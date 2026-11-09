"""Is the model near the noise floor, or does real headroom remain?

Why the global floor is not enough
----------------------------------
The floor quoted everywhere in this repository (0.0644) was measured at a single
parameter point, (sigma=100, mu=1). But the clouds grow with mu: at mu=10 the
support is roughly twice as wide, and the W2 between two independent samples of
the *same* distribution grows with the spread. Comparing the model's global mean
error against a single-point floor can therefore understate how close the model is
in the easy regions and overstate the gap in the hard ones.

This script measures the floor **locally**, at several (sigma, mu) points, by
simulating independent replicates and taking the mean pairwise W2. It then divides
the model's error on those same cells by the local floor, which is the honest
per-cell measure of "how far from the statistical limit are we".

The decomposition
-----------------
For a measured model error and a sampling floor, two bounds are reported because
W2 does not obey a Hilbert-space Pythagorean identity:

  * quadrature  sqrt(model^2 - floor^2)   optimistic, assumes independence
  * difference  model - floor             pessimistic, assumes worst-case alignment

Both are reported; neither is claimed to be exact.

Usage:
    python scripts/local_floor.py --replicates 3
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream" / "MF-DeepParticle" / "deepparticle"))

from fullscale import grid as G  # noqa: E402
from fullscale.evaluate import score_pair  # noqa: E402
from twoparam.data import unit_ball  # noqa: E402

OUT = G.RESULTS / "local_floor"

# a spread of regimes: weak shear / weak diffusion, the middle, and both extremes
POINTS = [(10.0, 0.1), (100.0, 1.0), (100.0, 10.0), (200.0, 10.0), (200.0, 0.1)]


def simulate(sigma: float, mu: float, *, n: int, seed: int, device="cuda") -> np.ndarray:
    from training.protocol import simulate_original

    initial = unit_ball(n, seed)
    np.random.seed(seed + 13)
    torch.manual_seed(seed + 13)
    return simulate_original(torch.from_numpy(initial), float(sigma), mu=float(mu),
                             chi=16 * math.pi, delta=0.1, t=0.02, dt=0.005,
                             chunk=256, device=device).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--out", type=Path, default=OUT / "local_floor.json")
    a = ap.parse_args()

    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda t: print(f"[{time.strftime('%H:%M:%S')}] {t}", flush=True)

    # model errors per cell, from the converged-budget run
    detail_path = G.RESULTS / "optimal_budget_ablation" / "detail.csv"
    model_error = {}
    if detail_path.is_file():
        import csv
        with detail_path.open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                model_error[row["key"]] = float(row["w2_two_param"])
        log(f"loaded model errors for {len(model_error)} cells")

    results = []
    for sigma, mu in POINTS:
        log(f"=== sigma={sigma:g} mu={mu:g} ===")
        clouds = []
        for k in range(a.replicates):
            seed = 700_000 + 1_009 * k
            started = time.perf_counter()
            clouds.append(simulate(sigma, mu, n=a.n, seed=seed))
            log(f"  replicate {k} in {time.perf_counter() - started:.0f}s")
        pairs = []
        for i in range(len(clouds)):
            for j in range(i + 1, len(clouds)):
                d = score_pair(clouds[i].astype(np.float64), clouds[j].astype(np.float64),
                               G.BINS)["w2"]
                pairs.append(d)
        floor = float(np.mean(pairs))
        i = G.SIGMA_GRID.index(sigma)
        j = G.MU_GRID.index(mu)
        key = G.Cell(i, j).key
        err = model_error.get(key)
        entry = {"sigma": sigma, "mu": mu, "key": key,
                 "floor": floor, "floor_values": pairs,
                 "n_replicates": len(clouds),
                 "model_error": err,
                 "ratio": (err / floor) if err else None}
        if err:
            entry["excess_difference"] = err - floor
            entry["excess_quadrature"] = math.sqrt(max(err * err - floor * floor, 0.0))
        results.append(entry)
        log(f"  local floor {floor:.5f}   model {err if err else float('nan'):.5f}   "
            f"ratio {entry['ratio'] if entry['ratio'] else float('nan'):.2f}x")
        del clouds
        torch.cuda.empty_cache()

    print(f"\n{'sigma':>7} {'mu':>8} {'local floor':>12} {'model W2':>10} {'ratio':>7} "
          f"{'excess(diff)':>13} {'excess(quad)':>13}")
    for r in results:
        if not r["model_error"]:
            continue
        print(f"{r['sigma']:7.0f} {r['mu']:8.3f} {r['floor']:12.5f} "
              f"{r['model_error']:10.5f} {r['ratio']:7.2f} "
              f"{r['excess_difference']:13.5f} {r['excess_quadrature']:13.5f}")

    floors = [r["floor"] for r in results]
    print(f"\nlocal floor range: {min(floors):.5f} .. {max(floors):.5f}  "
          f"(ratio {max(floors) / min(floors):.2f}x)")
    print("  the floor is parameter dependent, so a global single-point floor")
    print("  gives a misleading per-cell picture")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"n": a.n, "bins": G.BINS, "points": results},
                                indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
