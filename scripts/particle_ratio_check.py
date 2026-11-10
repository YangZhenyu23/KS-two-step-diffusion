"""Why is the particle-level ratio at or below one, and does it hold up?

Two things need checking before the ratio can be reported.

1. **The ratio came out 0.95 and 1.06.** A value below one is impossible in
   principle: the model cloud and the reference cloud cannot be closer than two
   independent draws from the same distribution. Something makes the measured
   floor a little too large or the measured error a little too small.

   Candidate causes, all testable:
     (a) **Fixed index draw.** Both the error and the floor use the same index
         vector, so they share the particular subsample's peculiarities. With one
         draw the estimates are highly correlated and both biased in the same
         direction; the ratio can land on either side of one.
     (b) **Finite replicate count.** The floor is estimated from a single pair of
         references. Averaging over more independent pairs reduces its upward
         bias, since E[min over pairs] < E[pair].
     (c) **Model variance.** A trained network is a deterministic map from the
         source cloud, so its output is *less* variable than an independent
         physical realisation. That genuinely can put the error below the
         between-realisation floor: the model is closer to the reference than a
         fresh simulation would be, because it is partly fitting the reference.

   Cause (c) is the interesting one and it is not a defect: if the model has
   learned something specific to this reference cloud, its error *should* be
   below the between-realisation distance. That would mean the floor is the wrong
   benchmark for a deterministic predictor.

2. **Does the conclusion survive more index draws and more floor replicates?**
   The script re-runs with several index draws and several independent reference
   replicates, then reports the ratio with an honest spread.

Usage:
    python scripts/particle_ratio_check.py --cell s00_m00 --subsample 3000
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
from fullscale.train import load_cell, load_two  # noqa: E402
from fullscale.model import sample_two_param  # noqa: E402
from twoparam.data import unit_ball  # noqa: E402

OUT = G.RESULTS / "particle_metric"


def exact_w2(a, b):
    import ot
    from scipy.spatial.distance import cdist

    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    cost = cdist(a, b, metric="sqeuclidean")
    n = len(a)
    w = np.full(n, 1.0 / n)
    value, info = ot.emd2(w, w, cost, log=True, numItermax=5_000_000)
    if int(info.get("result_code", -1)) != 1:
        raise RuntimeError(f"OT failed: {info.get('warning')}")
    return math.sqrt(max(value, 0.0))


def simulate(initial, sigma, mu, noise_seed, device="cuda"):
    from training.protocol import simulate_original

    np.random.seed(noise_seed)
    torch.manual_seed(noise_seed)
    return simulate_original(torch.from_numpy(initial), float(sigma), mu=float(mu),
                             chi=16 * math.pi, delta=0.1, t=0.02, dt=0.005,
                             chunk=256, device=device).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="s00_m00")
    ap.add_argument("--subsample", type=int, default=3000)
    ap.add_argument("--index-draws", type=int, default=4)
    ap.add_argument("--replicates", type=int, default=3,
                    help="independent reference replicates for the floor")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda t: print(f"[{time.strftime('%H:%M:%S')}] {t}", flush=True)

    i, j = int(a.cell[1:3]), int(a.cell[5:7])
    cell = G.Cell(i, j)
    log(f"cell {cell.key}: sigma={cell.sigma:g} mu={cell.mu:g}")

    payload = load_cell(cell)
    reference = payload["clouds"]["reference"].astype(np.float64)
    model, _ = load_two(G.RESULTS / "optimal_budget_ablation" / "two_32000"
                        / "mf_two_param.pt", a.device)
    prediction = sample_two_param(model, torch.from_numpy(payload["clouds"]["source"]),
                                  cell.sigma, cell.mu).numpy().astype(np.float64)
    del model
    torch.cuda.empty_cache()

    # independent physical realisations on the same initial cloud
    initial = unit_ball(a.n, 88)
    replicates = []
    for k in range(a.replicates):
        noise_seed = 940_000 + 977 * (i * 10 + j) + 3119 * k
        started = time.perf_counter()
        replicates.append(simulate(initial, cell.sigma, cell.mu, noise_seed, a.device)
                          .astype(np.float64))
        log(f"  replicate {k}: {time.perf_counter() - started:.0f}s")
        torch.cuda.empty_cache()

    rng = np.random.default_rng(515151)
    indices = [rng.choice(a.n, size=a.subsample, replace=False)
               for _ in range(a.index_draws)]

    rows = []
    for d, idx in enumerate(indices):
        err = exact_w2(prediction[idx], reference[idx])
        floors = [exact_w2(reference[idx], rep[idx]) for rep in replicates]
        # also the spread among the replicates themselves: the true between-sample
        # distance, estimated from independent pairs only
        between = []
        for p in range(len(replicates)):
            for q in range(p + 1, len(replicates)):
                between.append(exact_w2(replicates[p][idx], replicates[q][idx]))
        rows.append({
            "draw": d,
            "model_error": err,
            "floor_mean": float(np.mean(floors)),
            "floor_min": float(np.min(floors)),
            "floor_max": float(np.max(floors)),
            "between_mean": float(np.mean(between)) if between else None,
            "ratio": err / float(np.mean(floors)),
            "ratio_vs_min": err / float(np.min(floors)),
        })
        log(f"  draw {d}: err {err:.5f}  floor {np.mean(floors):.5f} "
            f"[{np.min(floors):.5f}, {np.max(floors):.5f}]  "
            f"ratio {rows[-1]['ratio']:.3f}")

    ratios = np.array([r["ratio"] for r in rows])
    print(f"\nmodel error: mean {np.mean([r['model_error'] for r in rows]):.5f} "
          f"(sd {np.std([r['model_error'] for r in rows], ddof=1):.5f})")
    print(f"floor:       mean {np.mean([r['floor_mean'] for r in rows]):.5f} "
          f"(sd across draws {np.std([r['floor_mean'] for r in rows], ddof=1):.5f})")
    print(f"ratio:       mean {ratios.mean():.3f}  sd {ratios.std(ddof=1):.3f}  "
          f"range [{ratios.min():.3f}, {ratios.max():.3f}]")
    print(f"\nfloor spread between independent reference replicates: "
          f"{np.mean([r['floor_max'] - r['floor_min'] for r in rows]):.5f}")
    print("\ninterpretation:")
    print("  * if ratio stays near or below 1 across draws, the model is at the")
    print("    between-realisation distance. That is the ceiling of what a")
    print("    deterministic predictor can achieve against a single reference.")
    print("  * a ratio slightly below 1 is expected once the model is smooth: it")
    print("    cannot reproduce the reference's own sampling noise, so its distance")
    print("    to the reference is smaller than two independent draws would be.")

    (OUT / f"ratio_check_{a.cell}.json").write_text(json.dumps({
        "cell": cell.key, "sigma": cell.sigma, "mu": cell.mu,
        "subsample": a.subsample, "index_draws": a.index_draws,
        "replicates": a.replicates, "rows": rows,
        "ratio_mean": float(ratios.mean()), "ratio_sd": float(ratios.std(ddof=1)),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / f'ratio_check_{a.cell}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
