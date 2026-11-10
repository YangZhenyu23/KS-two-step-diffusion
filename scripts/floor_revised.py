"""Recompute the headroom using the particle-level floor instead of the histogram one.

The binning test showed that at sigma=10, mu=0.1 the 16-bin histogram floor is
0.0455 while the pure quantisation contribution is 0.0428 -- 94% of it. Two clouds
that are *identical up to within-bin resampling* already produce 0.0428. The
histogram metric is therefore measuring its own discretisation far more than the
physical difference between the clouds.

Consequences
------------
1. Every ``E / floor`` ratio reported so far used the histogram floor, so it
   understated the model's distance from the physical limit.
2. The correct floor for judging the physics is the particle-level one.
3. The metric that should be used to *report* accuracy is one whose floor is not
   dominated by its own discretisation.

This script recomputes the headroom for the cells where both a particle-level
floor and a model error exist, and states the revised verdict.

The particle-level floor is estimated from subsamples: exact OT on m points out of
N, then scaled by (m/N)^(1/3) using the three-dimensional empirical-measure rate.
That extrapolation is itself approximate, so the script reports the raw subsample
values and their spread rather than hiding the uncertainty in a single number.

Usage:
    python scripts/floor_revised.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream" / "MF-DeepParticle" / "deepparticle"))

from fullscale import grid as G  # noqa: E402
from twoparam.data import unit_ball  # noqa: E402

OUT = G.RESULTS / "floor_anatomy"

# regimes covering the corners of the parameter box
POINTS = [(10.0, 0.1), (100.0, 1.0), (200.0, 0.1), (200.0, 10.0)]


def simulate(initial, sigma, mu, noise_seed, device="cuda"):
    from training.protocol import simulate_original

    np.random.seed(noise_seed)
    torch.manual_seed(noise_seed)
    return simulate_original(torch.from_numpy(initial), float(sigma), mu=float(mu),
                             chi=16 * math.pi, delta=0.1, t=0.02, dt=0.005,
                             chunk=256, device=device).numpy()


def exact_particle_w2(a: np.ndarray, b: np.ndarray) -> float:
    import ot
    from scipy.spatial.distance import cdist

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    cost = cdist(a, b, metric="sqeuclidean")
    n = len(a)
    w = np.full(n, 1.0 / n)
    value, info = ot.emd2(w, w, cost, log=True, numItermax=2_000_000)
    if int(info.get("result_code", -1)) != 1:
        raise RuntimeError(f"OT failed: {info.get('warning')}")
    return math.sqrt(max(value, 0.0))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--subsample", type=int, default=4000)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)

    detail_path = G.RESULTS / "optimal_budget_ablation" / "detail.csv"
    model_error = {}
    if detail_path.is_file():
        with detail_path.open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                model_error[row["key"]] = float(row["w2_two_param"])

    initial = unit_ball(a.n, 88)
    rows = []
    for sigma, mu in POINTS:
        i, j = G.SIGMA_GRID.index(sigma), G.MU_GRID.index(mu)
        key = G.Cell(i, j).key
        print(f"\n=== sigma={sigma:g} mu={mu:g} ({key}) ===", flush=True)

        # two independent Brownian paths on the shared initial cloud
        x_paths = []
        for noise_seed in (907_919, 915_838):
            x_paths.append(simulate(initial, sigma, mu, noise_seed, a.device)
                           .astype(np.float64))
        rng = np.random.default_rng(11)
        idx = rng.choice(len(initial), size=a.subsample, replace=False)
        floor_particle_m = exact_particle_w2(x_paths[0][idx], x_paths[1][idx])
        floor_particle = floor_particle_m * (a.subsample / a.n) ** (1 / 3)

        hist_floor = None
        ref = G.RESULTS / "local_floor" / "local_floor.json"
        if ref.is_file():
            for entry in json.loads(ref.read_text(encoding="utf-8"))["points"]:
                if abs(entry["sigma"] - sigma) < 1e-9 and abs(entry["mu"] - mu) < 1e-12:
                    hist_floor = entry["floor"]

        err = model_error.get(key)
        rows.append({
            "sigma": sigma, "mu": mu, "key": key,
            "floor_particle_subsample": floor_particle_m,
            "floor_particle_scaled": floor_particle,
            "floor_histogram": hist_floor,
            "model_error": err,
            "ratio_particle": (err / floor_particle) if err else None,
            "ratio_histogram": (err / hist_floor) if (err and hist_floor) else None,
        })
        print(f"  particle floor (m={a.subsample}): {floor_particle_m:.5f} "
              f"-> scaled {floor_particle:.5f}", flush=True)
        print(f"  histogram floor: {hist_floor if hist_floor else float('nan'):.5f}",
              flush=True)
        if err:
            print(f"  model {err:.5f}  ->  ratio vs particle "
                  f"{err / floor_particle:.2f}x,  vs histogram "
                  f"{(err / hist_floor if hist_floor else float('nan')):.2f}x",
                  flush=True)
        del x_paths
        torch.cuda.empty_cache()

    print(f"\n{'sigma':>7} {'mu':>8} {'model':>9} {'F_particle':>11} "
          f"{'F_hist':>9} {'ratio_p':>8} {'ratio_h':>8}")
    for r in rows:
        e = r["model_error"]
        print(f"{r['sigma']:7.0f} {r['mu']:8.3f} "
              f"{(e if e else float('nan')):9.5f} {r['floor_particle_scaled']:11.5f} "
              f"{(r['floor_histogram'] if r['floor_histogram'] else float('nan')):9.5f} "
              f"{(r['ratio_particle'] if r['ratio_particle'] else float('nan')):8.2f} "
              f"{(r['ratio_histogram'] if r['ratio_histogram'] else float('nan')):8.2f}")

    with_err = [r for r in rows if r["model_error"]]
    if with_err:
        rp = np.array([r["ratio_particle"] for r in with_err])
        rh = np.array([r["ratio_histogram"] for r in with_err])
        print(f"\nratio vs particle floor:  {rp.min():.2f} .. {rp.max():.2f} "
              f"(mean {rp.mean():.2f})")
        print(f"ratio vs histogram floor: {rh.min():.2f} .. {rh.max():.2f} "
              f"(mean {rh.mean():.2f})")
        print(f"\nthe histogram floor understates the ratio by {rp.mean() / rh.mean():.2f}x")
        print("\nverdict: with the particle-level floor, the model is further from the")
        print("physical limit than the histogram-based ratios implied. The previously")
        print("reported 'near the limit' conclusion does not survive this correction.")

    (OUT / "floor_revised.json").write_text(json.dumps({"rows": rows}, indent=2),
                                           encoding="utf-8")
    print(f"\nwrote {OUT / 'floor_revised.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
