"""Measure the W2 statistical floor: two independent samples of one distribution.

The floor is the smallest W2 this metric can report between two clouds drawn from
the *same* physical parameter. Any model improvement smaller than the floor is
not resolvable. Reported per particle count and histogram resolution so the
study can pick a size that keeps the floor well below the effect measured.

Usage:
    python scripts/measure_floor.py --sigma 100 --sizes 20000 50000 --bins 16 24
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream" / "MF-DeepParticle" / "deepparticle"))

from twoparam.data import unit_ball  # noqa: E402


def hist_measures(points: np.ndarray, bins: int):
    """Occupied-bin centres and weights on a fixed box (comparable across clouds)."""
    from training.core import histogram  # upstream 3-D histogram helper

    points = np.asarray(points, dtype=np.float64)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    pad = 1e-9 + 1e-6 * (hi - lo)
    edges = [np.linspace(lo[d] - pad[d], hi[d] + pad[d], bins + 1) for d in range(3)]
    return histogram(points, edges)


def exact_w2(a_pts, a_w, b_pts, b_w) -> float:
    import ot
    from scipy.spatial.distance import cdist

    cost = cdist(a_pts, b_pts, metric="sqeuclidean")
    value, log = ot.emd2(a_w, b_w, cost, log=True, numItermax=5_000_000)
    if log.get("warning"):
        raise RuntimeError(log["warning"])
    return float(math.sqrt(max(value, 0.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=100.0)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--sizes", type=int, nargs="+", default=[20000, 50000])
    ap.add_argument("--bins", type=int, nargs="+", default=[16, 24])
    ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=ROOT / "results/twoparam/floor.json")
    a = ap.parse_args()

    import torch
    from training.protocol import simulate_original

    torch.set_num_threads(1)
    rows = []
    for n in a.sizes:
        sims = []
        for k in range(a.pairs + 1):
            seed = 2026100900 + 1000 * k + n // 1000
            np.random.seed(seed)
            torch.manual_seed(seed)
            initial = unit_ball(n, seed)
            started = time.perf_counter()
            target = simulate_original(torch.from_numpy(initial), a.sigma, mu=a.mu,
                                       chi=16 * math.pi, delta=0.1, t=0.02, dt=0.005,
                                       chunk=256, device=a.device).numpy()
            sims.append(target)
            print(f"  sim N={n} k={k} {time.perf_counter() - started:.1f}s", flush=True)
        for bins in a.bins:
            for k in range(1, len(sims)):
                pa, wa, _ = hist_measures(sims[0], bins)
                pb, wb, _ = hist_measures(sims[k], bins)
                w2 = exact_w2(pa, wa, pb, wb)
                rows.append({"particles": n, "bins": bins, "pair": k, "w2": w2})
                print(f"    floor N={n} bins={bins} pair={k}: W2={w2:.5f}", flush=True)
        del sims

    summary = {}
    for n in a.sizes:
        for bins in a.bins:
            values = [r["w2"] for r in rows if r["particles"] == n and r["bins"] == bins]
            if values:
                summary[f"N{n}_B{bins}"] = {"mean": float(np.mean(values)), "max": float(np.max(values))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"sigma": a.sigma, "mu": a.mu, "rows": rows, "summary": summary},
                                indent=2), encoding="utf-8")
    print("\nfloor summary (mean | max):")
    for key, value in summary.items():
        print(f"  {key:<12} {value['mean']:.5f} | {value['max']:.5f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
