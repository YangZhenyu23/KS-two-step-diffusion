"""Calibrate the particle count and histogram resolution for the paper-scale run.

Two questions have to be answered before generating expensive data:

1. What is the W2 statistical floor at the candidate particle counts?
   The floor is ``W2`` between two independent samples of the *same* parameters.
   It is the smallest difference this metric can resolve, so a model error close
   to the floor is essentially perfect and the experiment is saturated.
2. How long does one cell take, so the total budget is known before starting?

The earlier 20k-particle study used 16 bins/axis and measured a floor of 0.093.
This script measures 100k (the upstream paper's evaluation size) at 16, 24 and
32 bins, plus timing, and writes ``results/fullscale/floor.json``.
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

from twoparam.data import unit_ball  # noqa: E402
from twoparam.evaluate import _edges, measure, solve  # noqa: E402

OUT = ROOT / "results" / "fullscale"


def simulate(initial, sigma, mu, seed, device="cuda"):
    from training.protocol import simulate_original

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return simulate_original(torch.from_numpy(initial), sigma, mu=mu, chi=16 * math.pi,
                             delta=0.1, t=0.02, dt=0.005, chunk=256, device=device).numpy()


def w2(a, b, bins):
    edges = _edges([a, b], bins)
    return solve(measure(a, edges, bins), measure(b, edges, bins))["w2"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[100_000])
    ap.add_argument("--sigma", type=float, default=100.0)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--bins", type=int, nargs="+", default=[16, 24, 32])
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=OUT / "floor.json")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    report = {"sigma": a.sigma, "mu": a.mu, "sizes": {}}

    for n in a.n:
        samples, seconds = [], []
        for k in range(a.pairs + 1):
            seed = 2026100900 + 1000 * k
            initial = unit_ball(n, seed)
            started = time.perf_counter()
            cloud = simulate(initial, a.sigma, a.mu, seed + 13, a.device)
            elapsed = time.perf_counter() - started
            seconds.append(elapsed)
            samples.append(cloud)
            print(f"  N={n} sample {k}: {elapsed:.1f}s", flush=True)
        entry = {"seconds_per_cell": float(np.mean(seconds)),
                 "seconds_all": [float(s) for s in seconds], "bins": {}}
        for bins in a.bins:
            values = [w2(samples[0], samples[k], bins) for k in range(1, len(samples))]
            entry["bins"][str(bins)] = {"mean": float(np.mean(values)),
                                        "max": float(np.max(values)),
                                        "values": [float(v) for v in values]}
            print(f"    floor N={n} bins={bins}: {np.mean(values):.5f} "
                  f"(max {np.max(values):.5f})", flush=True)
        report["sizes"][str(n)] = entry
        del samples
        torch.cuda.empty_cache()

    (a.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== floor vs bins ===")
    print(f"{'N':>8} " + " ".join(f"{b:>9}" for b in a.bins))
    for n, entry in report["sizes"].items():
        row = " ".join(f"{entry['bins'][str(b)]['mean']:9.5f}" for b in a.bins)
        print(f"{n:>8} {row}")
    print("\n=== cost ===")
    for n, entry in report["sizes"].items():
        per = entry["seconds_per_cell"]
        print(f"  N={n}: {per:.1f}s/cell  ->  100 cells = {per * 100 / 60:.0f} min")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
