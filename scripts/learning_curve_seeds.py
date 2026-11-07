"""Multi-seed repetition of the learning curve, to separate signal from variance.

The single-seed curve came out non-monotone:

    n=20   test W2 0.1803
    n=40   test W2 0.1943      <- worse than 20
    n=80   test W2 0.1388
    n=160  test W2 0.1527

A curve that goes down, up, down cannot be reported as "more data helps": with one
design per size, the wiggle is indistinguishable from design variance. This script
repeats each size with several independent designs (different sampling seeds, same
frozen test set) and reports mean, standard deviation and the range.

That turns the question into a statistical one: is the spread between designs at
a fixed size smaller or larger than the trend across sizes? If the design spread
is larger, the honest conclusion is "at these sizes the design choice matters as
much as the sample size", which is itself a useful and reportable finding.

Budget
------
Each run costs 2000 updates per cell, so a run at n=160 costs eight times a run at
n=20. To keep the total affordable the number of seeds decreases with size:
6 seeds for n=20 and n=40, 3 for n=80, 2 for n=160. The script reports the actual
per-size spread so the reader can judge whether that is enough.

Usage:
    python scripts/learning_curve_seeds.py
    python scripts/learning_curve_seeds.py --seeds-20 6 --seeds-40 6 --seeds-80 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fullscale import grid as G  # noqa: E402
from fullscale.grid import Cell, Config  # noqa: E402
from fullscale.train import train_two_param  # noqa: E402

from sample_efficiency import (  # noqa: E402
    OUT, UPDATES_PER_CELL, frozen_test_set, run_one, stratified_subset, write_csv,
    write_json, log,
)

SEEDS_PER_SIZE = {20: 6, 40: 6, 80: 3, 160: 2}


def main() -> int:
    ap = argparse.ArgumentParser()
    for n in SEEDS_PER_SIZE:
        ap.add_argument(f"--seeds-{n}", type=int, default=SEEDS_PER_SIZE[n])
    a = ap.parse_args()
    seed_counts = {n: getattr(a, f"seeds_{n}") for n in SEEDS_PER_SIZE}

    torch.set_num_threads(min(8, torch.get_num_threads()))
    test, pool = frozen_test_set()
    log(f"test {len(test)} cells, pool {len(pool)} cells")
    log(f"seeds per size: {seed_counts}")

    results = []
    for n, count in seed_counts.items():
        for k in range(count):
            design_seed = 20261107 + 10000 * k + n
            cells = stratified_subset(pool, n, seed=design_seed)
            name = f"seedcurve_n{n}_s{k}"
            record = run_one(name, cells, test, pool, reuse_existing=True)
            record["design_seed"] = design_seed
            record["repeat"] = k
            results.append(record)
            log(f"  n={n} repeat {k + 1}/{count}: TEST {record['test_mean_w2']:.5f}")

    write_csv(OUT / "learning_curve_seeds.csv", results)

    print(f"\n{'n cells':>8} {'runs':>5} {'mean':>9} {'sd':>8} {'min':>9} {'max':>9} "
          f"{'x floor':>8}")
    summary = {}
    for n in sorted(seed_counts):
        group = [r for r in results if r["n_cells"] == n]
        values = [r["test_mean_w2"] for r in group]
        row = {
            "n_cells": n,
            "runs": len(values),
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "over_floor": float(np.mean(values) / G.FLOOR_16),
            "values": values,
        }
        summary[n] = row
        print(f"{n:8d} {len(values):5d} {row['mean']:9.5f} {row['sd']:8.5f} "
              f"{row['min']:9.5f} {row['max']:9.5f} {row['over_floor']:8.2f}")

    sizes = sorted(summary)
    within = float(np.mean([summary[n]["sd"] for n in sizes if summary[n]["runs"] > 1]))
    trend = float(summary[sizes[0]]["mean"] - summary[sizes[-1]]["mean"])
    print(f"\nmean within-size spread (sd) : {within:.5f}")
    print(f"trend from n={sizes[0]} to n={sizes[-1]} : {trend:.5f}")
    print(f"ratio trend / spread         : {trend / within:.2f}x")
    print("  A ratio well above 1 means the data-size trend is larger than the")
    print("  design-to-design noise, so the curve is interpretable. A ratio near 1")
    print("  means design variance dominates and single-seed curves are unreliable.")

    write_json(OUT / "learning_curve_seeds.json", {
        "seeds_per_size": seed_counts,
        "n_test": len(test),
        "summary": {str(n): summary[n] for n in sizes},
        "within_size_sd": within,
        "trend": trend,
        "trend_over_spread": trend / within,
    })
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
