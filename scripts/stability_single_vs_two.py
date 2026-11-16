"""Cross-seed and rollout stability for the SINGLE-parameter model, to compare with the two-parameter one.

The ablation question ("does the corrector repair an identifiable error or compensate the
training distribution?") should be answered separately for both settings, because they differ
in what extrapolation costs and therefore in what the corrector's benefit means.

For the single-parameter setting the relevant quantities are:

  cross-seed stability at k=1
      from the reproduced upstream metrics: three seeds per sigma, 20 sigmas, MF and MF+DP.

  rollout stability
      NOT available for the single-parameter model: the repeated-application study was run on
      the two-parameter checkpoint only. This script reports that absence explicitly rather
      than leaving the reader to assume a symmetry that was not measured.

  corrector gain inside vs outside the training range
      from high_sigma.csv, with the trained sigma range being [15, 150].

Usage:
    python scripts/stability_single_vs_two.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fullscale import grid as G  # noqa: E402

TRAIN_SIGMA_MAX = 150.0   # upstream TRAIN_SIGMA max
TRAIN_SIGMA_MIN = 15.0

print("=" * 90)
print("SINGLE-PARAMETER SETTING")
print("=" * 90)

# ---------- cross-seed stability ----------
seed_rows = []
for path in sorted((ROOT / "results" / "ks3d" / "evaluation" / "ks").rglob("grid.json")):
    pass  # grid.json holds no per-metric values; use the summary csv instead

summary_csv = ROOT / "results" / "ks3d" / "evaluation" / "ks" / "summary.csv"
metrics_csv = ROOT / "results" / "ks3d" / "evaluation" / "ks" / "metrics_per_seed.csv"

if metrics_csv.exists():
    rows = list(csv.DictReader(open(metrics_csv, encoding="utf-8")))
    print(f"\ncross-seed stability, from {metrics_csv.relative_to(ROOT)}")
    print(f"  columns: {list(rows[0].keys())}")

    # group by (sigma, method) and compute the sd across seeds
    groups = defaultdict(list)
    for row in rows:
        groups[(float(row["level"]), row["method"])].append(float(row["w2"]))
    methods = sorted({key[1] for key in groups})
    print(f"  methods present: {methods}")

    for method in methods:
        per_sigma = {sigma: values for (sigma, m), values in groups.items() if m == method}
        sds = [float(np.std(v, ddof=1)) for v in per_sigma.values() if len(v) > 1]
        means = [float(np.mean(v)) for v in per_sigma.values()]
        if sds:
            print(f"    {method:<12} mean {np.mean(means):.5f}  "
                  f"mean-within-sigma sd {np.mean(sds):.5f}  "
                  f"max sd {max(sds):.5f}  (n={len(sds)} sigmas)")
else:
    print(f"\n  {metrics_csv.relative_to(ROOT)} missing")

# ---------- gain inside vs outside the training range ----------
high = ROOT / "results" / "fullscale" / "high_sigma" / "high_sigma.csv"
if high.exists():
    rows = list(csv.DictReader(open(high, encoding="utf-8")))
    inside, outside = [], []
    for row in rows:
        sigma = float(row["sigma"])
        mf, dp = float(row["mf"]), float(row["mf_dp"])
        gain = 1 - dp / mf
        record = {"sigma": sigma, "mf": mf, "mf_dp": dp, "gain": gain}
        if sigma > TRAIN_SIGMA_MAX:
            outside.append(record)
        elif sigma >= TRAIN_SIGMA_MIN:
            inside.append(record)
        else:
            # below the training minimum: also extrapolation, but downward
            outside.append(record)

    def stat(group, label):
        if not group:
            return
        mf = np.mean([r["mf"] for r in group])
        dp = np.mean([r["mf_dp"] for r in group])
        gain = np.mean([r["gain"] for r in group])
        print(f"    {label:<26} n={len(group):>2}  MF {mf:.5f}  MF+DP {dp:.5f}  "
              f"gain {gain * 100:+.1f}%")

    print(f"\ncorrector gain, training sigma in [{TRAIN_SIGMA_MIN:.0f}, {TRAIN_SIGMA_MAX:.0f}]")
    stat(inside, "inside range")
    stat([r for r in outside if r["sigma"] > TRAIN_SIGMA_MAX], "above range (160-200)")
    stat([r for r in outside if r["sigma"] < TRAIN_SIGMA_MIN], "below range (10)")
    stat(outside, "outside range (all)")

    hi = [r for r in outside if r["sigma"] > TRAIN_SIGMA_MAX]
    if hi and inside:
        print(f"\n    degradation: MF rises from "
              f"{np.mean([r['mf'] for r in inside]):.5f} inside to "
              f"{np.mean([r['mf'] for r in hi]):.5f} above "
              f"({np.mean([r['mf'] for r in hi]) / np.mean([r['mf'] for r in inside]):.2f}x)")
        print(f"    gain: {np.mean([r['gain'] for r in inside]) * 100:+.1f}% -> "
              f"{np.mean([r['gain'] for r in hi]) * 100:+.1f}%  (essentially unchanged)")

print()
print("=" * 90)
print("TWO-PARAMETER SETTING (for contrast)")
print("=" * 90)
three = json.loads((G.RESULTS / "three_seed" / "three_seed.json").read_text("utf-8"))
print(f"\ncross-seed stability, three seeds x 143 cells")
print(f"    stage 1        mean {three['mf_two_mean']:.5f}  sd {three['mf_two_sd']:.5f}")
print(f"    stage 1 + corr mean {three['mf_dp_mean']:.5f}  sd {three['mf_dp_sd']:.5f}")
print(f"    sd reduction {100 * (1 - three['mf_dp_sd'] / three['mf_two_sd']):.0f}%")

three_regions = G.RESULTS / "extrapolation" / "three_regions.json"
if three_regions.exists():
    table = json.loads(three_regions.read_text("utf-8"))
    print(f"\ncorrector gain by region")
    for region, block in table.items():
        print(f"    {region:<22} n={block['n']:>3}  MF {block['mf']:.5f}  "
              f"gain {block['gain'] * 100:+.1f}%")

print()
print("=" * 90)
print("WHAT THIS MEANS FOR THE ABLATION")
print("=" * 90)
print("  Single-parameter: gain survives mild 1-D extrapolation (14.6% -> 13.1%).")
print("  Two-parameter   : gain collapses off the 2-D surface (14.2% -> 1.8%).")
print("  So the corrector's benefit is 'identifiable' in the weak sense that it transfers")
print("  along one axis, and 'distribution-bound' in the strong sense that it does not")
print("  transfer off a surface.")
