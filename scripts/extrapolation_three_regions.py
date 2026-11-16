"""Split the parameter domain into three regions, not two.

The user's framing is "inside the training range vs beyond it", which is right but
incomplete for this design. The training cells are 45 points scattered over the interior
box, so the interior itself has two parts:

  R1  training cells themselves            - memorisation, not a test
  R2  interior but NOT a training cell     - interpolation: the parameter is within the
                                             sampled range, the exact setting is unseen
  R3  outside the training box             - extrapolation: the parameter exceeds every
                                             value ever shown

R2 is the honest "inside" number; quoting R1 would flatter the model. R3 is the real
extrapolation test. This script recomputes all three from the detail CSV, using the
existing predictions (no new model runs).

Also reported: the same split for the corrector's gain, since the question is specifically
whether the corrector still helps as we move from R2 to R3.

Usage:
    python scripts/extrapolation_three_regions.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fullscale import grid as G  # noqa: E402

OUT = G.RESULTS / "extrapolation"
summary = json.loads((OUT / "summary_s88.json").read_text("utf-8"))
train = set(summary["train_cells"])
rows = list(csv.DictReader(open(OUT / "detail_s88.csv", encoding="utf-8")))
floor = summary["noise_floor"]

SIGMA_MAX = 150.0
MU_MAX = 3.981071705535


def region_of(row) -> str:
    key = row["key"]
    sigma, mu = float(row["sigma"]), float(row["mu"])
    outside = sigma > SIGMA_MAX or mu > MU_MAX + 1e-9
    if outside:
        return "R3_outside"
    return "R1_training" if key in train else "R2_interior_heldout"


buckets = defaultdict(list)
for row in rows:
    buckets[region_of(row)].append(row)

print("=" * 88)
print("THREE REGIONS (interior split into memorised and held-out)")
print("=" * 88)
print(f"{'region':<22} {'n':>4} {'MF':>9} {'MF+DP':>9} {'gain':>8} {'MF/floor':>9}")

table = {}
for region in ("R1_training", "R2_interior_heldout", "R3_outside"):
    group = buckets.get(region, [])
    if not group:
        continue
    mf = sum(float(r["w2_mf"]) for r in group) / len(group)
    dp = sum(float(r["w2_mf_dp"]) for r in group) / len(group)
    gain = 1 - dp / mf
    table[region] = {"n": len(group), "mf": mf, "mf_dp": dp, "gain": gain,
                     "mf_over_floor": mf / floor}
    print(f"{region:<22} {len(group):>4} {mf:9.5f} {dp:9.5f} {gain * 100:+7.1f}% "
          f"{mf / floor:9.2f}")

print()
print("=" * 88)
print("R3 broken down by which parameter leaves the box")
print("=" * 88)
print(f"{'sub-zone':<22} {'n':>4} {'MF':>9} {'MF+DP':>9} {'gain':>8} {'MF/floor':>9}")
sub = defaultdict(list)
for row in buckets.get("R3_outside", []):
    sigma, mu = float(row["sigma"]), float(row["mu"])
    sigma_out = sigma > SIGMA_MAX
    mu_out = mu > MU_MAX + 1e-9
    label = ("corner (both out)" if sigma_out and mu_out
             else "sigma out only" if sigma_out else "mu out only")
    sub[label].append(row)
for label in ("sigma out only", "mu out only", "corner (both out)"):
    group = sub.get(label, [])
    if not group:
        continue
    mf = sum(float(r["w2_mf"]) for r in group) / len(group)
    dp = sum(float(r["w2_mf_dp"]) for r in group) / len(group)
    print(f"{label:<22} {len(group):>4} {mf:9.5f} {dp:9.5f} "
          f"{100 * (1 - dp / mf):+7.1f}% {mf / floor:9.2f}")

print()
print("=" * 88)
print("THE TWO SENTENCES THAT MATTER")
print("=" * 88)
if "R2_interior_heldout" in table and "R3_outside" in table:
    r2, r3 = table["R2_interior_heldout"], table["R3_outside"]
    print(f"interpolation  : MF {r2['mf']:.4f} ({r2['mf_over_floor']:.2f}x floor), "
          f"corrector gain {r2['gain'] * 100:+.1f}%")
    print(f"extrapolation  : MF {r3['mf']:.4f} ({r3['mf_over_floor']:.2f}x floor), "
          f"corrector gain {r3['gain'] * 100:+.1f}%")
    print(f"degradation    : {r3['mf'] / r2['mf']:.2f}x worse outside the box")
    print(f"corrector      : {r2['gain'] * 100:.1f}% -> {r3['gain'] * 100:.1f}% "
          f"(collapses)")

(OUT / "three_regions.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
