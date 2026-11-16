"""Check whether the interior evaluation includes the training cells.

The extrapolation study samples 45 training cells from the 135-cell interior and reports
the interior zone as n = 135. If those 135 include the 45 training cells, the interior
number is contaminated by memorisation and the interior-vs-extrapolation contrast is
overstated. This check settles it from the detail CSV.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fullscale import grid as G  # noqa: E402

OUT = G.RESULTS / "extrapolation"
summary = json.loads((OUT / "summary_s88.json").read_text("utf-8"))
train = set(summary["train_cells"])

rows = list(csv.DictReader(open(OUT / "detail_s88.csv", encoding="utf-8")))
print(f"detail rows: {len(rows)}")

# detect the zone column
if rows:
    print("columns:", list(rows[0].keys()))

def zone_of(key: str) -> str:
    i, j = int(key[1:3]), int(key[5:7])
    sigma, mu = G.SIGMA_GRID[i], G.MU_GRID[j]
    sigma_out = sigma > 150.0
    mu_out = mu > 3.981071705535 + 1e-9
    if sigma_out and mu_out:
        return "corner"
    if sigma_out:
        return "sigma_extrap"
    if mu_out:
        return "mu_extrap"
    return "interior"

from collections import defaultdict
zones = defaultdict(list)
for row in rows:
    zones[zone_of(row["key"])].append(row)

print(f"\n{'zone':<14} {'n':>4} {'of which TRAINING':>18} {'leak share':>10}")
for zone in ("interior", "sigma_extrap", "mu_extrap", "corner"):
    group = zones.get(zone, [])
    n_train = sum(1 for r in group if r["key"] in train)
    share = n_train / len(group) if group else 0.0
    print(f"{zone:<14} {len(group):>4} {n_train:>18} {share:>9.0%}")

print("\n--- if interior has leaking cells, recompute on held-out interior only ---")
held = [r for r in zones.get("interior", []) if r["key"] not in train]
if held and len(held) < len(zones.get("interior", [])):
    mf_col = "w2_mf" if "w2_mf" in held[0] else None
    dp_col = "w2_mf_dp" if "w2_mf_dp" in held[0] else None
    if mf_col and dp_col:
        mf = sum(float(r[mf_col]) for r in held) / len(held)
        dp = sum(float(r[dp_col]) for r in held) / len(held)
        print(f"  held-out interior only (n={len(held)}): MF {mf:.5f}  MF+DP {dp:.5f}  "
              f"gain {100*(1-dp/mf):+.1f}%")
        allint = zones.get("interior", [])
        mf_all = sum(float(r[mf_col]) for r in allint) / len(allint)
        dp_all = sum(float(r[dp_col]) for r in allint) / len(allint)
        print(f"  reported interior  (n={len(allint)}): MF {mf_all:.5f}  MF+DP {dp_all:.5f}  "
              f"gain {100*(1-dp_all/mf_all):+.1f}%")
        print(f"  -> contamination inflates the interior score by "
              f"{100*(1-mf_all/mf):.1f}% (MF) and {100*(1-dp_all/dp):.1f}% (MF+DP)")
else:
    print("  interior contains no training cells; the reported number is clean")
