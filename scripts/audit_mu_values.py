"""Show exactly which cells and which mu values each single-parameter control saw.

The question being answered: "what mu was the single-parameter model trained on?"
There are three different controls and they do NOT use the same mu, so the answer has
to be given per-control, from the training logs rather than from the script comments.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fullscale import grid as G

FC = ROOT / "results" / "fullscale" / "fair_control"

print("MU_GRID (%d values):" % len(G.MU_GRID))
for j, mu in enumerate(G.MU_GRID):
    mark = "   <-- mu = 1 exactly (BASELINE, index %d)" % j if abs(mu - 1.0) < 1e-12 else ""
    print(f"  index {j}: {mu:.6g}{mark}")

print()
for name in ("slice-20", "same-4", "same-cells"):
    log_path = FC / name / "training_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8"))
    keys = log["cells"]
    mus = sorted({float(G.MU_GRID[int(k[5:7])]) for k in keys})
    sigmas = sorted({float(G.SIGMA_GRID[int(k[1:3])]) for k in keys})
    print("=" * 76)
    print(f"{name}:  {len(keys)} cells, {log['steps']} steps")
    print(f"  distinct mu values  ({len(mus)}): {[round(m, 4) for m in mus]}")
    print(f"  contains mu == 1 ?  {any(abs(m - 1.0) < 1e-12 for m in mus)}")
    print(f"  distinct sigma ({len(sigmas)}): {[round(s, 1) for s in sigmas]}")
    print(f"  cells: {', '.join(keys)}")

print()
print("=" * 76)
print("the 40 LHS cells, showing which ones sit at mu = 1")
lhs_log = json.loads((ROOT / "results/fullscale/fair_control/same-cells/training_log.json")
                     .read_text(encoding="utf-8"))
at_one = [k for k in lhs_log["cells"] if int(k[5:7]) == G.MU_INDEX]
print(f"  {len(at_one)} of 40 LHS cells have mu = 1: {', '.join(at_one)}")
