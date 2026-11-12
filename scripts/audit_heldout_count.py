"""Settle the overlap count that decides whether the held-out set has 143 or 144 cells.

Two different numbers appear in the repository:
  * ``optimal_budget_ablation`` reports 143 held-out cells
  * ``fair_control`` reports 144

The difference is the overlap between the two training sets (the LHS-40 design and the
mu=1 row). This script recomputes both designs from disk and prints which LHS cells sit
at mu=1, so the arithmetic in the paper draft can be stated correctly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fullscale import grid as G  # noqa: E402

R = ROOT / "results" / "fullscale"

print(f"MU_INDEX = {G.MU_INDEX}, BASELINE_MU = {G.BASELINE_MU}")
print(f"mu=1 row: {[c.key for c in G.baseline_slice()]}\n")

# every candidate record of the LHS-40 design
candidates = {
    "fair_control/same-cells": R / "fair_control/same-cells/training_log.json",
    "train/lhs40": R / "train/lhs40/training_log.json",
    "optimal_budget_ablation/two_32000": R / "optimal_budget_ablation/two_32000/training_log.json",
}
designs = {}
for label, path in candidates.items():
    if not path.exists():
        print(f"{label}: log missing")
        continue
    keys = json.loads(path.read_text(encoding="utf-8"))["cells"]
    at_one = [k for k in keys if int(k[5:7]) == G.MU_INDEX]
    designs[label] = keys
    print(f"{label}")
    print(f"  {len(keys)} cells, {len(at_one)} at mu=1: {', '.join(at_one)}")

print()
for label, keys in designs.items():
    overlap = len([k for k in keys if int(k[5:7]) == G.MU_INDEX])
    held = 200 - len(keys) - 20 + overlap
    print(f"{label}: 200 - {len(keys)} - 20 + {overlap} = {held} held out")

# are the designs the same set of cells?
sets = {k: frozenset(v) for k, v in designs.items()}
labels = list(sets)
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        same = sets[labels[i]] == sets[labels[j]]
        print(f"\n{labels[i]} == {labels[j]} ? {same}")
        if not same:
            only_a = sorted(sets[labels[i]] - sets[labels[j]])
            only_b = sorted(sets[labels[j]] - sets[labels[i]])
            print(f"  only in first : {only_a}")
            print(f"  only in second: {only_b}")
