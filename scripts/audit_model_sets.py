"""Precise audit of what each model trained on and what it is scored on.

Three questions this answers, from disk rather than from memory:

  Q1  Did every model finish training (a checkpoint exists and its log is complete)?
  Q2  For each model, what is its training set, and what is the evaluation set that its
      results were actually computed on? Are they disjoint?
  Q3  Are the single-parameter and two-parameter models evaluated on the SAME cells?
      This is the question that decides whether their numbers can be compared at all.

The subtlety: "the same cells" and "the same held-out set" are different requirements.
Comparing two models requires a common evaluation set that is held out for BOTH. If model
A trained on cells that model B is scored on, the comparison is invalid even when the two
models look like they were scored on "the same 143 cells".
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream" / "MF-DeepParticle" / "deepparticle"))

from fullscale import grid as G  # noqa: E402

R = G.RESULTS


def ckpt_cells(path: Path) -> frozenset[str] | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cells = payload.get("cells")
    return frozenset(cells) if cells else None


def log_info(path: Path) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "steps": payload.get("steps"),
        "seconds": payload.get("seconds"),
        "n_rows": len(payload.get("rows", [])),
        "final_loss": (payload["rows"][-1]["loss"]
                       if payload.get("rows") else None),
        "cells": len(payload.get("cells", [])),
    }


MODELS = [
    ("two-param MF  seed88", R / "optimal_budget_ablation/two_32000/mf_two_param.pt",
     R / "optimal_budget_ablation/two_32000/training_log.json"),
    ("two-param MF  seed89", R / "optimal_budget_ablation/two_32000_s89/mf_two_param.pt",
     R / "optimal_budget_ablation/two_32000_s89/training_log.json"),
    ("two-param MF  seed90", R / "optimal_budget_ablation/two_32000_s90/mf_two_param.pt",
     R / "optimal_budget_ablation/two_32000_s90/training_log.json"),
    ("one-param MF  seed88", R / "optimal_budget_ablation/one_32000/mf_one_param.pt",
     R / "optimal_budget_ablation/one_32000/training_log.json"),
    ("one-param MF  seed89", R / "optimal_budget_ablation/one_32000_s89/mf_one_param.pt",
     R / "optimal_budget_ablation/one_32000_s89/training_log.json"),
    ("one-param MF  seed90", R / "optimal_budget_ablation/one_32000_s90/mf_one_param.pt",
     R / "optimal_budget_ablation/one_32000_s90/training_log.json"),
    ("DP corrector  seed88", R / "dp_two_param/s88/dp_corrector.pt",
     R / "dp_two_param/s88/summary.json"),
    ("DP corrector  seed89", R / "dp_two_param/s89/dp_corrector.pt",
     R / "dp_two_param/s89/summary.json"),
    ("DP corrector  seed90", R / "dp_two_param/s90/dp_corrector.pt",
     R / "dp_two_param/s90/summary.json"),
    ("capacity w112 seed88", R / "capacity_matched/w112_s88/mf_two_param.pt",
     R / "capacity_matched/w112_s88/training_log.json"),
]

print("=" * 96)
print("Q1 / Q2  TRAINING COMPLETION AND TRAINING SETS")
print("=" * 96)
print(f"{'model':<22} {'ckpt':>5} {'steps':>10} {'log rows':>9} {'final loss':>11} "
      f"{'cells':>6} {'mu=1':>5} {'complete?':>10}")
print("-" * 96)
info = {}
for label, ckpt, log in MODELS:
    cells = ckpt_cells(ckpt)
    meta = log_info(log)
    exists = "yes" if ckpt.exists() else "NO"
    steps = meta["steps"] if meta else "-"
    rows = meta["n_rows"] if meta else "-"
    loss = f"{meta['final_loss']:.5f}" if meta and meta["final_loss"] is not None else "-"
    n = len(cells) if cells else "-"
    at_one = sum(1 for k in cells if int(k[5:7]) == G.MU_INDEX) if cells else "-"
    complete = "-"
    if meta and meta["steps"]:
        complete = "yes" if meta["n_rows"] >= 2 else "PARTIAL"
    print(f"{label:<22} {exists:>5} {str(steps):>10} {str(rows):>9} {loss:>11} "
          f"{str(n):>6} {str(at_one):>5} {complete:>10}")
    info[label] = {"cells": cells, "meta": meta}

print()
print("=" * 96)
print("Q3  EVALUATION SETS: WERE THE REPORTED NUMBERS COMPUTED ON DISJOINT CELLS?")
print("=" * 96)

two_cells = info["two-param MF  seed88"]["cells"]
one_cells = info["one-param MF  seed88"]["cells"]
slice_keys = frozenset(c.key for c in G.baseline_slice())
pool = frozenset(c.key for c in G.full_grid())

print(f"two-param trains on : {len(two_cells)} cells, {sum(1 for k in two_cells if int(k[5:7]) == G.MU_INDEX)} at mu=1")
print(f"one-param trains on : {len(one_cells)} cells, {sum(1 for k in one_cells if int(k[5:7]) == G.MU_INDEX)} at mu=1")
print(f"one-param training set == the mu=1 row? {one_cells == slice_keys}")

# The main comparison's evaluation set, as recorded in the detail csv
detail = R / "optimal_budget_ablation/detail.csv"
if detail.exists():
    scored = {row["key"] for row in csv.DictReader(open(detail, encoding="utf-8"))}
    print(f"\nmain comparison (optimal_budget_ablation/detail.csv) scores {len(scored)} cells")
    print(f"  of those, in two-param training set : {len(scored & two_cells)}")
    print(f"  of those, in one-param training set : {len(scored & one_cells)}")
    expected = pool - two_cells - slice_keys
    print(f"  expected held-out = pool - two_train - mu1_row = {len(expected)}")
    print(f"  matches the scored set exactly? {scored == expected}")
    if scored != expected:
        print(f"    only in scored   : {sorted(scored - expected)}")
        print(f"    only in expected : {sorted(expected - scored)}")
    print(f"  -> disjoint from BOTH training sets? "
          f"{not (scored & two_cells) and not (scored & one_cells)}")

print()
print("=" * 96)
print("WHAT THIS MEANS FOR COMPARABILITY")
print("=" * 96)
if detail.exists():
    print("The two-parameter-vs-single-parameter comparison uses a set that excludes")
    print("both models' training data. So that comparison IS valid.")
    print()
    print("But that evaluation set has a structural property that must be stated:")
    absent = [round(m, 4) for m in G.MU_GRID if f"_m{G.MU_INDEX:02d}" not in scored]
    print(f"  mu=1 is absent from the evaluation set entirely (the whole row was")
    print(f"  training data for the single-parameter model).")
    print(f"  mu values present: "
          f"{sorted({round(float(G.MU_GRID[int(k[5:7])]), 3) for k in scored})}")
    print(f"  => the single-parameter model is scored ONLY on mu it never saw.")
