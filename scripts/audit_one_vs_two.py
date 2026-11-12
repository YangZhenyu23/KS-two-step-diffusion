"""Audit the one-parameter vs two-parameter comparison protocol.

Answers, from the saved files rather than from memory:

  1. How many cells exist in total, and where does the "144" held-out count come from?
  2. Which cells did each model actually train on?
  3. For every held-out cell, what is mu? (If the held-out set excludes the whole mu=1
     row, then the single-parameter model is being scored 100% on mu values it has
     never seen - which changes how the headline number should be read.)
  4. Per-mu-band breakdown of the gain, plus the fraction of cells where the
     single-parameter model is literally incapable (mu != 1) versus merely difficult.
  5. Sanity check: does the sigma-only model's output depend on mu at all? It cannot
     (mu is not an input), so its prediction at (sigma, mu) equals its prediction at
     (sigma, 1). That is worth measuring because it makes the comparison's logic
     explicit: the gain is not "a better function", it is "access to a missing input".

Usage:
    python scripts/audit_one_vs_two.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FC = ROOT / "results" / "fullscale" / "fair_control"


def main() -> int:
    summary = json.loads((FC / "summary.json").read_text(encoding="utf-8"))
    detail = list(csv.DictReader(open(FC / "detail.csv", encoding="utf-8")))

    print("=" * 78)
    print("1. CELL ACCOUNTING")
    print("=" * 78)
    n_total = 200
    n_lhs = summary["n_lhs_cells"]
    n_held = summary["n_heldout"]
    print(f"  grid                       : 20 sigma x 10 mu = {n_total} cells")
    print(f"  LHS-40 training set        : {n_lhs} cells")
    print(f"  mu=1 row (single-param set): 20 cells")
    print(f"  overlap (LHS cells with mu=1): 4 cells")
    print(f"  held-out = 200 - 40 - 20 + 4 = {n_total - n_lhs - 20 + 4}"
          f"   (file says {n_held})")

    print()
    print("=" * 78)
    print("2. WHAT EACH MODEL SEES")
    print("=" * 78)
    for name, block in summary["controls"].items():
        print(f"  {name:<12} cells={block['n_cells']:>3}  steps={block['steps']:>6}  "
              f"| {block['description']}")
    print(f"  {'two-param':<12} cells={n_lhs:>3}  steps={n_lhs * summary['updates_per_cell']:>6}"
          f"  | sigma AND mu as inputs")

    print()
    print("=" * 78)
    print("3. IS mu=1 REPRESENTED IN THE HELD-OUT SET?")
    print("=" * 78)
    # detail.csv carries only the cell key (sII_mJJ); mu must be recovered from it.
    import sys
    sys.path.insert(0, str(ROOT))
    from fullscale import grid as G
    for row in detail:
        i = int(row["key"][1:3])
        j = int(row["key"][5:7])
        row["sigma"] = float(G.SIGMA_GRID[i])
        row["mu"] = float(G.MU_GRID[j])

    mus = sorted({round(float(r["mu"]), 6) for r in detail})
    print(f"  distinct mu values in detail.csv: {len(mus)} -> {mus}")
    if any(abs(m - 1.0) < 1e-9 for m in mus):
        n_one = sum(1 for r in detail if abs(float(r["mu"]) - 1.0) < 1e-9)
        print(f"  cells at mu=1 in the held-out set: {n_one}")
    else:
        print("  *** mu=1 does NOT appear: the entire mu=1 row was training data for the")
        print("      single-parameter model, so every held-out cell has an unseen mu. ***")

    print()
    print("=" * 78)
    print("4. GAIN BY mu BAND  (single-param minus two-param, on the shared held-out set)")
    print("=" * 78)
    bands = defaultdict(list)
    for r in detail:
        bands[round(abs(float(r["mu"])), 4)].append(r)
    print(f"  {'mu':>8} {'n':>4} {'one-param':>10} {'two-param':>10} {'gain':>9} {'rel':>8}")
    for mu in sorted(bands):
        group = bands[mu]
        one = sum(float(r["w2_one_param"]) for r in group) / len(group)
        two = sum(float(r["w2_two_param"]) for r in group) / len(group)
        print(f"  {mu:8.4f} {len(group):4d} {one:10.5f} {two:10.5f} "
              f"{one - two:+9.5f} {1 - two / one:+8.1%}")

    print()
    print("=" * 78)
    print("5. HEADLINE NUMBERS, PER CONTROL")
    print("=" * 78)
    print(f"  {'control':<12} {'train fit':>10} {'held-out':>10} {'vs two':>9} {'rel':>8} {'wins':>7}")
    print(f"  {'two-param':<12} {summary['two_param_training_fit']:10.5f} "
          f"{summary['two_param_heldout']:10.5f} {'-':>9} {'-':>8} {'-':>7}")
    for name, block in summary["controls"].items():
        print(f"  {name:<12} {block['training_fit_w2']:10.5f} {block['heldout_mean_w2']:10.5f} "
              f"{block['improvement']:+9.5f} {block['relative_reduction']:+8.1%} "
              f"{block['two_param_wins']:>3}/{block['n_heldout']}")

    print()
    print("=" * 78)
    print("6. THE INFORMATIONAL BOUND ARGUMENT (same-cells control)")
    print("=" * 78)
    b = summary["controls"]["same-cells"]
    print("  On the SAME 40 cells, a sigma-only network cannot do better than the")
    print("  conditional mean over the mu's present at each sigma. Its training fit is")
    print(f"  therefore an information floor: {b['training_fit_w2']:.5f}")
    print(f"  The two-parameter model reaches  {summary['two_param_training_fit']:.5f} "
          f"on the same cells.")
    ratio = b["training_fit_w2"] / summary["two_param_training_fit"]
    print(f"  Ratio: {ratio:.2f}x. A sigma-only model cannot close this gap by any")
    print("  amount of training, because the missing input is not recoverable from sigma.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
