"""Establish the exact protocol behind every one-vs-two number in the repository.

Three comparisons exist and they differ in (a) which mu the sigma-only model trains on,
(b) the training budget, and (c) which cells are held out. Mixing them up is what makes
the headline percentage ambiguous, so this prints all three side by side from the saved
artifacts.

The three protocols
-------------------
P1  headline / three-seed
    sigma-only:  mu = 1 row, 20 cells, 32,000 updates per cell
    two-param :  LHS-40,     40 cells, 32,000 updates per cell
    held out  :  143 cells = 200 - 40 - 20 + 3   (the LHS grid contains 3 cells at mu=1)

P2  fair control, same-cells  (the information-gap diagnostic)
    sigma-only:  the same LHS-40 cells, but with mu REMOVED from its inputs
    two-param :  LHS-40
    held out  :  144 cells (excludes the whole mu=1 row conservatively)
    budget    :  2,000 per cell  (both models, so the comparison is budget-matched
                 but both are far from converged)

P3  fair control, slice-20
    sigma-only:  mu = 1 row, 20 cells
    two-param :  LHS-40
    held out  :  144 cells
    budget    :  2,000 per cell

What this script computes
-------------------------
* the budget each model actually received, from its training log;
* the overlap correction (how many LHS cells sit at mu = 1), which is why the held-out
  count is 143 in P1 and 144 in P2/P3;
* per-mu breakdown of P1, so the reader can see that the gain is largest where the
  sigma-only model has no information and smallest near mu = 1.

Usage:
    python scripts/protocol_audit.py
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

R = ROOT / "results" / "fullscale"


def budget(path: Path) -> tuple[int, int, float]:
    log = json.loads(path.read_text(encoding="utf-8"))
    cells = len(log.get("cells", []))
    steps = int(log.get("steps", 0))
    return cells, steps, steps / max(cells, 1)


def main() -> int:
    print("=" * 78)
    print("P1  HEADLINE / THREE-SEED PROTOCOL")
    print("=" * 78)
    lhs_keys = {c.key for c in G.full_grid()}  # placeholder, refined below
    design = json.loads((R / "optimal_budget_ablation/summary.json").read_text("utf-8"))
    print(f"  updates per cell          : {design['updates_per_cell']:,}")
    print(f"  held-out cells            : {design['n_held_out']}")
    print(f"  one-param held-out W2     : {design['one_param_mean_w2']:.5f}")
    print(f"  two-param held-out W2     : {design['two_param_mean_w2']:.5f}")
    print(f"  relative reduction        : {design['relative_reduction']:+.1%}")
    print(f"  per-cell wins (two/one)   : {design['two_param_wins']}/{design['one_param_wins']}")
    print(f"  corr(gain, |log10 mu|)    : {design['corr_gain_vs_mu_distance']:.3f}")

    # How many LHS cells sit at mu = 1: this is the overlap in the held-out arithmetic.
    fc_log = json.loads((R / "fair_control/same-cells/training_log.json").read_text("utf-8"))
    at_one = [k for k in fc_log["cells"] if int(k[5:7]) == G.MU_INDEX]
    print(f"\n  LHS cells with mu = 1     : {len(at_one)}  -> {', '.join(at_one)}")
    print(f"  held-out arithmetic       : 200 - 40 - 20 + {len(at_one)}"
          f" = {200 - 40 - 20 + len(at_one)}")

    print(f"\n  {'mu':>8} {'n':>4} {'one-param':>10} {'two-param':>10} {'gain':>9} {'rel':>8}")
    bands = design["by_mu_band"]
    for mu in sorted(bands, key=lambda s: float(s)):
        b = bands[mu]
        rel = 1 - b["two"] / b["one"]
        print(f"  {float(mu):8.3f} {b['n']:4d} {b['one']:10.5f} {b['two']:10.5f} "
              f"{b['one'] - b['two']:+9.5f} {rel:+8.1%}")

    print()
    print("=" * 78)
    print("BUDGETS, FROM THE TRAINING LOGS")
    print("=" * 78)
    print(f"  {'run':<34} {'cells':>6} {'steps':>9} {'per cell':>10}")
    for label, path in (
        ("two-param headline s88", R / "optimal_budget_ablation/two_32000/training_log.json"),
        ("one-param headline (mu=1 row)", R / "optimal_budget_ablation/one_32000/training_log.json"),
        ("two-param used by fair_control", R / "train/lhs40/training_log.json"),
        ("one-param same-cells", R / "fair_control/same-cells/training_log.json"),
        ("one-param slice-20", R / "fair_control/slice-20/training_log.json"),
        ("one-param same-4", R / "fair_control/same-4/training_log.json"),
    ):
        if not path.exists():
            print(f"  {label:<34}   (log absent)")
            continue
        cells, steps, per = budget(path)
        print(f"  {label:<34} {cells:6d} {steps:9d} {per:10.0f}")

    print()
    print("=" * 78)
    print("P2 / P3  FAIR-CONTROL PROTOCOL")
    print("=" * 78)
    fc = json.loads((R / "fair_control/summary.json").read_text("utf-8"))
    print(f"  updates per cell          : {fc['updates_per_cell']:,}")
    print(f"  held-out cells            : {fc['n_heldout']}")
    print(f"  {'control':<12} {'cells':>6} {'train fit':>10} {'held-out':>10} {'rel vs two':>11}")
    print(f"  {'two-param':<12} {fc['n_lhs_cells']:6d} {fc['two_param_training_fit']:10.5f} "
          f"{fc['two_param_heldout']:10.5f} {'-':>11}")
    for name, b in fc["controls"].items():
        print(f"  {name:<12} {b['n_cells']:6d} {b['training_fit_w2']:10.5f} "
              f"{b['heldout_mean_w2']:10.5f} {b['relative_reduction']:+11.1%}")

    print()
    print("=" * 78)
    print("WHY P2 IS THE PROTOCOL THAT ANSWERS 'DOES THE EXTRA INPUT HELP?'")
    print("=" * 78)
    same = fc["controls"]["same-cells"]
    print("  In P2 both models see the SAME 40 cells and the SAME budget. The only")
    print("  difference is that mu is removed from the sigma-only model's input vector.")
    print(f"  sigma-only training fit : {same['training_fit_w2']:.5f}")
    print(f"  two-param  training fit : {fc['two_param_training_fit']:.5f}")
    print(f"  gap                     : {same['training_fit_w2'] - fc['two_param_training_fit']:.5f}"
          f"  ({same['training_fit_w2'] / fc['two_param_training_fit']:.2f}x)")
    print("  This gap is measured on TRAINING cells, so it cannot be explained by")
    print("  generalisation or by mu-extrapolation. It is the irreducible cost of not")
    print("  knowing mu, because one (sigma -> cloud) map cannot represent ten clouds.")
    print()
    print("  Caveat that must be stated: P2 runs at 2,000 updates/cell while P1 runs at")
    print("  32,000, and the two-param fit at 2,000 (0.1532) is worse than at 32,000")
    print("  (0.1200). The RATIO is therefore measured at an undertrained operating point;")
    print("  the honest claim is 'the gap exists and is large', not 'the gap is exactly 1.63x'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
