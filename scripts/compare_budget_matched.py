"""Budget-matched comparison: one-parameter vs two-parameter at equal convergence.

The first comparison (``compare_one_vs_two.py``) held the *number of gradient
steps* fixed at 1000. That is not a fair test of the second parameter: the
two-parameter model spreads those 1000 steps over 20 cells (50 per cell) while
the one-parameter model spreads them over 10 (100 per cell), so the comparison
partly measures per-cell training budget rather than the value of mu.

This script removes that confound in three ways.

1. **Per-cell step parity.** The two-parameter run gets ``steps x n_cells``
   gradient updates, so each cell receives the same number of updates as the
   single-parameter control.

2. **Convergence check.** Loss histories are reported so a reader can see
   whether either model was still improving when training stopped.

3. **Two scoring protocols.**
   * ``mu = 1 slice``: both models scored on the ten sigma values that the
     one-parameter model was trained on. The two-parameter model has also seen
     some of these cells; the slice is reported for reference.
   * ``held-out cells``: cells in neither model's training set. This is the only
     protocol that tests generalisation on both parameters, and it is the one
     that answers "does the second parameter earn its keep".

The output is ``results/twoparam/compare/budget_matched.json`` plus a CSV with
the per-cell breakdown.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from twoparam import protocol as P  # noqa: E402
from twoparam.data import load_cell  # noqa: E402
from twoparam.evaluate import evaluate_cells, write_csv, write_json  # noqa: E402
from twoparam.model import TwoParamMeanflow, mf_sample  # noqa: E402
from twoparam.protocol import Config  # noqa: E402
from twoparam.train import load_model, train as train_two  # noqa: E402

from compare_one_vs_two import OneParamMeanflow, sample_one_param, train_one_param  # noqa: E402


def main() -> int:
    config = Config(n_train=P.N_TRAIN, device="cuda")
    out = P.ensure_results("compare")
    slice_points = P.baseline_slice()
    full = P.full_grid()
    index = P.grid_index()
    log = lambda t: print(t, flush=True)

    # ---- design ----------------------------------------------------------
    from twoparam.cli import _design
    design = _design("lhs20", n_points=20)
    lhs_points = [index[i, j] for i, j in design.points]
    lhs_keys = {p.key for p in lhs_points}
    slice_keys = {p.key for p in slice_points}
    held_keys = {p.key for p in full} - lhs_keys - slice_keys
    held_points = [p for p in full if p.key in held_keys]
    log(f"cells: LHS-20={len(lhs_keys)}  mu=1 slice={len(slice_keys)}  "
        f"held out from both={len(held_points)}")

    # ---- budget-matched training ----------------------------------------
    per_cell = 400          # updates per training cell
    one_steps = per_cell * len(slice_points)   # 10 cells
    two_steps = per_cell * len(lhs_points)     # 20 cells
    log(f"budget: {per_cell} updates/cell -> single-param {one_steps} steps, "
        f"two-param {two_steps} steps")

    log("\ntraining the single-parameter control (sigma only, mu = 1)")
    one = train_one_param(slice_points, config, steps=one_steps)

    log("\ntraining the two-parameter model (sigma, mu) on LHS-20")
    train_dir = P.ensure_results("train", "lhs20_budget")
    two_summary = train_two(P.RESULTS / "data" / "full", train_dir, lhs_points, config,
                            steps=two_steps, batch_size=256, seed=88, log=lambda t: None)
    two = load_model(train_dir / "checkpoints" / "mf_two_param.pt")
    log(f"  final loss {two_summary['final_loss']:.6g}")

    # ---- scoring ---------------------------------------------------------
    def score(points, label):
        pred_one, pred_two, ref = {}, {}, {}
        for point in points:
            cell = load_cell(P.RESULTS / "data" / "full", point.key)
            src = cell["clouds"]["source"]
            pred_one[point.key] = sample_one_param(one, src, point.sigma)
            pred_two[point.key] = mf_sample(two, torch.from_numpy(src), point.sigma, point.mu).numpy()
            ref[point.key] = cell["clouds"]["reference"]
        rows_one, _ = evaluate_cells(pred_one, ref, bins=config.bins, workers=4, log=lambda t: None)
        rows_two, _ = evaluate_cells(pred_two, ref, bins=config.bins, workers=4, log=lambda t: None)
        by_one = {r["key"]: r["w2"] for r in rows_one}
        by_two = {r["key"]: r["w2"] for r in rows_two}
        detail = []
        for point in points:
            detail.append({**point.meta(), "w2_one_param": by_one[point.key],
                           "w2_two_param": by_two[point.key],
                           "improvement": by_one[point.key] - by_two[point.key],
                           "ratio": by_two[point.key] / by_one[point.key],
                           "mu_off_baseline": abs(math.log10(point.mu))})
        m1 = float(np.mean([r["w2_one_param"] for r in detail]))
        m2 = float(np.mean([r["w2_two_param"] for r in detail]))
        gains = [r["improvement"] for r in detail]
        summary = {
            "label": label,
            "n_cells": len(detail),
            "one_param_mean_w2": m1,
            "two_param_mean_w2": m2,
            "mean_improvement": float(np.mean(gains)),
            "median_improvement": float(np.median(gains)),
            "relative_reduction": float(1 - m2 / m1) if m1 else None,
            "two_param_wins": int(sum(1 for v in gains if v > 0)),
            "one_param_wins": int(sum(1 for v in gains if v < 0)),
            "max_one_param_w2": float(np.max([r["w2_one_param"] for r in detail])),
            "max_two_param_w2": float(np.max([r["w2_two_param"] for r in detail])),
        }
        corr = None
        if len(detail) > 2:
            off = np.array([r["mu_off_baseline"] for r in detail])
            imp = np.array(gains)
            if off.std() > 0 and imp.std() > 0:
                corr = float(np.corrcoef(off, imp)[0, 1])
        summary["corr_improvement_vs_mu_distance"] = corr
        log(f"\n[{label}]  cells={len(detail)}")
        log(f"  single-param mean W2 = {m1:.5f}")
        log(f"  two-param    mean W2 = {m2:.5f}")
        log(f"  relative reduction   = {summary['relative_reduction']:+.1%}")
        log(f"  two-param wins {summary['two_param_wins']}/{len(detail)}")
        log(f"  corr(gain, |log10 mu|) = {corr}")
        return summary, detail

    slice_summary, slice_detail = score(slice_points, "mu=1 slice (seen by single-param)")
    held_summary, held_detail = score(held_points, "held out from both models")

    write_csv(out / "budget_matched_slice.csv", slice_detail)
    write_csv(out / "budget_matched_heldout.csv", held_detail)
    write_json(out / "budget_matched.json", {
        "protocol": {
            "updates_per_cell": per_cell,
            "single_param_steps": one_steps,
            "two_param_steps": two_steps,
            "single_param_cells": len(slice_points),
            "two_param_cells": len(lhs_points),
            "particles": config.n_train,
            "bins": config.bins,
            "note": ("Both models receive the same number of updates per training cell, "
                     "so the comparison is not confounded by per-cell budget."),
        },
        "slice": slice_summary,
        "heldout": held_summary,
        "two_param_training": two_summary,
    })
    log(f"\nwrote {out / 'budget_matched.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
