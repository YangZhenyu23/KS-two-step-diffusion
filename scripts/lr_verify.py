"""Was the learning-rate gain real, or an artifact of the measurement?

The schedule comparison produced a clean monotone ordering

    fixed 0.11532  >  cosine 0.10279  >  warmcos 0.09784

yet the *tail loss variance* did not change at all (ratio 1.00 for both). That
combination is suspicious: the stated diagnostic said "variance must fall", and it
did not. Either the schedules helped for a different reason, or the gain is noise.

Why the tail-loss diagnostic was unusable
----------------------------------------
The training log records the loss of a *single minibatch* every 200 steps. With
batch 4096 out of 40 cells and roughly lognormal per-sample displacement errors,
the single-batch loss has a very wide distribution: in all three runs the last-5%
window spans 0.012 to 1.08. Its variance is dominated by minibatch sampling, not
by the learning rate. So "tail loss sd" cannot detect optimisation stability at
all -- the diagnostic was wrong, not the schedules.

What this script checks instead
-------------------------------
1. **Paired per-cell comparison.** Score the three models on the same 143 held-out
   cells and compare cell by cell. A real gain should win consistently across
   cells, not just on the mean.
2. **Sign test.** Under the null (no difference) the win count is binomial; the
   script reports how many standard deviations the observed count is from half.
3. **Correlation between schedules.** If the improvement is a genuine property of
   the model, the three schedules' per-cell errors should be highly correlated
   and their *differences* should be small relative to the schedule-to-schedule
   spread of a single training run.

Because there is only one training seed per schedule, a single seed can only show
*consistency*, not significance. This script states that limitation explicitly
and reports what a second seed would need to show to settle the question.

Usage:
    python scripts/lr_verify.py
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

from fullscale import grid as G  # noqa: E402
from fullscale.evaluate import score_many  # noqa: E402
from fullscale.model import sample_two_param  # noqa: E402
from fullscale.train import load_cell, load_two  # noqa: E402

from sample_efficiency import frozen_test_set, stratified_subset, write_csv, write_json, log  # noqa: E402

OUT = G.RESULTS / "lr_schedule"


def main() -> int:
    test, pool = frozen_test_set()
    design = stratified_subset(pool, 40, seed=20261107)
    design_keys = {c.key for c in design}
    slice_keys = {c.key for c in G.baseline_slice()}
    held = [c for c in G.full_grid() if c.key not in design_keys and c.key not in slice_keys]
    log(f"evaluate on {len(held)} held-out cells")

    payloads = {c.key: load_cell(c) for c in held}

    scores = {}
    for schedule in ("fixed", "cosine", "warmcos"):
        path = OUT / f"{schedule}_model.pt"
        if not path.is_file():
            log(f"{schedule}: no saved model ({path.name}); skipping")
            continue
        model, _ = load_two(path)
        pairs = {}
        for cell in held:
            payload = payloads[cell.key]
            pairs[cell.key] = (payload["clouds"]["reference"],
                               sample_two_param(model, torch.from_numpy(payload["clouds"]["source"]),
                                                cell.sigma, cell.mu).numpy().astype(np.float32))
        rows = score_many(pairs, bins=G.BINS, workers=4, log=lambda t: None)
        scores[schedule] = {r["key"]: r["w2"] for r in rows}
        log(f"{schedule}: mean {np.mean(list(scores[schedule].values())):.5f}")
        del model
        torch.cuda.empty_cache()

    if len(scores) < 2:
        raise SystemExit("need at least two schedules with saved models; "
                         "re-run scripts/lr_schedule.py with --save-models")

    names = list(scores)
    detail = []
    for cell in held:
        row = {**cell.meta()}
        for name in names:
            row[f"w2_{name}"] = scores[name][cell.key]
        detail.append(row)
    write_csv(OUT / "per_cell.csv", detail)

    # ---- paired comparisons ----------------------------------------------
    report = {"n_cells": len(held), "means": {}, "pairs": {}}
    for name in names:
        report["means"][name] = float(np.mean([r[f"w2_{name}"] for r in detail]))

    print(f"\n{'comparison':<24} {'mean diff':>10} {'wins':>10} {'z':>7}  verdict")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            diffs = np.array([r[f"w2_{a}"] - r[f"w2_{b}"] for r in detail])
            wins = int((diffs > 0).sum())
            n = len(diffs)
            # sign test under the null of no difference
            z = (wins - n / 2) / math.sqrt(n / 4)
            verdict = ("real" if abs(z) > 2 else "not resolvable")
            key = f"{a}_vs_{b}"
            report["pairs"][key] = {"mean_diff": float(diffs.mean()), "wins": wins,
                                    "n": n, "z": float(z), "verdict": verdict}
            print(f"{key:<24} {diffs.mean():+10.5f} {wins:>4}/{n:<4} {z:+7.2f}  {verdict}")

    # ---- how spread out are the cells? -----------------------------------
    print("\nper-cell spread of the schedule difference")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            diffs = np.array([r[f"w2_{a}"] - r[f"w2_{b}"] for r in detail])
            print(f"  {a} vs {b}: mean {diffs.mean():+.5f}  sd {diffs.std(ddof=1):.5f}  "
                  f"median {np.median(diffs):+.5f}")

    write_json(OUT / "lr_verify.json", report)
    print(f"\nwrote {OUT / 'lr_verify.json'}")
    print("\nlimitation: one training seed per schedule. A monotone ordering across")
    print("three schedules on identical data and identical evaluation cells is")
    print("strong evidence of a real effect, but a second seed per schedule is what")
    print("a paper would need to claim a specific percentage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
