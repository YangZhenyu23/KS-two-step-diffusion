"""Training-budget sweep: is the remaining error due to under-training?

Why this is the only legal direction left
-----------------------------------------
The project's core advantage over DeepParticle is that MeanFlow generates a cloud
in a *single* forward pass. Any change that adds inference cost therefore
undermines the main contribution (docs/PROJECT_RATIONALE.md 2). That rules out
bigger networks (also measured useless), multi-step sampling, and the DP
corrector as a primary path. Data volume is already saturated (80 of 160 cells
suffice) and the sampling criterion has been ruled out.

What remains is the training budget, which leaves inference untouched.

The diagnostic logic
--------------------
Three outcomes are possible and they mean different things:

  * **both training fit and test error fall** -> under-trained. Increasing the
    budget is a free win: accuracy improves with identical inference cost.
  * **training fit falls but test error does not** -> over-fitting; the data is
    the limit. (Already contradicted by the learning curve, so this outcome
    would be surprising.)
  * **neither moves** -> the optimiser has converged and the remaining error is
    a limit of the representation, not of the budget.

The training-fit error is therefore the decisive measurement, and it is reported
alongside the test error for every budget.

Protocol
--------
Architecture, data, optimiser, learning rate and batch size are held fixed. Only
the updates per cell vary. The same 40-cell design and the same frozen test set
are used for every run, so differences are attributable to the budget alone.

Usage:
    python scripts/budget_sweep.py --budgets 2000 8000 --seeds 1
    python scripts/budget_sweep.py --budgets 32000 --seeds 2
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fullscale import grid as G  # noqa: E402
from fullscale.evaluate import score_many  # noqa: E402
from fullscale.grid import Config  # noqa: E402
from fullscale.model import sample_two_param  # noqa: E402
from fullscale.train import load_cell, load_two, train_two_param  # noqa: E402

from sample_efficiency import (  # noqa: E402
    OUT, frozen_test_set, stratified_subset, write_csv, write_json, log,
)

OUT_BUDGET = G.RESULTS / "budget_sweep"


def score(model, cells: list[G.Cell], payloads: dict) -> float:
    pairs = {}
    for cell in cells:
        payload = payloads[cell.key]
        pairs[cell.key] = (
            payload["clouds"]["reference"],
            sample_two_param(model, torch.from_numpy(payload["clouds"]["source"]),
                             cell.sigma, cell.mu).numpy().astype(np.float32),
        )
    rows = score_many(pairs, bins=G.BINS, workers=4, log=lambda t: None)
    return float(np.mean([r["w2"] for r in rows]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=int, nargs="+", default=[2000, 8000])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    test, pool = frozen_test_set()
    train_cells = stratified_subset(pool, 40, seed=20261107)
    log(f"train {len(train_cells)} cells, test {len(test)} cells, "
        f"budgets {a.budgets}, {a.seeds} seed(s)")

    train_payloads = {c.key: load_cell(c) for c in train_cells}
    test_payloads = {c.key: load_cell(c) for c in test}
    config = Config(n_train=G.N_PARTICLES, device="cuda")

    OUT_BUDGET.mkdir(parents=True, exist_ok=True)
    rows = []
    for budget in a.budgets:
        for seed in (88, 89)[: a.seeds]:
            name = f"b{budget}_s{seed}"
            work = OUT_BUDGET / name
            steps = budget * len(train_cells)
            log(f"--- {name}: {steps:,} steps ---")
            started = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                summary = train_two_param(train_cells, config, work, steps=steps,
                                          batch_size=a.batch_size, lr=a.lr, seed=seed)
            seconds = time.perf_counter() - started
            model, _ = load_two(work / "mf_two_param.pt")

            fit = score(model, train_cells, train_payloads)
            held = score(model, test, test_payloads)
            row = {
                "budget_per_cell": budget,
                "seed": seed,
                "steps": steps,
                "seconds": seconds,
                "final_loss": summary["final_loss"],
                "train_fit_w2": fit,
                "test_w2": held,
                "generalisation_gap": held - fit,
                "test_over_floor": held / G.FLOOR_16,
            }
            rows.append(row)
            log(f"  train-fit {fit:.5f}  TEST {held:.5f}  "
                f"gap {held - fit:+.5f}  ({held / G.FLOOR_16:.2f}x floor)  {seconds:.0f}s")
            del model
            torch.cuda.empty_cache()

    write_csv(OUT_BUDGET / "budget_sweep.csv", rows)

    print(f"\n{'budget':>7} {'seed':>5} {'steps':>10} {'train fit':>10} {'TEST':>9} "
          f"{'gap':>8} {'x floor':>8}")
    for r in rows:
        print(f"{r['budget_per_cell']:7d} {r['seed']:5d} {r['steps']:10,d} "
              f"{r['train_fit_w2']:10.5f} {r['test_w2']:9.5f} "
              f"{r['generalisation_gap']:+8.5f} {r['test_over_floor']:8.2f}")

    # verdict per budget
    by_budget = {}
    for r in rows:
        by_budget.setdefault(r["budget_per_cell"], []).append(r)
    summary_out = {}
    print()
    for budget in sorted(by_budget):
        group = by_budget[budget]
        fit = float(np.mean([r["train_fit_w2"] for r in group]))
        held = float(np.mean([r["test_w2"] for r in group]))
        summary_out[str(budget)] = {"n": len(group), "train_fit": fit, "test": held}
    base = summary_out[str(min(by_budget))]
    for budget in sorted(by_budget):
        entry = summary_out[str(budget)]
        entry["delta_vs_lowest"] = entry["test"] - base["test"]
        entry["fit_delta_vs_lowest"] = entry["train_fit"] - base["train_fit"]

    print("verdict:")
    for budget in sorted(summary_out):
        entry = summary_out[str(budget)]
        print(f"  budget {budget:>6}: fit {entry['train_fit']:.5f} "
              f"({entry['fit_delta_vs_lowest']:+.5f} vs lowest)  "
              f"test {entry['test']:.5f} ({entry['delta_vs_lowest']:+.5f})")
    print("\n  fit falls + test falls   -> under-trained; raise the budget (free win)")
    print("  fit falls + test flat    -> over-fitting; data-limited")
    print("  neither moves            -> representation limit, not budget")

    write_json(OUT_BUDGET / "budget_sweep.json", {"rows": rows, "summary": summary_out})
    print(f"\nwrote {OUT_BUDGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
