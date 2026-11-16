"""P0: the information-gap control, re-run at the headline budget and on the headline cells.

What was wrong with the existing control
----------------------------------------
``fair_control/same-cells`` already measures the information gap: a sigma-only network
trained on the same cells as the two-parameter model, with mu removed from its inputs.
Two problems made it unusable as the headline evidence:

  1. **Budget mismatch.** It ran at 2,000 updates per cell while the headline model ran at
     32,000. The two-parameter model's own training fit improves from 0.1532 at 2,000 steps
     to 0.1200 at 32,000, so the 1.63x ratio was measured at an arbitrary, undertrained
     operating point.
  2. **Different design.** ``fair_control.py`` builds its 40 cells with
     ``latin_hypercube(40)`` and compares against ``train/lhs40`` (design S4). The headline
     numbers come from ``optimal_budget_ablation/two_32000`` (design S1). S1 and S4 share
     only 4 cells, so the control was not even measuring the same experiment.

What this script does
---------------------
  1. Reads the training cells from the headline checkpoint (design S1) - never recomputes.
  2. Trains a sigma-only network on exactly those 40 cells, with mu allowed to vary
     (``allow_mixed_mu=True``) - this is the deliberate diagnostic, not a usable model.
  3. Uses the SAME budget as the headline: 32,000 updates per cell.
  4. Reports the training-fit error of both models on those 40 cells. The sigma-only fit is
     an information floor: one map sigma -> cloud cannot represent ten different clouds, so
     no amount of training removes this error.
  5. Also re-scores the two-parameter checkpoint's training fit at the same budget, so the
     ratio is like-for-like.

Cost: 40 cells x 32,000 = 1.28M steps, roughly 2 hours on the laptop 4060.

Usage:
    python scripts/p0_same_cells_budget.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream" / "MF-DeepParticle" / "deepparticle"))

from fullscale import grid as G  # noqa: E402
from fullscale.train import (  # noqa: E402
    Config, load_cell, load_one, load_two, train_one_param,
)
from fullscale.model import sample_one_param, sample_two_param  # noqa: E402
from fullscale.evaluate import score_many  # noqa: E402

OUT = G.RESULTS / "p0_same_cells"
BUDGET = 32000  # updates per cell, matching optimal_budget_ablation/two_32000


def main() -> int:
    torch.set_num_threads(min(8, torch.get_num_threads()))
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda text: print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)

    headline = G.RESULTS / "optimal_budget_ablation" / "two_32000" / "mf_two_param.pt"
    payload = torch.load(headline, map_location="cpu", weights_only=False)
    train_keys = list(payload["cells"])
    cells = [G.Cell(int(k[1:3]), int(k[5:7])) for k in train_keys]
    log(f"headline checkpoint trains on {len(cells)} cells: "
        f"{sum(1 for c in cells if abs(c.mu - 1.0) < 1e-12)} of them at mu=1")
    log(f"budget: {BUDGET} updates per cell -> {BUDGET * len(cells):,} steps")

    config = Config(n_train=G.N_PARTICLES, device="cuda")

    # ---- the sigma-only diagnostic on exactly these cells ------------------
    work = G.ensure("p0_same_cells", f"same_cells_{BUDGET}")
    checkpoint = work / "mf_one_param.pt"
    steps = BUDGET * len(cells)
    if checkpoint.exists():
        log("reusing existing checkpoint")
        one, _ = load_one(checkpoint, "cuda")
        seconds = None
    else:
        log("training sigma-only network on the headline cells (mixed mu)")
        result = train_one_param(cells, config, work, steps=steps, batch_size=4096,
                                 allow_mixed_mu=True)
        one, _ = load_one(checkpoint, "cuda")
        seconds = result["seconds"]
        log(f"  done in {seconds:.0f}s")

    # ---- the two-parameter model at the same budget ------------------------
    two, _ = load_two(headline, "cuda")

    def score(predict, cell_list, label) -> float:
        pairs = {}
        for cell in cell_list:
            item = load_cell(cell)
            pairs[cell.key] = (item["clouds"]["reference"],
                               predict(item["clouds"]["source"], cell))
        rows = score_many(pairs, bins=G.BINS, workers=4, log=lambda t: None)
        values = [r["w2"] for r in rows]
        log(f"  {label}: {np.mean(values):.5f}  (n={len(values)})")
        return float(np.mean(values))

    log("scoring training fit on the 40 training cells")
    one_train = score(lambda src, cell: sample_one_param(one, torch.from_numpy(src),
                                                         cell.sigma), cells, "one-param train fit")
    two_train = score(lambda src, cell: sample_two_param(two, torch.from_numpy(src),
                                                         cell.sigma, cell.mu,
                                                         steps=1).numpy().astype(np.float32),
                      cells, "two-param train fit")

    # ---- held-out, derived from the checkpoint -----------------------------
    slice_keys = {c.key for c in G.baseline_slice()}
    train_set = set(train_keys)
    held = [c for c in G.full_grid()
            if c.key not in train_set and c.key not in slice_keys]
    log(f"held-out cells (from checkpoint): {len(held)}")
    one_held = score(lambda src, cell: sample_one_param(one, torch.from_numpy(src),
                                                        cell.sigma), held, "one-param held-out")
    two_held = score(lambda src, cell: sample_two_param(two, torch.from_numpy(src),
                                                        cell.sigma, cell.mu,
                                                        steps=1).numpy().astype(np.float32),
                     held, "two-param held-out")

    gap = one_train - two_train
    summary = {
        "budget_per_cell": BUDGET,
        "steps": steps,
        "n_train_cells": len(cells),
        "n_train_cells_at_mu1": sum(1 for c in cells if abs(c.mu - 1.0) < 1e-12),
        "n_held_out": len(held),
        "one_param_training_seconds": seconds,
        "one_param_training_fit": one_train,
        "two_param_training_fit": two_train,
        "training_fit_gap": gap,
        "training_fit_ratio": one_train / two_train,
        "one_param_held_out": one_held,
        "two_param_held_out": two_held,
        "held_out_ratio": one_held / two_held,
        "train_cells": train_keys,
        "note": ("sigma-only network trained on the SAME 40 cells as the headline "
                 "two-parameter checkpoint, with mu varying across cells but absent from "
                 "its inputs. Its training-fit error is an information floor: one map "
                 "sigma -> cloud cannot represent ten distinct clouds."),
    }
    (OUT / f"p0_budget{BUDGET}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'':<26} {'one-param':>12} {'two-param':>12} {'ratio':>8}")
    print(f"{'training fit (40 cells)':<26} {one_train:12.5f} {two_train:12.5f} "
          f"{one_train / two_train:8.2f}x")
    print(f"{'held-out':<26} {one_held:12.5f} {two_held:12.5f} {one_held / two_held:8.2f}x")
    print(f"\ninformation gap at matched budget: {gap:.5f} "
          f"({one_train / two_train:.2f}x) on the training set itself")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
