"""Fair single-parameter control on the *same* training cells as the two-parameter model.

The first ablation compared:

    two-parameter  trained on LHS-40  (40 cells spread over the whole plane)
    one-parameter  trained on mu=1 row (20 cells, one slice)

This is not an equal-footing comparison: the two models see different numbers of
cells, different sigma coverage, and different total budget. A reviewer will ask
exactly that question, so this script removes the confound.

Three controls, all trained with the same updates per cell and scored on the same
held-out cells:

  C1  ``same-cells``   the sigma-only network trained on the **same 40 LHS cells**
                       as the two-parameter model. This is the closest possible
                       apples-to-apples comparison: identical data, identical
                       budget, identical architecture; the only difference is
                       that the sigma-only network cannot see mu. Because those 40
                       cells contain ten distinct mu values, the sigma-only model
                       is being asked to fit a function that is not a function of
                       its input -- which is the *definition* of the information
                       it is missing. The training loss therefore has an
                       irreducible floor, and that floor is a quantitative measure
                       of the value of mu.

  C2  ``slice-20``     sigma-only on the mu=1 row (the original control, 20 cells).

  C3  ``same-4``       sigma-only on the four mu=1 cells that happen to be inside
                       the LHS-40. Smallest possible data set, useful as a
                       lower bound.

Reported for each: mean W2 on the shared held-out set, and the **training** W2,
because a model that cannot lower its own training error is capacity-limited by
information, not by data volume.

Usage:
    python scripts/fair_control.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fullscale import grid as G  # noqa: E402
from fullscale.evaluate import score_many, write_csv, write_json  # noqa: E402
from fullscale.lhs import latin_hypercube  # noqa: E402
from fullscale.model import sample_one_param, sample_two_param  # noqa: E402
from fullscale.grid import Config  # noqa: E402
from fullscale.train import (  # noqa: E402
    load_cell, load_one, load_two, train_one_param, train_two_param,
)

UPDATES_PER_CELL = 2000


def main() -> int:
    config = Config(n_train=G.N_PARTICLES, device="cuda")
    out = G.ensure("fair_control")

    design = latin_hypercube(40)
    lhs_cells = [G.Cell(i, j) for i, j in design.points]
    lhs_keys = {c.key for c in lhs_cells}
    slice_cells = G.baseline_slice()
    slice_keys = {c.key for c in slice_cells}
    same4 = [c for c in lhs_cells if abs(c.mu - G.BASELINE_MU) < 1e-12]

    held = [c for c in G.full_grid() if c.key not in lhs_keys and c.key not in slice_keys]
    print(f"training sets: LHS-40={len(lhs_cells)}  mu=1 row={len(slice_cells)}  "
          f"LHS∩mu=1={len(same4)}")
    print(f"shared held-out cells: {len(held)} (excluded both training sets)\n")

    # ---- the reference two-parameter model (already trained) --------------
    two, _ = load_two(G.RESULTS / "train" / "lhs40" / "mf_two_param.pt")

    results = {}

    def score_model(predict_fn, cells, label):
        pairs = {}
        for cell in cells:
            payload = load_cell(cell)
            pairs[cell.key] = (payload["clouds"]["reference"],
                               predict_fn(payload["clouds"]["source"], cell))
        rows = score_many(pairs, bins=G.BINS, workers=4, log=lambda t: None)
        return {r["key"]: r["w2"] for r in rows}, rows

    def predict_two(source, cell):
        return sample_two_param(two, torch.from_numpy(source), cell.sigma, cell.mu).numpy().astype(np.float32)

    def predict_one(model):
        def inner(source, cell):
            return sample_one_param(model, torch.from_numpy(source), cell.sigma).numpy().astype(np.float32)
        return inner

    # held-out scores for the two-parameter model
    held_two, _ = score_model(predict_two, held, "two")
    print(f"two-parameter (LHS-40 training): held-out mean W2 = {np.mean(list(held_two.values())):.5f}")

    # training scores for the two-parameter model (does it fit its own data?)
    train_two, _ = score_model(predict_two, lhs_cells, "two-train")
    print(f"two-parameter training-fit mean W2        = {np.mean(list(train_two.values())):.5f}\n")

    configs = [
        ("same-cells", lhs_cells, "sigma-only on the same 40 LHS cells"),
        ("slice-20", slice_cells, "sigma-only on the mu=1 row (20 cells)"),
        ("same-4", same4, "sigma-only on the 4 LHS cells with mu=1"),
    ]

    detail_rows = []
    for name, cells, description in configs:
        print(f"--- {name}: {description} ---")
        work = G.ensure("fair_control", name)
        steps = UPDATES_PER_CELL * len(cells)
        summary = train_one_param(cells, config, work, steps=steps, batch_size=4096,
                                  allow_mixed_mu=(name == "same-cells"))
        model, _ = load_one(work / "mf_one_param.pt")

        held_one, _ = score_model(predict_one(model), held, name)
        train_one_scores, _ = score_model(predict_one(model), cells, f"{name}-train")

        m_held = float(np.mean(list(held_one.values())))
        m_train = float(np.mean(list(train_one_scores.values())))
        gains = [held_one[k] - held_two[k] for k in held_one]
        entry = {
            "name": name,
            "description": description,
            "n_cells": len(cells),
            "steps": steps,
            "updates_per_cell": UPDATES_PER_CELL,
            "training_fit_w2": m_train,
            "heldout_mean_w2": m_held,
            "two_param_heldout_w2": float(np.mean(list(held_two.values()))),
            "improvement": float(np.mean(gains)),
            "relative_reduction": float(1 - np.mean(list(held_two.values())) / m_held),
            "two_param_wins": int(sum(1 for g in gains if g > 0)),
            "n_heldout": len(gains),
        }
        results[name] = entry
        print(f"    training fit  W2 = {m_train:.5f}  ({m_train / G.FLOOR_16:.2f}x floor)")
        print(f"    held-out      W2 = {m_held:.5f}  ({m_held / G.FLOOR_16:.2f}x floor)")
        print(f"    two-param {entry['two_param_heldout_w2']:.5f} -> "
              f"reduction {entry['relative_reduction']:+.1%}, "
              f"wins {entry['two_param_wins']}/{len(gains)}\n")

        for key in held_one:
            detail_rows.append({"control": name, "key": key,
                                "w2_one_param": held_one[key],
                                "w2_two_param": held_two[key],
                                "improvement": held_one[key] - held_two[key]})

    write_csv(out / "detail.csv", detail_rows)
    write_json(out / "summary.json", {
        "updates_per_cell": UPDATES_PER_CELL,
        "n_lhs_cells": len(lhs_cells),
        "n_heldout": len(held),
        "two_param_training_fit": float(np.mean(list(train_two.values()))),
        "two_param_heldout": float(np.mean(list(held_two.values()))),
        "controls": results,
    })

    print("=" * 74)
    print(f"{'control':<12} {'cells':>6} {'train fit':>10} {'held-out':>10} "
          f"{'vs two-param':>13}")
    print("-" * 74)
    print(f"{'two-param':<12} {len(lhs_cells):6d} "
          f"{np.mean(list(train_two.values())):10.5f} "
          f"{np.mean(list(held_two.values())):10.5f} {'—':>13}")
    for name, entry in results.items():
        print(f"{name:<12} {entry['n_cells']:6d} {entry['training_fit_w2']:10.5f} "
              f"{entry['heldout_mean_w2']:10.5f} "
              f"{entry['relative_reduction']:+12.1%}")
    print("=" * 74)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
