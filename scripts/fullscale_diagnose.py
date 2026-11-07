"""Diagnose the partial ablation pattern along the mu = 1 row.

The partial results show the two-parameter model winning on the mu = 1 row for
large sigma and losing for small sigma, while also losing on every sampled
mu != 1 cell. Two explanations are plausible and they imply different fixes:

H1 **Data density.** The single-parameter control trains on all twenty cells of
   the mu = 1 row; the two-parameter model trains on the LHS-40 subset, which
   contains only four of those cells. On that row the control therefore has five
   times the supervision, which explains a loss concentrated at small sigma
   (where the physical signal is weakest).

H2 **Optimisation.** The two-parameter model has 40 cells to cover with the same
   400 updates per cell, so it may simply be undertrained.

This script separates them by measuring the *training* error of each model on
the cells it was actually trained on. If the two-parameter model fits its own
training cells well, H1 (coverage on the mu = 1 row) is the explanation. If its
training error is also high, H2 (budget) is.

Usage:
    python scripts/fullscale_diagnose.py
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
from fullscale.evaluate import score_pair  # noqa: E402
from fullscale.lhs import latin_hypercube  # noqa: E402
from fullscale.model import sample_one_param, sample_two_param  # noqa: E402
from fullscale.train import load_cell, load_one, load_two  # noqa: E402


def main() -> int:
    two, two_payload = load_two(G.RESULTS / "train" / "lhs40" / "mf_two_param.pt")
    one, one_payload = load_one(G.RESULTS / "train" / "baseline" / "mf_one_param.pt")
    print(f"two-param: trained {two_payload['step']} steps on {len(two_payload['cells'])} cells")
    print(f"one-param: trained {one_payload['step']} steps on {len(one_payload['cells'])} cells")

    two_cells = set(two_payload["cells"])
    one_cells = set(one_payload["cells"])
    design = latin_hydrogen = latin_hypercube(40)
    lhs_keys = {G.Cell(i, j).key for i, j in design.points}

    print(f"\ntwo-param training cells on the mu = 1 row: "
          f"{sorted(k for k in two_cells if k.endswith('m05'))}")
    print(f"one-param training cells (all of mu = 1): {len(one_cells)}")

    # ---- H2 check: does the two-parameter model fit its own training cells? --
    print("\nH2: two-parameter error on its OWN training cells")
    print(f"{'cell':<9} {'sigma':>7} {'mu':>9} {'W2 train':>10}")
    train_errors = []
    for key in sorted(two_cells):
        i, j = int(key[1:3]), int(key[5:7])
        cell = G.Cell(i, j)
        payload = load_cell(cell)
        prediction = sample_two_param(two, torch.from_numpy(payload["clouds"]["source"]),
                                      cell.sigma, cell.mu).numpy().astype(np.float32)
        w2 = score_pair(payload["clouds"]["reference"], prediction, G.BINS)["w2"]
        train_errors.append(w2)
        print(f"{key:<9} {cell.sigma:7.0f} {cell.mu:9.4f} {w2:10.5f}")
    print(f"  mean training W2 = {np.mean(train_errors):.5f}  "
          f"({np.mean(train_errors) / G.FLOOR_16:.2f}x floor)")

    # ---- H1 check: mu = 1 row, both models --------------------------------
    print("\nH1: the mu = 1 row, cell by cell")
    print(f"{'sigma':>7} {'one (trained)':>14} {'two (trained?)':>15} {'gain':>9}")
    rows = []
    for i in range(len(G.SIGMA_GRID)):
        cell = G.Cell(i, G.MU_INDEX)
        path = G.DATA / "clouds" / f"{cell.key}.npz"
        if not path.is_file():
            continue
        payload = load_cell(cell)
        source = torch.from_numpy(payload["clouds"]["source"])
        reference = payload["clouds"]["reference"]
        w1 = score_pair(reference, sample_one_param(one, source, cell.sigma).numpy().astype(np.float32),
                        G.BINS)["w2"]
        w2 = score_pair(reference, sample_two_param(two, source, cell.sigma, cell.mu).numpy().astype(np.float32),
                        G.BINS)["w2"]
        in_lhs = "yes" if cell.key in lhs_keys else "no"
        rows.append({"sigma": cell.sigma, "one": w1, "two": w2, "in_lhs": in_lhs})
        print(f"{cell.sigma:7.0f} {w1:14.5f} {w2:15.5f} {w1 - w2:+9.5f}   LHS={in_lhs}")

    in_lhs = [r for r in rows if r["in_lhs"] == "yes"]
    not_lhs = [r for r in rows if r["in_lhs"] == "no"]
    if in_lhs:
        print(f"\n  cells the two-param model trained on: n={len(in_lhs)}  "
              f"one={np.mean([r['one'] for r in in_lhs]):.5f}  "
              f"two={np.mean([r['two'] for r in in_lhs]):.5f}")
    if not_lhs:
        print(f"  cells it did NOT train on:            n={len(not_lhs)}  "
              f"one={np.mean([r['one'] for r in not_lhs]):.5f}  "
              f"two={np.mean([r['two'] for r in not_lhs]):.5f}")

    out = G.ensure("diagnosis")
    (out / "diagnosis.json").write_text(json.dumps({
        "two_param_train_mean_w2": float(np.mean(train_errors)),
        "two_param_steps": two_payload["step"],
        "one_param_steps": one_payload["step"],
        "mu1_row": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out / 'diagnosis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
