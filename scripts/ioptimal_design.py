"""Build an I-optimal design with maximin as a hard constraint, and compare it.

Three designs at 40 cells, all drawn from the same frozen training pool:

  ``lhs40``      maximin Latin hypercube (the current method)
  ``random40``   uniform random subset
  ``iopt40``     minimise the integrated W2-to-nearest-design objective, subject
                 to the parameter-space minimum spacing being at least that of
                 the maximin design

The third design keeps the user's requirement (a guaranteed minimum spacing) while
changing what is optimised: instead of the geometric criterion alone, it minimises

    J(D) = sum_{x in grid} min_{d in D} W2(cloud_x, cloud_d)^2

which is the squared distance in the paper's own error metric from every grid
point to the nearest training condition. Because J is computed in W2 units, it is
not a proxy for the paper's metric -- it is the integral of that metric.

Scoring
-------
Each design is trained with the identical protocol and scored on the frozen test
set, so the comparison is on the same footing as every other number in this
repository. The script reports both the design objectives and the realised test
errors.

Usage:
    python scripts/ioptimal_design.py --size 40
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fullscale import grid as G  # noqa: E402
from fullscale.grid import Cell  # noqa: E402
from fullscale.ioptimal import (  # noqa: E402
    ioptimal_design, load_or_build, maximin_distance, w2_objective,
)

from sample_efficiency import (  # noqa: E402
    OUT, frozen_test_set, run_one, stratified_subset, write_csv, write_json, log,
)


def describe(geometry, cells: list[Cell], label: str) -> dict:
    all_index = {c.key: k for k, c in enumerate(geometry.cells)}
    design_idx = np.array([all_index[c.key] for c in cells])
    domain_idx = np.arange(len(geometry.cells))
    _, mean_nearest = w2_objective(geometry, design_idx, domain_idx)
    return {
        "label": label,
        "n": len(cells),
        "min_spacing": maximin_distance(cells),
        "mean_nearest_w2": mean_nearest,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=40)
    ap.add_argument("--iterations", type=int, default=40000)
    a = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    test, pool = frozen_test_set()
    pool_keys = {c.key for c in pool}
    log(f"test {len(test)} cells, pool {len(pool)} cells, design size {a.size}")

    # The geometry is built over the full 200-cell grid so that J integrates over
    # the whole parameter domain, including the test cells.
    cells_all = G.full_grid()
    matrix_path = OUT / "bures_distance.npy"
    geometry = load_or_build(cells_all, matrix_path, log=log)

    # ---- the three designs -------------------------------------------------
    lhs_cells = stratified_subset(pool, a.size, seed=20261107)
    rng = np.random.default_rng(424242)
    chosen = rng.choice(len(pool), size=a.size, replace=False)
    random_cells = sorted((pool[int(c)] for c in chosen), key=lambda c: (c.i, c.j))

    target_spacing = maximin_distance(lhs_cells)
    log(f"maximin reference spacing: {target_spacing:.4f} (this is the hard constraint)")

    # pool restricted to itself for the design search
    pool_geometry = load_or_build(
        cells_all, matrix_path, log=lambda t: None)
    result = ioptimal_design(
        pool_geometry, a.size, min_spacing=target_spacing,
        candidate_keys=[c.key for c in pool],
        domain_keys=[c.key for c in cells_all],
        seed=20261107, iterations=a.iterations, log=log)
    all_index = {c.key: k for k, c in enumerate(cells_all)}
    iopt_cells = sorted((cells_all[all_index[k]] for k in result["cells"]),
                        key=lambda c: (c.i, c.j))
    if not {c.key for c in iopt_cells}.issubset(pool_keys):
        raise RuntimeError("I-optimal design drew a cell outside the training pool")

    rows = []
    for label, cells in (("lhs40", lhs_cells), ("random40", random_cells),
                         ("iopt40", iopt_cells)):
        if len({c.key for c in cells}) != a.size:
            raise RuntimeError(f"{label} has the wrong size")
        info = describe(geometry, cells, label)
        log(f"{label}: min_spacing {info['min_spacing']:.4f}  "
            f"mean nearest W2 {info['mean_nearest_w2']:.5f}  "
            f"objective J {info['mean_nearest_w2'] ** 2 * len(cells_all):.4f}")
        run = run_one(label, cells, test, pool, reuse_existing=True)
        info.update({"test_mean_w2": run["test_mean_w2"],
                     "training_fit_w2": run["training_fit_w2"],
                     "test_over_floor": run["test_over_floor"],
                     "cells": [c.key for c in cells]})
        rows.append(info)

    write_csv(OUT / "ioptimal_comparison.csv", rows)
    print(f"\n{'design':<10} {'min space':>10} {'mean near W2':>13} {'train fit':>10} "
          f"{'TEST W2':>9} {'x floor':>8}")
    for r in rows:
        print(f"{r['label']:<10} {r['min_spacing']:10.4f} {r['mean_nearest_w2']:13.5f} "
              f"{r['training_fit_w2']:10.5f} {r['test_mean_w2']:9.5f} "
              f"{r['test_over_floor']:8.2f}")

    by = {r["label"]: r for r in rows}
    write_json(OUT / "ioptimal_comparison.json", {
        "size": a.size,
        "spacing_constraint": target_spacing,
        "iterations": a.iterations,
        "designs": rows,
        "iopt_vs_lhs": by["iopt40"]["test_mean_w2"] - by["lhs40"]["test_mean_w2"],
        "iopt_vs_random": by["iopt40"]["test_mean_w2"] - by["random40"]["test_mean_w2"],
    })
    print(f"\nI-optimal vs LHS   : {by['iopt40']['test_mean_w2'] - by['lhs40']['test_mean_w2']:+.5f}")
    print(f"I-optimal vs random: {by['iopt40']['test_mean_w2'] - by['random40']['test_mean_w2']:+.5f}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
