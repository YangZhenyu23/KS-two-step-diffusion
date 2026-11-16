"""E1: can the SINGLE-parameter model be iterated?

Why this is the interesting counterpart to the two-parameter rollout study
-----------------------------------------------------------------------
The two-parameter map diverges under repeated application (results/fullscale/rollout). The
diagnosis given in the report is that the map is a fixed-horizon transport rather than a flow,
and that one application moves the cloud far outside the training distribution. If that
diagnosis is right, the single-parameter model should show the *same* failure, because it has
the same architecture and the same endpoint-only supervision. If instead it tolerates
iteration, then non-iterability is a property of the two-parameter family rather than of the
method, which would sharpen \S4.8 considerably.

Design
------
The single-parameter checkpoint is the upstream architecture: 6 inputs (z, t, r, log sigma),
no mu. It was trained at mu = 1, so every comparison here is at mu = 1, i.e. against the
mu = 1 row of the grid. This is the only regime the model covers.

Same protocol as the two-parameter study:
  * apply the one-step map k times, each application spanning [0, T]
  * compare against the physical solution at k*T, same initial cloud
  * report W2, W2/floor, step amplification, and rms radius

Reference cloud generation is the expensive part (O(N^2) per step), so it is done once per
(cell, k) and cached.

Usage:
    python scripts/e1_single_param_rollout.py --cells 3 --ks 1 2 4 8 --n 20000
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream" / "MF-DeepParticle" / "deepparticle"))

from fullscale import grid as G  # noqa: E402
from fullscale.train import load_one  # noqa: E402
from fullscale.model import sample_one_param  # noqa: E402
from twoparam.data import unit_ball  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from rollout_stability import exact_w2, simulate, invariants  # noqa: E402

OUT = G.RESULTS / "rollout_single"

# mu = 1 cells, spanning sigma. Held out from the 20-cell slice that trained the model
# is impossible here: the model saw ALL 20 mu=1 cells, so these are training cells.
# That is stated in the output rather than hidden: the comparison is about *stability*,
# not about held-out accuracy.
CELLS = [(1, G.MU_INDEX), (6, G.MU_INDEX), (13, G.MU_INDEX), (19, G.MU_INDEX)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=4)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--score-n", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda text: print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)

    checkpoint = G.RESULTS / "optimal_budget_ablation" / "one_32000" / "mf_one_param.pt"
    model, payload = load_one(checkpoint, a.device)
    log(f"loaded {checkpoint.relative_to(ROOT)}  (mu = {payload.get('mu', 1.0)})")

    initial = unit_ball(a.n, 88)
    rng = np.random.default_rng(20261117)
    index = rng.choice(a.n, size=min(a.score_n, a.n), replace=False)
    source = torch.from_numpy(initial.astype(np.float32))

    rows = []
    for position, (i, j) in enumerate(CELLS[: a.cells], start=1):
        cell = G.Cell(i, j)
        sigma, mu = cell.sigma, cell.mu
        log(f"=== {position}/{min(a.cells, len(CELLS))} {cell.key} "
            f"sigma={sigma:g} mu={mu:g} ===")

        generated = {}
        for k in a.ks:
            current = source
            for _ in range(k):
                current = sample_one_param(model, current, sigma)
            generated[k] = current.numpy().astype(np.float64)

        physical = {}
        for k in a.ks:
            seed = 6_600_000 + 41 * (i * 10 + j) + 113 * k
            physical[k] = simulate(initial, sigma, mu, G.FINAL_TIME * k, seed, a.device)

        previous_gen, previous_phys = None, None
        for k in a.ks:
            gen, phys = generated[k], physical[k]
            w2 = exact_w2(gen[index], phys[index])
            step = None if previous_gen is None else exact_w2(gen[index], previous_gen[index])
            phys_step = (None if previous_phys is None
                         else exact_w2(phys[index], previous_phys[index]))
            inv = invariants(gen)
            rows.append({
                "key": cell.key, "sigma": sigma, "mu": mu, "k": k,
                "w2_to_physical": w2, "step_w2": step, "physical_step_w2": phys_step,
                "amplification": (None if (step is None or not phys_step)
                                  else step / phys_step),
                "rms_radius": inv["radius_rms"],
                "centroid_x": inv["centroid"][0],
                "finite": inv["finite"],
            })
            previous_gen, previous_phys = gen, phys
            log(f"  k={k}: W2 {w2:.5f}"
                + (f"  step {step:.5f} (physics {phys_step:.5f})" if step is not None else ""))

    floors = []
    for i, j in CELLS[: a.cells]:
        cell = G.Cell(i, j)
        one = simulate(initial, cell.sigma, cell.mu, G.FINAL_TIME,
                       7_700_000 + i * 10 + j, a.device)
        two = simulate(initial, cell.sigma, cell.mu, G.FINAL_TIME,
                       8_800_000 + i * 10 + j, a.device)
        floors.append({"key": cell.key, "floor": exact_w2(one[index], two[index])})
    floor_mean = float(np.mean([f["floor"] for f in floors]))

    payload_out = {
        "setting": "single-parameter (6-input), mu = 1",
        "cells": [G.Cell(i, j).key for i, j in CELLS[: a.cells]],
        "ks": a.ks, "n": a.n, "score_n": int(len(index)),
        "floor_mean": floor_mean, "floors": floors, "rows": rows,
        "note": ("These cells are the model's own training cells: the single-parameter "
                 "checkpoint was trained on the whole mu=1 row, so no held-out mu=1 cell "
                 "exists. The measurement is about stability under iteration, not about "
                 "held-out accuracy."),
    }
    (OUT / "rollout_single.json").write_text(json.dumps(payload_out, indent=2),
                                             encoding="utf-8")

    print(f"\nfloor = {floor_mean:.5f}")
    print(f"{'cell':>8} {'k':>3} {'W2/phys':>9} {'x floor':>9} {'amp':>8} {'rms':>10}")
    for row in rows:
        amp = row["amplification"]
        print(f"{row['key']:>8} {row['k']:3d} {row['w2_to_physical']:9.5f} "
              f"{row['w2_to_physical'] / floor_mean:9.1f} "
              f"{(f'{amp:.2f}' if amp is not None else '-'):>8} {row['rms_radius']:10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
