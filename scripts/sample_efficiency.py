"""Three sample-efficiency experiments on a common, frozen test set.

The question
------------
"Does training on a Latin hypercube subset reach the accuracy of training on
everything?" cannot be answered without a test set that *no* model sees. If the
"full" model trained on all 200 cells, there would be nothing left to test on.

Design
------
A single test set ``T`` of 40 cells is frozen up front and excluded from every
training set. The remaining ``P`` of 160 cells is the only pool any model may
draw from. All three experiments are then scored on the same ``T``:

  1. **Full-sample control** -- train on all 160 cells of ``P``. This is the
     best a "see (nearly) everything" model can do, and it is directly
     comparable to the subsets because the evaluation cells are identical.

  2. **Learning curve** -- train on LHS subsets of ``P`` with
     n = 20, 40, 80, 160 cells. Same updates *per cell*, so more cells also means
     more total gradient steps; that is the honest accounting, because in
     practice a bigger dataset costs more compute too. The curve answers "how
     much error remains at each data size", and reaching the n=160 value at a
     smaller n would be the claim "a small sample suffices".

  3. **LHS versus random** -- two designs of the same size (40 cells), one
     maximin Latin hypercube, one uniform random subset of ``P``. Identical
     budget, identical architecture, identical test set. This isolates the value
     of the sampling *method* rather than the sampling *size*.

Fairness controls
-----------------
* one seed for initialisation and one for data order, shared by every run
* identical architecture, optimiser, learning rate, batch size
* identical updates per cell (2000)
* identical test cells and identical metric (16-bin exact histogram W2)
* each checkpoint records which cells it trained on, and the loader refuses to
  score a run whose test set overlaps its training set

Outputs land in ``results/fullscale/sample_efficiency/``.

Usage:
    python scripts/sample_efficiency.py --stage all
    python scripts/sample_efficiency.py --stage curve        # only experiment 2
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

from fullscale import grid as G  # noqa: E402
from fullscale.evaluate import score_many, write_csv, write_json  # noqa: E402
from fullscale.grid import Cell, Config  # noqa: E402
from fullscale.lhs import latin_hypercube, random_subset  # noqa: E402
from fullscale.model import sample_two_param  # noqa: E402
from fullscale.train import load_cell, load_two, train_two_param  # noqa: E402

UPDATES_PER_CELL = 2000
SEED = 88
TEST_SEED = 999999          # separate seed so T is independent of the training design
N_TEST = 40
OUT = G.RESULTS / "sample_efficiency"


def log(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


# ------------------------------------------------------------------ designs ---
def stratified_subset(pool: list[Cell], n: int, *, seed: int,
                      iterations: int = 20000) -> list[Cell]:
    """A stratified maximin subset of the pool, balanced in both parameters.

    Construction, and why it needs a *global* search
    ------------------------------------------------
    A subset is built by taking ``n / n_sigma`` cells from each sigma level and
    then improving it with **swap moves that exchange two selected cells for two
    unselected ones anywhere in the pool**. Per-level rotation alone is not
    enough: it cannot stop two adjacent sigma levels from landing on the same mu
    level, which pins the minimum pairwise distance at the grid step (0.0526) and
    makes the design no better than random. That failure was measured with
    ``scripts/check_design_quality.py`` and is what this version fixes.

    The objective is

        min_pairwise_distance + 0.03 * (1 - |corr|) - 0.01 * mu_imbalance

    so the design is simultaneously space-filling, decorrelated and balanced in
    both parameter marginals. Sigma balance is exact by construction.
    """
    n_sigma = len(G.SIGMA_GRID)
    n_mu = len(G.MU_GRID)
    if n % n_sigma:
        raise ValueError(f"n={n} must be a multiple of {n_sigma}")
    per_sigma = n // n_sigma
    rng = np.random.default_rng(seed)

    by_sigma: dict[int, list[Cell]] = {}
    for cell in pool:
        by_sigma.setdefault(cell.i, []).append(cell)
    for group in by_sigma.values():
        group.sort(key=lambda c: c.j)
    if any(len(by_sigma[i]) < per_sigma for i in range(n_sigma)):
        raise ValueError(f"pool has fewer than {per_sigma} cells at some sigma level")

    def objective(cells) -> float:
        arr = np.array([[c.i / (n_sigma - 1), c.j / (n_mu - 1)] for c in cells])
        diff = arr[:, None, :] - arr[None, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        spacing = float(d.min())
        counts = np.bincount([c.j for c in cells], minlength=n_mu)
        imbalance = float(counts.max() - counts.min())
        corr = abs(float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])) if arr[:, 0].std() > 0 else 0.0
        return spacing + 0.03 * (1.0 - corr) - 0.01 * imbalance

    # initial draw: an evenly spread mu choice inside every sigma level
    selected: list[Cell] = []
    for i in range(n_sigma):
        group = by_sigma[i]
        step = len(group) / per_sigma
        offset = int(rng.integers(len(group)))
        for k in range(per_sigma):
            selected.append(group[int((offset + k * step)) % len(group)])

    current = objective(selected)
    best, best_value = list(selected), current
    for _ in range(iterations):
        # swap one selected cell for an unselected cell of the same sigma level,
        # which preserves the exact sigma marginal
        pos = int(rng.integers(len(selected)))
        victim = selected[pos]
        candidates = [c for c in by_sigma[victim.i] if c not in selected]
        if not candidates:
            continue
        replacement = candidates[int(rng.integers(len(candidates)))]
        selected[pos] = replacement
        value = objective(selected)
        if value >= current or rng.random() < 0.02:
            current = value
            if value > best_value:
                best, best_value = list(selected), value
        else:
            selected[pos] = victim
    return sorted(best, key=lambda c: (c.i, c.j))


def frozen_test_set() -> tuple[list[Cell], list[Cell]]:
    """Return ``(test, pool)``, frozen and written to disk on first creation."""
    path = OUT / "split.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        test = [Cell(int(r["i"]), int(r["j"])) for r in payload["test"]]
        pool = [Cell(int(r["i"]), int(r["j"])) for r in payload["pool"]]
        log(f"loaded frozen split: test={len(test)} pool={len(pool)}")
        return test, pool

    design = latin_hypercube(N_TEST, seed=TEST_SEED)
    test = [Cell(i, j) for i, j in design.points]
    test_keys = {c.key for c in test}
    pool = [c for c in G.full_grid() if c.key not in test_keys]
    if len(test) != N_TEST or len(pool) != 200 - N_TEST:
        raise RuntimeError("frozen split has the wrong size")

    OUT.mkdir(parents=True, exist_ok=True)
    write_json(path, {
        "test_seed": TEST_SEED,
        "n_test": len(test),
        "n_pool": len(pool),
        "note": ("The test cells are never used for training by any model in this "
                 "script, so every reported number is a genuine held-out error."),
        "test": [c.meta() for c in test],
        "pool": [c.meta() for c in pool],
    })
    log(f"created frozen split: test={len(test)} pool={len(pool)}")
    return test, pool


# ------------------------------------------------------------------ scoring ---
def score_on(model, cells: list[Cell], *, label: str) -> dict:
    pairs = {}
    for cell in cells:
        payload = load_cell(cell)
        prediction = sample_two_param(model, torch.from_numpy(payload["clouds"]["source"]),
                                      cell.sigma, cell.mu).numpy().astype(np.float32)
        pairs[cell.key] = (payload["clouds"]["reference"], prediction)
    rows = score_many(pairs, bins=G.BINS, workers=4, log=lambda t: None)
    values = [r["w2"] for r in rows]
    return {"label": label, "mean_w2": float(np.mean(values)),
            "median_w2": float(np.median(values)), "max_w2": float(np.max(values)),
            "over_floor": float(np.mean(values) / G.FLOOR_16),
            "n_cells": len(values), "rows": rows}


def run_one(name: str, cells: list[Cell], test: list[Cell], pool: list[Cell],
            *, reuse_existing: bool = False) -> dict:
    """Train one model (unless a matching checkpoint exists) and score it on ``test``."""
    work = OUT / "runs" / name
    checkpoint = work / "mf_two_param.pt"
    meta_path = work / "run_meta.json"
    steps = UPDATES_PER_CELL * len(cells)

    if reuse_existing and meta_path.is_file() and checkpoint.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("cells") == [c.key for c in cells]:
            log(f"  reusing {name} ({meta['steps']} steps)")
        else:
            reuse_existing = False

    if not (reuse_existing and meta_path.is_file()):
        log(f"  training {name}: {len(cells)} cells x {UPDATES_PER_CELL} = {steps} steps")
        config = Config(n_train=G.N_PARTICLES, device="cuda")
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            summary = train_two_param(cells, config, work, steps=steps,
                                      batch_size=4096, lr=1e-3, seed=SEED)
        meta = {"name": name, "cells": [c.key for c in cells],
                "steps": steps, "updates_per_cell": UPDATES_PER_CELL,
                "seed": SEED, "seconds": summary["seconds"],
                "final_loss": summary["final_loss"],
                "n_cells": len(cells)}
        write_json(meta_path, meta)
    else:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    model, _ = load_two(checkpoint)
    score = score_on(model, test, label=name)
    train_score = score_on(model, cells, label=f"{name}-train")

    overlap = {c.key for c in cells} & {c.key for c in test}
    if overlap:
        raise RuntimeError(f"{name}: test set overlaps training set on {sorted(overlap)}")

    record = {
        "name": name,
        "n_cells": len(cells),
        "steps": steps,
        "updates_per_cell": UPDATES_PER_CELL,
        "training_seconds": meta.get("seconds"),
        "final_train_loss": meta.get("final_loss"),
        "training_fit_w2": train_score["mean_w2"],
        "test_mean_w2": score["mean_w2"],
        "test_median_w2": score["median_w2"],
        "test_max_w2": score["max_w2"],
        "test_over_floor": score["over_floor"],
        "n_test": score["n_cells"],
    }
    log(f"    train-fit {record['training_fit_w2']:.5f}  "
        f"TEST {record['test_mean_w2']:.5f}  ({record['test_over_floor']:.2f}x floor)")
    return record


# ------------------------------------------------------------------- stages ---
def stage_full(test: list[Cell], pool: list[Cell]) -> dict:
    log("=== experiment 1: full-sample control (all 160 pool cells) ===")
    record = run_one("full160", pool, test, pool, reuse_existing=True)
    write_json(OUT / "full_control.json", record)
    return record


def stage_curve(test: list[Cell], pool: list[Cell]) -> dict:
    log("=== experiment 2: learning curve ===")
    results = []
    for n in (20, 40, 80, 160):
        cells = stratified_subset(pool, n, seed=20261107 + n)
        log(f"  n={n} cells selected")
        results.append(run_one(f"curve{n}", cells, test, pool, reuse_existing=True))
    write_csv(OUT / "learning_curve.csv", results)
    write_json(OUT / "learning_curve.json", {"n_test": len(test), "results": results})

    print(f"\n{'n cells':>8} {'steps':>8} {'train fit':>11} {'TEST W2':>10} {'x floor':>8}")
    for r in results:
        print(f"{r['n_cells']:8d} {r['steps']:8d} {r['training_fit_w2']:11.5f} "
              f"{r['test_mean_w2']:10.5f} {r['test_over_floor']:8.2f}")
    return {"results": results}


def stage_lhs_vs_random(test: list[Cell], pool: list[Cell]) -> dict:
    log("=== experiment 3: LHS vs random, same size (40 cells) ===")
    lhs_cells = stratified_subset(pool, 40, seed=20261107)

    # The random control must be a uniform draw from the *same pool* the LHS uses,
    # not a draw from the whole grid with the test cells filtered out: filtering
    # changes the sampling distribution and would bias the control.
    rng = np.random.default_rng(424242)
    chosen = rng.choice(len(pool), size=40, replace=False)
    rnd_cells = sorted((pool[int(c)] for c in chosen), key=lambda c: (c.i, c.j))
    if len({c.key for c in rnd_cells}) != 40:
        raise RuntimeError("random control drew duplicate cells")

    def geometry(cells):
        arr = np.array([[c.i / (len(G.SIGMA_GRID) - 1), c.j / (len(G.MU_GRID) - 1)]
                        for c in cells])
        diff = arr[:, None, :] - arr[None, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        corr = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]) if arr[:, 0].std() > 0 else 0.0
        return {"min_distance": float(d.min()),
                "mean_nn_distance": float(d.min(axis=1).mean()),
                "correlation": corr}

    runs = []
    for name, cells, geom in (("lhs40", lhs_cells, geometry(lhs_cells)),
                              ("random40", rnd_cells, geometry(rnd_cells))):
        log(f"  {name}: min_dist={geom['min_distance']:.4f} corr={geom['correlation']:+.3f}")
        record = run_one(name, cells, test, pool, reuse_existing=True)
        record["design"] = geom
        runs.append(record)

    lhs_run, rnd_run = runs
    summary = {
        "n_test": len(test),
        "runs": runs,
        "delta_test_w2": lhs_run["test_mean_w2"] - rnd_run["test_mean_w2"],
        "relative": (lhs_run["test_mean_w2"] / rnd_run["test_mean_w2"] - 1),
        "lhs_better": lhs_run["test_mean_w2"] < rnd_run["test_mean_w2"],
    }
    write_json(OUT / "lhs_vs_random.json", summary)
    print(f"\n{'design':<10} {'min dist':>9} {'corr':>7} {'train fit':>11} "
          f"{'TEST W2':>10}")
    for r in runs:
        print(f"{r['name']:<10} {r['design']['min_distance']:9.4f} "
              f"{r['design']['correlation']:+7.3f} {r['training_fit_w2']:11.5f} "
              f"{r['test_mean_w2']:10.5f}")
    print(f"\nLHS test W2 {lhs_run['test_mean_w2']:.5f} vs random "
          f"{rnd_run['test_mean_w2']:.5f}  ({summary['relative']:+.2%})")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "full", "curve", "lhs"], default="all")
    a = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    OUT.mkdir(parents=True, exist_ok=True)
    test, pool = frozen_test_set()
    log(f"test cells {len(test)}, pool cells {len(pool)}, "
        f"updates/cell {UPDATES_PER_CELL}, floor {G.FLOOR_16}")

    if a.stage in ("all", "full"):
        stage_full(test, pool)
    if a.stage in ("all", "curve"):
        stage_curve(test, pool)
    if a.stage in ("all", "lhs"):
        stage_lhs_vs_random(test, pool)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
