"""Rebuild the paper's key comparison at the optimal training budget.

Every earlier comparison in this repository was trained at 2,000 updates per
cell, which the budget sweep showed to be under-trained (2.60x the noise floor
versus 1.87x at 32,000). The *directions* of those results should survive, but the
magnitudes cannot be quoted in a paper without being re-measured at a converged
budget.

This script re-runs the central comparison -- two parameters versus one -- with
everything else held at the values established as optimal:

    updates per cell   32,000
    architecture       64 wide, 3 SELU layers, plain log10 conditioning
    particles          100,000
    test set           the frozen 40 cells (never trained on)
    held-out cells     144, the same set used in the earlier ablation

The single-parameter control is trained on the mu = 1 row with the same updates
per cell, so the per-cell budget is matched exactly.

What is reported
----------------
* both models' mean test W2, their ratio, and both normalised by the noise floor
* the gain binned by distance from mu = 1, which is the signature that the model
  is using the second parameter rather than memorising a slice
* a paired per-cell win count

Usage:
    python scripts/ablation_optimal_budget.py --updates-per-cell 32000
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
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
from fullscale.model import sample_one_param, sample_two_param  # noqa: E402
from fullscale.train import load_cell, load_one, load_two, train_one_param, train_two_param  # noqa: E402

from sample_efficiency import (  # noqa: E402
    frozen_test_set, stratified_subset, write_csv, write_json, log,
)

OUT = G.RESULTS / "optimal_budget_ablation"
DEFAULT_UPDATES = 32000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates-per-cell", type=int, default=DEFAULT_UPDATES)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    test, pool = frozen_test_set()
    design = stratified_subset(pool, 40, seed=20261107)
    design_keys = {c.key for c in design}
    slice_cells = G.baseline_slice()
    slice_keys = {c.key for c in slice_cells}
    held = [c for c in G.full_grid()
            if c.key not in design_keys and c.key not in slice_keys]
    log(f"train two-param: {len(design)} cells | "
        f"train one-param: {len(slice_cells)} cells (mu=1 row) | "
        f"evaluate on {len(held)} cells neither saw")
    log(f"updates per cell: {a.updates_per_cell}  "
        f"-> {a.updates_per_cell * len(design):,} / "
        f"{a.updates_per_cell * len(slice_cells):,} steps")

    config = Config(n_train=G.N_PARTICLES, device="cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    payloads = {c.key: load_cell(c) for c in held}

    def score(model, sampler):
        pairs = {}
        for cell in held:
            payload = payloads[cell.key]
            pairs[cell.key] = (payload["clouds"]["reference"],
                               sampler(model, payload["clouds"]["source"], cell))
        rows = score_many(pairs, bins=G.BINS, workers=4, log=lambda t: None)
        return {r["key"]: r["w2"] for r in rows}

    # ---- two parameters ---------------------------------------------------
    two_dir = OUT / f"two_{a.updates_per_cell}"
    two_ckpt = two_dir / "mf_two_param.pt"
    if two_ckpt.is_file():
        log("two-parameter checkpoint exists, reusing")
    else:
        log("training two-parameter model ...")
        started = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            train_two_param(design, config, two_dir,
                            steps=a.updates_per_cell * len(design),
                            batch_size=a.batch_size, lr=a.lr, seed=88,
                            conditioning="log10")
        log(f"  finished in {time.perf_counter() - started:.0f}s")
    two_model, _ = load_two(two_ckpt)
    two_scores = score(two_model, lambda m, src, c:
                       sample_two_param(m, torch.from_numpy(src), c.sigma, c.mu).numpy().astype(np.float32))
    del two_model
    torch.cuda.empty_cache()

    # ---- one parameter ----------------------------------------------------
    one_dir = OUT / f"one_{a.updates_per_cell}"
    one_ckpt = one_dir / "mf_one_param.pt"
    if one_ckpt.is_file():
        log("one-parameter checkpoint exists, reusing")
    else:
        log("training one-parameter control ...")
        started = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            train_one_param(slice_cells, config, one_dir,
                            steps=a.updates_per_cell * len(slice_cells),
                            batch_size=a.batch_size, lr=a.lr, seed=88)
        log(f"  finished in {time.perf_counter() - started:.0f}s")
    one_model, _ = load_one(one_ckpt)
    one_scores = score(one_model, lambda m, src, c:
                       sample_one_param(m, torch.from_numpy(src), c.sigma).numpy().astype(np.float32))
    del one_model
    torch.cuda.empty_cache()

    # ---- compare ----------------------------------------------------------
    detail = []
    for cell in held:
        w_one, w_two = one_scores[cell.key], two_scores[cell.key]
        detail.append({**cell.meta(),
                       "w2_one_param": w_one, "w2_two_param": w_two,
                       "improvement": w_one - w_two, "ratio": w_two / w_one,
                       "mu_off_baseline": abs(math.log10(cell.mu))})
    detail.sort(key=lambda r: (r["mu_off_baseline"], r["sigma"]))
    write_csv(OUT / "detail.csv", detail)

    m_one = float(np.mean([r["w2_one_param"] for r in detail]))
    m_two = float(np.mean([r["w2_two_param"] for r in detail]))
    gains = np.array([r["improvement"] for r in detail])
    off = np.array([r["mu_off_baseline"] for r in detail])

    bands = {}
    for r in detail:
        bands.setdefault(round(r["mu_off_baseline"], 2), []).append(r)
    band_summary = {}
    print(f"\n{'|log10 mu|':>11} {'n':>4} {'one-param':>10} {'two-param':>10} {'gain':>9}")
    for band in sorted(bands):
        group = bands[band]
        o = float(np.mean([r["w2_one_param"] for r in group]))
        t = float(np.mean([r["w2_two_param"] for r in group]))
        band_summary[str(band)] = {"n": len(group), "one": o, "two": t, "gain": o - t}
        print(f"{band:11.2f} {len(group):4d} {o:10.5f} {t:10.5f} {o - t:+9.5f}")

    summary = {
        "updates_per_cell": a.updates_per_cell,
        "n_held_out": len(detail),
        "one_param_mean_w2": m_one,
        "two_param_mean_w2": m_two,
        "relative_reduction": float(1 - m_two / m_one),
        "one_over_floor": m_one / G.FLOOR_16,
        "two_over_floor": m_two / G.FLOOR_16,
        "two_param_wins": int((gains > 0).sum()),
        "one_param_wins": int((gains < 0).sum()),
        "median_improvement": float(np.median(gains)),
        "corr_gain_vs_mu_distance": float(np.corrcoef(off, gains)[0, 1])
        if off.std() > 0 and gains.std() > 0 else None,
        "by_mu_band": band_summary,
        "noise_floor": G.FLOOR_16,
    }
    write_json(OUT / "summary.json", summary)

    print(f"\nnoise floor {G.FLOOR_16:.5f}")
    print(f"one-parameter  {m_one:.5f}  ({m_one / G.FLOOR_16:.2f}x floor)")
    print(f"two-parameter  {m_two:.5f}  ({m_two / G.FLOOR_16:.2f}x floor)")
    print(f"reduction      {summary['relative_reduction']:+.1%}   "
          f"two-param wins {summary['two_param_wins']}/{len(detail)}")
    print(f"corr(gain, |log10 mu|) = {summary['corr_gain_vs_mu_distance']:.3f}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
