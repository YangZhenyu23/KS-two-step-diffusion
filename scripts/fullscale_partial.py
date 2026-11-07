"""Evaluate the trained models on whatever cells already exist.

The full 200-cell grid takes hours to simulate, but the 40 training cells plus
the 20-cell baseline row are available early. This script scores both models on
those cells so the run can be sanity-checked before the complete grid is ready.

It writes into ``results/fullscale/partial/`` and is deliberately separate from
the final evaluation so partial numbers can never be mistaken for final ones.

Usage:
    python scripts/fullscale_partial.py
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
from fullscale.grid import MU_INDEX  # noqa: E402
from fullscale.lhs import latin_hypercube  # noqa: E402
from fullscale.model import sample_one_param, sample_two_param  # noqa: E402
from fullscale.train import load_cell, load_one, load_two  # noqa: E402


def main() -> int:
    manifest = json.loads((G.DATA / "manifest.json").read_text(encoding="utf-8"))
    available = {(r["i"], r["j"]) for r in manifest.get("cells", [])}
    print(f"{len(available)} cells available")

    two, _ = load_two(G.RESULTS / "train" / "lhs40" / "mf_two_param.pt")
    one_path = G.RESULTS / "train" / "baseline" / "mf_one_param.pt"
    one = load_one(one_path)[0] if one_path.is_file() else None
    if one is None:
        print("single-parameter control not trained yet; scoring the two-parameter model only")

    design = latin_hypercube(40)
    train_keys = {G.Cell(i, j).key for i, j in design.points}

    # ---- two-parameter model on all available cells -----------------------
    pairs_two, meta = {}, []
    for cell in G.full_grid():
        if (cell.i, cell.j) not in available:
            continue
        payload = load_cell(cell)
        prediction = sample_two_param(two, torch.from_numpy(payload["clouds"]["source"]),
                                     cell.sigma, cell.mu).numpy().astype(np.float32)
        pairs_two[cell.key] = (payload["clouds"]["reference"], prediction)
        meta.append({**cell.meta(), "split": "train" if cell.key in train_keys else "heldout"})

    rows_two = score_many(pairs_two, bins=G.BINS, workers=4, log=lambda t: None)
    by_two = {r["key"]: r for r in rows_two}
    for entry in meta:
        entry.update({k: v for k, v in by_two[entry["key"]].items() if k != "key"})

    train_rows = [r for r in meta if r["split"] == "train"]
    held_rows = [r for r in meta if r["split"] == "heldout"]
    out = G.ensure("partial")
    write_csv(out / "partial_grid_metrics.csv", meta)

    print(f"\ntwo-parameter model on {len(meta)} available cells")
    print(f"  train cells   n={len(train_rows):3d}  mean W2 = "
          f"{np.mean([r['w2'] for r in train_rows]):.5f}")
    if held_rows:
        print(f"  held out      n={len(held_rows):3d}  mean W2 = "
              f"{np.mean([r['w2'] for r in held_rows]):.5f}")
    print(f"  floor = {G.FLOOR_16:.5f}")

    # ---- head to head on the cells both models can score ------------------
    summary = {"available_cells": len(meta), "floor": G.FLOOR_16,
               "two_param_train_mean": float(np.mean([r["w2"] for r in train_rows])),
               "two_param_heldout_mean": float(np.mean([r["w2"] for r in held_rows]))
               if held_rows else None}
    if one is not None:
        held_cells = [c for c in G.full_grid()
                      if (c.i, c.j) in available and c.key not in train_keys]
        print(f"\nhead-to-head on {len(held_cells)} held-out cells")
        pairs_one = {}
        for cell in held_cells:
            payload = load_cell(cell)
            pairs_one[cell.key] = (payload["clouds"]["reference"],
                                   sample_one_param(one, torch.from_numpy(payload["clouds"]["source"]),
                                                    cell.sigma).numpy().astype(np.float32))
        rows_one = score_many(pairs_one, bins=G.BINS, workers=4, log=lambda t: None)
        by_one = {r["key"]: r["w2"] for r in rows_one}
        detail = []
        for cell in held_cells:
            w1, w2 = by_one[cell.key], by_two[cell.key]["w2"]
            detail.append({**cell.meta(), "w2_one_param": w1, "w2_two_param": w2,
                           "improvement": w1 - w2, "ratio": w2 / w1,
                           "mu_off_baseline": abs(np.log10(cell.mu))})
        write_csv(out / "partial_ablation.csv", detail)
        m1 = float(np.mean([r["w2_one_param"] for r in detail]))
        m2 = float(np.mean([r["w2_two_param"] for r in detail]))
        gains = np.array([r["improvement"] for r in detail])
        print(f"  one-parameter mean W2 = {m1:.5f}  ({m1 / G.FLOOR_16:.2f}x floor)")
        print(f"  two-parameter mean W2 = {m2:.5f}  ({m2 / G.FLOOR_16:.2f}x floor)")
        print(f"  reduction {1 - m2 / m1:+.1%}   two-param wins {int((gains > 0).sum())}/{len(detail)}")
        summary["partial_ablation"] = {"n_cells": len(detail), "one_param_mean": m1,
                                       "two_param_mean": m2,
                                       "relative_reduction": float(1 - m2 / m1),
                                       "two_param_wins": int((gains > 0).sum())}

    write_json(out / "partial_summary.json", summary)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
