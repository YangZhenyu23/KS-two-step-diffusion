"""Retrain seed-88 stage 1 with warm-up + cosine on the headline cells, then the corrector.

The cells are read from the headline checkpoint. They are not recomputed with
stratified_subset, because that function changed during development and no longer
returns the 40 cells the headline model was trained on.
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
sys.path.insert(0, str(ROOT / "scripts"))

from fullscale import grid as G  # noqa: E402
from fullscale.evaluate import score_many, write_csv, write_json  # noqa: E402
from fullscale.model import TwoParamMeanflow, sample_two_param  # noqa: E402
from fullscale.train import _save, load_cell  # noqa: E402
from lr_schedule import train as train_scheduled  # noqa: E402
from dp_two_param import apply_corrector, train_corrector  # noqa: E402

HEADLINE = G.RESULTS / "optimal_budget_ablation" / "two_32000" / "mf_two_param.pt"
OUT = G.RESULTS / "harry_followup" / "warmcos_s88"


def log(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def cells_from_checkpoint(path: Path) -> list[G.Cell]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    keys = payload.get("cells")
    if not keys or len(keys) != 40:
        raise SystemExit(f"headline checkpoint does not record 40 cells: {path}")
    by_key = {cell.key: cell for cell in G.full_grid()}
    missing = [key for key in keys if key not in by_key]
    if missing:
        raise SystemExit(f"unknown cell keys: {missing[:5]}")
    return [by_key[key] for key in keys]


def held_out(train: list[G.Cell]) -> list[G.Cell]:
    train_keys = {cell.key for cell in train}
    slice_keys = {cell.key for cell in G.baseline_slice()}
    return [cell for cell in G.full_grid()
            if cell.key not in train_keys and cell.key not in slice_keys]


def main() -> int:
    torch.set_num_threads(min(8, torch.get_num_threads()))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    OUT.mkdir(parents=True, exist_ok=True)
    cells = cells_from_checkpoint(HEADLINE)
    held = held_out(cells)
    log(f"headline cells {len(cells)}, held-out {len(held)}")
    (OUT / "cells.json").write_text(json.dumps({
        "source": str(HEADLINE),
        "train": [cell.key for cell in cells],
        "held_out": [cell.key for cell in held],
    }, indent=2), encoding="utf-8")

    steps = 32_000 * len(cells)
    config = G.Config(n_train=G.N_PARTICLES, device="cuda")
    model = TwoParamMeanflow(width=64, hidden_layers=3, dimension=3, conditioning="log10")
    log(f"warmcos stage 1: {steps} steps, lr 1e-3 -> 1e-5, warmup 2%")
    started = time.perf_counter()
    model, records = train_scheduled(
        model, cells, config, steps=steps, schedule="warmcos",
        lr0=1e-3, eta_min=1e-5, batch_size=4096, warmup_frac=0.02, seed=88,
    )
    stage1_seconds = time.perf_counter() - started
    _save(OUT / "mf_two_param.pt", model, cells, config, steps, 88)
    (OUT / "stage1_log.json").write_text(json.dumps({
        "schedule": "warmcos", "seed": 88, "steps": steps,
        "seconds": stage1_seconds, "records": records,
        "cells": [cell.key for cell in cells],
    }), encoding="utf-8")
    log(f"stage 1 done in {stage1_seconds:.0f}s")

    result = train_corrector(
        model, cells, config, updates_per_cell=2000, batch=2000, alpha=1.0,
        blocks=2, residual_weight=1.0, ot_refresh=10, lr=1e-4, seed=88, log=log,
    )
    corrector = result["model"]
    torch.save({
        "model": {k: v.detach().cpu().clone() for k, v in corrector.state_dict().items()},
        "architecture": corrector.architecture(),
        "cells": [cell.key for cell in cells],
        "updates_per_cell": 2000, "batch": 2000, "alpha": 1.0,
        "residual_weight": 1.0, "ot_refresh": 10, "lr": 1e-4, "seed": 88,
        "mf_checkpoint": str(OUT / "mf_two_param.pt"),
    }, OUT / "dp_corrector.pt")
    log(f"corrector done in {result['seconds']:.0f}s")

    log(f"scoring {len(held)} held-out cells")
    pairs_mf, pairs_dp = {}, {}
    for position, cell in enumerate(held, start=1):
        payload = load_cell(cell)
        source = torch.from_numpy(payload["clouds"]["source"])
        reference = payload["clouds"]["reference"]
        mf = sample_two_param(model, source, cell.sigma, cell.mu).numpy().astype(np.float32)
        dp = apply_corrector(corrector, torch.from_numpy(mf), cell.sigma, cell.mu)
        pairs_mf[cell.key] = (reference, mf)
        pairs_dp[cell.key] = (reference, dp)
        if position % 40 == 0:
            log(f"  generated {position}/{len(held)}")
    rows_mf = score_many(pairs_mf, bins=G.BINS, workers=4, log=lambda text: None)
    rows_dp = score_many(pairs_dp, bins=G.BINS, workers=4, log=lambda text: None)
    by_mf = {row["key"]: row["w2"] for row in rows_mf}
    by_dp = {row["key"]: row["w2"] for row in rows_dp}
    detail = []
    for cell in held:
        detail.append({
            "key": cell.key, "sigma": cell.sigma, "mu": cell.mu,
            "w2_mf": by_mf[cell.key], "w2_mf_dp": by_dp[cell.key],
        })
    write_csv(OUT / "detail.csv", detail)
    mean_mf = float(np.mean([row["w2_mf"] for row in detail]))
    mean_dp = float(np.mean([row["w2_mf_dp"] for row in detail]))
    summary = {
        "schedule": "warmcos", "seed": 88,
        "n_train": len(cells), "n_held_out": len(held),
        "stage1_seconds": stage1_seconds,
        "corrector_seconds": result["seconds"],
        "mf_mean_w2": mean_mf, "mf_dp_mean_w2": mean_dp,
        "relative_reduction": float(1.0 - mean_dp / mean_mf),
        "dp_wins": int(sum(row["w2_mf_dp"] < row["w2_mf"] for row in detail)),
    }
    write_json(OUT / "summary.json", summary)
    log(f"held-out MF {mean_mf:.5f}  MF+DP {mean_dp:.5f}  "
        f"reduction {summary['relative_reduction']:+.1%}  "
        f"wins {summary['dp_wins']}/{len(held)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
