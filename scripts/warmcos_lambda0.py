"""Retrain the warmcos corrector with the residual penalty removed.

Same stage-1 checkpoint, same 40 cells, same 2,000 updates per cell, alpha 1.
Only lambda changes from 1 to 0. The stage-1 network is not updated.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fullscale import grid as G
from fullscale.evaluate import score_many, write_json
from fullscale.model import sample_two_param
from fullscale.train import load_cell, load_two
from dp_two_param import apply_corrector, train_corrector
from harry_warmcos_s88 import cells_from_checkpoint, held_out

WARM = G.RESULTS / "harry_followup" / "warmcos_s88"
OUT = G.RESULTS / "harry_followup" / "warmcos_s88_lambda0"
STAGE1 = WARM / "mf_two_param.pt"


def log(text: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {text}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "train.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> int:
    torch.set_num_threads(min(8, torch.get_num_threads()))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    OUT.mkdir(parents=True, exist_ok=True)
    cells = cells_from_checkpoint(STAGE1)
    held = held_out(cells)
    config = G.Config(n_train=G.N_PARTICLES, device="cuda")
    model, _ = load_two(STAGE1, "cuda")
    model.eval()
    log(f"lambda 0 corrector on {len(cells)} cells, score {len(held)} held-out")
    result = train_corrector(
        model, cells, config, updates_per_cell=2000, batch=2000, alpha=1.0,
        blocks=2, residual_weight=0.0, ot_refresh=10, lr=1e-4, seed=88, log=log,
    )
    corrector = result["model"]
    torch.save({
        "model": {key: value.detach().cpu().clone() for key, value in corrector.state_dict().items()},
        "architecture": corrector.architecture(),
        "cells": [cell.key for cell in cells],
        "updates_per_cell": 2000, "batch": 2000, "alpha": 1.0,
        "residual_weight": 0.0, "ot_refresh": 10, "lr": 1e-4, "seed": 88,
        "mf_checkpoint": str(STAGE1),
    }, OUT / "dp_corrector.pt")
    log(f"corrector done in {result['seconds']:.0f}s")

    pairs_mf, pairs_dp = {}, {}
    for position, cell in enumerate(held, start=1):
        payload = load_cell(cell)
        source = torch.from_numpy(payload["clouds"]["source"])
        reference = payload["clouds"]["reference"]
        with torch.no_grad():
            predicted = sample_two_param(model, source, cell.sigma, cell.mu).numpy().astype(np.float32)
        corrected = apply_corrector(corrector, torch.from_numpy(predicted), cell.sigma, cell.mu)
        pairs_mf[cell.key] = (reference, predicted)
        pairs_dp[cell.key] = (reference, corrected)
        if position % 40 == 0:
            log(f"  generated {position}/{len(held)}")
    rows_mf = score_many(pairs_mf, bins=G.BINS, workers=4, log=lambda text: None)
    rows_dp = score_many(pairs_dp, bins=G.BINS, workers=4, log=lambda text: None)
    by_mf = {row["key"]: row["w2"] for row in rows_mf}
    by_dp = {row["key"]: row["w2"] for row in rows_dp}
    detail = [{
        "key": cell.key, "sigma": cell.sigma, "mu": cell.mu,
        "w2_mf": by_mf[cell.key], "w2_mf_dp": by_dp[cell.key],
    } for cell in held]
    mean_mf = float(np.mean([row["w2_mf"] for row in detail]))
    mean_dp = float(np.mean([row["w2_mf_dp"] for row in detail]))
    summary = {
        "residual_weight": 0.0,
        "seed": 88,
        "n_train": len(cells),
        "n_held_out": len(held),
        "corrector_seconds": result["seconds"],
        "mf_mean_w2": mean_mf,
        "mf_dp_mean_w2": mean_dp,
        "relative_reduction": float(1.0 - mean_dp / mean_mf),
        "dp_wins": int(sum(row["w2_mf_dp"] < row["w2_mf"] for row in detail)),
        "lambda1_reference": {"mf_mean_w2": 0.09873952703702839, "mf_dp_mean_w2": 0.09153228233670548},
    }
    write_json(OUT / "summary.json", summary)
    (OUT / "detail.json").write_text(json.dumps(detail), encoding="utf-8")
    log(f"held-out {mean_mf:.5f} -> {mean_dp:.5f}  "
        f"reduction {summary['relative_reduction']:+.1%}  "
        f"wins {summary['dp_wins']}/{len(held)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
