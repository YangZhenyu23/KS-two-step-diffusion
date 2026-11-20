"""Per-cell reading of the two training-budget controls.

The means are already in ``budget_controls.json``. This script does not train.
It reloads the fixed-learning-rate stage 1, the extra MeanFlow run, the OT
fine-tune, and the paired corrector, and scores the same 143 held-out cells.

For each cell it also compares displacements of the same source particles:

    d_extra = x_extra - x_stage1
    d_ot    = x_ot - x_stage1
    d_corr  = x_corrector - x_stage1

Usage:
    python scripts/budget_control_detail.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fullscale import grid as G
from fullscale.evaluate import score_pair, write_json
from fullscale.model import sample_two_param
from fullscale.train import load_cell, load_two
from dp_two_param import TwoParamCorrector, apply_corrector
from harry_warmcos_s88 import HEADLINE, cells_from_checkpoint, held_out

OUT = G.RESULTS / "harry_followup"
MIN_NORM = 1e-4


def log(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def load_corrector(device: str) -> TwoParamCorrector:
    path = G.RESULTS / "dp_two_param" / "s88" / "dp_corrector.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    arch = checkpoint.get("architecture", {})
    model = TwoParamCorrector(
        width=arch.get("width", 64),
        alpha=float(checkpoint.get("alpha", 1.0)),
        blocks=checkpoint.get("blocks", arch.get("blocks", 2)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.alpha = float(checkpoint.get("alpha", 1.0))
    model.eval()
    return model


def pair_direction(base: np.ndarray, other: np.ndarray) -> dict:
    delta = other.astype(np.float64) - base.astype(np.float64)
    norm = np.linalg.norm(delta, axis=1)
    return {"delta": delta, "norm": norm, "mean_step": float(norm.mean())}


def cosine_against(reference_delta: np.ndarray, reference_norm: np.ndarray, other: dict) -> dict:
    keep = (reference_norm > MIN_NORM) & (other["norm"] > MIN_NORM)
    if not np.any(keep):
        cosine = 0.0
        cos2 = 0.0
    else:
        raw = np.sum(reference_delta[keep] * other["delta"][keep], axis=1)
        raw = raw / (reference_norm[keep] * other["norm"][keep])
        raw = np.clip(raw, -1.0, 1.0)
        cosine = float(raw.mean())
        cos2 = float(np.mean(raw ** 2))
    return {
        "mean_cosine": cosine,
        "mean_cos2": cos2,
        "mean_step": other["mean_step"],
        "fraction_used": float(np.mean(keep)),
    }


def main() -> int:
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    cells = cells_from_checkpoint(HEADLINE)
    held = held_out(cells)
    device = "cuda"
    stage1, _ = load_two(HEADLINE, device)
    extra, _ = load_two(OUT / "extra_updates_s88" / "mf_two_param.pt", device)
    ot, _ = load_two(OUT / "ot_finetune_s88" / "mf_two_param.pt", device)
    corrector = load_corrector(device)
    log(f"scoring {len(held)} held-out cells")

    rows = []
    started = time.perf_counter()
    for position, cell in enumerate(held, start=1):
        payload = load_cell(cell)
        source = torch.from_numpy(payload["clouds"]["source"])
        reference = payload["clouds"]["reference"]
        with torch.no_grad():
            base = sample_two_param(stage1, source, cell.sigma, cell.mu).numpy().astype(np.float32)
            extra_cloud = sample_two_param(extra, source, cell.sigma, cell.mu).numpy().astype(np.float32)
            ot_cloud = sample_two_param(ot, source, cell.sigma, cell.mu).numpy().astype(np.float32)
        corrected = apply_corrector(corrector, torch.from_numpy(base), cell.sigma, cell.mu)
        w2_base = score_pair(reference, base, G.BINS)["w2"]
        w2_extra = score_pair(reference, extra_cloud, G.BINS)["w2"]
        w2_ot = score_pair(reference, ot_cloud, G.BINS)["w2"]
        w2_dp = score_pair(reference, corrected, G.BINS)["w2"]
        corr = pair_direction(base, corrected)
        extra_dir = cosine_against(corr["delta"], corr["norm"], pair_direction(base, extra_cloud))
        ot_dir = cosine_against(corr["delta"], corr["norm"], pair_direction(base, ot_cloud))
        rows.append({
            "key": cell.key,
            "sigma": cell.sigma,
            "mu": cell.mu,
            "w2_stage1": w2_base,
            "w2_extra_mf": w2_extra,
            "w2_ot": w2_ot,
            "w2_corrector": w2_dp,
            "step_corrector": corr["mean_step"],
            "step_extra": extra_dir["mean_step"],
            "step_ot": ot_dir["mean_step"],
            "cosine_extra": extra_dir["mean_cosine"],
            "cos2_extra": extra_dir["mean_cos2"],
            "cosine_ot": ot_dir["mean_cosine"],
            "cos2_ot": ot_dir["mean_cos2"],
        })
        if position % 20 == 0 or position == len(held):
            log(f"{position}/{len(held)}  {time.perf_counter() - started:.0f}s")

    def mean(name: str) -> float:
        return float(np.mean([row[name] for row in rows]))

    stage1_mean = mean("w2_stage1")
    extra_mean = mean("w2_extra_mf")
    ot_mean = mean("w2_ot")
    corr_mean = mean("w2_corrector")
    corr_drop = stage1_mean - corr_mean
    summary = {
        "n_held_out": len(rows),
        "w2_stage1": stage1_mean,
        "w2_extra_mf": extra_mean,
        "w2_ot": ot_mean,
        "w2_corrector": corr_mean,
        "extra_vs_stage1": float(extra_mean / stage1_mean - 1.0),
        "ot_vs_stage1": float(1.0 - ot_mean / stage1_mean),
        "corrector_vs_stage1": float(1.0 - corr_mean / stage1_mean),
        "fraction_of_corrector_drop_recovered_by_ot": float((stage1_mean - ot_mean) / corr_drop),
        "cells_extra_lower": int(sum(row["w2_extra_mf"] < row["w2_stage1"] for row in rows)),
        "cells_ot_lower": int(sum(row["w2_ot"] < row["w2_stage1"] for row in rows)),
        "cells_corrector_lower": int(sum(row["w2_corrector"] < row["w2_stage1"] for row in rows)),
        "cells_ot_closer_than_corrector": int(sum(row["w2_ot"] <= row["w2_corrector"] for row in rows)),
        "direction": {
            "mean_cosine_extra": mean("cosine_extra"),
            "mean_cos2_extra": mean("cos2_extra"),
            "mean_cosine_ot": mean("cosine_ot"),
            "mean_cos2_ot": mean("cos2_ot"),
            "mean_step_stage1_to_extra": mean("step_extra"),
            "mean_step_stage1_to_ot": mean("step_ot"),
            "mean_step_corrector": mean("step_corrector"),
        },
        "published_means": {
            "w2_stage1": 0.12276673234794717,
            "w2_extra_mf": 0.1246544193496638,
            "w2_ot": 0.11910603363559288,
            "w2_corrector": 0.10429119590156308,
        },
        "per_cell": rows,
        "note": (
            "No retraining. Displacements are on the same source particles. "
            "Cosine compares each control's displacement with the corrector's."
        ),
    }
    write_json(OUT / "budget_controls_detail.json", summary)
    log(
        f"stage1 {stage1_mean:.5f}  extra {extra_mean:.5f}  "
        f"ot {ot_mean:.5f}  corrector {corr_mean:.5f}"
    )
    log(
        f"OT recovers {summary['fraction_of_corrector_drop_recovered_by_ot']:.1%} of the drop; "
        f"cosine ot {summary['direction']['mean_cosine_ot']:.3f}  "
        f"extra {summary['direction']['mean_cosine_extra']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
