"""Direction of the warmcos corrector against two-step, and sliced W2.

No retraining. Same seed-88 warmcos stage 1 and the corrector that was trained
on it. Held-out cells are the 143 keys in detail.csv.

For each cell the same source particles are pushed three ways. Index i is the
same particle, so the displacements can be compared:

    d_two  = x_two_step - x_one_step
    d_corr = x_corrector - x_one_step

Sliced W2 uses 64 projections and seed 88, the same count as the fixed-learning-rate
check. It is a lower bound from one-dimensional projections, not a replacement
for the 16-bin histogram.

Usage:
    python scripts/warmcos_direction_sliced.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch

from fullscale import grid as G
from fullscale.model import sample_two_param
from fullscale.train import load_cell, load_two
from dp_two_param import TwoParamCorrector, apply_corrector

WARM = G.RESULTS / "harry_followup" / "warmcos_s88"
OUT = G.RESULTS / "harry_followup" / "direction_and_sliced.json"
PROJECTIONS = 64
MIN_NORM = 1e-4


def log(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def sliced_w2(reference: np.ndarray, predicted: np.ndarray, directions: np.ndarray) -> float:
    left = np.sort(reference @ directions.T, axis=0)
    right = np.sort(predicted @ directions.T, axis=0)
    return float(np.sqrt(np.mean((left - right) ** 2)))


def direction_stats(one: np.ndarray, two: np.ndarray, corrected: np.ndarray) -> dict:
    d_two = two - one
    d_corr = corrected - one
    n_two = np.linalg.norm(d_two, axis=1)
    n_corr = np.linalg.norm(d_corr, axis=1)
    keep = (n_two > MIN_NORM) & (n_corr > MIN_NORM)
    cos = np.ones(len(one))
    if keep.any():
        cos_keep = np.sum(d_two[keep] * d_corr[keep], axis=1) / (n_two[keep] * n_corr[keep])
        cos = np.clip(cos_keep, -1.0, 1.0)
    mean_two = d_two.mean(axis=0)
    mean_corr = d_corr.mean(axis=0)
    denom = float(np.linalg.norm(mean_two) * np.linalg.norm(mean_corr))
    mean_cos = float(np.dot(mean_two, mean_corr) / denom) if denom > 0 else 0.0
    dx_two = d_two[:, 0]
    dx_corr = d_corr[:, 0]
    if dx_two.std() > 0 and dx_corr.std() > 0:
        dx_corr_coef = float(np.corrcoef(dx_two, dx_corr)[0, 1])
    else:
        dx_corr_coef = 0.0
    return {
        "mean_particle_cosine": float(cos.mean()) if keep.any() else 0.0,
        "median_particle_cosine": float(np.median(cos)) if keep.any() else 0.0,
        "mean_cos2": float(np.mean(cos ** 2)) if keep.any() else 0.0,
        "fraction_cosine_above_half": float(np.mean(cos > 0.5)) if keep.any() else 0.0,
        "fraction_used": float(keep.mean()),
        "cosine_of_mean_displacement": mean_cos,
        "dx_correlation": dx_corr_coef,
        "mean_step_two": float(n_two.mean()),
        "mean_step_corrector": float(n_corr.mean()),
    }


def load_corrector(device: str) -> TwoParamCorrector:
    checkpoint = torch.load(WARM / "dp_corrector.pt", map_location=device, weights_only=False)
    arch = checkpoint["architecture"]
    model = TwoParamCorrector(
        width=arch["width"], alpha=float(checkpoint["alpha"]), blocks=arch["blocks"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    detail = {row["key"]: row for row in csv.DictReader(
        (WARM / "detail.csv").open(encoding="utf-8"))}
    by_key = {cell.key: cell for cell in G.full_grid()}
    cells = [by_key[key] for key in detail]
    device = "cuda"
    model, _ = load_two(WARM / "mf_two_param.pt", device)
    model.eval()
    corrector = load_corrector(device)
    rng = np.random.default_rng(88)
    directions = rng.standard_normal((PROJECTIONS, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    rows = []
    started = time.perf_counter()
    for position, cell in enumerate(cells, start=1):
        payload = load_cell(cell)
        source = torch.from_numpy(payload["clouds"]["source"])
        reference = payload["clouds"]["reference"].astype(np.float64)
        with torch.no_grad():
            one = sample_two_param(model, source, cell.sigma, cell.mu, steps=1).numpy()
            two = sample_two_param(model, source, cell.sigma, cell.mu, steps=2).numpy()
            corrected = apply_corrector(
                corrector, torch.from_numpy(one.astype(np.float32)), cell.sigma, cell.mu)
        stats = direction_stats(one.astype(np.float64), two.astype(np.float64), corrected.astype(np.float64))
        stats.update({
            "key": cell.key,
            "sigma": cell.sigma,
            "mu": cell.mu,
            "sliced_mf": sliced_w2(reference, one.astype(np.float64), directions),
            "sliced_mf_dp": sliced_w2(reference, corrected.astype(np.float64), directions),
            "histogram_mf": float(detail[cell.key]["w2_mf"]),
            "histogram_mf_dp": float(detail[cell.key]["w2_mf_dp"]),
        })
        rows.append(stats)
        if position % 20 == 0 or position == len(cells):
            elapsed = time.perf_counter() - started
            log(f"{position}/{len(cells)}  {elapsed:.0f}s")

    sliced_mf = float(np.mean([row["sliced_mf"] for row in rows]))
    sliced_dp = float(np.mean([row["sliced_mf_dp"] for row in rows]))
    hist_mf = float(np.mean([row["histogram_mf"] for row in rows]))
    hist_dp = float(np.mean([row["histogram_mf_dp"] for row in rows]))
    summary = {
        "n": len(rows),
        "projections": PROJECTIONS,
        "sliced_seed": 88,
        "sliced_w2_mf": sliced_mf,
        "sliced_w2_mf_dp": sliced_dp,
        "sliced_relative_reduction": float(1.0 - sliced_dp / sliced_mf),
        "histogram_w2_mf": hist_mf,
        "histogram_w2_mf_dp": hist_dp,
        "histogram_relative_reduction": float(1.0 - hist_dp / hist_mf),
        "direction": {
            "mean_particle_cosine": float(np.mean([row["mean_particle_cosine"] for row in rows])),
            "median_of_cell_median_cosine": float(np.median([row["median_particle_cosine"] for row in rows])),
            "mean_cos2": float(np.mean([row["mean_cos2"] for row in rows])),
            "mean_fraction_cosine_above_half": float(np.mean([row["fraction_cosine_above_half"] for row in rows])),
            "mean_cosine_of_mean_displacement": float(np.mean([row["cosine_of_mean_displacement"] for row in rows])),
            "mean_dx_correlation": float(np.mean([row["dx_correlation"] for row in rows])),
            "mean_step_two": float(np.mean([row["mean_step_two"] for row in rows])),
            "mean_step_corrector": float(np.mean([row["mean_step_corrector"] for row in rows])),
        },
        "per_cell": rows,
        "note": (
            "Cosine compares d_two = x_two - x_one with d_corr = x_corrector - x_one "
            "on the same source particles. mean_cos2 is the fraction of the corrector "
            "step lying along the two-step direction."
        ),
    }
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"sliced {sliced_mf:.5f} -> {sliced_dp:.5f}  ({summary['sliced_relative_reduction']:+.1%})")
    log(f"histogram {hist_mf:.5f} -> {hist_dp:.5f}  ({summary['histogram_relative_reduction']:+.1%})")
    direction = summary["direction"]
    log(f"cosine {direction['mean_particle_cosine']:.3f}  "
        f"cos2 {direction['mean_cos2']:.3f}  "
        f"dx corr {direction['mean_dx_correlation']:.3f}")
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
