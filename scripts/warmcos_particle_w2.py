"""Exact particle W2 for the warmcos pair, on a shared subsample of 10,000.

The 16-bin histogram and the 64-projection sliced distance are already on disk.
This adds the coupling the histogram drops: optimal transport between the raw
points. One index vector per cell is reused for the reference, the one-step
cloud, and the corrected cloud.

Usage:
    python scripts/warmcos_particle_w2.py
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
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fullscale import grid as G
from fullscale.model import sample_two_param
from fullscale.train import load_cell, load_two
from dp_two_param import TwoParamCorrector, apply_corrector

WARM = G.RESULTS / "harry_followup" / "warmcos_s88"
OUT = G.RESULTS / "harry_followup" / "warmcos_particle_w2.json"
SUBSAMPLE = 10_000


def log(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def exact_w2(left: np.ndarray, right: np.ndarray) -> float:
    cost = cdist(left, right, metric="sqeuclidean")
    rows, columns = linear_sum_assignment(cost)
    return float(np.sqrt(cost[rows, columns].mean()))


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
    detail = {row["key"]: row for row in csv.DictReader((WARM / "detail.csv").open(encoding="utf-8"))}
    by_key = {cell.key: cell for cell in G.full_grid()}
    cells = [by_key[key] for key in detail]
    model, _ = load_two(WARM / "mf_two_param.pt", "cuda")
    model.eval()
    corrector = load_corrector("cuda")
    rows = []
    started = time.perf_counter()
    for position, cell in enumerate(cells, start=1):
        payload = load_cell(cell)
        source = torch.from_numpy(payload["clouds"]["source"])
        reference = payload["clouds"]["reference"]
        with torch.no_grad():
            stage1 = sample_two_param(model, source, cell.sigma, cell.mu).numpy()
        corrected = apply_corrector(corrector, torch.from_numpy(stage1), cell.sigma, cell.mu)
        index = np.random.default_rng(20261120 + position).choice(len(reference), SUBSAMPLE, replace=False)
        particle_mf = exact_w2(reference[index], stage1[index])
        particle_dp = exact_w2(reference[index], corrected[index])
        rows.append({
            "key": cell.key,
            "sigma": cell.sigma,
            "mu": cell.mu,
            "particle_mf": particle_mf,
            "particle_dp": particle_dp,
            "histogram_mf": float(detail[cell.key]["w2_mf"]),
            "histogram_dp": float(detail[cell.key]["w2_mf_dp"]),
        })
        if position % 10 == 0 or position == len(cells):
            log(f"{position}/{len(cells)}  {time.perf_counter() - started:.0f}s")
    particle_mf = float(np.mean([row["particle_mf"] for row in rows]))
    particle_dp = float(np.mean([row["particle_dp"] for row in rows]))
    hist_mf = float(np.mean([row["histogram_mf"] for row in rows]))
    hist_dp = float(np.mean([row["histogram_dp"] for row in rows]))
    summary = {
        "n": len(rows),
        "subsample": SUBSAMPLE,
        "particle_w2_mf": particle_mf,
        "particle_w2_mf_dp": particle_dp,
        "particle_relative_reduction": float(1.0 - particle_dp / particle_mf),
        "histogram_w2_mf": hist_mf,
        "histogram_w2_mf_dp": hist_dp,
        "histogram_relative_reduction": float(1.0 - hist_dp / hist_mf),
        "per_cell": rows,
        "note": "Exact OT on one shared subsample of 10,000 points per cell. Same warmcos stage 1 and its lambda=1 corrector.",
    }
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"particle {particle_mf:.5f} -> {particle_dp:.5f}  ({summary['particle_relative_reduction']:+.1%})")
    log(f"histogram {hist_mf:.5f} -> {hist_dp:.5f}  ({summary['histogram_relative_reduction']:+.1%})")


if __name__ == "__main__":
    main()
