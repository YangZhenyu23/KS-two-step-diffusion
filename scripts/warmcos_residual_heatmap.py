"""Density heatmaps of the warmcos corrector on two held-out cells.

The 16-bin histogram W2 is one number per cell. These figures keep the particles
and show the x-z projection, which is the shear plane of this Keller-Segel run.
The colour scale is shared across the three densities, and the two error panels
share one diverging scale, so a smaller red-blue pattern is a smaller residual.

Usage:
    python scripts/warmcos_residual_heatmap.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torch

from fullscale import grid as G
from fullscale.model import sample_two_param
from fullscale.train import load_cell, load_two
from dp_two_param import TwoParamCorrector, apply_corrector

OUT = ROOT / "results" / "report_figures"
WARM = G.RESULTS / "harry_followup" / "warmcos_s88"
DETAIL = WARM / "detail.csv"
BINS = 80

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "figure.dpi": 200,
})


def load_detail() -> dict[str, dict]:
    rows = {}
    with DETAIL.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row["key"]] = row
    return rows


def project(cloud: np.ndarray, edges_x, edges_z) -> np.ndarray:
    counts, _, _ = np.histogram2d(cloud[:, 0], cloud[:, 2], bins=[edges_x, edges_z])
    area = np.diff(edges_x)[:, None] * np.diff(edges_z)[None, :]
    return counts / counts.sum() / area


def l1(diff: np.ndarray, edges_x, edges_z) -> float:
    area = np.diff(edges_x)[:, None] * np.diff(edges_z)[None, :]
    return float(np.abs(diff * area).sum())


def correction_field(stage1: np.ndarray, corrected: np.ndarray, edges_x, edges_z, min_count: int = 30):
    ix = np.clip(np.digitize(stage1[:, 0], edges_x) - 1, 0, len(edges_x) - 2)
    iz = np.clip(np.digitize(stage1[:, 2], edges_z) - 1, 0, len(edges_z) - 2)
    shift = corrected - stage1
    acc = np.zeros((len(edges_x) - 1, len(edges_z) - 1))
    count = np.zeros_like(acc)
    np.add.at(acc, (ix, iz), shift[:, 0])
    np.add.at(count, (ix, iz), 1)
    mean = np.divide(acc, count, out=np.zeros_like(acc), where=count >= min_count)
    mean[count < min_count] = np.nan
    return mean


def predict(model, corrector, cell) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = load_cell(cell)
    source = torch.from_numpy(payload["clouds"]["source"])
    reference = payload["clouds"]["reference"].astype(np.float32)
    stage1 = sample_two_param(model, source, cell.sigma, cell.mu).numpy().astype(np.float32)
    corrected = apply_corrector(corrector, torch.from_numpy(stage1), cell.sigma, cell.mu)
    return reference, stage1, corrected


def draw(cell, reference, stage1, corrected, recorded: dict) -> None:
    stacked = np.concatenate([reference, stage1, corrected], axis=0)
    lo = stacked[:, [0, 2]].min(axis=0)
    hi = stacked[:, [0, 2]].max(axis=0)
    pad = 0.02 * np.maximum(hi - lo, 1e-3)
    edges_x = np.linspace(lo[0] - pad[0], hi[0] + pad[0], BINS + 1)
    edges_z = np.linspace(lo[1] - pad[1], hi[1] + pad[1], BINS + 1)
    densities = {
        "reference": project(reference, edges_x, edges_z),
        "stage 1": project(stage1, edges_x, edges_z),
        "stage 1 + corrector": project(corrected, edges_x, edges_z),
    }
    err_stage = densities["stage 1"] - densities["reference"]
    err_corr = densities["stage 1 + corrector"] - densities["reference"]
    field = correction_field(stage1, corrected, edges_x, edges_z)
    vmax = max(panel.max() for panel in densities.values())
    emax = max(np.abs(err_stage).max(), np.abs(err_corr).max())
    finite = field[np.isfinite(field)]
    fmax = float(np.quantile(np.abs(finite), 0.98)) if finite.size else 1.0
    extent = [edges_x[0], edges_x[-1], edges_z[0], edges_z[-1]]

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.4), layout="constrained")
    for ax, (name, panel) in zip(axes[0], densities.items()):
        mesh = ax.imshow(panel.T, origin="lower", extent=extent, cmap="viridis",
                         norm=Normalize(0, vmax), aspect="auto", interpolation="nearest")
        ax.set(title=name, xlabel="$x$", ylabel="$z$")
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    for ax, panel, name in (
        (axes[1, 0], err_stage, "stage 1 $-$ reference"),
        (axes[1, 1], err_corr, "corrector $-$ reference"),
    ):
        mesh = ax.imshow(panel.T, origin="lower", extent=extent, cmap="RdBu_r",
                         norm=Normalize(-emax, emax), aspect="auto", interpolation="nearest")
        ax.set(title=f"{name}\n$L^1$ {l1(panel, edges_x, edges_z):.3f}", xlabel="$x$", ylabel="$z$")
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    mesh = axes[1, 2].imshow(field.T, origin="lower", extent=extent, cmap="RdBu_r",
                             norm=Normalize(-fmax, fmax), aspect="auto", interpolation="nearest")
    axes[1, 2].set(title="mean $\\Delta x$ inside each bin", xlabel="$x$", ylabel="$z$")
    fig.colorbar(mesh, ax=axes[1, 2], fraction=0.046, pad=0.04)
    w2_mf = float(recorded["w2_mf"])
    w2_dp = float(recorded["w2_mf_dp"])
    fig.suptitle(
        f"held-out {cell.key}, $\\sigma={cell.sigma:.0f}$, $\\mu={cell.mu:.3g}$, seed 88\n"
        f"16-bin $W_2$: stage 1 {w2_mf:.4f}, with corrector {w2_dp:.4f}",
        fontsize=11,
    )
    path = OUT / f"fig_corrector_xz_{cell.key}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")
    print(f"    L1 stage1 {l1(err_stage, edges_x, edges_z):.4f}  "
          f"corrector {l1(err_corr, edges_x, edges_z):.4f}  "
          f"mean |dx| {float(np.mean(np.abs(corrected[:, 0] - stage1[:, 0]))):.5f}")


def draw_strip(model, corrector, cells, detail: dict) -> None:
    """Same colour scale on every cell, so a pale panel is a small correction."""
    clouds = []
    for cell in cells:
        reference, stage1, corrected = predict(model, corrector, cell)
        clouds.append((cell, stage1, corrected))
    lo = min(stage1[:, 0].min() for _, stage1, _ in clouds)
    hi = max(stage1[:, 0].max() for _, stage1, _ in clouds)
    zlo = min(stage1[:, 2].min() for _, stage1, _ in clouds)
    zhi = max(stage1[:, 2].max() for _, stage1, _ in clouds)
    edges_x = np.linspace(lo, hi, 61)
    edges_z = np.linspace(zlo, zhi, 61)
    fields = [correction_field(stage1, corrected, edges_x, edges_z) for _, stage1, corrected in clouds]
    finite = np.concatenate([field[np.isfinite(field)] for field in fields])
    fmax = float(np.quantile(np.abs(finite), 0.99))
    extent = [edges_x[0], edges_x[-1], edges_z[0], edges_z[-1]]
    fig, axes = plt.subplots(1, len(cells), figsize=(2.6 * len(cells) + 0.8, 3.3), layout="constrained")
    for ax, cell, field in zip(axes, cells, fields):
        mesh = ax.imshow(field.T, origin="lower", extent=extent, cmap="RdBu_r",
                         norm=Normalize(-fmax, fmax), aspect="auto", interpolation="nearest")
        row = detail[cell.key]
        rel = 1.0 - float(row["w2_mf_dp"]) / float(row["w2_mf"])
        ax.set(title=f"$\\sigma={cell.sigma:.0f}$\n$W_2$ lower by {rel:.1%}", xlabel="$x$", ylabel="$z$")
    fig.colorbar(mesh, ax=axes, fraction=0.046, pad=0.04, label="mean $\\Delta x$")
    fig.suptitle(r"held-out cells at $\mu=10$, seed 88, same colour scale", fontsize=11)
    path = OUT / "fig_corrector_dx_mu10.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}  shared |dx| scale {fmax:.4f}")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    OUT.mkdir(parents=True, exist_ok=True)
    detail = load_detail()
    by_key = {cell.key: cell for cell in G.full_grid()}
    device = "cuda"
    model, _ = load_two(WARM / "mf_two_param.pt", device)
    checkpoint = torch.load(WARM / "dp_corrector.pt", map_location=device, weights_only=False)
    arch = checkpoint["architecture"]
    corrector = TwoParamCorrector(
        width=arch["width"], alpha=float(checkpoint["alpha"]), blocks=arch["blocks"],
    ).to(device)
    corrector.load_state_dict(checkpoint["model"])
    corrector.eval()
    for key in ("s19_m09", "s07_m04"):
        cell = by_key[key]
        reference, stage1, corrected = predict(model, corrector, cell)
        draw(cell, reference, stage1, corrected, detail[key])
    at_mu10 = [by_key[key] for key, row in detail.items() if abs(float(row["mu"]) - 10) < 1e-6]
    at_mu10.sort(key=lambda cell: cell.sigma)
    picked = [at_mu10[index] for index in (0, len(at_mu10) // 3, 2 * len(at_mu10) // 3, len(at_mu10) - 1)]
    draw_strip(model, corrector, picked, detail)


if __name__ == "__main__":
    main()
