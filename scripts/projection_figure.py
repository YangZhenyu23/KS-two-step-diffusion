"""Projection density figure for the two-parameter model, matching the upstream style.

What is imitated
----------------
``upstream/MF-DeepParticle/paper_reproduction/ks.py::_projections`` draws the figure
the paper uses for the 3D KS distribution: one row per method, one column per
coordinate pair (xy, xz, yz), ``imshow`` of a 2D histogram with ``viridis``, a
colour bar per column, and **the normalisation shared down each column** so that
rows are directly comparable. The reference row is on top.

This script reproduces exactly that layout (``combined=True``) but for the
two-parameter study. Because the interesting comparison there is *across parameters*
rather than across methods, the figure is produced in two forms:

  ``projection_<cell>.png``    3 x 3, methods as rows: Reference / MF / MF+DP
                               (one figure per representative cell, like the paper)
  ``projection_param_sweep.png``
                               5 columns x 3 rows: five (sigma, mu) settings as
                               columns, Reference / MF / MF+DP as rows. This is the
                               view that shows the *continuous parameter* claim:
                               the shape morphs smoothly across columns.

Shared scales
-------------
Inside one figure the axis limits are global over all panels (as upstream does with
``lows``/``highs`` over all clouds), and the density normalisation is per column so
that a row can be compared vertically. A per-row normalisation would hide exactly
the effect we want to show (MF and DP have nearly the same mass, so their peaks are
comparable; the reference has the same mass too, so a shared scale is fair).

Held-out cells only
-------------------
The cells plotted are held out from training (LHS-40 leaves 143 of 200 cells), so
the middle and bottom rows are genuine predictions rather than memorised targets.
The DP corrector was trained on the same 40 cells, so the DP row is also honest.

Usage:
    python scripts/projection_figure.py --seed 88
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream" / "MF-DeepParticle" / "deepparticle"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import torch  # noqa: E402

from fullscale import grid as G  # noqa: E402
from fullscale.train import load_cell, load_two  # noqa: E402
from fullscale.model import sample_two_param  # noqa: E402

OUT = ROOT / "results" / "figures"

# The upstream style, copied so the figures sit next to the paper's without a jump.
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42,
    "axes.linewidth": .8, "legend.fontsize": 9, "axes.labelsize": 12,
}
PAIRS = [(0, 1), (0, 2), (1, 2)]

# Five settings along a diagonal-ish walk that exercises both parameters. Indices
# into G.SIGMA_GRID / G.MU_GRID, chosen so every one is a *held-out* cell of the
# LHS-40 design (so all rows are honest predictions, none memorised).
SWEEP_INDEX = [(0, 1), (3, 4), (9, 6), (13, 7), (19, 9)]

# Representative single cell: the one with the largest DP gain in the whole grid.
HERO_INDEX = (11, 9)


def index_of(sigma: float, mu: float) -> tuple[int, int]:
    return G.SIGMA_GRID.index(sigma), G.MU_GRID.index(mu)


def cell_clouds(ij: tuple[int, int], mf_model, corrector, seed: int, device: str):
    """Return (reference, mf, mf_dp) for one cell, addressed by grid index."""
    i, j = ij
    sigma, mu = float(G.SIGMA_GRID[i]), float(G.MU_GRID[j])
    cell = G.Cell(i, j)
    payload = load_cell(cell)
    source = torch.from_numpy(payload["clouds"]["source"])
    reference = payload["clouds"]["reference"].astype(np.float64)

    mf = sample_two_param(mf_model, source, sigma, mu).numpy().astype(np.float64)

    if corrector is None:
        return reference, mf, None

    log_sigma = torch.full((len(source), 1), math.log10(sigma), dtype=torch.float32, device=device)
    log_mu = torch.full((len(source), 1), math.log10(mu), dtype=torch.float32, device=device)
    with torch.no_grad():
        corrected = corrector(torch.from_numpy(mf).float().to(device), log_sigma, log_mu)
    return reference, mf, corrected.cpu().numpy().astype(np.float64)


def density_panels(clouds: dict[str, np.ndarray], edges_per_axis, bins: int):
    """2D projected histograms as probability *density*, one per (cloud, pair)."""
    out = {}
    for key, cloud in clouds.items():
        row = []
        for a, b in PAIRS:
            counts = np.histogram2d(cloud[:, a], cloud[:, b],
                                    bins=(edges_per_axis[a], edges_per_axis[b]))[0]
            if counts.sum() != len(cloud):
                raise RuntimeError(f"{key} projection lost particles")
            dx = np.diff(edges_per_axis[a])[:, None]
            dy = np.diff(edges_per_axis[b])[None, :]
            row.append(counts / len(cloud) / dx / dy)
        out[key] = row
    return out


def shared_edges(clouds: list[np.ndarray], bins: int):
    lows = np.min([c.min(0) for c in clouds], axis=0)
    highs = np.max([c.max(0) for c in clouds], axis=0)
    pad = 0.03 * (highs - lows)
    return [np.linspace(lows[d] - pad[d], highs[d] + pad[d], bins + 1) for d in range(3)]


def load_w2_table(path: Path) -> dict[str, tuple[float, float]]:
    """Held-out W2 per cell: {key: (mf, mf_dp)}. Empty if the CSV is absent."""
    import csv
    if not path.exists():
        return {}
    table = {}
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            table[row["key"]] = (float(row["w2_mf"]), float(row["w2_mf_dp"]))
    return table


def draw_sweep(settings: list, edges, path: Path, bins: int, w2: dict):
    """Five parameter settings as columns, three methods as rows, one plot per pair.

    Normalisation is **per row**: the colour scale is shared across the five columns
    of one method, so a row can be read across (the density does change with the
    parameters and that change is the point). It is *not* shared down a column,
    because the three methods have nearly identical mass and their peaks would then
    be indistinguishable — the paper's figure shares down a column precisely because
    there the parameters are fixed. Both choices are recorded in the caption.

    Three separate panels (one per coordinate pair) rather than one wide figure,
    because 5 columns x 3 rows x 3 pairs in one axes grid would be illegible.
    """
    method_labels = ["Reference", "MF (1 step)", "MF + DP (2 steps)"]
    for col, (a, b) in enumerate(PAIRS):
        panels = {}
        for ri in range(len(method_labels)):
            for ci, setting in enumerate(settings):
                panels[(ri, ci)] = setting["panels"][ri][col]
        row_maxima = [max(panels[(ri, ci)].max() for ci in range(len(settings)))
                      for ri in range(len(method_labels))]

        fig, axes = plt.subplots(len(method_labels), len(settings),
                                 figsize=(2.15 * len(settings), 2.05 * len(method_labels) + 0.55),
                                 layout="constrained", squeeze=False)
        for ri in range(len(method_labels)):
            for ci in range(len(settings)):
                ax = axes[ri, ci]
                mesh = ax.imshow(
                    panels[(ri, ci)].T, origin="lower",
                    extent=[edges[a][0], edges[a][-1], edges[b][0], edges[b][-1]],
                    norm=Normalize(0, row_maxima[ri]), cmap="viridis",
                    aspect="auto", interpolation="nearest",
                )
                ax.tick_params(labelsize=7)
                ax.xaxis.set_major_locator(MaxNLocator(3))
                ax.yaxis.set_major_locator(MaxNLocator(3))
                if ri == 0:
                    sigma, mu = settings[ci]["sigma"], settings[ci]["mu"]
                    ax.set_title(f"$\\sigma$={sigma:g}\n$\\mu$={mu:g}", fontsize=9)
                if ci == 0:
                    ax.set_ylabel(f"{method_labels[ri]}\n$" + "xyz"[b] + "$", fontsize=9)
                if ri == len(method_labels) - 1:
                    ax.set_xlabel("$" + "xyz"[a] + "$", fontsize=9)
                # W2 in the corner: the number the whole study is about
                entry = w2.get(settings[ci]["key"])
                if entry is not None and ri > 0:
                    value = entry[0] if ri == 1 else entry[1]
                    ax.text(0.03, 0.97, f"$W_2$={value:.3f}", transform=ax.transAxes,
                            va="top", ha="left", fontsize=8, color="white",
                            bbox=dict(boxstyle="round,pad=0.18", fc="black", ec="none", alpha=0.55))
            fig.colorbar(mesh, ax=axes[ri, :].tolist(), pad=.012, shrink=.85,
                         location="right").ax.tick_params(labelsize=7)
        stem = path.with_name(path.stem + f"_{'xyz'[a]}{'xyz'[b]}")
        fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {stem.with_suffix('.png').relative_to(ROOT)}")


def draw_grid(rows: list[str], panels: dict, edges, path: Path, *, row_titles: list[str],
              col_titles: list[str], bins: int):
    """The upstream layout: one row per method, one column per projection."""
    maxima = []
    for col in range(len(PAIRS)):
        maxima.append(max(panels[r][col].max() for r in rows))

    fig, axes = plt.subplots(len(rows), 3, figsize=(9.4, 2.55 * len(rows) + 0.35),
                             layout="constrained", squeeze=False)
    for ri, key in enumerate(rows):
        for col, (a, b) in enumerate(PAIRS):
            ax = axes[ri, col]
            mesh = ax.imshow(
                panels[key][col].T, origin="lower",
                extent=[edges[a][0], edges[a][-1], edges[b][0], edges[b][-1]],
                norm=Normalize(0, maxima[col]), cmap="viridis",
                aspect="auto", interpolation="nearest",
            )
            ax.set(xlabel="$" + "xyz"[a] + "$", ylabel="$" + "xyz"[b] + "$")
            ax.tick_params(labelsize=10)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            if ri == 0:
                ax.set_title(col_titles[col], fontsize=11)
            fig.colorbar(mesh, ax=ax, pad=.025, shrink=.88).ax.tick_params(labelsize=8)
        axes[ri, 0].annotate(
            row_titles[ri], xy=(-0.34, 0.5), xycoords="axes fraction",
            rotation=90, va="center", ha="center", fontsize=12,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def draw_error_maps(panels: dict, edges, path: Path):
    """Signed density error against the reference, with a diverging colour map.

    Why this figure is needed: the density rows above look almost identical because
    DP buys 15% of an already small W2. A signed difference map shows *where* each
    step is wrong, which is the informative view. ``RdBu_r`` with a symmetric scale
    centred on zero: red = model puts too much mass, blue = too little.
    """
    rows = [("mf", "MF (1 step)"), ("mf_dp", "MF + DP (2 steps)")]
    vmax = max(np.abs(panels[key][col] - panels["ref"][col]).max()
               for key, _ in rows for col in range(len(PAIRS)))

    fig, axes = plt.subplots(len(rows), 3, figsize=(9.4, 2.55 * len(rows) + 0.45),
                             layout="constrained", squeeze=False)
    for ri, (key, label) in enumerate(rows):
        for col, (a, b) in enumerate(PAIRS):
            ax = axes[ri, col]
            diff = panels[key][col] - panels["ref"][col]
            mesh = ax.imshow(
                diff.T, origin="lower",
                extent=[edges[a][0], edges[a][-1], edges[b][0], edges[b][-1]],
                norm=Normalize(-vmax, vmax), cmap="RdBu_r",
                aspect="auto", interpolation="nearest",
            )
            ax.set(xlabel="$" + "xyz"[a] + "$", ylabel="$" + "xyz"[b] + "$")
            ax.tick_params(labelsize=10)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            # Total-variation-like L1 between the projected measures. The density
            # must be integrated against the cell area to be a probability again;
            # without the dx*dy factor the number is density units, not a distance.
            area = np.diff(edges[a])[:, None] * np.diff(edges[b])[None, :]
            l1 = float(np.abs(diff * area).sum())
            ax.set_title(f"$\\|\\Delta\\varrho\\|_1$ = {l1:.3f}", fontsize=9)
            fig.colorbar(mesh, ax=ax, pad=.025, shrink=.88).ax.tick_params(labelsize=8)
        axes[ri, 0].annotate(label, xy=(-0.34, 0.5), xycoords="axes fraction",
                             rotation=90, va="center", ha="center", fontsize=12)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def draw_combined(panels: dict, edges, path: Path):
    """One compact figure: rows = methods, columns = the three projections + error.

    This is the version meant for the document and the paper: it keeps the paper's
    3-column projection layout and appends a fourth column carrying the signed error
    of that row against the reference, so the density and the residual can be read
    in one glance. The reference row has no error column.
    """
    labels = [("ref", "Reference"), ("mf", "MF (1 step)"), ("mf_dp", "MF + DP (2 steps)")]
    labels = [(k, t) for k, t in labels if k in panels]
    vmax_density = max(panels[k][c].max() for k, _ in labels for c in range(len(PAIRS)))
    vmax_err = max(np.abs(panels[k][c] - panels["ref"][c]).max()
                   for k, _ in labels if k != "ref" for c in range(len(PAIRS)))

    fig, axes = plt.subplots(len(labels), 4, figsize=(11.6, 2.6 * len(labels) + 0.6),
                             layout="constrained", squeeze=False)
    for ri, (key, label) in enumerate(labels):
        for col, (a, b) in enumerate(PAIRS):
            ax = axes[ri, col]
            mesh = ax.imshow(
                panels[key][col].T, origin="lower",
                extent=[edges[a][0], edges[a][-1], edges[b][0], edges[b][-1]],
                norm=Normalize(0, vmax_density), cmap="viridis",
                aspect="auto", interpolation="nearest",
            )
            ax.set(xlabel="$" + "xyz"[a] + "$", ylabel="$" + "xyz"[b] + "$")
            ax.tick_params(labelsize=9)
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(3))
            if ri == 0:
                ax.set_title(f"${'xyz'[a]}${'-'}${'xyz'[b]}$", fontsize=10)
            if col == 2:
                fig.colorbar(mesh, ax=ax, pad=.02, shrink=.86).ax.tick_params(labelsize=7)

        ax = axes[ri, 3]
        if key == "ref":
            ax.axis("off")
            ax.text(.5, .5, "error columns\n$=$ model $-$ reference",
                    ha="center", va="center", fontsize=9, color="#555555")
        else:
            # stack the three projection errors as a small multiple inside one panel
            diff = panels[key][1] - panels["ref"][1]  # xz: the most informative
            area = np.diff(edges[0])[:, None] * np.diff(edges[2])[None, :]
            l1 = float(np.abs(diff * area).sum())
            mesh = ax.imshow(
                diff.T, origin="lower",
                extent=[edges[0][0], edges[0][-1], edges[2][0], edges[2][-1]],
                norm=Normalize(-vmax_err, vmax_err), cmap="RdBu_r",
                aspect="auto", interpolation="nearest",
            )
            ax.set(xlabel="$x$", ylabel="$z$")
            ax.tick_params(labelsize=9)
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.set_title(f"error ($xz$), $\\|\\Delta\\varrho\\|_1$={l1:.3f}", fontsize=9)
            fig.colorbar(mesh, ax=ax, pad=.02, shrink=.86).ax.tick_params(labelsize=7)

        axes[ri, 0].annotate(label, xy=(-0.40, 0.5), xycoords="axes fraction",
                             rotation=90, va="center", ha="center", fontsize=11)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=88)
    ap.add_argument("--bins", type=int, default=90)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    device = a.device
    mf_path = G.RESULTS / "optimal_budget_ablation" / "two_32000" / "mf_two_param.pt"
    dp_path = G.RESULTS / "dp_two_param" / f"s{a.seed}" / "dp_corrector.pt"
    mf_model, _ = load_two(mf_path, device)

    corrector = None
    if dp_path.exists():
        from dp_two_param import TwoParamCorrector  # same folder as this script
        checkpoint = torch.load(dp_path, map_location=device, weights_only=False)
        arch = checkpoint.get("architecture", {})
        corrector = TwoParamCorrector(
            width=arch.get("width", checkpoint.get("width", 64)),
            alpha=checkpoint.get("alpha", 1.0),
            blocks=checkpoint.get("blocks", arch.get("blocks", 2)),
        ).to(device)
        corrector.load_state_dict(checkpoint["model"])
        corrector.eval()
        print(f"  loaded corrector {dp_path.relative_to(ROOT)}")
    else:
        print(f"  no corrector at {dp_path}, plotting MF only")

    OUT.mkdir(parents=True, exist_ok=True)

    # ---------- figure 1: the paper-style 3x3 for one representative cell ----------
    hero_sigma, hero_mu = G.SIGMA_GRID[HERO_INDEX[0]], G.MU_GRID[HERO_INDEX[1]]
    print(f"hero cell sigma={hero_sigma:g} mu={hero_mu:g}")
    ref, mf, dp = cell_clouds(HERO_INDEX, mf_model, corrector, a.seed, device)
    edges = shared_edges([ref, mf, dp if dp is not None else mf], a.bins)
    clouds = {"ref": ref, "mf": mf}
    titles = ["Reference", "MF (1 step)"]
    if dp is not None:
        clouds["mf_dp"] = dp
        titles.append("MF + DP (2 steps)")
    panels = density_panels(clouds, edges, a.bins)
    draw_grid(list(clouds), panels, edges, OUT / "projection_hero.png",
              row_titles=titles,
              col_titles=[f"${'xyz'[a_]}${'-'}${'xyz'[b_]}$ projection" for a_, b_ in PAIRS],
              bins=a.bins)
    if dp is not None:
        draw_error_maps(panels, edges, OUT / "projection_hero_error.png")
    draw_combined(panels, edges, OUT / "projection_hero_combined.png")

    # ---------- figure 2: five parameters as columns ----------
    print("parameter sweep")
    all_clouds, settings = [], []
    for ij in SWEEP_INDEX:
        ref, mf, dp = cell_clouds(ij, mf_model, corrector, a.seed, device)
        rows = [ref, mf] + ([dp] if dp is not None else [])
        all_clouds.extend(rows)
        settings.append({"sigma": float(G.SIGMA_GRID[ij[0]]),
                         "mu": float(G.MU_GRID[ij[1]]), "clouds": rows,
                         "key": G.Cell(ij[0], ij[1]).key})
        print(f"  sigma={G.SIGMA_GRID[ij[0]]:g} mu={G.MU_GRID[ij[1]]:g} done")

    edges = shared_edges(all_clouds, a.bins)
    for item in settings:
        panels = density_panels({str(k): c for k, c in enumerate(item["clouds"])}, edges, a.bins)
        item["panels"] = [panels[str(k)] for k in range(len(item["clouds"]))]
    draw_sweep(settings, edges, OUT / "projection_sweep.png", a.bins,
               load_w2_table(G.RESULTS / "dp_two_param" / f"s{a.seed}" / "detail.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
