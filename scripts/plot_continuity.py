"""Plots that make parameter continuity directly visible.

Three panels:
  1. W2 between neighbouring cells along sigma and along mu, versus the log
     parameter distance, with the noise floor drawn as a horizontal band.
  2. The noise-corrected signal, which should be a straight line through the
     origin if the family is Lipschitz continuous.
  3. Cloud geometry (anisotropy vs isotropic spread) showing that sigma and mu
     act on geometrically different features, i.e. the two-parameter design is
     not degenerate.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from twoparam.data import load_cell  # noqa: E402
from twoparam.protocol import MU_GRID, RESULTS, SIGMA_GRID  # noqa: E402

OUT = RESULTS / "continuity"


def main() -> int:
    sig = json.loads((OUT / "signal_noise.json").read_text(encoding="utf-8"))
    noise = sig["noise"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), layout="constrained")

    # ---- panel 1: measured W2 vs h, with the noise floor -------------------
    ax = axes[0]
    for key, color, label in (("sigma", "#2864B4", r"vary $\sigma$ (fixed $\mu=1$)"),
                              ("mu", "#D37B2F", r"vary $\mu$ (fixed $\sigma=100$)")):
        rows = sorted(sig[key], key=lambda r: r["h"])
        h = [r["h"] for r in rows]
        w = [r["w2"] for r in rows]
        ax.plot(h, w, "o-", color=color, lw=1.7, ms=5, label=label)
    ax.axhline(noise, color="#777777", ls="--", lw=1.2)
    ax.text(0.99, noise + 0.004, f"noise floor {noise:.3f}", ha="right", va="bottom",
            fontsize=7.5, color="#555555", transform=ax.get_yaxis_transform())
    ax.set_xlabel(r"parameter distance $h=\|\Delta\log_{10}\theta\|$")
    ax.set_ylabel(r"$W_2$ (16-bin exact OT)")
    ax.set_title("measured distance vs parameter distance", fontsize=9.5)
    ax.grid(alpha=.25)
    ax.legend(fontsize=7.5, frameon=False)

    # ---- panel 2: noise-corrected signal should be linear ------------------
    ax = axes[1]
    for key, color, label in (("sigma", "#2864B4", r"$\sigma$ direction"),
                              ("mu", "#D37B2F", r"$\mu$ direction")):
        rows = [r for r in sorted(sig[key], key=lambda r: r["h"]) if r["w2_corrected"] > 0]
        if not rows:
            continue
        h = np.array([r["h"] for r in rows])
        c = np.array([r["w2_corrected"] for r in rows])
        ax.plot(h, c, "o", color=color, ms=6, label=label)
        # least-squares line through the origin: the Lipschitz slope
        slope = float((h * c).sum() / (h * h).sum())
        grid = np.linspace(0, h.max() * 1.12, 50)
        ax.plot(grid, slope * grid, "-", color=color, lw=1.3, alpha=.8)
        ax.annotate(f"slope {slope:.2f}", (h[-1], c[-1]), textcoords="offset points",
                    xytext=(6, -2), fontsize=7.5, color=color)
    ax.set_xlabel(r"$h=\|\Delta\log_{10}\theta\|$")
    ax.set_ylabel(r"$\sqrt{W_2^2-\varepsilon_N^2}$")
    ax.set_title("noise-corrected signal is linear in $h$", fontsize=9.5)
    ax.grid(alpha=.25)
    ax.legend(fontsize=7.5, frameon=False)

    # ---- panel 3: the two parameters act on different geometry -------------
    ax = axes[2]
    data = RESULTS / "data" / "full"
    idx = [0, 2, 4, 6, 9]
    for j, color in ((0, "#7A1F1F"), (4, "#2864B4"), (9, "#168173")):
        geo = []
        for i in idx:
            ref = load_cell(data, f"s{i:02d}_m{j:02d}")["clouds"]["reference"].astype(np.float64)
            radial = np.sqrt(ref[:, 1] ** 2 + ref[:, 2] ** 2)
            geo.append((ref[:, 0].std() / radial.std(), np.percentile(radial, 95)))
        g = np.array(geo)
        ax.plot(g[:, 0], g[:, 1], "o-", color=color, ms=5, lw=1.6,
                label=rf"$\mu={MU_GRID[j]:.3g}$")
    ax.set_xlabel(r"anisotropy  std$(x)$ / std$(r_{yz})$")
    ax.set_ylabel(r"isotropic spread  $r_{yz}$ p95")
    ax.set_title(r"$\sigma$ moves anisotropy, $\mu$ moves spread", fontsize=9.5)
    ax.grid(alpha=.25)
    ax.legend(fontsize=7.5, frameon=False, title=r"curves vary $\sigma$", title_fontsize=7)

    for ax in axes:
        ax.tick_params(labelsize=8)

    path = OUT / "continuity_plot.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
