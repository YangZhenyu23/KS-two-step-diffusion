"""E5: the cross-seed reproducibility figure, both parameter settings side by side.

Three quantities on one axis, because the comparison that matters is *variance*, not level:

  single-parameter, stage 1        sd 0.08109 over 3 seeds x 20 sigma
  single-parameter, stage 1 + corr sd 0.01627
  two-parameter, stage 1           sd 0.00892 over 3 seeds x 143 cells
  two-parameter, stage 1 + corr    sd 0.00267

The two settings have very different error levels (the single-parameter model is far worse in
absolute terms because it cannot see mu), so plotting raw W2 would hide the effect. The figure
therefore plots, per setting, the *spread across seeds* relative to that setting's own mean:
the quantity the corrector is claimed to reduce.

Also shown: the per-sigma spread of the single-parameter model, to make clear the reduction is
not an artefact of averaging over cells.

Usage:
    python scripts/e5_reproducibility_figure.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fullscale import grid as G  # noqa: E402

OUT = ROOT / "results" / "report_figures"
STYLE = {"font.family": "DejaVu Sans", "font.size": 10, "axes.linewidth": .8,
         "axes.labelsize": 10, "legend.fontsize": 8, "figure.dpi": 200}
COLORS = {"mf": "#2864B4", "mf_dp": "#168173"}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- single parameter: per-sigma spread across three seeds ----
    metrics = ROOT / "results" / "ks3d" / "evaluation" / "ks" / "metrics_per_seed.csv"
    rows = list(csv.DictReader(open(metrics, encoding="utf-8")))
    groups = defaultdict(list)
    for row in rows:
        groups[(float(row["level"]), row["method"])].append(float(row["w2"]))

    levels = sorted({key[0] for key in groups})
    per_sigma = {}
    for method in ("mf", "mf_dp"):
        per_sigma[method] = {
            "mean": [float(np.mean(groups[(lv, method)])) for lv in levels],
            "sd": [float(np.std(groups[(lv, method)], ddof=1)) for lv in levels],
        }

    single = {}
    for method in ("mf", "mf_dp"):
        sds = np.array(per_sigma[method]["sd"])
        means = np.array(per_sigma[method]["mean"])
        single[method] = {"mean_sd": float(sds.mean()),
                          "mean_level": float(means.mean()),
                          "cv": float((sds / means).mean())}

    # ---- two parameter: three-seed summary ----
    three = json.loads((G.RESULTS / "three_seed/three_seed.json").read_text("utf-8"))
    two = {
        "mf": {"mean_sd": three["mf_two_sd"], "mean_level": three["mf_two_mean"]},
        "mf_dp": {"mean_sd": three["mf_dp_sd"], "mean_level": three["mf_dp_mean"]},
    }
    for block in two.values():
        block["cv"] = block["mean_sd"] / block["mean_level"]

    # ---- figure: two panels ----
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), layout="constrained")

    ax = axes[0]
    positions = [0, 1]
    width = 0.34
    for offset, method, label in ((-0.5, "mf", "stage 1"),
                                  (0.5, "mf_dp", "stage 1 + corrector")):
        values = [single[method]["mean_sd"], two[method]["mean_sd"]]
        bars = ax.bar([p + offset * width for p in positions], values, width,
                      label=label, color=COLORS[method])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value * 1.04, f"{value:.4f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(positions, ["single parameter\n($\\mu=1$, 20 $\\sigma$)",
                              "two parameters\n(143 cells)"])
    ax.set(ylabel="standard deviation across seeds", yscale="log")
    ax.grid(alpha=.23, ls="--", axis="y")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for offset, method, label in ((-0.5, "mf", "stage 1"),
                                  (0.5, "mf_dp", "stage 1 + corrector")):
        values = [100 * single[method]["cv"], 100 * two[method]["cv"]]
        bars = ax.bar([p + offset * width for p in positions], values, width,
                      label=label, color=COLORS[method])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.15, f"{value:.1f}%",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(positions, ["single parameter", "two parameters"])
    ax.set(ylabel="coefficient of variation across seeds (%)")
    ax.grid(alpha=.23, ls="--", axis="y")

    fig.savefig(OUT / "fig_reproducibility.png", bbox_inches="tight")
    plt.close(fig)

    payload = {"single_parameter": single, "two_parameter": two,
               "per_sigma_single": per_sigma,
               "note": ("The two settings have different absolute error levels, so the "
                        "figure compares spread, not level.")}
    (G.RESULTS / "reproducibility" ).mkdir(parents=True, exist_ok=True)
    (G.RESULTS / "reproducibility" / "reproducibility.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote results/report_figures/fig_reproducibility.png")
    print(f"  single-param  sd {single['mf']['mean_sd']:.5f} -> "
          f"{single['mf_dp']['mean_sd']:.5f} "
          f"({100 * (1 - single['mf_dp']['mean_sd'] / single['mf']['mean_sd']):.0f}% lower)")
    print(f"  two-param     sd {two['mf']['mean_sd']:.5f} -> {two['mf_dp']['mean_sd']:.5f} "
          f"({100 * (1 - two['mf_dp']['mean_sd'] / two['mf']['mean_sd']):.0f}% lower)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
