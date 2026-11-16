"""Figures for the report to Harry: the ones the report needs but the repo lacks.

Existing figures are all distribution snapshots (3x3 projections, parameter sweeps). The
report also needs the *quantitative* views:

  fig_rollout        W2 vs number of applications, log scale - the divergence result
  fig_rollout_method W2 at k=4 for MF / MF+DP / MF-two-step at matched cost
  fig_four_way       the four configurations at matched cost, with error bars where known
  fig_alpha          alpha sweep on dev with the test points marked
  fig_physics        anisotropy error per cell, model vs reference
  fig_extrapolation  the ablation: gain inside vs outside the training box

All numbers come from the JSON files already on disk; nothing is recomputed.

Usage:
    python scripts/report_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream" / "MF-DeepParticle" / "deepparticle"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from fullscale import grid as G  # noqa: E402

OUT = ROOT / "results" / "report_figures"
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 10, "axes.linewidth": .8,
    "axes.labelsize": 10, "legend.fontsize": 8, "figure.dpi": 200,
}
COLORS = {"mf": "#2864B4", "mf_dp": "#168173", "mf_2step": "#D37B2F",
          "capacity": "#8B6BAE"}


def load(name: str) -> dict | None:
    path = G.RESULTS / name
    if not path.exists():
        print(f"  skip (missing): {name}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fig_rollout() -> None:
    payload = load("rollout/rollout.json")
    if not payload:
        return
    floor = payload["floor_mean"]
    rows = payload["rows"]
    cells = sorted({r["key"] for r in rows})
    fig, ax = plt.subplots(figsize=(5.4, 3.6), layout="constrained")
    for cell in cells:
        series = sorted((r for r in rows if r["key"] == cell), key=lambda r: r["k"])
        ks = [r["k"] for r in series]
        values = [r["w2_to_physical"] for r in series]
        ax.plot(ks, values, marker="o", ms=4, lw=1.4, label=cell)
    ax.axhline(floor, color="#222222", ls="--", lw=1, label=f"floor {floor:.3f}")
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax.set(xlabel="applications of the one-step map $k$",
           ylabel="$W_2$ to the physical solution at $k\\cdot T$")
    ax.grid(alpha=.23, ls="--")
    ax.legend(frameon=False, ncol=2, fontsize=7)
    fig.savefig(OUT / "fig_rollout.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_rollout.png")


def fig_rollout_method() -> None:
    payload = load("rollout/rollout_compare.json")
    if not payload:
        return
    floor = payload["floor_mean"]
    rows = payload["rows"]
    cells = sorted({r["key"] for r in rows})
    methods = [("mf", "MF"), ("mf_dp", "MF + DP"), ("mf_2step", "MF, two-step")]
    fig, ax = plt.subplots(figsize=(5.4, 3.4), layout="constrained")
    width = 0.26
    positions = np.arange(len(cells))
    for offset, (key, label) in enumerate(methods):
        values = []
        for cell in cells:
            match = [r for r in rows if r["key"] == cell and r["method"] == key and r["k"] == 4]
            values.append(match[0]["w2_physical"] if match else np.nan)
        ax.bar(positions + (offset - 1) * width, values, width,
               label=label, color=COLORS[key])
    ax.axhline(floor, color="#222222", ls="--", lw=1, label=f"floor {floor:.3f}")
    ax.set_yscale("log")
    ax.set_xticks(positions, cells)
    ax.set(ylabel="$W_2$ after 4 applications",
           title="matched cost: every bar is two forward passes")
    ax.grid(alpha=.2, ls="--", axis="y")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "fig_rollout_method.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_rollout_method.png")


def fig_four_way() -> None:
    """Use the repaired numbers when available, else the reported ones."""
    repaired = load("repaired/repaired_four_way.json")
    four = load("four_way/four_way_s88.json")
    capacity = load("capacity_matched/capacity_matched.json")
    if not (repaired or four):
        return

    labels, values, colors = [], [], []
    if repaired:
        clean = repaired["results"]["clean"]
        labels.append("(a) MF, 1 step")
        values.append(clean["mf_dp_alpha0"])
        colors.append(COLORS["mf"])
        labels.append("(d) MF, 2 steps")
        values.append(clean["mf_2step"])
        colors.append(COLORS["mf_2step"])
        labels.append("(b) MF + DP, $\\alpha$=1")
        values.append(clean["mf_dp_alpha1"])
        colors.append(COLORS["mf_dp"])
        labels.append("(b') MF + DP, $\\alpha$=2")
        values.append(clean["mf_dp_alpha2"])
        colors.append(COLORS["mf_dp"])
    if capacity:
        row = capacity["rows"][0]
        labels.append("(c) single net, width 112\n(26,547 params)")
        values.append(row["held_out_mean_w2"])
        colors.append(COLORS["capacity"])

    fig, ax = plt.subplots(figsize=(6.0, 3.6), layout="constrained")
    positions = np.arange(len(labels))
    bars = ax.bar(positions, values, color=colors, width=.62)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.002, f"{value:.4f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(positions, labels, fontsize=8)
    ax.set(ylabel="held-out $W_2$", ylim=(0.085, 0.135))
    ax.grid(alpha=.2, ls="--", axis="y")
    fig.savefig(OUT / "fig_four_way.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_four_way.png")


def fig_alpha() -> None:
    payload = load("alpha_sweep/alpha_sweep_repaired_s88.json")
    if not payload:
        payload = load("alpha_sweep/alpha_sweep_s88.json")
    if not payload:
        return
    dev = {float(k): v for k, v in payload["dev_mean"].items()}
    test = payload.get("test", {})
    best = payload["selected_alpha"]

    fig, ax = plt.subplots(figsize=(5.0, 3.4), layout="constrained")
    xs = sorted(dev)
    ax.plot(xs, [dev[x] for x in xs], marker="o", ms=4, lw=1.6,
            color=COLORS["mf"], label="dev (used to select)")
    for key, value in test.items():
        alpha = float(key.replace("alpha_", ""))
        ax.scatter([alpha], [value], marker="s", s=46, zorder=5,
                   color=COLORS["mf_dp"], label="test (untouched)" if alpha == 1.0 else None)
    ax.axvline(best, color="#222222", ls=":", lw=1)
    ax.annotate(f"$\\alpha^*={best:g}$", xy=(best, ax.get_ylim()[0]),
                xytext=(4, 6), textcoords="offset points", fontsize=8)
    ax.set(xlabel="correction strength $\\alpha$", ylabel="mean $W_2$")
    ax.grid(alpha=.23, ls="--")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "fig_alpha.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_alpha.png")


def fig_physics() -> None:
    payload = load("physics/physical_quantities.json")
    if not payload:
        return
    rows = payload["rows"]
    cells = sorted({r["key"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), layout="constrained")
    for offset, (key, label) in enumerate((("mf", "MF"), ("mf_dp", "MF + DP"))):
        ref = [next(r for r in rows if r["key"] == c and r["method"] == key)["reference"]["anisotropy"]
               for c in cells]
        pred = [next(r for r in rows if r["key"] == c and r["method"] == key)["predicted"]["anisotropy"]
                for c in cells]
        axes[0].scatter(ref, pred, s=44, label=label,
                        color=COLORS[key], marker="o" if offset == 0 else "s")
    limits = [1.0, 1.5]
    axes[0].plot(limits, limits, color="#222222", ls="--", lw=1, label="exact")
    axes[0].set(xlabel="reference anisotropy $\\lambda_1/\\lambda_2$",
                ylabel="model anisotropy", xlim=limits, ylim=limits)
    axes[0].grid(alpha=.23, ls="--")
    axes[0].legend(frameon=False, fontsize=8)

    names = ["centroid_x", "rms_radius", "anisotropy", "eig_1", "eig_2"]
    pretty = ["centroid $x$", "rms radius", "anisotropy", "$\\lambda_1$", "$\\lambda_2$"]
    positions = np.arange(len(names))
    width = 0.36
    for offset, (key, label) in enumerate((("mf", "MF"), ("mf_dp", "MF + DP"))):
        errors = []
        for name in names:
            values = [abs(r["errors"][name]["relative"]) for r in rows
                      if r["method"] == key and r["errors"][name]["relative"] is not None]
            errors.append(np.mean(values) * 100 if values else np.nan)
        axes[1].bar(positions + (offset - 0.5) * width, errors, width,
                    label=label, color=COLORS[key])
    axes[1].set_xticks(positions, pretty, fontsize=8)
    axes[1].set(ylabel="mean $|$relative error$|$ (%)")
    axes[1].grid(alpha=.2, ls="--", axis="y")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "fig_physics.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_physics.png")


def fig_extrapolation() -> None:
    payload = load("extrapolation/summary_s88.json")
    if not payload:
        return
    zones = payload.get("zones")
    if not zones:
        print("  skip fig_extrapolation (no zones)")
        return
    order = ["interior", "sigma_extrap", "mu_extrap", "corner"]
    labels = [z for z in order if z in zones]
    pretty = {"interior": "interior\n(135 cells)", "sigma_extrap": "$\\sigma$ extrap\n(45)",
              "mu_extrap": "$\\mu$ extrap\n(15)", "corner": "corner\n(5)"}
    mf = [zones[z]["mf"] for z in labels]
    dp = [zones[z]["mf_dp"] for z in labels]
    gains = [zones[z]["dp_relative"] for z in labels]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), layout="constrained")
    positions = np.arange(len(labels))
    axes[0].bar(positions - .2, mf, .4, label="MF", color=COLORS["mf"])
    axes[0].bar(positions + .2, dp, .4, label="MF + DP", color=COLORS["mf_dp"])
    axes[0].set_xticks(positions, [pretty[z] for z in labels], fontsize=8)
    axes[0].set(ylabel="$W_2$")
    axes[0].grid(alpha=.2, ls="--", axis="y")
    axes[0].legend(frameon=False, fontsize=8)

    bars = axes[1].bar(positions, [g * 100 for g in gains], .55, color=COLORS["mf_dp"])
    for bar, gain in zip(bars, gains):
        axes[1].text(bar.get_x() + bar.get_width() / 2, gain * 100 + 0.3,
                     f"{gain * 100:.1f}%", ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(positions, [pretty[z] for z in labels], fontsize=8)
    axes[1].set(ylabel="corrector gain (%)")
    axes[1].grid(alpha=.2, ls="--", axis="y")
    fig.savefig(OUT / "fig_extrapolation.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_extrapolation.png")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        fig_rollout()
        fig_rollout_method()
        fig_four_way()
        fig_alpha()
        fig_physics()
        fig_extrapolation()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
