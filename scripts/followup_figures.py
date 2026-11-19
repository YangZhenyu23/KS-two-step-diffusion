"""Figures for the follow-up note. Every number is read from a JSON already on disk.

The (r, t) panel is the one exception: the stored file only keeps two fractions, so the
density is redrawn from the same sampler as ``fullscale.model.sample_times``.

Usage:
    python scripts/followup_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "report_figures"
FOLLOW = ROOT / "results" / "fullscale" / "harry_followup"
PAIRED = ROOT / "results" / "fullscale" / "dp_two_param"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 200,
})

BLUE = "#2864B4"
GREEN = "#168173"
ORANGE = "#D37B2F"
PURPLE = "#8B6BAE"
GRAY = "#8A8F98"
INK = "#222222"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def label_bar(ax, bar, value: float, dy: float, digits: int = 4) -> None:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + dy,
        f"{value:.{digits}f}",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=INK,
    )


def label_grouped(ax, groups: list[list[tuple[float, float]]], gap: float, digits: int = 4) -> None:
    """Place one label per bar, lifting any label that would sit on the one below it."""
    for items in groups:
        last = -1e9
        for x, value in sorted(items, key=lambda item: item[1]):
            y = value + gap * 0.35
            if y < last + gap:
                y = last + gap
            ax.text(x, y, f"{value:.{digits}f}", ha="center", va="bottom", fontsize=7, color=INK)
            last = y


def fig_schedule() -> None:
    warm = load(FOLLOW / "warmcos_s88" / "summary.json")
    paired = load(PAIRED / "s88" / "summary.json")
    groups = [
        ("fixed $10^{-3}$\ncorrector $-$15.0%", paired["mf_mean_w2"], paired["mf_dp_mean_w2"]),
        ("warm-up + cosine\ncorrector $-$7.3%", warm["mf_mean_w2"], warm["mf_dp_mean_w2"]),
    ]
    fig, ax = plt.subplots(figsize=(5.8, 3.6), layout="constrained")
    xs = np.arange(len(groups))
    width = 0.34
    stage = [row[1] for row in groups]
    corrected = [row[2] for row in groups]
    bars_a = ax.bar(xs - width / 2, stage, width, color=BLUE, label="stage 1")
    bars_b = ax.bar(xs + width / 2, corrected, width, color=GREEN, label="stage 1 + corrector")
    for bar, value in zip(bars_a, stage):
        label_bar(ax, bar, value, 0.0016)
    for bar, value in zip(bars_b, corrected):
        label_bar(ax, bar, value, 0.0016)
    ax.axhline(paired["mf_dp_mean_w2"], color=INK, ls="--", lw=0.9, zorder=0)
    ax.text(
        1.48, paired["mf_dp_mean_w2"] + 0.0012,
        "fixed-lr corrected",
        ha="right", va="bottom", fontsize=7.5, color=INK,
    )
    ax.set_xticks(xs, [row[0] for row in groups])
    ax.set(ylabel="held-out $W_2$, 143 cells, seed 88", ylim=(0.082, 0.142))
    ax.grid(alpha=0.2, ls="--", axis="y")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(OUT / "fig_followup_schedule.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_followup_schedule.png")


def fig_four_row() -> None:
    payload = load(FOLLOW / "four_row_warmcos.json")
    seeds = [payload["c_width112"]["per_seed"][key] for key in ("88", "89", "90")]
    labels = ["(a) one step", "(b) + corrector", "(c) width 112\nmean of 3 seeds", "(d) two steps"]
    values = [
        payload["a_one_step"],
        payload["b_corrector"],
        payload["c_width112"]["mean"],
        payload["d_two_step"],
    ]
    colors = [BLUE, GREEN, PURPLE, ORANGE]
    base = payload["a_one_step"]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.55), layout="constrained")
    positions = np.arange(4)
    bars = axes[0].bar(positions, values, color=colors, width=0.62)
    jitter = np.array([-0.13, 0.0, 0.13])
    axes[0].scatter(np.full(3, 2) + jitter, seeds, s=26, color=INK, zorder=5)
    for index, (bar, value) in enumerate(zip(bars, values)):
        if index == 2:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2, max(seeds) + 0.0022,
                f"{value:.4f}", ha="center", va="bottom", fontsize=7.5, color=INK,
            )
        else:
            label_bar(axes[0], bar, value, 0.0014)
    axes[0].set_xticks(positions, labels, fontsize=8)
    axes[0].set(ylabel="held-out $W_2$ on 112 common cells", ylim=(0.074, 0.120))
    axes[0].grid(alpha=0.2, ls="--", axis="y")

    gain_keys = [
        ("(b) + corrector", 100 * payload["b_relative_vs_a"], GREEN, None),
        ("(c) width 112\nmean of 3 seeds", 100 * payload["c_width112"]["relative_vs_a"], PURPLE,
         [100 * (base - seed) / base for seed in seeds]),
        ("(d) two steps", 100 * payload["d_relative_vs_a"], ORANGE, None),
    ]
    gain_x = np.arange(3)
    gain_values = [row[1] for row in gain_keys]
    gain_bars = axes[1].bar(gain_x, gain_values, color=[row[2] for row in gain_keys], width=0.62)
    axes[1].axhline(0, color=INK, lw=0.8)
    for bar, value, row in zip(gain_bars, gain_values, gain_keys):
        center = bar.get_x() + bar.get_width() / 2
        if row[3] is not None:
            axes[1].scatter(np.full(3, center) + jitter, row[3], s=26, color=INK, zorder=5)
        if value >= 0:
            text_y, valign = value + 0.55, "bottom"
        else:
            lowest = min([value, *row[3]]) if row[3] is not None else value
            text_y, valign = lowest - 1.6, "top"
        axes[1].text(center, text_y, f"{value:+.1f}%", ha="center", va=valign, fontsize=8, color=INK)
    axes[1].set_xticks(gain_x, [row[0] for row in gain_keys], fontsize=8)
    axes[1].set(ylabel="error reduction vs (a), percent", ylim=(-11, 20))
    axes[1].grid(alpha=0.2, ls="--", axis="y")
    fig.savefig(OUT / "fig_followup_four_row.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_followup_four_row.png")


def fig_budget() -> None:
    budget = load(FOLLOW / "budget_controls.json")
    paired = load(PAIRED / "s88" / "summary.json")
    labels = [
        "stage 1",
        "+80k MeanFlow\nupdates",
        "+80k OT updates\nsame network",
        "paired corrector\nsecond network",
    ]
    values = [
        budget["headline_mf"],
        budget["extra_mf_updates"],
        budget["ot_finetune_same_network"],
        paired["mf_dp_mean_w2"],
    ]
    colors = [BLUE, GRAY, ORANGE, GREEN]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), layout="constrained")
    bars = ax.bar(np.arange(4), values, color=colors, width=0.64)
    for bar, value in zip(bars, values):
        label_bar(ax, bar, value, 0.0012)
    ax.set_xticks(np.arange(4), labels, fontsize=8)
    ax.set(ylabel="held-out $W_2$, 143 cells, seed 88", ylim=(0.096, 0.140))
    ax.grid(alpha=0.2, ls="--", axis="y")
    fig.savefig(OUT / "fig_followup_budget.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_followup_budget.png")


def sample_intervals(n: int = 1_000_000, seed: int = 0):
    """Same rule as ``fullscale.model.sample_times``."""
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal((n, 2)).astype(np.float32) * 1.0 + np.float32(-0.4)
    samples = 1.0 / (1.0 + np.exp(-samples))
    t = np.maximum(samples[:, 0], samples[:, 1])
    r = np.minimum(samples[:, 0], samples[:, 1])
    chosen = rng.permutation(n)[: n // 2]
    r[chosen] = t[chosen]
    return r, t


def fig_intervals() -> None:
    stored = load(FOLLOW / "consistency.json")
    r, t = sample_intervals()
    gap = t - r
    short = float(np.mean(gap < 0.5))
    diagonal = float(np.mean(np.abs(gap) < 1e-8))
    print(f"  interval redraw: short={short:.4f} (stored {stored['fraction_interval_shorter_than_half']:.4f}), "
          f"diagonal={diagonal:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), layout="constrained")
    hexbin = axes[0].hexbin(
        r, t, gridsize=48, cmap="viridis", mincnt=1,
        norm=LogNorm(), linewidths=0, extent=(0, 1, 0, 1),
    )
    axes[0].plot([0, 1], [0, 1], color="white", lw=0.8, alpha=0.85)
    axes[0].plot([0, 0.5], [0.5, 1.0], color="white", ls="--", lw=1.1)
    axes[0].scatter([0.0], [1.0], s=64, marker="*", color="white", edgecolor=INK, linewidths=0.7,
                    zorder=5, clip_on=False, label="one step, $(r,t)=(0,1)$")
    axes[0].scatter([0.0, 0.5], [0.5, 1.0], s=32, marker="s", color=ORANGE, edgecolor="white",
                    linewidths=0.7, zorder=5, clip_on=False, label="two sub-steps, length 0.5")
    axes[0].set(xlabel="$r$", ylabel="$t$", xlim=(-0.04, 1.04), ylim=(-0.04, 1.04), aspect="equal")
    axes[0].legend(frameon=True, framealpha=0.92, fontsize=7, loc="lower right")
    fig.colorbar(hexbin, ax=axes[0], fraction=0.046, pad=0.04).set_label("draws per bin")

    quantiles = np.linspace(0, 1, 400)
    axes[1].plot(np.quantile(gap, quantiles), quantiles, color=BLUE, lw=1.8)
    axes[1].axvline(0.5, color=INK, ls="--", lw=0.9)
    axes[1].axhline(stored["fraction_interval_shorter_than_half"], color=GRAY, ls=":", lw=0.9)
    axes[1].scatter([0.5], [stored["fraction_interval_shorter_than_half"]], s=28, color=ORANGE, zorder=5)
    axes[1].annotate(
        "95.75% shorter than 0.5",
        xy=(0.5, stored["fraction_interval_shorter_than_half"]),
        xytext=(0.62, 0.78),
        fontsize=8, color=INK,
        arrowprops={"arrowstyle": "->", "color": INK, "lw": 0.8},
    )
    axes[1].annotate(
        "50% set to $r=t$",
        xy=(0.02, 0.50),
        xytext=(0.18, 0.30),
        fontsize=8, color=INK,
        arrowprops={"arrowstyle": "->", "color": INK, "lw": 0.8},
    )
    axes[1].set(xlabel="interval length $t-r$", ylabel="cumulative fraction",
                xlim=(-0.02, 1.02), ylim=(0, 1.02))
    axes[1].grid(alpha=0.2, ls="--")
    fig.savefig(OUT / "fig_followup_intervals.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_followup_intervals.png")


def fig_alpha() -> None:
    payload = load(FOLLOW / "alpha_seeds.json")
    alphas = ("0.0", "1.0", "2.0")
    colors = [BLUE, GREEN, ORANGE]
    labels = [r"$\alpha=0$", r"$\alpha=1$", r"$\alpha=2$"]
    seeds = ("89", "90")
    fig, ax = plt.subplots(figsize=(5.6, 3.5), layout="constrained")
    xs = np.arange(len(seeds))
    width = 0.24
    for offset, (alpha, color, label) in enumerate(zip(alphas, colors, labels)):
        values = [payload[seed][alpha] for seed in seeds]
        bars = ax.bar(xs + (offset - 1) * width, values, width, color=color, label=label)
        for bar, value in zip(bars, values):
            label_bar(ax, bar, value, 0.0012)
    ax.set_xticks(xs, [f"seed {seed}" for seed in seeds])
    ax.set(ylabel="mean $W_2$ on 93 test cells", ylim=(0.092, 0.148))
    ax.grid(alpha=0.2, ls="--", axis="y")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.savefig(OUT / "fig_followup_alpha.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_followup_alpha.png")


def fig_physics() -> None:
    payload = load(FOLLOW / "particle_and_physics.json")
    keys = [("centroid_x", "centroid $x$"), ("rms_radius", "rms radius"), ("anisotropy", "anisotropy")]
    stage = [payload["mean_abs_relative_error"]["mf"][key] for key, _ in keys]
    corrected = [payload["mean_abs_relative_error"]["mf_dp"][key] for key, _ in keys]
    fig, ax = plt.subplots(figsize=(5.8, 3.5), layout="constrained")
    xs = np.arange(len(keys))
    width = 0.34
    bars_a = ax.bar(xs - width / 2, stage, width, color=BLUE, label="stage 1")
    bars_b = ax.bar(xs + width / 2, corrected, width, color=GREEN, label="stage 1 + corrector")
    for bar, value in zip(bars_a, stage):
        label_bar(ax, bar, value, 0.0006, digits=4)
    for bar, value in zip(bars_b, corrected):
        label_bar(ax, bar, value, 0.0006, digits=4)
    ax.set_xticks(xs, [label for _, label in keys])
    ax.set(ylabel="mean absolute relative error, 143 cells", ylim=(0, 0.036))
    ax.grid(alpha=0.2, ls="--", axis="y")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(OUT / "fig_followup_physics.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_followup_physics.png")


def fig_multiseed() -> None:
    shared = load(FOLLOW / "multiseed_corrector" / "summary.json")
    seeds = ("88", "89", "90")
    stage, paired, common = [], [], []
    for seed in seeds:
        own = load(PAIRED / f"s{seed}" / "summary.json")
        stage.append(own["mf_mean_w2"])
        paired.append(own["mf_dp_mean_w2"])
        common.append(shared["per_seed"][seed]["mf_dp"])
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), layout="constrained")
    xs = np.arange(len(seeds))
    width = 0.24
    series = [
        (stage, BLUE, "stage 1"),
        (common, ORANGE, "one shared corrector"),
        (paired, GREEN, "corrector paired to that seed"),
    ]
    placed: list[list[tuple[float, float]]] = [[] for _ in seeds]
    for offset, (values, color, label) in enumerate(series):
        axes[0].bar(xs + (offset - 1) * width, values, width, color=color, label=label)
        for index, value in enumerate(values):
            placed[index].append((xs[index] + (offset - 1) * width, value))
    label_grouped(axes[0], placed, gap=0.0062)
    axes[0].set_xticks(xs, [f"seed {seed}" for seed in seeds])
    axes[0].set(ylabel="held-out $W_2$, each seed's 143 cells", ylim=(0.090, 0.158))
    axes[0].grid(alpha=0.2, ls="--", axis="y")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper right")

    paired_gain = [100 * (a - b) / a for a, b in zip(stage, paired)]
    shared_gain = [100 * (a - b) / a for a, b in zip(stage, common)]
    axes[1].bar(xs - width / 2, paired_gain, width, color=GREEN, label="paired")
    axes[1].bar(xs + width / 2, shared_gain, width, color=ORANGE, label="shared")
    axes[1].axhline(0, color=INK, lw=0.8)
    for x, value in zip(xs - width / 2, paired_gain):
        axes[1].text(x, value + 0.4, f"{value:+.1f}%", ha="center", va="bottom", fontsize=7.5)
    for x, value in zip(xs + width / 2, shared_gain):
        axes[1].text(x, value + 0.4, f"{value:+.1f}%", ha="center", va="bottom", fontsize=7.5)
    axes[1].set_xticks(xs, [f"seed {seed}" for seed in seeds])
    axes[1].set(ylabel="error reduction vs that seed's stage 1, percent", ylim=(0, 24))
    axes[1].grid(alpha=0.2, ls="--", axis="y")
    axes[1].legend(frameon=False, loc="upper right")
    fig.savefig(OUT / "fig_followup_multiseed.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_followup_multiseed.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_schedule()
    fig_four_row()
    fig_budget()
    fig_intervals()
    fig_alpha()
    fig_physics()
    fig_multiseed()


if __name__ == "__main__":
    main()
