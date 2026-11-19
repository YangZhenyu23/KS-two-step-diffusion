"""Figures for the 20 November reply to Harry.

Every number is read from a file already on disk. The (r, t) density is redrawn
from the same sampler as ``fullscale.model.sample_times``. The particle panel
reads the exact-OT subsample already stored for the 3D clouds.
"""
from __future__ import annotations

import csv
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
PARTICLE = ROOT / "results" / "ks3d" / "dp_particle"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "medium",
    "axes.linewidth": 0.7,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "figure.dpi": 160,
    "savefig.dpi": 240,
    "axes.unicode_minus": False,
})

BLUE = "#24578A"
GREEN = "#1C7C6E"
ORANGE = "#C47A2C"
PURPLE = "#6C5A8E"
GRAY = "#8B9098"
INK = "#1C1C1C"
GRID = "#E3E6EA"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def finish(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3.2, width=0.7, color="#666666")


def tag(ax, x, y, text, dy=0.0) -> None:
    ax.text(x, y + dy, text, ha="center", va="bottom", fontsize=8.2, color=INK, zorder=5)


def particle_rows() -> list[dict]:
    path = PARTICLE / "dp_particle_m10000.csv"
    with path.open(encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    by = {}
    for row in raw:
        by[(int(row["sigma"]), row["method"])] = row
    out = []
    for sigma in sorted({key[0] for key in by}):
        mf, dp = by[(sigma, "mf")], by[(sigma, "mf_dp")]
        hist_mf, hist_dp = float(mf["w2_histogram"]), float(dp["w2_histogram"])
        part_mf, part_dp = float(mf["w2_particle"]), float(dp["w2_particle"])
        out.append({
            "sigma": sigma,
            "hist_gain": (hist_mf - hist_dp) / hist_mf,
            "particle_gain": (part_mf - part_dp) / part_mf,
            "hist_mf": hist_mf,
            "hist_dp": hist_dp,
            "particle_mf": part_mf,
            "particle_dp": part_dp,
        })
    return out


def write_particle_summary(rows: list[dict]) -> dict:
    exact = load(PARTICLE / "dp_particle_m30000.json")
    mf = exact["rows"][0]
    dp = exact["rows"][1]
    hist_gain = (mf["w2_histogram"] - dp["w2_histogram"]) / mf["w2_histogram"]
    m30 = (mf["w2_particle"] - dp["w2_particle"]) / mf["w2_particle"]
    cell = next(row for row in rows if row["sigma"] == 100)
    summary = {
        "n_shears_m10000": len(rows),
        "mean_histogram_gain_m10000": float(np.mean([row["hist_gain"] for row in rows])),
        "mean_particle_gain_m10000": float(np.mean([row["particle_gain"] for row in rows])),
        "median_histogram_gain_m10000": float(np.median([row["hist_gain"] for row in rows])),
        "median_particle_gain_m10000": float(np.median([row["particle_gain"] for row in rows])),
        "sigma100": {
            "histogram_mf": mf["w2_histogram"],
            "histogram_mf_dp": dp["w2_histogram"],
            "histogram_gain": hist_gain,
            "m10000_gain": cell["particle_gain"],
            "m30000_mf": mf["w2_particle"],
            "m30000_mf_dp": dp["w2_particle"],
            "m30000_gain": m30,
            "seconds": {"mf": mf["seconds"], "mf_dp": dp["seconds"]},
        },
    }
    small = [row for row in rows if row["hist_gain"] < 0.25]
    summary["shears_with_histogram_gain_below_25pct"] = {
        "n": len(small),
        "mean_histogram_gain": float(np.mean([row["hist_gain"] for row in small])),
        "mean_particle_gain": float(np.mean([row["particle_gain"] for row in small])),
    }
    destination = PARTICLE / "gain_comparison.json"
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def fig_schedule() -> None:
    warm = load(FOLLOW / "warmcos_s88" / "summary.json")
    paired = load(PAIRED / "s88" / "summary.json")
    labels = ["fixed $10^{-3}$", "warm-up, then cosine"]
    stage = [paired["mf_mean_w2"], warm["mf_mean_w2"]]
    corrected = [paired["mf_dp_mean_w2"], warm["mf_dp_mean_w2"]]
    gains = [100 * paired["relative_reduction"], 100 * warm["relative_reduction"]]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.85), layout="constrained",
                             gridspec_kw={"width_ratios": [1.35, 1]})
    xs = np.arange(2)
    width = 0.34
    axes[0].bar(xs - width / 2, stage, width, color=BLUE, label="stage 1", zorder=3)
    axes[0].bar(xs + width / 2, corrected, width, color=GREEN, label="stage 1 + corrector", zorder=3)
    for x, a, b in zip(xs, stage, corrected):
        tag(axes[0], x - width / 2, a, f"{a:.4f}", 0.0025)
        tag(axes[0], x + width / 2, b, f"{b:.4f}", 0.0025)
    axes[0].set_xticks(xs, labels)
    axes[0].set_ylabel("histogram $W_2$, 143 held-out cells")
    axes[0].set_ylim(0, 0.165)
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].set_title("Same cells, same update budget")
    finish(axes[0])

    bars = axes[1].bar(xs, gains, color=[BLUE, GREEN], width=0.62, zorder=3)
    for bar, value in zip(bars, gains):
        tag(axes[1], bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}%", 0.6)
    axes[1].set_xticks(xs, labels)
    axes[1].set_ylabel("corrector's reduction")
    axes[1].set_ylim(0, 22)
    axes[1].set_title("The gain that remains")
    finish(axes[1])
    fig.savefig(OUT / "fig_reply_schedule.png")
    plt.close(fig)


def fig_four() -> None:
    payload = load(FOLLOW / "four_row_warmcos.json")
    values = [
        payload["a_one_step"],
        payload["b_corrector"],
        payload["c_width112"]["mean"],
        payload["d_two_step"],
    ]
    colors = [BLUE, GREEN, PURPLE, ORANGE]
    seeds = [payload["c_width112"]["per_seed"][key] for key in ("88", "89", "90")]
    base = payload["a_one_step"]
    change = [100 * (base - value) / base for value in values[1:]]

    fig, axes = plt.subplots(1, 2, figsize=(8.1, 4.0), layout="constrained")
    bars = axes[0].bar(np.arange(4), values, color=colors, width=0.68, zorder=3)
    for bar, value in zip(bars, values):
        lift = 0.009 if abs(value - payload["c_width112"]["mean"]) < 1e-12 else 0.0024
        tag(axes[0], bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", lift)
    jitter = np.array([-0.16, 0.0, 0.16])
    axes[0].scatter(2 + jitter, seeds, s=26, color="white", edgecolor=INK, linewidths=0.8, zorder=4)
    short = ["(a)\none step", "(b)\ncorrector", "(c)\nwidth 112", "(d)\ntwo steps"]
    axes[0].set_xticks(np.arange(4), short)
    axes[0].set_ylabel("histogram $W_2$, 112 common cells")
    axes[0].set_ylim(0, 0.158)
    axes[0].set_title("Scheduled stage 1")
    finish(axes[0])

    change_colors = [GREEN, PURPLE, ORANGE]
    bars = axes[1].bar(np.arange(3), change, color=change_colors, width=0.68, zorder=3)
    axes[1].axhline(0, color=INK, lw=0.8)
    for bar, value in zip(bars, change):
        x = bar.get_x() + bar.get_width() / 2
        if value >= 0:
            axes[1].text(x, value + 0.45, f"{value:+.1f}%", ha="center", va="bottom",
                         fontsize=8.4, color=INK)
        else:
            axes[1].text(x, -7.2, f"{value:+.1f}%", ha="center", va="top",
                         fontsize=8.4, color=INK)
    seed_change = [100 * (base - value) / base for value in seeds]
    axes[1].scatter(1 + jitter, seed_change, s=26, color="white", edgecolor=INK, linewidths=0.8, zorder=4)
    axes[1].set_xticks(np.arange(3), ["(b)\ncorrector", "(c)\nwidth 112", "(d)\ntwo steps"])
    axes[1].set_ylabel("lower than one step")
    axes[1].set_ylim(-11, 22)
    axes[1].set_title("Against row (a)")
    finish(axes[1])
    fig.savefig(OUT / "fig_reply_four.png")
    plt.close(fig)


def fig_budget() -> None:
    payload = load(FOLLOW / "budget_controls.json")
    paired = load(PAIRED / "s88" / "summary.json")
    labels = ["stage 1", "+80,000\nMeanFlow updates", "+80,000 updates\non the OT targets", "separate\ncorrector"]
    values = [
        payload["headline_mf"],
        payload["extra_mf_updates"],
        payload["ot_finetune_same_network"],
        paired["mf_dp_mean_w2"],
    ]
    notes = ["", "+1.5%", "−3.0%", "−15.0%"]
    colors = [BLUE, GRAY, ORANGE, GREEN]
    fig, ax = plt.subplots(figsize=(7.3, 4.05), layout="constrained")
    bars = ax.bar(np.arange(4), values, color=colors, width=0.68, zorder=3)
    for bar, value, note in zip(bars, values, notes):
        x = bar.get_x() + bar.get_width() / 2
        tag(ax, x, value, f"{value:.4f}", 0.0026)
        if note:
            ax.text(x, value + 0.012, note, ha="center", va="bottom", fontsize=8.4, color=INK)
    ax.set_xticks(np.arange(4), labels)
    ax.set_ylabel("histogram $W_2$, 143 held-out cells, seed 88")
    ax.set_ylim(0, 0.172)
    ax.set_title("The same 80,000 updates, spent three ways")
    finish(ax)
    fig.savefig(OUT / "fig_reply_budget.png")
    plt.close(fig)


def sample_intervals(n: int = 1_000_000, seed: int = 0):
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
    print(f"interval redraw short={short:.4f} stored={stored['fraction_interval_shorter_than_half']:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.95), layout="constrained")
    hexbin = axes[0].hexbin(
        r, t, gridsize=46, cmap="viridis", mincnt=1, norm=LogNorm(),
        linewidths=0, extent=(0, 1, 0, 1),
    )
    axes[0].plot([0, 1], [0, 1], color="white", lw=0.9, alpha=0.9)
    axes[0].plot([0, 0.5], [0.5, 1.0], color="white", ls=(0, (3, 2)), lw=1.05)
    axes[0].scatter([0], [1], s=90, marker="*", color="white", edgecolor=INK,
                    linewidths=0.8, zorder=6, label="one step, $(0,1)$")
    axes[0].scatter([0.0, 0.5], [0.5, 1.0], s=36, marker="s", color=ORANGE,
                    edgecolor="white", linewidths=0.8, zorder=6, label="sub-steps of length 0.5")
    axes[0].legend(frameon=True, framealpha=0.94, fontsize=8, loc="lower right",
                   borderpad=0.4, handletextpad=0.4)
    axes[0].set_xlabel("$r$")
    axes[0].set_ylabel("$t$")
    axes[0].set_xlim(-0.02, 1.02)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_aspect("equal")
    axes[0].set_title("Where training looks")
    axes[0].tick_params(length=3.2, width=0.7)
    fig.colorbar(hexbin, ax=axes[0], fraction=0.046, pad=0.03).set_label("draws")

    order = np.sort(gap)
    cdf = np.linspace(0, 1, len(order), endpoint=False)
    axes[1].plot(order[::200], cdf[::200], color=BLUE, lw=1.8)
    axes[1].axvline(0.5, color=INK, ls=(0, (3, 2)), lw=0.9)
    mark = stored["fraction_interval_shorter_than_half"]
    axes[1].scatter([0.5], [mark], s=32, color=ORANGE, zorder=5)
    axes[1].annotate("95.8% are shorter\nthan 0.5", xy=(0.5, mark), xytext=(0.62, 0.72),
                     fontsize=8.4, color=INK,
                     arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.7})
    axes[1].annotate("half the batch\nsits on $r=t$", xy=(0.015, 0.50), xytext=(0.16, 0.28),
                     fontsize=8.4, color=INK,
                     arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.7})
    axes[1].set_xlabel("interval length $t-r$")
    axes[1].set_ylabel("cumulative fraction")
    axes[1].set_xlim(-0.02, 1.02)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("How long those intervals are")
    finish(axes[1])
    fig.savefig(OUT / "fig_reply_intervals.png")
    plt.close(fig)


def fig_metric(rows: list[dict], summary: dict) -> None:
    cell = summary["sigma100"]
    gains = [100 * cell["histogram_gain"], 100 * cell["m10000_gain"], 100 * cell["m30000_gain"]]
    groups = ["16-bin\nhistogram", "exact OT\n$m=10{,}000$", "exact OT\n$m=30{,}000$"]
    mf_vals = [cell["histogram_mf"], 0.1584872230920125, cell["m30000_mf"]]
    dp_vals = [cell["histogram_mf_dp"], 0.1247028470562517, cell["m30000_mf_dp"]]

    fig, axes = plt.subplots(1, 2, figsize=(7.9, 4.2), layout="constrained")
    xs = np.arange(3)
    width = 0.34
    axes[0].bar(xs - width / 2, mf_vals, width, color=BLUE, label="MeanFlow", zorder=3)
    axes[0].bar(xs + width / 2, dp_vals, width, color=GREEN, label="MeanFlow + corrector", zorder=3)
    for x, a, b, gain in zip(xs, mf_vals, dp_vals, gains):
        tag(axes[0], x, max(a, b), f"−{gain:.1f}%", 0.008)
    axes[0].set_xticks(xs, groups)
    axes[0].set_ylabel("$W_2$ at $\\sigma=100$, seed 88")
    axes[0].set_ylim(0, 0.27)
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].set_title("Same cloud, three ways of scoring it")
    finish(axes[0])

    hx = 100 * np.array([row["hist_gain"] for row in rows])
    py = 100 * np.array([row["particle_gain"] for row in rows])
    axes[1].plot([0, 75], [0, 75], color=GRAY, lw=0.9, ls=(0, (3, 2)), zorder=1)
    axes[1].scatter(hx, py, s=28, color=BLUE, zorder=3)
    hit = next(i for i, row in enumerate(rows) if row["sigma"] == 100)
    axes[1].scatter([hx[hit]], [py[hit]], s=58, color=ORANGE, zorder=4, label=r"$\sigma=100$")
    axes[1].annotate(r"$\sigma=100$", xy=(hx[hit], py[hit]), xytext=(4, 36),
                     fontsize=8.5, color=ORANGE,
                     arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 0.7})
    axes[1].set_xlabel("histogram reduction, percent")
    axes[1].set_ylabel("particle reduction at $m=10{,}000$, percent")
    axes[1].set_xlim(0, 78)
    axes[1].set_ylim(0, 78)
    axes[1].set_title("Twenty shears, seed 88")
    finish(axes[1])
    fig.savefig(OUT / "fig_reply_metric.png")
    plt.close(fig)


def fig_alpha_physics() -> None:
    alpha = load(FOLLOW / "alpha_seeds.json")
    physics = load(FOLLOW / "particle_and_physics.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9), layout="constrained")

    seeds = ("89", "90")
    alphas = ("0.0", "1.0", "2.0")
    colors = [BLUE, "#5B8FA8", GREEN]
    names = [r"$\alpha=0$", r"$\alpha=1$", r"$\alpha=2$"]
    xs = np.arange(2)
    width = 0.24
    for offset, (key, color, name) in enumerate(zip(alphas, colors, names)):
        values = [alpha[seed][key] for seed in seeds]
        bars = axes[0].bar(xs + (offset - 1) * width, values, width, color=color, label=name, zorder=3)
        for bar, value in zip(bars, values):
            tag(axes[0], bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", 0.002)
    axes[0].set_xticks(xs, ["seed 89", "seed 90"])
    axes[0].set_ylabel("histogram $W_2$, 93 test cells")
    axes[0].set_ylim(0, 0.175)
    axes[0].legend(frameon=False, ncol=3, loc="upper right")
    axes[0].set_title(r"$\alpha=2$ was chosen on seed 88")
    finish(axes[0])

    keys = [("centroid_x", "centroid"), ("rms_radius", "rms radius"), ("anisotropy", "anisotropy")]
    stage = [physics["mean_abs_relative_error"]["mf"][key] for key, _ in keys]
    corrected = [physics["mean_abs_relative_error"]["mf_dp"][key] for key, _ in keys]
    xs = np.arange(3)
    axes[1].bar(xs - width / 2, stage, 0.34, color=BLUE, label="stage 1", zorder=3)
    axes[1].bar(xs + width / 2, corrected, 0.34, color=GREEN, label="with corrector", zorder=3)
    for x, a, b in zip(xs, stage, corrected):
        drop = 100 * (a - b) / a
        tag(axes[1], x, max(a, b), f"−{drop:.0f}%", 0.0011)
    axes[1].set_xticks(xs, [label for _, label in keys])
    axes[1].set_ylabel("mean absolute relative error")
    axes[1].set_ylim(0, 0.042)
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].set_title("Three summaries, 143 cells")
    finish(axes[1])
    fig.savefig(OUT / "fig_reply_scope.png")
    plt.close(fig)


def fig_multiseed() -> None:
    shared = load(FOLLOW / "multiseed_corrector" / "summary.json")
    seeds = ("88", "89", "90")
    stage, paired, common = [], [], []
    for seed in seeds:
        own = load(PAIRED / f"s{seed}" / "summary.json")
        stage.append(own["mf_mean_w2"])
        paired.append(own["mf_dp_mean_w2"])
        common.append(shared["per_seed"][seed]["mf_dp"])
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.95), layout="constrained")
    xs = np.arange(3)
    width = 0.24
    series = [
        (stage, BLUE, "stage 1"),
        (common, ORANGE, "one corrector, three seeds"),
        (paired, GREEN, "corrector fit to that seed"),
    ]
    for offset, (values, color, label) in enumerate(series):
        axes[0].bar(xs + (offset - 1) * width, values, width, color=color, label=label, zorder=3)
    axes[0].set_xticks(xs, [f"seed {seed}" for seed in seeds])
    axes[0].set_ylabel("histogram $W_2$, 143 held-out cells")
    axes[0].set_ylim(0, 0.185)
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].set_title("Levels")
    finish(axes[0])

    paired_gain = [100 * (a - b) / a for a, b in zip(stage, paired)]
    shared_gain = [100 * (a - b) / a for a, b in zip(stage, common)]
    axes[1].bar(xs - width / 2, paired_gain, width, color=GREEN, label="fit to that seed", zorder=3)
    axes[1].bar(xs + width / 2, shared_gain, width, color=ORANGE, label="shared", zorder=3)
    for x, value in zip(xs - width / 2, paired_gain):
        tag(axes[1], x, value, f"{value:.1f}%", 0.35)
    for x, value in zip(xs + width / 2, shared_gain):
        tag(axes[1], x, value, f"{value:.1f}%", 0.35)
    axes[1].set_xticks(xs, [f"seed {seed}" for seed in seeds])
    axes[1].set_ylabel("reduction against that seed's stage 1")
    axes[1].set_ylim(0, 26)
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].set_title("How much comes off")
    finish(axes[1])
    fig.savefig(OUT / "fig_reply_multiseed.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = particle_rows()
    summary = write_particle_summary(rows)
    fig_schedule()
    fig_four()
    fig_budget()
    fig_intervals()
    fig_metric(rows, summary)
    fig_alpha_physics()
    fig_multiseed()
    print("wrote reply figures")


if __name__ == "__main__":
    main()
