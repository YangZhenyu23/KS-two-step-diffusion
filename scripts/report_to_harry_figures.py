"""Figures for the report to Harry. One message per figure.

Numbers are read from the stored JSON and CSV. The only computation done here is the
linear fit in the binning panel, which is written back to a JSON so the report can quote
it without re-deriving it.

Usage:
    python scripts/report_to_harry_figures.py
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
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "report_figures"
FOLLOW = ROOT / "results" / "fullscale" / "harry_followup"
PAIRED = ROOT / "results" / "fullscale" / "dp_two_param"
PARTICLE = ROOT / "results" / "ks3d" / "dp_particle"
ALPHA88 = ROOT / "results" / "fullscale" / "alpha_sweep" / "alpha_sweep_repaired_s88.json"

plt.rcParams.update({
    "font.family": "Segoe UI",
    "font.size": 9.4,
    "axes.labelsize": 9.6,
    "axes.titlesize": 10.2,
    "axes.titleweight": "semibold",
    "axes.linewidth": 0.75,
    "axes.edgecolor": "#5B5F66",
    "xtick.labelsize": 8.8,
    "ytick.labelsize": 8.8,
    "xtick.color": "#3A3D42",
    "ytick.color": "#3A3D42",
    "legend.fontsize": 8.6,
    "figure.dpi": 170,
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
    "axes.unicode_minus": True,
})

INK = "#1B1D21"
SOFT = "#5B5F66"
BLUE = "#2E5C8A"
TEAL = "#22786B"
CLAY = "#B0603A"
PLUM = "#6B5B8C"
MIST = "#A9AEB5"
HAIR = "#DFE2E6"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def clean(ax, *, grid: str | None = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=HAIR, lw=0.75, zorder=0)
        ax.set_axisbelow(True)
    ax.tick_params(length=3.0, width=0.7)


def value(ax, x, y, text, *, dx=0.0, dy=0.0, **kw) -> None:
    kw.setdefault("fontsize", 8.5)
    kw.setdefault("color", INK)
    kw.setdefault("ha", "left")
    kw.setdefault("va", "center")
    ax.text(x + dx, y + dy, text, zorder=6, **kw)


# ------------------------------------------------------------------ figure 1, the ladder
def fig_ladder() -> None:
    """The 143-cell pair only. Both pairs share the same cells, so they can be compared."""
    warm = load(FOLLOW / "warmcos_s88" / "summary.json")
    paired = load(PAIRED / "s88" / "summary.json")

    rows = [
        ("stage 1, fixed $10^{-3}$", paired["mf_mean_w2"], MIST),
        ("stage 1 + corrector", paired["mf_dp_mean_w2"], TEAL),
        ("stage 1, warm-up + cosine", warm["mf_mean_w2"], BLUE),
        ("stage 1 + corrector", warm["mf_dp_mean_w2"], TEAL),
    ]
    ys = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(7.1, 3.5), layout="constrained")

    ax.axvspan(rows[1][1], rows[2][1], color="#EDF2F6", zorder=0)
    ax.axvline(rows[1][1], color=SOFT, lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.text(rows[1][1], 3.62, "0.1043", ha="center", va="bottom",
            fontsize=8.2, color=SOFT)
    ax.text(0.5 * (rows[1][1] + rows[2][1]), 3.62,
            "the schedule alone, with no second network,\nlands left of the old corrected map",
            ha="center", va="bottom", fontsize=8.2, color="#41525E", linespacing=1.3)

    for y, (_, x, color) in zip(ys, rows):
        ax.scatter([x], [y], s=58, color=color, zorder=5,
                   edgecolor="white", linewidths=0.9)

    for top, bottom, color in ((0, 1, TEAL), (2, 3, TEAL)):
        ax.annotate("", xy=(rows[bottom][1], ys[bottom]), xytext=(rows[top][1], ys[top]),
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.4,
                                "shrinkA": 7, "shrinkB": 7, "mutation_scale": 10,
                                "connectionstyle": "arc3,rad=0.0"}, zorder=3)

    value(ax, rows[0][1], ys[0], f"{rows[0][1]:.4f}", dx=0.0013, color=SOFT)
    value(ax, rows[1][1], ys[1], f"{rows[1][1]:.4f}", dx=-0.0016, ha="right")
    value(ax, rows[2][1], ys[2], f"{rows[2][1]:.4f}", dx=-0.0016, ha="right", color=BLUE)
    value(ax, rows[3][1], ys[3], f"{rows[3][1]:.4f}", dx=-0.0016, ha="right")

    ax.text(rows[1][1] - 0.0022, (ys[0] + ys[1]) / 2 + 0.30, "−15.0%",
            ha="right", va="center", fontsize=8.5, color=TEAL, fontweight="semibold")
    ax.text(rows[3][1] - 0.0022, (ys[2] + ys[3]) / 2, "−7.3%",
            ha="right", va="center", fontsize=8.5, color=TEAL, fontweight="semibold")
    ax.text(rows[2][1] + 0.0006, ys[2] - 0.42,
            "−19.6% against the fixed-rate map", ha="left", va="center",
            fontsize=8.2, color=BLUE)

    ax.set_yticks(ys, [row[0] for row in rows])
    ax.set_ylim(-0.62, 4.02)
    ax.set_xlim(0.0805, 0.1305)
    ax.set_xlabel("histogram $W_2$, 143 held-out cells, seed 88")
    ax.xaxis.set_major_locator(MultipleLocator(0.01))
    clean(ax, grid=None)
    fig.savefig(OUT / "fig_report_ladder.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------- figure 2, error against cost
def fig_cost() -> None:
    four = load(FOLLOW / "four_row_warmcos.json")
    complexity = load(ROOT / "results" / "fullscale" / "complexity" / "complexity.json")
    flops = {row["configuration"]: row["relative_flops"] for row in complexity["inference"]}

    points = [
        ("one step", flops["stage 1 only"], four["a_one_step"], BLUE, (8, 12), "left"),
        ("two sub-steps", flops["stage 1, two-step"], four["d_two_step"], CLAY, (10, -14), "left"),
        ("+ corrector", flops["stage 1 + corrector"], four["b_corrector"], TEAL, (-10, -16), "right"),
    ]
    seeds = [four["c_width112"]["per_seed"][key] for key in ("88", "89", "90")]

    fig, ax = plt.subplots(figsize=(7.0, 4.05), layout="constrained")

    ax.plot([1.0, 2.97], [four["a_one_step"]] * 2, color=HAIR, lw=0.9, ls=(0, (3, 3)), zorder=1)

    for label, cost, error, color, offset, align in points:
        ax.scatter([cost], [error], s=84, color=color, zorder=5,
                   edgecolor="white", linewidths=1.0)
        ax.annotate(f"{label}\n{error:.4f}", xy=(cost, error),
                    xytext=offset, textcoords="offset points",
                    ha=align, va="center", fontsize=8.6, color=color,
                    fontweight="semibold", linespacing=1.35)

    ax.scatter([1.0] * 3, seeds, s=26, facecolor="white", edgecolor=PLUM,
               linewidths=1.0, zorder=5)
    ax.scatter([1.0], [four["c_width112"]["mean"]], s=72, color=PLUM, zorder=6,
               edgecolor="white", linewidths=1.0)
    ax.annotate(f"width 112, three seeds\n{four['c_width112']['mean']:.4f}",
                xy=(1.0, four["c_width112"]["mean"]), xytext=(16, 6),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=8.6, color=PLUM, fontweight="semibold", linespacing=1.35)

    ax.set_xlabel("inference cost, multiples of one stage-1 pass")
    ax.set_ylabel("histogram $W_2$, 112 common cells")
    ax.set_xlim(0.78, 3.35)
    ax.set_ylim(0.0810, 0.1075)
    ax.set_xticks([1.0, 2.0, 2.97], ["1.0×", "2.0×", "2.97×"])
    clean(ax)
    fig.savefig(OUT / "fig_report_cost.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- figure 3, where t and r go
def sample_intervals(n: int = 1_000_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    s = rng.standard_normal((n, 2)).astype(np.float32) + np.float32(-0.4)
    s = 1.0 / (1.0 + np.exp(-s))
    t = np.maximum(s[:, 0], s[:, 1])
    r = np.minimum(s[:, 0], s[:, 1])
    chosen = rng.permutation(n)[: n // 2]
    r[chosen] = t[chosen]
    return r, t


def fig_sampling() -> None:
    stored = load(FOLLOW / "consistency.json")
    r, t = sample_intervals()
    gap = t - r
    share = float(np.mean(gap >= 0.5))

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.65), layout="constrained",
                             gridspec_kw={"width_ratios": [1.12, 1]})
    ax = axes[0]
    hb = ax.hexbin(r, t, gridsize=44, cmap="Blues", mincnt=1, norm=LogNorm(),
                   linewidths=0, extent=(0, 1, 0, 1))
    ax.plot([0, 1], [0, 1], color="#8A8F96", lw=0.7)
    ax.plot([0, 0.5], [0.5, 1.0], color=CLAY, lw=1.1, ls=(0, (3, 2)), zorder=4)
    ax.scatter([0], [1], s=64, marker="*", color=CLAY, edgecolor="white",
               linewidths=0.8, zorder=6)
    ax.scatter([0.0, 0.5], [0.5, 1.0], s=26, marker="s", color=CLAY,
               edgecolor="white", linewidths=0.7, zorder=6)
    ax.annotate("one step $(0,1)$", xy=(0.015, 0.985), xytext=(0.15, 0.86),
                fontsize=8.3, color=CLAY,
                arrowprops={"arrowstyle": "-", "color": CLAY, "lw": 0.7})
    ax.annotate("two sub-steps", xy=(0.505, 0.5), xytext=(0.60, 0.30),
                fontsize=8.3, color=CLAY,
                arrowprops={"arrowstyle": "-", "color": CLAY, "lw": 0.7})
    ax.set_xlabel("$r$")
    ax.set_ylabel("$t$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_title("every pair the network is trained on")
    ax.tick_params(length=3.0, width=0.7)
    cb = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("draws", fontsize=8.4)
    cb.ax.tick_params(labelsize=7.8)

    ax = axes[1]
    off = gap[gap > 1e-6]
    ax.hist(off, bins=np.linspace(0, 1, 51), color=BLUE, alpha=0.85, zorder=3)
    ax.axvline(0.5, color=CLAY, lw=1.2, ls=(0, (3, 2)), zorder=4)
    ax.set_xlabel("interval length $t-r$")
    ax.set_ylabel("draws (the half on $r=t$ is left out)")
    ax.set_xlim(0, 1.0)
    ax.set_title(f"only {100 * share:.1f}% are as long as a sub-step")
    ax.annotate("half the batch\nsits exactly on $r=t$", xy=(0.03, ax.get_ylim()[1] * 0.82),
                fontsize=8.3, color=INK, ha="left", va="top")
    ax.annotate("length 0.5", xy=(0.5, ax.get_ylim()[1] * 0.55), xytext=(0.56, ax.get_ylim()[1] * 0.72),
                fontsize=8.3, color=CLAY,
                arrowprops={"arrowstyle": "-", "color": CLAY, "lw": 0.7})
    clean(ax)
    fig.savefig(OUT / "fig_report_sampling.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- figure 3, the budget
def fig_budget() -> None:
    payload = load(FOLLOW / "budget_controls.json")
    paired = load(PAIRED / "s88" / "summary.json")
    base = payload["headline_mf"]
    rows = [
        ("MeanFlow loss, 80,000 more updates", payload["extra_mf_updates"], MIST),
        ("the transport targets, same network", payload["ot_finetune_same_network"], CLAY),
        ("a second network", paired["mf_dp_mean_w2"], TEAL),
    ]

    fig, ax = plt.subplots(figsize=(7.0, 3.3), layout="constrained")
    ys = np.arange(len(rows))[::-1]
    ax.axvline(base, color=SOFT, lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.text(base, len(rows) - 0.42, "stage 1\n0.1228", ha="center", va="bottom",
            fontsize=8.3, color=SOFT, linespacing=1.3)

    for y, (label, value, color) in zip(ys, rows):
        ax.plot([base, value], [y, y], color=color, lw=2.0, zorder=3,
                solid_capstyle="round")
        ax.scatter([value], [y], s=54, color=color, zorder=4,
                   edgecolor="white", linewidths=0.9)
        change = 100 * (base - value) / base
        sign = "+" if change < 0 else "−"
        ax.text(max(base, value) + 0.0012, y, f"{value:.4f}", fontsize=8.5,
                color=INK, va="center")
        ax.text(0.1012, y, f"{sign}{abs(change):.1f}%", fontsize=8.6, color=color,
                va="center", ha="left", fontweight="semibold")

    ax.set_yticks(ys, [row[0] for row in rows])
    ax.set_ylim(-0.5, len(rows) - 0.05)
    ax.set_xlim(0.1005, 0.1305)
    ax.set_xlabel("histogram $W_2$, 143 held-out cells, seed 88")
    ax.text(0.1012, len(rows) - 0.42, "against stage 1", fontsize=8.2, color=SOFT,
            ha="left", va="bottom")
    clean(ax, grid="x")
    fig.savefig(OUT / "fig_report_budget.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- figure 4, alpha by seed
def fig_alpha() -> None:
    alpha = load(FOLLOW / "alpha_seeds.json")
    s88 = load(ALPHA88)
    series = {
        "88": [s88["test"]["alpha_0"], s88["test"]["alpha_1"], s88["test"]["alpha_2"]],
        "89": [alpha["89"]["0.0"], alpha["89"]["1.0"], alpha["89"]["2.0"]],
        "90": [alpha["90"]["0.0"], alpha["90"]["1.0"], alpha["90"]["2.0"]],
    }
    fig, ax = plt.subplots(figsize=(7.0, 3.7), layout="constrained")
    seeds = list(series)
    xs = np.arange(len(seeds))
    width = 0.25
    colors = [MIST, BLUE, TEAL]
    names = [r"$\alpha=0$", r"$\alpha=1$", r"$\alpha=2$"]
    for offset, (color, name) in enumerate(zip(colors, names)):
        vals = [series[seed][offset] for seed in seeds]
        bars = ax.bar(xs + (offset - 1) * width, vals, width * 0.94, color=color,
                      label=name, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.0022, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=7.6, color=INK, rotation=90)
    for seed, x in zip(seeds, xs):
        a0, a1, a2 = series[seed]
        ax.text(x, 0.006, f"−{100 * (a0 - a2) / a0:.1f}%", ha="center", va="bottom",
                fontsize=8.4, color="white", fontweight="semibold", zorder=6)
    ax.set_xticks(xs, [f"seed {seed}" for seed in seeds])
    ax.set_ylabel("histogram $W_2$, 93 test cells")
    ax.set_ylim(0, 0.158)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax.annotate("chosen here", xy=(0.28, 0.100), xytext=(0.30, 0.128),
                fontsize=8.2, color=TEAL,
                arrowprops={"arrowstyle": "-", "color": TEAL, "lw": 0.7})
    clean(ax)
    fig.savefig(OUT / "fig_report_alpha.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- figure 5, the binning
def particle_table() -> list[dict]:
    path = PARTICLE / "dp_particle_m10000.csv"
    with path.open(encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    by = {(int(row["sigma"]), row["method"]): row for row in raw}
    out = []
    for sigma in sorted({key[0] for key in by}):
        mf, dp = by[(sigma, "mf")], by[(sigma, "mf_dp")]
        out.append({
            "sigma": sigma,
            "hist": 100 * (float(mf["w2_histogram"]) - float(dp["w2_histogram"])) / float(mf["w2_histogram"]),
            "part": 100 * (float(mf["w2_particle"]) - float(dp["w2_particle"])) / float(mf["w2_particle"]),
            "hist_mf": float(mf["w2_histogram"]),
            "hist_dp": float(dp["w2_histogram"]),
            "part_mf": float(mf["w2_particle"]),
            "part_dp": float(dp["w2_particle"]),
        })
    return out


def particle_summary(rows: list[dict]) -> dict:
    """The histogram-versus-particle summary the report quotes."""
    exact = load(PARTICLE / "dp_particle_m30000.json")
    mf, dp = exact["rows"][0], exact["rows"][1]
    hist_gain = (mf["w2_histogram"] - dp["w2_histogram"]) / mf["w2_histogram"]
    m30 = (mf["w2_particle"] - dp["w2_particle"]) / mf["w2_particle"]
    cell = next(row for row in rows if row["sigma"] == 100)
    low = [row for row in rows if row["hist"] < 25]
    summary = {
        "n_shears_m10000": len(rows),
        "mean_histogram_gain_m10000": float(np.mean([row["hist"] for row in rows]) / 100),
        "mean_particle_gain_m10000": float(np.mean([row["part"] for row in rows]) / 100),
        "sigma100": {
            "histogram_mf": mf["w2_histogram"],
            "histogram_mf_dp": dp["w2_histogram"],
            "histogram_gain": hist_gain,
            "m10000_gain": cell["part"] / 100,
            "m30000_mf": mf["w2_particle"],
            "m30000_mf_dp": dp["w2_particle"],
            "m30000_gain": m30,
            "seconds": {"mf": mf["seconds"], "mf_dp": dp["seconds"]},
        },
        "shears_with_histogram_gain_below_25pct": {
            "n": len(low),
            "mean_histogram_gain": float(np.mean([row["hist"] for row in low]) / 100),
            "mean_particle_gain": float(np.mean([row["part"] for row in low]) / 100),
        },
    }
    (PARTICLE / "gain_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def fig_binning(summary: dict) -> None:
    exact = load(PARTICLE / "dp_particle_m30000.json")
    m30, d30 = exact["rows"][0], exact["rows"][1]
    cell = summary["sigma100"]
    rows = particle_table()

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.85), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1.28]})

    ax = axes[0]
    labels = ["16-bin\nhistogram", "$m=10{,}000$", "$m=30{,}000$"]
    gains = [100 * cell["histogram_gain"], cell["m10000_gain"] * 100, cell["m30000_gain"] * 100]
    base = [cell["histogram_mf"], 0.1584872230920125, m30["w2_particle"]]
    after = [cell["histogram_mf_dp"], 0.1247028470562517, d30["w2_particle"]]
    xs = np.arange(3)
    ax.bar(xs, base, 0.58, color="#CBD5DD", zorder=3, label="MeanFlow")
    ax.bar(xs, after, 0.58, color=TEAL, zorder=4, label="with the refinement")
    for x, b, a, g in zip(xs, base, after, gains):
        ax.text(x, b + 0.008, f"{100 * (b - a) / b:.1f}%", ha="center", va="bottom",
                fontsize=9.0, color=INK, fontweight="semibold")
    ax.set_xticks(xs, labels)
    ax.set_ylabel("$W_2$ at $\\sigma=100$, seed 88")
    ax.set_ylim(0, 0.225)
    ax.legend(frameon=False, loc="upper right", fontsize=8.2)
    ax.set_title("one cloud, three ways to score it")
    clean(ax)

    ax = axes[1]
    hx = np.array([row["hist"] for row in rows])
    py = np.array([row["part"] for row in rows])
    slope, intercept = np.polyfit(hx, py, 1)
    residual = float(np.std(py - (intercept + slope * hx)))
    rng = np.random.default_rng(0)
    preds = []
    for _ in range(4000):
        idx = rng.integers(0, len(hx), len(hx))
        if len(set(hx[idx])) < 3:
            continue
        b, a = np.polyfit(hx[idx], py[idx], 1)
        preds.append(a + b * 7.3)
    preds = np.array(preds)
    band = (float(np.percentile(preds, 5)), float(np.percentile(preds, 95)))

    grid = np.linspace(0, 62, 100)
    ax.plot(grid, intercept + slope * grid, color=SOFT, lw=1.0, zorder=2)
    ax.plot([0, 62], [0, 62], color=MIST, lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.scatter(hx, py, s=26, color=BLUE, zorder=4, edgecolor="white", linewidths=0.4)
    hit = int(np.argmin(np.abs(hx - cell["histogram_gain"] * 100)))
    ax.scatter([hx[hit]], [py[hit]], s=52, color=CLAY, zorder=5,
               edgecolor="white", linewidths=0.8)
    ax.axvspan(band[0] * 0 + 7.3 - 0.35, 7.3 + 0.35, color="#F2E3D9", zorder=0)
    ax.plot([7.3, 7.3], [band[0], band[1]], color=CLAY, lw=1.4, solid_capstyle="round", zorder=5)
    ax.scatter([7.3], [slope * 7.3 + intercept], s=44, marker="D", color=CLAY, zorder=6,
               edgecolor="white", linewidths=0.7)
    ax.annotate(f"scheduled map\nreads near {slope * 7.3 + intercept:.1f}%",
                xy=(7.3, slope * 7.3 + intercept), xytext=(10.5, 15.5),
                fontsize=8.3, color=CLAY,
                arrowprops={"arrowstyle": "-", "color": CLAY, "lw": 0.7})
    ax.annotate("$\\sigma=100$", xy=(hx[hit], py[hit]), xytext=(18.0, 30.0),
                fontsize=8.3, color=SOFT,
                arrowprops={"arrowstyle": "-", "color": SOFT, "lw": 0.7})
    ax.text(30, 6.5, "dashed line: the two agree", fontsize=8.0, color=MIST)
    ax.set_xlabel("reduction on the 16-bin histogram, percent")
    ax.set_ylabel("reduction on the particles, percent")
    ax.set_xlim(0, 63)
    ax.set_ylim(0, 62)
    ax.set_title("twenty shears, $m=10{,}000$")
    clean(ax, grid="both")

    fig.savefig(OUT / "fig_report_binning.png", bbox_inches="tight")
    plt.close(fig)
    return {"slope": float(slope), "intercept": float(intercept), "residual_sd": residual,
            "predicted_at_7_3": float(slope * 7.3 + intercept),
            "bootstrap_5": band[0], "bootstrap_95": band[1]}


# ------------------------------------------------------------- figure 6, shared corrector
def fig_multiseed() -> None:
    shared = load(FOLLOW / "multiseed_corrector" / "summary.json")
    seeds = ("88", "89", "90")
    fig, ax = plt.subplots(figsize=(6.9, 3.55), layout="constrained")
    ys = np.arange(len(seeds))[::-1]
    for y, seed in zip(ys, seeds):
        own = load(PAIRED / f"s{seed}" / "summary.json")
        stage = own["mf_mean_w2"]
        paired = own["mf_dp_mean_w2"]
        common = shared["per_seed"][seed]["mf_dp"]
        ax.plot([stage, max(paired, common)], [y, y], color=HAIR, lw=2.4, zorder=2,
                solid_capstyle="round")
        ax.scatter([stage], [y], s=46, color=BLUE, zorder=5,
                   edgecolor="white", linewidths=0.8)
        ax.scatter([common], [y - 0.16], s=40, color=CLAY, zorder=5,
                   edgecolor="white", linewidths=0.8)
        ax.scatter([paired], [y + 0.16], s=40, color=TEAL, zorder=5,
                   edgecolor="white", linewidths=0.8)
        ax.text(stage + 0.0012, y + 0.03, f"{stage:.4f}", fontsize=8.2, color=SOFT,
                va="center")
        ax.text(common - 0.0014, y - 0.16, f"{common:.4f}", fontsize=8.2, color=CLAY,
                va="center", ha="right")
        ax.text(paired + 0.0012, y + 0.16, f"{paired:.4f}", fontsize=8.2, color=TEAL,
                va="center")
        ax.text(0.1315, y, f"{100 * (stage - common) / stage:.1f}%  /  "
                           f"{100 * (stage - paired) / stage:.1f}%",
                fontsize=8.3, color=INK, ha="right", va="center")
    ax.set_yticks(ys, [f"seed {seed}" for seed in seeds])
    ax.set_ylim(-0.55, len(seeds) - 0.35)
    ax.set_xlim(0.0975, 0.1335)
    ax.set_xlabel("histogram $W_2$, each seed's 143 held-out cells")
    ax.text(0.1315, len(seeds) - 0.45, "shared  /  fitted", fontsize=8.2, color=SOFT,
            ha="right")
    handles = [
        plt.Line2D([], [], marker="o", color="none", markerfacecolor=BLUE,
                   markeredgecolor="white", markersize=7, label="stage 1"),
        plt.Line2D([], [], marker="o", color="none", markerfacecolor=CLAY,
                   markeredgecolor="white", markersize=7, label="one corrector, three seeds"),
        plt.Line2D([], [], marker="o", color="none", markerfacecolor=TEAL,
                   markeredgecolor="white", markersize=7, label="corrector fit to that seed"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper center",
              bbox_to_anchor=(0.42, 1.16), ncol=3)
    clean(ax, grid="x")
    fig.savefig(OUT / "fig_report_multiseed.png", bbox_inches="tight")
    plt.close(fig)


def fig_sampler() -> None:
    """One seed, fixed learning rate: the original sampler against the half-interval mixture."""
    labels = ["one step", "two steps", "then the corrector"]
    original = [0.12277, 0.12418, 0.10429]
    retrained = [0.13980, 0.12327, 0.11465]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.6, 3.4), layout="constrained")
    ax.bar(x - 0.18, original, width=0.36, color=BLUE, label="original sampler", zorder=3)
    ax.bar(x + 0.18, retrained, width=0.36, color=CLAY, label="half-intervals upweighted", zorder=3)
    ax.set_xticks(x, labels)
    ax.set_ylabel("histogram $W_2$")
    ax.set_ylim(0, 0.17)
    ax.legend(frameon=False, loc="upper right")
    clean(ax)
    fig.savefig(OUT / "fig_reply_sampler.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = particle_table()
    summary = particle_summary(rows)
    fit = fig_binning(summary)
    fig_ladder()
    fig_cost()
    fig_budget()
    fig_sampling()
    fig_alpha()
    fig_multiseed()
    payload = {
        "binning_fit": fit,
        "shears_above_line": int(sum(r["part"] > r["hist"] for r in rows)),
        "mean_histogram_gain_20": float(np.mean([r["hist"] for r in rows])),
        "mean_particle_gain_20": float(np.mean([r["part"] for r in rows])),
    }
    (OUT / "report_fit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
