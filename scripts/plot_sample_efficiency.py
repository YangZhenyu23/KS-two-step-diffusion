"""Plots for the three sample-efficiency experiments.

Panels:
  1. **Learning curve** -- test W2 versus number of training cells with the
     per-size spread drawn as error bars, plus the noise floor.
  2. **LHS versus random** -- the two 40-cell designs on the parameter plane,
     annotated with their test errors and spacing statistics.
  3. **Error map** -- test error over (sigma, mu) for the full-sample model.

Usage:
    python scripts/plot_sample_efficiency.py
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

from fullscale import grid as G  # noqa: E402

OUT = G.RESULTS / "sample_efficiency"


def design_points(meta_path: Path):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    xs, ys = [], []
    for key in meta["cells"]:
        i, j = int(key[1:3]), int(key[5:7])
        xs.append(G.SIGMA_GRID[i])
        ys.append(G.MU_GRID[j])
    return xs, ys


def main() -> int:
    curve_path = OUT / "learning_curve.json"
    if not curve_path.is_file():
        raise SystemExit("run scripts/sample_efficiency.py first")

    curve = json.loads(curve_path.read_text(encoding="utf-8"))["results"]
    seeds_path = OUT / "learning_curve_seeds.json"
    lhs_path = OUT / "lhs_vs_random.json"
    full_path = OUT / "full_control.json"

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.1), layout="constrained")

    # ---- panel 1: learning curve -----------------------------------------
    ax = axes[0]
    n = [r["n_cells"] for r in curve]
    test = [r["test_mean_w2"] for r in curve]
    ax.plot(n, test, "o-", color="#2864B4", lw=1.8, ms=6, label="single design")
    if seeds_path.is_file():
        payload = json.loads(seeds_path.read_text(encoding="utf-8"))
        summary = payload["summary"]
        xs, means, sds = [], [], []
        for key in sorted(summary, key=lambda k: int(k)):
            row = summary[key]
            xs.append(row["n_cells"])
            means.append(row["mean"])
            sds.append(row["sd"])
        ax.errorbar(xs, means, yerr=sds, fmt="s--", color="#168173", lw=1.6, ms=6,
                    capsize=4, label="mean of several designs ± sd")
    ax.plot(n, [r["training_fit_w2"] for r in curve], "^:", color="#D37B2F", lw=1.3,
            ms=5, label="training fit")
    ax.axhline(G.FLOOR_16, color="#777777", ls=":", lw=1.2, label="noise floor")
    ax.set_xlabel("training cells")
    ax.set_ylabel(r"$W_2$ (held-out test set)")
    ax.set_title("learning curve", fontsize=9.5)
    ax.grid(alpha=.25)
    ax.legend(fontsize=7, frameon=False)

    # ---- panel 2: LHS vs random ------------------------------------------
    ax = axes[1]
    seeds_path2 = OUT / "lhs_vs_random_seeds.json"
    if seeds_path2.is_file():
        payload = json.loads(seeds_path2.read_text(encoding="utf-8"))
        summary = payload["summary"]
        for k, (method, colour, label) in enumerate(
                (("lhs", "#168173", "LHS (maximin)"), ("random", "#7A1F1F", "random"))):
            values = summary[method]["values"]
            xs = np.full(len(values), k, dtype=float) + np.linspace(-.07, .07, len(values))
            ax.scatter(xs, values, s=54, color=colour, alpha=.9,
                       edgecolors="white", linewidths=.7, zorder=3)
            mean = summary[method]["mean"]
            ax.hlines(mean, k - .26, k + .26, color=colour, lw=2.2, zorder=4)
            ax.annotate(f"mean {mean:.4f}\nsd {summary[method]['sd']:.4f}",
                        (k, mean), textcoords="offset points", xytext=(0, 16),
                        ha="center", fontsize=7.5, color=colour)
        ax.axhline(G.FLOOR_16, color="#777777", ls=":", lw=1.2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"LHS\n(mean min-d {summary['lhs']['mean_min_distance']:.3f})",
                            f"random\n(mean min-d {summary['random']['mean_min_distance']:.3f})"],
                           fontsize=8)
        ax.set_ylabel(r"$W_2$ on the test set")
        ratio = payload["difference"] / payload["se"] if payload["se"] else 0.0
        ax.set_title(f"LHS vs random, {payload['repeats']} designs each\n"
                     f"difference {payload['difference']:+.4f} = {ratio:+.1f} SE -> "
                     f"{payload['verdict']}", fontsize=9)
        ax.grid(alpha=.25, axis="y")
    elif lhs_path.is_file():
        payload = json.loads(lhs_path.read_text(encoding="utf-8"))
        style = {"lhs40": ("#168173", "o", "LHS"), "random40": ("#7A1F1F", "X", "random")}
        for run in payload["runs"]:
            name = run["name"]
            colour, marker, label = style[name]
            xs, ys = design_points(OUT / "runs" / name / "run_meta.json")
            ax.scatter(xs, ys, s=52, marker=marker, color=colour, alpha=.85,
                       edgecolors="white", linewidths=.6,
                       label=f"{label}: {run['test_mean_w2']:.4f}")
        ax.set_xlabel(r"$\sigma$")
        ax.set_ylabel(r"$\mu$")
        ax.set_yscale("log")
        ax.set_title("LHS vs random, 40 cells (single design)", fontsize=9.5)
        ax.grid(alpha=.25)
        ax.legend(fontsize=7, frameon=False)
    else:
        ax.text(.5, .5, "not run", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    # ---- panel 3: error map of the full-sample model ---------------------
    ax = axes[2]
    if full_path.is_file():
        record = json.loads(full_path.read_text(encoding="utf-8"))
        rows = record.get("rows")
        if rows:
            grid = np.full((len(G.MU_GRID), len(G.SIGMA_GRID)), np.nan)
            for row in rows:
                i, j = int(row["key"][1:3]), int(row["key"][5:7])
                grid[j, i] = row["w2"]
            im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
            step = 3
            ax.set_xticks(range(0, len(G.SIGMA_GRID), step))
            ax.set_xticklabels([f"{G.SIGMA_GRID[i]:.0f}"
                                for i in range(0, len(G.SIGMA_GRID), step)])
            ax.set_yticks(range(len(G.MU_GRID)))
            ax.set_yticklabels([f"{m:.3g}" for m in G.MU_GRID])
            ax.set_xlabel(r"$\sigma$")
            ax.set_ylabel(r"$\mu$")
            ax.set_title(f"test errors, full-sample model ({record['test_mean_w2']:.4f})",
                         fontsize=9.5)
            fig.colorbar(im, ax=ax, label=r"$W_2$")
        else:
            ax.axis("off")
            ax.text(.5, .6, f"full-sample model\ntest mean W2 {record['test_mean_w2']:.4f}\n"
                            f"({record['test_over_floor']:.2f}x noise floor)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.text(.5, .25, "per-cell errors not stored\nfor this run",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=7.5, color="#777777")
    else:
        ax.text(.5, .5, "not run", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    for ax in axes:
        ax.tick_params(labelsize=8)

    path = OUT / "sample_efficiency.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
