"""Paper-scale analysis: single vs two parameters, and parameter continuity.

Produces the numbers that go into the paper:

A. **Generalisation.** Mean W2 over the held-out cells (all 200 minus the 40 LHS
   training cells), split by training/held-out and reported against the
   measured statistical floor.

B. **Ablation.** The sigma-only control is trained on the mu = 1 row with the
   *same updates per cell* as the two-parameter model, then both are scored on
   the cells that neither model saw. This is the only protocol that isolates the
   value of mu from differences in budget or cell count.

C. **Where the second parameter pays.** Error as a function of distance from
   mu = 1. The single-parameter model cannot know mu at all, so its error must
   grow as mu moves away from the value it was trained at. That growth is the
   amount of physics the one-parameter design misses.

D. **Continuity.** W2 between neighbouring cells as a function of the log
   parameter distance, with the noise floor drawn in, plus the noise-corrected
   ratio that should be roughly constant for a Lipschitz family.

Usage:
    python scripts/fullscale_analysis.py
    python scripts/fullscale_analysis.py --no-plots
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fullscale import grid as G  # noqa: E402
from fullscale.lhs import latin_hypercube  # noqa: E402


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    out = []
    for row in rows:
        parsed = {}
        for key, value in row.items():
            try:
                parsed[key] = float(value)
            except (TypeError, ValueError):
                parsed[key] = value
        out.append(parsed)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    out = G.ensure("analysis")
    floor = G.FLOOR_16

    # ---- A: generalisation -------------------------------------------------
    grid_csv = G.RESULTS / "evaluate" / "lhs40" / "grid_metrics.csv"
    if not grid_csv.is_file():
        raise SystemExit("run `python -m fullscale.cli evaluate --design lhs40` first")
    rows = read_csv(grid_csv)
    train = [r for r in rows if r["split"] == "train"]
    held = [r for r in rows if r["split"] == "heldout"]
    slice_rows = [r for r in rows if abs(r["mu"] - G.BASELINE_MU) < 1e-12]
    print(f"A. generalisation  ({len(rows)} cells, {len(train)} train / {len(held)} held out)")
    print(f"   train     mean W2 {np.mean([r['w2'] for r in train]):.5f}")
    print(f"   held out  mean W2 {np.mean([r['w2'] for r in held]):.5f}")
    print(f"   all       mean W2 {np.mean([r['w2'] for r in rows]):.5f}  "
          f"({np.mean([r['w2'] for r in rows]) / floor:.2f}x floor {floor:.5f})")
    print(f"   mu=1 row  mean W2 {np.mean([r['w2'] for r in slice_rows]):.5f}")
    print(f"   certified {all(r['solver_result_code'] == 1 for r in rows)}")

    summary = {
        "floor_16bins": floor,
        "n_cells": len(rows),
        "train_mean_w2": float(np.mean([r["w2"] for r in train])),
        "heldout_mean_w2": float(np.mean([r["w2"] for r in held])),
        "all_mean_w2": float(np.mean([r["w2"] for r in rows])),
        "all_mean_over_floor": float(np.mean([r["w2"] for r in rows]) / floor),
        "mu1_row_mean_w2": float(np.mean([r["w2"] for r in slice_rows])),
        "all_max_w2": float(np.max([r["w2"] for r in rows])),
        "certified": all(r["solver_result_code"] == 1 for r in rows),
        "max_marginal_residual": float(np.max([r["marginal_residual"] for r in rows])),
        "max_primal_dual_gap": float(np.max([r["primal_dual_gap"] for r in rows])),
    }

    # ---- B and C: ablation ------------------------------------------------
    ablation_csv = G.RESULTS / "ablation" / "detail.csv"
    if ablation_csv.is_file():
        detail = read_csv(ablation_csv)
        m1 = float(np.mean([r["w2_one_param"] for r in detail]))
        m2 = float(np.mean([r["w2_two_param"] for r in detail]))
        gains = np.array([r["improvement"] for r in detail])
        off = np.array([r["mu_off_baseline"] for r in detail])
        print(f"\nB. ablation on {len(detail)} cells neither model trained on")
        print(f"   one-parameter  mean W2 {m1:.5f}  ({m1 / floor:.2f}x floor)")
        print(f"   two-parameter  mean W2 {m2:.5f}  ({m2 / floor:.2f}x floor)")
        print(f"   reduction      {1 - m2 / m1:+.1%}")
        print(f"   two-param wins {int((gains > 0).sum())}/{len(detail)}")
        corr = float(np.corrcoef(off, gains)[0, 1]) if off.std() > 0 and gains.std() > 0 else None
        print(f"   corr(gain, |log10 mu|) = {corr}")

        print("\nC. error vs distance from mu = 1")
        print(f"   {'|log10 mu|':>11} {'n':>3} {'one-param':>10} {'two-param':>10} {'gain':>9}")
        bands = {}
        for r in detail:
            bands.setdefault(round(r["mu_off_baseline"], 2), []).append(r)
        band_summary = {}
        for band in sorted(bands):
            group = bands[band]
            o = float(np.mean([r["w2_one_param"] for r in group]))
            t = float(np.mean([r["w2_two_param"] for r in group]))
            print(f"   {band:11.2f} {len(group):3d} {o:10.5f} {t:10.5f} {o - t:+9.5f}")
            band_summary[str(band)] = {"n": len(group), "one_param": o, "two_param": t,
                                       "gain": o - t}

        summary["ablation"] = {
            "n_cells": len(detail),
            "one_param_mean_w2": m1,
            "two_param_mean_w2": m2,
            "relative_reduction": float(1 - m2 / m1),
            "two_param_wins": int((gains > 0).sum()),
            "one_param_wins": int((gains < 0).sum()),
            "corr_gain_vs_mu_distance": corr,
            "by_mu_band": band_summary,
        }
    else:
        print("\nB. ablation: not available (run `python -m fullscale.cli ablation`)")

    # ---- D: continuity from the grid -------------------------------------
    print("\nD. continuity across neighbouring cells")
    values = {}
    for i in range(len(G.SIGMA_GRID)):
        for j in range(len(G.MU_GRID)):
            key = f"s{i:02d}_m{j:02d}"
            match = [r for r in rows if r["key"] == key]
            if match:
                values[i, j] = match[0]

    def cloud(cell):
        path = G.DATA / "clouds" / f"{cell.key}.npz"
        with np.load(path, allow_pickle=False) as payload:
            return payload["reference"].copy()

    def w2_between(a_cell, b_cell):
        from fullscale.evaluate import score_pair
        return score_pair(cloud(a_cell), cloud(b_cell), G.BINS)["w2"]

    sigma_ratios, mu_ratios = [], []
    for i in range(len(G.SIGMA_GRID)):
        for j in range(len(G.MU_GRID)):
            if (i, j) not in values:
                continue
            if (i + 1, j) in values:
                h = abs(math.log10(G.SIGMA_GRID[i + 1] / G.SIGMA_GRID[i]))
                d = w2_between(G.Cell(i, j), G.Cell(i + 1, j))
                sigma_ratios.append({"h": h, "w2": d, "ratio": d / h,
                                     "over_floor": d / floor})
            if (i, j + 1) in values:
                h = abs(math.log10(G.MU_GRID[j + 1] / G.MU_GRID[j]))
                d = w2_between(G.Cell(i, j), G.Cell(i, j + 1))
                mu_ratios.append({"h": h, "w2": d, "ratio": d / h, "over_floor": d / floor})

    for name, ratios in (("sigma", sigma_ratios), ("mu", mu_ratios)):
        if not ratios:
            print(f"   {name}: no complete neighbour pairs yet")
            continue
        arr = np.array([r["ratio"] for r in ratios])
        print(f"   {name}: n={len(ratios)}  W2/h max={arr.max():.3f} "
              f"median={np.median(arr):.3f}  p95={np.percentile(arr, 95):.3f}")
        summary.setdefault("continuity", {})[name] = {
            "n_pairs": len(ratios),
            "max_ratio": float(arr.max()),
            "median_ratio": float(np.median(arr)),
            "p95_ratio": float(np.percentile(arr, 95)),
        }

    (out / "analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out / 'analysis.json'}")

    if not a.no_plots:
        try:
            make_plots(rows, detail if ablation_csv.is_file() else None, out)
        except Exception as error:  # plotting is optional
            print(f"plots skipped: {error}")
    return 0


def make_plots(rows: list[dict], detail: list[dict] | None, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), layout="constrained")

    ax = axes[0]
    for mu, colour in ((0.1, "#7A1F1F"), (0.63095734448, "#D37B2F"),
                       (1.0, "#2864B4"), (3.981071705535, "#168173"),
                       (10.0, "#5B3A8C")):
        group = sorted([r for r in rows if abs(r["mu"] - mu) < 1e-9], key=lambda r: r["sigma"])
        if not group:
            continue
        ax.plot([r["sigma"] for r in group], [r["w2"] for r in group],
                "o-", ms=3.4, lw=1.5, color=colour, label=rf"$\mu={mu:.3g}$")
    ax.axhline(G.FLOOR_16, color="#777777", ls="--", lw=1.1)
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$W_2$ (16-bin exact OT)")
    ax.set_title("two-parameter model across the full grid", fontsize=9.5)
    ax.grid(alpha=.25)
    ax.legend(fontsize=7, frameon=False, ncol=2)

    ax = axes[1]
    if detail:
        bands = {}
        for r in detail:
            bands.setdefault(round(r["mu_off_baseline"], 2), []).append(r)
        xs = sorted(bands)
        one = [np.mean([r["w2_one_param"] for r in bands[b]]) for b in xs]
        two = [np.mean([r["w2_two_param"] for r in bands[b]]) for b in xs]
        ax.plot(xs, one, "s-", color="#D37B2F", ms=4.5, lw=1.6, label="one parameter ($\\sigma$ only)")
        ax.plot(xs, two, "o-", color="#2864B4", ms=4.5, lw=1.6, label="two parameters $(\\sigma,\\mu)$")
        ax.axhline(G.FLOOR_16, color="#777777", ls="--", lw=1.1)
        ax.set_xlabel(r"distance from the trained value $|\log_{10}\mu|$")
        ax.set_ylabel(r"$W_2$")
        ax.set_title("the cost of not knowing $\\mu$", fontsize=9.5)
        ax.legend(fontsize=7.5, frameon=False)
    else:
        ax.text(.5, .5, "ablation not run", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    ax.grid(alpha=.25)

    for ax in axes:
        ax.tick_params(labelsize=8)
    path = out / "fullscale_summary.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
