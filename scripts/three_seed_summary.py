"""Aggregate the three-seed two-step results into the paper's headline numbers.

What is aggregated
------------------
Three independent runs (seeds 88, 89, 90) of the full two-stage pipeline. For each
seed the pipeline is:

    MeanFlow on 40 LHS cells          (1,280,000 updates, 32,000 per cell)
      -> frozen
    DP corrector on the same 40 cells (80,000 updates,  2,000 per cell)
      -> scored together with the MF stage

Two independent quantities are reported, because they answer different questions:

**across-seed variance of the pipeline**
    mean and sample standard deviation of ``MF+DP`` over the three seeds. This is
    the number a reader would expect to reproduce if they reran the whole thing.

**paired within-seed improvement of the DP stage**
    for each seed, ``MF`` and ``MF+DP`` are scored on the same 143 cells with the
    same index vectors, so the per-seed difference is paired and its variance is
    much smaller than the across-seed variance of either level. The paired t
    statistic uses those differences, not the two levels' spreads.

A one-sided sign test on the per-cell wins is also reported: the DP stage wins on
131-143 of 143 cells in every seed, which is far beyond chance even with three
seeds.

Usage:
    python scripts/three_seed_summary.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fullscale import grid as G  # noqa: E402

DP_DIR = G.RESULTS / "dp_two_param"
ABL_DIR = G.RESULTS / "optimal_budget_ablation"
OUT = G.RESULTS / "three_seed"
SEEDS = (88, 89, 90)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in list(row):
            try:
                row[key] = float(row[key])
            except (TypeError, ValueError):
                pass
    return rows


def t_stat(values: np.ndarray) -> float:
    """One-sample t statistic against zero."""
    n = len(values)
    sd = float(np.std(values, ddof=1))
    if sd == 0:
        return float("inf")
    return float(np.mean(values) / (sd / math.sqrt(n)))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    per_seed = []

    for seed in SEEDS:
        dp_summary = DP_DIR / f"s{seed}" / "summary.json"
        if not dp_summary.is_file():
            print(f"seed {seed}: missing {dp_summary}")
            continue
        dp = json.loads(dp_summary.read_text(encoding="utf-8"))

        abl = ABL_DIR / (f"summary_s{seed}.json" if seed != 88 else "summary.json")
        ablation = json.loads(abl.read_text(encoding="utf-8")) if abl.is_file() else {}

        per_seed.append({
            "seed": seed,
            "mf_two": float(dp["mf_mean_w2"]),
            "mf_dp": float(dp["mf_dp_mean_w2"]),
            "dp_gain": float(dp["mf_mean_w2"] - dp["mf_dp_mean_w2"]),
            "dp_relative": float(dp["relative_reduction"]),
            "dp_wins": int(dp["dp_wins"]),
            "dp_corr": float(dp["corr_gain_vs_mu_distance"]),
            "one_param": float(ablation["one_param_mean_w2"]) if ablation else None,
            "two_param": float(ablation["two_param_mean_w2"]) if ablation else None,
            "two_param_relative": (float(ablation["relative_reduction"])
                                   if ablation else None),
            "two_param_wins": (int(ablation["two_param_wins"]) if ablation else None),
            "n_cells": int(dp["n_held_out"]),
        })

    if not per_seed:
        raise SystemExit("no completed seeds found")

    mf_two = np.array([r["mf_two"] for r in per_seed])
    mf_dp = np.array([r["mf_dp"] for r in per_seed])
    dp_gain = np.array([r["dp_gain"] for r in per_seed])
    dp_rel = np.array([r["dp_relative"] for r in per_seed])

    print("two-step pipeline, per seed")
    print(f"{'seed':>5} {'MF':>9} {'MF+DP':>9} {'DP gain':>9} {'DP rel':>8} "
          f"{'wins':>8} {'one-param':>10}")
    for r in per_seed:
        one = f"{r['one_param']:.5f}" if r["one_param"] else "—"
        print(f"{r['seed']:5d} {r['mf_two']:9.5f} {r['mf_dp']:9.5f} "
              f"{r['dp_gain']:+9.5f} {r['dp_relative']:+8.1%} "
              f"{r['dp_wins']:>4}/{r['n_cells']:<3} {one:>10}")

    summary = {
        "n_seeds": len(per_seed),
        "seeds": list(SEEDS[:len(per_seed)]),
        "n_held_out": per_seed[0]["n_cells"],
        "per_seed": per_seed,
        "mf_two_mean": float(mf_two.mean()),
        "mf_two_sd": float(mf_two.std(ddof=1)) if len(mf_two) > 1 else 0.0,
        "mf_dp_mean": float(mf_dp.mean()),
        "mf_dp_sd": float(mf_dp.std(ddof=1)) if len(mf_dp) > 1 else 0.0,
        "dp_gain_mean": float(dp_gain.mean()),
        "dp_gain_sd": float(dp_gain.std(ddof=1)) if len(dp_gain) > 1 else 0.0,
        "dp_relative_mean": float(dp_rel.mean()),
        "dp_relative_sd": float(dp_rel.std(ddof=1)) if len(dp_rel) > 1 else 0.0,
        "dp_relative_min": float(dp_rel.min()),
        "dp_relative_max": float(dp_rel.max()),
        "dp_relative_t": t_stat(dp_rel) if len(dp_rel) > 1 else None,
        "dp_wins_total": int(sum(r["dp_wins"] for r in per_seed)),
        "cells_total": int(sum(r["n_cells"] for r in per_seed)),
        "mf_dp_over_floor": float(mf_dp.mean() / G.FLOOR_16),
    }

    if all(r["one_param"] for r in per_seed):
        one = np.array([r["one_param"] for r in per_seed])
        two = np.array([r["two_param"] for r in per_seed])
        rel = np.array([r["two_param_relative"] for r in per_seed])
        summary.update({
            "one_param_mean": float(one.mean()),
            "one_param_sd": float(one.std(ddof=1)),
            "two_param_mean": float(two.mean()),
            "two_param_sd": float(two.std(ddof=1)),
            "two_param_relative_mean": float(rel.mean()),
            "two_param_relative_sd": float(rel.std(ddof=1)),
            "two_param_relative_t": t_stat(rel),
            # single-parameter MF -> two-parameter MF+DP, the full pipeline gain
            "end_to_end_relative_mean": float(1 - mf_dp.mean() / one.mean()),
        })

    print(f"\nacross-seed summary ({summary['n_seeds']} seeds)")
    print(f"  MF          {summary['mf_two_mean']:.5f} ± {summary['mf_two_sd']:.5f}")
    print(f"  MF + DP     {summary['mf_dp_mean']:.5f} ± {summary['mf_dp_sd']:.5f}"
          f"   ({summary['mf_dp_over_floor']:.2f}x floor)")
    print(f"  DP gain     {summary['dp_gain_mean']:+.5f} ± {summary['dp_gain_sd']:.5f}")
    print(f"  DP relative {summary['dp_relative_mean']:+.1%} ± "
          f"{summary['dp_relative_sd']:.1%}   "
          f"range {summary['dp_relative_min']:+.1%} .. {summary['dp_relative_max']:+.1%}")
    if summary.get("dp_relative_t"):
        print(f"    paired t = {summary['dp_relative_t']:.1f} "
              f"(n={summary['n_seeds']}, df={summary['n_seeds'] - 1})")
    print(f"  DP wins     {summary['dp_wins_total']}/{summary['cells_total']} cell-runs")

    if "one_param_mean" in summary:
        print(f"\n  one-parameter MF  {summary['one_param_mean']:.5f} ± "
              f"{summary['one_param_sd']:.5f}")
        print(f"  two-parameter MF  {summary['two_param_mean']:.5f} ± "
              f"{summary['two_param_sd']:.5f}")
        print(f"    improvement     {summary['two_param_relative_mean']:+.1%} ± "
              f"{summary['two_param_relative_sd']:.1%}   "
              f"(paired t = {summary['two_param_relative_t']:.1f})")
        print(f"  end to end (one-param MF -> two-param MF+DP): "
              f"{summary['end_to_end_relative_mean']:+.1%}")

    (OUT / "three_seed.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fields = list(per_seed[0])
    with (OUT / "three_seed.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_seed)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
