"""How does the DP stage behave on the high-sigma tail?

Motivation
----------
The three-seed runs already store per-cell errors for MF and MF+DP on 143
held-out cells. Fifty of those cells sit at sigma >= 160, which is the region
where the MeanFlow error grows fastest (0.85 at sigma=200 versus 0.16 near
sigma=100). No new training is needed to ask whether DP rescues that tail.

What is measured
----------------
For each seed and each sigma, the mean over the ten mu levels of

    MF error, MF+DP error, and the ratio between them.

Then the same is aggregated over seeds, for three sigma bands:

    low        10-80
    middle     90-150
    high       160-200

A trend worth reporting exists if the DP gain grows with sigma: that would say
the corrector is not merely polishing an already-good map but repairing the
regions where the one-step map fails.

Caveat on what this is not
--------------------------
The MF models here were trained on LHS cells that include sigma = 150 and 200, so
sigma >= 160 is *interpolation* for them, not extrapolation. This analysis
therefore characterises the behaviour of the two-stage pipeline in a hard region,
not its ability to extrapolate to unseen sigma. A genuine extrapolation test
requires retraining with sigma capped at 150.

Usage:
    python scripts/high_sigma_analysis.py
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
OUT = G.RESULTS / "high_sigma"
SEEDS = (88, 89, 90)


def load_detail(seed: int) -> list[dict]:
    path = DP_DIR / f"s{seed}" / "detail.csv"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in list(row):
            try:
                row[key] = float(row[key])
            except (TypeError, ValueError):
                pass
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    per_seed = {}
    for seed in SEEDS:
        rows = load_detail(seed)
        per_seed[seed] = rows
        sigmas = sorted({r["sigma"] for r in rows})
        if seed == SEEDS[0]:
            print(f"{len(rows)} cells per seed, sigma levels: "
                  f"{[int(s) for s in sigmas]}")

    # ---- per-sigma table, averaged over seeds ----------------------------
    sigmas = sorted({r["sigma"] for r in per_seed[SEEDS[0]]})
    print(f"\n{'sigma':>6} {'n_mu':>5} {'MF':>9} {'MF+DP':>9} {'gain':>9} "
          f"{'rel':>8}  per-seed rel")
    table = []
    for sigma in sigmas:
        mf_all, dp_all, rels = [], [], []
        for seed in SEEDS:
            group = [r for r in per_seed[seed] if r["sigma"] == sigma]
            if not group:
                continue
            mf = float(np.mean([r["w2_mf"] for r in group]))
            dp = float(np.mean([r["w2_mf_dp"] for r in group]))
            mf_all.append(mf)
            dp_all.append(dp)
            rels.append(1 - dp / mf)
        if not mf_all:
            continue
        mf_mean = float(np.mean(mf_all))
        dp_mean = float(np.mean(dp_all))
        rel_mean = float(np.mean(rels))
        table.append({"sigma": sigma, "n_seeds": len(mf_all),
                      "mf": mf_mean, "mf_dp": dp_mean,
                      "gain": mf_mean - dp_mean, "relative": rel_mean,
                      "per_seed_relative": rels})
        print(f"{sigma:6.0f} {len([r for r in per_seed[SEEDS[0]] if r['sigma'] == sigma]):5d} "
              f"{mf_mean:9.5f} {dp_mean:9.5f} {mf_mean - dp_mean:+9.5f} "
              f"{rel_mean:+8.1%}  "
              f"{', '.join(f'{v:+.1%}' for v in rels)}")

    # ---- by band ---------------------------------------------------------
    bands = {"low (10-80)": (10, 80), "middle (90-150)": (90, 150),
             "high (160-200)": (160, 200)}
    print(f"\n{'band':<16} {'MF':>9} {'MF+DP':>9} {'gain':>9} {'rel':>8} "
          f"{'rel sd':>8}")
    band_summary = {}
    for label, (lo, hi) in bands.items():
        entries = [r for r in table if lo <= r["sigma"] <= hi]
        if not entries:
            continue
        mf = float(np.mean([r["mf"] for r in entries]))
        dp = float(np.mean([r["mf_dp"] for r in entries]))
        rels = np.array([r["relative"] for r in entries])
        band_summary[label] = {"n_sigma": len(entries), "mf": mf, "mf_dp": dp,
                               "gain": mf - dp, "relative": float(rels.mean()),
                               "relative_sd": float(rels.std(ddof=1))
                               if len(rels) > 1 else 0.0}
        print(f"{label:<16} {mf:9.5f} {dp:9.5f} {mf - dp:+9.5f} "
              f"{rels.mean():+8.1%} {band_summary[label]['relative_sd']:8.1%}")

    # ---- does the gain grow with sigma? ---------------------------------
    sig = np.array([r["sigma"] for r in table])
    rel = np.array([r["relative"] for r in table])
    mf = np.array([r["mf"] for r in table])
    corr = float(np.corrcoef(sig, rel)[0, 1])
    corr_mf = float(np.corrcoef(sig, mf)[0, 1])
    print(f"\ncorr(sigma, DP relative gain) = {corr:+.3f}")
    print(f"corr(sigma, MF error)         = {corr_mf:+.3f}")
    print("  a positive first value says the corrector helps more as the one-step")
    print("  map degrades.")

    # ---- the worst cells -------------------------------------------------
    worst = []
    for seed in SEEDS:
        for row in per_seed[seed]:
            if row["sigma"] >= 160:
                worst.append({"seed": seed, "sigma": row["sigma"],
                              "mu": row["mu"],
                              "mf": row["w2_mf"], "mf_dp": row["w2_mf_dp"],
                              "rel": 1 - row["w2_mf_dp"] / row["w2_mf"]})
    worst.sort(key=lambda r: -r["mf"])
    print(f"\nthe five hardest cells (largest MF error), all seeds pooled")
    print(f"{'seed':>5} {'sigma':>6} {'mu':>9} {'MF':>9} {'MF+DP':>9} {'rel':>8}")
    for r in worst[:5]:
        print(f"{r['seed']:5d} {r['sigma']:6.0f} {r['mu']:9.3f} "
              f"{r['mf']:9.5f} {r['mf_dp']:9.5f} {r['rel']:+8.1%}")

    out = {
        "note": ("sigma >= 160 is interpolation for these models: the LHS training "
                 "cells include sigma = 150 and 200. This measures behaviour in a "
                 "hard region, not extrapolation to unseen sigma."),
        "per_sigma": table,
        "bands": band_summary,
        "corr_sigma_vs_dp_gain": corr,
        "corr_sigma_vs_mf_error": corr_mf,
        "hardest_cells": worst[:10],
    }
    (OUT / "high_sigma.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    fields = ["sigma", "n_seeds", "mf", "mf_dp", "gain", "relative"]
    with (OUT / "high_sigma.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
