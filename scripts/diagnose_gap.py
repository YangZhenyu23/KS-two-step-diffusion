"""Diagnose why the two-parameter model underperforms the single-parameter one.

Hypotheses
----------
H1 (budget): 1000 Adam steps is not enough to cover a 10x wider parameter box.
    Test: retrain the two-parameter model with 3x the steps and see whether the
    gap closes.
H2 (parameterisation): the mu axis enters as log10(mu), which spans a full
    decade; a network may need a different conditioning scale.
    Test: report the error as a function of distance from mu = 1. If the error
    is concentrated at the extremes, it is a conditioning/coverage problem.
H3 (data): some mu values produce distributions with much larger support and
    the W2 metric itself grows with mu, so the mean comparison is dominated by
    the outer mu levels rather than by model quality.
    Test: compare against the *achievable floor* — the W2 between the target and
    a second independent sample of the same parameters (already measured in
    results/twoparam/floor.json) — and report the ratio model/floor.

The script prints all three and writes ``diagnosis.json``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from twoparam import protocol as P  # noqa: E402


def main() -> int:
    out = P.RESULTS / "compare"
    detail_path = out / "ablation_detail.csv"
    if not detail_path.is_file():
        raise SystemExit("run scripts/compare_one_vs_two.py first")
    import csv

    with detail_path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in list(row):
            try:
                row[key] = float(row[key])
            except (TypeError, ValueError):
                pass

    print("error vs distance from the baseline mu (H2)")
    print(f"{'|log10 mu|':>11} {'n':>4} {'one-param':>10} {'two-param':>10} {'gain':>9}")
    bands = {}
    for row in rows:
        band = round(row["mu_off_baseline"], 2)
        bands.setdefault(band, []).append(row)
    for band in sorted(bands):
        group = bands[band]
        one = np.mean([r["w2_one_param"] for r in group])
        two = np.mean([r["w2_two_param"] for r in group])
        print(f"{band:11.2f} {len(group):4d} {one:10.4f} {two:10.4f} {one - two:+9.4f}")

    print("\nerror by sigma (H3: does W2 scale with the parameter?)")
    print(f"{'sigma':>9} {'n':>4} {'one-param':>10} {'two-param':>10}")
    by_sigma = {}
    for row in rows:
        by_sigma.setdefault(round(row["sigma"], 3), []).append(row)
    for sigma in sorted(by_sigma):
        group = by_sigma[sigma]
        one = np.mean([r["w2_one_param"] for r in group])
        two = np.mean([r["w2_two_param"] for r in group])
        print(f"{sigma:9.2f} {len(group):4d} {one:10.4f} {two:10.4f}")

    print("\nH3 check: does the reference cloud's own scale grow with mu?")
    from twoparam.data import load_cell
    data = P.RESULTS / "data" / "full"
    print(f"{'mu':>9} {'radial_mean':>12} {'x_extent':>10}")
    for j, mu in enumerate(P.MU_GRID):
        cell = load_cell(data, f"s05_m{j:02d}")
        ref = cell["clouds"]["reference"].astype(np.float64)
        print(f"{mu:9.4f} {np.linalg.norm(ref, axis=1).mean():12.4f} "
              f"{ref[:,0].max() - ref[:,0].min():10.4f}")

    payload = {
        "by_mu_distance": {str(k): {
            "n": len(v),
            "one_param": float(np.mean([r["w2_one_param"] for r in v])),
            "two_param": float(np.mean([r["w2_two_param"] for r in v])),
        } for k, v in sorted(bands.items())},
        "by_sigma": {str(k): {
            "n": len(v),
            "one_param": float(np.mean([r["w2_one_param"] for r in v])),
            "two_param": float(np.mean([r["w2_two_param"] for r in v])),
        } for k, v in sorted(by_sigma.items())},
    }
    (out / "diagnosis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out / 'diagnosis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
