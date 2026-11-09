"""How far is the model from the local noise floor, cell by cell?

The single global floor (0.0644) is misleading: measured locally the floor ranges
from 0.051 to 0.123 across the parameter box, a factor of 2.4. Dividing a global
mean error by a single-point floor therefore mixes regions that are near the
statistical limit with regions that are far from it.

This script takes every cell where both a local floor and a model error are
available, and reports

    ratio  = model W2 / local floor
    excess = model W2 - local floor          (pessimistic, triangle inequality)
    quad   = sqrt(max(model^2 - floor^2,0))  (optimistic, orthogonal)

Then it asks the question that decides what to do next: **is the remaining error
concentrated in cells that are near their floor, or in cells that are far from
it?** If most cells are near 1, the model has essentially saturated this metric
and effort should go elsewhere (a finer metric). If many cells are above 2, there
is real modelling headroom and it is worth chasing.

Usage:
    python scripts/floor_gap_analysis.py
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

OUT = G.RESULTS / "local_floor"


def main() -> int:
    floor_path = OUT / "local_floor.json"
    detail_path = G.RESULTS / "optimal_budget_ablation" / "detail.csv"
    if not floor_path.is_file():
        raise SystemExit("run scripts/local_floor.py first")
    if not detail_path.is_file():
        raise SystemExit("run scripts/ablation_optimal_budget.py first")

    floors = json.loads(floor_path.read_text(encoding="utf-8"))["points"]
    with detail_path.open(encoding="utf-8") as stream:
        model = {r["key"]: float(r["w2_two_param"]) for r in csv.DictReader(stream)}

    print(f"local floors measured at {len(floors)} points")
    print(f"model errors available on {len(model)} cells\n")

    matched = []
    for entry in floors:
        key = entry["key"]
        if key not in model:
            print(f"  {key} (sigma={entry['sigma']:g}, mu={entry['mu']:g}): "
                  f"floor {entry['floor']:.5f}, no model error "
                  f"(cell not in the held-out set)")
            continue
        err = model[key]
        floor = entry["floor"]
        matched.append({
            "key": key, "sigma": entry["sigma"], "mu": entry["mu"],
            "floor": floor, "model": err,
            "ratio": err / floor,
            "excess_difference": err - floor,
            "excess_quadrature": math.sqrt(max(err * err - floor * floor, 0.0)),
        })

    print(f"\n{'sigma':>7} {'mu':>8} {'floor':>9} {'model':>9} {'ratio':>7} "
          f"{'excess(diff)':>13} {'excess(quad)':>13}  verdict")
    for r in matched:
        v = ("at limit" if r["ratio"] < 1.3 else
             "near limit" if r["ratio"] < 1.7 else
             "clear headroom")
        print(f"{r['sigma']:7.0f} {r['mu']:8.3f} {r['floor']:9.5f} {r['model']:9.5f} "
              f"{r['ratio']:7.2f} {r['excess_difference']:13.5f} "
              f"{r['excess_quadrature']:13.5f}  {v}")

    if not matched:
        raise SystemExit("no cell has both a floor and a model error")

    ratios = np.array([r["ratio"] for r in matched])
    floors_arr = np.array([r["floor"] for r in matched])
    models_arr = np.array([r["model"] for r in matched])

    print(f"\nratio: min {ratios.min():.2f}  max {ratios.max():.2f}  "
          f"mean {ratios.mean():.2f}")
    print(f"local floor spread: {floors_arr.min():.5f} .. {floors_arr.max():.5f} "
          f"({floors_arr.max() / floors_arr.min():.2f}x)")
    if len(matched) > 2:
        corr = float(np.corrcoef(floors_arr, models_arr)[0, 1])
        print(f"corr(local floor, model error) = {corr:+.3f}")
        print("  a strong positive correlation means the model's error tracks the")
        print("  floor, i.e. the hard cells are hard for the model and for sampling")
        print("  alike. That would mean most of the error is irreducible.")

    # extrapolate to the full grid using the ratio found here
    mean_ratio = float(ratios.mean())
    print(f"\nusing the mean ratio {mean_ratio:.2f}x and the global mean model")
    print(f"error {models_arr.mean():.5f}: implied mean local floor")
    print(f"  {models_arr.mean() / mean_ratio:.5f}")
    print("  (a sanity check: it should land inside the measured floor range)")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "floor_gap.json").write_text(json.dumps({
        "matched": matched,
        "ratio_mean": mean_ratio, "ratio_min": float(ratios.min()),
        "ratio_max": float(ratios.max()),
        "floor_min": float(floors_arr.min()), "floor_max": float(floors_arr.max()),
        "implied_mean_floor": float(models_arr.mean() / mean_ratio),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'floor_gap.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
