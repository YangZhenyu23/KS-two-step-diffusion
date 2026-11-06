"""Where does the second parameter actually pay off?

The budget-matched comparison shows the two-parameter model losing *on average*.
That average hides the structure that matters: for cells with ``mu = 1`` the
single-parameter model is exactly right by construction, while for cells with
``mu`` far from 1 it is applying the wrong physics. If mu carries information,
the two-parameter model must win specifically in the far-mu regime.

This script reports the per-mu-band breakdown and the *cost of not knowing mu*:
the error the sigma-only model incurs as mu moves away from the value it was
trained at. It also prints the model error normalised by the physical floor, so
a reader can see how much of each number is irreducible statistical noise.

Usage:
    python scripts/analyse_mu_effect.py --csv results/twoparam/compare/budget_matched_heldout.csv
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

from twoparam import protocol as P  # noqa: E402


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
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
    ap.add_argument("--csv", type=Path,
                    default=P.RESULTS / "compare" / "budget_matched_heldout.csv")
    ap.add_argument("--out", type=Path, default=P.RESULTS / "compare" / "mu_effect.json")
    a = ap.parse_args()

    rows = load(a.csv)
    print(f"{len(rows)} held-out cells, all with mu != 1\n")

    print("=== error vs distance from mu = 1 ===")
    print(f"{'|log10 mu|':>11} {'n':>3} {'single-param':>13} {'two-param':>11} "
          f"{'gain':>9} {'rel.':>8}")
    bands: dict[float, list[dict]] = {}
    for row in rows:
        bands.setdefault(round(row["mu_off_baseline"], 2), []).append(row)
    band_summary = {}
    for band in sorted(bands):
        group = bands[band]
        one = float(np.mean([r["w2_one_param"] for r in group]))
        two = float(np.mean([r["w2_two_param"] for r in group]))
        rel = (one - two) / one if one else 0.0
        print(f"{band:11.2f} {len(group):3d} {one:13.5f} {two:11.5f} {one - two:+9.4f} {rel:+8.1%}")
        band_summary[str(band)] = {"n": len(group), "one_param": one, "two_param": two,
                                   "gain": one - two, "relative": rel}

    far = [r for r in rows if r["mu_off_baseline"] >= 0.8]
    near = [r for r in rows if r["mu_off_baseline"] <= 0.2]
    print(f"\nfar  (|log10 mu| >= 0.8, n={len(far)}): "
          f"single {np.mean([r['w2_one_param'] for r in far]):.5f}  "
          f"two {np.mean([r['w2_two_param'] for r in far]):.5f}")
    print(f"near (|log10 mu| <= 0.2, n={len(near)}): "
          f"single {np.mean([r['w2_one_param'] for r in near]):.5f}  "
          f"two {np.mean([r['w2_two_param'] for r in near]):.5f}")

    print("\n=== error normalised by the physical floor ===")
    floor_path = P.RESULTS / "floor.json"
    floor = None
    if floor_path.is_file():
        payload = json.loads(floor_path.read_text(encoding="utf-8"))
        values = payload["summary"].get("N20000_B16") or payload["summary"].get("N50000_B16")
        floor = values["mean"] if values else None
    if floor:
        one = float(np.mean([r["w2_one_param"] for r in rows]))
        two = float(np.mean([r["w2_two_param"] for r in rows]))
        print(f"floor (mu=1, sigma=100, N=20k, 16 bins) = {floor:.5f}")
        print(f"single-param / floor = {one / floor:.2f}x   ({one:.5f})")
        print(f"two-param    / floor = {two / floor:.2f}x   ({two:.5f})")
        print("A ratio near 1 means the model is at the statistical noise limit;")
        print("a ratio below 1 is impossible and would signal a metric error.")
    else:
        print("floor.json not found; run scripts/measure_floor.py first")

    print("\n=== the cost of not knowing mu ===")
    print("For each cell, compare the sigma-only model (which assumes mu = 1) against")
    print("the reference solution. The growth of that error with mu distance is the")
    print("amount of physics the single-parameter design is missing.")
    by_band_one = {b: v["one_param"] for b, v in band_summary.items()}
    baseline_band = min(by_band_one, key=lambda b: abs(float(b)))
    base_value = by_band_one[baseline_band]
    print(f"\nreference band |log10 mu| = {baseline_band}: single-param W2 = {base_value:.5f}")
    for band in sorted(by_band_one, key=float):
        ratio = by_band_one[band] / base_value
        print(f"  |log10 mu| = {band}: {by_band_one[band]:.5f}  ({ratio:.2f}x the near-baseline error)")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"bands": band_summary, "floor": floor,
                                 "far": {"n": len(far),
                                         "one_param": float(np.mean([r["w2_one_param"] for r in far])),
                                         "two_param": float(np.mean([r["w2_two_param"] for r in far]))}
                                 if far else None}, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
