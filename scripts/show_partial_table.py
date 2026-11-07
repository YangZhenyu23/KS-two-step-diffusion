"""Print a per-cell table from partial_ablation.csv.

Usage:
    python scripts/show_partial_table.py
"""
from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    path = ROOT / "results" / "fullscale" / "partial" / "partial_ablation.csv"
    if not path.is_file():
        raise SystemExit(f"missing {path}; run scripts/fullscale_partial.py first")
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in list(row):
            try:
                row[key] = float(row[key])
            except (TypeError, ValueError):
                pass
    rows.sort(key=lambda r: (r["mu_off_baseline"], r["sigma"]))

    header = (f"{'sigma':>7} {'mu':>9} {'one-param':>10} {'two-param':>10} "
              f"{'gain':>9} {'ratio':>7}  verdict")
    print(header)
    print("-" * len(header))
    for r in rows:
        verdict = "two-param" if r["improvement"] > 0 else "one-param"
        print(f"{r['sigma']:7.0f} {r['mu']:9.4f} {r['w2_one_param']:10.5f} "
              f"{r['w2_two_param']:10.5f} {r['improvement']:+9.5f} "
              f"{r['ratio']:7.3f}  {verdict}")

    # grouped by mu distance
    print("\nby distance from mu = 1")
    bands: dict[float, list[dict]] = {}
    for r in rows:
        bands.setdefault(round(r["mu_off_baseline"], 2), []).append(r)
    print(f"{'|log10 mu|':>11} {'n':>3} {'one-param':>10} {'two-param':>10} {'gain':>9} {'rel.':>8}")
    for band in sorted(bands):
        group = bands[band]
        one = sum(r["w2_one_param"] for r in group) / len(group)
        two = sum(r["w2_two_param"] for r in group) / len(group)
        rel = (one - two) / one if one else 0.0
        print(f"{band:11.2f} {len(group):3d} {one:10.5f} {two:10.5f} {one - two:+9.5f} {rel:+8.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
