"""Inspect the centroid_y/z values behind the meaningless relative errors.

``physical_quantities.py`` reported mean |relative error| of 7.47 for centroid_z. That
cannot be a real error: the y and z centroids are near zero by symmetry, so dividing by
them produces explosive ratios. The absolute values are the meaningful ones. This prints
both so the report can quote the right thing.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
payload = json.loads((ROOT / "results/fullscale/physics/physical_quantities.json").read_text("utf-8"))

print(f"{'cell':>8} {'method':>7} {'cent_y':>9} {'ref_y':>9} {'|abs|':>9} "
      f"{'cent_z':>9} {'ref_z':>9} {'|abs|':>9}")
for row in payload["rows"]:
    p, r = row["predicted"], row["reference"]
    print(f"{row['key']:>8} {row['method']:>7} {p['centroid_y']:9.5f} {r['centroid_y']:9.5f} "
          f"{abs(p['centroid_y'] - r['centroid_y']):9.5f} "
          f"{p['centroid_z']:9.5f} {r['centroid_z']:9.5f} "
          f"{abs(p['centroid_z'] - r['centroid_z']):9.5f}")

print("\nabsolute deviations (the meaningful statistic for quantities centred on zero):")
for method in ("mf", "mf_dp"):
    for name in ("centroid_y", "centroid_z"):
        values = [abs(row["predicted"][name] - row["reference"][name])
                  for row in payload["rows"] if row["method"] == method]
        print(f"  {method:>6} {name}: max {max(values):.5f}  mean {sum(values) / len(values):.5f}")
