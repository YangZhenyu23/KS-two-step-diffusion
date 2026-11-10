"""Can a change of length unit explain the gap between our W2 and the paper's?

Motivation
----------
The paper defines W2 on empirical measures (Eq. 7) and never mentions
normalisation, binning or a coordinate scale. Our particle-level values are
15-25x larger than the paper's, so the natural question is whether the paper
reports W2 in different units.

W2 is positively homogeneous of degree one under a global scaling of the
coordinates:

    W2(sX, sY) = s * W2(X, Y).

Therefore dividing our numbers by a factor s is legitimate and free: no
recomputation is needed, only division. This script applies that identity with
several defensible choices of s and asks which, if any, brings the results into
the paper's range.

Candidate scales
----------------
    unit          s = 1                     (raw coordinates)
    extent_x      s = the cloud's x-extent  (the shear direction)
    extent_max    s = the largest axis extent
    ball_diag     s = sqrt(3) (diameter of the unit-ball support)
    cloud_diag    s = the diagonal of the cloud's bounding box
    fitted        s = the factor that would exactly match the paper at each sigma

The last one is diagnostic: it reports the scale the paper would have to be
using. If that value is wildly different from every geometric scale of the
clouds, then a change of units cannot be the explanation.

What this does not test
-----------------------
Anisotropic rescaling (normalising each axis separately) is not a change of unit
and does not obey the homogeneity identity, so it cannot be done for free. It is
deliberately excluded; the homogeneity test answers the question that was asked.

Usage:
    python scripts/scale_hypothesis.py
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

CLOUDS = ROOT / "repro_ks3d" / "evaluation" / "ks" / "generated" / "clouds" / "ks3d"
PARTICLE = ROOT / "results" / "ks3d" / "dp_particle" / "dp_particle_m10000.csv"
OUT = ROOT / "results" / "ks3d" / "scale_hypothesis"

# paper Figure 2, left table
PAPER = {20: (0.0084, 0.0047), 40: (0.0068, 0.0046), 60: (0.0070, 0.0049),
         80: (0.0105, 0.0056), 100: (0.0132, 0.0059), 120: (0.0201, 0.0065),
         140: (0.0216, 0.0073), 160: (0.0403, 0.0082), 180: (0.0832, 0.0140),
         200: (0.1970, 0.0214)}


def main() -> int:
    if not PARTICLE.is_file():
        raise SystemExit("run scripts/dp_particle_w2.py first")
    with PARTICLE.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    mine: dict[int, dict[str, float]] = {}
    for row in rows:
        mine.setdefault(int(float(row["sigma"])), {})[row["method"]] = float(row["w2_particle"])

    OUT.mkdir(parents=True, exist_ok=True)
    geometric = []
    for sigma in sorted(mine):
        path = CLOUDS / f"sigma{sigma}.npz"
        data = np.load(path)
        ref = data["reference"].astype(np.float64)
        lo, hi = ref.min(axis=0), ref.max(axis=0)
        extent = hi - lo
        geometric.append({
            "sigma": sigma,
            "extent_x": float(extent[0]),
            "extent_y": float(extent[1]),
            "extent_z": float(extent[2]),
            "extent_max": float(extent.max()),
            "cloud_diag": float(np.linalg.norm(extent)),
            "ball_diag": math.sqrt(3.0),
        })
    by_sigma = {g["sigma"]: g for g in geometric}

    scales = {
        "unit": lambda g: 1.0,
        "extent_x": lambda g: g["extent_x"],
        "extent_max": lambda g: g["extent_max"],
        "ball_diag": lambda g: g["ball_diag"],
        "cloud_diag": lambda g: g["cloud_diag"],
    }

    print("geometric scales of the reference clouds")
    print(f"{'sigma':>6} {'ext_x':>8} {'ext_max':>8} {'diag':>8}")
    for g in geometric:
        print(f"{g['sigma']:6d} {g['extent_x']:8.3f} {g['extent_max']:8.3f} "
              f"{g['cloud_diag']:8.3f}")

    results = {}
    common = [s for s in sorted(mine) if s in PAPER]
    for name, fn in scales.items():
        mf_scaled, dp_scaled, mf_ratio, dp_ratio = [], [], [], []
        for sigma in common:
            s = fn(by_sigma[sigma])
            pmf, pdp = PAPER[sigma]
            mf_scaled.append(mine[sigma]["mf"] / s)
            dp_scaled.append(mine[sigma]["mf_dp"] / s)
            mf_ratio.append((mine[sigma]["mf"] / s) / pmf)
            dp_ratio.append((mine[sigma]["mf_dp"] / s) / pdp)
        results[name] = {
            "mf_mean_scaled": float(np.mean(mf_scaled)),
            "dp_mean_scaled": float(np.mean(dp_scaled)),
            "mf_over_paper_mean": float(np.mean(mf_ratio)),
            "dp_over_paper_mean": float(np.mean(dp_ratio)),
            "dp_over_paper_sd": float(np.std(dp_ratio, ddof=1)),
        }

    print(f"\n{'scale':<12} {'s range':>16} {'MF/paper':>10} {'DP/paper':>10} "
          f"{'DP sd':>8}")
    print("-" * 62)
    for name, fn in scales.items():
        vals = [fn(by_sigma[s]) for s in common]
        r = results[name]
        print(f"{name:<12} {min(vals):7.2f} - {max(vals):6.2f} "
              f"{r['mf_over_paper_mean']:10.2f} {r['dp_over_paper_mean']:10.2f} "
              f"{r['dp_over_paper_sd']:8.2f}")

    # ---- what scale would the paper have to use? -------------------------
    print("\nscale required to match the paper exactly (diagnostic)")
    print(f"{'sigma':>6} {'s for MF':>10} {'s for DP':>10} {'cloud diag':>11} "
          f"{'s / diag':>9}")
    fitted = []
    for sigma in common:
        pmf, pdp = PAPER[sigma]
        s_mf = mine[sigma]["mf"] / pmf
        s_dp = mine[sigma]["mf_dp"] / pdp
        diag = by_sigma[sigma]["cloud_diag"]
        fitted.append({"sigma": sigma, "s_mf": s_mf, "s_dp": s_dp,
                       "cloud_diag": diag, "s_dp_over_diag": s_dp / diag})
        print(f"{sigma:6d} {s_mf:10.1f} {s_dp:10.1f} {diag:11.3f} {s_dp / diag:9.1f}")

    arr = np.array([f["s_dp_over_diag"] for f in fitted])
    print(f"\nrequired scale / cloud diagonal: mean {arr.mean():.1f}x, "
          f"range {arr.min():.1f}-{arr.max():.1f}")
    print("  a pure change of unit would need this ratio to be constant and equal")
    print("  to 1 (i.e. the paper measuring in units of the cloud size).")

    verdict = ("a change of unit cannot explain the gap"
               if arr.min() > 2 or arr.max() / max(arr.min(), 1e-9) > 2
               else "a change of unit might explain the gap")
    print(f"\nverdict: {verdict}")

    (OUT / "scale_hypothesis.json").write_text(json.dumps({
        "geometric": geometric, "scales": results, "fitted": fitted,
        "required_scale_over_diag_mean": float(arr.mean()),
        "required_scale_over_diag_range": [float(arr.min()), float(arr.max())],
        "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

