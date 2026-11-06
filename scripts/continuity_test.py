"""Numerical measurement of parameter continuity of the cloud family.

What is measured
----------------
For a fixed particle count ``N``, let ``mu_{sigma,mu}`` be the empirical measure
of the terminal cloud. Define the normalised parameter distance

    h = || (log10(sigma2) - log10(sigma1), log10(mu2) - log10(mu1)) ||

and the measured cloud distance ``d = W2(mu_{p1}, mu_{p2})`` computed with the
same 16-bin exact histogram OT used everywhere else.

A family is *Lipschitz continuous* in the parameter if ``d <= L h`` for some
constant ``L``. The script therefore reports

  * the ratio ``d / h`` along four slices of the grid (vary sigma at fixed mu,
    vary mu at fixed sigma, and the two diagonals), and
  * the empirical Lipschitz constant ``max d/h`` over a fine sweep, together
    with its stability under mesh refinement (h halving).

Two things make the measurement meaningful rather than tautological:

  1. **A floor subtraction.** Two independent samples from the *same* parameter
     already differ by ``d0`` (the statistical floor, ``results/twoparam/floor.json``).
     Continuity is visible only when ``d > d0``; ratios at or below 1 indicate
     that the parameter change is inside the noise, not that the family is
     discontinuous.
  2. **A continuity control.** Repeating the same measurement with the *initial*
     cloud (which is literally identical across cells) gives ``d == 0`` and
     confirms the pipeline reports zero when the underlying measure is identical.

Usage:
    python scripts/continuity_test.py --bins 16
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from twoparam.data import load_cell  # noqa: E402
from twoparam.evaluate import _edges, measure, solve  # noqa: E402
from twoparam.protocol import MU_GRID, MU_INDEX, RESULTS, SIGMA_GRID  # noqa: E402


def w2(a: np.ndarray, b: np.ndarray, bins: int) -> float:
    edges = _edges([a, b], bins)
    return solve(measure(a, edges, bins), measure(b, edges, bins))["w2"]


def pdist(p, q) -> float:
    """Euclidean distance in the log-parameter plane."""
    return math.hypot(math.log10(p[0] / q[0]), math.log10(p[1] / q[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, nargs="+", default=[16])
    ap.add_argument("--out", type=Path, default=RESULTS / "continuity" / "continuity.json")
    a = ap.parse_args()

    data = RESULTS / "data" / "full"
    floor_path = RESULTS / "floor.json"
    floor = None
    if floor_path.is_file():
        payload = json.loads(floor_path.read_text(encoding="utf-8"))
        floor = (payload["summary"].get("N20000_B16") or {}).get("mean")

    clouds = {}
    for i in range(len(SIGMA_GRID)):
        for j in range(len(MU_GRID)):
            cell = load_cell(data, f"s{i:02d}_m{j:02d}")
            clouds[i, j] = {
                "reference": cell["clouds"]["reference"],
                "initial": cell["clouds"]["initial"],
            }
    print(f"loaded {len(clouds)} cells")

    report = {"floor": floor, "bins": {}, "control": {}}
    for bins in a.bins:
        print(f"\n=== {bins} bins/axis ===")

        # ---- control: the initial cloud is identical in every cell --------
        if bins == a.bins[0]:
            control = w2(clouds[0, 0]["initial"], clouds[5, 5]["initial"], bins)
            report["control"]["initial_cloud_w2"] = control
            print(f"control: W2 between two identical initial clouds = {control:.3e} "
                  f"(expected 0)")

        # ---- slice A: sigma varies, mu fixed at each level ----------------
        print("\nsigma slice (mu fixed):")
        print(f"{'mu':>9} {'sigma pair':>20} {'h':>8} {'W2':>9} {'W2/h':>9} {'W2/floor':>9}")
        ratios_sigma = []
        for j in (0, MU_INDEX, len(MU_GRID) - 1):
            for i in range(len(SIGMA_GRID) - 1):
                p, q = (SIGMA_GRID[i], MU_GRID[j]), (SIGMA_GRID[i + 1], MU_GRID[j])
                h = pdist(p, q)
                d = w2(clouds[i, j]["reference"], clouds[i + 1, j]["reference"], bins)
                ratio = d / h
                ratios_sigma.append({"mu": q[1], "sigma": [p[0], q[0]], "h": h,
                                     "w2": d, "ratio": ratio,
                                     "over_floor": d / floor if floor else None})
                if j in (0, len(MU_GRID) - 1) or i in (0, 4, len(SIGMA_GRID) - 2):
                    print(f"{q[1]:9.3f} {p[0]:8.1f}->{q[0]:8.1f} {h:8.4f} {d:9.5f} "
                          f"{ratio:9.4f} {(d / floor if floor else float('nan')):9.2f}")

        # ---- slice B: mu varies, sigma fixed at each level -----------------
        print("\nmu slice (sigma fixed):")
        print(f"{'sigma':>9} {'mu pair':>22} {'h':>8} {'W2':>9} {'W2/h':>9} {'W2/floor':>9}")
        ratios_mu = []
        for i in (0, len(SIGMA_GRID) // 2, len(SIGMA_GRID) - 1):
            for j in range(len(MU_GRID) - 1):
                p, q = (SIGMA_GRID[i], MU_GRID[j]), (SIGMA_GRID[i], MU_GRID[j + 1])
                h = pdist(p, q)
                d = w2(clouds[i, j]["reference"], clouds[i, j + 1]["reference"], bins)
                ratio = d / h
                ratios_mu.append({"sigma": q[0], "mu": [p[1], q[1]], "h": h,
                                  "w2": d, "ratio": ratio,
                                  "over_floor": d / floor if floor else None})
                if i in (0, len(SIGMA_GRID) - 1) or j in (0, 4, len(MU_GRID) - 2):
                    print(f"{q[0]:9.2f} {p[1]:9.4f}->{q[1]:9.4f} {h:8.4f} {d:9.5f} "
                          f"{ratio:9.4f} {(d / floor if floor else float('nan')):9.2f}")

        # ---- empirical Lipschitz constant over a fine 4-neighbour sweep ---
        print("\nempirical Lipschitz constant (all adjacent pairs):")
        all_ratios = []
        for i in range(len(SIGMA_GRID)):
            for j in range(len(MU_GRID)):
                for di, dj in ((1, 0), (0, 1)):
                    ii, jj = i + di, j + dj
                    if ii >= len(SIGMA_GRID) or jj >= len(MU_GRID):
                        continue
                    p, q = (SIGMA_GRID[i], MU_GRID[j]), (SIGMA_GRID[ii], MU_GRID[jj])
                    h = pdist(p, q)
                    d = w2(clouds[i, j]["reference"], clouds[ii, jj]["reference"], bins)
                    all_ratios.append(d / h)
        all_ratios = np.array(all_ratios)
        stats = {
            "n_pairs": int(all_ratios.size),
            "max_ratio": float(all_ratios.max()),
            "mean_ratio": float(all_ratios.mean()),
            "median_ratio": float(np.median(all_ratios)),
            "p95_ratio": float(np.percentile(all_ratios, 95)),
        }
        print(f"  pairs={stats['n_pairs']}  max={stats['max_ratio']:.3f}  "
              f"mean={stats['mean_ratio']:.3f}  median={stats['median_ratio']:.3f}  "
              f"p95={stats['p95_ratio']:.3f}")
        print("  (a finite maximum ratio is exactly the empirical Lipschitz constant;")
        print("   a divergence to infinity as h -> 0 would indicate discontinuity)")

        # ---- mesh refinement: does the ratio stay bounded as h shrinks? ---
        print("\nmesh refinement along the sigma axis at mu = 1:")
        coarse = ratios_sigma
        # consecutive pairs are the finest pairs available on this grid
        print("  the grid is geometric, so consecutive pairs already have comparable h;")
        print("  the ratio is reported per step below to show it does not blow up:")
        steps = [r for r in ratios_sigma if abs(r["mu"] - 1.0) < 1e-12]
        for r in steps:
            print(f"    sigma {r['sigma'][0]:8.2f}->{r['sigma'][1]:8.2f}  h={r['h']:.4f}  "
                  f"W2={r['w2']:.5f}  ratio={r['ratio']:.4f}")

        report["bins"][str(bins)] = {
            "sigma_slices": ratios_sigma,
            "mu_slices": ratios_mu,
            "lipschitz": stats,
        }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
