"""Is the parameter dependence a power law, or a law in *displacement* coordinates?

Motivation
----------
``docs/THEORY.md`` states

    W2(nu_theta1, nu_theta2) <= L (|dsigma| + sqrt(|dmu|))          (claim A)

with the 1/2 exponent attributed to sqrt(2 mu) being 1/2-Hoelder at zero. That step
is wrong: the synchronous coupling gives

    |sqrt(2 m1) - sqrt(2 m2)| = sqrt(2)|m1-m2| / (sqrt(m1)+sqrt(m2)) <= C|m1-m2|

on any closed interval with m >= m_min > 0. The exponent is 1 (Lipschitz); the 1/2
only appears in the degenerate limit m_min -> 0, which is not the regime here.

But there is a better statement available. The two physical parameters enter the SDE
only through *displacement scales* accumulated over [0,T]:

    shear      S = sigma * T            (advection distance of g, |g| <= 1)
    diffusion  G = sqrt(2 mu T)         (one-sigma Brownian displacement)

so the dimensionally natural law is

    W2 ~ |dS| + |dG|                                                (claim B)

This script compares claims A and B on the 200-cell grid. For every adjacent pair
we compute the observed W2 (floor-subtracted in quadrature) and the two candidate
predictors, then report

  * the ratio  observed / predictor, which should be *constant* for a correct law;
  * the coefficient of variation of that ratio, which is the single number that
    decides which law is right.

Claim B also predicts the strong anisotropy seen in the data: on log coordinates
dS/dlog10(sigma) = S ln10 is *proportional to sigma* (a 20x spread across the grid),
and dG/dlog10(mu) = G ln10 / 2 is proportional to sqrt(mu) (a 3.2x spread). A single
"median Lipschitz constant" therefore hides an order of magnitude, which is exactly
what a 1/2-Hoelder story cannot explain.

Usage:
    python scripts/displacement_law.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream" / "MF-DeepParticle" / "deepparticle"))

from fullscale import grid as G  # noqa: E402
from fullscale.evaluate import score_pair  # noqa: E402
from fullscale.train import load_cell  # noqa: E402

OUT = G.RESULTS / "displacement"
FLOOR = 0.06443  # 16-bin, 100k particles (results/fullscale/floor.json)
T = G.FINAL_TIME


def cloud(i: int, j: int) -> np.ndarray:
    return load_cell(G.Cell(i, j))["clouds"]["reference"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cache: dict[tuple[int, int], np.ndarray] = {}

    def get(i, j):
        if (i, j) not in cache:
            cache[(i, j)] = cloud(i, j)
        return cache[(i, j)]

    rows = []
    for i in range(len(G.SIGMA_GRID)):
        for j in range(len(G.MU_GRID)):
            s1, m1 = G.SIGMA_GRID[i], G.MU_GRID[j]
            for di, dj, axis in ((1, 0, "sigma"), (0, 1, "mu")):
                ii, jj = i + di, j + dj
                if ii >= len(G.SIGMA_GRID) or jj >= len(G.MU_GRID):
                    continue
                s2, m2 = G.SIGMA_GRID[ii], G.MU_GRID[jj]
                w2 = float(score_pair(get(i, j), get(ii, jj), G.BINS)["w2"])
                w2c = math.sqrt(max(w2 * w2 - FLOOR * FLOOR, 0.0))

                pred_a = abs(s2 - s1) + math.sqrt(abs(m2 - m1))
                pred_b = abs(s2 * T - s1 * T) + abs(math.sqrt(2 * m2 * T) - math.sqrt(2 * m1 * T))
                rows.append(
                    {
                        "axis": axis, "sigma1": s1, "mu1": m1, "sigma2": s2, "mu2": m2,
                        "w2": w2, "w2_corrected": w2c,
                        "pred_A_holder": pred_a, "pred_B_displacement": pred_b,
                        "ratio_A": w2c / pred_a if pred_a > 0 else None,
                        "ratio_B": w2c / pred_b if pred_b > 0 else None,
                    }
                )
        print(f"  sigma={G.SIGMA_GRID[i]:<6} row done ({len(rows)} pairs)")

    summary = {}
    for axis in ("sigma", "mu"):
        for key, label in (("ratio_A", "A_holder"), ("ratio_B", "B_displacement")):
            vals = np.array([r[key] for r in rows if r["axis"] == axis and r[key] is not None])
            summary[f"{axis}|{label}"] = {
                "n": int(len(vals)),
                "mean": float(vals.mean()),
                "median": float(np.median(vals)),
                "sd": float(vals.std(ddof=1)),
                "cv": float(vals.std(ddof=1) / vals.mean()),  # coefficient of variation
                "q05": float(np.quantile(vals, 0.05)),
                "q95": float(np.quantile(vals, 0.95)),
                "p95_over_p05": float(np.quantile(vals, 0.95) / max(np.quantile(vals, 0.05), 1e-12)),
            }

    # How much does the *predicted* log-coordinate slope vary across the grid?
    # This is the part a single median Lipschitz constant cannot represent.
    slope_pred = {
        "dS_dlog10sigma": [float(s * T * math.log(10)) for s in G.SIGMA_GRID],
        "dG_dlog10mu": [float(math.sqrt(2 * m * T) * math.log(10) / 2) for m in G.MU_GRID],
    }
    slope_spread = {
        "sigma_max_over_min": max(slope_pred["dS_dlog10sigma"]) / min(slope_pred["dS_dlog10sigma"]),
        "mu_max_over_min": max(slope_pred["dG_dlog10mu"]) / min(slope_pred["dG_dlog10mu"]),
    }

    payload = {
        "floor": FLOOR, "T": T, "rows": rows,
        "summary": summary, "slope_prediction": slope_pred, "slope_spread": slope_spread,
    }
    (OUT / "displacement_law.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== which law has a stable constant? ===")
    print(f"{'axis|law':<28} {'median':>9} {'cv':>8} {'p95/p05':>9}")
    for k, v in summary.items():
        print(f"{k:<28} {v['median']:9.4f} {v['cv']:8.3f} {v['p95_over_p05']:9.2f}")
    print("\n(cv = sd/mean of observed/predicted. A correct law has the smaller cv.)")
    print("\n=== predicted log-coordinate slope spread ===")
    print(f"  dS/dlog10(sigma): {slope_spread['sigma_max_over_min']:.1f}x across the grid")
    print(f"  dG/dlog10(mu)   : {slope_spread['mu_max_over_min']:.1f}x across the grid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
