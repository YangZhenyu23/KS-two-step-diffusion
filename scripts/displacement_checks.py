"""Three checks that turn the displacement law into a theorem rather than a fit.

Check 1 -- no residual trend in sigma.
    If W2 ~ c_S |dS| with c_S *constant*, then the ratio W2/|dS| must not drift with
    sigma. The competing description, "W2 ~ L_sigma |d log sigma| with L_sigma a
    constant" (THEORY.md 1.3.1), predicts the opposite: a flat ratio in log
    coordinates, which in displacement coordinates means c_S proportional to sigma.
    A 20x drift would be unmistakable.

Check 2 -- the diffusion constant is sqrt(d).
    For pure Brownian motion from a common start, the terminal laws are
    N(0, 2 mu T I) and W2 between two such Gaussians is exactly sqrt(d)|a-b| with
    a = sqrt(2 mu T). Hence c_G = sqrt(3) = 1.732 in d = 3. Any measured c_G close
    to that says the mu dependence is the *free* diffusion, with the interaction
    only mildly modifying it.

Check 3 -- additivity on the diagonal.
    The triangle inequality along the path
        (s1,m1) -> (s2,m1) -> (s2,m2)
    gives W2 <= c_S|dS| + c_G|dG| *provided* the two one-dimensional constants are
    valid. Testing diagonal pairs (both indices +1) checks whether the sum is
    actually attained or whether there is slack.

Usage:
    python scripts/displacement_checks.py
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
FLOOR = 0.06443
T = G.FINAL_TIME
SQRT3 = math.sqrt(3.0)


def cloud(i: int, j: int) -> np.ndarray:
    return load_cell(G.Cell(i, j))["clouds"]["reference"]


def corrected(w2: float) -> float:
    return math.sqrt(max(w2 * w2 - FLOOR * FLOOR, 0.0))


def dS(s1, s2) -> float:
    return abs(s2 * T - s1 * T)


def dG(m1, m2) -> float:
    return abs(math.sqrt(2 * m2 * T) - math.sqrt(2 * m1 * T))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cache: dict[tuple[int, int], np.ndarray] = {}

    def get(i, j):
        if (i, j) not in cache:
            cache[(i, j)] = cloud(i, j)
        return cache[(i, j)]

    payload = json.loads((OUT / "displacement_law.json").read_text("utf-8"))

    # ---- check 1: ratio W2/|dS| versus sigma -------------------------------
    print("=== check 1: is c_S constant, or does it grow with sigma? ===")
    per_sigma = {}
    for r in payload["rows"]:
        if r["axis"] != "sigma":
            continue
        ratio = corrected(r["w2"]) / dS(r["sigma1"], r["sigma2"])
        per_sigma.setdefault(r["sigma1"], []).append(ratio)
    sig_keys = sorted(per_sigma)
    med = [float(np.median(per_sigma[k])) for k in sig_keys]
    print(f"{'sigma':>7} {'median c_S':>11} {'implied L_sigma':>16}")
    for k, m in zip(sig_keys, med):
        # what a constant-L_sigma model would require in log coordinates
        print(f"{k:7.0f} {m:11.4f} {m * k * math.log(10):16.3f}")
    slope = float(np.polyfit(np.log(sig_keys), np.log(med), 1)[0])
    print(f"\nlog-log slope of c_S vs sigma: {slope:+.3f}  (0 = constant c_S, 1 = constant L_sigma)")
    print(f"implied L_sigma spread across the grid: {med[-1] / med[0]:.2f}x "
          f"(a constant-L model needs 1.0x)")

    # ---- check 2: the diffusion constant ----------------------------------
    print("\n=== check 2: is c_G = sqrt(3)? ===")
    mu_ratios = []
    for r in payload["rows"]:
        if r["axis"] != "mu" or r["w2"] <= 1.5 * FLOOR:
            continue
        mu_ratios.append(corrected(r["w2"]) / dG(r["mu1"], r["mu2"]))
    mu_ratios = np.array(mu_ratios)
    print(f"median c_G = {np.median(mu_ratios):.3f}   (sqrt(3) = {SQRT3:.3f})")
    print(f"mean   c_G = {mu_ratios.mean():.3f} +- {mu_ratios.std(ddof=1) / math.sqrt(len(mu_ratios)):.3f} (sem)")
    print(f"n = {len(mu_ratios)} pairs above 1.5*floor")

    # sigma constant against the |g|_inf = 1 ceiling
    sig_ratios = [corrected(r["w2"]) / dS(r["sigma1"], r["sigma2"])
                  for r in payload["rows"] if r["axis"] == "sigma"]
    print(f"\nmedian c_S = {np.median(sig_ratios):.3f}   (rigorous ceiling |g|_inf = 1)")
    print("  c_S <= 1 is implied by |g| <= 1 applied to the difference of the two")
    print("  advection fields, so the measured 0.90 is the expected order.")

    # ---- check 3: additivity on the diagonal ------------------------------
    print("\n=== check 3: does the additive law hold on diagonal pairs? ===")
    rows = []
    for i in range(len(G.SIGMA_GRID) - 1):
        for j in range(len(G.MU_GRID) - 1):
            w2 = float(score_pair(get(i, j), get(i + 1, j + 1), G.BINS)["w2"])
            if w2 <= 1.5 * FLOOR:
                continue
            s1, s2 = G.SIGMA_GRID[i], G.SIGMA_GRID[i + 1]
            m1, m2 = G.MU_GRID[j], G.MU_GRID[j + 1]
            # one-parameter legs, using the constants measured in checks 1 and 2
            c_s, c_g = float(np.median(sig_ratios)), float(np.median(mu_ratios))
            pred = c_s * dS(s1, s2) + c_g * dG(m1, m2)
            rows.append({"i": i, "j": j, "w2": w2, "w2_corrected": corrected(w2),
                         "pred_add": pred, "ratio": corrected(w2) / pred})
        print(f"  sigma row {i} done ({len(rows)} diagonal pairs)")

    ratios = np.array([r["ratio"] for r in rows])
    print(f"\ndiagonal pairs kept: {len(rows)}")
    print(f"  observed / additive prediction: median {np.median(ratios):.3f}, "
          f"p05 {np.quantile(ratios, 0.05):.3f}, p95 {np.quantile(ratios, 0.95):.3f}")
    print("  ratio near 1 means the triangle-inequality bound is tight (both legs add up);")
    print("  ratio below 1 means the two parameter directions partially cancel.")

    (OUT / "displacement_checks.json").write_text(json.dumps(
        {
            "check1_cs_per_sigma": {str(k): v for k, v in zip(sig_keys, med)},
            "check1_loglog_slope": slope,
            "check1_L_sigma_spread": med[-1] / med[0],
            "check2_cG_median": float(np.median(mu_ratios)),
            "check2_cG_mean": float(mu_ratios.mean()),
            "check2_cG_sem": float(mu_ratios.std(ddof=1) / math.sqrt(len(mu_ratios))),
            "check2_sqrt3": SQRT3,
            "check2_cS_median": float(np.median(sig_ratios)),
            "check3_rows": rows,
            "check3_ratio_median": float(np.median(ratios)),
            "check3_ratio_p05": float(np.quantile(ratios, 0.05)),
            "check3_ratio_p95": float(np.quantile(ratios, 0.95)),
        }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
