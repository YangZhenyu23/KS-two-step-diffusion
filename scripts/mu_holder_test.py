"""Test whether the dependence on mu is really 1/2-Hoelder or Lipschitz.

Why this script exists
----------------------
``docs/THEORY.md`` claims

    W2(nu_{sigma1,mu1}, nu_{sigma2,mu2}) <= L (|dsigma| + sqrt(|dmu|))

with the exponent 1/2 coming from ``sqrt(2 mu)`` being 1/2-Hoelder at zero. That
derivation does not survive inspection: the synchronous coupling gives

    |sqrt(2 mu1) - sqrt(2 mu2)| = sqrt(2) |mu1 - mu2| / (sqrt(mu1) + sqrt(mu2))

which is *Lipschitz* in mu whenever mu is bounded away from zero; the square root
only bites as mu -> 0. On the grid mu in [0.1, 10] the honest prediction is

    W2 ~ L_mu |dmu|        (linear, not sqrt)

and on log coordinates it is still linear in |d log10 mu|. This matters because
THEORY.md leans on the exponent to argue mu is *intrinsically* harder to resolve
than sigma (threshold ~ (eps/L)^2). If the truth is Lipschitz then the difficulty
of mu is a matter of magnitude (L_mu << L_sigma), not of a different exponent.

What is measured
----------------
For each sigma slice, walk along the mu ladder away from the mu=1 anchor and
regress log(W2 - floor) on log(h) using the smallest h available. The slope is the
empirical exponent:

    slope near 1.0 -> Lipschitz (theory should state |dmu|)
    slope near 0.5 -> 1/2-Hoelder (current THEORY.md claim)

The floor must be subtracted: with a floor of ~0.06 and signals of comparable size
at the smallest step, the raw slope is biased upward.

Reuses clouds already on disk, so no resimulation is needed.

Usage:
    python scripts/mu_holder_test.py
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

OUT = G.RESULTS / "holder"


def cell_cloud(sigma_index: int, mu_index: int) -> np.ndarray:
    return load_cell(G.Cell(sigma_index, mu_index))["clouds"]["reference"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    anchor_index = 5  # mu = 1.0
    rows = []
    for i in range(0, len(G.SIGMA_GRID), 3):
        anchor = cell_cloud(i, anchor_index)
        sig = float(G.SIGMA_GRID[i])
        for j in range(len(G.MU_GRID)):
            if j == anchor_index:
                continue
            mu = float(G.MU_GRID[j])
            h = abs(math.log10(mu / float(G.MU_GRID[anchor_index])))
            scored = score_pair(anchor, cell_cloud(i, j), 16)
            rows.append({"sigma": sig, "mu": mu, "h_log10": h, "w2": float(scored["w2"])})
        print(f"  sigma={sig:<6} done")

    floor = 0.0644
    fits = []
    for sig in sorted({r["sigma"] for r in rows}):
        slice_rows = sorted((r for r in rows if r["sigma"] == sig), key=lambda r: r["h_log10"])
        for k in (3, 5, 8):
            subset = slice_rows[:k]
            if len(subset) < 2:
                continue
            h = np.array([r["h_log10"] for r in subset])
            y = np.array([max(r["w2"] - floor, 1e-6) for r in subset])
            slope, _ = np.polyfit(np.log(h), np.log(y), 1)
            fits.append({"sigma": sig, "n_points": k, "exponent": float(slope)})

    summary = {}
    for k in (3, 5, 8):
        vals = [f["exponent"] for f in fits if f["n_points"] == k]
        if vals:
            summary[str(k)] = {
                "mean_exponent": float(np.mean(vals)),
                "median_exponent": float(np.median(vals)),
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n_slices": len(vals),
            }

    payload = {"floor": floor, "rows": rows, "fits": fits, "summary": summary}
    (OUT / "mu_holder.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== empirical exponent along mu ===")
    for k, v in summary.items():
        print(
            f"  {k} smallest steps: exponent {v['mean_exponent']:.3f} "
            f"(median {v['median_exponent']:.3f}, n={v['n_slices']} slices)"
        )
    print("\nInterpretation: ~1 means Lipschitz in mu, ~0.5 means 1/2-Hoelder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
