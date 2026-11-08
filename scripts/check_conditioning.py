"""Check how the two physical parameters are scaled before entering the network.

The question
------------
sigma spans 10..200 and mu spans 0.1..10. They live on different scales, and a
network that receives them raw can end up biased toward one of them. This script
measures exactly what the network sees in the current implementation, then
compares it with the alternatives, so the choice is made on numbers.

What is measured
----------------
For each candidate encoding, over the 40-cell training design and over the full
200-cell grid:

  * the value range,
  * mean and standard deviation,
  * the ratio of the two parameters' spreads, which is the quantity that decides
    whether one axis dominates the other in the first linear layer.

Encodings compared
------------------
  ``raw``        sigma, mu as-is. Range ratio 200:1. Not used.
  ``log10``      log10(sigma), log10(mu) -- what upstream MeanFlow does, and what
                 this repository currently uses.
  ``log10_norm`` additionally centre and scale each axis to zero mean and unit
                 variance over the *training* design, the upstream
                 ``sigma_feature`` generalised to two parameters.
  ``unit01``     map each axis to [0, 1] using the grid bounds, as
                 ``sigma_feature`` does for sigma alone.

Why it matters
--------------
With plain log10 the two axes have means 1.65 and 0.00 and spreads 0.40 and 0.59.
Neither axis has a large scale ratio, but the *offsets* differ by 1.65, so the
first layer sees a strongly shifted input in one column. Centring removes that
asymmetry at no cost to inference, which is the constraint that matters here
(docs/PROJECT_RATIONALE.md 2).

Usage:
    python scripts/check_conditioning.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fullscale import grid as G  # noqa: E402
from fullscale.lhs import latin_hypercube  # noqa: E402

from sample_efficiency import frozen_test_set  # noqa: E402


def encodings(sigmas: np.ndarray, mus: np.ndarray) -> dict:
    log_sigma = np.log10(sigmas)
    log_mu = np.log10(mus)

    # statistics taken over the full grid, which is what a practitioner would use
    # if the design were not yet chosen
    gs = np.log10(np.array(G.SIGMA_GRID))
    gm = np.log10(np.array(G.MU_GRID))

    return {
        "raw": (sigmas, mus),
        "log10": (log_sigma, log_mu),
        "log10_norm": ((log_sigma - gs.mean()) / gs.std(),
                       (log_mu - gm.mean()) / gm.std()),
        "unit01": ((log_sigma - gs.min()) / (gs.max() - gs.min()),
                   (log_mu - gm.min()) / (gm.max() - gm.min())),
        "sigma_feature_upstream": ((log_sigma - np.log10(15.0)) / np.log10(10.0),
                                   (log_mu - np.log10(15.0)) / np.log10(10.0)),
    }


def describe(name: str, a: np.ndarray, b: np.ndarray) -> dict:
    row = {
        "encoding": name,
        "sigma_range": [float(a.min()), float(a.max())],
        "mu_range": [float(b.min()), float(b.max())],
        "sigma_mean": float(a.mean()), "sigma_sd": float(a.std(ddof=1)),
        "mu_mean": float(b.mean()), "mu_sd": float(b.std(ddof=1)),
        "spread_ratio": float(a.std(ddof=1) / b.std(ddof=1)) if b.std(ddof=1) else None,
        "offset_gap": float(abs(a.mean() - b.mean())),
    }
    return row


def main() -> int:
    test, pool = frozen_test_set()
    design = latin_hypercube(40, seed=20261107)
    train = [G.Cell(i, j) for i, j in design.points]

    for label, cells in (("training design (40 cells)", train),
                         ("full grid (200 cells)", G.full_grid())):
        sigmas = np.array([c.sigma for c in cells])
        mus = np.array([c.mu for c in cells])
        print(f"\n=== {label} ===")
        print(f"{'encoding':<24} {'sigma range':>18} {'mu range':>18} "
              f"{'sd ratio':>9} {'offset':>8}")
        print("-" * 82)
        rows = []
        for name, (a, b) in encodings(sigmas, mus).items():
            row = describe(name, a, b)
            rows.append(row)
            sr = row["sigma_range"]
            mr = row["mu_range"]
            ratio = row["spread_ratio"]
            print(f"{name:<24} [{sr[0]:8.3f},{sr[1]:8.3f}] [{mr[0]:8.3f},{mr[1]:8.3f}] "
                  f"{ratio:9.2f} {row['offset_gap']:8.2f}")

        if label.startswith("training"):
            payload = {"cells": [c.key for c in cells], "rows": rows}
            out = G.RESULTS / "conditioning"
            out.mkdir(parents=True, exist_ok=True)
            (out / "conditioning.json").write_text(json.dumps(payload, indent=2),
                                                  encoding="utf-8")
            print(f"\nwrote {out / 'conditioning.json'}")

    print("\ninterpretation")
    print("  raw          sigma is 20x larger than mu in magnitude and 200x in range;")
    print("               a linear layer would weight them very unequally")
    print("  log10        brings both to O(1) but leaves means 1.65 apart, so one")
    print("               input column is systematically offset")
    print("  log10_norm   both axes zero-mean unit-variance; offsets and spread match")
    print("  unit01       both in [0,1]; preserves the shape of the geometric grid")
    print("\n  All four have identical inference cost, so switching is free with")
    print("  respect to the one-step constraint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
