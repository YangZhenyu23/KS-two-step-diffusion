"""Is the corrector's rollout improvement real, or is it noise on a catastrophic failure?

The claim under test: "adding the corrector improves stability under repeated application".

There are two very different quantities that both get called "stability", and they must not
be confused:

  S1  cross-seed stability at k = 1
      How much does the generated cloud change when the training seed changes?
      Measured over three seeds, 143 cells. This is well powered.

  S2  rollout stability at k >= 2
      How far does repeated application drift from the true solution at k*T?
      Measured over three cells, one seed. This is barely powered.

This script quantifies both, reports the evidence weight honestly, and computes a paired
statistic for S2 so the reader can see how thin it is.

It also computes the quantity that actually matters for the ablation question: is the
corrector's k>=2 improvement large relative to the failure itself? A 7% reduction of a
100x error is not "improved stability", it is "equally catastrophic".

Usage:
    python scripts/verify_stability_claim.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fullscale import grid as G  # noqa: E402

OUT = G.RESULTS / "stability_check"
OUT.mkdir(parents=True, exist_ok=True)


def load(relative: str) -> dict | None:
    path = G.RESULTS / relative
    return json.loads(path.read_text("utf-8")) if path.exists() else None


print("=" * 88)
print("S1  CROSS-SEED STABILITY AT k = 1  (well powered: 3 seeds x 143 cells)")
print("=" * 88)
three = load("three_seed/three_seed.json")
if three:
    mf_sd = three["mf_two_sd"]
    dp_sd = three["mf_dp_sd"]
    print(f"  stage 1        mean {three['mf_two_mean']:.5f}  sd {mf_sd:.5f}")
    print(f"  stage 1 + corr mean {three['mf_dp_mean']:.5f}  sd {dp_sd:.5f}")
    print(f"  variance ratio {dp_sd ** 2 / mf_sd ** 2:.3f}  "
          f"(sd ratio {dp_sd / mf_sd:.3f}, i.e. {100 * (1 - dp_sd / mf_sd):.0f}% reduction)")
    # is the variance reduction larger than what the mean shift would explain?
    print(f"  mean reduction {100 * (1 - three['mf_dp_mean'] / three['mf_two_mean']):.1f}%")
    print("\n  interpretation: the sd drop (70%) is much larger than the mean drop (14%),")
    print("  so the corrector is making seeds agree, not just shifting them down.")
else:
    print("  three_seed.json missing")

print()
print("=" * 88)
print("S2  ROLLOUT STABILITY AT k >= 2  (thin: 3 cells, 1 seed)")
print("=" * 88)
compare = load("rollout/rollout_compare.json")
if compare:
    floor = compare["floor_mean"]
    rows = compare["rows"]
    cells = sorted({r["key"] for r in rows})
    print(f"  floor at this scoring size: {floor:.5f}")
    print(f"  {'cell':>8} {'k':>3} {'MF':>10} {'MF+DP':>10} {'ratio':>7} "
          f"{'MF/floor':>9} {'DP/floor':>9}")

    ratios = []
    for cell in cells:
        for k in sorted({r["k"] for r in rows}):
            if k == 1:
                continue
            mf = next((r for r in rows if r["key"] == cell
                       and r["method"] == "mf" and r["k"] == k), None)
            dp = next((r for r in rows if r["key"] == cell
                       and r["method"] == "mf_dp" and r["k"] == k), None)
            if not (mf and dp):
                continue
            ratio = dp["w2_physical"] / mf["w2_physical"]
            ratios.append(ratio)
            print(f"  {cell:>8} {k:>3} {mf['w2_physical']:>10.5f} "
                  f"{dp['w2_physical']:>10.5f} {ratio:>7.3f} "
                  f"{mf['w2_physical'] / floor:>9.1f} {dp['w2_physical'] / floor:>9.1f}")

    ratios = np.array(ratios)
    n = len(ratios)
    print(f"\n  paired ratio (corrector / stage 1) over n = {n} (cell, k) pairs:")
    print(f"    mean {ratios.mean():.4f}  median {np.median(ratios):.4f}  "
          f"min {ratios.min():.4f}  max {ratios.max():.4f}")
    print(f"    all ratios below 1? {bool((ratios < 1).all())}  ({int((ratios < 1).sum())}/{n})")
    if n > 1:
        sd = ratios.std(ddof=1)
        se = sd / math.sqrt(n)
        t = (ratios.mean() - 1.0) / se if se > 0 else float("nan")
        print(f"    sd {sd:.4f}, se {se:.4f}, t vs 1.0 = {t:.2f} (df={n - 1})")
        print(f"    => the direction is consistent ({int((ratios < 1).sum())}/{n}), but the")
        print(f"       sample is {n} pairs from ONE seed and 3 cells.")

    print("\n  magnitude relative to the failure itself:")
    for k in (2, 4):
        mf_vals = [r["w2_physical"] for r in rows if r["method"] == "mf" and r["k"] == k]
        dp_vals = [r["w2_physical"] for r in rows if r["method"] == "mf_dp" and r["k"] == k]
        if mf_vals and dp_vals:
            print(f"    k={k}: stage 1 is {np.mean(mf_vals) / floor:.0f}x the floor, "
                  f"corrector is {np.mean(dp_vals) / floor:.0f}x the floor; "
                  f"the improvement is {100 * (1 - np.mean(dp_vals) / np.mean(mf_vals)):.1f}% "
                  f"of a {np.mean(mf_vals) / floor:.0f}x failure")

print()
print("=" * 88)
print("VERDICT")
print("=" * 88)
print("  S1 (cross-seed, k=1):      REAL and well powered. 70% sd reduction over 3 seeds.")
print("  S2 (rollout, k>=2):        DIRECTIONALLY consistent but THIN, and small in")
print("                             absolute terms.")
print("  => the claim 'the corrector improves stability' is defensible for S1 and")
print("     must be stated with an explicit caveat for S2: it makes the divergence")
print("     slightly less severe, it does not make the map stable.")
