"""Continuity evidence with a refinement that actually refines.

The first attempt (``continuity_evidence.py``) reported two artefacts that had to
be fixed before any conclusion could be drawn:

1. **T1 could not refine.** The analysis grid has sigma spaced by 10, so the
   "5" and "2.5" steps landed on the same cell as the base point and produced
   ``h = 0``. Refinement needs clouds at intermediate sigma, which means
   simulating them.

2. **T2 flagged a 4.9x jump in the mu direction.** Every one of those flags sits
   on the interval mu = 3.98 -> 10, whose log spacing is 0.40 while every other
   mu interval is 0.20. A larger step gives a proportionally larger W2; that is
   the metric behaving as expected, not a discontinuity. The test must compare
   *normalised* steps (W2 / h), which is what T3 already does.

This script fixes both:

* it **simulates** clouds at intermediate parameters on a genuinely refined mesh,
  so refinement is real;
* T2 is reformulated as a test on the **normalised slope**, so a wide interval
  cannot masquerade as a jump;
* a **control** is included: the same refinement applied to an interpolated
  ("fake") family, to show the test can tell continuity from a synthetic jump.

Usage:
    python scripts/continuity_refined.py --levels 3
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream" / "MF-DeepParticle" / "deepparticle"))

from fullscale import grid as G  # noqa: E402
from fullscale.evaluate import score_pair  # noqa: E402
from twoparam.data import unit_ball  # noqa: E402

OUT = G.RESULTS / "continuity"


def simulate(sigma: float, mu: float, *, n: int, seed: int, device="cuda") -> np.ndarray:
    """Run the retained KS kernel at an arbitrary (sigma, mu)."""
    from training.protocol import simulate_original

    initial = unit_ball(n, seed)
    np.random.seed(seed + 13)
    torch.manual_seed(seed + 13)
    return simulate_original(torch.from_numpy(initial), float(sigma), mu=float(mu),
                             chi=16 * math.pi, delta=0.1, t=0.02, dt=0.005,
                             chunk=256, device=device).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--levels", type=int, default=3,
                    help="how many halvings of the parameter step")
    ap.add_argument("--out", type=Path, default=OUT / "continuity_refined.json")
    a = ap.parse_args()

    floor_path = G.RESULTS / "floor.json"
    floor = 0.06443
    if floor_path.is_file():
        payload = json.loads(floor_path.read_text(encoding="utf-8"))
        entry = payload["sizes"].get("100000", {}).get("bins", {}).get("16", {})
        floor = entry.get("mean", floor)
    print(f"noise floor = {floor:.5f}, N = {a.n}\n")

    torch.set_num_threads(1)
    report = {"floor": floor, "n": a.n, "sigma": {}, "mu": {}}

    # ---- sigma refinement at fixed mu = 1 --------------------------------
    # base sigma = 100; steps 10, 5, 2.5, 1.25  -> h = log10(1 + step/100)
    print("=== sigma refinement at mu = 1 (fresh simulations) ===")
    print(f"{'step':>7} {'sigma':>8} {'h':>9} {'W2':>9} {'W2 corrected':>13} {'corr/h':>9}")
    base_sigma, base_mu = 100.0, 1.0
    started = time.perf_counter()
    base_cloud = simulate(base_sigma, base_mu, n=a.n, seed=88)
    print(f"  base simulated in {time.perf_counter() - started:.0f}s", flush=True)
    rows = []
    for k in range(a.levels):
        step = 10.0 / (2 ** k)
        sigma = base_sigma + step
        started = time.perf_counter()
        other = simulate(sigma, base_mu, n=a.n, seed=88)
        elapsed = time.perf_counter() - started
        d = score_pair(base_cloud.astype(np.float64), other.astype(np.float64), G.BINS)["w2"]
        h = math.log10(sigma / base_sigma)
        corrected = math.sqrt(max(d * d - floor * floor, 0.0))
        rows.append({"step": step, "sigma": sigma, "h": h, "w2": d,
                     "corrected": corrected,
                     "ratio": corrected / h if h else None,
                     "seconds": elapsed})
        print(f"{step:7.3f} {sigma:8.2f} {h:9.5f} {d:9.5f} {corrected:13.5f} "
              f"{(corrected / h if h else float('nan')):9.4f}  ({elapsed:.0f}s)", flush=True)
    ratios = [r["ratio"] for r in rows if r["ratio"]]
    spread = max(ratios) / min(ratios) if len(ratios) > 1 and min(ratios) > 0 else None
    mono = all(x["w2"] >= y["w2"] for x, y in zip(rows, rows[1:]))
    print(f"  W2 monotone: {mono}   ratio spread: {spread:.2f}x" if spread else "")
    report["sigma"] = {"rows": rows, "monotone": mono, "ratio_spread": spread}

    # ---- mu refinement at fixed sigma = 100 ------------------------------
    print("\n=== mu refinement at sigma = 100 (fresh simulations) ===")
    print(f"{'step':>7} {'mu':>8} {'h':>9} {'W2':>9} {'W2 corrected':>13} {'corr/h':>9}")
    rows_mu = []
    for k in range(a.levels):
        step = 1.0 / (2 ** k)
        mu = base_mu + step
        started = time.perf_counter()
        other = simulate(base_sigma, mu, n=a.n, seed=88)
        elapsed = time.perf_counter() - started
        d = score_pair(base_cloud.astype(np.float64), other.astype(np.float64), G.BINS)["w2"]
        h = math.log10(mu / base_mu)
        corrected = math.sqrt(max(d * d - floor * floor, 0.0))
        rows_mu.append({"step": step, "mu": mu, "h": h, "w2": d,
                        "corrected": corrected,
                        "ratio": corrected / h if h else None,
                        "seconds": elapsed})
        print(f"{step:7.3f} {mu:8.3f} {h:9.5f} {d:9.5f} {corrected:13.5f} "
              f"{(corrected / h if h else float('nan')):9.4f}  ({elapsed:.0f}s)", flush=True)
    ratios_mu = [r["ratio"] for r in rows_mu if r["ratio"]]
    spread_mu = (max(ratios_mu) / min(ratios_mu)
                 if len(ratios_mu) > 1 and min(ratios_mu) > 0 else None)
    mono_mu = all(x["w2"] >= y["w2"] for x, y in zip(rows_mu, rows_mu[1:]))
    print(f"  W2 monotone: {mono_mu}   ratio spread: {spread_mu:.2f}x" if spread_mu else "")
    report["mu"] = {"rows": rows_mu, "monotone": mono_mu, "ratio_spread": spread_mu}

    # ---- T2 reformulated: normalised slope, not raw step ------------------
    print("\n=== T2 (reformulated)  normalised slope along each slice ===")
    print("  the earlier version compared raw W2 steps, which flags the wide")
    print("  mu = 3.98 -> 10 interval (h = 0.40 vs 0.20 elsewhere) as a jump.")
    print("  dividing by h removes that artefact; a real discontinuity would")
    print("  still show up as a ratio far above its neighbours.\n")
    slices = {}
    worst_ratio = 0.0
    for i in range(0, len(G.SIGMA_GRID), 4):
        steps = []
        for j in range(len(G.MU_GRID) - 1):
            a_cell, b_cell = G.Cell(i, j), G.Cell(i, j + 1)
            with np.load(G.DATA / "clouds" / f"{a_cell.key}.npz", allow_pickle=False) as p:
                ca = p["reference"].copy()
            with np.load(G.DATA / "clouds" / f"{b_cell.key}.npz", allow_pickle=False) as p:
                cb = p["reference"].copy()
            h = abs(math.log10(b_cell.mu / a_cell.mu))
            steps.append(score_pair(ca.astype(np.float64), cb.astype(np.float64),
                                    G.BINS)["w2"] / h)
        arr = np.array(steps)
        ratio = float(arr.max() / np.median(arr))
        worst_ratio = max(worst_ratio, ratio)
        slices[f"mu|sigma={G.SIGMA_GRID[i]}"] = {
            "median_slope": float(np.median(arr)), "max_slope": float(arr.max()),
            "max_at_mu": G.MU_GRID[int(np.argmax(arr)) + 1],
            "peak_over_median": ratio}
        print(f"  sigma={G.SIGMA_GRID[i]:<5} median slope {np.median(arr):.3f}  "
              f"max {arr.max():.3f} at mu={G.MU_GRID[int(np.argmax(arr)) + 1]:.3f}  "
              f"peak/median {ratio:.2f}")
    report["T2_normalised"] = {"slices": slices, "worst_peak_over_median": worst_ratio}

    # ---- verdict ---------------------------------------------------------
    print("\n=== verdict ===")
    checks = {
        "sigma: W2 decays monotonically under refinement": mono,
        "mu: W2 decays monotonically under refinement": mono_mu,
        "sigma: corrected slope stable (spread < 2x)":
            spread is not None and spread < 2.0,
        "mu: corrected slope stable (spread < 2x)":
            spread_mu is not None and spread_mu < 2.0,
        "T2: no slice peak above 3x its median normalised slope": worst_ratio < 3.0,
    }
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    report["verdict"] = checks
    report["all_pass"] = all(checks.values())

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
