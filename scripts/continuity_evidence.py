"""Complete evidence package for parameter continuity, from data alone.

This script answers one question with measurements rather than assertions:
**do the particle clouds depend continuously on (sigma, mu)?**

Four independent tests, each able to falsify continuity on its own:

T1  **Monotone decay under parameter refinement.** Halve the parameter step twice
    at a fixed point. If the family is discontinuous, W2 stays bounded away from
    zero; if continuous, W2 falls monotonically toward the noise floor.

T2  **No jump along dense slices.** Walk along sigma at several fixed mu, and
    along mu at several fixed sigma, and look for a large step relative to the
    local trend. A discontinuity would appear as one interval whose W2 is far
    above its neighbours.

T3  **Bounded normalised slope.** For every adjacent pair, W2 divided by the log
    parameter distance. Continuity (Lipschitz) means this ratio stays bounded;
    a singularity would make it diverge as the step shrinks.

T4  **Lipschitz constant stability under refinement.** Estimate the maximum
    normalised slope at the full-grid spacing and at a doubled spacing. A true
    Lipschitz constant should not grow as the mesh is refined.

The script also separates *signal* from *sampling noise*, because two clouds from
the same parameter already differ by a measurable W2. Without that subtraction a
continuous family looks "stuck" at the noise floor and can be mistaken for
discontinuous.

Usage:
    python scripts/continuity_evidence.py
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
from fullscale.evaluate import score_pair  # noqa: E402

OUT = G.RESULTS / "continuity"


def cloud(cell: G.Cell) -> np.ndarray:
    path = G.DATA / "clouds" / f"{cell.key}.npz"
    with np.load(path, allow_pickle=False) as payload:
        return payload["reference"].copy()


def w2(a: G.Cell, b: G.Cell) -> float:
    return score_pair(cloud(a), cloud(b), G.BINS)["w2"]


def log_distance(a: G.Cell, b: G.Cell) -> float:
    return math.hypot(math.log10(a.sigma / b.sigma), math.log10(a.mu / b.mu))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    floor_path = G.RESULTS / "fullscale" / "floor.json"
    floor = 0.06443
    if floor_path.is_file():
        payload = json.loads(floor_path.read_text(encoding="utf-8"))
        entry = payload["sizes"].get("100000", {}).get("bins", {}).get("16", {})
        floor = entry.get("mean", floor)
    print(f"noise floor (100k particles, 16 bins) = {floor:.5f}\n")

    report = {"floor": floor, "tests": {}}

    # ---- T1: refinement at a fixed point ----------------------------------
    print("=== T1  refinement at (sigma=100, mu=1) ===")
    print(f"{'sigma step':>11} {'h':>9} {'W2':>9} {'W2 corrected':>13} {'corr/h':>9}")
    base = G.Cell(G.SIGMA_GRID.index(100.0), G.MU_INDEX)
    rows_t1 = []
    for step in (20.0, 10.0, 5.0, 2.5):
        target_sigma = base.sigma + step
        # nearest grid sigma to the requested value
        i = min(range(len(G.SIGMA_GRID)),
                key=lambda k: abs(G.SIGMA_GRID[k] - target_sigma))
        other = G.Cell(i, base.j)
        h = abs(math.log10(other.sigma / base.sigma))
        d = w2(base, other)
        corrected = math.sqrt(max(d * d - floor * floor, 0.0))
        rows_t1.append({"step": step, "sigma": other.sigma, "h": h, "w2": d,
                        "corrected": corrected,
                        "ratio": corrected / h if h else None})
        print(f"{step:11.1f} {h:9.5f} {d:9.5f} {corrected:13.5f} "
              f"{(corrected / h if h else float('nan')):9.4f}")
    mono = all(a["w2"] >= b["w2"] for a, b in zip(rows_t1, rows_t1[1:]))
    print(f"  W2 monotone decreasing as the step shrinks: {mono}")
    report["tests"]["T1_refinement"] = {"rows": rows_t1, "monotone": mono}

    # ---- T2: dense slices, looking for jumps ------------------------------
    print("\n=== T2  dense slices (look for a single large jump) ===")
    slice_reports = {}
    for label, mu_levels in (("sigma slices", [0.1, 1.0, 10.0]),
                             ("mu slices", None)):
        if mu_levels is not None:
            for mu in mu_levels:
                j = G.MU_GRID.index(mu)
                steps = []
                for i in range(len(G.SIGMA_GRID) - 1):
                    a, b = G.Cell(i, j), G.Cell(i + 1, j)
                    steps.append(w2(a, b))
                median = float(np.median(steps))
                peak = float(np.max(steps))
                peak_index = int(np.argmax(steps))
                ratio = peak / median if median else float("inf")
                slice_reports[f"sigma|mu={mu}"] = {
                    "median_step": median, "max_step": peak,
                    "max_at_sigma": G.SIGMA_GRID[peak_index],
                    "peak_over_median": ratio, "steps": steps}
                print(f"  mu={mu:<6} step W2 median {median:.5f}  max {peak:.5f} "
                      f"at sigma={G.SIGMA_GRID[peak_index]:.0f}  "
                      f"peak/median {ratio:.2f}")
        else:
            for sigma in (10.0, 100.0, 200.0):
                i = G.SIGMA_GRID.index(sigma)
                steps = []
                for j in range(len(G.MU_GRID) - 1):
                    a, b = G.Cell(i, j), G.Cell(i, j + 1)
                    steps.append(w2(a, b))
                median = float(np.median(steps))
                peak = float(np.max(steps))
                peak_index = int(np.argmax(steps))
                ratio = peak / median if median else float("inf")
                slice_reports[f"mu|sigma={sigma}"] = {
                    "median_step": median, "max_step": peak,
                    "max_at_mu": G.MU_GRID[peak_index],
                    "peak_over_median": ratio, "steps": steps}
                print(f"  sigma={sigma:<5} step W2 median {median:.5f}  max {peak:.5f} "
                      f"at mu={G.MU_GRID[peak_index]:.3f}  "
                      f"peak/median {ratio:.2f}")
    report["tests"]["T2_slices"] = slice_reports

    # ---- T3: normalised slope over all adjacent pairs ---------------------
    print("\n=== T3  normalised slope W2/h over all adjacent pairs ===")
    ratios = {"sigma": [], "mu": []}
    for i in range(len(G.SIGMA_GRID)):
        for j in range(len(G.MU_GRID)):
            if i + 1 < len(G.SIGMA_GRID):
                a, b = G.Cell(i, j), G.Cell(i + 1, j)
                h = log_distance(a, b)
                ratios["sigma"].append(w2(a, b) / h)
            if j + 1 < len(G.MU_GRID):
                a, b = G.Cell(i, j), G.Cell(i, j + 1)
                h = log_distance(a, b)
                ratios["mu"].append(w2(a, b) / h)
    t3 = {}
    for key, values in ratios.items():
        arr = np.array(values)
        t3[key] = {"n": int(arr.size), "max": float(arr.max()),
                   "median": float(np.median(arr)),
                   "p95": float(np.percentile(arr, 95)),
                   "mean": float(arr.mean())}
        print(f"  {key}: n={arr.size}  max {arr.max():.3f}  median "
              f"{np.median(arr):.3f}  p95 {np.percentile(arr, 95):.3f}")
    print("  a finite maximum is exactly the empirical Lipschitz constant;")
    print("  divergence as h shrinks would indicate a discontinuity")
    report["tests"]["T3_slopes"] = t3

    # ---- T4: is the Lipschitz constant stable under refinement? -----------
    print("\n=== T4  Lipschitz constant at two mesh spacings ===")
    fine, coarse = [], []
    for i in range(len(G.SIGMA_GRID) - 1):
        a, b = G.Cell(i, G.MU_INDEX), G.Cell(i + 1, G.MU_INDEX)
        fine.append(w2(a, b) / log_distance(a, b))
    for i in range(len(G.SIGMA_GRID) - 2):
        a, b = G.Cell(i, G.MU_INDEX), G.Cell(i + 2, G.MU_INDEX)
        coarse.append(w2(a, b) / log_distance(a, b))
    fine_max, coarse_max = float(np.max(fine)), float(np.max(coarse))
    print(f"  adjacent spacing (h):        max ratio {fine_max:.3f}")
    print(f"  doubled spacing (2h):        max ratio {coarse_max:.3f}")
    print(f"  coarse / fine = {coarse_max / fine_max:.3f}")
    print("  a value near or below 1 means the constant does not blow up under")
    print("  refinement, which is what continuity requires")
    report["tests"]["T4_lipschitz"] = {
        "fine_max": fine_max, "coarse_max": coarse_max,
        "ratio": coarse_max / fine_max}

    # ---- verdict ---------------------------------------------------------
    print("\n=== verdict ===")
    checks = {
        "T1 W2 decays monotonically under refinement": mono,
        "T2 no slice has a step above 4x its median": all(
            v["peak_over_median"] < 4.0 for v in slice_reports.values()),
        "T3 all normalised slopes bounded (< 20)": (
            t3["sigma"]["max"] < 20 and t3["mu"]["max"] < 20),
        "T4 Lipschitz constant stable under refinement (ratio < 1.5)": (
            coarse_max / fine_max < 1.5),
    }
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    report["verdict"] = checks
    report["all_pass"] = all(checks.values())

    (OUT / "continuity_evidence.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'continuity_evidence.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
