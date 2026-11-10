"""Particle-level W2 for MF+DP clouds, at a user-chosen subsample size.

Purpose
-------
The paper-scale reproduction in ``repro_ks3d/`` already stores, for each sigma:

    reference        the physical truth
    seed{88,89,90}_mf      MeanFlow one-step output
    seed{88,89,90}_mf_dp   MeanFlow + DP refinement

The histogram W2 for every method is already computed. This script adds the
**particle-level** W2 for the MF+DP clouds only, as requested, so the two
regimes can be compared.

Metric
------
Exact optimal transport on a subsample of m points drawn without replacement,
using one index vector per cell that is reused for every seed and every method.
Reuse matters: with a fixed index vector the comparison between seeds and between
methods is not contaminated by drawing different particles.

Cost matrix is m x m float64: 0.75 GiB at m = 10000, which is CPU RAM, not VRAM.
Network simplex on m = 10000 takes about 22 s per solve on this machine.

Usage:
    python scripts/dp_particle_w2.py --subsample 10000 --seeds 88
    python scripts/dp_particle_w2.py --subsample 10000 --seeds 88 89 90
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLOUDS = ROOT / "repro_ks3d" / "evaluation" / "ks" / "generated" / "clouds" / "ks3d"
OUT = ROOT / "results" / "ks3d" / "dp_particle"


def exact_w2(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """Exact W2 between two equal-size point clouds; returns (w2, solver_code)."""
    import ot
    from scipy.spatial.distance import cdist

    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    cost = cdist(a, b, metric="sqeuclidean")
    n = len(a)
    w = np.full(n, 1.0 / n)
    value, info = ot.emd2(w, w, cost, log=True, numItermax=5_000_000)
    code = int(info.get("result_code", -1))
    if code != 1:
        raise RuntimeError(f"exact OT failed: {info.get('warning')}")
    return math.sqrt(max(value, 0.0)), code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample", type=int, default=10000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[88])
    ap.add_argument("--methods", nargs="+", default=["mf_dp"],
                    choices=["mf", "dp", "mf_dp"])
    ap.add_argument("--index-seed", type=int, default=20261110)
    ap.add_argument("--sigma", type=float, nargs="+", default=None,
                    help="restrict to these sigma values")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(CLOUDS.glob("sigma*.npz"),
                   key=lambda p: int(p.stem.replace("sigma", "")))
    if a.sigma:
        wanted = {int(s) for s in a.sigma}
        files = [p for p in files if int(p.stem.replace("sigma", "")) in wanted]
    if not files:
        raise SystemExit(f"no clouds under {CLOUDS}")
    print(f"{len(files)} cells, m={a.subsample}, seeds={a.seeds}, "
          f"methods={a.methods}", flush=True)

    # histogram values already computed, for side-by-side comparison
    hist = {}
    hist_path = ROOT / "results" / "ks3d" / "evaluation" / "ks" / "metrics_per_seed.csv"
    if hist_path.is_file():
        with hist_path.open(encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                if row.get("model") != "ks3d":
                    continue
                key = (int(float(row["level"])), int(row["seed"]), row["method"])
                hist[key] = float(row["w2"])

    rows = []
    started_all = time.perf_counter()
    for position, path in enumerate(files, start=1):
        sigma = int(path.stem.replace("sigma", ""))
        data = np.load(path)
        reference = data["reference"].astype(np.float64)
        rng = np.random.default_rng(a.index_seed + sigma)
        idx = rng.choice(len(reference), size=a.subsample, replace=False)
        ref_sub = reference[idx]

        for method in a.methods:
            for seed in a.seeds:
                key = f"seed{seed}_{method}"
                if key not in data.files:
                    print(f"  sigma={sigma}: missing {key}, skipped", flush=True)
                    continue
                pred = data[key].astype(np.float64)
                t0 = time.perf_counter()
                w2, code = exact_w2(pred[idx], ref_sub)
                dt = time.perf_counter() - t0
                rows.append({
                    "sigma": sigma, "seed": seed, "method": method,
                    "particles_total": len(reference), "subsample": a.subsample,
                    "w2_particle": w2, "solver_code": code,
                    "seconds": round(dt, 2),
                    "w2_histogram": hist.get((sigma, seed, method)),
                })
                elapsed = time.perf_counter() - started_all
                eta = elapsed / len(rows) * (len(files) * len(a.seeds) * len(a.methods)
                                             - len(rows))
                print(f"  [{position}/{len(files)}] sigma={sigma:3d} {key:12s} "
                      f"W2_particle={w2:.5f}  hist="
                      f"{(hist.get((sigma, seed, method)) or float('nan')):.5f}  "
                      f"({dt:.0f}s, eta {eta / 60:.0f} min)", flush=True)
        del data, reference, ref_sub

    # ---- summary ----------------------------------------------------------
    fields = list(rows[0])
    with (OUT / f"dp_particle_m{a.subsample}.csv").open(
            "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_method: dict[str, list[dict]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)

    print(f"\n{'method':<8} {'n':>4} {'particle mean':>14} {'histogram mean':>15} "
          f"{'ratio':>8}")
    summary = {}
    for method, group in by_method.items():
        p = np.array([r["w2_particle"] for r in group])
        h = np.array([r["w2_histogram"] for r in group if r["w2_histogram"]])
        summary[method] = {
            "n": len(group),
            "particle_mean": float(p.mean()),
            "particle_sd": float(p.std(ddof=1)) if len(p) > 1 else 0.0,
            "histogram_mean": float(h.mean()) if h.size else None,
            "ratio_particle_over_hist": (float(p.mean() / h.mean())
                                         if h.size else None),
        }
        s = summary[method]
        print(f"{method:<8} {s['n']:4d} {s['particle_mean']:14.5f} "
              f"{(s['histogram_mean'] or float('nan')):15.5f} "
              f"{(s['ratio_particle_over_hist'] or float('nan')):8.3f}")

    # per-sigma table for the requested method set
    print(f"\n{'sigma':>6} " + " ".join(f"{m + '_p':>10} {m + '_h':>10}"
                                       for m in by_method))
    for sigma in sorted({r["sigma"] for r in rows}):
        cells = []
        for method in by_method:
            group = [r for r in rows if r["sigma"] == sigma and r["method"] == method]
            p = np.mean([r["w2_particle"] for r in group])
            h = [r["w2_histogram"] for r in group if r["w2_histogram"]]
            cells.append(f"{p:10.5f} {(np.mean(h) if h else float('nan')):10.5f}")
        print(f"{sigma:6d} " + " ".join(cells))

    (OUT / f"dp_particle_m{a.subsample}.json").write_text(
        json.dumps({"subsample": a.subsample, "index_seed": a.index_seed,
                    "summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
