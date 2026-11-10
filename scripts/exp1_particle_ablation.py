"""Experiment 1: two parameters versus one, measured with particle-level OT.

Only the *metric* changes relative to the histogram-based ablation. Nothing is
retrained and no simulation is repeated: the 200-cell dataset already stores both
``source`` and ``reference`` for every cell, and both models are on disk.

Two comparisons are produced per cell
-------------------------------------
``error_hist``  and ``error_particle`` for both models, so the improvement can be
quoted in either metric. The point of the exercise is to check whether the
headline +38.4% survives the change of ruler.

Baseline for the ratio
----------------------
The particle-level floor depends on the parameter point, and measuring it needs an
extra independent simulation (~83 s per point). Five points were already measured
across the box. Because the floor varies smoothly with the parameters and far less
than the model error does, each cell is assigned the floor of the nearest measured
point in log-parameter space. The script reports how far each cell is from its
assigned floor point so the approximation is visible, and it labels the assignment
so a later full-grid measurement can replace it without redoing the work.

Paired subsampling
------------------
The same index vector is reused for the model and the reference, so the comparison
is not contaminated by drawing different particles.

Usage:
    python scripts/exp1_particle_ablation.py --subsample 3000
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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream" / "MF-DeepParticle" / "deepparticle"))

from fullscale import grid as G  # noqa: E402
from fullscale.evaluate import score_pair  # noqa: E402
from fullscale.train import load_cell, load_one, load_two  # noqa: E402
from fullscale.model import sample_one_param, sample_two_param  # noqa: E402

OUT = G.RESULTS / "particle_ablation"


def exact_w2(a, b) -> float:
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
    if int(info.get("result_code", -1)) != 1:
        raise RuntimeError(f"OT failed: {info.get('warning')}")
    return math.sqrt(max(value, 0.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample", type=int, default=3000)
    ap.add_argument("--index-draws", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0,
                    help="evaluate only the first N held-out cells (0 = all)")
    a = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda t: print(f"[{time.strftime('%H:%M:%S')}] {t}", flush=True)

    # ---- held-out cells: same set as the histogram ablation -----------------
    detail = G.RESULTS / "optimal_budget_ablation" / "detail.csv"
    with detail.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    cells = [G.Cell(int(r["i"]), int(r["j"])) for r in rows]
    hist = {f"{r['key']}": (float(r["w2_one_param"]), float(r["w2_two_param"]))
            for r in rows}
    if a.limit:
        cells = cells[: a.limit]
    log(f"{len(cells)} held-out cells")

    # ---- measured particle floors, five points -----------------------------
    floor_path = G.RESULTS / "particle_metric" / "particle_floor_full.json"
    floor_rows = json.loads(floor_path.read_text(encoding="utf-8"))["rows"]
    floor_points = [(r["sigma"], r["mu"], r["floor_mean"]) for r in floor_rows]
    log(f"particle floors measured at {len(floor_points)} points")

    def nearest_floor(cell) -> tuple[float, float]:
        """Floor of the nearest measured point in log-parameter space, plus distance."""
        best, best_d = None, float("inf")
        for sigma, mu, value in floor_points:
            d = math.hypot(math.log10(cell.sigma / sigma), math.log10(cell.mu / mu))
            if d < best_d:
                best, best_d = value, d
        return best, best_d

    # ---- models ------------------------------------------------------------
    two, _ = load_two(G.RESULTS / "optimal_budget_ablation" / "two_32000"
                      / "mf_two_param.pt")
    one, _ = load_one(G.RESULTS / "optimal_budget_ablation" / "one_32000"
                      / "mf_one_param.pt")
    rng = np.random.default_rng(707070)
    indices = [rng.choice(G.N_PARTICLES, size=a.subsample, replace=False)
               for _ in range(a.index_draws)]

    results = []
    started_all = time.perf_counter()
    for position, cell in enumerate(cells, start=1):
        payload = load_cell(cell)
        reference = payload["clouds"]["reference"].astype(np.float64)
        source = payload["clouds"]["source"]

        pred_two = sample_two_param(two, torch.from_numpy(source),
                                    cell.sigma, cell.mu).numpy().astype(np.float64)
        pred_one = sample_one_param(one, torch.from_numpy(source),
                                    cell.sigma).numpy().astype(np.float64)

        err_two = [exact_w2(pred_two[idx], reference[idx]) for idx in indices]
        err_one = [exact_w2(pred_one[idx], reference[idx]) for idx in indices]
        h_one, h_two = hist[cell.key]
        floor, dist = nearest_floor(cell)

        row = {
            "key": cell.key, "i": cell.i, "j": cell.j,
            "sigma": cell.sigma, "mu": cell.mu,
            "error_particle_two": float(np.mean(err_two)),
            "error_particle_one": float(np.mean(err_one)),
            "particle_sd_two": float(np.std(err_two, ddof=1)) if len(err_two) > 1 else 0.0,
            "error_hist_two": h_two,
            "error_hist_one": h_one,
            "floor_particle": floor,
            "floor_log_distance": dist,
            "ratio_two": float(np.mean(err_two)) / floor,
            "ratio_one": float(np.mean(err_one)) / floor,
            "improvement_particle": float(np.mean(err_one) - np.mean(err_two)),
            "improvement_hist": h_one - h_two,
            "relative_particle": 1 - float(np.mean(err_two)) / float(np.mean(err_one)),
            "relative_hist": 1 - h_two / h_one,
        }
        results.append(row)
        if position % 10 == 0 or position == len(cells):
            elapsed = time.perf_counter() - started_all
            eta = elapsed / position * (len(cells) - position)
            log(f"  {position}/{len(cells)}  {cell.key}  "
                f"two {row['error_particle_two']:.4f} one {row['error_particle_one']:.4f}  "
                f"rel {row['relative_particle']:+.1%}  eta {eta / 60:.0f} min")
        del pred_two, pred_one
        torch.cuda.empty_cache()

    # ---- summary -----------------------------------------------------------
    e_two = np.array([r["error_particle_two"] for r in results])
    e_one = np.array([r["error_particle_one"] for r in results])
    h_two = np.array([r["error_hist_two"] for r in results])
    h_one = np.array([r["error_hist_one"] for r in results])
    ratios_two = np.array([r["ratio_two"] for r in results])
    dists = np.array([r["floor_log_distance"] for r in results])

    summary = {
        "subsample": a.subsample, "index_draws": a.index_draws,
        "n_cells": len(results),
        "particle_two_mean": float(e_two.mean()),
        "particle_one_mean": float(e_one.mean()),
        "particle_relative_reduction": float(1 - e_two.mean() / e_one.mean()),
        "particle_two_wins": int((e_one > e_two).sum()),
        "hist_two_mean": float(h_two.mean()),
        "hist_one_mean": float(h_one.mean()),
        "hist_relative_reduction": float(1 - h_two.mean() / h_one.mean()),
        "hist_two_wins": int((h_one > h_two).sum()),
        "ratio_two_mean": float(ratios_two.mean()),
        "ratio_two_min": float(ratios_two.min()),
        "ratio_two_max": float(ratios_two.max()),
        "floor_distance_max": float(dists.max()),
        "floor_distance_mean": float(dists.mean()),
    }

    print(f"\n{'':<22} {'two-param':>11} {'one-param':>11} {'relative':>10} {'wins':>9}")
    print(f"{'particle level':<22} {e_two.mean():11.5f} {e_one.mean():11.5f} "
          f"{summary['particle_relative_reduction']:+10.1%} "
          f"{summary['particle_two_wins']:>4}/{len(results)}")
    print(f"{'16-bin histogram':<22} {h_two.mean():11.5f} {h_one.mean():11.5f} "
          f"{summary['hist_relative_reduction']:+10.1%} "
          f"{summary['hist_two_wins']:>4}/{len(results)}")
    print(f"\ntwo-param error / particle floor: {ratios_two.min():.2f} .. "
          f"{ratios_two.max():.2f} (mean {ratios_two.mean():.2f})")
    print(f"floor assignment: each cell used the nearest of {len(floor_points)} "
          f"measured points, max log-distance {dists.max():.2f} "
          f"(mean {dists.mean():.2f})")
    print("\nthe floor assignment is an approximation: cells far from any measured")
    print("point may have a floor that differs from the one used. The distance")
    print("column in the CSV identifies those cells for a follow-up measurement.")

    with (OUT / f"exp1_particle_ablation_m{a.subsample}.csv").open(
            "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (OUT / f"exp1_particle_ablation_m{a.subsample}.json").write_text(
        json.dumps({"summary": summary, "rows": results}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
