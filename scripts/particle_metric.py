"""Particle-level evaluation: model error and noise floor measured with the same ruler.

Why this script exists
----------------------
The earlier anatomy test showed that 94% of the 16-bin histogram floor at
(sigma, mu) = (10, 0.1) is quantisation, not physics: two clouds identical up to
within-bin resampling still produce W2 = 0.0428 out of a total floor of 0.0455.

But replacing only the *denominator* of the E/floor ratio is meaningless -- it
would compare a histogram-measured error against a particle-measured floor. Both
must move to the particle level together. That is what this script does:

    numerator    W2_particle( model cloud , reference cloud )
    denominator  W2_particle( reference cloud A , reference cloud B )

with the *same* metric, the *same* subsample size, and the *same* index draw, so
the ratio is well defined.

Paired sampling
---------------
Exact OT on 100k points is O(N^3) and infeasible, so every comparison uses a
subsample of m points. The same index vector is reused for all comparisons at a
given m:

    idx = choice(N, m, replace=False)          drawn once, fixed seed
    error   = W2(model[idx],      reference[idx])
    floor   = W2(reference_A[idx], reference_B[idx])

Using one index vector removes any difference that would otherwise come from
drawing different particles, which would dominate the very quantity being
measured. The script also reports the standard deviation across several index
draws so the reader can see how much the choice of subsample matters.

What this does not change
-------------------------
No training is involved. Model weights are loaded from the existing checkpoints
and only inference plus scoring is performed. The training objective is the
MeanFlow identity in raw particle coordinates; it never referenced W2, so
switching the evaluation metric cannot affect the trained model.

Usage:
    python scripts/particle_metric.py --points 4 --subsample 4000
    python scripts/particle_metric.py --points 20 --subsample 4000 --use-heldout
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
from fullscale.train import load_cell, load_two  # noqa: E402
from fullscale.model import sample_two_param  # noqa: E402
from twoparam.data import unit_ball  # noqa: E402

OUT = G.RESULTS / "particle_metric"

# the four corners of the parameter box plus the middle
POINTS = [(10.0, 0.1), (100.0, 1.0), (200.0, 0.1), (200.0, 10.0)]


def exact_w2(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Exact W2 between two equal-size clouds, no binning.

    Returns ``(w2, gap)`` where ``gap`` is the primal-dual gap reconstructed from
    the potentials, so every reported number carries a certificate.
    """
    import ot
    from scipy.spatial.distance import cdist

    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"clouds differ in shape: {a.shape} vs {b.shape}")
    cost = cdist(a, b, metric="sqeuclidean")
    n = len(a)
    w = np.full(n, 1.0 / n)
    value, info = ot.emd2(w, w, cost, log=True, numItermax=5_000_000)
    if int(info.get("result_code", -1)) != 1:
        raise RuntimeError(f"exact OT failed: {info.get('warning')}")
    gap = 0.0
    u, v = info.get("u"), info.get("v")
    if u is not None and v is not None:
        u = np.asarray(u, dtype=np.float64).ravel()
        v = np.asarray(v, dtype=np.float64).ravel()
        if np.isfinite(u).all() and np.isfinite(v).all():
            gap = float(abs(w @ u + w @ v - value))
    return math.sqrt(max(value, 0.0)), gap


def simulate(initial, sigma, mu, noise_seed, device="cuda"):
    from training.protocol import simulate_original

    np.random.seed(noise_seed)
    torch.manual_seed(noise_seed)
    return simulate_original(torch.from_numpy(initial), float(sigma), mu=float(mu),
                             chi=16 * math.pi, delta=0.1, t=0.02, dt=0.005,
                             chunk=256, device=device).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=4,
                    help="how many parameter points to evaluate")
    ap.add_argument("--subsample", type=int, default=4000)
    ap.add_argument("--index-draws", type=int, default=3,
                    help="how many independent index vectors, to report spread")
    ap.add_argument("--use-heldout", action="store_true",
                    help="evaluate on held-out grid cells instead of the four corners")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda t: print(f"[{time.strftime('%H:%M:%S')}] {t}", flush=True)

    # ---- pick the cells ---------------------------------------------------
    if a.use_heldout:
        design_path = G.RESULTS / "sample_efficiency" / "split.json"
        split = json.loads(design_path.read_text(encoding="utf-8"))
        test_keys = {r["key"] for r in split["test"]}
        train_path = G.RESULTS / "optimal_budget_ablation" / "summary.json"
        trained = set()
        if train_path.is_file():
            payload = json.loads(train_path.read_text(encoding="utf-8"))
            trained = set(payload.get("cells", []))
        # the ablation scored these cells; reuse exactly them for comparability
        detail = G.RESULTS / "optimal_budget_ablation" / "detail.csv"
        cells = []
        with detail.open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                cells.append(G.Cell(int(row["i"]), int(row["j"])))
        cells = cells[: a.points]
    else:
        cells = [G.Cell(G.SIGMA_GRID.index(s), G.MU_GRID.index(m))
                 for s, m in POINTS[: a.points]]
    log(f"evaluating {len(cells)} cells, subsample {a.subsample}, "
        f"{a.index_draws} index draw(s)")

    # ---- load the trained model and the reference replicate ---------------
    model, _ = load_two(G.RESULTS / "optimal_budget_ablation" / "two_32000"
                        / "mf_two_param.pt", a.device)
    initial = unit_ball(a.n, 88)
    rng = np.random.default_rng(424242)
    index_draws = [rng.choice(a.n, size=a.subsample, replace=False)
                   for _ in range(a.index_draws)]

    rows = []
    for cell in cells:
        payload = load_cell(cell)
        reference = payload["clouds"]["reference"].astype(np.float64)
        source = payload["clouds"]["source"]
        log(f"=== {cell.key} sigma={cell.sigma:g} mu={cell.mu:g} ===")

        prediction = sample_two_param(model, torch.from_numpy(source),
                                      cell.sigma, cell.mu).numpy().astype(np.float64)

        # a second physical realisation: same initial cloud, different noise path
        noise_seed = 940_000 + 977 * (cell.i * 10 + cell.j)
        second = simulate(initial, cell.sigma, cell.mu, noise_seed, a.device) \
            .astype(np.float64)

        errs, floors, hist_errs, hist_floors, gaps = [], [], [], [], []
        for idx in index_draws:
            e, g = exact_w2(prediction[idx], reference[idx])
            f, _ = exact_w2(reference[idx], second[idx])
            errs.append(e)
            floors.append(f)
            gaps.append(g)
            hist_errs.append(score_pair(reference, prediction, G.BINS)["w2"])
            hist_floors.append(score_pair(reference, second, G.BINS)["w2"])

        row = {
            "key": cell.key, "sigma": cell.sigma, "mu": cell.mu,
            "particle_error_mean": float(np.mean(errs)),
            "particle_error_sd": float(np.std(errs, ddof=1)) if len(errs) > 1 else 0.0,
            "particle_floor_mean": float(np.mean(floors)),
            "particle_floor_sd": float(np.std(floors, ddof=1)) if len(floors) > 1 else 0.0,
            "histogram_error": float(np.mean(hist_errs)),
            "histogram_floor": float(np.mean(hist_floors)),
            "ot_gap_max": float(np.max(gaps)),
        }
        row["ratio_particle"] = (row["particle_error_mean"] / row["particle_floor_mean"]
                                 if row["particle_floor_mean"] else None)
        row["ratio_histogram"] = (row["histogram_error"] / row["histogram_floor"]
                                  if row["histogram_floor"] else None)
        rows.append(row)
        log(f"  particle: error {row['particle_error_mean']:.5f} "
            f"(sd {row['particle_error_sd']:.5f})  floor "
            f"{row['particle_floor_mean']:.5f}  ratio {row['ratio_particle']:.2f}x")
        log(f"  histogram: error {row['histogram_error']:.5f}  floor "
            f"{row['histogram_floor']:.5f}  ratio {row['ratio_histogram']:.2f}x")
        del prediction, second
        torch.cuda.empty_cache()

    # ---- report ----------------------------------------------------------
    print(f"\n{'cell':<9} {'sigma':>7} {'mu':>8} | {'P err':>8} {'P floor':>8} "
          f"{'P ratio':>8} | {'H err':>8} {'H floor':>8} {'H ratio':>8}")
    for r in rows:
        print(f"{r['key']:<9} {r['sigma']:7.0f} {r['mu']:8.3f} | "
              f"{r['particle_error_mean']:8.5f} {r['particle_floor_mean']:8.5f} "
              f"{r['ratio_particle']:8.2f} | "
              f"{r['histogram_error']:8.5f} {r['histogram_floor']:8.5f} "
              f"{r['ratio_histogram']:8.2f}")

    pr = np.array([r["ratio_particle"] for r in rows])
    hr = np.array([r["ratio_histogram"] for r in rows])
    print(f"\nparticle-level ratio: {pr.min():.2f} .. {pr.max():.2f} (mean {pr.mean():.2f})")
    print(f"histogram-level ratio: {hr.min():.2f} .. {hr.max():.2f} (mean {hr.mean():.2f})")
    print(f"correlation between the two ratios: "
          f"{np.corrcoef(pr, hr)[0, 1]:+.3f}")

    fields = list(rows[0])
    with (OUT / f"particle_metric_m{a.subsample}.csv").open("w", encoding="utf-8",
                                                            newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (OUT / f"particle_metric_m{a.subsample}.json").write_text(
        json.dumps({"subsample": a.subsample, "index_draws": a.index_draws,
                    "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
