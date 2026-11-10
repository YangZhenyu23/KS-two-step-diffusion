"""Full verification of the particle-level floor ratio across the parameter box.

The single-cell check at (10, 0.1) gave ratio = 0.951 +- 0.004 over four index
draws and three reference replicates. That is stable, but one corner of the
parameter box cannot support a claim about the whole study. This script repeats
the measurement at four corners plus the centre, and reports the ratio per cell.

What is measured per cell
-------------------------
    model_error   = W2_particle( model[idx]    , reference[idx] )
    floor         = mean over R independent realisations of
                    W2_particle( reference[idx], replicate[idx] )
    between       = mean over pairs of realisations of
                    W2_particle( replicate_p[idx], replicate_q[idx] )

``between`` is the cleanest floor: it involves no model at all, only two
independent physical simulations. If ``model_error`` is at or below ``between``,
the model is at the level of physical reproducibility, which is the strongest
statement available for a deterministic predictor.

Why a ratio below one is admissible
-----------------------------------
The model is a deterministic function of the source cloud. It carries no
independent sampling noise of its own, whereas a fresh replicate carries a new
Brownian path. A smooth deterministic predictor can therefore sit *closer* to the
reference than an independent draw does. Values just below one are the signature
of a model that reproduces the distribution's deterministic content rather than
memorising one realisation.

Cost
----
Each cell needs R + 1 simulations of 100k particles at ~80 s plus exact OT on the
subsamples. With 5 cells and R = 3 that is 20 simulations, about 30 minutes, plus
scoring.

Usage:
    python scripts/particle_floor_full.py --replicates 3 --subsample 3000
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

# five cells spanning the parameter box: four corners and the centre
CELLS = [(10.0, 0.1), (10.0, 10.0), (100.0, 1.0), (200.0, 0.1), (200.0, 10.0)]


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


def simulate(initial, sigma, mu, noise_seed, device="cuda"):
    from training.protocol import simulate_original

    np.random.seed(noise_seed)
    torch.manual_seed(noise_seed)
    return simulate_original(torch.from_numpy(initial), float(sigma), mu=float(mu),
                             chi=16 * math.pi, delta=0.1, t=0.02, dt=0.005,
                             chunk=256, device=device).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--subsample", type=int, default=3000)
    ap.add_argument("--index-draws", type=int, default=3)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda t: print(f"[{time.strftime('%H:%M:%S')}] {t}", flush=True)

    model, _ = load_two(G.RESULTS / "optimal_budget_ablation" / "two_32000"
                        / "mf_two_param.pt", a.device)
    initial = unit_ball(a.n, 88)
    rng = np.random.default_rng(606060)
    indices = [rng.choice(a.n, size=a.subsample, replace=False)
               for _ in range(a.index_draws)]

    rows = []
    for sigma, mu in CELLS:
        i, j = G.SIGMA_GRID.index(sigma), G.MU_GRID.index(mu)
        cell = G.Cell(i, j)
        log(f"=== {cell.key} sigma={sigma:g} mu={mu:g} ===")

        payload = load_cell(cell)
        reference = payload["clouds"]["reference"].astype(np.float64)
        prediction = sample_two_param(model, torch.from_numpy(payload["clouds"]["source"]),
                                      sigma, mu).numpy().astype(np.float64)

        replicates = []
        for k in range(a.replicates):
            seed = 1_230_000 + 977 * (i * 10 + j) + 3119 * k
            started = time.perf_counter()
            replicates.append(simulate(initial, sigma, mu, seed, a.device).astype(np.float64))
            log(f"  replicate {k}: {time.perf_counter() - started:.0f}s")
            torch.cuda.empty_cache()

        per_draw = []
        for d, idx in enumerate(indices):
            err = exact_w2(prediction[idx], reference[idx])
            floors = [exact_w2(reference[idx], r[idx]) for r in replicates]
            between = [exact_w2(replicates[p][idx], replicates[q][idx])
                       for p in range(len(replicates))
                       for q in range(p + 1, len(replicates))]
            hist_err = score_pair(reference, prediction, G.BINS)["w2"]
            hist_floor = float(np.mean([score_pair(reference, r, G.BINS)["w2"]
                                        for r in replicates]))
            per_draw.append({
                "draw": d, "error": err,
                "floor_mean": float(np.mean(floors)),
                "floor_sd": float(np.std(floors, ddof=1)) if len(floors) > 1 else 0.0,
                "between_mean": float(np.mean(between)) if between else None,
                "between_sd": float(np.std(between, ddof=1)) if len(between) > 1 else 0.0,
                "ratio": err / float(np.mean(floors)),
                "ratio_between": (err / float(np.mean(between))) if between else None,
                "histogram_error": hist_err,
                "histogram_floor": hist_floor,
                "histogram_ratio": hist_err / hist_floor if hist_floor else None,
            })
            log(f"  draw {d}: err {err:.5f}  floor {np.mean(floors):.5f}  "
                f"between {np.mean(between) if between else float('nan'):.5f}  "
                f"ratio {per_draw[-1]['ratio']:.3f}")

        ratios = np.array([r["ratio"] for r in per_draw])
        ratios_b = np.array([r["ratio_between"] for r in per_draw
                             if r["ratio_between"] is not None])
        row = {
            "key": cell.key, "sigma": sigma, "mu": mu,
            "error_mean": float(np.mean([r["error"] for r in per_draw])),
            "error_sd": float(np.std([r["error"] for r in per_draw], ddof=1)),
            "floor_mean": float(np.mean([r["floor_mean"] for r in per_draw])),
            "between_mean": float(np.mean([r["between_mean"] for r in per_draw
                                           if r["between_mean"] is not None]))
            if any(r["between_mean"] for r in per_draw) else None,
            "ratio_mean": float(ratios.mean()),
            "ratio_sd": float(ratios.std(ddof=1)) if len(ratios) > 1 else 0.0,
            "ratio_between_mean": float(ratios_b.mean()) if ratios_b.size else None,
            "histogram_error": float(np.mean([r["histogram_error"] for r in per_draw])),
            "histogram_floor": float(np.mean([r["histogram_floor"] for r in per_draw])),
            "histogram_ratio": float(np.mean([r["histogram_ratio"] for r in per_draw])),
            "replicates": a.replicates,
            "subsample": a.subsample,
        }
        rows.append(row)
        log(f"  -> particle ratio {row['ratio_mean']:.3f} "
            f"(sd {row['ratio_sd']:.3f})   "
            f"vs-between {row['ratio_between_mean']:.3f}   "
            f"histogram ratio {row['histogram_ratio']:.2f}")
        del prediction, replicates
        torch.cuda.empty_cache()

    print(f"\n{'cell':<9} {'sigma':>7} {'mu':>8} | {'err':>8} {'floor':>8} "
          f"{'ratio':>7} {'sd':>6} {'vs betw':>8} | {'H ratio':>8}")
    for r in rows:
        print(f"{r['key']:<9} {r['sigma']:7.0f} {r['mu']:8.3f} | "
              f"{r['error_mean']:8.5f} {r['floor_mean']:8.5f} "
              f"{r['ratio_mean']:7.3f} {r['ratio_sd']:6.3f} "
              f"{(r['ratio_between_mean'] if r['ratio_between_mean'] else float('nan')):8.3f} | "
              f"{r['histogram_ratio']:8.2f}")

    pr = np.array([r["ratio_mean"] for r in rows])
    hr = np.array([r["histogram_ratio"] for r in rows])
    print(f"\nparticle ratio: {pr.min():.3f} .. {pr.max():.3f} (mean {pr.mean():.3f})")
    print(f"histogram ratio: {hr.min():.2f} .. {hr.max():.2f} (mean {hr.mean():.2f})")
    print(f"the histogram ratio overstates the gap by {hr.mean() / pr.mean():.2f}x")

    with (OUT / "particle_floor_full.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "particle_floor_full.json").write_text(json.dumps({
        "replicates": a.replicates, "subsample": a.subsample,
        "index_draws": a.index_draws, "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
