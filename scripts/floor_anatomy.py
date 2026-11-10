"""Is the noise floor measured against the right source of randomness?

The concern
-----------
A model is trained to map a *shared* initial cloud (seed 88) to a terminal cloud.
The only randomness it faces at test time is the Brownian path. Yet the floor used
everywhere in this repository was measured between two runs that differ in **both**
the initial cloud and the Brownian path.

Those two quantities are not the same:

    F_indep  = W2( Phi(X0a, xi_a), Phi(X0b, xi_b) )     different X0 and xi
    F_cond   = W2( Phi(X0,  xi_a), Phi(X0,  xi_b) )     same X0, different xi

F_indep contains variability the model never has to handle, because the model
always receives X0a. If the task is "given X0 and the parameters, predict the
terminal cloud", then F_cond is the correct irreducible error, and F_indep
overstates the floor.

Why it matters
--------------
If F_cond is substantially smaller than F_indep, every ratio E/floor reported so
far is too small, i.e. the model has **more** headroom than previously concluded.
This test decides which floor is defensible.

Second question: how much of the floor is histogram quantisation?
---------------------------------------------------------------
W2 computed on B bins differs from the true particle-level W2 by O(Delta) with
Delta the bin width, independent of the sampling noise. Sweeping B and fitting

    F(B)^2 = s^2 + c * Delta(B)^2,      Delta(B) ∝ 1/B

separates the irreducible sampling part s from the quantisation part. If s is
close to F(16), quantisation is negligible and finer bins buy little.

Third question: is a lower-variance metric available?
-----------------------------------------------------
Sliced Wasserstein projects onto random directions, where the 1-D W2 is exact and
cheap, then averages. It is a lower bound on W2 but has far lower estimator
variance. Reported alongside so the trade-off is visible.

Usage:
    python scripts/floor_anatomy.py --points 3 --replicates 2
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
from fullscale.evaluate import measure, score_pair, shared_edges  # noqa: E402
from twoparam.data import unit_ball  # noqa: E402

OUT = G.RESULTS / "floor_anatomy"

# three regimes: weak/weak, middle, strong-shear/weak-diffusion
POINTS = [(10.0, 0.1), (100.0, 1.0), (200.0, 0.1)]
# 48 bins/cell in 3-D means up to 110k occupied bins and a dense cost matrix that
# POT's network simplex cannot allocate; 32 is the practical ceiling here.
BINS = (12, 16, 24, 32)


def simulate(initial: np.ndarray, sigma: float, mu: float, noise_seed: int,
             device="cuda") -> np.ndarray:
    from training.protocol import simulate_original

    np.random.seed(noise_seed)
    torch.manual_seed(noise_seed)
    return simulate_original(torch.from_numpy(initial), float(sigma), mu=float(mu),
                             chi=16 * math.pi, delta=0.1, t=0.02, dt=0.005,
                             chunk=256, device=device).numpy()


def w2_bins(a: np.ndarray, b: np.ndarray, bins: int) -> float:
    """Exact histogram W2 on a grid shared by the pair only."""
    return score_pair(a.astype(np.float64), b.astype(np.float64), bins)["w2"]

def sliced_w2(a: np.ndarray, b: np.ndarray, *, projections: int = 256,
              seed: int = 0) -> float:
    """Sliced W2: average of exact 1-D W2 over random directions.

    1-D optimal transport is a sort, so each projection costs O(N log N) and is
    exact; averaging over directions gives an estimator with much lower variance
    than a single high-dimensional OT solve. It is a lower bound on W2, not an
    unbiased estimate of it, so it is reported as a separate diagnostic rather
    than substituted for the paper's metric.
    """
    rng = np.random.default_rng(seed)
    d = a.shape[1]
    directions = rng.standard_normal((projections, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    total = 0.0
    for w in directions:
        pa = np.sort(a @ w)
        pb = np.sort(b @ w)
        total += float(np.mean((pa - pb) ** 2))
    return math.sqrt(max(total / projections, 0.0))


def fit_floor(bins_list, floors) -> dict:
    """Fit F(B)^2 = s^2 + c * (1/B)^2 and return the infinite-bin limit."""
    x = np.array([1.0 / b ** 2 for b in bins_list])
    y = np.array([f * f for f in floors])
    # least squares on a two-parameter linear model in x
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    s2, c = float(coef[0]), float(coef[1])
    return {"s2": max(s2, 0.0), "s_infinite_bins": math.sqrt(max(s2, 0.0)),
            "quantisation_coef": c}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=3)
    ap.add_argument("--replicates", type=int, default=2,
                    help="extra simulations sharing the same initial cloud")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda t: print(f"[{time.strftime('%H:%M:%S')}] {t}", flush=True)

    # reproduce the repository's shared initial cloud exactly
    shared_initial = unit_ball(a.n, 88)
    log(f"shared initial cloud rebuilt from seed 88, N={a.n}")

    # the existing cells are themselves one realisation: shared X0, their own noise
    report = {"n": a.n, "bins": list(BINS), "points": []}
    for sigma, mu in POINTS[: a.points]:
        i = G.SIGMA_GRID.index(sigma)
        j = G.MU_GRID.index(mu)
        cell = G.Cell(i, j)
        log(f"=== sigma={sigma:g} mu={mu:g} ({cell.key}) ===")

        with np.load(G.DATA / "clouds" / f"{cell.key}.npz", allow_pickle=False) as p:
            stored = p["reference"].copy()
            meta = json.loads(p["meta"].item())
        stored_initial = np.load(G.DATA / "initial.npy")
        same_initial = np.array_equal(stored_initial.astype(np.float32),
                                      shared_initial.astype(np.float32))
        log(f"  stored cloud uses the shared initial: {same_initial} "
            f"(its noise_seed={meta['noise_seed']})")

        # replicates: same X0, different Brownian paths
        clouds = [stored.astype(np.float64)]
        for k in range(a.replicates):
            noise_seed = 900_000 + 7_919 * (k + 1) + 137 * (i * 10 + j)
            started = time.perf_counter()
            clouds.append(simulate(shared_initial, sigma, mu, noise_seed, a.device)
                          .astype(np.float64))
            log(f"  replicate {k}: noise_seed={noise_seed} "
                f"({time.perf_counter() - started:.0f}s)")

        # ---- conditional floor (same X0) at several bin counts --------------
        per_bin = {}
        for bins in BINS:
            values = []
            for p in range(len(clouds)):
                for q in range(p + 1, len(clouds)):
                    values.append(w2_bins(clouds[p], clouds[q], bins))
            per_bin[bins] = float(np.mean(values))
            log(f"  F_cond({bins:2d} bins) = {per_bin[bins]:.5f} "
                f"from {len(values)} pair(s)")

        fit = fit_floor(list(BINS), [per_bin[b] for b in BINS])
        log(f"  fit: sampling-only floor {fit['s_infinite_bins']:.5f} "
            f"(vs {per_bin[16]:.5f} at 16 bins)")

        # ---- sliced metric on the same pair --------------------------------
        sliced = []
        for p in range(len(clouds)):
            for q in range(p + 1, len(clouds)):
                sliced.append(sliced_w2(clouds[p], clouds[q], seed=1234))
        log(f"  sliced W2 (256 directions) = {np.mean(sliced):.5f}")

        # ---- how much does the initial cloud contribute? ------------------
        # a second cloud from the SAME path but a different initial draw is not
        # possible to obtain cheaply, so report the stored value from
        # local_floor.json for reference
        ref_path = G.RESULTS / "local_floor" / "local_floor.json"
        f_indep = None
        if ref_path.is_file():
            entries = json.loads(ref_path.read_text(encoding="utf-8"))["points"]
            match = [e["floor"] for e in entries
                     if abs(e["sigma"] - sigma) < 1e-9 and abs(e["mu"] - mu) < 1e-12]
            if match:
                f_indep = float(match[0])

        report["points"].append({
            "sigma": sigma, "mu": mu, "key": cell.key,
            "same_initial": bool(same_initial),
            "conditional_floor_by_bins": {str(b): v for b, v in per_bin.items()},
            "independent_floor_16bins": f_indep,
            "ratio_indep_over_cond": (f_indep / per_bin[16]) if f_indep else None,
            "sliced_w2": float(np.mean(sliced)),
            "fit": fit,
            "n_replicates": len(clouds),
        })
        del clouds
        torch.cuda.empty_cache()

    # ---- summary ---------------------------------------------------------
    print(f"\n{'sigma':>7} {'mu':>8} {'F_cond(16)':>11} {'F_indep(16)':>12} "
          f"{'ratio':>7} {'s(infinite)':>12} {'sliced':>9}")
    for r in report["points"]:
        f16 = r["conditional_floor_by_bins"]["16"]
        fi = r["independent_floor_16bins"]
        ratio = r["ratio_indep_over_cond"]
        print(f"{r['sigma']:7.0f} {r['mu']:8.3f} {f16:11.5f} "
              f"{(fi if fi else float('nan')):12.5f} "
              f"{(ratio if ratio else float('nan')):7.2f} "
              f"{r['fit']['s_infinite_bins']:12.5f} {r['sliced_w2']:9.5f}")

    ratios = [r["ratio_indep_over_cond"] for r in report["points"]
              if r["ratio_indep_over_cond"]]
    if ratios:
        print(f"\nmean F_indep / F_cond = {np.mean(ratios):.2f}x")
        print("  > 1 means the floor previously used was inflated by variation the")
        print("  model never faces; the model therefore has more headroom than the")
        print("  earlier ratios suggested.")

    (OUT / "floor_anatomy.json").write_text(json.dumps(report, indent=2),
                                            encoding="utf-8")
    print(f"\nwrote {OUT / 'floor_anatomy.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
