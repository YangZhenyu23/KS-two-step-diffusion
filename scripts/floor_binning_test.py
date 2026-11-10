"""Is the histogram metric inflating the floor? Compare with the raw-particle floor.

The anatomy test found a large discrepancy:

    F_cond (16 bins, histogram W2)  0.0453
    sliced W2 (256 directions)      0.0013

A factor of 35 cannot come from the physics. Two candidate explanations:

H1 **Binning.** The 16-bin histogram replaces every particle by its bin centre.
   For two clouds from the same distribution, most particles fall in the same bin
   and cancel, but the occupied-bin weights still differ by binomial noise. The
   resulting W2 is dominated by the quantisation grid, not by the particles.

H2 **The sliced metric is not comparable.** Sliced W2 averages squared 1-D
   distances over directions, which is *not* W2 and is systematically much
   smaller. It is a lower bound, so 0.0013 does not mean the true W2 is 0.0013.

Both can be true at once. To separate them, the decisive measurement is the W2
between two clouds computed **at the particle level, with no binning at all**,
using exact OT on a subsample small enough for network simplex (a few thousand
points). That number is the honest physical floor at that subsample size, and it
scales predictably with the subsample size, so the full-N value can be estimated.

A second measurement isolates the binning contribution directly: compute the
histogram W2 between a cloud and a *shuffled* copy of itself within bins. That
measures pure weight noise with zero physical difference.

Usage:
    python scripts/floor_binning_test.py
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

OUT = G.RESULTS / "floor_anatomy"


def simulate(initial, sigma, mu, noise_seed, device="cuda"):
    from training.protocol import simulate_original

    np.random.seed(noise_seed)
    torch.manual_seed(noise_seed)
    return simulate_original(torch.from_numpy(initial), float(sigma), mu=float(mu),
                             chi=16 * math.pi, delta=0.1, t=0.02, dt=0.005,
                             chunk=256, device=device).numpy()


def exact_particle_w2(a: np.ndarray, b: np.ndarray) -> float:
    """Exact W2 between two equal-size point clouds, no binning."""
    import ot
    from scipy.spatial.distance import cdist

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    cost = cdist(a, b, metric="sqeuclidean")
    n = len(a)
    w = np.full(n, 1.0 / n)
    value, info = ot.emd2(w, w, cost, log=True, numItermax=2_000_000)
    if int(info.get("result_code", -1)) != 1:
        raise RuntimeError(f"OT failed: {info.get('warning')}")
    return math.sqrt(max(value, 0.0))


def binning_noise(a: np.ndarray, bins: int, seed: int = 0) -> float:
    """Histogram W2 between a cloud and a copy with the same bins but shuffled counts.

    Two clouds identical up to within-bin resampling: the bin *counts* are the same
    but the assignment of particles to bins is redrawn from the same multinomial.
    Any W2 computed from the resulting histograms is pure discretisation noise.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=np.float64)
    lo, hi = a.min(axis=0), a.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    pad = 1e-6 * span
    edges = [np.linspace(lo[d] - pad[d], hi[d] + pad[d], bins + 1) for d in range(a.shape[1])]
    idx = np.stack([np.clip(np.digitize(a[:, d], edges[d]) - 1, 0, bins - 1)
                    for d in range(a.shape[1])], axis=1)
    counts = {}
    for row in idx:
        key = tuple(row)
        counts[key] = counts.get(key, 0) + 1
    keys = list(counts)
    probs = np.array([counts[k] for k in keys], dtype=np.float64)
    probs /= probs.sum()
    # draw a second multinomial with the same expected counts
    redraw = rng.multinomial(len(a), probs)
    centres = [0.5 * (e[1:] + e[:-1]) for e in edges]

    def to_measure(weights):
        occupied = weights > 0
        pos = np.array([[centres[d][keys[k][d]] for d in range(len(centres))]
                        for k in range(len(keys)) if occupied[k]])
        w = weights[occupied] / weights.sum()
        return pos, w

    import ot
    from scipy.spatial.distance import cdist

    pos_a, w_a = to_measure(probs * len(a))
    pos_b, w_b = to_measure(redraw.astype(np.float64))
    cost = cdist(pos_a, pos_b, metric="sqeuclidean")
    value, info = ot.emd2(w_a, w_b, cost, log=True, numItermax=2_000_000)
    if int(info.get("result_code", -1)) != 1:
        raise RuntimeError(f"OT failed: {info.get('warning')}")
    return math.sqrt(max(value, 0.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=10.0)
    ap.add_argument("--mu", type=float, default=0.1)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--subsamples", type=int, nargs="+", default=[1000, 2000, 4000])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda t: print(f"[{time.strftime('%H:%M:%S')}] {t}", flush=True)

    initial = unit_ball(a.n, 88)
    log(f"two simulations at sigma={a.sigma:g} mu={a.mu:g}, shared X0, different noise")
    clouds = []
    for k, noise_seed in enumerate((907_919, 915_838)):
        started = time.perf_counter()
        clouds.append(simulate(initial, a.sigma, a.mu, noise_seed, a.device)
                      .astype(np.float64))
        log(f"  cloud {k}: {time.perf_counter() - started:.0f}s")
    X, Y = clouds

    report = {"sigma": a.sigma, "mu": a.mu, "n": a.n, "sigma_hist": {}, "particle": {}}

    # ---- histogram floor at several bin counts ---------------------------
    print(f"\n{'bins':>6} {'F_hist':>9}")
    for bins in (12, 16, 24, 32):
        value = score_pair(X, Y, bins)["w2"]
        report["sigma_hist"][str(bins)] = value
        print(f"{bins:6d} {value:9.5f}")

    # ---- pure binning noise: cloud vs within-bin resample of itself ------
    print(f"\n{'bins':>6} {'binning noise':>14}  (same particles, resampled bins)")
    for bins in (12, 16, 24, 32):
        value = binning_noise(X, bins, seed=7)
        report["sigma_hist"].setdefault("binning_only", {})[str(bins)] = value
        print(f"{bins:6d} {value:14.5f}")

    # ---- particle-level floor on subsamples ------------------------------
    print(f"\n{'subsample':>10} {'W2 particles':>14} {'scaled to N':>13}")
    for m in a.subsamples:
        if m > len(X):
            continue
        idx = np.random.default_rng(11).choice(len(X), size=m, replace=False)
        value = exact_particle_w2(X[idx], Y[idx])
        # W2 between empirical measures scales like N^{-1/3} in 3-D for the
        # sampling part, so the full-N floor is the subsample value times
        # (m/N)^{1/3}
        scaled = value * (m / len(X)) ** (1 / 3)
        report["particle"][str(m)] = {"w2": value, "scaled_to_n": scaled}
        print(f"{m:10d} {value:14.5f} {scaled:13.5f}")

    # ---- verdict ---------------------------------------------------------
    f16 = report["sigma_hist"]["16"]
    b16 = report["sigma_hist"]["binning_only"]["16"]
    particle = min((v["scaled_to_n"] for v in report["particle"].values()),
                   default=float("nan"))
    print(f"\nsummary at sigma={a.sigma:g} mu={a.mu:g}")
    print(f"  histogram floor (16 bins)        {f16:.5f}")
    print(f"  pure binning noise (16 bins)     {b16:.5f}   "
          f"({b16 / f16:.0%} of the floor)")
    print(f"  particle-level floor (extrap.)   {particle:.5f}")
    print("\n  if pure binning noise is a large fraction of the histogram floor,")
    print("  then the floor is mostly a discretisation artefact. If the particle")
    print("  level floor is much smaller than the histogram one, the reported")
    print("  E/floor ratios have been too optimistic and the model has more")
    print("  headroom than previously stated.")

    (OUT / "floor_binning_test.json").write_text(json.dumps(report, indent=2),
                                                 encoding="utf-8")
    print(f"\nwrote {OUT / 'floor_binning_test.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
