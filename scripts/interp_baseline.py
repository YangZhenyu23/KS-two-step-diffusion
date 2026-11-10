"""Interpolation baselines: what can be achieved without a network at all?

Why this comparison is needed
-----------------------------
The headline claim is that a two-parameter network learns the coupling between
sigma and mu. That claim is only meaningful if a network beats what a trivial
interpolator can do with the *same* training clouds. If nearest-neighbour or
inverse-distance interpolation already reaches the same W2, then the network's
contribution reduces to speed, not accuracy, and the paper must say so.

Three interpolators, all of which use only the 40 training clouds and the source
particles, and none of which involve any learning:

**nearest neighbour**
    Return the training cloud whose parameters are closest in the log plane.
    This is the "do nothing" baseline: it is what a lookup table gives.

**inverse distance weighting (IDW)**
    Draw each output particle from training cell ``d`` with probability
    proportional to ``dist(theta, theta_d)^(-p)`` with ``p = 2``. The output is a
    genuine mixture of the nearest few clouds, so it can beat nearest neighbour
    where the response varies smoothly.

**barycentric on the k nearest**
    Take the ``k = 4`` nearest training cells and draw with weights proportional
    to their inverse distance, which is IDW restricted to a local neighbourhood
    and avoids the long tail of far-away cells contributing spurious mass.

Two reference sets are used, because the two stages of the method have different
accuracy:

    A  interpolate the *reference* clouds, compare against MF alone
    B  interpolate the *MF+DP output* clouds, compare against MF+DP

Both are scored with the same 16-bin histogram W2 on the same 143 held-out cells
as every other ablation, so the numbers are directly comparable.

Usage:
    python scripts/interp_baseline.py
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

from fullscale import grid as G  # noqa: E402
from fullscale.evaluate import score_many, write_csv, write_json  # noqa: E402
from fullscale.model import sample_two_param  # noqa: E402
from fullscale.train import load_cell, load_two  # noqa: E402

from sample_efficiency import frozen_test_set, stratified_subset  # noqa: E402

OUT = G.RESULTS / "interp_baseline"
DP_CHECKPOINT = G.RESULTS / "dp_two_param" / "dp_corrector.pt"
MF_CHECKPOINT = G.RESULTS / "optimal_budget_ablation" / "two_32000" / "mf_two_param.pt"


def log_distance(a_sigma, a_mu, b_sigma, b_mu) -> float:
    return math.hypot(math.log10(a_sigma / b_sigma), math.log10(a_mu / b_mu))


def weights_nearest(cell, train, n) -> list[tuple[int, float]]:
    """One-hot on the closest training cell."""
    best, best_d = None, float("inf")
    for k, t in enumerate(train):
        d = log_distance(cell.sigma, cell.mu, t.sigma, t.mu)
        if d < best_d:
            best, best_d = k, d
    return [(best, 1.0)]


def weights_idw(cell, train, n, power: float = 2.0,
                floor: float = 1e-3) -> list[tuple[int, float]]:
    """Inverse-distance weights over every training cell."""
    raw = []
    for k, t in enumerate(train):
        d = max(log_distance(cell.sigma, cell.mu, t.sigma, t.mu), floor)
        raw.append((k, d ** (-power)))
    total = sum(w for _, w in raw)
    return [(k, w / total) for k, w in raw]


def weights_knn(cell, train, n, k: int = 4, power: float = 2.0,
                floor: float = 1e-3) -> list[tuple[int, float]]:
    """Inverse-distance weights restricted to the k nearest cells."""
    dists = [(log_distance(cell.sigma, cell.mu, t.sigma, t.mu), idx)
             for idx, t in enumerate(train)]
    dists.sort()
    chosen = dists[:k]
    raw = [(idx, max(d, floor) ** (-power)) for d, idx in chosen]
    total = sum(w for _, w in raw)
    return [(idx, w / total) for idx, w in raw]


def sample_from_weights(weights, sources: list[np.ndarray], n: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Draw n particles, each taken from a source cloud chosen by the weights.

    Drawing *particles* (not averaging coordinates) keeps the output an honest
    empirical measure: a mixture of clouds is a mixture of samples, and averaging
    coordinates across different cells would produce points that no physical
    trajectory visits.
    """
    probs = np.array([w for _, w in weights], dtype=np.float64)
    probs = probs / probs.sum()
    picks = rng.choice(len(weights), size=n, p=probs)
    out = np.empty((n, sources[0].shape[1]), dtype=np.float32)
    for j, (idx, _) in enumerate(weights):
        mask = picks == j
        count = int(mask.sum())
        if count == 0:
            continue
        src = sources[idx]
        take = rng.choice(len(src), size=count, replace=False)
        out[mask] = src[take]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261111)
    a = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda t: print(f"[{time.strftime('%H:%M:%S')}] {t}", flush=True)

    test, pool = frozen_test_set()
    design = stratified_subset(pool, 40, seed=20261107)
    design_keys = {c.key for c in design}
    slice_keys = {c.key for c in G.baseline_slice()}
    held = [c for c in G.full_grid()
            if c.key not in design_keys and c.key not in slice_keys]
    log(f"{len(design)} training cells, {len(held)} held-out cells")

    # ---- load the 40 training clouds, two flavours ----------------------
    mf_model, _ = load_two(MF_CHECKPOINT, "cuda")
    log("collecting training clouds: reference and MF+DP outputs")
    ref_sources, dp_sources = [], []
    # load the corrector directly to avoid a circular import
    corrector = None
    if DP_CHECKPOINT.is_file():
        sys.path.insert(0, str(ROOT / "scripts"))
        from dp_two_param import TwoParamCorrector, apply_corrector
        payload = torch.load(DP_CHECKPOINT, map_location="cpu", weights_only=True)
        arch = payload["architecture"]
        corrector = TwoParamCorrector(width=arch["width"], alpha=arch["alpha"],
                                      blocks=arch["blocks"],
                                      dimension=arch["dimension"])
        corrector.load_state_dict(payload["model"], strict=True)
        corrector = corrector.to("cuda").eval().requires_grad_(False)

    for cell in design:
        payload = load_cell(cell)
        ref_sources.append(payload["clouds"]["reference"].astype(np.float32))
        if corrector is not None:
            src = torch.from_numpy(payload["clouds"]["source"])
            mf = sample_two_param(mf_model, src, cell.sigma, cell.mu)
            dp_sources.append(apply_corrector(corrector, mf, cell.sigma, cell.mu))
        else:
            dp_sources.append(payload["clouds"]["reference"].astype(np.float32))
    log(f"  {len(ref_sources)} training clouds loaded")

    # ---- reference values for the two stages ---------------------------
    mf_only, _ = load_two(MF_CHECKPOINT, "cuda")
    pairs_stage = {"mf": {}, "mf_dp": {}}
    for cell in held:
        payload = load_cell(cell)
        src = torch.from_numpy(payload["clouds"]["source"])
        mf = sample_two_param(mf_only, src, cell.sigma, cell.mu).numpy().astype(np.float32)
        pairs_stage["mf"][cell.key] = (payload["clouds"]["reference"], mf)
        if corrector is not None:
            dp = apply_corrector(corrector, torch.from_numpy(mf), cell.sigma, cell.mu)
            pairs_stage["mf_dp"][cell.key] = (payload["clouds"]["reference"], dp)

    baseline = {}
    for stage, pairs in pairs_stage.items():
        if not pairs:
            continue
        rows = score_many(pairs, bins=G.BINS, workers=4, log=lambda t: None)
        baseline[stage] = float(np.mean([r["w2"] for r in rows]))
        log(f"baseline {stage}: {baseline[stage]:.5f}")

    # ---- interpolators -------------------------------------------------
    methods = {
        "nearest": lambda cell, tr: weights_nearest(cell, tr, len(tr)),
        "idw": lambda cell, tr: weights_idw(cell, tr, len(tr)),
        "knn4": lambda cell, tr: weights_knn(cell, tr, len(tr), k=4),
    }
    rng = np.random.default_rng(a.seed)
    n_particles = G.N_PARTICLES
    results = []
    for flavour, sources, stage in (("reference", ref_sources, "mf"),
                                    ("mf_dp", dp_sources, "mf_dp")):
        if stage not in baseline:
            continue
        for name, fn in methods.items():
            started = time.perf_counter()
            pairs = {}
            for cell in held:
                w = fn(cell, design)
                pred = sample_from_weights(w, sources, n_particles, rng)
                payload = load_cell(cell)
                pairs[cell.key] = (payload["clouds"]["reference"], pred)
            rows = score_many(pairs, bins=G.BINS, workers=4, log=lambda t: None)
            mean = float(np.mean([r["w2"] for r in rows]))
            results.append({
                "flavour": flavour, "stage": stage, "method": name,
                "mean_w2": mean,
                "model_w2": baseline[stage],
                "ratio_interp_over_model": mean / baseline[stage],
                "interp_worse_by": mean / baseline[stage] - 1,
                "seconds": round(time.perf_counter() - started, 1),
            })
            log(f"  {flavour:9s} {name:8s}: {mean:.5f}   "
                f"(model {baseline[stage]:.5f}, interp worse by "
                f"{mean / baseline[stage] - 1:+.1%})")

    # ---- report --------------------------------------------------------
    print(f"\n{'flavour':<10} {'method':<9} {'interp W2':>10} {'model W2':>9} "
          f"{'interp/model':>13}")
    for r in results:
        print(f"{r['flavour']:<10} {r['method']:<9} {r['mean_w2']:10.5f} "
              f"{r['model_w2']:9.5f} {r['ratio_interp_over_model']:13.2f}x")

    best = {}
    for r in results:
        key = r["flavour"]
        if key not in best or r["mean_w2"] < best[key]["mean_w2"]:
            best[key] = r
    print("\nbest interpolator vs the network:")
    for flavour, r in best.items():
        print(f"  {flavour}: {r['method']} gives {r['mean_w2']:.5f}, "
              f"network gives {r['model_w2']:.5f} "
              f"-> {'NETWORK WINS' if r['model_w2'] < r['mean_w2'] else 'INTERPOLATION WINS'}"
              f" by {abs(1 - r['model_w2'] / r['mean_w2']):.1%}")

    write_csv(OUT / "interp_baseline.csv", results)
    write_json(OUT / "interp_baseline.json", {"baseline": baseline, "rows": results,
                                              "n_held_out": len(held),
                                              "n_train": len(design)})
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
