"""E11: does the trained MeanFlow already encode the intermediate state?

Idea
----
MeanFlow learns u(z, r, t) over the whole (r, t) range, not just the single interval
[0, 1]. If that is true in practice, the network can answer "what does the cloud look like
at t = 0.5" even though no intermediate time slice was ever simulated. That would settle
the long-horizon question cheaply: the model would implicitly represent the process, and
only the *evaluation* would have been missing.

How to test without training anything new
-----------------------------------------
Two independent routes to the same check.

Route 1 - forward from the prior at half time
    x_half = x_0 - 0.5 * u(x_0, r=0, t=0.5)
    Compare with the true cloud at T/2 = 0.01.
    NOTE: this is only meaningful if the physical clock scales linearly with the flow
    coordinate t, which it does here because the training velocity is defined on the
    straight-line interpolant of (x_0, x_T). The check is whether W2(x_half, true_T/2)
    is small compared with W2(x_0, true_T/2).

Route 2 - backward from the generated terminal state
    x_half_from_T = x_T_hat + 0.5 * u(x_T_hat, r=0.5, t=1.0)
    i.e. ask for the average velocity over [0.5, 1] and step backwards.
    Compare with the true cloud at T/2.

Agreement between the two routes is itself informative: they use disjoint parts of the
(r, t) domain, so if both land near the truth the field is consistent, not memorised at
one corner.

What "the truth" is
-------------------
The simulator integrates with dt = 0.005 from 0 to T = 0.02, i.e. exactly four Euler steps.
T/2 = 0.01 is therefore *reachable* by the simulator at no extra cost: run two of the four
steps. This is a real reference, not an interpolation.

Cost
----
40 training cells + 40 held-out cells at 20k particles, two horizons each. No training.
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
from fullscale.train import load_two  # noqa: E402
from twoparam.data import unit_ball  # noqa: E402

OUT = G.RESULTS / "intermediate_state"


def simulate(initial: np.ndarray, sigma: float, mu: float, final_time: float,
             noise_seed: int, device: str) -> np.ndarray:
    from training.protocol import simulate_original

    np.random.seed(noise_seed)
    torch.manual_seed(noise_seed)
    return simulate_original(
        torch.from_numpy(initial), float(sigma), mu=float(mu),
        chi=16 * math.pi, delta=0.1, t=float(final_time), dt=0.005,
        chunk=256, device=device,
    ).numpy()


def exact_w2(a: np.ndarray, b: np.ndarray) -> float:
    import ot
    from scipy.spatial.distance import cdist

    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    cost = cdist(a, b, metric="sqeuclidean")
    n = len(a)
    w = np.full(n, 1.0 / n)
    value, info = ot.emd2(w, w, cost, log=True, numItermax=5_000_000)
    if int(info.get("result_code", -1)) != 1:
        raise RuntimeError(f"OT failed: {info.get('warning')}")
    return math.sqrt(max(value, 0.0))


def meanflow_step(model, x: torch.Tensor, sigma: float, mu: float,
                  r: float, t: float) -> torch.Tensor:
    """One average-velocity evaluation over [r, t], returning x - (t-r) u."""
    device = next(model.parameters()).device
    x = x.to(device)
    batch = len(x)
    ls = torch.full((batch, 1), math.log10(sigma), dtype=x.dtype, device=device)
    lm = torch.full((batch, 1), math.log10(mu), dtype=x.dtype, device=device)
    rr = torch.full((batch, 1), float(r), dtype=x.dtype, device=device)
    tt = torch.full((batch, 1), float(t), dtype=x.dtype, device=device)
    with torch.no_grad():
        return (x - (t - r) * model(x, rr, tt, ls, lm)).cpu()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--score-n", type=int, default=3000)
    ap.add_argument("--cells", type=int, default=40, help="how many cells to use")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda text: print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)

    half = G.FINAL_TIME / 2.0
    log(f"T = {G.FINAL_TIME}, T/2 = {half}, dt = {G.DT} "
        f"-> T/2 is exactly {round(half / G.DT)} Euler steps, so it is a real reference")

    checkpoint = G.RESULTS / "optimal_budget_ablation" / "two_32000" / "mf_two_param.pt"
    model, _ = load_two(checkpoint, a.device)
    initial = unit_ball(a.n, 88)

    # spread the cells over the box so the result is not one corner
    grid = G.full_grid()
    step = max(1, len(grid) // a.cells)
    cells = grid[::step][: a.cells]
    log(f"using {len(cells)} cells spread over the grid")

    rng = np.random.default_rng(20261116)
    index = rng.choice(a.n, size=min(a.score_n, a.n), replace=False)
    source = torch.from_numpy(initial.astype(np.float32))

    rows = []
    for position, cell in enumerate(cells, start=1):
        seed_base = 4_400_000 + 53 * (cell.i * 10 + cell.j)
        truth_half = simulate(initial, cell.sigma, cell.mu, half, seed_base, a.device)
        truth_full = simulate(initial, cell.sigma, cell.mu, G.FINAL_TIME,
                              seed_base + 1, a.device)

        # route 1: forward from the prior, stopping at half time
        pred_half_fwd = meanflow_step(model, source, cell.sigma, cell.mu,
                                      r=0.0, t=0.5).numpy().astype(np.float64)

        # the terminal generation, for route 2
        pred_full = meanflow_step(model, source, cell.sigma, cell.mu,
                                  r=0.0, t=1.0)
        # route 2: step backwards from the generated terminal state over [0.5, 1]
        pred_half_bwd = meanflow_step(model, pred_full, cell.sigma, cell.mu,
                                      r=0.5, t=1.0).numpy().astype(np.float64)

        row = {
            "key": cell.key, "sigma": cell.sigma, "mu": cell.mu,
            "w2_fwd_half_vs_truth_half": exact_w2(pred_half_fwd[index],
                                                  truth_half[index]),
            "w2_bwd_half_vs_truth_half": exact_w2(pred_half_bwd[index],
                                                  truth_half[index]),
            "w2_prior_vs_truth_half": exact_w2(initial[index], truth_half[index]),
            "w2_full_vs_truth_full": exact_w2(pred_full.numpy()[index], truth_full[index]),
            "w2_half_vs_full_truth": exact_w2(truth_half[index], truth_full[index]),
        }
        rows.append(row)
        log(f"  {position}/{len(cells)} {cell.key}: "
            f"fwd {row['w2_fwd_half_vs_truth_half']:.4f}  "
            f"bwd {row['w2_bwd_half_vs_truth_half']:.4f}  "
            f"prior {row['w2_prior_vs_truth_half']:.4f}")

    def summary(key: str) -> dict:
        values = [r[key] for r in rows]
        return {"mean": float(np.mean(values)), "median": float(np.median(values)),
                "sd": float(np.std(values, ddof=1)), "n": len(values)}

    payload = {
        "n_particles": a.n, "score_n": int(len(index)),
        "half_time": half, "final_time": G.FINAL_TIME,
        "n_cells": len(cells),
        "summary": {key: summary(key) for key in
                    ("w2_fwd_half_vs_truth_half", "w2_bwd_half_vs_truth_half",
                     "w2_prior_vs_truth_half", "w2_full_vs_truth_full",
                     "w2_half_vs_full_truth")},
        "rows": rows,
        "note": ("Route 1 evaluates u(x, r=0, t=0.5); route 2 evaluates u(x, r=0.5, t=1). "
                 "They use disjoint parts of the (r,t) domain. If both land near the true "
                 "half-time cloud, the field encodes the intermediate state rather than a "
                 "single endpoint."),
    }
    (OUT / "intermediate_state.json").write_text(json.dumps(payload, indent=2),
                                                 encoding="utf-8")

    print(f"\n{len(cells)} cells, {len(index)} points per exact OT:")
    print(f"  {'quantity':<40} {'mean':>9} {'median':>9} {'sd':>8}")
    labels = {
        "w2_prior_vs_truth_half": "prior vs true T/2  (baseline)",
        "w2_fwd_half_vs_truth_half": "model fwd from prior vs true T/2",
        "w2_bwd_half_vs_truth_half": "model bwd from x_T vs true T/2",
        "w2_half_vs_full_truth": "true T/2 vs true T  (scale)",
        "w2_full_vs_truth_full": "model x_T vs true T  (reference quality)",
    }
    for key, label in labels.items():
        block = payload["summary"][key]
        print(f"  {label:<40} {block['mean']:9.4f} {block['median']:9.4f} "
              f"{block['sd']:8.4f}")

    base = payload["summary"]["w2_prior_vs_truth_half"]["mean"]
    fwd = payload["summary"]["w2_fwd_half_vs_truth_half"]["mean"]
    bwd = payload["summary"]["w2_bwd_half_vs_truth_half"]["mean"]
    print(f"\n  forward route explains {100 * (1 - fwd / base):.1f}% of the "
          f"prior-to-half-time distance")
    print(f"  backward route explains {100 * (1 - bwd / base):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
