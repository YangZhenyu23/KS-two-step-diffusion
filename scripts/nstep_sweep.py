"""N-step MeanFlow sampling: does spending more forward passes on stage 1 help?

Context
-------
MeanFlow learns the average velocity u(z, r, t) over the *whole* (r, t) range, not just the
single interval [0, 1]. The consistency identity

    (t - r) u(z_t, r, t) = (s - r) u(z_s, r, s) + (t - s) u(z_t, s, t)

is satisfied by the ground-truth field for any intermediate s, and the authors report a
2-NFE FID of 2.20 against 3.43 at 1-NFE on ImageNet. So multi-step application of the SAME
weights is a legitimate baseline, and it is the "few-step flow baseline" that was asked for.

The repository only had the 2-step case (and only on the single-parameter model). This
script generalises it to N in {1, 2, 4, 8} on the two-parameter model, on the held-out set
derived from the checkpoint, and also reports MF_N + corrector so the two ways of spending
extra compute can be compared directly.

Interval decomposition
----------------------
Split [0, 1] into N equal intervals and apply them in order:

    for i in range(N):  t_hi = 1 - i/N,  t_lo = 1 - (i+1)/N
                        x <- x - (t_hi - t_lo) * u(x, r=t_lo, t=t_hi)

For N = 2 this reproduces the schedule already in ``model.py``: (1.0, 0.5), (0.5, 0.0).

Cost
----
Pure forward passes, no training. 143 cells x (1+2+4+8) applications x 2 variants, plus
exact histogram OT per configuration.

Usage:
    python scripts/nstep_sweep.py --steps 1 2 4 8
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
from fullscale.train import load_cell, load_two  # noqa: E402
from fullscale.evaluate import score_pair  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from rollout_compare import load_corrector  # noqa: E402

OUT = G.RESULTS / "nstep"


def sample_n_step(model, source: torch.Tensor, sigma: float, mu: float, steps: int) -> torch.Tensor:
    """Apply the same MeanFlow weights ``steps`` times over equal sub-intervals."""
    device = next(model.parameters()).device
    x = source.detach().to(device)
    batch = len(x)
    log_sigma = torch.full((batch, 1), math.log10(float(sigma)), dtype=x.dtype, device=device)
    log_mu = torch.full((batch, 1), math.log10(float(mu)), dtype=x.dtype, device=device)
    with torch.no_grad():
        for index in range(steps):
            t_hi = 1.0 - index / steps
            t_lo = 1.0 - (index + 1) / steps
            t = torch.full((batch, 1), t_hi, dtype=x.dtype, device=device)
            r = torch.full((batch, 1), t_lo, dtype=x.dtype, device=device)
            x = x - (t_hi - t_lo) * model(x, r, t, log_sigma, log_mu)
    return x.cpu()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--seed", type=int, default=88)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bins", type=int, default=G.BINS)
    a = ap.parse_args()

    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda text: print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)

    checkpoint = G.RESULTS / "optimal_budget_ablation" / "two_32000" / "mf_two_param.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    train_keys = set(payload["cells"])
    slice_keys = {c.key for c in G.baseline_slice()}
    held = [c for c in G.full_grid() if c.key not in train_keys and c.key not in slice_keys]
    log(f"held-out cells (from checkpoint): {len(held)}")

    model, _ = load_two(checkpoint, a.device)
    corrector = load_corrector(a.seed, a.device)
    device = next(corrector.parameters()).device

    # one pass to load clouds; every N reuses them
    clouds = {}
    for cell in held:
        item = load_cell(cell)
        clouds[cell.key] = (item["clouds"]["reference"], item["clouds"]["source"])

    def correct(cloud: torch.Tensor, sigma: float, mu: float) -> torch.Tensor:
        x = cloud.to(device)
        cs = torch.full((len(x), 1), math.log10(sigma), dtype=torch.float32, device=device)
        cm = torch.full((len(x), 1), math.log10(mu), dtype=torch.float32, device=device)
        with torch.no_grad():
            return corrector(x, cs, cm).cpu()

    rows = []
    for steps in a.steps:
        log(f"=== N = {steps} ({steps} forward pass{'es' if steps > 1 else ''}) ===")
        started = time.perf_counter()
        scores_mf, scores_dp = [], []
        for position, cell in enumerate(held, start=1):
            reference, source = clouds[cell.key]
            mf = sample_n_step(model, torch.from_numpy(source), cell.sigma, cell.mu, steps)
            scores_mf.append(float(score_pair(reference, mf.numpy().astype(np.float32),
                                              a.bins)["w2"]))
            mf_dp = correct(mf, cell.sigma, cell.mu)
            scores_dp.append(float(score_pair(reference, mf_dp.numpy().astype(np.float32),
                                              a.bins)["w2"]))
            if position % 40 == 0:
                log(f"  {position}/{len(held)}")
        elapsed = time.perf_counter() - started
        rows.append({
            "steps": steps,
            "forward_passes": steps,
            "mf_mean_w2": float(np.mean(scores_mf)),
            "mf_sd": float(np.std(scores_mf, ddof=1)),
            "mf_dp_mean_w2": float(np.mean(scores_dp)),
            "mf_dp_sd": float(np.std(scores_dp, ddof=1)),
            "n": len(scores_mf),
            "seconds": elapsed,
            "mf_over_floor": float(np.mean(scores_mf) / G.FLOOR_16),
            "mf_dp_over_floor": float(np.mean(scores_dp) / G.FLOOR_16),
        })
        log(f"  N={steps}: MF {rows[-1]['mf_mean_w2']:.5f}  "
            f"MF+DP {rows[-1]['mf_dp_mean_w2']:.5f}  ({elapsed:.0f}s)")

    base = next(r for r in rows if r["steps"] == 1)
    for row in rows:
        row["mf_relative_vs_1step"] = float(1 - row["mf_mean_w2"] / base["mf_mean_w2"])
        row["mf_dp_relative_vs_1step_mf"] = float(
            1 - row["mf_dp_mean_w2"] / base["mf_mean_w2"])

    result = {
        "seed": a.seed, "bins": a.bins, "n_held_out": len(held),
        "floor_16": G.FLOOR_16, "rows": rows,
        "note": ("Same weights at every N: this is the few-step baseline. Intervals are "
                 "equal splits of [0,1]; N=2 reproduces the schedule in model.py."),
    }
    (OUT / f"nstep_s{a.seed}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n{'N (NFE)':>8} {'MF':>10} {'vs 1-step':>10} {'MF+DP':>10} "
          f"{'vs 1-step MF':>13} {'/floor':>8}")
    for row in rows:
        print(f"{row['steps']:8d} {row['mf_mean_w2']:10.5f} "
              f"{row['mf_relative_vs_1step'] * 100:+9.1f}% {row['mf_dp_mean_w2']:10.5f} "
              f"{row['mf_dp_relative_vs_1step_mf'] * 100:+12.1f}% "
              f"{row['mf_over_floor']:8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
