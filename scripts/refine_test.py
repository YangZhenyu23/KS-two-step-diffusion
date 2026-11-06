"""Mesh-refinement test: the decisive numerical evidence for continuity.

If the cloud family is Lipschitz continuous in the parameters, then halving the
parameter step must roughly halve the observed W2. If the family were
discontinuous (or if the measurement were dominated by sampling noise), halving
the step would *not* halve the distance: the measured difference would either
stay flat (noise floor) or not scale with h at all.

This script therefore generates clouds on nested parameter steps

    sigma: 100 -> 105 -> 110 ... (step 10, 5, 2.5)
    mu:     1 -> 1.1 -> 1.2 ...  (step 0.1, 0.05)

and reports ``W2`` and ``W2/h`` for each refinement level. The expected signature
of continuity is

    W2 roughly proportional to h  =>  W2/h roughly constant

and the expected signature of the noise floor is

    W2 flat in h  =>  W2/h grows like 1/h

Both are measured, so the reader can see which regime each axis is in.

Usage:
    python scripts/refine_test.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream" / "MF-DeepParticle" / "deepparticle"))

from twoparam.data import unit_ball  # noqa: E402
from twoparam.evaluate import _edges, measure, solve  # noqa: E402
from twoparam.protocol import RESULTS  # noqa: E402


def simulate(initial, sigma, mu, seed, n, device="cuda"):
    from training.protocol import simulate_original

    np.random.seed(seed)
    torch.manual_seed(seed)
    return simulate_original(torch.from_numpy(initial[:n]), sigma, mu=mu, chi=16 * math.pi,
                             delta=0.1, t=0.02, dt=0.005, chunk=256, device=device).numpy()


def w2(a, b, bins=16):
    edges = _edges([a, b], bins)
    return solve(measure(a, edges, bins), measure(b, edges, bins))["w2"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--steps", type=float, nargs="+", default=[10.0, 5.0, 2.5])
    ap.add_argument("--sigma0", type=float, default=100.0)
    ap.add_argument("--mu0", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=RESULTS / "continuity" / "refinement.json")
    a = ap.parse_args()

    initial = unit_ball(a.n, 88)
    report = {"n": a.n, "sigma": [], "mu": []}

    print(f"N = {a.n}, base sigma = {a.sigma0}, base mu = {a.mu0}\n")

    print("=== sigma refinement (mu fixed at 1.0) ===")
    print(f"{'step':>7} {'h=log10 ratio':>14} {'W2':>9} {'W2/h':>9}")
    for step in a.steps:
        s_lo, s_hi = a.sigma0, a.sigma0 + step
        lo = simulate(initial, s_lo, a.mu0, 777, a.n)
        hi = simulate(initial, s_hi, a.mu0, 777, a.n)
        d = w2(lo, hi)
        h = math.log10(s_hi / s_lo)
        print(f"{step:7.2f} {h:14.5f} {d:9.5f} {d / h:9.4f}")
        report["sigma"].append({"step": step, "h": h, "w2": d, "ratio": d / h})
        del lo, hi
        torch.cuda.empty_cache()

    print("\n=== mu refinement (sigma fixed at 100) ===")
    print(f"{'step':>7} {'h=log10 ratio':>14} {'W2':>9} {'W2/h':>9}")
    for step in (1.0, 0.5, 0.25):
        m_lo, m_hi = a.mu0, a.mu0 * (1 + step)
        lo = simulate(initial, a.sigma0, m_lo, 777, a.n)
        hi = simulate(initial, a.sigma0, m_hi, 777, a.n)
        d = w2(lo, hi)
        h = math.log10(m_hi / m_lo)
        print(f"{step:7.2f} {h:14.5f} {d:9.5f} {d / h:9.4f}")
        report["mu"].append({"step": step, "h": h, "w2": d, "ratio": d / h})
        del lo, hi
        torch.cuda.empty_cache()

    print("\ninterpretation:")
    print("  W2 proportional to h  -> ratio constant  -> Lipschitz continuous")
    print("  W2 independent of h    -> ratio ~ 1/h     -> dominated by sampling noise")
    for axis in ("sigma", "mu"):
        ratios = [r["ratio"] for r in report[axis]]
        w2s = [r["w2"] for r in report[axis]]
        spread = max(ratios) / min(ratios) if min(ratios) > 0 else float("inf")
        print(f"  {axis}: ratio spread over the refinements = {spread:.2f}x, "
              f"W2 range = {min(w2s):.5f}..{max(w2s):.5f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
