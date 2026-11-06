"""Separate the continuity signal from the sampling noise.

The refine test showed W2 falling as the parameter step shrinks, but not
linearly: ``W2/h`` drifted upward. Two things contribute to a measured distance
between two clouds of *different* parameters:

    W2_measured^2  ~  W2_signal^2  +  W2_noise^2

The noise term is what two independent samples of the *same* parameter would
show. This script measures that term directly at the same particle count and
discretisation, then reports the noise-corrected signal

    W2_signal = sqrt(max(W2_measured^2 - W2_noise^2, 0))

If the corrected signal scales linearly with ``h``, the family is Lipschitz
continuous and the apparent sublinearity was purely the noise floor.

Usage:
    python scripts/signal_vs_noise.py
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
    ap.add_argument("--sigma0", type=float, default=100.0)
    ap.add_argument("--mu0", type=float, default=1.0)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--out", type=Path, default=RESULTS / "continuity" / "signal_noise.json")
    a = ap.parse_args()

    n = a.n
    print(f"N = {n}, reference point (sigma={a.sigma0}, mu={a.mu0})\n")

    # ---- noise: same parameters, independent initial cloud AND noise ------
    noise_samples = []
    for k in range(a.pairs + 1):
        seed = 5000 + 977 * k
        initial = unit_ball(n, seed)
        noise_samples.append(simulate(initial, a.sigma0, a.mu0, seed + 13, n))
        print(f"  noise sample {k} done", flush=True)
    noise_values = []
    for k in range(1, len(noise_samples)):
        noise_values.append(w2(noise_samples[0], noise_samples[k]))
    noise = float(np.mean(noise_values))
    noise_sd = float(np.std(noise_values))
    print(f"\nnoise floor (same parameters, independent samples): "
          f"{noise:.5f} +- {noise_sd:.5f}  n={len(noise_values)}")
    del noise_samples
    torch.cuda.empty_cache()

    # ---- signal: shared initial cloud, only the parameter changes ---------
    initial = unit_ball(n, 88)
    base = simulate(initial, a.sigma0, a.mu0, 777, n)

    report = {"n": n, "noise": noise, "noise_sd": noise_sd, "sigma": [], "mu": []}

    print("\n=== sigma: measured vs noise-corrected ===")
    print(f"{'step':>7} {'h':>9} {'W2_meas':>9} {'W2_corr':>9} {'corr/h':>9} {'meas/h':>9}")
    for step in (20.0, 10.0, 5.0, 2.5):
        hi = simulate(initial, a.sigma0 + step, a.mu0, 777, n)
        d = w2(base, hi)
        h = math.log10((a.sigma0 + step) / a.sigma0)
        corrected = math.sqrt(max(d * d - noise * noise, 0.0))
        print(f"{step:7.2f} {h:9.5f} {d:9.5f} {corrected:9.5f} "
              f"{corrected / h:9.4f} {d / h:9.4f}")
        report["sigma"].append({"step": step, "h": h, "w2": d, "w2_corrected": corrected,
                                "ratio_corrected": corrected / h, "ratio_measured": d / h})
        del hi
        torch.cuda.empty_cache()

    print("\n=== mu: measured vs noise-corrected ===")
    print(f"{'step':>7} {'h':>9} {'W2_meas':>9} {'W2_corr':>9} {'corr/h':>9} {'meas/h':>9}")
    for step in (1.0, 0.5, 0.25):
        hi = simulate(initial, a.sigma0, a.mu0 * (1 + step), 777, n)
        d = w2(base, hi)
        h = math.log10(a.mu0 * (1 + step) / a.mu0)
        corrected = math.sqrt(max(d * d - noise * noise, 0.0))
        print(f"{step:7.2f} {h:9.5f} {d:9.5f} {corrected:9.5f} "
              f"{corrected / h:9.4f} {d / h:9.4f}")
        report["mu"].append({"step": step, "h": h, "w2": d, "w2_corrected": corrected,
                             "ratio_corrected": corrected / h, "ratio_measured": d / h})
        del hi
        torch.cuda.empty_cache()

    print("\nverdict:")
    for axis in ("sigma", "mu"):
        r = [x["ratio_corrected"] for x in report[axis]]
        spread = max(r) / min(r) if min(r) > 0 else float("inf")
        mono = all(x["w2_corrected"] >= y["w2_corrected"]
                   for x, y in zip(report[axis], report[axis][1:]))
        print(f"  {axis}: corrected ratio spread = {spread:.2f}x, "
              f"corrected W2 monotone in h = {mono}")
        report[axis + "_verdict"] = {"ratio_spread": spread, "monotone": mono}

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
