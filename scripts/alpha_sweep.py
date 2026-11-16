"""Correction strength selected on a development split, reported on a disjoint test split.

The requirement this satisfies
------------------------------
"Choose any correction-strength or stopping settings on development cases, not the final
evaluation set."

There are two ways to sweep the correction strength alpha:

  (a) retrain the corrector once per alpha. Honest, but 2.6 h per setting and the
      resulting weights differ for reasons unrelated to alpha (optimiser trajectory).
  (b) sweep alpha at *inference*. For the zero-initialised linear output head the map is
      ``x + alpha * (W h + b)``, so alpha is exactly the deployment-time "how much do I
      trust the corrector" knob. It costs one forward pass per alpha and is the setting an
      operator would actually tune.

This script does (b) and says so. It is a statement about the deployment knob, not about
retraining at each alpha. Because the corrector was trained at alpha = 1, the sweep also
tells us whether alpha = 1 is near-optimal or whether the trained network is
systematically over/under-shooting.

The split
---------
The 143/144 held-out cells are divided deterministically into:

  dev   35% - used to pick alpha*
  test  65% - untouched until alpha* is fixed, then reported

Both splits are disjoint from the 40 training cells, so neither leaks the corrector's
training data. The split itself is by a fixed RNG seed and saved to disk so it is
reproducible and auditable.

Usage:
    python scripts/alpha_sweep.py --alphas 0 0.25 0.5 1 1.5 2 3 4
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
from fullscale.model import sample_two_param  # noqa: E402
from fullscale.evaluate import score_pair  # noqa: E402
from twoparam.data import unit_ball  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from rollout_compare import load_corrector  # noqa: E402

OUT = G.RESULTS / "alpha_sweep"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
    ap.add_argument("--dev-fraction", type=float, default=0.35)
    ap.add_argument("--split-seed", type=int, default=424242)
    ap.add_argument("--seed", type=int, default=88)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bins", type=int, default=G.BINS)
    a = ap.parse_args()

    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda text: print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)

    # ---- reproduce the exact training/held-out split used by dp_two_param.py ----
    sys.path.insert(0, str(ROOT / "scripts"))
    from dp_two_param import stratified_subset

    pool = G.full_grid()
    design = stratified_subset(pool, 40, seed=20261107)
    design_keys = {c.key for c in design}
    slice_keys = {c.key for c in G.baseline_slice()}
    held = [c for c in pool if c.key not in design_keys and c.key not in slice_keys]
    log(f"train cells {len(design)}, held out {len(held)}")

    rng = np.random.default_rng(a.split_seed)
    order = rng.permutation(len(held))
    n_dev = int(round(a.dev_fraction * len(held)))
    dev = [held[k] for k in order[:n_dev]]
    test = [held[k] for k in order[n_dev:]]
    log(f"dev {len(dev)}, test {len(test)}  (split seed {a.split_seed})")
    (OUT / "split.json").write_text(json.dumps({
        "dev": [c.key for c in dev], "test": [c.key for c in test],
        "dev_fraction": a.dev_fraction, "split_seed": a.split_seed,
    }, indent=2), encoding="utf-8")

    mf_model, _ = load_two(G.RESULTS / "optimal_budget_ablation" / "two_32000"
                           / "mf_two_param.pt", a.device)
    corrector = load_corrector(a.seed, a.device)
    device = next(corrector.parameters()).device

    # ---- precompute MF outputs once; every alpha reuses them ----
    cache = {}
    for position, cell in enumerate(held, start=1):
        payload = load_cell(cell)
        source = torch.from_numpy(payload["clouds"]["source"])
        mf_cloud = sample_two_param(mf_model, source, cell.sigma, cell.mu, steps=1)
        cache[cell.key] = {
            "reference": payload["clouds"]["reference"],
            "mf": mf_cloud,
            "sigma": cell.sigma, "mu": cell.mu,
        }
        if position % 40 == 0:
            log(f"  MF forward {position}/{len(held)}")

    def score_alpha(alpha: float, cells: list) -> dict:
        """W2 per cell at this alpha, with the corrector output scaled by alpha."""
        rows = {}
        for cell in cells:
            entry = cache[cell.key]
            mf_cloud = entry["mf"]
            if alpha == 0.0:
                prediction = mf_cloud
            else:
                x = mf_cloud.to(device)
                cs = torch.full((len(x), 1), math.log10(entry["sigma"]),
                                dtype=torch.float32, device=device)
                cm = torch.full((len(x), 1), math.log10(entry["mu"]),
                                dtype=torch.float32, device=device)
                with torch.no_grad():
                    # corrector computes x + 1.0 * MLP(x); subtract x to rescale the delta
                    corrected = corrector(x, cs, cm)
                prediction = (x + alpha * (corrected - x)).cpu()
            scored = score_pair(entry["reference"], prediction.numpy().astype(np.float32), a.bins)
            rows[cell.key] = float(scored["w2"])
        return rows

    # ---- sweep on dev ----
    log(f"sweeping {len(a.alphas)} alphas on {len(dev)} dev cells")
    dev_scores, test_scores = {}, {}
    for alpha in a.alphas:
        dev_scores[alpha] = score_alpha(alpha, dev)
        log(f"  alpha={alpha:<5} dev mean W2 = {np.mean(list(dev_scores[alpha].values())):.5f}")

    dev_mean = {alpha: float(np.mean(list(v.values()))) for alpha, v in dev_scores.items()}
    best_alpha = min(dev_mean, key=lambda k: dev_mean[k])
    log(f"selected alpha* = {best_alpha} (dev {dev_mean[best_alpha]:.5f})")

    # ---- report selected alpha on the untouched test split ----
    log(f"scoring alpha* = {best_alpha} on {len(test)} test cells")
    test_scores[best_alpha] = score_alpha(best_alpha, test)
    test_scores[1.0] = score_alpha(1.0, test)
    test_scores[0.0] = score_alpha(0.0, test)
    for alpha in (0.0, 1.0, best_alpha):
        values = [test_scores[alpha][c.key] for c in test]
        log(f"  test alpha={alpha:<5} mean {np.mean(values):.5f}")

    mf_test = [test_scores[0.0][c.key] for c in test]
    dp1_test = [test_scores[1.0][c.key] for c in test]
    best_test = [test_scores[best_alpha][c.key] for c in test]
    payload = {
        "alphas": a.alphas, "seed": a.seed, "bins": a.bins,
        "n_dev": len(dev), "n_test": len(test),
        "dev_fraction": a.dev_fraction, "split_seed": a.split_seed,
        "dev_mean": dev_mean,
        "selected_alpha": best_alpha,
        "test_mean": {
            "mf_only": float(np.mean(mf_test)),
            "alpha_1.0": float(np.mean(dp1_test)),
            f"alpha_{best_alpha}": float(np.mean(best_test)),
        },
        "test_gain_vs_mf": {
            "alpha_1.0": float(1 - np.mean(dp1_test) / np.mean(mf_test)),
            f"alpha_{best_alpha}": float(1 - np.mean(best_test) / np.mean(mf_test)),
        },
        "dev_scores": {str(k): v for k, v in dev_scores.items()},
        "test_scores": {str(k): v for k, v in test_scores.items()},
    }
    (OUT / f"alpha_sweep_s{a.seed}.json").write_text(json.dumps(payload, indent=2),
                                                     encoding="utf-8")

    print(f"\n{'alpha':>7} {'dev mean':>10}")
    for alpha in a.alphas:
        mark = "  <-- selected" if alpha == best_alpha else ""
        print(f"{alpha:7.2f} {dev_mean[alpha]:10.5f}{mark}")
    print(f"\ntest (untouched, n={len(test)}):")
    print(f"  MF only          {np.mean(mf_test):.5f}")
    print(f"  alpha = 1.0      {np.mean(dp1_test):.5f}  "
          f"({1 - np.mean(dp1_test) / np.mean(mf_test):+.1%} vs MF)")
    print(f"  alpha = {best_alpha:<4}     {np.mean(best_test):.5f}  "
          f"({1 - np.mean(best_test) / np.mean(mf_test):+.1%} vs MF)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
