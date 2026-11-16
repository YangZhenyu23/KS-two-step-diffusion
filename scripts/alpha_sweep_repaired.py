"""Alpha sweep re-run on a held-out set derived from the checkpoint, not recomputed.

The earlier ``alpha_sweep.py`` built its dev/test split from
``stratified_subset(pool, 40, seed=20261107)``, which no longer reproduces the training
design of the checkpoint being scored. 32 of its 144 cells were in that checkpoint's
training set. This version reads the training cells from the checkpoint, so the split is
clean.

Protocol unchanged otherwise: pick alpha on dev, freeze it, then report on test. The split
is redone under a fixed seed and saved, so it is auditable.

Usage:
    python scripts/alpha_sweep_repaired.py --alphas 0 0.5 1 1.5 2 2.5 3 4
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

sys.path.insert(0, str(ROOT / "scripts"))
from rollout_compare import load_corrector  # noqa: E402

OUT = G.RESULTS / "alpha_sweep"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    ap.add_argument("--dev-fraction", type=float, default=0.35)
    ap.add_argument("--split-seed", type=int, default=424242)
    ap.add_argument("--seed", type=int, default=88)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bins", type=int, default=G.BINS)
    a = ap.parse_args()

    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda text: print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)

    mf_path = G.RESULTS / "optimal_budget_ablation" / "two_32000" / "mf_two_param.pt"
    payload = torch.load(mf_path, map_location="cpu", weights_only=False)
    train_keys = frozenset(payload["cells"])
    slice_keys = {c.key for c in G.baseline_slice()}
    held = [c for c in G.full_grid()
            if c.key not in train_keys and c.key not in slice_keys]
    log(f"checkpoint trains on {len(train_keys)} cells; held out {len(held)}")

    rng = np.random.default_rng(a.split_seed)
    order = rng.permutation(len(held))
    n_dev = int(round(a.dev_fraction * len(held)))
    dev = [held[k] for k in order[:n_dev]]
    test = [held[k] for k in order[n_dev:]]
    log(f"dev {len(dev)}, test {len(test)} (split seed {a.split_seed}, "
        f"derived from the checkpoint's own training cells)")
    (OUT / "split_repaired.json").write_text(json.dumps({
        "dev": [c.key for c in dev], "test": [c.key for c in test],
        "dev_fraction": a.dev_fraction, "split_seed": a.split_seed,
        "derived_from": str(mf_path.relative_to(ROOT)),
    }, indent=2), encoding="utf-8")

    mf_model, _ = load_two(mf_path, a.device)
    corrector = load_corrector(a.seed, a.device)
    device = next(corrector.parameters()).device

    cache = {}
    for position, cell in enumerate(held, start=1):
        item = load_cell(cell)
        cache[cell.key] = {
            "reference": item["clouds"]["reference"],
            "mf": sample_two_param(mf_model, torch.from_numpy(item["clouds"]["source"]),
                                   cell.sigma, cell.mu, steps=1),
            "sigma": cell.sigma, "mu": cell.mu,
        }
        if position % 40 == 0:
            log(f"  MF forward {position}/{len(held)}")

    def mean_at(alpha: float, cells: list) -> tuple[float, dict]:
        per_cell = {}
        for cell in cells:
            entry = cache[cell.key]
            mf_cloud = entry["mf"]
            if alpha == 0.0:
                prediction = mf_cloud.numpy().astype(np.float32)
            else:
                x = mf_cloud.to(device)
                cs = torch.full((len(x), 1), math.log10(entry["sigma"]),
                                dtype=torch.float32, device=device)
                cm = torch.full((len(x), 1), math.log10(entry["mu"]),
                                dtype=torch.float32, device=device)
                with torch.no_grad():
                    corrected = corrector(x, cs, cm)
                prediction = (x + alpha * (corrected - x)).cpu().numpy().astype(np.float32)
            per_cell[cell.key] = float(score_pair(entry["reference"], prediction, a.bins)["w2"])
        return float(np.mean(list(per_cell.values()))), per_cell

    dev_mean = {}
    for alpha in a.alphas:
        value, _ = mean_at(alpha, dev)
        dev_mean[alpha] = value
        log(f"  dev alpha={alpha:<5} {value:.5f}")

    best = min(dev_mean, key=lambda k: dev_mean[k])
    log(f"selected alpha* = {best} on dev")

    test_results = {}
    for alpha in sorted({0.0, 1.0, best}):
        value, per_cell = mean_at(alpha, test)
        test_results[f"alpha_{alpha:g}"] = {"mean": value, "per_cell": per_cell}
        log(f"  test alpha={alpha:<5} {value:.5f}")

    mf_mean = test_results["alpha_0"]["mean"]
    summary = {
        "alphas": a.alphas, "seed": a.seed, "bins": a.bins,
        "n_dev": len(dev), "n_test": len(test),
        "dev_fraction": a.dev_fraction, "split_seed": a.split_seed,
        "derived_from_checkpoint": str(mf_path.relative_to(ROOT)),
        "dev_mean": dev_mean, "selected_alpha": best,
        "test": {k: v["mean"] for k, v in test_results.items()},
        "test_gain_vs_mf": {k: float(1 - v["mean"] / mf_mean)
                            for k, v in test_results.items()},
    }
    (OUT / f"alpha_sweep_repaired_s{a.seed}.json").write_text(
        json.dumps({**summary, "test_per_cell": {k: v["per_cell"]
                                                 for k, v in test_results.items()}},
                   indent=2), encoding="utf-8")

    print(f"\n{'alpha':>7} {'dev':>10}")
    for alpha in a.alphas:
        mark = "  <-- selected" if alpha == best else ""
        print(f"{alpha:7.2f} {dev_mean[alpha]:10.5f}{mark}")
    print(f"\ntest (n={len(test)}, untouched during selection):")
    for key, value in summary["test"].items():
        gain = summary["test_gain_vs_mf"][key]
        print(f"  {key:<12} {value:.5f}  ({gain:+.1%} vs MF alone)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
