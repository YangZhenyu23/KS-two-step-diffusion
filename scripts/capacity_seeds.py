"""Multi-seed check of the capacity and architecture sweep.

The single-seed capacity sweep gave

    mlp64       0.1419
    mlp128      0.1610      <- wider is worse
    mlp256      0.1456
    fourier128  0.1342      <- different encoding is better

A non-monotone width response and a 5% architecture gain are both the same order
as the design-to-design spread seen earlier (0.153 to 0.169 for identical
protocols). One seed per configuration therefore cannot separate "wider helps"
from "this draw was lucky".

This script repeats each configuration with several training seeds on the *same*
40-cell design, so the design is held fixed and only the optimisation randomness
varies. It then reports mean, sd and a paired comparison against the baseline
configuration.

Interpretation guide
--------------------
* if the width trend stays flat across seeds, capacity in parameters is not the
  limit and the paper should say so
* if fourier128 keeps its lead across seeds, the limit is the *parameter
  encoding*, not the parameter count -- which matches the paper's own
  description (sinusoidal embeddings of r, t, sigma) over the released code
  (plain concatenation)

Usage:
    python scripts/capacity_seeds.py --configs mlp64 mlp128 fourier128 --seeds 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fullscale import grid as G  # noqa: E402
from fullscale.evaluate import score_many  # noqa: E402
from fullscale.grid import Config  # noqa: E402
from fullscale.train import load_cell  # noqa: E402

from sample_efficiency import (  # noqa: E402
    OUT, frozen_test_set, stratified_subset, write_csv, write_json, log,
)
from capacity_sweep import build, predict, train_variant, UPDATES_PER_CELL  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["mlp64", "mlp128", "fourier128"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4096)
    a = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    test, pool = frozen_test_set()
    cells = stratified_subset(pool, 40, seed=20261107)
    steps = UPDATES_PER_CELL * len(cells)
    config = Config(n_train=G.N_PARTICLES, device="cuda")
    log(f"{len(test)} test cells, {len(cells)} training cells, {steps} steps per run, "
        f"{a.seeds} seeds per config")

    payloads = {cell.key: load_cell(cell) for cell in test}

    rows = []
    for name in a.configs:
        for seed in (88, 89, 90)[: a.seeds]:
            log(f"--- {name} seed {seed} ---")
            model = build(name)
            info = model.describe()
            model = train_variant(model, cells, config, steps=steps,
                                  batch_size=a.batch_size, seed=seed)
            pairs = {}
            for cell in test:
                payload = payloads[cell.key]
                pairs[cell.key] = (
                    payload["clouds"]["reference"],
                    predict(model, torch.from_numpy(payload["clouds"]["source"]),
                            cell.sigma, cell.mu))
            scored = score_many(pairs, bins=G.BINS, workers=4, log=lambda t: None)
            values = [r["w2"] for r in scored]
            row = {"config": name, **info, "seed": seed,
                   "test_mean_w2": float(np.mean(values)),
                   "test_over_floor": float(np.mean(values) / G.FLOOR_16)}
            rows.append(row)
            log(f"  TEST {row['test_mean_w2']:.5f}  ({row['test_over_floor']:.2f}x floor)")
            del model
            torch.cuda.empty_cache()

    out = OUT / "capacity"
    write_csv(out / "capacity_seeds.csv", rows)

    print(f"\n{'config':<14} {'params':>9} {'runs':>5} {'mean':>9} {'sd':>8} "
          f"{'min':>9} {'max':>9}")
    summary = {}
    for name in a.configs:
        group = [r for r in rows if r["config"] == name]
        values = [r["test_mean_w2"] for r in group]
        summary[name] = {
            "params": group[0]["params"],
            "runs": len(values),
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)), "max": float(np.max(values)),
            "values": values,
            "kind": group[0]["kind"],
            "width": group[0]["width"],
        }
        s = summary[name]
        print(f"{name:<14} {s['params']:>9,} {s['runs']:5d} {s['mean']:9.5f} "
              f"{s['sd']:8.5f} {s['min']:9.5f} {s['max']:9.5f}")

    base = "mlp64"
    if base in summary:
        print(f"\ncomparison against {base}:")
        for name in a.configs:
            if name == base:
                continue
            diff = summary[name]["mean"] - summary[base]["mean"]
            se = float(np.sqrt(summary[name]["sd"] ** 2 / max(summary[name]["runs"], 1)
                               + summary[base]["sd"] ** 2 / max(summary[base]["runs"], 1)))
            ratio = diff / se if se > 0 else float("nan")
            verdict = ("better" if ratio < -2 else "worse" if ratio > 2 else "not resolvable")
            print(f"  {name:<14} {diff:+.5f}  ({ratio:+.1f} SE)  {verdict}")
            summary[name]["vs_baseline"] = {"difference": diff, "se": se, "verdict": verdict}

    write_json(out / "capacity_seeds.json", {"steps": steps, "summary": summary})
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
