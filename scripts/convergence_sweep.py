"""Convergence sweep: how many updates per cell does each model need?

Prints a table of mean held-out W2 as a function of the update budget, for both
the single-parameter and the two-parameter model. The point is to show whether
either model was still improving when the comparison stopped, and whether the
two-parameter model eventually overtakes the single-parameter one once the extra
parameter axis is actually learned.

Usage:
    python scripts/convergence_sweep.py --budgets 200 400 800
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

from twoparam import protocol as P  # noqa: E402
from twoparam.data import load_cell  # noqa: E402
from twoparam.evaluate import evaluate_cells  # noqa: E402
from twoparam.model import mf_sample  # noqa: E402
from twoparam.protocol import Config  # noqa: E402
from twoparam.train import load_model, train as train_two  # noqa: E402

from compare_one_vs_two import sample_one_param, train_one_param  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=int, nargs="+", default=[200, 400, 800])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", type=Path,
                    default=P.RESULTS / "compare" / "convergence.json")
    a = ap.parse_args()

    config = Config(n_train=P.N_TRAIN, device="cuda")
    full = P.full_grid()
    index = P.grid_index()
    slice_points = P.baseline_slice()
    slice_keys = {p.key for p in slice_points}

    from twoparam.cli import _design
    design = _design("lhs20", n_points=20)
    lhs_points = [index[i, j] for i, j in design.points]
    lhs_keys = {p.key for p in lhs_points}
    held = [p for p in full if p.key not in lhs_keys and p.key not in slice_keys]

    # Preload once: sampling and scoring repeatedly dominates the runtime.
    sources, targets = {}, {}
    for point in full:
        cell = load_cell(P.RESULTS / "data" / "full", point.key)
        sources[point.key] = cell["clouds"]["source"]
        targets[point.key] = cell["clouds"]["reference"]

    def score_one(model, points):
        pred = {p.key: sample_one_param(model, sources[p.key], p.sigma) for p in points}
        ref = {p.key: targets[p.key] for p in points}
        rows, _ = evaluate_cells(pred, ref, bins=config.bins, workers=4, log=lambda t: None)
        return float(np.mean([r["w2"] for r in rows]))

    def score_two(model, points):
        pred = {p.key: mf_sample(model, torch.from_numpy(sources[p.key]), p.sigma, p.mu).numpy()
                for p in points}
        ref = {p.key: targets[p.key] for p in points}
        rows, _ = evaluate_cells(pred, ref, bins=config.bins, workers=4, log=lambda t: None)
        return float(np.mean([r["w2"] for r in rows]))

    results = []
    print(f"{'updates/cell':>13} {'one-param steps':>16} {'two-param steps':>16} "
          f"{'one held W2':>12} {'two held W2':>12} {'better':>8}")
    for per_cell in a.budgets:
        one_steps = per_cell * len(slice_points)
        two_steps = per_cell * len(lhs_points)
        one = train_one_param(slice_points, config, steps=one_steps, batch_size=a.batch_size)
        train_dir = P.ensure_results("train", f"lhs20_conv{per_cell}")
        summary = train_two(P.RESULTS / "data" / "full", train_dir, lhs_points, config,
                            steps=two_steps, batch_size=a.batch_size, seed=88, log=lambda t: None)
        two = load_model(train_dir / "checkpoints" / "mf_two_param.pt")
        w1 = score_one(one, held)
        w2 = score_two(two, held)
        better = "two" if w2 < w1 else "one"
        print(f"{per_cell:13d} {one_steps:16d} {two_steps:16d} "
              f"{w1:12.5f} {w2:12.5f} {better:>8}")
        results.append({"updates_per_cell": per_cell, "one_steps": one_steps,
                        "two_steps": two_steps, "one_heldout_w2": w1,
                        "two_heldout_w2": w2, "better": better,
                        "two_param_loss": summary["final_loss"],
                        "two_param_seconds": summary["seconds"]})
        del one, two
        torch.cuda.empty_cache()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"n_heldout": len(held), "runs": results,
                                 "batch_size": a.batch_size}, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
