"""Command-line entry points for the (sigma, mu) two-parameter study.

Usage (run from the repository root, with the GPU virtual environment)::

    # 1. design
    python -m twoparam.cli design

    # 2. generate every cell of the 10x10 grid (needed for the leave-out test)
    python -m twoparam.cli data --all

    # 3. train on the LHS-20 subset
    python -m twoparam.cli train --design lhs20

    # 4. evaluate on the full grid and on the baseline slice
    python -m twoparam.cli evaluate --design lhs20

    # 5. parameter-continuity measurement
    python -m twoparam.cli continuity
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from . import protocol as P
from .protocol import Config, Point


def _log(text: str) -> None:
    print(text, flush=True)


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _design(name: str, *, n_points: int = 20, seed: int = 20261107):
    from .lhs import latin_hypercube

    if name == "lhs20":
        # mu = 1 is the upstream baseline value. Forcing that level into the
        # design guarantees the single-parameter slice is anchored by real
        # training data rather than left to interpolation, which is what makes
        # the head-to-head comparison against the baseline meaningful.
        return latin_hypercube(n_points, seed=seed, force_include=[(0, P.MU_INDEX)])
    if name == "random20":
        from .lhs import random_subset
        return random_subset(n_points, seed=seed)
    raise ValueError(f"unknown design {name}")


def _points_from_design(design) -> list[Point]:
    index = P.grid_index()
    return [P.Point(i, j, P.SIGMA_GRID[i], P.MU_GRID[j], in_lhs=True) for i, j in design.points]


# --------------------------------------------------------------- commands ---
def cmd_design(args) -> dict:
    from .lhs import latin_hypercube, random_subset

    lhs = latin_hypercube(args.points, seed=args.seed)
    rnd = random_subset(args.points, seed=args.seed)
    payload = {
        "schema": "twoparam_design_v1",
        "sigma_grid": P.SIGMA_GRID,
        "mu_grid": P.MU_GRID,
        "grid_size": len(P.SIGMA_GRID) * len(P.MU_GRID),
        "n_points": args.points,
        "lhs": lhs.as_dict(),
        "random": rnd.as_dict(),
        "comparison": {
            "min_distance_ratio": lhs.min_distance / max(rnd.min_distance, 1e-12),
            "mean_nn_ratio": lhs.mean_nn_distance / max(rnd.mean_nn_distance, 1e-12),
        },
    }
    out = P.ensure_results("design") / "design.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"LHS-{args.points}  min_dist={lhs.min_distance:.4f} (random {rnd.min_distance:.4f})")
    _log(f"          mean_nn={lhs.mean_nn_distance:.4f} (random {rnd.mean_nn_distance:.4f})")
    _log(f"          corr={lhs.correlation:+.3f} (random {rnd.correlation:+.3f})")
    _log(f"          quadrants={lhs.quadrants} (random {rnd.quadrants})")
    _log(f"wrote {out}")
    return payload


def cmd_data(args) -> dict:
    from .data import generate_cells

    config = Config(n_train=args.n_train, n_eval=args.n_eval, device=args.device, seed=args.seed)
    if args.all:
        cells = P.full_grid()
        name = "full"
    else:
        design = _design(args.design, n_points=args.points, seed=args.seed_design)
        cells = _points_from_design(design)
        name = args.design
    destination = P.ensure_results("data", name)
    _log(f"generating {len(cells)} cells at N={config.n_train} -> {destination}")
    manifest = generate_cells(cells, destination, config, progress=lambda t: _log(f"  {t}"))
    _log(f"done: {len(manifest['cells'])} cells in {manifest['elapsed_seconds']:.1f}s")
    return manifest


def cmd_train(args) -> dict:
    from .train import train

    design = _design(args.design, n_points=args.points, seed=args.seed_design)
    points = _points_from_design(design)
    config = Config(n_train=args.n_train, device=args.device, seed=args.seed)
    data_dir = P.RESULTS / "data" / "full"
    if not (data_dir / "manifest.json").is_file():
        data_dir = P.RESULTS / "data" / args.design
    destination = P.ensure_results("train", args.design)
    _log(f"training on {len(points)} cells from {data_dir}")
    result = train(data_dir, destination, points, config, steps=args.steps,
                   batch_size=args.batch_size, lr=args.lr, seed=args.seed, log=_log)
    (destination / "train_summary.json").write_text(
        json.dumps({"design": args.design, "points": [p.meta() for p in points], **result},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def cmd_evaluate(args) -> dict:
    from .data import load_cell
    from .evaluate import evaluate_cells, write_csv, write_json
    from .train import generate, load_model

    design = _design(args.design, n_points=args.points, seed=args.seed_design)
    train_points = _points_from_design(design)
    train_keys = {p.key for p in train_points}
    config = Config(n_train=args.n_train, device=args.device)
    data_dir = P.RESULTS / "data" / "full"
    model = load_model(P.RESULTS / "train" / args.design / "checkpoints" / "mf_two_param.pt",
                       device=args.device)

    index = P.grid_index()
    evaluations = []

    # (a) full grid, split into the LHS subset and the held-out remainder
    full = P.full_grid()
    predictions, references, meta_rows = {}, {}, []
    for point in full:
        cell = load_cell(data_dir, point.key)
        predictions[point.key] = generate(model, cell["clouds"]["source"], point.sigma, point.mu)
        references[point.key] = cell["clouds"]["reference"]
        meta_rows.append({**point.meta(), "split": "train" if point.key in train_keys else "heldout"})
    rows, _ = evaluate_cells(predictions, references, bins=config.bins, workers=args.workers, log=_log)
    by_key = {r["key"]: r for r in rows}
    for meta in meta_rows:
        meta.update({k: v for k, v in by_key[meta["key"]].items() if k != "key"})
    out = P.ensure_results("evaluate", args.design)
    write_csv(out / "grid_metrics.csv", meta_rows)
    evaluations.append(("grid", meta_rows))

    # (b) the baseline slice (mu = 1) for the direct single-parameter comparison
    slice_points = [p for p in full if abs(p.mu - P.BASELINE_MU) < 1e-12]
    slice_rows = [r for r in meta_rows if abs(r["mu"] - P.BASELINE_MU) < 1e-12]
    write_csv(out / "baseline_slice_metrics.csv", slice_rows)
    evaluations.append(("baseline_slice", slice_rows))

    train_rows = [r for r in meta_rows if r["split"] == "train"]
    held_rows = [r for r in meta_rows if r["split"] == "heldout"]
    summary = {
        "design": args.design,
        "n_train_cells": len(train_rows),
        "n_heldout_cells": len(held_rows),
        "train_mean_w2": float(np.mean([r["w2"] for r in train_rows])) if train_rows else None,
        "heldout_mean_w2": float(np.mean([r["w2"] for r in held_rows])) if held_rows else None,
        "all_mean_w2": float(np.mean([r["w2"] for r in meta_rows])),
        "all_max_w2": float(np.max([r["w2"] for r in meta_rows])),
        "baseline_slice_mean_w2": float(np.mean([r["w2"] for r in slice_rows])) if slice_rows else None,
        "certified": all(r["solver_result_code"] == 1 for r in meta_rows),
        "max_marginal_residual": float(np.max([r["marginal_residual"] for r in meta_rows])),
        "max_primal_dual_gap": float(np.max([r["primal_dual_gap"] for r in meta_rows])),
    }
    write_json(out / "evaluation_summary.json", summary)
    _log("\nsummary:")
    for key, value in summary.items():
        _log(f"  {key}: {value}")
    return summary


def cmd_continuity(args) -> dict:
    from .continuity import run_continuity

    result = run_continuity(design=args.design, points=args.points, workers=args.workers,
                            bins=args.bins, device=args.device)
    return result


# ------------------------------------------------------------------ main ---
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--design", default="lhs20")
        p.add_argument("--points", type=int, default=20)
        p.add_argument("--seed-design", type=int, default=20261107)
        p.add_argument("--n-train", type=int, default=P.N_TRAIN)
        p.add_argument("--device", default="cuda")
        p.add_argument("--workers", type=int, default=4)

    d = sub.add_parser("design", help="build the LHS design and compare it to random sampling")
    d.add_argument("--points", type=int, default=20)
    d.add_argument("--seed", type=int, default=20261107)
    d.set_defaults(func=cmd_design)

    g = sub.add_parser("data", help="generate physical clouds on the grid")
    g.add_argument("--all", action="store_true", help="generate all 100 cells")
    g.add_argument("--design", default="lhs20")
    g.add_argument("--points", type=int, default=20)
    g.add_argument("--seed-design", type=int, default=20261107)
    g.add_argument("--n-train", type=int, default=P.N_TRAIN)
    g.add_argument("--n-eval", type=int, default=P.N_EVAL)
    g.add_argument("--seed", type=int, default=88)
    g.add_argument("--device", default="cuda")
    g.set_defaults(func=cmd_data)

    t = sub.add_parser("train", help="train the two-parameter MeanFlow")
    common(t)
    t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--seed", type=int, default=88)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="score the trained model on the full grid")
    common(e)
    e.set_defaults(func=cmd_evaluate)

    c = sub.add_parser("continuity", help="measure parameter continuity of the cloud family")
    c.add_argument("--design", default="lhs20")
    c.add_argument("--points", type=int, default=20)
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--bins", type=int, nargs="+", default=[16, 32])
    c.add_argument("--device", default="cuda")
    c.set_defaults(func=cmd_continuity)

    args = ap.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
