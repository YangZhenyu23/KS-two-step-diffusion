"""Command-line entry points for the full-scale two-parameter study.

Typical paper-scale sequence::

    python -m fullscale.cli design
    python -m fullscale.cli data                       # 200 cells, ~4.5 h at 100k
    python -m fullscale.cli train --design lhs40 --updates-per-cell 400
    python -m fullscale.cli train --design baseline --updates-per-cell 400
    python -m fullscale.cli evaluate --design lhs40
    python -m fullscale.cli ablation                   # one- vs two-parameter
    python -m fullscale.cli continuity                 # refinement + noise split

Every command is idempotent where it makes sense: data resumes from the manifest,
training writes into its own directory, evaluation refuses to run if the
checkpoint does not match the current grid.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import sys

import numpy as np

from . import grid as G
from .grid import Cell, Config, ensure


def log(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


# ------------------------------------------------------------------ design ---
def cmd_design(args) -> int:
    from .lhs import latin_hypercube, random_subset

    lhs = latin_hypercube(args.points, seed=args.seed)
    rnd = random_subset(args.points, seed=args.seed + 1)
    payload = {
        "schema": "fullscale_design_v1",
        "grid": {"sigma": G.SIGMA_GRID, "mu": G.MU_GRID,
                 "n_cells": len(G.SIGMA_GRID) * len(G.MU_GRID)},
        "lhs": lhs.as_dict(),
        "random": rnd.as_dict(),
        "comparison": {
            "min_distance_ratio": lhs.min_distance / max(rnd.min_distance, 1e-12),
            "mean_nn_ratio": lhs.mean_nn_distance / max(rnd.mean_nn_distance, 1e-12),
            "abs_correlation_lhs": abs(lhs.correlation),
            "abs_correlation_random": abs(rnd.correlation),
        },
    }
    out = ensure("design") / "design.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"LHS-{args.points}: min_dist={lhs.min_distance:.4f} "
        f"(random {rnd.min_distance:.4f}, {lhs.min_distance / rnd.min_distance:.2f}x)")
    log(f"             corr={lhs.correlation:+.4f} (random {rnd.correlation:+.4f})")
    log(f"             sigma counts {lhs.sigma_counts}")
    log(f"             mu counts    {lhs.mu_counts}")
    log(f"wrote {out}")
    return 0


# -------------------------------------------------------------------- data ---
def cmd_data(args) -> int:
    """Delegate to the resumable generator with the CLI's arguments."""
    from . import generate
    import sys as _sys

    argv = []
    if args.n:
        argv += ["--n", str(args.n)]
    if args.force:
        argv += ["--force"]
    if args.device:
        argv += ["--device", args.device]
    saved = _sys.argv
    try:
        _sys.argv = ["fullscale.generate"] + argv
        return generate.main(argv)
    finally:
        _sys.argv = saved


# ------------------------------------------------------------------- train ---
def _design_cells(name: str, args) -> list[Cell]:
    from .lhs import latin_hypercube, random_subset

    if name == "baseline":
        return G.baseline_slice()
    if name == "lhs40":
        return [Cell(i, j) for i, j in latin_hypercube(args.points, seed=args.seed).points]
    if name == "random40":
        return [Cell(i, j) for i, j in random_subset(args.points, seed=args.seed + 1).points]
    raise ValueError(f"unknown design {name}")


def cmd_train(args) -> int:
    from .train import train_one_param, train_two_param

    config = Config(n_train=args.n, device=args.device)
    cells = _design_cells(args.design, args)
    steps = args.updates_per_cell * len(cells)
    out = ensure("train", args.design)
    log(f"design={args.design}  cells={len(cells)}  updates/cell={args.updates_per_cell} "
        f"-> {steps} steps")
    if args.design == "baseline":
        result = train_one_param(cells, config, out, steps=steps,
                                 batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    else:
        result = train_two_param(cells, config, out, steps=steps,
                                 batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                                 checkpoint_every=args.checkpoint_every)
    (out / "summary.json").write_text(json.dumps({
        "design": args.design, "cells": [c.key for c in cells],
        "updates_per_cell": args.updates_per_cell, **result,
    }, indent=2), encoding="utf-8")
    return 0


# ---------------------------------------------------------------- evaluate ---
def cmd_evaluate(args) -> int:
    from .train import load_cell, load_two
    from .evaluate import score_many, write_csv, write_json
    from .model import sample_two_param
    import torch

    design_cells = _design_cells(args.design, args)
    train_keys = {c.key for c in design_cells}
    model, payload = load_two(ensure("train", args.design) / "mf_two_param.pt", args.device)
    if payload.get("sigma_grid") != G.SIGMA_GRID or payload.get("mu_grid") != G.MU_GRID:
        raise SystemExit("checkpoint was trained on a different grid")

    all_cells = G.full_grid()
    pairs, meta = {}, []
    started = time.perf_counter()
    for position, cell in enumerate(all_cells, start=1):
        payload_cell = load_cell(cell)
        source = payload_cell["clouds"]["source"]
        prediction = sample_two_param(model, torch.from_numpy(source), cell.sigma, cell.mu,
                                     steps=args.steps).numpy().astype(np.float32)
        pairs[cell.key] = (payload_cell["clouds"]["reference"], prediction)
        meta.append({**cell.meta(), "split": "train" if cell.key in train_keys else "heldout"})
        if position % 20 == 0:
            log(f"  generated {position}/{len(all_cells)} predictions "
                f"({time.perf_counter() - started:.0f}s)")

    rows = score_many(pairs, bins=args.bins, workers=args.workers, log=log)
    by_key = {r["key"]: r for r in rows}
    for entry in meta:
        entry.update({k: v for k, v in by_key[entry["key"]].items() if k != "key"})

    out = ensure("evaluate", args.design)
    write_csv(out / "grid_metrics.csv", meta)
    train_rows = [r for r in meta if r["split"] == "train"]
    held_rows = [r for r in meta if r["split"] == "heldout"]
    slice_rows = [r for r in meta if abs(r["mu"] - G.BASELINE_MU) < 1e-12]
    write_csv(out / "baseline_slice_metrics.csv", slice_rows)

    summary = {
        "design": args.design,
        "steps": args.steps,
        "n_cells": len(meta),
        "n_train_cells": len(train_rows),
        "n_heldout_cells": len(held_rows),
        "train_mean_w2": float(np.mean([r["w2"] for r in train_rows])),
        "heldout_mean_w2": float(np.mean([r["w2"] for r in held_rows])) if held_rows else None,
        "all_mean_w2": float(np.mean([r["w2"] for r in meta])),
        "all_max_w2": float(np.max([r["w2"] for r in meta])),
        "baseline_slice_mean_w2": float(np.mean([r["w2"] for r in slice_rows])),
        "floor_16bins": G.FLOOR_16,
        "all_mean_over_floor": float(np.mean([r["w2"] for r in meta]) / G.FLOOR_16),
        "certified": all(r["solver_result_code"] == 1 for r in meta),
        "max_marginal_residual": float(np.max([r["marginal_residual"] for r in meta])),
        "max_primal_dual_gap": float(np.max([r["primal_dual_gap"] for r in meta])),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(out / "evaluation_summary.json", summary)
    log("\nsummary:")
    for key, value in summary.items():
        log(f"  {key}: {value}")
    return 0


# ---------------------------------------------------------------- ablation ---
def cmd_ablation(args) -> int:
    """Score the one-parameter control and the two-parameter model on the same cells."""
    from .train import load_cell, load_one, load_two
    from .evaluate import score_many, write_csv, write_json
    from .model import sample_one_param, sample_two_param
    import torch

    two, two_payload = load_two(ensure("train", "lhs40") / "mf_two_param.pt", args.device)
    one, one_payload = load_one(ensure("train", "baseline") / "mf_one_param.pt", args.device)

    lhs_cells = _design_cells("lhs40", args)
    lhs_keys = {c.key for c in lhs_cells}
    slice_keys = {c.key for c in G.baseline_slice()}
    held = [c for c in G.full_grid() if c.key not in lhs_keys and c.key not in slice_keys]
    log(f"ablation cells: {len(held)} (excluded {len(lhs_keys)} LHS + {len(slice_keys)} baseline row)")

    pairs_one, pairs_two, meta = {}, {}, []
    for cell in held:
        payload = load_cell(cell)
        source = payload["clouds"]["source"]
        reference = payload["clouds"]["reference"]
        pairs_one[cell.key] = (reference, sample_one_param(
            one, torch.from_numpy(source), cell.sigma).numpy().astype(np.float32))
        pairs_two[cell.key] = (reference, sample_two_param(
            two, torch.from_numpy(source), cell.sigma, cell.mu,
            steps=args.steps).numpy().astype(np.float32))
        meta.append(cell.meta())

    rows_one = score_many(pairs_one, bins=args.bins, workers=args.workers, log=lambda t: None)
    rows_two = score_many(pairs_two, bins=args.bins, workers=args.workers, log=lambda t: None)
    by_one = {r["key"]: r["w2"] for r in rows_one}
    by_two = {r["key"]: r["w2"] for r in rows_two}

    detail = []
    for entry in meta:
        w1, w2 = by_one[entry["key"]], by_two[entry["key"]]
        detail.append({**entry, "w2_one_param": w1, "w2_two_param": w2,
                       "improvement": w1 - w2, "ratio": w2 / w1,
                       "mu_off_baseline": abs(math.log10(entry["mu"]))})
    detail.sort(key=lambda r: (r["mu_off_baseline"], r["sigma"]))

    gains = [r["improvement"] for r in detail]
    m1 = float(np.mean([r["w2_one_param"] for r in detail]))
    m2 = float(np.mean([r["w2_two_param"] for r in detail]))
    off = np.array([r["mu_off_baseline"] for r in detail])
    imp = np.array(gains)
    summary = {
        "n_cells": len(detail),
        "one_param_mean_w2": m1,
        "two_param_mean_w2": m2,
        "relative_reduction": float(1 - m2 / m1),
        "median_improvement": float(np.median(gains)),
        "two_param_wins": int(sum(1 for v in gains if v > 0)),
        "one_param_wins": int(sum(1 for v in gains if v < 0)),
        "corr_improvement_vs_mu_distance": float(np.corrcoef(off, imp)[0, 1])
        if off.std() > 0 and imp.std() > 0 else None,
        "floor_16bins": G.FLOOR_16,
        "one_param_over_floor": m1 / G.FLOOR_16,
        "two_param_over_floor": m2 / G.FLOOR_16,
    }
    out = ensure("ablation")
    write_csv(out / "detail.csv", detail)
    write_json(out / "summary.json", summary)
    log("\nablation:")
    for key, value in summary.items():
        log(f"  {key}: {value}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--points", type=int, default=40)
        p.add_argument("--seed", type=int, default=20261107)
        p.add_argument("--n", type=int, default=G.N_PARTICLES)
        p.add_argument("--bins", type=int, default=G.BINS)
        p.add_argument("--device", default="cuda")
        p.add_argument("--workers", type=int, default=4)

    d = sub.add_parser("design")
    common(d)
    d.set_defaults(func=cmd_design)

    g = sub.add_parser("data")
    g.add_argument("--n", type=int, default=None)
    g.add_argument("--device", default=None)
    g.add_argument("--force", action="store_true")
    g.set_defaults(func=cmd_data)

    t = sub.add_parser("train")
    common(t)
    t.add_argument("--design", default="lhs40")
    t.add_argument("--updates-per-cell", type=int, default=400)
    t.add_argument("--batch-size", type=int, default=4096)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--checkpoint-every", type=int, default=0)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate")
    common(e)
    e.add_argument("--design", default="lhs40")
    e.add_argument("--steps", type=int, default=1)
    e.set_defaults(func=cmd_evaluate)

    a = sub.add_parser("ablation")
    common(a)
    a.add_argument("--steps", type=int, default=1)
    a.set_defaults(func=cmd_ablation)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
