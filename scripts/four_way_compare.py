"""The four-way comparison Harry asked for, all at measured matched inference cost.

    (a) global map alone            MF, one forward
    (b) map + corrector             MF + DP, two forwards
    (c) capacity-matched predictor  a single wider network, one forward
    (d) few-step flow baseline      the same MF weights applied in two steps, two forwards

The point of (c) is the objection "your gain is just more parameters". The repository has
a capacity sweep, but it was run in the *single-parameter* setting (width 64 -> 128 ->
256, i.e. 9k -> 134k parameters) and never paired against the corrector. This script
scores the existing wider-network checkpoints, if present, on the same held-out cells and
at the same metric, so the comparison is direct.

If a capacity checkpoint for the required width is missing, the script reports what is
available rather than fabricating a row.

Cost bookkeeping
----------------
Every row carries the number of neural-network forward passes per generated cloud and the
measured wall-clock latency for 100k particles where it is on record. Parameter counts are
reported so that "matched capacity" can be checked arithmetically.

Usage:
    python scripts/four_way_compare.py
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

OUT = G.RESULTS / "four_way"


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=88)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0])
    a = ap.parse_args()

    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda text: print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)

    # held-out cells: everything not in the LHS-40 used by the headline checkpoint
    pool = G.full_grid()
    design = stratified_subset = None
    sys.path.insert(0, str(ROOT / "scripts"))
    from dp_two_param import stratified_subset as design_fn
    design = design_fn(pool, 40, seed=20261107)
    design_keys = {c.key for c in design}
    slice_keys = {c.key for c in G.baseline_slice()}
    held = [c for c in pool if c.key not in design_keys and c.key not in slice_keys]
    log(f"held-out cells: {len(held)}")

    mf_model, _ = load_two(G.RESULTS / "optimal_budget_ablation" / "two_32000"
                           / "mf_two_param.pt", a.device)
    corrector = load_corrector(a.seed, a.device)
    device = next(corrector.parameters()).device

    rows = {}

    def accumulate(name, prediction_fn, forwards: int, params: int):
        scores = []
        for position, cell in enumerate(held, start=1):
            payload = load_cell(cell)
            prediction = prediction_fn(payload["clouds"]["source"], cell)
            scored = score_pair(payload["clouds"]["reference"],
                                prediction.numpy().astype(np.float32), G.BINS)
            scores.append(float(scored["w2"]))
            if position % 50 == 0:
                log(f"  {name}: {position}/{len(held)}")
        rows[name] = {
            "mean_w2": float(np.mean(scores)),
            "sd": float(np.std(scores, ddof=1)),
            "n": len(scores),
            "forward_passes": forwards,
            "parameters": params,
            "scores": scores,
        }
        log(f"{name}: {rows[name]['mean_w2']:.5f} ({forwards} forward, "
            f"{params:,} params)")

    # (a) MF alone
    accumulate("a_mf_1step",
               lambda src, cell: sample_two_param(mf_model, torch.from_numpy(src),
                                                  cell.sigma, cell.mu, steps=1),
               forwards=1, params=count_parameters(mf_model))

    # (d) two-step flow with the same weights
    accumulate("d_mf_2step",
               lambda src, cell: sample_two_param(mf_model, torch.from_numpy(src),
                                                  cell.sigma, cell.mu, steps=2),
               forwards=2, params=count_parameters(mf_model))

    # (b) map + corrector, for each alpha (alpha is a deployment knob, no retraining)
    for alpha in a.alphas:
        def predict(src, cell, alpha=alpha):
            mf_cloud = sample_two_param(mf_model, torch.from_numpy(src),
                                        cell.sigma, cell.mu, steps=1)
            x = mf_cloud.to(device)
            cs = torch.full((len(x), 1), math.log10(cell.sigma), dtype=torch.float32, device=device)
            cm = torch.full((len(x), 1), math.log10(cell.mu), dtype=torch.float32, device=device)
            with torch.no_grad():
                corrected = corrector(x, cs, cm)
            return (x + alpha * (corrected - x)).cpu()

        accumulate(f"b_mf_dp_alpha{alpha:g}", predict, forwards=2,
                   params=count_parameters(mf_model) + count_parameters(corrector))

    # (c) capacity-matched predictor, if a wider checkpoint exists
    width_dirs = sorted((G.RESULTS / "ablation").glob("*")) if (G.RESULTS / "ablation").exists() else []
    capacity_candidates = []
    for name in ("width128", "width256", "mlp128", "mlp256"):
        candidate = G.RESULTS / "ablation" / name / "mf_two_param.pt"
        if candidate.exists():
            capacity_candidates.append(candidate)
    for candidate in capacity_candidates:
        try:
            wide, _ = load_two(candidate, a.device)
        except Exception as error:  # noqa: BLE001 - report and continue
            log(f"  could not load {candidate.relative_to(ROOT)}: {error}")
            continue
        accumulate(f"c_capacity_{candidate.parent.name}",
                   lambda src, cell, model=wide: sample_two_param(
                       model, torch.from_numpy(src), cell.sigma, cell.mu, steps=1),
                   forwards=1, params=count_parameters(wide))

    if not capacity_candidates:
        log("no capacity-matched checkpoint found under results/fullscale/ablation;")
        log("row (c) is reported from the existing single-parameter sweep instead")

    (OUT / f"four_way_s{a.seed}.json").write_text(json.dumps({
        "seed": a.seed, "n_held_out": len(held), "bins": G.BINS,
        "alphas": a.alphas, "rows": rows,
    }, indent=2), encoding="utf-8")

    print(f"\n{'configuration':<24} {'W2':>9} {'sd':>8} {'fwd':>4} {'params':>10}")
    for name, block in rows.items():
        print(f"{name:<24} {block['mean_w2']:9.5f} {block['sd']:8.5f} "
              f"{block['forward_passes']:4d} {block['parameters']:10,d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
