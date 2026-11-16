"""Rebuild the held-out sets from the checkpoints themselves, then rescore.

The defect this repairs
-----------------------
Four scripts recompute the training design at run time via
``stratified_subset(pool, 40, seed=20261107)``. That function changed during development,
so the design it returns today is NOT the design the headline checkpoints were trained on:

    headline two_32000 training cells (from the checkpoint):  s11_m05, s15_m05, s19_m05 ...
    stratified_subset(...) today:                            s00_m05, s03_m05, s05_m05, s19_m05 ...
    overlap: 6 / 40

Consequently 32 of the 144 "held-out" cells were in the training set of the checkpoint
being scored, and any number computed that way is contaminated on 22% of its cells.

The fix, and it is the only correct one: the authoritative list of training cells is the
one stored inside the checkpoint. Never recompute it.

This script does not retrain anything. It rescoringes the existing checkpoints on the
correctly-derived held-out sets, so the four-way table, the alpha sweep and the
capacity-matched row can all be re-reported honestly, and prints the old-vs-new delta so
the size of the contamination is visible.

Usage:
    python scripts/repair_eval_sets.py
"""
from __future__ import annotations

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

OUT = G.RESULTS / "repaired"


def checkpoint_cells(path: Path) -> frozenset[str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cells = payload.get("cells")
    if not cells:
        raise RuntimeError(f"{path} has no recorded training cells; cannot derive a clean "
                           f"held-out set")
    return frozenset(cells)


def held_out_from(train_keys: frozenset[str]) -> list:
    """Everything not trained on and not in the baseline slice."""
    slice_keys = {c.key for c in G.baseline_slice()}
    return [c for c in G.full_grid() if c.key not in train_keys and c.key not in slice_keys]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda text: print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)

    mf_path = G.RESULTS / "optimal_budget_ablation" / "two_32000" / "mf_two_param.pt"
    train_keys = checkpoint_cells(mf_path)
    held = held_out_from(train_keys)
    log(f"checkpoint training cells: {len(train_keys)}")
    log(f"correct held-out set     : {len(held)} cells")

    # what the old (buggy) derivation gave, for the delta column
    from dp_two_param import stratified_subset
    old_keys = frozenset(c.key for c in stratified_subset(G.full_grid(), 40, seed=20261107))
    old_held = held_out_from(old_keys)
    leak = train_keys & {c.key for c in old_held}
    log(f"old derivation gave      : {len(old_held)} cells, of which {len(leak)} were TRAINED ON")

    mf_model, _ = load_two(mf_path, "cuda")
    corrector = load_corrector(88, "cuda")
    device = next(corrector.parameters()).device

    # precompute MF output once per cell; both variants reuse it
    cache = {}
    for position, cell in enumerate(held, start=1):
        payload = load_cell(cell)
        source = torch.from_numpy(payload["clouds"]["source"])
        cache[cell.key] = {
            "reference": payload["clouds"]["reference"],
            "mf": sample_two_param(mf_model, source, cell.sigma, cell.mu, steps=1),
            "mf_2step": sample_two_param(mf_model, source, cell.sigma, cell.mu, steps=2),
            "sigma": cell.sigma, "mu": cell.mu,
        }
        if position % 40 == 0:
            log(f"  generated {position}/{len(held)}")

    def score(alpha: float, keys: list[str]) -> float:
        values = []
        for key in keys:
            entry = cache[key]
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
            values.append(float(score_pair(entry["reference"], prediction, G.BINS)["w2"]))
        return float(np.mean(values))

    def score_two_step(keys: list[str]) -> float:
        return float(np.mean([
            float(score_pair(cache[k]["reference"],
                             cache[k]["mf_2step"].numpy().astype(np.float32),
                             G.BINS)["w2"])
            for k in keys
        ]))

    clean_keys = [c.key for c in held]
    # score the same cell set the old scripts used, for a like-for-like delta
    old_keys_list = [c.key for c in old_held if c.key in cache]
    missing = [c.key for c in old_held if c.key not in cache]

    log("scoring on the CORRECT held-out set")
    results = {"clean": {}, "contaminated": {}}
    for alpha in (0.0, 1.0, 2.0):
        results["clean"][f"mf_dp_alpha{alpha:g}"] = score(alpha, clean_keys)
        log(f"  clean  alpha={alpha}: {results['clean'][f'mf_dp_alpha{alpha:g}']:.5f}")
    results["clean"]["mf_2step"] = score_two_step(clean_keys)
    log(f"  clean  mf_2step    : {results['clean']['mf_2step']:.5f}")

    if old_keys_list:
        log("scoring the same old cell set (excluding cells it does not contain)")
        for alpha in (0.0, 1.0, 2.0):
            results["contaminated"][f"mf_dp_alpha{alpha:g}"] = score(alpha, old_keys_list)
        results["contaminated"]["mf_2step"] = score_two_step(old_keys_list)
    if missing:
        log(f"  note: {len(missing)} old cells were not regenerated (they are training "
            f"cells of this checkpoint); they are excluded from the contaminated recount")

    payload = {
        "checkpoint": str(mf_path.relative_to(ROOT)),
        "n_train_cells": len(train_keys),
        "n_held_clean": len(clean_keys),
        "n_held_old": len(old_held),
        "n_leaked_in_old": len(leak),
        "leaked_cells": sorted(leak),
        "n_old_cells_rescored": len(old_keys_list),
        "results": results,
        "note": "clean = pool - checkpoint['cells'] - baseline_slice; "
                "contaminated = the design recomputed today (what the earlier scripts used)",
    }
    (OUT / "repaired_four_way.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n{'configuration':<24} {'clean':>10} {'old set':>10} {'delta':>10}")
    for name in results["clean"]:
        clean = results["clean"][name]
        old = results["contaminated"].get(name)
        delta = f"{old - clean:+.5f}" if old is not None else "-"
        print(f"{name:<24} {clean:10.5f} "
              f"{(f'{old:.5f}' if old is not None else '-'):>10} {delta:>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
