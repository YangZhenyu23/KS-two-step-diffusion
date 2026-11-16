"""Verify whether the evaluation cells used by recent scripts match the training cells of
the checkpoint being evaluated.

The concern
-----------
Several scripts recompute the 40 training cells at run time with
``stratified_subset(pool, 40, seed=20261107)``. That function was itself changed during
development (a stratification bug fix), so calling it today need not reproduce the design
that the headline checkpoint was actually trained on.

If the recomputed design differs from the checkpoint's training cells, then
``held = pool - recomputed_design`` still contains cells that ARE in the checkpoint's
training set. Those cells are memorised, and scoring on them is data leakage.

This script compares, for each checkpoint on disk:
  * the cells recorded inside the checkpoint payload,
  * the design that ``stratified_subset`` produces today,
and reports the overlap and the leak size.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream" / "MF-DeepParticle" / "deepparticle"))

from fullscale import grid as G  # noqa: E402
from dp_two_param import stratified_subset  # noqa: E402

CHECKPOINTS = [
    ("headline MF (two_32000)", G.RESULTS / "optimal_budget_ablation/two_32000/mf_two_param.pt"),
    ("headline MF seed89", G.RESULTS / "optimal_budget_ablation/two_32000_s89/mf_two_param.pt"),
    ("train/lhs40", G.RESULTS / "train/lhs40/mf_two_param.pt"),
    ("capacity_matched w112", G.RESULTS / "capacity_matched/w112_s88/mf_two_param.pt"),
    ("extrapolation MF", G.RESULTS / "extrapolation/mf_s88/mf_two_param.pt"),
]


def main() -> int:
    pool = G.full_grid()
    design_now = stratified_subset(pool, 40, seed=20261107)
    keys_now = frozenset(c.key for c in design_now)
    slice_keys = frozenset(c.key for c in G.baseline_slice())

    print("design produced by stratified_subset(pool, 40, seed=20261107) TODAY:")
    print(f"  {len(keys_now)} cells")
    at_one = sorted(k for k in keys_now if int(k[5:7]) == G.MU_INDEX)
    print(f"  cells at mu=1: {', '.join(at_one)}")
    print(f"  held-out if mu=1 row also excluded: {200 - len(keys_now) - 20 + len(at_one)}")

    print("\n" + "=" * 88)
    for label, path in CHECKPOINTS:
        if not path.exists():
            print(f"{label}: MISSING {path.relative_to(ROOT)}")
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        trained = frozenset(payload.get("cells", []))
        at_one_ckpt = sorted(k for k in trained if int(k[5:7]) == G.MU_INDEX)
        held_if_now = (pool and
                       frozenset(c.key for c in pool) - keys_now - slice_keys)
        leak = trained & held_if_now
        print(f"\n{label}")
        print(f"  checkpoint records {len(trained)} training cells, "
              f"{len(at_one_ckpt)} at mu=1: {', '.join(at_one_ckpt)}")
        print(f"  identical to today's design? {trained == keys_now}")
        print(f"  overlap with today's design: {len(trained & keys_now)} / {len(trained)}")
        print(f"  held-out computed from today's design: {len(held_if_now)} cells")
        print(f"  *** of those, {len(leak)} are in this checkpoint's TRAINING set ***")
        if leak:
            print(f"      leaked: {', '.join(sorted(leak))}")
            print(f"      => any score computed on 'today's held-out set' for this")
            print(f"         checkpoint is contaminated on {len(leak)}/{len(held_if_now)} "
                  f"cells ({len(leak) / len(held_if_now):.0%})")

    print("\n" + "=" * 88)
    print("what the correct exclusion set is")
    print("  The held-out set for a checkpoint must be")
    print("      pool - checkpoint['cells'] - baseline_slice")
    print("  i.e. read the training cells FROM THE CHECKPOINT, never recompute them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
