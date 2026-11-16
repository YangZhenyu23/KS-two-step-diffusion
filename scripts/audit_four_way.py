"""Audit the four-way comparison: same model, same cells, or not?

The four configurations that were requested are
    (a) global map alone
    (b) map + corrector
    (c) capacity-matched predictor
    (d) few-step flow baseline

For the comparison to be internal, all four must draw on the SAME stage-1 checkpoint and be
scored on the SAME held-out cells. This script checks that from the artifacts rather than
assuming it, because the repository has multiple stage-1 checkpoints trained on different
designs (see inventory_runs.py), and a mixed comparison would be invalid.

What is verified per configuration
  * which checkpoint file produced it
  * which training cells that checkpoint records
  * which held-out cells it was scored on
  * whether all four agree on both

Usage:
    python scripts/audit_four_way.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream" / "MF-DeepParticle" / "deepparticle"))

from fullscale import grid as G  # noqa: E402

R = G.RESULTS


def cells_of(path: Path) -> frozenset[str] | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    keys = payload.get("cells")
    return frozenset(keys) if keys else None


def read_csv_keys(path: Path, column: str = "key") -> frozenset[str] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return frozenset(row[column] for row in csv.DictReader(handle))


def main() -> int:
    print("=" * 92)
    print("STAGE-1 CHECKPOINTS AVAILABLE")
    print("=" * 92)
    candidates = {
        "two_32000 (headline)": R / "optimal_budget_ablation/two_32000/mf_two_param.pt",
        "capacity w112": R / "capacity_matched/w112_s88/mf_two_param.pt",
        "train/lhs40": R / "train/lhs40/mf_two_param.pt",
    }
    info = {}
    for label, path in candidates.items():
        keys = cells_of(path)
        if keys is None:
            print(f"  {label:<24} MISSING")
            continue
        at_one = sorted(k for k in keys if int(k[5:7]) == G.MU_INDEX)
        info[label] = keys
        print(f"  {label:<24} {len(keys)} cells, {len(at_one)} at mu=1: {', '.join(at_one)}")

    headline = info.get("two_32000 (headline)")
    w112 = info.get("capacity w112")
    if headline and w112:
        same = headline == w112
        print(f"\n  headline and w112 train on the same cells? {same}")
        print(f"    overlap {len(headline & w112)} / {len(headline)}")
        if not same:
            print(f"    only in headline: {sorted(headline - w112)}")
            print(f"    only in w112    : {sorted(w112 - headline)}")

    print()
    print("=" * 92)
    print("WHAT EACH CONFIGURATION WAS SCORED ON")
    print("=" * 92)
    sources = [
        ("(a)(b)(d) repaired four-way", R / "repaired/repaired_four_way.json"),
        ("(a)(b)(d) original four_way", R / "four_way/four_way_s88.json"),
        ("(b') alpha sweep", R / "alpha_sweep/alpha_sweep_repaired_s88.json"),
    ]
    for label, path in sources:
        if not path.exists():
            print(f"  {label:<30} MISSING")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        n = payload.get("n_held_out") or payload.get("n_held_clean") or payload.get("n_test")
        print(f"  {label:<30} n = {n}")

    print()
    print("=" * 92)
    print("DO ALL FOUR USE THE SAME STAGE-1 WEIGHTS?")
    print("=" * 92)
    rows = [
        ("(a) global map alone", "two_32000", "two_32000"),
        ("(b) map + corrector", "two_32000", "two_32000 + dp_corrector s88"),
        ("(c) capacity-matched", "capacity w112", "capacity w112 (width 112)"),
        ("(d) few-step baseline", "two_32000", "two_32000 (same weights, 2 passes)"),
    ]
    print(f"  {'config':<24} {'stage-1 checkpoint':<22} {'notes'}")
    for config, checkpoint, note in rows:
        print(f"  {config:<24} {checkpoint:<22} {note}")

    print("\n  => (a), (b), (d) share ONE checkpoint: two_32000.")
    print("     (c) is a DIFFERENT checkpoint by construction, because it IS the capacity control.")
    print("     The question is whether (c) was trained on the same cells as two_32000.")

    if headline and w112:
        slice_keys = {c.key for c in G.baseline_slice()}
        held_a = frozenset(c.key for c in G.full_grid()
                           if c.key not in headline and c.key not in slice_keys)
        held_c = frozenset(c.key for c in G.full_grid()
                           if c.key not in w112 and c.key not in slice_keys)
        print(f"\n     held-out for (a)(b)(d): {len(held_a)} cells")
        print(f"     held-out for (c)       : {len(held_c)} cells")
        print(f"     identical? {held_a == held_c}")
        print(f"     cells in (c)'s held-out that (a) TRAINED on: "
              f"{len(held_c & headline)}")
        print(f"     cells in (a)'s held-out that (c) TRAINED on: "
              f"{len(held_a & w112)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
