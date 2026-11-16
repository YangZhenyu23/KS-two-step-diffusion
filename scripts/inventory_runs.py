"""Inventory every training run in the repository: what cells, what mu, what budget.

Why this exists
---------------
The repository grew in stages (single-parameter reproduction -> two-parameter pilot ->
full 200-cell grid -> ablations), and different stages used **different LHS-40 designs**.
That means the phrase "held-out set" refers to several different cell sets depending on
which run is being discussed, and a reader - or a reviewer - can easily compare two
numbers that were never measured on the same cells.

This script builds the ground truth: for every run with a training log, list the exact
cells, the distinct mu values, the budget, and the implied held-out count. It then flags
any pair of runs whose training-cell sets differ, because those two runs' held-out
numbers are NOT directly comparable.

Usage:
    python scripts/inventory_runs.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fullscale import grid as G  # noqa: E402

RESULTS = ROOT / "results"


def parse_key(key: str) -> tuple[int, int]:
    return int(key[1:3]), int(key[5:7])


def describe_log(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    cells = payload.get("cells") or []
    if not cells:
        return None
    keys = [c if isinstance(c, str) else c.get("key") for c in cells]
    indices = [parse_key(k) for k in keys]
    mus = sorted({round(G.MU_GRID[j], 6) for _, j in indices})
    sigmas = sorted({G.SIGMA_GRID[i] for i, _ in indices})
    mu_one = [k for k in keys if parse_key(k)[1] == G.MU_INDEX]
    return {
        "path": path,
        "kind": payload.get("kind", "?"),
        "n_cells": len(keys),
        "steps": int(payload.get("steps", 0)),
        "per_cell": int(payload.get("steps", 0)) / max(len(keys), 1),
        "n_mu": len(mus),
        "mus": mus,
        "n_sigma": len(sigmas),
        "mu_one_cells": mu_one,
        "keys": frozenset(keys),
        "seconds": payload.get("seconds"),
    }


def main() -> int:
    logs = sorted(RESULTS.rglob("training_log.json"))
    runs = [r for r in (describe_log(p) for p in logs) if r]
    print(f"{len(runs)} training runs with a stored cell list\n")

    print(f"{'run':<56} {'kind':>10} {'cells':>6} {'steps':>9} {'/cell':>7} {'mus':>4} {'mu=1':>5}")
    print("-" * 106)
    for run in runs:
        relative = str(run["path"].relative_to(ROOT)).replace("\\", "/")
        relative = relative.replace("results/fullscale/", "").replace("/training_log.json", "")
        print(f"{relative:<56} {run['kind']:>10} {run['n_cells']:6d} {run['steps']:9,d} "
              f"{run['per_cell']:7.0f} {run['n_mu']:4d} {len(run['mu_one_cells']):5d}")

    # group runs by identical training-cell sets
    print("\n=== distinct training-cell sets ===")
    groups: dict[frozenset, list[dict]] = {}
    for run in runs:
        groups.setdefault(run["keys"], []).append(run)
    ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    set_labels: dict[int, str] = {}
    for number, (keys, members) in enumerate(ordered, start=1):
        mus = sorted({round(G.MU_GRID[j], 4) for _, j in (parse_key(k) for k in keys)})
        mu_one = sorted(k for k in keys if parse_key(k)[1] == G.MU_INDEX)
        held = 200 - len(keys) - 20 + len(mu_one)
        label = f"S{number}"
        set_labels[id(keys)] = label
        print(f"\n{label}: {len(keys)} cells, {len(mus)} mu values, "
              f"{len(mu_one)} at mu=1 -> {held} held out if the mu=1 row is also training")
        print(f"     mu values: {[round(m, 3) for m in mus]}")
        if mu_one:
            print(f"     cells at mu=1: {', '.join(mu_one)}")
        print(f"     used by ({len(members)}):")
        for member in members:
            relative = str(member["path"].relative_to(ROOT)).replace("\\", "/")
            print(f"       - {relative}")

    if len(ordered) > 1:
        print("\n=== WARNING: different training sets -> held-out numbers not comparable ===")
        base_keys = ordered[0][0]
        for keys, members in ordered[1:]:
            overlap = len(base_keys & keys)
            only_base = len(base_keys - keys)
            only_other = len(keys - base_keys)
            print(f"  S1 vs {set_labels[id(keys)]}: overlap {overlap}, "
                  f"only-in-S1 {only_base}, only-in-other {only_other}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
