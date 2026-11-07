"""Check the geometric quality of the subset constructors on the pool.

The LHS-versus-random comparison came out with *identical* minimum distance
(0.0526) for both designs, which cannot be right if the LHS constructor is doing
its job. This script measures the actual spacing statistics of every constructor
so the cause is visible rather than guessed.

A useful reference: the maximin Latin hypercube over the full 20x10 grid reaches
min distance 0.1053 in normalised units. Anything much below that is a
construction problem, not a property of the grid.

Usage:
    python scripts/check_design_quality.py
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fullscale import grid as G  # noqa: E402
from fullscale.lhs import latin_hypercube  # noqa: E402
from sample_efficiency import frozen_test_set, stratified_subset  # noqa: E402


def stats(cells, label: str) -> dict:
    n_sigma, n_mu = len(G.SIGMA_GRID), len(G.MU_GRID)
    arr = np.array([[i / (n_sigma - 1), j / (n_mu - 1)] for i, j in cells])
    diff = arr[:, None, :] - arr[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    corr = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]) if arr[:, 0].std() > 0 else 0.0
    sig = Counter(i for i, _ in cells)
    mu = Counter(j for _, j in cells)
    row = {
        "label": label,
        "n": len(cells),
        "min_dist": float(d.min()),
        "mean_nn": float(d.min(axis=1).mean()),
        "corr": corr,
        "sigma_balance": max(sig.values()) - min(sig.values()) if sig else 0,
        "mu_balance": max(mu.values()) - min(mu.values()) if mu else 0,
        "n_duplicates": len(cells) - len(set(cells)),
    }
    print(f"{label:<26} n={row['n']:3d}  min_dist={row['min_dist']:.4f}  "
          f"mean_nn={row['mean_nn']:.4f}  corr={corr:+.3f}  "
          f"sigma_range={row['sigma_balance']}  mu_range={row['mu_balance']}  "
          f"dups={row['n_duplicates']}")
    return row


def main() -> int:
    test, pool = frozen_test_set()
    pool_keys = {c.key for c in pool}
    print(f"pool: {len(pool)} cells\n")

    print("reference: maximin LHS on the FULL grid (ignores the frozen split)")
    full = latin_hypercube(40)
    stats([(i, j) for i, j in full.points], "latin_hypercube(full,40)")
    print()

    print("current stratified_subset on the pool")
    for n in (20, 40, 80):
        cells = stratified_subset(pool, n, seed=20261107 + n)
        stats([(c.i, c.j) for c in cells], f"stratified_subset(n={n})")

    print("\ndiagnosis: if min_dist equals the grid step 1/19 = 0.0526, then some")
    print("pair of cells sits in adjacent sigma levels with the SAME mu level.")
    print("With two picks per sigma level that is easy to happen by chance, and")
    print("per-level offsets cannot prevent it -- the constructor needs swap moves")
    print("over the whole point set, which is what fullscale.lhs does.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
