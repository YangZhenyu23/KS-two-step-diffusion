"""Sanity check for the frozen split and the stratified subset construction.

Verifies before any training that:
  * the test set and the training pool are disjoint and cover all 200 cells
  * each requested subset size is actually achievable and returns that many
    distinct cells
  * each subset is balanced in sigma (equal count per level) as designed

Usage:
    python scripts/check_split.py
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sample_efficiency import frozen_test_set, stratified_subset  # noqa: E402
from fullscale import grid as G  # noqa: E402


def main() -> int:
    test, pool = frozen_test_set()
    test_keys = {c.key for c in test}
    pool_keys = {c.key for c in pool}
    print(f"test cells {len(test)}   pool cells {len(pool)}   "
          f"total {len(test) + len(pool)}")
    print(f"overlap: {len(test_keys & pool_keys)}  (must be 0)")
    print(f"union covers all 200: {len(test_keys | pool_keys) == 200}")

    print("\nmu coverage in the test set:")
    mu_counts = Counter(c.j for c in test)
    print(f"  {[mu_counts.get(j, 0) for j in range(len(G.MU_GRID))]}")
    print("sigma coverage in the test set:")
    sigma_counts = Counter(c.i for c in test)
    print(f"  {[sigma_counts.get(i, 0) for i in range(len(G.SIGMA_GRID))]}")

    print("\nsubsets of the pool:")
    ok = True
    for n in (20, 40, 80, 160):
        cells = stratified_subset(pool, n, seed=20261107 + n)
        keys = {c.key for c in cells}
        sig = Counter(c.i for c in cells)
        mu = Counter(c.j for c in cells)
        balanced = len(set(sig.values())) == 1
        overlap = len(keys & test_keys)
        status = "OK" if (len(keys) == n and balanced and overlap == 0) else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  n={n:3d}: unique={len(keys):3d}  sigma/level={sorted(set(sig.values()))}  "
              f"mu counts={[mu.get(j, 0) for j in range(len(G.MU_GRID))]}  "
              f"test overlap={overlap}  {status}")

    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
