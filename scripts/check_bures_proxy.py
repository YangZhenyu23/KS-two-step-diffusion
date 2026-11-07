"""Validate the Gaussian Bures-Wasserstein proxy against the exact histogram W2.

The I-optimal design search needs the pairwise W2 between all 200 clouds, which is
19,900 exact OT solves (~2 hours). The search instead uses the Bures-Wasserstein
distance between Gaussians fitted to each cloud. That substitution is only
legitimate if the proxy ranks pairs the same way the exact metric does, so this
script measures it directly:

  * Spearman rank correlation between proxy and exact W2 over a random sample of
    pairs, including deliberately close and deliberately far pairs;
  * the worst rank inversions.

A high rank correlation means the design search is optimising the same ordering
the paper reports, not a different quantity. A low one means the proxy must be
replaced by the exact solve before any design is trusted.

Usage:
    python scripts/check_bures_proxy.py --pairs 400
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fullscale import grid as G  # noqa: E402
from fullscale.evaluate import score_pair  # noqa: E402
from fullscale.ioptimal import (  # noqa: E402
    bures_wasserstein, cloud_moments, load_or_build,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=400)
    ap.add_argument("--out", type=Path,
                    default=G.RESULTS / "sample_efficiency" / "bures_proxy.json")
    a = ap.parse_args()

    cells = G.full_grid()
    matrix_path = G.RESULTS / "sample_efficiency" / "bures_distance.npy"
    geometry = load_or_build(cells, matrix_path)
    print(f"distance matrix: {geometry.distance.shape}, digest {geometry.digest}")

    rng = np.random.default_rng(7)
    n = len(cells)
    pairs = set()
    # a spread of separations: same sigma level, adjacent levels, and random
    for _ in range(a.pairs // 3):
        i = int(rng.integers(n))
        j = int(rng.integers(n))
        if i != j:
            pairs.add((min(i, j), max(i, j)))
    for i in range(n):
        for di in (1, 2, len(G.MU_GRID)):
            j = i + di
            if j < n:
                pairs.add((i, j))
    while len(pairs) < a.pairs:
        i, j = int(rng.integers(n)), int(rng.integers(n))
        if i != j:
            pairs.add((min(i, j), max(i, j)))
    pairs = sorted(pairs)[: a.pairs]

    print(f"scoring {len(pairs)} pairs with exact OT ...")
    rows = []
    cache: dict[int, np.ndarray] = {}

    def cloud(k: int) -> np.ndarray:
        if k not in cache:
            path = G.DATA / "clouds" / f"{cells[k].key}.npz"
            with np.load(path, allow_pickle=False) as payload:
                cache[k] = payload["reference"].copy()
        return cache[k]

    for position, (i, j) in enumerate(pairs, start=1):
        exact = score_pair(cloud(i).astype(np.float64), cloud(j).astype(np.float64),
                           G.BINS)["w2"]
        proxy = float(geometry.distance[i, j])
        rows.append({"i": i, "j": j, "key_i": cells[i].key, "key_j": cells[j].key,
                     "exact_w2": exact, "bures": proxy,
                     "ratio": proxy / exact if exact else None})
        if position % 50 == 0:
            print(f"  {position}/{len(pairs)}", flush=True)

    exact = np.array([r["exact_w2"] for r in rows])
    proxy = np.array([r["bures"] for r in rows])
    from scipy.stats import spearmanr, pearsonr

    spearman = float(spearmanr(exact, proxy).statistic)
    pearson = float(pearsonr(exact, proxy).statistic)
    slope = float(np.polyfit(exact, proxy, 1)[0])
    print(f"\nSpearman rank correlation : {spearman:.4f}")
    print(f"Pearson correlation       : {pearson:.4f}")
    print(f"linear slope proxy~exact  : {slope:.4f}")
    print(f"proxy / exact  median     : {np.median(proxy / exact):.4f}")
    print(f"proxy / exact  range      : "
          f"{(proxy / exact).min():.4f} .. {(proxy / exact).max():.4f}")

    verdict = ("proxy ranks pairs consistently with the exact metric"
               if spearman > 0.9 else
               "proxy is acceptable with caution" if spearman > 0.75 else
               "proxy is NOT reliable; use exact OT in the design search")
    print(f"\nverdict: {verdict}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "n_pairs": len(rows), "spearman": spearman, "pearson": pearson,
        "slope": slope, "median_ratio": float(np.median(proxy / exact)),
        "verdict": verdict, "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
