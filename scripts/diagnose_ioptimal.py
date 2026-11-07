"""Diagnose why the I-optimal design produced the worst test error.

The first I-optimal run gave the lowest W2-coverage objective (J = 0.517 versus
0.808 for maximin) yet the worst test error (0.198 versus 0.167). Two very
different explanations have to be separated:

H1 **Implementation failure.** The log recorded ``0 accepted moves`` and the seed
   spacing relaxed from 0.1053 to 0.0526. If the exchange search never improved on
   its starting point, the "I-optimal" design is really just the fallback
   stratified draw, and nothing was optimised at all.

H2 **The criterion is wrong for this task.** Even a correctly optimised J may not
   predict model error, because J only measures how far the nearest training cloud
   is; it says nothing about whether a smooth model can interpolate between those
   clouds. A design that places cells where the clouds change fast will have a low
   J yet be hard to fit.

This script looks at what the design actually did: where the selected cells sit,
how the coverage objective is distributed over the domain, and whether the cells
cluster in the region where clouds change fastest.

Usage:
    python scripts/diagnose_ioptimal.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fullscale import grid as G  # noqa: E402
from fullscale.ioptimal import load_or_build, maximin_distance  # noqa: E402

OUT = G.RESULTS / "sample_efficiency"


def main() -> int:
    cells = G.full_grid()
    geometry = load_or_build(cells, OUT / "bures_distance.npy", log=lambda t: None)
    all_index = {c.key: k for k, c in enumerate(cells)}

    payload = json.loads((OUT / "ioptimal_comparison.json").read_text(encoding="utf-8"))
    designs = {row["label"]: row for row in payload["designs"]}

    print("=== where do the designs place their cells? ===\n")
    for label, row in designs.items():
        keys = row["cells"]
        sigma_counts = Counter(int(k[1:3]) for k in keys)
        mu_counts = Counter(int(k[5:7]) for k in keys)
        print(f"{label}")
        print(f"  min spacing      {row['min_spacing']:.4f}")
        print(f"  sigma per level  {sorted(sigma_counts.values())}")
        print(f"  mu counts        {[mu_counts.get(j, 0) for j in range(len(G.MU_GRID))]}")
        print(f"  test W2          {row['test_mean_w2']:.5f}")

    print("\n=== is the coverage objective dominated by the high-mu region? ===\n")
    for label, row in designs.items():
        design_idx = np.array([all_index[k] for k in row["cells"]])
        domain_idx = np.arange(len(cells))
        sub = geometry.distance[np.ix_(domain_idx, design_idx)]
        nearest = sub.min(axis=1)
        by_mu = {}
        for k, cell in enumerate(cells):
            by_mu.setdefault(cell.j, []).append(nearest[k])
        means = [float(np.mean(by_mu[j])) for j in range(len(G.MU_GRID))]
        print(f"{label:<10} mean nearest W2 by mu level:")
        print("           " + "  ".join(f"{m:.3f}" for m in means))
        contribution = [float(np.sum(np.square(by_mu[j]))) for j in range(len(G.MU_GRID))]
        total = sum(contribution)
        shares = [c / total for c in contribution]
        print("           J share   " + "  ".join(f"{s:.2f}" for s in shares))
    print(f"\n  mu levels: " + "  ".join(f"{m:.2f}" for m in G.MU_GRID))

    print("\n=== how much does J vary across a random sample of designs? ===\n")
    rng = np.random.default_rng(3)
    pool_path = OUT / "split.json"
    split = json.loads(pool_path.read_text(encoding="utf-8"))
    pool_keys = [r["key"] for r in split["pool"]]
    pool_idx = np.array([all_index[k] for k in pool_keys])
    domain_idx = np.arange(len(cells))
    js, spacings = [], []
    for _ in range(200):
        pick = rng.choice(pool_idx, size=40, replace=False)
        sub = geometry.distance[np.ix_(domain_idx, pick)]
        nearest = sub.min(axis=1)
        js.append(float((nearest ** 2).sum()))
        spacings.append(maximin_distance([cells[k] for k in pick]))
    js = np.array(js)
    spacings = np.array(spacings)
    print(f"  random 40-cell designs (n=200):")
    print(f"    J          mean {js.mean():.4f}  sd {js.std():.4f}  "
          f"min {js.min():.4f}  max {js.max():.4f}")
    print(f"    min space  mean {spacings.mean():.4f}  max {spacings.max():.4f}")
    print(f"  correlation(J, min_spacing) = {np.corrcoef(js, spacings)[0, 1]:+.3f}")
    print("  A strong negative correlation means J and the spacing requirement")
    print("  pull in opposite directions: minimising coverage cost necessarily")
    print("  reduces the minimum spacing, so the two criteria cannot both be")
    print("  improved at once.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
