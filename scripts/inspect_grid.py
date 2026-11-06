"""Quick physical sanity check: how do sigma and mu separately shape the cloud?

Prints the per-cell geometry of the reference clouds on a coarse sub-grid, so one
can see at a glance whether each parameter moves a *different* geometric feature
(parameter identifiability) or whether they act along the same axis (degeneracy).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from twoparam.data import load_cell  # noqa: E402
from twoparam.protocol import MU_GRID, RESULTS, SIGMA_GRID  # noqa: E402


def geometry(points: np.ndarray) -> dict:
    points = np.asarray(points, dtype=np.float64)
    x = points[:, 0]
    radial = np.sqrt(points[:, 1] ** 2 + points[:, 2] ** 2)
    return {
        "x_extent": float(x.max() - x.min()),
        "x_std": float(x.std()),
        "yz_p95": float(np.percentile(radial, 95)),
        "yz_std": float(radial.std()),
        "radial_mean": float(np.linalg.norm(points, axis=1).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=RESULTS / "data" / "full")
    ap.add_argument("--indices", type=int, nargs="+", default=[0, 2, 4, 6, 9])
    a = ap.parse_args()

    header = f"{'sigma':>9} {'mu':>9} | {'x_extent':>9} {'x_std':>7} {'yz_p95':>7} {'yz_std':>7}"
    print(header)
    print("-" * len(header))
    for i in a.indices:
        for j in a.indices:
            cell = load_cell(a.dir, f"s{i:02d}_m{j:02d}")
            g = geometry(cell["clouds"]["reference"])
            print(f"{SIGMA_GRID[i]:9.3f} {MU_GRID[j]:9.4f} | "
                  f"{g['x_extent']:9.3f} {g['x_std']:7.3f} {g['yz_p95']:7.3f} {g['yz_std']:7.3f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
