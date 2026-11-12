"""Compare our projection figure specification against the upstream one, item by item.

The paper text does not state the histogram binning or colour-bar convention for the
3D figure, so the upstream code ``ks.py::_projections`` is the only authority. This
script re-implements both conventions on the same cloud and prints the resulting
colour-bar ranges, which settles whether the paper's bars (reaching ~25-30) indicate
counts or probability density.

If our numbers match the upstream ``density=True`` convention, the figure is a faithful
imitation; if they instead match raw counts, our scale label is wrong and must be fixed.

Usage:
    python scripts/check_projection_convention.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream" / "MF-DeepParticle" / "deepparticle"))

from fullscale import grid as G  # noqa: E402
from fullscale.train import load_cell  # noqa: E402

PAIRS = [(0, 1), (0, 2), (1, 2)]


def upstream_edges(clouds: dict[str, np.ndarray]):
    """Exactly ``ks.py::_projections`` with bins=None: 0.05 steps, 0.05 pad."""
    lows = np.min([c.min(0) for c in clouds.values()], axis=0)
    highs = np.max([c.max(0) for c in clouds.values()], axis=0)
    return [np.arange(np.floor(lo / .05) * .05 - .05,
                      np.ceil(hi / .05) * .05 + .075, .05)
            for lo, hi in zip(lows, highs)]


def main() -> int:
    cell = G.Cell(11, 9)  # the hero cell, sigma=120 mu=10
    payload = load_cell(cell)
    reference = payload["clouds"]["reference"].astype(np.float64)
    clouds = {"ref": reference}
    edges = upstream_edges(clouds)
    print(f"cell {cell.key}  sigma={cell.sigma:g} mu={cell.mu:g}  N={len(reference)}")
    print("upstream edges (bins=None, step 0.05):")
    for axis, edge in enumerate(edges):
        print(f"  {'xyz'[axis]}: {edge[0]:.2f} .. {edge[-1]:.2f}  ({len(edge) - 1} bins)")

    print(f"\n{'projection':>10} {'max count':>11} {'max density':>13} {'ratio':>10}")
    for a, b in PAIRS:
        counts = np.histogram2d(reference[:, a], reference[:, b],
                                bins=(edges[a], edges[b]))[0]
        density = counts / len(reference) / np.diff(edges[a])[:, None] / np.diff(edges[b])[None, :]
        print(f"{'xyz'[a]}-{'xyz'[b]:>7} {counts.max():11.1f} {density.max():13.2f} "
              f"{counts.max() / density.max():10.1f}")

    print("\nThe published figure's colour bars reach roughly 25-30.")
    print("Raw counts at these edges top out in the tens; probability density is O(10) too,")
    print("but the ratio printed above is the factor between the two conventions, so the")
    print("two readings are NOT interchangeable - the label on the bar has to say which.")

    # What our figure actually plots
    print("\nour figure (bins=90, probability density):")
    lows, highs = reference.min(0), reference.max(0)
    pad = 0.03 * (highs - lows)
    our_edges = [np.linspace(lows[d] - pad[d], highs[d] + pad[d], 91) for d in range(3)]
    for a, b in PAIRS:
        counts = np.histogram2d(reference[:, a], reference[:, b],
                                bins=(our_edges[a], our_edges[b]))[0]
        density = counts / len(reference) / np.diff(our_edges[a])[:, None] / np.diff(our_edges[b])[None, :]
        print(f"  {'xyz'[a]}-{'xyz'[b]}: max density {density.max():.2f} "
              f"(max count {counts.max():.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
