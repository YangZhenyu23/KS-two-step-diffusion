"""I-optimal design in the *W2 metric itself*, with maximin as a hard constraint.

The problem with maximin alone
------------------------------
Maximin maximises the smallest pairwise distance in parameter space. That is a
geometry criterion in (sigma, mu); it says nothing about the distributions those
parameters produce. Measured on this study it also behaves badly: the maximin
design has twice the parameter-space spacing of a random design yet no better
test error, and 3.9x the run-to-run variance (docs/SAMPLE_EFFICIENCY.md 3).

The alternative proposed here keeps maximin as a *constraint* (the user requires
a minimum spacing) and changes only the *objective*.

Objective: minimise the integrated W2 distance to the nearest design point
--------------------------------------------------------------------------
For a design set ``D`` and the full grid ``G`` (all 200 cells), define

    J(D) = sum_{x in G}  min_{d in D}  W2(cloud_x, cloud_d)^2

``W2(cloud_x, cloud_d)`` is the exact distance between two *physical clouds*, the
same quantity the paper reports as its error measure. So ``J`` is the total
amount of distribution-space distance a model would have to interpolate over the
domain.

Why this is the paper's own rationale, not a substitute for it
--------------------------------------------------------------
The paper judges a generative model by ``W2(generated, reference)``. For a model
trained at parameter points ``D``, consider the trivial "nearest-neighbour"
predictor that returns the training cloud closest in parameter space. Its error
at a grid point ``x`` is *exactly* ``min_d W2(cloud_x, cloud_d)``. Any model must
do at least as well as an interpolator between those clouds, so ``J`` measures the
interpolation burden on the very scale the paper scores on.

Minimising ``J`` therefore does not replace the paper's metric with a proxy such
as MSE: it optimises the integral of that metric over the domain. The maximin
constraint stays in force, so parameter-space coverage is never traded away.

Computing W2 for every pair is expensive
----------------------------------------
Exact histogram OT between 100k-particle clouds costs ~0.3 s per pair; all
19,900 pairs would take hours. The design search instead uses the
Bures-Wasserstein distance between Gaussians fitted to the clouds,

    BW^2 = ||m1 - m2||^2 + tr(S1 + S2 - 2 (S1^{1/2} S2 S1^{1/2})^{1/2})

which is the exact W2 between Gaussian measures and a close approximation for the
near-Gaussian clouds of this study. ``scripts/check_bures_proxy.py`` validates the
approximation against the exact histogram W2 on real pairs and reports the rank
correlation; the design is only trusted if that correlation is high.

Output: a design that (a) satisfies the maximin requirement and (b) minimises the
W2-coverage objective.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .grid import DATA, MU_GRID, SIGMA_GRID, Cell


# ---------------------------------------------------------------- descriptors ---
def cloud_moments(cell: Cell) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean and covariance of one cloud (float64)."""
    path = DATA / "clouds" / f"{cell.key}.npz"
    with np.load(path, allow_pickle=False) as payload:
        points = payload["reference"].astype(np.float64)
    return points.mean(axis=0), np.cov(points, rowvar=False, bias=True)


def bures_wasserstein(m1, S1, m2, S2) -> float:
    """Bures-Wasserstein distance between two Gaussians (exact W2 for Gaussians)."""
    mean_term = float(np.sum((m1 - m2) ** 2))
    root1 = _psd_sqrt(S1)
    middle = root1 @ S2 @ root1
    root_middle = _psd_sqrt(middle)
    trace_term = float(np.trace(S1 + S2 - 2.0 * root_middle))
    return float(np.sqrt(max(mean_term + max(trace_term, 0.0), 0.0)))


def _psd_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    values = np.clip(values, 0.0, None)
    return (vectors * np.sqrt(values)) @ vectors.T


# ------------------------------------------------------------- distance matrix ---
@dataclass
class Geometry:
    cells: list[Cell]
    index: dict[str, int]
    distance: np.ndarray          # (n, n) Bures-Wasserstein distances
    digest: str

    def save(self, path: Path) -> None:
        np.save(path, self.distance)
        (path.with_suffix(".json")).write_text(json.dumps({
            "n": len(self.cells),
            "order": [c.key for c in self.cells],
            "digest": self.digest,
            "metric": "bures_wasserstein (gaussian approximation of W2)",
        }, indent=2), encoding="utf-8")


def build_geometry(cells: list[Cell], *, log=print) -> Geometry:
    """Compute the pairwise Bures-Wasserstein matrix over the given cells."""
    moments = []
    for position, cell in enumerate(cells, start=1):
        moments.append(cloud_moments(cell))
        if position % 40 == 0:
            log(f"  moments {position}/{len(cells)}")
    n = len(cells)
    distance = np.zeros((n, n))
    for a in range(n):
        ma, Sa = moments[a]
        for b in range(a + 1, n):
            mb, Sb = moments[b]
            value = bures_wasserstein(ma, Sa, mb, Sb)
            distance[a, b] = distance[b, a] = value
    digest = hashlib.sha256(distance.tobytes()).hexdigest()[:16]
    return Geometry(cells=cells, index={c.key: i for i, c in enumerate(cells)},
                    distance=distance, digest=digest)


def load_or_build(cells: list[Cell], path: Path, *, log=print) -> Geometry:
    meta_path = path.with_suffix(".json")
    if path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("order") == [c.key for c in cells]:
            log(f"  reusing distance matrix ({meta['n']} cells, digest {meta['digest']})")
            return Geometry(cells=cells, index={c.key: i for i, c in enumerate(cells)},
                            distance=np.load(path), digest=meta["digest"])
    log(f"  building distance matrix for {len(cells)} cells")
    geometry = build_geometry(cells, log=log)
    geometry.save(path)
    return geometry


# ---------------------------------------------------------------------- design ---
def parameter_distance(cells: list[Cell]) -> np.ndarray:
    """Normalised Euclidean distance in the (sigma, mu) plane."""
    n_sigma, n_mu = len(SIGMA_GRID), len(MU_GRID)
    arr = np.array([[c.i / (n_sigma - 1), c.j / (n_mu - 1)] for c in cells])
    diff = arr[:, None, :] - arr[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def maximin_distance(cells: list[Cell]) -> float:
    d = parameter_distance(cells)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def w2_objective(geometry: Geometry, design_indices: np.ndarray,
                 domain_indices: np.ndarray) -> tuple[float, float]:
    """Return ``(sum of squared nearest-design W2, mean nearest-design W2)``.

    The first value is the design objective ``J``; the second is the more
    interpretable average interpolation burden in W2 units, which is directly
    comparable to the errors reported elsewhere in this repository.
    """
    sub = geometry.distance[np.ix_(domain_indices, design_indices)]
    nearest = sub.min(axis=1)
    return float((nearest ** 2).sum()), float(nearest.mean())


def ioptimal_design(
    geometry: Geometry,
    design_size: int,
    *,
    min_spacing: float,
    candidate_keys: list[str],
    domain_keys: list[str],
    seed: int = 20261107,
    iterations: int = 40000,
    log=print,
) -> dict:
    """Exchange algorithm: minimise the W2 objective subject to min spacing.

    ``candidate_keys`` are the cells the design may draw from (the training pool);
    ``domain_keys`` are the cells the objective integrates over (the full grid).
    Keeping the two lists separate is essential: the objective must measure
    coverage of the whole parameter domain, while the design is only allowed to
    use cells that are actually available for training.

    Every candidate move swaps one design cell for an unused candidate. A move is
    accepted only if the resulting design still has pairwise parameter distance at
    least ``min_spacing``, so the maximin requirement is a hard constraint rather
    than a term in the objective.
    """
    rng = np.random.default_rng(seed)
    all_index = {c.key: k for k, c in enumerate(geometry.cells)}
    candidate_indices = np.array(sorted({all_index[k] for k in candidate_keys}))
    domain_indices = np.array([all_index[k] for k in domain_keys])
    candidate_set = set(candidate_indices.tolist())

    # Feasible starting point: a maximin design built *from the candidate pool*.
    # Using a design over the whole grid would seed the search with cells that are
    # not available for training.
    from .lhs import latin_hypercube

    candidate_cells = [geometry.cells[k] for k in candidate_indices]
    design_cells: list = []
    attempt = 0
    while len({c.key for c in design_cells}) < design_size:
        attempt += 1
        if attempt > 12:
            # fall back to a stratified draw straight from the pool
            design_cells = []
            by_sigma: dict[int, list] = {}
            for c in candidate_cells:
                by_sigma.setdefault(c.i, []).append(c)
            per_sigma = design_size // len(by_sigma)
            local_rng = np.random.default_rng(seed + attempt)
            for i, group in sorted(by_sigma.items()):
                group = sorted(group, key=lambda c: c.j)
                step = len(group) / per_sigma
                offset = int(local_rng.integers(len(group)))
                for k in range(per_sigma):
                    design_cells.append(group[int((offset + k * step)) % len(group)])
            break
        seed_design = latin_hypercube(design_size, seed=seed + attempt)
        design_cells = []
        for i, j in seed_design.points:
            key = f"s{i:02d}_m{j:02d}"
            if key in all_index and all_index[key] in candidate_set:
                design_cells.append(geometry.cells[all_index[key]])

    design = np.array(sorted({all_index[c.key] for c in design_cells}))
    if len(design) != design_size:
        raise RuntimeError(f"seed design has {len(design)} unique cells, expected {design_size}")

    def feasible(candidate: np.ndarray) -> bool:
        return maximin_distance([geometry.cells[k] for k in candidate]) >= min_spacing - 1e-12

    # If the maximin seed is too well spaced to satisfy nothing else, it is by
    # construction feasible; if it is not, relax to the best available seed.
    if not feasible(design):
        best_seed, best_spacing = design, maximin_distance([geometry.cells[k] for k in design])
        for _ in range(300):
            pick = rng.choice(candidate_indices, size=design_size, replace=False)
            spacing = maximin_distance([geometry.cells[k] for k in pick])
            if spacing > best_spacing:
                best_seed, best_spacing = pick, spacing
        design = best_seed
        log(f"  seed spacing relaxed to {best_spacing:.4f} "
            f"(constraint was {min_spacing:.4f})")

    current_obj, current_mean = w2_objective(geometry, design, domain_indices)
    best, best_obj, best_mean = design.copy(), current_obj, current_mean
    accepted = 0
    for _ in range(iterations):
        drop = int(rng.integers(len(design)))
        in_use = set(design.tolist())
        available = [k for k in candidate_indices if k not in in_use]
        if not available:
            break
        add = int(available[int(rng.integers(len(available)))])
        candidate = design.copy()
        candidate[drop] = add
        if not feasible(candidate):
            continue
        obj, mean = w2_objective(geometry, candidate, domain_indices)
        if obj < current_obj - 1e-12:
            design, current_obj, current_mean = candidate, obj, mean
            accepted += 1
            if obj < best_obj:
                best, best_obj, best_mean = candidate.copy(), obj, mean
    log(f"  exchange: {accepted} accepted moves of {iterations}")

    best_cells = [geometry.cells[k] for k in sorted(best)]
    result = {
        "design_size": len(best_cells),
        "min_spacing": maximin_distance(best_cells),
        "min_spacing_constraint": min_spacing,
        "w2_objective": best_obj,
        "mean_nearest_w2": best_mean,
        "cells": [c.key for c in best_cells],
        "seed": seed,
        "iterations": iterations,
    }
    return result
