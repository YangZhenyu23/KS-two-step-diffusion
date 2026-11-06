"""Latin hypercube sampling on the discrete (sigma, mu) grid.

What LHS is
-----------
A Latin hypercube design of ``n`` points in ``d`` dimensions stratifies each
axis into ``n`` equal-probability intervals and places exactly one sample in
every interval of every axis. Equivalently: if the ``n`` points are sorted by
their ``k``-th coordinate, the ``k``-th coordinates form a permutation of the
stratum representatives. This makes the one-dimensional marginals of the design
*exactly* uniform, which plain random sampling only achieves in expectation.

The classic reason to prefer it over random sampling is variance reduction:
for a monotone or nearly additive response, the LHS estimator of the mean has
variance ``O(n^-1)`` rather than ``O(n^-1/2)`` (Stein 1987), and the design
avoids the clumping and the empty voids of ``n`` i.i.d. uniform draws. McKay,
Beckman & Conover (1979) introduced the construction.

The discrete analogue used here
-------------------------------
The grid has ``A`` levels per axis and we want ``n = k*A`` points with
``n % A == 0``. Writing the design as ``k`` independent permutations
``pi_1 ... pi_k`` of ``{0,...,A-1}`` and pairing point ``s`` with level
``pi_s[i]`` of the second axis gives, by construction,

    * exactly ``k`` points on every sigma level,
    * exactly ``k`` points on every mu level,
    * hence exactly uniform marginals in both directions,

which is the exact discrete statement of the Latin hypercube property. Among
all such balanced designs we then pick the one that maximises the minimum
pairwise distance (a *maximin* criterion, Johnson, Moore & Ylvisaker 1990) by
simulated annealing over swap moves. Maximin keeps the design space-filling:
no two training conditions are redundant, and the interpolating network sees
the whole parameter square rather than a diagonal stripe.

Why this matters for the present study
--------------------------------------
``10 x 10 = 100`` conditions at 100k interacting particles each would cost about
2.4 hours of simulation, and every additional physical parameter multiplies the
cost again. Training on a maximin LHS of ``k=2`` per level keeps every sigma and
every mu level represented, so the fitted surface is still anchored at the
boundaries of the box, while cutting the training set by a factor of five.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .protocol import MU_GRID, SIGMA_GRID


@dataclass
class Design:
    """A selected subset of grid indices together with its quality metrics."""

    points: list[tuple[int, int]]
    per_level: int
    min_distance: float
    mean_nn_distance: float
    correlation: float
    quadrants: list[int]
    objective: float
    seed: int

    def as_dict(self, sigma_grid=None, mu_grid=None) -> dict:
        sigma_grid = sigma_grid or SIGMA_GRID
        mu_grid = mu_grid or MU_GRID
        return {
            "points": [
                {"i": int(i), "j": int(j), "sigma": float(sigma_grid[i]), "mu": float(mu_grid[j])}
                for i, j in self.points
            ],
            "per_level": self.per_level,
            "min_distance": self.min_distance,
            "mean_nn_distance": self.mean_nn_distance,
            "correlation": self.correlation,
            "quadrants": self.quadrants,
            "objective": self.objective,
            "seed": self.seed,
            "n_points": len(self.points),
        }


def _coords(points, n_axis):
    """Normalise grid indices to [0, 1]^2 so distances are comparable across grids."""
    arr = np.asarray(points, dtype=np.float64)
    return arr / max(n_axis - 1, 1)


def _min_distance(points, n_axis):
    uv = _coords(points, n_axis)
    diff = uv[:, None, :] - uv[None, :, :]
    dist = np.sqrt((diff**2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    return float(dist.min())


def _mean_nn_distance(points, n_axis):
    uv = _coords(points, n_axis)
    diff = uv[:, None, :] - uv[None, :, :]
    dist = np.sqrt((diff**2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    return float(dist.min(axis=1).mean())


def _correlation(points):
    arr = np.asarray(points, dtype=np.float64)
    if arr[:, 0].std() == 0 or arr[:, 1].std() == 0:
        return 0.0
    return float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])


def _quadrants(points, n_axis):
    """Count points in each of the four quadrants of the parameter square."""
    counts = [0, 0, 0, 0]
    mid = (n_axis - 1) / 2.0
    for i, j in points:
        idx = (0 if i < mid else 1) * 2 + (0 if j < mid else 1)
        counts[idx] += 1
    return counts


def _from_permutations(perms, n_axis):
    """Materialise ``k`` permutations of an ``n_axis`` grid into grid points."""
    points = []
    for perm in perms:
        for i in range(n_axis):
            points.append((i, int(perm[i])))
    return points


def _objective(points, n_axis, corr_weight=0.05):
    """Maximin with a mild decorrelation bonus.

    The minimum distance dominates; the correlation term breaks ties between
    designs that are equally spread, preferring the one whose indices are less
    linearly aligned. Perfectly correlated designs (a pure diagonal) are the
    worst case for a regression-like learner because the two axes become
    confounded.
    """
    d = _min_distance(points, n_axis)
    r = abs(_correlation(points))
    return d + corr_weight * (1.0 - r)


def _anneal(n_axis, per_level, rng, iterations, temperature=0.08, cooling=None):
    """Simulated annealing over swap moves inside each permutation."""
    if cooling is None:
        cooling = (1e-3 / temperature) ** (1.0 / max(iterations, 1))
    perms = [rng.permutation(n_axis) for _ in range(per_level)]
    points = _from_permutations(perms, n_axis)
    current = _objective(points, n_axis)
    best, best_points = current, list(points)
    temp = temperature
    for _ in range(iterations):
        p_idx = int(rng.integers(per_level))
        a, b = rng.integers(0, n_axis, size=2)
        if a == b:
            temp *= cooling
            continue
        perms[p_idx][a], perms[p_idx][b] = perms[p_idx][b], perms[p_idx][a]
        candidate_points = _from_permutations(perms, n_axis)
        candidate = _objective(candidate_points, n_axis)
        delta = candidate - current
        if delta >= 0 or rng.random() < math.exp(delta / max(temp, 1e-12)):
            current, points = candidate, candidate_points
            if candidate > best:
                best, best_points = candidate, list(candidate_points)
        else:
            perms[p_idx][a], perms[p_idx][b] = perms[p_idx][b], perms[p_idx][a]
        temp *= cooling
    return best, best_points


def latin_hypercube(
    n_points: int = 20,
    *,
    sigma_grid=None,
    mu_grid=None,
    seed: int = 20261107,
    restarts: int = 24,
    iterations: int = 4000,
    force_include=None,
) -> Design:
    """Return a maximin Latin hypercube design on the discrete parameter grid.

    ``force_include`` optionally lists ``(i, j)`` levels that must appear, which
    is how the mu = 1 slice (the single-parameter baseline) is guaranteed to be
    represented.
    """
    sigma_grid = sigma_grid if sigma_grid is not None else SIGMA_GRID
    mu_grid = mu_grid if mu_grid is not None else MU_GRID
    n_sigma, n_mu = len(sigma_grid), len(mu_grid)
    if n_sigma != n_mu:
        raise ValueError("The permutation construction needs a square grid")
    n_axis = n_sigma
    if n_points % n_axis:
        raise ValueError(f"n_points={n_points} must be a multiple of the {n_axis} grid levels")
    per_level = n_points // n_axis

    rng = np.random.default_rng(seed)
    best_obj, best_points = -np.inf, None
    for _ in range(restarts):
        obj, pts = _anneal(n_axis, per_level, rng, iterations)
        if force_include:
            missing = [p for p in force_include if p not in set(pts)]
            penalty = 0.5 * len(missing)
            obj -= penalty
        if obj > best_obj:
            best_obj, best_points = obj, pts

    best_points = sorted(best_points, key=lambda p: (p[0], p[1]))
    for p in force_include or []:
        if p not in best_points:
            best_points.append(tuple(p))
    best_points = sorted(set(map(tuple, best_points)), key=lambda p: (p[0], p[1]))

    return Design(
        points=[(int(i), int(j)) for i, j in best_points],
        per_level=per_level,
        min_distance=_min_distance(best_points, n_axis),
        mean_nn_distance=_mean_nn_distance(best_points, n_axis),
        correlation=_correlation(best_points),
        quadrants=_quadrants(best_points, n_axis),
        objective=float(best_obj),
        seed=seed,
    )


def random_subset(n_points: int = 20, *, seed: int = 7) -> Design:
    """Uniform random subset of the same size, used as a control for LHS."""
    rng = np.random.default_rng(seed)
    n_axis = len(SIGMA_GRID)
    all_cells = [(i, j) for i in range(n_axis) for j in range(n_axis)]
    chosen = rng.choice(len(all_cells), size=n_points, replace=False)
    points = sorted((all_cells[int(c)] for c in chosen), key=lambda p: (p[0], p[1]))
    return Design(
        points=points,
        per_level=0,
        min_distance=_min_distance(points, n_axis),
        mean_nn_distance=_mean_nn_distance(points, n_axis),
        correlation=_correlation(points),
        quadrants=_quadrants(points, n_axis),
        objective=float("nan"),
        seed=seed,
    )


if __name__ == "__main__":  # pragma: no cover - exploratory entry point
    design = latin_hypercube(20)
    print(f"LHS-20  min_dist={design.min_distance:.4f}  "
          f"mean_nn={design.mean_nn_distance:.4f}  corr={design.correlation:+.3f}")
    for i, j in design.points:
        print(f"  sigma={SIGMA_GRID[i]:>8.3f}  mu={MU_GRID[j]:<10.6g}  (i={i}, j={j})")
