"""Maximin Latin hypercube design on the 20 x 10 full-scale grid.

The pilot grid was square (10 x 10), which allowed the textbook construction
"k permutations of the axis indices". The full-scale grid is 20 sigma levels by
10 mu levels, so that construction no longer applies. This module builds a
stratified design directly and states exactly which properties it guarantees.

Construction
------------
For ``n = 2 * 20 = 40`` points:

1. **Sigma strata** — two independent random permutations ``p1``, ``p2`` of
   ``0..19``. Point group ``g`` uses sigma levels ``pg[k]``, so every sigma level
   appears exactly twice.
2. **Mu pairing** — for each group, mu indices are drawn as a *cyclic offset*
   of a random permutation ``q``:

   ```
   mu(group g, rank k) = q[k] + offset_g   (mod 10)
   ```

   Each offset is a random permutation slice, so within a group every mu level is
   used exactly twice (20 points / 10 levels), and across the 40 points every mu
   level appears exactly four times.
3. **Annealing** — mu offsets and the permutations are perturbed by swap /
   offset moves; the objective is
   ``min_pairwise_distance + 0.05 * (1 - |corr|)``
   so the design is simultaneously space-filling (maximin) and statistically
   separable (decorrelated). The best of many restarts is kept.

Guaranteed properties
---------------------
* every sigma level: exactly ``n / 20`` points
* every mu level: exactly ``n / 10`` points
* ``|corr(sigma_index, mu_index)|`` minimised, reported in the output
* minimum pairwise distance in the unit square maximised

``n = 40`` keeps 160 of the 200 cells free for held-out validation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .grid import MU_GRID, MU_INDEX, SIGMA_GRID

N_SIGMA = len(SIGMA_GRID)
N_MU = len(MU_GRID)


@dataclass
class Design:
    points: list[tuple[int, int]]
    min_distance: float
    mean_nn_distance: float
    correlation: float
    sigma_counts: list[int]
    mu_counts: list[int]
    objective: float
    seed: int

    def as_dict(self) -> dict:
        return {
            "n_points": len(self.points),
            "n_sigma_levels": N_SIGMA,
            "n_mu_levels": N_MU,
            "min_distance": self.min_distance,
            "mean_nn_distance": self.mean_nn_distance,
            "correlation": self.correlation,
            "sigma_counts": self.sigma_counts,
            "mu_counts": self.mu_counts,
            "objective": self.objective,
            "seed": self.seed,
            "points": [
                {"i": int(i), "j": int(j), "sigma": float(SIGMA_GRID[i]), "mu": float(MU_GRID[j])}
                for i, j in self.points
            ],
        }


def _uv(points):
    arr = np.asarray(points, dtype=np.float64)
    out = np.empty_like(arr)
    out[:, 0] = arr[:, 0] / max(N_SIGMA - 1, 1)
    out[:, 1] = arr[:, 1] / max(N_MU - 1, 1)
    return out


def _distances(points):
    uv = _uv(points)
    diff = uv[:, None, :] - uv[None, :, :]
    dist = np.sqrt((diff**2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    return dist


def _min_distance(points) -> float:
    return float(_distances(points).min())


def _mean_nn(points) -> float:
    return float(_distances(points).min(axis=1).mean())


def _correlation(points) -> float:
    arr = np.asarray(points, dtype=np.float64)
    a, b = arr[:, 0], arr[:, 1]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _objective(points, corr_weight=0.05) -> float:
    return _min_distance(points) + corr_weight * (1.0 - abs(_correlation(points)))


def _materialise(sigma_perms, mu_perm, offsets, per_mu):
    """Build the point list from the design parameters."""
    points = []
    for g, sigma_perm in enumerate(sigma_perms):
        offset = offsets[g]
        for k in range(N_SIGMA):
            mu_index = (int(mu_perm[k % N_MU]) + int(offset)) % N_MU
            points.append((int(sigma_perm[k]), mu_index))
    return points


def latin_hypercube(
    n_points: int = 40,
    *,
    seed: int = 20261107,
    restarts: int = 40,
    iterations: int = 6000,
) -> Design:
    if n_points % N_SIGMA:
        raise ValueError(f"n_points={n_points} must be a multiple of {N_SIGMA} sigma levels")
    per_sigma = n_points // N_SIGMA          # 2 groups
    if n_points % N_MU:
        raise ValueError(f"n_points={n_points} must be a multiple of {N_MU} mu levels")

    rng = np.random.default_rng(seed)
    best_obj, best_points = -np.inf, None

    for _ in range(restarts):
        sigma_perms = [rng.permutation(N_SIGMA) for _ in range(per_sigma)]
        mu_perm = rng.permutation(N_MU)
        offsets = rng.integers(0, N_MU, size=per_sigma)

        points = _materialise(sigma_perms, mu_perm, offsets, per_sigma)
        current = _objective(points)
        temp = 0.06
        cooling = (1e-3 / temp) ** (1.0 / max(iterations, 1))

        for _ in range(iterations):
            move = int(rng.integers(3))
            undo = None
            if move == 0:
                # swap two entries of one sigma permutation
                g = int(rng.integers(per_sigma))
                a, b = rng.integers(0, N_SIGMA, size=2)
                if a == b:
                    temp *= cooling
                    continue
                sigma_perms[g][a], sigma_perms[g][b] = sigma_perms[g][b], sigma_perms[g][a]
                undo = ("sigma", g, a, b)
            elif move == 1:
                # swap two entries of the mu permutation
                a, b = rng.integers(0, N_MU, size=2)
                if a == b:
                    temp *= cooling
                    continue
                mu_perm[a], mu_perm[b] = mu_perm[b], mu_perm[a]
                undo = ("mu", None, a, b)
            else:
                # shift one group's mu offset
                g = int(rng.integers(per_sigma))
                delta = int(rng.integers(1, N_MU))
                previous = int(offsets[g])
                offsets[g] = (previous + delta) % N_MU
                undo = ("offset", g, previous, None)

            candidate = _materialise(sigma_perms, mu_perm, offsets, per_sigma)
            obj = _objective(candidate)
            if obj >= current or rng.random() < math.exp((obj - current) / max(temp, 1e-12)):
                current = obj
                if obj > best_obj:
                    best_obj, best_points = obj, list(candidate)
            else:
                kind, g, x, y = undo
                if kind == "sigma":
                    sigma_perms[g][x], sigma_perms[g][y] = sigma_perms[g][y], sigma_perms[g][x]
                elif kind == "mu":
                    mu_perm[x], mu_perm[y] = mu_perm[y], mu_perm[x]
                else:
                    offsets[g] = x
            temp *= cooling

    best_points = sorted(set(best_points), key=lambda p: (p[0], p[1]))
    sigma_counts = [0] * N_SIGMA
    mu_counts = [0] * N_MU
    for i, j in best_points:
        sigma_counts[i] += 1
        mu_counts[j] += 1
    return Design(
        points=best_points,
        min_distance=_min_distance(best_points),
        mean_nn_distance=_mean_nn(best_points),
        correlation=_correlation(best_points),
        sigma_counts=sigma_counts,
        mu_counts=mu_counts,
        objective=float(best_obj),
        seed=seed,
    )


def random_subset(n_points: int = 40, *, seed: int = 7) -> Design:
    rng = np.random.default_rng(seed)
    all_cells = [(i, j) for i in range(N_SIGMA) for j in range(N_MU)]
    chosen = rng.choice(len(all_cells), size=n_points, replace=False)
    points = sorted((all_cells[int(c)] for c in chosen), key=lambda p: (p[0], p[1]))
    sigma_counts = [0] * N_SIGMA
    mu_counts = [0] * N_MU
    for i, j in points:
        sigma_counts[i] += 1
        mu_counts[j] += 1
    return Design(
        points=points,
        min_distance=_min_distance(points),
        mean_nn_distance=_mean_nn(points),
        correlation=_correlation(points),
        sigma_counts=sigma_counts,
        mu_counts=mu_counts,
        objective=float("nan"),
        seed=seed,
    )


if __name__ == "__main__":
    design = latin_hypercube(40)
    rnd = random_subset(40)
    print(f"LHS-40  min_dist={design.min_distance:.4f}  mean_nn={design.mean_nn_distance:.4f}  "
          f"corr={design.correlation:+.3f}")
    print(f"Random  min_dist={rnd.min_distance:.4f}  mean_nn={rnd.mean_nn_distance:.4f}  "
          f"corr={rnd.correlation:+.3f}")
    print(f"sigma counts (LHS)  {design.sigma_counts}")
    print(f"mu counts    (LHS)  {design.mu_counts}")
    print(f"sigma counts (rand) {rnd.sigma_counts}")
    print(f"mu counts    (rand) {rnd.mu_counts}")
    print(f"\nbaseline mu=1 (index {MU_INDEX}) has {design.mu_counts[MU_INDEX]} LHS points")
