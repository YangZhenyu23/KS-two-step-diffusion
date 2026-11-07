"""Grid definition for the paper-scale two-parameter study.

Differences from the pilot (``twoparam.protocol``):

    sigma   10 -> 20 values, the upstream *evaluation* grid (10, 20, ..., 200).
            The pilot used the eight-point training grid plus two endpoints; the
            paper-scale run uses the twenty-point evaluation grid so that every
            published sigma is represented and a figure can be compared point by
            point against the upstream Figure 2.
    mu      10 values, unchanged (geometric, one decade, mu = 1 in the middle).
    N       100_000 particles, matching the upstream evaluation protocol, so the
            W2 statistical floor is 0.0644 instead of 0.0931.
    bins    16, unchanged, for direct comparability with every earlier number
            in this repository.

Grid size is therefore 20 x 10 = 200 cells. At 81 s per 100k cell this is about
4.5 hours of simulation for the complete grid, which is what makes the
leave-out validation possible: train on an LHS subset, then score every one of
the 200 cells.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "fullscale"
DATA = RESULTS / "data" / "full"

# ---- physics: identical to twoparam.protocol and to upstream -----------------
CHI = 16.0 * math.pi
DELTA = 0.1
FINAL_TIME = 0.02
DT = 0.005
R0 = 1.0
DIMENSION = 3

# ---- grid --------------------------------------------------------------------
# The upstream evaluation grid, verbatim (protocol.eval_levels for ks3d).
SIGMA_GRID = [float(v) for v in range(10, 201, 10)]

# Geometric over one decade, mu = 1 exactly in the middle (the upstream value).
MU_GRID = [0.1, 0.158489319246, 0.251188643151, 0.398107170553, 0.63095734448,
           1.0, 1.584893192461, 2.511886431509, 3.981071705535, 10.0]

BASELINE_MU = 1.0
MU_INDEX = MU_GRID.index(BASELINE_MU)

N_PARTICLES = 100_000
N_EVAL = 100_000
BINS = 16

# Measured on this machine at 100k particles (scripts/calibrate_fullscale.py).
FLOOR_16 = 0.06443
FLOOR_24 = 0.05709
SECONDS_PER_CELL = 80.9


@dataclass(frozen=True)
class Cell:
    """One point of the (sigma, mu) grid."""

    i: int
    j: int

    @property
    def sigma(self) -> float:
        return SIGMA_GRID[self.i]

    @property
    def mu(self) -> float:
        return MU_GRID[self.j]

    @property
    def key(self) -> str:
        return f"s{self.i:02d}_m{self.j:02d}"

    def meta(self) -> dict:
        return {
            "key": self.key, "i": self.i, "j": self.j,
            "sigma": self.sigma, "mu": self.mu,
            "S": self.sigma * FINAL_TIME / R0,
            "C": CHI * FINAL_TIME / R0**3,
            "G": math.sqrt(2.0 * self.mu * FINAL_TIME) / R0,
            "delta_tilde": DELTA / R0,
        }


def full_grid() -> list[Cell]:
    return [Cell(i, j) for i in range(len(SIGMA_GRID)) for j in range(len(MU_GRID))]


def index_map() -> dict[tuple[int, int], Cell]:
    return {(c.i, c.j): c for c in full_grid()}


def baseline_slice() -> list[Cell]:
    """The mu = 1 row: it reproduces the single-parameter protocol exactly."""
    return [Cell(i, MU_INDEX) for i in range(len(SIGMA_GRID))]


@dataclass
class Config:
    n_train: int = N_PARTICLES
    bins: int = BINS
    device: str = "cuda"
    seed: int = 88
    chi: float = CHI
    delta: float = DELTA
    final_time: float = FINAL_TIME
    dt: float = DT

    def metadata(self) -> dict:
        return {
            "schema": "fullscale_v1",
            "n_train": self.n_train,
            "bins": self.bins,
            "chi": self.chi,
            "delta": self.delta,
            "final_time": self.final_time,
            "dt": self.dt,
            "sigma_grid": SIGMA_GRID,
            "mu_grid": MU_GRID,
        }


def ensure(*parts: str) -> Path:
    path = RESULTS.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
