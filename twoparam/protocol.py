"""Parameter grids and dimensionless groups for the (sigma, mu) study.

Physical model (identical to the retained upstream KS3D kernel)::

    dX_j = -(chi/N) sum_i grad K_delta(X_j - X_i) dt
           + sigma * g(X_j) dt
           + sqrt(2*mu) dW_j
    g(x) = (exp(-y^2 - z^2), 0, 0)

Fixed across the whole two-parameter study (upstream values):

    chi   = 16*pi         chemotactic sensitivity
    delta = 0.1           blob regularisation radius
    T     = 0.02          terminal time
    dt    = 0.005         Euler-Maruyama step (4 steps)
    R0    = 1.0           unit-ball initial support

Dimensionless groups (x = R0*xi, t = T*tau)::

    S = sigma*T/R0        shear forcing
    C = chi*T/R0^3        chemotactic forcing
    G = sqrt(2*mu*T)/R0   diffusive forcing
    delta_tilde = delta/R0

With chi, T, delta, R0 frozen, (sigma, mu) is a non-degenerate two-dimensional
parameterisation: sigma moves S alone, mu moves G alone, C and delta_tilde are
constants. That is exactly the property the (sigma, kappa) pair would *not* have,
because kappa belongs to the KPP branch of the code base and does not appear in
the Keller-Segel equations at all.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "MF-DeepParticle"
RESULTS = ROOT / "results" / "twoparam"

# ---------------------------------------------------------------- physics ---
CHI = 16.0 * math.pi
DELTA = 0.1
FINAL_TIME = 0.02
DT = 0.005
R0 = 1.0
DIMENSION = 3

# ------------------------------------------------------------ grid design ---
# sigma: the eight upstream training amplitudes, bracketed by the two endpoints
# of the upstream evaluation range (10 and 200). Eight of the ten values are
# therefore *exactly* the single-parameter training grid, which makes the mu=1
# slice of this study identical to the baseline protocol.
SIGMA_GRID = [
    10.0,
    15.0,
    20.84243392944336,
    28.960464477539062,
    40.240440368652344,
    55.913902282714844,
    77.69212341308594,
    107.95283508300781,
    150.0,
    200.0,
]

# mu: ten geometric values spanning one decade around the upstream value mu=1.
# The diffusive displacement G = sqrt(2*mu*T) then spans 0.063 ... 0.632, a
# decade, which is the same order as the shear range S = 0.2 ... 4.0.
# Written out explicitly so mu=1 is exactly 1.0 and the middle slice reproduces
# the upstream protocol bit for bit.
MU_GRID = [0.1, 0.158489319246, 0.251188643151, 0.398107170553, 0.63095734448,
           1.0, 1.584893192461, 2.511886431509, 3.981071705535, 10.0]

BASELINE_MU = 1.0
MU_INDEX = MU_GRID.index(BASELINE_MU)

N_TRAIN = 15_000
N_EVAL = 20_000
N_VISUAL = 6_000
BINS = 16


def dimensionless(sigma: float, mu: float) -> dict:
    """The four dimensionless numbers that fix the distribution family."""
    return {
        "S": sigma * FINAL_TIME / R0,
        "C": CHI * FINAL_TIME / R0**3,
        "G": math.sqrt(2.0 * mu * FINAL_TIME) / R0,
        "delta_tilde": DELTA / R0,
    }


def displacement_scales(sigma: float, mu: float) -> dict:
    """Displacement each physical effect can produce within one time window T."""
    return {
        "shear": sigma * FINAL_TIME,
        "diffusion": math.sqrt(2.0 * mu * FINAL_TIME),
        "chemotaxis": CHI * FINAL_TIME / (4 * math.pi * R0**2),
    }


@dataclass(frozen=True)
class Point:
    """One cell of the (sigma, mu) grid."""

    i: int
    j: int
    sigma: float
    mu: float
    in_lhs: bool = False

    @property
    def key(self) -> str:
        return f"s{self.i:02d}_m{self.j:02d}"

    @property
    def label(self) -> str:
        return f"sigma={self.sigma:.4g},mu={self.mu:.4g}"

    def meta(self) -> dict:
        return {
            "key": self.key,
            "i": self.i,
            "j": self.j,
            "sigma": self.sigma,
            "mu": self.mu,
            "in_lhs": self.in_lhs,
            **dimensionless(self.sigma, self.mu),
            **displacement_scales(self.sigma, self.mu),
        }


def full_grid() -> list[Point]:
    """All 10 x 10 = 100 cells, row-major in (sigma index, mu index)."""
    return [Point(i, j, s, m) for i, s in enumerate(SIGMA_GRID) for j, m in enumerate(MU_GRID)]


def grid_index() -> dict[tuple[int, int], Point]:
    return {(p.i, p.j): p for p in full_grid()}


def baseline_slice() -> list[Point]:
    """The mu = 1 slice: the ten sigma values of the single-parameter study."""
    return [Point(i, MU_INDEX, s, BASELINE_MU) for i, s in enumerate(SIGMA_GRID)]


@dataclass
class Config:
    """Everything that determines a numerical result, in one place."""

    n_train: int = N_TRAIN
    n_eval: int = N_EVAL
    n_visual: int = N_VISUAL
    bins: int = BINS
    device: str = "cuda"
    seed: int = 88
    chi: float = CHI
    delta: float = DELTA
    final_time: float = FINAL_TIME
    dt: float = DT
    sigma_grid: list[float] = field(default_factory=lambda: list(SIGMA_GRID))
    mu_grid: list[float] = field(default_factory=lambda: list(MU_GRID))

    def metadata(self) -> dict:
        return {
            "schema": "twoparam_v1",
            "n_train": self.n_train,
            "n_eval": self.n_eval,
            "n_visual": self.n_visual,
            "bins": self.bins,
            "chi": self.chi,
            "delta": self.delta,
            "final_time": self.final_time,
            "dt": self.dt,
            "sigma_grid": self.sigma_grid,
            "mu_grid": self.mu_grid,
        }


def ensure_results(*parts: str) -> Path:
    path = RESULTS.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
