"""Two-parameter (sigma, mu) extension of the KS3D two-step diffusion study.

This package is additive: it never imports or mutates ``upstream/MF-DeepParticle``
at runtime beyond reading its numerical kernels, and it writes every artefact
under its own output directory so the single-parameter pipeline stays intact.

Modules
-------
protocol   parameter grids, dimensionless groups, cloud metadata
lhs        Latin hypercube sampling over the (sigma, mu) plane
data       physical cloud generation via the retained float64 KS kernel
model      two-condition MeanFlow (5 network inputs -> 7)
train      mean-flow training over an LHS-selected subset of the grid
evaluate   exact histogram W2 on 16 bins/axis, same protocol as the baseline
continuity numerical measurement of parameter continuity
"""
from __future__ import annotations

__all__ = ["protocol", "lhs", "data", "model", "train", "evaluate", "continuity"]
