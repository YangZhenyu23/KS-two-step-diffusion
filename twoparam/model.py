"""Two-condition MeanFlow: the network takes (sigma, mu) instead of sigma alone.

Upstream ``Meanflow`` (``deepparticle/training/core.py``)::

    net: Linear(6 -> 64) SELU  Linear(64 -> 64) SELU  Linear(64 -> 64) SELU  Linear(64 -> 3)
    forward(z, r, t, sigma) = net(cat(z, t, r, log10(sigma)))

Two-parameter extension (this module)::

    net: Linear(7 -> 64) SELU  Linear(64 -> 64) SELU  Linear(64 -> 64) SELU  Linear(64 -> 3)
    forward(z, r, t, cond)  = net(cat(z, t, r, log10(sigma), log10(mu)))

The change is deliberately minimal — one extra input column, everything else
identical (width 64, three hidden layers, SELU, the same (z, t, r, ...) input
order and the same learning rate schedule). That keeps the comparison with the
single-parameter baseline an apples-to-apples architectural comparison: the only
thing that differs is how many physical parameters the network can see.

Why log10 for both
------------------
The parameter grid is geometric in both directions (sigma spans 10..200, mu
spans 0.1..10). Training on log coordinates makes the network see an evenly
populated input box rather than a box with all the samples crushed near the
origin, and it matches what upstream already does for sigma.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.func import jvp


class TwoParamMeanflow(nn.Module):
    """MeanFlow with two physical conditions; same body as the upstream network."""

    def __init__(self, width: int = 64, hidden_layers: int = 3, dimension: int = 3):
        super().__init__()
        self.width = int(width)
        self.hidden_layers = int(hidden_layers)
        self.dimension = int(dimension)
        self.condition_mode = "log10_sigma_log10_mu"
        layers: list[nn.Module] = [nn.Linear(dimension + 4, width), nn.SELU()]
        for _ in range(self.hidden_layers - 1):
            layers += [nn.Linear(width, width), nn.SELU()]
        layers.append(nn.Linear(width, dimension))
        self.net = nn.Sequential(*layers)

    # -- the exact upstream call signature, with one extra condition ----------
    def forward(self, z, r, t, logsigma, logmu):
        return self.net(torch.cat((z, t, r, logsigma, logmu), dim=-1))
    def forward_condition(self, z, r, t, sigma, mu):
        """Convenience wrapper taking physical (not log) parameters."""
        logsigma = torch.log10(torch.as_tensor(sigma, dtype=z.dtype, device=z.device))
        logmu = torch.log10(torch.as_tensor(mu, dtype=z.dtype, device=z.device))
        return self.forward(z, r, t, logsigma, logmu)

    def architecture(self) -> dict:
        """The identity block written into every checkpoint, mirroring upstream."""
        return {
            "kind": "mf",
            "width": self.width,
            "hidden_layers": self.hidden_layers,
            "activation": "SELU",
            "dimension": self.dimension,
            "conditioning": "log10(sigma), log10(mu)",
            "condition_mode": self.condition_mode,
            "input_order": ["z", "t", "r", "log10_sigma", "log10_mu"],
            "input_features": self.dimension + 4,
            "profile": "original_tuned_two_param",
        }


def sample_times_original(batch: int, mean: float = -0.4, std: float = 1.0, seed=None):
    """Copy of the upstream time-pair sampler, so training statistics are identical."""
    rng = np.random if seed is None else np.random.default_rng(seed)
    samples = rng.randn(batch, 2).astype(np.float32) * std + mean
    samples = 1.0 / (1.0 + np.exp(-samples))
    t = np.maximum(samples[:, 0], samples[:, 1])
    r = np.minimum(samples[:, 0], samples[:, 1])
    selected = np.random.permutation(batch)[: batch // 2]
    r[selected] = t[selected]
    return torch.from_numpy(r[:, None]), torch.from_numpy(t[:, None])


def meanflow_loss(error: torch.Tensor):
    """Mean over particles of the squared vector error (identical to upstream)."""
    if error.ndim != 2 or min(error.shape) == 0:
        raise ValueError("MF error must have shape (nonempty batch, vector dimension)")
    squared = error.square().sum(-1)
    return squared.mean(), squared


@torch.no_grad()
def mf_sample(model: TwoParamMeanflow, source: torch.Tensor, sigma: float, mu: float) -> torch.Tensor:
    """One-step generation ``x_mf = x - u_theta(x, r=0, t=1, sigma, mu)``."""
    device = next(model.parameters()).device
    x = source.detach().to(device)
    batch = len(x)
    r = torch.zeros(batch, 1, dtype=x.dtype, device=device)
    t = torch.ones(batch, 1, dtype=x.dtype, device=device)
    logsigma = torch.full((batch, 1), math.log10(float(sigma)), dtype=x.dtype, device=device)
    logmu = torch.full((batch, 1), math.log10(float(mu)), dtype=x.dtype, device=device)
    return (x - model(x, r, t, logsigma, logmu)).cpu()


@torch.no_grad()
def mf_sample_two_step(model: TwoParamMeanflow, source: torch.Tensor, sigma: float, mu: float,
                       schedule=((1.0, 0.5), (0.5, 0.0))) -> torch.Tensor:
    """Two-step generation: the same weights, two (r, t) intervals."""
    device = next(model.parameters()).device
    x = source.detach().to(device)
    batch = len(x)
    logsigma = torch.full((batch, 1), math.log10(float(sigma)), dtype=x.dtype, device=device)
    logmu = torch.full((batch, 1), math.log10(float(mu)), dtype=x.dtype, device=device)
    for t_hi, r_lo in schedule:
        t = torch.full((batch, 1), float(t_hi), dtype=x.dtype, device=device)
        r = torch.full((batch, 1), float(r_lo), dtype=x.dtype, device=device)
        velocity = model(x, r, t, logsigma, logmu)
        x = x - (float(t_hi) - float(r_lo)) * velocity
    return x.cpu()
