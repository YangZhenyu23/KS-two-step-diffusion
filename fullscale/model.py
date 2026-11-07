"""Two-condition MeanFlow for the full-scale run.

The network is the upstream MeanFlow with one extra conditioning column::

    upstream (sigma only)   Linear(6 -> 64) SELU x3  Linear(64 -> 3)
    this module             Linear(7 -> 64) SELU x3  Linear(64 -> 3)

    upstream  forward(z, r, t, sigma)         = net(cat(z, t, r, log10 sigma))
    here      forward(z, r, t, logsigma, logmu) = net(cat(z, t, r, log10 sigma, log10 mu))

Nothing else changes: same width (64), same depth (3 hidden layers), same SELU
activation, same input ordering ``(z, t, r, ...)``, same default PyTorch
initialisation, same Adam learning rate. The single-parameter control in
``fullscale.train`` uses the 6-input form, so the comparison isolates exactly one
variable: whether the network can see mu.

Endpoint convention
-------------------
The upstream training loop (``deepparticle/training/core.py:369``) is

    x0, x1 = target, initial          # x0 at t = 0, x1 at t = 1
    velocity = x1 - x0
    z = (1 - t) * x0 + t * x1

so the flow runs *target -> initial* and the sampler is ``x - u(x, r=0, t=1)``.
This module keeps that orientation exactly; ``scripts/selfcheck_two_param.py``
verifies it before every experiment.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


class TwoParamMeanflow(nn.Module):
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

    def forward(self, z, r, t, logsigma, logmu):
        return self.net(torch.cat((z, t, r, logsigma, logmu), dim=-1))

    def architecture(self) -> dict:
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
            "profile": "fullscale_two_param",
        }


class OneParamMeanflow(nn.Module):
    """The upstream architecture verbatim: sigma is the only condition."""

    def __init__(self, width: int = 64, hidden_layers: int = 3, dimension: int = 3):
        super().__init__()
        self.width = int(width)
        self.hidden_layers = int(hidden_layers)
        self.dimension = int(dimension)
        self.condition_mode = "log10_sigma"
        layers: list[nn.Module] = [nn.Linear(dimension + 3, width), nn.SELU()]
        for _ in range(self.hidden_layers - 1):
            layers += [nn.Linear(width, width), nn.SELU()]
        layers.append(nn.Linear(width, dimension))
        self.net = nn.Sequential(*layers)

    def forward(self, z, r, t, logsigma):
        return self.net(torch.cat((z, t, r, logsigma), dim=-1))

    def architecture(self) -> dict:
        return {
            "kind": "mf",
            "width": self.width,
            "hidden_layers": self.hidden_layers,
            "activation": "SELU",
            "dimension": self.dimension,
            "conditioning": "log10(sigma)",
            "condition_mode": self.condition_mode,
            "input_order": ["z", "t", "r", "log10_sigma"],
            "input_features": self.dimension + 3,
            "profile": "fullscale_one_param_control",
        }


def sample_times(batch: int, mean: float = -0.4, std: float = 1.0):
    """Upstream time-pair sampler, reproduced exactly.

    Two standard normals are squashed through a sigmoid; the larger becomes ``t``
    and the smaller ``r``. Half the batch is set to ``r = t`` so the network also
    sees the diagonal where the MeanFlow identity degenerates to a velocity.
    """
    samples = np.random.randn(batch, 2).astype(np.float32) * std + mean
    samples = 1.0 / (1.0 + np.exp(-samples))
    t = np.maximum(samples[:, 0], samples[:, 1])
    r = np.minimum(samples[:, 0], samples[:, 1])
    selected = np.random.permutation(batch)[: batch // 2]
    r[selected] = t[selected]
    return torch.from_numpy(r[:, None]), torch.from_numpy(t[:, None])


def meanflow_loss(error: torch.Tensor):
    if error.ndim != 2 or min(error.shape) == 0:
        raise ValueError("MF error must have shape (nonempty batch, vector dimension)")
    squared = error.square().sum(-1)
    return squared.mean(), squared


@torch.no_grad()
def sample_two_param(model: TwoParamMeanflow, source: torch.Tensor, sigma: float, mu: float,
                     steps: int = 1) -> torch.Tensor:
    """One-step (``steps=1``) or two-step (``steps=2``) generation.

    ``steps=1`` reproduces the upstream one-step sampler: ``x - u(x, 0, 1)``.
    ``steps=2`` splits [0, 1] into [0.5, 1] and [0, 0.5], each applied with the
    interval length as the step size, matching the upstream two-step schedule.
    """
    device = next(model.parameters()).device
    x = source.detach().to(device)
    batch = len(x)
    logsigma = torch.full((batch, 1), math.log10(float(sigma)), dtype=x.dtype, device=device)
    logmu = torch.full((batch, 1), math.log10(float(mu)), dtype=x.dtype, device=device)
    if steps == 1:
        t = torch.ones(batch, 1, dtype=x.dtype, device=device)
        r = torch.zeros(batch, 1, dtype=x.dtype, device=device)
        return (x - model(x, r, t, logsigma, logmu)).cpu()
    if steps == 2:
        for t_hi, t_lo in ((1.0, 0.5), (0.5, 0.0)):
            t = torch.full((batch, 1), float(t_hi), dtype=x.dtype, device=device)
            r = torch.full((batch, 1), float(t_lo), dtype=x.dtype, device=device)
            x = x - (float(t_hi) - float(t_lo)) * model(x, r, t, logsigma, logmu)
        return x.cpu()
    raise ValueError("steps must be 1 or 2")


@torch.no_grad()
def sample_one_param(model: OneParamMeanflow, source: torch.Tensor, sigma: float) -> torch.Tensor:
    device = next(model.parameters()).device
    x = source.detach().to(device)
    batch = len(x)
    r = torch.zeros(batch, 1, dtype=x.dtype, device=device)
    t = torch.ones(batch, 1, dtype=x.dtype, device=device)
    logsigma = torch.full((batch, 1), math.log10(float(sigma)), dtype=x.dtype, device=device)
    return (x - model(x, r, t, logsigma)).cpu()
