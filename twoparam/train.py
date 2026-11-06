"""MeanFlow training over an LHS-selected subset of the (sigma, mu) grid.

The loss is the upstream MeanFlow identity, unchanged except that the condition
vector now carries two parameters instead of one::

    v   = x1 - x0                        (target displacement)
    z   = (1 - t) x0 + t x1              (interpolant)
    u, du/dt = jvp(u_theta(z, r, t, cond))
    L   = mean_i || u - (v - (t - r) du/dt) ||^2

Every epoch draws a random *cell* (a (sigma, mu) pair) and then a random batch
inside that cell. This is the same per-condition batching upstream uses, so the
number of gradient steps is directly comparable to the baseline run.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import jvp

from .data import load_cell
from .model import TwoParamMeanflow, mf_sample, mf_sample_two_step, meanflow_loss, sample_times_original
from .protocol import Config, Point


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stack_cells(directory: Path, points: list[Point], config: Config, *, log=print):
    """Load every selected cell: one shared initial cloud plus per-cell targets.

    The initial cloud is drawn once and reused for every cell (see
    ``twoparam.data.generate_cells``), so it is loaded once here and asserted to
    be identical across cells. Targets stay per-cell: ``x1`` has shape (K, N, 3).
    """
    initial = None
    targets, keys = [], []
    for point in points:
        cell = load_cell(directory, point.key)
        start = cell["clouds"]["initial"]
        target = cell["clouds"]["reference"]
        if start.shape != (config.n_train, 3) or target.shape != (config.n_train, 3):
            raise ValueError(f"{point.key}: expected {config.n_train} particles, got {start.shape}")
        if initial is None:
            initial = start
        elif not np.array_equal(initial, start):
            raise ValueError(f"{point.key}: cells do not share one initial cloud; "
                             "the paired design assumption is violated")
        targets.append(target)
        keys.append(point.key)
        log(f"  loaded {point.key} sigma={point.sigma:.4g} mu={point.mu:.4g}")
    return torch.from_numpy(initial), torch.from_numpy(np.stack(targets)), keys


def train(
    data_directory: Path,
    output_directory: Path,
    points: list[Point],
    config: Config,
    *,
    steps: int = 1000,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 88,
    log=print,
) -> dict:
    """Train one two-parameter MeanFlow on the given cells.

    ``data_directory`` is read-only (the generated clouds); every artefact goes
    to ``output_directory`` so a data run is never mutated by training.
    """
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    initial_cloud, targets, keys = _stack_cells(data_directory, points, config, log=log)
    initial_cloud, targets = initial_cloud.to(device), targets.to(device)
    log_sigma = torch.tensor([math.log10(p.sigma) for p in points], dtype=torch.float32, device=device)
    log_mu = torch.tensor([math.log10(p.mu) for p in points], dtype=torch.float32, device=device)

    model = TwoParamMeanflow(width=64, hidden_layers=3, dimension=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rows = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        env = torch.randint(len(points), (1,), device=device).expand(batch_size)
        idx = torch.randint(config.n_train, (batch_size,), device=device)
        x_target = targets[env, idx]        # (batch, 3) physical target, endpoint t = 0
        x_source = initial_cloud[idx]       # (batch, 3) initial cloud, endpoint t = 1
        r, t = sample_times_original(batch_size)
        r, t = r.to(device), t.to(device)
        # Upstream endpoint convention (deepparticle/training/core.py:369):
        # x0 = target, x1 = initial, so the flow runs target --t=0--> initial --t=1.
        # Preserving this orientation is what makes the one-step sampler
        # x - u(x, r=0, t=1) land on the target cloud.
        velocity = x_source - x_target
        z = (1 - t) * x_target + t * x_source
        cond_sigma = log_sigma[env, None]
        cond_mu = log_mu[env, None]
        u, derivative = jvp(
            lambda zz, rr, tt: model(zz, rr, tt, cond_sigma, cond_mu),
            (z, r, t),
            (velocity, torch.zeros_like(r), torch.ones_like(t)),
        )
        desired = (velocity - (t - r) * derivative).detach()
        loss, squared = meanflow_loss(u - desired)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"MF loss is not finite at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == steps:
            rows.append({"step": step, "loss": float(loss.item()),
                         "mse": float(squared.detach().mean().item())})
            log(f"  step {step:>5}/{steps}  loss={loss.item():.6g}")
    seconds = time.perf_counter() - started

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    payload = {
        "model": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "step": steps,
        "seed": seed,
        "architecture": model.architecture(),
        "config": config.metadata(),
        "cells": keys,
        "condition_mode": model.condition_mode,
    }
    checkpoint = destination / "checkpoints" / "mf_two_param.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint)

    history = destination / "training_log.json"
    history.write_text(json.dumps({"steps": steps, "batch_size": batch_size, "lr": lr,
                                   "seconds": seconds, "cells": keys, "rows": rows,
                                   "seed": seed}, indent=2), encoding="utf-8")
    log(f"  trained {steps} steps in {seconds:.1f}s -> {checkpoint.name}")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "seconds": seconds,
        "cells": keys,
        "final_loss": rows[-1]["loss"] if rows else None,
        "seed": seed,
    }


def load_model(path: Path, device: str = "cuda") -> TwoParamMeanflow:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    architecture = payload.get("architecture", {})
    model = TwoParamMeanflow(width=architecture.get("width", 64),
                             hidden_layers=architecture.get("hidden_layers", 3),
                             dimension=architecture.get("dimension", 3))
    model.load_state_dict(payload["model"], strict=True)
    target = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    return model.to(target).eval().requires_grad_(False)


def generate(model: TwoParamMeanflow, source: np.ndarray, sigma: float, mu: float,
             *, two_step: bool = False) -> np.ndarray:
    """One- or two-step generation for a single (sigma, mu) condition."""
    tensor = torch.from_numpy(np.ascontiguousarray(source, dtype=np.float32))
    with torch.no_grad():
        if two_step:
            out = mf_sample_two_step(model, tensor, sigma, mu)
        else:
            out = mf_sample(model, tensor, sigma, mu)
    return out.numpy().astype(np.float32)
