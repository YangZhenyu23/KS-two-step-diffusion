"""Training for the full-scale two-parameter study.

Two models are trained under the *same* protocol so the comparison isolates the
value of the second parameter:

``two``   a 7-input MeanFlow conditioned on (sigma, mu); for ``n_cells = 40`` and
          ``u`` updates per cell it runs ``40*u`` steps.
``one``   the upstream 6-input MeanFlow conditioned on sigma only; it can only be
          trained on cells with a fixed mu, so it uses the ``mu = 1`` row
          (20 cells) and runs ``20*u`` steps. Its budget per cell is identical,
          which is the only fair way to compare across different cell counts.

Memory
------
100k particles x 3 coordinates x 4 bytes = 1.2 MB per cloud. Forty training
clouds plus their targets is about 96 MB, so the whole training set stays
resident on the GPU and each step is a pure gather. Batch size 4096 is used for
100k particles (the pilot used 256 for 20k); the larger batch keeps the gradient
noise per step comparable while the per-step cost grows with N.

Checkpoint format
-----------------
A dict with ``model`` (state dict), ``architecture``, ``step``, ``cells`` (the
training keys, in order), ``sigma_grid``, ``mu_grid``, ``n_train``, ``bins`` and
``seed``. ``evaluate`` refuses to load a checkpoint whose architecture or grid
does not match, so a stale checkpoint can never be silently scored against a new
dataset.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.func import jvp

from .grid import DATA, MU_GRID, SIGMA_GRID, Cell, Config, ensure
from .model import OneParamMeanflow, TwoParamMeanflow, meanflow_loss, sample_times


def log(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def load_cell(cell: Cell) -> dict:
    path = DATA / "clouds" / f"{cell.key}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing cloud {path}; run fullscale.generate first")
    with np.load(path, allow_pickle=False) as values:
        clouds = {name: values[name].copy() for name in values.files if name != "meta"}
        meta = json.loads(values["meta"].item())
    return {"clouds": clouds, "meta": meta}


def load_training_tensors(cells: list[Cell], device: torch.device, *, dtype=torch.float32):
    """Stack the training clouds: one shared initial (N,3), per-cell targets (K,N,3)."""
    initial = None
    targets = []
    for cell in cells:
        payload = load_cell(cell)
        start = payload["clouds"]["initial"]
        target = payload["clouds"]["reference"]
        if initial is None:
            initial = start
        elif not np.array_equal(initial, start):
            raise ValueError(f"{cell.key}: cells do not share one initial cloud")
        targets.append(target)
        log(f"  loaded {cell.key} sigma={cell.sigma:g} mu={cell.mu:g} "
            f"({len(target)} particles)")
    return (torch.from_numpy(initial).to(device),
            torch.from_numpy(np.stack(targets)).to(device))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_two_param(
    cells: list[Cell],
    config: Config,
    out_dir: Path,
    *,
    steps: int,
    batch_size: int = 4096,
    lr: float = 1e-3,
    seed: int = 88,
    checkpoint_every: int = 0,
    conditioning: str = "log10",
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))

    log(f"training two-parameter MeanFlow: {len(cells)} cells, {steps} steps, "
        f"batch {batch_size}, N={config.n_train}, conditioning={conditioning}")
    initial, targets = load_training_tensors(cells, device)

    model = TwoParamMeanflow(width=64, hidden_layers=3, dimension=3,
                             conditioning=conditioning).to(device)
    # Encode each cell's parameters through the model's own conditioning, so the
    # training inputs and the inference inputs are guaranteed to be identical.
    log_sigma, log_mu = model.condition(
        torch.tensor([c.sigma for c in cells], device=device),
        torch.tensor([c.mu for c in cells], device=device),
        dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rows = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        env = torch.randint(len(cells), (1,), device=device).expand(batch_size)
        idx = torch.randint(config.n_train, (batch_size,), device=device)
        x_target = targets[env, idx]
        x_source = initial[idx]
        r, t = sample_times(batch_size)
        r, t = r.to(device), t.to(device)
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
        loss, _ = meanflow_loss(u - desired)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 200 == 0 or step == steps:
            rows.append({"step": step, "loss": float(loss.item())})
            log(f"  step {step}/{steps} loss={loss.item():.6g}")
        if checkpoint_every and step % checkpoint_every == 0 and step != steps:
            _save(out_dir / f"mf_two_param_step{step}.pt", model, cells, config, step, seed)
    seconds = time.perf_counter() - started
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    path = out_dir / "mf_two_param.pt"
    _save(path, model, cells, config, steps, seed)
    (out_dir / "training_log.json").write_text(json.dumps({
        "kind": "two_param", "steps": steps, "batch_size": batch_size, "lr": lr,
        "seconds": seconds, "rows": rows, "cells": [c.key for c in cells],
        "n_cells": len(cells), "seed": seed,
    }, indent=2), encoding="utf-8")
    log(f"  done in {seconds:.1f}s -> {path.name} sha256={_sha256(path)[:16]}")
    return {"checkpoint": str(path), "sha256": _sha256(path), "steps": steps,
            "seconds": seconds, "final_loss": rows[-1]["loss"], "n_cells": len(cells)}


def train_one_param(
    cells: list[Cell],
    config: Config,
    out_dir: Path,
    *,
    steps: int,
    batch_size: int = 4096,
    lr: float = 1e-3,
    seed: int = 88,
    allow_mixed_mu: bool = False,
) -> dict:
    """The upstream sigma-only architecture.

    By default the control must see exactly one mu value, which keeps it a
    genuine single-parameter model. ``allow_mixed_mu=True`` relaxes this for the
    deliberate diagnostic where the sigma-only network is trained on cells that
    carry several mu values: it then cannot fit the data by construction, and the
    resulting irreducible training error measures exactly how much information
    mu carries.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu")
    mus = {c.mu for c in cells}
    if len(mus) != 1 and not allow_mixed_mu:
        raise ValueError("the single-parameter control must see exactly one mu value; "
                         "pass allow_mixed_mu=True only for the information-loss diagnostic")
    torch.manual_seed(seed)
    np.random.seed(seed)

    mu_label = f"mu={mus.pop()}" if len(mus) == 1 else f"{len(mus)} mixed mu values"
    log(f"training one-parameter control: {len(cells)} cells ({mu_label}), "
        f"{steps} steps, batch {batch_size}")
    initial, targets = load_training_tensors(cells, device)
    log_sigma = torch.tensor([math.log10(c.sigma) for c in cells],
                             dtype=torch.float32, device=device)

    model = OneParamMeanflow(width=64, hidden_layers=3, dimension=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rows = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        env = torch.randint(len(cells), (1,), device=device).expand(batch_size)
        idx = torch.randint(config.n_train, (batch_size,), device=device)
        x_target = targets[env, idx]
        x_source = initial[idx]
        r, t = sample_times(batch_size)
        r, t = r.to(device), t.to(device)
        velocity = x_source - x_target
        z = (1 - t) * x_target + t * x_source
        cond = log_sigma[env, None]
        u, derivative = jvp(lambda zz, rr, tt: model(zz, rr, tt, cond),
                            (z, r, t), (velocity, torch.zeros_like(r), torch.ones_like(t)))
        loss, _ = meanflow_loss(u - (velocity - (t - r) * derivative).detach())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 200 == 0 or step == steps:
            rows.append({"step": step, "loss": float(loss.item())})
            log(f"  step {step}/{steps} loss={loss.item():.6g}")
    seconds = time.perf_counter() - started
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    path = out_dir / "mf_one_param.pt"
    torch.save({"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "architecture": model.architecture(), "step": steps,
                "cells": [c.key for c in cells], "sigma_grid": SIGMA_GRID,
                "mu_grid": MU_GRID, "n_train": config.n_train, "bins": config.bins,
                "seed": seed}, path)
    (out_dir / "training_log.json").write_text(json.dumps({
        "kind": "one_param", "steps": steps, "batch_size": batch_size, "lr": lr,
        "seconds": seconds, "rows": rows, "cells": [c.key for c in cells], "seed": seed,
    }, indent=2), encoding="utf-8")
    log(f"  done in {seconds:.1f}s -> {path.name}")
    return {"checkpoint": str(path), "sha256": _sha256(path), "steps": steps,
            "seconds": seconds, "final_loss": rows[-1]["loss"], "n_cells": len(cells)}


def _save(path: Path, model: TwoParamMeanflow, cells: list[Cell], config: Config,
          step: int, seed: int) -> None:
    torch.save({
        "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "architecture": {**model.architecture(),
                         "conditioning_mode": model.conditioning},
        "step": step,
        "cells": [c.key for c in cells],
        "sigma_grid": SIGMA_GRID,
        "mu_grid": MU_GRID,
        "n_train": config.n_train,
        "bins": config.bins,
        "seed": seed,
    }, path)


def load_two(path: Path, device: str = "cuda") -> TwoParamMeanflow:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    arch = payload.get("architecture", {})
    # Older checkpoints predate the conditioning field; they were all trained with
    # plain log10, so that is the correct default for them.
    mode = arch.get("conditioning_mode") or "log10"
    if "condition_mode" in arch and "log10_norm" in str(arch["condition_mode"]):
        mode = "log10_norm"
    elif "condition_mode" in arch and "unit01" in str(arch["condition_mode"]):
        mode = "unit01"
    model = TwoParamMeanflow(width=arch.get("width", 64),
                             hidden_layers=arch.get("hidden_layers", 3),
                             dimension=arch.get("dimension", 3),
                             conditioning=mode)
    model.load_state_dict(payload["model"], strict=True)
    target = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    return model.to(target).eval().requires_grad_(False), payload


def load_one(path: Path, device: str = "cuda") -> OneParamMeanflow:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    arch = payload.get("architecture", {})
    model = OneParamMeanflow(width=arch.get("width", 64),
                             hidden_layers=arch.get("hidden_layers", 3),
                             dimension=arch.get("dimension", 3))
    model.load_state_dict(payload["model"], strict=True)
    target = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    return model.to(target).eval().requires_grad_(False), payload
