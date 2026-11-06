"""Physical cloud generation for the (sigma, mu) grid.

Uses the *retained upstream float64 kernel* (``training.protocol.simulate_original``)
without copying or re-deriving it, so the physics of this study is bit-compatible
with the single-parameter baseline. Only the ``mu`` argument is new: upstream
always passed ``mu=protocol.mu`` = 1.0.

Cloud layout (one ``.npz`` per grid cell, mirroring the upstream ``ks_cloud_v1``
convention so the baseline tooling can read these files unchanged)::

    meta        JSON string, full provenance
    initial     (N, 3) float32   unit-ball draw, shared across the whole cell set
    reference   (N, 3) float32   the physical target (the "truth")
    source      (N, 3) float32   inference source = the same unit-ball draw

``initial`` and ``source`` are identical by construction: in the upstream KS3D
protocol the MeanFlow always starts from a fresh unit ball, so the cloud to be
transported *is* the initial condition. Keeping both keys makes the files
drop-in compatible with ``paper_reproduction.ks_workflows``.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from .protocol import Config, Point, UPSTREAM

UPSTREAM_DEEPARTICLE = UPSTREAM / "deepparticle"


def _import_kernel():
    """Import the retained upstream simulation kernel without polluting sys.path."""
    import sys

    path = str(UPSTREAM_DEEPARTICLE)
    if path not in sys.path:
        sys.path.insert(0, path)
    from training.protocol import simulate_original  # type: ignore

    return simulate_original


def unit_ball(n: int, seed: int, dimension: int = 3) -> np.ndarray:
    """The upstream initial-condition draw: uniform in the unit ball, float64."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    radius = torch.rand(n, 1, generator=generator).pow(1.0 / dimension)
    direction = torch.randn(n, dimension, generator=generator)
    return (radius * direction / torch.linalg.vector_norm(direction, dim=1, keepdim=True)).numpy()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def simulate_point(point: Point, initial: np.ndarray, config: Config, *, noise_seed: int) -> dict:
    """Integrate one (sigma, mu) case and return the target cloud plus timings."""
    import torch

    simulate_original = _import_kernel()
    device = config.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available; pass --device cpu for a smoke test")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.manual_seed(noise_seed)
    np.random.seed(noise_seed)
    started = time.perf_counter()
    target = simulate_original(
        torch.from_numpy(initial),
        float(point.sigma),
        mu=float(point.mu),
        chi=float(config.chi),
        delta=float(config.delta),
        t=float(config.final_time),
        dt=float(config.dt),
        chunk=256,
        device=device,
    ).numpy()
    seconds = time.perf_counter() - started
    if not np.isfinite(target).all():
        raise FloatingPointError(f"Non-finite KS particles at {point.label}")
    return {"target": target.astype(np.float32), "seconds": seconds}


def write_cloud(path: Path, clouds: dict, meta: dict) -> dict:
    """Write one cell with the upstream key convention; return its manifest record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        meta=np.asarray(json.dumps(meta)),
        **{key: value for key, value in clouds.items()},
    )
    return {
        "file": path.name,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        **meta,
    }


def generate_cells(
    cells: list[Point],
    destination: Path,
    config: Config,
    *,
    progress=None,
) -> dict:
    """Generate one cloud per grid cell; resumable and hash-verified.

    The initial cloud is drawn once and shared by every cell. That is deliberate:
    it turns the cell set into a *paired* design, so differences between cells
    reflect the physical parameters only, never a change of initial sample.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    initial = unit_ball(config.n_train, config.seed, 3)
    initial_hash = hashlib.sha256(initial.tobytes()).hexdigest()

    manifest_path = destination / "manifest.json"
    records: list[dict] = []
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("initial_sha256") == initial_hash:
            records = [r for r in previous.get("cells", []) if (destination / r["file"]).is_file()]
    done = {r["key"] for r in records}

    started = time.perf_counter()
    for point in cells:
        if point.key in done:
            if progress:
                progress(f"skip {point.key} (already present)")
            continue
        cell_started = time.perf_counter()
        noise_seed = config.seed + 1009 * (point.i * len(config.mu_grid) + point.j)
        result = simulate_point(point, initial, config, noise_seed=noise_seed)
        meta = {
            "schema": "ks_cloud_v1",
            "model": "ks3d",
            "level": float(point.sigma),
            "sigma": float(point.sigma),
            "mu": float(point.mu),
            "particles": int(config.n_train),
            "dimension": 3,
            "bins_per_axis": int(config.bins),
            "chi": float(config.chi),
            "delta": float(config.delta),
            "final_time": float(config.final_time),
            "dt": float(config.dt),
            "initial_seed": int(config.seed),
            "initial_sha256": initial_hash,
            "noise_seed": int(noise_seed),
            "seconds": round(result["seconds"], 6),
            "cell_seconds": round(time.perf_counter() - cell_started, 6),
            "grid": {"i": point.i, "j": point.j, "key": point.key, "in_lhs": point.in_lhs},
            **{k: v for k, v in point.meta().items() if k not in ("key", "i", "j", "sigma", "mu")},
        }
        record = write_cloud(
            destination / "clouds" / f"{point.key}.npz",
            {"initial": initial.astype(np.float32), "source": initial.astype(np.float32),
             "reference": result["target"]},
            meta,
        )
        record["file"] = f"clouds/{record['file']}"
        records.append(record)
        manifest = {
            "schema": "twoparam_clouds_v1",
            "status": "running",
            "config": config.metadata(),
            "initial_sha256": initial_hash,
            "cells": records,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if progress:
            progress(f"{point.key} sigma={point.sigma:.4g} mu={point.mu:.4g} "
                     f"{result['seconds']:.2f}s")

    manifest = {
        "schema": "twoparam_clouds_v1",
        "status": "complete",
        "config": config.metadata(),
        "initial_sha256": initial_hash,
        "cells": records,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_cell(directory: Path, key: str) -> dict:
    """Load one cell, refusing silent shape or dtype violations."""
    with np.load(Path(directory) / "clouds" / f"{key}.npz", allow_pickle=False) as values:
        clouds = {name: values[name].copy() for name in values.files if name != "meta"}
        meta = json.loads(values["meta"].item())
    for name, value in clouds.items():
        if value.dtype != np.float32 or value.ndim != 2 or value.shape[1] != 3:
            raise ValueError(f"{key}/{name}: expected float32 (N,3), got {value.dtype} {value.shape}")
    return {"clouds": clouds, "meta": meta}


def load_initial(directory: Path, key: str) -> np.ndarray:
    return load_cell(directory, key)["clouds"]["initial"]
