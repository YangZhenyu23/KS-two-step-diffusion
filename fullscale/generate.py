"""Resumable generation of the 200-cell, 100k-particle physical dataset.

Every cell stores the *same* initial cloud (a single unit-ball draw, seed 88) and
its own terminal reference cloud. Sharing the initial condition is what makes the
parameter family observable as a continuous map: two cells differ only by
(sigma, mu), never by a resampling of the initial measure.

Each completed cell is written atomically and recorded in ``manifest.json`` with
its SHA256, so an interrupted run resumes where it stopped and a completed run
can be verified without recomputing anything.

Layout::

    results/fullscale/data/full/
        manifest.json
        clouds/s00_m00.npz ... s19_m09.npz      (200 files, ~14 MB each)
        initial.npy                             (shared initial cloud)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream" / "MF-DeepParticle" / "deepparticle"))

from fullscale.grid import (  # noqa: E402
    BINS, CHI, DATA, DELTA, DT, FINAL_TIME, MU_GRID, N_PARTICLES, SIGMA_GRID,
    Cell, Config, full_grid,
)


def log(text: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {text}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unit_ball(n: int, seed: int, dimension: int = 3) -> np.ndarray:
    """Upstream initial-condition draw: uniform in the unit ball, float64."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    radius = torch.rand(n, 1, generator=generator).pow(1.0 / dimension)
    direction = torch.randn(n, dimension, generator=generator)
    return (radius * direction / torch.linalg.vector_norm(direction, dim=1, keepdim=True)).numpy()


def simulate(initial: np.ndarray, cell: Cell, config: Config, noise_seed: int) -> np.ndarray:
    from training.protocol import simulate_original

    np.random.seed(noise_seed)
    torch.manual_seed(noise_seed)
    target = simulate_original(
        torch.from_numpy(initial), cell.sigma, mu=cell.mu, chi=config.chi,
        delta=config.delta, t=config.final_time, dt=config.dt, chunk=256,
        device=config.device,
    ).numpy()
    if not np.isfinite(target).all():
        raise FloatingPointError(f"non-finite cloud at {cell.key}")
    return target.astype(np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_PARTICLES)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only-sigma-index", type=int, nargs="+", default=None,
                    help="restrict to these sigma indices (for smoke tests)")
    ap.add_argument("--only-mu-index", type=int, nargs="+", default=None)
    ap.add_argument("--cells-from", type=Path, default=None,
                    help="JSON list of [i,j] cell indices to generate exclusively")
    ap.add_argument("--skip-cells-from", type=Path, default=None,
                    help="JSON list of [i,j] cell indices to omit")
    ap.add_argument("--force", action="store_true", help="regenerate cells already present")
    a = ap.parse_args(argv)

    config = Config(n_train=a.n, device=a.device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "clouds").mkdir(exist_ok=True)

    initial = unit_ball(config.n_train, config.seed)
    initial_hash = hashlib.sha256(initial.tobytes()).hexdigest()
    initial_path = DATA / "initial.npy"
    if initial_path.is_file() and not a.force:
        existing = np.load(initial_path)
        if existing.shape != initial.shape or not np.array_equal(existing, initial):
            raise SystemExit("existing initial.npy differs from the seed-88 draw; "
                             "delete it (or pass --force) to rebuild the dataset")
    else:
        np.save(initial_path, initial)
    log(f"initial cloud: N={config.n_train} sha256={initial_hash[:16]}")

    cells = full_grid()
    if a.cells_from:
        wanted = {(int(i), int(j)) for i, j in json.loads(Path(a.cells_from).read_text(encoding="utf-8"))}
        cells = [c for c in cells if (c.i, c.j) in wanted]
    if a.skip_cells_from:
        skipped = {(int(i), int(j)) for i, j in json.loads(Path(a.skip_cells_from).read_text(encoding="utf-8"))}
        cells = [c for c in cells if (c.i, c.j) not in skipped]
    if a.only_sigma_index:
        cells = [c for c in cells if c.i in set(a.only_sigma_index)]
    if a.only_mu_index:
        cells = [c for c in cells if c.j in set(a.only_mu_index)]
    log(f"selected {len(cells)} cells to generate")

    manifest_path = DATA / "manifest.json"
    previous = {}
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("initial_sha256") == initial_hash:
            previous = {r["key"]: r for r in payload.get("cells", [])}
            log(f"resuming: {len(previous)} cells on record")
        else:
            raise SystemExit("manifest belongs to a different initial cloud")

    started = time.perf_counter()
    done_count = 0
    for position, cell in enumerate(cells, start=1):
        record = previous.get(cell.key)
        path = DATA / "clouds" / f"{cell.key}.npz"
        if record and path.is_file() and not a.force:
            if path.stat().st_size == record.get("bytes") and sha256(path) == record["sha256"]:
                log(f"[{position}/{len(cells)}] {cell.key} sigma={cell.sigma:g} "
                    f"mu={cell.mu:g} cached")
                done_count += 1
                continue
            log(f"[{position}/{len(cells)}] {cell.key} hash mismatch, regenerating")

        cell_started = time.perf_counter()
        noise_seed = config.seed + 1009 * (cell.i * len(MU_GRID) + cell.j)
        target = simulate(initial, cell, config, noise_seed)
        seconds = time.perf_counter() - cell_started

        meta = {
            "schema": "ks_cloud_v1",
            "model": "ks3d",
            "level": cell.sigma,
            "sigma": cell.sigma,
            "mu": cell.mu,
            "particles": int(config.n_train),
            "dimension": 3,
            "bins_per_axis": BINS,
            "chi": config.chi,
            "delta": config.delta,
            "final_time": config.final_time,
            "dt": config.dt,
            "initial_seed": config.seed,
            "initial_sha256": initial_hash,
            "noise_seed": int(noise_seed),
            "seconds": round(seconds, 4),
            "grid": {"i": cell.i, "j": cell.j, "key": cell.key},
            **{k: v for k, v in cell.meta().items() if k not in ("key", "i", "j", "sigma", "mu")},
        }
        # np.savez_compressed appends ".npz" when the name lacks it, so write to an
        # explicit file object to keep the atomic-rename target predictable.
        tmp = path.with_suffix(".part")
        with tmp.open("wb") as stream:
            np.savez_compressed(stream, meta=np.asarray(json.dumps(meta)),
                                initial=initial.astype(np.float32),
                                source=initial.astype(np.float32),
                                reference=target)
        os.replace(tmp, path)
        record = {"key": cell.key, "file": f"clouds/{path.name}", "sha256": sha256(path),
                  "bytes": path.stat().st_size, "i": cell.i, "j": cell.j,
                  "sigma": cell.sigma, "mu": cell.mu, **meta}
        previous[cell.key] = record
        manifest_path.write_text(json.dumps({
            "schema": "fullscale_clouds_v1", "status": "running", "config": config.metadata(),
            "initial_sha256": initial_hash,
            "cells": sorted(previous.values(), key=lambda r: (r["i"], r["j"])),
            "elapsed_seconds": round(time.perf_counter() - started, 1),
        }, indent=1), encoding="utf-8")
        done_count += 1
        elapsed = time.perf_counter() - started
        remaining = len(cells) - position
        eta = (elapsed / position) * remaining if position else 0
        log(f"[{position}/{len(cells)}] {cell.key} sigma={cell.sigma:g} mu={cell.mu:g} "
            f"{seconds:.1f}s  ({done_count} done, eta {eta / 60:.0f} min)")

    manifest_path.write_text(json.dumps({
        "schema": "fullscale_clouds_v1",
        "status": "complete" if len(previous) >= len(full_grid()) else "partial",
        "config": config.metadata(),
        "initial_sha256": initial_hash,
        "cells": sorted(previous.values(), key=lambda r: (r["i"], r["j"])),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }, indent=1), encoding="utf-8")
    log(f"finished: {len(previous)}/{len(full_grid())} cells complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
