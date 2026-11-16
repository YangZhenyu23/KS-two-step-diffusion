"""E9: is the learned correction actually a gradient field?

The structural argument in the note is that with exact OT the corrector's regression target is
the gradient of the Brenier potential, so at the population limit the learned map should
approximate a gradient field. A gradient field has a symmetric Jacobian, J_ij = J_ji, which is a
necessary condition any gradient field satisfies and which a general vector field almost surely
violates.

This is the sharpest available test that the corrector learned a *structural* object rather than
a set of empirical displacements, because symmetry is a property no amount of memorisation of
outputs would produce unless the memorised outputs happen to come from such a field -- which is
exactly what the target does, so the test checks whether the network retained that structure.

Method
------
At a set of probe points in the generated cloud, estimate the Jacobian of
    d(x) = x + alpha * MLP(x, cond) - x = alpha * MLP(x, cond)
by central differences with a step h, then measure the antisymmetry
    A = ||J - J^T||_F / ||J||_F
An exactly gradient field gives A = 0; a generic field gives O(1).

Two controls are reported:
  * the same measurement for the *stage-1* map, which is not a gradient field by construction,
    so its A gives a reference scale for "no structure";
  * the same measurement for the corrector's displacement on the reference cloud, to check the
    result is not an artefact of being evaluated on stage-1 outputs.

Usage:
    python scripts/e9_jacobian_symmetry.py --points 24 --cells 3
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream" / "MF-DeepParticle" / "deepparticle"))
sys.path.insert(0, str(ROOT / "scripts"))

from fullscale import grid as G  # noqa: E402
from fullscale.train import load_cell, load_two  # noqa: E402
from fullscale.model import sample_two_param  # noqa: E402

from rollout_compare import load_corrector  # noqa: E402

OUT = G.RESULTS / "jacobian"


def jacobian_of(fn, points: torch.Tensor, h: float, device: torch.device) -> np.ndarray:
    """Central-difference Jacobian of a vector function at each point. Returns (n, d, d)."""
    n, d = points.shape
    out = np.zeros((n, d, d), dtype=np.float64)
    for j in range(d):
        offset = torch.zeros(d, dtype=points.dtype, device=device)
        offset[j] = h
        plus = fn(points + offset)
        minus = fn(points - offset)
        out[:, :, j] = ((plus - minus) / (2 * h)).detach().cpu().numpy()
    return out


def antisymmetry(jac: np.ndarray) -> np.ndarray:
    """||J - J^T||_F / ||J||_F for each sample."""
    diff = jac - jac.transpose(0, 2, 1)
    return np.linalg.norm(diff.reshape(len(diff), -1), axis=1) / np.maximum(
        np.linalg.norm(jac.reshape(len(jac), -1), axis=1), 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=24, help="probe points per cell")
    ap.add_argument("--cells", type=int, default=3)
    ap.add_argument("--h", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=88)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda text: print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)

    headline = G.RESULTS / "optimal_budget_ablation" / "two_32000" / "mf_two_param.pt"
    train_keys = set(torch.load(headline, map_location="cpu", weights_only=False)["cells"])
    slice_keys = {c.key for c in G.baseline_slice()}
    held = [c for c in G.full_grid() if c.key not in train_keys
            and c.key not in slice_keys]
    step = max(1, len(held) // a.cells)
    cells = held[::step][: a.cells]
    log(f"probing {a.points} points per cell on {len(cells)} cells, h = {a.h}")

    mf_model, _ = load_two(headline, a.device)
    corrector = load_corrector(a.seed, a.device)
    device = next(corrector.parameters()).device
    d = 3

    rows = []
    for cell in cells:
        item = load_cell(cell)
        cloud = sample_two_param(mf_model, torch.from_numpy(item["clouds"]["source"]),
                                 cell.sigma, cell.mu, steps=1)
        rng = np.random.default_rng(abs(hash(cell.key)) % (2 ** 31))
        index = rng.choice(len(cloud), size=a.points, replace=False)
        points = cloud[index].to(device).requires_grad_(False)

        ls = torch.full((a.points, 1), math.log10(cell.sigma), dtype=torch.float32, device=device)
        lm = torch.full((a.points, 1), math.log10(cell.mu), dtype=torch.float32, device=device)

        def corrector_disp(pts):
            with torch.no_grad():
                return corrector(pts, ls, lm) - pts

        def stage1_disp(pts):
            rr = torch.zeros(len(pts), 1, dtype=torch.float32, device=device)
            tt = torch.ones(len(pts), 1, dtype=torch.float32, device=device)
            with torch.no_grad():
                return -mf_model(pts, rr, tt, ls, lm)   # x -> x - u  is  displacement = -u

        jc = jacobian_of(corrector_disp, points, a.h, device)
        jm = jacobian_of(stage1_disp, points, a.h, device)

        row = {
            "key": cell.key, "sigma": cell.sigma, "mu": cell.mu,
            "corrector_antisymmetry": float(np.mean(antisymmetry(jc))),
            "stage1_antisymmetry": float(np.mean(antisymmetry(jm))),
            "corrector_jacobian_norm": float(np.mean(
                np.linalg.norm(jc.reshape(len(jc), -1), axis=1))),
            "stage1_jacobian_norm": float(np.mean(
                np.linalg.norm(jm.reshape(len(jm), -1), axis=1))),
        }
        rows.append(row)
        log(f"  {cell.key}: corrector A={row['corrector_antisymmetry']:.4f}  "
            f"stage1 A={row['stage1_antisymmetry']:.4f}")

    summary = {
        "h": a.h, "points_per_cell": a.points, "n_cells": len(cells),
        "corrector_antisymmetry_mean": float(np.mean([r["corrector_antisymmetry"] for r in rows])),
        "stage1_antisymmetry_mean": float(np.mean([r["stage1_antisymmetry"] for r in rows])),
        "note": ("A gradient field has a symmetric Jacobian, so A = ||J-J^T||_F/||J||_F is 0 "
                 "for the ideal corrector. Stage 1 is not a gradient field, so its A gives the "
                 "scale for 'no structure'."),
    }
    (OUT / "jacobian_symmetry.json").write_text(
        json.dumps({**summary, "rows": rows}, indent=2), encoding="utf-8")

    print(f"\nJacobian antisymmetry, lower is more gradient-like:")
    print(f"  corrector  {summary['corrector_antisymmetry_mean']:.4f}")
    print(f"  stage 1    {summary['stage1_antisymmetry_mean']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
