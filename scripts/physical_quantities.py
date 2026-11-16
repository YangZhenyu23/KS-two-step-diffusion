"""Task-relevant physical quantities along the trajectory, not just W2 at t=T.

Why: W2 at the final time is a single scalar and can be small while the physical
content of the cloud is wrong (two clouds can have the same W2 but different anisotropy,
which is the quantity the shear parameter sigma actually controls). Harry asked for
"trajectory or task-relevant physical quantities", so this reports the moments the KS
system is actually about.

The quantities
--------------
  centroid_x     driven by the shear sigma*g; analytically it should grow roughly as
                 sigma * T * <exp(-y^2-z^2)>, so it is the cleanest signal of whether
                 the model reproduces the advection.
  rms_radius     sqrt of the trace of the covariance; grows under diffusion.
  eig_{1,2,3}    principal standard deviations of the cloud.
  anisotropy     eig_1 / eig_2. **This is the quantity sigma controls**: sigma stretches
                 the cloud along x, so anisotropy should increase with sigma and be
                 nearly independent of mu. If the model tracks W2 but misses anisotropy,
                 it has the mass in roughly the right place with the wrong shape.

Measured at the model output and at the physical reference for the same cell, so the
comparison is like-for-like. Deviation is reported per quantity as an absolute and a
relative error.

Usage:
    python scripts/physical_quantities.py --cells 6 --n 20000 --score-n 4000
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

from fullscale import grid as G  # noqa: E402
from fullscale.train import load_cell, load_two  # noqa: E402
from fullscale.model import sample_two_param  # noqa: E402
from twoparam.data import unit_ball  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from rollout_stability import simulate  # noqa: E402

OUT = G.RESULTS / "physics"
CELLS = [(0, 1), (3, 4), (9, 6), (13, 7), (19, 9), (11, 9)]


def load_corrector(seed: int, device: str):
    from dp_two_param import TwoParamCorrector

    path = G.RESULTS / "dp_two_param" / f"s{seed}" / "dp_corrector.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    arch = checkpoint.get("architecture", {})
    model = TwoParamCorrector(
        width=arch.get("width", 64), alpha=checkpoint.get("alpha", 1.0),
        blocks=checkpoint.get("blocks", arch.get("blocks", 2)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def quantities(cloud: np.ndarray) -> dict:
    centred = cloud - cloud.mean(0)
    covariance = np.cov(centred.T)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    std = np.sqrt(np.maximum(eigenvalues, 0.0))
    return {
        "centroid_x": float(cloud[:, 0].mean()),
        "centroid_y": float(cloud[:, 1].mean()),
        "centroid_z": float(cloud[:, 2].mean()),
        "rms_radius": float(np.sqrt((centred ** 2).sum(1).mean())),
        "std_x": float(cloud[:, 0].std()),
        "std_y": float(cloud[:, 1].std()),
        "std_z": float(cloud[:, 2].std()),
        "eig_1": float(std[0]), "eig_2": float(std[1]), "eig_3": float(std[2]),
        "anisotropy": float(std[0] / max(std[1], 1e-12)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=6)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--score-n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=88)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda text: print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)

    mf_model, _ = load_two(G.RESULTS / "optimal_budget_ablation" / "two_32000"
                           / "mf_two_param.pt", a.device)
    corrector = load_corrector(a.seed, a.device)
    initial = unit_ball(a.n, 88)
    rng = np.random.default_rng(31415)
    index = rng.choice(a.n, size=min(a.score_n, a.n), replace=False)
    source = torch.from_numpy(initial.astype(np.float32))

    rows = []
    for position, (i, j) in enumerate(CELLS[: a.cells], start=1):
        cell = G.Cell(i, j)
        sigma, mu = cell.sigma, cell.mu
        log(f"=== {position}/{min(a.cells, len(CELLS))} {cell.key} "
            f"sigma={sigma:g} mu={mu:g} ===")

        reference = simulate(initial, sigma, mu, G.FINAL_TIME,
                             6_660_000 + 11 * (i * 10 + j), a.device)

        mf_cloud = sample_two_param(mf_model, source, sigma, mu, steps=1).numpy().astype(np.float64)
        device = next(corrector.parameters()).device
        cs = torch.full((len(mf_cloud), 1), math.log10(sigma), dtype=torch.float32, device=device)
        cm = torch.full((len(mf_cloud), 1), math.log10(mu), dtype=torch.float32, device=device)
        with torch.no_grad():
            dp_cloud = corrector(torch.from_numpy(mf_cloud).float().to(device), cs, cm)
        dp_cloud = dp_cloud.cpu().numpy().astype(np.float64)

        for method, cloud in (("mf", mf_cloud), ("mf_dp", dp_cloud)):
            q = quantities(cloud[index])
            truth = quantities(reference[index])
            entry = {"key": cell.key, "sigma": sigma, "mu": mu, "method": method,
                     "predicted": q, "reference": truth}
            entry["errors"] = {
                name: {
                    "absolute": q[name] - truth[name],
                    "relative": ((q[name] - truth[name]) / truth[name]
                                 if abs(truth[name]) > 1e-9 else None),
                }
                for name in q
            }
            rows.append(entry)
        log(f"  ok (aniso ref {quantities(reference[index])['anisotropy']:.3f}, "
            f"mf {quantities(mf_cloud[index])['anisotropy']:.3f}, "
            f"dp {quantities(dp_cloud[index])['anisotropy']:.3f})")

    (OUT / "physical_quantities.json").write_text(
        json.dumps({"cells": CELLS[: a.cells], "n": a.n, "score_n": int(len(index)),
                    "rows": rows}, indent=2), encoding="utf-8")

    print(f"\n{'cell':>8} {'sig':>5} {'mu':>7} {'method':>7} "
          f"{'cent_x':>8} {'ref':>8} {'rms':>7} {'ref':>7} {'aniso':>7} {'ref':>7} "
          f"{'aniso_e':>8}")
    for row in rows:
        p, r = row["predicted"], row["reference"]
        print(f"{row['key']:>8} {row['sigma']:5.0f} {row['mu']:7.3f} {row['method']:>7} "
              f"{p['centroid_x']:8.3f} {r['centroid_x']:8.3f} "
              f"{p['rms_radius']:7.3f} {r['rms_radius']:7.3f} "
              f"{p['anisotropy']:7.3f} {r['anisotropy']:7.3f} "
              f"{row['errors']['anisotropy']['relative'] if row['errors']['anisotropy']['relative'] is not None else float('nan'):8.3f}")

    # summary: mean absolute relative error per quantity, per method
    print("\nmean |relative error| by quantity and method")
    names = list(rows[0]["predicted"])
    print(f"  {'quantity':>12} " + " ".join(f"{m:>9}" for m in ("mf", "mf_dp")))
    for name in names:
        line = f"  {name:>12} "
        for method in ("mf", "mf_dp"):
            values = [abs(row["errors"][name]["relative"]) for row in rows
                      if row["method"] == method and row["errors"][name]["relative"] is not None]
            line += f"{(np.mean(values) if values else float('nan')):9.4f} "
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
