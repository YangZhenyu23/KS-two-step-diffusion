"""Does the corrector change rollout stability? MF vs MF+DP vs two-step MF at matched cost.

The question Harry asked precisely: "show whether the corrector improves typical cases
but introduces a small number of unstable rollouts". A one-step W2 cannot answer it,
because both methods look fine at k=1 and both explode by k=4.

Three maps are compared, all at exactly 2 forward passes per application so the cost is
matched against MF+DP:

  mf        the frozen MeanFlow, applied twice           (2 forwards)
  mf_dp     MeanFlow once, then the corrector once       (2 forwards: 1 MF + 1 MLP)
  mf_2step  the frozen MeanFlow with the (r,t)=(0.5,1),(0,0.5) schedule  (2 forwards)

plus ``mf`` at k=1 as the reference operating point (1 forward), which is the setting
the model was actually designed and reported for.

What is measured per application count k
----------------------------------------
  w2_phys    W2 to the physical solution at time k*T (same initial cloud)
  amp        step amplification: W2(cloud_k, cloud_{k-1}) / W2(phys_k, phys_{k-1}).
             A faithful time-stepper has amp ~ 1. amp > 1 means the model moves the
             cloud further per step than the physics does, i.e. it compounds error.
  ratio      w2_phys / floor

The k=1 row is the honest operating point. Everything beyond it is the extrapolation
test: the maps were never trained on inputs drawn from their own output distribution,
so divergence is expected. The point of the experiment is to quantify *how fast* and to
check whether the corrector makes it better or worse.

Usage:
    python scripts/rollout_compare.py --cells 3 --ks 1 2
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
from rollout_stability import exact_w2, simulate, invariants  # noqa: E402

OUT = G.RESULTS / "rollout"
CELLS = [(3, 4), (9, 6), (19, 9)]


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


def apply_corrector(corrector, cloud: torch.Tensor, sigma: float, mu: float) -> torch.Tensor:
    device = next(corrector.parameters()).device
    x = cloud.to(device)
    cs = torch.full((len(x), 1), math.log10(sigma), dtype=torch.float32, device=device)
    cm = torch.full((len(x), 1), math.log10(mu), dtype=torch.float32, device=device)
    with torch.no_grad():
        return corrector(x, cs, cm).cpu()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=3)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--score-n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=88)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda text: print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)

    # The MeanFlow checkpoint must be paired with its own corrector: a corrector trained on
    # seed 89's stage-1 output does not apply to seed 88's. Seed 88 uses the unsuffixed
    # directory, the others use the _sNN suffix.
    mf_dir = ("two_32000" if a.seed == 88 else f"two_32000_s{a.seed}")
    mf_model, _ = load_two(G.RESULTS / "optimal_budget_ablation" / mf_dir
                           / "mf_two_param.pt", a.device)
    corrector = load_corrector(a.seed, a.device)
    initial = unit_ball(a.n, 88)
    rng = np.random.default_rng(20261116)
    index = rng.choice(a.n, size=min(a.score_n, a.n), replace=False)
    source = torch.from_numpy(initial.astype(np.float32))

    rows = []
    for position, (i, j) in enumerate(CELLS[: a.cells], start=1):
        cell = G.Cell(i, j)
        sigma, mu = cell.sigma, cell.mu
        log(f"=== {position}/{min(a.cells, len(CELLS))} {cell.key} "
            f"sigma={sigma:g} mu={mu:g} ===")

        physical = {}
        for k in a.ks:
            seed = 8_100_000 + 17 * (i * 10 + j) + 103 * k
            physical[k] = simulate(initial, sigma, mu, G.FINAL_TIME * k, seed, a.device)

        # trajectories of the three maps, keyed by method
        tracks = {}
        for method in ("mf", "mf_dp", "mf_2step"):
            tracks[method] = {}
            for k in a.ks:
                cloud = source
                for _ in range(k):
                    if method == "mf":
                        cloud = sample_two_param(mf_model, cloud, sigma, mu, steps=1)
                    elif method == "mf_2step":
                        cloud = sample_two_param(mf_model, cloud, sigma, mu, steps=2)
                    else:
                        cloud = sample_two_param(mf_model, cloud, sigma, mu, steps=1)
                        cloud = apply_corrector(corrector, cloud, sigma, mu)
                tracks[method][k] = cloud.numpy().astype(np.float64)
            log(f"  {method}: generated")

        for method, track in tracks.items():
            previous = None
            for k in a.ks:
                cloud = track[k]
                w2 = exact_w2(cloud[index], physical[k][index])
                step = None if previous is None else exact_w2(cloud[index], previous[index])
                rows.append({
                    "key": cell.key, "sigma": sigma, "mu": mu, "method": method, "k": k,
                    "w2_physical": w2,
                    "step_w2": step,
                    "invariants": invariants(cloud),
                    "forward_passes_per_application": 2 if method != "mf" else 1,
                })
                previous = cloud

        # physical step sizes, for the amplification factor
        physical_steps = {}
        for left, right in zip(a.ks[:-1], a.ks[1:]):
            physical_steps[right] = exact_w2(physical[left][index], physical[right][index])
        for row in rows:
            if row["key"] != cell.key or row["k"] not in physical_steps:
                continue
            if row["step_w2"] is None:
                continue
            row["physics_step"] = physical_steps[row["k"]]
            row["amplification"] = row["step_w2"] / max(physical_steps[row["k"]], 1e-12)

    floors = []
    for i, j in CELLS[: a.cells]:
        cell = G.Cell(i, j)
        one = simulate(initial, cell.sigma, cell.mu, G.FINAL_TIME,
                       8_800_000 + i * 10 + j, a.device)
        two = simulate(initial, cell.sigma, cell.mu, G.FINAL_TIME,
                       9_900_000 + i * 10 + j, a.device)
        floors.append({"key": cell.key, "floor": exact_w2(one[index], two[index])})
    floor_mean = float(np.mean([f["floor"] for f in floors]))

    payload = {"cells": CELLS[: a.cells], "ks": a.ks, "n": a.n, "seed": a.seed,
               "score_n": int(len(index)), "floor_mean": floor_mean,
               "floors": floors, "rows": rows}
    (OUT / f"rollout_compare_s{a.seed}.json").write_text(json.dumps(payload, indent=2),
                                                         encoding="utf-8")

    print(f"\nfloor = {floor_mean:.5f}   (n={len(index)}, exact OT)")
    print(f"{'cell':>8} {'method':>9} {'k':>3} {'W2/phys':>9} {'floor':>7} "
          f"{'step':>9} {'phys':>8} {'amp':>7} {'r_rms':>9}")
    for row in rows:
        step = row["step_w2"]
        amp = row.get("amplification")
        physics_step = row.get("physics_step")
        step_text = f"{step:.5f}" if step is not None else "-"
        physics_text = f"{physics_step:.5f}" if physics_step is not None else "-"
        amp_text = f"{amp:.2f}" if amp is not None else "-"
        print(f"{row['key']:>8} {row['method']:>9} {row['k']:3d} {row['w2_physical']:9.5f} "
              f"{row['w2_physical'] / floor_mean:7.2f} {step_text:>9} {physics_text:>8} "
              f"{amp_text:>7} {row['invariants']['radius_rms']:9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
