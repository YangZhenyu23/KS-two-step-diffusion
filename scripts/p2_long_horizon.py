"""P2: a long-horizon model, so the long-horizon question can be asked honestly.

Why iteration is not an answer
------------------------------
``rollout_stability.py`` shows the one-step generator diverges under repeated application:
W2 grows from 0.09-0.24 at k=1 to 4.5-82 at k=4 and 1,968-41,678 at k=8, with the RMS
radius exploding to ~27,000. The mechanism is distribution shift: after one application the
cloud sits somewhere mu_0 never put it, so the second application is out of distribution.

That experiment answers "can this map be iterated" (no) but not "does the approach work at
longer horizons" (untested). The only honest way to ask the latter is to **train a separate
model at the longer horizon**, which is what this script does.

Design
------
Same 40 cells (read from the headline checkpoint, so the design is identical), same
architecture, same budget per cell. Only the physical final time changes:

    T'  = 4 T = 0.08        (4x the headline horizon)

The target clouds are re-simulated at T' with the same shared initial cloud. Note this is
NOT the same as composing four Brownian paths of length T: the diffusion accumulates as
sqrt(4)*sqrt(2 mu T), and the advection accumulates 4*sigma*T, so the cloud is genuinely
different, and it is the physically correct T'=0.08 state.

Then the same three-way comparison is available at the new horizon:
    MF (1 step)  vs  MF + DP  vs  capacity-matched single net
and the question "is the second stage worth it at 4x the horizon" has a matched answer.

Cost
----
Data: 40 cells x 1 new simulation at T'=0.08. The simulator is O(N^2) in wall clock, so a
20k-particle run at T'=0.08 costs ~4x the T=0.02 run at the same N.
Train: 40 cells x 32,000 = 1.28M steps, ~2 h.

Usage:
    python scripts/p2_long_horizon.py --multiple 4 --n 20000
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
from fullscale.train import Config, train_two_param, load_two  # noqa: E402
from fullscale.model import sample_two_param  # noqa: E402
from fullscale.evaluate import score_many  # noqa: E402
from twoparam.data import unit_ball  # noqa: E402

OUT = G.RESULTS / "long_horizon"
BUDGET = 32000


def simulate(initial: np.ndarray, sigma: float, mu: float, final_time: float,
             noise_seed: int, device: str) -> np.ndarray:
    from training.protocol import simulate_original

    np.random.seed(noise_seed)
    torch.manual_seed(noise_seed)
    return simulate_original(
        torch.from_numpy(initial), float(sigma), mu=float(mu),
        chi=16 * math.pi, delta=0.1, t=float(final_time), dt=0.005,
        chunk=256, device=device,
    ).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--multiple", type=int, default=4, help="horizon multiplier: T' = k*T")
    ap.add_argument("--n", type=int, default=20000, help="particles for the new simulation")
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda text: print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)

    final_time = G.FINAL_TIME * a.multiple
    log(f"horizon T' = {final_time:.3f} ({a.multiple}x the headline {G.FINAL_TIME}), "
        f"N = {a.n:,}")

    headline = G.RESULTS / "optimal_budget_ablation" / "two_32000" / "mf_two_param.pt"
    payload = torch.load(headline, map_location="cpu", weights_only=False)
    train_keys = list(payload["cells"])
    cells = [G.Cell(int(k[1:3]), int(k[5:7])) for k in train_keys]
    slice_keys = {c.key for c in G.baseline_slice()}
    held = [c for c in G.full_grid() if c.key not in set(train_keys)
            and c.key not in slice_keys]
    log(f"same design as the headline model: {len(cells)} train, {len(held)} held out")

    # ---- data at the new horizon -------------------------------------------
    cloud_dir = OUT / f"clouds_T{a.multiple}" / "clouds"
    cloud_dir.mkdir(parents=True, exist_ok=True)
    initial = unit_ball(a.n, 88)

    needed = cells + held
    pending = [c for c in needed if not (cloud_dir / f"{c.key}.npz").exists()]
    log(f"simulating {len(pending)}/{len(needed)} cells at T'={final_time:.3f}")
    for position, cell in enumerate(pending, start=1):
        seed = 5_500_000 + 37 * (cell.i * 10 + cell.j) + 97 * a.multiple
        started = time.perf_counter()
        target = simulate(initial, cell.sigma, cell.mu, final_time, seed, a.device)
        np.savez_compressed(cloud_dir / f"{cell.key}.npz",
                            initial=initial.astype(np.float32),
                            source=initial.astype(np.float32),
                            reference=target.astype(np.float32))
        if position % 20 == 0 or position == len(pending):
            log(f"  {position}/{len(pending)} ({time.perf_counter() - started:.0f}s last)")

    # ---- train MeanFlow at the new horizon ---------------------------------
    # ``train_two_param`` reads the clouds through ``fullscale.grid.DATA``; we therefore
    # temporarily point the loader at the new directory via a thin shim.
    import fullscale.train as train_module
    import fullscale.grid as grid_module

    original_load = train_module.load_cell

    def load_from_new(cell):
        with np.load(cloud_dir / f"{cell.key}.npz") as values:
            clouds = {name: values[name].copy() for name in values.files if name != "meta"}
        return {"clouds": clouds, "meta": {}}

    train_module.load_cell = load_from_new
    work = G.ensure("long_horizon", f"mf_T{a.multiple}")
    checkpoint = work / "mf_two_param.pt"
    config = Config(n_train=a.n, device=a.device)
    if checkpoint.exists():
        log("reusing existing long-horizon checkpoint")
        model, _ = load_two(checkpoint, a.device)
    else:
        log(f"training MeanFlow at T'={final_time:.3f}: "
            f"{len(cells)} cells x {a.budget} = {a.budget * len(cells):,} steps")
        result = train_two_param(cells, config, work, steps=a.budget * len(cells),
                                 batch_size=4096, seed=88)
        model, _ = load_two(checkpoint, a.device)
        log(f"  done in {result['seconds']:.0f}s")
    train_module.load_cell = original_load

    # ---- score --------------------------------------------------------------
    def score(predict, cell_list, label) -> float:
        pairs = {}
        for cell in cell_list:
            item = np.load(cloud_dir / f"{cell.key}.npz")
            pairs[cell.key] = (item["reference"],
                               predict(torch.from_numpy(item["source"]), cell))
        rows = score_many(pairs, bins=G.BINS, workers=4, log=lambda t: None)
        values = [r["w2"] for r in rows]
        log(f"  {label}: {np.mean(values):.5f}  (n={len(values)})")
        return float(np.mean(values))

    log("scoring")
    mf_held = score(lambda src, cell: sample_two_param(model, src, cell.sigma, cell.mu,
                                                       steps=1).numpy().astype(np.float32),
                    held, "MF 1-step, held out")
    mf_train = score(lambda src, cell: sample_two_param(model, src, cell.sigma, cell.mu,
                                                        steps=1).numpy().astype(np.float32),
                     cells, "MF 1-step, training fit")
    two_step = score(lambda src, cell: sample_two_param(model, src, cell.sigma, cell.mu,
                                                        steps=2).numpy().astype(np.float32),
                     held, "MF 2-step, held out")

    summary = {
        "horizon": final_time, "multiple": a.multiple, "n_particles": a.n,
        "budget_per_cell": a.budget, "n_train": len(cells), "n_held_out": len(held),
        "mf_held_out": mf_held, "mf_train_fit": mf_train, "mf_2step_held_out": two_step,
        "over_floor_16": mf_held / G.FLOOR_16,
        "note": ("The headline horizon is T=0.02. This run retrains at k*T so the "
                 "long-horizon question is answered by a model trained for it, not by "
                 "iterating the short-horizon map (which diverges)."),
    }
    (OUT / f"long_horizon_T{a.multiple}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nhorizon {a.multiple}xT:")
    print(f"  MF training fit   {mf_train:.5f}")
    print(f"  MF held-out       {mf_held:.5f}  ({mf_held / G.FLOOR_16:.2f}x 16-bin floor)")
    print(f"  MF 2-step held-out {two_step:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
