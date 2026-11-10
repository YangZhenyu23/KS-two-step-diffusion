"""Where is the time actually going? A micro-benchmark of the two workloads.

Two very different operations dominate this project, and they scale differently
with hardware:

**Simulation** (`simulate_original`): O(N^2) pairwise kernel, batched as
``chunk x N`` matmuls in float64, dominated by memory traffic and FP64 throughput.

**Training** (MeanFlow with ``jvp``): per step it does a forward pass plus a
forward-mode derivative, on a batch of particles, float32. Dominated by FP32
matrix-multiply throughput and kernel launch overhead.

**Exact OT** (`ot.emd2`): a network simplex on a dense cost matrix, single-threaded
C++. This is CPU-bound and completely unaffected by the GPU.

Measuring all three separately tells us whether a faster GPU would help, and by how
much. If the wall clock is dominated by the CPU-side OT solve, no GPU upgrade helps.

Usage:
    python scripts/hardware_probe.py
"""
from __future__ import annotations

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
from twoparam.data import unit_ball  # noqa: E402

OUT = G.RESULTS / "hardware"


def bench_simulation(n: int, repeats: int, device="cuda") -> dict:
    from training.protocol import simulate_original

    initial = unit_ball(n, 88)
    torch.cuda.synchronize()
    times = []
    for k in range(repeats):
        np.random.seed(100 + k)
        torch.manual_seed(100 + k)
        torch.cuda.synchronize()
        started = time.perf_counter()
        simulate_original(torch.from_numpy(initial), 100.0, mu=1.0, chi=16 * math.pi,
                          delta=0.1, t=0.02, dt=0.005, chunk=256, device=device)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
    return {"n": n, "repeats": repeats, "times": times,
            "mean": float(np.mean(times)), "min": float(np.min(times))}


def bench_training(steps: int, batch: int, device="cuda") -> dict:
    """One MeanFlow step: forward + jvp + backward, the actual training cost."""
    from torch.func import jvp
    from fullscale.model import TwoParamMeanflow, meanflow_loss, sample_times

    model = TwoParamMeanflow(width=64, hidden_layers=3, dimension=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    z = torch.randn(batch, 3, device=device)
    r = torch.rand(batch, 1, device=device)
    t = torch.ones(batch, 1, device=device)
    ls = torch.full((batch, 1), 1.7, device=device)
    lm = torch.zeros(batch, 1, device=device)
    vel = torch.randn(batch, 3, device=device)

    # warm-up
    for _ in range(5):
        u, d = jvp(lambda zz, rr, tt: model(zz, rr, tt, ls, lm), (z, r, t),
                   (vel, torch.zeros_like(r), torch.ones_like(t)))
        loss, _ = meanflow_loss(u - vel)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    started = time.perf_counter()
    for _ in range(steps):
        u, d = jvp(lambda zz, rr, tt: model(zz, rr, tt, ls, lm), (z, r, t),
                   (vel, torch.zeros_like(r), torch.ones_like(t)))
        loss, _ = meanflow_loss(u - vel)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    total = time.perf_counter() - started
    return {"steps": steps, "batch": batch, "seconds": total,
            "ms_per_step": 1000 * total / steps,
            "steps_per_second": steps / total}


def bench_ot(repeats: int) -> dict:
    """Exact OT on a dense cost matrix: pure CPU work."""
    import ot
    from scipy.spatial.distance import cdist

    times = []
    for n in (1000, 3000):
        points = np.random.default_rng(0).standard_normal((n, 3))
        w = np.full(n, 1.0 / n)
        pair_times = []
        for _ in range(repeats):
            cost = cdist(points, points, metric="sqeuclidean")
            started = time.perf_counter()
            ot.emd2(w, w, cost, numItermax=5_000_000)
            pair_times.append(time.perf_counter() - started)
        times.append({"n": n, "mean": float(np.mean(pair_times)),
                      "min": float(np.min(pair_times))})
    return {"cost_matrix_included": False, "by_size": times}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, torch.get_num_threads()))

    print("=" * 66)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"  SMs {props.multi_processor_count}, "
          f"memory {props.total_memory / 2**30:.1f} GiB, "
          f"cc {props.major}.{props.minor}")
    print("=" * 66)

    print("\n[1] simulation (O(N^2) float64 kernel)")
    sim = {}
    for n in (100_000,):
        r = bench_simulation(n, repeats=3)
        sim[str(n)] = r
        print(f"  N={n}: mean {r['mean']:.1f}s  min {r['min']:.1f}s")

    print("\n[2] training step (forward + jvp + backward, float32)")
    tr = bench_training(steps=200, batch=4096)
    print(f"  batch {tr['batch']}: {tr['ms_per_step']:.1f} ms/step, "
          f"{tr['steps_per_second']:.0f} steps/s")

    print("\n[3] exact OT (network simplex, CPU only)")
    ot_b = bench_ot(repeats=3)
    for entry in ot_b["by_size"]:
        print(f"  n={entry['n']}: {entry['mean'] * 1000:.0f} ms per solve")

    # ---- how much of a real experiment is each part? ---------------------
    print("\n[4] budget of one evaluation cell")
    sim_time = sim["100000"]["mean"]
    ot_3000 = [e for e in ot_b["by_size"] if e["n"] == 3000][0]["mean"]
    sim_per_cell = sim_time
    score_per_cell = 2 * ot_3000 + 0.6   # two solves plus histograms and IO
    print(f"  simulation   {sim_per_cell:6.1f} s  ({sim_per_cell / (sim_per_cell + score_per_cell):.0%})")
    print(f"  scoring      {score_per_cell:6.1f} s  ({score_per_cell / (sim_per_cell + score_per_cell):.0%})")
    print(f"  total/cell   {sim_per_cell + score_per_cell:6.1f} s")

    # ---- what a faster GPU would buy -------------------------------------
    # the 4060 laptop has ~15 TFLOPS FP32; a 4080 desktop ~49, roughly 3.2x.
    # Simulation is FP64-heavy and memory-bound, so it scales worse than the
    # nominal ratio; training is FP32-matmul-bound and scales closer to nominal.
    speedups = {
        "rtx_4080_desktop": {"flop_ratio": 3.2, "sim_eff": 0.55, "train_eff": 0.85},
        "rtx_4090_desktop": {"flop_ratio": 5.5, "sim_eff": 0.55, "train_eff": 0.85},
    }
    print("\n[5] projected wall clock for a 20-cell particle-level evaluation")
    base_cells = 20
    base_total = base_cells * (sim_per_cell + score_per_cell)
    print(f"  4060 laptop (measured): {base_total / 60:.1f} min")
    for name, s in speedups.items():
        sim_new = sim_per_cell / (1 + (s["flop_ratio"] - 1) * s["sim_eff"])
        score_new = score_per_cell / (1 + (s["flop_ratio"] - 1) * 0.02)
        new_total = base_cells * (sim_new + score_new)
        print(f"  {name}: {new_total / 60:.1f} min  "
              f"(sim {sim_new:.1f}s, score {score_new:.1f}s)")

    report = {"gpu": torch.cuda.get_device_name(0),
              "sm_count": props.multi_processor_count,
              "simulation": sim, "training": tr, "exact_ot": ot_b,
              "per_cell": {"simulation_s": sim_per_cell, "scoring_s": score_per_cell}}
    (OUT / "hardware_probe.json").write_text(json.dumps(report, indent=2),
                                             encoding="utf-8")
    print(f"\nwrote {OUT / 'hardware_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
