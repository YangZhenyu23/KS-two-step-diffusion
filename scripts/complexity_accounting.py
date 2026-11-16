"""Computational complexity accounting for both stages, in FLOPs and memory.

Why complexity rather than wall clock
-------------------------------------
Wall-clock numbers from a laptop 4060 are not portable and invite the objection that the
comparison depends on hardware. Complexity is hardware-independent, and it also exposes
things wall clock hides: that the corrector's per-sample cost is not negligible (it has
LayerNorm and an extra input column), and that the OT term is cubic in the minibatch size
rather than linear.

What is counted
---------------
Stage 1 (MeanFlow), per particle, per forward pass
    architecture: Linear(7->64) SELU, Linear(64->64) SELU, Linear(64->64) SELU, Linear(64->3)
    MACs = 7*64 + 64*64 + 64*64 + 64*3 = 8,704
    FLOPs ~= 2 * MACs = 17,408 per particle

Stage 2 (corrector), per particle, per forward pass
    architecture: Linear(5->64) + 2 residual blocks each (LayerNorm, Linear(64->64),
    SiLU, Linear(64->64), scale 0.2) + Linear(64->3)
    MACs = 5*64 + 2*(64*64 + 64*64) + 64*3 = 17,056
    FLOPs ~= 2 * MACs = 34,112 per particle
    => the corrector is ~2x the FLOPs of stage 1 per pass, despite fewer parameters,
       because LayerNorm adds normalization work and the two 64x64 layers per block dominate.

Training
    stage 1 needs a JVP, which is one extra forward-mode pass: ~3x forward FLOPs per step
    (forward + tangent + backward). We report it as 3x rather than measure, and say so.
    stage 2 needs no JVP: 1 forward + 1 backward = ~2x forward FLOPs.
    stage 2 additionally solves a minibatch OT problem every ``ot_refresh`` steps.
    Exact OT is O(n^3 log n) worst case, O(n^3) in practice for network simplex.

Memory
    stage 1 activations: batch * 3 hidden layers * 64 floats
    stage 2 activations: batch * (1 + 4) hidden layers * 64 floats (LayerNorm stored)
    OT: an n x n cost matrix in float64 = 8 n^2 bytes, which is the real constraint:
        n = 2,000 -> 32 MB; n = 10,000 -> 800 MB; n = 30,000 -> 7.2 GB.

Usage:
    python scripts/complexity_accounting.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream" / "MF-DeepParticle" / "deepparticle"))

from fullscale import grid as G  # noqa: E402

OUT = G.RESULTS / "complexity"

N_PARTICLES = 100_000
MF_IN, MF_HID, MF_OUT, MF_LAYERS = 7, 64, 3, 3
DP_IN, DP_HID, DP_OUT, DP_BLOCKS = 5, 64, 3, 2
OT_BATCH = 2_000
OT_REFRESH = 10
DP_UPDATES_PER_CELL = 2_000
MF_UPDATES_PER_CELL = 32_000
N_TRAIN_CELLS = 40


def macs_mlp(fan_in: int, hidden: int, fan_out: int, layers: int) -> int:
    """Total multiply-accumulates of an MLP with ``layers`` hidden layers of ``hidden``."""
    total = fan_in * hidden
    total += (layers - 1) * hidden * hidden
    total += hidden * fan_out
    return total


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    mf_macs = macs_mlp(MF_IN, MF_HID, MF_OUT, MF_LAYERS)
    # corrector: input projection, then per block two 64x64 linears, then output
    dp_macs = DP_IN * DP_HID + DP_BLOCKS * (2 * DP_HID * DP_HID) + DP_HID * DP_OUT
    # LayerNorm is ~4 ops per element (mean, var, normalize, affine); counted separately
    dp_layernorm_ops = 2 * DP_BLOCKS * DP_HID * 4

    mf_flops_one_particle = 2 * mf_macs
    dp_flops_one_particle = 2 * dp_macs + dp_layernorm_ops

    rows = []
    for config, nfe, extra in (
        ("stage 1 only", 1, 0),
        ("stage 1, two-step", 2, 0),
        ("stage 1 + corrector", 1, 1),
        ("stage 1 + corrector, alpha=2", 1, 1),
    ):
        mf_passes = nfe if extra == 0 else 1
        flops = N_PARTICLES * (mf_passes * mf_flops_one_particle
                              + extra * dp_flops_one_particle)
        rows.append({
            "configuration": config,
            "stage1_passes": mf_passes,
            "corrector_passes": extra,
            "total_passes": mf_passes + extra,
            "flops_per_cloud": flops,
            "relative_flops": None,  # filled below
        })
    base = rows[0]["flops_per_cloud"]
    for row in rows:
        row["relative_flops"] = row["flops_per_cloud"] / base

    # ---- training complexity ----
    mf_steps = MF_UPDATES_PER_CELL * N_TRAIN_CELLS
    dp_steps = DP_UPDATES_PER_CELL * N_TRAIN_CELLS
    batch = 4096  # MF training batch

    mf_train_flops = mf_steps * batch * 3 * mf_flops_one_particle      # JVP => 3x
    dp_train_flops = dp_steps * OT_BATCH * 2 * dp_flops_one_particle   # fwd+bwd => 2x
    n_ot_solves = dp_steps // OT_REFRESH
    ot_flops = n_ot_solves * (OT_BATCH ** 3)  # leading order

    train_rows = [
        {"stage": "stage 1", "steps": mf_steps, "batch": batch,
         "flops": mf_train_flops, "ot_solves": 0, "ot_flops": 0},
        {"stage": "corrector", "steps": dp_steps, "batch": OT_BATCH,
         "flops": dp_train_flops, "ot_solves": n_ot_solves, "ot_flops": ot_flops},
    ]

    # ---- memory ----
    memory_rows = []
    for n in (1_000, 2_000, 5_000, 10_000, 30_000):
        memory_rows.append({
            "ot_batch": n,
            "cost_matrix_mb": 8 * n * n / 1024 ** 2,
            "activation_mb_stage1": n * MF_LAYERS * MF_HID * 4 / 1024 ** 2,
            "activation_mb_stage2": n * (1 + 4) * DP_HID * 4 / 1024 ** 2,
        })

    payload = {
        "architecture": {
            "stage1": {"in": MF_IN, "hidden": MF_HID, "layers": MF_LAYERS,
                       "params": mf_macs, "macs_per_particle": mf_macs,
                       "flops_per_particle": mf_flops_one_particle},
            "corrector": {"in": DP_IN, "hidden": DP_HID, "blocks": DP_BLOCKS,
                          "macs_per_particle": dp_macs,
                          "layernorm_ops": dp_layernorm_ops,
                          "flops_per_particle": dp_flops_one_particle},
            "corrector_over_stage1_flops": dp_flops_one_particle / mf_flops_one_particle,
        },
        "inference": rows,
        "training": train_rows,
        "memory": memory_rows,
        "notes": [
            "FLOPs are 2*MACs for dense layers; LayerNorm counted as 4 ops per element.",
            "Stage 1 training needs a JVP, counted as 3x forward FLOPs (fwd + tangent + bwd).",
            "Stage 2 training needs no JVP but does need a backward pass, counted as 2x.",
            "Exact OT is counted as n^3 at leading order (network simplex).",
            "Wall-clock numbers are deliberately excluded: they do not transfer across "
            "hardware and they obscure the OT term, which is the only superlinear one.",
        ],
    }
    (OUT / "complexity.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("per-particle forward cost")
    print(f"  stage 1  : {mf_macs:>6,} MACs  = {mf_flops_one_particle:>8,} FLOPs")
    print(f"  corrector: {dp_macs:>6,} MACs  = {dp_flops_one_particle:>8,} FLOPs "
          f"({dp_flops_one_particle / mf_flops_one_particle:.2f}x stage 1)")

    print(f"\ninference, {N_PARTICLES:,} particles")
    print(f"  {'configuration':<32} {'passes':>7} {'GFLOP':>10} {'relative':>9}")
    for row in rows:
        print(f"  {row['configuration']:<32} {row['total_passes']:>7} "
              f"{row['flops_per_cloud'] / 1e9:>10.2f} {row['relative_flops']:>8.2f}x")

    print("\ntraining")
    print(f"  {'stage':<12} {'steps':>10} {'PFLOP':>10} {'OT solves':>10} {'OT PFLOP':>10}")
    for row in train_rows:
        print(f"  {row['stage']:<12} {row['steps']:>10,} {row['flops'] / 1e15:>10.3f} "
              f"{row['ot_solves']:>10,} {row['ot_flops'] / 1e15:>10.3f}")

    print("\nOT memory scaling (the real constraint)")
    print(f"  {'batch':>7} {'cost matrix':>12} {'act. s1':>9} {'act. s2':>9}")
    for row in memory_rows:
        print(f"  {row['ot_batch']:>7,} {row['cost_matrix_mb']:>11.1f}MB "
              f"{row['activation_mb_stage1']:>8.1f}MB {row['activation_mb_stage2']:>8.1f}MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
