"""E6 and E8: is the corrector structural, or has it memorised one model's error pattern?

Two tests that use the SAME trained corrector (seed 88) and differ only in what it is applied
to. Neither requires training.

E6 -- cross-model transfer
    Apply the seed-88 corrector to the outputs of *other* base models:
      - seed-88 stage 1   (in distribution: the corrector was trained on this)
      - seed-89 stage 1   (same architecture, different seed)
      - seed-90 stage 1
      - width-112 stage 1 (different architecture, 26,547 parameters)
    If the gain is structural -- it repairs a class of error shared by maps of this kind --
    it should transfer. If it memorised seed 88's particular residual, it should not.

    Each base model is scored on its own held-out cells against the same reference, so the
    comparison is always "same base model, corrector on vs off". The absolute levels differ
    between base models, which is the point: what is compared is the *relative* gain.

E8 -- what the corrector does to an already-correct distribution
    Apply the corrector to the *reference cloud itself*. A corrector that knows when no
    correction is needed should leave it nearly alone. One that has memorised a displacement
    to apply unconditionally will push it away.

Usage:
    python scripts/e6_e8_structural_tests.py
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
from fullscale.train import load_cell, load_two  # noqa: E402
from fullscale.model import sample_two_param  # noqa: E402
from fullscale.evaluate import score_pair  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from rollout_compare import load_corrector  # noqa: E402

OUT = G.RESULTS / "structural_tests"


def base_models() -> dict[str, tuple[Path, str]]:
    """Label -> (checkpoint, the design it was trained on)."""
    return {
        "seed88": (G.RESULTS / "optimal_budget_ablation/two_32000/mf_two_param.pt", "s1"),
        "seed89": (G.RESULTS / "optimal_budget_ablation/two_32000_s89/mf_two_param.pt", "s1"),
        "seed90": (G.RESULTS / "optimal_budget_ablation/two_32000_s90/mf_two_param.pt", "s1"),
        "width112": (G.RESULTS / "capacity_matched/w112_s88/mf_two_param.pt", "s6"),
    }


def main() -> int:
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda text: print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)

    corrector = load_corrector(88, "cuda")
    device = next(corrector.parameters()).device
    slice_keys = {c.key for c in G.baseline_slice()}

    def correct(cloud: torch.Tensor, sigma: float, mu: float) -> torch.Tensor:
        x = cloud.to(device)
        cs = torch.full((len(x), 1), math.log10(sigma), dtype=torch.float32, device=device)
        cm = torch.full((len(x), 1), math.log10(mu), dtype=torch.float32, device=device)
        with torch.no_grad():
            return corrector(x, cs, cm).cpu()

    # ---------------- E6: transfer across base models ----------------
    rows = []
    for label, (checkpoint, design) in base_models().items():
        train_cells = frozenset(
            torch.load(checkpoint, map_location="cpu", weights_only=False)["cells"])
        # Score each base model on the cells IT has not seen, intersected with the cells the
        # seed-88 corrector's own base model (seed88) has seen, so the corrector is applied in
        # the regime it was trained for.
        seed88_cells = frozenset(
            torch.load(base_models()["seed88"][0], map_location="cpu",
                       weights_only=False)["cells"])
        held = [c for c in G.full_grid()
                if c.key not in train_cells and c.key not in slice_keys]
        overlap = [c for c in held if c.key in
                   {x.key for x in G.full_grid() if x.key not in seed88_cells
                    and x.key not in slice_keys}]
        cells = overlap if overlap else held
        log(f"=== E6 {label}: {len(cells)} cells "
            f"({'shared held-out' if overlap else 'own held-out'}) ===")

        model, _ = load_two(checkpoint, "cuda")
        before, after = [], []
        for position, cell in enumerate(cells, start=1):
            item = load_cell(cell)
            reference = item["clouds"]["reference"]
            mf = sample_two_param(model, torch.from_numpy(item["clouds"]["source"]),
                                  cell.sigma, cell.mu, steps=1)
            dp = correct(mf, cell.sigma, cell.mu)
            before.append(float(score_pair(reference, mf.numpy().astype(np.float32),
                                           G.BINS)["w2"]))
            after.append(float(score_pair(reference, dp.numpy().astype(np.float32),
                                          G.BINS)["w2"]))
            if position % 40 == 0:
                log(f"  {position}/{len(cells)}")
        mf_mean = float(np.mean(before))
        dp_mean = float(np.mean(after))
        rows.append({
            "base_model": label, "design": design, "n_cells": len(cells),
            "w2_without": mf_mean, "w2_with": dp_mean,
            "gain": float(1 - dp_mean / mf_mean),
            "wins": int(sum(1 for a, b in zip(before, after) if b < a)),
        })
        log(f"  gain {rows[-1]['gain'] * 100:+.1f}%  "
            f"({rows[-1]['wins']}/{len(cells)} cells improved)")

    # ---------------- E8: applying the corrector to the reference itself ----------------
    log("=== E8: corrector applied to the reference cloud ===")
    anchor = base_models()["seed88"][0]
    train_cells = frozenset(
        torch.load(anchor, map_location="cpu", weights_only=False)["cells"])
    held = [c for c in G.full_grid()
            if c.key not in train_cells and c.key not in slice_keys][:60]
    displacement, delta_w2 = [], []
    for cell in held:
        item = load_cell(cell)
        reference = item["clouds"]["reference"]
        moved = correct(torch.from_numpy(reference), cell.sigma, cell.mu)
        delta = (moved - torch.from_numpy(reference)).numpy()
        # mean displacement per particle, and the resulting change in W2 to the target
        displacement.append(float(np.sqrt((delta ** 2).sum(1)).mean()))
        delta_w2.append(float(score_pair(reference, moved.numpy().astype(np.float32),
                                         G.BINS)["w2"]))
    payload = {
        "e6_transfer": rows,
        "e8_reference_probe": {
            "n_cells": len(held),
            "mean_displacement": float(np.mean(displacement)),
            "w2_after_self_correction": float(np.mean(delta_w2)),
            "note": ("Applying the corrector to a perfect cloud should leave it almost "
                     "unchanged if the corrector knows when no correction is needed."),
        },
        "corrector": "dp_two_param/s88/dp_corrector.pt",
    }
    (OUT / "structural_tests.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nE6 cross-model transfer (seed-88 corrector applied to other base models):")
    print(f"  {'base model':<12} {'cells':>6} {'without':>9} {'with':>9} {'gain':>8} {'wins':>9}")
    for row in rows:
        print(f"  {row['base_model']:<12} {row['n_cells']:>6} {row['w2_without']:9.5f} "
              f"{row['w2_with']:9.5f} {row['gain'] * 100:+7.1f}% "
              f"{row['wins']:>4}/{row['n_cells']:<4}")

    print(f"\nE8 applied to the reference itself ({len(held)} cells):")
    print(f"  mean per-particle displacement : {np.mean(displacement):.5f}")
    print(f"  W2 to the true target after moving: {np.mean(delta_w2):.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
