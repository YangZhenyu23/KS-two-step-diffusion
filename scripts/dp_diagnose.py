"""Diagnose the DP corrector: why does refinement make the clouds worse?

The first DP run produced W2 = 0.524 versus 0.156 for MeanFlow alone, a factor of
3.4 worse on every one of the 144 held-out cells. That is not a weak improvement,
it is a broken configuration, and the cause has to be found before any claim is
made about DP.

Three candidates, each measured rather than guessed:

H1 **Corrected displacement too large.** ``alpha = 10`` multiplies the correction.
   If the corrector learns a unit-magnitude output, the applied displacement is 10x
   the cloud's own scale. Measured: ``||x_dp - x_mf||`` relative to ``||x_mf||``.

H2 **Undertrained corrector.** 60 updates per cell may be far too few for the
   alpha-scaled parameterisation to settle. Measured: training loss trajectory and
   the effect of a tenfold budget.

H3 **The matching is stale.** ``ot_refresh = 50`` with 2400 total steps means only
   48 matchings; if the corrector moves fast, the cached permutation becomes wrong.
   Measured: loss right after each refresh versus right before the next one.

The script prints all three diagnostics and writes ``diagnosis.json``.

Usage:
    python scripts/dp_diagnose.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fullscale import grid as G  # noqa: E402
from fullscale.evaluate import score_pair  # noqa: E402
from fullscale.lhs import latin_hypercube  # noqa: E402
from fullscale.model import sample_two_param  # noqa: E402
from fullscale.train import load_cell, load_two  # noqa: E402

from dp_refine import apply_corrector, load_corrector  # noqa: E402


def main() -> int:
    print("H1: magnitude of the applied correction")
    mf, _ = load_two(G.RESULTS / "train" / "lhs40" / "mf_two_param.pt")
    corrector = load_corrector(G.RESULTS / "dp" / "dp_corrector.pt")
    payload = json.loads((G.RESULTS / "dp" / "training_log.json").read_text(encoding="utf-8"))
    alpha = payload["alpha"]
    print(f"  alpha = {alpha}")

    design = latin_hypercube(40)
    lhs_cells = [G.Cell(i, j) for i, j in design.points]
    print(f"\n  {'cell':<9} {'|x_mf| rms':>11} {'|delta| rms':>12} {'ratio':>7} {'W2 mf':>8} {'W2 dp':>8}")
    ratios, worsenings = [], []
    for cell in lhs_cells[:8]:
        data = load_cell(cell)
        source = torch.from_numpy(data["clouds"]["source"])
        reference = data["clouds"]["reference"]
        x_mf = sample_two_param(mf, source, cell.sigma, cell.mu)
        x_dp = apply_corrector(corrector, x_mf, cell.sigma, cell.mu)
        delta = (x_dp - x_mf).numpy()
        scale_mf = float(np.sqrt((x_mf.numpy() ** 2).sum(-1).mean()))
        scale_d = float(np.sqrt((delta ** 2).sum(-1).mean()))
        w_mf = score_pair(reference, x_mf.numpy().astype(np.float32), G.BINS)["w2"]
        w_dp = score_pair(reference, x_dp.numpy().astype(np.float32), G.BINS)["w2"]
        ratio = scale_d / scale_mf if scale_mf else float("nan")
        ratios.append(ratio)
        worsenings.append(w_dp - w_mf)
        print(f"  {cell.key:<9} {scale_mf:11.4f} {scale_d:12.4f} {ratio:7.3f} {w_mf:8.4f} {w_dp:8.4f}")
    print(f"\n  mean |delta| / |x_mf| = {np.mean(ratios):.3f}")
    print(f"  mean W2 increase       = {np.mean(worsenings):+.4f}")
    print("  interpretation: a corrector meant to be a *near-identity* map should")
    print("  have this ratio well below 1. A ratio near or above 1 means the")
    print("  correction is as large as the cloud itself, i.e. alpha is too big for")
    print("  the residual this MeanFlow already achieves.")

    print("\nH2: training loss trajectory (are the weights still improving?)")
    rows = payload["rows"]
    for r in rows[:4] + rows[-4:]:
        print(f"  step {r['step']:5d}  loss={r['loss']:8.4f}  transport={r['transport']:8.4f}  "
              f"residual={r['residual']:8.4f}")
    losses = [r["loss"] for r in rows]
    print(f"  first={losses[0]:.4f}  last={losses[-1]:.4f}  "
          f"min={min(losses):.4f}  (a loss that oscillates without a downward trend "
          f"means the budget or the step size is wrong)")

    print("\nH3: is the cached matching still valid?")
    refreshed = [r for r in rows if r["ot_refreshed"]]
    stale = [r for r in rows if r["coupling_age"] >= payload["ot_refresh"] - 1]
    if refreshed and stale:
        print(f"  loss right after refresh: {np.mean([r['loss'] for r in refreshed]):.4f}")
        print(f"  loss just before refresh: {np.mean([r['loss'] for r in stale]):.4f}")
        print("  a large gap means the corrector drifts away from the cached matching")

    out = G.ensure("dp")
    (out / "diagnosis.json").write_text(json.dumps({
        "alpha": alpha,
        "mean_correction_ratio": float(np.mean(ratios)),
        "mean_w2_increase": float(np.mean(worsenings)),
        "loss_first": losses[0], "loss_last": losses[-1], "loss_min": min(losses),
        "ot_refresh": payload["ot_refresh"],
        "updates_per_cell": payload["updates_per_cell"],
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out / 'diagnosis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
