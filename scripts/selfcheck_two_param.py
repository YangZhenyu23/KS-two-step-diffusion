"""Self-check: does one training step actually reduce error in the right direction?

This is a smoke test, not a result. It trains a few hundred steps on two
neighbouring cells and asserts that

  1. the sampler ``x - u(x, r=0, t=1)`` applied to the *initial* cloud lands
     closer to the physical target than the initial cloud itself, and
  2. mean W2 drops below the initial-cloud baseline for both cells.

If the endpoint convention in training were flipped, (1) would fail loudly
here instead of silently producing a plausible-looking but wrong W2 table.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from twoparam import protocol as P  # noqa: E402
from twoparam.data import load_cell  # noqa: E402
from twoparam.evaluate import _edges, measure, solve  # noqa: E402
from twoparam.model import TwoParamMeanflow, mf_sample  # noqa: E402
from twoparam.protocol import Config  # noqa: E402
from twoparam.train import train  # noqa: E402


def w2_of(pred: np.ndarray, ref: np.ndarray, bins: int = 16) -> float:
    edges = _edges([ref, pred], bins)
    return solve(measure(ref, edges, bins), measure(pred, edges, bins))["w2"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--cells", nargs="+", default=["s04_m05", "s05_m05"])
    ap.add_argument("--bins", type=int, default=16)
    a = ap.parse_args()

    config = Config(n_train=P.N_TRAIN, device="cuda")
    data = P.RESULTS / "data" / "full"
    index = P.grid_index()
    points = []
    for key in a.cells:
        i, j = int(key[1:3]), int(key[5:7])
        points.append(index[i, j])

    out = P.RESULTS / "selfcheck"
    summary = train(data, out, points, config, steps=a.steps, batch_size=256, log=lambda t: None)
    print(f"trained {a.steps} steps  final_loss={summary['final_loss']:.6g}")
    print(f"cells: {summary['cells']}")

    import torch
    model = TwoParamMeanflow().to("cuda")
    payload = torch.load(out / "checkpoints" / "mf_two_param.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"])
    model.eval().requires_grad_(False)

    failures = []
    print(f"\n{'cell':<10} {'W2(init->tgt)':>14} {'W2(pred->tgt)':>14} {'ratio':>7}")
    for point in points:
        cell = load_cell(data, point.key)
        initial = cell["clouds"]["initial"]
        reference = cell["clouds"]["reference"]
        pred = mf_sample(model, torch.from_numpy(initial), point.sigma, point.mu).numpy()

        baseline = w2_of(initial, reference, a.bins)
        achieved = w2_of(pred, reference, a.bins)
        ratio = achieved / baseline
        print(f"{point.key:<10} {baseline:14.5f} {achieved:14.5f} {ratio:7.3f}")
        if not achieved < baseline:
            failures.append(f"{point.key}: sampler did not beat the initial cloud "
                            f"({achieved:.5f} >= {baseline:.5f})")

    if failures:
        print("\nFAIL")
        for line in failures:
            print("  " + line)
        return 1
    print("\nPASS: the sampler improves on the initial cloud in every checked cell")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
