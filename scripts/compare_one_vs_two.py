"""Fair head-to-head: one-parameter vs two-parameter MeanFlow.

The upstream baseline numbers (W2 ~ 0.35 at 100k particles) cannot be compared
directly with this study, because the particle count, the number of sigma levels
and the training budget all differ. So this script trains a *second* model under
exactly the same conditions as the two-parameter one — same data, same particle
count, same step budget, same evaluation grid — differing only in that it sees
sigma alone (trained on the mu = 1 slice) or both parameters (trained on the
LHS-20 subset).

Two comparisons are produced:

A. **Single-parameter control.** Trained on the 10 sigma values of the mu = 1
   slice. Scored on that same slice. This is the "what the old design achieves
   with the same budget" reference.

B. **LHS ablation (the real test of whether mu carries information).**
   Both models are scored on the *same held-out cells*. The two-parameter model
   sees (sigma, mu); the one-parameter model sees sigma alone. If mu matters,
   the two-parameter model must win on cells whose mu is far from 1, because
   for those cells the one-parameter model has no way to know mu at all.

Comparison B is the scientifically decisive one: it isolates the value of the
second parameter from every other difference.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.func import jvp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from twoparam import protocol as P  # noqa: E402
from twoparam.data import load_cell  # noqa: E402
from twoparam.evaluate import evaluate_cells, write_csv, write_json  # noqa: E402
from twoparam.model import TwoParamMeanflow, meanflow_loss, sample_times_original  # noqa: E402
from twoparam.protocol import Config  # noqa: E402


class OneParamMeanflow(nn.Module):
    """The upstream architecture, verbatim: 6 inputs, sigma as the only condition."""

    def __init__(self, width: int = 64, hidden_layers: int = 3, dimension: int = 3):
        super().__init__()
        self.width, self.hidden_layers, self.dimension = int(width), int(hidden_layers), int(dimension)
        layers: list[nn.Module] = [nn.Linear(dimension + 3, width), nn.SELU()]
        for _ in range(self.hidden_layers - 1):
            layers += [nn.Linear(width, width), nn.SELU()]
        layers.append(nn.Linear(width, dimension))
        self.net = nn.Sequential(*layers)

    def forward(self, z, r, t, logsigma):
        return self.net(torch.cat((z, t, r, logsigma), dim=-1))


def train_one_param(points, config, *, steps=1000, batch_size=256, lr=1e-3, seed=88, device="cuda"):
    """Train the sigma-only model on the mu = 1 slice."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    data = P.RESULTS / "data" / "full"
    initial = None
    targets = []
    for point in points:
        cell = load_cell(data, point.key)
        if initial is None:
            initial = cell["clouds"]["initial"]
        targets.append(cell["clouds"]["reference"])
    initial = torch.from_numpy(initial).to(dev)
    targets = torch.from_numpy(np.stack(targets)).to(dev)
    log_sigma = torch.tensor([math.log10(p.sigma) for p in points], dtype=torch.float32, device=dev)

    model = OneParamMeanflow().to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(1, steps + 1):
        env = torch.randint(len(points), (1,), device=dev).expand(batch_size)
        idx = torch.randint(config.n_train, (batch_size,), device=dev)
        x_target = targets[env, idx]
        x_source = initial[idx]
        r, t = sample_times_original(batch_size)
        r, t = r.to(dev), t.to(dev)
        velocity = x_source - x_target
        z = (1 - t) * x_target + t * x_source
        cond = log_sigma[env, None]
        u, derivative = jvp(lambda zz, rr, tt: model(zz, rr, tt, cond),
                            (z, r, t), (velocity, torch.zeros_like(r), torch.ones_like(t)))
        loss, _ = meanflow_loss(u - (velocity - (t - r) * derivative).detach())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"one-param loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.eval().requires_grad_(False)


@torch.no_grad()
def sample_one_param(model, source: np.ndarray, sigma: float, device="cuda") -> np.ndarray:
    dev = torch.device(device)
    x = torch.from_numpy(np.ascontiguousarray(source, dtype=np.float32)).to(dev)
    batch = len(x)
    r = torch.zeros(batch, 1, device=dev)
    t = torch.ones(batch, 1, device=dev)
    logsigma = torch.full((batch, 1), math.log10(float(sigma)), device=dev)
    return (x - model(x, r, t, logsigma)).cpu().numpy().astype(np.float32)


def main() -> int:
    config = Config(n_train=P.N_TRAIN, device="cuda")
    out = P.ensure_results("compare")
    slice_points = P.baseline_slice()
    full = P.full_grid()
    index = P.grid_index()

    log = lambda t: print(t, flush=True)
    log(f"training the single-parameter control on {len(slice_points)} sigma values (mu = 1)")
    one = train_one_param(slice_points, config, steps=1000)

    # ---- A: both models on the mu = 1 slice -------------------------------
    predictions, references = {}, {}
    for point in slice_points:
        cell = load_cell(P.RESULTS / "data" / "full", point.key)
        predictions[f"one:{point.key}"] = sample_one_param(one, cell["clouds"]["source"], point.sigma)
        references[f"one:{point.key}"] = cell["clouds"]["reference"]
    rows_a, _ = evaluate_cells(predictions, references, bins=config.bins, workers=4,
                               log=lambda t: None)
    a_summary = {
        "single_param_mean_w2": float(np.mean([r["w2"] for r in rows_a])),
        "single_param_max_w2": float(np.max([r["w2"] for r in rows_a])),
    }
    log(f"A. single-parameter control on the mu=1 slice: mean W2={a_summary['single_param_mean_w2']:.5f}")

    # ---- B: LHS ablation on held-out cells --------------------------------
    from twoparam.cli import _design
    from twoparam.train import load_model

    design = _design("lhs20", n_points=20)
    train_keys = {index[i, j].key for i, j in design.points}
    two = load_model(P.RESULTS / "train" / "lhs20" / "checkpoints" / "mf_two_param.pt")

    held = [p for p in full if p.key not in train_keys]
    log(f"B. ablation on {len(held)} held-out cells")
    pred_one, pred_two, ref = {}, {}, {}
    from twoparam.train import generate as generate_two

    for point in held:
        cell = load_cell(P.RESULTS / "data" / "full", point.key)
        pred_one[point.key] = sample_one_param(one, cell["clouds"]["source"], point.sigma)
        pred_two[point.key] = generate_two(two, cell["clouds"]["source"], point.sigma, point.mu)
        ref[point.key] = cell["clouds"]["reference"]

    rows_one, _ = evaluate_cells(pred_one, ref, bins=config.bins, workers=4, log=lambda t: None)
    rows_two, _ = evaluate_cells(pred_two, ref, bins=config.bins, workers=4, log=lambda t: None)
    by_one = {r["key"]: r["w2"] for r in rows_one}
    by_two = {r["key"]: r["w2"] for r in rows_two}

    detail = []
    for point in held:
        w1, w2 = by_one[point.key], by_two[point.key]
        detail.append({
            **point.meta(),
            "w2_one_param": w1,
            "w2_two_param": w2,
            "improvement": w1 - w2,
            "ratio": w2 / w1,
            "mu_off_baseline": abs(math.log10(point.mu) - math.log10(P.BASELINE_MU)),
        })
    detail.sort(key=lambda r: r["mu_off_baseline"])

    improvements = [r["improvement"] for r in detail]
    b_summary = {
        "n_heldout": len(detail),
        "one_param_mean_w2": float(np.mean([r["w2_one_param"] for r in detail])),
        "two_param_mean_w2": float(np.mean([r["w2_two_param"] for r in detail])),
        "mean_improvement": float(np.mean(improvements)),
        "median_improvement": float(np.median(improvements)),
        "two_param_wins": int(sum(1 for v in improvements if v > 0)),
        "one_param_wins": int(sum(1 for v in improvements if v < 0)),
        "mean_relative_reduction": float(np.mean([1 - r["ratio"] for r in detail])),
    }
    # Does the gain grow as mu moves away from the value the one-param model saw?
    off = np.array([r["mu_off_baseline"] for r in detail])
    imp = np.array(improvements)
    if off.std() > 0 and imp.std() > 0:
        b_summary["corr_improvement_vs_mu_distance"] = float(np.corrcoef(off, imp)[0, 1])
    else:
        b_summary["corr_improvement_vs_mu_distance"] = None

    log(f"B. one-param mean W2={b_summary['one_param_mean_w2']:.5f}  "
        f"two-param mean W2={b_summary['two_param_mean_w2']:.5f}")
    log(f"   two-param wins {b_summary['two_param_wins']}/{len(detail)} cells, "
        f"mean relative reduction {b_summary['mean_relative_reduction']:+.1%}")
    log(f"   correlation(gain, |log10 mu|) = {b_summary['corr_improvement_vs_mu_distance']}")

    write_csv(out / "ablation_detail.csv", detail)
    write_json(out / "comparison.json", {"A_single_param_slice": a_summary, "B_lhs_ablation": b_summary,
                                         "n_train_cells": len(train_keys)})
    log(f"\nwrote {out / 'ablation_detail.csv'}")
    log(f"wrote {out / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
