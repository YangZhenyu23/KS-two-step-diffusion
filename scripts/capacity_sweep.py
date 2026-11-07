"""Does network capacity, not data volume, limit the current error?

Motivation
----------
Every error in this study sits at roughly 2.3x the measurement floor (0.0644),
and the learning curve is flat from 80 to 160 training cells
(docs/SAMPLE_EFFICIENCY.md). Both facts point at the model, not the data. This
script tests that directly by sweeping capacity while holding everything else
fixed:

    width     64 -> 128 -> 256
    depth     3  -> 5   -> 7 hidden layers
    steps     2000 per cell in every run

If error falls as capacity grows at fixed data, the bottleneck is the model. If
it stays put, the representation itself is the limit (for example the plain
concatenation of log-sigma and log-mu, or the absence of a Fourier feature).

Architecture note
-----------------
The paper's own description of the MeanFlow network mentions sinusoidal
embeddings of (r, t, sigma), light skip connections and a small-gain linear head,
while the released code uses plain scalar concatenation and three SELU layers
(docs/PAPER_COMPARISON.md 5). This script therefore reports two families:

  ``mlp``       plain concatenation, the released architecture
  ``fourier``   sinusoidal features of (r, t, log-sigma, log-mu), plus skips

so a capacity result can be separated from an architecture-change result.

Usage:
    python scripts/capacity_sweep.py --configs mlp64 mlp128 mlp256 fourier128
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.func import jvp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fullscale import grid as G  # noqa: E402
from fullscale.grid import Cell, Config  # noqa: E402
from fullscale.model import sample_times  # noqa: E402
from fullscale.train import load_training_tensors  # noqa: E402

from sample_efficiency import (  # noqa: E402
    OUT, frozen_test_set, score_on, stratified_subset, write_csv, write_json, log,
)

UPDATES_PER_CELL = 2000


class DenseMeanflow(nn.Module):
    """MeanFlow with configurable width and depth, plain conditioning."""

    def __init__(self, width=64, hidden=3, dimension=3):
        super().__init__()
        self.width, self.hidden, self.dimension = width, hidden, dimension
        self.kind = "mlp"
        layers: list[nn.Module] = [nn.Linear(dimension + 4, width), nn.SELU()]
        for _ in range(hidden - 1):
            layers += [nn.Linear(width, width), nn.SELU()]
        layers.append(nn.Linear(width, dimension))
        self.net = nn.Sequential(*layers)

    def forward(self, z, r, t, logsigma, logmu):
        return self.net(torch.cat((z, t, r, logsigma, logmu), dim=-1))

    def describe(self) -> dict:
        return {"kind": "mlp", "width": self.width, "hidden_layers": self.hidden,
                "params": sum(p.numel() for p in self.parameters())}


class FourierMeanflow(nn.Module):
    """MeanFlow with sinusoidal condition features and light input skips.

    The condition vector (r, t, log-sigma, log-mu) is lifted to
    ``[sin(2^k pi c), cos(2^k pi c)]`` over a small octave range, which lets a
    small network represent sharp dependence on the parameters. The raw condition
    is concatenated as well so low-frequency behaviour is not lost.
    """

    def __init__(self, width=64, hidden=3, dimension=3, octaves=4):
        super().__init__()
        self.width, self.hidden, self.dimension = width, hidden, dimension
        self.octaves = octaves
        self.kind = "fourier"
        n_feat = 4 * (2 * octaves + 1)
        self.lift = nn.Parameter(
            (2.0 ** torch.arange(octaves)).float().view(1, octaves) * math.pi,
            requires_grad=False)
        self.input = nn.Linear(dimension + n_feat, width)
        self.blocks = nn.ModuleList(nn.Linear(width, width) for _ in range(hidden))
        self.head = nn.Linear(width, dimension)
        # small-gain initialisation on the last layer
        nn.init.normal_(self.head.weight, std=1e-2)
        nn.init.zeros_(self.head.bias)

    def features(self, r, t, logsigma, logmu):
        cond = torch.cat((r, t, logsigma, logmu), dim=-1)      # (n, 4)
        scaled = cond.unsqueeze(-1) * self.lift                # (n, 4, octaves)
        feats = [cond]
        feats.append(torch.sin(scaled).flatten(1))
        feats.append(torch.cos(scaled).flatten(1))
        return torch.cat(feats, dim=-1)

    def forward(self, z, r, t, logsigma, logmu):
        h = self.input(torch.cat((z, self.features(r, t, logsigma, logmu)), dim=-1))
        h = torch.nn.functional.selu(h)
        skip = h
        for index, block in enumerate(self.blocks):
            h = torch.nn.functional.selu(block(h))
            if index % 2 == 1:                 # light skip connections
                h = h + skip
                skip = h
        return self.head(h)

    def describe(self) -> dict:
        return {"kind": "fourier", "width": self.width, "hidden_layers": self.hidden,
                "octaves": self.octaves,
                "params": sum(p.numel() for p in self.parameters())}


def build(name: str):
    table = {
        "mlp64": lambda: DenseMeanflow(width=64, hidden=3),
        "mlp128": lambda: DenseMeanflow(width=128, hidden=3),
        "mlp256": lambda: DenseMeanflow(width=256, hidden=3),
        "mlp512": lambda: DenseMeanflow(width=512, hidden=4),
        "fourier128": lambda: FourierMeanflow(width=128, hidden=4),
        "fourier256": lambda: FourierMeanflow(width=256, hidden=4),
    }
    if name not in table:
        raise ValueError(f"unknown config {name}")
    return table[name]()


def train_variant(model, cells: list[Cell], config: Config, *,
                  steps: int, batch_size: int = 4096, lr: float = 1e-3, seed: int = 88):
    device = torch.device("cuda")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(device)
    initial, targets = load_training_tensors(cells, device)
    log_sigma = torch.tensor([math.log10(c.sigma) for c in cells], dtype=torch.float32, device=device)
    log_mu = torch.tensor([math.log10(c.mu) for c in cells], dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(1, steps + 1):
        env = torch.randint(len(cells), (1,), device=device).expand(batch_size)
        idx = torch.randint(config.n_train, (batch_size,), device=device)
        x_target = targets[env, idx]
        x_source = initial[idx]
        r, t = sample_times(batch_size)
        r, t = r.to(device), t.to(device)
        velocity = x_source - x_target
        z = (1 - t) * x_target + t * x_source
        cond_sigma = log_sigma[env, None]
        cond_mu = log_mu[env, None]
        u, derivative = jvp(lambda zz, rr, tt: model(zz, rr, tt, cond_sigma, cond_mu),
                            (z, r, t),
                            (velocity, torch.zeros_like(r), torch.ones_like(t)))
        loss = ((u - (velocity - (t - r) * derivative).detach()).square().sum(-1)).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.eval().requires_grad_(False)


@torch.no_grad()
def predict(model, cloud: torch.Tensor, sigma: float, mu: float) -> np.ndarray:
    device = next(model.parameters()).device
    x = cloud.detach().to(device)
    n = len(x)
    r = torch.zeros(n, 1, dtype=x.dtype, device=device)
    t = torch.ones(n, 1, dtype=x.dtype, device=device)
    ls = torch.full((n, 1), math.log10(float(sigma)), dtype=x.dtype, device=device)
    lm = torch.full((n, 1), math.log10(float(mu)), dtype=x.dtype, device=device)
    return (x - model(x, r, t, ls, lm)).cpu().numpy().astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+",
                    default=["mlp64", "mlp128", "mlp256", "fourier128"])
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    test, pool = frozen_test_set()
    cells = stratified_subset(pool, 40, seed=20261107)
    steps = UPDATES_PER_CELL * len(cells)
    config = Config(n_train=G.N_PARTICLES, device="cuda")
    log(f"{len(test)} test cells, {len(cells)} training cells, {steps} steps per run")

    rows = []
    for name in a.configs:
        log(f"--- {name} ---")
        model = build(name)
        info = model.describe()
        log(f"  parameters: {info['params']:,}")
        try:
            import time
            started = time.perf_counter()
            model = train_variant(model, cells, config, steps=steps,
                                  batch_size=a.batch_size, lr=a.lr)
            seconds = time.perf_counter() - started
        except torch.cuda.OutOfMemoryError:
            log(f"  out of memory at width {info.get('width')}; skipping")
            torch.cuda.empty_cache()
            rows.append({"config": name, **info, "status": "oom"})
            continue

        def predictor(cloud, cell):
            return predict(model, torch.from_numpy(cloud), cell.sigma, cell.mu)

        # score_on expects a model with the standard signature; adapt via closure
        class _Adapter:
            pass

        pairs = {}
        from fullscale.train import load_cell
        from fullscale.evaluate import score_many
        for cell in test:
            payload = load_cell(cell)
            pairs[cell.key] = (payload["clouds"]["reference"],
                               predictor(payload["clouds"]["source"], cell))
        scored = score_many(pairs, bins=G.BINS, workers=4, log=lambda t: None)
        values = [r["w2"] for r in scored]
        row = {
            "config": name, **info, "status": "ok",
            "training_seconds": seconds,
            "test_mean_w2": float(np.mean(values)),
            "test_median_w2": float(np.median(values)),
            "test_max_w2": float(np.max(values)),
            "test_over_floor": float(np.mean(values) / G.FLOOR_16),
            "n_test": len(values),
        }
        rows.append(row)
        log(f"  TEST {row['test_mean_w2']:.5f}  ({row['test_over_floor']:.2f}x floor)  "
            f"{seconds:.0f}s")
        del model
        torch.cuda.empty_cache()

    out = OUT / "capacity"
    write_csv(out / "capacity_sweep.csv", rows)
    write_json(out / "capacity_sweep.json", {"steps": steps, "rows": rows})

    print(f"\n{'config':<14} {'params':>10} {'hidden':>7} {'TEST W2':>9} {'x floor':>8}")
    for r in rows:
        if r.get("status") != "ok":
            print(f"{r['config']:<14} {r.get('params', 0):>10,} {'-':>7} {'OOM':>9} {'-':>8}")
            continue
        print(f"{r['config']:<14} {r['params']:>10,} {r['hidden_layers']:>7} "
              f"{r['test_mean_w2']:9.5f} {r['test_over_floor']:8.2f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
