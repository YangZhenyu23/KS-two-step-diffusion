"""E7: does the corrector need to have seen the cell it is correcting?

The cheapest discriminator between a structural corrector and a memorising one is how much
training coverage it needs. A corrector that learns "how this class of stage-1 error should be
corrected" should work from very few cells. One that memorises per-cell displacements will
degrade sharply when the training set is thinned.

Design
------
Train the same corrector on 4, 10, 20 and 40 cells (always a subset of the 40-cell design, so
every configuration is scored on the same 143 held-out cells), then measure gain against stage 1
alone. Everything else -- batch, updates per cell, alpha, OT refresh, learning rate -- is held
fixed.

The comparison of interest is not the absolute curve but where it saturates. If 4 cells already
gives most of the 40-cell gain, the corrector is not relying on per-cell experience.

Cost: 4 + 10 + 20 + 40 = 74 cell-training units at 2,000 updates each.

Usage:
    python scripts/e7_few_shot_corrector.py --cells 4 10 20 40
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
sys.path.insert(0, str(ROOT / "scripts"))

from fullscale import grid as G  # noqa: E402
from fullscale.train import load_cell, load_two  # noqa: E402
from fullscale.model import sample_two_param  # noqa: E402
from fullscale.evaluate import score_pair  # noqa: E402

from dp_two_param import (  # noqa: E402
    TwoParamCorrector, exact_assignment, apply_corrector,
)

OUT = G.RESULTS / "few_shot"


def train_on_cells(mf_model, cells, config, *, updates_per_cell, batch, alpha, blocks,
                   ot_refresh, lr, seed, device):
    """A trimmed copy of ``dp_two_param.train_corrector`` that returns the trained model."""
    log_sigma = torch.tensor([math.log10(c.sigma) for c in cells],
                             dtype=torch.float32, device=device)
    log_mu = torch.tensor([math.log10(c.mu) for c in cells],
                          dtype=torch.float32, device=device)

    mf_clouds, targets = [], []
    for cell in cells:
        payload = load_cell(cell)
        source = torch.from_numpy(payload["clouds"]["source"])
        with torch.no_grad():
            mf_clouds.append(
                sample_two_param(mf_model, source, cell.sigma, cell.mu).to(device))
        targets.append(torch.from_numpy(payload["clouds"]["reference"]).to(device))

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TwoParamCorrector(width=64, alpha=alpha, blocks=blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total = updates_per_cell * len(cells)

    matched, batch_idx, env_fixed = None, None, 0
    for step in range(1, total + 1):
        model.train()
        env = int(torch.randint(len(cells), (1,)).item())
        refresh = batch_idx is None or (step - 1) % ot_refresh == 0
        if refresh:
            env_fixed = env
            batch_idx = torch.randperm(config.n_train, device=device)[:batch]
            x_mf = mf_clouds[env_fixed][batch_idx]
            x_tgt = targets[env_fixed][batch_idx]
            cs = log_sigma[env_fixed].expand(batch, 1)
            cm = log_mu[env_fixed].expand(batch, 1)
            with torch.no_grad():
                current = model(x_mf, cs, cm).cpu().numpy()
            perm, _, _ = exact_assignment(current, x_tgt.cpu().numpy())
            matched = torch.as_tensor(perm, dtype=torch.long, device=device)
        else:
            x_mf = mf_clouds[env_fixed][batch_idx]
            x_tgt = targets[env_fixed][batch_idx]
            cs = log_sigma[env_fixed].expand(batch, 1)
            cm = log_mu[env_fixed].expand(batch, 1)

        corrected = model(x_mf, cs, cm)
        loss = ((corrected - x_tgt[matched]) ** 2).sum(1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, nargs="+", default=[4, 10, 20, 40])
    ap.add_argument("--updates-per-cell", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--ot-refresh", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--seed", type=int, default=88)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda text: print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)

    from fullscale.grid import Config
    config = Config(n_train=G.N_PARTICLES, device=a.device)

    headline = G.RESULTS / "optimal_budget_ablation" / "two_32000" / "mf_two_param.pt"
    train_keys = list(torch.load(headline, map_location="cpu", weights_only=False)["cells"])
    pool = [G.Cell(int(k[1:3]), int(k[5:7])) for k in train_keys]
    slice_keys = {c.key for c in G.baseline_slice()}
    held = [c for c in G.full_grid()
            if c.key not in set(train_keys) and c.key not in slice_keys]
    log(f"pool {len(pool)} cells, held out {len(held)}")

    mf_model, _ = load_two(headline, a.device)
    device = torch.device("cuda")

    log("scoring stage 1 alone")
    baseline, payloads = {}, {}
    for cell in held:
        item = load_cell(cell)
        payloads[cell.key] = item
        mf = sample_two_param(mf_model, torch.from_numpy(item["clouds"]["source"]),
                              cell.sigma, cell.mu, steps=1)
        baseline[cell.key] = float(score_pair(item["clouds"]["reference"],
                                              mf.numpy().astype(np.float32),
                                              G.BINS)["w2"])
    base_mean = float(np.mean(list(baseline.values())))
    log(f"  stage 1 held-out {base_mean:.5f}")

    rows = []
    for count in a.cells:
        step = max(1, len(pool) // count)
        subset = pool[::step][:count]
        log(f"=== training corrector on {len(subset)} cells ===")
        model = train_on_cells(
            mf_model, subset, config,
            updates_per_cell=a.updates_per_cell, batch=a.batch, alpha=a.alpha,
            blocks=a.blocks, ot_refresh=a.ot_refresh, lr=a.lr, seed=a.seed,
            device=a.device,
        )
        scores = {}
        for cell in held:
            item = payloads[cell.key]
            mf = sample_two_param(mf_model, torch.from_numpy(item["clouds"]["source"]),
                                  cell.sigma, cell.mu, steps=1)
            dp = apply_corrector(model, mf, cell.sigma, cell.mu)
            scores[cell.key] = float(score_pair(item["clouds"]["reference"], dp,
                                                G.BINS)["w2"])
        mean = float(np.mean(list(scores.values())))
        wins = sum(1 for k in scores if scores[k] < baseline[k])
        rows.append({"n_cells": len(subset), "held_out_w2": mean,
                     "gain_vs_stage1": float(1 - mean / base_mean),
                     "wins": wins, "n_held_out": len(scores)})
        log(f"  {len(subset)} cells -> {mean:.5f}  ({1 - mean / base_mean:+.1%})  "
            f"{wins}/{len(scores)} improved")
        del model
        torch.cuda.empty_cache()

    payload = {"updates_per_cell": a.updates_per_cell, "batch": a.batch,
               "alpha": a.alpha, "ot_refresh": a.ot_refresh, "lr": a.lr,
               "stage1_held_out": base_mean, "n_held_out": len(held), "rows": rows,
               "note": ("Every configuration is scored on the same held-out cells; only the "
                        "number of training cells changes.")}
    (OUT / "few_shot.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nstage 1 alone: {base_mean:.5f}")
    print(f"{'cells':>6} {'held-out':>10} {'gain':>9} {'wins':>12}")
    for row in rows:
        print(f"{row['n_cells']:6d} {row['held_out_w2']:10.5f} "
              f"{row['gain_vs_stage1'] * 100:+8.1f}% "
              f"{row['wins']:>4}/{row['n_held_out']:<6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
