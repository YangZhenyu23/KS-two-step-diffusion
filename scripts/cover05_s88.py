"""Retrain seed-88 stage 1 with intervals the two-step evaluation actually uses.

The original sampler leaves only about 4% of intervals at least as long as 0.5.
Two-step inference always evaluates [1, 0.5] and [0.5, 0]. This run keeps the
fixed learning rate, the 40 headline cells and 32,000 updates per cell, and
changes only the time sampler:

- 40% original logit-normal pairs, not forced onto the diagonal
- 20% diagonal r = t
- 20% the one-step pair (0, 1)
- 10% the first sub-step (0, 0.5)
- 10% the second sub-step (0.5, 1)

The corrector is then trained on this stage 1 with the original protocol.
Held-out scores compare one step, two steps, and one step plus the corrector
with the fixed-rate headline that used the original sampler.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import jvp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fullscale import grid as G
from fullscale.evaluate import score_many, write_json
from fullscale.model import TwoParamMeanflow, meanflow_loss, sample_two_param
from fullscale.train import _save, load_cell, load_training_tensors, load_two
from dp_two_param import apply_corrector, train_corrector
from harry_warmcos_s88 import HEADLINE, cells_from_checkpoint, held_out

OUT = G.RESULTS / "harry_followup" / "cover05_s88"


def log(text: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {text}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "train.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sample_times_cover(batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixture that keeps the original sampler and adds the evaluated intervals."""
    samples = torch.sigmoid(torch.randn(batch, 2, device=device) - 0.4)
    t = torch.maximum(samples[:, 0], samples[:, 1])
    r = torch.minimum(samples[:, 0], samples[:, 1])
    draw = torch.rand(batch, device=device)
    diagonal = (draw >= 0.40) & (draw < 0.60)
    full = (draw >= 0.60) & (draw < 0.80)
    lower = (draw >= 0.80) & (draw < 0.90)
    upper = draw >= 0.90
    r = torch.where(diagonal, t, r)
    r = torch.where(full, torch.zeros_like(r), r)
    t = torch.where(full, torch.ones_like(t), t)
    r = torch.where(lower, torch.zeros_like(r), r)
    t = torch.where(lower, torch.full_like(t, 0.5), t)
    r = torch.where(upper, torch.full_like(r, 0.5), r)
    t = torch.where(upper, torch.ones_like(t), t)
    return r[:, None], t[:, None]


def sampler_report(device: torch.device) -> dict:
    r, t = sample_times_cover(1_000_000, device)
    length = (t - r).squeeze()
    report = {
        "n": 1_000_000,
        "fraction_diagonal": float(torch.isclose(r, t).float().mean()),
        "fraction_shorter_than_half": float((length < 0.5).float().mean()),
        "fraction_at_least_half": float((length >= 0.5).float().mean()),
        "fraction_full_interval": float(((r.squeeze() == 0) & (t.squeeze() == 1)).float().mean()),
        "fraction_lower_half": float(((r.squeeze() == 0) & torch.isclose(t.squeeze(), torch.tensor(0.5, device=device))).float().mean()),
        "fraction_upper_half": float((torch.isclose(r.squeeze(), torch.tensor(0.5, device=device)) & (t.squeeze() == 1)).float().mean()),
        "mixture": "40% original logit-normal, 20% diagonal, 20% (0,1), 10% (0,0.5), 10% (0.5,1)",
    }
    log(
        f"sampler: diagonal {report['fraction_diagonal']:.1%}, "
        f"length>=0.5 {report['fraction_at_least_half']:.1%}"
    )
    return report


def train_stage1(cells, config, steps: int):
    device = torch.device("cuda")
    torch.manual_seed(88)
    model = TwoParamMeanflow(width=64, hidden_layers=3, dimension=3, conditioning="log10").to(device)
    initial, targets = load_training_tensors(cells, device)
    log_sigma, log_mu = model.condition(
        torch.tensor([cell.sigma for cell in cells], device=device),
        torch.tensor([cell.mu for cell in cells], device=device),
        dtype=torch.float32, device=device)
    lr = torch.tensor(1e-3, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, capturable=True)
    batch = 4096
    loss_box: dict[str, torch.Tensor] = {}

    def step() -> None:
        env = torch.randint(len(cells), (1,), device=device).expand(batch)
        idx = torch.randint(config.n_train, (batch,), device=device)
        x_target = targets[env, idx]
        x_source = initial[idx]
        r, t = sample_times_cover(batch, device)
        velocity = x_source - x_target
        z = (1 - t) * x_target + t * x_source
        u, derivative = jvp(
            lambda zz, rr, tt: model(zz, rr, tt, log_sigma[env, None], log_mu[env, None]),
            (z, r, t), (velocity, torch.zeros_like(r), torch.ones_like(t)))
        loss, _ = meanflow_loss(u - (velocity - (t - r) * derivative).detach())
        loss_box["loss"] = loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    started = time.perf_counter()
    records = []
    for index in range(4, steps):
        graph.replay()
        done = index + 1
        if done % 20000 == 0 or done == steps:
            torch.cuda.synchronize()
            loss_value = float(loss_box["loss"])
            records.append({"step": done, "loss": loss_value})
            log(f"  step {done}/{steps} loss={loss_value:.6g}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, records, time.perf_counter() - started


def mean_w2(pairs) -> float:
    rows = score_many(pairs, bins=G.BINS, workers=4, log=lambda text: None)
    return float(np.mean([row["w2"] for row in rows]))


def clouds_for(model, corrector, held) -> dict:
    one, two, corrected = {}, {}, {}
    for position, cell in enumerate(held, start=1):
        payload = load_cell(cell)
        source = torch.from_numpy(payload["clouds"]["source"])
        reference = payload["clouds"]["reference"]
        with torch.no_grad():
            first = sample_two_param(model, source, cell.sigma, cell.mu, steps=1).numpy().astype(np.float32)
            second = sample_two_param(model, source, cell.sigma, cell.mu, steps=2).numpy().astype(np.float32)
        one[cell.key] = (reference, first)
        two[cell.key] = (reference, second)
        if corrector is not None:
            refined = apply_corrector(corrector, torch.from_numpy(first), cell.sigma, cell.mu)
            corrected[cell.key] = (reference, refined)
        if position % 40 == 0:
            log(f"  generated {position}/{len(held)}")
    return {"one": one, "two": two, "corrected": corrected}


def main() -> int:
    torch.set_num_threads(min(8, torch.get_num_threads()))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    OUT.mkdir(parents=True, exist_ok=True)
    cells = cells_from_checkpoint(HEADLINE)
    held = held_out(cells)
    config = G.Config(n_train=G.N_PARTICLES, device="cuda")
    steps = 32_000 * len(cells)
    report = sampler_report(torch.device("cuda"))
    log(f"cover05 stage 1: {len(cells)} cells, {steps} steps, fixed lr 1e-3")
    model, records, stage1_seconds = train_stage1(cells, config, steps)
    _save(OUT / "mf_two_param.pt", model, cells, config, steps, 88)
    log(f"stage 1 done in {stage1_seconds:.0f}s")

    result = train_corrector(
        model, cells, config, updates_per_cell=2000, batch=2000, alpha=1.0,
        blocks=2, residual_weight=1.0, ot_refresh=10, lr=1e-4, seed=88, log=log,
    )
    corrector = result["model"]
    torch.save({
        "model": {key: value.detach().cpu().clone() for key, value in corrector.state_dict().items()},
        "architecture": corrector.architecture(),
        "cells": [cell.key for cell in cells],
        "updates_per_cell": 2000, "batch": 2000, "alpha": 1.0,
        "residual_weight": 1.0, "ot_refresh": 10, "lr": 1e-4, "seed": 88,
        "mf_checkpoint": str(OUT / "mf_two_param.pt"),
    }, OUT / "dp_corrector.pt")
    log(f"corrector done in {result['seconds']:.0f}s")

    log("scoring headline and cover05 on the same 143 cells")
    baseline, _ = load_two(HEADLINE, "cuda")
    old = clouds_for(baseline, None, held)
    new = clouds_for(model, corrector, held)
    summary = {
        "sampler": report,
        "seed": 88,
        "learning_rate": 1e-3,
        "updates_per_cell": 32000,
        "n_train": len(cells),
        "n_held_out": len(held),
        "stage1_seconds": stage1_seconds,
        "corrector_seconds": result["seconds"],
        "stage1_log": records[:: max(len(records) // 20, 1)],
        "headline_one_step": mean_w2(old["one"]),
        "headline_two_step": mean_w2(old["two"]),
        "cover_one_step": mean_w2(new["one"]),
        "cover_two_step": mean_w2(new["two"]),
        "cover_corrector": mean_w2(new["corrected"]),
    }
    summary["one_step_change"] = summary["cover_one_step"] / summary["headline_one_step"] - 1.0
    summary["two_step_change"] = summary["cover_two_step"] / summary["headline_two_step"] - 1.0
    summary["cover_two_vs_one"] = summary["cover_two_step"] / summary["cover_one_step"] - 1.0
    summary["cover_corrector_vs_one"] = 1.0 - summary["cover_corrector"] / summary["cover_one_step"]
    write_json(OUT / "summary.json", summary)
    (OUT / "stage1_log.json").write_text(json.dumps({
        "seconds": stage1_seconds, "records": records,
    }), encoding="utf-8")
    log(
        f"one {summary['headline_one_step']:.5f} -> {summary['cover_one_step']:.5f} "
        f"({summary['one_step_change']:+.1%})"
    )
    log(
        f"two {summary['headline_two_step']:.5f} -> {summary['cover_two_step']:.5f} "
        f"({summary['two_step_change']:+.1%})"
    )
    log(
        f"corrector {summary['cover_corrector']:.5f}  "
        f"vs its one-step {summary['cover_corrector_vs_one']:+.1%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
