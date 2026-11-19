"""Retrain seed-88 stage 1 with warm-up + cosine on the headline cells, then the corrector.

The cells are read from the headline checkpoint. They are not recomputed with
stratified_subset, because that function changed during development and no longer
returns the 40 cells the headline model was trained on.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.func import jvp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fullscale import grid as G  # noqa: E402
from fullscale.evaluate import score_many, write_csv, write_json  # noqa: E402
from fullscale.model import TwoParamMeanflow, meanflow_loss, sample_two_param  # noqa: E402
from fullscale.train import _save, load_cell, load_training_tensors  # noqa: E402
from lr_schedule import schedule_value  # noqa: E402
from dp_two_param import apply_corrector, train_corrector  # noqa: E402

HEADLINE = G.RESULTS / "optimal_budget_ablation" / "two_32000" / "mf_two_param.pt"
OUT = G.RESULTS / "harry_followup" / "warmcos_s88"


def log(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def cells_from_checkpoint(path: Path) -> list[G.Cell]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    keys = payload.get("cells")
    if not keys or len(keys) != 40:
        raise SystemExit(f"headline checkpoint does not record 40 cells: {path}")
    by_key = {cell.key: cell for cell in G.full_grid()}
    missing = [key for key in keys if key not in by_key]
    if missing:
        raise SystemExit(f"unknown cell keys: {missing[:5]}")
    return [by_key[key] for key in keys]


def sample_times_cuda(batch: int, device: torch.device, mean: float = -0.4, std: float = 1.0):
    """Same rule as sample_times, executed on the GPU."""
    samples = torch.sigmoid(torch.randn(batch, 2, device=device) * std + mean)
    left, right = samples[:, 0], samples[:, 1]
    t = torch.maximum(left, right)
    r = torch.minimum(left, right)
    chosen = torch.randperm(batch, device=device)[: batch // 2]
    r = r.clone()
    r[chosen] = t[chosen]
    return r[:, None], t[:, None]


def train_warmcos(cells: list[G.Cell], config: G.Config, steps: int, seed: int = 88,
                  width: int = 64):
    """Headline update, with the step captured in a CUDA graph.

    At width 64 the network has 9,027 parameters, so a plain Python step spends
    most of its time launching kernels. Replaying one captured step is about
    four times faster and does not change the batch, the loss, or the schedule.
    ``width`` is 112 only for the capacity-matched row of the same schedule.
    """
    device = torch.device("cuda")
    torch.manual_seed(seed)
    model = TwoParamMeanflow(width=width, hidden_layers=3, dimension=3, conditioning="log10").to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"warmcos width {width}, {n_params} parameters, seed {seed}")
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
        r, t = sample_times_cuda(batch, device)
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

    def learning_rate(index: int) -> float:
        return schedule_value("warmcos", index, steps, 1e-3, 1e-5, 0.02)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for index in range(3):
            lr.fill_(learning_rate(index))
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    lr.fill_(learning_rate(3))
    with torch.cuda.graph(graph):
        step()

    records = []
    started = time.perf_counter()
    for index in range(4, steps):
        lr.fill_(learning_rate(index))
        graph.replay()
        done = index + 1
        if done % 20000 == 0 or done == steps:
            torch.cuda.synchronize()
            loss_value = float(loss_box["loss"])
            records.append({"step": done, "loss": loss_value, "lr": learning_rate(index)})
            log(f"  step {done}/{steps} loss={loss_value:.6g} lr={learning_rate(index):.3g}")
    seconds = time.perf_counter() - started
    return model.eval().requires_grad_(False), records, seconds


def held_out(train: list[G.Cell]) -> list[G.Cell]:
    train_keys = {cell.key for cell in train}
    slice_keys = {cell.key for cell in G.baseline_slice()}
    return [cell for cell in G.full_grid()
            if cell.key not in train_keys and cell.key not in slice_keys]


def main() -> int:
    torch.set_num_threads(min(8, torch.get_num_threads()))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    OUT.mkdir(parents=True, exist_ok=True)
    cells = cells_from_checkpoint(HEADLINE)
    held = held_out(cells)
    log(f"headline cells {len(cells)}, held-out {len(held)}")
    (OUT / "cells.json").write_text(json.dumps({
        "source": str(HEADLINE),
        "train": [cell.key for cell in cells],
        "held_out": [cell.key for cell in held],
    }, indent=2), encoding="utf-8")

    steps = 32_000 * len(cells)
    config = G.Config(n_train=G.N_PARTICLES, device="cuda")
    log(f"warmcos stage 1: {steps} steps, lr 1e-3 -> 1e-5, warmup 2%, CUDA graph")
    model, records, stage1_seconds = train_warmcos(cells, config, steps, seed=88)
    _save(OUT / "mf_two_param.pt", model, cells, config, steps, 88)
    (OUT / "stage1_log.json").write_text(json.dumps({
        "schedule": "warmcos", "seed": 88, "steps": steps,
        "seconds": stage1_seconds, "records": records,
        "cells": [cell.key for cell in cells],
    }), encoding="utf-8")
    log(f"stage 1 done in {stage1_seconds:.0f}s")

    result = train_corrector(
        model, cells, config, updates_per_cell=2000, batch=2000, alpha=1.0,
        blocks=2, residual_weight=1.0, ot_refresh=10, lr=1e-4, seed=88, log=log,
    )
    corrector = result["model"]
    torch.save({
        "model": {k: v.detach().cpu().clone() for k, v in corrector.state_dict().items()},
        "architecture": corrector.architecture(),
        "cells": [cell.key for cell in cells],
        "updates_per_cell": 2000, "batch": 2000, "alpha": 1.0,
        "residual_weight": 1.0, "ot_refresh": 10, "lr": 1e-4, "seed": 88,
        "mf_checkpoint": str(OUT / "mf_two_param.pt"),
    }, OUT / "dp_corrector.pt")
    log(f"corrector done in {result['seconds']:.0f}s")

    log(f"scoring {len(held)} held-out cells")
    pairs_mf, pairs_dp = {}, {}
    for position, cell in enumerate(held, start=1):
        payload = load_cell(cell)
        source = torch.from_numpy(payload["clouds"]["source"])
        reference = payload["clouds"]["reference"]
        mf = sample_two_param(model, source, cell.sigma, cell.mu).numpy().astype(np.float32)
        dp = apply_corrector(corrector, torch.from_numpy(mf), cell.sigma, cell.mu)
        pairs_mf[cell.key] = (reference, mf)
        pairs_dp[cell.key] = (reference, dp)
        if position % 40 == 0:
            log(f"  generated {position}/{len(held)}")
    rows_mf = score_many(pairs_mf, bins=G.BINS, workers=4, log=lambda text: None)
    rows_dp = score_many(pairs_dp, bins=G.BINS, workers=4, log=lambda text: None)
    by_mf = {row["key"]: row["w2"] for row in rows_mf}
    by_dp = {row["key"]: row["w2"] for row in rows_dp}
    detail = []
    for cell in held:
        detail.append({
            "key": cell.key, "sigma": cell.sigma, "mu": cell.mu,
            "w2_mf": by_mf[cell.key], "w2_mf_dp": by_dp[cell.key],
        })
    write_csv(OUT / "detail.csv", detail)
    mean_mf = float(np.mean([row["w2_mf"] for row in detail]))
    mean_dp = float(np.mean([row["w2_mf_dp"] for row in detail]))
    summary = {
        "schedule": "warmcos", "seed": 88,
        "n_train": len(cells), "n_held_out": len(held),
        "stage1_seconds": stage1_seconds,
        "corrector_seconds": result["seconds"],
        "mf_mean_w2": mean_mf, "mf_dp_mean_w2": mean_dp,
        "relative_reduction": float(1.0 - mean_dp / mean_mf),
        "dp_wins": int(sum(row["w2_mf_dp"] < row["w2_mf"] for row in detail)),
    }
    write_json(OUT / "summary.json", summary)
    log(f"held-out MF {mean_mf:.5f}  MF+DP {mean_dp:.5f}  "
        f"reduction {summary['relative_reduction']:+.1%}  "
        f"wins {summary['dp_wins']}/{len(held)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
