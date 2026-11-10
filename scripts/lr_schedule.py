"""Cosine learning-rate schedule versus the fixed rate, at equal budget.

Motivation
----------
At 32,000 updates per cell the fixed rate lr = 1e-3 reached test W2 = 0.1207, and
at 128,000 it *rose* to 0.1261 even though the training-fit error also rose
(0.1200 -> 0.1252). Both endpoint errors moving together excludes over-fitting.
The loss traces explain it: the running minimum is identical in every run
(0.0120), but the error at the *final* step degrades as training lengthens:

    budget    mean loss of last 10 records   global minimum
     8,000            0.1228                     0.0122
    32,000            0.0822                     0.0120
   128,000            0.3295                     0.0120

That is the signature of a constant step size: the iterate bounces around the
minimum with an amplitude proportional to lr, and whichever point it happens to
occupy when training stops becomes the model. More steps do not help because the
bounce does not shrink.

A decaying schedule is the standard remedy. With lr_t -> 0 the stochastic
approximation iterate converges to a stationary point instead of orbiting it.

What is compared
----------------
Three schedules at identical total budget (32,000 updates per cell, 40 cells):

    fixed     lr_t = 1e-3                                 (the current baseline)
    cosine    lr_t = eta_min + (lr0 - eta_min) * (1 + cos(pi t / T)) / 2
    warmcos   cosine with a short linear warm-up over the first 2% of steps

Reported for each: training-fit W2, held-out test W2, the generalisation gap,
and the mean/variance of the loss over the final 5% of training. A schedule that
actually fixes the problem should show a *lower final loss variance* as well as a
lower test error.

The final-window statistics matter because they are the direct diagnostic: if the
last-5% loss variance does not shrink, the schedule did not stabilise the
iterate, and any error change is noise.

Usage:
    python scripts/lr_schedule.py --schedules fixed cosine warmcos
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
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
from fullscale.evaluate import score_many  # noqa: E402
from fullscale.grid import Config  # noqa: E402
from fullscale.model import TwoParamMeanflow, meanflow_loss, sample_times, sample_two_param  # noqa: E402
from fullscale.train import load_cell, load_two, load_training_tensors  # noqa: E402

from sample_efficiency import (  # noqa: E402
    frozen_test_set, stratified_subset, write_csv, write_json, log,
)

OUT = G.RESULTS / "lr_schedule"
UPDATES_PER_CELL = 32000
SEED = 88


def schedule_value(name: str, step: int, total: int, lr0: float, eta_min: float,
                   warmup_frac: float) -> float:
    """Learning rate at a given step for the named schedule."""
    if name == "fixed":
        return lr0
    if name == "cosine":
        progress = step / max(total - 1, 1)
        return eta_min + 0.5 * (lr0 - eta_min) * (1.0 + math.cos(math.pi * progress))
    if name == "warmcos":
        warm = max(int(warmup_frac * total), 1)
        if step < warm:
            return lr0 * (step + 1) / warm
        progress = (step - warm) / max(total - warm - 1, 1)
        return eta_min + 0.5 * (lr0 - eta_min) * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"unknown schedule {name}")


def train(model, cells, config, *, steps: int, schedule: str, lr0: float,
          eta_min: float, batch_size: int, warmup_frac: float, seed: int):
    device = torch.device("cuda")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(device)
    initial, targets = load_training_tensors(cells, device)
    cs, cm = model.condition(
        torch.tensor([c.sigma for c in cells], device=device),
        torch.tensor([c.mu for c in cells], device=device),
        dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr0)
    records = []
    for step in range(1, steps + 1):
        current_lr = schedule_value(schedule, step - 1, steps, lr0, eta_min, warmup_frac)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        env = torch.randint(len(cells), (1,), device=device).expand(batch_size)
        idx = torch.randint(config.n_train, (batch_size,), device=device)
        x_target = targets[env, idx]
        x_source = initial[idx]
        r, t = sample_times(batch_size)
        r, t = r.to(device), t.to(device)
        velocity = x_source - x_target
        z = (1 - t) * x_target + t * x_source
        u, derivative = jvp(lambda zz, rr, tt: model(zz, rr, tt, cs[env, None], cm[env, None]),
                            (z, r, t), (velocity, torch.zeros_like(r), torch.ones_like(t)))
        loss, _ = meanflow_loss(u - (velocity - (t - r) * derivative).detach())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 200 == 0 or step == steps:
            records.append({"step": step, "loss": float(loss.item()), "lr": current_lr})
    return model.eval().requires_grad_(False), records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedules", nargs="+", default=["fixed", "cosine", "warmcos"])
    ap.add_argument("--updates-per-cell", type=int, default=UPDATES_PER_CELL)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eta-min", type=float, default=1e-5)
    ap.add_argument("--warmup-frac", type=float, default=0.02)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--seeds", type=int, nargs="+", default=[SEED],
                    help="training seeds; more than one is needed to claim a "
                         "reproducible percentage")
    a = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    test, pool = frozen_test_set()
    cells = stratified_subset(pool, 40, seed=20261107)
    steps = a.updates_per_cell * len(cells)
    config = Config(n_train=G.N_PARTICLES, device="cuda")
    log(f"{len(cells)} train cells, {len(test)} test cells, {steps:,} steps per schedule")

    train_payloads = {c.key: load_cell(c) for c in cells}
    test_payloads = {c.key: load_cell(c) for c in test}
    OUT.mkdir(parents=True, exist_ok=True)

    def score(model, targets_cells, payloads):
        pairs = {}
        for cell in targets_cells:
            payload = payloads[cell.key]
            pairs[cell.key] = (payload["clouds"]["reference"],
                               sample_two_param(model, torch.from_numpy(payload["clouds"]["source"]),
                                                cell.sigma, cell.mu).numpy().astype(np.float32))
        rows = score_many(pairs, bins=G.BINS, workers=4, log=lambda t: None)
        return float(np.mean([r["w2"] for r in rows]))

    rows = []
    for schedule in a.schedules:
        for seed in a.seeds:
            tag = f"{schedule}_s{seed}"
            log(f"--- schedule: {schedule}, seed {seed} ---")
            model = TwoParamMeanflow(width=64, hidden_layers=3, dimension=3, conditioning="log10")
            started = time.perf_counter()
            model, records = train(model, cells, config, steps=steps, schedule=schedule,
                                   lr0=a.lr, eta_min=a.eta_min, batch_size=a.batch_size,
                                   warmup_frac=a.warmup_frac, seed=seed)
            seconds = time.perf_counter() - started
            fit = score(model, cells, train_payloads)
            held = score(model, test, test_payloads)

            losses = np.array([r["loss"] for r in records])
            tail = losses[int(len(losses) * 0.95):]
            row = {
                "schedule": schedule,
                "seed": seed,
                "steps": steps,
                "seconds": seconds,
                "train_fit_w2": fit,
                "test_w2": held,
                "gap": held - fit,
                "test_over_floor": held / G.FLOOR_16,
                "final_loss": float(losses[-1]),
                "loss_min": float(losses.min()),
                "tail_loss_mean": float(tail.mean()),
                "tail_loss_sd": float(tail.std(ddof=1)) if len(tail) > 1 else 0.0,
                "tail_over_min": float(tail.mean() / losses.min()) if losses.min() > 0 else None,
            }
            rows.append(row)
            log(f"  fit {fit:.5f}  TEST {held:.5f}  tail loss {row['tail_loss_mean']:.4f} "
                f"(sd {row['tail_loss_sd']:.4f}, /min {row['tail_over_min']:.1f}x)  "
                f"{seconds:.0f}s")
            # Persist the weights. Without this, a later per-cell verification or a
            # different evaluation set cannot be run without retraining, and the
            # comparison is unreproducible from the artefacts alone.
            torch.save({
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "architecture": {**model.architecture(),
                                 "conditioning_mode": model.conditioning},
                "schedule": schedule, "seed": seed, "steps": steps, "lr0": a.lr,
                "eta_min": a.eta_min, "warmup_frac": a.warmup_frac,
                "n_train": config.n_train, "bins": config.bins,
                "sigma_grid": G.SIGMA_GRID, "mu_grid": G.MU_GRID,
            }, OUT / f"{tag}_model.pt")
            write_json(OUT / f"{tag}_log.json",
                       {"schedule": schedule, "seed": seed, "records": records,
                        "summary": row})
            del model
            torch.cuda.empty_cache()

    write_csv(OUT / "lr_schedule.csv", rows)

    # ---- summary by schedule (mean over seeds) ---------------------------
    by_schedule = {}
    for r in rows:
        by_schedule.setdefault(r["schedule"], []).append(r)

    print(f"\n{'schedule':<10} {'runs':>5} {'train fit':>10} {'TEST mean':>10} "
          f"{'sd':>9} {'x floor':>8}")
    summary = {}
    for schedule, group in by_schedule.items():
        tests = np.array([r["test_w2"] for r in group])
        fits = np.array([r["train_fit_w2"] for r in group])
        summary[schedule] = {
            "runs": len(group),
            "train_fit_mean": float(fits.mean()),
            "test_mean": float(tests.mean()),
            "test_sd": float(tests.std(ddof=1)) if len(tests) > 1 else 0.0,
            "test_values": [float(v) for v in tests],
            "over_floor": float(tests.mean() / G.FLOOR_16),
        }
        s = summary[schedule]
        print(f"{schedule:<10} {s['runs']:5d} {s['train_fit_mean']:10.5f} "
              f"{s['test_mean']:10.5f} {s['test_sd']:9.5f} {s['over_floor']:8.2f}")

    base_name = a.schedules[0]
    if base_name in summary:
        print(f"\ncomparison against {base_name}:")
        for schedule in a.schedules[1:]:
            if schedule not in summary:
                continue
            diff = summary[schedule]["test_mean"] - summary[base_name]["test_mean"]
            se = math.sqrt(summary[schedule]["test_sd"] ** 2 / max(summary[schedule]["runs"], 1)
                           + summary[base_name]["test_sd"] ** 2 / max(summary[base_name]["runs"], 1))
            ratio = diff / se if se > 0 else float("nan")
            verdict = ("better" if ratio < -2 else "worse" if ratio > 2
                       else "not resolvable")
            print(f"  {schedule:<10} {diff:+.5f}  "
                  f"({ratio:+.1f} SE over {summary[schedule]['runs']} seed(s))  {verdict}")
            summary[schedule]["vs_baseline"] = {"difference": diff, "se": se,
                                                "verdict": verdict}

    print("\nnote: the earlier 'tail loss sd' diagnostic was unusable -- single-minibatch")
    print("loss over batch 4096 spans 0.012..1.08 in every run, so its variance is")
    print("dominated by minibatch sampling, not by the learning rate. The decision")
    print("statistic is now the held-out test error over seeds.")

    write_json(OUT / "lr_schedule.json", {"steps": steps, "rows": rows,
                                          "summary": summary})
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
