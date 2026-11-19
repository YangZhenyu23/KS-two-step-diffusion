"""Finish the two Harry items that are not on disk yet.

1. Four-row table on the warm-up + cosine stage 1.
   Rows (a), (b) and (d) use the seed-88 warmcos map and its corrector.
   Row (c) retrains width 112 with the same schedule, on the cells stored in
   the existing width-112 checkpoints, three seeds. The scored cells are the
   intersection held out by the headline design, every width-112 design, and
   the mu=1 row.

2. One corrector trained on the fixed-learning-rate stage-1 maps of seeds
   88, 89 and 90 together. Each map keeps its own training cells. The
   assignment is the same exact minibatch coupling as the headline corrector;
   SciPy's Hungarian algorithm matched POT's optimal cost and is what this
   run uses. Alpha stays 1, the headline value, and is not re-selected.

Already finished jobs (budget controls, consistency, alpha on seeds 89/90,
sliced W2, the warmcos seed-88 headline) are not repeated.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fullscale import grid as G  # noqa: E402
from fullscale.evaluate import score_many, write_json  # noqa: E402
from fullscale.model import sample_two_param  # noqa: E402
from fullscale.train import _save, load_cell, load_two  # noqa: E402
from dp_two_param import TwoParamCorrector, apply_corrector  # noqa: E402
from harry_warmcos_s88 import (  # noqa: E402
    HEADLINE, cells_from_checkpoint, train_warmcos,
)

OUT = G.RESULTS / "harry_followup"
WARM = OUT / "warmcos_s88"
WIDE_OLD = {
    88: G.RESULTS / "capacity_matched" / "w112_s88" / "mf_two_param.pt",
    89: G.RESULTS / "capacity_matched" / "w112_s89" / "mf_two_param.pt",
    90: G.RESULTS / "capacity_matched" / "w112_s90" / "mf_two_param.pt",
}
WIDE_NEW = {seed: OUT / "warmcos_w112" / f"s{seed}" for seed in (88, 89, 90)}
FIXED = {
    88: HEADLINE,
    89: G.RESULTS / "optimal_budget_ablation" / "two_32000_s89" / "mf_two_param.pt",
    90: G.RESULTS / "optimal_budget_ablation" / "two_32000_s90" / "mf_two_param.pt",
}
FOUR = OUT / "four_row_warmcos.json"
MAP_ROWS = OUT / "four_row_warmcos_abd.json"
MULTI = OUT / "multiseed_corrector" / "summary.json"


def log(text: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {text}"
    print(line, flush=True)
    with (OUT / "remaining.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def keys_of(path: Path) -> frozenset:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    keys = payload.get("cells")
    if not keys or len(keys) != 40:
        raise SystemExit(f"{path} does not record 40 cells")
    return frozenset(keys)


def common_cells() -> list[G.Cell]:
    banned = set(keys_of(HEADLINE))
    for path in WIDE_OLD.values():
        banned |= set(keys_of(path))
    banned |= {cell.key for cell in G.baseline_slice()}
    cells = [cell for cell in G.full_grid() if cell.key not in banned]
    log(f"four-row common held-out cells: {len(cells)}")
    return cells


def mean_w2(pairs: dict) -> tuple[float, dict]:
    rows = score_many(pairs, bins=G.BINS, workers=4, log=lambda text: None)
    by_key = {row["key"]: float(row["w2"]) for row in rows}
    return float(np.mean(list(by_key.values()))), by_key


def load_warm_corrector(device: str):
    checkpoint = torch.load(WARM / "dp_corrector.pt", map_location=device, weights_only=False)
    arch = checkpoint.get("architecture", {})
    model = TwoParamCorrector(
        width=arch.get("width", 64), alpha=float(checkpoint.get("alpha", 1.0)),
        blocks=checkpoint.get("blocks", arch.get("blocks", 2)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def score_map_rows(cells: list[G.Cell]) -> dict:
    device = "cuda"
    model, _ = load_two(WARM / "mf_two_param.pt", device)
    corrector = load_warm_corrector(device)
    one, two, both = {}, {}, {}
    for position, cell in enumerate(cells, start=1):
        payload = load_cell(cell)
        source = torch.from_numpy(payload["clouds"]["source"])
        reference = payload["clouds"]["reference"]
        step1 = sample_two_param(model, source, cell.sigma, cell.mu, steps=1).numpy().astype(np.float32)
        step2 = sample_two_param(model, source, cell.sigma, cell.mu, steps=2).numpy().astype(np.float32)
        corrected = apply_corrector(corrector, torch.from_numpy(step1), cell.sigma, cell.mu)
        one[cell.key] = (reference, step1)
        two[cell.key] = (reference, step2)
        both[cell.key] = (reference, corrected)
        if position % 40 == 0:
            log(f"  generated {position}/{len(cells)}")
    mean_a, by_a = mean_w2(one)
    mean_d, _ = mean_w2(two)
    mean_b, by_b = mean_w2(both)
    wins = sum(by_b[key] < by_a[key] for key in by_a)
    log(f"  (a) one-step {mean_a:.5f}  (d) two-step {mean_d:.5f}  "
        f"(b) corrector {mean_b:.5f}  wins {wins}/{len(cells)}")
    del model, corrector
    torch.cuda.empty_cache()
    return {
        "a_one_step": mean_a,
        "d_two_step": mean_d,
        "d_relative_vs_a": float(1.0 - mean_d / mean_a),
        "b_corrector": mean_b,
        "b_relative_vs_a": float(1.0 - mean_b / mean_a),
        "b_wins": wins,
        "n": len(cells),
    }


def probe_width(cells: list[G.Cell], width: int) -> None:
    """Time a few eager steps before the long run. The choice is still the graph."""
    from torch.func import jvp

    from fullscale.model import TwoParamMeanflow, meanflow_loss
    from fullscale.train import load_training_tensors
    from harry_warmcos_s88 import sample_times_cuda

    device = torch.device("cuda")
    torch.manual_seed(0)
    model = TwoParamMeanflow(width=width, hidden_layers=3, dimension=3, conditioning="log10").to(device)
    initial, targets = load_training_tensors(cells[:4], device)
    log_sigma, log_mu = model.condition(
        torch.tensor([cell.sigma for cell in cells[:4]], device=device),
        torch.tensor([cell.mu for cell in cells[:4]], device=device),
        dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = 4096

    def once() -> None:
        env = torch.randint(4, (1,), device=device).expand(batch)
        idx = torch.randint(G.N_PARTICLES, (batch,), device=device)
        x_target = targets[env, idx]
        x_source = initial[idx]
        r, t = sample_times_cuda(batch, device)
        velocity = x_source - x_target
        z = (1 - t) * x_target + t * x_source
        u, derivative = jvp(
            lambda zz, rr, tt: model(zz, rr, tt, log_sigma[env, None], log_mu[env, None]),
            (z, r, t), (velocity, torch.zeros_like(r), torch.ones_like(t)))
        loss, _ = meanflow_loss(u - (velocity - (t - r) * derivative).detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    for _ in range(3):
        once()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(8):
        once()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) / 8 * 1000
    log(f"width {width} eager step {elapsed:.2f} ms; training will replay a CUDA graph")
    del model, optimizer, initial, targets
    torch.cuda.empty_cache()


def train_wide(cells_by_seed: dict[int, list[G.Cell]]) -> None:
    probe_width(cells_by_seed[88], 112)
    config = G.Config(n_train=G.N_PARTICLES, device="cuda")
    for seed, cells in cells_by_seed.items():
        run = WIDE_NEW[seed]
        destination = run / "mf_two_param.pt"
        if destination.exists():
            log(f"width-112 seed {seed} already trained")
            continue
        run.mkdir(parents=True, exist_ok=True)
        steps = 32_000 * len(cells)
        log(f"width-112 warmcos seed {seed}: {steps} steps on {len(cells)} stored cells")
        model, records, seconds = train_warmcos(cells, config, steps, seed=seed, width=112)
        _save(destination, model, cells, config, steps, seed)
        (run / "stage1_log.json").write_text(json.dumps({
            "schedule": "warmcos", "width": 112, "seed": seed, "steps": steps,
            "seconds": seconds, "records": records,
            "cells": [cell.key for cell in cells],
        }), encoding="utf-8")
        log(f"width-112 seed {seed} done in {seconds:.0f}s")
        del model
        torch.cuda.empty_cache()


def score_wide(cells: list[G.Cell], baseline: float) -> dict:
    per_seed = {}
    for seed in (88, 89, 90):
        model, _ = load_two(WIDE_NEW[seed] / "mf_two_param.pt", "cuda")
        pairs = {}
        for cell in cells:
            payload = load_cell(cell)
            source = torch.from_numpy(payload["clouds"]["source"])
            prediction = sample_two_param(
                model, source, cell.sigma, cell.mu, steps=1).numpy().astype(np.float32)
            pairs[cell.key] = (payload["clouds"]["reference"], prediction)
        mean, _ = mean_w2(pairs)
        per_seed[str(seed)] = mean
        log(f"  (c) width-112 seed {seed} {mean:.5f}  "
            f"relative {1.0 - mean / baseline:+.1%}")
        del model
        torch.cuda.empty_cache()
    values = np.array(list(per_seed.values()))
    return {
        "per_seed": per_seed,
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "relative_vs_a": float(1.0 - values.mean() / baseline),
    }


def four_row() -> None:
    if FOUR.exists():
        log("four-row json already on disk")
        return
    cells = common_cells()
    if MAP_ROWS.exists():
        rows = json.loads(MAP_ROWS.read_text(encoding="utf-8"))
        log(f"reusing rows (a)(b)(d) from {MAP_ROWS.name}")
    else:
        rows = score_map_rows(cells)
        write_json(MAP_ROWS, rows)
    by_seed = {seed: cells_from_checkpoint(WIDE_OLD[seed]) for seed in (88, 89, 90)}
    train_wide(by_seed)
    rows["c_width112"] = score_wide(cells, rows["a_one_step"])
    rows["cell_keys"] = [cell.key for cell in cells]
    rows["note"] = (
        "Rows (a)(b)(d) are the seed-88 warmcos map. Row (c) is width 112 trained "
        "with the same warmcos schedule on each existing checkpoint's own 40 cells. "
        "The cell set is held out by the headline design and by all three width-112 designs."
    )
    write_json(FOUR, rows)
    log(f"FOUR ROW DONE (b) {rows['b_relative_vs_a']:+.1%}  "
        f"(c) {rows['c_width112']['relative_vs_a']:+.1%}  "
        f"(d) {rows['d_relative_vs_a']:+.1%}")


def hungarian(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    cost = cdist(current, target, metric="sqeuclidean")
    rows, columns = linear_sum_assignment(cost)
    if not np.array_equal(rows, np.arange(len(cost))):
        raise RuntimeError("Hungarian assignment did not cover every row")
    return columns.astype(np.int64)


def train_multiseed() -> None:
    destination = OUT / "multiseed_corrector"
    if MULTI.exists():
        log("multiseed corrector summary already on disk")
        return
    destination.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    groups = []
    for seed, path in FIXED.items():
        model, _ = load_two(path, "cuda")
        cells = cells_from_checkpoint(path)
        log(f"pre-computing seed {seed} stage 1 on {len(cells)} training cells")
        for cell in cells:
            payload = load_cell(cell)
            source = torch.from_numpy(payload["clouds"]["source"])
            with torch.no_grad():
                predicted = sample_two_param(model, source, cell.sigma, cell.mu)
            groups.append({
                "seed": seed,
                "key": cell.key,
                "mf": predicted.to(device),
                "target": torch.from_numpy(payload["clouds"]["reference"]).to(device),
                "log_sigma": math_log10(cell.sigma),
                "log_mu": math_log10(cell.mu),
            })
        del model
        torch.cuda.empty_cache()

    torch.manual_seed(88)
    corrector = TwoParamCorrector(width=64, alpha=1.0, blocks=2).to(device)
    optimizer = torch.optim.Adam(corrector.parameters(), lr=1e-4)
    updates_per_cell = 2000
    total = updates_per_cell * len(groups)
    batch = 2000
    refresh = 10
    rng = np.random.default_rng(88)
    log(f"multiseed corrector: {len(groups)} map-cell pairs x {updates_per_cell} "
        f"= {total} steps, Hungarian refresh {refresh}")
    started = time.perf_counter()
    matched = None
    batch_idx = None
    chosen = None
    loss_value = None
    for step in range(1, total + 1):
        corrector.train()
        if batch_idx is None or (step - 1) % refresh == 0:
            chosen = groups[int(rng.integers(0, len(groups)))]
            batch_idx = torch.randperm(G.N_PARTICLES, device=device)[:batch]
            x_mf = chosen["mf"][batch_idx]
            x_tgt = chosen["target"][batch_idx]
            cs = torch.full((batch, 1), chosen["log_sigma"], device=device)
            cm = torch.full((batch, 1), chosen["log_mu"], device=device)
            with torch.no_grad():
                current = corrector(x_mf, cs, cm).detach().cpu().numpy()
            matched = torch.as_tensor(
                hungarian(current, x_tgt.detach().cpu().numpy()),
                dtype=torch.long, device=device)
        else:
            x_mf = chosen["mf"][batch_idx]
            x_tgt = chosen["target"][batch_idx]
            cs = torch.full((batch, 1), chosen["log_sigma"], device=device)
            cm = torch.full((batch, 1), chosen["log_mu"], device=device)
        refined = corrector(x_mf, cs, cm)
        transport = (refined - x_tgt[matched]).square().sum(-1).mean()
        residual = (refined - x_mf).square().sum(-1).mean()
        loss = transport + residual
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % max(total // 10, 1) == 0:
            loss_value = float(loss.detach())
            log(f"  multiseed step {step}/{total} loss={loss_value:.5g}")
    seconds = time.perf_counter() - started
    corrector.eval()
    for parameter in corrector.parameters():
        parameter.requires_grad_(False)
    torch.save({
        "model": {key: value.detach().cpu().clone() for key, value in corrector.state_dict().items()},
        "architecture": corrector.architecture(),
        "alpha": 1.0,
        "seeds": [88, 89, 90],
        "updates_per_cell": updates_per_cell,
        "assignment": "scipy.linear_sum_assignment",
        "seconds": seconds,
    }, destination / "dp_corrector.pt")
    log(f"multiseed corrector trained in {seconds:.0f}s")

    per_seed = {}
    for seed, path in FIXED.items():
        model, _ = load_two(path, "cuda")
        train = cells_from_checkpoint(path)
        train_keys = {cell.key for cell in train}
        slice_keys = {cell.key for cell in G.baseline_slice()}
        held = [cell for cell in G.full_grid()
                if cell.key not in train_keys and cell.key not in slice_keys]
        alone, corrected = {}, {}
        for cell in held:
            payload = load_cell(cell)
            source = torch.from_numpy(payload["clouds"]["source"])
            reference = payload["clouds"]["reference"]
            base = sample_two_param(model, source, cell.sigma, cell.mu).numpy().astype(np.float32)
            refined = apply_corrector(corrector, torch.from_numpy(base), cell.sigma, cell.mu)
            alone[cell.key] = (reference, base)
            corrected[cell.key] = (reference, refined)
        mean_mf, by_mf = mean_w2(alone)
        mean_dp, by_dp = mean_w2(corrected)
        wins = sum(by_dp[key] < by_mf[key] for key in by_mf)
        per_seed[str(seed)] = {
            "n": len(held),
            "mf": mean_mf,
            "mf_dp": mean_dp,
            "relative_reduction": float(1.0 - mean_dp / mean_mf),
            "wins": wins,
        }
        log(f"  seed {seed} held-out {len(held)}: {mean_mf:.5f} -> {mean_dp:.5f} "
            f"({per_seed[str(seed)]['relative_reduction']:+.1%}, {wins}/{len(held)})")
        del model
        torch.cuda.empty_cache()
    write_json(MULTI, {
        "alpha": 1.0,
        "updates_per_cell": updates_per_cell,
        "n_groups": len(groups),
        "seconds": seconds,
        "per_seed": per_seed,
        "note": "One corrector, three fixed-lr stage-1 maps, each scored on its own held-out cells.",
    })
    log("MULTISEED DONE")


def math_log10(value: float) -> float:
    import math
    return float(math.log10(value))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    four_row()
    train_multiseed()
    log("REMAINING DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
