"""Run the remaining Harry follow-ups after the seed-88 warmcos job finishes.

Timing choice, measured on this machine:

- A MeanFlow update is launch-bound: 3.6 ms at batch 4096 and also at 16384.
  Replaying a CUDA graph is 0.80 ms. Later stage-1 updates use that replay.
- An exact 2000-point assignment is 1.89 s with POT's network simplex and 0.38 s
  with SciPy's Hungarian algorithm, and the optimal cost matches. The fine-tune
  uses Hungarian. Graphing the tiny network step beside it would not shorten
  the job, because the assignment is the clock.
"""
from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from torch.func import jvp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fullscale import grid as G  # noqa: E402
from fullscale.evaluate import score_many, write_json  # noqa: E402
from fullscale.model import TwoParamMeanflow, meanflow_loss, sample_two_param  # noqa: E402
from fullscale.train import _save, load_cell, load_training_tensors, load_two  # noqa: E402
from dp_two_param import TwoParamCorrector, apply_corrector  # noqa: E402
from floor_anatomy import sliced_w2  # noqa: E402
from harry_warmcos_s88 import (  # noqa: E402
    HEADLINE, cells_from_checkpoint, held_out, sample_times_cuda, train_warmcos,
)

OUT = G.RESULTS / "harry_followup"
WARM = OUT / "warmcos_s88"
SPLIT = G.RESULTS / "alpha_sweep" / "split_repaired.json"
MF = {
    88: HEADLINE,
    89: G.RESULTS / "optimal_budget_ablation" / "two_32000_s89" / "mf_two_param.pt",
    90: G.RESULTS / "optimal_budget_ablation" / "two_32000_s90" / "mf_two_param.pt",
}
DP = {seed: G.RESULTS / "dp_two_param" / f"s{seed}" / "dp_corrector.pt" for seed in (88, 89, 90)}


def log(text: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {text}"
    print(line, flush=True)
    with (OUT / "queue.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def warmcos_alive() -> bool:
    command = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
        "| Select-Object -ExpandProperty CommandLine"
    )
    try:
        output = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", command],
            text=True, errors="replace",
        )
    except subprocess.CalledProcessError:
        return False
    return "harry_warmcos_s88.py" in output


def wait_for_warmcos() -> None:
    summary = WARM / "summary.json"
    log("waiting for warmcos_s88/summary.json")
    while not summary.is_file():
        if not warmcos_alive():
            raise SystemExit("warmcos training exited before writing summary.json")
        time.sleep(20)
    log("warmcos summary is on disk")


def timed(name: str, fn) -> None:
    log(f"START {name}")
    started = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - started
    log(f"DONE {name} in {elapsed:.0f}s")
    path = OUT / "timings.json"
    previous = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    previous[name] = elapsed
    write_json(path, previous)


def assignment(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    cost = cdist(current, target, metric="sqeuclidean")
    rows, cols = linear_sum_assignment(cost)
    perm = np.empty(len(cost), dtype=np.int64)
    perm[rows] = cols
    return perm


def continue_fixed(cells, steps: int, out_dir: Path) -> None:
    """80,000 more MeanFlow updates at the fixed learning rate, CUDA-graph replay."""
    device = torch.device("cuda")
    model, _ = load_two(HEADLINE, "cuda")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.train()
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
    for index in range(4, steps):
        graph.replay()
        if (index + 1) % 20000 == 0 or index + 1 == steps:
            log(f"  extra MF step {index + 1}/{steps}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save(out_dir / "mf_two_param.pt", model, cells, G.Config(), steps, 88)


def finetune_ot(cells, steps: int, out_dir: Path) -> None:
    """Fine-tune the one-step map on exact minibatch assignments. No second network."""
    device = torch.device("cuda")
    model, _ = load_two(HEADLINE, "cuda")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.train()
    initial, targets = load_training_tensors(cells, device)
    log_sigma = torch.tensor([math.log10(cell.sigma) for cell in cells], device=device)
    log_mu = torch.tensor([math.log10(cell.mu) for cell in cells], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    batch = 2000
    refresh = 10
    matched = None
    env_fixed = 0
    idx = None
    for step in range(1, steps + 1):
        if idx is None or (step - 1) % refresh == 0:
            env_fixed = int(torch.randint(len(cells), (1,)).item())
            idx = torch.randperm(G.N_PARTICLES, device=device)[:batch]
            x_source = initial[idx]
            y = targets[env_fixed][idx]
            cs = log_sigma[env_fixed].expand(batch, 1)
            cm = log_mu[env_fixed].expand(batch, 1)
            r = torch.zeros(batch, 1, device=device)
            t = torch.ones(batch, 1, device=device)
            with torch.no_grad():
                pred = (x_source - model(x_source, r, t, cs, cm)).detach().cpu().numpy()
            matched = torch.as_tensor(
                assignment(pred, y.detach().cpu().numpy()), dtype=torch.long, device=device)
        else:
            x_source = initial[idx]
            y = targets[env_fixed][idx]
            cs = log_sigma[env_fixed].expand(batch, 1)
            cm = log_mu[env_fixed].expand(batch, 1)
            r = torch.zeros(batch, 1, device=device)
            t = torch.ones(batch, 1, device=device)
        pred = x_source - model(x_source, r, t, cs, cm)
        loss = (pred - y[matched]).square().sum(-1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % max(steps // 10, 1) == 0 or step == steps:
            log(f"  OT fine-tune {step}/{steps} loss={float(loss):.5g}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save(out_dir / "mf_two_param.pt", model, cells, G.Config(), steps, 88)


def score_model(model, cells) -> float:
    pairs = {}
    for cell in cells:
        payload = load_cell(cell)
        source = torch.from_numpy(payload["clouds"]["source"])
        pred = sample_two_param(model, source, cell.sigma, cell.mu).numpy().astype(np.float32)
        pairs[cell.key] = (payload["clouds"]["reference"], pred)
    rows = score_many(pairs, bins=G.BINS, workers=4, log=lambda text: None)
    return float(np.mean([row["w2"] for row in rows]))


def budget_controls() -> None:
    cells = cells_from_checkpoint(HEADLINE)
    held = held_out(cells)
    steps = 2000 * len(cells)
    extra = OUT / "extra_updates_s88"
    ot = OUT / "ot_finetune_s88"
    continue_fixed(cells, steps, extra)
    finetune_ot(cells, steps, ot)
    extra_model, _ = load_two(extra / "mf_two_param.pt", "cuda")
    ot_model, _ = load_two(ot / "mf_two_param.pt", "cuda")
    base, _ = load_two(HEADLINE, "cuda")
    write_json(OUT / "budget_controls.json", {
        "n_held_out": len(held),
        "extra_updates": steps,
        "headline_mf": score_model(base, held),
        "extra_mf_updates": score_model(extra_model, held),
        "ot_finetune_same_network": score_model(ot_model, held),
        "note": "OT uses SciPy linear_sum_assignment. On a 2000-point batch it "
                "matched POT emd's optimal cost and took 0.38 s instead of 1.89 s.",
    })


def consistency() -> None:
    cells = cells_from_checkpoint(HEADLINE)
    held = held_out(cells)
    model, _ = load_two(HEADLINE, "cuda")
    device = torch.device("cuda")
    draw_r, draw_t = sample_times_cuda(1_000_000, device)
    gap = (draw_t - draw_r).squeeze()
    pairs_between = {}
    gaps = []
    for cell in held:
        payload = load_cell(cell)
        source = torch.from_numpy(payload["clouds"]["source"])
        one = sample_two_param(model, source, cell.sigma, cell.mu, steps=1).numpy()
        two = sample_two_param(model, source, cell.sigma, cell.mu, steps=2).numpy()
        pairs_between[cell.key] = (one.astype(np.float32), two.astype(np.float32))
        gaps.append(float(np.linalg.norm(one - two, axis=1).mean()))
    rows = score_many(pairs_between, bins=G.BINS, workers=4, log=lambda text: None)
    write_json(OUT / "consistency.json", {
        "n_held_out": len(held),
        "time_samples": 1_000_000,
        "fraction_interval_shorter_than_half": float((gap < 0.5).float().mean()),
        "fraction_diagonal_rt": float(torch.isclose(draw_r, draw_t).float().mean()),
        "mean_particle_gap": float(np.mean(gaps)),
        "mean_w2_one_vs_two": float(np.mean([row["w2"] for row in rows])),
    })


def load_dp(seed: int, alpha: float, device: str):
    checkpoint = torch.load(DP[seed], map_location=device, weights_only=False)
    arch = checkpoint.get("architecture", {})
    model = TwoParamCorrector(
        width=arch.get("width", 64), alpha=alpha,
        blocks=checkpoint.get("blocks", arch.get("blocks", 2)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.alpha = float(alpha)
    model.eval()
    return model


def moments(cloud: np.ndarray) -> dict:
    centred = cloud - cloud.mean(0)
    eigenvalues = np.sort(np.linalg.eigvalsh(np.cov(centred.T)))[::-1]
    std = np.sqrt(np.maximum(eigenvalues, 0.0))
    return {
        "centroid_x": float(cloud[:, 0].mean()),
        "rms_radius": float(np.sqrt((centred ** 2).sum(1).mean())),
        "anisotropy": float(std[0] / std[1]) if std[1] else None,
    }


def alpha_metrics_physics() -> None:
    device = "cuda"
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    test_keys = set(split["test"])
    alpha_rows = {}
    for seed in (89, 90):
        mf, _ = load_two(MF[seed], device)
        held = held_out(cells_from_checkpoint(MF[seed]))
        chosen = [cell for cell in held if cell.key in test_keys]
        pairs = {0.0: {}, 1.0: {}, 2.0: {}}
        correctors = {alpha: load_dp(seed, alpha, device) for alpha in (1.0, 2.0)}
        for cell in chosen:
            payload = load_cell(cell)
            source = torch.from_numpy(payload["clouds"]["source"])
            reference = payload["clouds"]["reference"]
            base = sample_two_param(mf, source, cell.sigma, cell.mu).numpy().astype(np.float32)
            pairs[0.0][cell.key] = (reference, base)
            for alpha, corrector in correctors.items():
                pred = apply_corrector(corrector, torch.from_numpy(base), cell.sigma, cell.mu)
                pairs[alpha][cell.key] = (reference, pred)
        totals = {
            alpha: [row["w2"] for row in score_many(group, bins=G.BINS, workers=4, log=lambda text: None)]
            for alpha, group in pairs.items()
        }
        means = {str(alpha): float(np.mean(values)) for alpha, values in totals.items()}
        means["n_test"] = len(chosen)
        means["gain_alpha2"] = float(1.0 - means["2.0"] / means["0.0"])
        alpha_rows[str(seed)] = means
        log(f"  seed {seed} test n={len(chosen)} alpha2 gain {means['gain_alpha2']:+.1%}")
    write_json(OUT / "alpha_seeds.json", alpha_rows)

    mf, _ = load_two(MF[88], device)
    corrector = load_dp(88, 1.0, device)
    held = held_out(cells_from_checkpoint(MF[88]))
    sliced = {"mf": [], "mf_dp": []}
    physical = {"mf": [], "mf_dp": []}
    for cell in held:
        payload = load_cell(cell)
        source = torch.from_numpy(payload["clouds"]["source"])
        reference = payload["clouds"]["reference"]
        base = sample_two_param(mf, source, cell.sigma, cell.mu).numpy().astype(np.float64)
        refined = apply_corrector(corrector, torch.from_numpy(base.astype(np.float32)),
                                   cell.sigma, cell.mu).astype(np.float64)
        reference = reference.astype(np.float64)
        sliced["mf"].append(sliced_w2(reference, base, projections=64, seed=88))
        sliced["mf_dp"].append(sliced_w2(reference, refined, projections=64, seed=88))
        ref_m = moments(reference)
        for name, cloud in (("mf", base), ("mf_dp", refined)):
            got = moments(cloud)
            physical[name].append({
                key: abs(got[key] - ref_m[key]) / abs(ref_m[key]) if ref_m[key] else None
                for key in ("centroid_x", "rms_radius", "anisotropy")
            })
    def average(rows):
        return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    write_json(OUT / "particle_and_physics.json", {
        "n_held_out": len(held),
        "sliced_w2_projections": 64,
        "sliced_w2_mf": float(np.mean(sliced["mf"])),
        "sliced_w2_mf_dp": float(np.mean(sliced["mf_dp"])),
        "mean_abs_relative_error": {"mf": average(physical["mf"]), "mf_dp": average(physical["mf_dp"])},
        "note": "64 projections, not 256: each projection sorts 1e5 particles, and 64 "
                "already compares the two clouds on the same directions. Energy distance "
                "was not used; it is quadratic in the particle count.",
    })


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    wait_for_warmcos()
    timed("budget_controls", budget_controls)
    timed("consistency", consistency)
    timed("alpha_sliced_physics", alpha_metrics_physics)
    log("QUEUE DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
