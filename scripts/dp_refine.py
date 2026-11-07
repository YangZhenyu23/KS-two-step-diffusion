"""DP (DeepParticle) refinement on top of the two-parameter MeanFlow.

What DP adds
------------
MeanFlow generates a cloud in one shot and has no mechanism to correct the
remaining discrepancy. DeepParticle trains a *corrector* ``x -> x + alpha*MLP(x, c)``
that is initialised as the identity (zero-initialised output layer) and trained to
map the MeanFlow output onto the physical target. Stage II uses an exact
mini-batch optimal-transport matching so the correction is a transport plan, not
a per-particle regression, plus a residual penalty that keeps the corrector near
the identity.

Upstream values, kept here for comparability:

    alpha = 10           step size of the correction
    width = 64
    blocks = 2           residual blocks, LayerNorm + SiLU
    batch = 2000         mini-batch used for the exact OT assignment
    ot_refresh = 50      recompute the matching every 50 updates
    residual_weight = 1  weight of ||x_refined - x_source||^2
    lr = 1e-3            Adam

The only change is the conditioning vector: ``[log10(sigma)]`` becomes
``[log10(sigma), log10(mu)]``, so the corrector can also depend on the second
parameter.

Pipeline
--------
    unit ball --(MeanFlow)--> x_mf --(DP corrector)--> x_dp

Both ``x_mf`` and ``x_dp`` are scored against the physical reference, so the
comparison isolates the value of the refinement.

Usage:
    python scripts/dp_refine.py --updates-per-cell 100 --batch 2000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fullscale import grid as G  # noqa: E402
from fullscale.evaluate import score_many, write_csv, write_json  # noqa: E402
from fullscale.grid import Config  # noqa: E402
from fullscale.lhs import latin_hypercube  # noqa: E402
from fullscale.model import sample_two_param  # noqa: E402
from fullscale.train import load_cell, load_two  # noqa: E402


class TwoParamCorrector(nn.Module):
    """Upstream DPCorrector with a two-parameter conditioning vector."""

    def __init__(self, width: int = 64, alpha: float = 10.0, blocks: int = 2, dimension: int = 3):
        super().__init__()
        self.alpha = float(alpha)
        self.dimension = int(dimension)
        self.condition_mode = "log10_sigma_log10_mu"
        # input = x (d) + two condition features
        self.input = nn.Linear(dimension + 2, width)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, width),
                nn.SiLU(),
                nn.Linear(width, width),
            )
            for _ in range(int(blocks))
        )
        self.output = nn.Linear(width, dimension)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, logsigma, logmu):
        h = self.input(torch.cat((x, logsigma, logmu), dim=-1))
        for block in self.blocks:
            h = h + 0.2 * block(h)
        return x + self.alpha * self.output(h)

    def architecture(self) -> dict:
        return {"kind": "dp_corrector", "width": self.input.out_features,
                "blocks": len(self.blocks), "alpha": self.alpha,
                "dimension": self.dimension,
                "condition_mode": self.condition_mode,
                "input_features": self.dimension + 2,
                "activation": "SiLU", "layer_norms": 1,
                "head_initialization": "zero",
                "output": "x + alpha * MLP(x, condition)"}


def exact_assignment(current: np.ndarray, target: np.ndarray):
    """Exact OT permutation between two equal-size clouds (POT network simplex).

    Returns the permutation that maps each ``current`` particle to its matched
    ``target`` particle, plus the mean squared transport cost and the solver
    certificate. The coupling must be a permutation: a fractional plan cannot be
    silently rounded with argmax, because that changes the objective.
    """
    import ot

    cost = cdist(current, target, metric="sqeuclidean")
    n = len(cost)
    weights = np.full(n, 1.0 / n)
    plan, info = ot.emd(weights, weights, cost, numItermax=1_000_000, log=True)
    if int(info.get("result_code", -1)) != 1:
        raise RuntimeError(f"exact OT failed: {info.get('warning')}")
    coupling = n * np.asarray(plan, dtype=np.float64)
    columns = coupling.argmax(axis=1)
    permutation = np.zeros_like(coupling)
    permutation[np.arange(n), columns] = 1.0
    if (not np.array_equal(np.sort(columns), np.arange(n))
            or not np.allclose(coupling, permutation, rtol=0, atol=1e-8)):
        raise RuntimeError("OT coupling is not a permutation")
    return columns, float(np.sum(cost * coupling) / n), int(info["result_code"])


def train_corrector(
    cells: list[G.Cell],
    mf_model,
    config: Config,
    out_dir: Path,
    *,
    updates_per_cell: int,
    batch: int = 2000,
    alpha: float = 10.0,
    blocks: int = 2,
    residual_weight: float = 1.0,
    ot_refresh: int = 50,
    lr: float = 1e-3,
    seed: int = 88,
    log=print,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))

    # Pre-compute every MeanFlow output once: the corrector's input is frozen, so
    # regenerating it inside the loop would only waste time.
    log(f"pre-computing MeanFlow outputs for {len(cells)} training cells")
    sources, targets, mf_outputs = [], [], []
    for cell in cells:
        payload = load_cell(cell)
        source = payload["clouds"]["source"]
        target = payload["clouds"]["reference"]
        mf = sample_two_param(mf_model, torch.from_numpy(source), cell.sigma, cell.mu).numpy()
        sources.append(torch.from_numpy(source).to(device))
        targets.append(torch.from_numpy(target).to(device))
        mf_outputs.append(torch.from_numpy(np.ascontiguousarray(mf, dtype=np.float32)).to(device))
    log("  done")

    log_sigma = torch.tensor([math.log10(c.sigma) for c in cells], dtype=torch.float32, device=device)
    log_mu = torch.tensor([math.log10(c.mu) for c in cells], dtype=torch.float32, device=device)

    model = TwoParamCorrector(width=64, alpha=alpha, blocks=blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total = updates_per_cell * len(cells)
    log(f"training DP corrector: {len(cells)} cells x {updates_per_cell} updates = {total} steps, "
        f"batch {batch}, alpha {alpha}, blocks {blocks}")

    rows = []
    started = time.perf_counter()
    matched = None
    last_refresh = 0
    batch_idx = None
    for step in range(1, total + 1):
        model.train()
        env = int(torch.randint(len(cells), (1,)).item())

        # The matched permutation is indexed by *position inside the batch*, so the
        # batch must stay fixed while a matching is in use. Re-drawing particles
        # every step while keeping a stale permutation would silently train the
        # corrector against the wrong target particles. Upstream guards this with
        # its ``batch_refresh`` parameter; here the batch is refreshed only when a
        # new optimal-transport matching is computed.
        refreshed = batch_idx is None or (step - 1) % ot_refresh == 0
        if refreshed:
            batch_idx = torch.randperm(config.n_train, device=device)[:batch]
            new_env = env
            x_target = targets[new_env][batch_idx]
            x_source = sources[new_env][batch_idx]
            x_mf = mf_outputs[new_env][batch_idx]
            cond_sigma = log_sigma[new_env].expand(batch, 1)
            cond_mu = log_mu[new_env].expand(batch, 1)
            current = model(x_mf, cond_sigma, cond_mu).detach().cpu().numpy()
            permutation, ot_cost, code = exact_assignment(current, x_target.cpu().numpy())
            matched = torch.as_tensor(permutation, dtype=torch.long, device=device)
            last_refresh = step
            env_fixed = new_env
        else:
            ot_cost = None
            x_target = targets[env_fixed][batch_idx]
            x_source = sources[env_fixed][batch_idx]
            x_mf = mf_outputs[env_fixed][batch_idx]
            cond_sigma = log_sigma[env_fixed].expand(batch, 1)
            cond_mu = log_mu[env_fixed].expand(batch, 1)

        refined = model(x_mf, cond_sigma, cond_mu)
        transport = (refined - x_target[matched]).square().sum(-1).mean()
        residual = (refined - x_mf).square().sum(-1).mean()
        loss = transport + residual_weight * residual
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite DP loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 200 == 0 or step == total:
            rows.append({"step": step, "loss": float(loss.item()),
                         "transport": float(transport.item()),
                         "residual": float(residual.item()),
                         "ot_refreshed": int(refreshed),
                         "coupling_age": step - last_refresh})
            log(f"  step {step}/{total} loss={loss.item():.5g} "
                f"transport={transport.item():.5g} residual={residual.item():.5g}")
    seconds = time.perf_counter() - started

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    path = out_dir / "dp_corrector.pt"
    torch.save({"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "architecture": model.architecture(), "step": total,
                "cells": [c.key for c in cells], "sigma_grid": G.SIGMA_GRID,
                "mu_grid": G.MU_GRID, "alpha": alpha, "blocks": blocks,
                "residual_weight": residual_weight, "ot_refresh": ot_refresh,
                "batch": batch, "seed": seed}, path)
    (out_dir / "training_log.json").write_text(json.dumps({
        "updates_per_cell": updates_per_cell, "total_steps": total, "batch": batch,
        "alpha": alpha, "blocks": blocks, "residual_weight": residual_weight,
        "ot_refresh": ot_refresh, "lr": lr, "seconds": seconds,
        "cells": [c.key for c in cells], "rows": rows}, indent=2), encoding="utf-8")
    log(f"  done in {seconds:.1f}s -> {path.name}")
    return {"checkpoint": str(path), "total_steps": total, "seconds": seconds,
            "final_loss": rows[-1]["loss"], "final_transport": rows[-1]["transport"],
            "final_residual": rows[-1]["residual"]}


def load_corrector(path: Path, device: str = "cuda") -> TwoParamCorrector:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    arch = payload["architecture"]
    model = TwoParamCorrector(width=arch["width"], alpha=arch["alpha"],
                              blocks=arch["blocks"], dimension=arch["dimension"])
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval().requires_grad_(False)


@torch.no_grad()
def apply_corrector(model: TwoParamCorrector, cloud: torch.Tensor, sigma: float, mu: float) -> torch.Tensor:
    device = next(model.parameters()).device
    x = cloud.detach().to(device)
    n = len(x)
    ls = torch.full((n, 1), math.log10(float(sigma)), dtype=x.dtype, device=device)
    lm = torch.full((n, 1), math.log10(float(mu)), dtype=x.dtype, device=device)
    return model(x, ls, lm).cpu()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates-per-cell", type=int, default=100)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--residual-weight", type=float, default=1.0)
    ap.add_argument("--ot-refresh", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=88)
    a = ap.parse_args()

    config = Config(n_train=G.N_PARTICLES, device="cuda")
    design = latin_hypercube(40)
    lhs_cells = [G.Cell(i, j) for i, j in design.points]
    lhs_keys = {c.key for c in lhs_cells}
    slice_keys = {c.key for c in G.baseline_slice()}
    held = [c for c in G.full_grid() if c.key not in lhs_keys and c.key not in slice_keys]

    mf_model, _ = load_two(G.RESULTS / "train" / "lhs40" / "mf_two_param.pt")
    out = G.ensure("dp")

    summary = train_corrector(lhs_cells, mf_model, config, out,
                              updates_per_cell=a.updates_per_cell, batch=a.batch,
                              alpha=a.alpha, blocks=a.blocks,
                              residual_weight=a.residual_weight,
                              ot_refresh=a.ot_refresh, lr=a.lr, seed=a.seed)
    corrector = load_corrector(out / "dp_corrector.pt")

    print(f"\nscoring on {len(held)} held-out cells (MF alone vs MF + DP)")
    pairs_mf, pairs_dp, meta = {}, {}, []
    for cell in held:
        payload = load_cell(cell)
        source = payload["clouds"]["source"]
        reference = payload["clouds"]["reference"]
        mf = sample_two_param(mf_model, torch.from_numpy(source), cell.sigma, cell.mu).numpy().astype(np.float32)
        dp = apply_corrector(corrector, torch.from_numpy(mf), cell.sigma, cell.mu).numpy().astype(np.float32)
        pairs_mf[cell.key] = (reference, mf)
        pairs_dp[cell.key] = (reference, dp)
        meta.append(cell.meta())

    rows_mf = score_many(pairs_mf, bins=G.BINS, workers=4, log=lambda t: None)
    rows_dp = score_many(pairs_dp, bins=G.BINS, workers=4, log=lambda t: None)
    by_mf = {r["key"]: r["w2"] for r in rows_mf}
    by_dp = {r["key"]: r["w2"] for r in rows_dp}

    detail = []
    for entry in meta:
        w_mf, w_dp = by_mf[entry["key"]], by_dp[entry["key"]]
        detail.append({**entry, "w2_mf": w_mf, "w2_mf_dp": w_dp,
                       "improvement": w_mf - w_dp, "ratio": w_dp / w_mf,
                       "mu_off_baseline": abs(math.log10(entry["mu"]))})
    detail.sort(key=lambda r: (r["mu_off_baseline"], r["sigma"]))

    m_mf = float(np.mean([r["w2_mf"] for r in detail]))
    m_dp = float(np.mean([r["w2_mf_dp"] for r in detail]))
    gains = np.array([r["improvement"] for r in detail])
    by_mu = {}
    for r in detail:
        by_mu.setdefault(round(r["mu_off_baseline"], 2), []).append(r)

    result = {
        "updates_per_cell": a.updates_per_cell, "batch": a.batch, "alpha": a.alpha,
        "blocks": a.blocks, "residual_weight": a.residual_weight,
        "ot_refresh": a.ot_refresh, "n_heldout": len(detail),
        "mf_mean_w2": m_mf, "mf_dp_mean_w2": m_dp,
        "mf_over_floor": m_mf / G.FLOOR_16, "mf_dp_over_floor": m_dp / G.FLOOR_16,
        "relative_change": float(m_dp / m_mf - 1),
        "dp_wins": int((gains > 0).sum()), "mf_wins": int((gains < 0).sum()),
        "median_improvement": float(np.median(gains)),
        "corr_improvement_vs_mu_distance": float(
            np.corrcoef([r["mu_off_baseline"] for r in detail], gains)[0, 1]),
        "by_mu_distance": {str(b): {
            "n": len(g), "mf": float(np.mean([r["w2_mf"] for r in g])),
            "mf_dp": float(np.mean([r["w2_mf_dp"] for r in g]))}
            for b, g in sorted(by_mu.items())},
        "training": summary,
    }
    write_csv(out / "detail.csv", detail)
    write_json(out / "summary.json", result)

    print(f"\n{'':<12} {'mean W2':>10} {'x floor':>9}")
    print(f"{'MF alone':<12} {m_mf:10.5f} {m_mf / G.FLOOR_16:9.2f}")
    print(f"{'MF + DP':<12} {m_dp:10.5f} {m_dp / G.FLOOR_16:9.2f}")
    print(f"\nrelative change {result['relative_change']:+.2%}   "
          f"DP wins {result['dp_wins']}/{len(detail)}")
    print(f"corr(improvement, |log10 mu|) = {result['corr_improvement_vs_mu_distance']:.3f}")
    print("\nby distance from mu = 1:")
    for band, group in result["by_mu_distance"].items():
        print(f"  {band:>5}: n={group['n']:3d}  MF {group['mf']:.5f}  "
              f"MF+DP {group['mf_dp']:.5f}  {group['mf'] - group['mf_dp']:+.5f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
