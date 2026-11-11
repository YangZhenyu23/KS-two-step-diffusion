"""Two-step diffusion on the full two-parameter grid: MeanFlow + DP refinement.

What this closes
----------------
Earlier the two-parameter results reported MeanFlow alone. The method this
project studies is *two-step*: a one-step MeanFlow map followed by a near-identity
DP corrector, so the headline numbers must include the DP stage. This script
trains the corrector on the converged MF checkpoint and reports both stages.

Protocol
--------
    stage 1   MF   : x_mf = x - u_theta(x, r=0, t=1, log sigma, log mu)   (frozen)
    stage 2   DP   : x_dp = x_mf + alpha * MLP(x_mf, log sigma, log mu)

The corrector is trained on the *same* 40 LHS cells as the MF model, with the
loss of the upstream implementation:

    L = ||x_dp - x_matched||^2 + lambda * ||x_dp - x_mf||^2

where x_matched is the exact OT re-matching of the target cloud against the
current prediction. The matching must be recomputed whenever the batch changes:
the permutation is indexed by position inside the batch, so a stale coupling
would silently train the corrector against the wrong particles. This was a real
bug earlier in the project and the guard is kept here.

Evaluation
----------
Histogram W2 (16 bins/axis, per-cell grid, exact OT with certificate) is used
throughout, matching the metric of the rest of the repo. The script reports:

    MF alone            baseline
    MF + DP             two-step
    improvement         ratio

on the same 143 held-out cells used by every other ablation.

Usage:
    python scripts/dp_two_param.py --updates-per-cell 2000 --alpha 1.0 --ot-refresh 10
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
from torch import nn
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fullscale import grid as G  # noqa: E402
from fullscale.evaluate import score_many, write_csv, write_json  # noqa: E402
from fullscale.model import sample_two_param  # noqa: E402
from fullscale.train import load_cell, load_two  # noqa: E402

from sample_efficiency import frozen_test_set, stratified_subset  # noqa: E402

OUT = G.RESULTS / "dp_two_param"
DEFAULT_MF = G.RESULTS / "optimal_budget_ablation" / "two_32000" / "mf_two_param.pt"


class TwoParamCorrector(nn.Module):
    """Near-identity ResNet corrector conditioned on both parameters.

    Mirrors the upstream DPCorrector: width 64, residual blocks of
    LayerNorm/SiLU, zero-initialised output head so the map starts as the
    identity.
    """

    def __init__(self, width: int = 64, alpha: float = 1.0, blocks: int = 2,
                 dimension: int = 3):
        super().__init__()
        self.alpha = float(alpha)
        self.dimension = int(dimension)
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
                "condition_mode": "log10_sigma_log10_mu",
                "input_features": self.dimension + 2,
                "output": "x + alpha * MLP(x, cond)"}


def exact_assignment(current: np.ndarray, target: np.ndarray):
    """Exact OT permutation between equal-size clouds, with a permutation check."""
    import ot

    cost = cdist(current, target, metric="sqeuclidean")
    n = len(cost)
    w = np.full(n, 1.0 / n)
    plan, info = ot.emd(w, w, cost, numItermax=2_000_000, log=True)
    code = int(info.get("result_code", -1))
    if code != 1:
        raise RuntimeError(f"exact OT failed: {info.get('warning')}")
    coupling = n * np.asarray(plan, dtype=np.float64)
    columns = coupling.argmax(axis=1)
    perm = np.zeros_like(coupling)
    perm[np.arange(n), columns] = 1.0
    if (not np.array_equal(np.sort(columns), np.arange(n))
            or not np.allclose(coupling, perm, rtol=0, atol=1e-8)):
        raise RuntimeError("OT coupling is not a permutation")
    return columns, float(np.sum(cost * coupling) / n), code


def train_corrector(mf_model, cells, config, *, updates_per_cell: int,
                    batch: int, alpha: float, blocks: int,
                    residual_weight: float, ot_refresh: int, lr: float,
                    seed: int, log=print) -> dict:
    device = torch.device("cuda")

    log(f"pre-computing MF outputs for {len(cells)} cells")
    mf_clouds, targets = [], []
    for cell in cells:
        payload = load_cell(cell)
        source = torch.from_numpy(payload["clouds"]["source"])
        with torch.no_grad():
            mf = sample_two_param(mf_model, source, cell.sigma, cell.mu)
        mf_clouds.append(mf.to(device))
        targets.append(torch.from_numpy(payload["clouds"]["reference"]).to(device))
    log("  done")

    log_sigma = torch.tensor([math.log10(c.sigma) for c in cells],
                             dtype=torch.float32, device=device)
    log_mu = torch.tensor([math.log10(c.mu) for c in cells],
                          dtype=torch.float32, device=device)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TwoParamCorrector(width=64, alpha=alpha, blocks=blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total = updates_per_cell * len(cells)
    log(f"training corrector: {len(cells)} cells x {updates_per_cell} = {total} steps, "
        f"batch {batch}, alpha {alpha}")

    rows = []
    started = time.perf_counter()
    matched = None
    batch_idx = None
    env_fixed = 0
    for step in range(1, total + 1):
        model.train()
        env = int(torch.randint(len(cells), (1,)).item())

        # The permutation indexes positions inside the batch, so the batch must
        # stay fixed while a matching is in use. Refreshing the batch together
        # with the coupling is what keeps the target and the permutation aligned.
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
            perm, ot_cost, code = exact_assignment(current, x_tgt.cpu().numpy())
            matched = torch.as_tensor(perm, dtype=torch.long, device=device)
        else:
            x_mf = mf_clouds[env_fixed][batch_idx]
            x_tgt = targets[env_fixed][batch_idx]
            cs = log_sigma[env_fixed].expand(batch, 1)
            cm = log_mu[env_fixed].expand(batch, 1)

        refined = model(x_mf, cs, cm)
        transport = (refined - x_tgt[matched]).square().sum(-1).mean()
        residual = (refined - x_mf).square().sum(-1).mean()
        loss = transport + residual_weight * residual
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % max(total // 10, 1) == 0 or step == total:
            rows.append({"step": step, "loss": float(loss.item()),
                         "transport": float(transport.item()),
                         "residual": float(residual.item())})
            log(f"  step {step}/{total} loss={loss.item():.5g} "
                f"transport={transport.item():.5g} residual={residual.item():.5g}")

    seconds = time.perf_counter() - started
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return {"model": model, "seconds": seconds, "rows": rows,
            "total_steps": total}


@torch.no_grad()
def apply_corrector(model, cloud: torch.Tensor, sigma: float, mu: float) -> np.ndarray:
    device = next(model.parameters()).device
    x = cloud.to(device)
    n = len(x)
    ls = torch.full((n, 1), math.log10(float(sigma)), dtype=x.dtype, device=device)
    lm = torch.full((n, 1), math.log10(float(mu)), dtype=x.dtype, device=device)
    return model(x, ls, lm).cpu().numpy().astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates-per-cell", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--residual-weight", type=float, default=1.0)
    ap.add_argument("--ot-refresh", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=88)
    ap.add_argument("--mf-checkpoint", type=Path, default=None,
                    help="MF weights to correct; defaults to the seed-88 model, "
                         "but each DP seed must use its own MF model, otherwise the "
                         "seed pairing is meaningless")
    a = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    out_dir = OUT / f"s{a.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = lambda t: print(f"[{time.strftime('%H:%M:%S')}] {t}", flush=True)
    config = G  # namespace for n_train

    from fullscale.grid import Config
    cfg = Config(n_train=G.N_PARTICLES, device="cuda")

    test, pool = frozen_test_set()
    design = stratified_subset(pool, 40, seed=20261107)
    design_keys = {c.key for c in design}
    slice_keys = {c.key for c in G.baseline_slice()}
    held = [c for c in G.full_grid()
            if c.key not in design_keys and c.key not in slice_keys]
    log(f"train cells {len(design)}, held-out cells {len(held)}")

    mf_path = a.mf_checkpoint
    if mf_path is None:
        # default to the MF model trained with the same seed, so the MF/DP pair
        # shares one run; fall back to the seed-88 model only for seed 88
        candidate = (G.RESULTS / "optimal_budget_ablation" / f"two_32000_s{a.seed}"
                     / "mf_two_param.pt")
        mf_path = candidate if candidate.is_file() else DEFAULT_MF
    if not mf_path.is_file():
        raise SystemExit(f"MF checkpoint not found: {mf_path}")
    mf_model, mf_payload = load_two(mf_path, "cuda")
    log(f"MF checkpoint: {mf_path.relative_to(ROOT)}")
    if mf_payload.get("seed") not in (None, a.seed):
        log(f"WARNING: MF checkpoint was trained with seed {mf_payload.get('seed')}, "
            f"but this DP run uses seed {a.seed}; the pair is mixed")

    result = train_corrector(mf_model, design, cfg,
                             updates_per_cell=a.updates_per_cell, batch=a.batch,
                             alpha=a.alpha, blocks=a.blocks,
                             residual_weight=a.residual_weight,
                             ot_refresh=a.ot_refresh, lr=a.lr,
                             seed=a.seed, log=log)
    corrector = result["model"]

    torch.save({"model": {k: v.detach().cpu().clone()
                          for k, v in corrector.state_dict().items()},
                "architecture": corrector.architecture(),
                "updates_per_cell": a.updates_per_cell, "batch": a.batch,
                "alpha": a.alpha, "blocks": a.blocks,
                "ot_refresh": a.ot_refresh, "lr": a.lr, "seed": a.seed},
               out_dir / "dp_corrector.pt")

    # ---- evaluate: MF alone vs MF + DP, histogram W2 --------------------
    log(f"scoring {len(held)} held-out cells")
    pairs_mf, pairs_dp, metas = {}, {}, []
    for position, cell in enumerate(held, start=1):
        payload = load_cell(cell)
        source = torch.from_numpy(payload["clouds"]["source"])
        reference = payload["clouds"]["reference"]
        mf = sample_two_param(mf_model, source, cell.sigma, cell.mu).numpy().astype(np.float32)
        dp = apply_corrector(corrector, torch.from_numpy(mf), cell.sigma, cell.mu)
        pairs_mf[cell.key] = (reference, mf)
        pairs_dp[cell.key] = (reference, dp)
        metas.append(cell.meta())
        if position % 40 == 0:
            log(f"  generated {position}/{len(held)}")

    rows_mf = score_many(pairs_mf, bins=G.BINS, workers=4, log=lambda t: None)
    rows_dp = score_many(pairs_dp, bins=G.BINS, workers=4, log=lambda t: None)
    by_mf = {r["key"]: r["w2"] for r in rows_mf}
    by_dp = {r["key"]: r["w2"] for r in rows_dp}

    detail = []
    for meta in metas:
        w_mf, w_dp = by_mf[meta["key"]], by_dp[meta["key"]]
        detail.append({**meta, "w2_mf": w_mf, "w2_mf_dp": w_dp,
                       "improvement": w_mf - w_dp,
                       "ratio": w_dp / w_mf if w_mf else None,
                       "mu_off_baseline": abs(math.log10(meta["mu"]))})
    detail.sort(key=lambda r: (r["mu_off_baseline"], r["sigma"]))
    write_csv(out_dir / "detail.csv", detail)

    m_mf = float(np.mean([r["w2_mf"] for r in detail]))
    m_dp = float(np.mean([r["w2_mf_dp"] for r in detail]))
    gains = np.array([r["improvement"] for r in detail])

    bands = {}
    for r in detail:
        bands.setdefault(round(r["mu_off_baseline"], 2), []).append(r)
    band_summary = {}
    print(f"\n{'|log10 mu|':>11} {'n':>4} {'MF':>9} {'MF+DP':>9} {'improvement':>12}")
    for band in sorted(bands):
        group = bands[band]
        a_mf = float(np.mean([r["w2_mf"] for r in group]))
        a_dp = float(np.mean([r["w2_mf_dp"] for r in group]))
        band_summary[str(band)] = {"n": len(group), "mf": a_mf, "mf_dp": a_dp,
                                   "gain": a_mf - a_dp}
        print(f"{band:11.2f} {len(group):4d} {a_mf:9.5f} {a_dp:9.5f} "
              f"{a_mf - a_dp:+12.5f}")

    summary = {
        "updates_per_cell": a.updates_per_cell, "batch": a.batch,
        "alpha": a.alpha, "blocks": a.blocks, "ot_refresh": a.ot_refresh,
        "lr": a.lr, "seed": a.seed, "n_held_out": len(detail),
        "mf_mean_w2": m_mf, "mf_dp_mean_w2": m_dp,
        "relative_reduction": float(1 - m_dp / m_mf),
        "mf_over_floor": m_mf / G.FLOOR_16,
        "mf_dp_over_floor": m_dp / G.FLOOR_16,
        "dp_wins": int((gains > 0).sum()),
        "mf_wins": int((gains < 0).sum()),
        "median_improvement": float(np.median(gains)),
        "corr_gain_vs_mu_distance": float(
            np.corrcoef([r["mu_off_baseline"] for r in detail], gains)[0, 1]),
        "by_mu_band": band_summary,
        "training_seconds": result["seconds"],
        "noise_floor": G.FLOOR_16,
    }
    write_json(out_dir / "summary.json", summary)

    print(f"\nnoise floor {G.FLOOR_16:.5f}")
    print(f"  MF        {m_mf:.5f}  ({m_mf / G.FLOOR_16:.2f}x floor)")
    print(f"  MF + DP   {m_dp:.5f}  ({m_dp / G.FLOOR_16:.2f}x floor)")
    print(f"  reduction {summary['relative_reduction']:+.1%}   "
          f"DP wins {summary['dp_wins']}/{len(detail)}")
    print(f"  corr(gain, |log10 mu|) = {summary['corr_gain_vs_mu_distance']:.3f}")
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

