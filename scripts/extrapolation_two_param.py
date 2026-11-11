"""Two-parameter extrapolation: train on the box interior, test outside it.

Split
-----
The 20x10 parameter grid is cut into an interior and three extrapolation zones:

    interior        sigma <= 150  and  mu <= 3.98        15 x 9 = 135 cells
    sigma-extrap    sigma >= 160  and  mu <= 3.98         5 x 9 =  45
    mu-extrap       sigma <= 150  and  mu  = 10          15 x 1 =  15
    corner          sigma >= 160  and  mu  = 10           5 x 1 =   5

Nothing in any extrapolation zone is ever seen during training. This is the
question the earlier "high sigma" analysis could not answer, because those
models had trained on sigma = 150 and 200 as LHS cells and were therefore
interpolating.

Sampling
--------
The core method is unchanged: a maximin Latin hypercube over the training box.
From 135 interior cells we take 45, stratified as 3 per sigma level and 5 per mu
level, with the maximin + decorrelation objective used everywhere else. The
design is drawn only from the interior, which is what makes the zones outside
genuinely unseen.

Pipeline
--------
Both stages of the method are trained, because the claim under test is about the
complete two-step pipeline:

    MeanFlow on the 45 LHS cells      (32,000 updates per cell = 1,440,000)
      -> frozen
    DP corrector on the same 45 cells (2,000 updates per cell = 90,000)

Evaluation reports the four zones separately for MF and for MF+DP, so the
degradation from interior to extrapolation is visible, and so is whatever the
corrector recovers.

Usage:
    python scripts/extrapolation_two_param.py --seed 88
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fullscale import grid as G  # noqa: E402
from fullscale.evaluate import score_many, write_csv, write_json  # noqa: E402
from fullscale.grid import Cell, Config  # noqa: E402
from fullscale.model import sample_two_param  # noqa: E402
from fullscale.train import load_cell, load_two, train_two_param  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from dp_two_param import TwoParamCorrector, apply_corrector, train_corrector  # noqa: E402

OUT = G.RESULTS / "extrapolation"

SIGMA_TRAIN_MAX = 150.0
MU_TRAIN_MAX = 3.981071705535   # every mu level except 10.0


def zone_of(cell: Cell) -> str:
    sigma_out = cell.sigma > SIGMA_TRAIN_MAX
    mu_out = cell.mu > MU_TRAIN_MAX + 1e-9
    if sigma_out and mu_out:
        return "corner"
    if sigma_out:
        return "sigma_extrap"
    if mu_out:
        return "mu_extrap"
    return "interior"


def maximin_design(cells: list[Cell], n_sigma_per: int, n_mu_per: int,
                   seed: int) -> list[Cell]:
    """Maximin Latin hypercube over a rectangular sub-grid.

    ``n_sigma_per`` cells are taken from every sigma level and ``n_mu_per`` from
    every mu level, which fixes both marginals. Local swap moves then maximise the
    minimum pairwise distance in the normalised parameter square.
    """
    by_sigma: dict[int, list[Cell]] = {}
    for c in cells:
        by_sigma.setdefault(c.i, []).append(c)
    for group in by_sigma.values():
        group.sort(key=lambda c: c.j)
    sigmas = sorted(by_sigma)
    mu_levels = sorted({c.j for c in cells})
    n_mu = len(mu_levels)

    rng = np.random.default_rng(seed)
    chosen: list[Cell] = []
    for i in sigmas:
        group = by_sigma[i]
        # spread the picks across the mu range deterministically, then shuffle
        step = len(group) / n_sigma_per
        picks = [group[int((k + 0.5) * step) % len(group)] for k in range(n_sigma_per)]
        chosen.extend(picks)
    chosen = list({c.key: c for c in chosen}.values())

    def objective(cs: list[Cell]) -> float:
        arr = np.array([[c.i, c.j / max(n_mu - 1, 1)] for c in cs], dtype=float)
        arr[:, 0] /= max(len(sigmas) - 1, 1)
        diff = arr[:, None, :] - arr[None, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        spacing = float(d.min())
        # keep the mu marginal balanced: without this term the maximin search
        # leaves some mu levels with 3 picks and others with 7
        counts = np.bincount([c.j for c in cs], minlength=n_mu).astype(float)
        imbalance = float(counts.max() - counts.min())
        return spacing - 0.02 * imbalance

    current = objective(chosen)
    for _ in range(20000):
        pos = int(rng.integers(len(chosen)))
        victim = chosen[pos]
        pool = [c for c in by_sigma[victim.i] if c not in chosen]
        if not pool:
            continue
        cand = chosen.copy()
        cand[pos] = pool[int(rng.integers(len(pool)))]
        value = objective(cand)
        if value >= current:
            chosen, current = cand, value
    return sorted(chosen, key=lambda c: (c.i, c.j))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=88)
    ap.add_argument("--updates-per-cell", type=int, default=32000)
    ap.add_argument("--dp-updates-per-cell", type=int, default=2000)
    ap.add_argument("--dp-batch", type=int, default=2000)
    ap.add_argument("--dp-alpha", type=float, default=1.0)
    ap.add_argument("--dp-ot-refresh", type=int, default=10)
    ap.add_argument("--dp-lr", type=float, default=1e-4)
    a = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda t: print(f"[{time.strftime('%H:%M:%S')}] {t}", flush=True)
    cfg = Config(n_train=G.N_PARTICLES, device="cuda")

    all_cells = G.full_grid()
    zones: dict[str, list[Cell]] = {}
    for cell in all_cells:
        zones.setdefault(zone_of(cell), []).append(cell)
    for name, group in zones.items():
        log(f"zone {name}: {len(group)} cells")

    interior = zones["interior"]
    # 45 of the 135 interior cells: 3 per sigma level
    train = maximin_design(interior, n_sigma_per=3, n_mu_per=0, seed=20261111)
    if len(train) != 45:
        raise SystemExit(f"design returned {len(train)} cells, expected 45")
    log(f"LHS training design: {len(train)} cells "
        f"({len({c.i for c in train})} sigma levels, {len({c.j for c in train})} mu levels)")
    log(f"  training sigma range {min(c.sigma for c in train):g}-"
        f"{max(c.sigma for c in train):g}, "
        f"mu range {min(c.mu for c in train):g}-{max(c.mu for c in train):.4g}")

    # ---- stage 1: MeanFlow ----------------------------------------------
    mf_dir = OUT / f"mf_s{a.seed}"
    mf_ckpt = mf_dir / "mf_two_param.pt"
    if mf_ckpt.is_file():
        log("MF checkpoint exists, reusing")
    else:
        log(f"training MF: {len(train)} cells x {a.updates_per_cell}")
        started = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            summary = train_two_param(train, cfg, mf_dir,
                                      steps=a.updates_per_cell * len(train),
                                      batch_size=4096, lr=1e-3, seed=a.seed,
                                      conditioning="log10")
        log(f"  MF done in {time.perf_counter() - started:.0f}s "
            f"(final loss {summary['final_loss']:.4g})")
    mf_model, _ = load_two(mf_ckpt)

    # ---- stage 2: DP corrector ------------------------------------------
    dp_dir = OUT / f"dp_s{a.seed}"
    dp_ckpt = dp_dir / "dp_corrector.pt"
    if dp_ckpt.is_file():
        log("DP checkpoint exists, reusing")
        payload = torch.load(dp_ckpt, map_location="cpu", weights_only=True)
        arch = payload["architecture"]
        corrector = TwoParamCorrector(width=arch["width"], alpha=arch["alpha"],
                                      blocks=arch["blocks"],
                                      dimension=arch["dimension"])
        corrector.load_state_dict(payload["model"], strict=True)
        corrector = corrector.to("cuda").eval().requires_grad_(False)
    else:
        log(f"training DP: {len(train)} cells x {a.dp_updates_per_cell}")
        dp_dir.mkdir(parents=True, exist_ok=True)
        result = train_corrector(mf_model, train, cfg,
                                 updates_per_cell=a.dp_updates_per_cell,
                                 batch=a.dp_batch, alpha=a.dp_alpha, blocks=2,
                                 residual_weight=1.0,
                                 ot_refresh=a.dp_ot_refresh, lr=a.dp_lr,
                                 seed=a.seed, log=log)
        corrector = result["model"]
        torch.save({"model": {k: v.detach().cpu().clone()
                              for k, v in corrector.state_dict().items()},
                    "architecture": corrector.architecture(),
                    "updates_per_cell": a.dp_updates_per_cell,
                    "seed": a.seed}, dp_ckpt)
        log(f"  DP done in {result['seconds']:.0f}s")

    # ---- evaluate every zone --------------------------------------------
    log("scoring all 200 cells (MF and MF+DP)")
    pairs_mf, pairs_dp, meta = {}, {}, []
    for position, cell in enumerate(all_cells, start=1):
        payload = load_cell(cell)
        src = torch.from_numpy(payload["clouds"]["source"])
        ref = payload["clouds"]["reference"]
        mf = sample_two_param(mf_model, src, cell.sigma, cell.mu).numpy().astype(np.float32)
        dp = apply_corrector(corrector, torch.from_numpy(mf), cell.sigma, cell.mu)
        pairs_mf[cell.key] = (ref, mf)
        pairs_dp[cell.key] = (ref, dp)
        meta.append({**cell.meta(), "zone": zone_of(cell)})
        if position % 50 == 0:
            log(f"  generated {position}/{len(all_cells)}")

    rows_mf = score_many(pairs_mf, bins=G.BINS, workers=4, log=lambda t: None)
    rows_dp = score_many(pairs_dp, bins=G.BINS, workers=4, log=lambda t: None)
    by_mf = {r["key"]: r["w2"] for r in rows_mf}
    by_dp = {r["key"]: r["w2"] for r in rows_dp}

    detail = []
    for m in meta:
        w_mf, w_dp = by_mf[m["key"]], by_dp[m["key"]]
        detail.append({**m, "w2_mf": w_mf, "w2_mf_dp": w_dp,
                       "dp_gain": w_mf - w_dp,
                       "dp_relative": 1 - w_dp / w_mf if w_mf else None})
    write_csv(OUT / f"detail_s{a.seed}.csv", detail)

    zone_summary = {}
    print(f"\n{'zone':<14} {'n':>4} {'MF':>9} {'MF+DP':>9} {'DP gain':>9} {'rel':>8}")
    for name in ("interior", "sigma_extrap", "mu_extrap", "corner"):
        group = [r for r in detail if r["zone"] == name]
        if not group:
            continue
        mf = float(np.mean([r["w2_mf"] for r in group]))
        dp = float(np.mean([r["w2_mf_dp"] for r in group]))
        rel = float(np.mean([r["dp_relative"] for r in group]))
        zone_summary[name] = {"n": len(group), "mf": mf, "mf_dp": dp,
                              "dp_gain": mf - dp, "dp_relative": rel}
        print(f"{name:<14} {len(group):4d} {mf:9.5f} {dp:9.5f} {mf - dp:+9.5f} "
              f"{rel:+8.1%}")

    interior_mf = zone_summary["interior"]["mf"]
    interior_dp = zone_summary["interior"]["mf_dp"]
    print(f"\ndegradation from interior (MF)")
    for name in ("sigma_extrap", "mu_extrap", "corner"):
        if name not in zone_summary:
            continue
        z = zone_summary[name]
        print(f"  {name:<14} MF {z['mf'] / interior_mf:.2f}x  "
              f"MF+DP {z['mf_dp'] / interior_dp:.2f}x")

    out = {"seed": a.seed, "n_train": len(train),
           "train_cells": [c.key for c in train],
           "updates_per_cell": a.updates_per_cell,
           "dp_updates_per_cell": a.dp_updates_per_cell,
           "zones": zone_summary,
           "interior_mf": interior_mf, "interior_mf_dp": interior_dp,
           "noise_floor": G.FLOOR_16}
    write_json(OUT / f"summary_s{a.seed}.json", out)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
