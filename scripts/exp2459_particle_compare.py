"""Experiments 2, 4, 5 and 9 with particle-level scoring, on existing artefacts only.

All four ask the same question in different forms: given several trained models,
how do they rank when the ruler is the particle-level exact OT instead of the
16-bin histogram? They share one code path, differing only in which models are
compared.

    exp2  budget sweep       b2000 / b8000 / b32000 / b128000
    exp4  learning curve     curve20 / curve40 / curve80 / curve160
    exp5  LHS vs random      cmp_lhs_r0..3 / cmp_rnd_r0..3
    exp9  subsample size     b32000 scored at m = 1000 / 2000 / 3000 / 4000

Nothing is retrained and no physical simulation is run: every model and every
reference cloud already exists on disk.

Design decisions
----------------
* **Fixed evaluation cells.** Each experiment scores on the same frozen test cells
  it used originally, so the particle-level number is directly comparable to the
  histogram number already reported.
* **Paired indices.** One index vector per cell, reused for every model, so model
  differences cannot come from drawing different particles.
* **Subsample size.** Default 3000 for cross-model comparisons. Experiment 9
  varies it explicitly because the absolute value depends on m; only the ordering
  and the relative differences are claimed to be m-independent.

Usage:
    python scripts/exp2459_particle_compare.py --experiment exp2
    python scripts/exp2459_particle_compare.py --experiment exp4 exp5
    python scripts/exp2459_particle_compare.py --experiment exp9
"""
from __future__ import annotations

import argparse
import csv
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
from fullscale.evaluate import score_pair  # noqa: E402
from fullscale.model import sample_two_param  # noqa: E402
from fullscale.train import load_cell, load_two  # noqa: E402

OUT = G.RESULTS / "particle_compare"
SEED = G.RESULTS / "sample_efficiency" / "split.json"


def exact_w2(a, b) -> float:
    import ot
    from scipy.spatial.distance import cdist

    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    cost = cdist(a, b, metric="sqeuclidean")
    n = len(a)
    w = np.full(n, 1.0 / n)
    value, info = ot.emd2(w, w, cost, log=True, numItermax=5_000_000)
    if int(info.get("result_code", -1)) != 1:
        raise RuntimeError(f"OT failed: {info.get('warning')}")
    return math.sqrt(max(value, 0.0))


def freeze_cells() -> list[G.Cell]:
    payload = json.loads(SEED.read_text(encoding="utf-8"))
    return [G.Cell(int(r["i"]), int(r["j"])) for r in payload["test"]]


def score_models(models: dict[str, Path], cells: list[G.Cell], *,
                 subsample: int, draws: int, log) -> dict[str, dict]:
    """Score every model on every cell; return per-model per-cell results."""
    payloads = {c.key: load_cell(c) for c in cells}
    rng = np.random.default_rng(818181)
    indices = [rng.choice(G.N_PARTICLES, size=subsample, replace=False)
               for _ in range(draws)]

    out: dict[str, dict] = {name: {} for name in models}
    for name, path in models.items():
        started = time.perf_counter()
        model, _ = load_two(path)
        for cell in cells:
            payload = payloads[cell.key]
            reference = payload["clouds"]["reference"].astype(np.float64)
            prediction = sample_two_param(model, torch.from_numpy(payload["clouds"]["source"]),
                                          cell.sigma, cell.mu).numpy().astype(np.float64)
            particle = [exact_w2(prediction[idx], reference[idx]) for idx in indices]
            histogram = score_pair(reference, prediction, G.BINS)["w2"]
            out[name][cell.key] = {
                "particle": float(np.mean(particle)),
                "histogram": histogram,
            }
            del prediction
        log(f"  {name}: {time.perf_counter() - started:.0f}s")
        del model
        torch.cuda.empty_cache()
    return out


def summarise(scores: dict[str, dict], order: list[str]) -> list[dict]:
    rows = []
    for name in order:
        per_cell = scores[name]
        p = np.array([v["particle"] for v in per_cell.values()])
        h = np.array([v["histogram"] for v in per_cell.values()])
        rows.append({"model": name, "n_cells": len(p),
                     "particle_mean": float(p.mean()),
                     "particle_sd": float(p.std(ddof=1)),
                     "histogram_mean": float(h.mean()),
                     "histogram_sd": float(h.std(ddof=1))})
    return rows


def print_table(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print(f"{'model':<20} {'particle':>10} {'histogram':>11} "
          f"{'vs first (p)':>13} {'vs first (h)':>13}")
    base_p = rows[0]["particle_mean"]
    base_h = rows[0]["histogram_mean"]
    for r in rows:
        print(f"{r['model']:<20} {r['particle_mean']:10.5f} "
              f"{r['histogram_mean']:11.5f} "
              f"{1 - r['particle_mean'] / base_p:+13.1%} "
              f"{1 - r['histogram_mean'] / base_h:+13.1%}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", nargs="+", default=["exp2"],
                    choices=["exp2", "exp4", "exp5", "exp9"])
    ap.add_argument("--subsample", type=int, default=3000)
    ap.add_argument("--draws", type=int, default=2)
    a = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda t: print(f"[{time.strftime('%H:%M:%S')}] {t}", flush=True)
    cells = freeze_cells()
    log(f"{len(cells)} frozen test cells, subsample {a.subsample}, {a.draws} draws")

    report = {}

    if "exp2" in a.experiment:
        log("=== exp2: budget sweep ===")
        models = {f"budget_{b}": G.RESULTS / "budget_sweep" / f"b{b}_s88" / "mf_two_param.pt"
                  for b in (2000, 8000, 32000, 128000)}
        scores = score_models(models, cells, subsample=a.subsample, draws=a.draws, log=log)
        rows = summarise(scores, list(models))
        print_table(rows, "budget sweep (particle vs histogram)")
        report["exp2"] = rows

    if "exp4" in a.experiment:
        log("=== exp4: learning curve ===")
        models = {f"cells_{n}": G.RESULTS / "sample_efficiency" / "runs" / f"curve{n}"
                  / "mf_two_param.pt" for n in (20, 40, 80, 160)}
        scores = score_models(models, cells, subsample=a.subsample, draws=a.draws, log=log)
        rows = summarise(scores, list(models))
        print_table(rows, "learning curve (particle vs histogram)")
        report["exp4"] = rows

    if "exp5" in a.experiment:
        log("=== exp5: LHS vs random, four designs each ===")
        models = {}
        for k in range(4):
            models[f"lhs_r{k}"] = (G.RESULTS / "sample_efficiency" / "runs"
                                   / f"cmp_lhs_r{k}" / "mf_two_param.pt")
            models[f"rnd_r{k}"] = (G.RESULTS / "sample_efficiency" / "runs"
                                   / f"cmp_rnd_r{k}" / "mf_two_param.pt")
        scores = score_models(models, cells, subsample=a.subsample, draws=a.draws, log=log)
        lhs = np.array([np.mean([v["particle"] for v in scores[f"lhs_r{k}"].values()])
                        for k in range(4)])
        rnd = np.array([np.mean([v["particle"] for v in scores[f"rnd_r{k}"].values()])
                        for k in range(4)])
        lhs_h = np.array([np.mean([v["histogram"] for v in scores[f"lhs_r{k}"].values()])
                          for k in range(4)])
        rnd_h = np.array([np.mean([v["histogram"] for v in scores[f"rnd_r{k}"].values()])
                          for k in range(4)])
        se_p = math.sqrt(lhs.var(ddof=1) / 4 + rnd.var(ddof=1) / 4)
        se_h = math.sqrt(lhs_h.var(ddof=1) / 4 + rnd_h.var(ddof=1) / 4)
        print(f"\nLHS vs random (4 designs each)")
        print(f"  particle : LHS {lhs.mean():.5f} (sd {lhs.std(ddof=1):.5f})  "
              f"random {rnd.mean():.5f} (sd {rnd.std(ddof=1):.5f})")
        print(f"             difference {lhs.mean() - rnd.mean():+.5f} = "
              f"{(lhs.mean() - rnd.mean()) / se_p:+.1f} SE")
        print(f"  histogram: LHS {lhs_h.mean():.5f} (sd {lhs_h.std(ddof=1):.5f})  "
              f"random {rnd_h.mean():.5f} (sd {rnd_h.std(ddof=1):.5f})")
        print(f"             difference {lhs_h.mean() - rnd_h.mean():+.5f} = "
              f"{(lhs_h.mean() - rnd_h.mean()) / se_h:+.1f} SE")
        report["exp5"] = {
            "particle": {"lhs": lhs.tolist(), "random": rnd.tolist(),
                         "lhs_mean": float(lhs.mean()), "rnd_mean": float(rnd.mean()),
                         "difference": float(lhs.mean() - rnd.mean()),
                         "se": se_p,
                         "z": float((lhs.mean() - rnd.mean()) / se_p)},
            "histogram": {"lhs_mean": float(lhs_h.mean()), "rnd_mean": float(rnd_h.mean()),
                          "difference": float(lhs_h.mean() - rnd_h.mean()),
                          "se": se_h,
                          "z": float((lhs_h.mean() - rnd_h.mean()) / se_h)},
        }

    if "exp9" in a.experiment:
        log("=== exp9: subsample sensitivity (b32000) ===")
        model_path = G.RESULTS / "budget_sweep" / "b32000_s88" / "mf_two_param.pt"
        rows = []
        for m in (1000, 2000, 3000, 4000):
            scores = score_models({"b32000": model_path}, cells,
                                  subsample=m, draws=a.draws, log=log)
            per_cell = scores["b32000"]
            p = np.array([v["particle"] for v in per_cell.values()])
            rows.append({"subsample": m, "particle_mean": float(p.mean()),
                         "particle_sd": float(p.std(ddof=1)),
                         "n_cells": len(p)})
            log(f"  m={m}: {p.mean():.5f}")
        print(f"\n{'subsample':>10} {'particle mean':>14} {'sd':>9}")
        for r in rows:
            print(f"{r['subsample']:10d} {r['particle_mean']:14.5f} "
                  f"{r['particle_sd']:9.5f}")
        vals = np.array([r["particle_mean"] for r in rows])
        print(f"\nspread across m: {vals.max() / vals.min():.2f}x")
        print("  the absolute value depends on m; only the ordering of models and")
        print("  the relative differences should be treated as m-independent.")
        report["exp9"] = rows

    (OUT / "particle_compare.json").write_text(json.dumps(report, indent=2),
                                               encoding="utf-8")
    log(f"wrote {OUT / 'particle_compare.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
