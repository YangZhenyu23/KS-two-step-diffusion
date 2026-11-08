"""Does normalising the two parameter scales actually improve accuracy?

The two physical parameters live on different scales, so the way they enter the
network matters:

    sigma  10..200      log10 range 1.00..2.30, mean 1.65, sd 0.40
    mu     0.1..10      log10 range -1.0..1.00, mean 0.00, sd 0.59

With plain ``log10`` the first linear layer receives two O(1) columns whose means
differ by 2.0 and whose spreads differ by a factor of 1.7. That is a conditioning
defect, not a modelling choice, and it costs nothing to fix: every candidate
encoding is a handful of flops per particle, so inference speed is untouched and
the single-step constraint is preserved.

This script trains the same architecture on the same 40 cells with three
encodings and scores them on the same frozen test set:

    log10        current behaviour
    log10_norm   both axes zero mean, unit variance
    unit01       both log axes mapped onto [0, 1]

Several seeds per encoding, because a 5% difference is the size of the seed
spread established in docs/SAMPLE_EFFICIENCY.md.

Usage:
    python scripts/conditioning_ablation.py --seeds 2
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fullscale import grid as G  # noqa: E402
from fullscale.evaluate import score_many  # noqa: E402
from fullscale.grid import Config  # noqa: E402
from fullscale.model import sample_two_param  # noqa: E402
from fullscale.train import load_cell, load_two, train_two_param  # noqa: E402

from sample_efficiency import (  # noqa: E402
    OUT, frozen_test_set, stratified_subset, write_csv, write_json, log,
)

UPDATES_PER_CELL = 2000
OUT_COND = G.RESULTS / "conditioning"


def score(model, cells, payloads) -> float:
    pairs = {}
    for cell in cells:
        payload = payloads[cell.key]
        pairs[cell.key] = (
            payload["clouds"]["reference"],
            sample_two_param(model, torch.from_numpy(payload["clouds"]["source"]),
                             cell.sigma, cell.mu).numpy().astype(np.float32),
        )
    rows = score_many(pairs, bins=G.BINS, workers=4, log=lambda t: None)
    return float(np.mean([r["w2"] for r in rows]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["log10", "log10_norm", "unit01"])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4096)
    a = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    test, pool = frozen_test_set()
    train_cells = stratified_subset(pool, 40, seed=20261107)
    steps = UPDATES_PER_CELL * len(train_cells)
    config = Config(n_train=G.N_PARTICLES, device="cuda")
    log(f"{len(train_cells)} train cells, {len(test)} test cells, {steps} steps, "
        f"{a.seeds} seed(s), modes {a.modes}")

    train_payloads = {c.key: load_cell(c) for c in train_cells}
    test_payloads = {c.key: load_cell(c) for c in test}

    rows = []
    for mode in a.modes:
        for seed in (88, 89, 90)[: a.seeds]:
            name = f"{mode}_s{seed}"
            work = OUT_COND / name
            log(f"--- {name} ---")
            started = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                summary = train_two_param(train_cells, config, work, steps=steps,
                                          batch_size=a.batch_size, seed=seed,
                                          conditioning=mode)
            seconds = time.perf_counter() - started
            model, _ = load_two(work / "mf_two_param.pt")
            fit = score(model, train_cells, train_payloads)
            held = score(model, test, test_payloads)
            row = {"mode": mode, "seed": seed, "steps": steps, "seconds": seconds,
                   "final_loss": summary["final_loss"],
                   "train_fit_w2": fit, "test_w2": held,
                   "generalisation_gap": held - fit,
                   "test_over_floor": held / G.FLOOR_16}
            rows.append(row)
            log(f"  fit {fit:.5f}  TEST {held:.5f}  ({held / G.FLOOR_16:.2f}x floor)  "
                f"{seconds:.0f}s")
            del model
            torch.cuda.empty_cache()

    write_csv(OUT_COND / "conditioning_ablation.csv", rows)

    print(f"\n{'mode':<12} {'runs':>5} {'train fit':>10} {'TEST mean':>10} "
          f"{'sd':>8} {'x floor':>8}")
    summary_out = {}
    for mode in a.modes:
        group = [r for r in rows if r["mode"] == mode]
        values = [r["test_w2"] for r in group]
        summary_out[mode] = {
            "runs": len(values),
            "train_fit": float(np.mean([r["train_fit_w2"] for r in group])),
            "test_mean": float(np.mean(values)),
            "test_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "values": values,
            "over_floor": float(np.mean(values) / G.FLOOR_16),
        }
        s = summary_out[mode]
        print(f"{mode:<12} {s['runs']:5d} {s['train_fit']:10.5f} {s['test_mean']:10.5f} "
              f"{s['test_sd']:8.5f} {s['over_floor']:8.2f}")

    base = "log10"
    if base in summary_out:
        print(f"\ncomparison against {base} (current behaviour):")
        for mode in a.modes:
            if mode == base:
                continue
            diff = summary_out[mode]["test_mean"] - summary_out[base]["test_mean"]
            se = float(np.sqrt(summary_out[mode]["test_sd"] ** 2 / max(summary_out[mode]["runs"], 1)
                               + summary_out[base]["test_sd"] ** 2 / max(summary_out[base]["runs"], 1)))
            ratio = diff / se if se > 0 else float("nan")
            verdict = ("better" if ratio < -2 else "worse" if ratio > 2 else "not resolvable")
            print(f"  {mode:<12} {diff:+.5f}  ({ratio:+.1f} SE)  {verdict}")
            summary_out[mode]["vs_log10"] = {"difference": diff, "se": se, "verdict": verdict}

    write_json(OUT_COND / "conditioning_ablation.json",
               {"steps": steps, "modes": a.modes, "summary": summary_out, "rows": rows})
    print(f"\nwrote {OUT_COND}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
