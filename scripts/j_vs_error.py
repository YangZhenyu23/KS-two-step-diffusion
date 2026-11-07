"""Is the W2-coverage objective J actually predictive of model error?

The I-optimal design achieved the lowest J (0.517) yet the worst test error
(0.198). That single data point cannot tell us whether J is uninformative or
whether something else went wrong, because n = 1.

This script builds *many* designs with a range of J values, trains a model for a
representative subset, and measures the correlation between J and the realised
test error. Only if that correlation is strong is J a usable design criterion.

Cheap designs first
-------------------
Computing J costs nothing (the distance matrix is cached), so several hundred
designs are generated and their J recorded. A handful spanning the J range are
then trained. This gives a J-versus-error scatter rather than a single pair.

What would each outcome mean
----------------------------
* strong positive correlation -> J is a good criterion; the I-optimal outlier was
  an accident of one draw
* no correlation -> J does not predict error; the coverage criterion is the wrong
  quantity for this task and the maximin result should be reported as neutral
* negative correlation -> minimising J actively hurts, which would suggest the
  binding constraint is smoothness of the response surface, not proximity to a
  training cloud

Usage:
    python scripts/j_vs_error.py --train 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fullscale import grid as G  # noqa: E402
from fullscale.ioptimal import load_or_build, maximin_distance  # noqa: E402

from sample_efficiency import (  # noqa: E402
    OUT, frozen_test_set, run_one, write_csv, write_json, log,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=int, default=300)
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--size", type=int, default=40)
    a = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    test, pool = frozen_test_set()
    cells_all = G.full_grid()
    geometry = load_or_build(cells_all, OUT / "bures_distance.npy", log=log)
    all_index = {c.key: k for k, c in enumerate(cells_all)}
    pool_idx = np.array([all_index[c.key] for c in pool])
    domain_idx = np.arange(len(cells_all))

    # ---- generate candidates spanning the J range -------------------------
    log(f"generating {a.candidates} candidate designs")
    rng = np.random.default_rng(20261107)
    seen: set[frozenset] = set()
    candidates = []
    while len(candidates) < a.candidates:
        pick = tuple(sorted(rng.choice(pool_idx, size=a.size, replace=False).tolist()))
        key = frozenset(pick)
        if key in seen:
            continue
        seen.add(key)
        sub = geometry.distance[np.ix_(domain_idx, np.array(pick))]
        nearest = sub.min(axis=1)
        candidates.append({
            "indices": list(pick),
            "J": float((nearest ** 2).sum()),
            "mean_nearest_w2": float(nearest.mean()),
            "min_spacing": maximin_distance([cells_all[k] for k in pick]),
        })
    js = np.array([c["J"] for c in candidates])
    log(f"  J range {js.min():.4f} .. {js.max():.4f}  (mean {js.mean():.4f})")

    # ---- pick a spread across the J range, all at the *same* spacing ------
    # Spacing is held fixed so the comparison isolates J. Random draws nearly all
    # share the modal spacing of 0.0526, so restricting to that band keeps the
    # comparison clean.
    modal = float(np.median([c["min_spacing"] for c in candidates]))
    band = [c for c in candidates if abs(c["min_spacing"] - modal) < 1e-9]
    log(f"  {len(band)} candidates share the modal spacing {modal:.4f}")
    band.sort(key=lambda c: c["J"])
    if len(band) < a.train:
        raise SystemExit("not enough candidates in the spacing band")
    picks = [band[int(round(k * (len(band) - 1) / max(a.train - 1, 1)))]
             for k in range(a.train)]

    rows = []
    for rank, candidate in enumerate(picks):
        cells = [cells_all[k] for k in candidate["indices"]]
        name = f"jscan_{rank}"
        run = run_one(name, cells, test, pool, reuse_existing=True)
        rows.append({"rank": rank, "name": name, "J": candidate["J"],
                     "mean_nearest_w2": candidate["mean_nearest_w2"],
                     "min_spacing": candidate["min_spacing"],
                     "training_fit_w2": run["training_fit_w2"],
                     "test_mean_w2": run["test_mean_w2"],
                     "test_over_floor": run["test_over_floor"]})
        log(f"  rank {rank}: J {candidate['J']:.4f} -> TEST {run['test_mean_w2']:.5f}")

    write_csv(OUT / "j_vs_error.csv", rows)
    J = np.array([r["J"] for r in rows])
    err = np.array([r["test_mean_w2"] for r in rows])
    corr = float(np.corrcoef(J, err)[0, 1]) if J.std() > 0 and err.std() > 0 else None
    from scipy.stats import spearmanr
    rho = float(spearmanr(J, err).statistic) if J.std() > 0 and err.std() > 0 else None

    print(f"\n{'rank':>4} {'J':>8} {'mean near':>10} {'train fit':>10} {'TEST W2':>9}")
    for r in rows:
        print(f"{r['rank']:4d} {r['J']:8.4f} {r['mean_nearest_w2']:10.5f} "
              f"{r['training_fit_w2']:10.5f} {r['test_mean_w2']:9.5f}")
    print(f"\ncorrelation(J, test error) Pearson  {corr:+.3f}" if corr is not None
          else "\nnot enough variation to correlate")
    print(f"correlation(J, test error) Spearman {rho:+.3f}" if rho is not None else "")

    write_json(OUT / "j_vs_error.json", {
        "n_candidates": a.candidates, "n_trained": a.train,
        "spacing_band": modal, "pearson": corr, "spearman": rho, "rows": rows,
    })
    print(f"\nwrote {OUT}")
    print("\ninterpretation: a positive correlation means lower J predicts lower")
    print("error. Near-zero means J is not a usable criterion for this task.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
