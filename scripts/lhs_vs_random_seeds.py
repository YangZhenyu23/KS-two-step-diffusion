"""Multi-seed comparison of LHS versus random at fixed size.

The single-seed comparison is not resolvable. Two LHS-40 designs on the same pool
gave test errors of 0.1513 and 0.1667, a spread of 0.015, while the LHS-versus-
random gap in one draw was 0.018. When the design-to-design spread is the same
size as the effect, one seed says nothing.

This script repeats the comparison with several independent designs of each kind
and reports the two distributions. The question becomes: is the mean difference
between the methods larger than the spread within each method?

Note on what is being tested. A maximin criterion maximises the *minimum*
pairwise distance, which pushes points toward the boundary of the box and can
leave the interior relatively sparse. For a smooth response surface the error
that matters is the integrated (average) error over the domain, not the worst-case
gap between neighbouring training points. It is therefore entirely possible for a
maximin design to have better spacing by its own criterion yet train a worse
model. This script measures which happens, rather than assuming.

Usage:
    python scripts/lhs_vs_random_seeds.py --repeats 4
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

from sample_efficiency import (  # noqa: E402
    OUT, frozen_test_set, run_one, stratified_subset, write_csv, write_json, log,
)


def geometry(cells) -> dict:
    n_sigma, n_mu = len(G.SIGMA_GRID), len(G.MU_GRID)
    arr = np.array([[c.i / (n_sigma - 1), c.j / (n_mu - 1)] for c in cells])
    diff = arr[:, None, :] - arr[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    corr = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]) if arr[:, 0].std() > 0 else 0.0
    return {"min_distance": float(d.min()), "mean_nn_distance": float(d.min(axis=1).mean()),
            "correlation": corr}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--size", type=int, default=40)
    a = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    test, pool = frozen_test_set()
    log(f"test {len(test)}, pool {len(pool)}, size {a.size}, {a.repeats} repeats each")

    results = []
    for k in range(a.repeats):
        # LHS design
        lhs_seed = 20261107 + 1000 * k
        lhs_cells = stratified_subset(pool, a.size, seed=lhs_seed)
        rec = run_one(f"cmp_lhs_r{k}", lhs_cells, test, pool, reuse_existing=True)
        rec.update({"method": "lhs", "repeat": k, "design_seed": lhs_seed,
                    "design": geometry(lhs_cells)})
        results.append(rec)
        log(f"  LHS   repeat {k}: TEST {rec['test_mean_w2']:.5f} "
            f"(min d {rec['design']['min_distance']:.4f})")

        # random design, uniform from the same pool
        rng = np.random.default_rng(555000 + 1000 * k)
        chosen = rng.choice(len(pool), size=a.size, replace=False)
        rnd_cells = sorted((pool[int(c)] for c in chosen), key=lambda c: (c.i, c.j))
        rec = run_one(f"cmp_rnd_r{k}", rnd_cells, test, pool, reuse_existing=True)
        rec.update({"method": "random", "repeat": k, "design_seed": 555000 + 1000 * k,
                    "design": geometry(rnd_cells)})
        results.append(rec)
        log(f"  RANDOM repeat {k}: TEST {rec['test_mean_w2']:.5f} "
            f"(min d {rec['design']['min_distance']:.4f})")

    write_csv(OUT / "lhs_vs_random_seeds.csv", results)

    print(f"\n{'method':<8} {'runs':>5} {'mean':>9} {'sd':>8} {'min':>9} {'max':>9} "
          f"{'mean min_d':>11}")
    summary = {}
    for method in ("lhs", "random"):
        group = [r for r in results if r["method"] == method]
        values = [r["test_mean_w2"] for r in group]
        mins = [r["design"]["min_distance"] for r in group]
        summary[method] = {
            "runs": len(values),
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)), "max": float(np.max(values)),
            "mean_min_distance": float(np.mean(mins)),
            "values": values,
        }
        s = summary[method]
        print(f"{method:<8} {s['runs']:5d} {s['mean']:9.5f} {s['sd']:8.5f} "
              f"{s['min']:9.5f} {s['max']:9.5f} {s['mean_min_distance']:11.4f}")

    diff = summary["lhs"]["mean"] - summary["random"]["mean"]
    pooled = float(np.sqrt((summary["lhs"]["sd"] ** 2 + summary["random"]["sd"] ** 2) / 2))
    # Welch-style standard error for the difference of two means
    se = float(np.sqrt(summary["lhs"]["sd"] ** 2 / summary["lhs"]["runs"]
                       + summary["random"]["sd"] ** 2 / summary["random"]["runs"]))
    print(f"\nmean difference (lhs - random) = {diff:+.5f}")
    print(f"pooled within-method sd        = {pooled:.5f}")
    print(f"standard error of difference   = {se:.5f}")
    if se > 0:
        print(f"difference / SE                = {diff / se:+.2f}")
    verdict = ("LHS better" if diff < -2 * se else
               "random better" if diff > 2 * se else
               "not resolvable at this sample size")
    print(f"verdict (2-sigma rule)         : {verdict}")
    print("\nnote: with a maximin criterion the design is boundary-heavy by")
    print("construction; better spacing by that criterion need not mean lower")
    print("average test error, which is what is measured here.")

    write_json(OUT / "lhs_vs_random_seeds.json", {
        "size": a.size, "repeats": a.repeats, "n_test": len(test),
        "summary": summary, "difference": diff, "pooled_sd": pooled, "se": se,
        "verdict": verdict,
    })
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
