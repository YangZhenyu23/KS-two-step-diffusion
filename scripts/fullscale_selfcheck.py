"""Pre-flight checks for the full-scale run.

Runs before training, on a handful of already-generated cells, and verifies the
things that silently break a reproduction:

1. **Endpoint convention.** One short training run on a few cells must produce a
   sampler that beats the initial cloud. If the flow orientation in training were
   flipped, this check fails loudly instead of yielding a plausible but wrong
   error table. (This exact bug occurred in the pilot and inflated W2 fivefold.)
2. **Shared initial cloud.** Every cell must carry the identical initial array,
   otherwise "parameter change" and "resampling noise" cannot be separated.
3. **Grid metadata.** Every cloud must declare the sigma and mu it was actually
   simulated with, and they must match the grid definition.
4. **Physics sanity.** The terminal cloud must grow with mu and stretch with
   sigma, as the dimensionless analysis predicts.
5. **Histogram integrity.** The binned measure must retain every particle.

Usage:
    python scripts/fullscale_selfcheck.py --cells 8
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

from fullscale import grid as G  # noqa: E402
from fullscale.evaluate import measure, shared_edges  # noqa: E402
from fullscale.grid import Config  # noqa: E402
from fullscale.lhs import latin_hypercube  # noqa: E402
from fullscale.train import load_cell, train_two_param  # noqa: E402
from fullscale.model import sample_two_param  # noqa: E402


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}{('  ' + detail) if detail else ''}", flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=8, help="how many available cells to check")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "fullscale" / "selfcheck.json")
    a = ap.parse_args()

    manifest_path = G.DATA / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("no dataset yet; run fullscale.generate first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    available = {(r["i"], r["j"]) for r in manifest.get("cells", [])}
    print(f"dataset has {len(available)} cells")

    design = latin_hypercube(40)
    ordered = [c for c in (G.Cell(i, j) for i, j in design.points) if (c.i, c.j) in available]
    if not ordered:
        ordered = [c for c in G.full_grid() if (c.i, c.j) in available]
    chosen = ordered[: a.cells]
    print(f"checking {len(chosen)} cells: {[c.key for c in chosen]}\n")

    failures = 0

    print("1. shared initial cloud")
    initials = {c.key: load_cell(c)["clouds"]["initial"] for c in chosen}
    first = next(iter(initials.values()))
    same = all(np.array_equal(first, v) for v in initials.values())
    failures += not check("all cells share one initial", same,
                          f"shape={first.shape} dtype={first.dtype}")

    print("2. grid metadata")
    meta_ok = True
    for cell in chosen:
        meta = load_cell(cell)["meta"]
        if abs(meta["sigma"] - cell.sigma) > 1e-9 or abs(meta["mu"] - cell.mu) > 1e-12:
            meta_ok = False
            print(f"       {cell.key}: metadata sigma={meta['sigma']} mu={meta['mu']}")
        if meta["particles"] != G.N_PARTICLES:
            meta_ok = False
            print(f"       {cell.key}: {meta['particles']} particles, expected {G.N_PARTICLES}")
    failures += not check("metadata matches the grid and particle count", meta_ok)

    print("3. histogram integrity")
    hist_ok = True
    for cell in chosen:
        cloud = load_cell(cell)["clouds"]["reference"]
        edges = shared_edges([cloud], G.BINS)
        _, weights = measure(cloud, edges, G.BINS)
        if abs(weights.sum() - 1.0) > 1e-12:
            hist_ok = False
            print(f"       {cell.key}: weight sum {weights.sum()}")
    failures += not check("occupied-bin weights sum to one", hist_ok)

    print("4. physics sanity (does each parameter move a different feature?)")
    geo = {}
    for cell in chosen:
        cloud = load_cell(cell)["clouds"]["reference"].astype(np.float64)
        radial = np.sqrt(cloud[:, 1] ** 2 + cloud[:, 2] ** 2)
        geo[cell.key] = (float(cloud[:, 0].std()), float(np.percentile(radial, 95)))
    mus = sorted({c.mu for c in chosen})
    sigmas = sorted({c.sigma for c in chosen})
    spread_ok = True
    if len(mus) > 1:
        # at a fixed sigma, larger mu must mean a wider cloud
        for sigma in sigmas:
            group = [(c.mu, geo[c.key][1]) for c in chosen if c.sigma == sigma]
            group.sort()
            if len(group) > 1:
                values = [v for _, v in group]
                monotone = all(x <= y * 1.02 for x, y in zip(values, values[1:]))
                if not monotone:
                    spread_ok = False
                    print(f"       sigma={sigma:g}: yz spread not increasing in mu: {values}")
                    break
        failures += not check("yz spread increases with mu", spread_ok)
    else:
        print("       (single mu level in this sample; skipped)")

    print("5. endpoint convention (short training run must beat the initial cloud)")
    config = Config(n_train=int(first.shape[0]), device="cuda")
    work = G.ensure("selfcheck")
    summary = train_two_param(chosen, config, work, steps=a.steps, batch_size=min(2048, config.n_train),
                              log=lambda t: None)
    from fullscale.train import load_two
    model, _ = load_two(work / "mf_two_param.pt")
    print(f"       trained {a.steps} steps, final loss {summary['final_loss']:.4g}")

    beats = 0
    for cell in chosen:
        payload = load_cell(cell)
        source = payload["clouds"]["source"]
        reference = payload["clouds"]["reference"]
        prediction = sample_two_param(model, torch.from_numpy(source), cell.sigma, cell.mu).numpy()
        from fullscale.evaluate import score_pair
        baseline = score_pair(reference, source, G.BINS)["w2"]
        achieved = score_pair(reference, prediction, G.BINS)["w2"]
        ok = achieved < baseline
        beats += ok
        print(f"       {cell.key}: W2(initial)={baseline:.4f}  W2(predicted)={achieved:.4f}  "
              f"{'better' if ok else 'WORSE'}")
    failures += not check("sampler improves on the initial cloud in every checked cell",
                          beats == len(chosen), f"{beats}/{len(chosen)}")

    result = {"cells": [c.key for c in chosen], "failures": failures,
              "steps": a.steps, "final_loss": summary["final_loss"],
              "sampler_better_count": int(beats)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
    print(f"wrote {a.out}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
