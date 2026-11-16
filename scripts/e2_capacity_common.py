"""Recompute the capacity comparison across all three seeds on the common held-out cells.

The single-seed result in the note said width-112 reached +4.9 % against stage 1. With seeds
89 and 90 added, the three-seed means must be compared on the SAME cells as the two-stage
model, so this script:

  1. loads each width-112 checkpoint,
  2. derives the cells it did NOT train on,
  3. intersects with the cells the headline checkpoint did not train on,
  4. rescores stage 1, the corrector, and all three width-112 seeds on that intersection.

Without step 3 the comparison mixes cell sets, which is the error that was already found and
fixed once in this project.

Usage:
    python scripts/e2_capacity_common.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream" / "MF-DeepParticle" / "deepparticle"))
sys.path.insert(0, str(ROOT / "scripts"))

from fullscale import grid as G  # noqa: E402
from fullscale.train import load_cell, load_two  # noqa: E402
from fullscale.model import sample_two_param  # noqa: E402
from fullscale.evaluate import score_pair  # noqa: E402

from rollout_compare import load_corrector  # noqa: E402

OUT = G.RESULTS / "capacity_matched"
WIDE = {
    88: OUT / "w112_s88" / "mf_two_param.pt",
    89: OUT / "w112_s89" / "mf_two_param.pt",
    90: OUT / "w112_s90" / "mf_two_param.pt",
}


def main() -> int:
    torch.set_num_threads(1)
    log = lambda text: print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)

    headline = G.RESULTS / "optimal_budget_ablation" / "two_32000" / "mf_two_param.pt"
    headline_cells = frozenset(
        torch.load(headline, map_location="cpu", weights_only=False)["cells"])
    slice_keys = frozenset(c.key for c in G.baseline_slice())

    wide_cells = {}
    for seed, path in WIDE.items():
        if not path.exists():
            log(f"missing {path.relative_to(ROOT)}")
            continue
        wide_cells[seed] = frozenset(
            torch.load(path, map_location="cpu", weights_only=False)["cells"])

    log(f"headline trains on {len(headline_cells)}; "
        f"wide seeds available: {sorted(wide_cells)}")

    # common held-out: not trained by the headline model AND not trained by THIS wide seed
    results = {}
    mf_model, _ = load_two(headline, "cuda")
    corrector = load_corrector(88, "cuda")
    device = next(corrector.parameters()).device

    def correct(cloud, sigma, mu):
        x = cloud.to(device)
        cs = torch.full((len(x), 1), math.log10(sigma), dtype=torch.float32, device=device)
        cm = torch.full((len(x), 1), math.log10(mu), dtype=torch.float32, device=device)
        with torch.no_grad():
            return corrector(x, cs, cm).cpu()

    for seed, paths in ((None, None), (88, WIDE[88]), (89, WIDE[89]), (90, WIDE[90])):
        if seed is not None and seed not in wide_cells:
            continue
        if seed is None:
            cells = [c for c in G.full_grid()
                     if c.key not in headline_cells and c.key not in slice_keys]
            label = "stage1"
        else:
            cells = [c for c in G.full_grid()
                     if c.key not in headline_cells
                     and c.key not in wide_cells[seed]
                     and c.key not in slice_keys]
            label = f"wide_s{seed}"
        log(f"=== {label}: {len(cells)} common held-out cells ===")

        wide = None
        if seed is not None:
            wide, _ = load_two(paths, "cuda")

        scores = {"stage1": [], "corrector": [], "wide": []}
        for position, cell in enumerate(cells, start=1):
            item = load_cell(cell)
            reference = item["clouds"]["reference"]
            source = torch.from_numpy(item["clouds"]["source"])
            mf = sample_two_param(mf_model, source, cell.sigma, cell.mu, steps=1)
            scores["stage1"].append(float(score_pair(reference, mf.numpy().astype(np.float32),
                                                     G.BINS)["w2"]))
            dp = correct(mf, cell.sigma, cell.mu)
            scores["corrector"].append(float(score_pair(reference,
                                                        dp.numpy().astype(np.float32),
                                                        G.BINS)["w2"]))
            if wide is not None:
                wc = sample_two_param(wide, source, cell.sigma, cell.mu, steps=1)
                scores["wide"].append(float(score_pair(reference,
                                                       wc.numpy().astype(np.float32),
                                                       G.BINS)["w2"]))
            if position % 40 == 0:
                log(f"  {position}/{len(cells)}")

        entry = {key: {"mean": float(np.mean(v)), "n": len(v)}
                 for key, v in scores.items() if v}
        base = entry["stage1"]["mean"]
        for key, block in entry.items():
            block["relative_vs_stage1"] = float(1 - block["mean"] / base)
        results[label] = entry
        log(f"  stage1 {entry['stage1']['mean']:.5f}  "
            f"corrector {entry['corrector']['mean']:.5f} "
            f"({entry['corrector']['relative_vs_stage1'] * 100:+.1f}%)"
            + (f"  wide {entry['wide']['mean']:.5f} "
               f"({entry['wide']['relative_vs_stage1'] * 100:+.1f}%)"
               if "wide" in entry else ""))

    # aggregate the three wide seeds (each on its own common set, which is legitimate because
    # the comparison in each row is internal)
    wide_gains = [results[f"wide_s{s}"]["wide"]["relative_vs_stage1"]
                  for s in (88, 89, 90) if f"wide_s{s}" in results]
    corr_gains = [results[f"wide_s{s}"]["corrector"]["relative_vs_stage1"]
                  for s in (88, 89, 90) if f"wide_s{s}" in results]
    payload = {
        "per_seed": results,
        "wide_gain_mean": float(np.mean(wide_gains)) if wide_gains else None,
        "wide_gain_sd": float(np.std(wide_gains, ddof=1)) if len(wide_gains) > 1 else None,
        "corrector_gain_mean": float(np.mean(corr_gains)) if corr_gains else None,
        "corrector_gain_sd": float(np.std(corr_gains, ddof=1)) if len(corr_gains) > 1 else None,
        "note": ("Each seed's row compares width-112 against stage 1 on the cells that BOTH "
                 "the headline model and that wide seed left out, so every contrast is "
                 "internal to its own cell set."),
    }
    (OUT / "capacity_common.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nper-seed, each on its own common held-out cells:")
    print(f"  {'row':<12} {'cells':>6} {'stage1':>9} {'corrector':>10} {'width112':>10}")
    for label, block in results.items():
        wide = block.get("wide", {}).get("mean")
        print(f"  {label:<12} {block['stage1']['n']:>6} {block['stage1']['mean']:9.5f} "
              f"{block['corrector']['mean']:10.5f} "
              f"{(f'{wide:.5f}' if wide else '-'):>10}")
    print(f"\n  width-112 gain : {payload['wide_gain_mean'] * 100:+.1f}% "
          f"+- {payload['wide_gain_sd'] * 100 if payload['wide_gain_sd'] else 0:.1f}%")
    print(f"  corrector gain : {payload['corrector_gain_mean'] * 100:+.1f}% "
          f"+- {payload['corrector_gain_sd'] * 100 if payload['corrector_gain_sd'] else 0:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
