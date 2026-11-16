"""Put the capacity-matched model on the SAME held-out set as the other three configurations.

The problem
-----------
``capacity_matched/w112_s88`` was trained on design S6 while the headline checkpoint uses
design S1; the two designs share only 6 of 40 cells. Their held-out sets therefore differ
(143 vs 144 cells) and neither is a subset of the other:

    cells in (c)'s held-out that (a) trained on: 32
    cells in (a)'s held-out that (c) trained on: 31

So the (c) number reported earlier (0.11604) is NOT comparable to (a)/(b)/(d) (0.12277 /
0.10429 / 0.12418). Using it in the same table would be a protocol error of exactly the kind
that invalidates a capacity argument.

The fix, and the only honest one
--------------------------------
Score (c) on the intersection: cells that are held out for *both* models, i.e.

    common = full_grid  - two_32000 cells - w112 cells - mu=1 row

Everything else must be regenerated from the w112 checkpoint on the shared cells, which is a
forward pass only (the model is already trained).

What this script reports
------------------------
  * size of the common held-out set
  * W_2 for (a) stage-1 alone, (b) stage-1 + corrector, (c) w112, (d) two-stage-1 passes,
    all on the SAME common cells
  * an explicit note of how much of the earlier gap was an artefact of the mismatch

Usage:
    python scripts/four_way_common.py
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

from fullscale import grid as G  # noqa: E402
from fullscale.train import load_cell, load_two  # noqa: E402
from fullscale.model import sample_two_param  # noqa: E402
from fullscale.evaluate import score_pair  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from rollout_compare import load_corrector  # noqa: E402

OUT = G.RESULTS / "four_way"


def main() -> int:
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda text: print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)

    headline_path = G.RESULTS / "optimal_budget_ablation" / "two_32000" / "mf_two_param.pt"
    wide_path = G.RESULTS / "capacity_matched" / "w112_s88" / "mf_two_param.pt"

    headline_cells = frozenset(
        torch.load(headline_path, map_location="cpu", weights_only=False)["cells"])
    wide_cells = frozenset(
        torch.load(wide_path, map_location="cpu", weights_only=False)["cells"])
    slice_keys = frozenset(c.key for c in G.baseline_slice())

    common = [c for c in G.full_grid()
              if c.key not in headline_cells
              and c.key not in wide_cells
              and c.key not in slice_keys]
    log(f"headline trains on {len(headline_cells)}, w112 on {len(wide_cells)}, "
        f"overlap {len(headline_cells & wide_cells)}")
    log(f"common held-out for BOTH models: {len(common)} cells")

    headline, _ = load_two(headline_path, "cuda")
    wide, _ = load_two(wide_path, "cuda")
    corrector = load_corrector(88, "cuda")
    device = next(corrector.parameters()).device

    def correct(cloud: torch.Tensor, sigma: float, mu: float) -> torch.Tensor:
        x = cloud.to(device)
        cs = torch.full((len(x), 1), math.log10(sigma), dtype=torch.float32, device=device)
        cm = torch.full((len(x), 1), math.log10(mu), dtype=torch.float32, device=device)
        with torch.no_grad():
            return corrector(x, cs, cm).cpu()

    results = {key: [] for key in ("a_headline_1pass", "b_headline_dp",
                                   "c_w112_1pass", "d_headline_2pass")}

    for position, cell in enumerate(common, start=1):
        item = load_cell(cell)
        reference = item["clouds"]["reference"]
        source = torch.from_numpy(item["clouds"]["source"])

        mf_a = sample_two_param(headline, source, cell.sigma, cell.mu, steps=1)
        mf_d = sample_two_param(headline, source, cell.sigma, cell.mu, steps=2)
        wide_c = sample_two_param(wide, source, cell.sigma, cell.mu, steps=1)
        dp_b = correct(mf_a, cell.sigma, cell.mu)

        for key, cloud in (("a_headline_1pass", mf_a), ("d_headline_2pass", mf_d),
                           ("c_w112_1pass", wide_c), ("b_headline_dp", dp_b)):
            results[key].append(
                float(score_pair(reference, cloud.numpy().astype(np.float32), G.BINS)["w2"]))

        if position % 40 == 0:
            log(f"  {position}/{len(common)}")

    summary = {key: {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)),
                     "n": len(values)} for key, values in results.items()}
    base = summary["a_headline_1pass"]["mean"]
    for key, block in summary.items():
        block["relative_vs_a"] = float(1 - block["mean"] / base)

    payload = {
        "n_common_held_out": len(common),
        "headline_cells": sorted(headline_cells),
        "w112_cells": sorted(wide_cells),
        "overlap": len(headline_cells & wide_cells),
        "common_cells": [c.key for c in common],
        "results": summary,
        "note": ("All four configurations scored on the SAME cells: those held out by BOTH "
                 "the headline checkpoint and the width-112 checkpoint. The previously "
                 "reported (c) value was computed on a different cell set and is not "
                 "comparable."),
    }
    (OUT / "four_way_common_s88.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nall four on the same {len(common)} held-out cells:")
    print(f"  {'configuration':<26} {'W_2':>9} {'sd':>8} {'vs (a)':>9}")
    for key in ("a_headline_1pass", "d_headline_2pass", "c_w112_1pass", "b_headline_dp"):
        block = summary[key]
        print(f"  {key:<26} {block['mean']:9.5f} {block['sd']:8.5f} "
              f"{block['relative_vs_a'] * 100:+8.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
