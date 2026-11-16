"""Row (c): a capacity-matched single network, trained and scored against the corrector.

The objection being tested
--------------------------
"Your two-network design wins because it has 26.5k parameters while the baseline has 9.0k.
A single wider network with the same parameter count would do as well."

The existing capacity sweep in the repository does NOT answer this: it varies width but
never matches the *combined* parameter count of MF + corrector, and it never scores the
two designs side by side on the same cells with the same budget.

Design
------
  MF-64       9,027 params, 1 forward          baseline
  MF+DP      26,502 params, 2 forwards         the proposal
  MF-96      20,035 params, 1 forward          slightly under matched
  MF-128     34,435 params, 1 forward          slightly over matched
  MF-112     26,435 params, 1 forward          closest to matched  <- the deciding row

MF-112 lands within 0.3% of 26,502 parameters, so if the corrector's gain were purely a
capacity effect, MF-112 should match it. Everything is trained with the identical budget
(same updates per cell), identical 40 cells, identical held-out set, identical metric.

To keep the run time bounded only the matched width is trained by default; the neighbour
widths are included because a single-point comparison can be non-monotone (the repository
already observed non-monotone width response, so a lone point proves little).

Usage:
    python scripts/capacity_matched.py --widths 112 --seeds 88 89 90
    python scripts/capacity_matched.py --widths 96 112 128 --seeds 88
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream" / "MF-DeepParticle" / "deepparticle"))

from fullscale import grid as G  # noqa: E402
from fullscale.train import (Config, load_cell, train_two_param, load_two)  # noqa: E402
from fullscale.model import TwoParamMeanflow, sample_two_param  # noqa: E402
from fullscale.evaluate import score_many  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from dp_two_param import stratified_subset  # noqa: E402

OUT = G.RESULTS / "capacity_matched"
UPDATES_PER_CELL = 32000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="+", default=[112])
    ap.add_argument("--seeds", type=int, nargs="+", default=[88, 89, 90])
    ap.add_argument("--updates-per-cell", type=int, default=UPDATES_PER_CELL)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    OUT.mkdir(parents=True, exist_ok=True)
    log = lambda text: print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)

    pool = G.full_grid()
    cells = stratified_subset(pool, 40, seed=20261107)
    design_keys = {c.key for c in cells}
    slice_keys = {c.key for c in G.baseline_slice()}
    held = [c for c in pool if c.key not in design_keys and c.key not in slice_keys]
    steps = a.updates_per_cell * len(cells)
    log(f"{len(held)} held-out cells, {len(cells)} training cells, {steps:,} steps per run")

    payloads = {cell.key: load_cell(cell) for cell in held}
    config = Config(n_train=G.N_PARTICLES, device=a.device)

    rows = []
    for width in a.widths:
        for seed in a.seeds:
            run_dir = G.ensure("capacity_matched", f"w{width}_s{seed}")
            checkpoint = run_dir / "mf_two_param.pt"
            model = TwoParamMeanflow(width=width, hidden_layers=3, dimension=3)
            params = sum(p.numel() for p in model.parameters())
            log(f"--- width {width} ({params:,} params) seed {seed} ---")

            if checkpoint.exists():
                log("  reusing existing checkpoint")
                model, _ = load_two(checkpoint, a.device)
            else:
                result = train_two_param(cells, config, run_dir, steps=steps,
                                         batch_size=a.batch_size, seed=seed,
                                         width=width)
                model, _ = load_two(run_dir / "mf_two_param.pt", a.device)
                log(f"  trained in {result['seconds']:.0f}s")

            pairs = {}
            for cell in held:
                payload = payloads[cell.key]
                pairs[cell.key] = (
                    payload["clouds"]["reference"],
                    sample_two_param(model, torch.from_numpy(payload["clouds"]["source"]),
                                     cell.sigma, cell.mu, steps=1).numpy().astype(np.float32),
                )
            scored = score_many(pairs, bins=G.BINS, workers=4, log=lambda t: None)
            values = [r["w2"] for r in scored]
            row = {"width": width, "params": params, "seed": seed,
                   "held_out_mean_w2": float(np.mean(values)),
                   "over_floor": float(np.mean(values) / G.FLOOR_16)}
            rows.append(row)
            log(f"  held-out {row['held_out_mean_w2']:.5f} "
                f"({row['over_floor']:.2f}x floor)")
            del model
            torch.cuda.empty_cache()

    (OUT / "capacity_matched.json").write_text(json.dumps({
        "updates_per_cell": a.updates_per_cell, "n_held_out": len(held),
        "n_train": len(cells), "rows": rows,
    }, indent=2), encoding="utf-8")

    print(f"\n{'width':>6} {'params':>9} {'runs':>5} {'mean':>9} {'sd':>8}")
    for width in sorted({r["width"] for r in rows}):
        group = [r for r in rows if r["width"] == width]
        values = np.array([r["held_out_mean_w2"] for r in group])
        print(f"{width:6d} {group[0]['params']:9,d} {len(group):5d} {values.mean():9.5f} "
              f"{(values.std(ddof=1) if len(values) > 1 else 0.0):8.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
