"""Generate the LHS training cells first, then the remaining validation cells.

The complete 200-cell grid takes about 4.5 hours at 100k particles. Training only
needs the 40 LHS cells, so this script runs in two phases:

  phase 1  the 40 LHS cells          (~54 min)  -> training can start
  phase 2  the other 160 cells       (~3.6 h)   -> full-grid held-out validation

Both phases share the same generator and manifest, so if phase 2 is interrupted
nothing is lost: rerun and it resumes from the manifest. Pass ``--phase 1`` to do
only the training set, or ``--phase 2`` to fill in the rest.

The point is wall-clock efficiency: the model can be trained and its behaviour
inspected while the validation data is still being produced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fullscale import generate  # noqa: E402
from fullscale.grid import MU_GRID  # noqa: E402
from fullscale.lhs import latin_hypercube  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["1", "2", "both"], default="both")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--points", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261107)
    a = ap.parse_args()

    design = latin_hypercube(a.points, seed=a.seed)
    lhs_sigma = sorted({i for i, _ in design.points})
    lhs_mu = sorted({j for _, j in design.points})
    print(f"LHS-{a.points}: {len(lhs_sigma)} sigma levels x {len(lhs_mu)} mu levels "
          f"-> {len(design.points)} cells", flush=True)

    cells_json = ROOT / "results" / "fullscale" / "data" / "lhs_cells.json"
    cells_json.parent.mkdir(parents=True, exist_ok=True)
    cells_json.write_text(json.dumps([[int(i), int(j)] for i, j in design.points]), encoding="utf-8")

    if a.phase in ("1", "both"):
        print("\n=== phase 1: LHS training cells ===", flush=True)
        if generate.main(["--n", str(a.n), "--device", a.device,
                          "--cells-from", str(cells_json)]):
            return 1

    if a.phase in ("2", "both"):
        print("\n=== phase 2: remaining validation cells ===", flush=True)
        if generate.main(["--n", str(a.n), "--device", a.device,
                          "--skip-cells-from", str(cells_json)]):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
