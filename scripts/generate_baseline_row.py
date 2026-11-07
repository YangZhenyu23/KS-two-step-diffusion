"""Generate the mu = 1 row needed by the single-parameter control.

The sigma-only control model must be trained on one fixed mu, otherwise it would
be seeing mu information it is not allowed to condition on. The natural choice is
mu = 1, the upstream value, which makes the control identical to the published
single-parameter protocol.

The LHS-40 training set only contains four cells of that row, so this script
fills in the remaining sixteen. It writes its own cell list and delegates to the
resumable generator.

Usage:
    python scripts/generate_baseline_row.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fullscale import generate  # noqa: E402
from fullscale.grid import MU_INDEX, SIGMA_GRID  # noqa: E402


def main() -> int:
    cells = [[i, MU_INDEX] for i in range(len(SIGMA_GRID))]
    path = ROOT / "results" / "fullscale" / "data" / "baseline_cells.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cells), encoding="utf-8")
    print(f"baseline row: {len(cells)} cells at mu index {MU_INDEX} "
          f"(mu = {1.0})", flush=True)
    return generate.main(["--n", "100000", "--cells-from", str(path)])


if __name__ == "__main__":
    raise SystemExit(main())
