"""Check the training budget behind every model used in the one-vs-two comparison.

Fairness requires equal updates per cell. This prints the budget actually recorded in
each run's training log, because the comparison script loads a two-parameter
checkpoint from ``train/lhs40`` while the headline results use
``optimal_budget_ablation/two_32000`` -- if those two have different budgets, the
comparison is against an undertrained model and the gain is understated (or the
control is mis-scaled).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "fullscale"

RUNS = [
    ("two-param (used by fair_control)", R / "train/lhs40/training_log.json"),
    ("two-param (headline, 32k/cell)", R / "optimal_budget_ablation/two_32000/training_log.json"),
    ("two-param seed89 (headline)", R / "optimal_budget_ablation/two_32000_s89/training_log.json"),
    ("two-param seed90 (headline)", R / "optimal_budget_ablation/two_32000_s90/training_log.json"),
    ("one-param same-cells", R / "fair_control/same-cells/training_log.json"),
    ("one-param slice-20", R / "fair_control/slice-20/training_log.json"),
    ("one-param same-4", R / "fair_control/same-4/training_log.json"),
]

print(f"{'run':<38} {'cells':>6} {'steps':>9} {'per cell':>10}")
print("-" * 68)
for label, path in RUNS:
    if not path.exists():
        print(f"{label:<38}   (missing: {path.relative_to(ROOT)})")
        continue
    log = json.loads(path.read_text(encoding="utf-8"))
    cells = len(log.get("cells", []))
    steps = int(log.get("steps", 0))
    print(f"{label:<38} {cells:6d} {steps:9d} {steps / max(cells, 1):10.0f}")
