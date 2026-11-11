"""Inspect the mu pairs behind the displacement_law ratios.

The coefficient of variation came out enormous (p95/p05 ~ 1e12), which means some
ratios are essentially zero: the observed W2 for those pairs is at or below the
floor, so ``sqrt(w2^2 - floor^2)`` clips to zero. Those pairs carry no information
and must be excluded before judging a law.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
payload = json.loads((ROOT / "results" / "fullscale" / "displacement" / "displacement_law.json").read_text("utf-8"))
FLOOR = payload["floor"]

print(f"floor = {FLOOR}")
rows = [r for r in payload["rows"] if r["axis"] == "mu"]
rows.sort(key=lambda r: r["pred_B_displacement"])

print(f"\n{'dmu':>9} {'predA':>9} {'predB':>9} {'w2':>9} {'w2corr':>9} {'ratioB':>9}")
for r in rows:
    d = abs(r["mu2"] - r["mu1"])
    print(
        f"{d:9.4f} {r['pred_A_holder']:9.4f} {r['pred_B_displacement']:9.4f} "
        f"{r['w2']:9.5f} {r['w2_corrected']:9.5f} "
        f"{(r['ratio_B'] if r['ratio_B'] is not None else float('nan')):9.4f}"
    )

n_below = sum(1 for r in rows if r["w2"] <= FLOOR)
print(f"\n{sum(1 for r in rows if r['w2'] <= FLOOR)} of {len(rows)} mu pairs sit at or below the floor")
