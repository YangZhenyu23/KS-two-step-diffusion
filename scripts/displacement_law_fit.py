"""Decide between the two candidate laws using only pairs that clear the floor.

Rules
-----
* A pair enters the fit only if ``w2 > 1.5 * floor``; below that the observation is
  dominated by the 16-bin quantisation artefact and the deconvolution
  ``sqrt(w2^2 - floor^2)`` is numerically meaningless.
* Both laws are fitted **through the origin** (a law of the form W2 = c * predictor
  has no intercept), and scored by R^2 and by the coefficient of variation of the
  ratio.

Two things are being tested:
  1. Does the exponent 1/2 on |dmu| belong in the theory at all? (§1.3 of THEORY.md)
  2. Is the dependence better described by the *displacement* scales S = sigma T and
     G = sqrt(2 mu T) than by the raw parameter values?
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
payload = json.loads((ROOT / "results" / "fullscale" / "displacement" / "displacement_law.json").read_text("utf-8"))
FLOOR = payload["floor"]


def fit(rows, key):
    x = np.array([r[key] for r in rows])
    y = np.array([r["w2_corrected"] for r in rows])
    c = float(np.dot(x, y) / np.dot(x, x))  # least squares through origin
    pred = c * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ratio = y / x
    return {
        "n": int(len(rows)),
        "constant": c,
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "ratio_median": float(np.median(ratio)),
        "ratio_cv": float(ratio.std(ddof=1) / ratio.mean()),
        "ratio_p05": float(np.quantile(ratio, 0.05)),
        "ratio_p95": float(np.quantile(ratio, 0.95)),
    }


out = {}
for axis in ("sigma", "mu"):
    all_rows = [r for r in payload["rows"] if r["axis"] == axis]
    kept = [r for r in all_rows if r["w2"] > 1.5 * FLOOR]
    out[axis] = {
        "n_total": len(all_rows),
        "n_kept": len(kept),
        "n_below_floor": sum(1 for r in all_rows if r["w2"] <= FLOOR),
        "A_holder": fit(kept, "pred_A_holder"),
        "B_displacement": fit(kept, "pred_B_displacement"),
    }

(ROOT / "results" / "fullscale" / "displacement" / "law_comparison.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8"
)

print(f"floor = {FLOOR}, keep w2 > {1.5 * FLOOR:.4f}\n")
for axis, block in out.items():
    print(f"=== {axis}: {block['n_kept']}/{block['n_total']} pairs kept "
          f"({block['n_below_floor']} at/below floor) ===")
    print(f"{'law':<18} {'const':>8} {'R^2':>7} {'med ratio':>10} {'cv':>7} {'p05':>7} {'p95':>7}")
    for name, label in (("A_holder", "A: |ds|+sqrt|dm|"), ("B_displacement", "B: dS+dG")):
        f = block[name]
        print(f"{label:<18} {f['constant']:8.4f} {f['r2']:7.3f} {f['ratio_median']:10.4f} "
              f"{f['ratio_cv']:7.3f} {f['ratio_p05']:7.3f} {f['ratio_p95']:7.3f}")
    print()
