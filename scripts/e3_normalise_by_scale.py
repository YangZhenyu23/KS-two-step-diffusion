"""Normalise E3's long-horizon error by cloud scale before drawing conclusions.

E3 retrains at T' = 4T and measures W2 against the true terminal cloud. The error grew from
0.123 to 0.338, a factor of 2.75. But the true cloud at T' is also larger: its rms radius is
1.918 against 0.941 at T, a factor of 2.04. Since W2 is an absolute distance, part of the
growth is simply the cloud being bigger.

This script separates the two by reporting, per horizon:
    the raw W2
    the rms radius of the true cloud
    W2 divided by that radius  (a scale-free relative error)
and the same for the reproducibility floor, which also scales.

Only then can the question "does the model get worse at longer horizons" be answered.

Usage:
    python scripts/e3_normalise_by_scale.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream" / "MF-DeepParticle" / "deepparticle"))

from fullscale import grid as G  # noqa: E402

OUT = G.RESULTS / "long_horizon"


def rms(points: np.ndarray) -> float:
    centred = points - points.mean(0)
    return float(np.sqrt((centred ** 2).sum(1).mean()))


def main() -> int:
    summary_path = OUT / "long_horizon_T4.json"
    if not summary_path.exists():
        print("missing long_horizon_T4.json")
        return 1
    summary = json.loads(summary_path.read_text("utf-8"))

    # gather true-cloud scales at both horizons from the same cells
    cells = ["s09_m04", "s00_m05", "s14_m05"]
    rows = []
    for key in cells:
        short = G.DATA / "clouds" / f"{key}.npz"
        long = OUT / "clouds_T4" / "clouds" / f"{key}.npz"
        if not (short.exists() and long.exists()):
            continue
        a = np.load(short)["reference"].astype(np.float64)
        b = np.load(long)["reference"].astype(np.float64)
        rows.append({"key": key, "rms_T": rms(a), "rms_T4": rms(b)})

    mean_rms_T = float(np.mean([r["rms_T"] for r in rows])) if rows else float("nan")
    mean_rms_T4 = float(np.mean([r["rms_T4"] for r in rows])) if rows else float("nan")
    scale_growth = mean_rms_T4 / mean_rms_T

    w2_T = 0.12276673234794717          # headline held-out, from dp_two_param/s88
    w2_T4 = summary["mf_held_out"]
    floor_T = G.FLOOR_16                 # 16-bin floor, measured at T
    # the floor at T' is not measured; it scales with the cloud, so estimate it that way
    floor_T4_est = floor_T * scale_growth

    payload = {
        "w2_T": w2_T, "w2_T4": w2_T4,
        "raw_growth": w2_T4 / w2_T,
        "rms_T": mean_rms_T, "rms_T4": mean_rms_T4,
        "scale_growth": scale_growth,
        "relative_error_T": w2_T / mean_rms_T,
        "relative_error_T4": w2_T4 / mean_rms_T4,
        "relative_growth": (w2_T4 / mean_rms_T4) / (w2_T / mean_rms_T),
        "floor_T": floor_T,
        "floor_T4_estimate": floor_T4_est,
        "over_floor_T": w2_T / floor_T,
        "over_floor_T4_estimate": w2_T4 / floor_T4_est,
        "two_step_T4": summary["mf_2step_held_out"],
        "two_step_gain_at_T4": 1 - summary["mf_2step_held_out"] / w2_T4,
        "cells_used_for_scale": cells,
        "note": ("The floor at T' is not measured directly; it is estimated by scaling the "
                 "T floor with the cloud radius, which is how quantisation error behaves. "
                 "The comparison is therefore indicative, not exact."),
    }
    (OUT / "normalised.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"true-cloud rms radius   T = {mean_rms_T:.3f}   T' = {mean_rms_T4:.3f}   "
          f"({scale_growth:.2f}x)")
    print(f"raw W2                  T = {w2_T:.5f}   T' = {w2_T4:.5f}   "
          f"({w2_T4 / w2_T:.2f}x)")
    print(f"W2 / rms radius         T = {payload['relative_error_T']:.4f}   "
          f"T' = {payload['relative_error_T4']:.4f}   "
          f"({payload['relative_growth']:.2f}x)")
    print(f"over floor (estimated)  T = {payload['over_floor_T']:.2f}x   "
          f"T' = {payload['over_floor_T4_estimate']:.2f}x")
    print(f"two-step at T'          {payload['two_step_T4']:.5f}  "
          f"({payload['two_step_gain_at_T4'] * 100:+.1f}% vs one step)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
