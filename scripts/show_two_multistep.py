"""Print the two multi-step experiments side by side, to make a distinction explicit.

They look similar and are constantly confused:

  rollout      applies the trained one-step map k times, each application spanning the SAME
               interval [0, T]. The cloud therefore represents physical time k*T, and the
               reference for comparison is the simulator run to k*T. This measures whether the
               generator can be used as a time stepper / whether repeated application is stable.

  n-step       keeps the horizon fixed at T but splits [0, 1] into N sub-intervals, so the
               cloud always represents the SAME physical time T. This measures whether the
               numerical decomposition of one transport improves with more evaluations.

They answer different questions and must not be quoted interchangeably.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fullscale import grid as G  # noqa: E402

rollout = json.loads((G.RESULTS / "rollout" / "rollout.json").read_text("utf-8"))
nstep = json.loads((G.RESULTS / "nstep" / "nstep_s88.json").read_text("utf-8"))
floor = rollout["floor_mean"]

print("A. repeated application of the FULL map  -> physical time k*T")
print(f"   floor at this scoring size = {floor:.4f}")
print(f"   {'k':>3} {'W_2 range':>22} {'x floor':>18}")
for k in sorted({r["k"] for r in rollout["rows"]}):
    values = [r["w2_to_physical"] for r in rollout["rows"] if r["k"] == k]
    print(f"   {k:3d} {min(values):10.3f} - {max(values):9.3f} "
          f"{min(values) / floor:8.1f} - {max(values) / floor:7.1f}")

print()
print("B. splitting [0, 1] into N sub-intervals  -> physical time stays T")
print(f"   {'N':>3} {'mean W_2':>10} {'vs N=1':>9} {'x floor(16bin)':>15}")
for row in nstep["rows"]:
    print(f"   {row['steps']:3d} {row['mf_mean_w2']:10.5f} "
          f"{row['mf_relative_vs_1step'] * 100:+8.1f}% {row['mf_over_floor']:15.2f}")

print()
print("C. the same distinction for the corrector")
print("   stage 1 + corrector stays at T (one pass each), so it is NOT a long-horizon result.")
print(f"   its W_2 = {nstep['rows'][0]['mf_dp_mean_w2']:.5f} at T, "
      f"{nstep['rows'][0]['mf_dp_over_floor']:.2f}x the 16-bin floor")
