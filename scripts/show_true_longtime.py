"""What does the TRUE solution do as time grows?

Before asking whether the model captures long-time behaviour, we need to know what the
long-time behaviour IS. This reads the physical references already computed during the
rollout study (no new simulation) and prints the true cloud's invariants at T, 2T, 4T, 8T.

The physical setting matters here: a laminar shear g(x) = (exp(-y^2-z^2), 0, 0) advects mass
in +x indefinitely, while diffusion spreads it. There is no steady state to converge to.
That is a property of the PDE, not of the model, and it changes what "long-horizon" can
even mean.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fullscale import grid as G  # noqa: E402

payload = json.loads((G.RESULTS / "rollout" / "rollout.json").read_text("utf-8"))
rows = payload["rows"]
cells = sorted({r["key"] for r in rows})
ks = sorted({r["k"] for r in rows})

print("TRUE solution (physics), from the rollout study's reference simulations")
print(f"  n = {payload['n_simulate']:,} particles, T = {G.FINAL_TIME}\n")

for cell in cells:
    print(f"cell {cell}  (sigma={rows[0]['sigma']:.0f})")
    print(f"  {'k':>3} {'time':>8} {'centroid_x':>11} {'rms_radius':>11} "
          f"{'anisotropy':>11}")
    for k in ks:
        match = [r for r in rows if r["key"] == cell and r["k"] == k]
        if not match:
            continue
        truth = match[0]["physical"]
        print(f"  {k:3d} {G.FINAL_TIME * k:8.3f} {truth['centroid'][0]:11.3f} "
              f"{truth['radius_rms']:11.3f} {truth['anisotropy']:11.3f}")
    print()

print("key observation: centroid_x and rms_radius keep growing with k at every cell.")
print("there is no steady state; the shear advects and the diffusion spreads without bound.")
