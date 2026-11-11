"""Print the mu ladder for one sigma slice, so the shape of W2(h) is visible."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
payload = json.loads((ROOT / "results" / "fullscale" / "holder" / "mu_holder.json").read_text("utf-8"))

for sig in (10.0, 100.0, 190.0):
    rows = sorted((r for r in payload["rows"] if r["sigma"] == sig), key=lambda r: r["h_log10"])
    print(f"\nsigma = {sig}  (anchor mu = 1.0, floor 0.0644)")
    print("      h      mu        W2     W2-floor")
    for r in rows:
        print(f"{r['h_log10']:7.4f} {r['mu']:8.4f} {r['w2']:9.5f} {max(r['w2'] - 0.0644, 0.0):9.5f}")
