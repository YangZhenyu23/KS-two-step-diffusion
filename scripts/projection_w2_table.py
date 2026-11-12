"""Print the W2 values for the cells used in the projection figures."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = {r["key"]: r for r in csv.DictReader(
    open(ROOT / "results/fullscale/dp_two_param/s88/detail.csv", encoding="utf-8"))}

for k in ("s11_m09", "s00_m01", "s03_m04", "s08_m06", "s09_m06", "s09_m05",
          "s13_m07", "s19_m09"):
    r = rows.get(k)
    if r is None:
        print(k, "not held out")
        continue
    print(
        f"{k}  sigma={float(r['sigma']):5.0f}  mu={float(r['mu']):8.4f}  "
        f"mf={float(r['w2_mf']):.5f}  dp={float(r['w2_mf_dp']):.5f}  "
        f"gain={float(r['improvement']):+.5f}  ratio={float(r['ratio']):.3f}"
    )
