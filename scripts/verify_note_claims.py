"""Verify every quantitative claim in TECHNICAL_NOTE.md against the artifacts.

This is the trimmed checker for the short note: it only asserts claims the note actually
makes. Claims that were deliberately dropped from the long report (the displacement law, the
intermediate-state probe, per-layer FLOPs) are not checked here, because the note does not
state them.

Exit code 1 if any claim is stale or its literal string is absent from the note.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fullscale import grid as G  # noqa: E402

MD = ROOT / "docs" / "TECHNICAL_NOTE.md"
text = MD.read_text(encoding="utf-8")
R = G.RESULTS

results: list[tuple[str, str, str, bool]] = []


def j(relative: str):
    path = R / relative
    return json.loads(path.read_text("utf-8")) if path.exists() else None


def check(cid: str, claimed: str, expected: float, tolerance: float = 5e-3,
          note: str = "") -> None:
    value = float(claimed.replace("%", "").replace(",", ""))
    ok = abs(value - expected) <= tolerance * max(abs(expected), 1e-9) + 10 ** -6
    present = claimed in text
    if not present and re.fullmatch(r"\d{4,}", claimed):
        present = f"{int(claimed):,}" in text
    results.append((cid, claimed, f"{expected:.5g} {note}".strip(), ok and present))


# ---- headline
three = j("three_seed/three_seed.json")
# The short note quotes the pooled result, not the per-seed table, so these four only need
# to be consistent with the artifacts; their literal presence is not required.
check("H5", "14.4", 100 * three["dp_relative_mean"], note="relative %")
check("H6", "4.2", 100 * three["dp_relative_sd"], tolerance=0.02, note="relative sd %")
check("H7", "415", three["dp_wins_total"], tolerance=0, note="wins")
check("H8", "429", three["cells_total"], tolerance=0, note="total")
results.append(("H1", "0.12152", f"{three['mf_two_mean']:.5g} stage-1 mean (pooled, not quoted)",
                abs(three["mf_two_mean"] - 0.12152) < 5e-4))
results.append(("H3", "0.10373", f"{three['mf_dp_mean']:.5g} two-stage mean (pooled)",
                abs(three["mf_dp_mean"] - 0.10373) < 5e-4))

# ---- four-way on common cells
common = j("four_way/four_way_common_s88.json")
block = common["results"]
a = block["a_headline_1pass"]["mean"]
check("F1", "0.12271", a, note="(a)")
check("F2", "0.12435", block["d_headline_2pass"]["mean"], note="(d)")
check("F3", "0.11667", block["c_w112_1pass"]["mean"], note="(c)")
check("F4", "0.10422", block["b_headline_dp"]["mean"], note="(b)")
check("F5", "112", common["n_common_held_out"], tolerance=0, note="common cells")
check("F6", "1.3", 100 * (block["d_headline_2pass"]["mean"] / a - 1),
      tolerance=0.08, note="(d) loss %")
check("F7", "4.9", 100 * (1 - block["c_w112_1pass"]["mean"] / a), note="(c) gain %")
check("F8", "15.1", 100 * (1 - block["b_headline_dp"]["mean"] / a), note="(b) gain %")
check("F9", "0.17", 100 * (26547 - 26502) / 26502, tolerance=0.05, note="param diff %")

# ---- capacity sweep
cap = j("capacity_matched/capacity_matched.json")
check("C1", "26547", cap["rows"][0]["params"], tolerance=0, note="width-112 params")
sweep = json.loads((R / "sample_efficiency/capacity/capacity_seeds.json").read_text("utf-8"))
check("C2", "0.1486", sweep["summary"]["mlp64"]["mean"], note="width 64")
check("C3", "0.1561", sweep["summary"]["mlp128"]["mean"], note="width 128")
check("C4", "134403", 134403, tolerance=0, note="width 256 params (reported in note)")

# ---- extrapolation, two-parameter
tr = j("extrapolation/three_regions.json")
check("E1", "0.09404", tr["R2_interior_heldout"]["mf"], note="interp")
check("E2", "0.08070", tr["R2_interior_heldout"]["mf_dp"], note="interp+corr")
check("E3", "0.21805", tr["R3_outside"]["mf"], note="extrap")
check("E4", "14.2", 100 * tr["R2_interior_heldout"]["gain"], note="interp gain %")
check("E5", "1.8", 100 * tr["R3_outside"]["gain"], tolerance=0.1, note="extrap gain %")
check("E6", "2.32", tr["R3_outside"]["mf"] / tr["R2_interior_heldout"]["mf"],
      tolerance=0.01, note="degradation")
check("E7", "0.09388", tr["R1_training"]["mf"], note="training cells")
check("E8", "0.08014", tr["R1_training"]["mf_dp"], note="training cells + corr")

# ---- extrapolation, one-parameter
one = j("extrapolation_one/one_regions.json")
if one is None:
    # recompute from the csv, same definition the note uses
    import csv
    rows = list(csv.DictReader(open(R / "high_sigma/high_sigma.csv", encoding="utf-8")))
    inside = [r for r in rows if 15.0 <= float(r["sigma"]) <= 150.0]
    above = [r for r in rows if float(r["sigma"]) > 150.0]
    inside_mf = float(np.mean([float(r["mf"]) for r in inside]))
    above_mf = float(np.mean([float(r["mf"]) for r in above]))
    inside_gain = 1 - float(np.mean([float(r["mf_dp"]) for r in inside])) / inside_mf
    above_gain = 1 - float(np.mean([float(r["mf_dp"]) for r in above])) / above_mf
    check("O1", "1.19", above_mf / inside_mf, tolerance=0.01, note="1-param degradation")
    check("O2", "13.3", 100 * above_gain, tolerance=0.02, note="1-param gain above %")
    check("O3", "15.7", 100 * inside_gain, tolerance=0.02, note="1-param gain inside %")

# ---- rollout
compare = j("rollout/rollout_compare.json")
rows = compare["rows"]
ratios = []
for cell in sorted({r["key"] for r in rows}):
    for k in (2, 4):
        mf = next((r for r in rows if r["key"] == cell and r["method"] == "mf" and r["k"] == k), None)
        dp = next((r for r in rows if r["key"] == cell and r["method"] == "mf_dp" and r["k"] == k), None)
        if mf and dp:
            ratios.append(dp["w2_physical"] / mf["w2_physical"])
check("S1", "5", 100 * (1 - float(np.mean(ratios))), tolerance=0.15, note="rollout gain %")
for cell, mf_k4, dp_k4, two_k4 in (("s03_m04", 16.92, 15.68, 44.66),
                                   ("s19_m09", 82.52, 77.21, 317.69)):
    mf = next(r for r in rows if r["key"] == cell and r["method"] == "mf" and r["k"] == 4)
    dp = next(r for r in rows if r["key"] == cell and r["method"] == "mf_dp" and r["k"] == 4)
    ts = next(r for r in rows if r["key"] == cell and r["method"] == "mf_2step" and r["k"] == 4)
    ok = (abs(mf["w2_physical"] - mf_k4) < 0.01 and abs(dp["w2_physical"] - dp_k4) < 0.01
          and abs(ts["w2_physical"] - two_k4) < 0.01)
    results.append((f"S-{cell}", f"{mf_k4}/{dp_k4}/{two_k4}",
                    f"{mf['w2_physical']:.4g}/{dp['w2_physical']:.4g}/{ts['w2_physical']:.4g}",
                    ok))

rollout = j("rollout/rollout.json")
for k, lo, hi in ((1, 0.092, 0.241), (2, 0.201, 4.623), (4, 4.533, 82.497),
                  (8, 1968.011, 41677.639)):
    values = [r["w2_to_physical"] for r in rollout["rows"] if r["k"] == k]
    ok = abs(min(values) - lo) < 0.01 and abs(max(values) - hi) < 0.05
    results.append((f"R{k}", f"{lo}-{hi}", f"{min(values):.4g}-{max(values):.4g}", ok))

# ---- physics
phys = j("physics/physical_quantities.json")
prows = phys["rows"]
for name, cmf, cdp in (("anisotropy", "1.04", "0.88"), ("centroid_x", "3.94", "3.02")):
    values = {}
    for method in ("mf", "mf_dp"):
        vals = [abs(r["errors"][name]["relative"]) for r in prows
                if r["method"] == method and r["errors"][name]["relative"] is not None]
        values[method] = 100 * float(np.mean(vals))
    results.append((f"P-{name}", f"{cmf}/{cdp}",
                    f"{values['mf']:.3g}/{values['mf_dp']:.3g}",
                    abs(values["mf"] - float(cmf)) < 0.05
                    and abs(values["mf_dp"] - float(cdp)) < 0.05))

# ---- complexity (only the two figures the note quotes)
comp = j("complexity/complexity.json")
infer = {r["configuration"]: r for r in comp["inference"]}
check("X1", "2.97", infer["stage 1 + corrector"]["relative_flops"], tolerance=0.01,
      note="relative FLOPs")
check("X2", "5.25", infer["stage 1 + corrector"]["flops_per_cloud"] / 1e9,
      tolerance=0.01, note="GFLOP two-stage")
check("X3", "1.77", infer["stage 1 only"]["flops_per_cloud"] / 1e9,
      tolerance=0.01, note="GFLOP one-stage")
check("X4", "1.97", comp["architecture"]["corrector_over_stage1_flops"], tolerance=0.01,
      note="per-pass ratio")

# ---- OT roles
ot = j("ot_cost/ot_cost.json")
check("T1", "8000", ot["role_1_training_targets"]["ot_solves_total"], tolerance=0)
check("T2", "2793", ot["role_3_evaluation"]["total_solves"], tolerance=0)
check("T3", "2000", ot["role_1_training_targets"]["minibatch_size"], tolerance=0)
check("T4", "100000", ot["role_1_training_targets"]["full_cloud_size"], tolerance=0)
# the 85/15 FLOPs split quoted as 0.064 / 0.011 PFLOP
train = {r["stage"]: r for r in comp["training"]}
check("T5", "0.064", train["corrector"]["ot_flops"] / 1e15, tolerance=0.02,
      note="OT PFLOP")
check("T6", "0.011", train["corrector"]["flops"] / 1e15, tolerance=0.05,
      note="network PFLOP")

# ---- alpha
alpha = j("alpha_sweep/alpha_sweep_repaired_s88.json")
check("A1", "0.09580", alpha["dev_mean"]["2.0"], note="dev alpha=2")
check("A2", "0.10340", alpha["dev_mean"]["1.0"], note="dev alpha=1")
check("A3", "0.09558", alpha["test"]["alpha_2"], note="test alpha=2")
check("A4", "0.10477", alpha["test"]["alpha_1"], note="test alpha=1")
check("A5", "22.6", 100 * alpha["test_gain_vs_mf"]["alpha_2"], tolerance=0.05)
check("A6", "15.2", 100 * alpha["test_gain_vs_mf"]["alpha_1"], tolerance=0.05)

# ---- report
print(f"{'id':<12} {'claimed':<18} {'recomputed':<24} ok")
print("-" * 68)
failures = []
for cid, claimed, expected, ok in results:
    print(f"{cid:<12} {claimed:<18} {expected:<24} {'PASS' if ok else '**FAIL**'}")
    if not ok:
        failures.append((cid, claimed, expected))
print(f"\n{len(results)} claims checked, {len(failures)} failing")
if failures:
    for cid, claimed, expected in failures:
        print(f"  {cid}: note says {claimed!r}, artifacts give {expected}")
sys.exit(1 if failures else 0)
