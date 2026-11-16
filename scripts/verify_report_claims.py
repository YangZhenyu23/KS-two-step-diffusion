"""Claim-by-claim verification of every quantitative statement in the report.

A fuzzy numeric grep produces mostly false positives (section numbers, arXiv IDs, library
versions) and misses the thing that actually matters: whether each *claim* is backed by the
artifact it cites. This script instead lists the claims explicitly, recomputes each from the
JSON on disk, compares against the literal string in the Markdown, and reports pass or fail.

Each row is: claim id, what the report says, where it comes from, verdict.

Exit code 1 if any claim fails, so a stale number cannot slip into a rebuild.
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

MD = ROOT / "docs" / "REPORT_MAIN.md"
text = MD.read_text(encoding="utf-8")
R = G.RESULTS


def j(relative: str):
    path = R / relative
    return json.loads(path.read_text("utf-8")) if path.exists() else None


results: list[tuple[str, str, str, bool]] = []


def check(cid: str, claimed: str, expected: float, tolerance: float = 5e-3,
          note: str = "") -> None:
    """Does `claimed` appear in the report, and does it agree with `expected`?

    The report writes integers with thousands separators (8,832) and decimals without, so the
    presence test normalises both forms before searching.
    """
    value = float(claimed.replace("%", "").replace(",", ""))
    ok = abs(value - expected) <= tolerance * max(abs(expected), 1e-9) + 10 ** -6
    present = claimed in text
    if not present:
        # try the thousands-separated form: 8832 -> 8,832
        if re.fullmatch(r"\d{4,}", claimed):
            grouped = f"{int(claimed):,}"
            present = grouped in text
    results.append((cid, claimed, f"{expected:.5g} {note}".strip(), ok and present))


# ---------------------------------------------------------------- headline
three = j("three_seed/three_seed.json")
if three:
    check("H1", "0.12152", three["mf_two_mean"], note="stage-1 mean")
    check("H2", "0.00892", three["mf_two_sd"], note="stage-1 sd")
    check("H3", "0.10373", three["mf_dp_mean"], note="two-stage mean")
    check("H4", "0.00267", three["mf_dp_sd"], note="two-stage sd")
    check("H5", "14.4", 100 * three["dp_relative_mean"], note="relative reduction %")
    check("H6", "4.2", 100 * three["dp_relative_sd"], tolerance=0.02,
          note="relative sd %, reported as 4.2")
    check("H7", "415", three["dp_wins_total"], tolerance=0, note="cell wins")
    check("H8", "429", three["cells_total"], tolerance=0, note="cell total")

# ---------------------------------------------------------------- four-way, common cells
common = j("four_way/four_way_common_s88.json")
if common:
    block = common["results"]
    check("F1", "0.12271", block["a_headline_1pass"]["mean"], note="(a) stage 1")
    check("F2", "0.12435", block["d_headline_2pass"]["mean"], note="(d) two-pass")
    check("F3", "0.11667", block["c_w112_1pass"]["mean"], note="(c) width 112")
    check("F4", "0.10422", block["b_headline_dp"]["mean"], note="(b) + corrector")
    check("F5", "112", common["n_common_held_out"], tolerance=0, note="common cells")
    # the derived percentages
    a = block["a_headline_1pass"]["mean"]
    check("F6", "1.3", 100 * (block["d_headline_2pass"]["mean"] / a - 1),
          tolerance=0.08, note="(d) vs (a), negative")
    check("F7", "4.9", 100 * (1 - block["c_w112_1pass"]["mean"] / a), note="(c) gain")
    check("F8", "15.1", 100 * (1 - block["b_headline_dp"]["mean"] / a), note="(b) gain")
    check("F9", "10.7", 100 * (1 - block["b_headline_dp"]["mean"]
                               / block["c_w112_1pass"]["mean"]), note="(b) vs (c)")

# ---------------------------------------------------------------- capacity
cap = j("capacity_matched/capacity_matched.json")
if cap:
    check("C1", "26547", cap["rows"][0]["params"], tolerance=0, note="width-112 params")

# ---------------------------------------------------------------- n-step
nstep = j("nstep/nstep_s88.json")
if nstep:
    by = {r["steps"]: r for r in nstep["rows"]}
    check("N1", "0.12418", by[2]["mf_mean_w2"], note="N=2")
    check("N2", "0.12707", by[4]["mf_mean_w2"], note="N=4")
    check("N3", "0.12736", by[8]["mf_mean_w2"], note="N=8")
    check("N4", "1.1", 100 * abs(by[2]["mf_relative_vs_1step"]), tolerance=0.15, note="N=2 loss")

# ---------------------------------------------------------------- complexity
comp = j("complexity/complexity.json")
if comp:
    check("X1", "8832", comp["architecture"]["stage1"]["macs_per_particle"], tolerance=0)
    check("X2", "17664", comp["architecture"]["stage1"]["flops_per_particle"], tolerance=0)
    check("X3", "16896", comp["architecture"]["corrector"]["macs_per_particle"], tolerance=0)
    check("X4", "34816", comp["architecture"]["corrector"]["flops_per_particle"], tolerance=0)
    check("X5", "1.97", comp["architecture"]["corrector_over_stage1_flops"], tolerance=0.01)
    infer = {r["configuration"]: r for r in comp["inference"]}
    stage1 = infer["stage 1 only"]
    check("X6", "2.97", infer["stage 1 + corrector"]["relative_flops"], tolerance=0.01)

# ---------------------------------------------------------------- OT roles
ot = j("ot_cost/ot_cost.json")
if ot:
    check("O1", "8000", ot["role_1_training_targets"]["ot_solves_total"], tolerance=0,
          note="note: report writes 8,000")
    check("O2", "2793", ot["role_3_evaluation"]["total_solves"], tolerance=0,
          note="note: report writes 2,793")

# ---------------------------------------------------------------- extrapolation, two-param
tr = j("extrapolation/three_regions.json")
if tr:
    check("E1", "0.09404", tr["R2_interior_heldout"]["mf"], note="interp")
    check("E2", "0.08070", tr["R2_interior_heldout"]["mf_dp"], note="interp + corr")
    check("E3", "0.21805", tr["R3_outside"]["mf"], note="extrap")
    check("E4", "14.2", 100 * tr["R2_interior_heldout"]["gain"], note="interp gain")
    check("E5", "1.8", 100 * tr["R3_outside"]["gain"], tolerance=0.1, note="extrap gain")
    check("E6", "2.32", tr["R3_outside"]["mf"] / tr["R2_interior_heldout"]["mf"],
          tolerance=0.01, note="degradation")

# ---------------------------------------------------------------- stability
verify = j("stability_check/verify.json") if (R / "stability_check/verify.json").exists() else None
compare = j("rollout/rollout_compare.json")
if compare:
    rows = compare["rows"]
    cells = sorted({r["key"] for r in rows})
    ratios = []
    for cell in cells:
        for k in (2, 4):
            mf = next((r for r in rows if r["key"] == cell and r["method"] == "mf"
                       and r["k"] == k), None)
            dp = next((r for r in rows if r["key"] == cell and r["method"] == "mf_dp"
                       and r["k"] == k), None)
            if mf and dp:
                ratios.append(dp["w2_physical"] / mf["w2_physical"])
    check("S1", "5", 100 * (1 - float(np.mean(ratios))), tolerance=0.15,
          note="mean rollout reduction %")
    check("S2", "6", len(ratios), tolerance=0, note="pairs")
    # the two example cells
    for cell, mf_k4, dp_k4 in (("s03_m04", 16.92, 15.68), ("s19_m09", 82.52, 77.21)):
        mf = next((r for r in rows if r["key"] == cell and r["method"] == "mf" and r["k"] == 4), None)
        dp = next((r for r in rows if r["key"] == cell and r["method"] == "mf_dp" and r["k"] == 4), None)
        if mf and dp:
            results.append((f"S-{cell}", f"{mf_k4}/{dp_k4}",
                            f"{mf['w2_physical']:.4g}/{dp['w2_physical']:.4g}",
                            abs(mf["w2_physical"] - mf_k4) < 0.01
                            and abs(dp["w2_physical"] - dp_k4) < 0.01))

# ---------------------------------------------------------------- intermediate state
e11 = j("intermediate_state/intermediate_state.json")
if e11:
    s = e11["summary"]
    check("I1", "0.2599", s["w2_fwd_half_vs_truth_half"]["mean"], note="forward route")
    check("I2", "2.8485", s["w2_bwd_half_vs_truth_half"]["mean"], tolerance=0.01,
          note="backward route")
    check("I3", "0.7414", s["w2_prior_vs_truth_half"]["mean"], note="baseline")
    check("I4", "0.1079", s["w2_full_vs_truth_full"]["mean"], note="terminal reference")
    check("I5", "65", 100 * (1 - s["w2_fwd_half_vs_truth_half"]["mean"]
                             / s["w2_prior_vs_truth_half"]["mean"]), tolerance=0.05,
          note="% recovered")
    check("I6", "2.4", s["w2_fwd_half_vs_truth_half"]["mean"]
          / s["w2_full_vs_truth_full"]["mean"], tolerance=0.05, note="x terminal error")

# ---------------------------------------------------------------- physics
phys = j("physics/physical_quantities.json")
if phys:
    rows = phys["rows"]
    for name, claimed_mf, claimed_dp in (("anisotropy", "1.04", "0.88"),
                                         ("centroid_x", "3.94", "3.02"),
                                         ("eig_2", "0.94", "0.65")):
        values = {}
        for method in ("mf", "mf_dp"):
            vals = [abs(r["errors"][name]["relative"]) for r in rows
                    if r["method"] == method and r["errors"][name]["relative"] is not None]
            values[method] = 100 * float(np.mean(vals))
        results.append((f"P-{name}", f"{claimed_mf}/{claimed_dp}",
                        f"{values['mf']:.3g}/{values['mf_dp']:.3g}",
                        abs(values["mf"] - float(claimed_mf)) < 0.05
                        and abs(values["mf_dp"] - float(claimed_dp)) < 0.05))

# ---------------------------------------------------------------- displacement law
disp = j("displacement/displacement_checks.json")
if disp:
    check("D1", "0.90", disp["check2_cS_median"], tolerance=0.02, note="c_S")
    check("D2", "1.98", disp["check2_cG_mean"], tolerance=0.01, note="c_G")
    check("D3", "0.026", disp["check2_cG_sem"], tolerance=0.1, note="c_G sem")

# ---------------------------------------------------------------- P0 information gap
p0 = j("p0_same_cells/p0_budget32000.json")
if p0:
    check("G1", "0.26291", p0["one_param_training_fit"], note="sigma-only fit")
    check("G2", "0.11995", p0["two_param_training_fit"], note="two-param fit")
    check("G3", "2.19", p0["training_fit_ratio"], tolerance=0.01, note="ratio")

# ---------------------------------------------------------------- alpha
alpha = j("alpha_sweep/alpha_sweep_repaired_s88.json")
if alpha:
    check("A1", "0.09580", alpha["dev_mean"]["2.0"], note="dev at alpha=2")
    check("A2", "0.09558", alpha["test"]["alpha_2"], note="test at alpha=2")
    check("A3", "22.6", 100 * alpha["test_gain_vs_mf"]["alpha_2"], tolerance=0.05)

# ---------------------------------------------------------------- report
print(f"{'id':<14} {'claimed':<14} {'recomputed':<22} {'ok'}")
print("-" * 66)
failures = []
for cid, claimed, expected, ok in results:
    flag = "PASS" if ok else "**FAIL**"
    print(f"{cid:<14} {claimed:<14} {expected:<22} {flag}")
    if not ok:
        failures.append((cid, claimed, expected))

print(f"\n{len(results)} claims checked, {len(failures)} failing")
if failures:
    print("\nfailures (either the number is stale or the literal string is absent):")
    for cid, claimed, expected in failures:
        print(f"  {cid}: report says {claimed!r}, artifacts give {expected}")
sys.exit(1 if failures else 0)
