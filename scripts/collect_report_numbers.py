"""Collect every number that will appear in the report into one auditable file.

The report has grown across many rounds and a few figures were superseded (repaired
held-out sets, the common-cell four-way table, corrected complexity accounting). Before
writing a clean document, every number must come from one place so the prose cannot drift
from the artifacts.

Writes results/report_numbers.json and prints a compact digest.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream" / "MF-DeepParticle" / "deepparticle"))

from fullscale import grid as G  # noqa: E402

R = G.RESULTS
OUT = R / "report_numbers.json"


def j(relative: str):
    path = R / relative
    return json.loads(path.read_text("utf-8")) if path.exists() else None


numbers = {}

# --- three-seed headline
three = j("three_seed/three_seed.json")
if three:
    numbers["headline"] = {
        "seeds": three["seeds"],
        "mf_mean": three["mf_two_mean"], "mf_sd": three["mf_two_sd"],
        "dp_mean": three["mf_dp_mean"], "dp_sd": three["mf_dp_sd"],
        "relative_mean": three["dp_relative_mean"], "relative_sd": three["dp_relative_sd"],
        "t": three["dp_relative_t"],
        "wins": three["dp_wins_total"], "cells": three["cells_total"],
        "n_held_out": three["n_held_out"],
        "one_param_mean": three["one_param_mean"],
        "two_param_mean": three["two_param_mean"],
        "two_param_relative": three["two_param_relative_mean"],
        "two_param_t": three["two_param_relative_t"],
        "end_to_end": three["end_to_end_relative_mean"],
    }

# --- four-way on the common cells (the clean version)
common = j("four_way/four_way_common_s88.json")
if common:
    numbers["four_way_common"] = {
        "n_common": common["n_common_held_out"],
        "overlap_of_designs": common["overlap"],
        "values": {k: v["mean"] for k, v in common["results"].items()},
        "relative_vs_a": {k: v["relative_vs_a"] for k, v in common["results"].items()},
    }

# --- n-step
nstep = j("nstep/nstep_s88.json")
if nstep:
    numbers["nstep"] = {r["steps"]: {"mf": r["mf_mean_w2"], "mf_dp": r["mf_dp_mean_w2"],
                                     "rel": r["mf_relative_vs_1step"]}
                        for r in nstep["rows"]}

# --- complexity
complexity = j("complexity/complexity.json")
if complexity:
    numbers["complexity"] = {
        "stage1_flops": complexity["architecture"]["stage1"]["flops_per_particle"],
        "stage1_macs": complexity["architecture"]["stage1"]["macs_per_particle"],
        "dp_flops": complexity["architecture"]["corrector"]["flops_per_particle"],
        "dp_macs": complexity["architecture"]["corrector"]["macs_per_particle"],
        "ratio": complexity["architecture"]["corrector_over_stage1_flops"],
        "inference": {row["configuration"]: {"passes": row["passes"] if "passes" in row
                                             else row["total_passes"],
                                             "gflop": row["flops_per_cloud"] / 1e9,
                                             "relative": row["relative_flops"]}
                      for row in complexity["inference"]},
        "training": complexity["training"],
        "memory": complexity["memory"],
    }

# --- OT roles
ot = j("ot_cost/ot_cost.json")
if ot:
    numbers["ot"] = {
        "train_solves": ot["role_1_training_targets"]["ot_solves_total"],
        "minibatch": ot["role_1_training_targets"]["minibatch_size"],
        "full_cloud": ot["role_1_training_targets"]["full_cloud_size"],
        "refresh": ot["role_1_training_targets"]["refresh_interval_updates"],
        "solver": ot["role_1_training_targets"]["solver"],
        "deploy_solves": 0,
        "eval_solves": ot["role_3_evaluation"]["total_solves"],
    }

# --- extrapolation, both settings
three_regions = j("extrapolation/three_regions.json")
if three_regions:
    numbers["extrapolation_two"] = three_regions

high = ROOT / "results" / "fullscale" / "high_sigma" / "high_sigma.csv"
if high.exists():
    rows = list(csv.DictReader(open(high, encoding="utf-8")))
    inside = [r for r in rows if 15.0 <= float(r["sigma"]) <= 150.0]
    above = [r for r in rows if float(r["sigma"]) > 150.0]
    below = [r for r in rows if float(r["sigma"]) < 15.0]

    def agg(group):
        mf = float(np.mean([float(r["mf"]) for r in group]))
        dp = float(np.mean([float(r["mf_dp"]) for r in group]))
        return {"n": len(group), "mf": mf, "mf_dp": dp, "gain": 1 - dp / mf}

    numbers["extrapolation_one"] = {
        "inside": agg(inside), "above": agg(above), "below": agg(below),
        "train_min": 15.0, "train_max": 150.0,
        "degradation": agg(above)["mf"] / agg(inside)["mf"],
    }

# --- stability
verify = {}
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
    verify["rollout_ratio_mean"] = float(np.mean(ratios))
    verify["rollout_pairs"] = len(ratios)
    verify["rollout_all_below_one"] = bool((np.array(ratios) < 1).all())
    verify["floor"] = compare["floor_mean"]

rollout = j("rollout/rollout.json")
if rollout:
    verify["rollout_by_k"] = {}
    for k in sorted({r["k"] for r in rollout["rows"]}):
        values = [r["w2_to_physical"] for r in rollout["rows"] if r["k"] == k]
        verify["rollout_by_k"][k] = {
            "min": min(values), "max": max(values),
            "over_floor_min": min(values) / rollout["floor_mean"],
            "over_floor_max": max(values) / rollout["floor_mean"],
        }

metrics_csv = ROOT / "results" / "ks3d" / "evaluation" / "ks" / "metrics_per_seed.csv"
if metrics_csv.exists():
    rows = list(csv.DictReader(open(metrics_csv, encoding="utf-8")))
    from collections import defaultdict
    groups = defaultdict(list)
    for row in rows:
        groups[(float(row["level"]), row["method"])].append(float(row["w2"]))
    for method in ("mf", "mf_dp"):
        per = [v for (lv, m), v in groups.items() if m == method and len(v) > 1]
        sds = [float(np.std(v, ddof=1)) for v in per]
        means = [float(np.mean(v)) for v in per]
        verify[f"single_param_{method}"] = {"mean": float(np.mean(means)),
                                            "mean_sd": float(np.mean(sds))}
numbers["stability"] = verify

# --- physics
physics = j("physics/physical_quantities.json")
if physics:
    rows = physics["rows"]
    table = {}
    for name in ("centroid_x", "rms_radius", "anisotropy", "eig_1", "eig_2"):
        entry = {}
        for method in ("mf", "mf_dp"):
            values = [abs(r["errors"][name]["relative"]) for r in rows
                      if r["method"] == method and r["errors"][name]["relative"] is not None]
            entry[method] = float(np.mean(values))
        table[name] = entry
    numbers["physics"] = table

# --- displacement law
disp = j("displacement/displacement_checks.json")
if disp:
    numbers["displacement"] = {
        "cS_median": disp.get("check2_cS_median"),
        "cG_mean": disp.get("check2_cG_mean"),
        "cG_sem": disp.get("check2_cG_sem"),
        "sqrt3": disp.get("check2_sqrt3"),
        "L_sigma_spread": disp.get("check1_L_sigma_spread"),
        "loglog_slope": disp.get("check1_loglog_slope"),
        "check3_median": disp.get("check3_ratio_median"),
    }

# --- information gap (P0)
p0 = j("p0_same_cells/p0_budget32000.json")
if p0:
    numbers["p0"] = {
        "budget": p0["budget_per_cell"],
        "one_train": p0["one_param_training_fit"],
        "two_train": p0["two_param_training_fit"],
        "ratio": p0["training_fit_ratio"],
        "gap": p0["training_fit_gap"],
    }

# --- alpha
alpha = j("alpha_sweep/alpha_sweep_repaired_s88.json")
if alpha:
    numbers["alpha"] = {
        "dev": alpha["dev_mean"], "selected": alpha["selected_alpha"],
        "test": alpha["test"], "gains": alpha["test_gain_vs_mf"],
        "n_dev": alpha["n_dev"], "n_test": alpha["n_test"],
    }

# --- capacity
capacity = j("capacity_matched/capacity_matched.json")
if capacity:
    row = capacity["rows"][0]
    numbers["capacity"] = {"width": row["width"], "params": row["params"],
                           "w2": row["held_out_mean_w2"]}

OUT.write_text(json.dumps(numbers, indent=2), encoding="utf-8")

print(f"wrote {OUT.relative_to(ROOT)}")
for section, block in numbers.items():
    print(f"\n[{section}]")
    if isinstance(block, dict):
        for key, value in list(block.items())[:8]:
            if isinstance(value, float):
                print(f"  {key:<28} {value:.5f}")
            elif isinstance(value, (int, str, bool)):
                print(f"  {key:<28} {value}")
            else:
                print(f"  {key:<28} <{type(value).__name__}>")
