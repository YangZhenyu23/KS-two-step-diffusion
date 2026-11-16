"""Verify which NETWORK ARCHITECTURE each of the four configurations uses.

The previous audit checked training cells and held-out cells. The user's point is that this
is not enough: if the four configurations use different *architectures* (7-input two-parameter
vs 6-input single-parameter), then identical cell sets do not make the comparison meaningful.
A wider single-parameter net and a two-parameter net are different model families.

For each configuration this reads the checkpoint and prints the architecture actually stored
inside it, because that is authoritative and cannot drift from what was trained.

What to look for
    input_features = 7  -> two-parameter network (z, t, r, log sigma, log mu)
    input_features = 6  -> single-parameter network (z, t, r, log sigma)

Usage:
    python scripts/audit_architectures.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream" / "MF-DeepParticle" / "deepparticle"))

from fullscale import grid as G  # noqa: E402

R = G.RESULTS

CONFIGS = [
    ("(a) global map alone", R / "optimal_budget_ablation/two_32000/mf_two_param.pt"),
    ("(b) map + corrector: stage 1", R / "optimal_budget_ablation/two_32000/mf_two_param.pt"),
    ("(b) map + corrector: corrector", R / "dp_two_param/s88/dp_corrector.pt"),
    ("(c) capacity-matched predictor", R / "capacity_matched/w112_s88/mf_two_param.pt"),
    ("(d) few-step flow baseline", R / "optimal_budget_ablation/two_32000/mf_two_param.pt"),
    ("--- reference: single-param net ---", R / "optimal_budget_ablation/one_32000/mf_one_param.pt"),
]


def describe(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    arch = payload.get("architecture", {})
    state = payload.get("model", payload.get("state_dict", {}))

    # infer input width from the first linear layer, which is authoritative
    first_key = None
    for key in state:
        if key.endswith("weight") and ("input" in key or "net.0" in key):
            first_key = key
            break
    in_features = None
    params = None
    if first_key is not None:
        in_features = int(state[first_key].shape[1])
    try:
        params = sum(v.numel() for v in state.values())
    except Exception:
        params = None

    return {
        "architecture": arch,
        "in_features": in_features,
        "params": params,
        "kind": arch.get("kind", "?"),
        "cells": len(payload.get("cells", [])) if payload.get("cells") else None,
    }


def main() -> int:
    print("=" * 96)
    print("ARCHITECTURE ACTUALLY STORED IN EACH CHECKPOINT")
    print("=" * 96)
    print(f"{'configuration':<36} {'kind':>12} {'in_feat':>8} {'params':>9} {'cond':>22}")
    print("-" * 96)
    findings = {}
    for label, path in CONFIGS:
        if not path.exists():
            print(f"{label:<36} {'MISSING':>12}")
            continue
        info = describe(path)
        arch = info["architecture"]
        cond = arch.get("conditioning") or arch.get("condition_mode") or (
            "log10(sigma)" if info["in_features"] == 6 else "?")
        print(f"{label:<36} {info['kind']:>12} {str(info['in_features']):>8} "
              f"{str(info['params']):>9} {str(cond)[:22]:>22}")
        findings[label] = info

    print()
    print("=" * 96)
    print("VERDICT")
    print("=" * 96)
    a = findings.get("(a) global map alone")
    b = findings.get("(b) map + corrector: stage 1")
    c = findings.get("(c) capacity-matched predictor")
    d = findings.get("(d) few-step flow baseline")
    one = findings.get("--- reference: single-param net ---")

    if a and b and d:
        same = (a["in_features"] == b["in_features"] == d["in_features"]
                and a["architecture"] == b["architecture"] == d["architecture"])
        print(f"(a)(b)(d) share the same stage-1 architecture and checkpoint: {same}")
        print(f"  input features: {a['in_features']} -> "
              f"{'TWO-parameter (7 inputs)' if a['in_features'] == 7 else 'single-parameter (6 inputs)'}")

    if a and c:
        print(f"\n(c) is a {'SAME' if a['in_features'] == c['in_features'] else 'DIFFERENT'} "
              f"model family as (a):")
        print(f"  (a) input features {a['in_features']}, params {a['params']:,}")
        print(f"  (c) input features {c['in_features']}, params {c['params']:,}")
        if a["in_features"] == c["in_features"]:
            print("  => both are two-parameter networks; the only difference is width. "
                  "This is exactly what a capacity control must be.")
        else:
            print("  => WARNING: these are different model families. The capacity control is "
                  "invalid, because a wider single-parameter net cannot see mu at all.")

    if one:
        print(f"\nreference single-parameter checkpoint: input features {one['in_features']}, "
              f"params {one['params']:,}")
        print("  (this one is NOT used in the four-way table; it is the mu-extrapolation baseline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
