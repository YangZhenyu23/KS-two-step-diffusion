"""Complete OT cost accounting, separating three places OT is actually used.

The requirement is: "Report both the cost of constructing transport targets during training
and deployment cost; avoiding global OT is different from eliminating all OT computation,
particularly when the corrector still uses minibatch W2 or warm-start OT."

Three distinct OT roles exist in this project, and conflating them is what the sentence warns
against. They are separated here:

  1. TRAINING TARGET CONSTRUCTION
     Every `ot_refresh` steps the corrector's regression target is rebuilt by solving an exact
     OT problem on a minibatch of 2,000 particles. This is minibatch W2 - exactly the case the
     requirement names. No Sinkhorn, no warm start: `ot.emd`, network simplex.

  2. DEPLOYMENT (inference)
     Zero OT. The trained corrector is a pure feed-forward map; nothing is matched at
     inference time. This is the only genuine saving, and it is worth stating precisely:
     what is avoided is a *global* OT solve between full clouds, not OT as a concept.

  3. EVALUATION
     Scoring any generated cloud against a reference requires a W2 computation, which is an OT
     solve. It is exact (`ot.emd2`), on 16-bin histograms in the headline protocol and on
     subsampled particles in the particle-level protocol. This cost is borne by us as
     experimenters, not by a user of the model - but it exists and must be reported, because
     otherwise "OT-free" overstates the case.

Also reported: the minibatch size used for targets vs the full cloud size, since a minibatch
approximation is not the same object as the global OT it approximates.

Usage:
    python scripts/ot_cost_audit.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream" / "MF-DeepParticle" / "deepparticle"))

from fullscale import grid as G  # noqa: E402

OUT = G.RESULTS / "ot_cost"

# from the code, read not guessed
TARGET_BATCH = 2_000          # dp_two_param.py: batch=2000, ot.emd on that batch
OT_REFRESH = 10               # dp_two_param.py: refresh every 10 updates
DP_UPDATES_PER_CELL = 2_000
N_TRAIN_CELLS = 40
N_PARTICLES = 100_000
HEADLINE_BINS = 16
PARTICLE_LEVEL_SUBSAMPLE = 3_000
LARGEST_EXACT = 30_000


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    dp_steps = DP_UPDATES_PER_CELL * N_TRAIN_CELLS
    target_solves = dp_steps // OT_REFRESH

    # evaluation solves actually performed across the study, from the artifacts
    eval_solves = {
        "headline protocol (three seeds, 143 cells)": 3 * 143,
        "DP stage evaluation (three seeds, 143 cells)": 3 * 143,
        "n-step sweep (4 configurations, 143 cells)": 4 * 143,
        "four-way on common cells (4 configs, 112 cells)": 4 * 112,
        "alpha sweep (8 alphas dev + 3 test)": 8 * 50 + 3 * 93,
        "extrapolation (200 cells)": 200,
        "rollout study (6 cells x 4 horizons)": 6 * 4,
        "physical quantities (6 cells x 2 methods)": 6 * 2,
    }
    total_eval = sum(eval_solves.values())

    payload = {
        "role_1_training_targets": {
            "solver": "ot.emd (POT network simplex, exact)",
            "sinkhorn_used": False,
            "warm_start_used": False,
            "minibatch_size": TARGET_BATCH,
            "full_cloud_size": N_PARTICLES,
            "minibatch_fraction": TARGET_BATCH / N_PARTICLES,
            "refresh_interval_updates": OT_REFRESH,
            "corrector_updates_total": dp_steps,
            "ot_solves_total": target_solves,
            "cost_per_solve": "O(n^3) time, O(n^2) memory with n = minibatch size",
            "memory_per_solve_mb": 8 * TARGET_BATCH ** 2 / 1024 ** 2,
            "is_global_ot": False,
            "note": ("This is the minibatch W2 that the requirement names. It is a "
                     "2,000-of-100,000 approximation to the global OT it stands in for."),
        },
        "role_2_deployment": {
            "ot_solves": 0,
            "note": ("The trained corrector is a feed-forward map; no matching happens at "
                     "inference. What is saved is the *global* OT between full clouds, not "
                     "OT as a concept."),
        },
        "role_3_evaluation": {
            "solver": "ot.emd2 (POT network simplex, exact)",
            "sinkhorn_used": False,
            "headline_representation": f"{HEADLINE_BINS}-bin histogram",
            "particle_level_subsample": PARTICLE_LEVEL_SUBSAMPLE,
            "largest_exact_feasible": LARGEST_EXACT,
            "per_configuration": eval_solves,
            "total_solves": total_eval,
            "note": ("Borne by the experimenter, not by a model user. Reported because "
                     "'OT-free' would otherwise overstate the case: the metric itself is an "
                     "OT solve."),
        },
        "summary": {
            "training_target_solves": target_solves,
            "deployment_solves": 0,
            "evaluation_solves": total_eval,
            "statement": ("OT appears in two of three roles. Deployment is genuinely OT-free; "
                          "training and evaluation are not."),
        },
    }
    (OUT / "ot_cost.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("role 1 - training target construction")
    print(f"  solver            : {payload['role_1_training_targets']['solver']}")
    print(f"  sinkhorn / warm   : none / none")
    print(f"  minibatch         : {TARGET_BATCH:,} of {N_PARTICLES:,} particles "
          f"({TARGET_BATCH / N_PARTICLES:.1%})")
    print(f"  refresh interval  : every {OT_REFRESH} updates")
    print(f"  solves total      : {target_solves:,}  "
          f"({dp_steps:,} updates / {OT_REFRESH})")
    print(f"  memory per solve  : {payload['role_1_training_targets']['memory_per_solve_mb']:.0f} MB")
    print(f"  complexity        : O(n^3) time, O(n^2) memory")

    print("\nrole 2 - deployment")
    print("  OT solves         : 0")

    print("\nrole 3 - evaluation (the role usually left out)")
    print(f"  solver            : ot.emd2, exact, no Sinkhorn")
    print(f"  headline rep.     : {HEADLINE_BINS}-bin histogram")
    print(f"  {'batch':<52} {'solves':>7}")
    for label, count in eval_solves.items():
        print(f"  {label:<52} {count:>7,}")
    print(f"  {'TOTAL':<52} {total_eval:>7,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
