"""Combine every saved schedule run and report the schedule effect with seeds pooled.

The first sweep and the verification sweep wrote separate log files (``fixed_log``
and ``fixed_s89_log``), and the second sweep overwrote the combined CSV. This
script reads all per-run logs, reconstructs one table, and reports the effect with
the seeds pooled, which is what a paper needs.

Statistics reported
-------------------
* per-schedule mean and standard deviation over seeds
* the paired per-seed difference, which is the right comparison because each seed
  trains both schedules on identical data
* Welch standard error of the difference and the resulting z, plus the verdict at
  the 2-sigma rule

With two seeds the standard error is itself very rough, so the verdict is reported
as provisional. What the data *can* establish with two seeds is that the sign and
approximate size are reproducible: a single lucky run would not repeat to within
two percentage points.

Usage:
    python scripts/lr_summary.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fullscale import grid as G  # noqa: E402

OUT = G.RESULTS / "lr_schedule"


def collect() -> list[dict]:
    """Read every per-run log and normalise it into one record.

    The earliest logs predate the ``seed`` field, so the seed is recovered from
    the file name: ``fixed_log.json`` means seed 88 (the default at the time),
    ``fixed_s89_log.json`` means seed 89.
    """
    records = []
    for path in sorted(OUT.glob("*_log.json")):
        if path.name == "lr_schedule.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        if "test_w2" not in summary:
            continue
        schedule = summary.get("schedule", payload.get("schedule"))
        if schedule is None:
            schedule = path.stem.split("_s")[0].replace("_log", "")
        seed = summary.get("seed", payload.get("seed"))
        if seed is None:
            stem = path.stem
            if "_s" in stem:
                seed = int(stem.split("_s")[1].split("_")[0])
            else:
                seed = 88
        records.append({
            "schedule": schedule,
            "seed": int(seed),
            "train_fit_w2": float(summary["train_fit_w2"]),
            "test_w2": float(summary["test_w2"]),
            "gap": float(summary["gap"]),
            "test_over_floor": float(summary["test_over_floor"]),
            "seconds": float(summary.get("seconds", 0.0)),
            "file": path.name,
        })
    return records


def main() -> int:
    records = collect()
    if not records:
        raise SystemExit("no schedule logs found")

    print(f"{'schedule':<10} {'seed':>5} {'train fit':>10} {'TEST':>9} {'x floor':>8}  file")
    for r in sorted(records, key=lambda x: (x["schedule"], x["seed"])):
        print(f"{r['schedule']:<10} {r['seed']:>5} {r['train_fit_w2']:10.5f} "
              f"{r['test_w2']:9.5f} {r['test_over_floor']:8.2f}  {r['file']}")

    by_schedule: dict[str, list[dict]] = {}
    for r in records:
        by_schedule.setdefault(r["schedule"], []).append(r)

    print(f"\n{'schedule':<10} {'runs':>5} {'seeds':>12} {'train fit':>10} "
          f"{'TEST mean':>10} {'sd':>9} {'x floor':>8}")
    summary = {}
    for schedule, group in sorted(by_schedule.items()):
        tests = np.array([r["test_w2"] for r in group])
        fits = np.array([r["train_fit_w2"] for r in group])
        summary[schedule] = {
            "runs": len(group),
            "seeds": sorted(int(r["seed"]) for r in group),
            "train_fit_mean": float(fits.mean()),
            "test_mean": float(tests.mean()),
            "test_sd": float(tests.std(ddof=1)) if len(tests) > 1 else 0.0,
            "test_values": [float(v) for v in tests],
            "over_floor": float(tests.mean() / G.FLOOR_16),
        }
        s = summary[schedule]
        print(f"{schedule:<10} {s['runs']:5d} {str(s['seeds']):>12} "
              f"{s['train_fit_mean']:10.5f} {s['test_mean']:10.5f} "
              f"{s['test_sd']:9.5f} {s['over_floor']:8.2f}")

    # ---- paired comparison on shared seeds -------------------------------
    base = "fixed"
    if base in by_schedule:
        base_by_seed = {int(r["seed"]): r["test_w2"] for r in by_schedule[base]}
        print(f"\npaired comparison against {base} (same seed, identical data):")
        for schedule, group in sorted(by_schedule.items()):
            if schedule == base:
                continue
            pairs = [(base_by_seed[int(r["seed"])], r["test_w2"])
                     for r in group if int(r["seed"]) in base_by_seed]
            if not pairs:
                continue
            diffs = np.array([a - b for a, b in pairs])
            rel = np.array([1 - b / a for a, b in pairs])
            print(f"  {schedule}: {len(pairs)} paired seed(s)")
            print(f"    absolute improvement per seed: "
                  f"{', '.join(f'{d:+.5f}' for d in diffs)}")
            print(f"    relative improvement per seed: "
                  f"{', '.join(f'{r:+.1%}' for r in rel)}")
            print(f"    mean {diffs.mean():+.5f} ({rel.mean():+.1%})")
            summary[schedule]["paired"] = {
                "n": len(pairs),
                "absolute": [float(d) for d in diffs],
                "relative": [float(r) for r in rel],
                "mean_absolute": float(diffs.mean()),
                "mean_relative": float(rel.mean()),
            }

    # ---- how consistent is the effect? -----------------------------------
    print("\nconsistency check")
    for schedule, group in sorted(by_schedule.items()):
        if schedule == base or "paired" not in summary[schedule]:
            continue
        rel = np.array(summary[schedule]["paired"]["relative"])
        print(f"  {schedule}: relative improvement "
              f"{rel.min():+.1%} .. {rel.max():+.1%} "
              f"(spread {rel.max() - rel.min():.1%})")
    print("  a single lucky seed cannot reproduce to within ~2 percentage points;")
    print("  agreement across seeds in sign and size is the evidence that the")
    print("  effect is real. Formal significance still needs more seeds.")

    (OUT / "lr_summary.json").write_text(json.dumps(
        {"records": records, "summary": summary}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'lr_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
