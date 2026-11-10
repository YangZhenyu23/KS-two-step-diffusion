"""Plot the schedule comparison: per-seed errors and the training-loss traces.

Two panels:

  1. **Held-out error per schedule per seed.** Paired by seed so the comparison is
     on identical data. The point is that the ordering is the same seed by seed,
     not just on the mean.
  2. **Learning-rate trajectories.** Shows what the three schedules actually do,
     so the reader can see that the improvement tracks the rate decay rather than
     being an unexplained side effect.

A third panel plots the per-record training loss on a log scale, which makes
visible why the "tail loss variance" diagnostic used earlier was unusable: the
single-minibatch loss fluctuates over two orders of magnitude in every run,
regardless of the schedule.

Usage:
    python scripts/plot_lr_schedule.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fullscale import grid as G  # noqa: E402

OUT = G.RESULTS / "lr_schedule"


def main() -> int:
    summary_path = OUT / "lr_summary.json"
    if not summary_path.is_file():
        raise SystemExit("run scripts/lr_summary.py first")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    records = payload["records"]
    summary = payload["summary"]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0), layout="constrained")

    # ---- panel 1: paired per-seed errors ---------------------------------
    ax = axes[0]
    order = [s for s in ("fixed", "cosine", "warmcos") if s in summary]
    colours = {"fixed": "#7A1F1F", "cosine": "#D37B2F", "warmcos": "#168173"}
    for k, schedule in enumerate(order):
        group = [r for r in records if r["schedule"] == schedule]
        for r in group:
            jitter = 0.0
            ax.scatter(k + jitter, r["test_w2"], s=70, color=colours[schedule],
                       edgecolors="white", linewidths=.8, zorder=3)
        mean = summary[schedule]["test_mean"]
        ax.hlines(mean, k - .25, k + .25, color=colours[schedule], lw=2.4, zorder=4)
        ax.annotate(f"{mean:.4f}", (k, mean), textcoords="offset points",
                    xytext=(0, 14), ha="center", fontsize=8,
                    color=colours[schedule])
    ax.axhline(G.FLOOR_16, color="#777777", ls=":", lw=1.2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=9)
    ax.set_ylabel(r"$W_2$ on the held-out set")
    ax.set_title("per-seed test error (dots) and mean (bar)", fontsize=9.5)
    ax.grid(alpha=.25, axis="y")

    # ---- panel 2: learning-rate trajectories -----------------------------
    ax = axes[1]
    for name, colour in (("fixed", "#7A1F1F"), ("cosine", "#D37B2F"),
                         ("warmcos", "#168173")):
        path = OUT / f"{name}_log.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        recs = data["records"]
        steps = [r["step"] for r in recs]
        lrs = [r["lr"] for r in recs]
        ax.plot(steps, lrs, lw=1.6, color=colour, label=name)
    ax.set_xlabel("training step")
    ax.set_ylabel("learning rate")
    ax.set_yscale("log")
    ax.set_title("the schedules that were compared", fontsize=9.5)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, frameon=False)

    # ---- panel 3: why the tail-loss diagnostic failed --------------------
    ax = axes[2]
    for name, colour in (("fixed", "#7A1F1F"), ("warmcos", "#168173")):
        path = OUT / f"{name}_log.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        recs = data["records"]
        steps = [r["step"] for r in recs]
        losses = [max(r["loss"], 1e-4) for r in recs]
        ax.plot(steps, losses, lw=0.7, color=colour, alpha=.75, label=name)
    ax.set_xlabel("training step")
    ax.set_ylabel("single-minibatch loss")
    ax.set_yscale("log")
    ax.set_title("single-batch loss spans two orders of magnitude\n"
                 "(its variance cannot detect schedule stability)", fontsize=9)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, frameon=False)

    for ax in axes:
        ax.tick_params(labelsize=8)

    path = OUT / "lr_schedule.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
