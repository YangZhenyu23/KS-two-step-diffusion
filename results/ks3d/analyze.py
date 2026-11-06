"""Compare a fresh KS3D reproduction against the upstream historical metrics.

Reads (read-only):
  <run>/evaluation/metrics_per_seed.csv                      new run
  <repo>/output/ks_full_histogram_100k_bins16_20261022/...   historical (upstream)
Writes: <out>/comparison.csv, comparison.md, curves.png

Example (run from the repository root):
  python results/ks3d/analyze.py --run repro_ks3d --out repro_ks3d/analysis

The upstream snapshot is located automatically; override with --repo.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

# Locate the upstream snapshot relative to this file: <root>/results/ks3d/analyze.py
_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO = _ROOT / "upstream" / "MF-DeepParticle"
HIST_REL = Path("output/ks_full_histogram_100k_bins16_20261022/metrics_per_seed.csv")
METHODS = ("mf", "dp", "mf_dp")


def load(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("model") == "ks3d"]
    out = {}
    for r in rows:
        out[(float(r["level"]), int(r["seed"]), r["method"])] = float(r["w2"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO,
                    help=f"upstream snapshot root (default: {DEFAULT_REPO})")
    a = ap.parse_args()
    hist = a.repo / HIST_REL
    if not hist.is_file():
        raise SystemExit(
            f"historical metrics not found: {hist}\n"
            f"Pass --repo <upstream root> if the snapshot lives elsewhere."
        )
    candidates = [a.run / "evaluation/metrics_per_seed.csv",
                  a.run / "evaluation/ks/metrics_per_seed.csv"]
    source = next((p for p in candidates if p.is_file()), None)
    if source is None:
        raise SystemExit(f"no metrics_per_seed.csv under {a.run}/evaluation")
    print(f"new metrics: {source}")
    print(f"historical:  {hist}")
    new = load(source)
    old = load(hist)
    keys = sorted(set(new) & set(old))
    missing = sorted(set(new) ^ set(old))
    a.out.mkdir(parents=True, exist_ok=True)

    diffs = [abs(new[k] - old[k]) for k in keys]
    print(f"matched rows: {len(keys)}  unmatched: {len(missing)}")
    print(f"max |ΔW2|  = {max(diffs):.6g}")
    print(f"mean |ΔW2| = {statistics.mean(diffs):.6g}")
    print(f"max relative = {max(d / old[k] for d, k in zip(diffs, keys)):.3%}")

    levels = sorted({k[0] for k in keys})
    rows = []
    for level in levels:
        for method in METHODS:
            ks = [k for k in keys if k[0] == level and k[2] == method]
            if not ks:
                continue
            n = [new[k] for k in ks]
            o = [old[k] for k in ks]
            d = [abs(x - y) for x, y in zip(n, o)]
            rows.append({"level": level, "method": method, "n": len(ks),
                         "new_mean": statistics.mean(n), "new_sd": statistics.stdev(n) if len(n) > 1 else 0.,
                         "old_mean": statistics.mean(o), "old_sd": statistics.stdev(o) if len(o) > 1 else 0.,
                         "mean_abs_diff": statistics.mean(d), "max_abs_diff": max(d),
                         "ratio": statistics.mean(n) / statistics.mean(o)})

    with (a.out / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines = ["| sigma | method | new W2 (mean±sd) | historical (mean±sd) | ratio | max abs diff |",
             "|---:|---|---|---|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['level']:g} | {r['method']} | {r['new_mean']:.4f} ± {r['new_sd']:.4f} | "
                     f"{r['old_mean']:.4f} ± {r['old_sd']:.4f} | {r['ratio']:.3f} | {r['max_abs_diff']:.2e} |")
    (a.out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    per_method = {}
    for method in METHODS:
        group = [r for r in rows if r["method"] == method]
        per_method[method] = {
            "mean_new": statistics.mean(r["new_mean"] for r in group),
            "mean_old": statistics.mean(r["old_mean"] for r in group),
            "max_abs_diff": max(r["max_abs_diff"] for r in group),
            "worst_relative": max(abs(r["ratio"] - 1) for r in group),
        }
    print("\nper-method summary:")
    for m, v in per_method.items():
        print(f"  {m:6s} mean new={v['mean_new']:.4f} old={v['mean_old']:.4f} "
              f"max|Δ|={v['max_abs_diff']:.4f} worst rel={v['worst_relative']:.3%}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {"mf": "#2864B4", "dp": "#D37B2F", "mf_dp": "#168173"}
        labels = {"mf": "MF (1 step)", "dp": "DP", "mf_dp": "MF + DP"}
        fig, ax = plt.subplots(figsize=(6.4, 3.9), layout="constrained")
        for method in METHODS:
            group = [r for r in rows if r["method"] == method]
            x = [r["level"] for r in group]
            ax.plot(x, [r["new_mean"] for r in group], color=colors[method], lw=1.8,
                    marker="o", ms=3.2, label=f"{labels[method]} (this run)")
            ax.plot(x, [r["old_mean"] for r in group], color=colors[method], lw=1.2,
                    ls="--", alpha=.75, label=f"{labels[method]} (paper)")
        ax.set_xlabel(r"$\sigma$")
        ax.set_ylabel(r"$W_2$ (16-bin histogram OT)")
        ax.grid(alpha=.25)
        ax.legend(fontsize=7, ncol=2, frameon=False)
        fig.savefig(a.out / "curves.png", dpi=180)
        plt.close(fig)
        print(f"\nfigure: {a.out / 'curves.png'}")
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)
    return rows


if __name__ == "__main__":
    main()
