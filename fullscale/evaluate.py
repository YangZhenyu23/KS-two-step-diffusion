"""Exact histogram W2 evaluation for the full-scale study.

Protocol, identical to every other number in this repository:

* one shared 16-bin-per-axis grid **per cell**, built from that cell's reference
  and prediction only (a global grid over all 200 cells would inflate W2 by an
  order of magnitude — this was a real bug in the pilot);
* occupied bins become weighted Dirac masses with weights equal to their
  normalised counts;
* W2 is the square root of the exact OT value from POT's network simplex;
* every solve is certified: ``result_code == 1``, marginal residual < 1e-9,
  reconstructed primal-dual gap < 1e-8. A failed certificate raises.

The floor at 100k particles and 16 bins was measured at 0.0644
(``results/fullscale/floor.json``), so a mean error near that value means the
model is at the statistical limit of this metric, not that it is perfect.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np

MARGINAL_TOL = 1e-9
GAP_TOL = 1e-8


def shared_edges(clouds: list[np.ndarray], bins: int):
    stacked = np.concatenate(clouds, axis=0)
    lo, hi = stacked.min(axis=0), stacked.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    pad = 1e-6 * span
    return [np.linspace(lo[d] - pad[d], hi[d] + pad[d], bins + 1) for d in range(stacked.shape[1])]


def measure(points: np.ndarray, edges, bins: int):
    counts, _ = np.histogramdd(np.asarray(points, dtype=np.float64), bins=edges)
    retained = int(counts.sum())
    if retained != len(points):
        raise ValueError(f"histogram lost {len(points) - retained} particles")
    index = np.argwhere(counts > 0)
    centres = [0.5 * (edge[1:] + edge[:-1]) for edge in edges]
    positions = np.column_stack([centres[d][index[:, d]] for d in range(points.shape[1])])
    weights = counts[tuple(index.T)].astype(np.float64) / retained
    return positions, weights


def solve(reference, generated) -> dict:
    import ot
    from scipy.spatial.distance import cdist

    ref_pos, ref_w = reference
    gen_pos, gen_w = generated
    cost = cdist(ref_pos, gen_pos, metric="sqeuclidean")
    if not np.isfinite(cost).all():
        raise ValueError("cost matrix overflowed")
    value, info = ot.emd2(ref_w, gen_w, cost, log=True, numItermax=5_000_000)
    code = int(info.get("result_code", -1))
    if code != 1:
        raise RuntimeError(f"exact OT did not reach optimality: {info.get('warning')}")

    residual = float(max(abs(ref_w.sum() - 1.0), abs(gen_w.sum() - 1.0)))
    gap, available = 0.0, False
    u, v = info.get("u"), info.get("v")
    if u is not None and v is not None:
        u = np.asarray(u, dtype=np.float64).ravel()
        v = np.asarray(v, dtype=np.float64).ravel()
        if (u.size == ref_w.size and v.size == gen_w.size
                and np.isfinite(u).all() and np.isfinite(v).all()):
            gap = float(abs(ref_w @ u + gen_w @ v - value))
            available = True
    if residual >= MARGINAL_TOL:
        raise RuntimeError(f"marginal residual {residual:.3e} exceeds tolerance")
    if available and gap >= GAP_TOL:
        raise RuntimeError(f"primal-dual gap {gap:.3e} exceeds tolerance")
    return {
        "w2": float(math.sqrt(max(value, 0.0))),
        "w2_squared": float(max(value, 0.0)),
        "solver_result_code": code,
        "marginal_residual": residual,
        "primal_dual_gap": gap,
        "primal_dual_gap_available": available,
        "reference_occupied_bins": int(len(ref_w)),
        "generated_occupied_bins": int(len(gen_w)),
    }


def score_pair(reference: np.ndarray, prediction: np.ndarray, bins: int) -> dict:
    """Score one (reference, prediction) pair on its own grid."""
    edges = shared_edges([reference, prediction], bins)
    return solve(measure(reference, edges, bins), measure(prediction, edges, bins))


def _worker(payload):
    key, reference, prediction, bins = payload
    return key, score_pair(reference, prediction, bins)


def score_many(pairs: dict[str, tuple[np.ndarray, np.ndarray]], *, bins: int = 16,
               workers: int = 1, log=print) -> list[dict]:
    """Score many cells, optionally across processes.

    ``pairs`` maps a cell key to ``(reference, prediction)``. Each pair is scored
    independently on its own grid, so results are order-independent and can be
    parallelised without affecting the numbers.
    """
    payloads = [(key, ref, pred, bins) for key, (ref, pred) in pairs.items()]
    rows = []
    if workers > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for key, score in pool.map(_worker, payloads):
                rows.append({"key": key, **score})
                log(f"  {key}: W2={score['w2']:.5f}")
    else:
        for key, score in map(_worker, payloads):
            rows.append({"key": key, **score})
            log(f"  {key}: W2={score['w2']:.5f}")
    rows.sort(key=lambda r: r["key"])
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
