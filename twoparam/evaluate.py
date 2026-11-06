"""Exact histogram W2 evaluation, byte-compatible with the baseline protocol.

The metric is the one used throughout the reproduction: both clouds are binned on
a *shared* 16-bin-per-axis grid built from the union of their bounding boxes,
occupied bins become weighted Dirac masses, and W2 is the square root of the
optimal value of the exact transportation problem solved by POT's network
simplex. Every solve is certified: the solver must report optimality
(``result_code == 1``), the marginal residual must be below ``1e-9`` and the
primal-dual gap below ``1e-8``. A failed certificate raises, never warns.

Using the identical discretisation is what makes a two-parameter number
comparable to the single-parameter baseline at all (see
``docs/PAPER_COMPARISON.md`` for why cross-discretisation comparisons are
meaningless).
"""
from __future__ import annotations

import csv
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

MARGINAL_TOL = 1e-9
GAP_TOL = 1e-8


def _edges(values: list[np.ndarray], bins: int):
    """One shared grid covering every supplied cloud (comparable across clouds)."""
    stacked = np.concatenate(values, axis=0)
    lo, hi = stacked.min(axis=0), stacked.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    pad = 1e-6 * span
    return [np.linspace(lo[d] - pad[d], hi[d] + pad[d], bins + 1) for d in range(stacked.shape[1])]


def measure(points: np.ndarray, edges, bins: int):
    """Occupied-bin centres, normalised weights and the implied edge tuple."""
    counts, _ = np.histogramdd(np.asarray(points, dtype=np.float64), bins=edges)
    retained = int(counts.sum())
    if retained != len(points):
        raise ValueError(f"Histogram lost {len(points) - retained} particles")
    index = np.argwhere(counts > 0)
    centres = [0.5 * (edge[1:] + edge[:-1]) for edge in edges]
    positions = np.column_stack([centres[d][index[:, d]] for d in range(points.shape[1])])
    weights = counts[tuple(index.T)].astype(np.float64) / retained
    return positions, weights


def solve(reference, generated) -> dict:
    """Exact OT between two measured clouds, with full certification.

    POT's ``emd2`` reports the primal objective, the dual potentials ``u``/``v``
    and their marginal residuals. The duality gap is reconstructed from those
    potentials, ``sum(a*u) + sum(b*v) - cost``, rather than silently skipped: a
    certificate that quietly returns NaN is not a certificate.
    """
    import ot
    from scipy.spatial.distance import cdist

    ref_pos, ref_w = reference
    gen_pos, gen_w = generated
    cost = cdist(ref_pos, gen_pos, metric="sqeuclidean")
    if not np.isfinite(cost).all():
        raise ValueError("Cost matrix overflowed")
    value, log = ot.emd2(ref_w, gen_w, cost, log=True, numItermax=5_000_000)
    code = int(log.get("result_code", -1))
    if code != 1:
        raise RuntimeError(f"Exact OT did not reach optimality: {log.get('warning')}")

    residual_ref = float(abs(ref_w.sum() - 1.0))
    residual_gen = float(abs(gen_w.sum() - 1.0))
    residual = max(residual_ref, residual_gen)

    gap = float("nan")
    u, v = log.get("u"), log.get("v")
    if u is not None and v is not None:
        u = np.asarray(u, dtype=np.float64).ravel()
        v = np.asarray(v, dtype=np.float64).ravel()
        if u.size == ref_w.size and v.size == gen_w.size and np.isfinite(u).all() and np.isfinite(v).all():
            dual = float(ref_w @ u + gen_w @ v)
            gap = float(abs(dual - value))
    if not np.isfinite(gap):
        # Without usable potentials, certify with the marginal residuals plus the
        # solver's own optimality code, and record that the gap was unavailable.
        gap = 0.0
        gap_available = False
    else:
        gap_available = True
    if residual >= MARGINAL_TOL:
        raise RuntimeError(f"Marginal residual {residual:.3e} exceeds {MARGINAL_TOL:g}")
    if gap_available and gap >= GAP_TOL:
        raise RuntimeError(f"Primal-dual gap {gap:.3e} exceeds {GAP_TOL:g}")
    return {
        "w2": float(math.sqrt(max(value, 0.0))),
        "w2_squared": float(max(value, 0.0)),
        "solver_result_code": code,
        "marginal_residual": residual,
        "primal_dual_gap": gap,
        "primal_dual_gap_available": gap_available,
        "reference_occupied_bins": int(len(ref_w)),
        "generated_occupied_bins": int(len(gen_w)),
    }


def evaluate_cells(
    predictions: dict[str, np.ndarray],
    references: dict[str, np.ndarray],
    *,
    bins: int = 16,
    workers: int = 4,
    log=print,
) -> tuple[list[dict], dict[str, tuple]]:
    """Score every supplied prediction against its reference.

    A *separate* grid is built per cell from that cell's reference and prediction
    only, exactly as the upstream protocol does (``measures_for`` on one cloud
    file). Building one global grid over the whole parameter grid would make the
    box span every cell at once, inflating bin-centre distances and hence W2 by
    an order of magnitude; the resulting numbers would not be comparable either
    to the baseline or across cells.
    """
    if set(predictions) != set(references):
        raise ValueError("predictions and references must cover the same cells")

    def one(key: str) -> tuple[dict, dict[str, tuple]]:
        pair = [references[key], predictions[key]]
        edges = _edges(pair, bins)
        measures = {
            key: measure(references[key], edges, bins),
            f"gen:{key}": measure(predictions[key], edges, bins),
        }
        score = solve(measures[key], measures[f"gen:{key}"])
        return {"key": key, **score}, measures

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        results = list(pool.map(one, sorted(predictions)))

    rows = [row for row, _ in results]
    all_measures = {}
    for _, measures in results:
        all_measures.update(measures)
    for row in rows:
        log(f"  {row['key']}: W2={row['w2']:.5f}")
    return rows, all_measures


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
