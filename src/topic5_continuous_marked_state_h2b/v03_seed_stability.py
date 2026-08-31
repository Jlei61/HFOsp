"""Cross-seed stability in decoder geometry for H2b v0.3 qualification."""
from __future__ import annotations

import hashlib
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _standardise(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    centre = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    active = np.isfinite(scale) & (scale > 1e-8)
    if not bool(active.any()):
        return np.empty((len(matrix), 0), dtype=np.float64)
    return (matrix[:, active] - centre[active]) / scale[active]


def _condensed_distance(values: np.ndarray) -> np.ndarray:
    matrix = _standardise(values)
    if matrix.shape[1] == 0 or len(matrix) < 2:
        return np.zeros(len(matrix) * (len(matrix) - 1) // 2, dtype=np.float64)
    rows, columns = np.triu_indices(len(matrix), 1)
    return np.linalg.norm(matrix[rows] - matrix[columns], axis=1)


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.array(left, dtype=np.float64, copy=True)
    y = np.array(right, dtype=np.float64, copy=True)
    if x.shape != y.shape or x.size < 2:
        return None
    x = x - np.mean(x)
    y = y - np.mean(y)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > 1e-20 else None


def linear_cka(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.array(left, dtype=np.float64, copy=True)
    y = np.array(right, dtype=np.float64, copy=True)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        return None
    x -= np.mean(x, axis=0, keepdims=True)
    y -= np.mean(y, axis=0, keepdims=True)
    if not bool(np.isfinite(x).all() and np.isfinite(y).all()):
        return None
    cross = float(np.sum((x.T @ y) ** 2))
    left_norm = float(np.sum((x.T @ x) ** 2))
    right_norm = float(np.sum((y.T @ y) ** 2))
    denominator = np.sqrt(left_norm * right_norm)
    return cross / denominator if denominator > 1e-20 else None


def procrustes_similarity(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or len(x) < 2:
        return None
    x = x - np.mean(x, axis=0, keepdims=True)
    y = y - np.mean(y, axis=0, keepdims=True)
    x_norm = float(np.linalg.norm(x))
    y_norm = float(np.linalg.norm(y))
    if x_norm <= 1e-20 or y_norm <= 1e-20:
        return None
    x /= x_norm
    y /= y_norm
    u, _, vt = np.linalg.svd(y.T @ x, full_matrices=False)
    aligned = y @ (u @ vt)
    return _correlation(x.ravel(), aligned.ravel())


def _common_rows(traces: Sequence[dict[str, np.ndarray]],
                 max_anchors: int) -> tuple[np.ndarray, list[np.ndarray]]:
    keyed = []
    for trace in traces:
        time = np.asarray(trace["anchor_time"], dtype=np.float64)
        session = np.asarray(trace["anchor_session"], dtype=np.int64)
        keys = [(int(label), float(value)) for label, value in zip(session, time)]
        _require(len(keys) == len(set(keys)), "trace has duplicate session/time anchors")
        keyed.append({key: index for index, key in enumerate(keys)})
    common = sorted(set.intersection(*(set(value) for value in keyed)))
    _require(len(common) >= 3, "fewer than three common cross-seed anchors")
    if len(common) > int(max_anchors):
        take = np.linspace(0, len(common) - 1, int(max_anchors)).round().astype(int)
        common = [common[index] for index in np.unique(take)]
    session = np.asarray([key[0] for key in common], dtype=np.int64)
    rows = [np.asarray([mapping[key] for key in common], dtype=np.int64)
            for mapping in keyed]
    return session, rows


def _within_session_permutation(session: np.ndarray,
                                rng: np.random.Generator) -> np.ndarray:
    output = np.arange(len(session), dtype=np.int64)
    for label in np.unique(session):
        rows = np.flatnonzero(session == label)
        output[rows] = rng.permutation(rows)
    return output


def cross_seed_stability(
    subject: str,
    seeds: Sequence[int],
    traces: Sequence[dict[str, np.ndarray]],
    *,
    max_anchors: int = 256,
    n_permutations: int = 100,
) -> dict[str, Any]:
    _require(len(seeds) == len(traces), "seed/trace count mismatch")
    _require(len(set(map(int, seeds))) == len(seeds), "duplicate seed")
    session, rows = _common_rows(traces, max_anchors=max_anchors)
    decoder = [np.asarray(trace["persistent_decoder"], dtype=np.float64)[row]
               for trace, row in zip(traces, rows)]
    state = [np.asarray(trace["persistent_state"], dtype=np.float64)[row]
             for trace, row in zip(traces, rows)]
    distance = [_condensed_distance(value) for value in decoder]
    seed_material = int.from_bytes(
        hashlib.sha256(subject.encode("utf-8")).digest()[:8], "little"
    )
    rng = np.random.default_rng(seed_material)
    pair_rows = []
    for left, right in combinations(range(len(seeds)), 2):
        observed = _correlation(distance[left], distance[right])
        null = []
        for _ in range(int(n_permutations)):
            permutation = _within_session_permutation(session, rng)
            permuted = _condensed_distance(decoder[right][permutation])
            value = _correlation(distance[left], permuted)
            if value is not None and np.isfinite(value):
                null.append(float(value))
        q95 = float(np.quantile(null, 0.95)) if null else None
        pair_rows.append({
            "seed_left": int(seeds[left]),
            "seed_right": int(seeds[right]),
            "decoder_distance_correlation": observed,
            "decoder_linear_cka": linear_cka(decoder[left], decoder[right]),
            "latent_procrustes_similarity": procrustes_similarity(
                state[left], state[right]
            ),
            "permuted_distance_correlation_q95": q95,
            "n_finite_permutations": len(null),
            "distance_stability_above_null": bool(
                observed is not None and q95 is not None and observed > q95
            ),
        })
    observed_values = [row["decoder_distance_correlation"] for row in pair_rows
                       if row["decoder_distance_correlation"] is not None]
    cka_values = [row["decoder_linear_cka"] for row in pair_rows
                  if row["decoder_linear_cka"] is not None]
    procrustes_values = [row["latent_procrustes_similarity"] for row in pair_rows
                         if row["latent_procrustes_similarity"] is not None]
    above = sum(bool(row["distance_stability_above_null"]) for row in pair_rows)
    return {
        "status": "COMPLETE",
        "subject": str(subject),
        "seeds": list(map(int, seeds)),
        "n_common_anchors": int(len(session)),
        "n_sessions": int(len(np.unique(session))),
        "n_seed_pairs": len(pair_rows),
        "n_permutations_per_pair": int(n_permutations),
        "median_decoder_distance_correlation": (
            float(np.median(observed_values)) if observed_values else None
        ),
        "median_decoder_linear_cka": float(np.median(cka_values)) if cka_values else None,
        "median_latent_procrustes_similarity": (
            float(np.median(procrustes_values)) if procrustes_values else None
        ),
        "pairs_above_seed_permuted_null": int(above),
        "fraction_pairs_above_seed_permuted_null": (
            float(above / len(pair_rows)) if pair_rows else 0.0
        ),
        "preliminary_Q5_pass": bool(len(seeds) >= 3 and pair_rows and above > len(pair_rows) / 2),
        "pair_rows": pair_rows,
    }


def load_trace(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def load_cell_manifest(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
