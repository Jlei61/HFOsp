"""Pre-outcome spatial-pattern and response contracts for FCXR-LC3 E5."""
from __future__ import annotations

import numpy as np


SCHEMA_VERSION = "fcxr-lc3-spatial-response-1.0"
RESPONSE_TIMES_MS = (50.0, 150.0, 300.0, 500.0)
RATE_WINDOW_MS = 50.0
SPATIAL_CONTROL_SEED = 731


def _take(mask, score, n):
    idx = np.flatnonzero(mask)
    if idx.size < int(n):
        raise ValueError(f"candidate mask has {idx.size} cells but {n} are required")
    order = np.lexsort((idx, np.asarray(score, float)[idx]))
    out = np.zeros(mask.size, dtype=bool)
    out[idx[order[:int(n)]]] = True
    return out


def build_equal_local_masks(pos, src, snk, axis, *, core_r: float,
                            random_seed: int = SPATIAL_CONTROL_SEED):
    """Build equal-count core/axial/transverse/shuffled local E-cell masks."""

    pos = np.asarray(pos, float)
    src = np.asarray(src, float)
    snk = np.asarray(snk, float)
    axis = np.asarray(axis, float)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("pos must have shape (NE,2)")
    axis = axis / np.linalg.norm(axis)
    perp_axis = np.array([-axis[1], axis[0]])
    midpoint = 0.5 * (src + snk)
    rel = pos - midpoint
    along = rel @ axis
    perp = rel @ perp_axis
    half_len = 0.5 * float(np.linalg.norm(snk - src)) + float(core_r)

    raw_a = np.linalg.norm(pos - src, axis=1) <= core_r
    raw_b = np.linalg.norm(pos - snk, axis=1) <= core_r
    n = int(min(raw_a.sum(), raw_b.sum()))
    if n < 1:
        raise ValueError("both registered cores must contain E cells")
    core_a = _take(raw_a, np.linalg.norm(pos - src, axis=1), n)
    core_b = _take(raw_b, np.linalg.norm(pos - snk, axis=1), n)

    raw_axial = (np.abs(perp) <= core_r) & (np.abs(along) <= half_len)
    raw_transverse = (np.abs(along) <= core_r) & (np.abs(perp) <= half_len)
    # Primary score favors the narrow dimension; the weak secondary term spreads
    # the selected cells along the intended long dimension instead of choosing a
    # compact disk at the crossing.
    axial = _take(raw_axial, np.abs(perp) + 1e-6 * np.abs(along), n)
    transverse = _take(raw_transverse, np.abs(along) + 1e-6 * np.abs(perp), n)

    rng = np.random.default_rng(int(random_seed))
    shuffled = np.zeros(pos.shape[0], dtype=bool)
    shuffled[rng.choice(pos.shape[0], size=n, replace=False)] = True

    used = core_a | core_b | axial | transverse
    radial = np.linalg.norm(rel, axis=1)
    surround = _take(~used, radial, n)
    out = dict(
        core_A=core_a, core_B=core_b, axial=axial, transverse=transverse,
        shuffled_axial=shuffled, surround=surround,
    )
    if len({int(v.sum()) for v in out.values()}) != 1:
        raise RuntimeError("local spatial masks are not equal-count")
    return out


def positive_patterns(local_masks: dict) -> dict:
    """Return unit-per-active-cell local patterns (global controls are later)."""

    required = ("core_A", "core_B", "axial", "transverse", "shuffled_axial")
    missing = [name for name in required if name not in local_masks]
    if missing:
        raise ValueError(f"missing local patterns: {missing}")
    counts = {int(np.count_nonzero(local_masks[name])) for name in required}
    if len(counts) != 1:
        raise ValueError("positive local patterns must have equal active-cell counts")
    return {name: np.asarray(local_masks[name], float) for name in required}


def global_control_patterns(ne: int, n_local: int) -> dict:
    """Separate global patterns matching local positive charge or RMS at A=1."""

    if not (0 < int(n_local) <= int(ne)):
        raise ValueError("n_local must lie in (0,NE]")
    charge_amp = float(n_local) / float(ne)
    rms_amp = np.sqrt(float(n_local) / float(ne))
    return dict(
        global_charge_matched=np.full(int(ne), charge_amp),
        global_rms_matched=np.full(int(ne), rms_amp),
    )


def _unit_l2(value):
    value = np.asarray(value, float)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("basis vector has zero/non-finite L2 norm")
    return value / norm


def build_signed_basis(local_masks: dict, *, random_seed: int = SPATIAL_CONTROL_SEED):
    """Build the locked nine-dimensional physical input/output basis."""

    ne = len(next(iter(local_masks.values())))
    core_a = np.asarray(local_masks["core_A"], float)
    core_b = np.asarray(local_masks["core_B"], float)
    axial = np.asarray(local_masks["axial"], float)
    transverse = np.asarray(local_masks["transverse"], float)
    surround = np.asarray(local_masks["surround"], float)
    rng = np.random.default_rng(int(random_seed))
    random_1 = rng.choice(np.array([-1.0, 1.0]), size=ne)
    random_2 = rng.choice(np.array([-1.0, 1.0]), size=ne)
    basis = dict(
        global_mode=np.ones(ne), core_A=core_a, core_B=core_b,
        axial_symmetric=axial,
        axial_antisymmetric=core_a - core_b,
        transverse=transverse, surround=surround,
        random_731=random_1, random_732=random_2,
    )
    return {name: _unit_l2(value) for name, value in basis.items()}


def rate_fields(raster, *, dt_ms: float, response_times_ms=RESPONSE_TIMES_MS,
                window_ms: float = RATE_WINDOW_MS):
    """Per-cell trailing-window rates at registered finite response times."""

    raster = np.asarray(raster, bool)
    if raster.ndim != 2 or not (dt_ms > 0 and window_ms > 0):
        raise ValueError("raster must be step x cell and dt/window must be positive")
    out = {}
    for t_ms in response_times_ms:
        hi = min(raster.shape[0], int(round(float(t_ms) / dt_ms)))
        lo = max(0, hi - int(round(window_ms / dt_ms)))
        if hi <= lo:
            raise ValueError(f"empty response window at {t_ms} ms")
        out[float(t_ms)] = raster[lo:hi].sum(axis=0) / ((hi - lo) * dt_ms * 1e-3)
    return out


def first_passage_ms(raster, *, dt_ms: float):
    raster = np.asarray(raster, bool)
    out = np.full(raster.shape[1], np.nan, dtype=float)
    active = raster.any(axis=0)
    out[active] = np.argmax(raster[:, active], axis=0) * float(dt_ms)
    return out


def projected_response_matrix(plus_fields: dict, minus_fields: dict,
                              basis: dict, *, epsilon_l2: float):
    """Central-difference response matrices in the locked physical basis."""

    if not np.isfinite(epsilon_l2) or epsilon_l2 <= 0:
        raise ValueError("epsilon_l2 must be finite and positive")
    names = list(basis)
    times = sorted(next(iter(plus_fields.values())))
    matrices = {}
    for t_ms in times:
        mat = np.zeros((len(names), len(names)), dtype=float)
        for j, in_name in enumerate(names):
            delta = (np.asarray(plus_fields[in_name][t_ms], float)
                     - np.asarray(minus_fields[in_name][t_ms], float)) / (2.0 * epsilon_l2)
            for i, out_name in enumerate(names):
                mat[i, j] = float(np.dot(basis[out_name], delta))
        matrices[float(t_ms)] = mat
    return names, matrices


def svd_summary(matrix, basis_names):
    matrix = np.asarray(matrix, float)
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    if not np.all(np.isfinite(s)):
        raise ValueError("response matrix SVD is non-finite")
    v0 = vt[0]
    u0 = u[:, 0]
    idx = {name: i for i, name in enumerate(basis_names)}
    overlaps = {}
    for name in ("global_mode", "core_A", "core_B", "axial_symmetric",
                 "axial_antisymmetric", "transverse"):
        if name in idx:
            overlaps[name] = dict(input_abs=float(abs(v0[idx[name]])),
                                  output_abs=float(abs(u0[idx[name]])))
    return dict(
        singular_values=s.astype(float).tolist(), sigma_max=float(s[0]),
        dominant_input=v0.astype(float).tolist(),
        dominant_output=u0.astype(float).tolist(), overlaps=overlaps,
    )
