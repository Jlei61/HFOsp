"""Direction-axis disambiguation helpers for Topic 4 SEF-ITP H2b.

Plan: docs/archive/topic4/sef_itp_direction_axis/phase_h2b_direction_axis_plan_2026-05-25.md

H2b is a per-subject mechanism disambiguation supplementary to H2 spatial-layer
cohort claim. It answers: when source/sink role-swap exists, is the swap reading
the same spatial axis backwards (SEF-ITP physical picture), or are template A and
template B two independent sources?

NOT a cohort claim. NOT a SOZ-progression test. Cohort output = per-subject 3-state
verdict counts, stratified by swap_class. No Wilcoxon, no cohort p-value.

5 layers:
  Layer 1 — Template-level axis vector v_c = c_sink - c_source; primary cos(v_A, -v_B)
            + role-shuffle permutation null.
  Layer 2 — Event-level direction; per-cluster cos(u_event, u_A) histogram.
  Layer 3 — Axis projection slope; slope_B_on_axisA is the discriminative test.
  Layer 4 — PCA degeneracy detector (near-1D shaft layout); OVERRIDING flag.
  Layer 5 — SOZ / ictal-early relation (secondary; not in verdict gate).

Verdict alphabet (per archive plan §3.7):
  degenerate_geometry  — Layer 4 fires OR n_universe<6 (overrides Layer 1-3)
  axis_reversal        — Layer 1 cos(A,-B)>=0.5 AND perm p<0.05 AND Layer 3 slope_B_on_axisA<0 with p<0.05
  dual_source          — |cos(A,-B)|<0.5 (angle in 60..120 deg) AND Layer 3 r2_B_on_axisA<0.2
  same_direction       — cos(A,+B)>=0.5 (templates share direction)
  inconclusive         — everything else
  exit_no_universe     — n_universe < 6

Constants in this module are framework-time locks. Do not adjust post-hoc.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__version__ = "v1.0.0"

# ============================================================================
# Framework-time locked constants (archive plan §3 lock 2026-05-25)
# ============================================================================
DEGENERACY_LAMBDA_RATIO_12_THRESHOLD: float = 0.10
DEGENERACY_LAMBDA_RATIO_23_THRESHOLD: float = 0.05
COS_AXIS_REVERSAL_THRESHOLD: float = 0.5     # cos(60 deg)
COS_SAME_DIRECTION_THRESHOLD: float = 0.5    # cos(60 deg)
DUAL_SOURCE_COS_ABS_MAX: float = 0.5         # |cos|<0.5 → angle in (60, 120) deg
R2_DUAL_SOURCE_MAX: float = 0.20
# v1.0.2 (2026-05-25, audit catch #3): docstring promised r² gate for axis_reversal_shaped /
# same_direction_shaped but implementation only checked cos+slope-sign. Low r² with negative
# slope (noise-driven) could silently enter axis_reversal_shaped. Lock 0.20 = same threshold
# as R2_DUAL_SOURCE_MAX; the descriptive layer requires r² >= 0.20 to assert a shape.
R2_DESCRIPTIVE_MIN: float = 0.20
SLOPE_PVALUE_THRESHOLD: float = 0.05
AXIS_PERM_PVALUE_THRESHOLD: float = 0.05
DEFAULT_N_PERM: int = 1000
DEFAULT_SEED: int = 0
DEFAULT_K_EVENT: int = 2
MIN_UNIVERSE_SIZE: int = 6
MIN_EVENT_ELIGIBLE_FACTOR: int = 2  # n_eligible >= 2 * k_event


# ============================================================================
# Data containers
# ============================================================================


@dataclass
class TemplateAxis:
    """Per-cluster template axis. Indices reference ORIGINAL channel order."""

    cluster_id: int
    source_indices: List[int]
    sink_indices: List[int]
    source_centroid: np.ndarray
    sink_centroid: np.ndarray
    v_axis: np.ndarray
    axis_length: float


@dataclass
class DegeneracyResult:
    lambda_eigenvalues: Tuple[float, float, float]
    lambda_ratio_12: float
    lambda_ratio_23: float
    is_near_1d: bool
    is_near_2d: bool


@dataclass
class AxisPairAlignment:
    cos_A_pos_B: float
    cos_A_neg_B: float
    angle_A_pos_B_deg: float
    angle_A_neg_B_deg: float
    null_cos_A_neg_B_median: float
    null_cos_A_neg_B_95: float
    p_one_sided_axis_reversal: float
    n_perm_completed: int


@dataclass
class AxisProjectionSlope:
    cluster_id: int
    slope: float
    intercept: float
    r2: float
    spearman_rho: float
    n_points: int
    null_slope_median: float
    p_one_sided_neg_slope: float


@dataclass
class EventDirectionStats:
    cluster_id: int
    k_event: int
    n_events_total: int
    n_events_eligible: int
    cos_with_u_A_median: float
    cos_with_u_A_q25: float
    cos_with_u_A_q75: float
    mean_unit_vector: List[float]
    mean_resultant_length: float


@dataclass
class SOZSet:
    """One SOZ-set readout (mapped-full or joint-universe subset)."""

    name: str
    n: int
    centroid: Optional[List[float]]
    d_source_A_to_SOZ: Optional[float]
    d_sink_A_to_SOZ: Optional[float]
    d_source_B_to_SOZ: Optional[float]
    d_sink_B_to_SOZ: Optional[float]
    min_d_source_A_chs_to_SOZ_chs: Optional[float]
    min_d_sink_A_chs_to_SOZ_chs: Optional[float]
    min_d_source_B_chs_to_SOZ_chs: Optional[float]
    min_d_sink_B_chs_to_SOZ_chs: Optional[float]


@dataclass
class SOZRelation:
    """SOZ relation reported against TWO sets (v1.0.2 audit fix #5):

    - mapped_full: all soz_mask channels that have coords (broader; clinical-SOZ-grounded)
    - joint_universe: soz_mask AND universe (joint_valid AND mapped); narrower; matches Layer 0

    Both sets are reported. joint-universe alone risked silently dropping SOZ channels that
    weren't in this pair's joint_valid set, making 'source close to SOZ' look misleadingly
    coupled to participation-field selection.
    """

    mapped_full: Optional[SOZSet]
    joint_universe: Optional[SOZSet]
    exit_reason: Optional[str]
    # Backward-compat top-level fields (mirror joint_universe for any existing readers):
    soz_centroid: Optional[List[float]]
    n_soz_in_universe: int
    d_source_A_to_SOZ: Optional[float]
    d_sink_A_to_SOZ: Optional[float]
    d_source_B_to_SOZ: Optional[float]
    d_sink_B_to_SOZ: Optional[float]
    min_d_source_A_chs_to_SOZ_chs: Optional[float]
    min_d_sink_A_chs_to_SOZ_chs: Optional[float]
    min_d_source_B_chs_to_SOZ_chs: Optional[float]
    min_d_sink_B_chs_to_SOZ_chs: Optional[float]


@dataclass
class SubjectVerdict:
    label: str
    reason: str


@dataclass
class DescriptiveGeometry:
    """Geometric-pattern descriptive label, *independent* of permutation-null gates.

    Reports what the geometry looks like (axis-reversal-shaped / dual-source-shaped /
    same-direction-shaped / unclear). Useful when the strict verdict reads
    'inconclusive' because the permutation null saturates (e.g., when decision_k is
    large relative to n_universe). NEVER overrides the strict verdict; only adds
    descriptive context.
    """

    label: str  # "axis_reversal_shaped" | "dual_source_shaped" | "same_direction_shaped" | "unclear"
    reason: str


# ============================================================================
# Layer 0 — universe + endpoint extraction
# ============================================================================


def compute_universe_mask(
    joint_valid: np.ndarray, mapped_mask: np.ndarray
) -> np.ndarray:
    """Universe = joint_valid AND mapped_mask (bool, length n_channels)."""
    joint_valid = np.asarray(joint_valid, dtype=bool)
    mapped_mask = np.asarray(mapped_mask, dtype=bool)
    if joint_valid.shape != mapped_mask.shape:
        raise ValueError(
            f"shape mismatch: joint_valid {joint_valid.shape} vs mapped_mask {mapped_mask.shape}"
        )
    return joint_valid & mapped_mask


def derive_source_sink_within_universe(
    rank_dense: np.ndarray,
    universe_mask: np.ndarray,
    decision_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split one cluster's dense rank into source-half (lowest k) and sink-half (highest k).

    Uses the same argsort rule as ``src.rank_displacement.derive_swap_endpoint``:
    lowest-k dense rank = source, highest-k dense rank = sink. Confined to universe.

    Returns:
        (source_indices, sink_indices) into ORIGINAL channel order (not universe-internal).

    Raises:
        ValueError if universe_mask.sum() < 2*decision_k (caller must gate).
    """
    rank_dense = np.asarray(rank_dense, dtype=float)
    universe_mask = np.asarray(universe_mask, dtype=bool)
    if rank_dense.shape != universe_mask.shape:
        raise ValueError(
            f"shape mismatch: rank_dense {rank_dense.shape} vs universe_mask {universe_mask.shape}"
        )
    universe_idx = np.where(universe_mask)[0]
    if universe_idx.size < 2 * decision_k:
        raise ValueError(
            f"universe size {universe_idx.size} < 2*decision_k={2 * decision_k}; "
            f"caller must check before calling"
        )
    sub_ranks = rank_dense[universe_idx]
    order = np.argsort(sub_ranks, kind="stable")
    source_local = order[:decision_k]
    sink_local = order[-decision_k:]
    source_global = np.sort(universe_idx[source_local])
    sink_global = np.sort(universe_idx[sink_local])
    return source_global, sink_global


def compute_template_axis(
    coords: np.ndarray,
    source_indices: np.ndarray,
    sink_indices: np.ndarray,
    cluster_id: int,
) -> TemplateAxis:
    """Build the template axis vector c_sink - c_source from coords (mm)."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be (n, 3); got {coords.shape}")
    src = np.asarray(source_indices, dtype=int)
    snk = np.asarray(sink_indices, dtype=int)
    if src.size == 0 or snk.size == 0:
        raise ValueError("source / sink indices must be non-empty")
    c_source = coords[src].mean(axis=0)
    c_sink = coords[snk].mean(axis=0)
    v_axis = c_sink - c_source
    return TemplateAxis(
        cluster_id=int(cluster_id),
        source_indices=[int(x) for x in src],
        sink_indices=[int(x) for x in snk],
        source_centroid=c_source.astype(float),
        sink_centroid=c_sink.astype(float),
        v_axis=v_axis.astype(float),
        axis_length=float(np.linalg.norm(v_axis)),
    )


# ============================================================================
# Layer 4 — degeneracy detector (PCA on universe coords)
# ============================================================================


def detect_degeneracy(
    coords: np.ndarray,
    universe_mask: np.ndarray,
    near_1d_threshold: float = DEGENERACY_LAMBDA_RATIO_12_THRESHOLD,
    near_2d_threshold: float = DEGENERACY_LAMBDA_RATIO_23_THRESHOLD,
) -> DegeneracyResult:
    """PCA on universe coords; flag near-1D (overriding) and near-2D (descriptive)."""
    coords = np.asarray(coords, dtype=float)
    universe_mask = np.asarray(universe_mask, dtype=bool)
    pts = coords[universe_mask]
    if pts.shape[0] < 3:
        return DegeneracyResult(
            lambda_eigenvalues=(float("nan"), float("nan"), float("nan")),
            lambda_ratio_12=float("nan"),
            lambda_ratio_23=float("nan"),
            is_near_1d=True,
            is_near_2d=False,
        )
    centered = pts - pts.mean(axis=0)
    cov = centered.T @ centered / max(1, pts.shape[0] - 1)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    l1, l2, l3 = float(eigvals[0]), float(eigvals[1]), float(eigvals[2])
    if l1 <= 1e-12:
        return DegeneracyResult(
            lambda_eigenvalues=(l1, l2, l3),
            lambda_ratio_12=float("nan"),
            lambda_ratio_23=float("nan"),
            is_near_1d=True,
            is_near_2d=False,
        )
    ratio_12 = l2 / l1
    ratio_23 = l3 / l1
    is_near_1d = ratio_12 < near_1d_threshold
    is_near_2d = (not is_near_1d) and (ratio_23 < near_2d_threshold)
    return DegeneracyResult(
        lambda_eigenvalues=(l1, l2, l3),
        lambda_ratio_12=float(ratio_12),
        lambda_ratio_23=float(ratio_23),
        is_near_1d=bool(is_near_1d),
        is_near_2d=bool(is_near_2d),
    )


# ============================================================================
# Layer 1 — axis pair alignment + permutation null
# ============================================================================


def _safe_unit(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return None
    return v / n


def compute_axis_pair_alignment(
    coords: np.ndarray,
    universe_mask: np.ndarray,
    template_A: TemplateAxis,
    template_B: TemplateAxis,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = DEFAULT_SEED,
) -> AxisPairAlignment:
    """Compute cos(v_A, ±v_B) + permutation null on role-label shuffle within U.

    Null hypothesis: source/sink role labels carry no geometric information about
    axis pairing. Permutation in U = source_A ∪ sink_A ∪ source_B ∪ sink_B
    redistributes role tags (S_A, K_A, S_B, K_B each of size decision_k); recomputes
    centroids and v vectors; gets a null distribution of cos(v_A_null, -v_B_null).

    A small p_one_sided_axis_reversal means observed cos(v_A, -v_B) is more positive
    than role-shuffled, i.e. the v_A / v_B antipodal alignment is unlikely under
    random role assignment.
    """
    coords = np.asarray(coords, dtype=float)
    universe_mask = np.asarray(universe_mask, dtype=bool)
    u_A = _safe_unit(template_A.v_axis)
    u_B = _safe_unit(template_B.v_axis)
    if u_A is None or u_B is None:
        return AxisPairAlignment(
            cos_A_pos_B=float("nan"),
            cos_A_neg_B=float("nan"),
            angle_A_pos_B_deg=float("nan"),
            angle_A_neg_B_deg=float("nan"),
            null_cos_A_neg_B_median=float("nan"),
            null_cos_A_neg_B_95=float("nan"),
            p_one_sided_axis_reversal=float("nan"),
            n_perm_completed=0,
        )
    cos_pos = float(np.clip(np.dot(u_A, u_B), -1.0, 1.0))
    cos_neg = float(np.clip(np.dot(u_A, -u_B), -1.0, 1.0))
    angle_pos = float(np.degrees(np.arccos(cos_pos)))
    angle_neg = float(np.degrees(np.arccos(cos_neg)))

    union = (
        set(map(int, template_A.source_indices))
        | set(map(int, template_A.sink_indices))
        | set(map(int, template_B.source_indices))
        | set(map(int, template_B.sink_indices))
    )
    U = np.array(sorted(union), dtype=int)
    n_U = U.size
    n_sA = len(template_A.source_indices)
    n_kA = len(template_A.sink_indices)
    n_sB = len(template_B.source_indices)
    n_kB = len(template_B.sink_indices)

    # Per-cluster INDEPENDENT role draws: each cluster's source/sink must be
    # disjoint internally (a channel can't be both source and sink of the SAME
    # template), but the two clusters draw independently — under the antipodal
    # hypothesis the same channel can legitimately be source_A and sink_B.
    # Requires only |U| >= max(n_sA + n_kA, n_sB + n_kB).
    needed_per_cluster_A = n_sA + n_kA
    needed_per_cluster_B = n_sB + n_kB
    null_cos_neg: List[float] = []
    if n_U >= max(needed_per_cluster_A, needed_per_cluster_B, 2):
        rng = np.random.default_rng(seed)
        for _ in range(n_perm):
            permA = rng.permutation(n_U)
            sA = U[permA[:n_sA]]
            kA = U[permA[n_sA:n_sA + n_kA]]
            permB = rng.permutation(n_U)
            sB = U[permB[:n_sB]]
            kB = U[permB[n_sB:n_sB + n_kB]]
            vA_null = coords[kA].mean(axis=0) - coords[sA].mean(axis=0)
            vB_null = coords[kB].mean(axis=0) - coords[sB].mean(axis=0)
            uA_null = _safe_unit(vA_null)
            uB_null = _safe_unit(vB_null)
            if uA_null is None or uB_null is None:
                continue
            null_cos_neg.append(float(np.clip(np.dot(uA_null, -uB_null), -1.0, 1.0)))
    if null_cos_neg:
        arr = np.array(null_cos_neg, dtype=float)
        null_med = float(np.median(arr))
        null_95 = float(np.quantile(arr, 0.95))
        p_val = float((arr >= cos_neg).sum() + 1) / float(arr.size + 1)
    else:
        null_med = float("nan")
        null_95 = float("nan")
        p_val = float("nan")
    return AxisPairAlignment(
        cos_A_pos_B=cos_pos,
        cos_A_neg_B=cos_neg,
        angle_A_pos_B_deg=angle_pos,
        angle_A_neg_B_deg=angle_neg,
        null_cos_A_neg_B_median=null_med,
        null_cos_A_neg_B_95=null_95,
        p_one_sided_axis_reversal=p_val,
        n_perm_completed=len(null_cos_neg),
    )


# ============================================================================
# Layer 3 — axis projection slope
# ============================================================================


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
    ry = np.argsort(np.argsort(y, kind="stable"), kind="stable").astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(rx, ry) / denom)


def compute_axis_projection_slope(
    rank_dense: np.ndarray,
    universe_mask: np.ndarray,
    coords: np.ndarray,
    u_axis_A: np.ndarray,
    cluster_id: int,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = DEFAULT_SEED,
) -> AxisProjectionSlope:
    """Project universe channels onto axis_A; regress rank vs projection.

    The discriminative test is ``cluster_id == B`` (rank_B vs proj_axisA):
      - axis_reversal → slope < 0 with high r²
      - dual_source   → slope ≈ 0 with low r²
      - same_direction → slope > 0 with high r²

    Cluster A's slope on its own axis is trivially positive monotone (sanity).

    Permutation null: shuffle the universe-channel rank labels; recompute slope.
    """
    rank_dense = np.asarray(rank_dense, dtype=float)
    universe_mask = np.asarray(universe_mask, dtype=bool)
    coords = np.asarray(coords, dtype=float)
    u_A = _safe_unit(u_axis_A)
    if u_A is None:
        return AxisProjectionSlope(
            cluster_id=int(cluster_id),
            slope=float("nan"), intercept=float("nan"),
            r2=float("nan"), spearman_rho=float("nan"),
            n_points=0,
            null_slope_median=float("nan"),
            p_one_sided_neg_slope=float("nan"),
        )
    idx = np.where(universe_mask)[0]
    if idx.size < 3:
        return AxisProjectionSlope(
            cluster_id=int(cluster_id),
            slope=float("nan"), intercept=float("nan"),
            r2=float("nan"), spearman_rho=float("nan"),
            n_points=int(idx.size),
            null_slope_median=float("nan"),
            p_one_sided_neg_slope=float("nan"),
        )
    x = coords[idx] @ u_A   # projection on u_A
    y = rank_dense[idx]
    # Linear regression slope (y = a + b*x)
    x_c = x - x.mean()
    y_c = y - y.mean()
    var_x = float((x_c * x_c).sum())
    if var_x < 1e-12:
        return AxisProjectionSlope(
            cluster_id=int(cluster_id),
            slope=float("nan"), intercept=float("nan"),
            r2=float("nan"), spearman_rho=float("nan"),
            n_points=int(idx.size),
            null_slope_median=float("nan"),
            p_one_sided_neg_slope=float("nan"),
        )
    slope = float((x_c * y_c).sum() / var_x)
    intercept = float(y.mean() - slope * x.mean())
    var_y = float((y_c * y_c).sum())
    r2 = float(((x_c * y_c).sum()) ** 2 / (var_x * var_y)) if var_y > 1e-12 else float("nan")
    rho = _spearman_rho(x, y)
    # Permutation null: shuffle y (rank labels) across universe channels
    rng = np.random.default_rng(seed)
    null_slopes: List[float] = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        y_perm_c = y_perm - y_perm.mean()
        null_slopes.append(float((x_c * y_perm_c).sum() / var_x))
    null_arr = np.array(null_slopes, dtype=float)
    null_med = float(np.median(null_arr))
    # one-sided: how often does null <= observed (test for negative slope)
    p_neg = float((null_arr <= slope).sum() + 1) / float(null_arr.size + 1)
    return AxisProjectionSlope(
        cluster_id=int(cluster_id),
        slope=slope,
        intercept=intercept,
        r2=r2,
        spearman_rho=rho,
        n_points=int(idx.size),
        null_slope_median=null_med,
        p_one_sided_neg_slope=p_neg,
    )


# ============================================================================
# Layer 2 — event-level direction (optional; needs per-event ranks/bools/labels)
# ============================================================================


def compute_event_direction_stats(
    ranks_full: np.ndarray,
    bools_full: np.ndarray,
    event_labels: np.ndarray,
    universe_mask: np.ndarray,
    coords: np.ndarray,
    u_axis_A: np.ndarray,
    cluster_id: int,
    k_event: int = DEFAULT_K_EVENT,
) -> EventDirectionStats:
    """Per-event direction vector for one cluster, projected to u_A as 1D cos.

    Args:
        ranks_full: (n_ch, n_events) per-event lag rank (raw lagPatRank — phantom
            channels carry int ranks but `bools_full` filters them out).
        bools_full: (n_ch, n_events) per-event participation bool.
        event_labels: (n_events,) cluster id per event.
        universe_mask: (n_ch,) bool — joint_valid ∧ mapped_mask.
        coords: (n_ch, 3) mm.
        u_axis_A: unit vector of template A (for cos projection).
        cluster_id: which cluster to filter events on.
        k_event: per-event top-k early / last-k late.

    For each event in this cluster:
        1. eligible channels = universe ∧ bools_full[:, e]
        2. need n_eligible >= 2*k_event
        3. early_k = argsort(rank_e[eligible])[:k_event], late_k = [-k_event:]
        4. v_event = mean(coords[late_k]) - mean(coords[early_k])
        5. cos_e = (v_event / ||v_event||) · u_A
    """
    ranks_full = np.asarray(ranks_full, dtype=float)
    bools_full = np.asarray(bools_full, dtype=bool)
    event_labels = np.asarray(event_labels, dtype=int)
    universe_mask = np.asarray(universe_mask, dtype=bool)
    coords = np.asarray(coords, dtype=float)
    u_A = _safe_unit(u_axis_A)
    n_ch, n_ev = ranks_full.shape
    if u_A is None or n_ev == 0 or event_labels.size != n_ev:
        return EventDirectionStats(
            cluster_id=int(cluster_id),
            k_event=int(k_event),
            n_events_total=0,
            n_events_eligible=0,
            cos_with_u_A_median=float("nan"),
            cos_with_u_A_q25=float("nan"),
            cos_with_u_A_q75=float("nan"),
            mean_unit_vector=[float("nan"), float("nan"), float("nan")],
            mean_resultant_length=float("nan"),
        )
    event_mask = event_labels == cluster_id
    cluster_events = np.where(event_mask)[0]
    cos_list: List[float] = []
    unit_vectors: List[np.ndarray] = []
    universe_idx = np.where(universe_mask)[0]
    for e in cluster_events:
        elig = universe_mask & bools_full[:, e]
        elig_idx = np.where(elig)[0]
        if elig_idx.size < MIN_EVENT_ELIGIBLE_FACTOR * k_event:
            continue
        sub_ranks = ranks_full[elig_idx, e]
        order = np.argsort(sub_ranks, kind="stable")
        early = elig_idx[order[:k_event]]
        late = elig_idx[order[-k_event:]]
        v = coords[late].mean(axis=0) - coords[early].mean(axis=0)
        u = _safe_unit(v)
        if u is None:
            continue
        cos_list.append(float(np.clip(np.dot(u, u_A), -1.0, 1.0)))
        unit_vectors.append(u)
    if not cos_list:
        return EventDirectionStats(
            cluster_id=int(cluster_id),
            k_event=int(k_event),
            n_events_total=int(cluster_events.size),
            n_events_eligible=0,
            cos_with_u_A_median=float("nan"),
            cos_with_u_A_q25=float("nan"),
            cos_with_u_A_q75=float("nan"),
            mean_unit_vector=[float("nan"), float("nan"), float("nan")],
            mean_resultant_length=float("nan"),
        )
    arr = np.array(cos_list, dtype=float)
    uv = np.vstack(unit_vectors)
    mean_uv = uv.mean(axis=0)
    R = float(np.linalg.norm(mean_uv))
    return EventDirectionStats(
        cluster_id=int(cluster_id),
        k_event=int(k_event),
        n_events_total=int(cluster_events.size),
        n_events_eligible=int(arr.size),
        cos_with_u_A_median=float(np.median(arr)),
        cos_with_u_A_q25=float(np.quantile(arr, 0.25)),
        cos_with_u_A_q75=float(np.quantile(arr, 0.75)),
        mean_unit_vector=[float(x) for x in mean_uv],
        mean_resultant_length=R,
    )


# ============================================================================
# Layer 5 — SOZ relation (secondary)
# ============================================================================


def _build_soz_set(
    name: str,
    coords: np.ndarray,
    soz_member_mask: np.ndarray,
    template_A: TemplateAxis,
    template_B: TemplateAxis,
) -> Optional[SOZSet]:
    """Build one SOZSet (centroid + 4 source/sink centroid distances + 4 min-pair distances)."""
    n = int(soz_member_mask.sum())
    if n == 0:
        return None
    soz_coords = coords[soz_member_mask]
    c_SOZ = soz_coords.mean(axis=0)

    def _centroid_dist(centroid: np.ndarray) -> float:
        return float(np.linalg.norm(centroid - c_SOZ))

    def _min_pair_dist(indices: Sequence[int]) -> float:
        if not indices:
            return float("nan")
        pts = coords[list(indices)]
        diffs = pts[:, None, :] - soz_coords[None, :, :]
        return float(np.linalg.norm(diffs, axis=-1).min())

    return SOZSet(
        name=name,
        n=n,
        centroid=[float(x) for x in c_SOZ],
        d_source_A_to_SOZ=_centroid_dist(template_A.source_centroid),
        d_sink_A_to_SOZ=_centroid_dist(template_A.sink_centroid),
        d_source_B_to_SOZ=_centroid_dist(template_B.source_centroid),
        d_sink_B_to_SOZ=_centroid_dist(template_B.sink_centroid),
        min_d_source_A_chs_to_SOZ_chs=_min_pair_dist(template_A.source_indices),
        min_d_sink_A_chs_to_SOZ_chs=_min_pair_dist(template_A.sink_indices),
        min_d_source_B_chs_to_SOZ_chs=_min_pair_dist(template_B.source_indices),
        min_d_sink_B_chs_to_SOZ_chs=_min_pair_dist(template_B.sink_indices),
    )


def compute_soz_relation(
    coords: np.ndarray,
    universe_mask: np.ndarray,
    soz_mask: Optional[np.ndarray],
    template_A: TemplateAxis,
    template_B: TemplateAxis,
    mapped_mask: Optional[np.ndarray] = None,
) -> SOZRelation:
    """Distance readouts from each cluster's source/sink to SOZ. Descriptive only.

    v1.0.2 audit fix #5: report TWO SOZ-set cuts:
      - mapped_full: soz_mask ∧ mapped_mask  (broader; not restricted to joint_valid)
      - joint_universe: soz_mask ∧ universe_mask  (= soz ∧ joint_valid ∧ mapped)

    If ``mapped_mask`` is None, the mapped-full set falls back to ``soz_mask`` restricted
    to channels with finite coords (caller-side mapped-mask reconstruction).
    """
    coords = np.asarray(coords, dtype=float)
    universe_mask = np.asarray(universe_mask, dtype=bool)
    if soz_mask is None:
        return SOZRelation(
            mapped_full=None, joint_universe=None, exit_reason="no_soz_mask",
            soz_centroid=None, n_soz_in_universe=0,
            d_source_A_to_SOZ=None, d_sink_A_to_SOZ=None,
            d_source_B_to_SOZ=None, d_sink_B_to_SOZ=None,
            min_d_source_A_chs_to_SOZ_chs=None,
            min_d_sink_A_chs_to_SOZ_chs=None,
            min_d_source_B_chs_to_SOZ_chs=None,
            min_d_sink_B_chs_to_SOZ_chs=None,
        )
    soz_mask = np.asarray(soz_mask, dtype=bool)
    if mapped_mask is None:
        # Fallback: any channel with finite coords counts as mapped
        mapped_mask = np.isfinite(coords).all(axis=1)
    else:
        mapped_mask = np.asarray(mapped_mask, dtype=bool)

    soz_mapped_full_mask = soz_mask & mapped_mask
    soz_joint_mask = soz_mask & universe_mask

    mapped_full = _build_soz_set("mapped_full", coords, soz_mapped_full_mask, template_A, template_B)
    joint_universe = _build_soz_set("joint_universe", coords, soz_joint_mask, template_A, template_B)

    exit_reason = None
    if mapped_full is None and joint_universe is None:
        exit_reason = "soz_empty_in_mapped_and_universe"
    elif joint_universe is None:
        exit_reason = "soz_empty_in_universe"  # mapped-full still informative

    # Backward-compat top-level fields mirror joint_universe (which was the v1.0.1 behavior)
    ju = joint_universe
    return SOZRelation(
        mapped_full=mapped_full,
        joint_universe=joint_universe,
        exit_reason=exit_reason,
        soz_centroid=ju.centroid if ju else None,
        n_soz_in_universe=ju.n if ju else 0,
        d_source_A_to_SOZ=ju.d_source_A_to_SOZ if ju else None,
        d_sink_A_to_SOZ=ju.d_sink_A_to_SOZ if ju else None,
        d_source_B_to_SOZ=ju.d_source_B_to_SOZ if ju else None,
        d_sink_B_to_SOZ=ju.d_sink_B_to_SOZ if ju else None,
        min_d_source_A_chs_to_SOZ_chs=ju.min_d_source_A_chs_to_SOZ_chs if ju else None,
        min_d_sink_A_chs_to_SOZ_chs=ju.min_d_sink_A_chs_to_SOZ_chs if ju else None,
        min_d_source_B_chs_to_SOZ_chs=ju.min_d_source_B_chs_to_SOZ_chs if ju else None,
        min_d_sink_B_chs_to_SOZ_chs=ju.min_d_sink_B_chs_to_SOZ_chs if ju else None,
    )


# ============================================================================
# Verdict (combines Layer 1, 3, 4)
# ============================================================================


def assess_descriptive_geometry(
    n_universe: int,
    degeneracy: DegeneracyResult,
    alignment: AxisPairAlignment,
    slope_B_on_axisA: AxisProjectionSlope,
) -> DescriptiveGeometry:
    """Geometric pattern label, ignoring permutation-null gates.

    Captures the geometric shape observed regardless of whether the permutation null
    can resolve it. Useful when role-shuffle null saturates (decision_k large vs
    n_universe) but the cos-similarity geometry is clear.
    """
    if n_universe < MIN_UNIVERSE_SIZE:
        return DescriptiveGeometry(label="unclear", reason=f"n_universe={n_universe} < {MIN_UNIVERSE_SIZE}")
    if degeneracy.is_near_1d:
        return DescriptiveGeometry(
            label="unclear",
            reason=f"near-1D layout (lambda_ratio_12={degeneracy.lambda_ratio_12:.3f}); direction not interpretable",
        )
    cos_neg = alignment.cos_A_neg_B
    cos_pos = alignment.cos_A_pos_B
    slope_B = slope_B_on_axisA.slope
    r2_B = slope_B_on_axisA.r2
    if not np.isfinite(cos_neg):
        return DescriptiveGeometry(label="unclear", reason="alignment NaN")
    # v1.0.2 audit fix: axis_reversal_shaped + same_direction_shaped now require r² >= 0.20
    # in addition to cos+slope-sign (closes silent noise-driven false-positive path).
    if (cos_neg >= COS_AXIS_REVERSAL_THRESHOLD
            and np.isfinite(slope_B) and slope_B < 0
            and np.isfinite(r2_B) and r2_B >= R2_DESCRIPTIVE_MIN):
        return DescriptiveGeometry(
            label="axis_reversal_shaped",
            reason=(f"cos(A,-B)={cos_neg:.3f}>=0.5 AND slope_B_on_axisA={slope_B:.3f}<0 "
                    f"AND r2_B_on_axisA={r2_B:.3f}>=0.20"),
        )
    if (cos_pos >= COS_SAME_DIRECTION_THRESHOLD
            and np.isfinite(slope_B) and slope_B > 0
            and np.isfinite(r2_B) and r2_B >= R2_DESCRIPTIVE_MIN):
        return DescriptiveGeometry(
            label="same_direction_shaped",
            reason=(f"cos(A,+B)={cos_pos:.3f}>=0.5 AND slope_B_on_axisA={slope_B:.3f}>0 "
                    f"AND r2_B_on_axisA={r2_B:.3f}>=0.20"),
        )
    if abs(cos_neg) < DUAL_SOURCE_COS_ABS_MAX and np.isfinite(r2_B) and r2_B < R2_DUAL_SOURCE_MAX:
        return DescriptiveGeometry(
            label="dual_source_shaped",
            reason=f"|cos(A,-B)|={abs(cos_neg):.3f}<0.5 AND r2_B_on_axisA={r2_B:.3f}<0.20",
        )
    return DescriptiveGeometry(
        label="unclear",
        reason=(f"cos(A,-B)={cos_neg:.3f}, slope_B={slope_B:.3f}, r2_B={r2_B:.3f}; "
                f"no descriptive pattern triggers (r2 < 0.20 with cos-shape signal is unclear, not shape)"),
    )


def assess_subject_verdict(
    n_universe: int,
    degeneracy: DegeneracyResult,
    alignment: AxisPairAlignment,
    slope_B_on_axisA: AxisProjectionSlope,
) -> SubjectVerdict:
    """Per-subject verdict per archive plan §3.7. Lock - no post-hoc adjustment."""
    if n_universe < MIN_UNIVERSE_SIZE:
        return SubjectVerdict(label="exit_no_universe",
                              reason=f"n_universe={n_universe} < {MIN_UNIVERSE_SIZE}")
    if degeneracy.is_near_1d:
        return SubjectVerdict(
            label="degenerate_geometry",
            reason=(f"PCA lambda_ratio_12={degeneracy.lambda_ratio_12:.3f} "
                    f"< {DEGENERACY_LAMBDA_RATIO_12_THRESHOLD}; universe is near-1D"),
        )
    if not (np.isfinite(alignment.cos_A_neg_B) and np.isfinite(slope_B_on_axisA.slope)):
        return SubjectVerdict(label="inconclusive",
                              reason="axis pair alignment or slope is NaN (degenerate axis vector)")

    cos_neg = alignment.cos_A_neg_B
    cos_pos = alignment.cos_A_pos_B
    p_align = alignment.p_one_sided_axis_reversal
    slope_B = slope_B_on_axisA.slope
    p_slope_neg = slope_B_on_axisA.p_one_sided_neg_slope
    r2_B = slope_B_on_axisA.r2

    # axis_reversal: cos(A,-B) >= 0.5 AND align p < 0.05 AND slope_B < 0 AND p_slope < 0.05
    axis_reversal = (
        cos_neg >= COS_AXIS_REVERSAL_THRESHOLD
        and np.isfinite(p_align) and p_align < AXIS_PERM_PVALUE_THRESHOLD
        and slope_B < 0
        and np.isfinite(p_slope_neg) and p_slope_neg < SLOPE_PVALUE_THRESHOLD
    )
    if axis_reversal:
        return SubjectVerdict(
            label="axis_reversal",
            reason=(f"cos(A,-B)={cos_neg:.3f}>=0.5 (perm p={p_align:.3f}) AND "
                    f"slope_B_on_axisA={slope_B:.3f}<0 (perm p={p_slope_neg:.3f})"),
        )

    # same_direction: cos(A,+B) >= 0.5 (templates share direction)
    if cos_pos >= COS_SAME_DIRECTION_THRESHOLD:
        return SubjectVerdict(
            label="same_direction",
            reason=(f"cos(A,+B)={cos_pos:.3f}>=0.5; templates share direction "
                    f"(typical of swap_class=none)"),
        )

    # dual_source: |cos(A,-B)| < 0.5 (angle in 60..120 deg) AND r2_B_on_axisA < 0.2
    if abs(cos_neg) < DUAL_SOURCE_COS_ABS_MAX and np.isfinite(r2_B) and r2_B < R2_DUAL_SOURCE_MAX:
        return SubjectVerdict(
            label="dual_source",
            reason=(f"|cos(A,-B)|={abs(cos_neg):.3f}<0.5 (orthogonal-ish axes) AND "
                    f"r2_B_on_axisA={r2_B:.3f}<0.20"),
        )
    return SubjectVerdict(
        label="inconclusive",
        reason=(f"cos(A,-B)={cos_neg:.3f}, perm p={p_align if np.isfinite(p_align) else float('nan'):.3f}; "
                f"slope_B_on_axisA={slope_B:.3f}, p_neg={p_slope_neg if np.isfinite(p_slope_neg) else float('nan'):.3f}; "
                f"r2_B={r2_B:.3f}; no verdict triggers fire"),
    )


# ============================================================================
# JSON-serialization helpers
# ============================================================================


def _arr_to_list(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return [_arr_to_list(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return float(x) if isinstance(x, np.floating) else int(x)
    return x


def serialize_template_axis(ta: TemplateAxis) -> Dict[str, Any]:
    return {
        "cluster_id": ta.cluster_id,
        "source_indices": ta.source_indices,
        "sink_indices": ta.sink_indices,
        "source_centroid": _arr_to_list(ta.source_centroid),
        "sink_centroid": _arr_to_list(ta.sink_centroid),
        "v_axis": _arr_to_list(ta.v_axis),
        "axis_length": ta.axis_length,
    }


def serialize_alignment(a: AxisPairAlignment) -> Dict[str, Any]:
    return asdict(a)


def serialize_slope(s: AxisProjectionSlope) -> Dict[str, Any]:
    return asdict(s)


def serialize_event_stats(es: EventDirectionStats) -> Dict[str, Any]:
    return asdict(es)


def serialize_soz_relation(s: SOZRelation) -> Dict[str, Any]:
    return {
        "exit_reason": s.exit_reason,
        "mapped_full": asdict(s.mapped_full) if s.mapped_full is not None else None,
        "joint_universe": asdict(s.joint_universe) if s.joint_universe is not None else None,
        # Backward-compat top-level mirrors joint_universe
        "soz_centroid": s.soz_centroid,
        "n_soz_in_universe": s.n_soz_in_universe,
        "d_source_A_to_SOZ": s.d_source_A_to_SOZ,
        "d_sink_A_to_SOZ": s.d_sink_A_to_SOZ,
        "d_source_B_to_SOZ": s.d_source_B_to_SOZ,
        "d_sink_B_to_SOZ": s.d_sink_B_to_SOZ,
        "min_d_source_A_chs_to_SOZ_chs": s.min_d_source_A_chs_to_SOZ_chs,
        "min_d_sink_A_chs_to_SOZ_chs": s.min_d_sink_A_chs_to_SOZ_chs,
        "min_d_source_B_chs_to_SOZ_chs": s.min_d_source_B_chs_to_SOZ_chs,
        "min_d_sink_B_chs_to_SOZ_chs": s.min_d_sink_B_chs_to_SOZ_chs,
    }


def serialize_degeneracy(d: DegeneracyResult) -> Dict[str, Any]:
    return {
        "lambda_eigenvalues": list(d.lambda_eigenvalues),
        "lambda_ratio_12": d.lambda_ratio_12,
        "lambda_ratio_23": d.lambda_ratio_23,
        "is_near_1d": d.is_near_1d,
        "is_near_2d": d.is_near_2d,
    }
