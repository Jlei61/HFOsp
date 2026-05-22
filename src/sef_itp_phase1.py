"""SEF-ITP Phase 1: spatial geometry hypotheses H1 / H2 / H6.

Plan-of-record: docs/archive/topic4/sef_itp_phase1/plan_2026-05-21.md
Upstream framework: docs/topic4_sef_itp_framework.md v1.0.2

Contract (v1.0.2 lock):
  - All compute functions take coords + valid pool indices EXPLICITLY.
    No implicit coord loader. No default valid_mask=None.
  - Distance metrics: euclidean_3d (main, requires coords) | shaft_ordinal (fallback,
    requires only channel names). cortical_surface / SC reserved for future.
  - Null constructions are pre-registered:
      H6: shaft-stratified shuffle of participation rate
      H1: matched random sampling with shaft + participation + HFO rate, with
          degradation order (drop_hfo_rate first, drop_participation second)
      H1c: centroid from VALID channels MINUS endpoint set (advisor fix:
           avoids circularity where high-participation centroid is partially
           defined by the endpoint set being tested)
      H2: role-label shuffle within (S_A ∪ K_A ∪ S_B ∪ K_B) pool
  - All RNG via np.random.default_rng with explicit seed.
  - Channel ordering: every function entry that takes (channel_names, coords,
    participation, ...) asserts these are aligned (same length / same indexing).

This module is data-agnostic (synthetic inputs accepted). Real-data wiring is
gated on Topic 0 Phase 0a (phantom-rank fix complete) + Phase 0b (coord-loader
PR, NOT YET STARTED — see docs/topic0_methodology_audits.md §3.2).
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# Channel name parsing (shaft prefix + ordinal)
# =============================================================================


def _parse_channel(name: str) -> Tuple[Optional[str], int]:
    """Parse SEEG channel name into (shaft_prefix, ordinal).

    Handles monopolar 'A1', bipolar 'A1-A2', primed 'A\\'1-A\\'2'.
    Returns (None, 0) if cannot parse.
    """
    if "-" in name:
        name = name.split("-", 1)[0].strip()
    # find digit boundary
    for i, c in enumerate(name):
        if c.isdigit():
            if i == 0:
                return None, 0
            try:
                return name[:i], int(name[i:])
            except ValueError:
                return None, 0
    return (name if name else None), 0


# =============================================================================
# Distance metrics
# =============================================================================


def pairwise_3d_euclidean(coords: np.ndarray) -> np.ndarray:
    """Pairwise 3D Euclidean distance matrix from (n_ch, 3) coords."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be (n_ch, 3), got {coords.shape}")
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def pairwise_shaft_ordinal(channel_names: Sequence[str]) -> np.ndarray:
    """Pairwise distance from channel names: within-shaft = |idx diff|, cross-shaft = +inf.

    Use this when 3D coords are not available (current state per Topic 0 §3.2).
    """
    parsed = [_parse_channel(nm) for nm in channel_names]
    n = len(parsed)
    D = np.full((n, n), np.inf)
    for i in range(n):
        D[i, i] = 0.0
        si, oi = parsed[i]
        for j in range(i + 1, n):
            sj, oj = parsed[j]
            if si is not None and si == sj:
                d = float(abs(oi - oj))
                D[i, j] = d
                D[j, i] = d
    return D


# =============================================================================
# H6 — Participation field spatial segregation
# =============================================================================


def compute_participation_rate(events_bool: np.ndarray) -> np.ndarray:
    """For events_bool (n_ch, n_events) boolean array, return per-channel participation rate."""
    arr = np.asarray(events_bool)
    if arr.ndim != 2:
        raise ValueError(f"events_bool must be 2D, got {arr.shape}")
    return arr.astype(bool).mean(axis=1)


def compute_h6_segregation(
    participation: np.ndarray,
    channel_names: Sequence[str],
    coords: Optional[np.ndarray] = None,
    distance_metric: Literal["euclidean", "shaft_ordinal"] = "shaft_ordinal",
    n_permutations: int = 1000,
    rng: Optional[np.random.Generator] = None,
    high_threshold: float = 0.5,
) -> Dict:
    """H6: participation field spatial segregation.

    Tests whether high-participation and low-participation channels are spatially
    segregated, against shaft-stratified shuffle null.

    Verdict: PASS | NULL | PARTIAL | FAIL | EXCLUDED_SINGLE_SHAFT

    Single-shaft subjects are EXCLUDED (not fallback to global shuffle, per advisor
    2026-05-21 fix: within-shaft = global = no constraint = no information).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    participation = np.asarray(participation, dtype=float)
    n = len(participation)
    if n != len(channel_names):
        raise ValueError(
            f"channel_names ({len(channel_names)}) != participation ({n})"
        )
    if coords is not None:
        coords = np.asarray(coords, dtype=float)
        if coords.shape[0] != n:
            raise ValueError(
                f"coords ({coords.shape[0]}) != participation ({n})"
            )

    shafts = [_parse_channel(nm)[0] for nm in channel_names]
    valid_shafts = [s for s in shafts if s is not None]
    n_shafts = len(set(valid_shafts))

    if n_shafts <= 1:
        return {
            "verdict": "EXCLUDED_SINGLE_SHAFT",
            "n_shafts": n_shafts,
            "n_channels": n,
            "reason": "Single-shaft subject; shaft-stratified shuffle is degenerate (advisor 2026-05-21 fix)",
        }

    if distance_metric == "euclidean":
        if coords is None:
            raise ValueError(
                "euclidean distance requires coords (None given). "
                "Use distance_metric='shaft_ordinal' when coords unavailable."
            )
        D = pairwise_3d_euclidean(coords)
    elif distance_metric == "shaft_ordinal":
        D = pairwise_shaft_ordinal(channel_names)
    else:
        raise ValueError(f"unknown distance_metric: {distance_metric}")

    high_mask = participation >= high_threshold
    low_mask = ~high_mask

    if high_mask.sum() < 2 or low_mask.sum() < 2:
        return {
            "verdict": "INSUFFICIENT_SPLIT",
            "n_high": int(high_mask.sum()),
            "n_low": int(low_mask.sum()),
            "high_threshold": float(high_threshold),
        }

    morans_i_actual = _morans_i(participation, D)
    centroid_d_actual = _centroid_distance(D, high_mask, low_mask)
    silhouette_actual = _silhouette_basic(D, high_mask, low_mask)

    morans_i_null: List[float] = []
    centroid_d_null: List[float] = []
    silhouette_null: List[float] = []

    for _ in range(n_permutations):
        shuffled = _shaft_stratified_shuffle(participation, shafts, rng)
        h_m = shuffled >= high_threshold
        l_m = ~h_m
        if h_m.sum() < 2 or l_m.sum() < 2:
            continue
        morans_i_null.append(_morans_i(shuffled, D))
        centroid_d_null.append(_centroid_distance(D, h_m, l_m))
        silhouette_null.append(_silhouette_basic(D, h_m, l_m))

    if len(morans_i_null) < 100:
        return {
            "verdict": "INSUFFICIENT_NULL",
            "n_null": len(morans_i_null),
        }

    morans_arr = np.asarray(morans_i_null)
    centroid_arr = np.asarray(centroid_d_null)
    silhouette_arr = np.asarray(silhouette_null)

    # one-sided: real value > null
    morans_p = float((np.sum(morans_arr >= morans_i_actual) + 1) / (len(morans_arr) + 1))
    centroid_p = float((np.sum(centroid_arr >= centroid_d_actual) + 1) / (len(centroid_arr) + 1))
    silhouette_p = float((np.sum(silhouette_arr >= silhouette_actual) + 1) / (len(silhouette_arr) + 1))

    n_significant = sum(p < 0.05 for p in [morans_p, centroid_p, silhouette_p])
    if n_significant >= 2:
        verdict = "PASS"
    elif n_significant == 0:
        verdict = "NULL"
    else:
        verdict = "PARTIAL"

    return {
        "verdict": verdict,
        "n_channels": n,
        "n_shafts": n_shafts,
        "distance_metric_used": distance_metric,
        "high_threshold": float(high_threshold),
        "n_high_participation": int(high_mask.sum()),
        "n_low_participation": int(low_mask.sum()),
        "morans_i_actual": float(morans_i_actual),
        "morans_i_null_median": float(np.median(morans_arr)),
        "morans_i_null_p": morans_p,
        "centroid_distance_actual": float(centroid_d_actual),
        "centroid_distance_null_median": float(np.median(centroid_arr)),
        "centroid_distance_null_p": centroid_p,
        "silhouette_actual": float(silhouette_actual),
        "silhouette_null_median": float(np.median(silhouette_arr)),
        "silhouette_null_p": silhouette_p,
        "n_permutations": n_permutations,
        "n_null_used": len(morans_arr),
    }


def _shaft_stratified_shuffle(
    values: np.ndarray,
    shafts: Sequence[Optional[str]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute values within each shaft; preserves cross-shaft mean per shaft."""
    out = values.copy()
    by_shaft: Dict[Optional[str], List[int]] = {}
    for i, s in enumerate(shafts):
        by_shaft.setdefault(s, []).append(i)
    for _, idxs in by_shaft.items():
        if len(idxs) > 1:
            perm = rng.permutation(out[idxs])
            for k, i in enumerate(idxs):
                out[i] = perm[k]
    return out


def _morans_i(values: np.ndarray, distance: np.ndarray) -> float:
    """Moran's I with inverse-distance weights (w_ii=0; w_ij=1/d_ij for finite d>0)."""
    n = len(values)
    finite = np.isfinite(distance) & (distance > 0)
    W = np.zeros_like(distance, dtype=float)
    W[finite] = 1.0 / distance[finite]
    W_sum = W.sum()
    if W_sum == 0:
        return 0.0
    mean = values.mean()
    dev = values - mean
    denom = (dev ** 2).sum()
    if denom == 0:
        return 0.0
    numerator = (W * np.outer(dev, dev)).sum()
    return float((n / W_sum) * (numerator / denom))


def _centroid_distance(D: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Mean cross-group distance between mask_a and mask_b."""
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return 0.0
    cross = D[np.ix_(mask_a, mask_b)]
    finite = cross[np.isfinite(cross)]
    if len(finite) == 0:
        return 0.0
    return float(finite.mean())


def _silhouette_basic(D: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Mean silhouette score given two-group split (binary mask_a / mask_b)."""
    n = len(mask_a)
    scores: List[float] = []
    for i in range(n):
        if mask_a[i]:
            same = mask_a.copy()
            same[i] = False
            other = mask_b
        else:
            same = mask_b.copy()
            same[i] = False
            other = mask_a
        if same.sum() == 0 or other.sum() == 0:
            continue
        a_d = D[i, same]
        b_d = D[i, other]
        af = a_d[np.isfinite(a_d)]
        bf = b_d[np.isfinite(b_d)]
        if len(af) == 0 or len(bf) == 0:
            continue
        ai, bi = af.mean(), bf.mean()
        denom = max(ai, bi)
        if denom == 0:
            continue
        scores.append((bi - ai) / denom)
    return float(np.mean(scores)) if scores else 0.0


# =============================================================================
# H1 — Endpoint compactness, three independent layers (v1.0.3, 2026-05-21)
#
# H1 asks three different questions about endpoints. v1 plan collapsed them
# into one PASS/NULL/FAIL verdict — but user audit 2026-05-21 pointed out
# that "NULL on strict matched-null" doesn't mean "endpoints not compact",
# only "extra compactness beyond shaft geometry is weak". Three layers:
#
#   1. h1_descriptive       — absolute distance / diameter / radius of gyration.
#                             No null, no verdict. Report numbers.
#
#   2. h1_strict (was h1_compactness) — endpoint vs matched random null.
#                             Tests EXTRA compactness BEYOND shaft geometry.
#                             PASS / NULL / FAIL_DIFFUSE.
#
#   3. h1_envelope          — endpoint inside pathological participation field.
#                             NECESSARY condition. PASS / FAIL only (no NULL).
#
# Overall H1 verdict:
#   if envelope == FAIL: H1 = FAIL (necessary condition overrides)
#   else: H1 = strict.verdict
# =============================================================================


def _assert_coords_are_mm(coord_units: Optional[str], func_name: str) -> None:
    """Coord_units contract check (v1.0.5, advisor 2026-05-21 #1 fix).

    Phase 1 main analyses (H1 strict / H1 descriptive / H2 spatial) require mm
    coordinates. Voxel coords from Epilepsiae default loader must NOT silently
    feed main analysis (coord_loader plan §4 invariant #6).

    coord_units=None is permitted (backward-compat for synthetic tests passing
    raw numpy arrays without metadata); when provided, MUST be "mm".
    """
    if coord_units is not None and coord_units != "mm":
        raise ValueError(
            f"{func_name}: coord_units must be 'mm' for main analysis; "
            f"got {coord_units!r}. For Epilepsiae voxel coords, pass mri_affine "
            f"to load_subject_coords() to convert to ras_mm_via_affine first."
        )


def compute_h1_descriptive(
    members: Sequence[int],
    coords: Optional[np.ndarray],
    channel_names: Sequence[str],
    distance_metric: Literal["euclidean", "shaft_ordinal"] = "euclidean",
    coord_units: Optional[str] = None,
) -> Dict:
    """H1 descriptive layer: absolute spatial statistics, NO null, NO verdict.

    Returns mean pairwise distance, diameter (max pairwise), radius of gyration.
    Use to describe "how spread out are the endpoints" without making any
    PASS/NULL/FAIL claim — those belong to h1_strict (matched-null) or
    h1_envelope (necessary condition).

    coord_units: when provided, must be 'mm' for euclidean distance metric.
    Raises ValueError if voxel coords are passed (v1.0.5 contract).
    """
    if distance_metric == "euclidean":
        _assert_coords_are_mm(coord_units, "compute_h1_descriptive")
    members = list(members)
    if len(members) < 2:
        return {"verdict": "DESCRIPTIVE_INSUFFICIENT", "n_members": len(members)}

    if distance_metric == "euclidean":
        if coords is None:
            return {
                "verdict": "DESCRIPTIVE_GATED_ON_COORDS",
                "reason": "euclidean descriptive layer requires 3D coords (Topic 0 §3.2 coord loader)",
            }
        coords = np.asarray(coords, dtype=float)
        D = pairwise_3d_euclidean(coords)
        sub_coords = coords[members]
        centroid = sub_coords.mean(axis=0)
        radius_gyration = float(np.sqrt(np.mean(np.sum((sub_coords - centroid) ** 2, axis=1))))
    elif distance_metric == "shaft_ordinal":
        D = pairwise_shaft_ordinal(channel_names)
        radius_gyration = None  # not meaningful in 1D ordinal
    else:
        raise ValueError(f"unknown distance_metric: {distance_metric}")

    sub = D[np.ix_(members, members)]
    iu = np.triu_indices_from(sub, k=1)
    vals = sub[iu]
    finite = vals[np.isfinite(vals)]

    if len(finite) == 0:
        return {
            "verdict": "DESCRIPTIVE_NO_FINITE_DISTANCES",
            "distance_metric_used": distance_metric,
            "n_members": len(members),
            "reason": "all pairwise distances are inf (likely cross-shaft with shaft_ordinal)",
        }

    return {
        "verdict": "DESCRIPTIVE",  # not a PASS/NULL — just describing
        "mean_pairwise_distance": float(finite.mean()),
        "diameter": float(finite.max()),
        "radius_of_gyration": radius_gyration,
        "distance_metric_used": distance_metric,
        "n_members": len(members),
        "n_finite_pairs": int(len(finite)),
        "n_inf_pairs": int(len(vals) - len(finite)),
    }


# =============================================================================
# H1 strict layer + envelope layer (existing functions, kept; semantic renames in docstring)
# =============================================================================


def compute_h1_compactness(
    members: Sequence[int],
    candidate_pool: Sequence[int],
    coords: Optional[np.ndarray],
    channel_names: Sequence[str],
    participation: np.ndarray,
    hfo_rate: Optional[np.ndarray] = None,
    distance_metric: Literal["euclidean", "shaft_ordinal"] = "euclidean",
    n_null: int = 1000,
    rng: Optional[np.random.Generator] = None,
    coord_units: Optional[str] = None,
) -> Dict:
    """H1 STRICT layer (v1.0.7, 2026-05-22).

    Tests whether endpoints are MORE COMPACT than a shaft-matched random
    subset drawn from the ENTIRE SEEG implantation pool. Asking:
    "Is there extra spatial compactness beyond what same-shaft anchoring
    would predict, when 'random' is sampled across the whole implantation
    grid (not just lagPat-selected channels)?"

    v1.0.7 null contract:
      - candidate_pool = entire SEEG implantation, MINUS endpoint
      - matched only on shaft distribution (preserves the geometric
        backbone — without it the null is dominated by cross-shaft pairs
        with very large mean distance, trivially making actual look compact)
      - participation rate / HFO rate constraints REMOVED (v1.0.6 used them
        but the pool was lagPat-restricted, producing a circular null
        where endpoint and null both came from the same high-HFO subset)

    NULL means "no extra compactness beyond same-shaft expectation";
    PASS means endpoint compactness exceeds same-shaft random null;
    FAIL_DIFFUSE means endpoint is MORE diffuse than same-shaft random.

    coord_units: when provided, must be 'mm' for euclidean distance metric.
    Raises ValueError if voxel coords are passed (v1.0.5 contract).
    """
    if distance_metric == "euclidean":
        _assert_coords_are_mm(coord_units, "compute_h1_compactness")
    if rng is None:
        rng = np.random.default_rng(0)

    members = list(members)
    if len(members) < 2:
        return {"verdict": "INSUFFICIENT_CHANNELS", "n_members": len(members)}

    n_total = len(channel_names)
    if coords is not None:
        coords = np.asarray(coords, dtype=float)
        if coords.shape[0] != n_total:
            raise ValueError(
                f"coords ({coords.shape[0]}) != channel_names ({n_total})"
            )
    if len(participation) != n_total:
        raise ValueError(
            f"participation ({len(participation)}) != channel_names ({n_total})"
        )
    if hfo_rate is not None and len(hfo_rate) != n_total:
        raise ValueError(
            f"hfo_rate ({len(hfo_rate)}) != channel_names ({n_total})"
        )

    if distance_metric == "euclidean":
        if coords is None:
            raise ValueError("euclidean requires coords; use shaft_ordinal when unavailable")
        D = pairwise_3d_euclidean(coords)
    elif distance_metric == "shaft_ordinal":
        D = pairwise_shaft_ordinal(channel_names)
    else:
        raise ValueError(f"unknown distance_metric: {distance_metric}")

    C_actual = _mean_pairwise_distance(D, members)

    shafts = [_parse_channel(nm)[0] for nm in channel_names]

    null_dists, relaxation_tier = _matched_random_null(
        members=members,
        candidate_pool=candidate_pool,
        shafts=shafts,
        participation=participation,
        hfo_rate=hfo_rate,
        n_null=n_null,
        rng=rng,
        D=D,
    )

    if len(null_dists) < 100:
        return {
            "verdict": "INSUFFICIENT_NULL",
            "n_null_obtained": len(null_dists),
            "relaxation_tier": relaxation_tier,
            "C_actual": float(C_actual),
        }

    null_arr = np.asarray(null_dists)
    null_lower_p = float((np.sum(null_arr <= C_actual) + 1) / (len(null_arr) + 1))
    null_upper_p = float((np.sum(null_arr >= C_actual) + 1) / (len(null_arr) + 1))

    if null_lower_p < 0.05:
        verdict = "PASS"  # actual < null → compact
    elif null_upper_p < 0.05:
        verdict = "FAIL_DIFFUSE"  # actual > null → diffuse
    else:
        verdict = "NULL"

    return {
        "verdict": verdict,
        "C_actual": float(C_actual),
        "C_null_median": float(np.median(null_arr)),
        "null_lower_p": null_lower_p,
        "null_upper_p": null_upper_p,
        "distance_metric_used": distance_metric,
        "n_members": len(members),
        "n_null_samples": len(null_arr),
        "relaxation_tier": relaxation_tier,
    }


def compute_h1c_envelope(
    endpoint_indices: Sequence[int],
    non_endpoint_pool: Sequence[int],
    coords: np.ndarray,
    n_null: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """H1c: endpoint envelope within pathological participation field.

    v1.0.7 (2026-05-22): parameter renamed `valid_indices` → `non_endpoint_pool`
    to match v1.0.7 semantics — caller passes the non-endpoint pool directly
    (typically all mapped SEEG channels MINUS endpoint). Internal subtraction
    removed.

    Centroid is the spatial mean of non_endpoint_pool. Null draws "fake
    endpoint" subsets of size |endpoint| from non_endpoint_pool, recomputes
    centroid with the fake endpoint removed, and reports the ratio
    distribution.

    Returns ratio = mean_d(endpoint→centroid) / mean_d(non_endpoint→centroid).
    PASS if ratio not significantly larger than null (endpoint enveloped).
    FAIL if ratio significantly larger than null (endpoint escapes field).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    coords = np.asarray(coords, dtype=float)
    pool = list(non_endpoint_pool)
    endpoint_set = set(endpoint_indices)

    if endpoint_set & set(pool):
        raise ValueError(
            "non_endpoint_pool must NOT contain endpoint_indices "
            "(v1.0.7: caller is responsible for excluding endpoints)"
        )

    if len(pool) < 3:
        return {
            "verdict": "INSUFFICIENT_NON_ENDPOINT",
            "n_non_endpoint": len(pool),
        }

    non_ep_coords = coords[pool]
    centroid = non_ep_coords.mean(axis=0)

    ep_coords = coords[list(endpoint_set)]
    d_endpoint = float(np.linalg.norm(ep_coords - centroid, axis=1).mean())
    d_non_endpoint = float(np.linalg.norm(non_ep_coords - centroid, axis=1).mean())

    if d_non_endpoint <= 0:
        return {"verdict": "DEGENERATE_CENTROID", "d_endpoint": d_endpoint}

    ratio = d_endpoint / d_non_endpoint

    # null: random "fake endpoint" subsets of same size from non_endpoint_pool;
    # recompute centroid with leave-out logic (mirrors actual computation).
    null_ratios: List[float] = []
    pool_arr = np.asarray(pool)
    for _ in range(n_null):
        choice = rng.choice(pool_arr, size=len(endpoint_indices), replace=False)
        cset = set(choice.tolist())
        non_c = [i for i in pool if i not in cset]
        if len(non_c) < 3:
            continue
        non_c_coords = coords[non_c]
        c_centroid = non_c_coords.mean(axis=0)
        d_c = float(np.linalg.norm(coords[list(cset)] - c_centroid, axis=1).mean())
        d_nc = float(np.linalg.norm(non_c_coords - c_centroid, axis=1).mean())
        if d_nc > 0:
            null_ratios.append(d_c / d_nc)

    if len(null_ratios) < 100:
        return {"verdict": "INSUFFICIENT_NULL", "n_null": len(null_ratios), "ratio": ratio}

    null_arr = np.asarray(null_ratios)
    upper_p = float((np.sum(null_arr >= ratio) + 1) / (len(null_arr) + 1))

    if upper_p > 0.05:
        verdict = "PASS"  # ratio not anomalously high → endpoint enveloped
    else:
        verdict = "FAIL"  # ratio anomalously high → endpoint outside field

    return {
        "verdict": verdict,
        "ratio_endpoint_to_non_endpoint": ratio,
        "ratio_null_median": float(np.median(null_arr)),
        "upper_p": upper_p,
        "n_endpoint": len(endpoint_indices),
        "n_non_endpoint": len(pool),
        "n_null": len(null_arr),
    }


def _mean_pairwise_distance(D: np.ndarray, indices: Sequence[int]) -> float:
    n = len(indices)
    if n < 2:
        return 0.0
    sub = D[np.ix_(indices, indices)]
    iu = np.triu_indices_from(sub, k=1)
    vals = sub[iu]
    finite = vals[np.isfinite(vals)]
    return float(finite.mean()) if len(finite) > 0 else float("inf")


def _matched_random_null(
    members: Sequence[int],
    candidate_pool: Sequence[int],
    shafts: Sequence[Optional[str]],
    participation: np.ndarray,  # kept for back-compat; ignored in v1.0.7
    hfo_rate: Optional[np.ndarray],  # kept for back-compat; ignored in v1.0.7
    n_null: int,
    rng: np.random.Generator,
    D: np.ndarray,
) -> Tuple[List[float], str]:
    """Shaft-stratified random null over the full SEEG implantation pool.

    v1.0.7 (2026-05-22 user catch): previous matched-by-participation /
    matched-by-hfo_rate constraints, combined with a candidate_pool drawn
    from lagPat-selected channels (PR-6 valid_mask), produced a circular
    null: endpoints AND null samples both came from the same pre-filtered
    high-HFO subset. Result: 50% of clusters returned
    INSUFFICIENT_ENVELOPE / INSUFFICIENT_NULL across the n=23 cohort, and
    the testable subset measured "endpoint vs other high-HFO channels"
    rather than "endpoint vs the entire implantation".

    v1.0.7 fix:
      - candidate_pool is now the entire SEEG implantation (caller's
        responsibility — pass all-mm-mapped channels minus endpoint)
      - matching is shaft-only; participation/hfo_rate constraints removed
        (they were nuisance variables that also encoded the endpoint
        signal we are trying to test).

    Args:
        participation, hfo_rate: kept in signature for backward compat
            with v1.0.6 callers; IGNORED in v1.0.7 random sampling.

    Returns:
        (list of null distances, relaxation_tier)
        relaxation_tier ∈ {"shaft_matched", "shaft_infeasible"}
    """
    members_set = set(members)
    pool = [i for i in candidate_pool if i not in members_set]
    n_target = len(members)
    if len(pool) < n_target:
        return [], "insufficient_pool"

    # Target shaft distribution (same-shaft constraint preserved — without it,
    # the null would mostly sample cross-shaft pairs with very large mean
    # distance, biasing actual towards "compact" by default).
    target_shaft_counts: Dict[Optional[str], int] = {}
    for i in members:
        s = shafts[i]
        target_shaft_counts[s] = target_shaft_counts.get(s, 0) + 1

    pool_by_shaft: Dict[Optional[str], List[int]] = {}
    for i in pool:
        pool_by_shaft.setdefault(shafts[i], []).append(i)

    def _try_match_shaft() -> Optional[List[int]]:
        sel: List[int] = []
        for s, cnt in target_shaft_counts.items():
            avail = pool_by_shaft.get(s, [])
            if len(avail) < cnt:
                return None
            pick = rng.choice(avail, size=cnt, replace=False)
            sel.extend(int(x) for x in pick)
        return sel

    null_dists: List[float] = []
    max_attempts = n_null * 3
    attempts = 0
    while len(null_dists) < n_null and attempts < max_attempts:
        attempts += 1
        sel = _try_match_shaft()
        if sel is None:
            return null_dists, "shaft_infeasible"
        null_dists.append(_mean_pairwise_distance(D, sel))

    return null_dists, "shaft_matched"


# =============================================================================
# H2 — Source/sink reversal index
# =============================================================================


def compute_h2_set_reversal(
    S_A: Sequence[int],
    K_A: Sequence[int],
    S_B: Sequence[int],
    K_B: Sequence[int],
    n_null: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """H2 main analysis 1: set-based reversal Jaccard.

    R_set = J_swap − J_same
      J_swap = mean(Jaccard(S_A, K_B), Jaccard(K_A, S_B))   # reversal pairs
      J_same = mean(Jaccard(S_A, K_A), Jaccard(K_B, S_B))   # same-template ends

    Null: shuffle role labels within (S_A ∪ K_A ∪ S_B ∪ K_B) pool.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    sa, ka, sb, kb = set(S_A), set(K_A), set(S_B), set(K_B)
    sizes = [len(sa), len(ka), len(sb), len(kb)]
    if min(sizes) == 0:
        return {"verdict": "EMPTY_SET", "sizes": sizes}

    j_swap = (_jaccard_set(sa, kb) + _jaccard_set(ka, sb)) / 2.0
    j_same = (_jaccard_set(sa, ka) + _jaccard_set(kb, sb)) / 2.0
    r_set = j_swap - j_same

    # null: shuffle role labels in union pool
    union = list(sa | ka | sb | kb)
    if len(union) < sum(sizes):
        # roles overlap (e.g., source in A ∩ sink in B by chance, allowed under
        # synthetic perfect-swap). Use sampling with replacement for null.
        sample_with_replacement = True
    else:
        sample_with_replacement = False

    null_r: List[float] = []
    for _ in range(n_null):
        if sample_with_replacement:
            sets = []
            for sz in sizes:
                pick = rng.choice(union, size=sz, replace=True)
                sets.append(set(pick.tolist()))
        else:
            shuffled = rng.permutation(union)
            cs = np.cumsum([0] + sizes)
            sets = [set(shuffled[cs[i] : cs[i + 1]].tolist()) for i in range(4)]
        nsa, nka, nsb, nkb = sets
        j_swap_n = (_jaccard_set(nsa, nkb) + _jaccard_set(nka, nsb)) / 2.0
        j_same_n = (_jaccard_set(nsa, nka) + _jaccard_set(nkb, nsb)) / 2.0
        null_r.append(j_swap_n - j_same_n)

    null_arr = np.asarray(null_r)
    upper_p = float((np.sum(null_arr >= r_set) + 1) / (len(null_arr) + 1))

    if upper_p < 0.05:
        verdict = "PASS"
    elif r_set < -0.1:
        verdict = "FAIL"
    else:
        verdict = "NULL"

    return {
        "verdict": verdict,
        "J_swap": float(j_swap),
        "J_same": float(j_same),
        "R_set": float(r_set),
        "R_set_null_median": float(np.median(null_arr)),
        "upper_p": upper_p,
        "n_null": len(null_arr),
        "null_sampling": "with_replacement" if sample_with_replacement else "permutation",
    }


def compute_h2_spatial_reversal(
    S_A: Sequence[int],
    K_A: Sequence[int],
    S_B: Sequence[int],
    K_B: Sequence[int],
    coords: np.ndarray,
    n_null: int = 1000,
    rng: Optional[np.random.Generator] = None,
    coord_units: Optional[str] = None,
) -> Dict:
    """H2 main analysis 2: spatial reversal index (centroid distance based).

    d_swap = d(c(S_A), c(K_B)) + d(c(K_A), c(S_B))   # reversal pair distances
    d_same = d(c(S_A), c(K_A)) + d(c(K_B), c(S_B))   # same-template pair distances
    R_spatial = d_same / (d_swap + d_same)

    Expected: R_spatial > 0.5 under reversal hypothesis (swap pairs closer than same).

    coord_units: when provided, must be 'mm' (v1.0.5 contract; reject voxel coords).
    """
    _assert_coords_are_mm(coord_units, "compute_h2_spatial_reversal")
    if rng is None:
        rng = np.random.default_rng(0)

    coords = np.asarray(coords, dtype=float)
    sa, ka, sb, kb = list(S_A), list(K_A), list(S_B), list(K_B)
    sizes = [len(sa), len(ka), len(sb), len(kb)]
    if min(sizes) == 0:
        return {"verdict": "EMPTY_SET", "sizes": sizes}

    def _cent(indices: Sequence[int]) -> np.ndarray:
        return coords[list(indices)].mean(axis=0)

    c_sa, c_ka, c_sb, c_kb = _cent(sa), _cent(ka), _cent(sb), _cent(kb)

    d_swap = float(np.linalg.norm(c_sa - c_kb) + np.linalg.norm(c_ka - c_sb))
    d_same = float(np.linalg.norm(c_sa - c_ka) + np.linalg.norm(c_kb - c_sb))
    if d_swap + d_same == 0:
        return {"verdict": "DEGENERATE", "d_swap": d_swap, "d_same": d_same}
    r_spatial = d_same / (d_swap + d_same)

    union = list(set(sa) | set(ka) | set(sb) | set(kb))
    sample_with_replacement = len(union) < sum(sizes)

    null_r: List[float] = []
    for _ in range(n_null):
        if sample_with_replacement:
            sets = [rng.choice(union, size=sz, replace=True).tolist() for sz in sizes]
        else:
            shuffled = rng.permutation(union)
            cs = np.cumsum([0] + sizes)
            sets = [shuffled[cs[i] : cs[i + 1]].tolist() for i in range(4)]
        if any(len(s) == 0 for s in sets):
            continue
        c0, c1, c2, c3 = [_cent(s) for s in sets]
        dsw_n = float(np.linalg.norm(c0 - c3) + np.linalg.norm(c1 - c2))
        dsa_n = float(np.linalg.norm(c0 - c1) + np.linalg.norm(c2 - c3))
        if dsw_n + dsa_n > 0:
            null_r.append(dsa_n / (dsw_n + dsa_n))

    if len(null_r) < 100:
        return {
            "verdict": "INSUFFICIENT_NULL",
            "n_null_obtained": len(null_r),
            "R_spatial": float(r_spatial),
        }

    null_arr = np.asarray(null_r)
    upper_p = float((np.sum(null_arr >= r_spatial) + 1) / (len(null_arr) + 1))

    if upper_p < 0.05:
        verdict = "PASS"
    elif r_spatial < 0.45:
        verdict = "FAIL"
    else:
        verdict = "NULL"

    return {
        "verdict": verdict,
        "d_swap": d_swap,
        "d_same": d_same,
        "R_spatial": float(r_spatial),
        "R_spatial_null_median": float(np.median(null_arr)),
        "upper_p": upper_p,
        "n_null_used": len(null_arr),
        "null_sampling": "with_replacement" if sample_with_replacement else "permutation",
    }


def _jaccard_set(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


# =============================================================================
# Integrated H1 (a + b + c)
# =============================================================================


def compute_h1_full(
    source_indices: Sequence[int],
    sink_indices: Sequence[int],
    valid_indices: Sequence[int],
    coords: Optional[np.ndarray],
    channel_names: Sequence[str],
    participation: np.ndarray,
    hfo_rate: Optional[np.ndarray] = None,
    distance_metric: Literal["euclidean", "shaft_ordinal"] = "euclidean",
    n_null: int = 1000,
    rng: Optional[np.random.Generator] = None,
    coord_units: Optional[str] = None,
) -> Dict:
    """H1 integrated runner (v1.0.3 three-layer, 2026-05-21).

    Reports three INDEPENDENT layers (per plan archive doc §3.2):

      1. descriptive: source/sink absolute distances + diameter + Rg
                      (no verdict, just numbers)

      2. strict_source / strict_sink: matched-null compactness check
                      (asks: extra compactness beyond shaft geometry?)
                      verdicts: PASS / NULL / FAIL_DIFFUSE

      3. envelope: endpoint NOT outside pathological participation field
                   (necessary condition)
                   verdicts: PASS / FAIL only

    Overall verdict rule:
      - envelope == FAIL          → overall FAIL (necessary condition overrides)
      - strict both PASS          → overall PASS
      - one strict PASS / one NULL → overall partial_PASS
      - both strict NULL          → overall NULL

    User audit 2026-05-21: NULL on strict layer does NOT mean "endpoints not
    compact"; it means "no extra compactness beyond shaft geometry". Report
    all three layers in cohort summary; do NOT collapse.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # --- Descriptive layer (numbers only, no verdict) ---
    desc_source = compute_h1_descriptive(
        members=source_indices,
        coords=coords,
        channel_names=channel_names,
        distance_metric=distance_metric,
        coord_units=coord_units,
    )
    desc_sink = compute_h1_descriptive(
        members=sink_indices,
        coords=coords,
        channel_names=channel_names,
        distance_metric=distance_metric,
        coord_units=coord_units,
    )

    # --- Strict layer (matched-null compactness) ---
    strict_source = compute_h1_compactness(
        members=source_indices,
        candidate_pool=valid_indices,
        coords=coords,
        channel_names=channel_names,
        participation=participation,
        hfo_rate=hfo_rate,
        distance_metric=distance_metric,
        n_null=n_null,
        rng=rng,
        coord_units=coord_units,
    )
    strict_sink = compute_h1_compactness(
        members=sink_indices,
        candidate_pool=valid_indices,
        coords=coords,
        channel_names=channel_names,
        participation=participation,
        hfo_rate=hfo_rate,
        distance_metric=distance_metric,
        n_null=n_null,
        rng=rng,
        coord_units=coord_units,
    )

    # --- Envelope layer (necessary condition) ---
    envelope: Dict
    if coords is None:
        envelope = {
            "verdict": "SKIPPED_NO_COORDS",
            "reason": "H1 envelope requires 3D coords (Topic 0 §3.2 coord loader)",
        }
    else:
        endpoint = list(set(source_indices) | set(sink_indices))
        # v1.0.7: valid_indices is already non_endpoint pool (mapped SEEG - endpoint)
        non_endpoint_pool = [i for i in valid_indices if i not in set(endpoint)]
        envelope = compute_h1c_envelope(
            endpoint_indices=endpoint,
            non_endpoint_pool=non_endpoint_pool,
            coords=coords,
            n_null=n_null,
            rng=rng,
        )

    # --- Overall verdict (v1.0.6 fix, 2026-05-21 advisor catch) ---
    #
    # Three independent dimensions get mapped to one overall label:
    #
    #   strict source state ∈ {COMPACT, NULL, DIFFUSE, UNTESTABLE}
    #   strict sink   state ∈ {COMPACT, NULL, DIFFUSE, UNTESTABLE}
    #   envelope      state ∈ {PASS, FAIL, GATED, INDETERMINATE}
    #
    # v1.0.5 conflated {PASS, INSUFFICIENT_NULL} as `partial_PASS` — readers see
    # "one side scientifically passed, the other side tested but didn't"
    # which is wrong (the other side was UNTESTABLE not NULL). v1.0.5 also
    # silently collapsed any FAIL_DIFFUSE result into `NULL`, erasing the
    # anti-compact signal.
    #
    # v1.0.6 enumerates each combination explicitly. UNTESTABLE never gets
    # smuggled into a binary verdict.
    env_v = envelope.get("verdict")
    src_v = strict_source.get("verdict")
    snk_v = strict_sink.get("verdict")

    def _classify_strict(v: Optional[str]) -> str:
        if v == "PASS":
            return "COMPACT"
        if v == "NULL":
            return "NULL"
        if v == "FAIL_DIFFUSE":
            return "DIFFUSE"
        # INSUFFICIENT_CHANNELS, INSUFFICIENT_NULL, etc. — couldn't decide
        return "UNTESTABLE"

    src_state = _classify_strict(src_v)
    snk_state = _classify_strict(snk_v)

    # Envelope classification — only PASS/FAIL run as "tested";
    # SKIPPED_NO_COORDS is distinguishable from other indeterminate causes
    # because the user can tell whether to wait for coords or fix the data.
    if env_v == "PASS":
        env_state = "PASS"
    elif env_v == "FAIL":
        env_state = "FAIL"
    elif env_v == "SKIPPED_NO_COORDS":
        env_state = "GATED"
    else:
        env_state = "INDETERMINATE"

    # Envelope-level short-circuit
    if env_state == "GATED":
        overall = "INCOMPLETE_GATED_ON_COORDS"
    elif env_state == "INDETERMINATE":
        overall = "INCONCLUSIVE_ENVELOPE_INDETERMINATE"
    elif env_state == "FAIL":
        # necessary condition failed → overrides strict layer
        overall = "FAIL"
    else:
        # envelope PASS — now combine strict source × strict sink
        # DIFFUSE on either side = strict-layer anti-compact signal — surface it
        if src_state == "DIFFUSE" or snk_state == "DIFFUSE":
            overall = "FAIL_DIFFUSE"
        elif src_state == "COMPACT" and snk_state == "COMPACT":
            overall = "PASS"
        elif src_state == "COMPACT" and snk_state == "NULL":
            overall = "partial_PASS"
        elif src_state == "NULL" and snk_state == "COMPACT":
            overall = "partial_PASS"
        elif src_state == "COMPACT" and snk_state == "UNTESTABLE":
            overall = "PASS_one_side_untestable"
        elif src_state == "UNTESTABLE" and snk_state == "COMPACT":
            overall = "PASS_one_side_untestable"
        elif src_state == "NULL" and snk_state == "NULL":
            overall = "NULL"
        elif src_state == "NULL" and snk_state == "UNTESTABLE":
            overall = "NULL_one_side_untestable"
        elif src_state == "UNTESTABLE" and snk_state == "NULL":
            overall = "NULL_one_side_untestable"
        else:  # both UNTESTABLE
            overall = "UNTESTABLE_BOTH_SIDES"

    return {
        "h1_overall_verdict": overall,
        "descriptive": {
            "source": desc_source,
            "sink": desc_sink,
        },
        "strict": {
            "source": strict_source,
            "sink": strict_sink,
        },
        "envelope": envelope,
        # NOTE: h1a/h1b/h1c backward-compat keys REMOVED 2026-05-21 (user audit:
        # "保留旧键会诱导后续代码继续把 descriptive / strict / envelope 揉回一个旧 H1 叙事").
        # Branch not yet released; no migration burden.
    }
