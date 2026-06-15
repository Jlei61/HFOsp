"""Topic 5 C-line — subtype x activation-direction (pure functions, TDD).

Two layers, kept strictly separate (review Point 1):
  - AXIS layer (mode="axis"): u=(cos2θ, sin2θ); θ and θ+π are the SAME axis. This is the
    only layer eligible to explain the sign-free A-line split-half instability.
  - POLARITY layer (mode="pol"): v=(cosθ, sinθ); θ vs θ+π differ. Descriptive direction
    read-out only; CANNOT explain the A-line (A-line is sign-free).

Alignment is by seizure_id (z-ER seizure_ids_kept ↔ T0 audit seizure_id), never by position.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.topic5_axis_direction import (
    axial_mean,
    axial_resultant_length,
    circular_mean,
    resultant_length,
)

PI = np.pi
TWO_PI = 2.0 * np.pi


# --- angular distances --------------------------------------------------------
def axial_distance(a: float, b: float) -> float:
    """Axial angular distance ∈ [0, π/2] (θ and θ+π collapse to the same axis)."""
    delta = abs((a - b) % PI)
    return float(min(delta, PI - delta))


def circular_distance(a: float, b: float) -> float:
    """Directional angular distance ∈ [0, π] (full-circle, polarity-aware)."""
    delta = abs((a - b) % TWO_PI)
    return float(min(delta, TWO_PI - delta))


# --- per-subtype separation statistic -----------------------------------------
def _subtype_centers(angles: np.ndarray, labels: np.ndarray, mode: str):
    """Return (ordered subtype list, center per subtype, size per subtype)."""
    center_fn = axial_mean if mode == "axis" else circular_mean
    subs = sorted(set(int(x) for x in labels))
    centers, sizes = [], []
    for s in subs:
        a = angles[labels == s]
        centers.append(center_fn(a))
        sizes.append(int(a.size))
    return subs, centers, sizes


def subtype_separation_stat(angles, labels, mode: str = "axis") -> float:
    """Between-subtype angular separation.

    mode="axis" → size-weighted mean pairwise `axial_distance` of subtype axial means (∈[0,π/2]).
    mode="pol"  → size-weighted mean pairwise `circular_distance` of subtype circular means (∈[0,π]).
    For k=2 this reduces to the single pairwise distance.
    """
    if mode not in ("axis", "pol"):
        raise ValueError(f"mode must be 'axis' or 'pol', got {mode!r}")
    angles = np.asarray(angles, float)
    labels = np.asarray(labels)
    dist_fn = axial_distance if mode == "axis" else circular_distance
    subs, centers, sizes = _subtype_centers(angles, labels, mode)
    if len(subs) < 2:
        return float("nan")
    num = 0.0
    den = 0.0
    for i in range(len(subs)):
        for j in range(i + 1, len(subs)):
            ci, cj = centers[i], centers[j]
            if not (np.isfinite(ci) and np.isfinite(cj)):
                continue
            w = sizes[i] * sizes[j]
            num += w * dist_fn(ci, cj)
            den += w
    return float(num / den) if den > 0 else float("nan")


# --- within-subject permutation test ------------------------------------------
def within_subject_perm_p(angles, labels, mode: str = "axis", B: int = 2000,
                          rng: Optional[np.random.Generator] = None,
                          min_subtype_size: int = 3) -> Dict[str, Any]:
    """Permute subtype labels within the subject; p = (1+#{T_perm≥T_obs})/(B+1).

    Subtypes with fewer than `min_subtype_size` aligned seizures are DROPPED first (their
    per-subtype centre is unreliable); the test then runs on the kept subtypes. Bad-data
    gate: fewer than 2 kept subtypes → eligibility="insufficient_subtypes", p=None.
    """
    angles = np.asarray(angles, float)
    labels = np.asarray(labels)
    if rng is None:
        rng = np.random.default_rng()
    subs, _, sizes = _subtype_centers(angles, labels, mode)
    size_of = dict(zip(subs, sizes))
    kept = [s for s in subs if size_of[s] >= min_subtype_size]
    dropped = [s for s in subs if size_of[s] < min_subtype_size]
    base: Dict[str, Any] = {"T_obs": None, "p": None, "B": B,
                            "subtype_sizes": size_of, "dropped_subtypes": dropped,
                            "k": len(kept)}
    if len(kept) < 2:
        base["n"] = int(sum(size_of[s] for s in kept))
        base["eligibility"] = "insufficient_subtypes"
        return base
    keep_mask = np.isin(labels, kept)
    ang_k = angles[keep_mask]
    lab_k = labels[keep_mask]
    base["n"] = int(ang_k.size)
    t_obs = subtype_separation_stat(ang_k, lab_k, mode=mode)
    ge = 0
    for _ in range(B):
        perm = rng.permutation(lab_k)
        if subtype_separation_stat(ang_k, perm, mode=mode) >= t_obs - 1e-12:
            ge += 1
    base["T_obs"] = float(t_obs)
    base["p"] = float((1 + ge) / (B + 1))
    base["eligibility"] = "ok"
    return base


# --- direction clustering (Step-0 gate) ---------------------------------------
def direction_clustering(angles, r_min: float = 0.5) -> Dict[str, Any]:
    """Axial + directional concentration of one subject's per-seizure angles.

    `clustered` = R_axial >= r_min (a coarse 'is there an axis at all' check before the
    subtype test; the runner additionally compares against a uniform permutation null).
    """
    angles = np.asarray(angles, float)
    r_axial = axial_resultant_length(angles)
    r_dir = resultant_length(angles)
    clustered = bool(np.isfinite(r_axial) and r_axial >= r_min)
    return {"R_axial": float(r_axial), "R_dir": float(r_dir), "clustered": clustered,
            "n": int(np.isfinite(angles).sum())}


# --- geometry quality ---------------------------------------------------------
def coord_aspect_ratio(x, y) -> float:
    """Second/first singular value of the centered (n×2) contact-coordinate cloud.

    ~0 = near-1D (collinear; gradient angle fragile); ~1 = isotropic (well-spread 2D).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    M = np.column_stack([x[ok] - x[ok].mean(), y[ok] - y[ok].mean()])
    s = np.linalg.svd(M, compute_uv=False)
    if s[0] < 1e-12:
        return 0.0
    return float(s[1] / s[0])


# --- alignment (join on seizure_id; fail-loud on namespace mismatch) ----------
def align_subtype_to_direction(
    seizure_id_to_subtype: Dict[str, int],
    idx_to_seizure_id: Dict[int, str],
    eligible_idxs: Sequence[int],
    angles_by_idx: Dict[int, float],
) -> List[Dict[str, Any]]:
    """Inner-join T0 eligible seizures to z-ER subtypes BY seizure_id.

    Keeps a seizure iff: idx in eligible_idxs, idx maps to a seizure_id, that seizure_id has
    a subtype label != -1, and its angle is finite. Drops everything else silently.

    Raises ValueError when both namespaces are non-empty but share ZERO seizure_id — this
    is the position-blind-match trap (§6.2); the caller must check the audit CSV / use the
    onset fallback rather than aligning by list position.
    """
    eligible_ids = {idx_to_seizure_id[i] for i in eligible_idxs if i in idx_to_seizure_id}
    zer_ids = set(seizure_id_to_subtype.keys())
    if eligible_ids and zer_ids and not (eligible_ids & zer_ids):
        raise ValueError(
            "zero seizure_id overlap between z-ER (e.g. "
            f"{sorted(zer_ids)[:3]}) and T0-eligible (e.g. {sorted(eligible_ids)[:3]}) — "
            "namespace mismatch; do NOT align by position. Check t0_eligibility_audit.csv "
            "or use the onset fallback."
        )
    out: List[Dict[str, Any]] = []
    for idx in eligible_idxs:
        sid = idx_to_seizure_id.get(idx)
        if sid is None or sid not in seizure_id_to_subtype:
            continue
        subtype = int(seizure_id_to_subtype[sid])
        if subtype == -1:
            continue
        theta = angles_by_idx.get(idx, float("nan"))
        if not np.isfinite(theta):
            continue
        out.append({"seizure_id": sid, "idx": int(idx), "theta": float(theta), "subtype": subtype})
    return out


# --- odd/even subtype imbalance (A-line connection) ---------------------------
def oddeven_subtype_imbalance(even_idxs, odd_idxs, idx_to_subtype: Dict[int, int]) -> float:
    """Total-variation distance ∈ [0,1] between the subtype distributions of the labeled
    seizures falling in the even vs odd half of the A-line parity split.

    NaN if either half has no labeled seizure.
    """
    def dist(idxs):
        labs = [idx_to_subtype[i] for i in idxs if i in idx_to_subtype]
        return labs

    e_labs = dist(even_idxs)
    o_labs = dist(odd_idxs)
    if not e_labs or not o_labs:
        return float("nan")
    subs = sorted(set(e_labs) | set(o_labs))
    e_frac = np.array([e_labs.count(s) for s in subs], float) / len(e_labs)
    o_frac = np.array([o_labs.count(s) for s in subs], float) / len(o_labs)
    return float(0.5 * np.abs(e_frac - o_frac).sum())
