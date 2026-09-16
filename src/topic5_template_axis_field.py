"""Pure helpers for Topic 5 single-template gradient axes and shared-plane fields.

The module deliberately separates the interictal-only construction (axis pair,
collinearity, shared bisector) from the downstream ictal field readout.  It has no
filesystem I/O; the cohort runner owns joins, caching and serialization.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from src.dab_gradient_axis import RCOND, compute_dab_gradient_axis
from src.topic5_contact_similarity import kernel_smooth_at_contacts, median_nn_spacing


TEMPLATE_AXIS_DEFINITION = "template_propagation_axis_v2"
TEMPLATE_AXIS_DIRECTION = "positive_early_to_late"
INTERICTAL_FIELD_CONTRACT = "topic5_interictal_template_fields_v1"
INTERICTAL_FIELD_FINGERPRINT_ALGORITHM = "sha256_v1p1_nonfinite_canonical"


def z_earliness(rank: Sequence[float]) -> np.ndarray:
    """Return earliness = -z(rank); fail closed to zeros for a constant vector."""
    x = np.asarray(rank, float)
    out = np.full(x.shape, np.nan, float)
    ok = np.isfinite(x)
    if int(ok.sum()) == 0:
        return out
    sd = float(np.std(x[ok]))
    out[ok] = -(x[ok] - float(np.mean(x[ok]))) / sd if sd > 1e-12 else 0.0
    return out


def _as_template_propagation_axis(earliness_fit: Mapping[str, object]) -> Dict[str, object]:
    """Convert an earliness-gradient fit into an early-to-late propagation axis.

    ``compute_dab_gradient_axis`` deliberately orients ``u`` toward increasing input.
    For ``e_T=-z(rank_T)``, increasing input means later-to-earlier.  A propagation
    direction has the opposite sign: early-to-late.  The raw gradient is retained
    under explicit ``earliness_gradient_*`` keys so the two meanings cannot drift
    back into one ambiguous ``u`` field.
    """
    raw = dict(earliness_fit)
    out = dict(raw)
    out["axis_definition"] = TEMPLATE_AXIS_DEFINITION
    out["direction_convention"] = TEMPLATE_AXIS_DIRECTION
    out["scalar_definition"] = "earliness=-z(rank); rank increases early_to_late"
    out["propagation_relation"] = "u=-normalize(gradient(earliness))"
    out["earliness_gradient_axis_definition"] = raw.get("axis_definition")

    if raw.get("status") != "ok":
        return out

    e_u = np.asarray(raw["u"], float)
    e_beta = np.asarray(raw["beta"], float)
    e_along = np.asarray(raw["along"], float)
    out.update({
        "u": -e_u,
        "along": -e_along,
        "propagation_vector": -e_beta,
        "earliness_gradient_u": e_u,
        "earliness_gradient_beta": e_beta,
        "earliness_gradient_along": e_along,
        "beta_role": "removed_from_top_level; use earliness_gradient_beta or propagation_vector",
        "mu_early": raw.get("mu_A"),
        "mu_late": raw.get("mu_B"),
        "p_early": raw.get("p_A"),
        "p_late": raw.get("p_B"),
        "pole_early_idx": raw.get("pole_A_idx"),
        "pole_late_idx": raw.get("pole_B_idx"),
    })
    # These D_AB-specific names would invert the biological meaning for a
    # single-template earliness field.  Keep only the explicit early/late aliases.
    for key in ("beta", "mu_A", "mu_B", "p_A", "p_B", "pole_A_idx", "pole_B_idx"):
        out.pop(key, None)
    return out


def compute_template_propagation_axis(
    coords: np.ndarray,
    earliness: np.ndarray,
    shafts: Sequence[str],
    *,
    n_boot: int = 200,
    seed: int = 0,
) -> Dict[str, object]:
    """Fit one template axis whose positive direction is early-to-late."""
    raw = compute_dab_gradient_axis(coords, earliness, shafts, n_boot=n_boot, seed=seed)
    return _as_template_propagation_axis(raw)


def axis_passes_qc(axis: Mapping[str, object], *, boot_min: float = 0.80,
                   loso_min: float = 0.50) -> bool:
    """Frozen formal axis QC from the design contract."""
    if axis.get("status") != "ok":
        return False
    vals = (axis.get("bootstrap_cosine"), axis.get("loso_cosine"))
    if not all(v is not None and np.isfinite(float(v)) for v in vals):
        return False
    return bool(
        int(axis.get("n", 0)) >= 6
        and int(axis.get("n_shafts", 0) or 0) >= 2
        and int(axis.get("effective_rank", 0)) >= 2
        and float(axis["bootstrap_cosine"]) >= boot_min
        and float(axis["loso_cosine"]) >= loso_min
    )


def assess_axis_direction_validity(
    axis: Mapping[str, object], *, boot_min: float = 0.80, loso_min: float = 0.50
) -> Dict[str, object]:
    """Return explicit, reusable direction-validity tiers for one template axis.

    Estimability, two-dimensional sampling geometry, and resampling stability are
    intentionally separate.  A failed strict-stability gate never erases an
    otherwise estimable early-to-late direction.
    """
    estimable = bool(axis.get("status") == "ok")
    n = int(axis.get("n", 0) or 0)
    n_shafts = int(axis.get("n_shafts", 0) or 0)
    effective_rank = int(axis.get("effective_rank", 0) or 0)
    boot = axis.get("bootstrap_cosine")
    loso = axis.get("loso_cosine")
    boot_ok = bool(boot is not None and np.isfinite(float(boot)) and float(boot) >= boot_min)
    loso_ok = bool(loso is not None and np.isfinite(float(loso)) and float(loso) >= loso_min)
    geometry_2d = bool(estimable and n_shafts >= 2 and effective_rank >= 2)
    strict = bool(
        estimable and n >= 6 and geometry_2d and boot_ok and loso_ok
    )
    reasons = []
    if not estimable:
        reasons.append(f"axis_status:{axis.get('status', 'missing')}")
    if estimable and n < 6:
        reasons.append("fewer_than_6_contacts")
    if estimable and n_shafts < 2:
        reasons.append("single_shaft_geometry")
    if estimable and effective_rank < 2:
        reasons.append("effective_rank_below_2")
    if estimable and not boot_ok:
        reasons.append("contact_bootstrap_below_threshold")
    if estimable and not loso_ok:
        reasons.append("leave_one_shaft_out_below_threshold_or_unavailable")
    return {
        "axis_definition": TEMPLATE_AXIS_DEFINITION,
        "direction_convention": TEMPLATE_AXIS_DIRECTION,
        "estimable": estimable,
        "geometry_2d_supported": geometry_2d,
        "contact_bootstrap_stable": boot_ok,
        "leave_one_shaft_out_stable": loso_ok,
        "strict_stability_pass": strict,
        "thresholds": {
            "minimum_contacts": 6,
            "minimum_shafts_for_2d": 2,
            "minimum_effective_rank_for_2d": 2,
            "bootstrap_cosine_min": float(boot_min),
            "loso_cosine_min": float(loso_min),
        },
        "reason_codes": reasons,
    }


def classify_axis_pair(cosine: float, *, line_threshold: float = 0.50) -> Dict[str, object]:
    """Separate line collinearity (|cos|) from same/reversed direction (sign)."""
    c = float(np.clip(cosine, -1.0, 1.0))
    abs_c = abs(c)
    collinear = bool(abs_c >= line_threshold)
    relation = "same" if collinear and c >= 0 else "reversed" if collinear else "different"
    return {
        "cosine": c,
        "abs_cosine": abs_c,
        "line_angle_deg": float(np.degrees(np.arccos(abs_c))),
        "collinear": collinear,
        "relation": relation,
        "line_threshold": float(line_threshold),
    }


def shared_bisector(u_a: Sequence[float], u_b: Sequence[float]) -> Dict[str, object]:
    """Align B to A as an unoriented line, then return their unit angular bisector."""
    a = np.asarray(u_a, float)
    b = np.asarray(u_b, float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    cosine = float(np.clip(a @ b, -1.0, 1.0))
    sign = 1.0 if cosine >= 0 else -1.0
    b_aligned = sign * b
    s = a + b_aligned
    norm = float(np.linalg.norm(s))
    if not np.isfinite(norm) or norm < 1e-12:
        return {"status": "degenerate_bisector", "u": None, "cosine": cosine}
    return {
        "status": "ok",
        "u": s / norm,
        "u_b_aligned": b_aligned,
        "b_alignment_sign": int(sign),
        "cosine": cosine,
    }


def _fit_propagation_axis(coords: np.ndarray, earliness: np.ndarray) -> Optional[np.ndarray]:
    """Fast bootstrap-only early-to-late counterpart of the earliness gradient."""
    x = np.asarray(coords, float)
    v = np.asarray(earliness, float)
    xc = x - x.mean(0)
    yc = v - v.mean()
    beta, *_ = np.linalg.lstsq(xc, yc, rcond=RCOND)
    bn = float(np.linalg.norm(beta))
    if not np.isfinite(bn) or bn < 1e-9:
        return None
    u = beta / bn
    along = xc @ u
    if np.std(along) < 1e-12 or np.corrcoef(along, v)[0, 1] < 0:
        u = -u
    return -u


def paired_axis_bootstrap(coords: np.ndarray, e_a: np.ndarray, e_b: np.ndarray,
                          full_u_a: np.ndarray, full_u_b: np.ndarray, *,
                          n_boot: int = 500, seed: int = 0,
                          line_threshold: float = 0.50) -> Dict[str, object]:
    """Paired contact bootstrap for A/B line and direction-relation stability."""
    x = np.asarray(coords, float)
    a = np.asarray(e_a, float)
    b = np.asarray(e_b, float)
    rng = np.random.default_rng(seed)
    full_cos = float(np.asarray(full_u_a) @ np.asarray(full_u_b))
    cosines = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(x), len(x))
        ua = _fit_propagation_axis(x[idx], a[idx])
        ub = _fit_propagation_axis(x[idx], b[idx])
        if ua is not None and ub is not None:
            cosines.append(float(np.clip(ua @ ub, -1.0, 1.0)))
    if not cosines:
        return {"n_valid": 0, "p_collinear": np.nan, "p_sign_stable": np.nan,
                "cosine_q": {"p2_5": np.nan, "p50": np.nan, "p97_5": np.nan},
                "robust_collinear": False}
    c = np.asarray(cosines, float)
    sign_stable = np.sign(c) == np.sign(full_cos)
    p_col = float(np.mean(np.abs(c) >= line_threshold))
    p_sign = float(np.mean(sign_stable))
    return {
        "n_valid": int(len(c)),
        "p_collinear": p_col,
        "p_sign_stable": p_sign,
        "cosine_q": {"p2_5": float(np.percentile(c, 2.5)),
                     "p50": float(np.percentile(c, 50)),
                     "p97_5": float(np.percentile(c, 97.5))},
        "robust_collinear": bool(p_col >= 0.80 and p_sign >= 0.80),
    }


def compute_template_axis_pair(coords: np.ndarray, rank_a: np.ndarray, rank_b: np.ndarray,
                               shafts: Sequence[str], *, n_axis_boot: int = 200,
                               n_pair_boot: int = 500, seed: int = 0,
                               line_threshold: float = 0.50) -> Dict[str, object]:
    """Fit A/B axes, freeze QC, classify their line relation and build a bisector."""
    x = np.asarray(coords, float)
    ea, eb = z_earliness(rank_a), z_earliness(rank_b)
    valid = np.isfinite(x).all(1) & np.isfinite(ea) & np.isfinite(eb)
    x, ea, eb = x[valid], ea[valid], eb[valid]
    sh = np.asarray(shafts, object)[valid]
    ax_a = compute_template_propagation_axis(x, ea, sh, n_boot=n_axis_boot, seed=seed)
    ax_b = compute_template_propagation_axis(x, eb, sh, n_boot=n_axis_boot, seed=seed + 1)
    out: Dict[str, object] = {"axis_a": ax_a, "axis_b": ax_b, "n_joint": int(len(x))}
    if ax_a.get("status") != "ok" or ax_b.get("status") != "ok":
        out.update({"status": "axis_undefined", "axis_pair_estimable": False,
                    "geometry_2d_supported": False, "strict_stability_pass": False,
                    "axis_pair_qc_pass": False})
        return out
    ua, ub = np.asarray(ax_a["u"], float), np.asarray(ax_b["u"], float)
    relation = classify_axis_pair(float(ua @ ub), line_threshold=line_threshold)
    relation["direction_convention"] = TEMPLATE_AXIS_DIRECTION
    relation["signed_cosine_meaning"] = "positive=same early_to_late direction; negative=reversed"
    boot = paired_axis_bootstrap(x, ea, eb, ua, ub, n_boot=n_pair_boot, seed=seed + 2,
                                 line_threshold=line_threshold)
    bis = shared_bisector(ua, ub) if relation["collinear"] else {"status": "not_collinear", "u": None}
    if bis.get("status") == "ok":
        bis["direction_convention"] = TEMPLATE_AXIS_DIRECTION
        bis["orientation_reference"] = "axis_a; axis_b sign-aligned as an unoriented line"
    geometry_2d = bool(
        int(ax_a.get("n_shafts", 0) or 0) >= 2
        and int(ax_b.get("n_shafts", 0) or 0) >= 2
        and int(ax_a.get("effective_rank", 0)) >= 2
        and int(ax_b.get("effective_rank", 0)) >= 2
    )
    strict_stability = bool(axis_passes_qc(ax_a) and axis_passes_qc(ax_b))
    out.update({
        "status": "ok",
        "axis_pair_estimable": True,
        "geometry_2d_supported": geometry_2d,
        "strict_stability_pass": strict_stability,
        # Backward-compatible alias.  This never means that failed subjects lack axes.
        "axis_pair_qc_pass": strict_stability,
        "relation": relation,
        "pair_bootstrap": boot,
        "shared_axis": bis,
    })
    return out


def make_normalized_plane(coords: np.ndarray, u: Sequence[float], *,
                          origin: Optional[Sequence[float]] = None) -> Dict[str, object]:
    """Project contacts to an axis-aligned plane with one shared robust length scale."""
    x = np.asarray(coords, float)
    axis = np.asarray(u, float)
    axis = axis / np.linalg.norm(axis)
    xbar = x.mean(0) if origin is None else np.asarray(origin, float)
    rel = x - xbar
    along = rel @ axis
    resid = rel - np.outer(along, axis)
    try:
        w = np.linalg.svd(resid, full_matrices=False)[2][0]
    except np.linalg.LinAlgError:
        return {"status": "svd_failed"}
    trans = resid @ w
    lo, hi = np.percentile(along, [2.5, 97.5])
    scale = float(hi - lo)
    if not np.isfinite(scale) or scale < 1e-9:
        return {"status": "degenerate_axis_span"}
    pts = np.column_stack((along / scale, trans / scale))
    sigma = float(median_nn_spacing(pts))
    if not np.isfinite(sigma) or sigma <= 0:
        return {"status": "degenerate_spacing"}
    return {"status": "ok", "points": pts, "u": axis, "w": w, "origin": xbar,
            "scale_mm": scale, "sigma": sigma}


def make_field_scorer(template_values: Sequence[float], plane_points: np.ndarray,
                      support: Sequence[float], sigma: float) -> Dict[str, object]:
    """Precompute one template field on one fixed contact plane."""
    vals = np.asarray(template_values, float)
    pts = np.asarray(plane_points, float)
    sup = np.asarray(support, float)
    sig2 = 2.0 * float(sigma) ** 2
    d2_id = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
    mirror_eval = pts.copy()
    mirror_eval[:, 1] *= -1
    d2_mirror = ((mirror_eval[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
    weight_id = np.exp(-d2_id / sig2) * sup[None, :]
    weight_mirror = np.exp(-d2_mirror / sig2) * sup[None, :]
    field = _smooth_from_weights(vals, weight_id)
    return {"template_field": field, "points": pts, "support": sup, "sigma": float(sigma),
            "weight_id": weight_id, "weight_mirror": weight_mirror}


def build_interictal_template_field_record(
    *,
    subject_id: str,
    dataset: str,
    subject: str,
    stable_k: int,
    names: Sequence[str],
    coords: np.ndarray,
    rank_ta: Sequence[float],
    rank_tb: Sequence[float],
    shafts: Sequence[str],
    support_ta: Sequence[float],
    support_tb: Sequence[float],
    support_source: str,
    template_event_counts: Optional[Mapping[str, object]] = None,
    n_axis_boot: int = 200,
    n_pair_boot: int = 500,
    line_threshold: float = 0.50,
    seed: int = 0,
) -> Dict[str, object]:
    """Freeze one subject's interictal TA/TB axes and contact-evaluated fields.

    No ictal input is accepted.  Axes use every joint-valid mapped contact.  Field
    models use the fixed subset with positive support in both templates, and store
    their kernel weights so future onset definitions can be scored without
    rebuilding an axis, plane, bandwidth, or interictal field.
    """
    names_arr = np.asarray([str(x) for x in names], object)
    x = np.asarray(coords, float)
    ra = np.asarray(rank_ta, float)
    rb = np.asarray(rank_tb, float)
    sh = np.asarray([str(x) for x in shafts], object)
    sa = np.asarray(support_ta, float)
    sb = np.asarray(support_tb, float)
    n = len(names_arr)
    if not (x.shape == (n, 3) and len(ra) == len(rb) == len(sh) == len(sa) == len(sb) == n):
        raise ValueError("interictal template field inputs are not contact-aligned")
    if len(set(names_arr.tolist())) != n:
        raise ValueError("interictal template field contact names must be unique")

    pair = compute_template_axis_pair(
        x, ra, rb, sh, n_axis_boot=n_axis_boot, n_pair_boot=n_pair_boot,
        seed=seed, line_threshold=line_threshold,
    )
    record: Dict[str, object] = {
        "contract": INTERICTAL_FIELD_CONTRACT,
        "subject_id": str(subject_id),
        "dataset": str(dataset),
        "subject": str(subject),
        "stable_k": int(stable_k),
        "template_labels": {"a": "TA", "b": "TB"},
        "axis_definition": TEMPLATE_AXIS_DEFINITION,
        "axis_direction_convention": TEMPLATE_AXIS_DIRECTION,
        "status": pair.get("status"),
        "names": names_arr,
        "coords": x,
        "shafts": sh,
        "rank_a": ra,
        "rank_b": rb,
        "earliness_a": z_earliness(ra),
        "earliness_b": z_earliness(rb),
        "support_a": sa,
        "support_b": sb,
        "support_source": str(support_source),
        "template_event_counts": dict(template_event_counts or {}),
        "axis_pair": pair,
    }
    if pair.get("status") != "ok":
        record["direction_validity"] = {
            "ta": assess_axis_direction_validity(pair.get("axis_a", {})),
            "tb": assess_axis_direction_validity(pair.get("axis_b", {})),
            "pair": {
                "axis_pair_estimable": False,
                "geometry_2d_supported": False,
                "strict_stability_pass": False,
            },
        }
        record["interictal_field"] = {"status": "axis_not_available"}
        return record

    va = assess_axis_direction_validity(pair["axis_a"])
    vb = assess_axis_direction_validity(pair["axis_b"])
    record["direction_validity"] = {
        "ta": va,
        "tb": vb,
        "pair": {
            "axis_pair_estimable": bool(pair.get("axis_pair_estimable")),
            "geometry_2d_supported": bool(pair.get("geometry_2d_supported")),
            "strict_stability_pass": bool(pair.get("strict_stability_pass")),
            "relation": pair.get("relation"),
            "paired_contact_bootstrap": pair.get("pair_bootstrap"),
        },
    }

    keep = (
        np.isfinite(x).all(1) & np.isfinite(ra) & np.isfinite(rb)
        & np.isfinite(sa) & np.isfinite(sb) & (sa > 0) & (sb > 0)
    )
    if int(keep.sum()) < 6:
        record["interictal_field"] = {
            "status": "insufficient_positive_joint_support",
            "n_contacts": int(keep.sum()),
        }
        return record

    fnames = names_arr[keep]
    fx, fra, frb = x[keep], ra[keep], rb[keep]
    fsa, fsb = sa[keep], sb[keep]
    fea, feb = z_earliness(fra), z_earliness(frb)
    axa, axb = pair["axis_a"], pair["axis_b"]
    own_a = make_normalized_plane(fx, axa["u"], origin=axa["xbar"])
    own_b = make_normalized_plane(fx, axb["u"], origin=axb["xbar"])
    if own_a.get("status") != "ok" or own_b.get("status") != "ok":
        record["interictal_field"] = {
            "status": "own_plane_failed",
            "n_contacts": int(keep.sum()),
            "own_ta_status": own_a.get("status"),
            "own_tb_status": own_b.get("status"),
        }
        return record

    scorers = {
        "own_a": make_field_scorer(fea, own_a["points"], fsa, own_a["sigma"]),
        "own_b": make_field_scorer(feb, own_b["points"], fsb, own_b["sigma"]),
    }
    planes = {"own_a": own_a, "own_b": own_b}
    if pair["relation"].get("collinear") and pair["shared_axis"].get("status") == "ok":
        shared = make_normalized_plane(fx, pair["shared_axis"]["u"], origin=axa["xbar"])
        if shared.get("status") == "ok":
            scorers["shared_a"] = make_field_scorer(fea, shared["points"], fsa, shared["sigma"])
            scorers["shared_b"] = make_field_scorer(feb, shared["points"], fsb, shared["sigma"])
            planes["shared"] = shared

    record["interictal_field"] = {
        "status": "ok",
        "field_contact_policy": "joint_valid_mapped_and_positive_support_in_both_TA_TB",
        "contact_order": fnames,
        "coords": fx,
        "shafts": sh[keep],
        "rank_a": fra,
        "rank_b": frb,
        "earliness_a": fea,
        "earliness_b": feb,
        "support_a": fsa,
        "support_b": fsb,
        "n_contacts": int(keep.sum()),
        "planes": planes,
        "field_models": scorers,
        "reuse_contract": {
            "join": "align future activation by exact channel name to contact_order",
            "frozen": ["axis", "plane", "bandwidth", "support", "template_field", "kernel_weights"],
            "forbidden": "do not refit axis or plane from ictal/onset values",
        },
    }
    record["interictal_field"]["fingerprint_algorithm"] = INTERICTAL_FIELD_FINGERPRINT_ALGORITHM
    record["interictal_field"]["fingerprint_sha256"] = interictal_field_fingerprint(record)
    return record


def interictal_field_fingerprint(record: Mapping[str, object]) -> str:
    """Hash every frozen numerical component needed by future activation scoring."""
    if record.get("contract") != INTERICTAL_FIELD_CONTRACT:
        raise ValueError(f"unsupported interictal field contract: {record.get('contract')}")
    field = record.get("interictal_field") or {}
    if field.get("status") != "ok":
        raise ValueError(f"interictal field unavailable: {field.get('status')}")
    algorithm = field.get("fingerprint_algorithm")
    if algorithm != INTERICTAL_FIELD_FINGERPRINT_ALGORITHM:
        raise ValueError(f"unsupported interictal field fingerprint algorithm: {algorithm}")
    digest = hashlib.sha256()

    def add_text(value: object) -> None:
        # Production JSON converts unavailable NumPy NaN scalars to ``null``.
        # Treat both representations as the same semantic value so a frozen
        # record validates identically before and after JSON serialization.
        if value is None or (
            isinstance(value, (float, np.floating)) and not np.isfinite(float(value))
        ):
            value = "<nonfinite>"
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")

    def add_array(value: object) -> None:
        array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())

    add_text(record.get("contract"))
    add_text(algorithm)
    add_text(record.get("subject_id"))
    add_text(record.get("axis_definition"))
    add_text(record.get("axis_direction_convention"))
    for name in field.get("contact_order", []):
        add_text(name)
    for key in ("coords", "rank_a", "rank_b", "earliness_a", "earliness_b",
                "support_a", "support_b"):
        add_array(field[key])
    pair = record.get("axis_pair") or {}
    for key in ("axis_a", "axis_b"):
        axis = pair.get(key) or {}
        for array_key in ("u", "along", "earliness_gradient_u", "propagation_vector"):
            add_array(axis[array_key])
        for scalar_key in ("n", "n_shafts", "effective_rank", "R2",
                           "bootstrap_cosine", "loso_cosine"):
            add_text(axis.get(scalar_key))
    relation = pair.get("relation") or {}
    for key in ("cosine", "abs_cosine", "line_angle_deg", "collinear", "relation"):
        add_text(relation.get(key))
    pair_bootstrap = pair.get("pair_bootstrap") or {}
    for key in ("p_collinear", "p_sign_stable", "robust_collinear"):
        add_text(pair_bootstrap.get(key))
    if (pair.get("shared_axis") or {}).get("status") == "ok":
        add_array(pair["shared_axis"]["u"])
    for key in sorted((field.get("planes") or {}).keys()):
        plane = field["planes"][key]
        add_text(key)
        for array_key in ("points", "u", "w", "origin"):
            add_array(plane[array_key])
        add_text(float(plane["scale_mm"]))
        add_text(float(plane["sigma"]))
    for key in sorted((field.get("field_models") or {}).keys()):
        model = field["field_models"][key]
        add_text(key)
        for array_key in ("template_field", "points", "support", "weight_id", "weight_mirror"):
            add_array(model[array_key])
        add_text(float(model["sigma"]))
    return digest.hexdigest()


def interictal_field_quality_tier(record: Mapping[str, object]) -> str:
    """Return a descriptive input-quality tier without changing eligibility."""
    field = record.get("interictal_field") or {}
    if field.get("status") != "ok":
        return "field_unavailable"
    pair = record.get("axis_pair") or {}
    if not bool(pair.get("geometry_2d_supported")):
        return "geometry_unsupported"
    if bool(pair.get("strict_stability_pass")):
        return "strict_2d"
    return "non_strict_2d"


def scorers_from_interictal_record(record: Mapping[str, object]) -> Dict[str, Dict[str, object]]:
    """Load frozen NumPy scorer dictionaries from a serialized interictal record."""
    if record.get("contract") != INTERICTAL_FIELD_CONTRACT:
        raise ValueError(f"unsupported interictal field contract: {record.get('contract')}")
    field = record.get("interictal_field") or {}
    if field.get("status") != "ok":
        raise ValueError(f"interictal field unavailable: {field.get('status')}")
    algorithm = field.get("fingerprint_algorithm")
    if algorithm != INTERICTAL_FIELD_FINGERPRINT_ALGORITHM:
        raise ValueError(f"unsupported interictal field fingerprint algorithm: {algorithm}")
    expected = field.get("fingerprint_sha256")
    if not expected or str(expected) != interictal_field_fingerprint(record):
        raise ValueError("interictal field fingerprint mismatch")
    models = field.get("field_models") or {}
    required = ("own_a", "own_b")
    if not all(k in models for k in required):
        raise ValueError("interictal record is missing own TA/TB field models")
    out: Dict[str, Dict[str, object]] = {}
    for key, model in models.items():
        out[str(key)] = {
            "template_field": np.asarray(model["template_field"], float),
            "points": np.asarray(model["points"], float),
            "support": np.asarray(model["support"], float),
            "sigma": float(model["sigma"]),
            "weight_id": np.asarray(model["weight_id"], float),
            "weight_mirror": np.asarray(model["weight_mirror"], float),
        }
    return out


def align_activation_to_interictal_field(
    record: Mapping[str, object], activation_names: Sequence[str], activation: Sequence[float]
) -> Dict[str, object]:
    """Name-align one future activation vector to the frozen interictal field order."""
    source_names = [str(x) for x in activation_names]
    values = np.asarray(activation, float)
    if len(source_names) != len(values):
        raise ValueError("activation names and values have different lengths")
    if len(set(source_names)) != len(source_names):
        raise ValueError("activation channel names must be unique")
    field = record.get("interictal_field") or {}
    target_names = [str(x) for x in field.get("contact_order", [])]
    source_index = {name: i for i, name in enumerate(source_names)}
    aligned = np.full(len(target_names), np.nan, float)
    matched = np.zeros(len(target_names), bool)
    for i, name in enumerate(target_names):
        if name in source_index:
            aligned[i] = values[source_index[name]]
            matched[i] = True
    return {
        "values": aligned,
        "matched_mask": matched,
        "n_target": len(target_names),
        "n_matched": int(matched.sum()),
        "n_finite": int(np.isfinite(aligned).sum()),
        "missing_names": [target_names[i] for i in np.where(~matched)[0]],
    }


def _smooth_from_weights(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Fast kernel smoothing with finite-value re-normalization."""
    v = np.asarray(values, float)
    fin = np.isfinite(v)
    out = np.full(weights.shape[0], np.nan)
    if not np.any(fin):
        return out
    denom = weights[:, fin].sum(axis=1)
    ok = denom > 1e-12
    out[ok] = (weights[ok][:, fin] @ v[fin]) / denom[ok]
    return out


def _smooth_matrix_from_weights(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Batch equivalent of :func:`_smooth_from_weights` for rows of values."""
    v = np.asarray(values, float)
    if v.ndim != 2:
        raise ValueError("values must have shape (n_draw, n_contact)")
    finite = np.isfinite(v)
    numerator = np.nan_to_num(v, nan=0.0) @ weights.T
    denominator = finite.astype(float) @ weights.T
    out = np.full(numerator.shape, np.nan)
    ok = denominator > 1e-12
    out[ok] = numerator[ok] / denominator[ok]
    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _template_projection_z(template: np.ndarray, activation: np.ndarray) -> float:
    """Project activation onto a standardized frozen template.

    The result has the same units as ``activation`` (baseline robust-z for the
    ictal-field consumer).  Unlike Pearson correlation, it retains the spatial
    contrast amplitude while remaining insensitive to a spatially uniform
    offset.  Algebraically this is ``corr(template, activation) *
    std(activation)`` on the pairwise-finite support.
    """
    t = np.asarray(template, float)
    a = np.asarray(activation, float)
    ok = np.isfinite(t) & np.isfinite(a)
    if int(ok.sum()) < 3:
        return np.nan
    tc = t[ok] - float(np.mean(t[ok]))
    tsd = float(np.std(tc))
    if not np.isfinite(tsd) or tsd < 1e-12:
        return np.nan
    return float(np.mean((tc / tsd) * a[ok]))


def score_field(scorer: Mapping[str, object], activation: Sequence[float]) -> Dict[str, object]:
    """Score morphology and amplitude-aware expression of one frozen field.

    ``signed_r`` / ``abs_r`` preserve the historical scale-free morphology
    contract. ``signed_projection_z`` / ``abs_projection_z`` quantify how
    strongly that morphology is expressed in the activation field.  When a
    correlation candidate exists, both readouts use the same abs-correlation
    selected identity/mirror orientation so amplitude cannot reselect the
    geometry after looking at energy.
    """
    value = np.asarray(activation, float)
    if "weight_id" in scorer:
        act_id = _smooth_from_weights(value, np.asarray(scorer["weight_id"], float))
        act_mirror = _smooth_from_weights(value, np.asarray(scorer["weight_mirror"], float))
    else:  # backward-compatible scorer dictionaries
        pts = np.asarray(scorer["points"], float)
        sup = np.asarray(scorer["support"], float)
        sigma = float(scorer["sigma"])
        act_id = kernel_smooth_at_contacts(value, pts, pts, sup, sigma)
        mirrored_eval = pts.copy()
        mirrored_eval[:, 1] *= -1
        act_mirror = kernel_smooth_at_contacts(value, pts, mirrored_eval, sup, sigma)
    tpl = np.asarray(scorer["template_field"], float)
    c_id, c_mirror = _pearson(tpl, act_id), _pearson(tpl, act_mirror)
    q_id = _template_projection_z(tpl, act_id)
    q_mirror = _template_projection_z(tpl, act_mirror)
    candidates = [("identity", c_id), ("mirror", c_mirror)]
    candidates = [(k, c) for k, c in candidates if np.isfinite(c)]
    if not candidates:
        q_candidates = [("identity", q_id), ("mirror", q_mirror)]
        q_candidates = [(k, q) for k, q in q_candidates if np.isfinite(q)]
        q_choice, projection = (
            max(q_candidates, key=lambda z: abs(z[1]))
            if q_candidates else (None, np.nan)
        )
        return {
            "signed_r": np.nan,
            "abs_r": np.nan,
            "mirror_choice": None,
            "r_identity": c_id,
            "r_mirror": c_mirror,
            "signed_projection_z": float(projection),
            "abs_projection_z": abs(float(projection)),
            "projection_mirror_choice": q_choice,
            "projection_identity_z": q_id,
            "projection_mirror_z": q_mirror,
        }
    choice, signed = max(candidates, key=lambda z: abs(z[1]))
    projection = q_id if choice == "identity" else q_mirror
    return {
        "signed_r": float(signed),
        "abs_r": abs(float(signed)),
        "mirror_choice": choice,
        "r_identity": c_id,
        "r_mirror": c_mirror,
        "signed_projection_z": float(projection),
        "abs_projection_z": abs(float(projection)),
        "projection_mirror_choice": choice,
        "projection_identity_z": q_id,
        "projection_mirror_z": q_mirror,
    }


def _row_pearson(template: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Pearson(template, row) for every row, with pairwise-finite masking."""
    t = np.asarray(template, float)
    y = np.asarray(values, float)
    mask = np.isfinite(y) & np.isfinite(t)[None, :]
    n = mask.sum(axis=1).astype(float)
    tx = np.where(mask, t[None, :], 0.0)
    yy = np.where(mask, y, 0.0)
    sx, sy = tx.sum(1), yy.sum(1)
    sxx, syy = (tx * tx).sum(1), (yy * yy).sum(1)
    sxy = (tx * yy).sum(1)
    cov = sxy - sx * sy / np.maximum(n, 1)
    vx = sxx - sx * sx / np.maximum(n, 1)
    vy = syy - sy * sy / np.maximum(n, 1)
    denom = np.sqrt(np.maximum(vx, 0) * np.maximum(vy, 0))
    out = np.full(len(y), np.nan)
    ok = (n >= 3) & (denom > 1e-12)
    out[ok] = cov[ok] / denom[ok]
    return out


def _row_template_projection_z(template: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Batch counterpart of :func:`_template_projection_z`.

    Each row is evaluated on its own pairwise-finite support.  The frozen
    template is standardized on that support and the activation is left in its
    native units, so the result retains robust-z spatial-contrast amplitude.
    """
    t = np.asarray(template, float)
    y = np.asarray(values, float)
    if y.ndim != 2:
        raise ValueError("values must have shape (n_draw, n_field_point)")
    mask = np.isfinite(y) & np.isfinite(t)[None, :]
    n = mask.sum(axis=1).astype(float)
    tx = np.where(mask, t[None, :], 0.0)
    yy = np.where(mask, y, 0.0)
    t_mean = tx.sum(axis=1) / np.maximum(n, 1.0)
    tc = np.where(mask, t[None, :] - t_mean[:, None], 0.0)
    t_sd = np.sqrt((tc * tc).sum(axis=1) / np.maximum(n, 1.0))
    out = np.full(len(y), np.nan)
    ok = (n >= 3) & np.isfinite(t_sd) & (t_sd > 1e-12)
    if np.any(ok):
        out[ok] = (
            ((tc[ok] / t_sd[ok, None]) * yy[ok]).sum(axis=1)
            / n[ok]
        )
    return out


def score_field_batch(scorer: Mapping[str, object], activations: np.ndarray) -> Dict[str, np.ndarray]:
    """Vectorized, numerically equivalent morphology and projection score."""
    v = np.asarray(activations, float)
    if "weight_id" not in scorer:
        rows = [score_field(scorer, row) for row in v]
        return {"signed_r": np.asarray([r["signed_r"] for r in rows]),
                "abs_r": np.asarray([r["abs_r"] for r in rows]),
                "signed_projection_z": np.asarray(
                    [r["signed_projection_z"] for r in rows]
                ),
                "abs_projection_z": np.asarray(
                    [r["abs_projection_z"] for r in rows]
                )}
    act_id = _smooth_matrix_from_weights(v, np.asarray(scorer["weight_id"], float))
    act_mirror = _smooth_matrix_from_weights(v, np.asarray(scorer["weight_mirror"], float))
    tpl = np.asarray(scorer["template_field"], float)
    c_id, c_mirror = _row_pearson(tpl, act_id), _row_pearson(tpl, act_mirror)
    q_id = _row_template_projection_z(tpl, act_id)
    q_mirror = _row_template_projection_z(tpl, act_mirror)
    choose_mirror = np.abs(c_mirror) > np.abs(c_id)
    only_mirror_corr = ~np.isfinite(c_id) & np.isfinite(c_mirror)
    no_corr = ~np.isfinite(c_id) & ~np.isfinite(c_mirror)
    choose_mirror[only_mirror_corr] = True
    only_mirror_q = no_corr & ~np.isfinite(q_id) & np.isfinite(q_mirror)
    both_q = no_corr & np.isfinite(q_id) & np.isfinite(q_mirror)
    choose_mirror[only_mirror_q] = True
    choose_mirror[both_q] = np.abs(q_mirror[both_q]) > np.abs(q_id[both_q])
    signed = np.where(choose_mirror, c_mirror, c_id)
    projection = np.where(choose_mirror, q_mirror, q_id)
    return {
        "signed_r": signed,
        "abs_r": np.abs(signed),
        "signed_projection_z": projection,
        "abs_projection_z": np.abs(projection),
    }


def score_scorer_bundle(scorers: Mapping[str, Mapping[str, object]],
                        activation: Sequence[float]) -> Dict[str, float]:
    """Score all fixed fields and recompute own/shared maxAB for this activation."""
    result: Dict[str, float] = {}
    for name, scorer in scorers.items():
        s = score_field(scorer, activation)
        result[f"{name}_signed"] = float(s["signed_r"])
        result[f"{name}_abs"] = float(s["abs_r"])
        result[f"{name}_signed_projection_z"] = float(s["signed_projection_z"])
        result[f"{name}_abs_projection_z"] = float(s["abs_projection_z"])
    for prefix in ("own", "shared"):
        vals = [result.get(f"{prefix}_{t}_abs", np.nan) for t in ("a", "b")]
        finite = [v for v in vals if np.isfinite(v)]
        if finite:
            result[f"{prefix}_maxab"] = float(max(finite))
        q_vals = [
            result.get(f"{prefix}_{t}_abs_projection_z", np.nan)
            for t in ("a", "b")
        ]
        q_finite = [v for v in q_vals if np.isfinite(v)]
        if q_finite:
            result[f"{prefix}_maxab_projection_z"] = float(max(q_finite))
    return result


def score_scorer_bundle_batch(scorers: Mapping[str, Mapping[str, object]],
                              activations: np.ndarray) -> Dict[str, np.ndarray]:
    """Batch counterpart that still reselects A/B max independently for every row."""
    result: Dict[str, np.ndarray] = {}
    for name, scorer in scorers.items():
        s = score_field_batch(scorer, activations)
        result[f"{name}_signed"] = s["signed_r"]
        result[f"{name}_abs"] = s["abs_r"]
        result[f"{name}_signed_projection_z"] = s["signed_projection_z"]
        result[f"{name}_abs_projection_z"] = s["abs_projection_z"]
    for prefix in ("own", "shared"):
        keys = [f"{prefix}_{t}_abs" for t in ("a", "b") if f"{prefix}_{t}_abs" in result]
        if keys:
            result[f"{prefix}_maxab"] = np.nanmax(np.vstack([result[k] for k in keys]), axis=0)
        q_keys = [
            f"{prefix}_{t}_abs_projection_z"
            for t in ("a", "b")
            if f"{prefix}_{t}_abs_projection_z" in result
        ]
        if q_keys:
            result[f"{prefix}_maxab_projection_z"] = np.nanmax(
                np.vstack([result[k] for k in q_keys]), axis=0
            )
    return result
