"""Pure helpers for Topic 5 single-template gradient axes and shared-plane fields.

The module deliberately separates the interictal-only construction (axis pair,
collinearity, shared bisector) from the downstream ictal field readout.  It has no
filesystem I/O; the cohort runner owns joins, caching and serialization.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from src.dab_gradient_axis import RCOND, compute_dab_gradient_axis
from src.topic5_contact_similarity import kernel_smooth_at_contacts, median_nn_spacing


TEMPLATE_AXIS_DEFINITION = "template_propagation_axis_v2"
TEMPLATE_AXIS_DIRECTION = "positive_early_to_late"


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


def score_field(scorer: Mapping[str, object], activation: Sequence[float]) -> Dict[str, object]:
    """Correct abs-max identity/mirror contact-field correlation for one template."""
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
    candidates = [("identity", c_id), ("mirror", c_mirror)]
    candidates = [(k, c) for k, c in candidates if np.isfinite(c)]
    if not candidates:
        return {"signed_r": np.nan, "abs_r": np.nan, "mirror_choice": None,
                "r_identity": c_id, "r_mirror": c_mirror}
    choice, signed = max(candidates, key=lambda z: abs(z[1]))
    return {"signed_r": float(signed), "abs_r": abs(float(signed)),
            "mirror_choice": choice, "r_identity": c_id, "r_mirror": c_mirror}


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


def score_field_batch(scorer: Mapping[str, object], activations: np.ndarray) -> Dict[str, np.ndarray]:
    """Vectorized, numerically equivalent identity/mirror abs-max field score."""
    v = np.asarray(activations, float)
    if "weight_id" not in scorer:
        rows = [score_field(scorer, row) for row in v]
        return {"signed_r": np.asarray([r["signed_r"] for r in rows]),
                "abs_r": np.asarray([r["abs_r"] for r in rows])}
    act_id = _smooth_matrix_from_weights(v, np.asarray(scorer["weight_id"], float))
    act_mirror = _smooth_matrix_from_weights(v, np.asarray(scorer["weight_mirror"], float))
    tpl = np.asarray(scorer["template_field"], float)
    c_id, c_mirror = _row_pearson(tpl, act_id), _row_pearson(tpl, act_mirror)
    choose_mirror = np.abs(c_mirror) > np.abs(c_id)
    signed = np.where(choose_mirror, c_mirror, c_id)
    signed[~np.isfinite(c_id) & np.isfinite(c_mirror)] = c_mirror[~np.isfinite(c_id) & np.isfinite(c_mirror)]
    return {"signed_r": signed, "abs_r": np.abs(signed)}


def score_scorer_bundle(scorers: Mapping[str, Mapping[str, object]],
                        activation: Sequence[float]) -> Dict[str, float]:
    """Score all fixed fields and recompute own/shared maxAB for this activation."""
    result: Dict[str, float] = {}
    for name, scorer in scorers.items():
        s = score_field(scorer, activation)
        result[f"{name}_signed"] = float(s["signed_r"])
        result[f"{name}_abs"] = float(s["abs_r"])
    for prefix in ("own", "shared"):
        vals = [result.get(f"{prefix}_{t}_abs", np.nan) for t in ("a", "b")]
        finite = [v for v in vals if np.isfinite(v)]
        if finite:
            result[f"{prefix}_maxab"] = float(max(finite))
    return result


def score_scorer_bundle_batch(scorers: Mapping[str, Mapping[str, object]],
                              activations: np.ndarray) -> Dict[str, np.ndarray]:
    """Batch counterpart that still reselects A/B max independently for every row."""
    result: Dict[str, np.ndarray] = {}
    for name, scorer in scorers.items():
        s = score_field_batch(scorer, activations)
        result[f"{name}_signed"] = s["signed_r"]
        result[f"{name}_abs"] = s["abs_r"]
    for prefix in ("own", "shared"):
        keys = [f"{prefix}_{t}_abs" for t in ("a", "b") if f"{prefix}_{t}_abs" in result]
        if keys:
            result[f"{prefix}_maxab"] = np.nanmax(np.vstack([result[k] for k in keys]), axis=0)
    return result
