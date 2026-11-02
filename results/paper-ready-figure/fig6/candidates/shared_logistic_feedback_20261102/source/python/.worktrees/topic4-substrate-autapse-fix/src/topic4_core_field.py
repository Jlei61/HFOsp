"""Topic 4 axis-constrained data-driven pathology field (spec rev3).

Pure computation: no simulation, no engine import.
"""
from __future__ import annotations

import numpy as np

M_DEFAULT = 9
EPS = 1e-3
TAU_H = 0.25
A0 = 1.5
B0 = 1.5
AXIAL_MARGIN = 2.0
SIGMA_S_FACTOR = 1.2
SHIFT_MM = 3.0


def axis_coords(pos, center, u_axis):
    """Axial (s) and transverse (r) coordinates. u_axis is undirected: flipping
    its sign negates both, which every score must be invariant to."""
    pos = np.asarray(pos, float)
    u = np.asarray(u_axis, float)
    u = u / np.linalg.norm(u)
    u_perp = np.array([-u[1], u[0]])
    d = pos - np.asarray(center, float)[None, :]
    return d @ u, d @ u_perp


def axial_basis_centers(s_support, M=M_DEFAULT):
    return np.linspace(float(s_support[0]), float(s_support[1]), int(M))


def partition_of_unity(s, kappa, sigma_s):
    """Normalised Gaussian bases: rows sum to exactly 1 (spec 4.1)."""
    s = np.asarray(s, float)
    kappa = np.asarray(kappa, float)
    logw = -((s[:, None] - kappa[None, :]) ** 2) / (2.0 * float(sigma_s) ** 2)
    logw -= logw.max(axis=1, keepdims=True)
    w = np.exp(logw)
    return w / w.sum(axis=1, keepdims=True)


from scipy.special import expit
from scipy.stats import truncnorm

V_BASE = 18.0
V_RESET = 11.0
CORE_MEAN = 17.5
CORE_STD = 1.0


def sample_core_quantiles(n_E, seed):
    """One uniform quantile per E neuron, drawn once and frozen. Position- and
    field-independent, so every arm shares the same latent draw."""
    return np.random.default_rng(int(seed)).uniform(0.0, 1.0, size=int(n_E))


def core_thresholds(u, core_mean=CORE_MEAN, core_std=CORE_STD, v_reset=V_RESET):
    """Truncated-normal inverse transform: same distribution as the engine's
    rejection sampler, but deterministic per neuron. Bitwise reproduction of the
    legacy draw is impossible -- rejection makes its stream position data-dependent."""
    a = (float(v_reset) - float(core_mean)) / float(core_std)
    return truncnorm.ppf(np.asarray(u, float), a=a, b=np.inf,
                         loc=float(core_mean), scale=float(core_std))


def signed_depth(v_core, v_base=V_BASE):
    return float(v_base) - np.asarray(v_core, float)


def project_to_budget(q, target_count, tau_h=TAU_H, eps=EPS, max_iter=200):
    """Bisect lambda so that sum_i h_i == target_count.

    h is strictly decreasing in lambda, so the root is unique. This is a
    LEVEL-SET operation: the region's size is pinned by the budget and q only
    sets its shape (spec 4.4).
    """
    q = np.asarray(q, float)
    if not np.isfinite(q).all():
        raise ValueError("project_to_budget: q contains non-finite values")
    if (q + eps <= 0).any():
        raise ValueError("project_to_budget: q + eps must be positive")
    target = float(target_count)
    if not np.isfinite(target) or not (0.0 < target < q.size):
        raise ValueError(
            f"project_to_budget: target_count must lie in (0, {q.size}), got {target}")

    lq = np.log(q + eps)
    lo, hi = lq.min() - 20.0, lq.max() + 20.0
    for _ in range(max_iter):
        lam = 0.5 * (lo + hi)
        if expit((lq - lam) / tau_h).sum() > target:
            lo = lam
        else:
            hi = lam
    lam = 0.5 * (lo + hi)
    return expit((lq - lam) / tau_h), lam


def build_vth(h, d, n_total, n_E, v_base=V_BASE):
    """Per-neuron threshold vector for the engine. I neurons keep baseline."""
    vth = np.full(int(n_total), float(v_base))
    vth[:int(n_E)] = float(v_base) - np.asarray(h, float) * np.asarray(d, float)
    return vth


ARM_NAMES = (
    "manual_hard",        # legacy engine path (rejection sampler + np.minimum)
    "manual_projected",   # SAME hard mask, latent-quantile draws  -> comparison A
    "manual_smooth",      # smoothed two-core through the budget   -> baseline for B1-B4
    "uniform_axial",
    "width_wide",
    "width_narrow",
    "transverse_plus",
    "transverse_minus",
)

# Shape comparisons and the quantity each one must actually move (spec 4.4).
# rms_axial and aspect are BOTH dominated by the inter-core separation, which is
# 13.32 mm on the real geometry: two cores at +-6.66 mm have axial rms 6.71, and a
# uniform strip over the 22.1 mm support has 22.14/sqrt(12) = 6.39. They coincide by
# arithmetic accident, so neither metric can see the difference here. Core
# concentration and transverse rms are uncontaminated and separate these fields by
# 3.2x and 3.9x. Thresholds unchanged (2026-08-06, after the real-geometry pre-flight).
SHAPE_CHECKS = {
    "B1": dict(a="manual_smooth", b="uniform_axial",
               metric="core_concentration", kind="fold", threshold=0.20),
    "B2": dict(a="manual_smooth", b="width_wide",
               metric="rms_transverse", kind="fold", threshold=0.50),
    "B3": dict(a="manual_smooth", b="width_narrow",
               metric="rms_transverse", kind="fold", threshold=0.50),
    "B4": dict(a="manual_smooth", b="transverse_plus",
               metric="centroid_transverse", kind="abs", threshold=1.5),
}


def manual_mask(pos, src_xy, snk_xy, core_r):
    """The legacy two-disk core mask, in sheet coordinates."""
    pos = np.asarray(pos, float)
    d = np.minimum(((pos - np.asarray(src_xy, float)) ** 2).sum(1),
                   ((pos - np.asarray(snk_xy, float)) ** 2).sum(1))
    return d <= float(core_r) ** 2


def two_core_q(s, r, sep, rho=1.0, a0=A0, b0=B0, r_shift=0.0, eps=EPS):
    """Two elliptical cores at s = +-sep/2, transverse offset r_shift.

    rho is the FIXED-AREA aspect ratio (a = a0*rho, b = b0/rho), so it reshapes
    the region instead of merely blurring its edge (spec 4.4).
    """
    a, b = float(a0) * float(rho), float(b0) / float(rho)
    s = np.asarray(s, float)
    rr = np.asarray(r, float) - float(r_shift)
    q = np.zeros_like(s)
    for c in (-float(sep) / 2.0, float(sep) / 2.0):
        q = np.maximum(q, np.exp(-((s - c) ** 2 / (2 * a ** 2) + rr ** 2 / (2 * b ** 2))))
    return q + eps


def uniform_axial_q(s, r, kappa, sigma_s, sigma_perp, eps=EPS):
    """Flat axial profile: pi_m == 1/M on a partition-of-unity basis."""
    M = len(kappa)
    profile = partition_of_unity(np.asarray(s, float), kappa, sigma_s) @ np.full(M, 1.0 / M)
    return profile * np.exp(-np.asarray(r, float) ** 2 / (2 * float(sigma_perp) ** 2)) + eps


def arm_h(name, s, r, geom, target_count, manual_mask_E=None):
    """h field for one Stage 1 arm.

    manual_hard is not built here (legacy engine path). manual_projected is the
    hard mask verbatim -- it changes the DRAWS, not the geometry (spec 4.3.1).
    """
    if name == "manual_projected":
        if manual_mask_E is None:
            raise ValueError("manual_projected requires manual_mask_E")
        return np.asarray(manual_mask_E, bool).astype(float)
    sep = geom["sep"]
    if name == "manual_smooth":
        q = two_core_q(s, r, sep, rho=1.0)
    elif name == "uniform_axial":
        kappa = axial_basis_centers(geom["s_support"], geom["M"])
        q = uniform_axial_q(s, r, kappa, SIGMA_S_FACTOR * (kappa[1] - kappa[0]),
                            geom["sigma_perp"])
    elif name == "width_wide":
        q = two_core_q(s, r, sep, rho=0.5)
    elif name == "width_narrow":
        q = two_core_q(s, r, sep, rho=2.0)
    elif name == "transverse_plus":
        q = two_core_q(s, r, sep, rho=1.0, r_shift=+geom["shift_mm"])
    elif name == "transverse_minus":
        q = two_core_q(s, r, sep, rho=1.0, r_shift=-geom["shift_mm"])
    else:
        raise ValueError(f"arm_h does not build {name!r}")
    h, _ = project_to_budget(q, target_count)
    return h


def shape_metrics(h, s, r, core_centers_s=None, core_radius=1.5):
    """h-weighted geometry. Uses h, never h*d -- d is signed and h*d is not a
    non-negative mass (spec 9 / P0-7).

    `core_concentration` is the share of h within `core_radius` of the nearer core
    centre along the axis. Unlike rms_axial it is not dominated by the inter-core
    separation, so it can tell a two-core field from a uniform corridor.
    """
    h = np.asarray(h, float)
    w = h.sum()
    rms_ax = float(np.sqrt((h * np.asarray(s, float) ** 2).sum() / w))
    rms_tr = float(np.sqrt((h * np.asarray(r, float) ** 2).sum() / w))
    out = dict(rms_axial=rms_ax, rms_transverse=rms_tr,
               aspect=rms_tr / rms_ax if rms_ax > 0 else np.inf,
               centroid_transverse=float((h * np.asarray(r, float)).sum() / w),
               centroid_axial=float((h * np.asarray(s, float)).sum() / w),
               budget=float(w))
    if core_centers_s is not None:
        s_arr = np.asarray(s, float)
        near = np.min(np.abs(s_arr[:, None] - np.asarray(core_centers_s, float)[None, :]),
                      axis=1)
        out["core_concentration"] = float((h * (near <= float(core_radius))).sum() / w)
    return out


def preflight_shape(h_by_arm, s, r, target_count, checks=None, sep=None):
    """Refuse to launch 96 simulations on a vacuous shape comparison.

    Only the B comparisons are checked. manual_hard and manual_projected SHOULD
    be near-identical -- an all-pairs correlation gate would reject the correct
    implementation (third-review P0-2). Correlation is reported as a diagnostic,
    never as the gate.
    """
    checks = checks or SHAPE_CHECKS
    centers = None if sep is None else (-float(sep) / 2.0, +float(sep) / 2.0)
    metrics = {name: shape_metrics(h, s, r, core_centers_s=centers)
               for name, h in h_by_arm.items()}
    out, ok_all = {}, True
    for key, c in checks.items():
        ma, mb = metrics[c["a"]][c["metric"]], metrics[c["b"]][c["metric"]]
        if c["kind"] == "fold":
            # "differ by X%" as a FOLD change: require max/min >= 1 + X.
            # Encoding it as |a-b|/max would make 0.50 mean "exactly 2x", which is
            # precisely what rho=0.5 / rho=2.0 produce -- a knife edge in both
            # directions. The threshold NUMBERS are unchanged; only the arithmetic
            # that turns them into a test is corrected (2026-08-06).
            hi, lo = max(abs(ma), abs(mb)), max(min(abs(ma), abs(mb)), 1e-12)
            observed = hi / lo - 1.0
        elif c["kind"] == "rel":
            observed = abs(ma - mb) / max(abs(ma), abs(mb), 1e-12)
        else:
            observed = abs(ma - mb)
        ok = bool(observed >= c["threshold"])
        ok_all &= ok
        out[key] = dict(ok=ok, a=c["a"], b=c["b"], metric=c["metric"],
                        observed=float(observed), threshold=c["threshold"],
                        correlation=float(np.corrcoef(h_by_arm[c["a"]],
                                                      h_by_arm[c["b"]])[0, 1]))
    budget_err = {n: abs(m["budget"] - target_count) / target_count
                  for n, m in metrics.items()}
    worst = max(budget_err.values())
    ok_all &= bool(worst < 1e-6)
    return dict(ok=bool(ok_all), checks=out, metrics=metrics,
                worst_budget_error=float(worst))
