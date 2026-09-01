"""Stage 3: the field is free to sit anywhere on the sheet (spec section 9).

Stage 2 parameterised the field in the axis's own coordinate system -- an
axial profile times a transverse envelope -- which made "the field lies on the
axis" a construction rather than a finding. Here the field is a mixture of
free-centre anisotropic Gaussians over the sheet. The only thing still frozen
is the E->E connectivity anisotropy.

K = 3, deliberately: K = 1 cannot express two cores and K = 2 would write them
into the prior, which is the very thing under test.
"""
from __future__ import annotations

import numpy as np

from src.topic4_core_field import EPS, axis_coords, project_to_budget

K_COMPONENTS = 3
SIGMA_MIN_MM, SIGMA_MAX_MM = 0.4, 6.0
CENTER_MARGIN_MM = 2.0          # same value as AXIAL_MARGIN; centres stay off the edge


def _clip_center(c, L):
    """Confine a component centre to a DISC, not a square box.

    A square constraint box is invariant under 90-degree rotations about the
    sheet centre but not under an arbitrary one, so clipping to it is itself a
    four-fold direction prior -- precisely the bug the arbitrary-angle
    invariance test exists to catch. A disc is equivariant under every rotation.
    """
    mid = np.full(2, float(L) / 2.0)
    R = float(L) / 2.0 - CENTER_MARGIN_MM
    d = np.asarray(c, float) - mid
    n = float(np.hypot(d[0], d[1]))
    return mid + d * (R / n) if n > R else mid + d


def n_free(K=K_COMPONENTS):
    """5K per component plus K-1 weight logits.

    The K-th logit is pinned at zero rather than handed to the optimiser:
    softmax is shift invariant, so that direction is redundant and would only
    waste a covariance direction in CMA-ES.
    """
    return 5 * int(K) + int(K) - 1


def latent_to_theta(latent, K=K_COMPONENTS, L=20.0, logit_limit=4.0):
    """Decode O(1) optimizer coordinates into the legacy physical theta format.

    The first Stage 3 fit asked one isotropic CMA-ES covariance to learn scales
    spanning millimetres, log-millimetres, radians, and weight logits. It also
    hard-clipped centres and sigmas, creating large latent plateaus. This smooth
    decoder preconditions every coordinate and approaches, but never crosses, the
    physical bounds. The returned theta remains compatible with `unpack` and all
    existing renderers.
    """
    z = np.asarray(latent, float)
    if z.size != n_free(K):
        raise ValueError(f"expected {n_free(K)} latent values for K={K}, got {z.size}")
    theta = np.zeros_like(z)
    midpoint = np.full(2, float(L) / 2.0)
    radius = float(L) / 2.0 - CENTER_MARGIN_MM
    log_min, log_max = np.log(SIGMA_MIN_MM), np.log(SIGMA_MAX_MM)

    for k in range(int(K)):
        b = 5 * k
        direction = z[b:b + 2]
        norm = float(np.linalg.norm(direction))
        radial = radius * np.tanh(norm / 1.5)
        theta[b:b + 2] = (midpoint if norm < 1e-12 else
                           midpoint + radial * direction / norm)
        unit = 1.0 / (1.0 + np.exp(-np.clip(z[b + 2:b + 4], -40.0, 40.0)))
        theta[b + 2:b + 4] = log_min + unit * (log_max - log_min)
        theta[b + 4] = np.pi / (1.0 + np.exp(-float(np.clip(z[b + 4], -40.0, 40.0))))

    if int(K) > 1:
        theta[5 * K:] = float(logit_limit) * np.tanh(z[5 * K:] / 2.0)
    return theta


def unpack(theta, K=K_COMPONENTS, L=20.0):
    theta = np.asarray(theta, float)
    logits = np.append(theta[5 * K:5 * K + K - 1], 0.0)
    logits = logits - logits.max()
    w = np.exp(logits) / np.exp(logits).sum()
    out = []
    for k in range(K):
        b = 5 * k
        out.append(dict(
            center=_clip_center(theta[b:b + 2], L),
            sigma_par=float(np.clip(np.exp(theta[b + 2]), SIGMA_MIN_MM, SIGMA_MAX_MM)),
            sigma_perp=float(np.clip(np.exp(theta[b + 3]), SIGMA_MIN_MM, SIGMA_MAX_MM)),
            phi=float(theta[b + 4]), weight=float(w[k])))
    return out


def params_to_q(theta, pos_xy, K=K_COMPONENTS, L=20.0):
    """Raw field before the budget projection, in sheet coordinates."""
    pos = np.asarray(pos_xy, float)
    q = np.zeros(len(pos))
    for comp in unpack(theta, K, L):
        c, s = np.cos(comp["phi"]), np.sin(comp["phi"])
        d = pos - comp["center"]
        u = d[:, 0] * c + d[:, 1] * s              # along the component's own axis
        v = -d[:, 0] * s + d[:, 1] * c             # across it
        q += comp["weight"] * np.exp(
            -0.5 * ((u / comp["sigma_par"]) ** 2 + (v / comp["sigma_perp"]) ** 2))
    return q + EPS


def params_to_h(theta, pos_xy, K=K_COMPONENTS, L=20.0, target_count=1129.0):
    h, _ = project_to_budget(params_to_q(theta, pos_xy, K, L), target_count)
    return h


def probe_q(pos_xy, center_xy, sigma_mm):
    """Leg A's single isotropic blob: position and size are its only freedoms.

    Isotropic on purpose -- an elongated probe would smuggle an orientation
    back into a sweep whose whole point is that no direction is privileged.
    """
    d = np.asarray(pos_xy, float) - np.asarray(center_xy, float)
    return np.exp(-0.5 * (d ** 2).sum(1) / float(sigma_mm) ** 2) + EPS


def spatial_diagnostics(h, pos_xy, center, axis_unit, deltas=(1.0, 2.0, 3.0)):
    """Where the field sits, weighted by h.

    Weighting by h*d would be wrong: d = V_base - V_core keeps its sign and
    about 31% of it is negative, so h*d is not a non-negative spatial mass --
    the centroid could land outside the field and C_axis could leave [0, 1].
    """
    h = np.asarray(h, float)
    s, r = axis_coords(np.asarray(pos_xy, float), np.asarray(center, float),
                       np.asarray(axis_unit, float))
    m = h.sum()
    if m <= 0:
        raise ValueError("empty field: h sums to zero")
    return dict(
        r_bar=float((h * r).sum() / m),
        s_bar=float((h * s).sum() / m),
        rms_transverse=float(np.sqrt((h * r ** 2).sum() / m)),
        rms_axial=float(np.sqrt((h * s ** 2).sum() / m)),
        c_axis={float(d): float(h[np.abs(r) < float(d)].sum() / m) for d in deltas},
        mass=float(m))


def high_scoring_region(scores, n_valid, top_frac=0.10, min_valid=3):
    """Cells in the top `top_frac` of the map that are also well enough sampled.

    Deliberately a REGION, not the argmax. Picking the single best of ninety-odd
    cells at four seeds each is a winner's curse; the returned mask is what gets
    re-run on independent seeds so the optimism can be measured rather than
    inherited (plan, Leg A discipline).
    """
    s = np.asarray(scores, float)
    v = np.asarray(n_valid, int)
    eligible = np.isfinite(s) & (v >= int(min_valid))
    if not eligible.any():
        return np.zeros(s.shape, bool)
    k = max(1, int(np.ceil(eligible.sum() * float(top_frac))))
    cut = np.sort(s[eligible])[::-1][k - 1]
    return eligible & (s >= cut)
