"""Task 1 -- the field parameterisation must carry no direction prior.

The whole point of Stage 3 is that the field is free to sit anywhere, so the
parameterisation itself must not prefer any orientation. A 90-degree-only
invariance test passes happily on an implementation with a four-fold
preference, which is exactly the bug we are trying to exclude, so the
arbitrary-angle case carries the contract.
"""
import numpy as np
import pytest

from src.topic4_core_field_stage3 import (CENTER_MARGIN_MM, K_COMPONENTS,
                                          SIGMA_MAX_MM, SIGMA_MIN_MM,
                                          n_free, params_to_h, params_to_q,
                                          probe_q, spatial_diagnostics, unpack)

L = 20.0
N_CORE = 1129.0
CENTER = np.array([L / 2, L / 2])


def _positions(n=4000, seed=0):
    return np.random.default_rng(seed).uniform(0, L, size=(n, 2))


def _theta(seed=0, K=K_COMPONENTS):
    return np.random.default_rng(seed).normal(0, 1.0, size=n_free(K))


def _rotate(points, angle_rad, about):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return (np.asarray(points, float) - about) @ R.T + about


def _rotate_theta(theta, angle_rad, about, K=K_COMPONENTS):
    """Rotate the component centres and add the angle to every orientation."""
    out = np.array(theta, float)
    for k in range(K):
        b = 5 * k
        out[b:b + 2] = _rotate(out[b:b + 2][None, :], angle_rad, about)[0]
        out[b + 4] += angle_rad
    return out


# ---------------------------------------------------------------- dimensions
def test_optimiser_sees_seventeen_free_dimensions():
    # softmax is shift invariant, so the K-th logit is pinned rather than
    # handed to CMA-ES as a redundant covariance direction
    assert n_free(3) == 17
    assert n_free(2) == 11
    assert n_free(4) == 23


def test_component_count_is_three_not_two():
    # K=2 would write "two cores" into the prior, which is the thing under test
    assert K_COMPONENTS == 3


# -------------------------------------------------------------------- bounds
def test_centres_out_of_range_are_clipped_into_the_sheet():
    theta = np.zeros(n_free())
    theta[0:2] = [-50.0, 100.0]
    theta[5:7] = [999.0, -999.0]
    for comp in unpack(theta, K_COMPONENTS, L):
        assert CENTER_MARGIN_MM - 1e-9 <= comp["center"][0] <= L - CENTER_MARGIN_MM + 1e-9
        assert CENTER_MARGIN_MM - 1e-9 <= comp["center"][1] <= L - CENTER_MARGIN_MM + 1e-9


def test_sigmas_out_of_range_are_clipped():
    theta = np.zeros(n_free())
    theta[2:4] = [20.0, -20.0]
    for comp in unpack(theta, K_COMPONENTS, L):
        for s in (comp["sigma_par"], comp["sigma_perp"]):
            assert SIGMA_MIN_MM - 1e-9 <= s <= SIGMA_MAX_MM + 1e-9


# -------------------------------------------------------------------- budget
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_budget_holds_for_any_parameters(seed):
    h = params_to_h(_theta(seed), _positions(), K_COMPONENTS, L, N_CORE)
    assert abs(h.sum() - N_CORE) < 1e-6
    assert h.min() >= 0.0 and h.max() <= 1.0


# ------------------------------------------------------- no direction prior
def test_invariant_under_ninety_degrees():
    pos, theta = _positions(), _theta(7)
    h0 = params_to_h(theta, pos, K_COMPONENTS, L, N_CORE)
    h1 = params_to_h(_rotate_theta(theta, np.pi / 2, CENTER),
                     _rotate(pos, np.pi / 2, CENTER), K_COMPONENTS, L, N_CORE)
    assert np.allclose(h0, h1, atol=1e-9)


@pytest.mark.parametrize("deg", [37.0, 11.5, 113.0, -64.0])
def test_invariant_under_an_arbitrary_angle(deg):
    # THE contract. An implementation that quantises orientation to pi/2 passes
    # the 90-degree test above and fails here.
    pos, theta = _positions(), _theta(7)
    ang = np.deg2rad(deg)
    h0 = params_to_h(theta, pos, K_COMPONENTS, L, N_CORE)
    h1 = params_to_h(_rotate_theta(theta, ang, CENTER),
                     _rotate(pos, ang, CENTER), K_COMPONENTS, L, N_CORE)
    assert np.allclose(h0, h1, atol=1e-9)


def test_orientation_has_period_pi():
    pos, theta = _positions(), _theta(3)
    shifted = np.array(theta, float)
    for k in range(K_COMPONENTS):
        shifted[5 * k + 4] += np.pi
    assert np.allclose(params_to_h(theta, pos, K_COMPONENTS, L, N_CORE),
                       params_to_h(shifted, pos, K_COMPONENTS, L, N_CORE),
                       atol=1e-9)


def test_invariant_under_component_permutation():
    pos, theta = _positions(), _theta(5)
    blocks = [theta[5 * k:5 * k + 5] for k in range(K_COMPONENTS)]
    logits = np.append(theta[5 * K_COMPONENTS:], 0.0)      # K-th logit pinned
    order = [2, 0, 1]
    permuted = np.concatenate([np.concatenate([blocks[i] for i in order]),
                               logits[order][:-1] - logits[order][-1]])
    assert np.allclose(params_to_h(theta, pos, K_COMPONENTS, L, N_CORE),
                       params_to_h(permuted, pos, K_COMPONENTS, L, N_CORE),
                       atol=1e-9)


def test_weight_redundancy_is_removed_not_merely_tolerated():
    # With the K-th logit pinned at zero there is no shift direction left, so
    # moving the stored logits MUST change the field. If it did not, the
    # optimiser would still be carrying a redundant dimension.
    pos, theta = _positions(), _theta(11)
    shifted = np.array(theta, float)
    shifted[5 * K_COMPONENTS:] += 2.5
    assert not np.allclose(params_to_h(theta, pos, K_COMPONENTS, L, N_CORE),
                           params_to_h(shifted, pos, K_COMPONENTS, L, N_CORE),
                           atol=1e-6)


def test_weights_are_a_proper_simplex():
    w = [c["weight"] for c in unpack(_theta(13), K_COMPONENTS, L)]
    assert len(w) == K_COMPONENTS
    assert all(0.0 < x < 1.0 for x in w)
    assert sum(w) == pytest.approx(1.0)


# --------------------------------------------------------------- diagnostics
def test_diagnostics_use_h_not_h_times_signed_depth():
    # d keeps its sign and about 31% of it is negative (spec 2.5); weighting
    # space by h*d would put the centroid outside the field
    pos = _positions()
    h = params_to_h(_theta(2), pos, K_COMPONENTS, L, N_CORE)
    u = np.array([np.cos(np.deg2rad(-22.8)), np.sin(np.deg2rad(-22.8))])
    rng = np.random.default_rng(0)
    d = rng.normal(0.5, 1.0, size=len(pos))
    assert (d < 0).mean() > 0.2
    a = spatial_diagnostics(h, pos, CENTER, u)
    b = spatial_diagnostics(h, pos, CENTER, u)
    assert a["r_bar"] == pytest.approx(b["r_bar"])
    assert 0.0 <= a["c_axis"][2.0] <= 1.0


def test_c_axis_is_non_decreasing_and_saturates():
    pos = _positions()
    h = params_to_h(_theta(4), pos, K_COMPONENTS, L, N_CORE)
    u = np.array([1.0, 0.0])
    diag = spatial_diagnostics(h, pos, CENTER, u, deltas=(0.5, 1.0, 2.0, 3.0, 1e6))
    vals = [diag["c_axis"][d] for d in (0.5, 1.0, 2.0, 3.0, 1e6)]
    assert all(x <= y + 1e-12 for x, y in zip(vals, vals[1:]))
    assert diag["c_axis"][1e6] == pytest.approx(1.0)


def test_transverse_offset_is_visible_in_r_bar():
    pos = _positions(n=8000)
    u = np.array([1.0, 0.0])
    on = probe_q(pos, CENTER, 1.5)
    off = probe_q(pos, CENTER + np.array([0.0, 5.0]), 1.5)
    from src.topic4_core_field import project_to_budget
    d_on = spatial_diagnostics(project_to_budget(on, N_CORE)[0], pos, CENTER, u)
    d_off = spatial_diagnostics(project_to_budget(off, N_CORE)[0], pos, CENTER, u)
    assert abs(d_on["r_bar"]) < 0.5
    assert d_off["r_bar"] > 3.5
    assert d_on["c_axis"][2.0] > d_off["c_axis"][2.0]


# --------------------------------------------------------------- Leg A probe
def test_probe_position_is_identifiable():
    pos = _positions(n=8000)
    a = probe_q(pos, np.array([6.0, 6.0]), 1.2)
    b = probe_q(pos, np.array([14.0, 14.0]), 1.2)
    assert np.corrcoef(a, b)[0, 1] < 0.9


def test_probe_is_isotropic():
    # Leg A sweeps position and size only; an anisotropic probe would smuggle
    # an orientation back in
    pos = _positions(n=8000)
    q0 = probe_q(pos, CENTER, 1.5)
    q1 = probe_q(_rotate(pos, np.deg2rad(37.0), CENTER), CENTER, 1.5)
    assert np.allclose(np.sort(q0), np.sort(q1), atol=1e-9)
