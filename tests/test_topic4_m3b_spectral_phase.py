"""M3B-R2 spectral phase-map — TDD roadmap.

Two layers:

* **TDD-0 (real, GREEN now):** the artifact/claim contract — `STATUS.md` exists, declares the
  artifacts, freezes the allowed verdict categories and forbidden claims. Pure-data; no eigensolver.
* **TDD-1..13 / TDD-15 (pending roadmap):** every named test from
  `docs/superpowers/plans/2026-06-27-sef-hfo-m3b-spectral-phase-map-plan.md` is listed below as a
  PARAMETRIZED SKIP that names the producing module `src/topic4_m3b_spectral_phase.py` (not yet
  written). These are explicit placeholders (SKIPPED, never hollow-passing): when the spectral module
  is designed, each section's stub is replaced by real RED tests. The M3A-overlay contract (plan
  TDD-14) is already implemented at the contract layer in `tests/test_sef_hfo_m3_interface.py`.
"""
from pathlib import Path

import numpy as np
import pytest

import src.topic4_m3b_spectral_phase as spm

_STATUS = (Path(__file__).resolve().parents[1]
           / "results/topic4_sef_hfo/m3b_spectral_phase_map/STATUS.md")

_DECLARED_ARTIFACTS = (
    "homogeneous_dispersion.json", "finite_jacobian_grid.json", "mode_metrics.csv",
    "control_summary.json", "mode_readout_projection.json", "m3a_interface_audit.json",
    "slow_trajectory_overlay.csv", "snn_spotcheck_summary.json", "figures/README.md",
)
_ALLOWED_VERDICTS = (
    "SPM-PASS full bridge", "SPM-PASS spontaneous mechanism", "SPM-PASS frozen map",
    "SPM-BOUNDED negative", "SPM-MODEL mismatch", "SPM-UNRESOLVED",
)
_FORBIDDEN_CLAIMS = (
    "W causes seizure", "proves clinical seizure onset",
)


# ---------------------------------------------------------------------------
# TDD-0: claim boundary and artifact contract (real, GREEN)
# ---------------------------------------------------------------------------
def test_status_exists_and_mentions_round1_m3b_next_m3a():
    assert _STATUS.exists(), f"missing TDD-0 STATUS.md at {_STATUS}"
    text = _STATUS.read_text(encoding="utf-8")
    assert "Round-1" in text
    assert "M3B-R2" in text
    assert "M3A handoff" in text


def test_artifact_paths_are_declared():
    text = _STATUS.read_text(encoding="utf-8")
    for art in _DECLARED_ARTIFACTS:
        assert art in text, f"declared artifact {art!r} not listed in STATUS.md"


def test_no_forbidden_claims_in_status():
    text = _STATUS.read_text(encoding="utf-8")
    # the guardrail itself must be frozen before any eigenvalue exists
    assert "## Forbidden claims" in text
    for claim in _FORBIDDEN_CLAIMS:
        assert claim in text, f"forbidden claim {claim!r} not frozen in STATUS.md"
    for verdict in _ALLOWED_VERDICTS:
        assert verdict in text, f"allowed verdict {verdict!r} not declared in STATUS.md"


# ---------------------------------------------------------------------------
# TDD-1: grid / kernel / state-vector contract (real)
# ---------------------------------------------------------------------------
def test_pack_unpack_state_roundtrip():
    g = spm.Grid(n=5, L=12.0)
    rng = np.random.default_rng(0)
    state = {f: rng.standard_normal((g.n, g.n)) for f in spm.STATE_FIELDS}
    z = spm.pack_state(state, g)
    back = spm.unpack_state(z, g)
    for f in spm.STATE_FIELDS:
        assert np.array_equal(back[f], state[f]), f"{f} did not roundtrip"


def test_state_size_equals_6_times_grid_size():
    g = spm.Grid(n=4, L=12.0)
    state = {f: np.zeros((g.n, g.n)) for f in spm.STATE_FIELDS}
    z = spm.pack_state(state, g)
    assert z.shape == (6 * g.size,)
    assert z.shape == (6 * g.n * g.n,)


def test_no_core_mask_empty_or_uniform():
    g = spm.Grid(n=32, L=12.0)
    cm = spm.make_core_mask(g, kind="none")
    assert cm.mask.dtype == bool
    assert not cm.mask.any(), "no-core mask must be empty (no distinguished core)"
    assert cm.area_fraction == 0.0


def test_single_core_mask_area_reasonable():
    g = spm.Grid(n=48, L=12.0)
    cm = spm.make_core_mask(g, kind="single", radius=1.5)
    assert cm.mask.any() and not cm.mask.all()
    # a 1.5mm-radius disk on a 12mm sheet is a few % of area: not a single pixel, not the whole sheet
    assert 0.01 < cm.area_fraction < 0.30, cm.area_fraction


def test_two_core_mask_centers_on_ee_axis():
    g = spm.Grid(n=64, L=12.0)
    cm = spm.make_core_mask(g, kind="two", radius=1.2, separation=5.0, theta=spm.THETA_EE)
    assert len(cm.centers) == 2
    # the two recorded centers lie on the theta line through the origin: their connecting vector
    # is parallel to (cos theta, sin theta)
    (x1, y1), (x2, y2) = cm.centers
    ang = np.arctan2(y1 - y2, x1 - x2)
    ang = (ang + np.pi / 2) % np.pi - np.pi / 2          # fold to a line orientation
    assert abs(ang - spm.THETA_EE) < np.deg2rad(2), np.rad2deg(ang)
    # and the True-pixel cloud's principal axis is along theta too
    X, Y = g.coords()
    xs, ys = X[cm.mask], Y[cm.mask]
    Sxx = float(np.mean((xs - xs.mean()) ** 2))
    Syy = float(np.mean((ys - ys.mean()) ** 2))
    Sxy = float(np.mean((xs - xs.mean()) * (ys - ys.mean())))
    pang = 0.5 * np.arctan2(2 * Sxy, Sxx - Syy)
    assert abs(pang - spm.THETA_EE) < np.deg2rad(5), np.rad2deg(pang)


def test_ar1_kernel_is_isotropic():
    g = spm.Grid(n=48, L=12.0)
    # ell_perp enlarged so the geometry is well-resolved (kernel-builder orientation check, not the
    # physical sub-pixel width)
    K = spm.ee_kernel(g, ar=1.0, ell_perp=1.0)
    _ang, aspect = spm.kernel_principal_axis(K, g)
    assert abs(aspect - 1.0) < 0.05, aspect


def test_ar2_kernel_major_axis_is_45deg():
    g = spm.Grid(n=48, L=12.0)
    K = spm.ee_kernel(g, ar=2.0, ell_perp=1.0, theta=spm.THETA_EE)
    ang, aspect = spm.kernel_principal_axis(K, g)
    assert aspect > 1.5, aspect                          # elongated, not isotropic
    assert abs(ang - np.pi / 4) < np.deg2rad(3), np.rad2deg(ang)


# ---------------------------------------------------------------------------
# TDD-2: LIF transfer + local gain (transfer/gain wrappers real; q_global/q_core remain pending)
# ---------------------------------------------------------------------------
def test_phi_lif_monotonic_in_mu():
    sig = 5.0
    mus = np.linspace(0.0, 25.0, 30)
    for pop in ("E", "I"):
        rates = np.array([spm.phi_lif(m, sig, pop=pop) for m in mus])
        assert np.all(np.diff(rates) >= -1e-12), f"{pop} Phi not monotonic in mu"
        assert rates[-1] > rates[0]


def test_dphi_dmu_matches_finite_difference():
    from src.sef_hfo_lif import lif_gains, mean_field
    op = mean_field(1.0)
    # (a) the per-cell local gain reproduces the validated 0-D lif_gains at the op point
    g_val = lif_gains(op)
    assert abs(spm.local_gain(op["muE"], op["sE"], pop="E") - g_val["E"]) < 1e-9
    assert abs(spm.local_gain(op["muI"], op["sI"], pop="I") - g_val["I"]) < 1e-9
    # (b) the finite-difference slope is converged: two step sizes agree
    d1 = spm.dphi_dmu(op["muE"], op["sE"], pop="E", h=1e-3)
    d2 = spm.dphi_dmu(op["muE"], op["sE"], pop="E", h=5e-3)
    assert abs(d1 - d2) <= 1e-2 * abs(d1) + 1e-12


def test_gain_finite_under_low_and_high_drive():
    sig = 5.0
    for mu in (-5.0, 0.0, 10.0, 17.5, 30.0, 60.0):     # deep sub-threshold to supra-threshold
        for pop in ("E", "I"):
            assert np.isfinite(spm.local_gain(mu, sig, pop=pop)), (pop, mu)


def test_gain_nonnegative_in_valid_regime():
    sig = 5.0
    for mu in np.linspace(-5.0, 40.0, 40):
        for pop in ("E", "I"):
            gg = spm.local_gain(mu, sig, pop=pop)
            assert gg >= -1e-9, (pop, mu, gg)          # Phi increasing -> gain >= 0


# ---------------------------------------------------------------------------
# TDD-2 (cont): q_global / q_core inhibition scaling (real)
# ---------------------------------------------------------------------------
def test_q_global_reduces_effective_inhibition_in_expected_direction():
    g = spm.Grid(n=12, L=12.0)
    core = spm.make_core_mask(g, kind="none")
    ker = spm.build_kernels(g)
    exc = spm.build_excitability_field(g, core)
    # the field-level lever: lower q_global scales the effective I->E weight down everywhere
    inh_hi = spm.build_inhibition_field(g, core, q_global=1.0)
    inh_lo = spm.build_inhibition_field(g, core, q_global=0.9)
    assert np.allclose(inh_hi.q, 1.0) and np.allclose(inh_lo.q, 0.9)
    # the dynamical consequence: weaker inhibition -> higher steady E rate (gentle disinhibition;
    # this is an inhibition-stabilized net, so q<=0.8 already runs away to saturation)
    op_hi = spm.solve_operating_point(g, ker, exc, inh_hi, ratio=1.0)
    op_lo = spm.solve_operating_point(g, ker, exc, inh_lo, ratio=1.0)
    assert op_hi.converged and op_lo.converged
    assert op_lo.rE.mean() > op_hi.rE.mean(), (op_hi.rE.mean(), op_lo.rE.mean())


def test_q_core_affects_core_e_cells_only():
    g = spm.Grid(n=16, L=12.0)
    core = spm.make_core_mask(g, kind="single", radius=1.5)
    inh = spm.build_inhibition_field(g, core, q_global=1.0, q_core=0.4)
    assert np.allclose(inh.q[core.mask], 0.4), "core cells must carry q_global*q_core"
    assert np.allclose(inh.q[~core.mask], 1.0), "surround must be untouched by q_core"


# ---------------------------------------------------------------------------
# TDD-3: operating point (real)
# ---------------------------------------------------------------------------
def _no_core_op(n=12, **kw):
    g = spm.Grid(n=n, L=12.0)
    core = spm.make_core_mask(g, kind="none")
    ker = spm.build_kernels(g)
    exc = spm.build_excitability_field(g, core)
    inh = spm.build_inhibition_field(g, core)
    return g, spm.solve_operating_point(g, ker, exc, inh, **kw)


def test_no_core_operating_point_is_spatially_uniform():
    from src.sef_hfo_lif import mean_field
    _g, op = _no_core_op()
    assert op.converged
    assert float(op.rE.std()) < 1e-9 and float(op.rI.std()) < 1e-9, "no-core op must be uniform"
    mf = mean_field(1.0)
    assert abs(op.rE.mean() - mf["nuE"]) < 0.15 * mf["nuE"], (op.rE.mean(), mf["nuE"])


def test_core_excitability_raises_core_rE_or_lowers_threshold_effectively():
    g = spm.Grid(n=16, L=12.0)
    core = spm.make_core_mask(g, kind="single", radius=1.5)
    ker = spm.build_kernels(g)
    inh = spm.build_inhibition_field(g, core)
    exc0 = spm.build_excitability_field(g, core, mu_core=0.0)
    exc1 = spm.build_excitability_field(g, core, mu_core=0.6)
    op0 = spm.solve_operating_point(g, ker, exc0, inh, ratio=1.0)
    op1 = spm.solve_operating_point(g, ker, exc1, inh, ratio=1.0)
    assert op0.converged and op1.converged
    # core excitability raises the rate inside the core
    assert op1.rE[core.mask].mean() > op0.rE[core.mask].mean()
    assert op1.rE[core.mask].mean() > op1.rE[~core.mask].mean()


def test_operating_point_residual_below_tol_when_converged():
    _g, op = _no_core_op(tol=1e-9)
    assert op.converged
    assert op.residual < 1e-8, op.residual


def test_bad_params_return_unresolved_not_stable():
    # extreme recurrent gain has no clean self-limited rest: the solver must NOT return a clean op
    g = spm.Grid(n=12, L=12.0)
    core = spm.make_core_mask(g, kind="none")
    ker = spm.build_kernels(g)
    exc = spm.build_excitability_field(g, core)
    inh = spm.build_inhibition_field(g, core)
    op = spm.solve_operating_point(g, ker, exc, inh, ratio=1.0, w_ee_mult=10.0)
    assert op.status != "resolved", op.status        # unresolved or saturated, never silently stable


def test_operating_point_source_is_recorded():
    _g, op = _no_core_op(source="ratefield_steady")
    assert op.source == "ratefield_steady"
    _g2, op2 = _no_core_op(source="snn_baseline")
    assert op2.source == "snn_baseline"


def test_high_rate_saturation_is_flagged_not_axial():
    # strong disinhibition breaks the E/I balance -> runaway to the refractory ceiling (NOT high
    # external drive, which an inhibition-stabilized net absorbs by recruiting inhibition).
    g = spm.Grid(n=12, L=12.0)
    core = spm.make_core_mask(g, kind="none")
    ker = spm.build_kernels(g)
    exc = spm.build_excitability_field(g, core)
    inh = spm.build_inhibition_field(g, core, q_global=0.4)
    op = spm.solve_operating_point(g, ker, exc, inh, ratio=1.0)
    assert op.saturated, (op.rE.max(), op.status)
    assert op.status == "saturated"                  # downstream must not call this axial


# ---------------------------------------------------------------------------
# TDD-4: homogeneous Brunel-style dispersion sanity (real)
# ---------------------------------------------------------------------------
def test_lambda_k_returns_finite_values_on_small_k_grid():
    gE, gI = spm.homogeneous_gains(1.0, 1.5)
    disp = spm.homogeneous_dispersion(gE, gI, w_ee_mult=1.5, nk=11, kmax=3.0)
    assert disp["lambda_re"].shape == (11,) and disp["lambda_im"].shape == (11,)
    assert np.all(np.isfinite(disp["lambda_re"])) and np.all(np.isfinite(disp["lambda_im"]))
    assert disp["regime"] in {"stable", "candidate", "unstable"}


def test_lambda_k_symmetry_k_and_minus_k():
    gE, gI = spm.homogeneous_gains(1.0, 1.5)
    for kx, ky in [(1.2, 0.7), (0.0, 2.0), (2.5, -1.0)]:
        lam_p = spm.dispersion_lambda(kx, ky, gE, gI, w_ee_mult=1.5)
        lam_m = spm.dispersion_lambda(-kx, -ky, gE, gI, w_ee_mult=1.5)
        assert abs(lam_p - lam_m) < 1e-12, (kx, ky, lam_p, lam_m)


def test_ar1_dispersion_is_rotation_consistent_approximately():
    gE, gI = spm.homogeneous_gains(1.0, 1.5)
    k = 1.5
    growth = []
    for ang in np.linspace(0, np.pi, 6, endpoint=False):
        lam = spm.dispersion_lambda(k * np.cos(ang), k * np.sin(ang), gE, gI, w_ee_mult=1.5,
                                    ell_par=0.4, ell_perp=0.4)   # AR1: isotropic E->E
        growth.append(lam.real)
    assert np.ptp(growth) < 1e-9, growth     # isotropic kernel -> no direction dependence


def test_ar2_dispersion_prefers_expected_axis_when_anisotropy_present():
    # AR2 elongated E->E kernel (ell_par=0.54 > ell_perp=0.27): at finite k the growth is
    # anisotropic. A wavevector ACROSS the axis (= a spatial RIDGE ALONG the E->E axis) has the
    # larger E->E Fourier amplitude, so it grows faster -> the elongated scaffold favours ridges
    # along its long axis (plan §5 ridge-vs-wavevector 90deg relationship).
    gE, gI = spm.homogeneous_gains(1.0, 1.5)
    th, k = spm.THETA_EE, 1.5
    along = spm.dispersion_lambda(k * np.cos(th), k * np.sin(th), gE, gI, w_ee_mult=1.5,
                                  ell_par=0.54, ell_perp=0.27).real
    across = spm.dispersion_lambda(-k * np.sin(th), k * np.cos(th), gE, gI, w_ee_mult=1.5,
                                   ell_par=0.54, ell_perp=0.27).real
    assert across > along + 1e-3, (along, across)


def test_homogeneous_dispersion_json_schema():
    import json
    gE, gI = spm.homogeneous_gains(1.0, 1.5)
    disp = spm.homogeneous_dispersion(gE, gI, w_ee_mult=1.5, nk=9)
    required = {"k", "lambda_re", "lambda_im", "k_star", "re_max", "omega", "freq_Hz",
               "is_hopf", "regime", "along_axis"}
    assert required <= set(disp), required - set(disp)
    js = spm.dispersion_to_json(disp)
    json.dumps(js)                            # must be serializable
    assert isinstance(js["k"], list) and isinstance(js["re_max"], float)


# ---------------------------------------------------------------------------
# TDD-5: finite Jacobian builder / JVP (real)
# ---------------------------------------------------------------------------
def _no_core_jac(n=6, L=4.0, ell_perp=0.6, w_ee_mult=1.3):
    g = spm.Grid(n=n, L=L)
    ker = spm.build_kernels(g, ell_perp=ell_perp)
    core = spm.make_core_mask(g, kind="none")
    op = spm.solve_operating_point(g, ker, spm.build_excitability_field(g, core),
                                   spm.build_inhibition_field(g, core), ratio=1.0, w_ee_mult=w_ee_mult)
    return g, ker, op


def test_dense_jacobian_shape():
    g, ker, op = _no_core_jac()
    J = spm.build_jacobian_dense(g, ker, op)
    assert J.shape == (6 * g.size, 6 * g.size)


def test_linear_operator_matvec_matches_dense():
    g, ker, op = _no_core_jac()
    J = spm.build_jacobian_dense(g, ker, op)
    lo = spm.jacobian_linear_operator(g, ker, op)
    rng = np.random.default_rng(0)
    for _ in range(3):
        v = rng.standard_normal(6 * g.size)
        assert np.max(np.abs(J @ v - lo.matvec(v))) < 1e-10


def test_jvp_matches_finite_difference_tiny_grid():
    g, ker, op = _no_core_jac()
    J = spm.build_jacobian_dense(g, ker, op)
    zst = spm.op_state_vector(op, ker, g)
    rng = np.random.default_rng(3)
    v = rng.standard_normal(6 * g.size)
    # J is the Jacobian of the nonlinear field RHS at the op (sigma frozen). eps small enough to
    # stay within one piecewise-linear LUT cell of Phi.
    fd = (spm.field_rhs(zst + 1e-6 * v, g, ker, op) - spm.field_rhs(zst - 1e-6 * v, g, ker, op)) / 2e-6
    assert np.max(np.abs(J @ v - fd)) < 1e-7


def test_synaptic_blocks_have_expected_time_constant_signs():
    from src.sef_hfo_lif import TAU_AMPA, TAU_GABA, TAU_ME, TAU_MI
    g, ker, op = _no_core_jac()
    J = spm.build_jacobian_dense(g, ker, op)
    N = g.size

    def diag_blk(b):
        return np.diag(J[b * N:(b + 1) * N, b * N:(b + 1) * N])

    assert np.allclose(diag_blk(0), -1.0 / TAU_ME) and np.allclose(diag_blk(1), -1.0 / TAU_MI)
    assert np.allclose(diag_blk(2), -1.0 / TAU_AMPA) and np.allclose(diag_blk(4), -1.0 / TAU_AMPA)
    assert np.allclose(diag_blk(3), -1.0 / TAU_GABA) and np.allclose(diag_blk(5), -1.0 / TAU_GABA)


def test_inhibitory_blocks_have_expected_negative_effect_on_rE():
    g, ker, op = _no_core_jac()
    J = spm.build_jacobian_dense(g, ker, op)
    N = g.size
    GE_EE = np.diag(J[0:N, 2 * N:3 * N])          # rE <- sEE (recurrent E)
    GE_EI = np.diag(J[0:N, 3 * N:4 * N])          # rE <- sEI (I->E inhibition)
    GI_II = np.diag(J[1 * N:2 * N, 5 * N:6 * N])  # rI <- sII (I->I inhibition)
    assert np.all(GE_EE >= 0)                      # excitation promotes E growth
    assert np.all(GE_EI <= 0) and np.any(GE_EI < 0)   # inhibition depresses E growth
    assert np.all(GI_II <= 0)


def test_no_core_jacobian_eigs_match_homogeneous_dispersion_samples():
    g, ker, op = _no_core_jac(n=8)
    J = spm.build_jacobian_dense(g, ker, op)
    evJ = np.linalg.eigvals(J)
    gE, gI = float(op.gE.ravel()[0]), float(op.gI.ravel()[0])

    def mode_roots(mi, ni):
        return spm._rate_branch_roots(ker.ghat_EE[mi, ni], ker.ghat_I[mi, ni], gE, gI,
                                      wee=op.wee_mult * spm.W_EE, wEI=spm.W_EI, wII=spm.W_II)

    # the leading rate eigenvalue of J == the max-growth dispersion sample over grid modes (exact)
    lead_disp = max(mode_roots(mi, ni).real.max() for mi in range(g.n) for ni in range(g.n))
    lam, _vec, _p = spm.leading_rate_eigenpair(J, g)
    assert abs(lam.real - lead_disp) < 1e-9, (lam.real, lead_disp)
    # every grid-mode rate-branch root appears in eig(J) (loose: the defective synaptic-pole cluster
    # limits np.eigvals precision near -1/TAU_GABA)
    for mi in range(g.n):
        for ni in range(g.n):
            for r in mode_roots(mi, ni):
                assert np.min(np.abs(evJ - r)) < 1e-2


def test_core_excitability_increases_growth_of_core_overlap_mode():
    g = spm.Grid(n=10, L=5.0)
    ker = spm.build_kernels(g, ell_perp=0.6)
    core = spm.make_core_mask(g, kind="single", radius=0.9)
    inh = spm.build_inhibition_field(g, core)
    op0 = spm.solve_operating_point(g, ker, spm.build_excitability_field(g, core, mu_core=0.0),
                                    inh, ratio=1.0, w_ee_mult=1.3)
    op1 = spm.solve_operating_point(g, ker, spm.build_excitability_field(g, core, mu_core=0.8),
                                    inh, ratio=1.0, w_ee_mult=1.3)
    lam0, v0, _ = spm.leading_rate_eigenpair(spm.build_jacobian_dense(g, ker, op0), g)
    lam1, v1, _ = spm.leading_rate_eigenpair(spm.build_jacobian_dense(g, ker, op1), g)
    assert lam1.real > lam0.real                                       # raises growth
    assert spm.core_overlap(v1, g, core) > spm.core_overlap(v0, g, core)  # of a more core-localized mode


# ---------------------------------------------------------------------------
# TDD-6: eigenpair extraction with left/right modes (real)
# ---------------------------------------------------------------------------
def _core_jac(n=8, L=5.0, ell_perp=0.6, mu_core=0.8, w_ee_mult=1.4):
    g = spm.Grid(n=n, L=L)
    ker = spm.build_kernels(g, ell_perp=ell_perp)
    core = spm.make_core_mask(g, kind="single", radius=0.9)
    op = spm.solve_operating_point(g, ker, spm.build_excitability_field(g, core, mu_core=mu_core),
                                   spm.build_inhibition_field(g, core), ratio=1.0, w_ee_mult=w_ee_mult)
    J = spm.build_jacobian_dense(g, ker, op)
    return g, ker, core, op, J


def test_right_eigen_residual_norm_small():
    g, _ker, _core, _op, J = _core_jac()
    res = spm.rate_eigenpairs(J, g, n_modes=4)
    assert res.status == "resolved"
    assert res.residual_right < 1e-8, res.residual_right


def test_left_eigen_residual_norm_small():
    g, _ker, _core, _op, J = _core_jac()
    res = spm.rate_eigenpairs(J, g, n_modes=4)
    assert res.residual_left < 1e-8, res.residual_left


def test_complex_conjugate_pairs_are_handled():
    g, _ker, _core, _op, J = _core_jac()
    ev = np.linalg.eigvals(J)
    cplx = ev[np.abs(ev.imag) > 1e-9]
    assert cplx.size > 0                                  # the op has oscillatory (Hopf-like) modes
    for c in cplx:
        assert np.min(np.abs(ev - np.conj(c))) < 1e-9     # every complex eig has its conjugate
    res = spm.rate_eigenpairs(J, g, n_modes=4)
    assert np.iscomplexobj(res.eigenvalues) and res.residual_right < 1e-8


def test_modes_sorted_by_real_part():
    g, _ker, _core, _op, J = _core_jac()
    res = spm.rate_eigenpairs(J, g, n_modes=5)
    assert np.all(np.diff(res.eigenvalues.real) <= 1e-12), res.eigenvalues


def test_left_right_biorthogonality_after_normalization():
    g, _ker, _core, _op, J = _core_jac()
    res = spm.rate_eigenpairs(J, g, n_modes=4)
    assert res.biorthogonality_error < spm.BIORTHOGONALITY_TOL, res.biorthogonality_error


def test_unstable_or_failed_eigs_mark_unresolved():
    g, _ker, _core, _op, J = _core_jac()
    Jbad = J.copy()
    Jbad[0, 0] = np.nan
    assert spm.rate_eigenpairs(Jbad, g).status == "unresolved"


# ---------------------------------------------------------------------------
# TDD-7: synthetic-mode metrics + classifier (real)
# ---------------------------------------------------------------------------
def _axis_fields(n=24, L=12.0):
    g = spm.Grid(n=n, L=L)
    th = spm.THETA_EE
    X, Y = g.coords()
    u = np.cos(th) * X + np.sin(th) * Y          # along the E->E axis
    v = -np.sin(th) * X + np.cos(th) * Y         # across the axis
    return g, X, Y, u, v


def test_core_localized_mode_has_high_core_overlap_low_globality():
    g, X, Y, u, v = _axis_fields()
    core = spm.make_core_mask(g, kind="single", radius=1.5)
    blob = np.exp(-(X ** 2 + Y ** 2) / (2 * 0.8 ** 2))
    assert spm.core_overlap(blob, g, core) > 0.8
    assert spm.globality(blob, g) < 0.1


def test_axial_elongated_ridge_has_high_elongation_axis_score():
    g, X, Y, u, v = _axis_fields()
    ridge = np.exp(-(v ** 2) / (2 * 0.6 ** 2) - (u ** 2) / (2 * 4.0 ** 2))   # tube ALONG the axis
    iso = np.exp(-(X ** 2 + Y ** 2) / (2 * 2.0 ** 2))
    assert spm.elongation_axis_score(ridge, g) > 0.7
    assert abs(spm.elongation_axis_score(iso, g)) < 0.1


def test_phase_gradient_wave_has_expected_phase_gradient_axis_score():
    g, X, Y, u, v = _axis_fields()
    assert spm.phase_gradient_axis_score(np.exp(1j * 2.0 * u), g) > 0.5      # wavevector along axis
    assert spm.phase_gradient_axis_score(np.exp(1j * 2.0 * v), g) < -0.5     # wavevector across axis


def test_global_low_k_mode_has_high_participation_ratio():
    g, X, Y, u, v = _axis_fields()
    assert spm.globality(np.ones((g.n, g.n)), g) > 0.9
    assert spm.globality(np.exp(-(X ** 2 + Y ** 2) / (2 * 0.8 ** 2)), g) < 0.2


def test_off_axis_mode_is_not_called_axial():
    g, X, Y, u, v = _axis_fields()
    core = spm.make_core_mask(g, kind="single", radius=1.5)
    offridge = np.exp(-(u ** 2) / (2 * 0.6 ** 2) - (v ** 2) / (2 * 4.0 ** 2))   # tube ACROSS the axis
    assert spm.off_axis_score(offridge, g) > 0.7
    cls = spm.classify_mode(growth=0.01, core_overlap_=spm.core_overlap(offridge, g, core),
                            globality_=spm.globality(offridge, g),
                            elongation_axis=spm.elongation_axis_score(offridge, g),
                            off_axis=spm.off_axis_score(offridge, g))
    assert cls != "axial", cls


def test_axis_score_handles_90deg_wavevector_vs_ridge_ambiguity():
    g, X, Y, u, v = _axis_fields()
    # a ridge ALONG the axis carries a wavevector ACROSS the axis (90 deg) -> the two scores must
    # disagree, not be silently collapsed (plan §5)
    ridge = np.exp(-(v ** 2) / (2 * 0.6 ** 2) - (u ** 2) / (2 * 4.0 ** 2))
    elong = spm.elongation_axis_score(ridge, g)
    pg = spm.phase_gradient_axis_score(ridge, g)
    assert elong > 0.5 and pg < -0.5 and elong * pg < 0, (elong, pg)


def test_non_normal_toy_has_negative_alpha_but_high_finite_time_gain():
    M = np.array([[-0.1, 8.0], [0.0, -0.2]])
    assert np.max(np.linalg.eigvals(M).real) < 0          # every eigenvalue stable
    assert spm.transient_gain(M, [0.0, 1.0], 2.0) > 2.0   # yet large non-normal transient growth


def test_core_controllability_uses_left_not_right_eigenvector():
    g, _ker, core, _op, J = _core_jac()
    res = spm.rate_eigenpairs(J, g, n_modes=4)
    ctrl_left = spm.core_controllability(res.left[:, 0], g, core)
    ctrl_right = spm.core_controllability(res.right[:, 0], g, core)
    # the operator is non-normal: left and right eigenvectors give different controllability
    assert abs(ctrl_left - ctrl_right) > 1e-3, (ctrl_left, ctrl_right)


# ---------------------------------------------------------------------------
# TDD-8: single-point golden cases (real)
# ---------------------------------------------------------------------------
def _gp(mu_core, q_global, w_ee_mult=1.3, n=8, L=5.0):
    g = spm.Grid(n=n, L=L)
    ker = spm.build_kernels(g, ell_perp=0.6)
    core = spm.make_core_mask(g, kind="single", radius=0.9)
    return spm.analyze_spectral_point(g, ker, core, mu_core=mu_core, q_global=q_global,
                                      w_ee_mult=w_ee_mult)


def test_low_excitability_point_is_stable_or_local():
    p = _gp(0.0, 1.0, w_ee_mult=1.0)
    assert p.mode_class in {"stable", "local"}, p.mode_class


def test_increasing_core_excitability_raises_core_overlap_or_growth():
    p0, p1 = _gp(0.0, 1.0), _gp(0.8, 1.0)
    assert (p1.core_overlap > p0.core_overlap) or (p1.alpha_1 > p0.alpha_1)


def test_increasing_global_disinhibition_raises_globality_or_low_k_energy():
    # the low-k/global mode approaches instability as inhibition weakens (q drops)
    p0, p1 = _gp(0.0, 1.0), _gp(0.0, 0.95)
    assert (p1.globality > p0.globality) or (p1.alpha_1 > p0.alpha_1), (p0.alpha_1, p1.alpha_1)


def test_high_rate_saturation_is_flagged_runaway_not_axial():
    p = _gp(0.0, 0.3, w_ee_mult=1.4)
    assert p.mode_class == "runaway" and p.mode_class != "axial"


# ---------------------------------------------------------------------------
# TDD-9: pilot phase map (real)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pilot_map():
    g = spm.Grid(n=8, L=5.0)
    ker = spm.build_kernels(g, ell_perp=0.6)
    core = spm.make_core_mask(g, kind="single", radius=0.9)
    return spm.build_phase_map(g, ker, core, x_values=[0.0, 0.5, 1.0],
                               y_values=[1.0, 0.97, 0.94], w_ee_mult=1.3)


def test_phase_map_3x3_runs_to_completion(pilot_map):
    assert len(pilot_map) == 9


def test_all_grid_points_have_status(pilot_map):
    valid_cls = {"stable", "local", "axial", "mixed", "global", "runaway", "unresolved"}
    for p in pilot_map:
        assert p.op_status in {"resolved", "saturated", "unresolved"}
        assert p.mode_class in valid_cls


def test_unresolved_fraction_below_threshold(pilot_map):
    assert spm.unresolved_fraction(pilot_map) < 0.5


def test_mode_metrics_csv_has_required_columns(pilot_map):
    rows = spm.phase_map_to_rows(pilot_map)
    assert len(rows) == len(pilot_map)
    for r in rows:
        assert set(spm.MODE_METRICS_COLUMNS) <= set(r)


def test_phase_map_has_nontrivial_variation_in_alpha_or_mode_class(pilot_map):
    classes = {p.mode_class for p in pilot_map}
    alphas = [p.alpha_1 for p in pilot_map if p.alpha_1 == p.alpha_1]
    assert len(classes) >= 2 or (len(alphas) >= 2 and np.ptp(alphas) > 1e-3)


# ---------------------------------------------------------------------------
# TDD-10: controls / ablations (real)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def controls_summary():
    g = spm.Grid(n=8, L=5.0)
    return spm.run_controls(g, mu_core=0.8, w_ee_mult=1.3)


def test_no_core_does_not_reproduce_core_localized_story(controls_summary):
    s = controls_summary
    assert abs(s["no_core"]["core_localization"]) < 1e-6        # no core excitability -> no localization
    assert s["core"]["core_localization"] > s["no_core"]["core_localization"] + 0.01


def test_ar1_weakens_or_removes_45deg_axial_preference(controls_summary):
    s = controls_summary
    assert abs(s["ar1_isotropic"]["dispersion_anisotropy"]) < 1e-6      # isotropic -> no axis preference
    assert s["ar2_anisotropic"]["dispersion_anisotropy"] > s["ar1_isotropic"]["dispersion_anisotropy"]


def test_off_axis_core_rotates_or_weakens_axis_score_as_expected(controls_summary):
    s = controls_summary["off_axis_core"]
    # the off-axis core still gathers power at ITS location (localization follows the core geometry)
    assert s["core_localization_at_offcore"] > 0.01
    # ... but the E->E scaffold reinforces an on-axis core more, so it is weaker than on-axis
    assert s["core_localization_at_offcore"] < s["core_localization_on_axis_ref"]


def test_shuffled_core_thresholds_do_not_create_same_clean_axis_consistently(controls_summary):
    s = controls_summary["shuffled_core"]
    # scattered core cells localize less coherently than a contiguous scaffold
    assert s["core_localization_mean"] < s["contiguous_core_localization"]


def test_controls_summary_contains_all_required_controls(controls_summary):
    assert set(spm.REQUIRED_CONTROLS) <= set(controls_summary)


# ---------------------------------------------------------------------------
# TDD-11: rate-field dynamic spot checks (real)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rf_responses():
    g = spm.Grid(n=24, L=10.0)
    core = spm.make_core_mask(g, kind="single", radius=1.2)
    ker2 = spm.build_kernels(g, ell_perp=0.7)               # AR2 anisotropic scaffold
    ker1 = spm.build_kernels(g, ar=1.0, ell_perp=0.7)       # AR1 isotropic control
    exc0 = spm.build_excitability_field(g, core, mu_core=0.0)
    inh = spm.build_inhibition_field(g, core)
    ar2 = spm.simulate_ratefield_response(g, ker2, exc0, inh, w_ee_mult=1.0, stim_amp=8.0)
    ar1 = spm.simulate_ratefield_response(g, ker1, exc0, inh, w_ee_mult=1.0, stim_amp=8.0)
    exc_hi = spm.build_excitability_field(g, core, mu_core=0.7)
    runaway = spm.simulate_ratefield_response(g, ker2, exc_hi, inh, w_ee_mult=1.4, stim_amp=8.0)
    return ar2, ar1, runaway


def test_axial_spectral_point_produces_axial_ratefield_response(rf_responses):
    ar2, ar1, _ = rf_responses
    # the anisotropic E->E scaffold makes the kick response elongate along the axis; the isotropic
    # control does not (Round-1 recovered this 45deg axis from a kick-driven rate field)
    assert ar2.response_axis_score > 0.2, ar2.response_axis_score
    assert ar2.response_axis_score > ar1.response_axis_score


def test_global_spectral_point_produces_higher_active_fraction(rf_responses):
    ar2, _ar1, runaway = rf_responses
    assert runaway.max_active > ar2.max_active    # global recruitment lights up more than self-limited


def test_stable_point_returns_to_baseline_after_pulse(rf_responses):
    ar2, _ar1, _ = rf_responses
    assert ar2.max_active > 0.0 and ar2.returned   # the kick does excite, then returns to baseline


def test_runaway_risk_point_is_flagged_if_no_return(rf_responses):
    _ar2, _ar1, runaway = rf_responses
    assert (not runaway.returned) and runaway.max_active > spm._RUNAWAY_FRAC


# ---------------------------------------------------------------------------
# TDD-12: SNN frozen-state spot checks (real)
# ---------------------------------------------------------------------------
def test_snn_param_mapping_is_documented_for_each_point():
    m = spm.snn_param_mapping(mu_core=0.5, q_global=0.9, w_ee_mult=1.3)
    assert m["g"] < spm._SNN_BASE_G               # q<1 -> disinhibition
    assert m["w_EE"] > spm._SNN_BASE_W_EE         # w_ee_mult>1 -> stronger recurrent E
    assert m["core_v_th"] < spm._SNN_BASE_V_TH    # mu_core lowers core threshold
    assert "transforms" in m and len(m["transforms"]) == 3


def test_axial_point_not_systematically_r4b():
    # a self-limited local event (returns to baseline, low recruitment) must NOT be called R4b
    axial = dict(baseline=4.0, peak=40.0, tail=4.0, returned=True, inside=200.0, outside=50.0,
                 ratio=0.25, peak_active_frac=0.1)
    assert spm.classify_snn_r_class(axial) != "R4b"
    runaway = dict(baseline=4.0, peak=600.0, tail=600.0, returned=False, inside=500.0,
                   outside=1200.0, ratio=2.4, peak_active_frac=1.0)
    assert spm.classify_snn_r_class(runaway) == "R4b"


def test_global_risk_point_has_higher_active_mass_than_axial_point():
    glob = dict(baseline=4.0, peak=300.0, tail=8.0, returned=True, inside=300.0, outside=900.0,
                ratio=3.0, peak_active_frac=0.8)
    axial = dict(baseline=4.0, peak=40.0, tail=4.0, returned=True, inside=200.0, outside=50.0,
                 ratio=0.25, peak_active_frac=0.15)
    assert spm.snn_active_mass(glob) > spm.snn_active_mass(axial)


def test_systematic_spectrum_snn_mismatch_triggers_stop_status():
    stop = spm.snn_spectrum_consistency([("axial", "R4b"), ("local", "R4b"), ("stable", "R4a")])
    assert stop["status"] == "stop"
    ok = spm.snn_spectrum_consistency([("axial", "R2"), ("global", "R4a")])
    assert ok["status"] == "ok"


def test_snn_recruitment_axis_detects_axial_vs_isotropic():
    n = 12
    xs = np.repeat(np.linspace(-1, 1, n), n)
    ys = np.tile(np.linspace(-1, 1, n), n)
    pos_E = np.column_stack([xs, ys])
    th = spm.THETA_EE
    v = -np.sin(th) * xs + np.cos(th) * ys
    axial_mask = np.abs(v) < 0.2                       # a tube ALONG the 45deg axis
    iso_mask = (xs ** 2 + ys ** 2) < 0.3               # a central isotropic blob
    dt = 0.1
    nsteps = int(200 / dt)

    def spk(mask):
        b = np.zeros((nsteps, pos_E.shape[0]), bool)
        b[int(150 / dt):int(180 / dt), mask] = True
        return b

    assert spm._snn_recruitment_axis(spk(axial_mask), pos_E, dt, 150, 180) > 0.3
    assert abs(spm._snn_recruitment_axis(spk(iso_mask), pos_E, dt, 150, 180)) < 0.15


@pytest.mark.slow
def test_snn_runs_emit_classification_fields():
    # one real (tiny) SNN frozen-state pilot must run end-to-end and emit the R-class + metric fields
    rec = spm.run_snn_spotcheck(mu_core=0.3, q_global=1.0, w_ee_mult=1.0)
    assert set(spm.SNN_SPOTCHECK_FIELDS) <= set(rec)
    assert rec["R_class"] in spm.SNN_R_CLASSES
    assert "recruitment_axis" in rec


@pytest.mark.slow
def test_snn_spotcheck_grid_runs_and_classifies():
    g = spm.run_snn_spotcheck_grid([(0.0, 1.0), (0.6, 1.0)], seeds=(1,), kick_mult=1.0,
                                   spectral_mode_classes={"0.0,1.0": "axial", "0.6,1.0": "axial"})
    assert g["n_points"] == 2 and set(g["per_point"]) == {"0.0,1.0", "0.6,1.0"}
    for pp in g["per_point"].values():
        assert pp["modal_R_class"] in spm.SNN_R_CLASSES and "mean_recruitment_axis" in pp
    assert "snn_grid_pass_axial" in g and g["consistency"]["status"] in {"ok", "stop"}


# ---------------------------------------------------------------------------
# TDD-13: mode / event readout projection (real)
# ---------------------------------------------------------------------------
def _axial_field(g):
    th = spm.THETA_EE
    X, Y = g.coords()
    u = np.cos(th) * X + np.sin(th) * Y
    v = -np.sin(th) * X + np.cos(th) * Y
    return np.exp(-(v ** 2) / (2 * 0.8 ** 2) - (u ** 2) / (2 * 3.0 ** 2))


def test_mock_mode_to_virtual_seeg_record_schema():
    g = spm.Grid(n=24, L=10.0)
    rec = spm.project_mode_to_record(_axial_field(g), g)
    assert {"dataset", "subject", "template_id", "channels", "n_channels", "scalars", "flags"} <= set(rec)
    assert rec["n_channels"] >= 6
    assert {"axis_length_mm", "transverse_width_mm", "rank_vs_xnorm_spearman"} <= set(rec["scalars"])
    assert len(rec["channels"]) == rec["n_channels"]


def test_real_mode_projection_has_required_scalars_and_channels():
    g, _ker, _core, _op, J = _core_jac()
    res = spm.rate_eigenpairs(J, g, n_modes=2)
    eE = np.abs(spm.mode_e_field(res.right[:, 0], g))
    rec = spm.project_mode_to_record(eE, g)
    assert rec.get("n_channels", 0) >= 6
    assert "scalars" in rec and len(rec.get("channels", [])) > 0


def test_snn_event_projection_uses_same_masked_readout_conventions():
    g = spm.Grid(n=24, L=10.0)
    X, Y = g.coords()
    blob = np.exp(-(X ** 2 + Y ** 2) / (2 * 0.8 ** 2))     # localized -> many non-participants
    _rec, interm = spm.project_mode_to_record(blob, g, return_intermediates=True)
    nonpart = ~interm["bools"].ravel()
    assert nonpart.any()
    # non-participating contacts carry NaN ranks (no phantom integer ranks) — the masked convention
    assert np.all(np.isnan(interm["masked_ranks"].ravel()[nonpart]))


def test_compare_model_to_cohort_runs_without_schema_adapter_hacks():
    from src.propagation_contact_plane_readout import compare_model_to_cohort, make_plane_grid
    g = spm.Grid(n=24, L=10.0)
    model = spm.project_mode_to_record(_axial_field(g), g, model_id="model")
    reals = [spm.project_mode_to_record(_axial_field(g) * (1 + 0.05 * k), g, model_id=f"s{k}")
             for k in range(3)]
    X, Y = make_plane_grid()
    res = compare_model_to_cohort(model, reals, X, Y)
    assert "field_placement" in res and "scalar_placement" in res


def test_geometry_null_failure_forces_placement_only_verdict():
    assert spm.readout_bridge_verdict("failed") == "placement_only"
    assert spm.readout_bridge_verdict("passed") == "bridge"
    # 'not_run' is distinct from 'failed': only the schema/projection connected, no placement computed
    assert spm.readout_bridge_verdict("not_run") == "projection_only"


# ---------------------------------------------------------------------------
# TDD-15: figures / verdict / claim audit (real)
# ---------------------------------------------------------------------------
_OUT = Path(__file__).resolve().parents[1] / "results/topic4_sef_hfo/m3b_spectral_phase_map"
_FIG = _OUT / "figures"
_REQUIRED_ARTIFACTS = (
    "STATUS.md", "homogeneous_dispersion.json", "finite_jacobian_grid.json", "mode_metrics.csv",
    "control_summary.json", "mode_readout_projection.json", "m3a_interface_audit.json",
    "snn_spotcheck_summary.json", "ratefield_spotcheck_summary.json",
)
_REQUIRED_FIGURES = (
    "homogeneous_dispersion.png", "example_modes.png", "phase_map_mode_class.png",
    "phase_map_gap_gain.png", "non_normal_gain_controllability.png", "mode_readout_projection.png",
    "snn_spotcheck_grid.png", "slow_trajectory_overlay.png",
)


def test_required_artifacts_exist():
    import json
    for art in _REQUIRED_ARTIFACTS:
        assert (_OUT / art).exists(), f"missing artifact {art} (run scripts/build_m3b_spectral_outputs.py)"
    # the M3A overlay artifact is correctly the REFUSED audit (M3A absent); the overlay CSV is absent
    audit = json.loads((_OUT / "m3a_interface_audit.json").read_text(encoding="utf-8"))
    assert audit.get("overlay_verdict") == "refused"


def test_required_figures_exist_or_are_marked_na():
    readme = (_FIG / "README.md").read_text(encoding="utf-8")
    for fig in _REQUIRED_FIGURES:
        assert (_FIG / fig).exists() or fig in readme, f"figure {fig} neither generated nor marked N/A"


def test_verdict_category_is_one_of_allowed_values():
    status = (_OUT / "STATUS.md").read_text(encoding="utf-8")
    assert any(v in status for v in spm.ALLOWED_VERDICTS)
    # this run's evidence: specific controls + §5 non-normal axial substrate present; no SNN grid /
    # M3A overlay / readout-null bridge -> frozen-map tier
    v = spm.m3b_verdict(phase_map_resolved=True, model_matches_dynamics=True, controls_pass=True,
                        non_normal_axial_pass=True, snn_grid_pass=False, m3a_overlay_pass=False,
                        readout_null_pass=False)
    assert v in spm.ALLOWED_VERDICTS and v == "SPM-PASS frozen map"


def test_verdict_is_fail_closed_on_controls_and_axial():
    base = dict(phase_map_resolved=True, model_matches_dynamics=True, controls_pass=True,
                non_normal_axial_pass=True, snn_grid_pass=False, m3a_overlay_pass=False,
                readout_null_pass=False)
    # a control failure CANNOT return a PASS tier (fail-closed; controls_pass is load-bearing)
    assert spm.m3b_verdict(**{**base, "controls_pass": False}) == "SPM-BOUNDED negative"
    # a missing §5 axial substrate CANNOT return a PASS tier
    assert spm.m3b_verdict(**{**base, "non_normal_axial_pass": False}) == "SPM-BOUNDED negative"
    # the bridge tiers each need their own gate
    assert spm.m3b_verdict(**{**base, "snn_grid_pass": True}) == "SPM-PASS spontaneous mechanism"
    assert spm.m3b_verdict(**{**base, "snn_grid_pass": True, "m3a_overlay_pass": True,
                              "readout_null_pass": True}) == "SPM-PASS full bridge"


def test_full_bridge_requires_phase_map_snn_m3a_readout_null_pass():
    full = dict(phase_map_coherent=True, snn_predicts_spotchecks=True, m3a_trajectory_valid=True,
                readout_null_pass=True)
    assert spm.full_bridge_gate(**full) == "SPM-PASS full bridge"
    for k in ("phase_map_coherent", "snn_predicts_spotchecks", "m3a_trajectory_valid", "readout_null_pass"):
        partial = dict(full)
        partial[k] = False
        assert spm.full_bridge_gate(**partial) != "SPM-PASS full bridge"


def test_no_forbidden_claims_in_status_and_readme():
    status = (_OUT / "STATUS.md").read_text(encoding="utf-8")
    readme = (_FIG / "README.md").read_text(encoding="utf-8")
    assert "## Forbidden claims" in status                      # guardrail stays frozen
    # neither the verdict prose nor the figure README overclaims a full bridge / seizure proof
    for bad in ("proves clinical seizure", "proves seizure onset", "证明发作机制成立", "full bridge established"):
        assert bad not in readme
    assert any(v in status for v in spm.ALLOWED_VERDICTS)       # the written verdict is an allowed category


# ---------------------------------------------------------------------------
# Path (a): non-normal transient axial readout (§5 PRIMARY metric)
# ---------------------------------------------------------------------------
def _resolved_core_jac(mu_core=0.6, ell_perp=0.6, ar=2.0):
    g = spm.Grid(n=10, L=5.0)
    ker = spm.build_kernels(g, ar=ar, ell_perp=ell_perp)
    core = spm.make_core_mask(g, kind="single", radius=0.9)
    op = spm.solve_operating_point(g, ker, spm.build_excitability_field(g, core, mu_core=mu_core),
                                   spm.build_inhibition_field(g, core), ratio=1.0, w_ee_mult=1.3)
    assert op.status == "resolved"
    return g, ker, core, spm.build_jacobian_dense(g, ker, op)


def test_core_kick_transiently_amplified_and_self_limited():
    g, _ker, core, J = _resolved_core_jac()
    r = spm.non_normal_axial_readout(J, g, core)
    # non-normal: a core kick is amplified above 1 then decays back (self-limited), even though the
    # leading eigenvalue is stable/global
    assert r["transient_amplified"] and r["peak_gain"] > 1.2
    assert r["self_limited"]


def test_core_kick_transient_is_axial_but_leading_mode_is_not():
    g, ker, core, J = _resolved_core_jac()
    r = spm.non_normal_axial_readout(J, g, core)
    assert r["axial"] and r["max_axis"] > 0.2          # the transient spreads along the E->E axis
    # ... while the leading eigenmode itself is NOT axial (the signal is in the transient, not the mode)
    res = spm.rate_eigenpairs(J, g, n_modes=2)
    assert abs(spm.elongation_axis_score(spm.mode_e_field(res.right[:, 0], g), g)) < 0.1


def test_isotropic_scaffold_has_no_axial_transient():
    # the axial transient is scaffold-specific: an isotropic (AR1) E->E kernel gives no axial transient
    g, _ker, core, J = _resolved_core_jac(ar=1.0)
    r = spm.non_normal_axial_readout(J, g, core)
    assert not r["axial"] and r["max_axis"] < 0.05
