"""TDD tests for topic4_criticality.load_crit_config (Task 0, plan Step 2).

Contract (verbatim from task brief .superpowers/sdd/task-0-brief.md Step 2):
  - load_crit_config(path=None) loads config/topic4_criticality.yaml by default.
  - Locks units, verdict threshold-sweep, quality-gate floors, and the
    slow_to_ratefield terminology block (review additions #4/#8/#9/#15/P1-1).
"""
import pytest

from src.topic4_criticality import load_crit_config


# --- verbatim from task-0-brief.md Step 2 ---

def test_config_locks_units_verdicts_and_review_additions():
    c = load_crit_config()
    assert c["operator"]["alpha_units"] == "per_ms"
    assert c["verdict"]["alpha_near_zero_tol_per_ms"] == 0.002          # #4 per-ms, low default
    assert "threshold_sweep" in c["verdict"]                             # #4
    assert c["quality_gate"]["rate_scale_floor"] > 0                     # #8
    assert c["branching"]["branch_cluster_field_tol"] > 0                # #9
    assert set(c["slow_to_ratefield"]) == {"q_I", "g_K", "h_G"}          # P1-1
    assert c["finite_time_gain"]["mode"] == "directional_core"          # #15


# --- Task 2 review #1 gap-fill: _invert_phase_transform's only prior coverage was via
# build_conditional_atlas in test_topic4_crit_integration.py, which is @pytest.mark.integration
# + figdata-gated (silently skipped without the gitignored results/ tree). This is a fast,
# no-figdata characterization test of the algebraic inverse itself. ---

def test_invert_phase_transform_round_trips_apply_transform():
    from src.sef_hfo_m3_interface import _apply_transform
    from src.sef_hfo_m3a_export import default_precalib_mapping_and_ranges
    from src.topic4_criticality import _invert_phase_transform

    mapping, _ranges = default_precalib_mapping_and_ranges("m3a_v2_2_approach")
    recip_t = mapping["coordinates"]["phase_x_core"]["transform"]   # real coeffs: a=1/3, b=-1/3
    affine_t = {"type": "affine", "a": 2.0, "b": 0.1, "clip": [0.0, 1.0]}
    identity_t = {"type": "identity"}

    cases = [
        (identity_t, [0.0, 0.3, 0.7, 1.0]),
        (affine_t, [0.1, 0.25, 0.4]),        # a*x+b stays inside [0,1]: no clip distortion
        (recip_t, [0.3, 0.5, 0.75, 1.0]),    # q_core in [0.25,1] -> phase in [0,1]: no clip
    ]
    for transform, xs in cases:
        for x in xs:
            phase = _apply_transform(transform, x)
            assert _invert_phase_transform(transform, phase) == pytest.approx(x, abs=1e-9)


# --- Task 2.5: slow_to_ratefield -- g_K/h_G wired into solve_operating_point's muE (#P1-1 sign
# lock). Brief .superpowers/sdd/task-2.5-brief.md Step 1 test translated to the REAL m3b API per
# the controller's VERIFIED T2.5 notes (.superpowers/sdd/progress.md): ExcitabilityField.from_core /
# InhibitionField.uniform / positional make_core_mask do not exist; use build_excitability_field /
# build_inhibition_field / make_core_mask(kind=, radius=). EigResult.eigenvalues is correct as-is. ---

def test_slow_to_ratefield_h_G_sign_lowers_excitability():
    """Raising h_G (global recovery current) must not raise alpha_1 or mean rE (#P1-1)."""
    from src.topic4_m3b_spectral_phase import (
        Grid, build_kernels, make_core_mask, build_excitability_field, build_inhibition_field,
        solve_operating_point, build_jacobian_dense, rate_eigenpairs,
    )
    g = Grid(n=6, L=5.0)
    k = build_kernels(g, ell_perp=0.6)
    core = make_core_mask(g, kind="single", radius=0.9)
    exc = build_excitability_field(g, core, mu_core=1.0)
    inh = build_inhibition_field(g, core, q_global=0.94)

    op0 = solve_operating_point(g, k, exc, inh)
    a0 = rate_eigenpairs(build_jacobian_dense(g, k, op0), g).eigenvalues[0].real

    hi = solve_operating_point(g, k, exc, inh, hG_scalar=2.0, eta_G=1.0)  # more global recovery current
    a1 = rate_eigenpairs(build_jacobian_dense(g, k, hi), g).eigenvalues[0].real

    assert a1 <= a0 + 1e-9        # h_G suppresses E -> alpha1 not higher
    assert hi.rE.mean() <= op0.rE.mean() + 1e-9


def test_slow_to_ratefield_g_K_sign_lowers_excitability():
    """Raising g_K (per-cell K-current field), the newly-wired path alongside h_G, must not raise
    alpha_1 or mean rE (#P1-1)."""
    import numpy as np
    from src.topic4_m3b_spectral_phase import (
        Grid, build_kernels, make_core_mask, build_excitability_field, build_inhibition_field,
        solve_operating_point, build_jacobian_dense, rate_eigenpairs,
    )
    g = Grid(n=6, L=5.0)
    k = build_kernels(g, ell_perp=0.6)
    core = make_core_mask(g, kind="single", radius=0.9)
    exc = build_excitability_field(g, core, mu_core=1.0)
    inh = build_inhibition_field(g, core, q_global=0.94)

    op0 = solve_operating_point(g, k, exc, inh)
    a0 = rate_eigenpairs(build_jacobian_dense(g, k, op0), g).eigenvalues[0].real

    hi = solve_operating_point(g, k, exc, inh, gK_field=np.full((g.n, g.n), 2.0), eta_K=1.0)
    a1 = rate_eigenpairs(build_jacobian_dense(g, k, hi), g).eigenvalues[0].real

    assert a1 <= a0 + 1e-9        # g_K suppresses E -> alpha1 not higher
    assert hi.rE.mean() <= op0.rE.mean() + 1e-9


def test_slow_to_ratefield_sign_ok_all_three_pass():
    """The #P1-1 sign-test deliverable: slow_to_ratefield_sign_ok reads eta_K/eta_G off
    load_crit_config()'s slow_to_ratefield block and confirms q_I (already-wired via inh.q),
    g_K, and h_G all lower excitability when raised. Structured per-var dict (user P1-3)
    distinguishes q_I's rate-only criterion from g_K/h_G's strict alpha_1+rate criterion."""
    from src.topic4_criticality import slow_to_ratefield_sign_ok, load_crit_config

    r = slow_to_ratefield_sign_ok(load_crit_config())
    assert r["q_I"]["rate_not_higher"] is True
    assert r["q_I"]["alpha1_not_required"] is True
    assert r["g_K"]["rate_not_higher"] is True
    assert r["g_K"]["alpha1_not_higher"] is True
    assert r["h_G"]["rate_not_higher"] is True
    assert r["h_G"]["alpha1_not_higher"] is True


# --- Task 3a-1: operating-point quality gate -- rate_mismatch (abs+rel floor), adiabatic_index,
# and qualify_point's fail-closed missing-field + alpha-drift gates. Verbatim from
# .superpowers/sdd/task-3a-1-brief.md Step 1. ---

def test_quality_gate_all_review_fixes():
    import numpy as np
    from src.topic4_criticality import rate_mismatch, adiabatic_index, qualify_point, load_crit_config

    c = load_crit_config()
    a, r = rate_mismatch(np.array([2.1, 2.1]), np.array([2.0, 2.0]), 0.05)
    assert abs(r - 0.05) < 1e-9
    aq, rq = rate_mismatch(np.array([0.06, 0.0]), np.array([0.0, 0.0]), 0.05)   # #8 quiet branch: floor prevents blow-up
    assert rq < 1.0
    assert abs(adiabatic_index(1.0, -2.0, 1.0) - 0.5) < 1e-9   # eps=1e-9 floor perturbs bit-exact ==0.5; same tol as rate_mismatch check above
    ok, why = qualify_point({"converged": True, "saturated": False, "residual_rms": 1e-6, "rate_mismatch_abs": 0.01,
        "rate_mismatch_rel": 0.01, "slow_mismatch_rel": 0.01, "adiabatic_index": 0.05, "alpha_drift_index": 0.05}, c)
    assert ok and why == "qualified"
    bad, why2 = qualify_point({"converged": True, "saturated": False, "residual_rms": 1e-6, "rate_mismatch_abs": 0.01,
        "rate_mismatch_rel": 0.01, "slow_mismatch_rel": 0.01, "adiabatic_index": 0.05, "alpha_drift_index": 0.9}, c)
    assert (not bad) and why2 == "alpha_drift_too_fast"                         # #7 enforced
    miss, why3 = qualify_point({"converged": True, "saturated": False, "residual_rms": 1e-6}, c)
    assert (not miss) and why3.startswith("missing_")                          # #5 fail-closed


# --- user fix 1-1: every gate is a '>=' comparison, and 'nan >= tol' is False in Python, so a
# bad point with e.g. residual_rms=nan passed every gate and returned (True, "qualified") --
# a fail-OPEN hole. qualify_point must reject non-finite (nan or inf) values on the 6 NUMERIC
# required fields (not the two booleans converged/saturated). ---

def test_qualify_point_rejects_nonfinite_numeric_fields():
    import numpy as np
    from src.topic4_criticality import qualify_point, load_crit_config

    c = load_crit_config()
    baseline = {"converged": True, "saturated": False, "residual_rms": 1e-6, "rate_mismatch_abs": 0.01,
        "rate_mismatch_rel": 0.01, "slow_mismatch_rel": 0.01, "adiabatic_index": 0.05, "alpha_drift_index": 0.05}

    ok, why = qualify_point(dict(baseline), c)
    assert ok and why == "qualified"                                          # all-finite baseline still passes

    numeric_fields = ["residual_rms", "rate_mismatch_abs", "rate_mismatch_rel",
                       "slow_mismatch_rel", "adiabatic_index", "alpha_drift_index"]
    for field in numeric_fields:
        for bad_value in (np.nan, np.inf):
            f = dict(baseline)
            f[field] = bad_value
            bad, why_bad = qualify_point(f, c)
            assert (not bad) and why_bad == f"nonfinite_{field}", (field, bad_value, why_bad)


# --- Task 3a-2: solve_operating_point(init=) warm-start (scalar|array, #10) + solve_branches
# field-distance branch protocol (#9) with a deterministic random_small seed (#11). Brief
# .superpowers/sdd/task-3a-2-brief.md Steps 1/5, API-translated per the controller's VERIFIED
# T3a-2/T2.5 notes (.superpowers/sdd/progress.md): ExcitabilityField.from_core/InhibitionField.uniform/
# positional make_core_mask do not exist; use build_excitability_field/build_inhibition_field/
# make_core_mask(kind=, radius=) as in the T2.5 tests above. ---

def test_init_accepts_scalar_and_array():
    from src.topic4_m3b_spectral_phase import (
        Grid, build_kernels, make_core_mask, build_excitability_field, build_inhibition_field,
        solve_operating_point,
    )
    g = Grid(n=6, L=5.0)
    k = build_kernels(g, ell_perp=0.6)
    core = make_core_mask(g, kind="single", radius=0.9)
    exc = build_excitability_field(g, core, mu_core=0.9)
    inh = build_inhibition_field(g, core, q_global=0.94)

    lo = solve_operating_point(g, k, exc, inh, init={"rE": 1e-3, "rI": 1e-3})        # scalar
    prev = solve_operating_point(g, k, exc, inh, init={"rE": lo.rE, "rI": lo.rI})    # array (#10)
    assert prev.rE.shape == (g.n, g.n)


def test_branches_labeled_by_field_distance_deterministic():
    from src.topic4_m3b_spectral_phase import (
        Grid, build_kernels, make_core_mask, build_excitability_field, build_inhibition_field,
    )
    from src.topic4_criticality import solve_branches, load_crit_config

    g = Grid(n=6, L=5.0)
    k = build_kernels(g, ell_perp=0.6)
    core = make_core_mask(g, kind="single", radius=0.9)
    exc = build_excitability_field(g, core, mu_core=0.9)
    inh = build_inhibition_field(g, core, q_global=0.94)

    b1 = solve_branches(g, k, exc, inh, load_crit_config(), seed_key=(2, 3))
    b2 = solve_branches(g, k, exc, inh, load_crit_config(), seed_key=(2, 3))    # #11 same seed_key -> identical
    assert [x.branch_id for x in b1] == [x.branch_id for x in b2]
    assert all(hasattr(x, "branch_field_distance_to_low") for x in b1)          # #9 field-level

    allowed_reasons = {"low_branch", "high_branch", "saturated_branch", "ambiguous_branch"}
    for x in b1:
        assert hasattr(x, "branch_rate_mean")
        assert hasattr(x, "branch_alpha1")
        assert hasattr(x, "branch_residual")
        assert x.branch_selected_reason in allowed_reasons
    # the low_rate-seeded solve must register as (part of) the low branch in this non-saturated regime
    assert any(x.branch_selected_reason == "low_branch" for x in b1)


def test_solve_branches_random_small_seed_key_none_does_not_crash():
    """#11 seed_key=None must use a fixed literal fallback (never hash(None))."""
    from src.topic4_m3b_spectral_phase import (
        Grid, build_kernels, make_core_mask, build_excitability_field, build_inhibition_field,
    )
    from src.topic4_criticality import solve_branches, load_crit_config

    g = Grid(n=6, L=5.0)
    k = build_kernels(g, ell_perp=0.6)
    core = make_core_mask(g, kind="single", radius=0.9)
    exc = build_excitability_field(g, core, mu_core=0.9)
    inh = build_inhibition_field(g, core, q_global=0.94)

    b1 = solve_branches(g, k, exc, inh, load_crit_config())
    b2 = solve_branches(g, k, exc, inh, load_crit_config())
    assert [x.branch_id for x in b1] == [x.branch_id for x in b2]


# --- Task 3a-3: eigen-metrics on the frozen Jacobian's spectrum -- next_distinct_gap fixes
# spectral_gap's (TDD-7) conjugate-pair blind spot (raw a1-a2 array-order reads 0 when the leading
# mode is a complex-conjugate pair); leading_subspace_indices generalizes "the leading mode" to
# "the leading invariant subspace" (the pair, or a near-degenerate real group); pair_loading then
# builds a real non-negative spatial field from that subspace via the existing mode_e_field
# state-unpacking helper (not a hardcoded re-derivation). Verbatim from
# .superpowers/sdd/task-3a-3-brief.md Step 1. ---

def test_next_distinct_gap_and_leading_subspace():
    import numpy as np
    from src.topic4_m3b_spectral_phase import next_distinct_gap, leading_subspace_indices

    ev = np.array([-0.1 + 3j, -0.1 - 3j, -0.5 + 0j])
    assert abs(next_distinct_gap(ev, min_sep=1e-3) - 0.4) < 1e-9            # skips conjugate member
    assert set(leading_subspace_indices(ev, min_sep=1e-3, imag_tol=1e-3)) == {0, 1}   # conj pair


def test_pair_loading_uses_state_helper_nonneg():
    import numpy as np
    from src.topic4_m3b_spectral_phase import Grid, pair_loading

    g = Grid(n=2, L=1.0)
    N = g.size
    v1 = np.zeros(6 * N, complex)
    v1[0] = 1 + 1j
    R = np.column_stack([v1, np.conj(v1)])
    load = pair_loading(R, (0, 1), g)                        # via mode_e_field, not hardcoded unpacking
    assert load.shape == (g.n, g.n) and np.all(load >= 0)


# --- Task 3a-3 review F2: leading_subspace_indices's complex branch paired the leading eigenvalue
# with argmin(|ev - conj(ev[i])|) UNCONDITIONALLY. If the true conjugate partner was truncated out
# of the array (e.g. rate_eigenpairs(n_modes=...) split a pair), that unconditional pairing
# silently mis-pairs: alone, it self-pairs to (0,0) (inflating downstream loadings by sqrt(2));
# with an unrelated real mode present, it fake-pairs to (0,1). Fixed to only accept the candidate
# as a genuine partner (different index AND within imag_tol of the true conjugate); otherwise it
# falls back to the lone-mode tuple (i,). ---

def test_leading_subspace_indices_no_fake_pair_when_partner_truncated():
    import numpy as np
    from src.topic4_m3b_spectral_phase import leading_subspace_indices

    # true partner missing (only an unrelated real mode is present) -> must not fake-pair with it
    ev = np.array([-0.1 + 3j, -0.5])
    assert leading_subspace_indices(ev, min_sep=1e-3, imag_tol=1e-3) == (0,)


def test_leading_subspace_indices_lone_complex_mode_not_self_paired():
    import numpy as np
    from src.topic4_m3b_spectral_phase import leading_subspace_indices

    # single complex eigenvalue, no partner anywhere in the array -> must not self-pair to (0, 0)
    ev = np.array([-0.1 + 3j])
    assert leading_subspace_indices(ev, min_sep=1e-3, imag_tol=1e-3) == (0,)


# --- Task 3a-3 review F1: left_mode_input_projection had zero test coverage. Its
# biorthonormalization (psi = L[:,k] / conj(c), c = vdot(L[:,k], R[:,k])) is subtle: the reviewer
# proved that abs(vdot(l/c, b)) and abs(vdot(l/conj(c), b)) are mathematically IDENTICAL (both
# reduce to abs(vdot(l, b)) / abs(c)), so a value-only test cannot catch a conj(c) bug. Covered in
# three parts: (1) a hand-verified value check for wiring/sum/sqrt/idx bugs, (2) the
# scale-invariance property that DOES distinguish the two forms (only the correct
# self-normalizing psi is invariant to an arbitrary rescale of the RAW left column), and (3) a
# regression tie-in showing the idx=(0,) single-mode case reduces exactly to the existing trusted
# core_controllability helper. ---

def test_left_mode_input_projection_value_check():
    import numpy as np
    from src.topic4_m3b_spectral_phase import left_mode_input_projection

    L = np.array([[3., 0.], [0., 5.]])
    R = np.eye(2)
    b = np.array([1., 1.])
    # c0=3, psi0=l0/3=[1,0], vdot(psi0,b)=1; c1=5, psi1=l1/5=[0,1], vdot(psi1,b)=1; sqrt(1^2+1^2)
    assert left_mode_input_projection(L, R, (0, 1), b) == pytest.approx(np.sqrt(2), abs=1e-9)


def test_left_mode_input_projection_scale_invariant_to_raw_left_column():
    import numpy as np
    from src.topic4_m3b_spectral_phase import left_mode_input_projection

    L = np.array([[3., 0.], [0., 5.]], dtype=complex)
    R = np.eye(2, dtype=complex)
    b = np.array([1., 1.], dtype=complex)
    base = left_mode_input_projection(L, R, (0,), b)

    L_scaled = L.copy()
    L_scaled[:, 0] *= (2 + 3j)      # arbitrary nonzero complex rescale of the RAW left column
    scaled = left_mode_input_projection(L_scaled, R, (0,), b)
    assert scaled == pytest.approx(base, abs=1e-9)


# --- Task 3a-4: non-normality -- numerical_abscissa (max eigenvalue of J's Hermitian part; can be
# positive even when every eigenvalue of J has negative real part -- the non-normality signature,
# #16 conj().T complex-safe), directional_finite_time_gain_curve (REUSES
# topic4_m3b_spectral_phase.transient_gain's matrix-free ||exp(J*T) b||/||b|| -- same per-horizon
# directional gain, not re-implemented per §6.1, #15), and transient_amplification_present's
# alpha1>=0 guard (modal growth is not a stable transient, #17). Verbatim from
# .superpowers/sdd/task-3a-4-brief.md Step 1. ---

def test_nonnormality_review_fixes():
    import numpy as np
    from src.topic4_criticality import numerical_abscissa, transient_amplification_present

    J = np.array([[-1., 10.], [0., -2.]])
    assert numerical_abscissa(J) > 0                                          # #16
    assert transient_amplification_present({"10": 3.0}, alpha1=-0.5)          # stable + gain -> True
    assert not transient_amplification_present({"10": 3.0}, alpha1=0.2)       # #17 alpha>=0 -> modal growth, not transient


# --- supplementary: the brief's verbatim test above never exercises transient_amplification_present's
# other branch (stable AND gain at-or-below threshold -> False) -- add it directly so both branches
# of the 2-line gate are covered, not just the alpha1>=0 short-circuit. ---

def test_transient_amplification_present_false_when_gain_at_or_below_threshold():
    from src.topic4_criticality import transient_amplification_present

    assert not transient_amplification_present({"10": 1.5}, alpha1=-0.5)      # ==thresh -> not '>' -> False
    assert not transient_amplification_present({"10": 1.0}, alpha1=-0.5)      # stable but no amplification


# --- supplementary: directional_finite_time_gain_curve itself has ZERO direct invocation in the
# brief's Step-1 test (transient_amplification_present is only exercised with a literal curve dict)
# -- the same zero-coverage gap the T3a-3 review caught on left_mode_input_projection (F1: "naive
# value-test can't catch a bug because the wrong and right forms produce identical output" -- here
# the risk is inverted: an UNTESTED reuse wiring could silently regress to a no-op or wrong shape
# and nothing would fail). Cross-check the reused function's output against an INDEPENDENTLY
# hand-computed ||exp(J*T) b||/||b|| (test-side scipy.linalg.expm is fine -- §6.1 only forbids a
# duplicate expm implementation in src) AND against transient_gain directly, on the SAME non-normal
# J as above with a non-eigenvector direction b=[0,1] that genuinely transiently amplifies (peak
# gain ~2.3 at T=1ms, verified by direct numerical scan before writing this test). ---

def test_directional_finite_time_gain_curve_matches_expm_and_transient_gain():
    import numpy as np
    from scipy.linalg import expm
    from src.topic4_criticality import directional_finite_time_gain_curve
    from src.topic4_m3b_spectral_phase import transient_gain

    J = np.array([[-1., 10.], [0., -2.]])   # same non-normal example as test_nonnormality_review_fixes
    b = np.array([0., 1.])                  # NOT an eigenvector of J -> shows genuine transient growth
    horizons = [1, 2, 5]

    curve = directional_finite_time_gain_curve(J, b, horizons)
    assert set(curve.keys()) == {"1", "2", "5"}
    for T in horizons:
        expected = np.linalg.norm(expm(J * T) @ b) / np.linalg.norm(b)
        assert curve[str(T)] == pytest.approx(expected, rel=1e-9)
        assert curve[str(T)] == pytest.approx(transient_gain(J, b, T), rel=1e-9)   # exact reuse tie-in
    assert max(curve.values()) > 1.5        # b=[0,1] genuinely transiently amplifies before decaying


# --- supplementary: numerical_abscissa's OWN review finding #16 is specifically "use .conj().T,
# keep complex-safe" -- but the brief's Step-1 test uses a REAL J, where .conj().T and plain .T are
# identical, so that test cannot actually distinguish the fix from a plain-.T regression. A 2x2
# COMPLEX J also cannot distinguish them (2x2 Hermitian eigenvalues depend only on the OFF-DIAGONAL
# MAGNITUDE, which conjugation never changes -- verified by hand before writing this test); n>=3 is
# required because eigenvalues there generically depend on the relative PHASE between off-diagonal
# entries too. This 3x3 complex J was verified numerically to make plain .T and .conj().T diverge
# (1.619 vs 1.549) before being committed as a test. ---

def test_numerical_abscissa_is_complex_safe_not_plain_transpose():
    import numpy as np
    from src.topic4_criticality import numerical_abscissa

    J = np.array([
        [0.0 + 0.0j, 1.0 + 2.0j, 0.5 - 1.0j],
        [-3.0 + 0.0j, 0.0 + 0.0j, 2.0 + 0.5j],
        [1.0 + 0.0j, -1.0 + 1.0j, 0.0 + 0.0j],
    ])
    correct = numerical_abscissa(J)                                            # uses .conj().T
    wrong = float(np.max(np.linalg.eigvalsh(0.5 * (J + J.T)).real))             # #16 regression: plain .T
    assert correct == pytest.approx(1.5487428421097742, abs=1e-9)
    assert wrong == pytest.approx(1.6190508031916435, abs=1e-9)
    assert abs(correct - wrong) > 1e-3                                          # genuinely distinguishes the fix


def test_left_mode_input_projection_matches_core_controllability_single_mode():
    from src.topic4_m3b_spectral_phase import (
        Grid, build_kernels, make_core_mask, build_excitability_field, build_inhibition_field,
        solve_operating_point, build_jacobian_dense, rate_eigenpairs,
        core_perturbation_vector, core_controllability, left_mode_input_projection,
    )

    g = Grid(n=6, L=5.0)
    k = build_kernels(g, ell_perp=0.6)
    core = make_core_mask(g, kind="single", radius=0.9)
    exc = build_excitability_field(g, core, mu_core=1.0)
    inh = build_inhibition_field(g, core, q_global=0.94)

    op = solve_operating_point(g, k, exc, inh)
    res = rate_eigenpairs(build_jacobian_dense(g, k, op), g)
    b_core = core_perturbation_vector(g, core)

    subspace = left_mode_input_projection(res.left, res.right, (0,), b_core)
    single = core_controllability(res.left[:, 0], g, core)
    assert subspace == pytest.approx(single, abs=1e-9)


# --- Task 3a-5a: classify_trajectory 3-way pre-registered verdict (brief Steps 1-4). Pure
# LOGIC over synthetic point-dicts (no SNN). Verbatim from .superpowers/sdd/task-3a-5-brief.md
# Step 1 (naming #3.1, jump-window #18, fraction #19, ambiguity #20, continuation #2). ---

import numpy as np
from src.topic4_criticality import classify_trajectory


def _pts(alphas, t0=0, dt=10, branch="low_branch"):
    return [{"time_ms": t0 + i * dt, "alpha1": a, "qualified": True, "branch_id": branch,
             "branch_continuation_checked": True}
            for i, a in enumerate(alphas)]


def test_verdicts():
    c = load_crit_config()
    smooth = _pts(np.linspace(-0.5, -0.001, 8)) + [
        {"time_ms": 80, "alpha1": None, "qualified": False, "saturated": True,
         "branch_id": "saturated_branch"}]
    r = classify_trajectory(smooth, c)
    assert r["verdict"] == "smooth_CSD"
    assert r["alpha1_closest_to_zero_pre_onset"] == max(
        p["alpha1"] for p in smooth if p["qualified"])   # #3.1 max not min
    hard = _pts(np.linspace(-0.6, -0.2, 8)) + [
        {"time_ms": 85, "alpha1": None, "qualified": False, "saturated": True,
         "branch_id": "saturated_branch", "branch_continuation_checked": True,
         "continuation_status": "low_branch_remains_far_from_alpha0_until_jump"}]
    r2 = classify_trajectory(hard, c)
    assert r2["verdict"] == "hard_jump_no_CSD" and r2["jump_distance_to_alpha0"] == abs(hard[7]["alpha1"])


def test_hard_requires_continuation_and_window_and_fraction_and_ambiguity():
    c = load_crit_config()
    # #2 no continuation -> unresolved
    noc = _pts(np.linspace(-0.6, -0.2, 8))
    noc[-1]["branch_continuation_checked"] = False
    noc += [{"time_ms": 85, "alpha1": None, "qualified": False, "saturated": True,
             "branch_id": "saturated_branch"}]
    assert classify_trajectory(noc, c)["verdict"] == "unresolved_operating_point"
    # #18 saturation outside window -> unresolved
    late = _pts(np.linspace(-0.6, -0.2, 8)) + [
        {"time_ms": 10000, "alpha1": None, "qualified": False, "saturated": True,
         "branch_id": "saturated_branch", "branch_continuation_checked": True}]
    assert classify_trajectory(late, c)["verdict"] == "unresolved_operating_point"
    # #19 too few qualified fraction -> unresolved
    many = _pts(np.linspace(-0.5, -0.001, 5)) + [
        {"time_ms": 100 + i, "alpha1": None, "qualified": False, "branch_id": "low_branch"}
        for i in range(95)]
    assert classify_trajectory(many, c)["verdict"] == "unresolved_operating_point"
    # #20 ambiguous near transition -> unresolved
    amb = _pts(np.linspace(-0.5, -0.001, 8))
    amb[-1]["branch_id"] = "ambiguous_branch"
    assert classify_trajectory(amb, c)["verdict"] == "unresolved_operating_point"


# --- Task 3a-5a review (Important): classify_trajectory read q[-1]/alphas[-1] assuming
# `points` arrives pre-sorted ascending by time_ms, but nothing sorted or asserted that. A
# caller assembling points from more than one source (e.g. this file's own _pts(...) +
# [saturated(...)] idiom, or a future T3a-5b producer) could hand in a differently-ordered
# list with the SAME values and get a DIFFERENT verdict silently -- the reviewer reproduced a
# smooth_CSD -> unresolved flip just by shuffling list order. DETERMINISTIC reorder (full
# list(reversed(...)), no unseeded shuffle) of the brief's own smooth_CSD and hard_jump_no_CSD
# fixtures from test_verdicts above. ---

def test_classify_trajectory_verdict_is_reorder_invariant():
    c = load_crit_config()

    smooth = _pts(np.linspace(-0.5, -0.001, 8)) + [
        {"time_ms": 80, "alpha1": None, "qualified": False, "saturated": True,
         "branch_id": "saturated_branch"}]
    r = classify_trajectory(smooth, c)
    r_reordered = classify_trajectory(list(reversed(smooth)), c)
    assert r_reordered["verdict"] == r["verdict"] == "smooth_CSD"
    assert r_reordered["last_stable_alpha1"] == r["last_stable_alpha1"]
    assert r_reordered["alpha1_closest_to_zero_pre_onset"] == r["alpha1_closest_to_zero_pre_onset"]

    hard = _pts(np.linspace(-0.6, -0.2, 8)) + [
        {"time_ms": 85, "alpha1": None, "qualified": False, "saturated": True,
         "branch_id": "saturated_branch", "branch_continuation_checked": True,
         "continuation_status": "low_branch_remains_far_from_alpha0_until_jump"}]
    assert classify_trajectory(list(reversed(hard)), c)["verdict"] == "hard_jump_no_CSD"
