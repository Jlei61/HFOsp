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
    g_K, and h_G all lower excitability when raised."""
    from src.topic4_criticality import slow_to_ratefield_sign_ok, load_crit_config

    result = slow_to_ratefield_sign_ok(load_crit_config())
    assert result == {"q_I": True, "g_K": True, "h_G": True}


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
