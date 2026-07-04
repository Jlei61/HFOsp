import numpy as np
from src.topic5_v3p_preictal_trajectory import load_v3p_config, theil_sen_slope, spearman_trend, slope_over_windows, within_compartment_flux, global_axial_energy, residualize_slope, extract_window_metrics
from src.topic5_v3_mode_transition import subspace_projectors, axis_nonaxis_vectors, rank_forward, load_v3_config

def test_v3p_config_keys():
    c = load_v3p_config()
    assert c["preictal"]["phases"] == ["P0", "P1", "P2", "P3"]
    assert c["preictal"]["span_full_rel"] == [-120.0, -10.0]
    assert c["preictal"]["span_guard_rel"] == [-120.0, -20.0]      # rev1 onset guard
    assert c["trend"]["estimator"] == "theil_sen"
    assert c["gates"]["h3b_require"] == ["p_label", "p_rate", "lag1_specific_positive"]
    assert c["gates"]["h3c_require"] == ["p_label", "p_phase", "p_block"]
    assert c["gates"]["require_both_spans"] is True
    assert c["nulls_v3p"]["rate_null_per_window"] is True and c["nulls_v3p"]["time_order_null"] is True
    assert c["residualization"]["primary_adjudicator"] == "label_null_slope"
    assert c["co_primary"]["correction"] == "holm"
    assert c["co_primary"]["endpoints"] == ["net_offaxis_flux_surplus_slope", "mode_shift_density_surplus_slope"]
    assert c["inherit_v3_config"] == "topic5_v3" and c["cohorts"]["primary"] == "narrow"

def test_theil_sen_and_spearman_trend():
    t = np.arange(20.0)
    y = 0.3 * t + 1.0
    assert abs(theil_sen_slope(y, t) - 0.3) < 1e-9
    assert spearman_trend(y, t) > 0.999
    y_out = y.copy(); y_out[7] = 500.0                       # one wild outlier window
    assert abs(theil_sen_slope(y_out, t) - 0.3) < 0.05       # robust
    y_flat = np.zeros(20)
    assert abs(theil_sen_slope(y_flat, t)) < 1e-9 and np.isnan(spearman_trend(y_flat, t))

def test_slope_over_windows_nan_safe_and_dispatch():
    t = np.arange(12.0); y = 0.5 * t
    y2 = y.copy(); y2[3] = np.nan; t2 = t.copy()
    assert abs(slope_over_windows(y2, t2, "theil_sen") - 0.5) < 1e-9   # drops the NaN window
    assert abs(slope_over_windows(y, t, "ols") - 0.5) < 1e-9
    assert np.isnan(slope_over_windows(y[:1], t[:1], "theil_sen"))      # <2 points

def test_within_compartment_flux_self_sustain():
    # 4 contacts, N = {2,3}; strong 2->3 and 3->2 mass, diagonal already zero
    atm = np.zeros((4, 4))
    atm[2, 3] = 0.6; atm[3, 2] = 0.4; atm[0, 1] = 0.9   # axis-internal, ignored by N block
    val = within_compartment_flux(atm, np.array([2, 3]), "source_mean")
    assert abs(val - 0.5) < 1e-9                          # mean of active N-source outgoing-into-N mass (0.6, 0.4)
    assert within_compartment_flux(atm, np.array([2, 3]) , "source_mean") == \
           within_compartment_flux(np.pad(atm, ((0,1),(0,1))), np.array([2, 3]), "source_mean")  # padding never-active row invariant

def test_global_axial_energy():
    env = np.array([[1.0, -1.0], [2.0, -2.0], [0.0, 0.0]])   # 3 contacts x 2 t; mean|.| rows = 1,2,0
    g, a = global_axial_energy(env, np.array([0, 1]))
    assert abs(g - 1.0) < 1e-9 and abs(a - 1.5) < 1e-9        # global mean over all rows; axial over rows 0,1

def test_residualize_strips_global_and_is_conservative_under_collinearity():
    t = np.arange(20.0)
    glob = 0.4 * t                                   # global energy IS collinear with time
    # non-axis metric = 2*global + a genuinely orthogonal rise
    orth = np.where(t % 2 == 0, 1.0, -1.0) * 0.1     # zero net slope, orthogonal to t
    vals = 2.0 * glob + orth
    resid_slope = residualize_slope(vals, t, [glob], "theil_sen")
    assert abs(resid_slope) < 0.02                   # collinear global stripped -> conservative ~0 (documented floor)

def test_residualize_keeps_orthogonal_trend():
    t = np.arange(20.0)
    glob = np.sin(t)                                 # global uncorrelated with linear time
    vals = 0.3 * t + 2.0 * glob
    resid_slope = residualize_slope(vals, t, [glob], "ols")
    assert abs(resid_slope - 0.3) < 0.05             # orthogonal non-axis trend survives

def test_residualize_nan_when_insufficient_dof_for_covariate_count():
    t = np.arange(3.0)
    vals = np.array([10.0, -20.0, 30.0])
    cov1 = np.array([1.0, 2.0, 3.0]); cov2 = np.array([0.0, 1.0, 0.5])
    assert np.isnan(residualize_slope(vals, t, [cov1, cov2], "ols"))   # 3 windows, 3 design cols -> no residual DOF -> nan (was artifactual ~0)
    # and 1-covariate with 3 windows still proceeds (unchanged behavior):
    assert np.isfinite(residualize_slope(np.array([0.3, 0.6, 0.9]), t, [cov1], "ols"))

def test_extract_window_metrics_keys_and_flux_sign():
    rng = np.random.default_rng(0); names = [f"A{i}" for i in range(6)]
    axis = names[:3]; nonaxis = names[3:]
    P_A, P_N = subspace_projectors(names, axis, nonaxis)
    rf = rank_forward({n: float(i) for i, n in enumerate(axis)})
    e_am, e_ag, e_nm = axis_nonaxis_vectors(names, rf, axis, nonaxis)
    # scripted axis->non-axis cascade: axis fires at t, non-axis at t+1
    n_t = 200; env = rng.standard_normal((6, n_t)) * 0.1
    env[:3, :-1] += 4.0 * (rng.random((3, n_t - 1)) > 0.6)
    env[3:, 1:] += env[:3, :-1]                       # non-axis echoes axis one step later
    geom = {"names": names, "axis_idx": np.array([0,1,2]), "nonaxis_idx": np.array([3,4,5]),
            "P_A": P_A, "P_N": P_N, "e_axis_mean": e_am, "e_nonaxis_mean": e_nm, "rank_forward": rf}
    m = extract_window_metrics(env, geom, load_v3_config())
    for k in ["net_offaxis_flux_lag1","net_offaxis_flux_lag0","mode_shift_density","mode_singular_gap",
              "nonaxis_activation_rate","n_activation_events","global_energy","axial_energy",
              "N_self_sustain_lag1","N_self_sustain_lag0","gain_axis","gain_nonaxis","beta_axis_strength"]:
        assert k in m
    # lag1 (delayed A->N flow) exceeds lag0 (same-time co-activation) for a scripted one-step cascade
    assert np.isfinite(m["net_offaxis_flux_lag1"]) and m["net_offaxis_flux_lag1"] > m["net_offaxis_flux_lag0"]
