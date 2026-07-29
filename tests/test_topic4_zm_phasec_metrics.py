import numpy as np
import pytest

from src.topic4_zm_phasec_metrics import (
    PhaseCThresholds,
    activity_and_spatial_entropy,
    aggregate_phasec_taxonomy,
    classify_phasec_seed,
    firing_and_ceiling_metrics,
    isi_cv2_and_refractory_lock,
    jeffreys_interval,
    paired_local_gain,
    pairwise_spike_count_correlation,
    phasec_metrics_from_raster,
    phasec_bootstrap_units,
    refractory_ceiling_hz,
    spatial_active_area_from_rate_grid,
)


DT = 0.5
TAU_REF = 2.0
N_STEPS = 4000


def _renewal_raster(n=120, seed=0, mean_isi_ms=7.0):
    """Irregular renewal process with the model's hard 2-ms refractory."""
    rng = np.random.default_rng(seed)
    x = np.zeros((N_STEPS, n), bool)
    for j in range(n):
        t = float(rng.uniform(0, mean_isi_ms))
        while t < N_STEPS * DT:
            x[min(N_STEPS - 1, int(np.floor(t / DT))), j] = True
            t += TAU_REF + float(rng.exponential(mean_isi_ms - TAU_REF))
    return x


def _periodic_raster(n=120, period_ms=TAU_REF):
    x = np.zeros((N_STEPS, n), bool)
    period = int(round(period_ms / DT))
    for j in range(n):
        # Phase staggering prevents this fixture from being only a synchronous-flash test.
        x[j % period :: period, j] = True
    return x


def _linear_gain():
    return paired_local_gain([
        dict(delta_mV=0.05, rate_vth_minus_hz=101.0, rate_vth_plus_hz=99.0,
             rate_baseline_hz=100.0),
        dict(delta_mV=0.10, rate_vth_minus_hz=102.0, rate_vth_plus_hz=98.0,
             rate_baseline_hz=100.0),
    ])


def test_ceiling_fraction_and_core_are_explicit():
    x = np.column_stack([_periodic_raster(20), _renewal_raster(20, seed=1)])
    core = np.zeros(40, bool)
    core[:20] = True
    out, rates = firing_and_ceiling_metrics(
        x, DT, TAU_REF, core_mask=core, ceiling_rate_fraction=0.8
    )
    assert out["max_refractory_rate_hz"] == 500.0
    assert out["ceiling_fraction_core"] == 1.0
    assert out["ceiling_fraction_active_core"] == 1.0
    assert out["ceiling_fraction_all"] == pytest.approx(0.5)
    assert np.all(rates[:20] == 500.0)


def test_refractory_ceiling_uses_discrete_engine_steps():
    assert refractory_ceiling_hz(2.0, 0.1) == pytest.approx(500.0)
    assert refractory_ceiling_hz(2.0, 0.3) == pytest.approx(1000.0 / 2.1)


def test_isi_cv2_and_ref_lock_separate_irregular_from_periodic():
    irregular = _renewal_raster(seed=3, mean_isi_ms=12.0)
    periodic = _periodic_raster()
    ai, *_ = isi_cv2_and_refractory_lock(irregular, DT, TAU_REF)
    sat, *_ = isi_cv2_and_refractory_lock(periodic, DT, TAU_REF)
    assert ai["isi_cv2"]["median"] > 0.70
    assert ai["refractory_locked_fraction"] < 0.25
    assert sat["isi_cv2"]["median"] == pytest.approx(0.0)
    assert sat["refractory_locked_fraction"] == 1.0


def test_pairwise_corr_and_shift_null_detect_common_modulation():
    rng = np.random.default_rng(4)
    x = np.zeros((N_STEPS, 80), bool)
    # Common 25-ms activity windows plus independent within-window spikes.
    for start in range(0, N_STEPS, 100):
        active = slice(start, min(start + 50, N_STEPS))
        x[active] = rng.random((active.stop - active.start, x.shape[1])) < 0.05
    out = pairwise_spike_count_correlation(
        x, DT, bin_ms=5.0, n_null=80, null_seed=9, min_spikes=3
    )
    assert out["status"] == "ok"
    assert out["observed_median"] > out["null_q95"]
    assert out["excess_over_null"] > 0.05


def test_fixed_panels_are_activity_independent_and_bootstrap_units_are_compact():
    x = _renewal_raster(n=80, seed=5, mean_isi_ms=12.0)
    # These inactive cells stay in the fixed panel; the routine may exclude
    # their zero-variance columns but must not replace them with active cells.
    x[:, :8] = False
    analysis = np.arange(40)
    pairwise = np.r_[np.arange(8), np.arange(40, 48)]
    pos = np.random.default_rng(6).uniform(0, 20, size=(80, 2))
    metrics = phasec_metrics_from_raster(
        x,
        DT,
        TAU_REF,
        core_mask=np.arange(80) < 40,
        positions=pos,
        L=20.0,
        analysis_panel_ids=analysis,
        pairwise_panel_ids=pairwise,
    )
    assert metrics["analysis_counts"]["analysis_panel_n"] == 40
    assert metrics["analysis_counts"]["pairwise_panel_n"] == 16
    assert metrics["pairwise_5ms"]["n_neurons"] <= 8
    units = phasec_bootstrap_units(
        x,
        DT,
        TAU_REF,
        core_mask=np.arange(80) < 40,
        analysis_panel_ids=analysis,
        pairwise_panel_ids=pairwise,
        positions=pos,
        L=20.0,
        pairwise_n_null=8,
    )
    assert units["rho80_active_core_by_block_window"].shape == (4, 6)
    assert units["isi_cv2_by_panel_neuron"].shape == (40,)
    assert units["refractory_isi_fraction_by_panel_neuron"].shape == (40,)
    assert units["block_isi_cv2_by_panel_neuron"].shape == (4, 40)
    assert units["block_refractory_isi_fraction_by_panel_neuron"].shape == (4, 40)
    assert units["block_refractory_isi_numerator_by_stratum"].shape == (4, 2)
    assert units["block_refractory_isi_denominator_by_stratum"].shape == (4, 2)
    assert tuple(units["refractory_isi_stratum_names"]) == (
        "core", "surround"
    )
    assert units["active_grid_fraction_by_block"].shape == (4,)
    assert units["active_area_fraction_by_block_window"].shape == (4, 20)
    assert units["pair_corr_by_block_and_pair"].shape == (4, 120)
    assert units["pair_null_median_by_block_and_draw"].shape == (4, 3, 8)
    assert tuple(units["pair_null_stratum_names"]) == (
        "core_core",
        "core_surround",
        "surround_surround",
    )
    assert units["spatial_area_denominator"].item() == (
        "anatomy_occupied_E_grid_bins"
    )


def test_bootstrap_ceiling_windows_preserve_time_neuron_axis_and_block_truth():
    """rho80 is computed per active-core neuron, never per time sample."""
    steps_per_block = int(round(500.0 / DT))
    x = np.zeros((2 * steps_per_block, 8), bool)
    core = np.arange(8) < 4

    # Block 0: every core neuron fires at exactly 80% of the implemented
    # refractory ceiling (400 Hz), with phases staggered across neurons.
    near_ceiling_period = int(round(2.5 / DT))
    for neuron in np.flatnonzero(core):
        x[neuron % near_ceiling_period:steps_per_block:
          near_ceiling_period, neuron] = True

    # Block 1: the same core neurons remain active (10 Hz) but are far below
    # the ceiling.  Thus the root block x six-window truth is exactly 1 then 0.
    quiet_period = int(round(100.0 / DT))
    for neuron in np.flatnonzero(core):
        start = steps_per_block + neuron
        x[start::quiet_period, neuron] = True

    # Keep surround members non-degenerate and ensure the fixed pair panel
    # spans core-core, core-surround, and surround-surround strata.
    for neuron in np.flatnonzero(~core):
        x[neuron::int(round(25.0 / DT)), neuron] = True
    positions = np.column_stack([
        np.arange(8, dtype=float) % 4 + 0.5,
        np.arange(8, dtype=float) // 4 + 0.5,
    ])
    units = phasec_bootstrap_units(
        x,
        DT,
        TAU_REF,
        core_mask=core,
        analysis_panel_ids=np.arange(8),
        pairwise_panel_ids=np.array([0, 1, 4, 5]),
        positions=positions,
        L=4.0,
        pairwise_n_null=2,
        n_grid=4,
    )
    rho = units["rho80_active_core_by_block_window"]
    assert rho.shape == (2, 6)
    assert np.array_equal(rho[0], np.ones(6))
    assert np.array_equal(rho[1], np.zeros(6))


def test_pooled_refractory_probability_units_are_event_weighted_and_keep_cross_block_isi():
    """f_ref sufficient statistics count ISIs, not per-neuron fractions."""
    steps_per_block = int(round(500.0 / DT))
    x = np.zeros((2 * steps_per_block, 4), bool)
    # Core neuron 0 supplies many refractory-near intervals.  Surround neuron
    # 2 supplies only two long intervals, including one crossing 500 ms.
    x[::int(round(4.0 / DT)), 0] = True
    for step in (0, steps_per_block - 2, steps_per_block + 2):
        x[step, 2] = True
    # Keep the other fixed-pair neurons non-degenerate.
    x[::int(round(25.0 / DT)), 1] = True
    x[::int(round(30.0 / DT)), 3] = True
    core = np.asarray([True, True, False, False])
    pos = np.asarray([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]])
    units = phasec_bootstrap_units(
        x,
        DT,
        TAU_REF,
        core_mask=core,
        analysis_panel_ids=np.arange(4),
        pairwise_panel_ids=np.arange(4),
        positions=pos,
        L=2.0,
        pairwise_n_null=2,
        n_grid=2,
    )
    numerator = units["block_refractory_isi_numerator_by_stratum"]
    denominator = units["block_refractory_isi_denominator_by_stratum"]
    direct_num = np.zeros(2, int)
    direct_den = np.zeros(2, int)
    for neuron in range(4):
        times = np.flatnonzero(x[:, neuron]) * DT
        intervals = np.diff(times)
        stratum = 0 if core[neuron] else 1
        direct_num[stratum] += int(np.sum(intervals <= TAU_REF + 2 * DT))
        direct_den[stratum] += int(intervals.size)
    np.testing.assert_array_equal(numerator.sum(axis=0), direct_num)
    np.testing.assert_array_equal(denominator.sum(axis=0), direct_den)
    # The cross-boundary surround interval is retained exactly once.
    assert denominator[1, 1] >= 1


def test_spatial_active_area_uses_rate_grid_not_active_neuron_fraction():
    n_grid = 4
    per_cell = 4
    n = n_grid * n_grid * per_cell
    positions = []
    for iy in range(n_grid):
        for ix in range(n_grid):
            positions.extend([
                ((ix + 0.5), (iy + 0.5))
                for _ in range(per_cell)
            ])
    positions = np.asarray(positions, float)
    steps = int(round(1000.0 / DT))
    hotspot = np.zeros((steps, n), bool)
    spread = np.zeros_like(hotspot)
    window_steps = int(round(25.0 / DT))
    for start in range(0, steps, window_steps):
        # Eight active neurons, but restricted to two spatial bins.
        hotspot[start, :8] = True
        # The same eight-neuron fraction, distributed over eight bins.
        spread[start, np.arange(0, 8 * per_cell, per_cell)] = True
    common = dict(
        dt_ms=DT,
        tau_ref_ms=TAU_REF,
        core_mask=np.arange(n) < n // 2,
        analysis_panel_ids=np.arange(40),
        pairwise_panel_ids=np.r_[np.arange(8), np.arange(n // 2, n // 2 + 8)],
        positions=positions,
        L=float(n_grid),
        pairwise_n_null=8,
        n_grid=n_grid,
        spatial_active_floor_hz=5.0,
    )
    hotspot_units = phasec_bootstrap_units(hotspot, **common)
    spread_units = phasec_bootstrap_units(spread, **common)
    assert hotspot.mean() == spread.mean()
    assert np.median(
        hotspot_units["active_area_fraction_by_block_window"]
    ) == pytest.approx(2 / 16)
    assert np.median(
        spread_units["active_area_fraction_by_block_window"]
    ) == pytest.approx(8 / 16)
    assert hotspot_units["spatial_grid_all_E_bins_occupied"].item()


def test_spatial_active_area_denominator_excludes_anatomically_empty_bins():
    positions = np.repeat(
        np.asarray([[0.5, 0.5], [1.5, 0.5]], float), 20, axis=0
    )
    x = np.zeros((int(round(1000.0 / DT)), len(positions)), bool)
    x[::int(round(25.0 / DT)), :20] = True
    units = phasec_bootstrap_units(
        x,
        DT,
        TAU_REF,
        core_mask=np.arange(len(positions)) < 20,
        analysis_panel_ids=np.arange(20),
            pairwise_panel_ids=np.r_[np.arange(8), np.arange(20, 28)],
        positions=positions,
        L=4.0,
        pairwise_n_null=8,
        n_grid=4,
        spatial_active_floor_hz=5.0,
    )
    assert units["spatial_grid_n_occupied_E"].item() == 2
    assert not units["spatial_grid_all_E_bins_occupied"].item()
    np.testing.assert_allclose(units["active_grid_fraction_by_block"], 0.5)
    assert np.median(
        units["active_area_fraction_by_block_window"]
    ) == pytest.approx(0.5)


def test_fine_rate_grid_active_area_uses_local_rate_and_occupied_bins():
    rates = np.asarray([
        [[10.0, 0.0], [1.0, 999.0]],
        [[10.0, 6.0], [1.0, 999.0]],
    ])
    anatomy = np.asarray([[4, 4], [4, 0]])
    area = spatial_active_area_from_rate_grid(
        rates, anatomy, active_floor_hz=5.0
    )
    np.testing.assert_allclose(area, [1 / 3, 2 / 3])


def test_spatial_entropy_separates_uniform_from_hotspot():
    n = 100
    side = 10
    xx, yy = np.meshgrid(np.arange(side) + 0.5, np.arange(side) + 0.5)
    pos = np.column_stack([xx.ravel(), yy.ravel()])
    uniform = np.zeros((200, n), bool)
    hotspot = np.zeros_like(uniform)
    # Every 5-ms bin covers the full grid in one arm, one corner in the other.
    uniform[::10, :] = True
    hotspot[::10, :4] = True
    u = activity_and_spatial_entropy(
        uniform, DT, bin_ms=5.0, positions=pos, L=10.0, n_grid=10
    )
    h = activity_and_spatial_entropy(
        hotspot, DT, bin_ms=5.0, positions=pos, L=10.0, n_grid=10
    )
    assert u["spatial_entropy"]["median"] > 0.95
    assert h["spatial_entropy"]["median"] < 0.35
    assert u["active_grid_fraction"]["median"] > h["active_grid_fraction"]["median"]


def test_paired_local_gain_requires_two_linear_monotone_central_pairs():
    good = _linear_gain()
    assert good["linearity_pass"]
    assert good["gain_hz_per_mV_median"] == pytest.approx(20.0)
    bad = paired_local_gain([
        dict(delta_mV=0.05, rate_vth_minus_hz=101, rate_vth_plus_hz=99,
             rate_baseline_hz=100),
        dict(delta_mV=0.10, rate_vth_minus_hz=108, rate_vth_plus_hz=99,
             rate_baseline_hz=100),
    ])
    assert not bad["linearity_pass"]


def test_ai_poisson_fixture_is_asynchronous_tonic_candidate():
    x = _renewal_raster(n=180, seed=11, mean_isi_ms=12.0)
    pos = np.random.default_rng(1).uniform(0, 20, size=(x.shape[1], 2))
    metrics = phasec_metrics_from_raster(
        x, DT, TAU_REF, positions=pos, L=20.0, pairwise_null_seed=2
    )
    out = classify_phasec_seed(metrics, local_gain=_linear_gain())
    assert out["klass"] == "balanced_asynchronous_tonic_candidate"


def test_refractory_periodic_fixture_is_plateau():
    x = _periodic_raster(n=180)
    metrics = phasec_metrics_from_raster(x, DT, TAU_REF)
    # Perfectly periodic bins have zero temporal variance and therefore no
    # defined pairwise correlation.  Give the seed classifier an eligible,
    # conservative zero-excess online pairwise summary; raster-level ref-lock,
    # Fano and ceiling evidence remain real.
    metrics["pairwise_5ms"] = {
        "status": "ok",
        "excess_over_null": 0.0,
        "n_neurons": 180,
    }
    out = classify_phasec_seed(metrics, local_gain=_linear_gain())
    assert out["klass"] == "refractory_limited_plateau"


def test_mixed_fixture_is_not_forced_to_either_positive_class():
    x = np.column_stack([
        _periodic_raster(n=100),
        _renewal_raster(n=100, seed=13, mean_isi_ms=12.0),
    ])
    metrics = phasec_metrics_from_raster(x, DT, TAU_REF, pairwise_null_seed=3)
    out = classify_phasec_seed(metrics, local_gain=_linear_gain())
    assert out["klass"] == "mixed_or_unresolved"
    assert "both_ai_like_and_refractory_like_subpopulations" in out["reasons"]


def test_seed_classifier_fails_closed_when_gain_or_metrics_missing():
    x = _renewal_raster(seed=17)
    metrics = phasec_metrics_from_raster(x, DT, TAU_REF)
    assert classify_phasec_seed(metrics)["klass"] == "no_evidence"
    broken = dict(metrics)
    broken["isi"] = {}
    assert classify_phasec_seed(broken, local_gain=_linear_gain())["klass"] == "no_evidence"


def test_jeffreys_and_aggregate_taxonomy_are_fail_closed():
    post = jeffreys_interval(3, 3)
    assert 0 < post["lo"] < post["median"] < post["hi"] < 1
    ai = "balanced_asynchronous_tonic_candidate"
    sat = "refractory_limited_plateau"
    same = aggregate_phasec_taxonomy([
        {"seed": 1, "klass": ai},
        {"seed": 3, "klass": ai},
        {"seed": 4, "klass": ai},
    ])
    assert same["verdict"] == f"replicated_{ai}"
    mixed = aggregate_phasec_taxonomy([
        {"seed": 1, "klass": ai},
        {"seed": 3, "klass": ai},
        {"seed": 4, "klass": sat},
    ])
    assert mixed["verdict"] == "heterogeneous_or_unresolved"
    short = aggregate_phasec_taxonomy([
        {"seed": 1, "klass": ai},
        {"seed": 3, "klass": ai},
    ])
    assert short["verdict"] == "no_evidence"
