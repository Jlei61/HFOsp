from __future__ import annotations

import numpy as np

from src.topic5_group_event_state.v037.h2b import (
    FixedBaselineResidualHazard,
    HAZARD_BAND_EDGES,
    LowCapacityDiscreteHazard,
    _causal_grid_lookup,
    _clock_features,
    _fit_nested_hazards,
    _fit_masked_state_residual_ridge,
    _phase_circular_shift,
    _robust_pca,
    _standardise_predictors,
)


def test_h2b_survival_baseline_is_low_capacity_but_keeps_five_minute_bins() -> None:
    import torch

    model = LowCapacityDiscreteHazard(3).double()
    assert model.bin_logit.numel() == 6
    assert HAZARD_BAND_EDGES == (0, 1, 3, 6, 12, 24, 72)
    with torch.no_grad():
        model.bin_logit.copy_(torch.arange(6, dtype=torch.float64))
    x = torch.zeros((72, 3), dtype=torch.float64)
    logits = model(x, torch.arange(72))
    assert logits.shape == (72,)
    # Adjacent five-minute bins share a baseline only inside a declared band.
    assert logits[1] == logits[2]
    assert logits[2] != logits[3]


def test_h2b_wrong_time_shift_can_cross_carry_segments_but_stays_in_phase_rows() -> None:
    state = np.arange(12, dtype=np.float32)[:, None]
    time = np.arange(12, dtype=np.float64) * 3600.0
    rows = np.arange(2, 10, dtype=np.int64)
    shifted, valid = _phase_circular_shift(state, time, rows, 3.0 * 3600.0)
    assert np.all(valid[rows])
    assert not np.any(valid[:2]) and not np.any(valid[10:])
    assert sorted(shifted[rows, 0].tolist()) == sorted(state[rows, 0].tolist())
    assert np.all(np.abs(time[shifted[rows, 0].astype(int)] - time[rows]) >= 3.0 * 3600.0)


def test_h2b_pca_fit_is_unchanged_by_future_rows() -> None:
    rng = np.random.default_rng(17)
    value = rng.normal(size=(40, 7))
    fit = np.arange(20)
    first, first_audit = _robust_pca(value, fit, 4, 9)
    altered = value.copy()
    altered[20:] = rng.normal(loc=100.0, scale=20.0, size=altered[20:].shape)
    second, second_audit = _robust_pca(altered, fit, 4, 9)
    assert np.allclose(first[fit], second[fit])
    for key in ("centre", "scale", "components", "mean", "output_centre", "output_scale"):
        assert np.allclose(first_audit[key], second_audit[key])


def test_h2b_grid_lookup_never_reads_a_future_anchor() -> None:
    anchor = np.asarray([10.0, 20.0, 30.0, 110.0, 120.0])
    segment = np.asarray([0, 0, 0, 1, 1])
    bounds = np.asarray([[0.0, 50.0], [100.0, 150.0]])
    feature = {"x": np.arange(anchor.size, dtype=np.float32)[:, None]}
    out, valid = _causal_grid_lookup(
        np.asarray([9.0, 10.0, 19.0, 35.0, 105.0, 115.0, 151.0]),
        anchor, segment, bounds, feature,
    )
    assert np.array_equal(valid, [False, True, True, True, False, True, False])
    assert np.array_equal(out["x"][valid, 0], [0.0, 0.0, 2.0, 3.0])


def test_h2b_predictor_scaling_uses_fit_only_and_preserves_intercept() -> None:
    x = np.asarray([[1.0, 0.0], [1.0, 2.0], [1.0, 100.0], [1.0, 200.0]])
    fit = np.asarray([0, 1])
    scaled = _standardise_predictors(x, fit, intercept=True)
    changed = x.copy(); changed[2:, 1] += 10000.0
    scaled_changed = _standardise_predictors(changed, fit, intercept=True)
    assert np.allclose(scaled[:2], scaled_changed[:2])
    assert np.array_equal(scaled[:, 0], np.ones(4))


def test_h2b_clock_features_are_periodic_and_bounded() -> None:
    epoch = np.asarray([0.0, 86400.0, 43200.0])
    value = _clock_features(epoch)
    assert value.shape == (3, 2)
    assert np.all(np.abs(value) <= 1.0 + 1e-7)
    assert np.allclose(value[0], value[1], atol=1e-6)


def test_h2b_hazard_gate_counts_distinct_seizures_not_dense_rows() -> None:
    anchor = np.arange(0.0, 2400.0, 300.0)
    phase = np.asarray(["FIT"] * 4 + ["INNER"] * 2 + ["SELECTION"] * 2)
    bounds = {"20pct": -1.0, "60pct": 1200.0, "70pct": 1800.0, "80pct": 2400.0}
    support = np.asarray([[0.0, 2400.0]])
    # One seizure can generate many positive person-period rows.  It must not
    # satisfy the minimum of three independent FIT outcomes.
    seizures = [{"onset_epoch": 900.0, "offset_epoch": 950.0},
                {"onset_epoch": 1500.0, "offset_epoch": 1550.0},
                {"onset_epoch": 2100.0, "offset_epoch": 2150.0}]
    result = _fit_nested_hazards(
        anchor, phase, support, bounds, seizures, {"baseline": np.zeros((anchor.size, 1))}
    )
    assert result["baseline"]["status"] == "NOT_ESTIMABLE"
    assert result["support"]["seizures_by_phase"]["FIT"] == 1


def test_h2b_state_extension_freezes_the_selected_baseline_exactly() -> None:
    import torch

    baseline = LowCapacityDiscreteHazard(2).double()
    with torch.no_grad():
        baseline.bin_logit.copy_(torch.linspace(-4.0, -2.0, 6, dtype=torch.float64))
        baseline.beta.copy_(torch.as_tensor([0.3, -0.7], dtype=torch.float64))
    extension = FixedBaselineResidualHazard(baseline, total_width=5, prefix_width=2).double()
    x = torch.as_tensor(
        [[1.0, 2.0, 0.0, 0.0, 0.0], [-1.0, 0.5, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    bins = torch.as_tensor([0, 20], dtype=torch.long)
    with torch.no_grad():
        expected = baseline(x[:, :2], bins)
        observed = extension(x, bins)
    assert torch.equal(extension.bin_logit, baseline.bin_logit)
    assert torch.equal(extension.base_beta, baseline.beta)
    assert torch.allclose(observed, expected)
    assert set(dict(extension.named_parameters())) == {"beta"}


def test_h2b_field_state_fit_is_a_residual_over_frozen_base() -> None:
    rng = np.random.default_rng(13)
    base_x = np.column_stack((np.ones(40), rng.normal(size=40)))
    state_x = rng.normal(size=(40, 2))
    base_coef = np.asarray([[1.2], [-0.4]])
    truth = np.asarray([[0.7], [-0.2]])
    y = base_x @ base_coef + state_x @ truth
    valid = np.ones_like(y, dtype=bool)
    state_coef = _fit_masked_state_residual_ridge(
        base_x, state_x, base_coef, y, valid, np.arange(30),
        alpha=1e-8, centre=np.zeros(1), scale=np.ones(1),
    )
    assert np.allclose(state_coef, truth, atol=1e-5)
    # The helper returns only an appended residual; there is no path by which
    # it can alter or silently re-estimate the inherited field coefficients.
    assert np.array_equal(base_coef, np.asarray([[1.2], [-0.4]]))


def test_h2b_zero_state_extension_cannot_recalibrate_baseline_hazard() -> None:
    anchor = np.arange(0.0, 18000.0, 300.0)
    phase = np.full(anchor.size, "SELECTION", dtype="<U12")
    phase[anchor < 10800.0] = "FIT"
    phase[(anchor >= 10800.0) & (anchor < 14400.0)] = "INNER"
    bounds = {"20pct": -1.0, "60pct": 10800.0, "70pct": 14400.0, "80pct": 18000.0}
    seizures = [
        {"onset_epoch": t, "offset_epoch": t + 30.0}
        for t in (1800.0, 5400.0, 9000.0, 12600.0, 16200.0)
    ]
    baseline = np.zeros((anchor.size, 1), dtype=np.float64)
    result = _fit_nested_hazards(
        anchor, phase, np.asarray([[0.0, 18000.0]]), bounds, seizures,
        {
            "B_context": baseline,
            "B_context_plus_S_dual": np.column_stack((baseline, np.zeros(anchor.size))),
        },
    )
    base = result["B_context"]["selection_censored_likelihood"]["log_score"]
    # Saved fit/scaler diagnostics must survive the actual card JSON boundary.
    import json
    assert json.loads(json.dumps(result))["B_context"]["fitted_readout"]["class"] == "LowCapacityDiscreteHazard"
    state = result["B_context_plus_S_dual"]["selection_censored_likelihood"]["log_score"]
    assert result["B_context_plus_S_dual"]["nested_base"] == "B_context"
    assert state == base
    for name in ('B_context', 'B_context_plus_S_dual'):
        assert result[name]['training']['optimizer'] == 'LBFGS'
        assert np.isfinite(result[name]['training']['trace'][-1]['gradient_max_abs'])
        assert 'state_dict' in result[name]['fitted_readout']
        assert len(result[name]['recipe_audits']) == 6


def test_h2b_clinical_inventory_count_does_not_replace_scored_distinct_onsets() -> None:
    anchor = np.array([0., 10800., 14400.])
    phase = np.array(['FIT', 'INNER', 'SELECTION'])
    bounds = {'20pct': -1., '60pct': 10800., '70pct': 14400., '80pct': 18000.}
    seizures = [{'onset_epoch': t, 'offset_epoch': t + 30.} for t in (1800., 5400., 9000., 12600., 16200.)]
    support = np.column_stack((anchor, anchor + 300.))
    result = _fit_nested_hazards(anchor, phase, support, bounds, seizures,
                                 {'B_history': np.zeros((3, 1))})
    assert result['support']['seizures_by_phase']['FIT'] == 3
    # One FIT anchor can contribute only its next known seizure, even though
    # the clinical inventory contains three FIT onsets. A clinical event may
    # still be known when EEG support has a gap; do not erase that positive.
    assert result['support']['observed_seizures_by_phase']['FIT'] == 1
    assert result['B_history']['status'] == 'NOT_ESTIMABLE'


def test_h2b_persistent_background_is_nested_over_current_background() -> None:
    anchor = np.arange(0.0, 18000.0, 300.0)
    phase = np.full(anchor.size, "SELECTION", dtype="<U12")
    phase[anchor < 10800.0] = "FIT"
    phase[(anchor >= 10800.0) & (anchor < 14400.0)] = "INNER"
    bounds = {"20pct": -1.0, "60pct": 10800.0, "70pct": 14400.0, "80pct": 18000.0}
    seizures = [
        {"onset_epoch": t, "offset_epoch": t + 30.0}
        for t in (1800.0, 5400.0, 9000.0, 12600.0, 16200.0)
    ]
    current = np.column_stack((
        np.zeros(anchor.size, dtype=np.float64),
        np.sin(anchor / 3600.0),
    ))
    zero = np.zeros((anchor.size, 2), dtype=np.float64)
    result = _fit_nested_hazards(
        anchor, phase, np.asarray([[0.0, 18000.0]]), bounds, seizures,
        {
            "B_history_current_background": current,
            "B_history_current_background_plus_S_background": np.column_stack((current, zero)),
            "B_history_current_background_plus_random_background": np.column_stack((current, zero)),
        },
    )
    base = result["B_history_current_background"]["selection_censored_likelihood"]["log_score"]
    for name in (
        "B_history_current_background_plus_S_background",
        "B_history_current_background_plus_random_background",
    ):
        assert result[name]["nested_base"] == "B_history_current_background"
        observed = result[name]["selection_censored_likelihood"]["log_score"]
        assert observed == base
