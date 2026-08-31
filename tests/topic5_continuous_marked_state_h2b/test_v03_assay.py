from __future__ import annotations

import numpy as np

from src.topic5_continuous_marked_state_h2b.v03_assay import (
    AssayTemplate,
    _lead_rows_for_onsets,
    build_template,
    residualise_train_test,
    run_replicate,
    simulate_world,
    wilson_interval,
)


def _template() -> AssayTemplate:
    rng = np.random.default_rng(4)
    n = 500
    state = np.zeros((n, 4))
    for index in range(1, n):
        state[index] = 0.97 * state[index - 1] + rng.normal(scale=0.2, size=4)
    memoryless = rng.normal(scale=0.2, size=(n, 4))
    history = rng.normal(size=(n, 11))
    observation = rng.normal(size=(n, 12))
    phase = 2 * np.pi * np.arange(n) / 288.0
    history[:, 8] = np.sin(phase)
    history[:, 9] = np.cos(phase)
    return build_template(
        time_epoch=np.arange(n, dtype=np.float64) * 300.0,
        segment=np.zeros(n, dtype=np.int64),
        deterministic_history=history,
        current_observation=observation,
        persistent_decoder=state,
        memoryless_decoder=memoryless,
        n_seizures=14,
    )


def test_assay_world_preserves_fixed_seizure_count_and_coverage() -> None:
    template = _template()
    generated = simulate_world(
        template, "persistent_state", np.random.default_rng(9), effect_scale=2.0,
    )
    assert len(generated["onset_index"]) == template.n_seizures
    assert np.all(np.diff(generated["onset_index"]) > 0)
    assert len(generated["outcome"]) == len(template.time_epoch)


def test_persistent_world_is_detectable_at_strong_effect() -> None:
    observed = run_replicate(
        _template(), "persistent_state", 22, initial_k=3, effect_scale=3.5,
    )
    assert observed["transfer"]["status"] == "COMPLETE"
    assert observed["transfer"]["T_relative_logloss_improvement"] > 0
    assert "M_relative_logloss_improvement" in observed["transfer"]
    assert "lag_degradation" in observed["transfer"]


def test_synthetic_onsets_have_same_segment_30_minute_leads() -> None:
    template = _template()
    onset, lead = _lead_rows_for_onsets(template)
    assert len(onset) == len(lead) > 0
    assert np.all(template.segment[onset] == template.segment[lead])
    observed = template.time_epoch[onset] - template.time_epoch[lead]
    assert np.all(np.abs(observed - 30.0 * 60.0) <= 7.5 * 60.0)


def test_residualisation_never_uses_test_rows_to_fit() -> None:
    rng = np.random.default_rng(3)
    x_train = rng.normal(size=(80, 3))
    x_test = rng.normal(size=(20, 3))
    y_train = x_train @ rng.normal(size=(3, 2))
    y_test = x_test @ rng.normal(size=(3, 2))
    train_a, _ = residualise_train_test(y_train, y_test, x_train, x_test)
    train_b, _ = residualise_train_test(
        y_train, y_test + 1e6, x_train, x_test + 1e6,
    )
    assert np.array_equal(train_a, train_b)


def test_wilson_interval_contains_observed_fraction() -> None:
    lower, upper = wilson_interval(80, 100)
    assert lower is not None and lower < 0.8
    assert upper is not None and upper > 0.8
