from __future__ import annotations

import numpy as np

from src.topic5_continuous_marked_state_h2b.v03_assay import (
    AssayTemplate,
    build_template,
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
    phase = 2 * np.pi * np.arange(n) / 288.0
    history[:, 8] = np.sin(phase)
    history[:, 9] = np.cos(phase)
    return build_template(
        time_epoch=np.arange(n, dtype=np.float64) * 300.0,
        segment=np.zeros(n, dtype=np.int64),
        deterministic_history=history,
        persistent_state=state,
        memoryless_state=memoryless,
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
    assert observed["transfer"]["relative_logloss_improvement"] > 0


def test_wilson_interval_contains_observed_fraction() -> None:
    lower, upper = wilson_interval(80, 100)
    assert lower is not None and lower < 0.8
    assert upper is not None and upper > 0.8
