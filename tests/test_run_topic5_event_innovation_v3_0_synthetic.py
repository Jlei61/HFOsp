from __future__ import annotations

import numpy as np

from scripts.run_topic5_event_innovation_v3_0_synthetic import (
    controlled_accumulation,
    run_calibration,
    simulate_event_indexed_state,
)


def test_synthetic_state_uses_event_innovation_in_next_transition():
    transition = np.eye(2) * 0.8
    impulse = np.eye(2) * 0.5
    data = simulate_event_indexed_state(
        200,
        transition,
        impulse,
        state_noise=0.0,
        measurement_noise=0.0,
        seed=2,
    )
    expected = data["latent_pre"] @ transition.T + data["true_innovation"] @ impulse.T
    np.testing.assert_allclose(data["latent_future"], expected)


def test_aligned_innovations_accumulate_more_than_cancelling():
    result = controlled_accumulation(
        np.eye(2) * 0.95, np.eye(2) * 0.2, window=20
    )
    assert result["aligned_minus_cancelling"] > 0


def test_full_synthetic_acceptance_is_human_free(tmp_path):
    state = run_calibration(tmp_path, seed=17)
    assert state["status"] == "SYNTHETIC_IDENTIFIABILITY_COMPLETE"
    assert state["human_data_read"] is False
    assert state["one_step_is_one_complete_event"] is True
