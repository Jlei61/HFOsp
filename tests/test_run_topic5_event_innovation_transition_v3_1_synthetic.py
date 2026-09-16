from __future__ import annotations

from scripts.run_topic5_event_innovation_transition_v3_1_synthetic import (
    discrete_switching_scores,
    run_calibration,
)


def test_discrete_control_wins_for_true_switching_process():
    score = discrete_switching_scores(seed=4)
    assert score["switching_gain"] > 0


def test_v31_synthetic_acceptance_is_matched_and_human_free(tmp_path):
    state = run_calibration(tmp_path, seed=11)
    assert state["status"] == "SYNTHETIC_TRANSITION_IDENTIFICATION_COMPLETE"
    assert state["human_data_read"] is False
    assert state["v3_0_handoff_read"] is False
    assert state["shared_parameter_registry"]["only_added_parameter"] == "event_transition_B"
