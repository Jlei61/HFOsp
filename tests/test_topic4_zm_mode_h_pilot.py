from src.topic4_zm_mode_h_pilot import adjudicate_mode_h_pilot


def _row(**updates):
    row = {
        "episode_status": "onset_persistent", "onset_ms": 1000.0,
        "offset_ms": None, "runaway": False, "H_peak": 0.5,
        "returning_event": False, "returning_distribution": False,
        "post_onset_deep_gap_fraction": 0.3,
    }
    row.update(updates)
    return row


def _panel():
    return {
        "baseline": _row(H_peak=0.0),
        "rho05_gate": _row(), "rho05_nomgate": _row(),
        "rho1_gate": _row(), "rho1_nomgate": _row(),
    }


def test_exit_requires_matched_no_m_gate_to_persist():
    rows = _panel()
    rows["rho05_gate"] = _row(
        episode_status="onset_durable_offset", offset_ms=5000.0,
        returning_event=True,
    )
    verdict = adjudicate_mode_h_pilot(rows)
    assert verdict["verdict"] == "M_GATED_EXIT_WITH_RETURNING_EVENT"


def test_offset_in_both_arms_is_not_a_causal_m_gate_exit():
    rows = _panel()
    rows["rho05_gate"] = _row(episode_status="onset_durable_offset", offset_ms=5000.0)
    rows["rho05_nomgate"] = _row(episode_status="onset_durable_offset", offset_ms=5100.0)
    assert adjudicate_mode_h_pilot(rows)["verdict"] == "NO_LIFECYCLE_DIRECTION"


def test_unengaged_sensor_fails_before_dynamics_are_interpreted():
    rows = _panel()
    for key in rows:
        rows[key]["H_peak"] = 0.01
    assert adjudicate_mode_h_pilot(rows)["verdict"] == "NO_GO_H_NOT_ENGAGED"


def test_tighter_m_gate_can_use_the_same_no_gate_control():
    rows = _panel()
    rows["rho1_mc30"] = _row(
        episode_status="onset_durable_offset", offset_ms=4200.0,
    )
    verdict = adjudicate_mode_h_pilot(rows)
    assert verdict["verdict"] == "M_GATED_EXIT_WITHOUT_INTERICTAL_RETURN"
    assert verdict["causal_pairs"][0]["m_mode_half"] == 30.0
