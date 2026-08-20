import pytest

from scripts.aggregate_topic4_zm_joint_morphology_confirmation import (
    summarize_confirmation,
)


FROZEN = {
    "I_th_EI": 76.0,
    "tau_z": 5000.0,
    "tau_adp": 500.0,
    "eta_m": 0.007,
    "E_to_E_dose": 1.0,
    "E_to_I_dose": 0.02,
}


def _record(seed, passed, onset):
    return {
        "seed": seed,
        "parameters": dict(FROZEN),
        "final_joint_eligible": True,
        "verdict": (
            "JOINT_SUSTAINED_HIGH_OSCILLATORY_STATE_CANARY_PASS"
            if passed else "JOINT_ICTAL_MORPHOLOGY_CANARY_FAIL"
        ),
        "runaway_morphology": {"scientific_onset_ms": onset},
    }


def test_confirmation_accepts_two_of_three_and_uses_median_passing_onset():
    output = summarize_confirmation([
        _record(1831, True, 5000.0),
        _record(1832, False, 4000.0),
        _record(1833, True, 6000.0),
    ], FROZEN)
    assert output["accepted"] is True
    assert output["n_pass"] == 2
    assert output["representative_seed"] == 1831


def test_confirmation_rejects_parameter_drift():
    records = [_record(1831 + index, True, 5000.0 + index) for index in range(3)]
    records[1]["parameters"]["E_to_I_dose"] = 0.03
    with pytest.raises(RuntimeError, match="parameter E_to_I_dose drifted"):
        summarize_confirmation(records, FROZEN)
