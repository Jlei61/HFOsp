from src.topic5_continuous_marked_state_r1.synthetic_recovery import (
    run_synthetic_recovery,
)


def test_in_family_persistent_truth_beats_baseline_and_wrong_time() -> None:
    result = run_synthetic_recovery(seed=4, epochs=60)
    assert result["recovered"] is True
    assert result["filtered_minus_wrong_time"] < -0.02
