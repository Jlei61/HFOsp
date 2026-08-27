from src.topic5_continuous_marked_state_r1.optimizer_synthetic import (
    run_optimizer_synthetic,
)


def test_optimizer_synthetic_recovers_in_family_signal() -> None:
    result = run_optimizer_synthetic(
        seed=4, truth="positive", epochs=60, learning_rate=3e-3
    )
    assert result["selected_epoch"] > 0
    assert result["test_minus_baseline"] < 0
    assert result["test_minus_wrong_time"] < 0
    assert result["recovered"] is True
