from scripts.topic5_continuous_marked_state_r1.aggregate_r1_6_optimizer_confirmation import (
    classify_subject,
)


def test_confirmation_classification_separates_failure_modes():
    assert classify_subject(
        stable=3, stable_independent=1, selected_nonzero=4,
        train_favourable=5, overfit_pass=3,
    ) == "OPTIMIZATION_ROBUST_SUPPORT"
    assert classify_subject(
        stable=1, stable_independent=0, selected_nonzero=2,
        train_favourable=5, overfit_pass=3,
    ) == "OPTIMIZER_SENSITIVE_SUPPORT"
    assert classify_subject(
        stable=0, stable_independent=0, selected_nonzero=0,
        train_favourable=5, overfit_pass=3,
    ) == "GENERALISATION_FAILURE_OR_CURRENT_MODEL_NONIDENTIFIABLE"
    assert classify_subject(
        stable=0, stable_independent=0, selected_nonzero=0,
        train_favourable=0, overfit_pass=0,
    ) == "OPTIMIZATION_FAILURE_OR_INSUFFICIENT_DIAGNOSTIC"
