from scripts.reconcile_topic5_stateful_event_rnn_v2_7_validation import (
    RECOVERED_SUBJECTS,
)


def test_recovery_subjects_are_explicit_and_unique():
    assert len(RECOVERED_SUBJECTS) == 3
    assert len(set(RECOVERED_SUBJECTS)) == 3
    assert all(subject.startswith("epilepsiae_") for subject in RECOVERED_SUBJECTS)
