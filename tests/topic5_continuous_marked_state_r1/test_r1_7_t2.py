from src.topic5_continuous_marked_state_r1.r1_7_t2 import (
    R1_7_T2_REVISION, is_expected_support_limit,
)


def test_r1_7_t2_revision_is_frozen() -> None:
    assert R1_7_T2_REVISION == "r1_7a_d_mechanism_t2_r2_n100_v1"


def test_only_declared_support_failures_are_non_estimable() -> None:
    assert is_expected_support_limit(
        ValueError("state-matched placebo has too few TRAIN donors")
    )
    assert is_expected_support_limit(
        ValueError("T2-R2.0 H10 has no within-segment pairs")
    )
    assert not is_expected_support_limit(ValueError("placebo arrays disagree"))
    assert not is_expected_support_limit(ValueError("checkpoint hash mismatch"))
