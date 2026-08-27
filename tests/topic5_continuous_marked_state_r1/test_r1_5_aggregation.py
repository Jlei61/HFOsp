from scripts.topic5_continuous_marked_state_r1.aggregate_r1_5 import (
    summarise_seed_evidence,
)
from scripts.topic5_continuous_marked_state_r1.run_r1_3_target_observer import (
    R1_5_REVISION,
    requires_recorded_segment_lock,
)
from scripts.topic5_continuous_marked_state_r1.run_r1_5_queue import complete
from scripts.topic5_continuous_marked_state_r1.finalize_r1_5_h3_long import (
    fmt_count,
)


def _payload(epoch: int, checkpoint_hash: str, value: float = -0.1) -> dict:
    return {
        "fit_trace": {"selected_total_epoch": epoch},
        "checkpoint_sha256": checkpoint_hash,
        "validation": {
            "persistent_minus_memoryless": {
                "joint_nll_per_event": value,
            },
            "mark_endpoints": {
                "persistent_minus_memoryless": {
                    "first_group_subset_nll_per_event": value,
                    "continuation_subset_nll_per_event": value,
                },
            },
            "strict_matched_wrong_time": {
                "correct_minus_wrong_median": {
                    "joint_nll_per_event": value,
                },
            },
        },
    }


def test_epoch_zero_is_not_a_directional_negative_or_positive() -> None:
    values = [_payload(0, f"zero-{seed}") for seed in range(5)]
    summary = summarise_seed_evidence(values, "independent_extension")
    assert summary["updated_seeds"] == 0
    assert summary["persistent_estimable_seeds"] == 0
    assert summary["persistent_favourable_seeds"] == 0
    assert summary["joint_stable_seeds"] == 0
    assert summary["stable_explicit_t1_for_h3"] is False


def test_report_count_formatter_does_not_add_contrast_sign() -> None:
    assert fmt_count(None) == "NA"
    assert fmt_count(3.0) == "3"
    assert fmt_count(2.5) == "2.5"


def test_patient_stability_requires_three_distinct_checkpoints() -> None:
    values = [
        _payload(0, "epoch-zero"),
        _payload(1, "a"),
        _payload(1, "a"),
        _payload(1, "b"),
        _payload(1, "c"),
    ]
    summary = summarise_seed_evidence(values, "independent_extension")
    assert summary["updated_seeds"] == 4
    assert summary["persistent_favourable_seeds"] == 4
    assert summary["persistent_estimable_seeds"] == 4
    assert summary["joint_stable_seeds"] == 4
    assert summary["joint_stable_distinct_checkpoints"] == 3
    assert summary["stable_explicit_t1_for_h3"] is True

    for value in values[1:]:
        value["checkpoint_sha256"] = "same"
    summary = summarise_seed_evidence(values, "independent_extension")
    assert summary["joint_stable_seeds"] == 4
    assert summary["joint_stable_distinct_checkpoints"] == 1
    assert summary["stable_explicit_t1_for_h3"] is False


def test_r1_5_requires_recorded_segment_lock() -> None:
    assert requires_recorded_segment_lock(R1_5_REVISION) is True


def test_resume_rejects_pre_segment_lock_result(tmp_path) -> None:
    path = tmp_path / "result.json"
    payload = {
        "status": "COMPLETE",
        "sealed_opened": False,
        "experiment_label": R1_5_REVISION,
        "target_observer_runner_revision": (
            "r1_3_target_observer_segment_locked_v2"
        ),
        "target_observer_runner_sha256": "expected",
        "recorded_coverage_segment_lock_required": True,
        "validation": {
            "strict_matched_wrong_time": {
                "audit": {"same_recorded_coverage_segment": True},
            },
        },
    }
    import json
    path.write_text(json.dumps(payload))
    assert complete(
        path, experiment=R1_5_REVISION, runner_sha256="expected"
    ) is True
    payload["validation"]["strict_matched_wrong_time"]["audit"][
        "same_recorded_coverage_segment"
    ] = False
    path.write_text(json.dumps(payload))
    assert complete(
        path, experiment=R1_5_REVISION, runner_sha256="expected"
    ) is False
