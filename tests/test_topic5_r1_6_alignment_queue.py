import json

from scripts.topic5_continuous_marked_state_r1 import (
    run_r1_6_alignment_optimizer_queue as queue,
)


def _artifact(*, executed=(4, 4), trajectory_length=9):
    return {
        "status": "COMPLETE",
        "revision": queue.R1_6_REVISION,
        "stage": "optimizer_selection",
        "config_id": "cfg",
        "subject": "subject",
        "seed": 2,
        "development_validation_scored": False,
        "epoch_zero_seen_alignment_selection": False,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "fit_trace": {
            "executed_epochs_by_stage": {
                "observer_alignment": executed[0],
                "joint_alignment": executed[1],
            },
            "trajectory": [
                {"evaluated_train_metrics": {}, "optimizer_steps": 0}
                for _ in range(trajectory_length)
            ],
        },
    }


def test_valid_selection_accepts_consistent_early_stopping(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(_artifact(executed=(3, 3), trajectory_length=7)))
    assert queue.valid_selection(path, "cfg", "subject", 2)


def test_valid_selection_rejects_truncated_or_overlong_trace(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(_artifact(executed=(3, 3), trajectory_length=6)))
    assert not queue.valid_selection(path, "cfg", "subject", 2)
    path.write_text(json.dumps(_artifact(executed=(4, 4), trajectory_length=10)))
    assert not queue.valid_selection(path, "cfg", "subject", 2)
