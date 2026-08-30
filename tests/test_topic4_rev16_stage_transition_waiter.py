from scripts import wait_topic4_rev16_joint_response_then_launch_selection as waiter


def _controller(status=waiter.QUEUE_COMPLETE):
    return {
        "status": status, "n_jobs": 120, "n_complete": 120,
        "n_failed": 0, "n_invalid_artifact": 0,
    }


def _aggregate(status="COMPLETE"):
    return {
        "schema_id": waiter.AGGREGATE_SCHEMA,
        "status": status,
        "inventory": {
            "present_validated": 120, "complete_cartesian_product": True,
        },
        "joint_response_tensor": {"shape": [3, 48]},
        "robust_directions": [{"family": "mean_j14"}],
        "provenance": {"formal_ready": True},
    }


def test_stage_transition_waits_for_both_queue_and_aggregate():
    assert waiter.classify(
        {**_controller("REV16_M4_SHELL_QUEUE_RUNNING"), "n_complete": 114}, None,
    ) == "wait"
    assert waiter.classify(_controller(), None) == "wait"
    assert waiter.classify(_controller(), _aggregate("INCOMPLETE")) == "wait"
    assert waiter.classify(_controller(), _aggregate()) == "ready"


def test_stage_transition_fails_closed_on_invalid_complete_artifact():
    failed = {**_controller(), "n_failed": 1}
    assert waiter.classify(failed, _aggregate()) == "failed"
    invalid = _aggregate()
    invalid["inventory"]["present_validated"] = 119
    assert waiter.classify(_controller(), invalid) == "failed"
    assert waiter.classify(
        _controller("REV16_M4_SHELL_QUEUE_FAILED"), None,
    ) == "failed"
