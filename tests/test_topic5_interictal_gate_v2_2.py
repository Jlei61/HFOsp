from __future__ import annotations

from src.topic5_interictal_gate_v2_2 import evaluate_interictal_target_gate


def _claim2(passed: bool = True) -> dict:
    value = "PASS" if passed else "FAIL"
    return {"claim2_next": value, "claim2_future": value}


def _metadata(ready: bool) -> dict:
    return {
        "source_contact_metadata_ready": ready,
        "primary_transfer_metadata_ready": ready,
        "early_ictal_transfer_allowed": ready,
        "blocker": "missing exact source",
    }


def test_all_four_claims_and_metadata_are_required() -> None:
    gate = evaluate_interictal_target_gate(
        claim2=_claim2(),
        claim3={"claim3_random_axis": "PASS"},
        claim4={"claim4_shared_scaffold": "PASS"},
        target_metadata=_metadata(True),
    )
    assert gate["interictal_pass"]
    assert gate["early_ictal_values_unlocked"]


def test_metadata_block_keeps_target_sealed_after_interictal_pass() -> None:
    gate = evaluate_interictal_target_gate(
        claim2=_claim2(),
        claim3={"claim3_random_axis": "PASS"},
        claim4={"claim4_shared_scaffold": "PASS"},
        target_metadata=_metadata(False),
    )
    assert gate["interictal_pass"]
    assert not gate["early_ictal_values_unlocked"]
    assert gate["blockers"] == ["missing exact source"]


def test_absent_downstream_claims_are_locked_after_claim2_failure() -> None:
    gate = evaluate_interictal_target_gate(
        claim2=_claim2(False),
        claim3=None,
        claim4=None,
        target_metadata=_metadata(False),
    )
    assert gate["failed_items"] == ["claim2_next", "claim2_future"]
    assert gate["locked_items"] == [
        "claim3_random_axis",
        "claim4_shared_scaffold",
    ]
    assert gate["claim_statuses"] == {
        "claim2_next": "FAIL",
        "claim2_future": "FAIL",
        "claim3_random_axis": "LOCKED_NOT_RUN",
        "claim4_shared_scaffold": "LOCKED_NOT_RUN",
    }
    assert (
        gate["early_ictal_transfer_status"]
        == "BLOCKED_INTERICTAL_GATE_AND_MISSING_SOURCE_METADATA"
    )
    assert not gate["early_ictal_values_unlocked"]
