"""Pure interictal-to-target gate logic for Topic-5 v2.2."""
from __future__ import annotations

from typing import Any


def evaluate_interictal_target_gate(
    *,
    claim2: dict[str, Any],
    claim3: dict[str, Any] | None,
    claim4: dict[str, Any] | None,
    target_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the frozen gate without calling locked stages failures.

    Claim 2 is an executed scientific test.  Claims 3 and 4 are downstream
    tests that must remain locked when Claim 2 fails.  Keeping those states
    distinct is essential for a faithful pre-registered closeout.
    """
    claim2_next = str(claim2.get("claim2_next", "NOT_RUN")).upper()
    claim2_future = str(claim2.get("claim2_future", "NOT_RUN")).upper()
    claim2_pass = claim2_next == "PASS" and claim2_future == "PASS"
    if claim3 is None:
        claim3_status = "LOCKED_NOT_RUN" if not claim2_pass else "NOT_RUN"
    else:
        claim3_status = str(
            claim3.get("claim3_random_axis", "NOT_RUN")
        ).upper()
    claim3_pass = claim3_status == "PASS"
    if claim4 is None:
        claim4_status = (
            "LOCKED_NOT_RUN"
            if not (claim2_pass and claim3_pass)
            else "NOT_RUN"
        )
    else:
        claim4_status = str(
            claim4.get("claim4_shared_scaffold", "NOT_RUN")
        ).upper()
    claim_statuses = {
        "claim2_next": claim2_next,
        "claim2_future": claim2_future,
        "claim3_random_axis": claim3_status,
        "claim4_shared_scaffold": claim4_status,
    }
    items = {
        name: status == "PASS" for name, status in claim_statuses.items()
    }
    interictal_pass = all(items.values())
    metadata_ready = bool(
        target_metadata.get("source_contact_metadata_ready")
        and target_metadata.get("primary_transfer_metadata_ready")
        and target_metadata.get("early_ictal_transfer_allowed")
    )
    target_unlocked = bool(interictal_pass and metadata_ready)
    failed_items = [
        name for name, status in claim_statuses.items() if status == "FAIL"
    ]
    locked_items = [
        name
        for name, status in claim_statuses.items()
        if status == "LOCKED_NOT_RUN"
    ]
    blockers = []
    if failed_items:
        blockers.append(
            "interictal gate failed: " + ", ".join(failed_items)
        )
    if locked_items:
        blockers.append(
            "downstream claims locked by the frozen stop rule: "
            + ", ".join(locked_items)
        )
    if not metadata_ready:
        blockers.append(
            str(
                target_metadata.get(
                    "blocker",
                    "exact per-seizure clinical-onset source metadata is unavailable",
                )
            )
        )
    if interictal_pass and metadata_ready:
        transfer_status = "UNLOCKED"
    elif not interictal_pass and not metadata_ready:
        transfer_status = (
            "BLOCKED_INTERICTAL_GATE_AND_MISSING_SOURCE_METADATA"
        )
    elif not interictal_pass:
        transfer_status = "BLOCKED_INTERICTAL_GATE"
    else:
        transfer_status = "BLOCKED_MISSING_SOURCE_METADATA"
    return {
        "items": items,
        "claim_statuses": claim_statuses,
        "interictal_pass": interictal_pass,
        "source_metadata_ready": metadata_ready,
        "early_ictal_values_unlocked": target_unlocked,
        "early_ictal_transfer_status": transfer_status,
        "failed_items": failed_items,
        "locked_items": locked_items,
        "blockers": blockers,
    }
