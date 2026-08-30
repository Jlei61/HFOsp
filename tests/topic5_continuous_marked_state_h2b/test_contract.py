from __future__ import annotations

from pathlib import Path

import pytest

from src.topic5_continuous_marked_state_h2b import contract


def test_frozen_leads_and_primary_estimand() -> None:
    payload = contract.run_contract_payload()
    assert payload["lead_minutes"] == [5, 15, 30, 60, 120]
    assert payload["primary_lead_minutes"] == 30
    assert payload["boundary"]["seizure_loss_updates_state"] is False
    assert payload["patient_first"] is True
    assert payload["seed_is_patient_replicate"] is False


@pytest.mark.parametrize(
    ("n_seizures", "tier"),
    [
        (10, "primary_chronological"),
        (9, "sensitivity_loso"),
        (5, "sensitivity_loso"),
        (4, "descriptive_case_series"),
        (2, "descriptive_case_series"),
        (1, "not_estimable"),
    ],
)
def test_support_tier_boundaries(n_seizures: int, tier: str) -> None:
    assert contract.support_tier(n_seizures) == tier


def test_output_path_cannot_escape_or_enter_r1_7() -> None:
    assert contract.assert_safe_output_path(contract.RESULT_ROOT / "manifests/x.json")
    assert contract.assert_safe_output_path(
        contract.V0_2_RESULT_ROOT / "manifests/x.json"
    )
    assert contract.assert_safe_output_path(
        contract.CANONICAL_V0_2_RESULT_ROOT / "reports/machine_audit.json"
    )
    with pytest.raises(ValueError):
        contract.assert_safe_output_path(Path("/tmp/hfosp_r17_20260827/x.json"))
    with pytest.raises(ValueError):
        contract.assert_safe_output_path(Path("/tmp/not_h2b.json"))
