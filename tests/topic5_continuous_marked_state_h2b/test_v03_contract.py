from __future__ import annotations

import json

from src.topic5_continuous_marked_state_h2b.contract import H2B_V0_3_REVISION
from src.topic5_continuous_marked_state_h2b.v03_contract import (
    DEFAULT_CONTRACT,
    assert_frozen_contract_matches,
    freeze_contract,
    load_and_validate_contract,
)


def test_v03_contract_is_nested_outcome_blind_and_development_only() -> None:
    contract = load_and_validate_contract()
    matrices = contract["design_matrices"]
    assert matrices["M2"][:-1] == matrices["M1"]
    assert matrices["M4"][:-1] == matrices["M3"]
    assert contract["state_qualification"]["outcome_blind"] is True
    assert contract["boundaries"]["formal_test_partition_opened"] is False
    assert contract["boundaries"]["sealed_opened"] is False
    assert contract["geometry"]["umap_role"] == "visualisation_only"


def test_v03_contract_freeze_is_hash_bound(tmp_path, monkeypatch) -> None:
    from src.topic5_continuous_marked_state_h2b import contract as boundary

    monkeypatch.setattr(boundary, "CANONICAL_V0_3_RESULT_ROOT", tmp_path)
    monkeypatch.setattr(boundary, "V0_3_RESULT_ROOT", tmp_path)
    output = tmp_path / "analysis_contract.json"
    frozen = freeze_contract(DEFAULT_CONTRACT, output)
    observed = json.loads(output.read_text(encoding="utf-8"))
    assert observed == frozen
    assert frozen["revision"] == H2B_V0_3_REVISION
    assert_frozen_contract_matches(observed)
