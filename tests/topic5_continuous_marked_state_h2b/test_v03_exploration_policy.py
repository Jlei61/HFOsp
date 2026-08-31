from __future__ import annotations

import json

from src.topic5_continuous_marked_state_h2b.v03_exploration_policy import (
    DEFAULT_POLICY,
    assert_frozen_exploration_policy_matches,
    freeze_exploration_policy,
    load_and_validate_exploration_policy,
)


def test_exploration_policy_has_only_safety_hard_gates() -> None:
    policy = load_and_validate_exploration_policy()
    assert set(policy["hard_gates"]) == {
        "source_purity", "estimand_integrity", "temporal_integrity",
        "artifact_integrity", "resource_safety",
    }
    assert policy["reporting"]["negative_result_never_global_blocker"] is True
    assert "phenotype_bridge" in policy["claim_specific_not_global_gates"]


def test_exploration_policy_freeze_is_hash_bound(tmp_path, monkeypatch) -> None:
    from src.topic5_continuous_marked_state_h2b import contract as boundary

    monkeypatch.setattr(boundary, "CANONICAL_V0_3_RESULT_ROOT", tmp_path)
    monkeypatch.setattr(boundary, "V0_3_RESULT_ROOT", tmp_path)
    output = tmp_path / "exploration_policy.json"
    frozen = freeze_exploration_policy(DEFAULT_POLICY, output)
    assert json.loads(output.read_text(encoding="utf-8")) == frozen
    assert_frozen_exploration_policy_matches(frozen)
