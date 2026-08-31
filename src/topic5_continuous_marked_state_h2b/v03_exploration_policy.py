"""Validation and freezing for the H2b v0.3 exploration-policy addendum."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contract import (
    CANONICAL_V0_3_RESULT_ROOT,
    H2B_V0_3_REVISION,
    atomic_json,
    sha256_file,
    utc_now,
)


DEFAULT_POLICY = (
    Path(__file__).resolve().parents[2]
    / "config/topic5_continuous_marked_state_h2b_v0_3_exploration_policy.json"
)


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def load_and_validate_exploration_policy(
    path: Path | str = DEFAULT_POLICY,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(
        payload.get("schema_revision") == "h2b_v0_3_exploration_policy_v1",
        "unexpected exploration-policy revision",
    )
    _require(payload.get("status") == "FROZEN_DEVELOPMENT_ADDENDUM",
             "exploration policy is not frozen")
    _require(payload.get("changes_estimands_or_data_boundaries") is False,
             "exploration policy may not change estimands or data boundaries")
    expected_hard = {
        "source_purity", "estimand_integrity", "temporal_integrity",
        "artifact_integrity", "resource_safety",
    }
    _require(set(payload.get("hard_gates") or {}) == expected_hard,
             "hard gates drifted from the minimal safety set")
    claim_gates = payload.get("claim_specific_not_global_gates") or {}
    _require(set(claim_gates) == {
        "A1_state_qualification", "A2_assay_power", "T_increment",
        "M_memory", "D_specificity", "IED_source_ablation",
        "phenotype_bridge",
    }, "claim-specific exploration tracks are incomplete")
    reporting = payload.get("reporting") or {}
    _require(reporting.get("negative_result_never_global_blocker") is True,
             "negative results were promoted to a global blocker")
    _require(reporting.get("engineering_pass_is_not_scientific_support") is True,
             "engineering/science boundary drift")
    for key in ("formal_test_partition_opened", "sealed_opened", "h3_or_t2_run"):
        _require(reporting.get(key) is False, f"forbidden boundary opened: {key}")
    return payload


def freeze_exploration_policy(
    policy_path: Path | str = DEFAULT_POLICY,
    output_path: Path | str = CANONICAL_V0_3_RESULT_ROOT / "exploration_policy.json",
) -> dict[str, Any]:
    source = Path(policy_path).resolve()
    policy = load_and_validate_exploration_policy(source)
    frozen: dict[str, Any] = {
        "status": "FROZEN",
        "revision": H2B_V0_3_REVISION,
        "created_utc": utc_now(),
        "policy_path": str(source),
        "policy_sha256": sha256_file(source),
        "policy": policy,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "h3_or_t2_run": False,
    }
    atomic_json(output_path, frozen)
    return frozen


def assert_frozen_exploration_policy_matches(
    frozen: Mapping[str, Any], policy_path: Path | str = DEFAULT_POLICY,
) -> None:
    source = Path(policy_path).resolve()
    load_and_validate_exploration_policy(source)
    _require(frozen.get("status") == "FROZEN", "policy receipt is not frozen")
    _require(frozen.get("revision") == H2B_V0_3_REVISION,
             "policy receipt revision drift")
    _require(frozen.get("policy_sha256") == sha256_file(source),
             "policy receipt SHA256 drift")
    for key in ("formal_test_partition_opened", "sealed_opened", "h3_or_t2_run"):
        _require(frozen.get(key) is False, f"policy receipt opened {key}")
