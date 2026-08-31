"""Machine validation for the frozen H2b v0.3 development contract."""
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


DEFAULT_CONTRACT = (
    Path(__file__).resolve().parents[2]
    / "config/topic5_continuous_marked_state_h2b_v0_3.json"
)


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def load_and_validate_contract(path: Path | str = DEFAULT_CONTRACT) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    _require(payload.get("schema_revision") == "h2b_v0_3_analysis_contract_v1",
             "unexpected v0.3 contract revision")
    _require(payload.get("status") == "FROZEN_DEVELOPMENT",
             "v0.3 contract is not frozen development")
    _require(payload.get("v0_2_informed_redesign") is True,
             "v0.3 must disclose that v0.2 outcomes informed the redesign")
    _require(payload.get("independent_confirmation") is False,
             "v0.3 cannot be labelled independent confirmation")

    boundary = payload.get("boundaries") or {}
    _require(boundary.get("development_only") is True,
             "v0.3 must remain development-only")
    for key in (
        "formal_test_partition_opened", "sealed_opened", "h3_or_t2_run",
        "physical_clock_run", "clinical_predictor_claim_allowed",
        "mechanism_or_causality_claim_allowed",
    ):
        _require(boundary.get(key) is False, f"v0.3 boundary {key} is not false")
    source_spec = payload.get("source") or {}
    _require(source_spec.get("seizure_supervision_updates_state") is False,
             "seizure supervision may not update the frozen state")
    _require(source_spec.get("all_state_components_frozen_before_seizure_analysis") is True,
             "state components are not explicitly frozen")

    time_axis = payload.get("time_axis") or {}
    _require(int(time_axis.get("primary_horizon_minutes", -1)) == 30,
             "30 min is the frozen primary horizon")
    _require(int(time_axis.get("anchor_grid_minutes", -1)) == 5,
             "v0.3 requires the frozen five-minute grid")
    _require(time_axis.get("causal_cutoff") == "t <= anchor_time",
             "causal cutoff drift")

    matrices = payload.get("design_matrices") or {}
    expected = {
        "M0": ["C", "H"],
        "M1": ["C", "H", "O"],
        "M2": ["C", "H", "O", "Z_persistent"],
        "M3": ["C", "H", "O", "Z_memoryless"],
        "M4": ["C", "H", "O", "Z_memoryless", "R_persistent_history"],
    }
    _require(matrices == expected, "v0.3 nested design matrices drifted")
    _require(matrices["M2"][:-1] == matrices["M1"],
             "M2 is not a strict state increment over M1")
    _require(matrices["M4"][:-1] == matrices["M3"],
             "M4 is not a strict history-residual increment over M3")

    qualification = payload.get("state_qualification") or {}
    _require(qualification.get("outcome_blind") is True,
             "state qualification must be seizure-outcome blind")
    geometry = payload.get("geometry") or {}
    _require(geometry.get("fit_population")
             == "outer-training clean interictal continuous trajectories only",
             "manifold geometry is not training-fold clean-interictal only")
    _require(geometry.get("umap_role") == "visualisation_only",
             "UMAP was promoted beyond visualisation")
    _require(geometry.get("seizure_label_supervised_embedding_forbidden") is True,
             "seizure-supervised geometry is not forbidden")
    gates = payload.get("gates") or {}
    _require(gates.get("sealed_unlock") == "forbidden in v0.3",
             "sealed partition is not fail-closed")
    return payload


def freeze_contract(
    contract_path: Path | str = DEFAULT_CONTRACT,
    output_path: Path | str = CANONICAL_V0_3_RESULT_ROOT / "analysis_contract.json",
) -> dict[str, Any]:
    source = Path(contract_path).resolve()
    contract = load_and_validate_contract(source)
    frozen: dict[str, Any] = {
        "status": "FROZEN",
        "revision": H2B_V0_3_REVISION,
        "created_utc": utc_now(),
        "contract_path": str(source),
        "contract_sha256": sha256_file(source),
        "contract": contract,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "h3_or_t2_run": False,
        "seizure_gradient_path": False,
    }
    atomic_json(output_path, frozen)
    return frozen


def assert_frozen_contract_matches(
    frozen: Mapping[str, Any], contract_path: Path | str = DEFAULT_CONTRACT,
) -> None:
    source = Path(contract_path).resolve()
    load_and_validate_contract(source)
    _require(frozen.get("status") == "FROZEN", "analysis contract is not frozen")
    _require(frozen.get("revision") == H2B_V0_3_REVISION,
             "frozen analysis contract revision drift")
    _require(frozen.get("contract_sha256") == sha256_file(source),
             "frozen analysis contract SHA256 drift")
    for key in ("formal_test_partition_opened", "sealed_opened", "h3_or_t2_run",
                "seizure_gradient_path"):
        _require(frozen.get(key) is False, f"frozen analysis contract {key} is not false")
