"""Machine-readable H3 audit schema."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


SCHEMA_VERSION = "group_event_state.v034_h3_estimability.v1"
FORBIDDEN_RESULT_KEYS = {"dev_test", "sealed", "formal_test", "seizure_probe", "paper_ready"}


def _walk_keys(value: Any):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_keys(child)


def build_machine_report(*, subjects: list[dict], canary: dict, arm_contracts: list[dict], config: dict) -> dict:
    core = []
    for subject in subjects:
        core.extend(subject.get("core_eligible_designs", []))
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "CPU_AUDIT_COMPLETE_NO_HUMAN_MODEL_FIT",
        "data_scope": {
            "structural_coverage_metadata": True,
            "seizure_boundaries_used_only_to_cut_coverage": True,
            "event_times_used_phases": ["state_train", "inner_val"],
            "model_outputs_read": False,
            "development_outcomes_read": False,
            "sealed_partition_read": False,
        },
        "config": dict(config),
        "arm_contracts": list(arm_contracts),
        "synthetic_canary": dict(canary),
        "subjects": list(subjects),
        "core_eligible_designs": core,
        "n_subjects": len(subjects),
        "n_core_eligible_designs": len(core),
    }
    validate_machine_report(report)
    return report


def validate_machine_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("wrong H3 audit schema version")
    scope = report.get("data_scope", {})
    if scope.get("model_outputs_read") or scope.get("development_outcomes_read") or scope.get("sealed_partition_read"):
        raise ValueError("H3 estimability audit must not read model/development/sealed outcomes")
    # Exact tokens are banned.  Compound audit labels such as
    # ``development_outcomes_read`` are intentionally part of the scope above.
    found = set(_walk_keys(report)) & FORBIDDEN_RESULT_KEYS
    if found:
        raise ValueError(f"forbidden result keys in estimability report: {sorted(found)}")
