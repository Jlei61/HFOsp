#!/usr/bin/env python3
"""Validate the code and artifact boundary for the unified Topic 4 Fig4/5 line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = ROOT / "config/topic4_fig45_data_driven_zm_integration.json"
EXPECTED_SCHEMA = "topic4_fig45_data_driven_zm_integration_v1"


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def validate_contract(
    contract: dict[str, Any],
    *,
    repo_root: Path = ROOT,
    artifact_root: Path | None = None,
    require_artifacts: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    if contract.get("schema_version") != EXPECTED_SCHEMA:
        errors.append("unexpected schema_version")

    configured_root = artifact_root or Path(str(contract.get("artifact_root", "")))
    if not configured_root.is_absolute():
        errors.append("artifact_root must be absolute")
    elif _is_within(configured_root, repo_root):
        errors.append("artifact_root must be outside the Git repository")

    figures = contract.get("figures", {})
    if set(figures) != {"fig4", "fig5"}:
        errors.append("contract must define exactly fig4 and fig5")
    else:
        fig4 = figures["fig4"]
        fig5 = figures["fig5"]
        if fig5.get("inherits_frozen_substrate_from") != fig4.get("model_id"):
            errors.append("fig5 must inherit the exact frozen Fig4 substrate")
        if set(fig5.get("added_state_variables", [])) != {"Z", "M"}:
            errors.append("fig5 may add exactly the Z and M state variables")
        for figure_id, figure in figures.items():
            for producer in figure.get("producers", []):
                producer_path = repo_root / producer
                if producer.startswith("results/") or not producer_path.is_file():
                    errors.append(f"{figure_id}: invalid producer {producer}")
            if require_artifacts and not (configured_root / figure["artifact_subdir"]).is_dir():
                errors.append(f"{figure_id}: artifact directory is missing")

    policy = contract.get("repository_policy", {})
    if policy.get("runtime_products") != "EXTERNAL_ONLY":
        errors.append("runtime products must remain external")
    forbidden = set(policy.get("forbidden_main_paths", []))
    if "results/" not in forbidden:
        errors.append("results/ must be forbidden from the main integration commit")

    artifacts_ready = (
        configured_root.is_dir()
        and all(
            (configured_root / row["artifact_subdir"]).is_dir()
            for row in figures.values()
        )
        if set(figures) == {"fig4", "fig5"}
        else False
    )
    status = "INVALID" if errors else (
        "CONTRACT_VALID_ARTIFACTS_READY" if artifacts_ready
        else "CONTRACT_VALID_ARTIFACTS_PENDING"
    )
    return {
        "status": status,
        "artifact_root": str(configured_root),
        "artifacts_ready": artifacts_ready,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--require-artifacts", action="store_true")
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    report = validate_contract(
        contract,
        artifact_root=args.artifact_root,
        require_artifacts=args.require_artifacts,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] != "INVALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
