#!/usr/bin/env python3
"""Validate the code and artifact boundary for the unified Topic 4 Fig4/5 line."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = ROOT / "config/topic4_fig45_data_driven_zm_integration.json"
EXPECTED_SCHEMA = "topic4_fig45_data_driven_zm_integration_v2"


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_contract(
    contract: dict[str, Any],
    *,
    repo_root: Path = ROOT,
    artifact_root: Path | None = None,
    require_artifacts: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    locked_panels: list[str] = []
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
            if figure.get("artifact_locked") and not (
                configured_root / figure["artifact_subdir"]
            ).is_dir():
                errors.append(f"{figure_id}: locked artifact directory is missing")

        fig5_root = configured_root / fig5["artifact_subdir"]
        for panel_id, panel in fig5.get("locked_panels", {}).items():
            panel_key = f"fig5.{panel_id}"
            panel_root = configured_root / panel.get("artifact_subdir", "")
            if not _is_within(panel_root, fig5_root):
                errors.append(f"{panel_key}: artifact_subdir must be inside Fig5")
                continue
            seeds = panel.get("confirmation_seeds", [])
            if len(set(seeds)) < 3 or panel.get("passed_seeds") != len(set(seeds)):
                errors.append(f"{panel_key}: three unique passing confirmation seeds required")
            artifact_errors = False
            for relative_name, expected_sha256 in panel.get("required_artifacts", {}).items():
                artifact = panel_root / relative_name
                if not artifact.is_file():
                    errors.append(f"{panel_key}: missing artifact {relative_name}")
                    artifact_errors = True
                elif _sha256(artifact) != expected_sha256:
                    errors.append(f"{panel_key}: sha256 mismatch for {relative_name}")
                    artifact_errors = True
            if not artifact_errors and panel.get("required_artifacts"):
                locked_panels.append(panel_key)

    policy = contract.get("repository_policy", {})
    if policy.get("runtime_products") != "EXTERNAL_ONLY":
        errors.append("runtime products must remain external")
    forbidden = set(policy.get("forbidden_main_paths", []))
    if "results/" not in forbidden:
        errors.append("results/ must be forbidden from the main integration commit")

    figure_artifacts_ready = {
        figure_id: bool(figure.get("artifact_locked"))
        and (configured_root / figure["artifact_subdir"]).is_dir()
        for figure_id, figure in figures.items()
    }
    artifacts_ready = (
        set(figure_artifacts_ready) == {"fig4", "fig5"}
        and all(figure_artifacts_ready.values())
    )
    if require_artifacts and not artifacts_ready:
        errors.append("full Fig4/Fig5 artifact set is not locked")
    status = "INVALID" if errors else (
        "CONTRACT_VALID_ARTIFACTS_READY" if artifacts_ready
        else "CONTRACT_VALID_PARTIALLY_LOCKED" if locked_panels
        else "CONTRACT_VALID_ARTIFACTS_PENDING"
    )
    return {
        "status": status,
        "artifact_root": str(configured_root),
        "artifacts_ready": artifacts_ready,
        "figure_artifacts_ready": figure_artifacts_ready,
        "locked_panels": locked_panels,
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
