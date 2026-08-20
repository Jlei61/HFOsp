#!/usr/bin/env python3
"""Freeze the rev4 discovery boundary before any candidate is rescored.

This producer is deliberately the only place that knows which inputs the
pre-freeze audit may read. Every later rev4 producer imports ``guard_forbidden``
from here, so "did this script touch a clinical ictal endpoint?" has one answer
instead of one answer per script.

The verification is fail-closed by design: a missing or drifted immutable input
raises with the exact path, because a boundary file that silently records a
different substrate is worse than no boundary file at all.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_core_field_runner import atomic_write_json  # noqa: E402

AUDIT_CONFIG = "config/topic4_data_driven_zm_discovery_audit_v1.json"
BOUNDARY_STATUS = "DEVELOPMENT_ONLY_RETROSPECTIVE_DISCOVERY_AUDIT"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_audit_config(path=AUDIT_CONFIG):
    resolved = ROOT / path
    if not resolved.exists():
        raise RuntimeError(f"discovery audit config missing: {path}")
    config = json.loads(resolved.read_text())
    if config.get("status") != BOUNDARY_STATUS:
        raise RuntimeError("the audit config must declare the non-blind status")
    return config


class ForbiddenInputError(RuntimeError):
    """A pre-freeze producer tried to read a clinical ictal endpoint."""


def guard_forbidden(config, path):
    """Hard-fail if ``path`` is a clinical ictal endpoint (spec section 8).

    Called by every pre-freeze producer on every path it opens. Warning instead
    of raising would let a selection quietly become target-informed, which is
    precisely the failure this revision exists to prevent.
    """
    rule = config["forbidden_pre_freeze_inputs"]
    text = str(path).replace(os.sep, "/")
    relative = text
    root = str(ROOT).replace(os.sep, "/")
    if relative.startswith(root):
        relative = relative[len(root):].lstrip("/")
    for prefix in rule["path_prefixes"]:
        if relative.startswith(prefix) or text.startswith(prefix):
            raise ForbiddenInputError(
                f"forbidden pre-freeze input (prefix {prefix!r}): {path}")
    lowered = text.lower()
    for token in rule["path_substrings"]:
        if token in lowered:
            raise ForbiddenInputError(
                f"forbidden pre-freeze input (substring {token!r}): {path}")
    return str(path)


def verify_immutable_inputs(config):
    """Resolve and hash every declared input, or fail with the exact path."""
    records = {}
    for key, record in config["immutable_inputs"].items():
        path = ROOT / record["path"]
        if not path.exists():
            raise RuntimeError(f"immutable input missing: {record['path']}")
        guard_forbidden(config, record["path"])
        observed = sha256_file(path)
        records[key] = {"path": record["path"], "expected": record["sha256"],
                        "observed": observed,
                        "match": observed == record["sha256"]}
        if observed != record["sha256"]:
            raise RuntimeError(
                f"immutable input hash changed: {record['path']} "
                f"expected {record['sha256']} observed {observed}")
    for key, record in config["governing_documents"].items():
        path = ROOT / record["path"]
        if not path.exists():
            raise RuntimeError(f"governing document missing: {record['path']}")
        observed = sha256_file(path)
        records[f"document_{key}"] = {
            "path": record["path"], "expected": record["sha256"],
            "observed": observed, "match": observed == record["sha256"]}
        if observed != record["sha256"]:
            raise RuntimeError(
                f"governing document changed: {record['path']}")
    return records


def montage_of_record(config):
    """The 15 contacts, their shafts, sheet positions and display order."""
    contract = json.loads(
        (ROOT / config["immutable_inputs"]["contact_contract"]["path"]).read_text())
    contacts = contract["contacts"]
    return {
        "n_contacts": len(contacts),
        "display_order": [row["contact_name"] for row in contacts],
        "contact_index": [int(row["contact_index"]) for row in contacts],
        "shaft_ids": [row["shaft_id"] for row in contacts],
        "sheet_xy_mm": [[float(v) for v in row["sheet_xy_mm"]] for row in contacts],
        "shaft_groups": {
            shaft: [row["contact_name"] for row in contacts
                    if row["shaft_id"] == shaft]
            for shaft in sorted({row["shaft_id"] for row in contacts})},
        "source": config["immutable_inputs"]["contact_contract"]["path"],
    }


def _inventory_dir(config, directory, pattern, *, hash_npz):
    root = ROOT / directory
    if not root.exists():
        return {"path": directory, "present": False, "files": []}
    files = []
    for path in sorted(root.glob(pattern)):
        guard_forbidden(config, path)
        row = {"name": path.name, "bytes": path.stat().st_size,
               "sha256": sha256_file(path)}
        companion = path.with_suffix(".npz")
        if hash_npz and companion.exists():
            row["npz_bytes"] = companion.stat().st_size
            row["npz_sha256"] = sha256_file(companion)
        elif hash_npz:
            row["npz_bytes"] = None
            row["npz_sha256"] = None
        files.append(row)
    return {"path": directory, "present": True, "n_files": len(files),
            "files": files}


def candidate_inventory(config):
    roots = config["candidate_inventory_roots"]
    inventory = {
        "exact_carryover_and_pathway_arms": _inventory_dir(
            config, roots["exact_carryover_and_pathway_arms"], "*.json",
            hash_npz=True),
        "calibrated_transition_candidates": [
            _inventory_dir(config, directory, "*.json", hash_npz=True)
            for directory in roots["calibrated_transition_candidates"]],
        "repertoire_reference_workers": _inventory_dir(
            config, roots["repertoire_reference_workers"], "*.npz",
            hash_npz=False),
        "existing_fig5_artifacts": [
            _inventory_dir(config, directory + "/figures", "*.json",
                           hash_npz=False)
            for directory in roots["existing_fig5_artifacts"]],
    }
    return inventory


def git_provenance():
    def _git(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(ROOT), *args],
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:  # pragma: no cover - provenance is best effort
            return None
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_worktree": str(ROOT),
        "tracked_dirty_files": [
            line for line in (_git("status", "--porcelain") or "").splitlines()
            if line and not line.startswith("??")],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=AUDIT_CONFIG)
    args = parser.parse_args()

    config = load_audit_config(args.config)
    output_root = ROOT / config["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)

    hashes = verify_immutable_inputs(config)
    boundary = {
        "status": BOUNDARY_STATUS,
        "schema_version": config["schema_version"],
        "scientific_role": config["scientific_role"],
        "not_blind_rationale": config["not_blind_rationale"],
        "analyst_exposure": config["analyst_exposure"],
        "allowed_pre_freeze_inputs": config["allowed_pre_freeze_inputs"],
        "forbidden_pre_freeze_inputs": config["forbidden_pre_freeze_inputs"],
        "exact_fig4_carryover": config["exact_fig4_carryover"],
        "model_ictal_v2": config["model_ictal_v2"],
        "repertoire_gate": config["repertoire_gate"],
        "motif_reuse": config["motif_reuse"],
        "montage_of_record": montage_of_record(config),
        "immutable_input_hashes": hashes,
        "candidate_inventory": candidate_inventory(config),
        "preserved_unchanged": config["supersedes_nothing"],
        "claim_boundary": config["claim_boundary"],
    }
    atomic_write_json(boundary, str(output_root / "discovery_boundary.json"))

    provenance = {
        "status": BOUNDARY_STATUS,
        "producer": "scripts/freeze_topic4_zm_discovery_boundary.py",
        "audit_config": {"path": args.config,
                         "sha256": sha256_file(ROOT / args.config)},
        "governing_documents": config["governing_documents"],
        **git_provenance(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "simulation_launched": False,
        "clinical_ictal_target_read": False,
    }
    atomic_write_json(provenance, str(output_root / "provenance.json"))
    print(json.dumps({
        "status": BOUNDARY_STATUS,
        "immutable_inputs_verified": len(hashes),
        "n_trajectory_json": boundary["candidate_inventory"][
            "exact_carryover_and_pathway_arms"]["n_files"],
        "n_calibration_json": sum(
            row.get("n_files", 0) for row in
            boundary["candidate_inventory"]["calibrated_transition_candidates"]),
    }, indent=1))


if __name__ == "__main__":
    main()
