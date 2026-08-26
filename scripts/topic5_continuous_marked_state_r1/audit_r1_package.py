#!/usr/bin/env python3
"""Fail-closed artifact/provenance audit for the bounded R1 development package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.bridge_e1 import BRIDGE_E1_REVISION
from src.topic5_continuous_marked_state_r1.raw_observation import RAW_OBSERVATION_REVISION
from src.topic5_continuous_marked_state_r1.synthetic_recovery import SYNTHETIC_REVISION
from src.topic5_continuous_marked_state_r1.t1_pilot import T1_PILOT_REVISION


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _check_result(path: Path, *, revision_key: str | None = None,
                  revision: str | None = None) -> dict:
    value = _load(path)
    if value.get("status") != "COMPLETE":
        raise ValueError(f"incomplete artifact: {path}")
    if value.get("contract") != contract.REVISION:
        raise ValueError(f"contract mismatch: {path}")
    if value.get("sealed_opened") is not False:
        raise ValueError(f"sealed status is not false: {path}")
    if revision_key and value.get(revision_key) != revision:
        raise ValueError(f"revision mismatch: {path}")
    checkpoint = value.get("checkpoint")
    expected_hash = value.get("checkpoint_sha256")
    if checkpoint and expected_hash:
        if contract.sha256_file(checkpoint) != expected_hash:
            raise ValueError(f"checkpoint hash mismatch: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args()
    root = contract.RESULT_ROOT
    checks: dict[str, object] = {}

    coverage = _load(root / "coverage/COVERAGE_MANIFEST.json")
    rows = coverage.get("subjects", coverage.get("rows", []))
    if isinstance(rows, dict):
        rows = list(rows.values())
    coverage_subjects = {
        row.get("subject") for row in rows if isinstance(row, dict)
    }
    missing_coverage = sorted(set(contract.PILOT_SUBJECTS) - coverage_subjects)
    checks["coverage"] = {
        "n_subjects": len(coverage_subjects),
        "missing_pilot_subjects": missing_coverage,
    }
    if missing_coverage:
        raise ValueError(f"missing pilot coverage: {missing_coverage}")

    baseline_rows = []
    for subject in contract.PILOT_SUBJECTS:
        path = root / "baselines" / subject / "seed_0/result.json"
        if path.exists():
            baseline_rows.append(_check_result(path))
        elif not args.allow_pending:
            raise FileNotFoundError(path)
    checks["baselines"] = {
        "complete_subjects": [row["subject"] for row in baseline_rows],
        "expected_subjects": list(contract.PILOT_SUBJECTS),
    }

    bridge_rows = []
    for subject in contract.BRIDGE_E1_SUBJECTS:
        path = root / "bridge_e1" / subject / "seed_0/result.json"
        if path.exists():
            row = _check_result(
                path, revision_key="bridge_e1_revision",
                revision=BRIDGE_E1_REVISION,
            )
            if row.get("raw_observation_revision") != RAW_OBSERVATION_REVISION:
                raise ValueError(f"raw observation revision mismatch: {path}")
            if max(row["zero_raw_initial_parity_abs"].values()) > 1e-6:
                raise ValueError(f"zero-raw parity failed: {path}")
            bridge_rows.append(row)
        elif not args.allow_pending:
            raise FileNotFoundError(path)
    checks["bridge_e1"] = {
        "complete_subjects": [row["subject"] for row in bridge_rows],
        "expected_subjects": list(contract.BRIDGE_E1_SUBJECTS),
    }

    synthetic = _check_result(root / "synthetic/t1_recovery.json")
    if any(row.get("synthetic_revision") != SYNTHETIC_REVISION
           for row in synthetic["rows"]):
        raise ValueError("synthetic revision mismatch")
    checks["synthetic"] = {
        "n_recovered": synthetic["n_recovered"],
        "n_seeds": synthetic["n_seeds"],
    }

    t1_rows = []
    for path in sorted((root / "t1_pilot").glob("*/t1_*_seed_0/result.json")):
        t1_rows.append(_check_result(
            path, revision_key="t1_pilot_revision", revision=T1_PILOT_REVISION
        ))
    if not t1_rows and not args.allow_pending:
        raise FileNotFoundError("no T1 pilot results")
    checks["t1_pilot"] = {
        "complete_arms": [f"{row['subject']}:{row['arm']}" for row in t1_rows],
    }

    source_paths = sorted(
        list((contract.REPO_ROOT / "src/topic5_continuous_marked_state_r1").glob("*.py"))
        + list((contract.REPO_ROOT / "scripts/topic5_continuous_marked_state_r1").glob("*.py"))
        + list((contract.REPO_ROOT / "tests/topic5_continuous_marked_state_r1").glob("*.py"))
    )
    source_hashes = {
        str(path.relative_to(contract.REPO_ROOT)): contract.sha256_file(path)
        for path in source_paths
    }
    output = {
        "status": "PASS",
        "contract": contract.REVISION,
        "sealed_opened": False,
        "allow_pending": bool(args.allow_pending),
        "checks": checks,
        "source_hashes": source_hashes,
        "claim_boundary": (
            "engineering/provenance audit of a development package; PASS is not "
            "scientific acceptance of H1 or H2a"
        ),
    }
    target = root / "manifests/FINAL_PACKAGE_AUDIT.json"
    contract.atomic_json(target, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
