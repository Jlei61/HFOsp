#!/usr/bin/env python3
"""Require exact array parity before the rev21 seed-factorization audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(
            payload, indent=2, sort_keys=True, allow_nan=False,
        ) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare all persisted dtypes while treating floating NaNs as equal."""
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape:
        return False
    if left.dtype.kind in "fc" and right.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def compare_npz(reference_path: Path, candidate_path: Path) -> dict:
    with np.load(reference_path, allow_pickle=False) as reference, np.load(
            candidate_path, allow_pickle=False) as candidate:
        reference_keys = set(reference.files)
        candidate_keys = set(candidate.files)
        common = sorted(reference_keys & candidate_keys)
        rows = {}
        for key in common:
            left, right = np.asarray(reference[key]), np.asarray(candidate[key])
            rows[key] = {
                "shape_equal": left.shape == right.shape,
                "dtype_equal": left.dtype == right.dtype,
                "array_equal": arrays_equal(left, right),
            }
        return {
            "reference_only_keys": sorted(reference_keys - candidate_keys),
            "candidate_only_keys": sorted(candidate_keys - reference_keys),
            "arrays": rows,
            "all_common_arrays_equal": bool(rows and all(
                row["array_equal"] for row in rows.values()
            )),
            "key_sets_equal": reference_keys == candidate_keys,
        }


def compare_runtime_modules(candidate_json: Path, current_commit: str) -> dict:
    payload = json.loads(candidate_json.read_text())
    provenance = payload.get("provenance", {})
    frozen = provenance.get("runtime_module_sha256", {})
    if not frozen:
        raise RuntimeError("candidate JSON has no runtime-module provenance")
    rows = {}
    for relative, expected in frozen.items():
        try:
            blob = subprocess.check_output(
                ["git", "show", f"{current_commit}:{relative}"], cwd=ROOT,
            )
            observed = hashlib.sha256(blob).hexdigest()
            rows[relative] = {
                "expected_sha256": expected,
                "current_commit_sha256": observed,
                "match": observed == expected,
            }
        except subprocess.CalledProcessError:
            rows[relative] = {
                "expected_sha256": expected,
                "current_commit_sha256": None,
                "match": False,
            }
    config_path = provenance.get("config_path")
    config_expected = provenance.get("config_sha256")
    if config_path and config_expected:
        blob = subprocess.check_output(
            ["git", "show", f"{current_commit}:{config_path}"], cwd=ROOT,
        )
        config_observed = hashlib.sha256(blob).hexdigest()
    else:
        config_observed = None
    return {
        "candidate_expected_commit": provenance.get("expected_git_commit"),
        "current_commit": current_commit,
        "module_count": len(rows),
        "modules": rows,
        "all_runtime_modules_match_current_commit": bool(
            rows and all(row["match"] for row in rows.values())
        ),
        "config_path": config_path,
        "config_expected_sha256": config_expected,
        "config_current_commit_sha256": config_observed,
        "config_matches_current_commit": bool(
            config_expected and config_observed == config_expected
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-json", type=Path)
    parser.add_argument("--current-commit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison = compare_npz(args.reference, args.candidate)
    runtime = None
    if bool(args.candidate_json) != bool(args.current_commit):
        parser.error("--candidate-json and --current-commit must be supplied together")
    if args.candidate_json:
        runtime = compare_runtime_modules(args.candidate_json, args.current_commit)
    passed = bool(
        comparison["key_sets_equal"]
        and comparison["all_common_arrays_equal"]
        and (runtime is None or (
            runtime["all_runtime_modules_match_current_commit"]
            and runtime["config_matches_current_commit"]
        ))
    )
    payload = {
        "schema_id": "topic4_rev21_seed_parity_audit_v1",
        "status": "PASS" if passed else "FAIL",
        "reference": str(args.reference),
        "reference_sha256": _sha256(args.reference),
        "candidate": str(args.candidate),
        "candidate_sha256": _sha256(args.candidate),
        "comparison": comparison,
        "current_runtime_audit": runtime,
    }
    _atomic_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "output": str(args.output)},
                     indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
