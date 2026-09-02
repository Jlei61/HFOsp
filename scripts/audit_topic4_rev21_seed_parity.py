#!/usr/bin/env python3
"""Require exact array parity before the rev21 seed-factorization audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


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
                "array_equal": bool(left.shape == right.shape
                                    and np.array_equal(left, right, equal_nan=True)),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison = compare_npz(args.reference, args.candidate)
    passed = comparison["key_sets_equal"] and comparison["all_common_arrays_equal"]
    payload = {
        "schema_id": "topic4_rev21_seed_parity_audit_v1",
        "status": "PASS" if passed else "FAIL",
        "reference": str(args.reference),
        "reference_sha256": _sha256(args.reference),
        "candidate": str(args.candidate),
        "candidate_sha256": _sha256(args.candidate),
        "comparison": comparison,
    }
    _atomic_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "output": str(args.output)},
                     indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
