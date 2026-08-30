#!/usr/bin/env python3
"""Materialise one subject's primary and matched-wrong-time H2b risk tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch as _torch  # noqa: F401; load compatible native runtime first

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_continuous_marked_state_h2b.contract import (  # noqa: E402
    V0_2_RESULT_ROOT,
    atomic_json,
    sha256_file,
    utc_now,
)
from src.topic5_continuous_marked_state_h2b.pilot import (  # noqa: E402
    build_cohort_risk_table,
    state_cache_to_anchor_frame,
)
from src.topic5_continuous_marked_state_r1.coverage import (  # noqa: E402
    CoverageTable,
)


PRIMARY_ARMS = ("B_history", "B_observation", "B_state", "memoryless")
WRONG_TIME_ARMS = (*PRIMARY_ARMS, "wrong_time")


def run(subject: str, result_root: Path, controls_per_case: int) -> dict:
    root = result_root.resolve()
    subject_risk = root / "risk_sets" / subject
    input_manifest_path = subject_risk / "input_manifest.json"
    input_manifest = json.loads(input_manifest_path.read_text())
    if input_manifest.get("status") != "COMPLETE":
        raise ValueError(f"{subject}: input manifest is not COMPLETE")
    if input_manifest.get("formal_test_partition_opened") is not False:
        raise ValueError(f"{subject}: formal test partition was opened")
    coverage_path = Path(input_manifest["coverage_path"])
    if sha256_file(coverage_path) != input_manifest["coverage_sha256"]:
        raise ValueError(f"{subject}: coverage hash drifted")
    coverage = CoverageTable.load(coverage_path)
    query_path = Path(input_manifest["query_path"])
    exclusion_path = Path(input_manifest["global_exclusion_path"])
    seizure_path = Path(input_manifest["seizure_path"])
    for label, path in {
        "query": query_path,
        "exclusion": exclusion_path,
        "seizure": seizure_path,
    }.items():
        expected = input_manifest[f"{label if label != 'exclusion' else 'global_exclusion'}_sha256"]
        if sha256_file(path) != expected:
            raise ValueError(f"{subject}: {label} input hash drifted")

    cache_paths = sorted(
        (root / "state_cache" / subject).glob("seed_*/states.npz"),
        key=lambda path: int(path.parent.name.split("_")[-1]),
    )
    if not cache_paths:
        raise ValueError(f"{subject}: no completed frozen state cache")
    frames = []
    cache_hashes = {}
    for cache_path in cache_paths:
        seed = int(cache_path.parent.name.split("_")[-1])
        frames.append(state_cache_to_anchor_frame(
            cache_path=cache_path,
            query_path=query_path,
            coverage=coverage,
            global_exclusion_path=exclusion_path,
            seed=seed,
            patient_id=subject,
        ))
        cache_hashes[str(seed)] = sha256_file(cache_path)

    primary_path = subject_risk / "primary_risk_sets.csv"
    _, primary_audit = build_cohort_risk_table(
        anchor_frames=frames,
        seizure_path=seizure_path,
        output_path=primary_path,
        controls_per_case=controls_per_case,
        arms=PRIMARY_ARMS,
        require_wrong_time=False,
    )
    wrong_path = subject_risk / "matched_wrong_time_risk_sets.csv"
    try:
        _, wrong_audit = build_cohort_risk_table(
            anchor_frames=frames,
            seizure_path=seizure_path,
            output_path=wrong_path,
            controls_per_case=controls_per_case,
            arms=WRONG_TIME_ARMS,
            require_wrong_time=True,
        )
        wrong_status = "COMPLETE"
        wrong_reason = None
    except ValueError as exc:
        wrong_audit = None
        wrong_status = "NOT_ESTIMABLE"
        wrong_reason = str(exc)

    summary = {
        "status": "COMPLETE",
        "revision": "h2b_cross_task_subject_risk_tables_v0_2",
        "created_utc": utc_now(),
        "subject": subject,
        "n_state_cache_seeds": len(cache_paths),
        "state_cache_sha256_by_seed": cache_hashes,
        "controls_per_case": int(controls_per_case),
        "primary": {
            "status": "COMPLETE",
            "path": str(primary_path),
            "sha256": sha256_file(primary_path),
            "n_risk_sets": primary_audit["n_risk_sets"],
            "risk_set_hash": primary_audit["risk_set_hash"],
        },
        "matched_wrong_time": {
            "status": wrong_status,
            "reason": wrong_reason,
            "path": str(wrong_path) if wrong_status == "COMPLETE" else None,
            "sha256": sha256_file(wrong_path) if wrong_status == "COMPLETE" else None,
            "n_risk_sets": (
                wrong_audit["n_risk_sets"] if wrong_audit is not None else 0
            ),
            "risk_set_hash": (
                wrong_audit["risk_set_hash"] if wrong_audit is not None else None
            ),
        },
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "h3_or_t2_run": False,
    }
    atomic_json(subject_risk / "risk_table_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--result-root", type=Path, default=V0_2_RESULT_ROOT)
    parser.add_argument("--controls-per-case", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(
        run(args.subject, args.result_root, args.controls_per_case),
        indent=2, sort_keys=True,
    ))


if __name__ == "__main__":
    main()
