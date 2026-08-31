#!/usr/bin/env python3
"""Verify rebuilt interictal inputs against frozen R1.7B artifacts.

This is deliberately a verifier, not a producer.  It is used when an older
producer completed its expensive stages but could not write a v0.3 receipt.
No seizure outcome is read here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_continuous_marked_state_h2b.contract import (  # noqa: E402
    CANONICAL_V0_3_RESULT_ROOT,
    atomic_json,
    sha256_file,
    utc_now,
)


R17B_ROOT = Path(
    "/home/honglab/leijiaxin/HFOsp/results/epi_prssm/"
    "continuous_marked_state/r1/r1_7b_cohort_extension"
)
PRODUCER = Path(__file__).resolve()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _history_baseline_equal(
    baseline_path: Path, checkpoint_paths: list[Path],
) -> tuple[bool, list[dict]]:
    """Require every timing/mark history tensor to equal every frozen seed."""
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    rows: list[dict] = []
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False,
        )["model"]
        tensor_checks = []
        for group, prefix in (
            ("timing", "timing_baseline."), ("mark", "mark_baseline."),
        ):
            for name, value in baseline[group]["history"].items():
                key = prefix + name
                tensor_checks.append({
                    "tensor": key,
                    "bitwise_equal": bool(
                        key in checkpoint and torch.equal(value, checkpoint[key])
                    ),
                })
        rows.append({
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "all_history_tensors_bitwise_equal": bool(
                tensor_checks and all(row["bitwise_equal"] for row in tensor_checks)
            ),
            "tensor_checks": tensor_checks,
        })
    return bool(
        rows and all(row["all_history_tensors_bitwise_equal"] for row in rows)
    ), rows


def verify_subject(subject: str, result_root: Path) -> dict:
    upstream = result_root / "upstream_r1_2"
    verification = result_root / "upstream_r1_3_verification"
    frozen_manifest_path = R17B_ROOT / "cache" / subject / "manifest.json"
    design_manifest_path = upstream / "cache" / subject / "manifest.json"
    verification_manifest_path = (
        verification / "cache" / subject / "manifest.json"
    )
    design_path = upstream / "cache" / subject / "full_design.npz"
    embedding_path = upstream / "cache" / subject / "explicit_embedding.npy"
    explicit_path = (
        verification / "cache" / subject / "explicit_normalised.npy"
    )
    baseline_path = upstream / "baselines" / subject / "seed_0/models.pt"
    required = (
        frozen_manifest_path, design_manifest_path, verification_manifest_path,
        design_path, embedding_path, explicit_path, baseline_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{subject}: missing rebuild artifacts: {missing}")

    frozen = _json(frozen_manifest_path)
    design_manifest = _json(design_manifest_path)
    verification_manifest = _json(verification_manifest_path)
    checkpoints = sorted((R17B_ROOT / "fits" / subject).glob("seed_*/model.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"{subject}: no frozen R1.7B checkpoints")
    baseline_equal, baseline_rows = _history_baseline_equal(
        baseline_path, checkpoints,
    )
    checks = {
        "design_cache_complete": design_manifest.get("status") == "COMPLETE",
        "verification_cache_complete": (
            verification_manifest.get("status") == "COMPLETE"
        ),
        "design_matches_own_manifest": (
            sha256_file(design_path) == design_manifest.get("design_sha256")
        ),
        "embedding_matches_own_manifest": (
            sha256_file(embedding_path)
            == design_manifest.get("explicit_embedding_sha256")
        ),
        "design_matches_frozen_r1_7b": (
            sha256_file(design_path) == frozen.get("design_sha256")
        ),
        "normalised_explicit_matches_frozen_r1_7b": (
            sha256_file(explicit_path) == frozen.get("explicit_sha256")
        ),
        "history_baseline_matches_all_frozen_r1_7b_seeds": baseline_equal,
        "sealed_not_opened": (
            design_manifest.get("sealed_opened") is False
            and verification_manifest.get("sealed_opened") is False
        ),
    }
    payload = {
        "status": "COMPLETE" if all(checks.values()) else "FAIL",
        "revision": "h2b_v0_3_upstream_rebuild_equivalence_v1",
        "created_utc": utc_now(),
        "subject": subject,
        "checks": checks,
        "baseline_checkpoint_comparisons": baseline_rows,
        "artifacts": {
            "design": str(design_path),
            "design_sha256": sha256_file(design_path),
            "embedding": str(embedding_path),
            "embedding_sha256": sha256_file(embedding_path),
            "baseline": str(baseline_path),
            "baseline_sha256": sha256_file(baseline_path),
            "verification_explicit": str(explicit_path),
            "verification_explicit_sha256": sha256_file(explicit_path),
            "frozen_r1_7b_cache_manifest": str(frozen_manifest_path),
            "frozen_r1_7b_cache_manifest_sha256": sha256_file(
                frozen_manifest_path
            ),
        },
        "verifier": str(PRODUCER),
        "verifier_sha256": sha256_file(PRODUCER),
        "seizure_outcome_read": False,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
    }
    atomic_json(
        result_root / "manifests/upstream_rebuild" / f"{subject}.json",
        payload,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument(
        "--result-root", type=Path, default=CANONICAL_V0_3_RESULT_ROOT,
    )
    args = parser.parse_args()
    root = args.result_root.resolve()
    rows = [verify_subject(subject, root) for subject in args.subjects]
    summary = {
        "status": (
            "COMPLETE" if all(row["status"] == "COMPLETE" for row in rows)
            else "FAIL"
        ),
        "revision": "h2b_v0_3_upstream_rebuild_verification_queue_v1",
        "created_utc": utc_now(),
        "subjects": list(args.subjects),
        "rows": rows,
        "seizure_outcome_read": False,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
    }
    atomic_json(root / "UPSTREAM_PREPARATION_STATUS.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
