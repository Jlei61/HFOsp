#!/usr/bin/env python3
"""Audit every canonical Topic 5 interictal field after JSON serialization."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_template_axis_field import (  # noqa: E402
    INTERICTAL_FIELD_FINGERPRINT_ALGORITHM,
    interictal_field_fingerprint,
    interictal_field_quality_tier,
    scorers_from_interictal_record,
)

DEFAULT_ROOT = ROOT / "results/interictal_propagation_masked/template_gradient_fields"


def _hashed_scalar_null_paths(record: Mapping[str, object]) -> list[str]:
    """List JSON-null scalar paths that were ambiguous under legacy sha256_v1."""
    paths: list[str] = []
    pair = record.get("axis_pair") or {}
    for axis_name in ("axis_a", "axis_b"):
        axis = pair.get(axis_name) or {}
        for key in ("n", "n_shafts", "effective_rank", "R2", "bootstrap_cosine", "loso_cosine"):
            if axis.get(key) is None:
                paths.append(f"axis_pair.{axis_name}.{key}")
    relation = pair.get("relation") or {}
    for key in ("cosine", "abs_cosine", "line_angle_deg", "collinear", "relation"):
        if relation.get(key) is None:
            paths.append(f"axis_pair.relation.{key}")
    bootstrap = pair.get("pair_bootstrap") or {}
    for key in ("p_collinear", "p_sign_stable", "robust_collinear"):
        if bootstrap.get(key) is None:
            paths.append(f"axis_pair.pair_bootstrap.{key}")
    return paths


def audit_record(path: Path) -> dict[str, object]:
    record = json.loads(path.read_text())
    field = record.get("interictal_field") or {}
    ready = field.get("status") == "ok"
    scalar_null_paths = _hashed_scalar_null_paths(record) if ready else []
    recomputed = None
    fingerprint_valid = None
    fingerprint_error = None
    scorer_valid = None
    scorer_error = None
    if ready:
        try:
            recomputed = interictal_field_fingerprint(record)
            fingerprint_valid = recomputed == field.get("fingerprint_sha256")
        except Exception as exc:  # audit must retain the failure row
            fingerprint_valid = False
            fingerprint_error = f"{type(exc).__name__}:{exc}"
        try:
            scorers_from_interictal_record(record)
            scorer_valid = True
        except Exception as exc:  # audit must retain the failure row
            scorer_valid = False
            scorer_error = f"{type(exc).__name__}:{exc}"
    pair = record.get("axis_pair") or {}
    return {
        "subject_id": record.get("subject_id", path.stem),
        "dataset": record.get("dataset"),
        "subject": record.get("subject"),
        "record_status": record.get("status"),
        "field_status": field.get("status"),
        "field_ready": ready,
        "axis_quality_tier": interictal_field_quality_tier(record),
        "geometry_2d_supported": pair.get("geometry_2d_supported"),
        "strict_stability_pass": pair.get("strict_stability_pass"),
        "fingerprint_algorithm": field.get("fingerprint_algorithm"),
        "expected_algorithm": INTERICTAL_FIELD_FINGERPRINT_ALGORITHM,
        "stored_fingerprint_sha256": field.get("fingerprint_sha256"),
        "recomputed_fingerprint_sha256": recomputed,
        "fingerprint_valid_after_json": fingerprint_valid,
        "fingerprint_error": fingerprint_error,
        "scorer_load_valid": scorer_valid,
        "scorer_load_error": scorer_error,
        "legacy_nan_null_serialization_risk": bool(scalar_null_paths),
        "hashed_scalar_null_paths": ";".join(scalar_null_paths),
        "artifact": str(path.relative_to(ROOT)),
    }


def run(root: Path) -> dict[str, object]:
    paths = sorted((root / "per_subject").glob("*.json"))
    if not paths:
        raise RuntimeError(f"no per-subject field artifacts under {root}")
    rows = [audit_record(path) for path in paths]
    columns = list(rows[0])
    with (root / "fingerprint_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    ready = [row for row in rows if row["field_ready"]]
    invalid = [row for row in ready if not row["fingerprint_valid_after_json"]]
    unloadable = [row for row in ready if not row["scorer_load_valid"]]
    risk = [row for row in ready if row["legacy_nan_null_serialization_risk"]]
    summary = {
        "audit_contract": "topic5_interictal_field_fingerprint_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "field_root": str(root.relative_to(ROOT)),
        "expected_algorithm": INTERICTAL_FIELD_FINGERPRINT_ALGORITHM,
        "counts": {
            "subject_artifacts": len(rows),
            "field_ready": len(ready),
            "fingerprint_valid_after_json": sum(bool(row["fingerprint_valid_after_json"]) for row in ready),
            "scorer_load_valid": sum(bool(row["scorer_load_valid"]) for row in ready),
            "legacy_nan_null_serialization_risk": len(risk),
        },
        "record_status_counts": dict(sorted(Counter(str(row["record_status"]) for row in rows).items())),
        "field_status_counts": dict(sorted(Counter(str(row["field_status"]) for row in rows).items())),
        "axis_quality_tier_counts": dict(sorted(Counter(str(row["axis_quality_tier"]) for row in rows).items())),
        "legacy_risk_subjects": [str(row["subject_id"]) for row in risk],
        "invalid_fingerprint_subjects": [str(row["subject_id"]) for row in invalid],
        "unloadable_scorer_subjects": [str(row["subject_id"]) for row in unloadable],
        "audit_ok": not invalid and not unloadable,
    }
    (root / "fingerprint_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["audit_ok"]:
        raise RuntimeError("field fingerprint audit failed closed")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    run(args.field_root.resolve())


if __name__ == "__main__":
    main()
