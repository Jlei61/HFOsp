#!/usr/bin/env python3
"""Freeze and verify empirical interictal-field inputs before target unseal."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
DEFAULT_FIELD_ROOT = Path(
    "/home/honglab/leijiaxin/HFOsp/results/"
    "interictal_propagation_masked/template_gradient_fields/per_subject"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--field-root", type=Path, default=DEFAULT_FIELD_ROOT)
    args = parser.parse_args()
    out = args.out_root.resolve()
    if (out / "TARGET_UNSEAL_AUTHORIZATION.json").exists():
        raise RuntimeError("empirical-field provenance must be frozen before target authorization")
    census_path = out / "FULL_PARENT_FIT_CENSUS.csv"
    routing_path = out / "EARLY_ICTAL_ROUTING_METADATA.csv"
    census = pd.read_csv(census_path)
    routing = pd.read_csv(routing_path)
    field_root = args.field_root.resolve()
    required = {"subject", "fit_id", "field_sha256"}
    if not required.issubset(census.columns):
        raise RuntimeError(f"census lacks field provenance columns: {sorted(required - set(census))}")
    failures: list[str] = []
    rows = []
    for record in census[list(required)].drop_duplicates().itertuples(index=False):
        field_path = (field_root / f"{record.subject}.json").resolve()
        if not field_path.exists():
            failures.append(f"MISSING:{field_path}")
            actual = None
        else:
            actual = sha256_file(field_path)
            if actual != str(record.field_sha256):
                failures.append(f"HASH_MISMATCH:{field_path}")
        rows.append({
            "subject": str(record.subject), "fit_id": str(record.fit_id),
            "path": str(field_path), "expected_sha256": str(record.field_sha256),
            "actual_sha256": actual,
        })
    early_subjects = set(routing.subject.astype(str))
    census_subjects = set(census.subject.astype(str))
    missing_early = sorted(early_subjects - census_subjects)
    failures.extend(f"EARLY_SUBJECT_WITHOUT_FIELD:{subject}" for subject in missing_early)
    payload = {
        "contract": "topic5_v0_5_empirical_field_input_prefreeze",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failures else "FAIL",
        "target_values_read": False,
        "census_sha256": sha256_file(census_path),
        "routing_sha256": sha256_file(routing_path),
        "fit_rows": len(census),
        "unique_field_files": len({row["path"] for row in rows}),
        "spatial_patients": int(census.subject.nunique()),
        "early_patients_covered": len(early_subjects & census_subjects),
        "early_patients_expected": len(early_subjects),
        "failures": failures,
        "fields": rows,
    }
    destination = out / "EMPIRICAL_FIELD_INPUT_PREFREEZE_MANIFEST.json"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(destination)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
