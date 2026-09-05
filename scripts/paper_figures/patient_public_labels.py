"""Resolve artifact subject identifiers to manuscript-safe public labels.

Yuquan's private crosswalk is intentionally read at run time and is never
copied into tracked figure metadata.  The Epilepsiae order is the locked order
used by Supplementary Table S2 and the legacy cohort scripts.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
YUQUAN_CROSSWALK = ROOT / "docs/paper-draft/.private/yuquan_crosswalk.md"
EPILEPSIAE_TABLE_ORDER = (
    "1096",
    "1084",
    "958",
    "922",
    "590",
    "1150",
    "442",
    "1073",
    "253",
    "1146",
    "916",
    "620",
    "583",
    "548",
    "384",
    "139",
    "1125",
    "1077",
    "818",
    "635",
)


@lru_cache(maxsize=1)
def _yuquan_public_by_folder() -> dict[str, str]:
    if not YUQUAN_CROSSWALK.exists():
        raise FileNotFoundError(
            "Yuquan public labels require the private manuscript crosswalk: "
            f"{YUQUAN_CROSSWALK}"
        )
    mapping: dict[str, str] = {}
    row_pattern = re.compile(r"^\|\s*(Y\d+)\s*\|\s*([^|]+?)\s*\|")
    for line in YUQUAN_CROSSWALK.read_text(encoding="utf-8").splitlines():
        match = row_pattern.match(line)
        if match:
            mapping[match.group(2).strip()] = match.group(1)
    if len(mapping) != 20:
        raise RuntimeError(
            "Expected 20 Yuquan manuscript mappings, "
            f"found {len(mapping)} in {YUQUAN_CROSSWALK}"
        )
    return mapping


def public_patient_label(dataset: str, subject: str) -> str:
    """Return the locked manuscript label for one artifact subject."""
    dataset = str(dataset).lower()
    subject = str(subject)
    if dataset == "yuquan":
        mapping = _yuquan_public_by_folder()
        if subject not in mapping:
            raise KeyError(f"Yuquan subject is not in Table S1 crosswalk: {subject}")
        return mapping[subject]
    if dataset == "epilepsiae":
        try:
            return f"E{EPILEPSIAE_TABLE_ORDER.index(subject) + 1}"
        except ValueError as exc:
            raise KeyError(
                f"Epilepsiae subject is not in the locked Table S2 order: {subject}"
            ) from exc
    raise ValueError(f"Unsupported dataset: {dataset}")


def artifact_subject_from_public(dataset: str, patient_id: str) -> str:
    """Resolve a locked public label back to its local artifact identifier."""
    dataset = str(dataset).lower()
    patient_id = str(patient_id)
    if dataset == "yuquan":
        reverse = {
            public: folder for folder, public in _yuquan_public_by_folder().items()
        }
        if patient_id not in reverse:
            raise KeyError(f"Unknown Yuquan public patient label: {patient_id}")
        return reverse[patient_id]
    if dataset == "epilepsiae":
        if not re.fullmatch(r"E(?:[1-9]|1\d|20)", patient_id):
            raise KeyError(f"Unknown Epilepsiae public patient label: {patient_id}")
        return EPILEPSIAE_TABLE_ORDER[int(patient_id[1:]) - 1]
    raise ValueError(f"Unsupported dataset: {dataset}")
