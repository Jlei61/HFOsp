"""Layer A v2.3 per-subject JSON loading + onset-matrix construction.

Extracted from ``scripts/plot_ictal_er_atlas.py`` so downstream
modules (e.g. ``src/ictal_seizure_clustering.py`` for topic5 PR-1)
can consume v2.3 outputs without an ``src → scripts`` dependency.

Public API:

- ``REQUIRED_SCHEMA`` — schema_version string the v2.3 cohort writes.
- ``LAYER_A_DIR`` / ``PER_SUBJECT_DIR`` / ``SENTINEL_DIR`` — canonical paths.
- ``load_per_subject_json(subject, *, source, schema_required)`` — load + schema gate.
- ``list_cohort_subjects(*, source)`` — all subjects with v2.3 JSON in dir.
- ``build_onset_matrix(per_er_record, channels)`` — (onset[n_ch, n_sz], statuses, seizure_ids).
- ``seizure_idx_in_order(per_er_record)`` — seizure_idx list in JSON order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


_THIS = Path(__file__).resolve()
ROOT = _THIS.parents[1]

# v2.3 schema constant — must match scripts/run_ictal_er_rank.py SCHEMA_VERSION
REQUIRED_SCHEMA = "pr_t3_1_layer_a_v2_3_timing"

LAYER_A_DIR = ROOT / "results" / "data_driven_soz" / "layer_a_ictal_er_rank"
PER_SUBJECT_DIR = LAYER_A_DIR / "per_subject"
SENTINEL_DIR = LAYER_A_DIR / "_sentinel"


def load_per_subject_json(
    subject: str,
    *,
    source: str = "per_subject",
    schema_required: bool = True,
) -> Dict:
    """Load v2.3 per-subject JSON. ``source`` ∈ {per_subject, _sentinel}.

    Hard-fails with ValueError if schema_version doesn't match the v2.3
    contract (REQUIRED_SCHEMA), so v2.2 JSONs cannot leak through.
    """
    sid = subject.replace("/", "_")
    if source == "per_subject":
        path = PER_SUBJECT_DIR / f"{sid}.json"
    elif source == "_sentinel":
        path = SENTINEL_DIR / f"{sid}.json"
    else:
        raise ValueError(f"unknown source={source}")
    if not path.exists():
        raise FileNotFoundError(f"per-subject JSON missing: {path}")

    with open(path, "r") as fh:
        d = json.load(fh)
    if schema_required:
        sv = d.get("schema_version")
        if sv != REQUIRED_SCHEMA:
            raise ValueError(
                f"{path} schema_version={sv!r}, expected {REQUIRED_SCHEMA!r}. "
                f"v2.2 JSON is not consumable — backup is in per_subject_v2_2/."
            )
    return d


def list_cohort_subjects(*, source: str = "per_subject") -> List[str]:
    """All subjects with a v2.3 per-subject JSON in the given dir."""
    src_dir = PER_SUBJECT_DIR if source == "per_subject" else SENTINEL_DIR
    if not src_dir.exists():
        return []
    out: List[str] = []
    for p in sorted(src_dir.glob("*.json")):
        if p.name in {"cohort_summary.json", "sanity_report.json"}:
            continue
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if d.get("schema_version") == REQUIRED_SCHEMA:
            out.append(d["subject"])
    return out


def seizure_idx_in_order(per_er_record: Dict) -> List[int]:
    """Return seizure_idx values in the order seizure_records were written."""
    return [int(r["seizure_idx"]) for r in per_er_record.get("seizure_records", [])]


def build_onset_matrix(
    per_er_record: Dict,
    channels: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (onset_matrix [n_ch, n_sz], status_array [n_sz], seizure_ids).

    onset_matrix cells: t_onset_sec (float) or NaN (CUSUM not triggered /
    seizure status != ok / channel missing in channel_onsets).

    Row j corresponds to the j-th entry in per_er_record["seizure_records"]
    (which is in seizure_idx order in v2.3 JSON).
    """
    sz_records = per_er_record.get("seizure_records", [])
    n_sz = len(sz_records)
    onset = np.full((len(channels), n_sz), np.nan, dtype=np.float64)
    statuses: List[str] = []
    seizure_ids: List[str] = []
    for j, rec in enumerate(sz_records):
        statuses.append(rec.get("status", "unknown"))
        seizure_ids.append(rec.get("seizure_id", str(rec.get("seizure_idx", j))))
        co = rec.get("channel_onsets") or {}
        for i, ch in enumerate(channels):
            entry = co.get(ch)
            if not entry:
                continue
            t = entry.get("t_onset_sec")
            if t is not None and np.isfinite(t):
                onset[i, j] = float(t)
    return onset, np.array(statuses), seizure_ids
