"""Cohort query/support construction for H2b cross-task transfer v0.2.

This module contains no learned seizure model.  It converts frozen seizure
onsets, admissible recorded coverage and inference-readable background anchors
into exact pre-seizure queries plus a common control grid.  The state extractor
then reads those times without receiving a seizure label.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
# Load cuda_env's compatible C++ runtime before pandas native extensions.
import torch as _torch  # noqa: F401
import pandas as pd

from src.topic5_continuous_marked_state_r1.coverage import CoverageTable

from .contract import (
    LEAD_MINUTES,
    POSTICTAL_GUARD_MINUTES,
    PRIMARY_LEAD_MINUTES,
    V0_2_RESULT_ROOT,
    atomic_csv,
    atomic_json,
    sha256_file,
    support_tier,
    utc_now,
)
from .state_extraction import (
    InferenceRawAnchorReader,
    load_frozen_design_artifact,
)


COHORT_QUERY_REVISION = "h2b_cross_task_cohort_query_support_v0_2"
CURRENT_OBSERVATION_MAX_AGE_SECONDS = 30.0


@dataclass(frozen=True)
class SubjectQueryBundle:
    query_rows: list[dict[str, Any]]
    seizure_rows: list[dict[str, Any]]
    support_rows: list[dict[str, Any]]
    exclusion_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def _window_segment(
        coverage: CoverageTable, cutoff: float, onset: float,
        ) -> int | None:
    hit = np.flatnonzero(
        (coverage.start <= float(cutoff))
        # Coverage segments used here are closed at a seizure boundary: a
        # segment whose stop equals onset means recording is complete through
        # the pre-onset horizon.  Query anchors themselves remain < stop.
        & (float(onset) <= coverage.stop)
    )
    return int(hit[0]) if len(hit) == 1 else None


def _latest_causal_observation_age(
        grid_time: np.ndarray, grid_segment: np.ndarray,
        *, cutoff: float, segment: int,
        ) -> float:
    eligible = (
        (np.asarray(grid_segment, dtype=np.int64) == int(segment))
        & (np.asarray(grid_time, dtype=np.float64) <= float(cutoff))
    )
    if not bool(np.any(eligible)):
        return float("inf")
    return float(cutoff) - float(np.max(np.asarray(grid_time)[eligible]))


def build_subject_query_bundle(
        *, subject: str,
        seizure_rows: Iterable[Mapping[str, Any]],
        coverage: CoverageTable,
        grid_time_epoch: np.ndarray,
        grid_segment: np.ndarray,
        grid_continuity_session: np.ndarray,
        ) -> SubjectQueryBundle:
    """Build exact case queries and the event-independent control grid."""
    if coverage.subject != str(subject):
        raise ValueError("coverage/subject mismatch")
    grid_time = np.asarray(grid_time_epoch, dtype=np.float64)
    grid_segment = np.asarray(grid_segment, dtype=np.int64)
    grid_session = np.asarray(grid_continuity_session, dtype=np.int64)
    if not (len(grid_time) == len(grid_segment) == len(grid_session)):
        raise ValueError("inference grid arrays do not align")
    if len(grid_time) and np.any(np.diff(grid_time) < 0):
        raise ValueError("inference grid is not chronological")

    seizures = sorted(
        [dict(row) for row in seizure_rows],
        key=lambda row: (float(row["onset_epoch"]), str(row["seizure_id"])),
    )
    development_seizures = [
        row for row in seizures
        if float(row["onset_epoch"]) < float(coverage.dev_end_epoch)
    ]
    exclusion_intervals = []
    for seizure in development_seizures:
        onset = float(seizure["onset_epoch"])
        offset = float(seizure.get("offset_epoch", onset))
        if not np.isfinite(offset):
            offset = onset
        exclusion_intervals.append((
            onset, offset + POSTICTAL_GUARD_MINUTES * 60.0,
        ))
    support: list[dict[str, Any]] = []
    eligible_by_seizure: dict[str, list[dict[str, Any]]] = {}
    for seizure in seizures:
        seizure_id = str(seizure["seizure_id"])
        onset = float(seizure["onset_epoch"])
        in_development = onset < float(coverage.dev_end_epoch)
        for lead in LEAD_MINUTES:
            cutoff = onset - 60.0 * int(lead)
            segment = (
                _window_segment(coverage, cutoff, onset)
                if in_development else None
            )
            age = (
                _latest_causal_observation_age(
                    grid_time, grid_segment, cutoff=cutoff, segment=segment,
                )
                if segment is not None else float("inf")
            )
            fresh = bool(
                np.isfinite(age)
                and age >= -1e-9
                and age <= CURRENT_OBSERVATION_MAX_AGE_SECONDS + 1e-9
            )
            cutoff_exclusion_clear = all(
                not (left <= cutoff <= right)
                for left, right in exclusion_intervals
            )
            eligible = bool(
                in_development and segment is not None and fresh
                and cutoff_exclusion_clear
            )
            if not in_development:
                reason = "outside_development_partition"
            elif segment is None:
                reason = "lead_window_crosses_gap_or_excluded_interval"
            elif not fresh:
                reason = "no_causal_observation_within_30_seconds"
            elif not cutoff_exclusion_clear:
                reason = "cutoff_in_ictal_or_postictal_interval"
            else:
                reason = "eligible"
            row = {
                "subject": str(subject),
                "seizure_id": seizure_id,
                "lead_minutes": int(lead),
                "primary_lead": int(lead) == PRIMARY_LEAD_MINUTES,
                "onset_epoch": np.float64(onset),
                "cutoff_epoch": np.float64(cutoff),
                "in_development_partition": bool(in_development),
                "complete_recorded_lead_window": segment is not None,
                "coverage_segment_index": int(segment) if segment is not None else None,
                "current_observation_age_seconds": (
                    np.float64(age) if np.isfinite(age) else None
                ),
                "eligible": eligible,
                "cutoff_ictal_postictal_exclusion_clear": bool(
                    cutoff_exclusion_clear
                ),
                "exclusion_reason": reason,
            }
            support.append(row)
            if eligible:
                eligible_by_seizure.setdefault(seizure_id, []).append(row)

    case_segments = sorted({
        int(row["coverage_segment_index"])
        for rows in eligible_by_seizure.values() for row in rows
    })
    query_by_key: dict[tuple[int, float], dict[str, Any]] = {}
    selected_grid = np.isin(grid_segment, np.asarray(case_segments, dtype=np.int64))
    for time, segment, session in zip(
            grid_time[selected_grid], grid_segment[selected_grid],
            grid_session[selected_grid]):
        query_by_key[(int(segment), float(time))] = {
            "anchor_time_epoch": np.float64(time),
            "coverage_segment_index": int(segment),
            "continuity_session": int(session),
            "query_role": "control_candidate",
            "case_seizure_id": "",
            "case_lead_minutes": "",
            "exclusion_start_epoch": None,
            "exclusion_stop_epoch": None,
        }

    seizure_lookup = {str(row["seizure_id"]): row for row in seizures}
    for seizure_id, rows in eligible_by_seizure.items():
        seizure = seizure_lookup[seizure_id]
        onset = float(seizure["onset_epoch"])
        offset = float(seizure.get("offset_epoch", onset))
        if not np.isfinite(offset):
            offset = onset
        for support_row in rows:
            segment = int(support_row["coverage_segment_index"])
            cutoff = float(support_row["cutoff_epoch"])
            lead = int(support_row["lead_minutes"])
            key = (segment, cutoff)
            existing = query_by_key.get(key)
            if existing and existing.get("case_seizure_id") not in ("", seizure_id):
                raise ValueError("two seizures share one exact pre-seizure query")
            if existing and existing.get("case_lead_minutes") not in ("", lead):
                raise ValueError("two lead times share one exact pre-seizure query")
            query_by_key[key] = {
                **(existing or {}),
                "anchor_time_epoch": np.float64(cutoff),
                "coverage_segment_index": segment,
                "continuity_session": int(coverage.session[segment]),
                "query_role": "case_and_control_grid" if existing else "case",
                "case_seizure_id": seizure_id,
                "case_lead_minutes": lead,
                "exclusion_start_epoch": np.float64(cutoff),
                "exclusion_stop_epoch": np.float64(
                    offset + POSTICTAL_GUARD_MINUTES * 60.0
                ),
            }

    query_rows = sorted(
        query_by_key.values(),
        key=lambda row: (
            float(row["anchor_time_epoch"]),
            int(row["coverage_segment_index"]),
        ),
    )
    stem = str(subject).replace("epilepsiae_", "e").replace("yuquan_", "y_")
    for index, row in enumerate(query_rows):
        row["query_id"] = f"{stem}_q{index:08d}"

    # Keep every development seizure in this table, including seizures without
    # a usable case anchor.  ``build_risk_sets`` uses the full onset list to
    # reject controls whose future horizon contains any seizure; omitting an
    # unsupported seizure would silently turn a positive horizon into a
    # negative control.  Post-development onsets are not imported.
    frozen_seizures: list[dict[str, Any]] = []
    for seizure in development_seizures:
        seizure_id = str(seizure["seizure_id"])
        rows = eligible_by_seizure.get(seizure_id, [])
        primary = [row for row in rows if row["primary_lead"]]
        representative = primary[0] if primary else (rows[0] if rows else None)
        frozen_seizures.append({
            "patient_id": str(subject),
            "seizure_id": seizure_id,
            "onset_time": np.float64(seizure["onset_epoch"]),
            "segment_id": (
                str(int(representative["coverage_segment_index"]))
                if representative is not None else "UNMAPPED_NO_CASE_SUPPORT"
            ),
            "primary_30min_supported": bool(primary),
        })
    frozen_seizures.sort(key=lambda row: (float(row["onset_time"]), row["seizure_id"]))

    exclusions = []
    for seizure in development_seizures:
        onset = float(seizure["onset_epoch"])
        offset = float(seizure.get("offset_epoch", onset))
        if not np.isfinite(offset):
            offset = onset
        exclusions.append({
            "seizure_id": str(seizure["seizure_id"]),
            "interval_start_epoch": np.float64(onset),
            "interval_stop_epoch": np.float64(
                offset + POSTICTAL_GUARD_MINUTES * 60.0
            ),
            "interval_role": "ictal_plus_postictal_exclusion",
        })

    by_lead = {
        str(lead): sum(
            bool(row["eligible"]) and int(row["lead_minutes"]) == int(lead)
            for row in support
        )
        for lead in LEAD_MINUTES
    }
    primary_n = int(by_lead[str(PRIMARY_LEAD_MINUTES)])
    summary = {
        "status": "COMPLETE",
        "revision": COHORT_QUERY_REVISION,
        "subject": str(subject),
        "n_inventory_seizures": len(seizures),
        "n_seizures_imported_into_development_outputs": len(
            development_seizures
        ),
        "post_development_seizure_identifiers_persisted": False,
        "n_development_seizures": sum(
            float(row["onset_epoch"]) < float(coverage.dev_end_epoch)
            for row in seizures
        ),
        "n_eligible_by_lead": by_lead,
        "n_primary_eligible_seizures": primary_n,
        "support_tier": support_tier(primary_n),
        "n_query_rows": len(query_rows),
        "n_control_grid_rows": int(selected_grid.sum()),
        "n_case_query_rows": sum(bool(row["eligible"]) for row in support),
        "case_coverage_segment_indices": case_segments,
        "state_source_uses_seizure_labels": False,
        "seizure_labels_used_only_for_query_and_exclusion": True,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
    }
    return SubjectQueryBundle(
        query_rows=query_rows,
        seizure_rows=frozen_seizures,
        support_rows=support,
        exclusion_rows=exclusions,
        summary=summary,
    )


def prepare_subject_query_inputs(
        *, subject: str,
        seizure_crosswalk_path: str | Path,
        coverage_path: str | Path,
        design_path: str | Path,
        design_sha256: str,
        design_manifest_path: str | Path | None = None,
        result_root: str | Path = V0_2_RESULT_ROOT,
        ) -> dict[str, Any]:
    """Read frozen upstream artifacts and atomically persist one subject bundle."""
    root = Path(result_root).resolve()
    crosswalk_path = Path(seizure_crosswalk_path).resolve()
    coverage_path = Path(coverage_path).resolve()
    design_path = Path(design_path).resolve()
    crosswalk = pd.read_csv(crosswalk_path)
    selected = crosswalk[crosswalk["subject"].astype(str) == str(subject)]
    if selected.empty:
        raise ValueError(f"{subject}: no seizure rows in frozen crosswalk")
    if selected["seizure_id"].astype(str).duplicated().any():
        raise ValueError(f"{subject}: seizure IDs are not unique")
    design, _, resolved_manifest = load_frozen_design_artifact(
        design_path, expected_sha256=design_sha256,
        expected_subject=subject, manifest_path=design_manifest_path,
    )
    coverage = CoverageTable.load(coverage_path)
    reader = InferenceRawAnchorReader(
        subject, design.event_time,
        source_repo_root="/home/honglab/leijiaxin/HFOsp",
    )
    grid = reader.inference_anchor_inventory(coverage)
    bundle = build_subject_query_bundle(
        subject=subject,
        seizure_rows=selected.to_dict(orient="records"),
        coverage=coverage,
        grid_time_epoch=grid[0], grid_segment=grid[1],
        grid_continuity_session=grid[2],
    )
    subject_root = root / "risk_sets" / subject
    query_path = subject_root / "state_queries.csv"
    seizure_path = subject_root / "seizures.csv"
    support_path = subject_root / "support_by_seizure_lead.csv"
    exclusion_path = subject_root / "global_exclusions.csv"
    atomic_csv(query_path, bundle.query_rows)
    atomic_csv(seizure_path, bundle.seizure_rows)
    atomic_csv(support_path, bundle.support_rows)
    atomic_csv(exclusion_path, bundle.exclusion_rows)
    summary = {
        **bundle.summary,
        "created_utc": utc_now(),
        "query_path": str(query_path),
        "query_sha256": sha256_file(query_path),
        "seizure_path": str(seizure_path),
        "seizure_sha256": sha256_file(seizure_path),
        "support_path": str(support_path),
        "support_sha256": sha256_file(support_path),
        "global_exclusion_path": str(exclusion_path),
        "global_exclusion_sha256": sha256_file(exclusion_path),
        "seizure_crosswalk_path": str(crosswalk_path),
        "seizure_crosswalk_sha256": sha256_file(crosswalk_path),
        "coverage_path": str(coverage_path),
        "coverage_sha256": sha256_file(coverage_path),
        "design_path": str(design_path),
        "design_sha256": sha256_file(design_path),
        "design_manifest_path": (
            str(resolved_manifest) if resolved_manifest is not None else None
        ),
        "design_manifest_sha256": (
            sha256_file(resolved_manifest) if resolved_manifest is not None else None
        ),
        "raw_cache_source_hashes": reader.source_hashes,
    }
    output = subject_root / "input_manifest.json"
    atomic_json(output, summary)
    return summary
