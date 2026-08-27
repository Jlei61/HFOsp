"""Phase-0 checkpoint, seizure-crosswalk, and support inventory for H2b.

This module is deliberately read-only with respect to the upstream state model.
It follows the R1.6 machine audit to an exact result file, follows that result to
an exact checkpoint, and recomputes every SHA256 before seizure metadata is read.
No model is instantiated and no seizure probe is trained here.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contract import (
    H2B_REVISION,
    LEAD_MINUTES,
    PRIMARY_LEAD_MINUTES,
    R1_6_MACHINE_AUDIT,
    R1_6_ROOT,
    RunBoundary,
    sha256_file,
    support_tier,
    utc_now,
    validate_lead_minutes,
)


INVENTORY_REVISION = "h2b_phase0_inventory_v0_1"
E384_SUBJECT = "epilepsiae_384"
E384_STABLE_SEEDS = (1, 3, 4)
SOURCE_TASK = "continuous_background_and_ied_timing_mark"
YUQUAN_STATE_ID = re.compile(r"^(?P<record>.+)_(?P<index>[0-9]+)$")
RAW_WINDOW_SECONDS = 30.0


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_result_checkpoint(source_repo_root: Path, result_path: Path,
                               checkpoint_value: str) -> Path:
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_absolute():
        checkpoint = source_repo_root / checkpoint
    checkpoint = checkpoint.resolve()
    expected = result_path.with_name("model.pt").resolve()
    if checkpoint != expected:
        raise ValueError(
            f"R1.6 result points outside its audited cell: {checkpoint} != {expected}"
        )
    return checkpoint


def load_r16_checkpoint_inventory(
    *,
    audit_path: Path = R1_6_MACHINE_AUDIT,
    result_root: Path = R1_6_ROOT,
    source_repo_root: Path = Path("/home/honglab/leijiaxin/HFOsp"),
    subject: str = E384_SUBJECT,
    seeds: Sequence[int] = E384_STABLE_SEEDS,
) -> dict[str, Any]:
    """Resolve stable E384 checkpoints through the audit, never by globbing."""
    audit_path = Path(audit_path).resolve()
    result_root = Path(result_root).resolve()
    source_repo_root = Path(source_repo_root).resolve()
    audit = _json(audit_path)
    if audit.get("status") != "COMPLETE":
        raise ValueError("R1.6 machine audit is not COMPLETE")
    if audit.get("formal_test_partition_opened") is not False:
        raise ValueError("R1.6 machine audit opened formal test data")
    if audit.get("sealed_opened") is not False:
        raise ValueError("R1.6 machine audit opened sealed data")
    by_subject = (audit.get("confirmation") or {}).get("by_subject") or {}
    subject_audit = by_subject.get(subject) or {}
    if subject_audit.get("stable_checkpoints") != len(tuple(seeds)):
        raise ValueError(
            f"{subject}: audit stable checkpoint count disagrees with frozen seeds"
        )
    confirmation = (audit.get("manifests") or {}).get("confirmation") or {}
    audited_files = confirmation.get("files") or []
    by_path = {str(row["path"]): row for row in audited_files}
    if len(by_path) != len(audited_files):
        raise ValueError("R1.6 confirmation audit contains duplicate result paths")

    entries: list[dict[str, Any]] = []
    for seed in tuple(int(value) for value in seeds):
        relative = (
            "confirmation/prefix_high_lr_e12_c128/nested_extended_budget/"
            f"{subject}/seed_{seed}/result.json"
        )
        audit_row = by_path.get(relative)
        if audit_row is None:
            raise ValueError(f"R1.6 machine audit lacks frozen result {relative}")
        result_path = (result_root / relative).resolve()
        result_sha256 = sha256_file(result_path)
        if result_sha256 != audit_row.get("sha256"):
            raise ValueError(f"R1.6 result SHA256 mismatch: {result_path}")
        result = _json(result_path)
        required = {
            "status": "COMPLETE",
            "subject": subject,
            "seed": seed,
            "stable_checkpoint": True,
            "formal_test_partition_opened": False,
            "sealed_opened": False,
            "development_validation_used_for_selection": False,
        }
        mismatches = {
            key: (result.get(key), expected)
            for key, expected in required.items() if result.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"inadmissible R1.6 result {result_path}: {mismatches}")
        checkpoint = _resolve_result_checkpoint(
            source_repo_root, result_path, str(result["checkpoint"])
        )
        checkpoint_sha256 = sha256_file(checkpoint)
        if checkpoint_sha256 != result.get("checkpoint_sha256"):
            raise ValueError(f"R1.6 checkpoint SHA256 mismatch: {checkpoint}")
        entries.append({
            "subject": subject,
            "seed": seed,
            "seed_role": result.get("seed_role"),
            "result_path": str(result_path),
            "result_sha256_expected": str(audit_row["sha256"]),
            "result_sha256_observed": result_sha256,
            "result_sha256_match": True,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256_expected": str(result["checkpoint_sha256"]),
            "checkpoint_sha256_observed": checkpoint_sha256,
            "checkpoint_sha256_match": True,
            "state_revision": result.get("revision"),
            "confirmation_revision": result.get("confirmation_revision"),
            "selected_prefix_config": result.get("selected_prefix_config"),
            "selected_alignment_config": result.get("selected_config"),
            "stable_checkpoint": True,
            "state_source_task": SOURCE_TASK,
            "state_source_uses_seizure_labels": False,
            "state_frozen_before_seizure_task": True,
            "formal_test_partition_opened": False,
            "sealed_opened": False,
        })
    return {
        "status": "COMPLETE",
        "contract": H2B_REVISION,
        "inventory_revision": INVENTORY_REVISION,
        "created_utc": utc_now(),
        "source_machine_audit": str(audit_path),
        "source_machine_audit_sha256": sha256_file(audit_path),
        "source_result_root": str(result_root),
        "subject": subject,
        "stable_seeds": list(map(int, seeds)),
        "n_checkpoints": len(entries),
        "entries": entries,
        "r1_7_used": False,
        "seizure_probe_trained": False,
    }


def load_epilepsiae_sql_seizures(sql_path: Path, *, subject: str) -> list[dict[str, Any]]:
    """Read seizure truth directly from one unique Epilepsiae SQL export."""
    from src.epilepsiae_dataset import _parse_sql_subject, _parse_ts

    sql_path = Path(sql_path).resolve()
    short = subject.split("_", 1)[1]
    parsed = _parse_sql_subject(sql_path)
    if parsed.get("subject") != short:
        raise ValueError(f"SQL subject {parsed.get('subject')} != requested {short}")
    rows: list[dict[str, Any]] = []
    for seizure in parsed.get("seizures") or []:
        clinical_onset = _parse_ts(seizure.get("clin_onset"))
        eeg_onset = _parse_ts(seizure.get("eeg_onset"))
        if clinical_onset is not None:
            onset, onset_kind = float(clinical_onset), "clinical"
            offset = _parse_ts(seizure.get("clin_offset"))
        elif eeg_onset is not None:
            onset, onset_kind = float(eeg_onset), "eeg"
            offset = _parse_ts(seizure.get("eeg_offset"))
        else:
            continue
        rows.append({
            "subject": subject,
            "dataset": "epilepsiae",
            "state_seizure_id": str(seizure["seizure_id"]),
            "canonical_seizure_id": str(seizure["seizure_id"]),
            "recording_code": str(seizure["recording_id"]),
            "block_id": str(seizure["block_id"]),
            "onset_epoch": onset,
            "canonical_onset_epoch": onset,
            "offset_epoch": None if offset is None else float(offset),
            "onset_kind": onset_kind,
            "classification": seizure.get("classification"),
            "pattern": seizure.get("pattern"),
            "vigilance": seizure.get("vigilance"),
            "match_route": "sql_seizure_id",
            "onset_difference_seconds": 0.0,
            "onset_exact_match": True,
            "ambiguous": False,
            "matched": True,
            "metadata_source": str(sql_path),
            "metadata_source_sha256": sha256_file(sql_path),
            "metadata_truth": "Epilepsiae SQL recording/block/seizure",
        })
    rows.sort(key=lambda row: (float(row["onset_epoch"]), row["canonical_seizure_id"]))
    return rows


def build_yuquan_crosswalk(
    state_rows: Iterable[Mapping[str, Any]],
    canonical_rows: Iterable[Mapping[str, Any]],
    *,
    subject: str,
) -> list[dict[str, Any]]:
    """Map ``<record>_<index>`` IDs through recording code with exact onset parity."""
    short = subject.split("_", 1)[1]
    canonical = [dict(row) for row in canonical_rows]
    canonical = [
        row for row in canonical
        if str(row.get("subject")) in {short, subject}
    ]
    canonical.sort(key=lambda row: (
        str(row.get("record", row.get("recording_code", ""))),
        float(row["eeg_onset_epoch"]),
        str(row.get("seizure_id", "")),
    ))
    keyed: dict[tuple[str, int], list[dict[str, Any]]] = {}
    counters: Counter[str] = Counter()
    for row in canonical:
        record = str(row.get("record", row.get("recording_code", "")))
        index = counters[record]
        counters[record] += 1
        keyed.setdefault((record, index), []).append(row)

    output: list[dict[str, Any]] = []
    for raw in state_rows:
        row = dict(raw)
        state_id = str(row.get("state_seizure_id", row.get("seizure_id", "")))
        parsed = YUQUAN_STATE_ID.match(state_id)
        record = parsed.group("record") if parsed else None
        index = int(parsed.group("index")) if parsed else None
        candidates = [] if parsed is None else keyed.get((str(record), int(index)), [])
        ambiguous = len(candidates) > 1
        canonical_row = candidates[0] if len(candidates) == 1 else None
        onset = float(row["onset_epoch"])
        canonical_onset = (
            float(canonical_row["eeg_onset_epoch"])
            if canonical_row is not None else float("nan")
        )
        delta = onset - canonical_onset
        exact = bool(np.isfinite(delta) and delta == 0.0)
        matched = bool(canonical_row is not None and not ambiguous and exact)
        output.append({
            **row,
            "subject": subject,
            "dataset": "yuquan",
            "state_seizure_id": state_id,
            "recording_code": record,
            "index_within_record": index,
            "canonical_seizure_id": (
                None if canonical_row is None else str(canonical_row.get("seizure_id"))
            ),
            "canonical_onset_epoch": (
                None if canonical_row is None else canonical_onset
            ),
            "onset_difference_seconds": None if not np.isfinite(delta) else float(delta),
            "onset_exact_match": exact,
            "ambiguous": ambiguous,
            "matched": matched,
            "match_route": (
                "id_did_not_parse" if parsed is None else
                "ambiguous_record_code+index" if ambiguous else
                "unmatched_record_code+index" if canonical_row is None else
                "record_code+index_onset_mismatch" if not exact else
                "record_code+index"
            ),
        })
    return output


def build_inference_minute_mask(
    *,
    covered: np.ndarray,
    session_id: np.ndarray,
    cached: np.ndarray,
    n_valid_contacts: np.ndarray,
    n_contacts: int,
    min_valid_contact_fraction: float,
) -> np.ndarray:
    """Unlabelled raw-cache eligibility, intentionally excluding training guards.

    ``guard_free`` and ``minute_usable`` are label-derived training gates and are
    therefore absent from this API.  They must not determine whether a frozen
    state model can consume an otherwise readable raw observation at inference.
    """
    arrays = [
        np.asarray(covered, dtype=bool),
        np.asarray(session_id, dtype=np.int64),
        np.asarray(cached, dtype=bool),
        np.asarray(n_valid_contacts, dtype=np.int64),
    ]
    if any(value.ndim != 1 for value in arrays):
        raise ValueError("raw inference index fields must be one-dimensional")
    if len({len(value) for value in arrays}) != 1:
        raise ValueError("raw inference index fields disagree in length")
    if int(n_contacts) < 1:
        raise ValueError("raw cache has no contacts")
    threshold = float(min_valid_contact_fraction)
    if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("raw valid-contact fraction threshold is invalid")
    return (
        arrays[0] & (arrays[1] >= 0) & arrays[2]
        & ((arrays[3] / float(n_contacts)) >= threshold)
    )


def load_inference_raw_anchors(
    *, subject: str, coverage: Any, raw_cache_dir: Path | None = None,
    event_times: np.ndarray | None = None,
) -> dict[str, Any]:
    """Use the state extractor's authoritative inference-only anchor reader."""
    from .state_extraction import InferenceRawAnchorReader, _sha256_arrays

    if raw_cache_dir is not None:
        raise ValueError(
            "custom raw cache roots are forbidden; use the frozen reader source"
        )
    if event_times is None:
        raise ValueError("hash-verified FullAnchorDesign event times are required")
    reader = InferenceRawAnchorReader(
        subject, np.asarray(event_times, dtype=np.float64)
    )
    anchor_time, anchor_segment, _, anchor_minute, training_guard = (
        reader.inference_anchor_inventory(coverage)
    )
    readable = np.asarray([
        reader.can_read(float(value)) for value in anchor_time
    ], dtype=bool)
    if not bool(readable.all()):
        bad = anchor_time[~readable]
        raise ValueError(
            f"{subject}: authoritative inference anchors are not exactly readable "
            f"at {bad[:5].tolist()}"
        )
    frame = reader.window_index
    n_contacts = int(reader.raw.shape[1])
    threshold = float(reader.inference_min_valid_contact_fraction)
    independently_rebuilt = build_inference_minute_mask(
        covered=frame.covered.to_numpy(dtype=bool),
        session_id=frame.session_id.to_numpy(dtype=np.int64),
        cached=reader.cached,
        n_valid_contacts=frame.n_valid_contacts.to_numpy(dtype=np.int64),
        n_contacts=n_contacts,
        min_valid_contact_fraction=threshold,
    )
    if not np.array_equal(independently_rebuilt, reader.inference_usable):
        raise ValueError(f"{subject}: inventory/reader inference minute masks disagree")
    cache_dir = reader.cache_dir.resolve()
    window_index_path = cache_dir / "window_index_refined.parquet"
    cache_index_path = cache_dir / "cache_index.parquet"
    stats_path = cache_dir / "train_stats.json"
    raw_path = cache_dir / "raw_256hz.zarr"
    artifact_path = cache_dir / "artifact_mask.zarr"
    stats_sha256 = sha256_file(stats_path)
    if reader.source_hashes.get("train_stats") != stats_sha256:
        raise ValueError(f"{subject}: train-stats hash disagrees with state reader")
    return {
        "anchor_time": anchor_time.astype(np.float64),
        "anchor_segment": anchor_segment.astype(np.int64),
        "anchor_minute_index": anchor_minute.astype(np.int64),
        "raw_cache_dir": cache_dir,
        "window_index_path": window_index_path,
        "window_index_sha256": sha256_file(window_index_path),
        "cache_index_path": cache_index_path,
        "cache_index_sha256": sha256_file(cache_index_path),
        "raw_zarr_path": raw_path,
        "raw_zarr_metadata_sha256": sha256_file(raw_path / "zarr.json"),
        "artifact_zarr_path": artifact_path,
        "artifact_zarr_metadata_sha256": sha256_file(artifact_path / "zarr.json"),
        "train_stats_path": stats_path,
        "train_stats_sha256": stats_sha256,
        "train_stats_sha256_match_state_extraction_reader": True,
        "train_stats_n_contacts_match_raw": True,
        "inference_min_valid_contact_fraction": threshold,
        "n_contacts": n_contacts,
        "minimum_valid_contacts": int(math.ceil(threshold * n_contacts)),
        "n_index_minutes": int(len(frame)),
        "n_cached_minutes": int(reader.cached.sum()),
        "n_inference_unlabelled_minutes": int(reader.inference_usable.sum()),
        "n_training_guard_excluded_but_inference_usable_minutes": int(
            np.sum(reader.inference_usable & ~reader.training_guard_free)
        ),
        "n_exact_causal_anchors": int(len(anchor_time)),
        "anchor_time_segment_sha256": _sha256_arrays(
            anchor_time.astype(np.float64), anchor_segment.astype(np.int64)
        ),
        "anchor_inventory_matches_state_extraction_reader": True,
        "training_guard_columns_used_for_inference": False,
        "n_valid_contacts_crosschecked_against_artifact_mask": True,
    }


def load_state_support_arrays(source_repo_root: Path, *, subject: str
                              ) -> dict[str, Any]:
    """Load guarded training anchors plus independent unlabelled inference anchors."""
    from src.topic5_continuous_marked_state_r1.coverage import CoverageTable

    source_repo_root = Path(source_repo_root).resolve()
    cache_manifest_path = (
        source_repo_root
        / "results/epi_prssm/continuous_marked_state/r1/r1_5/cache"
        / subject / "manifest.json"
    )
    cache_manifest = _json(cache_manifest_path)
    if cache_manifest.get("status") != "COMPLETE":
        raise ValueError(f"{subject}: R1.6 observation cache is not COMPLETE")
    if cache_manifest.get("sealed_opened") is not False:
        raise ValueError(f"{subject}: R1.6 observation cache opened sealed data")
    design_path = Path(cache_manifest["design"]).resolve()
    if sha256_file(design_path) != cache_manifest.get("design_sha256"):
        raise ValueError(f"{subject}: R1.6 anchor design SHA256 mismatch")
    with np.load(design_path, allow_pickle=False) as data:
        training_anchor_time = data["anchor_time"].astype(np.float64)
        training_anchor_session = data["anchor_session"].astype(np.int64)
        event_times = data["event_time"].astype(np.float64)

    coverage_path = (
        source_repo_root
        / "results/epi_prssm/continuous_marked_state/r1/r1_2/coverage"
        / f"{subject}.npz"
    )
    coverage_manifest_path = coverage_path.with_suffix(".manifest.json")
    coverage_manifest = _json(coverage_manifest_path)
    if sha256_file(coverage_path) != coverage_manifest.get("output_sha256"):
        raise ValueError(f"{subject}: R1.6 coverage SHA256 mismatch")
    coverage = CoverageTable.load(coverage_path)
    if coverage.subject != subject:
        raise ValueError(f"coverage subject {coverage.subject} != {subject}")
    inference = load_inference_raw_anchors(
        subject=subject, coverage=coverage, event_times=event_times
    )
    return {
        "coverage": coverage,
        "training_anchor_time": training_anchor_time,
        "training_anchor_session": training_anchor_session,
        "inference_anchor_time": inference["anchor_time"],
        "inference_anchor_segment": inference["anchor_segment"],
        "design_path": design_path,
        "design_sha256": cache_manifest["design_sha256"],
        "cache_manifest_path": cache_manifest_path,
        "cache_manifest_sha256": sha256_file(cache_manifest_path),
        "coverage_path": coverage_path,
        "coverage_sha256": coverage_manifest["output_sha256"],
        "coverage_manifest_path": coverage_manifest_path,
        "coverage_revision": coverage_manifest.get("coverage_revision"),
        "postictal_guard_seconds": coverage_manifest.get("postictal_guard_seconds"),
        **{
            key: value for key, value in inference.items()
            if key not in {"anchor_time", "anchor_segment", "anchor_minute_index"}
        },
    }


def _coverage_segment(coverage: Any, left: float, right: float,
                      *, atol: float = 1e-6) -> int | None:
    hit = np.flatnonzero(
        (coverage.start <= float(left) + atol)
        & (coverage.stop >= float(right) - atol)
    )
    return int(hit[0]) if len(hit) == 1 else None


def _control_count(coverage: Any, anchor_time: np.ndarray,
                   seizure_onsets: np.ndarray, lead_seconds: float) -> int:
    count = 0
    for anchor in np.asarray(anchor_time, dtype=np.float64):
        if anchor >= float(coverage.dev_end_epoch):
            continue
        segment = _coverage_segment(coverage, float(anchor), float(anchor + lead_seconds))
        if segment is None:
            continue
        if np.any((seizure_onsets > anchor) & (seizure_onsets <= anchor + lead_seconds)):
            continue
        count += 1
    return count


def summarise_seizure_support(
    seizure_rows: Sequence[Mapping[str, Any]],
    *,
    coverage: Any,
    training_anchor_time: np.ndarray,
    training_anchor_session: np.ndarray,
    inference_anchor_time: np.ndarray,
    inference_anchor_segment: np.ndarray | None = None,
    leads: Sequence[int] = LEAD_MINUTES,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Build patient/lead support and a seizure-level exclusion audit.

    A case is usable only when ``[onset-lead, onset)`` lies inside one recorded
    coverage segment and at least one exact, unlabelled, causal 30 s observation
    is readable at or before the cut-off in that same segment.  The old R1.5
    training-guarded anchors are reported separately and never gate eligibility.
    """
    validate_lead_minutes(leads)
    training_anchor_time = np.asarray(training_anchor_time, dtype=np.float64)
    training_anchor_session = np.asarray(training_anchor_session, dtype=np.int64)
    inference_anchor_time = np.asarray(inference_anchor_time, dtype=np.float64)
    if len(training_anchor_time) != len(training_anchor_session):
        raise ValueError("training anchor time/session arrays disagree")
    if inference_anchor_segment is None:
        inferred_segments: list[int] = []
        for value in inference_anchor_time:
            segment = _coverage_segment(
                coverage, float(value - RAW_WINDOW_SECONDS), float(value)
            )
            inferred_segments.append(-1 if segment is None else int(segment))
        inference_anchor_segment = np.asarray(inferred_segments, dtype=np.int64)
    else:
        inference_anchor_segment = np.asarray(inference_anchor_segment, dtype=np.int64)
    if len(inference_anchor_time) != len(inference_anchor_segment):
        raise ValueError("inference anchor time/segment arrays disagree")
    if np.any(inference_anchor_segment < 0):
        raise ValueError("inference anchor crosses excluded or unrecorded coverage")
    all_onsets = np.asarray([float(row["onset_epoch"]) for row in seizure_rows])
    detail: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    subject = str(seizure_rows[0]["subject"]) if seizure_rows else str(coverage.subject)
    for lead_minutes in leads:
        lead_seconds = float(lead_minutes) * 60.0
        counts: Counter[str] = Counter()
        for source in seizure_rows:
            onset = float(source["onset_epoch"])
            in_development = onset < float(coverage.dev_end_epoch)
            cutoff = onset - lead_seconds
            segment = (
                _coverage_segment(coverage, cutoff, onset)
                if in_development else None
            )
            complete = segment is not None
            training_anchor = False
            latest_training_anchor = None
            training_anchor_gap = None
            inference_available = False
            latest_inference_anchor = None
            inference_freshness = None
            if complete:
                training_mask = (
                    (training_anchor_time >= coverage.start[segment])
                    & (training_anchor_time <= cutoff)
                    & (training_anchor_session == coverage.session[segment])
                )
                training_candidates = training_anchor_time[training_mask]
                if len(training_candidates):
                    latest_training_anchor = float(training_candidates.max())
                    training_anchor_gap = float(cutoff - latest_training_anchor)
                    training_anchor = True
                inference_mask = (
                    (inference_anchor_time <= cutoff)
                    & (inference_anchor_segment == int(segment))
                )
                inference_candidates = inference_anchor_time[inference_mask]
                if len(inference_candidates):
                    latest_inference_anchor = float(inference_candidates.max())
                    inference_freshness = float(cutoff - latest_inference_anchor)
                    inference_available = True
            eligible = bool(
                source.get("matched") is True
                and source.get("onset_exact_match") is True
                and in_development and complete and inference_available
            )
            if not in_development:
                reason = "outside_development_partition"
            elif source.get("matched") is not True or source.get("onset_exact_match") is not True:
                reason = "crosswalk_not_exact"
            elif not complete:
                reason = "lead_window_crosses_gap_or_unrecorded_time"
            elif not inference_available:
                reason = "no_h2b_inference_observation_at_or_before_cutoff_in_segment"
            else:
                reason = "eligible"
            counts[reason] += 1
            detail.append({
                "subject": subject,
                "seizure_id": source.get("canonical_seizure_id"),
                "recording_code": source.get("recording_code"),
                "lead_minutes": int(lead_minutes),
                "primary_lead": int(lead_minutes) == PRIMARY_LEAD_MINUTES,
                "onset_epoch": onset,
                "cutoff_epoch": cutoff,
                "in_development_partition": in_development,
                "complete_recorded_lead_window": complete,
                "crosses_gap_or_unrecorded_time": bool(in_development and not complete),
                "coverage_segment": segment,
                "training_guarded_anchor_exists": training_anchor,
                "latest_training_guarded_anchor_epoch": latest_training_anchor,
                "training_guarded_anchor_freshness_seconds": training_anchor_gap,
                "h2b_inference_observation_available_at_cutoff": inference_available,
                "latest_h2b_inference_observation_epoch": latest_inference_anchor,
                "h2b_inference_observation_freshness_seconds": inference_freshness,
                "eligible": eligible,
                "exclusion_reason": reason,
            })
        n_eligible = counts["eligible"]
        n_controls = _control_count(
            coverage, inference_anchor_time, all_onsets, lead_seconds
        )
        summaries.append({
            "subject": subject,
            "lead_minutes": int(lead_minutes),
            "primary_lead": int(lead_minutes) == PRIMARY_LEAD_MINUTES,
            "n_seizures_total": len(seizure_rows),
            "n_seizures_development": int(np.sum(all_onsets < coverage.dev_end_epoch)),
            "n_exact_crosswalk": int(sum(
                row.get("matched") is True and row.get("onset_exact_match") is True
                for row in seizure_rows
            )),
            "n_complete_recorded_lead_window": int(sum(
                row["complete_recorded_lead_window"]
                for row in detail if row["lead_minutes"] == int(lead_minutes)
            )),
            "n_training_guarded_anchor_exists": int(sum(
                row["training_guarded_anchor_exists"]
                for row in detail if row["lead_minutes"] == int(lead_minutes)
            )),
            "n_h2b_inference_observation_available_at_cutoff": int(sum(
                row["h2b_inference_observation_available_at_cutoff"]
                for row in detail if row["lead_minutes"] == int(lead_minutes)
            )),
            "n_eligible_seizures": int(n_eligible),
            "n_candidate_control_state_anchors": n_controls,
            "n_candidate_controls_after_ictal_postictal_exclusion": n_controls,
            "control_count_unit": (
                "unlabelled exact causal 30 s inference anchors; not independent seizures"
            ),
            "support_tier": support_tier(n_eligible),
            "formal_test_partition_opened": False,
            "sealed_opened": False,
        })
    funnel = {
        "subject": subject,
        "n_seizures_total": len(seizure_rows),
        "n_seizures_development": int(np.sum(all_onsets < coverage.dev_end_epoch)),
        "by_lead": {
            str(lead): dict(Counter(
                row["exclusion_reason"] for row in detail
                if row["lead_minutes"] == int(lead)
            ))
            for lead in leads
        },
        "primary_lead_minutes": PRIMARY_LEAD_MINUTES,
        "primary_support_tier": next(
            row["support_tier"] for row in summaries if row["primary_lead"]
        ),
    }
    return summaries, funnel, detail


def target_inventory(source_repo_root: Path, *, subject: str) -> dict[str, Any]:
    """Report frozen pre-existing phenotype targets without deriving new labels."""
    source_repo_root = Path(source_repo_root).resolve()
    subtype = (
        source_repo_root
        / "results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/per_subject"
        / f"{subject}__zer_binned.json"
    )
    recruitment = (
        source_repo_root
        / "results/topic5_ictal_recruitment/ictal_field_long_cache"
        / f"{subject}.json"
    )
    recruitment_count = 0
    if recruitment.exists():
        payload = _json(recruitment)
        recruitment_count = len(payload.get("eligible_idxs") or [])
    return {
        "preexisting_seizure_subtype_available": subtype.exists(),
        "preexisting_seizure_subtype_path": str(subtype) if subtype.exists() else None,
        "preexisting_seizure_subtype_sha256": sha256_file(subtype) if subtype.exists() else None,
        "preexisting_early_recruitment_available": recruitment.exists(),
        "preexisting_early_recruitment_path": (
            str(recruitment) if recruitment.exists() else None
        ),
        "preexisting_early_recruitment_sha256": (
            sha256_file(recruitment) if recruitment.exists() else None
        ),
        "n_preexisting_early_recruitment_seizures": int(recruitment_count),
        "targets_reclustered_from_h2b_state": False,
    }


def exclusion_funnel_payload(*, checkpoint_inventory: Mapping[str, Any],
                             support_funnel: Mapping[str, Any],
                             targets: Mapping[str, Any],
                             source_arrays: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "COMPLETE",
        "contract": H2B_REVISION,
        "inventory_revision": INVENTORY_REVISION,
        "created_utc": utc_now(),
        "phase": "Phase 0 read-only inventory",
        "checkpoint_funnel": {
            "subject": checkpoint_inventory["subject"],
            "stable_seeds_expected": checkpoint_inventory["stable_seeds"],
            "checkpoint_hash_verified": checkpoint_inventory["n_checkpoints"],
            "checkpoint_hash_failed": 0,
        },
        "seizure_funnel": dict(support_funnel),
        "target_availability": dict(targets),
        "support_sources": {
            key: str(source_arrays[key]) if isinstance(source_arrays[key], Path)
            else source_arrays[key]
            for key in (
                "design_path", "design_sha256", "cache_manifest_path",
                "cache_manifest_sha256", "coverage_path", "coverage_sha256",
                "coverage_manifest_path", "coverage_revision",
                "postictal_guard_seconds", "raw_cache_dir",
                "window_index_path", "window_index_sha256",
                "cache_index_path", "cache_index_sha256", "raw_zarr_path",
                "raw_zarr_metadata_sha256", "artifact_zarr_path",
                "artifact_zarr_metadata_sha256", "train_stats_path",
                "train_stats_sha256", "inference_min_valid_contact_fraction",
                "train_stats_sha256_match_state_extraction_reader",
                "train_stats_n_contacts_match_raw", "n_contacts",
                "minimum_valid_contacts", "n_index_minutes",
                "n_cached_minutes", "n_inference_unlabelled_minutes",
                "n_training_guard_excluded_but_inference_usable_minutes",
                "n_exact_causal_anchors", "anchor_time_segment_sha256",
                "anchor_inventory_matches_state_extraction_reader",
                "training_guard_columns_used_for_inference",
                "n_valid_contacts_crosschecked_against_artifact_mask",
            )
        },
        "boundary": RunBoundary().__dict__,
        "seizure_probe_trained": False,
        "r1_7_used": False,
    }
