#!/usr/bin/env python3
"""G0 target-blind causal-prefix audit for the cross-event HistoryRNN.

The script may read target routing metadata, target contact names, and NPZ key
inventory.  It never accesses an early-ictal energy array.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_topic5_interictal_operator_dataset import (  # noqa: E402
    _raw_subject_dir,
)
from src.interictal_propagation import load_subject_propagation_events  # noqa: E402
from src.topic5_history_rnn import (  # noqa: E402
    build_continuous_segment_ids,
    exact_contact_join,
    select_causal_prefix,
)


GUARD_SECONDS = 10.0 * 60.0
POSTICTAL_SECONDS = 70.0 * 60.0
MIN_CONTACTS = 6
MIN_HISTORY_EVENTS = 8
MIN_G2_PATIENTS = 8
MIN_DISTINCT_HISTORIES = 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_inventory(fit1: Path) -> pd.DataFrame:
    frame = pd.read_csv(fit1)
    strict = (
        frame.loc[
            (frame.group_id == "strict_broadband")
            & (frame.time_reference == "clinical_onset"),
            ["subject", "seizure_idx"],
        ]
        .drop_duplicates()
        .sort_values(["subject", "seizure_idx"])
        .reset_index(drop=True)
    )
    if strict.subject.nunique() != 16 or len(strict) != 106:
        raise RuntimeError(
            f"strict clinical-onset denominator drifted: "
            f"{strict.subject.nunique()}/16 patients, {len(strict)}/106 seizures"
        )
    return strict


def _seizure_tables(path: Path) -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(path, dtype={"subject": str, "recording_id": str})
    frame = frame.loc[frame.clin_onset_epoch.notna()].copy()
    return {
        f"epilepsiae_{subject}": group.sort_values("clin_onset_epoch").reset_index(
            drop=True
        )
        for subject, group in frame.groupby("subject")
    }


def _previous_postictal_end(seizures: pd.DataFrame, onset_epoch: float) -> float:
    previous = seizures.loc[seizures.clin_onset_epoch < float(onset_epoch)]
    if previous.empty:
        return -np.inf
    last = previous.iloc[-1]
    offset = last.get("eeg_offset_epoch", np.nan)
    if not np.isfinite(float(offset)):
        offset = last.get("clin_offset_epoch", np.nan)
    if not np.isfinite(float(offset)):
        return np.inf
    return float(offset) + POSTICTAL_SECONDS


def _block_metadata(frame: pd.DataFrame) -> dict[str, dict]:
    required = {
        "block_stem",
        "recording_id",
        "block_no",
        "block_start_epoch",
        "block_end_epoch",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"block inventory missing columns: {sorted(missing)}")
    ordered = frame.sort_values(["block_start_epoch", "block_stem"]).reset_index(
        drop=True
    )
    ordered["sequence_index"] = np.arange(len(ordered), dtype=int)
    return {
        str(row.block_stem): {
            "recording_id": str(row.recording_id),
            "block_no": int(row.block_no),
            "sequence_index": int(row.sequence_index),
            "block_start_epoch": float(row.block_start_epoch),
            "block_end_epoch": float(row.block_end_epoch),
        }
        for row in ordered.itertuples()
    }


def _timeline(
    subject: str,
    dataset_npz: Path,
    patient_blocks: pd.DataFrame,
) -> dict[str, np.ndarray]:
    with np.load(dataset_npz, allow_pickle=False) as data:
        times = np.asarray(data["event_abs_time"], np.float64)
        source_index = np.asarray(data["event_source_index"], np.int64)
        contacts = np.asarray(data["contact_names"])
    raw = load_subject_propagation_events(_raw_subject_dir(subject))
    raw_times = np.asarray(raw["event_abs_times"], np.float64)
    raw_block_id = np.asarray(raw["block_ids"], np.int64)
    if np.any(source_index < 0) or np.any(source_index >= raw_times.size):
        raise RuntimeError(f"{subject}: event_source_index outside raw event array")
    if not np.allclose(times, raw_times[source_index], atol=1e-6, rtol=0):
        raise RuntimeError(f"{subject}: dataset/raw event time alignment drifted")
    event_block_id = raw_block_id[source_index]
    raw_names = np.asarray([str(value) for value in raw["record_names"]])
    if np.any(event_block_id < 0) or np.any(event_block_id >= raw_names.size):
        raise RuntimeError(f"{subject}: raw block id outside record_names")
    block_stem = raw_names[event_block_id]
    metadata = _block_metadata(patient_blocks)
    segment, reset = build_continuous_segment_ids(
        block_stem,
        metadata,
        allow_cross_recording_contiguous=True,
    )
    recording = np.asarray(
        [str(metadata[str(stem)]["recording_id"]) for stem in block_stem]
    )
    delta_t = np.diff(times, prepend=times[0])
    delta_t[reset] = 0.0
    return {
        "event_time": times,
        "event_source_index": source_index,
        "event_block_id": event_block_id,
        "event_block_stem": block_stem,
        "event_recording_id": recording,
        "event_segment_id": segment,
        "event_reset": reset,
        "event_delta_t_sec": delta_t,
        "contacts": contacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=ROOT,
        help="Root holding ignored result artifacts; code still comes from this worktree.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/topic5_history_rnn_early_ictal_field/g0_causal_prefix",
    )
    args = parser.parse_args()
    artifact = args.artifact_root.resolve()
    output = args.output_dir.resolve()
    dataset = artifact / "results/topic5_interictal_rank_distribution/dataset_v0_4"
    fit1 = (
        artifact
        / "results/topic5_state_conditioned_predictor/fit12_clinical_bb150/fit1/"
        "fig6_fit1_clinical_onset_scaffold_event.csv"
    )
    seizures_path = artifact / "results/epilepsiae_seizure_inventory.csv"
    blocks_path = artifact / "results/epilepsiae_block_inventory.csv"
    target_root = artifact / "results/topic5_ictal_recruitment/t0_feature_cache_bb150_1_150"

    strict = _strict_inventory(fit1)
    seizures_by_subject = _seizure_tables(seizures_path)
    blocks = pd.read_csv(blocks_path, dtype={"subject": str, "recording_id": str})
    event_rows: list[dict] = []
    subject_rows: list[dict] = []
    timeline_dir = output / "timeline"
    timeline_dir.mkdir(parents=True, exist_ok=True)

    for subject, targets in strict.groupby("subject", sort=True):
        short = subject.split("_", 1)[1]
        npz_path = dataset / "per_subject" / f"{subject}.npz"
        timeline = _timeline(
            subject,
            npz_path,
            blocks.loc[blocks.subject.astype(str) == short],
        )
        target_path = target_root / f"{subject}.npz"
        if not target_path.exists():
            raise FileNotFoundError(target_path)
        # NPZ access is lazy.  Only the channels metadata array is materialized;
        # no bb150_auc target array is indexed or deserialized.
        with np.load(target_path, allow_pickle=False) as target_npz:
            target_keys = set(target_npz.files)
            target_contacts = np.asarray(target_npz["channels"])
        join = exact_contact_join(timeline["contacts"], target_contacts)
        seizures = seizures_by_subject[subject]
        if len(seizures) <= int(targets.seizure_idx.max()):
            raise RuntimeError(f"{subject}: target seizure index outside inventory")
        patient_rows = []
        for target in targets.itertuples(index=False):
            seizure_index = int(target.seizure_idx)
            key = f"bb150_auc__{seizure_index}"
            seizure = seizures.iloc[seizure_index]
            onset = float(seizure.clin_onset_epoch)
            previous_postictal_end = _previous_postictal_end(seizures, onset)
            prefix = select_causal_prefix(
                timeline["event_time"],
                timeline["event_segment_id"],
                timeline["event_recording_id"],
                seizure_recording_id=str(seizure.recording_id),
                clinical_onset_epoch=onset,
                guard_seconds=GUARD_SECONDS,
                previous_postictal_end_epoch=previous_postictal_end,
            )
            index = prefix.event_indices
            n_events = int(index.size)
            row = {
                "subject": subject,
                "seizure_idx": seizure_index,
                "seizure_id": str(seizure.seizure_id),
                "recording_id": str(seizure.recording_id),
                "clinical_onset_epoch": onset,
                "guard_seconds": GUARD_SECONDS,
                "previous_postictal_end_epoch": (
                    previous_postictal_end if np.isfinite(previous_postictal_end) else np.nan
                ),
                "n_causal_events": n_events,
                "causal_history_span_hours": (
                    float(
                        (timeline["event_time"][index[-1]] - timeline["event_time"][index[0]])
                        / 3600.0
                    )
                    if n_events >= 2
                    else 0.0
                ),
                "last_event_gap_hours": (
                    float((onset - timeline["event_time"][index[-1]]) / 3600.0)
                    if n_events
                    else np.nan
                ),
                "segment_id": int(prefix.segment_id),
                "last_event_index": int(prefix.last_event_index),
                "history_fingerprint": (
                    f"{subject}:{prefix.segment_id}:{prefix.last_event_index}" if n_events else ""
                ),
                "target_key_present": key in target_keys,
                "n_interictal_contacts": int(len(timeline["contacts"])),
                "n_exact_joined_contacts": int(join.size),
                "contact_gate_pass": bool(join.size >= MIN_CONTACTS),
                "history_gate_pass": bool(n_events >= MIN_HISTORY_EVENTS),
                "g2_metadata_eligible": bool(
                    key in target_keys
                    and join.size >= MIN_CONTACTS
                    and n_events >= MIN_HISTORY_EVENTS
                ),
                "exclusion_reason": prefix.exclusion_reason,
            }
            event_rows.append(row)
            patient_rows.append(row)

        patient = pd.DataFrame(patient_rows)
        eligible = patient.loc[patient.g2_metadata_eligible]
        subject_rows.append(
            {
                "subject": subject,
                "n_strict_seizures": int(len(patient)),
                "n_g2_metadata_eligible_seizures": int(len(eligible)),
                "n_distinct_eligible_histories": int(
                    eligible.history_fingerprint.nunique()
                ),
                "n_contacts": int(len(timeline["contacts"])),
                "n_continuous_segments": int(
                    np.unique(timeline["event_segment_id"]).size
                ),
                "g2_patient_eligible": bool(len(eligible) > 0),
                "g3_pairing_eligible": bool(
                    eligible.history_fingerprint.nunique() >= 2
                ),
                "g3_residual_candidate": bool(
                    eligible.history_fingerprint.nunique() >= 3
                ),
            }
        )
        np.savez_compressed(
            timeline_dir / f"{subject}.npz",
            **{key: value for key, value in timeline.items() if key != "contacts"},
            contact_names=timeline["contacts"],
            target_contact_index=join,
        )

    seizure_frame = pd.DataFrame(event_rows)
    subject_frame = pd.DataFrame(subject_rows)
    seizure_frame.to_csv(output / "seizure_causal_history_inventory.csv", index=False)
    subject_frame.to_csv(output / "subject_causal_history_inventory.csv", index=False)
    eligible = seizure_frame.loc[seizure_frame.g2_metadata_eligible]
    n_g2_patients = int(subject_frame.g2_patient_eligible.sum())
    n_distinct = int(eligible.history_fingerprint.nunique())
    target_key_complete = bool(seizure_frame.target_key_present.all())
    contact_join_complete = bool(seizure_frame.contact_gate_pass.all())
    g0_pass = bool(
        target_key_complete
        and contact_join_complete
        and n_g2_patients >= MIN_G2_PATIENTS
        and n_distinct >= MIN_DISTINCT_HISTORIES
    )
    summary = {
        "contract": "topic5_history_rnn_early_ictal_field_v0_1",
        "status": "COMPLETE",
        "target_values_read": False,
        "target_arrays_deserialized": False,
        "target_metadata_arrays_deserialized": ["channels"],
        "frozen_endpoint": {
            "time_reference": "clinical_onset",
            "window_sec": [0.0, 10.0],
            "band_hz": [1.0, 150.0],
            "guard_seconds": GUARD_SECONDS,
            "postictal_reset_seconds": POSTICTAL_SECONDS,
        },
        "strict_inventory": {
            "n_patients": int(seizure_frame.subject.nunique()),
            "n_seizures": int(len(seizure_frame)),
            "target_key_complete": target_key_complete,
            "exact_contact_join_complete": contact_join_complete,
        },
        "causal_history_inventory": {
            "minimum_events": MIN_HISTORY_EVENTS,
            "n_g2_metadata_eligible_patients": n_g2_patients,
            "n_g2_metadata_eligible_seizures": int(len(eligible)),
            "n_distinct_eligible_histories": n_distinct,
            "n_g3_pairing_eligible_patients": int(
                subject_frame.g3_pairing_eligible.sum()
            ),
            "n_g3_residual_candidate_patients": int(
                subject_frame.g3_residual_candidate.sum()
            ),
        },
        "g0_thresholds": {
            "minimum_joined_contacts": MIN_CONTACTS,
            "minimum_history_events": MIN_HISTORY_EVENTS,
            "minimum_g2_patients": MIN_G2_PATIENTS,
            "minimum_distinct_histories": MIN_DISTINCT_HISTORIES,
        },
        "g0_verdict": "PASS" if g0_pass else "FAIL",
        "next_action": (
            "IMPLEMENT_AND_RUN_G1_TARGET_SEALED"
            if g0_pass
            else "STOP_AND_REPAIR_CAUSAL_TIMELINE"
        ),
        "input_hashes": {
            "fit1_metadata_csv": _sha256(fit1),
            "seizure_inventory_csv": _sha256(seizures_path),
            "block_inventory_csv": _sha256(blocks_path),
            "rank_dataset_manifest": _sha256(dataset / "dataset_manifest.json"),
        },
    }
    (output / "G0_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not g0_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
