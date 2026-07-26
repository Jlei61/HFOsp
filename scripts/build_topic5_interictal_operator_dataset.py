#!/usr/bin/env python3
"""Build the v0.3 within-event masked-rank dataset without reading ictal values.

Candidate patients use the exact frozen Fit-2 12 h prefix block IDs. Auxiliary
Epilepsiae and Yuquan patients use all fail-closed definite-interictal blocks.
Every output event is one independent within-event recruitment sequence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_topic5_state_conditioned_dataset import (  # noqa: E402
    _event_mask_from_blocks,
    _inventory,
    _raw_subject_dir,
    _seizure_intervals,
    eligible_blocks,
)
from src.interictal_propagation import load_subject_propagation_events  # noqa: E402
from src.seeg_coord_loader import load_subject_coords  # noqa: E402
from src.topic5_interictal_operator import (  # noqa: E402
    CONTACT_FEATURE_NAMES,
    build_contact_features,
    encode_recruitment_matrix,
)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_manifest_fingerprint(subject_dir: Path) -> tuple[str, list[dict]]:
    lagpat = sorted(subject_dir.glob("*_lagPat_withFreqCent.npz"))
    if not lagpat:
        lagpat = sorted(subject_dir.glob("*_lagPat.npz"))
    records = []
    for path in lagpat:
        stat = path.stat()
        records.append(
            {
                "path": str(path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest(), records


def _load_subject_index(cfg: dict) -> list[str]:
    root = ROOT / cfg["cohort"]["masked_subject_index"]
    subjects = []
    for path in sorted(root.glob("*.json")):
        record = json.loads(path.read_text())
        dataset = str(record.get("dataset", ""))
        subject = str(record.get("subject", ""))
        if dataset in set(cfg["cohort"]["datasets"]) and subject:
            subjects.append(f"{dataset}_{subject}")
    subjects = sorted(set(subjects))
    if not subjects:
        raise RuntimeError("masked propagation subject index is empty")
    return subjects


def _candidate_contract(cfg: dict) -> tuple[list[str], pd.DataFrame]:
    attrition_path = ROOT / cfg["cohort"]["candidate_prefix_attrition"]
    attrition = pd.read_csv(attrition_path)
    passed = attrition.prefix_field_pass.astype(str).str.lower().isin(
        ("true", "1", "yes")
    )
    candidates = sorted(attrition.loc[passed, "subject"].astype(str).unique())
    expected = int(cfg["cohort"]["candidate_subjects_expected"])
    if len(candidates) != expected:
        raise RuntimeError(
            f"candidate subject contract drift: expected {expected}, got {len(candidates)}"
        )

    # Only routing/provenance columns are loaded. observed/null/energy values
    # are deliberately not read before Stage B.
    target_path = ROOT / cfg["cohort"]["accepted_strict_bb150_event_table"]
    routing = pd.read_csv(
        target_path,
        usecols=["subject", "seizure_idx", "group_id", "time_reference", "band"],
    )
    routing = routing[routing.group_id.astype(str) == "strict_broadband"].copy()
    if set(routing.time_reference.astype(str)) != {"clinical_onset"}:
        raise RuntimeError("strict BB150 routing is not clinical-onset aligned")
    if set(routing.band.astype(str)) != {"broadband_1_150"}:
        raise RuntimeError("strict BB150 routing band drift")
    n_expected = int(cfg["cohort"]["candidate_strict_seizures_expected"])
    if routing.subject.nunique() != expected or len(routing) != n_expected:
        raise RuntimeError(
            "strict BB150 candidate contract drift: "
            f"expected {expected}/{n_expected}, got "
            f"{routing.subject.nunique()}/{len(routing)}"
        )
    if set(routing.subject.astype(str)) != set(candidates):
        raise RuntimeError("candidate prefix cohort and strict BB150 cohort disagree")
    return candidates, routing


def _frozen_prefix_blocks(subject: str, cfg: dict) -> tuple[np.ndarray, dict]:
    path = ROOT / cfg["cohort"]["candidate_prefix_records"] / f"{subject}.json"
    record = json.loads(path.read_text())
    provenance = record.get("prefix_provenance") or {}
    blocks = np.asarray(provenance.get("prefix_block_ids") or [], dtype=int)
    expected = int(cfg["cohort"]["calibration_hours"])
    if blocks.size != expected:
        raise RuntimeError(
            f"{subject}: expected {expected} frozen prefix blocks, got {blocks.size}"
        )
    return blocks, provenance


def _auxiliary_interictal_blocks(
    subject: str, events: Dict[str, object], cfg: dict
) -> np.ndarray:
    dataset = subject.split("_", 1)[0]
    inventory = _inventory(subject)
    intervals = _seizure_intervals(subject, inventory)
    post_guard = float(cfg["cohort"]["seizure_guard_post_minutes"]) * 60.0
    if dataset == "epilepsiae":
        return np.flatnonzero(
            eligible_blocks(
                events,
                intervals,
                post_guard_sec=post_guard,
                timezone="Europe/Berlin",
            )
        )
    if dataset != "yuquan":
        raise ValueError(f"unknown dataset: {dataset}")

    # Yuquan lagPat blocks are recording-sized (usually 2 h), not Epilepsiae
    # 1 h blocks. Reusing eligible_blocks() would classify every normal 7200 s
    # adjacency as a >5400 s discontinuity and delete the entire cohort.
    # Recurrence is within one event, so cross-record gaps and day/night
    # boundaries are irrelevant here. We only exclude recording blocks that
    # overlap a seizure or its frozen post-ictal guard.
    starts = np.asarray(events["block_start_times"], float)
    if not intervals:
        return np.flatnonzero(np.isfinite(starts))
    record_names = [str(x) for x in events["record_names"]]
    block_inventory_path = ROOT / "results/dataset_inventory/yuquan_block_inventory.csv"
    block_inventory = pd.read_csv(block_inventory_path)
    short_subject = subject.split("_", 1)[1]
    block_inventory = block_inventory[
        block_inventory.subject.astype(str) == short_subject
    ]
    end_by_record = {
        str(row.block_stem): float(row.block_end_epoch)
        for row in block_inventory.itertuples()
        if np.isfinite(float(row.block_end_epoch))
    }
    return np.flatnonzero(
        yuquan_interictal_block_mask(
            starts,
            record_names,
            intervals,
            post_guard_sec=post_guard,
            end_by_record=end_by_record,
        )
    )


def yuquan_interictal_block_mask(
    starts: np.ndarray,
    record_names: Sequence[str],
    intervals,
    *,
    post_guard_sec: float,
    end_by_record: Mapping[str, float],
) -> np.ndarray:
    """Yuquan recording-block eligibility without the Epilepsiae 1 h gap rule."""
    starts = np.asarray(starts, float)
    if len(record_names) != starts.size:
        raise ValueError("record_names and starts must be aligned")
    good = np.isfinite(starts)
    finite_gap = np.diff(starts[np.isfinite(starts)])
    finite_gap = finite_gap[(finite_gap > 0) & (finite_gap < 10_000)]
    fallback_duration = float(np.median(finite_gap)) if finite_gap.size else 7200.0
    for block_id, (record_name, start) in enumerate(zip(record_names, starts)):
        if not good[block_id]:
            continue
        end = float(end_by_record.get(str(record_name), start + fallback_duration))
        for _, onset, offset in intervals:
            if start < offset + float(post_guard_sec) and end > onset:
                good[block_id] = False
                break
    return good


def _load_geometry(subject: str, names: Sequence[str], cfg: dict):
    dataset, sid = subject.split("_", 1)
    allow_voxel = bool(cfg["contact_features"]["allow_epilepsiae_voxel_fallback"])
    try:
        record = load_subject_coords(
            dataset,
            sid,
            names,
            allow_voxel_fallback=allow_voxel,
        )
        coords = np.asarray(record.coords_array_in_requested_order, float)
        status = {
            "status": "ok",
            "coord_space": str(record.coord_space),
            "coord_units": str(record.coord_units),
            "provenance": _jsonable(record.provenance),
            "n_mapped": int(np.sum(record.mapped_mask_in_requested_order)),
        }
    except Exception as exc:
        if not bool(cfg["contact_features"]["allow_missing_geometry"]):
            raise
        coords = np.full((len(names), 3), np.nan)
        status = {
            "status": f"unavailable:{type(exc).__name__}",
            "error": str(exc)[:300],
            "n_mapped": 0,
        }
    return coords, status


def _tie_audit(
    lag_event_contact: np.ndarray,
    participation_event_contact: np.ndarray,
    thresholds_ms: Iterable[float],
) -> dict:
    differences = []
    exact = 0
    for lag, part in zip(lag_event_contact, participation_event_contact):
        values = np.sort(lag[part & np.isfinite(lag)])
        if values.size < 2:
            continue
        delta = np.diff(values)
        differences.extend(delta.tolist())
        exact += int(np.sum(delta == 0.0))
    delta = np.asarray(differences, float)
    out = {
        "n_adjacent_transitions": int(delta.size),
        "exact_tie_fraction": float(exact / delta.size) if delta.size else np.nan,
    }
    for threshold in thresholds_ms:
        key = f"near_tie_fraction_le_{float(threshold):g}ms"
        out[key] = (
            float(np.mean(delta <= float(threshold) / 1000.0 + 1e-12))
            if delta.size
            else np.nan
        )
    for quantile in (0.1, 0.5, 0.9):
        out[f"adjacent_lag_q{int(100 * quantile):02d}_ms"] = (
            float(np.quantile(delta, quantile) * 1000.0) if delta.size else np.nan
        )
    return out


def build_subject(
    subject: str,
    cfg: dict,
    candidates: set[str],
    out_dir: Path,
    *,
    overwrite: bool,
    force_all_interictal: bool = False,
) -> dict:
    dataset, _ = subject.split("_", 1)
    subject_dir = _raw_subject_dir(subject)
    events = load_subject_propagation_events(subject_dir)
    times = np.asarray(events["event_abs_times"], float)
    ranks = np.asarray(events["ranks"], float)
    bools = np.asarray(events["bools"], bool)
    lag_raw = np.asarray(events["lag_raw"], float)
    names = [str(x) for x in events["channel_names"]]
    candidate = subject in candidates
    if candidate and not force_all_interictal:
        blocks, prefix_provenance = _frozen_prefix_blocks(subject, cfg)
        block_policy = "frozen_fit2_12h_prefix"
    else:
        blocks = _auxiliary_interictal_blocks(subject, events, cfg)
        prefix_provenance = {}
        block_policy = "all_fail_closed_definite_interictal_blocks"

    selected = np.flatnonzero(
        _event_mask_from_blocks(events, blocks) & np.isfinite(times)
    )
    selected = selected[np.argsort(times[selected], kind="stable")]
    n_events_before_participant_gate = int(selected.size)
    min_participants = int(cfg["cohort"]["min_participants"])
    selected = selected[np.sum(bools[:, selected], axis=0) >= min_participants]
    if selected.size == 0:
        raise RuntimeError(f"{subject}: no event survives participant gate")

    local_rank, group_ids, group_counts = encode_recruitment_matrix(
        ranks[:, selected],
        bools[:, selected],
        lag_raw[:, selected],
        tie_tolerance_seconds=float(cfg["event_encoding"]["tie_tolerance_seconds"]),
    )
    min_groups = int(cfg["cohort"]["min_recruitment_sets"])
    keep = group_counts >= min_groups
    selected = selected[keep]
    local_rank = local_rank[keep]
    group_ids = group_ids[keep]
    group_counts = group_counts[keep]
    participation = bools[:, selected].T
    lag_event_contact = lag_raw[:, selected].T.astype(np.float32)
    event_times = times[selected].astype(np.float64)
    if selected.size == 0:
        raise RuntimeError(f"{subject}: no event survives recruitment-set gate")

    split = int(np.floor(float(cfg["cohort"]["heldout_calibration_fraction"]) * len(selected)))
    split = min(max(split, 1), len(selected) - 1)
    event_split = np.ones(len(selected), dtype=np.uint8)
    event_split[:split] = 0
    support = np.mean(participation[:split], axis=0)
    coords, geometry_status = _load_geometry(subject, names, cfg)
    contact_features, feature_metadata = build_contact_features(names, support, coords)

    arrays = {
        "event_local_rank": local_rank.astype(np.float32),
        "event_group_ids": group_ids.astype(np.int16),
        "event_group_count": group_counts.astype(np.int16),
        "event_participation": participation.astype(np.uint8),
        "event_lag_raw": lag_event_contact,
        "event_abs_time": event_times,
        "event_source_index": selected.astype(np.int64),
        "event_split": event_split,
        "contact_features": contact_features,
        "contact_feature_names": np.asarray(CONTACT_FEATURE_NAMES),
        "contact_names": np.asarray(names),
        "contact_coords": coords.astype(np.float32),
        "prefix_participation_support": support.astype(np.float32),
        "selected_block_ids": np.asarray(blocks, dtype=np.int32),
    }
    per_subject = out_dir / "per_subject"
    per_subject.mkdir(parents=True, exist_ok=True)
    npz_path = per_subject / f"{subject}.npz"
    json_path = per_subject / f"{subject}.json"
    if (npz_path.exists() or json_path.exists()) and not overwrite:
        raise FileExistsError(
            f"{subject}: output exists; pass --overwrite for a deterministic rebuild"
        )
    np.savez_compressed(npz_path, **arrays)

    source_fingerprint, source_files = _source_manifest_fingerprint(subject_dir)
    tie = _tie_audit(
        lag_event_contact,
        participation,
        cfg["event_encoding"]["near_tie_audit_ms"],
    )
    participant_count = np.sum(participation, axis=1)
    metadata = {
        "contract": cfg["contract"]["name"],
        "contract_version": cfg["contract"]["version"],
        "dataset": dataset,
        "subject": subject,
        "candidate_target_patient": candidate,
        "block_policy": block_policy,
        "selected_block_ids": blocks,
        "prefix_provenance": prefix_provenance,
        "n_contacts": len(names),
        "n_events_before_participant_gate": n_events_before_participant_gate,
        "n_events": int(len(selected)),
        "n_train_calibration_events": int(split),
        "n_heldout_events": int(len(selected) - split),
        "train_fraction_realized": float(split / len(selected)),
        "participant_count_median": float(np.median(participant_count)),
        "participant_count_q10": float(np.quantile(participant_count, 0.1)),
        "participant_count_q90": float(np.quantile(participant_count, 0.9)),
        "recruitment_group_count_median": float(np.median(group_counts)),
        "recruitment_group_count_q10": float(np.quantile(group_counts, 0.1)),
        "recruitment_group_count_q90": float(np.quantile(group_counts, 0.9)),
        "tie_audit": tie,
        "tie_tolerance_seconds_primary": float(
            cfg["event_encoding"]["tie_tolerance_seconds"]
        ),
        "contact_feature_metadata": feature_metadata,
        "geometry": geometry_status,
        "source_file_manifest_sha256": source_fingerprint,
        "source_files": source_files,
        "dataset_npz_sha256": _sha256(npz_path),
        "forbidden_inputs_present": {
            "inter_event_interval": False,
            "event_rate": False,
            "time_to_seizure": False,
            "seizure_seed": False,
            "ictal_target": False,
            "patient_or_channel_string_model_feature": False,
        },
    }
    json_path.write_text(
        json.dumps(_jsonable(metadata), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "dataset": dataset,
        "subject": subject,
        "candidate_target_patient": candidate,
        "status": "ok",
        "reason": "",
        "block_policy": block_policy,
        "n_contacts": len(names),
        "n_selected_blocks": int(len(blocks)),
        "n_events_before_participant_gate": n_events_before_participant_gate,
        "n_events": int(len(selected)),
        "n_train_calibration_events": int(split),
        "n_heldout_events": int(len(selected) - split),
        "participant_count_median": float(np.median(participant_count)),
        "group_count_median": float(np.median(group_counts)),
        "geometry_mapped": int(feature_metadata["n_geometry_mapped"]),
        **tie,
    }


def _target_fingerprint(cfg: dict, candidates: Sequence[str]) -> dict:
    event_table = ROOT / cfg["cohort"]["accepted_strict_bb150_event_table"]
    cache_root = ROOT / cfg["cohort"]["accepted_bb150_cache_root"]
    caches = {}
    for subject in candidates:
        files = {}
        for suffix in (".json", ".npz"):
            path = cache_root / f"{subject}{suffix}"
            if not path.exists():
                raise FileNotFoundError(path)
            files[path.name] = _sha256(path)
        caches[subject] = files
    return {
        "accepted_event_table": str(event_table.relative_to(ROOT)),
        "accepted_event_table_sha256": _sha256(event_table),
        "accepted_bb150_cache_root": str(cache_root.relative_to(ROOT)),
        "per_subject_cache_sha256": caches,
        "target_values_loaded_by_dataset_builder": False,
        "purpose": "Phase-0 provenance lock only; Stage-A arrays contain no ictal value",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_interictal_operator_static_readout.yaml",
    )
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = yaml.safe_load(config_path.read_text())
    out_dir = (
        args.out_dir
        if args.out_dir is not None and args.out_dir.is_absolute()
        else ROOT / args.out_dir
        if args.out_dir is not None
        else ROOT / cfg["outputs"]["dataset"]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates, routing = _candidate_contract(cfg)
    indexed = _load_subject_index(cfg)
    if args.subjects:
        subjects = [str(x) for x in args.subjects]
    elif args.candidate_only:
        subjects = candidates
    else:
        subjects = indexed
    unknown = sorted(set(subjects) - set(indexed))
    if unknown:
        raise RuntimeError(f"subjects absent from masked index: {unknown}")

    rows = []
    for position, subject in enumerate(subjects, start=1):
        print(f"[phase0 {position}/{len(subjects)}] {subject}", flush=True)
        try:
            row = build_subject(
                subject,
                cfg,
                set(candidates),
                out_dir,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            row = {
                "dataset": subject.split("_", 1)[0],
                "subject": subject,
                "candidate_target_patient": subject in set(candidates),
                "status": "failed",
                "reason": f"{type(exc).__name__}:{exc}",
            }
        rows.append(row)
        print(
            f"  -> {row['status']} events={row.get('n_events', 0)} "
            f"reason={row.get('reason', '')}",
            flush=True,
        )

    audit = pd.DataFrame(rows).sort_values(["dataset", "subject"])
    audit.to_csv(out_dir / "subject_audit.csv", index=False)
    fingerprint = _target_fingerprint(cfg, candidates)
    (out_dir / "target_contract_fingerprint.json").write_text(
        json.dumps(fingerprint, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    candidate_rows = audit[audit.candidate_target_patient.astype(bool)]
    candidate_gate_pass = bool(np.all(candidate_rows.status.astype(str) == "ok"))
    full_pool_requested = set(subjects) == set(indexed)
    source_pool_complete = bool(
        full_pool_requested and np.all(audit.status.astype(str) == "ok")
    )
    manifest = {
        "contract": cfg["contract"]["name"],
        "contract_version": cfg["contract"]["version"],
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "source_spec": cfg["contract"]["source_spec"],
        "source_spec_sha256": _sha256(ROOT / cfg["contract"]["source_spec"]),
        "n_subjects_requested": len(subjects),
        "n_subjects_ok": int(np.sum(audit.status == "ok")),
        "n_subjects_failed": int(np.sum(audit.status != "ok")),
        "n_candidate_subjects": len(candidates),
        "n_candidate_strict_bb150_seizures": int(len(routing)),
        "n_events_ok": int(audit.loc[audit.status == "ok", "n_events"].sum()),
        "candidate_subjects": candidates,
        "dataset_balance": audit[audit.status == "ok"]
        .groupby("dataset")
        .size()
        .astype(int)
        .to_dict(),
        "split_contract": "chronological first 80% calibration, last 20% held out",
        "sampling_contract": "dataset-balanced then patient-balanced then event sampling",
        "recurrence_time": "within-one-interictal-event recruitment pseudo-time only",
        "target_values_read": False,
        "candidate_gate_pass": candidate_gate_pass,
        "full_source_pool_requested": full_pool_requested,
        "source_pool_complete": source_pool_complete,
        "phase0_pass": bool(candidate_gate_pass and source_pool_complete),
    }
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(_jsonable(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
