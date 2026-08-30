"""E384 development-instrument integration for H2b.

This module joins the independently audited Phase-0 inventory, causal state
cache, and conditional risk-set probe.  It never trains or updates the frozen
state model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
# Load cuda_env's compatible C++ runtime before pandas native extensions.
import torch as _torch  # noqa: F401
import pandas as pd

from src.topic5_continuous_marked_state_r1.coverage import CoverageTable
from src.topic5_continuous_marked_state_r1.history import (
    BASE_HISTORY_NAMES,
    history_names,
)
from src.topic5_continuous_marked_state_r1.raw_observation import EXPLICIT_NAMES

from .contract import (
    LEAD_MINUTES,
    POSTICTAL_GUARD_MINUTES,
    RESULT_ROOT,
    atomic_csv,
    atomic_json,
    sha256_file,
    utc_now,
)
from .risk_probe import build_risk_sets
from .state_extraction import (
    InferenceRawAnchorReader,
    load_frozen_design,
)


PILOT_REVISION = "h2b_e384_instrument_integration_v0_1"
E384_SUBJECT = "epilepsiae_384"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _segment_for_time(coverage: CoverageTable, value: float) -> int | None:
    hit = np.flatnonzero(
        (coverage.start <= float(value)) & (float(value) < coverage.stop)
    )
    return int(hit[0]) if len(hit) == 1 else None


def _mask_signature(mask: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
    return hashlib.sha256(value.tobytes()).hexdigest()


def _outside_intervals(value: float,
                       intervals: Iterable[tuple[float, float]]) -> bool:
    return all(not (float(left) <= float(value) <= float(right))
               for left, right in intervals)


def prepare_e384_query_inputs(
    *, source_repo_root: Path,
    result_root: Path = RESULT_ROOT,
) -> dict[str, Any]:
    """Freeze exact case queries, candidate-control anchors, and exclusions."""
    source_repo = Path(source_repo_root).resolve()
    root = Path(result_root).resolve()
    manifest_root = root / "manifests"
    funnel_path = manifest_root / "exclusion_funnel.json"
    crosswalk_path = manifest_root / "seizure_crosswalk.csv"
    funnel = _json(funnel_path)
    crosswalk = pd.read_csv(crosswalk_path)
    design, design_manifest, design_manifest_path = load_frozen_design(
        source_repo, E384_SUBJECT
    )
    coverage_path = (
        source_repo
        / "results/epi_prssm/continuous_marked_state/r1/r1_2/coverage"
        / f"{E384_SUBJECT}.npz"
    )
    coverage = CoverageTable.load(coverage_path)

    detail = funnel.get("per_seizure_support") or []
    primary = [
        row for row in detail
        if int(row["lead_minutes"]) == 30 and row.get("eligible") is True
    ]
    primary_ids = {str(row["seizure_id"]) for row in primary}
    eligible_support = [row for row in detail if row.get("eligible") is True]
    supported_ids = {str(row["seizure_id"]) for row in eligible_support}
    if not primary_ids:
        raise ValueError("E384 has no primary-lead development case for the pilot")
    seizure_lookup = {
        str(row.canonical_seizure_id): row
        for row in crosswalk.itertuples(index=False)
        if str(row.canonical_seizure_id) in supported_ids
    }
    if set(seizure_lookup) != supported_ids:
        raise ValueError("eligible support IDs do not map uniquely to the crosswalk")
    case_segments = sorted({
        int(row["coverage_segment"]) for row in eligible_support
    })

    reader = InferenceRawAnchorReader(E384_SUBJECT, design.event_time)
    grid_time, grid_segment, grid_session, _, _ = reader.inference_anchor_inventory(
        coverage, allowed_segments=case_segments,
    )
    query_by_key: dict[tuple[int, float], dict[str, Any]] = {}
    for time, segment, session in zip(grid_time, grid_segment, grid_session):
        query_by_key[(int(segment), float(time))] = {
            "anchor_time_epoch": np.float64(time),
            "coverage_segment_index": int(segment),
            "continuity_session": int(session),
            "query_role": "control_candidate",
            "case_seizure_id": "",
            "case_lead_minutes": "",
            "exclusion_start_epoch": np.nan,
            "exclusion_stop_epoch": np.nan,
        }

    case_rows = [
        row for row in eligible_support
        if int(row["lead_minutes"]) in LEAD_MINUTES
    ]
    for support in case_rows:
        seizure_id = str(support["seizure_id"])
        segment = int(support["coverage_segment"])
        cutoff = float(support["cutoff_epoch"])
        onset = float(support["onset_epoch"])
        lead = int(support["lead_minutes"])
        if _segment_for_time(coverage, cutoff) != segment:
            raise ValueError("case support segment drifted from frozen coverage")
        source = seizure_lookup[seizure_id]
        offset = float(source.offset_epoch) if pd.notna(source.offset_epoch) else onset
        key = (segment, cutoff)
        existing = query_by_key.get(key, {})
        query_by_key[key] = {
            **existing,
            "anchor_time_epoch": np.float64(cutoff),
            "coverage_segment_index": segment,
            "continuity_session": int(coverage.session[segment]),
            "query_role": (
                "case_and_control_grid" if existing else "case"
            ),
            "case_seizure_id": seizure_id,
            "case_lead_minutes": lead,
            # A wrong-time donor may be earlier in the same segment, but may
            # not overlap this lead-specific target interval or postictal guard.
            "exclusion_start_epoch": np.float64(onset - lead * 60.0),
            "exclusion_stop_epoch": np.float64(
                offset + POSTICTAL_GUARD_MINUTES * 60.0
            ),
        }

    query_rows = sorted(
        query_by_key.values(),
        key=lambda row: (float(row["anchor_time_epoch"]),
                         int(row["coverage_segment_index"])),
    )
    for index, row in enumerate(query_rows):
        row["query_id"] = f"e384_q{index:06d}"
    query_path = root / "risk_sets/e384_state_queries.csv"
    atomic_csv(query_path, query_rows)

    global_rows = []
    for row in crosswalk.itertuples(index=False):
        onset = float(row.onset_epoch)
        offset = float(row.offset_epoch) if pd.notna(row.offset_epoch) else onset
        global_rows.append({
            "seizure_id": str(row.canonical_seizure_id),
            "interval_start_epoch": np.float64(onset),
            "interval_stop_epoch": np.float64(
                offset + POSTICTAL_GUARD_MINUTES * 60.0
            ),
            "interval_role": "ictal_plus_postictal_exclusion",
        })
    exclusion_path = root / "risk_sets/e384_global_exclusions.csv"
    atomic_csv(exclusion_path, global_rows)

    seizures = []
    representative_support = {}
    for support in eligible_support:
        representative_support.setdefault(str(support["seizure_id"]), support)
    for seizure_id in sorted(supported_ids):
        support = representative_support[seizure_id]
        source = seizure_lookup[seizure_id]
        seizures.append({
            "patient_id": E384_SUBJECT,
            "seizure_id": seizure_id,
            "onset_time": np.float64(source.onset_epoch),
            "segment_id": str(int(support["coverage_segment"])),
            "primary_30min_supported": bool(seizure_id in primary_ids),
        })
    seizure_path = root / "risk_sets/e384_primary_seizures.csv"
    atomic_csv(seizure_path, seizures)

    payload = {
        "status": "COMPLETE",
        "pilot_revision": PILOT_REVISION,
        "created_utc": utc_now(),
        "subject": E384_SUBJECT,
        "primary_seizure_ids": sorted(primary_ids),
        "primary_seizure_count": len(primary_ids),
        "all_lead_supported_seizure_ids": sorted(supported_ids),
        "all_lead_supported_seizure_count": len(supported_ids),
        "case_query_count": len(case_rows),
        "candidate_query_count": len(query_rows),
        "case_coverage_segment_indices": case_segments,
        "coverage_segment_semantics": "unique_coverage_table_row_index",
        "continuity_session_semantics": "history_only",
        "query_path": str(query_path),
        "query_sha256": sha256_file(query_path),
        "global_exclusion_path": str(exclusion_path),
        "global_exclusion_sha256": sha256_file(exclusion_path),
        "seizure_path": str(seizure_path),
        "seizure_sha256": sha256_file(seizure_path),
        "inventory_funnel": str(funnel_path),
        "inventory_funnel_sha256": sha256_file(funnel_path),
        "crosswalk": str(crosswalk_path),
        "crosswalk_sha256": sha256_file(crosswalk_path),
        "frozen_design_manifest": str(design_manifest_path),
        "frozen_design_manifest_sha256": sha256_file(design_manifest_path),
        "frozen_design_sha256": design_manifest["design_sha256"],
        "training_guard_free_used_for_inference": False,
        "seizure_label_used_in_state_update": False,
    }
    output = manifest_root / "e384_pilot_inputs.json"
    atomic_json(output, payload)
    return payload


def state_cache_to_anchor_frame(
    *, cache_path: Path,
    query_path: Path,
    coverage: CoverageTable,
    global_exclusion_path: Path,
    seed: int,
    patient_id: str = E384_SUBJECT,
) -> pd.DataFrame:
    """Convert one frozen state cache to the arm-neutral anchor table."""
    cache_path = Path(cache_path).resolve()
    manifest_path = cache_path.with_suffix(".manifest.json")
    manifest = _json(manifest_path)
    if sha256_file(cache_path) != manifest.get("cache_sha256"):
        raise ValueError("state cache hash does not match its manifest")
    if manifest.get("all_current_observations_fresh") is not True:
        raise ValueError("state cache contains a stale current observation")
    query = pd.read_csv(query_path)
    exclusions = pd.read_csv(global_exclusion_path)
    global_intervals = [
        (float(row.interval_start_epoch), float(row.interval_stop_epoch))
        for row in exclusions.itertuples(index=False)
    ]
    with np.load(cache_path, allow_pickle=False) as data:
        query_id = data["query_id"].astype(str)
        if not np.array_equal(query_id, query.query_id.astype(str).to_numpy()):
            raise ValueError("state cache/query IDs disagree")
        anchor_time = data["anchor_time_epoch"].astype(np.float64)
        segment = data["coverage_segment_index"].astype(np.int64)
        persistent = data["persistent_state"].astype(np.float64)
        memoryless = data["memoryless_observation_code"].astype(np.float64)
        observation = data["current_explicit_summary"].astype(np.float64)
        history = data["deterministic_history"].astype(np.float64)
        mask = data["current_contact_mask"].astype(bool)
        available = data["observation_available"].astype(bool)
        age = data["observation_age_seconds"].astype(np.float64)
        donor_time = data["wrong_time_donor_time_epoch"].astype(np.float64)
        donor_state = data["wrong_time_donor_state"].astype(np.float64)
        donor_valid = data["wrong_time_valid"].astype(bool)

    history_width = int(history.shape[1])
    mark_width = history_width - len(BASE_HISTORY_NAMES)
    if mark_width < 0 or mark_width % 3:
        raise ValueError("deterministic history width cannot encode the frozen mark set")
    resolved_history_names = history_names(mark_width // 3)
    if list(manifest.get("deterministic_history_names", [])) != list(
        resolved_history_names
    ):
        raise ValueError("state manifest deterministic-history names/width drifted")
    if observation.shape[1] != 2 * len(EXPLICIT_NAMES):
        raise ValueError("explicit observation summary width drifted")
    rows: list[dict[str, Any]] = []
    for index, source in query.iterrows():
        valid_ids = np.flatnonzero(donor_valid[index])
        wrong_valid = bool(len(valid_ids))
        chosen = int(valid_ids[0]) if wrong_valid else -1
        wrong_time = float(donor_time[index, chosen]) if wrong_valid else np.nan
        wrong_segment = _segment_for_time(coverage, wrong_time) if wrong_valid else None
        target_intervals = list(global_intervals)
        if pd.notna(source.exclusion_start_epoch) and pd.notna(source.exclusion_stop_epoch):
            target_intervals.append((
                float(source.exclusion_start_epoch),
                float(source.exclusion_stop_epoch),
            ))
        row: dict[str, Any] = {
            "patient_id": str(patient_id),
            "seed": int(seed),
            "anchor_id": str(source.query_id),
            "anchor_time": np.float64(anchor_time[index]),
            "segment_id": str(int(segment[index])),
            "segment_start": np.float64(coverage.start[int(segment[index])]),
            "segment_end": np.float64(coverage.stop[int(segment[index])]),
            "observation_available": bool(available[index] and age[index] <= 30.0),
            # The frozen contract requires equally available current
            # observations, not an identical 80-contact artifact mask.  Keep
            # the exact mask hash for audit without turning it into a hidden
            # case/control matching gate.
            "observation_signature": "fresh_current_observation_le_30s",
            "contact_mask_signature": _mask_signature(mask[index]),
            # Controls come from an event-independent observation grid.  They
            # therefore need an explicit frozen seizure/postictal screen;
            # hard-coding False would silently admit postictal controls.
            "in_ictal_or_postictal": bool(
                not _outside_intervals(float(anchor_time[index]), global_intervals)
            ),
            "wrong_time_donor_valid": wrong_valid,
            "wrong_time_same_segment": bool(
                wrong_valid and wrong_segment == int(segment[index])
            ),
            "wrong_time_exclusion_clear": bool(
                wrong_valid and _outside_intervals(wrong_time, target_intervals)
            ),
            "wrong_time_donor_time": np.float64(wrong_time),
            "current_observation_age_seconds": np.float64(age[index]),
            "query_role": str(source.query_role),
            "case_seizure_id": str(source.case_seizure_id),
            "case_lead_minutes": source.case_lead_minutes,
        }
        row.update({
            f"history__{name}": float(history[index, position])
            for position, name in enumerate(resolved_history_names)
        })
        row.update({
            f"observation__mean_{name}": float(observation[index, position])
            for position, name in enumerate(EXPLICIT_NAMES)
        })
        row.update({
            f"observation__sd_{name}": float(
                observation[index, len(EXPLICIT_NAMES) + position]
            )
            for position, name in enumerate(EXPLICIT_NAMES)
        })
        row.update({
            f"state__z{position}": float(persistent[index, position])
            for position in range(persistent.shape[1])
        })
        row.update({
            f"memoryless__z{position}": float(memoryless[index, position])
            for position in range(memoryless.shape[1])
        })
        if wrong_valid:
            row.update({
                f"wrong_time__z{position}": float(donor_state[index, chosen, position])
                for position in range(donor_state.shape[2])
            })
        else:
            row.update({
                f"wrong_time__z{position}": np.nan
                for position in range(donor_state.shape[2])
            })
        rows.append(row)
    return pd.DataFrame(rows)


def build_cohort_risk_table(
    *, anchor_frames: Iterable[pd.DataFrame],
    seizure_path: Path,
    output_path: Path,
    controls_per_case: int = 5,
    arms: tuple[str, ...] = (
        "B_history", "B_observation", "B_state", "memoryless",
    ),
    require_wrong_time: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    anchors = pd.concat(list(anchor_frames), ignore_index=True)
    eligible = anchors["observation_available"].astype(bool)
    if require_wrong_time:
        eligible &= (
            anchors["wrong_time_donor_valid"].astype(bool)
            & anchors["wrong_time_same_segment"].astype(bool)
            & anchors["wrong_time_exclusion_clear"].astype(bool)
        )
    anchors = anchors.loc[eligible].copy()
    seizures = pd.read_csv(
        seizure_path, dtype={
            "patient_id": str, "seizure_id": str, "segment_id": str,
        },
    )
    if "primary_30min_supported" in seizures:
        values = seizures["primary_30min_supported"]
        if not pd.api.types.is_bool_dtype(values):
            lowered = values.astype(str).str.strip().str.lower()
            if not set(lowered.unique()).issubset({"true", "false"}):
                raise ValueError("primary_30min_supported must be strict boolean")
            seizures["primary_30min_supported"] = lowered.map({
                "true": True, "false": False,
            }).astype(bool)
    seizures["onset_time"] = seizures["onset_time"].astype(np.float64)
    risk, audit = build_risk_sets(
        anchors, seizures,
        lead_minutes=LEAD_MINUTES,
        controls_per_case=int(controls_per_case),
        random_seed=1729,
        arms=arms,
        require_wrong_time=require_wrong_time,
    )
    if risk.empty:
        raise ValueError("H2b cohort input produced no estimable risk set")
    atomic_csv(output_path, risk.replace({np.nan: None}).to_dict(orient="records"))
    audit = {
        **audit,
        "integration_revision": "h2b_cross_task_risk_table_v0_2",
        "risk_table": str(output_path),
        "risk_table_sha256": sha256_file(output_path),
        "n_anchor_rows_before_contract_filter": int(len(eligible)),
        "n_anchor_rows_after_contract_filter": int(eligible.sum()),
        "all_seeds_share_candidate_anchor_ids": all(
            set(group.anchor_id.astype(str)) == set(
                anchors[anchors.seed == anchors.seed.min()].anchor_id.astype(str)
            )
            for _, group in anchors.groupby("seed")
        ),
        "wrong_time_required_for_entry": bool(require_wrong_time),
        "arms": list(arms),
        "source_sha256": {
            "src/topic5_continuous_marked_state_h2b/pilot.py": sha256_file(
                Path(__file__).resolve()
            ),
            "src/topic5_continuous_marked_state_h2b/risk_probe.py": sha256_file(
                Path(__file__).resolve().with_name("risk_probe.py")
            ),
        },
    }
    atomic_json(output_path.with_suffix(".manifest.json"), audit)
    return risk, audit


def build_e384_risk_table(
    *, anchor_frames: Iterable[pd.DataFrame],
    seizure_path: Path,
    output_path: Path,
    controls_per_case: int = 5,
    arms: tuple[str, ...] = (
        "B_history", "B_observation", "B_state", "memoryless",
    ),
    require_wrong_time: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Backward-compatible v0.1 wrapper around the cohort-neutral builder."""
    risk, audit = build_cohort_risk_table(
        anchor_frames=anchor_frames,
        seizure_path=seizure_path,
        output_path=output_path,
        controls_per_case=controls_per_case,
        arms=arms,
        require_wrong_time=require_wrong_time,
    )
    audit = {**audit, "pilot_revision": PILOT_REVISION}
    atomic_json(output_path.with_suffix(".manifest.json"), audit)
    return risk, audit
