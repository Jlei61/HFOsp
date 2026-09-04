"""Leakage-safe human data adapter for the v0.3.4 S_P pilot."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.topic5_group_event_state.v033_training_lab.contact_grammar import DATASET_ROOT
from src.topic5_group_event_state.v033_training_lab.sg_o2 import (
    GrammarPairs,
    HUMAN_INPUT_ROOT,
    SGO2HumanData,
    SGO2TrainConfig,
    _feature_routes,
    _future_pairs,
    load_sg_o2_human_data,
)
from src.topic5_group_event_state.v033_training_lab.data import (
    robust_scale_apply,
    robust_scale_fit,
)
from src.topic5_group_event_state.v033_training_lab.paths import file_hash

from .contracts import EVALUATION_SUBJECTS, TUNING_SUBJECTS, TrainConfig, assert_safe_phases


# v0.3.5 broadens the already-built, prefix-only evaluation adapter to three
# explicitly registered exploratory patients.  Keep this list local instead of
# mutating the frozen v0.3.4 role registry: all three carry the same
# ``explicit_non_tuning_override`` manifest and the loader still stops at the
# registered 80% boundary.
V035_EXTENSION_SUBJECTS = (
    "epilepsiae_1096",
    "epilepsiae_384",
    "epilepsiae_1125",
)
V035_COHORT_EXPANSION_SUBJECTS = (
    "epilepsiae_1077", "epilepsiae_958", "yuquan_chengshuai",
    "yuquan_pengzihang", "yuquan_xuxinyi", "yuquan_zhangbichen",
    "yuquan_zhangjiaqi", "yuquan_zhangkexuan",
)
REGISTERED_PREFIX_SUBJECTS = (
    EVALUATION_SUBJECTS + V035_EXTENSION_SUBJECTS + V035_COHORT_EXPANSION_SUBJECTS
)


@dataclass(frozen=True)
class SpatialData:
    subject: str
    event_time: np.ndarray
    event_segment: np.ndarray
    event_token: np.ndarray
    train_event_mask: np.ndarray
    group_ids: np.ndarray
    group_count: np.ndarray
    participation: np.ndarray
    positive_extent: np.ndarray
    relative_lag: np.ndarray
    lag_valid: np.ndarray
    anchor_time: np.ndarray
    last_event_pos: np.ndarray
    phase: np.ndarray
    train_pairs: GrammarPairs
    selection_pairs: GrammarPairs
    provenance: Mapping[str, Any]
    anchor_truth: np.ndarray | None = None

    @property
    def n_contacts(self) -> int:
        return int(self.participation.shape[1])


def _source_rows(subject: str, event_time: np.ndarray, dataset_root: Path) -> tuple[Path, np.ndarray]:
    root = Path(dataset_root) / subject
    scalars = np.load(root / "scalars.npz")
    order = np.asarray(scalars["interictal_index"], dtype=np.int64)
    time = np.asarray(scalars["t_abs"], dtype=np.float64)[order]
    pos = np.searchsorted(time, event_time, side="left")
    if np.any(pos >= time.size) or not np.array_equal(time[pos], event_time):
        raise ValueError("v0.3.4 event prefix does not map exactly to source dataset")
    return root, order[pos]


def _contactwise_robust(values: np.ndarray, valid: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = np.where(valid & train[:, None], values, np.nan)
    centre = np.nanmedian(selected, axis=0)
    centre = np.where(np.isfinite(centre), centre, 0.0)
    mad = np.nanmedian(np.abs(selected - centre[None]), axis=0)
    scale = np.where(np.isfinite(mad) & (mad > 1e-7), 1.4826 * mad, 1.0)
    out = (values - centre[None]) / scale[None]
    out = np.where(valid, out, 0.0)
    return out.astype(np.float32), centre.astype(np.float32), scale.astype(np.float32)


def _burn_in_pairs(
    pairs: GrammarPairs,
    *,
    anchor_time: np.ndarray,
    anchor_segment: np.ndarray,
    event_time: np.ndarray,
    event_segment: np.ndarray,
    seconds: float,
) -> GrammarPairs:
    if seconds <= 0:
        return pairs
    segment_start = {
        int(seg): float(event_time[np.flatnonzero(event_segment == seg)[0]])
        for seg in np.unique(event_segment)
    }
    keep_local = np.array([
        anchor_time[row] - segment_start[int(anchor_segment[row])] >= float(seconds)
        for row in pairs.anchor_rows
    ], dtype=bool)
    kept = np.flatnonzero(keep_local)
    if kept.size == 0:
        raise ValueError("no anchors remain after the physical-time burn-in")
    remap = np.full(keep_local.size, -1, dtype=np.int64)
    remap[kept] = np.arange(kept.size)
    pair_keep = keep_local[pairs.pair_anchor]
    new_anchor = remap[pairs.pair_anchor[pair_keep]]
    # Rebuild equal-anchor/mean-event weights after filtering.
    counts = np.bincount(new_anchor, minlength=kept.size).astype(np.float64)
    weight = 1.0 / (kept.size * counts[new_anchor])
    return GrammarPairs(
        anchor_rows=pairs.anchor_rows[kept],
        pair_anchor=new_anchor,
        pair_event=pairs.pair_event[pair_keep],
        pair_weight=weight,
    ).validate()


def _load_evaluation_prefix_base(
    subject: str,
    *,
    input_root: Path,
    dataset_root: Path,
) -> SGO2HumanData:
    """Local v0.3.4 adapter for the four non-tuning evaluation prefixes.

    The v0.3.3 loader intentionally refuses these patients.  This duplicate is
    kept local so that the old tuning contract is not broadened retroactively.
    It indexes only events and targets before the registered 80% boundary.
    """

    if subject not in REGISTERED_PREFIX_SUBJECTS:
        raise PermissionError(f"not a locked S_P evaluation subject: {subject}")
    manifest_path = Path(input_root) / subject / "manifest_v3.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_path = Path(str(manifest.get("input_path", "")))
    if manifest.get("format") != "group_event_state_v0_3_3_human_r0_input_manifest" \
            or manifest.get("subject") != subject \
            or manifest.get("role") != "explicit_non_tuning_override" \
            or manifest.get("sealed") is not False \
            or manifest.get("development_evaluation_used_for_fitting") is not False:
        raise PermissionError("S_P evaluation manifest is not a locked non-tuning prefix")
    if not artifact_path.is_file() or file_hash(artifact_path) != manifest.get("input_npz_sha256"):
        raise ValueError("S_P evaluation input bytes differ from the locked manifest")
    with np.load(artifact_path, allow_pickle=False) as stored:
        metadata = json.loads(str(np.asarray(stored["metadata_json"]).item()))
        event_time_all = np.asarray(stored["event_time"], dtype=np.float64)
        event_carry_all = np.asarray(stored["event_carry"], dtype=np.int64)
        feature_all = np.asarray(stored["event_features_r0"], dtype=np.float64)
        valid_all = np.asarray(stored["event_feature_valid"], dtype=bool)
        train_event_all = np.asarray(stored["train_event_mask"], dtype=bool)
        anchor_time_all = np.asarray(stored["anchor_time"], dtype=np.float64)
        anchor_carry_all = np.asarray(stored["anchor_carry"], dtype=np.int64)
        last_all = np.asarray(stored["last_event_pos"], dtype=np.int64)
        phase_all = np.asarray(stored["phase"]).astype(str)
        eligible_all = np.asarray(stored["eligible_by_horizon"], dtype=bool)
        target_segment_all = np.asarray(stored["target_segment"], dtype=np.int64)
        target_bounds = np.asarray(stored["target_segment_bounds"], dtype=np.float64)
    if metadata.get("subject") != subject or metadata.get("sealed") is not False \
            or metadata.get("input_view") != "R0":
        raise ValueError("S_P evaluation R0 metadata identity differs")
    names = tuple(str(v) for v in metadata.get("event_feature_names_r0", []))
    if feature_all.shape != valid_all.shape or len(names) != feature_all.shape[1]:
        raise ValueError("S_P evaluation feature arrays do not align")

    selection_stop = float(manifest["report"]["phase_boundaries_epoch"]["80pct"])
    keep_events = event_time_all < selection_stop
    event_time = event_time_all[keep_events]
    event_carry = event_carry_all[keep_events]
    feature = feature_all[keep_events]
    valid = valid_all[keep_events]
    train_event = train_event_all[keep_events]
    if event_time.size == 0 or not np.any(train_event):
        raise ValueError("S_P evaluation has no permitted training events")

    allowed_anchor = np.isin(phase_all, ("STATE_TRAIN", "STATE_SELECTION")) \
        & eligible_all.all(axis=1)
    if np.any(allowed_anchor & (anchor_time_all >= selection_stop)):
        raise PermissionError("S_P evaluation anchor crosses the 80% boundary")
    anchor_source = np.flatnonzero(allowed_anchor)
    assert_safe_phases(phase_all[anchor_source])
    anchor_time = anchor_time_all[anchor_source]
    anchor_carry = anchor_carry_all[anchor_source]
    last_event_pos = last_all[anchor_source]
    phase = phase_all[anchor_source]
    anchor_target = target_segment_all[anchor_source]
    if np.any(last_event_pos >= event_time.size):
        raise PermissionError("S_P evaluation history reaches a development event")

    subject_root = Path(dataset_root) / subject
    index = json.loads((subject_root / "index.json").read_text(encoding="utf-8"))
    scalars = np.load(subject_root / "scalars.npz")
    raw_order = np.asarray(scalars["interictal_index"], dtype=np.int64)
    raw_time = np.asarray(scalars["t_abs"], dtype=np.float64)[raw_order]
    stream_index = np.searchsorted(raw_time, event_time, side="left")
    if np.any(stream_index >= raw_time.size) \
            or not np.array_equal(raw_time[stream_index], event_time):
        raise ValueError("S_P evaluation prefix does not map to source stream")
    raw_rows = raw_order[stream_index]
    group_meta = index["arrays"]["tied_group_id"]
    source_groups = np.load(subject_root / group_meta["file"], mmap_mode="r")
    group_ids = np.asarray(source_groups[raw_rows], dtype=np.int64)
    group_count = np.maximum(group_ids.max(1) + 1, 0).astype(np.int64)

    event_target = np.full(event_time.size, -1, dtype=np.int64)
    for segment, (left, right) in enumerate(target_bounds):
        inside = (event_time >= left) & (event_time < right)
        if np.any(event_target[inside] >= 0):
            raise ValueError("S_P evaluation target segments overlap")
        event_target[inside] = int(segment)
    raw = np.where(valid, feature, np.nan)
    scaler = robust_scale_fit(raw, train_event)
    x_scaled = robust_scale_apply(raw, scaler)
    x_scaled[~valid] = 0.0
    train_rows = np.flatnonzero(phase == "STATE_TRAIN")
    selection_rows = np.flatnonzero(phase == "STATE_SELECTION")
    train_pairs = _future_pairs(
        train_rows, anchor_time=anchor_time, anchor_target=anchor_target,
        event_time=event_time, event_target=event_target, horizon=1800.0,
    )
    inner_pairs = _future_pairs(
        selection_rows, anchor_time=anchor_time, anchor_target=anchor_target,
        event_time=event_time, event_target=event_target, horizon=1800.0,
    )
    if float(anchor_time[train_pairs.anchor_rows].max()) \
            >= float(anchor_time[inner_pairs.anchor_rows].min()):
        raise PermissionError("S_P evaluation selection is not after training")
    mark_columns, scaffold_columns = _feature_routes(names)
    return SGO2HumanData(
        subject=subject, event_time=event_time, event_carry=event_carry,
        x_scaled=np.ascontiguousarray(x_scaled), train_event_mask=train_event,
        group_ids=group_ids, group_count=group_count,
        anchor_time=anchor_time, anchor_carry=anchor_carry,
        last_event_pos=last_event_pos, phase=phase,
        train_pairs=train_pairs, inner_pairs=inner_pairs, feature_names=names,
        mark_columns=mark_columns, scaffold_columns=scaffold_columns,
        provenance={
            "human_manifest": str(manifest_path),
            "human_manifest_sha256": file_hash(manifest_path),
            "human_input": str(artifact_path),
            "human_input_sha256": file_hash(artifact_path),
            "human_content_hash": str(manifest["input_artifact_hash"]),
            "split_hash": str(manifest["split_hash"]),
            "role": "locked_recipe_evaluation",
            "normalization": "robust; fit on CALIBRATION_plus_STATE_TRAIN event mask only",
            "input_stop": "80pct boundary; no DEVELOPMENT_EVALUATION events used",
            "target_phases": ["STATE_TRAIN", "STATE_SELECTION"],
            "development_targets_read": False,
            "development_targets_exposed": False,
            "sealed_partition_opened": False,
            "seizure_outcomes_read": False,
        },
    )


def load_human_spatial_data(
    subject: str,
    *,
    train_config: TrainConfig,
    input_root: Path = HUMAN_INPUT_ROOT,
    dataset_root: Path = DATASET_ROOT,
) -> SpatialData:
    """Load only the permitted prefix; never index development/seizure targets."""

    train_config.validate()
    if subject in TUNING_SUBJECTS:
        legacy_cfg = SGO2TrainConfig(run_kind="full_training", max_steps=1)
        base = load_sg_o2_human_data(
            subject, train_cfg=legacy_cfg, input_root=input_root, dataset_root=dataset_root
        )
    elif subject in REGISTERED_PREFIX_SUBJECTS:
        base = _load_evaluation_prefix_base(
            subject, input_root=input_root, dataset_root=dataset_root,
        )
    else:
        raise PermissionError(f"subject is not registered for v0.3.4 S_P: {subject}")
    assert_safe_phases(base.phase)
    root, rows = _source_rows(subject, base.event_time, dataset_root)
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))

    def load_array(name: str) -> np.ndarray:
        meta = index["arrays"][name]
        return np.asarray(np.load(root / meta["file"], mmap_mode="r")[rows])

    participation = load_array("participation").astype(bool)
    contact_ok = load_array("contact_ok").astype(bool)
    group_ids = load_array("tied_group_id").astype(np.int64)
    relative_lag = load_array("relative_delay").astype(np.float32)
    if not np.array_equal(participation, group_ids >= 0):
        raise ValueError("participation and tied-group membership differ")
    lag_valid = participation & contact_ok & np.isfinite(relative_lag)
    lag_z, lag_centre, lag_scale = _contactwise_robust(
        relative_lag, lag_valid, base.train_event_mask
    )
    group_count = np.maximum(group_ids.max(1) + 1, 0).astype(np.int64)
    positive_extent = participation.sum(1).astype(np.float32)
    denom = np.maximum(group_count[:, None] - 1, 1)
    group_position = np.where(participation, group_ids / denom, 0.0).astype(np.float32)
    # No legacy_rank enters this token.  This is important because the source
    # rank arrays contain finite phantom ranks for nonparticipants.
    token = np.concatenate([
        base.x_scaled.astype(np.float32),
        participation.astype(np.float32),
        group_position,
        lag_z,
    ], axis=1)
    anchor_segment = base.anchor_carry
    train_pairs = _burn_in_pairs(
        base.train_pairs,
        anchor_time=base.anchor_time,
        anchor_segment=anchor_segment,
        event_time=base.event_time,
        event_segment=base.event_carry,
        seconds=train_config.burn_in_seconds,
    )
    selection_pairs = _burn_in_pairs(
        base.inner_pairs,
        anchor_time=base.anchor_time,
        anchor_segment=anchor_segment,
        event_time=base.event_time,
        event_segment=base.event_carry,
        seconds=train_config.burn_in_seconds,
    )
    return SpatialData(
        subject=subject,
        event_time=base.event_time,
        event_segment=base.event_carry,
        event_token=np.ascontiguousarray(token),
        train_event_mask=base.train_event_mask,
        group_ids=group_ids,
        group_count=group_count,
        participation=participation,
        positive_extent=positive_extent,
        relative_lag=np.where(lag_valid, relative_lag, 0.0).astype(np.float32),
        lag_valid=lag_valid,
        anchor_time=base.anchor_time,
        last_event_pos=base.last_event_pos,
        phase=base.phase,
        train_pairs=train_pairs,
        selection_pairs=selection_pairs,
        provenance={
            **dict(base.provenance),
            "spatial_arrays": {
                name: index["arrays"][name]["file"]
                for name in ("participation", "tied_group_id", "relative_delay", "contact_ok")
            },
            "event_token": "R0 + masked participation + tied-group position + exact relative lag",
            "legacy_rank_used": False,
            "lag_scaling_fit": "STATE_TRAIN participating valid contacts only",
            "lag_centre": lag_centre.tolist(),
            "lag_scale": lag_scale.tolist(),
            "burn_in_seconds": train_config.burn_in_seconds,
            "future_horizon_seconds": 1800.0,
            "development_targets_read": False,
            "sealed_partition_opened": False,
            "seizure_outcomes_read": False,
        },
    )


def sample_equal_anchor_pairs(
    pairs: GrammarPairs,
    *,
    rng: np.random.Generator,
    n_anchors: int,
    events_per_anchor: int,
) -> GrammarPairs:
    """Sample anchors, then events within anchors, without event-rate weighting."""

    n_local = int(pairs.anchor_rows.size)
    chosen = rng.choice(n_local, size=min(int(n_anchors), n_local), replace=False)
    pair_anchor: list[int] = []
    pair_event: list[int] = []
    for new_local, old_local in enumerate(chosen):
        available = np.flatnonzero(pairs.pair_anchor == int(old_local))
        if available.size > int(events_per_anchor):
            available = rng.choice(available, size=int(events_per_anchor), replace=False)
        pair_anchor.extend([new_local] * int(available.size))
        pair_event.extend(pairs.pair_event[available].tolist())
    pa = np.asarray(pair_anchor, dtype=np.int64)
    counts = np.bincount(pa, minlength=chosen.size).astype(np.float64)
    weight = 1.0 / (chosen.size * counts[pa])
    return GrammarPairs(
        anchor_rows=pairs.anchor_rows[chosen],
        pair_anchor=pa,
        pair_event=np.asarray(pair_event, dtype=np.int64),
        pair_weight=weight,
    ).validate()
