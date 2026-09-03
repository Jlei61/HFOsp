"""Fail-closed loader for Agent C's immutable human R0 training input.

The loader deliberately does not import Agent C's worktree.  It validates the
published bytes and independently reconstructs strict history and future
count targets.  State carry follows continuous recorded sessions; targets and
bootstrap blocks follow seizure-cut coverage segments.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .data import DataView, robust_scale_apply, robust_scale_fit


FORMAT = "group_event_state_v0_3_3_materialized_human_r0_input"
BIN_CONVENTION = "left_closed_right_open_[t+a,t+b)"
TRAIN_PHASE = "STATE_TRAIN"
INNER_VAL_PHASE = "STATE_SELECTION"
DEVELOPMENT_PHASE = "DEVELOPMENT_EVALUATION"


class HumanArtifactHeld(PermissionError):
    """The immutable human artifact or canonical evaluator is not released."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_last_event(
    event_time: np.ndarray,
    event_carry: np.ndarray,
    anchor_time: np.ndarray,
    anchor_carry: np.ndarray,
) -> np.ndarray:
    out = np.full(anchor_time.size, -1, dtype=np.int64)
    for carry in np.unique(anchor_carry):
        events = np.flatnonzero(event_carry == carry)
        anchors = np.flatnonzero(anchor_carry == carry)
        if events.size == 0:
            continue
        local = np.searchsorted(event_time[events], anchor_time[anchors], side="left") - 1
        ok = local >= 0
        out[anchors[ok]] = events[local[ok]]
    return out


def _target_counts(
    event_time: np.ndarray,
    event_target: np.ndarray,
    anchor_time: np.ndarray,
    anchor_target: np.ndarray,
    bins: Sequence[Sequence[float]],
) -> np.ndarray:
    out = np.zeros((anchor_time.size, len(bins)), dtype=np.int64)
    for row, (time, target) in enumerate(zip(anchor_time, anchor_target)):
        same = event_target == target
        for column, (left, right) in enumerate(bins):
            out[row, column] = int(np.sum(
                same & (event_time >= time + float(left)) & (event_time < time + float(right))
            ))
    return out


def load_materialized_human_r0_view(
    request: Mapping[str, Any],
    *,
    bins: tuple[tuple[float, float], ...],
    scaling: str,
) -> tuple[DataView, dict[str, Any]]:
    """Load only STATE_TRAIN and STATE_SELECTION; never expose development targets."""

    iv = dict(request["input_view"])
    subject = str(iv.get("subject", ""))
    artifact_path = Path(str(iv.get("artifact_path", "")))
    manifest_path = Path(str(iv.get("artifact_manifest", "")))
    evaluator_path = Path(str(iv.get("evaluator_contract", "")))
    release_path = Path(str(iv.get("canonical_evaluator_release", "")))
    for label, path in (
        ("human R0 artifact", artifact_path), ("human R0 manifest", manifest_path),
        ("canonical evaluator", evaluator_path), ("canonical evaluator release", release_path),
    ):
        if not path.is_file():
            raise HumanArtifactHeld(f"{label} is missing")

    manifest_hash = _sha256_file(manifest_path)
    if manifest_hash != str(iv.get("artifact_manifest_sha256", "")):
        raise ValueError("human artifact manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    role = str(manifest.get("role", ""))
    expansion = dict(request.get("human_trainability_experiment") or {})
    expansion_authorized = expansion.get("cohort_expansion_authorized") is True \
        and expansion.get("scientific_interpretation") == "optimization_diagnostic_only"
    role_allowed = role == "tuning" or (
        role == "explicit_non_tuning_override" and expansion_authorized
    )
    if manifest.get("format") != "group_event_state_v0_3_3_human_r0_input_manifest" \
            or manifest.get("subject") != subject or not role_allowed:
        raise HumanArtifactHeld(
            "human manifest is neither a locked tuning artifact nor an explicitly "
            "authorized trainability-expansion artifact"
        )
    if manifest.get("sealed") is not False \
            or manifest.get("development_evaluation_used_for_fitting") is not False:
        raise PermissionError("human manifest is sealed or used development evaluation for fitting")

    evaluator_hash = _sha256_file(evaluator_path)
    release_hash = _sha256_file(release_path)
    if evaluator_hash != str(iv.get("evaluator_contract_sha256", "")) \
            or evaluator_hash != str(manifest.get("evaluator_contract_sha256", "")):
        raise ValueError("human artifact evaluator identity mismatch")
    if release_hash != str(iv.get("canonical_evaluator_release_sha256", "")):
        raise ValueError("canonical evaluator release SHA256 mismatch")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("status") != "RELEASED_DEVELOPMENT_ONLY" \
            or release.get("canonical_sha256") != evaluator_hash \
            or release.get("sealed_partition_opened") is not False:
        raise HumanArtifactHeld("canonical evaluator is not released for development")

    artifact_hash = _sha256_file(artifact_path)
    if artifact_hash != str(iv.get("artifact_sha256", "")) \
            or artifact_hash != str(request.get("input_hash", "")) \
            or artifact_hash != str(manifest.get("input_npz_sha256", "")):
        raise ValueError("human R0 NPZ byte hash mismatch")
    if Path(str(manifest.get("input_path", ""))).resolve() != artifact_path.resolve():
        raise ValueError("manifest input path differs from requested human R0 artifact")

    h_path = Path(str(manifest.get("h_mark_path", "")))
    if not h_path.is_file() or _sha256_file(h_path) != str(manifest.get("h_mark_npz_sha256", "")):
        raise HumanArtifactHeld("bound H_mark artifact is missing or changed")
    with np.load(h_path, allow_pickle=False) as stored_h:
        h_content_hash = str(np.asarray(stored_h["artifact_hash"]).item())
    baseline = dict(request.get("baseline_H") or {})
    if baseline.get("name") != "H_mark" \
            or baseline.get("hash") != h_content_hash \
            or h_content_hash != str(manifest.get("h_mark_artifact_hash", "")):
        raise ValueError("human H_mark content identity mismatch")

    with np.load(artifact_path, allow_pickle=False) as stored:
        metadata = json.loads(str(np.asarray(stored["metadata_json"]).item()))
        content_hash = str(np.asarray(stored["artifact_hash"]).item())
        arrays = {name: np.asarray(stored[name]) for name in stored.files
                  if name not in {"metadata_json", "artifact_hash"}}
    if content_hash != str(iv.get("artifact_content_sha256", "")) \
            or content_hash != str(manifest.get("input_artifact_hash", "")):
        raise ValueError("human R0 content hash mismatch")
    if metadata.get("format") != FORMAT or int(metadata.get("schema_version", -1)) != 1 \
            or metadata.get("subject") != subject or metadata.get("sealed") is not False \
            or metadata.get("bin_convention") != BIN_CONVENTION:
        raise ValueError("unexpected materialized human R0 schema")
    if metadata.get("science_code_commit") != request.get("science_code_commit") \
            or manifest.get("science_code_commit") != request.get("science_code_commit"):
        raise ValueError("human science-code identity mismatch")
    if metadata.get("data_registry_key") != iv.get("data_registry_key") \
            or metadata.get("data_registry_key") != manifest.get("data_registry_key"):
        raise ValueError("human data-registry identity mismatch")
    if metadata.get("split_hash") != request.get("split_hash") \
            or metadata.get("split_hash") != manifest.get("split_hash"):
        raise ValueError("human split identity mismatch")
    if metadata.get("h_mark_artifact_hash") != h_content_hash:
        raise ValueError("human input is not bound to the requested H_mark")

    required = {
        "anchor_id", "anchor_time", "phase", "anchor_carry", "target_segment",
        "carry_bounds", "target_segment_bounds", "last_event_pos", "eligible_by_horizon",
        "target_counts", "log_mu_h_mark", "nb_log_r_train_frozen", "event_id", "event_time",
        "event_carry", "event_features_r0", "event_feature_valid", "train_event_mask",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"materialized human arrays missing: {missing}")

    anchor_id = np.asarray(arrays["anchor_id"]).astype(str)
    anchor_time = np.asarray(arrays["anchor_time"], dtype=np.float64)
    phase = np.asarray(arrays["phase"]).astype(str)
    anchor_carry = np.asarray(arrays["anchor_carry"], dtype=np.int64)
    target_segment = np.asarray(arrays["target_segment"], dtype=np.int64)
    carry_bounds = np.asarray(arrays["carry_bounds"], dtype=np.float64)
    target_bounds = np.asarray(arrays["target_segment_bounds"], dtype=np.float64)
    last = np.asarray(arrays["last_event_pos"], dtype=np.int64)
    eligible = np.asarray(arrays["eligible_by_horizon"], dtype=bool)
    counts_all = np.asarray(arrays["target_counts"])
    log_mu_all = np.asarray(arrays["log_mu_h_mark"], dtype=np.float64)
    log_r = np.asarray(arrays["nb_log_r_train_frozen"], dtype=np.float64)
    event_id = np.asarray(arrays["event_id"]).astype(str)
    event_time = np.asarray(arrays["event_time"], dtype=np.float64)
    event_carry = np.asarray(arrays["event_carry"], dtype=np.int64)
    features = np.asarray(arrays["event_features_r0"], dtype=np.float64)
    feature_valid = np.asarray(arrays["event_feature_valid"], dtype=bool)
    train_event = np.asarray(arrays["train_event_mask"], dtype=bool)
    n, m = anchor_time.size, event_time.size
    if len(bins) != 3 or tuple(bins) != ((0.0, 300.0), (300.0, 900.0), (900.0, 1800.0)):
        raise ValueError("human R0 loader only accepts the canonical disjoint 5/15/30 minute bins")
    if any(value.shape != (n,) for value in (anchor_id, phase, anchor_carry, target_segment, last)) \
            or any(value.shape != (m,) for value in (event_id, event_carry, train_event)) \
            or eligible.shape != (n, 3) or counts_all.shape != (n, 3) or log_mu_all.shape != (n, 3) \
            or log_r.shape != (3,) or features.shape != feature_valid.shape or features.shape[0] != m:
        raise ValueError("materialized human arrays are not aligned")
    if np.unique(anchor_id).size != n or np.unique(event_id).size != m or np.any(np.diff(event_time) < 0):
        raise ValueError("human anchor/event identity or ordering is invalid")
    if not np.issubdtype(counts_all.dtype, np.integer) or not np.isfinite(log_r).all() \
            or not np.isfinite(features[feature_valid]).all():
        raise ValueError("human counts, dispersion or valid features are invalid")
    for label, bounds in (("carry", carry_bounds), ("target", target_bounds)):
        if bounds.ndim != 2 or bounds.shape[1:] != (2,) or not np.isfinite(bounds).all() \
                or np.any(bounds[:, 1] <= bounds[:, 0]):
            raise ValueError(f"human {label} bounds are invalid")
    if np.any(anchor_carry < 0) or np.any(anchor_carry >= carry_bounds.shape[0]) \
            or np.any(event_carry < 0) or np.any(event_carry >= carry_bounds.shape[0]) \
            or np.any(target_segment < 0) or np.any(target_segment >= target_bounds.shape[0]):
        raise ValueError("human carry/target ids do not index their registered bounds")
    if np.any(anchor_time < carry_bounds[anchor_carry, 0]) \
            or np.any(anchor_time >= carry_bounds[anchor_carry, 1]) \
            or np.any(event_time < carry_bounds[event_carry, 0]) \
            or np.any(event_time >= carry_bounds[event_carry, 1]) \
            or np.any(anchor_time < target_bounds[target_segment, 0]) \
            or np.any(anchor_time >= target_bounds[target_segment, 1]):
        raise ValueError("human event/anchor time lies outside its registered boundary")
    expected_last = _strict_last_event(event_time, event_carry, anchor_time, anchor_carry)
    if not np.array_equal(last, expected_last):
        raise ValueError("human last_event_pos is not strictly pre-anchor within state carry")

    event_target = np.full(m, -1, dtype=np.int64)
    for segment, (left, right) in enumerate(target_bounds):
        inside = (event_time >= left) & (event_time < right)
        if np.any(event_target[inside] >= 0):
            raise ValueError("overlapping target segments make event assignment ambiguous")
        event_target[inside] = segment
    recomputed = _target_counts(event_time, event_target, anchor_time, target_segment, bins)
    if np.any(eligible[:, 1] & ~eligible[:, 0]) or np.any(eligible[:, 2] & ~eligible[:, 1]) \
            or not np.array_equal(counts_all[eligible], recomputed[eligible]) \
            or np.any(counts_all[~eligible] != -1) or not np.isfinite(log_mu_all[eligible]).all():
        raise ValueError("human future counts/H_mark disagree with target segments and eligibility")

    full = eligible.all(axis=1)
    phase_index = {
        "train": np.flatnonzero((phase == TRAIN_PHASE) & full),
        "inner_val": np.flatnonzero((phase == INNER_VAL_PHASE) & full),
    }
    if any(index.size == 0 for index in phase_index.values()):
        raise ValueError("human R0 needs non-empty STATE_TRAIN and STATE_SELECTION anchors")
    exposed = np.zeros(n, dtype=bool)
    exposed[np.r_[phase_index["train"], phase_index["inner_val"]]] = True
    if np.any(exposed & (phase == DEVELOPMENT_PHASE)):
        raise PermissionError("development-evaluation anchors entered the training view")
    counts = np.full((n, 3), -1, dtype=np.int64)
    counts[exposed] = counts_all[exposed]
    log_mu = np.full((n, 3), np.nan, dtype=np.float64)
    log_mu[exposed] = log_mu_all[exposed]

    raw = np.where(feature_valid, features, np.nan)
    if scaling == "robust":
        scaler_stats = robust_scale_fit(raw, train_event)
        x_scaled = robust_scale_apply(raw, scaler_stats)
    elif scaling == "zscore":
        train_values = raw[train_event]
        center = np.nanmean(train_values, axis=0)
        scale = np.nanstd(train_values, axis=0)
        degenerate = ~np.isfinite(center) | ~np.isfinite(scale) | (scale <= 1e-9)
        center = np.where(degenerate, 0.0, center)
        scale = np.where(degenerate, 1.0, scale)
        x_scaled = np.nan_to_num((raw - center) / scale, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        scaler_stats = {"method": "zscore", "center": center.tolist(), "scale": scale.tolist(),
                        "degenerate": degenerate.tolist(), "fit": "CALIBRATION_plus_STATE_TRAIN_events_only"}
    else:
        raise ValueError(f"unknown scaling {scaling!r}")
    x_scaled[~feature_valid] = 0.0
    names = tuple(str(value) for value in metadata.get("event_feature_names_r0", []))
    if len(names) != features.shape[1]:
        raise ValueError("human R0 feature names do not match feature width")

    view = DataView(
        subject=subject, bins=bins, horizon=max(right for _left, right in bins),
        event_times=event_time, event_segment=event_carry,
        x_scaled=np.ascontiguousarray(x_scaled), train_event_mask=train_event,
        t_anchor=anchor_time, anchor_segment=anchor_carry, last_event_pos=last,
        segment_bounds=carry_bounds, phase_index=phase_index, counts=counts,
        log_mu_h=log_mu, log_r_h=log_r, h_source="materialized_agent_c_H_mark",
        missing_h_bins=[], split_hash=str(request["split_hash"]), input_hash=artifact_hash,
        scaling=scaling, feature_names=names,
        fingerprint={"npz_sha256": artifact_hash, "content_sha256": content_hash,
                     "manifest_sha256": manifest_hash, "h_mark_artifact_hash": h_content_hash,
                     "evaluator_sha256": evaluator_hash, "release_sha256": release_hash},
        scaler_stats=scaler_stats,
        h_meta={"artifact_manifest": str(manifest_path), "baseline_H": baseline,
                "phase_contract": {"train": TRAIN_PHASE, "inner_val": INNER_VAL_PHASE,
                                   "development_evaluation_exposed": False},
                "bin_convention": BIN_CONVENTION},
        bundle=None, target_segment=target_segment, target_segment_bounds=target_bounds,
    )
    meta = {
        "kind": "R0", "subject": subject, "scaffold": "materialized_agent_c_human_r0",
        "human_data_used": True, "artifact_path": str(artifact_path), "artifact_sha256": artifact_hash,
        "artifact_content_sha256": content_hash, "bin_convention": BIN_CONVENTION,
        "scaling": scaling, "definition_owner": "agent_c", "release_present": True,
        "development_evaluation_exposed": False,
    }
    return view, meta
