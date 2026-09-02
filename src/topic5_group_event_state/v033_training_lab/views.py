"""Build the data view a request asks for (design §2/§3, assumption A8).

``toy`` and ``synthetic`` views never need a release.  A synthetic view may be
planted on a *human scaffold* (real event tokens, synthetic counts) only for
subjects that were already development patients in v0.3.2 -- no untouched
replication patient is ever touched before the release.  ``R0`` / ``R1`` are
human views and are refused without the execution release.

The R0 / R1 token definitions belong to Agent C; the builders here are the
v0.3.2 event token (R1-like, contact-resolved participation / leader columns)
and its summary-only projection (R0-like).  Both carry ``definition_owner``.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.topic5_group_event_state.v032_model.data import SubjectBundle, load_subject_bundle
from src.topic5_group_event_state.v032_model.features import TrainStandardizer

from .data import DEFAULT_BINS, DataView, build_view, robust_scale_apply, robust_scale_fit
from .request import is_human_view
from .synthetic import plant_residual_signal
from .toy import toy_bundle

PRERELEASE_HUMAN_SCAFFOLD_SUBJECTS = ("epilepsiae_1146",)
CONTACT_COLUMN_PREFIXES = ("participation[", "leader[")


class ViewHeld(PermissionError):
    """Raised when a human view is requested without an execution release."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _materialized_synthetic_view(
    request: Mapping[str, Any], *, bins: tuple[tuple[float, float], ...], scaling: str,
) -> tuple[DataView, dict[str, Any]]:
    """Load Agent C's immutable synthetic NPZ without importing its worktree code."""

    iv = dict(request["input_view"])
    artifact_path = Path(str(iv.get("artifact_path", "")))
    manifest_path = Path(str(iv.get("artifact_manifest", "")))
    if not artifact_path.is_file() or not manifest_path.is_file():
        raise ViewHeld("materialized synthetic artifact or manifest is missing")
    manifest_bytes_hash = _sha256_file(manifest_path)
    if manifest_bytes_hash != str(iv.get("artifact_manifest_sha256", "")):
        raise ValueError("synthetic artifact manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    file_hash = _sha256_file(artifact_path)
    expected_file_hash = str(iv.get("artifact_sha256", ""))
    if file_hash != expected_file_hash or file_hash != str(request.get("input_hash", "")) \
            or file_hash != str(manifest.get("npz_sha256", "")):
        raise ValueError("materialized synthetic NPZ byte hash mismatch")
    if manifest.get("sealed") is not False or manifest.get("human_data_used") is not False:
        raise PermissionError("synthetic sentinel must be unsealed and contain no human data")
    expected_convention = "left_closed_right_open_[t+a,t+b)"
    if manifest.get("bin_convention") != expected_convention \
            or request.get("scientific_target", {}).get("bin_convention") != expected_convention:
        raise ValueError("synthetic sentinel target convention is not canonical [t+a,t+b)")
    with np.load(artifact_path, allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata_json"]))
        content_hash = str(stored["artifact_hash"])
        arrays = {name: np.asarray(stored[name]) for name in stored.files
                  if name not in {"metadata_json", "artifact_hash"}}
    if content_hash != str(iv.get("artifact_content_sha256", "")) \
            or content_hash != str(manifest.get("artifact_content_sha256", "")):
        raise ValueError("materialized synthetic content hash mismatch")
    if metadata.get("format") != "group_event_state_v0_3_3_materialized_synthetic_training_input" \
            or metadata.get("bin_convention") != expected_convention:
        raise ValueError("unexpected materialized synthetic schema")
    if metadata.get("science_code_commit") != request.get("science_code_commit") \
            or manifest.get("science_code_commit") != request.get("science_code_commit"):
        raise ValueError("synthetic science-code identity mismatch")
    if manifest.get("split_hash") != request.get("split_hash"):
        raise ValueError("synthetic split hash mismatch")
    baseline = request.get("baseline_H", {})
    if baseline.get("name") != "H_mark" \
            or baseline.get("hash") != manifest.get("baseline_H", {}).get("hash"):
        raise ValueError("synthetic H_mark identity mismatch")

    required = {
        "row_id", "anchor_time", "anchor_segment", "anchor_valid_until", "split",
        "eligible_by_horizon", "target_counts", "log_mu_h_mark", "nb_log_r_train_frozen",
        "event_id", "event_time", "event_segment", "event_features_r0", "event_feature_valid",
        "train_event_mask", "last_event_pos",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"materialized synthetic arrays missing: {missing}")
    event_time = np.asarray(arrays["event_time"], dtype=np.float64)
    event_segment = np.asarray(arrays["event_segment"], dtype=np.int64)
    anchor_time = np.asarray(arrays["anchor_time"], dtype=np.float64)
    anchor_segment = np.asarray(arrays["anchor_segment"], dtype=np.int64)
    last = np.asarray(arrays["last_event_pos"], dtype=np.int64)
    split = np.asarray(arrays["split"]).astype(str)
    eligible = np.asarray(arrays["eligible_by_horizon"], dtype=bool)
    counts = np.asarray(arrays["target_counts"])
    log_mu = np.asarray(arrays["log_mu_h_mark"], dtype=np.float64)
    log_r = np.asarray(arrays["nb_log_r_train_frozen"], dtype=np.float64)
    features = np.asarray(arrays["event_features_r0"], dtype=np.float64)
    feature_valid = np.asarray(arrays["event_feature_valid"], dtype=bool)
    train_event = np.asarray(arrays["train_event_mask"], dtype=bool)
    n, m = anchor_time.size, event_time.size
    if event_segment.shape != (m,) or last.shape != (n,) or anchor_segment.shape != (n,) \
            or eligible.shape != (n, len(bins)) or counts.shape != eligible.shape \
            or log_mu.shape != eligible.shape or log_r.shape != (len(bins),):
        raise ValueError("materialized synthetic arrays are not aligned")
    if features.shape != feature_valid.shape or features.shape[0] != m or train_event.shape != (m,):
        raise ValueError("materialized synthetic event features are not aligned")
    if not np.issubdtype(counts.dtype, np.integer) or not np.isfinite(log_r).all():
        raise ValueError("materialized counts/log_r have invalid type or values")
    if not np.array_equal(np.unique(anchor_segment), np.arange(int(anchor_segment.max()) + 1)):
        raise ValueError("materialized segment ids must be dense from zero")
    recomputed = np.zeros_like(counts, dtype=np.int64)
    expected_last = np.full(n, -1, dtype=np.int64)
    for i, (t0, seg) in enumerate(zip(anchor_time, anchor_segment)):
        same = event_segment == seg
        prior = np.flatnonzero(same & (event_time < t0))
        expected_last[i] = int(prior[-1]) if prior.size else -1
        for j, (lo, hi) in enumerate(bins):
            recomputed[i, j] = int(np.sum(same & (event_time >= t0 + lo) & (event_time < t0 + hi)))
    if not np.array_equal(last, expected_last):
        raise ValueError("materialized last_event_pos is not strictly pre-anchor")
    if not np.array_equal(counts[eligible], recomputed[eligible]) \
            or np.any(counts[~eligible] != -1):
        raise ValueError("materialized future counts disagree with canonical [t+a,t+b) bins")
    phase_index = {
        "train": np.flatnonzero((split == "TRAIN") & eligible.all(axis=1)),
        "inner_val": np.flatnonzero((split == "INNER_VALIDATION") & eligible.all(axis=1)),
    }
    if any(index.size == 0 for index in phase_index.values()):
        raise ValueError("materialized sentinel needs non-empty TRAIN and INNER_VALIDATION anchors")

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
        x_scaled = ((raw - center) / scale).astype(np.float32)
        x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        scaler_stats = {"method": "zscore", "center": center.tolist(), "scale": scale.tolist(),
                        "degenerate": degenerate.tolist(), "fit": "TRAIN_events_only"}
    else:
        raise ValueError(f"unknown scaling {scaling!r}")
    x_scaled[~feature_valid] = 0.0
    max_seg = int(anchor_segment.max())
    bounds = np.zeros((max_seg + 1, 2), dtype=np.float64)
    valid_until = np.asarray(arrays["anchor_valid_until"], dtype=np.float64)
    for seg in range(max_seg + 1):
        event_sel = event_time[event_segment == seg]
        anchor_sel = anchor_time[anchor_segment == seg]
        bounds[seg, 0] = min(float(event_sel.min()), float(anchor_sel.min()))
        bounds[seg, 1] = float(valid_until[anchor_segment == seg].max())
    feature_names = tuple(str(v) for v in metadata.get("event_feature_names_r0", []))
    if len(feature_names) != features.shape[1]:
        raise ValueError("materialized event feature names do not match feature width")
    synthetic_subject = str(request.get("request_id", "materialized_synthetic")).replace("-", "_")
    view = DataView(
        subject=synthetic_subject, bins=bins, horizon=max(b for _a, b in bins),
        event_times=event_time, event_segment=event_segment,
        x_scaled=np.ascontiguousarray(x_scaled), train_event_mask=train_event,
        t_anchor=anchor_time, anchor_segment=anchor_segment, last_event_pos=last,
        segment_bounds=bounds, phase_index=phase_index, counts=counts.astype(np.int64),
        log_mu_h=log_mu, log_r_h=log_r, h_source="materialized_agent_c_H_mark",
        missing_h_bins=[], split_hash=str(request["split_hash"]), input_hash=file_hash,
        scaling=scaling, feature_names=feature_names,
        fingerprint={"npz_sha256": file_hash, "content_sha256": content_hash,
                     "manifest_sha256": manifest_bytes_hash},
        scaler_stats=scaler_stats,
        h_meta={"artifact_manifest": str(manifest_path), "baseline_H": dict(baseline),
                "bin_convention": expected_convention},
        bundle=None,
    )
    meta = {"kind": "synthetic", "subject": view.subject, "scaffold": "materialized_agent_c",
            "human_data_used": False, "artifact_path": str(artifact_path), "artifact_sha256": file_hash,
            "artifact_content_sha256": content_hash, "bin_convention": expected_convention,
            "scaling": scaling, "definition_owner": "agent_c", "release_present": False}
    return view, meta


def summary_only_bundle(bundle: SubjectBundle) -> SubjectBundle:
    """R0-like projection: drop per-contact participation / leader columns."""

    keep = np.array([not n.startswith(CONTACT_COLUMN_PREFIXES) for n in bundle.feature_names], dtype=bool)
    std = bundle.standardizer
    reduced = TrainStandardizer(mean=std.mean[keep], scale=std.scale[keep], zero_variance=std.zero_variance[keep])
    return replace(bundle, x_raw=bundle.x_raw[:, keep], x_std=bundle.x_std[:, keep],
                   feature_names=tuple(n for n, k in zip(bundle.feature_names, keep) if k), standardizer=reduced)


def view_for_request(request: Mapping[str, Any], *, release_present: bool, scaling: str = "zscore") -> tuple[DataView, dict[str, Any]]:
    iv = dict(request["input_view"])
    kind = str(iv.get("kind", ""))
    bins = tuple(tuple(float(v) for v in b) for b in request["scientific_target"].get("bins_seconds", DEFAULT_BINS))
    meta: dict[str, Any] = {"kind": kind, "bins_seconds": [list(b) for b in bins], "scaling": scaling,
                            "definition_owner": "agent_c", "release_present": bool(release_present)}
    if kind == "toy":
        bundle = toy_bundle(int(iv.get("seed", 0)))
        meta.update({"subject": "toy", "view_definition": "toy two-segment bundle (v0.3.2 test toy)"})
        return build_view(bundle, bins=bins, scaling=scaling), meta
    if kind == "synthetic":
        if iv.get("materialized_arrays_only") is True:
            return _materialized_synthetic_view(request, bins=bins, scaling=scaling)
        base_subject = iv.get("base_subject")
        if base_subject:
            if not release_present and base_subject not in PRERELEASE_HUMAN_SCAFFOLD_SUBJECTS:
                raise ViewHeld(f"synthetic scaffold {base_subject!r} is not allowed before the release "
                               f"(allowed: {PRERELEASE_HUMAN_SCAFFOLD_SUBJECTS})")
            bundle = load_subject_bundle(str(base_subject), allow_provisional_h=False)
            meta.update({"subject": str(base_subject), "scaffold": "human_event_tokens",
                         "view_definition": "v0.3.2 event token scaffold with synthetic residual-positive counts"})
        else:
            bundle = toy_bundle(int(iv.get("seed", 0)))
            meta.update({"subject": "toy", "scaffold": "toy", "view_definition": "toy scaffold with synthetic counts"})
        base = build_view(bundle, bins=bins, scaling=scaling)
        spec = dict(iv.get("synthetic") or {})
        planted, info = plant_residual_signal(base, beta=float(spec.get("beta", 0.7)),
                                              dispersion_r=float(spec.get("dispersion_r", 5.0)),
                                              generator_seed=int(spec.get("generator_seed", 1)),
                                              noise_seed=int(spec.get("noise_seed", 2)))
        meta["synthetic"] = {k: v for k, v in info.items() if k not in ("z", "log_mu_true")}
        return planted, meta
    if is_human_view(iv):
        if not release_present:
            raise ViewHeld("human input view requested without V0_3_3_EXECUTION_RELEASE.json")
        subject = str(iv["subject"])
        bundle = load_subject_bundle(subject, allow_provisional_h=False)
        if kind == "R0":
            bundle = summary_only_bundle(bundle)
            meta["view_definition"] = "v0.3.2 event token without per-contact columns (R0-like summary token)"
        else:
            meta["view_definition"] = "v0.3.2 event token (R1-like contact-resolved token)"
        meta["subject"] = subject
        return build_view(bundle, bins=bins, scaling=scaling), meta
    raise ValueError(f"unknown input_view.kind {kind!r}")
