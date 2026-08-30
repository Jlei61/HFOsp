"""Causal frozen-state extraction for H2b cross-task transfer.

The seizure task is deliberately absent from this module.  It accepts query
times only after a Continuous Marked State checkpoint has been reconstructed,
verified and frozen.  Every returned value is therefore a readout of the
interictal model, never an update driven by a seizure label.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from src.topic5_continuous_marked_state_r1.history import BASE_HISTORY_NAMES
from src.topic5_continuous_marked_state_r1.history import (
    DeterministicHistory,
    HistoryScaler,
)
from src.topic5_continuous_marked_state_r1.data import R1EventStream
from src.topic5_continuous_marked_state_r1.observer import ObservationTransformer
from src.topic5_continuous_marked_state_r1.coverage import CoverageTable
from src.topic5_continuous_marked_state_r1.r1_2 import FullAnchorDesign, load_full_design
from src.topic5_continuous_marked_state_r1.r1_3 import FullTargetObserverStateModel
from src.topic5_continuous_marked_state_r1.raw_observation import (
    ANALYSIS_RATE_HZ,
    EXPLICIT_NAMES,
    WINDOW_SECONDS,
    RawAnchorReader,
    _raw_cache_dir,
)

from . import contract as h2b_contract


H2B_STATE_EXTRACTION_REVISION = "h2b_cross_task_causal_state_extraction_v0_2"
SUPPORTED_CHECKPOINT_REVISIONS = {
    "r1_6_optimizer_identifiability_nested_selection_v1",
    "r1_7a_prospective_state_replication_v1",
    "r1_7b_extended_development_cohort_v1",
}
INTERICTAL_SOURCE_TASK = "continuous_background_and_ied_timing_mark"
INFERENCE_OBSERVATION_REVISION = (
    "h2b_inference_covered_cached_artifact_valid_without_training_guard_v1"
)


sha256_file = h2b_contract.sha256_file


def freeze_and_assert(model: torch.nn.Module) -> torch.nn.Module:
    """Freeze a state model and fail if any trainable parameter remains."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    if trainable:
        raise AssertionError(f"frozen state model still has trainable parameters: {trainable}")
    if model.training or any(module.training for module in model.modules()):
        raise AssertionError("frozen state model is not fully in eval mode")
    return model


def _strip_prefix(state: Mapping[str, torch.Tensor], prefix: str
                  ) -> dict[str, torch.Tensor]:
    return {
        key[len(prefix):]: value
        for key, value in state.items() if key.startswith(prefix)
    }


def load_frozen_r16_checkpoint(
        checkpoint: str | Path, *, expected_sha256: str,
        expected_subject: str | None = None,
        expected_seed: int | None = None,
        device: str | torch.device = "cpu",
        require_stable_result: bool = True,
        require_complete_result: bool = True,
        ) -> tuple[FullTargetObserverStateModel, dict]:
    """Reconstruct one audited R1.6 explicit-observer checkpoint, fail closed.

    The R1.6 payload stores a complete state dict but not constructor metadata.
    Dimensions are recovered from tensor shapes; architecture-changing
    revisions must be added explicitly to ``SUPPORTED_CHECKPOINT_REVISIONS``.
    """
    path = Path(checkpoint).resolve()
    observed_hash = sha256_file(path)
    if observed_hash != str(expected_sha256):
        raise ValueError(
            f"checkpoint SHA256 mismatch: expected {expected_sha256}, got {observed_hash}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("unsupported R1.6 checkpoint payload")
    revision = str(payload.get("revision", ""))
    if revision not in SUPPORTED_CHECKPOINT_REVISIONS:
        raise ValueError(f"unsupported interictal checkpoint revision {revision!r}")
    subject = str(payload.get("subject", ""))
    seed = int(payload.get("seed", -1))
    if expected_subject is not None and subject != str(expected_subject):
        raise ValueError(f"checkpoint subject mismatch: {subject} != {expected_subject}")
    if expected_seed is not None and seed != int(expected_seed):
        raise ValueError(f"checkpoint seed mismatch: {seed} != {expected_seed}")

    result_path = path.with_name("result.json")
    result_hash = None
    if require_complete_result or require_stable_result:
        if not result_path.exists():
            raise ValueError(f"checkpoint has no adjacent result.json: {path}")
        result = json.loads(result_path.read_text())
        if result.get("status") != "COMPLETE":
            raise ValueError("interictal result is not COMPLETE")
        if require_stable_result and result.get("stable_checkpoint") is not True:
            raise ValueError("interictal result does not declare a stable checkpoint")
        if result.get("formal_test_partition_opened") is not False:
            raise ValueError("R1.6 result opened the formal test partition")
        if result.get("sealed_opened") is not False:
            raise ValueError("R1.6 result opened sealed data")
        if result.get("checkpoint_sha256") != observed_hash:
            raise ValueError("R1.6 result/checkpoint hash disagreement")
        if str(result.get("subject")) != subject or int(result.get("seed", -1)) != seed:
            raise ValueError("R1.6 result/checkpoint identity disagreement")
        result_hash = sha256_file(result_path)

    state = payload["model"]
    required = (
        "timing_baseline.weight", "mark_baseline.adjacency",
        "state_contact.weight", "state.generator.mu",
        "observer.explicit.0.weight", "observer.raw.tokenizer.weight",
        "observer.raw.position", "observer.shaft.weight",
    )
    missing = [name for name in required if name not in state]
    if missing:
        raise ValueError(f"R1.6 checkpoint lacks constructor fields: {missing}")
    history_dim = int(state["timing_baseline.weight"].shape[-1])
    n_contacts = int(state["state_contact.weight"].shape[0])
    state_dim = int(state["state.generator.mu"].numel())
    explicit_dim = int(state["observer.explicit.0.weight"].shape[1])
    d_model = int(state["observer.explicit.0.weight"].shape[0])
    patch_samples = int(state["observer.raw.tokenizer.weight"].shape[-1])
    max_patches = int(state["observer.raw.position"].shape[1])
    max_shafts = int(state["observer.shaft.weight"].shape[0])
    temporal_layers = len({
        key.split(".")[4] for key in state
        if key.startswith("observer.raw.transformer.layers.")
    })
    spatial_layers = len({
        key.split(".")[3] for key in state
        if key.startswith("observer.spatial.layers.")
    })
    if temporal_layers < 1 or spatial_layers < 1:
        raise ValueError("checkpoint observer layer inventory is invalid")
    adjacency = state["mark_baseline.adjacency"].detach().cpu().numpy()
    baseline = {
        "timing": {"history": _strip_prefix(state, "timing_baseline.")},
        "mark": {"history": _strip_prefix(state, "mark_baseline.")},
    }
    observer = ObservationTransformer(
        explicit_dim, d_model=d_model, patch_samples=patch_samples,
        # R1.6's frozen architecture uses four heads.  A future architecture
        # must carry a new checkpoint revision rather than being guessed here.
        n_heads=4, temporal_layers=temporal_layers,
        spatial_layers=spatial_layers, max_shafts=max_shafts,
        raw_enabled=True,
    )
    # Preserve a non-default position length if a later checkpoint within the
    # frozen revision was written with it.
    if max_patches != observer.raw.position.shape[1]:
        observer.raw.position = torch.nn.Parameter(torch.zeros(1, max_patches, d_model))
    model = FullTargetObserverStateModel(
        baseline, history_dim, n_contacts, adjacency, observer,
        use_raw=False, state_dim=state_dim,
    ).to(device)
    model.load_state_dict(state, strict=True)
    freeze_and_assert(model)
    provenance = {
        "state_extraction_revision": H2B_STATE_EXTRACTION_REVISION,
        "source_task": INTERICTAL_SOURCE_TASK,
        "checkpoint": str(path),
        "checkpoint_sha256": observed_hash,
        "checkpoint_revision": revision,
        "checkpoint_confirmation_revision": payload.get("confirmation_revision"),
        "checkpoint_subject": subject,
        "checkpoint_seed": seed,
        "checkpoint_result": str(result_path) if result_path.exists() else None,
        "checkpoint_result_sha256": result_hash,
        "checkpoint_result_stable": (
            result.get("stable_checkpoint") is True
            if (require_complete_result or require_stable_result) else None
        ),
        "state_frozen": True,
        "all_parameters_require_grad_false": True,
        "seizure_gradient_path": False,
        "formal": False,
        "sealed": False,
    }
    return model, provenance


@dataclass(frozen=True)
class FrozenAnchorInputs:
    design: FullAnchorDesign
    explicit: np.ndarray
    contact_mask: np.ndarray
    coordinates: np.ndarray
    coordinate_valid: np.ndarray
    shaft_index: np.ndarray
    manifest: dict
    manifest_path: Path


class InferenceRawAnchorReader(RawAnchorReader):
    """Inference-only raw reader that does not reuse the labelled TRAIN guard.

    The upstream ``minute_usable`` is a training field and already contains
    ``guard_free``.  H2b reconstructs availability only from observable storage
    and artifact fields, then applies the independent R1.2 admissible-coverage
    table to exclude ictal/postictal time and reset state at gaps.
    """

    def __init__(self, subject: str, event_times: np.ndarray, *,
                 source_repo_root: str | Path = "/home/honglab/leijiaxin/HFOsp"):
        # Do not call ``RawAnchorReader.__init__``: it resolves contact metadata
        # from the active worktree's ignored result tree.  H2b runs in an
        # isolated worktree and must name the read-only canonical artifact root
        # explicitly instead of silently depending on a local ignored file.
        import pandas as pd
        import zarr

        self.subject = str(subject)
        self.cache_dir = _raw_cache_dir(self.subject)
        required = {
            "raw": self.cache_dir / "raw_256hz.zarr",
            "artifact": self.cache_dir / "artifact_mask.zarr",
            "train_stats": self.cache_dir / "train_stats.json",
            "window_index": self.cache_dir / "window_index_refined.parquet",
            "cache_index": self.cache_dir / "cache_index.parquet",
        }
        missing = [str(path) for path in required.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{subject}: missing raw cache inputs {missing}")
        self.raw = zarr.open_array(str(required["raw"]), mode="r")
        self.artifact = zarr.open_array(str(required["artifact"]), mode="r")
        stats = json.loads(required["train_stats"].read_text())
        self.count_scale = np.asarray(stats["int16_scale_uv"], dtype=np.float32)
        self.raw_center = np.asarray(stats["raw_center_uv"], dtype=np.float32)
        self.raw_scale = np.asarray(stats["raw_scale_uv"], dtype=np.float32)
        self.window_index = pd.read_parquet(required["window_index"]).sort_values(
            "minute_index"
        )
        cache_index = pd.read_parquet(required["cache_index"])
        n_minutes = int(self.window_index.minute_index.max()) + 1
        self.minute_start = np.full(n_minutes, np.nan, dtype=np.float64)
        self.minute_session = np.full(n_minutes, -1, dtype=np.int64)
        row_index = self.window_index.minute_index.to_numpy(dtype=np.int64)
        if not np.array_equal(row_index, np.arange(n_minutes)):
            raise ValueError(f"{subject}: raw minute grid is not canonical")
        self.minute_start[row_index] = self.window_index.minute_start_epoch.to_numpy(
            dtype=np.float64
        )
        self.minute_session[row_index] = self.window_index.session_id.to_numpy(
            dtype=np.int64
        )
        self.first_epoch = float(self.minute_start[0])
        self.event_times = np.asarray(event_times, dtype=np.float64)
        self.window_samples = int(WINDOW_SECONDS * ANALYSIS_RATE_HZ)
        self.minute_samples = 60 * ANALYSIS_RATE_HZ
        self._decoded_minute_index = -1
        self._decoded_minute_raw = None
        self._decoded_minute_artifact = None

        metadata_path = (
            Path(source_repo_root).resolve()
            / "results/epi_prssm/raw_seeg_state/r0_1/data/contact_metadata.parquet"
        )
        metadata = pd.read_parquet(metadata_path)
        metadata = metadata[
            metadata.subject.astype(str) == self.subject
        ].sort_values("channel_index")
        metadata = metadata[metadata.contact_valid.astype(bool)]
        if len(metadata) != self.raw.shape[1]:
            raise ValueError(
                f"{subject}: metadata/raw contact counts disagree "
                f"({len(metadata)} vs {self.raw.shape[1]})"
            )
        if not np.array_equal(
                metadata.channel_index.to_numpy(), np.arange(len(metadata))):
            raise ValueError(f"{subject}: raw contact metadata is not canonical")
        self.contact_names = metadata.channel_name.astype(str).to_numpy()
        coordinate = metadata[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=np.float32)
        self.coordinate_valid = (
            metadata.coord_valid.to_numpy(dtype=bool)
            & np.isfinite(coordinate).all(1)
        )
        centre = (
            coordinate[self.coordinate_valid].mean(0)
            if bool(self.coordinate_valid.any()) else np.zeros(3, dtype=np.float32)
        )
        centred = np.where(self.coordinate_valid[:, None], coordinate - centre, 0.0)
        scale = (
            centred[self.coordinate_valid].std(0)
            if bool(self.coordinate_valid.any()) else np.ones(3, dtype=np.float32)
        )
        scale = np.where(np.isfinite(scale) & (scale > 1e-3), scale, 1.0)
        self.coordinates = (centred / scale).astype(np.float32)
        _, shaft_index = np.unique(
            metadata.shaft.astype(str).to_numpy(), return_inverse=True
        )
        self.shaft_index = shaft_index.astype(np.int64)

        cache_index = self._read_cache_index()
        cached = np.zeros(len(self.window_index), dtype=bool)
        cache_minute = cache_index.minute_index.to_numpy(dtype=np.int64)
        if np.any(cache_minute < 0) or np.any(cache_minute >= len(cached)):
            raise ValueError(f"{subject}: cache index is outside the raw minute grid")
        cached[cache_minute] = cache_index.cached.to_numpy(dtype=bool)
        stats_path = required["train_stats"]
        training_threshold = float(stats["minute_min_valid_contact_fraction"])
        # The subject-specific valid-contact threshold is part of the frozen
        # raw-cache contract.  It is read from the hash-verified TRAIN stats,
        # never relaxed to RawAnchorReader's generic 50% decoding floor.
        inference_threshold = training_threshold
        n_contacts = int(self.raw.shape[1])
        if int(stats["n_contacts"]) != n_contacts:
            raise ValueError(f"{subject}: train-stats/raw contact count mismatch")
        frame = self.window_index
        n_valid = frame.n_valid_contacts.to_numpy(dtype=np.int64)
        inference_usable = (
            frame.covered.to_numpy(dtype=bool)
            & (frame.session_id.to_numpy(dtype=np.int64) >= 0)
            & cached
            & ((n_valid / float(n_contacts)) >= inference_threshold)
        )
        training_guard_free = frame.guard_free.to_numpy(dtype=bool)
        training_minute_usable = frame.minute_usable.to_numpy(dtype=bool)
        training_observable = (
            frame.covered.to_numpy(dtype=bool)
            & (frame.session_id.to_numpy(dtype=np.int64) >= 0)
            & cached
            & ((n_valid / float(n_contacts)) >= training_threshold)
        )
        expected_training = training_observable & training_guard_free
        if not np.array_equal(training_minute_usable & cached, expected_training):
            raise ValueError(
                f"{subject}: frozen minute_usable no longer factors into "
                "inference availability and the labelled training guard"
            )
        # Cross-check the artifact/contact fraction against the actual mask.
        artifact = np.asarray(self.artifact[:], dtype=bool)
        if artifact.shape != (len(frame), n_contacts):
            raise ValueError(f"{subject}: artifact-mask shape mismatch")
        observed_n_valid = (~artifact).sum(1).astype(np.int64)
        if not np.array_equal(observed_n_valid[cached], n_valid[cached]):
            raise ValueError(f"{subject}: artifact mask and n_valid_contacts disagree")

        self.cached = cached
        self.training_guard_free = training_guard_free
        self.training_minute_usable = training_minute_usable
        self.inference_usable = inference_usable
        self.inference_min_valid_contact_fraction = inference_threshold
        self.training_min_valid_contact_fraction = training_threshold
        # Inherited ``read`` and ``can_read`` now operate on the inference gate.
        self.usable = inference_usable.copy()
        self.source_hashes = self._source_hashes(stats_path)
        self.source_hashes["contact_metadata"] = sha256_file(metadata_path)

    def _read_cache_index(self):
        import pandas as pd

        return pd.read_parquet(self.cache_dir / "cache_index.parquet")

    def _source_hashes(self, stats_path: Path) -> dict[str, str]:
        paths = {
            "window_index_refined": self.cache_dir / "window_index_refined.parquet",
            "cache_index": self.cache_dir / "cache_index.parquet",
            "train_stats": stats_path,
            "raw_cache_build_status": self.cache_dir / "BUILD_STATUS.json",
            "raw_cache_target_status": self.cache_dir / "TARGET_STATUS.json",
            "raw_zarr_metadata": self.cache_dir / "raw_256hz.zarr/zarr.json",
            "artifact_zarr_metadata": self.cache_dir / "artifact_mask.zarr/zarr.json",
        }
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"{self.subject}: inference raw-cache provenance missing {missing}"
            )
        return {name: sha256_file(path) for name, path in paths.items()}

    def inference_anchor_inventory(
            self, coverage: CoverageTable, *,
            allowed_segments: Iterable[int] | None = None,
            upper_time_by_segment: Mapping[int, float] | None = None,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return causal 30 s anchors wholly inside admissible coverage."""
        if coverage.subject != self.subject:
            raise ValueError("raw reader/coverage subject mismatch")
        minute = np.flatnonzero(self.inference_usable)
        anchor = np.concatenate([
            self.minute_start[minute] + 30.0,
            self.minute_start[minute] + 60.0,
        ]).astype(np.float64)
        source_minute = np.concatenate([minute, minute]).astype(np.int64)
        order = np.argsort(anchor, kind="stable")
        anchor, source_minute = anchor[order], source_minute[order]
        # Add the frozen causal-inpainting requirement (at least two pre-anchor
        # background samples).  Artifact/contact eligibility was already
        # checked vectorially above, so avoid thousands of duplicate Zarr reads
        # through ``can_read``.  The inherited mask clips IEDs at the anchor.
        readable = np.asarray([
            int((~self._ied_core_mask(float(value))).sum()) >= 2
            for value in anchor
        ], dtype=bool)
        anchor, source_minute = anchor[readable], source_minute[readable]
        segment = np.full(len(anchor), -1, dtype=np.int64)
        continuity_session = np.full(len(anchor), -1, dtype=np.int64)
        window_start = anchor - 30.0
        for row, (left, right, label) in enumerate(zip(
                coverage.start, coverage.stop, coverage.session)):
            hit = (window_start >= float(left)) & (anchor <= float(right))
            if bool(np.any(segment[hit] >= 0)):
                raise ValueError("inference observation maps to overlapping coverage")
            # State reset and wrong-time matching use the unique coverage-row
            # index.  ``coverage.session`` is a separate continuity label used
            # only by deterministic IED history and can repeat after a gap.
            segment[hit] = int(row)
            continuity_session[hit] = int(label)
        keep = segment >= 0
        if allowed_segments is not None:
            keep &= np.isin(segment, np.asarray(list(allowed_segments), dtype=np.int64))
        if upper_time_by_segment is not None:
            for label in np.unique(segment[keep]):
                if int(label) not in upper_time_by_segment:
                    keep[segment == label] = False
                else:
                    keep[(segment == label) & (
                        anchor > float(upper_time_by_segment[int(label)])
                    )] = False
        anchor, segment, continuity_session, source_minute = (
            value[keep] for value in (
                anchor, segment, continuity_session, source_minute
            )
        )
        training_guard_free = self.training_guard_free[source_minute]
        if len(anchor) and np.any(np.diff(anchor) < 0):
            raise AssertionError("inference anchors are not chronological")
        return (
            anchor, segment, continuity_session, source_minute,
            training_guard_free,
        )


@dataclass(frozen=True)
class InferenceAnchorInputs:
    anchor_time_epoch: np.ndarray
    coverage_segment_index: np.ndarray
    continuity_session: np.ndarray
    source_minute_index: np.ndarray
    training_guard_free: np.ndarray
    explicit: np.ndarray
    contact_mask: np.ndarray
    coordinates: np.ndarray
    coordinate_valid: np.ndarray
    shaft_index: np.ndarray
    provenance: dict


def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


def load_frozen_explicit_scaler(source_repo_root: str | Path, subject: str,
                                *, result_path: str | Path | None = None
                                ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load the bridge-selected TRAIN scaler without refitting on H2b time."""
    path = Path(result_path).resolve() if result_path is not None else (
        Path(source_repo_root).resolve()
        / "results/epi_prssm/continuous_marked_state/r1/r1_2/bridge_e1"
        / subject / "seed_0/result.json"
    )
    payload = json.loads(path.read_text())
    if payload.get("status") != "COMPLETE" or payload.get("sealed_opened") is not False:
        raise ValueError(f"{subject}: frozen bridge scaler source is inadmissible")
    if payload.get("explicit_scaler_source") != "selected_train_anchors":
        # Existing artifacts use this exact declaration; fail on silent drift.
        raise ValueError(f"{subject}: explicit scaler is not TRAIN-only")
    mean = np.asarray(payload["explicit_mean"], dtype=np.float32)
    scale = np.asarray(payload["explicit_scale"], dtype=np.float32)
    if mean.shape != (len(EXPLICIT_NAMES),) or scale.shape != mean.shape:
        raise ValueError(f"{subject}: frozen explicit scaler shape mismatch")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError(f"{subject}: frozen explicit scaler is invalid")
    return mean, scale, {
        "explicit_scaler_source": "frozen_bridge_train_anchors_only",
        "explicit_scaler_result": str(path),
        "explicit_scaler_result_sha256": sha256_file(path),
    }


def build_inference_anchor_inputs(
        reader: InferenceRawAnchorReader, coverage: CoverageTable, *,
        explicit_mean: np.ndarray, explicit_scale: np.ndarray,
        allowed_segments: Iterable[int] | None = None,
        upper_time_by_segment: Mapping[int, float] | None = None,
        ) -> InferenceAnchorInputs:
    """Decode inference-only observations and apply the frozen TRAIN scaler."""
    anchor, segment, continuity_session, source_minute, guard_free = (
        reader.inference_anchor_inventory(
        coverage, allowed_segments=allowed_segments,
        upper_time_by_segment=upper_time_by_segment,
        )
    )
    observations = [reader.read(float(value)) for value in anchor]
    if any(value is None for value in observations):
        bad = [float(anchor[index]) for index, value in enumerate(observations) if value is None]
        raise RuntimeError(
            f"{reader.subject}: inference-eligible raw anchors became unreadable {bad[:5]}"
        )
    if not observations:
        raise ValueError(f"{reader.subject}: no inference observations in requested coverage")
    explicit = np.stack([value.explicit for value in observations]).astype(np.float32)
    contact_mask = np.stack([value.contact_mask for value in observations]).astype(bool)
    explicit = ((explicit - explicit_mean) / explicit_scale).astype(np.float32)
    input_hash = _sha256_arrays(
        anchor.astype(np.float64), segment.astype(np.int64),
        continuity_session.astype(np.int64),
        explicit, contact_mask,
    )
    provenance = {
        "inference_observation_revision": INFERENCE_OBSERVATION_REVISION,
        "inference_availability_components": [
            "covered", "session_id_nonnegative", "cached",
            "artifact_valid_contact_fraction",
            "causal_ied_inpaint_background_samples_at_least_two",
        ],
        "inference_min_valid_contact_fraction": (
            reader.inference_min_valid_contact_fraction
        ),
        "inference_min_valid_contact_fraction_source": (
            "hash_verified_train_stats.minute_min_valid_contact_fraction"
        ),
        "training_guard_free_used_for_inference": False,
        "training_minute_usable_used_for_inference": False,
        "training_guard_free_retained_as_provenance": True,
        "n_inference_anchors": int(len(anchor)),
        "n_inference_anchors_excluded_by_old_training_guard": int((~guard_free).sum()),
        "admissible_coverage_excludes_ictal_postictal_and_gaps": True,
        "coverage_segment_semantics": "unique_coverage_table_row_index",
        "continuity_session_semantics": "coverage_session_for_deterministic_history_only",
        "state_reset_at_every_coverage_segment_row": True,
        "seizure_labels_enter_state_update": False,
        "seizure_label_role": "coverage_exclusion_reset_and_query_construction_only",
        "derived_inference_input_sha256": input_hash,
        "raw_cache_source_hashes": reader.source_hashes,
    }
    return InferenceAnchorInputs(
        anchor_time_epoch=anchor.astype(np.float64),
        coverage_segment_index=segment.astype(np.int64),
        continuity_session=continuity_session.astype(np.int64),
        source_minute_index=source_minute.astype(np.int64),
        training_guard_free=guard_free.astype(bool),
        explicit=explicit,
        contact_mask=contact_mask,
        coordinates=reader.coordinates,
        coordinate_valid=reader.coordinate_valid,
        shaft_index=reader.shaft_index,
        provenance=provenance,
    )


def _contact_geometry(source_repo_root: Path, subject: str,
                      n_contacts: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import pandas as pd

    path = (
        source_repo_root
        / "results/epi_prssm/raw_seeg_state/r0_1/data/contact_metadata.parquet"
    )
    frame = pd.read_parquet(path)
    frame = frame[frame.subject.astype(str) == subject].sort_values("channel_index")
    frame = frame[frame.contact_valid.astype(bool)]
    if len(frame) != int(n_contacts):
        raise ValueError(
            f"{subject}: contact metadata/model mismatch ({len(frame)} != {n_contacts})"
        )
    if not np.array_equal(frame.channel_index.to_numpy(), np.arange(len(frame))):
        raise ValueError(f"{subject}: contact metadata order is not canonical")
    coordinate = frame[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=np.float32)
    valid = frame.coord_valid.to_numpy(dtype=bool) & np.isfinite(coordinate).all(1)
    centre = coordinate[valid].mean(0) if bool(valid.any()) else np.zeros(3, dtype=np.float32)
    centred = np.where(valid[:, None], coordinate - centre, 0.0)
    scale = centred[valid].std(0) if bool(valid.any()) else np.ones(3, dtype=np.float32)
    scale = np.where(np.isfinite(scale) & (scale > 1e-3), scale, 1.0)
    coordinates = (centred / scale).astype(np.float32)
    _, shaft = np.unique(frame.shaft.astype(str).to_numpy(), return_inverse=True)
    return coordinates, valid.astype(bool), shaft.astype(np.int64)


def load_frozen_anchor_inputs(source_repo_root: str | Path, subject: str
                              ) -> FrozenAnchorInputs:
    """Load the hash-verified R1.6 observation cache from a read-only repo."""
    source_repo_root = Path(source_repo_root).resolve()
    manifest_path = (
        source_repo_root
        / "results/epi_prssm/continuous_marked_state/r1/r1_5/cache"
        / subject / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "COMPLETE" or manifest.get("sealed_opened") is not False:
        raise ValueError(f"{subject}: observation cache is not admissible")
    paths = {
        "design": Path(manifest["design"]),
        "explicit": Path(manifest["explicit"]),
        "contact_mask": Path(manifest["contact_mask"]),
    }
    # Old manifests contain absolute canonical paths.  Resolve relative paths
    # against the read-only source repo if encountered.
    for key, value in list(paths.items()):
        if not value.is_absolute():
            paths[key] = source_repo_root / value
        expected = manifest[f"{key}_sha256"]
        if sha256_file(paths[key]) != expected:
            raise ValueError(f"{subject}: {key} cache hash mismatch")
    design = load_full_design(paths["design"])
    explicit = np.load(paths["explicit"], mmap_mode="r")
    contact_mask = np.load(paths["contact_mask"], mmap_mode="r")
    if explicit.ndim != 3 or explicit.shape[0] != len(design.anchor_time):
        raise ValueError(f"{subject}: explicit cache shape mismatch")
    if contact_mask.shape != explicit.shape[:2]:
        raise ValueError(f"{subject}: contact-mask cache shape mismatch")
    # The checkpoint's mark decoder can use a smaller IED-contact set than the
    # all-bipolar background observer (E384: 9 versus 80).  Observer contact
    # count/order therefore comes from this hash-verified cache and canonical
    # contact metadata, never from ``state_contact.out_features``.
    n_observer_contacts = int(explicit.shape[1])
    coordinates, coordinate_valid, shaft_index = _contact_geometry(
        source_repo_root, subject, n_observer_contacts
    )
    return FrozenAnchorInputs(
        design=design, explicit=explicit, contact_mask=contact_mask,
        coordinates=coordinates, coordinate_valid=coordinate_valid,
        shaft_index=shaft_index, manifest=manifest, manifest_path=manifest_path,
    )


def load_frozen_design(source_repo_root: str | Path, subject: str
                       ) -> tuple[FullAnchorDesign, dict, Path]:
    """Load only the causal event/design support, not guarded observations."""
    manifest_path = (
        Path(source_repo_root).resolve()
        / "results/epi_prssm/continuous_marked_state/r1/r1_5/cache"
        / subject / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "COMPLETE" or manifest.get("sealed_opened") is not False:
        raise ValueError(f"{subject}: frozen design manifest is inadmissible")
    design_path = Path(manifest["design"])
    if not design_path.is_absolute():
        design_path = Path(source_repo_root).resolve() / design_path
    if sha256_file(design_path) != manifest["design_sha256"]:
        raise ValueError(f"{subject}: frozen design hash mismatch")
    return load_full_design(design_path), manifest, manifest_path


def load_frozen_design_artifact(
        design_path: str | Path, *, expected_sha256: str,
        expected_subject: str,
        manifest_path: str | Path | None = None,
        ) -> tuple[FullAnchorDesign, dict, Path | None]:
    """Load an explicitly named, hash-locked full design.

    R1.7 was produced in an isolated worktree and some manifests retain the
    now-deleted producer path.  H2b therefore resolves the canonical copied
    artifact itself, verifies its frozen digest, and treats a supplied manifest
    as provenance rather than as a path redirect.
    """
    path = Path(design_path).resolve()
    observed = sha256_file(path)
    if observed != str(expected_sha256):
        raise ValueError(
            f"{expected_subject}: frozen design SHA256 mismatch: "
            f"expected {expected_sha256}, got {observed}"
        )
    design = load_full_design(path)
    if design.subject != str(expected_subject):
        raise ValueError(
            f"frozen design subject mismatch: {design.subject} != {expected_subject}"
        )
    resolved_manifest: Path | None = None
    if manifest_path is not None:
        resolved_manifest = Path(manifest_path).resolve()
        manifest = json.loads(resolved_manifest.read_text())
        if manifest.get("status") != "COMPLETE":
            raise ValueError(f"{expected_subject}: design manifest is not COMPLETE")
        if manifest.get("sealed_opened") is not False:
            raise ValueError(f"{expected_subject}: design manifest opened sealed data")
        if str(manifest.get("subject")) != str(expected_subject):
            raise ValueError(f"{expected_subject}: design manifest subject mismatch")
        if manifest.get("design_sha256") != observed:
            raise ValueError(f"{expected_subject}: design manifest hash mismatch")
    else:
        manifest = {
            "status": "COMPLETE",
            "sealed_opened": False,
            "subject": str(expected_subject),
            "design": str(path),
            "design_sha256": observed,
            "manifest_absent_but_artifact_hash_locked": True,
        }
    return design, manifest, resolved_manifest


def materialize_observation_embeddings(
        model: FullTargetObserverStateModel, inputs: FrozenAnchorInputs, *,
        device: str | torch.device = "cpu", batch_size: int = 256,
        ) -> np.ndarray:
    """Apply the frozen, explicit-only R1.6 observer independently per anchor."""
    freeze_and_assert(model)
    if model.use_raw:
        raise ValueError("R1.6 H2b extraction expects the frozen explicit observer arm")
    result: list[np.ndarray] = []
    with torch.inference_mode():
        for lo in range(0, len(inputs.design.anchor_time), int(batch_size)):
            hi = min(lo + int(batch_size), len(inputs.design.anchor_time))
            explicit = np.array(inputs.explicit[lo:hi], copy=True)
            mask = np.array(inputs.contact_mask[lo:hi], copy=True)
            batch = hi - lo
            dummy = np.zeros((batch, explicit.shape[1], 1), dtype=np.float32)
            value = model.observation_embedding({
                "explicit": torch.as_tensor(explicit, device=device),
                "waveform": torch.as_tensor(dummy, device=device),
                "sample_valid": torch.ones_like(
                    torch.as_tensor(dummy, device=device), dtype=torch.bool
                ),
                "contact_mask": torch.as_tensor(mask, device=device),
                "coordinates": torch.as_tensor(
                    np.broadcast_to(
                        inputs.coordinates, (batch, *inputs.coordinates.shape)
                    ).copy(), device=device,
                ),
                "coordinate_valid": torch.as_tensor(
                    np.broadcast_to(
                        inputs.coordinate_valid, (batch, len(inputs.coordinate_valid))
                    ).copy(), device=device,
                ),
                "shaft_index": torch.as_tensor(
                    np.broadcast_to(
                        inputs.shaft_index, (batch, len(inputs.shaft_index))
                    ).copy(), dtype=torch.long, device=device,
                ),
            })
            result.append(value.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(result) if result else np.empty((0, model.state.dim), np.float32)


def materialize_inference_observation_embeddings(
        model: FullTargetObserverStateModel, inputs: InferenceAnchorInputs, *,
        device: str | torch.device = "cpu", batch_size: int = 256,
        ) -> np.ndarray:
    """Embed inference-only explicit windows with the frozen R1.6 observer."""
    freeze_and_assert(model)
    if model.use_raw:
        raise ValueError("R1.6 H2b extraction expects the frozen explicit observer arm")
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for lo in range(0, len(inputs.anchor_time_epoch), int(batch_size)):
            hi = min(lo + int(batch_size), len(inputs.anchor_time_epoch))
            explicit = np.array(inputs.explicit[lo:hi], copy=True)
            mask = np.array(inputs.contact_mask[lo:hi], copy=True)
            batch = hi - lo
            dummy = torch.zeros(
                (batch, explicit.shape[1], 1), dtype=torch.float32, device=device
            )
            value = model.observation_embedding({
                "explicit": torch.as_tensor(explicit, device=device),
                "waveform": dummy,
                "sample_valid": torch.ones_like(dummy, dtype=torch.bool),
                "contact_mask": torch.as_tensor(mask, device=device),
                "coordinates": torch.as_tensor(
                    np.broadcast_to(
                        inputs.coordinates, (batch, *inputs.coordinates.shape)
                    ).copy(), device=device,
                ),
                "coordinate_valid": torch.as_tensor(
                    np.broadcast_to(
                        inputs.coordinate_valid, (batch, len(inputs.coordinate_valid))
                    ).copy(), device=device,
                ),
                "shaft_index": torch.as_tensor(
                    np.broadcast_to(
                        inputs.shaft_index, (batch, len(inputs.shaft_index))
                    ).copy(), dtype=torch.long, device=device,
                ),
            })
            rows.append(value.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(rows) if rows else np.empty((0, 64), dtype=np.float32)


def explicit_observation_summary(explicit: np.ndarray,
                                 contact_mask: np.ndarray) -> np.ndarray:
    """Fixed-width masked mean+SD for a low-capacity seizure probe."""
    explicit = np.asarray(explicit, dtype=np.float32)
    contact_mask = np.asarray(contact_mask, dtype=bool)
    if explicit.ndim != 3 or contact_mask.shape != explicit.shape[:2]:
        raise ValueError("explicit/contact-mask arrays disagree")
    weight = contact_mask[..., None].astype(np.float64)
    count = weight.sum(1).clip(min=1.0)
    mean = (explicit.astype(np.float64) * weight).sum(1) / count
    delta = explicit.astype(np.float64) - mean[:, None, :]
    variance = delta ** 2
    variance = (variance * weight).sum(1) / count
    summary = np.concatenate([mean, np.sqrt(np.maximum(variance, 0.0))], axis=1)
    no_observation = ~contact_mask.any(1)
    summary[no_observation] = np.nan
    return summary.astype(np.float32)


def exact_deterministic_history(
        *, design: FullAnchorDesign, subject: str,
        history_scaler: Mapping[str, Sequence[float]],
        query_time_epoch: np.ndarray, query_continuity_session: np.ndarray,
        scaled: bool = True,
        ) -> np.ndarray:
    """Rebuild the frozen strictly-pre-query IED history at exact query times.

    ``FullAnchorDesign`` retains the admissible event times and marks needed by
    the deterministic baseline.  This avoids carrying the previous 30 s
    anchor's history forward as though it were exact at a seizure lead time.
    """
    group_ids = np.asarray(design.event_group_ids, dtype=np.int64)
    participation = group_ids >= 0
    n_contacts = int(group_ids.shape[1])
    stream = R1EventStream(
        subject=str(subject), dataset=str(subject).split("_", 1)[0],
        event_time=np.asarray(design.event_time, dtype=np.float64),
        split=np.asarray(design.event_split, dtype=np.int8),
        session=np.asarray(design.event_session, dtype=np.int64),
        participation=participation,
        group_ids=group_ids,
        group_count=np.asarray(design.event_group_count, dtype=np.int64),
        load=participation.mean(1).astype(np.float32),
        contact_names=np.asarray([f"c{index}" for index in range(n_contacts)]),
        contact_features=np.zeros((n_contacts, 1), dtype=np.float32),
        adjacency=np.zeros((1, n_contacts, n_contacts), dtype=np.float32),
        source_hashes={},
    )
    # Do not call ``R1EventStream.validate`` here: its final split check resolves
    # the split manifest relative to the H2b worktree, whereas these inputs are
    # deliberately read-only artifacts from the canonical source repo.  The
    # loaded ``FullAnchorDesign`` has already validated chronology, mark shapes,
    # source anchors and split support.  Recheck the fields used below locally.
    if np.any(np.diff(stream.event_time) < 0):
        raise ValueError("deterministic-history events are not chronological")
    if np.any(stream.group_ids[~stream.participation] != -1):
        raise ValueError("deterministic-history stream contains phantom group IDs")
    session_start = {
        int(label): float(start)
        for label, start in zip(design.session_label, design.session_start)
    }
    value = DeterministicHistory(stream, session_start).evaluate(
        np.asarray(query_time_epoch, dtype=np.float64),
        np.asarray(query_continuity_session, dtype=np.int64),
    )
    if not scaled:
        return value.astype(np.float32)
    scaler = HistoryScaler(
        mean=np.asarray(history_scaler["mean"], dtype=np.float32),
        scale=np.asarray(history_scaler["scale"], dtype=np.float32),
    )
    return scaler.transform(value)


@dataclass(frozen=True)
class CausalStateFeatures:
    anchor_time_epoch: np.ndarray
    coverage_segment_index: np.ndarray
    persistent_state: np.ndarray
    memoryless_observation_code: np.ndarray
    current_explicit_observation: np.ndarray
    current_contact_mask: np.ndarray
    current_explicit_summary: np.ndarray
    deterministic_history: np.ndarray
    last_observation_time_epoch: np.ndarray
    observation_age_seconds: np.ndarray
    observation_available: np.ndarray
    causal_observation_count: np.ndarray
    gap_reset: np.ndarray

    def validate(self) -> None:
        n = len(self.anchor_time_epoch)
        arrays = (
            self.coverage_segment_index, self.persistent_state,
            self.memoryless_observation_code, self.current_explicit_observation,
            self.current_contact_mask, self.current_explicit_summary,
            self.deterministic_history, self.last_observation_time_epoch,
            self.observation_age_seconds,
            self.observation_available, self.causal_observation_count, self.gap_reset,
        )
        if any(len(value) != n for value in arrays):
            raise ValueError("causal state feature arrays disagree")
        if self.anchor_time_epoch.dtype != np.float64:
            raise ValueError("absolute anchor time must be stored as float64")
        if self.coverage_segment_index.dtype != np.int64:
            raise ValueError("coverage segment must be stored as int64")
        if self.observation_age_seconds.dtype != np.float64:
            raise ValueError("observation age must be stored as float64")
        if np.any(self.observation_age_seconds[np.isfinite(
                self.observation_age_seconds)] < -1e-9):
            raise ValueError("current observation age cannot be negative")


def assert_anchor_outputs_bitwise_equal(reference: CausalStateFeatures,
                                        perturbed: CausalStateFeatures) -> None:
    """Instrument check: post-anchor perturbations cannot change anchor output."""
    fields = (
        "anchor_time_epoch", "coverage_segment_index", "persistent_state",
        "memoryless_observation_code", "current_explicit_observation",
        "current_contact_mask", "current_explicit_summary",
        "deterministic_history", "last_observation_time_epoch",
        "observation_age_seconds",
        "observation_available", "causal_observation_count", "gap_reset",
    )
    failures = []
    for name in fields:
        left = getattr(reference, name)
        right = getattr(perturbed, name)
        if not np.array_equal(left, right, equal_nan=True):
            failures.append(name)
    if failures:
        raise AssertionError(
            "post-anchor perturbation changed causal outputs: " + ", ".join(failures)
        )


def _segment_start_map(segment_start: Mapping[int, float] | Sequence[float],
                       segment_labels: np.ndarray) -> dict[int, float]:
    if isinstance(segment_start, Mapping):
        result = {int(key): float(value) for key, value in segment_start.items()}
    else:
        values = np.asarray(segment_start, dtype=np.float64)
        labels = np.unique(segment_labels)
        if len(values) != len(labels):
            raise ValueError("segment starts do not align with segment labels")
        result = {int(label): float(value) for label, value in zip(labels, values)}
    missing = set(np.unique(segment_labels).tolist()) - set(result)
    if missing:
        raise ValueError(f"missing segment starts for {sorted(missing)}")
    return result


def extract_causal_state_features(
        model: FullTargetObserverStateModel, *,
        observation_time_epoch: np.ndarray,
        observation_coverage_segment_index: np.ndarray,
        observation_embedding: np.ndarray,
        explicit_observation: np.ndarray,
        contact_mask: np.ndarray,
        anchor_time_epoch: np.ndarray,
        anchor_coverage_segment_index: np.ndarray,
        deterministic_history: np.ndarray,
        segment_start: Mapping[int, float] | Sequence[float],
        max_current_observation_age_seconds: float | None = None,
        ) -> CausalStateFeatures:
    """Extract exact ``z(t)`` using observations with time ``<= t`` only.

    State is reset to zero at every coverage-segment start.  Persistent and
    memoryless arms then share the same latest observation and autonomous flow
    to the exact query time.
    """
    freeze_and_assert(model)
    observation_time = np.asarray(observation_time_epoch, dtype=np.float64)
    observation_segment = np.asarray(
        observation_coverage_segment_index, dtype=np.int64
    )
    embedding = np.asarray(observation_embedding, dtype=np.float32)
    explicit = np.asarray(explicit_observation, dtype=np.float32)
    mask = np.asarray(contact_mask, dtype=bool)
    query_time = np.asarray(anchor_time_epoch, dtype=np.float64)
    query_segment = np.asarray(anchor_coverage_segment_index, dtype=np.int64)
    history = np.asarray(deterministic_history, dtype=np.float32)
    if any(value.ndim != 1 for value in (
            observation_time, observation_segment, query_time, query_segment)):
        raise ValueError("time and segment arrays must be one dimensional")
    if len(observation_time) != len(observation_segment) or len(embedding) != len(observation_time):
        raise ValueError("observation arrays disagree")
    if len(explicit) != len(observation_time) or mask.shape != explicit.shape[:2]:
        raise ValueError("explicit observation arrays disagree")
    if len(query_time) != len(query_segment) or len(history) != len(query_time):
        raise ValueError("query arrays disagree")
    if not np.isfinite(observation_time).all() or not np.isfinite(query_time).all():
        raise ValueError("absolute times must be finite")
    if np.any(np.diff(observation_time) < 0):
        raise ValueError("observations must be globally chronological")
    starts = _segment_start_map(segment_start, np.concatenate([
        observation_segment, query_segment
    ]))
    for label in np.unique(observation_segment):
        local = observation_time[observation_segment == label]
        if np.any(np.diff(local) < 0) or np.any(local < starts[int(label)]):
            raise ValueError(f"segment {label}: invalid observation chronology")
    for label in np.unique(query_segment):
        local = query_time[query_segment == label]
        if np.any(local < starts[int(label)]):
            raise ValueError(f"segment {label}: query precedes coverage start")

    n_query = len(query_time)
    state_dim = int(model.state.dim)
    contacts, features = explicit.shape[1:]
    persistent = np.full((n_query, state_dim), np.nan, dtype=np.float32)
    memoryless = np.full_like(persistent, np.nan)
    current_explicit = np.full((n_query, contacts, features), np.nan, dtype=np.float32)
    current_mask = np.zeros((n_query, contacts), dtype=bool)
    latest_time = np.full(n_query, np.nan, dtype=np.float64)
    observation_age = np.full(n_query, np.nan, dtype=np.float64)
    available = np.zeros(n_query, dtype=bool)
    count = np.zeros(n_query, dtype=np.int64)
    gap_reset = np.ones(n_query, dtype=bool)
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype

    with torch.inference_mode():
        for label in np.unique(query_segment):
            query_ids = np.flatnonzero(query_segment == label)
            query_order = query_ids[np.argsort(query_time[query_ids], kind="stable")]
            observation_ids = np.flatnonzero(observation_segment == label)
            observation_ids = observation_ids[
                np.argsort(observation_time[observation_ids], kind="stable")
            ]
            state = torch.zeros(state_dim, dtype=dtype, device=device)
            cursor = float(starts[int(label)])
            position = 0
            for query_id in query_order:
                target = float(query_time[query_id])
                while (position < len(observation_ids)
                       and float(observation_time[observation_ids[position]]) <= target):
                    observation_id = int(observation_ids[position])
                    time = float(observation_time[observation_id])
                    state = model.state.generator.propagate(
                        state, (time - cursor) / 60.0
                    )
                    state = model.state.correction(
                        state,
                        torch.as_tensor(embedding[observation_id], dtype=dtype, device=device),
                        enabled=True,
                    )
                    cursor = time
                    position += 1
                exact = model.state.generator.propagate(
                    state, (target - cursor) / 60.0
                )
                persistent[query_id] = exact.detach().cpu().numpy().astype(np.float32)
                count[query_id] = position
                if position:
                    observation_id = int(observation_ids[position - 1])
                    source_time = float(observation_time[observation_id])
                    age = target - source_time
                    if age < -1e-9:
                        raise AssertionError("current observation follows query anchor")
                    latest_time[query_id] = source_time
                    observation_age[query_id] = max(age, 0.0)
                    fresh = (
                        max_current_observation_age_seconds is None
                        or age <= float(max_current_observation_age_seconds) + 1e-9
                    )
                    if not fresh:
                        continue
                    code = model.state.correction(
                        model.state.generator.mu,
                        torch.as_tensor(embedding[observation_id], dtype=dtype, device=device),
                        enabled=True,
                    )
                    code = model.state.generator.propagate(
                        code, (target - source_time) / 60.0
                    )
                    memoryless[query_id] = code.detach().cpu().numpy().astype(np.float32)
                    current_explicit[query_id] = explicit[observation_id]
                    current_mask[query_id] = mask[observation_id]
                    available[query_id] = True
                    gap_reset[query_id] = position == 1

    summary = explicit_observation_summary(current_explicit, current_mask)
    result = CausalStateFeatures(
        anchor_time_epoch=query_time.astype(np.float64, copy=False),
        coverage_segment_index=query_segment.astype(np.int64, copy=False),
        persistent_state=persistent,
        memoryless_observation_code=memoryless,
        current_explicit_observation=current_explicit,
        current_contact_mask=current_mask,
        current_explicit_summary=summary,
        deterministic_history=history,
        last_observation_time_epoch=latest_time,
        observation_age_seconds=observation_age,
        observation_available=available,
        causal_observation_count=count,
        gap_reset=gap_reset,
    )
    result.validate()
    return result


def wrong_time_confounders(history_unscaled: np.ndarray,
                           explicit_summary: np.ndarray) -> np.ndarray:
    """Confounders for soft within-segment wrong-time donor ranking."""
    history = np.asarray(history_unscaled, dtype=np.float64)
    explicit_summary = np.asarray(explicit_summary, dtype=np.float64)
    if history.ndim != 2 or explicit_summary.ndim != 2 or len(history) != len(explicit_summary):
        raise ValueError("wrong-time confounder arrays disagree")
    index = {name: BASE_HISTORY_NAMES.index(name) for name in (
        "log_time_since_previous_event", "count_trace_30s", "count_trace_2m",
        "count_trace_10m", "tod_sin", "tod_cos", "log_session_elapsed_minutes",
    )}
    valid_fraction_mean = explicit_summary[:, explicit_summary.shape[1] // 2 - 1]
    return np.column_stack([
        history[:, index["tod_sin"]], history[:, index["tod_cos"]],
        history[:, index["log_session_elapsed_minutes"]],
        history[:, index["log_time_since_previous_event"]],
        history[:, index["count_trace_30s"]], history[:, index["count_trace_2m"]],
        history[:, index["count_trace_10m"]], valid_fraction_mean,
    ]).astype(np.float32)


@dataclass(frozen=True)
class WrongTimeCandidates:
    donor_index: np.ndarray
    donor_time_epoch: np.ndarray
    donor_state: np.ndarray
    donor_confounders: np.ndarray
    distance: np.ndarray
    valid: np.ndarray


def build_wrong_time_candidates(
        *, target_time_epoch: np.ndarray, target_segment: np.ndarray,
        target_confounders: np.ndarray, donor_time_epoch: np.ndarray,
        donor_segment: np.ndarray, donor_state: np.ndarray,
        donor_confounders: np.ndarray, n_donors: int = 20,
        min_separation_seconds: float = 1800.0,
        global_exclusion_intervals: Iterable[tuple[float, float]] = (),
        target_exclusion_start: np.ndarray | None = None,
        target_exclusion_stop: np.ndarray | None = None,
        ) -> WrongTimeCandidates:
    """Soft-rank wrong-time donors in the same recorded coverage segment.

    No multivariate caliper is applied.  Matching variables determine only a
    deterministic distance order and remain available for explicit probe
    adjustment downstream.
    """
    target_time = np.asarray(target_time_epoch, dtype=np.float64)
    target_segment = np.asarray(target_segment, dtype=np.int64)
    target_cov = np.asarray(target_confounders, dtype=np.float64)
    donor_time = np.asarray(donor_time_epoch, dtype=np.float64)
    donor_segment = np.asarray(donor_segment, dtype=np.int64)
    donor_state = np.asarray(donor_state, dtype=np.float32)
    donor_cov = np.asarray(donor_confounders, dtype=np.float64)
    if len(target_time) != len(target_segment) or len(target_cov) != len(target_time):
        raise ValueError("wrong-time target arrays disagree")
    if len(donor_time) != len(donor_segment) or len(donor_state) != len(donor_time):
        raise ValueError("wrong-time donor arrays disagree")
    if len(donor_cov) != len(donor_time) or donor_cov.shape[1] != target_cov.shape[1]:
        raise ValueError("wrong-time confounder dimensions disagree")
    k = int(n_donors)
    if k < 1:
        raise ValueError("n_donors must be positive")
    if target_exclusion_start is None:
        target_exclusion_start = np.full(len(target_time), np.nan)
    if target_exclusion_stop is None:
        target_exclusion_stop = np.full(len(target_time), np.nan)
    target_exclusion_start = np.asarray(target_exclusion_start, dtype=np.float64)
    target_exclusion_stop = np.asarray(target_exclusion_stop, dtype=np.float64)
    if len(target_exclusion_start) != len(target_time) or len(target_exclusion_stop) != len(target_time):
        raise ValueError("target exclusion arrays disagree")
    globally_allowed = np.ones(len(donor_time), dtype=bool)
    for left, right in global_exclusion_intervals:
        if not float(left) <= float(right):
            raise ValueError("global exclusion interval is reversed")
        globally_allowed &= ~((donor_time >= float(left)) & (donor_time <= float(right)))

    out_index = np.full((len(target_time), k), -1, dtype=np.int64)
    out_time = np.full((len(target_time), k), np.nan, dtype=np.float64)
    out_state = np.full((len(target_time), k, donor_state.shape[1]), np.nan, np.float32)
    out_cov = np.full((len(target_time), k, donor_cov.shape[1]), np.nan, np.float32)
    out_distance = np.full((len(target_time), k), np.nan, np.float32)
    valid = np.zeros((len(target_time), k), dtype=bool)
    for row, (time, segment) in enumerate(zip(target_time, target_segment)):
        eligible = globally_allowed & (donor_segment == segment)
        eligible &= np.abs(donor_time - time) >= float(min_separation_seconds)
        left, right = target_exclusion_start[row], target_exclusion_stop[row]
        if np.isfinite(left) and np.isfinite(right):
            if left > right:
                raise ValueError("target exclusion interval is reversed")
            eligible &= ~((donor_time >= left) & (donor_time <= right))
        candidate = np.flatnonzero(eligible & np.isfinite(donor_cov).all(1))
        if not len(candidate) or not np.isfinite(target_cov[row]).all():
            continue
        scale = np.std(donor_cov[candidate], axis=0)
        scale = np.where(scale > 1e-6, scale, 1.0)
        delta = (donor_cov[candidate] - target_cov[row]) / scale
        distance = np.sqrt(np.sum(delta ** 2, axis=1))
        order = np.lexsort((donor_time[candidate], distance))[:k]
        selected = candidate[order]
        width = len(selected)
        out_index[row, :width] = selected
        out_time[row, :width] = donor_time[selected]
        out_state[row, :width] = donor_state[selected]
        out_cov[row, :width] = donor_cov[selected]
        out_distance[row, :width] = distance[order]
        valid[row, :width] = True
    return WrongTimeCandidates(
        donor_index=out_index, donor_time_epoch=out_time,
        donor_state=out_state, donor_confounders=out_cov,
        distance=out_distance, valid=valid,
    )


def atomic_state_cache(path: str | Path, *, features: CausalStateFeatures,
                       query_id: np.ndarray, provenance: dict,
                       wrong_time: WrongTimeCandidates | None = None) -> dict:
    """Atomically write an array cache followed by a hash-bound manifest."""
    target = h2b_contract.assert_safe_output_path(path)
    if target.suffix != ".npz":
        raise ValueError("state cache path must end in .npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".npz.tmp")
    max_source_time = features.last_observation_time_epoch.astype(np.float64)
    causal = (~np.isfinite(max_source_time)) | (
        max_source_time <= features.anchor_time_epoch
    )
    if not bool(causal.all()):
        raise ValueError("state cache contains a source observation after its anchor")
    required_provenance = (
        "checkpoint_sha256", "checkpoint_result_sha256", "source_hashes",
        "current_observation_max_age_seconds",
    )
    missing = [key for key in required_provenance if not provenance.get(key)]
    if missing:
        raise ValueError(f"state cache provenance lacks {missing}")
    freshness_limit = float(provenance["current_observation_max_age_seconds"])
    if not bool(features.observation_available.all()):
        raise ValueError("state cache contains a query without a fresh current observation")
    if bool(np.any(features.observation_age_seconds > freshness_limit + 1e-9)):
        raise ValueError("state cache contains a stale current observation")
    arrays = {
        "query_id": np.asarray(query_id).astype(str),
        "anchor_time_epoch": features.anchor_time_epoch.astype(np.float64),
        "coverage_segment_index": features.coverage_segment_index.astype(np.int64),
        "persistent_state": features.persistent_state.astype(np.float32),
        "memoryless_observation_code": features.memoryless_observation_code.astype(np.float32),
        "current_explicit_observation": features.current_explicit_observation.astype(np.float32),
        "current_contact_mask": features.current_contact_mask.astype(bool),
        "current_explicit_summary": features.current_explicit_summary.astype(np.float32),
        "deterministic_history": features.deterministic_history.astype(np.float32),
        "last_observation_time_epoch": features.last_observation_time_epoch.astype(np.float64),
        "max_source_time_epoch": max_source_time,
        "observation_age_seconds": features.observation_age_seconds.astype(np.float64),
        "observation_available": features.observation_available.astype(bool),
        "causal_observation_count": features.causal_observation_count.astype(np.int64),
        "gap_reset": features.gap_reset.astype(bool),
    }
    if len(arrays["query_id"]) != len(features.anchor_time_epoch):
        raise ValueError("query IDs do not align with state features")
    if wrong_time is not None:
        arrays.update({
            "wrong_time_donor_index": wrong_time.donor_index,
            "wrong_time_donor_time_epoch": wrong_time.donor_time_epoch.astype(np.float64),
            "wrong_time_donor_state": wrong_time.donor_state.astype(np.float32),
            "wrong_time_donor_confounders": wrong_time.donor_confounders.astype(np.float32),
            "wrong_time_distance": wrong_time.distance.astype(np.float32),
            "wrong_time_valid": wrong_time.valid.astype(bool),
        })
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, target)
    manifest = {
        **provenance,
        "status": "COMPLETE",
        "state_extraction_revision": H2B_STATE_EXTRACTION_REVISION,
        "cache": str(target.resolve()),
        "cache_sha256": sha256_file(target),
        "n_queries": int(len(features.anchor_time_epoch)),
        "absolute_time_dtype": "float64",
        "time_dtype": "float64",
        "segment_dtype": "int64",
        "coverage_segment_semantics": "unique_coverage_table_row_index",
        "state_reset_at_every_coverage_segment_row": True,
        "anchor_rule": "observations_with_time_less_than_or_equal_to_anchor_only",
        "max_source_time_le_anchor": True,
        "current_observation_max_age_seconds": freshness_limit,
        "all_current_observations_fresh": True,
        "gap_policy": "reset_at_recorded_coverage_segment_start",
        "gap_reset": True,
        "state_frozen": True,
        "all_parameters_frozen": True,
        "seizure_gradient_path": False,
        "formal": False,
        "sealed": False,
    }
    manifest_path = target.with_suffix(".manifest.json")
    h2b_contract.atomic_json(manifest_path, manifest)
    return manifest
