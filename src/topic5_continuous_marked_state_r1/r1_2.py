"""R1.2 full-recorded-support state experiment with frozen observers.

The sampled R1 pilot scored only the 30 s support immediately following a
selected raw anchor.  R1.2 instead keeps every TRAIN/validation event and every
recorded-time quadrature row.  Raw observations correct the state when they are
available; between observations the same state evolves autonomously.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from . import contract
from .baseline import ExactHistoryMarkDecoder, HistoryIntensity
from .bridge_e1 import (
    BridgeE1Design,
    build_bridge_e1_design,
    make_paired_models,
)
from .coverage import CoverageTable, merge_labeled_intervals
from .data import R1EventStream, load_event_stream
from .design import _quadrature_grid
from .history import DeterministicHistory, HistoryScaler, session_start_map
from .mark_likelihood import tied_group_mark_log_prob
from .raw_observation import RawAnchorReader
from .state import ControlledPersistentState


R1_2_REVISION = "r1_2_full_recorded_support_inventory_coverage_v2"
CACHE_REVISION = "r1_2_all_admissible_development_anchors_dual_embedding_v2"
FULL_STREAM_REVISION = "r1_2_preictal_admissible_full_event_stream_v1"
FULL_COVERAGE_REVISION = "r1_2_full_block_inventory_minus_ictal_postictal_v1"


def _subtract_intervals(start: np.ndarray, stop: np.ndarray,
                        label: np.ndarray, cut_start: np.ndarray,
                        cut_stop: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out_start: list[float] = []
    out_stop: list[float] = []
    out_label: list[int] = []
    cuts = sorted(zip(np.asarray(cut_start, dtype=float), np.asarray(cut_stop, dtype=float)))
    for left, right, group in zip(start, stop, label):
        pieces = [(float(left), float(right))]
        for cut_left, cut_right in cuts:
            if cut_right <= left or cut_left >= right:
                continue
            next_pieces = []
            for piece_left, piece_right in pieces:
                if cut_right <= piece_left or cut_left >= piece_right:
                    next_pieces.append((piece_left, piece_right))
                    continue
                if piece_left < cut_left:
                    next_pieces.append((piece_left, min(piece_right, cut_left)))
                if cut_right < piece_right:
                    next_pieces.append((max(piece_left, cut_right), piece_right))
            pieces = next_pieces
        for piece_left, piece_right in pieces:
            if piece_right > piece_left:
                out_start.append(piece_left)
                out_stop.append(piece_right)
                out_label.append(int(group))
    return (
        np.asarray(out_start, dtype=np.float64),
        np.asarray(out_stop, dtype=np.float64),
        np.asarray(out_label, dtype=np.int64),
    )


def build_full_admissible_coverage(subject: str) -> tuple[CoverageTable, dict]:
    """All metadata-recorded development time, minus ictal and 2 h post-ictal."""
    import pandas as pd
    from src.topic5_epi_prssm.preictal_stream import POST_ICTAL_GUARD_SECONDS
    from src.topic5_epi_prssm.seizure_labels import load_seizures

    manifest_path = contract.RAW_STATE_ROOT / "data/dataset_manifest.parquet"
    blocks = pd.read_parquet(manifest_path)
    blocks = blocks[blocks.subject.astype(str) == subject].sort_values("block_start_epoch")
    if blocks.empty:
        raise ValueError(f"{subject}: no full block inventory rows")
    train_end, dev_end = contract.load_split(subject)
    raw_start = blocks.block_start_epoch.to_numpy(dtype=np.float64)
    raw_stop = np.minimum(blocks.block_end_epoch.to_numpy(dtype=np.float64), dev_end)
    raw_session = blocks.session_id.to_numpy(dtype=np.int64)
    keep = raw_stop > raw_start
    start, stop, session = merge_labeled_intervals(
        raw_start[keep], raw_stop[keep], raw_session[keep]
    )
    seizures = [value for value in load_seizures(subject) if value.onset_epoch < dev_end]
    cut_start = np.asarray([value.onset_epoch for value in seizures], dtype=np.float64)
    cut_stop = np.asarray([
        value.offset_epoch + POST_ICTAL_GUARD_SECONDS for value in seizures
    ], dtype=np.float64)
    start, stop, session = _subtract_intervals(
        start, stop, session, cut_start, cut_stop
    )
    # A seizure is an unmodelled intervention in T1.  The first admissible
    # segment after it must therefore open a new latent/history session; carrying
    # the pre-seizure state through the excluded ictal/postictal interval would
    # silently assert that the seizure had no state effect.  Ordinary metadata
    # gaps retain their upstream continuity label.
    seizure_generation = np.searchsorted(
        np.sort(cut_start), start, side="left"
    ).astype(np.int64)
    continuity = np.zeros(len(start), dtype=np.int64)
    next_label = 0
    for index in range(1, len(start)):
        if (session[index] != session[index - 1]
                or seizure_generation[index] != seizure_generation[index - 1]):
            next_label += 1
        continuity[index] = next_label
    session = continuity
    value = CoverageTable(
        subject=subject, start=start, stop=stop, session=session,
        train_end_epoch=train_end, dev_end_epoch=dev_end,
        source_hashes={
            "raw_state_dataset_manifest": contract.sha256_file(manifest_path),
            "full_event_stream": contract.sha256_file(
                contract.UPSTREAM_ROOT / "full_event_stream/per_subject" / f"{subject}.npz"
            ),
        },
    )
    value.validate()
    meta = {
        "status": "COMPLETE",
        "contract": contract.REVISION,
        "r1_2_revision": R1_2_REVISION,
        "coverage_revision": FULL_COVERAGE_REVISION,
        "subject": subject,
        "n_source_blocks": int(len(blocks)),
        "n_seizures_before_dev_end": int(len(seizures)),
        "postictal_guard_seconds": float(POST_ICTAL_GUARD_SECONDS),
        "preictal_excluded": False,
        "n_coverage_segments": int(len(start)),
        "n_continuity_sessions": int(len(np.unique(session))),
        "state_reset_after_ictal_postictal_exclusion": True,
        "train_recorded_seconds": float(sum(
            max(0.0, min(float(right), train_end) - float(left))
            for left, right in zip(start, stop)
        )),
        "validation_recorded_seconds": float(sum(
            max(0.0, min(float(right), dev_end) - max(float(left), train_end))
            for left, right in zip(start, stop)
        )),
        "source_hashes": value.source_hashes,
        "sealed_opened": False,
    }
    return value, meta


def write_full_admissible_coverage(subject: str, output_root: Path | None = None) -> dict:
    output_root = output_root or contract.RESULT_ROOT / "r1_2"
    coverage, meta = build_full_admissible_coverage(subject)
    path = Path(output_root) / "coverage" / f"{subject}.npz"
    coverage.save(path)
    meta["output"] = str(path)
    meta["output_sha256"] = contract.sha256_file(path)
    contract.atomic_json(path.with_suffix(".manifest.json"), meta)
    return meta


def load_full_admissible_event_stream(subject: str,
                                      coverage: CoverageTable | None = None
                                      ) -> R1EventStream:
    """Full reconstructed IED stream, retaining preictal and dropping ictal/post."""
    from src.topic5_epi_prssm.preictal_stream import build_admissible_stream
    from src.topic5_epi_prssm.seizure_labels import load_seizures

    coverage = coverage or CoverageTable.load(
        contract.RESULT_ROOT / "r1_2/coverage" / f"{subject}.npz"
    )
    seizures = load_seizures(subject)
    admissible = build_admissible_stream(
        subject, [(value.onset_epoch, value.offset_epoch) for value in seizures]
    )
    frozen = load_event_stream(subject)
    times = np.asarray(admissible.event_time, dtype=np.float64)
    keep = times < coverage.dev_end_epoch
    times = times[keep]
    split = np.where(times < coverage.train_end_epoch, 0, 1).astype(np.int8)
    session = np.full(len(times), -1, dtype=np.int64)
    for left, right, label in zip(coverage.start, coverage.stop, coverage.session):
        hit = (times >= left) & (times < right)
        if np.any(session[hit] >= 0):
            raise ValueError(f"{subject}: full-stream event maps to overlapping coverage")
        session[hit] = int(label)
    if np.any(session < 0):
        bad = times[session < 0]
        raise ValueError(
            f"{subject}: {len(bad)} admissible full-stream events outside coverage"
        )
    tensors = admissible.tensors
    participation = tensors.participation.detach().cpu().numpy().astype(bool)[keep]
    group_ids = tensors.group_ids.detach().cpu().numpy().astype(np.int64)[keep]
    group_count = tensors.n_groups.detach().cpu().numpy().astype(np.int64)[keep]
    load = tensors.load.detach().cpu().numpy().astype(np.float32)[keep]
    full_path = contract.UPSTREAM_ROOT / "full_event_stream/per_subject" / f"{subject}.npz"
    value = R1EventStream(
        subject=subject, dataset=frozen.dataset,
        event_time=times, split=split, session=session,
        participation=participation, group_ids=group_ids,
        group_count=group_count, load=load,
        contact_names=frozen.contact_names,
        contact_features=frozen.contact_features,
        adjacency=frozen.adjacency,
        source_hashes={
            **frozen.source_hashes,
            "full_event_stream": contract.sha256_file(full_path),
            "r1_2_coverage": contract.sha256_file(
                contract.RESULT_ROOT / "r1_2/coverage" / f"{subject}.npz"
            ),
        },
    )
    value.validate()
    return value


@dataclass(frozen=True)
class FullAnchorDesign:
    subject: str
    anchor_time: np.ndarray
    anchor_split: np.ndarray
    anchor_session: np.ndarray
    anchor_history: np.ndarray
    event_time: np.ndarray
    event_split: np.ndarray
    event_session: np.ndarray
    event_source_anchor: np.ndarray
    event_history: np.ndarray
    event_group_ids: np.ndarray
    event_group_count: np.ndarray
    quadrature_time: np.ndarray
    quadrature_split: np.ndarray
    quadrature_session: np.ndarray
    quadrature_source_anchor: np.ndarray
    quadrature_history: np.ndarray
    quadrature_weight_seconds: np.ndarray
    session_label: np.ndarray
    session_start: np.ndarray

    def validate(self) -> None:
        n_anchor = len(self.anchor_time)
        if any(len(value) != n_anchor for value in (
            self.anchor_split, self.anchor_session, self.anchor_history,
        )):
            raise ValueError("R1.2 anchor arrays disagree")
        n_event = len(self.event_time)
        if any(len(value) != n_event for value in (
            self.event_split, self.event_session, self.event_source_anchor,
            self.event_history, self.event_group_ids, self.event_group_count,
        )):
            raise ValueError("R1.2 event arrays disagree")
        n_q = len(self.quadrature_time)
        if any(len(value) != n_q for value in (
            self.quadrature_split, self.quadrature_session,
            self.quadrature_source_anchor, self.quadrature_history,
            self.quadrature_weight_seconds,
        )):
            raise ValueError("R1.2 quadrature arrays disagree")
        if not np.all(np.diff(self.anchor_time) >= 0):
            raise ValueError("R1.2 anchors are not chronological")
        if not np.all(np.diff(self.event_time) >= 0):
            raise ValueError("R1.2 events are not chronological")
        if np.any(self.quadrature_weight_seconds <= 0):
            raise ValueError("R1.2 quadrature contains non-positive weights")
        if set(np.unique(self.anchor_split).tolist()) != {0, 1}:
            raise ValueError("R1.2 requires TRAIN and validation anchors")
        if set(np.unique(self.event_split).tolist()) != {0, 1}:
            raise ValueError("R1.2 requires TRAIN and validation events")
        for source, time, session, name in (
            (self.event_source_anchor, self.event_time, self.event_session, "event"),
            (self.quadrature_source_anchor, self.quadrature_time,
             self.quadrature_session, "quadrature"),
        ):
            if np.any(source < -1) or np.any(source >= n_anchor):
                raise ValueError(f"R1.2 {name} source anchor out of range")
            observed = source >= 0
            if np.any(self.anchor_time[source[observed]] > time[observed] + 1e-9):
                raise ValueError(f"R1.2 {name} uses a future anchor")
            if np.any(self.anchor_session[source[observed]] != session[observed]):
                raise ValueError(f"R1.2 {name} crosses sessions")
        if not np.isfinite(self.anchor_history).all():
            raise ValueError("R1.2 anchor history is non-finite")
        if not np.isfinite(self.event_history).all():
            raise ValueError("R1.2 event history is non-finite")
        if not np.isfinite(self.quadrature_history).all():
            raise ValueError("R1.2 quadrature history is non-finite")
        if not np.array_equal(np.unique(self.session_label), self.session_label):
            raise ValueError("R1.2 session labels must be unique and sorted")
        if len(self.session_label) != len(self.session_start):
            raise ValueError("R1.2 session arrays disagree")

    def anchor_ids(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.anchor_split == {"train": 0, "validation": 1}[split])

    def session_start_for(self, labels: np.ndarray) -> np.ndarray:
        position = np.searchsorted(self.session_label, labels)
        if np.any(position >= len(self.session_label)):
            raise ValueError("unknown R1.2 session")
        if np.any(self.session_label[position] != labels):
            raise ValueError("unknown R1.2 session")
        return self.session_start[position]


def _latest_anchor_source(query_time: np.ndarray, query_session: np.ndarray,
                          anchor_time: np.ndarray, anchor_session: np.ndarray
                          ) -> np.ndarray:
    result = np.full(len(query_time), -1, dtype=np.int64)
    for label in np.unique(query_session):
        query = np.flatnonzero(query_session == label)
        anchor = np.flatnonzero(anchor_session == label)
        if not len(anchor):
            continue
        order = anchor[np.argsort(anchor_time[anchor], kind="stable")]
        # Event likelihood is evaluated from z(t-).  An observation ending at
        # exactly the event time is only assimilated after that event and cannot
        # be its own source state.
        position = np.searchsorted(anchor_time[order], query_time[query], side="left") - 1
        valid = position >= 0
        result[query[valid]] = order[position[valid]]
    return result


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def save_full_design(path: Path, design: FullAnchorDesign) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    arrays = {
        field.name: getattr(design, field.name)
        for field in fields(FullAnchorDesign) if field.name != "subject"
    }
    arrays["subject"] = np.asarray(design.subject)
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def load_full_design(path: Path) -> FullAnchorDesign:
    with np.load(path, allow_pickle=False) as data:
        kwargs = {
            field.name: data[field.name]
            for field in fields(FullAnchorDesign) if field.name != "subject"
        }
        value = FullAnchorDesign(subject=str(data["subject"].item()), **kwargs)
    value.validate()
    return value


def _bridge_scaler(subject: str, baseline_path: Path, bridge_result: dict,
                   stream: R1EventStream, coverage: CoverageTable,
                   ) -> tuple[np.ndarray, np.ndarray, BridgeE1Design, RawAnchorReader]:
    frozen_scaler = (
        np.asarray(bridge_result["explicit_mean"], dtype=np.float32),
        np.asarray(bridge_result["explicit_scale"], dtype=np.float32),
    )
    sampled, reader, _ = build_bridge_e1_design(
        subject, baseline_path,
        max_train_anchors=int(bridge_result["n_train_anchors"]),
        max_validation_anchors=int(bridge_result["n_validation_anchors"]),
        quadrature_order=4,
        stream=stream,
        coverage=coverage,
        explicit_scaler=frozen_scaler,
    )
    return sampled.explicit_mean, sampled.explicit_scale, sampled, reader


def _candidate_anchors(reader: RawAnchorReader, coverage: CoverageTable
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time, split, _ = reader.anchor_times()
    keep_time: list[float] = []
    keep_split: list[int] = []
    keep_session: list[int] = []
    for value, split_code in zip(time, split):
        if not reader.can_read(float(value)):
            continue
        match = np.flatnonzero((coverage.start <= value) & (value < coverage.stop))
        if len(match) != 1:
            continue
        keep_time.append(float(value))
        keep_split.append(int(split_code))
        keep_session.append(int(coverage.session[match[0]]))
    return (
        np.asarray(keep_time, dtype=np.float64),
        np.asarray(keep_split, dtype=np.int8),
        np.asarray(keep_session, dtype=np.int64),
    )


def _observer_batch(observations: list, explicit: np.ndarray,
                    *, device: torch.device | str) -> dict[str, torch.Tensor]:
    batch = len(observations)
    coordinates = observations[0].coordinates
    coordinate_valid = observations[0].coordinate_valid
    shaft_index = observations[0].shaft_index
    return {
        "explicit": torch.as_tensor(explicit, device=device),
        "waveform": torch.as_tensor(
            np.stack([value.waveform for value in observations]), device=device
        ),
        "sample_valid": torch.as_tensor(
            np.stack([value.sample_valid for value in observations]), device=device
        ),
        "contact_mask": torch.as_tensor(
            np.stack([value.contact_mask for value in observations]), device=device
        ),
        "coordinates": torch.as_tensor(
            np.broadcast_to(coordinates, (batch, *coordinates.shape)).copy(),
            device=device,
        ),
        "coordinate_valid": torch.as_tensor(
            np.broadcast_to(coordinate_valid, (batch, len(coordinate_valid))).copy(),
            device=device,
        ),
        "shaft_index": torch.as_tensor(
            np.broadcast_to(shaft_index, (batch, len(shaft_index))).copy(),
            dtype=torch.long, device=device,
        ),
    }


def build_full_anchor_cache(subject: str, *, device: torch.device | str = "cuda",
                            anchor_batch_size: int = 8,
                            output_root: Path | None = None) -> dict:
    """Freeze the paired Bridge observers and cache both embeddings once."""
    output_root = output_root or contract.RESULT_ROOT / "r1_2"
    baseline_path = Path(output_root) / "baselines" / subject / "seed_0/models.pt"
    bridge_dir = Path(output_root) / "bridge_e1" / subject / "seed_0"
    bridge_result_path = bridge_dir / "result.json"
    bridge_checkpoint_path = bridge_dir / "models.pt"
    for path in (baseline_path, bridge_result_path, bridge_checkpoint_path):
        if not path.exists():
            raise FileNotFoundError(path)
    bridge_result = json.loads(bridge_result_path.read_text())
    if bridge_result.get("status") != "COMPLETE":
        raise ValueError(f"{subject}: Bridge-E1 is incomplete")
    if bridge_result.get("contract") != contract.REVISION:
        raise ValueError(f"{subject}: Bridge-E1 contract mismatch")
    if bridge_result.get("r1_2_revision") != R1_2_REVISION:
        raise ValueError(f"{subject}: Bridge-E1 R1.2 revision mismatch")
    if bridge_result.get("sealed_opened") is not False:
        raise ValueError(f"{subject}: Bridge-E1 sealed flag is not false")
    if contract.sha256_file(bridge_checkpoint_path) != bridge_result["checkpoint_sha256"]:
        raise ValueError(f"{subject}: Bridge-E1 checkpoint hash mismatch")

    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    coverage_path = Path(output_root) / "coverage" / f"{subject}.npz"
    coverage = CoverageTable.load(coverage_path)
    stream = load_full_admissible_event_stream(subject, coverage)
    explicit_mean, explicit_scale, sampled, reader = _bridge_scaler(
        subject, baseline_path, bridge_result, stream, coverage
    )
    explicit_model, raw_model = make_paired_models(
        baseline, sampled, stream.adjacency, seed=0, device=device
    )
    bridge_checkpoint = torch.load(
        bridge_checkpoint_path, map_location=device, weights_only=False
    )
    explicit_model.load_state_dict(bridge_checkpoint["explicit"])
    raw_model.load_state_dict(bridge_checkpoint["explicit_raw"])
    explicit_model.eval()
    raw_model.eval()
    for model in (explicit_model, raw_model):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    candidate_time, candidate_split, candidate_session = _candidate_anchors(reader, coverage)
    kept_time: list[np.ndarray] = []
    kept_split: list[np.ndarray] = []
    kept_session: list[np.ndarray] = []
    explicit_embedding: list[np.ndarray] = []
    raw_embedding: list[np.ndarray] = []
    unreadable = 0
    with torch.inference_mode():
        for lo in range(0, len(candidate_time), int(anchor_batch_size)):
            hi = min(lo + int(anchor_batch_size), len(candidate_time))
            observations = []
            local = []
            for index in range(lo, hi):
                value = reader.read(float(candidate_time[index]))
                if value is None:
                    unreadable += 1
                    continue
                observations.append(value)
                local.append(index)
            if not observations:
                continue
            explicit = np.stack([value.explicit for value in observations])
            explicit = ((explicit - explicit_mean) / explicit_scale).astype(np.float32)
            batch = _observer_batch(observations, explicit, device=device)
            explicit_embedding.append(
                explicit_model.observer(**batch).detach().cpu().numpy().astype(np.float32)
            )
            raw_embedding.append(
                raw_model.observer(**batch).detach().cpu().numpy().astype(np.float32)
            )
            local = np.asarray(local, dtype=np.int64)
            kept_time.append(candidate_time[local])
            kept_split.append(candidate_split[local])
            kept_session.append(candidate_session[local])
    if not kept_time:
        raise ValueError(f"{subject}: no readable full anchors")
    anchor_time = np.concatenate(kept_time)
    anchor_split = np.concatenate(kept_split)
    anchor_session = np.concatenate(kept_session)
    explicit_embedding_array = np.concatenate(explicit_embedding)
    raw_embedding_array = np.concatenate(raw_embedding)
    if not np.isfinite(explicit_embedding_array).all():
        raise ValueError(f"{subject}: explicit embedding is non-finite")
    if not np.isfinite(raw_embedding_array).all():
        raise ValueError(f"{subject}: raw embedding is non-finite")

    scaler = HistoryScaler(
        mean=np.asarray(baseline["history_scaler"]["mean"], dtype=np.float32),
        scale=np.asarray(baseline["history_scaler"]["scale"], dtype=np.float32),
    )
    starts = session_start_map(stream, coverage.session, coverage.start)
    history = DeterministicHistory(stream, starts)
    event_keep = stream.split < 2
    event_time = stream.event_time[event_keep]
    event_split = stream.split[event_keep]
    event_session = stream.session[event_keep]
    event_history = scaler.transform(history.evaluate(event_time, event_session))
    event_source = _latest_anchor_source(
        event_time, event_session, anchor_time, anchor_session
    )
    q_time_rows = []
    q_weight_rows = []
    q_session_rows = []
    q_split_rows = []
    for split_name, code in (("train", 0), ("validation", 1)):
        q_time, q_weight, q_session = _quadrature_grid(
            stream, coverage, split_name, 4
        )
        q_time_rows.append(q_time)
        q_weight_rows.append(q_weight)
        q_session_rows.append(q_session)
        q_split_rows.append(np.full(len(q_time), code, dtype=np.int8))
    q_time = np.concatenate(q_time_rows)
    q_weight = np.concatenate(q_weight_rows)
    q_session = np.concatenate(q_session_rows)
    q_split = np.concatenate(q_split_rows)
    order = np.argsort(q_time, kind="stable")
    q_time, q_weight, q_session, q_split = (
        value[order] for value in (q_time, q_weight, q_session, q_split)
    )
    q_history = scaler.transform(history.evaluate(q_time, q_session))
    q_source = _latest_anchor_source(q_time, q_session, anchor_time, anchor_session)
    session_label = np.unique(coverage.session[coverage.start < coverage.dev_end_epoch])
    session_start = np.asarray([
        coverage.start[coverage.session == label].min() for label in session_label
    ], dtype=np.float64)
    design = FullAnchorDesign(
        subject=subject,
        anchor_time=anchor_time.astype(np.float64),
        anchor_split=anchor_split.astype(np.int8),
        anchor_session=anchor_session.astype(np.int64),
        anchor_history=scaler.transform(history.evaluate(anchor_time, anchor_session)),
        event_time=event_time.astype(np.float64),
        event_split=event_split.astype(np.int8),
        event_session=event_session.astype(np.int64),
        event_source_anchor=event_source,
        event_history=event_history,
        event_group_ids=stream.group_ids[event_keep].astype(np.int64),
        event_group_count=stream.group_count[event_keep].astype(np.int64),
        quadrature_time=q_time.astype(np.float64),
        quadrature_split=q_split,
        quadrature_session=q_session.astype(np.int64),
        quadrature_source_anchor=q_source,
        quadrature_history=q_history,
        quadrature_weight_seconds=q_weight.astype(np.float64),
        session_label=session_label.astype(np.int64),
        session_start=session_start,
    )
    design.validate()
    for split_name, code in (("train", 0), ("validation", 1)):
        contract.assert_development_times(
            subject, design.event_time[design.event_split == code], split_name
        )
        contract.assert_development_times(
            subject, design.anchor_time[design.anchor_split == code], split_name
        )

    output = Path(output_root) / "cache" / subject
    design_path = output / "full_design.npz"
    explicit_path = output / "explicit_embedding.npy"
    raw_path = output / "explicit_raw_embedding.npy"
    save_full_design(design_path, design)
    _atomic_npy(explicit_path, explicit_embedding_array)
    _atomic_npy(raw_path, raw_embedding_array)
    expected_train, expected_validation = (
        int(np.sum(candidate_split == code)) for code in (0, 1)
    )
    manifest = {
        "status": "COMPLETE",
        "contract": contract.REVISION,
        "r1_2_revision": R1_2_REVISION,
        "cache_revision": CACHE_REVISION,
        "subject": subject,
        "n_train_anchors": int(np.sum(anchor_split == 0)),
        "n_validation_anchors": int(np.sum(anchor_split == 1)),
        "expected_train_anchors": expected_train,
        "expected_validation_anchors": expected_validation,
        "n_unreadable_anchors": int(unreadable),
        "n_train_events_full_recorded_support": int(np.sum(event_split == 0)),
        "n_validation_events_full_recorded_support": int(np.sum(event_split == 1)),
        "train_recorded_seconds": float(q_weight[q_split == 0].sum()),
        "validation_recorded_seconds": float(q_weight[q_split == 1].sum()),
        "explicit_scaler_source": "frozen_bridge_selected_train_anchors",
        "bridge_selected_epochs": bridge_result["selected_epochs"],
        "bridge_result": str(bridge_result_path),
        "bridge_result_sha256": contract.sha256_file(bridge_result_path),
        "bridge_checkpoint": str(bridge_checkpoint_path),
        "bridge_checkpoint_sha256": contract.sha256_file(bridge_checkpoint_path),
        "baseline_checkpoint": str(baseline_path),
        "baseline_checkpoint_sha256": contract.sha256_file(baseline_path),
        "coverage": str(coverage_path),
        "coverage_sha256": contract.sha256_file(coverage_path),
        "full_event_stream_revision": FULL_STREAM_REVISION,
        "full_coverage_revision": FULL_COVERAGE_REVISION,
        "design": str(design_path),
        "design_sha256": contract.sha256_file(design_path),
        "explicit_embedding": str(explicit_path),
        "explicit_embedding_sha256": contract.sha256_file(explicit_path),
        "explicit_raw_embedding": str(raw_path),
        "explicit_raw_embedding_sha256": contract.sha256_file(raw_path),
        "observer_frozen": True,
        "full_recorded_support": True,
        "sealed_opened": False,
    }
    if unreadable or manifest["n_train_anchors"] != expected_train or manifest[
        "n_validation_anchors"
    ] != expected_validation:
        raise ValueError(f"{subject}: full-anchor denominator changed during cache build")
    contract.atomic_json(output / "manifest.json", manifest)
    return manifest


def load_full_anchor_cache(subject: str, *, arm: str,
                           output_root: Path | None = None
                           ) -> tuple[FullAnchorDesign, np.ndarray, dict]:
    if arm not in {"explicit", "explicit_raw"}:
        raise ValueError("R1.2 arm must be explicit or explicit_raw")
    output_root = output_root or contract.RESULT_ROOT / "r1_2"
    root = Path(output_root) / "cache" / subject
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("status") != "COMPLETE" or manifest.get("sealed_opened") is not False:
        raise ValueError(f"{subject}: invalid R1.2 cache manifest")
    if manifest.get("r1_2_revision") != R1_2_REVISION:
        raise ValueError(f"{subject}: R1.2 cache revision mismatch")
    design_path = Path(manifest["design"])
    embedding_path = Path(manifest[
        "explicit_embedding" if arm == "explicit" else "explicit_raw_embedding"
    ])
    expected_hash = manifest[
        "explicit_embedding_sha256" if arm == "explicit"
        else "explicit_raw_embedding_sha256"
    ]
    if contract.sha256_file(design_path) != manifest["design_sha256"]:
        raise ValueError(f"{subject}: R1.2 design hash mismatch")
    if contract.sha256_file(embedding_path) != expected_hash:
        raise ValueError(f"{subject}: R1.2 embedding hash mismatch")
    design = load_full_design(design_path)
    embedding = np.load(embedding_path, mmap_mode="r")
    if embedding.shape != (len(design.anchor_time), 64):
        raise ValueError(f"{subject}: R1.2 embedding shape mismatch")
    return design, embedding, manifest


class FrozenEmbeddingStateModel(nn.Module):
    """Exact frozen history baseline plus one learned persistent state."""

    def __init__(self, baseline_checkpoint: dict, history_dim: int,
                 n_contacts: int, adjacency: np.ndarray,
                 *, observation_dim: int = 64, state_dim: int = 8):
        super().__init__()
        self.state = ControlledPersistentState(observation_dim, state_dim)
        self.timing_baseline = HistoryIntensity(history_dim, history_visible=True)
        self.timing_baseline.load_state_dict(baseline_checkpoint["timing"]["history"])
        self.mark_baseline = ExactHistoryMarkDecoder(
            history_dim, n_contacts, adjacency, history_visible=True
        )
        self.mark_baseline.load_state_dict(baseline_checkpoint["mark"]["history"])
        for module in (self.timing_baseline, self.mark_baseline):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.state_timing = nn.Linear(state_dim, 1, bias=False)
        self.state_contact = nn.Linear(state_dim, n_contacts, bias=False)
        self.state_size = nn.Linear(state_dim, n_contacts + 1, bias=False)
        for module in (self.state_timing, self.state_contact, self.state_size):
            nn.init.zeros_(module.weight)

    def timing_log_rate(self, history: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.timing_baseline(history) + self.state_timing(state).squeeze(-1)

    def mark_terms(self, history: torch.Tensor, state: torch.Tensor,
                   group_ids: torch.Tensor, group_count: torch.Tensor):
        size, contact = self.mark_baseline.logits(history, group_ids, group_count)
        size = size + self.state_size(state).unsqueeze(1)
        contact = contact + self.state_contact(state).unsqueeze(1)
        return tied_group_mark_log_prob(group_ids, group_count, size, contact)


@dataclass(frozen=True)
class FullT1Metrics:
    joint_nll_per_event: float
    timing_nll_per_event: float
    mark_nll_per_event: float
    group_size_nll_per_event: float
    subset_nll_per_event: float
    n_events: int
    n_anchors: int
    recorded_seconds: float


def _flow(model: FrozenEmbeddingStateModel, state: torch.Tensor,
          delta_minutes: float, transition_cache: dict[float, torch.Tensor]
          ) -> torch.Tensor:
    key = round(float(max(delta_minutes, 0.0)), 9)
    transition = transition_cache.get(key)
    if transition is None:
        transition = torch.matrix_exp(
            model.state.generator.matrix().to(state.dtype) * key
        )
        transition_cache[key] = transition
    mu = model.state.generator.mu
    return mu + torch.matmul(state - mu, transition.transpose(-1, -2))


def filtered_anchor_states(model: FrozenEmbeddingStateModel,
                           design: FullAnchorDesign, embedding: np.ndarray,
                           *, device: torch.device | str,
                           correction_enabled: bool = True,
                           validation_correction_off: bool = False,
                           max_anchor: int | None = None) -> torch.Tensor:
    """Causal state scan; TRAIN naturally warm-starts validation."""
    n = len(design.anchor_time) if max_anchor is None else min(
        int(max_anchor) + 1, len(design.anchor_time)
    )
    output = torch.zeros((len(design.anchor_time), model.state.dim), device=device)
    for label in design.session_label:
        anchors = np.flatnonzero(
            (design.anchor_session == label) & (np.arange(len(design.anchor_time)) < n)
        )
        if not len(anchors):
            continue
        anchors = anchors[np.argsort(design.anchor_time[anchors], kind="stable")]
        state = output.new_zeros(model.state.dim)
        cursor = float(design.session_start_for(np.asarray([label]))[0])
        transition_cache: dict[float, torch.Tensor] = {}
        for anchor in anchors:
            time = float(design.anchor_time[anchor])
            state_minus = _flow(
                model, state, (time - cursor) / 60.0, transition_cache
            )
            enabled = correction_enabled and not (
                validation_correction_off and int(design.anchor_split[anchor]) == 1
            )
            observation = torch.as_tensor(
                np.array(embedding[anchor], copy=True), device=device
            )
            state = model.state.correction(state_minus, observation, enabled=enabled)
            output[anchor] = state
            cursor = time
    return output


def memoryless_anchor_states(model: FrozenEmbeddingStateModel,
                             design: FullAnchorDesign, embedding: np.ndarray,
                             *, device: torch.device | str,
                             max_anchor: int | None = None) -> torch.Tensor:
    """Independent observation codes with no carry between anchors.

    Every anchor starts from the learned generator mean ``mu`` and applies the
    same observation correction as the persistent model.  Query states still
    flow causally forward from the target anchor.  This isolates cross-anchor
    carry from the information in the current 30 s observation window.
    """
    n = len(design.anchor_time) if max_anchor is None else min(
        int(max_anchor) + 1, len(design.anchor_time)
    )
    output = torch.zeros((len(design.anchor_time), model.state.dim), device=device)
    if n <= 0:
        return output
    anchor = np.arange(n, dtype=np.int64)
    observation = torch.as_tensor(
        np.array(embedding[anchor], copy=True), device=device
    )
    mean = model.state.generator.mu.unsqueeze(0).expand(n, -1)
    output[:n] = model.state.correction(mean, observation, enabled=True)
    return output


def matched_wrong_time_permutation(design: FullAnchorDesign, *, split: str,
                                   min_separation_seconds: float = 300.0
                                   ) -> tuple[np.ndarray, np.ndarray]:
    target = design.anchor_ids(split)
    permutation = np.arange(len(design.anchor_time), dtype=np.int64)
    matched = np.zeros(len(design.anchor_time), dtype=bool)
    feature_index = np.asarray([
        value for value in (2, 3, 4, 5, 6, 7, 8, 9, 10)
        if value < design.anchor_history.shape[1]
    ], dtype=np.int64)
    for row in target:
        candidate = target[
            (design.anchor_session[target] == design.anchor_session[row])
            & (np.abs(design.anchor_time[target] - design.anchor_time[row])
               >= float(min_separation_seconds))
        ]
        if not len(candidate):
            continue
        delta = design.anchor_history[candidate][:, feature_index] - design.anchor_history[
            row, feature_index
        ]
        donor = int(candidate[int(np.argmin(np.sum(delta.astype(np.float64) ** 2, axis=1)))])
        permutation[row] = donor
        matched[row] = True
    return permutation, matched


def _query_states(model: FrozenEmbeddingStateModel, design: FullAnchorDesign,
                  anchor_state: torch.Tensor, source: np.ndarray,
                  time: np.ndarray, session: np.ndarray,
                  rows: np.ndarray, *, state_permutation: np.ndarray | None,
                  device: torch.device | str) -> torch.Tensor:
    rows = np.asarray(rows, dtype=np.int64)
    result = []
    for lo in range(0, len(rows), 8192):
        take = rows[lo:lo + 8192]
        selected_source = source[take]
        source_state = torch.zeros((len(take), model.state.dim), device=device)
        source_time = design.session_start_for(session[take]).astype(np.float64)
        observed = selected_source >= 0
        if bool(observed.any()):
            donor = selected_source[observed]
            if state_permutation is not None:
                donor = np.asarray(state_permutation, dtype=np.int64)[donor]
            source_state[torch.as_tensor(observed, device=device)] = anchor_state[
                torch.as_tensor(donor, dtype=torch.long, device=device)
            ]
            # A donor state replaces the target state at the target anchor time;
            # propagation still uses the target's true event/quadrature offset.
            source_time[observed] = design.anchor_time[selected_source[observed]]
        delta = torch.as_tensor(
            (time[take] - source_time) / 60.0,
            dtype=source_state.dtype, device=device,
        ).clamp(min=0.0)
        matrix = model.state.generator.matrix().to(source_state.dtype)
        transition = torch.matrix_exp(matrix.unsqueeze(0) * delta[:, None, None])
        mu = model.state.generator.mu
        result.append(mu.unsqueeze(0) + torch.matmul(
            transition, (source_state - mu).unsqueeze(-1)
        ).squeeze(-1))
    return torch.cat(result, dim=0) if result else anchor_state.new_empty((0, model.state.dim))


def evaluate_full_t1(model: FrozenEmbeddingStateModel, design: FullAnchorDesign,
                     embedding: np.ndarray, split: str, *,
                     device: torch.device | str,
                     correction_enabled: bool = True,
                     validation_correction_off: bool = False,
                     state_permutation: np.ndarray | None = None,
                     matched_anchor_mask: np.ndarray | None = None,
                     anchor_state_mode: str = "persistent",
                     time_lower: float | None = None,
                     time_upper: float | None = None,
                     anchor_state_override: torch.Tensor | None = None,
                     ) -> FullT1Metrics:
    model.eval()
    code = {"train": 0, "validation": 1}[split]
    with torch.no_grad():
        if anchor_state_override is not None:
            anchor_state = anchor_state_override.to(device)
            if anchor_state.shape != (len(design.anchor_time), model.state.dim):
                raise ValueError("anchor_state_override shape disagrees with design")
        elif anchor_state_mode == "persistent":
            anchor_state = filtered_anchor_states(
                model, design, embedding, device=device,
                correction_enabled=correction_enabled,
                validation_correction_off=validation_correction_off,
            )
        elif anchor_state_mode == "memoryless":
            if not correction_enabled or validation_correction_off:
                raise ValueError(
                    "memoryless mode is the current-observation correction arm; "
                    "correction-off flags are not defined for it"
                )
            anchor_state = memoryless_anchor_states(
                model, design, embedding, device=device
            )
        else:
            raise ValueError(f"unknown anchor_state_mode {anchor_state_mode!r}")
        event_keep = design.event_split == code
        q_keep = design.quadrature_split == code
        if time_lower is not None:
            event_keep &= design.event_time >= float(time_lower)
            q_keep &= design.quadrature_time >= float(time_lower)
        if time_upper is not None:
            event_keep &= design.event_time < float(time_upper)
            q_keep &= design.quadrature_time < float(time_upper)
        if matched_anchor_mask is not None:
            matched_anchor_mask = np.asarray(matched_anchor_mask, dtype=bool)
            event_keep &= (
                (design.event_source_anchor >= 0)
                & matched_anchor_mask[np.maximum(design.event_source_anchor, 0)]
            )
            q_keep &= (
                (design.quadrature_source_anchor >= 0)
                & matched_anchor_mask[np.maximum(design.quadrature_source_anchor, 0)]
            )
        event_rows = np.flatnonzero(event_keep)
        q_rows = np.flatnonzero(q_keep)
        event_state = _query_states(
            model, design, anchor_state, design.event_source_anchor,
            design.event_time, design.event_session, event_rows,
            state_permutation=state_permutation, device=device,
        )
        q_state = _query_states(
            model, design, anchor_state, design.quadrature_source_anchor,
            design.quadrature_time, design.quadrature_session, q_rows,
            state_permutation=state_permutation, device=device,
        )
        event_log = 0.0
        mark_log = 0.0
        size_log = 0.0
        subset_log = 0.0
        for lo in range(0, len(event_rows), 4096):
            take = event_rows[lo:lo + 4096]
            state = event_state[lo:lo + len(take)]
            history = torch.as_tensor(design.event_history[take], device=device)
            event_log += float(model.timing_log_rate(history, state).sum())
            mark = model.mark_terms(
                history, state,
                torch.as_tensor(design.event_group_ids[take], dtype=torch.long, device=device),
                torch.as_tensor(design.event_group_count[take], dtype=torch.long, device=device),
            )
            mark_log += float(mark.event_log_prob.sum())
            size_log += float(mark.group_size_log_prob.sum())
            subset_log += float(mark.subset_log_prob.sum())
        survival = 0.0
        for lo in range(0, len(q_rows), 65536):
            take = q_rows[lo:lo + 65536]
            state = q_state[lo:lo + len(take)]
            history = torch.as_tensor(design.quadrature_history[take], device=device)
            log_rate = model.timing_log_rate(history, state)
            weight = torch.as_tensor(
                design.quadrature_weight_seconds[take],
                dtype=log_rate.dtype, device=device,
            )
            survival += float(torch.sum(weight * torch.exp(torch.clamp(log_rate, max=20.0))))
    denominator = max(len(event_rows), 1)
    timing = (survival - event_log) / denominator
    mark_nll = -mark_log / denominator
    anchor_keep = design.anchor_split == code
    if time_lower is not None:
        anchor_keep &= design.anchor_time >= float(time_lower)
    if time_upper is not None:
        anchor_keep &= design.anchor_time < float(time_upper)
    anchor_ids = np.flatnonzero(anchor_keep)
    return FullT1Metrics(
        joint_nll_per_event=timing + mark_nll,
        timing_nll_per_event=timing,
        mark_nll_per_event=mark_nll,
        group_size_nll_per_event=-size_log / denominator,
        subset_nll_per_event=-subset_log / denominator,
        n_events=int(len(event_rows)),
        n_anchors=int(len(anchor_ids)),
        recorded_seconds=float(design.quadrature_weight_seconds[q_rows].sum()),
    )


def _train_epoch(model: FrozenEmbeddingStateModel, design: FullAnchorDesign,
                 embedding: np.ndarray, optimizer: torch.optim.Optimizer, *,
                 device: torch.device | str, anchor_ids: np.ndarray,
                 query_time_upper: float, chunk_anchors: int = 256,
                 grad_clip_norm: float | None = 1.0,
                 step_state: dict[str, int] | None = None,
                 diagnostics: dict | None = None) -> None:
    selected = np.zeros(len(design.anchor_time), dtype=bool)
    selected[np.asarray(anchor_ids, dtype=np.int64)] = True
    event_allowed = np.flatnonzero(
        (design.event_split == 0) & (design.event_time < float(query_time_upper))
        & (design.event_source_anchor >= 0)
        & selected[np.maximum(design.event_source_anchor, 0)]
    )
    q_allowed = np.flatnonzero(
        (design.quadrature_split == 0)
        & (design.quadrature_time < float(query_time_upper))
        & (design.quadrature_source_anchor >= 0)
        & selected[np.maximum(design.quadrature_source_anchor, 0)]
    )

    def grouped_rows(rows: np.ndarray, source: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
        if not len(rows):
            return rows, np.zeros(len(design.anchor_time) + 1, dtype=np.int64)
        order = np.argsort(source[rows], kind="stable")
        sorted_rows = rows[order]
        count = np.bincount(
            source[sorted_rows], minlength=len(design.anchor_time)
        )
        boundary = np.concatenate([[0], np.cumsum(count, dtype=np.int64)])
        return sorted_rows, boundary

    event_sorted, event_boundary = grouped_rows(
        event_allowed, design.event_source_anchor
    )
    q_sorted, q_boundary = grouped_rows(q_allowed, design.quadrature_source_anchor)

    def rows_for(chunk: np.ndarray, sorted_rows: np.ndarray,
                 boundary: np.ndarray) -> np.ndarray:
        pieces = [
            sorted_rows[boundary[int(anchor)]:boundary[int(anchor) + 1]]
            for anchor in chunk
            if boundary[int(anchor) + 1] > boundary[int(anchor)]
        ]
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)

    event_total = int(len(event_allowed))
    chunks = max(int(math.ceil(len(anchor_ids) / max(int(chunk_anchors), 1))), 1)
    scale = float(chunks) / max(event_total, 1)
    if diagnostics is not None:
        diagnostics.update({
            "optimizer_steps": 0, "events": event_total,
            "anchors": int(len(anchor_ids)),
            "sessions": int(sum(
                bool(np.any(selected & (design.anchor_session == label)))
                for label in design.session_label
            )),
            "objective_numerator": 0.0, "preclip_norm_max": 0.0,
            "postclip_norm_max": 0.0, "clip_count": 0,
            "nonfinite_gradient_steps": 0, "learning_rate_last": {},
            "gradient_group_max": {},
        })
    for label in design.session_label:
        anchors = np.flatnonzero(selected & (design.anchor_session == label))
        if not len(anchors):
            continue
        anchors = anchors[np.argsort(design.anchor_time[anchors], kind="stable")]
        state = torch.zeros(model.state.dim, device=device)
        cursor = float(design.session_start_for(np.asarray([label]))[0])
        for lo in range(0, len(anchors), int(chunk_anchors)):
            chunk = anchors[lo:lo + int(chunk_anchors)]
            transition_cache: dict[float, torch.Tensor] = {}
            chunk_states = []
            for anchor in chunk:
                time = float(design.anchor_time[anchor])
                state_minus = _flow(
                    model, state, (time - cursor) / 60.0, transition_cache
                )
                observation = torch.as_tensor(
                    np.array(embedding[anchor], copy=True), device=device
                )
                state = model.state.correction(state_minus, observation, enabled=True)
                chunk_states.append(state)
                cursor = time
            chunk_states_tensor = torch.stack(chunk_states)
            local_lookup = np.full(len(design.anchor_time), -1, dtype=np.int64)
            local_lookup[chunk] = np.arange(len(chunk), dtype=np.int64)
            event_rows = rows_for(chunk, event_sorted, event_boundary)
            q_rows = rows_for(chunk, q_sorted, q_boundary)
            if not len(event_rows) and not len(q_rows):
                state = state.detach()
                continue
            event_source = local_lookup[design.event_source_anchor[event_rows]]
            q_source = local_lookup[design.quadrature_source_anchor[q_rows]]
            event_source_state = chunk_states_tensor[
                torch.as_tensor(event_source, dtype=torch.long, device=device)
            ]
            q_source_state = chunk_states_tensor[
                torch.as_tensor(q_source, dtype=torch.long, device=device)
            ]
            event_delta = torch.as_tensor(
                (design.event_time[event_rows]
                 - design.anchor_time[design.event_source_anchor[event_rows]]) / 60.0,
                dtype=event_source_state.dtype, device=device,
            ).clamp(min=0.0)
            q_delta = torch.as_tensor(
                (design.quadrature_time[q_rows]
                 - design.anchor_time[design.quadrature_source_anchor[q_rows]]) / 60.0,
                dtype=q_source_state.dtype, device=device,
            ).clamp(min=0.0)
            matrix = model.state.generator.matrix().to(event_source_state.dtype)
            if len(event_rows):
                transition = torch.matrix_exp(matrix.unsqueeze(0) * event_delta[:, None, None])
                mu = model.state.generator.mu
                event_state = mu.unsqueeze(0) + torch.matmul(
                    transition, (event_source_state - mu).unsqueeze(-1)
                ).squeeze(-1)
                event_history = torch.as_tensor(design.event_history[event_rows], device=device)
                event_log = model.timing_log_rate(event_history, event_state).sum()
                mark = model.mark_terms(
                    event_history, event_state,
                    torch.as_tensor(design.event_group_ids[event_rows], dtype=torch.long, device=device),
                    torch.as_tensor(design.event_group_count[event_rows], dtype=torch.long, device=device),
                )
                mark_log = mark.event_log_prob.sum()
            else:
                event_log = state.new_zeros(())
                mark_log = state.new_zeros(())
            if len(q_rows):
                transition = torch.matrix_exp(matrix.unsqueeze(0) * q_delta[:, None, None])
                mu = model.state.generator.mu
                q_state = mu.unsqueeze(0) + torch.matmul(
                    transition, (q_source_state - mu).unsqueeze(-1)
                ).squeeze(-1)
                q_history = torch.as_tensor(design.quadrature_history[q_rows], device=device)
                q_log = model.timing_log_rate(q_history, q_state)
                weight = torch.as_tensor(
                    design.quadrature_weight_seconds[q_rows],
                    dtype=q_log.dtype, device=device,
                )
                survival = torch.sum(weight * torch.exp(torch.clamp(q_log, max=20.0)))
            else:
                survival = state.new_zeros(())
            objective = survival - event_log - mark_log
            loss = objective * scale
            optimizer.zero_grad(set_to_none=True)
            if step_state is not None:
                step = int(step_state.get("step", 0))
                warmup_steps = int(step_state.get("warmup_steps", 0))
                factor = (
                    min(1.0, float(step + 1) / float(warmup_steps))
                    if warmup_steps > 0 else 1.0
                )
                for group in optimizer.param_groups:
                    group["lr"] = float(group.get("base_lr", group["lr"])) * factor
                step_state["step"] = step + 1
            loss.backward()
            named = list(model.named_parameters())
            group_prefix = {
                "stable_generator": "state.generator",
                "observation_correction": "state.correction",
                "state_readout_timing": "state_timing",
                "state_readout_contact": "state_contact",
                "state_readout_size": "state_size",
            }
            if diagnostics is not None:
                for label_name, prefix in group_prefix.items():
                    square = sum(
                        float(value.grad.detach().float().square().sum())
                        for name, value in named
                        if name.startswith(prefix) and value.grad is not None
                    )
                    norm = math.sqrt(square)
                    previous = diagnostics["gradient_group_max"].get(
                        label_name, 0.0
                    )
                    diagnostics["gradient_group_max"][label_name] = max(
                        previous, norm
                    )
            parameters = [
                value for value in model.parameters() if value.requires_grad
            ]
            if grad_clip_norm is None:
                preclip = math.sqrt(sum(
                    float(value.grad.detach().float().square().sum())
                    for value in parameters if value.grad is not None
                ))
            else:
                preclip = float(torch.nn.utils.clip_grad_norm_(
                    parameters, float(grad_clip_norm)
                ))
            if not math.isfinite(preclip):
                if diagnostics is not None:
                    diagnostics["nonfinite_gradient_steps"] += 1
                raise RuntimeError("R1.2 prefix encountered a non-finite gradient")
            optimizer.step()
            if diagnostics is not None:
                diagnostics["optimizer_steps"] += 1
                diagnostics["objective_numerator"] += float(objective.detach())
                diagnostics["preclip_norm_max"] = max(
                    diagnostics["preclip_norm_max"], preclip
                )
                diagnostics["postclip_norm_max"] = max(
                    diagnostics["postclip_norm_max"],
                    preclip if grad_clip_norm is None else min(
                        preclip, float(grad_clip_norm)
                    ),
                )
                if grad_clip_norm is not None and preclip > float(grad_clip_norm):
                    diagnostics["clip_count"] += 1
                diagnostics["learning_rate_last"] = {
                    str(group.get("group_name", index)): float(group["lr"])
                    for index, group in enumerate(optimizer.param_groups)
                }
            state = state.detach()
    if diagnostics is not None:
        diagnostics["train_joint_nll_per_event"] = (
            diagnostics["objective_numerator"] / max(event_total, 1)
        )
        diagnostics["clip_fraction"] = (
            diagnostics["clip_count"]
            / max(diagnostics["optimizer_steps"], 1)
        )


def fit_full_t1(model: FrozenEmbeddingStateModel, design: FullAnchorDesign,
                embedding: np.ndarray, *, device: torch.device | str,
                epochs: int = 6, learning_rate: float = 3e-4,
                chunk_anchors: int = 256) -> FrozenEmbeddingStateModel:
    """Time-ordered truncated-BPTT fit with epoch-zero inner selection."""
    train = design.anchor_ids("train")
    if len(train) < 10:
        raise ValueError("R1.2 needs at least ten TRAIN anchors")
    cut = int(np.clip(math.floor(0.8 * len(train)), 1, len(train) - 1))
    inner_train = train[:cut]
    boundary = float(design.anchor_time[train[cut]])
    train_end, _ = contract.load_split(design.subject)
    initial = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    best_epoch = 0
    best_value = evaluate_full_t1(
        model, design, embedding, "train", device=device,
        time_lower=boundary, time_upper=train_end,
    ).joint_nll_per_event
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(learning_rate), weight_decay=1e-3,
    )
    for epoch in range(1, int(epochs) + 1):
        model.train()
        _train_epoch(
            model, design, embedding, optimizer, device=device,
            anchor_ids=inner_train, query_time_upper=boundary,
            chunk_anchors=chunk_anchors,
        )
        value = evaluate_full_t1(
            model, design, embedding, "train", device=device,
            time_lower=boundary, time_upper=train_end,
        ).joint_nll_per_event
        if value < best_value:
            best_value = value
            best_epoch = epoch

    model.load_state_dict(initial)
    if best_epoch:
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(learning_rate), weight_decay=1e-3,
        )
        for _ in range(best_epoch):
            model.train()
            _train_epoch(
                model, design, embedding, optimizer, device=device,
                anchor_ids=train, query_time_upper=train_end,
                chunk_anchors=chunk_anchors,
            )
    model.selected_epochs = int(best_epoch)
    model.inner_validation_joint_nll = float(best_value)
    model.truncated_bptt_anchors = int(chunk_anchors)
    return model.eval()
