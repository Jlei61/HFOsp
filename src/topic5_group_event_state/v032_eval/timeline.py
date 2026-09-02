"""One patient's v0.3.2 evaluation view: segments, partition, marks, anchors.

This mirrors ``v02.subject.load_subject_timeline`` but every TRAIN-only object
(mark PCA, mark standardisation, contact vocabulary) is estimated on the
``base_fit`` prefix of the v0.3.2 partition, and anchor eligibility is judged
against the v0.3.2 phases instead of the old 70/10/20 split.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.topic5_group_event_state.contract import contact_shaft
from src.topic5_group_event_state.v02.marks import EventMarks, build_event_marks
from src.topic5_group_event_state.v02.targets import FutureTargetBuilder
from src.topic5_group_event_state.v02.timeline import (
    AnchorGrid,
    CoverageSegment,
    assign_events_to_segments,
    build_anchor_grid,
    build_carry_segments,
    sessions_from_inventory,
)

from .partition import EVAL_PHASES, EvalPartition, eval_partition


@dataclass
class EvalTimeline:
    subject: str
    dataset: str
    config: dict[str, Any]
    index: Mapping[str, Any]
    segments: list[CoverageSegment]
    partition: EvalPartition
    stream_positions: np.ndarray      # positions into the v0.1 interictal stream
    raw_positions: np.ndarray         # positions into the raw consolidated arrays
    event_times: np.ndarray
    event_segment: np.ndarray
    event_session: np.ndarray
    participation: np.ndarray         # (N, C) bool, all lagPat contacts
    relative_delay: np.ndarray        # (N, C) float32
    tied_group_id: np.ndarray         # (N, C) int16
    group_count: np.ndarray           # (N,) int64
    contact_names: tuple[str, ...]
    contact_shafts: tuple[str, ...]
    vocab_mask: np.ndarray            # (C,) bool  -- base_fit prefix vocabulary
    contact_valid_base: np.ndarray    # (C,) bool  -- contact_ok on base_fit events
    marks: EventMarks
    builder: FutureTargetBuilder
    grid: AnchorGrid
    seizures: list[dict[str, Any]]
    excluded: dict[str, int]

    @property
    def n_events(self) -> int:
        return int(self.event_times.size)

    @property
    def n_contacts(self) -> int:
        return int(self.participation.shape[1])

    @property
    def n_vocab(self) -> int:
        return int(self.vocab_mask.sum())

    @property
    def horizons_seconds(self) -> tuple[float, ...]:
        return tuple(float(h) for h in self.config["timeline"]["horizons_seconds"])

    def anchor_phase_labels(self) -> np.ndarray:
        return self.partition.labels_of(self.grid.t_anchor)

    def event_phase_labels(self) -> np.ndarray:
        return self.partition.labels_of(self.event_times)

    def segment_start_map(self) -> dict[int, float]:
        return {int(s.segment_id): float(s.start_epoch) for s in self.segments}

    def session_start_map(self) -> dict[int, float]:
        out: dict[int, float] = {}
        for seg in self.segments:
            out[int(seg.session_id)] = min(out.get(int(seg.session_id), np.inf), float(seg.start_epoch))
        return out

    def anchor_indices(self, phase: str, horizon_index: int) -> np.ndarray:
        """Eligible anchors of one phase whose whole window lies in that phase."""

        horizon = self.horizons_seconds[horizon_index]
        mask = (
            self.partition.mask_for_phase(self.grid.t_anchor, phase)
            & self.grid.eligible[:, horizon_index]
            & self.partition.window_within_phase(self.grid.t_anchor, horizon)
        )
        return np.flatnonzero(mask)

    def window_counts(self, anchors: np.ndarray, horizon_index: int) -> np.ndarray:
        return (
            self.grid.window_hi[anchors, horizon_index]
            - self.grid.window_lo[anchors, horizon_index]
        ).astype(np.int64)

    def event_indices(self, phase: str) -> np.ndarray:
        return np.flatnonzero(self.partition.mask_for_phase(self.event_times, phase))

    def positive_k_mask(self) -> np.ndarray:
        return self.group_count >= 2

    def summary(self) -> dict[str, Any]:
        labels = self.event_phase_labels()
        return {
            "subject": self.subject,
            "dataset": self.dataset,
            "n_events": self.n_events,
            "n_contacts": self.n_contacts,
            "n_vocab": self.n_vocab,
            "n_segments": len(self.segments),
            "n_anchors": int(self.grid.n_anchors),
            "partition": self.partition.as_dict(),
            "events_by_phase": {
                name: int((labels == i).sum()) for i, name in enumerate(EVAL_PHASES)
            },
            "excluded": dict(self.excluded),
        }


def _load_session_rows(subject: str, inventory: Path) -> list[Mapping[str, str]]:
    rows = [r for r in csv.DictReader(Path(inventory).open()) if r["subject"] == subject]
    if not rows:
        raise FileNotFoundError(f"{subject}: no session rows in {inventory}")
    return rows


def _group_count(tied_group_id: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(tied_group_id, dtype=np.int64).max(axis=1) + 1, 0)


def load_eval_timeline(subject: str, config: Mapping[str, Any]) -> EvalTimeline:
    root = Path(config["dataset_root"]) / subject
    index = json.loads((root / "index.json").read_text())
    scalars = np.load(root / "scalars.npz")
    tl_cfg = config["timeline"]

    order = np.asarray(scalars["interictal_index"], dtype=np.int64)
    t_all = np.asarray(scalars["t_abs"], dtype=np.float64)
    t_stream = t_all[order]

    sessions = sessions_from_inventory(_load_session_rows(subject, Path(config["session_inventory"])))
    seizures = [dict(s) for s in index.get("seizures", [])]
    segments = build_carry_segments(
        sessions, seizures,
        postictal_exclusion_seconds=float(tl_cfg["postictal_exclusion_seconds"]),
        min_segment_seconds=float(tl_cfg["min_segment_seconds"]),
    )
    if not segments:
        raise ValueError(f"{subject}: no coverage segment survives the seizure exclusion")
    seg_of = assign_events_to_segments(t_stream, segments)
    keep = seg_of >= 0
    stream_positions = np.flatnonzero(keep)
    event_times = t_stream[keep]
    event_segment = seg_of[keep]
    event_session = np.asarray([segments[i].session_id for i in event_segment], dtype=np.int64)
    excluded = {
        "n_interictal_events_total": int(order.size),
        "n_events_outside_segments": int((~keep).sum()),
        "n_events_kept": int(keep.sum()),
        "n_ictal_events_upstream": int(index["n_events_ictal"]),
    }
    partition = eval_partition(segments, config["partition"]["boundary_fractions"])
    base_fit_positions = np.flatnonzero(partition.mask_for_phase(event_times, "base_fit"))
    if base_fit_positions.size == 0:
        raise ValueError(f"{subject}: base_fit prefix holds no event")

    raw_positions = order[stream_positions]
    participation = np.asarray(np.load(root / "participation.npy", mmap_mode="r")[raw_positions])
    relative_delay = np.asarray(np.load(root / "relative_delay.npy", mmap_mode="r")[raw_positions])
    tied_group_id = np.asarray(np.load(root / "tied_group_id.npy", mmap_mode="r")[raw_positions])
    band_features = np.asarray(np.load(root / "band_features.npy", mmap_mode="r")[raw_positions])
    contact_ok = np.asarray(np.load(root / "contact_ok.npy", mmap_mode="r")[raw_positions])

    # Contact vocabulary and montage validity come from the base_fit prefix only.
    vocab_min = int(config["measurement"]["vocab_min_events"])
    vocab_mask = participation[base_fit_positions].sum(axis=0) >= vocab_min
    contact_valid_base = contact_ok[base_fit_positions].any(axis=0)

    marks = build_event_marks(
        participation, relative_delay, band_features,
        band_available=tuple(bool(b) for b in index["band_available"]),
        band_names=tuple(str(b) for b in index["bands"]),
        train_positions=base_fit_positions,
        n_components=int(tl_cfg["embedding_components"]),
        seed=int(tl_cfg["embedding_seed"]),
    )
    del band_features
    builder = FutureTargetBuilder(marks)
    grid = build_anchor_grid(
        segments, partition, event_times,
        horizons_seconds=tuple(float(h) for h in tl_cfg["horizons_seconds"]),
        grid_seconds=float(tl_cfg["anchor_grid_seconds"]),
        min_warmup_seconds=float(tl_cfg["min_warmup_seconds"]),
    )
    names = tuple(str(c["lagpat_label"]) for c in index["contacts"])
    shafts = tuple(str(c.get("shaft") or contact_shaft(c["lagpat_label"]) or "?") for c in index["contacts"])
    return EvalTimeline(
        subject=subject,
        dataset=str(index["dataset"]),
        config=dict(config),
        index=index,
        segments=segments,
        partition=partition,
        stream_positions=stream_positions,
        raw_positions=raw_positions,
        event_times=event_times,
        event_segment=event_segment,
        event_session=event_session,
        participation=np.asarray(participation, dtype=bool),
        relative_delay=np.asarray(relative_delay, dtype=np.float32),
        tied_group_id=np.asarray(tied_group_id, dtype=np.int16),
        group_count=_group_count(tied_group_id),
        contact_names=names,
        contact_shafts=shafts,
        vocab_mask=np.asarray(vocab_mask, dtype=bool),
        contact_valid_base=np.asarray(contact_valid_base, dtype=bool),
        marks=marks,
        builder=builder,
        grid=grid,
        seizures=seizures,
        excluded=excluded,
    )
