"""Assemble one patient's v0.2 view: segments, split, marks, anchors, baseline.

This is the only place that reads the v0.1 dataset directory, and it never reads
``index.json::split_bounds_on_interictal_index`` -- that field is the v0.1
event-count split, which CC 7.1 replaces.  There is deliberately no
``split=None`` fallback: a default would silently restore it.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .baseline import BaselineFeatures, build_baseline_features
from .marks import DEFAULT_EMBEDDING_COMPONENTS, EventMarks, build_event_marks, summarise
from .targets import FutureTargetBuilder
from .timeline import (
    ANCHOR_GRID_SECONDS,
    CoverageSegment,
    HORIZONS_SECONDS,
    MIN_SEGMENT_SECONDS,
    MIN_WARMUP_SECONDS,
    POSTICTAL_EXCLUSION_SECONDS,
    SPLIT_FRACTIONS,
    SPLIT_NAMES,
    AnchorGrid,
    PhysicalTimeSplit,
    assign_events_to_segments,
    build_anchor_grid,
    build_carry_segments,
    effective_independent_windows,
    physical_time_split,
    sessions_from_inventory,
)

DATASET_ROOT = Path("/data/hfosp_group_event_state_v0_1/dataset")
SESSION_INVENTORY = Path(
    "/home/honglab/leijiaxin/HFOsp/results/epi_prssm/group_event_state/v0_1"
    "/contiguous_session_inventory.csv"
)


@dataclass(frozen=True)
class SubjectTimelineConfig:
    postictal_exclusion_seconds: float = POSTICTAL_EXCLUSION_SECONDS
    min_segment_seconds: float = MIN_SEGMENT_SECONDS
    grid_seconds: float = ANCHOR_GRID_SECONDS
    min_warmup_seconds: float = MIN_WARMUP_SECONDS
    horizons_seconds: tuple[float, ...] = HORIZONS_SECONDS
    split_fractions: tuple[float, float, float] = SPLIT_FRACTIONS
    embedding_components: int = DEFAULT_EMBEDDING_COMPONENTS
    embedding_seed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "postictal_exclusion_seconds": self.postictal_exclusion_seconds,
            "min_segment_seconds": self.min_segment_seconds,
            "grid_seconds": self.grid_seconds,
            "min_warmup_seconds": self.min_warmup_seconds,
            "horizons_seconds": list(self.horizons_seconds),
            "split_fractions": list(self.split_fractions),
            "embedding_components": self.embedding_components,
            "embedding_seed": self.embedding_seed,
        }


@dataclass
class SubjectTimeline:
    """Everything downstream needs about one patient, with nothing model-specific."""

    subject: str
    dataset: str
    config: SubjectTimelineConfig
    index: Mapping[str, Any]
    segments: list[CoverageSegment]
    split: PhysicalTimeSplit
    stream_positions: np.ndarray     # positions into the v0.1 interictal stream
    event_times: np.ndarray          # float64 absolute epochs of those events
    event_segment: np.ndarray
    marks: EventMarks
    builder: FutureTargetBuilder
    grid: AnchorGrid
    baseline: BaselineFeatures
    excluded: dict[str, int]

    @property
    def n_contacts(self) -> int:
        return self.marks.n_contacts

    @property
    def n_dims(self) -> int:
        return self.marks.n_continuous

    def train_event_positions(self) -> np.ndarray:
        return np.flatnonzero(self.event_times < self.split.boundary_epochs[0])

    def anchor_mask(self, split_name: str, horizon_index: int) -> np.ndarray:
        return self.grid.split_mask(split_name) & self.grid.eligible[:, horizon_index]

    def window_stats(self, split_name: str, horizon_index: int):
        m = self.anchor_mask(split_name, horizon_index)
        return self.builder.window_stats(
            self.grid.window_lo[m, horizon_index], self.grid.window_hi[m, horizon_index]
        )

    def coverage_report(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "recorded_seconds": dict(self.split.recorded_seconds),
            "total_recorded_seconds": self.split.total_recorded_seconds,
            "n_segments": len(self.segments),
            "n_anchors_total": self.grid.n_anchors,
            "excluded": dict(self.excluded),
            "horizons": {},
        }
        for h_i, horizon in enumerate(self.config.horizons_seconds):
            entry: dict[str, Any] = {}
            for name in SPLIT_NAMES:
                mask = self.anchor_mask(name, h_i)
                entry[name] = {
                    "n_anchors": int(mask.sum()),
                    "n_independent_windows": effective_independent_windows(
                        self.segments, self.split, name, horizon
                    ),
                    "n_events_in_windows": int(
                        (self.grid.window_hi[mask, h_i] - self.grid.window_lo[mask, h_i]).sum()
                    ),
                }
            out["horizons"][f"{int(horizon)}s"] = entry
        return out


def _load_session_rows(subject: str, inventory: Path) -> list[Mapping[str, str]]:
    rows = [r for r in csv.DictReader(Path(inventory).open()) if r["subject"] == subject]
    if not rows:
        raise FileNotFoundError(f"{subject}: no session rows in {inventory}")
    return rows


def load_subject_timeline(
    subject: str,
    *,
    dataset_root: Path = DATASET_ROOT,
    session_inventory: Path = SESSION_INVENTORY,
    config: SubjectTimelineConfig = SubjectTimelineConfig(),
) -> SubjectTimeline:
    root = Path(dataset_root) / subject
    index = json.loads((root / "index.json").read_text())
    scalars = np.load(root / "scalars.npz")

    order = np.asarray(scalars["interictal_index"], dtype=np.int64)
    t_all = np.asarray(scalars["t_abs"], dtype=np.float64)
    t_stream = t_all[order]

    sessions = sessions_from_inventory(_load_session_rows(subject, session_inventory))
    seizures = list(index.get("seizures", []))
    segments = build_carry_segments(
        sessions,
        seizures,
        postictal_exclusion_seconds=config.postictal_exclusion_seconds,
        min_segment_seconds=config.min_segment_seconds,
    )
    if not segments:
        raise ValueError(f"{subject}: no coverage segment survives the seizure exclusion")

    seg_of = assign_events_to_segments(t_stream, segments)
    keep = seg_of >= 0
    stream_positions = np.flatnonzero(keep)
    event_times = t_stream[keep]
    event_segment = seg_of[keep]
    excluded = {
        "n_interictal_events_total": int(order.size),
        "n_events_outside_segments": int((~keep).sum()),
        "n_events_kept": int(keep.sum()),
        "n_ictal_events_upstream": int(index["n_events_ictal"]),
    }

    split = physical_time_split(segments, config.split_fractions)
    train_positions = np.flatnonzero(event_times < split.boundary_epochs[0])
    if train_positions.size == 0:
        raise ValueError(f"{subject}: TRAIN split holds no event")

    raw_positions = order[stream_positions]
    participation = np.asarray(
        np.load(root / "participation.npy", mmap_mode="r")[raw_positions]
    )
    relative_delay = np.asarray(
        np.load(root / "relative_delay.npy", mmap_mode="r")[raw_positions]
    )
    band_features = np.asarray(
        np.load(root / "band_features.npy", mmap_mode="r")[raw_positions]
    )
    marks = build_event_marks(
        participation,
        relative_delay,
        band_features,
        band_available=tuple(bool(b) for b in index["band_available"]),
        band_names=tuple(str(b) for b in index["bands"]),
        train_positions=train_positions,
        n_components=config.embedding_components,
        seed=config.embedding_seed,
    )
    del participation, relative_delay, band_features

    builder = FutureTargetBuilder(marks)
    grid = build_anchor_grid(
        segments,
        split,
        event_times,
        horizons_seconds=config.horizons_seconds,
        grid_seconds=config.grid_seconds,
        min_warmup_seconds=config.min_warmup_seconds,
    )
    onsets = np.array([float(s["onset_epoch"]) for s in seizures], dtype=np.float64)
    offsets = np.array([float(s["offset_epoch"]) for s in seizures], dtype=np.float64)
    baseline = build_baseline_features(
        grid, segments, event_times, event_segment, marks,
        seizure_onsets=onsets, seizure_offsets=offsets,
    )
    return SubjectTimeline(
        subject=subject,
        dataset=str(index["dataset"]),
        config=config,
        index=index,
        segments=segments,
        split=split,
        stream_positions=stream_positions,
        event_times=event_times,
        event_segment=event_segment,
        marks=marks,
        builder=builder,
        grid=grid,
        baseline=baseline,
        excluded=excluded,
    )


def timeline_summary(tl: SubjectTimeline) -> dict[str, Any]:
    report = tl.coverage_report()
    report.update(
        {
            "subject": tl.subject,
            "dataset": tl.dataset,
            "config": tl.config.as_dict(),
            "n_contacts": tl.n_contacts,
            "marks": summarise(tl.marks),
            "baseline_features": tl.baseline.n_features,
            "baseline_notes": tl.baseline.notes,
            "split_boundary_epochs": [float(v) for v in tl.split.boundary_epochs],
            "segment_seconds": [float(s.duration_seconds) for s in tl.segments],
        }
    )
    return report


def trainability(tl: SubjectTimeline, *, min_anchors: int = 1) -> dict[str, Any]:
    """Pre-registered eligibility: an anchor in every split at the shortest horizon.

    Short coverage removes *long horizons only*; it is recorded as
    ``insufficient_coverage`` for that horizon and is never a scientific negative.
    """

    per_horizon: dict[str, Any] = {}
    trainable = True
    for h_i, horizon in enumerate(tl.config.horizons_seconds):
        counts = {n: int(tl.anchor_mask(n, h_i).sum()) for n in SPLIT_NAMES}
        ok = all(v >= min_anchors for v in counts.values())
        per_horizon[f"{int(horizon)}s"] = {
            "anchors": counts,
            "status": "ok" if ok else "insufficient_coverage",
        }
        if h_i == 0:
            trainable = ok
    return {
        "subject": tl.subject,
        "trainable": bool(trainable),
        "per_horizon": per_horizon,
    }
