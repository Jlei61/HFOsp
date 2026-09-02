"""Shared assembly: from patient name to trained-arm-ready tensors, once.

Kept separate from ``train`` so that the perturbation stage, the innovation stage
and the trainer all build the *same* timeline from the *same* cuts.  If any of
them rebuilt it slightly differently, the perturbation would be scored against a
denominator the model was never trained on, and nothing in the numbers would say so.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .io import payload_hash
from .stream import SubjectStream, load_stream
from .support import (
    Interval,
    MAIN_HORIZONS_MINUTES,
    POSTICTAL_EXCLUSION_SECONDS,
    build_coverage_segments,
    cut_intervals_at_seizures,
    load_block_time_ranges,
    load_seizures,
    segment_anchor_grid,
    segment_bounds,
    select_disjoint_anchors,
    split_by_physical_time,
)
from .timeline import build_timelines
from .train import SubjectTensors, prepare_subject

V0_1_RESULTS = Path("/home/honglab/leijiaxin/HFOsp/results/epi_prssm/group_event_state/v0_1")
DATASET_ROOT = Path("/data/hfosp_group_event_state_v0_1/dataset")
AGENT_C_ROOT = Path("/data/hfosp_group_event_state_v0_2/agent_c")


@dataclass
class SubjectContext:
    subject: str
    stream: SubjectStream
    intervals: list[Interval]
    tensors: SubjectTensors
    horizons: list[int]
    disjoint: dict[int, list[tuple[int, int]]]   # horizon -> [(segment, anchor_index)]
    support_hash: str

    def anchor_time(self, segment: int, anchor_index: int) -> float:
        return float(self.tensors.timelines[segment].anchor_time[anchor_index])


def load_subject(
    subject: str,
    device: torch.device,
    *,
    horizons: Sequence[int] = MAIN_HORIZONS_MINUTES,
    postictal_exclusion_s: float = POSTICTAL_EXCLUSION_SECONDS,
    dataset_root: Path = DATASET_ROOT,
    inventory_csv: Path = V0_1_RESULTS / "block_inventory.csv",
    feature_root: Path = AGENT_C_ROOT / "features",
    background_root: Path = AGENT_C_ROOT / "background",
) -> SubjectContext:
    horizons = [int(h) for h in horizons]
    dataset_dir = Path(dataset_root) / subject

    block_ranges = load_block_time_ranges(inventory_csv, subject)
    seizures = load_seizures(dataset_dir / "index.json")
    segments = build_coverage_segments(block_ranges)
    cut = cut_intervals_at_seizures(
        segments, seizures, postictal_exclusion_s=postictal_exclusion_s
    )
    intervals = split_by_physical_time(cut)
    if not intervals:
        raise ValueError(f"{subject}: no usable interval survives the boundary cuts")

    stream = load_stream(dataset_dir, Path(feature_root) / f"{subject}.npz")
    with np.load(Path(background_root) / f"{subject}.npz") as bg:
        anchor_background = (bg["anchor_time"], bg["anchor_features"])

    timelines = build_timelines(intervals, stream.t_abs, anchor_background)
    tensors = prepare_subject(stream.features, timelines, intervals, horizons, device)

    # The independent denominator, defined once, here.  Every reported statistic
    # is a function of this list, so an eligibility count and an estimator count
    # cannot drift apart.
    bounds = segment_bounds(intervals)
    disjoint: dict[int, list[tuple[int, int]]] = {}
    for horizon in horizons:
        rows: list[tuple[int, int]] = []
        for seg_pos, segment_id in enumerate(sorted(bounds)):
            lo, hi = bounds[segment_id]
            grid = segment_anchor_grid(lo, hi)
            members = [i for i in intervals if i.segment_id == segment_id]
            chosen = {
                round(a, 6)
                for a, _split, _seg in select_disjoint_anchors(
                    grid, members, horizon, disjoint_exposure=False
                )
            }
            times = timelines[seg_pos].anchor_time
            for idx, t in enumerate(times):
                if round(float(t), 6) in chosen:
                    rows.append((seg_pos, idx))
        disjoint[horizon] = rows

    support_hash = payload_hash(
        {
            "subject": subject,
            "postictal_exclusion_s": postictal_exclusion_s,
            "horizons": horizons,
            "block_ranges": [[round(a, 3), round(b, 3)] for a, b in block_ranges],
            "seizures": [[round(a, 3), round(b, 3)] for a, b in seizures],
        }
    )
    return SubjectContext(
        subject=subject,
        stream=stream,
        intervals=intervals,
        tensors=tensors,
        horizons=horizons,
        disjoint=disjoint,
        support_hash=support_hash,
    )


def disjoint_mask(
    ctx: SubjectContext, horizon: int, segments: np.ndarray, anchor_ids: np.ndarray
) -> np.ndarray:
    """Which collected rows are members of the pre-registered disjoint set."""

    allowed = set(ctx.disjoint[int(horizon)])
    return np.asarray(
        [(int(s), int(a)) in allowed for s, a in zip(segments, anchor_ids)], dtype=bool
    )


def context_summary(ctx: SubjectContext) -> dict[str, Any]:
    per_split_hours: dict[str, float] = {}
    for interval in ctx.intervals:
        per_split_hours[interval.split] = (
            per_split_hours.get(interval.split, 0.0) + interval.duration / 3600.0
        )
    return {
        "subject": ctx.subject,
        "dataset": ctx.stream.dataset,
        "n_events_interictal": int(ctx.stream.features.n_events),
        "n_contacts": int(ctx.stream.n_contacts),
        "n_segments": len(ctx.tensors.timelines),
        "n_timeline_steps": int(sum(tl.n_steps for tl in ctx.tensors.timelines)),
        "split_hours": per_split_hours,
        "n_disjoint_blocks": {str(h): len(ctx.disjoint[h]) for h in ctx.horizons},
        "n_mark_features": int(ctx.stream.features.mark_features.shape[1]),
        "n_count_features": int(ctx.stream.features.count_features.shape[1]),
        "n_drive_features": int(ctx.tensors.n_drive_features),
        "support_hash": ctx.support_hash,
    }
