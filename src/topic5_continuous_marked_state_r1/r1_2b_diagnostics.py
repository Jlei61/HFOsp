"""Core persistent-state diagnostics used to close R1.2b.

These are scientific contrasts, not additional training arms.  They reuse the
fitted checkpoint and exact validation support to distinguish cross-anchor
state carry from a window-local observation code, and to localise mark gains.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from .r1_2 import (
    FullAnchorDesign,
    FrozenEmbeddingStateModel,
    _query_states,
    filtered_anchor_states,
    memoryless_anchor_states,
)


DIAGNOSTIC_REVISION = "r1_2b_persistent_memoryless_strict_swap_v1"


def strict_matched_wrong_time_permutations(
    design: FullAnchorDesign,
    observation_coverage: np.ndarray,
    *,
    anchor_segment: np.ndarray | None = None,
    split: str = "validation",
    n_donors: int = 5,
    min_separation_seconds: float = 1800.0,
    time_lower: float | None = None,
    time_upper: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return several causally valid matched wrong-time donors per anchor.

    Matching uses the frozen deterministic-history coordinates for time since
    the previous event, 30 s/2 min/10 min counts, last/recent load, previous
    event extent, time of day and session position, plus raw-window contact
    coverage.  The history coordinates are already scaled from TRAIN.  R1.4
    supplies the recorded-coverage segment explicitly so an apparently
    continuous clinical session cannot bridge an unrecorded gap.  The default
    preserves the frozen R1.3 same-session diagnostic.
    """
    code = {"train": 0, "validation": 1}[split]
    target_keep = design.anchor_split == code
    if time_lower is not None:
        target_keep &= design.anchor_time >= float(time_lower)
    if time_upper is not None:
        target_keep &= design.anchor_time < float(time_upper)
    target = np.flatnonzero(target_keep)
    coverage = np.asarray(observation_coverage, dtype=np.float64)
    if coverage.shape != (len(design.anchor_time),):
        raise ValueError("observation coverage must have one value per anchor")
    if anchor_segment is None:
        donor_group = np.asarray(design.anchor_session, dtype=np.int64)
        group_kind = "event_session"
    else:
        donor_group = np.asarray(anchor_segment, dtype=np.int64)
        if donor_group.shape != (len(design.anchor_time),):
            raise ValueError("anchor segment must have one value per anchor")
        if np.any(donor_group < 0):
            raise ValueError("anchor segment contains an invalid label")
        group_kind = "recorded_coverage_segment"
    # BASE_HISTORY_NAMES indices 1..10 exclude only has_previous_event.
    feature = np.asarray(design.anchor_history[:, 1:11], dtype=np.float64)
    train = design.anchor_split == 0
    cov_mean = float(np.mean(coverage[train])) if bool(train.any()) else 0.0
    cov_scale = float(np.std(coverage[train])) if bool(train.any()) else 1.0
    cov_scale = cov_scale if cov_scale > 1e-6 else 1.0
    feature = np.column_stack([feature, (coverage - cov_mean) / cov_scale])
    permutations = np.broadcast_to(
        np.arange(len(design.anchor_time), dtype=np.int64),
        (int(n_donors), len(design.anchor_time)),
    ).copy()
    matched = np.zeros(len(design.anchor_time), dtype=bool)
    candidate_counts = np.zeros(len(design.anchor_time), dtype=np.int64)
    distances: list[float] = []
    for row in target:
        candidate = target[
            (donor_group[target] == donor_group[row])
            & (
                np.abs(design.anchor_time[target] - design.anchor_time[row])
                >= float(min_separation_seconds)
            )
        ]
        candidate_counts[row] = len(candidate)
        if len(candidate) < int(n_donors):
            continue
        delta = feature[candidate] - feature[row]
        distance = np.sum(delta * delta, axis=1)
        order = np.lexsort((candidate, distance))[: int(n_donors)]
        donor = candidate[order]
        permutations[:, row] = donor
        distances.extend(distance[order].tolist())
        matched[row] = True
    audit = {
        "n_validation_anchors": int(len(target)),
        "n_matched_anchors": int(matched.sum()),
        "n_donors": int(n_donors),
        "minimum_separation_seconds": float(min_separation_seconds),
        "same_session": bool(anchor_segment is None),
        "same_recorded_coverage_segment": bool(anchor_segment is not None),
        "donor_group_kind": group_kind,
        "history_feature_indices": list(range(1, 11)),
        "observation_coverage_included": True,
        "candidate_count_median": (
            float(np.median(candidate_counts[target])) if len(target) else 0.0
        ),
        "selected_distance_median": (
            float(np.median(distances)) if distances else None
        ),
    }
    return permutations, matched, audit


@dataclass(frozen=True)
class MarkEndpointMetrics:
    mark_nll_per_event: float
    selecting_group_size_nll_per_event: float
    stop_nll_per_event: float
    first_group_subset_nll_per_event: float
    continuation_subset_nll_per_event: float
    same_prefix_continuation_nll_per_event: float | None
    n_events: int
    n_continuation_events: int
    n_same_prefix_continuation_events: int


def _repeated_first_prefix(group_ids: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Mark rows whose exact first tied group repeats within the scored set."""
    eligible = np.asarray(group_ids[rows] == 0, dtype=np.uint8)
    key = [np.packbits(value).tobytes() for value in eligible]
    count: dict[bytes, int] = {}
    for value in key:
        count[value] = count.get(value, 0) + 1
    return np.asarray([count[value] >= 2 for value in key], dtype=bool)


def evaluate_mark_endpoints(
    model: FrozenEmbeddingStateModel,
    design: FullAnchorDesign,
    embedding: np.ndarray,
    *,
    device: torch.device | str,
    split: str = "validation",
    anchor_state_mode: str = "persistent",
    state_permutation: np.ndarray | None = None,
    matched_anchor_mask: np.ndarray | None = None,
    time_lower: float | None = None,
    time_upper: float | None = None,
    anchor_state_override: torch.Tensor | None = None,
) -> MarkEndpointMetrics:
    """Split exact sequential mark likelihood into interpretable endpoints.

    ``continuation_subset`` is the teacher-forced identity likelihood after the
    first tied group, so it is sensitive to ordered recruitment beyond onset.
    ``same_prefix_continuation`` applies the same score only where the exact
    first tied group occurs at least twice in the scored validation set.
    """
    model.eval()
    code = {"train": 0, "validation": 1}[split]
    with torch.no_grad():
        if anchor_state_override is not None:
            anchor_state = anchor_state_override.to(device)
            if anchor_state.shape != (len(design.anchor_time), model.state.dim):
                raise ValueError("anchor_state_override shape disagrees with design")
        elif anchor_state_mode == "persistent":
            anchor_state = filtered_anchor_states(
                model, design, embedding, device=device
            )
        elif anchor_state_mode == "memoryless":
            anchor_state = memoryless_anchor_states(
                model, design, embedding, device=device
            )
        else:
            raise ValueError(f"unknown anchor_state_mode {anchor_state_mode!r}")
        keep = design.event_split == code
        if time_lower is not None:
            keep &= design.event_time >= float(time_lower)
        if time_upper is not None:
            keep &= design.event_time < float(time_upper)
        if matched_anchor_mask is not None:
            matched_anchor_mask = np.asarray(matched_anchor_mask, dtype=bool)
            keep &= (
                (design.event_source_anchor >= 0)
                & matched_anchor_mask[np.maximum(design.event_source_anchor, 0)]
            )
        rows = np.flatnonzero(keep)
        state = _query_states(
            model, design, anchor_state, design.event_source_anchor,
            design.event_time, design.event_session, rows,
            state_permutation=state_permutation, device=device,
        )
        repeated = _repeated_first_prefix(design.event_group_ids, rows)
        totals = np.zeros(6, dtype=np.float64)
        n_continuation = 0
        n_same_prefix = 0
        for lo in range(0, len(rows), 4096):
            take = rows[lo:lo + 4096]
            local_state = state[lo:lo + len(take)]
            terms = model.mark_terms(
                torch.as_tensor(design.event_history[take], device=device),
                local_state,
                torch.as_tensor(
                    design.event_group_ids[take], dtype=torch.long, device=device
                ),
                torch.as_tensor(
                    design.event_group_count[take], dtype=torch.long, device=device
                ),
            )
            steps = torch.arange(
                terms.group_size_step_log_prob.shape[1], device=device
            ).unsqueeze(0)
            selecting = terms.select_step
            terminal = terms.active_step & ~terms.select_step
            first = selecting & (steps == 0)
            continuation = selecting & (steps >= 1)
            repeated_local = torch.as_tensor(
                repeated[lo:lo + len(take)], dtype=torch.bool, device=device
            ).unsqueeze(-1)
            same_prefix = continuation & repeated_local
            totals += np.asarray([
                float(terms.event_log_prob.sum()),
                float(terms.group_size_step_log_prob[selecting].sum()),
                float(terms.group_size_step_log_prob[terminal].sum()),
                float(terms.subset_step_log_prob[first].sum()),
                float(terms.subset_step_log_prob[continuation].sum()),
                float(terms.subset_step_log_prob[same_prefix].sum()),
            ])
            has_continuation = np.asarray(design.event_group_count[take] >= 2)
            n_continuation += int(has_continuation.sum())
            n_same_prefix += int((has_continuation & repeated[lo:lo + len(take)]).sum())
    denominator = max(len(rows), 1)
    return MarkEndpointMetrics(
        mark_nll_per_event=-totals[0] / denominator,
        selecting_group_size_nll_per_event=-totals[1] / denominator,
        stop_nll_per_event=-totals[2] / denominator,
        first_group_subset_nll_per_event=-totals[3] / denominator,
        continuation_subset_nll_per_event=-totals[4] / max(n_continuation, 1),
        same_prefix_continuation_nll_per_event=(
            -totals[5] / n_same_prefix if n_same_prefix else None
        ),
        n_events=int(len(rows)),
        n_continuation_events=int(n_continuation),
        n_same_prefix_continuation_events=int(n_same_prefix),
    )


def median_metric_dict(values: list[dict]) -> dict:
    """Patient/seed-local median across several matched wrong-time donors."""
    result: dict[str, float | int | None] = {}
    for key in values[0]:
        row = [value[key] for value in values if value[key] is not None]
        result[key] = float(np.median(row)) if row else None
    return result


def metric_contrast(left: dict, right: dict) -> dict:
    result = {}
    for key, value in left.items():
        if key.startswith("n_") or value is None or right.get(key) is None:
            continue
        result[key] = float(value) - float(right[key])
    return result
