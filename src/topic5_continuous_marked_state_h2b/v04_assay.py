"""Real-coverage semi-synthetic utilities for H2b v0.4 route assays."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .v03_hazard import HazardDesign


def zscore(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    centre = np.mean(array, axis=0, keepdims=True)
    scale = np.std(array, axis=0, keepdims=True)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (array - centre) / scale


def synthetic_slow_pair(time_epoch: np.ndarray, segment: np.ndarray) -> np.ndarray:
    """Two deterministic slow coordinates that reset at real coverage gaps."""
    time = np.asarray(time_epoch, dtype=np.float64)
    group = np.asarray(segment, dtype=np.int64)
    result = np.zeros((len(time), 2), dtype=np.float64)
    for label in np.unique(group):
        rows = np.flatnonzero(group == label)
        rows = rows[np.argsort(time[rows], kind="stable")]
        if not len(rows):
            continue
        elapsed_hours = (time[rows] - float(time[rows[0]])) / 3600.0
        phase_offset = (int(label) * 0.61803398875) % 1.0
        result[rows, 0] = np.sin(
            2.0 * np.pi * (elapsed_hours / 5.5 + phase_offset)
        )
        result[rows, 1] = np.cos(
            2.0 * np.pi * (elapsed_hours / 9.0 + 0.5 * phase_offset)
        )
    return zscore(result)


def inject_slow_state(design: HazardDesign, amplitude: float = 2.5) -> tuple[np.ndarray, np.ndarray]:
    """Create a fixed-width positive-control state with two slow active axes.

    The remaining dimensions are structural zeros.  This deliberately tests
    whether the route estimator can recover an effect that is present in its
    admissible representation, rather than conflating estimator power with the
    real checkpoint's instrument quality.  Real-data analyses never call this
    function.
    """
    persistent = zscore(design.persistent_state)
    slow = synthetic_slow_pair(design.time_epoch, design.segment)
    if persistent.shape[1] < 2:
        raise ValueError("synthetic route assay requires at least two state dimensions")
    result = np.zeros_like(persistent)
    # The positive control has one slow coordinate with two distant ends.
    # That is the minimal nonlinear union that a two-route readout should
    # recover and a one-route distance should not conflate with another axis.
    result[:, 0] = float(amplitude) * slow[:, 0] + 0.02 * persistent[:, 0]
    return result, slow


def candidate_anchor_rows(
    design: HazardDesign,
    *,
    horizon_minutes: float = 30.0,
    minimum_prior_rows: int = 30,
) -> np.ndarray:
    time = np.asarray(design.time_epoch, dtype=np.float64)
    group = np.asarray(design.segment, dtype=np.int64)
    horizon = float(horizon_minutes) * 60.0
    rows = []
    for index in range(int(minimum_prior_rows), len(time)):
        local = np.flatnonzero(group == group[index])
        if not len(local):
            continue
        if time[index] + horizon <= float(np.max(time[local])) + 1e-9:
            rows.append(index)
    return np.asarray(rows, dtype=np.int64)


def sample_synthetic_onsets(
    design: HazardDesign,
    score: np.ndarray,
    *,
    rng: np.random.Generator,
    n_seizures: int = 10,
    horizon_minutes: float = 30.0,
    minimum_separation_minutes: float = 180.0,
    balance: np.ndarray | None = None,
    strength: float = 2.5,
    maximum_attempts: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select exact lead anchors by weighted, separated Gumbel ranking."""
    candidate = candidate_anchor_rows(design, horizon_minutes=horizon_minutes)
    if len(candidate) < int(n_seizures):
        raise ValueError("too few real-coverage synthetic seizure candidates")
    values = zscore(np.asarray(score, dtype=np.float64).reshape(-1, 1))[:, 0]
    if len(values) != len(design.time_epoch):
        raise ValueError("synthetic score rows disagree")
    group = np.asarray(design.segment, dtype=np.int64)
    time = np.asarray(design.time_epoch, dtype=np.float64)
    separation = float(minimum_separation_minutes) * 60.0
    balance_value = None if balance is None else np.asarray(balance, dtype=bool)
    if balance_value is not None and len(balance_value) != len(time):
        raise ValueError("route-balance labels disagree")

    def admissible(row: int, selected: list[int]) -> bool:
        return all(
            group[previous] != group[row]
            or abs(float(time[previous] - time[row])) >= separation
            for previous in selected
        )

    for _ in range(int(maximum_attempts)):
        utility = float(strength) * values[candidate] + rng.gumbel(size=len(candidate))
        selected: list[int] = []
        if balance_value is None:
            for local in np.argsort(-utility, kind="stable"):
                row = int(candidate[local])
                if admissible(row, selected):
                    selected.append(row)
                if len(selected) == int(n_seizures):
                    break
        else:
            targets = (int(n_seizures) // 2, int(n_seizures) - int(n_seizures) // 2)
            # Alternate routes so one early route cannot consume all temporal
            # support needed by the other.
            ordered = {
                flag: [
                    int(candidate[local]) for local in np.argsort(-utility, kind="stable")
                    if int(balance_value[candidate[local]]) == flag
                ]
                for flag in (0, 1)
            }
            cursors = [0, 0]
            counts = [0, 0]
            while sum(counts) < int(n_seizures):
                progressed = False
                for flag in (0, 1):
                    if counts[flag] >= targets[flag]:
                        continue
                    while cursors[flag] < len(ordered[flag]):
                        row = ordered[flag][cursors[flag]]
                        cursors[flag] += 1
                        if admissible(row, selected):
                            selected.append(row)
                            counts[flag] += 1
                            progressed = True
                            break
                if not progressed:
                    break
        if len(selected) == int(n_seizures):
            anchor = np.asarray(sorted(selected, key=lambda row: time[row]), dtype=np.int64)
            onset = time[anchor] + float(horizon_minutes) * 60.0
            onset_group = group[anchor]
            return onset, onset_group, anchor
    raise ValueError("could not sample separated synthetic seizures")


def apply_synthetic_postictal_exclusion(
    design: HazardDesign,
    onset_time: np.ndarray,
    onset_original_segment: np.ndarray,
    *,
    postictal_minutes: float = 120.0,
    maximum_grid_gap_seconds: float = 450.0,
) -> tuple[HazardDesign, np.ndarray]:
    """Remove synthetic postictal rows and split every resulting coverage gap."""
    onset = np.asarray(onset_time, dtype=np.float64)
    onset_group = np.asarray(onset_original_segment, dtype=np.int64)
    keep = np.ones(len(design.time_epoch), dtype=bool)
    postictal = float(postictal_minutes) * 60.0
    for event, label in zip(onset, onset_group):
        keep &= ~(
            (design.segment == int(label))
            & (design.time_epoch >= float(event) - 1e-9)
            & (design.time_epoch <= float(event) + postictal + 1e-9)
        )
    take = np.flatnonzero(keep)
    if len(take) < 30:
        raise ValueError("synthetic postictal exclusion removed too much coverage")
    new_segment = np.full(len(take), -1, dtype=np.int64)
    next_label = 0
    for old_label in np.unique(design.segment[take]):
        local = np.flatnonzero(design.segment[take] == old_label)
        local = local[np.argsort(design.time_epoch[take[local]], kind="stable")]
        previous = None
        for position in local:
            current = float(design.time_epoch[take[position]])
            if previous is None or current - previous > float(maximum_grid_gap_seconds):
                next_label += 1
            new_segment[position] = next_label
            previous = current
    mapped_onset_segment = []
    for event, old_label in zip(onset, onset_group):
        local = np.flatnonzero(
            (design.segment[take] == int(old_label))
            & (design.time_epoch[take] < float(event))
        )
        if not len(local):
            raise ValueError("synthetic onset lacks a retained causal anchor")
        position = int(local[np.argmax(design.time_epoch[take[local]])])
        mapped_onset_segment.append(int(new_segment[position]))
    value = HazardDesign(
        source_index=design.source_index[take],
        time_epoch=design.time_epoch[take],
        segment=new_segment,
        history=design.history[take],
        current_observation=design.current_observation[take],
        persistent_state=design.persistent_state[take],
        memoryless_state=design.memoryless_state[take],
        onset_time=onset,
        onset_segment=np.asarray(mapped_onset_segment, dtype=np.int64),
    )
    value.validate()
    return value, take


def replace_states(
    design: HazardDesign,
    *,
    persistent_state: np.ndarray,
    memoryless_state: np.ndarray | None = None,
) -> HazardDesign:
    return replace(
        design,
        persistent_state=np.asarray(persistent_state, dtype=np.float64),
        memoryless_state=(
            design.memoryless_state if memoryless_state is None
            else np.asarray(memoryless_state, dtype=np.float64)
        ),
    )
