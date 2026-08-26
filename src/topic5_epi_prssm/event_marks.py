"""Frozen event representation: tie-aware recruitment groups, masks and node marks.

One position in a patient sequence is one complete interictal event.  An event is
a tuple of tied-rank contact sets ``(S_1, ..., S_K)`` recovered from the explicit
``event_group_ids`` identity, never by comparing equal rank values.  Contacts that
did not participate carry a mask, never a phantom rank.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .contracts import (
    DATASET_ROOT,
    FROZEN,
    SOURCE_MAPPING_ROOT,
    ForbiddenInputError,
    sha256_file,
)
from .sessions import SessionTable, build_sessions, session_boundaries

SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST = 0, 1, 2
SPLIT_NAMES = {SPLIT_TRAIN: "train", SPLIT_VALIDATION: "validation", SPLIT_TEST: "test"}

#: Columns of ``contact_features`` that are admitted as static node covariates.
#: ``prefix_participation_support*`` are rejected: their estimation partition is
#: not recoverable from the artefact, so re-using them would put an unverifiable
#: whole-record repertoire estimate into a train-only baseline.
ADMITTED_CONTACT_FEATURES = (
    "within_shaft_position",
    "shaft_size_fraction",
    "coord_x_centered_scaled",
    "coord_y_centered_scaled",
    "coord_z_centered_scaled",
    "geometry_present",
)
REJECTED_CONTACT_FEATURES = (
    "prefix_participation_support",
    "prefix_participation_support_centered",
)


@dataclass(frozen=True)
class PatientEvents:
    """Everything one patient contributes, with masks and splits already fixed."""

    subject: str
    dataset: str
    contact_names: np.ndarray            # (N,) str
    contact_coords: np.ndarray           # (N, 3) float32, mm
    contact_features: np.ndarray         # (N, F) float32, admitted static covariates only
    contact_feature_names: tuple[str, ...]

    participation: np.ndarray            # (E, N) bool
    group_ids: np.ndarray                # (E, N) int16, -1 where not participating
    group_count: np.ndarray              # (E,) int16
    normalized_rank: np.ndarray          # (E, N) float32, nan where not participating
    onset_indicator: np.ndarray          # (E, N) bool, group_id == 0
    event_time: np.ndarray               # (E,) float64
    delta_t: np.ndarray                  # (E,) float64, nan at each session opening
    split: np.ndarray                    # (E,) int8, 0 train / 1 validation / 2 test
    sessions: SessionTable
    session_opening: np.ndarray          # (E,) bool
    load: np.ndarray                     # (E,) float32, participating fraction
    source_hashes: dict[str, str]

    @property
    def n_events(self) -> int:
        return int(self.participation.shape[0])

    @property
    def n_contacts(self) -> int:
        return int(self.participation.shape[1])

    def split_mask(self, name: str) -> np.ndarray:
        inverse = {v: k for k, v in SPLIT_NAMES.items()}
        return self.split == inverse[name]

    def node_marks(self) -> np.ndarray:
        """``Y[e, i] = [participation, normalized rank, onset indicator]``.

        The rank channel is zero-filled where the contact did not participate and
        the participation channel is the mask that says so; no consumer may read
        the rank channel without the participation channel.
        """
        rank = np.where(self.participation, np.nan_to_num(self.normalized_rank), 0.0)
        return np.stack(
            [self.participation.astype(np.float32), rank.astype(np.float32),
             self.onset_indicator.astype(np.float32)],
            axis=-1,
        )


def available_subjects() -> tuple[str, ...]:
    return tuple(sorted(p.stem for p in DATASET_ROOT.glob("*.npz")))


@lru_cache(maxsize=64)
def load_patient(subject: str) -> PatientEvents:
    """Load, validate and split one patient.  Fails closed on any contract breach."""
    dataset_path = DATASET_ROOT / f"{subject}.npz"
    mapping_path = SOURCE_MAPPING_ROOT / f"{subject}.npz"
    with np.load(dataset_path, allow_pickle=True) as z:
        participation = np.asarray(z["event_participation"]).astype(bool)
        group_ids = np.asarray(z["event_group_ids"]).astype(np.int16)
        group_count = np.asarray(z["event_group_count"]).astype(np.int16)
        normalized_rank = np.asarray(z["event_local_rank"]).astype(np.float32)
        event_time = np.asarray(z["event_abs_time"]).astype(np.float64)
        legacy_split = np.asarray(z["event_split"]).astype(np.int8)
        contact_names = np.asarray(z["contact_names"]).astype(str)
        contact_coords = np.asarray(z["contact_coords"]).astype(np.float32)
        raw_features = np.asarray(z["contact_features"]).astype(np.float32)
        raw_feature_names = tuple(np.asarray(z["contact_feature_names"]).astype(str).tolist())
    with np.load(mapping_path, allow_pickle=True) as m:
        record_names = np.asarray(m["event_source_record_name"]).astype(str)

    _validate_event_encoding(subject, participation, group_ids, group_count, normalized_rank, event_time)

    keep = [i for i, name in enumerate(raw_feature_names) if name in ADMITTED_CONTACT_FEATURES]
    rejected = [n for n in raw_feature_names if n in REJECTED_CONTACT_FEATURES]
    if set(rejected) - set(REJECTED_CONTACT_FEATURES):
        raise ForbiddenInputError(f"{subject}: unexpected rejected feature set {rejected}")
    contact_features = raw_features[:, keep]
    contact_feature_names = tuple(raw_feature_names[i] for i in keep)

    sessions = build_sessions(subject, event_time, record_names)
    opening = session_boundaries(sessions.session_index)
    delta_t = np.full(len(event_time), np.nan)
    delta_t[1:] = np.diff(event_time)
    delta_t[opening] = np.nan

    split = _three_way_split(legacy_split)
    onset = group_ids == 0
    load = participation.sum(axis=1).astype(np.float32) / float(participation.shape[1])

    return PatientEvents(
        subject=subject,
        dataset=subject.split("_", 1)[0],
        contact_names=contact_names,
        contact_coords=contact_coords,
        contact_features=contact_features,
        contact_feature_names=contact_feature_names,
        participation=participation,
        group_ids=group_ids,
        group_count=group_count,
        normalized_rank=normalized_rank,
        onset_indicator=onset,
        event_time=event_time,
        delta_t=delta_t,
        split=split,
        sessions=sessions,
        session_opening=opening,
        load=load,
        source_hashes={
            "dataset_npz": sha256_file(dataset_path),
            "source_mapping_npz": sha256_file(mapping_path),
        },
    )


def _validate_event_encoding(subject, participation, group_ids, group_count, rank, times) -> None:
    if participation.shape != group_ids.shape or participation.shape != rank.shape:
        raise ValueError(f"{subject}: participation / group / rank shapes disagree")
    if np.any(np.diff(times) < 0):
        raise ValueError(f"{subject}: event_abs_time is not chronological")
    if np.any(group_ids[~participation] != -1):
        raise ValueError(f"{subject}: non-participating contacts carry a group id (phantom rank)")
    if np.any(np.isfinite(rank[~participation])):
        raise ValueError(f"{subject}: non-participating contacts carry a finite rank (phantom rank)")
    if np.any(~np.isfinite(rank[participation])):
        raise ValueError(f"{subject}: participating contacts carry a non-finite rank")
    if np.any(group_ids[participation] < 0):
        raise ValueError(f"{subject}: participating contacts carry a negative group id")
    n_participants = participation.sum(axis=1)
    if np.any(n_participants < 1):
        raise ValueError(f"{subject}: an event has no participating contact")
    # explicit group identity must be a dense 0..K-1 labelling per event
    for e in np.flatnonzero(group_count != n_participants)[:64]:
        labels = np.unique(group_ids[e][participation[e]])
        if labels.min() != 0 or labels.max() != len(labels) - 1:
            raise ValueError(f"{subject}: event {e} group ids are not a dense 0..K-1 labelling")
    if np.any(group_count > n_participants):
        raise ValueError(f"{subject}: group_count exceeds the number of participants")


def _three_way_split(legacy_split: np.ndarray) -> np.ndarray:
    """train / validation / test in chronological order.

    The dataset's own last-20% partition becomes the untouched test.  The first
    80% calibration partition is cut 75/25 chronologically, realising the frozen
    0.60 / 0.20 / 0.20 fractions without moving the sealed boundary.
    """
    split = np.full(len(legacy_split), SPLIT_TEST, dtype=np.int8)
    calibration = np.flatnonzero(legacy_split == 0)
    if len(calibration) == 0:
        raise ValueError("empty calibration partition")
    train_fraction = FROZEN["split_fractions"][0] / (
        FROZEN["split_fractions"][0] + FROZEN["split_fractions"][1]
    )
    cut = int(round(len(calibration) * train_fraction))
    cut = int(np.clip(cut, 1, len(calibration) - 1))
    split[calibration[:cut]] = SPLIT_TRAIN
    split[calibration[cut:]] = SPLIT_VALIDATION
    return split


def recruitment_groups(events: PatientEvents, index: int) -> list[np.ndarray]:
    """Ordered list of tied contact sets for one event, from explicit group identity."""
    mask = events.participation[index]
    gids = events.group_ids[index]
    order = np.unique(gids[mask])
    return [np.flatnonzero(mask & (gids == g)) for g in order]
