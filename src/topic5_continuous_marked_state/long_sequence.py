"""Full-event timeline used by recurrent T1/T2 models.

Unlike Bridge arrays, this dataset never drops an IED because its background
observation is unavailable.  Every immediate event-to-event transition in the
development partitions is present; continuous-SEEG features are attached with
an explicit availability mask.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import contract
from .bridge import BridgeArrays, _arm_matrix, _explicit_history
from .exposure import (
    cross_fitted_load_innovation,
    cross_fitted_participation_innovation,
    pre_event_innovation_predictors,
)


LONG_SEQUENCE_REVISION = "full_event_timeline_sparse_observation_v1"


@dataclass
class FullEventSequence:
    subject: str
    history: np.ndarray
    observation: np.ndarray
    observation_available: np.ndarray
    current_time: np.ndarray
    next_time: np.ndarray
    current_event_index: np.ndarray
    next_event_index: np.ndarray
    session: np.ndarray
    split: np.ndarray
    log_next_iei: np.ndarray
    next_participation: np.ndarray
    next_rank: np.ndarray
    next_stop_fraction: np.ndarray
    load_innovation: np.ndarray
    participation_innovation: np.ndarray

    def validate(self) -> None:
        n = len(self.split)
        for value in (
            self.history, self.observation, self.observation_available,
            self.current_time, self.next_time, self.current_event_index,
            self.next_event_index, self.session, self.log_next_iei,
            self.next_participation, self.next_rank, self.next_stop_fraction,
            self.load_innovation, self.participation_innovation,
        ):
            if len(value) != n:
                raise ValueError(f"{self.subject}: unequal full-sequence rows")
        if not np.array_equal(self.next_event_index, self.current_event_index + 1):
            raise ValueError(f"{self.subject}: full sequence skipped an event")
        if np.any(self.next_time <= self.current_time):
            raise ValueError(f"{self.subject}: non-positive event interval")
        if set(np.unique(self.split).tolist()) - {0, 1}:
            raise ValueError(f"{self.subject}: sealed row in full sequence")
        for code, name in ((0, "train"), (1, "validation")):
            mask = self.split == code
            contract.assert_development_times(self.subject, self.current_time[mask], name)
            contract.assert_development_times(self.subject, self.next_time[mask], name)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **{
                key: value for key, value in self.__dict__.items()
                if key != "subject"
            }, subject=np.asarray(self.subject))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "FullEventSequence":
        with np.load(path, allow_pickle=False) as z:
            kwargs = {key: z[key] for key in z.files if key != "subject"}
            value = cls(subject=str(z["subject"].item()), **kwargs)
        value.validate()
        return value


def build_full_event_sequence(subject: str,
                              observation_arm: str = "b1_spectral") -> FullEventSequence:
    payload = torch.load(contract.COHORT_CACHE, map_location="cpu", weights_only=False)[subject]
    times = payload["event_time"].numpy().astype(np.float64)
    session = payload["session_index"].numpy().astype(np.int64)
    participation = payload["participation"].numpy().astype(bool)
    n_groups = payload["n_groups"].numpy().astype(np.int64)
    marks = payload["marks"].numpy().astype(np.float32)
    load = payload["load"].numpy().astype(np.float32)
    history = _explicit_history(
        times, session, participation, n_groups, load, marks[:, :, 1],
        str(payload["dataset"]),
    )
    bound = contract.load_split(subject)
    event_split = np.full(len(times), 2, dtype=np.int8)
    event_split[times < bound.dev_end_epoch] = 1
    event_split[times < bound.train_end_epoch] = 0
    predictors = pre_event_innovation_predictors(history, participation)
    load_innovation = cross_fitted_load_innovation(predictors, load, event_split)
    participation_innovation = cross_fitted_participation_innovation(
        predictors, participation, event_split
    )

    pair_ok = (
        (session[1:] == session[:-1])
        & (event_split[1:] == event_split[:-1])
        & (np.diff(times) > 0)
        & (event_split[:-1] < 2)
    )
    idx = np.flatnonzero(pair_ok)
    nxt = idx + 1

    bridge = BridgeArrays.load(
        contract.RESULT_ROOT / "bridge/features" / f"{subject}.npz"
    )
    bridge_matrix, _ = _arm_matrix(bridge, observation_arm)
    bridge_observation = bridge_matrix[:, bridge.history.shape[1]:]
    observation = np.zeros((len(idx), contract.OBSERVATION_DIM), dtype=np.float32)
    observation_available = np.zeros(len(idx), dtype=bool)
    row_by_event = {int(event): row for row, event in enumerate(idx.tolist())}
    for bridge_row, event in enumerate(bridge.current_event_index.tolist()):
        target_row = row_by_event.get(int(event))
        if target_row is not None:
            observation[target_row] = bridge_observation[bridge_row]
            observation_available[target_row] = True

    result = FullEventSequence(
        subject=subject,
        history=history[idx].astype(np.float32),
        observation=observation,
        observation_available=observation_available,
        current_time=times[idx], next_time=times[nxt],
        current_event_index=idx.astype(np.int64),
        next_event_index=nxt.astype(np.int64),
        session=session[idx], split=event_split[idx],
        log_next_iei=np.log(np.maximum(times[nxt] - times[idx], 1e-3)).astype(np.float32),
        next_participation=participation[nxt].astype(np.float32),
        next_rank=marks[nxt, :, 1].astype(np.float32),
        next_stop_fraction=(n_groups[nxt] / participation.shape[1]).astype(np.float32),
        load_innovation=load_innovation[idx].astype(np.float32),
        participation_innovation=participation_innovation[idx].astype(np.float32),
    )
    result.validate()
    return result


def write_full_event_sequence(subject: str, output: Path) -> dict:
    sequence = build_full_event_sequence(subject)
    sequence.save(output)
    manifest = {
        "contract": contract.REVISION,
        "long_sequence_revision": LONG_SEQUENCE_REVISION,
        "subject": subject,
        "n_rows": int(len(sequence.split)),
        "n_train": int(np.sum(sequence.split == 0)),
        "n_validation": int(np.sum(sequence.split == 1)),
        "n_contacts": int(sequence.next_participation.shape[1]),
        "n_observation_available": int(sequence.observation_available.sum()),
        "observation_available_fraction": float(sequence.observation_available.mean()),
        "all_transitions_immediate_next_event": True,
        "sealed_opened": False,
        "output": str(output),
    }
    manifest_path = output.with_suffix(".manifest.json")
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp, manifest_path)
    return manifest
