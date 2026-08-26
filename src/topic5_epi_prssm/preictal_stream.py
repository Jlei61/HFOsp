"""Build model-ready tensors on the rebuilt full event stream.

The frozen per-contact quantities -- contact features, patient baseline, graph --
come from the cohort cache and are *not* recomputed here: the model is frozen, and
only the observations change.  What changes is which events the observer is allowed
to consume.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .contracts import OUTPUT_ROOT
from .cohort import load_tensors
from .model import SESSION_OPEN_DELTA_T, PatientTensors
from .sessions import build_sessions

STREAM_ROOT = OUTPUT_ROOT / "full_event_stream/per_subject"

#: frozen in the upstream dataset contract; an event inside a seizure or inside the
#: 120 min that follow its offset is not an interictal observation
POST_ICTAL_GUARD_SECONDS = 7200.0


@dataclass
class AdmissibleStream:
    """Events an online system would legitimately have observed."""

    tensors: PatientTensors
    event_time: np.ndarray
    in_definite_interictal: np.ndarray
    n_events_full: int
    n_events_admissible: int
    n_events_removed_ictal_or_postictal: int
    n_events_beyond_definite_interictal: int


def load_full_stream(subject: str) -> dict:
    with np.load(STREAM_ROOT / f"{subject}.npz", allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def build_admissible_stream(subject: str, seizure_intervals, *,
                            device: str = "cpu") -> AdmissibleStream:
    """Drop only ictal and post-ictal events; keep every pre-ictal one.

    ``seizure_intervals`` is a sequence of ``(onset_epoch, offset_epoch)``.
    """
    raw = load_full_stream(subject)
    frozen = load_tensors([subject])[0]
    times = np.asarray(raw["event_abs_time"], dtype=np.float64)
    keep = np.ones(len(times), dtype=bool)
    for onset, offset in seizure_intervals:
        stop = float(offset) + POST_ICTAL_GUARD_SECONDS
        keep &= ~((times >= float(onset)) & (times <= stop))
    n_removed = int((~keep).sum())

    participation = np.asarray(raw["event_participation"], dtype=bool)[keep]
    group_ids = np.asarray(raw["event_group_ids"], dtype=np.int64)[keep]
    group_count = np.asarray(raw["event_group_count"], dtype=np.int64)[keep]
    local_rank = np.asarray(raw["event_local_rank"], dtype=np.float32)[keep]
    record_names = np.asarray(raw["event_record_name"]).astype(str)[keep]
    in_definite = np.asarray(raw["in_definite_interictal"], dtype=bool)[keep]
    event_time = times[keep]

    sessions = build_sessions(subject, event_time, record_names)
    opening = np.zeros(len(event_time), dtype=bool)
    if len(opening):
        opening[0] = True
        opening[1:] = np.diff(sessions.session_index) != 0
    delta = np.full(len(event_time), np.nan)
    delta[1:] = np.diff(event_time)
    delta[opening] = np.nan
    delta[~np.isfinite(delta)] = SESSION_OPEN_DELTA_T
    delta = np.maximum(delta, 0.0)

    marks = np.stack([
        participation.astype(np.float32),
        np.where(participation, np.nan_to_num(local_rank), 0.0).astype(np.float32),
        (group_ids == 0).astype(np.float32),
    ], axis=-1)
    load = participation.sum(axis=1).astype(np.float32) / float(frozen.n_contacts)

    # the split label follows the frozen chronological boundaries, so a downstream
    # consumer can still tell development events from sealed-test-period events
    frozen_time = np.asarray(frozen.event_time)
    frozen_split = frozen.split.cpu().numpy()
    train_end = float(frozen_time[frozen_split == 0].max())
    val_end = float(frozen_time[frozen_split == 1].max())
    split = np.full(len(event_time), 2, dtype=np.int64)
    split[event_time <= val_end] = 1
    split[event_time <= train_end] = 0

    to = lambda a, d=torch.float32: torch.as_tensor(np.asarray(a), dtype=d, device=device)
    tensors = PatientTensors(
        subject=subject, dataset=frozen.dataset,
        participation=to(participation, torch.bool),
        group_ids=to(group_ids, torch.long),
        n_groups=to(group_count, torch.long),
        marks=to(marks), delta_t=to(delta), log_delta_t=to(np.log1p(delta)),
        session_open=to(opening.astype(np.float32)), load=to(load),
        split=to(split, torch.long), event_time=event_time,
        adjacency=frozen.adjacency, node_features=frozen.node_features,
        baseline_order=frozen.baseline_order,
        baseline_participation=frozen.baseline_participation,
        baseline_stop=frozen.baseline_stop,
        n_contacts=frozen.n_contacts, n_events=int(len(event_time)),
        meta={**frozen.meta, "stream": "admissible_full_stream",
              "session_index": sessions.session_index,
              "n_sessions": sessions.n_sessions},
    )
    return AdmissibleStream(
        tensors=tensors, event_time=event_time, in_definite_interictal=in_definite,
        n_events_full=int(len(times)), n_events_admissible=int(len(event_time)),
        n_events_removed_ictal_or_postictal=n_removed,
        n_events_beyond_definite_interictal=int((~in_definite).sum()),
    )


def observation_coverage(event_time: np.ndarray, probe: float, window: float) -> float:
    """Fraction of a look-back window that actually holds observed events.

    Measured as the share of ten equal sub-windows that contain at least one event,
    so a probe preceded by a recording gap is not matched against one preceded by
    continuous recording.
    """
    edges = np.linspace(probe - window, probe, 11)
    counts = np.histogram(event_time, bins=edges)[0]
    return float((counts > 0).mean())


def multiscale_rate(event_time: np.ndarray, probe: float,
                    windows=(1800.0, 7200.0, 14400.0, 28800.0)) -> dict[str, float]:
    """Events per hour in several look-back windows ending at ``probe``."""
    out = {}
    for window in windows:
        lo = np.searchsorted(event_time, probe - window, side="left")
        hi = np.searchsorted(event_time, probe, side="right")
        out[f"rate_{int(window)}s"] = (hi - lo) / (window / 3600.0)
    lo = np.searchsorted(event_time, probe - 7200.0, side="left")
    hi = np.searchsorted(event_time, probe, side="right")
    if hi - lo >= 3:
        out["median_iei"] = float(np.median(np.diff(event_time[lo:hi])))
    else:
        out["median_iei"] = float("nan")
    return out
