"""Frozen data semantics for Group-Event State v0.1.

This module deliberately contains no model code.  Its job is to make one
complete group event traceable back to the exact source block and native sample
range, while keeping reference montage and unavailable frequency bands explicit.

The legacy ``lagPatRaw`` value is a spectrogram centroid on a stitched segment
timeline.  Only participant-masked *within-event differences* are portable.
It is never interpreted as a detector onset or as an absolute record time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Mapping, Sequence

import numpy as np


CONTRACT_NAME = "topic5_group_event_state_v0_1"
CONTRACT_VERSION = "0.1.0"

# The packed group window is the scientific event core.  These shoulders are
# available to the waveform encoder but are not used to redefine event identity.
EVENT_CONTEXT_PRE_SECONDS = 0.250
EVENT_CONTEXT_POST_SECONDS = 0.250

# Bands are computed at native sampling rate.  A band is absent, rather than
# zero, whenever its upper edge plus the Nyquist guard is unsupported.
ANALYSIS_BANDS_HZ: Mapping[str, tuple[float, float]] = {
    "ied_low": (1.0, 30.0),
    "gamma": (30.0, 70.0),
    "low_ripple": (80.0, 110.0),
    "ripple": (80.0, 150.0),
    "fast_ripple": (150.0, 250.0),
}
BAND_NYQUIST_GUARD_HZ = 8.0

# The legacy centroid is read off a spectrogram with ``nperseg = 0.05 * fs`` and
# 80% overlap, so successive centroid estimates are one 10 ms hop apart at every
# sampling rate in this cohort.  Two participants closer than one hop are not
# resolvably ordered; they form one tied recruitment group.  The exact delay is
# always kept alongside, so this label never replaces the continuous value.
CENTROID_HOP_SECONDS = 0.010
TIE_TOLERANCE_SECONDS = CENTROID_HOP_SECONDS

# Two consecutive recording blocks written by the same acquisition run abut to
# within a sample or two.  Anything larger is an unobserved interval and must
# break the state sequence rather than be bridged.
SEAM_TOLERANCE_SECONDS = 2.0


@dataclass(frozen=True)
class LagPatVariant:
    """One legacy packer's matched (lagPat, packedTimes) file pair.

    The two packers disagree on the event list for the same recording
    (``FC10477Q``: 2965 rows from the older packer, 2601 from ``withFreqCent``),
    so the packed-time file must be selected by *variant*, never by existence.
    """

    name: str
    lagpat_suffix: str
    packed_suffix: str
    has_frequency_centroid: bool


LAGPAT_VARIANTS: tuple[LagPatVariant, ...] = (
    LagPatVariant(
        name="withFreqCent",
        lagpat_suffix="_lagPat_withFreqCent.npz",
        packed_suffix="_packedTimes_withFreqCent.npy",
        has_frequency_centroid=True,
    ),
    LagPatVariant(
        name="legacy",
        lagpat_suffix="_lagPat.npz",
        packed_suffix="_packedTimes.npy",
        has_frequency_centroid=False,
    ),
)

# No labels from these families may enter the interictal representation learner.
FORBIDDEN_INPUT_FAMILIES = frozenset(
    {
        "seizure_label",
        "seizure_onset",
        "preictal_label",
        "soz",
        "resection",
        "clinical_outcome",
        "template_label",
        "kmeans_label",
        "future_event",
        "future_mark",
    }
)


@dataclass(frozen=True)
class EventSourcePointer:
    """Lossless pointer from a model event to its native recording.

    ``core_*`` is the original packed group-event interval. ``context_*`` adds
    the fixed shoulders above. Sample indices are half-open and index the native
    recording before re-referencing or resampling.
    """

    dataset: str
    subject: str
    record_name: str
    source_block_id: int
    source_event_row: int
    raw_path: str
    head_path: str | None
    native_rate_hz: float
    block_start_epoch: float
    core_start_seconds: float
    core_end_seconds: float
    core_start_sample: int
    core_stop_sample: int
    context_start_sample: int
    context_stop_sample: int
    detector_reference: str

    @property
    def event_abs_time(self) -> float:
        return float(self.block_start_epoch + self.core_start_seconds)


def supported_band_mask(
    native_rate_hz: float,
    bands: Mapping[str, tuple[float, float]] = ANALYSIS_BANDS_HZ,
) -> dict[str, bool]:
    """Return explicit per-band availability under the native Nyquist limit."""

    nyquist = float(native_rate_hz) / 2.0
    return {
        str(name): bool(float(high) + BAND_NYQUIST_GUARD_HZ < nyquist)
        for name, (_low, high) in bands.items()
    }


def relative_participant_delay(
    lag_raw: np.ndarray,
    participation: np.ndarray,
) -> np.ndarray:
    """Participant-masked centroid delay relative to the event's first centroid.

    Non-participants are always NaN even when the legacy file contains a finite
    phantom value.  The function accepts ``(..., contacts)`` arrays.
    """

    lag = np.asarray(lag_raw, dtype=np.float64)
    mask = np.asarray(participation, dtype=bool)
    if lag.shape != mask.shape:
        raise ValueError(f"lag/participation shape mismatch: {lag.shape} != {mask.shape}")
    valid = mask & np.isfinite(lag)
    masked = np.where(valid, lag, np.nan)
    with np.errstate(all="ignore"):
        first = np.nanmin(masked, axis=-1, keepdims=True)
    out = masked - first
    out[~valid] = np.nan
    return out


def tied_recruitment_groups(
    relative_delay: np.ndarray,
    participation: np.ndarray,
    *,
    tolerance_seconds: float = TIE_TOLERANCE_SECONDS,
) -> list[list[int]]:
    """Group one event's participants into resolvably-ordered recruitment steps.

    Participants are sorted by relative delay and split wherever the step to the
    next participant exceeds one centroid hop (single linkage).  Non-participants
    never appear.  Returns contact indices, groups ordered earliest-first.

    The legacy ``argsort(argsort(x))`` rank destroys exactly this information by
    forcing a total order on values that the producer cannot resolve.
    """

    delay = np.asarray(relative_delay, dtype=np.float64).reshape(-1)
    mask = np.asarray(participation, dtype=bool).reshape(-1)
    if delay.shape != mask.shape:
        raise ValueError(f"delay/participation shape mismatch: {delay.shape} != {mask.shape}")
    idx = np.flatnonzero(mask & np.isfinite(delay))
    if idx.size == 0:
        return []
    order = idx[np.argsort(delay[idx], kind="stable")]
    groups: list[list[int]] = [[int(order[0])]]
    for prev, cur in zip(order[:-1], order[1:]):
        if float(delay[cur] - delay[prev]) > float(tolerance_seconds):
            groups.append([int(cur)])
        else:
            groups[-1].append(int(cur))
    return groups


_CONTACT_RE = re.compile(r"^\s*([A-Za-z]+['\u2032]?)(\d+)\s*$")


def adjacent_bipolar_label(anode_label: str) -> str | None:
    """Translate legacy Yuquan ``E11`` into its true ``E11-E12`` montage label."""

    match = _CONTACT_RE.match(str(anode_label))
    if match is None:
        return None
    shaft, number = match.groups()
    return f"{shaft.upper()}{int(number)}-{shaft.upper()}{int(number) + 1}"


def map_detector_channels(
    dataset: str,
    lagpat_labels: Sequence[str],
    detector_labels: Sequence[str],
) -> tuple[list[str | None], list[str]]:
    """Map lagPat rows to the exact detector montage without guessing silently.

    Epilepsiae legacy group events are CAR contact rows and should match by exact
    label.  Yuquan legacy lagPat rows use the anode label for an adjacent bipolar
    detector channel, so ``A1`` maps to ``A1-A2``.
    """

    available = {str(value).strip().upper(): str(value) for value in detector_labels}
    mapped: list[str | None] = []
    failures: list[str] = []
    for raw_label in lagpat_labels:
        label = str(raw_label).strip().upper()
        candidate = label
        if candidate not in available and str(dataset).lower() == "yuquan":
            candidate = adjacent_bipolar_label(label) or ""
        if candidate in available:
            mapped.append(available[candidate])
        else:
            mapped.append(None)
            failures.append(str(raw_label))
    return mapped, failures


def seconds_to_native_sample(seconds: float, native_rate_hz: float) -> int:
    """Round a source time to its nearest native sample deterministically."""

    if not math.isfinite(float(seconds)) or not math.isfinite(float(native_rate_hz)):
        raise ValueError("seconds and native_rate_hz must be finite")
    if float(native_rate_hz) <= 0:
        raise ValueError("native_rate_hz must be positive")
    return int(np.rint(float(seconds) * float(native_rate_hz)))


def build_source_pointer(
    *,
    dataset: str,
    subject: str,
    record_name: str,
    source_block_id: int,
    source_event_row: int,
    raw_path: str,
    head_path: str | None,
    native_rate_hz: float,
    block_start_epoch: float,
    core_start_seconds: float,
    core_end_seconds: float,
    detector_reference: str,
    n_native_samples: int | None = None,
) -> EventSourcePointer:
    """Construct and range-check one native event pointer."""

    if core_end_seconds < core_start_seconds:
        raise ValueError("event core ends before it starts")
    core_start = seconds_to_native_sample(core_start_seconds, native_rate_hz)
    core_stop = seconds_to_native_sample(core_end_seconds, native_rate_hz)
    context_start = max(
        0,
        seconds_to_native_sample(
            core_start_seconds - EVENT_CONTEXT_PRE_SECONDS, native_rate_hz
        ),
    )
    context_stop = seconds_to_native_sample(
        core_end_seconds + EVENT_CONTEXT_POST_SECONDS, native_rate_hz
    )
    if n_native_samples is not None:
        if core_start < 0 or core_stop > int(n_native_samples):
            raise ValueError("event core lies outside the native recording")
        context_stop = min(context_stop, int(n_native_samples))
    if core_stop <= core_start:
        raise ValueError("event core contains no native sample")
    return EventSourcePointer(
        dataset=str(dataset),
        subject=str(subject),
        record_name=str(record_name),
        source_block_id=int(source_block_id),
        source_event_row=int(source_event_row),
        raw_path=str(raw_path),
        head_path=None if head_path is None else str(head_path),
        native_rate_hz=float(native_rate_hz),
        block_start_epoch=float(block_start_epoch),
        core_start_seconds=float(core_start_seconds),
        core_end_seconds=float(core_end_seconds),
        core_start_sample=core_start,
        core_stop_sample=core_stop,
        context_start_sample=context_start,
        context_stop_sample=context_stop,
        detector_reference=str(detector_reference),
    )



def contact_shaft(label: str) -> str | None:
    """Electrode shaft of a single-contact label (``GD8`` -> ``GD``)."""

    match = _CONTACT_RE.match(str(label))
    return None if match is None else match.group(1).upper()


def bipolar_anode(label: str) -> str:
    """Anode of a montage label (``E11-E12`` -> ``E11``; ``GD8`` -> ``GD8``)."""

    text = str(label).strip()
    return text.split("-", 1)[0].strip().upper() if "-" in text else text.upper()
