"""Seizure onset times.

This module is the only place seizure labels enter the project, and it refuses
to load anything until ``INTERICTAL_MODEL_FREEZE.json`` exists on disk.  That is
Hard Gate B expressed as code rather than as a convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import (
    EPILEPSIAE_SEIZURE_INVENTORY, OUTPUT_ROOT, YUQUAN_SEIZURE_ROOT, ForbiddenInputError,
)

FREEZE = OUTPUT_ROOT / "manifests/INTERICTAL_MODEL_FREEZE.json"

#: local-time offsets used only for the day/night nuisance covariate
TIMEZONE_OFFSET_HOURS = {"epilepsiae": 1.0, "yuquan": 8.0}
DAY_START_HOUR, DAY_END_HOUR = 8, 20


@dataclass(frozen=True)
class Seizure:
    subject: str
    seizure_id: str
    onset_epoch: float
    offset_epoch: float        # end of the seizure; the post-ictal guard starts here
    onset_kind: str            # "clinical" or "eeg"
    local_hour: float
    day_night: str
    vigilance: str | None = None
    classification: str | None = None


def require_freeze() -> dict:
    if not FREEZE.exists():
        raise ForbiddenInputError(
            "Hard Gate B: seizure labels may not be read before "
            f"{FREEZE} exists. Freeze the interictal model family first.")
    return json.loads(FREEZE.read_text())


def load_seizures(subject: str) -> list[Seizure]:
    require_freeze()
    dataset, short = subject.split("_", 1)
    return _epilepsiae(subject, short) if dataset == "epilepsiae" else _yuquan(subject, short)


def _local(epoch: float, dataset: str) -> tuple[float, str]:
    offset = TIMEZONE_OFFSET_HOURS[dataset]
    stamp = datetime.fromtimestamp(epoch, tz=timezone.utc) + timedelta(hours=offset)
    hour = stamp.hour + stamp.minute / 60.0
    return hour, ("day" if DAY_START_HOUR <= hour < DAY_END_HOUR else "night")


def _epilepsiae(subject: str, short: str) -> list[Seizure]:
    frame = pd.read_csv(EPILEPSIAE_SEIZURE_INVENTORY)
    rows = frame[frame["subject"].astype(str) == short]
    out: list[Seizure] = []
    for row in rows.itertuples(index=False):
        onset = row.clin_onset_epoch
        offset = row.clin_offset_epoch
        kind = "clinical"
        if not np.isfinite(onset):
            onset, offset, kind = row.eeg_onset_epoch, row.eeg_offset_epoch, "eeg"
        if not np.isfinite(onset):
            continue
        if not np.isfinite(offset) or offset <= onset:
            # a missing offset must not silently shorten the post-ictal guard
            offset = float(onset) + 120.0
        hour, day_night = _local(float(onset), "epilepsiae")
        out.append(Seizure(subject, str(row.seizure_id), float(onset), float(offset), kind,
                           hour, day_night,
                           vigilance=str(getattr(row, "vigilance", "")) or None,
                           classification=str(getattr(row, "classification", "")) or None))
    return sorted(out, key=lambda s: s.onset_epoch)


def _yuquan(subject: str, short: str) -> list[Seizure]:
    path = YUQUAN_SEIZURE_ROOT / f"pr1_seizure_{short}.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    out: list[Seizure] = []
    for record in payload.get("files", []):
        for index, interval in enumerate(record.get("seizure_intervals", [])):
            onset = float(interval["onset_epoch"])
            offset = float(interval.get("offset_epoch", onset + 120.0))
            if not np.isfinite(offset) or offset <= onset:
                offset = onset + 120.0
            hour, day_night = _local(onset, "yuquan")
            out.append(Seizure(subject, f"{record['record']}_{index}", onset, offset, "eeg",
                               hour, day_night))
    return sorted(out, key=lambda s: s.onset_epoch)


def cohort_seizures(subjects) -> dict[str, list[Seizure]]:
    return {s: load_seizures(s) for s in subjects}
