"""Exact recorded-coverage segments for R1 survival likelihoods."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np

from . import contract


COVERAGE_REVISION = "metadata_recorded_segment_union_v1"


def merge_intervals(start: np.ndarray, stop: np.ndarray,
                    *, tolerance_seconds: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Return the union of finite positive intervals in chronological order."""
    start = np.asarray(start, dtype=np.float64)
    stop = np.asarray(stop, dtype=np.float64)
    if start.shape != stop.shape or start.ndim != 1:
        raise ValueError("coverage starts/stops must be equal-length 1-D arrays")
    if not len(start):
        raise ValueError("coverage is empty")
    if not np.isfinite(start).all() or not np.isfinite(stop).all():
        raise ValueError("coverage contains non-finite boundary")
    if np.any(stop <= start):
        raise ValueError("coverage contains a non-positive interval")
    order = np.argsort(start, kind="stable")
    start, stop = start[order], stop[order]
    out_start = [float(start[0])]
    out_stop = [float(stop[0])]
    for left, right in zip(start[1:], stop[1:]):
        # Only true overlap/abutment is unioned.  The frozen session resolver
        # clamps a small *negative* metadata gap caused by rounding; it never
        # converts a small positive gap into recorded time.  Doing the latter
        # silently inflated survival exposure by one to two seconds at dozens
        # of Epilepsiae block boundaries.
        overlap = out_stop[-1] - float(left)
        if overlap > float(tolerance_seconds):
            raise ValueError(
                f"coverage intervals overlap by {overlap:.6f}s, beyond tolerance"
            )
        if float(left) <= out_stop[-1]:
            out_stop[-1] = max(out_stop[-1], float(right))
        else:
            out_start.append(float(left))
            out_stop.append(float(right))
    return np.asarray(out_start), np.asarray(out_stop)


def merge_labeled_intervals(start: np.ndarray, stop: np.ndarray,
                            label: np.ndarray,
                            *, tolerance_seconds: float = 2.0
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Union overlap/abutment only within the same continuity label."""
    start = np.asarray(start, dtype=np.float64)
    stop = np.asarray(stop, dtype=np.float64)
    label = np.asarray(label, dtype=np.int64)
    if start.shape != stop.shape or start.shape != label.shape or start.ndim != 1:
        raise ValueError("labeled coverage arrays disagree")
    order = np.argsort(start, kind="stable")
    start, stop, label = start[order], stop[order], label[order]
    out_start: list[float] = []
    out_stop: list[float] = []
    out_label: list[int] = []
    for left, right, current_label in zip(start, stop, label):
        if not np.isfinite(left) or not np.isfinite(right) or right <= left:
            raise ValueError("invalid labeled coverage interval")
        if not out_start:
            out_start.append(float(left)); out_stop.append(float(right)); out_label.append(int(current_label))
            continue
        overlap = out_stop[-1] - float(left)
        if overlap > float(tolerance_seconds):
            raise ValueError(f"coverage intervals overlap by {overlap:.6f}s")
        if int(current_label) == out_label[-1] and float(left) <= out_stop[-1]:
            out_stop[-1] = max(out_stop[-1], float(right))
        else:
            # A continuity reset is allowed at an exactly abutting boundary.
            # A true overlap across two labels would double-count recorded time.
            if float(left) < out_stop[-1]:
                raise ValueError("different continuity labels overlap in recorded time")
            out_start.append(float(left)); out_stop.append(float(right)); out_label.append(int(current_label))
    return np.asarray(out_start), np.asarray(out_stop), np.asarray(out_label)


def clip_intervals(start: np.ndarray, stop: np.ndarray,
                   lower: float, upper: float) -> tuple[np.ndarray, np.ndarray]:
    """Intersect coverage with ``[lower, upper)``."""
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("invalid clipping bounds")
    left = np.maximum(np.asarray(start, dtype=np.float64), float(lower))
    right = np.minimum(np.asarray(stop, dtype=np.float64), float(upper))
    keep = right > left
    return left[keep], right[keep]


def recorded_duration_between(start: np.ndarray, stop: np.ndarray,
                              left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Recorded duration inside each pair of query boundaries."""
    start = np.asarray(start, dtype=np.float64)
    stop = np.asarray(stop, dtype=np.float64)
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("query boundary shapes disagree")
    if np.any(right < left):
        raise ValueError("query right boundary precedes left boundary")
    flat_l = left.reshape(-1, 1)
    flat_r = right.reshape(-1, 1)
    overlap = np.maximum(
        np.minimum(flat_r, stop.reshape(1, -1))
        - np.maximum(flat_l, start.reshape(1, -1)),
        0.0,
    )
    return overlap.sum(axis=1).reshape(left.shape)


@dataclass(frozen=True)
class CoverageTable:
    subject: str
    start: np.ndarray
    stop: np.ndarray
    session: np.ndarray
    train_end_epoch: float
    dev_end_epoch: float
    source_hashes: dict[str, str]

    def validate(self) -> None:
        if (self.start.shape != self.stop.shape or self.start.shape != self.session.shape
                or self.start.ndim != 1):
            raise ValueError(f"{self.subject}: invalid coverage shapes")
        if not len(self.start) or np.any(self.stop <= self.start):
            raise ValueError(f"{self.subject}: empty/non-positive coverage")
        if np.any(self.start[1:] < self.stop[:-1]):
            raise ValueError(f"{self.subject}: overlapping coverage union")
        if self.start.max() >= self.dev_end_epoch and not np.any(self.start < self.dev_end_epoch):
            raise ValueError(f"{self.subject}: no development coverage")

    def split_segments(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        left, right, _ = self.split_segments_with_session(split)
        return left, right

    def split_segments_with_session(
        self, split: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if split == "train":
            lower, upper = float(self.start.min()), self.train_end_epoch
        elif split == "validation":
            lower, upper = self.train_end_epoch, self.dev_end_epoch
        else:
            raise ValueError(f"unknown development split {split!r}")
        left = np.maximum(self.start, lower)
        right = np.minimum(self.stop, upper)
        keep = right > left
        result = (left[keep], right[keep], self.session[keep])
        if not len(result[0]):
            raise ValueError(f"{self.subject}: no {split} coverage")
        return result

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as handle:
            np.savez_compressed(
                handle,
                subject=np.asarray(self.subject),
                start=self.start,
                stop=self.stop,
                session=self.session,
                train_end_epoch=np.asarray(self.train_end_epoch),
                dev_end_epoch=np.asarray(self.dev_end_epoch),
                source_hashes_json=np.asarray(json.dumps(self.source_hashes, sort_keys=True)),
            )
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "CoverageTable":
        with np.load(path, allow_pickle=False) as data:
            value = cls(
                subject=str(data["subject"].item()),
                start=data["start"].astype(np.float64),
                stop=data["stop"].astype(np.float64),
                session=data["session"].astype(np.int64),
                train_end_epoch=float(data["train_end_epoch"].item()),
                dev_end_epoch=float(data["dev_end_epoch"].item()),
                source_hashes=json.loads(str(data["source_hashes_json"].item())),
            )
        value.validate()
        return value


def build_coverage(subject: str) -> tuple[CoverageTable, dict]:
    """Build metadata coverage and audit it against the older duration artifact."""
    from src.topic5_epi_prssm.event_marks import SOURCE_MAPPING_ROOT, load_patient
    from src.topic5_epi_prssm.sessions import build_sessions

    events = load_patient(subject)
    mapping_path = SOURCE_MAPPING_ROOT / f"{subject}.npz"
    with np.load(mapping_path, allow_pickle=True) as mapping:
        record_names = np.asarray(mapping["event_source_record_name"]).astype(str)
    table = build_sessions(subject, events.event_time, record_names)
    raw_start = np.asarray([block.start_epoch for block in table.blocks])
    raw_stop = np.asarray([block.stop_epoch for block in table.blocks])
    start, stop, segment_session = merge_labeled_intervals(
        raw_start, raw_stop, np.asarray(table.block_session, dtype=np.int64)
    )
    train_end, dev_end = contract.load_split(subject)
    value = CoverageTable(
        subject=subject,
        start=start,
        stop=stop,
        session=segment_session,
        train_end_epoch=train_end,
        dev_end_epoch=dev_end,
        source_hashes={
            **events.source_hashes,
            "source_mapping_npz": contract.sha256_file(mapping_path),
        },
    )
    value.validate()

    event_time = np.asarray(events.event_time, dtype=np.float64)
    inside = np.zeros(len(event_time), dtype=bool)
    for left, right in zip(start, stop):
        inside |= (event_time >= left) & (event_time < right)
    if not bool(inside.all()):
        bad = event_time[~inside]
        raise ValueError(
            f"{subject}: {len(bad)} events fall outside metadata coverage; "
            f"first={bad[0]:.6f}"
        )

    old_path = contract.UPSTREAM_ROOT / "recorded_intervals" / f"{subject}.npz"
    parity = None
    if old_path.exists():
        with np.load(old_path, allow_pickle=False) as old:
            expected = old["recorded"].astype(np.float64)
        got = np.zeros(len(event_time), dtype=np.float64)
        if len(event_time) > 1:
            got[1:] = recorded_duration_between(
                start, stop, event_time[:-1], event_time[1:]
            )
        difference = np.abs(got - expected)
        parity = {
            "old_artifact": str(old_path),
            "max_abs_seconds": float(difference.max()),
            "median_abs_seconds": float(np.median(difference)),
            "n_over_1ms": int(np.sum(difference > 1e-3)),
        }
    manifest = {
        "contract": contract.REVISION,
        "coverage_revision": COVERAGE_REVISION,
        "subject": subject,
        "n_source_blocks": int(len(table.blocks)),
        "n_merged_segments": int(len(start)),
        "n_events": int(len(event_time)),
        "all_events_inside_coverage": True,
        "recorded_hours_before_dev_end": float(
            recorded_duration_between(
                start, stop, np.asarray([start.min()]), np.asarray([dev_end])
            )[0] / 3600.0
        ),
        "duration_parity": parity,
        "source_hashes": value.source_hashes,
        "sealed_opened": False,
    }
    return value, manifest


def write_coverage(subject: str, output: Path) -> dict:
    value, manifest = build_coverage(subject)
    value.save(output)
    manifest["output"] = str(output)
    contract.atomic_json(output.with_suffix(".manifest.json"), manifest)
    return manifest
