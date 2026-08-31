"""Read-only audit of complete group-event artifacts against native recordings.

The audit answers, per candidate patient, the questions that decide whether a
patient can enter Group-Event State v0.1 at all:

* does every packed group event still point at a readable native sample range?
* do ``packedTimes`` (event clock) and ``lagPat*`` (event content) describe the
  same event list, from the *same* producer variant?
* which montage did the detector actually run on, and can it be rebuilt?
* where does continuous coverage actually break, so that state is never carried
  unconditionally across hours nobody recorded?
* which analysis bands the native sampling rate supports -- as *missing*, never
  as zero.

Nothing here trains, selects, or filters on any outcome.  Seizure intervals are
used only to mark the true ictal interval for exclusion and to expose
distance-to-seizure bookkeeping; no seizure label is emitted as a model input.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .raw_views import resolve_montage
from .contract import (
    LAGPAT_VARIANTS,
    LagPatVariant,
    SEAM_TOLERANCE_SECONDS,
    map_detector_channels,
    relative_participant_delay,
    supported_band_mask,
    tied_recruitment_groups,
)


YUQUAN_ARTIFACT_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ARTIFACT_ROOT = Path(
    "/mnt/epilepsia_data/interilca_inter_results/all_data_lns"
)

# Bytes hashed from each end of a multi-gigabyte native recording.  This is a
# fingerprint, deliberately *not* a whole-file digest: the cohort holds several
# terabytes of raw and a full pass would cost hours per audit.
RAW_FINGERPRINT_EDGE_BYTES = 1 << 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_fingerprint(path: Path) -> dict[str, Any]:
    """Cheap identity for a native recording that is too large to hash whole.

    Returns size, mtime and the digests of the leading and trailing 1 MiB.  The
    ``kind`` field records that this is an edge fingerprint so no reader can
    mistake it for a full-content hash.
    """

    path = Path(path)
    if not path.exists():
        return {"kind": "absent", "path": str(path)}
    stat = path.stat()
    size = int(stat.st_size)
    with path.open("rb") as handle:
        head = handle.read(RAW_FINGERPRINT_EDGE_BYTES)
        if size > RAW_FINGERPRINT_EDGE_BYTES:
            handle.seek(max(0, size - RAW_FINGERPRINT_EDGE_BYTES))
            tail = handle.read(RAW_FINGERPRINT_EDGE_BYTES)
        else:
            tail = b""
    return {
        "kind": f"edge_sha256_{RAW_FINGERPRINT_EDGE_BYTES}B",
        "path": str(path),
        "size_bytes": size,
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256_head": hashlib.sha256(head).hexdigest(),
        "sha256_tail": hashlib.sha256(tail).hexdigest() if tail else "",
    }


def read_inventory(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: Mapping[str, str], *keys: str) -> float:
    for key in keys:
        value = row.get(key, "")
        if value not in (None, ""):
            try:
                return float(value)
            except ValueError:
                continue
    return float("nan")


def _path(row: Mapping[str, str], *keys: str) -> str:
    for key in keys:
        value = str(row.get(key, "") or "")
        if value:
            return value
    return ""


def inventory_index(
    epilepsiae_inventory: Path,
    yuquan_inventory: Path,
) -> dict[tuple[str, str, str], dict[str, str]]:
    out: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in read_inventory(epilepsiae_inventory):
        key = ("epilepsiae", str(row["subject"]), str(row["block_stem"]))
        if key in out:
            raise ValueError(f"duplicate inventory key {key}")
        out[key] = row
    for row in read_inventory(yuquan_inventory):
        key = ("yuquan", str(row["subject"]), str(row["block_stem"]))
        if key in out:
            raise ValueError(f"duplicate inventory key {key}")
        out[key] = row
    return out


def seizure_index(
    epilepsiae_seizures: Path | None,
    yuquan_seizures: Path | None,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Per-(dataset, subject) ictal intervals in epoch seconds.

    Epilepsiae rows additionally carry ``pattern`` / ``classification`` so H2b
    can report sensitivity by seizure phenotype instead of pooling them.
    """

    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for dataset, path in (
        ("epilepsiae", epilepsiae_seizures),
        ("yuquan", yuquan_seizures),
    ):
        if path is None or not Path(path).exists():
            continue
        for row in read_inventory(Path(path)):
            onset = _float(row, "eeg_onset_epoch")
            offset = _float(row, "eeg_offset_epoch")
            if not (math.isfinite(onset) and math.isfinite(offset)):
                continue
            key = (dataset, str(row["subject"]))
            out.setdefault(key, []).append(
                {
                    "seizure_id": str(row.get("seizure_id", "")),
                    "onset_epoch": float(onset),
                    "offset_epoch": float(max(offset, onset)),
                    "pattern": str(row.get("pattern", "")),
                    "classification": str(row.get("classification", "")),
                    "vigilance": str(row.get("vigilance", "")),
                }
            )
    for key in out:
        out[key].sort(key=lambda item: item["onset_epoch"])
    return out


def subject_artifact_dir(subject: str) -> tuple[str, str, Path]:
    dataset, patient = str(subject).split("_", 1)
    if dataset == "yuquan":
        return dataset, patient, YUQUAN_ARTIFACT_ROOT / patient
    if dataset == "epilepsiae":
        return dataset, patient, EPILEPSIAE_ARTIFACT_ROOT / patient / "all_recs"
    raise ValueError(f"unsupported dataset in {subject}")


def _record_name(lag_path: Path) -> str:
    for variant in LAGPAT_VARIANTS:
        if lag_path.name.endswith(variant.lagpat_suffix):
            return lag_path.name[: -len(variant.lagpat_suffix)]
    raise ValueError(f"not a lagPat path: {lag_path}")


def lagpat_variant_of(lag_path: Path) -> LagPatVariant:
    for variant in LAGPAT_VARIANTS:
        if lag_path.name.endswith(variant.lagpat_suffix):
            return variant
    raise ValueError(f"not a lagPat path: {lag_path}")


def _packed_path(lag_path: Path, record_name: str) -> Path:
    """Pair packedTimes with the *same producer variant* as this lagPat file.

    The two legacy packers disagree on the event list (chengshuai FC10477Q:
    2965 old rows vs 2601 ``withFreqCent`` rows).  Choosing the packed file by
    existence rather than by variant silently pairs one packer's event clock
    with the other packer's event content.
    """

    return lag_path.with_name(f"{record_name}{lagpat_variant_of(lag_path).packed_suffix}")


def _gpu_path(
    dataset: str,
    artifact_dir: Path,
    record_name: str,
    inventory_row: Mapping[str, str],
) -> Path:
    explicit = _path(inventory_row, "raw_gpu_path")
    if explicit and Path(explicit).exists():
        return Path(explicit)
    return artifact_dir / f"{record_name}_gpu.npz"


def _raw_path(dataset: str, inventory_row: Mapping[str, str]) -> Path:
    if dataset == "yuquan":
        return Path(_path(inventory_row, "edf_path", "data_path"))
    return Path(_path(inventory_row, "data_path"))


def _head_path(dataset: str, inventory_row: Mapping[str, str]) -> Path | None:
    if dataset == "yuquan":
        return None
    value = _path(inventory_row, "head_path")
    return Path(value) if value else None


def _detector_reference(dataset: str, gpu_files: Sequence[str], gpu: Any) -> str:
    """Name the montage the detector actually ran on.

    Yuquan ``_gpu.npz`` stores ``reference_type='bipolar'`` plus the explicit
    ``bipolar_pairs``.  The Epilepsiae producer stores no tag; its detector path
    (``epilepsiae_detectHFOs.avg_rerefAndDrop_eeg``) subtracts the mean over the
    retained intracranial channels, i.e. a global common average reference.  The
    token records that this was read off the producer, not guessed.
    """

    if "reference_type" in gpu_files:
        value = np.asarray(gpu["reference_type"]).reshape(-1)
        if value.size:
            return str(value[0])
    if dataset == "epilepsiae":
        return "car_global_intracranial_from_producer"
    return "unknown"


@dataclass(frozen=True)
class BlockAudit:
    dataset: str
    subject: str
    record_name: str
    status: str
    fail_reasons: tuple[str, ...]
    lagpat_variant: str
    n_events: int
    n_contacts: int
    native_rate_hz: float
    block_start_epoch: float
    block_end_epoch: float
    block_duration_sec: float
    detector_reference: str
    montage_provenance: str
    n_detector_channels: int
    raw_path: str
    raw_exists: bool
    head_path: str | None
    head_exists: bool | None
    lagpat_path: str
    packed_path: str
    gpu_path: str
    packed_exists: bool
    gpu_exists: bool
    event_count_aligned: bool
    contact_mapping_exact: bool
    contact_mapping_failures: tuple[str, ...]
    participating_lag_finite_fraction: float
    nonparticipant_finite_lag_fraction: float
    event_core_seconds_median: float | None
    lag_span_median_ms: float | None
    lag_span_max_ms: float | None
    lag_span_exceeds_core_fraction: float
    n_participants_median: float | None
    tied_group_count_median: float | None
    lag_frequency_available: bool
    start_t_matches_inventory: bool
    packed_within_recording: bool
    waveform_pointer_reconstructable: bool
    first_event_epoch: float
    last_event_epoch: float
    band_support: Mapping[str, bool]
    source_hashes: Mapping[str, str]
    raw_fingerprint: Mapping[str, Any]
    event_start_seconds: np.ndarray = field(repr=False, default=None)  # type: ignore[assignment]
    event_end_seconds: np.ndarray = field(repr=False, default=None)  # type: ignore[assignment]

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row.pop("event_start_seconds", None)
        row.pop("event_end_seconds", None)
        row["fail_reasons"] = "|".join(self.fail_reasons)
        row["contact_mapping_failures"] = "|".join(self.contact_mapping_failures)
        row["band_support"] = json.dumps(dict(self.band_support), sort_keys=True)
        row["source_hashes"] = json.dumps(dict(self.source_hashes), sort_keys=True)
        row["raw_fingerprint"] = json.dumps(dict(self.raw_fingerprint), sort_keys=True)
        return row


def audit_block(
    subject: str,
    lag_path: Path,
    inventory: Mapping[tuple[str, str, str], Mapping[str, str]],
    *,
    hash_artifacts: bool = True,
) -> BlockAudit:
    dataset, patient, artifact_dir = subject_artifact_dir(subject)
    record_name = _record_name(lag_path)
    variant = lagpat_variant_of(lag_path)
    inv_key = (dataset, patient, record_name)
    row = inventory.get(inv_key)
    if row is None:
        raise KeyError(f"missing block inventory row {inv_key}")

    packed_path = _packed_path(lag_path, record_name)
    gpu_path = _gpu_path(dataset, artifact_dir, record_name, row)
    raw_path = _raw_path(dataset, row)
    head_path = _head_path(dataset, row)
    packed_exists = packed_path.exists()
    gpu_exists = gpu_path.exists()
    raw_exists = raw_path.exists()
    head_exists = None if head_path is None else head_path.exists()

    with np.load(lag_path, allow_pickle=True) as lag:
        required = {"lagPatRaw", "eventsBool", "chnNames", "start_t"}
        missing = sorted(required - set(lag.files))
        if missing:
            raise ValueError(f"{lag_path}: missing {missing}")
        lag_raw = np.asarray(lag["lagPatRaw"], dtype=np.float64)
        participation = np.asarray(lag["eventsBool"]).astype(bool)
        labels = [str(value) for value in lag["chnNames"]]
        start_t = float(np.asarray(lag["start_t"]).reshape(-1)[0])
        lag_frequency_available = "lagPatFreq" in lag.files
    if lag_raw.shape != participation.shape:
        raise ValueError(f"{lag_path}: lagPatRaw/eventsBool shape mismatch")
    n_contacts, n_events = lag_raw.shape
    if n_contacts != len(labels):
        raise ValueError(f"{lag_path}: contact label count mismatch")

    duration = _float(row, "duration_sec", "head_duration_sec")
    packed_count = -1
    packed_within_recording = False
    event_start = np.zeros(0, dtype=np.float64)
    event_end = np.zeros(0, dtype=np.float64)
    core_median = None
    if packed_exists:
        packed = np.asarray(np.load(packed_path), dtype=np.float64)
        if packed.ndim == 2 and packed.shape[1] >= 2:
            packed_count = int(packed.shape[0])
            event_start = packed[:, 0].copy()
            event_end = packed[:, 1].copy()
            core_median = float(np.median(event_end - event_start))
            packed_within_recording = bool(
                packed.size > 0
                and np.all(np.isfinite(packed[:, :2]))
                and float(np.min(event_start)) >= -1e-6
                and (
                    not math.isfinite(duration)
                    or float(np.max(event_end)) <= duration + 1e-3
                )
            )

    detector_labels: list[str] = []
    detector_reference = "missing"
    montage_provenance = "unresolved"
    failures: list[str] = list(labels)
    if gpu_exists or (dataset == "yuquan" and raw_exists):
        try:
            montage = resolve_montage(
                dataset, labels, gpu_path if gpu_exists else None, raw_path
            )
            detector_labels = list(montage.detector_labels)
            detector_reference = montage.reference
            montage_provenance = montage.provenance
            if montage.provenance == "gpu_npz":
                _mapped, failures = map_detector_channels(dataset, labels, detector_labels)
            else:
                failures = list(montage.unresolvable)
        except Exception as exc:  # keep the reason, never a silent pass
            montage_provenance = f"error:{type(exc).__name__}"

    valid = participation & np.isfinite(lag_raw)
    n_participating = int(np.sum(participation))
    finite_fraction = (
        float(np.sum(valid) / n_participating) if n_participating else float("nan")
    )
    n_nonpart = int(np.sum(~participation))
    nonpart_finite = (
        float(np.sum((~participation) & np.isfinite(lag_raw)) / n_nonpart)
        if n_nonpart
        else 0.0
    )

    rel = relative_participant_delay(lag_raw.T, participation.T).T
    with np.errstate(all="ignore"):
        spans = np.nanmax(np.where(participation, rel, np.nan), axis=0)
    finite_spans = spans[np.isfinite(spans)]
    lag_span_ms = float(np.median(finite_spans) * 1e3) if finite_spans.size else None
    lag_span_max_ms = float(np.max(finite_spans) * 1e3) if finite_spans.size else None
    if finite_spans.size and core_median is not None:
        span_exceeds = float(np.mean(finite_spans > core_median + 1e-9))
    else:
        span_exceeds = float("nan")
    n_part_per_event = participation.sum(axis=0)
    n_part_median = float(np.median(n_part_per_event)) if n_events else None
    tie_counts = [
        len(tied_recruitment_groups(rel[:, ei], participation[:, ei]))
        for ei in range(0, n_events, max(1, n_events // 256))
    ]
    tie_median = float(np.median(tie_counts)) if tie_counts else None

    native_rate = _float(row, "sample_rate", "head_sample_rate", "sample_rate_sql")
    block_start = _float(row, "block_start_epoch", "head_start_epoch")
    block_end = _float(row, "block_end_epoch")
    event_count_aligned = bool(packed_count == n_events)
    contact_mapping_exact = bool(not failures and detector_labels)
    start_aligned = bool(math.isfinite(start_t) and abs(start_t - block_start) <= 1e-3)

    reasons: list[str] = []
    if not raw_exists:
        reasons.append("raw_missing")
    if head_exists is False:
        reasons.append("head_missing")
    if not packed_exists:
        reasons.append("packed_missing")
    if not gpu_exists and montage_provenance != "derived_from_label_rule_no_gpu":
        reasons.append("gpu_missing")
    if not event_count_aligned:
        reasons.append(f"event_count_mismatch({packed_count}!={n_events})")
    if not contact_mapping_exact:
        reasons.append("contact_mapping_failed")
    if not packed_within_recording:
        reasons.append("packed_outside_recording")
    if not start_aligned:
        reasons.append("start_t_inventory_mismatch")
    if not math.isfinite(native_rate):
        reasons.append("native_rate_missing")
    if math.isfinite(finite_fraction) and finite_fraction < 1.0:
        reasons.append("participant_lag_not_finite")
    if math.isfinite(span_exceeds) and span_exceeds > 0.0:
        reasons.append("delay_span_exceeds_event_core")

    reconstructable = not [
        r
        for r in reasons
        if r
        not in {"participant_lag_not_finite", "delay_span_exceeds_event_core"}
    ]
    status = "PASS" if not reasons else "FAIL"

    source_hashes: dict[str, str] = {}
    if hash_artifacts:
        source_hashes["lagpat"] = sha256_file(lag_path)
        if packed_exists:
            source_hashes["packed"] = sha256_file(packed_path)
        if gpu_exists:
            source_hashes["gpu"] = sha256_file(gpu_path)
        if head_path is not None and head_exists:
            source_hashes["head"] = sha256_file(head_path)

    first_epoch = float(block_start + event_start[0]) if event_start.size else float("nan")
    last_epoch = float(block_start + event_start[-1]) if event_start.size else float("nan")

    return BlockAudit(
        dataset=dataset,
        subject=subject,
        record_name=record_name,
        status=status,
        fail_reasons=tuple(reasons),
        lagpat_variant=variant.name,
        n_events=n_events,
        n_contacts=n_contacts,
        native_rate_hz=native_rate,
        block_start_epoch=block_start,
        block_end_epoch=block_end,
        block_duration_sec=duration,
        detector_reference=detector_reference,
        montage_provenance=montage_provenance,
        n_detector_channels=len(detector_labels),
        raw_path=str(raw_path),
        raw_exists=raw_exists,
        head_path=None if head_path is None else str(head_path),
        head_exists=head_exists,
        lagpat_path=str(lag_path),
        packed_path=str(packed_path),
        gpu_path=str(gpu_path),
        packed_exists=packed_exists,
        gpu_exists=gpu_exists,
        event_count_aligned=event_count_aligned,
        contact_mapping_exact=contact_mapping_exact,
        contact_mapping_failures=tuple(failures),
        participating_lag_finite_fraction=finite_fraction,
        nonparticipant_finite_lag_fraction=nonpart_finite,
        event_core_seconds_median=core_median,
        lag_span_median_ms=lag_span_ms,
        lag_span_max_ms=lag_span_max_ms,
        lag_span_exceeds_core_fraction=span_exceeds,
        n_participants_median=n_part_median,
        tied_group_count_median=tie_median,
        lag_frequency_available=lag_frequency_available,
        start_t_matches_inventory=start_aligned,
        packed_within_recording=packed_within_recording,
        waveform_pointer_reconstructable=reconstructable,
        first_event_epoch=first_epoch,
        last_event_epoch=last_epoch,
        band_support=supported_band_mask(native_rate),
        source_hashes=source_hashes,
        raw_fingerprint=raw_fingerprint(raw_path),
        event_start_seconds=event_start,
        event_end_seconds=event_end,
    )


def lagpat_paths(subject: str) -> list[Path]:
    """All lagPat artifacts for one subject, from a single producer variant.

    Mixing variants inside one subject would mix two different event lists, so
    the richer ``withFreqCent`` variant wins whenever it exists and the older
    variant is used only when it is the only one present.
    """

    _dataset, _patient, root = subject_artifact_dir(subject)
    for variant in LAGPAT_VARIANTS:
        paths = sorted(root.glob(f"*{variant.lagpat_suffix}"))
        if paths:
            return paths
    return []


def coverage_sessions(
    blocks: Sequence[BlockAudit],
    all_blocks_in_inventory: Sequence[Mapping[str, str]],
    *,
    seam_tolerance_sec: float = SEAM_TOLERANCE_SECONDS,
) -> list[dict[str, Any]]:
    """Split a subject's packed blocks into genuinely continuous sessions.

    A session breaks when either (a) the clock gap to the next packed block
    exceeds the seam tolerance, or (b) the recording continued in a block that
    has no group-event artifact.  Case (b) matters because such an hour is
    *unobserved*, not event-free; carrying state across it would invent history.
    """

    ordered = sorted(blocks, key=lambda b: (b.block_start_epoch, b.record_name))
    packed_stems = {b.record_name for b in ordered}
    recorded = sorted(
        (
            (float(r["block_start_epoch"]), float(r["block_end_epoch"]), str(r["block_stem"]))
            for r in all_blocks_in_inventory
            if r.get("block_start_epoch") not in (None, "")
            and r.get("block_end_epoch") not in (None, "")
        ),
    )
    sessions: list[dict[str, Any]] = []
    current: list[BlockAudit] = []

    def _flush() -> None:
        if not current:
            return
        sessions.append(
            {
                "session_index": len(sessions),
                "n_blocks": len(current),
                "n_events": int(sum(b.n_events for b in current)),
                "start_epoch": float(current[0].block_start_epoch),
                "end_epoch": float(current[-1].block_end_epoch),
                "duration_hours": float(
                    (current[-1].block_end_epoch - current[0].block_start_epoch) / 3600.0
                ),
                "record_names": [b.record_name for b in current],
            }
        )

    for block in ordered:
        if current:
            prev = current[-1]
            gap = block.block_start_epoch - prev.block_end_epoch
            unobserved_between = any(
                start >= prev.block_end_epoch - seam_tolerance_sec
                and end <= block.block_start_epoch + seam_tolerance_sec
                and stem not in packed_stems
                for start, end, stem in recorded
            )
            if gap > seam_tolerance_sec or unobserved_between:
                _flush()
                current = []
        current.append(block)
    _flush()
    return sessions


def _event_table(blocks: Sequence[BlockAudit]) -> dict[str, np.ndarray]:
    """Absolute event clock for a subject, ordered in real time."""

    starts, ends, block_idx, rows = [], [], [], []
    for bi, block in enumerate(
        sorted(blocks, key=lambda b: (b.block_start_epoch, b.record_name))
    ):
        if block.event_start_seconds is None or not block.event_start_seconds.size:
            continue
        starts.append(block.block_start_epoch + block.event_start_seconds)
        ends.append(block.block_start_epoch + block.event_end_seconds)
        block_idx.append(np.full(block.event_start_seconds.size, bi, dtype=np.int64))
        rows.append(np.arange(block.event_start_seconds.size, dtype=np.int64))
    if not starts:
        return {k: np.zeros(0) for k in ("start", "end", "block", "row")}
    start = np.concatenate(starts)
    order = np.argsort(start, kind="stable")
    return {
        "start": start[order],
        "end": np.concatenate(ends)[order],
        "block": np.concatenate(block_idx)[order],
        "row": np.concatenate(rows)[order],
    }


def ictal_masks(
    event_start: np.ndarray,
    event_end: np.ndarray,
    seizures: Sequence[Mapping[str, Any]],
    *,
    post_guard_sec: float = 0.0,
    pre_guard_sec: float = 0.0,
) -> dict[str, np.ndarray]:
    """Mark events overlapping a true ictal interval; keep preictal events.

    Also returns signed distances to the nearest seizure so downstream analyses
    can stratify by lead time without any of that entering representation
    training.
    """

    n = int(event_start.size)
    is_ictal = np.zeros(n, dtype=bool)
    time_to_next = np.full(n, np.inf)
    time_since_prev = np.full(n, np.inf)
    if n and seizures:
        onsets = np.array([s["onset_epoch"] for s in seizures], dtype=np.float64)
        offsets = np.array([s["offset_epoch"] for s in seizures], dtype=np.float64)
        for onset, offset in zip(onsets, offsets):
            is_ictal |= (event_end > onset - pre_guard_sec) & (
                event_start < offset + post_guard_sec
            )
        for i in range(n):
            future = onsets[onsets >= event_start[i]]
            past = offsets[offsets <= event_start[i]]
            if future.size:
                time_to_next[i] = float(future[0] - event_start[i])
            if past.size:
                time_since_prev[i] = float(event_start[i] - past[-1])
    return {
        "is_ictal": is_ictal,
        "time_to_next_seizure_sec": time_to_next,
        "time_since_prev_seizure_sec": time_since_prev,
    }


def audit_subject(
    subject: str,
    inventory: Mapping[tuple[str, str, str], Mapping[str, str]],
    seizures: Mapping[tuple[str, str], list[dict[str, Any]]] | None = None,
    *,
    hash_artifacts: bool = True,
    pointer_sample_size: int = 32,
) -> dict[str, Any]:
    paths = lagpat_paths(subject)
    if not paths:
        raise FileNotFoundError(f"{subject}: no lagPat artifacts")
    blocks = [
        audit_block(subject, path, inventory, hash_artifacts=hash_artifacts)
        for path in paths
    ]
    dataset, patient, _root = subject_artifact_dir(subject)
    inv_rows = [
        row for (ds, sub, _stem), row in inventory.items() if ds == dataset and sub == patient
    ]
    sessions = coverage_sessions(blocks, inv_rows)
    table = _event_table(blocks)
    sz = list((seizures or {}).get((dataset, patient), []))
    masks = ictal_masks(table["start"], table["end"], sz)

    event_total = int(sum(b.n_events for b in blocks))
    ok_blocks = [b for b in blocks if b.waveform_pointer_reconstructable]
    reconstructable = int(sum(b.n_events for b in ok_blocks))
    n_ictal = int(masks["is_ictal"].sum())
    band_events = {
        name: int(sum(b.n_events for b in blocks if b.band_support[name]))
        for name in blocks[0].band_support
    }
    session_sizes = [s["n_events"] for s in sessions] or [0]
    rates = sorted({round(float(b.native_rate_hz), 3) for b in blocks if math.isfinite(b.native_rate_hz)})
    contacts = sorted({int(b.n_contacts) for b in blocks})
    dt = np.diff(table["start"]) if table["start"].size > 1 else np.zeros(0)

    rng = np.random.default_rng(abs(hash(subject)) % (2**32))
    sample_idx = (
        rng.choice(table["start"].size, size=min(pointer_sample_size, table["start"].size), replace=False)
        if table["start"].size
        else np.zeros(0, dtype=int)
    )
    ordered_blocks = sorted(blocks, key=lambda b: (b.block_start_epoch, b.record_name))
    pointer_samples = []
    for i in np.sort(sample_idx):
        blk = ordered_blocks[int(table["block"][i])]
        pointer_samples.append(
            {
                "event_abs_time": float(table["start"][i]),
                "record_name": blk.record_name,
                "source_event_row": int(table["row"][i]),
                "core_start_sample": int(round((table["start"][i] - blk.block_start_epoch) * blk.native_rate_hz)),
                "core_stop_sample": int(round((table["end"][i] - blk.block_start_epoch) * blk.native_rate_hz)),
                "native_rate_hz": float(blk.native_rate_hz),
                "raw_path": blk.raw_path,
                "detector_reference": blk.detector_reference,
            }
        )

    return {
        "subject": subject,
        "dataset": dataset,
        "patient": patient,
        "lagpat_variant": blocks[0].lagpat_variant,
        "n_blocks": len(blocks),
        "n_blocks_pass": int(sum(b.status == "PASS" for b in blocks)),
        "n_blocks_recorded_in_inventory": len(inv_rows),
        "block_fail_reasons": sorted({r for b in blocks for r in b.fail_reasons}),
        "n_events": event_total,
        "n_waveform_pointer_reconstructable": reconstructable,
        "waveform_pointer_fraction": float(reconstructable / event_total) if event_total else float("nan"),
        "n_contacts": contacts,
        "native_rate_hz": rates,
        "detector_reference": sorted({b.detector_reference for b in blocks}),
        "montage_provenance": sorted({b.montage_provenance for b in blocks}),
        "n_detector_channels": sorted({b.n_detector_channels for b in blocks}),
        "lag_frequency_available": bool(all(b.lag_frequency_available for b in blocks)),
        "n_contiguous_sessions": len(sessions),
        "max_events_in_contiguous_session": int(max(session_sizes)),
        "median_events_in_contiguous_session": float(np.median(session_sizes)),
        "longest_session_hours": float(max((s["duration_hours"] for s in sessions), default=0.0)),
        "sessions_ge_1k_events": int(sum(1 for s in sessions if s["n_events"] >= 1000)),
        "sessions_ge_5k_events": int(sum(1 for s in sessions if s["n_events"] >= 5000)),
        "sessions_ge_10k_events": int(sum(1 for s in sessions if s["n_events"] >= 10000)),
        "n_seizures": len(sz),
        "seizure_patterns": sorted({s["pattern"] for s in sz if s["pattern"]}),
        "n_events_ictal": n_ictal,
        "n_events_interictal": event_total - n_ictal,
        "n_events_preictal_1h": int(
            np.sum((~masks["is_ictal"]) & (masks["time_to_next_seizure_sec"] <= 3600.0))
        ),
        "median_inter_event_sec": float(np.median(dt)) if dt.size else float("nan"),
        "p95_inter_event_sec": float(np.percentile(dt, 95)) if dt.size else float("nan"),
        "recording_span_hours": float(
            (table["start"][-1] - table["start"][0]) / 3600.0 if table["start"].size > 1 else 0.0
        ),
        "events_by_supported_band": band_events,
        "sessions": sessions,
        "pointer_samples": pointer_samples,
        "blocks": [b.to_row() for b in blocks],
    }


def audit_cohort(
    subjects: Sequence[str],
    inventory: Mapping[tuple[str, str, str], Mapping[str, str]],
    seizures: Mapping[tuple[str, str], list[dict[str, Any]]] | None = None,
    *,
    hash_artifacts: bool = True,
    progress: bool = False,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for subject in subjects:
        try:
            records.append(
                audit_subject(subject, inventory, seizures, hash_artifacts=hash_artifacts)
            )
            if progress:
                print(f"  audited {subject}: {records[-1]['n_events']} events", flush=True)
        except Exception as exc:  # a partial audit file is worse than a flagged failure
            failures.append({"subject": subject, "error": f"{type(exc).__name__}: {exc}"})
            if progress:
                print(f"  FAILED {subject}: {type(exc).__name__}: {exc}", flush=True)
    total = int(sum(r["n_events"] for r in records))
    reconstructable = int(sum(r["n_waveform_pointer_reconstructable"] for r in records))
    band_names = sorted({n for r in records for n in r["events_by_supported_band"]})
    return {
        "contract": "topic5_group_event_state_v0_1_source_audit",
        "status": "PASS" if not failures else "PARTIAL",
        "n_subjects_requested": len(subjects),
        "n_subjects_audited": len(records),
        "n_subject_failures": len(failures),
        "n_events": total,
        "n_events_interictal": int(sum(r["n_events_interictal"] for r in records)),
        "n_waveform_pointer_reconstructable": reconstructable,
        "waveform_pointer_fraction": float(reconstructable / total) if total else float("nan"),
        "n_subjects_with_10k_session": int(sum(1 for r in records if r["sessions_ge_10k_events"])),
        "n_subjects_with_5k_session": int(sum(1 for r in records if r["sessions_ge_5k_events"])),
        "n_subjects_with_1k_session": int(sum(1 for r in records if r["sessions_ge_1k_events"])),
        "events_by_supported_band": {
            name: int(sum(r["events_by_supported_band"].get(name, 0) for r in records))
            for name in band_names
        },
        "subjects": records,
        "failures": failures,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"unserialisable {type(value)!r}")


def write_json_atomic(payload: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default, allow_nan=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def write_csv_atomic(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("", encoding="utf-8")
        os.replace(tmp, path)
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    os.replace(tmp, path)


def discover_subjects(
    *,
    epilepsiae_root: Path = EPILEPSIAE_ARTIFACT_ROOT,
    yuquan_root: Path = YUQUAN_ARTIFACT_ROOT,
) -> list[str]:
    """Every patient that owns at least one group-event artifact on disk."""

    found: list[str] = []
    for dataset, root in (("epilepsiae", epilepsiae_root), ("yuquan", yuquan_root)):
        if not Path(root).exists():
            continue
        for directory in sorted(Path(root).iterdir()):
            if not directory.is_dir():
                continue
            base = directory / "all_recs" if dataset == "epilepsiae" else directory
            if not base.is_dir():
                continue
            if any(base.glob(f"*{v.lagpat_suffix}") for v in LAGPAT_VARIANTS):
                found.append(f"{dataset}_{directory.name}")
    return sorted(found)


def discover_existing_34(dataset_root: Path) -> list[str]:
    per_subject = Path(dataset_root) / "per_subject"
    return sorted(path.stem for path in per_subject.glob("*.npz"))
