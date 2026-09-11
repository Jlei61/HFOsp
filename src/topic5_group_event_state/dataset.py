"""Consolidate block shards into one per-patient event sequence, and serve it.

Two responsibilities, deliberately separated:

``consolidate_subject``
    walks a patient's block shards in real time order and writes one memory-
    mappable array per field plus an ``index.json``.  This is where the event
    *clock* is assembled: contiguous coverage sessions, real inter-event
    intervals, background-anchor alignment, and the true ictal intervals that are
    excluded from the interictal stream.

``SubjectSequence``
    serves that consolidated stream to the trainer.  It never returns an event's
    own content for the step where that event is being predicted.

A sequence never crosses a coverage-session boundary.  Recording gaps are not
bridged and unobserved hours are not treated as event-free.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DATASET_FORMAT_VERSION = "group_event_state_dataset_v0_1_0"

# Chronological development split.  Test is the latest slice of the patient's
# own recording, so a state model is judged on time it has never seen.
SPLIT_FRACTIONS = (0.70, 0.10, 0.20)

# Summaries of the recent past that the memoryless baseline is given, so that
# "the state model beat the baseline" cannot mean "the baseline had no history".
HISTORY_LAGS = (1, 5, 20)


@dataclass(frozen=True)
class Session:
    session_id: int
    start_index: int
    stop_index: int

    @property
    def n_events(self) -> int:
        return self.stop_index - self.start_index


def _shard_order(cache_dir: Path) -> list[tuple[float, Path, dict]]:
    entries = []
    for manifest_path in sorted(Path(cache_dir).glob("*.manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        shard = manifest_path.with_name(f"{manifest['record_name']}.npz")
        if shard.exists():
            entries.append((float(manifest["block_start_epoch"]), shard, manifest))
    entries.sort(key=lambda item: (item[0], item[1].name))
    return entries


def _session_ids(
    manifests: Sequence[dict],
    session_rows: Sequence[Mapping[str, str]],
) -> dict[str, int]:
    """Map each record to the coverage session the source audit assigned it."""

    record_to_session: dict[str, int] = {}
    for row in session_rows:
        first, last = str(row["first_record"]), str(row["last_record"])
        session = int(row["session_index"])
        started = False
        for manifest in manifests:
            name = manifest["record_name"]
            if name == first:
                started = True
            if started:
                record_to_session[name] = session
            if started and name == last:
                break
    return record_to_session


def consolidate_subject(
    subject: str,
    cache_dir: Path,
    out_dir: Path,
    *,
    session_rows: Sequence[Mapping[str, str]],
    seizures: Sequence[Mapping[str, Any]],
    overwrite: bool = False,
) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.json"
    if index_path.exists() and not overwrite:
        return json.loads(index_path.read_text())

    entries = _shard_order(cache_dir)
    if not entries:
        raise FileNotFoundError(f"{subject}: no shards under {cache_dir}")
    manifests = [m for _t, _p, m in entries]
    head = manifests[0]
    for m in manifests[1:]:
        if m["n_contacts"] != head["n_contacts"] or m["stored_views"] != head["stored_views"]:
            raise ValueError(f"{subject}: shards disagree on contact universe / views")
        if m["n_context_samples"] != head["n_context_samples"]:
            raise ValueError(f"{subject}: shards disagree on context width")

    n_events = int(sum(m["n_events"] for m in manifests))
    n_contacts = int(head["n_contacts"])
    n_bands = len(head["bands"])
    n_pairs = len(head["cross_band_pairs"])
    views = list(head["stored_views"])
    n_ctx = int(head["n_context_samples"])
    n_env = int(head["envelope_bins"])
    n_bg_feat = len(head["background"]["feature_names"])
    n_band_feat = len(head["band_feature_names"])

    specs = {
        "waveform": ((n_events, n_contacts, len(views), n_ctx), np.float16),
        "band_envelope": ((n_events, n_contacts, n_bands, n_env), np.float16),
        "band_features": ((n_events, n_contacts, n_bands, n_band_feat), np.float32),
        "cross_band_lag": ((n_events, n_contacts, n_pairs), np.float32),
        "participation": ((n_events, n_contacts), np.bool_),
        "contact_ok": ((n_events, n_contacts), np.bool_),
        "relative_delay": ((n_events, n_contacts), np.float32),
        "tied_group_id": ((n_events, n_contacts), np.int16),
        "legacy_rank": ((n_events, n_contacts), np.int16),
        "background": ((n_events, n_contacts, n_bg_feat), np.float32),
    }
    arrays = {
        name: np.lib.format.open_memmap(
            out_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
        )
        for name, (shape, dtype) in specs.items()
    }

    t_abs = np.zeros(n_events, dtype=np.float64)
    has_waveform = np.zeros(n_events, dtype=bool)
    core_seconds = np.zeros(n_events, dtype=np.float32)
    session_of_event = np.zeros(n_events, dtype=np.int32)
    block_of_event = np.zeros(n_events, dtype=np.int32)
    row_of_event = np.zeros(n_events, dtype=np.int32)
    background_age = np.full(n_events, np.inf, dtype=np.float32)

    record_to_session = _session_ids(manifests, session_rows)
    cursor = 0
    for block_index, (_start, shard_path, manifest) in enumerate(entries):
        with np.load(shard_path) as z:
            n = int(manifest["n_events"])
            sl = slice(cursor, cursor + n)
            wave = np.stack([z[f"waveform_{v}"] for v in views], axis=2)
            arrays["waveform"][sl] = wave
            arrays["band_envelope"][sl] = z["band_envelope"]
            arrays["band_features"][sl] = z["band_features"]
            arrays["cross_band_lag"][sl] = z["cross_band_lag_s"]
            arrays["participation"][sl] = z["participation"]
            arrays["contact_ok"][sl] = z["contact_ok"]
            arrays["relative_delay"][sl] = z["relative_delay_s"]
            arrays["tied_group_id"][sl] = z["tied_group_id"]
            arrays["legacy_rank"][sl] = z["legacy_rank"]
            t_abs[sl] = z["event_abs_time"]
            has_waveform[sl] = z["has_waveform"]
            core_seconds[sl] = z["core_seconds_raw"]
            block_of_event[sl] = block_index
            row_of_event[sl] = np.arange(n)
            session_of_event[sl] = record_to_session.get(manifest["record_name"], -1)

            anchor_t = z["background_time_s"]
            anchor_f = z["background_features"]
            core_t = z["core_start_seconds"]
            if anchor_t.size:
                # last anchor that *finished* before this event core started
                anchor_end = anchor_t + float(manifest["background"]["window_seconds"])
                pos = np.searchsorted(anchor_end, core_t, side="right") - 1
                valid = pos >= 0
                arrays["background"][sl][valid] = anchor_f[pos[valid]]
                age = np.full(n, np.inf, dtype=np.float32)
                age[valid] = (core_t[valid] - anchor_end[pos[valid]]).astype(np.float32)
                background_age[sl] = age
        cursor += n
    for array in arrays.values():
        array.flush()

    order = np.argsort(t_abs, kind="stable")
    if not np.array_equal(order, np.arange(n_events)):
        raise ValueError(f"{subject}: shards did not arrive in time order")

    dt_prev = np.full(n_events, np.nan, dtype=np.float32)
    dt_prev[1:] = np.diff(t_abs)
    session_start = np.zeros(n_events, dtype=bool)
    session_start[0] = True
    session_start[1:] = session_of_event[1:] != session_of_event[:-1]
    dt_prev[session_start] = np.nan

    is_ictal = np.zeros(n_events, dtype=bool)
    time_to_next_seizure = np.full(n_events, np.inf, dtype=np.float64)
    time_since_prev_seizure = np.full(n_events, np.inf, dtype=np.float64)
    if seizures:
        onsets = np.array([float(s["onset_epoch"]) for s in seizures])
        offsets = np.array([float(s["offset_epoch"]) for s in seizures])
        core_end = t_abs + core_seconds
        for onset, offset in zip(onsets, offsets):
            is_ictal |= (core_end > onset) & (t_abs < offset)
        idx_next = np.searchsorted(np.sort(onsets), t_abs, side="left")
        sorted_on = np.sort(onsets)
        ok = idx_next < sorted_on.size
        time_to_next_seizure[ok] = sorted_on[idx_next[ok]] - t_abs[ok]
        sorted_off = np.sort(offsets)
        idx_prev = np.searchsorted(sorted_off, t_abs, side="right") - 1
        ok = idx_prev >= 0
        time_since_prev_seizure[ok] = t_abs[ok] - sorted_off[idx_prev[ok]]

    keep = ~is_ictal
    sessions: list[Session] = []
    for session_id in sorted(set(session_of_event.tolist())):
        member = np.flatnonzero((session_of_event == session_id) & keep)
        if member.size == 0:
            continue
        sessions.append(Session(int(session_id), int(member[0]), int(member[-1]) + 1))

    interictal_index = np.flatnonzero(keep)
    n_keep = interictal_index.size
    n_train = int(round(n_keep * SPLIT_FRACTIONS[0]))
    n_val = int(round(n_keep * SPLIT_FRACTIONS[1]))
    split_bounds = {
        "train": [0, n_train],
        "val": [n_train, n_train + n_val],
        "test": [n_train + n_val, n_keep],
    }

    scalars = {
        "t_abs": t_abs,
        "dt_prev": dt_prev,
        "session_of_event": session_of_event,
        "session_start": session_start,
        "block_of_event": block_of_event,
        "row_of_event": row_of_event,
        "has_waveform": has_waveform,
        "core_seconds": core_seconds,
        "background_age": background_age,
        "is_ictal": is_ictal,
        "time_to_next_seizure": time_to_next_seizure,
        "time_since_prev_seizure": time_since_prev_seizure,
        "interictal_index": interictal_index,
    }
    tmp = out_dir / "scalars.npz.tmp"
    with tmp.open("wb") as handle:
        np.savez(handle, **scalars)
    os.replace(tmp, out_dir / "scalars.npz")

    index = {
        "format": DATASET_FORMAT_VERSION,
        "subject": subject,
        "dataset": head["dataset"],
        "n_events": n_events,
        "n_events_interictal": int(n_keep),
        "n_events_ictal": int(is_ictal.sum()),
        "n_events_with_waveform": int(has_waveform.sum()),
        "n_contacts": n_contacts,
        "native_rate_hz": head["native_rate_hz"],
        "detector_reference": head["detector_reference"],
        "montage_provenance": head.get("montage_provenance", "unknown"),
        "bipolar_equals_detector": head["bipolar_equals_detector"],
        "views": views,
        "bands": head["bands"],
        "band_available": head["band_available"],
        "band_feature_names": head["band_feature_names"],
        "cross_band_pairs": head["cross_band_pairs"],
        "n_context_samples": n_ctx,
        "n_core_samples": head["n_core_samples"],
        "core_seconds_nominal": head["core_seconds_nominal"],
        "envelope_bins": n_env,
        "background_feature_names": head["background"]["feature_names"],
        "contacts": head["contacts"],
        "tie_tolerance_seconds": head["tie_tolerance_seconds"],
        "sessions": [
            {"session_id": s.session_id, "start_index": s.start_index, "stop_index": s.stop_index}
            for s in sessions
        ],
        "split_bounds_on_interictal_index": split_bounds,
        "split_fractions": list(SPLIT_FRACTIONS),
        "n_seizures": len(seizures),
        "seizures": [dict(s) for s in seizures],
        "n_blocks": len(entries),
        "source_shards": [str(p) for _t, p, _m in entries],
        "arrays": {name: {"file": f"{name}.npy", "shape": list(spec[0]), "dtype": np.dtype(spec[1]).name}
                   for name, spec in specs.items()},
    }
    tmp_index = out_dir / "index.json.tmp"
    tmp_index.write_text(json.dumps(index, indent=2, sort_keys=True, default=float))
    os.replace(tmp_index, index_path)
    return index


def recent_history_features(
    t_abs: np.ndarray,
    participation: np.ndarray,
    relative_delay: np.ndarray,
    order: np.ndarray,
    lags: Sequence[int] = HISTORY_LAGS,
) -> np.ndarray:
    """Fixed summaries of the k previous events, in the sequence's own order.

    The memoryless baseline reads only these: how often events had been arriving,
    how large and how spread out they were, and which contacts took part.  Every
    column is strictly a function of events **before** the one being predicted, so
    anything the state model gains over this cannot be "it remembered the last
    event" -- and cannot be "the baseline was handed the answer" either.
    """

    n = order.size
    n_contacts = participation.shape[1]
    blocks: list[np.ndarray] = []
    t = t_abs[order]
    size = participation[order].sum(1).astype(np.float32)
    span = np.nan_to_num(relative_delay[order]).max(1).astype(np.float32)
    part = participation[order].astype(np.float32)

    for lag in lags:
        # Shifted by one event on purpose.  ``t[i] - t[i-lag]`` contains the very
        # interval the timing head has to predict, so the baseline would be handed
        # its own target; at event i the baseline may only know intervals that
        # closed at or before event i-1.
        dt = np.full(n, np.nan, dtype=np.float32)
        if n > lag + 1:
            dt[lag + 1 :] = (t[lag:-1] - t[: -lag - 1]).astype(np.float32)
        blocks.append(np.log1p(np.clip(dt, 0.0, None))[:, None])
        for source in (size, span):
            roll = np.full(n, np.nan, dtype=np.float32)
            csum = np.concatenate([[0.0], np.cumsum(source)])
            roll[lag:] = ((csum[lag:-1] - csum[: -lag - 1]) / lag).astype(np.float32)
            blocks.append(roll[:, None])
        rate = np.full((n, n_contacts), np.nan, dtype=np.float32)
        csum = np.concatenate([np.zeros((1, n_contacts)), np.cumsum(part, axis=0)])
        rate[lag:] = ((csum[lag:-1] - csum[: -lag - 1]) / lag).astype(np.float32)
        blocks.append(rate)
    return np.concatenate(blocks, axis=1)


class SubjectSequence:
    """Memory-mapped access to one patient's consolidated interictal stream."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.index = json.loads((self.root / "index.json").read_text())
        self.scalars = dict(np.load(self.root / "scalars.npz"))
        self.arrays = {
            name: np.load(self.root / meta["file"], mmap_mode="r")
            for name, meta in self.index["arrays"].items()
        }
        self.order = self.scalars["interictal_index"]
        self.session_of_event = self.scalars["session_of_event"][self.order]
        t = self.scalars["t_abs"][self.order]
        dt = np.full(t.size, np.nan, dtype=np.float32)
        dt[1:] = np.diff(t)
        new_session = np.zeros(t.size, dtype=bool)
        new_session[0] = True
        new_session[1:] = self.session_of_event[1:] != self.session_of_event[:-1]
        dt[new_session] = np.nan
        self.t_abs = t
        self.dt_prev = dt
        self.new_session = new_session
        # `contact_ok` in a shard records whether that event's waveform window
        # resolved, so the ~1 event per block whose filter window fell off the
        # edge has every contact marked invalid.  Contact validity is a property
        # of the montage and is constant across the recording; leaving it
        # per-event produced fully-masked attention rows, which return NaN and
        # poison the whole state chain from step 0.
        self.contact_valid = np.asarray(self.arrays["contact_ok"]).any(axis=0)
        self.history = recent_history_features(
            self.scalars["t_abs"],
            np.asarray(self.arrays["participation"]),
            np.asarray(self.arrays["relative_delay"]),
            self.order,
        )
        self.history[~np.isfinite(self.history)] = 0.0

    def __len__(self) -> int:
        return int(self.order.size)

    def split_slice(self, name: str) -> tuple[int, int]:
        lo, hi = self.index["split_bounds_on_interictal_index"][name]
        return int(lo), int(hi)

    def streams(self, lo: int, hi: int, n_streams: int) -> list[tuple[int, int]]:
        """Split a range into contiguous segments that can be run in parallel.

        Each segment carries its own state chain, so segments are long (train
        events / n_streams) and truncated BPTT inside them is unchanged.  This is
        a training-throughput device only: evaluation always runs the single true
        chronological chain.
        """

        total = hi - lo
        if n_streams <= 1 or total < n_streams * 2:
            return [(lo, hi)]
        edges = np.linspace(lo, hi, n_streams + 1).astype(int)
        return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]

    def chunks(self, lo: int, hi: int, chunk: int) -> list[tuple[int, int, bool]]:
        """Contiguous training chunks that never straddle a session boundary."""

        out: list[tuple[int, int, bool]] = []
        i = lo
        while i < hi:
            j = min(i + chunk, hi)
            breaks = np.flatnonzero(self.new_session[i + 1 : j]) + i + 1
            if breaks.size:
                j = int(breaks[0])
            out.append((i, j, bool(self.new_session[i])))
            i = j
        return out

    def gather(self, lo: int, hi: int) -> dict[str, np.ndarray]:
        return self.gather_positions(np.arange(lo, hi))

    def gather_positions(self, pos: np.ndarray) -> dict[str, np.ndarray]:
        """Gather arbitrary positions of the interictal stream (stream batching)."""

        lo, hi = pos, None  # kept for the shared body below
        idx = self.order[pos]
        return {
            "t_abs": self.t_abs[pos],
            "dt_prev": self.dt_prev[pos],
            "new_session": self.new_session[pos],
            "history": self.history[pos],
            "waveform": np.asarray(self.arrays["waveform"][idx]),
            "band_envelope": np.asarray(self.arrays["band_envelope"][idx]),
            "band_features": np.asarray(self.arrays["band_features"][idx]),
            "cross_band_lag": np.asarray(self.arrays["cross_band_lag"][idx]),
            "participation": np.asarray(self.arrays["participation"][idx]),
            "contact_ok": np.broadcast_to(self.contact_valid, (idx.size, self.contact_valid.size)).copy(),
            "rel_delay": np.asarray(self.arrays["relative_delay"][idx]),
            "tied_group_id": np.asarray(self.arrays["tied_group_id"][idx]),
            "legacy_rank": np.asarray(self.arrays["legacy_rank"][idx]),
            "background": np.asarray(self.arrays["background"][idx]),
            "background_age": self.scalars["background_age"][idx],
            "has_waveform": self.scalars["has_waveform"][idx],
            "time_to_next_seizure": self.scalars["time_to_next_seizure"][idx],
        }
