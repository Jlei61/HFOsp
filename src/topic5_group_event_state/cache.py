"""Block-sharded multimodal cache for Group-Event State v0.1.

One shard holds every complete group event of one recording block: the native
waveform in each reference view, per-band energy trajectories with an explicit
availability mask, exact continuous recruitment delays, tied recruitment groups,
and the background-SEEG anchors that sit *before* each event core.

The cache is a derived convenience.  The lossless source of every event stays
the pointer written into the shard manifest (file, sample range, montage,
fingerprint), so a shard can always be rebuilt or checked against raw.

Nothing in a shard depends on any seizure label, cluster label or outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.signal import firwin, filtfilt, hilbert, resample_poly

from .contract import (
    ANALYSIS_BANDS_HZ,
    EVENT_CONTEXT_POST_SECONDS,
    EVENT_CONTEXT_PRE_SECONDS,
    TIE_TOLERANCE_SECONDS,
    relative_participant_delay,
    supported_band_mask,
    tied_recruitment_groups,
)
from .raw_views import (
    VIEW_BIPOLAR,
    VIEW_DETECTOR,
    VIEW_SHAFT_CAR,
    ContactUniverse,
    build_contact_universe,
    build_event_views,
    build_view_plan,
    clean_contact,
    open_block_reader,
    resolve_montage,
)

CACHE_FORMAT_VERSION = "group_event_state_cache_v0_1_0"

# Filtering shoulder discarded after bandpass+Hilbert, so no stored sample is
# contaminated by a filter transient.
FILTER_PAD_SECONDS = 0.5
FIR_NUMTAPS = 201

# Envelopes are stored on a fixed model grid so one encoder can read both
# sampling rates without resampling the waveform itself.
ENVELOPE_BINS = 192

# Background SEEG is an auxiliary observation on a fixed rule: a grid of short
# windows that never overlap a packed event core.
BACKGROUND_GRID_SECONDS = 30.0
BACKGROUND_WINDOW_SECONDS = 2.0

BAND_FEATURE_NAMES = (
    "peak_time_s",
    "centroid_time_s",
    "log_integrated_energy",
    "width_s",
    "log_peak_amplitude",
)
BACKGROUND_FEATURE_SUFFIXES = ("log_var", "lag1_ac", "spectral_edge_hz")


def _band_pairs(band_names: Sequence[str]) -> list[tuple[int, int]]:
    return [
        (i, j)
        for i in range(len(band_names))
        for j in range(i + 1, len(band_names))
    ]


def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    taps = firwin(FIR_NUMTAPS, [lo / (fs / 2.0), hi / (fs / 2.0)], pass_zero=False)
    return filtfilt(taps, [1.0], x, axis=-1)


def _envelope(x: np.ndarray) -> np.ndarray:
    return np.abs(hilbert(x, axis=-1))


def _resample_to_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Area-preserving resample of the last axis onto a fixed bin count."""

    n = x.shape[-1]
    if n == n_bins:
        return x
    edges = np.linspace(0, n, n_bins + 1)
    cum = np.concatenate(
        [np.zeros(x.shape[:-1] + (1,), dtype=np.float64), np.cumsum(x, axis=-1)], axis=-1
    )
    lo = np.floor(edges[:-1]).astype(int)
    hi = np.clip(np.ceil(edges[1:]).astype(int), 1, n)
    out = (np.take(cum, hi, axis=-1) - np.take(cum, lo, axis=-1)) / np.maximum(
        (hi - lo), 1
    )
    return out


def _xcorr_peak_lag(a: np.ndarray, b: np.ndarray, fs: float, max_lag_s: float) -> np.ndarray:
    """Lag (seconds) of ``b`` relative to ``a``, positive when ``b`` follows.

    Vectorised over every leading axis; the search is clipped to ``max_lag_s``
    so a spurious far-field peak cannot masquerade as a within-event delay.
    """

    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    n = a.shape[-1]
    nfft = int(2 ** math.ceil(math.log2(2 * n - 1)))
    fa = np.fft.rfft(a, n=nfft, axis=-1)
    fb = np.fft.rfft(b, n=nfft, axis=-1)
    cc = np.fft.irfft(fb * np.conj(fa), n=nfft, axis=-1)
    cc = np.concatenate([cc[..., -(n - 1):], cc[..., :n]], axis=-1)
    lags = np.arange(-(n - 1), n) / float(fs)
    keep = np.abs(lags) <= float(max_lag_s)
    cc = cc[..., keep]
    lags = lags[keep]
    return lags[np.argmax(cc, axis=-1)]


@dataclass(frozen=True)
class BlockSpec:
    dataset: str
    subject: str
    record_name: str
    raw_path: str
    head_path: str | None
    gpu_path: str
    lagpat_path: str
    packed_path: str
    block_start_epoch: float
    native_rate_hz: float


def load_universe(spec: BlockSpec) -> tuple[ContactUniverse, str, list[str]]:
    with np.load(spec.lagpat_path, allow_pickle=True) as lag:
        labels = [str(v) for v in lag["chnNames"]]
    montage = resolve_montage(
        spec.dataset, labels, Path(spec.gpu_path), Path(spec.raw_path)
    )
    if montage.unresolvable:
        raise ValueError(
            f"{spec.record_name}: unresolvable montage rows {montage.unresolvable}"
        )
    universe = build_contact_universe(
        spec.dataset,
        spec.subject,
        labels,
        montage.detector_labels,
        montage.reference,
        montage.bipolar_pairs,
    )
    return universe, montage.provenance, list(montage.detector_labels)


def _stored_views(universe: ContactUniverse) -> tuple[str, ...]:
    """Only reference views that differ are written; the alias is in the manifest."""

    if universe.bipolar_equals_detector:
        return (VIEW_DETECTOR, VIEW_SHAFT_CAR)
    return (VIEW_DETECTOR, VIEW_BIPOLAR, VIEW_SHAFT_CAR)


def build_block_shard(
    spec: BlockSpec,
    out_dir: Path,
    *,
    chunk_events: int = 128,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Materialise one block's events into ``<out_dir>/<record>.npz`` atomically."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec.record_name}.npz"
    manifest_path = out_dir / f"{spec.record_name}.manifest.json"
    if out_path.exists() and manifest_path.exists() and not overwrite:
        return json.loads(manifest_path.read_text())

    started = time.time()
    reader = open_block_reader(
        spec.dataset, Path(spec.raw_path), Path(spec.head_path) if spec.head_path else None
    )
    universe, montage_provenance, detector_labels = load_universe(spec)
    car_idx = (
        [reader.index[clean_contact(d)] for d in detector_labels]
        if spec.dataset == "epilepsiae"
        else None
    )
    fs = float(reader.native_rate_hz)
    if abs(fs - spec.native_rate_hz) > 1e-6:
        raise ValueError(
            f"{spec.record_name}: reader rate {fs} != inventory rate {spec.native_rate_hz}"
        )

    with np.load(spec.lagpat_path, allow_pickle=True) as lag:
        participation = np.asarray(lag["eventsBool"]).astype(bool).T  # (events, contacts)
        lag_raw = np.asarray(lag["lagPatRaw"], dtype=np.float64).T
        legacy_rank = np.asarray(lag["lagPatRank"]).astype(np.int16).T
        legacy_freq = (
            np.asarray(lag["lagPatFreq"], dtype=np.float32).T
            if "lagPatFreq" in lag.files
            else None
        )
    packed = np.asarray(np.load(spec.packed_path), dtype=np.float64)
    n_events, n_contacts = participation.shape
    if packed.shape[0] != n_events:
        raise ValueError(
            f"{spec.record_name}: packed rows {packed.shape[0]} != lagPat events {n_events}"
        )

    band_names = tuple(ANALYSIS_BANDS_HZ)
    support = supported_band_mask(fs)
    band_available = np.array([support[b] for b in band_names], dtype=bool)
    pairs = _band_pairs(band_names)

    rel_delay = relative_participant_delay(lag_raw, participation).astype(np.float32)
    tie_id = np.full((n_events, n_contacts), -1, dtype=np.int16)
    for ei in range(n_events):
        for gi, group in enumerate(
            tied_recruitment_groups(rel_delay[ei], participation[ei])
        ):
            tie_id[ei, group] = gi

    # Sample widths come from ONE nominal core duration per block, not from each
    # event independently: rounding t0*fs and t1*fs separately makes a constant
    # 0.11 s core alternate between 112 and 113 samples purely on the fractional
    # phase of t0, which discarded 37% of a block for a sub-millisecond artefact
    # three orders of magnitude below the 10 ms centroid resolution.
    core_seconds_raw = packed[:, 1] - packed[:, 0]
    n_core = int(round(float(np.median(core_seconds_raw)) * fs))
    core_offset = int(round(EVENT_CONTEXT_PRE_SECONDS * fs))
    n_ctx = core_offset + n_core + int(round(EVENT_CONTEXT_POST_SECONDS * fs))
    core_start = np.rint(packed[:, 0] * fs).astype(np.int64)
    core_stop = core_start + n_core
    ctx_start = core_start - core_offset
    ctx_stop = ctx_start + n_ctx
    pad = int(round(FILTER_PAD_SECONDS * fs))

    # Every event stays on the clock.  Only the *waveform* of an event whose
    # filter window runs off the block edge is unavailable; dropping the event
    # itself would silently shorten the inter-event intervals the state model
    # integrates over.
    has_waveform = (ctx_start - pad >= 0) & (ctx_stop + pad <= reader.n_samples)
    plan = build_view_plan(reader, universe, car_idx)

    views = _stored_views(universe)
    waveform = {v: np.zeros((n_events, n_contacts, n_ctx), dtype=np.float16) for v in views}
    envelopes = np.zeros((n_events, n_contacts, len(band_names), ENVELOPE_BINS), dtype=np.float16)
    band_features = np.full(
        (n_events, n_contacts, len(band_names), len(BAND_FEATURE_NAMES)), np.nan, np.float32
    )
    cross_band_lag = np.full((n_events, n_contacts, len(pairs)), np.nan, np.float32)
    contact_ok = np.zeros((n_events, n_contacts), dtype=bool)

    max_lag_s = float(n_core) / fs

    for lo in range(0, n_events, chunk_events):
        hi = min(lo + chunk_events, n_events)
        idx = [ei for ei in range(lo, hi) if has_waveform[ei]]
        if not idx:
            continue
        padded = np.empty((len(idx), n_contacts, n_ctx + 2 * pad), dtype=np.float32)
        raw_views: dict[str, np.ndarray] = {
            v: np.empty((len(idx), n_contacts, n_ctx), dtype=np.float32) for v in views
        }
        for k, ei in enumerate(idx):
            built = build_event_views(
                reader,
                universe,
                int(ctx_start[ei] - pad),
                int(ctx_stop[ei] + pad),
                plan=plan,
            )
            padded[k] = built[VIEW_DETECTOR]
            for v in views:
                raw_views[v][k] = built[v][:, pad : pad + n_ctx]
        for v in views:
            waveform[v][idx] = raw_views[v].astype(np.float16)
        contact_ok[idx] = ~np.isnan(padded[:, :, 0])

        flat = np.nan_to_num(padded.reshape(-1, padded.shape[-1]), copy=False)
        env_by_band = []
        for bi, band in enumerate(band_names):
            if not band_available[bi]:
                env_by_band.append(None)
                continue
            lo_hz, hi_hz = ANALYSIS_BANDS_HZ[band]
            env = _envelope(_bandpass(flat, fs, lo_hz, hi_hz))[:, pad : pad + n_ctx]
            env = env.astype(np.float32).reshape(len(idx), n_contacts, n_ctx)
            env_by_band.append(env)
            envelopes[idx, :, bi, :] = _resample_to_bins(env, ENVELOPE_BINS).astype(np.float16)

            core = env[:, :, core_offset : core_offset + n_core]
            t = np.arange(n_core) / fs
            peak_bin = np.argmax(core, axis=-1)
            band_features[idx, :, bi, 0] = peak_bin / fs
            power = core.astype(np.float64) ** 2
            denom = power.sum(-1) + 1e-30
            band_features[idx, :, bi, 1] = (power * t).sum(-1) / denom
            band_features[idx, :, bi, 2] = np.log(denom / fs)
            half = core.max(-1, keepdims=True) * 0.5
            band_features[idx, :, bi, 3] = (core >= half).sum(-1) / fs
            band_features[idx, :, bi, 4] = np.log(core.max(-1) + 1e-30)

        for pi, (a, b) in enumerate(pairs):
            if env_by_band[a] is None or env_by_band[b] is None:
                continue
            cross_band_lag[idx, :, pi] = _xcorr_peak_lag(
                env_by_band[a], env_by_band[b], fs, max_lag_s
            ).astype(np.float32)

    background = _background_anchors(
        reader, universe, plan, fs, packed, band_names, band_available
    )

    payload: dict[str, Any] = {
        "event_abs_time": (spec.block_start_epoch + packed[:, 0]).astype(np.float64),
        "core_start_seconds": packed[:, 0].astype(np.float64),
        "core_end_seconds": packed[:, 1].astype(np.float64),
        "core_start_sample": core_start,
        "core_stop_sample": core_stop,
        "ctx_start_sample": ctx_start,
        "ctx_stop_sample": ctx_stop,
        "has_waveform": has_waveform,
        "core_seconds_raw": core_seconds_raw.astype(np.float32),
        "contact_ok": contact_ok,
        "participation": participation,
        "relative_delay_s": rel_delay,
        "tied_group_id": tie_id,
        "legacy_rank": legacy_rank,
        "band_available": band_available,
        "band_envelope": envelopes,
        "band_features": band_features,
        "cross_band_lag_s": cross_band_lag,
        "background_time_s": background["time_s"],
        "background_features": background["features"],
    }
    if legacy_freq is not None:
        payload["legacy_freq_centroid"] = legacy_freq
    for view, array in waveform.items():
        payload[f"waveform_{view}"] = array

    tmp = out_path.with_suffix(".npz.tmp")
    # np.savez appends ".npz" to a *path* whose name does not end in .npz, which
    # would write beside the tempfile and defeat the atomic rename.  Handing it
    # an open file object keeps the name exactly as given.
    with tmp.open("wb") as handle:
        np.savez(handle, **payload)
    os.replace(tmp, out_path)

    manifest = {
        "format": CACHE_FORMAT_VERSION,
        "dataset": spec.dataset,
        "subject": spec.subject,
        "record_name": spec.record_name,
        "block_start_epoch": spec.block_start_epoch,
        "native_rate_hz": fs,
        "n_events": int(n_events),
        "n_events_with_waveform": int(has_waveform.sum()),
        "n_contacts": int(n_contacts),
        "n_context_samples": int(n_ctx),
        "n_core_samples": int(n_core),
        "n_core_seconds_unique": int(np.unique(np.round(core_seconds_raw, 6)).size),
        "core_seconds_nominal": float(n_core / fs),
        "context_pre_seconds": EVENT_CONTEXT_PRE_SECONDS,
        "context_post_seconds": EVENT_CONTEXT_POST_SECONDS,
        "core_offset_samples": core_offset,
        "envelope_bins": ENVELOPE_BINS,
        "bands": list(band_names),
        "band_available": band_available.tolist(),
        "band_edges_hz": {k: list(v) for k, v in ANALYSIS_BANDS_HZ.items()},
        "band_feature_names": list(BAND_FEATURE_NAMES),
        "cross_band_pairs": [[band_names[a], band_names[b]] for a, b in pairs],
        "stored_views": list(views),
        "bipolar_equals_detector": universe.bipolar_equals_detector,
        "detector_reference": universe.detector_reference,
        "montage_provenance": montage_provenance,
        "tie_tolerance_seconds": TIE_TOLERANCE_SECONDS,
        "contacts": [
            {
                "lagpat_label": c.lagpat_label,
                "detector_label": c.detector_label,
                "anode": c.anode,
                "cathode": c.cathode,
                "shaft": c.shaft,
                "number": c.number,
            }
            for c in universe.contacts
        ],
        "source": {
            "raw_path": spec.raw_path,
            "head_path": spec.head_path,
            "gpu_path": spec.gpu_path,
            "lagpat_path": spec.lagpat_path,
            "packed_path": spec.packed_path,
        },
        "background": {
            "grid_seconds": BACKGROUND_GRID_SECONDS,
            "window_seconds": BACKGROUND_WINDOW_SECONDS,
            "n_anchors": int(background["time_s"].size),
            "feature_names": background["feature_names"],
        },
        "build_seconds": round(time.time() - started, 2),
        "shard_bytes": int(out_path.stat().st_size),
    }
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp_manifest, manifest_path)
    return manifest


def _background_anchors(
    reader,
    universe: ContactUniverse,
    plan,
    fs: float,
    packed: np.ndarray,
    band_names: Sequence[str],
    band_available: np.ndarray,
) -> dict[str, Any]:
    """Fixed-grid background windows that never touch a packed event core.

    The rule is deterministic and event-agnostic: walk a 30 s grid, take the
    first 2 s window on that grid step that overlaps no packed event, and drop
    the step otherwise.  This keeps background an *observation of the state*
    rather than a second event clock.
    """

    duration = reader.n_samples / fs
    grid = np.arange(0.0, max(0.0, duration - BACKGROUND_WINDOW_SECONDS), BACKGROUND_GRID_SECONDS)
    starts = packed[:, 0]
    stops = packed[:, 1]
    feature_names = [
        f"{band}_log_power" for band, ok in zip(band_names, band_available) if ok
    ] + list(BACKGROUND_FEATURE_SUFFIXES)

    times: list[float] = []
    rows: list[np.ndarray] = []
    for t0 in grid:
        t1 = t0 + BACKGROUND_WINDOW_SECONDS
        if np.any((stops > t0) & (starts < t1)):
            continue
        a = int(round(t0 * fs))
        b = a + int(round(BACKGROUND_WINDOW_SECONDS * fs))
        if b > reader.n_samples:
            continue
        x = build_event_views(reader, universe, a, b, plan=plan)[VIEW_DETECTOR]
        if np.isnan(x).any():
            continue
        feats = []
        for bi, band in enumerate(band_names):
            if not band_available[bi]:
                continue
            lo_hz, hi_hz = ANALYSIS_BANDS_HZ[band]
            filtered = _bandpass(x.astype(np.float64), fs, lo_hz, hi_hz)
            feats.append(np.log(np.mean(filtered**2, axis=-1) + 1e-30))
        centred = x - x.mean(axis=-1, keepdims=True)
        var = np.mean(centred**2, axis=-1)
        feats.append(np.log(var + 1e-30))
        feats.append(
            np.sum(centred[:, :-1] * centred[:, 1:], axis=-1) / (np.sum(centred**2, axis=-1) + 1e-30)
        )
        spec = np.abs(np.fft.rfft(centred, axis=-1)) ** 2
        freqs = np.fft.rfftfreq(centred.shape[-1], d=1.0 / fs)
        cum = np.cumsum(spec, axis=-1)
        edge_idx = np.argmax(cum >= 0.9 * cum[:, -1:], axis=-1)
        feats.append(freqs[edge_idx])
        times.append(float(t0))
        rows.append(np.stack(feats, axis=-1).astype(np.float32))

    if not rows:
        n_feat = len(feature_names)
        return {
            "time_s": np.zeros(0, dtype=np.float64),
            "features": np.zeros((0, len(universe), n_feat), dtype=np.float32),
            "feature_names": feature_names,
        }
    return {
        "time_s": np.asarray(times, dtype=np.float64),
        "features": np.stack(rows, axis=0),
        "feature_names": feature_names,
    }
