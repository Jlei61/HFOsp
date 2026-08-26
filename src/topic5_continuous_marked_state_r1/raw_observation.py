"""Causal 30 s raw-SEEG observations on an event-independent 30 s clock."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from . import contract


RAW_OBSERVATION_REVISION = "r1_causal_30s_all_bipolar_ied_inpaint_v2"
ANALYSIS_RATE_HZ = 256
WINDOW_SECONDS = 30.0
IED_CORE_HALF_WIDTH_SECONDS = 1.0
SPECTRAL_EDGES_HZ = np.asarray([1.0, 4.0, 8.0, 13.0, 30.0, 60.0, 100.0])
EXPLICIT_NAMES = (
    "log_power_1_4", "log_power_4_8", "log_power_8_13",
    "log_power_13_30", "log_power_30_60", "log_power_60_100",
    "mean_abs", "std", "mean_abs_diff", "zero_crossing_rate",
    "lag1_autocorrelation", "abs_p95", "valid_fraction",
)


@dataclass(frozen=True)
class RawObservation:
    subject: str
    anchor_time: float
    waveform: np.ndarray        # (C,T), TRAIN-normalised raw
    sample_valid: np.ndarray    # (C,T)
    contact_mask: np.ndarray    # (C,)
    explicit: np.ndarray        # (C,13)
    coordinates: np.ndarray     # (C,3), patient-centred/scaled
    coordinate_valid: np.ndarray
    shaft_index: np.ndarray     # categorical shaft identity
    contact_names: np.ndarray

    def validate(self) -> None:
        contacts, samples = self.waveform.shape
        if samples != int(WINDOW_SECONDS * ANALYSIS_RATE_HZ):
            raise ValueError("raw observation has wrong duration")
        if self.sample_valid.shape != self.waveform.shape:
            raise ValueError("raw/sample-valid shape mismatch")
        for value in (
            self.contact_mask, self.coordinates, self.coordinate_valid,
            self.shaft_index, self.contact_names, self.explicit,
        ):
            if len(value) != contacts:
                raise ValueError("raw observation contact axes disagree")
        if self.explicit.shape != (contacts, len(EXPLICIT_NAMES)):
            raise ValueError("explicit feature shape disagrees")
        if not np.isfinite(self.waveform).all() or not np.isfinite(self.explicit).all():
            raise ValueError("raw observation contains non-finite value")
        if not bool(self.contact_mask.any()):
            raise ValueError("raw observation has no usable contact")


def _raw_cache_dir(subject: str) -> Path:
    dataset = subject.split("_", 1)[0]
    base = {
        "epilepsiae": Path("/mnt/yuquan_data/hfosp_cache/raw_seeg_state_r0_1"),
        "yuquan": Path("/mnt/epilepsia_data/hfosp_cache/raw_seeg_state_r0_1"),
    }[dataset]
    return base / subject


def _explicit_features(waveform: np.ndarray,
                       valid: np.ndarray) -> np.ndarray:
    """Per-contact fixed features; invalid samples never become zeros as data."""
    contacts, samples = waveform.shape
    filled = np.where(valid, waveform, 0.0)
    count = valid.sum(-1).clip(min=1)
    mean = filled.sum(-1) / count
    mean_abs = np.abs(filled).sum(-1) / count
    centred = np.where(valid, waveform - mean[:, None], 0.0)
    variance = (centred ** 2).sum(-1) / count
    std = np.sqrt(np.maximum(variance, 1e-8))
    pair = valid[:, 1:] & valid[:, :-1]
    pair_count = pair.sum(-1).clip(min=1)
    difference = np.where(pair, np.diff(filled, axis=-1), 0.0)
    mean_abs_diff = np.abs(difference).sum(-1) / pair_count
    crossing = np.where(pair, filled[:, 1:] * filled[:, :-1] < 0, False)
    zero_crossing = crossing.sum(-1) / pair_count
    autocorrelation = (
        np.where(pair, centred[:, 1:] * centred[:, :-1], 0.0).sum(-1)
        / np.maximum(pair_count * variance, 1e-6)
    )
    abs_p95 = np.zeros(contacts, dtype=np.float32)
    for contact in range(contacts):
        values = np.abs(waveform[contact, valid[contact]])
        if len(values):
            abs_p95[contact] = float(np.percentile(values, 95))

    nper = 512
    nseg = samples // nper
    segment = filled[:, :nseg * nper].reshape(contacts, nseg, nper)
    segment_valid = valid[:, :nseg * nper].reshape(contacts, nseg, nper)
    good = segment_valid.mean(-1) >= 0.80
    window = np.hanning(nper).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(segment * window[None, None, :], axis=-1)) ** 2
    frequency = np.fft.rfftfreq(nper, d=1.0 / ANALYSIS_RATE_HZ)
    bands = []
    for left, right in zip(SPECTRAL_EDGES_HZ[:-1], SPECTRAL_EDGES_HZ[1:]):
        use = (frequency >= left) & (frequency < right)
        band = np.log1p(spectrum[..., use].mean(-1))
        value = np.divide(
            np.where(good, band, 0.0).sum(-1), good.sum(-1),
            out=np.zeros(contacts, dtype=np.float64), where=good.sum(-1) > 0,
        )
        bands.append(value)
    return np.column_stack([
        np.stack(bands, axis=1),
        mean_abs, std, mean_abs_diff, zero_crossing, autocorrelation,
        abs_p95, valid.mean(-1),
    ]).astype(np.float32)


class RawAnchorReader:
    def __init__(self, subject: str, event_times: np.ndarray):
        import pandas as pd
        import zarr

        self.subject = subject
        self.cache_dir = _raw_cache_dir(subject)
        required = (
            self.cache_dir / "raw_256hz.zarr",
            self.cache_dir / "artifact_mask.zarr",
            self.cache_dir / "train_stats.json",
            self.cache_dir / "window_index_refined.parquet",
            self.cache_dir / "cache_index.parquet",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{subject}: missing raw cache inputs {missing}")
        self.raw = zarr.open_array(str(required[0]), mode="r")
        self.artifact = zarr.open_array(str(required[1]), mode="r")
        stats = json.loads(required[2].read_text())
        self.count_scale = np.asarray(stats["int16_scale_uv"], dtype=np.float32)
        self.raw_center = np.asarray(stats["raw_center_uv"], dtype=np.float32)
        self.raw_scale = np.asarray(stats["raw_scale_uv"], dtype=np.float32)
        self.window_index = pd.read_parquet(required[3]).sort_values("minute_index")
        cache_index = pd.read_parquet(required[4])
        n_minutes = int(self.window_index.minute_index.max()) + 1
        self.minute_start = np.full(n_minutes, np.nan, dtype=np.float64)
        self.minute_session = np.full(n_minutes, -1, dtype=np.int64)
        self.usable = np.zeros(n_minutes, dtype=bool)
        row_index = self.window_index.minute_index.to_numpy(dtype=int)
        self.minute_start[row_index] = self.window_index.minute_start_epoch.to_numpy(dtype=float)
        self.minute_session[row_index] = self.window_index.session_id.to_numpy(dtype=int)
        self.usable[row_index] = (
            self.window_index.covered.to_numpy(dtype=bool)
            & self.window_index.guard_free.to_numpy(dtype=bool)
            & self.window_index.minute_usable.to_numpy(dtype=bool)
        )
        cached = np.zeros(n_minutes, dtype=bool)
        ci = cache_index.minute_index.to_numpy(dtype=int)
        cached[ci] = cache_index.cached.to_numpy(dtype=bool)
        self.usable &= cached
        self.first_epoch = float(self.minute_start[0])
        self.event_times = np.asarray(event_times, dtype=np.float64)
        self.window_samples = int(WINDOW_SECONDS * ANALYSIS_RATE_HZ)
        self.minute_samples = 60 * ANALYSIS_RATE_HZ
        # Zarr is chunked by complete minute.  The 30 s clock asks for the two
        # halves consecutively; retaining one decoded chunk avoids decompressing
        # every minute twice without retaining a patient-scale raw array.
        self._decoded_minute_index = -1
        self._decoded_minute_raw: np.ndarray | None = None
        self._decoded_minute_artifact: np.ndarray | None = None

        metadata_path = contract.RAW_STATE_ROOT / "data/contact_metadata.parquet"
        metadata = pd.read_parquet(metadata_path)
        metadata = metadata[metadata.subject.astype(str) == subject].sort_values("channel_index")
        metadata = metadata[metadata.contact_valid.astype(bool)]
        if len(metadata) != self.raw.shape[1]:
            raise ValueError(
                f"{subject}: metadata/raw contact counts disagree "
                f"({len(metadata)} vs {self.raw.shape[1]})"
            )
        if not np.array_equal(metadata.channel_index.to_numpy(), np.arange(len(metadata))):
            raise ValueError(f"{subject}: raw contact metadata is not in channel_index order")
        self.contact_names = metadata.channel_name.astype(str).to_numpy()
        coordinate = metadata[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=np.float32)
        self.coordinate_valid = metadata.coord_valid.to_numpy(dtype=bool) & np.isfinite(coordinate).all(1)
        if bool(self.coordinate_valid.any()):
            centre = coordinate[self.coordinate_valid].mean(axis=0)
        else:
            centre = np.zeros(3, dtype=np.float32)
        centred = np.where(self.coordinate_valid[:, None], coordinate - centre, 0.0)
        if bool(self.coordinate_valid.any()):
            scale = centred[self.coordinate_valid].std(axis=0)
        else:
            scale = np.ones(3, dtype=np.float32)
        scale = np.where(np.isfinite(scale) & (scale > 1e-3), scale, 1.0)
        self.coordinates = (centred / scale).astype(np.float32)
        _, shaft_index = np.unique(metadata.shaft.astype(str).to_numpy(), return_inverse=True)
        self.shaft_index = shaft_index.astype(np.int64)

    def anchor_times(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return event-independent 30 s anchors, their split and cache session."""
        train_end, dev_end = contract.load_split(self.subject)
        minute = np.flatnonzero(self.usable)
        anchors = np.concatenate([
            self.minute_start[minute] + 30.0,
            self.minute_start[minute] + 60.0,
        ])
        sessions = np.concatenate([
            self.minute_session[minute], self.minute_session[minute]
        ])
        order = np.argsort(anchors, kind="stable")
        anchors, sessions = anchors[order], sessions[order]
        keep = anchors < dev_end
        anchors, sessions = anchors[keep], sessions[keep]
        split = np.where(anchors < train_end, 0, 1).astype(np.int8)
        return anchors, split, sessions

    def _window_location(self, anchor_time: float) -> tuple[int, int] | None:
        """Locate a causal 30 s window without decoding its raw minute."""
        start = float(anchor_time) - WINDOW_SECONDS
        minute = int(math.floor((start - self.first_epoch) / 60.0 + 1e-9))
        if minute < 0 or minute >= len(self.usable) or not self.usable[minute]:
            return None
        offset = int(round((start - self.minute_start[minute]) * ANALYSIS_RATE_HZ))
        if offset < 0 or offset + self.window_samples > self.minute_samples:
            return None
        return minute, offset

    def _ied_core_mask(self, anchor_time: float) -> np.ndarray:
        """Return the causal IED-core mask shared by eligibility and decoding."""
        start = float(anchor_time) - WINDOW_SECONDS
        event_lo = np.searchsorted(
            self.event_times, start - IED_CORE_HALF_WIDTH_SECONDS, side="left"
        )
        event_hi = np.searchsorted(
            # An event after the anchor is not yet observable, even when its
            # offline +/-1 s annotation overlaps the input window.
            self.event_times, anchor_time, side="right"
        )
        ied_core = np.zeros(self.window_samples, dtype=bool)
        for event_time in self.event_times[event_lo:event_hi]:
            left = max(0, int(math.floor(
                (float(event_time) - IED_CORE_HALF_WIDTH_SECONDS - start)
                * ANALYSIS_RATE_HZ
            )))
            right = min(self.window_samples, int(math.ceil(
                (float(event_time) + IED_CORE_HALF_WIDTH_SECONDS - start)
                * ANALYSIS_RATE_HZ
            )))
            if right > left:
                ied_core[left:right] = True
        return ied_core

    def can_read(self, anchor_time: float) -> bool:
        """Exact cheap eligibility check used to freeze raw-anchor denominators."""
        location = self._window_location(anchor_time)
        if location is None:
            return False
        minute, _ = location
        contact_bad = np.asarray(self.artifact[minute], dtype=bool)
        if float((~contact_bad).mean()) < 0.50:
            return False
        # With fewer than two background samples, interpolation is undefined.
        return int((~self._ied_core_mask(anchor_time)).sum()) >= 2

    def read(self, anchor_time: float, *, compute_explicit: bool = True
             ) -> RawObservation | None:
        start = float(anchor_time) - WINDOW_SECONDS
        location = self._window_location(anchor_time)
        if location is None:
            return None
        minute, offset = location
        if self._decoded_minute_index != minute:
            minute_lo = minute * self.minute_samples
            self._decoded_minute_raw = np.asarray(
                self.raw[minute_lo:minute_lo + self.minute_samples],
                dtype=np.float32,
            ).T
            self._decoded_minute_artifact = np.asarray(
                self.artifact[minute], dtype=bool
            )
            self._decoded_minute_index = minute
        if self._decoded_minute_raw is None or self._decoded_minute_artifact is None:
            raise RuntimeError("raw minute cache failed to initialise")
        raw = self._decoded_minute_raw[:, offset:offset + self.window_samples]
        waveform = raw * self.count_scale[:, None]
        waveform = (
            waveform - self.raw_center[:, None]
        ) / np.maximum(self.raw_scale[:, None], 1e-4)
        # Only recording/artifact validity is observable by the network.  IED
        # locations are used causally for background inpainting below but their
        # mask pattern is not exposed; deterministic IED history belongs in r(t).
        sample_valid = np.ones_like(waveform, dtype=bool)
        contact_bad = self._decoded_minute_artifact
        sample_valid[contact_bad] = False
        ied_core = self._ied_core_mask(anchor_time)
        # Contact eligibility is a recording/artifact property, not a function
        # of how many IED cores happened inside this window.  Basing it on the
        # post-IED-mask valid fraction would preferentially discard high-rate
        # intervals from the point-process exposure set.
        contact_mask = ~contact_bad
        if bool(ied_core.any()):
            index = np.arange(self.window_samples)
            keep = ~ied_core
            if int(keep.sum()) < 2:
                contact_mask[:] = False
            else:
                # Linear interpolation removes the event core without revealing
                # a zero/missing-value stencil to the raw or explicit branch.
                waveform = waveform.copy()
                for contact in np.flatnonzero(contact_mask):
                    waveform[contact, ied_core] = np.interp(
                        index[ied_core], index[keep], waveform[contact, keep]
                    ).astype(np.float32)
        if contact_mask.mean() < 0.50:
            return None
        # R1.3 caches these deterministic features once per frozen anchor.  Raw
        # target-training still streams the waveform, but must not repeat 13
        # FFT/statistic computations for every seed and epoch.
        explicit = (
            _explicit_features(waveform, sample_valid)
            if compute_explicit else
            np.zeros((len(waveform), len(EXPLICIT_NAMES)), dtype=np.float32)
        )
        clean = np.where(sample_valid, waveform, 0.0).astype(np.float32)
        value = RawObservation(
            subject=self.subject, anchor_time=float(anchor_time),
            waveform=clean, sample_valid=sample_valid,
            contact_mask=contact_mask, explicit=explicit,
            coordinates=self.coordinates, coordinate_valid=self.coordinate_valid,
            shaft_index=self.shaft_index, contact_names=self.contact_names,
        )
        value.validate()
        return value
