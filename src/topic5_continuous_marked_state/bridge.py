"""Build and fit the development-only B0--B3 information Bridge.

The Bridge predicts the exact next inter-event interval and next spatial mark
from the current event.  It is deliberately not called a state model.  The
background window ends at the current event, known IED cores are converted to
missing samples, and train-only PCA maps every observation arm to the same
32-dimensional slot.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch import nn

from . import contract


BASE_HISTORY_NAMES = (
    "log_previous_iei",
    "trace_30s",
    "trace_2m",
    "last_event_load",
    "recent_mean_load_5",
    "previous_mark_jaccard",
    "group_fraction",
    "tod_sin",
    "tod_cos",
    "log_session_elapsed_minutes",
)


def history_names(n_contacts: int) -> tuple[str, ...]:
    return BASE_HISTORY_NAMES + tuple(
        f"current_participation_c{i}" for i in range(n_contacts)
    ) + tuple(
        f"current_rank_c{i}" for i in range(n_contacts)
    ) + tuple(
        f"participation_trace_2m_c{i}" for i in range(n_contacts)
    )

RAW_METRIC_NAMES = (
    "mean_abs",
    "std",
    "mean_abs_diff",
    "zero_crossing_rate",
    "lag1_autocorrelation",
    "abs_p95",
    "valid_fraction",
)

SPECTRAL_EDGES_HZ = np.asarray([1.0, 4.0, 8.0, 13.0, 30.0, 60.0, 100.0])


@dataclass
class BridgeArrays:
    subject: str
    history: np.ndarray
    spectral: np.ndarray
    raw: np.ndarray
    log_next_iei: np.ndarray
    participation: np.ndarray
    rank: np.ndarray
    stop_fraction: np.ndarray
    split: np.ndarray
    current_time: np.ndarray
    next_time: np.ndarray
    current_event_index: np.ndarray
    observation_valid_fraction: np.ndarray

    def validate(self) -> None:
        n = len(self.split)
        arrays = (
            self.history, self.spectral, self.raw, self.log_next_iei,
            self.participation, self.rank, self.stop_fraction,
            self.current_time, self.next_time, self.current_event_index,
            self.observation_valid_fraction,
        )
        if any(len(x) != n for x in arrays):
            raise ValueError(f"{self.subject}: Bridge arrays have unequal rows")
        if set(np.unique(self.split).tolist()) - {0, 1}:
            raise ValueError(f"{self.subject}: sealed split entered Bridge arrays")
        if np.any(self.next_time <= self.current_time):
            raise ValueError(f"{self.subject}: non-positive next-event interval")
        if not np.isfinite(self.history).all():
            raise ValueError(f"{self.subject}: non-finite explicit history")
        if not np.isfinite(self.spectral).all() or not np.isfinite(self.raw).all():
            raise ValueError(f"{self.subject}: non-finite observation feature")
        for code, name in ((0, "train"), (1, "validation")):
            m = self.split == code
            contract.assert_development_times(self.subject, self.current_time[m], name)
            contract.assert_development_times(self.subject, self.next_time[m], name)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as f:
            np.savez_compressed(
                f, subject=np.asarray(self.subject), history=self.history,
                spectral=self.spectral, raw=self.raw,
                log_next_iei=self.log_next_iei,
                participation=self.participation, rank=self.rank,
                stop_fraction=self.stop_fraction, split=self.split,
                current_time=self.current_time, next_time=self.next_time,
                current_event_index=self.current_event_index,
                observation_valid_fraction=self.observation_valid_fraction,
                history_names=np.asarray(history_names(self.participation.shape[1])),
                raw_metric_names=np.asarray(RAW_METRIC_NAMES),
                spectral_edges_hz=SPECTRAL_EDGES_HZ,
            )
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "BridgeArrays":
        with np.load(path, allow_pickle=False) as z:
            out = cls(
                subject=str(z["subject"].item()), history=z["history"],
                spectral=z["spectral"], raw=z["raw"],
                log_next_iei=z["log_next_iei"],
                participation=z["participation"], rank=z["rank"],
                stop_fraction=z["stop_fraction"], split=z["split"],
                current_time=z["current_time"], next_time=z["next_time"],
                current_event_index=z["current_event_index"],
                observation_valid_fraction=z["observation_valid_fraction"],
            )
        out.validate()
        return out


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _uniform_take(indices: np.ndarray, limit: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= limit:
        return indices
    positions = np.linspace(0, len(indices) - 1, limit).round().astype(np.int64)
    return indices[np.unique(positions)]


def _explicit_history(event_time: np.ndarray, session: np.ndarray,
                      participation: np.ndarray, n_groups: np.ndarray,
                      load: np.ndarray, rank: np.ndarray, dataset: str) -> np.ndarray:
    n = len(event_time)
    n_contacts = participation.shape[1]
    out = np.zeros((n, len(history_names(n_contacts))), dtype=np.float32)
    trace30 = trace120 = 0.0
    participation_trace = np.zeros(n_contacts, dtype=np.float64)
    recent: list[float] = []
    session_start = float(event_time[0])
    zone = ZoneInfo("Europe/Berlin" if dataset == "epilepsiae" else "Asia/Shanghai")
    for i in range(n):
        opening = i == 0 or session[i] != session[i - 1]
        if opening:
            trace30 = trace120 = 0.0
            participation_trace.fill(0.0)
            recent = []
            session_start = float(event_time[i])
            dt = 0.0
            jaccard = 0.0
        else:
            dt = max(float(event_time[i] - event_time[i - 1]), 0.0)
            trace30 *= math.exp(-dt / 30.0)
            trace120 *= math.exp(-dt / 120.0)
            participation_trace *= math.exp(-dt / 120.0)
            union = np.logical_or(participation[i], participation[i - 1]).sum()
            inter = np.logical_and(participation[i], participation[i - 1]).sum()
            jaccard = float(inter / union) if union else 0.0
        trace30 += 1.0
        trace120 += 1.0
        participation_trace += participation[i].astype(np.float64)
        recent.append(float(load[i]))
        if len(recent) > 5:
            recent.pop(0)
        local = datetime.fromtimestamp(float(event_time[i]), tz=zone)
        phase = 2.0 * math.pi * (local.hour * 3600 + local.minute * 60 + local.second) / 86400.0
        base = np.asarray([
            math.log1p(dt), trace30, trace120, float(load[i]),
            float(np.mean(recent)), jaccard,
            float(n_groups[i]) / float(participation.shape[1]),
            math.sin(phase), math.cos(phase),
            math.log1p(max(float(event_time[i]) - session_start, 0.0) / 60.0),
        ], dtype=np.float32)
        out[i] = np.concatenate([
            base,
            participation[i].astype(np.float32),
            np.where(participation[i], rank[i], 0.0).astype(np.float32),
            participation_trace.astype(np.float32),
        ])
    return out


class BackgroundReader:
    """Read causal 30 s windows from the existing train-normalised raw cache."""

    def __init__(self, subject: str, all_event_times: np.ndarray):
        import pandas as pd
        import zarr

        self.subject = subject
        self.cache_dir = contract.raw_cache_dir(subject)
        self.raw = zarr.open_array(str(self.cache_dir / "raw_256hz.zarr"), mode="r")
        stats = json.loads((self.cache_dir / "train_stats.json").read_text())
        self.count_scale = np.asarray(stats["int16_scale_uv"], dtype=np.float32)
        self.raw_center = np.asarray(stats["raw_center_uv"], dtype=np.float32)
        self.raw_scale = np.asarray(stats["raw_scale_uv"], dtype=np.float32)
        self.artifact = np.asarray(
            zarr.open_array(str(self.cache_dir / "artifact_mask.zarr"), mode="r"),
            dtype=bool,
        )
        wi = pd.read_parquet(self.cache_dir / "window_index_refined.parquet")
        ci = pd.read_parquet(self.cache_dir / "cache_index.parquet")
        self.first_epoch = float(wi["minute_start_epoch"].iloc[0])
        n_min = int(wi["minute_index"].max()) + 1
        self.usable = np.zeros(n_min, dtype=bool)
        widx = wi["minute_index"].to_numpy(dtype=int)
        self.usable[widx] = (
            wi["minute_usable"].to_numpy(dtype=bool)
            & wi["covered"].to_numpy(dtype=bool)
            & wi["guard_free"].to_numpy(dtype=bool)
        )
        cached = np.zeros(n_min, dtype=bool)
        cidx = ci["minute_index"].to_numpy(dtype=int)
        cached[cidx] = ci["cached"].to_numpy(dtype=bool)
        self.usable &= cached
        self.event_times = np.asarray(all_event_times, dtype=np.float64)
        self.fs = contract.ANALYSIS_RATE_HZ
        self.window_samples = int(round(contract.BACKGROUND_SECONDS * self.fs))
        self.minute_samples = 60 * self.fs
        self._minute_cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def _minute(self, index: int) -> np.ndarray:
        if index in self._minute_cache:
            value = self._minute_cache.pop(index)
            self._minute_cache[index] = value
            return value
        lo = index * self.minute_samples
        value = np.asarray(self.raw[lo:lo + self.minute_samples, :], dtype=np.float32)
        self._minute_cache[index] = value
        while len(self._minute_cache) > 8:
            self._minute_cache.popitem(last=False)
        return value

    def _raw_window(self, s0: int, s1: int) -> np.ndarray:
        first = s0 // self.minute_samples
        last = (s1 - 1) // self.minute_samples
        pieces = []
        for minute in range(first, last + 1):
            lo = max(s0 - minute * self.minute_samples, 0)
            hi = min(s1 - minute * self.minute_samples, self.minute_samples)
            pieces.append(self._minute(minute)[lo:hi])
        return np.concatenate(pieces, axis=0)

    def features(self, anchor_time: float, *, include_raw: bool = True,
                 include_spectral: bool = True) -> tuple[np.ndarray, np.ndarray, float] | None:
        start = float(anchor_time) - contract.BACKGROUND_SECONDS
        end = float(anchor_time)
        s0 = int(round((start - self.first_epoch) * self.fs))
        s1 = int(round((end - self.first_epoch) * self.fs))
        if s0 < 0 or s1 > self.raw.shape[0] or s1 - s0 != self.window_samples:
            return None
        m0 = int(math.floor((start - self.first_epoch) / 60.0))
        m1 = int(math.floor((np.nextafter(end, start) - self.first_epoch) / 60.0))
        minutes = np.arange(m0, m1 + 1, dtype=int)
        if minutes.size == 0 or minutes.min() < 0 or minutes.max() >= len(self.usable):
            return None
        if not bool(self.usable[minutes].all()):
            return None
        counts = self._raw_window(s0, s1)
        x = counts * self.count_scale[None, :]
        x = (x - self.raw_center[None, :]) / np.maximum(self.raw_scale[None, :], 1e-4)

        contact_bad = np.asarray(self.artifact[minutes, :], dtype=bool).any(axis=0)
        time_valid = np.ones(len(x), dtype=bool)
        lo = np.searchsorted(
            self.event_times, start - contract.IED_CORE_HALF_WIDTH_SECONDS, side="left"
        )
        hi = np.searchsorted(
            self.event_times, end + contract.IED_CORE_HALF_WIDTH_SECONDS, side="right"
        )
        for event_time in self.event_times[lo:hi]:
            a = max(0, int(math.floor(
                (float(event_time) - contract.IED_CORE_HALF_WIDTH_SECONDS - start) * self.fs
            )))
            b = min(len(x), int(math.ceil(
                (float(event_time) + contract.IED_CORE_HALF_WIDTH_SECONDS - start) * self.fs
            )))
            if b > a:
                time_valid[a:b] = False
        valid_fraction = float(time_valid.mean() * (~contact_bad).mean())
        if time_valid.mean() < 0.50 or (~contact_bad).mean() < 0.50:
            return None
        x[~time_valid, :] = np.nan
        x[:, contact_bad] = np.nan
        raw_feature = _raw_features(x) if include_raw else np.empty(0, dtype=np.float32)
        spectral_feature = (
            _spectral_features(x, self.fs)
            if include_spectral else np.empty(0, dtype=np.float32)
        )
        return spectral_feature, raw_feature, valid_fraction


def _raw_features(x: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x)
    filled = np.where(finite, x, 0.0)
    count = finite.sum(axis=0).clip(min=1)
    mean_abs = np.abs(filled).sum(axis=0) / count
    mean = filled.sum(axis=0) / count
    var = ((np.where(finite, x - mean[None, :], 0.0)) ** 2).sum(axis=0) / count
    std = np.sqrt(np.maximum(var, 1e-8))
    pair = finite[1:] & finite[:-1]
    pair_count = pair.sum(axis=0).clip(min=1)
    diff = np.where(pair, np.diff(filled, axis=0), 0.0)
    madiff = np.abs(diff).sum(axis=0) / pair_count
    zc = (np.where(pair, filled[1:] * filled[:-1] < 0, False)).sum(axis=0) / pair_count
    ac_num = np.where(pair, (filled[1:] - mean) * (filled[:-1] - mean), 0.0).sum(axis=0)
    ac = ac_num / np.maximum(pair_count * var, 1e-6)
    abs95 = np.zeros(x.shape[1], dtype=np.float32)
    for contact in range(x.shape[1]):
        values = np.abs(x[finite[:, contact], contact])
        if values.size:
            abs95[contact] = float(np.percentile(values, 95))
    valid = finite.mean(axis=0)
    feat = np.stack([mean_abs, std, madiff, zc, ac, abs95, valid], axis=1)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32).ravel()


def _spectral_features(x: np.ndarray, fs: int) -> np.ndarray:
    nper = 512
    nseg = x.shape[0] // nper
    trimmed = x[:nseg * nper].reshape(nseg, nper, x.shape[1])
    finite = np.isfinite(trimmed)
    good_segment = finite.mean(axis=1) >= 0.80
    filled = np.where(finite, trimmed, 0.0)
    window = np.hanning(nper).astype(np.float32)[None, :, None]
    fft = np.fft.rfft(filled * window, axis=1)
    power = (np.abs(fft) ** 2).astype(np.float32)
    power = np.where(good_segment[:, None, :], power, np.nan)
    power_count = np.isfinite(power).sum(axis=0)
    mean_power = np.nansum(power, axis=0) / np.maximum(power_count, 1)
    mean_power[power_count == 0] = np.nan
    freq = np.fft.rfftfreq(nper, d=1.0 / fs)
    bands = []
    for lo, hi in zip(SPECTRAL_EDGES_HZ[:-1], SPECTRAL_EDGES_HZ[1:]):
        mask = (freq >= lo) & (freq < hi)
        band = mean_power[mask]
        band_count = np.isfinite(band).sum(axis=0)
        band_mean = np.nansum(band, axis=0) / np.maximum(band_count, 1)
        band_mean[band_count == 0] = np.nan
        bands.append(np.log1p(band_mean))
    finite_all = np.isfinite(x)
    count = finite_all.sum(axis=0).clip(min=1)
    mean = np.where(finite_all, x, 0.0).sum(axis=0) / count
    variance = np.where(finite_all, (x - mean[None, :]) ** 2, 0.0).sum(axis=0) / count
    pair = finite_all[1:] & finite_all[:-1]
    pcount = pair.sum(axis=0).clip(min=1)
    ac = np.where(pair, (np.nan_to_num(x[1:]) - mean) * (np.nan_to_num(x[:-1]) - mean), 0.0).sum(axis=0)
    ac /= np.maximum(pcount * variance, 1e-6)
    feat = np.stack([*bands, np.log1p(variance), ac], axis=1)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32).ravel()


def build_bridge_arrays(subject: str, *, max_train: int = contract.MAX_TRAIN_PAIRS,
                        max_validation: int = contract.MAX_VALIDATION_PAIRS) -> BridgeArrays:
    payload = torch.load(contract.COHORT_CACHE, map_location="cpu", weights_only=False)[subject]
    times = payload["event_time"].numpy().astype(np.float64)
    upstream_split = payload["split"].numpy().astype(np.int8)
    session = payload["session_index"].numpy().astype(np.int64)
    participation = payload["participation"].numpy().astype(bool)
    n_groups = payload["n_groups"].numpy().astype(np.int64)
    marks = payload["marks"].numpy().astype(np.float32)
    load = payload["load"].numpy().astype(np.float32)
    history_all = _explicit_history(
        times, session, participation, n_groups, load, marks[:, :, 1],
        str(payload["dataset"])
    )
    # Re-derive the Bridge split from the one wall-clock source of truth.
    # The upstream integer split includes the event exactly equal to each
    # boundary in the earlier partition, while this contract is strictly
    # half-open. Trusting both silently moves one event per patient.
    bound = contract.load_split(subject)
    split = np.full(len(times), 2, dtype=np.int8)
    split[times < bound.dev_end_epoch] = 1
    split[times < bound.train_end_epoch] = 0
    if int((upstream_split != split).sum()) > 2:
        raise ValueError(
            f"{subject}: upstream-vs-clock split differs by more than boundary equality"
        )
    same_session = session[1:] == session[:-1]
    same_split = split[1:] == split[:-1]
    positive = np.diff(times) > 0
    pair_ok = same_session & same_split & positive & (split[:-1] < 2)
    train = _uniform_take(np.flatnonzero(pair_ok & (split[:-1] == 0)), max_train)
    valid = _uniform_take(np.flatnonzero(pair_ok & (split[:-1] == 1)), max_validation)
    selected = np.concatenate([train, valid])
    selected_split = np.concatenate([
        np.zeros(len(train), dtype=np.int8), np.ones(len(valid), dtype=np.int8)
    ])

    reader = BackgroundReader(subject, times)
    rows = []
    for event_index, split_code in zip(selected.tolist(), selected_split.tolist()):
        obs = reader.features(float(times[event_index]))
        if obs is None:
            continue
        spectral, raw, fraction = obs
        rows.append((event_index, split_code, spectral, raw, fraction))
    if not rows:
        raise ValueError(f"{subject}: no Bridge pair has a usable masked background window")
    idx = np.asarray([r[0] for r in rows], dtype=np.int64)
    sp = np.asarray([r[1] for r in rows], dtype=np.int8)
    next_idx = idx + 1
    out = BridgeArrays(
        subject=subject,
        history=history_all[idx],
        spectral=np.stack([r[2] for r in rows]),
        raw=np.stack([r[3] for r in rows]),
        log_next_iei=np.log(np.maximum(times[next_idx] - times[idx], 1e-3)).astype(np.float32),
        participation=participation[next_idx].astype(np.float32),
        rank=marks[next_idx, :, 1].astype(np.float32),
        stop_fraction=(n_groups[next_idx] / participation.shape[1]).astype(np.float32),
        split=sp,
        current_time=times[idx], next_time=times[next_idx],
        current_event_index=idx,
        observation_valid_fraction=np.asarray([r[4] for r in rows], dtype=np.float32),
    )
    out.validate()
    return out


def write_bridge_dataset(subject: str, output: Path, *, max_train: int,
                         max_validation: int) -> dict:
    arrays = build_bridge_arrays(
        subject, max_train=max_train, max_validation=max_validation
    )
    arrays.save(output)
    split = contract.load_split(subject)
    manifest = {
        "contract": contract.REVISION,
        "subject": subject,
        "output": str(output.resolve()),
        "sha256": _sha256(output),
        "n_rows": int(len(arrays.split)),
        "n_train": int((arrays.split == 0).sum()),
        "n_validation": int((arrays.split == 1).sum()),
        "n_event_contacts": int(arrays.participation.shape[1]),
        "spectral_feature_dim_raw": int(arrays.spectral.shape[1]),
        "raw_feature_dim_raw": int(arrays.raw.shape[1]),
        "explicit_history_dim": int(arrays.history.shape[1]),
        "explicit_history_semantics": "base renewal/session covariates plus current full participation/rank and 2-minute contact participation trace",
        "median_observation_valid_fraction": float(np.median(arrays.observation_valid_fraction)),
        "max_train_current_time": float(arrays.current_time[arrays.split == 0].max()),
        "min_validation_current_time": float(arrays.current_time[arrays.split == 1].min()),
        "max_validation_next_time": float(arrays.next_time[arrays.split == 1].max()),
        "train_end_epoch": split.train_end_epoch,
        "dev_end_epoch": split.dev_end_epoch,
        "ied_core_half_width_seconds": contract.IED_CORE_HALF_WIDTH_SECONDS,
        "background_seconds": contract.BACKGROUND_SECONDS,
        "sealed_opened": False,
        "raw_semantics": "fixed masked raw-derived E0 features; not final Transformer embedding",
        "split_semantics": "re-derived from half-open wall-clock train/dev bounds",
    }
    _atomic_json(output.with_suffix(".manifest.json"), manifest)
    return manifest


class BridgeHead(nn.Module):
    """Identical linear exact-time and mark heads for all four feature arms."""

    def __init__(self, input_dim: int, n_contacts: int, *, time_sigma: float,
                 rank_sigma: float, stop_sigma: float):
        super().__init__()
        self.time_mean = nn.Linear(input_dim, 1)
        self.participation = nn.Linear(input_dim, n_contacts)
        self.rank = nn.Linear(input_dim, n_contacts)
        self.stop = nn.Linear(input_dim, 1)
        # The scale is estimated once from TRAIN targets and then fixed for all
        # arms. Letting each arm shrink its own sigma made a small pilot appear
        # catastrophically bad solely through overconfidence.
        self.register_buffer("time_sigma", torch.tensor(float(time_sigma)))
        self.register_buffer("rank_sigma", torch.tensor(float(rank_sigma)))
        self.register_buffer("stop_sigma", torch.tensor(float(stop_sigma)))

    def losses(self, x: torch.Tensor, log_iei: torch.Tensor,
               participation: torch.Tensor, rank: torch.Tensor,
               stop_fraction: torch.Tensor) -> dict[str, torch.Tensor]:
        time_mu = self.time_mean(x).squeeze(-1)
        timing = 0.5 * ((log_iei - time_mu) / self.time_sigma) ** 2 + self.time_sigma.log() + 0.5 * math.log(2 * math.pi)
        logits = self.participation(x)
        part = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, participation, reduction="none"
        ).mean(-1)
        rank_mu = self.rank(x)
        rank_terms = 0.5 * ((rank - rank_mu) / self.rank_sigma) ** 2 + self.rank_sigma.log() + 0.5 * math.log(2 * math.pi)
        rank_nll = (rank_terms * participation).sum(-1) / participation.sum(-1).clamp(min=1)
        stop_mu = self.stop(x).squeeze(-1)
        stop_nll = 0.5 * ((stop_fraction - stop_mu) / self.stop_sigma) ** 2 + self.stop_sigma.log() + 0.5 * math.log(2 * math.pi)
        mark = part + rank_nll + stop_nll
        return {"timing_nll": timing, "participation_nll": part,
                "rank_nll": rank_nll, "stop_nll": stop_nll,
                "mark_nll": mark, "joint_nll": timing + mark,
                "time_mu": time_mu, "participation_prob": logits.sigmoid(),
                "rank_mu": rank_mu, "stop_mu": stop_mu}


def _arm_matrix(arrays: BridgeArrays, arm: str) -> tuple[np.ndarray, dict]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    train = arrays.split == 0
    history_scaler = StandardScaler().fit(arrays.history[train])
    history = history_scaler.transform(arrays.history).astype(np.float32)
    if arm == "b0_history":
        obs = np.zeros((len(history), contract.OBSERVATION_DIM), dtype=np.float32)
        explained = 0.0
    else:
        if arm == "b1_spectral":
            source = arrays.spectral
        elif arm == "b2_raw":
            source = arrays.raw
        elif arm == "b3_both":
            source = np.concatenate([arrays.spectral, arrays.raw], axis=1)
        else:
            raise ValueError(f"unknown Bridge arm {arm!r}")
        scaler = StandardScaler().fit(source[train])
        standardized_train = scaler.transform(source[train])
        # About twenty train events per active component is a pragmatic E0
        # regularisation rule. The slot remains 32-D in every arm, so the head
        # architecture and parameter count are unchanged.
        support_cap = max(1, standardized_train.shape[0] // 20)
        n_components = min(contract.OBSERVATION_DIM, support_cap,
                           standardized_train.shape[0] - 1, standardized_train.shape[1])
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
        pca.fit(standardized_train)
        reduced = pca.transform(scaler.transform(source)).astype(np.float32)
        obs = np.zeros((len(source), contract.OBSERVATION_DIM), dtype=np.float32)
        obs[:, :n_components] = reduced
        explained = float(pca.explained_variance_ratio_.sum())
    return np.concatenate([history, obs], axis=1), {
        "history_dim": int(history.shape[1]),
        "observation_slot_dim": int(obs.shape[1]),
        "pca_explained_variance": explained,
    }


class FrozenHistoryResidualHead(nn.Module):
    """A shared frozen history head plus an observation-only residual."""

    def __init__(self, baseline: BridgeHead, observation_dim: int):
        super().__init__()
        self.baseline = baseline
        for parameter in self.baseline.parameters():
            parameter.requires_grad_(False)
        n_contacts = baseline.participation.out_features
        self.obs_time = nn.Linear(observation_dim, 1, bias=False)
        self.obs_participation = nn.Linear(observation_dim, n_contacts, bias=False)
        self.obs_rank = nn.Linear(observation_dim, n_contacts, bias=False)
        self.obs_stop = nn.Linear(observation_dim, 1, bias=False)
        for module in (
            self.obs_time, self.obs_participation,
            self.obs_rank, self.obs_stop,
        ):
            nn.init.zeros_(module.weight)

    def losses(self, history: torch.Tensor, observation: torch.Tensor,
               log_iei: torch.Tensor, participation: torch.Tensor,
               rank: torch.Tensor, stop_fraction: torch.Tensor) -> dict[str, torch.Tensor]:
        time_mu = (
            self.baseline.time_mean(history) + self.obs_time(observation)
        ).squeeze(-1)
        timing = (
            0.5 * ((log_iei - time_mu) / self.baseline.time_sigma) ** 2
            + self.baseline.time_sigma.log() + 0.5 * math.log(2 * math.pi)
        )
        logits = (
            self.baseline.participation(history)
            + self.obs_participation(observation)
        )
        part = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, participation, reduction="none"
        ).mean(-1)
        rank_mu = self.baseline.rank(history) + self.obs_rank(observation)
        rank_terms = (
            0.5 * ((rank - rank_mu) / self.baseline.rank_sigma) ** 2
            + self.baseline.rank_sigma.log() + 0.5 * math.log(2 * math.pi)
        )
        rank_nll = (
            (rank_terms * participation).sum(-1)
            / participation.sum(-1).clamp(min=1)
        )
        stop_mu = (
            self.baseline.stop(history) + self.obs_stop(observation)
        ).squeeze(-1)
        stop_nll = (
            0.5 * ((stop_fraction - stop_mu) / self.baseline.stop_sigma) ** 2
            + self.baseline.stop_sigma.log() + 0.5 * math.log(2 * math.pi)
        )
        mark = part + rank_nll + stop_nll
        return {
            "timing_nll": timing, "participation_nll": part,
            "rank_nll": rank_nll, "stop_nll": stop_nll,
            "mark_nll": mark, "joint_nll": timing + mark,
            "time_mu": time_mu, "participation_prob": logits.sigmoid(),
            "rank_mu": rank_mu, "stop_mu": stop_mu,
        }


def fit_bridge_arm(arrays: BridgeArrays, arm: str, seed: int = 0,
                   epochs: int = 300, lr: float = 3e-3,
                   weight_decay: float | None = None) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    matrix, transform = _arm_matrix(arrays, arm)
    split_at = int(transform["history_dim"])
    history = torch.as_tensor(matrix[:, :split_at], dtype=torch.float32)
    observation = torch.as_tensor(matrix[:, split_at:], dtype=torch.float32)
    y_time = torch.as_tensor(arrays.log_next_iei, dtype=torch.float32)
    y_part = torch.as_tensor(arrays.participation, dtype=torch.float32)
    y_rank = torch.as_tensor(arrays.rank, dtype=torch.float32)
    y_stop = torch.as_tensor(arrays.stop_fraction, dtype=torch.float32)
    train_idx = torch.as_tensor(np.flatnonzero(arrays.split == 0), dtype=torch.long)
    valid_idx = torch.as_tensor(np.flatnonzero(arrays.split == 1), dtype=torch.long)
    time_sigma = float(y_time[train_idx].std(unbiased=False).clamp(min=0.25))
    participating_rank = y_rank[train_idx][y_part[train_idx] > 0.5]
    rank_sigma = float(participating_rank.std(unbiased=False).clamp(min=0.20))
    stop_sigma = float(y_stop[train_idx].std(unbiased=False).clamp(min=0.10))
    def new_baseline() -> BridgeHead:
        candidate = BridgeHead(
            history.shape[1], y_part.shape[1], time_sigma=time_sigma,
            rank_sigma=rank_sigma, stop_sigma=stop_sigma,
        )
        for module in candidate.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
        return candidate

    def clone_baseline(source: BridgeHead) -> BridgeHead:
        candidate = new_baseline()
        candidate.load_state_dict(source.state_dict(), strict=True)
        return candidate

    def evaluate_baseline(model: BridgeHead, index: torch.Tensor) -> dict:
        with torch.no_grad():
            out = model.losses(
                history[index], y_time[index], y_part[index],
                y_rank[index], y_stop[index],
            )
        return metric_dict(out, index)

    def evaluate_residual(model: FrozenHistoryResidualHead,
                          index: torch.Tensor) -> dict:
        with torch.no_grad():
            out = model.losses(
                history[index], observation[index], y_time[index],
                y_part[index], y_rank[index], y_stop[index],
            )
        return metric_dict(out, index)

    def metric_dict(out: dict[str, torch.Tensor], index: torch.Tensor) -> dict:
        metrics = {k: float(out[k].mean()) for k in (
            "joint_nll", "timing_nll", "mark_nll", "participation_nll",
            "rank_nll", "stop_nll"
        )}
        pred_seconds = out["time_mu"].exp()
        true_seconds = y_time[index].exp()
        metrics["next_iei_mae_seconds"] = float((pred_seconds - true_seconds).abs().mean())
        metrics["participation_accuracy"] = float(
            ((out["participation_prob"] >= 0.5) == (y_part[index] >= 0.5)).float().mean()
        )
        return metrics

    def optimise_baseline(model: BridgeHead, index: torch.Tensor,
                          penalty_weight: float, max_iter: int) -> int:
        # Every head is linear and every likelihood term is convex. A
        # deterministic full-batch LBFGS fit removes optimiser seed noise.
        optimizer = torch.optim.LBFGS(
            model.parameters(), lr=0.5, max_iter=int(max_iter), history_size=20,
            tolerance_grad=1e-7, tolerance_change=1e-9,
            line_search_fn="strong_wolfe",
        )
        calls = 0

        def closure():
            nonlocal calls
            calls += 1
            optimizer.zero_grad(set_to_none=True)
            losses = model.losses(
                history[index], y_time[index], y_part[index],
                y_rank[index], y_stop[index],
            )
            penalty = sum(
                p.square().sum() for p in model.parameters() if p.ndim >= 2
            )
            objective = losses["joint_nll"].mean() + penalty_weight * penalty
            objective.backward()
            return objective

        optimizer.step(closure)
        model.eval()
        return calls

    def optimise_residual(model: FrozenHistoryResidualHead,
                          index: torch.Tensor, penalty_weight: float,
                          max_iter: int) -> int:
        trainable = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.LBFGS(
            trainable, lr=0.5, max_iter=int(max_iter), history_size=20,
            tolerance_grad=1e-7, tolerance_change=1e-9,
            line_search_fn="strong_wolfe",
        )
        calls = 0

        def closure():
            nonlocal calls
            calls += 1
            optimizer.zero_grad(set_to_none=True)
            losses = model.losses(
                history[index], observation[index], y_time[index],
                y_part[index], y_rank[index], y_stop[index],
            )
            penalty = sum(parameter.square().sum() for parameter in trainable)
            objective = losses["joint_nll"].mean() + penalty_weight * penalty
            objective.backward()
            return objective

        optimizer.step(closure)
        model.eval()
        return calls

    # Observation covariates can be numerous relative to the small Yuquan
    # pilots. A single arbitrary ridge coefficient made those arms explode on
    # later time. Select the one common coefficient using only the final 20%
    # of TRAIN, then refit all TRAIN once; development validation is untouched.
    regularization_grid = (
        (float(weight_decay),) if weight_decay is not None
        else (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
    )
    cut = max(1, min(len(train_idx) - 1, int(math.floor(0.80 * len(train_idx)))))
    inner_train_idx = train_idx[:cut]
    inner_valid_idx = train_idx[cut:]
    baseline_selection = []
    selection_iter = max(80, int(math.ceil(epochs / 2)))
    for candidate_weight in regularization_grid:
        candidate = new_baseline()
        calls = optimise_baseline(
            candidate, inner_train_idx, candidate_weight, selection_iter
        )
        score = evaluate_baseline(candidate, inner_valid_idx)["joint_nll"]
        baseline_selection.append({
            "weight_decay": float(candidate_weight),
            "inner_validation_joint_nll": float(score),
            "closure_calls": int(calls),
        })
    selected_weight = min(
        baseline_selection, key=lambda row: row["inner_validation_joint_nll"]
    )["weight_decay"]
    inner_baseline = new_baseline()
    optimise_baseline(
        inner_baseline, inner_train_idx, selected_weight, int(epochs)
    )
    final_baseline = new_baseline()
    baseline_closure_calls = optimise_baseline(
        final_baseline, train_idx, selected_weight, int(epochs)
    )

    residual_selection = []
    if arm == "b0_history":
        selected_residual_weight = None
        residual_closure_calls = 0
        model = FrozenHistoryResidualHead(
            final_baseline, contract.OBSERVATION_DIM
        )
    else:
        for candidate_weight in regularization_grid:
            candidate = FrozenHistoryResidualHead(
                clone_baseline(inner_baseline), contract.OBSERVATION_DIM
            )
            calls = optimise_residual(
                candidate, inner_train_idx, candidate_weight, selection_iter
            )
            score = evaluate_residual(candidate, inner_valid_idx)["joint_nll"]
            residual_selection.append({
                "weight_decay": float(candidate_weight),
                "inner_validation_joint_nll": float(score),
                "closure_calls": int(calls),
            })
        selected_residual_weight = min(
            residual_selection,
            key=lambda row: row["inner_validation_joint_nll"],
        )["weight_decay"]
        model = FrozenHistoryResidualHead(
            final_baseline, contract.OBSERVATION_DIM
        )
        residual_closure_calls = optimise_residual(
            model, train_idx, selected_residual_weight, int(epochs)
        )

    n_parameters = sum(p.numel() for p in model.parameters())
    train_metrics = evaluate_residual(model, train_idx)
    validation_metrics = evaluate_residual(model, valid_idx)
    baseline_train_metrics = evaluate_baseline(final_baseline, train_idx)
    baseline_validation_metrics = evaluate_baseline(final_baseline, valid_idx)
    if train_metrics["joint_nll"] > baseline_train_metrics["joint_nll"] + 1e-6:
        raise ValueError(
            f"{arrays.subject} {arm}: residual fit worsened TRAIN NLL despite "
            "the exact zero-effect initialization"
        )
    return {
        "contract": contract.REVISION,
        "fit_revision": contract.FIT_REVISION,
        "subject": arrays.subject,
        "arm": arm,
        "seed": int(seed),
        "optimizer": "deterministic_full_batch_lbfgs",
        "seed_semantics": (
            "audit label only; zero-initialised convex full-batch fit is "
            "expected to be seed invariant"
        ),
        "best_epoch": None,
        "optimizer_max_iter": int(epochs),
        "closure_calls": int(baseline_closure_calls + residual_closure_calls),
        "regularization_selection": {
            "semantics": (
                "history baseline and observation residual selected separately "
                "on chronological final 20% of TRAIN; development validation untouched"
            ),
            "n_inner_train": int(len(inner_train_idx)),
            "n_inner_validation": int(len(inner_valid_idx)),
            "history": {
                "selected_weight_decay": float(selected_weight),
                "grid": baseline_selection,
            },
            "observation_residual": {
                "selected_weight_decay": (
                    float(selected_residual_weight)
                    if selected_residual_weight is not None else None
                ),
                "grid": residual_selection,
            },
        },
        "n_parameters": int(n_parameters),
        "n_train": int(len(train_idx)),
        "n_validation": int(len(valid_idx)),
        "feature_transform": transform,
        "fixed_train_target_scales": {
            "time_sigma": time_sigma, "rank_sigma": rank_sigma,
            "stop_sigma": stop_sigma,
        },
        "shared_frozen_history_baseline": {
            "train": baseline_train_metrics,
            "validation": baseline_validation_metrics,
        },
        "train": train_metrics,
        "validation": validation_metrics,
        "sealed_opened": False,
        "claim_boundary": (
            "Bridge-E0 information diagnostic with a shared frozen history "
            "baseline; not a persistent-state result"
        ),
    }
