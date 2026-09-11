"""Event token features ``X_e`` and TRAIN-only standardisation (design §2).

Every column is a function of the event itself (and of static montage
geometry).  The inter-event interval is *not* a column: physical time enters the
model only through the state decay.  Raw waveform, background SEEG and any
seizure label are excluded from the primary token by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.topic5_group_event_state.dataset import SubjectSequence
from src.topic5_group_event_state.v02.marks import BAND_FEATURE_ENERGY, BAND_FEATURE_PEAK
from src.topic5_group_event_state.v02.subject import load_subject_timeline

from .paths import DATASET_ROOT, SESSION_INVENTORY, atomic_write_npz, file_hash

FEATURE_VERSION = "v032_event_token_1"
BAND_FEATURE_LOG_PEAK_AMPLITUDE = 4  # ``log_peak_amplitude`` in the frozen band_feature_names
CONFIDENCE_BAND_INDEX = 0            # ``ied_low`` carries the detector's own band


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean over axis 1 of ``values`` where ``mask`` (broadcastable) holds; NaN if empty."""

    ok = mask & np.isfinite(values)
    total = np.where(ok, values, 0.0).sum(axis=1)
    count = ok.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _masked_max(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ok = mask & np.isfinite(values)
    filled = np.where(ok, values, -np.inf).max(axis=1)
    return np.where(np.isfinite(filled), filled, np.nan)


def event_token_features(
    *,
    participation: np.ndarray,
    relative_delay: np.ndarray,
    tied_group_id: np.ndarray,
    band_features: np.ndarray,
    cross_band_lag: np.ndarray,
    contact_valid: np.ndarray,
    coords: np.ndarray | None,
    core_seconds: np.ndarray,
    has_waveform: np.ndarray,
    band_available: Sequence[bool],
    band_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Per-event token matrix ``(n_events, D)`` and its column names."""

    part = np.asarray(participation, dtype=bool) & np.asarray(contact_valid, dtype=bool)[None, :]
    n, c = part.shape
    part_f = part.astype(np.float64)
    n_part = part_f.sum(axis=1)
    n_valid = float(np.asarray(contact_valid, dtype=bool).sum())
    tied = np.asarray(tied_group_id, dtype=np.int64)
    delay = np.asarray(relative_delay, dtype=np.float64)
    bands = list(band_names) if band_names is not None else [f"{i}" for i in range(band_features.shape[2])]
    available = [bool(v) for v in band_available]
    columns: list[np.ndarray] = []
    names: list[str] = []

    def add(block: np.ndarray, labels: Sequence[str]) -> None:
        block = np.asarray(block, dtype=np.float64)
        if block.ndim == 1:
            block = block[:, None]
        if block.shape[1] != len(labels):
            raise ValueError(f"{len(labels)} labels for {block.shape[1]} columns")
        columns.append(block)
        names.extend(labels)

    # --- participation / tied-group summary ---------------------------------------
    add(part_f, [f"participation[{i}]" for i in range(c)])
    leader = (tied == 0) & part
    add(leader.astype(np.float64), [f"leader[{i}]" for i in range(c)])
    add(n_part / max(n_valid, 1.0), ["extent_fraction"])
    add(np.log1p(n_part), ["extent_log1p_n"])
    tied_masked = np.where(part, tied, -1)
    n_groups = tied_masked.max(axis=1) + 1
    safe_n = np.maximum(n_part, 1.0)
    first_size = leader.sum(axis=1)
    max_group = np.zeros(n)
    mean_group = np.zeros(n)
    for e in range(n):
        ids = tied_masked[e][tied_masked[e] >= 0]
        if ids.size:
            sizes = np.bincount(ids)
            sizes = sizes[sizes > 0]
            max_group[e] = sizes.max()
            mean_group[e] = sizes.mean()
    add(np.where(n_part > 0, n_groups / safe_n, 0.0), ["tied_n_groups_per_participant"])
    add(first_size / safe_n, ["tied_first_group_fraction"])
    add(max_group / safe_n, ["tied_largest_group_fraction"])
    add(mean_group / safe_n, ["tied_mean_group_fraction"])

    # --- exact delay ---------------------------------------------------------------
    delay_masked = np.where(part & np.isfinite(delay), delay, np.nan)
    with np.errstate(all="ignore"):
        span = np.nanmax(delay_masked, axis=1) - np.nanmin(delay_masked, axis=1)
        mean_delay = np.nanmean(delay_masked, axis=1)
        std_delay = np.nanstd(delay_masked, axis=1)
        median_delay = np.nanmedian(delay_masked, axis=1)
    multi = n_part >= 2
    add(np.where(multi, np.nan_to_num(span), 0.0), ["delay_span_s"])
    add(np.where(multi, np.nan_to_num(mean_delay), 0.0), ["delay_mean_s"])
    add(np.where(multi, np.nan_to_num(std_delay), 0.0), ["delay_std_s"])
    add(np.where(multi, np.nan_to_num(median_delay), 0.0), ["delay_median_s"])

    # --- spatial dispersion (static geometry) -------------------------------------
    if coords is not None and np.isfinite(coords).all() and coords.shape == (c, 3):
        xyz = np.asarray(coords, dtype=np.float64)
        pairwise = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
        quad = np.einsum("nc,cd,nd->n", part_f, pairwise, part_f)
        mean_pair = np.where(multi, quad / np.maximum(n_part * (n_part - 1.0), 1.0), 0.0)
        centroid = (part_f @ xyz) / safe_n[:, None]
        sq = (part_f @ (xyz ** 2).sum(axis=1)) / safe_n - (centroid ** 2).sum(axis=1)
        rms = np.where(multi, np.sqrt(np.clip(sq, 0.0, None)), 0.0)
        leader_f = leader.astype(np.float64)
        leader_centroid = (leader_f @ xyz) / np.maximum(leader_f.sum(axis=1), 1.0)[:, None]
        leader_dist = np.where(multi, np.linalg.norm(leader_centroid - centroid, axis=1), 0.0)
    else:
        mean_pair = rms = leader_dist = np.zeros(n)
    add(mean_pair, ["dispersion_mean_pairwise_mm"])
    add(rms, ["dispersion_rms_radius_mm"])
    add(leader_dist, ["dispersion_leader_to_centroid_mm"])
    add((~multi).astype(np.float64), ["dispersion_single_participant_flag"])

    # --- multiband summary ---------------------------------------------------------
    bf = np.asarray(band_features, dtype=np.float64)
    for b, band in enumerate(bands):
        if available[b]:
            energy = _masked_mean(bf[:, :, b, BAND_FEATURE_ENERGY], part)
            peak_amp = _masked_mean(bf[:, :, b, BAND_FEATURE_LOG_PEAK_AMPLITUDE], part)
            peak_amp_max = _masked_max(bf[:, :, b, BAND_FEATURE_LOG_PEAK_AMPLITUDE], part)
            peak_time = _masked_mean(bf[:, :, b, BAND_FEATURE_PEAK], part)
        else:
            energy = peak_amp = peak_amp_max = peak_time = np.zeros(n)
        add(energy, [f"band_{band}_mean_log_energy"])
        add(peak_amp, [f"band_{band}_mean_log_peak_amp"])
        add(peak_amp_max, [f"band_{band}_max_log_peak_amp"])
        add(peak_time, [f"band_{band}_mean_peak_time_s"])
    lag = np.asarray(cross_band_lag, dtype=np.float64)
    for p in range(lag.shape[2]):
        add(_masked_mean(lag[:, :, p], part), [f"crossband_{p}_mean_lag_s"])

    # --- detector-confidence proxies ------------------------------------------------
    cb = CONFIDENCE_BAND_INDEX
    conf_max = (
        _masked_max(bf[:, :, cb, BAND_FEATURE_LOG_PEAK_AMPLITUDE], part) if available[cb] else np.zeros(n)
    )
    conf_energy = _masked_mean(bf[:, :, cb, BAND_FEATURE_ENERGY], part) if available[cb] else np.zeros(n)
    add(conf_max, ["confidence_max_log_peak_amp_band0"])
    add(conf_energy, ["confidence_mean_log_energy_band0"])
    add(np.asarray(core_seconds, dtype=np.float64), ["confidence_core_seconds"])
    add(np.asarray(has_waveform, dtype=np.float64), ["confidence_has_waveform"])

    # --- coverage / reference -------------------------------------------------------
    add(np.full(n, n_valid / max(c, 1)), ["coverage_valid_fraction"])
    add(np.full(n, n_valid), ["coverage_n_valid"])

    x = np.concatenate(columns, axis=1).astype(np.float32)
    return x, tuple(names)


@dataclass
class TrainStandardizer:
    """Per-column mean/scale estimated on TRAIN rows only, frozen afterwards."""

    mean: np.ndarray
    scale: np.ndarray
    zero_variance: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, train_mask: np.ndarray) -> "TrainStandardizer":
        rows = np.asarray(x, dtype=np.float64)[np.asarray(train_mask, dtype=bool)]
        if rows.shape[0] == 0:
            raise ValueError("standardizer needs at least one TRAIN row")
        with np.errstate(all="ignore"):
            mean = np.nanmean(rows, axis=0)
            scale = np.nanstd(rows, axis=0)
        degenerate = ~np.isfinite(mean) | ~np.isfinite(scale) | (scale <= 1e-9)
        mean = np.where(degenerate, 0.0, mean)
        scale = np.where(degenerate, 1.0, scale)
        return cls(mean=mean, scale=scale, zero_variance=degenerate)

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        z[:, self.zero_variance] = 0.0
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        return z.astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "zero_variance": self.zero_variance.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainStandardizer":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            zero_variance=np.asarray(payload["zero_variance"], dtype=bool),
        )


def _coords(seq: SubjectSequence) -> np.ndarray | None:
    path = Path(seq.root) / "coords.npy"
    if not path.exists():
        return None
    coords = np.load(path)
    if coords.shape != (int(seq.index["n_contacts"]), 3) or not np.isfinite(coords).all():
        return None
    return coords.astype(np.float64)


def feature_fingerprint(subject: str, *, dataset_root: Path = DATASET_ROOT) -> dict[str, Any]:
    root = Path(dataset_root) / subject
    return {
        "feature_version": FEATURE_VERSION,
        "dataset_index_sha256": file_hash(root / "index.json"),
        "dataset_scalars_sha256": file_hash(root / "scalars.npz"),
        "session_inventory_sha256": file_hash(SESSION_INVENTORY),
    }


def build_subject_features(
    subject: str,
    *,
    dataset_root: Path = DATASET_ROOT,
    out_root: Path,
    overwrite: bool = False,
) -> Path:
    """Compute and cache the raw (unstandardised) token matrix for one patient."""

    out_root = Path(out_root)
    out = out_root / f"{subject}.npz"
    meta_path = out_root / f"{subject}.json"
    fingerprint = feature_fingerprint(subject, dataset_root=dataset_root)
    if out.exists() and meta_path.exists() and not overwrite:
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") == fingerprint:
            return out
    seq = SubjectSequence(Path(dataset_root) / subject)
    timeline = load_subject_timeline(subject)
    idx = np.asarray(seq.order[timeline.stream_positions], dtype=np.int64)
    arrays = seq.arrays
    x, names = event_token_features(
        participation=np.asarray(arrays["participation"][idx]),
        relative_delay=np.asarray(arrays["relative_delay"][idx]),
        tied_group_id=np.asarray(arrays["tied_group_id"][idx]),
        band_features=np.asarray(arrays["band_features"][idx]),
        cross_band_lag=np.asarray(arrays["cross_band_lag"][idx]),
        contact_valid=np.asarray(seq.contact_valid, dtype=bool),
        coords=_coords(seq),
        core_seconds=np.asarray(seq.scalars["core_seconds"][idx]),
        has_waveform=np.asarray(seq.scalars["has_waveform"][idx]),
        band_available=tuple(bool(b) for b in seq.index["band_available"]),
        band_names=tuple(str(b) for b in seq.index["bands"]),
    )
    atomic_write_npz(
        out,
        {
            "x_raw": x,
            "names": np.asarray(names, dtype=object),
            "event_times": timeline.event_times,
            "event_segment": timeline.event_segment,
            "stream_positions": timeline.stream_positions,
        },
    )
    from .paths import atomic_write_json

    atomic_write_json(
        meta_path,
        {
            "subject": subject,
            "n_events": int(x.shape[0]),
            "n_features": int(x.shape[1]),
            "names": list(names),
            "fingerprint": fingerprint,
            "excludes": ["raw_waveform", "background_seeg", "seizure_labels", "inter_event_interval"],
        },
    )
    return out
