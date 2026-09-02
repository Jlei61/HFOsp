"""Reading the frozen v0.1 event stream, and only the parts H3 is entitled to.

Deliberately *not* a subclass of ``SubjectSequence``.  That class answers "give me
the interictal stream split 70/10/20 by event count", and H3 asks a different
question -- "give me the interictal stream indexed by recorded wall-clock time" --
so reusing its splits would import an answer to the wrong question (CLAUDE.md
§6.1).  What is reused is exactly the part that is question-neutral: the
memory-mapped arrays and the ictal exclusion the v0.1 data contract froze.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .features import EventFeatures, build_event_features
from .io import write_npz_atomic


@dataclass
class SubjectStream:
    subject: str
    dataset: str
    t_abs: np.ndarray             # (n,) float64, interictal only, time-ordered
    features: EventFeatures
    n_contacts: int
    band_available: np.ndarray
    index: dict[str, Any]


def waveform_log_rms(
    waveform_path: Path,
    participation: np.ndarray,
    *,
    chunk: int = 2048,
) -> np.ndarray:
    """Per-event, per-view log RMS over the contacts that actually took part.

    Streamed in chunks: the widest patient's waveform array is 11 GB and holding
    it resident would evict everything else on a shared box.  Non-participants are
    excluded rather than averaged in -- their trace is background, and pooling it
    would dilute exactly the amplitude contrast ``M2`` is asked about.
    """

    array = np.load(waveform_path, mmap_mode="r")  # (n, C, V, T) float16
    n, n_contacts, n_views, _n_samples = array.shape
    if participation.shape != (n, n_contacts):
        raise ValueError(
            f"participation {participation.shape} does not match waveform {array.shape[:2]}"
        )
    out = np.zeros((n, n_views), dtype=np.float32)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        block = np.asarray(array[lo:hi], dtype=np.float32)
        mask = participation[lo:hi][:, :, None, None].astype(np.float32)
        power = np.nan_to_num(block) ** 2 * mask
        denom = mask.sum(axis=(1, 3)).clip(min=1.0)
        out[lo:hi] = np.log1p(np.sqrt(power.sum(axis=(1, 3)) / denom)).astype(np.float32)
    return out


def build_subject_features(dataset_dir: Path, waveform_cache: Path | None = None) -> EventFeatures:
    """Frozen count/mark vocabularies for one patient's interictal stream."""

    dataset_dir = Path(dataset_dir)
    index = json.loads((dataset_dir / "index.json").read_text())
    scalars = np.load(dataset_dir / "scalars.npz")
    order = scalars["interictal_index"]

    participation = np.asarray(np.load(dataset_dir / "participation.npy", mmap_mode="r")[order])
    if waveform_cache is not None and Path(waveform_cache).exists():
        rms = np.load(waveform_cache)["waveform_rms"]
    else:
        rms_all = waveform_log_rms(
            dataset_dir / "waveform.npy",
            np.asarray(np.load(dataset_dir / "participation.npy", mmap_mode="r")),
        )
        rms = rms_all[order]
        if waveform_cache is not None:
            write_npz_atomic(Path(waveform_cache), waveform_rms=rms)

    arrays = {
        "participation": participation,
        "relative_delay": np.asarray(np.load(dataset_dir / "relative_delay.npy", mmap_mode="r")[order]),
        "tied_group_id": np.asarray(np.load(dataset_dir / "tied_group_id.npy", mmap_mode="r")[order]),
        "band_features": np.asarray(np.load(dataset_dir / "band_features.npy", mmap_mode="r")[order]),
        "cross_band_lag": np.asarray(np.load(dataset_dir / "cross_band_lag.npy", mmap_mode="r")[order]),
        "waveform_rms": rms,
    }

    t_abs = scalars["t_abs"][order].astype(np.float64)
    # Session-relative interval; a session's first event gets NaN, which the
    # feature builder turns into an explicit "no previous interval" channel
    # rather than a zero that would claim the event followed instantly.
    session = scalars["session_of_event"][order]
    dt_prev = np.full(t_abs.size, np.nan, dtype=np.float64)
    if t_abs.size > 1:
        dt_prev[1:] = np.diff(t_abs)
        dt_prev[1:][session[1:] != session[:-1]] = np.nan
    dt_prev[0] = np.nan

    return build_event_features(arrays, t_abs, dt_prev, index["band_available"])


def load_stream(dataset_dir: Path, feature_cache: Path) -> SubjectStream:
    """Load a patient's frozen H3 vocabularies from cache, building nothing."""

    dataset_dir = Path(dataset_dir)
    index = json.loads((dataset_dir / "index.json").read_text())
    cache = np.load(Path(feature_cache))
    meta = json.loads(Path(str(feature_cache).replace(".npz", ".json")).read_text())
    features = EventFeatures(
        t_abs=cache["t_abs"].astype(np.float64),
        count_features=cache["count_features"],
        mark_features=cache["mark_features"],
        mark_group_slices={k: tuple(v) for k, v in meta["mark_group_slices"].items()},
        count_feature_names=meta["count_feature_names"],
        mark_feature_names=meta["mark_feature_names"],
        participation=cache["participation"].astype(bool),
        size=cache["size"].astype(np.float32),
        band_available=np.asarray(meta["band_available"], dtype=bool),
    )
    return SubjectStream(
        subject=index["subject"],
        dataset=index["dataset"],
        t_abs=features.t_abs,
        features=features,
        n_contacts=int(index["n_contacts"]),
        band_available=features.band_available,
        index=index,
    )
