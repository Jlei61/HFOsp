"""One interictal group event -> the mark that a future block has to predict.

The mark is deliberately split into the four families the plan names separately
(SP A2), because a state that only sharpens ``size`` is an extent model and a
state that only sharpens ``participation`` is a spatial model, and the numbers
have to be able to say which:

``participation``   which contacts took part               (C Bernoullis)
``size`` / ``span`` how far recruitment got before it stopped   (extent / STOP)
``band_energy`` / ``band_peak``  the multiband expression   (per available band)
``embedding``       a K-free repertoire coordinate          (TRAIN-frozen PCA)

The embedding exists because the alternative repertoire target -- KMeans cluster
identity -- makes every downstream conclusion depend on K and on the seed.  A
continuous coordinate has no such knob (CC 8).

Everything here is estimated on TRAIN positions only: the PCA basis, the
per-dimension centring and the scaling.  ``train_positions`` is a required
keyword with no default, because a default would silently restore the leak.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


# Index of ``log_integrated_energy`` and ``peak_time_s`` inside the frozen
# ``band_feature_names`` of the v0.1 shard (DC 9).
BAND_FEATURE_ENERGY = 2
BAND_FEATURE_PEAK = 0

DEFAULT_EMBEDDING_COMPONENTS = 8
MAX_EMBEDDING_FIT_EVENTS = 50_000


@dataclass(frozen=True)
class MarkEmbeddingSpec:
    """The TRAIN-frozen linear map from raw event features to the repertoire axis."""

    feature_mean: np.ndarray        # (F,)
    feature_scale: np.ndarray       # (F,)
    components: np.ndarray          # (d, F)
    embedding_mean: np.ndarray      # (d,)
    embedding_scale: np.ndarray     # (d,)
    explained_variance_ratio: np.ndarray
    n_fit_events: int
    n_components: int

    def transform(self, features: np.ndarray) -> np.ndarray:
        z = (np.asarray(features, dtype=np.float64) - self.feature_mean) / self.feature_scale
        return (z @ self.components.T - self.embedding_mean) / self.embedding_scale


@dataclass(frozen=True)
class EventMarks:
    """Per-event mark, in the exact form the future-block target consumes."""

    participation: np.ndarray       # (N, C) bool
    continuous: np.ndarray          # (N, D) float64, TRAIN-standardised
    valid: np.ndarray               # (N,) bool -- every continuous dim finite
    block_slices: dict[str, slice]
    continuous_names: tuple[str, ...]
    band_names_available: tuple[str, ...]
    embedding_spec: MarkEmbeddingSpec
    continuous_mean: np.ndarray     # (D,) TRAIN mean, before standardising
    continuous_scale: np.ndarray    # (D,) TRAIN scale, before standardising
    n_span_imputed: int = 0         # events whose recruitment span had to be set to 0

    @property
    def n_events(self) -> int:
        return int(self.participation.shape[0])

    @property
    def n_contacts(self) -> int:
        return int(self.participation.shape[1])

    @property
    def n_continuous(self) -> int:
        return int(self.continuous.shape[1])


def _raw_features(
    participation: np.ndarray,
    relative_delay: np.ndarray,
    band_features: np.ndarray,
    band_keep: np.ndarray,
) -> np.ndarray:
    """Per-event feature matrix the repertoire PCA is fitted on.

    ``[participation (C) | masked delay (C) | masked per-contact band energy (C*Ba)]``
    -- i.e. who took part, when they took part, and with what spectral content.
    Non-participants contribute a structural zero in the delay and energy blocks;
    that zero means "did not take part", which the participation block states
    explicitly alongside, so it is not an imputed value.
    """

    part = np.asarray(participation, dtype=bool)
    n, c = part.shape
    delay = np.nan_to_num(np.asarray(relative_delay, dtype=np.float64)) * part
    energy = np.asarray(band_features, dtype=np.float64)[:, :, band_keep, BAND_FEATURE_ENERGY]
    energy = np.nan_to_num(energy) * part[:, :, None]
    return np.concatenate(
        [part.astype(np.float64), delay, energy.reshape(n, -1)], axis=1
    )


def _masked_band_mean(
    band_features: np.ndarray, participation: np.ndarray, band_keep: np.ndarray, feature: int
) -> np.ndarray:
    """Mean over *participating* contacts, per available band; NaN if none finite."""

    values = np.asarray(band_features, dtype=np.float64)[:, :, band_keep, feature]
    mask = np.asarray(participation, dtype=bool)[:, :, None] & np.isfinite(values)
    total = np.where(mask, values, 0.0).sum(axis=1)
    count = mask.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return out


def _continuous_blocks(
    participation: np.ndarray,
    relative_delay: np.ndarray,
    band_features: np.ndarray,
    band_keep: np.ndarray,
) -> tuple[np.ndarray, list[tuple[str, int]], int]:
    part = np.asarray(participation, dtype=bool)
    size = part.sum(axis=1).astype(np.float64)[:, None]
    delay = np.asarray(relative_delay, dtype=np.float64)
    # An event whose participants all carry a non-finite delay would get span 0
    # here rather than being flagged out of the conditional-mark score, which is
    # the one imputation left in this file.  Measured incidence in this cohort is
    # 0.000% (participant delays are 100% finite, DC 3), and the count is carried
    # in the summary so a future cohort cannot inherit the assumption silently.
    with np.errstate(all="ignore"):
        span = np.nanmax(np.where(part, delay, np.nan), axis=1)[:, None]
    n_span_imputed = int((~np.isfinite(span)).sum())
    span = np.nan_to_num(span, nan=0.0)
    energy = _masked_band_mean(band_features, part, band_keep, BAND_FEATURE_ENERGY)
    peak = _masked_band_mean(band_features, part, band_keep, BAND_FEATURE_PEAK)
    blocks = np.concatenate([size, span, energy, peak], axis=1)
    layout = [("size", 1), ("span", 1), ("band_energy", energy.shape[1]),
              ("band_peak", peak.shape[1])]
    return blocks, layout, n_span_imputed


def fit_mark_embedding(
    features: np.ndarray,
    *,
    train_positions: np.ndarray,
    n_components: int = DEFAULT_EMBEDDING_COMPONENTS,
    seed: int = 0,
) -> MarkEmbeddingSpec:
    """PCA basis of the repertoire, estimated on TRAIN rows only (clause C7)."""

    pos = np.asarray(train_positions, dtype=np.int64)
    if pos.size == 0:
        raise ValueError("mark embedding needs at least one TRAIN event")
    rng = np.random.default_rng(seed)
    if pos.size > MAX_EMBEDDING_FIT_EVENTS:
        pos = np.sort(rng.choice(pos, size=MAX_EMBEDDING_FIT_EVENTS, replace=False))
    x = np.asarray(features, dtype=np.float64)[pos]
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    z = (x - mean) / scale
    d = int(min(n_components, z.shape[1], max(z.shape[0] - 1, 1)))
    _u, s, vt = np.linalg.svd(z, full_matrices=False)
    components = vt[:d]
    var = (s ** 2) / max(z.shape[0] - 1, 1)
    ratio = var[:d] / max(float(var.sum()), 1e-12)
    emb = z @ components.T
    emb_mean = emb.mean(axis=0)
    emb_scale = emb.std(axis=0)
    emb_scale = np.where(emb_scale > 1e-9, emb_scale, 1.0)
    return MarkEmbeddingSpec(
        feature_mean=mean,
        feature_scale=scale,
        components=components,
        embedding_mean=emb_mean,
        embedding_scale=emb_scale,
        explained_variance_ratio=ratio,
        n_fit_events=int(pos.size),
        n_components=d,
    )


def build_event_marks(
    participation: np.ndarray,
    relative_delay: np.ndarray,
    band_features: np.ndarray,
    *,
    band_available: Sequence[bool],
    band_names: Sequence[str],
    train_positions: np.ndarray,
    n_components: int = DEFAULT_EMBEDDING_COMPONENTS,
    seed: int = 0,
) -> EventMarks:
    """Assemble the four mark families and freeze their TRAIN standardisation."""

    band_keep = np.flatnonzero(np.asarray(band_available, dtype=bool))
    if band_keep.size == 0:
        raise ValueError("no supported frequency band for this patient")
    kept_names = tuple(str(band_names[i]) for i in band_keep)

    blocks, layout, n_span_imputed = _continuous_blocks(
        participation, relative_delay, band_features, band_keep
    )
    features = _raw_features(participation, relative_delay, band_features, band_keep)
    spec = fit_mark_embedding(
        features, train_positions=train_positions, n_components=n_components, seed=seed
    )
    embedding = spec.transform(features)
    raw = np.concatenate([blocks, embedding], axis=1)
    layout = layout + [("embedding", embedding.shape[1])]

    pos = np.asarray(train_positions, dtype=np.int64)
    train_rows = raw[pos]
    finite_train = np.isfinite(train_rows).all(axis=1)
    if not finite_train.any():
        raise ValueError("every TRAIN event has a non-finite mark")
    mean = train_rows[finite_train].mean(axis=0)
    scale = train_rows[finite_train].std(axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    standardised = (raw - mean) / scale

    valid = np.isfinite(standardised).all(axis=1)
    # C6: a non-finite mark is *flagged out of the conditional-mark score*, never
    # replaced by a plausible number.  The event still counts towards p(N).
    standardised = np.where(valid[:, None], standardised, 0.0)

    slices: dict[str, slice] = {}
    names: list[str] = []
    cursor = 0
    for block_name, width in layout:
        slices[block_name] = slice(cursor, cursor + width)
        if block_name in ("band_energy", "band_peak"):
            names += [f"{block_name}:{b}" for b in kept_names]
        elif width == 1:
            names.append(block_name)
        else:
            names += [f"{block_name}:{i}" for i in range(width)]
        cursor += width

    return EventMarks(
        n_span_imputed=n_span_imputed,
        participation=np.asarray(participation, dtype=bool),
        continuous=standardised,
        valid=valid,
        block_slices=slices,
        continuous_names=tuple(names),
        band_names_available=kept_names,
        embedding_spec=spec,
        continuous_mean=mean,
        continuous_scale=scale,
    )


def apply_event_marks(
    participation: np.ndarray,
    relative_delay: np.ndarray,
    band_features: np.ndarray,
    spec: MarkEmbeddingSpec,
    reference: EventMarks,
) -> np.ndarray:
    """Re-embed rows with an already-frozen spec (used to prove C7 in tests)."""

    band_keep = np.flatnonzero(
        np.isin(
            np.arange(band_features.shape[2]),
            _band_keep_from_reference(reference, band_features.shape[2]),
        )
    )
    blocks, _layout, _n_imputed = _continuous_blocks(
        participation, relative_delay, band_features, band_keep
    )
    features = _raw_features(participation, relative_delay, band_features, band_keep)
    raw = np.concatenate([blocks, spec.transform(features)], axis=1)
    return (raw - reference.continuous_mean) / reference.continuous_scale


def _band_keep_from_reference(reference: EventMarks, n_bands_total: int) -> np.ndarray:
    width = len(reference.band_names_available)
    if width == n_bands_total:
        return np.arange(n_bands_total)
    raise ValueError(
        "re-embedding a different band set than the reference was fitted on; "
        "pass the same band_available mask"
    )


def summarise(marks: EventMarks) -> dict[str, Any]:
    return {
        "n_events": marks.n_events,
        "n_contacts": marks.n_contacts,
        "n_continuous_dims": marks.n_continuous,
        "continuous_names": list(marks.continuous_names),
        "bands_available": list(marks.band_names_available),
        "fraction_valid_mark": float(marks.valid.mean()),
        "n_events_with_span_imputed_to_zero": int(marks.n_span_imputed),
        "embedding_components": int(marks.embedding_spec.n_components),
        "embedding_explained_variance_ratio": [
            float(v) for v in marks.embedding_spec.explained_variance_ratio
        ],
        "embedding_fit_events": int(marks.embedding_spec.n_fit_events),
    }
