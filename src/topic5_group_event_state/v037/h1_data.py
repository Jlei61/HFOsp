"""Causal H1 data assembly for the v0.3.7 shared observer.

The builder keeps event burden and conditional grammar separate.  Grammar is
not a legacy rank: it includes community use/coupling, continuous contact lag,
multiband expression, cross-band timing and compact summaries of the stored
bipolar/CAR event waveforms.  Every dictionary, residualisation and scale is
fitted before INNER/SELECTION.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.decomposition import PCA

from src.topic5_group_event_state.dataset import SubjectSequence
from src.topic5_group_event_state.v02.marks import build_event_marks, summarise
from src.topic5_group_event_state.v035.contracts import INPUT_ROOT, RateTrainConfig
from src.topic5_group_event_state.v035.dynamic_rate import RateData, load_rate_data
from src.topic5_group_event_state.v035.grammar_targets import (
    GrammarBlockTargets,
    GrammarDictionary,
    _event_grammar_arrays,
    aggregate_grammar_blocks,
    fit_grammar_dictionary,
)


@dataclass(frozen=True)
class H1SubjectData:
    subject: str
    rate: RateData
    event_time: np.ndarray
    event_segment: np.ndarray
    burden_mark: np.ndarray
    grammar_mark: np.ndarray
    grammar_dictionary: GrammarDictionary
    grammar_targets: GrammarBlockTargets
    grammar_mark_mean_target: np.ndarray
    grammar_mark_mean_valid: np.ndarray
    burden_mark_mean_target: np.ndarray
    burden_mark_mean_valid: np.ndarray
    representation_provenance: dict[str, Any]
    preprocessing_operators: dict[str, Any] = field(default_factory=dict)


def _event_rows_for_rate(subject: str, rate: RateData) -> tuple[SubjectSequence, np.ndarray, np.ndarray, np.ndarray]:
    manifest = json.loads((INPUT_ROOT / subject / "manifest_v3.json").read_text(encoding="utf-8"))
    with np.load(manifest["input_path"], allow_pickle=False) as z:
        source_time = np.asarray(z["event_time"], dtype=np.float64)
    keep = source_time < float(rate.phase_boundaries["80pct"])
    source_time = source_time[keep]
    segment = np.full(source_time.size, -1, dtype=np.int64)
    for lo, hi in rate.observed_support_bounds:
        parent = np.flatnonzero(
            (rate.segment_bounds[:, 0] <= float(lo) + 1e-9)
            & (rate.segment_bounds[:, 1] >= float(hi) - 1e-9)
        )
        if parent.size != 1:
            raise ValueError("observed support must map to one carry segment")
        inside = (source_time >= float(lo)) & (source_time < float(hi))
        if np.any(segment[inside] >= 0):
            raise ValueError("observed support pieces overlap")
        segment[inside] = int(parent[0])
    covered = segment >= 0
    source_time, segment = source_time[covered], segment[covered]
    sequence = SubjectSequence(Path("/data/hfosp_group_event_state_v0_1/dataset") / subject)
    source_position = np.searchsorted(sequence.t_abs, source_time)
    if np.any(source_position >= len(sequence)) or not np.array_equal(sequence.t_abs[source_position], source_time):
        raise ValueError("registered H1 events do not map exactly to the v0.1 stream")
    raw_rows = sequence.order[source_position]
    return sequence, source_time, segment, raw_rows


def _phase_rows(time: np.ndarray, bounds: dict[str, float]) -> np.ndarray:
    return np.flatnonzero(time < float(bounds["60pct"]))


def _community_mean(values: np.ndarray, participation: np.ndarray, community: np.ndarray) -> np.ndarray:
    """Mean contact feature in each community, preserving event order."""

    n, _c, width = values.shape
    k = int(community.max()) + 1
    out = np.zeros((n, k, width), dtype=np.float32)
    for group in range(k):
        contact = np.flatnonzero(community == group)
        mask = participation[:, contact, None] & np.isfinite(values[:, contact])
        count = mask.sum(axis=1)
        total = np.where(mask, values[:, contact], 0.0).sum(axis=1)
        out[:, group] = total / np.maximum(count, 1)
    return out.reshape(n, -1)


def _waveform_community_features(
    sequence: SubjectSequence,
    raw_rows: np.ndarray,
    participation: np.ndarray,
    community: np.ndarray,
    *,
    chunk_events: int = 1024,
) -> np.ndarray:
    n = raw_rows.size
    k = int(community.max()) + 1
    n_view = int(sequence.arrays["waveform"].shape[2])
    output = np.zeros((n, k * n_view * 3), dtype=np.float32)
    for start in range(0, n, int(chunk_events)):
        stop = min(n, start + int(chunk_events))
        wave = np.asarray(sequence.arrays["waveform"][raw_rows[start:stop]], dtype=np.float32)
        finite = np.isfinite(wave)
        clean = np.where(finite, wave, 0.0)
        denom = np.maximum(finite.sum(axis=-1), 1)
        rms = np.sqrt((clean.square().sum(axis=-1) if hasattr(clean, "square") else (clean * clean).sum(axis=-1)) / denom)
        peak = np.max(np.abs(clean), axis=-1)
        line = np.mean(np.abs(np.diff(clean, axis=-1)), axis=-1)
        feature = np.stack((rms, peak, line), axis=-1).reshape(stop - start, wave.shape[1], -1)
        output[start:stop] = _community_mean(
            feature, participation[start:stop], community
        )
    return output


def _rich_cache_signature(sequence, raw_rows, participation, tied, dictionary, fit_rows):
    """Hash the actual pre-80% rows and fitted dictionary consumed by this cache."""
    digest = hashlib.sha256(Path(__file__).read_bytes())
    def add(value):
        array = np.ascontiguousarray(value)
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    for value in (raw_rows, fit_rows, participation, tied, dictionary.community_of_contact,
                  dictionary.event_repertoire_embedding, dictionary.event_repertoire_label,
                  dictionary.repertoire_centres, np.asarray(sequence.index['band_available'])):
        add(value)
    digest.update(json.dumps({key: sequence.index.get(key) for key in ('bands', 'views', 'cross_band_pairs')}, sort_keys=True).encode())
    # Only selected interictal rows are fingerprinted; no development/sealed
    # waveform values are needed for cache validation.
    for name in ('band_features', 'cross_band_lag', 'waveform'):
        digest.update(name.encode())
        for start in range(0, len(raw_rows), 1024):
            add(sequence.arrays[name][raw_rows[start:start + 1024]])
    return digest.hexdigest()


def _rich_grammar_features(
    sequence: SubjectSequence,
    raw_rows: np.ndarray,
    participation: np.ndarray,
    tied: np.ndarray,
    dictionary: GrammarDictionary,
    fit_rows: np.ndarray,
    *,
    cache_path: Path | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    fingerprint = _rich_cache_signature(sequence, raw_rows, participation, tied, dictionary, fit_rows)
    if cache_path is not None:
        cache_path = cache_path.with_name(f'{cache_path.stem}.{fingerprint}.npz')
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as z:
            provenance = json.loads(str(z['provenance_json'].item()))
            value = np.asarray(z['grammar_mark'], dtype=np.float32)
            if provenance.get('selected_input_and_dictionary_sha256') != fingerprint or value.shape[0] != len(raw_rows):
                raise ValueError('rich grammar cache fingerprint or row alignment mismatch')
            return value, provenance
    occupancy, coupling, coupling_valid = _event_grammar_arrays(participation, tied, dictionary)
    coupling = coupling * coupling_valid[:, None]
    repertoire = np.zeros((participation.shape[0], dictionary.n_repertoires), dtype=np.float32)
    valid_rep = dictionary.event_repertoire_label >= 0
    repertoire[np.flatnonzero(valid_rep), dictionary.event_repertoire_label[valid_rep]] = 1.0

    available = np.flatnonzero(np.asarray(sequence.index["band_available"], dtype=bool))
    band = np.asarray(sequence.arrays["band_features"][raw_rows], dtype=np.float32)
    energy = band[:, :, available, 2]
    peak = band[:, :, available, 0]
    cross = np.asarray(sequence.arrays["cross_band_lag"][raw_rows], dtype=np.float32)
    community = dictionary.community_of_contact
    energy_c = _community_mean(energy, participation, community)
    peak_c = _community_mean(peak, participation, community)
    cross_c = _community_mean(cross, participation, community)
    waveform_c = _waveform_community_features(sequence, raw_rows, participation, community)
    raw = np.concatenate(
        (
            occupancy,
            coupling,
            repertoire,
            dictionary.event_repertoire_embedding,
            energy_c,
            peak_c,
            cross_c,
            waveform_c,
        ),
        axis=1,
    ).astype(np.float64)
    centre = np.nanmedian(raw[fit_rows], axis=0)
    scale = 1.4826 * np.nanmedian(np.abs(raw[fit_rows] - centre), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    standard = np.clip((np.where(np.isfinite(raw), raw, centre) - centre) / scale, -12.0, 12.0)
    provenance = {
        "format": "group_event_state_v0_3_7_rich_grammar_mark_v1",
        "inputs": [
            "contact participation",
            "continuous contact delay",
            "tied-group community coupling",
            "per-contact multiband energy and peak time",
            "per-contact cross-band lag",
            "stored event waveform RMS, peak and line length in every reference view",
        ],
        "spatial_aggregation": "patient-specific FIT-only contact communities",
        "normalization": "median/MAD on pre-INNER rows only",
        "n_features": int(standard.shape[1]),
        "n_fit_events": int(fit_rows.size),
        'selected_input_and_dictionary_sha256': fingerprint,
        'fitted_centre': centre.tolist(), 'fitted_scale': scale.tolist(),
        'cache_contract': 'actual selected arrays + row order + FIT dictionary + producer hash; no legacy fallback',
    }
    value = standard.astype(np.float32)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f'{cache_path.name}.{os.getpid()}.tmp.npz')
        np.savez_compressed(
            temporary,
            grammar_mark=value,
            provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
        )
        os.replace(temporary, cache_path)
    return value, provenance


def _residualise_grammar(
    grammar: np.ndarray,
    burden: np.ndarray,
    fit_rows: np.ndarray,
    *,
    n_components: int = 24,
    seed: int = 20260904,
) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.concatenate((np.ones((burden.shape[0], 1)), burden.astype(np.float64)), axis=1)
    xf = x[fit_rows]
    yf = grammar[fit_rows].astype(np.float64)
    penalty = np.eye(x.shape[1], dtype=np.float64) * 1e-3
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(xf.T @ xf + penalty, xf.T @ yf)
    residual = grammar.astype(np.float64) - x @ beta
    centre = np.median(residual[fit_rows], axis=0)
    scale = 1.4826 * np.median(np.abs(residual[fit_rows] - centre), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    residual = np.clip((residual - centre) / scale, -12.0, 12.0).astype(np.float32)
    fit_for_pca = fit_rows
    if fit_for_pca.size > 50_000:
        rng = np.random.default_rng(int(seed))
        fit_for_pca = np.sort(rng.choice(fit_for_pca, 50_000, replace=False))
    width = min(int(n_components), residual.shape[1], max(1, fit_for_pca.size - 1))
    pca = PCA(n_components=width, svd_solver="randomized", random_state=int(seed))
    pca.fit(residual[fit_for_pca])
    compressed = pca.transform(residual).astype(np.float32)
    return compressed, {
        "method": "FIT-only ridge residual of grammar on burden plus intercept",
        "ridge": 1e-3,
        "n_nuisance": int(x.shape[1]),
        "raw_grammar_width": int(grammar.shape[1]),
        "compressed_width": int(width),
        "compression": "FIT-only randomized PCA after burden residualisation",
        "pca_fit_events": int(fit_for_pca.size),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        'fitted_operator': {'ridge_coefficient': beta.tolist(), 'residual_centre': centre.tolist(),
                            'residual_scale': scale.tolist(), 'pca_mean': pca.mean_.tolist(),
                            'pca_components': pca.components_.tolist(), 'pca_fit_rows': fit_for_pca.tolist()},
    }


def _future_means(
    anchor_time: np.ndarray,
    horizons: Sequence[float],
    target_valid: np.ndarray,
    event_time: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.zeros((anchor_time.size, len(horizons), values.shape[1]), dtype=np.float32)
    valid = np.zeros((anchor_time.size, len(horizons)), dtype=bool)
    prefix = np.concatenate(
        (np.zeros((1, values.shape[1]), dtype=np.float64), np.cumsum(values, axis=0, dtype=np.float64)), axis=0
    )
    left = np.searchsorted(event_time, anchor_time, side="left")
    for j, horizon in enumerate(horizons):
        right = np.searchsorted(event_time, anchor_time + float(horizon), side="left")
        count = right - left
        use = target_valid[:, j] & (count > 0)
        target[use, j] = ((prefix[right[use]] - prefix[left[use]]) / count[use, None]).astype(np.float32)
        valid[:, j] = use
    return target, valid


def build_h1_subject_data(
    subject: str,
    *,
    horizons_seconds: Sequence[float] = (1800.0, 7200.0, 21600.0, 28800.0),
    taus_seconds: Sequence[float] = (600.0, 1800.0, 3600.0, 7200.0, 14400.0, 28800.0, 57600.0),
    cache_root: Path | None = Path('/data/hfosp_group_event_state_v0_3_8_review_repair/verified_event_representation'),
    seed: int = 20260904,
) -> H1SubjectData:
    config = RateTrainConfig(
        horizons_seconds=tuple(float(v) for v in horizons_seconds),
        taus_seconds=tuple(float(v) for v in taus_seconds),
        window_contract="observed_support",
        merge_artificial_cuts_seconds=600.0,
        minimum_exposure_fraction=0.8,
        split_contract="shared_multi_horizon_observed_time",
    )
    rate = load_rate_data(subject, config)
    sequence, event_time, event_segment, raw_rows = _event_rows_for_rate(subject, rate)
    part = np.asarray(sequence.arrays["participation"][raw_rows], dtype=bool)
    delay = np.asarray(sequence.arrays["relative_delay"][raw_rows], dtype=np.float32)
    tied = np.asarray(sequence.arrays["tied_group_id"][raw_rows], dtype=np.int16)
    band = np.asarray(sequence.arrays["band_features"][raw_rows], dtype=np.float32)
    fit_rows = _phase_rows(event_time, dict(rate.phase_boundaries))
    marks = build_event_marks(
        part,
        delay,
        band,
        band_available=sequence.index["band_available"],
        band_names=sequence.index["bands"],
        train_positions=fit_rows,
        n_components=8,
        seed=int(seed),
    )
    burden_columns = np.r_[
        np.arange(marks.block_slices["size"].start, marks.block_slices["span"].stop),
        np.arange(marks.block_slices["band_energy"].start, marks.block_slices["band_energy"].stop),
    ]
    burden = marks.continuous[:, burden_columns].astype(np.float32)
    dictionary = fit_grammar_dictionary(
        part,
        delay,
        tied,
        band,
        band_available=sequence.index["band_available"],
        band_names=sequence.index["bands"],
        fit_rows=fit_rows,
        seed=int(seed),
        requested_communities=4,
        requested_repertoires=6,
    )
    cache_path = None if cache_root is None else Path(cache_root) / subject / f"seed{seed}.npz"
    grammar_raw, rich_provenance = _rich_grammar_features(
        sequence, raw_rows, part, tied, dictionary, fit_rows, cache_path=cache_path
    )
    grammar, residual_provenance = _residualise_grammar(
        grammar_raw, burden, fit_rows, n_components=24, seed=int(seed)
    )
    targets = aggregate_grammar_blocks(
        grid_time=rate.anchor_time,
        horizons_seconds=rate.horizons_seconds,
        future_valid=rate.target_valid,
        event_time=event_time,
        participation=part,
        tied_group_id=tied,
        dictionary=dictionary,
    )
    grammar_mean, grammar_mean_valid = _future_means(
        rate.anchor_time, rate.horizons_seconds, rate.target_valid, event_time, grammar
    )
    burden_mean, burden_mean_valid = _future_means(
        rate.anchor_time, rate.horizons_seconds, rate.target_valid, event_time, burden
    )
    provenance = {
        "event_mark_summary": summarise(marks),
        "rich_grammar": rich_provenance,
        "burden_residualisation": residual_provenance,
        "dictionary_fit_rows_before_inner": True,
        "event_count": int(event_time.size),
        "event_contact_count": int(part.shape[1]),
        "maximum_source_time": float(event_time.max()),
        "inner_boundary": float(rate.phase_boundaries["60pct"]),
        "selection_not_used_for_representation_fit": True,
        "development_targets_read": False,
        "seizure_targets_read": False,
        "sealed_partition_opened": False,
    }
    return H1SubjectData(
        subject=subject,
        rate=rate,
        event_time=event_time,
        event_segment=event_segment,
        burden_mark=burden,
        grammar_mark=grammar,
        grammar_dictionary=dictionary,
        grammar_targets=targets,
        grammar_mark_mean_target=grammar_mean,
        grammar_mark_mean_valid=grammar_mean_valid,
        burden_mark_mean_target=burden_mean,
        burden_mark_mean_valid=burden_mean_valid,
        representation_provenance=provenance,
        preprocessing_operators={
            'event_mark_embedding': marks.embedding_spec,
            'event_mark_continuous_mean': marks.continuous_mean,
            'event_mark_continuous_scale': marks.continuous_scale,
            'burden_columns': burden_columns,
            'grammar_dictionary': dictionary,
            'rich_feature_normalization': rich_provenance,
            'burden_residualisation': residual_provenance,
            'source_raw_rows': raw_rows,
        },
    )
