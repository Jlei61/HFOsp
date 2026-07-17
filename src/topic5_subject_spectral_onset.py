"""Patient-specific, phenotype-gated refinement of spectral onset time.

This layer never creates an onset for an event without a sustained broadband
episode.  For eligible events it ranks multiple episode-linked change points
against a leave-one-seizure-out patient prototype.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Sequence

import numpy as np

from src.topic5_spectral_onset import (
    PreparedSpectralEvent,
    SpectralCalibration,
    SpectralDiagnostics,
    SpectralEpisode,
)


@dataclass(frozen=True)
class SubjectOnsetConfig:
    candidate_before_state_sec: float = 10.0
    candidate_after_state_sec: float = 3.0
    candidate_min_separation_sec: float = 0.5
    max_candidates_per_episode: int = 12
    min_prototype_events: int = 3
    prototype_coherence_threshold: float = 0.65
    prototype_weight: float = 0.65
    near_tie_score: float = 0.03
    trajectory_offsets_sec: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0)
    n_boot: int = 100
    consistency_tolerance_sec: float = 1.0
    temporal_min_other_events: int = 3
    temporal_min_supporting_events: int = 2
    temporal_neighbor_k: int = 2
    temporal_radius_quantile: float = 0.90
    temporal_radius_floor_sec: float = 2.0


@dataclass(frozen=True)
class SeedSignature:
    event_key: str
    signature: np.ndarray
    time_sec: float
    generic_score: float


@dataclass(frozen=True)
class SubjectPrototype:
    available: bool
    used: bool
    n_training_events: int
    coherence: float
    signature: np.ndarray | None


@dataclass(frozen=True)
class TemporalSupport:
    available: bool
    supported: bool
    n_training_events: int
    radius_sec: float
    n_supporting_events: int


@dataclass(frozen=True)
class CandidatePoint:
    episode_index: int
    time_index: int
    time_sec: float
    episode_start_sec: float
    consensus_step_strength: float
    spectral_breadth: float
    spatial_support: float
    state_proximity: float
    generic_score: float
    signature: np.ndarray
    prototype_similarity: float = float("nan")
    final_score: float = float("nan")


@dataclass(frozen=True)
class RefinedOnsetResult:
    phenotype_status: str
    timing_status: str
    has_candidate_time: bool
    has_accepted_time: bool
    t_candidate_sec: float
    t_best_sec: float
    episode_index: int | None
    best_score: float
    second_score: float
    score_margin: float
    prototype_available: bool
    prototype_used: bool
    n_training_events: int
    prototype_coherence: float
    prototype_similarity: float
    temporal_support_available: bool
    temporal_support_radius_sec: float
    temporal_n_supporting_events: int
    bootstrap_q05_sec: float
    bootstrap_q95_sec: float
    bootstrap_width_sec: float
    selection_consistency_1s: float
    n_candidates: int
    candidates: tuple[CandidatePoint, ...]


def _normalize(vector: np.ndarray) -> np.ndarray:
    x = np.asarray(vector, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    norm = float(np.linalg.norm(x))
    return x / norm if norm > 0.0 else np.zeros_like(x)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    a = _normalize(left)
    b = _normalize(right)
    if not np.any(a) or not np.any(b):
        return float("nan")
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def build_loso_prototype(
    seeds: Sequence[SeedSignature],
    *,
    target_event_key: str,
    config: SubjectOnsetConfig = SubjectOnsetConfig(),
) -> SubjectPrototype:
    training = [seed for seed in seeds if seed.event_key != target_event_key]
    if len(training) < config.min_prototype_events:
        return SubjectPrototype(
            available=False,
            used=False,
            n_training_events=len(training),
            coherence=float("nan"),
            signature=None,
        )
    stack = np.stack([_normalize(seed.signature) for seed in training])
    prototype = _normalize(np.nanmedian(stack, axis=0))
    similarities = np.asarray(
        [cosine_similarity(seed.signature, prototype) for seed in training], dtype=float
    )
    coherence = float(np.nanmedian(similarities))
    return SubjectPrototype(
        available=True,
        used=bool(np.isfinite(coherence) and coherence >= config.prototype_coherence_threshold),
        n_training_events=len(training),
        coherence=coherence,
        signature=prototype,
    )


def assess_temporal_support(
    seeds: Sequence[SeedSignature],
    *,
    target_event_key: str,
    candidate_time_sec: float,
    config: SubjectOnsetConfig = SubjectOnsetConfig(),
) -> TemporalSupport:
    """Test whether other seizures support the target's relative timing mode.

    The radius is patient-adaptive: it is the 90th percentile of the distance to
    each training event's second-nearest neighbour, with a 2 s resolution floor.
    Requiring two other events means an accepted timing mode contains at least
    three seizures including the target, while allowing multiple modes.
    """
    times = np.asarray(
        [
            seed.time_sec
            for seed in seeds
            if seed.event_key != target_event_key and np.isfinite(seed.time_sec)
        ],
        dtype=float,
    )
    if times.size < config.temporal_min_other_events:
        return TemporalSupport(
            available=False,
            supported=False,
            n_training_events=int(times.size),
            radius_sec=float("nan"),
            n_supporting_events=0,
        )
    neighbour_distances: list[float] = []
    for index, value in enumerate(times):
        others = np.delete(times, index)
        distances = np.sort(np.abs(others - value))
        if distances.size >= config.temporal_neighbor_k:
            neighbour_distances.append(
                float(distances[config.temporal_neighbor_k - 1])
            )
    adaptive = (
        float(np.quantile(neighbour_distances, config.temporal_radius_quantile))
        if neighbour_distances
        else float("nan")
    )
    radius = max(
        float(config.temporal_radius_floor_sec),
        adaptive if np.isfinite(adaptive) else 0.0,
    )
    n_support = int(np.sum(np.abs(times - float(candidate_time_sec)) <= radius))
    return TemporalSupport(
        available=True,
        supported=n_support >= config.temporal_min_supporting_events,
        n_training_events=int(times.size),
        radius_sec=radius,
        n_supporting_events=n_support,
    )


def connected_episode_indices(
    episodes: Sequence[SpectralEpisode],
    *,
    eeg_onset_sec: float,
    clinical_onset_sec: float,
    assignment_post_sec: float,
) -> list[int]:
    anchor_start = float(min(eeg_onset_sec, clinical_onset_sec))
    anchor_end = float(max(eeg_onset_sec, clinical_onset_sec) + assignment_post_sec)
    return [
        index
        for index, episode in enumerate(episodes)
        if episode.end_sec >= anchor_start and episode.start_sec <= anchor_end
    ]


def _candidate_indices(
    diagnostics: SpectralDiagnostics,
    episode: SpectralEpisode,
    *,
    config: SubjectOnsetConfig,
) -> list[int]:
    t = diagnostics.rel_t
    z = diagnostics.consensus_step_z
    use = np.flatnonzero(
        (t >= episode.start_sec - config.candidate_before_state_sec)
        & (t <= episode.start_sec + config.candidate_after_state_sec)
        & np.isfinite(z)
    )
    if use.size == 0:
        return []
    local: list[int] = []
    for index in use:
        left = z[index - 1] if index > 0 else -np.inf
        right = z[index + 1] if index + 1 < z.size else -np.inf
        if z[index] >= left and z[index] > right:
            local.append(int(index))
    if not local:
        local = [int(use[int(np.nanargmax(z[use]))])]
    ranked = sorted(local, key=lambda index: float(z[index]), reverse=True)
    selected: list[int] = []
    for index in ranked:
        if all(
            abs(float(t[index] - t[other])) >= config.candidate_min_separation_sec
            for other in selected
        ):
            selected.append(index)
        if len(selected) >= config.max_candidates_per_episode:
            break
    mandatory = (
        int(np.argmin(np.abs(t - episode.change_sec))),
        int(np.argmin(np.abs(t - episode.start_sec))),
    )
    for index in mandatory:
        if all(
            abs(float(t[index] - t[other])) >= 0.5 * config.candidate_min_separation_sec
            for other in selected
        ):
            selected.append(index)
    return sorted(selected, key=lambda index: float(t[index]))


def _trajectory_block(
    event: PreparedSpectralEvent,
    time_index: int,
    contact_indices: np.ndarray,
    *,
    config: SubjectOnsetConfig,
) -> np.ndarray:
    center_time = float(event.rel_t[time_index])
    sample_indices = [
        int(np.argmin(np.abs(event.rel_t - (center_time + offset))))
        for offset in config.trajectory_offsets_sec
    ]
    values = np.nanquantile(
        event.smoothed[:, contact_indices][:, :, sample_indices], 0.75, axis=1
    )
    pre_count = max(1, sum(offset < 0.0 for offset in config.trajectory_offsets_sec))
    baseline = np.nanmean(values[:, :pre_count], axis=1, keepdims=True)
    trajectory = np.clip(values - baseline, -6.0, 12.0)
    return _normalize(trajectory.reshape(-1))


def extract_candidate(
    event: PreparedSpectralEvent,
    calibration: SpectralCalibration,
    episode: SpectralEpisode,
    *,
    episode_index: int,
    time_index: int,
    config: SubjectOnsetConfig = SubjectOnsetConfig(),
    contact_indices: np.ndarray | None = None,
) -> CandidatePoint:
    n_band, n_contact, _ = event.smoothed.shape
    contacts = (
        np.arange(n_contact, dtype=int)
        if contact_indices is None
        else np.asarray(contact_indices, dtype=int)
    )
    step_z = (
        event.cell_step[:, contacts, time_index]
        - calibration.cell_step_center[:, contacts]
    ) / calibration.cell_step_scale[:, contacts]
    post_high = (
        event.cell_post_level[:, contacts, time_index]
        > calibration.level_threshold[:, contacts]
    )
    active = (step_z >= 3.0) & post_high
    band_step = np.nanquantile(np.maximum(step_z, 0.0), 0.75, axis=1)
    step_profile = _normalize(np.clip(band_step, 0.0, 12.0))
    support_profile = np.mean(active, axis=1)
    support_signature = _normalize(support_profile)
    trajectory = _trajectory_block(event, time_index, contacts, config=config)
    signature = _normalize(np.concatenate([step_profile, support_signature, trajectory]))

    min_contacts = max(2, int(np.ceil(0.25 * contacts.size)))
    band_supported = np.sum(active, axis=1) >= min_contacts
    contact_supported = np.sum(active, axis=0) >= 3
    spectral_breadth = float(np.mean(band_supported))
    spatial_support = float(np.mean(contact_supported))
    consensus_strength = float(np.nanmedian(band_step))
    strength_score = float(np.clip(consensus_strength / 6.0, 0.0, 1.0))
    time_sec = float(event.rel_t[time_index])
    state_proximity = float(np.exp(-abs(time_sec - episode.start_sec) / 5.0))
    generic = float(
        0.40 * strength_score
        + 0.20 * spectral_breadth
        + 0.20 * spatial_support
        + 0.20 * state_proximity
    )
    return CandidatePoint(
        episode_index=int(episode_index),
        time_index=int(time_index),
        time_sec=time_sec,
        episode_start_sec=float(episode.start_sec),
        consensus_step_strength=consensus_strength,
        spectral_breadth=spectral_breadth,
        spatial_support=spatial_support,
        state_proximity=state_proximity,
        generic_score=generic,
        signature=signature,
    )


def generate_candidates(
    event: PreparedSpectralEvent,
    diagnostics: SpectralDiagnostics,
    episode_indices: Sequence[int],
    *,
    config: SubjectOnsetConfig = SubjectOnsetConfig(),
    contact_indices: np.ndarray | None = None,
) -> list[CandidatePoint]:
    candidates: list[CandidatePoint] = []
    for episode_index in episode_indices:
        episode = diagnostics.episodes[int(episode_index)]
        for time_index in _candidate_indices(diagnostics, episode, config=config):
            candidates.append(
                extract_candidate(
                    event,
                    diagnostics.calibration,
                    episode,
                    episode_index=int(episode_index),
                    time_index=time_index,
                    config=config,
                    contact_indices=contact_indices,
                )
            )
    return candidates


def score_candidates(
    candidates: Sequence[CandidatePoint],
    prototype: SubjectPrototype,
    *,
    config: SubjectOnsetConfig = SubjectOnsetConfig(),
) -> list[CandidatePoint]:
    out: list[CandidatePoint] = []
    for candidate in candidates:
        similarity = (
            cosine_similarity(candidate.signature, prototype.signature)
            if prototype.available and prototype.signature is not None
            else float("nan")
        )
        if prototype.used and np.isfinite(similarity):
            similarity_01 = 0.5 * (similarity + 1.0)
            final = (
                config.prototype_weight * similarity_01
                + (1.0 - config.prototype_weight) * candidate.generic_score
            )
        else:
            final = candidate.generic_score
        out.append(
            replace(
                candidate,
                prototype_similarity=float(similarity),
                final_score=float(final),
            )
        )
    return out


def select_best_candidate(
    candidates: Sequence[CandidatePoint],
    *,
    near_tie_score: float,
) -> tuple[CandidatePoint, float]:
    if not candidates:
        raise ValueError("at least one candidate is required")
    max_score = max(float(candidate.final_score) for candidate in candidates)
    near = [
        candidate
        for candidate in candidates
        if max_score - float(candidate.final_score) <= float(near_tie_score)
    ]
    best = min(near, key=lambda candidate: candidate.time_sec)
    alternatives = [candidate.final_score for candidate in candidates if candidate is not best]
    second = float(max(alternatives)) if alternatives else float("nan")
    return best, second


def _bootstrap_times(
    event: PreparedSpectralEvent,
    diagnostics: SpectralDiagnostics,
    episode_indices: Sequence[int],
    training_seeds: Sequence[SeedSignature],
    base_prototype: SubjectPrototype,
    *,
    target_event_key: str,
    best_time: float,
    config: SubjectOnsetConfig,
    seed: int,
) -> tuple[float, float, float]:
    if config.n_boot <= 0:
        return best_time, best_time, 1.0
    rng = np.random.default_rng(int(seed))
    n_contact = event.smoothed.shape[1]
    training = [item for item in training_seeds if item.event_key != target_event_key]
    times: list[float] = []
    for _ in range(config.n_boot):
        contacts = rng.integers(0, n_contact, size=n_contact)
        candidates = generate_candidates(
            event,
            diagnostics,
            episode_indices,
            config=config,
            contact_indices=contacts,
        )
        if not candidates:
            continue
        if base_prototype.used:
            sampled = [training[int(i)] for i in rng.integers(0, len(training), len(training))]
            signature = _normalize(
                np.nanmedian(np.stack([item.signature for item in sampled]), axis=0)
            )
            prototype = replace(base_prototype, signature=signature)
        else:
            prototype = base_prototype
        scored = score_candidates(candidates, prototype, config=config)
        selected, _ = select_best_candidate(scored, near_tie_score=config.near_tie_score)
        times.append(selected.time_sec)
    if not times:
        return float("nan"), float("nan"), float("nan")
    values = np.asarray(times, dtype=float)
    q05, q95 = (float(value) for value in np.quantile(values, [0.05, 0.95]))
    consistency = float(np.mean(np.abs(values - best_time) <= config.consistency_tolerance_sec))
    return q05, q95, consistency


def refine_event_onset(
    event_key: str,
    event: PreparedSpectralEvent,
    diagnostics: SpectralDiagnostics,
    *,
    connected_indices: Sequence[int],
    training_seeds: Sequence[SeedSignature],
    config: SubjectOnsetConfig = SubjectOnsetConfig(),
    seed: int = 20260714,
) -> RefinedOnsetResult:
    prototype = build_loso_prototype(
        training_seeds, target_event_key=event_key, config=config
    )
    if not connected_indices:
        phenotype_status = (
            "prior_candidate_manual_only" if diagnostics.episodes else "phenotype_absent"
        )
        return RefinedOnsetResult(
            phenotype_status=phenotype_status,
            timing_status=(
                "no_time_prior_candidate_manual_only"
                if phenotype_status == "prior_candidate_manual_only"
                else "no_time_phenotype_absent"
            ),
            has_candidate_time=False,
            has_accepted_time=False,
            t_candidate_sec=float("nan"),
            t_best_sec=float("nan"),
            episode_index=None,
            best_score=float("nan"),
            second_score=float("nan"),
            score_margin=float("nan"),
            prototype_available=prototype.available,
            # The patient prototype may exist, but without an eligible episode
            # there are no candidates to score.  Keep availability as provenance
            # while reporting that the prototype was not actually used.
            prototype_used=False,
            n_training_events=prototype.n_training_events,
            prototype_coherence=prototype.coherence,
            prototype_similarity=float("nan"),
            temporal_support_available=False,
            temporal_support_radius_sec=float("nan"),
            temporal_n_supporting_events=0,
            bootstrap_q05_sec=float("nan"),
            bootstrap_q95_sec=float("nan"),
            bootstrap_width_sec=float("nan"),
            selection_consistency_1s=float("nan"),
            n_candidates=0,
            candidates=(),
        )
    candidates = generate_candidates(event, diagnostics, connected_indices, config=config)
    if not candidates:
        raise RuntimeError("phenotype-positive event produced no onset candidates")
    scored = score_candidates(candidates, prototype, config=config)
    best, second = select_best_candidate(scored, near_tie_score=config.near_tie_score)
    q05, q95, consistency = _bootstrap_times(
        event,
        diagnostics,
        connected_indices,
        training_seeds,
        prototype,
        target_event_key=event_key,
        best_time=best.time_sec,
        config=config,
        seed=seed,
    )
    temporal = assess_temporal_support(
        training_seeds,
        target_event_key=event_key,
        candidate_time_sec=best.time_sec,
        config=config,
    )
    if not temporal.available:
        timing_status = "candidate_no_subject_timing_template"
    elif not temporal.supported:
        timing_status = "candidate_temporally_unanchored"
    else:
        timing_status = "accepted_subject_recurrent"
    accepted = timing_status == "accepted_subject_recurrent"
    return RefinedOnsetResult(
        phenotype_status="phenotype_present",
        timing_status=timing_status,
        has_candidate_time=True,
        has_accepted_time=accepted,
        t_candidate_sec=best.time_sec,
        t_best_sec=best.time_sec if accepted else float("nan"),
        episode_index=best.episode_index,
        best_score=best.final_score,
        second_score=second,
        score_margin=(
            float(best.final_score - second) if np.isfinite(second) else float("nan")
        ),
        prototype_available=prototype.available,
        prototype_used=prototype.used,
        n_training_events=prototype.n_training_events,
        prototype_coherence=prototype.coherence,
        prototype_similarity=best.prototype_similarity,
        temporal_support_available=temporal.available,
        temporal_support_radius_sec=temporal.radius_sec,
        temporal_n_supporting_events=temporal.n_supporting_events,
        bootstrap_q05_sec=q05,
        bootstrap_q95_sec=q95,
        bootstrap_width_sec=(q95 - q05 if np.isfinite(q05) and np.isfinite(q95) else float("nan")),
        selection_consistency_1s=consistency,
        n_candidates=len(scored),
        candidates=tuple(scored),
    )


def result_to_dict(result: RefinedOnsetResult) -> dict:
    payload = asdict(result)
    payload["candidates"] = [asdict(candidate) for candidate in result.candidates]
    for candidate in payload["candidates"]:
        candidate["signature"] = np.asarray(candidate["signature"], dtype=float).tolist()
    return payload


def config_to_dict(config: SubjectOnsetConfig) -> dict:
    return asdict(config)
