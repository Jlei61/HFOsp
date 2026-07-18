import numpy as np

from src.topic5_spectral_onset import (
    SpectralOnsetConfig,
    calibration_samples,
    detect_spectral_episodes,
    fit_spectral_calibration,
    prepare_spectral_event,
)
from src.topic5_subject_spectral_onset import (
    CandidatePoint,
    SeedSignature,
    SubjectOnsetConfig,
    SubjectPrototype,
    assess_temporal_support,
    build_loso_prototype,
    refine_event_onset,
    score_candidates,
    select_best_candidate,
)


def _candidate(time, generic, signature):
    return CandidatePoint(
        episode_index=0,
        time_index=int(round(time * 10)),
        time_sec=float(time),
        episode_start_sec=0.0,
        consensus_step_strength=4.0,
        spectral_breadth=0.8,
        spatial_support=0.5,
        state_proximity=1.0,
        generic_score=float(generic),
        signature=np.asarray(signature, dtype=float),
    )


def test_loso_prototype_excludes_target_seed() -> None:
    seeds = [
        SeedSignature("target", np.array([0.0, 1.0]), 0.0, 1.0),
        SeedSignature("a", np.array([1.0, 0.0]), 0.0, 1.0),
        SeedSignature("b", np.array([1.0, 0.0]), 0.0, 1.0),
        SeedSignature("c", np.array([1.0, 0.0]), 0.0, 1.0),
    ]
    prototype = build_loso_prototype(seeds, target_event_key="target")
    assert prototype.available is True
    assert prototype.used is True
    assert prototype.n_training_events == 3
    assert np.allclose(prototype.signature, [1.0, 0.0])


def test_prototype_requires_three_other_seizures() -> None:
    seeds = [
        SeedSignature("a", np.array([1.0, 0.0]), 0.0, 1.0),
        SeedSignature("b", np.array([1.0, 0.0]), 0.0, 1.0),
    ]
    prototype = build_loso_prototype(seeds, target_event_key="target")
    assert prototype.available is False
    assert prototype.used is False


def test_patient_prototype_can_outweigh_larger_generic_late_peak() -> None:
    prototype = SubjectPrototype(True, True, 4, 0.9, np.array([1.0, 0.0]))
    early = _candidate(-1.0, 0.50, [1.0, 0.0])
    late = _candidate(3.0, 0.80, [0.0, 1.0])
    scored = score_candidates([early, late], prototype)
    best, _ = select_best_candidate(scored, near_tie_score=0.03)
    assert best.time_sec == -1.0


def test_without_coherent_prototype_generic_quality_is_used() -> None:
    prototype = SubjectPrototype(True, False, 4, 0.2, np.array([1.0, 0.0]))
    early = _candidate(-1.0, 0.50, [1.0, 0.0])
    late = _candidate(3.0, 0.80, [0.0, 1.0])
    scored = score_candidates([early, late], prototype)
    best, _ = select_best_candidate(scored, near_tie_score=0.03)
    assert best.time_sec == 3.0


def test_near_tied_candidates_choose_earliest_onset() -> None:
    candidates = [
        _candidate(-2.0, 0.71, [1.0]),
        _candidate(1.0, 0.73, [1.0]),
    ]
    candidates = [
        CandidatePoint(**{**candidate.__dict__, "final_score": candidate.generic_score})
        for candidate in candidates
    ]
    best, _ = select_best_candidate(candidates, near_tie_score=0.03)
    assert best.time_sec == -2.0


def test_temporal_support_is_loso_and_allows_recurrent_modes() -> None:
    seeds = [
        SeedSignature("target", np.ones(2), 15.0, 1.0),
        SeedSignature("a", np.ones(2), -0.5, 1.0),
        SeedSignature("b", np.ones(2), 0.0, 1.0),
        SeedSignature("c", np.ones(2), 0.5, 1.0),
        SeedSignature("d", np.ones(2), 5.0, 1.0),
        SeedSignature("e", np.ones(2), 5.5, 1.0),
        SeedSignature("f", np.ones(2), 6.0, 1.0),
    ]
    recurrent = assess_temporal_support(
        seeds, target_event_key="target", candidate_time_sec=5.2
    )
    assert recurrent.available is True
    assert recurrent.supported is True
    assert recurrent.n_supporting_events >= 2

    isolated = assess_temporal_support(
        seeds, target_event_key="target", candidate_time_sec=15.0
    )
    assert isolated.available is True
    assert isolated.supported is False


def _detected_event(with_episode: bool):
    rng = np.random.default_rng(1)
    t = np.arange(-120.0, 20.1, 0.1)
    bg_z = rng.normal(0.0, 0.1, size=(5, 8, t.size))
    target_z = rng.normal(0.0, 0.1, size=(5, 8, t.size))
    if with_episode:
        use = np.flatnonzero((t >= -5.0) & (t < 15.0))
        target_z[np.ix_([0, 1, 3, 4], [0, 1, 2, 3], use)] += 5.0
    spectral_config = SpectralOnsetConfig(n_boot=0)
    bg = prepare_spectral_event(bg_z, t, config=spectral_config)
    target = prepare_spectral_event(target_z, t, config=spectral_config)
    calibration = fit_spectral_calibration(
        [calibration_samples(bg)], config=spectral_config
    )
    diagnostics = detect_spectral_episodes(
        target,
        calibration,
        search=(-80.0, 15.0),
        config=spectral_config,
        seed=2,
    )
    return target, diagnostics


def test_refinement_assigns_time_only_when_phenotype_is_eligible() -> None:
    target, diagnostics = _detected_event(True)
    assert diagnostics.episodes
    config = SubjectOnsetConfig(n_boot=5)
    probe = refine_event_onset(
        "target",
        target,
        diagnostics,
        connected_indices=[0],
        training_seeds=[],
        config=config,
    )
    assert probe.has_candidate_time is True
    probe_candidate = min(
        probe.candidates,
        key=lambda candidate: abs(candidate.time_sec - probe.t_candidate_sec),
    )
    training = [
        SeedSignature(
            key,
            probe_candidate.signature,
            probe.t_candidate_sec + offset,
            probe_candidate.generic_score,
        )
        for key, offset in zip(("a", "b", "c"), (-0.2, 0.0, 0.2))
    ]
    accepted = refine_event_onset(
        "target",
        target,
        diagnostics,
        connected_indices=[0],
        training_seeds=training,
        config=config,
    )
    assert accepted.phenotype_status == "phenotype_present"
    assert accepted.has_candidate_time is True
    assert accepted.has_accepted_time is True
    assert np.isfinite(accepted.t_best_sec)

    prior_only = refine_event_onset(
        "target",
        target,
        diagnostics,
        connected_indices=[],
        training_seeds=[
            SeedSignature(key, np.ones(70), 0.0, 1.0)
            for key in ("a", "b", "c")
        ],
        config=config,
    )
    assert prior_only.phenotype_status == "prior_candidate_manual_only"
    assert prior_only.has_accepted_time is False
    assert np.isnan(prior_only.t_best_sec)
    assert prior_only.prototype_available is True
    assert prior_only.prototype_used is False

    absent_target, absent_diagnostics = _detected_event(False)
    absent = refine_event_onset(
        "target",
        absent_target,
        absent_diagnostics,
        connected_indices=[],
        training_seeds=[],
        config=config,
    )
    assert absent.phenotype_status == "phenotype_absent"
    assert absent.has_accepted_time is False
