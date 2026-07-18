import numpy as np

from src.topic5_spectral_onset import (
    SpectralOnsetConfig,
    assign_target_episode,
    calibration_samples,
    detect_spectral_episodes,
    fit_spectral_calibration,
    prepare_spectral_event,
)


def _event(*, episodes=(), seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(-120.0, 20.1, 0.1)
    z = rng.normal(0.0, 0.10, size=(5, 8, t.size))
    for lo, hi, gain in episodes:
        use = (t >= lo) & (t < hi)
        # Four bands spanning low and high frequency, four co-localized contacts.
        z[np.ix_([0, 1, 3, 4], [0, 1, 2, 3], np.flatnonzero(use))] += gain
    return z, t


def _detect(target_episodes, *, background_seed=99, target_seed=1, n_boot=30):
    cfg = SpectralOnsetConfig(n_boot=n_boot)
    bg_z, t = _event(seed=background_seed)
    target_z, _ = _event(episodes=target_episodes, seed=target_seed)
    background = prepare_spectral_event(bg_z, t, config=cfg)
    target = prepare_spectral_event(target_z, t, config=cfg)
    calibration = fit_spectral_calibration([calibration_samples(background)], config=cfg)
    diagnostics = detect_spectral_episodes(
        target,
        calibration,
        search=(-80.0, 15.0),
        config=cfg,
        seed=7,
    )
    return diagnostics, cfg


def test_detects_sustained_spatial_multiband_episode_and_precise_change() -> None:
    diagnostics, cfg = _detect([(-10.0, 15.0, 5.0)])
    assert len(diagnostics.episodes) == 1
    episode = diagnostics.episodes[0]
    assert abs(episode.change_sec - (-10.0)) <= 0.25
    assert episode.n_step_bands >= 3
    assert episode.n_step_contacts >= diagnostics.min_spatial_contacts
    assert episode.precise_time is True
    assignment = assign_target_episode(
        diagnostics.episodes,
        eeg_onset_sec=0.0,
        clinical_onset_sec=0.0,
        config=cfg,
    )
    assert assignment.status == "confirmed_precise_T"
    assert assignment.target_index == 0


def test_rejects_short_broadband_transient() -> None:
    diagnostics, _ = _detect([(-10.0, -8.0, 5.0)])
    assert diagnostics.episodes == []


def test_persistence_uses_future_window_occupancy_not_uninterrupted_duration() -> None:
    # Repeating 1.8 s on / 1.2 s off epochs occupy 60% of each 5 s-scale
    # neighbourhood, but no single raw-state run lasts five seconds.
    intermittent = []
    start = -10.0
    while start < 15.0:
        intermittent.append((start, min(start + 1.8, 15.0), 5.0))
        start += 3.0
    diagnostics, _ = _detect(intermittent, n_boot=0)
    assert diagnostics.episodes
    assert diagnostics.episodes[0].duration_sec >= 5.0


def test_prior_episode_is_not_assigned_to_later_annotated_seizure() -> None:
    diagnostics, cfg = _detect([(-50.0, -35.0, 5.0)])
    assert len(diagnostics.episodes) == 1
    assignment = assign_target_episode(
        diagnostics.episodes,
        eeg_onset_sec=0.0,
        clinical_onset_sec=0.0,
        config=cfg,
    )
    assert assignment.status == "separate_prior_episode"
    assert assignment.target_index is None
    assert assignment.n_prior_episodes == 1


def test_assignment_chooses_connected_episode_not_stronger_prior_episode() -> None:
    diagnostics, cfg = _detect(
        [(-50.0, -35.0, 8.0), (-5.0, 15.0, 4.0)],
        n_boot=20,
    )
    assert len(diagnostics.episodes) == 2
    assignment = assign_target_episode(
        diagnostics.episodes,
        eeg_onset_sec=0.0,
        clinical_onset_sec=0.0,
        config=cfg,
    )
    assert assignment.target_index == 1
    assert diagnostics.episodes[assignment.target_index].change_sec > -10.0


def test_fewer_than_three_bands_does_not_define_broadband_episode() -> None:
    rng = np.random.default_rng(3)
    t = np.arange(-120.0, 20.1, 0.1)
    target_z = rng.normal(0.0, 0.10, size=(5, 8, t.size))
    use = (t >= -10.0) & (t < 15.0)
    target_z[np.ix_([0, 4], [0, 1, 2, 3], np.flatnonzero(use))] += 5.0
    bg_z, _ = _event(seed=9)
    cfg = SpectralOnsetConfig(n_boot=0)
    target = prepare_spectral_event(target_z, t, config=cfg)
    bg = prepare_spectral_event(bg_z, t, config=cfg)
    calibration = fit_spectral_calibration([calibration_samples(bg)], config=cfg)
    diagnostics = detect_spectral_episodes(
        target, calibration, search=(-80.0, 15.0), config=cfg
    )
    assert diagnostics.episodes == []


def test_imprecise_broadband_episode_still_gets_timing_interval() -> None:
    cfg = SpectralOnsetConfig(step_z_threshold=100.0, n_boot=12)
    bg_z, t = _event(seed=17)
    target_z, _ = _event(episodes=[(-10.0, 15.0, 5.0)], seed=18)
    bg = prepare_spectral_event(bg_z, t, config=cfg)
    target = prepare_spectral_event(target_z, t, config=cfg)
    calibration = fit_spectral_calibration([calibration_samples(bg)], config=cfg)
    diagnostics = detect_spectral_episodes(
        target, calibration, search=(-80.0, 15.0), config=cfg, seed=19
    )
    assert len(diagnostics.episodes) == 1
    episode = diagnostics.episodes[0]
    assert episode.automatic_change_gate is False
    assert np.isfinite(episode.bootstrap_q05_sec)
    assert np.isfinite(episode.bootstrap_q95_sec)
    assert episode.precise_time is False
