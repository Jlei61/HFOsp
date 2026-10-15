import numpy as np

from src.topic4_native_activity_regularizer import (
    summarize_network_windows,
    unsupported_activity_for_window,
)


def _support(n_bins=5):
    support = np.zeros((2, n_bins, n_bins), float)
    for index in range(n_bins):
        support[1, index, index] = 0.01
    support[1, 1, 0] = 0.01
    support[1, 2, 1] = 0.01
    support[1, 3, 1] = 0.01
    return support


def _score(movie, *, reset_frames=2):
    return unsupported_activity_for_window(
        movie, _support(), start_frame=2, stop_frame=len(movie),
        baseline_start_frame=0, baseline_stop_frame=2,
        minimum_active_neurons=2, minimum_parent_support=0.001,
        reset_frames=reset_frames, background_quantile=0.95,
    )


def test_continuous_front_and_branch_are_supported_outside_any_core():
    movie = np.zeros((7, 1, 5), float)
    movie[2, 0, 0] = 10
    movie[3, 0, 1] = 8
    movie[4, 0, 2] = 6
    movie[4, 0, 3] = 6
    result = _score(movie)
    assert result.n_episode_seeds == 1
    assert result.unsupported_fraction == 0.0


def test_two_sequential_self_limited_episodes_receive_two_seed_exemptions():
    movie = np.zeros((10, 1, 5), float)
    movie[2:4, 0, 0] = 10
    movie[6:8, 0, 4] = 10
    result = _score(movie, reset_frames=2)
    assert result.n_episode_seeds == 2
    assert result.unsupported_fraction == 0.0


def test_distant_hotspot_before_front_arrival_increases_penalty_and_cannot_self_legitimise():
    clean = np.zeros((7, 1, 5), float)
    clean[2, 0, 0] = 10
    clean[3, 0, 1] = 8
    clean[4, 0, 2] = 6
    contaminated = clean.copy()
    contaminated[3:6, 0, 4] = 12
    clean_result = _score(clean)
    contaminated_result = _score(contaminated)
    assert clean_result.unsupported_fraction == 0.0
    assert contaminated_result.unsupported_fraction > 0.4
    assert contaminated_result.unsupported_mass == 36.0


def test_disconnected_same_frame_seed_is_not_exempt_and_background_is_separate():
    movie = np.zeros((7, 1, 5), float)
    movie[:2, 0, 4] = 1  # reported background, below the active threshold
    movie[2, 0, 0] = 10
    movie[2, 0, 4] = 8
    result = _score(movie)
    assert result.exempt_seed_mass == 10.0
    assert result.unsupported_mass > 0.0
    assert result.background_mass_per_frame == 1.0
    summary = summarize_network_windows([result])
    assert summary["n_events"] == 1
    assert summary["mean_background_mass_per_frame"] == 1.0


def test_invalid_or_empty_support_contract_is_rejected():
    movie = np.zeros((5, 1, 5), float)
    bad = np.zeros((2, 4, 4), float)
    try:
        unsupported_activity_for_window(
            movie, bad, start_frame=2, stop_frame=5,
            baseline_start_frame=0, baseline_stop_frame=2,
        )
    except ValueError as exc:
        assert "support_by_lag" in str(exc)
    else:
        raise AssertionError("mismatched support grid was accepted")


def test_delayed_psp_tail_supports_activity_after_exact_axonal_arrival_frame():
    movie = np.zeros((8, 1, 5), float)
    movie[2, 0, 0] = 10
    movie[5, 0, 1] = 8
    result = unsupported_activity_for_window(
        movie, _support(), start_frame=2, stop_frame=len(movie),
        baseline_start_frame=0, baseline_stop_frame=2,
        minimum_active_neurons=2, minimum_parent_support=0.001,
        reset_frames=5, response_tail_frames=4, response_tail_fraction=0.5,
    )
    assert result.n_episode_seeds == 1
    assert result.unsupported_fraction == 0.0
