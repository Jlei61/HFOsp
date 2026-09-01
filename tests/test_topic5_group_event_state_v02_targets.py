"""A0 regression tests: per-event marks and the sparse future-block targets.

The load-bearing test is ``test_window_statistics_match_brute_force``: the whole
point of the prefix-sum builder is that it is *exactly* the dense computation,
so it is checked against a literal loop over the events in every window.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.topic5_group_event_state.v02 import marks as M
from src.topic5_group_event_state.v02 import targets as TG


def _toy(n_events: int = 400, n_contacts: int = 6, n_bands: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    part = rng.random((n_events, n_contacts)) < 0.5
    part[~part.any(1), 0] = True  # every packed event has a participant
    delay = np.where(part, rng.random((n_events, n_contacts)) * 0.05, np.nan)
    band = rng.normal(size=(n_events, n_contacts, n_bands, 5)).astype(np.float32)
    return part, delay.astype(np.float32), band


# --------------------------------------------------------------------- C6 / C7


def test_conditional_mark_covers_all_four_families() -> None:
    """C6: participation, size/STOP, continuous embedding and multiband."""

    part, delay, band = _toy()
    marks = M.build_event_marks(
        part, delay, band, band_available=(True, True, True),
        band_names=("a", "b", "c"),
        train_positions=np.arange(280), n_components=4, seed=0,
    )
    assert set(marks.block_slices) == {"size", "span", "band_energy", "band_peak", "embedding"}
    assert marks.participation.shape == part.shape
    assert marks.continuous.shape[1] == 1 + 1 + 3 + 3 + 4


def test_unavailable_bands_are_dropped_not_zero_filled() -> None:
    """C6/DC 7: a band the sampling rate cannot represent is missing, not silent."""

    part, delay, band = _toy(n_bands=3)
    marks = M.build_event_marks(
        part, delay, band, band_available=(True, False, True),
        band_names=("a", "b", "c"),
        train_positions=np.arange(280), n_components=4, seed=0,
    )
    assert marks.band_names_available == ("a", "c")
    assert marks.continuous.shape[1] == 1 + 1 + 2 + 2 + 4
    suffixes = {n.split(":", 1)[1] for n in marks.continuous_names if n.startswith("band_")}
    assert suffixes == {"a", "c"}


def test_mark_embedding_refuses_to_be_fitted_without_a_train_selection() -> None:
    """C7: repertoire geometry may only be estimated on TRAIN."""

    part, delay, band = _toy()
    with pytest.raises(TypeError):
        M.build_event_marks(  # type: ignore[call-arg]
            part, delay, band, band_available=(True, True, True),
            band_names=("a", "b", "c"), n_components=4, seed=0,
        )
    with pytest.raises(ValueError):
        M.build_event_marks(
            part, delay, band, band_available=(True, True, True),
            band_names=("a", "b", "c"),
            train_positions=np.zeros(0, dtype=np.int64), n_components=4, seed=0,
        )


def test_embedding_is_frozen_on_train_and_reused_verbatim_on_test() -> None:
    """C7: the same rows must embed identically whatever else is in the array."""

    part, delay, band = _toy()
    train = np.arange(280)
    a = M.build_event_marks(part, delay, band, band_available=(True,) * 3,
                            band_names=("a", "b", "c"), train_positions=train,
                            n_components=4, seed=0)
    b = M.apply_event_marks(part[300:], delay[300:], band[300:], a.embedding_spec, a)
    assert np.allclose(a.continuous[300:], b, atol=1e-5)


def test_events_with_a_non_finite_mark_are_flagged_not_imputed() -> None:
    """C6: ~0.05% of participant-band cells are NaN; they must not become zeros."""

    part, delay, band = _toy()
    band[7, :, 0, 2] = np.nan  # kill one band for one event, all contacts
    marks = M.build_event_marks(part, delay, band, band_available=(True,) * 3,
                                band_names=("a", "b", "c"),
                                train_positions=np.arange(280), n_components=4, seed=0)
    assert not marks.valid[7]
    assert bool(marks.valid.sum()) and marks.valid.mean() > 0.99
    assert np.isfinite(marks.continuous[marks.valid]).all()


# --------------------------------------------------------------------- C5


def test_window_statistics_match_brute_force() -> None:
    """C5: the prefix-sum builder is exactly the dense per-window computation."""

    part, delay, band = _toy(n_events=500, seed=3)
    marks = M.build_event_marks(part, delay, band, band_available=(True,) * 3,
                                band_names=("a", "b", "c"),
                                train_positions=np.arange(350), n_components=4, seed=0)
    builder = TG.FutureTargetBuilder(marks)
    rng = np.random.default_rng(1)
    lo = rng.integers(0, 480, size=40)
    hi = lo + rng.integers(0, 20, size=40)
    stats = builder.window_stats(lo, hi)

    for i, (a, b) in enumerate(zip(lo, hi)):
        assert stats.count[i] == b - a
        sel = slice(int(a), int(b))
        assert np.array_equal(stats.sum_participation[i], marks.participation[sel].sum(0))
        v = marks.valid[sel]
        x = marks.continuous[sel][v]
        assert stats.n_valid_mark[i] == v.sum()
        assert np.allclose(stats.sum_x[i], x.sum(0), atol=1e-6)
        assert np.allclose(stats.sum_x2[i], (x ** 2).sum(0), atol=1e-6)


def test_builder_never_materialises_an_event_by_horizon_by_contact_tensor() -> None:
    """C5: memory must be O(events x dims), not O(anchors x horizon x contacts)."""

    part, delay, band = _toy(n_events=2000, n_contacts=8, seed=5)
    marks = M.build_event_marks(part, delay, band, band_available=(True,) * 3,
                                band_names=("a", "b", "c"),
                                train_positions=np.arange(1400), n_components=4, seed=0)
    builder = TG.FutureTargetBuilder(marks)
    footprint = builder.prefix_bytes()
    n_ev, n_c, n_d = 2000, 8, marks.continuous.shape[1]
    assert footprint <= (n_ev + 1) * (n_c + 2 * n_d + 2) * 8 + 4096
    # a dense (anchor, horizon, contact) target for 3 horizons would be larger
    assert footprint < 3 * n_ev * n_c * 8 * 4


def test_empty_window_has_no_mark_terms_but_still_has_a_count() -> None:
    """C5/A2: p(N) is defined everywhere; p(mark | N>0) only where N>0."""

    part, delay, band = _toy(n_events=50)
    marks = M.build_event_marks(part, delay, band, band_available=(True,) * 3,
                                band_names=("a", "b", "c"),
                                train_positions=np.arange(35), n_components=4, seed=0)
    stats = TG.FutureTargetBuilder(marks).window_stats(np.array([10]), np.array([10]))
    assert stats.count[0] == 0
    assert stats.n_valid_mark[0] == 0
    assert np.all(stats.sum_x[0] == 0.0)
