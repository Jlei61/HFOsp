import numpy as np
import pytest

from src.topic5_tspectral_type_refinement import (
    TypeRefinementConfig,
    refine_frozen_type_onset,
    select_quiet_baseline,
    smooth_band_contacts,
)


def _base() -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(-120.0, 20.1, 0.1)
    z = np.zeros((6, 8, t.size), dtype=float)
    # Make the conventional distal interval active so the resting-window
    # selector must choose another genuinely quiet pre-onset segment.
    z[:, :, (t >= -120.0) & (t < -90.0)] = 2.0
    return z, t


def test_frozen_gamma_type_uses_gamma_without_reclassifying() -> None:
    z, t = _base()
    z[4, :6, t >= 4.0] = 7.0
    z[:3, :, t >= 4.0] = -2.0
    got = refine_frozen_type_onset(z, t, "gamma_nonbroadband", 4.0)
    assert got.simple_phenotype == "gamma_nonbroadband"
    assert got.detected is True
    assert abs(got.onset_sec - 4.0) <= 0.2


def test_frozen_broadband_requires_five_of_six_bands() -> None:
    z, t = _base()
    z[:5, :6, t >= -1.0] = 6.0
    got = refine_frozen_type_onset(z, t, "broadband_1_150", -1.0)
    assert got.detected is True
    assert got.n_required_bands == 5
    assert got.n_band_post_sustained >= 5


def test_frozen_low_frequency_type_uses_two_of_three_low_bands() -> None:
    z, t = _base()
    z[0:2, :6, t >= 8.0] = 6.0
    got = refine_frozen_type_onset(z, t, "low_frequency_only", 8.0)
    assert got.detected is True
    # The centered 2 s smoother intentionally places the sustained-rise edge
    # slightly before an ideal step; the refinement still localizes the same
    # early low-frequency transition.
    assert abs(got.onset_sec - 8.0) <= 0.7


def test_other_label_is_rejected_instead_of_forced() -> None:
    z, t = _base()
    with pytest.raises(ValueError, match="simple_phenotype"):
        refine_frozen_type_onset(z, t, "other", 0.0)


def test_left_censored_gamma_change_is_not_called_an_onset() -> None:
    z, t = _base()
    # With anchor=0 the local search begins at -10 s.  A transition just
    # outside that boundary is unresolved, not a valid onset at the boundary.
    z[4, :6, t >= -10.5] = 7.0
    got = refine_frozen_type_onset(z, t, "gamma_nonbroadband", 0.0)
    assert got.detected is False


def test_quiet_baseline_avoids_high_distal_segment() -> None:
    z, t = _base()
    smooth = smooth_band_contacts(z, t, smooth_sec=2.0)
    baseline = select_quiet_baseline(
        smooth,
        t,
        config=TypeRefinementConfig(),
    )
    assert baseline.start_sec >= -90.0
