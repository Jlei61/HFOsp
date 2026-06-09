import numpy as np
import pytest

from src.topic5_echo_gate import spearman_common


def test_spearman_common_identical_and_reverse():
    a = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert spearman_common(a, a, min_ch=8) == pytest.approx(1.0)
    assert spearman_common(a, a[::-1].copy(), min_ch=8) == pytest.approx(-1.0)


def test_spearman_common_too_few_returns_nan():
    a = np.array([0.0, 1.0, 2.0, np.nan, np.nan, np.nan, np.nan, np.nan])
    b = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    # only 3 common finite -> below min_ch
    assert np.isnan(spearman_common(a, b, min_ch=8))


def test_spearman_common_phantom_channel_excluded():
    # template has NaN (masked phantom) at index 7 even though seizure has a value there.
    # Result must equal Spearman over indices 0..6 only (phantom never enters).
    templ = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan])
    seiz = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 99.0])
    from scipy.stats import spearmanr
    expected = spearmanr(seiz[:7], templ[:7]).statistic
    assert spearman_common(seiz, templ, min_ch=6) == pytest.approx(expected)


def test_spearman_common_shape_mismatch_raises():
    with pytest.raises(ValueError):
        spearman_common(np.zeros(8), np.zeros(7), min_ch=5)
