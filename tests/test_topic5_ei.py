"""TDD for src.topic5_ei — EI-like must REWARD early + strong (delay in the denominator)."""
import numpy as np

from src.topic5_ei import onset_delays, ei_like

HOP = 0.1


def _traces():
    z = np.zeros((3, 100))
    z[0, 0:] = 3.0          # ch0 crosses z>2 at frame 0 (earliest)
    z[1, 50:] = 3.0         # ch1 crosses at frame 50 (= 5.0 s later)
    # ch2 never crosses
    return z


def test_onset_delays_relative_to_earliest():
    d = onset_delays(_traces(), hop_sec=HOP, z_onset=2.0)
    assert np.isclose(d[0], 0.0)        # earliest -> delay 0
    assert np.isclose(d[1], 5.0)        # 50 frames * 0.1 s
    assert np.isnan(d[2])               # never crossed


def test_ei_like_rewards_early_not_late():
    z = _traces()
    auc = np.array([5.0, 5.0, 5.0])     # equal strength -> only timing differs
    ei = ei_like(z, auc, hop_sec=HOP, z_onset=2.0, tau=1.0)
    assert ei[0] > ei[1] > ei[2]        # early > late > never (delay is a PENALTY)
    assert np.isclose(ei[0], 5.0)       # 5 / (0 + 1)
    assert np.isclose(ei[1], 5.0 / 6.0)  # 5 / (5 + 1)
    assert ei[2] == 0.0                 # no onset -> 0, never NaN-imputed


def test_ei_like_negative_auc_clamped_to_zero_energy():
    z = _traces()
    ei = ei_like(z, np.array([-3.0, 5.0, 5.0]), hop_sec=HOP, z_onset=2.0, tau=1.0)
    assert ei[0] == 0.0                 # below-baseline HFA -> ER clamped to 0
