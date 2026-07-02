import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.topic4_axis_vs_core import (linear_montage, split_source_axis,  # noqa: E402
                                     select_footprint, onset_time_field, runaway_delay_ms)


def test_linear_montage_centered_and_spaced():
    c = np.array([10.0, 10.0]); u = np.array([1.0, 0.0])
    contacts, names = linear_montage(c, u, n_contacts=11, pitch=1.2)
    assert contacts.shape == (11, 2) and len(names) == 11
    assert np.allclose(contacts[5], c)                        # middle contact at centre
    assert np.allclose(np.diff(contacts[:, 0]), 1.2)          # even spacing along u
    assert np.allclose(contacts[:, 1], 10.0)                  # perpendicular coord constant


def test_split_source_axis():
    c = np.array([10.0, 10.0]); u = np.array([1.0, 0.0])
    contacts, _ = linear_montage(c, u, n_contacts=11, pitch=1.2)
    src, ax = split_source_axis(contacts, c, core_radius=3.0)
    assert set(src.tolist()) == {3, 4, 5, 6, 7}               # within 3 mm of centre (|i-5|*1.2<=3)
    assert set(ax.tolist()) == {0, 1, 2, 8, 9, 10}


def test_select_footprint_symmetric_and_fair():
    c = np.array([10.0, 10.0]); u = np.array([1.0, 0.0])
    contacts, _ = linear_montage(c, u, n_contacts=11, pitch=1.2)
    src, ax = split_source_axis(contacts, c, core_radius=3.0)
    core, axis = select_footprint(contacts, c, u, src, ax, N=4)
    assert len(core) == 4 and len(axis) == 4                  # fixed footprint
    assert set(core).issubset(set(src.tolist()))             # core ⊆ source
    assert set(axis).issubset(set(ax.tolist()))              # axis ⊆ downstream
    proj = (contacts - c) @ u
    assert (proj[axis] > 0).sum() == 2 and (proj[axis] < 0).sum() == 2   # symmetric both sides


def test_select_footprint_asserts():
    c = np.array([10.0, 10.0]); u = np.array([1.0, 0.0])
    contacts, _ = linear_montage(c, u, n_contacts=11, pitch=1.2)
    src, ax = split_source_axis(contacts, c, core_radius=3.0)
    with pytest.raises(AssertionError):
        select_footprint(contacts, c, u, src, ax, N=3)       # odd N
    with pytest.raises(AssertionError):
        select_footprint(contacts, c, u, src, ax, N=6)       # N >= n_source(5): core fully coverable


def test_onset_time_field():
    # 3 E cells: cell0 first at step 2, cell1 at step 5, cell2 never
    spk = np.zeros((10, 3), bool)
    spk[2, 0] = True; spk[7, 0] = True
    spk[5, 1] = True
    got = onset_time_field(spk, dt=0.1)
    assert np.isclose(got[0], 0.2) and np.isclose(got[1], 0.5)
    assert np.isnan(got[2])


def test_runaway_delay_ms():
    assert np.isclose(runaway_delay_ms(1591.9, 757.5, 2500.0), 834.4)
    assert np.isclose(runaway_delay_ms(None, 757.5, 2500.0), 2500.0 - 757.5)   # prevented within T
    assert np.isnan(runaway_delay_ms(100.0, None, 2500.0))                     # no baseline runaway
