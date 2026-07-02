import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
for _p in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "scripts", "paper_figures"),
           os.path.join(ROOT, "src", "snn_engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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


def _fake_small_core_S(L=20.0, core_radius=3.0):
    # deterministic synthetic sheet: E cells on a grid, I cells appended; centre core.
    import numpy as np
    xs = np.linspace(1, L - 1, 24)
    gx, gy = np.meshgrid(xs, xs)
    posE = np.column_stack([gx.ravel(), gy.ravel()])
    posI = posE[:50] + 0.1
    pos = np.vstack([posE, posI])
    NE = len(posE); N = len(pos)
    labels = np.zeros(N, int); labels[NE:] = 1
    center = np.array([L / 2, L / 2]); u = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)])
    core_mask = np.zeros(N, bool)
    core_mask[:NE] = np.linalg.norm(posE - center, axis=1) <= core_radius
    return dict(net={"pos": pos}, posE=posE, posI=posI, N=N, NE=NE, labels=labels,
               center=center, axis_unit=u, L=L, core_mask=core_mask,
               layout={"kind": "stage4_patch", "foci": [center.tolist()], "core_r": core_radius})


def test_build_small_core_targets_fairness():
    from run_stage4_axis_vs_core_stim import build_small_core_targets
    S = _fake_small_core_S(core_radius=3.0)
    t = build_small_core_targets(S, core_radius=3.0, n_contacts=11, pitch=1.2, r_stim=2.0, N=4)
    is_E = np.asarray(S["labels"]) == 0
    assert t["core_mask"].shape[0] == S["N"] and t["axis_mask"].shape[0] == S["N"]
    assert (t["core_mask"] & ~is_E).sum() == 0 and (t["axis_mask"] & ~is_E).sum() == 0   # E only
    assert len(t["core_contact_idx"]) == 4 == len(t["axis_contact_idx"])                 # fixed footprint
    assert t["core_mask"].sum() > 0 and t["axis_mask"].sum() > 0
    assert (t["core_mask"] & t["axis_mask"]).sum() == 0                                  # disjoint clamp sets
