import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.topic4_offaxis_surround_stim import (axis_frame, project_contacts,  # noqa: E402
    classify_axis_corridor, select_offaxis_surround_contacts,
    select_onaxis_corridor_contacts, onaxis_effective_halfwidth, electrode_e_mask)


def _planar_grid():
    # planar montage: axis along x (source [2,5] -> sink [12,5]); grid contacts; core near src/sink.
    xs = [0, 2, 4, 6, 8, 10, 12, 14]; ys = [1, 3, 5, 7, 9]
    contacts = np.array([[x, y] for y in ys for x in xs], float)
    src = np.array([2.0, 5.0]); sink = np.array([12.0, 5.0])
    core = ((np.linalg.norm(contacts - src, axis=1) <= 1.5)
            | (np.linalg.norm(contacts - sink, axis=1) <= 1.5))
    return contacts, src, sink, core


def test_axis_frame_and_projection():
    contacts, src, sink, _ = _planar_grid()
    fr = axis_frame(src, sink)
    assert np.allclose(fr["center"], [7, 5]) and np.isclose(fr["inter_core_mm"], 10.0)
    assert np.allclose(fr["axis_unit"], [1, 0]) and np.allclose(fr["perp_unit"], [0, 1])
    pr = project_contacts(contacts, fr)
    i = int(np.flatnonzero((contacts[:, 0] == 8) & (contacts[:, 1] == 9))[0])  # [8,9]
    assert np.isclose(pr["along"][i], 1.0) and np.isclose(pr["off"][i], 4.0)   # along=8-7, off=9-5


def test_select_offaxis_balanced_and_disjoint():
    contacts, src, sink, core = _planar_grid()
    fr = axis_frame(src, sink)
    sel = select_offaxis_surround_contacts(contacts, fr, core, N=4,
                                           corridor_halfwidth_mm=1.5, offaxis_min_mm=2.5)
    assert len(sel) == 4
    pr = project_contacts(contacts, fr)
    assert (pr["off"][sel] > 0).sum() == 2 and (pr["off"][sel] < 0).sum() == 2   # balanced both sides
    assert not core[sel].any()                                                   # never a core contact
    corridor = classify_axis_corridor(contacts, fr, 1.5)
    assert not corridor[sel].any()                                               # never in the axis corridor
    assert (np.abs(pr["off"][sel]) >= 2.5).all()                                 # genuinely off-axis


def test_onaxis_same_N_and_in_corridor():
    contacts, src, sink, core = _planar_grid()
    fr = axis_frame(src, sink)
    off = select_offaxis_surround_contacts(contacts, fr, core, N=4,
                                           corridor_halfwidth_mm=1.5, offaxis_min_mm=2.5)
    on = select_onaxis_corridor_contacts(contacts, fr, core, N=4, corridor_halfwidth_mm=1.5)
    assert len(on) == len(off) == 4                                              # same footprint
    corridor = classify_axis_corridor(contacts, fr, 1.5)
    assert corridor[on].all() and not core[on].any()                            # in corridor, not core
    assert set(on.tolist()).isdisjoint(set(off.tolist()))                        # on/off disjoint


def test_onaxis_falls_back_to_nearest_axis_when_corridor_sparse():
    # all contacts are 3 mm off-axis (none within a 1.5 mm corridor); on-axis must still find N by
    # falling back to nearest-axis, and the effective halfwidth flags the degraded comparator.
    xs = [3, 5, 7, 9, 11]
    contacts = np.array([[x, 5 + s] for s in (-3.0, 3.0) for x in xs], float)
    fr = axis_frame([3.0, 5.0], [11.0, 5.0])
    on = select_onaxis_corridor_contacts(contacts, fr, np.zeros(len(contacts), bool),
                                         N=4, corridor_halfwidth_mm=1.5)
    assert len(on) == 4
    assert onaxis_effective_halfwidth(contacts, fr, on) >= 1.5   # degraded: beyond nominal corridor


def test_insufficient_offaxis_raises():
    contacts, src, sink, core = _planar_grid()
    fr = axis_frame(src, sink)
    with pytest.raises(ValueError):
        select_offaxis_surround_contacts(contacts, fr, core, N=20,             # too many requested
                                         corridor_halfwidth_mm=1.5, offaxis_min_mm=2.5)
    with pytest.raises(ValueError):
        select_offaxis_surround_contacts(contacts, fr, core, N=4,
                                         corridor_halfwidth_mm=1.5, offaxis_min_mm=99.0)  # none that far off


def test_electrode_e_mask_deterministic():
    posE = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0], [10.0, 0.0]])
    contacts = np.array([[0.0, 0.0], [10.0, 0.0]])
    assert electrode_e_mask(posE, contacts, [0], 1.5).tolist() == [True, True, False, False]
    assert electrode_e_mask(posE, contacts, [0, 1], 1.5).tolist() == [True, True, False, True]
