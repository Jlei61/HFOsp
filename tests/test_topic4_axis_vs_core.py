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
                                     select_footprint, onset_time_field, runaway_delay_ms,
                                     count_events_pre_runaway)


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


def _synthetic_train_af(nbins=1000):
    # 3 near-baseline train bumps + a big late detonation peak (mimics the kick trajectory)
    af = np.full(nbins, 0.005)
    for c in (200, 450, 700):
        af[c - 20:c + 20] = 0.030
    af[755:900] = 0.480
    return af


def test_count_events_pre_runaway_train_not_buried_by_late_peak():
    from src.sef_hfo_events import detect_events
    af = _synthetic_train_af(); bin_w, runaway_ms = 1.0, 760.0
    n, bar = count_events_pre_runaway(af, bin_w, runaway_ms, detect_events)
    assert n == 3, f"expected 3 pre-runaway train bumps, got {n}"          # sensitive pre-runaway bar
    # the naive whole-record 0.5x-peak bar is set by the 0.48 detonation and buries the ~0.03 train
    floor = float(np.percentile(af[5:50], 95))
    naive_bar = floor + 0.5 * (af.max() - floor)
    assert len(detect_events(af, bin_w, event_on_frac=naive_bar)) <= 1     # record-peak confound


def test_count_events_pre_runaway_immediate_burst_is_one():
    from src.sef_hfo_events import detect_events
    af = np.full(200, 0.005); af[60:190] = 0.48                            # one burst (past baseline win)
    n, _ = count_events_pre_runaway(af, 1.0, 60.0, detect_events)          # runaway<300 -> whole record
    assert n == 1


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


def test_figure_b_renders_from_fixture(tmp_path):
    import plot_fig_stage4_axis_vs_core_difficulty as F
    small = {"config": {"N": 4}, "n_source_contacts": 5,
             "contacts": [[i * 1.0, 10.0] for i in range(11)],
             "core_contact_idx": [4, 5, 6, 7], "axis_contact_idx": [2, 3, 8, 9],
             "arms": {"no_stim": {"runaway_ms": 50.0},
                      "core_stim": {"runaway_ms": 120.0, "runaway_delay_ms": 70.0},
                      "axis_stim": {"runaway_ms": 160.0, "runaway_delay_ms": 110.0}}}
    geom = {"foci": [[3.5, 8.5], [16.5, 8.5]], "L": 20.0, "core_r": 1.5,
            "names": [f"ICL{i}" for i in range(1, 12)],
            "contacts": [[3.5 + i * 1.3, 8.5] for i in range(11)],
            "core_contacts": ["ICL8", "ICL9", "ICL10", "ICL11"],
            "axis_contacts": ["ICL4", "ICL5", "ICL6", "ICL7"]}
    out = tmp_path / "figs"; out.mkdir()
    F.render_figure_b(small, F.KICK_REF, out, kick_geom=geom)   # exercise the E1146 geometry path
    assert (out / "axis_vs_core.png").exists() and (out / "axis_vs_core.png").stat().st_size > 0
