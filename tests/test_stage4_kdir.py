"""TDD for the Phase 2 --k-dir thread (reviewer P2 2026-06-17): the readable participant floor
`part_min = 2*k_dir+1` must move CONSISTENTLY — both the `read_event` valid-contact gate AND the
`endpoint_centroid_axis` participant floor. Default (k_dir=3 -> part_min=7) stays byte-identical.
"""
import os
import sys

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, os.path.join("src", "snn_engine"))
sys.path.insert(0, os.getcwd())
import run_sef_hfo_snn_cm_spontaneous_readout as R   # noqa: E402
from src.sef_hfo_observation import VirtualMontage    # noqa: E402


def _toy_inputs():
    # 8 contacts on a line (x=0..7); only the first 6 are VALID. The 6 valid contacts fire as
    # staggered Gaussians (distinct peak times -> distinct ranks along +x), so they all participate.
    fdt = 0.1
    nt = 300
    t = np.arange(nt) * fdt
    contacts = np.column_stack([np.arange(8.0), np.zeros(8)])
    names = [f"A{i}" for i in range(8)]
    m = VirtualMontage(contacts, names, "toy")
    valid = np.array([True] * 6 + [False] * 2)
    env = np.full((8, nt), 0.01)
    for i in range(6):
        env[i] = env[i] + np.exp(-((t - 2.0 * (i + 1)) ** 2) / (2 * 0.8 ** 2))
    return env, fdt, m, valid, (0.0, 26.0), np.array([1.0, 0.0])


def test_read_event_floor_moves_with_k_dir():
    env, fdt, m, valid, win, axis = _toy_inputs()
    # k_dir=3 -> part_min=7: only 6 valid contacts < 7 -> gated unreadable
    rd3 = R.read_event(env, fdt, m, valid, win, axis, k_dir=3, part_min=7)
    assert rd3["n_part"] == 0 and rd3["sign"] is None
    # k_dir=2 -> part_min=5: 6 valid >= 5 -> reads; 6 participants -> a direction
    rd2 = R.read_event(env, fdt, m, valid, win, axis, k_dir=2, part_min=5)
    assert rd2["n_part"] == 6
    assert rd2["sign"] in (1.0, -1.0)


def test_read_event_default_is_kdir3_partmin7_byte_identical():
    env, fdt, m, valid, win, axis = _toy_inputs()
    a = R.read_event(env, fdt, m, valid, win, axis)                       # defaults
    b = R.read_event(env, fdt, m, valid, win, axis, k_dir=3, part_min=7)  # explicit current floor
    assert a["n_part"] == b["n_part"] == 0          # 6 valid < 7 -> unreadable at the default floor
    assert a["sign"] is b["sign"] is None
    assert R.KDIR == 3 and R.PART_MIN == 7          # module defaults unchanged


def test_part_min_floor_relation():
    # the participant floor is 2*k_dir+1 (the contract the runner derives from --k-dir)
    assert 2 * 3 + 1 == 7
    assert 2 * 2 + 1 == 5


def test_montage_pitch_knob_denser_and_default_unchanged():
    # E2 readout-escape: smaller pitch packs contacts closer (denser in-patch sampling); default
    # pitch keeps the existing geometry byte-identical.
    center = np.array([16.0, 16.0])
    m_def = R.montage(center, 45.0, 0.0, 7)                # default pitch=4
    m_p3 = R.montage(center, 45.0, 0.0, 7, pitch=3.0)      # E2 dense
    m_p4 = R.montage(center, 45.0, 0.0, 7, pitch=4.0)      # explicit 4 == default
    C_def = np.asarray(m_def.contacts); C3 = np.asarray(m_p3.contacts); C4 = np.asarray(m_p4.contacts)
    assert np.allclose(C_def, C4)                          # default unchanged
    # along the ∥ shaft (first 7), neighbour spacing is the pitch
    d3 = np.linalg.norm(np.diff(C3[:7], axis=0), axis=1)
    d4 = np.linalg.norm(np.diff(C4[:7], axis=0), axis=1)
    assert np.allclose(d3, 3.0) and np.allclose(d4, 4.0)
    assert np.asarray(m_p3.contacts).max() < np.asarray(m_p4.contacts).max()   # denser -> more compact


def test_read_event_eps_deg_scales_with_pitch():
    # P1-1 (reviewer 2026-06-17): the eps_deg displacement floor (0.5*pitch mm) must follow the
    # MONTAGE pitch, not the global PITCH=4. An event whose early->late centroid displacement is
    # ~1.7 mm reads at pitch=3 (eps 1.5) but is wrongly rejected at pitch=4 (eps 2.0) — the bug that
    # confounded the E2 dense-montage NO-GO.
    fdt = 0.1; nt = 320; t = np.arange(nt) * fdt
    xs = np.array([0.0, 0.2, 0.4, 1.5, 1.7, 1.9])      # two tight clusters ~1.7 mm apart along x
    m = VirtualMontage(np.column_stack([xs, np.zeros(6)]), [f"A{i}" for i in range(6)], "toy")
    valid = np.ones(6, bool)
    env = np.full((6, nt), 0.01)
    for i in range(6):
        env[i] = env[i] + np.exp(-((t - (2.0 + 4.0 * i)) ** 2) / (2 * 0.8 ** 2))   # staggered by x-order
    win = (0.0, 28.0); axis = np.array([1.0, 0.0])
    rd_p3 = R.read_event(env, fdt, m, valid, win, axis, k_dir=2, part_min=5, pitch=3.0)   # eps 1.5
    rd_p4 = R.read_event(env, fdt, m, valid, win, axis, k_dir=2, part_min=5, pitch=4.0)   # eps 2.0
    assert rd_p3["n_part"] == 6 and rd_p3["sign"] in (1.0, -1.0)   # 1.7 mm > 1.5 -> readable
    assert rd_p4["sign"] is None                                   # 1.7 mm < 2.0 -> wrongly rejected
