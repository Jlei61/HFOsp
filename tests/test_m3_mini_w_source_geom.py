"""Geometry contract for the mini-W_event 5-source layout (design §1).

5 sources around the core center at offset R_src along the E->E long axis (theta=45)
and the off-axis (135). No SNN.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from run_m3_mini_w_pilot import source_xy, SOURCE_NAMES, _cfg  # noqa: E402


def test_pilot_cfg_matches_ceiling_working_point():
    # The ceiling used a 5x5 grid (thresholds.json n_bins=25, NOT in config.sweep_parameters).
    # On 5x5 the sheet center L/2 is a bin CENTROID; on 4x4 it is the 4-bin junction (inflates
    # r95_ea/far_ea, flips the argmin source bin). This guard prevents regressing to 4.
    cfg = _cfg(False)
    assert cfg["n_bins_per_axis"] == 5, "pilot must use the ceiling's 5x5 grid (n_bins=25)"
    assert cfg["L"] == 20.0 and cfg["density"] == 100.0 and cfg["seeds"] == 12
    assert cfg["t_kick"] == 100.0 and cfg["T"] == 500.0
    # the sheet center (L/2) must be a 1D grid centroid, not a bin boundary/junction
    n = cfg["n_bins_per_axis"]; L = cfg["L"]
    centers = (np.arange(n) + 0.5) * (L / n)
    assert np.any(np.isclose(centers, L / 2)), \
        f"center L/2={L/2} is not a bin centroid on a {n}x{n} grid -> junction pathology"


def test_five_sources_named():
    assert SOURCE_NAMES == ["center", "+axis", "-axis", "+offaxis", "-offaxis"]


def test_center_is_core_center():
    assert np.allclose(source_xy("center", center=(10.0, 10.0), r_src=4.0), [10.0, 10.0])


def test_axis_sources_lie_on_45deg_line():
    c = (10.0, 10.0); r = 4.0
    plus = source_xy("+axis", center=c, r_src=r)
    minus = source_xy("-axis", center=c, r_src=r)
    off = r / np.sqrt(2)                       # 4*cos45
    assert np.allclose(plus, [10 + off, 10 + off])
    assert np.allclose(minus, [10 - off, 10 - off])
    # both exactly R_src from center
    assert np.isclose(np.linalg.norm(np.subtract(plus, c)), r)
    assert np.isclose(np.linalg.norm(np.subtract(minus, c)), r)


def test_offaxis_is_perpendicular_to_axis():
    c = (10.0, 10.0); r = 4.0
    off = r / np.sqrt(2)
    assert np.allclose(source_xy("+offaxis", center=c, r_src=r), [10 - off, 10 + off])
    assert np.allclose(source_xy("-offaxis", center=c, r_src=r), [10 + off, 10 - off])
    # axis and off-axis directions are orthogonal
    axis_dir = np.subtract(source_xy("+axis", center=c, r_src=r), c)
    off_dir = np.subtract(source_xy("+offaxis", center=c, r_src=r), c)
    assert np.isclose(np.dot(axis_dir, off_dir), 0.0, atol=1e-9)
