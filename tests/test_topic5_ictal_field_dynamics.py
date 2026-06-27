import numpy as np
import pytest
from src import topic5_ictal_field_dynamics as fd


def test_source_core_compact_takes_top2():
    pos = {"a": (0, 0), "b": (5, 0), "c": (40, 0)}
    core, uncertain, dist = fd.source_core(["a", "b", "c"], pos, compact_mm=15.0)
    assert core == ["a", "b"] and uncertain is False and dist == pytest.approx(5.0)


def test_source_core_scattered_falls_back_to_single():
    pos = {"a": (0, 0), "b": (40, 0)}
    core, uncertain, dist = fd.source_core(["a", "b"], pos, compact_mm=15.0)
    assert core == ["a"] and uncertain is True and dist == pytest.approx(40.0)


def test_source_core_single_mapped_is_uncertain():
    core, uncertain, dist = fd.source_core(["a"], {"a": (1, 1)}, compact_mm=15.0)
    assert core == ["a"] and uncertain is True and np.isnan(dist)


def test_axis_partition_mece_four_groups():
    pos = {"a": (0, 0), "b": (40, 0), "mid": (20, 1), "off": (20, 30), "endish": (4, 1)}
    r = fd.axis_partition(["a", "b", "mid", "off", "endish"], pos, ["a"], ["b"])
    g = r["groups"]
    assert g["a"] == "source_core" and g["b"] == "source_core"
    assert g["mid"] == "axial_mid"
    assert g["off"] == "non_axial"
    assert set(g.values()) <= {"source_core", "axis_end_noncore", "axial_mid", "non_axial"}
    assert len(g) == 5


def test_axis_partition_degenerate_when_cores_coincide():
    pos = {"a": (10, 0), "b": (10.5, 0), "x": (40, 0), "y": (0, 30)}
    r = fd.axis_partition(["a", "b", "x", "y"], pos, ["a"], ["b"])
    assert r["axis_degenerate"] is True


def test_positive_mass_share_stable_with_negative_mean():
    zmean = {"a": 4.0, "b": -3.0, "c": 2.0, "d": -1.0}
    groups = {"a": "axial_mid", "b": "non_axial", "c": "non_axial", "d": "source_core"}
    pms = fd.positive_mass_share(zmean, groups)
    assert pms["axial_mid"] == pytest.approx(4 / 6) and pms["non_axial"] == pytest.approx(2 / 6)
    assert pms["source_core"] == 0.0
    assert sum(pms.values()) == pytest.approx(1.0)


def test_positive_mass_share_all_nonpositive_is_zero():
    pms = fd.positive_mass_share({"a": -1.0, "b": -2.0}, {"a": "axial_mid", "b": "non_axial"})
    assert all(v == 0.0 for v in pms.values())


def test_field_gradient_recovers_known_direction():
    pos = {f"c{i}_{j}": (float(i), float(j)) for i in range(5) for j in range(3)}
    zmean = {n: pos[n][0] for n in pos}
    ang, mag = fd.field_gradient(zmean, pos)
    assert fd.fold_angle_deg(ang, 0.0) < 1.0 and mag > 0


def test_fold_angle_deg_axis_invariant():
    assert fd.fold_angle_deg(170.0, 0.0) == pytest.approx(10.0)
    assert fd.fold_angle_deg(95.0, 0.0) == pytest.approx(85.0)


def test_field_synchrony_identical_traces_is_one():
    base = np.array([0.0, 1.0, 2.0, 3.0, 2.0])
    s = fd.field_synchrony({"a": base, "b": base * 1.0, "c": base + 0.5})
    assert s == pytest.approx(1.0)


def test_participation_fraction():
    assert fd.participation({"a": 3.0, "b": 1.0, "c": 2.5, "d": -1.0}, thresh=2.0) == pytest.approx(0.5)


def test_offset_pre_onset_overlap_short_seizure():
    assert fd.offset_pre_onset_overlap(-60.0, 25.0) is True
    assert fd.offset_pre_onset_overlap(-10.0, 25.0) is False


def test_parity_max_abs_diff_ignores_nan():
    a = np.array([1.0, 2.0, np.nan]); b = np.array([1.0, 2.0005, 9.0])
    assert fd.parity_max_abs_diff(a, b) == pytest.approx(0.0005, abs=1e-6)
