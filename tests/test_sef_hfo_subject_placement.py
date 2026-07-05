"""TDD for subject-specific SNN placement helpers (field-swap plan §3A/§3B)."""
import numpy as np
import pytest

from src.sef_hfo_observation import from_real_geometry, VirtualMontage
from src.sef_hfo_subject_placement import (
    load_swap_endpoints, load_subject_montage, register_to_sheet)


# ---- from_real_geometry (2D-precomputed path + loud 3D fail) ----

def _geom(names_xy, scale=10.0):
    return {"norm_scale_mm": scale,
            "channels": [{"name": n, "x_norm": x, "y_norm": y} for n, x, y in names_xy]}


def test_from_real_geometry_builds_mm_montage():
    g = _geom([("A", 0.0, 0.0), ("B", 1.0, 0.0), ("C", 0.0, 1.0)], scale=10.0)
    m = from_real_geometry(g)
    assert m.names == ["A", "B", "C"]
    assert np.allclose(m.contacts[1], [10.0, 0.0])  # x_norm 1.0 * scale 10
    assert m.spans_2d()


def test_from_real_geometry_raises_without_2d_coords():
    with pytest.raises(NotImplementedError):
        from_real_geometry({"channels": [{"name": "A", "lag": 1.0}]})


def test_from_real_geometry_raises_when_collapsed_to_line():
    g = _geom([("A", 0.0, 0.0), ("B", 0.5, 0.0), ("C", 1.0, 0.0)])  # all on x-axis
    with pytest.raises(ValueError):
        from_real_geometry(g)


# ---- register_to_sheet (isotropic, frame-consistent, inside-sheet, loud overlap) ----

def _montage(coords, names):
    return VirtualMontage(np.asarray(coords, float), list(names), "test")


def test_register_all_contacts_inside_sheet():
    m = _montage([[0, 0], [40, 0], [0, 40], [40, 40], [20, 20]], list("ABCDE"))
    r = register_to_sheet(m, ["A"], ["D"], L=20.0, margin=2.0)
    Cs = np.asarray(r["montage_sheet"].contacts)
    assert (Cs >= 2.0 - 1e-6).all() and (Cs <= 18.0 + 1e-6).all()


def test_register_isotropic_preserves_aspect():
    # a wide rectangle must stay a rectangle (no per-axis squashing)
    m = _montage([[0, 0], [40, 0], [0, 10], [40, 10]], list("ABCD"))
    r = register_to_sheet(m, ["A"], ["B"], L=20.0, margin=2.0)
    Cs = np.asarray(r["montage_sheet"].contacts)
    w = Cs[:, 0].ptp(); h = Cs[:, 1].ptp()
    assert np.isclose(w / h, 4.0, rtol=1e-6)  # original 40:10 preserved


def test_register_centroids_and_axis():
    # source at left, sink at right -> forward axis points +x (theta ~ 0)
    m = _montage([[0, 5], [40, 5], [20, 5]], ["S", "K", "M"])
    r = register_to_sheet(m, ["S"], ["K"], L=20.0, margin=2.0)
    assert r["source_centroid"][0] < r["sink_centroid"][0]
    assert abs(r["theta_deg"]) < 1e-6
    assert np.allclose(r["center"], 0.5 * (r["source_centroid"] + r["sink_centroid"]))


def test_register_loud_fail_on_missing_swap_node():
    m = _montage([[0, 0], [10, 0], [0, 10]], list("ABC"))
    with pytest.raises(ValueError, match="missing from geometry"):
        register_to_sheet(m, ["A"], ["Z"], L=20.0)  # Z absent


def test_register_core_anchored_sets_inter_core_and_centers():
    # cores 22mm apart in patient frame -> core-anchored to 14mm, midpoint at sheet center
    m = _montage([[0, 10], [22, 10], [40, 10], [11, 10]], ["S", "K", "FAR", "M"])
    r = register_to_sheet(m, ["S"], ["K"], L=20.0, target_inter_core_mm=14.0)
    assert r["anchor"] == "core_anchored"
    assert np.isclose(r["inter_core_mm_sheet"], 14.0, atol=1e-6)
    assert np.allclose(r["center"], [10.0, 10.0], atol=1e-6)
    # the far contact (40mm) scales by 14/22 and may sit outside the sheet -> reported
    assert r["n_contacts_offsheet"] >= 1


def test_register_multi_contact_cores():
    m = _montage([[0, 0], [2, 0], [40, 40], [38, 40], [20, 20]], list("PQRST"))
    r = register_to_sheet(m, ["P", "Q"], ["R", "S"], L=20.0, margin=2.0)
    # centroid of P,Q is averaged in sheet frame
    Cs = {n: c for n, c in zip(r["montage_sheet"].names, np.asarray(r["montage_sheet"].contacts))}
    assert np.allclose(r["source_centroid"], 0.5 * (Cs["P"] + Cs["Q"]))


# ---- real-data smoke: E958 (clean narrow case) ----

def test_e958_real_data_placement_smoke():
    sw = load_swap_endpoints("epilepsiae_958", "narrow")
    assert sw["swap_class"] == "strict" and sw["decision_k"] == 3
    m = load_subject_montage("epilepsiae_958", "narrow", "t_a")
    r = register_to_sheet(m, sw["source"], sw["sink"], L=20.0, margin=2.0)
    Cs = np.asarray(r["montage_sheet"].contacts)
    assert (Cs >= -1e-6).all() and (Cs <= 20.0 + 1e-6).all()
    assert r["inter_core_mm_sheet"] > 3.0  # cores separated in the sheet
