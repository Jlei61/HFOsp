"""Contracts for the registered four-way temporal-geometry label."""
import pytest

from src.topic4_fcxr_lc3_pathlabel import (
    drift_toward_return,
    has_high_branch,
    return_bracket,
    temporal_geometry_label,
    window_is_adequate,
)


def _low(n=6, total_ms=1500.0, label="INTERICTAL_WORKPOINT"):
    return [dict(d_label=f"D{i}", a_x=1.0, state_kind="low",
                 resolved_label=label, total_ms=total_ms) for i in range(n)]


def _high(fields=("D10", "D30"), survive_at=(1.0, 0.8), quiet_at=(0.65, 0.5),
          total_ms=5000.0):
    cells = []
    for f in fields:
        for a in survive_at:
            cells.append(dict(d_label=f, a_x=a, state_kind="high",
                              resolved_label="FINITE_HIGH_FIXED", total_ms=total_ms))
        for a in quiet_at:
            cells.append(dict(d_label=f, a_x=a, state_kind="high",
                              resolved_label="INTERICTAL_WORKPOINT", total_ms=total_ms))
    return cells


def _vec(a_x, mean_slope, regional):
    return dict(state_kind="high", a_x=a_x, dot_mean_a_X_per_s=mean_slope,
                regional_X_change=dict(zip(
                    ("core_A", "core_B", "axial", "off_axis"), regional)))


# --- window adequacy: the rule that makes absence readable -------------------

def test_a_window_shorter_than_ignition_is_inadequate():
    a = window_is_adequate(_low(total_ms=1500.0), reference_ms=4000.0)
    assert not a["adequate"] and a["shortfall_ms"] == 2500.0


def test_a_window_reaching_ignition_is_adequate():
    a = window_is_adequate(_low(total_ms=8000.0), reference_ms=4000.0)
    assert a["adequate"] and a["n_adequate"] == 6 and a["shortfall_ms"] == 0.0


def test_adequacy_refuses_a_nonpositive_reference():
    with pytest.raises(ValueError, match="positive ignition time"):
        window_is_adequate(_low(), reference_ms=0.0)


# --- brackets ----------------------------------------------------------------

def test_return_bracket_is_derived_per_wear_field():
    r = return_bracket(_high())
    assert r["present"] and r["n_bracketed"] == 2
    assert r["per_field"]["D10"] == (0.65, 0.8)


def test_a_field_that_never_stops_surviving_yields_no_bracket():
    cells = [dict(d_label="D10", a_x=a, state_kind="high",
                  resolved_label="FINITE_HIGH_FIXED", total_ms=5000.0)
             for a in (1.0, 0.8, 0.65)]
    assert return_bracket(cells)["present"] is False


def test_high_branch_detection_covers_orbit_as_well_as_fixed():
    assert has_high_branch([dict(resolved_label="FINITE_HIGH_ORBIT")])
    assert not has_high_branch([dict(resolved_label="ELEVATED_EVENT_TRAIN")])


# --- drift: mean and regions are reported separately -------------------------

def test_a_region_moving_toward_return_is_not_erased_by_a_rising_mean():
    """The measured case: mean relay rises while both cores fall."""
    d = drift_toward_return([_vec(0.8, +0.0359, (-0.0142, -0.0238, +0.0113, +0.0116))],
                            bracket_top=0.8)
    assert d["mean_reaches_return"] is False
    assert d["any_region_reaches_return"] is True
    assert d["regions_toward_return"] == {"core_A": 1, "core_B": 1}


def test_drift_below_the_bracket_is_not_counted():
    d = drift_toward_return([_vec(0.5, -0.05, (-0.01,) * 4)], bracket_top=0.8)
    assert d["measured"] is False


# --- the four-way decision ---------------------------------------------------

def test_an_unwatched_quiet_side_is_unresolved_not_absent():
    """1500 ms cells against a 4000 ms ignition cannot say the quiet state stays."""
    out = temporal_geometry_label(
        low_cells=_low(total_ms=1500.0), high_cells=_high(),
        vectors=[_vec(0.8, +0.036, (-0.014, -0.024, +0.011, +0.012))],
        ignition_times_ms=[5000.0, 6000.0, 4000.0])
    assert out["label"] == "DX_MAP_UNRESOLVED"
    assert "screen window, not the tissue" in out["reason"]
    assert out["fastest_observed_ignition_ms"] == 4000.0


def test_a_watched_quiet_side_that_never_departs_is_absent():
    out = temporal_geometry_label(
        low_cells=_low(total_ms=8000.0), high_cells=_high(),
        vectors=[_vec(0.8, -0.02, (-0.01,) * 4)],
        ignition_times_ms=[4000.0])
    assert out["label"] == "DX_GEOMETRIC_PATH_ABSENT"


def test_both_brackets_with_no_component_toward_return_is_misaligned():
    out = temporal_geometry_label(
        low_cells=_low(total_ms=8000.0, label="FINITE_HIGH_FIXED"),
        high_cells=_high(),
        vectors=[_vec(0.8, +0.036, (+0.01, +0.02, +0.011, +0.012))],
        ignition_times_ms=[4000.0])
    assert out["label"] == "DX_DYNAMIC_VECTOR_MISALIGNED"


def test_both_brackets_with_drift_connecting_them_is_present():
    out = temporal_geometry_label(
        low_cells=_low(total_ms=8000.0, label="FINITE_HIGH_FIXED"),
        high_cells=_high(),
        vectors=[_vec(0.8, -0.02, (-0.01, +0.02, +0.011, +0.012))],
        ignition_times_ms=[4000.0])
    assert out["label"] == "DX_GEOMETRIC_PATH_PRESENT"


def test_departure_without_a_return_bracket_is_absent():
    cells = [dict(d_label="D10", a_x=a, state_kind="high",
                  resolved_label="FINITE_HIGH_FIXED", total_ms=5000.0)
             for a in (1.0, 0.8)]
    out = temporal_geometry_label(
        low_cells=_low(total_ms=8000.0, label="FINITE_HIGH_FIXED"),
        high_cells=cells, vectors=[], ignition_times_ms=[4000.0])
    assert out["label"] == "DX_GEOMETRIC_PATH_ABSENT"
    assert "stops surviving" in out["reason"]


def test_the_label_records_that_it_gates_nothing():
    out = temporal_geometry_label(
        low_cells=_low(), high_cells=_high(), vectors=[], ignition_times_ms=[4000.0])
    assert "neither opens nor closes" in out["authorizes_nothing"]
