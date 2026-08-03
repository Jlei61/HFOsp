"""Boundary and fallback selection for LC3 slow-flow probes."""
from src.topic4_fcxr_lc3_slowflow import D_ORDER, X_ORDER, select_slowflow_landmarks


def _rows(boundary=False):
    rows = []
    for state in ("low", "high"):
        for d in D_ORDER:
            for x in X_ORDER:
                high = boundary and state == "low" and D_ORDER.index(d) >= 3 and x >= 0.5
                rows.append(dict(
                    row_id=f"{state}_{d}_{x}", d_label=d, a_x=x, state_kind=state,
                    sentinel=False,
                    resolved_label="FINITE_HIGH_FIXED" if high else "INTERICTAL_WORKPOINT",
                ))
    return rows


def test_no_boundary_uses_locked_12_point_fallback():
    got = select_slowflow_landmarks(_rows(boundary=False))
    assert len(got) == 12
    assert {(r["d_label"], r["state_kind"]) for r in got} == {
        ("D_healthy", "low"), ("D50", "low"), ("Dmax", "high")}


def test_boundaries_include_both_sides_but_never_exceed_twenty():
    got = select_slowflow_landmarks(_rows(boundary=True))
    assert 12 <= len(got) <= 20
    coords = {(r["d_label"], r["a_x"], r["state_kind"]) for r in got}
    assert ("D30", 1.0, "low") in coords
    assert ("D50", 1.0, "low") in coords


def test_incomplete_map_fails_closed():
    try:
        select_slowflow_landmarks(_rows()[:-1])
    except ValueError as exc:
        assert "84-row" in str(exc)
    else:
        raise AssertionError("incomplete map must fail")
