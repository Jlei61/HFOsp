import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "run_topic4_fcxr_lc2_gx1", ROOT / "scripts" / "run_topic4_fcxr_lc2_gx1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _arm(point, arm, label, numerical=False, slope=0.0, ceiling=0.0):
    return dict(point_id=point, family="H1", rho_fraction=0.025, theta_scale=1.0,
                theta=1.0, rho=0.54, arm=arm, numerical_failure=numerical,
                workpoint_label=label, required_low_workpoint_label=label,
                state_tail_1s=dict(h_slope_per_s=slope, h_mean=2.0,
                                  ceiling_fraction=ceiling))


def _passing_point(point="p", rho=0.025, theta_scale=1.0, family="H1"):
    rows = [
        _arm(point, "healthy_low", "INTERICTAL_WORKPOINT"),
        _arm(point, "susceptible_low", "INTERICTAL_WORKPOINT"),
        _arm(point, "susceptible_high", "FINITE_HIGH_ORBIT"),
    ]
    for row in rows:
        row.update(rho_fraction=rho, theta_scale=theta_scale, family=family)
    return rows


def test_strip_manifest_is_exactly_twelve_points_times_three_arms():
    g = _module()
    rows = g.build_strip_rows()
    assert len(rows) == 36
    assert len({r["point_id"] for r in rows}) == 12
    assert {r["arm"] for r in rows} == {"healthy_low", "susceptible_low", "susceptible_high"}
    assert {r["rho_fraction"] for r in rows} == {0.025, 0.05, 0.075}
    assert max(r["rho_fraction"] for r in rows) < 0.10
    assert {r["theta_scale"] for r in rows} == {1.0, 1.25}
    assert all(r["M"] is r["K"] is r["A"] is r["ELR"] is False for r in rows)
    assert all(r["T_ms"] == 4000.0 and r["no_kick"] for r in rows)


def test_strip_point_requires_healthy_and_susceptible_low_plus_susceptible_high():
    g = _module()
    assert g._strip_point_pass(_passing_point())["pass_point"]
    rows = _passing_point()
    rows[0]["workpoint_label"] = "ELEVATED_EVENT_TRAIN"
    assert g._strip_point_pass(rows)["label"] == "HEALTHY_LOW_IGNITES"
    rows = _passing_point()
    rows[1]["workpoint_label"] = "FINITE_HIGH_FIXED"
    assert g._strip_point_pass(rows)["label"] == "SUSCEPTIBLE_LOW_IGNITES"
    rows = _passing_point()
    rows[2]["workpoint_label"] = "INTERICTAL_WORKPOINT"
    assert g._strip_point_pass(rows)["label"] == "SUSCEPTIBLE_HIGH_NOT_MAINTAINED"


def test_strip_window_requires_within_family_grid_adjacency():
    g = _module()
    adjacent = (_passing_point("a", 0.025, 1.0, "H1") +
                _passing_point("b", 0.05, 1.0, "H1"))
    out = g.aggregate_strip(adjacent)
    assert out["verdict"] == "NATURAL_SELECTIVITY_WINDOW_CANDIDATE"
    assert out["n_window_points"] == 2

    cross_family = (_passing_point("a", 0.025, 1.0, "H1") +
                    _passing_point("b", 0.05, 1.0, "H6"))
    out = g.aggregate_strip(cross_family)
    assert out["verdict"] == "ISOLATED_SELECTIVITY_POINT"
    assert out["n_window_points"] == 0


def test_strip_anchor_prefers_lower_rho_then_higher_theta_then_h1():
    g = _module()
    points = []
    for pid, family, rho, theta in (("h6", "H6", 0.025, 1.25),
                                    ("h1_lowtheta", "H1", 0.025, 1.0),
                                    ("h1_hightheta", "H1", 0.025, 1.25)):
        points.append(dict(point_id=pid, family=family, rho_fraction=rho,
                           theta_scale=theta, theta=1.0, rho=0.54,
                           in_adjacent_window=True))
    assert g.select_strip_anchor(dict(point_rows=points))["point_id"] == "h1_hightheta"


def test_x_manifest_uses_four_locked_availabilities_and_eight_tau_window():
    g = _module()
    rows = g.build_x_rows(dict(point_rows=[]))
    assert [r["x_availability"] for r in rows] == [1.0, 0.5, 0.1, 0.0]
    assert len({r["T_ms"] for r in rows}) == 1
    assert rows[0]["T_ms"] >= 8.0 * rows[0]["tau_ms"]
    assert rows[0]["anchor_source"] == "archived_H6_k05_r10_no_strip_window"
    assert all(r["required_low_min_ms"] == 2000.0 for r in rows)
    assert all(r["M"] is r["K"] is r["A"] is r["ELR"] is False for r in rows)


def test_frozen_fork_return_window_supports_a_stricter_downstream_floor():
    text = (ROOT / "scripts" / "run_topic4_fcxr_lc2_forks.py").read_text()
    assert 'row.get("required_low_min_ms", 1000.0)' in text


def _xrow(x, wp, post=None, numerical=False):
    return dict(x_availability=x, numerical_failure=numerical, workpoint_label=wp,
                required_low_workpoint_label=post or wp)


def test_x_authority_verdict_distinguishes_range_from_maximal_bypass():
    g = _module()
    high = "FINITE_HIGH_ORBIT"
    rows = [_xrow(1.0, high), _xrow(0.5, high), _xrow(0.1, high),
            _xrow(0.0, high, "INTERICTAL_WORKPOINT")]
    assert g.classify_x_authority(rows)["verdict"] == "X_PATH_REACHABLE_RANGE_INSUFFICIENT"

    rows[-1] = _xrow(0.0, high)
    assert g.classify_x_authority(rows)["verdict"] == "H_ACTUATOR_BYPASSES_X_AT_MAXIMAL_SHUTDOWN"

    rows[2] = _xrow(0.1, high, "INTERICTAL_WORKPOINT")
    rows[3] = _xrow(0.0, high, "INTERICTAL_WORKPOINT")
    assert g.classify_x_authority(rows)["verdict"] == "X_OFFSET_ALREADY_REACHABLE_IN_CURRENT_PATH"


def test_x_authority_rejects_missing_high_anchor_or_numerical_failure():
    g = _module()
    rows = [_xrow(1.0, "INTERICTAL_WORKPOINT"),
            _xrow(0.5, "INTERICTAL_WORKPOINT"),
            _xrow(0.1, "INTERICTAL_WORKPOINT"),
            _xrow(0.0, "INTERICTAL_WORKPOINT")]
    assert g.classify_x_authority(rows)["reason"] == "anchor_high_not_established"
    rows[0] = _xrow(1.0, "FINITE_HIGH_FIXED", numerical=True)
    assert g.classify_x_authority(rows)["reason"] == "numerical_failure"
