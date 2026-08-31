from __future__ import annotations

import pytest

from scripts.topic5_continuous_marked_state_h2b.build_v03_closeout import (
    _assert_no_active_downstream,
    _strict_route_payload,
)


def test_strict_route_closes_every_downstream_branch() -> None:
    route = _strict_route_payload("2026-08-31T00:00:00+00:00")
    assert route["status"] == "NOT_RELEASED_A1_AND_A2_FAILED"
    assert route["A3_A5_hazard_lag"] == "NOT_RUN_GATE_CLOSED"
    assert route["A6_OOS_manifold_flow"] == "NOT_RUN_GATE_CLOSED"
    assert route["A7_IED_objective_ablation"] == "NOT_RUN_GATE_CLOSED"
    assert route["A8_frozen_phenotype_bridge"] == "NOT_RUN_GATE_CLOSED"
    assert route["biological_negative_allowed"] is False


def test_active_downstream_guard_allows_only_geometry_route_receipt(tmp_path) -> None:
    geometry = tmp_path / "geometry"
    geometry.mkdir()
    (geometry / "ROUTE_STATUS.json").write_text("{}", encoding="utf-8")
    _assert_no_active_downstream(tmp_path)
    (tmp_path / "hazard_full_grid").mkdir()
    with pytest.raises(ValueError, match="hazard"):
        _assert_no_active_downstream(tmp_path)
