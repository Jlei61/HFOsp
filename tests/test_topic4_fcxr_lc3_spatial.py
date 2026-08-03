import numpy as np
import pytest
import importlib.util
from pathlib import Path

from src.topic4_fcxr_lc3_spatial import (
    build_equal_local_masks,
    build_signed_basis,
    global_control_patterns,
    positive_patterns,
    projected_response_matrix,
    rate_fields,
    svd_summary,
)


def _runner_module():
    path = Path(__file__).parents[1] / "scripts" / "run_topic4_fcxr_lc3_spatial.py"
    spec = importlib.util.spec_from_file_location("run_topic4_fcxr_lc3_spatial", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _geometry():
    x, y = np.meshgrid(np.linspace(-5, 5, 41), np.linspace(-5, 5, 41))
    return np.c_[x.ravel(), y.ravel()], np.array([-2.0, 0.0]), np.array([2.0, 0.0])


def test_local_masks_are_equal_count_and_reproducible():
    pos, src, snk = _geometry()
    a = build_equal_local_masks(pos, src, snk, np.array([1.0, 0.0]), core_r=1.0)
    b = build_equal_local_masks(pos, src, snk, np.array([1.0, 0.0]), core_r=1.0)
    assert len({int(mask.sum()) for mask in a.values()}) == 1
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
    assert not np.array_equal(a["axial"], a["transverse"])


def test_global_controls_match_charge_or_rms_but_not_both():
    controls = global_control_patterns(100, 25)
    local = np.r_[np.ones(25), np.zeros(75)]
    assert controls["global_charge_matched"].sum() == pytest.approx(local.sum())
    assert np.sqrt(np.mean(controls["global_rms_matched"] ** 2)) == pytest.approx(
        np.sqrt(np.mean(local ** 2)))
    assert np.linalg.norm(controls["global_charge_matched"]) != pytest.approx(
        np.linalg.norm(local))


def test_signed_basis_is_locked_nine_dimensional_and_unit_l2():
    pos, src, snk = _geometry()
    masks = build_equal_local_masks(pos, src, snk, np.array([1.0, 0.0]), core_r=1.0)
    positive_patterns(masks)
    basis = build_signed_basis(masks)
    assert len(basis) == 9
    for value in basis.values():
        assert np.linalg.norm(value) == pytest.approx(1.0)


def test_rate_fields_and_projected_svd_recover_identity_response():
    raster = np.zeros((100, 3), bool)
    raster[10, 0] = True; raster[40, 1] = True; raster[80, 2] = True
    fields = rate_fields(raster, dt_ms=1.0, response_times_ms=(50.0, 100.0), window_ms=50.0)
    np.testing.assert_array_equal(fields[50.0], [20.0, 20.0, 0.0])
    np.testing.assert_array_equal(fields[100.0], [0.0, 0.0, 20.0])

    basis = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    plus = {
        "a": {50.0: np.array([2.0, 0.0])},
        "b": {50.0: np.array([0.0, 2.0])},
    }
    minus = {name: {50.0: -value[50.0]} for name, value in plus.items()}
    names, matrices = projected_response_matrix(plus, minus, basis, epsilon_l2=2.0)
    np.testing.assert_array_equal(matrices[50.0], np.eye(2))
    summary = svd_summary(matrices[50.0], names)
    assert summary["sigma_max"] == pytest.approx(1.0)


def test_positive_first_passage_separates_new_recruitment_from_background():
    runner = _runner_module()
    times = runner.RESPONSE_TIMES_MS
    zero_fields = {t: np.zeros(4) for t in times}
    arm = dict(
        fields=zero_fields, active=np.array([True, True, False, False]),
        first_passage=np.array([5.0, 12.0, np.nan, np.nan]),
        accounting={}, max_population_rate_hz=1.0,
        refractory_ceiling_fraction=0.0, artifact={},
    )
    sham = dict(
        fields=zero_fields, active=np.array([True, False, False, False]),
        first_passage=np.array([10.0, np.nan, np.nan, np.nan]),
        max_population_rate_hz=1.0, refractory_ceiling_fraction=0.0,
    )
    substrate = dict(posE=np.array([[0.0, 0.0], [1.0, 0.0],
                                    [2.0, 0.0], [3.0, 0.0]]), L=20.0)
    regions = {"all": np.ones(4, bool)}
    metrics = runner._positive_metrics(
        substrate, arm, sham, np.ones(4), 1.0, regions)
    assert metrics["first_passage_newly_recruited_region_median_ms"]["all"] == 12.0
    assert metrics["shared_active_first_passage_shift_region_median_ms"]["all"] == -5.0
    assert metrics["first_passage_scope"] == "raw_arm_activity_including_background"
