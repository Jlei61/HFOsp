"""Regression tests for the Stage-0F v1.1 engineering repair."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

import scripts.run_topic4_spatial_slowfast_stage0f_v1_1 as runner
import src.topic4_spatial_slowfast_stage0f_v1_1 as stage0f
from src.topic4_spatial_slowfast_stage0c import PoolParameters
from src.topic4_spatial_slowfast_stage0e import _empty_audit


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return yaml.safe_load(
        (ROOT / "config/topic4_spatial_slowfast_stage0f_v1_1.yaml").read_text(encoding="utf-8")
    )


class _SimpleTransfer:
    def support_mask(self, mu, sigma):
        return np.ones(np.broadcast(mu, sigma).shape, dtype=bool)

    def rate(self, mu, sigma, pop):
        del sigma, pop
        return np.full(np.asarray(mu).shape, 0.01)

    def rate_with_derivatives(self, mu, sigma, pop):
        rate = self.rate(mu, sigma, pop)
        return rate, np.zeros_like(rate), np.zeros_like(rate)


def _natural_state() -> np.ndarray:
    return np.asarray([0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.5, 0.15])


def test_v1_1_config_validates_and_rejects_scope_expansion() -> None:
    cfg = _config()
    runner._validate_config(cfg)
    drifted = deepcopy(cfg)
    drifted["scope"]["adaptive_finite_difference_step"] = True
    with pytest.raises(ValueError, match="scope expansion"):
        runner._validate_config(drifted)


def test_all_locked_upstream_hashes_verify_before_execution() -> None:
    rows = runner._verify_locked_inputs(_config())
    assert rows
    assert all(row["pass"] for row in rows.values())


def test_execution_lock_covers_runtime_dependencies_and_environment(tmp_path: Path) -> None:
    cfg = _config()
    paths = runner._locked_input_paths(cfg)
    assert {"sef_hfo_lif_module", "atomic_helper_runner"}.issubset(paths)
    output = tmp_path / "lock"
    output.mkdir()
    lock = runner._write_execution_lock(
        output,
        ROOT / "config/topic4_spatial_slowfast_stage0f_v1_1.yaml",
        {"fixture": {"pass": True}},
    )
    assert set(lock["numerical_environment_pre_execution"]) == {
        "python",
        "python_implementation",
        "numpy",
        "scipy",
    }


@pytest.mark.parametrize("failure_phase", ["provenance", "transfer"])
def test_preflight_failure_never_creates_canonical_root(
    monkeypatch, tmp_path: Path, failure_phase: str
) -> None:
    cfg = _config()
    output = tmp_path / "stage0f_smooth_transfer_variational_certificate_v1_1"
    if failure_phase == "provenance":
        monkeypatch.setattr(
            runner,
            "_verify_locked_inputs",
            lambda _cfg: (_ for _ in ()).throw(RuntimeError("provenance mismatch")),
        )
    else:
        monkeypatch.setattr(runner, "_verify_locked_inputs", lambda _cfg: {})
        monkeypatch.setattr(
            runner,
            "_load_transfer",
            lambda _cfg: (_ for _ in ()).throw(RuntimeError("transfer load")),
        )
    with pytest.raises(RuntimeError):
        runner._preflight(
            ROOT / "config/topic4_spatial_slowfast_stage0f_v1_1.yaml",
            cfg,
            output,
        )
    assert not output.exists()


def test_value_only_rate_does_not_evaluate_spline_derivatives() -> None:
    class _SplineSpy:
        def __init__(self):
            self.calls = []

        def ev(self, x, y, **kwargs):
            self.calls.append(dict(kwargs))
            return np.zeros(np.asarray(x).shape)

    transfer = object.__new__(stage0f.SmoothSiegertTransferV11)
    transfer.domain = stage0f.SmoothDomain(-160.0, 80.0, 3.0, 20.0)
    transfer._spline = _SplineSpy()
    rate = transfer.rate(np.asarray([0.0, 1.0]), np.asarray([5.0, 6.0]), "E")
    assert np.all(np.isfinite(rate))
    assert transfer._spline.calls == [{}]
    transfer.rate_with_derivatives(np.asarray([0.0]), np.asarray([5.0]), "E")
    assert any(call.get("dx") == 1 for call in transfer._spline.calls)
    assert any(call.get("dy") == 1 for call in transfer._spline.calls)


def test_boundary_aware_stencil_records_central_forward_and_backward() -> None:
    params = PoolParameters(0.85, 15.0)
    scales = np.ones(9) * 0.01
    interior = _natural_state()
    _, middle = stage0f.boundary_aware_rhs_jacobian(
        interior, params, _SimpleTransfer(), scales=scales, relative_step=1e-5, absolute_floor=1e-9
    )
    assert middle["counts"] == {"central": 9, "forward": 0, "backward": 0}
    lower = interior.copy()
    lower[7] = 1e-12
    _, low = stage0f.boundary_aware_rhs_jacobian(
        lower, params, _SimpleTransfer(), scales=scales, relative_step=1e-5, absolute_floor=1e-9
    )
    assert low["coordinate_stencils"][7] == "forward"
    upper = interior.copy()
    upper[7] = 1.0 - 1e-12
    _, high = stage0f.boundary_aware_rhs_jacobian(
        upper, params, _SimpleTransfer(), scales=scales, relative_step=1e-5, absolute_floor=1e-9
    )
    assert high["coordinate_stencils"][7] == "backward"
    assert not low["first_order_fallback_used"] and not low["adaptive_step_used"]


def test_boundary_aware_stencil_fails_when_locked_two_h_probe_has_no_domain() -> None:
    state = _natural_state()
    state[7] = 0.05
    scales = np.ones(9) * 1e-6
    scales[7] = 1.0
    with pytest.raises(FloatingPointError, match="no locked second-order stencil"):
        stage0f.boundary_aware_rhs_jacobian(
            state,
            PoolParameters(0.85, 15.0),
            _SimpleTransfer(),
            scales=scales,
            relative_step=0.6,
            absolute_floor=1e-9,
        )


def test_shooting_gate_uses_event_restarted_p_and_p2_not_plot_trace_second_crossing() -> None:
    cfg = _config()["shooting"]
    shooting = {
        "converged": True,
        "residual": np.asarray([1e-4, 1e-6, 1e-10, 1e-12]),
        "period_ms": np.asarray([600.0, 600.0, 600.0, 600.0]),
    }
    cycle = {
        "valid": True,
        "aligned_cycle_residual": 1e-5,
        "second_closure_residual": 0.5,
    }
    restarted = {"valid": True, "p_closure": 1e-12, "p2_closure": 2e-12}
    summary = stage0f.shooting_summary_v1_1(shooting, cycle, restarted, cfg)
    assert summary["pass"]
    assert summary["non_event_restarted_second_closure_diagnostic_only"] == pytest.approx(0.5)
    broken = deepcopy(restarted)
    broken["p2_closure"] = 2e-7
    assert not stage0f.shooting_summary_v1_1(shooting, cycle, broken, cfg)["pass"]


def test_nominal_map_identity_requires_full_crossing_period_and_two_transversalities() -> None:
    scales = np.ones(9)
    state = _natural_state()
    restarted = {
        "valid": True,
        "return_time_ms": np.asarray([600.0, 1200.0]),
        "return_state": np.asarray([state, state]),
        "physical_support_audit": {"clean": True},
    }
    variation = {
        "valid": True,
        "period_ms": 600.0,
        "crossing_state": state.copy(),
        "continuous_rhs_transversality_per_ms": 0.005,
        "discrete_bracket_transversality_per_ms": 0.004,
        "nominal_audit": {"clean": True},
    }
    cfg = _config()["variational"]
    assert stage0f.nominal_map_identity_summary(restarted, variation, scales, cfg)["pass"]
    variation["discrete_bracket_transversality_per_ms"] = 0.0
    assert not stage0f.nominal_map_identity_summary(restarted, variation, scales, cfg)["pass"]
    variation["discrete_bracket_transversality_per_ms"] = 0.004
    variation["crossing_state"] = state.copy()
    variation["crossing_state"][0] += 1e-4
    assert not stage0f.nominal_map_identity_summary(restarted, variation, scales, cfg)["pass"]


def test_whole_return_jv_recovers_known_linear_map(monkeypatch) -> None:
    expected = np.diag(np.linspace(0.02, 0.09, 8))

    def fake_integrate(initial, params, transfer, *, dt_ms, n_returns, section):
        del params, transfer, dt_ms, n_returns, section
        baseline = 0.5 * (initial[0] + initial[1])
        returned = np.tile(baseline, (16, 1))
        for index in range(16):
            returned[index, :8] += expected @ (initial[index, :8] - baseline[:8])
        audit = _empty_audit(16)
        audit["n_euler_states"][:] = 10
        audit["peak_rE_hz"][:] = 20.0
        audit["moment_min"][:] = 0.0
        audit["moment_max"][:] = 1.0
        return {
            "valid": np.ones(16, dtype=bool),
            "return_state": returned[None, :, :],
            "return_time_ms": np.ones((1, 16)) * 600.0,
            "transversality_per_ms": np.ones((1, 16)) * 0.005,
            "audit": audit,
            "crossing_audit": [[{"clean": True} for _ in range(16)]],
        }

    monkeypatch.setattr(stage0f, "integrate_to_returns_batch", fake_integrate)
    fixed = _natural_state()
    scales = np.ones(9)
    row = stage0f.whole_return_poincare_jv(
        fixed,
        PoolParameters(0.85, 15.0),
        _SimpleTransfer(),
        dt_ms=0.125,
        section=stage0f.SectionDefinition(),
        scales=scales,
        epsilon_relative=1e-3,
    )
    assert row["valid"]
    np.testing.assert_allclose(row["jv_matrix"], expected, atol=2e-14)
    assert row["initial_probe_audit"]["clean"]
    assert row["crossing_audits_clean"]
    assert len(row["crossing_audit"]) == 16
    assert row["return_period_minimum_ms"] == pytest.approx(600.0)
    assert row["return_period_maximum_ms"] == pytest.approx(600.0)
    assert row["minimum_transversality_per_ms"] == pytest.approx(0.005)
    assert row["maximum_transversality_per_ms"] == pytest.approx(0.005)


def test_whole_return_failure_preserves_initial_and_partial_crossing_audits(
    monkeypatch,
) -> None:
    def fake_integrate(initial, params, transfer, *, dt_ms, n_returns, section):
        del initial, params, transfer, dt_ms, n_returns, section
        audit = _empty_audit(16)
        return {
            "valid": np.asarray([True] * 15 + [False]),
            "audit": audit,
            "crossing_audit": [
                [{"clean": True, "probe": index} for index in range(15)] + [None]
            ],
        }

    monkeypatch.setattr(stage0f, "integrate_to_returns_batch", fake_integrate)
    row = stage0f.whole_return_poincare_jv(
        _natural_state(),
        PoolParameters(0.85, 15.0),
        _SimpleTransfer(),
        dt_ms=0.125,
        section=stage0f.SectionDefinition(),
        scales=np.ones(9),
        epsilon_relative=1e-3,
    )
    assert not row["valid"]
    assert row["initial_probe_audit"]["clean"]
    assert row["crossing_audit_count"] == 15
    assert len(row["crossing_audit"]) == 15


def test_whole_return_jv_summary_requires_both_epsilon_and_chain_agreement() -> None:
    cfg = _config()["whole_return_jv"]
    matrix = np.diag(np.linspace(0.02, 0.09, 8))
    audit = [{"clean": True}] * 16
    rows = [
        {
            "valid": True,
            "epsilon_relative": epsilon,
            "jv_matrix": matrix.copy(),
            "spectral_radius_diagnostic": 0.09,
            "per_probe_audit": audit,
            "initial_probe_audit": {"clean": True},
            "crossing_audits_clean": True,
            "crossing_audit": audit,
            "period_band_pass": True,
        }
        for epsilon in (1e-3, 3e-4)
    ]
    assert stage0f.whole_return_jv_summary(rows, matrix, cfg, norm_floor=1e-8)["pass"]
    rows[1]["jv_matrix"][0, 1] = 0.2
    assert not stage0f.whole_return_jv_summary(rows, matrix, cfg, norm_floor=1e-8)["pass"]
    rows[1]["jv_matrix"] = matrix.copy()
    rows[1]["crossing_audit"][0] = {"clean": False}
    assert not stage0f.whole_return_jv_summary(rows, matrix, cfg, norm_floor=1e-8)["pass"]


def test_variational_transfer_parity_rows_are_actual_euler_states(monkeypatch) -> None:
    n = 600
    states = np.tile(_natural_state(), (n, 1))
    states[:, 0] += np.linspace(0.0, 0.001, n)
    crossing = np.full(9, 9.0)
    variation = {
        "nominal_state_trace": states,
        "nominal_time_ms": np.arange(n) * 0.125,
        "event_crossing_state": crossing,
    }

    def fake_exact(mu, sigma, pop):
        return _SimpleTransfer().rate_with_derivatives(mu, sigma, pop)

    monkeypatch.setattr(stage0f, "exact_siegert_rate_derivatives", fake_exact)
    rows = stage0f.variational_transfer_parity_rows(
        variation,
        PoolParameters(0.85, 15.0),
        _SimpleTransfer(),
        dt_ms=0.125,
        n_samples=512,
    )
    assert len(rows) == 1024
    assert {row["source"] for row in rows} == {"variational_nominal_euler_state"}
    assert len({row["source_state_index"] for row in rows if row["population"] == "E"}) == 512
    assert max(row["source_state_index"] for row in rows) < n
    assert not any(np.isclose(row["mu_mv"], 9.0) for row in rows)


def _minimal_point_result(params: PoolParameters) -> dict:
    return {
        "z": float(params.z),
        "alpha_G": float(params.alpha_g),
        "outcome": "periodic_orbit_derivative_unresolved",
        "derivative_certified": False,
        "failed_gates": ["fixture_gate"],
        "stage1_open": False,
        "space_open": False,
    }


def test_invalid_variational_result_serializes_partial_report(tmp_path: Path) -> None:
    result = _minimal_point_result(PoolParameters(0.85, 15.0))
    artifacts = {
        "variational": {
            "base": {"valid": False, "reason": "locked_stencil_unavailable"}
        }
    }
    counts = runner._save_point(tmp_path, result, artifacts)
    point = tmp_path / "per_point/z_0p85_alpha_15"
    assert counts == runner._empty_saved_counts()
    report = json.loads(
        (point / "discrete_variational_report.json").read_text(encoding="utf-8")
    )
    assert report["base"]["reason"] == "locked_stencil_unavailable"
    partial = json.loads(
        (point / "partial_artifact_report.json").read_text(encoding="utf-8")
    )
    assert partial["variational_levels_valid"] == {"base": False}


@pytest.mark.parametrize(
    "fault_location",
    ["boundary_stencil", "integration", "whole_return_probe", "exact_quadrature"],
)
def test_captured_numerical_faults_become_engineering_failure_and_block_later_point(
    monkeypatch, tmp_path: Path, fault_location: str
) -> None:
    def fake_run(params, transfer, stage0e_inputs, cfg):
        del transfer, stage0e_inputs, cfg
        result = _minimal_point_result(params)
        if fault_location == "boundary_stencil":
            artifacts = {
                "variational": {
                    "base": {
                        "valid": False,
                        "reason": "variational_derivative_exception",
                        "exception_type": "FloatingPointError",
                        "exception_message": "no locked second-order stencil",
                    }
                }
            }
        elif fault_location == "integration":
            artifacts = {
                "variational": {
                    "base": {
                        "valid": False,
                        "reason": "integration_exception",
                        "exception_type": "RuntimeError",
                        "exception_message": "integration fault",
                    }
                }
            }
        elif fault_location == "whole_return_probe":
            artifacts = {
                "whole_return_jv": {
                    "base": [
                        {
                            "valid": False,
                            "reason": "whole_return_integration_exception",
                            "exception_type": "RuntimeError",
                            "exception_message": "probe fault",
                            "initial_probe_audit": {"clean": True},
                            "crossing_audit": [],
                        }
                    ]
                }
            }
        else:
            result["base_exact_transfer_parity_exception"] = {
                "exception_type": "FloatingPointError",
                "exception_message": "quadrature fault",
            }
            artifacts = {}
        return result, artifacts

    monkeypatch.setattr(runner, "run_point_certificate_v1_1", fake_run)
    (
        rows,
        _,
        _,
        _,
        _,
        exceptions,
    ) = runner._run_locked_points_fail_closed(
        tmp_path,
        _config(),
        _SimpleTransfer(),
        {15: {}, 16: {}},
    )
    assert [row["outcome"] for row in rows] == [
        "numerical_failure",
        "not_executed_due_to_engineering_failure",
    ]
    assert exceptions
    for alpha in (15, 16):
        point = tmp_path / f"per_point/z_0p85_alpha_{alpha}"
        assert (point / "point_outcome.json").stat().st_size > 0
        assert (point / "partial_artifact_report.json").stat().st_size > 0


def test_point_serialization_fault_is_fail_closed(monkeypatch, tmp_path: Path) -> None:
    def fake_run(params, transfer, stage0e_inputs, cfg):
        del transfer, stage0e_inputs, cfg
        return _minimal_point_result(params), {}

    original_save = runner._save_point
    calls = {"count": 0}

    def fail_first_save(output, result, artifacts):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("injected serialization fault")
        return original_save(output, result, artifacts)

    monkeypatch.setattr(runner, "run_point_certificate_v1_1", fake_run)
    monkeypatch.setattr(runner, "_save_point", fail_first_save)
    rows, _, _, _, _, exceptions = runner._run_locked_points_fail_closed(
        tmp_path,
        _config(),
        _SimpleTransfer(),
        {15: {}, 16: {}},
    )
    assert [row["outcome"] for row in rows] == [
        "numerical_failure",
        "not_executed_due_to_engineering_failure",
    ]
    assert any(row["phase"] == "point_artifact_serialization" for row in exceptions)
    for alpha in (15, 16):
        point = tmp_path / f"per_point/z_0p85_alpha_{alpha}"
        assert (point / "point_outcome.json").is_file()
        assert (point / "partial_artifact_report.json").is_file()


@pytest.mark.parametrize(
    ("fault", "execution_exception_expected"),
    [
        ("boundary_stencil", True),
        ("integration", True),
        ("whole_return_probe", True),
        ("exact_quadrature", True),
        ("serialization", True),
        ("invalid_variational", False),
    ],
)
def test_post_preflight_faults_finalize_complete_two_point_bundle(
    monkeypatch,
    tmp_path: Path,
    fault: str,
    execution_exception_expected: bool,
) -> None:
    cfg = _config()
    output = tmp_path / "stage0f_smooth_transfer_variational_certificate_v1_1"
    cfg["result_root"] = str(output)
    config_path = tmp_path / f"stage0f_v1_1_{fault}.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    monkeypatch.setattr(
        runner,
        "_preflight",
        lambda config_path, cfg, output: (
            {"fixture": {"pass": True}},
            _SimpleTransfer(),
            {15: {}, 16: {}},
        ),
    )

    def fake_finalize(output, config_path, cfg, lock):
        del config_path, cfg
        lock["upstream_inputs_post_execution"] = lock[
            "upstream_inputs_pre_execution"
        ]
        lock["stage0f_v1_1_sources_post_execution"] = lock[
            "stage0f_v1_1_sources_pre_execution"
        ]
        lock["numerical_environment_post_execution"] = lock[
            "numerical_environment_pre_execution"
        ]
        lock["all_inputs_and_sources_unchanged_during_execution"] = True
        runner._atomic_json(output / "EXECUTION_LOCK.json", lock)
        return True

    monkeypatch.setattr(runner, "_finalize_execution_lock", fake_finalize)

    def fake_run(params, transfer, stage0e_inputs, cfg):
        del transfer, stage0e_inputs, cfg
        result = _minimal_point_result(params)
        if fault == "integration":
            raise RuntimeError("injected integration fault")
        if fault == "boundary_stencil":
            return result, {
                "variational": {
                    "base": {
                        "valid": False,
                        "exception_type": "FloatingPointError",
                        "exception_message": "injected stencil fault",
                    }
                }
            }
        if fault == "whole_return_probe":
            return result, {
                "whole_return_jv": {
                    "base": [
                        {
                            "valid": False,
                            "exception_type": "RuntimeError",
                            "exception_message": "injected whole-return fault",
                        }
                    ]
                }
            }
        if fault == "exact_quadrature":
            result["base_exact_transfer_parity_exception"] = {
                "exception_type": "FloatingPointError",
                "exception_message": "injected quadrature fault",
            }
            return result, {}
        return result, {
            "variational": {
                "base": {"valid": False, "reason": "invalid_without_exception"}
            }
        }

    monkeypatch.setattr(runner, "run_point_certificate_v1_1", fake_run)
    if fault == "serialization":
        original_save = runner._save_point
        calls = {"count": 0}

        def fail_first_save(output, result, artifacts):
            calls["count"] += 1
            if calls["count"] == 1:
                raise OSError("injected serialization fault")
            return original_save(output, result, artifacts)

        monkeypatch.setattr(runner, "_save_point", fail_first_save)

    monkeypatch.setattr(
        runner.sys,
        "argv",
        [str(runner.RUNNER), "--config", str(config_path), "--confirm-run"],
    )
    runner.main()
    summary = json.loads(
        (output / "stage0f_v1_1_variational_summary.json").read_text(
            encoding="utf-8"
        )
    )
    lock = json.loads((output / "EXECUTION_LOCK.json").read_text(encoding="utf-8"))
    assert summary["execution_exception"] is execution_exception_expected
    assert summary["engineering_pass"] is (not execution_exception_expected)
    assert lock["all_inputs_and_sources_unchanged_during_execution"] is True
    assert summary["figure_metadata"]["panels_with_content"] == 4
    assert (output / "STATUS.md").stat().st_size > 0
    for alpha in (15, 16):
        point = output / f"per_point/z_0p85_alpha_{alpha}"
        assert (point / "point_outcome.json").stat().st_size > 0
        assert (point / "partial_artifact_report.json").stat().st_size > 0


def test_failure_figure_has_explicit_content_in_all_panels(tmp_path: Path) -> None:
    rows = [
        {"alpha_G": 15.0, "outcome": "periodic_orbit_derivative_unresolved", "failed_gates": ["base_smooth_shooting"]},
        {"alpha_G": 16.0, "outcome": "periodic_orbit_derivative_unresolved", "failed_gates": ["base_smooth_shooting"]},
    ]
    metadata = runner._plot(tmp_path, rows, [{}, {}])
    assert metadata["panels_with_content"] == 4
    assert metadata["failure_annotations"] >= 4
    assert len(metadata["partial_point_annotations"]) == 2
    assert all(metadata["panel_content"].values())
    assert (tmp_path / "figures/stage0f_v1_1_variational_certificate.png").stat().st_size > 0


def test_plotting_fault_finalizes_engineering_failure_bundle(
    monkeypatch, tmp_path: Path
) -> None:
    cfg = _config()
    output = tmp_path / "stage0f_smooth_transfer_variational_certificate_v1_1"
    cfg["result_root"] = str(output)
    config_path = tmp_path / "stage0f_v1_1_fault.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    monkeypatch.setattr(
        runner,
        "_preflight",
        lambda config_path, cfg, output: (
            {},
            _SimpleTransfer(),
            {15: {}, 16: {}},
        ),
    )
    monkeypatch.setattr(
        runner,
        "run_point_certificate_v1_1",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("injected numerical fault")
        ),
    )
    monkeypatch.setattr(
        runner,
        "_plot",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("injected plotting fault")
        ),
    )
    monkeypatch.setattr(
        runner.sys,
        "argv",
        [
            str(runner.RUNNER),
            "--config",
            str(config_path),
            "--confirm-run",
        ],
    )
    runner.main()
    summary = json.loads(
        (output / "stage0f_v1_1_variational_summary.json").read_text(
            encoding="utf-8"
        )
    )
    lock = json.loads((output / "EXECUTION_LOCK.json").read_text(encoding="utf-8"))
    assert summary["execution_exception"]
    assert not summary["engineering_pass"]
    assert summary["verdict"] == "STAGE0F_ENGINEERING_OR_PROVENANCE_FAIL"
    assert "plotting_exception" in summary["figure_metadata"]
    assert lock["all_inputs_and_sources_unchanged_during_execution"] is False
    assert (output / "STATUS.md").stat().st_size > 0
    assert (
        output / "figures/stage0f_v1_1_variational_certificate.png"
    ).stat().st_size > 0
    for alpha in (15, 16):
        assert (
            output / f"per_point/z_0p85_alpha_{alpha}/point_outcome.json"
        ).stat().st_size > 0


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x\n", encoding="utf-8")


def test_artifact_completeness_locks_exact_complete_path_counts(tmp_path: Path) -> None:
    required = [
        "EXECUTION_LOCK.json",
        "parameter_point_outcomes.json",
        "parameter_point_outcomes.csv",
        "figure_metadata.json",
        "figures/stage0f_v1_1_variational_certificate.png",
        "figures/stage0f_v1_1_variational_certificate.pdf",
        "figures/README.md",
        "stage0f_v1_1_variational_summary.json",
        "STATUS.md",
        "exact_transfer_parity.csv",
        "derivative_consistency.csv",
    ]
    for alpha in (15, 16):
        prefix = f"per_point/z_0p85_alpha_{alpha}"
        required.extend(
            [
                f"{prefix}/point_outcome.json",
                f"{prefix}/partial_artifact_report.json",
                f"{prefix}/exact_transfer_parity.csv",
                f"{prefix}/smooth_base_cycle_trace.npz",
                f"{prefix}/smooth_half_cycle_trace.npz",
                f"{prefix}/discrete_variational_certificate.npz",
                f"{prefix}/discrete_variational_report.json",
                f"{prefix}/whole_return_jv_audit.npz",
                f"{prefix}/whole_return_jv_report.json",
            ]
        )
    for name in required:
        _touch(tmp_path / name)
    expected_certificate_keys = {
        f"{dt_label}_nominal_{suffix}": np.zeros(1)
        for dt_label in ("base", "half")
        for suffix in ("time_ms", "state_trace", "moment_trace")
    }
    expected_certificate_keys.update(
        {
            f"{dt_label}_{method}_{suffix}": np.zeros(1)
            for dt_label in ("base", "half")
            for method in stage0f.DERIVATIVE_LABELS_V11
            for suffix in (
                "poincare_matrix",
                "full_event_tangent_normalized",
                "multiplier_real",
                "multiplier_imag",
            )
        }
    )
    expected_jv_keys = {
        f"{dt_label}_epsilon_{index}_jv_matrix": np.zeros((8, 8))
        for dt_label in ("base", "half")
        for index in range(2)
    }
    for alpha in (15, 16):
        point = tmp_path / f"per_point/z_0p85_alpha_{alpha}"
        np.savez_compressed(
            point / "discrete_variational_certificate.npz",
            **expected_certificate_keys,
        )
        np.savez_compressed(point / "whole_return_jv_audit.npz", **expected_jv_keys)

    variation = {
        "valid": True,
        "poincare_matrices": {
            method: np.zeros((8, 8)) for method in stage0f.DERIVATIVE_LABELS_V11
        },
        "full_event_tangents_normalized": {
            method: np.zeros((9, 8)) for method in stage0f.DERIVATIVE_LABELS_V11
        },
        "multipliers": {
            method: np.zeros(8, dtype=complex)
            for method in stage0f.DERIVATIVE_LABELS_V11
        },
    }
    jv_rows = [
        {"valid": True, "epsilon_relative": epsilon, "jv_matrix": np.zeros((8, 8))}
        for epsilon in (1e-3, 3e-4)
    ]
    artifacts = [
        {
            "cycles": {"base": {}, "half": {}},
            "variational": {"base": deepcopy(variation), "half": deepcopy(variation)},
            "whole_return_jv": {"base": deepcopy(jv_rows), "half": deepcopy(jv_rows)},
        }
        for _ in range(2)
    ]
    counts = [
        {"cycle_traces": 2, "certificate_matrices": 6, "certificate_multipliers": 48, "whole_return_jv_arrays": 4},
        {"cycle_traces": 2, "certificate_matrices": 6, "certificate_multipliers": 48, "whole_return_jv_arrays": 4},
    ]
    figure = {"panels_with_content": 4, "panel_content": {"A": True, "B": True, "C": True, "D": True}, "failure_annotations": 0}
    parity_rows = []
    for alpha in (15.0, 16.0):
        for dt_ms in (0.125, 0.0625):
            for population in ("E", "I"):
                parity_rows.extend(
                    {
                        "z": 0.85,
                        "alpha_G": alpha,
                        "dt_ms": dt_ms,
                        "population": population,
                        "sample_index": index,
                        "source_state_index": index + 10,
                        "source": "variational_nominal_euler_state",
                    }
                    for index in range(512)
                )
    derivative_rows = [
        {"alpha_G": alpha, "dt_label": dt_label}
        for alpha in (15.0, 16.0)
        for dt_label in ("base", "half")
    ]
    summary = runner.artifact_completeness_summary(
        tmp_path,
        [{"alpha_G": 15.0}, {"alpha_G": 16.0}],
        artifacts,
        parity_rows,
        derivative_rows,
        counts,
        figure,
        _config(),
    )
    assert summary["pass"]
    duplicated = deepcopy(parity_rows)
    duplicated[1]["sample_index"] = duplicated[0]["sample_index"]
    assert not runner.artifact_completeness_summary(
        tmp_path,
        [{"alpha_G": 15.0}, {"alpha_G": 16.0}],
        artifacts,
        duplicated,
        derivative_rows,
        counts,
        figure,
        _config(),
    )["pass"]
    counts[0]["certificate_multipliers"] = 47
    assert not runner.artifact_completeness_summary(
        tmp_path,
        [{"alpha_G": 15.0}, {"alpha_G": 16.0}],
        artifacts,
        parity_rows,
        derivative_rows,
        counts,
        figure,
        _config(),
    )["pass"]


def test_real_near_boundary_orbit_uses_forward_mu_g_and_matches_chain_rule() -> None:
    cfg = _config()
    transfer = runner._load_transfer(cfg)
    point = ROOT / str(cfg["stage0e_root"]) / "per_point/z_0p85_alpha_15"
    with np.load(point / "base_cycle_trace.npz", allow_pickle=False) as payload:
        states = np.asarray(payload["state"], dtype=float)
    outcome = yaml.safe_load((point / "point_outcome.json").read_text(encoding="utf-8"))
    state = states[int(np.argmin(states[:, 7]))]
    scales = np.asarray(outcome["state_scales"], dtype=float)
    analytic = stage0f.analytic_rhs_jacobian(state, PoolParameters(0.85, 15.0), transfer)
    finite, metadata = stage0f.boundary_aware_rhs_jacobian(
        state,
        PoolParameters(0.85, 15.0),
        transfer,
        scales=scales,
        relative_step=1e-5,
        absolute_floor=1e-9,
    )
    assert metadata["coordinate_stencils"][7] == "forward"
    difference = stage0f.normalized_frobenius_difference(analytic, finite, norm_floor=1e-12)
    assert difference < 1e-5
