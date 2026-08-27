from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import freeze_topic4_rev15_m3_coordinate_atlas as freezer
from scripts import monitor_topic4_rev15_m3_coordinate_atlas as monitor
from scripts import run_topic4_rev15_m3_coordinate_atlas_worker as worker
from src.topic4_core_field import (
    core_thresholds,
    project_to_budget,
    sample_core_quantiles,
    signed_depth,
)
from src.topic4_rev14_fourier_field import array_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_rev15_m3_coordinate_atlas.json"


def _config():
    return json.loads(CONFIG.read_text())


def test_complete_m3_coordinate_atlas_has_exact_sign_pairs():
    candidates, audit = freezer.build_candidates(_config())
    selectable = [row for row in candidates if row["selection_eligible"]]
    assert len(candidates) == 58
    assert len(selectable) == 56
    assert audit["ordered_coordinate_count"] == 28
    assert {row["candidate_id"] for row in candidates if not row["selection_eligible"]} == {
        "exact_off", "uniform_node",
    }
    for coordinate_index in range(28):
        rows = [
            row for row in selectable
            if row["coordinate_atlas"]["coordinate_index"] == coordinate_index
        ]
        assert len(rows) == 2
        rows.sort(key=lambda row: row["fourier_coordinate"]["sign"])
        negative = np.asarray(rows[0]["fourier_coordinate"]["coefficients"])
        positive = np.asarray(rows[1]["fourier_coordinate"]["coefficients"])
        assert np.array_equal(negative, -positive)
        assert rows[0]["coordinate_atlas"]["ordered_basis_only"] is True


def test_atlas_generation_contract_is_observation_free_and_node_only():
    config = _config()
    freezer._validate_config(config)
    assert config["pathways"] == {
        "learned_E_to_E_redistribution": "off",
        "learned_E_to_I_redistribution": "off",
        "Z_M": "off",
    }
    assert config["search"]["active_network_seeds"] == [2331]
    assert "contact" not in config["m3_design"]["candidate_generation_inputs"]
    assert "patient" not in config["m3_design"]["candidate_generation_inputs"]


def test_worker_wrapper_switches_only_freezer_and_status_contract():
    previous = (
        worker.base.freezer,
        worker.base.WORKER_STATUS,
        worker.base.PREPARE_STATUS,
        worker.base.EXPECTED_PATHWAYS,
    )
    try:
        worker.configure_base()
        assert worker.base.freezer is freezer
        assert worker.base.WORKER_STATUS == worker.WORKER_STATUS
        assert worker.base.EXPECTED_PATHWAYS == freezer.EXPECTED_PATHWAYS
    finally:
        (
            worker.base.freezer,
            worker.base.WORKER_STATUS,
            worker.base.PREPARE_STATUS,
            worker.base.EXPECTED_PATHWAYS,
        ) = previous


def test_real_coordinate_reaches_projection_and_preserves_node_contract():
    config = _config()
    rows, _ = freezer.build_candidates(config)
    candidate = next(row for row in rows if row["candidate_id"] == "m3_c00_p_r08")
    assert candidate["field_kind"] == "absolute_paired_phase_fourier_m3"

    rng = np.random.default_rng(1501)
    n_e, n_i = 96, 24
    positions = rng.uniform(0.0, 20.0, size=(n_e, 2))
    exact_h, _ = project_to_budget(
        np.exp(np.sin(positions[:, 0] / 3.0)), target_count=31.0,
    )
    quantile_seed = 77
    core_mean, core_std, v_base = 17.5, 1.0, 18.0
    frozen_depth = signed_depth(core_thresholds(
        sample_core_quantiles(n_e, quantile_seed), core_mean, core_std,
    ))
    exact_vtheta = np.full(n_e + n_i, v_base, dtype=np.float64)
    exact_vtheta[:n_e] = v_base - exact_h * frozen_depth
    substrate = SimpleNamespace(
        h_e=exact_h,
        vtheta=exact_vtheta,
        n_e=n_e,
        positions_e=positions,
        engine={
            "v_base": v_base, "L": 20.0,
            "core_mean": core_mean, "core_std": core_std,
        },
        stage={"N_core_manual": 31.0, "quantile_seed": quantile_seed},
    )
    config["node_mapping"]["expected_target_h_mass"] = 31.0
    config["node_mapping"]["signed_depth_contract"] = {
        "expected_n_e": n_e,
        "quantile_seed": quantile_seed,
        "core_mean_mV": core_mean,
        "core_std_mV": core_std,
        "v_base_mV": v_base,
        "sha256": array_sha256(frozen_depth),
    }

    projection = worker.base._project_candidate(candidate, substrate, config)

    assert np.isclose(projection["h"].sum(), 31.0, rtol=0.0, atol=1e-8)
    assert np.array_equal(projection["vtheta"][n_e:], exact_vtheta[n_e:])
    assert projection["audit"]["mapping"] == (
        "absolute_fourier_s_to_mass_projected_h"
    )
    assert np.isclose(
        projection["audit"]["centered_latent_surface_rms"],
        0.8,
        rtol=0.0,
        atol=2e-12,
    )


def test_controller_contract_uses_rev15_worker_and_adaptive_hard_cap():
    names = (
        "WORKER", "COMPLETE_STATUS", "MANIFEST_STATUS", "MANIFEST_SCHEMA",
        "CONTROLLER_SCHEMA", "DEFAULT_UNIT_PREFIX", "QUEUE_EMERGENCY_STATUS",
        "PREWARM_INCOMPLETE_STATUS", "QUEUE_COMPLETE_STATUS",
        "QUEUE_DRAINING_STATUS", "QUEUE_FAILED_STATUS", "PREWARM_WAIT_STATUS",
        "PREWARM_RUNNING_STATUS", "QUEUE_WAIT_STATUS", "QUEUE_RUNNING_STATUS",
    )
    previous = {name: getattr(monitor.base, name) for name in names}
    try:
        monitor.configure_base()
        assert monitor.base.WORKER.name == "run_topic4_rev15_m3_coordinate_atlas_worker.py"
        assert monitor.base.COMPLETE_STATUS == worker.WORKER_STATUS
        assert monitor.base.MANIFEST_STATUS == freezer.STATUS
        assert monitor.base.DEFAULT_UNIT_PREFIX == monitor.DEFAULT_UNIT_PREFIX
        assert _config()["resources"]["maximum_workers"] == 14
        assert _config()["resources"]["monitor_interval_seconds"] == 600
    finally:
        for name, value in previous.items():
            setattr(monitor.base, name, value)
