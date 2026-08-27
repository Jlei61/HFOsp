from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import freeze_topic4_rev15_m3_coordinate_atlas as freezer
from scripts import monitor_topic4_rev15_m3_coordinate_atlas as monitor
from scripts import run_topic4_rev15_m3_coordinate_atlas_worker as worker


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
