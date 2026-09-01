import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.audit_topic4_rev10_d6_2_joint_continuous_field_surface import (
    _joint_signal,
)
from scripts.freeze_topic4_rev10_d6_2_joint_continuous_field_surface import (
    candidate_library,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_rev10_d6_2_joint_continuous_field_surface.json"


def _contrast(natural_q05, crossfit_q05):
    return {
        "natural_alignment_delta": {
            "network_bootstrap_q05": natural_q05,
        },
        "crossfit_margin_delta": {
            "network_bootstrap_q05": crossfit_q05,
        },
    }


def test_d62_contract_keeps_one_continuous_field_and_closed_mechanisms():
    config = json.loads(CONFIG.read_text())
    assert config["field_search"]["candidate_count"] == 7
    assert config["field_search"]["component_count"] is None
    assert config["field_search"]["peak_count_constraint"] is None
    assert config["search"]["confirmation_network_seeds"] == list(range(1361, 1367))
    assert config["search"]["simulation"]["duration_ms"] == 16000.0
    assert config["search"]["beta"] == "closed"
    assert "slow-variable transfer" in config["forbidden"]
    assert "contact coordinates" in config["field_search"]["forbidden_builder_inputs"]


def test_d62_library_spans_source_fields_without_creating_components():
    config = json.loads(CONFIG.read_text())
    source = json.loads(
        (ROOT / config["inputs"]["d6_1_manifest"]["path"]).read_text()
    )
    rows, audit = candidate_library(config, source)
    by_id = {row["candidate_id"]: row for row in rows}
    base = np.asarray(by_id["edge_noop"]["node_field"]["coefficients"], float)
    natural = np.asarray(
        by_id["d6_f09_sin_p0p4"]["node_field"]["coefficients"], float,
    )
    geometry = np.asarray(
        by_id["d6_f05_sin_m0p8"]["node_field"]["coefficients"], float,
    )
    midpoint = np.asarray(
        by_id["d62_a0p5_b0p5"]["node_field"]["coefficients"], float,
    )
    assert np.allclose(midpoint, base + 0.5 * (natural - base) + 0.5 * (geometry - base))
    assert all(row["node_field"]["component_count"] is None for row in rows)
    assert all(row["node_field"]["peak_count_constraint"] is None for row in rows)
    assert all(np.all(np.asarray(row["coefficients"]) == 0.0) for row in rows)
    assert -1.0 <= audit["direction_cosine"] <= 1.0


def test_d62_joint_signal_is_small_and_explicit():
    assert _joint_signal(_contrast(0.01, 0.02), density_support=4, n_runaway=0)
    assert not _joint_signal(_contrast(-0.01, 0.02), density_support=6, n_runaway=0)
    assert not _joint_signal(_contrast(0.01, 0.02), density_support=3, n_runaway=0)
    assert not _joint_signal(_contrast(0.01, 0.02), density_support=6, n_runaway=1)


def test_d62_entrypoints_are_directly_executable():
    for script in (
        "freeze_topic4_rev10_d6_2_joint_continuous_field_surface.py",
        "audit_topic4_rev10_d6_2_joint_continuous_field_surface.py",
    ):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--help"],
            cwd=ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
