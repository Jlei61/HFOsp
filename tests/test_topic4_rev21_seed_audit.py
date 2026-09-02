import hashlib
import json
import subprocess
import numpy as np
from pathlib import Path

from scripts.aggregate_topic4_rev21_seed_audit import crossed_variance
from scripts.audit_topic4_rev21_seed_parity import (
    arrays_equal, compare_runtime_modules,
)


def test_parity_comparator_supports_strings_and_floating_nans():
    assert arrays_equal(np.asarray(["ICL1", "SCL1"]),
                        np.asarray(["ICL1", "SCL1"]))
    assert not arrays_equal(np.asarray(["ICL1"]), np.asarray(["ICL2"]))
    assert arrays_equal(np.asarray([1.0, np.nan]),
                        np.asarray([1.0, np.nan]))


def test_crossed_variance_attributes_pure_topology_effect():
    row = crossed_variance([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
    assert row["status"] == "OK"
    assert np.isclose(row["variance_share_topology"], 1.0)
    assert np.isclose(row["variance_share_dynamics"], 0.0)


def test_crossed_variance_attributes_pure_dynamics_effect():
    row = crossed_variance([[1.0, 2.0, 4.0], [1.0, 2.0, 4.0]])
    assert np.isclose(row["variance_share_topology"], 0.0)
    assert np.isclose(row["variance_share_dynamics"], 1.0)


def test_crossed_variance_refuses_missing_cells():
    assert crossed_variance([[1.0, np.nan], [2.0, 3.0]])["status"] == "NOT_ESTIMABLE"


def test_seed_audit_keeps_patient_heldout_sealed():
    source = Path("scripts/aggregate_topic4_rev21_seed_audit.py").read_text()
    assert 'config["inputs"]["patient_heldout_npz"]' not in source
    assert '"patient_heldout_opened": False' in source


def test_runtime_audit_accepts_identical_modules_at_current_commit(tmp_path):
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True,
    ).strip()
    module = "src/topic4_zm_ictal_transition.py"
    config = "config/topic4_rev21_dual_core_zm_transition.json"
    module_blob = subprocess.check_output(["git", "show", f"{commit}:{module}"])
    config_blob = subprocess.check_output(["git", "show", f"{commit}:{config}"])
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({
        "provenance": {
            "expected_git_commit": commit,
            "runtime_module_sha256": {
                module: hashlib.sha256(module_blob).hexdigest(),
            },
            "config_path": config,
            "config_sha256": hashlib.sha256(config_blob).hexdigest(),
        },
    }))
    result = compare_runtime_modules(candidate, commit)
    assert result["all_runtime_modules_match_current_commit"]
    assert result["config_matches_current_commit"]
