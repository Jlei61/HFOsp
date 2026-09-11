import numpy as np
from pathlib import Path

from scripts.aggregate_topic4_rev21_seed_audit import crossed_variance
from scripts.audit_topic4_rev21_seed_parity import arrays_equal


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
