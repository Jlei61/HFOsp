import numpy as np

from scripts.aggregate_topic4_rev21_seed_audit import crossed_variance


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
