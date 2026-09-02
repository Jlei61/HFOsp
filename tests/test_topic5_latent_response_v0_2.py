import numpy as np

from src.topic5_latent_response_v0_2 import deterministic_sets, prefix_ranks_for_references
from scripts.freeze_topic5_cross_patient_geometry_mappings_v0_2 import median_spacing
from scripts.freeze_topic5_spatial_patch_contract_v0_2 import patch_directions
from scripts.run_topic5_data_field_alignment_v0_2 import centered_valid, signed_scores


def test_deterministic_sets_respects_repeat_mask_size_and_tie_break():
    logits = np.array([[1.0, 5.0, 5.0, 4.0]])
    recruited = np.array([[1, 0, 0, 0]], dtype=bool)
    selected = deterministic_sets(logits, recruited, np.array([2]), np.array([True]))
    assert selected.tolist() == [[0, 1, 1, 0]]


def test_prefix_ranks_keeps_only_observed_prefix():
    ranks = np.array([[0, 2, 1, -1]])
    result = prefix_ranks_for_references(ranks, np.array([0]), np.array([1]))
    assert result.tolist() == [[0, -1, 1, -1]]


def test_patch_directions_are_geometry_only_unit_gaussians():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    directions, spacing, width = patch_directions(nodes)
    assert np.isclose(spacing, 1.0)
    assert np.isclose(width, 2.0)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert directions[0, 0] > directions[0, 1] > directions[0, 2]


def test_geometry_mapping_spacing_and_signed_field_do_not_oracle_flip():
    assert np.isclose(median_spacing(np.array([-0.5, 0.0, 0.5])), 0.5)
    field, valid, ok = centered_valid(np.array([0.0, 1.0, 2.0, 3.0]))
    assert ok and valid.all()
    assert signed_scores(field, field)[0] > 0.99
    assert signed_scores(-field, field)[0] < -0.99
