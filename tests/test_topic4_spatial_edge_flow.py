import numpy as np
from scipy import sparse

from src.topic4_core_connectivity import incoming_ee_weight
from src.topic4_spatial_edge_flow import (
    FEATURE_NAMES,
    sample_spatial_edge_features,
    spatial_vector_edge_features,
    spatial_vector_edge_logits,
    spatial_vector_ee_flow,
    spatial_vector_field,
)


def _network():
    first = sparse.csc_matrix(np.asarray([
        [0.0, 2.0, 1.0, 0.0],
        [1.0, 0.0, 3.0, 0.0],
        [2.0, 1.0, 0.0, 0.0],
        [4.0, 5.0, 6.0, 0.0],
    ]))
    second = sparse.csc_matrix(np.asarray([
        [0.0, 0.0, 0.5, 0.0],
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]))
    return {
        "NE": 3, "NI": 1, "ampa_by_delay": [first, second],
        "gaba_by_delay": [sparse.csc_matrix((4, 1))],
        "ampa_cached": object(), "_ampa_derived_cache_keys": ["ampa_cached"],
    }


def _positions():
    return np.asarray([[2.0, 3.0], [5.0, 8.0], [11.0, 6.0]])


def test_vector_edge_features_are_directed_and_contact_free():
    target = np.asarray([[2.0, 3.0], [7.0, 8.0]])
    source = np.asarray([[5.0, 4.0], [4.0, 2.0]])
    forward = spatial_vector_edge_features(target, source, L=20.0, length_scale=2.0)
    reverse = spatial_vector_edge_features(source, target, L=20.0, length_scale=2.0)
    np.testing.assert_allclose(forward, -reverse, atol=1e-15)
    assert forward.shape == (2, 12)
    assert len(FEATURE_NAMES) == 12


def test_vector_field_and_edge_logits_share_coefficients():
    target = np.asarray([[2.0, 3.0], [7.0, 8.0]])
    source = np.asarray([[5.0, 4.0], [4.0, 2.0]])
    coefficients = np.linspace(-0.2, 0.3, 12)
    midpoint = 0.5 * (target + source)
    vector = spatial_vector_field(midpoint, coefficients, L=20.0)
    expected = np.sum(vector * (source - target) / 2.0, axis=1)
    observed = spatial_vector_edge_logits(
        target, source, coefficients, L=20.0, length_scale=2.0,
    )
    np.testing.assert_allclose(observed, expected)


def test_spatial_flow_noop_and_nonzero_preserve_structure():
    net, positions = _network(), _positions()
    noop, audit0 = spatial_vector_ee_flow(
        net, positions, np.zeros(12), L=20.0, length_scale=2.0,
    )
    assert audit0["exact_noop"] and audit0["ampa_data_unchanged"]
    assert "ampa_cached" not in noop
    before = incoming_ee_weight(net["ampa_by_delay"], 3)
    mapped, audit = spatial_vector_ee_flow(
        net, positions, np.linspace(-0.1, 0.1, 12),
        L=20.0, length_scale=2.0, ratio_sample_limit=4,
    )
    np.testing.assert_allclose(
        incoming_ee_weight(mapped["ampa_by_delay"], 3), before,
        rtol=0.0, atol=1e-12,
    )
    assert audit["max_abs_incoming_E_error"] <= 1e-9
    assert audit["topology_unchanged"]
    assert audit["delay_assignment_unchanged"]
    assert audit["e_to_i_unchanged"] and audit["gaba_unchanged"]
    assert not audit["ampa_data_unchanged"]


def test_spatial_feature_audit_handles_empty_delay_bins():
    net, positions = _network(), _positions()
    empty = sparse.csc_matrix((4, 4), dtype=float)
    sample = sample_spatial_edge_features(
        [empty, *net["ampa_by_delay"], empty], positions,
        L=20.0, length_scale=2.0, sample_limit=100,
    )
    assert sample["features"].shape[1] == 12
    assert np.all(np.isfinite(sample["feature_abs_max"]))
    covariance = (
        sample["feature_gram"] / sample["n_ee_delay_entries"]
        - np.outer(sample["feature_sum"], sample["feature_sum"])
        / sample["n_ee_delay_entries"] ** 2
    )
    assert np.all(np.linalg.eigvalsh(covariance) >= -1e-12)


def test_clipped_spatial_flow_has_audited_dose_and_ratio_bound():
    net, positions = _network(), _positions()
    clip = 0.2
    _, audit = spatial_vector_ee_flow(
        net, positions, np.linspace(-4.0, 4.0, 12),
        L=20.0, length_scale=2.0, raw_logit_clip=clip,
    )
    assert audit["logit_dose"]["raw_rms"] > audit["logit_dose"]["applied_rms"]
    assert audit["logit_dose"]["clipped_edge_fraction"] > 0.0
    assert audit["logit_dose"]["applied_abs_max"] <= clip
    assert audit["edge_ratio"]["min"] >= np.exp(-2.0 * clip) - 1e-10
    assert audit["edge_ratio"]["max"] <= np.exp(2.0 * clip) + 1e-10
