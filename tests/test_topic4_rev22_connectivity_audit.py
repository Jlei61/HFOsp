import numpy as np
from scipy import sparse

from src.topic4_local_connectivity import continuous_local_e_source_flow
from src.topic4_rev20_dual_core_mechanism import fixed_topology_ee_ellipse_redistribution
from src.topic4_rev22_connectivity_audit import (
    candidate_pathway_weights,
    candidate_pathway_weights_from_logits,
    candidate_structure,
    flatten_pathway,
    pathway_logits,
    target_normalized_reweight,
)


def _toy_net(seed=4, n_e=30, n_i=8, degree=10):
    rng = np.random.default_rng(seed)
    n = n_e + n_i
    positions = rng.uniform(0.0, 5.0, size=(n, 2))
    bins = [sparse.lil_matrix((n, n_e)), sparse.lil_matrix((n, n_e))]
    for target in range(n):
        choices = [source for source in range(n_e) if source != target]
        for source in rng.choice(choices, size=degree, replace=False):
            bins[int(rng.integers(0, 2))][target, source] = rng.uniform(0.2, 1.0)
    return {
        "NE": n_e,
        "NI": n_i,
        "pos": positions,
        "ampa_by_delay": [matrix.tocsc() for matrix in bins],
        "gaba_by_delay": [sparse.csc_matrix(rng.uniform(0.0, 1.0, size=(n, n_i)))],
    }


def test_target_normalization_preserves_each_target_budget():
    rows = np.array([0, 0, 0, 1, 1, 2])
    data = np.array([1.0, 2.0, 0.5, 1.5, 0.7, 4.0])
    logits = np.array([-0.2, 0.3, 0.1, 0.4, -0.5, 0.8])
    transformed = target_normalized_reweight(rows, data, logits, 3)
    np.testing.assert_allclose(
        np.bincount(rows, weights=transformed, minlength=3),
        np.bincount(rows, weights=data, minlength=3),
        rtol=0.0,
        atol=1e-12,
    )


def test_fast_final_weights_match_accepted_producer_order():
    net = _toy_net()
    n_e = net["NE"]
    positions = np.asarray(net["pos"], float)
    rng = np.random.default_rng(11)
    h_all = rng.uniform(0.0, 1.0, size=len(positions))
    coefficients = rng.normal(0.0, 0.18, size=(2, 6))
    parameters = dict(
        g_ee=0.7,
        g_etoi=1.25,
        length_scale_ee=0.38,
        length_scale_etoi=0.25,
        angle_deg=37.5,
        aspect_ratio=1.8,
        raw_logit_clip=0.75,
    )
    ee = flatten_pathway(net, "E_to_E")
    etoi = flatten_pathway(net, "E_to_I")
    fast = candidate_pathway_weights(
        ee,
        etoi,
        positions=positions,
        h_all=h_all,
        coefficients=coefficients,
        **parameters,
    )

    ellipse, _ = fixed_topology_ee_ellipse_redistribution(
        net,
        positions,
        length_scale=parameters["length_scale_ee"],
        angle_deg=parameters["angle_deg"],
        aspect_ratio=parameters["aspect_ratio"],
    )
    produced, _ = continuous_local_e_source_flow(
        ellipse,
        positions,
        h_all,
        coefficients * np.asarray([[parameters["g_ee"]], [parameters["g_etoi"]]]),
        l_ee=parameters["length_scale_ee"],
        l_e_to_i=parameters["length_scale_etoi"],
        raw_logit_clip=parameters["raw_logit_clip"],
    )
    produced_ee = flatten_pathway(produced, "E_to_E")
    produced_etoi = flatten_pathway(produced, "E_to_I")
    assert np.array_equal(ee["bin"], produced_ee["bin"])
    assert np.array_equal(etoi["bin"], produced_etoi["bin"])
    np.testing.assert_allclose(fast["E_to_E"], produced_ee["data"], rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(fast["E_to_I"], produced_etoi["data"], rtol=2e-14, atol=2e-14)

    structure = candidate_structure(ee, etoi, fast, positions)
    assert structure["E_to_E"]["maximum_abs_incoming_error"] <= 1e-12
    assert structure["E_to_I"]["maximum_abs_incoming_error"] <= 1e-12
    assert 0.0 <= structure["E_to_E"]["achieved_geometry"]["achieved_angle_deg"] < 180.0


def test_cached_logits_match_direct_fast_path():
    net = _toy_net()
    n_e = net["NE"]
    positions = np.asarray(net["pos"], float)
    h_all = np.linspace(0.0, 1.0, len(positions))
    coefficients = np.arange(12, dtype=float).reshape(2, 6) / 30.0
    ee = flatten_pathway(net, "E_to_E")
    etoi = flatten_pathway(net, "E_to_I")
    direct = candidate_pathway_weights(
        ee, etoi, positions=positions, h_all=h_all, coefficients=coefficients,
        g_ee=0.35, g_etoi=1.2, length_scale_ee=0.38,
        length_scale_etoi=0.25, angle_deg=51.0, aspect_ratio=2.2,
        raw_logit_clip=0.75,
    )
    logits = {
        "E_to_E": pathway_logits(
            dict(ee, n_e=n_e), positions, h_all, coefficients[0],
            length_scale=0.38, raw_logit_clip=None,
        ),
        "E_to_I": pathway_logits(
            dict(etoi, n_e=n_e), positions, h_all, coefficients[1],
            length_scale=0.25, raw_logit_clip=None,
        ),
    }
    cached = candidate_pathway_weights_from_logits(
        ee, etoi, positions=positions, base_logits=logits,
        g_ee=0.35, g_etoi=1.2, length_scale_ee=0.38,
        angle_deg=51.0, aspect_ratio=2.2, raw_logit_clip=0.75,
    )
    np.testing.assert_array_equal(direct["E_to_E"], cached["E_to_E"])
    np.testing.assert_array_equal(direct["E_to_I"], cached["E_to_I"])


def test_invalid_candidate_parameters_fail_closed():
    net = _toy_net()
    positions = np.asarray(net["pos"], float)
    ee = flatten_pathway(net, "E_to_E")
    etoi = flatten_pathway(net, "E_to_I")
    with np.testing.assert_raises(ValueError):
        candidate_pathway_weights(
            ee, etoi, positions=positions, h_all=np.zeros(len(positions)),
            coefficients=np.zeros((2, 6)), g_ee=-0.1, g_etoi=0.0,
            length_scale_ee=0.38, length_scale_etoi=0.25,
            angle_deg=45.0, aspect_ratio=2.0, raw_logit_clip=0.75,
        )
