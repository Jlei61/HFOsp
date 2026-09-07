"""Scientific regression: zero-probability and self edges cannot be sampled."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/snn_engine"))
import connectivity as con
import connectivity_rot as rot
from params import Params


@pytest.mark.parametrize("kind", ["isotropic", "rotated", "pruned"])
@pytest.mark.parametrize("count", [1, 2, 8])
def test_self_and_underflow_sources_are_excluded(kind, count):
    points = np.array([[0., 0.], [.1, 0.], [0., .1], [1000., 1000.]])
    rng = np.random.default_rng(7)
    if kind == "isotropic":
        chosen = con._sample_partners(points[0], points, count, .38, 0, rng, self_local=0)
    else:
        chosen = rot._sample_partners_rot(
            points[0], points, count, .38*np.sqrt(2), .38/np.sqrt(2), .2,
            rng, self_local=0, prune_radius=.5 if kind == "pruned" else None)
    assert len(chosen) == min(count, 2)
    assert set(chosen).issubset({1, 2})
    assert len(set(chosen)) == len(chosen)


def test_all_zero_weights_excluding_only_self_returns_empty():
    points = np.array([[0., 0.], [1000., 1000.]])
    assert con._sample_partners(points[0], points, 4, .1, 0, np.random.default_rng(1), self_local=0).size == 0
    assert rot._sample_partners_rot(points[0], points, 4, .1, .1, 0, np.random.default_rng(1), self_local=0).size == 0


def test_correction_preserves_positive_priorities_and_future_random_stream():
    weights = np.array([0., .7, .3, .1])
    old_rng, new_rng = np.random.default_rng(9), np.random.default_rng(9)
    draws = old_rng.standard_exponential(4)
    new = con._positive_weight_keys(weights, new_rng)
    assert np.isinf(new[0])
    np.testing.assert_array_equal(new[1:], draws[1:]/weights[1:])
    np.testing.assert_array_equal(old_rng.random(10), new_rng.random(10))


def test_full_small_graph_has_no_autapses_and_exact_pathway_degrees():
    p = Params(L=2, density=30, C_EE=12, C_IE=10, C_EI=8, C_II=7, seed=3)
    rng = np.random.default_rng(p.seed)
    pos, labels, ne, ni = con.place_neurons(p, rng)
    net = rot.build_connectivity_rot(p, pos, labels, ne, ni, rng, theta_EE=.3, AR=2)
    a = sum(net["ampa_by_delay"]); g = sum(net["gaba_by_delay"])
    assert not a[:ne].diagonal().any()
    assert not g[ne:].diagonal().any()
    np.testing.assert_array_equal(np.diff(a.tocsr().indptr), [12]*ne+[10]*ni)
    np.testing.assert_array_equal(np.diff(g.tocsr().indptr), [8]*ne+[7]*ni)


def test_sampler_identity_prevents_reuse_of_old_cache_key():
    from src.topic4_core_field_runner import connectivity_config, cache_key
    cfg = connectivity_config(Params(), -22.8, 2, git_commit="same-commit")
    assert cfg["partner_sampler_version"] == con.PARTNER_SAMPLER_VERSION
    legacy = {key:value for key,value in cfg.items() if key != "partner_sampler_version"}
    assert cache_key(cfg) != cache_key(legacy)


@pytest.mark.parametrize("field", ["dt", "w_EE", "tau_m_E", "tau_r_AMPA"])
def test_stored_weight_and_delay_parameters_change_cache_identity(field):
    from src.topic4_core_field_runner import connectivity_config, cache_key
    p = Params()
    before = connectivity_config(p, -22.8, 2, git_commit="same-commit")
    setattr(p, field, getattr(p, field)*.5)
    after = connectivity_config(p, -22.8, 2, git_commit="same-commit")
    assert cache_key(before) != cache_key(after)
