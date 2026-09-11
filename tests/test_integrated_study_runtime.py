"""Cross-version integration must not silently substitute models or samplers."""
import inspect
import numpy as np
from src.snn_engine.params import Params
from src.topic4_corrected_graph_cache import get_network, connectivity_config
from src import topic4_legacy_dual_core_transition as legacy
from src import topic4_initial_state_substrate as initial
from src import topic4_zm_ictal_transition as default
from scripts import run_topic4_legacy_rev21_worker as old_worker
from scripts import run_topic4_rev12_node_worker as seeded_worker


def test_workers_use_their_model_version_and_default_entry_stays_separate():
    assert old_worker.build_substrate is legacy.build_substrate
    assert old_worker.make_slow is legacy.make_slow
    assert seeded_worker.build_substrate is initial.build_substrate
    assert 'topology_seed' in inspect.signature(seeded_worker.build_substrate).parameters
    assert default.build_substrate is not legacy.build_substrate
    assert default.build_substrate is not initial.build_substrate


def test_corrected_cache_has_no_self_edges_and_keeps_exact_degrees(tmp_path):
    p = Params(L=1., density=400., seed=9, C_EE=10, C_IE=10, C_EI=5, C_II=5)
    net, ne, ni, hit = get_network(p, 31., 2., str(tmp_path), git_commit='fixture')
    assert not hit
    ampa = sum(net['ampa_by_delay'])
    gaba = sum(net['gaba_by_delay'])
    assert np.count_nonzero(ampa[:ne].diagonal()) == 0
    assert np.count_nonzero(gaba[ne:].diagonal()) == 0
    np.testing.assert_array_equal(ampa.getnnz(axis=1), np.full(ne + ni, 10))
    np.testing.assert_array_equal(gaba.getnnz(axis=1), np.full(ne + ni, 5))
    cached, ne2, ni2, hit = get_network(p, 31., 2., str(tmp_path), git_commit='fixture')
    assert hit and (ne2, ni2) == (ne, ni)
    np.testing.assert_array_equal(net['pos'], cached['pos'])
    for name in ('ampa_by_delay', 'gaba_by_delay'):
        assert all((a != b).nnz == 0 for a, b in zip(net[name], cached[name]))
    assert connectivity_config(p, 31., 2., git_commit='fixture')['partner_sampler_version'] == 'positive_weight_no_autapse_v2'
