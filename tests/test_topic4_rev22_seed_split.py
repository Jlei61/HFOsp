"""rev22-DCI Task 2: topology and dynamics seeds split without changing legacy output."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))
sys.path.insert(0, str(ROOT))

from src.topic4_rev20_dual_core_mechanism import _hash_sparse_bins  # noqa: E402
from src.topic4_zm_ictal_transition import build_substrate, load_round_config  # noqa: E402

CONFIG = ROOT / "config/topic4_data_driven_zm_ictal_transition_v1.json"
SHARED_CACHE = ARTIFACT_ROOT / "results/topic4_sef_hfo/data_driven_core_field_rev9/network_cache"
CANDIDATE = "joint_04_control"
SEED_A, SEED_B = 2521, 2522   # both networks are cached from the rev20 confirmation runs


def _build(seed, **kw):
    return build_substrate(
        load_round_config(CONFIG), CANDIDATE, seed,
        cache_dir=str(SHARED_CACHE), artifact_root=ARTIFACT_ROOT, **kw,
    )


def _graph_hashes(sub):
    return {
        "topology": _hash_sparse_bins(sub.net["ampa_by_delay"], include_data=False),
        "ampa": _hash_sparse_bins(sub.net["ampa_by_delay"]),
        "gaba": _hash_sparse_bins(sub.net["gaba_by_delay"]),
        "delays": int(sub.net["max_delay_steps"]),
    }


def _rng_state(sub):
    return sub.net["rng"].bit_generator.state["state"]["state"]


@pytest.fixture(scope="module")
def legacy():
    return _build(SEED_A)


def test_default_path_is_legacy_mode(legacy):
    assert legacy.seed_mode == "legacy"
    assert legacy.topology_seed == SEED_A and legacy.dynamics_seed == SEED_A
    assert legacy.params.seed == SEED_A
    assert legacy.network_cache["hit"] is True
    assert legacy.extras["seed_contract"] == {
        "seed": SEED_A, "topology_seed": SEED_A, "dynamics_seed": SEED_A, "seed_mode": "legacy",
    }


@pytest.mark.slow
@pytest.mark.integration
def test_equal_explicit_seeds_reproduce_legacy_substrate(legacy):
    same = _build(SEED_A, topology_seed=SEED_A, dynamics_seed=SEED_A)
    assert same.seed_mode == "legacy"
    assert same.network_cache["cache_sha256"] == legacy.network_cache["cache_sha256"]
    assert _graph_hashes(same) == _graph_hashes(legacy)
    assert np.array_equal(same.positions_e, legacy.positions_e)
    assert np.array_equal(same.h_e, legacy.h_e)
    assert np.array_equal(same.delta_vtheta, legacy.delta_vtheta)
    assert np.array_equal(same.edge_coefficients, legacy.edge_coefficients)
    assert _rng_state(same) == _rng_state(legacy)


@pytest.mark.slow
@pytest.mark.integration
def test_dynamics_seed_changes_only_the_rng_stream(legacy):
    split = _build(SEED_A, topology_seed=SEED_A, dynamics_seed=SEED_B)
    assert split.seed_mode == "split"
    assert split.params.seed == SEED_A
    assert split.network_cache["cache_sha256"] == legacy.network_cache["cache_sha256"]
    assert _graph_hashes(split) == _graph_hashes(legacy)
    assert np.array_equal(split.positions_e, legacy.positions_e)
    assert np.array_equal(split.h_e, legacy.h_e)
    assert np.array_equal(split.delta_vtheta, legacy.delta_vtheta)
    assert np.array_equal(split.edge_coefficients, legacy.edge_coefficients)
    assert _rng_state(split) != _rng_state(legacy)
    assert _rng_state(split) == _rng_state(_build(SEED_B))


@pytest.mark.slow
@pytest.mark.integration
def test_topology_seed_changes_the_graph_but_not_the_parameter_contract(legacy):
    split = _build(SEED_B, topology_seed=SEED_B, dynamics_seed=SEED_A)
    assert split.seed_mode == "split"
    assert split.params.seed == SEED_B
    assert split.network_cache["hit"] is True
    assert split.network_cache["cache_sha256"] != legacy.network_cache["cache_sha256"]
    assert _graph_hashes(split)["topology"] != _graph_hashes(legacy)["topology"]
    assert not np.array_equal(split.positions_e, legacy.positions_e)
    # frozen parameter contract is unchanged
    assert np.array_equal(split.edge_coefficients, legacy.edge_coefficients)
    assert split.extras["frozen_node_field_sha256"] == legacy.extras["frozen_node_field_sha256"]
    assert split.extras["pathway_dose"] == legacy.extras["pathway_dose"]
    assert split.extras["ellipse_audit"]["exact_noop"] == legacy.extras["ellipse_audit"]["exact_noop"]
    assert np.isclose(split.h_e.sum(), legacy.h_e.sum(), atol=1e-8)
    assert _rng_state(split) == _rng_state(legacy)
