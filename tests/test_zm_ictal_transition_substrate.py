"""Substrate rebuild must reproduce the frozen rev11-NLC substrate exactly."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))
sys.path.insert(0, str(ROOT))

from src.topic4_zm_ictal_transition import (  # noqa: E402
    build_substrate, load_round_config, verify_frozen_inputs)

CONFIG = ROOT / "config/topic4_data_driven_zm_ictal_transition_v1.json"
ARCHIVE = ROOT / ("results/topic4_sef_hfo/data_driven_local_connectivity_rev11_nlc"
                  "/frozen_substrate_confirmation/workers")
CACHE = ROOT / "results/topic4_sef_hfo/data_driven_zm_ictal_transition/network_cache"


def test_frozen_inputs_verify():
    config = load_round_config(CONFIG)
    report = verify_frozen_inputs(config)
    assert report["all_match"] is True
    assert len(report["records"]) >= 14


def test_frozen_inputs_raise_on_drift(tmp_path):
    config = load_round_config(CONFIG)
    config["inputs"]["zm_baseline"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="input hash changed"):
        verify_frozen_inputs(config)


@pytest.mark.slow
@pytest.mark.integration
def test_seed_1561_substrate_matches_archive():
    config = load_round_config(CONFIG)
    sub = build_substrate(config, "joint_04_control", 1561, cache_dir=str(CACHE))
    archived = json.loads((ARCHIVE / "joint_04_control_seed_1561.json").read_text())

    assert sub.n_e == 32000 and sub.n_i == 8000
    assert sub.network_cache["cache_sha256"] == archived["network"]["cache_source"]["cache_sha256"]
    assert sub.network_cache["frozen_cache_key"] == archived["network"]["cache_source"]["frozen_cache_key"]

    with np.load(ARCHIVE / "joint_04_control_seed_1561.npz", allow_pickle=False) as z:
        assert np.array_equal(sub.h_e.astype(np.float32), z["h"])
        assert np.array_equal(sub.positions_e.astype(np.float32), z["positions_E"])
        assert np.array_equal(sub.delta_vtheta.astype(np.float32), z["delta_vtheta"])
        assert np.array_equal(sub.h_i.astype(np.float32), z["h_I_for_edge"])
        assert np.array_equal(sub.edge_coefficients, z["edge_coefficients"])
        assert list(sub.contact_names) == list(z["contact_names"])
        assert np.array_equal(sub.contact_xy, z["contact_xy_mm"])
    assert sub.edge_audit["coefficients_sha256"] == archived["edge_audit"]["coefficients_sha256"]
    assert np.isclose(sub.h_e.sum(), 1129.0, atol=1e-8)
    assert np.allclose(sub.axis_unit, [0.92182673, -0.38760221], atol=1e-8)
    assert bool(np.all(sub.valid_contacts))


@pytest.mark.slow
@pytest.mark.integration
def test_pathway_gains_are_outgoing_and_not_conserved():
    """Incoming budget is conserved per target by contract, so only the
    OUTGOING side carries the mapper's effect."""
    config = load_round_config(CONFIG)
    sub = build_substrate(config, "joint_04_control", 1561, cache_dir=str(CACHE))
    for gain in (sub.ee_out_gain, sub.etoi_out_gain):
        finite = gain[np.isfinite(gain)]
        assert finite.size > 0.9 * sub.n_e
        assert finite.min() < 0.99 and finite.max() > 1.01   # genuinely redistributed
        assert np.isfinite(finite).all()


@pytest.mark.slow
@pytest.mark.integration
def test_node_baseline_arm_has_a_noop_edge_mapper():
    config = load_round_config(CONFIG)
    sub = build_substrate(config, "node_baseline", 1561, cache_dir=str(CACHE))
    assert np.allclose(sub.edge_coefficients, 0.0)
    assert np.allclose(sub.ee_out_gain[np.isfinite(sub.ee_out_gain)], 1.0, atol=1e-9)


@pytest.mark.slow
@pytest.mark.integration
def test_transformed_substrate_preserves_field_mass_and_leaves_geometry_fixed():
    config = load_round_config(CONFIG)
    plain = build_substrate(config, "joint_04_control", 1561, cache_dir=str(CACHE))
    rotated = build_substrate(config, "joint_04_control", 1561, cache_dir=str(CACHE),
                              field_transform="r180")
    assert np.isclose(rotated.h_e.sum(), 1129.0, atol=1e-8)
    assert not np.allclose(rotated.h_e, plain.h_e)
    # the control moves the FIELD, never the geometry it is registered against
    assert np.array_equal(rotated.contact_xy, plain.contact_xy)
    assert np.allclose(rotated.axis_unit, plain.axis_unit)
    assert np.array_equal(rotated.positions_e, plain.positions_e)
    bounds = np.array([0.5, 0.5, 0.15, 0.15, 0.15, 0.15])
    assert np.all(np.abs(rotated.edge_coefficients) <= bounds + 1e-12)
    assert np.allclose(np.linalg.norm(rotated.edge_coefficients[:, 4:], axis=1),
                       np.linalg.norm(plain.edge_coefficients[:, 4:], axis=1))
