"""Pre-outcome contracts for the FCXR-LC3 102-row frozen geometry."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

from model import build_network
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig
from params import Params
from src.topic4_fcxr_lc3_geometry import (
    H1_POINT_ID,
    H6_POINT_ID,
    PRIMARY_D_LABELS,
    build_geometry_manifest_rows,
    classify_geometry_tail,
    compact_checkpoint_diagnostics,
    configured_state_hash,
    extension_required,
    paired_field_shape_metrics,
    load_prepared_checkpoint,
    save_prepared_checkpoint,
    validate_geometry_manifest,
)


def test_paired_field_shape_metrics_exposes_same_mean_different_support():
    a = np.array([0.0, 0.0, 1.0, 1.0])
    b = np.array([0.5, 0.5, 0.5, 0.5])
    got = paired_field_shape_metrics(a, b)
    assert got["support_fraction_a"] == 0.5
    assert got["support_fraction_b"] == 1.0
    assert got["support_jaccard"] == 0.5
    assert got["relative_l2_difference"] > 0.0
    assert got["pearson_cellwise"] is None
from src.topic4_fcxr_lc3 import run_fcxr_loop


def _case(seed=17, *, frozen=False):
    p = Params(L=4.0, density=80.0, T=120.0, dt=0.1, nu_ext_ratio=0.92, seed=seed)
    net = build_network(p, verbose=False)
    ne, n = net["NE"], net["NE"] + net["NI"]
    vth = np.full(n, p.V_th)
    vth[: min(8, ne)] -= 0.8
    zf = np.linspace(0.75, 1.0, ne) if frozen else None
    xf = np.linspace(0.7, 1.0, ne) if frozen else None
    cfg = MZSlowVarsConfig(
        membrane_mode="full_conductance", E_E=58.0, c_E=1.0,
        ff_conductance=False, rec_conductance=True, rec_sat_g=21.6,
        v_match=18.0, e_gaba=0.0, e_k=0.0,
        max_total_conductance=99.0, fail_on_clip=True,
        use_h_lc2=True, tau_h_lc2=80.0, theta_h_lc2=0.03,
        k_h_lc2=0.02, rho_h_lc2=0.2,
        use_x=True, x_relay_frozen_E=xf, tau_y=120.0, tau_x=800.0,
        x_min=0.1, y_gate=2.0, K_y=4.0, hill_n=4,
        use_z=not frozen, z_frozen_E=zf, I_th_EI=1.0, tau_z=3000.0,
    )
    slow = MZSlowVars(n, 18.0, cfg, NE=ne, core_mask_E=np.zeros(ne, bool))
    net["rng"] = np.random.default_rng(seed)
    return p, net, slow, vth


def _fields():
    return {label: dict(field_sha256=f"field-{label}", source_path=f"/{label}.npz",
                        source_sha256=f"source-{label}") for label in PRIMARY_D_LABELS}


def _states():
    return {(point, state): f"state-{point}-{state}"
            for point in (H1_POINT_ID, H6_POINT_ID) for state in ("low", "high")}


def test_manifest_is_exactly_84_primary_plus_18_sentinel_before_outcomes():
    rows = build_geometry_manifest_rows(
        fields=_fields(), prepared_state_hashes=_states(), output_root="/tmp/lc3")
    audit = validate_geometry_manifest(rows)
    assert audit == dict(status="PASS", n_rows=102, n_h1=84, n_h6=18,
                         n_low=51, n_high=51,
                         schema="fcxr-lc3-geometry-contract-1.0")
    assert len({r["row_id"] for r in rows}) == 102
    assert sum(r["sentinel"] for r in rows) == 18
    assert all(r["noise_seed"] == 401 and r["no_kick"] for r in rows)


def test_manifest_fails_closed_on_missing_field_or_state_hash():
    fields = _fields(); fields.pop("D30")
    with pytest.raises(ValueError, match="six primary"):
        build_geometry_manifest_rows(fields=fields, prepared_state_hashes=_states(),
                                     output_root="/tmp/lc3")
    states = _states(); states.pop((H6_POINT_ID, "high"))
    with pytest.raises(ValueError, match="low/high exactly"):
        build_geometry_manifest_rows(fields=_fields(), prepared_state_hashes=states,
                                     output_root="/tmp/lc3")


def test_configured_hash_binds_h_parameters_ignored_by_dynamic_hash():
    p, net, slow, vth = _case(frozen=True)
    state = run_fcxr_loop(p, net, slow=slow, n_steps=10, capture_final=True,
                          store_spikes=False, v_th_per_neuron=vth)["checkpoint"]
    other = compact_checkpoint_diagnostics(state)
    other.slow.cfg.theta_h_lc2 += 0.1
    assert configured_state_hash(other) != configured_state_hash(state)


def test_compacted_checkpoint_roundtrip_resumes_exactly(tmp_path):
    p, net, slow, vth = _case(frozen=True)
    first = run_fcxr_loop(p, net, slow=slow, n_steps=60, capture_final=True,
                          store_spikes=False, v_th_per_neuron=vth)["checkpoint"]
    path = tmp_path / "prepared.pkl"
    record = save_prepared_checkpoint(str(path), first, metadata={"kind": "test"})
    loaded = load_prepared_checkpoint(str(path), expected_file_sha256=record["file_sha256"])
    assert loaded["metadata"] == {"kind": "test"}
    assert configured_state_hash(loaded["state"]) == record["configured_state_hash"]

    # Use separate but identical networks because each continuation consumes RNG state.
    p1, net1, _slow1, vth1 = _case(frozen=True)
    a = run_fcxr_loop(p1, net1, start=first, n_steps=50, capture_final=True,
                      store_spikes=True, v_th_per_neuron=vth1)
    p2, net2, _slow2, vth2 = _case(frozen=True)
    b = run_fcxr_loop(p2, net2, start=loaded["state"], n_steps=50, capture_final=True,
                      store_spikes=True, v_th_per_neuron=vth2)
    np.testing.assert_array_equal(a["rate_E"], b["rate_E"])
    np.testing.assert_array_equal(a["E_spk_bool"], b["E_spk_bool"])
    assert configured_state_hash(a["checkpoint"]) == configured_state_hash(b["checkpoint"])


def _classify(rate, *, counts=None, h=None, start=0.0, finite=True, clip=0.0,
              tau_eff=3.0):
    rate = np.asarray(rate, float)
    return classify_geometry_tail(
        rate_hz=rate, dt_ms=1.0, baseline_roll_hi_hz=10.0,
        analysis_start_ms=start,
        per_cell_tail_spike_counts=np.zeros(100) if counts is None else counts,
        tail_duration_ms=1000.0, tau_ref_e_ms=2.0,
        h_mean_trace=np.ones(rate.size) if h is None else h,
        theta_h=1.0, finite=finite, clip_frac_max=clip,
        tau_eff_min_ms=tau_eff,
    )


def test_classifier_separates_interictal_finite_high_and_tonic_ceiling():
    low = _classify(np.zeros(1500))
    high = _classify(np.full(1500, 60.0), h=np.full(1500, 2.0))
    tonic_counts = np.zeros(100); tonic_counts[:5] = 450.0
    tonic = _classify(np.full(1500, 100.0), counts=tonic_counts,
                      h=np.full(1500, 2.0))
    assert low["label"] == "INTERICTAL_WORKPOINT"
    assert high["label"] == "FINITE_HIGH_FIXED"
    assert tonic["label"] == "SATURATED_TONIC_BRANCH"


def test_decaying_h_with_high_rate_is_unresolved_not_a_branch():
    h = np.linspace(2.0, 0.2, 1500)
    out = _classify(np.full(1500, 60.0), h=h)
    assert out["workpoint_label"] == "FINITE_HIGH_FIXED"
    assert out["label"] == "HIGH_RATE_H_DECAY_UNRESOLVED"


def test_numerical_failure_has_priority_and_never_requests_extension():
    out = _classify(np.full(1500, 60.0), clip=0.01)
    assert out["label"] == "NUMERICAL_UNSAFE"
    assert not extension_required(state_kind="high", label=out["label"])


@pytest.mark.parametrize(
    "state,label,expected",
    [
        ("low", "INTERICTAL_WORKPOINT", False),
        ("low", "FINITE_HIGH_FIXED", True),
        ("low", "ELEVATED_EVENT_TRAIN", True),
        ("high", "FINITE_HIGH_ORBIT", False),
        ("high", "INTERICTAL_WORKPOINT", True),
        ("high", "METASTABLE_TRANSIENT", True),
        ("high", "SATURATED_TONIC_BRANCH", False),
    ],
)
def test_extension_rule_is_state_relative_and_outcome_independent(state, label, expected):
    assert extension_required(state_kind=state, label=label) is expected
