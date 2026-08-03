"""Pre-outcome contracts for the FCXR-LC3 102-row frozen geometry."""
from __future__ import annotations

import json
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
    EXTENDED_ROW_RSS_SCALE,
    H1_POINT_ID,
    H6_POINT_ID,
    MAP_WORKER_MEM_FLOOR_GIB,
    MAP_WORKER_RESERVE,
    MAX_MAP_WORKERS,
    PRIMARY_D_LABELS,
    build_geometry_manifest_rows,
    choose_map_workers,
    classify_geometry_tail,
    compact_checkpoint_diagnostics,
    configured_state_hash,
    extension_required,
    geometry_manifest_summary,
    install_registered_noise_rng,
    REGISTERED_NOISE_SEED,
    paired_field_shape_metrics,
    load_prepared_checkpoint,
    prepared_state_is_reusable,
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
from src.topic4_fcxr_lc3 import clone_loop_state, run_fcxr_loop


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


def test_manifest_summary_nests_the_audit_instead_of_shadowing_its_status():
    rows = build_geometry_manifest_rows(
        fields=_fields(), prepared_state_hashes=_states(), output_root="/tmp/lc3")
    audit = validate_geometry_manifest(rows)
    # The 2026-08-04 map launch died here: the audit already owns ``status``, so
    # splatting it beside a second ``status`` raised TypeError *after* the 102-row
    # manifest had been written.  The summary must keep both and stay serialisable.
    with pytest.raises(TypeError):
        dict(status="LOCKED", **audit)
    summary = geometry_manifest_summary(audit)
    assert summary["status"] == "LOCKED"
    assert summary["audit"]["status"] == "PASS"
    assert summary["audit"]["n_rows"] == 102
    assert json.loads(json.dumps(summary)) == summary


def _workers(mem_gib, rss_gib, *, swap=0.0, base=0.0, cpu=80):
    return choose_map_workers(mem_available_gib=mem_gib, swap_used_mib=swap,
                              swap_baseline_mib=base, single_rss_gib=rss_gib,
                              cpu_count=cpu)


def _registered_rule(mem_gib, rss_gib):
    """The two-worker rule this function generalises; it is the guaranteed floor."""
    return 2 if mem_gib >= MAP_WORKER_MEM_FLOOR_GIB + 2.0 * MAP_WORKER_RESERVE * rss_gib else 1


def test_map_workers_never_fall_below_the_registered_two_worker_rule():
    for mem_gib in (100.0, 140.0, 175.0, 225.0, 400.0):
        for rss_gib in (2.0, 5.0, 10.0, 12.0):
            assert _workers(mem_gib, rss_gib) >= _registered_rule(mem_gib, rss_gib)


def test_map_workers_only_exceed_two_when_the_worst_case_row_still_fits():
    for mem_gib in (100.0, 140.0, 175.0, 225.0, 400.0):
        for rss_gib in (2.0, 5.0, 10.0, 12.0):
            n = _workers(mem_gib, rss_gib)
            assert 1 <= n <= MAX_MAP_WORKERS
            if n > _registered_rule(mem_gib, rss_gib):
                reserved = n * MAP_WORKER_RESERVE * EXTENDED_ROW_RSS_SCALE * rss_gib
                assert reserved <= mem_gib - MAP_WORKER_MEM_FLOOR_GIB


def test_map_workers_collapse_to_one_on_rising_swap_or_no_headroom():
    assert _workers(225.0, 5.0, swap=1000.0, base=700.0) == 1   # +300 MiB swap
    assert _workers(96.0, 5.0) == 1                              # exactly at the floor
    assert _workers(60.0, 5.0) == 1                              # below the floor


def test_map_workers_are_bounded_by_cpu_and_by_the_hard_cap():
    assert _workers(4000.0, 1.0, cpu=6) == 4                     # cpu_count - 2
    assert _workers(4000.0, 1.0, cpu=80) == MAX_MAP_WORKERS


def test_map_workers_grow_with_memory_and_shrink_with_row_footprint():
    assert _workers(140.0, 5.0) < _workers(225.0, 5.0)
    assert _workers(225.0, 12.0) < _workers(225.0, 2.0)


def test_map_workers_reject_non_finite_or_empty_inputs():
    for bad in (dict(mem_gib=float("nan"), rss_gib=5.0),
                dict(mem_gib=225.0, rss_gib=0.0),
                dict(mem_gib=225.0, rss_gib=float("inf"))):
        with pytest.raises(ValueError, match="finite positive"):
            _workers(**bad)
    with pytest.raises(ValueError, match="finite positive"):
        _workers(225.0, 5.0, cpu=0)


def _prepared_record(**over):
    rec = dict(status="ACCEPTED_CANONICAL_STATE", point_id=H1_POINT_ID, state_kind="high",
               checkpoint=dict(file_sha256="abc"))
    rec.update(over)
    return rec


def test_accepted_prepared_state_is_resumed_instead_of_recomputed():
    assert prepared_state_is_reusable(_prepared_record(), point_id=H1_POINT_ID,
                                      state_kind="high", checkpoint_sha256="abc")
    injected = _prepared_record(status="ACCEPTED_SENTINEL_INJECTED_LOW_STATE",
                                point_id=H6_POINT_ID, state_kind="low")
    assert prepared_state_is_reusable(injected, point_id=H6_POINT_ID,
                                      state_kind="low", checkpoint_sha256="abc")


@pytest.mark.parametrize("record,sha", [
    (_prepared_record(status="PREPARED_STATE_UNRESOLVED"), "abc"),   # never reuse a reject
    (_prepared_record(), "drifted"),                                 # checkpoint changed on disk
    (_prepared_record(point_id=H6_POINT_ID), "abc"),                 # wrong point
    (_prepared_record(state_kind="low"), "abc"),                     # wrong basin
    (_prepared_record(checkpoint={}), "abc"),                        # no recorded hash
    (None, "abc"),
])
def test_unresolved_or_drifted_prepared_state_is_never_reused(record, sha):
    assert not prepared_state_is_reusable(record, point_id=H1_POINT_ID,
                                          state_kind="high", checkpoint_sha256=sha)


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


def test_registered_noise_rng_is_installed_with_the_manifest_seed():
    net = install_registered_noise_rng({})
    assert (net["rng"].standard_normal()
            == np.random.default_rng(REGISTERED_NOISE_SEED).standard_normal())
    rows = build_geometry_manifest_rows(
        fields=_fields(), prepared_state_hashes=_states(), output_root="/tmp/lc3")
    assert {r["noise_seed"] for r in rows} == {REGISTERED_NOISE_SEED}
    with pytest.raises(ValueError, match="substrate network dict"):
        install_registered_noise_rng(None)


def test_substrate_without_a_noise_generator_cannot_step():
    # build_substrate does not create net["rng"], so the geometry map's worker
    # context has to install it; without this the very first map row dies here.
    p, net, slow, vth = _case(frozen=True)
    del net["rng"]
    with pytest.raises(KeyError, match="rng"):
        run_fcxr_loop(p, net, slow=slow, n_steps=5, capture_final=False,
                      store_spikes=False, v_th_per_neuron=vth)
    install_registered_noise_rng(net)
    run_fcxr_loop(p, net, slow=slow, n_steps=5, capture_final=False,
                  store_spikes=False, v_th_per_neuron=vth)


def test_continuation_ignores_the_construction_seed_so_worker_count_cannot_move_a_row():
    # Every map row is a continuation, and the loop overwrites the bit-generator
    # state from its prepared checkpoint before its first draw.  This is what makes
    # the map worker count and submission order a pure resource choice.
    p0, net0, slow0, vth0 = _case(frozen=True)
    start = run_fcxr_loop(p0, net0, slow=slow0, n_steps=40, capture_final=True,
                          store_spikes=False, v_th_per_neuron=vth0)["checkpoint"]
    outs = []
    for noise_seed in (REGISTERED_NOISE_SEED, 999):
        p, net, _slow, vth = _case(frozen=True)  # identical connectivity, different rng
        install_registered_noise_rng(net, noise_seed=noise_seed)
        # Each row deep-clones its start state, exactly as replace_frozen_fields does;
        # stepping the same in-memory state twice would advance its slow counter.
        outs.append(run_fcxr_loop(p, net, start=clone_loop_state(start), n_steps=40,
                                  capture_final=True, store_spikes=True,
                                  v_th_per_neuron=vth))
    np.testing.assert_array_equal(outs[0]["rate_E"], outs[1]["rate_E"])
    np.testing.assert_array_equal(outs[0]["E_spk_bool"], outs[1]["E_spk_bool"])
    assert configured_state_hash(outs[0]["checkpoint"]) == configured_state_hash(
        outs[1]["checkpoint"])
