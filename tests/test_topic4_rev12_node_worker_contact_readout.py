from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from scripts.run_topic4_rev12_node_worker import (
    _directed_parent_contract,
    _event_contact_readout,
    _segmentation_variants,
    _validate_scientific_role,
)
from src.sef_hfo_observation import VirtualMontage
from src.topic4_node_dualmode import (
    binned_contact_envelope,
    excitatory_psp_tail_support_ms,
    local_ee_delay_quantile_ms,
    sheet_contact_sampling_weights,
)


def test_worker_uses_only_the_selected_directed_root_for_contact_ranks():
    counts = np.zeros((5, 3, 3), float)
    labels = np.zeros_like(counts, int)
    counts[:, 0, 0] = [0.0, 4.0, 8.0, 4.0, 0.0]
    labels[1:4, 0, 0] = 1
    counts[:, 0, 2] = [0.0, 20.0, 10.0, 20.0, 0.0]
    labels[1:4, 0, 2] = 2
    positions = np.vstack([
        np.repeat([[0.5, 0.5]], 20, axis=0),
        np.repeat([[1.5, 0.5]], 20, axis=0),
        np.repeat([[2.5, 0.5]], 20, axis=0),
    ])
    contacts = np.asarray([[0.5, 0.5], [2.5, 0.5]])
    montage = VirtualMontage(contacts, ["A1", "A2"], "test")
    population = np.zeros((3, 3), float)
    population[0] = 20.0
    weights = sheet_contact_sampling_weights(
        contacts, population,
        bin_mm=1.0, kernel_width_mm=0.1,
    )
    envelope = binned_contact_envelope(
        counts, weights, frame_ms=1.0, smooth_ms=0.01,
    )

    onsets, ranks, audit = _event_contact_readout(
        events=[{"cascade_id": 1, "t_on": 0.0, "t_off": 5.0}],
        envelope=envelope, envelope_dt_ms=1.0, montage=montage,
        valid_contacts=np.ones(2, bool), positions_e=positions,
        movie={
            "activity_counts": counts, "frame_ms": 1.0,
            "bin_mm": 1.0, "sheet_mm": 3.0,
        },
        lineage_labels=labels,
        readout={
            "source": "lineage_restricted_sheet_activity",
            "kernel_width_mm": 0.1, "smooth_ms": 0.01,
            "minimum_full_trace_pearson": 0.999,
            "participation_margin_fraction": 0.1,
            "timing_fraction": 0.5,
        },
    )

    assert np.isfinite(onsets[0, 0])
    assert np.isnan(onsets[0, 1])
    assert ranks[0, 0] == 0.0
    assert np.isnan(ranks[0, 1])
    assert audit["source"] == "lineage_restricted_sheet_activity"
    assert audit["parity_status"] == "PASS"


def test_worker_exact_neuron_readout_uses_the_original_contact_kernel():
    counts = np.zeros((5, 3, 3), float)
    labels = np.zeros_like(counts, int)
    labels[1:4, 0, 0] = 1
    labels[1:4, 0, 2] = 2
    positions = np.asarray([[0.5, 0.5], [1.5, 0.5], [2.5, 0.5]])
    montage = VirtualMontage(
        np.asarray([[0.5, 0.5], [2.5, 0.5]]), ["A1", "A2"], "test",
    )
    spikes = np.zeros((10, 3), bool)
    spikes[2:8:2, 0] = True
    spikes[2:8:2, 2] = True
    onsets, ranks, audit = _event_contact_readout(
        events=[{"cascade_id": 1, "t_on": 0.0, "t_off": 10.0}],
        envelope=np.zeros((2, 5)), envelope_dt_ms=2.0, montage=montage,
        valid_contacts=np.ones(2, bool), positions_e=positions,
        movie={
            "activity_counts": counts, "frame_ms": 2.0,
            "bin_mm": 1.0, "sheet_mm": 3.0,
        },
        lineage_labels=labels, spikes=spikes, spike_dt_ms=1.0,
        readout={
            "source": "lineage_restricted_neuron_activity",
            "kernel_width_mm": 0.1, "smooth_ms": 0.01,
            "participation_margin_fraction": 0.1,
            "timing_fraction": 0.5,
        },
    )
    assert np.isfinite(onsets[0, 0]) and np.isnan(onsets[0, 1])
    assert ranks[0, 0] == 0.0 and np.isnan(ranks[0, 1])
    assert audit["parity_status"] == "EXACT_SHARED_PER_NEURON_KERNEL"


def test_persistent_lineage_memory_comes_from_fast_state_and_delay():
    contract = _directed_parent_contract(
        {
            "name": "persistent_directed_spatiotemporal_lineage",
            "fast_state_decay_multiples": 5.0,
            "forward_parent_neighborhood_bins": 1,
        },
        params=SimpleNamespace(tau_m_E=20.0, tau_d_GABA=18.0),
        net={"max_delay_steps": 50}, engine={"dt": 0.1}, frame_ms=2.0,
    )
    assert contract["causal_memory_ms"] == 105.0
    assert contract["forward_parent_frame_gap"] == 53
    assert contract["forward_parent_neighborhood_bins"] == 1


def test_local_ee_delay_uses_only_edges_inside_the_movie_parent_cone():
    positions = np.asarray([[0.2, 0.2], [1.2, 0.2], [4.2, 0.2]])
    matrices = [sparse.csr_matrix((3, 3)) for _ in range(5)]
    matrices[2] = sparse.csr_matrix((
        np.asarray([1.0]), (np.asarray([1]), np.asarray([0])),
    ), shape=(3, 3))
    matrices[4] = sparse.csr_matrix((
        np.asarray([1.0]), (np.asarray([2]), np.asarray([0])),
    ), shape=(3, 3))
    assert local_ee_delay_quantile_ms(
        matrices, positions, dt_ms=0.1, bin_mm=1.0,
        neighborhood_bins=1, quantile=1.0,
    ) == 0.2


def test_psp_tail_memory_uses_excitatory_path_and_local_delay():
    psp_ms = excitatory_psp_tail_support_ms(
        tau_r_ms=0.7, tau_d_ms=3.5, tau_m_ms=20.0,
        dt_ms=0.1, tail_fraction=0.1,
    )
    assert 57.0 < psp_ms < 59.0
    matrices = [sparse.csr_matrix((2, 2)) for _ in range(4)]
    matrices[3] = sparse.csr_matrix(np.asarray([[0.0, 1.0], [0.0, 0.0]]))
    contract = _directed_parent_contract(
        {
            "name": "persistent_root_coactivity_episode",
            "causal_memory_method": "local_ee_psp_tail",
            "psp_tail_fraction": 0.1,
            "local_ee_delay_quantile": 1.0,
            "movie_bin_mm": 1.0,
            "forward_parent_neighborhood_bins": 1,
        },
        params=SimpleNamespace(
            tau_m_E=20.0, tau_r_AMPA=0.7, tau_d_AMPA=3.5,
        ),
        net={
            "max_delay_steps": 50, "NE": 2,
            "pos": np.asarray([[0.2, 0.2], [1.2, 0.2]]),
            "ampa_by_delay": matrices,
        },
        engine={"dt": 0.1}, frame_ms=2.0,
    )
    assert np.isclose(contract["local_or_global_delay_support_ms"], 0.3)
    assert np.isclose(contract["causal_memory_ms"], psp_ms + 0.3)
    assert contract["forward_parent_frame_gap"] == int(np.ceil((psp_ms + 0.3) / 2.0))


def test_worker_accepts_engine_derived_event_identity_canary_role():
    _validate_scientific_role(
        "development_only_engine_derived_event_identity_canary"
    )
    _validate_scientific_role(
        "development_only_full_fit_replay_after_event_identity_fix"
    )
    _validate_scientific_role(
        "development_only_causal_continuation_and_capacity"
    )
    _validate_scientific_role(
        "development_only_global_continuous_node_field_screen"
    )
    _validate_scientific_role(
        "development_only_scalar_node_gain_canary"
    )
    with pytest.raises(RuntimeError, match="scientific role"):
        _validate_scientific_role("patient_selected_event_window")


def test_causal_root_sensitivity_changes_one_axis_at_a_time():
    variants = _segmentation_variants({
        "causal_memory_method": "local_ee_psp_tail",
        "psp_tail_fraction": 0.5,
        "sensitivity_psp_tail_fractions": [0.8, 0.5, 0.2],
        "minimum_dominance": 0.7,
        "sensitivity_minimum_dominances": [0.6, 0.7, 0.8],
    })
    assert len(variants) == 5
    assert sum(row["is_primary"] for row in variants) == 1
    assert {(row["memory_value"], row["minimum_dominance"]) for row in variants} == {
        (0.8, 0.7), (0.5, 0.7), (0.2, 0.7),
        (0.5, 0.6), (0.5, 0.8),
    }


def test_edge_supported_sensitivity_changes_one_axis_at_a_time():
    variants = _segmentation_variants({
        "name": "edge_supported_causal_family_observation",
        "causal_memory_method": "local_ee_psp_tail",
        "psp_tail_fraction": 0.5,
        "sensitivity_psp_tail_fractions": [0.8, 0.5, 0.2],
        "minimum_dominance": 0.7,
        "sensitivity_minimum_dominances": [0.6, 0.7, 0.8],
        "minimum_parent_support": 0.001,
        "sensitivity_minimum_parent_supports": [0.0003, 0.001, 0.003],
        "edge_delay_rounding": "nearest",
        "sensitivity_edge_delay_roundings": ["floor", "nearest", "ceil"],
    })
    assert len(variants) == 9
    assert sum(row["is_primary"] for row in variants) == 1
    primary = next(row for row in variants if row["is_primary"])
    assert primary == {
        "memory_parameter": "psp_tail_fraction",
        "memory_value": 0.5,
        "minimum_dominance": 0.7,
        "minimum_parent_support": 0.001,
        "edge_delay_rounding": "nearest",
        "is_primary": True,
    }
