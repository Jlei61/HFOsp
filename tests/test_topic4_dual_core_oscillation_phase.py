import numpy as np
from scipy import sparse

from src.topic4_dual_core_oscillation_phase import (
    classify_coarse_trajectory,
    coarsen_delay_operators,
    contact_oscillation_assay,
    population_cycle_modulation,
)
from src.topic4_dual_core_spatial_z_delay import CoarseDelayOperators


def test_delay_coarsening_conserves_each_pathway_weight():
    blocks = [sparse.csr_matrix([[float(value)]]) for value in range(1, 8)]
    history = sparse.hstack(blocks, format="csr")
    operators = CoarseDelayOperators(
        dt_ms=0.1, max_delay_steps=7,
        w_ee_history=history,
        w_ei_history=2 * history,
        w_ie_history=3 * history,
        w_ii_history=4 * history,
    )
    coarse = coarsen_delay_operators(operators, factor=5)
    assert coarse.dt_ms == 0.5
    assert coarse.max_delay_steps == 1
    assert np.isclose(coarse.w_ee_history.sum(), history.sum())
    assert np.isclose(coarse.w_ii_history.sum(), 4 * history.sum())


def test_cycle_modulation_recovers_deep_40_hz_population_rhythm():
    dt = 0.5
    time = np.arange(0.0, 1000.0, dt) / 1000.0
    rate = 180.0 + 72.0 * np.sin(2.0 * np.pi * 40.0 * time)
    result = population_cycle_modulation(rate, dt_ms=dt)
    assert result["dominant_hz"] == 40.0
    assert result["modulation_depth"] > 0.7


def test_classifier_separates_oscillatory_from_tonic_recruited_state():
    dt = 0.5
    time = np.arange(0.0, 1500.0, dt) / 1000.0
    oscillatory = 180.0 + 55.0 * np.sin(2.0 * np.pi * 40.0 * time)
    tonic = np.full_like(oscillatory, 180.0)
    regional = [170.0, 165.0, 150.0]
    first = classify_coarse_trajectory(
        oscillatory, dt_ms=dt, regional_tail_rate_hz=regional)
    second = classify_coarse_trajectory(
        tonic, dt_ms=dt, regional_tail_rate_hz=regional)
    assert first["state"] == "oscillatory_recruited"
    assert second["state"] == "tonic_recruited"


def test_classifier_does_not_call_a_recruited_core_low():
    dt = 0.5
    rate = np.full(int(1000.0 / dt), 40.0)
    result = classify_coarse_trajectory(
        rate, dt_ms=dt, regional_tail_rate_hz=[200.0, 70.0, 15.0])
    assert result["state"] == "intermediate"


def test_contact_assay_requires_target_peak_and_rms_increase():
    dt = 0.5
    base_time = np.arange(0.0, 500.0, dt) / 1000.0
    late_time = np.arange(0.0, 1000.0, dt) / 1000.0
    baseline = np.column_stack([
        np.sin(2.0 * np.pi * 40.0 * base_time),
        np.sin(2.0 * np.pi * 40.0 * base_time),
        np.sin(2.0 * np.pi * 12.0 * base_time),
    ])
    late = np.column_stack([
        3.0 * np.sin(2.0 * np.pi * 40.0 * late_time),
        0.5 * np.sin(2.0 * np.pi * 40.0 * late_time),
        3.0 * np.sin(2.0 * np.pi * 12.0 * late_time),
    ])
    result = contact_oscillation_assay(baseline, late, dt_ms=dt)
    assert result["n_dominant_in_target_band"] == 2
    assert result["n_band_rms_increased"] == 2
    assert result["n_contacts_passing_both"] == 1
    assert result["passing_contact_mask"] == [True, False, False]
    assert result["n_persistent_contacts"] == 1
    assert result["persistent_contact_mask"] == [True, False, False]


def test_contact_assay_rejects_sparse_bursts_hidden_by_coarse_windows():
    dt = 0.5
    base_time = np.arange(0.0, 500.0, dt) / 1000.0
    late_ms = np.arange(0.0, 1000.0, dt)
    late_time = late_ms / 1000.0
    baseline = np.sin(2.0 * np.pi * 40.0 * base_time)[:, None]
    envelope = np.zeros_like(late_ms)
    for start in (20.0, 270.0, 520.0, 770.0):
        envelope[(late_ms >= start) & (late_ms < start + 70.0)] = 1.0
    late = (4.0 * envelope * np.sin(
        2.0 * np.pi * 40.0 * late_time))[:, None]
    result = contact_oscillation_assay(baseline, late, dt_ms=dt)
    assert result["coarse_persistent_contact_mask"] == [True]
    assert result["fine_persistent_contact_mask"] == [False]
    assert result["persistent_contact_mask"] == [False]
