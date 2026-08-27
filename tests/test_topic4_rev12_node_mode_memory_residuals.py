import numpy as np

from scripts.audit_topic4_rev12_node_mode_memory_residuals import (
    block_permutation_audit,
    classify_mode_memory,
    conditional_size_iei_permutation_audit,
    local_occupancy_permutation_audit,
    ordered_blocks,
    recording_block_bootstrap,
    sequence_statistics,
    summarize_candidate,
)


def test_ordered_blocks_never_bridge_recording_boundaries():
    sequences = ordered_blocks(
        np.array([0, 1, 1, 0, 1, 0]),
        np.array([0, 0, 1, 1, 1, 0]),
        np.array([2.0, 1.0, 3.0, 1.0, 2.0, 3.0]),
        minimum_events=3,
    )
    assert [row.tolist() for row in sequences] == [[1, 0, 0], [0, 1, 1]]
    assert sequence_statistics(sequences)["transition_counts"].sum() == 4


def test_block_permutation_detects_strong_mode_memory():
    sequences = [np.r_[np.zeros(40, int), np.ones(40, int)] for _ in range(3)]
    audit = block_permutation_audit(sequences, draws=256, seed=7)
    assert audit["same_mode_null"]["excess_over_null_median"] > 0.4
    assert audit["same_mode_null"]["upper_tail_p"] < 0.01
    assert audit["permutation_endpoints"]["lag2_same_fraction"][
        "excess_over_null_median"
    ] > 0.4
    assert audit["permutation_endpoints"]["run_length_mean"][
        "excess_over_null_median"
    ] > 10
    assert audit["permutation_endpoints"]["run_fraction_ge_3"][
        "upper_tail_p"
    ] < 0.01
    assert audit["permutation_endpoints"]["run_fraction_ge_5"][
        "upper_tail_p"
    ] < 0.01


def test_gap_sensitivity_uses_only_well_separated_adjacent_events():
    labels = [np.array([0, 0, 1, 1])]
    starts = [np.array([0.0, 0.2, 2.0, 2.2])]
    ends = [np.array([0.1, 1.9, 2.1, 2.3])]
    audit = block_permutation_audit(
        labels, draws=64, seed=9, time_sequences=starts,
        end_time_sequences=ends,
        gap_thresholds=[0.1, 1.0],
    )
    assert audit["gap_sensitivity"]["0.1"]["n_pairs"] == 3
    assert audit["gap_sensitivity"]["1.0"]["n_pairs"] == 0
    assert audit["start_to_start_interval_sensitivity"]["1.0"]["n_pairs"] == 1
    assert audit["gap_definition"] == (
        "next_event_start_seconds - previous_event_end_seconds"
    )


def test_conditional_size_iei_null_absorbs_size_mediated_serial_order():
    labels = np.r_[np.zeros(20, int), np.ones(20, int)]
    starts = np.arange(40, dtype=float)
    records = [{
        "block": "b0",
        "labels": labels,
        "start_times": starts,
        "end_times": starts + 0.2,
        "event_sizes": np.where(labels == 0, 3, 4),
    }]
    audit = conditional_size_iei_permutation_audit(
        records, draws=64, seed=11, maximum_lag=2,
    )
    assert audit["same_mode_null"]["excess_over_null_median"] == 0.0
    assert audit["iei_role"] == "conditioning_diagnostic_not_a_post_end_gap"
    assert audit["n_iei_quantiles"] == 5


def test_local_occupancy_null_absorbs_slow_mode_occupancy_drift():
    labels = np.r_[np.zeros(40, int), np.ones(40, int)]
    starts = np.arange(80, dtype=float)
    records = [{
        "block": "b0",
        "labels": labels,
        "start_times": starts,
        "end_times": starts + 0.1,
        "event_sizes": np.full(80, 5),
    }]
    for window in (10.0, 30.0, 60.0, 300.0):
        audit = local_occupancy_permutation_audit(
            records, window_seconds=window, draws=64,
            seed=13 + int(window), maximum_lag=2,
        )
        assert audit["window_seconds"] == window
        if window == 10.0:
            assert audit["same_mode_null"]["excess_over_null_median"] == 0.0


def test_recording_block_bootstrap_reports_both_weighting_contracts():
    sequences = [
        np.r_[np.zeros(10, int), np.ones(10, int)] for _ in range(6)
    ]
    audit = recording_block_bootstrap(sequences, draws=128, seed=17)
    event_weighted = audit["event_weighted_same_mode_excess"]
    equal_block = audit["equal_block_same_mode_excess"]
    assert audit["bootstrap_unit"] == "recording_block"
    assert event_weighted["estimate"] > 0.4
    assert equal_block["estimate"] > 0.4
    assert event_weighted["ci95"][0] > 0
    assert equal_block["ci95"][0] > 0


def test_memory_classifier_rejects_persistent_state_when_heldout_fades_by_one_second():
    def row(excess, p):
        return {"excess_over_null_median": excess, "upper_tail_p": p}

    splits = {
        "train": {
            "same_mode_null": row(0.04, 0.001),
            "post_end_gap_sensitivity": {
                "0.5": row(0.03, 0.001), "1.0": row(0.03, 0.001),
            },
            "short_range_controls": {"all_confirmed": True},
        },
        "heldout": {
            "same_mode_null": row(0.03, 0.001),
            "post_end_gap_sensitivity": {
                "0.5": row(0.02, 0.001), "1.0": row(0.01, 0.06),
            },
            "short_range_controls": {"all_confirmed": True},
        },
    }
    decision = classify_mode_memory(
        splits, threshold=0.02, quantile=0.95,
        short_gap_seconds=0.5, persistent_gap_seconds=1.0,
    )
    assert decision["status"] == "PATIENT_MODE_MEMORY_CONFINED_TO_SUBSECOND_EVENT_HISTORY"
    assert decision["short_lived_activity_dependent_recovery_authorized"]
    assert decision["event_burst_autocorrelation_not_excluded"]
    assert not decision["persistent_state_authorized"]


def test_candidate_summary_preserves_four_components_and_precedence_classes():
    mode = {
        "recruitment": 1.0, "precedence": 2.0, "profile": 3.0, "cloud": 4.0,
        "raw": {"precedence_classes": {
            "ICL-ICL": 0.1, "SCL-SCL": 0.2, "ICL-SCL": 0.3,
        }},
        "effective_events": 6.0,
    }
    row = {
        "candidate_id": "candidate", "n_networks": 1, "mean_events": 12,
        "mean_soft_objective": 2.0, "mean_soft_causal_direction": 0.4,
        "mean_soft_causal_monotonicity": 0.5,
        "per_network": [{
            "soft_objective": {"modes": {"0": mode, "1": mode}},
            "soft_causal_direction": {"modes": {
                "0": {"alignment_score": 0.6},
                "1": {"alignment_score": 0.7},
            }},
        }],
    }
    summary = summarize_candidate(row)
    assert summary["modes"]["0"]["cloud"] == 4.0
    assert summary["modes"]["1"]["precedence_classes_raw"]["ICL-SCL"] == 0.3
    assert summary["modes"]["1"]["direction_alignment"] == 0.7
