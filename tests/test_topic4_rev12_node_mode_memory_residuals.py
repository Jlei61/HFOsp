import numpy as np

from scripts.audit_topic4_rev12_node_mode_memory_residuals import (
    block_permutation_audit,
    classify_mode_memory,
    ordered_blocks,
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


def test_gap_sensitivity_uses_only_well_separated_adjacent_events():
    labels = [np.array([0, 0, 1, 1])]
    times = [np.array([0.0, 0.2, 2.0, 2.2])]
    audit = block_permutation_audit(
        labels, draws=64, seed=9, time_sequences=times,
        gap_thresholds=[0.1, 1.0],
    )
    assert audit["gap_sensitivity"]["0.1"]["n_pairs"] == 3
    assert audit["gap_sensitivity"]["1.0"]["n_pairs"] == 1
    assert audit["gap_sensitivity"]["1.0"]["same_mode_fraction"] == 0.0


def test_memory_classifier_rejects_persistent_state_when_heldout_fades_by_one_second():
    def row(excess, p):
        return {"excess_over_null_median": excess, "upper_tail_p": p}

    splits = {
        "train": {
            "same_mode_null": row(0.04, 0.001),
            "gap_sensitivity": {"0.5": row(0.03, 0.001), "1.0": row(0.03, 0.001)},
        },
        "heldout": {
            "same_mode_null": row(0.03, 0.001),
            "gap_sensitivity": {"0.5": row(0.02, 0.001), "1.0": row(0.01, 0.06)},
        },
    }
    decision = classify_mode_memory(
        splits, threshold=0.02, quantile=0.95,
        short_gap_seconds=0.5, persistent_gap_seconds=1.0,
    )
    assert decision["status"] == "PATIENT_MODE_MEMORY_CONFINED_TO_SUBSECOND_EVENT_HISTORY"
    assert decision["short_lived_activity_dependent_recovery_authorized"]
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
