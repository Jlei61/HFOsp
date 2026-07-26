import pandas as pd

from scripts.analyze_topic5_persistent_path_pilot import (
    METRICS,
    _count_gate,
    _mode_gate,
)


def test_count_gate_requires_both_patient_seed_and_subject_majority():
    frame = pd.DataFrame(
        {
            "subject": ["a"] * 3 + ["b"] * 3 + ["c"] * 3,
            "benefit": [1, 1, 1, 1, 1, -1, 1, -1, -1],
        }
    )
    result = _count_gate(frame, min_patient_seed=6, min_subjects=2)
    assert result["n_patient_seed_better"] == 6
    assert result["n_subject_median_better"] == 2
    assert result["pass"]
    frame.loc[3, "benefit"] = -1
    assert not _count_gate(
        frame, min_patient_seed=6, min_subjects=2
    )["pass"]


def test_mode_gate_requires_all_controls_stability_and_structure_lesion():
    comparison = pd.DataFrame(
        [
            {
                "mode_count": 2,
                "baseline": baseline,
                "metric": metric,
                "pass": True,
            }
            for baseline in (
                "no_history",
                "merged_path",
                "weight_shuffle",
                "mode_shuffle",
            )
            for metric in METRICS
        ]
    )
    stability = pd.DataFrame(
        [
            {"metric": metric, "pass": True}
            for metric in METRICS
        ]
    )
    lesion = pd.DataFrame(
        [
            {
                "mode_count": 2,
                "lesion": lesion_name,
                "metric": metric,
                "pass": lesion_name == "mode_collapse",
            }
            for lesion_name in ("mode_collapse", "drop_dominant_mode")
            for metric in METRICS
        ]
    )
    result = _mode_gate(2, comparison, stability, lesion)
    assert result["hard_gate_pass"]
    comparison.loc[
        (comparison.baseline == "mode_shuffle")
        & (comparison.metric == "precedence_mae"),
        "pass",
    ] = False
    assert not _mode_gate(
        2, comparison, stability, lesion
    )["hard_gate_pass"]
