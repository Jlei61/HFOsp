from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analyze_topic5_persistent_path_internal_dynamics import (
    _sample_eval_indices,
    _summarize_trajectories,
)


def test_sample_eval_indices_spans_full_chronology_without_duplicates() -> None:
    indices = np.arange(100, 300)
    selected = _sample_eval_indices(indices, 17)
    assert len(selected) == 17
    assert len(np.unique(selected)) == 17
    assert selected[0] == 100
    assert selected[-1] == 299


def test_trajectory_summary_is_patient_first_across_seeds() -> None:
    rows = []
    for subject, offset in (("s1", 0.0), ("s2", 1.0)):
        for seed in (1, 2, 3):
            rows.append(
                {
                    "subject": subject,
                    "dataset": "d",
                    "seed": seed,
                    "event_index": seed,
                    "progress_bin": 0.5,
                    "posterior_entropy_normalized": offset + seed,
                    "posterior_max": 0.5,
                    "posterior_weighted_excitation": 0.1,
                    "posterior_weighted_inhibition": 0.2,
                    "mode0_probability": 0.5,
                    "forward_probability": 0.5,
                }
            )
    patient, cohort = _summarize_trajectories(pd.DataFrame(rows))
    assert len(patient) == 6
    entropy = cohort[
        cohort.metric.eq("posterior_entropy_normalized")
    ].iloc[0]
    assert entropy.n_patients == 2
    assert entropy["median"] == 2.5
