from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analyze_topic5_persistent_path_formal import (
    benjamini_hochberg,
    comparison_benefits,
    patient_statistics,
)


def test_bh_is_monotone_in_rank_and_bounded() -> None:
    raw = np.asarray([0.04, 0.001, 0.02, np.nan])
    adjusted = benjamini_hochberg(raw)
    assert np.isnan(adjusted[-1])
    assert np.all((adjusted[:3] >= raw[:3]) & (adjusted[:3] <= 1.0))
    order = np.argsort(raw[:3])
    assert np.all(np.diff(adjusted[:3][order]) >= 0)


def test_formal_benefit_uses_lower_error_as_better() -> None:
    rows = []
    for subject_index in range(34):
        subject = f"s{subject_index:02d}"
        for seed in (20260726, 20260727, 20260728):
            for mode_count, control, value in (
                (0, "no_history", 0.30),
                (1, "merged_path", 0.25),
                (2, "intact", 0.10),
                (2, "weight_shuffle", 0.20),
                (2, "mode_shuffle", 0.15),
            ):
                rows.append(
                    {
                        "subject": subject,
                        "seed": seed,
                        "mode_count": mode_count,
                        "control": control,
                        "lesion": "none",
                        "participation_mae": value,
                        "rank_wasserstein": value,
                        "heldout_event_nll": value,
                        "precedence_mae": value,
                        "path_sliced_wasserstein": value,
                    }
                )
    benefits = comparison_benefits(pd.DataFrame(rows))
    selected = benefits[
        (benefits.baseline == "no_history")
        & (benefits.metric == "participation_mae")
    ]
    np.testing.assert_allclose(selected.benefit, 0.20)
    patient, stats = patient_statistics(
        benefits, group_column="baseline", primary_only=True
    )
    assert len(patient) == 34 * 4 * 2
    assert stats["pass"].all()


def test_patient_statistics_supports_development_excluded_sensitivity() -> None:
    rows = []
    for subject_index in range(31):
        for seed in (20260726, 20260727, 20260728):
            rows.append(
                {
                    "baseline": "b",
                    "metric": "participation_mae",
                    "subject": f"s{subject_index}",
                    "seed": seed,
                    "benefit": 0.1,
                }
            )
    _, stats = patient_statistics(
        pd.DataFrame(rows),
        group_column="baseline",
        primary_only=True,
        expected_patients=31,
    )
    assert stats.iloc[0]["pass"]
