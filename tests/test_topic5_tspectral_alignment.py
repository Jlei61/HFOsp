import numpy as np
import pandas as pd

from scripts.build_topic5_tspectral_aligned_cache import (
    _cache_zero_time,
    align_rel_time,
)
from scripts.summarize_topic5_subject_spectral_onset import build_cohort_summary


def test_cache_zero_time_supports_yuquan_eeg_only_and_legacy_clinical_rows() -> None:
    yuquan = pd.Series(
        {
            "t_spectral_best_rel_cache_zero_sec": 4.3,
            "t_spectral_best_rel_clinical_sec": np.nan,
        }
    )
    legacy = pd.Series({"t_spectral_best_rel_clinical_sec": -1.2})

    assert _cache_zero_time(yuquan) == 4.3
    assert _cache_zero_time(legacy) == -1.2
    assert np.allclose(align_rel_time(np.array([3.3, 4.3, 5.3]), 4.3), [-1, 0, 1])


def test_yuquan_cohort_summary_does_not_invent_clinical_onset_statistics() -> None:
    events = pd.DataFrame(
        [
            {
                "phenotype_status": "phenotype_present",
                "timing_status": "accepted_subject_recurrent",
                "has_candidate_t": True,
                "has_accepted_t_best": True,
                "t_spectral_best_rel_eeg_sec": 4.3,
                "clinical_onset_available": False,
                "annotation_mode": "eeg_only",
                "cache_tier": "narrow_sensitivity",
                "prototype_used": True,
            }
        ]
    )
    subjects = pd.DataFrame(
        [
            {
                "fraction_phenotype_present": 1.0,
                "t_best_median_rel_eeg_sec": 4.3,
                "t_best_median_rel_clinical_sec": np.nan,
                "bootstrap_width_median_sec": 0.0,
                "selection_consistency_1s_median": 1.0,
                "median_abs_distance_to_eeg_sec": 4.3,
                "median_abs_distance_to_clinical_sec": np.nan,
            }
        ]
    )

    summary = build_cohort_summary(events, subjects)

    assert summary["annotation_mode_counts"] == {"eeg_only": 1}
    assert summary["subject_t_best_median_rel_clinical_q25_median_q75_sec"] is None
    alignment = summary["annotation_alignment_descriptive_not_independent"]
    assert alignment["median_subject_abs_distance_to_clinical_sec"] is None
    assert alignment["n_subjects_eeg_closer"] == 0
    assert alignment["n_subjects_clinical_closer"] == 0
