from scripts.summarize_topic5_spectral_onset_cohort import (
    build_cohort_summary,
    build_refined_rows,
    build_subject_summary,
    timing_tier,
)


def _row(status, *, stable="", t="", subject="epilepsiae_1", seizure_idx=0):
    return {
        "subject": subject,
        "seizure_idx": str(seizure_idx),
        "seizure_id": f"sz{seizure_idx}",
        "auto_status": status,
        "auto_stable_candidate_time": stable,
        "auto_t_spectral_rel_eeg_sec": t,
        "eeg_onset_rel_clinical_sec": "-10",
        "auto_bootstrap_q05_rel_eeg_sec": "",
        "auto_bootstrap_q95_rel_eeg_sec": "",
        "auto_bootstrap_ci_width_sec": "",
        "auto_complete_change_gate": "",
        "auto_n_step_bands": "",
        "auto_n_step_contacts": "",
        "auto_low_step_supported": "",
        "auto_high_step_supported": "",
        "auto_n_episodes": "1" if t != "" else "0",
        "auto_n_prior_episodes": "0",
    }


def test_timing_tier_separates_primary_stable_and_unstable() -> None:
    assert timing_tier(_row("confirmed_precise_T", t="1")) == "primary_precise"
    assert (
        timing_tier(_row("broadband_but_imprecise_T", stable="True", t="2"))
        == "sensitivity_stable_candidate"
    )
    assert (
        timing_tier(_row("broadband_but_imprecise_T", stable="False", t="3"))
        == "exploratory_unstable_candidate"
    )


def test_refined_rows_do_not_force_primary_time_for_candidate() -> None:
    rows = build_refined_rows(
        [
            _row("confirmed_precise_T", t="1.5", seizure_idx=0),
            _row("broadband_but_imprecise_T", stable="True", t="3.0", seizure_idx=1),
            _row("no_detectable_broadband_transition", seizure_idx=2),
        ]
    )
    assert rows[0]["primary_t_spectral_rel_eeg_sec"] == 1.5
    assert rows[1]["primary_t_spectral_rel_eeg_sec"] == ""
    assert rows[1]["sensitivity_t_spectral_rel_eeg_sec"] == 3.0
    assert rows[2]["candidate_t_spectral_rel_eeg_sec"] == ""


def test_subject_summary_uses_subject_seizure_denominator() -> None:
    refined = build_refined_rows(
        [
            _row("confirmed_precise_T", t="1", seizure_idx=0),
            _row("confirmed_precise_T", t="3", seizure_idx=1),
            _row("no_detectable_broadband_transition", seizure_idx=2),
            _row("no_detectable_broadband_transition", seizure_idx=3),
        ]
    )
    subject = build_subject_summary(refined)[0]
    assert subject["n_seizures"] == 4
    assert subject["fraction_primary_precise"] == 0.5
    assert subject["primary_t_rel_eeg_median_sec"] == 2.0


def test_cohort_summary_reports_subject_level_timing() -> None:
    refined = build_refined_rows(
        [
            _row("confirmed_precise_T", t="1", subject="epilepsiae_1", seizure_idx=0),
            _row("confirmed_precise_T", t="3", subject="epilepsiae_1", seizure_idx=1),
            _row("confirmed_precise_T", t="5", subject="epilepsiae_2", seizure_idx=0),
        ]
    )
    subjects = build_subject_summary(refined)
    cohort = build_cohort_summary(refined, subjects)
    timing = cohort["timing_relative_to_eeg_onset_subject_level"]
    assert timing["primary_n_subjects_with_defined_median"] == 2
    assert timing["primary_subject_median_q25_median_q75_sec"] == [2.75, 3.5, 4.25]
