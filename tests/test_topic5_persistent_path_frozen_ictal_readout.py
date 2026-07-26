from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_topic5_persistent_path_frozen_ictal_readout import (
    FEATURE_COLUMNS,
    _all_contact_null,
    _fit_condition_loso,
    _within_patient_scale,
)


def test_within_patient_scale_is_centered_and_nonconstant() -> None:
    scaled = _within_patient_scale(np.asarray([1.0, 2.0, 4.0, 8.0]))
    assert np.isclose(np.median(scaled), 0.0)
    assert np.std(scaled) > 0


def test_channel_null_is_reproducible() -> None:
    prediction = np.arange(8, dtype=float)
    target = np.asarray([0, 2, 1, 4, 3, 7, 5, 6], float)
    first = _all_contact_null(
        prediction, target, subject="s", n_perm=100, seed=1
    )
    second = _all_contact_null(
        prediction, target, subject="s", n_perm=100, seed=1
    )
    np.testing.assert_array_equal(first, second)


def test_loso_readout_keeps_patients_out_of_training() -> None:
    rows = []
    for subject_index in range(8):
        subject = f"s{subject_index}"
        for contact in range(8):
            feature = contact / 7
            row = {
                "subject": subject,
                "condition": "intact",
                "contact_name": f"c{contact}",
                "clinical_bb150_raw": feature,
                "clinical_bb150_scaled": feature,
            }
            row.update({column: 0.0 for column in FEATURE_COLUMNS})
            row["joint_rank_bin_0"] = feature
            row["nonparticipation_probability"] = 1.0 - feature
            rows.append(row)
    _, subjects = _fit_condition_loso(
        pd.DataFrame(rows),
        condition="intact",
        n_perm=100,
        seed=1,
    )
    assert len(subjects) == 8
    assert (subjects.n_train_subjects == 7).all()
    assert (subjects.rho_data > 0.99).all()
