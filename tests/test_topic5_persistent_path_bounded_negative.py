from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analyze_topic5_persistent_path_bounded_negative import (
    _json_default,
    comparison_gate_matrix,
)


def _check_rows(kind: str, name: str) -> pd.DataFrame:
    key = "baseline" if kind == "comparison" else "lesion"
    return pd.DataFrame(
        [
            {
                "mode_count": 2,
                key: name,
                "metric": "precedence_mae",
                "n_patient_seed_better": 6,
                "n_subject_median_better": 2,
                "pass": True,
            },
            {
                "mode_count": 2,
                key: name,
                "metric": "path_sliced_wasserstein",
                "n_patient_seed_better": 7,
                "n_subject_median_better": 3,
                "pass": True,
            },
        ]
    )


def test_gate_matrix_preserves_keyword_named_pass_column() -> None:
    comparisons = _check_rows("comparison", "no_history")
    lesions = _check_rows("lesion", "graph")
    matrix = comparison_gate_matrix(comparisons, lesions)
    assert len(matrix) == 2
    assert matrix["both_metric_gate_pass"].all()
    assert set(matrix["precedence_n_better"]) == {6}
    assert set(matrix["whole_path_n_better"]) == {7}


def test_json_default_unwraps_numpy_scalars() -> None:
    assert _json_default(np.int64(3)) == 3
    assert _json_default(np.float64(0.25)) == 0.25
