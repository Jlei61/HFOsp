from __future__ import annotations

import numpy as np

from scripts.build_topic5_persistent_path_cross_state_features import (
    _distribution_rows,
)


def test_distribution_rows_form_a_full_probability_vector() -> None:
    groups = np.asarray([[0, 1, -1], [1, 0, 2], [-1, 0, 1]])
    counts = np.asarray([2, 3, 2])
    rows = _distribution_rows(
        groups,
        counts,
        subject="s",
        dataset="d",
        seed=1,
        condition="intact",
        contact_names=np.asarray(["a", "b", "c"]),
    )
    assert len(rows) == 3
    for row in rows:
        total = row["nonparticipation_probability"] + sum(
            row[f"joint_rank_bin_{index}"] for index in range(10)
        )
        assert np.isclose(total, 1.0)
