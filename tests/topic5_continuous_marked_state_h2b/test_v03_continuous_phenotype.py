from __future__ import annotations

import numpy as np
import pytest

from scripts.topic5_continuous_marked_state_h2b.run_v03_continuous_phenotype import (
    _nearest_lead_row,
    _split,
    run,
)


def test_continuous_target_splits_are_chronological_when_supported() -> None:
    observed = [_split(10, index) for index in range(10)]
    assert [row[0] for row in observed] == [
        "TRAIN", "TRAIN", "TRAIN", "TRAIN", "TRAIN", "TRAIN",
        "SELECT", "SELECT", "TEST", "TEST",
    ]
    assert {row[1] for row in observed} == {"primary_chronological"}


def test_small_continuous_target_sets_remain_exploratory() -> None:
    assert _split(5, 0) == ("TRAIN", "sensitivity_loso")
    assert _split(3, 0) == ("TRAIN", "descriptive_case_series")
    assert _split(1, 0) == ("TRAIN", "not_estimable")


def test_nearest_lead_anchor_stays_in_segment_and_before_onset() -> None:
    time = np.asarray([0.0, 300.0, 600.0, 900.0, 1200.0])
    segment = np.asarray([0, 0, 0, 1, 1])
    row = _nearest_lead_row(
        time, segment, onset=2400.0, label=0, lead_minutes=30.0,
    )
    assert row == 2
    assert _nearest_lead_row(
        time, segment, onset=5000.0, label=0, lead_minutes=30.0,
    ) is None


def test_continuous_phenotype_fails_closed_when_a1_a2_fail(tmp_path) -> None:
    root = tmp_path / "v03"
    (root / "qualification").mkdir(parents=True)
    (root / "qualification/state_qualified_manifest.json").write_text(
        '{"subjects": []}', encoding="utf-8",
    )
    target = tmp_path / "target.csv"
    target.write_text("subject,seizure_idx,r3_observed,r3_null_median\n", encoding="utf-8")
    with pytest.raises(ValueError, match="A8 not released"):
        run(v02=tmp_path / "v02", root=root, target_source=target)
