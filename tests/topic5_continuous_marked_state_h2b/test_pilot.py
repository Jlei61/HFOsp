import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.topic5_continuous_marked_state_h2b.pilot import (
    _mask_signature,
    _outside_intervals,
    state_cache_to_anchor_frame,
)
from src.topic5_continuous_marked_state_h2b.contract import sha256_file
from src.topic5_continuous_marked_state_r1.coverage import CoverageTable
from src.topic5_continuous_marked_state_r1.history import history_names
from src.topic5_continuous_marked_state_r1.raw_observation import EXPLICIT_NAMES


def test_mask_signature_is_deterministic_and_order_sensitive():
    left = _mask_signature(np.asarray([True, False, True]))
    right = _mask_signature(np.asarray([True, False, True]))
    changed = _mask_signature(np.asarray([True, True, False]))
    assert left == right
    assert left != changed


def test_wrong_time_interval_check_is_closed_interval():
    intervals = [(10.0, 20.0), (30.0, 40.0)]
    assert _outside_intervals(9.0, intervals)
    assert not _outside_intervals(10.0, intervals)
    assert not _outside_intervals(35.0, intervals)
    assert _outside_intervals(41.0, intervals)


def test_anchor_frame_marks_ictal_and_postictal_grid_rows(tmp_path: Path):
    query_path = tmp_path / "queries.csv"
    pd.DataFrame([
        {
            "query_id": "inside", "query_role": "control_candidate",
            "case_seizure_id": "", "case_lead_minutes": "",
            "exclusion_start_epoch": np.nan, "exclusion_stop_epoch": np.nan,
        },
        {
            "query_id": "outside", "query_role": "control_candidate",
            "case_seizure_id": "", "case_lead_minutes": "",
            "exclusion_start_epoch": np.nan, "exclusion_stop_epoch": np.nan,
        },
    ]).to_csv(query_path, index=False)
    exclusion_path = tmp_path / "exclusions.csv"
    pd.DataFrame([{
        "interval_start_epoch": 100.0, "interval_stop_epoch": 200.0,
    }]).to_csv(exclusion_path, index=False)
    coverage = CoverageTable(
        subject="p1", start=np.asarray([0.0]), stop=np.asarray([1000.0]),
        session=np.asarray([0], dtype=np.int64), train_end_epoch=700.0,
        dev_end_epoch=900.0, source_hashes={"fixture": "f" * 64},
    )
    coverage.validate()
    n, state_dim, contacts = 2, 2, 1
    hnames = history_names(contacts)
    cache = tmp_path / "states.npz"
    np.savez_compressed(
        cache,
        query_id=np.asarray(["inside", "outside"]),
        anchor_time_epoch=np.asarray([150.0, 250.0], dtype=np.float64),
        coverage_segment_index=np.zeros(n, dtype=np.int64),
        persistent_state=np.zeros((n, state_dim), dtype=np.float32),
        memoryless_observation_code=np.zeros((n, state_dim), dtype=np.float32),
        current_explicit_summary=np.zeros(
            (n, 2 * len(EXPLICIT_NAMES)), dtype=np.float32,
        ),
        deterministic_history=np.zeros((n, len(hnames)), dtype=np.float32),
        current_contact_mask=np.ones((n, contacts), dtype=bool),
        observation_available=np.ones(n, dtype=bool),
        observation_age_seconds=np.zeros(n, dtype=np.float64),
        wrong_time_donor_time_epoch=np.asarray([[300.0], [300.0]], dtype=np.float64),
        wrong_time_donor_state=np.zeros((n, 1, state_dim), dtype=np.float32),
        wrong_time_valid=np.ones((n, 1), dtype=bool),
    )
    cache.with_suffix(".manifest.json").write_text(json.dumps({
        "cache_sha256": sha256_file(cache),
        "all_current_observations_fresh": True,
        "deterministic_history_names": list(hnames),
    }))
    frame = state_cache_to_anchor_frame(
        cache_path=cache, query_path=query_path, coverage=coverage,
        global_exclusion_path=exclusion_path, seed=0, patient_id="p1",
    )
    assert frame["in_ictal_or_postictal"].tolist() == [True, False]
