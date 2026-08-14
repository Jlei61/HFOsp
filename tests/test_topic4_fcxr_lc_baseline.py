import json
from pathlib import Path

import pytest

from src.topic4_fcxr_lc_baseline import (
    DEFAULT_SNAPSHOT,
    load_classifier_snapshot,
    validate_classifier_snapshot,
)
from src.topic4_mz_fcxr_lifecycle import window_regime


def test_tracked_snapshot_exposes_only_the_frozen_classifier_contract():
    baseline = load_classifier_snapshot()
    assert baseline["schema"] == "fcxr_lc1_seed1_classifier_snapshot_v1"
    assert baseline["frozen_event_bar"] == 0.03978125
    assert baseline["af_bin_ms"] == 1.0
    assert baseline["floor_af"] == 3.125e-05
    assert baseline["band"] == {
        "event_lookback_ms": 8000.0,
        "event_rate_hi": 3.15,
        "event_rate_lo": 0.086,
        "recruit_p90": 0.0717,
        "roll_hi": 9.7382291667,
        "win_ms": 1000.0,
    }
    assert baseline["original_full_contract"]["available"] is False


def test_snapshot_provenance_is_present_and_original_hash_is_locked():
    baseline = load_classifier_snapshot(DEFAULT_SNAPSHOT)
    assert baseline["original_full_contract"]["sha256"] == (
        "fd3e0d05ef730c30a484a071046e6a92d8f5e775b2035646dc89f4b4e8367c53"
    )


def test_snapshot_validation_fails_closed_on_missing_classifier_field():
    payload = json.loads(Path(DEFAULT_SNAPSHOT).read_text())
    del payload["band"]["roll_hi"]
    with pytest.raises(ValueError, match="missing fields"):
        validate_classifier_snapshot(payload)


def test_snapshot_preserves_registered_interictal_dense_and_ictal_boundaries():
    band = load_classifier_snapshot()["band"]
    assert window_regime(
        {"occ": 0.0, "event_rate_hz": 0.125, "recruit_frac": 0.05}, band
    ) == "INTERICTAL"
    assert window_regime(
        {"occ": 0.11, "event_rate_hz": 1.0, "recruit_frac": 0.05}, band
    ) == "DENSE"
    assert window_regime(
        {"occ": 0.50, "event_rate_hz": 3.0, "recruit_frac": 0.072}, band
    ) == "ICTAL"
