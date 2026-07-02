"""Tests for the Phase-2 shared data plumbing (scripts/_topic5_v2_crit_io.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._topic5_v2_crit_io import (  # noqa: E402
    state_prefix, shaft_of, window_index_range, load_subject_preictal,
)
from src.topic5_v2_criticality import load_phase2_config  # noqa: E402


def test_state_prefix_maps_bb_and_rejects_unknown():
    assert state_prefix("legacy_bb_1_45") == "bb"
    assert state_prefix("legacy_hfa_60_100") == "hfa"
    with pytest.raises(ValueError):
        state_prefix("ripple_full_80_250")  # not stored in the long cache


def test_shaft_of_strips_trailing_contact_number():
    assert shaft_of("HL10") == "HL"
    assert shaft_of("TLA1") == "TLA"
    assert shaft_of("GA'1") == "GA'"
    assert shaft_of("X") == "X"


def test_window_index_range_is_contiguous_half_open():
    relt = np.linspace(-120.0, 0.0, 1201)  # 0.1 s steps
    rng = window_index_range(relt, -30.0, 0.0)
    assert rng is not None
    start, stop = rng
    assert relt[start] >= -30.0 - 1e-9 and relt[stop - 1] <= 0.0 + 1e-9
    assert relt[start - 1] < -30.0  # first inside-window sample
    assert window_index_range(relt, 50.0, 60.0) is None  # no postictal in a preictal axis


@pytest.mark.integration
@pytest.mark.parametrize("ds_sid,substrate", [("epilepsiae_139", "broad"),
                                              ("epilepsiae_958", "narrow")])
def test_load_subject_preictal_real(ds_sid, substrate):
    cfg = load_phase2_config()
    sub = load_subject_preictal(ds_sid, substrate, cfg)
    assert sub["status"] == "ok", sub.get("skip_reason")
    assert sub["available_pre_sec"] >= cfg["preictal"]["min_required_pre_sec"]
    assert sub["n_contacts"] > 4 and sub["n_seizures"] >= 1
    # G_HFO ranks + geometry keyed to the same matched contacts
    assert set(sub["mapped"]) <= set(sub["ta_rank"])
    assert set(sub["shaft_by_name"]) == set(sub["mapped"])
    for s in sub["seizures"]:
        assert s["E"].shape[0] == sub["n_contacts"]
        assert s["E"].shape[1] == s["relt"].size
        assert s["relt"].min() >= cfg["preictal"]["span_rel"][0] - 1e-6
        assert s["relt"].max() <= cfg["preictal"]["span_rel"][1] + 1e-6
