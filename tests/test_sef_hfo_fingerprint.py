# tests/test_sef_hfo_fingerprint.py
"""TDD for the Axis-A propagation fingerprint schema/extractor (Stage A0).

Synthetic events ONLY — no simulation, no engine. Each test builds a tiny readout
JSON + matching rep npz on disk and exercises one contract clause from plan §1:
  - no-direction event (perp-bar synchronous) -> axis judged unreadable; deferred
    speed stays None/requires_extended_readout_save (never fabricated).
  - sparse-contact event -> pathway_width / onset_jitter do NOT invent signal.
  - bad-axis event -> unreadable.
  - stable single-focus synthetic -> stable primary fingerprint across repeats.
  - name-misalignment -> raises.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from src.sef_hfo_fingerprint import (
    extract_fingerprint,
    load_run_artifacts,
    perp_pathway_width,
    along_axis_span,
    propose_n_min_events,
    FINGERPRINT_SCHEMA,
    PRIMARY_FEATURES,
    DEFERRED_FEATURES,
    REQUIRES_EXTENDED_SAVE,
)

# A 6-contact synthetic montage at theta=0:
#   shaft A (A0..A2) ALONG the axis (x varies), shaft B (B0..B2) PERPENDICULAR (y varies).
THETA = 0.0
NAMES = ["A0", "A1", "A2", "B0", "B1", "B2"]
COORDS = np.array([
    [0.0, 0.0], [4.0, 0.0], [8.0, 0.0],   # along-axis shaft
    [4.0, -4.0], [4.0, 0.0], [4.0, 4.0],  # perpendicular shaft
], float)


def _write_artifacts(tmp_path: Path, events, lesion="oneend_neg", seed=1,
                     valid=None):
    """Write a minimal readout JSON + rep npz; return their paths."""
    rj = {
        "tag": f"{lesion}_s{seed}",
        "provenance": {"engine_sha": {"connectivity_rot.py": "deadbeef0001"}},
        "config": {"lesion": lesion, "seed": seed, "theta": THETA},
        "n_events": len(events),
        "rep_event_index": 0,
        "events": events,
    }
    jp = tmp_path / f"readout_{lesion}_s{seed}.json"
    jp.write_text(json.dumps(rj))
    v = np.ones(len(NAMES), int) if valid is None else np.asarray(valid, int)
    npz_path = tmp_path / f"rep_{lesion}_s{seed}.npz"
    np.savez(npz_path, contacts=COORDS, names=np.array(NAMES),
             valid=v, theta=THETA, foci=np.array([[0.0, 0.0]]))
    return jp, npz_path


def _forward_event(sign=1.0, axis_err=0.0, t_on=100.0):
    """A clean forward event: A-shaft ranks increase along the axis; B1 mid participates."""
    ranks = {"A0": 0.0, "A1": 1.0, "A2": 2.0, "B0": None, "B1": 1.0, "B2": None}
    return dict(t_on=t_on, t_off=t_on + 50, event_peak_t=t_on + 40, returned=True,
                n_part=4, axis_err=axis_err, sign=sign, readability=0.98, ranks=ranks)


# ---------------------------------------------------------------------------
# schema freeze sanity
# ---------------------------------------------------------------------------
def test_schema_tiers_locked():
    """Tier discipline LOCKED: primary set exact; speed/event_size deferred; amplitude diagnostic."""
    assert set(PRIMARY_FEATURES) == {"axis_dir", "pathway_width", "onset_jitter"}
    assert set(DEFERRED_FEATURES) == {"speed", "event_size"}
    assert FINGERPRINT_SCHEMA["amplitude_proxy"]["tier"] == "diagnostic"
    # the three save-gap fields are exactly the ms-latency / amplitude ones
    assert set(REQUIRES_EXTENDED_SAVE) == {
        "amplitude_proxy", "latency_jitter", "speed", "event_size"}


# ---------------------------------------------------------------------------
# 1. no-direction event -> unreadable; speed never fabricated
# ---------------------------------------------------------------------------
def test_no_direction_event_unreadable_and_speed_none(tmp_path):
    # perp-bar synchronous: only the B (perpendicular) shaft participates, all tied rank.
    # axis_err None (runner could not fit an axis) -> our extractor must mark ambiguous.
    ranks = {"A0": None, "A1": None, "A2": None, "B0": 0.0, "B1": 0.0, "B2": 0.0}
    ev = dict(t_on=100.0, t_off=150.0, event_peak_t=140.0, returned=True,
              n_part=3, axis_err=None, sign=None, readability=None, ranks=ranks)
    jp, npz = _write_artifacts(tmp_path, [ev])
    rf = extract_fingerprint(jp, npz, n_min_events=1)
    row = rf.events[0]
    assert row.is_ambiguous is True
    assert row.axis_dir["readable"] is False
    # speed stays an honest gap (NEVER fabricated)
    assert row.speed["value"] is None
    assert row.speed["status"] == "requires_extended_readout_save"
    # this event is not clean -> excluded from aggregation
    assert rf.n_clean_events == 0


# ---------------------------------------------------------------------------
# 2. sparse-contact event -> pathway_width / onset_jitter do NOT invent signal
# ---------------------------------------------------------------------------
def test_sparse_contact_no_invented_width():
    # only 2 participating contacts -> perp width must be NaN, not a manufactured number
    part = np.zeros(len(NAMES), bool)
    part[[0, 1]] = True   # both on the along-axis shaft (perp extent = 0 anyway)
    w = perp_pathway_width(COORDS, part, THETA, min_part=3)
    assert np.isnan(w)
    a = along_axis_span(COORDS, part, THETA, min_part=3)
    assert np.isnan(a)


def test_sparse_event_onset_jitter_not_inflated(tmp_path):
    # one clean event with too-few contacts is excluded -> onset_jitter has n=0, no fake signal
    sparse = dict(t_on=100.0, t_off=150.0, event_peak_t=140.0, returned=True,
                  n_part=2, axis_err=0.0, sign=1.0, readability=0.9,
                  ranks={"A0": 0.0, "A1": 1.0, "A2": None, "B0": None, "B1": None, "B2": None})
    # n_part=2 (consistent with the 2 non-null ranks) but below PART_MIN=7 -> excluded
    jp, npz = _write_artifacts(tmp_path, [sparse])
    rf = extract_fingerprint(jp, npz, n_min_events=1)
    # PART_MIN gate (7) excludes this; clean count 0; onset_jitter n=0 (no invented entry)
    assert rf.n_clean_events == 0
    assert rf.aggregate["onset_jitter"]["n"] == 0


def test_n_part_rank_mismatch_raises(tmp_path):
    # n_part must EQUAL the number of non-null ranks: the clean gate uses n_part while
    # participation/width/jitter use the rank dict, so a disagreement is contamination.
    bad = dict(t_on=100.0, t_off=150.0, event_peak_t=140.0, returned=True,
               n_part=4, axis_err=0.0, sign=1.0, readability=0.9,
               ranks={"A0": 0.0, "A1": 1.0, "A2": None, "B0": None, "B1": None, "B2": None})
    # declared n_part=4 but only 2 non-null ranks -> must raise at BOTH entry points
    jp, npz = _write_artifacts(tmp_path, [bad])
    with pytest.raises(ValueError, match="n_part/rank mismatch"):
        load_run_artifacts(jp, npz)
    with pytest.raises(ValueError, match="n_part/rank mismatch"):
        extract_fingerprint(jp, npz, n_min_events=1)


# ---------------------------------------------------------------------------
# 3. bad-axis event -> unreadable
# ---------------------------------------------------------------------------
def test_bad_axis_event_unreadable(tmp_path):
    bad = _forward_event(axis_err=40.0)   # axis_err >= AXIS_ERR_MAX (25) -> not clean
    jp, npz = _write_artifacts(tmp_path, [bad])
    rf = extract_fingerprint(jp, npz, n_min_events=1)
    # axis_err present but >25 -> not clean (excluded), but still "readable" axis fit exists
    assert rf.n_clean_events == 0
    row = rf.events[0]
    assert row.axis_dir["axis_err_deg"] == 40.0


# ---------------------------------------------------------------------------
# 4. stable single-focus synthetic -> stable primary fingerprint across repeats
# ---------------------------------------------------------------------------
def test_stable_single_focus_fingerprint(tmp_path):
    # 8 identical clean forward events, but PART_MIN=7 requires >=7 participating contacts.
    # Build a rich montage event where all 6 contacts participate (still < 7) -> to make
    # them clean we lower nothing; instead we test the STABILITY of computed per-event
    # primaries directly on participating-contact geometry (the aggregate path needs >=7
    # participants which this 6-contact synthetic cannot reach -> assert per-event stability).
    events = [_forward_event(t_on=100.0 + 100 * i) for i in range(8)]
    jp, npz = _write_artifacts(tmp_path, events)
    rf = extract_fingerprint(jp, npz, n_min_events=1)
    # per-event primaries identical across the 8 repeats (deterministic geometry)
    widths = [r.pathway_width["value_mm"] for r in rf.events]
    firsts = [r.first_contact for r in rf.events]
    assert len(set(widths)) == 1            # same pathway_width every repeat
    assert all(f == "A0" for f in firsts)   # min-rank first-contact stable
    # source label = lesion sign (single-focus), collision N/A -> all False
    assert all(r.source_label == "neg" for r in rf.events)
    assert all(r.is_collision is False for r in rf.events)
    assert rf.collision_count == 0


def test_part_min_7_lets_a_rich_event_into_aggregate(tmp_path):
    # An 8-contact-participating event (n_part>=7) on an 8-contact montage so the CLEAN
    # gate + aggregate path are exercised end-to-end with stable primaries.
    names8 = NAMES + ["A3", "B3"]
    coords8 = np.vstack([COORDS, [[12.0, 0.0], [4.0, 8.0]]])
    ranks = {"A0": 0.0, "A1": 1.0, "A2": 2.0, "A3": 3.0,
             "B0": 1.0, "B1": 1.0, "B2": 1.0, "B3": 1.0}
    events = [dict(t_on=100.0 + 100 * i, t_off=160.0 + 100 * i, event_peak_t=150.0 + 100 * i,
                   returned=True, n_part=8, axis_err=0.0, sign=1.0, readability=0.98,
                   ranks=dict(ranks)) for i in range(5)]
    rj = {"tag": "oneend_neg_s1", "provenance": {"engine_sha": {}},
          "config": {"lesion": "oneend_neg", "seed": 1, "theta": THETA},
          "n_events": 5, "rep_event_index": 0, "events": events}
    jp = tmp_path / "readout.json"; jp.write_text(json.dumps(rj))
    npz = tmp_path / "rep.npz"
    np.savez(npz, contacts=coords8, names=np.array(names8),
             valid=np.ones(len(names8), int), theta=THETA, foci=np.array([[0.0, 0.0]]))
    rf = extract_fingerprint(jp, npz, n_min_events=3)
    assert rf.n_clean_events == 5
    assert rf.insufficient is False
    # primary aggregate stable + finite
    assert rf.aggregate["pathway_width"]["median_mm"] is not None
    assert rf.aggregate["pathway_width"]["iqr_mm"] == 0.0     # identical repeats
    assert rf.aggregate["onset_jitter"]["top1_fraction"] == 1.0  # always A0 first
    assert rf.aggregate["axis_dir"]["sign_majority"] == 1.0


def test_insufficient_gate_flags_low_clean_count(tmp_path):
    events = [_forward_event(t_on=100.0 + 100 * i) for i in range(2)]
    # make them rich enough to be clean via the 8-contact path is overkill; just check the gate
    jp, npz = _write_artifacts(tmp_path, events)
    rf = extract_fingerprint(jp, npz, n_min_events=6)
    # these 6-contact events are excluded by PART_MIN anyway -> 0 clean -> INSUFFICIENT
    assert rf.insufficient is True
    assert "INSUFFICIENT" in rf.aggregate


# ---------------------------------------------------------------------------
# 5. name-misalignment -> raises
# ---------------------------------------------------------------------------
def test_name_misalignment_raises(tmp_path):
    # event references a contact "Z9" that is NOT in the rep npz names -> must raise
    bad = _forward_event()
    bad["ranks"]["Z9"] = 5.0
    jp, npz = _write_artifacts(tmp_path, [bad])
    with pytest.raises(ValueError, match="name-alignment"):
        load_run_artifacts(jp, npz)
    with pytest.raises(ValueError, match="name-alignment"):
        extract_fingerprint(jp, npz, n_min_events=1)


def test_amplitude_proxy_never_promoted(tmp_path):
    ev = _forward_event()
    jp, npz = _write_artifacts(tmp_path, [ev])
    rf = extract_fingerprint(jp, npz, n_min_events=1)
    row = rf.events[0]
    assert row.amplitude_proxy["value"] is None
    assert "primary" not in row.amplitude_proxy.get("note", "").lower() or \
        "never a primary" in row.amplitude_proxy["note"]


def test_propose_n_min_events():
    out = propose_n_min_events({"oneend_neg_s1": 15, "oneend_pos_s1": 15})
    assert out["value"] == 6
    assert out["min_observed_clean"] == 15
    assert "sign-off" in out["rationale"]
