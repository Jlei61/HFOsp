"""TDD for SEF-HFO SOZ-localization cohort + channel-universe + montage bridge.

Contract source: docs/superpowers/plans/2026-06-06-sef-hfo-soz-localization-rate-vs-geometry.md
  §3.0 channel universe (缺测=missing 非0), §3.2 montage bridge (single exact /
  bipolar first-contact X->X-next unique, else missing), Task 1 (three-source
  intersection + SOZ非空 + total_hours>=12 gate + soz_coverage + comparison_a_eligible).
  Bridge convention verified in docs/archive/topic4/sef_hfo/channel_universe_montage_diagnostic_2026-06-06.md.
"""
from __future__ import annotations

import pytest

from src.sef_hfo_soz_localization import (
    classify_montage,
    build_rate_lookup,
    geom_valid_contacts,
    compute_channel_universe,
    build_cohort,
)


# ---------- montage classification (§3.2) ----------

def test_classify_montage_bipolar_when_most_names_hyphenated():
    cm = [{"ch_name": n, "n_events": 100, "event_rate": 1.0}
          for n in ["E9-E10", "E10-E11", "K3-K4", "K4-K5"]]
    assert classify_montage(cm) == "bipolar"


def test_classify_montage_single_when_no_hyphen():
    cm = [{"ch_name": n, "n_events": 100, "event_rate": 1.0}
          for n in ["A1", "A2", "HRA1", "BFRA5"]]
    assert classify_montage(cm) == "single"


# ---------- rate lookup / montage bridge (§3.2) ----------

def test_build_rate_lookup_single_exact_name_with_min_events_gate():
    cm = [
        {"ch_name": "A1", "n_events": 100, "event_rate": 50.0},
        {"ch_name": "A2", "n_events": 100, "event_rate": 40.0},
        {"ch_name": "A3", "n_events": 10, "event_rate": 5.0},  # below MIN_CH_EVENTS=30
    ]
    look = build_rate_lookup(cm, "single", min_ch_events=30)
    assert look == {"A1": 50.0, "A2": 40.0}  # A3 gated out, not zero-filled


def test_build_rate_lookup_bipolar_first_contact_unique_else_missing():
    # K3-K4, K5-K6, K6-K7 present; K8-K9 below gate; no K7-K8, no K9-K10.
    cm = [
        {"ch_name": "K3-K4", "n_events": 100, "event_rate": 30.0},
        {"ch_name": "K5-K6", "n_events": 100, "event_rate": 20.0},
        {"ch_name": "K6-K7", "n_events": 100, "event_rate": 25.0},
        {"ch_name": "K8-K9", "n_events": 10, "event_rate": 9.0},   # gated out
    ]
    look = build_rate_lookup(cm, "bipolar", min_ch_events=30)
    # first-contact convention: contact X -> unique pair 'X-next'
    assert look["K3"] == 30.0   # K3-K4
    assert look["K5"] == 20.0   # K5-K6
    assert look["K6"] == 25.0   # K6-K7
    assert "K8" not in look     # K8-K9 below gate -> missing
    assert "K7" not in look     # only K6-K7 (K7 is SECOND contact) -> not first-contact -> missing
    assert "K4" not in look     # K4 only second contact of K3-K4 -> missing


# ---------- geom valid contacts (joint_valid, §3.1/§6.1) ----------

def test_geom_valid_contacts_respects_joint_valid():
    pair = {
        "channel_names": ["A1", "A2", "A3", "A4"],
        "joint_valid": [True, True, False, True],
        "rank_a_dense_full": [0, 1, 0, 2],
        "rank_b_dense_full": [2, 1, 0, 0],
    }
    assert geom_valid_contacts(pair) == ["A1", "A2", "A4"]


# ---------- channel universe = geom_valid ∩ bridged-rate (§3.0) ----------

def test_channel_universe_single_intersects_geom_and_rate_no_zero_fill():
    pair = {"channel_names": ["A1", "A2", "A3"],
            "joint_valid": [True, True, True],
            "rank_a_dense_full": [0, 1, 2], "rank_b_dense_full": [2, 1, 0]}
    cm = [
        {"ch_name": "A1", "n_events": 100, "event_rate": 50.0},
        {"ch_name": "A2", "n_events": 10, "event_rate": 5.0},   # below gate
        # A3 not in rate file at all
    ]
    u = compute_channel_universe(pair, cm, min_ch_events=30)
    assert u["montage"] == "single"
    assert u["universe"] == ["A1"]                 # A2 gated, A3 no rate -> excluded
    assert set(u["geom_missing_no_rate"]) == {"A2", "A3"}
    assert "A3" not in u["rate_lookup"] and "A2" not in u["rate_lookup"]


# ---------- build_cohort: intersection, SOZ-nonempty, hours gate, coverage, eligibility ----------

def _single_subject(dataset, subject, total_hours, geom_n, soz_core):
    chans = [f"C{i}" for i in range(geom_n)]
    pair = {"channel_names": chans, "joint_valid": [True] * geom_n,
            "rank_a_dense_full": list(range(geom_n)),
            "rank_b_dense_full": list(range(geom_n))[::-1]}
    cm = [{"ch_name": c, "n_events": 100, "event_rate": 100.0 - i}
          for i, c in enumerate(chans)]
    return {"dataset": dataset, "subject": subject, "total_hours": total_hours,
            "soz_core": soz_core, "channel_metrics": cm, "geom_pair": pair}


def test_build_cohort_drops_incomplete_sources_and_empty_soz():
    candidates = [
        _single_subject("epilepsiae", "ok", 100, 8, ["C0", "C1", "C2"]),
        {"dataset": "epilepsiae", "subject": "no_geom", "total_hours": 100,
         "soz_core": ["C0"], "channel_metrics": [], "geom_pair": None},
        {"dataset": "epilepsiae", "subject": "empty_soz", "total_hours": 100,
         "soz_core": [], "channel_metrics": [{"ch_name": "C0", "n_events": 100, "event_rate": 1.0}],
         "geom_pair": {"channel_names": ["C0"], "joint_valid": [True],
                       "rank_a_dense_full": [0], "rank_b_dense_full": [0]}},
    ]
    cohort = build_cohort(candidates, min_hours=12, min_ch_events=30)
    kept = {r["subject"] for r in cohort["kept"]}
    excl = {r["subject"]: r["reason"] for r in cohort["excluded"]}
    assert kept == {"ok"}
    assert "no_geom" in excl and "empty_soz" in excl


def test_build_cohort_hours_gate_excludes_low_hours():
    candidates = [
        _single_subject("yuquan", "good", 24, 8, ["C0", "C1"]),
        _single_subject("yuquan", "pengzihang_like", 2.0, 8, ["C0", "C1"]),
    ]
    cohort = build_cohort(candidates, min_hours=12, min_ch_events=30)
    assert {r["subject"] for r in cohort["kept"]} == {"good"}
    reasons = {r["subject"]: r["reason"] for r in cohort["excluded"]}
    assert "pengzihang_like" in reasons and "hours" in reasons["pengzihang_like"].lower()


def test_build_cohort_coverage_and_eligibility():
    # eligible subject: |U|=8>=5, |SOZ∩U|=3>=2, coverage=1.0
    elig = _single_subject("epilepsiae", "elig", 100, 8, ["C0", "C1", "C2"])
    # low-coverage subject: SOZ on channels outside geometry -> coverage 0, A-ineligible
    lowcov = _single_subject("yuquan", "lowcov", 24, 6, ["Z1", "Z2", "Z3"])
    cohort = build_cohort([elig, lowcov], min_hours=12, min_ch_events=30)
    recs = {r["subject"]: r for r in cohort["kept"]}
    assert recs["elig"]["soz_coverage"] == pytest.approx(1.0)
    assert recs["elig"]["comparison_a_eligible"] is True
    assert recs["elig"]["n_universe"] == 8
    assert recs["lowcov"]["soz_coverage"] == pytest.approx(0.0)
    assert recs["lowcov"]["comparison_a_eligible"] is False  # |SOZ∩U|=0 < 2


def test_build_cohort_all_soz_in_universe_is_ineligible():
    # entire universe inside SOZ -> no non-SOZ channels -> AUC undefined -> A-ineligible
    subj = _single_subject("epilepsiae", "all_soz", 100, 6, ["C0", "C1", "C2", "C3", "C4", "C5"])
    cohort = build_cohort([subj], min_hours=12, min_ch_events=30)
    rec = cohort["kept"][0]
    assert rec["n_soz_in_universe"] == rec["n_universe"]      # all-SOZ
    assert rec["comparison_a_eligible"] is False              # no contrast for AUC


def test_build_cohort_bipolar_subject_bridged_into_universe():
    # bipolar yuquan: geom single-contact K3/K5; rate bipolar K3-K4/K5-K6
    pair = {"channel_names": ["K3", "K5", "K7"], "joint_valid": [True, True, True],
            "rank_a_dense_full": [0, 1, 2], "rank_b_dense_full": [2, 1, 0]}
    cm = [
        {"ch_name": "K3-K4", "n_events": 100, "event_rate": 30.0},
        {"ch_name": "K4-K5", "n_events": 100, "event_rate": 22.0},
        {"ch_name": "K5-K6", "n_events": 100, "event_rate": 20.0},
        {"ch_name": "K6-K7", "n_events": 100, "event_rate": 25.0},
        # no K7-K8 -> K7 missing
    ]
    subj = {"dataset": "yuquan", "subject": "bip", "total_hours": 24,
            "soz_core": ["K3", "K5"], "channel_metrics": cm, "geom_pair": pair}
    cohort = build_cohort([subj], min_hours=12, min_ch_events=30)
    rec = cohort["kept"][0]
    assert rec["montage"] == "bipolar"
    assert set(rec["universe"]) == {"K3", "K5"}     # K7 has no first-contact pair -> missing
    assert "K7" in rec["geom_missing_no_rate"]
    assert rec["soz_coverage"] == pytest.approx(1.0)  # both SOZ (K3,K5) in universe
    # §6: stored rate maps EXACTLY to the universe (no non-universe phantom keys, no 0-fill)
    assert set(rec["rate_in_universe"].keys()) == set(rec["universe"])
    assert rec["rate_in_universe"]["K3"] == 30.0 and rec["rate_in_universe"]["K5"] == 20.0
