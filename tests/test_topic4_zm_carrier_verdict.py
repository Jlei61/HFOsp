"""Pre-registered carrier-gate verdict logic (spec 2026-07-24). Tested on SYNTHETIC metric traces / dicts
so the thresholds are validated independently of any simulation (the 'don't tune thresholds on the traces
you then classify' discipline, mirroring src/sef_hfo_m4_termination.py).

Two verdicts, kept strictly separate from the old termination `fragment` label:
  ictal_carrier_verdict  -> {fail_hfo_like_train, fail_plateau, fail_runaway,
                             candidate_source_only, candidate_observed_carrier}
  lifecycle_verdict      -> {no_onset, prevention, persistent, terminate_to_silence,
                             terminate_then_reignite, terminate_and_recover}  (gated behind carrier pass)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.topic4_zm_carrier_verdict import (  # noqa: E402
    analyze_macroepisode, is_sustained, ictal_carrier_verdict, lifecycle_verdict,
    MIN_MACRO_MS, OCCUPANCY_MIN, MAX_GAP_MS)

DT = 5.0  # ms, BIN_FINE


# ---------------------------------------------------------------- analyze_macroepisode
def test_flat_trace_has_no_onset_and_is_not_sustained():
    e = np.full(400, 0.1)
    m = analyze_macroepisode(e, DT)
    assert m["onset_ms"] is None
    assert not is_sustained(m)


def test_constant_high_after_baseline_is_a_sustained_macroepisode():
    e = np.concatenate([np.full(60, 0.1), np.full(700, 5.0)])   # 300 ms baseline + 3500 ms high
    m = analyze_macroepisode(e, DT)
    assert m["onset_ms"] is not None
    assert m["duration_ms"] >= 3000.0
    assert m["occupancy"] > 0.98
    assert is_sustained(m)


def test_sparse_burst_train_with_long_gaps_is_not_sustained():
    # 100 ms bursts separated by 500 ms sub-floor gaps (> MAX_GAP 250 ms) -> episodes don't merge
    seg = np.concatenate([np.full(20, 5.0), np.full(100, 0.1)])   # 100 ms on + 500 ms off
    e = np.concatenate([np.full(60, 0.1)] + [seg] * 10)
    m = analyze_macroepisode(e, DT)
    assert m["onset_ms"] is not None                    # it does have events
    assert m["duration_ms"] < MIN_MACRO_MS              # but no single macroepisode reaches 2 s
    assert not is_sustained(m)


def test_dense_bursts_within_gap_tolerance_merge_into_a_carrier():
    # 100 ms bursts separated by 150 ms gaps (< MAX_GAP 250 ms) for > 2 s, high duty -> sustained
    seg = np.concatenate([np.full(20, 5.0), np.full(30, 0.1)])    # 100 ms on + 150 ms off = 40% duty...
    e = np.concatenate([np.full(60, 0.1)] + [seg] * 40)          # 40 cycles = 10 s
    m = analyze_macroepisode(e, DT)
    assert m["duration_ms"] >= MIN_MACRO_MS             # merges across <=250 ms gaps into a long span
    assert m["max_gap_ms"] <= MAX_GAP_MS + 1e-6         # internal gaps bounded by construction
    # occupancy here is ~40% (dense-ish but gappy) -> NOT a carrier; the occupancy gate is what separates
    assert m["occupancy"] < OCCUPANCY_MIN
    assert not is_sustained(m)


# ---------------------------------------------------------------- ictal_carrier_verdict
def _sustained_macro():
    return dict(onset_ms=300.0, duration_ms=8000.0, occupancy=0.95, max_gap_ms=100.0)


def _unsustained_macro():
    return dict(onset_ms=300.0, duration_ms=400.0, occupancy=0.4, max_gap_ms=500.0)


def _carrier_metrics(**over):
    m = dict(runaway_early_stop_ms=None, tail_escalating=False, whole_field_flash=False,
             saturated_plateau=False, has_recruitment=True, src_macro=_sustained_macro(),
             src_sep_count=3, obs_n_sustained_contacts=3, obs_highfreq_enhanced=True,
             obs_best_macro=_sustained_macro(), obs_sep_count=4)
    m.update(over); return m


def test_runaway_flag_gives_fail_runaway_regardless_of_shape():
    lab, _ = ictal_carrier_verdict(_carrier_metrics(runaway_early_stop_ms=2871.8))
    assert lab == "fail_runaway"
    lab2, _ = ictal_carrier_verdict(_carrier_metrics(tail_escalating=True))
    assert lab2 == "fail_runaway"


def test_whole_field_flash_or_no_recruitment_gives_fail_plateau():
    assert ictal_carrier_verdict(_carrier_metrics(whole_field_flash=True))[0] == "fail_plateau"
    assert ictal_carrier_verdict(_carrier_metrics(saturated_plateau=True))[0] == "fail_plateau"
    assert ictal_carrier_verdict(_carrier_metrics(has_recruitment=False))[0] == "fail_plateau"


def test_unsustained_source_gives_fail_hfo_like_train():
    lab, _ = ictal_carrier_verdict(_carrier_metrics(src_macro=_unsustained_macro()))
    assert lab == "fail_hfo_like_train"
    # gate A also fails if fewer than 2 of 3 separation dims (A7)
    lab2, _ = ictal_carrier_verdict(_carrier_metrics(src_sep_count=1))
    assert lab2 == "fail_hfo_like_train"


def test_gateA_pass_but_gateB_fail_gives_source_only():
    assert ictal_carrier_verdict(_carrier_metrics(obs_n_sustained_contacts=1))[0] == "candidate_source_only"
    assert ictal_carrier_verdict(_carrier_metrics(obs_highfreq_enhanced=False))[0] == "candidate_source_only"
    assert ictal_carrier_verdict(_carrier_metrics(obs_best_macro=_unsustained_macro()))[0] == "candidate_source_only"
    assert ictal_carrier_verdict(_carrier_metrics(obs_sep_count=2))[0] == "candidate_source_only"


def test_full_gateA_and_gateB_pass_gives_observed_carrier():
    lab, detail = ictal_carrier_verdict(_carrier_metrics())
    assert lab == "candidate_observed_carrier"
    assert detail["gate_A"] is True and detail["gate_B"] is True


# ---------------------------------------------------------------- lifecycle_verdict (gated)
def test_lifecycle_not_emitted_when_carrier_failed():
    for fail in ("fail_hfo_like_train", "fail_plateau", "fail_runaway"):
        assert lifecycle_verdict(fail, dict(onset_detected=True, terminated=True)) == "carrier_not_established"


def test_lifecycle_persistent_when_carrier_holds_and_no_termination():
    v = lifecycle_verdict("candidate_source_only", dict(onset_detected=True, terminated=False))
    assert v == "persistent"


def test_lifecycle_terminate_and_recover_and_prevention():
    assert lifecycle_verdict("candidate_observed_carrier",
                             dict(onset_detected=True, terminated=True, interictal_recovered=True)) == "terminate_and_recover"
    assert lifecycle_verdict("candidate_observed_carrier",
                             dict(onset_detected=True, terminated=True, reignited=True)) == "terminate_then_reignite"
    assert lifecycle_verdict("candidate_observed_carrier",
                             dict(onset_detected=True, terminated=True)) == "terminate_to_silence"
    assert lifecycle_verdict("candidate_observed_carrier",
                             dict(prevented=True, onset_detected=False)) == "prevention"
    assert lifecycle_verdict("candidate_observed_carrier",
                             dict(onset_detected=False)) == "no_onset"
