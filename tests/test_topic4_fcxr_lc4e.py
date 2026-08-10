from __future__ import annotations

import numpy as np

from src.topic4_fcxr_lc4e import adjudicate_shared_screen, derive_shared_candidate


def _trace(current, rate=None, af=None, *, h_off=1.0):
    current = np.asarray(current, np.float32)
    n = current.size
    return {
        "adap_current": current,
        "trace_dt_ms": np.asarray([10.0]),
        "rate_E": np.asarray(rate if rate is not None else np.zeros(n)),
        "rate_dt_ms": np.asarray([10.0]),
        "af": np.asarray(af if af is not None else np.zeros(n * 10)),
        "af_bin_ms": np.asarray([1.0]),
        "snapshot_H_off_axis": np.asarray([h_off]),
    }


def test_candidate_changes_only_spatial_mode():
    c = dict(g_m_max=734.0, deadzone=46.0)
    lock = {"status": "L0_PASS", "candidate": c}
    rec = {"gate": {"verdict": "OFFSET_LATENCY_REPAIR_INSUFFICIENT", "onset_ms": 11000.0},
           "numerical": {"numerical_unsafe": False, "clip_frac_max": 0.0}}
    cur = np.zeros(1800); cur[1183:] = 1.0
    out = derive_shared_candidate(lock, rec, _trace(cur))
    assert out["changed_fields"] == {"m_hill_spatial_mode": "shared"}
    assert {k: v for k, v in out["candidate"].items() if k != "m_hill_spatial_mode"} == c


def test_positive_architecture_requires_prefix_identity_and_shared_gate():
    cur = np.zeros(1800); cur[1183:] = 1.0
    tr = _trace(cur)
    local = {"gate": {"verdict": "OFFSET_LATENCY_REPAIR_INSUFFICIENT",
                       "onset_ms": 11000.0, "bout_ms": 7000.0}}
    shared = {"gate": {"verdict": "L1_ENTRY_OFFSET_ALIGNED", "passed": True,
                        "onset_ms": 11000.0, "offset_ms": 14000.0,
                        "bout_ms": 3000.0, "n_returning_before_onset": 10}}
    out = adjudicate_shared_screen(local_record=local, shared_record=shared,
                                   local_trace=tr, shared_trace=tr)
    assert out["verdict"] == "SPATIALLY_SHARED_OFFSET_CANDIDATE"
    assert out["passed"] is True


def test_prefix_mismatch_has_priority_over_a_pretty_offset():
    cur = np.zeros(1800); cur[1183:] = 1.0
    local_t = _trace(cur)
    shared_t = _trace(cur, rate=np.r_[1.0, np.zeros(1799)])
    local = {"gate": {"verdict": "OFFSET_LATENCY_REPAIR_INSUFFICIENT"}}
    shared = {"gate": {"verdict": "L1_ENTRY_OFFSET_ALIGNED", "passed": True}}
    out = adjudicate_shared_screen(local_record=local, shared_record=shared,
                                   local_trace=local_t, shared_trace=shared_t)
    assert out["verdict"] == "CAUSAL_PREFIX_MISMATCH"
    assert out["passed"] is False


def test_subsecond_suppression_is_not_promoted_to_a_carrier():
    cur = np.zeros(1800); cur[1183:] = 1.0
    tr = _trace(cur)
    local = {"gate": {"verdict": "OFFSET_LATENCY_REPAIR_INSUFFICIENT"}}
    shared = {"gate": {"verdict": "TERMINATOR_PREVENTS_QUALIFYING_ENTRY", "passed": False}}
    out = adjudicate_shared_screen(local_record=local, shared_record=shared,
                                   local_trace=tr, shared_trace=tr)
    assert out["verdict"] == "SHARED_EXECUTOR_OVERFAST"
    assert out["passed"] is False
