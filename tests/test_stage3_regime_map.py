"""Guard the Stage-3 regime-map science contract (2026-06-15): event bucketing + ADVANCE GATE
thresholds + cell tag parsing. Hard thresholds are a science contract — they must not run unguarded."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import analyze_stage3_regime_map as rm   # noqa: E402


# ---------- cell_of ----------
def test_cell_of_parses_mean_and_sep():
    assert rm.cell_of("rm_m17.0_sep0.7_s3") == (17.0, 0.7)
    assert rm.cell_of("rm_m16.5_sep0.6_s1") == (16.5, 0.6)


# ---------- classify_event (5 mutually-exclusive buckets) ----------
def test_collision_takes_priority_over_npart():
    # co-ignition is a source-level outcome regardless of spread
    assert rm.classify_event(dict(hidden_source_label="collision", clean_for_timing=False, n_part=3)) == "collision"
    assert rm.classify_event(dict(hidden_source_label="collision", clean_for_timing=True, n_part=9)) == "collision"


def test_clean_global_uses_clean_for_timing_not_npart():
    # FATAL-2: clean uses the build_sidecar clean_for_timing contract verbatim
    assert rm.classify_event(dict(hidden_source_label="neg", clean_for_timing=True, n_part=8)) == "clean_global_neg"
    assert rm.classify_event(dict(hidden_source_label="pos", clean_for_timing=True, n_part=8)) == "clean_global_pos"


def test_local_is_strict_npart_lt_7():
    # FATAL-1: local is STRICTLY non-collision, non-clean, n_part<7
    assert rm.classify_event(dict(hidden_source_label="neg", clean_for_timing=False, n_part=3)) == "local"
    assert rm.classify_event(dict(hidden_source_label="ambiguous", clean_for_timing=False, n_part=4)) == "local"


def test_npart_ge7_unreadable_is_dirty_global_NOT_local():
    # FATAL-1: a spread-but-unreadable event must NOT pollute local
    assert rm.classify_event(dict(hidden_source_label="neg", clean_for_timing=False, n_part=9)) == "dirty_global"
    assert rm.classify_event(dict(hidden_source_label="ambiguous", clean_for_timing=False, n_part=10)) == "dirty_global"


# ---------- gate (5 verdicts + 3 fail directions) ----------
def _d(n=20, local=0, collision=0, cg_neg=0, cg_pos=0, dirty_global=0):
    return dict(n=n, local=local, collision=collision, cg_neg=cg_neg, cg_pos=cg_pos, dirty_global=dirty_global)


def test_gate_too_cold_when_undersampled():
    assert rm.gate(_d(n=5, cg_neg=3, cg_pos=3))[0] == "too_cold/undersampled"


def test_gate_collision_dominated_hot():
    assert rm.gate(_d(n=20, collision=8, cg_neg=3, cg_pos=3))[0] == "fail:collision_dominated"


def test_gate_local_dominated_cold():
    assert rm.gate(_d(n=20, local=19))[0] == "fail:local_dominated"


def test_gate_pass_relay_both_ends():
    assert rm.gate(_d(n=20, cg_neg=2, cg_pos=2, dirty_global=5))[0] == "PASS:relay_both_ends"


def test_gate_middle_one_end_dominant():
    v, mid = rm.gate(_d(n=20, cg_neg=5, cg_pos=0, dirty_global=4))
    assert v == "fail:one_end_or_no_clean_relay" and mid == "one_end_dominant"


def test_gate_middle_no_clean_relay():
    # low collision, non-local, spreads (dirty_global) but neither end clean
    v, mid = rm.gate(_d(n=20, cg_neg=1, cg_pos=0, dirty_global=10, local=5))
    assert v == "fail:one_end_or_no_clean_relay" and mid == "no_clean_relay"


def test_gate_thresholds_are_the_locked_values():
    assert (rm.K_END, rm.COLL_MAX, rm.LOCAL_MAX, rm.MIN_EV) == (2, 0.30, 0.90, 10)


# ---------- provenance audit (catches engine-drift mid-scout) ----------
def _prov(git, eng):
    return dict(git_sha=git, engine_sha={"kick_probe.py": eng})


def test_provenance_audit_single_ok():
    p = rm.audit_provenance([_prov("abc", "k1"), _prov("abc", "k1"), _prov("abc", "k1")])
    assert p["single_provenance"] is True and p["n_readouts"] == 3


def test_provenance_audit_flags_mixed_engine():
    # the 2026-06-15 failure mode: engine edited mid-scout -> two kick_probe shas
    p = rm.audit_provenance([_prov("abc", "k1"), _prov("abc", "k2")])
    assert p["single_provenance"] is False


def test_provenance_audit_flags_mixed_git():
    p = rm.audit_provenance([_prov("abc", "k1"), _prov("def", "k1")])
    assert p["single_provenance"] is False
