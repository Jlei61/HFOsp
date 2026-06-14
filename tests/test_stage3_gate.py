"""TDD for src.sef_hfo_stage3.pilot_gate — the pre-registered Stage 3 regime-screen gate.

The gate must ENCODE THE CONCLUSION (MEMORY: acceptance gates must encode the conclusion, not just
existence): a (sep,std,mean[,drive]) cell only 'passes' if it is a usable BALANCED bidirectional
low-collision regime, and a bad regime must fail with a specific reason (not silently pass).
"""
from src.sef_hfo_stage3 import pilot_gate


def test_gate_passes_balanced_low_collision_cell():
    ok, reason, flags = pilot_gate(collision_rate=0.10, neg_clean=5, pos_clean=4,
                                   n_events=20, ambiguous=2, bidir_seed_frac=1.0)
    assert ok is True and reason == "pass"
    assert flags["collision_ok"] and flags["source_balance_ok"]


def test_gate_fails_high_collision():
    ok, reason, _ = pilot_gate(collision_rate=0.50, neg_clean=5, pos_clean=5,
                               n_events=20, ambiguous=1, bidir_seed_frac=1.0)
    assert ok is False and reason == "high_collision"


def test_gate_fails_source_imbalance_one_end_dominant():
    # one end never produces clean source events -> not a two-source train, even at low collision.
    ok, reason, _ = pilot_gate(collision_rate=0.10, neg_clean=0, pos_clean=5,
                               n_events=20, ambiguous=3, bidir_seed_frac=1.0)
    assert ok is False and reason == "source_imbalance"


def test_gate_fails_no_events_too_cold():
    # drive too low -> nothing ignites -> 'no_events', NOT a spurious low-collision pass.
    ok, reason, _ = pilot_gate(collision_rate=0.0, neg_clean=0, pos_clean=0,
                               n_events=0, ambiguous=0, bidir_seed_frac=0.0)
    assert ok is False and reason == "no_events"


def test_gate_unidirectional_only_when_sign_is_trusted():
    # if the readout sign is trusted (sign_ok) a one-direction-only cell fails as unidirectional;
    # if sign is NOT trusted, direction is ignored and the source-based gate decides.
    base = dict(collision_rate=0.10, neg_clean=5, pos_clean=5, n_events=20, ambiguous=2)
    ok_trust, reason_trust, _ = pilot_gate(**base, bidir_seed_frac=0.0, sign_ok=True)
    ok_notrust, reason_notrust, _ = pilot_gate(**base, bidir_seed_frac=0.0, sign_ok=False)
    assert ok_trust is False and reason_trust == "unidirectional"
    assert ok_notrust is True and reason_notrust == "pass"


def test_gate_reports_ambiguous_rate():
    _, _, flags = pilot_gate(collision_rate=0.10, neg_clean=0, pos_clean=0,
                             n_events=25, ambiguous=14, bidir_seed_frac=1.0)
    assert flags["ambiguous_rate"] == round(14 / 25, 3)
