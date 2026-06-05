"""TDD for src/sef_hfo_events — Step 1 noise-driven event detection + OU noise.

Detector tests run on SYNTHETIC active-fraction traces (fast, no field sim) to
lock the regime logic against the frozen contract §1. Noise tests lock the
amp=0 identity and seed reproducibility; one short field run verifies the OU
drive is wired through integrate_lif_field.
"""
import numpy as np

from src.sef_hfo_events import (
    EVENT_ON_FRAC,
    FRAC_TIME_ON_MAX,
    classify_run,
    detect_events,
    make_ou_noise,
)
from src.sef_hfo_lif import ELL_PAR, ELL_PERP, L_INH, integrate_lif_field, mean_field

DT = 0.25
VAL = 0.2  # synthetic ON value, clearly above EVENT_ON_FRAC=0.05


def _times(t_max):
    return np.arange(0, t_max, DT)


def _rect(t, start, stop, val=VAL):
    """Rectangular ON pulse on [start, stop) ms — deterministic on/off + gaps."""
    return np.where((t >= start) & (t < stop), val, 0.0)


# ---------------------------------------------------------------------------
# Detector regime logic (synthetic coherence-activity traces)
# ---------------------------------------------------------------------------

def test_single_pulse_is_one_discrete_event():
    t = _times(300.0)
    ext = _rect(t, 90.0, 130.0)  # 40ms event, returns to 0
    op = mean_field(1.0)
    res = classify_run(ext, DT, op, window="A")
    assert res["label"] == "discrete_events", res
    assert res["n_events"] == 1, res["n_events"]


def test_two_close_pulses_merge_to_one():
    t = _times(300.0)
    ext = _rect(t, 90.0, 100.0) + _rect(t, 108.0, 120.0)  # gap 8ms < 12ms -> merge
    evs = detect_events(ext, DT)
    assert len(evs) == 1, [(e["t_on"], e["t_off"]) for e in evs]


def test_two_separated_pulses_are_two_events():
    t = _times(300.0)
    ext = _rect(t, 90.0, 100.0) + _rect(t, 130.0, 142.0)  # gap 30ms > 12ms -> two
    evs = detect_events(ext, DT)
    assert len(evs) == 2, [(e["t_on"], e["t_off"]) for e in evs]


def test_sustained_plateau():
    t = _times(520.0)
    ext = np.where(t >= 50.0, 0.2, 0.0)  # on from 50ms, never returns (>400ms)
    op = mean_field(1.0)
    res = classify_run(ext, DT, op, window="A")
    assert res["label"] == "sustained", res


def test_flickering_is_sustained_not_discrete():
    """Many short self-terminating pulses occupying >30% of the run = sustained.

    Locks the advisor temporal-separation criterion: discreteness requires
    events be temporally separated (mostly quiescent), not continuous flicker.
    """
    t = _times(300.0)
    ext = np.zeros_like(t)
    for k in range(2, 10):  # 8 pulses of 15ms every 25ms -> frac_time_on = 0.40
        ext = ext + _rect(t, k * 25.0, k * 25.0 + 15.0)
    op = mean_field(1.0)
    res = classify_run(ext, DT, op, window="A")
    assert res["frac_time_on"] >= 0.30, res["frac_time_on"]
    assert res["label"] == "sustained", res


def test_mixed_returned_and_nonreturned_is_not_discrete():
    """A normal self-terminating event + a trailing NON-returning ON segment must
    NOT be labeled discrete (critical review repro: discrete_events [True, False]).

    Both segments are short (<400ms) and frac_on<0.30, so only the all-returned
    requirement can catch it.
    """
    t = _times(300.0)
    ext = _rect(t, 50.0, 90.0) + _rect(t, 280.0, 300.0)  # 2nd runs to end -> cannot return
    evs = detect_events(ext, DT)
    assert [e["returned"] for e in evs] == [True, False], [e["returned"] for e in evs]
    op = mean_field(1.0)
    res = classify_run(ext, DT, op, window="A")
    assert res["frac_time_on"] < FRAC_TIME_ON_MAX, res["frac_time_on"]
    assert res["all_returned"] is False
    assert res["label"] == "sustained", res


def test_runaway():
    t = _times(250.0)
    ext = np.where(t >= 50.0, 0.7, 0.0)  # high and stays high to the end (not returned)
    op = mean_field(1.0)
    res = classify_run(ext, DT, op, window="A")
    assert res["label"] == "runaway", res


def test_captured_high_window_B_only():
    """Elevated, non-returning, moderate plateau (<400ms) + field at the HIGH root.

    Window B (bistable, 2 roots) -> captured_high. Same trace under window A
    (no capture check) must NOT be labeled captured_high.
    """
    t = _times(300.0)
    ext = _rect(t, 50.0, 250.0)  # 200ms, < SUSTAINED_MS, does not return
    opB = mean_field(1.0, w_ee_mult=1.4)
    assert len(opB["roots"]) >= 2
    hi = opB["roots"][-1]["nuE"]
    rE_final = np.full((16, 16), hi)  # field settled at the high root
    resB = classify_run(ext, DT, opB, rE_final=rE_final, window="B")
    assert resB["label"] == "captured_high", resB
    resA = classify_run(ext, DT, mean_field(1.0), rE_final=rE_final, window="A")
    assert resA["label"] != "captured_high", resA


def test_extinction_when_below_threshold():
    t = _times(300.0)
    ext = np.full_like(t, 0.5 * EVENT_ON_FRAC)  # always below ON threshold
    op = mean_field(1.0)
    res = classify_run(ext, DT, op, window="A")
    assert res["label"] == "extinction_only", res
    assert res["n_events"] == 0


def test_coherence_measure_rejects_noise_speckle():
    """End-to-end: low-σ OU noise -> coherence-active-fraction stays sub-threshold.

    Locks the v1.1 coherence amendment: per-pixel speckle that crossed the RAW
    active-fraction floor (raw_ext ~0.05 at σ=2.0) does NOT cross the coherence
    floor (EVENT_ON_FRAC=0.05), so a sub-event noise run classifies extinction.
    """
    op = mean_field(1.0)
    n, L = 64, 16.0
    stim = make_ou_noise(n, L, DT, sigma_noise=2.0, seed=0)
    ext, _front, coh = integrate_lif_field(
        op, stim, dt=DT, t_max=1200.0, b_a=0.0, n=n, L=L,
        ell_par=ELL_PAR, ell_perp=ELL_PERP, l_inh=L_INH, coh_len=ELL_PAR,
    )
    res = classify_run(coh, DT, op, window="A")
    assert res["label"] == "extinction_only", res


# ---------------------------------------------------------------------------
# OU noise
# ---------------------------------------------------------------------------

def test_noise_amp_zero_is_identity():
    """sigma_noise=0 -> the field run is identical to the deterministic run."""
    op = mean_field(1.0)
    n, L = 32, 16.0
    zero_noise = make_ou_noise(n, L, DT, sigma_noise=0.0, seed=0)
    assert zero_noise(0.0) == 0.0 and zero_noise(10.0) == 0.0
    ext_noise, _ = integrate_lif_field(op, zero_noise, dt=DT, t_max=40.0, n=n, L=L)
    ext_det, _ = integrate_lif_field(op, lambda t: 0.0, dt=DT, t_max=40.0, n=n, L=L)
    assert np.array_equal(ext_noise, ext_det)


def test_noise_seed_reproducible_and_distinct():
    n, L = 32, 16.0
    f0a = make_ou_noise(n, L, DT, sigma_noise=2.0, seed=7)
    f0b = make_ou_noise(n, L, DT, sigma_noise=2.0, seed=7)
    f1 = make_ou_noise(n, L, DT, sigma_noise=2.0, seed=8)
    a = [f0a(i * DT) for i in range(5)]
    b = [f0b(i * DT) for i in range(5)]
    c = [f1(i * DT) for i in range(5)]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))   # same seed -> identical
    assert not np.array_equal(a[-1], c[-1])                  # different seed -> differ


def test_noise_steady_state_std_matches_sigma():
    n, L = 48, 16.0
    sigma = 2.0
    f = make_ou_noise(n, L, DT, sigma_noise=sigma, tau_noise=5.0, ell_noise=0.5, seed=3)
    # advance well past tau_noise, then sample the per-pixel std over the field
    field = None
    for i in range(4000):
        field = f(i * DT)
    assert 0.5 * sigma < field.std() < 1.6 * sigma, field.std()


# ---------------------------------------------------------------------------
# Per-operating-point detector recalibration (contract §10.2)
# The amplitude bar (event_on_frac on the coherence active-fraction) is set
# PER operating point, between the σ=0/σ_ref floor and the deterministic-kick
# event-peak — NOT a single constant calibrated at one drive (that single-point
# calibration is the §9.7 confound: at near-zero rest, slow noise hovers above a
# fixed bar between events -> false "sustained").
# ---------------------------------------------------------------------------

def test_event_on_frac_from_refs_is_midpoint_between_floor_and_peak():
    """§10.2(A): bar = floor + 0.5*(peak-floor), pre-registered midpoint rule."""
    from src.sef_hfo_events import event_on_frac_from_refs
    bar = event_on_frac_from_refs(floor=0.02, peak=0.30)
    assert abs(bar - 0.16) < 1e-12, bar
    # comfortably above the noise floor and below the genuine event peak
    assert 0.02 < bar < 0.30


def test_event_on_frac_from_refs_loud_fails_when_floor_not_below_peak():
    """§10.2(A): if noise floor reaches the event amplitude, the operating point
    is undetectable — loud-fail, never silently emit a usable bar."""
    import pytest

    from src.sef_hfo_events import UndetectableOperatingPoint, event_on_frac_from_refs
    with pytest.raises(UndetectableOperatingPoint):
        event_on_frac_from_refs(floor=0.30, peak=0.30)
    with pytest.raises(UndetectableOperatingPoint):
        event_on_frac_from_refs(floor=0.35, peak=0.30)


def test_recalibrated_bar_rejects_subevent_flicker_as_extinction():
    """§9.7 fix: a near-zero-rest operating point where slow noise hovers at
    amplitude 0.08 — ABOVE the default fixed bar 0.05 (mis-called sustained) but
    BELOW a per-op bar recalibrated to 0.16 (correctly extinction)."""
    t = _times(300.0)
    flicker = np.full_like(t, 0.08)   # above default 0.05, below recalibrated 0.16
    op = mean_field(0.6)
    # default fixed bar (0.05): the flicker reads as continuous activity
    assert classify_run(flicker, DT, op, window="A")["label"] == "sustained"
    # per-op recalibrated bar (0.16): correctly extinction (no genuine event)
    res = classify_run(flicker, DT, op, window="A", event_on_frac=0.16)
    assert res["label"] == "extinction_only", res
    assert res["n_events"] == 0


def test_recalibrated_bar_preserves_clean_event_and_shape_constants():
    """Invariant (§10.2 B): recalibrating only the amplitude bar must NOT drop a
    genuine event nor change the shape/time classification — a clean 40ms,
    amplitude-0.30 self-terminating pulse stays one discrete event at bar 0.16."""
    t = _times(300.0)
    ext = _rect(t, 90.0, 130.0, val=0.30)
    op = mean_field(0.6)
    res = classify_run(ext, DT, op, window="A", event_on_frac=0.16)
    assert res["label"] == "discrete_events", res
    assert res["n_events"] == 1, res


def test_calibrate_detector_uses_worst_floor_and_kick_peak():
    """§10.2(A)/§10.5: bar from reference coherence series (σ=0/σ_ref, worst-case
    floor = max over references) and the deterministic-kick peak. Pure over the
    series — the runner supplies them (anti-circular: never the noise grid)."""
    from src.sef_hfo_events import calibrate_detector
    ref0 = np.full(100, 0.02)            # σ=0 floor
    refR = np.full(100, 0.03)            # σ_ref (sub-threshold) floor — higher
    kick = np.concatenate([np.zeros(40), np.full(20, 0.30), np.zeros(40)])  # event peak 0.30
    cal = calibrate_detector([ref0, refR], kick)
    assert abs(cal["floor"] - 0.03) < 1e-12, cal      # worst (max) of the references
    assert abs(cal["peak"] - 0.30) < 1e-12, cal
    assert abs(cal["event_on_frac"] - (0.03 + 0.5 * 0.27)) < 1e-12, cal  # midpoint = 0.165


def test_calibrate_detector_loud_fails_at_undetectable_operating_point():
    """§10.2(A): if sub-threshold noise reaches the event amplitude (floor >= peak),
    the operating point is non-excitable/undetectable -> loud-fail (drive excluded
    from the interictal band, §10.3)."""
    import pytest

    from src.sef_hfo_events import UndetectableOperatingPoint, calibrate_detector
    ref_hot = np.full(100, 0.30)         # noise floor already at event amplitude
    kick = np.full(100, 0.30)
    with pytest.raises(UndetectableOperatingPoint):
        calibrate_detector([ref_hot], kick)
