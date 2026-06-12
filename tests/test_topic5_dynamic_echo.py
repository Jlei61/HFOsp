"""Synthetic TDD for the Stage-2b dynamic-pattern-echo pure-math module.

Spec: docs/superpowers/specs/2026-06-11-topic5-stage2b-dynamic-pattern-echo-design.md
Plan: docs/superpowers/plans/2026-06-11-topic5-stage2b-dynamic-pattern-echo.md

The sign-flip test (Task 1) and the max-null-not-false-positive test (Task 4) are
the load-bearing non-circular checks: align_score sign reversed = whole conclusion
reversed; max-over-time null absorbs the time-selection so random data is not
falsely significant.
"""
import numpy as np
import pytest

from src.topic5_dynamic_echo import (
    align_score,
    activation_and_slope,
    echo_curve,
    echo_curve_null,
    echo_peak_pvalue,
    slope_latencies,
    ramp_strength,
    region_aggregate,
)


# ---------------------------------------------------------------------------
# Task 1 — align_score (sign-locked, §2.1)
# ---------------------------------------------------------------------------
def test_align_score_intensity_positive_when_template_early_is_stronger():
    # template_rank small = source/early. value large = stronger. If template-early
    # contacts ARE stronger, value should DECREASE with rank -> Spearman<0 -> align=+.
    template_rank = np.arange(10, dtype=float)            # 0=earliest
    value = (10 - template_rank) + 0.01                   # earliest = strongest
    assert align_score(template_rank, value, kind="intensity", min_ch=8) > 0.9


def test_align_score_intensity_sign_flips():
    template_rank = np.arange(10, dtype=float)
    value = template_rank.copy()                          # earliest = WEAKEST (anti)
    assert align_score(template_rank, value, kind="intensity", min_ch=8) < -0.9


def test_align_score_latency_positive_when_template_early_is_earlier():
    # latency_rank small = earlier. template-early earlier -> ranks agree -> Spearman+ -> align=+.
    template_rank = np.arange(10, dtype=float)
    latency_rank = np.arange(10, dtype=float)
    assert align_score(template_rank, latency_rank, kind="latency", min_ch=8) > 0.9


def test_align_score_too_few_common_is_nan():
    a = np.array([0.0, 1, 2, np.nan, np.nan, np.nan, np.nan, np.nan])
    b = np.arange(8, dtype=float)
    assert np.isnan(align_score(a, b, kind="intensity", min_ch=8))


def test_align_score_bad_kind_raises():
    a = np.arange(10, dtype=float)
    with pytest.raises(ValueError):
        align_score(a, a, kind="bogus", min_ch=8)


# ---------------------------------------------------------------------------
# Task 2 — activation_and_slope (Savitzky-Golay derivative, §2)
# ---------------------------------------------------------------------------
def test_activation_and_slope_derivative_sign():
    hop = 0.1
    n = 200
    z = np.zeros((1, n))
    z[0, 50:] = np.linspace(0, 10, n - 50)        # rising ramp after frame 50
    act, dz = activation_and_slope(z, hop=hop)
    assert np.allclose(act, z)                     # activation = z itself
    assert dz[0, 100] > 0                          # positive slope on the ramp
    assert abs(dz[0, 20]) < 0.05                   # ~0 on the flat baseline


# ---------------------------------------------------------------------------
# Task 3 — echo_curve (align_score(t) + echo_peak/echo_mean)
# ---------------------------------------------------------------------------
def test_echo_curve_peaks_when_template_early_strengthens_over_time():
    n_ch, n_t = 12, 100
    t = np.arange(n_t) * 0.1                       # 0..10 s
    template_rank = np.arange(n_ch, dtype=float)
    value = np.zeros((n_ch, n_t))
    for c in range(n_ch):
        value[c] = np.clip((t - 0.3 * c), 0, None)   # earlier-rank contacts rise first
    res = echo_curve(template_rank, value, t, kind="intensity", min_ch=8,
                     mean_window=(0.0, 5.0))
    assert res["echo_peak"] > 0.5
    assert 0.0 <= res["t_peak"] <= 10.0
    assert res["echo_mean"] > 0.3


def test_echo_curve_flat_when_no_priority():
    n_ch, n_t = 12, 100
    t = np.arange(n_t) * 0.1
    template_rank = np.arange(n_ch, dtype=float)
    rng = np.random.default_rng(0)
    value = rng.standard_normal((n_ch, n_t))       # no template relationship
    res = echo_curve(template_rank, value, t, kind="intensity", min_ch=8,
                     mean_window=(0.0, 5.0))
    assert abs(res["echo_mean"]) < 0.4


# ---------------------------------------------------------------------------
# Task 4 — echo_curve_null (max-over-time null, §2.2 — THE key gate)
# ---------------------------------------------------------------------------
def test_max_null_not_falsely_significant_on_random():
    n_ch, n_t = 14, 80
    t = np.arange(n_t) * 0.1
    template_rank = np.arange(n_ch, dtype=float)
    rng = np.random.default_rng(3)
    value = rng.standard_normal((n_ch, n_t))                 # no real echo
    obs = echo_curve(template_rank, value, t, kind="intensity", min_ch=8,
                     mean_window=(0, 5))["echo_peak"]
    null = echo_curve_null(template_rank, value, t, kind="intensity", min_ch=8,
                           null_mode="channel", blocks=None, B=300, rng=rng)
    p = echo_peak_pvalue(obs, null)
    assert p > 0.05                                         # max-null absorbs the time-selection


def test_max_null_significant_on_real_echo():
    n_ch, n_t = 14, 80
    t = np.arange(n_t) * 0.1
    template_rank = np.arange(n_ch, dtype=float)
    value = np.zeros((n_ch, n_t))
    for c in range(n_ch):
        value[c] = np.clip(t - 0.25 * c, 0, None)            # strong template echo
    obs = echo_curve(template_rank, value, t, kind="intensity", min_ch=8,
                     mean_window=(0, 5))["echo_peak"]
    null = echo_curve_null(template_rank, value, t, kind="intensity", min_ch=8,
                           null_mode="channel", blocks=None, B=300,
                           rng=np.random.default_rng(4))
    assert echo_peak_pvalue(obs, null) < 0.05


# ---------------------------------------------------------------------------
# Task 5 — slope_latencies (eligibility-gated, §3.2)
# ---------------------------------------------------------------------------
def test_slope_latencies_orders_by_rise_time_and_gates_flat():
    hop = 0.1
    t = np.arange(100) * hop                              # 0..10s
    z = np.zeros((3, 100))
    z[0, 10:] = np.clip(t[10:] - t[10], 0, 8)            # rises early (contact 0)
    z[1, 40:] = np.clip(t[40:] - t[40], 0, 8)            # rises later (contact 1)
    z[2] = 0.1 * np.random.default_rng(0).standard_normal(100)  # flat noise -> ineligible
    out = slope_latencies(z, t_axis=t, z_min=2.0, delta_min=1.0)
    assert out["t_peak"][0] < out["t_peak"][1]           # earlier contact peaks first
    assert np.isnan(out["t_peak"][2])                    # flat channel gated to NaN
    assert np.isnan(out["t50_rise"][2])


# ---------------------------------------------------------------------------
# Task 6 — ramp_strength (per-window AUC + slope, §3.3)
# ---------------------------------------------------------------------------
def test_ramp_strength_early_window_higher_for_early_riser():
    hop = 0.1
    t = np.arange(100) * hop
    z = np.zeros((2, 100))
    z[0, 0:] = np.clip(t, 0, 5)                 # rises immediately -> high 0-2 AUC
    z[1, 70:] = np.clip(t[70:] - t[70], 0, 5)   # rises late -> low 0-2 AUC
    out = ramp_strength(z, t_axis=t, windows=((0, 2), (2, 5), (5, 10)))
    assert out["AUC"][(0, 2)][0] > out["AUC"][(0, 2)][1]


# ---------------------------------------------------------------------------
# Task 7 — region_aggregate (§3.4)
# ---------------------------------------------------------------------------
def test_region_aggregate_medians_by_group_and_drops_singletons():
    value = np.array([1.0, 3.0, 10.0, 20.0, 5.0])
    groups = np.array(["A", "A", "B", "B", "C"])     # C is singleton -> dropped (min 2)
    reg_val, reg_labels = region_aggregate(value, groups, min_group=2)
    assert reg_labels == ["A", "B"]
    assert np.allclose(reg_val, [2.0, 15.0])
