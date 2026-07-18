"""Topic 5 间期单事件包络场 — 纯数学的承重不变量。

只测「错了会静默出一张看起来正常但其实是错的图」的那几条。三条最要命的：
(1) unit-support vs 模板加权 —— 用模板 support 给单事件加权是循环论证；
(2) 事件级 t=0 不能被一个没有真峰的弱通道锚跑（139 的 HRB1）；
(3) 代表事件挑选不能被单一维度的长尾带偏，且绝不许读传播斜率。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.topic5_interictal_event_field import (
    ENVELOPE_QUANTITY, baseline_robust_z, detector_excursions, display_window_ms,
    envelope_peak_times, envelope_peak_z, estimator_agreement, event_level_t0,
    event_spreads_ms, frame_times_ms, group_event_interval,
    peak_times_in_event, pooled_vmax, rank_candidates, select_medoid_event,
    stored_lag_rel_ms,
    template_likeness, values_at,
)
from scripts.paper_figures.fig1_spectrogram_utils import (
    centroid_alignment_audit,
    compute_group_event_spectrogram_stack,
)


# ---------------------------------------------------------------- 合同 2：amplitude 不是 energy
def test_quantity_is_amplitude_not_energy():
    """return_hil_enve_norm = sum(|hilbert(x)|)，没有平方。面向读者的字样不许写 energy。"""
    assert "amplitude" in ENVELOPE_QUANTITY
    assert "energy" not in ENVELOPE_QUANTITY.lower()
    assert "power" not in ENVELOPE_QUANTITY.lower()


# ---------------------------------------------------------------- 事件跨度
def test_event_spread_uses_participants_only():
    lag = np.array([[10.100], [10.130], [10.115], [99.0]])      # 最后一个是非参与的幽灵值
    bools = np.array([[True], [True], [True], [False]])
    assert event_spreads_ms(lag, bools)[0] == pytest.approx(30.0)   # 不是 88900


def test_event_spread_is_nan_not_zero_when_unmeasurable():
    """测不了必须 NaN。返回 0 会被 medoid 当成「跨度最典型」的完美候选。"""
    lag = np.array([[1.0], [2.0], [3.0]])
    bools = np.array([[True], [True], [False]])
    assert np.isnan(event_spreads_ms(lag, bools, min_participating=3)[0])


def test_stored_lag_is_relative_and_ghosts_are_nan():
    """存档绝对值在拼接时间轴上，只有事件内相对差有意义；非参与必须 NaN 不是 0。"""
    out = stored_lag_rel_ms(np.array([10.100, 10.130, 10.115, 99.0]),
                            np.array([True, True, True, False]))
    assert out[0] == pytest.approx(0.0) and out[1] == pytest.approx(30.0)
    assert np.isnan(out[3])


# ---------------------------------------------------------------- 合同 3：代表事件挑选
def test_medoid_prefers_typical_over_the_single_prettiest():
    """承重：单挑「最像模板」会系统性挑到长尾事件。medoid 必须挑各维都典型的那个。"""
    cand = [0, 1, 2]
    feats = {0: [96.0, 10.0, 40.0], 1: [27.0, 10.0, 12.0], 2: [80.0, 9.0, 35.0]}
    pos, _ = select_medoid_event(cand, feats, [27.0, 10.0, 12.0])
    assert pos == 1


def test_medoid_balances_dimensions_not_just_spread():
    """多维的意义：跨度对但参与数离谱的事件不该赢。"""
    cand = [0, 1]
    feats = {0: [27.0, 3.0, 12.0], 1: [31.0, 10.0, 12.0]}      # 0 跨度更准但只有 3 个触点
    pos, _ = select_medoid_event(cand, feats, [27.0, 10.0, 12.0],
                                 scale=[4.0, 1.0, 4.0])
    assert pos == 1


def test_medoid_rejects_dimension_mismatch_loudly():
    with pytest.raises(ValueError, match="dim"):
        select_medoid_event([0], {0: [1.0, 2.0]}, [1.0, 2.0, 3.0])


def test_rank_candidates_gates_and_orders_by_template_likeness():
    tr = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    masked = np.stack([tr[::-1], tr, tr + 0.01], axis=1)        # 0=反向 1=完美 2=近乎完美
    bools = np.ones((6, 3), bool)
    cand, rhos, info = rank_candidates(masked, bools, np.array([0, 0, 0]), 0, tr,
                                       np.array([30.0, 30.0, 30.0]), 30.0, 6.0, top_k=2)
    assert cand[0] in (1, 2) and 0 not in cand                   # 反向那个排最后，被 top_k 挡掉
    assert rhos[1] == pytest.approx(1.0)


def test_rank_candidates_breaks_ties_by_typicality_not_by_recording_order():
    """承重回归：epilepsiae_958 实测 8 万事件里 top-25 全是 rho=1.000。稳定排序会让 top_k
    恒取事件序号最小的一批 = 录制最早期的事件，候选池带隐藏时间偏差。
    并列必须按「离典型值多远」打破。"""
    tr = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    n = 40
    masked = np.tile(tr[:, None], (1, n))                        # 全部 rho=1.000，全并列
    bools = np.ones((6, n), bool)
    spreads = np.full(n, 99.0)
    spreads[37] = 27.0                                           # 只有靠后的 #37 跨度典型
    cand, _, info = rank_candidates(masked, bools, np.zeros(n, int), 0, tr, spreads,
                                    27.0, 6.0, top_k=3)
    assert info["n_tied_at_top"] == n                             # 确认确实全并列
    assert cand[0] == 37                                          # 典型的那个排第一，不是 #0
    assert 0 not in cand[:1]


def test_rank_candidates_drops_events_with_unmeasurable_spread():
    tr = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    masked = np.stack([tr, tr], axis=1)
    cand, _, _ = rank_candidates(masked, np.ones((5, 2), bool), np.array([0, 0]), 0, tr,
                                 np.array([np.nan, 40.0]), 40.0, 5.0)
    assert cand == [1]


def test_rank_candidates_raises_rather_than_returning_a_thin_event():
    tr = np.arange(5.0)
    masked = np.full((5, 1), np.nan); masked[:2, 0] = [0.0, 1.0]
    bools = np.zeros((5, 1), bool); bools[:2, 0] = True
    with pytest.raises(ValueError, match="gate"):
        rank_candidates(masked, bools, np.array([0]), 0, tr, np.array([27.0]), 27.0, 5.0,
                        min_participating=5)


def test_template_likeness_needs_enough_common_channels():
    assert np.isnan(template_likeness(np.array([0.0, 1.0, np.nan]), np.array([0.0, 1.0, 2.0]),
                                      np.array([True, True, False]), min_common=3))


# ---------------------------------------------------------------- 基线 z
def test_baseline_excludes_the_event_and_its_guard():
    """基线若含事件本体，事件把 MAD 撑大、把自己压扁 -> 图上波看不见。"""
    fs, n = 512.0, 2048
    t = np.arange(n) / fs
    env = np.ones((1, n)) + np.random.default_rng(0).normal(0, 0.01, (1, n))
    env[0, (t >= 1.0) & (t <= 1.2)] = 50.0
    z = baseline_robust_z(env, t, 1.0, 1.2, guard_sec=0.05)
    assert np.nanmax(z[0, (t >= 1.0) & (t <= 1.2)]) > 100
    assert abs(np.nanmedian(z[0, t < 0.9])) < 1.0


def test_flat_channel_becomes_nan_not_fake_signal():
    """MAD=0 的死通道除以 0 -> 必须 NaN（没测到），不能变 inf 点亮整张场。"""
    t = np.arange(2048) / 512.0
    assert np.isnan(baseline_robust_z(np.ones((1, 2048)), t, 1.0, 1.2, guard_sec=0.05)).all()


def test_baseline_raises_when_pad_too_thin():
    t = np.arange(100) / 512.0
    with pytest.raises(ValueError, match="baseline too short"):
        baseline_robust_z(np.ones((1, 100)), t, t[0], t[-1])


# ---------------------------------------------------------------- 事件级 t=0
def test_event_level_t0_is_the_first_sustained_rise_across_all_participants():
    fs, n = 512.0, 1024
    t = np.arange(n) / fs
    z = np.zeros((2, n))
    z[0, 400:500] = 40.0                       # ch0 早
    z[1, 600:700] = 40.0                       # ch1 晚
    t0 = event_level_t0(z, t, np.array([True, True]), 0.0, 2.0)
    assert t0 == pytest.approx(t[400], abs=2 / fs)


def test_event_level_t0_is_not_dragged_early_by_pre_event_noise_near_z3():
    """承重回归 —— 这是 epilepsiae_1146 上真实发生的 bug。

    审阅提的「z>=3 持续 5ms」规则：z=3 只是 3 倍 MAD，噪声里常见，撑 5ms 毫无难度。
    实测该规则把 t0 钉在打包窗左缘，而真事件的包络峰在 +96..+145ms —— 整个显示窗
    画的是事件前的爬升。阈值必须对事件自身幅度自标定。
    """
    fs, n = 512.0, 1024
    t = np.arange(n) / fs
    z = np.zeros((2, n))
    z[0, 50:400] = 3.4                         # 事件前：参与触点在 z≈3 附近晃了 700ms
    z[1, 50:400] = 3.2
    z[0, 600:700] = 40.0                       # 真事件
    z[1, 640:740] = 38.0
    t0 = event_level_t0(z, t, np.array([True, True]), 0.0, 2.0)
    assert t0 > t[550]                         # 不许被 z≈3 的噪声拽到 t[50]
    assert t0 == pytest.approx(t[600], abs=6 / fs)


def test_event_level_t0_ignores_a_weak_channel_with_no_real_peak():
    """139 的 HRB1：探测器算它参与，但包络只有噪声毛刺，撑不住 —— 不许锚上去。"""
    fs, n = 512.0, 1024
    t = np.arange(n) / fs
    z = np.zeros((2, n))
    z[0, 100] = 9.0                            # 单点毛刺
    z[1, 600:700] = 40.0
    t0 = event_level_t0(z, t, np.array([True, True]), 0.0, 2.0)
    assert t0 == pytest.approx(t[600], abs=4 / fs)


def test_event_level_t0_ignores_non_participating_contacts():
    fs, n = 512.0, 1024
    t = np.arange(n) / fs
    z = np.zeros((2, n))
    z[0, 100:200] = 40.0                       # 未参与的热触点（1146 的 SCL9）
    z[1, 600:700] = 40.0
    t0 = event_level_t0(z, t, np.array([False, True]), 0.0, 2.0)
    assert t0 == pytest.approx(t[600], abs=4 / fs)


def test_event_level_t0_threshold_self_scales_to_the_event_amplitude():
    """弱事件和强事件都该锚到各自的起始，而不是被一个绝对阈值切在不同的相对位置。"""
    fs, n = 512.0, 1024
    t = np.arange(n) / fs
    out = []
    for amp in (12.0, 120.0):
        z = np.zeros((1, n))
        z[0, 300:500] = np.linspace(0, amp, 200)
        out.append(event_level_t0(z, t, np.array([True]), 0.0, 2.0, frac_of_peak=0.25, min_z=1.0))
    assert out[0] == pytest.approx(out[1], abs=4 / fs)    # 两者锚在同一个相对位置


def test_event_level_t0_raises_when_nothing_ever_rises():
    t = np.arange(1024) / 512.0
    with pytest.raises(ValueError, match="usable envelope onset"):
        event_level_t0(np.zeros((2, 1024)), t, np.array([True, True]), 0.0, 2.0)


def test_event_level_t0_raises_when_no_contact_participates():
    t = np.arange(1024) / 512.0
    with pytest.raises(ValueError, match="no participating contact"):
        event_level_t0(np.ones((2, 1024)), t, np.array([False, False]), 0.0, 2.0)


# ---------------------------------------------------------------- 探测器自己的事件定义
def test_excursions_split_two_events_separated_by_more_than_min_gap():
    """承重：一个 200ms 窗里可以装下不止一次事件。间隔 > min_gap 的必须切成两次。"""
    fs, n = 512.0, 1024
    t = np.arange(n) / fs
    env = np.ones((1, n))
    env[0, 100:160] = 20.0                       # 第一次: ~117ms
    env[0, 300:360] = 20.0                       # 第二次: 隔了 274ms >> min_gap
    ex = detector_excursions(env, t, min_gap_ms=20.0, min_last_ms=50.0)
    assert len(ex[0]) == 2


def test_excursions_merge_two_blips_closer_than_min_gap():
    fs, n = 512.0, 1024
    t = np.arange(n) / fs
    env = np.ones((1, n))
    env[0, 100:130] = 20.0
    env[0, 135:165] = 20.0                       # 只隔 ~10ms < min_gap=20ms -> 合并成 ~127ms 一次
    ex = detector_excursions(env, t, min_gap_ms=20.0, min_last_ms=50.0)
    assert len(ex[0]) == 1
    assert (ex[0][0][1] - ex[0][0][0]) == pytest.approx(0.127, abs=0.01)


def test_excursions_reject_a_merged_run_longer_than_max_last():
    """max_last 也是探测器合同的一部分：合并后超长的不算一次 HFO 事件。"""
    fs, n = 512.0, 1024
    t = np.arange(n) / fs
    env = np.ones((1, n))
    env[0, 100:225] = 20.0                       # ~244ms > max_last=200ms
    assert detector_excursions(env, t, max_last_ms=200.0)[0] == []


def test_excursions_drop_blips_shorter_than_min_last():
    fs, n = 512.0, 1024
    t = np.arange(n) / fs
    env = np.ones((1, n))
    env[0, 100:110] = 20.0                       # ~20ms < min_last=50ms
    assert detector_excursions(env, t, min_last_ms=50.0)[0] == []


def test_group_event_is_where_most_contacts_fire_together():
    """群体事件本来就是"多通道同时超阈"定义的，所以该由并发度定，不由谁最响定。"""
    fs, n = 512.0, 1024
    t = np.arange(n) / fs
    env = np.ones((3, n))
    for i in range(3):
        env[i, 100:200] = 20.0                   # 三个通道一起 -> 这才是群体事件
    env[0, 600:700] = 99.0                       # ch0 后面还有一次很响的独奏
    ex = detector_excursions(env, t)
    lo, hi = group_event_interval(ex, t)
    assert lo == pytest.approx(t[100], abs=4 / fs) and hi == pytest.approx(t[199], abs=4 / fs)


def test_peak_times_ignore_the_later_separate_event():
    """承重回归 —— epilepsiae_1146 TB 上真实发生的：低轴端触点在 ~120ms 还有一团，
    那是被 min_gap 分开的**另一次事件**。整窗 argmax 会把它当成"这次的峰"，传播读出被拖长一倍。"""
    fs, n = 512.0, 1024
    t = np.arange(n) / fs
    env = np.ones((2, n))
    env[0, 100:200] = 20.0
    env[1, 120:220] = 20.0
    env[1, 600:700] = 99.0                       # ch1 稍后的另一次事件，而且更响
    ex = detector_excursions(env, t)
    iv = group_event_interval(ex, t)
    pk = peak_times_in_event(env, t, ex, iv)
    assert pk[1] < t[250]                        # 不许锁到 t[600] 那次
    assert np.isfinite(pk[0])


def test_peak_times_are_nan_for_a_contact_not_in_this_event():
    fs, n = 512.0, 1024
    t = np.arange(n) / fs
    env = np.ones((2, n))
    env[0, 100:200] = 20.0
    env[1, 700:800] = 20.0                       # ch1 只在别处放电
    ex = detector_excursions(env, t)
    pk = peak_times_in_event(env, t, ex, group_event_interval(ex, t))
    assert np.isfinite(pk[0]) and np.isnan(pk[1])


# ---------------------------------------------------------------- 峰值 / 一致性
def test_envelope_peak_times_and_heights():
    fs, n = 512.0, 512
    t = np.arange(n) / fs
    z = np.zeros((2, n)); z[0, 100] = 5.0; z[1, 200] = 9.0
    assert envelope_peak_times(z, t, 0.0, 1.0)[1] == pytest.approx(200 / fs)
    assert envelope_peak_z(z, t, 0.0, 1.0)[1] == pytest.approx(9.0)


def test_estimator_agreement_is_offset_free():
    stored = np.array([0.0, 10.0, 20.0, 30.0])
    rho, mad, n = estimator_agreement(stored + 137.0, stored)
    assert rho == pytest.approx(1.0) and mad == pytest.approx(0.0) and n == 4


def test_estimator_agreement_reports_disagreement():
    rho, _, n = estimator_agreement(np.array([0.0, 10.0, 20.0]), np.array([20.0, 10.0, 0.0]))
    assert rho == pytest.approx(-1.0) and n == 3


# ---------------------------------------------------------------- 时间窗 / 取帧 / 色标
def test_window_tightens_to_the_measured_peaks_not_the_200ms_ruler():
    """27ms 的事件不该在 200ms 的尺子里空放。min_post 是地板，所以窗 = -10..+40。"""
    t_lo, t_hi = display_window_ms([np.array([0.0, 10.0, 27.0])])
    assert (t_lo, t_hi) == (-15.0, 42.0)          # 最早的峰前留 15ms 铺垫，最晚的峰后留 15ms
    assert (t_hi - t_lo) < 200.0


def test_window_covers_every_peak_it_draws():
    """承重回归 —— epilepsiae_1146 TB 上真实发生的 bug：窗口用**存档跨度**(57ms)算，
    但图上画的是**包络峰**(跨到 120ms)，peak-order 线直接冲出热图右缘。画什么就用什么定窗。"""
    peaks = [np.array([0.0, 20.0, 40.0]), np.array([35.0, 60.0, 120.0])]   # TB 拖到 120
    t_lo, t_hi = display_window_ms(peaks)
    assert t_hi >= 120.0                       # 必须把 120 那个峰包进来
    assert t_lo <= 0.0


def test_window_is_shared_so_TA_and_TB_stay_comparable():
    a, b = np.array([0.0, 27.0]), np.array([10.0, 80.0])
    assert display_window_ms([a, b]) == display_window_ms([b, a])


def test_window_includes_peaks_earlier_than_the_default_pre_roll():
    t_lo, _ = display_window_ms([np.array([-40.0, 0.0, 20.0])], pre_ms=10.0, margin_ms=15.0)
    assert t_lo <= -55.0                       # 不许把 -40 的峰切掉


def test_window_is_capped_so_one_runaway_peak_cannot_blow_it_up():
    assert display_window_ms([np.array([0.0, 5000.0])])[1] == 190.0


def test_window_handles_an_all_nan_event():
    assert display_window_ms([np.array([np.nan, np.nan])])[1] == 40.0


def test_frame_times_span_the_window_inclusive():
    ft = frame_times_ms(-10.0, 50.0, 7)
    assert len(ft) == 7 and ft[0] == -10.0 and ft[-1] == 50.0


def test_values_at_averages_a_short_window_to_kill_single_sample_flicker():
    fs, n = 512.0, 512
    t = np.arange(n) / fs
    z = np.zeros((1, n)); z[0, 100] = 100.0     # 单采样点尖刺
    spike = values_at(z, t, 0.0, t[100] * 1000, avg_ms=0.0)
    smooth = values_at(z, t, 0.0, t[100] * 1000, avg_ms=10.0)
    assert smooth < spike                        # 平均窗压掉了闪烁


def test_pooled_vmax_is_shared_across_both_events():
    """不逐行不逐事件归一化 —— 否则 A/B 强弱不可比，反向也可能是归一化 artifact。"""
    a, b = np.array([[1.0, 2.0]]), np.array([[100.0, 200.0]])
    assert pooled_vmax([a, b]) == pooled_vmax([b, a])
    assert pooled_vmax([a, b]) > 50.0            # 大的那个事件参与决定色标


def test_pooled_vmax_floors_at_one_and_ignores_nan():
    assert pooled_vmax([np.array([0.0, np.nan, 0.1])]) == 1.0


# ---------------------------------------------------------------- Fig1a 公共 producer 合同
def _fig1a_payload(amplitudes=(20.0, 20.0), starts=(0.09, 0.19)):
    fs, duration = 1000.0, 0.32
    t = np.arange(int(fs * duration)) / fs
    x = np.zeros((len(amplitudes), t.size))
    for row, (amp, start) in enumerate(zip(amplitudes, starts)):
        m = (t >= start) & (t < start + 0.045)
        x[row, m] = amp * np.sin(2 * np.pi * 150 * t[m])
    return compute_group_event_spectrogram_stack(
        x, fs, np.asarray([duration]), spec_window="hamming", spec_win_sec=0.05,
        spec_overlap_sec=0.04, spec_freq_range=(50.0, 300.0), gaussian_sigma=1.5,
        enhancement_threshold=0.70,
    )


def test_fig1a_centroid_tracks_the_displayed_enhancement():
    normed, times, freqs, centers = _fig1a_payload()
    assert centers[0, 0, 0] < centers[1, 0, 0]
    audit = centroid_alignment_audit(
        normed, times, freqs, centers, window_sec=0.32, acceptance_threshold=0.70,
    )
    assert audit["all_centroids_pass"] and audit["n_centroids"] == 2


def test_fig1a_normalises_each_contact_within_the_event():
    normed, _, freqs, _ = _fig1a_payload(amplitudes=(100.0, 1.0), starts=(0.12, 0.12))
    n_freq = len(freqs)
    assert normed[:n_freq].max() == pytest.approx(1.0)
    assert normed[n_freq:].max() == pytest.approx(1.0)


def test_fig1a_centroid_uses_the_dominant_connected_component():
    fs, duration = 1000.0, 0.32
    t = np.arange(int(fs * duration)) / fs
    x = np.zeros((1, t.size))
    for amp, start in ((20.0, 0.085), (14.0, 0.205)):
        m = (t >= start) & (t < start + 0.04)
        x[0, m] += amp * np.sin(2 * np.pi * 150 * t[m])
    _, _, _, centers = compute_group_event_spectrogram_stack(
        x, fs, np.asarray([duration]), spec_window="hamming", spec_win_sec=0.05,
        spec_overlap_sec=0.04, spec_freq_range=(50.0, 300.0), gaussian_sigma=1.5,
        enhancement_threshold=0.70,
    )
    assert centers[0, 0, 0] < 0.16     # 不落到两个 burst 之间的能量谷
