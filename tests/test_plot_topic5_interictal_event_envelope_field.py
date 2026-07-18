"""端到端图层合同 —— 44 个纯函数测试全部通过，却漏掉了图上画错对象这一整类错误。

这些测试锁的是「图上画的东西 == 正文引用的统计对象」，用 fake event 驱动真实渲染代码，
不碰 /mnt 原始数据。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.plot_topic5_interictal_event_envelope_field as P
import scripts.paper_figures.screen_fig2c_tb_event_candidates as S
import src.topic5_interictal_event_field as ief


@pytest.fixture
def fz():
    """两根杆：ICL 沿轴铺开（应判为 axial -> 橙），SCL 挤在一小段（transverse -> 青）。"""
    ax = np.array([-16.0, -8.0, 0.0, 8.0, 16.0, -3.0, -1.0, 1.0])
    pts = np.column_stack([ax, np.zeros(8)])
    return dict(names=[f"ICL{i}" for i in range(1, 6)] + [f"SCL{i}" for i in range(1, 4)],
                points_norm=pts / 20.0, points_mm=pts, analysis_sigma=0.2,
                analysis_sigma_mm=4.0, display_sigma_mm=6.0,
                display_xlim_mm=(-25.0, 25.0), display_ylim_mm=(-20.0, 20.0),
                transverse_sign=1, transverse_basis_w=[1.0, 0.0, 0.0], scale_mm=20.0,
                ax_mm=ax, shafts=np.array(["ICL"] * 5 + ["SCL"] * 3), axial_shaft="ICL",
                rank_a=np.arange(8.0), rank_b=np.arange(8.0)[::-1],
                support_a=np.linspace(0.2, 1.0, 8), support_b=np.linspace(1.0, 0.2, 8),
                relation={"line_angle_deg": 47.0, "cosine": -0.68},
                boot={"robust_collinear": False}, fingerprint="deadbeef" * 8)


def _event(fz, *, peaks_ms, part, fs=1024.0):
    n = int(0.6 * fs)
    t = np.arange(n) / fs
    t0 = 0.2
    z = np.zeros((len(fz["names"]), n))
    for i, pm in enumerate(peaks_ms):
        if np.isfinite(pm):
            k = int((t0 + pm / 1000.0) * fs)
            z[i, max(k - 10, 0):k + 10] = 20.0
    spec = np.ones((len(fz["names"]), 4, 20), float)
    return dict(env_z=z, t=t, t0=t0, fs=fs, part=np.asarray(part, bool),
                usable=np.asarray(part, bool) & np.isfinite(peaks_ms),
                centroid_ms=np.asarray(peaks_ms, float),
                peak_ms=np.asarray(peaks_ms, float),
                centroid_freq_index=np.full(len(fz["names"]), 1.5),
                spec=spec, spec_t=np.linspace(0.01, 0.20, 20),
                spec_t_ms=np.linspace(-40.0, 150.0, 20), spec_freq_hz=np.arange(4.0),
                tile_lo_ms=-50.0, tile_hi_ms=200.0,
                stored=np.where(part, np.asarray(peaks_ms, float), np.nan),
                peak_z=np.full(len(fz["names"]), 20.0), w_lo=0.15, w_hi=0.4,
                packed_window_ms=250.0, event_interval=(0.18, 0.35), event_interval_ms=170.0,
                stem="fake_0001", t_in_block=1.0, pos=0, n_part=int(np.sum(part)),
                n_usable=int(np.sum(np.asarray(part, bool) & np.isfinite(peaks_ms))), n_snr=8)


@pytest.fixture
def ta(fz):
    # ICL 沿 +轴 依次点火；SCL 整根杆同时；ICL5 未参与
    return _event(fz, peaks_ms=np.array([5.0, 15.0, 25.0, 35.0, np.nan, 20.0, 21.0, 22.0]),
                  part=[1, 1, 1, 1, 0, 1, 1, 1])


@pytest.fixture
def tb(fz):
    # ICL 反向；SCL 同时
    return _event(fz, peaks_ms=np.array([35.0, 25.0, 15.0, 5.0, np.nan, 20.0, 21.0, 22.0]),
                  part=[1, 1, 1, 1, 0, 1, 1, 1])


# ------------------------------------------------------------------ 统计对象一致性
def test_stats_report_all_participants_and_each_shaft(fz, ta):
    st = P.event_stats(ta, fz)
    assert set(st) == {"all_participants", "shaft_ICL", "shaft_SCL"}
    assert st["shaft_ICL"]["fig1a_centroid_vs_axis_rho"] == pytest.approx(1.0)
    assert st["all_participants"]["n"] == 7                                     # ICL5 未参与


def test_stats_exclude_non_participating_contacts(fz, ta):
    """承重：正文引用的是"参与触点"，统计对象必须与之一致。"""
    st = P.event_stats(ta, fz)
    assert st["shaft_ICL"]["n"] == 4          # 5 个 ICL 里 ICL5 未参与 -> 只算 4 个


def test_stats_recover_opposite_signs_for_reversed_events(fz, ta, tb):
    a = P.event_stats(ta, fz)["shaft_ICL"]["fig1a_centroid_vs_axis_rho"]
    b = P.event_stats(tb, fz)["shaft_ICL"]["fig1a_centroid_vs_axis_rho"]
    assert a > 0 and b < 0


def test_direction_clarity_recovers_reverse_and_middle_contact_signal(fz, ta, tb):
    """候选筛选只读质心/包络数值，且必须显式审计沿轴中段触点。"""
    a = P.event_direction_clarity(ta, fz)
    b = P.event_direction_clarity(tb, fz)
    assert a["centroid_vs_axis_rho"] > 0
    assert b["centroid_vs_axis_rho"] < 0
    assert a["slope_ms_per_mm"] > 0
    assert b["slope_ms_per_mm"] < 0
    assert b["left_minus_right_centroid_ms"] > 0
    assert b["right_to_left_monotonic_fraction"] == pytest.approx(1.0)
    assert b["middle_contacts"] == ["ICL3"]
    assert b["n_middle_usable"] == 1
    assert b["middle_peak_z_min"] == pytest.approx(20.0)
    assert b["shaft_counts"]["ICL"]["n_usable"] == 4
    assert b["shaft_counts"]["SCL"]["n_usable"] == 3


def test_tb_screen_strict_gate_requires_both_shafts(fz, tb):
    row = P.event_direction_clarity(tb, fz)
    assert S._gate_tier(row, n_middle_expected=1) == "strict"
    row["shaft_counts"]["SCL"]["n_usable"] = 1
    assert S._gate_tier(row, n_middle_expected=1) == "outside"


def test_tb_screen_prioritizes_strict_middle_complete_event_over_block_diversity():
    """中段完整性是用户指出的失败模式；不能为了 block 去重先塞入 relaxed 事件。"""
    rows = [
        dict(event_pos=1, block="A", gate_tier="strict", screen_score=0.9),
        dict(event_pos=2, block="A", gate_tier="strict", screen_score=0.8),
        dict(event_pos=3, block="B", gate_tier="relaxed", screen_score=1.0),
    ]
    got = S._select_distinct_blocks(rows, 2)
    assert [r["event_pos"] for r in got] == [1, 2]


# ------------------------------------------------------------------ readout 图层合同（fig1a 形式）
def _readout_art(fz, e, st):
    fig, ax = plt.subplots()
    order = list(np.argsort(fz["ax_mm"]))
    P._readout(ax, e, fz, order, (-50.0, 200.0), st, "t", template="TA")
    tracks = [ln for ln in ax.get_lines() if ln.get_linestyle() == "-"
              and ln.get_color() == P.TEMPLATE_COLORS["TA"] and len(ln.get_xdata())]
    dots = [c for c in ax.collections if isinstance(c, PathCollection) and len(c.get_offsets())]
    title = ax.get_title()
    ylabels = [tick.get_text() for tick in ax.get_yticklabels()]
    plt.close(fig)
    return tracks, dots, title, order, ylabels


def test_readout_marks_exactly_the_participating_contacts(fz, ta):
    """承重回归：旧版把全部触点（含未参与的）连成一条线 —— 图上主证据 != 正文统计对象。"""
    tracks, dots, _, _, _ = _readout_art(fz, ta, P.event_stats(ta, fz))
    assert sum(len(d.get_offsets()) for d in dots) == int(ta["usable"].sum()) == 7


def test_readout_draws_one_centroid_track_not_per_shaft_lines(fz, ta):
    """用户明确要求：不看单杆的 peak。fig1a 的形式是**一条**质心连线，不按杆拆。"""
    tracks, _, _, _, _ = _readout_art(fz, ta, P.event_stats(ta, fz))
    assert len(tracks) == 1
    assert tracks[0].get_color() == P.TEMPLATE_COLORS["TA"]


def test_readout_track_is_the_centroid_not_the_envelope_argmax(fz, ta):
    """质心是管线自己的估计量（= lagPatRaw 的定义）。换成 argmax 就是另一把尺子。"""
    tracks, _, _, order, _ = _readout_art(fz, ta, P.event_stats(ta, fz))
    xs = list(tracks[0].get_xdata())
    want = [ta["centroid_ms"][c] for c in order if ta["usable"][c]]
    assert xs == pytest.approx(want)


def test_readout_title_is_only_the_template_name(fz, ta):
    """窄版 Fig1a readout 的统计量留在 JSON/README，标题只承担 TA/TB 识别。"""
    _, _, title, _, _ = _readout_art(fz, ta, P.event_stats(ta, fz))
    assert title == "t"


def test_readout_uses_the_locked_fig1a_palette():
    assert P.FIG1A_CMAP == "coolwarm"
    assert P.CENTROID_FACE == "#ffb000"
    assert P.TEMPLATE_COLORS == {"TA": "#B2182B", "TB": "#2166AC"}


def test_readout_ylabels_show_contact_names_without_axis_numbers(fz, ta):
    _, _, _, order, ylabels = _readout_art(fz, ta, P.event_stats(ta, fz))
    want = [fz["names"][c] for c in order if ta["usable"][c]]
    assert ylabels == want


def test_TA_and_TB_tracks_use_red_blue_semantics(fz, ta, tb):
    fig, axes = plt.subplots(1, 2)
    order = list(np.argsort(fz["ax_mm"]))
    out_a = P._readout(
        axes[0], ta, fz, order, (-50.0, 200.0), P.event_stats(ta, fz), "TA",
        template="TA",
    )
    out_b = P._readout(
        axes[1], tb, fz, order, (-50.0, 200.0), P.event_stats(tb, fz), "TB",
        template="TB",
    )
    assert any(line.get_color() == "#B2182B" for line in axes[0].get_lines())
    assert any(line.get_color() == "#2166AC" for line in axes[1].get_lines())
    assert out_a[1] and out_b[1]
    plt.close(fig)


def test_left_readout_row_label_and_top_xlabel_contract(fz, ta):
    fig, ax = plt.subplots()
    order = list(np.argsort(fz["ax_mm"]))
    P._readout(
        ax, ta, fz, order, (-50.0, 200.0), P.event_stats(ta, fz),
        template="TA", show_xlabel=False, row_label="TA",
    )
    assert ax.get_title() == ""
    assert ax.get_xlabel() == ""
    assert ax.get_ylabel() == "TA"
    assert ax.yaxis.label.get_color() == P.TEMPLATE_COLORS["TA"]
    assert ax.get_box_aspect() == pytest.approx(1.0)
    plt.close(fig)


# ------------------------------------------------------------------ shared field 图层合同
def test_shared_geometry_uses_one_deterministic_transverse_sign_for_both_rows():
    shared = {
        "points": [[-0.5, -0.2], [0.5, 0.3]],
        "scale_mm": 20.0,
        "w": [0.1, -0.9, 0.2],
    }
    pts, sign, _, _ = P._shared_display_geometry(shared)
    assert sign == -1
    assert pts[:, 0] == pytest.approx([-10.0, 10.0])
    assert pts[:, 1] == pytest.approx([4.0, -6.0])
    # render() consumes this one shared points_mm array for TA and TB; no row-specific mirror exists.
    src = Path(P.__file__).read_text()
    assert "points_mm_a" not in src and "points_mm_b" not in src


def test_event_field_is_exactly_the_established_physical_mm_smoother(fz, ta):
    from scripts.plot_contact_plane_static import _smooth_rank_field_mm

    vals = np.arange(len(fz["names"]), dtype=float)
    support = ta["part"].astype(float)
    got = P._event_field(fz, vals, support)
    pts = fz["points_mm"]
    ref = _smooth_rank_field_mm(
        pts[:, 0], pts[:, 1], vals, support, fz["display_xlim_mm"],
        fz["display_ylim_mm"], fz["display_sigma_mm"],
    )
    for a, b in zip(got, ref):
        assert np.allclose(a, b, equal_nan=True)


def test_event_field_uses_fixed_six_mm_display_bandwidth(fz):
    assert fz["display_sigma_mm"] == P.DEFAULT_DISPLAY_SIGMA_MM == 6.0


# ------------------------------------------------------------------ support 模式
def test_main_figure_uses_participant_only_support(fz, ta):
    """承重：未参与触点比参与触点还热，让它们进平滑 = 场里最亮处不属于这个事件。"""
    sup = P._support("participant", fz, ta, "a")
    assert np.array_equal(sup, ta["part"].astype(float))
    assert sup[4] == 0.0                                     # ICL5 未参与 -> 权重 0


def test_qc_modes_are_distinct_from_the_main_mode(fz, ta):
    assert np.all(P._support("unit", fz, ta, "a") == 1.0)
    assert np.array_equal(P._support("template", fz, ta, "a"), fz["support_a"])
    with pytest.raises(ValueError):
        P._support("nonsense", fz, ta, "a")


# ------------------------------------------------------------------ 窗宽不许硬编码
def test_packed_window_is_read_from_the_artifact_not_hardcoded(fz, ta):
    """承重回归：1146 的打包窗是 250ms，139 是 200ms。曾经把 200 硬编码进标题/JSON。"""
    assert ta["packed_window_ms"] == pytest.approx(1000.0 * (ta["w_hi"] - ta["w_lo"]))
    src = Path(P.__file__).read_text()
    assert "packed_window_ms=200" not in src and "packed_window_ms\": 200" not in src


def test_no_module_level_200ms_constant():
    src = Path(P.__file__).read_text()
    code = "\n".join(l for l in src.splitlines()
                     if not l.strip().startswith("#") and "200ms" not in l and "200 ms" not in l)
    assert "200.0" not in code.split("DET = ")[0]            # 只有 detector max_last 允许是 200


# ------------------------------------------------------------------ vmax 覆盖整个显示窗
def test_vmax_is_computed_over_the_full_window_not_just_the_frames(fz):
    """承重回归：readout 画的是连续时间，只用 7 帧算 vmax 会让帧间的峰意外饱和。"""
    n, fs = 1000, 1000.0
    z_full = np.ones((1, n))
    z_full[0, 470:490] = 60.0                                # 窄爆发，藏在两帧之间
    frame_idx = [0, 150, 300, 450, 600, 750, 900]            # 7 帧全部错过它
    frames = z_full[:, frame_idx]
    v_frames, v_full = ief.pooled_vmax([frames]), ief.pooled_vmax([z_full])
    assert v_frames == pytest.approx(1.0)                    # 只看帧 -> 以为最大值就是基线
    assert v_full > 20 * v_frames                            # 完整窗才看得到真峰


def test_readout_title_never_prints_nan_rho(fz):
    """n<3 时标题仍只显示模板名，不泄漏 NaN 统计量。"""
    e = _event(fz, peaks_ms=np.array([5.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
               part=[1, 0, 0, 0, 0, 0, 0, 0])
    _, _, title, _, _ = _readout_art(fz, e, P.event_stats(e, fz))
    assert "nan" not in title


def test_compact_layout_contract():
    assert P.FIELD_CBAR_WIDTH_RATIO < 0.05
    assert P.FIELD_CBAR_WIDTH_RATIO > 0.03
    assert P.GROUP_GAP_WIDTH_RATIO > P.FIELD_CBAR_WIDTH_RATIO
    assert P.READOUT_WIDTH_RATIO == pytest.approx(1.0)
    assert P.SPEC_CBAR_WIDTH_RATIO < P.FIELD_CBAR_WIDTH_RATIO
    assert P.FIGURE_WIDTH_IN >= 12.5
    assert 4.5 < P.FIGURE_HEIGHT_IN < 5.0
    assert P.FIELD_TICK_LABELSIZE >= 8
    assert P.CONTACT_TICK_LABELSIZE >= 8
    assert P.READOUT_TICK_LABELSIZE >= 9
    assert P.CBAR_TICK_LABELSIZE >= 8
    assert P.AXIS_LABELSIZE >= 12
    assert P.FRAME_TITLE_SIZE >= 12
    assert P.TEMPLATE_LABEL_SIZE >= 12
    assert P.MAIN_TITLE_SIZE >= 15
    assert (P.READOUT_COL, P.READOUT_CBAR_COL, P.GROUP_GAP_COL, P.FRAME_COL_START) == (0, 1, 2, 3)
    assert P._subject_title("epilepsiae_1146") == "E1146"


def test_frame_times_are_never_duplicated():
    """承重回归：曾用「质心时刻的分位」取帧 —— 质心聚在 0-25ms 内，多个分位落到同一毫秒，
    图上出现两个 "+2 ms" 的重复帧。"""
    fts = ief.frame_times_ms(-15.0, 40.0, P.N_FRAMES)
    assert len(np.unique(np.round(fts, 1))) == len(fts) == P.N_FRAMES
    assert np.all(np.diff(fts) > 0)


def test_frame_window_is_tighter_than_the_old_plus_96ms_tail(fz, ta, tb):
    lo, hi = P._frame_window(ta, tb)
    assert (lo, hi) == pytest.approx((-8.0, 39.0))
    assert P.N_FRAMES == 6
    assert P.FRAME_MAX_POST_MS == 50.0


def test_gif_time_grid_keeps_true_endpoint_and_two_ms_contract():
    got = P._gif_frame_times(-8.0, 50.0)
    assert got.tolist() == pytest.approx(np.arange(-8.0, 50.1, 2.0))
    assert got[-1] == pytest.approx(50.0)
    with pytest.raises(ValueError):
        P._gif_frame_times(-8.0, 50.0, step_ms=0.0)


def test_paper_candidate_naming_contract():
    assert P._paper_stem("epilepsiae_1146") == (
        "fig2c_candidate_E1146_interictal_event_envelope_field"
    )
    assert P.PAPER_SCHEMA_ID == "fig2c_interictal_event_envelope_field_candidate_v1"
    assert P.GIF_FPS == 12


def test_small_gif_uses_same_frozen_geometry_and_writes_real_animation(tmp_path, fz, ta, tb):
    from PIL import Image

    out = tmp_path / "ab.gif"
    meta = P.render_gif(
        "epilepsiae_1146", fz, ta, tb, P.event_stats(ta, fz), P.event_stats(tb, fz),
        out, step_ms=30.0, fps=2, dpi=45,
    )
    assert out.exists() and out.stat().st_size > 0
    assert meta["display_sigma_mm"] == pytest.approx(6.0)
    assert meta["vmax"] > 0
    assert meta["frame_times_ms"] == pytest.approx([-8.0, 22.0, 39.0])
    with Image.open(out) as im:
        assert im.n_frames == 3
