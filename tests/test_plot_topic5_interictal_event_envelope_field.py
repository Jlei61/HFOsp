"""端到端图层合同 —— 44 个纯函数测试全部通过，却漏掉了图上画错对象这一整类错误。

这些测试锁的是「图上画的东西 == 正文引用的统计对象」，用 fake event 驱动真实渲染代码，
不碰 /mnt 原始数据。
"""
from __future__ import annotations

import json
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
import scripts.paper_figures.build_main_figures_1_2 as F12
import scripts.paper_figures.plot_fig2c_interictal_event_envelope_field as PF
import scripts.paper_figures.screen_fig2c_ta_event_candidates as TAS
import scripts.paper_figures.screen_fig2c_tb_event_candidates as S
import src.topic5_interictal_event_field as ief


@pytest.fixture
def fz():
    """两根杆：ICL 沿轴铺开（应判为 axial -> 橙），SCL 挤在一小段（transverse -> 青）。"""
    ax = np.array([-16.0, -8.0, 0.0, 8.0, 16.0, -3.0, -1.0, 1.0])
    pts = np.column_stack([ax, np.zeros(8)])
    out = dict(names=[f"ICL{i}" for i in range(1, 6)] + [f"SCL{i}" for i in range(1, 4)],
               points_norm=pts / 20.0, points_mm=pts, analysis_sigma=0.2,
               analysis_sigma_mm=4.0, display_sigma_mm=6.0,
               display_xlim_mm=(-25.0, 25.0), display_ylim_mm=(-20.0, 20.0),
               transverse_sign=1, transverse_basis_w=[1.0, 0.0, 0.0], scale_mm=20.0,
               ax_mm=ax, shafts=np.array(["ICL"] * 5 + ["SCL"] * 3), axial_shaft="ICL",
               rank_a=np.arange(8.0), rank_b=np.arange(8.0)[::-1],
               support_a=np.linspace(0.2, 1.0, 8), support_b=np.linspace(1.0, 0.2, 8),
               relation={"line_angle_deg": 47.0, "cosine": -0.68},
               boot={"robust_collinear": False}, fingerprint="deadbeef" * 8,
               template_field_mode="shared")
    payloads = {}
    for lab, vals, sup in (
        ("TA", np.linspace(0.0, 1.0, 8), out["support_a"]),
        ("TB", np.linspace(1.0, 0.0, 8), out["support_b"]),
    ):
        payloads[lab] = dict(
            ds_sid="epilepsiae_1146", names=out["names"], xs=pts[:, 0], ys=pts[:, 1],
            sup=np.asarray(sup, float), soz=np.zeros(8, bool), src_a=set(), src_b=set(),
            frame=dict(xlim=out["display_xlim_mm"], ylim=out["display_ylim_mm"],
                       sigma_mm=6.0),
            vals=np.asarray(vals, float), transverse_sign=1,
            rank_values=(np.arange(8.0) if lab == "TA" else np.arange(8.0)[::-1]),
            transverse_alignment_rmse_mm=0.0,
        )
    out["template_payloads"] = payloads
    return out


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


def test_ta_screen_flips_direction_metrics_to_left_to_right(fz, ta):
    row = TAS._ta_metrics(ta, fz)
    assert row["centroid_vs_axis_rho"] > 0
    assert row["right_minus_left_centroid_ms"] > 0
    assert row["left_to_right_monotonic_fraction"] == pytest.approx(1.0)
    assert row["envelope_q99"] > 0
    assert TAS._gate_tier(row, n_middle_expected=1) == "strict"
    row["shaft_counts"]["SCL"]["n_usable"] = 1
    assert TAS._gate_tier(row, n_middle_expected=1) == "outside"


def test_ta_screen_amplitude_strength_changes_ranking_when_other_metrics_tie():
    def row(pos, amplitude):
        return dict(
            event_pos=pos,
            centroid_vs_axis_rho=0.9,
            envelope_q99=amplitude,
            contact_peak_median=amplitude,
            middle_peak_z_min=10.0,
            left_to_right_monotonic_fraction=0.9,
            right_minus_left_centroid_ms=20.0,
            rho_vs_template=0.9,
            axial_envelope_q99=amplitude,
            axial_contact_peak_median=amplitude,
            static_frame_axis_rho=0.9,
            static_frame_axis_span_mm=20.0,
            n_active_axial_static_frames=3,
            axial_completion_by_50_fraction=1.0,
        )

    rows = TAS._score_rows([row(1, 10.0), row(2, 40.0)])
    assert rows[1]["screen_score"] > rows[0]["screen_score"]


def test_tb_screen_prioritizes_strict_middle_complete_event_over_block_diversity():
    """中段完整性是用户指出的失败模式；不能为了 block 去重先塞入 relaxed 事件。"""
    rows = [
        dict(event_pos=1, block="A", gate_tier="strict", screen_score=0.9),
        dict(event_pos=2, block="A", gate_tier="strict", screen_score=0.8),
        dict(event_pos=3, block="B", gate_tier="relaxed", screen_score=1.0),
    ]
    got = S._select_distinct_blocks(rows, 2)
    assert [r["event_pos"] for r in got] == [1, 2]


def test_tb_screen_readme_marks_the_locked_fig2c_event(tmp_path):
    def row(event_pos, filename):
        return dict(
            event_pos=event_pos, figure_name=filename, axial_shaft="ICL",
            centroid_vs_axis_rho=-0.9, n_middle_usable=3, n_middle_expected=3,
            middle_peak_z_min=12.0, left_minus_right_centroid_ms=41.0,
            shaft_counts={
                "ICL": {"n_usable": 11, "n_participating": 11},
                "SCL": {"n_usable": 4, "n_participating": 4},
            },
        )

    current = row(2521, "current.png")
    selected = [row(829, "candidate_04.png")]
    path = S._write_readme(
        tmp_path, selected, current, 58.2, selected_for_fig2c_event_pos=829,
    )
    text = path.read_text(encoding="utf-8")
    assert "当前锁定 TB event=829" in text
    assert "当前 Fig2-C 选定 TB 单事件" in text
    assert "旧 medoid TB 参照事件" in text


def test_explicit_exemplar_is_label_checked_and_auditable(monkeypatch, fz):
    ev = {"labels": np.array([0, 1])}
    expected = {"stem": "fake_0009"}
    monkeypatch.setattr(P, "build_event", lambda *args: expected)
    event, meta = P.load_explicit_exemplar(
        fz, ev, {}, "1146", event_pos=1, label=1, tname="TB",
    )
    assert event is expected
    assert meta["selected_event_pos"] == 1
    assert "direction-qualified illustrative exemplar" in meta["caveat"]
    with pytest.raises(ValueError, match="not requested TA"):
        P.load_explicit_exemplar(fz, ev, {}, "1146", event_pos=1, label=0, tname="TA")


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
        title="Sample from TA", template="TA", show_xlabel=True, row_label="TA",
    )
    assert ax.get_title() == "Sample from TA"
    assert ax.get_xlabel() == "time (ms)"
    assert ax.get_ylabel() == "TA"
    assert ax.yaxis.label.get_color() == P.TEMPLATE_COLORS["TA"]
    assert ax.get_box_aspect() == pytest.approx(P.READOUT_BOX_ASPECT)
    assert ax.title.get_position()[0] == pytest.approx(0.96)
    assert ax.title.get_ha() == "right"
    plt.close(fig)


def test_readout_uses_common_window_intersection_to_remove_white_edge_bars(fz, ta, tb):
    ta = dict(ta, tile_lo_ms=-110.0, tile_hi_ms=150.0)
    tb = dict(tb, tile_lo_ms=-90.0, tile_hi_ms=130.0)
    assert P._common_readout_xlim(ta, tb) == pytest.approx((-90.0, 130.0))
    with pytest.raises(ValueError, match="at least two events"):
        P._common_readout_xlim(ta)


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


def test_frozen_template_panel_uses_viridis_rank_not_event_envelope(fz):
    fig, ax = plt.subplots()
    im = P._template_panel(ax, fz, "TA", show_y=True, show_x=True)
    assert im.get_cmap().name == "viridis"
    assert im.get_clim() == pytest.approx((0.0, 1.0))
    assert ax.get_title() == "TA template"
    assert ax.get_xlabel() == "shared TA axis (mm)"
    assert ax.get_ylabel() == "y (mm)"
    assert ax.collections[0].get_sizes()[0] == pytest.approx(P.TEMPLATE_CONTACT_SIZE)
    plt.close(fig)


def test_template_colorbar_uses_actual_ranks_and_top_title(fz):
    fig, cax = plt.subplots(figsize=(1.0, 3.0))
    cb, rank_range = P._template_rank_colorbar(fig, cax, fz, "TA")
    assert rank_range == pytest.approx((0.0, 7.0))
    assert cb.ax.get_title(loc="left") == "ranks"
    assert cb.ax.get_title() == ""
    assert cb.ax.yaxis.label.get_text() == ""
    assert cb.get_ticks() == pytest.approx([0.0, 3.5, 7.0])
    labels = [tick.get_text() for tick in cb.ax.get_yticklabels()]
    assert labels == ["0  early", "3.5", "7  late"]
    plt.close(fig)


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
    """承重回归：readout 画的是连续时间，只用离散帧算 vmax 会让帧间的峰意外饱和。"""
    n, fs = 1000, 1000.0
    z_full = np.ones((1, n))
    z_full[0, 470:490] = 60.0                                # 窄爆发，藏在两帧之间
    frame_idx = [0, 225, 450, 675, 900]                      # 5 帧全部错过它
    frames = z_full[:, frame_idx]
    v_frames, v_full = ief.pooled_vmax([frames]), ief.pooled_vmax([z_full])
    assert v_frames == pytest.approx(1.0)                    # 只看帧 -> 以为最大值就是基线
    assert v_full > 20 * v_frames                            # 完整窗才看得到真峰


def test_event_field_normalization_uses_one_complete_window_q99_per_event(ta, tb):
    louder_tb = dict(tb, env_z=np.asarray(tb["env_z"], float) * 2.0)
    scales = P._event_normalization_scales(ta, louder_tb, -8.0, 50.0)
    assert scales["TA"] > 0
    assert scales["TB"] == pytest.approx(2.0 * scales["TA"])
    assert P.FIELD_NORMALIZATION_ID == (
        "per_event_participant_q99_over_complete_display_window"
    )


def test_static_frame_normalization_uses_robust_participant_top3_reference():
    raw = np.array([1.0, 3.0, 5.0, 7.0, 100.0])
    participant = np.array([True, True, True, True, False])
    shown, scale = P._static_frame_relative_values(raw, participant)
    assert scale == pytest.approx((3.0 + 5.0 + 7.0) / 3.0)
    assert shown[:4] == pytest.approx([0.2, 0.6, 1.0, 1.0])
    assert shown[4] == pytest.approx(1.0)  # nonparticipant is clipped but excluded by support
    assert P.STATIC_FIELD_NORMALIZATION_ID == (
        "per_frame_participant_top3_mean_robust_z"
    )
    assert P.STATIC_FIELD_CBAR_LABEL == "Relative HFO envelope"


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
    assert P.READOUT_WIDTH_RATIO < 0.8
    assert P.READOUT_BOX_ASPECT > 1.0
    assert P.SPEC_CBAR_WIDTH_RATIO < P.FIELD_CBAR_WIDTH_RATIO
    assert 12.5 <= P.FIGURE_WIDTH_IN <= 13.2
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
    assert P.TEMPLATE_FIELD_COL > P.FIELD_CBAR_COL
    assert P.TEMPLATE_CBAR_COL == P.N_LAYOUT_COLS - 1
    assert P._subject_title("epilepsiae_1146") == "E1146"


def test_frame_times_are_never_duplicated():
    """承重回归：曾用「质心时刻的分位」取帧 —— 质心聚在 0-25ms 内，多个分位落到同一毫秒，
    图上出现两个 "+2 ms" 的重复帧。"""
    fts = P._static_frame_times(-15.0, 40.0, P.N_FRAMES)
    assert len(np.unique(np.round(fts, 1))) == len(fts) == P.N_FRAMES
    assert np.all(np.diff(fts) > 0)
    assert np.min(np.diff(fts)) >= 10.0


def test_static_frame_times_avoid_redundant_zero_and_plus_four_ms():
    got = P._static_frame_times(-8.0, 50.0)
    assert got == pytest.approx([-8.0, 11.333333, 30.666667, 50.0])
    assert 0.0 not in got
    assert P._time_label(11.333333) == "+11 ms"


def test_joint_visible_selector_requires_both_rows_and_distinct_contact_states(
    fz, ta, tb, monkeypatch,
):
    # The compact synthetic pulse fixture peaks at joint visibility 1/3; lower only this
    # data-dependent floor while retaining the production state-distance and direction gates.
    monkeypatch.setattr(P, "STATIC_FRAME_MIN_JOINT_VISIBILITY", 0.30)
    # Rectangular 20-ms synthetic pulses overlap at the first handoff; the production exemplar
    # regression below locks the non-zero endpoint gate on the raw E1146 event pair.
    monkeypatch.setattr(P, "STATIC_FRAME_MIN_ENDPOINT_HANDOFF", -1.0)
    monkeypatch.setattr(P, "STATIC_FRAME_MIN_FULL_VISIBILITY", 0.0)
    monkeypatch.setattr(P, "STATIC_FRAME_MIN_FULL_CENTROID_STEP_MM", -100.0)
    monkeypatch.setattr(P, "STATIC_FRAME_MIN_HOTSPOT_STEP_MM", -100.0)
    scales = P._event_normalization_scales(ta, tb, -8.0, 50.0)
    times, audit = P._select_joint_visible_frame_times(
        ta, tb, fz, -8.0, 50.0, scales,
    )
    assert len(times) == P.N_FRAMES
    assert np.min(np.diff(times)) >= P.STATIC_FRAME_MIN_GAP_MS
    assert np.ptp(np.diff(times)) < 1e-9
    assert audit["selection_mode"] == "equal_interval_full_field_hotspot_v6"
    assert audit["times_are_equally_spaced"] is True
    assert audit["equal_interval_ms"] == pytest.approx(np.diff(times)[0])
    assert min(audit["joint_visibility"]) >= P.STATIC_FRAME_MIN_JOINT_VISIBILITY
    assert min(audit["joint_contact_state_step_rms"]) >= P.STATIC_FRAME_MIN_STATE_DISTANCE
    assert min(audit["joint_endpoint_handoff_step"]) >= -1.0
    assert len(audit["joint_full_centroid_step_mm"]) == P.N_FRAMES - 1
    assert len(audit["joint_top3_hotspot_step_mm"]) == P.N_FRAMES - 1
    assert audit["ta_centroid_vs_time_rho"] >= P.STATIC_FRAME_MIN_CENTROID_RHO
    assert audit["tb_centroid_vs_time_rho"] <= -P.STATIC_FRAME_MIN_CENTROID_RHO


def test_endpoint_contrast_tracks_actual_left_to_right_handoff(fz, ta):
    # Four participating ICL contacts are ordered at -16, -8, 0, +8 mm.  The fixed outer
    # thirds therefore compare the first two contacts with the last contact.
    states = np.asarray([
        [1.0, 0.8, 0.1, 0.0],
        [0.3, 0.2, 0.7, 1.0],
    ])
    contrast = P._axial_endpoint_contrast(ta, fz, states)
    assert contrast == pytest.approx([-0.9, 0.75])
    assert np.diff(contrast)[0] > P.STATIC_FRAME_MIN_ENDPOINT_HANDOFF


def test_frame_window_is_tighter_than_the_old_plus_96ms_tail(fz, ta, tb):
    lo, hi = P._frame_window(ta, tb)
    assert (lo, hi) == pytest.approx((-8.0, 39.0))
    assert P.N_FRAMES == 4
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
    assert P.PAPER_SCHEMA_ID == "fig2c_interictal_event_envelope_field_candidate_v10"
    assert F12.FIG2C_ACCEPTED_SCHEMA == P.PAPER_SCHEMA_ID
    assert P.STATIC_FRAME_MIN_JOINT_VISIBILITY == pytest.approx(0.24)
    assert P.STATIC_FRAME_MIN_ENDPOINT_HANDOFF == pytest.approx(0.10)
    assert P.STATIC_FRAME_MIN_FULL_VISIBILITY == pytest.approx(0.30)
    assert P.STATIC_FRAME_MIN_FULL_CENTROID_STEP_MM == pytest.approx(2.0)
    assert P.STATIC_FRAME_MIN_HOTSPOT_STEP_MM == pytest.approx(4.0)
    assert "paper-ready-figure" not in str(PF.DEFAULT_OUT)
    assert P.GIF_FPS == 12
    assert PF.LOCKED_TA_EVENT_POS["epilepsiae_1146"] == 6344
    assert PF.LOCKED_TB_EVENT_POS["epilepsiae_1146"] == 937


def test_fig2_builder_rewrites_staging_paths_to_canonical_panel_names(tmp_path):
    figures = tmp_path / "figures"
    figures.mkdir()
    metadata_path = tmp_path / "fig2_panelc_metadata.json"
    metadata_path.write_text(json.dumps({
        "static": {"figure": "/old/staging.png", "extra_outputs": ["/old/staging.pdf"]},
        "gif": {"figure": "/old/staging.gif"},
    }))
    F12._canonicalize_fig2c_metadata(metadata_path, figures)
    metadata = json.loads(metadata_path.read_text())
    assert Path(metadata["static"]["figure"]).name == "fig2-panelc.png"
    assert Path(metadata["static"]["extra_outputs"][0]).name == "fig2-panelc.pdf"
    assert Path(metadata["gif"]["figure"]).name == "fig2-panelc.gif"


def test_event_field_uses_muted_bluegray_not_salient_magma():
    assert P.CMAP_NAME == "fig2c_muted_bluegray"
    assert P.CMAP.name == P.CMAP_NAME
    lo = np.asarray(P.CMAP(0.0)[:3])
    hi = np.asarray(P.CMAP(1.0)[:3])
    assert lo.mean() > 0.9
    assert hi.mean() < 0.45
    assert hi.max() - hi.min() < 0.22
    assert P.FIELD_DISPLAY_GAMMA == pytest.approx(0.5)
    assert P.FIELD_DISPLAY_NORM(0.25) == pytest.approx(0.5)
    assert P.FIELD_DISPLAY_NORM_ID == "fixed_power_norm_gamma_0p50"


def test_small_gif_uses_same_frozen_geometry_and_writes_real_animation(tmp_path, fz, ta, tb):
    from PIL import Image

    out = tmp_path / "ab.gif"
    meta = P.render_gif(
        "epilepsiae_1146", fz, ta, tb, P.event_stats(ta, fz), P.event_stats(tb, fz),
        out, step_ms=30.0, fps=2, dpi=45,
    )
    assert out.exists() and out.stat().st_size > 0
    assert meta["display_sigma_mm"] == pytest.approx(6.0)
    assert meta["vmax"] == pytest.approx(1.0)
    assert meta["normalization_mode"] == P.FIELD_NORMALIZATION_ID
    assert meta["display_norm"] == P.FIELD_DISPLAY_NORM_ID
    assert meta["display_gamma"] == pytest.approx(0.5)
    assert set(meta["normalization_scales_robust_z"]) == {"TA", "TB"}
    assert meta["frame_times_ms"] == pytest.approx([-8.0, 22.0, 39.0])
    with Image.open(out) as im:
        assert im.n_frames == 3
