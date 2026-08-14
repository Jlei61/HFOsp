#!/usr/bin/env python3
"""Topic 5 间期单事件包络场 —— 冻结 TA/TB 共享平面上的两个示例事件。

问：这个被试的两套间期传播模板（TA/TB），对应的真实事件里，参与触点的 HFO 包络峰
是不是沿冻结共享轴呈相反的先后顺序？

主图：行 = 模板，左 = Fig1a 原 producer 的 normalized spectrogram + 主高频增强连通区
质心轨迹，右 = participant-only 包络场随时间。readout 只画参与触点，并按同一 shared-axis
顺序排列；红线是“触点顺序上的质心轨迹”，不是跨杆的物理连线。

承重合同（每条都是被真实数据打脸后定的，改回去会重现 bug）：
 1. 打包窗宽 **per-subject**，从 w_hi-w_lo 实时读（1146=250ms，139=200ms）。窗宽是尺子长度，
    不是测出来的传播时长。**不许硬编码。**
 2. 右侧包络用单带 `return_hil_enve`；左侧直接复用 Fig1a 的 magnitude + Gaussian sigma=1.5、
    per-channel/per-event max normalization、主峰 >=70% 连通区质心和真实 STFT cell edges。
 3. 一次事件 = 探测器自己的规则（`ief.detector_excursions`，rel/abs=2.0、min_gap=20ms、
    min_last=50ms）切出来、且跟**群体并发段**重叠的那一次。整窗 argmax 会把后面另一次事件
    当成"这次的峰"。
 4. 主图 = participant-only。未参与触点比参与触点还热，让它们进平滑 = 场里最亮的地方不属于
    这个事件。all-contact 与 template-weighted 只作 QC。
 5. 所有统计在代码里算、写进 JSON、由程序填进图上；**不手填**（手填已导致 README 的 TB
    存档数停留在换 exemplar 前的旧值）。

冻结合同：axis / shared plane / contact_order 来自 template_gradient_fields，先过 fingerprint
校验，**不重拟合**。显示沿用 TA/TB field producer：物理 mm 坐标、shared A/B 同一个 transverse
正号、同一 x/y extent、固定 6 mm display kernel；分析用的冻结 sigma 不被修改。

⚠️ 这**不是**独立数据验证：共享轴、TA/TB 标签、exemplar 筛选、参与触点、事件窗全部来自模板
管线；新的只有包络定时。正确措辞 = raw-EEG-derived envelope timing cross-check on a
previously frozen interictal axis。

用法: python scripts/plot_topic5_interictal_event_envelope_field.py --subject epilepsiae_1146
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.topic5_interictal_event_field as ief
from scripts.paper_figures.fig1_spectrogram_utils import (
    ALGORITHM_ID as FIG1A_ALGORITHM_ID,
    centroid_alignment_audit,
    compute_group_event_spectrogram_stack,
    full_extent_edges,
)
from scripts.plot_contact_plane_static import _limits_with_padding, _smooth_rank_field_mm
from scripts.plot_topic5_interictal_template_ab_fields import (
    DEFAULT_DISPLAY_SIGMA_MM,
    _canonical_transverse_sign,
    build_interictal_ab_panel_payloads,
    draw_interictal_rank_field_panel,
)
from src.interictal_propagation import load_subject_propagation_events
from src.lagpat_rank_audit import mask_phantom_ranks
from src.preprocessing import load_epilepsiae_block
from src.propagation_skeleton_geometry import assign_events_to_templates
from src.topic5_template_axis_field import scorers_from_interictal_record
from src.utils.bqk_utils import band_filt, notch_filt, return_hil_enve

FROZEN = _ROOT / "results/interictal_propagation_masked/template_gradient_fields/per_subject"
LAGPAT_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
INVENTORY = _ROOT / "results/epilepsiae_block_inventory.csv"
OUT = _ROOT / "results/interictal_propagation_masked/event_envelope_fields"

BAND = (80.0, 250.0)
PAD_SEC, GUARD_SEC = 1.0, 0.05
DET = dict(rel_thresh=2.0, abs_thresh=2.0, min_gap_ms=20.0,      # legacy yuquan 口径
           min_last_ms=50.0, max_last_ms=200.0)                  # config/default.yaml
ONSET_FRAC, ONSET_MIN_Z, ONSET_SUSTAIN_MS = 0.25, 5.0, 5.0
TOP_K, SNR_MIN_Z, SNR_MIN_CH = 40, 5.0, 5
AXIAL_MIN_CH = 6       # 沿轴杆上至少 6 个触点的峰可测 —— 4 个点量不出梯度。
                       # 这是**可测性**门（看有几个点，不看斜率），不是挑漂亮的。
FRAME_AVG_MS, N_FRAMES = 3.0, 4
CMAP_NAME = "fig2c_muted_bluegray"
CMAP = LinearSegmentedColormap.from_list(
    CMAP_NAME,
    ["#f7f8fa", "#dfe7eb", "#b5c8d0", "#7f9eaa", "#456b78"],
)
FIELD_DISPLAY_GAMMA = 0.50
FIELD_DISPLAY_NORM_ID = "fixed_power_norm_gamma_0p50"
FIELD_DISPLAY_NORM = PowerNorm(gamma=FIELD_DISPLAY_GAMMA, vmin=0.0, vmax=1.0)
STATIC_FIELD_NORMALIZATION_ID = "per_frame_participant_top3_mean_robust_z"
STATIC_FIELD_CBAR_LABEL = "Relative HFO envelope"
FRAME_PRE_MS, FRAME_MARGIN_MS, FRAME_MIN_POST_MS, FRAME_MAX_POST_MS = 8.0, 4.0, 35.0, 50.0
GIF_STEP_MS, GIF_FPS = 2.0, 12
NOTCH_HZ = (50.0, 100.0, 150.0, 200.0, 250.0)
FIG1A_CMAP = "coolwarm"
CENTROID_FACE = "#ffb000"
TEMPLATE_COLORS = {"TA": "#B2182B", "TB": "#2166AC"}
FIELD_CBAR_WIDTH_RATIO = 0.045
TEMPLATE_CBAR_WIDTH_RATIO = 0.045
GROUP_GAP_WIDTH_RATIO = 0.08
TEMPLATE_GAP_WIDTH_RATIO = 0.14
READOUT_WIDTH_RATIO = 0.72
READOUT_BOX_ASPECT = 1.18
SPEC_CBAR_WIDTH_RATIO = 0.032
TEMPLATE_FIELD_WIDTH_RATIO = 1.0
TEMPLATE_CONTACT_SIZE = 38
TEMPLATE_CONTACT_OUTLINE_LW = 1.2
FIGURE_WIDTH_IN = 12.8
FIGURE_HEIGHT_IN = 4.9
FIELD_TICK_LABELSIZE = 8
CONTACT_TICK_LABELSIZE = 8
READOUT_TICK_LABELSIZE = 9
CBAR_TICK_LABELSIZE = 8
CBAR_LABELSIZE = 9
TICK_LENGTH = 1.5
AXIS_LABELSIZE = 12
FRAME_TITLE_SIZE = 12
TEMPLATE_LABEL_SIZE = 12
MAIN_TITLE_SIZE = 15
READOUT_COL = 0
READOUT_CBAR_COL = 1
GROUP_GAP_COL = 2
FRAME_COL_START = 3
FIELD_CBAR_COL = FRAME_COL_START + N_FRAMES
TEMPLATE_GAP_COL = FIELD_CBAR_COL + 1
TEMPLATE_FIELD_COL = TEMPLATE_GAP_COL + 1
TEMPLATE_CBAR_COL = TEMPLATE_FIELD_COL + 1
N_LAYOUT_COLS = TEMPLATE_CBAR_COL + 1
PAPER_SCHEMA_ID = "fig2c_interictal_event_envelope_field_candidate_v10"
FIELD_NORMALIZATION_ID = "per_event_participant_q99_over_complete_display_window"
STATIC_FRAME_GRID_STEP_MS = 2.0
STATIC_FRAME_MIN_GAP_MS = 8.0
STATIC_FRAME_MIN_JOINT_VISIBILITY = 0.24
STATIC_FRAME_MIN_AXIS_SPAN_MM = 12.0
STATIC_FRAME_MIN_STATE_DISTANCE = 0.09
STATIC_FRAME_MIN_CENTROID_RHO = 0.70
STATIC_FRAME_MIN_ENDPOINT_HANDOFF = 0.10
STATIC_FRAME_MIN_FULL_VISIBILITY = 0.30
STATIC_FRAME_MIN_FULL_CENTROID_STEP_MM = 2.0
STATIC_FRAME_MIN_HOTSPOT_STEP_MM = 4.0


def _inventory(subject):
    with open(INVENTORY) as fh:
        return {r["block_stem"]: r for r in csv.DictReader(fh) if r["subject"] == subject}


def _rho(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan"), int(m.sum())
    r = spearmanr(np.asarray(a)[m], np.asarray(b)[m]).correlation
    return (float(r) if np.isfinite(r) else float("nan")), int(m.sum())


def _shared_display_geometry(shared_plane):
    """Map one frozen shared plane to the established physical-mm display frame."""
    pts = np.asarray(shared_plane["points"], float) * float(shared_plane["scale_mm"])
    sign = _canonical_transverse_sign(shared_plane["w"])
    pts[:, 1] *= sign
    xlim = _limits_with_padding(pts[:, 0], include_zero=True, min_span=35.0)
    ylim = _limits_with_padding(pts[:, 1], include_zero=True, min_span=35.0)
    return pts, int(sign), xlim, ylim


def load_frozen(ds_sid):
    rec = json.load(open(FROZEN / f"{ds_sid}.json"))
    scorers_from_interictal_record(rec)                          # <- fingerprint gate (raises)
    template_a, template_b, template_mode = build_interictal_ab_panel_payloads(
        rec, display_sigma_mm=DEFAULT_DISPLAY_SIGMA_MM,
    )
    f, ap = rec["interictal_field"], rec["axis_pair"]
    sh = (f.get("planes") or {}).get("shared")
    if f["status"] != "ok" or not sh or sh.get("status") != "ok":
        raise ValueError(f"{ds_sid}: no usable frozen shared plane")
    if ap["relation"]["relation"] != "reversed":
        raise ValueError(f"{ds_sid}: relation={ap['relation']['relation']}, not reversed")
    pts_norm = np.asarray(sh["points"], float)
    scale = float(sh["scale_mm"])
    pts_mm, transverse_sign, xlim, ylim = _shared_display_geometry(sh)
    shafts = np.array([str(x) for x in f["shafts"]])
    ax_mm = pts_mm[:, 0]
    span = {s: float(np.ptp(ax_mm[shafts == s])) for s in set(shafts) if (shafts == s).sum() > 1}
    axial = max(span, key=span.get) if span else None            # 沿轴跨度最大的杆 -> 橙
    return dict(names=[str(x) for x in f["contact_order"]], points_norm=pts_norm,
                points_mm=pts_mm, analysis_sigma=float(sh["sigma"]),
                analysis_sigma_mm=float(sh["sigma"]) * scale,
                display_sigma_mm=float(DEFAULT_DISPLAY_SIGMA_MM),
                display_xlim_mm=xlim, display_ylim_mm=ylim,
                transverse_sign=int(transverse_sign), transverse_basis_w=sh["w"],
                scale_mm=scale, ax_mm=ax_mm, shafts=shafts, axial_shaft=axial,
                rank_a=np.asarray(f["rank_a"], float), rank_b=np.asarray(f["rank_b"], float),
                support_a=np.asarray(f["support_a"], float),
                support_b=np.asarray(f["support_b"], float),
                relation=ap["relation"], boot=ap["pair_bootstrap"],
                fingerprint=f["fingerprint_sha256"],
                template_payloads={"TA": template_a, "TB": template_b},
                template_field_mode=template_mode)


def load_events(fz, subject):
    """narrow pool（= 冻结场的 source substrate）。⚠️ 绝不用 _broad 的 KMeans 标签 —— 那是
    另一次聚类、另一套通道（CLAUDE.md §6.2 层级错配）。"""
    pool = load_subject_propagation_events(LAGPAT_ROOT / subject / "all_recs")
    ev_names = [str(x) for x in pool["channel_names"]]
    bools = np.asarray(pool["bools"], bool)
    masked = mask_phantom_ranks(np.asarray(pool["ranks"], float), bools, normalize=True)
    idx = {n: i for i, n in enumerate(ev_names)}
    if [n for n in fz["names"] if n not in idx]:
        raise ValueError(f"{subject}: frozen contacts absent from the event pool")
    sel = [idx[n] for n in fz["names"]]                          # exact channel-name join
    return dict(masked=masked[sel], bools=bools[sel], lag_raw=pool["lag_raw"][sel],
                labels=assign_events_to_templates(masked[sel], fz["rank_a"], fz["rank_b"]),
                block_ids=np.asarray(pool["block_ids"]),
                abs_times=np.asarray(pool["event_abs_times"], float),
                record_names=np.asarray(pool["record_names"]),
                block_start=np.asarray(pool["block_start_times"], float))


def build_event(ev, pos, inv, subject, fz):
    bid = int(ev["block_ids"][pos])
    stem = str(ev["record_names"][bid])
    t_in_block = float(ev["abs_times"][pos]) - float(ev["block_start"][bid])
    row = inv[stem]
    pt = np.load(LAGPAT_ROOT / subject / "all_recs" / f"{stem}_packedTimes.npy")
    j = int(np.argmin(np.abs(pt[:, 0] - t_in_block)))
    if abs(float(pt[j, 0]) - t_in_block) > 1e-3:
        raise ValueError(f"{stem}: event t={t_in_block:.4f}s matches no packed window")
    w_lo, w_hi = float(pt[j, 0]), float(pt[j, 1])                # 合同 1：窗宽实时读，不硬编码
    crop = max(w_lo - PAD_SEC, 0.0)
    res = load_epilepsiae_block(row["data_path"], row["head_path"], reference="car",
                                segment_sec=None, crop_start_sec=crop,
                                crop_duration_sec=(w_hi - w_lo) + 2 * PAD_SEC)
    if [n for n in fz["names"] if n not in res.ch_names]:
        raise ValueError(f"{stem}: contacts missing from raw")
    t = np.arange(res.data.shape[1]) / res.sfreq + crop
    X = res.data[[res.ch_names.index(n) for n in fz["names"]]]
    clean = notch_filt(X, res.sfreq, [f for f in NOTCH_HZ if f < res.sfreq / 2.0])
    band = band_filt(clean, res.sfreq, list(BAND))
    env = return_hil_enve(clean, res.sfreq, list(BAND))
    part = np.asarray(ev["bools"][:, pos], bool)
    exc = ief.detector_excursions(env, t, **DET)                 # 合同 3：探测器自己的事件定义
    iv = ief.group_event_interval([exc[i] for i in np.where(part)[0]], t)
    if iv is None:
        raise ValueError(f"{stem}: no concurrent supra-threshold interval among participants")

    # Fig1a 合同：一个 packed event = 一个 spectrogram tile；显示量、归一化、质心支持和
    # marker 坐标全部来自同一个公共 helper。这里不再另算“类似 Fig1a”的全局质心。
    tile_mask = (t >= w_lo) & (t < w_hi)
    band_tile = np.asarray(band[:, tile_mask], float)
    tile_sec = band_tile.shape[1] / float(res.sfreq)
    if band_tile.shape[1] < int(round(0.12 * res.sfreq)):
        raise ValueError(f"{stem}: packed event tile too short for Fig1a spectrogram")
    spec_stack, spec_t, spec_f, centers = compute_group_event_spectrogram_stack(
        band_tile,
        res.sfreq,
        np.asarray([tile_sec]),
        spec_window="hamming",
        spec_win_sec=0.05,
        spec_overlap_sec=0.04,
        spec_freq_range=(50.0, 300.0),
        gaussian_sigma=1.5,
        enhancement_threshold=0.70,
    )
    alignment = centroid_alignment_audit(
        spec_stack, spec_t, spec_f, centers, window_sec=tile_sec, acceptance_threshold=0.70,
    )
    n_freq = len(spec_f)
    spec = spec_stack.reshape(len(fz["names"]), n_freq, spec_stack.shape[1])
    cent_rel = np.asarray(centers[:, 0, 0], float)
    cent_freq_idx = np.asarray(centers[:, 0, 1], float)
    cent_abs = w_lo + cent_rel
    if not np.isfinite(cent_abs[part]).any():
        raise ValueError(f"{stem}: no finite Fig1a centroid among participants")
    t0 = float(np.nanmin(cent_abs[part]))                         # t=0 = 最早 Fig1a 质心
    pk_ms = (cent_abs - t0) * ief.MS
    stored = ief.stored_lag_rel_ms(ev["lag_raw"][:, pos], part)
    pkz = ief.envelope_peak_z(ief.baseline_robust_z(env, t, w_lo, w_hi, guard_sec=GUARD_SEC),
                              t, iv[0], iv[1])
    # Fig1a 主峰连通区质心与 lagPat 全局频谱重心不是同一估计量；这里只作透明的 agreement audit。
    _pm = part & np.isfinite(pk_ms)
    if _pm.sum() >= 3:
        par_rho = _rho(pk_ms[_pm], stored[_pm])[0]
        par_mad = float(np.nanmedian(np.abs(pk_ms[_pm] - stored[_pm])))
    else:
        par_rho, par_mad = float("nan"), float("nan")
    env_z = ief.baseline_robust_z(env, t, w_lo, w_hi, guard_sec=GUARD_SEC)
    return dict(pos=int(pos), stem=stem, t_in_block=t_in_block, env_z=env_z, t=t, t0=t0,
                band_tile=band_tile, spec=spec, spec_t=spec_t,
                spec_t_ms=(w_lo + spec_t - t0) * ief.MS, spec_freq_hz=spec_f,
                centroid_ms=pk_ms, centroid_freq_index=cent_freq_idx,
                tile_lo_ms=(w_lo - t0) * ief.MS, tile_hi_ms=(w_lo + tile_sec - t0) * ief.MS,
                fig1a_centroid_alignment=alignment,
                parity_vs_stored_rho=par_rho, parity_vs_stored_median_abs_dt_ms=par_mad,
                fs=float(res.sfreq), w_lo=w_lo, w_hi=w_hi,
                packed_window_ms=(w_hi - w_lo) * ief.MS, event_interval=iv,
                event_interval_ms=(iv[1] - iv[0]) * ief.MS, part=part,
                usable=part & np.isfinite(pk_ms), stored=stored, peak_ms=pk_ms, peak_z=pkz,
                spread_ms=float(np.nanmax(stored)), n_part=int(part.sum()),
                n_usable=int((part & np.isfinite(pk_ms)).sum()),
                n_snr=int(np.sum(pkz[part] >= SNR_MIN_Z)))


def event_stats(e, fz):
    """所有承重统计在这里算一次 -> JSON -> 由程序填到图上。**任何数字都不许手填。**"""
    out = {}
    groups = [("all_participants", e["usable"])] + \
             [(f"shaft_{s}", e["usable"] & (fz["shafts"] == s)) for s in sorted(set(fz["shafts"]))]
    for tag, m in groups:
        r_cent, n_cent = _rho(e["centroid_ms"][m], fz["ax_mm"][m])
        r_st, n_st = _rho(e["stored"][m], fz["ax_mm"][m])
        pm = e["centroid_ms"][m]
        pm = pm[np.isfinite(pm)]
        out[tag] = dict(n=int(m.sum()), fig1a_centroid_vs_axis_rho=r_cent,
                        n_fig1a_centroid=n_cent,
                        stored_lag_vs_axis_rho=r_st, n_stored=n_st,
                        centroid_range_ms=[float(pm.min()), float(pm.max())] if pm.size else None)
    return out


def build_exemplar_pool(
    fz, ev, inv, subject, label, tname, *, top_k=TOP_K,
    candidate_filter=None, max_raw_candidates=None,
):
    """Build the raw-EEG candidate pool once so medoid and transparent screens share inputs."""
    spreads = ief.event_spreads_ms(ev["lag_raw"], ev["bools"])
    own = np.asarray(ev["labels"]) == label
    tgt_sp = float(np.nanmedian(spreads[own]))
    tgt_np = float(np.median(ev["bools"][:, own].sum(axis=0)))
    ranked_cand, rhos, info = ief.rank_candidates(
        ev["masked"], ev["bools"], ev["labels"], label,
        fz["rank_a"] if tname == "TA" else fz["rank_b"],
        spreads, tgt_sp, tgt_np, top_k=top_k,
    )
    cand = [int(i) for i in ranked_cand
            if candidate_filter is None or bool(candidate_filter(int(i)))]
    n_after_filter = len(cand)
    if max_raw_candidates is not None:
        cand = cand[:int(max_raw_candidates)]
    built, feats = {}, {}
    for i in cand:
        try:
            e = build_event(ev, i, inv, subject, fz)
        except (ValueError, KeyError, FileNotFoundError):
            continue
        n_ax = int((e["usable"] & (fz["shafts"] == fz["axial_shaft"])).sum())
        if e["n_snr"] < SNR_MIN_CH or e["n_usable"] < SNR_MIN_CH or n_ax < AXIAL_MIN_CH:
            continue
        built[i] = e
        feats[i] = [e["spread_ms"], float(e["n_part"]), float(np.nanmedian(e["peak_z"][e["part"]]))]
    if not built:
        raise ValueError(f"{tname}: no candidate survives the SNR / usable-peak gate")
    return built, feats, dict(
        n_events_in_template=int(own.sum()), target_spread_ms=tgt_sp,
        target_n_participating=tgt_np, candidate_positions=[int(x) for x in cand],
        n_ranked_candidates=int(len(ranked_cand)),
        n_after_candidate_filter=int(n_after_filter),
        rhos={int(k): float(v) for k, v in rhos.items()}, info=info,
    )


def event_direction_clarity(e, fz):
    """Readout-only direction/continuity metrics; never inspect rendered field pixels."""
    shaft = fz["axial_shaft"]
    axial_all = np.asarray(fz["shafts"] == shaft, bool)
    axial = axial_all & np.asarray(e["usable"], bool)
    x = np.asarray(fz["ax_mm"], float)
    t = np.asarray(e["centroid_ms"], float)
    peak_z = np.asarray(e["peak_z"], float)
    if axial.sum() < 3:
        raise ValueError("direction clarity requires at least three usable axial-shaft contacts")

    xa = x[axial]
    ta = t[axial]
    q1, q2 = np.quantile(x[axial_all], [1.0 / 3.0, 2.0 / 3.0])
    middle = axial & (x >= q1) & (x <= q2)
    left = axial & (x <= q1)
    right = axial & (x >= q2)
    order = np.argsort(xa)[::-1]  # physical right -> left on the shared TA axis
    dt = np.diff(ta[order])
    rho, n = _rho(ta, xa)
    slope = float(np.polyfit(xa, ta, 1)[0])
    left_t = t[left & np.isfinite(t)]
    right_t = t[right & np.isfinite(t)]
    endpoint_delay = (
        float(np.median(left_t) - np.median(right_t))
        if left_t.size and right_t.size else float("nan")
    )
    middle_peak = peak_z[middle & np.isfinite(peak_z)]
    axial_peak = peak_z[axial & np.isfinite(peak_z)]
    shaft_counts = {}
    for shaft_name in sorted(set(fz["shafts"])):
        shaft_mask = np.asarray(fz["shafts"] == shaft_name, bool)
        shaft_counts[str(shaft_name)] = dict(
            n_participating=int(np.sum(shaft_mask & np.asarray(e["part"], bool))),
            n_usable=int(np.sum(shaft_mask & np.asarray(e["usable"], bool))),
            n_peak_z_ge_5=int(np.sum(shaft_mask & np.asarray(e["part"], bool)
                                     & np.isfinite(peak_z) & (peak_z >= SNR_MIN_Z))),
        )
    return dict(
        axial_shaft=str(shaft), n_axial_usable=int(axial.sum()), n_direction=int(n),
        centroid_vs_axis_rho=float(rho), slope_ms_per_mm=slope,
        right_to_left_monotonic_fraction=float(np.mean(dt > 0)) if dt.size else float("nan"),
        left_minus_right_centroid_ms=endpoint_delay,
        middle_contacts=[str(nm) for nm, keep in zip(fz["names"], middle) if keep],
        n_middle_usable=int(middle.sum()),
        middle_peak_z_min=float(np.min(middle_peak)) if middle_peak.size else float("nan"),
        middle_peak_z_median=float(np.median(middle_peak)) if middle_peak.size else float("nan"),
        axial_peak_z_min=float(np.min(axial_peak)) if axial_peak.size else float("nan"),
        axial_peak_z_median=float(np.median(axial_peak)) if axial_peak.size else float("nan"),
        shaft_counts=shaft_counts,
    )


def pick_exemplar(fz, ev, inv, subject, label, tname):
    """两步：(1) 门 + 像模板排序（并列按离典型值多远打破）；(2) 只对候选池碰原始信号，
    过信噪门后多维 medoid（跨度/参与数/峰高）。**挑选过程绝不读取图上的传播斜率。**

    ⚠️ top_k 只占总事件极小一部分 -> 这不是无条件的"代表事件"，只是
    high-template-concordance exemplar with typical spread and participation within the pool。
    """
    built, feats, ctx = build_exemplar_pool(fz, ev, inv, subject, label, tname)
    pos, dist = ief.select_medoid_event(
        list(built), feats,
        [ctx["target_spread_ms"], ctx["target_n_participating"],
         float(np.median([f[2] for f in feats.values()]))],
    )
    info = ctx["info"]
    rhos = ctx["rhos"]
    print(f"  [{tname}] n={ctx['n_events_in_template']} "
          f"typical spread={ctx['target_spread_ms']:.1f}ms | eligible={info['n_eligible']} "
          f"(tied at rho={info['top_rho']:.3f}: {info['n_tied_at_top']}) | "
          f"pool={len(built)}/{len(ctx['candidate_positions'])} | medoid #{pos} rho={rhos[pos]:.3f}",
          flush=True)
    return built[pos], dict(n_events_in_template=ctx["n_events_in_template"],
                            target_spread_ms=ctx["target_spread_ms"],
                            gate=f"n_snr>={SNR_MIN_CH}, n_usable>={SNR_MIN_CH}, "
                                 f"axial-shaft usable>={AXIAL_MIN_CH} (measurability only; "
                                 f"the slope is never read during selection)",
                            target_n_participating=ctx["target_n_participating"],
                            n_eligible=info["n_eligible"],
                            n_tied_at_top=info["n_tied_at_top"],
                            pool_size=len(ctx["candidate_positions"]),
                            pool_survived_gates=len(built), rho_vs_template=float(rhos[pos]),
                            medoid_distance=dist,
                            caveat="high-template-concordance exemplar with typical spread and "
                                   "participation within the top-K pool; NOT an unconditional "
                                   "representative of all TA/TB events")


def load_explicit_exemplar(fz, ev, inv, subject, event_pos, label, tname):
    """Load one previously audited event without silently re-running exemplar selection."""
    event_pos = int(event_pos)
    if event_pos < 0 or event_pos >= len(ev["labels"]):
        raise IndexError(f"{tname} event_pos={event_pos} is outside the event table")
    if int(ev["labels"][event_pos]) != int(label):
        raise ValueError(
            f"event_pos={event_pos} is assigned to template {int(ev['labels'][event_pos])}, "
            f"not requested {tname} label {int(label)}"
        )
    event = build_event(ev, event_pos, inv, subject, fz)
    print(
        f"  [{tname}] explicitly locked direction-qualified event #{event_pos} "
        f"({event['stem']})",
        flush=True,
    )
    meta = dict(
        selection_mode="explicit_event_pos_after_locked_candidate_screen",
        selected_event_pos=event_pos,
        caveat=(
            "direction-qualified illustrative exemplar selected under the locked candidate-screen "
            "contract; NOT an unconditional representative of all TA/TB events"
        ),
    )
    return event, meta


# ------------------------------------------------------------------ 渲染
def _support(mode, fz, e, which):
    if mode == "participant":
        return e["part"].astype(float)                # 主图（合同 4）
    if mode == "unit":
        return np.ones(len(fz["names"]))              # QC：全部冻结触点
    if mode == "template":
        return fz["support_a"] if which == "a" else fz["support_b"]   # QC：循环论证对照
    raise ValueError(mode)


def _event_field(fz, values, support):
    """Call the established Topic 3/5 physical-mm field smoother without reimplementation."""
    pts = np.asarray(fz["points_mm"], float)
    return _smooth_rank_field_mm(
        pts[:, 0], pts[:, 1], np.clip(np.asarray(values, float), 0.0, None),
        np.asarray(support, float), fz["display_xlim_mm"], fz["display_ylim_mm"],
        fz["display_sigma_mm"],
    )


def _panel(ax, grid, T, pts, cvals, part, vmax, title, *, show_y=False, show_x=False):
    """Established Topic 3/5 physical-mm field grammar with a shared A/B frame."""
    X, Y = grid
    ax.imshow(T, origin="lower",
              extent=[X.min(), X.max(), Y.min(), Y.max()], aspect="equal", cmap=CMAP,
              norm=FIELD_DISPLAY_NORM, interpolation="bilinear")
    ax.scatter(pts[part, 0], pts[part, 1], c=np.clip(cvals[part], 0, vmax), cmap=CMAP,
               norm=FIELD_DISPLAY_NORM, s=34, edgecolors="#34434a", linewidths=0.9, zorder=3)
    ax.scatter(pts[~part, 0], pts[~part, 1], facecolors="none", edgecolors="0.62", s=22,
               linewidths=0.7, zorder=3)
    ax.set_title(title, fontsize=FRAME_TITLE_SIZE)
    ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max())
    ax.tick_params(axis="both", labelsize=FIELD_TICK_LABELSIZE, length=TICK_LENGTH)
    if not show_y:
        ax.set_yticklabels([])
    if show_x:
        ax.set_xlabel("shared TA axis (mm)", fontsize=AXIS_LABELSIZE)


def _frame_window(ea, eb):
    """Tight frame window around the displayed centroids, without an empty late tail."""
    return ief.display_window_ms(
        [e["centroid_ms"][e["usable"]] for e in (ea, eb)],
        pre_ms=FRAME_PRE_MS,
        margin_ms=FRAME_MARGIN_MS,
        min_post_ms=FRAME_MIN_POST_MS,
        max_post_ms=FRAME_MAX_POST_MS,
    )


def _static_frame_times(t_lo, t_hi, n_frames=N_FRAMES):
    """Use a sparse uniform grid; the readout cursor, not a redundant frame, marks t=0."""
    t_lo, t_hi = float(t_lo), float(t_hi)
    if int(n_frames) < 3:
        raise ValueError("static frame grid requires at least three frames")
    times = ief.frame_times_ms(t_lo, t_hi, int(n_frames))
    if len(times) != int(n_frames) or len(np.unique(np.round(times, 1))) != len(times):
        raise ValueError(f"invalid static frame times: {np.round(times, 1)}")
    return np.asarray(times, float)


def _display_vmax_events(events, t_lo, t_hi):
    """One scale for any candidate set, computed from complete intervals, not sampled frames."""
    win = [
        np.clip(
            e["env_z"][e["part"]][
                :, (e["t"] >= e["t0"] + t_lo / ief.MS)
                & (e["t"] <= e["t0"] + t_hi / ief.MS)
            ],
            0.0,
            None,
        )
        for e in events
    ]
    return ief.pooled_vmax(win, q=99.0)


def _display_vmax(ea, eb, t_lo, t_hi):
    return _display_vmax_events((ea, eb), t_lo, t_hi)


def _event_normalization_scales(ea, eb, t_lo, t_hi, override=None):
    """Return one robust-z denominator per event; the displayed field is clipped to 0..1."""
    if override is None:
        scales = {
            "TA": _display_vmax_events((ea,), t_lo, t_hi),
            "TB": _display_vmax_events((eb,), t_lo, t_hi),
        }
    else:
        scales = {lab: float(override[lab]) for lab in ("TA", "TB")}
    if any(not np.isfinite(value) or value <= 0 for value in scales.values()):
        raise ValueError("normalization scales must be positive and finite")
    return scales


def _static_frame_relative_values(raw_values, participant):
    """Normalize one static frame by its three strongest participating contacts.

    Static small multiples are intended to show where the envelope is concentrated.  A single
    complete-window q99 makes low-amplitude but direction-qualified late frames nearly white after
    the fixed 6-mm smoother.  The top-three mean is a robust frame-local reference: it does not let
    one contact set the scale, and the full-field visibility gate prevents quiet noise frames from
    being amplified into the paper panel.
    """
    raw = np.clip(np.asarray(raw_values, float), 0.0, None)
    part = np.asarray(participant, bool)
    if raw.ndim != 1 or part.shape != raw.shape:
        raise ValueError("static frame values and participant mask must be aligned vectors")
    if int(part.sum()) < 3:
        raise ValueError("static frame normalization requires at least three participants")
    scale = float(np.mean(np.sort(raw[part])[-3:]))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("static frame top-three participant scale must be positive and finite")
    return np.clip(raw / scale, 0.0, 1.0), scale


def _normalized_axial_frame_metrics(event, fz, frame_times, scale):
    """Contact-level visibility and axis centroid; rendered field pixels are never inspected."""
    visibility, centroids, _ = _normalized_axial_frame_states(
        event, fz, frame_times, scale,
    )
    return visibility, centroids


def _normalized_axial_frame_states(event, fz, frame_times, scale):
    """Return visibility, centroid and axial contact state for each candidate frame."""
    part = np.asarray(event["part"], bool)
    axial = part & np.asarray(fz["shafts"] == fz["axial_shaft"], bool)
    if int(axial.sum()) < 3:
        raise ValueError("joint-visible frame selection requires >=3 axial participants")
    x = np.asarray(fz["ax_mm"], float)[axial]
    visibility, centroids, states = [], [], []
    for frame_ms in np.asarray(frame_times, float):
        raw = np.clip(
            ief.values_at(
                event["env_z"], event["t"], event["t0"], frame_ms,
                avg_ms=FRAME_AVG_MS,
            ),
            0.0, None,
        )
        values = np.clip(raw / float(scale), 0.0, 1.0)[axial]
        states.append(values)
        top_k = min(3, len(values))
        visibility.append(float(np.mean(np.sort(values)[-top_k:])))
        weight = float(np.sum(values))
        centroids.append(float(np.sum(x * values) / weight) if weight > 0 else float("nan"))
    return (
        np.asarray(visibility, float),
        np.asarray(centroids, float),
        np.asarray(states, float),
    )


def _normalized_full_frame_states(event, fz, frame_times, scale):
    """Return full-participant visibility, centroid and top-3 hotspot along the shared axis.

    These quantities are computed from the exact contact values and frozen coordinates consumed
    by the final 2-D field.  They deliberately include every participating shaft, closing the old
    mismatch where an ICL-only selector could approve frames whose rendered all-shaft hotspot did
    not move.
    """
    part = np.asarray(event["part"], bool)
    if int(part.sum()) < 3:
        raise ValueError("full-field frame selection requires >=3 participants")
    x = np.asarray(fz["ax_mm"], float)[part]
    visibility, centroids, hotspots, states = [], [], [], []
    for frame_ms in np.asarray(frame_times, float):
        raw = np.clip(
            ief.values_at(
                event["env_z"], event["t"], event["t0"], frame_ms,
                avg_ms=FRAME_AVG_MS,
            ),
            0.0, None,
        )
        values = np.clip(raw / float(scale), 0.0, 1.0)[part]
        states.append(values)
        top_k = min(3, len(values))
        top = np.argsort(values)[-top_k:]
        top_values = values[top]
        visibility.append(float(np.mean(top_values)))
        weight = float(np.sum(values))
        top_weight = float(np.sum(top_values))
        centroids.append(float(np.sum(x * values) / weight) if weight > 0 else float("nan"))
        hotspots.append(
            float(np.sum(x[top] * top_values) / top_weight)
            if top_weight > 0 else float("nan")
        )
    return (
        np.asarray(visibility, float),
        np.asarray(centroids, float),
        np.asarray(hotspots, float),
        np.asarray(states, float),
    )


def _axial_endpoint_contrast(event, fz, states):
    """Return right-minus-left energy on the fixed outer thirds of the axial shaft.

    This contact-level readability metric does not inspect rendered field pixels.  Unlike a
    centroid, it only changes when energy is actually handed from one shaft end to the other, so
    a diffuse energy plateau cannot pass as a distinct propagation frame.
    """
    part = np.asarray(event["part"], bool)
    shaft = np.asarray(fz["shafts"] == fz["axial_shaft"], bool)
    axial = part & shaft
    x_all = np.asarray(fz["ax_mm"], float)[shaft]
    x = np.asarray(fz["ax_mm"], float)[axial]
    states = np.asarray(states, float)
    if states.ndim != 2 or states.shape[1] != len(x):
        raise ValueError("axial endpoint contrast received a misaligned contact-state matrix")
    q1, q2 = np.quantile(x_all, [1.0 / 3.0, 2.0 / 3.0])
    left, right = x <= q1, x >= q2
    if not left.any() or not right.any():
        raise ValueError("axial endpoint contrast requires participants at both shaft ends")
    return np.mean(states[:, right], axis=1) - np.mean(states[:, left], axis=1)


def _select_joint_visible_frame_times(ea, eb, fz, t_lo, t_hi, scales, n_frames=N_FRAMES):
    """Pick equally spaced frames with opposite motion in the final all-shaft field inputs.

    Selection uses normalized contact envelopes only.  The strict path searches arithmetic time
    sequences on the biological 2-ms grid.  It retains the axial direction gates, then additionally
    requires the full-participant centroid and top-3 hotspot to move at every step in the direction
    that the final 2-D renderer is expected to show.  A generic uniform fallback keeps other
    subjects renderable.
    """
    n_frames = int(n_frames)
    grid = np.arange(
        float(t_lo), float(t_hi) + 0.25 * STATIC_FRAME_GRID_STEP_MS,
        STATIC_FRAME_GRID_STEP_MS,
    )
    if grid.size < n_frames:
        fallback = _static_frame_times(t_lo, t_hi, n_frames)
        return fallback, {"selection_mode": "uniform_fallback_grid_too_short"}
    try:
        vis_a, cen_a, state_a = _normalized_axial_frame_states(
            ea, fz, grid, scales["TA"],
        )
        vis_b, cen_b, state_b = _normalized_axial_frame_states(
            eb, fz, grid, scales["TB"],
        )
        endpoint_a = _axial_endpoint_contrast(ea, fz, state_a)
        endpoint_b = _axial_endpoint_contrast(eb, fz, state_b)
        full_vis_a, full_cen_a, hot_a, _ = _normalized_full_frame_states(
            ea, fz, grid, scales["TA"],
        )
        full_vis_b, full_cen_b, hot_b, _ = _normalized_full_frame_states(
            eb, fz, grid, scales["TB"],
        )
    except ValueError:
        fallback = _static_frame_times(t_lo, t_hi, n_frames)
        return fallback, {"selection_mode": "uniform_fallback_axial_geometry"}
    joint = np.minimum(vis_a, vis_b)
    full_joint = np.minimum(full_vis_a, full_vis_b)
    feasible = []
    late_floor = float(t_hi) - STATIC_FRAME_MIN_GAP_MS
    min_step_idx = int(np.ceil(STATIC_FRAME_MIN_GAP_MS / STATIC_FRAME_GRID_STEP_MS))
    for step_idx in range(min_step_idx, len(grid)):
        for start_idx in range(len(grid)):
            idx = start_idx + np.arange(n_frames, dtype=int) * step_idx
            if idx[-1] >= len(grid):
                break
            times = grid[idx]
            if times[0] > STATIC_FRAME_GRID_STEP_MS or times[-1] < late_floor:
                continue
            if (
                not np.all(np.isfinite(cen_a[idx]))
                or not np.all(np.isfinite(cen_b[idx]))
                or not np.all(np.isfinite(full_cen_a[idx]))
                or not np.all(np.isfinite(full_cen_b[idx]))
                or not np.all(np.isfinite(hot_a[idx]))
                or not np.all(np.isfinite(hot_b[idx]))
            ):
                continue
            if float(np.min(joint[idx])) < STATIC_FRAME_MIN_JOINT_VISIBILITY:
                continue
            rho_a = float(spearmanr(times, cen_a[idx]).correlation)
            rho_b = float(spearmanr(times, cen_b[idx]).correlation)
            if (
                not np.isfinite(rho_a)
                or not np.isfinite(rho_b)
                or rho_a < STATIC_FRAME_MIN_CENTROID_RHO
                or rho_b > -STATIC_FRAME_MIN_CENTROID_RHO
            ):
                continue
            span_a = float(cen_a[idx][-1] - cen_a[idx][0])
            span_b = float(cen_b[idx][0] - cen_b[idx][-1])
            if span_a < STATIC_FRAME_MIN_AXIS_SPAN_MM or span_b < STATIC_FRAME_MIN_AXIS_SPAN_MM:
                continue
            state_step_a = np.sqrt(np.mean(np.diff(state_a[idx], axis=0) ** 2, axis=1))
            state_step_b = np.sqrt(np.mean(np.diff(state_b[idx], axis=0) ** 2, axis=1))
            joint_state_step = np.minimum(state_step_a, state_step_b)
            if float(np.min(joint_state_step)) < STATIC_FRAME_MIN_STATE_DISTANCE:
                continue
            handoff_a = np.diff(endpoint_a[idx])
            handoff_b = -np.diff(endpoint_b[idx])
            joint_handoff = np.minimum(handoff_a, handoff_b)
            if float(np.min(joint_handoff)) < STATIC_FRAME_MIN_ENDPOINT_HANDOFF:
                continue
            full_centroid_step_a = np.diff(full_cen_a[idx])
            full_centroid_step_b = -np.diff(full_cen_b[idx])
            joint_full_centroid_step = np.minimum(
                full_centroid_step_a, full_centroid_step_b,
            )
            hotspot_step_a = np.diff(hot_a[idx])
            hotspot_step_b = -np.diff(hot_b[idx])
            joint_hotspot_step = np.minimum(hotspot_step_a, hotspot_step_b)
            if float(np.min(full_joint[idx])) < STATIC_FRAME_MIN_FULL_VISIBILITY:
                continue
            if (
                float(np.min(joint_full_centroid_step))
                < STATIC_FRAME_MIN_FULL_CENTROID_STEP_MM
            ):
                continue
            if float(np.min(joint_hotspot_step)) < STATIC_FRAME_MIN_HOTSPOT_STEP_MM:
                continue
            score = float(
                2.0 * np.min(joint_hotspot_step)
                + np.mean(joint_hotspot_step)
                + np.min(joint_full_centroid_step)
                + 0.5 * np.mean(joint_full_centroid_step)
                + 2.0 * np.min(joint_handoff)
                + np.mean(joint_handoff)
                + 2.0 * np.min(joint[idx])
                + np.mean(joint[idx])
                + 2.0 * np.min(joint_state_step)
                + np.mean(joint_state_step)
                + 0.1 * (rho_a - rho_b)
            )
            feasible.append(
                (
                    score, tuple(float(x) for x in times), idx, span_a, span_b,
                    rho_a, rho_b, state_step_a, state_step_b, joint_state_step,
                    handoff_a, handoff_b, joint_handoff,
                    full_centroid_step_a, full_centroid_step_b,
                    joint_full_centroid_step, hotspot_step_a, hotspot_step_b,
                    joint_hotspot_step,
                    float(times[1] - times[0]),
                )
            )
    if not feasible:
        fallback = _static_frame_times(t_lo, t_hi, n_frames)
        return fallback, {
            "selection_mode": "uniform_fallback_no_strict_equal_interval_solution",
            "grid_step_ms": STATIC_FRAME_GRID_STEP_MS,
            "times_are_equally_spaced": True,
        }
    (
        score, selected, idx, span_a, span_b, rho_a, rho_b,
        state_step_a, state_step_b, joint_state_step,
        handoff_a, handoff_b, joint_handoff,
        full_centroid_step_a, full_centroid_step_b, joint_full_centroid_step,
        hotspot_step_a, hotspot_step_b, joint_hotspot_step, equal_interval_ms,
    ) = max(
        feasible, key=lambda row: (row[0], tuple(-x for x in row[1]))
    )
    return np.asarray(selected, float), {
        "selection_mode": "equal_interval_full_field_hotspot_v6",
        "rendered_pixels_used": False,
        "times_are_equally_spaced": True,
        "equal_interval_ms": equal_interval_ms,
        "grid_step_ms": STATIC_FRAME_GRID_STEP_MS,
        "minimum_gap_ms": STATIC_FRAME_MIN_GAP_MS,
        "minimum_joint_visibility": STATIC_FRAME_MIN_JOINT_VISIBILITY,
        "minimum_axis_span_mm": STATIC_FRAME_MIN_AXIS_SPAN_MM,
        "minimum_contact_state_distance": STATIC_FRAME_MIN_STATE_DISTANCE,
        "minimum_abs_centroid_time_rho": STATIC_FRAME_MIN_CENTROID_RHO,
        "minimum_endpoint_handoff_per_step": STATIC_FRAME_MIN_ENDPOINT_HANDOFF,
        "minimum_full_participant_visibility": STATIC_FRAME_MIN_FULL_VISIBILITY,
        "minimum_full_centroid_step_mm": STATIC_FRAME_MIN_FULL_CENTROID_STEP_MM,
        "minimum_top3_hotspot_step_mm": STATIC_FRAME_MIN_HOTSPOT_STEP_MM,
        "objective_score": float(score),
        "joint_visibility": [float(x) for x in joint[idx]],
        "ta_axis_centroids_mm": [float(x) for x in cen_a[idx]],
        "tb_axis_centroids_mm": [float(x) for x in cen_b[idx]],
        "ta_centroid_vs_time_rho": rho_a,
        "tb_centroid_vs_time_rho": rho_b,
        "ta_contact_state_step_rms": [float(x) for x in state_step_a],
        "tb_contact_state_step_rms": [float(x) for x in state_step_b],
        "joint_contact_state_step_rms": [float(x) for x in joint_state_step],
        "ta_right_minus_left_endpoint_energy": [float(x) for x in endpoint_a[idx]],
        "tb_right_minus_left_endpoint_energy": [float(x) for x in endpoint_b[idx]],
        "ta_endpoint_handoff_step": [float(x) for x in handoff_a],
        "tb_endpoint_handoff_step": [float(x) for x in handoff_b],
        "joint_endpoint_handoff_step": [float(x) for x in joint_handoff],
        "joint_full_participant_visibility": [float(x) for x in full_joint[idx]],
        "ta_full_centroids_mm": [float(x) for x in full_cen_a[idx]],
        "tb_full_centroids_mm": [float(x) for x in full_cen_b[idx]],
        "ta_full_centroid_step_mm": [float(x) for x in full_centroid_step_a],
        "tb_full_centroid_step_mm": [float(x) for x in full_centroid_step_b],
        "joint_full_centroid_step_mm": [float(x) for x in joint_full_centroid_step],
        "ta_top3_hotspots_mm": [float(x) for x in hot_a[idx]],
        "tb_top3_hotspots_mm": [float(x) for x in hot_b[idx]],
        "ta_top3_hotspot_step_mm": [float(x) for x in hotspot_step_a],
        "tb_top3_hotspot_step_mm": [float(x) for x in hotspot_step_b],
        "joint_top3_hotspot_step_mm": [float(x) for x in joint_hotspot_step],
        "ta_axis_span_mm": span_a,
        "tb_axis_span_mm": span_b,
    }


def _gif_frame_times(t_lo, t_hi, step_ms=GIF_STEP_MS):
    if not np.isfinite(step_ms) or step_ms <= 0:
        raise ValueError("GIF step_ms must be positive and finite")
    times = np.arange(float(t_lo), float(t_hi) + 0.25 * step_ms, float(step_ms))
    if times.size == 0 or times[-1] < float(t_hi) - 0.25 * step_ms:
        times = np.append(times, float(t_hi))
    else:
        times[-1] = min(times[-1], float(t_hi))
    return times


def _time_label(x):
    return "0 ms" if abs(float(x)) < 0.5 else f"{float(x):+.0f} ms"


def _common_readout_xlim(*events):
    """Use the common recorded window so neither row contains an event-specific white strip."""
    if len(events) < 2:
        raise ValueError("readout x-limit requires at least two events")
    lo = float(max(e["tile_lo_ms"] for e in events))
    hi = float(min(e["tile_hi_ms"] for e in events))
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError(f"events have no common readout window: {lo:g}..{hi:g} ms")
    for event in events:
        centroids = np.asarray(event["centroid_ms"], float)[np.asarray(event["usable"], bool)]
        if centroids.size and (np.nanmin(centroids) < lo or np.nanmax(centroids) > hi):
            raise ValueError("common readout window would crop a displayed centroid")
    return lo, hi


def _readout(
    ax, e, fz, order, xlim, _stats, title=None, *, template,
    show_xlabel=True, row_label=None,
):
    """Exact Fig1a stacked spectrogram + dominant-enhancement centroid trajectory."""
    sel = [c for c in order if e["usable"][c]]
    if not sel:
        raise ValueError("readout has no participating contact with a finite Fig1a centroid")
    n_freq = len(e["spec_freq_hz"])
    shown = np.concatenate([e["spec"][c] for c in sel], axis=0)
    t_edges = full_extent_edges(
        np.asarray(e["spec_t_ms"], float), float(e["tile_lo_ms"]), float(e["tile_hi_ms"]),
    )
    y_edges = np.arange(len(sel) * n_freq + 1, dtype=float)
    im = ax.pcolormesh(t_edges, y_edges, shown, cmap=FIG1A_CMAP, vmin=0.0, vmax=1.0,
                       shading="flat", rasterized=True)
    for k in range(1, len(sel)):
        ax.axhline(k * n_freq, color="#c7c7c7", linewidth=0.45, linestyle="--")
    xs = np.asarray([e["centroid_ms"][c] for c in sel], float)
    ys = np.asarray(
        [k * n_freq + e["centroid_freq_index"][c] + 0.5 for k, c in enumerate(sel)], float,
    )
    semantic = TEMPLATE_COLORS[template]
    ax.plot(xs, ys, "-", color=semantic, lw=1.0, alpha=0.95, zorder=4)
    ax.scatter(xs, ys, s=13, facecolors=CENTROID_FACE, edgecolors=semantic,
               linewidth=0.35, zorder=5)
    ax.set_xlim(*xlim)
    ax.set_ylim(len(sel) * n_freq, 0.0)
    ax.set_box_aspect(READOUT_BOX_ASPECT)
    ax.margins(x=0.0)
    ax.set_yticks((np.arange(len(sel)) + 0.5) * n_freq)
    ax.set_yticklabels([fz["names"][c] for c in sel], fontsize=CONTACT_TICK_LABELSIZE)
    if show_xlabel:
        ax.set_xlabel("time (ms)", fontsize=AXIS_LABELSIZE)
    if title:
        ax.set_title(
            title, fontsize=TEMPLATE_LABEL_SIZE, color=semantic, fontweight="bold",
            x=0.96, ha="right", pad=3,
        )
    if row_label:
        ax.set_ylabel(
            row_label, fontsize=TEMPLATE_LABEL_SIZE,
            color=semantic, fontweight="bold",
        )
    ax.tick_params(axis="x", labelsize=READOUT_TICK_LABELSIZE, length=TICK_LENGTH)
    ax.tick_params(axis="y", labelsize=CONTACT_TICK_LABELSIZE, length=0)
    return im, sel, xs, ys


def _label_colorbar(cb, text):
    cb.set_label(text, fontsize=CBAR_LABELSIZE, labelpad=3)
    cb.ax.tick_params(labelsize=CBAR_TICK_LABELSIZE, length=TICK_LENGTH)


def _template_panel(ax, fz, template, *, show_y, show_x):
    """Embed the frozen population-template rank field via its canonical renderer."""
    payload = fz["template_payloads"][template]
    draw_interictal_rank_field_panel(
        ax, payload, template, compact=False, panel_title=f"{template} template",
        contact_size=TEMPLATE_CONTACT_SIZE,
        contact_outline_lw=TEMPLATE_CONTACT_OUTLINE_LW,
    )
    ax.set_title(
        f"{template} template", fontsize=TEMPLATE_LABEL_SIZE,
        color=TEMPLATE_COLORS[template], fontweight="bold",
    )
    ax.set_xlabel("shared TA axis (mm)" if show_x else "", fontsize=AXIS_LABELSIZE)
    ax.set_ylabel("y (mm)" if show_y else "", fontsize=AXIS_LABELSIZE)
    ax.tick_params(axis="both", labelsize=FIELD_TICK_LABELSIZE, length=TICK_LENGTH)
    if not show_y:
        ax.set_yticklabels([])
    return ax.images[-1]


def _template_rank_colorbar(fig, cax, fz, template):
    """Show the frozen template's actual rank numbers while preserving viridis colours."""
    values = np.asarray(fz["template_payloads"][template]["rank_values"], float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError(f"{template}: fewer than two finite frozen ranks")
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi <= lo:
        raise ValueError(f"{template}: frozen rank range is degenerate")
    mid = float(0.5 * (lo + hi))
    ticks = [lo, mid, hi]

    def fmt(value):
        return f"{value:.0f}" if np.isclose(value, np.round(value)) else f"{value:g}"

    cb = fig.colorbar(
        ScalarMappable(Normalize(lo, hi), cmap="viridis"), cax=cax,
    )
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{fmt(lo)}  early", fmt(mid), f"{fmt(hi)}  late"])
    cb.ax.set_title("ranks", fontsize=CBAR_LABELSIZE, pad=4, loc="left")
    cb.ax.tick_params(labelsize=CBAR_TICK_LABELSIZE, length=TICK_LENGTH)
    return cb, (lo, hi)


def _subject_title(ds_sid):
    dataset, sid = ds_sid.split("_", 1)
    return f"E{sid}" if dataset == "epilepsiae" else sid


def render(
    ds_sid, fz, ea, eb, sa, sb, out_png, *, support_mode="participant",
    extra_outputs=(), dpi=125, frame_window=None, normalization_scales_override=None,
):
    pts = fz["points_mm"]
    sup = {"TA": _support(support_mode, fz, ea, "a"), "TB": _support(support_mode, fz, eb, "b")}
    t_lo, t_hi = _frame_window(ea, eb) if frame_window is None else map(float, frame_window)
    # Complete-window q99 is used by the deterministic frame selector and retained for audit.
    # The four static small multiples are then normalized frame-wise by the participant top-three
    # mean so low-amplitude late frames remain spatially legible after the fixed 6-mm smoother.
    scales = _event_normalization_scales(
        ea, eb, t_lo, t_hi, override=normalization_scales_override,
    )
    fts, frame_selection = _select_joint_visible_frame_times(
        ea, eb, fz, t_lo, t_hi, scales, N_FRAMES,
    )
    vmax = 1.0

    order = list(np.argsort(fz["ax_mm"]))
    readout_xlim = _common_readout_xlim(ea, eb)
    fig, axes = plt.subplots(2, N_LAYOUT_COLS,
                             figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
                             layout="constrained",
                             gridspec_kw={
                                 "width_ratios": [READOUT_WIDTH_RATIO, SPEC_CBAR_WIDTH_RATIO,
                                                  GROUP_GAP_WIDTH_RATIO]
                                 + [1] * N_FRAMES
                                 + [FIELD_CBAR_WIDTH_RATIO, TEMPLATE_GAP_WIDTH_RATIO,
                                    TEMPLATE_FIELD_WIDTH_RATIO, TEMPLATE_CBAR_WIDTH_RATIO],
                             })
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(w_pad=0.02, h_pad=0.02, wspace=0.015, hspace=0.015)
    frame_scales = {"TA": [], "TB": []}
    for r, (lab, e, st) in enumerate((("TA", ea, sa), ("TB", eb, sb))):
        for c, x in enumerate(fts):
            raw = np.clip(
                ief.values_at(e["env_z"], e["t"], e["t0"], float(x), avg_ms=FRAME_AVG_MS),
                0.0, None,
            )
            v, frame_scale = _static_frame_relative_values(raw, e["part"])
            frame_scales[lab].append(float(frame_scale))
            X, Y, T, _, _ = _event_field(fz, v, sup[lab])
            field_ax = axes[r, FRAME_COL_START + c]
            _panel(field_ax, (X, Y), T, pts, v, e["part"], vmax, _time_label(x),
                   show_y=(c == 0), show_x=(r == 1 and c == N_FRAMES // 2))
        field_cax = axes[r, FIELD_CBAR_COL]
        field_cax.set_box_aspect(1.0 / FIELD_CBAR_WIDTH_RATIO)
        field_cb = fig.colorbar(
            ScalarMappable(FIELD_DISPLAY_NORM, cmap=CMAP), cax=field_cax,
        )
        field_cb.set_ticks([0.0, 0.5, 1.0])
        _label_colorbar(field_cb, STATIC_FIELD_CBAR_LABEL)
        axes[r, GROUP_GAP_COL].set_axis_off()
        axes[r, TEMPLATE_GAP_COL].set_axis_off()
        spec_im, _, _, _ = _readout(
            axes[r, READOUT_COL], e, fz, order, readout_xlim, st,
            title=f"Sample from {lab}", template=lab, show_xlabel=True, row_label=lab,
        )
        spec_cax = axes[r, READOUT_CBAR_COL]
        spec_cax.set_box_aspect(1.0 / SPEC_CBAR_WIDTH_RATIO)
        spec_cb = fig.colorbar(spec_im, cax=spec_cax)
        spec_cb.set_ticks([0, 1])
        _label_colorbar(spec_cb, "Normalized magnitude")
        spec_cb.outline.set_visible(False)
        _template_panel(
            axes[r, TEMPLATE_FIELD_COL], fz, lab,
            show_y=True, show_x=(r == 1),
        )
        template_cax = axes[r, TEMPLATE_CBAR_COL]
        template_cax.set_box_aspect(1.0 / TEMPLATE_CBAR_WIDTH_RATIO)
        _template_rank_colorbar(fig, template_cax, fz, lab)
    fig.suptitle(
        _subject_title(ds_sid), x=0.01, ha="left",
        fontsize=MAIN_TITLE_SIZE, fontweight="bold",
    )
    outputs = [Path(out_png), *(Path(p) for p in extra_outputs)]
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] {Path(out_png).name}  frame-relative 0..1 "
          f"(selector q99 TA={scales['TA']:.1f}, TB={scales['TB']:.1f})  "
          f"window {t_lo:+.0f}..{t_hi:+.0f} ms ({support_mode})", flush=True)
    return dict(t_lo_ms=float(t_lo), t_hi_ms=float(t_hi), vmax=float(vmax),
                frame_times_ms=[float(x) for x in fts], support_mode=support_mode,
                cmap=CMAP_NAME, normalization_mode=STATIC_FIELD_NORMALIZATION_ID,
                display_norm=FIELD_DISPLAY_NORM_ID,
                display_gamma=float(FIELD_DISPLAY_GAMMA),
                frame_normalization_scales_robust_z=frame_scales,
                frame_normalization_reference=(
                    "within each displayed static frame, mean of the three strongest "
                    "participating-contact envelopes equals 1; values are clipped to 0..1"
                ),
                selection_normalization_mode=FIELD_NORMALIZATION_ID,
                complete_window_q99_robust_z_for_selection_and_audit={
                    lab: float(value) for lab, value in scales.items()
                },
                static_frame_selection=frame_selection,
                field_colorbars=("one labeled relative HFO envelope bar per TA/TB row; each "
                                 "static frame uses its own participant top-three mean reference"),
                readout_colorbars="one labeled normalized-magnitude 0..1 bar per TA/TB row",
                template_colorbars=("one viridis bar per TA/TB row titled ranks; actual frozen "
                                    "rank numbers with separate early/late endpoint text"),
                template_rank_ranges={
                    lab: [float(np.nanmin(fz["template_payloads"][lab]["rank_values"])),
                          float(np.nanmax(fz["template_payloads"][lab]["rank_values"]))]
                    for lab in ("TA", "TB")
                },
                panel_order=("single-event readout | readout colorbar | gap | event envelope "
                             "frames | envelope colorbar | gap | frozen template rank field | "
                             "template colorbar"),
                readout_xlim_ms=[float(x) for x in readout_xlim],
                readout_xlim_mode="common intersection of the two recorded STFT windows",
                display_sigma_mm=float(fz["display_sigma_mm"]),
                display_xlim_mm=[float(x) for x in fz["display_xlim_mm"]],
                display_ylim_mm=[float(x) for x in fz["display_ylim_mm"]],
                transverse_sign=int(fz["transverse_sign"]), figure=str(out_png),
                extra_outputs=[str(p) for p in outputs[1:]], dpi=int(dpi))


def render_gif(
    ds_sid, fz, ea, eb, sa, sb, out_gif, *, support_mode="participant",
    step_ms=GIF_STEP_MS, fps=GIF_FPS, dpi=120,
    frame_window=None, normalization_scales_override=None,
):
    """Animate the same two exemplars and frozen field contract used by the static candidate."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("GIF fps must be positive and finite")
    pts = np.asarray(fz["points_mm"], float)
    sup = {
        "TA": _support(support_mode, fz, ea, "a"),
        "TB": _support(support_mode, fz, eb, "b"),
    }
    t_lo, t_hi = _frame_window(ea, eb) if frame_window is None else map(float, frame_window)
    frame_times = _gif_frame_times(t_lo, t_hi, step_ms=step_ms)
    scales = _event_normalization_scales(
        ea, eb, t_lo, t_hi, override=normalization_scales_override,
    )
    vmax = 1.0
    order = list(np.argsort(fz["ax_mm"]))
    readout_xlim = _common_readout_xlim(ea, eb)

    fig, axes = plt.subplots(
        2, 8, figsize=(9.0, FIGURE_HEIGHT_IN), layout="constrained",
        gridspec_kw={
            "width_ratios": [READOUT_WIDTH_RATIO, SPEC_CBAR_WIDTH_RATIO,
                             GROUP_GAP_WIDTH_RATIO, 1.0, FIELD_CBAR_WIDTH_RATIO,
                             TEMPLATE_GAP_WIDTH_RATIO, TEMPLATE_FIELD_WIDTH_RATIO,
                             TEMPLATE_CBAR_WIDTH_RATIO],
        },
    )
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(w_pad=0.02, h_pad=0.02, wspace=0.015, hspace=0.015)

    cursor_lines, field_images, field_scatters, field_titles = [], [], [], []
    for r, (lab, e, st) in enumerate((("TA", ea, sa), ("TB", eb, sb))):
        spec_im, _, _, _ = _readout(
            axes[r, 0], e, fz, order, readout_xlim, st,
            title=f"Sample from {lab}", template=lab, show_xlabel=True, row_label=lab,
        )
        cursor_lines.append(
            axes[r, 0].axvline(
                frame_times[0], color="black", linestyle="--", linewidth=1.2,
                alpha=0.8, zorder=7,
            )
        )
        spec_cax = axes[r, 1]
        spec_cax.set_box_aspect(1.0 / SPEC_CBAR_WIDTH_RATIO)
        spec_cb = fig.colorbar(spec_im, cax=spec_cax)
        spec_cb.set_ticks([0, 1])
        _label_colorbar(spec_cb, "Normalized magnitude")
        spec_cb.outline.set_visible(False)
        axes[r, 2].set_axis_off()

        raw = np.clip(
            ief.values_at(e["env_z"], e["t"], e["t0"], frame_times[0],
                          avg_ms=FRAME_AVG_MS),
            0.0, None,
        )
        v = np.clip(raw / scales[lab], 0.0, 1.0)
        X, Y, T, _, _ = _event_field(fz, v, sup[lab])
        field_ax = axes[r, 3]
        _panel(
            field_ax, (X, Y), T, pts, v, e["part"], vmax,
            _time_label(frame_times[0]), show_y=True, show_x=(r == 1),
        )
        field_images.append(field_ax.images[-1])
        field_scatters.append(field_ax.collections[0])
        field_titles.append(field_ax.title)
        field_cax = axes[r, 4]
        field_cax.set_box_aspect(1.0 / FIELD_CBAR_WIDTH_RATIO)
        field_cb = fig.colorbar(
            ScalarMappable(FIELD_DISPLAY_NORM, cmap=CMAP),
            cax=field_cax,
        )
        field_cb.set_ticks([0.0, 0.5, 1.0])
        _label_colorbar(field_cb, "Normalized HFO envelope")
        axes[r, 5].set_axis_off()
        _template_panel(
            axes[r, 6], fz, lab, show_y=True, show_x=(r == 1),
        )
        template_cax = axes[r, 7]
        template_cax.set_box_aspect(1.0 / TEMPLATE_CBAR_WIDTH_RATIO)
        _template_rank_colorbar(fig, template_cax, fz, lab)

    fig.suptitle(
        _subject_title(ds_sid), x=0.01, ha="left",
        fontsize=MAIN_TITLE_SIZE, fontweight="bold",
    )

    def update(frame_idx):
        x = float(frame_times[frame_idx])
        artists = []
        for r, (lab, e) in enumerate((("TA", ea), ("TB", eb))):
            raw = np.clip(
                ief.values_at(e["env_z"], e["t"], e["t0"], x,
                              avg_ms=FRAME_AVG_MS),
                0.0, None,
            )
            v = np.clip(raw / scales[lab], 0.0, 1.0)
            _, _, T, _, _ = _event_field(fz, v, sup[lab])
            field_images[r].set_data(T)
            field_scatters[r].set_array(np.clip(v[e["part"]], 0.0, vmax))
            field_titles[r].set_text(_time_label(x))
            cursor_lines[r].set_xdata([x, x])
            artists.extend((field_images[r], field_scatters[r], field_titles[r], cursor_lines[r]))
        return artists

    out_gif = Path(out_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(
        fig, update, frames=len(frame_times), interval=1000.0 / float(fps),
        blit=False, repeat=True,
    )
    anim.save(
        out_gif,
        writer=PillowWriter(
            fps=float(fps),
            metadata={"title": f"{_subject_title(ds_sid)} interictal TA/TB envelope propagation"},
        ),
        dpi=dpi,
    )
    plt.close(fig)
    print(
        f"  [gif] {out_gif.name}  n={len(frame_times)}  biological step={step_ms:g} ms  "
        f"playback={fps:g} fps",
        flush=True,
    )
    return dict(
        figure=str(out_gif), support_mode=support_mode,
        frame_times_ms=[float(x) for x in frame_times], n_frames=int(len(frame_times)),
        biological_step_ms=float(step_ms), playback_fps=float(fps),
        playback_duration_sec=float(len(frame_times) / fps),
        frame_average_ms=float(FRAME_AVG_MS), t_lo_ms=float(t_lo), t_hi_ms=float(t_hi),
        vmax=float(vmax), cmap=CMAP_NAME, normalization_mode=FIELD_NORMALIZATION_ID,
        display_norm=FIELD_DISPLAY_NORM_ID, display_gamma=float(FIELD_DISPLAY_GAMMA),
        normalization_scales_robust_z={lab: float(value) for lab, value in scales.items()},
        display_sigma_mm=float(fz["display_sigma_mm"]),
        readout_cursor="black dashed line at the displayed field time",
        panel_order=("single-event readout | normalized-magnitude colorbar | current envelope "
                     "field | normalized-envelope colorbar | frozen template rank field | "
                     "rank colorbar"),
        readout_xlim_ms=[float(x) for x in readout_xlim],
        readout_xlim_mode="common intersection of the two recorded STFT windows",
        template_field=("frozen viridis propagation rank; colorbar shows actual frozen rank "
                        "numbers and separate early/late endpoint text"),
        template_rank_ranges={
            lab: [float(np.nanmin(fz["template_payloads"][lab]["rank_values"])),
                  float(np.nanmax(fz["template_payloads"][lab]["rank_values"]))]
            for lab in ("TA", "TB")
        },
    )


def _paper_stem(ds_sid):
    return f"fig2c_candidate_{_subject_title(ds_sid)}_interictal_event_envelope_field"


def _write_paper_ready_readme(figures_dir, ds_sid, js, static_meta, gif_meta):
    figures_dir = Path(figures_dir)
    stem = _paper_stem(ds_sid)
    ta = js["exemplar"]["TA"]["stats"][f"shaft_{js['axial_shaft']}"]
    tb = js["exemplar"]["TB"]["stats"][f"shaft_{js['axial_shaft']}"]
    frame_text = ", ".join(
        _time_label(x).removesuffix(" ms") for x in static_meta["frame_times_ms"]
    )
    gif_block = ""
    if gif_meta is not None:
        gif_block = f"""
### {stem}.gif

同一对 TA/TB exemplar 的动态版本。生物学时间从 {gif_meta['t_lo_ms']:+.0f} 到 {gif_meta['t_hi_ms']:+.0f} ms，帧间隔 {gif_meta['biological_step_ms']:.0f} ms；播放速度 {gif_meta['playback_fps']:.0f} fps 只用于观看，不代表真实时间倍率。左侧黑色虚线与中间当前 envelope field 帧严格同步，最右冻结 template rank field 保持不动。

**关注点**：比较 TA/TB 热区沿冻结 shared axis 的相反移动；GIF 与静态 candidate 使用同一 exemplar、support、几何、6 mm display kernel 和 colormap，但量纲承担不同任务：GIF 固定每事件完整窗 q99 以保留连续幅度演化，静态小图逐帧以最强三个参与触点的均值归一化以突出空间位置。两者都不能用于比较 TA/TB 绝对 robust-z 幅度。
"""
    text = f"""# Fig2-C candidate：E1146 间期单事件包络传播场

### {stem}.png / .pdf

Fig2-C representative-subject 单事件候选：每行只放一个 exemplar（TA 一次、TB 一次），不是多事件 train。三组内容依次为：左侧 `Sample from TA/TB` normalized-magnitude spectrogram 与质心轨迹；中间 participant-only 单带 HFO Hilbert amplitude envelope 场；最右冻结群体 TA/TB propagation-rank field。静态帧为 `{frame_text} ms`，由 contact-level equal-interval full-field selector 在 2 ms 网格上确定：全部时间间隔完全相等；除轴杆共同可见度、状态分离、质心方向和端点交接门外，最终二维场实际使用的全部参与触点还必须共同可见度至少 {STATIC_FRAME_MIN_FULL_VISIBILITY:.2f}，每一步全参与触点质心至少移动 {STATIC_FRAME_MIN_FULL_CENTROID_STEP_MM:.1f} mm，top-3 热点至少移动 {STATIC_FRAME_MIN_HOTSPOT_STEP_MM:.1f} mm（TA 向右、TB 向左）。选择过程只读取接触点包络与冻结坐标，不读取渲染像素。为避免完整窗 q99 把有效但幅度较低的后帧压成近白色，四幅静态小图分别以该帧最强三个参与触点的 robust-z envelope 均值为 1 并 clip 到 0–1；colorbar 因此写 `{STATIC_FIELD_CBAR_LABEL}`。低饱和蓝灰 `{CMAP_NAME}` 继续固定使用 `PowerNorm(gamma={FIELD_DISPLAY_GAMMA:.2f})`。这种 frame-relative 显示只比较空间集中位置，不比较帧间、TA/TB 间绝对幅度；连续幅度演化保留在 GIF 的 complete-window q99 尺度中。最右使用 `viridis`，colorbar 顶部写 `ranks` 并显示 artifact 实际 rank，最低/最高端分别附 early/late。左侧两行取真实 STFT 窗的共同交集，避免无数据白边；三个 colorbar 均写明物理量。

当前 E1146 的沿轴杆 {js['axial_shaft']} 质心-轴 Spearman 为 TA {ta['fig1a_centroid_vs_axis_rho']:+.3f}、TB {tb['fig1a_centroid_vs_axis_rho']:+.3f}。显示核固定为 6 mm，只控制画布连续性，不替换冻结分析 kernel。

**关注点**：该图是 raw-EEG-derived envelope timing 在既有冻结间期轴上的 representative cross-check；不是 template-free 验证、cohort 统计、跨未采样组织的 traveling-wave 证明或机制证明。
{gif_block}
### {stem}_metadata.json

记录冻结 fingerprint、exemplar、静态帧、GIF 帧时刻、显示 kernel、色标、producer 与 claim boundary。后续修改图时先核 metadata，再比较 PNG/GIF。

**关注点**：任何后续间期传播 frame 图必须先读 `docs/fig2c_interictal_event_envelope_field_spec.md`，并复用 canonical producer；不得复制 renderer 后另改时间零点、support、平面、sigma 或色标。
"""
    path = figures_dir / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def package_paper_ready(
    ds_sid, fz, ea, eb, sa, sb, js, figures_dir, *, make_gif=True,
    gif_step_ms=GIF_STEP_MS, gif_fps=GIF_FPS,
):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    stem = _paper_stem(ds_sid)
    png = figures_dir / f"{stem}.png"
    pdf = figures_dir / f"{stem}.pdf"
    static_meta = render(
        ds_sid, fz, ea, eb, sa, sb, png, support_mode="participant",
        extra_outputs=(pdf,), dpi=600,
    )
    gif_meta = None
    if make_gif:
        gif_meta = render_gif(
            ds_sid, fz, ea, eb, sa, sb, figures_dir / f"{stem}.gif",
            support_mode="participant", step_ms=gif_step_ms, fps=gif_fps,
        )
    metadata = dict(
        schema_id=PAPER_SCHEMA_ID,
        status="paper-ready Fig2-C candidate; representative subject, not final locked panel",
        ds_sid=ds_sid,
        canonical_producer="scripts/paper_figures/build_main_figures_1_2.py",
        source_producer="scripts/paper_figures/plot_fig2c_interictal_event_envelope_field.py",
        core_renderer="scripts/plot_topic5_interictal_event_envelope_field.py",
        source_metadata=str(OUT / f"{ds_sid}_event_envelope_field.json"),
        frozen_fingerprint=js["frozen_fingerprint"],
        frozen_contract=js["frozen_contract"],
        claim_scope=js["claim_scope"],
        exemplar=js["exemplar"],
        static=static_meta,
        gif=gif_meta,
    )
    meta_path = figures_dir / f"{stem}_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=float),
                         encoding="utf-8")
    readme = _write_paper_ready_readme(figures_dir, ds_sid, js, static_meta, gif_meta)
    print(f"  [paper-meta] {meta_path}", flush=True)
    print(f"  [paper-readme] {readme}", flush=True)
    return dict(
        figures_dir=str(figures_dir), png=str(png), pdf=str(pdf),
        gif=(None if gif_meta is None else gif_meta["figure"]),
        metadata=str(meta_path), readme=str(readme), schema_id=PAPER_SCHEMA_ID,
    )


def _write_cache(ds_sid, fz, ea, eb):
    """Freeze the exact arrays consumed by the field frames and Fig1a readouts."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{ds_sid}_event_envelope_field_cache.npz"
    payload = {
        "contact_order": np.asarray(fz["names"]),
        "points_mm": np.asarray(fz["points_mm"], float),
        "shafts": np.asarray(fz["shafts"]),
        "display_xlim_mm": np.asarray(fz["display_xlim_mm"], float),
        "display_ylim_mm": np.asarray(fz["display_ylim_mm"], float),
        "display_sigma_mm": np.asarray(fz["display_sigma_mm"], float),
        "transverse_sign": np.asarray(fz["transverse_sign"], int),
    }
    for lab, e in (("TA", ea), ("TB", eb)):
        payload.update({
            f"{lab}_event_pos": np.asarray(e["pos"], int),
            f"{lab}_block": np.asarray(e["stem"]),
            f"{lab}_participant": np.asarray(e["part"], bool),
            f"{lab}_envelope_robust_z": np.asarray(e["env_z"], float),
            f"{lab}_envelope_time_from_first_centroid_ms": (
                np.asarray(e["t"], float) - float(e["t0"])
            ) * ief.MS,
            f"{lab}_band_tile": np.asarray(e["band_tile"], float),
            f"{lab}_fig1a_spec_norm": np.asarray(e["spec"], float),
            f"{lab}_fig1a_spec_time_from_first_centroid_ms": np.asarray(e["spec_t_ms"], float),
            f"{lab}_fig1a_spec_freq_hz": np.asarray(e["spec_freq_hz"], float),
            f"{lab}_fig1a_centroid_ms": np.asarray(e["centroid_ms"], float),
            f"{lab}_fig1a_centroid_freq_index": np.asarray(e["centroid_freq_index"], float),
        })
    np.savez_compressed(path, **payload)
    return path


def _write_readme(ds_sid, fz, js):
    """Generate figure notes from the same metadata that populated the canvas."""
    fdir = OUT / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    ta = js["exemplar"]["TA"]
    tb = js["exemplar"]["TB"]
    key = f"shaft_{fz['axial_shaft']}"
    ar = ta["stats"][key]["fig1a_centroid_vs_axis_rho"]
    br = tb["stats"][key]["fig1a_centroid_vs_axis_rho"]
    main = f"{ds_sid}_event_envelope_field.png"
    unit = f"{ds_sid}_event_envelope_field_qc_unit.png"
    template = f"{ds_sid}_event_envelope_field_qc_template.png"
    text = f"""# 间期单事件包络场（冻结 TA/TB shared field）

### {main}

左侧复用 Fig1a 的 Gaussian-smoothed magnitude、逐触点逐事件归一化、主增强区质心和真实 STFT cell 边界，并明确标题为单次 `Sample from TA/TB`；上下两行都写 `time (ms)`，x limits 取两次真实 STFT 窗的共同交集，标题在轴内靠右避开 colorbar，黑色竖线标记 `t=0`。中间两行在完全相同的 shared-plane 物理毫米坐标上显示 participant-only HFO envelope。4 个静态帧由严格等间距、轴向方向门和最终二维场全部参与触点的逐步质心/top-3 热点移动门共同确定，不从渲染像素手挑；每帧再以本帧最强三个参与触点的 robust-z envelope 均值归一化到 1，使幅度较低但已通过可见度门的后帧不会被完整窗 q99 压成近白色。低饱和蓝灰顺序色图固定使用 `PowerNorm(gamma={FIELD_DISPLAY_GAMMA:.2f})`；静态中间场只比较空间集中位置，不比较帧间或 TA/TB 绝对幅度。最右两幅调用冻结群体 TA/TB template-rank field 公共 renderer；`viridis` colorbar 顶部写 `ranks`，显示 artifact 实际 rank 数值并在最低/最高端分别附 early/late，两行 y-label 均简写为 `y (mm)`。沿轴杆 {fz['axial_shaft']} 的 Fig1a 质心-轴 Spearman 为 TA {ar:+.3f}、TB {br:+.3f}。

**关注点**：比较两行质心轨迹与左侧热区移动是否同号相反；不要把单被试两次示例升级成跨二维组织的 traveling wave 证据。

### {unit}

与主图共享几何、时间轴、色标和 6 mm display kernel，但让全部冻结触点进入场平滑，用于检查未参与触点是否改变热区。左侧 readout 仍只显示本次事件的参与触点。

**关注点**：这是 all-contact QC，不作为事件传播主证据。

### {template}

与主图共享几何、时间轴和色标，但以模板 participation support 加权场平滑。该版本会把模板的空间签名重新带回示例事件，只用于检查视觉结果对 support 的敏感性。

**关注点**：这是循环性敏感性图，不作为主结论。
"""
    path = fdir / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def run(
    ds_sid, *, paper_ready_dir=None, make_gif=False,
    gif_step_ms=GIF_STEP_MS, gif_fps=GIF_FPS, ta_event_pos=None, tb_event_pos=None,
):
    dataset, subject = ds_sid.split("_", 1)
    if dataset != "epilepsiae":
        raise NotImplementedError(f"{ds_sid}: only the Epilepsiae raw path is wired")
    fz = load_frozen(ds_sid)
    print(f"[{ds_sid}] frozen OK: {len(fz['names'])} contacts, axial shaft={fz['axial_shaft']}, "
          f"reversed {fz['relation']['line_angle_deg']:.1f}°, "
          f"robust_collinear={fz['boot']['robust_collinear']}, "
          f"shared transverse sign={fz['transverse_sign']:+d}", flush=True)
    ev = load_events(fz, subject)
    inv = _inventory(subject)
    if ta_event_pos is None:
        ea, ma = pick_exemplar(fz, ev, inv, subject, 0, "TA")
    else:
        ea, ma = load_explicit_exemplar(
            fz, ev, inv, subject, ta_event_pos, 0, "TA",
        )
    if tb_event_pos is None:
        eb, mb = pick_exemplar(fz, ev, inv, subject, 1, "TB")
    else:
        eb, mb = load_explicit_exemplar(
            fz, ev, inv, subject, tb_event_pos, 1, "TB",
        )
    sa, sb = event_stats(ea, fz), event_stats(eb, fz)
    for lab, e, st in (("TA", ea, sa), ("TB", eb, sb)):
        a = st[f"shaft_{fz['axial_shaft']}"]
        print(f"  [{lab}] {e['stem']} | packed window {e['packed_window_ms']:.0f} ms, concurrent "
              f"interval {e['event_interval_ms']:.0f} ms | {e['n_usable']}/{e['n_part']} "
              f"participants usable | Fig1a centroid vs axis: "
              f"all={st['all_participants']['fig1a_centroid_vs_axis_rho']:+.3f} "
              f"(n={st['all_participants']['n']}), {fz['axial_shaft']}="
              f"{a['fig1a_centroid_vs_axis_rho']:+.3f} (n={a['n']}) | stored: "
              f"all={st['all_participants']['stored_lag_vs_axis_rho']:+.3f}, "
              f"{fz['axial_shaft']}={a['stored_lag_vs_axis_rho']:+.3f}", flush=True)

    fdir = OUT / "figures"; fdir.mkdir(parents=True, exist_ok=True)
    figs = {m: render(ds_sid, fz, ea, eb, sa, sb,
                      fdir / f"{ds_sid}_event_envelope_field{'' if m == 'participant' else '_qc_' + m}.png",
                      support_mode=m) for m in ("participant", "unit", "template")}
    cache = _write_cache(ds_sid, fz, ea, eb)
    js = dict(ds_sid=ds_sid, band_hz=list(BAND), quantity=ief.ENVELOPE_QUANTITY,
              envelope_fn="return_hil_enve (single band; NOT return_hil_enve_norm)",
              preprocessing=dict(reference="CAR", notch_hz=list(NOTCH_HZ), band_hz=list(BAND)),
              fig1a_readout=dict(
                  algorithm_id=FIG1A_ALGORITHM_ID,
                  display_quantity="Gaussian-smoothed magnitude, normalized per contact per event max",
                  spectrogram=dict(window="hamming", nperseg_sec=0.05, noverlap_sec=0.04,
                                   freq_range_hz=[50.0, 300.0], gaussian_sigma=1.5),
                  centroid_support="8-connected component containing the within-event maximum at >=70%",
                  marker_registration="pcolormesh on real STFT time-cell edges; frequency row center +0.5",
              ),
              detector_event_rule=DET, frozen_fingerprint=fz["fingerprint"],
              frozen_contract="axis/shared plane/contact_order reused; nothing refit",
              display_contract=dict(
                  source="plot_topic5_interictal_template_ab_fields.py",
                  points_unit="mm", display_sigma_mm=fz["display_sigma_mm"],
                  analysis_sigma_mm=fz["analysis_sigma_mm"],
                  shared_transverse_sign=fz["transverse_sign"],
                  xlim_mm=list(fz["display_xlim_mm"]), ylim_mm=list(fz["display_ylim_mm"]),
                  note="display-only 6 mm bandwidth; frozen analysis kernel unchanged",
              ),
              claim_scope="raw-EEG-derived envelope timing cross-check on a previously frozen "
                          "interictal axis; NOT an independent data validation (axis, TA/TB "
                          "labels, exemplar selection, participants and window all come from the "
                          "template pipeline)",
              axial_shaft=fz["axial_shaft"], relation=fz["relation"], pair_bootstrap=fz["boot"],
              exemplar={lab: dict(meta, event_pos=e["pos"], block=e["stem"],
                                  t_in_block_sec=e["t_in_block"],
                                  packed_window_ms=e["packed_window_ms"],
                                  concurrent_interval_ms=e["event_interval_ms"], fs_hz=e["fs"],
                                  n_participating=e["n_part"],
                                  n_with_usable_fig1a_centroid=e["n_usable"],
                                  stored_spread_ms=e["spread_ms"],
                                  fig1a_centroid_vs_stored_rho=e["parity_vs_stored_rho"],
                                  fig1a_centroid_vs_stored_median_abs_dt_ms=(
                                      e["parity_vs_stored_median_abs_dt_ms"]),
                                  fig1a_centroid_alignment=e["fig1a_centroid_alignment"], stats=st)
                        for lab, e, meta, st in (("TA", ea, ma, sa), ("TB", eb, mb, sb))},
              cache=str(cache), figures=figs)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{ds_sid}_event_envelope_field.json"
    json.dump(js, open(p, "w"), indent=2, ensure_ascii=False, default=float)
    readme = _write_readme(ds_sid, fz, js)
    if paper_ready_dir is not None:
        js["paper_ready"] = package_paper_ready(
            ds_sid, fz, ea, eb, sa, sb, js, paper_ready_dir,
            make_gif=make_gif, gif_step_ms=gif_step_ms, gif_fps=gif_fps,
        )
        p.write_text(json.dumps(js, indent=2, ensure_ascii=False, default=float),
                     encoding="utf-8")
    print(f"  [meta] {p}", flush=True)
    print(f"  [cache] {cache}", flush=True)
    print(f"  [readme] {readme}", flush=True)
    return js


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", default="epilepsiae_1146")
    ap.add_argument("--paper-ready-dir", type=Path)
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--gif-step-ms", type=float, default=GIF_STEP_MS)
    ap.add_argument("--gif-fps", type=float, default=GIF_FPS)
    ap.add_argument("--ta-event-pos", type=int)
    ap.add_argument("--tb-event-pos", type=int)
    args = ap.parse_args()
    if args.gif and args.paper_ready_dir is None:
        ap.error("--gif requires --paper-ready-dir")
    run(
        args.subject, paper_ready_dir=args.paper_ready_dir, make_gif=args.gif,
        gif_step_ms=args.gif_step_ms, gif_fps=args.gif_fps,
        ta_event_pos=args.ta_event_pos, tb_event_pos=args.tb_event_pos,
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
