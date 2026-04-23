"""PR-6-A Step 2 sentinel driver.

Loads ictal-onset windows for the two PR-6-A sentinel subjects, runs the
ER + baseline z-score primitives from :mod:`src.ictal_onset_extraction`
under both gamma_ER and broad_ER band configurations, and writes
multi-channel z-ER trace figures that the user inspects to confirm SOZ
channels show a clear z-ER rise around clinical onset.

Sentinel cohort (locked in archive plan §3 末尾):
    - sentinel_A: epilepsiae/548  — k=2, in PR-5-A retained main, in
      forward/reverse subset, n_seizures=31 (highest among the 9-subset).
    - sentinel_B: epilepsiae/916  — k=2, in PR-5-A retained main,
      ordinary k=2 (NOT in forward/reverse subset), strong grade,
      n_seizures=51.

Output:
    results/interictal_propagation/ictal_alignment/_sentinel_step2/
        <subject>_<seizure_idx>_<gamma_ER|broad_ER>.png
        sentinel_step2_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as mgs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.ictal_onset_extraction import (
    BROAD_ER_BANDS,
    GAMMA_ER_BANDS,
    baseline_zscore_er,
    compute_er,
    extract_seizure_window,
    resolve_baseline_window,
)
from src.plot_style import (
    COL_SOZ,
    FS_LABEL,
    FS_TICK,
    FS_TITLE,
    savefig_pub,
    style_panel,
)

COL_HHI = "#7E3FA1"              # High-HI index only
COL_HHI_ICTAL = "#C65A6B"        # High-HI ∩ ictal; keep close to ictal red
COL_BG = "#9B9B9B"               # other / control channels
N_BG_TRACES = 5                  # control traces: highest |z-ER| post-clin in display window
TRACE_YGAP = 6.0                 # vertical gap between stacked z-ER traces (z units)
RAW_YGAP = 5.0                   # vertical gap between stacked raw traces
DISPLAY_TMIN = -200.0
DISPLAY_TMAX = 200.0
RAW_TRACE_CLIP = 6.0
RAW_PLOT_TARGET_HZ = 128.0

# Figure margin layout (figure-fraction coordinates).
# Main plotting column (traces + heatmap) lives in [GS_LEFT, GS_RIGHT].
# Legend & colorbar sit in [GS_RIGHT + GAP_W, ...] outside the GridSpec.
GS_LEFT = 0.07
GS_RIGHT = 0.78
GS_TOP = 0.94
GS_BOTTOM = 0.06
SIDE_GAP = 0.012                 # gap between main column right edge and legend/cbar
CBAR_W = 0.012                   # colorbar width in figure fraction


SENTINEL_COHORT = [
    {
        "key": "sentinel_A",
        "subject": "epilepsiae/548",
        "rationale": "k=2, PR-5-A retained main, in 9-subset forward/reverse, n_seizures=31 (max in subset)",
    },
    {
        "key": "sentinel_B",
        "subject": "epilepsiae/916",
        "rationale": "k=2, PR-5-A retained main, NOT in forward/reverse subset, ordinary k=2, n_seizures=51",
    },
]

OUT_DIR = _PROJECT_ROOT / "results" / "interictal_propagation" / "ictal_alignment" / "_sentinel_step2"


def _load_focus_rel() -> dict:
    path = _PROJECT_ROOT / "results" / "epilepsiae_electrode_focus_rel.json"
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _focal_channels(subject: str, focus_rel: dict) -> List[str]:
    sid = subject.split("/", 1)[1]
    rec = focus_rel.get(sid, {})
    return list(rec.get("i", []))


def _load_lagpat(subject: str) -> Tuple[List[str], List[Dict]]:
    """Load PR-1 lag-pattern info for a subject.

    Returns
    -------
    lagpat_channels : list[str]
        Channel names that participated in the PR-1 valid event set
        (i.e. the channels for which a template_rank is defined).
    clusters : list[dict]
        One entry per cluster, sorted by ``n_events`` descending::

            {
                "cluster_id": int,
                "n_events": int,
                "fraction": float,
                "rank_by_channel": {ch_name: rank_int, ...},
            }
    """

    sid = subject.split("/", 1)[1]
    pj_path = (
        _PROJECT_ROOT
        / "results"
        / "interictal_propagation"
        / "per_subject"
        / f"epilepsiae_{sid}.json"
    )
    with open(pj_path, "r", encoding="utf-8") as fh:
        d = json.load(fh)

    lagpat_channels = list(d["channel_names"])
    clusters_raw = d["adaptive_cluster"]["clusters"]
    clusters: List[Dict] = []
    for c in clusters_raw:
        tpl = list(c["template_rank"])
        if len(tpl) != len(lagpat_channels):
            raise ValueError(
                f"{subject}: template_rank length {len(tpl)} != "
                f"len(channel_names) {len(lagpat_channels)}"
            )
        rank_by_channel = {
            ch: (None if r is None or (isinstance(r, float) and np.isnan(r)) else int(r))
            for ch, r in zip(lagpat_channels, tpl)
        }
        clusters.append(
            {
                "cluster_id": int(c["cluster_id"]),
                "n_events": int(c["n_events"]),
                "fraction": float(c["fraction"]),
                "rank_by_channel": rank_by_channel,
            }
        )
    clusters.sort(key=lambda x: -x["n_events"])
    return lagpat_channels, clusters


EVENT_LINE_STYLES = {
    "clin_onset":    {"color": "black",   "lw": 1.4, "ls": "-",  "alpha": 0.85},
    "baseline_edge": {"color": "#555555", "lw": 1.0, "ls": "--", "alpha": 0.7},
    "eeg_onset":     {"color": "#1f4e79", "lw": 1.2, "ls": "-.", "alpha": 0.85},
}


def _draw_event_lines(
    ax: plt.Axes,
    *,
    baseline_edge_sec: float = -60.0,
    eeg_onset_rel_sec: float | None = None,
    draw_zero: bool = False,
) -> None:
    s = EVENT_LINE_STYLES
    ax.axvline(0.0, **s["clin_onset"], zorder=10)
    ax.axvline(baseline_edge_sec, **s["baseline_edge"], zorder=10)
    if eeg_onset_rel_sec is not None and abs(eeg_onset_rel_sec) > 0.5:
        ax.axvline(eeg_onset_rel_sec, **s["eeg_onset"], zorder=10)
    if draw_zero:
        ax.axhline(0, color="black", lw=0.4, alpha=0.25, zorder=1)


def _pick_display_cluster(clusters: List[Dict]) -> Dict:
    """Prefer cluster 1 for display; fall back to the dominant cluster."""

    if not clusters:
        raise ValueError("No propagation clusters available for display")
    for cluster in clusters:
        if int(cluster["cluster_id"]) == 1:
            return cluster
    return clusters[0]


def _channel_role(
    ch_name: str,
    *,
    focal_upper: set[str],
    high_hi_upper: set[str],
) -> str:
    ch_upper = ch_name.upper()
    if ch_upper in high_hi_upper and ch_upper in focal_upper:
        return "high_hi_ictal"
    if ch_upper in high_hi_upper:
        return "high_hi_index"
    if ch_upper in focal_upper:
        return "ictal"
    return "other"


def _role_color(role: str) -> str:
    return {
        "high_hi_ictal": COL_HHI_ICTAL,
        "high_hi_index": COL_HHI,
        "ictal": COL_SOZ,
        "other": COL_BG,
    }[role]


def _role_linewidth(role: str, *, raw: bool) -> float:
    if raw:
        return 1.0 if role.startswith("high_hi") else 0.8
    return 1.35 if role.startswith("high_hi") else 0.9


def _role_alpha(role: str, *, raw: bool) -> float:
    if raw:
        return 0.85 if role.startswith("high_hi") else 0.65
    return 0.95 if role.startswith("high_hi") else 0.7


def _set_stacked_ticks(
    ax: plt.Axes,
    display_entries: List[Dict],
    *,
    ygap: float,
    ylabel: str,
) -> None:
    yticks = [k * ygap for k in range(len(display_entries))]
    ylabels = [entry["channel"] for entry in display_entries]
    ycolors = [_role_color(entry["role"]) for entry in display_entries]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=10)
    for tlbl, col in zip(ax.get_yticklabels(), ycolors):
        tlbl.set_color(col)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)


def _build_display_entries(
    *,
    ch_names: List[str],
    focal_upper: set[str],
    high_hi_upper: set[str],
    cluster: Dict,
    control_idx_top5: np.ndarray,
    valid_mask: np.ndarray,
) -> List[Dict]:
    name_to_idx = {nm.upper(): i for i, nm in enumerate(ch_names)}
    display_entries: List[Dict] = []

    for ch, rank in sorted(
        cluster["rank_by_channel"].items(),
        key=lambda item: (item[1] is None, item[1]),
    ):
        if rank is None:
            continue
        idx = name_to_idx.get(ch.upper())
        if idx is None or not valid_mask[idx]:
            continue
        display_entries.append(
            {
                "channel": ch,
                "idx": idx,
                "rank": int(rank),
                "role": _channel_role(
                    ch,
                    focal_upper=focal_upper,
                    high_hi_upper=high_hi_upper,
                ),
                "source": "high_hi",
            }
        )

    for idx in control_idx_top5:
        if not valid_mask[idx]:
            continue
        ch = ch_names[int(idx)]
        display_entries.append(
            {
                "channel": ch,
                "idx": int(idx),
                "rank": None,
                "role": _channel_role(
                    ch,
                    focal_upper=focal_upper,
                    high_hi_upper=high_hi_upper,
                ),
                "source": "control",
            }
        )

    return display_entries


def _robust_scale_trace(trace: np.ndarray) -> np.ndarray:
    med = float(np.median(trace))
    mad = float(np.median(np.abs(trace - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-9:
        scale = float(np.std(trace))
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    z = (trace - med) / scale
    return np.clip(z, -RAW_TRACE_CLIP, RAW_TRACE_CLIP)


def _plot_raw_panel(
    ax: plt.Axes,
    *,
    signal: np.ndarray,
    t_axis: np.ndarray,
    display_entries: List[Dict],
    eeg_onset_rel_sec: float | None,
    baseline_edge_sec: float = -60.0,
) -> None:
    disp_mask = (t_axis >= DISPLAY_TMIN) & (t_axis <= DISPLAY_TMAX)
    if not disp_mask.any():
        raise ValueError("Raw panel display mask is empty")
    t_disp = t_axis[disp_mask]
    dt = float(np.median(np.diff(t_disp))) if t_disp.size > 1 else 0.0
    stride = 1
    if dt > 0:
        stride = max(1, int(round(1.0 / (dt * RAW_PLOT_TARGET_HZ))))
    t_plot = t_disp[::stride]

    for stack_pos, entry in enumerate(display_entries):
        trace = signal[entry["idx"], disp_mask]
        trace_scaled = _robust_scale_trace(trace)[::stride]
        offset = stack_pos * RAW_YGAP
        role = entry["role"]
        ax.plot(
            t_plot,
            trace_scaled + offset,
            color=_role_color(role),
            lw=_role_linewidth(role, raw=True),
            alpha=_role_alpha(role, raw=True),
            zorder=4,
        )

    _set_stacked_ticks(ax, display_entries, ygap=RAW_YGAP, ylabel="SEEG (scaled)")
    ax.set_ylim(-RAW_YGAP * 0.7, (len(display_entries) - 1) * RAW_YGAP + RAW_YGAP * 1.2)
    _draw_event_lines(
        ax,
        eeg_onset_rel_sec=eeg_onset_rel_sec,
        baseline_edge_sec=baseline_edge_sec,
    )
    style_panel(ax)
    ax.set_xlim(DISPLAY_TMIN, DISPLAY_TMAX)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(100))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(25))


def _plot_trace_panel(
    ax: plt.Axes,
    *,
    z: np.ndarray,
    t_axis_er: np.ndarray,
    cluster: Dict,
    display_entries: List[Dict],
    eeg_onset_rel_sec: float | None,
    baseline_edge_sec: float = -60.0,
) -> Dict:
    """Single High-HI-ranked z-ER panel plus control traces."""

    high_hi_entries = [entry for entry in display_entries if entry["source"] == "high_hi"]
    if not display_entries:
        ax.text(
            0.5,
            0.5,
            "no valid High-HI channels in this seizure window",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=FS_LABEL,
            color="#777",
        )
        style_panel(ax)
        return {
            "display_cluster_id": int(cluster["cluster_id"]),
            "display_cluster_n_events": int(cluster["n_events"]),
            "n_high_hi_channels_in_template": int(
                sum(1 for r in cluster["rank_by_channel"].values() if r is not None)
            ),
            "n_high_hi_channels_valid_in_window": 0,
            "n_control_traces": 0,
            "display_order": [],
        }

    for stack_pos, entry in enumerate(display_entries):
        offset = stack_pos * TRACE_YGAP
        role = entry["role"]
        ax.plot(
            t_axis_er,
            z[entry["idx"]] + offset,
            color=_role_color(role),
            lw=_role_linewidth(role, raw=False),
            alpha=_role_alpha(role, raw=False),
            zorder=5 if entry["source"] == "high_hi" else 2,
        )

    _set_stacked_ticks(
        ax,
        display_entries,
        ygap=TRACE_YGAP,
        ylabel=f"High-HI z-ER (cluster {cluster['cluster_id']})",
    )
    ax.set_ylim(-TRACE_YGAP * 0.7, (len(display_entries) - 1) * TRACE_YGAP + TRACE_YGAP * 1.2)
    _draw_event_lines(
        ax,
        eeg_onset_rel_sec=eeg_onset_rel_sec,
        baseline_edge_sec=baseline_edge_sec,
    )
    style_panel(ax)
    ax.set_xlim(DISPLAY_TMIN, DISPLAY_TMAX)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(100))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(25))

    return {
        "display_cluster_id": int(cluster["cluster_id"]),
        "display_cluster_n_events": int(cluster["n_events"]),
        "display_cluster_fraction": float(cluster["fraction"]),
        "n_high_hi_channels_in_template": int(
            sum(1 for r in cluster["rank_by_channel"].values() if r is not None)
        ),
        "n_high_hi_channels_valid_in_window": len(high_hi_entries),
        "n_control_traces": int(sum(entry["source"] == "control" for entry in display_entries)),
        "display_order": [entry["channel"] for entry in display_entries],
        "high_hi_ictal_in_window": sorted(
            entry["channel"] for entry in high_hi_entries if entry["role"] == "high_hi_ictal"
        ),
        "high_hi_index_in_window": sorted(
            entry["channel"] for entry in high_hi_entries if entry["role"] == "high_hi_index"
        ),
        "ictal_only_controls_in_window": sorted(
            entry["channel"]
            for entry in display_entries
            if entry["source"] == "control" and entry["role"] == "ictal"
        ),
        "other_controls_in_window": sorted(
            entry["channel"]
            for entry in display_entries
            if entry["source"] == "control" and entry["role"] == "other"
        ),
    }


def _select_bg_traces(
    z: np.ndarray,
    t_axis_er: np.ndarray,
    ch_names: List[str],
    high_hi_upper: set[str],
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Non-High-HI channels with largest max |z-ER| over the displayed post-onset window."""

    name_to_idx = {nm.upper(): i for i, nm in enumerate(ch_names)}
    non_high_hi_idx = np.array(
        [i for nm, i in name_to_idx.items() if nm not in high_hi_upper and valid_mask[i]],
        dtype=int,
    )
    if non_high_hi_idx.size == 0:
        return np.array([], dtype=int), ""

    post_mask = (t_axis_er >= 0.0) & (t_axis_er <= DISPLAY_TMAX)
    if not post_mask.any():
        return np.array([], dtype=int), ""

    with np.errstate(invalid="ignore"):
        score = np.nanmax(np.abs(z[non_high_hi_idx][:, post_mask]), axis=1)
    order = np.argsort(-score)
    sel = non_high_hi_idx[order[:N_BG_TRACES]]
    t1 = float(t_axis_er[post_mask][0])
    t2 = float(t_axis_er[post_mask][-1])
    rule = f"max |z-ER| over [{t1:.0f}, {t2:.0f}] s (post clin_onset, displayed window)"
    return sel, rule


def _plot_heatmap_panel(
    ax: plt.Axes,
    *,
    z: np.ndarray,
    t_axis_er: np.ndarray,
    ch_names: List[str],
    focal_upper: set[str],
    high_hi_upper: set[str],
    valid_mask: np.ndarray,
    eeg_onset_rel_sec: float | None,
    baseline_edge_sec: float = -60.0,
):
    """Channels x time heatmap with semantic y-groups."""

    n_ch = z.shape[0]
    post_mask = (t_axis_er >= 0.0) & (t_axis_er <= 30.0)
    post_max = np.full(n_ch, -np.inf, dtype=float)
    if post_mask.any():
        with np.errstate(invalid="ignore"):
            tmp = np.nanmax(z[:, post_mask], axis=1)
        post_max = np.where(valid_mask, tmp, -np.inf)

    is_focal = np.array([nm.upper() in focal_upper for nm in ch_names], dtype=bool)
    is_high_hi = np.array([nm.upper() in high_hi_upper for nm in ch_names], dtype=bool)

    seg_hhi_ictal = np.where(is_high_hi & is_focal & valid_mask)[0]
    seg_hhi_only = np.where(is_high_hi & ~is_focal & valid_mask)[0]
    seg_ictal_only = np.where(~is_high_hi & is_focal & valid_mask)[0]
    seg_other = np.where(~is_high_hi & ~is_focal & valid_mask)[0]

    seg_hhi_ictal = seg_hhi_ictal[np.argsort(-post_max[seg_hhi_ictal])]
    seg_hhi_only = seg_hhi_only[np.argsort(-post_max[seg_hhi_only])]
    seg_ictal_only = seg_ictal_only[np.argsort(-post_max[seg_ictal_only])]
    seg_other = seg_other[np.argsort(-post_max[seg_other])]

    order = np.concatenate([seg_hhi_ictal, seg_hhi_only, seg_ictal_only, seg_other])
    if order.size == 0:
        heat = np.zeros((1, 1), dtype=float)
        extent = [DISPLAY_TMIN, DISPLAY_TMAX, 1, 0]
        vmax = 2.0
    else:
        xmask = (t_axis_er >= DISPLAY_TMIN) & (t_axis_er <= DISPLAY_TMAX)
        heat = z[order][:, xmask]
        extent = [float(t_axis_er[xmask][0]), float(t_axis_er[xmask][-1]), len(order), 0]
        vmax = float(np.nanpercentile(np.abs(heat), 99))
        vmax = max(vmax, 2.0)

    im = ax.imshow(
        heat,
        aspect="auto",
        origin="upper",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=extent,
        interpolation="nearest",
    )

    seg_lens = [len(seg_hhi_ictal), len(seg_hhi_only), len(seg_ictal_only), len(seg_other)]
    boundaries = np.cumsum(seg_lens)
    for b in boundaries[:-1]:
        if b > 0:
            ax.axhline(b, color="black", lw=1.0, alpha=0.85)

    seg_labels = [
        f"High-HI ∩ ictal (n={seg_lens[0]})",
        f"High-HI index (n={seg_lens[1]})",
        f"ictal only (n={seg_lens[2]})",
        f"other (n={seg_lens[3]})",
    ]
    seg_colors = [COL_HHI_ICTAL, COL_HHI, COL_SOZ, COL_BG]
    starts = np.concatenate([[0], boundaries[:-1]])
    label_y = starts + np.array(seg_lens) / 2.0
    keep = np.array(seg_lens) > 0
    kept_labels = [lbl for lbl, k in zip(seg_labels, keep) if k]
    kept_colors = [col for col, k in zip(seg_colors, keep) if k]
    ax.set_yticks(label_y[keep])
    ax.set_yticklabels(kept_labels, fontsize=10)
    for tlbl, col in zip(ax.get_yticklabels(), kept_colors):
        tlbl.set_color(col)
    ax.yaxis.set_ticks_position("left")
    ax.yaxis.set_tick_params(left=True, labelleft=True, right=False, labelright=False)
    ax.set_xlabel("time relative to clin_onset (s)", fontsize=FS_LABEL)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(100))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(25))
    style_panel(ax)
    _draw_event_lines(
        ax,
        eeg_onset_rel_sec=eeg_onset_rel_sec,
        baseline_edge_sec=baseline_edge_sec,
    )
    ax.set_xlim(DISPLAY_TMIN, DISPLAY_TMAX)

    return (
        {
            "heatmap_segments": {
                "high_hi_ictal": int(seg_lens[0]),
                "high_hi_index": int(seg_lens[1]),
                "ictal_only": int(seg_lens[2]),
                "other": int(seg_lens[3]),
            },
            "heatmap_vmax": float(vmax),
        },
        im,
    )


def _band_max(z: np.ndarray, idx_list, mask: np.ndarray) -> float:
    idx_list = list(idx_list)
    if not idx_list:
        return float("nan")
    sub = z[np.ix_(idx_list, mask)]
    if sub.size == 0:
        return float("nan")
    return float(np.nanmedian(np.nanmax(sub, axis=1)))


def _plot_zER_panel(
    z: np.ndarray,
    t_axis_er: np.ndarray,
    raw_signal: np.ndarray,
    t_axis_raw: np.ndarray,
    ch_names: List[str],
    focal_set: set[str],
    lagpat_channels: List[str],
    clusters: List[Dict],
    title: str,
    outfile: Path,
    band_label: str,
    eeg_onset_rel_sec: float | None = None,
    baseline_edge_sec: float = -60.0,
) -> Dict:
    """Raw + single High-HI z-ER panel + full-channel heatmap.

    Returns numerical summary written to sentinel_step2_summary.json.
    """
    valid_mask = ~np.isnan(z).any(axis=1)
    n_ch = z.shape[0]
    focal_upper = {c.upper() for c in focal_set}
    high_hi_upper = {c.upper() for c in lagpat_channels}
    display_cluster = _pick_display_cluster(clusters)

    bg_idx_top5, bg_select_rule = _select_bg_traces(
        z, t_axis_er, ch_names, high_hi_upper, valid_mask,
    )
    control_ch_names = [ch_names[i] for i in bg_idx_top5]
    display_entries = _build_display_entries(
        ch_names=ch_names,
        focal_upper=focal_upper,
        high_hi_upper=high_hi_upper,
        cluster=display_cluster,
        control_idx_top5=bg_idx_top5,
        valid_mask=valid_mask,
    )

    fig_h = 13.4
    fig_w = 13.0
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    height_ratios = [5.2, 5.0, 6.2]
    gs = mgs.GridSpec(
        3, 1,
        figure=fig,
        height_ratios=height_ratios,
        left=GS_LEFT, right=GS_RIGHT,
        top=GS_TOP, bottom=GS_BOTTOM,
        hspace=0.22,
    )

    ax_raw = fig.add_subplot(gs[0, 0])
    ax_er = fig.add_subplot(gs[1, 0], sharex=ax_raw)
    ax_heat = fig.add_subplot(gs[2, 0], sharex=ax_raw)

    _plot_raw_panel(
        ax_raw,
        signal=raw_signal,
        t_axis=t_axis_raw,
        display_entries=display_entries,
        eeg_onset_rel_sec=eeg_onset_rel_sec,
        baseline_edge_sec=baseline_edge_sec,
    )
    plt.setp(ax_raw.get_xticklabels(), visible=False)

    er_panel_info = _plot_trace_panel(
        ax_er,
        z=z,
        t_axis_er=t_axis_er,
        cluster=display_cluster,
        display_entries=display_entries,
        eeg_onset_rel_sec=eeg_onset_rel_sec,
        baseline_edge_sec=baseline_edge_sec,
    )
    plt.setp(ax_er.get_xticklabels(), visible=False)

    heat_info, im = _plot_heatmap_panel(
        ax_heat,
        z=z, t_axis_er=t_axis_er, ch_names=ch_names,
        focal_upper=focal_upper, high_hi_upper=high_hi_upper,
        valid_mask=valid_mask,
        eeg_onset_rel_sec=eeg_onset_rel_sec,
        baseline_edge_sec=baseline_edge_sec,
    )

    heat_pos = ax_heat.get_position()
    cax = fig.add_axes([
        GS_RIGHT + SIDE_GAP, heat_pos.y0, CBAR_W, heat_pos.height,
    ])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("z-ER", fontsize=FS_LABEL)
    cbar.ax.tick_params(labelsize=FS_TICK - 2)

    post_mask = (t_axis_er >= 0.0) & (t_axis_er <= 30.0)
    pre_mask = (t_axis_er >= -30.0) & (t_axis_er < 0.0)
    focal_idx = [i for i in range(n_ch)
                 if ch_names[i].upper() in focal_upper and valid_mask[i]]
    nonfocal_idx = [i for i in range(n_ch)
                    if ch_names[i].upper() not in focal_upper and valid_mask[i]]
    high_hi_idx = [i for i in range(n_ch)
                   if ch_names[i].upper() in high_hi_upper and valid_mask[i]]
    high_hi_ictal_idx = [i for i in high_hi_idx if ch_names[i].upper() in focal_upper]
    high_hi_only_idx = [i for i in high_hi_idx if ch_names[i].upper() not in focal_upper]

    summary = {
        "n_channels_total": int(n_ch),
        "n_channels_valid": int(valid_mask.sum()),
        "n_focal_valid": len(focal_idx),
        "n_nonfocal_valid": len(nonfocal_idx),
        "n_high_hi_valid": len(high_hi_idx),
        "n_high_hi_ictal_valid": len(high_hi_ictal_idx),
        "n_high_hi_index_valid": len(high_hi_only_idx),
        "focal_zER_pre30s_max_median": _band_max(z, focal_idx, pre_mask),
        "focal_zER_post30s_max_median": _band_max(z, focal_idx, post_mask),
        "nonfocal_zER_pre30s_max_median": _band_max(z, nonfocal_idx, pre_mask),
        "nonfocal_zER_post30s_max_median": _band_max(z, nonfocal_idx, post_mask),
        "high_hi_zER_pre30s_max_median": _band_max(z, high_hi_idx, pre_mask),
        "high_hi_zER_post30s_max_median": _band_max(z, high_hi_idx, post_mask),
        "high_hi_ictal_zER_post30s_max_median": _band_max(z, high_hi_ictal_idx, post_mask),
        "high_hi_index_zER_post30s_max_median": _band_max(z, high_hi_only_idx, post_mask),
        "control_traces": control_ch_names,
        "control_selection_rule": bg_select_rule,
        "er_panel": er_panel_info,
        "heatmap": heat_info,
        "display_window_sec": [DISPLAY_TMIN, DISPLAY_TMAX],
        "band": band_label,
    }

    from matplotlib.lines import Line2D
    role_presence = {
        "high_hi_ictal": bool(high_hi_ictal_idx),
        "high_hi_index": bool(high_hi_only_idx),
        "ictal": bool([i for i in focal_idx if ch_names[i].upper() not in high_hi_upper]),
        "other": bool([i for i in nonfocal_idx if ch_names[i].upper() not in high_hi_upper]),
    }
    role_labels = {
        "high_hi_ictal": "High-HI ∩ ictal",
        "high_hi_index": "High-HI index",
        "ictal": "ictal",
        "other": "other",
    }
    legend_handles = [
        Line2D([0], [0], color=_role_color(role), lw=2.0, label=role_labels[role])
        for role, present in role_presence.items()
        if present
    ]
    legend_handles.extend([
        Line2D([0], [0], **EVENT_LINE_STYLES["clin_onset"],
               label="clin_onset (t=0)"),
        Line2D([0], [0], **EVENT_LINE_STYLES["baseline_edge"],
               label=f"baseline edge ({baseline_edge_sec:+.0f} s)"),
    ])
    if eeg_onset_rel_sec is not None and abs(eeg_onset_rel_sec) > 0.5:
        legend_handles.append(
            Line2D([0], [0], **EVENT_LINE_STYLES["eeg_onset"],
                   label=f"eeg_onset (Δ={eeg_onset_rel_sec:+.1f}s)")
        )
    ax0_pos = ax_raw.get_position()
    fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(GS_RIGHT + SIDE_GAP, ax0_pos.y1),
        bbox_transform=fig.transFigure,
        frameon=False,
        ncol=1,
        fontsize=FS_TICK - 3,
        handlelength=2.0,
        labelspacing=0.9,
        borderaxespad=0.0,
    )

    fig.suptitle(title, fontsize=FS_TITLE, y=GS_TOP + 0.025)
    savefig_pub(fig, outfile, dpi=180)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seizures", type=int, default=2,
                        help="Per-sentinel number of seizures to plot (default 2)")
    parser.add_argument("--seizure-indices", type=int, nargs="*", default=None,
                        help="Override: explicit zero-based seizure indices (applied to both sentinels)")
    parser.add_argument("--max-attempts", type=int, default=20,
                        help="Skip seizures whose pre/post window crosses block boundary; try up to this many")
    parser.add_argument("--pre-sec", type=float, default=300.0)
    parser.add_argument("--post-sec", type=float, default=300.0)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    focus_rel = _load_focus_rel()

    summary = {
        "step": "PR-6-A Step 2 sentinel ER + baseline z-score visual inspection",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sentinels": [],
    }

    bands = (GAMMA_ER_BANDS, BROAD_ER_BANDS)

    for sentinel in SENTINEL_COHORT:
        subj = sentinel["subject"]
        focal_set = set(_focal_channels(subj, focus_rel))
        lagpat_channels, clusters = _load_lagpat(subj)
        sentinel_log = {
            "key": sentinel["key"],
            "subject": subj,
            "rationale": sentinel["rationale"],
            "focal_channels": sorted(focal_set),
            "lagpat_channels": list(lagpat_channels),
            "clusters_used": [
                {
                    "cluster_id": c["cluster_id"],
                    "n_events": c["n_events"],
                    "fraction": c["fraction"],
                }
                for c in clusters
            ],
            "seizures": [],
        }
        print(
            f"\n=== {sentinel['key']}  {subj}  "
            f"focal(i)={len(focal_set)}  lagpat={len(lagpat_channels)}  "
            f"clusters={len(clusters)} ==="
        )

        if args.seizure_indices is not None:
            candidate_idx = list(args.seizure_indices)
        else:
            candidate_idx = list(range(args.max_attempts))

        ok_seizures = 0
        for seizure_idx in candidate_idx:
            if ok_seizures >= args.n_seizures and args.seizure_indices is None:
                break
            try:
                window = extract_seizure_window(
                    subj, seizure_idx,
                    pre_sec=args.pre_sec, post_sec=args.post_sec,
                    results_root=_PROJECT_ROOT / "results",
                    reference="car",
                )
            except (ValueError, IndexError) as exc:
                print(f"  seizure {seizure_idx}: SKIP ({exc})")
                continue

            print(f"  seizure {seizure_idx}: block={window.block_stem} "
                  f"fs={window.fs} n_ch={window.signal.shape[0]} "
                  f"n_samples={window.signal.shape[1]}")

            eeg_onset_rel_sec: float | None = None
            if window.eeg_onset_epoch is not None and window.clin_onset_epoch is not None:
                eeg_onset_rel_sec = float(
                    window.eeg_onset_epoch - window.clin_onset_epoch
                )

            seizure_record = {
                "seizure_idx": seizure_idx,
                "seizure_id": window.seizure_id,
                "block_stem": window.block_stem,
                "clin_onset_epoch": window.clin_onset_epoch,
                "eeg_onset_epoch": window.eeg_onset_epoch,
                "eeg_onset_rel_sec": eeg_onset_rel_sec,
                "n_channels": window.signal.shape[0],
                "fs": window.fs,
                "bands": {},
            }

            for band in bands:
                er = compute_er(
                    window.signal, fs=window.fs,
                    fast_band=band["fast"], slow_band=band["slow"],
                    win_sec=1.0, hop_sec=0.1,
                )
                hop_sec = 0.1
                win_sec = 1.0
                bl_win = resolve_baseline_window(
                    er.shape[1],
                    hop_sec=hop_sec,
                    pre_sec=window.pre_sec,
                    buffer_sec=60.0,
                    eeg_onset_rel_sec=eeg_onset_rel_sec,
                )

                baseline_diag = {
                    "baseline_start_rel_sec": bl_win.start_sec,
                    "baseline_end_rel_sec": bl_win.end_sec,
                    "baseline_valid_sec": bl_win.valid_sec,
                    "baseline_clipped_by_eeg_onset": bl_win.clipped_by_eeg_onset,
                    "eeg_onset_rel_sec": eeg_onset_rel_sec,
                    "baseline_valid": bl_win.valid,
                }

                if not bl_win.valid:
                    print(
                        f"    {band['key']}: BASELINE-INVALID "
                        f"(valid_sec={bl_win.valid_sec:.1f}s, "
                        f"clipped_by_eeg_onset={bl_win.clipped_by_eeg_onset}); "
                        f"skipping plot for this band"
                    )
                    seizure_record["bands"][band["key"]] = {
                        "band": band["key"],
                        "skipped": True,
                        "skip_reason": "baseline_invalid",
                        **baseline_diag,
                    }
                    continue

                z = baseline_zscore_er(
                    er,
                    baseline_idx_window=(bl_win.start_idx, bl_win.end_idx),
                    hop_sec=hop_sec,
                )

                t_axis_er = (np.arange(er.shape[1]) * hop_sec + win_sec / 2.0) - window.pre_sec

                title = f"{subj}  |  seizure_id={window.seizure_id}  |  band={band['key']}"
                sid = subj.split("/", 1)[1]
                outfile = OUT_DIR / f"epilepsiae_{sid}_{seizure_idx:02d}_{band['key']}.png"
                stats = _plot_zER_panel(
                    z, t_axis_er, window.signal, window.t_axis, list(window.ch_names),
                    focal_set, lagpat_channels, clusters,
                    title, outfile, band["key"],
                    eeg_onset_rel_sec=eeg_onset_rel_sec,
                    baseline_edge_sec=bl_win.end_sec,
                )
                stats["png"] = str(outfile.relative_to(_PROJECT_ROOT))
                stats.update(baseline_diag)
                seizure_record["bands"][band["key"]] = stats
                clip_tag = "(clipped)" if bl_win.clipped_by_eeg_onset else ""
                print(
                    f"    {band['key']}: focal post30s={stats['focal_zER_post30s_max_median']:.2f}  "
                    f"vs pre30s={stats['focal_zER_pre30s_max_median']:.2f}  "
                    f"baseline=[{bl_win.start_sec:.0f},{bl_win.end_sec:.0f}]s {clip_tag} -> {outfile.name}"
                )

            sentinel_log["seizures"].append(seizure_record)
            ok_seizures += 1

        summary["sentinels"].append(sentinel_log)

    summary_path = OUT_DIR / "sentinel_step2_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"\nSummary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
