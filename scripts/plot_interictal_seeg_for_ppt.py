"""Render an interictal SEEG raw-trace panel for PPT comparison.

Companion to ``scripts/plot_ictal_er_atlas.py per-seizure``: same channel
selection (PR-1 cluster + top-5 controls picked from the seizure z-ER) and
the same role-based color scheme as
``epilepsiae_548_seizure_12.png``, applied to a clean interictal segment
~2.3 h before that seizure. Used for PPT to contrast interictal background
against the ictal panel.

Usage::

    python scripts/plot_interictal_seeg_for_ppt.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ictal_onset_extraction import (  # noqa: E402
    GAMMA_ER_BANDS,
    compute_er,
    extract_seizure_window,
    resolve_baseline_window,
)
from src.plot_style import FS_LABEL, FS_TITLE, savefig_pub, style_panel  # noqa: E402

# Reuse the canonical archive helpers (role colors, display entries).
ARCHIVE_DIR = ROOT / "scripts" / "archive" / "topic1"
if str(ARCHIVE_DIR) not in sys.path:
    sys.path.insert(0, str(ARCHIVE_DIR))
import sentinel_pr6a_step2 as arch  # noqa: E402

arch._PROJECT_ROOT = ROOT  # match plot_ictal_er_atlas behaviour

SUBJECT = "epilepsiae/548"
SEIZURE_IDX = 12

# Interictal segment locked in for the 548 sentinel:
#   - block 54802102_0019 starts at epoch 1138659194; 17 lagPat events fall
#     in the t=[1400, 1800] s sub-window — moderate density, clearly
#     visible spacing on the raw panel.
#   - center is ~140 min before sz12 clin_onset (1138669228.4) and ~107 min
#     after the previous block boundary; safely outside the pre-ictal zone.
INTERICTAL_BLOCK_STEM = "54802102_0019"
# 50 s zoom [1670, 1720]: same 4 lagPat events (1685.8 / 1690.3 / 1693.0
# close-together burst, then 25 s of quiet, then an isolated 1718.5
# event). No xticks/labels per user request — PPT only.
INTERICTAL_T0_IN_BLOCK = 1670.0
INTERICTAL_T1_IN_BLOCK = 1720.0

# Visualization-only filtering applied AFTER bipolar + notch in
# load_epilepsiae_block:
#   * bandpass [5, 200] Hz: HP=5 cleans drift sharper than HP=1; LP=200
#     drops 200–500 Hz wideband noise but keeps 80–200 Hz ripple HFO
#     energy where most lagPat events live at fs=1024 Hz.
#   * 1-second sliding-median baseline subtraction in the time domain —
#     a hard de-trender that removes any residual slow drift the HP let
#     through.
VIS_BANDPASS_LO_HZ = 5.0
VIS_BANDPASS_HI_HZ = 100.0
VIS_FILTER_ORDER = 4
BASELINE_WINDOW_SEC = 1.0
RAW_TRACE_CLIP_TIGHT = 3.0
SHADING_PAD_SEC = 0.3
SHADING_COLOR = "#f5a142"
SHADING_ALPHA = 0.35

OUT_PATH = (
    ROOT / "results" / "data_driven_soz" / "layer_a_ictal_er_rank"
    / "atlas_v2_3" / "figures" / "per_seizure"
    / "epilepsiae_548_interictal_before_seizure_12.png"
)


def _load_v23_focal(subject: str) -> List[str]:
    import json

    sid = subject.split("/", 1)[1]
    p = (
        ROOT / "results" / "data_driven_soz" / "layer_a_ictal_er_rank"
        / "per_subject" / f"epilepsiae_{sid}.json"
    )
    with open(p, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    return list(d.get("focal_channels") or [])


def _load_block_inventory_row(stem: str) -> Dict:
    import csv

    p = ROOT / "results" / "epilepsiae_block_inventory.csv"
    with open(p, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("block_stem") == stem:
                return r
    raise FileNotFoundError(f"block_stem {stem} not found in inventory")


def _resolve_channel_indices(
    display_entries: List[Dict], target_ch_names: List[str]
) -> List[Dict]:
    """Re-resolve each display entry's signal index against a different block.

    Different recordings can re-order the channel list. We match by name
    (case-insensitive) and drop any display entry that is missing in the
    interictal block.
    """
    name_to_idx = {nm.upper(): i for i, nm in enumerate(target_ch_names)}
    out = []
    for e in display_entries:
        idx = name_to_idx.get(e["channel"].upper())
        if idx is None:
            print(f"  [warn] channel {e['channel']} missing in interictal block; dropped",
                  flush=True)
            continue
        out.append({**e, "idx": int(idx)})
    return out


def _build_bipolar_display_entries(
    bipolar_pairs_named: List,
    bipolar_pair_labels: List[str],
    cluster_rank_by_ch: Dict[str, int],
    focal_set: set,
    n_controls: int = 5,
    signal_for_control_score: np.ndarray | None = None,
) -> List[Dict]:
    """Map seizure-figure CAR display entries onto bipolar pairs.

    Role per pair (any-touch rule):
      * touches high_hi AND focal : high_hi_ictal
      * touches high_hi only      : high_hi_index
      * touches focal only        : ictal
      * otherwise                 : other

    Ranking: cluster pairs sorted by min(cluster rank of in-cluster contact).
    Each cluster channel contributes the *upward* pair (e.g. HL7 -> HL7-HL8)
    when available; falls back to the downward pair if not. We de-dup so
    each pair appears once.
    """
    high_hi_upper = {c.upper() for c in cluster_rank_by_ch}
    focal_upper = {c.upper() for c in focal_set}
    rank_by_upper = {c.upper(): r for c, r in cluster_rank_by_ch.items()
                     if r is not None}

    def _pair_role(a: str, b: str) -> str:
        a_u, b_u = a.upper(), b.upper()
        in_hh = (a_u in high_hi_upper) or (b_u in high_hi_upper)
        in_f = (a_u in focal_upper) or (b_u in focal_upper)
        if in_hh and in_f:
            return "high_hi_ictal"
        if in_hh:
            return "high_hi_index"
        if in_f:
            return "ictal"
        return "other"

    def _pair_min_rank(a: str, b: str):
        ranks = [rank_by_upper[c.upper()]
                 for c in (a, b) if c.upper() in rank_by_upper]
        return min(ranks) if ranks else None

    n_pairs = len(bipolar_pair_labels)
    pair_role = [_pair_role(*bipolar_pairs_named[i]) for i in range(n_pairs)]
    pair_rank = [_pair_min_rank(*bipolar_pairs_named[i]) for i in range(n_pairs)]

    entries: List[Dict] = []

    # 1) Cluster pairs: every pair whose role starts with "high_hi", sorted
    #    by the min cluster rank inside the pair.
    cluster_pair_idx = [i for i in range(n_pairs) if pair_role[i].startswith("high_hi")]
    cluster_pair_idx.sort(key=lambda i: (pair_rank[i] if pair_rank[i] is not None else 1e9,
                                          bipolar_pair_labels[i]))
    for i in cluster_pair_idx:
        entries.append({
            "channel": bipolar_pair_labels[i],
            "idx": i,
            "rank": pair_rank[i],
            "role": pair_role[i],
            "source": "high_hi",
        })

    # 2) Ictal-only pairs (focal touch but no cluster touch).
    ictal_only_idx = [i for i in range(n_pairs) if pair_role[i] == "ictal"]
    ictal_only_idx.sort(key=lambda i: bipolar_pair_labels[i])
    for i in ictal_only_idx:
        entries.append({
            "channel": bipolar_pair_labels[i],
            "idx": i,
            "rank": None,
            "role": "ictal",
            "source": "ictal_only",
        })

    # 3) Top-N control pairs by RMS amplitude (proxy for "informative
    #    background trace") since we no longer have a per-seizure z-ER.
    other_idx = [i for i in range(n_pairs) if pair_role[i] == "other"]
    if other_idx and signal_for_control_score is not None:
        rms = np.array([
            float(np.sqrt(np.mean(signal_for_control_score[i] ** 2)))
            for i in other_idx
        ])
        order = np.argsort(-rms)[:n_controls]
        chosen = [other_idx[k] for k in order]
    else:
        chosen = other_idx[:n_controls]
    for i in chosen:
        entries.append({
            "channel": bipolar_pair_labels[i],
            "idx": i,
            "rank": None,
            "role": "other",
            "source": "control",
        })

    return entries


def _bandpass_visual(signal: np.ndarray, fs: float) -> np.ndarray:
    """Zero-phase bandpass [VIS_BANDPASS_LO, VIS_BANDPASS_HI] Hz for display."""
    nyq = 0.5 * fs
    hi = min(VIS_BANDPASS_HI_HZ, nyq * 0.98)
    sos = butter(
        VIS_FILTER_ORDER,
        [VIS_BANDPASS_LO_HZ / nyq, hi / nyq],
        btype="bandpass", output="sos",
    )
    return sosfiltfilt(sos, signal, axis=-1)


def _subtract_moving_baseline(signal: np.ndarray, fs: float,
                                window_sec: float = BASELINE_WINDOW_SEC
                                ) -> np.ndarray:
    """Subtract the 1-s sliding median per channel.

    Sharper drift removal than a narrow HP filter without smearing HFO
    events: a moving median over 1 s is robust to short bursts (they
    barely move the median over a 1024-sample window) so each event
    survives intact.
    """
    from scipy.ndimage import median_filter
    win = int(round(window_sec * fs))
    if win % 2 == 0:
        win += 1
    out = np.empty_like(signal)
    for i in range(signal.shape[0]):
        med = median_filter(signal[i], size=win, mode="reflect")
        out[i] = signal[i] - med
    return out


def _global_scale(
    signal: np.ndarray, display_idx: List[int],
    quiet_mask: np.ndarray | None,  # kept for signature compat; unused here
) -> float:
    """Pick ONE scale anchored on event amplitudes, shared by every trace.

    99th percentile of |signal| across all displayed channels and time
    samples captures typical event peaks. Dividing by 3 makes those
    peaks land near the ±3 clip; quiet baseline samples (well below the
    99th percentile) display at ~10–20 % of the trace lane and look
    visibly stationary. Channels with no events stay tiny; channels
    with events show clear bursts at the same vertical scale.
    """
    sub = signal[display_idx]
    p99 = float(np.percentile(np.abs(sub), 99))
    if not np.isfinite(p99) or p99 < 1e-9:
        p99 = float(np.percentile(np.abs(sub), 99.9))
    if not np.isfinite(p99) or p99 < 1e-9:
        return 1.0
    return p99 / RAW_TRACE_CLIP_TIGHT


def _scale_with_global(trace: np.ndarray, scale: float) -> np.ndarray:
    """Subtract median, divide by global scale, clip at ±RAW_TRACE_CLIP_TIGHT."""
    med = float(np.median(trace))
    z = (trace - med) / scale
    return np.clip(z, -RAW_TRACE_CLIP_TIGHT, RAW_TRACE_CLIP_TIGHT)


def _plot_raw_panel_no_event_lines(
    ax: plt.Axes,
    *,
    signal: np.ndarray,
    t_axis: np.ndarray,
    display_entries: List[Dict],
    t_min: float,
    t_max: float,
    quiet_mask: np.ndarray | None = None,
) -> None:
    """Mirror ``arch._plot_raw_panel`` but skip clin_onset / baseline lines.

    We can't reuse ``arch._plot_raw_panel`` directly because it always
    draws an axvline at t=0 (clinical onset) and another at the baseline
    edge. Both are meaningless for a free interictal segment, so we
    reproduce the trace-stacking + tick-coloring logic and drop the
    event-line call.
    """
    disp_mask = (t_axis >= t_min) & (t_axis <= t_max)
    if not disp_mask.any():
        raise ValueError("Raw panel display mask is empty")
    t_disp = t_axis[disp_mask]
    dt = float(np.median(np.diff(t_disp))) if t_disp.size > 1 else 0.0
    stride = max(1, int(round(1.0 / (dt * arch.RAW_PLOT_TARGET_HZ)))) if dt > 0 else 1
    t_plot = t_disp[::stride]

    qmask_disp = quiet_mask[disp_mask] if quiet_mask is not None else None
    sub_signal = signal[:, disp_mask]
    global_scale = _global_scale(
        sub_signal, [e["idx"] for e in display_entries], qmask_disp,
    )
    for stack_pos, entry in enumerate(display_entries):
        trace = sub_signal[entry["idx"]]
        trace_scaled = _scale_with_global(trace, global_scale)[::stride]
        offset = stack_pos * arch.RAW_YGAP
        role = entry["role"]
        ax.plot(
            t_plot, trace_scaled + offset,
            color=arch._role_color(role),
            lw=arch._role_linewidth(role, raw=True),
            alpha=arch._role_alpha(role, raw=True),
            zorder=4,
        )

    arch._set_stacked_ticks(ax, display_entries, ygap=arch.RAW_YGAP,
                             ylabel="SEEG (scaled)")
    ax.set_ylim(
        -arch.RAW_YGAP * 0.7,
        (len(display_entries) - 1) * arch.RAW_YGAP + arch.RAW_YGAP * 1.2,
    )
    style_panel(ax)
    ax.set_xlim(t_min, t_max)
    ax.set_xticks([])
    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    for spine in ("top", "bottom"):
        ax.spines[spine].set_visible(False)


def main() -> int:
    # 1. Look up seizure 12 metadata (clin_onset_epoch, seizure_id) for the
    #    title; we don't need the signal because the panel is interictal-only.
    import csv
    sid = SUBJECT.split("/", 1)[1]
    with open(ROOT / "results" / "epilepsiae_seizure_inventory.csv", "r",
              encoding="utf-8") as fh:
        sz_rows = [r for r in csv.DictReader(fh)
                    if r["subject"] == sid and r.get("clin_onset_epoch")]
    sz_rows.sort(key=lambda r: float(r["clin_onset_epoch"]))
    sz12 = sz_rows[SEIZURE_IDX]
    sz12_clin_onset_epoch = float(sz12["clin_onset_epoch"])
    sz12_seizure_id = sz12["seizure_id"]
    fs = None  # not needed; bipolar signal carries its own fs

    focal_set = {c.upper() for c in _load_v23_focal(SUBJECT)}
    lagpat_channels, clusters = arch._load_lagpat(SUBJECT)
    display_cluster = arch._pick_display_cluster(clusters)
    cluster_rank_by_ch = display_cluster["rank_by_channel"]
    print(f"[ref] PR-1 cluster {display_cluster['cluster_id']} channels (rank):")
    for ch, r in sorted(cluster_rank_by_ch.items(), key=lambda x: (x[1] is None, x[1])):
        print(f"  {ch:>6s}  rank={r}")

    # 2. Load the interictal block in bipolar reference.
    blk = _load_block_inventory_row(INTERICTAL_BLOCK_STEM)
    block_start_epoch = float(blk["block_start_epoch"])

    from src.preprocessing import load_epilepsiae_block

    pre = load_epilepsiae_block(
        blk["data_path"], blk["head_path"],
        reference="bipolar", segment_sec=200.0,
    )
    inter_fs = float(pre.sfreq)
    i0 = int(round(INTERICTAL_T0_IN_BLOCK * inter_fs))
    i1 = int(round(INTERICTAL_T1_IN_BLOCK * inter_fs))
    inter_signal = pre.data[:, i0:i1].copy()
    inter_signal = _bandpass_visual(inter_signal, inter_fs)
    inter_signal = _subtract_moving_baseline(inter_signal, inter_fs)
    inter_t = np.arange(inter_signal.shape[1]) / inter_fs + INTERICTAL_T0_IN_BLOCK

    # 3. Build bipolar display entries with per-pair role + cluster ranking.
    pair_labels = list(pre.ch_names)             # ["GA1-GA2", ...]
    bipolar_pairs_named = pre.bipolar_pairs       # [("GA1","GA2"), ...]
    inter_entries = _build_bipolar_display_entries(
        bipolar_pairs_named=bipolar_pairs_named,
        bipolar_pair_labels=pair_labels,
        cluster_rank_by_ch=cluster_rank_by_ch,
        focal_set=focal_set,
        n_controls=5,
        signal_for_control_score=inter_signal,
    )
    print(f"[interictal] bipolar display order ({len(inter_entries)} pairs):")
    for i, e in enumerate(inter_entries):
        rk = e["rank"]
        rk_s = f"r={rk}" if rk is not None else "r=-"
        print(f"  {i:2d}  {e['channel']:>10s}  role={e['role']:>14s}  {rk_s}")

    # 4. Pull in the lagPat event times within this window for top-row markers.
    pt_path = (
        Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
        / SUBJECT.split("/", 1)[1] / "all_recs"
        / f"{INTERICTAL_BLOCK_STEM}_packedTimes_withFreqCent.npy"
    )
    pt = np.load(pt_path)
    in_win = (pt[:, 0] >= INTERICTAL_T0_IN_BLOCK) & (pt[:, 0] <= INTERICTAL_T1_IN_BLOCK)
    event_starts = pt[in_win, 0]
    event_ends = pt[in_win, 1]
    print(f"[interictal] block {INTERICTAL_BLOCK_STEM} window [{INTERICTAL_T0_IN_BLOCK:.0f},"
          f" {INTERICTAL_T1_IN_BLOCK:.0f}] s: {event_starts.size} lagPat events")

    # Quiet mask: True wherever the sample is OUTSIDE any event ±1 s pad.
    # We feed this to the scaler so MAD reflects the true baseline floor;
    # events then float to 5–10 sigma above that floor (instead of being
    # absorbed into a per-trace MAD that mixes baseline + events).
    EVENT_PAD_SEC = 2.0
    quiet_mask = np.ones_like(inter_t, dtype=bool)
    for s, e in zip(event_starts, event_ends):
        in_evt = (inter_t >= s - EVENT_PAD_SEC) & (inter_t <= e + EVENT_PAD_SEC)
        quiet_mask &= ~in_evt
    quiet_frac = float(quiet_mask.mean())
    print(f"[scale] quiet mask covers {quiet_frac:.1%} of the window "
          f"(used for MAD-only baseline normalization)")

    # 5. Render figure (single raw panel; no heatmap, no event lines).
    fig_w = 18.0
    fig_h = 9.0
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    ax = fig.add_axes([0.07, 0.10, 0.86, 0.78])

    _plot_raw_panel_no_event_lines(
        ax,
        signal=inter_signal,
        t_axis=inter_t,
        display_entries=inter_entries,
        t_min=INTERICTAL_T0_IN_BLOCK,
        t_max=INTERICTAL_T1_IN_BLOCK,
        quiet_mask=quiet_mask,
    )

    # lagPat events: prominent shaded vertical bands across all traces.
    for s, e in zip(event_starts, event_ends):
        ax.axvspan(s - SHADING_PAD_SEC, e + SHADING_PAD_SEC,
                    facecolor=SHADING_COLOR, alpha=SHADING_ALPHA,
                    edgecolor="none", zorder=1)

    ax.set_xlabel("")

    # Title: subject, time-relative-to-seizure context, block id, n_events.
    win_center_epoch = block_start_epoch + 0.5 * (
        INTERICTAL_T0_IN_BLOCK + INTERICTAL_T1_IN_BLOCK
    )
    pre_sz_min = (sz12_clin_onset_epoch - win_center_epoch) / 60.0
    title = (
        f"epilepsiae 548  |  interictal segment ≈ {pre_sz_min:.0f} min before "
        f"seizure 12 (id={sz12_seizure_id})  |  bipolar reference\n"
        f"block {INTERICTAL_BLOCK_STEM}  t=[{INTERICTAL_T0_IN_BLOCK:.0f}, "
        f"{INTERICTAL_T1_IN_BLOCK:.0f}] s within block  |  "
        f"{event_starts.size} lagPat events (orange shading)"
    )
    fig.suptitle(title, fontsize=FS_TITLE, y=0.96)

    # Match seizure-figure legend (role colors only — no event lines apply).
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=arch._role_color("high_hi_ictal"), lw=2.0,
                label="High-HI ∩ ictal"),
        Line2D([0], [0], color=arch._role_color("high_hi_index"), lw=2.0,
                label="High-HI index"),
        Line2D([0], [0], color=arch._role_color("ictal"), lw=2.0, label="ictal"),
        Line2D([0], [0], color=arch._role_color("other"), lw=2.0, label="other"),
        mpatches.Patch(facecolor=SHADING_COLOR, alpha=SHADING_ALPHA,
                       edgecolor="none", label="lagPat event"),
    ]
    fig.legend(handles=legend_handles, loc="upper right",
                bbox_to_anchor=(0.99, 0.995), frameon=False, fontsize=11)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    savefig_pub(fig, OUT_PATH)
    print(f"[saved] {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
