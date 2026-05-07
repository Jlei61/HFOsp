"""Ictal ER-onset Timing Atlas (Layer A v2.3).

Renders two figure types per subject for visual diagnosis of failure
modes in ER-rank as a SOZ proxy:

- ``per-subject``: channel x seizure t_ER_onset matrix, dual-band stacked
  (gamma_ER on top, broad_ER on bottom). Cool = pre-clinical onset
  (t < 0), warm = post-clinical, gray = CUSUM not triggered. Channels
  sorted by sort_band r_sz ascending.
- ``per-seizure``: dual-band side-by-side (left gamma / right broad),
  each column = raw SEEG row + full-channel z-ER heatmap row, the
  heatmap overlaid with per-row t_ER_onset markers and top-5 trace
  overlays. Time window [-120, +30]s relative to clinical onset.

CLI subcommands::

  per-seizure --subject epilepsiae/548 --seizure-idx 0
  per-subject --subject epilepsiae/548 [--include-seizures]
  cohort                              [--include-seizures]

Spec: ``docs/superpowers/specs/2026-05-08-ictal-er-atlas-design.md``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as mgs
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plot_style import FS_LABEL, FS_TICK, FS_TITLE, savefig_pub  # noqa: E402

# ---------------------------------------------------------------------------
# Schema / paths

REQUIRED_SCHEMA = "pr_t3_1_layer_a_v2_3_timing"
LAYER_A_DIR = ROOT / "results" / "data_driven_soz" / "layer_a_ictal_er_rank"
PER_SUBJECT_DIR = LAYER_A_DIR / "per_subject"
SENTINEL_DIR = LAYER_A_DIR / "_sentinel"
ATLAS_OUT_DIR = LAYER_A_DIR / "atlas_v2_3" / "figures"
PER_SUBJECT_OUT_DIR = ATLAS_OUT_DIR / "per_subject"
PER_SEIZURE_OUT_DIR = ATLAS_OUT_DIR / "per_seizure"

DETECTION_WINDOW_SEC = (-120.0, 30.0)   # heatmap display window (matches v2.3 detection)
CMAP_HEATMAP = "RdBu_r"                  # diverging; midpoint at 0 = clinical onset

# Tick coloring (spec §4.3 / §5.3)
COL_TICK_SOZ = "#c0392b"
COL_TICK_HIGHHI = "#34495e"
COL_TICK_OTHER = "#95a5a6"

# Status strip palette (spec §5.3)
STATUS_COLORS = {
    "ok": "#2ecc71",
    "onset_unreached": "#d5d5d5",
    "onset_tied": "#8e44ad",
    "baseline_invalid": "#1a1a1a",
    "not_loaded": "#ffffff",
    "boundary_skip": "#3498db",
}

# ---------------------------------------------------------------------------
# Data loading


def _load_per_subject_json(subject: str, *, source: str = "per_subject",
                            schema_required: bool = True) -> Dict:
    """Load v2.3 per-subject JSON. ``source`` ∈ {per_subject, _sentinel}."""
    sid = subject.replace("/", "_")
    if source == "per_subject":
        path = PER_SUBJECT_DIR / f"{sid}.json"
    elif source == "_sentinel":
        path = SENTINEL_DIR / f"{sid}.json"
    else:
        raise ValueError(f"unknown source={source}")
    if not path.exists():
        raise FileNotFoundError(f"per-subject JSON missing: {path}")

    with open(path, "r") as fh:
        d = json.load(fh)
    if schema_required:
        sv = d.get("schema_version")
        if sv != REQUIRED_SCHEMA:
            raise ValueError(
                f"{path} schema_version={sv!r}, expected {REQUIRED_SCHEMA!r}. "
                f"v2.2 JSON is not consumable by atlas_v2_3 — backup is in "
                f"per_subject_v2_2/."
            )
    return d


def _list_cohort_subjects(*, source: str = "per_subject") -> List[str]:
    """All subjects with a v2.3 per-subject JSON."""
    src_dir = PER_SUBJECT_DIR if source == "per_subject" else SENTINEL_DIR
    if not src_dir.exists():
        return []
    out: List[str] = []
    for p in sorted(src_dir.glob("*.json")):
        if p.name in {"cohort_summary.json", "sanity_report.json"}:
            continue
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if d.get("schema_version") == REQUIRED_SCHEMA:
            out.append(d["subject"])
    return out


# ---------------------------------------------------------------------------
# Per-subject matrix construction


def _seizure_idx_in_order(per_er_record: Dict) -> List[int]:
    return [int(r["seizure_idx"]) for r in per_er_record.get("seizure_records", [])]


def _build_onset_matrix(
    per_er_record: Dict,
    channels: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (onset_matrix [n_ch, n_sz], status_array [n_sz], seizure_ids).

    onset_matrix cells: t_onset_sec (float) or NaN (CUSUM not triggered /
    seizure status != ok).
    """
    sz_records = per_er_record.get("seizure_records", [])
    n_sz = len(sz_records)
    onset = np.full((len(channels), n_sz), np.nan, dtype=np.float64)
    statuses: List[str] = []
    seizure_ids: List[str] = []
    for j, rec in enumerate(sz_records):
        statuses.append(rec.get("status", "unknown"))
        seizure_ids.append(rec.get("seizure_id", str(rec.get("seizure_idx", j))))
        co = rec.get("channel_onsets") or {}
        for i, ch in enumerate(channels):
            entry = co.get(ch)
            if not entry:
                continue
            t = entry.get("t_onset_sec")
            if t is not None and np.isfinite(t):
                onset[i, j] = float(t)
    return onset, np.array(statuses), seizure_ids


def _select_sort_band(per_subject: Dict) -> str:
    """Spec §5.2 sort_band rule: stable > moderate > unstable > insufficient."""
    ph = per_subject.get("producer_health", {})
    rank = {"stable": 3, "moderate": 2, "unstable": 1, "insufficient": 0}
    g = rank.get(ph.get("gamma_ER", "insufficient"), 0)
    b = rank.get(ph.get("broad_ER", "insufficient"), 0)
    if g >= b:
        return "gamma_ER"
    return "broad_ER"


def _sort_band_unreliable(per_subject: Dict) -> bool:
    ph = per_subject.get("producer_health", {})
    g = ph.get("gamma_ER", "insufficient")
    b = ph.get("broad_ER", "insufficient")
    return g in {"unstable", "insufficient"} and b in {"unstable", "insufficient"}


def _channel_order(per_subject: Dict, sort_band: str) -> Tuple[List[str], List[str]]:
    """Sort channels by sort_band r_sz ascending; r_sz=None at end.

    Returns (sorted_channels, [r_sz_repr]).
    """
    rec = per_subject.get("per_er", {}).get(sort_band, {})
    rsz = rec.get("r_sz", {})
    chs = list(rsz.keys())
    finite = [(c, rsz[c]) for c in chs if rsz[c] is not None]
    nonfinite = [c for c in chs if rsz[c] is None]
    finite.sort(key=lambda x: x[1])
    return [c for c, _ in finite] + nonfinite, [str(rsz[c]) for c in chs]


def _channel_role(channel: str, focal_set: set) -> str:
    """Per spec §5.3. High-HI tagging requires LagPat input, deferred."""
    return "soz" if channel in focal_set else "other"


def _channel_tick_color(role: str) -> str:
    return {
        "soz": COL_TICK_SOZ,
        "high_hi": COL_TICK_HIGHHI,
    }.get(role, COL_TICK_OTHER)


def _row_order_per_seizure(
    ch_names: Sequence[str],
    focal_set: set,
    onsets: Dict[str, Optional[float]],
) -> List[int]:
    """Heatmap row order for per-seizure figure (spec §4.1).

    Tier 1: SOZ (focal_set membership), Tier 2: non-SOZ.
    Within each tier: sorted by t_ER_onset asc; NaN/None at the end of
    its tier. Returns a permutation of ``range(len(ch_names))``.
    """
    big = float("inf")
    rows = []
    for i, ch in enumerate(ch_names):
        tier = 0 if ch in focal_set else 1
        t = onsets.get(ch)
        t_key = t if (t is not None and np.isfinite(t)) else big
        rows.append((tier, t_key, i))
    rows.sort()
    return [r[2] for r in rows]


# ---------------------------------------------------------------------------
# Per-subject summary figure


def _draw_onset_matrix_band(
    ax: plt.Axes,
    onset: np.ndarray,
    statuses: np.ndarray,
    *,
    band_label: str,
    show_x_ticks: bool = True,
) -> matplotlib.image.AxesImage:
    """Draw one band's t_onset matrix (channels x seizures)."""
    n_ch, n_sz = onset.shape
    norm = TwoSlopeNorm(vmin=DETECTION_WINDOW_SEC[0],
                        vcenter=0.0,
                        vmax=DETECTION_WINDOW_SEC[1])
    im = ax.imshow(onset, aspect="auto", origin="upper", cmap=CMAP_HEATMAP,
                    norm=norm, interpolation="nearest")

    # Gray fill for NaN (= CUSUM not triggered or seizure non-ok)
    nan_mask = np.isnan(onset)
    if nan_mask.any():
        gray = np.zeros((*onset.shape, 4))
        gray[nan_mask] = (0.835, 0.835, 0.835, 1.0)
        ax.imshow(gray, aspect="auto", origin="upper", interpolation="nearest")

    # Hatching for non-ok seizures (whole column)
    for j, st in enumerate(statuses):
        if st != "ok":
            ax.add_patch(mpatches.Rectangle(
                (j - 0.5, -0.5), 1.0, n_ch,
                fill=False, hatch="///", edgecolor="black",
                linewidth=0, alpha=0.35,
            ))

    ax.set_xlim(-0.5, n_sz - 0.5)
    ax.set_ylim(n_ch - 0.5, -0.5)
    ax.set_ylabel(band_label, fontsize=FS_LABEL)
    if not show_x_ticks:
        # sharex prevents removing ticks; hide labels instead so the
        # status strip below is the only x-axis labelled.
        plt.setp(ax.get_xticklabels(), visible=False)
    return im


def _draw_status_strip(ax: plt.Axes, statuses: np.ndarray) -> None:
    n_sz = len(statuses)
    for j, st in enumerate(statuses):
        col = STATUS_COLORS.get(st, "#cccccc")
        ax.add_patch(mpatches.Rectangle(
            (j - 0.5, 0), 1.0, 1.0,
            facecolor=col, edgecolor="white", linewidth=0.4,
        ))
    ax.set_xlim(-0.5, n_sz - 0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks(np.arange(n_sz))
    ax.set_xticklabels([str(j) for j in range(n_sz)], fontsize=FS_TICK - 4)
    ax.set_xlabel("seizure_idx", fontsize=FS_LABEL)


def _draw_cov_bar(
    ax: plt.Axes,
    channels: Sequence[str],
    rsz_valid_count: Dict[str, int],
    n_ok: int,
    *,
    band_label: str,
) -> None:
    n_ch = len(channels)
    fractions = []
    for ch in channels:
        c = rsz_valid_count.get(ch, 0)
        fractions.append(c / n_ok if n_ok > 0 else 0.0)
    ax.barh(np.arange(n_ch), fractions, color="#27ae60", height=0.85)
    ax.set_xlim(0, 1)
    ax.set_ylim(n_ch - 0.5, -0.5)
    # sharey with heatmap means we can't permanently empty the y-ticks;
    # explicitly suppress the labels on cov bar's left side.
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_xticks([0, 1.0])
    ax.set_xticklabels(["0", "1"], fontsize=FS_TICK - 5)
    ax.tick_params(axis="x", pad=1)
    ax.set_xlabel(f"cov/{n_ok}", fontsize=FS_TICK - 4, labelpad=1)


def _format_subject_title(per_subject: Dict, sort_band: str) -> str:
    subj = per_subject["subject"]
    n_sz_total = per_subject.get("n_seizures_total", 0)
    parts = [f"{subj}", f"n_sz_total={n_sz_total}"]
    for er_key in ("gamma_ER", "broad_ER"):
        rec = per_subject.get("per_er", {}).get(er_key, {})
        n_ok = rec.get("n_seizures_ok", 0)
        n_ur = rec.get("n_seizures_onset_unreached", 0)
        s_sz = rec.get("s_sz")
        s_sz_repr = f"{s_sz:.2f}" if isinstance(s_sz, (int, float)) and s_sz is not None else "—"
        ph = per_subject.get("producer_health", {}).get(er_key, "?")
        cc = per_subject.get("clinical_concordance", {}).get(er_key, "?")
        parts.append(
            f"{er_key.split('_')[0]}: ok={n_ok} ur={n_ur} s_sz={s_sz_repr}"
            f" ph={ph} cc={cc}"
        )
    suffix = f"sort_band={sort_band}"
    if _sort_band_unreliable(per_subject):
        suffix += " (sort_band unreliable — both bands unstable/insufficient)"
    return "  |  ".join(parts) + "\n" + suffix


def render_per_subject(per_subject: Dict, out_path: Path) -> Path:
    """Render the channels x seizures dual-band t_onset matrix figure."""
    sort_band = _select_sort_band(per_subject)
    channels, _rsz_repr = _channel_order(per_subject, sort_band)
    focal_set = set(per_subject.get("focal_channels") or [])

    n_ch = len(channels)
    n_sz = per_subject.get("n_seizures_total", 0) or 0
    # actual n_sz comes from seizure_records (may differ from n_total if some were skipped)
    seizure_records = per_subject["per_er"].get(sort_band, {}).get("seizure_records", [])
    n_sz = max(n_sz, len(seizure_records))

    # Spec §5.4: scale to fit [12,30] width and [8,30] height.
    fig_w = max(14.0, 8.0 + 0.35 * n_sz)
    fig_w = min(fig_w, 30.0)
    fig_h = max(9.0, 0.20 * n_ch * 2 + 4)
    fig_h = min(fig_h, 30.0)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = mgs.GridSpec(
        nrows=3, ncols=3,
        figure=fig,
        height_ratios=[1.0, 1.0, 0.05],
        width_ratios=[0.02, 1.0, 0.07],
        left=0.07, right=0.95, top=0.90, bottom=0.10,
        hspace=0.18, wspace=0.025,
    )

    ax_g = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, 1], sharex=ax_g, sharey=ax_g)
    ax_status = fig.add_subplot(gs[2, 1], sharex=ax_g)
    ax_cov_g = fig.add_subplot(gs[0, 2], sharey=ax_g)
    ax_cov_b = fig.add_subplot(gs[1, 2], sharey=ax_g)

    # --- gamma band ---
    g_rec = per_subject["per_er"].get("gamma_ER", {})
    g_onset, g_statuses, _ = _build_onset_matrix(g_rec, channels)
    if g_onset.shape[1] == 0:
        # n_ok=0 fallback per spec §5.4 boundary case
        g_onset = np.full((n_ch, max(n_sz, 1)), np.nan)
        g_statuses = np.array(["onset_unreached"] * g_onset.shape[1])
    im_g = _draw_onset_matrix_band(ax_g, g_onset, g_statuses,
                                    band_label="gamma_ER", show_x_ticks=False)
    _g_n_ok = g_rec.get("n_seizures_ok", 0)
    _draw_cov_bar(ax_cov_g, channels, g_rec.get("r_sz_valid_count", {}),
                  _g_n_ok, band_label="gamma_ER")
    if _g_n_ok == 0:
        ax_g.set_title("GAMMA_ER (n_ok=0, no detection)",
                        fontsize=FS_TICK, color="#c0392b")

    # --- broad band ---
    b_rec = per_subject["per_er"].get("broad_ER", {})
    b_onset, b_statuses, _ = _build_onset_matrix(b_rec, channels)
    if b_onset.shape[1] == 0:
        b_onset = np.full((n_ch, max(n_sz, 1)), np.nan)
        b_statuses = np.array(["onset_unreached"] * b_onset.shape[1])
    im_b = _draw_onset_matrix_band(ax_b, b_onset, b_statuses,
                                    band_label="broad_ER", show_x_ticks=False)
    _b_n_ok = b_rec.get("n_seizures_ok", 0)
    _draw_cov_bar(ax_cov_b, channels, b_rec.get("r_sz_valid_count", {}),
                  _b_n_ok, band_label="broad_ER")
    if _b_n_ok == 0:
        ax_b.set_title("BROAD_ER (n_ok=0, no detection)",
                        fontsize=FS_TICK, color="#c0392b")

    # --- status strip (use the gamma band's statuses by convention; both
    #     bands process identical seizure_idx so the strip is shared) ---
    _draw_status_strip(ax_status, g_statuses)

    # --- y-tick labels on gamma + broad heatmaps (rotated, colored by role) ---
    for ax in (ax_g, ax_b):
        ax.set_yticks(np.arange(n_ch))
        ax.set_yticklabels(channels, fontsize=FS_TICK - 4)
        for tick, ch in zip(ax.get_yticklabels(), channels):
            tick.set_color(_channel_tick_color(_channel_role(ch, focal_set)))

    # --- shared horizontal colorbar at bottom ---
    cbar_ax = fig.add_axes([0.25, 0.025, 0.5, 0.010])
    cbar = fig.colorbar(im_g, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("t_ER_onset (s, relative to clinical onset)",
                    fontsize=FS_TICK - 2)
    cbar.ax.tick_params(labelsize=FS_TICK - 4)

    # --- title ---
    fig.suptitle(_format_subject_title(per_subject, sort_band),
                  fontsize=FS_TITLE - 2, y=0.97)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig_pub(fig, out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Per-seizure figure (placeholder; uses raw extraction)


def render_per_seizure(subject: str, seizure_idx: int, out_path: Path,
                        *, per_subject_json: Optional[Dict] = None) -> Path:
    """Render dual-band (raw + heatmap with onset markers) per-seizure figure.

    Re-extracts the seizure window and recomputes z-ER for the heatmap
    background; t_ER_onset markers come from the v2.3 per-subject JSON
    (so they are guaranteed consistent with r_sz).
    """
    from src.ictal_onset_extraction import (
        BROAD_ER_BANDS, GAMMA_ER_BANDS,
        baseline_zscore_er, compute_er, extract_seizure_window,
        resolve_baseline_window,
    )

    if per_subject_json is None:
        per_subject_json = _load_per_subject_json(subject)

    sw = extract_seizure_window(
        subject, seizure_idx,
        pre_sec=300.0, post_sec=30.0,
        results_root=ROOT / "results", reference="car",
    )
    eeg_rel = (
        sw.eeg_onset_epoch - sw.clin_onset_epoch
        if sw.eeg_onset_epoch is not None else None
    )
    focal_set = set(per_subject_json.get("focal_channels") or [])

    fig_w = 24.0
    fig_h = 12.0
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = mgs.GridSpec(
        nrows=2, ncols=2, figure=fig,
        height_ratios=[1.0, 2.0], width_ratios=[1.0, 1.0],
        left=0.05, right=0.97, top=0.92, bottom=0.08,
        hspace=0.18, wspace=0.10,
    )

    bands = (GAMMA_ER_BANDS, BROAD_ER_BANDS)

    # Choose top-5 overlay channels based on focal_set (or producer top-5)
    sort_band = _select_sort_band(per_subject_json)
    channels, _ = _channel_order(per_subject_json, sort_band)
    if focal_set:
        overlay_chs = [c for c in channels if c in focal_set][:5]
        overlay_label = "top-5 focal"
    else:
        overlay_chs = channels[:5]
        overlay_label = "top-5 producer (no clinical labels)"

    # Find this seizure's record per band for t_onset markers
    def _channel_onsets_for_seizure(band_key: str) -> Dict[str, Optional[float]]:
        rec = per_subject_json["per_er"].get(band_key, {})
        for r in rec.get("seizure_records", []):
            if int(r.get("seizure_idx", -1)) == int(seizure_idx):
                co = r.get("channel_onsets") or {}
                return {ch: (entry or {}).get("t_onset_sec")
                        for ch, entry in co.items()}
        return {}

    for col, band in enumerate(bands):
        band_key = band["key"]
        ax_raw = fig.add_subplot(gs[0, col])
        ax_heat = fig.add_subplot(gs[1, col], sharex=ax_raw)

        er = compute_er(
            sw.signal, fs=sw.fs,
            fast_band=band["fast"], slow_band=band["slow"],
            win_sec=1.0, hop_sec=0.1,
        )
        n_t = er.shape[1]
        bw = resolve_baseline_window(
            n_t, hop_sec=0.1, pre_sec=sw.pre_sec, eeg_onset_rel_sec=eeg_rel,
        )
        if bw.valid:
            z = baseline_zscore_er(er, (bw.start_idx, bw.end_idx), hop_sec=0.1)
        else:
            z = np.full_like(er, np.nan, dtype=np.float64)
        t_axis = (np.arange(n_t) * 0.1 + 0.5) - sw.pre_sec

        # --- Raw row: stack overlay channels ---
        offset = 0.0
        for ch in overlay_chs:
            if ch in sw.ch_names:
                ci = sw.ch_names.index(ch)
                trace = sw.signal[ci]
                # robust scale
                scale = np.nanstd(trace) or 1.0
                t_raw = np.arange(trace.shape[0]) / sw.fs - sw.pre_sec
                ax_raw.plot(t_raw, trace / scale + offset,
                             color=COL_TICK_SOZ if ch in focal_set else COL_TICK_OTHER,
                             lw=0.5, alpha=0.85)
                ax_raw.text(t_axis[0] - 1.0, offset, ch,
                             fontsize=FS_TICK - 4, ha="right", va="center",
                             color=COL_TICK_SOZ if ch in focal_set else COL_TICK_OTHER)
            offset += 4.0
        ax_raw.axvline(0.0, color="black", lw=1.5)
        if eeg_rel is not None and abs(eeg_rel) > 0.5:
            ax_raw.axvline(eeg_rel, color="#8b0000", linestyle="--", lw=1.0)
        ax_raw.set_xlim(*DETECTION_WINDOW_SEC)
        ax_raw.set_yticks([])
        ax_raw.set_title(f"{band_key}: raw SEEG ({overlay_label})",
                          fontsize=FS_LABEL)
        plt.setp(ax_raw.get_xticklabels(), visible=False)

        # --- Heatmap row ---
        # Spec §4.1: rows grouped by SOZ role; within group sorted by
        # t_ER_onset asc; NaN at end. Both bands share the same row order
        # (gamma's ordering applies to broad).
        n_ch = z.shape[0]
        onsets = _channel_onsets_for_seizure(band_key)
        if col == 0:
            order = _row_order_per_seizure(
                sw.ch_names[:n_ch], focal_set, onsets,
            )
            shared_row_order = order
        else:
            order = shared_row_order  # noqa: F821 — set on col=0
        z_ord = z[order]
        ch_ord = [sw.ch_names[i] for i in order]

        norm = TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=3.0)
        im = ax_heat.imshow(
            z_ord, aspect="auto", origin="upper", cmap="RdBu_r",
            norm=norm, interpolation="nearest",
            extent=[t_axis[0], t_axis[-1], n_ch - 0.5, -0.5],
        )
        # onset markers per channel (in the same row ordering)
        for new_ci, ch in enumerate(ch_ord):
            t_on = onsets.get(ch)
            if t_on is not None and np.isfinite(t_on):
                ax_heat.plot(t_on, new_ci, marker="*", markersize=4,
                              color="white", markeredgecolor="black",
                              markeredgewidth=0.4)
        ax_heat.axvline(0.0, color="black", lw=1.5)
        if eeg_rel is not None and abs(eeg_rel) > 0.5:
            ax_heat.axvline(eeg_rel, color="#8b0000", linestyle="--", lw=1.0)
        ax_heat.axvline(bw.end_sec, color="#9a9a9a", linestyle=":", lw=0.8)
        ax_heat.set_xlim(*DETECTION_WINDOW_SEC)
        ax_heat.set_ylim(n_ch - 0.5, -0.5)
        ax_heat.set_xlabel("time relative to clinical onset (s)",
                            fontsize=FS_LABEL)
        ax_heat.set_yticks(np.arange(n_ch))
        ax_heat.set_yticklabels(ch_ord, fontsize=FS_TICK - 5)
        for tick, ch in zip(ax_heat.get_yticklabels(), ch_ord):
            tick.set_color(_channel_tick_color(
                _channel_role(ch, focal_set)))
        if col == 1:
            cbar = fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.02)
            cbar.set_label("z-ER", fontsize=FS_TICK)

    fig.suptitle(
        f"{subject}  |  seizure_idx={seizure_idx}  |  seizure_id={sw.seizure_id}",
        fontsize=FS_TITLE,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig_pub(fig, out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI


def _per_subject_out_path(subject: str) -> Path:
    sid = subject.replace("/", "_")
    return PER_SUBJECT_OUT_DIR / f"{sid}.png"


def _per_seizure_out_path(subject: str, seizure_idx: int) -> Path:
    sid = subject.replace("/", "_")
    return PER_SEIZURE_OUT_DIR / f"{sid}_seizure_{int(seizure_idx):02d}.png"


def cmd_per_subject(args: argparse.Namespace) -> int:
    subject = args.subject
    src = "_sentinel" if args.from_sentinel else "per_subject"
    per_subject = _load_per_subject_json(subject, source=src)
    out_path = _per_subject_out_path(subject)
    if not args.no_skip_existing and out_path.exists():
        print(f"[skip] {out_path} exists", flush=True)
        return 0
    t0 = time.time()
    render_per_subject(per_subject, out_path)
    print(f"[per-subject] {subject} → {out_path}  ({time.time()-t0:.1f}s)",
           flush=True)
    if args.include_seizures:
        n_sz = per_subject.get("n_seizures_total", 0)
        for sz_idx in range(n_sz):
            sz_path = _per_seizure_out_path(subject, sz_idx)
            if not args.no_skip_existing and sz_path.exists():
                continue
            try:
                render_per_seizure(subject, sz_idx, sz_path,
                                    per_subject_json=per_subject)
                print(f"  [seizure {sz_idx}] → {sz_path}", flush=True)
            except Exception as exc:
                print(f"  [seizure {sz_idx}] ERROR: {exc}", flush=True)
    return 0


def cmd_per_seizure(args: argparse.Namespace) -> int:
    subject = args.subject
    sz_idx = int(args.seizure_idx)
    src = "_sentinel" if args.from_sentinel else "per_subject"
    per_subject = _load_per_subject_json(subject, source=src)
    out_path = _per_seizure_out_path(subject, sz_idx)
    if not args.no_skip_existing and out_path.exists():
        print(f"[skip] {out_path} exists", flush=True)
        return 0
    t0 = time.time()
    render_per_seizure(subject, sz_idx, out_path,
                        per_subject_json=per_subject)
    print(f"[per-seizure] {subject}#{sz_idx} → {out_path}  ({time.time()-t0:.1f}s)",
           flush=True)
    return 0


def cmd_cohort(args: argparse.Namespace) -> int:
    src = "_sentinel" if args.from_sentinel else "per_subject"
    subjects = _list_cohort_subjects(source=src)
    if not subjects:
        print(f"[cohort] no v2.3 per-subject JSONs found in {src}/", flush=True)
        return 1
    print(f"[cohort] rendering {len(subjects)} subjects from {src}/", flush=True)
    n_done = 0
    for subj in subjects:
        try:
            sub_args = argparse.Namespace(
                subject=subj,
                from_sentinel=args.from_sentinel,
                no_skip_existing=args.no_skip_existing,
                include_seizures=args.include_seizures,
            )
            cmd_per_subject(sub_args)
            n_done += 1
        except Exception as exc:
            print(f"[cohort] {subj} FAILED: {exc}", flush=True)
    print(f"[cohort] done {n_done}/{len(subjects)}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--no-skip-existing", action="store_true",
                         help="Re-render PNG even if it already exists.")
    common.add_argument("--from-sentinel", action="store_true",
                         help="Read per-subject JSON from _sentinel/ instead of per_subject/.")
    common.add_argument("--include-seizures", action="store_true",
                         help="Also render per-seizure PNGs (per-subject/cohort modes).")

    pps = sub.add_parser("per-subject", parents=[common])
    pps.add_argument("--subject", required=True, help="e.g. epilepsiae/548")

    pse = sub.add_parser("per-seizure", parents=[common])
    pse.add_argument("--subject", required=True)
    pse.add_argument("--seizure-idx", required=True, type=int)

    sub.add_parser("cohort", parents=[common])

    args = parser.parse_args()
    if args.cmd == "per-subject":
        return cmd_per_subject(args)
    if args.cmd == "per-seizure":
        return cmd_per_seizure(args)
    if args.cmd == "cohort":
        return cmd_cohort(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
