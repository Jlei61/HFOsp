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

from src.atlas_loading import (  # noqa: E402  topic5 PR-1 dep-direction fix
    LAYER_A_DIR,
    PER_SUBJECT_DIR,
    REQUIRED_SCHEMA,
    SENTINEL_DIR,
    build_onset_matrix as _build_onset_matrix_impl,
    list_cohort_subjects as _list_cohort_subjects_impl,
    load_per_subject_json as _load_per_subject_json_impl,
    seizure_idx_in_order as _seizure_idx_in_order_impl,
)
from src.plot_style import FS_LABEL, FS_TICK, FS_TITLE, savefig_pub  # noqa: E402

# ---------------------------------------------------------------------------
# Atlas-specific output paths

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
# Backward-compat shims to src.atlas_loading (keeps existing tests / callers
# that import the underscore-prefixed names from this module).


def _load_per_subject_json(subject: str, *, source: str = "per_subject",
                            schema_required: bool = True) -> Dict:
    return _load_per_subject_json_impl(
        subject, source=source, schema_required=schema_required,
    )


def _list_cohort_subjects(*, source: str = "per_subject") -> List[str]:
    return _list_cohort_subjects_impl(source=source)


def _seizure_idx_in_order(per_er_record: Dict) -> List[int]:
    return _seizure_idx_in_order_impl(per_er_record)


def _build_onset_matrix(
    per_er_record: Dict,
    channels: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    return _build_onset_matrix_impl(per_er_record, channels)


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
# Per-seizure figure — reuses the archive plotter helpers from
# scripts/archive/topic1/sentinel_pr6a_step2.py for raw + heatmap rendering.
# Layout: drop the middle z-ER trace row; arrange gamma|broad as two cols.


def _import_archive_plotter():
    """Lazy import of helpers from the archive sentinel plotter.

    The archive script owns the canonical raw + heatmap rendering style
    (publication-grade role coloring, top-5 control trace selection,
    role-segmented heatmap with proper dividers + group labels). This
    reuses those helpers without modifying the archive script.

    The archive computes ``_PROJECT_ROOT`` from its own file location
    (``scripts/archive/topic1/sentinel_pr6a_step2.py`` → ``scripts/archive``)
    which is wrong for our use; monkey-patch it back to the real repo root
    so ``_load_lagpat`` and ``_load_focus_rel`` find the right files.
    """
    archive_dir = ROOT / "scripts" / "archive" / "topic1"
    if str(archive_dir) not in sys.path:
        sys.path.insert(0, str(archive_dir))
    import sentinel_pr6a_step2 as _arch  # type: ignore
    _arch._PROJECT_ROOT = ROOT
    return _arch


def _heatmap_row_order_archive_compat(
    z: np.ndarray,
    t_axis_er: np.ndarray,
    ch_names: list,
    focal_upper: set,
    high_hi_upper: set,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Replicate the row order used by archive _plot_heatmap_panel.

    Channels grouped: High-HI ∩ ictal → High-HI index → ictal only → other;
    within each group sorted by descending post-onset max |z-ER|. Returned
    array is a permutation of channel indices and matches the row order in
    the rendered heatmap, so onset markers can be placed at row i for
    channel ch_names[order[i]].
    """
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
    return np.concatenate([seg_hhi_ictal, seg_hhi_only, seg_ictal_only, seg_other])


def render_per_seizure(subject: str, seizure_idx: int, out_path: Path,
                        *, per_subject_json: Optional[Dict] = None) -> Path:
    """Dual-band per-seizure figure: left=gamma | right=broad.

    Each column = (raw SEEG row) over (full-channel z-ER heatmap row),
    matching the archive's publication style. The middle High-HI z-ER
    trace panel is dropped per spec §4.1 brainstorm decision.

    t_ER_onset markers (✦ on each heatmap row) come from the v2.3
    per-subject JSON channel_onsets — guaranteed consistent with r_sz.
    """
    from src.ictal_onset_extraction import (
        BROAD_ER_BANDS, GAMMA_ER_BANDS,
        baseline_zscore_er, compute_er, extract_seizure_window,
        resolve_baseline_window,
    )
    arch = _import_archive_plotter()

    if per_subject_json is None:
        per_subject_json = _load_per_subject_json(subject)

    # Load lagpat for High-HI cluster info (from PR-1 per_subject JSON,
    # not the v2.3 Layer A JSON). 1084-class subjects with focal=[] still
    # work because lagpat is independent of clinical labels.
    try:
        lagpat_channels, clusters = arch._load_lagpat(subject)
        display_cluster = arch._pick_display_cluster(clusters)
    except (FileNotFoundError, ValueError) as exc:
        # No PR-1 lagpat available — fall back to empty High-HI set.
        # Heatmap will only show ictal-only + other groups.
        lagpat_channels, display_cluster = [], None
        print(f"  [warn] {subject}: lagpat unavailable ({exc}); "
              f"High-HI groups will be empty", flush=True)

    # Extract the seizure window. Try widest window first; shrink if the
    # seizure is near a block boundary. This guarantees we still render
    # something even when the seizure is too close to block_end for
    # post_sec=300 to fit.
    sw = None
    last_exc: Exception | None = None
    for post_attempt in (300.0, 200.0, 100.0, 60.0, 30.0):
        try:
            sw = extract_seizure_window(
                subject, seizure_idx,
                pre_sec=300.0, post_sec=post_attempt,
                results_root=ROOT / "results", reference="car",
            )
            break
        except (ValueError, IndexError) as exc:
            last_exc = exc
            continue
    if sw is None:
        raise ValueError(
            f"{subject} seizure {seizure_idx}: window extraction failed "
            f"at all post_sec attempts (last: {last_exc})"
        )
    eeg_rel = (
        sw.eeg_onset_epoch - sw.clin_onset_epoch
        if sw.eeg_onset_epoch is not None else None
    )
    focal_set = set(per_subject_json.get("focal_channels") or [])
    focal_upper = {c.upper() for c in focal_set}
    high_hi_upper = {ch.upper() for ch in lagpat_channels}

    # Layout: 3 rows (raw / heatmap / colorbar) x 2 cols (gamma | broad).
    # Drop the middle z-ER trace panel per brainstorm decision.
    fig_w = 24.0
    fig_h = 13.0
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = mgs.GridSpec(
        nrows=2, ncols=2, figure=fig,
        height_ratios=[1.0, 1.6], width_ratios=[1.0, 1.0],
        left=0.06, right=0.86, top=0.93, bottom=0.07,
        hspace=0.16, wspace=0.18,
    )

    bands = (GAMMA_ER_BANDS, BROAD_ER_BANDS)
    last_im = None

    # Find this seizure's onset record per band
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

        # ER + baseline z-score for this band
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
            z = arch.baseline_zscore_er(
                er, (bw.start_idx, bw.end_idx), hop_sec=0.1,
            )
            baseline_edge_sec = bw.end_sec
        else:
            z = np.full_like(er, np.nan, dtype=np.float64)
            baseline_edge_sec = -60.0
        t_axis_er = (np.arange(n_t) * 0.1 + 0.5) - sw.pre_sec
        t_axis_raw = np.arange(sw.signal.shape[1]) / sw.fs - sw.pre_sec
        valid_mask = ~np.isnan(z).any(axis=1)

        # --- Raw row (archive _plot_raw_panel) ---
        bg_idx_top5, _ = arch._select_bg_traces(
            z, t_axis_er, sw.ch_names, high_hi_upper, valid_mask,
        )
        if display_cluster is None:
            # Synthetic cluster wrapping all valid channels with rank None
            display_cluster_local = {
                "cluster_id": 0,
                "n_events": 0,
                "fraction": 0.0,
                "rank_by_channel": {},
            }
        else:
            display_cluster_local = display_cluster
        display_entries = arch._build_display_entries(
            ch_names=sw.ch_names,
            focal_upper=focal_upper,
            high_hi_upper=high_hi_upper,
            cluster=display_cluster_local,
            control_idx_top5=bg_idx_top5,
            valid_mask=valid_mask,
        )
        arch._plot_raw_panel(
            ax_raw,
            signal=sw.signal,
            t_axis=t_axis_raw,
            display_entries=display_entries,
            eeg_onset_rel_sec=eeg_rel,
            baseline_edge_sec=baseline_edge_sec,
        )
        ax_raw.set_title(band_key, fontsize=FS_LABEL, pad=6)
        plt.setp(ax_raw.get_xticklabels(), visible=False)

        # --- Heatmap row (archive _plot_heatmap_panel) ---
        _, im = arch._plot_heatmap_panel(
            ax_heat,
            z=z, t_axis_er=t_axis_er, ch_names=sw.ch_names,
            focal_upper=focal_upper, high_hi_upper=high_hi_upper,
            valid_mask=valid_mask,
            eeg_onset_rel_sec=eeg_rel,
            baseline_edge_sec=baseline_edge_sec,
        )
        last_im = im

        # Overlay t_ER_onset markers per row. The archive heatmap orders
        # rows by [hhi_ictal, hhi_only, ictal_only, other] then by post_max
        # within each group; replicate that order here so markers land on
        # the correct visual row.
        order = _heatmap_row_order_archive_compat(
            z, t_axis_er, sw.ch_names, focal_upper, high_hi_upper, valid_mask,
        )
        onsets = _channel_onsets_for_seizure(band_key)
        for visual_row, ch_idx in enumerate(order):
            ch = sw.ch_names[ch_idx]
            t_on = onsets.get(ch)
            if t_on is not None and np.isfinite(t_on):
                ax_heat.plot(t_on, visual_row + 0.5, marker="*", markersize=8,
                              color="white", markeredgecolor="black",
                              markeredgewidth=0.7, zorder=12)

    # --- Shared horizontal colorbar at the right of the figure ---
    if last_im is not None:
        cbar_ax = fig.add_axes([0.88, 0.20, 0.012, 0.55])
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label("z-ER", fontsize=FS_LABEL)
        cbar.ax.tick_params(labelsize=FS_TICK - 2)

    # --- Shared legend at top-right ---
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=arch._role_color("high_hi_ictal"), lw=2.0,
                label="High-HI ∩ ictal"),
        Line2D([0], [0], color=arch._role_color("high_hi_index"), lw=2.0,
                label="High-HI index"),
        Line2D([0], [0], color=arch._role_color("ictal"), lw=2.0,
                label="ictal"),
        Line2D([0], [0], color=arch._role_color("other"), lw=2.0,
                label="other"),
        Line2D([0], [0], **arch.EVENT_LINE_STYLES["clin_onset"],
                label="clin_onset (t=0)"),
        Line2D([0], [0], **arch.EVENT_LINE_STYLES["baseline_edge"],
                label="baseline edge"),
        Line2D([0], [0], marker="*", color="white", markeredgecolor="black",
                markersize=8, lw=0, label="t_ER_onset (CUSUM)"),
    ]
    if eeg_rel is not None and abs(eeg_rel) > 0.5:
        legend_handles.insert(
            -1,
            Line2D([0], [0], **arch.EVENT_LINE_STYLES["eeg_onset"],
                    label=f"eeg_onset (Δ={eeg_rel:+.1f}s)"),
        )
    fig.legend(
        handles=legend_handles,
        loc="upper left", bbox_to_anchor=(0.87, 0.93),
        bbox_transform=fig.transFigure,
        frameon=False, ncol=1, fontsize=FS_TICK - 3,
        handlelength=2.0, labelspacing=0.9, borderaxespad=0.0,
    )

    fig.suptitle(
        f"{subject}  |  seizure_idx={seizure_idx}  |  seizure_id={sw.seizure_id}",
        fontsize=FS_TITLE, y=0.97,
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
