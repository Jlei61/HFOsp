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
from contextlib import contextmanager
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as mgs
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
EEG_ZOOM_PRE_SEC = 90.0
EEG_ZOOM_POST_SEC = 90.0
_MIN_DISPLAY_OVERLAP_SEC = 60.0  # below this, treat eeg_onset as bogus → zoom on t=0
HEATMAP_ROLE_ORDER = ("high_hi_ictal", "high_hi_index", "ictal", "other")
HEATMAP_ROLE_LABEL = {
    "high_hi_ictal": "High-HI ∩ ictal",
    "high_hi_index": "High-HI index",
    "ictal": "ictal only",
    "other": "other",
}

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


def _lagpat_json_to_display_clusters(subject: str, d: Dict) -> Tuple[List[str], List[Dict]]:
    """Convert a PR-2 adaptive-cluster JSON into archive display clusters."""
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
            ch: (
                None if r is None or (isinstance(r, float) and np.isnan(r))
                else int(r)
            )
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


def _load_display_lagpat(subject: str) -> Tuple[List[str], List[Dict]]:
    """Load the High-HI / Lagpat channel pool for per-seizure display.

    Prefer the **broad** channel-pool lagpat (``interictal_propagation_masked_broad``,
    top-N by event count) — High-HI is a *range* of channels, and the broad pool
    covers subjects whose narrow Lagpat channels are absent from the ictal
    recording (e.g. yuquan zhaojinrui: narrow 4ch→0 ictal match, broad 20ch→10).
    Fall back to the narrow masked pool, then the unmasked legacy path.
    """
    sid = subject.replace("/", "_")
    candidates = [
        ROOT / "results" / "interictal_propagation_masked_broad" / "per_subject" / f"{sid}.json",
        ROOT / "results" / "interictal_propagation_masked" / "per_subject" / f"{sid}.json",
        ROOT / "results" / "interictal_propagation" / "per_subject" / f"{sid}.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                return _lagpat_json_to_display_clusters(subject, json.load(fh))
    raise FileNotFoundError(
        f"{subject}: no interictal propagation per_subject JSON found in "
        f"{', '.join(str(p.parent) for p in candidates)}"
    )


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


def _display_window_around_eeg(
    eeg_onset_rel_sec: Optional[float],
    *,
    pre_sec: float = EEG_ZOOM_PRE_SEC,
    post_sec: float = EEG_ZOOM_POST_SEC,
) -> Tuple[float, float]:
    """Return a display window centered on EEG onset.

    Epilepsiae has separate clinical and EEG onset annotations, so
    ``eeg_onset_rel_sec`` can be negative relative to clinical onset.
    Yuquan only has EEG onset; the loader uses that as t=0 and passes
    ``None`` here, which correctly falls back to a window around zero.
    """
    if eeg_onset_rel_sec is None:
        center = 0.0
    else:
        center = float(eeg_onset_rel_sec)
        if not np.isfinite(center):
            center = 0.0
    return (center - float(pre_sec), center + float(post_sec))


def _clip_display_window_to_signal(
    display_window: Tuple[float, float],
    *,
    pre_sec: float,
    post_sec: float,
) -> Tuple[float, float]:
    """Clip a requested display window to the extracted signal span.

    If the requested (EEG-centred) window barely overlaps the signal — which
    happens when ``eeg_onset`` is a bogus annotation hundreds/thousands of
    seconds off (some Epilepsiae seizures) — fall back to a ±EEG_ZOOM window
    around t=0 (clinical onset / reference) rather than the full signal span,
    so every figure stays zoomed near onset.
    """
    lo, hi = float(display_window[0]), float(display_window[1])
    lo = max(lo, -float(pre_sec))
    hi = min(hi, float(post_sec))
    if hi - lo < _MIN_DISPLAY_OVERLAP_SEC:
        return (
            max(-EEG_ZOOM_PRE_SEC, -float(pre_sec)),
            min(EEG_ZOOM_POST_SEC, float(post_sec)),
        )
    return (lo, hi)


def _alignment_reference(
    dataset: str,
    eeg_rel: Optional[float],
    *,
    pre_sec: float,
    post_sec: float,
) -> Tuple[float, bool]:
    """Pick the t=0 reference: EEG (electrographic) onset when usable, else clinical.

    Returns ``(align_ref_sec, ref_is_eeg)``. ``align_ref_sec`` is the offset of
    the reference in the clinical-onset frame; subtract it from clin-frame times
    to get display (EEG-aligned) times. Yuquan's ``extract_seizure_window``
    already uses EEG onset / reference as t=0, so ``align_ref_sec=0``. For
    Epilepsiae, align to EEG onset only when its ±zoom window overlaps the
    signal — a bogus ``eeg_onset`` annotation (hundreds/thousands of s off)
    falls back to clinical onset.
    """
    if dataset == "yuquan":
        return 0.0, True
    if eeg_rel is not None and np.isfinite(eeg_rel):
        lo = max(float(eeg_rel) - EEG_ZOOM_PRE_SEC, -float(pre_sec))
        hi = min(float(eeg_rel) + EEG_ZOOM_POST_SEC, float(post_sec))
        if hi - lo >= _MIN_DISPLAY_OVERLAP_SEC:
            return float(eeg_rel), True
    return 0.0, False


@contextmanager
def _archive_display_window(arch, display_window: Tuple[float, float]) -> Iterator[None]:
    """Temporarily set archive plotter display constants for raw traces."""
    old = (arch.DISPLAY_TMIN, arch.DISPLAY_TMAX)
    arch.DISPLAY_TMIN = float(display_window[0])
    arch.DISPLAY_TMAX = float(display_window[1])
    try:
        yield
    finally:
        arch.DISPLAY_TMIN, arch.DISPLAY_TMAX = old


def _select_bg_traces_in_window(
    z: np.ndarray,
    t_axis_er: np.ndarray,
    ch_names: Sequence[str],
    high_hi_upper: set,
    valid_mask: np.ndarray,
    display_window: Tuple[float, float],
    *,
    n_bg: int = 5,
) -> np.ndarray:
    """Select non-High-HI controls with the largest |z-ER| in the display window."""
    non_high_hi_idx = np.array(
        [
            i for i, nm in enumerate(ch_names)
            if nm.upper() not in high_hi_upper and bool(valid_mask[i])
        ],
        dtype=int,
    )
    if non_high_hi_idx.size == 0:
        return np.array([], dtype=int)
    xmask = (
        (t_axis_er >= float(display_window[0]))
        & (t_axis_er <= float(display_window[1]))
    )
    if not xmask.any():
        return np.array([], dtype=int)
    with np.errstate(invalid="ignore"):
        score = np.nanmax(np.abs(z[non_high_hi_idx][:, xmask]), axis=1)
    score = np.where(np.isfinite(score), score, -np.inf)
    order = np.argsort(-score)
    return non_high_hi_idx[order[:int(n_bg)]]


def _heatmap_order_from_display_entries(
    display_entries: Sequence[Dict],
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Use the raw-panel channel set for the heatmap, grouped by semantic role."""
    order: List[int] = []
    counts = {role: 0 for role in HEATMAP_ROLE_ORDER}
    seen: set[int] = set()
    for role in HEATMAP_ROLE_ORDER:
        for entry in display_entries:
            if entry.get("role") != role:
                continue
            idx = int(entry["idx"])
            if idx in seen:
                continue
            seen.add(idx)
            order.append(idx)
            counts[role] += 1
    return np.asarray(order, dtype=int), counts


def _ordered_display_rows(
    display_entries: Sequence[Dict],
) -> Tuple[List[Dict], Dict[str, int]]:
    """Role-grouped row list shared by the aligned raw + heatmap panels.

    Returns ``(rows, counts)`` where ``rows[p]`` = ``{idx, role, channel}``
    for the channel drawn at visual row ``p`` (top = 0). Both the raw SEEG
    panel and the z-ER heatmap iterate this single ordering so that raw
    trace ``p`` and heatmap row ``p`` are the same channel — the y-axis
    alignment the two side-by-side panels rely on.
    """
    counts = {role: 0 for role in HEATMAP_ROLE_ORDER}
    rows: List[Dict] = []
    seen: set[int] = set()
    for role in HEATMAP_ROLE_ORDER:
        for entry in display_entries:
            if entry.get("role") != role:
                continue
            idx = int(entry["idx"])
            if idx in seen:
                continue
            seen.add(idx)
            rows.append({"idx": idx, "role": role, "channel": entry["channel"]})
            counts[role] += 1
    return rows, counts


MAX_SEQUENCE_CH = 8  # at most this many channels per per-seizure figure


def _select_sequence_rows(
    z_sel: np.ndarray,
    t_axis_er: np.ndarray,
    ch_names: Sequence[str],
    *,
    high_hi_upper: set,
    focal_upper: set,
    valid_mask: np.ndarray,
    display_window: Tuple[float, float],
    onsets: Dict[str, Optional[float]],
    align_ref_sec: float,
    max_ch: int = MAX_SEQUENCE_CH,
) -> List[Dict]:
    """Pick ≤``max_ch`` Lagpat/High-HI channels with the clearest ictal sequence.

    Selection band is the *broad* band (``z_sel`` / ``onsets`` are broad). The
    clearest-sequence rule:

    - candidate pool = High-HI (Lagpat) channels that are valid in this band;
    - selection key = ``(has a visible in-window onset, peak |z-ER| in-window)``
      descending — channels that actually participate in the recruitment
      sequence rank first, then the strongest activation;
    - keep the top ``max_ch``;
    - order top→bottom by **onset time ascending** (earliest recruited at top;
      channels without a visible onset sink to the bottom by peak desc) so the
      connecting line through the onset markers reads as the recruitment order.

    Only High-HI/Lagpat channels are shown — no non-Lagpat fill. If a subject
    has zero High-HI channels in the ictal recording (even after the broad pool),
    the returned list is empty and the caller skips the figure.

    Each returned row carries the **display-frame** onset (``onset_disp``) for
    the connecting line / marker (already shifted by ``align_ref_sec``).
    """
    lo, hi = float(display_window[0]), float(display_window[1])
    xmask = (t_axis_er >= lo) & (t_axis_er <= hi)
    cand: List[Dict] = []
    for i, nm in enumerate(ch_names):
        if nm.upper() not in high_hi_upper or not bool(valid_mask[i]):
            continue
        if xmask.any():
            seg = z_sel[i, xmask]
            finite = seg[np.isfinite(seg)]
            peak = float(np.max(np.abs(finite))) if finite.size else 0.0
        else:
            peak = 0.0
        raw_onset = onsets.get(nm)
        onset_disp = None
        if raw_onset is not None and np.isfinite(raw_onset):
            od = float(raw_onset) - float(align_ref_sec)
            if lo <= od <= hi:
                onset_disp = od
        cand.append({
            "idx": i,
            "channel": nm,
            "role": ("high_hi_ictal" if nm.upper() in focal_upper
                     else "high_hi_index"),
            "peak": peak,
            "onset_disp": onset_disp,
        })
    if not cand:
        return []
    # selection: prefer a visible onset, then strongest peak
    cand.sort(key=lambda c: (c["onset_disp"] is None, -c["peak"]))
    selected = cand[:int(max_ch)]
    # order top->bottom by onset ascending (no-onset last, by peak desc)
    selected.sort(key=lambda c: (
        c["onset_disp"] is None,
        c["onset_disp"] if c["onset_disp"] is not None else 0.0,
        -c["peak"],
    ))
    return selected


def _apply_time_locator(ax: plt.Axes, display_window: Tuple[float, float]) -> None:
    span = float(display_window[1]) - float(display_window[0])
    if span <= 180.0:
        ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
    else:
        ax.xaxis.set_major_locator(mticker.MultipleLocator(100))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(25))


def _raw_label_fontsize(n_rows: int) -> float:
    """Channel-name tick fontsize that stays legible as row count grows."""
    if n_rows <= 8:
        return 11.0
    if n_rows <= 16:
        return 8.0
    if n_rows <= 24:
        return 6.5
    if n_rows <= 34:
        return 5.5
    return 4.5


def _plot_aligned_raw_panel(
    ax: plt.Axes,
    *,
    arch,
    signal: np.ndarray,
    t_axis_raw: np.ndarray,
    rows: Sequence[Dict],
    display_window: Tuple[float, float],
    eeg_onset_rel_sec: Optional[float],
    baseline_edge_sec: float,
    show_xlabel: bool,
    x_label: str,
):
    """Raw SEEG traces in heatmap-row coordinates (y-aligned with the heatmap).

    Channel ``rows[p]`` is drawn centred at ``y = p + 0.5`` and its robust-
    scaled waveform stays within the ``[p, p+1]`` band, so the panel shares
    the heatmap's ``ylim = (n, 0)`` row geometry exactly.
    """
    n = len(rows)
    disp = (t_axis_raw >= float(display_window[0])) & (
        t_axis_raw <= float(display_window[1])
    )
    t_disp = t_axis_raw[disp]
    dt = float(np.median(np.diff(t_disp))) if t_disp.size > 1 else 0.0
    stride = max(1, int(round(1.0 / (dt * arch.RAW_PLOT_TARGET_HZ)))) if dt > 0 else 1
    t_plot = t_disp[::stride]
    amp = 0.42 / float(arch.RAW_TRACE_CLIP)  # ±CLIP maps to ±0.42 of a row
    for p, r in enumerate(rows):
        ts = arch._robust_scale_trace(signal[r["idx"], disp])[::stride]
        # positive deflection points up (toward smaller y, origin upper)
        y = (p + 0.5) - ts * amp
        ax.plot(
            t_plot, y,
            color=arch._role_color(r["role"]),
            lw=arch._role_linewidth(r["role"], raw=True),
            alpha=arch._role_alpha(r["role"], raw=True),
            zorder=4,
        )
    ax.set_ylim(n, 0)
    arch.style_panel(ax)
    # custom (small) channel-name ticks AFTER style_panel (which forces FS_TICK)
    fs = _raw_label_fontsize(n)
    ax.set_yticks([p + 0.5 for p in range(n)])
    ax.set_yticklabels([r["channel"] for r in rows], fontsize=fs)
    for tlbl, r in zip(ax.get_yticklabels(), rows):
        tlbl.set_color(arch._role_color(r["role"]))
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(axis="y", labelsize=_raw_label_fontsize(n), length=2)
    arch._draw_event_lines(
        ax, eeg_onset_rel_sec=eeg_onset_rel_sec, baseline_edge_sec=baseline_edge_sec,
    )
    ax.set_xlim(float(display_window[0]), float(display_window[1]))
    _apply_time_locator(ax, display_window)
    if show_xlabel:
        ax.set_xlabel(x_label, fontsize=FS_LABEL)
    else:
        plt.setp(ax.get_xticklabels(), visible=False)


def _plot_aligned_heatmap_panel(
    ax: plt.Axes,
    *,
    arch,
    z: np.ndarray,
    t_axis_er: np.ndarray,
    rows: Sequence[Dict],
    display_window: Tuple[float, float],
    eeg_onset_rel_sec: Optional[float],
    baseline_edge_sec: float,
    show_xlabel: bool,
    x_label: str,
):
    """z-ER heatmap over the same onset-ordered rows as the raw panel.

    Rows are ordered by recruitment (onset) time, not grouped, so there are no
    group separators; the per-channel names live on the LEFT raw panel and both
    panels share ``ylim = (n, 0)``.
    """
    n = len(rows)
    order = [r["idx"] for r in rows]
    xmask = (
        (t_axis_er >= float(display_window[0]))
        & (t_axis_er <= float(display_window[1]))
    )
    if n == 0 or not xmask.any():
        heat = np.zeros((max(1, n), 1), dtype=float)
        extent = [float(display_window[0]), float(display_window[1]), max(1, n), 0]
        vmax = 2.0
    else:
        heat = z[order][:, xmask]
        extent = [
            float(t_axis_er[xmask][0]),
            float(t_axis_er[xmask][-1]),
            n,
            0,
        ]
        finite_abs = np.abs(heat[np.isfinite(heat)])
        vmax = float(np.percentile(finite_abs, 99)) if finite_abs.size else 2.0
        vmax = max(vmax, 2.0)

    im = ax.imshow(
        heat, aspect="auto", origin="upper", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, extent=extent, interpolation="nearest",
    )
    ax.set_ylim(n, 0)
    arch.style_panel(ax)
    ax.set_yticks([])  # channel names live on the left raw panel
    arch._draw_event_lines(
        ax, eeg_onset_rel_sec=eeg_onset_rel_sec, baseline_edge_sec=baseline_edge_sec,
    )
    ax.set_xlim(float(display_window[0]), float(display_window[1]))
    _apply_time_locator(ax, display_window)
    if show_xlabel:
        ax.set_xlabel(x_label, fontsize=FS_LABEL)
    else:
        plt.setp(ax.get_xticklabels(), visible=False)

    return ({"heatmap_vmax": float(vmax), "n_rows": int(n)}, im)


class _FigureSkipped(Exception):
    """Raised when a per-seizure figure has no displayable High-HI channels."""


def render_per_seizure(subject: str, seizure_idx: int, out_path: Path,
                        *, per_subject_json: Optional[Dict] = None) -> Path:
    """Dual-band per-seizure figure: horizontal layout.

    Two band rows (gamma top / broad bottom); within each row the left
    panel is the raw SEEG and the right panel is the z-ER heatmap. The two
    panels share the same role-grouped channel order and ``ylim``, so raw
    trace ``p`` lines up with heatmap row ``p`` (per-channel names on the
    far-left, role-group labels on the heatmap's right). The heatmap uses
    the same channel set as the raw panel, so the ``other`` block is a
    small set of selected controls rather than every non-index channel.

    t_ER_onset markers (✦ on each heatmap row) come from the v2.3
    per-subject JSON channel_onsets — guaranteed consistent with r_sz.
    """
    from src.ictal_onset_extraction import (
        BROAD_ER_BANDS, GAMMA_ER_BANDS,
        baseline_zscore_er, compute_er, detect_er_onset_preview,
        extract_seizure_window, resolve_baseline_window,
    )
    arch = _import_archive_plotter()
    dataset = subject.split("/", 1)[0]

    if per_subject_json is None:
        per_subject_json = _load_per_subject_json(subject)

    # Load lagpat for High-HI cluster info (from PR-2 per_subject JSON,
    # not the v2.3 Layer A JSON). 1084-class subjects with focal=[] still
    # work because lagpat is independent of clinical labels.
    try:
        lagpat_channels, clusters = _load_display_lagpat(subject)
        display_cluster = arch._pick_display_cluster(clusters)
    except (FileNotFoundError, ValueError) as exc:
        # No PR-2 lagpat available — fall back to empty High-HI set.
        # Heatmap will only show ictal-only + other groups.
        lagpat_channels, display_cluster = [], None
        print(f"  [warn] {subject}: lagpat unavailable ({exc}); "
              f"High-HI groups will be empty", flush=True)

    # Extract the seizure window. Try widest window first; shrink if the
    # seizure is near a block boundary. This guarantees we still render
    # something even when the seizure is too close to block_end for
    # post_sec=300 to fit.
    # Yuquan: the High-HI/Lagpat channels are bipolar (alias-to-left); draw the
    # ictal z-ER in the same montage so the channel names match. Epilepsiae
    # High-HI matches CAR directly, so it stays CAR.
    ref = "bipolar" if dataset == "yuquan" else "car"
    alias_left = (dataset == "yuquan")
    sw = None
    last_exc: Exception | None = None
    for post_attempt in (300.0, 200.0, 100.0, 60.0, 30.0):
        try:
            sw = extract_seizure_window(
                subject, seizure_idx,
                pre_sec=300.0, post_sec=post_attempt,
                results_root=ROOT / "results", reference=ref,
                alias_bipolar_to_left=alias_left,
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
    # Align t=0 to the EEG (electrographic) onset when usable; clinical onset
    # becomes a secondary marker. Yuquan is already EEG/ref-aligned. Bogus
    # eeg_onset annotations fall back to t=0 = clinical onset.
    align_ref_sec, ref_is_eeg = _alignment_reference(
        dataset, eeg_rel, pre_sec=sw.pre_sec, post_sec=sw.post_sec,
    )
    lo_sig = -float(sw.pre_sec) - align_ref_sec
    hi_sig = float(sw.post_sec) - align_ref_sec
    display_window = (
        max(-EEG_ZOOM_PRE_SEC, lo_sig),
        min(EEG_ZOOM_POST_SEC, hi_sig),
    )
    x_label = "Time (s)"  # t=0 reference is named in the legend (ref_legend)
    if dataset == "yuquan":
        ref_legend = "eeg_onset/ref (t=0)"
        other_onset_disp = None
    elif ref_is_eeg:
        ref_legend = "eeg_onset (t=0)"
        other_onset_disp = 0.0 - align_ref_sec  # clinical onset display position
    else:
        ref_legend = "clin_onset (t=0)"
        other_onset_disp = None  # eeg_onset bogus → not shown
    focal_set = set(per_subject_json.get("focal_channels") or [])
    focal_upper = {c.upper() for c in focal_set}
    high_hi_upper = {ch.upper() for ch in lagpat_channels}

    # Layout: 2 rows (gamma top / broad bottom) x 2 cols (raw SEEG | z-ER).
    # Within each band row the raw panel (left) and heatmap (right) share the
    # same onset-ordered channel set + ylim, so SEEG trace p aligns with
    # heatmap row p (sharey within the row).
    fig_w = 20.0
    fig_h = 12.5
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = mgs.GridSpec(
        nrows=2, ncols=2, figure=fig,
        height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0],
        left=0.08, right=0.82, top=0.91, bottom=0.08,
        hspace=0.22, wspace=0.10,
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

    # ---- Precompute both bands (ER → baseline-z → display-frame axes) ----
    band_blobs: Dict[str, Dict] = {}
    for band in bands:
        bkey = band["key"]
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
            z = arch.baseline_zscore_er(er, (bw.start_idx, bw.end_idx), hop_sec=0.1)
            baseline_edge_disp = bw.end_sec - align_ref_sec
        else:
            z = np.full_like(er, np.nan, dtype=np.float64)
            baseline_edge_disp = -60.0 - align_ref_sec
        t_axis_er = ((np.arange(n_t) * 0.1 + 0.5) - sw.pre_sec) - align_ref_sec

        # Per-channel ER onset in the DISPLAY frame.
        #  - Epilepsiae (CAR): reuse the Layer A CAR onsets (names match), shifted.
        #  - Yuquan (bipolar): Layer A is CAR and does NOT match the bipolar
        #    channel names, so detect the onset in-figure from the displayed
        #    bipolar z-ER with the same CUSUM detector.
        onsets_disp: Dict[str, Optional[float]] = {}
        if dataset == "yuquan":
            det_mask = (t_axis_er >= baseline_edge_disp) & (t_axis_er <= display_window[1])
            det_idx = np.where(det_mask)[0]
            det_win = (int(det_idx[0]), int(det_idx[-1]) + 1) if det_idx.size >= 2 else None
            for i, nm in enumerate(sw.ch_names):
                if det_win is None:
                    onsets_disp[nm] = None
                    continue
                ev = detect_er_onset_preview(z[i], t_axis_er, det_win,
                                             bias=0.5, threshold=5.0)
                onsets_disp[nm] = float(ev.onset_sec) if ev.detected else None
        else:
            la = _channel_onsets_for_seizure(bkey)
            for nm in sw.ch_names:
                v = la.get(nm)
                onsets_disp[nm] = (float(v) - align_ref_sec) if (
                    v is not None and np.isfinite(v)) else None

        band_blobs[bkey] = {
            "z": z,
            "t_axis_er": t_axis_er,
            "t_axis_raw": (np.arange(sw.signal.shape[1]) / sw.fs - sw.pre_sec) - align_ref_sec,
            "baseline_edge_disp": baseline_edge_disp,
            "valid_mask": ~np.isnan(z).any(axis=1),
            "onsets_disp": onsets_disp,  # DISPLAY frame, None where no onset
        }

    # ---- Select ≤8 sequence channels by the BROAD band (Lagpat ∩ clearest) ----
    sb = band_blobs[BROAD_ER_BANDS["key"]]
    rows = _select_sequence_rows(
        sb["z"], sb["t_axis_er"], sw.ch_names,
        high_hi_upper=high_hi_upper, focal_upper=focal_upper,
        valid_mask=sb["valid_mask"], display_window=display_window,
        onsets=sb["onsets_disp"], align_ref_sec=0.0, max_ch=MAX_SEQUENCE_CH,
    )
    if not rows:
        plt.close(fig)
        raise _FigureSkipped(
            f"{subject} seizure {seizure_idx}: no High-HI/Lagpat channels present "
            f"in the ictal recording — figure skipped"
        )

    # ---- Plot both band rows with the shared, onset-ordered channel set ----
    for row_i, band in enumerate(bands):
        bkey = band["key"]
        blob = band_blobs[bkey]
        show_xlabel = (row_i == len(bands) - 1)
        ax_raw = fig.add_subplot(gs[row_i, 0])
        # No sharey: both panels set identical ylim=(n,0) in the same gridspec
        # row, so rows align without the heatmap's set_yticks([]) wiping the raw
        # panel's channel-name ticks.
        ax_heat = fig.add_subplot(gs[row_i, 1])

        _plot_aligned_raw_panel(
            ax_raw, arch=arch, signal=sw.signal, t_axis_raw=blob["t_axis_raw"],
            rows=rows, display_window=display_window,
            eeg_onset_rel_sec=other_onset_disp,
            baseline_edge_sec=blob["baseline_edge_disp"],
            show_xlabel=show_xlabel, x_label=x_label,
        )
        ax_raw.set_title(f"{bkey} · raw SEEG", fontsize=FS_LABEL,
                          loc="left", fontweight="bold", pad=6)

        _, im = _plot_aligned_heatmap_panel(
            ax_heat, arch=arch, z=blob["z"], t_axis_er=blob["t_axis_er"],
            rows=rows, display_window=display_window,
            eeg_onset_rel_sec=other_onset_disp,
            baseline_edge_sec=blob["baseline_edge_disp"],
            show_xlabel=show_xlabel, x_label=x_label,
        )
        ax_heat.set_title(f"{bkey} · z-ER", fontsize=FS_LABEL,
                           loc="left", fontweight="bold", pad=6)
        last_im = im

        # Peak markers (✦, big) at this band's onset, connected top→bottom to
        # show the recruitment sequence.
        onsets_disp = blob["onsets_disp"]
        seq_pts = []
        for p, r in enumerate(rows):
            t_disp = onsets_disp.get(r["channel"])
            if t_disp is None or not np.isfinite(t_disp):
                continue
            if not (display_window[0] <= t_disp <= display_window[1]):
                continue
            seq_pts.append((t_disp, p + 0.5))
        if len(seq_pts) >= 2:
            xs, ys = zip(*seq_pts)
            ax_heat.plot(xs, ys, "-", color="#10131a", lw=1.8, alpha=0.8, zorder=12)
        for t_disp, yy in seq_pts:
            ax_heat.plot(t_disp, yy, marker="*", markersize=18,
                          color="white", markeredgecolor="black",
                          markeredgewidth=1.2, zorder=14)

    # --- Shared colorbar at the far right (clear of heatmap group labels) ---
    if last_im is not None:
        cbar_ax = fig.add_axes([0.90, 0.30, 0.014, 0.40])
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label("z-ER", fontsize=FS_LABEL)
        cbar.ax.tick_params(labelsize=FS_TICK - 2)

    # --- Shared legend at top-right ---
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=arch._role_color("high_hi_ictal"), lw=2.0,
                label="High-HI ∩ SOZ"),
        Line2D([0], [0], color=arch._role_color("high_hi_index"), lw=2.0,
                label="High-HI (Lagpat)"),
        Line2D([0], [0], **arch.EVENT_LINE_STYLES["clin_onset"],
                label=ref_legend),
        Line2D([0], [0], **arch.EVENT_LINE_STYLES["baseline_edge"],
                label="baseline edge"),
        Line2D([0], [0], marker="*", color="white", markeredgecolor="black",
                markersize=15, markeredgewidth=1.2, lw=0,
                label="t_ER_onset (peak)"),
        Line2D([0], [0], color="#10131a", lw=1.8, alpha=0.8,
                label="recruitment order"),
    ]
    if (other_onset_disp is not None and abs(other_onset_disp) > 0.5
            and display_window[0] <= other_onset_disp <= display_window[1]):
        legend_handles.insert(
            3,
            Line2D([0], [0], **arch.EVENT_LINE_STYLES["eeg_onset"],
                    label=f"clin_onset (Δ={other_onset_disp:+.1f}s)"),
        )
    fig.legend(
        handles=legend_handles,
        loc="upper left", bbox_to_anchor=(0.855, 0.985),
        bbox_transform=fig.transFigure,
        frameon=False, ncol=1, fontsize=FS_TICK - 3,
        handlelength=2.0, labelspacing=0.9, borderaxespad=0.0,
    )

    fig.suptitle(
        f"{subject}  |  seizure_idx={seizure_idx}  |  seizure_id={sw.seizure_id}"
        f"  |  display=[{display_window[0]:.0f},{display_window[1]:.0f}]s",
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
    if getattr(args, "out_dir", None):
        out_path = (Path(args.out_dir)
                    / f"{subject.replace('/', '_')}_seizure_{sz_idx:02d}.png")
    else:
        out_path = _per_seizure_out_path(subject, sz_idx)
    if not args.no_skip_existing and out_path.exists():
        print(f"[skip] {out_path} exists", flush=True)
        return 0
    t0 = time.time()
    try:
        render_per_seizure(subject, sz_idx, out_path,
                            per_subject_json=per_subject)
    except _FigureSkipped as exc:
        if out_path.exists():
            out_path.unlink()  # remove any stale figure from an earlier run
        print(f"[skip] {exc}", flush=True)
        return 0
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
    pse.add_argument("--out-dir", default=None,
                     help="Override output directory (e.g. a preview folder).")

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
