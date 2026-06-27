"""Detection-grade seizure onset/end REVIEW panels (visual label correction).

Goal: let a human eyeball the precise onset / end of every labeled seizure by
looking at most channels' electrical activity in a window that brackets the
labeled onset and offset. This is a QC / annotation-correction utility, NOT a
paper figure — low DPI, no high-resolution rendering required.

Two products per subject (both requested 2026-06-21):
  - per-seizure DETAIL: one figure per seizure, all channels stacked, window
    spans [earliest-onset-margin .. latest-offset+margin], vertical lines at
    every onset/offset annotation. Use this to read/correct precise times.
  - per-subject OVERVIEW: one contact-sheet of small activity-envelope panels,
    one per seizure, to quickly scan which seizures' labels look suspicious.

Label source (read-only): the seizure inventories under ``results/``
(``{dataset}_seizure_inventory.csv``). Epilepsiae carries clinical AND EEG
onset/offset; Yuquan carries EEG onset/offset only.

Montage: per-dataset detection reference (traced in run_topic5_ictal_recruitment) —
epilepsiae='car', yuquan='bipolar'.

Usage::

    python scripts/plot_seizure_onset_review.py --dataset epilepsiae --subject 548
    python scripts/plot_seizure_onset_review.py --dataset epilepsiae --all
    python scripts/plot_seizure_onset_review.py --all-datasets
    python scripts/plot_seizure_onset_review.py --dataset yuquan --subject litengsheng \
        --pre-margin 30 --post-margin 30
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ictal_onset_extraction import (  # noqa: E402
    _read_csv_rows,
    _resolve_inventory_paths,
    extract_seizure_window,
)
from src.event_periodicity import (  # noqa: E402  (canonical SOZ name matching)
    _normalize_channel_name,
    match_bipolar_soz,
)

_SOZ_CACHE: dict = {}


def _soz_set(dataset: str, sid: str) -> set:
    """Normalized clinician-SOZ contact set for one subject (empty if no list)."""
    if dataset not in _SOZ_CACHE:
        p = SOZ_PATHS.get(dataset)
        _SOZ_CACHE[dataset] = json.loads(p.read_text()) if p and p.exists() else {}
    return {_normalize_channel_name(c) for c in _SOZ_CACHE[dataset].get(sid, [])}


def soz_channel_flags(ch_names, soz_set) -> list:
    """Per-channel bool: does this montage channel touch the SOZ set? Reuses the
    canonical ``match_bipolar_soz`` (single montage = exact contact; bipolar =
    any constituent contact in SOZ). All-False if the subject has no SOZ list."""
    if not soz_set:
        return [False] * len(list(ch_names))
    return [match_bipolar_soz(n, soz_set) == "soz" for n in ch_names]

# Per-dataset detection reference (matches run_topic5_ictal_recruitment ICTAL_REFERENCE).
DATASET_REFERENCE = {"epilepsiae": "car", "yuquan": "bipolar"}
OUT_ROOT = _ROOT / "results" / "seizure_onset_review"
SOZ_PATHS = {
    "epilepsiae": _ROOT / "results" / "epilepsiae_soz_core_channels.json",
    "yuquan": _ROOT / "results" / "yuquan_soz_core_channels.json",
}

# Marker drawing: color by annotation source, line style by onset/offset.
ANN_COLOR = {"clin": "crimson", "eeg": "royalblue"}
KIND_STYLE = {"onset": "-", "offset": "--"}
SOZ_COLOR = "#1a9e3e"     # clinician-annotated SOZ channels (traces + labels + envelope)
NONSOZ_COLOR = "0.30"

# Display constants (detection-grade; readability over fidelity).
CLIP = 4.0       # clip MAD-normalized traces to ±CLIP so lanes don't collide
GAP = 9.0        # vertical gap between stacked channels (MAD units)
BP_LO = 1.0      # bandpass low edge (Hz) — kill DC drift
BP_HI = 80.0     # bandpass high edge (Hz) — trim HF noise (capped at <Nyquist)
SINGLE_MAX_SPAN = 160.0   # span (s) above which the detail splits into zoom panels
ONSET_ZOOM_PRE = 20.0     # onset-zoom starts this many s before the earliest onset
ONSET_ZOOM_POST = 70.0    # onset-zoom reaches at least this many s past the earliest onset
END_ZOOM_PRE = 70.0       # end-zoom panel starts this many s before the offset


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_plot_seizure_onset_review.py)
# ---------------------------------------------------------------------------
def _to_float(row: dict, key: str) -> Optional[float]:
    """Parse ``row[key]`` as float; return None if absent/blank/unparseable."""
    val = row.get(key, "")
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def seizure_marker_times(row: dict, dataset: str):
    """Onset/offset markers for one seizure, in seconds relative to the t=0 reference.

    t=0 reference matches ``extract_seizure_window``'s ``onset_field``:
    ``clin_onset_epoch`` for epilepsiae, ``eeg_onset_epoch`` for yuquan.

    Returns ``(markers, t0_epoch)`` where each marker is
    ``{"label", "t_rel", "kind" in {onset,offset}, "ann" in {clin,eeg}}``.
    Markers whose epoch is missing are omitted (e.g. incomplete intervals).
    """
    if dataset == "epilepsiae":
        t0 = _to_float(row, "clin_onset_epoch")
        if t0 is None:
            raise ValueError("epilepsiae seizure row missing clin_onset_epoch")
        specs = [
            ("clin onset", "clin_onset_epoch", "onset", "clin"),
            ("clin offset", "clin_offset_epoch", "offset", "clin"),
            ("eeg onset", "eeg_onset_epoch", "onset", "eeg"),
            ("eeg offset", "eeg_offset_epoch", "offset", "eeg"),
        ]
    elif dataset == "yuquan":
        t0 = _to_float(row, "eeg_onset_epoch")
        if t0 is None:
            raise ValueError("yuquan seizure row missing eeg_onset_epoch")
        specs = [
            ("eeg onset", "eeg_onset_epoch", "onset", "eeg"),
            ("eeg offset", "eeg_offset_epoch", "offset", "eeg"),
        ]
    else:
        raise ValueError(f"unsupported dataset {dataset!r}")

    markers = []
    for label, key, kind, ann in specs:
        ep = _to_float(row, key)
        if ep is None:
            continue
        markers.append({"label": label, "t_rel": ep - t0, "kind": kind, "ann": ann})
    return markers, t0


def window_margins(markers, *, pre_margin: float, post_margin: float,
                   default_post: float):
    """(pre_sec, post_sec) for extract_seizure_window so the window brackets
    every marker with the requested margins.

    pre_sec covers the EARLIEST onset annotation (eeg onset can precede the
    clinical t=0 reference → negative t_rel), post_sec covers the LATEST offset.
    Seizures with no offset annotation fall back to ``default_post``.
    """
    onset_rels = [m["t_rel"] for m in markers if m["kind"] == "onset"]
    offset_rels = [m["t_rel"] for m in markers if m["kind"] == "offset"]
    lo = min([0.0] + onset_rels)              # ≤ 0
    hi = max(offset_rels) if offset_rels else float(default_post)
    pre_sec = float(pre_margin) - lo          # lo ≤ 0 → ≥ pre_margin
    post_sec = hi + float(post_margin)
    return pre_sec, post_sec


def reference_shift(markers, ref_onset: str):
    """Display-origin shift (s) so the chosen onset reads as t=0, plus its label.

    Markers come in the loader frame (t=0 = clin onset for epilepsiae, eeg onset
    for yuquan). ``ref_onset='eeg'`` re-centers the DISPLAY on the electrographic
    onset by returning that marker's t_rel as the shift; it falls back to the
    loader frame (shift 0) when there is no eeg-onset annotation. This is purely
    cosmetic — the window/block/ordering math stays in the loader frame.
    """
    if ref_onset == "eeg":
        for m in markers:
            if m["ann"] == "eeg" and m["kind"] == "onset":
                return m["t_rel"], "eeg onset"
    return 0.0, "clin onset"


def duration_bits(markers):
    """Per-annotation seizure durations (offset−onset of the same source) as title
    strings. Computed in the loader frame so they stay TRUE durations regardless of
    which onset the display is centered on."""
    bits = []
    for ann in ("clin", "eeg"):
        ons = [m["t_rel"] for m in markers if m["ann"] == ann and m["kind"] == "onset"]
        offs = [m["t_rel"] for m in markers if m["ann"] == ann and m["kind"] == "offset"]
        if ons and offs:
            bits.append(f"{ann} dur≈{offs[0] - ons[0]:.1f}s")
    return bits


def robust_normalize_mad(sig: np.ndarray, *, eps: float = 1e-9) -> np.ndarray:
    """Per-channel robust z-score: (x - median) / (1.4826 * MAD). Last axis = time."""
    sig = np.asarray(sig, float)
    med = np.nanmedian(sig, axis=-1, keepdims=True)
    centered = sig - med
    mad = np.nanmedian(np.abs(centered), axis=-1, keepdims=True)
    scale = mad * 1.4826 + eps
    return centered / scale


def activity_envelope(norm_sig: np.ndarray, fs: float, *,
                      smooth_sec: float = 1.0) -> np.ndarray:
    """Cross-channel mean of |normalized signal|, boxcar-smoothed → 1D activity
    trace for the overview contact-sheet."""
    env = np.nanmean(np.abs(np.asarray(norm_sig, float)), axis=0)
    w = max(1, int(round(smooth_sec * fs)))
    if w <= 1:
        return env
    kernel = np.ones(w) / w
    return np.convolve(env, kernel, mode="same")


# ---------------------------------------------------------------------------
# Signal conditioning
# ---------------------------------------------------------------------------
def _bandpass(sig: np.ndarray, fs: float, *, lo: float = BP_LO,
              hi: float = BP_HI) -> np.ndarray:
    """Light bandpass for visual detection (drift + HF-noise removal). Falls back
    to median-detrend if the segment is too short for zero-phase filtering."""
    from scipy.signal import butter, sosfiltfilt

    nyq = 0.5 * fs
    hi_eff = min(hi, 0.9 * nyq)
    if not (0 < lo < hi_eff < nyq):
        return sig - np.nanmedian(sig, axis=-1, keepdims=True)
    sos = butter(4, [lo / nyq, hi_eff / nyq], btype="band", output="sos")
    clean = np.nan_to_num(sig, nan=0.0)
    try:
        return sosfiltfilt(sos, clean, axis=-1)
    except ValueError:  # segment shorter than filter padding
        return sig - np.nanmedian(sig, axis=-1, keepdims=True)


_BLK_CACHE: dict = {}


def _block_room(dataset: str, sid: str, row: dict, t0: float, results_root: Path):
    """Available (max_pre, max_post) seconds before/after t0 inside the seizure's
    block, from the block inventory. None if the block row can't be found.

    This lets us clamp pre/post INDEPENDENTLY to each block edge — a block that
    starts only ~8 s before onset must not force the post window (the seizure
    end) to shrink too."""
    key = (dataset, str(results_root))
    if key not in _BLK_CACHE:
        _, blk_csv = _resolve_inventory_paths(results_root, dataset=dataset)
        _BLK_CACHE[key] = _read_csv_rows(blk_csv)
    seizure_join = "block_id" if dataset == "epilepsiae" else "record"
    bid = row.get(seizure_join)
    for b in _BLK_CACHE[key]:
        if b["subject"] == sid and b["block_id"] == bid:
            return t0 - float(b["block_start_epoch"]), float(b["block_end_epoch"]) - t0
    return None, None


def _load_seizure_window(subject: str, idx: int, pre_sec: float, post_sec: float,
                         reference: str):
    """extract_seizure_window with a small rounding-safety retry. Callers should
    pre-clamp pre/post to the block room via _block_room so this rarely fires."""
    last_err = None
    for shrink in (1.0, 0.97, 0.9):
        try:
            return extract_seizure_window(
                subject, idx, pre_sec=pre_sec * shrink, post_sec=post_sec * shrink,
                reference=reference,
            )
        except ValueError as exc:
            last_err = exc
            continue
    print(f"    [skip] {subject} seizure {idx}: {last_err}")
    return None


def ordered_seizure_rows(dataset: str, sid: str, results_root: Path):
    """Seizure rows for ``sid`` in EXACTLY the order extract_seizure_window indexes
    them (so our enumerate idx == its seizure_idx). Mirrors its filter+sort."""
    sz_csv, _ = _resolve_inventory_paths(results_root, dataset=dataset)
    rows = [r for r in _read_csv_rows(sz_csv) if r["subject"] == sid]
    onset_field = "clin_onset_epoch" if dataset == "epilepsiae" else "eeg_onset_epoch"
    rows = [r for r in rows if r.get(onset_field)]
    rows.sort(key=lambda r: float(r[onset_field]))
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _draw_markers(ax, markers, *, panel_xlim, data_range, label=True):
    """Vertical onset/offset lines for one panel.

    A marker inside ``panel_xlim`` is a full vertical line. A marker outside the
    panel gets a colored edge arrow + time so it is never silently cropped. The
    ``[off-block]`` legend suffix is reserved for markers beyond the loaded
    recording ``data_range`` (e.g. an end that falls in the next block) — NOT for
    markers merely outside this panel's zoom (those are visible in the sibling
    panel). Legend lists this panel's in-view markers plus any off-block ones.
    Returns deduped legend handles."""
    from matplotlib.lines import Line2D

    plo, phi = panel_xlim
    dlo, dhi = data_range
    seen = set()
    handles = []
    for m in markers:
        color = ANN_COLOR.get(m["ann"], "k")
        style = KIND_STYLE.get(m["kind"], "-")
        lw = 1.7 if m["kind"] == "onset" else 1.3
        t = m["t_rel"]
        in_panel = plo <= t <= phi
        off_block = t < dlo - 1e-6 or t > dhi + 1e-6
        if in_panel:
            ax.axvline(t, color=color, ls=style, lw=lw, alpha=0.9, zorder=5)
        else:
            edge = plo if t < plo else phi
            arrow = "←" if t < plo else "→"
            ax.annotate(
                f"{arrow}{t:+.0f}s", xy=(edge, 1.0), xycoords=("data", "axes fraction"),
                ha=("left" if t < plo else "right"), va="bottom", fontsize=7,
                color=color, clip_on=False, annotation_clip=False,
            )
            ax.plot([edge], [1.0], marker=("<" if t < plo else ">"), color=color,
                    transform=ax.get_xaxis_transform(), clip_on=False, zorder=6,
                    markersize=7)
        if label and (in_panel or off_block) and m["label"] not in seen:
            seen.add(m["label"])
            suffix = " [off-block]" if off_block else ""
            handles.append(Line2D([], [], color=color, ls=style, lw=lw,
                                  label=f"{m['label']} ({t:+.1f}s){suffix}"))
    return handles


def _detail_windows(t, markers, *, post_margin):
    """Time sub-windows (label, lo, hi) for the detail panels. Short seizures get
    ONE full-span panel; long seizures (>SINGLE_MAX_SPAN) split into an onset-zoom
    and an end-zoom so the onset/offset transitions stay readable instead of being
    crushed into a too-wide axis. If the offset is off-block/missing, the second
    panel zooms the recording end instead."""
    lo, hi = float(t[0]), float(t[-1])
    if hi - lo <= SINGLE_MAX_SPAN:
        return [("onset + end view", lo, hi)]
    # Anchor the onset zoom on the EARLIEST onset annotation (the electrographic /
    # eeg onset when present — it precedes and is more precise than clinical onset),
    # and stretch to also include the latest onset so both markers stay visible.
    onsets = [m["t_rel"] for m in markers if m["kind"] == "onset"]
    a = min(onsets) if onsets else 0.0
    b = max(onsets) if onsets else 0.0
    onset_lo = max(lo, a - ONSET_ZOOM_PRE)
    onset_hi = min(hi, max(a + ONSET_ZOOM_POST, b + ONSET_ZOOM_PRE))
    wins = [("onset zoom", onset_lo, onset_hi)]
    offs = [m["t_rel"] for m in markers
            if m["kind"] == "offset" and lo <= m["t_rel"] <= hi]
    if offs:
        o = max(offs)
        wins.append(("end zoom", max(lo, o - END_ZOOM_PRE), min(hi, o + post_margin)))
    else:
        wins.append(("recording-end zoom",
                     max(lo, hi - (END_ZOOM_PRE + post_margin)), hi))
    return wins


def plot_seizure_detail(sw, t, markers, soz_flags, *, dataset, ref_label, dur_bits,
                        out_path, post_margin):
    """All channels stacked, MAD-normalized, with onset/offset lines. Clinician-SOZ
    channels are drawn in green (traces + labels). One panel for short seizures;
    onset-zoom + end-zoom columns for long ones (_detail_windows). ``t`` and
    ``markers`` are pre-shifted so t=0 = ``ref_label``."""
    from matplotlib.lines import Line2D

    sig = _bandpass(np.asarray(sw.signal, float), sw.fs)
    Y = np.clip(robust_normalize_mad(sig), -CLIP, CLIP)
    n_ch = Y.shape[0]
    t = np.asarray(t, float)
    ch_names = list(sw.ch_names)
    yoff = np.array([(n_ch - 1 - i) * GAP for i in range(n_ch)])  # ch0 at TOP
    ch_color = [SOZ_COLOR if soz_flags[i] else NONSOZ_COLOR for i in range(n_ch)]
    any_soz = any(soz_flags)

    wins = _detail_windows(t, markers, post_margin=post_margin)
    ncols = len(wins)
    height = max(6.0, min(0.165 * n_ch + 2.0, 40.0))
    panel_w = 14.0 if ncols == 1 else 8.6   # wide single panel; narrower zoom columns
    fig, axes = plt.subplots(1, ncols, figsize=(panel_w * ncols, height),
                             squeeze=False, sharey=True)
    fig.patch.set_facecolor("white")
    name_fs = 6 if n_ch <= 48 else (4 if n_ch <= 96 else 3)

    for c, (wlabel, wlo, whi) in enumerate(wins):
        ax = axes[0][c]
        sel = (t >= wlo) & (t <= whi)
        ts = t[sel]
        for i in range(n_ch):
            ax.plot(ts, Y[i, sel] + yoff[i], lw=0.4, color=ch_color[i],
                    zorder=(3 if soz_flags[i] else 2))
        if c == 0:
            for i in range(n_ch):
                ax.text(wlo, yoff[i], f"{ch_names[i]} ", va="center", ha="right",
                        fontsize=name_fs,
                        color=(SOZ_COLOR if soz_flags[i] else "0.15"),
                        fontweight=("bold" if soz_flags[i] else "normal"),
                        clip_on=False)
        handles = _draw_markers(ax, markers, panel_xlim=(wlo, whi),
                                data_range=(t[0], t[-1]))
        if any_soz:
            handles.append(Line2D([], [], color=SOZ_COLOR, lw=1.2,
                                  label="clinician SOZ channel"))
        ax.set_xlim(wlo, whi)
        ax.set_ylim(-GAP, (n_ch - 1) * GAP + GAP)
        ax.set_yticks([])
        ax.set_xlabel("time rel. onset reference (s)", fontsize=11)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10, steps=[1, 2, 5, 10]))
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.grid(axis="x", which="major", color="0.6", lw=0.6, alpha=0.5)
        ax.grid(axis="x", which="minor", color="0.8", lw=0.4, alpha=0.4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if ncols > 1:
            ax.set_title(wlabel, fontsize=11)
        if handles:
            ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle(
        f"{dataset}/{sw.subject.split('/')[-1]}  seizure {sw.seizure_id}  "
        f"(t=0 @ {ref_label}, {DATASET_REFERENCE[dataset]} montage, {n_ch} ch)"
        + ("   " + ", ".join(dur_bits) if dur_bits else ""),
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_subject_overview(subject_label, entries, *, dataset, out_path):
    """Contact-sheet: one small panel per seizure showing how much electrical
    activity (cross-channel mean |MAD-z|) there is over time, with onset/offset
    lines. A green overlay is the same average over ONLY the clinician-SOZ
    channels — lets you scan whether SOZ activity leads/aligns with the labels
    and which seizures' labels look off, before opening the full detail figure."""
    n = len(entries)
    if n == 0:
        return
    any_soz = any(ent.get("soz_env") is not None for ent in entries)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.4 * nrows),
                             squeeze=False)
    fig.patch.set_facecolor("white")
    for k, ent in enumerate(entries):
        ax = axes[k // ncols][k % ncols]
        ax.plot(ent["t"], ent["env"], lw=0.9, color="0.25", zorder=2,
                label="all channels")
        if ent.get("soz_env") is not None:
            ax.plot(ent["t"], ent["soz_env"], lw=1.0, color=SOZ_COLOR, zorder=3,
                    label="SOZ channels")
        rng = (float(ent["t"][0]), float(ent["t"][-1]))
        _draw_markers(ax, ent["markers"], panel_xlim=rng, data_range=rng, label=False)
        ax.set_xlim(*rng)
        ax.set_ylim(bottom=0)
        ax.set_title(f"sz {ent['idx']}  {ent['seizure_id']}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(axis="x", color="0.85", lw=0.4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if k == 0 and any_soz:
            ax.legend(loc="upper right", fontsize=6, framealpha=0.85)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    soz_note = "; green=SOZ-only mean" if any_soz else ""
    fig.suptitle(
        f"{subject_label}: per-seizure activity envelope "
        f"(cross-channel mean |MAD-z|{soz_note}; crimson=clin, blue=eeg, "
        f"solid=onset, dashed=offset)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def render_subject(dataset: str, sid: str, *, pre_margin: float, post_margin: float,
                   default_post: float, ref_onset: str, do_detail: bool,
                   do_overview: bool, overwrite: bool, max_seizures: Optional[int],
                   results_root: Path):
    subject = f"{dataset}/{sid}"
    reference = DATASET_REFERENCE[dataset]
    rows = ordered_seizure_rows(dataset, sid, results_root)
    if max_seizures is not None:
        rows = rows[:max_seizures]
    if not rows:
        print(f"  {subject}: no seizures with onset epoch — skip")
        return
    out_dir = OUT_ROOT / dataset / sid
    ov_path = out_dir / f"{sid}__overview.png"
    overview_needed = do_overview and (overwrite or not ov_path.exists())
    soz_set = _soz_set(dataset, sid)
    print(f"  {subject}: {len(rows)} seizures, {len(soz_set)} SOZ contact(s) -> {out_dir}")

    entries = []
    for idx, row in enumerate(rows):
        markers, t0 = seizure_marker_times(row, dataset)
        detail_path = out_dir / f"{sid}__sz{idx:02d}_{row['seizure_id']}.png"
        need_detail = do_detail and (overwrite or not detail_path.exists())
        # The overview needs every seizure's envelope, so load unless nothing
        # here is wanted (cheap reruns skip already-rendered subjects entirely).
        if not need_detail and not overview_needed:
            continue

        pre_sec, post_sec = window_margins(
            markers, pre_margin=pre_margin, post_margin=post_margin,
            default_post=default_post,
        )
        # Clamp each side to its block edge independently (a tight pre side must
        # not cut the seizure end off the post side).
        max_pre, max_post = _block_room(dataset, sid, row, t0, results_root)
        if max_pre is not None:
            if pre_sec > max_pre - 0.05:
                pre_sec = max(0.0, max_pre - 0.05)
            if post_sec > max_post - 0.05:
                post_sec = max(0.0, max_post - 0.05)
        sw = _load_seizure_window(subject, idx, pre_sec, post_sec, reference)
        if sw is None:
            continue
        # Loud guard against inventory-ordering drift vs extract_seizure_window.
        if abs(sw.clin_onset_epoch - t0) > 1e-3:
            raise RuntimeError(
                f"{subject} idx {idx}: onset epoch mismatch "
                f"(inventory {t0} vs loaded {sw.clin_onset_epoch}) — ordering drift"
            )

        soz_flags = soz_channel_flags(sw.ch_names, soz_set)
        # Re-center the DISPLAY on the chosen onset (window/block math above stays
        # in the loader frame). dur_bits are TRUE per-source durations, frame-free.
        shift, ref_label = reference_shift(markers, ref_onset)
        dur_bits = duration_bits(markers)
        t_disp = np.asarray(sw.t_axis, float) - shift
        markers_disp = [{**m, "t_rel": m["t_rel"] - shift} for m in markers]
        if need_detail:
            plot_seizure_detail(sw, t_disp, markers_disp, soz_flags, dataset=dataset,
                                ref_label=ref_label, dur_bits=dur_bits,
                                out_path=detail_path, post_margin=post_margin)
        if overview_needed:
            norm = robust_normalize_mad(_bandpass(np.asarray(sw.signal, float), sw.fs))
            env = activity_envelope(norm, sw.fs)
            soz_mask = np.asarray(soz_flags, bool)
            soz_env = activity_envelope(norm[soz_mask], sw.fs) if soz_mask.any() else None
            entries.append({
                "idx": idx, "seizure_id": row["seizure_id"],
                "t": t_disp, "env": env, "soz_env": soz_env, "markers": markers_disp,
            })

    if overview_needed and entries:
        plot_subject_overview(f"{dataset}/{sid}", entries,
                              dataset=dataset, out_path=ov_path)


def _all_subjects(dataset: str, results_root: Path):
    sz_csv, _ = _resolve_inventory_paths(results_root, dataset=dataset)
    onset_field = "clin_onset_epoch" if dataset == "epilepsiae" else "eeg_onset_epoch"
    subs = sorted({r["subject"] for r in _read_csv_rows(sz_csv) if r.get(onset_field)})
    return subs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["epilepsiae", "yuquan"])
    ap.add_argument("--subject", action="append", default=[],
                    help="subject id (repeatable); omit with --all for whole dataset")
    ap.add_argument("--all", action="store_true", help="all subjects in --dataset")
    ap.add_argument("--all-datasets", action="store_true",
                    help="every subject of BOTH datasets")
    ap.add_argument("--pre-margin", type=float, default=30.0)
    ap.add_argument("--post-margin", type=float, default=30.0)
    ap.add_argument("--default-post", type=float, default=120.0,
                    help="post window (s) for seizures with no offset annotation")
    ap.add_argument("--ref-onset", choices=["eeg", "clin"], default="eeg",
                    help="display t=0 reference (epilepsiae): eeg (electrographic, "
                         "default) or clin onset; yuquan is always eeg")
    ap.add_argument("--no-detail", action="store_true")
    ap.add_argument("--no-overview", action="store_true")
    ap.add_argument("--max-seizures", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    results_root = _ROOT / "results"
    jobs = []  # (dataset, sid)
    if args.all_datasets:
        for ds in ("epilepsiae", "yuquan"):
            jobs += [(ds, s) for s in _all_subjects(ds, results_root)]
    else:
        if not args.dataset:
            ap.error("need --dataset (with --subject/--all) or --all-datasets")
        if args.all:
            jobs += [(args.dataset, s) for s in _all_subjects(args.dataset, results_root)]
        elif args.subject:
            jobs += [(args.dataset, s) for s in args.subject]
        else:
            ap.error("need --subject or --all")

    print(f"[seizure-onset-review] {len(jobs)} subject(s)")
    for ds, sid in jobs:
        try:
            render_subject(
                ds, sid, pre_margin=args.pre_margin, post_margin=args.post_margin,
                default_post=args.default_post, ref_onset=args.ref_onset,
                do_detail=not args.no_detail, do_overview=not args.no_overview,
                overwrite=args.overwrite, max_seizures=args.max_seizures,
                results_root=results_root,
            )
        except Exception as exc:  # one bad subject must not kill a long batch
            print(f"  [ERROR] {ds}/{sid}: {type(exc).__name__}: {exc}")
    print("[seizure-onset-review] done")


if __name__ == "__main__":
    main()
