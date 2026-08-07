"""Paper-ready subject-specific SNN KMeans panel (Topic 4 Fig4B).

Companion to ``plot_fig_subject_snn.py``. Consumes one or more fixed-parameter
spontaneous twoend E1146 seed readouts and draws the KMeans=2 rank diagnostic in the
**mature canonical style** -- the same plotting functions that produce
``results/interictal_propagation_masked/figures/per_subject/<dataset>_<subject>_propagation.png``
(`_plot_rank_histogram` / `_plot_rank_heatmap` + `_plot_cluster_boundaries` /
`_plot_cluster_rank_fig4` from scripts/plot_interictal_propagation.py). No simulation rerun.

Formal Figure 4 keeps the original single-row four-block layout:

  TA/TB-clustered heatmap | rank distribution | mean-rank profiles | model-vs-real matrix

The first three blocks reuse the locked Figure-1 plotting grammar; the fourth
retains the previously accepted model-vs-real validation panel.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/topic4_sef_hfo/field_swap_subject_snn"

sys.path.insert(0, str(ROOT))
from src.interictal_propagation import compute_adaptive_cluster_stereotypy  # noqa: E402
# MATURE canonical heatmap plotter (same function that draws the per_subject propagation fig).
# Formal Figure 4 uses the exact Figure-1 canonical ridgeline helper for per-channel rank;
# legacy/preview layouts retain _hist_aligned.  The template profile keeps the same line + SD-band
# grammar while using direction-aware TA/TB colours.
from scripts.plot_interictal_propagation import (  # noqa: E402
    _plot_cluster_boundaries, _plot_rank_heatmap, _plot_rank_histogram)
# model-vs-real-template similarity matrix (rightmost panel) + real-template loader.
from scripts.paper_figures.plot_fig_subject_snn_realvsmodel import (  # noqa: E402
    _real_templates, _sim_matrix, _matrix_panel)


TA_COLOR = "#d62728"
TB_COLOR = "#1f77b4"
TA_SUB_COLORS = [TA_COLOR, "#f28e8c", "#b2182b"]
TB_SUB_COLORS = [TB_COLOR, "#6baed6", "#08519c"]
_CL_COLORS = [TA_COLOR, TB_COLOR, "#2ca02c", "#ff7f0e"]
MIN_DIR_EVENTS = 3   # both forward AND reverse need >= this many events for the fwd-rev x t_a/t_b matrix


def _hist_aligned(
    ax,
    R,
    max_rank,
    names,
    viridis_n,
    *,
    y_centers=None,
    ylim=None,
    show_ylabels=True,
):
    """Per-channel rank histogram on the heatmap's 1-unit coordinate (channel display-row i in band
    [i,i+1]) so its y-axis aligns with the heatmap + cluster panels. Canonical viridis-per-channel look."""
    n_ch = R.shape[0]
    y_centers = np.asarray(y_centers if y_centers is not None else np.arange(n_ch) + 0.5, dtype=float)
    row_step = float(np.nanmedian(np.abs(np.diff(np.sort(y_centers))))) if n_ch > 1 else 1.0
    for i in range(n_ch):
        vals = R[i][np.isfinite(R[i])]
        if vals.size == 0:
            continue
        hist, _ = np.histogram(vals, bins=np.arange(max_rank + 2) - 0.5)
        h = hist / max(1, vals.size) * row_step * 0.74
        ax.bar(np.arange(max_rank + 1), h, bottom=y_centers[i] - h / 2.0, width=1.0,
               color=plt.cm.viridis(i / max(1, viridis_n - 1)), alpha=0.8, linewidth=0)
        ax.axhline(y_centers[i], color="0.9", lw=0.3)
    ax.set_ylim(ylim if ylim is not None else (0, n_ch))
    ax.set_yticks(y_centers)
    ax.set_yticklabels(names if show_ylabels else [])
    if not show_ylabels:
        ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_xlim(-0.5, max_rank + 0.5); ax.set_xlabel("Rank")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _cluster_aligned(
    ax,
    R,
    labels,
    max_rank,
    names,
    *,
    y_centers=None,
    ylim=None,
    cluster_display=None,
    cluster_order=None,
    show_ylabels=True,
):
    """Per-cluster mean+/-std rank line on the heatmap's 1-unit coordinate (channel center i+0.5)."""
    n_ch = R.shape[0]
    y = np.asarray(y_centers if y_centers is not None else np.arange(n_ch) + 0.5, dtype=float)
    order = cluster_order if cluster_order is not None else sorted(set(labels.tolist()))
    for k, cid in enumerate(order):
        mask = labels == cid
        mean = np.full(n_ch, np.nan); std = np.full(n_ch, np.nan)
        for i in range(n_ch):
            v = R[i, mask]; v = v[np.isfinite(v)]
            if v.size:
                mean[i] = v.mean(); std[i] = v.std()
        fin = np.isfinite(mean)
        display = (cluster_display or {}).get(int(cid), {})
        col = display.get("color", _CL_COLORS[k % len(_CL_COLORS)])
        label = display.get("label", f"C{int(cid)}")
        ax.fill_betweenx(y[fin], (mean - std)[fin], (mean + std)[fin], color=col, alpha=0.15, lw=0)
        ax.plot(mean[fin], y[fin], "-o", color=col, lw=2.0, ms=5, label=f"{label} (n={int(mask.sum())})")
    ax.set_ylim(ylim if ylim is not None else (0, n_ch)); ax.set_yticks(y)
    ax.set_yticklabels(names if show_ylabels else [])
    if not show_ylabels:
        ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_xlim(-0.5, max_rank + 0.5); ax.set_xlabel("Rank")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _load(tag: str):
    return (json.load(open(RUN / f"readout_{tag}.json")),
            np.load(RUN / f"figdata_{tag}.npz", allow_pickle=True))


def _duration_ms(figdata):
    times = np.asarray(figdata["times"], dtype=float)
    if times.size < 2:
        return float(times[-1]) if times.size else float("nan")
    return float(times[-1] + np.median(np.diff(times)))


def _validate_seed_pool(loaded):
    """Fail loudly if nominal seed replicates do not share one model/readout contract."""
    if not loaded:
        raise ValueError("at least one input tag is required")
    tag0, readout0, figdata0 = loaded[0]
    subject0 = str(readout0.get("subject") or "_".join(tag0.split("_")[:2]))
    names0 = [str(x) for x in figdata0["names"]]
    contacts0 = np.asarray(figdata0["contacts"], dtype=float)
    foci0 = np.asarray(figdata0["foci"], dtype=float)
    scalar_keys = ("L", "core_r", "core_mean", "theta_deg")
    seeds = []
    for tag, readout, figdata in loaded:
        subject = str(readout.get("subject") or "_".join(tag.split("_")[:2]))
        if subject != subject0:
            raise RuntimeError(f"mixed subjects in seed pool: {subject0} vs {subject} ({tag})")
        if readout.get("lesion") != readout0.get("lesion"):
            raise RuntimeError(f"mixed lesion contracts in seed pool: {tag}")
        if readout.get("placement") != readout0.get("placement"):
            raise RuntimeError(f"mixed placement contracts in seed pool: {tag}")
        if int(readout.get("k_dir", 2)) != int(readout0.get("k_dir", 2)):
            raise RuntimeError(f"mixed k_dir contracts in seed pool: {tag}")
        if [str(x) for x in figdata["names"]] != names0:
            raise RuntimeError(f"contact-name/order mismatch in seed pool: {tag}")
        if not np.allclose(np.asarray(figdata["contacts"], float), contacts0):
            raise RuntimeError(f"contact geometry mismatch in seed pool: {tag}")
        if not np.allclose(np.asarray(figdata["foci"], float), foci0):
            raise RuntimeError(f"core-focus mismatch in seed pool: {tag}")
        for key in scalar_keys:
            if not np.isclose(float(figdata[key]), float(figdata0[key])):
                raise RuntimeError(f"{key} mismatch in seed pool: {tag}")
        if not np.isclose(_duration_ms(figdata), _duration_ms(figdata0)):
            raise RuntimeError(f"simulation-duration mismatch in seed pool: {tag}")
        seeds.append(int(readout["seed"]))
    if len(set(seeds)) != len(seeds):
        raise RuntimeError(f"duplicate seeds in seed pool: {seeds}")
    return subject0


def _seed_stratified_direction_permutation(labels, signs, seed_ids, n_perm=10_000):
    """Association P while preserving each seed's direction counts."""
    labels = np.asarray(labels, dtype=int)
    signs = np.asarray(signs, dtype=float)
    seed_ids = np.asarray(seed_ids, dtype=int)
    observed, _ = _direction_purity(labels, signs)
    rng = np.random.default_rng(20260721)
    exceed = 0
    for _ in range(int(n_perm)):
        shuffled = signs.copy()
        for seed in np.unique(seed_ids):
            idx = np.flatnonzero(seed_ids == seed)
            shuffled[idx] = rng.permutation(shuffled[idx])
        null_purity, _ = _direction_purity(labels, shuffled)
        exceed += int(null_purity >= observed - 1e-12)
    return float((exceed + 1) / (int(n_perm) + 1))


def _loso_cluster_statistics(ranks, bools, signs, seed_ids, names, min_participation):
    rows = []
    for held_out in sorted(set(np.asarray(seed_ids, dtype=int).tolist())):
        keep = np.asarray(seed_ids, dtype=int) != held_out
        if int(np.sum(keep)) < 2:
            continue
        fit = compute_adaptive_cluster_stereotypy(
            ranks[:, keep], bools[:, keep], names, k_range=(2, 2),
            use_masked_features=True, min_participation=min_participation,
            n_sample=60, n_tau_seeds=2,
        )
        lab = np.asarray(fit.get("labels", []), dtype=int)
        if lab.shape != (int(np.sum(keep)),):
            continue
        purity, conf = _direction_purity(lab, np.asarray(signs)[keep])
        shared, shared_n = _shared_corr(ranks[:, keep], lab)
        rows.append({
            "held_out_seed": int(held_out),
            "n_events": int(np.sum(keep)),
            "direction_purity": float(purity),
            "within_cluster_tau_mean": float(fit.get("within_cluster_tau_mean", np.nan)),
            "shared_overlap_corr": float(shared),
            "shared_overlap_n_channels": int(shared_n),
            "direction_confusion": conf.tolist(),
        })
    return rows


def _rank_matrix(events, names):
    ranks = np.full((len(names), len(events)), np.nan, dtype=float)
    for j, ev in enumerate(events):
        ev_ranks = ev.get("ranks") or {}
        for i, name in enumerate(names):
            v = ev_ranks.get(name)
            if v is not None:
                ranks[i, j] = float(v)
    return ranks, np.isfinite(ranks)


def _axis_order(fd, names):
    all_names = [str(x) for x in fd["names"]]
    contacts = np.asarray(fd["contacts"], dtype=float)
    reg = fd["reg"].item()
    center = np.asarray(reg["center"], dtype=float); axis = np.asarray(reg["axis_unit"], dtype=float)
    idx = [all_names.index(n) for n in names]
    proj = (contacts[idx] - center) @ axis
    return [names[i] for i in np.argsort(proj)]


def _direction_purity(labels, signs):
    clusters = sorted(set(labels.tolist()))
    conf = np.zeros((len(clusters), 2), dtype=int)
    for row, c in enumerate(clusters):
        s = signs[labels == c]
        conf[row, 0] = int(np.sum(s > 0)); conf[row, 1] = int(np.sum(s < 0))
    if conf.shape == (2, 2):
        purity = max(conf[0, 0] + conf[1, 1], conf[0, 1] + conf[1, 0]) / max(conf.sum(), 1)
    else:
        purity = float(np.max(conf, axis=1).sum() / max(conf.sum(), 1))
    return float(purity), conf


def _cluster_display_from_direction(labels, conf):
    """Map unsupervised cluster ids to template names for display.

    In this panel the model templates are defined by event direction:
    forward-majority cluster -> t_a, reverse-majority cluster -> t_b.
    If KMeans only splits one direction, display the clusters as same-template
    subclusters (for example, t_b-1/t_b-2) instead of faking a t_a/t_b pair.
    """
    clusters = sorted(set(labels.tolist()))
    majority = []
    for row, cid in enumerate(clusters):
        is_forward = conf[row, 0] >= conf[row, 1]
        majority.append((int(cid), "forward" if is_forward else "reverse"))
    dir_counts = {
        direction: sum(1 for _, d in majority if d == direction)
        for direction in ("forward", "reverse")
    }
    dir_seen = {"forward": 0, "reverse": 0}
    display = {}
    for cid, direction in majority:
        dir_seen[direction] += 1
        is_forward = direction == "forward"
        base = "TA" if is_forward else "TB"
        if dir_counts[direction] > 1:
            label = f"{base}-{dir_seen[direction]}"
            colors = TA_SUB_COLORS if is_forward else TB_SUB_COLORS
            color = colors[(dir_seen[direction] - 1) % len(colors)]
        else:
            label = base
            color = TA_COLOR if is_forward else TB_COLOR
        display[int(cid)] = {
            "label": label,
            "color": color,
            "direction": direction,
        }
    return display


def _shared_corr(ranks, labels):
    clusters = sorted(set(labels.tolist()))
    if len(clusters) != 2:
        return float("nan"), 0

    def mean_rank(mask):
        out = np.full(ranks.shape[0], np.nan)
        for i in range(ranks.shape[0]):
            v = ranks[i, mask]; v = v[np.isfinite(v)]
            if v.size:
                out[i] = float(np.mean(v))
        return out
    m0, m1 = mean_rank(labels == clusters[0]), mean_rank(labels == clusters[1])
    shared = np.isfinite(m0) & np.isfinite(m1)
    if int(shared.sum()) < 3:
        return float("nan"), int(shared.sum())
    return float(spearmanr(m0[shared], m1[shared]).correlation), int(shared.sum())


def _cluster_display_order(labels, cluster_display):
    """Return cluster ids in reader-facing template order.

    Raw KMeans ids are arbitrary. For the figure, show template semantics:
    t_a/forward first, then t_b/reverse, with same-template subclusters kept
    in their numeric suffix order. Metadata still stores the raw C ids.
    """
    def key(cid):
        info = cluster_display[int(cid)]
        label = info["label"]
        direction_rank = 0 if info["direction"] == "forward" else 1
        suffix = 0
        if "-" in label:
            try:
                suffix = int(label.rsplit("-", 1)[1])
            except ValueError:
                suffix = 0
        return (direction_rank, suffix, int(cid))

    return sorted(set(labels.tolist()), key=key)


def _formal_row_axes(fig, outer, row):
    """Figure-1-identical heatmap / colorbar / right-summary row geometry."""
    left = outer[row, 0].subgridspec(2, 1, height_ratios=[20, 1], hspace=0.06)
    colorbar_column = outer[row, 1].subgridspec(2, 1, height_ratios=[20, 1], hspace=0.06)
    right = outer[row, 2].subgridspec(2, 1, height_ratios=[20, 1], hspace=0.06)
    ax_main = fig.add_subplot(left[0])
    ax_bottom = fig.add_subplot(left[1], sharex=ax_main)
    ax_cbar = fig.add_subplot(colorbar_column[0])
    ax_cbar_dummy = fig.add_subplot(colorbar_column[1])
    ax_cbar_dummy.axis("off")
    ax_right = fig.add_subplot(right[0])
    ax_right_dummy = fig.add_subplot(right[1], sharex=ax_right)
    ax_right_dummy.axis("off")
    return ax_main, ax_bottom, ax_cbar, ax_right


def _formal_colorbar(fig, image, cax):
    cbar = fig.colorbar(image, cax=cax, orientation="vertical")
    cbar.set_label("First → Last", fontsize=11.5)
    cbar.ax.tick_params(labelsize=9.5, length=2.5)
    return cbar


def _plot_direction_strip(ax, signs):
    """Model analogue of Figure 1's day/night strip; uses observed event direction."""
    signs = np.asarray(signs, float)
    rgba = np.empty((1, len(signs), 4), float)
    rgba[0, signs > 0] = matplotlib.colors.to_rgba(TA_COLOR)
    rgba[0, signs < 0] = matplotlib.colors.to_rgba(TB_COLOR)
    rgba[0, signs == 0] = matplotlib.colors.to_rgba("0.75")
    ax.imshow(
        rgba,
        aspect="auto",
        interpolation="nearest",
        extent=[0.0, float(len(signs)), 0.0, 1.0],
        origin="lower",
    )
    ax.set_xlim(0.0, float(len(signs)))
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(
        "Model events (time-ordered)  ·  strip: forward (red) / reverse (blue)",
        fontsize=11.5,
        labelpad=2.0,
    )


def _compose_formal_standard(
    *,
    tag,
    fig_name,
    output_stem,
    panel_letter,
    min_participation,
    display_min_channel_frac,
    dropped_display_channels,
    readout,
    events,
    ranks,
    bools,
    ordered_names,
    signs,
    labels,
    cluster_display,
    purity,
    conf,
    active_inter,
    shared_corr,
    shared_n,
    within_tau,
    result,
    n_fwd,
    n_rev,
    bidirectional,
    Msim,
    Psim,
):
    """Render formal Panel C with the locked Figure-1 two-row visual grammar."""
    n_ch, n_ev = ranks.shape
    if not np.all(np.diff([float(ev["t_on"]) for ev in events]) >= 0):
        raise ValueError("formal time-ordered heatmap requires chronological readout events")
    channel_order = np.arange(n_ch)
    cluster_order = _cluster_display_order(labels, cluster_display)
    ev_order = np.concatenate([np.where(labels == cid)[0] for cid in cluster_order])
    display_id = {int(cid): i for i, cid in enumerate(cluster_order)}
    labels_sorted = np.asarray([display_id[int(cid)] for cid in labels[ev_order]], dtype=int)

    fig = plt.figure(figsize=(12.4, 7.25), facecolor="white")
    outer = fig.add_gridspec(
        2,
        3,
        width_ratios=[8.4, 0.18, 1.42],
        height_ratios=[1.0, 1.0],
        hspace=0.24,
        wspace=0.14,
        left=0.070,
        right=0.985,
        bottom=0.075,
        top=0.925,
    )

    # Top row: the same clean events in chronological order.
    ax_time, ax_strip, cax_time, ax_dist = _formal_row_axes(fig, outer, 0)
    im_time = _plot_rank_heatmap(
        ax_time,
        ranks,
        ordered_names,
        "",
        show_ylabels=True,
        display_bools=bools,
        ytick_fontsize=10.5,
        xtick_fontsize=9.0,
    )
    ax_time.tick_params(axis="x", labelbottom=False)
    ax_time.set_xlabel("")
    ax_time.text(
        0.0,
        1.075,
        f"Model E1146  |  n={n_ev}",
        transform=ax_time.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.0,
        fontweight="bold",
    )
    _plot_direction_strip(ax_strip, signs)
    _plot_rank_histogram(
        ax_dist,
        ranks,
        bools,
        np.arange(n_ev),
        channel_order,
        ordered_names,
        "Rank dist.",
        show_ylabels=False,
        label_fontsize=11.5,
        title_fontsize=11.0,
        xtick_fontsize=9.5,
    )
    _formal_colorbar(fig, im_time, cax_time)

    # Bottom row: identical events sorted into direction-aligned KMeans groups.
    ax_cluster, ax_cluster_dummy, cax_cluster, ax_profile = _formal_row_axes(fig, outer, 1)
    ax_cluster_dummy.axis("off")
    im_cluster = _plot_rank_heatmap(
        ax_cluster,
        ranks[:, ev_order],
        ordered_names,
        "",
        show_ylabels=True,
        display_bools=bools[:, ev_order],
        ytick_fontsize=10.5,
        xtick_fontsize=9.0,
    )
    boundary = int(np.sum(labels_sorted == 0))
    gap_half = 0.18
    ax_cluster.axvspan(
        boundary - gap_half,
        boundary + gap_half,
        facecolor="white",
        edgecolor="0.60",
        hatch="////",
        linewidth=0.0,
        zorder=12,
    )
    _plot_cluster_boundaries(
        ax_cluster,
        labels_sorted,
        n_ch,
        line_color="0.55",
        line_width=0.0,
        line_style="-",
        label_fontsize=11.0,
        label_box=False,
        boundary_band=False,
        label_names=[cluster_display[int(cid)]["label"] for cid in cluster_order],
        label_y_offset=0.42,
    )
    for txt in ax_cluster.texts:
        for info in cluster_display.values():
            if txt.get_text().startswith(info["label"]):
                txt.set_color(info["color"])
    ax_cluster.set_xlabel("Model events (clustered)", fontsize=11.5)
    ax_cluster.tick_params(axis="x", labelsize=9.0)
    heat_ylim = ax_cluster.get_ylim()
    heat_ticks = ax_cluster.get_yticks()
    _cluster_aligned(
        ax_profile,
        ranks,
        labels,
        n_ch - 1,
        ordered_names,
        y_centers=heat_ticks,
        ylim=heat_ylim,
        cluster_display=cluster_display,
        cluster_order=cluster_order,
        show_ylabels=False,
    )
    ax_profile.tick_params(axis="x", labelsize=9.5)
    ax_profile.xaxis.label.set_fontsize(11.5)
    _formal_colorbar(fig, im_cluster, cax_cluster)

    if panel_letter:
        fig.text(0.012, 0.955, str(panel_letter), fontsize=22, fontweight="bold")

    outdir = ROOT / f"results/paper-ready-figure/{fig_name}/figures"
    outdir.mkdir(parents=True, exist_ok=True)
    stem = output_stem or f"{fig_name}_kmeans2"
    png, pdf = outdir / f"{stem}.png", outdir / f"{stem}.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metadata = {
        "figure": stem,
        "companion_to": fig_name,
        "input_tag": tag,
        "preview_style": False,
        "formal_layout": True,
        "panel_letter": panel_letter,
        "layout": "Figure-1 standard: time-ordered heatmap + direction strip + rank distribution above; TA/TB-clustered heatmap + mean-rank profiles below",
        "plotters": "canonical _plot_rank_heatmap, _plot_rank_histogram, _plot_cluster_boundaries; direction-aware line+SD mean-rank profiles",
        "event_filter": f"sign is not None and n_part >= 2*k_dir ({2 * int(readout.get('k_dir', 2))})",
        "time_order": [float(ev["t_on"]) for ev in events],
        "top_strip": "observed model event direction: forward red, reverse blue; no fabricated day/night labels",
        "display_min_channel_frac": float(display_min_channel_frac),
        "dropped_display_channels": dropped_display_channels,
        "n_events": n_ev,
        "n_forward": n_fwd,
        "n_reverse": n_rev,
        "bidirectional": bidirectional,
        "min_dir_events_gate": MIN_DIR_EVENTS,
        "channels_displayed": ordered_names,
        "kmeans": {
            "k": 2,
            "min_participation": int(min_participation),
            "labels": labels.tolist(),
            "cluster_sizes": {f"C{c}": int(np.sum(labels == c)) for c in sorted(set(labels.tolist()))},
            "display_cluster_order": [f"C{int(c)}" for c in cluster_order],
            "display_labels": {
                f"C{c}": {
                    "label": cluster_display[int(c)]["label"],
                    "color": cluster_display[int(c)]["color"],
                    "direction": cluster_display[int(c)]["direction"],
                }
                for c in sorted(set(labels.tolist()))
            },
            "direction_confusion_rows_cluster_cols_forward_reverse": conf.tolist(),
            "direction_purity": purity,
            "within_cluster_tau_mean": within_tau,
            "active_inter_cluster_corr": active_inter,
            "shared_overlap_corr": shared_corr,
            "shared_overlap_n_channels": shared_n,
            "candidate_forward_reverse_pairs": result.get("candidate_forward_reverse_pairs", []),
        },
        "similarity_matrix_panel": ({
            "valid": True,
            "shown_on_main_canvas": False,
            "reason_not_shown": "formal Panel C follows the locked Figure-1 2x2 visual contract",
            "model_templates_built_by": "event sign (forward = sign>0 events, reverse = sign<0 events), NOT cluster label",
            "rows_model_forward_reverse_cols_data_forward_reverse_spearman": Msim.tolist(),
            "directional_perm_p": Psim.tolist(),
            "perm_p_direction": "diagonal one-sided positive, off-diagonal one-sided negative (swap-predicted)",
        } if bidirectional else {
            "valid": False,
            "shown_on_main_canvas": False,
            "reason": f"one-direction readout (fwd={n_fwd}, rev={n_rev}); need >= {MIN_DIR_EVENTS} each",
        }),
        "notes": [
            "Plotting-only; no SNN rerun.",
            "The same 14 clean directional events appear in both rows; only their column order changes.",
            "The thin top strip encodes model forward/reverse direction because simulation events have no day/night state.",
            "The model-vs-real similarity matrix is retained numerically in metadata and removed from the formal Panel-C canvas.",
        ],
    }
    (outdir / f"{stem}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {png}\nwrote {pdf}")
    print(json.dumps(metadata["kmeans"], indent=2))
    return outdir


def compose(
    tag,
    fig_name,
    min_participation,
    montage="narrow",
    preview_style=False,
    display_min_channel_frac=0.0,
    output_stem=None,
    panel_letter=None,
    formal_layout=False,
    tags=None,
):
    input_tags = list(tags) if tags is not None else [tag]
    loaded = [(input_tag, *_load(input_tag)) for input_tag in input_tags]
    subject = _validate_seed_pool(loaded)
    _, readout, figdata = loaded[0]
    k_dir = int(readout.get("k_dir", 2))
    events = []
    per_seed = []
    for input_tag, seed_readout, seed_figdata in loaded:
        seed = int(seed_readout["seed"])
        selected = []
        for ev in seed_readout["events"]:
            if ev.get("sign") is None or ev.get("n_part", 0) < 2 * k_dir:
                continue
            item = dict(ev)
            item["_seed"] = seed
            item["_source_tag"] = input_tag
            selected.append(item)
        n_seed_fwd = int(sum(float(ev["sign"]) > 0 for ev in selected))
        n_seed_rev = int(sum(float(ev["sign"]) < 0 for ev in selected))
        trace_duration_ms = _duration_ms(seed_figdata)
        duration_ms = float(seed_readout.get("paired_simulation_duration_ms", trace_duration_ms))
        per_seed.append({
            "seed": seed,
            "input_tag": input_tag,
            "duration_ms": duration_ms,
            "trace_duration_ms": trace_duration_ms,
            "n_events": int(len(selected)),
            "n_forward": n_seed_fwd,
            "n_reverse": n_seed_rev,
            "forward_fraction": float(n_seed_fwd / len(selected)) if selected else None,
            "bidirectional": bool(n_seed_fwd > 0 and n_seed_rev > 0),
            "clean_event_rate_hz": float(len(selected) / (duration_ms / 1000.0)),
        })
        events.extend(selected)
    if len(events) < 2:
        raise RuntimeError(f"not enough clean directional events in {input_tags}: {len(events)}")

    all_names = sorted({n for ev in events for n, v in (ev.get("ranks") or {}).items() if v is not None})
    ordered_names = _axis_order(figdata, all_names)          # source -> sink channel order
    ranks, bools = _rank_matrix(events, ordered_names)       # (n_ch, n_ev) in display order
    dropped_display_channels = []
    if display_min_channel_frac > 0:
        frac = np.mean(np.isfinite(ranks), axis=1)
        keep = frac >= float(display_min_channel_frac)
        if int(keep.sum()) < 3:
            raise RuntimeError(
                f"display_min_channel_frac={display_min_channel_frac} leaves "
                f"only {int(keep.sum())} channels"
            )
        dropped_display_channels = [
            {"channel": n, "participation_frac": float(frac[i])}
            for i, n in enumerate(ordered_names)
            if not bool(keep[i])
        ]
        ordered_names = [n for i, n in enumerate(ordered_names) if bool(keep[i])]
        ranks = ranks[keep, :]
        bools = bools[keep, :]
    signs = np.asarray([float(ev["sign"]) for ev in events])
    event_seed_ids = np.asarray([int(ev["_seed"]) for ev in events], dtype=int)
    n_ch, n_ev = ranks.shape
    channel_order = np.arange(n_ch)                           # ranks already in ordered_names order

    result = compute_adaptive_cluster_stereotypy(
        ranks, bools, ordered_names, k_range=(2, 2),
        use_masked_features=True, min_participation=min_participation)
    labels = np.asarray(result.get("labels", []), dtype=int)
    if labels.shape != (n_ev,):
        raise RuntimeError(f"KMeans labels length {labels.shape} != event count {n_ev}")

    purity, conf = _direction_purity(labels, signs)
    cluster_display = _cluster_display_from_direction(labels, conf)
    if formal_layout:
        for info in cluster_display.values():
            info["label"] = (
                "model forward" if info["direction"] == "forward" else "model reverse"
            )
    active_inter = float(result["inter_cluster_corr_matrix"][0][1])
    shared_corr, shared_n = _shared_corr(ranks, labels)
    within_tau = float(result.get("within_cluster_tau_mean", np.nan))
    direction_perm_p = _seed_stratified_direction_permutation(
        labels, signs, event_seed_ids, n_perm=10_000,
    )
    loso = _loso_cluster_statistics(
        ranks, bools, signs, event_seed_ids, ordered_names, min_participation,
    ) if len(input_tags) > 1 else []

    # model-vs-real-template similarity matrix (rightmost panel). GATE (2026-06-27): the model
    # forward/reverse templates are built from events of each SIGN directly (NOT by mapping the two
    # KMeans clusters -- for a one-direction readout the two clusters are two sub-patterns of the
    # SAME direction, so labelling one "forward" and the other "reverse" is fake). The fwd-rev x
    # real-t_a/t_b matrix is only valid when BOTH directions are present with >= MIN_DIR_EVENTS each;
    # otherwise the panel is N/A (one-direction diagnostic only).
    n_fwd = int(np.sum(signs > 0)); n_rev = int(np.sum(signs < 0))
    bidirectional = (n_fwd >= MIN_DIR_EVENTS) and (n_rev >= MIN_DIR_EVENTS)

    def _sign_meanrank(mask):
        out = {}
        for i, n in enumerate(ordered_names):
            v = ranks[i, mask]; v = v[np.isfinite(v)]
            if v.size:
                out[n] = float(v.mean())
        return out
    if bidirectional:
        model_tpl = {"forward": _sign_meanrank(signs > 0), "reverse": _sign_meanrank(signs < 0)}
        Msim, Psim = _sim_matrix(model_tpl, _real_templates(subject, montage))
    else:
        model_tpl, Msim, Psim = None, None, None

    # ---- figure: 3 blocks drawn with the MATURE canonical helpers ----
    # k=2 heatmap leftmost; all blocks share the SAME channel order (ordered_names) AND y-orientation
    # (the canonical cluster-rank helper internally inverts its y-axis, so we re-invert it to align).
    # Canonical fonts are tuned for the full-page per_subject figure; shrink them for a compact
    # paper-ready panel, and crop the rank axis to the ACTUAL max rank (events recruit a subset of
    # contacts, so within-event ranks only reach ~max_rank << n_ch -> avoid the empty right margin).
    # `ranks` retains finite placeholder ranks for non-participating contacts.
    # They must not determine the displayed rank range: the plotted statistic is
    # explicitly conditional on within-event participation (`bools`).
    valid_rank_values = ranks[np.asarray(bools, dtype=bool)]
    valid_rank_values = valid_rank_values[np.isfinite(valid_rank_values)]
    max_rank = int(np.nanmax(valid_rank_values)) if valid_rank_values.size else 1

    def _shrink(ax, title=11.5, label=10, tick=8.5):
        ax.title.set_fontsize(title)
        ax.xaxis.label.set_fontsize(label); ax.yaxis.label.set_fontsize(label)
        for t in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            t.set_fontsize(tick)
        ax.tick_params(axis="both", labelsize=tick)

    if formal_layout:
        # Match Panel B's canvas and vertical bounds.  Formal Panel C contains
        # only the clustered event heatmap, mean-rank profile, and model-data
        # matrix; the redundant per-channel rank-distribution panel is omitted.
        fig_size = (19.2, 4.75)
        width_ratios = [5.70, 1.65, 2.40]
        wspace = 0.22
        top = 0.88
        bottom = 0.15
        fs = {
            "title": 15.0,
            "label": 13.8,
            "tick": 12.2,
            "cluster": 14.0,
            "matrix_tick": 14.0,
            "matrix_star": 19.0,
        }
    elif preview_style:
        fig_size = (17.8, 5.1)
        width_ratios = [5.45, 0.58, 1.02, 1.34]
        wspace = 0.32
        top = 0.86
        bottom = 0.24
        fs = {"title": 15.5, "label": 13.0, "tick": 11.2, "cluster": 12.5}
    else:
        fig_size = (17.2, 4.6)
        width_ratios = [5.35, 0.62, 1.02, 1.32]
        wspace = 0.34
        top = 0.88
        bottom = 0.22
        fs = {"title": 11.5, "label": 10.0, "tick": 8.5, "cluster": 8.5}

    fig = plt.figure(figsize=fig_size, facecolor="white")
    gs = fig.add_gridspec(
        1, 3 if formal_layout else 4,
        width_ratios=width_ratios,
        wspace=wspace,
        left=0.045 if formal_layout else 0.05,
        # Pull the formal grid inward so the matrix's inset colorbar, rather
        # than the matrix axis, lands on Panel B's outer right boundary.
        right=0.947 if formal_layout else 0.975,
        top=top,
        bottom=bottom,
    )

    # block 1 (LEFT): clustered event heatmap (canonical pcolormesh) + cluster boundaries
    cluster_order = _cluster_display_order(labels, cluster_display)
    ev_order = np.concatenate([np.where(labels == cid)[0] for cid in cluster_order])
    display_id = {int(cid): i for i, cid in enumerate(cluster_order)}
    labels_for_boundaries = np.asarray([display_id[int(cid)] for cid in labels[ev_order]], dtype=int)
    heatmap_gs = gs[0, 0].subgridspec(1, 2, width_ratios=[1.0, 0.045], wspace=0.035)
    ax_hm = fig.add_subplot(heatmap_gs[0, 0])
    cax_hm = fig.add_subplot(heatmap_gs[0, 1])
    im = _plot_rank_heatmap(ax_hm, ranks[:, ev_order], ordered_names, "",
                            show_ylabels=True, display_bools=bools[:, ev_order])
    if formal_layout:
        boundary = int(np.sum(labels_for_boundaries == 0))
        ax_hm.axvspan(
            boundary - 0.18,
            boundary + 0.18,
            facecolor="white",
            edgecolor="0.60",
            hatch="////",
            linewidth=0.0,
            zorder=12,
        )
        _plot_cluster_boundaries(
            ax_hm,
            labels_for_boundaries,
            n_ch,
            line_color="0.55",
            line_width=0.0,
            line_style="-",
            label_fontsize=fs["cluster"],
            label_box=False,
            boundary_band=False,
            label_names=[cluster_display[int(c)]["label"] for c in cluster_order],
            label_y_offset=0.42,
        )
    elif preview_style:
        _plot_cluster_boundaries(
            ax_hm,
            labels_for_boundaries,
            n_ch,
            line_color="#d00000",
            line_width=3.4,
            line_style="-",
            label_fontsize=fs["cluster"],
            label_box=True,
            boundary_band=True,
            label_names=[cluster_display[int(c)]["label"] for c in cluster_order],
        )
    else:
        _plot_cluster_boundaries(
            ax_hm,
            labels_for_boundaries,
            n_ch,
            label_names=[cluster_display[int(c)]["label"] for c in cluster_order],
        )
    ax_hm.set_xlabel(
        (f"Model events ({len(input_tags)} seeds; n={n_ev})"
         if formal_layout and len(input_tags) > 1 else
         "Model events (clustered)" if formal_layout else "group events"),
        fontsize=fs["label"],
    )
    for txt in ax_hm.texts:
        txt.set_fontsize(fs["cluster"])
        for info in cluster_display.values():
            if txt.get_text().startswith(info["label"]):
                txt.set_color(info["color"])
                txt.set_fontweight("bold")
                bbox = txt.get_bbox_patch()
                if bbox is not None:
                    bbox.set_edgecolor(info["color"])
    _shrink(ax_hm, title=fs["title"], label=fs["label"], tick=fs["tick"])
    ax_hm.set_title("" if formal_layout else "Clustered group events",
                    fontsize=fs["title"], pad=18 if preview_style else 16)
    cb = fig.colorbar(im, cax=cax_hm, orientation="vertical")
    cb.set_label("First → last", fontsize=fs["label"])
    cb.ax.tick_params(labelsize=fs["tick"])
    hm_ylim = ax_hm.get_ylim()
    hm_yticks = ax_hm.get_yticks()
    hm_ylabels = [tick.get_text() for tick in ax_hm.get_yticklabels()]

    # Preview/diagnostic mode retains the per-channel rank distribution.  It
    # is intentionally absent from the formal three-block Panel C.
    if not formal_layout:
        ax_h = fig.add_subplot(gs[0, 1])
        _hist_aligned(
            ax_h,
            ranks,
            max_rank,
            hm_ylabels,
            n_ch,
            y_centers=hm_yticks,
            ylim=hm_ylim,
            show_ylabels=not preview_style,
        )
        ax_h.set_title("Rank dist.", fontsize=fs["title"])
        _shrink(ax_h, title=fs["title"], label=fs["label"], tick=fs["tick"])

    # Mean-rank profiles use the same copied heatmap y ticks.
    ax_cr = fig.add_subplot(gs[0, 1 if formal_layout else 2])
    _cluster_aligned(
        ax_cr,
        ranks,
        labels,
        max_rank,
        hm_ylabels,
        y_centers=hm_yticks,
        ylim=hm_ylim,
        cluster_display=cluster_display,
        cluster_order=cluster_order,
        show_ylabels=not formal_layout,
    )
    ax_cr.set_title("" if formal_layout else "Cluster rank profile", fontsize=fs["title"])
    if formal_layout:
        legend = ax_cr.get_legend()
        if legend is not None:
            legend.remove()
    else:
        ax_cr.legend(
            frameon=False,
            prop={"size": fs["tick"]},
            loc="upper right",
            borderaxespad=0.2,
            handlelength=1.35,
        )
    if formal_layout:
        ax_cr.set_xticks(np.arange(max_rank + 1))
    _shrink(ax_cr, title=fs["title"], label=fs["label"], tick=fs["tick"])

    # block 4 (RIGHT): model-vs-real template similarity matrix -- only when BOTH directions present
    # (gate). Keep the 2x2 matrix square by putting its colorbar in a separate sub-axis; a normal
    # fig.colorbar(ax=...) silently steals width from the matrix axis and can make the panel look
    # visually compressed in the paper layout.
    ax_m = fig.add_subplot(gs[0, 2 if formal_layout else 3])
    if bidirectional:
        im_m = _matrix_panel(ax_m, Msim, Psim)
        if formal_layout:
            ax_m.set_xticklabels(
                ["data forward", "data reverse"],
                fontsize=fs["matrix_tick"],
                fontweight="bold",
            )
            ax_m.set_yticklabels(
                ["model forward", "model reverse"],
                fontsize=fs["matrix_tick"],
            )
            ax_m.get_xticklabels()[0].set_color(TA_COLOR)
            ax_m.get_xticklabels()[1].set_color(TB_COLOR)
            for txt in ax_m.texts:
                txt.set_fontsize(fs["matrix_star"])
                txt.set_fontweight("bold")
            ax_m.set_xlabel("")
            ax_m.set_ylabel("")
            ax_m.set_title("")
        else:
            ax_m.set_title("model vs real templates\n(★ perm p)", fontsize=fs["label"])
        ax_m.set_box_aspect(1.0)
        cax_m = inset_axes(
            ax_m,
            width="7%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=(1.08, 0.0, 1.0, 1.0),
            bbox_transform=ax_m.transAxes,
            borderpad=0,
        )
        cb_m = fig.colorbar(im_m, cax=cax_m)
        cb_m.set_label("ρ" if formal_layout else "ρ model vs real", fontsize=fs["label"], fontweight="bold" if formal_layout else "normal")
        cb_m.ax.tick_params(labelsize=fs["tick"])
        if formal_layout:
            for tick in cb_m.ax.get_yticklabels():
                tick.set_fontsize(fs["tick"])
    else:
        ax_m.axis("off")
        ax_m.set_box_aspect(1.0)
        ax_m.text(0.5, 0.5, f"model vs real\nN/A\n\none-direction only\n(fwd={n_fwd}, rev={n_rev})\n"
                  f"≥{MIN_DIR_EVENTS} each needed",
                  ha="center", va="center", fontsize=fs["label"], color="#b00",
                  transform=ax_m.transAxes,
                  bbox=dict(boxstyle="round", fc="#fff3f3", ec="#b00", lw=1.0))

    if panel_letter:
        fig.text(0.012, 0.93, str(panel_letter), fontsize=22, fontweight="bold")

    outdir = ROOT / f"results/paper-ready-figure/{fig_name}/figures"
    outdir.mkdir(parents=True, exist_ok=True)
    stem = output_stem or f"{fig_name}_kmeans2"
    png, pdf = outdir / f"{stem}.png", outdir / f"{stem}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight", pad_inches=0.0, facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.0, facecolor="white")
    plt.close(fig)

    seed_forward_fractions = np.asarray([
        row["forward_fraction"] for row in per_seed if row["forward_fraction"] is not None
    ], dtype=float)
    loso_summary = {}
    for key in ("direction_purity", "within_cluster_tau_mean", "shared_overlap_corr"):
        vals = np.asarray([row[key] for row in loso], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            loso_summary[key] = {
                "median": float(np.median(vals)),
                "range": [float(np.min(vals)), float(np.max(vals))],
            }
    paired_arm_pool = str(readout.get("lesion")) == "driven_pooled"
    replication_stats = {
        "independent_unit": (
            "paired network seed (source-only and sink-only arms)"
            if paired_arm_pool else "simulation seed"
        ),
        "n_seeds": int(len(input_tags)),
        "seeds": [int(row["seed"]) for row in per_seed],
        "per_seed": per_seed,
        "total_clean_directional_events": int(n_ev),
        "total_forward": int(n_fwd),
        "total_reverse": int(n_rev),
        "total_simulation_duration_ms": float(sum(row["duration_ms"] for row in per_seed)),
        "bidirectional_seeds": int(sum(row["bidirectional"] for row in per_seed)),
        "forward_fraction_across_seeds": ({
            "median": float(np.median(seed_forward_fractions)),
            "iqr": [
                float(np.quantile(seed_forward_fractions, 0.25)),
                float(np.quantile(seed_forward_fractions, 0.75)),
            ],
        } if seed_forward_fractions.size else None),
        "cluster_direction_association": {
            "statistic": "direction purity",
            "observed": float(purity),
            "permutation": "shuffle direction labels within each seed; KMeans labels fixed",
            "n_permutations": 10_000,
            "p_value": float(direction_perm_p),
        },
        "leave_one_seed_out": {
            "tau_monte_carlo": {"n_sample": 60, "n_tau_seeds": 2},
            "rows": loso,
            "summary": loso_summary,
        },
        "interpretation_boundary": (
            "Events are pooled for the display and KMeans fit, but the paired network seed is "
            "the independent replication unit; event count is not treated as n independent "
            "simulations."
            if paired_arm_pool else
            "Events are pooled for the display and KMeans fit, but seed is the independent "
            "replication unit; event count is not treated as n independent simulations."
        ),
    }

    metadata = {
        "figure": stem, "companion_to": fig_name,
        "input_tag": input_tags[0] if len(input_tags) == 1 else None,
        "input_tags": input_tags,
        "preview_style": bool(preview_style),
        "formal_layout": bool(formal_layout),
        "panel_letter": panel_letter,
        "rank_axis_limits": [0, max_rank],
        "layout": (
            "Panel-B-aligned three-block layout: clustered heatmap | mean-rank profile | "
            "model-vs-data correlation matrix"
            if formal_layout else "legacy/preview single-row layout"
        ),
        "plotters": (
            "canonical _plot_rank_heatmap + _plot_cluster_boundaries; direction-aware line+SD template profile; "
            "square _matrix_panel with compact model/dataset tick semantics"
            if formal_layout else
            "canonical heatmap/boundaries + _hist_aligned + _cluster_aligned"
        ),
        "event_filter": f"sign is not None and n_part >= 2*k_dir ({2 * k_dir})",
        "display_min_channel_frac": float(display_min_channel_frac),
        "dropped_display_channels": dropped_display_channels,
        "n_events": n_ev, "n_forward": n_fwd, "n_reverse": n_rev,
        "bidirectional": bidirectional, "min_dir_events_gate": MIN_DIR_EVENTS,
        "replication_statistics": replication_stats,
        "channels_displayed": ordered_names,
        "kmeans": {
            "k": 2, "min_participation": int(min_participation), "labels": labels.tolist(),
            "cluster_sizes": {f"C{c}": int(np.sum(labels == c)) for c in sorted(set(labels.tolist()))},
            "display_cluster_order": [f"C{int(c)}" for c in cluster_order],
            "display_labels": {
                f"C{c}": {
                    "label": cluster_display[int(c)]["label"],
                    "color": cluster_display[int(c)]["color"],
                    "direction": cluster_display[int(c)]["direction"],
                }
                for c in sorted(set(labels.tolist()))
            },
            "direction_confusion_rows_cluster_cols_forward_reverse": conf.tolist(),
            "direction_purity": purity,
            "seed_stratified_direction_permutation_p": direction_perm_p,
            "within_cluster_tau_mean": within_tau,
            "active_inter_cluster_corr": active_inter,
            "shared_overlap_corr": shared_corr, "shared_overlap_n_channels": shared_n,
            "candidate_forward_reverse_pairs": result.get("candidate_forward_reverse_pairs", []),
        },
        "similarity_matrix_panel": ({
            "valid": True,
            "model_templates_built_by": "event sign (forward = sign>0 events, reverse = sign<0 events), NOT cluster label",
            "rows_model_forward_reverse_cols_data_forward_reverse_spearman": Msim.tolist(),
            "directional_perm_p": Psim.tolist(),
            "perm_p_direction": "diagonal one-sided positive, off-diagonal one-sided negative (swap-predicted)",
            "stars": "panel shows directional channel-shuffle permutation p as stars only; color = Spearman ρ",
        } if bidirectional else {
            "valid": False,
            "reason": f"one-direction readout (fwd={n_fwd}, rev={n_rev}); need >= {MIN_DIR_EVENTS} each. "
                      "fwd/rev x t_a/t_b mapping would be fake -> panel shown as N/A.",
        }),
        "notes": [
            "Plotting/statistics only; the fixed-parameter multi-seed simulation artifacts already existed.",
            "Formal Panel C uses a Panel-B-aligned three-block layout; Rank dist. is omitted.",
            "The clustered heatmap uses a white hatched model-forward/model-reverse separator.",
            "Model forward/reverse mean-rank profiles use the same line + SD-band grammar; their redundant legend is omitted because the adjacent heatmap fixes the color mapping.",
            "Model fwd/rev templates built by EVENT SIGN (not cluster label); similarity matrix GATED "
            "on >= MIN_DIR_EVENTS of each direction (one-direction subjects -> N/A, not a fake swap).",
            "Full/active imputed inter-cluster corr is not the main direction readout; "
            "shared-overlap corr + direction purity are the cleaner direction judges.",
            "If both KMeans clusters have the same direction, display labels remain direction-specific "
            "subclusters rather than fabricating a forward/reverse pair.",
        ],
    }
    stats_json = outdir.parent / f"{stem}_statistics.json"
    stats_csv = outdir.parent / f"{stem}_per_seed.csv"
    stats_json.write_text(json.dumps(replication_stats, indent=2), encoding="utf-8")
    with stats_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed[0].keys()))
        writer.writeheader()
        writer.writerows(per_seed)
    metadata["statistics_outputs"] = [
        str(stats_json.relative_to(ROOT)), str(stats_csv.relative_to(ROOT)),
    ]
    (outdir / f"{stem}_metadata.json").write_text(json.dumps(metadata, indent=2))
    readme = outdir / "README.md"
    if not readme.exists():
        readme.write_text(
            f"# {fig_name}\n\n"
            "### " + stem + ".png / .pdf\n\n"
            "KMeans k=2 subject-level preview rendered from the subject-SNN readout artifact. "
            "This preview uses larger fonts, a narrower rank-distribution panel with no repeated "
            "y-axis labels, and a stronger red cluster boundary on the clustered event heatmap.\n\n"
            "**关注点**：先看左侧 KMeans heatmap 的红色分界是否清楚，再看 rank distribution 是否不抢占宽度。\n",
            encoding="utf-8",
        )
    print(f"wrote {png}\nwrote {pdf}")
    print(json.dumps(metadata["kmeans"], indent=2))
    return outdir


def main():
    os.chdir(ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="epilepsiae_1146_twoend_equal_tsrc_s3")
    ap.add_argument(
        "--tags",
        default=None,
        help="Comma-separated fixed-parameter seed tags to pool; overrides --tag for Panel C.",
    )
    ap.add_argument("--fig-name", default="fig_subject_snn_epilepsiae_1146")
    ap.add_argument("--min-participation", type=int, default=3)
    ap.add_argument("--montage", default="narrow", choices=["narrow", "broad"])
    ap.add_argument(
        "--preview-style",
        action="store_true",
        help="Render the enlarged-font / narrow-rank-distribution preview style without changing defaults.",
    )
    ap.add_argument("--output-stem", default=None)
    ap.add_argument("--panel-letter", default=None)
    ap.add_argument("--formal-layout", action="store_true")
    ap.add_argument(
        "--display-min-channel-frac",
        type=float,
        default=0.0,
        help=(
            "Optional active-contact display/KMeans filter. Channels participating in less than this "
            "fraction of clean events are excluded from the KMeans panel and similarity matrix. "
            "Default 0 keeps the full readout channel set."
        ),
    )
    a = ap.parse_args()
    compose(
        a.tag,
        a.fig_name,
        a.min_participation,
        a.montage,
        preview_style=a.preview_style,
        display_min_channel_frac=a.display_min_channel_frac,
        output_stem=a.output_stem,
        panel_letter=a.panel_letter,
        formal_layout=a.formal_layout,
        tags=[x.strip() for x in a.tags.split(",") if x.strip()] if a.tags else None,
    )


if __name__ == "__main__":
    main()
