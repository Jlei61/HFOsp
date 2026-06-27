"""Paper-ready subject-specific SNN KMeans panel (Topic 4 Fig4B).

Companion to ``plot_fig_subject_snn.py``. Consumes the SAME spontaneous twoend
E1146 readout used in Fig4A and draws the legacy KMeans=2 rank diagnostic in the
**mature canonical style** -- the same plotting functions that produce
``results/interictal_propagation_masked/figures/per_subject/<dataset>_<subject>_propagation.png``
(`_plot_rank_histogram` / `_plot_rank_heatmap` + `_plot_cluster_boundaries` /
`_plot_cluster_rank_fig4` from scripts/plot_interictal_propagation.py). No simulation rerun.

Three blocks (one row): per-channel rank distribution | clustered event heatmap | cluster rank profiles.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/topic4_sef_hfo/field_swap_subject_snn"

sys.path.insert(0, str(ROOT))
from src.interictal_propagation import compute_adaptive_cluster_stereotypy  # noqa: E402
# MATURE canonical heatmap plotter (same function that draws the per_subject propagation fig).
# Per-channel rank + cluster profile are drawn by _hist_aligned / _cluster_aligned below: same
# canonical visual style but on the heatmap's 1-unit channel coordinate so all 3 left panels' y-axes
# align channel-for-channel (sharey) -- the canonical ridge histogram uses 0.15 spacing and cannot align.
from scripts.plot_interictal_propagation import (  # noqa: E402
    _plot_rank_heatmap, _plot_cluster_boundaries)
# model-vs-real-template similarity matrix (rightmost panel) + real-template loader.
from scripts.paper_figures.plot_fig_subject_snn_realvsmodel import (  # noqa: E402
    _real_templates, _sim_matrix, _matrix_panel)


_CL_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
MIN_DIR_EVENTS = 3   # both forward AND reverse need >= this many events for the fwd-rev x t_a/t_b matrix


def _hist_aligned(ax, R, max_rank, names, viridis_n):
    """Per-channel rank histogram on the heatmap's 1-unit coordinate (channel display-row i in band
    [i,i+1]) so its y-axis aligns with the heatmap + cluster panels. Canonical viridis-per-channel look."""
    n_ch = R.shape[0]
    for i in range(n_ch):
        vals = R[i][np.isfinite(R[i])]
        if vals.size == 0:
            continue
        hist, _ = np.histogram(vals, bins=np.arange(max_rank + 2) - 0.5)
        h = hist / max(1, vals.size) * 0.82
        ax.bar(np.arange(max_rank + 1), h, bottom=i + 0.09, width=1.0,
               color=plt.cm.viridis(i / max(1, viridis_n - 1)), alpha=0.8, linewidth=0)
        ax.axhline(i, color="0.9", lw=0.3)
    ax.set_ylim(0, n_ch); ax.set_yticks(np.arange(n_ch) + 0.5); ax.set_yticklabels(names)
    ax.set_xlim(-0.5, max_rank + 0.5); ax.set_xlabel("Rank")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _cluster_aligned(ax, R, labels, max_rank, names):
    """Per-cluster mean+/-std rank line on the heatmap's 1-unit coordinate (channel center i+0.5)."""
    n_ch = R.shape[0]
    y = np.arange(n_ch) + 0.5
    for k, cid in enumerate(sorted(set(labels.tolist()))):
        mask = labels == cid
        mean = np.full(n_ch, np.nan); std = np.full(n_ch, np.nan)
        for i in range(n_ch):
            v = R[i, mask]; v = v[np.isfinite(v)]
            if v.size:
                mean[i] = v.mean(); std[i] = v.std()
        fin = np.isfinite(mean); col = _CL_COLORS[k % len(_CL_COLORS)]
        ax.fill_betweenx(y[fin], (mean - std)[fin], (mean + std)[fin], color=col, alpha=0.15, lw=0)
        ax.plot(mean[fin], y[fin], "-o", color=col, lw=2.0, ms=5, label=f"C{int(cid)} (n={int(mask.sum())})")
    ax.set_ylim(0, n_ch); ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlim(-0.5, max_rank + 0.5); ax.set_xlabel("Rank")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _load(tag: str):
    return (json.load(open(RUN / f"readout_{tag}.json")),
            np.load(RUN / f"figdata_{tag}.npz", allow_pickle=True))


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


def compose(tag, fig_name, min_participation, montage="narrow"):
    readout, figdata = _load(tag)
    k_dir = int(readout.get("k_dir", 2))
    events = [ev for ev in readout["events"]
              if ev.get("sign") is not None and ev.get("n_part", 0) >= 2 * k_dir]
    if len(events) < 2:
        raise RuntimeError(f"not enough clean directional events in {tag}: {len(events)}")

    all_names = sorted({n for ev in events for n, v in (ev.get("ranks") or {}).items() if v is not None})
    ordered_names = _axis_order(figdata, all_names)          # source -> sink channel order
    ranks, bools = _rank_matrix(events, ordered_names)       # (n_ch, n_ev) in display order
    signs = np.asarray([float(ev["sign"]) for ev in events])
    n_ch, n_ev = ranks.shape
    channel_order = np.arange(n_ch)                           # ranks already in ordered_names order

    result = compute_adaptive_cluster_stereotypy(
        ranks, bools, ordered_names, k_range=(2, 2),
        use_masked_features=True, min_participation=min_participation)
    labels = np.asarray(result.get("labels", []), dtype=int)
    if labels.shape != (n_ev,):
        raise RuntimeError(f"KMeans labels length {labels.shape} != event count {n_ev}")

    purity, conf = _direction_purity(labels, signs)
    active_inter = float(result["inter_cluster_corr_matrix"][0][1])
    shared_corr, shared_n = _shared_corr(ranks, labels)
    within_tau = float(result.get("within_cluster_tau_mean", np.nan))

    # model-vs-real-template similarity matrix (rightmost panel). GATE (2026-06-27): the model
    # forward/reverse templates are built from events of each SIGN directly (NOT by mapping the two
    # KMeans clusters -- for a one-direction readout the two clusters are two sub-patterns of the
    # SAME direction, so labelling one "forward" and the other "reverse" is fake). The fwd-rev x
    # real-t_a/t_b matrix is only valid when BOTH directions are present with >= MIN_DIR_EVENTS each;
    # otherwise the panel is N/A (one-direction diagnostic only).
    subject = "_".join(tag.split("_")[:2])
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
    max_rank = int(np.nanmax(ranks)) if np.isfinite(ranks).any() else 1

    def _shrink(ax, title=11.5, label=10, tick=8.5):
        ax.title.set_fontsize(title)
        ax.xaxis.label.set_fontsize(label); ax.yaxis.label.set_fontsize(label)
        for t in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            t.set_fontsize(tick)
        ax.tick_params(axis="both", labelsize=tick)

    fig = plt.figure(figsize=(16.0, 4.4), facecolor="white")
    gs = fig.add_gridspec(1, 4, width_ratios=[4.6, 1.0, 1.3, 0.85], wspace=0.40,
                          left=0.05, right=0.975, top=0.88, bottom=0.22)

    # block 1 (LEFT): clustered event heatmap (canonical pcolormesh) + cluster boundaries
    ev_order = np.argsort(labels, kind="stable")
    ax_hm = fig.add_subplot(gs[0, 0])
    im = _plot_rank_heatmap(ax_hm, ranks[:, ev_order], ordered_names, "",
                            show_ylabels=True, display_bools=bools[:, ev_order])
    _plot_cluster_boundaries(ax_hm, labels[ev_order], n_ch)
    ax_hm.set_xlabel("pop events", fontsize=10)
    for txt in ax_hm.texts:               # canonical C0/C1 cluster labels -> smaller
        txt.set_fontsize(8.5)
    _shrink(ax_hm)
    ax_hm.set_title("Pop events (clustered)", fontsize=11.5, pad=16)  # lift above C0/C1 labels
    cb = fig.colorbar(im, ax=ax_hm, orientation="horizontal", fraction=0.045, pad=0.20)  # below x-label
    cb.set_label("First → Last", fontsize=8.5); cb.ax.tick_params(labelsize=7.5)

    # block 2 (MIDDLE): per-channel rank distribution, SAME 1-unit channel y as the heatmap (sharey)
    ax_h = fig.add_subplot(gs[0, 1], sharey=ax_hm)
    _hist_aligned(ax_h, ranks, max_rank, ordered_names, n_ch)
    ax_h.set_title("Per-channel rank", fontsize=11.5)
    _shrink(ax_h)

    # block 3 (MIDDLE-RIGHT): cluster rank profiles, SAME 1-unit channel y (sharey -> aligned)
    ax_cr = fig.add_subplot(gs[0, 2], sharey=ax_hm)
    _cluster_aligned(ax_cr, ranks, labels, max_rank, ordered_names)
    ax_cr.set_title("Cluster rank profile", fontsize=11.5)
    ax_cr.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    _shrink(ax_cr)

    # block 4 (RIGHT): model-vs-real template similarity matrix -- only when BOTH directions present
    # (gate). One-direction readout -> N/A panel (the fwd/rev x t_a/t_b mapping would be fake).
    ax_m = fig.add_subplot(gs[0, 3])
    if bidirectional:
        im_m = _matrix_panel(ax_m, Msim, Psim)
        ax_m.set_title("model vs real\n(★ perm p)", fontsize=10)
        fig.colorbar(im_m, ax=ax_m, fraction=0.05, pad=0.08, label="ρ model vs real")
    else:
        ax_m.axis("off")
        ax_m.text(0.5, 0.5, f"model vs real\nN/A\n\none-direction only\n(fwd={n_fwd}, rev={n_rev})\n"
                  f"≥{MIN_DIR_EVENTS} each needed",
                  ha="center", va="center", fontsize=9.5, color="#b00",
                  transform=ax_m.transAxes,
                  bbox=dict(boxstyle="round", fc="#fff3f3", ec="#b00", lw=1.0))

    outdir = ROOT / f"results/paper-ready-figure/{fig_name}/figures"
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{fig_name}_kmeans2"
    png, pdf = outdir / f"{stem}.png", outdir / f"{stem}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metadata = {
        "figure": stem, "companion_to": fig_name, "input_tag": tag,
        "plotters": "canonical _plot_rank_heatmap + _plot_cluster_boundaries (heatmap); "
                    "_hist_aligned + _cluster_aligned (canonical-styled, on heatmap 1-unit coord so the "
                    "3 left panels share an aligned channel y-axis via sharey)",
        "event_filter": f"sign is not None and n_part >= 2*k_dir ({2 * k_dir})",
        "n_events": n_ev, "n_forward": n_fwd, "n_reverse": n_rev,
        "bidirectional": bidirectional, "min_dir_events_gate": MIN_DIR_EVENTS,
        "channels_displayed": ordered_names,
        "kmeans": {
            "k": 2, "min_participation": int(min_participation), "labels": labels.tolist(),
            "cluster_sizes": {f"C{c}": int(np.sum(labels == c)) for c in sorted(set(labels.tolist()))},
            "direction_confusion_rows_cluster_cols_forward_reverse": conf.tolist(),
            "direction_purity": purity, "within_cluster_tau_mean": within_tau,
            "active_inter_cluster_corr": active_inter,
            "shared_overlap_corr": shared_corr, "shared_overlap_n_channels": shared_n,
            "candidate_forward_reverse_pairs": result.get("candidate_forward_reverse_pairs", []),
        },
        "similarity_matrix_panel": ({
            "valid": True,
            "model_templates_built_by": "event sign (forward = sign>0 events, reverse = sign<0 events), NOT cluster label",
            "rows_model_fwd_rev_cols_data_ta_tb_spearman": Msim.tolist(),
            "directional_perm_p": Psim.tolist(),
            "perm_p_direction": "diagonal one-sided positive, off-diagonal one-sided negative (swap-predicted)",
            "stars": "panel shows directional channel-shuffle permutation p as stars only; color = Spearman ρ",
        } if bidirectional else {
            "valid": False,
            "reason": f"one-direction readout (fwd={n_fwd}, rev={n_rev}); need >= {MIN_DIR_EVENTS} each. "
                      "fwd/rev x t_a/t_b mapping would be fake -> panel shown as N/A.",
        }),
        "notes": [
            "Plotting-only; no SNN rerun.",
            "Rank heatmap + rank distribution + cluster profiles drawn by the MATURE canonical "
            "functions (same as the per_subject propagation figure), not hand-rolled.",
            "Model fwd/rev templates built by EVENT SIGN (not cluster label); similarity matrix GATED "
            "on >= MIN_DIR_EVENTS of each direction (one-direction subjects -> N/A, not a fake swap).",
            "Full/active imputed inter-cluster corr is not the main direction readout; "
            "shared-overlap corr + direction purity are the cleaner direction judges.",
        ],
    }
    (outdir / f"{stem}_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"wrote {png}\nwrote {pdf}")
    print(json.dumps(metadata["kmeans"], indent=2))
    return outdir


def main():
    os.chdir(ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="epilepsiae_1146_twoend_equal_tsrc_s3")
    ap.add_argument("--fig-name", default="fig_subject_snn_epilepsiae_1146")
    ap.add_argument("--min-participation", type=int, default=3)
    ap.add_argument("--montage", default="narrow", choices=["narrow", "broad"])
    a = ap.parse_args()
    compose(a.tag, a.fig_name, a.min_participation, a.montage)


if __name__ == "__main__":
    main()
