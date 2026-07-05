"""core_model_s3-style subject-specific SNN figure (field-swap plan §5).

Consumes one subject run (readout_<tag>.json + figdata_<tag>.npz from
run_sef_hfo_subject_snn.py). Workflow (user 2026-06-26): FIRST confirm forward/
reverse exists, THEN cluster (KMeans, same masked pipeline as real data), THEN draw.

3 rows, patient-specific:
  A forward  : left = source-space propagation gradient (per-neuron onset, viridis)
               + two swap cores + axis; right = patient-electrode LFP read-out (|LFP|
               peak sweep, contacts ordered by axis position).
  B reverse  : same for a representative reverse event.
  C KMeans   : per-event rank heatmap, events grouped by the 2 clusters (the same
               compute_adaptive_cluster_stereotypy / masked-feature pipeline as real
               subjects); annotated with inter-cluster corr + forward/reverse pair.
"""
import sys
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.getcwd())
from src.interictal_propagation import compute_adaptive_cluster_stereotypy

RUN_DIR = "results/topic4_sef_hfo/field_swap_subject_snn"


def build_rank_matrix(events, valid_names):
    """R (n_contact x n_event), B bools from per-event rank dicts (valid contacts)."""
    R = np.full((len(valid_names), len(events)), np.nan)
    for e, ev in enumerate(events):
        ranks = ev.get("ranks") or {}
        for i, n in enumerate(valid_names):
            v = ranks.get(n)
            if v is not None:
                R[i, e] = v
    B = np.isfinite(R)
    return R, B


def axis_order(contacts, names, valid_names, center, axis_unit):
    idx = [list(names).index(n) for n in valid_names]
    proj = (np.asarray(contacts)[idx] - np.asarray(center)) @ np.asarray(axis_unit)
    return np.argsort(proj), proj


def panel_gradient(ax, posE, onset, reg, contacts, names, title):
    fin = np.isfinite(onset)
    sub = np.zeros(len(posE), bool); sub[::4] = True  # subsample for speed
    q = sub & ~fin
    ax.scatter(posE[q, 0], posE[q, 1], s=2, c="0.85", zorder=1)
    p = sub & fin
    sc = ax.scatter(posE[p, 0], posE[p, 1], s=4, c=onset[p], cmap="viridis", zorder=2)
    src = np.array(reg["source_centroid"]); snk = np.array(reg["sink_centroid"])
    ax.plot([src[0], snk[0]], [src[1], snk[1]], "k--", lw=1.5, zorder=3)
    ax.scatter(*src, marker="X", s=200, c="#d62728", edgecolors="k", zorder=5, label="source core")
    ax.scatter(*snk, marker="X", s=200, c="#1f77b4", edgecolors="k", zorder=5, label="sink core")
    C = np.asarray(contacts)
    ax.scatter(C[:, 0], C[:, 1], s=18, facecolors="none", edgecolors="k", lw=0.6, zorder=4)
    ax.set_title(title, fontsize=10); ax.set_aspect("equal")
    ax.set_xlabel("sheet x (mm)"); ax.set_ylabel("y (mm)")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="onset (ms)")


def panel_lfp(ax, fig, ev_meta, order, valid_names, proj, title):
    lfp = fig["lfp_trace"]; times = fig["times"]; names = list(fig["names"])
    vidx = [names.index(n) for n in valid_names]
    t_on, t_off = ev_meta["t_on"], ev_meta["t_off"]
    pad = 0.3 * (t_off - t_on) + 20
    m = (times >= t_on - pad) & (times <= t_off + pad)
    tt = times[m]
    for rank_pos, oi in enumerate(order):
        tr = np.abs(lfp[m, vidx[oi]])
        tr = tr / (tr.max() + 1e-9)
        ax.plot(tt, tr + rank_pos * 1.1, lw=0.8, color="0.35")
        pk = tt[int(np.argmax(np.abs(lfp[m, vidx[oi]])))]
        ax.plot(pk, rank_pos * 1.1 + 0.5, "o", ms=3, color="#d62728")
    ax.axvspan(t_on, t_off, color="0.9", zorder=0)
    ax.set_yticks([i * 1.1 + 0.5 for i in range(len(order))])
    ax.set_yticklabels([valid_names[oi] for oi in order], fontsize=6)
    ax.set_xlabel("time (ms)"); ax.set_title(title, fontsize=10)
    ax.set_ylabel("contacts (ordered source->sink)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--run-dir", default=RUN_DIR)
    ap.add_argument("--min-participation", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="draw even if not bidirectional")
    a = ap.parse_args()

    out = json.load(open(os.path.join(a.run_dir, f"readout_{a.tag}.json")))
    fd = np.load(os.path.join(a.run_dir, f"figdata_{a.tag}.npz"), allow_pickle=True)
    reg = fd["reg"].item()
    print(f"[{a.tag}] dir fwd/rev = {out['dir_forward']}/{out['dir_reverse']}  "
          f"clean fwd/rev = {out['clean_forward']}/{out['clean_reverse']}  bidir={out['bidirectional']}")
    if not out["bidirectional"] and not a.force:
        print("NOT bidirectional -> stopping before KMeans (user ordering: 先看正反向再 kmeans). "
              "Use --force to draw anyway.")
        return

    # ---- KMeans (same masked pipeline as real subjects) ----
    events = out["events"]
    valid_names = sorted({n for ev in events for n in (ev.get("ranks") or {})})
    R, B = build_rank_matrix(events, valid_names)
    cl = compute_adaptive_cluster_stereotypy(R, B, valid_names, k_range=(2, 2),
                                             use_masked_features=True,
                                             min_participation=a.min_participation)
    corr = cl.get("inter_cluster_corr_matrix")
    inter = corr[0][1] if (corr and len(corr) > 1) else float("nan")
    frpairs = cl.get("candidate_forward_reverse_pairs")
    labels = np.asarray(cl.get("labels", []), int)
    print(f"  KMeans k=2: inter-cluster corr={inter:.3f}  forward_reverse_pairs={frpairs}  "
          f"cluster sizes={[c['n_events'] for c in cl.get('clusters', [])]}")
    # cluster <-> direction correspondence
    signs = np.array([ev["sign"] if ev["sign"] is not None else 0 for ev in events])
    if labels.size == len(events):
        for c in sorted(set(labels)):
            s = signs[labels == c]
            print(f"  cluster {c}: n={ (labels==c).sum() }  mean_sign={s.mean():+.2f} "
                  f"(fwd={int((s>0).sum())} rev={int((s<0).sum())})")

    # ---- figure ----
    order, proj = axis_order(fd["contacts"], fd["names"], valid_names, reg["center"], reg["axis_unit"])
    posE = fd["posE"]; rep_fwd = fd["rep_fwd"].item(); rep_rev = fd["rep_rev"].item()
    fig = plt.figure(figsize=(14, 13))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.1, 1.0], height_ratios=[1, 1, 1])

    if rep_fwd:
        panel_gradient(fig.add_subplot(gs[0, 0]), posE, rep_fwd["onset"], reg, fd["contacts"], fd["names"],
                       f"A  forward event: source-space propagation (t={rep_fwd['meta']['t_on']:.0f}ms, "
                       f"n_part={rep_fwd['meta']['n_part']})")
        panel_lfp(fig.add_subplot(gs[0, 1]), fd, rep_fwd["meta"], order, valid_names, proj,
                  "A  forward: patient-electrode |LFP| read-out")
    if rep_rev:
        panel_gradient(fig.add_subplot(gs[1, 0]), posE, rep_rev["onset"], reg, fd["contacts"], fd["names"],
                       f"B  reverse event: source-space propagation (t={rep_rev['meta']['t_on']:.0f}ms, "
                       f"n_part={rep_rev['meta']['n_part']})")
        panel_lfp(fig.add_subplot(gs[1, 1]), fd, rep_rev["meta"], order, valid_names, proj,
                  "B  reverse: patient-electrode |LFP| read-out")

    # C: rank heatmap grouped by cluster
    axc = fig.add_subplot(gs[2, :])
    if labels.size == len(events):
        evorder = np.argsort(labels)
        Rsorted = R[np.ix_(order, evorder)]
        im = axc.imshow(Rsorted, aspect="auto", cmap="viridis", interpolation="nearest")
        # cluster boundary
        bnd = np.searchsorted(labels[evorder], np.unique(labels)[1:])
        for b in bnd:
            axc.axvline(b - 0.5, color="r", lw=2)
        axc.set_yticks(range(len(order))); axc.set_yticklabels([valid_names[oi] for oi in order], fontsize=6)
        axc.set_xlabel("events (grouped by KMeans cluster, red line = boundary)")
        axc.set_ylabel("contacts (source->sink)")
        plt.colorbar(im, ax=axc, fraction=0.02, pad=0.01, label="rank")
    axc.set_title(f"C  KMeans k=2 (masked pipeline, same as real): inter-cluster corr = {inter:.2f}  |  "
                  f"forward/reverse pair = {frpairs}", fontsize=10)

    fig.suptitle(f"Subject-specific SNN (swap cores -> patient electrodes): {out['subject']} "
                 f"[{out['montage']}, {out['anchor']}, inter-core {out['inter_core_sheet']}mm]\n"
                 f"spontaneous twoend_equal, dir fwd/rev={out['dir_forward']}/{out['dir_reverse']} "
                 f"(clean {out['clean_forward']}/{out['clean_reverse']})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    figdir = os.path.join(a.run_dir, "figures")
    os.makedirs(figdir, exist_ok=True)
    fp = os.path.join(figdir, f"core_model_subject_{a.tag}.png")
    fig.savefig(fp, dpi=130, bbox_inches="tight")
    print(f"[written] {fp}")


if __name__ == "__main__":
    main()
