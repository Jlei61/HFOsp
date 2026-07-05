"""Separate-then-pool subject SNN figure (field-swap plan §3C.2 + §5), s3_brakeoff style.

Pools a SOURCE-core-only run (forward) + a SINK-core-only run (reverse) -- the validated
pooled_bidir method, robust to one-core dominance -- then:
  A/B rows: core map + propagation gradient (left) + MULTI-EVENT TRAIN read-out on the
            patient electrodes (right, peak-locus per event, like core_model_s3_brakeoff).
  C row   : the ACTUAL unsupervised KMeans (cluster-grouped heatmap) + a KMeans-cluster x
            origin CONFUSION matrix (does kmeans recover source/sink?) + honest metrics.

Honesty contract (review 2026-06-26):
  - C is the real KMeans, NOT origin-sorting. The origin-pooled shared-contact reversal is
    reported as a separate metric, labelled as such.
  - k_dir=2 is a sparse-electrode relaxation (4 contacts -> direction). The figure annotates
    the k_dir=3 sensitivity (how many events keep a direction under the stricter estimator).
"""
import sys
import os
import json
import argparse
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.getcwd())
from src.interictal_propagation import compute_adaptive_cluster_stereotypy
from src.sef_hfo_observation import endpoint_centroid_axis
from src.sef_hfo_subject_placement import MONTAGE_TREES
from scripts.plot_subject_snn_core_figure import build_rank_matrix, axis_order, panel_gradient

RUN_DIR = "results/topic4_sef_hfo/field_swap_subject_snn"


def patient_field_panel(ax, subject, montage, template, core_a, core_b, title):
    """The patient's REAL interictal field: real contacts colored by typical_rank (viridis,
    low=early), with the two template-source cores outlined. Anchors the reader: the model cores
    sit at this field's early regions."""
    geo = MONTAGE_TREES[montage]["geo"]
    g = json.load(open(os.path.join(geo, f"{subject}_{template}.json")))
    s = g["norm_scale_mm"]
    xs, ys, rk, nm = [], [], [], []
    for c in g["channels"]:
        xs.append(c["x_norm"] * s); ys.append(c["y_norm"] * s)
        rk.append(c.get("typical_rank", np.nan)); nm.append(c["name"])
    sc = ax.scatter(xs, ys, c=rk, cmap="viridis", s=160, edgecolors="0.3", zorder=2)
    for x, y, n in zip(xs, ys, nm):
        if n in core_a:
            ax.scatter([x], [y], s=320, facecolors="none", edgecolors="#d62728", lw=2.4, zorder=3)
        if n in core_b:
            ax.scatter([x], [y], s=320, facecolors="none", edgecolors="#1f77b4", lw=2.4, zorder=3)
        ax.annotate(n, (x, y), fontsize=5.5, ha="center", va="center", zorder=4)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=9)
    ax.set_xlabel("contact x (mm)"); ax.set_ylabel("y (mm)")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="field typical_rank (low=early)")


def load(tag, run_dir):
    return (json.load(open(os.path.join(run_dir, f"readout_{tag}.json"))),
            np.load(os.path.join(run_dir, f"figdata_{tag}.npz"), allow_pickle=True))


def best_rep(fd):
    cands = [c for c in (fd["rep_fwd"].item(), fd["rep_rev"].item()) if c]
    return max(cands, key=lambda c: c["meta"]["n_part"]) if cands else None


def dirs_at_kdir(out, fd, kdir):
    """post-hoc: recompute direction sign for each event at a given k_dir from saved ranks."""
    names = list(fd["names"]); C = np.asarray(fd["contacts"]); au = np.array(fd["reg"].item()["axis_unit"])
    fwd = rev = 0
    for e in out["events"]:
        ranks = e.get("ranks") or {}
        vn = list(ranks)
        if len(vn) < 2 * kdir:
            continue
        idx = [names.index(n) for n in vn]
        r = np.array([ranks[n] if ranks[n] is not None else np.nan for n in vn])
        b = np.isfinite(r)
        if b.sum() < 2 * kdir:
            continue
        ax = endpoint_centroid_axis(r, b, C[idx], k_dir=kdir, eps_deg=2.0)
        if ax is None:
            continue
        s = np.sign(np.dot(ax, au))
        fwd += s > 0; rev += s < 0
    return int(fwd), int(rev)


def train_readout(ax, fd, events, valid_names, order, title):
    """Multi-event TRAIN: |LFP| traces stacked by axis position, per-event peak locus (slant=dir),
    over the whole window (like core_model_s3_brakeoff fused read-out)."""
    lfp = np.abs(fd["lfp_trace"]); times = fd["times"]; names = list(fd["names"])
    vidx = [names.index(valid_names[oi]) for oi in order]
    step = max(1, len(times) // 4000)
    ts = times[::step]
    OFF = 1.25
    for row, ci in enumerate(vidx):
        tr = lfp[::step, ci]; tr = tr / (tr.max() + 1e-9)
        ax.plot(ts, tr + row * OFF, lw=0.5, color="0.45")
    for e in events:
        t_on, t_off = e["t_on"], e["t_off"]
        ax.axvspan(t_on, t_off, color="0.88", lw=0, zorder=0)
        m = (times >= t_on) & (times <= t_off)
        if not m.any():
            continue
        pts = []
        for row, ci in enumerate(vidx):
            seg = lfp[m, ci]
            if seg.max() < 1e-9:
                continue
            pk = times[m][int(np.argmax(seg))]
            ax.plot(pk, row * OFF + 0.5, "o", ms=2.5, mfc="k", mec="w", mew=0.4, zorder=4)
            pts.append((pk, row * OFF + 0.5))
        if len(pts) >= 2:
            pts.sort(); px, py = zip(*pts)
            ax.plot(px, py, "-", color="k", lw=0.8, alpha=0.7, zorder=3)
    ax.set_yticks([r * OFF + 0.5 for r in range(len(order))])
    ax.set_yticklabels([valid_names[oi] for oi in order], fontsize=6)
    ax.set_xlabel("time (ms)"); ax.set_ylabel("contacts (source->sink)")
    ax.set_title(title, fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-tag", required=True)
    ap.add_argument("--sink-tag", required=True)
    ap.add_argument("--run-dir", default=RUN_DIR)
    ap.add_argument("--min-participation", type=int, default=3)
    ap.add_argument("--label", default=None)
    a = ap.parse_args()

    so, sfd = load(a.source_tag, a.run_dir)
    ko, kfd = load(a.sink_tag, a.run_dir)
    reg = sfd["reg"].item(); subj = so["subject"]

    # k_dir sensitivity (post-hoc)
    s_k2, s_k3 = dirs_at_kdir(so, sfd, 2)[0], dirs_at_kdir(so, sfd, 3)
    k_dir = so.get("k_dir", 2)
    sens = {"source": (dirs_at_kdir(so, sfd, 2), dirs_at_kdir(so, sfd, 3)),
            "sink": (dirs_at_kdir(ko, kfd, 2), dirs_at_kdir(ko, kfd, 3))}
    print(f"[{subj}] k_dir sensitivity: source k2={sens['source'][0]} k3={sens['source'][1]} | "
          f"sink k2={sens['sink'][0]} k3={sens['sink'][1]}")

    pooled = [(ev, "source") for ev in so["events"]] + [(ev, "sink") for ev in ko["events"]]
    valid_names = sorted({n for ev, _ in pooled for n in (ev.get("ranks") or {})})
    events = [ev for ev, _ in pooled]
    origin = np.array([o for _, o in pooled])
    R, B = build_rank_matrix(events, valid_names)
    keep = B.any(axis=0)
    R, B, events, origin = R[:, keep], B[:, keep], [e for e, k in zip(events, keep) if k], origin[keep]

    # shared-contact reversal (clean metric)
    is_src = origin == "source"
    def mean_rank(mask):
        mr = np.full(len(valid_names), np.nan)
        for i in range(len(valid_names)):
            v = R[i, mask][np.isfinite(R[i, mask])]
            if v.size:
                mr[i] = v.mean()
        return mr
    mr_s, mr_k = mean_rank(is_src), mean_rank(~is_src)
    shared = np.isfinite(mr_s) & np.isfinite(mr_k)
    proj_all = (np.asarray(sfd["contacts"])[[list(sfd["names"]).index(n) for n in valid_names]]
                - np.asarray(reg["center"])) @ np.asarray(reg["axis_unit"])
    shared_corr = spearmanr(mr_s[shared], mr_k[shared]).correlation if shared.sum() >= 3 else np.nan
    src_axis = spearmanr(proj_all[shared], mr_s[shared]).correlation if shared.sum() >= 3 else np.nan
    snk_axis = spearmanr(proj_all[shared], mr_k[shared]).correlation if shared.sum() >= 3 else np.nan

    # ACTUAL unsupervised KMeans (same masked pipeline as real subjects)
    cl = compute_adaptive_cluster_stereotypy(R, B, valid_names, k_range=(2, 2),
                                             use_masked_features=True, min_participation=a.min_participation)
    corr = cl.get("inter_cluster_corr_matrix")
    inter = corr[0][1] if (corr and len(corr) > 1) else np.nan
    labels = np.asarray(cl.get("labels", []), int)
    nv = cl.get("n_valid_events", labels.size)
    # map kmeans labels to events: the pipeline labels only its valid events; align by valid mask
    valid_ev = None
    if labels.size != len(events):
        from src.interictal_propagation import _valid_event_indices
        try:
            valid_ev = _valid_event_indices(B, min_participating=a.min_participation)
        except Exception:
            valid_ev = np.arange(len(events))
    conf_txt = "labels<->events mismatch"
    if labels.size == len(events) or (valid_ev is not None and labels.size == len(valid_ev)):
        ev_idx = np.arange(len(events)) if labels.size == len(events) else np.asarray(valid_ev)
        orig_lab = origin[ev_idx]
        conf = np.zeros((2, 2), int)  # rows=kmeans cluster, cols=origin(source,sink)
        for lab, og in zip(labels, orig_lab):
            conf[lab % 2, 0 if og == "source" else 1] += 1
        # purity = best assignment of clusters to origins
        purity = max(conf[0, 0] + conf[1, 1], conf[0, 1] + conf[1, 0]) / max(conf.sum(), 1)
        conf_txt = f"kmeans purity vs origin={purity:.2f}"
        print(f"  KMeans confusion (rows=cluster, cols=source/sink): {conf.tolist()}  {conf_txt}")
    print(f"  shared={int(shared.sum())}/{len(valid_names)} src-axis={src_axis:+.2f} sink-axis={snk_axis:+.2f} "
          f"reversed={shared_corr:+.2f} | kmeans inter-corr(imputed)={inter:+.2f}")

    # ---- figure ----
    order, proj = axis_order(sfd["contacts"], sfd["names"], valid_names, reg["center"], reg["axis_unit"])
    rep_s, rep_k = best_rep(sfd), best_rep(kfd)
    core_a = list(reg.get("source_names", [])); core_b = list(reg.get("sink_names", []))
    fig = plt.figure(figsize=(15, 18))
    gs = fig.add_gridspec(4, 3, width_ratios=[1.0, 1.5, 0.9], height_ratios=[0.95, 1, 1, 1.05])

    # Row 0: the patient's REAL interictal field (anchor) -- cores sit at each template's early region
    if core_a and core_b:
        patient_field_panel(fig.add_subplot(gs[0, 0]), subj, so["montage"], "t_a", core_a, core_b,
                            f"patient field t_a (real contacts): red=t_a source core, blue=t_b source core")
        patient_field_panel(fig.add_subplot(gs[0, 1]), subj, so["montage"], "t_b", core_a, core_b,
                            f"patient field t_b (the reverse template)")
        axleg = fig.add_subplot(gs[0, 2]); axleg.axis("off")
        axleg.text(0.0, 0.5, "PLACEMENT (user-corrected):\ncores = earliest-3 electrodes\nof EACH template\n"
                   f"core A (t_a source): {','.join(core_a)}\ncore B (t_b source): {','.join(core_b)}\n"
                   "real geometry plane-fit ->\n13mm sep (no anchoring)\nstage3 params m17.5/std1.0",
                   fontsize=8, va="top", family="monospace")

    if rep_s:
        panel_gradient(fig.add_subplot(gs[1, 0]), sfd["posE"], rep_s["onset"], reg, sfd["contacts"], sfd["names"],
                       f"A source-core MODEL: source-space propagation (rep event)")
        train_readout(fig.add_subplot(gs[1, 1:]), sfd, so["events"], valid_names, order,
                      f"A source-core run: patient-electrode |LFP| TRAIN ({len(so['events'])} events; k_dir=2 dir {sens['source'][0][0]}/{sens['source'][0][1]})")
    if rep_k:
        panel_gradient(fig.add_subplot(gs[2, 0]), kfd["posE"], rep_k["onset"], reg, kfd["contacts"], kfd["names"],
                       f"B sink-core MODEL: source-space propagation (rep event)")
        train_readout(fig.add_subplot(gs[2, 1:]), kfd, ko["events"], valid_names, order,
                      f"B sink-core run: patient-electrode |LFP| TRAIN ({len(ko['events'])} events; k_dir=2 dir {sens['sink'][0][0]}/{sens['sink'][0][1]})")

    # C-left: ACTUAL kmeans cluster-grouped heatmap
    axc = fig.add_subplot(gs[3, :2])
    sh_idx = [i for i in order if shared[i]]
    if labels.size == len(events) or (valid_ev is not None and labels.size == len(valid_ev)):
        ev_idx = np.arange(len(events)) if labels.size == len(events) else np.asarray(valid_ev)
        evorder = ev_idx[np.argsort(labels)]
        im = axc.imshow(R[np.ix_(sh_idx, evorder)], aspect="auto", cmap="viridis", interpolation="nearest")
        bnd = np.searchsorted(np.sort(labels), np.unique(labels)[1:])
        for b in bnd:
            axc.axvline(b - 0.5, color="r", lw=2)
        axc.set_xlabel("events grouped by UNSUPERVISED KMeans cluster (red=boundary)")
    else:
        im = axc.imshow(R[np.ix_(sh_idx, np.arange(len(events)))], aspect="auto", cmap="viridis")
        axc.set_xlabel("events")
    axc.set_yticks(range(len(sh_idx))); axc.set_yticklabels([valid_names[i] for i in sh_idx], fontsize=7)
    axc.set_ylabel("shared contacts (source->sink)")
    plt.colorbar(im, ax=axc, fraction=0.02, pad=0.01, label="within-event rank")
    axc.set_title(f"C  KMeans k=2 on full montage: inter-corr(imputed)={inter:+.2f}  ({conf_txt})", fontsize=9)

    # C-right: confusion matrix + honest metrics text
    axt = fig.add_subplot(gs[3, 2]); axt.axis("off")
    if 'conf' in dir() and conf.sum() > 0:
        axi = axt.inset_axes([0.1, 0.55, 0.8, 0.4])
        axi.imshow(conf, cmap="Blues")
        for (i, j), v in np.ndenumerate(conf):
            axi.text(j, i, str(v), ha="center", va="center", fontsize=11)
        axi.set_xticks([0, 1]); axi.set_xticklabels(["src", "sink"], fontsize=7)
        axi.set_yticks([0, 1]); axi.set_yticklabels(["clu0", "clu1"], fontsize=7)
        axi.set_title("KMeans x origin", fontsize=8)
    txt = (f"HONEST METRICS\n"
           f"shared contacts: {int(shared.sum())}/{len(valid_names)}\n"
           f"source rank-vs-axis: {src_axis:+.2f}\n"
           f"sink rank-vs-axis:  {snk_axis:+.2f}\n"
           f"source-vs-sink (reversed): {shared_corr:+.2f}\n\n"
           f"k_dir SENSITIVITY (fwd/rev):\n"
           f" source k2={sens['source'][0]} k3={sens['source'][1]}\n"
           f" sink   k2={sens['sink'][0]} k3={sens['sink'][1]}\n"
           f" -> forward leg needs k_dir=2\n\n"
           f"kmeans inter-corr (imputed,\n  non-shared-contact artifact): {inter:+.2f}")
    axt.text(0.0, 0.5, txt, fontsize=8, va="top", family="monospace")

    fig.suptitle(f"Subject SNN separate-then-pool (INSTRUMENT-ALIGNMENT, not spontaneous): {a.label or subj}  "
                 f"[{so['anchor']}, inter-core {so['inter_core_sheet']}mm, k_dir={k_dir}]", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    figdir = os.path.join(a.run_dir, "figures"); os.makedirs(figdir, exist_ok=True)
    fp = os.path.join(figdir, f"core_model_pooled_{subj}.png")
    fig.savefig(fp, dpi=130, bbox_inches="tight"); print(f"[written] {fp}")


if __name__ == "__main__":
    main()
