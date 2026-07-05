#!/usr/bin/env python3
"""Topic 5 — class-field vs template-field, 6-panel comparison (one figure per subject).

User ask (2026-06-25): per subject show 6 fields in the A-line left-panel form on one physical
plane — the 2 aggregate-template fields (A/B), the 2 event-aggregated class fields (A/B), and the
2 pre-ictal activation fields at different lead times. Layout (2 rows x 3 cols):

    template A   |  template B   |  pre-ictal  -10..0 s
    class A      |  class B      |  pre-ictal -120..-90 s

Column 1/2 let you read template-vs-class (top vs bottom) for each propagation class; column 3
shows the two pre-ictal targets the max-AB statistic is computed against. All viridis, contacts as
dots, clinical seizure-onset contacts ringed black (overlay only). Broad 20-contact substrate,
common template-A display frame. EXPLORATORY secondary; NOT a replay claim.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import src.topic5_event_resolved_alignment as erm
from src.rank_displacement import swap_node_groups_at_k
from scripts.plot_contact_plane_static import (_subject_display_frame, _display_points,
                                               _smooth_rank_field_mm, _attach_real_coords)

GEOM_BROAD = _ROOT / "results/spatial_modulation/propagation_geometry_broad/observation_readout/real_subjects"
BROAD_RD = _ROOT / "results/interictal_propagation_masked_broad/rank_displacement/per_subject"
WIN_CACHES = _ROOT / "results/topic5_ictal_recruitment/axis_alignment/window_caches"
OUT = _ROOT / "results/topic5_ictal_recruitment/event_resolved_alignment/class_vs_template/figures"
PRE = [("pre_prox_m10_0", "pre-ictal  -10..0 s"), ("pre_distal_m120_m90", "pre-ictal  -120..-90 s")]
SRC_A_COL, SRC_B_COL = "#d62728", "#1f77b4"   # swap node: source-in-A (red) / source-in-B (blue)
LABEL_HALO = [pe.withStroke(linewidth=1.0, foreground="black")]   # subtle label outline


def _swap_legend_handles():
    return [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor=SRC_A_COL,
                       markeredgewidth=2.4, markersize=12, label="swap node — source in template A (leads A, trails B)"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor=SRC_B_COL,
                       markeredgewidth=2.4, markersize=12, label="swap node — source in template B (leads B, trails A)"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="0.6", markeredgecolor="k",
                       markersize=10, label="clinical seizure-onset contact (overlay)")]


def _broad_swap_groups(ds_sid):
    """(set source-in-A names, set source-in-B names, swap_class) from BROAD rank-displacement swap_sweep."""
    f = BROAD_RD / f"{ds_sid}.json"
    if not f.exists():
        return set(), set(), None
    d = json.load(open(f)); pp = d.get("primary_pair") or (d.get("pairs") or [{}])[0]
    ss = pp.get("swap_sweep", {}); k = ss.get("decision_k")
    if ss.get("swap_class") not in ("strict", "candidate") or not k:
        return set(), set(), ss.get("swap_class")
    jv = np.asarray(pp["joint_valid"], bool)
    ra = np.asarray(pp["rank_a_dense_full"], float); rb = np.asarray(pp["rank_b_dense_full"], float)
    sa, sb = swap_node_groups_at_k(ra, rb, jv, jv, int(k))
    nm = pp["channel_names"]
    return {nm[i] for i in sa}, {nm[i] for i in sb}, ss.get("swap_class")


def _rank01(vals):
    v = np.asarray(vals, float); out = np.full(v.shape, np.nan); ok = np.isfinite(v)
    if ok.sum() >= 2:
        out[ok] = np.argsort(np.argsort(v[ok])) / (ok.sum() - 1)
    elif ok.sum() == 1:
        out[ok] = 0.5
    return out


def _window_mean(ds_sid, window, key="bb_auc"):
    npz, mj = WIN_CACHES / window / f"{ds_sid}.npz", WIN_CACHES / window / f"{ds_sid}.json"
    if not npz.exists() or not mj.exists():
        return {}
    data = np.load(npz, allow_pickle=True); meta = json.load(open(mj))
    names = [str(x) for x in data["channels"]]
    arrs = [np.asarray(data[f"{key}__{i}"], float) for i in meta.get("eligible_idxs", [])
            if f"{key}__{i}" in data.files]
    if not arrs:
        return {}
    return {n: float(v) for n, v in zip(names, np.nanmean(np.vstack(arrs), axis=0)) if np.isfinite(v)}


def _panel(ax, xs, ys, vals, support, xlim, ylim, sigma, title, cbar_label, soz,
           names=None, src_a=None, src_b=None):
    _, _, T, _, _ = _smooth_rank_field_mm(xs, ys, vals, support, xlim, ylim, sigma)
    im = ax.imshow(T, origin="lower", extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                   aspect="equal", cmap="viridis", vmin=0, vmax=1)
    okv = np.isfinite(vals)
    xx, yy = np.asarray(xs)[okv], np.asarray(ys)[okv]
    soz_ok = [z for z, v in zip(soz, vals) if np.isfinite(v)]
    ax.scatter(xx, yy, c=np.asarray(vals)[okv], cmap="viridis", vmin=0, vmax=1, s=58, zorder=3,
               edgecolors=["k" if z else "white" for z in soz_ok],
               linewidths=[1.6 if z else 1.0 for z in soz_ok])      # thicker electrode frame
    # swap-node rings, colored by source role (same contact -> same color in every panel)
    if names is not None and (src_a or src_b):
        nn = [n for n, v in zip(names, vals) if np.isfinite(v)]
        for grp, col in ((src_a or set(), SRC_A_COL), (src_b or set(), SRC_B_COL)):
            m = np.array([n in grp for n in nn])
            if m.any():
                pc = ax.scatter(xx[m], yy[m], s=185, facecolors="none", edgecolors=col,
                                linewidths=2.6, zorder=5)
                pc.set_path_effects([pe.withStroke(linewidth=5.0, foreground="black")])   # black outline -> pops
        sw = (src_a or set()) | (src_b or set())
        for x, y, n in zip(xx, yy, nn):
            if n in sw:
                ax.text(x, y + (ylim[1] - ylim[0]) * 0.028, n, ha="center", va="bottom",
                        fontsize=5.5, color="0.92", path_effects=LABEL_HALO, zorder=6)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label(cbar_label, fontsize=8)


def plot_subject(ds_sid):
    dataset, subject = ds_sid.split("_", 1)
    ta_f, tb_f = GEOM_BROAD / f"{ds_sid}_t_a.json", GEOM_BROAD / f"{ds_sid}_t_b.json"
    if not (ta_f.exists() and tb_f.exists()):
        print(f"[skip] {ds_sid}: no broad planes"); return None
    try:
        bundle = erm.load_event_labels_ranks(dataset, subject)
    except Exception as e:
        print(f"[skip] {ds_sid}: {e}"); return None
    plane_a = json.loads(ta_f.read_text()); plane_b = json.loads(tb_f.read_text())
    order = bundle["channel_names"]
    ta_rank = np.array([{c["name"]: c.get("typical_rank") for c in plane_a["channels"]}.get(n, np.nan) for n in order], float)
    tb_rank = np.array([{c["name"]: c.get("typical_rank") for c in plane_b["channels"]}.get(n, np.nan) for n in order], float)
    cmap = erm.map_clusters_to_templates(np.array(bundle["cluster_template_ranks"][0], float),
                                         np.array(bundle["cluster_template_ranks"][1], float), ta_rank, tb_rank)
    if cmap["ambiguous"]:
        print(f"[skip] {ds_sid}: ambiguous cluster map"); return None
    label_A = [k for k, t in cmap["map"].items() if t == "t_a"][0]
    label_B = [k for k, t in cmap["map"].items() if t == "t_b"][0]
    cvA = erm.class_aggregate_contact_values(bundle, label_A)
    cvB = erm.class_aggregate_contact_values(bundle, label_B)
    nA = int(np.sum(bundle["labels"] == label_A)); nB = int(np.sum(bundle["labels"] == label_B))

    # one shared display frame from the broad t_a record
    _attach_real_coords([plane_a])
    frame = _subject_display_frame([plane_a])
    if frame is None:
        print(f"[skip] {ds_sid}: no frame"); return None
    xs, ys = _display_points(plane_a, frame)
    names = [c["name"] for c in plane_a["channels"]]
    soz = np.array([bool(c.get("is_soz")) for c in plane_a["channels"]])
    xlim, ylim, sigma = frame["xlim"], frame["ylim"], frame["sigma_mm"]
    src_a, src_b, swap_class = _broad_swap_groups(ds_sid)   # role-swap nodes (red=source A / blue=source B)

    def vals_sup(value_map, support_map=None):
        v = _rank01([value_map.get(n, np.nan) for n in names])
        s = (np.array([support_map.get(n, 0.0) for n in names], float) if support_map is not None
             else np.where(np.isfinite(v), 1.0, 0.0))
        return v, s

    tplA, sA = vals_sup({c["name"]: c["typical_rank"] for c in plane_a["channels"]},
                        {c["name"]: c.get("support", 1.0) for c in plane_a["channels"]})
    tplB, sB = vals_sup({c["name"]: c["typical_rank"] for c in plane_b["channels"]},
                        {c["name"]: c.get("support", 1.0) for c in plane_b["channels"]})
    clsA, scA = vals_sup({n: d["value"] for n, d in cvA.items()}, {n: d["support"] for n, d in cvA.items()})
    clsB, scB = vals_sup({n: d["value"] for n, d in cvB.items()}, {n: d["support"] for n, d in cvB.items()})
    pre1, sp1 = vals_sup(_window_mean(ds_sid, PRE[0][0]))
    pre2, sp2 = vals_sup(_window_mean(ds_sid, PRE[1][0]))

    sw = dict(names=names, src_a=src_a, src_b=src_b)
    fig, ax = plt.subplots(2, 3, figsize=(17.5, 10.5), constrained_layout=True)
    _panel(ax[0, 0], xs, ys, tplA, sA, xlim, ylim, sigma, "aggregate template A", "early(0)->late(1)", soz, **sw)
    _panel(ax[0, 1], xs, ys, tplB, sB, xlim, ylim, sigma, "aggregate template B", "early(0)->late(1)", soz, **sw)
    _panel(ax[0, 2], xs, ys, pre1, sp1, xlim, ylim, sigma, PRE[0][1], "activation low(0)->high(1)", soz, **sw)
    _panel(ax[1, 0], xs, ys, clsA, scA, xlim, ylim, sigma, f"class A field (all {nA} events)", "early(0)->late(1)", soz, **sw)
    _panel(ax[1, 1], xs, ys, clsB, scB, xlim, ylim, sigma, f"class B field (all {nB} events)", "early(0)->late(1)", soz, **sw)
    _panel(ax[1, 2], xs, ys, pre2, sp2, xlim, ylim, sigma, PRE[1][1], "activation low(0)->high(1)", soz, **sw)
    for a in (ax[1, 0], ax[1, 1], ax[1, 2]):
        a.set_xlabel("along template-A axis (mm)")
    for a in (ax[0, 0], ax[1, 0]):
        a.set_ylabel("transverse (mm)")
    pretty = ds_sid.replace("epilepsiae_", "E").replace("yuquan_", "Y-")
    swtag = (f"swap={swap_class}, {len(src_a) + len(src_b)} nodes" if (src_a or src_b)
             else f"swap={swap_class or 'none'}")
    fig.suptitle(f"Patient {pretty} — aggregate templates vs event-aggregated class fields vs pre-ictal, "
                 f"role-SWAP nodes ringed ({swtag}; broad plane; EXPLORATORY, NOT a replay claim)", fontsize=12)
    fig.text(0.5, 0.038, "col 1/2 = propagation class A/B (top: aggregate template · bottom: all that class's events, "
             "weight-normalized);  col 3 = pre-ictal activation targets.", ha="center", fontsize=8.5, color="0.35")
    fig.legend(handles=_swap_legend_handles(), loc="lower center", ncol=3, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.005))     # legend OUTSIDE the panels
    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / f"{ds_sid}_class_vs_template_fields.png"
    fig.savefig(fp, dpi=125, bbox_inches="tight"); plt.close(fig)
    return fp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()
    subs = args.subjects or sorted(p.stem.replace("_t_a", "") for p in GEOM_BROAD.glob("*_t_a.json"))
    made = []
    for ds_sid in subs:
        fp = plot_subject(ds_sid)
        if fp:
            print(f"[fig] {fp.name}"); made.append(fp.name)
    print(f"[done] {len(made)} figures -> {OUT}")


if __name__ == "__main__":
    main()
