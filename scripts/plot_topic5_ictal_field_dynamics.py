"""Topic 5 发作内 field 动力学 — 两层图（per-seizure composite + subject-level 聚合）。
复用 field-vs-ictal 的 display frame / 平滑 / 色标逻辑（_smooth_rank_field_mm + viridis）。
设计: docs/superpowers/specs/2026-06-28-topic5-ictal-field-dynamics-design.md §6（含 P1 修复）。"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.run_topic5_ictal_field_dynamics import load_context, CACHE, OUT, SUBJECTS, _slice, _zmean_by_name
from scripts.plot_contact_plane_static import _smooth_rank_field_mm

GROUP_COL = {"source_core": "#d62728", "axial_mid": "#ffcc00",
             "non_axial": "#1f77b4", "axis_end_noncore": "#999999"}
PROG_KEYS = [("align_maxab", "field-axis align (maxAB)"), ("pms_axial_mid", "axial-mid pos-mass share"),
             ("pms_non_axial", "non-axial pos-mass share"), ("sync_median_corr", "synchrony (median corr)")]


def _f(x):
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return float("nan")


def _rank01(v):
    v = np.asarray(v, float); out = np.full(v.shape, np.nan); ok = np.isfinite(v)
    if ok.sum() >= 2:
        out[ok] = np.argsort(np.argsort(v[ok])) / (ok.sum() - 1)
    elif ok.sum() == 1:
        out[ok] = 0.5
    return out


def _field_ax(ax, ctx, zmean_by_name, title, *, show_field=True):
    frame = ctx["frame"]; xlim, ylim, sigma = frame["xlim"], frame["ylim"], frame["sigma_mm"]
    names = list(ctx["mapped"])
    xs = np.array([ctx["pos"][n][0] for n in names]); ys = np.array([ctx["pos"][n][1] for n in names])
    if show_field:
        vals = _rank01([zmean_by_name.get(n, np.nan) for n in names])
        _, _, T, _, _ = _smooth_rank_field_mm(xs, ys, vals, np.ones_like(xs), xlim, ylim, sigma)
        ax.imshow(T, origin="lower", extent=[xlim[0], xlim[1], ylim[0], ylim[1]], aspect="equal",
                  cmap="viridis", vmin=0, vmax=1)
    g = ctx["part"]["groups"]
    ax.scatter(xs, ys, c=[GROUP_COL.get(g.get(n), "w") for n in names], s=46, edgecolors="k",
               linewidths=0.7, zorder=3)
    PA, PB = ctx["part"]["P_A"], ctx["part"]["P_B"]
    ax.plot([PA[0], PB[0]], [PA[1], PB[1]], "w--" if show_field else "k--", lw=1.8, zorder=4)
    ax.set_title(title, fontsize=10); ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xticks([]); ax.set_yticks([])


def _seizure_snaps(ctx, idx, meta, data, cache_names):
    s = meta["seizure"][str(idx)]; off = s["eeg_offset_rel"]
    zt, relt = data[f"bb_zt__{idx}"], data[f"bb_relt__{idx}"]
    snaps = []
    for frac in (0.0, 0.33, 0.66, 1.0):
        lo = max(0.0, frac * off - 5.0)
        zmv, _ = _slice(zt, relt, lo, lo + 10.0)
        if zmv is not None:
            snaps.append((frac, _zmean_by_name(zmv, cache_names, ctx["mapped"])))
    return snaps


def plot_per_seizure(ds_sid, ctx, meta, data, cache_names, rows_by_sz):
    fig_dir = OUT / "figures" / "per_seizure" / ds_sid
    fig_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for idx in meta["eligible_idxs"]:
        s = meta["seizure"][str(idx)]
        if s["eeg_duration_sec"] < 40 or idx not in rows_by_sz:
            continue
        snaps = _seizure_snaps(ctx, idx, meta, data, cache_names)
        rs = sorted([r for r in rows_by_sz[idx] if r["window_kind"] == "onset_slide"
                     and r["band"] == "bb" and _f(r["ictal_fraction"]) >= 0.5],
                    key=lambda r: _f(r["t_center_rel_onset"]))
        if not rs or not snaps:
            continue
        fig, ax = plt.subplots(2, 4, figsize=(17, 8), layout="constrained")
        for j, (frac, zmn) in enumerate(snaps[:4]):
            _field_ax(ax[0, j], ctx, zmn, f"progress {int(frac*100)}%")
        prog = [_f(r["progress_frac"]) for r in rs]
        for key, lbl in PROG_KEYS:
            ax[1, 0].plot(prog, [_f(r[key]) for r in rs], marker="o", ms=3, label=lbl)
        ax[1, 0].set_title("metrics vs progress"); ax[1, 0].set_xlabel("progress"); ax[1, 0].legend(fontsize=7)
        for key, col in (("pms_source_core", GROUP_COL["source_core"]),
                         ("pms_axial_mid", GROUP_COL["axial_mid"]),
                         ("pms_non_axial", GROUP_COL["non_axial"])):
            ax[1, 1].plot(prog, [_f(r[key]) for r in rs], marker="o", ms=3, color=col, label=key)
        ax[1, 1].set_title("pos-mass share vs progress"); ax[1, 1].legend(fontsize=7)
        offs = sorted([r for r in rows_by_sz[idx] if r["window_kind"] == "offset_aligned"
                       and r["band"] == "bb" and r["pre_onset_overlap"] == "False"],
                      key=lambda r: _f(r["t_center_rel_offset"]))
        if offs:
            ax[1, 2].plot([_f(r["t_center_rel_offset"]) for r in offs],
                          [_f(r["pms_axial_mid"]) for r in offs], "o-", color=GROUP_COL["axial_mid"], label="axial_mid")
            ax[1, 2].plot([_f(r["t_center_rel_offset"]) for r in offs],
                          [_f(r["pms_non_axial"]) for r in offs], "o-", color=GROUP_COL["non_axial"], label="non_axial")
            ax[1, 2].legend(fontsize=7)
        ax[1, 2].axvline(0, color="k", ls=":"); ax[1, 2].set_title("offset zoom (rel offset s)")
        ax[1, 3].plot(prog, [_f(r["sync_median_corr"]) for r in rs], "o-", color="purple")
        ax[1, 3].set_title("synchrony vs progress"); ax[1, 3].set_xlabel("progress")
        fig.suptitle(f"{ds_sid} sz{idx} (dur {s['eeg_duration_sec']:.0f}s) - within-seizure field dynamics (bb)", fontsize=13)
        fig.savefig(fig_dir / f"{ds_sid}_sz{idx}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
        n += 1
    return n


def plot_subject_level(ds_sid, ctx, rows, subj):
    fdir = OUT / "figures"; fdir.mkdir(parents=True, exist_ok=True)
    onset = [r for r in rows if r["window_kind"] == "onset_slide" and r["band"] == "bb"
             and _f(r["ictal_fraction"]) >= 0.5]
    by_sz = defaultdict(list)
    for r in onset:
        by_sz[r["seizure_idx"]].append(r)
    bins = np.linspace(0, 1, 6)
    # (1) progress summary
    fig, ax = plt.subplots(1, len(PROG_KEYS), figsize=(5 * len(PROG_KEYS), 4), layout="constrained")
    for k, (key, lbl) in enumerate(PROG_KEYS):
        for sz, rs in by_sz.items():
            rs = sorted(rs, key=lambda r: _f(r["progress_frac"]))
            ax[k].plot([_f(r["progress_frac"]) for r in rs], [_f(r[key]) for r in rs], color="0.75", lw=0.8)
        mx, my = [], []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            vals = [_f(r[key]) for r in onset if b0 <= _f(r["progress_frac"]) < b1 and np.isfinite(_f(r[key]))]
            if vals:
                mx.append((b0 + b1) / 2); my.append(np.nanmedian(vals))
        ax[k].plot(mx, my, "k-o", lw=2); ax[k].set_title(lbl); ax[k].set_xlabel("progress")
    fig.suptitle(f"{ds_sid} - progress summary (spaghetti+median; n_sz={len(by_sz)}; "
                 f"degen={subj['axis']['axis_degenerate']})")
    fig.savefig(fdir / f"{ds_sid}_progress.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    # (2) offset summary
    offa = [r for r in rows if r["window_kind"] == "offset_aligned" and r["band"] == "bb"
            and r["pre_onset_overlap"] == "False"]
    fig, ax = plt.subplots(1, len(PROG_KEYS), figsize=(5 * len(PROG_KEYS), 4), layout="constrained")
    tcs = sorted({_f(r["t_center_rel_offset"]) for r in offa})
    for k, (key, lbl) in enumerate(PROG_KEYS):
        for r in offa:
            ax[k].scatter(_f(r["t_center_rel_offset"]), _f(r[key]), s=10, color="0.6")
        my = [np.nanmedian([_f(r[key]) for r in offa if _f(r["t_center_rel_offset"]) == tc
                            and np.isfinite(_f(r[key]))] or [np.nan]) for tc in tcs]
        ax[k].plot(tcs, my, "k-o", lw=2); ax[k].axvline(0, color="r", ls=":")
        ax[k].set_title(lbl); ax[k].set_xlabel("rel offset (s)")
    fig.suptitle(f"{ds_sid} - offset-aligned summary (offset=0; pre_onset_overlap excluded)")
    fig.savefig(fdir / f"{ds_sid}_offset.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    # (3) seizure-by-seizure heatmap (pms_axial_mid over progress bins)
    szs = sorted(by_sz)
    if szs:
        M = np.full((len(szs), len(bins) - 1), np.nan)
        for i, sz in enumerate(szs):
            for j, (b0, b1) in enumerate(zip(bins[:-1], bins[1:])):
                vals = [_f(r["pms_axial_mid"]) for r in by_sz[sz]
                        if b0 <= _f(r["progress_frac"]) < b1 and np.isfinite(_f(r["pms_axial_mid"]))]
                if vals:
                    M[i, j] = np.nanmedian(vals)
        fig, axh = plt.subplots(figsize=(7, max(3, 0.32 * len(szs) + 1)), layout="constrained")
        im = axh.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=np.nanmax(M) if np.isfinite(M).any() else 1)
        axh.set_xticks(range(len(bins) - 1)); axh.set_xticklabels([f"{int(b*100)}" for b in bins[:-1]])
        axh.set_yticks(range(len(szs))); axh.set_yticklabels([f"sz{s}" for s in szs], fontsize=7)
        axh.set_xlabel("progress %"); axh.set_title(f"{ds_sid} - axial-mid pos-mass share (rows=seizures)")
        fig.colorbar(im, ax=axh, fraction=0.046, label="pms_axial_mid")
        fig.savefig(fdir / f"{ds_sid}_seizure_heatmap.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    # (4) source geometry QC
    fig, axg = plt.subplots(figsize=(7, 6), layout="constrained")
    _field_ax(axg, ctx, {n: 0.0 for n in ctx["mapped"]}, "", show_field=False)
    ax_ax = subj["axis"]
    axg.set_title(f"{ds_sid} 4-partition + axis QC\n(uncertA/B={ax_ax['source_focus_uncertain_a']}/"
                  f"{ax_ax['source_focus_uncertain_b']}, distA/B={ax_ax['source_top2_dist_a_mm']:.0f}/"
                  f"{ax_ax['source_top2_dist_b_mm']:.0f}mm, degen={ax_ax['axis_degenerate']})", fontsize=10)
    fig.savefig(fdir / f"{ds_sid}_geometry_qc.png", dpi=120, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="*", default=SUBJECTS)
    args = ap.parse_args()
    rows_all = list(csv.DictReader(open(OUT / "per_seizure_metrics.csv")))
    for ds_sid in args.subjects:
        if not (CACHE / f"{ds_sid}.npz").exists():
            print(f"[skip] {ds_sid} no cache", flush=True); continue
        ctx = load_context(ds_sid)
        meta = json.load(open(CACHE / f"{ds_sid}.json"))
        data = np.load(CACHE / f"{ds_sid}.npz", allow_pickle=True)
        cache_names = [str(x) for x in data["channels"]]
        subj = json.load(open(OUT / "per_subject" / f"{ds_sid}.json"))
        rows = [r for r in rows_all if r["ds_sid"] == ds_sid]
        rbs = defaultdict(list)
        for r in rows:
            rbs[int(r["seizure_idx"])].append(r)
        npf = plot_per_seizure(ds_sid, ctx, meta, data, cache_names, rbs)
        plot_subject_level(ds_sid, ctx, rows, subj)
        print(f"[fig] {ds_sid}: {npf} per-seizure + 4 subject-level", flush=True)
    print("FIGS DONE", flush=True)


if __name__ == "__main__":
    main()
