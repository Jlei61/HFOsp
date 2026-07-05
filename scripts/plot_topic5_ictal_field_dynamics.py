"""Topic 5 发作内 field 动力学 — 两层图，复用 field_vs_ictal_swap 的渲染 (_field_panel)。
每张 field 图都锚到该 subject 发作前 frame；红=source in A / 蓝=source in B（分布式 swap source 集合）。
设计: docs/superpowers/specs/2026-06-28-topic5-ictal-field-dynamics-design.md §6（含 P1 修复 + 第三轮 review）。

per-seizure (dur>=40s, 非 parity_fail) 3 张:
  1) 时刻演化行  [间期A | 间期B | ictal 0/25/50/75/100%]
  2) 指标随进程曲线 (band=bb, ictal_fraction>=0.5)
  3) 终止对齐场行 [间期A | 间期B | off -45/-20/-5/+15s]
subject-level 4 张:
  1) 间期 A|B 锚 (大图, labels+cbar)
  2) 全发作平均早期 ictal 场
  3) 进程 summary (spaghetti+median)
  4) 终止 summary
走廊 (axial_mid) 为空的 subject: 对应曲线标 'corridor n=0 (not measurable)'。
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.run_topic5_ictal_field_dynamics import CACHE, OUT_BY_SUB, SUBJECTS_BY_SUB, _load_subject
from scripts.plot_topic5_swap_nodes_fields import _arrays, _legend_handles
from scripts.plot_topic5_field_vs_ictal_swap import _field_panel, _rank01

OUT = OUT_BY_SUB["broad"]   # rebound per --substrate in main()

PROG_KEYS = [("align_maxab", "field-axis align (maxAB)"), ("pms_axial_mid", "axial-mid pos-mass share"),
             ("pms_non_axial", "non-axial pos-mass share"), ("sync_median_corr", "synchrony (median corr)")]
OFFSET_CENTERS = [(-45, "off -45s"), (-20, "off -20s"), (-5, "off -5s"), (15, "off +15s")]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _dats(ds_sid, substrate):
    """dat_a / dat_b (= field_vs_ictal _field_panel inputs on shared frame) + interictal A/B rank01.
    ungated(swap + non-swap) via _load_subject;红蓝圈 src_a/src_b = swap 集合或 template-earliest top-K。"""
    ta, tb, frame, src_a, src_b, swap_class, decision_k = _load_subject(ds_sid, substrate)
    dat0 = dict(ds_sid=ds_sid, ta=ta, tb=tb, frame=frame, src_a=src_a, src_b=src_b,
                ss={"swap_class": swap_class, "decision_k": decision_k})
    na, xa, ya, ia, sa, za = _arrays(ta, frame)
    nb, xb, yb, ib, sb, zb = _arrays(tb, frame)
    dat_a = {**dat0, "names": na, "xs": np.asarray(xa), "ys": np.asarray(ya), "sup": sa, "soz": za}
    dat_b = {**dat0, "names": nb, "xs": np.asarray(xb), "ys": np.asarray(yb), "sup": sb, "soz": zb}
    return dat0, dat_a, dat_b, ia, ib


def _moment_vals(names, cidx, zt, relt, lo, hi):
    m = (relt >= lo) & (relt <= hi)
    if not m.any():
        return _rank01([np.nan] * len(names))
    zm = np.nanmean(zt[:, m], axis=1)
    return _rank01([zm[cidx[n]] if n in cidx else np.nan for n in names])


def _spaghetti(ax, by_sz, key, bins):
    for sz, rs in by_sz.items():
        rs = sorted(rs, key=lambda r: _f(r["progress_frac"]))
        ax.plot([_f(r["progress_frac"]) for r in rs], [_f(r[key]) for r in rs], color="0.78", lw=0.8)
    allr = [r for rs in by_sz.values() for r in rs]
    mx, my = [], []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        vals = [_f(r[key]) for r in allr if b0 <= _f(r["progress_frac"]) < b1 and np.isfinite(_f(r[key]))]
        if vals:
            mx.append((b0 + b1) / 2); my.append(np.nanmedian(vals))
    if my:
        ax.plot(mx, my, "k-o", lw=2)
    else:
        ax.text(0.5, 0.5, "corridor n=0\n(not measurable)" if key == "pms_axial_mid" else "n/a",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="0.4")


def _animate(ds_sid, dat_a, dat_b, ia, ib, data, meta, cidx, suphdr):
    """Field-evolution GIF: [interictal A | interictal B | sliding ictal field], onset->offset+30s.
    场图是示意性的——z-ER 对发作前 baseline 归一，发作中后期 reliability 下降；这里画窗内 rank01(相对空间形状)。"""
    cand = [(meta["seizure"][str(i)]["eeg_duration_sec"], i) for i in meta["eligible_idxs"]
            if meta["seizure"][str(i)]["eeg_duration_sec"] >= 40
            and f"bb_zt__{i}" in data.files]
    if not cand:
        return None
    dur, idx = max(cand)
    s = meta["seizure"][str(idx)]; off = s["eeg_offset_rel"]
    zt, relt = data[f"bb_zt__{idx}"], data[f"bb_relt__{idx}"]
    win, step = 10.0, 1.5   # 细步长 + 高重叠 -> 帧间场变化平滑（减少跳变）
    los = list(np.arange(0.0, max(off + 30.0 - win, step), step))
    fig, ax = plt.subplots(1, 3, figsize=(12.6, 4.6), layout="constrained")
    _field_panel(ax[0], dat_a, ia, "interictal A", "", compact=True)
    _field_panel(ax[1], dat_b, ib, "interictal B", "", compact=True)

    def update(k):
        lo = los[k]; tc = lo + win / 2.0
        ax[2].clear()
        vals = _moment_vals(dat_a["names"], cidx, zt, relt, lo, lo + win)
        post = "  POST-OFFSET" if tc > off else ""
        prog = int(100 * min(tc / off, 1.0)) if off > 0 else 0
        _field_panel(ax[2], dat_a, vals, f"ictal t={tc:.0f}s  ({prog}% progress){post}", "", compact=True)

    fig.suptitle(f"{ds_sid} sz{idx} (dur {dur:.0f}s) — field evolution onset→offset (illustrative:\n"
                 f"window rank01 of baseline-z; z-ER vs pre-seizure baseline less reliable late-seizure)\n{suphdr}",
                 fontsize=11)
    anim = FuncAnimation(fig, update, frames=len(los), interval=125)
    gif = OUT / "figures" / f"{ds_sid}_field_evolution.gif"
    anim.save(gif, writer=PillowWriter(fps=8)); plt.close(fig)
    print(f"  [anim] {ds_sid}: sz{idx} ({dur:.0f}s, {len(los)} frames) -> {gif.name}", flush=True)
    return idx


def plot_subject(ds_sid, rows, substrate):
    fdir = OUT / "figures"; fdir.mkdir(parents=True, exist_ok=True)
    psdir = OUT / "figures" / "per_seizure" / ds_sid; psdir.mkdir(parents=True, exist_ok=True)
    dat, dat_a, dat_b, ia, ib = _dats(ds_sid, substrate)
    data = np.load(CACHE / f"{ds_sid}.npz", allow_pickle=True)
    meta = json.load(open(CACHE / f"{ds_sid}.json"))
    cache_names = [str(x) for x in data["channels"]]
    cidx = {n: i for i, n in enumerate(cache_names)}
    ss = dat["ss"]
    suphdr = f"{ds_sid} (swap={ss.get('swap_class')}, k={ss.get('decision_k')}; red=source in A, blue=source in B)"

    # ---- subject (1): interictal A | B anchor (big) ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.6), layout="constrained")
    _field_panel(ax[0], dat_a, ia, "interictal propagation order — template A", "early (0) -> late (1)",
                 compact=False, labels=True, cbar=True)
    _field_panel(ax[1], dat_b, ib, "interictal propagation order — template B", "early (0) -> late (1)",
                 compact=False, labels=True, cbar=True)
    fig.suptitle(f"{suphdr} — interictal A|B anchor (pre-seizure layout)", fontsize=14)
    fig.legend(handles=_legend_handles(), loc="outside lower center", ncol=3, fontsize=12, frameon=False)
    fig.savefig(fdir / f"{ds_sid}_interictal_AB.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # ---- subject (2): mean early-ictal field (mean [0,10]s z across eligible seizures) ----
    acc = []
    for idx in meta["eligible_idxs"]:
        relt = data[f"bb_relt__{idx}"]; m = (relt >= 0) & (relt <= 10)
        if m.any():
            acc.append(np.nanmean(data[f"bb_zt__{idx}"][:, m], axis=1))
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 6.4), layout="constrained")
    if acc:
        mean_z = np.nanmean(np.vstack(acc), axis=0)
        vals = _rank01([mean_z[cidx[n]] if n in cidx else np.nan for n in dat_a["names"]])
        _field_panel(ax, dat_a, vals, "mean early-ictal activation (0–10s, all eligible sz)",
                     "activation rank", compact=False, labels=True, cbar=True)
    fig.suptitle(suphdr, fontsize=13)
    fig.legend(handles=_legend_handles(), loc="outside lower center", ncol=3, fontsize=11, frameon=False)
    fig.savefig(fdir / f"{ds_sid}_mean_ictal.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # ---- subject (3,4): progress + offset summaries (bb, ictal_fraction>=0.5) ----
    onset = [r for r in rows if r["window_kind"] == "onset_slide" and r["band"] == "bb"
             and _f(r["ictal_fraction"]) >= 0.5]
    by_sz = defaultdict(list)
    for r in onset:
        by_sz[r["seizure_idx"]].append(r)
    bins = np.linspace(0, 1, 6)
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.2), layout="constrained")
    for k, (key, lbl) in enumerate(PROG_KEYS):
        _spaghetti(ax[k], by_sz, key, bins); ax[k].set_title(lbl); ax[k].set_xlabel("progress")
    fig.suptitle(f"{ds_sid} — progress summary (spaghetti+median; n_sz={len(by_sz)})")
    fig.savefig(fdir / f"{ds_sid}_progress.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    offa = [r for r in rows if r["window_kind"] == "offset_aligned" and r["band"] == "bb"
            and r["pre_onset_overlap"] == "False"]
    tcs = sorted({_f(r["t_center_rel_offset"]) for r in offa})
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.2), layout="constrained")
    for k, (key, lbl) in enumerate(PROG_KEYS):
        for r in offa:
            ax[k].scatter(_f(r["t_center_rel_offset"]), _f(r[key]), s=10, color="0.6")
        my = [np.nanmedian([_f(r[key]) for r in offa if _f(r["t_center_rel_offset"]) == tc
                            and np.isfinite(_f(r[key]))] or [np.nan]) for tc in tcs]
        ax[k].plot(tcs, my, "k-o", lw=2); ax[k].axvline(0, color="r", ls=":")
        ax[k].set_title(lbl); ax[k].set_xlabel("rel offset (s)")
    fig.suptitle(f"{ds_sid} — termination-aligned summary (offset=0; pre_onset_overlap excluded)")
    fig.savefig(fdir / f"{ds_sid}_offset.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    # ---- per-seizure (dur>=40s, non parity_fail) ----
    rbs = defaultdict(list)
    for r in rows:
        rbs[int(r["seizure_idx"])].append(r)
    n_ps = 0
    for idx in meta["eligible_idxs"]:
        s = meta["seizure"][str(idx)]; off = s["eeg_offset_rel"]
        if s["eeg_duration_sec"] < 40 or idx not in rbs:
            continue
        if any(r["parity_fail"] == "True" for r in rbs[idx]):
            continue
        zt, relt = data[f"bb_zt__{idx}"], data[f"bb_relt__{idx}"]
        # (1) moment-evolution row
        fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
        fig, ax = plt.subplots(1, 2 + len(fracs), figsize=(3.05 * (2 + len(fracs)), 4.1), layout="constrained")
        _field_panel(ax[0], dat_a, ia, "interictal A", "", compact=True)
        _field_panel(ax[1], dat_b, ib, "interictal B", "", compact=True)
        for j, fr in enumerate(fracs):
            lo = float(np.clip(fr * off - 5.0, 0.0, max(off - 10.0, 0.0)))
            _field_panel(ax[2 + j], dat_a, _moment_vals(dat_a["names"], cidx, zt, relt, lo, lo + 10),
                         f"ictal ~{int(fr*100)}%", "", compact=True)
        fig.suptitle(f"{ds_sid} sz{idx} (dur {s['eeg_duration_sec']:.0f}s) — within-seizure field evolution\n{suphdr}",
                     fontsize=12)
        fig.legend(handles=_legend_handles(), loc="outside lower center", ncol=3, fontsize=10, frameon=False)
        fig.savefig(psdir / f"{ds_sid}_sz{idx}_evolution.png", dpi=115, bbox_inches="tight"); plt.close(fig)
        # (2) metric trajectory
        rs = sorted([r for r in rbs[idx] if r["window_kind"] == "onset_slide" and r["band"] == "bb"
                     and _f(r["ictal_fraction"]) >= 0.5], key=lambda r: _f(r["progress_frac"]))
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.6), layout="constrained")
        prog = [_f(r["progress_frac"]) for r in rs]
        for key, lbl in PROG_KEYS:
            ax.plot(prog, [_f(r[key]) for r in rs], marker="o", ms=3, label=lbl)
        ax.set_xlabel("progress (onset->offset)"); ax.set_title(f"{ds_sid} sz{idx} — metrics vs progress (bb)")
        ax.legend(fontsize=8)
        fig.savefig(psdir / f"{ds_sid}_sz{idx}_trajectory.png", dpi=115, bbox_inches="tight"); plt.close(fig)
        # (3) termination field row
        fig, ax = plt.subplots(1, 2 + len(OFFSET_CENTERS), figsize=(3.05 * (2 + len(OFFSET_CENTERS)), 4.1),
                               layout="constrained")
        _field_panel(ax[0], dat_a, ia, "interictal A", "", compact=True)
        _field_panel(ax[1], dat_b, ib, "interictal B", "", compact=True)
        for j, (c, lbl) in enumerate(OFFSET_CENTERS):
            _field_panel(ax[2 + j], dat_a, _moment_vals(dat_a["names"], cidx, zt, relt, off + c - 5, off + c + 5),
                         lbl, "", compact=True)
        fig.suptitle(f"{ds_sid} sz{idx} — termination-aligned field (offset=0)\n{suphdr}", fontsize=12)
        fig.legend(handles=_legend_handles(), loc="outside lower center", ncol=3, fontsize=10, frameon=False)
        fig.savefig(psdir / f"{ds_sid}_sz{idx}_termination.png", dpi=115, bbox_inches="tight"); plt.close(fig)
        n_ps += 1
    _animate(ds_sid, dat_a, dat_b, ia, ib, data, meta, cidx, suphdr)
    return n_ps


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--substrate", choices=list(OUT_BY_SUB), default="broad")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--anim-only", action="store_true", help="只重渲染 GIF 动画（不重做 PNG）")
    args = ap.parse_args()
    OUT = OUT_BY_SUB[args.substrate]
    subjects = args.subjects or SUBJECTS_BY_SUB[args.substrate]
    rows_all = list(csv.DictReader(open(OUT / "per_seizure_metrics.csv")))
    for ds_sid in subjects:
        if not (CACHE / f"{ds_sid}.npz").exists():
            print(f"[skip] {ds_sid} no cache", flush=True); continue
        if args.anim_only:
            dat, dat_a, dat_b, ia, ib = _dats(ds_sid, args.substrate)
            data = np.load(CACHE / f"{ds_sid}.npz", allow_pickle=True)
            meta = json.load(open(CACHE / f"{ds_sid}.json"))
            cidx = {n: i for i, n in enumerate(str(x) for x in data["channels"])}
            ss = dat["ss"]
            suphdr = (f"{ds_sid} (swap={ss.get('swap_class')}, k={ss.get('decision_k')}; "
                      f"red=source in A, blue=source in B)")
            _animate(ds_sid, dat_a, dat_b, ia, ib, data, meta, cidx, suphdr)
            continue
        rows = [r for r in rows_all if r["ds_sid"] == ds_sid]
        n_ps = plot_subject(ds_sid, rows, args.substrate)
        print(f"[fig] {ds_sid}: 4 subject-level + {n_ps}x3 per-seizure", flush=True)
    print("FIGS DONE", flush=True)


if __name__ == "__main__":
    main()
