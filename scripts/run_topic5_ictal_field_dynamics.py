"""Topic 5 发作内 field 动力学 — 驱动：loader（非 swap-gated）+ 扫窗写 CSV/JSON。
设计: docs/superpowers/specs/2026-06-28-topic5-ictal-field-dynamics-design.md（含 P1 修复）。"""
from __future__ import annotations
import argparse, csv, json, sys, warnings
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

from src import topic5_ictal_field_dynamics as fd
from src.propagation_contact_plane_readout import (make_plane_grid, R_smooth_rank,
                                                   corr_pair_mirror_invariant, S_THRESH, OVERLAP_MIN)
from src.topic5_axis_alignment import matched_channels, make_field_record
from scripts.plot_topic5_swap_nodes_fields import SUBSTRATE, _subject_data, _arrays
from scripts.plot_contact_plane_static import _subject_display_frame, _attach_real_coords

CACHE = _ROOT / "results/topic5_ictal_recruitment/ictal_field_long_cache"   # substrate-independent (per-subject ictal z)
V2REF = _ROOT / "results/topic5_ictal_recruitment/t0_feature_cache_v2_windows"
OUT_BY_SUB = {"broad": _ROOT / "results/topic5_ictal_recruitment/field_dynamics",
              "narrow": _ROOT / "results/topic5_ictal_recruitment/field_dynamics_narrow"}
OUT = OUT_BY_SUB["broad"]   # rebound per --substrate in main()
# broad = swap-positive(8) + E916(non-swap, 48 sz; 调查证明非 swap 也可用); narrow = parallel batch.
# 走廊/轴的数学只需每模板最早端点(compact core)，不必 swap -> ungated loader 通吃 swap + non-swap。
SUBJECTS_BY_SUB = {
    "broad": ["epilepsiae_139", "epilepsiae_253", "epilepsiae_1077", "epilepsiae_1096", "epilepsiae_1125",
              "epilepsiae_1150", "epilepsiae_620", "epilepsiae_635", "epilepsiae_916"],
    "narrow": ["epilepsiae_1096", "epilepsiae_1125", "epilepsiae_1146", "epilepsiae_253",
               "epilepsiae_384", "epilepsiae_442", "epilepsiae_958"],
}
PARITY_TOL = 1e-3
ONSET_WIN, ONSET_STEP = 10.0, 5.0
OFFSET_WINS = [(-60, -30), (-30, -10), (-10, 0), (0, 30)]
BANDS = ("bb", "hfa")   # bb primary, hfa secondary — 两者都分析


def _abs_corr(Fi, Fj):
    r = corr_pair_mirror_invariant(Fi["T"], Fi["S"], Fj["T"], Fj["S"], S_THRESH, OVERLAP_MIN)["corr"]
    return abs(r) if r is not None and np.isfinite(r) else np.nan


def _load_subject(ds_sid, substrate):
    """UNGATED subject loader (swap + non-swap). Returns ta, tb, frame, src_a_disp, src_b_disp(集合,
    图红蓝圈用), swap_class, decision_k. swap-positive -> 用 swap source 集合作圈; 非 swap -> fallback
    用每模板最早 top-K_disp 集合作圈(K_disp=clip(decision_k,3,n//3))。轴/走廊数学另用 compact core(见 load_context)。"""
    rd_dir, geo_dir = SUBSTRATE[substrate]
    dat = _subject_data(ds_sid, rd_dir, geo_dir)   # swap-gated; None if non-swap
    if dat is not None:
        ss = dat["ss"]
        return dat["ta"], dat["tb"], dat["frame"], set(dat["src_a"]), set(dat["src_b"]), \
            ss.get("swap_class"), ss.get("decision_k")
    rd = json.load(open(rd_dir / f"{ds_sid}.json"))
    pp = rd.get("primary_pair") or rd["pairs"][0]
    sw = (pp.get("swap_sweep") or {})
    decision_k = int(sw.get("decision_k") or 3)
    ta = json.load(open(geo_dir / f"{ds_sid}_t_a.json"))
    tb = json.load(open(geo_dir / f"{ds_sid}_t_b.json"))
    recs = [ta, tb]
    _attach_real_coords(recs)
    frame = _subject_display_frame(recs)

    def _topk(rec):
        rk = sorted(((c["name"], c["typical_rank"]) for c in rec["channels"]
                     if np.isfinite(c.get("typical_rank", np.nan))), key=lambda x: x[1])
        kd = int(np.clip(decision_k, 3, max(3, len(rk) // 3)))
        return set(n for n, _ in rk[:kd])

    return ta, tb, frame, _topk(ta), _topk(tb), sw.get("swap_class"), decision_k


def load_context(ds_sid, substrate="broad"):
    """轴/走廊: core_a/core_b = 每模板最早 COMPACT core(top-2 if <15mm else single; 非 decision_k 整串)。
    图红蓝圈 = src_*_disp(swap 集合或 template-earliest top-K)。interictal A/B 场 = t_a/t_b typical_rank。
    ungated -> swap + non-swap 通吃(E916/narrow non-swap 也能跑)。"""
    ta, tb, frame, src_a_disp, src_b_disp, swap_class, decision_k = _load_subject(ds_sid, substrate)
    names_geo, xs, ys, inter, sup, soz = _arrays(ta, frame)
    pos = {n: (float(x), float(y)) for n, x, y in zip(names_geo, xs, ys)
           if np.isfinite(x) and np.isfinite(y)}
    data = np.load(CACHE / f"{ds_sid}.npz", allow_pickle=True)
    cache_names = [str(x) for x in data["channels"]]
    matched = matched_channels(ta, {n: 0.0 for n in cache_names})
    names_m = [c["name"] for c in matched]
    mapped = [n for n in names_m if n in pos]
    # AXIS/corridor cores = COMPACT extreme-early core of each template (top-2 if <15mm else single).
    # NOT the full swap source SET (k=10 sets eat the whole middle -> axial_mid=0). Full sets kept
    # for figure rings only (display). 139 -> axial_mid=3; bilateral 253 -> 0 (no mid electrodes).
    src_a_full = [n for n in sorted(src_a_disp) if n in pos]
    src_b_full = [n for n in sorted(src_b_disp) if n in pos]
    ta_rank = {c["name"]: c["typical_rank"] for c in ta["channels"]
               if c["name"] in pos and np.isfinite(c.get("typical_rank", np.nan))}
    tb_rank = {c["name"]: c["typical_rank"] for c in tb["channels"]
               if c["name"] in pos and np.isfinite(c.get("typical_rank", np.nan))}
    core_a, uncert_a, dist_a = fd.source_core(sorted(ta_rank, key=ta_rank.get), pos)
    core_b, uncert_b, dist_b = fd.source_core(sorted(tb_rank, key=tb_rank.get), pos)
    part = fd.axis_partition(mapped, pos, core_a, core_b)
    X, Y = make_plane_grid()
    F_inter_a = R_smooth_rank(make_field_record(matched, [float(c["typical_rank"]) for c in matched]),
                              X, Y, None, S_THRESH)
    sigma = F_inter_a["sigma_xy"]
    mb = matched_channels(tb, {n: 0.0 for n in cache_names})
    brank = {c["name"]: float(c["typical_rank"]) for c in mb}
    inter_b = [brank.get(n, np.nan) for n in names_m]
    F_inter_b = (R_smooth_rank(make_field_record(matched, inter_b), X, Y, sigma, S_THRESH)
                 if np.isfinite(inter_b).sum() >= 4 else None)
    return dict(ds_sid=ds_sid, names_m=names_m, mapped=mapped, pos=pos, matched=matched,
                swap_class=swap_class, decision_k=decision_k,
                X=X, Y=Y, sigma=sigma, F_inter_a=F_inter_a, F_inter_b=F_inter_b,
                core_a=core_a, core_b=core_b, uncert_a=uncert_a, uncert_b=uncert_b,
                dist_a=dist_a, dist_b=dist_b, src_a_full=src_a_full, src_b_full=src_b_full,
                part=part, frame=frame, ta=ta, tb=tb)


def window_maxab(ctx, vals_by_name):
    """vals_by_name: {name->raw window-mean z}. 传 raw（与 bb_auc 同路径），R_smooth_rank 内部处理。"""
    vals = [vals_by_name.get(n, np.nan) for n in ctx["names_m"]]
    Fw = R_smooth_rank(make_field_record(ctx["matched"], vals), ctx["X"], ctx["Y"], ctx["sigma"], S_THRESH)
    ra = _abs_corr(ctx["F_inter_a"], Fw)
    if ctx["F_inter_b"] is None:
        return ra
    rb = _abs_corr(ctx["F_inter_b"], Fw)
    v = [x for x in (ra, rb) if np.isfinite(x)]
    return float(max(v)) if v else float("nan")


def _slice(zt, relt, lo, hi):
    m = (relt >= lo) & (relt <= hi)
    if m.sum() == 0:
        return None, None
    return np.nanmean(zt[:, m], axis=1), zt[:, m]


def _ictal_fraction(relt, lo, hi, eeg_offset_rel):
    m = (relt >= lo) & (relt <= hi)
    if m.sum() == 0:
        return float("nan")
    wt = relt[m]
    return float(np.mean((wt >= 0) & (wt <= eeg_offset_rel)))


def _zmean_by_name(zmean_vec, cache_names, mapped):
    idx = {n: i for i, n in enumerate(cache_names)}
    return {n: float(zmean_vec[idx[n]]) for n in mapped
            if n in idx and np.isfinite(zmean_vec[idx[n]])}


def _ztraces_by_name(sub, cache_names, mapped):
    idx = {n: i for i, n in enumerate(cache_names)}
    return {n: sub[idx[n]] for n in mapped if n in idx}


def _metrics_row(ctx, zmean_by_name, ztr_by_name):
    g = ctx["part"]["groups"]
    pms = fd.positive_mass_share(zmean_by_name, g)
    allmean = fd.group_mean(zmean_by_name, {n: "all" for n in zmean_by_name}, "all")
    sc = fd.group_mean(zmean_by_name, g, "source_core")
    am = fd.group_mean(zmean_by_name, g, "axial_mid")
    na = fd.group_mean(zmean_by_name, g, "non_axial")
    en = fd.group_mean(zmean_by_name, g, "axis_end_noncore")
    ang, mag = fd.field_gradient(zmean_by_name, ctx["pos"])

    def share(grp):
        vals = [v for n, v in zmean_by_name.items() if g.get(n) == grp and np.isfinite(v)]
        return float(np.mean([v > 0 for v in vals])) if vals else float("nan")

    nonax = [v for n, v in zmean_by_name.items() if g.get(n) == "non_axial" and np.isfinite(v)]
    return dict(n_matched=len(zmean_by_name), align_maxab=window_maxab(ctx, zmean_by_name),
                grad_angle=ang, grad_mag=mag, source_core_mean_z=sc, axis_end_noncore_mean_z=en,
                axial_mid_mean_z=am, non_axial_mean_z=na,
                axialmid_minus_nonaxial=(am - na) if np.isfinite(am) and np.isfinite(na) else float("nan"),
                source_core_minus_all=(sc - allmean) if np.isfinite(sc) and np.isfinite(allmean) else float("nan"),
                pms_source_core=pms["source_core"], pms_axis_end_noncore=pms["axis_end_noncore"],
                pms_axial_mid=pms["axial_mid"], pms_non_axial=pms["non_axial"],
                source_core_pos_share=share("source_core"), axial_mid_pos_share=share("axial_mid"),
                non_axial_pos_share=share("non_axial"),
                non_axial_p95_z=float(np.nanpercentile(nonax, 95)) if nonax else float("nan"),
                sync_median_corr=fd.field_synchrony(ztr_by_name),
                participation=fd.participation(zmean_by_name))


def _parity_fail(ds_sid, idx, long_npz):
    """长 cache bb_auc[idx] vs v2_windows bb_auc[idx]，max|Δ|>tol → True。参考缺失 → False(警告)。"""
    ref = V2REF / f"{ds_sid}.npz"
    if not ref.exists():
        return False, "no_ref"
    r = np.load(ref, allow_pickle=True)
    k = f"bb_auc__{idx}"
    if k not in r.files or k not in long_npz.files:
        return False, "no_key"
    diff = fd.parity_max_abs_diff(long_npz[k], r[k])
    return (diff > PARITY_TOL), f"{diff:.2e}"


def run_subject(ds_sid, substrate):
    ctx = load_context(ds_sid, substrate)
    meta = json.load(open(CACHE / f"{ds_sid}.json"))
    data = np.load(CACHE / f"{ds_sid}.npz", allow_pickle=True)
    cache_names = [str(x) for x in data["channels"]]
    rows, n_long, n_parity_fail = [], 0, 0
    gcounts = {gname: sum(v == gname for v in ctx["part"]["groups"].values()) for gname in fd.GROUPS}
    base = dict(ds_sid=ds_sid, subject=ds_sid.split("_", 1)[1],
                axis_degenerate=ctx["part"]["axis_degenerate"], swap_class=ctx["swap_class"],
                decision_k=ctx["decision_k"], source_focus_uncertain_a=ctx["uncert_a"],
                source_focus_uncertain_b=ctx["uncert_b"], n_src_a_full=len(ctx["src_a_full"]),
                n_src_b_full=len(ctx["src_b_full"]), n_source_core=gcounts["source_core"],
                n_axis_end_noncore=gcounts["axis_end_noncore"], n_axial_mid=gcounts["axial_mid"],
                n_non_axial=gcounts["non_axial"])
    for idx in meta["eligible_idxs"]:
        s = meta["seizure"][str(idx)]
        off, dur = s["eeg_offset_rel"], s["eeg_duration_sec"]
        pf, _info = _parity_fail(ds_sid, idx, data)
        if pf:
            n_parity_fail += 1
            continue
        if dur >= 40:
            n_long += 1
        for band in BANDS:
            zt, relt = data[f"{band}_zt__{idx}"], data[f"{band}_relt__{idx}"]
            lo = 0.0
            while lo + ONSET_WIN <= max(off + 1e-6, ONSET_WIN):
                zmv, sub = _slice(zt, relt, lo, lo + ONSET_WIN)
                tc = lo + ONSET_WIN / 2
                if zmv is not None:
                    zmn = _zmean_by_name(zmv, cache_names, ctx["mapped"])
                    if len(zmn) >= 6:
                        r = dict(base, seizure_idx=idx, seizure_id=s["seizure_id"],
                                 window_kind="onset_slide", band=band, t_center_rel_onset=tc,
                                 t_center_rel_offset=tc - off,
                                 progress_frac=(tc / off) if off > 0 else float("nan"),
                                 pre_onset_overlap=False, post_offset_overlap=bool((lo + ONSET_WIN) > off),
                                 ictal_fraction=_ictal_fraction(relt, lo, lo + ONSET_WIN, off),
                                 parity_fail=False)
                        r.update(_metrics_row(ctx, zmn, _ztraces_by_name(sub, cache_names, ctx["mapped"])))
                        rows.append(r)
                lo += ONSET_STEP
            for wlo, whi in OFFSET_WINS:
                zmv, sub = _slice(zt, relt, off + wlo, off + whi)
                if zmv is None:
                    continue
                zmn = _zmean_by_name(zmv, cache_names, ctx["mapped"])
                if len(zmn) < 6:
                    continue
                r = dict(base, seizure_idx=idx, seizure_id=s["seizure_id"],
                         window_kind="offset_aligned", band=band,
                         t_center_rel_onset=off + (wlo + whi) / 2, t_center_rel_offset=(wlo + whi) / 2,
                         progress_frac=float("nan"),
                         pre_onset_overlap=fd.offset_pre_onset_overlap(wlo, off),
                         post_offset_overlap=bool(whi > 0),
                         ictal_fraction=_ictal_fraction(relt, off + wlo, off + whi, off), parity_fail=False)
                r.update(_metrics_row(ctx, zmn, _ztraces_by_name(sub, cache_names, ctx["mapped"])))
                rows.append(r)
    # drift per (seizure, band)：各窗 grad_angle - 该 (sz,band) onset 首窗 grad_angle
    by = {}
    for r in rows:
        by.setdefault((r["seizure_idx"], r["band"]), []).append(r)
    for key, rs in by.items():
        onset_rs = sorted([x for x in rs if x["window_kind"] == "onset_slide"],
                          key=lambda x: x["t_center_rel_onset"])
        ref_ang = onset_rs[0]["grad_angle"] if onset_rs else float("nan")
        for r in rs:
            r["drift_vs_onset"] = fd.fold_angle_deg(r["grad_angle"], ref_ang)
            r["angle_to_interictal_axis"] = fd.fold_angle_deg(r["grad_angle"], 0.0)
    subj = dict(ds_sid=ds_sid, n_eligible=len(meta["eligible_idxs"]),
                n_used=len({k[0] for k in by}), n_parity_fail=n_parity_fail, n_long_seizures=n_long,
                axis=dict(L=ctx["part"]["L"], bbox_diag=ctx["part"]["bbox_diag"],
                          axis_degenerate=ctx["part"]["axis_degenerate"],
                          swap_class=ctx["swap_class"], decision_k=ctx["decision_k"],
                          core_a=ctx["core_a"], core_b=ctx["core_b"],
                          source_focus_uncertain_a=ctx["uncert_a"], source_focus_uncertain_b=ctx["uncert_b"],
                          source_top2_dist_a_mm=ctx["dist_a"], source_top2_dist_b_mm=ctx["dist_b"],
                          src_a_full=ctx["src_a_full"], src_b_full=ctx["src_b_full"],
                          n_axial_mid=sum(v == "axial_mid" for v in ctx["part"]["groups"].values())))
    return rows, subj


CSV_COLS = ["ds_sid", "subject", "seizure_idx", "seizure_id", "window_kind", "t_center_rel_onset",
            "t_center_rel_offset", "progress_frac", "pre_onset_overlap", "post_offset_overlap",
            "ictal_fraction", "parity_fail", "band", "swap_class", "decision_k",
            "source_focus_uncertain_a", "source_focus_uncertain_b", "n_src_a_full", "n_src_b_full",
            "n_matched", "n_source_core", "n_axis_end_noncore", "n_axial_mid", "n_non_axial",
            "axis_degenerate", "align_maxab", "drift_vs_onset", "angle_to_interictal_axis",
            "grad_mag", "source_core_mean_z", "axis_end_noncore_mean_z", "axial_mid_mean_z",
            "non_axial_mean_z", "axialmid_minus_nonaxial", "source_core_minus_all", "pms_source_core",
            "pms_axis_end_noncore", "pms_axial_mid", "pms_non_axial", "source_core_pos_share",
            "axial_mid_pos_share", "non_axial_pos_share", "non_axial_p95_z", "sync_median_corr",
            "participation"]


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--substrate", choices=list(OUT_BY_SUB), default="broad")
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()
    OUT = OUT_BY_SUB[args.substrate]
    subjects = args.subjects or SUBJECTS_BY_SUB[args.substrate]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_subject").mkdir(exist_ok=True)
    all_rows = []
    for ds_sid in subjects:
        if not (CACHE / f"{ds_sid}.npz").exists():
            print(f"[skip] {ds_sid} no long cache", flush=True); continue
        rows, subj = run_subject(ds_sid, args.substrate)
        all_rows += rows
        json.dump(subj, open(OUT / "per_subject" / f"{ds_sid}.json", "w"), indent=2, ensure_ascii=False)
        print(f"[{ds_sid}] {len(rows)} window-rows, {subj['n_used']} sz, parity_fail={subj['n_parity_fail']}, "
              f"degen={subj['axis']['axis_degenerate']}, swap={subj['axis']['swap_class']}, "
              f"n_axial_mid={subj['axis']['n_axial_mid']}", flush=True)
    with open(OUT / "per_seizure_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader(); w.writerows(all_rows)
    print(f"[done] {len(all_rows)} rows -> {OUT/'per_seizure_metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
