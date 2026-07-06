"""Topic5 间期顺序场 → 发作早期能量空间外推 — Phase 2 cohort 正式版。

主张量 = F_core_only（场只用 narrow core 建，隐身电极不进场）；A/B 两模板 max。
per-subject：F_core_only + {channel,within_shaft,anchor} permutation null；
逐发作序列 F_core_sz/F_loo_sz/C1_sz/C2_sz/radius_sz/pca1_geometry_sz → per-subject paired Wilcoxon。
基线 pca1_geometry = 沿隐身电极坐标主 linear 轴投影（**非** shaft-direction）。
cohort：subject=独立单位（不池化 seizure）；14-hypothesis BH-FDR（{bb,hfa}×7）。
Spec: docs/superpowers/specs/2026-07-01-topic5-energy-field-extrapolation-design.md
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon, binomtest

from src.topic5_field_extrapolation import (
    load_broad_axis_record, channel_names_from_pool, broad_minus_narrow,
    ictal_paired_features, predicted_interictal_order, maxabscorr_series,
    field_null_p, quantile_bin_labels, _abs_spear,
    DEF_BROAD_POOL, DEF_NARROW_POOL,
)

OUT = Path("results/topic5_ictal_recruitment/field_extrapolation")
ALPHA = 0.05
DELTA_FC = 0.03
SUBJECTS = ["epilepsiae_583", "epilepsiae_922", "yuquan_zhangkexuan", "epilepsiae_1077",
            "epilepsiae_1096", "epilepsiae_1125", "epilepsiae_1150", "epilepsiae_139",
            "epilepsiae_253", "epilepsiae_384", "epilepsiae_620", "epilepsiae_635",
            "epilepsiae_916", "epilepsiae_590", "epilepsiae_1084", "epilepsiae_1146"]
BANDS = ["bb_auc", "hfa_auc"]


def _median(series):
    s = [v for v in series if np.isfinite(v)]
    return float(np.median(s)) if s else float("nan")


def _paired(a_series, b_series):
    """F_core_sz vs Y_sz：共同有限发作上 median diff + 单尾 Wilcoxon(F>Y) p。"""
    a = np.asarray(a_series, float)
    b = np.asarray(b_series, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return {"median_diff": float("nan"), "p": float("nan"), "n": int(m.sum())}
    d = a[m] - b[m]
    md = float(np.median(d))
    if np.allclose(d, 0):
        return {"median_diff": md, "p": 1.0, "n": int(m.sum())}
    try:
        p = float(wilcoxon(a[m], b[m], alternative="greater").pvalue)
    except ValueError:
        p = float("nan")
    return {"median_diff": md, "p": p, "n": int(m.sum())}


def _pca1_proj(coords):
    """coords (n,3) → 投影到第一主轴 + 非退化(2nd/1st eig 比)。"""
    X = np.asarray(coords, float)
    fin = np.all(np.isfinite(X), axis=1)
    if fin.sum() < 3:
        return None, 0.0
    Xc = X[fin] - X[fin].mean(0)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    ev = s ** 2
    ratio = float(ev[1] / ev[0]) if ev[0] > 0 and ev.size > 1 else 0.0
    proj = np.full(X.shape[0], np.nan)
    proj[fin] = (X[fin] - X[fin].mean(0)) @ vt[0]
    return proj, ratio


def evaluate_subject(ds_sid, activation, *, n_null=2000, sigma_xy=None, seed=0):
    rec_a = load_broad_axis_record(ds_sid, template="t_a")
    if rec_a is None:
        return {"subject": ds_sid, "band": activation, "status": "no_broad_geometry"}
    rec_b = load_broad_axis_record(ds_sid, template="t_b")
    recs = [rec_a] + ([rec_b] if rec_b is not None else [])
    narrow = set(channel_names_from_pool(ds_sid, DEF_NARROW_POOL))
    hidden = broad_minus_narrow(channel_names_from_pool(ds_sid, DEF_BROAD_POOL), list(narrow))
    cache, paired = ictal_paired_features(ds_sid, "bact", activation)   # 对齐 (bact, energy)
    if not paired:
        return {"subject": ds_sid, "band": activation, "status": "no_seizure_cache"}
    bact_sz = [p for p, _ in paired]
    sz = [t for _, t in paired]
    cset = set(cache)

    def _valid(rec):
        bn = {c["name"]: c for c in rec["channels"]}
        return {n for n in hidden if n in bn and np.isfinite(bn[n].get("typical_rank", np.nan))}
    names = [n for n in hidden if n in cset and all(n in _valid(r) for r in recs)]
    pred_core = [predicted_interictal_order(r, names, loo=True, sigma_xy=sigma_xy, core_names=narrow)
                 for r in recs]
    pred_loo = [predicted_interictal_order(r, names, loo=True, sigma_xy=sigma_xy) for r in recs]
    names = [n for n in names if all(np.isfinite(d.get(n, np.nan)) for d in pred_core + pred_loo)]
    if len(names) < 3:
        return {"subject": ds_sid, "band": activation, "status": "insufficient_hidden",
                "n_hidden": len(names)}
    cidx = {n: i for i, n in enumerate(cache)}
    ci = np.array([cidx[n] for n in names], int)
    byname = [{c["name"]: c for c in r["channels"]} for r in recs]

    xcore = [np.array([d[n] for n in names], float) for d in pred_core]
    xloo = [np.array([d[n] for n in names], float) for d in pred_loo]
    xown = [np.array([bn[n]["typical_rank"] for n in names], float) for bn in byname]
    src = min((c["along_axis_mm"] for c in rec_a["channels"]
               if np.isfinite(c.get("along_axis_mm", np.nan))), default=0.0)
    xrad = [np.array([byname[0][n]["along_axis_mm"] - src for n in names], float)]
    coords = np.array([byname[0][n].get("coord_mm", [np.nan] * 3) for n in names], float)
    # pca1_geometry 基线 = 沿隐身电极坐标的主 linear 轴投影 (dominant 采样方向)。
    # 注意(review): 这**不是**"单杆方向 / shaft-direction baseline"; 只在 >=2D(pca_ratio>=.05)算。
    proj, pca_ratio = _pca1_proj(coords)
    xpca1 = [proj] if (proj is not None and pca_ratio >= 0.05) else None

    s_core = maxabscorr_series(xcore, ci, sz)
    s_loo = maxabscorr_series(xloo, ci, sz)
    s_c1 = maxabscorr_series(xown, ci, sz)
    s_rad = maxabscorr_series(xrad, ci, sz)
    s_pca1 = maxabscorr_series(xpca1, ci, sz) if xpca1 else [np.nan] * len(sz)
    s_c2 = [(_abs_spear(bact_sz[k][ci][np.isfinite(sz[k][ci])], sz[k][ci][np.isfinite(sz[k][ci])])
             if np.isfinite(sz[k][ci]).sum() >= 3 else np.nan) for k in range(len(sz))]

    F_core = _median(s_core)
    # within_shaft null: 真·按电极所在杆分组 permute(合法用 shaft 成员; 与上面 pca1 基线无关)。
    # **稳定整数编码**(非 hash()): Python 字符串 hash 每进程随机 → 会改 _permute_within_labels 的组处理顺序
    # → 改 RNG 抽样序 → 结果不可复现(实测 hfa within_shaft n_pass 3↔4 漂)。sorted-unique 映射保证确定性。
    _shaft_names = [byname[0][n].get("shaft", "?") for n in names]
    _uniq = {s: i for i, s in enumerate(sorted(set(_shaft_names)))}
    shaft_ids = np.array([_uniq[s] for s in _shaft_names], int)
    labels = {
        "channel": [np.zeros(len(names), int)] * len(sz),
        "within_shaft": [shaft_ids] * len(sz),
        "anchor": [quantile_bin_labels(bact_sz[k][ci], n_bins=4) for k in range(len(sz))],
    }
    nulls = {k: field_null_p(xcore, ci, sz, F_core, lab, n=n_null, seed=seed)
             for k, lab in labels.items()}

    pair = {"C1": _paired(s_core, s_c1), "C2": _paired(s_core, s_c2),
            "radius": _paired(s_core, s_rad), "pca1_geometry": _paired(s_core, s_pca1)}
    _gt = lambda a, b: (a == a and b == b and a - b > DELTA_FC)
    # 弱 screen(仅 channel null + C1/C2 margin) — 名字如实, 不是完整场优势
    screen_c1c2 = (nulls["channel"]["p_value"] < ALPHA
                   and _gt(F_core, _median(s_c1)) and _gt(F_core, _median(s_c2)))
    # 严格 screen = 三层 null 全过 + 赢 C1/C2/radius/pca1 全部 margin(才配叫 field-advantage)
    screen_strict = (all(nulls[k]["p_value"] == nulls[k]["p_value"] and nulls[k]["p_value"] < ALPHA
                         for k in ("channel", "within_shaft", "anchor"))
                     and all(_gt(F_core, _median(s)) for s in (s_c1, s_c2, s_rad, s_pca1)))
    return {"subject": ds_sid, "band": activation, "status": "ok",
            "n_hidden": len(names), "n_seizures": len([v for v in s_core if np.isfinite(v)]),
            "n_templates": len(recs), "pca_ratio": pca_ratio, "pca1_usable": xpca1 is not None,
            "F_core_only": F_core, "F_loo": _median(s_loo),
            "C1": _median(s_c1), "C2": _median(s_c2), "radius": _median(s_rad),
            "pca1_geometry": _median(s_pca1),
            "null_channel_p": nulls["channel"]["p_value"], "null_within_shaft_p": nulls["within_shaft"]["p_value"],
            "null_anchor_p": nulls["anchor"]["p_value"],
            "paired": pair,
            "screen_channel_c1c2_only": bool(screen_c1c2), "screen_strict": bool(screen_strict),
            "series": {"F_core": s_core, "F_loo": s_loo, "C1": s_c1, "C2": s_c2,
                       "radius": s_rad, "pca1_geometry": s_pca1}}


def _bh_fdr(pvals):
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    idx = np.where(ok)[0]
    m = idx.size
    if m == 0:
        return q
    order = idx[np.argsort(p[idx])]
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        r = m - rank + 1
        prev = min(prev, p[i] * m / r)
        q[i] = prev
    return q


def aggregate(per_subject_rows):
    """subject = 独立单位。每 band×hypothesis 一个 cohort p；BH-FDR over 全部。"""
    HYP_NULL = [("channel", "null_channel_p"), ("within_shaft", "null_within_shaft_p"),
                ("anchor", "null_anchor_p")]
    HYP_PAIR = ["C1", "C2", "radius", "pca1_geometry"]
    rows = []
    for band in BANDS:
        ok = [r for r in per_subject_rows if r.get("band") == band and r.get("status") == "ok"]
        n = len(ok)
        for name, key in HYP_NULL:
            npass = sum(1 for r in ok if np.isfinite(r[key]) and r[key] < ALPHA)
            nval = sum(1 for r in ok if np.isfinite(r[key]))
            p = binomtest(npass, nval, ALPHA, alternative="greater").pvalue if nval else float("nan")
            rows.append({"band": band, "hypothesis": f"F_core>{name}_null", "kind": "binomial",
                         "n_subjects": nval, "n_pass": npass, "cohort_p": float(p)})
        for y in HYP_PAIR:
            diffs = [r["paired"][y]["median_diff"] for r in ok
                     if np.isfinite(r["paired"][y]["median_diff"])]
            n_pos = sum(1 for d in diffs if d > 0)
            if len(diffs) >= 3 and not np.allclose(diffs, 0):
                try:
                    p = float(wilcoxon(diffs, alternative="greater").pvalue)
                except ValueError:
                    p = float("nan")
            else:
                p = float("nan")
            rows.append({"band": band, "hypothesis": f"F_core>{y}", "kind": "wilcoxon_subject",
                         "n_subjects": len(diffs), "n_pos_diff": n_pos,
                         "median_of_subject_diffs": float(np.median(diffs)) if diffs else float("nan"),
                         "cohort_p": p})
    qs = _bh_fdr([r["cohort_p"] for r in rows])
    for r, q in zip(rows, qs):
        r["fdr_q"] = float(q) if np.isfinite(q) else None
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=SUBJECTS)
    ap.add_argument("--bands", nargs="*", default=BANDS)
    ap.add_argument("--n-null", type=int, default=2000)
    ap.add_argument("--sigma-xy", type=float, default=None)
    ap.add_argument("--compute", action="store_true", help="跑 per-subject（否则只 aggregate 现有）")
    args = ap.parse_args()
    pdir = OUT / "cohort_per_subject"
    pdir.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.compute:
        for band in args.bands:
            for sid in args.subjects:
                r = evaluate_subject(sid, band, n_null=args.n_null, sigma_xy=args.sigma_xy)
                json.dump(r, open(pdir / f"{sid}__{band}.json", "w"), indent=2)
                rows.append(r)
                st = r.get("status")
                if st != "ok":
                    print(f"{sid:20} {band:8} {st}")
                    continue
                g = lambda v: f"{v:.3f}" if isinstance(v, float) and v == v else "nan"
                print(f"{sid:20} {band:8} hid={r['n_hidden']:>2} sz={r['n_seizures']:>2} T={r['n_templates']} "
                      f"| Fco={g(r['F_core_only'])} C1={g(r['C1'])} C2={g(r['C2'])} rad={g(r['radius'])} "
                      f"pca1={g(r['pca1_geometry'])} | ch_p={g(r['null_channel_p'])} ws_p={g(r['null_within_shaft_p'])} "
                      f"an_p={g(r['null_anchor_p'])} | c1c2scr={'Y' if r['screen_channel_c1c2_only'] else '-'} "
                      f"strict={'Y' if r['screen_strict'] else '-'}")
    else:
        for f in sorted(pdir.glob("*.json")):
            rows.append(json.load(open(f)))
    agg = aggregate(rows)
    json.dump(agg, open(OUT / "energy_field_extrapolation_FINAL.json", "w"), indent=2)
    n_ok = len(set(r["subject"] for r in rows if r.get("status") == "ok"))
    n_skip = len(set(r["subject"] for r in rows if r.get("status") == "no_broad_geometry"))
    lines = [f"# Energy-field extrapolation cohort FINAL (subject=unit; primary=F_core_only)",
             f"executed 2026-07-01 via `scripts/run_topic5_energy_field_cohort.py --compute --n-null 2000`.",
             f"**COHORT = {n_ok}-subject broad-geometry cohort** ({n_skip} subjects skipped: no broad geometry). "
             f"E590/E1084/E1146 added 2026-07-01 (upstream broad propagation via build_broad_lagpat_patch_epilepsiae.py; "
             f"E590/E1146 have only n_hidden=4 = low-power).",
             f"pca1_geometry baseline = dominant linear axis of hidden coords (NOT shaft-direction).",
             f"alpha={ALPHA} delta_fc={DELTA_FC}; BH-FDR over {len(agg)} hypotheses\n",
             "| band | hypothesis | kind | n_subj | detail | cohort_p | fdr_q |",
             "|---|---|---|---|---|---|---|"]
    for r in agg:
        det = (f"n_pass={r.get('n_pass')}" if r["kind"] == "binomial"
               else f"n_pos={r.get('n_pos_diff')} med_diff={r.get('median_of_subject_diffs')}")
        q = r["fdr_q"]
        lines.append(f"| {r['band']} | {r['hypothesis']} | {r['kind']} | {r['n_subjects']} | {det} | "
                     f"{r['cohort_p']:.4f} | {q if q is None else round(q,4)} |")
    (OUT / "energy_field_extrapolation_FINAL.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
