"""Topic 5 发作内 field 动力学 — 趋势统计：'轴向中段占比随进程下降 / 非轴向上升' 的量化。

设计纪律: 发作=重复单位（窗内强自相关，不能当独立观测）。两层:
  per-seizure: Spearman ρ(progress, pms_axial_mid) 期望<0; ρ(progress, pms_non_axial) 期望>0; + OLS 斜率。
  per-subject: 对该被试各发作的 ρ 做 Wilcoxon signed-rank（单边, H0 中位=0）, 报 n / 中位 ρ / 同向比例 / p。
仅用 band=bb、window_kind=onset_slide、ictal_fraction>=0.5、走廊可测(pms_axial_mid 非 NaN)的窗。
PILOT/exploratory: 不做正式 cohort 检验, 只描述几个被试方向一致/显著。z-ER 中后期偏示意; pms 是占比相对稳。
"""
from __future__ import annotations
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, wilcoxon, linregress

_ROOT = Path(__file__).resolve().parents[1]
OUT_BY_SUB = {"broad": _ROOT / "results/topic5_ictal_recruitment/field_dynamics",
              "narrow": _ROOT / "results/topic5_ictal_recruitment/field_dynamics_narrow"}
OUT = OUT_BY_SUB["broad"]   # rebound per --substrate in main()
MIN_WIN = 4          # 每次发作至少这么多可用窗才算 ρ（短发作 <4 窗 → 不进趋势）
MIN_SZ_WILCOXON = 6  # 每被试至少这么多发作才报 Wilcoxon p


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _per_seizure_rho(rows, key):
    """{seizure_idx: (rho, slope, n_win)} for onset bb ictal>=0.5 windows with finite key."""
    by = defaultdict(list)
    for r in rows:
        if (r["window_kind"] == "onset_slide" and r["band"] == "bb"
                and _f(r["ictal_fraction"]) >= 0.5):
            p, v = _f(r["progress_frac"]), _f(r[key])
            if np.isfinite(p) and np.isfinite(v):
                by[r["seizure_idx"]].append((p, v))
    out = {}
    for sz, pts in by.items():
        if len(pts) < MIN_WIN:
            continue
        p = np.array([a for a, _ in pts]); v = np.array([b for _, b in pts])
        if np.std(v) == 0:
            continue
        rho = spearmanr(p, v).correlation
        slope = linregress(p, v).slope
        out[sz] = (float(rho), float(slope), len(pts))
    return out


def _subject_stat(rhos, side):
    """rhos: list of per-seizure ρ. side='less'(expect<0) or 'greater'(expect>0).
    Returns dict with n, median_rho, frac_dir, wilcoxon_p (or None)."""
    arr = np.array(rhos, float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return dict(n=0, median_rho=None, frac_dir=None, wilcoxon_p=None)
    frac = float(np.mean(arr < 0 if side == "less" else arr > 0))
    p = None
    if len(arr) >= MIN_SZ_WILCOXON and np.any(arr != 0):
        try:
            p = float(wilcoxon(arr, alternative=side).pvalue)
        except ValueError:
            p = None
    return dict(n=int(len(arr)), median_rho=float(np.median(arr)), frac_dir=frac, wilcoxon_p=p)


def main():
    global OUT
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--substrate", choices=list(OUT_BY_SUB), default="broad")
    OUT = OUT_BY_SUB[ap.parse_args().substrate]
    rows_all = list(csv.DictReader(open(OUT / "per_seizure_metrics.csv")))
    subs = sorted(set(r["ds_sid"] for r in rows_all))
    report = {}
    print(f"{'subject':16s} {'corridor':9s} | axial_mid (expect DECLINE, rho<0)        | non_axial (expect RISE, rho>0)")
    print(f"{'':16s} {'n_sz':9s} | n  med_rho  frac<0  wilcoxon_p             | n  med_rho  frac>0  wilcoxon_p")
    for ds in subs:
        rows = [r for r in rows_all if r["ds_sid"] == ds]
        ax = _per_seizure_rho(rows, "pms_axial_mid")
        na = _per_seizure_rho(rows, "pms_non_axial")
        s_ax = _subject_stat([v[0] for v in ax.values()], "less")
        s_na = _subject_stat([v[0] for v in na.values()], "greater")
        report[ds] = dict(axial_mid=s_ax, non_axial=s_na,
                          per_seizure_axial={k: v[0] for k, v in ax.items()},
                          per_seizure_nonaxial={k: v[0] for k, v in na.items()})

        def fmt(s):
            mr = f"{s['median_rho']:+.2f}" if s['median_rho'] is not None else "  -- "
            fd = f"{s['frac_dir']:.2f}" if s['frac_dir'] is not None else " -- "
            wp = f"p={s['wilcoxon_p']:.3f}" if s['wilcoxon_p'] is not None else f"(n<{MIN_SZ_WILCOXON})"
            return f"{s['n']:>2d}  {mr}    {fd}    {wp:18s}"
        cor = "n=0(NA)" if s_ax["n"] == 0 else f"{s_ax['n']}sz"
        print(f"{ds:16s} {cor:9s} | {fmt(s_ax)} | {fmt(s_na)}")
    # cohort descriptive (pilot — count only)
    meas = [ds for ds in subs if report[ds]["axial_mid"]["n"] > 0]
    ax_dir = sum(report[ds]["axial_mid"]["median_rho"] < 0 for ds in meas)
    na_dir = sum(report[ds]["non_axial"]["median_rho"] > 0 for ds in meas)
    ax_sig = sum((report[ds]["axial_mid"]["wilcoxon_p"] or 1) < 0.05 for ds in meas)
    na_sig = sum((report[ds]["non_axial"]["wilcoxon_p"] or 1) < 0.05 for ds in meas)
    cohort = dict(n_corridor_subjects=len(meas),
                  axial_median_rho_negative=f"{ax_dir}/{len(meas)}",
                  nonaxial_median_rho_positive=f"{na_dir}/{len(meas)}",
                  axial_wilcoxon_sig=f"{ax_sig}/{len(meas)}", nonaxial_wilcoxon_sig=f"{na_sig}/{len(meas)}")
    print(f"\nCOHORT (pilot, descriptive): corridor-measurable={len(meas)}; "
          f"axial median_rho<0 in {ax_dir}/{len(meas)}, non_axial median_rho>0 in {na_dir}/{len(meas)}; "
          f"per-subject Wilcoxon p<0.05: axial {ax_sig}/{len(meas)}, non_axial {na_sig}/{len(meas)}")
    json.dump(dict(per_subject=report, cohort=cohort, params=dict(MIN_WIN=MIN_WIN, MIN_SZ_WILCOXON=MIN_SZ_WILCOXON)),
              open(OUT / "trend_stats.json", "w"), indent=2, ensure_ascii=False)
    print(f"[done] -> {OUT/'trend_stats.json'}")


if __name__ == "__main__":
    main()
