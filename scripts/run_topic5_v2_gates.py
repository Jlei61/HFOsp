"""Topic 5 V2 Phase 1 — Task 14 (script part): Gate A/B/C decision table + max-over-bands FWER null.

测了什么（白话）：前面的 null 脚本已经给出每个频带"发作刚起头的能量空间图 vs 间期'谁先谁后'顺序几何
图（G_HFO）"的对齐，以及这份对齐超没超过两种巧合（把发作能量图在同一根电极杆内洗牌、把间期顺序按
'谁 HFO 多'重排）。这个脚本把这些证据合成一张判决表：每个频带最深走到哪一档——
  · Gate A（空间+顺序都过）：既不是单纯空间平滑造成的像，也不是单纯跟着 HFO 富集触点走。
  · Gate B（频带特异）：扣掉"所有频带共有的那张宽谱招募场"之后还对得上，且它不是"扫了一堆频带里
    碰运气挑出来最好看的那个"（家族式 max-over-bands 控制）。
  · Gate C（HFO/ripple 特异）：是 ripple 频带，且扣掉 1/f 背景后的振荡余量仍对得上。

怎么测的（把 null 脚本每被试每频带的证据折成队列级判决）：
  · cohort_delta / cohort_null_z / cohort_empirical_p = 对被试取中位（把"每个被试实测 − 它自己零假设
    中位"的余量在被试间取中位）；空间零假设是主统计量。
  · max_over_bands_p（家族式多重比较控制，从 perm-long 长表算）：先把每个频带的余量按"减掉该频带
    零假设中位"对齐（否则低频天然更平滑、余量基线更高，会污染"取最大"）；再看洗牌分布里"所有频带
    中最大的那个余量"当作"碰运气能达到的天花板"，实测某频带的余量有没有超过这个天花板分布。赢家
    频带拿到的正好是 brief 写的 P(perm 最大余量 ≥ 实测最大余量)，其余更弱频带 p 更大。
  · 判决用纯函数 gate_pass_flags/gate_tier（src.topic5_v2_band_scan）：Gate A 硬要求空间洗牌"够强"
    （within_shaft_strong 才算正式；退化到跨杆/全被试洗只当描述）、顺序零假设可信（非 weak_downgrade，
    且 missing 会被喂 NaN p 挡住）。

揭示了什么：不是单个 PASS/NULL，而是每频带一个档位——strongest（走到 Gate C）/ frequency_specific
（到 Gate B）/ broadband_recruitment（只到 Gate A，G_HFO 预测的是宽谱招募，也是有效阳性）/ weak_negative
（连 Gate A 都没过）。对小样本单被试（如 epilepsiae_139），空间洗牌退化成 subject_wide_weak、顺序退化成
weak_downgrade，因此某些频带即便 p<alpha、余量为正也进不了正式 Gate A，如实标 weak_negative（不硬凑过）。

代号补注：G_HFO=间期 typical_rank；spatial null=within_shaft 洗牌（issue #10）；order null=HFO 率保留、
timing 打乱（Patch F/Task 9）；common_resid=扣掉 LOBO 共有场的残差（Task 10b, Gate B）；aperiodic_resid=
扣掉 1/f 地板的振荡余量（Task 11b, Gate C）；max-over-bands=Westfall-Young max-T（Patch E perm-long）；
gate_pass_flags/gate_tier=Task 14 纯函数（issue #17）；subject=统计单元。

Reads Task-13 outputs (feature-INFIXED names; Task 13 concern #2 bare names clobber across features):
  {outdir}/{substrate}/phase1_null_{feature}_perm_subject_long.parquet   (spatial max-over-bands + cohort)
  {outdir}/{substrate}/phase1_null_{feature}_subject_summary.csv          (per-subject delta/z/p + strengths)
  {outdir}/{substrate}/phase1_null_common_resid_subject_summary.csv       (Gate B input; if present)
  {outdir}/{substrate}/phase1_null_aperiodic_resid_subject_summary.csv    (Gate C input; if present)
Writes: {outdir}/{substrate}/phase1_gate_summary.csv (+ phase1_gate_meta.json).

设计: docs/superpowers/specs/2026-07-01-topic5-v2-hfo-critical-mode-design.md; plan Task 14 (issue #17, Patch E/H).
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.topic5_v2_band_scan import load_phase1_config, gate_pass_flags, gate_tier  # noqa: E402

V2_ROOT = _ROOT / "results/topic5_ictal_recruitment/v2_band_scan"

# Weakest tier wins when folding subjects -> cohort (a FORMAL within_shaft_strong / order 'strong'
# cohort requires EVERY contributing subject at that tier; one fallback subject downgrades it).
_SPATIAL_TIER = {"within_shaft_strong": 3, "distance_bin_fallback": 2, "subject_wide_weak": 1}
_ORDER_TIER = {"strong": 3, "weak_downgrade": 2, "missing": 1}

# The primary Gate A/B statistic is the within-shaft spatial null; Gate B/C residual features are
# also tested against their own spatial null (does the residual field's alignment beat spatial smoothness).
PRIMARY_NULL_TYPE = "spatial"

GATE_COLS = ["axis_set", "cohort", "band", "feature", "gate_A_spatial_pass", "gate_A_order_pass",
             "gate_B_frequency_specific_pass", "gate_C_HFO_specific_pass", "cohort_delta",
             "cohort_null_z", "cohort_empirical_p", "max_over_bands_p", "n_subjects_valid",
             "interpretation_tier", "spatial_null_strength", "order_null_strength"]


def _f(x):
    """Native python float (numpy/pandas scalar or None -> float/NaN).

    Forward fix #1 (Task-14-module review): gate_pass_flags `and`-chains comparison results, so
    every numeric input must be a python float — a numpy scalar makes `spatial_p < alpha` a
    numpy.bool_, and the chained result then breaks `is True`/`is False` semantics downstream."""
    if x is None:
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# --------------------------------------------------------------------------- cohort folds (from summary)
def _cohort_from_summary(summ_df, feature, null_type):
    """Per-band cohort delta/null_z/empirical_p = median OVER SUBJECTS of the per-subject summary
    columns (delta = obs − its own null_median), for one feature + null_type. n_subjects_valid =
    subjects with a finite delta for that band. Returns band -> {delta, null_z, empirical_p, n}."""
    sub = summ_df[(summ_df["feature"] == feature) & (summ_df["null_type"] == null_type)]
    out = {}
    for band, g in sub.groupby("band"):
        d = pd.to_numeric(g["delta"], errors="coerce").to_numpy(float)
        z = pd.to_numeric(g["null_z"], errors="coerce").to_numpy(float)
        p = pd.to_numeric(g["empirical_p"], errors="coerce").to_numpy(float)
        out[str(band)] = {"delta": float(np.nanmedian(d)) if np.isfinite(d).any() else float("nan"),
                          "null_z": float(np.nanmedian(z)) if np.isfinite(z).any() else float("nan"),
                          "empirical_p": float(np.nanmedian(p)) if np.isfinite(p).any() else float("nan"),
                          "n": int(np.isfinite(d).sum())}
    return out


def _cohort_strengths(summ_df, feature):
    """Per-band cohort (spatial_strength, order_strength) = the WEAKEST tier across contributing
    subjects. Read from the spatial-null rows (one per subject/band; each row carries BOTH strength
    columns). Weakest-wins is the honest formal tier (P1-c: only within_shaft_strong supports a
    formal Gate A; a single fallback subject cannot be laundered into a strong cohort)."""
    sub = summ_df[(summ_df["feature"] == feature) & (summ_df["null_type"] == "spatial")]
    out = {}
    for band, g in sub.groupby("band"):
        sp = [str(v) for v in g["spatial_null_strength"] if str(v) in _SPATIAL_TIER]
        orr = [str(v) for v in g["order_null_strength"] if str(v) in _ORDER_TIER]
        out[str(band)] = (min(sp, key=lambda s: _SPATIAL_TIER[s]) if sp else None,
                          min(orr, key=lambda s: _ORDER_TIER[s]) if orr else None)
    return out


# --------------------------------------------------------------------------- max-over-bands FWER null
def _max_over_bands_p(perm_df, feature, null_type, family_bands):
    """Family-wise max-over-bands p per band (Westfall-Young max-T on null-centered cohort deltas).

    per (band b, perm_id p): cohort_stat[b,p] = median over subjects of perm_subject_median.
    null_median[b] = median over p>=0 of cohort_stat[b,p]        (per-band centering constant).
    obs_cohort_delta[b]  = cohort_stat[b,-1]     − null_median[b].
    perm_cohort_delta[b,p] = cohort_stat[b,p]    − null_median[b]  (centering makes bands comparable
                                                                     before the max — else the max is
                                                                     dominated by the smoothest band).
    perm_max_delta[p]    = max over b∈family of perm_cohort_delta[b,p].
    max_over_bands_p[b]  = add-one P(perm_max_delta >= obs_cohort_delta[b]).

    The winning (observed-max) band gets exactly the brief's P(perm max-band delta >= observed
    max-band delta); every weaker band gets a larger p ("this band is not simply the best of many
    bands sampled by chance"). Returns band -> p for bands in `family_bands`."""
    sub = perm_df[(perm_df["feature"] == feature) & (perm_df["null_type"] == null_type)
                  & (perm_df["band"].isin(family_bands))].copy()
    sub["perm_subject_median"] = pd.to_numeric(sub["perm_subject_median"], errors="coerce")
    coh = (sub.groupby(["band", "perm_id"])["perm_subject_median"]
           .apply(lambda s: float(np.nanmedian(s.to_numpy(float))) if np.isfinite(s.to_numpy(float)).any()
                  else float("nan")).reset_index())
    piv = coh.pivot(index="perm_id", columns="band", values="perm_subject_median")
    null_piv = piv[piv.index >= 0]
    if -1 not in piv.index or null_piv.empty:
        return {b: float("nan") for b in family_bands}
    null_median = null_piv.median(axis=0)                          # per band (skipna)
    perm_delta = null_piv.subtract(null_median, axis=1)            # (n_perm, n_band), null-centered
    perm_max_delta = perm_delta.max(axis=1).to_numpy(float)        # per perm, max over bands (skipna)
    perm_max_delta = perm_max_delta[np.isfinite(perm_max_delta)]
    n_perm = int(perm_max_delta.size)
    obs_delta = (piv.loc[-1] - null_median)
    out = {}
    for b in piv.columns:
        od = float(obs_delta[b])
        if not np.isfinite(od) or n_perm == 0:
            out[str(b)] = float("nan")
        else:
            out[str(b)] = float((1 + int(np.sum(perm_max_delta >= od))) / (1 + n_perm))
    for b in family_bands:                                         # bands with no rows -> NaN
        out.setdefault(str(b), float("nan"))
    return out


# --------------------------------------------------------------------------- optional Gate B/C features
def _read_feature_cohort(sub_dir, feature, null_type):
    """Per-band cohort {delta, empirical_p} for a Gate B/C residual feature (common_resid /
    aperiodic_resid), if its feature-infixed summary exists; {} otherwise (-> NaN inputs -> gate fails
    honestly rather than fabricate a residual pass)."""
    p = sub_dir / f"phase1_null_{feature}_subject_summary.csv"
    if not p.exists():
        return {}
    return _cohort_from_summary(pd.read_csv(p), feature, null_type)


# --------------------------------------------------------------------------- driver
def build_gate_rows(sub_dir, substrate, feature, cfg):
    alpha = float(cfg["nulls"]["alpha"])
    repro_bands = set(cfg["repro_bands"].values())                # legacy reproduction bands are NOT hypotheses

    perm_path = sub_dir / f"phase1_null_{feature}_perm_subject_long.parquet"
    summ_path = sub_dir / f"phase1_null_{feature}_subject_summary.csv"
    if not perm_path.exists():
        raise FileNotFoundError(f"missing feature-infixed perm-long: {perm_path}")
    if not summ_path.exists():
        raise FileNotFoundError(f"missing feature-infixed subject summary: {summ_path}")
    perm_df = pd.read_parquet(perm_path)
    summ_df = pd.read_csv(summ_path)

    family_bands = sorted(set(str(b) for b in perm_df["band"].unique()) - repro_bands)

    coh_spatial = _cohort_from_summary(summ_df, feature, PRIMARY_NULL_TYPE)
    coh_order = _cohort_from_summary(summ_df, feature, "order")
    strengths = _cohort_strengths(summ_df, feature)
    mob = _max_over_bands_p(perm_df, feature, PRIMARY_NULL_TYPE, family_bands)
    coh_common = _read_feature_cohort(sub_dir, "common_resid", PRIMARY_NULL_TYPE)
    coh_aperi = _read_feature_cohort(sub_dir, "aperiodic_resid", PRIMARY_NULL_TYPE)

    rows = []
    for band in family_bands:
        sp = coh_spatial.get(band, {})
        orr = coh_order.get(band, {})
        sp_strength, or_strength = strengths.get(band, (None, None))
        band_mob = _f(mob.get(band))

        # Forward fix #2: a 'missing' (or absent) order null must NOT slip through gate_A_order via the
        # `strength != 'weak_downgrade'` clause — feed NaN order_p/delta so `order_p < alpha` fails.
        if or_strength is None or or_strength == "missing":
            order_p, order_delta = float("nan"), float("nan")
        else:
            order_p, order_delta = _f(orr.get("empirical_p")), _f(orr.get("delta"))

        cr = coh_common.get(band, {})
        ap = coh_aperi.get(band, {})

        # Forward fix #1: every numeric input is a native python float (via _f) before the and-chain.
        flags = gate_pass_flags(
            spatial_p=_f(sp.get("empirical_p")), spatial_delta=_f(sp.get("delta")),
            spatial_strength=(sp_strength if sp_strength is not None else "subject_wide_weak"),
            order_p=order_p, order_delta=order_delta,
            order_strength=(or_strength if or_strength is not None else "missing"),
            common_resid_p=_f(cr.get("empirical_p")), common_resid_delta=_f(cr.get("delta")),
            aperiodic_p=_f(ap.get("empirical_p")), aperiodic_delta=_f(ap.get("delta")),
            band_max_over_bands_p=band_mob, band=band, fs_subset=substrate, alpha=alpha)
        tier = gate_tier(flags, band)

        rows.append({
            "axis_set": substrate, "cohort": substrate, "band": band, "feature": feature,
            "gate_A_spatial_pass": bool(flags["gate_A_spatial"]),
            "gate_A_order_pass": bool(flags["gate_A_order"]),
            "gate_B_frequency_specific_pass": bool(flags["gate_B_freq_specific"]),
            "gate_C_HFO_specific_pass": bool(flags["gate_C_hfo_specific"]),
            "cohort_delta": _f(sp.get("delta")), "cohort_null_z": _f(sp.get("null_z")),
            "cohort_empirical_p": _f(sp.get("empirical_p")), "max_over_bands_p": band_mob,
            "n_subjects_valid": int(sp.get("n", 0)), "interpretation_tier": tier,
            "spatial_null_strength": sp_strength, "order_null_strength": or_strength})
    return rows, family_bands


def _write_csv(path, cols, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--substrate", choices=["broad", "narrow"], default="broad")
    ap.add_argument("--feature", default="raw",
                    help="primary Gate A alignment feature whose nulls drive the gate (default raw); "
                         "Gate B/C always read common_resid/aperiodic_resid summaries if present")
    ap.add_argument("--outdir", default=None, help="root (reads/writes {outdir}/{substrate}); default results tree")
    args = ap.parse_args()

    cfg = load_phase1_config()
    outroot = Path(args.outdir) if args.outdir else V2_ROOT
    sub_dir = outroot / args.substrate
    sub_dir.mkdir(parents=True, exist_ok=True)

    rows, family_bands = build_gate_rows(sub_dir, args.substrate, args.feature, cfg)
    _write_csv(sub_dir / "phase1_gate_summary.csv", GATE_COLS, rows)

    tiers = {}
    for r in rows:
        tiers[r["interpretation_tier"]] = tiers.get(r["interpretation_tier"], 0) + 1
    meta = {"generated_by": "run_topic5_v2_gates.py", "substrate": args.substrate,
            "feature": args.feature, "alpha": float(cfg["nulls"]["alpha"]),
            "primary_null_type": PRIMARY_NULL_TYPE, "n_bands": len(rows), "family_bands": family_bands,
            "excluded_repro_bands": sorted(set(cfg["repro_bands"].values())),
            "common_resid_present": (sub_dir / "phase1_null_common_resid_subject_summary.csv").exists(),
            "aperiodic_resid_present": (sub_dir / "phase1_null_aperiodic_resid_subject_summary.csv").exists(),
            "tier_counts": tiers}
    (sub_dir / "phase1_gate_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(f"[done] {len(rows)} gate rows ({args.substrate}, feature={args.feature}) -> {sub_dir} "
          f"tiers={tiers}", flush=True)


if __name__ == "__main__":
    main()
