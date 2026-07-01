"""Topic 5 V2 Phase-1 alignment tables — early-ictal band field vs interictal HFO geometry.

测了什么：发作刚起头的那 20 秒里，每个频带（δ 到 ripple）的能量在电极阵列上"点亮"成
一张空间图；间期时同一批触点各自有一个"平时谁先谁后"的几何顺序（typical_rank）。这个脚本
把两者对齐——发作早期某频带的能量场，长得像不像间期那张顺序几何图。

怎么测的：对每个被试、每个发作、每个频带、每个"合格早窗"（窗口和发作前 20s 有重叠 且 窗内至少
一半时间在发作里），把频带 z 迹在窗内取均值 → 每触点一个数 → 只保留固定分析掩膜里的触点 →
(1) 场层面 align_abs_maxab = window_maxab（平滑场与间期 A/B 场的最大 |相关|）；(2) 触点层面
contact_alignment（窗值 vs 间期 typical_rank 的 signed Spearman，定向模板 a）。窗 → 发作中位数 →
被试中位数（被试才是统计单元；broad/narrow 永不合并）。

固定掩膜 = 该被试的 analysis_channels（跨所有频带同一批触点），落地为 analysis_channels ∩ mapped ∩
cache（analysis_channels 含无几何坐标的触点，需与 mapped 求交）。--feature raw 读 v2 band cache；
common_resid/aperiodic_resid 读残差 cache（Task 10b/11b，未建成，分支已就位）。

设计: docs/superpowers/specs/2026-07-01-topic5-v2-hfo-critical-mode-design.md; plan Task 7 (P1-d)。
"""
from __future__ import annotations
import argparse, csv, json, sys, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

from scripts.run_topic5_ictal_field_dynamics import (  # noqa: E402
    load_context, window_maxab, _slice, _zmean_by_name, _ictal_fraction,
    SUBJECTS_BY_SUB, CACHE as LONG_CACHE)
from src.topic5_v2_band_scan import contact_alignment, load_phase1_config  # noqa: E402

V2_ROOT = _ROOT / "results/topic5_ictal_recruitment/v2_band_scan"
RAW_CACHE_DIR = V2_ROOT / "cache"                    # canonical sidecar (fixed mask + per-band QC)
FEATURE_CACHE_DIR = {                                # z-trace source per --feature
    "raw": RAW_CACHE_DIR,
    "common_resid": V2_ROOT / "common_resid_cache",   # built by Task 10b (same keys as band cache)
    "aperiodic_resid": V2_ROOT / "aperiodic_resid_cache",  # built by Task 11b (same keys)
}
MIN_CONTACTS = 4          # smallest N any alignment metric accepts (contact_alignment needs >=4)
METRIC_COLS = ["align_abs_maxab", "align_signed_oriented", "align_signed_posthoc_max",
               "signed_spearman_a", "signed_spearman_b", "n_contacts"]

WINDOW_COLS = ["subject", "axis_set", "seizure", "band", "feature", "win_start_rel", "win_end_rel",
               "win_center_rel", "ictal_fraction", "strict_onset", "align_abs_maxab",
               "align_signed_oriented", "align_signed_posthoc_max", "signed_spearman_a",
               "signed_spearman_b", "n_contacts", "band_eff_frac", "fs_edge_flag", "used_fixed_mask"]
SEIZURE_COLS = ["subject", "axis_set", "seizure", "band", "feature", "used_fixed_mask", "n_windows",
                "n_strict_onset"] + METRIC_COLS
SUBJECT_COLS = ["subject", "axis_set", "band", "feature", "used_fixed_mask", "n_seizures"] \
    + METRIC_COLS + ["fixed_mask_delta", "unmasked_align_abs_maxab", "n_channels_dropped_by_fixed_mask"]


def _config_bands(cfg):
    """All analysis bands (7 primary half-open + 4 composite closed) + legacy repro band."""
    primary = [b[0] for b in cfg["bands"]["primary"]]
    composites = [b[0] for b in cfg["bands"]["composites"]]
    legacy = cfg["repro_bands"]["bb"]                # legacy_bb_1_45
    return primary + composites + [legacy], primary, legacy


def _epoch_grid(cfg):
    """Fixed epoch grid (config `epoch`). A window (start, start+w) is a candidate iff it OVERLAPS
    the early-ictal region (0, main_rel[1]): win_end_rel>0 AND win_start_rel<main_rel[1] — this
    interval-overlap rule admits onset-straddling windows (start<0) and past-20s windows, which the
    `strict_onset` flag then distinguishes. Grid anchored at main_rel[0], stepped by field_step_sec.
    ictal_fraction is applied per-seizure (needs the offset), not here."""
    e = cfg["epoch"]
    w, step = float(e["field_window_sec"]), float(e["field_step_sec"])
    r0, r1 = float(e["main_rel"][0]), float(e["main_rel"][1])
    k_lo = int(np.floor((-w - r0) / step)) - 1
    k_hi = int(np.ceil((r1 - r0) / step)) + 1
    grid = []
    for k in range(k_lo, k_hi + 1):
        st = r0 + k * step
        en = st + w
        if en > 0 and st < r1:                       # VALID-EARLY overlap rule (part 1)
            grid.append((float(st), float(en)))
    return grid, r1


def _resolve_ta_tb_rank(ctx):
    """Interictal per-contact typical_rank maps for templates A/B (contact-level geometry),
    re-derived exactly as load_context builds them (geometry-mapped + finite rank only)."""
    def _rk(rec):
        return {c["name"]: float(c["typical_rank"]) for c in rec["channels"]
                if c["name"] in ctx["pos"] and np.isfinite(c.get("typical_rank", np.nan))}
    return _rk(ctx["ta"]), _rk(ctx["tb"])


def _seizure_idxs(fcache):
    return sorted({int(f.rsplit("__", 1)[1]) for f in fcache.files if "__zt__" in f})


def _band_good_names(zt, cache_names, qc):
    """--sensitivity per-band good set: finite-z AND not flagged bad_channel for this (band,seizure)
    (mirrors Task-6 `good = any-finite ∧ ¬bad`). PRIMARY never calls this — it uses the fixed mask."""
    bad = set(qc.get("bad_channels", []))
    finite = np.any(np.isfinite(zt), axis=1)
    n = min(len(cache_names), zt.shape[0])
    return {cache_names[i] for i in range(n) if finite[i] and cache_names[i] not in bad}


def run_subject(ds_sid, substrate, feature, cfg, sensitivity=False, feature_cache_dir=None):
    """Emit window rows for one subject. Returns (window_rows, legacy_unmasked_rows, n_dropped).
    legacy_unmasked_rows = full-pool (unmasked) align_abs_maxab of the legacy band, per window —
    the same-window baseline the QC-2 (P1-d) fixed-mask delta is measured against.
    ``feature_cache_dir`` overrides the z-trace cache dir (default = canonical FEATURE_CACHE_DIR
    per feature); used for isolated smoke tests of the residual caches (10b/11b)."""
    ctx = load_context(ds_sid, substrate)
    long_meta = json.loads((LONG_CACHE / f"{ds_sid}.json").read_text())
    off_by_idx = {int(k): float(v["eeg_offset_rel"]) for k, v in long_meta["seizure"].items()}

    feat_dir = feature_cache_dir or FEATURE_CACHE_DIR[feature]
    fpath = feat_dir / f"{ds_sid}.npz"
    if not fpath.exists():
        raise FileNotFoundError(
            f"{feature} cache missing for {ds_sid}: {fpath}. "
            f"(raw=v2 band cache Task 6; common_resid=Task 10b; aperiodic_resid=Task 11b — not built yet.)")
    fcache = np.load(fpath, allow_pickle=True)
    raw_side = json.loads((RAW_CACHE_DIR / f"{ds_sid}.json").read_text())  # mask + QC are feature-independent
    cache_names = [str(x) for x in fcache["channels"]]

    # ---- SCIENCE CONTRACT: subject-fixed analysis mask (issue #8, T6b revision) ----
    basis = raw_side.get("analysis_channels_basis")
    assert basis == "primary_bands_validity", (
        f"{ds_sid} analysis_channels_basis={basis!r} — expected 'primary_bands_validity' "
        f"(the Task-6b validity revision; NOT the old 'primary_bands'). Refusing to align on a stale mask.")
    analysis_channels = list(raw_side["analysis_channels"])
    assert set(analysis_channels) <= set(cache_names), (
        f"{ds_sid} analysis_channels escapes cache channels (mask ill-formed).")
    pool = [n for n in ctx["mapped"] if n in set(cache_names)]                 # mapped ∩ cache (unmasked pool)
    fixed_mask = [n for n in pool if n in set(analysis_channels)]              # analysis_channels ∩ mapped ∩ cache
    # analysis_channels legitimately contains non-geometry-mapped contacts, so it is NOT a subset of
    # mapped∩cache; the EFFECTIVE fixed mask (the intersection) is what must stay inside mapped∩cache.
    assert set(fixed_mask) <= (set(ctx["mapped"]) & set(cache_names)), \
        f"{ds_sid} effective fixed mask escapes mapped∩cache."
    assert fixed_mask, f"{ds_sid} empty fixed mask after intersecting analysis_channels ∩ mapped ∩ cache."
    n_dropped = int(raw_side.get("n_channels_dropped_by_fixed_mask", 0))
    mask_set = set(fixed_mask)

    ta_rank, tb_rank = _resolve_ta_tb_rank(ctx)
    all_bands, _primary, legacy_band = _config_bands(cfg)
    grid, r1 = _epoch_grid(cfg)
    ictal_min = float(cfg["epoch"]["ictal_fraction_min"])
    subject = ds_sid.split("_", 1)[1]
    band_qc = raw_side.get("bands", {})

    rows, legacy_unmasked = [], []
    for idx in _seizure_idxs(fcache):
        off = off_by_idx.get(idx)
        if off is None:
            continue
        for band in all_bands:
            zt_key, relt_key = f"{band}__zt__{idx}", f"{band}__relt__{idx}"
            if zt_key not in fcache.files:            # band Nyquist-skipped for this subject/seizure
                continue
            zt, relt = fcache[zt_key], fcache[relt_key]
            qc = band_qc.get(band, {}).get(str(idx), {})
            eff = float(qc.get("eff_frac", float("nan")))
            edge = bool(qc.get("fs_edge_flag", False))
            win_mask = (_band_good_names(zt, cache_names, qc) & set(pool)) if sensitivity else mask_set
            for st, en in grid:
                if not (en > 0 and st < r1):
                    continue
                zmv, _sub = _slice(zt, relt, st, en)
                if zmv is None:
                    continue
                icf = _ictal_fraction(relt, st, en, off)
                if not (np.isfinite(icf) and icf >= ictal_min):   # VALID-EARLY rule (part 2)
                    continue
                zmn_pool = _zmean_by_name(zmv, cache_names, pool)          # full-pool window means
                zmn = {n: v for n, v in zmn_pool.items() if n in win_mask}  # RESTRICT to fixed mask
                if len(zmn) < MIN_CONTACTS:
                    continue
                ca = contact_alignment(zmn, ta_rank, tb_rank, oriented_template="a")
                rows.append(dict(
                    subject=subject, axis_set=substrate, seizure=idx, band=band, feature=feature,
                    win_start_rel=round(st, 3), win_end_rel=round(en, 3),
                    win_center_rel=round((st + en) / 2.0, 3), ictal_fraction=round(icf, 4),
                    strict_onset=bool(st >= 0 and en <= r1),
                    align_abs_maxab=window_maxab(ctx, zmn),
                    align_signed_oriented=ca["align_signed_oriented"],
                    align_signed_posthoc_max=ca["align_signed_posthoc_max"],
                    signed_spearman_a=ca["signed_spearman_a"], signed_spearman_b=ca["signed_spearman_b"],
                    n_contacts=int(ca["n_contacts_a"]), band_eff_frac=eff, fs_edge_flag=edge,
                    used_fixed_mask=(not sensitivity)))
                if band == legacy_band and not sensitivity:               # QC-2 same-window unmasked baseline
                    legacy_unmasked.append(dict(seizure=idx,
                                                align_abs_maxab=window_maxab(ctx, zmn_pool)))
    return rows, legacy_unmasked, n_dropped


def _median_over(items, metric_cols):
    out = {}
    for m in metric_cols:
        vals = [it[m] for it in items if it.get(m) is not None and np.isfinite(it[m])]
        out[m] = float(np.nanmedian(vals)) if vals else float("nan")
    return out


def _seizure_summary(window_rows):
    groups = defaultdict(list)
    for r in window_rows:
        groups[(r["subject"], r["axis_set"], r["seizure"], r["band"], r["feature"],
                r["used_fixed_mask"])].append(r)
    out = []
    for (subject, axis_set, seizure, band, feature, ufm), rs in groups.items():
        d = dict(subject=subject, axis_set=axis_set, seizure=seizure, band=band, feature=feature,
                 used_fixed_mask=ufm, n_windows=len(rs),
                 n_strict_onset=int(sum(bool(r["strict_onset"]) for r in rs)))
        d.update(_median_over(rs, METRIC_COLS))
        out.append(d)
    return out


def _subject_summary(seizure_rows, feature, legacy_band, legacy_unmasked_by_sub, n_dropped_by_sub):
    """Subject = median over seizure medians (★ statistical unit). For the legacy band, attach the
    QC-2 (P1-d) record: fixed_mask_delta = fixed-mask subject median − same-window UNMASKED subject
    median; both are aggregated window→seizure→subject identically. RECORD-ONLY unless the subject's
    n_channels_dropped_by_fixed_mask==0 (then the fixed mask is a no-op on the pool and the delta MUST
    be ~0 — a hard invariant, checked by the caller)."""
    groups = defaultdict(list)
    for r in seizure_rows:
        groups[(r["subject"], r["axis_set"], r["band"], r["feature"], r["used_fixed_mask"])].append(r)
    out = []
    for (subject, axis_set, band, feature_, ufm), rs in groups.items():
        d = dict(subject=subject, axis_set=axis_set, band=band, feature=feature_,
                 used_fixed_mask=ufm, n_seizures=len(rs))
        d.update(_median_over(rs, METRIC_COLS))
        d["fixed_mask_delta"] = ""
        d["unmasked_align_abs_maxab"] = ""
        d["n_channels_dropped_by_fixed_mask"] = ""
        if band == legacy_band and ufm is True:
            n_dropped = n_dropped_by_sub.get((subject, axis_set), 0)
            unmasked_windows = legacy_unmasked_by_sub.get((subject, axis_set), [])
            # window→seizure→subject median of the unmasked baseline (same aggregation as the fixed side)
            by_sz = defaultdict(list)
            for w in unmasked_windows:
                by_sz[w["seizure"]].append(w)
            sz_med = [float(np.nanmedian([w["align_abs_maxab"] for w in ws
                                          if np.isfinite(w["align_abs_maxab"])])) for ws in by_sz.values()]
            sz_med = [v for v in sz_med if np.isfinite(v)]
            unmasked = float(np.nanmedian(sz_med)) if sz_med else float("nan")
            fixed = d["align_abs_maxab"]
            d["unmasked_align_abs_maxab"] = unmasked
            d["fixed_mask_delta"] = (fixed - unmasked) if np.isfinite(fixed) and np.isfinite(unmasked) \
                else float("nan")
            d["n_channels_dropped_by_fixed_mask"] = n_dropped
        out.append(d)
    return out


def _write_csv(path, cols, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feature", choices=list(FEATURE_CACHE_DIR), default="raw")
    ap.add_argument("--substrate", choices=list(SUBJECTS_BY_SUB), default="broad")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--outdir", default=None, help="output root (default results/.../v2_band_scan)")
    ap.add_argument("--feature-cache-dir", default=None,
                    help="override the --feature z-trace cache dir (default: canonical per-feature dir). "
                         "Production leaves this unset; used for isolated smoke tests of 10b/11b residual caches.")
    ap.add_argument("--sensitivity", action="store_true",
                    help="band-wise (per-band good) mask instead of the subject-fixed mask (NOT primary)")
    args = ap.parse_args()
    feat_dir = Path(args.feature_cache_dir) if args.feature_cache_dir else FEATURE_CACHE_DIR[args.feature]
    cfg = load_phase1_config()
    _all_bands, _primary, legacy_band = _config_bands(cfg)
    tol = float(cfg["tolerances"]["legacy_subject_median_abs"])
    subjects = args.subjects or SUBJECTS_BY_SUB[args.substrate]
    outroot = Path(args.outdir) if args.outdir else V2_ROOT
    outdir = outroot / args.substrate
    outdir.mkdir(parents=True, exist_ok=True)

    all_windows, legacy_unmasked_by_sub, n_dropped_by_sub = [], {}, {}
    for ds_sid in subjects:
        if not (feat_dir / f"{ds_sid}.npz").exists():
            print(f"[skip] {ds_sid} no {args.feature} cache", flush=True)
            continue
        rows, legacy_unmasked, n_dropped = run_subject(
            ds_sid, args.substrate, args.feature, cfg, sensitivity=args.sensitivity,
            feature_cache_dir=feat_dir)
        all_windows += rows
        subject = ds_sid.split("_", 1)[1]
        legacy_unmasked_by_sub[(subject, args.substrate)] = legacy_unmasked
        n_dropped_by_sub[(subject, args.substrate)] = n_dropped
        print(f"[{ds_sid}] {len(rows)} window-rows, n_dropped_by_fixed_mask={n_dropped}, "
              f"axis_set={args.substrate}, feature={args.feature}", flush=True)

    seizure_rows = _seizure_summary(all_windows)
    subject_rows = _subject_summary(seizure_rows, args.feature, legacy_band,
                                    legacy_unmasked_by_sub, n_dropped_by_sub)

    # QC-2 (P1-d): fixed-mask delta is RECORD-ONLY, EXCEPT when zero channels were dropped — then the
    # fixed mask cannot change the legacy result and the delta MUST be within tolerance (hard invariant).
    for d in subject_rows:
        if d["band"] == legacy_band and d["used_fixed_mask"] is True \
                and d["n_channels_dropped_by_fixed_mask"] == 0:
            delta = d["fixed_mask_delta"]
            if not np.isfinite(delta):
                continue  # degenerate subject (no valid legacy early windows) -> record-only; not a mask bug, don't abort the cohort run (review T7-M3)
            assert abs(delta) <= tol, (
                f"QC-2 invariant violated: {d['subject']} legacy_bb fixed_mask_delta={delta} > tol={tol} "
                f"with n_channels_dropped_by_fixed_mask==0 (fixed mask must be a no-op on the pool).")

    infix = "_sensitivity" if args.sensitivity else ""
    stem = f"phase1_alignment_{args.feature}{infix}"
    _write_csv(outdir / f"{stem}_window_long.csv", WINDOW_COLS, all_windows)
    _write_csv(outdir / f"{stem}_seizure_summary.csv", SEIZURE_COLS, seizure_rows)
    _write_csv(outdir / f"{stem}_subject_summary.csv", SUBJECT_COLS, subject_rows)
    print(f"[done] {len(all_windows)} windows / {len(seizure_rows)} seizure-band / "
          f"{len(subject_rows)} subject-band rows -> {outdir}/{stem}_*.csv", flush=True)


if __name__ == "__main__":
    main()
