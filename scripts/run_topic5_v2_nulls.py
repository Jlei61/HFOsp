"""Topic 5 V2 Phase 1 — Task 13: two-layer null orchestration (the Gate A statistical core).

测了什么（白话）：发作刚起头那 20 秒，某个频带的能量在电极阵列上点亮成一张空间图；间期时同一批触
点有一张"平时谁先谁后"的顺序几何图（G_HFO）。前面的 alignment 脚本量了两者"长得像不像"（一个 0–1
的相关数）。但"像"可能只是因为(1)能量本来就空间平滑、相邻触点自然像；(2)只是跟着"哪些触点 HFO 多"
走。这个脚本回答的就是命门问题：这份"像"到底有没有超过这两种巧合。

怎么测的（两把尺子，都把"被试"当统计单元，窗→发作中位→被试中位）：
  • 空间零假设：把发作那张能量图在"同一根电极杆内部"随机洗牌很多次（洗值不洗触点位置），每洗一次
    重算一遍"像不像"。如果真信号只是空间平滑，洗牌后的"像"应该跟没洗差不多；如果实测的"像"明显
    比洗牌分布高，就说明不是单纯平滑能解释的。只有真能"杆内洗"（within_shaft_strong）的被试才
    有资格进正式 Gate A（杆太短、退化到跨杆洗的只当描述）。
  • 顺序零假设：保留每个触点"参与了多少次 HFO"（HFO 富集地形原样不动），只把每个事件内部的时间
    先后打乱、重建间期几何。若"像"只是跟着 HFO 富集触点走，这样重建的假几何也一样能对上；若实测
    的"像"明显高于这个假几何分布，就说明是间期的"顺序"本身携带了信息，而不是富集地形。

揭示了什么：不是"PASS/NULL"，而是每被试每频带一个"实测的像 vs 洗牌/假几何的像"对比 + 经验 p。
真正的 Gate A 判决在 Task 14；这里只把每次置换的被试中位数如实落成一张 permutation-level 长表
（max-over-bands 需要它）+ 一张 per-(被试,频带,零假设类型) 摘要。第三张 confound 表是"把 G_HFO 顺序
先扣掉 HFO 率/功率/杆位地形后再对齐"的确定性调整（不是零假设，除非 --confound-null）。

代号补注：G_HFO=间期 typical_rank；spatial null=spatial_constrained_permute（within_shaft，issue #10）；
order null=order_null_rank_pair（Patch F，Task 9，保留 participation/HFO 率、打乱 timing）；confound=
confound_residual_rank（Task 12）× phase1_confound_maps.json（Task 12a）；subject=统计单元；perm-long=Patch E。

设计: docs/superpowers/specs/2026-07-01-topic5-v2-hfo-critical-mode-design.md; plan Task 13 (Patch E/H).
"""
from __future__ import annotations
import argparse, hashlib, json, sys, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

# REUSE (do not reinvent): the exact observed-statistic backbone the alignment script uses.
from scripts.run_topic5_ictal_field_dynamics import (  # noqa: E402
    load_context, window_maxab, _slice, _zmean_by_name, _ictal_fraction,
    SUBJECTS_BY_SUB, CACHE as LONG_CACHE)
from scripts.run_topic5_v2_alignment import (  # noqa: E402  (window/epoch/fixed-mask logic mirrored from Task 7)
    _config_bands, _epoch_grid, _resolve_ta_tb_rank, _seizure_idxs,
    FEATURE_CACHE_DIR, RAW_CACHE_DIR, MIN_CONTACTS)
from src.topic5_axis_alignment import make_field_record  # noqa: E402
from src.propagation_contact_plane_readout import R_smooth_rank, S_THRESH  # noqa: E402
from src.propagation_skeleton_geometry import parse_shaft  # noqa: E402
from src.topic5_v2_band_scan import (  # noqa: E402
    load_phase1_config, spatial_constrained_permute, rebuild_typical_rank,
    order_null_rank_pair, _order_null_one_template, confound_residual_rank)
import src.topic5_event_resolved_alignment as erm  # noqa: E402
# REUSE the Task-9 dep-check loaders so order_null_strength here == the dep-check's (same helpers).
from scripts.run_topic5_v2_order_null_depcheck import (  # noqa: E402
    _SUBSTRATE as ORDER_SUBSTRATE, _load_bundle, _spearman_shared)

V2_ROOT = _ROOT / "results/topic5_ictal_recruitment/v2_band_scan"
_STRENGTH_TIER = {"within_shaft_strong": 3, "distance_bin_fallback": 2, "subject_wide_weak": 1}

PERM_COLS = ["subject", "axis_set", "feature", "null_type", "band", "perm_id", "perm_subject_median"]
SUMMARY_COLS = ["subject", "axis_set", "feature", "band", "null_type", "obs_subject_median",
                "null_median", "null_mad", "null_z", "delta", "empirical_p",
                "spatial_null_strength", "order_null_strength", "n_perm_effective"]
CONFOUND_COLS = ["subject", "axis_set", "feature", "band", "covariate", "obs_align_to_G_HFO_resid",
                 "resid_null_z", "resid_empirical_p"]


# --------------------------------------------------------------------------- observed precompute
def precompute_observed(ds_sid, substrate, feature, cfg, feat_dir):
    """Load ctx + feature cache; per (band, seizure, window) store the fixed-mask window vals_by_name
    ONCE (issue #16) so perms only re-permute/re-smooth/re-correlate, never re-slice the cache.

    Mirrors run_topic5_v2_alignment.run_subject's PRIMARY fixed-mask path EXACTLY (same epoch grid,
    same valid-early + ictal_fraction rule, same analysis_channels ∩ mapped ∩ cache mask), minus the
    metric compute. Returns (ctx, ta_rank, tb_rank, obs_windows, fixed_mask) where
    obs_windows[band] = list of (seizure_idx, vals_by_name)."""
    ctx = load_context(ds_sid, substrate)
    long_meta = json.loads((LONG_CACHE / f"{ds_sid}.json").read_text())
    off_by_idx = {int(k): float(v["eeg_offset_rel"]) for k, v in long_meta["seizure"].items()}

    fpath = feat_dir / f"{ds_sid}.npz"
    if not fpath.exists():
        raise FileNotFoundError(f"{feature} cache missing for {ds_sid}: {fpath}")
    fcache = np.load(fpath, allow_pickle=True)
    raw_side = json.loads((RAW_CACHE_DIR / f"{ds_sid}.json").read_text())  # mask/QC are feature-independent
    cache_names = [str(x) for x in fcache["channels"]]

    basis = raw_side.get("analysis_channels_basis")
    assert basis == "primary_bands_validity", (
        f"{ds_sid} analysis_channels_basis={basis!r} — expected 'primary_bands_validity' (Task-6b). "
        f"Refusing to run nulls on a stale mask.")
    analysis_channels = list(raw_side["analysis_channels"])
    assert set(analysis_channels) <= set(cache_names), f"{ds_sid} analysis_channels escapes cache channels."
    pool = [n for n in ctx["mapped"] if n in set(cache_names)]                # mapped ∩ cache
    fixed_mask = [n for n in pool if n in set(analysis_channels)]             # analysis_channels ∩ mapped ∩ cache
    assert set(fixed_mask) <= (set(ctx["mapped"]) & set(cache_names)), f"{ds_sid} fixed mask escapes mapped∩cache."
    assert fixed_mask, f"{ds_sid} empty fixed mask after intersecting analysis_channels ∩ mapped ∩ cache."
    mask_set = set(fixed_mask)

    all_bands, _primary, _legacy = _config_bands(cfg)
    grid, r1 = _epoch_grid(cfg)
    ictal_min = float(cfg["epoch"]["ictal_fraction_min"])
    obs = defaultdict(list)
    for idx in _seizure_idxs(fcache):
        off = off_by_idx.get(idx)
        if off is None:
            continue
        for band in all_bands:
            zt_key, relt_key = f"{band}__zt__{idx}", f"{band}__relt__{idx}"
            if zt_key not in fcache.files:                # band Nyquist-skipped for this subject/seizure
                continue
            zt, relt = fcache[zt_key], fcache[relt_key]
            for st, en in grid:
                if not (en > 0 and st < r1):
                    continue
                zmv, _sub = _slice(zt, relt, st, en)
                if zmv is None:
                    continue
                icf = _ictal_fraction(relt, st, en, off)
                if not (np.isfinite(icf) and icf >= ictal_min):
                    continue
                zmn_pool = _zmean_by_name(zmv, cache_names, pool)
                zmn = {n: v for n, v in zmn_pool.items() if n in mask_set}    # RESTRICT to fixed mask
                if len(zmn) < MIN_CONTACTS:
                    continue
                obs[band].append((idx, zmn))
    ta_rank, tb_rank = _resolve_ta_tb_rank(ctx)
    return ctx, ta_rank, tb_rank, dict(obs), fixed_mask


# --------------------------------------------------------------------------- statistic + fields
def _subject_median(ctx_like, windows):
    """FULL statistic aggregation: window_maxab per window → nanmedian over windows in a seizure →
    nanmedian over seizure medians (subject = the unit). Identical to the alignment script's
    window→seizure→subject fold of align_abs_maxab."""
    by_sz = defaultdict(list)
    for idx, vals in windows:
        by_sz[idx].append(window_maxab(ctx_like, vals))
    sz_med = []
    for vs in by_sz.values():
        f = [v for v in vs if np.isfinite(v)]
        if f:
            sz_med.append(float(np.nanmedian(f)))
    return float(np.nanmedian(sz_med)) if sz_med else float("nan")


def _field_from_rank(ctx, rank_by_name):
    """Rebuild an interictal field on ctx's plane with typical_rank := rank_by_name (by name), NaN
    where absent. Same sigma as everything else (ctx['sigma']) so the null/adjusted geometry is
    smoothed on identical footing to the observed. <4 finite contacts → None (matches load_context's
    F_inter_b guard: nothing meaningful to smooth)."""
    vals = [rank_by_name.get(c["name"], np.nan) for c in ctx["matched"]]
    if int(np.isfinite(np.asarray(vals, float)).sum()) < 4:
        return None
    return R_smooth_rank(make_field_record(ctx["matched"], vals), ctx["X"], ctx["Y"], ctx["sigma"], S_THRESH)


def _weaker(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a if _STRENGTH_TIER[a] <= _STRENGTH_TIER[b] else b


# --------------------------------------------------------------------------- spatial null
def _spatial_perm_window(vals_by_name, ctx, rng, min_group):
    """One within-shaft spatial permutation of a window's ictal per-contact values (issue #10).
    Returns (perm_vals_by_name, spatial_null_strength)."""
    names = list(vals_by_name)
    shaft_by = {n: parse_shaft(n)[0] for n in names}
    coord_by = {n: ctx["pos"][n] for n in names}
    perm, st = spatial_constrained_permute(names, vals_by_name, shaft_by, coord_by, rng,
                                            "within_shaft", min_group)
    return perm, st["spatial_null_strength"]


# --------------------------------------------------------------------------- order null inputs
def load_order_inputs(ds_sid, substrate, ctx, min_corr):
    """Reuse the Task-9 dep-check loading (C1-proven event tables + cluster→template map) to get the
    per-template participation/lag tables AND the order_null_strength for THIS subject.

    order_null_strength here == the dep-check's (same `_load_bundle`, `map_clusters_to_templates`,
    `rebuild_typical_rank(agg='median')`, `_spearman_shared`, same min_corr) — computed inline so we
    do not depend on the depcheck CSV having been run first. Returns a dict:
      {strength, channel_names, events_a, lag_a, events_b, lag_b}  (events_* None if that template
      has no events). strength='missing' → no trustworthy order null (caller emits NaN order p/delta)."""
    ds, subj = ds_sid.split("_", 1)
    ocfg = ORDER_SUBSTRATE[substrate]
    try:
        bundle = _load_bundle(ds, subj, ocfg)
    except (FileNotFoundError, ValueError, KeyError):
        return {"strength": "missing"}
    order = bundle["channel_names"]
    ta_by = {c["name"]: c.get("typical_rank") for c in ctx["ta"]["channels"]}
    tb_by = {c["name"]: c.get("typical_rank") for c in ctx["tb"]["channels"]}
    ta = np.array([ta_by[n] if (n in ta_by and ta_by[n] is not None) else np.nan for n in order], float)
    tb = np.array([tb_by[n] if (n in tb_by and tb_by[n] is not None) else np.nan for n in order], float)
    cmap = erm.map_clusters_to_templates(bundle["cluster_template_ranks"][0],
                                         bundle["cluster_template_ranks"][1], ta, tb)
    if cmap["ambiguous"]:
        return {"strength": "missing"}
    inv = {v: k for k, v in cmap["map"].items()}
    out = {"channel_names": order, "events_a": None, "lag_a": None, "events_b": None, "lag_b": None}
    corrs = {}
    for tid, producer, key in (("t_a", ta, "a"), ("t_b", tb, "b")):
        sel = bundle["labels"] == inv[tid]
        if int(sel.sum()) == 0:
            continue
        eb = bundle["bools"][:, sel].T                    # (n_events, n_ch)
        lg = bundle["masked"][:, sel].T                   # masked normalized rank = the quantity the null permutes
        out[f"events_{key}"], out[f"lag_{key}"] = eb, lg
        rebuilt = rebuild_typical_rank(eb, lg, agg="median")   # median = producer aggregator (Task-9 default)
        rho = _spearman_shared(rebuilt, producer)
        if rho is not None:
            corrs[key] = rho
    if not corrs:
        return {"strength": "missing"}
    out["strength"] = "strong" if all(c >= min_corr for c in corrs.values()) else "weak_downgrade"
    return out


def _order_null_ctx(ctx, order_inp, rng):
    """One HFO-rate-preserving order-null realization of the interictal geometry, coherent across
    ALL bands (built ONCE per perm). Returns a ctx-shaped dict with F_inter_a/F_inter_b swapped to
    the null fields, or None if neither template can be rebuilt."""
    ea, la = order_inp["events_a"], order_inp["lag_a"]
    eb, lb = order_inp["events_b"], order_inp["lag_b"]
    if ea is not None:
        ra_null, rb_null = order_null_rank_pair(ea, la, eb, lb, rng)   # rb_null=None if eb is None
    elif eb is not None:
        ra_null, rb_null = None, _order_null_one_template(eb, lb, rng)  # A-absent fallback (rare)
    else:
        return None
    order = order_inp["channel_names"]

    def _byname(rank):
        return {order[i]: float(rank[i]) for i in range(len(order)) if np.isfinite(rank[i])}

    F_a = _field_from_rank(ctx, _byname(ra_null)) if ra_null is not None else None
    F_b = _field_from_rank(ctx, _byname(rb_null)) if rb_null is not None else None
    if F_a is None and F_b is None:
        return None
    if F_a is None:                                        # window_maxab needs F_inter_a non-None
        F_a, F_b = F_b, None
    return dict(ctx, F_inter_a=F_a, F_inter_b=F_b)


# --------------------------------------------------------------------------- summary stats
def _summary_stats(obs_v, meds):
    """obs vs the null distribution of subject medians. empirical_p = add-one-smoothed one-sided
    P(perm >= obs); null_z = (obs - null_median)/null_mad (MAD-units); delta = obs - null_median."""
    arr = np.array([m for m in meds if np.isfinite(m)], float)
    n = int(arr.size)
    if n == 0 or not np.isfinite(obs_v):
        return dict(null_median=float("nan"), null_mad=float("nan"), null_z=float("nan"),
                    delta=float("nan"), empirical_p=float("nan"), n_perm_effective=n)
    null_median = float(np.median(arr))
    null_mad = float(np.median(np.abs(arr - null_median)))
    delta = float(obs_v - null_median)
    null_z = float(delta / null_mad) if null_mad > 0 else float("nan")
    empirical_p = float((1 + int(np.sum(arr >= obs_v))) / (1 + n))       # add-one smoothing
    return dict(null_median=null_median, null_mad=null_mad, null_z=null_z, delta=delta,
                empirical_p=empirical_p, n_perm_effective=n)


# --------------------------------------------------------------------------- confound adjustment
def _confound_covariate_maps(maps, config_covs):
    """Present, non-empty covariate maps in config order (+ soz if the map exists), name->float."""
    present = [c for c in config_covs if c in maps and maps[c]]
    if "soz" in maps and maps["soz"] and "soz" not in present:
        present.append("soz")
    return {c: {k: float(v) for k, v in maps[c].items()} for c in present}


def build_confound_adjusted(ds_sid, substrate, feature, ctx, ta_rank, obs_windows, cfg,
                            confound_maps_path, confound_null, rng, min_group):
    """Deterministic confound ADJUSTMENT table (Patch H; NOT a null). Residualize G_HFO (template-A
    typical_rank) against each confound covariate (Task 12), rebuild the interictal field from the
    residualized rank, and re-measure the observed alignment (obs_align_to_G_HFO_resid) per band.

    COMBINED confound drops the baseline/broadband collinearity (broadband_1_250 ≈ baseline_band_power)
    — single per-covariate residuals keep both. With --confound-null (count>0), the SPATIAL null is
    re-run on the adjusted geometry (resid_null_z/resid_empirical_p); off by default (expensive, not a
    formal Gate A tier). Order-null-on-adjusted is a deferred extension (see report)."""
    if not confound_maps_path.exists():
        return [], "no_confound_maps_file"
    data = json.loads(confound_maps_path.read_text())
    if ds_sid not in data:
        return [], "subject_absent_from_confound_maps"
    cov_maps = _confound_covariate_maps(data[ds_sid], list(cfg["nulls"]["confound_covariates"]))
    if not cov_maps:
        return [], "no_usable_covariates"

    res_all = confound_residual_rank(ta_rank, cov_maps)                  # single over ALL covariates
    resid_by_cov = dict(res_all["single"])
    pruned = {c: m for c, m in cov_maps.items() if c != "broadband_1_250"}  # collinearity-pruned COMBINED
    combined = confound_residual_rank(ta_rank, pruned)["combined"]
    if combined is not None:
        resid_by_cov["combined"] = combined

    subject = ds_sid.split("_", 1)[1]
    bands = sorted(obs_windows)
    rows = []
    for cov, resid_map in resid_by_cov.items():
        F_resid = _field_from_rank(ctx, {k: float(v) for k, v in resid_map.items()})
        if F_resid is None:
            continue
        ctx_r = dict(ctx, F_inter_a=F_resid, F_inter_b=None)            # align to residualized G_HFO(A) only
        for band in bands:
            obs_r = _subject_median(ctx_r, obs_windows[band])
            row = dict(subject=subject, axis_set=substrate, feature=feature, band=band, covariate=cov,
                       obs_align_to_G_HFO_resid=obs_r, resid_null_z="", resid_empirical_p="")
            if confound_null and np.isfinite(obs_r):                    # SPATIAL null on the adjusted rank
                sp_meds = []
                for _ in range(int(confound_null)):
                    pw = [(idx, _spatial_perm_window(v, ctx, rng, min_group)[0])
                          for idx, v in obs_windows[band]]
                    sp_meds.append(_subject_median(ctx_r, pw))
                s = _summary_stats(obs_r, sp_meds)
                row["resid_null_z"], row["resid_empirical_p"] = s["null_z"], s["empirical_p"]
            rows.append(row)
    return rows, ("confound_null_on" if confound_null else "deterministic_only")


# --------------------------------------------------------------------------- per-subject driver
def run_subject_nulls(ds_sid, substrate, feature, cfg, n_perm, seed, feat_dir, order_min_corr,
                      confound_maps_path, confound_null, min_group_override=None):
    ctx, ta_rank, tb_rank, obs_windows, fixed_mask = precompute_observed(
        ds_sid, substrate, feature, cfg, feat_dir)
    subject = ds_sid.split("_", 1)[1]
    # §1 min3 sensitivity: --min-group overrides the primary min_group=4 (config). Lower min_group
    # lets shorter shafts (>= min_group contacts) qualify for within-shaft permutation. NOTE: min3 is
    # a SENSITIVITY tier only (spec §1) — it does NOT promote to formal primary Gate A.
    min_group = int(min_group_override) if min_group_override is not None else int(cfg["nulls"]["min_group_for_shaft"])
    bands = sorted(obs_windows)
    if not bands:
        return [], [], [], {"ds_sid": ds_sid, "n_bands": 0, "reason": "no_valid_early_windows"}

    obs_med = {b: _subject_median(ctx, obs_windows[b]) for b in bands}
    order_inp = load_order_inputs(ds_sid, substrate, ctx, order_min_corr)
    order_strength = order_inp["strength"]

    # Reproducible, subject-order-independent RNG: subject-hashed seed, then independent per-perm
    # children per null type (perm p is one coherent null realization; order rebuilds ONCE per perm).
    sub_hash = int(hashlib.sha1(subject.encode()).hexdigest()[:8], 16)
    ss_sp, ss_or = np.random.SeedSequence([int(seed), sub_hash]).spawn(2)
    sp_children = ss_sp.spawn(n_perm)
    or_children = ss_or.spawn(n_perm)

    spatial_meds = {b: [] for b in bands}
    order_meds = {b: [] for b in bands}
    spatial_strength = {b: None for b in bands}

    for p in range(n_perm):
        rng_sp = np.random.default_rng(sp_children[p])
        for b in bands:
            pw = []
            for idx, vals in obs_windows[b]:
                perm_vals, strength = _spatial_perm_window(vals, ctx, rng_sp, min_group)
                pw.append((idx, perm_vals))
                if p == 0:
                    spatial_strength[b] = _weaker(spatial_strength[b], strength)
            spatial_meds[b].append(_subject_median(ctx, pw))
        if order_strength != "missing":
            ctx_p = _order_null_ctx(ctx, order_inp, np.random.default_rng(or_children[p]))
            for b in bands:
                order_meds[b].append(_subject_median(ctx_p, obs_windows[b]) if ctx_p is not None else float("nan"))

    perm_rows, summary_rows = [], []
    for b in bands:
        obs_v = obs_med[b]
        base = dict(subject=subject, axis_set=substrate, feature=feature, band=b)
        # observed rows (perm_id=-1) for both null types (the observed statistic is identical for both)
        perm_rows.append(dict(base, null_type="spatial", perm_id=-1, perm_subject_median=obs_v))
        perm_rows.append(dict(base, null_type="order", perm_id=-1, perm_subject_median=obs_v))
        for p, m in enumerate(spatial_meds[b]):
            perm_rows.append(dict(base, null_type="spatial", perm_id=p, perm_subject_median=m))
        for p, m in enumerate(order_meds[b]):
            perm_rows.append(dict(base, null_type="order", perm_id=p, perm_subject_median=m))

        s_sp = _summary_stats(obs_v, spatial_meds[b])
        summary_rows.append(dict(base, null_type="spatial", obs_subject_median=obs_v,
                                 spatial_null_strength=spatial_strength[b],
                                 order_null_strength=order_strength, **s_sp))
        if order_strength == "missing":
            # 'missing' order null → NaN p/delta so it can NEVER spuriously pass gate_A_order downstream.
            summary_rows.append(dict(base, null_type="order", obs_subject_median=obs_v,
                                     null_median=float("nan"), null_mad=float("nan"),
                                     null_z=float("nan"), delta=float("nan"), empirical_p=float("nan"),
                                     n_perm_effective=0, spatial_null_strength=spatial_strength[b],
                                     order_null_strength=order_strength))
        else:
            s_or = _summary_stats(obs_v, order_meds[b])
            summary_rows.append(dict(base, null_type="order", obs_subject_median=obs_v,
                                     spatial_null_strength=spatial_strength[b],
                                     order_null_strength=order_strength, **s_or))

    confound_rows, confound_note = build_confound_adjusted(
        ds_sid, substrate, feature, ctx, ta_rank, obs_windows, cfg, confound_maps_path,
        confound_null, np.random.default_rng([int(seed), sub_hash, 7]), min_group)
    info = {"ds_sid": ds_sid, "n_bands": len(bands), "order_null_strength": order_strength,
            "confound_note": confound_note,
            "spatial_strength_worst": _worst_strength(spatial_strength.values())}
    return perm_rows, summary_rows, confound_rows, info


def _worst_strength(strengths):
    worst = None
    for s in strengths:
        worst = _weaker(worst, s)
    return worst


# --------------------------------------------------------------------------- IO
def _write_csv(path, cols, rows):
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def assert_no_cohort_clobber(combined_path, n_this, allow_overwrite):
    """Refuse to overwrite a larger cohort combined output with a smaller (subset) run.

    A `--subjects <single>` job writes a 1-subject combined summary that silently clobbers
    the cohort artifact (this bit us: a broad-1146 rerun overwrote the broad-17 combined).
    Compare against the EXISTING file's subject count rather than a hardcoded cohort N because
    SUBJECTS_BY_SUB holds the stale 7/9 lists, not the true 20/17 cohort. Cohort shrinks that
    are intentional pass --allow-overwrite-combined.
    """
    if allow_overwrite or not combined_path.exists():
        return
    try:
        n_existing = int(pd.read_csv(combined_path, usecols=["subject"])["subject"].nunique())
    except Exception:
        return  # unreadable / schema drift → do not block a legitimate rewrite
    if n_existing > n_this:
        raise RuntimeError(
            f"Refusing to overwrite {combined_path.name} (has {n_existing} subjects) with a smaller "
            f"{n_this}-subject run — this prevents a single-/subset-subject job from clobbering the "
            f"cohort combined output. Aggregate the full cohort (--subjects <full list>) or pass "
            f"--allow-overwrite-combined for an intentional cohort shrink.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feature", choices=list(FEATURE_CACHE_DIR), default="raw")
    ap.add_argument("--substrate", choices=list(SUBJECTS_BY_SUB), default="broad")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--n-perm", type=int, default=None, help="default = config nulls.n_perm_smoke (20)")
    ap.add_argument("--outdir", default=None, help="output root (default results/.../v2_band_scan)")
    ap.add_argument("--feature-cache-dir", default=None,
                    help="override the --feature z-trace cache dir (residual-cache smoke tests)")
    ap.add_argument("--confound-maps", default=None,
                    help="phase1_confound_maps.json (default: {outdir|results}/.../{substrate}/phase1_confound_maps.json)")
    ap.add_argument("--confound-null", action="store_true",
                    help="also run a SPATIAL null (n_perm draws) on each confound-adjusted rank "
                         "(resid_null_z/resid_empirical_p); default off (deterministic adjustment only)")
    ap.add_argument("--min-group", type=int, default=None,
                    help="override nulls.min_group_for_shaft (§1 min3 SENSITIVITY within-shaft; "
                         "default = config 4 = formal primary). min3 does NOT promote to formal Gate A.")
    ap.add_argument("--allow-overwrite-combined", action="store_true",
                    help="override the guard that refuses to overwrite a larger cohort combined output "
                         "with a smaller subset run (intentional cohort shrink only)")
    args = ap.parse_args()

    cfg = load_phase1_config()
    n_perm = int(args.n_perm) if args.n_perm is not None else int(cfg["nulls"]["n_perm_smoke"])
    seed = int(cfg["nulls"]["seed"])
    order_min_corr = float(cfg["nulls"]["order_null_min_corr_to_geo"])
    feat_dir = Path(args.feature_cache_dir) if args.feature_cache_dir else FEATURE_CACHE_DIR[args.feature]
    subjects = args.subjects or SUBJECTS_BY_SUB[args.substrate]
    outroot = Path(args.outdir) if args.outdir else V2_ROOT
    outdir = outroot / args.substrate
    outdir.mkdir(parents=True, exist_ok=True)
    confound_maps_path = (Path(args.confound_maps) if args.confound_maps
                          else outroot / args.substrate / "phase1_confound_maps.json")
    confound_null_n = n_perm if args.confound_null else 0

    # Per-subject CHECKPOINT + RESUME (long full-n_perm runs get killed by session teardown; a restart
    # with identical args reloads finished subjects from _partial and only re-runs the rest, so at most
    # one subject's work is lost per kill). run_key pins the reuse contract — a partial from a different
    # n_perm/seed/min_group/order_min_corr is ignored and recomputed.
    partial_dir = outdir / f"_partial_{args.feature}"
    partial_dir.mkdir(parents=True, exist_ok=True)
    run_key = {"n_perm": n_perm, "seed": seed, "min_group_override": args.min_group,
               "order_min_corr": order_min_corr, "confound_null_n": confound_null_n,
               "substrate": args.substrate, "feature": args.feature}
    all_perm, all_summary, all_confound, per_subject = [], [], [], []
    for ds_sid in subjects:
        if not (feat_dir / f"{ds_sid}.npz").exists():
            print(f"[skip] {ds_sid} no {args.feature} cache", flush=True)
            continue
        mark = partial_dir / f"{ds_sid}.marker.json"
        if mark.exists() and json.loads(mark.read_text()).get("run_key") == run_key:
            perm_rows = pd.read_parquet(partial_dir / f"{ds_sid}.perm.parquet").to_dict("records")
            summary_rows = pd.read_csv(partial_dir / f"{ds_sid}.summary.csv").to_dict("records")
            cpath = partial_dir / f"{ds_sid}.confound.csv"
            confound_rows = (pd.read_csv(cpath).to_dict("records")
                             if cpath.exists() and cpath.stat().st_size > 0 else [])
            info = json.loads(mark.read_text())["info"]
            print(f"[resume] {ds_sid} from checkpoint ({len(perm_rows)} perm rows)", flush=True)
        else:
            perm_rows, summary_rows, confound_rows, info = run_subject_nulls(
                ds_sid, args.substrate, args.feature, cfg, n_perm, seed, feat_dir, order_min_corr,
                confound_maps_path, confound_null_n, min_group_override=args.min_group)
            pd.DataFrame(perm_rows, columns=PERM_COLS).to_parquet(partial_dir / f"{ds_sid}.perm.parquet", index=False)
            _write_csv(partial_dir / f"{ds_sid}.summary.csv", SUMMARY_COLS, summary_rows)
            _write_csv(partial_dir / f"{ds_sid}.confound.csv", CONFOUND_COLS, confound_rows)
            mark.write_text(json.dumps({"run_key": run_key, "info": info}))
            print(f"[{ds_sid}] {info['n_bands']} bands, order={info.get('order_null_strength')}, "
                  f"spatial_worst={info.get('spatial_strength_worst')}, confound={info.get('confound_note')}, "
                  f"n_perm={n_perm} [checkpointed]", flush=True)
        all_perm += perm_rows
        all_summary += summary_rows
        all_confound += confound_rows
        per_subject.append(info)

    stem = f"phase1_null_{args.feature}"
    # Overwrite guard (both the feature-infixed and the bare-alias combined) BEFORE writing anything,
    # so a subset run aborts without leaving a half-clobbered artifact tree.
    n_this = len({r["subject"] for r in all_summary})
    for name in (f"{stem}_subject_summary.csv", "phase1_null_subject_summary.csv"):
        assert_no_cohort_clobber(outdir / name, n_this, args.allow_overwrite_combined)
    # perm-long parquet: canonical feature-infixed + bare alias (the smoke asserts the bare name; the
    # infixed copy survives across features so the multi-feature full run never clobbers itself).
    perm_df = pd.DataFrame(all_perm, columns=PERM_COLS)
    for name in (f"{stem}_perm_subject_long.parquet", "phase1_null_perm_subject_long.parquet"):
        perm_df.to_parquet(outdir / name, engine="pyarrow", index=False)
    for name in (f"{stem}_subject_summary.csv", "phase1_null_subject_summary.csv"):
        _write_csv(outdir / name, SUMMARY_COLS, all_summary)
    for name in (f"phase1_confound_adjusted_{args.feature}_subject.csv", "phase1_confound_adjusted_subject.csv"):
        _write_csv(outdir / name, CONFOUND_COLS, all_confound)

    meta = {"generated_by": "run_topic5_v2_nulls.py", "feature": args.feature,
            "substrate": args.substrate, "seed": seed, "n_perm": n_perm,
            "spatial_mode": cfg["nulls"]["spatial"], "min_group_for_shaft": cfg["nulls"]["min_group_for_shaft"],
            "min_group_override": args.min_group,
            "order_null_min_corr_to_geo": order_min_corr, "alpha": cfg["nulls"]["alpha"],
            "confound_maps": str(confound_maps_path), "confound_maps_present": confound_maps_path.exists(),
            "confound_null_n": confound_null_n, "subjects": subjects, "per_subject": per_subject}
    (outdir / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    (outdir / "phase1_null_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(f"[done] {len(all_perm)} perm rows / {len(all_summary)} summary rows / "
          f"{len(all_confound)} confound rows -> {outdir} (feature={args.feature}, n_perm={n_perm})", flush=True)


if __name__ == "__main__":
    main()
