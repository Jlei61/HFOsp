"""Topic 5 V2 — aperiodic-corrected band-excess cache + robust-z (Task 11b, Gate C input).

测了什么 / why: Gate C 问的是——某个频带（首要目标是 ripple 80–250 Hz）在电极阵列上"点亮"的
能量，究竟是真的有**超出 1/f 背景**的振荡性余量（像局部有个鼓包），还是只是宽谱 1/f 背景整体
被抬高（功率大但没有频带特异的鼓包）？为公平回答它，我们对每个触点、每个时间点，先在 log-log
坐标上把 1/f 背景拟合成一条直线（over [fit_lo,fit_hi]，排除工频与非正值），再看目标频带里"实测
功率减去这条 1/f 地板、只留正的部分"有多少——这就是 aperiodic-corrected band excess power。若这个
余量在触点阵列上仍有稳定的空间几何（对齐脚本随后拿它去和间期 HFO 几何对齐），才算这个频带真的
带有超出 1/f 的振荡性结构，而不是背景整体变亮。

WHAT THIS CACHE IS: for every subject / eligible seizure / PRIMARY band B / channel c / time-bin tt,
we store the baseline-robust-z of a per-(c,tt) excess-power TRACE. Structure/keys are IDENTICAL to
the raw v2 band cache (``{B}__zt__{idx}`` (n_ch,n_bins float32), ``{B}__relt__{idx}``, ``channels``)
so ``run_topic5_v2_alignment.py --feature aperiodic_resid`` reads it exactly like the raw cache
(``FEATURE_CACHE_DIR['aperiodic_resid']`` = ``V2_ROOT/aperiodic_resid_cache``). Only the 7 PRIMARY
(half-open) bands are built — Gate C targets ripple; composites / legacy are not needed here.

HOW (single spectrogram, fit-once-per-(c,tt)): per seizure we compute the spectrogram ONCE via
``_spectrogram_on_hop`` (the SAME single-spectrogram pattern the band cache uses) → ``(f,t,Sxx)``
of shape ``(n_ch,n_freq,n_time)``, ``line_noise_bin_mask(f, …)`` once. The log-log 1/f fit over
``[fit_lo,fit_hi]`` depends only on ``(c,tt)``, NOT the band, so ``_excess_traces`` fits it ONCE per
``(c,tt)`` (vectorized closed-form OLS over the whole ``(n_ch,n_time)`` grid) and reuses it for every
band's excess — instead of refitting 7× inside ``aperiodic_corrected_excess_power``. The vectorized
math is byte-for-byte the Task-11 helper (verified cell-by-cell by
``test_aperiodic_vectorized_excess_matches_helper``); the (essentially-never) columns holding a
non-positive/non-finite PSD bin fall back to the scalar helper per ``(c,tt)`` so the result stays
exactly the helper's. ``excess_power`` is NaN when the fit is not ``ok`` (``n_fit<10`` or
``fit_r2<min_r2``) or the band has no valid bin (Task-11 contract) — those cells are excluded.

Each per-band excess TRACE is then baseline-robust-z'd (``robust_z_with_flags`` / ``baseline_robust_z``)
against the SAME baseline segment as the band cache (``resolve_baseline_window`` with
``GUARD_SEC`` / ``MIN_BASELINE_SEC``, ``eeg_onset_rel_sec`` = per-seizure EEG offset) — the excess
trace is robust-z'd DIRECTLY (not log'd: excess is a linear ≥0 quantity that is legitimately 0 in
quiet bins, so ``log`` would be undefined).

The reused ``analysis_channels`` sidecar (basis ``primary_bands_validity``) is copied from the raw
band cache sidecar EXACTLY as Task 10b did, so the alignment's fixed-mask + basis asserts hold. The
1/f fit params are not in the phase-1 config, so we use the Task-11 helper defaults
(``fit_lo=1``, ``fit_hi=200``, ``min_r2=0.5``).

Depends on the raw v2 band cache sidecar (Task 6) for ``analysis_channels`` (like Task 10b), and on
the SAME seizure-window loader as the band cache (``iter_subject_seizure_windows``) for the raw
signal it re-spectrograms. Output: results/.../v2_band_scan/aperiodic_resid_cache/{ds_sid}.npz
(+ .json = raw sidecar copy + aperiodic provenance); ``--outdir`` overrides the cache dir directly.

Plan: docs/superpowers/plans/2026-07-01-topic5-v2-phase1-band-scan-backbone.md Task 11b.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="invalid value encountered")

from scripts.build_topic5_ictal_field_long_cache import (  # noqa: E402
    iter_subject_seizure_windows, HOP, GUARD_SEC, MIN_BASELINE_SEC)
from scripts.run_topic5_ictal_field_dynamics import SUBJECTS_BY_SUB  # noqa: E402
from src import topic5_ictal_recruitment as recruit  # noqa: E402
from src.topic5_ictal_recruitment import _spectrogram_on_hop  # noqa: E402
from src.ictal_onset_extraction import resolve_baseline_window  # noqa: E402
from src.topic5_v2_band_scan import (  # noqa: E402
    load_phase1_config, line_noise_bin_mask, band_bin_selection,
    robust_z_with_flags, aperiodic_corrected_excess_power)

V2_ROOT = _ROOT / "results/topic5_ictal_recruitment/v2_band_scan"
RAW_CACHE_DIR = V2_ROOT / "cache"                        # source raw band cache sidecar (Task 6)
OUT_DIR_DEFAULT = V2_ROOT / "aperiodic_resid_cache"      # canonical; == FEATURE_CACHE_DIR["aperiodic_resid"]

# 1/f fit range + r2 gate: not in config -> Task-11 helper defaults (kept in sync with the helper).
FIT_LO, FIT_HI, MIN_R2 = 1.0, 200.0, 0.5


def _primary_band_specs(cfg):
    """The 7 PRIMARY (HALF-OPEN [lo,hi)) bands as (name, lo, hi), config order. Gate C targets
    ripple; only primaries are built (composites/legacy not needed for the aperiodic residual)."""
    return [(row[0], float(row[1]), float(row[2])) for row in cfg["bands"]["primary"]]


def _excess_traces(freqs, Sxx, line_mask, band_specs, fit_lo, fit_hi, min_r2, return_qc=False):
    """Vectorized per-(channel,time-bin) aperiodic-corrected band excess power (Gate C raw material).

    Mirrors ``src.topic5_v2_band_scan.aperiodic_corrected_excess_power`` EXACTLY (asserted cell-by-cell
    by ``test_aperiodic_vectorized_excess_matches_helper``), computed once over the whole
    ``(n_ch,n_time)`` grid instead of 7× per-cell: the log-log 1/f fit over ``[fit_lo,fit_hi]``
    (closed range, line-noise + non-positive/non-finite bins excluded) depends only on ``(c,tt)``, so
    it is fit ONCE per ``(c,tt)`` and reused for every band's excess extraction.

    Returns ``{name: (n_ch,n_time) float64}`` with NaN where the fit is not ``ok`` (n valid fit bins
    < 10 or ``fit_r2 < min_r2``) or the band has no valid bin (Task-11 contract). Columns holding a
    non-positive/non-finite PSD bin in the fit range (never for a clean PSD) fall back to the scalar
    helper per ``(c,tt)`` so the result stays byte-for-byte the helper's.

    With ``return_qc=True`` ALSO returns ``(out, fit_qc)`` — a purely OBSERVATIONAL 1/f-fit QC dict
    (W1 Task 1.1, Gate C input) read off the arrays this function already computes; the excess traces
    ``out`` are byte-for-byte identical whether or not QC is requested (the default ``return_qc=False``
    path is untouched). ``fit_qc`` counts every ``(channel,time-bin)`` as one attempted fit:
    ``fraction_failed_fits`` = fraction with ``r2<min_r2`` OR ``n_fit<10`` OR a non-finite/non-positive
    fit bin (all three collapse into ``~ok``); ``median_fit_r2`` is over the cells whose r2 is genuinely
    computed (``all_fit_ok``, i.e. excluding the scalar-helper patch cells whose vectorized r2 is a
    placeholder); ``n_valid_freq_bins`` = fit bins after line exclusion; ``line_noise_bins_excluded`` =
    fit-range bins dropped as mains harmonics (via ``band_bin_selection``)."""
    freqs = np.asarray(freqs, float)
    n_ch, _n_freq, n_time = Sxx.shape
    fin = np.isfinite(Sxx) & (Sxx > 0) & (freqs > 0)[None, :, None]     # helper finite_pos: isfinite & psd>0 & f>0

    fit_mask, _, n_fit_range_total = band_bin_selection(freqs, fit_lo, fit_hi, line_mask, half_open=False)  # CLOSED, like the helper
    n_fit = int(fit_mask.sum())
    # common case: every fit bin is positive-finite -> closed-form OLS over ALL (c,tt) at once.
    all_fit_ok = np.all(fin[:, fit_mask, :], axis=1)                    # (n_ch,n_time); False -> patch via helper
    xf = np.log10(freqs[fit_mask])                                     # (n_fit,) same for every (c,tt)
    Yf = np.log10(np.where(fin[:, fit_mask, :], Sxx[:, fit_mask, :], 1.0))  # placeholder 1.0 in to-be-patched cols
    xbar = float(xf.mean())
    xc = xf - xbar
    sxx = float((xc ** 2).sum())                                       # >0 (fit freqs distinct)
    ybar = Yf.mean(axis=1)                                             # (n_ch,n_time)
    sxy = np.tensordot(xc, Yf, axes=([0], [1]))                        # sum_f xc*Yf -> (n_ch,n_time)
    slope = sxy / sxx
    offset = ybar - slope * xbar
    # OLS identity ss_res = ss_tot - slope^2*sxx (ss_reg = slope^2*sum(xc^2)); avoids materializing the
    # (n_ch,n_fit,n_time) prediction array -> ~halves peak memory. ss_tot = sum(Yf^2) - n_fit*ybar^2.
    ss_tot = (Yf ** 2).sum(axis=1) - n_fit * ybar ** 2
    ss_reg = slope ** 2 * sxx
    with np.errstate(invalid="ignore", divide="ignore"):
        fit_r2 = np.where(ss_tot > 0, ss_reg / ss_tot, 1.0)          # = 1 - ss_res/ss_tot; helper: 1.0 when ss_tot==0
    ok = (fit_r2 >= min_r2) & (n_fit >= 10) & all_fit_ok               # all_fit_ok defers bad-bin cols to the patch

    out = {}
    for name, lo, hi in band_specs:
        band_mask, _, _ = band_bin_selection(freqs, lo, hi, line_mask, half_open=True)  # primaries are half-open
        excess = np.full((n_ch, n_time), np.nan, dtype=np.float64)
        if band_mask.any():
            logfb = np.log10(freqs[band_mask])
            psd_b = Sxx[:, band_mask, :]
            pred_b = 10.0 ** (slope[:, None, :] * logfb[None, :, None] + offset[:, None, :])
            # excluded (non-finite/non-positive) band bins contribute 0, exactly as the helper drops
            # them; below-floor finite bins already contribute 0 via max(psd-floor,0).
            contrib = np.where(fin[:, band_mask, :], np.maximum(psd_b - pred_b, 0.0), 0.0)
            summed = contrib.sum(axis=1)
            band_any = np.any(fin[:, band_mask, :], axis=1)           # helper: band_valid.any()
            valid = ok & band_any
            excess[valid] = summed[valid]
        out[name] = excess

    # exact-correctness patch: the (essentially-never) columns whose fit range holds a non-positive/
    # non-finite bin — the helper excludes that bin and refits the rest; the vectorized path can't, so
    # recompute those (c,tt) with the scalar helper so the cache is byte-for-byte the helper everywhere.
    need = ~all_fit_ok
    if need.any():
        for c, tt in np.argwhere(need):
            psd = Sxx[c, :, tt]
            for name, lo, hi in band_specs:
                out[name][c, tt] = aperiodic_corrected_excess_power(
                    freqs, psd, lo, hi, line_mask, fit_lo=fit_lo, fit_hi=fit_hi,
                    min_r2=min_r2, half_open=True)["excess_power"]
    if not return_qc:
        return out
    # OBSERVATIONAL 1/f-fit QC (W1 Task 1.1) off the SAME arrays; does not touch ``out``.
    n_total = int(ok.size)                                             # every (channel,time-bin) is one attempted fit
    n_failed = int((~ok).sum())                                        # ~ok = r2<min_r2 OR n_fit<10 OR bad fit bin
    median_r2 = float(np.median(fit_r2[all_fit_ok])) if bool(all_fit_ok.any()) else float("nan")
    fit_qc = {
        "median_fit_r2": median_r2,
        "fraction_failed_fits": (n_failed / n_total) if n_total else float("nan"),
        "n_failed_fits": n_failed, "n_total_fits": n_total,           # raw counts -> cohort-POOLED fraction
        "n_valid_freq_bins": int(n_fit),                              # fit bins after line exclusion
        "line_noise_bins_excluded": int(n_fit_range_total - n_fit),   # mains-harmonic bins dropped in the fit range
    }
    return out, fit_qc


def build_subject(ds_sid, substrate, band_specs, cfg, out_dir):
    """Build the aperiodic-corrected band-excess robust-z cache for one subject.

    Returns True iff an npz was written. Reuses the raw band-cache sidecar (Task 6) for the
    ``analysis_channels`` / ``primary_bands_validity`` fixed-mask contract (like Task 10b), and the
    SAME seizure-window loader as the band cache for the raw signal it re-spectrograms."""
    side_path = RAW_CACHE_DIR / f"{ds_sid}.json"
    if not side_path.exists():
        print(f"  [{ds_sid}] no raw band-cache sidecar in {RAW_CACHE_DIR} (run Task-6 band cache "
              f"first for analysis_channels), skip", flush=True)
        return False
    side = json.loads(side_path.read_text())

    ln = cfg["line_noise"]
    harmonics = list(ln["harmonics_hz"])
    halfwidth = float(ln["halfwidth_hz"])
    spec_win = float(cfg["power"]["spectrogram_win_sec"])
    spec_hop = float(cfg["power"]["spectrogram_hop_sec"])

    arrays = {}
    bands_qc = {}                 # {band: {str(idx): qc}}
    seizure_fit_qc = {}           # {str(idx): 1/f-fit QC (band-independent) + residual_variance_by_band}
    band_seizure_fail = []        # [(band, idx, reason)] robust-z degeneracies
    skipped_bands = {}            # {band: reason} Nyquist (subject-level)
    drops = []
    channels = None
    fs = None
    seizure_idxs = []

    for idx, sw, eeg_rel in iter_subject_seizure_windows(ds_sid, substrate, drops=drops):
        ch = [recruit.bipolar_alias_label(c) for c in sw.ch_names]
        if channels is None:
            channels, fs = ch, float(sw.fs)
        elif ch != channels:
            drops.append({"idx": idx, "reason": f"chan_mismatch:{len(ch)}vs{len(channels)}"})
            continue
        try:
            f, t, Sxx = _spectrogram_on_hop(sw.signal, sw.fs, spec_win, spec_hop)
        except ValueError as e:                        # win_sec needs more samples than we have
            drops.append({"idx": idx, "reason": f"spectrogram:{e}"})
            continue
        line_mask = line_noise_bin_mask(f, harmonics, halfwidth)
        nyq = float(sw.fs) / 2.0
        present = [(n, lo, hi) for (n, lo, hi) in band_specs if hi < nyq]   # Nyquist gate (half-open hi<nyq)
        for n, _lo, hi in band_specs:
            if hi >= nyq:
                skipped_bands.setdefault(n, f"band hi {hi} >= Nyquist {nyq} for fs={sw.fs}")
        if not present:
            drops.append({"idx": idx, "reason": "all_primary_bands_nyquist_skipped"})
            continue

        excess, fit_qc = _excess_traces(f, Sxx, line_mask, present, FIT_LO, FIT_HI, MIN_R2,
                                        return_qc=True)  # excess {name:(n_ch,n_time) f64}; fit_qc observational
        n_time = Sxx.shape[2]
        relt = (np.asarray(t, float) - float(sw.pre_sec)).astype(np.float32)
        # baseline segment depends on n_time + pre_sec + eeg_rel (NOT the band) -> resolve ONCE per
        # seizure; SAME convention as the band cache (resolve_baseline_window + robust_z_with_flags).
        bl = resolve_baseline_window(n_time, hop_sec=spec_hop, pre_sec=sw.pre_sec,
                                     buffer_sec=GUARD_SEC, eeg_onset_rel_sec=eeg_rel,
                                     min_baseline_valid_sec=MIN_BASELINE_SEC)
        wrote_any = False
        resid_var_by_band = {}                         # {band: nanvar of its aperiodic-corrected excess trace}
        for name, _lo, _hi in present:
            exc = excess[name]                         # (n_ch,n_time) f64; NaN on bad fit / no band bin
            try:                                       # baseline-robust-z the EXCESS trace directly (not log'd)
                z, low = robust_z_with_flags(exc, (bl.start_idx, bl.end_idx), spec_hop, MIN_BASELINE_SEC)
            except ValueError as e:                    # degenerate baseline window
                band_seizure_fail.append((name, idx, str(e)))
                continue
            arrays[f"{name}__zt__{idx}"] = z.astype(np.float32)
            arrays[f"{name}__relt__{idx}"] = relt
            n_cells = int(exc.size)
            bands_qc.setdefault(name, {})[str(idx)] = {
                "n_bins": int(n_time),
                "frac_cells_fit_ok": round(float(np.isfinite(exc).sum()) / max(n_cells, 1), 4),
                "n_low_baseline_channels": int(np.asarray(low, bool).sum()),
                "baseline_idx": [int(bl.start_idx), int(bl.end_idx)],
            }
            resid_var_by_band[name] = float(np.nanvar(exc))   # W1 Task 1.1 QC: variance of the excess trace
            wrote_any = True
        if wrote_any:
            seizure_idxs.append(idx)
            # per-seizure 1/f-fit QC (band-independent scalars) + this seizure's per-band excess variance
            seizure_fit_qc[str(idx)] = {**fit_qc, "residual_variance_by_band": resid_var_by_band}

    if channels is None or not arrays:
        print(f"  [{ds_sid}] nothing cached ({len(drops)} loader drops)", flush=True)
        return False

    # channel consistency: the sidecar we reuse must describe the same montage as this npz, or its
    # analysis_channels would index the wrong contacts downstream (AGENTS.md channel_names ordering).
    if list(side.get("channels", [])) != channels:
        print(f"  [{ds_sid}] SKIP: montage differs from raw band-cache sidecar "
              f"({len(channels)} vs {len(side.get('channels', []))}); refusing to reuse a stale mask", flush=True)
        return False

    # Reuse the raw sidecar so analysis_channels / basis / fixed-mask contract are IDENTICAL (Task 10b
    # pattern); add aperiodic provenance under a fresh key without clobbering the raw QC.
    meta = dict(side)
    meta["feature"] = "aperiodic_resid"
    meta["aperiodic"] = {
        "kind": "aperiodic_corrected_band_excess_power_robust_z",
        "fit_range_hz": [FIT_LO, FIT_HI], "min_r2": MIN_R2,
        "excess_note": "per (channel,time-bin): log-log 1/f OLS over [fit_lo,fit_hi] (closed, line/"
                       "non-positive excluded); excess = sum max(psd - 1/f floor, 0) over the HALF-OPEN "
                       "primary band; NaN when fit ok=False (n_fit<10 or r2<min_r2) or no valid band bin",
        "robust_z_note": "baseline_robust_z of the RAW excess trace (not log'd), SAME baseline segment "
                         "(resolve_baseline_window + GUARD_SEC/MIN_BASELINE_SEC) as the v2 band cache",
        "target_bands": "primary only (7 half-open); Gate C targets ripple",
        "source_raw_sidecar": str(side_path),
        "skipped_bands": skipped_bands,               # Nyquist (subject-level)
        "band_seizure_fail": [{"band": b, "idx": i, "reason": r} for b, i, r in band_seizure_fail],
        "bands": bands_qc,                            # per (band, idx) aperiodic QC
        "fit_qc": seizure_fit_qc,                      # per-seizure 1/f-fit QC (W1 Task 1.1, Gate C input)
        "drops": drops,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays["channels"] = np.array(channels)
    np.savez_compressed(out_dir / f"{ds_sid}.npz", **arrays)
    json.dump(meta, open(out_dir / f"{ds_sid}.json", "w"), indent=2, ensure_ascii=False)
    print(f"  [{ds_sid}] wrote {len(seizure_idxs)} sz x {len(bands_qc)} primary bands | "
          f"analysis_channels={len(side.get('analysis_channels', []))}/{len(channels)} | "
          f"skipped={list(skipped_bands)} | {len(drops)} drops", flush=True)
    return True


def _aggregate_cohort_qc(out_dir, threshold=0.2):
    """Roll up the per-subject 1/f-fit QC (each sidecar's ``aperiodic.fit_qc``) into a cohort
    ``aperiodic_qc.json`` (W1 Task 1.1, Gate C input).

    测了什么 / why: Gate C 判 ripple 的能量在 1/f 背景之外是否还有余量。这个判断只有当那条 1/f
    背景线本身拟合得好时才可信。这里把每个被试、每个发作、每个 (触点,时间点) 的拟合好坏汇总成一个
    队列数字：拟合失败的比例。若失败比例够低 (< threshold)，则 Gate C 的"没看到 ripple 特异余量"是
    一个真阴性；否则 Gate C 只能当描述性结论。

    Reads WHATEVER subject sidecars are present that carry a ``fit_qc`` block, so a partial /
    single-subject run yields a partial cohort file (sidecars predating the QC are skipped). The LOCK:
    cohort ``fraction_failed_fits`` POOLED over every (channel,time-bin) fit across all subjects/
    seizures (not a mean of per-subject fractions) < ``threshold`` -> ``aperiodic_trustworthy``."""
    per_subject = []
    band_pool = {}                                     # band -> {failed, total, r2:[per-(subj,sz) median]}
    tot_failed = tot_total = 0
    subj_medians = []
    for jp in sorted(Path(out_dir).glob("*.json")):
        if jp.name == "aperiodic_qc.json":
            continue                                   # never parse our own output as a subject sidecar
        try:
            meta = json.loads(jp.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        fit_qc = (meta.get("aperiodic") or {}).get("fit_qc") or {}
        if not fit_qc:
            continue                                   # sidecar predates the QC (or nothing cached)
        s_failed = s_total = 0
        s_r2 = []
        for _idx, sz in fit_qc.items():
            f_, t_ = int(sz.get("n_failed_fits", 0)), int(sz.get("n_total_fits", 0))
            s_failed += f_
            s_total += t_
            r2 = sz.get("median_fit_r2")
            r2 = float(r2) if (r2 is not None and np.isfinite(r2)) else None
            if r2 is not None:
                s_r2.append(r2)
            for band in (sz.get("residual_variance_by_band") or {}):   # a band's presence marks it built here
                bp = band_pool.setdefault(band, {"failed": 0, "total": 0, "r2": []})
                bp["failed"] += f_
                bp["total"] += t_
                if r2 is not None:
                    bp["r2"].append(r2)
        subj_frac = (s_failed / s_total) if s_total else float("nan")
        subj_med = float(np.median(s_r2)) if s_r2 else float("nan")
        tot_failed += s_failed
        tot_total += s_total
        if np.isfinite(subj_med):
            subj_medians.append(subj_med)
        per_subject.append({"subject": jp.stem, "n_seizures": len(fit_qc),
                            "median_fit_r2": subj_med, "fraction_failed_fits": subj_frac,
                            "n_failed_fits": s_failed, "n_total_fits": s_total})

    cohort_frac = (tot_failed / tot_total) if tot_total else float("nan")
    by_band = {b: {"median_fit_r2": (float(np.median(v["r2"])) if v["r2"] else float("nan")),
                   "fraction_failed_fits": (v["failed"] / v["total"]) if v["total"] else float("nan"),
                   "n_failed_fits": v["failed"], "n_total_fits": v["total"]}
               for b, v in sorted(band_pool.items())}
    cohort = {
        "aperiodic_trustworthy": bool(np.isfinite(cohort_frac) and cohort_frac < threshold),
        "fraction_failed_fits": cohort_frac,           # cohort, POOLED over all (channel,time-bin) fits
        "trustworthy_threshold": threshold,
        "median_fit_r2": float(np.median(subj_medians)) if subj_medians else float("nan"),
        "n_subjects": len(per_subject),
        "n_failed_fits": tot_failed, "n_total_fits": tot_total,
        "fit_range_hz": [FIT_LO, FIT_HI], "min_r2": MIN_R2,
        "by_band": by_band,
        "per_subject": per_subject,
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    json.dump(cohort, open(Path(out_dir) / "aperiodic_qc.json", "w"), indent=2, ensure_ascii=False)
    verdict = "TRUSTWORTHY" if cohort["aperiodic_trustworthy"] else "DESCRIPTIVE-ONLY"
    print(f"[v2-aperiodic-qc] {len(per_subject)} subj | cohort fraction_failed_fits={cohort_frac:.4f} "
          f"(thr {threshold}) -> aperiodic_trustworthy={cohort['aperiodic_trustworthy']} [{verdict}] | "
          f"median_fit_r2={cohort['median_fit_r2']:.4f} -> {Path(out_dir) / 'aperiodic_qc.json'}", flush=True)
    return cohort


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="explicit subject list (default = SUBJECTS_BY_SUB[substrate])")
    ap.add_argument("--substrate", choices=list(SUBJECTS_BY_SUB), default="broad",
                    help="default subject cohort; the aperiodic cache itself is substrate-independent")
    ap.add_argument("--outdir", default=None,
                    help="override the aperiodic-cache DIR directly (writes {outdir}/{ds_sid}.npz + .json); "
                         "default results/.../v2_band_scan/aperiodic_resid_cache")
    ap.add_argument("--restart", action="store_true", help="rebuild even if the aperiodic npz exists")
    args = ap.parse_args()

    cfg = load_phase1_config()
    band_specs = _primary_band_specs(cfg)
    out_dir = Path(args.outdir) if args.outdir else OUT_DIR_DEFAULT
    subs = args.subjects or SUBJECTS_BY_SUB[args.substrate]
    print(f"[v2-aperiodic] {len(subs)} subjects | 1/f-corrected excess over {len(band_specs)} primary "
          f"bands (fit [{FIT_LO},{FIT_HI}]Hz, min_r2={MIN_R2}) -> {out_dir}", flush=True)
    for ds_sid in subs:
        if (out_dir / f"{ds_sid}.npz").exists() and not args.restart:
            print(f"[cache] {ds_sid} exists, skip", flush=True)
            continue
        print(f"[cache] {ds_sid} ...", flush=True)
        try:
            build_subject(ds_sid, args.substrate, band_specs, cfg, out_dir)
        except Exception as e:
            print(f"  SUBJECT ERROR {type(e).__name__}: {e}", flush=True)
    _aggregate_cohort_qc(out_dir)                       # cohort 1/f-fit QC roll-up over whatever sidecars exist
    print("V2 APERIODIC CACHE DONE", flush=True)


if __name__ == "__main__":
    main()
