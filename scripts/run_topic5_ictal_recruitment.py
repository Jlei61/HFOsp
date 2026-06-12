"""Topic 5 Stage-2 recruitment-time instrument runner (I/O + orchestration).

Spec: docs/superpowers/specs/2026-06-10-topic5-ictal-recruitment-stage2-design.md
Plan: docs/superpowers/plans/2026-06-10-topic5-ictal-recruitment-stage2.md
Staged gate: trace-montage -> audit -> sentinel (MANUAL) -> per-subject -> cohort.
This file implements Phase 2 (trace-montage / audit / sentinel). Phase 3 (per-subject /
cohort) is added later, only after the sentinel manual gate passes.

P0 invariants:
- §3.4 montage (PER-DATASET, traced 2026-06-10): ictal features use the dataset's
  detection reference — yuquan='bipolar' (alias-left), epilepsiae='car';
  assert_channel_identity hard-fails on a montage mismatch.
- §7.1 echo reuses src.topic5_echo_gate (no reinvented statistic) — Phase 3.
"""
from __future__ import annotations

import argparse
import csv
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

from src import topic5_ictal_recruitment as recruit
from src import topic5_echo_gate as echo
from src.ictal_onset_extraction import (extract_seizure_window, compute_er,
                                        resolve_baseline_window, GAMMA_ER_BANDS)
from src.propagation_skeleton_geometry import parse_shaft  # noqa: F401  (Phase 3)

MASKED_ROOT = Path("results/interictal_propagation_masked")
OUT_ROOT = Path("results/topic5_ictal_recruitment")
HOP = 0.1
MIN_CH = 8
RECRUIT_POST_SEC = 15.0
GLOBAL_ONSET_FRAC = 0.15
FPR_TARGET_PER_HOUR = 1.0
MIN_POOLED_BASELINE_SEC = 600.0
MIN_BASELINE_SEC = 60.0
# Phase-2 load cap: pooled-lambda only needs ~600s baseline (~3 seizures of ~240s each),
# and the audit/sentinel only need a representative sample. Cap EDF loads per subject so
# 26-seizure subjects don't take ~20 min. Phase 3 (per-subject echo) lifts this cap since
# each seizure is then a data point.
LAMBDA_MAX_SEIZURES = 4
BASELINE_PRE_SEC = 300.0
PRE_ONSET_CHANGE_SEC = -10.0
EARLY_K = 3
B = 2000
RNG_SEED = 20260610

# PER-DATASET montage (traced 2026-06-10; spec §3.4 v2.1).
TEMPLATE_MONTAGE = {"yuquan": "bipolar_aliased_left", "epilepsiae": "car"}
ICTAL_REFERENCE = {"yuquan": "bipolar", "epilepsiae": "car"}
NULL_D_MODE = {"yuquan": "region_matched", "epilepsiae": "mni_nn"}

FUSED_FEATURES = ("line_length", "broadband", "hfa", "spectral_edge")
AMP_FEATURES = ("line_length", "broadband", "hfa")
SPECTRAL_FEATURE = "spectral_edge"
FEATURE_WIN = {"line_length": 1.0, "broadband": 1.0, "hfa": 0.5,
               "spectral_edge": 1.0, "er": 1.0}


# ---------------------------------------------------------------------------
# helpers: dataset id, masked template, feature traces, z, onsets
# ---------------------------------------------------------------------------
def _ds_of(ds_sid: str) -> str:
    return ds_sid.split("_", 1)[0]


def _subj_of(ds_sid: str) -> str:
    return ds_sid.split("_", 1)[1]


def _load_masked_template(ds_sid):
    """Masked Main-A narrow template (Stage-1 contract). None if unusable."""
    mj = MASKED_ROOT / "rank_displacement" / "per_subject" / f"{ds_sid}.json"
    if not mj.exists():
        return None
    d = json.load(open(mj))
    if d.get("stable_k") != 2 or not d.get("pairs"):
        return None
    dataset = d["dataset"]
    ch = list(d["channel_names"])
    pair = d["pairs"][0]
    jv = np.asarray(pair["joint_valid"], dtype=bool)
    templates = [echo.masked_template_rank_1d(np.asarray(pair["rank_a_full"], float), jv),
                 echo.masked_template_rank_1d(np.asarray(pair["rank_b_full"], float), jv)]
    return {"channels": ch, "dataset": dataset, "templates": templates, "template_k": 2,
            "swap_class": pair.get("swap_sweep", {}).get("swap_class", "na"),
            "template_montage": TEMPLATE_MONTAGE[dataset]}


def _n_seizures(subj_path) -> int:
    """Count seizures the extractor can index (rows with an onset epoch)."""
    dataset = subj_path.split("/", 1)[0]
    sid = subj_path.split("/", 1)[1]
    for cand in (Path("results/dataset_inventory") / f"{dataset}_seizure_inventory.csv",
                 Path("results") / f"{dataset}_seizure_inventory.csv"):
        if cand.exists():
            onset_field = "clin_onset_epoch" if dataset == "epilepsiae" else "eeg_onset_epoch"
            with open(cand) as fh:
                return sum(1 for r in csv.DictReader(fh)
                           if r.get("subject") == sid and r.get(onset_field))
    return 0


def _feature_traces(signal, fs):
    """4 fused-feature traces + held-out ER trace. HFA -> (None,None) if fs too low."""
    traces = {}
    traces["line_length"] = recruit.line_length_trace(signal, fs, win_sec=1.0, hop_sec=HOP)
    traces["broadband"] = recruit.band_power_trace(signal, fs, band=(1.0, 45.0),
                                                   win_sec=1.0, hop_sec=HOP)
    try:
        traces["hfa"] = recruit.band_power_trace(signal, fs, band=(80.0, 150.0),
                                                 win_sec=0.5, hop_sec=HOP)
    except ValueError:
        traces["hfa"] = (None, None)
    traces["spectral_edge"] = recruit.spectral_edge_trace(signal, fs, edge=0.9,
                                                          win_sec=1.0, hop_sec=HOP)
    er = compute_er(signal, fs, GAMMA_ER_BANDS["fast"], GAMMA_ER_BANDS["slow"],
                    win_sec=1.0, hop_sec=HOP)
    traces["er"] = (er, np.arange(er.shape[1]) * HOP)
    return traces


def _raw_traces(signal, fs):
    """Cacheable unit: raw feature traces {feat: trace_2d} (HFA may be absent). No z."""
    tr = _feature_traces(signal, fs)
    return {k: (tr[k][0] if tr[k][0] is not None else None) for k in list(FUSED_FEATURES) + ["er"]}


def _z_from_traces(raw, pre_sec, eeg_onset_rel_sec, *, detrend="none"):
    """Raw traces -> (optional detrend) -> robust-z + baseline frames. Sweepable (cheap).
    Returns (z_by_feat, nfr_by_feat, baseline_frames_by_feat, available_features) or None."""
    avail = [k for k in FUSED_FEATURES if raw.get(k) is not None]
    if len(avail) < 2:
        return None
    z, nfr, base = {}, {}, {}
    for k in [c for c in avail + ["er"] if raw.get(c) is not None]:
        tr = recruit.detrend_trace(raw[k], mode=detrend, hop_sec=HOP, win_sec=30.0)
        nfr[k] = tr.shape[1]
        bl = resolve_baseline_window(nfr[k], hop_sec=HOP, pre_sec=pre_sec,
                                     eeg_onset_rel_sec=eeg_onset_rel_sec)
        if not bl.valid:
            return None
        z[k] = recruit.baseline_robust_z(tr, (bl.start_idx, bl.end_idx),
                                         hop_sec=HOP, min_baseline_valid_sec=MIN_BASELINE_SEC)
        base[k] = z[k][:, bl.start_idx:bl.end_idx]
    return z, nfr, base, avail


def _build_z(signal, fs, pre_sec, eeg_onset_rel_sec, *, detrend="none"):
    """Load path: signal -> raw traces -> z. Thin wrapper over _z_from_traces."""
    return _z_from_traces(_raw_traces(signal, fs), pre_sec, eeg_onset_rel_sec, detrend=detrend)


def _sec_to_frame(sec, *, pre_sec, win_sec, n_frames):
    frame = int(round((float(sec) + float(pre_sec) - float(win_sec) / 2.0) / HOP))
    return int(np.clip(frame, 0, n_frames))


def _onsets_in(z, nfr, avail_keys, lo_sec, hi_sec, lambdas, pre_sec, *, detector=("cusum", 0.5)):
    """Per-feature onset SECONDS in [lo_sec, hi_sec]; also per-(feat) no-onset count.
    detector = ('cusum', bias) -> Page-Hinkley change-point at per-feature lambda;
               ('zcross', z_cross) -> fixed-z sustained-crossing BACKUP comparator."""
    mode = detector[0]
    out, no_onset = {}, {}
    n_ch = z[avail_keys[0]].shape[0]
    for k in avail_keys:
        win = FEATURE_WIN[k]
        lo_f = _sec_to_frame(lo_sec, pre_sec=pre_sec, win_sec=win, n_frames=nfr[k])
        hi_f = _sec_to_frame(hi_sec, pre_sec=pre_sec, win_sec=win, n_frames=nfr[k])
        arr = np.full(n_ch, np.nan)
        miss = 0
        for c in range(z[k].shape[0]):
            if not np.isfinite(z[k][c]).any():
                continue
            if mode == "cusum":
                r = recruit.detect_contact_onset(z[k][c], lam=lambdas[k],
                                                 detection_idx_window=(lo_f, hi_f), hop_sec=HOP,
                                                 win_sec=win, pre_sec=pre_sec, bias=detector[1])
            else:  # zcross
                r = recruit.detect_contact_onset_zcross(z[k][c], z_cross=detector[1],
                                                        detection_idx_window=(lo_f, hi_f), hop_sec=HOP,
                                                        win_sec=win, pre_sec=pre_sec)
            if r["detected"]:
                arr[c] = r["onset_sec"]
            else:
                miss += 1
        out[k] = arr
        no_onset[k] = miss / max(1, z[k].shape[0])
    return out, no_onset


def _recruitment_from_z(z, nfr, channels, lambdas, pre_sec, *, detector=("cusum", 0.5)):
    """Two-pass recruitment from pre-built z. detector selects the per-contact onset
    method (see _onsets_in). For cusum a finite per-feature lambda is required; zcross
    needs no lambda. None if unresolved."""
    if detector[0] == "cusum":
        avail = [k for k in FUSED_FEATURES if k in z and np.isfinite(lambdas.get(k, np.nan))]
    else:
        avail = [k for k in FUSED_FEATURES if k in z]
    if len(avail) < 2:
        return None
    n_ch = z[avail[0]].shape[0]
    # PASS 1: provisional onset SECONDS over [-30,+30]s -> t_global
    p1, _ = _onsets_in(z, nfr, avail, -30.0, 30.0, lambdas, pre_sec, detector=detector)
    _, prov_onset = recruit.fuse_recruitment_rank(p1)
    g = recruit.resolve_global_onset(prov_onset, n_valid=n_ch, frac=GLOBAL_ONSET_FRAC)
    if not g["global_onset_resolved"]:
        return {"global_onset_resolved": False, "available_features": avail}
    t_global = g["t_global"]
    # PASS 2: per-contact onset SECONDS in recruitment band
    er_ok = "er" in z and (detector[0] == "zcross" or np.isfinite(lambdas.get("er", np.nan)))
    er_keys = avail + (["er"] if er_ok else [])
    p2, no_onset = _onsets_in(z, nfr, er_keys, t_global - 2.0, t_global + RECRUIT_POST_SEC,
                              lambdas, pre_sec, detector=detector)
    n_preonset = 0
    for k in avail:
        mask_pre = np.isfinite(p1[k]) & (p1[k] < PRE_ONSET_CHANGE_SEC)
        n_preonset += int(mask_pre.sum())
        p2[k][mask_pre] = np.nan
    fused_rank, fused_onset = recruit.fuse_recruitment_rank({k: p2[k] for k in avail})
    per_rank = {k: recruit._rank_of(p2[k]) for k in avail}
    have_amp = [k for k in AMP_FEATURES if k in per_rank]
    if len(have_amp) >= 2:
        ag = recruit.feature_agreement(
            {**per_rank, SPECTRAL_FEATURE: per_rank.get(SPECTRAL_FEATURE, np.full(n_ch, np.nan))},
            amplitude=tuple(have_amp), spectral=SPECTRAL_FEATURE, early_k=EARLY_K)
    else:
        ag = {"feature_agreement_flag": False}
    return {"global_onset_resolved": True, "t_global_sec": float(t_global),
            "recruitment_rank": fused_rank, "fused_onset": fused_onset,
            "er_onset": p2.get("er"), "per_feature_onset": p2,
            "n_recruited": int(np.isfinite(fused_rank).sum()),
            "n_preonset_change": n_preonset, "no_onset_rate": no_onset,
            "agreement": ag, "channels": list(channels), "available_features": avail}


def compute_seizure_recruitment(signal, fs, pre_sec, channels, lambdas, *,
                                eeg_onset_rel_sec=None):
    """One seizure -> recruitment (Phase 1 smoke + sentinel entry point). None if unusable."""
    built = _build_z(signal, fs, pre_sec, eeg_onset_rel_sec)
    if built is None:
        return None
    z, nfr, _base, _avail = built
    res = _recruitment_from_z(z, nfr, channels, lambdas, pre_sec)
    if res is None or not res.get("global_onset_resolved"):
        return None
    return res


# ---------------------------------------------------------------------------
# subject-level lambda calibration (pooled baseline across the subject's seizures)
# ---------------------------------------------------------------------------
def _subject_lambdas(subj_path, *, verbose=False):
    """Pool baseline z across the subject's seizures per feature -> calibrated lambda.
    Returns (lambdas dict, info dict). Loads each seizure EDF once."""
    dataset = subj_path.split("/", 1)[0]
    ref = ICTAL_REFERENCE[dataset]
    n_sz = _n_seizures(subj_path)
    pooled = {}                       # feat -> list of baseline-frame arrays (n_ch, n_bf)
    z_cache = []                      # list of (z, nfr, channels, eeg_rel) per loadable seizure
    for idx in range(n_sz):
        if len(z_cache) >= LAMBDA_MAX_SEIZURES:   # Phase-2 cap (enough for pooled lambda + sample)
            break
        try:
            sw = extract_seizure_window(subj_path, idx, pre_sec=BASELINE_PRE_SEC,
                                        post_sec=30.0, reference=ref)
        except Exception as e:
            if verbose:
                print(f"  [{subj_path} sz{idx}] load skip: {type(e).__name__}")
            continue
        eeg_rel = ((sw.eeg_onset_epoch - sw.clin_onset_epoch)
                   if sw.eeg_onset_epoch is not None else None)
        built = _build_z(sw.signal, sw.fs, sw.pre_sec, eeg_rel)
        if built is None:
            continue
        z, nfr, base, avail = built
        for k in list(avail) + ["er"]:
            pooled.setdefault(k, []).append(base[k])
        z_cache.append({"z": z, "nfr": nfr, "channels": list(sw.ch_names),
                        "eeg_rel": eeg_rel, "fs": sw.fs, "idx": idx})
        if verbose:
            print(f"  [{subj_path} sz{idx}] loaded ({len(z_cache)}/{LAMBDA_MAX_SEIZURES}) "
                  f"fs={sw.fs} avail={avail}", flush=True)
    lambdas, lam_info = {}, {}
    for k, frames in pooled.items():
        # concatenate baseline frames across seizures along time
        nch = max(f.shape[0] for f in frames)
        cat = np.concatenate([f for f in frames if f.shape[0] == nch], axis=1) \
            if len({f.shape[0] for f in frames}) == 1 else np.concatenate(frames, axis=1)
        out = recruit.calibrate_feature_lambda(cat, fpr_target_per_hour=FPR_TARGET_PER_HOUR,
                                               hop_sec=HOP, min_pooled_baseline_sec=MIN_POOLED_BASELINE_SEC)
        lambdas[k] = out["lambda"]
        lam_info[k] = out
    return lambdas, {"z_cache": z_cache, "lam_info": lam_info, "n_seizures_total": n_sz}


# ---------------------------------------------------------------------------
# Task 12 — B0 eligibility audit
# ---------------------------------------------------------------------------
def _iter_cohort_subjects():
    for mj in sorted((MASKED_ROOT / "rank_displacement" / "per_subject").glob("*.json")):
        yield mj.stem


def _audit_one_subject(ds_sid, *, verbose=False):
    tmpl = _load_masked_template(ds_sid)
    if tmpl is None:
        return None
    dataset = tmpl["dataset"]
    subj_path = f"{dataset}/{_subj_of(ds_sid)}"
    lambdas, info = _subject_lambdas(subj_path, verbose=verbose)
    z_cache = info["z_cache"]
    if not z_cache:
        return {"subject_id": ds_sid, "dataset": dataset, "n_seizures_total": info["n_seizures_total"],
                "n_seizures_loadable": 0, "n_seizures_eligible": 0, "MIN_CH_pass": False,
                "template_montage": tmpl["template_montage"], "ictal_montage": ICTAL_REFERENCE[dataset]}
    # montage / channel-identity contract (§3.4): aliased ictal names vs template names
    tmpl_set = set(tmpl["channels"])
    sample_ch = z_cache[0]["channels"]
    aliased = [recruit.bipolar_alias_label(c) for c in sample_ch]
    n_match = sum(1 for a in aliased if a in tmpl_set)
    recruit.assert_channel_identity(template_montage=tmpl["template_montage"],
                                    ictal_montage=ICTAL_REFERENCE[dataset])
    # per-seizure recruitment from cached z
    fs = z_cache[0]["fs"]
    n_recr, glob_ok, agree, preonset = [], 0, 0, 0
    for zc in z_cache:
        res = _recruitment_from_z(zc["z"], zc["nfr"], zc["channels"], lambdas, BASELINE_PRE_SEC)
        if res is None:
            continue
        if res.get("global_onset_resolved"):
            glob_ok += 1
            n_recr.append(res["n_recruited"])
            agree += int(res["agreement"].get("feature_agreement_flag", False))
            preonset += res.get("n_preonset_change", 0)
    n_recr = n_recr or [0]
    per_feat_avail = {k: bool(k in lambdas and np.isfinite(lambdas.get(k, np.nan)))
                      for k in FUSED_FEATURES}
    return {
        "subject_id": ds_sid, "dataset": dataset, "fs": fs,
        "n_seizures_total": info["n_seizures_total"], "n_seizures_loadable": len(z_cache),
        "n_seizures_eligible": glob_ok,
        "n_channels_template_narrow": len(tmpl["channels"]), "template_k": tmpl["template_k"],
        "swap_class": tmpl["swap_class"],
        "n_channels_recruited_min": int(np.min(n_recr)),
        "n_channels_recruited_median": float(np.median(n_recr)),
        "n_channels_recruited_max": int(np.max(n_recr)),
        "per_feature_available": json.dumps(per_feat_avail),
        "global_onset_resolved_fraction": round(glob_ok / max(1, len(z_cache)), 3),
        "feature_agreement_flag_fraction": round(agree / max(1, glob_ok), 3) if glob_ok else 0.0,
        "template_montage": tmpl["template_montage"], "ictal_montage": ICTAL_REFERENCE[dataset],
        "channel_identity_contract": ("matched_bipolar_aliased_left" if dataset == "yuquan"
                                      else "matched_car"),
        "n_channels_montage_matched": n_match,
        "calibration_unstable_per_feature": json.dumps(
            {k: bool(info["lam_info"].get(k, {}).get("calibration_unstable", True))
             for k in FUSED_FEATURES}),
        "pooled_baseline_sec": round(float(np.median(
            [v.get("pooled_baseline_sec", 0.0) for v in info["lam_info"].values()] or [0.0])), 1),
        "no_onset_rate_per_feature": "computed_per_seizure",
        "null_d_mode": NULL_D_MODE[dataset],
        "n_preonset_change_contacts": preonset,
        "MIN_CH_pass": bool(np.median(n_recr) >= MIN_CH),
    }


def cmd_audit(args):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    only = set(args.subjects) if args.subjects else None
    rows = []
    for ds_sid in _iter_cohort_subjects():
        if only and ds_sid not in only:
            continue
        print(f"[audit] {ds_sid} ...", flush=True)
        try:
            info = _audit_one_subject(ds_sid, verbose=args.verbose)
        except Exception as e:
            print(f"  ERROR {type(e).__name__}: {e}", flush=True)
            continue
        if info is not None:
            rows.append(info)
            print(f"  loadable={info.get('n_seizures_loadable')} eligible={info.get('n_seizures_eligible')} "
                  f"med_recr={info.get('n_channels_recruited_median')} montage_match={info.get('n_channels_montage_matched')}",
                  flush=True)
    cols = ["subject_id", "dataset", "fs", "n_seizures_total", "n_seizures_loadable",
            "n_seizures_eligible", "n_channels_template_narrow", "template_k", "swap_class",
            "n_channels_recruited_min", "n_channels_recruited_median", "n_channels_recruited_max",
            "per_feature_available", "global_onset_resolved_fraction",
            "feature_agreement_flag_fraction", "template_montage", "ictal_montage",
            "channel_identity_contract", "n_channels_montage_matched",
            "calibration_unstable_per_feature", "pooled_baseline_sec",
            "no_onset_rate_per_feature", "null_d_mode", "n_preonset_change_contacts", "MIN_CH_pass"]
    out_csv = OUT_ROOT / ("b0_recruitment_audit.csv" if only is None
                          else "b0_recruitment_audit_subset.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out_csv} ({len(rows)} subjects)")
    drop = sum(1 for r in rows if not r.get("MIN_CH_pass"))
    print(f"[MIN_CH={MIN_CH} locked] subjects failing median-recruited>=MIN_CH: {drop} (reported, not tuned)")


# ---------------------------------------------------------------------------
# Task 10 — trace-montage
# ---------------------------------------------------------------------------
def cmd_trace_montage(args):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rec = {"template_montage": TEMPLATE_MONTAGE, "ictal_reference": ICTAL_REFERENCE,
           "traced": "2026-06-10", "spec": "§3.4 v2.1 (per-dataset)",
           "evidence": "config/default.yaml bipolar+alias_left; subject_params /yuquan=bipolar "
                       "/epilepsiae=car; epilepsiae masked templates show consecutive CAR contacts"}
    json.dump(rec, open(OUT_ROOT / "montage_trace.json", "w"), indent=2, ensure_ascii=False)
    print("montage trace:", json.dumps(rec, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Task 13 — sentinel overlay (MANUAL GATE)
# ---------------------------------------------------------------------------
def cmd_sentinel(args):
    """Sentinel overlays for the manual gate. Groups seizures by subject so lambda is
    pooled-calibrated ONCE per subject (each calibration reloads <=cap EDFs)."""
    (OUT_ROOT / "sentinel").mkdir(parents=True, exist_ok=True)
    import importlib
    from collections import defaultdict
    plotter = importlib.import_module("scripts.plot_topic5_ictal_recruitment")
    by_subj = defaultdict(list)
    for spec in args.seizures:
        subj, sz = spec.rsplit(":", 1)
        by_subj[subj].append(int(sz))
    for subj, szs in by_subj.items():
        ds = subj.split("/", 1)[0]
        print(f"[sentinel] {subj} seizures={szs} (ref={ICTAL_REFERENCE[ds]}) — calibrating lambda ...",
              flush=True)
        lambdas, info = _subject_lambdas(subj, verbose=True)
        lam_print = {k: (round(lambdas[k], 2) if np.isfinite(lambdas.get(k, np.nan)) else None)
                     for k in FUSED_FEATURES}
        print(f"  lambdas={lam_print}", flush=True)
        for sz in szs:
            tag = f"{subj.replace('/', '_')}_{sz}"
            try:
                sw = extract_seizure_window(subj, sz, pre_sec=BASELINE_PRE_SEC, post_sec=30.0,
                                            reference=ICTAL_REFERENCE[ds])
            except Exception as e:
                print(f"  sz{sz} load FAILED: {type(e).__name__}: {e}", flush=True)
                continue
            eeg_rel = ((sw.eeg_onset_epoch - sw.clin_onset_epoch)
                       if sw.eeg_onset_epoch is not None else None)
            out = compute_seizure_recruitment(sw.signal, sw.fs, sw.pre_sec, sw.ch_names, lambdas,
                                              eeg_onset_rel_sec=eeg_rel)
            if out is None:
                print(f"  sz{sz}: UNRESOLVED (global onset not reached or baseline invalid)", flush=True)
                continue
            rec = {"seizure": f"{subj}:{sz}", "t_global_sec": out["t_global_sec"],
                   "n_recruited": out["n_recruited"], "n_preonset_change": out["n_preonset_change"],
                   "lambdas": lam_print,
                   "no_onset_rate": {k: round(float(v), 3) for k, v in out["no_onset_rate"].items()},
                   "agreement": {k: (round(float(v), 3) if isinstance(v, (int, float)) and not isinstance(v, bool)
                                     else bool(v) if isinstance(v, (bool, np.bool_)) else v)
                                 for k, v in out["agreement"].items()},
                   "available_features": out["available_features"]}
            json.dump(rec, open(OUT_ROOT / "sentinel" / f"{tag}.json", "w"), indent=2)
            plotter.plot_sentinel_overlay(sw, out, lambdas, BASELINE_PRE_SEC,
                                          OUT_ROOT / "sentinel" / f"{tag}.png")
            print(f"  sz{sz}: t_global={out['t_global_sec']:.1f}s n_recruited={out['n_recruited']} "
                  f"agree_flag={out['agreement'].get('feature_agreement_flag')} "
                  f"amp_agree={out['agreement'].get('amplitude_family_agreement')} "
                  f"spectral_support={out['agreement'].get('spectral_support')}", flush=True)
    print("\nSENTINEL DONE — human visual gate. Do NOT run per-subject/cohort until sign-off.")


# ---------------------------------------------------------------------------
# Detector Repair Stage — feature cache + bias/detrend/detector sweep (NO cohort)
# ---------------------------------------------------------------------------
CACHE_DIR = OUT_ROOT / "sentinel_cache"


def _cache_subject(subj, target_idxs):
    """Load subject's baseline-pool (<=cap) + target seizures ONCE; cache RAW feature
    traces + full context (fs/channels/pre_sec/eeg_rel/montage/baseline-pool idxs) so the
    detector sweep needs no EDF reloads. Raw traces (not the raw EEG window) are cached:
    they are the sweepable unit for detrend/robust-z/bias/zcross; raw-window caching is
    deferred (large) until a feature/window sweep is needed."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dataset = subj.split("/", 1)[0]
    ds_sid = f"{dataset}_{subj.split('/', 1)[1]}"
    ref = ICTAL_REFERENCE[dataset]
    tmpl = _load_masked_template(ds_sid)
    arrays = {}
    meta = {"dataset": dataset, "subject": subj, "ictal_reference": ref,
            "template_montage": tmpl["template_montage"] if tmpl else ICTAL_REFERENCE[dataset],
            "pre_sec": BASELINE_PRE_SEC, "hop_sec": HOP, "eeg_rel_by_idx": {},
            "loaded_idxs": [], "target_idxs": list(target_idxs)}
    loaded = []
    for idx in range(_n_seizures(subj)):
        is_target = idx in target_idxs
        if len(loaded) >= LAMBDA_MAX_SEIZURES and not is_target:
            continue
        try:
            sw = extract_seizure_window(subj, idx, pre_sec=BASELINE_PRE_SEC, post_sec=30.0,
                                        reference=ref)
        except Exception as e:
            print(f"  [{subj} sz{idx}] load skip: {type(e).__name__}", flush=True)
            continue
        raw = _raw_traces(sw.signal, sw.fs)
        if len([k for k in FUSED_FEATURES if raw.get(k) is not None]) < 2:
            continue
        for k in [c for c in list(FUSED_FEATURES) + ["er"] if raw.get(c) is not None]:
            arrays[f"tr__{k}__{idx}"] = raw[k].astype(np.float32)
        meta["eeg_rel_by_idx"][str(idx)] = (
            (sw.eeg_onset_epoch - sw.clin_onset_epoch) if sw.eeg_onset_epoch is not None else None)
        meta["fs"] = float(sw.fs)
        meta["channels"] = list(sw.ch_names)
        loaded.append(idx)
        print(f"  [{subj} sz{idx}] cached ({len(loaded)}) target={is_target} fs={sw.fs}", flush=True)
    meta["loaded_idxs"] = loaded
    meta["baseline_pool_idxs"] = loaded[:LAMBDA_MAX_SEIZURES]
    np.savez_compressed(CACHE_DIR / f"{ds_sid}.npz", **arrays)
    json.dump(meta, open(CACHE_DIR / f"{ds_sid}.json", "w"), indent=2)
    print(f"  cached {ds_sid}: loaded={loaded} pool={meta['baseline_pool_idxs']}", flush=True)


def cmd_cache(args):
    from collections import defaultdict
    by_subj = defaultdict(list)
    for spec in args.seizures:
        subj, sz = spec.rsplit(":", 1)
        by_subj[subj].append(int(sz))
    for subj, szs in by_subj.items():
        print(f"[cache] {subj} targets={szs}", flush=True)
        _cache_subject(subj, szs)
    print("CACHE DONE.")


def _load_cache(ds_sid):
    npz, mj = CACHE_DIR / f"{ds_sid}.npz", CACHE_DIR / f"{ds_sid}.json"
    if not npz.exists() or not mj.exists():
        return None
    data = np.load(npz)
    meta = json.load(open(mj))
    raw_by_idx = {idx: {k: (data[f"tr__{k}__{idx}"].astype(np.float64)
                            if f"tr__{k}__{idx}" in data.files else None)
                        for k in list(FUSED_FEATURES) + ["er"]}
                  for idx in meta["loaded_idxs"]}
    return raw_by_idx, meta


def cmd_sweep(args):
    """Detector ablation from cache (no EDF reload). Grid: detrend x detector, where
    detector in {cusum(bias), zcross(z_cross BACKUP)}. Writes detector_sweep.csv."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    from collections import defaultdict
    detectors = [("cusum", b) for b in args.bias_grid] + [("zcross", z) for z in args.zcross_grid]
    by_subj = defaultdict(list)
    for spec in args.seizures:
        subj, sz = spec.rsplit(":", 1)
        ds = subj.split("/", 1)[0]
        by_subj[f"{ds}_{subj.split('/', 1)[1]}"].append(int(sz))
    rows = []
    for ds_sid, targets in by_subj.items():
        cached = _load_cache(ds_sid)
        if cached is None:
            print(f"[sweep] {ds_sid}: NO CACHE (run `cache` first)", flush=True)
            continue
        raw_by_idx, meta = cached
        pre_sec, pool_idxs = meta["pre_sec"], meta["baseline_pool_idxs"]
        for detrend in args.detrend_grid:
            # z for every loaded seizure under this detrend (shared across detectors)
            zinfo = {}
            for idx in meta["loaded_idxs"]:
                built = _z_from_traces(raw_by_idx[idx], pre_sec,
                                       meta["eeg_rel_by_idx"].get(str(idx)), detrend=detrend)
                if built is not None:
                    z, nfr, base, avail = built
                    zinfo[idx] = {"z": z, "nfr": nfr, "base": base}
            # pooled baseline frames per feature (shared)
            pooled = defaultdict(list)
            for idx in pool_idxs:
                if idx in zinfo:
                    for k, arr in zinfo[idx]["base"].items():
                        pooled[k].append(arr)
            for det in detectors:
                if det[0] == "cusum":
                    lambdas = {}
                    for k, frames in pooled.items():
                        out = recruit.calibrate_feature_lambda(
                            np.concatenate(frames, axis=1), fpr_target_per_hour=FPR_TARGET_PER_HOUR,
                            hop_sec=HOP, min_pooled_baseline_sec=MIN_POOLED_BASELINE_SEC, bias=det[1])
                        lambdas[k] = out["lambda"]
                    lam_med = float(np.nanmedian([lambdas[k] for k in FUSED_FEATURES if k in lambdas]))
                    lam_sat = bool(lam_med >= 99.9)
                    det_label = f"cusum_b{det[1]}"
                else:
                    lambdas, lam_med, lam_sat, det_label = {}, float("nan"), False, f"zcross_z{det[1]}"
                for idx in targets:
                    base_row = {"subject": ds_sid, "seizure": idx, "detector": det_label,
                                "detrend": detrend, "lambda_med": round(lam_med, 2) if np.isfinite(lam_med) else "",
                                "lambda_saturated": lam_sat}
                    if idx not in zinfo:
                        rows.append({**base_row, "status": "no_z"}); continue
                    zi = zinfo[idx]
                    res = _recruitment_from_z(zi["z"], zi["nfr"], meta["channels"], lambdas,
                                              pre_sec, detector=det)
                    if res is None or not res.get("global_onset_resolved"):
                        rows.append({**base_row, "status": "unresolved"}); continue
                    ag = res["agreement"]
                    rows.append({**base_row, "status": "ok",
                                 "t_global_sec": round(res["t_global_sec"], 2),
                                 "n_recruited": res["n_recruited"],
                                 "feature_agreement_flag": bool(ag.get("feature_agreement_flag")),
                                 "amp_agree": round(float(ag.get("amplitude_family_agreement", float("nan"))), 3),
                                 "early_K_overlap": round(float(ag.get("early_K_overlap", float("nan"))), 3),
                                 "spectral_support": round(float(ag.get("spectral_support", float("nan"))), 3)})
                    print(f"  {ds_sid} sz{idx} {det_label} detrend={detrend}: lam={lam_med:.1f}"
                          f"{'(SAT)' if lam_sat else ''} agree={ag.get('feature_agreement_flag')} "
                          f"amp={ag.get('amplitude_family_agreement'):.2f} "
                          f"earlyK={ag.get('early_K_overlap'):.2f} nrec={res['n_recruited']}", flush=True)
    import csv as _csv
    cols = ["subject", "seizure", "detector", "detrend", "lambda_med", "lambda_saturated",
            "t_global_sec", "n_recruited", "feature_agreement_flag", "amp_agree",
            "early_K_overlap", "spectral_support", "status"]
    with open(OUT_ROOT / "detector_sweep.csv", "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {OUT_ROOT/'detector_sweep.csv'} ({len(rows)} rows)")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("trace-montage").set_defaults(func=cmd_trace_montage)
    pa = sub.add_parser("audit"); pa.add_argument("--subjects", nargs="*", default=None)
    pa.add_argument("--verbose", action="store_true"); pa.set_defaults(func=cmd_audit)
    ps = sub.add_parser("sentinel"); ps.add_argument("--seizures", nargs="+", required=True)
    ps.set_defaults(func=cmd_sentinel)
    pc = sub.add_parser("cache"); pc.add_argument("--seizures", nargs="+", required=True)
    pc.set_defaults(func=cmd_cache)
    pw = sub.add_parser("sweep"); pw.add_argument("--seizures", nargs="+", required=True)
    pw.add_argument("--bias-grid", nargs="*", type=float, default=[0.5, 1.0, 1.5, 2.0])
    pw.add_argument("--zcross-grid", nargs="*", type=float, default=[3.0, 4.0])
    pw.add_argument("--detrend-grid", nargs="*", default=["none", "rolling_median", "rolling_quantile"])
    pw.set_defaults(func=cmd_sweep)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
