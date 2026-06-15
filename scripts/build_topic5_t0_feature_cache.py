"""Topic 5 T0 — early-ictal feature cache for the analysis_eligible seizures.

Builds, per (subject, seizure) that passed the T0 eligibility audit, the per-contact
early-ictal activation features that A (primary) and B (secondary) consume:
  - broadband_auc_0_10s  : A-line PRIMARY activation = mean baseline-robust-z (1-45 Hz)
                           power over [0,10]s post-onset, per contact.
  - hfa_auc_0_10s        : same for 60-100 Hz (B / EI input).
  - ramp_slope_0_10s     : per-contact z-slope over [0,10]s (secondary).
  - hfa_zt               : the FULL [-pre,+post] HFA robust-z trace per contact (so B's EI /
                           CUSUM onset and every alignment window can be computed later
                           WITHOUT reloading the EDF).
v2 multi-window superset (--store-bb-zt / --post-sec / --pre-feature-window): also stores the
FULL [-pre,+post] broadband robust-z trace (bb_zt) and per-bin rel-to-onset time vectors
(bb_relt / hfa_relt) so all 6 alignment windows (post [0,5]/[5,10]/[0,10]/[0,20],
pre proximal [-10,0], pre distal [-120,-90] neg-control) are obtained by SLICING one
extraction instead of re-running the EDF 6x. The [0,10]s summaries are kept (v2 = v1 superset).
Channels are stored under the alias-left label (= the interictal template/axis convention),
so A joins the ictal field to the axis record BY NAME.

Reuses (no reinvented I/O): scripts.run_topic5_t0_eligibility (adaptive-pre + montage +
inventory), src.topic5_ictal_recruitment (band_power_trace / baseline_robust_z), and
src.topic5_t0_features (windowing + summaries, TDD'd). Resumable per subject; output is
audit-only data, no statistics. Run detached for the full cohort (I/O-bound, hours).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="invalid value encountered")

from src import topic5_ictal_recruitment as recruit
from src.ictal_onset_extraction import extract_seizure_window, resolve_baseline_window
from src.topic5_t0_features import activation_mean, ramp_slope, window_indices
from src.topic5_ei import ei_like
from scripts.run_topic5_t0_eligibility import (
    _eeg_rel_inv, _inventory_rows, ICTAL_REFERENCE, PRE_FLOOR, GUARD_SEC,
    TARGET_BASELINE_SEC, POST_SEC, HOP, BROAD_BAND, HFA_BAND, MIN_BASELINE_SEC)

AUDIT_CSV = Path("results/topic5_ictal_recruitment/t0_eligibility_audit.csv")
CACHE_DIR = Path("results/topic5_ictal_recruitment/t0_feature_cache")
T0_SEC, T1_SEC = 0.0, 10.0

# v2 multi-window cache knobs — set by main() from argparse (single source of truth,
# same global-swap pattern as run_topic5_t0_eligibility's path swap). Defaults below keep
# the legacy single-window behaviour if main() is bypassed. POST_SEC is imported above
# (default 15.0) and re-bound by --post-sec.
STORE_BB_ZT = False          # --store-bb-zt: store the FULL broadband z trace + relt
PRE_FEATURE_SEC = None       # --pre-feature-window: floor on pre so [-120,-90] is covered


def _pre_target(dataset, inv):
    """Same adaptive pre-window as the eligibility audit (single source of truth).
    With --pre-feature-window the pre is floored to PRE_FEATURE_SEC so the distal
    [-120,-90] negative-control window is fully inside the stored trace (the spectrogram
    first bin sits at win_sec/2 - pre, so a floor=120 only just reaches ~-119.5)."""
    pre = max(PRE_FLOOR, min(abs(_eeg_rel_inv(dataset, inv)), 300.0)
              + GUARD_SEC + TARGET_BASELINE_SEC)
    if PRE_FEATURE_SEC is not None:
        pre = max(pre, float(PRE_FEATURE_SEC))
    return pre


def _eligible_by_subject():
    rows = [r for r in csv.DictReader(open(AUDIT_CSV))
            if str(r["analysis_eligible"]).strip().lower() in ("true", "1", "yes")]
    d = defaultdict(list)
    for r in rows:
        d[r["subject_id"]].append(int(r["seizure_idx"]))
    return d


def _onset_window(times, pre_sec):
    rel = np.asarray(times, float) - float(pre_sec)
    return np.where((rel >= T0_SEC) & (rel <= T1_SEC))[0]


def _features_one(sw, eeg_rel):
    """Per-contact features for one seizure window.

    Returns the legacy [0,10]s summaries (bb_auc/hfa_auc/ramp/bact) PLUS the FULL-window
    robust-z traces (covering [-pre,+post]) and their per-bin rel-to-onset time vectors,
    so the v2 cache supports every alignment window by slicing (window_activation) without
    re-extracting the EDF. bb/hfa use different win_sec (1.0/0.5) -> different bin grids ->
    a separate relt per band."""
    bb, bt = recruit.band_power_trace(sw.signal, sw.fs, band=BROAD_BAND, win_sec=1.0, hop_sec=HOP)
    hf, ht = recruit.band_power_trace(sw.signal, sw.fs, band=HFA_BAND, win_sec=0.5, hop_sec=HOP)
    blb = resolve_baseline_window(bb.shape[1], hop_sec=HOP, pre_sec=sw.pre_sec, buffer_sec=GUARD_SEC,
                                  eeg_onset_rel_sec=eeg_rel, min_baseline_valid_sec=MIN_BASELINE_SEC)
    blh = resolve_baseline_window(hf.shape[1], hop_sec=HOP, pre_sec=sw.pre_sec, buffer_sec=GUARD_SEC,
                                  eeg_onset_rel_sec=eeg_rel, min_baseline_valid_sec=MIN_BASELINE_SEC)
    z_bb = recruit.baseline_robust_z(bb, (blb.start_idx, blb.end_idx), hop_sec=HOP,
                                     min_baseline_valid_sec=MIN_BASELINE_SEC)
    z_hf = recruit.baseline_robust_z(hf, (blh.start_idx, blh.end_idx), hop_sec=HOP,
                                     min_baseline_valid_sec=MIN_BASELINE_SEC)
    # per-bin time relative to clinical onset (= band_power_trace times - pre_sec); self-
    # describing so downstream window slicing doesn't need pre_sec/hop again.
    bb_relt = (np.asarray(bt, float) - float(sw.pre_sec)).astype(np.float32)
    hfa_relt = (np.asarray(ht, float) - float(sw.pre_sec)).astype(np.float32)
    win_bb = _onset_window(bt, sw.pre_sec)
    win_hf = _onset_window(ht, sw.pre_sec)
    bb_auc = activation_mean(z_bb, win_bb)
    hfa_auc = activation_mean(z_hf, win_hf)
    ramp = ramp_slope(z_bb, win_bb, hop_sec=HOP)
    # FULL-window traces (v2 superset). hfa_zt is now the WHOLE [-pre,+post] trace (per the
    # v2 contract). bb_zt is stored only when --store-bb-zt (it ~doubles npz size). The EI
    # post-process must re-slice [0,10] from hfa_zt via hfa_relt (see _add_ei_subject).
    hfa_zt = z_hf.astype(np.float32)
    bb_zt = z_bb.astype(np.float32) if STORE_BB_ZT else None
    # per-channel baseline activity = mean RAW broadband power over the baseline window
    # (the anchor for the anchor-matched null: 'is the ictal activation just tracking how
    # active the channel already is at baseline?').
    bact = (np.nanmean(bb[:, blb.start_idx:blb.end_idx], axis=1)
            if blb.end_idx > blb.start_idx else np.full(bb.shape[0], np.nan))
    return bb_auc, hfa_auc, ramp, hfa_zt, bact, bb_zt, bb_relt, hfa_relt


def _cache_subject(ds_sid, idxs):
    dataset, sid = ds_sid.split("_", 1)
    ref = ICTAL_REFERENCE[dataset]
    inv_rows, _ = _inventory_rows(dataset, sid)
    arrays = {}
    meta = {"dataset": dataset, "subject": sid, "hop_sec": HOP, "t_window": [T0_SEC, T1_SEC],
            "band_broad": list(BROAD_BAND), "band_hfa": list(HFA_BAND),
            "fs": None, "channels": None, "eligible_idxs": [],
            "primary_feature": "broadband_auc_0_10s (mean baseline-robust-z 1-45Hz over [0,10]s)",
            # v2 multi-window cache provenance:
            "post_sec": POST_SEC, "store_bb_zt": bool(STORE_BB_ZT),
            "pre_feature_sec": PRE_FEATURE_SEC,
            "full_trace_keys": "hfa_zt__/bb_zt__ are FULL [-pre,+post] robust-z traces; "
                               "bb_relt__/hfa_relt__ are per-bin rel-to-onset times "
                               "(= band_power_trace times - pre_sec); slice via "
                               "src.topic5_t0_features.window_activation. Legacy "
                               "bb_auc/hfa_auc/ramp/bact still [0,10]s (v2 = superset of v1)."}
    for idx in idxs:
        inv = inv_rows[idx] if idx < len(inv_rows) else {}
        pre = _pre_target(dataset, inv)
        try:
            sw = extract_seizure_window(f"{dataset}/{sid}", idx, pre_sec=pre, post_sec=POST_SEC,
                                        reference=ref)
        except Exception as e:
            print(f"  [{ds_sid} sz{idx}] load skip ({type(e).__name__}) — eligible CSV drift?", flush=True)
            continue
        eeg_rel = ((sw.eeg_onset_epoch - sw.clin_onset_epoch)
                   if sw.eeg_onset_epoch is not None else None)
        try:
            bb_auc, hfa_auc, ramp, hfa_zt, bact, bb_zt, bb_relt, hfa_relt = _features_one(sw, eeg_rel)
        except Exception as e:
            print(f"  [{ds_sid} sz{idx}] feature error {type(e).__name__}: {e}", flush=True)
            continue
        ch = [recruit.bipolar_alias_label(c) for c in sw.ch_names]
        if meta["channels"] is None:
            meta["channels"] = ch
            meta["fs"] = float(sw.fs)
        elif len(ch) != len(meta["channels"]):
            print(f"  [{ds_sid} sz{idx}] channel-count mismatch {len(ch)}!={len(meta['channels'])}, skip",
                  flush=True)
            continue
        arrays[f"bb_auc__{idx}"] = bb_auc.astype(np.float32)
        arrays[f"hfa_auc__{idx}"] = hfa_auc.astype(np.float32)
        arrays[f"ramp__{idx}"] = ramp.astype(np.float32)
        arrays[f"hfa_zt__{idx}"] = hfa_zt            # FULL [-pre,+post] trace (v2)
        arrays[f"hfa_relt__{idx}"] = hfa_relt
        arrays[f"bb_relt__{idx}"] = bb_relt
        if bb_zt is not None:
            arrays[f"bb_zt__{idx}"] = bb_zt          # FULL broadband trace (--store-bb-zt)
        arrays[f"bact__{idx}"] = bact.astype(np.float32)
        meta["eligible_idxs"].append(idx)
        print(f"  [{ds_sid} sz{idx}] cached: {int(np.isfinite(bb_auc).sum())}/{len(ch)} finite bb_auc",
              flush=True)
    if not meta["eligible_idxs"]:
        print(f"  [{ds_sid}] no seizures cached", flush=True)
        return
    arrays["channels"] = np.array(meta["channels"])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_DIR / f"{ds_sid}.npz", **arrays)
    json.dump(meta, open(CACHE_DIR / f"{ds_sid}.json", "w"), indent=2, ensure_ascii=False)
    print(f"  [{ds_sid}] wrote {len(meta['eligible_idxs'])} seizures + {len(meta['channels'])} channels",
          flush=True)


def _augment_subject(ds_sid):
    """Add per-channel baseline-activity (`bact__<idx>`) to an EXISTING npz without
    recomputing the other features. Reloads each seizure (broadband only) for the baseline
    mean. Idempotent + non-destructive (keeps bb_auc/hfa/etc.)."""
    npz_f, mj = CACHE_DIR / f"{ds_sid}.npz", CACHE_DIR / f"{ds_sid}.json"
    if not npz_f.exists():
        print(f"  [{ds_sid}] no npz, skip", flush=True)
        return
    existing = dict(np.load(npz_f, allow_pickle=True))
    if any(k.startswith("bact__") for k in existing):
        print(f"  [{ds_sid}] already has baseline activity, skip", flush=True)
        return
    meta = json.load(open(mj))
    dataset, sid = meta["dataset"], meta["subject"]
    ref = ICTAL_REFERENCE[dataset]
    inv_rows, _ = _inventory_rows(dataset, sid)
    n_ch = len(existing["channels"])
    added = 0
    for idx in meta["eligible_idxs"]:
        inv = inv_rows[idx] if idx < len(inv_rows) else {}
        try:
            sw = extract_seizure_window(f"{dataset}/{sid}", idx, pre_sec=_pre_target(dataset, inv),
                                        post_sec=POST_SEC, reference=ref)
            bb, _ = recruit.band_power_trace(sw.signal, sw.fs, band=BROAD_BAND, win_sec=1.0, hop_sec=HOP)
            eeg_rel = ((sw.eeg_onset_epoch - sw.clin_onset_epoch)
                       if sw.eeg_onset_epoch is not None else None)
            bl = resolve_baseline_window(bb.shape[1], hop_sec=HOP, pre_sec=sw.pre_sec, buffer_sec=GUARD_SEC,
                                         eeg_onset_rel_sec=eeg_rel, min_baseline_valid_sec=MIN_BASELINE_SEC)
            bact = (np.nanmean(bb[:, bl.start_idx:bl.end_idx], axis=1)
                    if bl.end_idx > bl.start_idx else np.full(bb.shape[0], np.nan)).astype(np.float32)
        except Exception as e:
            print(f"  [{ds_sid} sz{idx}] augment skip {type(e).__name__}", flush=True)
            continue
        if bact.shape[0] != n_ch:
            print(f"  [{ds_sid} sz{idx}] channel mismatch {bact.shape[0]}!={n_ch}, skip", flush=True)
            continue
        existing[f"bact__{idx}"] = bact
        added += 1
    np.savez_compressed(npz_f, **existing)
    print(f"  [{ds_sid}] augmented baseline activity for {added} seizures", flush=True)


def _add_ei_subject(ds_sid):
    """Add per-channel EI-like (`ei_like__<idx>`) to an existing npz from the cached HFA
    trace (`hfa_zt`) + `hfa_auc`. Pure cache post-process — NO EDF reload. Idempotent."""
    npz_f = CACHE_DIR / f"{ds_sid}.npz"
    if not npz_f.exists():
        return
    existing = dict(np.load(npz_f, allow_pickle=True))
    if any(k.startswith("ei_like__") for k in existing):
        print(f"  [{ds_sid}] already has ei_like, skip", flush=True)
        return
    meta = json.load(open(CACHE_DIR / f"{ds_sid}.json"))
    added = 0
    for idx in meta["eligible_idxs"]:
        zt_k, auc_k = f"hfa_zt__{idx}", f"hfa_auc__{idx}"
        if zt_k not in existing or auc_k not in existing:
            continue
        # ei_like / onset_delays count delay from frame 0 of the passed trace, so the trace
        # MUST start at onset. v2 hfa_zt is now the FULL [-pre,+post] trace -> re-slice the
        # [0,10]s onset window via hfa_relt before EI. v1 caches (no relt) stay as-is.
        zt = existing[zt_k]
        relt_k = f"hfa_relt__{idx}"
        if relt_k in existing:
            zt = zt[:, window_indices(existing[relt_k], T0_SEC, T1_SEC)]
        existing[f"ei_like__{idx}"] = ei_like(zt, existing[auc_k], hop_sec=HOP).astype(np.float32)
        added += 1
    np.savez_compressed(npz_f, **existing)
    print(f"  [{ds_sid}] added ei_like for {added} seizures", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--restart", action="store_true", help="rebuild even if npz exists")
    ap.add_argument("--augment-baseline", action="store_true",
                    help="add baseline-activity (anchor for anchor-matched null) to existing npz")
    ap.add_argument("--add-ei", action="store_true",
                    help="add EI-like from cached HFA trace to existing npz (no reload)")
    ap.add_argument("--out-dir", default="results/topic5_ictal_recruitment/t0_feature_cache_v2_windows",
                    help="cache output dir (default = the v2 multi-window dir; do NOT overwrite v1)")
    ap.add_argument("--post-sec", type=float, default=20.0,
                    help="post-onset extract length (default 20s so [0,20] axis window fits)")
    ap.add_argument("--store-bb-zt", action="store_true",
                    help="store the FULL broadband robust-z trace (bb_zt__/bb_relt__); ~doubles npz")
    ap.add_argument("--pre-feature-window", nargs="?", type=float, const=130.0, default=None,
                    help="floor the adaptive pre (s) so the distal [-120,-90] window is fully "
                         "inside the stored trace; bare flag = 130s (default adaptive pre=120 "
                         "only reaches ~-119.5)")
    args = ap.parse_args()

    # v2 multi-window knobs -> module globals (single source of truth for every helper,
    # same global-swap discipline as the masked-path swap in the propagation runners).
    global CACHE_DIR, POST_SEC, STORE_BB_ZT, PRE_FEATURE_SEC
    CACHE_DIR = Path(args.out_dir)
    POST_SEC = float(args.post_sec)
    STORE_BB_ZT = bool(args.store_bb_zt)
    PRE_FEATURE_SEC = args.pre_feature_window

    if args.add_ei:
        subs = sorted(p.stem for p in CACHE_DIR.glob("*.npz"))
        if args.subjects:
            subs = [s for s in subs if s in set(args.subjects)]
        print(f"[add-ei] {len(subs)} cached subjects", flush=True)
        for ds_sid in subs:
            try:
                _add_ei_subject(ds_sid)
            except Exception as e:
                print(f"  [{ds_sid}] ERROR {type(e).__name__}: {e}", flush=True)
        print("ADD-EI DONE", flush=True)
        return

    if args.augment_baseline:
        subs = sorted(p.stem for p in CACHE_DIR.glob("*.npz"))
        if args.subjects:
            subs = [s for s in subs if s in set(args.subjects)]
        print(f"[augment-baseline] {len(subs)} cached subjects", flush=True)
        for ds_sid in subs:
            print(f"[augment] {ds_sid} ...", flush=True)
            try:
                _augment_subject(ds_sid)
            except Exception as e:
                print(f"  SUBJECT ERROR {type(e).__name__}: {e}", flush=True)
        print("AUGMENT DONE", flush=True)
        return
    elig = _eligible_by_subject()
    subs = sorted(elig)
    if args.subjects:
        subs = [s for s in subs if s in set(args.subjects)]
    print(f"[t0-feature-cache] {len(subs)} subjects with eligible seizures "
          f"({sum(len(elig[s]) for s in subs)} seizures)", flush=True)
    for ds_sid in subs:
        if (CACHE_DIR / f"{ds_sid}.npz").exists() and not args.restart:
            print(f"[cache] {ds_sid} already cached, skip", flush=True)
            continue
        print(f"[cache] {ds_sid} ({len(elig[ds_sid])} eligible) ...", flush=True)
        try:
            _cache_subject(ds_sid, elig[ds_sid])
        except Exception as e:
            print(f"  SUBJECT ERROR {type(e).__name__}: {e}", flush=True)
    print("CACHE BUILD DONE", flush=True)


if __name__ == "__main__":
    main()
