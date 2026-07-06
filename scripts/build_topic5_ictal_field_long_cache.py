"""Topic 5 — 长窗 ictal robust-z field cache (onset-130s .. offset+90s) for field-dynamics pilot.

复用 build_topic5_t0_feature_cache._features_one（同 baseline robust-z / band 口径），只把 post_sec
按每次发作自适应到 eeg offset+90s（现有 v2_windows 只到 +20s，不够到 offset）。写到 parallel dir，
不动现有 t0_feature_cache*。channels = bipolar_alias_label（与几何/axis record 同名约定）。

设计: docs/superpowers/specs/2026-06-28-topic5-ictal-field-dynamics-design.md §2（含 P1 修复）。
"""
from __future__ import annotations
import argparse, csv, json, math, sys, warnings
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="invalid value encountered")

import scripts.build_topic5_t0_feature_cache as cb
from scripts.run_topic5_t0_eligibility import (
    _inventory_rows, ICTAL_REFERENCE, HOP, GUARD_SEC, MIN_BASELINE_SEC)
from src.ictal_onset_extraction import extract_seizure_window

AUDIT = _ROOT / "results/topic5_ictal_recruitment/t0_eligibility_audit.csv"
OUT = _ROOT / "results/topic5_ictal_recruitment/ictal_field_long_cache"
SUBJECTS = ["epilepsiae_442", "epilepsiae_548", "epilepsiae_583",
            "epilepsiae_384", "epilepsiae_958", "epilepsiae_1084"]
POST_PAD = 90.0
MAX_ICTAL_SEC = 600.0   # span 上限（疑似 status；亦防 OOM）


def _eligible_complete(ds_sid, inv_rows):
    """analysis_eligible idx (audit) ∩ has_complete_eeg_interval (inventory)."""
    elig = set()
    for r in csv.DictReader(open(AUDIT)):
        if r["subject_id"] == ds_sid and str(r["analysis_eligible"]).strip().lower() in ("true", "1", "yes"):
            elig.add(int(r["seizure_idx"]))
    out = []
    for idx in sorted(elig):
        inv = inv_rows[idx] if idx < len(inv_rows) else {}
        if str(inv.get("has_complete_eeg_interval", "")).strip().lower() in ("true", "1", "yes", "t"):
            out.append(idx)
    return out


def _anchor_epoch(inv):
    """Onset anchor (epoch) for span/pre/post windowing. Clinical onset when present (epilepsiae);
    EEG-onset fallback when ``clin_onset_epoch`` is absent/blank (yuquan — its inventory has ONLY
    ``eeg_onset_epoch``, and ``extract_seizure_window`` already anchors yuquan windows on eeg_onset,
    so BOTH producer sites here — iter + build_subject — MUST mirror that same anchor or span/pre/post
    would disagree). epilepsiae keeps clin_onset -> ``float(raw)`` is byte-identical to the committed
    long cache (fallback never fires). Centralized (§5/§6.1) so the two sites can't drift apart."""
    raw = inv.get("clin_onset_epoch")
    return float(raw) if raw not in (None, "", "None") else float(inv["eeg_onset_epoch"])


def iter_subject_seizure_windows(ds_sid, substrate=None, drops=None):
    """Yield ``(idx, sw, eeg_rel)`` for each eligible + extractable seizure of ``ds_sid``.

    Factored verbatim out of :func:`build_subject` so the v2 multi-band cache can reuse the exact
    same seizure-window loop: same eligibility (``_eligible_complete``), same per-seizure
    ``post_sec`` (``ceil(span) + POST_PAD``), same ``span > MAX_ICTAL_SEC`` drop, and the same
    baseline-resolution inputs (``sw`` + ``eeg_rel``). Loader-side drops
    (``inv_field`` / ``duration_too_long_for_pilot`` / ``extract``) are appended to ``drops`` when a
    list is passed — lazy iteration keeps their interleaving with the consumer's own drops identical
    to the pre-refactor loop. ``substrate`` is accepted for signature uniformity with the band-cache
    consumer; the long ictal window is substrate-independent, so it never changes what is yielded.

    ``cb.PRE_FEATURE_SEC = 130.0`` is set here (pre floor, was set by ``build_subject`` pre-refactor)
    so ``_pre_target`` floors ``pre`` identically for every caller — the yielded ``sw.pre_sec``
    therefore matches the committed long cache whether the loop runs from ``build_subject`` or a
    standalone reuse (e.g. the v2 band cache).
    """
    cb.PRE_FEATURE_SEC = 130.0     # pre floor（与 v2_windows 一致）
    dataset, sid = ds_sid.split("_", 1)
    ref = ICTAL_REFERENCE[dataset]
    inv_rows, _ = _inventory_rows(dataset, sid)
    for idx in _eligible_complete(ds_sid, inv_rows):
        inv = inv_rows[idx]
        try:
            eeg_dur = float(inv["eeg_duration_sec"])
            clin_on = _anchor_epoch(inv)   # clin_onset (epilepsiae) or eeg_onset fallback (yuquan)
            eeg_off_rel = float(inv["eeg_offset_epoch"]) - clin_on
            eeg_on_rel = float(inv["eeg_onset_epoch"]) - clin_on  # parse guards inv_field drop (used by consumer)
        except (KeyError, TypeError, ValueError) as e:
            if drops is not None:
                drops.append({"idx": idx, "reason": f"inv_field:{type(e).__name__}"})
            continue
        span = max(eeg_off_rel, eeg_dur)   # P1: 覆盖 eeg offset，即使 eeg_onset 晚于 clin_onset(384 ~+36s)
        if span > MAX_ICTAL_SEC:
            if drops is not None:
                drops.append({"idx": idx, "reason": f"duration_too_long_for_pilot:{span:.0f}s"})
            continue
        pre = cb._pre_target(dataset, inv)
        post = math.ceil(span) + POST_PAD
        try:
            sw = extract_seizure_window(f"{dataset}/{sid}", idx, pre_sec=pre, post_sec=post, reference=ref)
        except Exception as e:
            if drops is not None:
                drops.append({"idx": idx, "reason": f"extract:{type(e).__name__}"})
            continue
        eeg_rel = (sw.eeg_onset_epoch - sw.clin_onset_epoch) if sw.eeg_onset_epoch is not None else None
        yield idx, sw, eeg_rel


def build_subject(ds_sid):
    cb.STORE_BB_ZT = True          # 存完整 broadband z trace
    dataset, sid = ds_sid.split("_", 1)
    inv_rows, _ = _inventory_rows(dataset, sid)   # re-read (read-only) for per-seizure provenance meta
    arrays, drops = {}, []
    meta = {"dataset": dataset, "subject": sid, "hop_sec": cb.HOP if hasattr(cb, "HOP") else 0.1,
            "channels": None, "fs": None, "eligible_idxs": [], "seizure": {}, "post_pad": POST_PAD,
            "baseline": {"guard_sec": 60.0,
                         "note": "robust-z baseline=[-pre_sec,-60] adaptive (resolve_baseline_window, "
                                 "eeg-rel clipped); per-seizure pre_sec in seizure[idx]"}}
    for idx, sw, eeg_rel in iter_subject_seizure_windows(ds_sid, drops=drops):
        inv = inv_rows[idx]
        clin_on = _anchor_epoch(inv)   # same anchor as iter_subject_seizure_windows (yuquan -> eeg_onset)
        eeg_dur = float(inv["eeg_duration_sec"])
        eeg_off_rel = float(inv["eeg_offset_epoch"]) - clin_on
        eeg_on_rel = float(inv["eeg_onset_epoch"]) - clin_on
        post = float(sw.post_sec)
        try:
            bb_auc, hfa_auc, ramp, hfa_zt, bact, bb_zt, bb_relt, hfa_relt = cb._features_one(sw, eeg_rel)
        except Exception as e:
            drops.append({"idx": idx, "reason": f"features:{type(e).__name__}"}); continue
        ch = [cb.recruit.bipolar_alias_label(c) for c in sw.ch_names]
        if meta["channels"] is None:
            meta["channels"] = ch; meta["fs"] = float(sw.fs)
        elif len(ch) != len(meta["channels"]):
            drops.append({"idx": idx, "reason": f"chan_count:{len(ch)}!={len(meta['channels'])}"}); continue
        arrays[f"bb_zt__{idx}"] = bb_zt; arrays[f"bb_relt__{idx}"] = bb_relt
        arrays[f"hfa_zt__{idx}"] = hfa_zt; arrays[f"hfa_relt__{idx}"] = hfa_relt
        arrays[f"bb_auc__{idx}"] = bb_auc.astype(np.float32)
        arrays[f"hfa_auc__{idx}"] = hfa_auc.astype(np.float32)
        meta["eligible_idxs"].append(idx)
        meta["seizure"][str(idx)] = {"seizure_id": sw.seizure_id, "pre_sec": float(sw.pre_sec),
                                     "post_sec": float(post), "eeg_onset_rel": eeg_on_rel,
                                     "eeg_offset_rel": eeg_off_rel, "eeg_duration_sec": eeg_dur}
        print(f"  [{ds_sid} sz{idx}] cached post={post:.0f}s dur={eeg_dur:.0f}s off_rel={eeg_off_rel:.0f}s",
              flush=True)
    meta["drops"] = drops
    if not meta["eligible_idxs"]:
        print(f"  [{ds_sid}] nothing cached ({len(drops)} drops)", flush=True); return
    arrays["channels"] = np.array(meta["channels"])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{ds_sid}.npz", **arrays)
    json.dump(meta, open(OUT / f"{ds_sid}.json", "w"), indent=2, ensure_ascii=False)
    print(f"  [{ds_sid}] wrote {len(meta['eligible_idxs'])} sz, {len(drops)} drops", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="*", default=SUBJECTS)
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()
    for ds_sid in args.subjects:
        if (OUT / f"{ds_sid}.npz").exists() and not args.restart:
            print(f"[cache] {ds_sid} exists, skip", flush=True); continue
        print(f"[cache] {ds_sid} ...", flush=True)
        try:
            build_subject(ds_sid)
        except Exception as e:
            print(f"  SUBJECT ERROR {type(e).__name__}: {e}", flush=True)
    print("LONG CACHE DONE", flush=True)


if __name__ == "__main__":
    main()
