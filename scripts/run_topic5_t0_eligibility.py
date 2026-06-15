"""Topic 5 A-line T0 — early-ictal eligibility audit (CSV + roll-up ONLY, no feature cache).

Plan: docs/archive/topic5/network_axis_pivot_plan_2026-06-13.md (§四 T0, §五 A-line).

What this answers (per seizure): can we build the 0–10 s ictal activation map AND align it
to the subject's interictal propagation axis? It does NOT run the (failed) first-onset
detector — it only loads the window, computes the broadband (+HFA) activation traces,
checks the clean baseline, and checks channel overlap with the interictal readout record.

Cohort = subjects that HAVE a usable interictal axis record (the only ones A can run on),
read from the topic3↔4 contact-plane readout records. Eligibility rules (frozen, see plan):
  - usable interictal record  : channels[] non-empty AND n_channels>=6 AND
                                flags.low_contact_count==false; status only excludes
                                'descriptive_only'/'error:' (NOT status=='ok').
  - baseline                  : [-90s,-60s] i.e. >=30s clean BEFORE the [-60s,0] guard
                                (resolve_baseline_window buffer_sec=60, min_valid=30).
  - A-line montage join       : denominator = readout RECORD channels; need >=80% resolve
                                in the ictal montage AND >= MIN_CH=6 matched.
  - boundary                  : has_complete_eeg_interval AND gap-to-prev-seizure >= 300s.

Reuses (no reinvented I/O): src.ictal_onset_extraction.{extract_seizure_window,
resolve_baseline_window}; src.topic5_ictal_recruitment.{band_power_trace,
bipolar_alias_label}. Per-dataset montage: yuquan=bipolar, epilepsiae=car.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

from src.ictal_onset_extraction import extract_seizure_window, resolve_baseline_window
from src.topic5_ictal_recruitment import band_power_trace, bipolar_alias_label

GEOM_ROOT = Path("results/spatial_modulation/propagation_geometry/observation_readout/real_subjects")
OUT_ROOT = Path("results/topic5_ictal_recruitment")
ICTAL_REFERENCE = {"yuquan": "bipolar", "epilepsiae": "car"}

HOP = 0.1
# Adaptive pre-window, sized PER SEIZURE to its eeg_rel so the eeg-onset-aware baseline is
# always ~60s ending at the [-60s,0] guard:  pre = max(PRE_FLOOR, |eeg_rel| + GUARD + TARGET).
# A fixed pre_sec is wrong both ways: 300 over-rejects near-block seizures (and is slow);
# 120 squeezes the baseline below 30s for early-EEG-onset seizures (observed epilepsiae_1077
# drop 9->6). eeg_rel = eeg_onset - clin_onset (<=0 when EEG leads), read from the inventory.
PRE_FLOOR = 120.0
TARGET_BASELINE_SEC = 60.0
POST_SEC = 15.0
GUARD_SEC = 60.0           # [-60s,0] peri-onset guard; baseline must end at -60s
MIN_BASELINE_SEC = 30.0    # >=30s clean baseline before the guard ([-90,-60] primary)
MIN_CH = 6                 # geometry MIN_CONTACTS (NOT Stage-2's 8)
MONTAGE_FRAC = 0.80        # >=80% of readout-record channels must resolve in ictal montage
MIN_GAP_PREV_SEC = 300.0   # baseline window length; previous seizure must be >= this away
BROAD_BAND = (1.0, 45.0)   # A-line primary activation (broadband_auc_0_10s)
HFA_BAND = (60.0, 100.0)   # B-line / EI secondary (fast activity)

COLS = ["subject_id", "dataset", "seizure_idx", "seizure_id", "n_axis_channels",
        "attempted", "load_ok", "fs", "n_channels_ictal", "n_montage_resolved",
        "montage_frac", "montage_ok", "broadband_ok", "hfa_ok", "baseline_valid",
        "has_complete_eeg_interval", "gap_to_prev_sec", "gap_prev_ok", "day_night",
        "cacheable", "analysis_eligible", "b_eligible", "reason"]


def _usable_record_channels(path: Path):
    """Return [channel names] if the readout record is usable (schema rule), else None."""
    try:
        d = json.load(open(path))
    except Exception:
        return None
    status = d.get("status")
    if status is not None and (str(status).startswith("error") or status == "descriptive_only"):
        return None
    chans = d.get("channels") or []
    if not chans:
        return None
    n_ch = d.get("n_channels", len(chans))
    if n_ch < MIN_CH:
        return None
    if (d.get("flags") or {}).get("low_contact_count"):
        return None
    return [c["name"] for c in chans]


def _cohort(only=None):
    """Yield (ds_sid, axis_channels) for subjects with a usable PRIMARY-template (t_a) record."""
    for p in sorted(GEOM_ROOT.glob("*_t_a.json")):
        ds_sid = p.name[: -len("_t_a.json")]
        if only and ds_sid not in only:
            continue
        ch = _usable_record_channels(p)
        if ch:
            yield ds_sid, ch


def _inventory_rows(dataset: str, sid: str):
    """Ordered inventory rows for the subject (same filter/order as the extractor's index)."""
    onset_field = "clin_onset_epoch" if dataset == "epilepsiae" else "eeg_onset_epoch"
    for cand in (Path("results/dataset_inventory") / f"{dataset}_seizure_inventory.csv",
                 Path("results") / f"{dataset}_seizure_inventory.csv"):
        if cand.exists():
            with open(cand) as fh:
                rows = [r for r in csv.DictReader(fh)
                        if r.get("subject") == sid and r.get(onset_field)]
            return rows, onset_field
    return [], onset_field


def _full_inventory_n():
    """Total seizures across BOTH dataset inventories (the unfiltered denominator).
    This audit only ATTEMPTS the subset that has a usable interictal axis record, so the
    summary reports this full count separately to avoid a misleading 'X/594' framing."""
    out = {}
    for ds in ("epilepsiae", "yuquan"):
        of = "clin_onset_epoch" if ds == "epilepsiae" else "eeg_onset_epoch"
        for cand in (Path("results/dataset_inventory") / f"{ds}_seizure_inventory.csv",
                     Path("results") / f"{ds}_seizure_inventory.csv"):
            if cand.exists():
                with open(cand) as fh:
                    out[ds] = sum(1 for r in csv.DictReader(fh) if r.get(of))
                break
    return out


def _eeg_rel_inv(dataset, inv):
    """eeg_onset - clin_onset from the inventory (<=0 when EEG leads); 0 for yuquan
    (eeg-onset-referenced, so the loader's t=0 already IS the eeg onset)."""
    if dataset == "epilepsiae":
        try:
            return float(inv.get("eeg_onset_epoch")) - float(inv.get("clin_onset_epoch"))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _audit_subject(ds_sid, axis_channels, *, limit=None, verbose=False):
    dataset = ds_sid.split("_", 1)[0]
    sid = ds_sid.split("_", 1)[1]
    subj_path = f"{dataset}/{sid}"
    ref = ICTAL_REFERENCE[dataset]
    inv_rows, onset_field = _inventory_rows(dataset, sid)
    axis_set = set(axis_channels)
    n_axis = len(axis_channels)
    out = []
    prev_onset = None
    for idx, inv in enumerate(inv_rows):
        if limit is not None and idx >= limit:
            break
        row = {c: "" for c in COLS}
        row.update(subject_id=ds_sid, dataset=dataset, seizure_idx=idx,
                   seizure_id=inv.get("seizure_id", ""), n_axis_channels=n_axis,
                   attempted=True, load_ok=False, montage_ok=False, broadband_ok=False,
                   hfa_ok=False, baseline_valid=False, cacheable=False,
                   analysis_eligible=False, b_eligible=False)
        complete = str(inv.get("has_complete_eeg_interval", "")).strip().lower() in ("true", "1", "yes", "t")
        row["has_complete_eeg_interval"] = complete
        row["day_night"] = inv.get("eeg_onset_day_night", "")
        # gap to previous seizure
        if dataset == "epilepsiae":
            gp = inv.get("seizure_interval_from_prev_sec", "")
            try:
                gap = float(gp) if gp not in ("", None) else None
            except ValueError:
                gap = None
        else:
            try:
                onset = float(inv.get(onset_field))
                gap = (onset - prev_onset) if prev_onset is not None else None
                prev_onset = onset
            except (TypeError, ValueError):
                gap = None
        gap_prev_ok = (gap is None) or (gap >= MIN_GAP_PREV_SEC)
        row["gap_to_prev_sec"] = round(gap, 1) if gap is not None else ""
        row["gap_prev_ok"] = gap_prev_ok
        # load window (per-dataset montage); pre sized to this seizure's eeg_rel so the
        # eeg-onset-aware baseline is ~60s and never squeezed by an early EEG onset
        # cap |eeg_rel| at 300s: a real EEG-onset lead is never >5min; larger = a bogus
        # timestamp (epilepsiae_139 sz0 had ~90h) and must not trigger a giant load.
        pre_target = max(PRE_FLOOR, min(abs(_eeg_rel_inv(dataset, inv)), 300.0)
                         + GUARD_SEC + TARGET_BASELINE_SEC)
        try:
            sw = extract_seizure_window(subj_path, idx, pre_sec=pre_target, post_sec=POST_SEC,
                                        reference=ref)
        except Exception as e:
            msg = str(e)
            if "before block_start" in msg or "before record" in msg:
                row["reason"] = "pre_window_before_block"   # insufficient baseline (legit drop)
            elif "after block_end" in msg or "after record" in msg:
                row["reason"] = "post_window_after_block"    # seizure at block end / bogus onset
            else:
                row["reason"] = f"load_error:{type(e).__name__}"
            out.append(row)
            if verbose:
                print(f"    sz{idx}: {row['reason']}", flush=True)
            continue
        row["load_ok"] = True
        row["fs"] = float(sw.fs)
        row["n_channels_ictal"] = len(sw.ch_names)
        eeg_rel = ((sw.eeg_onset_epoch - sw.clin_onset_epoch)
                   if sw.eeg_onset_epoch is not None else None)
        # broadband (A primary) + HFA (B/EI secondary)
        nfr = 0
        try:
            bb, _ = band_power_trace(sw.signal, sw.fs, band=BROAD_BAND, win_sec=1.0, hop_sec=HOP)
            broadband_ok = bb is not None and bb.shape[1] > 0
            nfr = bb.shape[1] if broadband_ok else 0
        except Exception:
            broadband_ok = False
        row["broadband_ok"] = broadband_ok
        # HFA availability is purely a Nyquist question (band_power_trace raises iff the band
        # exceeds Nyquist); derive it from fs to avoid a second spectrogram per seizure.
        row["hfa_ok"] = bool(sw.fs / 2.0 > HFA_BAND[1])
        # clean baseline before the [-60s,0] guard
        if broadband_ok:
            bl = resolve_baseline_window(nfr, hop_sec=HOP, pre_sec=sw.pre_sec,
                                         buffer_sec=GUARD_SEC, eeg_onset_rel_sec=eeg_rel,
                                         min_baseline_valid_sec=MIN_BASELINE_SEC)
            row["baseline_valid"] = bool(bl.valid)
        # A-line montage join: >=80% of RECORD channels resolve in ictal montage, >=MIN_CH matched
        aliased = set(bipolar_alias_label(c) for c in sw.ch_names)
        n_res = sum(1 for ac in axis_channels if ac in aliased)
        frac = n_res / max(1, n_axis)
        montage_ok = (n_res >= MIN_CH) and (frac >= MONTAGE_FRAC)
        row["n_montage_resolved"] = n_res
        row["montage_frac"] = round(frac, 3)
        row["montage_ok"] = montage_ok
        # gates
        cacheable = bool(broadband_ok)
        analysis_eligible = bool(cacheable and row["baseline_valid"] and montage_ok
                                 and complete and gap_prev_ok)
        row["cacheable"] = cacheable
        row["analysis_eligible"] = analysis_eligible
        row["b_eligible"] = bool(analysis_eligible and row["hfa_ok"])
        if not broadband_ok:
            row["reason"] = "broadband_fail"
        elif not complete:
            row["reason"] = "incomplete_eeg_interval"
        elif not gap_prev_ok:
            row["reason"] = f"baseline_gap<{int(MIN_GAP_PREV_SEC)}s"
        elif not row["baseline_valid"]:
            row["reason"] = "baseline_invalid"
        elif not montage_ok:
            row["reason"] = f"montage<80%_or<{MIN_CH}(res={n_res}/{n_axis})"
        else:
            row["reason"] = "ok"
        out.append(row)
        if verbose:
            print(f"    sz{idx}: {row['reason']} fs={row['fs']} montage={n_res}/{n_axis} "
                  f"elig={analysis_eligible}", flush=True)
    return out


def _as_bool(v):
    return v is True or str(v).strip().lower() in ("true", "1", "yes")


def _summary(rows, cohort_n=None):
    by_ds = {}
    for ds in sorted({r["dataset"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == ds]
        by_ds[ds] = {
            "n_seizures_attempted": len(sub),
            "n_cacheable": sum(1 for r in sub if _as_bool(r["cacheable"])),
            "n_analysis_eligible": sum(1 for r in sub if _as_bool(r["analysis_eligible"])),
            "n_b_eligible": sum(1 for r in sub if _as_bool(r["b_eligible"])),
            "n_subjects": len({r["subject_id"] for r in sub}),
            "n_subjects_ge1_eligible": len({r["subject_id"] for r in sub if _as_bool(r["analysis_eligible"])}),
        }
    reasons = Counter(r["reason"] for r in rows if not _as_bool(r["analysis_eligible"]))
    inv = _full_inventory_n()
    return {
        "audit_scope": ("axis-linked ictal eligibility — seizures of subjects that HAVE a "
                        "usable interictal axis record; this is NOT a full-inventory audit"),
        "full_inventory_n": sum(inv.values()),
        "full_inventory_by_dataset": inv,
        "axis_linked_attempted_n": len(rows),
        "cohort_n_subjects": cohort_n,
        "n_subjects_with_seizures": len({r["subject_id"] for r in rows}),
        "n_seizures_attempted": len(rows),
        "n_cacheable": sum(1 for r in rows if _as_bool(r["cacheable"])),
        "n_analysis_eligible": sum(1 for r in rows if _as_bool(r["analysis_eligible"])),
        "n_b_eligible": sum(1 for r in rows if _as_bool(r["b_eligible"])),
        "n_subjects": len({r["subject_id"] for r in rows}),
        "n_subjects_ge1_eligible": len({r["subject_id"] for r in rows if _as_bool(r["analysis_eligible"])}),
        "by_dataset": by_ds,
        "ineligible_reasons": dict(reasons.most_common()),
        "frozen_params": {"baseline": "[-90,-60] (>=30s clean, guard=[-60,0])",
                          "montage_frac": MONTAGE_FRAC, "min_ch": MIN_CH,
                          "min_gap_prev_sec": MIN_GAP_PREV_SEC,
                          "broad_band": BROAD_BAND, "hfa_band": HFA_BAND},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None, help="restrict to these ds_sid")
    ap.add_argument("--limit-seizures", type=int, default=None, help="smoke: cap seizures/subject")
    ap.add_argument("--out", default=str(OUT_ROOT / "t0_eligibility_audit.csv"))
    ap.add_argument("--restart", action="store_true", help="ignore existing CSV, redo from scratch")
    ap.add_argument("--resummarize", action="store_true",
                    help="regenerate the summary JSON from the existing CSV (no audit re-run)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    only = set(args.subjects) if args.subjects else None
    cohort = list(_cohort(only))
    out_csv = Path(args.out)

    if args.resummarize:
        rows = list(csv.DictReader(open(out_csv)))
        summ = _summary(rows, cohort_n=len(cohort))
        out_json = out_csv.with_name(out_csv.stem.replace("_audit", "") + "_summary.json")
        json.dump(summ, open(out_json, "w"), indent=2, ensure_ascii=False)
        print(f"resummarized {out_json} from {len(rows)} rows: "
              f"full_inventory={summ['full_inventory_n']} attempted={summ['axis_linked_attempted_n']} "
              f"eligible={summ['n_analysis_eligible']} subj_with_sz={summ['n_subjects_with_seizures']}/{summ['cohort_n_subjects']}")
        return

    # Resume: skip subjects already in the CSV; append per-subject so a kill never restarts
    # from zero (the full run is I/O-bound and gets killed mid-way).
    done = set()
    if out_csv.exists() and not args.restart:
        with open(out_csv) as fh:
            done = {r["subject_id"] for r in csv.DictReader(fh)}
    fresh = args.restart or not out_csv.exists()
    print(f"[t0-eligibility] cohort = {len(cohort)} subjects with a usable interictal axis "
          f"record; {len(done)} already done (resume), {len(cohort) - len(done)} to go", flush=True)

    fh = open(out_csv, "w" if fresh else "a", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
    if fresh:
        w.writeheader()
        fh.flush()
    for ds_sid, axis_ch in cohort:
        if ds_sid in done:
            print(f"[t0] {ds_sid} — already done, skip", flush=True)
            continue
        print(f"[t0] {ds_sid} (axis_channels={len(axis_ch)}) ...", flush=True)
        try:
            sub_rows = _audit_subject(ds_sid, axis_ch, limit=args.limit_seizures,
                                      verbose=args.verbose)
        except Exception as e:
            print(f"  SUBJECT ERROR {type(e).__name__}: {e}", flush=True)
            continue
        for r in sub_rows:
            w.writerow(r)
        fh.flush()                                    # persist this subject before the next load
        n_el = sum(1 for r in sub_rows if r["analysis_eligible"])
        print(f"  seizures={len(sub_rows)} analysis_eligible={n_el}", flush=True)
    fh.close()

    rows = list(csv.DictReader(open(out_csv)))        # re-read full CSV (incl. prior resume rows)
    summ = _summary(rows, cohort_n=len(cohort))
    out_json = out_csv.with_name(out_csv.stem.replace("_audit", "") + "_summary.json")
    json.dump(summ, open(out_json, "w"), indent=2, ensure_ascii=False)

    print(f"\nwrote {out_csv} ({len(rows)} seizure rows)")
    print(f"wrote {out_json}")
    print(f"\n=== ROLL-UP ===")
    print(f"attempted        : {summ['n_seizures_attempted']}")
    print(f"cacheable        : {summ['n_cacheable']}")
    print(f"analysis_eligible: {summ['n_analysis_eligible']}  "
          f"(B-eligible w/ HFA: {summ['n_b_eligible']})")
    print(f"subjects         : {summ['n_subjects']}  (>=1 eligible seizure: {summ['n_subjects_ge1_eligible']})")
    for ds, s in summ["by_dataset"].items():
        print(f"  {ds:11s}: attempted={s['n_seizures_attempted']} "
              f"eligible={s['n_analysis_eligible']} subj_ge1={s['n_subjects_ge1_eligible']}/{s['n_subjects']}")
    print(f"ineligible reasons: {summ['ineligible_reasons']}")


if __name__ == "__main__":
    main()
