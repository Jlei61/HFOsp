"""Topic 5 Stage-1 ictal-template-echo gate runner (proxy triage).

Spec: docs/superpowers/specs/2026-06-08-topic5-ictal-template-echo-gate-design.md (v4)

P0 invariants enforced here:
- §3.6 phantom-safe: primary templates come ONLY from
  results/interictal_propagation_masked/ (rank_a_full/rank_b_full + joint_valid),
  masked via topic5_echo_gate.masked_template_rank_1d (1-D np.where contract).
  The old unmasked bridge loader / q1prime JSON template_rank are NOT used.
- P0-B: ictal atlas via src.atlas_loading (NOT a hand-rolled atlas_v2_3 path).
- P0-1: proxy triage NEVER vetoes Stage 2 — verdicts only set Stage-2 priority and
  the artifact must not contain the veto word (cohort lint guard).

Scope (spec v4 §3.2): GENERIC template echo. primary = ALL subjects with a stable
masked template (k=2, n_valid>=MIN_CH). swap_class is a pre-registered STRATIFIER.
Negative control = between-subject (Null D) + bad-data regression.
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
from scipy.stats import rankdata

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
warnings.filterwarnings("ignore", message="Mean of empty slice")

from src import atlas_loading
from src import topic5_echo_gate as echo
from src.propagation_skeleton_geometry import parse_shaft

MASKED_DIR = Path("results/interictal_propagation_masked/rank_displacement/per_subject")
OUT_ROOT = Path("results/topic5_ictal_template_echo")
MIN_CH = 8                 # P1-1 locked
B = 2000                   # §5.3 lock
TIE_MAX = 0.3              # atlas-quality
N_ANCHOR_BINS = 4          # Null C earliness bins
BAND_PRIMARY = "broad_ER"  # user-locked 2026-06-08 (gamma_ER = sensitivity)
RNG_SEED = 20260608


def subject_atlas_quality(seiz_matrix, joint_valid):
    """Atlas-quality on the RIGHT channel set + per-seizure ties (review P1 x2).
    - mean ictal rank is masked to joint_valid BEFORE the quality check, so template-
      invalid channels (which never enter the primary correlation) cannot prop up
      quality.
    - tie fraction is measured PER SEIZURE (averaging across seizures hides ties);
      gate on the per-seizure median."""
    joint_valid = np.asarray(joint_valid, dtype=bool)
    tie_fracs = []
    for r in np.asarray(seiz_matrix, float):
        rv = r[joint_valid]                       # mask to template-valid channels first
        fin = rv[np.isfinite(rv)]
        if fin.size:
            _, c = np.unique(fin, return_counts=True)
            tie_fracs.append(float(np.sum(c[c > 1]) / fin.size))
    tie_med = float(np.median(tie_fracs)) if tie_fracs else 1.0
    tie_mx = float(np.max(tie_fracs)) if tie_fracs else 1.0
    with np.errstate(invalid="ignore"):
        mean_ictal = np.where(np.all(np.isnan(seiz_matrix), axis=0), np.nan,
                              np.nanmean(seiz_matrix, axis=0))
    mean_ictal_q = np.where(joint_valid, mean_ictal, np.nan)   # P1: mask to joint_valid
    aq = echo.compute_atlas_quality(mean_ictal_q, tie_max=TIE_MAX, min_channels=MIN_CH)
    flag = "pass" if (aq["atlas_quality_flag"] == "pass" and tie_med <= TIE_MAX) else "fail"
    return {"atlas_quality_flag": flag, "rank_tie_fraction_perseizure_median": round(tie_med, 3),
            "rank_tie_fraction_perseizure_max": round(tie_mx, 3),
            "rank_dynamic_range": aq["rank_dynamic_range"], "n_ranked_channels": aq["n_ranked_channels"]}


def ictal_rank_from_onsets(channel_onsets, channels):
    """Per-channel ictal rank over `channels`: rank ascending t_onset_sec
    (earliest recruited -> lowest rank, matching the template convention);
    NaN where the channel has no onset (not recruited / CUSUM not triggered)."""
    onsets = np.array(
        [(channel_onsets.get(ch) or {}).get("t_onset_sec", None) for ch in channels],
        dtype=object)
    t = np.array([np.nan if v is None else float(v) for v in onsets], dtype=float)
    rank = np.full(t.shape, np.nan)
    finite = np.isfinite(t)
    if finite.sum() >= 2:
        rank[finite] = rankdata(t[finite], method="average") - 1.0   # 0-based like template
    return rank


def load_subject(stem):
    """stem e.g. 'epilepsiae_1146'. Returns a dict with masked templates + ictal
    seizure ranks aligned to the SAME channel order, or None if unusable."""
    mj = MASKED_DIR / f"{stem}.json"
    if not mj.exists():
        return None
    d = json.load(open(mj))
    if d.get("stable_k") != 2 or not d.get("pairs"):
        return None
    dataset, subject = d["dataset"], d["subject"]
    channels = list(d["channel_names"])
    pair = d["pairs"][0]
    joint_valid = np.asarray(pair["joint_valid"], dtype=bool)
    t_a = echo.masked_template_rank_1d(np.asarray(pair["rank_a_full"], float), joint_valid)
    t_b = echo.masked_template_rank_1d(np.asarray(pair["rank_b_full"], float), joint_valid)
    templates = [t_a, t_b]
    swap_class = pair.get("swap_sweep", {}).get("swap_class")
    n_valid = int(joint_valid.sum())

    # --- ictal side: atlas onsets aligned to the SAME channel order (name match) ---
    try:
        atlas = atlas_loading.load_per_subject_json(f"{dataset}/{subject}", source="per_subject")
    except Exception:
        return None
    band = atlas.get("per_er", {}).get(BAND_PRIMARY)
    if band is None:
        return None
    atlas_channels = set()
    for rec in band.get("seizure_records", []):
        atlas_channels |= set((rec.get("channel_onsets") or {}).keys())
    # Alignment is by EXACT channel NAME (ictal_rank_from_onsets looks up each template
    # channel in channel_onsets), so a channel absent from the atlas can never be
    # MIS-aligned — it simply gets NaN (no onset) and drops from the common set. The
    # item-10 "hard fail" is for channel-ORDER inconsistency, which name-lookup forbids
    # by construction; partial montage overlap (common for yuquan vs its ictal atlas) is
    # handled conservatively by the MIN_CH common-channel gate below, not a hard fail.
    n_name_overlap = sum(1 for c in channels if c in atlas_channels)
    seiz_ranks = []
    seiz_ids = []
    for rec in band.get("seizure_records", []):
        if rec.get("status") != "ok":
            continue
        r = ictal_rank_from_onsets(rec.get("channel_onsets") or {}, channels)
        common = np.isfinite(r) & np.isfinite(t_a)      # template-valid (NaN if invalid) ∩ onset
        if int(np.sum(np.isfinite(r) & joint_valid)) >= MIN_CH:
            seiz_ranks.append(r)
            seiz_ids.append(rec.get("seizure_id"))
    if len(seiz_ranks) < 2:
        return None
    seiz_matrix = np.vstack(seiz_ranks)

    # atlas quality on the joint_valid-masked mean ictal rank + per-seizure ties (P1)
    aq = subject_atlas_quality(seiz_matrix, joint_valid)
    with np.errstate(invalid="ignore"):
        mean_ictal = np.where(np.all(np.isnan(seiz_matrix), axis=0), np.nan,
                              np.nanmean(seiz_matrix, axis=0))

    # Null C anchor bins = quartiles of mean ictal earliness (None where no onsets)
    anchor_bins = np.full(len(channels), None, dtype=object)
    fin = np.isfinite(mean_ictal)
    if fin.sum() >= N_ANCHOR_BINS:
        qs = np.quantile(mean_ictal[fin], np.linspace(0, 1, N_ANCHOR_BINS + 1)[1:-1])
        binid = np.digitize(mean_ictal[fin], qs)
        anchor_bins[np.where(fin)[0]] = binid.astype(object)

    return {
        "stem": stem, "dataset": dataset, "subject": subject,
        "channels": channels, "templates": templates, "swap_class": swap_class,
        "template_k": 2, "n_valid": n_valid, "n_name_overlap": int(n_name_overlap),
        "soz_channels": d.get("soz_channels", []),
        "seizure_ranks": seiz_matrix, "seizure_ids": seiz_ids,
        "atlas_quality_flag": aq["atlas_quality_flag"],
        "rank_tie_fraction": aq["rank_tie_fraction_perseizure_median"],
        "rank_tie_fraction_max": aq["rank_tie_fraction_perseizure_max"],
        "rank_dynamic_range": aq["rank_dynamic_range"], "anchor_bins": anchor_bins,
        "construct_validity_flag": "pending",
    }


def _iter_stems():
    atlas_subjects = set(atlas_loading.list_cohort_subjects())   # 'dataset/sid' keys
    for mj in sorted(MASKED_DIR.glob("*.json")):
        d = json.load(open(mj))
        if d.get("stable_k") != 2:
            continue
        key = f"{d['dataset']}/{d['subject']}"
        if key in atlas_subjects:
            yield mj.stem


# --------------------------------------------------------------------------- audit
def cmd_audit(args):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for stem in _iter_stems():
        sub = load_subject(stem)
        if sub is None:
            rows.append({"subject_id": stem, "note": "unusable (<2 eligible seizures / band missing)"})
            continue
        sm = sub["seizure_ranks"]
        ncommon = [int(np.sum(np.isfinite(r) & np.isfinite(sub["templates"][0]))) for r in sm]
        rows.append({
            "subject_id": stem, "dataset": sub["dataset"],
            "n_seizures_eligible": len(sm), "n_channels_template": sub["n_valid"],
            "n_name_overlap": sub["n_name_overlap"],
            "n_channels_common_min": min(ncommon), "n_channels_common_median": int(np.median(ncommon)),
            "n_channels_common_max": max(ncommon), "rank_tie_fraction": round(sub["rank_tie_fraction"], 3),
            "rank_dynamic_range": round(sub["rank_dynamic_range"], 2), "template_k": 2,
            "swap_class": sub["swap_class"], "ictal_rank_source": f"ER_atlas:{BAND_PRIMARY}",
            "atlas_quality_flag": sub["atlas_quality_flag"], "construct_validity_flag": "pending",
            "phantom_mask_applied": True, "valid_mask_source": "rank_displacement.joint_valid",
            "alignment_guard_pass": True,
            "deanchor_eligible": len(sm) >= 4,
            "deanchor_anchor_reliability": round(echo.anchor_reliability(sm), 3),
        })
    cols = ["subject_id", "dataset", "n_seizures_eligible", "n_channels_template",
            "n_name_overlap", "n_channels_common_min", "n_channels_common_median",
            "n_channels_common_max", "rank_tie_fraction", "rank_dynamic_range",
            "template_k", "swap_class",
            "ictal_rank_source", "atlas_quality_flag", "construct_validity_flag",
            "phantom_mask_applied", "valid_mask_source", "alignment_guard_pass",
            "deanchor_eligible", "deanchor_anchor_reliability", "note"]
    with open(OUT_ROOT / "b0_eligibility_audit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    n_ok = sum(1 for r in rows if r.get("alignment_guard_pass") and r.get("atlas_quality_flag") == "pass")
    drop8 = sum(1 for r in rows if (r.get("n_channels_common_median") or 0) < MIN_CH)
    print(f"wrote {OUT_ROOT/'b0_eligibility_audit.csv'} ({len(rows)} subjects; "
          f"{n_ok} pass atlas-quality; {drop8} have median common<{MIN_CH})")


# ------------------------------------------------------------------- per-subject
def cmd_per_subject(args):
    rng = np.random.default_rng(RNG_SEED)
    (OUT_ROOT / "per_subject").mkdir(parents=True, exist_ok=True)
    n = 0
    for stem in _iter_stems():
        try:
            sub = load_subject(stem)
        except ValueError:
            continue
        if sub is None or sub["atlas_quality_flag"] == "fail":
            continue
        templates = sub["templates"]
        shafts = np.array([parse_shaft(c)[0] for c in sub["channels"]], dtype=object)
        cap = echo.shaft_block_capacity(shafts)
        seiz = sub["seizure_ranks"]
        per_seizure = []
        for k in range(seiz.shape[0]):
            rec = {"seizure_idx": k}
            modes = [("channel", None), ("within_shaft", shafts),
                     ("anchor_matched", sub["anchor_bins"])]
            if not cap["insufficient_block_exchange"]:
                modes.append(("shaft_block", shafts))
            for mode, blocks in modes:
                rec[mode] = echo.compute_echo_strength(
                    seiz[k], templates, B=B, rng=rng, min_ch=MIN_CH,
                    null_mode=mode, blocks=blocks)
            per_seizure.append(rec)
        deanchor = (echo.compute_deanchor_echo(seiz, templates, B=B, rng=rng, min_ch=MIN_CH)
                    if seiz.shape[0] >= 4 else None)
        out = {
            "subject": stem, "dataset": sub["dataset"], "swap_class": sub["swap_class"],
            "template_k": 2, "atlas_quality_flag": sub["atlas_quality_flag"],
            "construct_validity_flag": sub["construct_validity_flag"],
            "shaft_block_capacity": cap, "anchor_reliability": echo.anchor_reliability(seiz),
            "n_seizures": int(seiz.shape[0]),
            "per_seizure": per_seizure, "deanchor": deanchor,
            "channels": list(sub["channels"]),
            "template_ranks": [list(map(_jsonnum, t)) for t in templates],
            "seizure_ranks": [list(map(_jsonnum, r)) for r in seiz],
        }
        json.dump(out, open(OUT_ROOT / "per_subject" / f"{stem}.json", "w"),
                  indent=2, default=_jsonnum)
        n += 1
    print(f"per-subject done ({n} subjects)")


def _jsonnum(o):
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, (np.floating, np.integer)):
        return None if (isinstance(o, np.floating) and not np.isfinite(o)) else o.item()
    return o


# ------------------------------------------------------------------------ cohort
def cmd_cohort(args):
    subs = [json.load(open(p)) for p in sorted((OUT_ROOT / "per_subject").glob("*.json"))]

    def pool_mode(mode, subset=None):
        recs = []
        for s in subs:
            if subset == "strict_candidate" and s["swap_class"] not in ("strict", "candidate"):
                continue
            if subset == "none" and s["swap_class"] != "none":
                continue
            for ps in s["per_seizure"]:
                m = ps.get(mode)
                if m is None:
                    continue
                recs.append({"subject": s["subject"], "e_k": m.get("e_k")})
        return echo.pool_echo_subject_level(recs)

    # P0: Null D split by dataset; yuquan A/B/C/D shaft names are WITHIN-patient
    # numbering, NOT cross-patient anatomical labels — exact-name cross-patient alignment
    # would treat coincidental name collisions as the same location, so yuquan is skipped.
    rng = np.random.default_rng(RNG_SEED + 1)
    bs_recs = []
    n_overlap_seen = []
    null_d_notes = {
        "epilepsiae": "name-aligned within epilepsiae (labels assumed cross-patient anatomical)",
        "yuquan": "SKIPPED — A/B/C/D shaft names are within-patient numbering, not "
                  "cross-patient anatomical labels; exact-name alignment is invalid",
    }
    for s in subs:
        if s["dataset"] != "epilepsiae":     # P0: yuquan name-align invalid -> skip Null D
            continue
        foreign = [(t, o["channels"]) for o in subs
                   if o["dataset"] == "epilepsiae" and o["subject"] != s["subject"]
                   for t in o["template_ranks"]]
        if not foreign:
            continue
        if len(foreign) > 8:
            idx = rng.choice(len(foreign), size=8, replace=False)
            foreign = [foreign[i] for i in idx]
        for seiz in s["seizure_ranks"]:
            r = echo.between_subject_control(np.array(seiz, float), s["channels"], foreign,
                                             B=B, rng=rng, min_ch=MIN_CH)
            n_overlap_seen.append(r.get("n_foreign_overlapping", 0))
            if r.get("n_foreign_overlapping", 0) > 0:
                bs_recs.append({"subject": s["subject"], "e_k": r["e_k"]})

    bd_recs = [{"subject": s["subject"], "e_k_baddata": ps["channel"]["e_k_baddata"]}
               for s in subs for ps in s["per_seizure"] if ps.get("channel")]

    deanchor_recs = []
    for s in subs:
        if not s.get("deanchor"):
            continue
        for r in s["deanchor"]:
            deanchor_recs.append({"subject": s["subject"], "e_k": r.get("e_k")})

    cv = [s.get("construct_validity_flag", "pending") for s in subs]
    construct_status = "pass" if (cv and all(x == "pass" for x in cv)) else "pending"
    summary = {
        "scope": "generic_template_echo", "n_subjects_loaded": len(subs),
        "construct_validity_status": construct_status,
        "primary_channel_all": pool_mode("channel"),
        "primary_within_shaft_all": pool_mode("within_shaft"),
        "primary_anchor_matched_all": pool_mode("anchor_matched"),
        "deanchor_all": echo.pool_echo_subject_level(deanchor_recs),
        "stratifier_swap_strict_candidate": pool_mode("channel", "strict_candidate"),
        "stratifier_swap_none": pool_mode("channel", "none"),
        "negative_between_subject_epilepsiae": echo.pool_echo_subject_level(bs_recs),
        "null_d_notes": null_d_notes,
        "between_subject_n_overlap_median": float(np.median(n_overlap_seen)) if n_overlap_seen else 0.0,
        "between_subject_n_overlap_max": int(max(n_overlap_seen)) if n_overlap_seen else 0,
        "bad_data_regression": echo.bad_data_regression(bd_recs),
    }
    summary["verdict"] = _assign_verdict(summary)
    assert "暂缓" not in json.dumps(summary, ensure_ascii=False)   # P0-1 no-veto guard
    json.dump(summary, open(OUT_ROOT / "cohort_echo_summary.json", "w"), indent=2)
    print("verdict:", summary["verdict"]["label"], "|", summary["verdict"]["why"])


def _assign_verdict(summary):
    p = summary["primary_channel_all"]
    has_sens = (np.isfinite(p.get("sign_p_onesided", np.nan)) and
                np.isfinite((p.get("boot_ci95") or [np.nan])[0]))
    neg = summary["negative_between_subject_epilepsiae"]
    bad = summary["bad_data_regression"]
    construct_ok = summary.get("construct_validity_status") == "pass"
    # Null D applicability: needs an actual cross-patient name-overlap to run. With
    # disjoint electrode labels it is INAPPLICABLE (n_subjects==0 / not enough overlap) —
    # that is NOT a clean pass; flag it so the verdict cannot claim subject-specificity.
    neg_applicable = (neg.get("n_subjects") or 0) >= 6
    neg_clean = neg_applicable and not ((neg.get("wilcoxon_p_onesided") or 1) < 0.05)
    bad_clean = not ((bad.get("wilcoxon_p_onesided") or 1) < 0.05)
    if p["n_subjects"] < 6:
        return {"label": "没看清", "why": "n_subjects<6"}
    # P1: construct-validity sentinel is a precondition for ANY standing verdict.
    if not construct_ok:
        return {"label": "代理计算跑通·控制未闭环",
                "why": "construct-validity sentinel is 'pending' — cannot certify the ER "
                       "proxy order reflects propagation (有形状≠是传播); only 'proxy "
                       "pipeline runs', NOT a standing verdict. Run the sentinel first.",
                "neg_applicable": neg_applicable}
    wp = p.get("wilcoxon_p_onesided")
    primary_sig = wp is not None and wp < 0.05 and p["median_E_s"] > 0 and has_sens and bad_clean
    if not primary_sig:
        return {"label": "代理阴性/没看清",
                "why": "primary not significant or sensitivities/bad-data not clean; "
                       "ER proxy gave no continuation evidence — Stage 2 by scientific value, not vetoed",
                "neg_applicable": neg_applicable}
    a = summary["primary_within_shaft_all"]
    c = summary["primary_anchor_matched_all"]
    a_sig = (a.get("wilcoxon_p_onesided") or 1) < 0.05
    c_sig = (c.get("wilcoxon_p_onesided") or 1) < 0.05
    # primary is significant (inclusive echo). Subject-SPECIFICITY cannot be claimed
    # without an applicable Null D — so cap at inclusive when Null D is inapplicable.
    if not neg_applicable:
        return {"label": "站住·inclusive（特异性未判定）",
                "why": f"inclusive echo holds (p={wp:.3f}); within-shaft {'survives' if a_sig else 'flat'}, "
                       f"anchor-matched {'survives' if c_sig else 'FLAT'}; "
                       "Null D (between-subject) INAPPLICABLE — patients have disjoint channel "
                       "names, so subject-specificity is NOT established",
                "neg_applicable": False}
    if not neg_clean:
        return {"label": "站住·inclusive（非特异：Null D 显著）",
                "why": "between-subject control also significant -> anatomy-general, not subject-specific"}
    specific = a_sig or c_sig
    return {"label": "站住·含具体通路" if specific else "站住·稳定锚为主",
            "why": "inclusive echo holds; A/C " + ("survive" if specific else "flatten"),
            "neg_applicable": True}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit").set_defaults(func=cmd_audit)
    sub.add_parser("per-subject").set_defaults(func=cmd_per_subject)
    sub.add_parser("cohort").set_defaults(func=cmd_cohort)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
