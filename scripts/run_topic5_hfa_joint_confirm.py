#!/usr/bin/env python3
"""Topic 5 A-line — frozen confirmatory re-run of the ONE finding that survived the most
conservative null: fast-activity (HFA 60-100 Hz) interictal-axis vs ictal-activation alignment
beating the JOINT (shaft x activity-bin) shuffle. The cohort FINAL table flagged this as the
only metric x null cell that is fine-grained AND activity-independent. Because it lives in a
sensitivity-tier metric, it cannot be promoted to a primary claim off the same exploratory pass;
this script LOCKS its parameters and re-runs it cleanly + adds a within-subject split-half
robustness check + a negative-control gate.

This is NOT a held-out validation — it re-uses the same 18-subject Epilepsiae cohort. It answers
"is the hfa x joint result stable to (a) a clean locked re-run, (b) splitting each subject's
seizures odd/even, (c) a destroy-the-alignment negative control" — robustness, not independence.
A truly independent confirmation needs a second cohort (not available).

Reuses _subject / _cohort_stats from run_topic5_axis_alignment (no re-implementation).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.run_topic5_axis_alignment import _subject, _cohort_stats, CACHE_DIR, ACTIVATION_KEY
from src.topic5_axis_alignment import seizure_parity_subsets

OUT = _ROOT / "results/topic5_ictal_recruitment/axis_alignment/hfa_joint_confirm.json"
ACT = ACTIVATION_KEY["hfa"]          # hfa_auc — the metric under confirmation
B = 2000                              # locked
RNG_SEED = 20260615                   # frozen seed for this confirmation


def _epi_cohort(rows):
    ok = [r for r in rows if r and r.get("status") == "ok"]
    epi = [r for r in ok if r["dataset"] == "epilepsiae"]
    cs = _cohort_stats(epi)
    return cs, epi


def _parity_meta(ds_sid):
    meta = json.load(open(CACHE_DIR / f"{ds_sid}.json"))
    return seizure_parity_subsets(meta["eligible_idxs"])


def _run_arm(cached, label, *, sz_subsets=None, negative_control=False):
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for ds_sid in cached:
        sub = sz_subsets.get(ds_sid) if sz_subsets else None
        if sz_subsets is not None and not sub:        # empty half -> drop (single-seizure subject)
            continue
        r = _subject(ds_sid, B=B, rng=rng, activation=ACT, sz_subset=sub,
                     negative_control=negative_control)
        if r:
            rows.append(r)
    cs, epi = _epi_cohort(rows)
    print(f"\n=== {label}  (Epilepsiae n={cs.get('n')}) ===", flush=True)
    for k, lbl in [("channel", "coarse"), ("within_shaft", "within-shaft"),
                   ("anchor_matched", "activity"), ("joint", "JOINT (shaft x activity)")]:
        if cs.get(f"n_pass_{k}") is not None:
            print(f"  {lbl:24s} {cs[f'n_pass_{k}']}/{cs['n']}  "
                  f"binom={cs[f'binom_p_{k}']}  Wilcox={cs[f'wilcoxon_p_{k}']}  "
                  f"LOSO-worst={cs.get(f'loso_wilcoxon_max_p_{k}')}", flush=True)
    return {"label": label, "cohort": cs,
            "per_subject": [{"subject_id": r["subject_id"], "dataset": r["dataset"],
                             "real_median_abs_corr": r.get("real_median_abs_corr"),
                             "n_seizures": r.get("n_seizures"),
                             "pass_joint_null": r.get("pass_joint_null")}
                            for r in rows if r.get("status") == "ok"]}


def main():
    cached = sorted(p.stem for p in CACHE_DIR.glob("*.npz"))
    print(f"[hfa-joint-confirm] activation=hfa B={B} seed={RNG_SEED} | {len(cached)} subjects",
          flush=True)
    parity = {s: _parity_meta(s) for s in cached}
    even = {s: parity[s][0] for s in cached}
    odd = {s: parity[s][1] for s in cached}

    arms = {
        "full_locked_rerun": _run_arm(cached, "FULL (locked re-run)"),
        "split_half_even": _run_arm(cached, "SPLIT-HALF even", sz_subsets=even),
        "split_half_odd": _run_arm(cached, "SPLIT-HALF odd", sz_subsets=odd),
        "negative_control": _run_arm(cached, "NEGATIVE CONTROL (alignment destroyed)",
                                     negative_control=True),
    }
    # verdict: joint robust iff full + both halves significant by Wilcoxon AND negative control fails
    def _jp(arm):
        return arms[arm]["cohort"].get("wilcoxon_p_joint")
    full_p, ev_p, od_p, nc_p = _jp("full_locked_rerun"), _jp("split_half_even"), _jp("split_half_odd"), _jp("negative_control")
    robust = all(p is not None and p < 0.05 for p in (full_p, ev_p, od_p)) and (nc_p is None or nc_p >= 0.05)
    verdict = {
        "metric": "hfa (60-100 Hz)", "null": "joint (shaft x activity-bin)", "B": B,
        "joint_wilcoxon_p": {"full": full_p, "even": ev_p, "odd": od_p, "negative_control": nc_p},
        "split_half_robust": bool(robust),
        "interpretation": ("locked re-run + both seizure-halves beat the JOINT null and the "
                           "destroy-alignment control fails -> fine, activity-independent fast-"
                           "activity alignment is stable within this cohort"
                           if robust else
                           "joint result NOT stable across locked re-run / split-half / control"),
        "scope": "SAME 18-subject Epilepsiae cohort — robustness, NOT held-out independence; "
                 "promotion to primary needs a second cohort.",
    }
    out = {"verdict": verdict, "arms": arms}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n=== VERDICT: split_half_robust={robust} ===")
    print(f"joint Wilcoxon p — full={full_p} even={ev_p} odd={od_p} neg_ctrl={nc_p}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
