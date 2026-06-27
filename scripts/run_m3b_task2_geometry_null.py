"""M3B Round-1 Task 2 — geometry null for the model→cohort field bridge (INSTRUMENT-PROBE).

Scope lock (P1-1): the model field is the KICK-DRIVEN LIF RATE FIELD (a labeled instrument probe),
NOT a spontaneous-M3 event. Every verdict below is an "instrument-probe bridge" statement; it does
NOT claim a spontaneous mechanism reproduced the real scaffold.

Plan §3.1/§4/§8: `compare_model_to_cohort` placement alone does NOT strip geometry (its baseline is
cohort-internal). The "shared SCAFFOLD (not just shared geometry)" claim requires the model's cohort
field-match to beat a GEOMETRY-STRIPPED null.

§6.1 question-match (why this is NOT the literal A-line four-tier null): the A-line null joins the
interictal axis ↔ ICTAL activation BY CHANNEL NAME within a real subject. The model has virtual
contacts (A0..C5), no name-match, no paired ictal field — it cannot enter that statistic. The
faithful execution of the plan's INTENT is the same geometry-stripping PRINCIPLE on the model:
permute the model's per-contact recruitment RANK while holding contact geometry/support fixed,
rebuild the smoothed rank field, and ask whether the real 45°-structured field matches the cohort
BETTER than a random rank arrangement on the same contacts.

Tiers applicable to the model: channel (permute across all contacts) + within_shaft (permute within
each model shaft A/B/C). anchor_matched/joint bin by ICTAL ACTIVITY -> N/A for the model.

P1-2 fix: the statistic is SUBJECT-FIRST folded (median within (dataset,subject), then median across
subjects) to match the formal `compare_model_to_cohort` field-placement denominator — NOT a raw
median over t_a/t_b records (which double-weights multi-template subjects).

Run from worktree root: python scripts/run_m3b_task2_geometry_null.py
"""
import os
import sys
import json
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.getcwd())
from src import propagation_contact_plane_readout as R                            # noqa: E402

REAL_DIR = ("/home/honglab/leijiaxin/HFOsp/results/spatial_modulation/"
            "propagation_geometry/observation_readout/real_subjects")
MODEL_REC = "results/topic4_sef_hfo/m3b_bridge/task1_pilot/model_record_lif_rate_45deg.json"
OUT = "results/topic4_sef_hfo/m3b_bridge/task2_bridge"
B = 2000
SEED = 0

X, Y = R.make_plane_grid()


def _load_reals(d, two_d_only=False):
    recs = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        r = json.loads(open(os.path.join(d, fn)).read())
        if r.get("status") in ("no_events", "descriptive_only") or not r.get("channels"):
            continue
        if two_d_only and r.get("flags", {}).get("one_dimensional_sampling"):
            continue
        recs.append(r)
    return recs


def _field(rec):
    f = R.smooth_field(rec, X, Y, sigma_xy=None, scalar="rank", s_thresh=R.S_THRESH)
    return f["T"], f["S"]


def _subject_first_median_corr(model_T, model_S, real_items):
    """Subject-first folded median |corr| of the model field to the real cohort.
    real_items: list of (dataset, subject, (T, S)). Fold within (dataset,subject) first."""
    rows = []
    for ds, subj, (Tr, Sr) in real_items:
        c = R.corr_pair_mirror_invariant(model_T, model_S, Tr, Sr,
                                         s_thresh=R.S_THRESH, overlap_min=R.OVERLAP_MIN)["corr"]
        if c is not None and np.isfinite(c):
            rows.append({"dataset": ds, "subject": subj, "corr": abs(c)})
    folded = R.subject_first_fold(rows, "corr")       # one value per subject
    return (float(np.median(folded)) if folded else float("nan")), len(folded)


def _permuted_record(rec, idx_groups, rng):
    chans = [dict(c) for c in rec["channels"]]
    ranks = np.array([c["typical_rank"] for c in chans], float)
    for grp in idx_groups:
        if len(grp) > 1:
            ranks[grp] = ranks[rng.permutation(grp)]
    for i, c in enumerate(chans):
        c["typical_rank"] = float(ranks[i])
    return {**rec, "channels": chans}


def _null(model_rec, idx_groups, real_items, rng):
    vals = []
    for _ in range(B):
        mr = _permuted_record(model_rec, idx_groups, rng)
        mt, ms = _field(mr)
        v, _n = _subject_first_median_corr(mt, ms, real_items)
        if np.isfinite(v):
            vals.append(v)
    return np.array(vals, float)


def run(two_d_only=False):
    os.makedirs(OUT, exist_ok=True)
    reals = _load_reals(REAL_DIR, two_d_only=two_d_only)
    model_rec = json.loads(open(MODEL_REC).read())
    real_items = [(r["dataset"], r["subject"], _field(r)) for r in reals]
    mt, ms = _field(model_rec)
    real_corr, n_subj = _subject_first_median_corr(mt, ms, real_items)

    n_ch = len(model_rec["channels"])
    all_idx = [list(range(n_ch))]                                  # channel tier
    shafts = defaultdict(list)
    for i, c in enumerate(model_rec["channels"]):
        shafts[c.get("shaft", "?")].append(i)
    shaft_groups = [g for g in shafts.values()]                    # within_shaft tier
    eff_shuffle_n = sum(len(g) for g in shaft_groups if len(g) > 1)

    rng = np.random.default_rng(SEED)
    null_channel = _null(model_rec, all_idx, real_items, rng)
    null_shaft = _null(model_rec, shaft_groups, real_items, rng)

    def _tier(null):
        p95 = float(np.percentile(null, 95)) if null.size else float("nan")
        p = float((1 + int(np.sum(null >= real_corr))) / (1 + null.size)) if null.size else float("nan")
        return {"null_p95": p95, "null_median": float(np.median(null)) if null.size else float("nan"),
                "p_value": p, "beats_null": bool(real_corr > p95), "B_effective": int(null.size)}

    out = {
        "task": "M3B round-1 Task 2 geometry null — INSTRUMENT-PROBE bridge (kick-rate-field model field)",
        "bridge_scope": "kick-driven LIF rate field, labeled instrument probe (NOT spontaneous M3 mechanism)",
        "statistic": "subject-first median over real subjects of |corr_pair_mirror_invariant(model_field, real_field)|",
        "real_model_to_real_median_corr": real_corr,
        "n_real_records": len(reals),
        "n_real_subjects_subject_first": n_subj,
        "n_model_contacts": n_ch,
        "within_shaft_effective_shuffle_n": int(eff_shuffle_n),
        "tiers": {
            "channel": _tier(null_channel),
            "within_shaft": _tier(null_shaft),
            "anchor_matched": "N/A — needs ictal activity bins (model has none)",
            "joint": "N/A — needs ictal activity bins (model has none)",
        },
        "B": B, "seed": SEED,
        "interpretation": (
            "beats channel null => the model's cohort match depends on its 45deg PROPAGATION structure, "
            "not just contact geometry (shared interictal scaffold, instrument-probe). does NOT beat => "
            "placement-only (shared geometry)."),
    }
    out["real_cohort"] = "2D-only" if two_d_only else "all"
    fname = "task2_geometry_null_2d_only.json" if two_d_only else "task2_geometry_null.json"
    with open(os.path.join(OUT, fname), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-2d-only", action="store_true",
                    help="restrict the real cohort to 2D-sampled records (P1-4 fairness check)")
    a = ap.parse_args()
    run(two_d_only=a.real_2d_only)
