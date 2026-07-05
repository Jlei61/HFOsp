"""M3B Round-1 Task 2 Part 3 — model propagation axis vs the real ICTAL-early axis (INSTRUMENT-PROBE).

Scope lock (P1-1): the model field is the KICK-DRIVEN LIF RATE FIELD (a labeled instrument probe).
Verdicts are instrument-probe statements, NOT a spontaneous-mechanism claim.

Plan Task 2: "compare the model axis to the ictal-early axis." The SAME model 45deg field (Task-1
record) is compared to each subject's median broadband ICTAL activation (bb_auc = A-line PRIMARY)
laid on that subject's interictal axis frame via `make_field_record` (reuse, §6). Two legs (§6.3):
  - placement   : does the model field land inside the real-to-real ICTAL field-similarity dist?
  - geometry null: does the match beat the model rank-shuffle null (channel + within_shaft)?

P1-3 fix: PRIMARY cohort = Epilepsiae only (Topic 5 A-line primary is 18 Epilepsiae; Yuquan is a
single case, not a cohort). Yuquan is reported separately as descriptive, never folded into primary.

§6.1: model rank-shuffle geometry null (the literal A-line per-name within-subject null does not
apply to the model). anchor_matched/joint need a model ictal anchor -> N/A.

Run from worktree root: python scripts/run_m3b_task2_part3_ictal_axis.py
"""
import os
import sys
import json
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.getcwd())
from src import propagation_contact_plane_readout as R                            # noqa: E402
from src.topic5_axis_alignment import matched_channels, make_field_record        # noqa: E402

ROOTM = "/home/honglab/leijiaxin/HFOsp/results"
AXIS_DIR = f"{ROOTM}/spatial_modulation/propagation_geometry/observation_readout/real_subjects"
CACHE_DIR = f"{ROOTM}/topic5_ictal_recruitment/t0_feature_cache"
MODEL_REC = "results/topic4_sef_hfo/m3b_bridge/task1_pilot/model_record_lif_rate_45deg.json"
OUT = "results/topic4_sef_hfo/m3b_bridge/task2_bridge"
B = 2000
SEED = 0

X, Y = R.make_plane_grid()


def _field(rec):
    f = R.smooth_field(rec, X, Y, sigma_xy=None, scalar="rank", s_thresh=R.S_THRESH)
    return f["T"], f["S"]


def _corr(a, b):
    c = R.corr_pair_mirror_invariant(a[0], a[1], b[0], b[1], R.S_THRESH, R.OVERLAP_MIN)["corr"]
    return abs(c) if (c is not None and np.isfinite(c)) else np.nan


def build_ictal_fields(activation="bb_auc"):
    out = {}
    for fn in sorted(os.listdir(CACHE_DIR)):
        if not fn.endswith(".npz"):
            continue
        sid = fn[:-4]
        axis_f = f"{AXIS_DIR}/{sid}_t_a.json"
        if not os.path.exists(axis_f):
            continue
        axis = json.load(open(axis_f))
        if not axis.get("channels"):
            continue
        data = np.load(f"{CACHE_DIR}/{sid}.npz", allow_pickle=True)
        meta = json.load(open(f"{CACHE_DIR}/{sid}.json"))
        cache_names = [str(x) for x in data["channels"]]
        cidx = {n: i for i, n in enumerate(cache_names)}
        matched = matched_channels(axis, {n: 0.0 for n in cache_names})
        if len(matched) < 6:
            continue
        m_in_cache = np.array([cidx[c["name"]] for c in matched])
        cols = []
        for idx in meta["eligible_idxs"]:
            key = f"{activation}__{idx}"
            if key in data.files:
                v = data[key][m_in_cache].astype(float)
                if np.isfinite(v).sum() >= 6:
                    cols.append(v)
        if not cols:
            continue
        med = np.nanmedian(np.vstack(cols), axis=0)
        out[sid] = make_field_record(matched, med)
    return out


def _permuted_model(model_rec, idx_groups, rng):
    chans = [dict(c) for c in model_rec["channels"]]
    ranks = np.array([c["typical_rank"] for c in chans], float)
    for grp in idx_groups:
        if len(grp) > 1:
            ranks[grp] = ranks[rng.permutation(grp)]
    for i, c in enumerate(chans):
        c["typical_rank"] = float(ranks[i])
    return {**model_rec, "channels": chans}


def _median_corr_to(model_field, fields):
    cs = [_corr(model_field, f) for f in fields]
    cs = [c for c in cs if np.isfinite(c)]
    return float(np.median(cs)) if cs else float("nan")


def _assess(model_rec, ictal_fields, shaft_groups, all_idx):
    """placement (vs real-to-real LOSO) + geometry null on a given ictal-field cohort."""
    model_field = _field(model_rec)
    real_corr = _median_corr_to(model_field, ictal_fields)
    r2r = []
    for i in range(len(ictal_fields)):
        others = [ictal_fields[j] for j in range(len(ictal_fields)) if j != i]
        r2r.append(_median_corr_to(ictal_fields[i], others))
    r2r = np.array([v for v in r2r if np.isfinite(v)], float)
    placement_pct = float(100.0 * np.mean(r2r <= real_corr)) if r2r.size else float("nan")
    rng = np.random.default_rng(SEED)

    def _null(groups):
        vals = []
        for _ in range(B):
            mf = _field(_permuted_model(model_rec, groups, rng))
            v = _median_corr_to(mf, ictal_fields)
            if np.isfinite(v):
                vals.append(v)
        a = np.array(vals, float)
        p95 = float(np.percentile(a, 95)) if a.size else float("nan")
        p = float((1 + int(np.sum(a >= real_corr))) / (1 + a.size)) if a.size else float("nan")
        return {"null_p95": p95, "null_median": float(np.median(a)) if a.size else float("nan"),
                "p_value": p, "beats_null": bool(real_corr > p95), "B_effective": int(a.size)}

    ch_null = _null(all_idx)
    sh_null = _null(shaft_groups)
    inside = np.isfinite(placement_pct) and placement_pct >= 5.0
    if ch_null["beats_null"]:
        verdict = ("instrument-probe: model lands in the ictal cohort AND beats the geometry null => "
                   "the model's 45deg scaffold matches the real ICTAL-early gradient BEYOND geometry "
                   "(sign-free collinear, NOT directional replay).")
    elif inside:
        verdict = ("PLACEMENT-ONLY (instrument-probe): model lands inside the ictal cohort but does NOT "
                   "beat the geometry null => ictal match is geometry-level, not a structure-beating "
                   "scaffold match. Consistent with the data's OWN coarse interictal<->ictal alignment "
                   "(A-line: only the coarse skeleton survives the channel null; fine ictal alignment weak).")
    else:
        verdict = "model does NOT land in the ictal cohort (placement outside) — no ictal bridge."
    return {"real_model_to_ictal_median_corr": real_corr,
            "placement_percentile_in_real_to_real_ictal": placement_pct, "n_r2r": int(r2r.size),
            "geometry_null": {"channel": ch_null, "within_shaft": sh_null,
                              "anchor_matched": "N/A — model has no ictal-activity anchor",
                              "joint": "N/A — model has no ictal-activity anchor"},
            "verdict": verdict}


def run(activation="bb_auc"):
    os.makedirs(OUT, exist_ok=True)
    model_rec = json.loads(open(MODEL_REC).read())
    ictal = build_ictal_fields(activation)
    sids = sorted(ictal)
    epi = [s for s in sids if s.startswith("epilepsiae_")]
    yuq = [s for s in sids if s.startswith("yuquan")]

    n_ch = len(model_rec["channels"])
    all_idx = [list(range(n_ch))]
    shafts = defaultdict(list)
    for i, c in enumerate(model_rec["channels"]):
        shafts[c.get("shaft", "?")].append(i)
    shaft_groups = list(shafts.values())

    # PRIMARY = Epilepsiae cohort only (P1-3)
    primary = _assess(model_rec, [_field(ictal[s]) for s in epi], shaft_groups, all_idx)

    # Yuquan = descriptive only (single case, NOT a cohort) — just the model↔yuquan corr
    model_field = _field(model_rec)
    yuq_desc = {s: _corr(model_field, _field(ictal[s])) for s in yuq}

    out = {
        "task": "M3B round-1 Task 2 Part 3 — model field vs real ICTAL-early field (INSTRUMENT-PROBE)",
        "bridge_scope": "kick-driven LIF rate field, instrument probe (NOT spontaneous mechanism)",
        "activation": activation + (" (broadband = A-line PRIMARY)" if activation == "bb_auc"
                                    else " (HFA 60-100Hz = A-line fine-aligned sensitivity metric)"),
        "PRIMARY_cohort": "epilepsiae",
        "n_primary_epilepsiae": len(epi),
        "primary_subjects": epi,
        "primary": primary,
        "yuquan_descriptive": {"note": "single case, NOT a cohort — descriptive only (Topic 5 contract)",
                               "model_to_yuquan_ictal_corr": yuq_desc},
        "B": B, "seed": SEED,
    }
    fname = "task2_part3_ictal_axis.json" if activation == "bb_auc" else f"task2_part3_ictal_axis_{activation}.json"
    with open(os.path.join(OUT, fname), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--activation", choices=["bb_auc", "hfa_auc"], default="bb_auc",
                    help="ictal activation metric: bb_auc (broadband, A-line PRIMARY) or "
                         "hfa_auc (HFA 60-100Hz, the data's fine-aligned sensitivity metric)")
    a = ap.parse_args()
    run(activation=a.activation)
