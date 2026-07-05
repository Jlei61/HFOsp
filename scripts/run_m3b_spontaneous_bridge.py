"""M3B exploration — SPONTANEOUS-field bridge (the path Round-1 deferred under P1-1).

Round-1 bridged the KICK instrument-probe. Here the model field is the model's OWN SPONTANEOUS
events: a lesion-nucleated, noise-driven (NO kick) cm-SNN train (surround sub-critical, so events
nucleate ONLY from the low-threshold lesion — not a near-critical whole-sheet artifact), produced
by `run_sef_hfo_snn_cm_spontaneous_readout.py`. Each spontaneous record (multi-event lagPat) ->
`build_record_from_events` (mean-rank template, like the Task-1 adapter) -> bridge to the real
interictal cohort (placement + model-rank-shuffle geometry null), the SAME Round-1 machinery.

This tests the STRONGER claim: do the model's SPONTANEOUS interictal-like events reproduce the real
interictal scaffold beyond geometry? HONEST CAVEATS (do not over-claim): lesion-DRIVEN spontaneous
(homogeneous sheet has no robust spontaneous events — static-μ); finite-size noise; the lesion
location SETS the propagation direction; per-seed single-realization risk -> need multiple seeds.

Run from worktree root: python scripts/run_m3b_spontaneous_bridge.py [--tags t1 t2 ...]
"""
import os
import sys
import json
import glob
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.getcwd())
from src import propagation_contact_plane_readout as R                            # noqa: E402
from run_m3b_task2_geometry_null import (_load_reals, _field, _subject_first_median_corr,   # noqa: E402
                                         _permuted_record, REAL_DIR)
from scripts.run_contact_plane_readout import build_record_from_events           # noqa: E402

SPONT_DIR = ("/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/observation_layer/"
             "snn_cm_spontaneous/record")
OUT = "results/topic4_sef_hfo/m3b_bridge/spontaneous"
DEFAULT_TAGS = ["oneend_neg_s1", "oneend_pos_s1"]
B = 2000
SEED = 0
X, Y = R.make_plane_grid()


def load_spont_record(tag):
    d = f"{SPONT_DIR}/{tag}"
    npz = np.load(glob.glob(f"{d}/*_lagPat_withFreqCent.npz")[0], allow_pickle=True)
    mont = json.load(open(glob.glob(f"{d}/*_montage.json")[0]))
    coords2d = np.asarray(mont["contact_coords"], float)
    coords3d = np.column_stack([coords2d, np.zeros(len(coords2d))])
    rec = build_record_from_events(
        dataset="model", subject=f"spont_{tag}", template_id="t_a",
        names=[str(x) for x in npz["chnNames"]],
        ranks=np.asarray(npz["lagPatRank"], float),
        bools=np.asarray(npz["eventsBool"]).astype(bool),
        lag_raw=np.asarray(npz["lagPatRaw"], float),
        coords=coords3d, mapped=np.ones(len(npz["chnNames"]), bool), soz_core=set(),
        montage="single", lag_time_unit="s", spacing_mm=4.0)
    return rec, int(npz["lagPatRank"].shape[1])


def _placement_and_null(model_rec, real_items):
    mf = _field(model_rec)
    real_corr, _ = _subject_first_median_corr(mf[0], mf[1], real_items)
    # real-to-real placement distribution (subject-first)
    by_subj = defaultdict(list)
    for ds, subj, fld in real_items:
        by_subj[(ds, subj)].append(fld)
    r2r = []
    for k in by_subj:
        fi = by_subj[k][0]
        others = [(d, s, f) for (d, s), fs in by_subj.items() if (d, s) != k for f in fs]
        v, _ = _subject_first_median_corr(fi[0], fi[1], others)
        if np.isfinite(v):
            r2r.append(v)
    r2r = np.array(r2r)
    placement_pct = float(100.0 * np.mean(r2r <= real_corr)) if r2r.size else float("nan")
    # geometry null
    n_ch = len(model_rec["channels"])
    shafts = defaultdict(list)
    for i, c in enumerate(model_rec["channels"]):
        shafts[c.get("shaft", "?")].append(i)
    rng = np.random.default_rng(SEED)

    def _null(groups):
        v = []
        for _ in range(B):
            mr = _permuted_record(model_rec, groups, rng)
            x, _n = _subject_first_median_corr(_field(mr)[0], _field(mr)[1], real_items)
            if np.isfinite(x):
                v.append(x)
        a = np.array(v)
        p95 = float(np.percentile(a, 95)) if a.size else float("nan")
        p = float((1 + int(np.sum(a >= real_corr))) / (1 + a.size)) if a.size else float("nan")
        return {"null_p95": p95, "p_value": p, "beats_null": bool(real_corr > p95)}
    ch = _null([list(range(n_ch))])
    sh = _null(list(shafts.values()))
    return real_corr, placement_pct, ch, sh


def run(tags):
    os.makedirs(OUT, exist_ok=True)
    reals = _load_reals(REAL_DIR)
    real_items = [(r["dataset"], r["subject"], _field(r)) for r in reals]
    rows = []
    for tag in tags:
        rec, n_ev = load_spont_record(tag)
        if rec.get("status") in ("descriptive_only",) or not rec.get("channels"):
            rows.append({"tag": tag, "n_events": n_ev, "status": rec.get("status", "no_channels")})
            continue
        real_corr, pct, ch, sh = _placement_and_null(rec, real_items)
        rows.append({
            "tag": tag, "n_events": n_ev, "n_channels": len(rec["channels"]),
            "status": rec.get("status", "ok"),
            "one_dimensional_sampling": rec.get("flags", {}).get("one_dimensional_sampling"),
            "model_to_real_median_corr": real_corr,
            "placement_percentile": pct,
            "geometry_null": {"channel": ch, "within_shaft": sh},
            "lands_inside": bool(np.isfinite(pct) and pct >= 5.0),
            "beats_geometry_channel": ch["beats_null"],
        })
    out = {
        "task": "M3B SPONTANEOUS-field bridge (lesion-nucleated noise-driven cm-SNN, NO kick)",
        "vs": "Round-1 used the kick instrument-probe; this uses the model's OWN spontaneous events",
        "caveats": "lesion-DRIVEN (homogeneous sheet has no spontaneous events); finite-size noise; "
                   "lesion location sets direction; per-seed single-realization -> multi-seed for robustness",
        "real_cohort_n_subjects": len({(r['dataset'], r['subject']) for r in reals}),
        "B": B, "realizations": rows,
    }
    with open(os.path.join(OUT, "spontaneous_bridge.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    a = ap.parse_args()
    run(a.tags)
