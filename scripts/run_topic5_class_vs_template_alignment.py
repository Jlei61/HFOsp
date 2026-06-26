#!/usr/bin/env python3
"""Topic 5 — class-field vs template-field alignment (max-over-AB), cohort.

User ask (2026-06-25): compare the EVENT-AGGREGATED per-class interictal field against the
AGGREGATE template field, using the SAME statistic as the A-line template result and the SAME
"max" rule (candidate = max over {A,B}; null pays the selection cost). Targets = the two
pre-ictal windows (−10..0 s, −120..−90 s) shown in the 6-panel figure, plus the seizure-onset
window (0..10 s) for continuity with the A-line.

Per subject (broad cohort, 12 epi): build template-A/B fields (typical_rank) and class-A/B fields
(all events of that class, weight-normalized) on the broad t_a / t_b planes; for each target
window compute max-AB per-seizure alignment + channel-shuffle null (selection cost paid) for BOTH
representations. Cohort summary compares template vs class (paired). EXPLORATORY secondary; no
cohort verdict — awaits advisor (user) sign-off.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import src.topic5_event_resolved_alignment as erm
from src.propagation_contact_plane_readout import make_plane_grid

GEOM_BROAD = _ROOT / "results/spatial_modulation/propagation_geometry_broad/observation_readout/real_subjects"
WIN_CACHES = _ROOT / "results/topic5_ictal_recruitment/axis_alignment/window_caches"
OUT = _ROOT / "results/topic5_ictal_recruitment/event_resolved_alignment/class_vs_template"
RNG_SEED = 20260625
TARGETS = ["pre_prox_m10_0", "pre_distal_m120_m90", "post_0_10"]   # 2 pre-ictal + onset (continuity)


def _target_seizures(ds_sid, window, key="bb_auc"):
    npz = WIN_CACHES / window / f"{ds_sid}.npz"; js = WIN_CACHES / window / f"{ds_sid}.json"
    if not npz.exists() or not js.exists():
        return None
    data = np.load(npz, allow_pickle=True); meta = json.load(open(js))
    names = [str(x) for x in data["channels"]]
    out = []
    for idx in meta.get("eligible_idxs", []):
        k = f"{key}__{idx}"
        if k not in data.files:
            continue
        vals = np.asarray(data[k], float)
        out.append({n: float(v) for n, v in zip(names, vals) if np.isfinite(v)})
    return out


def _run_subject(ds_sid, *, n_null, rng):
    dataset, subject = ds_sid.split("_", 1)
    ta_f, tb_f = GEOM_BROAD / f"{ds_sid}_t_a.json", GEOM_BROAD / f"{ds_sid}_t_b.json"
    if not (ta_f.exists() and tb_f.exists()):
        return {"subject_id": ds_sid, "status": "no_broad_planes"}
    try:
        bundle = erm.load_event_labels_ranks(dataset, subject)
    except Exception as e:
        return {"subject_id": ds_sid, "status": f"load_error:{e}"}
    plane_a = json.loads(ta_f.read_text()); plane_b = json.loads(tb_f.read_text())

    order = bundle["channel_names"]
    ta_rank = np.array([{c["name"]: c.get("typical_rank") for c in plane_a["channels"]}.get(n, np.nan)
                        for n in order], float)
    tb_rank = np.array([{c["name"]: c.get("typical_rank") for c in plane_b["channels"]}.get(n, np.nan)
                        for n in order], float)
    cmap = erm.map_clusters_to_templates(np.array(bundle["cluster_template_ranks"][0], float),
                                         np.array(bundle["cluster_template_ranks"][1], float),
                                         ta_rank, tb_rank)
    if cmap["ambiguous"]:
        return {"subject_id": ds_sid, "status": "cluster_map_ambiguous",
                "diag_minus_offdiag": cmap["diag_minus_offdiag"]}
    label_A = [k for k, t in cmap["map"].items() if t == "t_a"][0]
    label_B = [k for k, t in cmap["map"].items() if t == "t_b"][0]

    X, Y = make_plane_grid()
    sigma_a = erm.class_template_sigma(plane_a, X=X, Y=Y)
    sigma_b = erm.class_template_sigma(plane_b, X=X, Y=Y)

    # template fields (typical_rank, plane aggregate support)
    F_tplA = erm.field_from_contact_values(plane_a, {c["name"]: c["typical_rank"] for c in plane_a["channels"]},
                                           sigma=sigma_a, X=X, Y=Y)
    F_tplB = erm.field_from_contact_values(plane_b, {c["name"]: c["typical_rank"] for c in plane_b["channels"]},
                                           sigma=sigma_b, X=X, Y=Y)
    # class fields (event-aggregate, participation support)
    cvA = erm.class_aggregate_contact_values(bundle, label_A)
    cvB = erm.class_aggregate_contact_values(bundle, label_B)
    F_clsA = erm.field_from_contact_values(plane_a, {n: d["value"] for n, d in cvA.items()},
                                           support_by_name={n: d["support"] for n, d in cvA.items()},
                                           sigma=sigma_a, X=X, Y=Y)
    F_clsB = erm.field_from_contact_values(plane_b, {n: d["value"] for n, d in cvB.items()},
                                           support_by_name={n: d["support"] for n, d in cvB.items()},
                                           sigma=sigma_b, X=X, Y=Y)
    if any(f is None for f in (F_tplA, F_tplB, F_clsA, F_clsB)):
        return {"subject_id": ds_sid, "status": "field_build_failed"}

    res = {"subject_id": ds_sid, "dataset": dataset, "status": "ok",
           "cluster_map": cmap["map"], "n_events_valid_full": int(bundle["labels"].size),
           "windows": {}}
    for w in TARGETS:
        tgt = _target_seizures(ds_sid, w)
        if not tgt:
            res["windows"][w] = {"status": "no_window_cache"}; continue
        both = erm.maxab_two_reps_vs_target(
            {"template": (F_tplA, F_tplB), "class": (F_clsA, F_clsB)},
            plane_a, plane_b, sigma_a, sigma_b, tgt, n_null=n_null, rng=rng, X=X, Y=Y)
        tpl, cls = both["template"], both["class"]
        res["windows"][w] = {"template": tpl, "class": cls,
                             "class_minus_template": (
                                 (cls["real_median_maxab"] - tpl["real_median_maxab"])
                                 if (tpl.get("status") == "ok" and cls.get("status") == "ok") else None)}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="*", default=None, help="default = all broad-cohort epi")
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--force", action="store_true", help="recompute even if per_subject json exists")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    if args.subjects:
        subs = args.subjects
    else:
        labels = {p.name[:-5] for p in (_ROOT / "results/interictal_propagation_masked_broad/per_subject").glob("*.json")}
        subs = sorted(s for s in labels if (GEOM_BROAD / f"{s}_t_a.json").exists()
                      and (GEOM_BROAD / f"{s}_t_b.json").exists())
    outdir = Path(args.out); psdir = outdir / "per_subject"; psdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    for ds_sid in subs:
        if not args.force and (psdir / f"{ds_sid}.json").exists():
            print(f"[skip-exists] {ds_sid}", flush=True); continue
        print(f"[run] {ds_sid} ...", flush=True)
        res = _run_subject(ds_sid, n_null=args.n_null, rng=rng)
        json.dump(res, open(psdir / f"{ds_sid}.json", "w"), indent=2)
        if res["status"] == "ok":
            for w, d in res["windows"].items():
                t, c = d.get("template", {}), d.get("class", {})
                if t.get("status") == "ok" and c.get("status") == "ok":
                    print(f"    {w}: tpl={t['real_median_maxab']:.3f}(p{t['pass_channel_null']}) "
                          f"cls={c['real_median_maxab']:.3f}(p{c['pass_channel_null']}) "
                          f"Δ={d['class_minus_template']:+.3f}", flush=True)
        else:
            print(f"    status={res['status']}", flush=True)

    # cohort summary read FROM DISK (so partial/batched/resumed runs still summarize)
    rows = []
    for f in sorted(psdir.glob("*.json")):
        res = json.load(open(f))
        if res.get("status") != "ok":
            continue
        for w, d in res.get("windows", {}).items():
            t, c = d.get("template", {}), d.get("class", {})
            if t.get("status") == "ok" and c.get("status") == "ok":
                rows.append({"subject_id": res["subject_id"], "window": w,
                             "template_maxab": t["real_median_maxab"], "template_pass": t["pass_channel_null"],
                             "class_maxab": c["real_median_maxab"], "class_pass": c["pass_channel_null"],
                             "class_minus_template": d["class_minus_template"]})
    cohort = {}
    for w in TARGETS:
        wr = [r for r in rows if r["window"] == w]
        if not wr:
            continue
        tpl = np.array([r["template_maxab"] for r in wr]); cls = np.array([r["class_maxab"] for r in wr])
        cohort[w] = {"n": len(wr),
                     "template_pass": int(sum(bool(r["template_pass"]) for r in wr)),
                     "class_pass": int(sum(bool(r["class_pass"]) for r in wr)),
                     "template_median_maxab": float(np.median(tpl)),
                     "class_median_maxab": float(np.median(cls)),
                     "class_minus_template_median": float(np.median(cls - tpl)),
                     "n_class_gt_template": int(np.sum(cls > tpl))}
    json.dump({"note": "EXPLORATORY — class vs template max-AB alignment; per-subject channel-null "
               "pass uncorrected; no cohort verdict (awaits advisor sign-off).",
               "targets": TARGETS, "n_null": args.n_null, "rows": rows, "cohort": cohort},
              open(outdir / "cohort_summary.json", "w"), indent=2)
    print(f"[done] {len(rows)} (subject×window) rows -> {outdir/'cohort_summary.json'}")
    for w, d in cohort.items():
        print(f"  {w}: n={d['n']} tpl_pass={d['template_pass']} cls_pass={d['class_pass']} "
              f"Δmed={d['class_minus_template_median']:+.3f} cls>tpl={d['n_class_gt_template']}/{d['n']}")


if __name__ == "__main__":
    main()
