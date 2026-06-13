#!/usr/bin/env python3
"""Track E1 (Yuquan-only clinical capstone) — template-resection coverage metrics.

Computes the PREDICTOR side now (templates/热凝 all present). The OUTCOME side is
gated on hospital follow-up labels (absent from repo; case docs only to 24h post-op),
so this writes a FROZEN EMPTY outcome schema for the user to fill, and a merged file
tagged outcome_status="labels_missing_not_analyzed". No association stats are run.

Spec: docs/superpowers/specs/2026-06-13-yuquan-template-resection-outcome-design.md

Outputs under results/template_resection_outcome/:
  yuquan_template_resection_metrics.csv   predictor metrics, one row per coverage-ok subject
  yuquan_outcome_labels.csv               frozen empty schema (subject filled, labels empty)
  yuquan_outcome_merged_metrics.csv       metrics + empty outcome cols + outcome_status column
  cohort_summary.json                     cohort layering, swap_class dist, coverage gradients, discordant list
"""
import csv
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

# reuse, don't re-invent: 热凝/onset parsing + doc resolution (spec §10)
from yuquan_template_ablation_coverage import (  # noqa: E402
    norm,
    ablated_contacts,
    onset_contacts,
    doc_for,
)
from src.template_resection_outcome import (  # noqa: E402
    coverage,
    shared_endpoint_core,
    hfo_rate_core,
    discordant_candidate,
)

PROP_DIR = os.path.join(ROOT, "results/interictal_propagation_masked/per_subject")
ANCHOR_DIR = os.path.join(ROOT, "results/interictal_propagation_masked/template_anchoring/per_subject")
RANKDISP_DIR = os.path.join(ROOT, "results/interictal_propagation_masked/rank_displacement/per_subject")
SOZ_JSON = os.path.join(ROOT, "results/yuquan_soz_core_channels.json")
NET_JSON = os.path.join(ROOT, "results/lagpat_broad/yuquan_clinical_networks.json")
PERCHAN_DIR = os.path.join(ROOT, "results/spatial_modulation/per_channel_metrics/yuquan")
OUT = os.path.join(ROOT, "results/template_resection_outcome")

MIN_CH_EVENTS = 30  # frozen (spec §4.2); matches src.sef_hfo_soz_localization.MIN_CH_EVENTS

# Old static case hints — LINE-driven leads only, NOT frozen truth (spec §5).
PRIOR_HINTS = {
    "huanghanwen": "旧dominant-source:源头完全没毁",
    "hanyuxuan": "旧dominant-source:源头没毁(新跨模板early_end=0.5)",
    "zhangkexuan": "网络大段没毁(右岛叶-岛盖);报告热凝后发作2",
    "songzishuo": "网络覆盖<50%",
    "chenziyang": "模板源≠临床起源;报告热凝后SZ2",
    "zhaojinrui": "多次热凝(第4天第二次);覆盖虚高(模板仅源头那撮)",
}

METRIC_COLS = [
    "subject",
    "template_endpoint_coverage", "n_endpoint",
    "early_end_coverage", "n_source",
    "shared_endpoint_core_coverage", "n_shared_endpoint",
    "clinical_soz_coverage", "n_soz",
    "hfo_rate_topk_sozsize_coverage", "n_hfo_rate_core", "n_hfo_rate_available",
    "clinical_network_coverage", "n_clinical_network",
    "template_coreness_endpoint_coverage",
    "late_end_coverage",
    "n_ablated", "n_resected", "resected_status",
    "n_treated_total", "treated_total_status", "multi_session",
    "swap_class", "forward_reverse_reproduced", "surgery_type",
    "template_anchor_status", "template_anchor_exit_reason", "has_template_endpoint_metric",
    "hfo_rate_core_status", "hfo_rate_core_exit_reason",
    "metric_status", "exit_reason",
    "discordant_candidate", "notes_prior_case_hint", "notes",
]

OUTCOME_COLS = [
    "subject", "surgery_type", "engel_class", "ilae_class", "followup_months",
    "recurrence", "recurrence_date", "procedure_date", "last_followup_date",
    "source", "confidence", "notes",
]


def normset(it):
    """Channel-name set, normalized (prime->ASCII, upper). All set ops are by NAME."""
    return {norm(c).upper() for c in it}


def detect_multi_session(doc_path):
    """Heuristic: a second 热凝 session mentioned in the report (spec §3)."""
    t = open(doc_path, encoding="utf-8", errors="ignore").read()
    return bool(re.search(r"第\s*[二2三3]\s*次.{0,6}热凝|二次热凝|再次热凝|两次热凝", t))


def round_or_blank(x):
    return "" if x is None else round(x, 3)


def load_anchoring(subject):
    """Template endpoint metrics from template_anchoring (NA-safe). Returns dict of
    per-template endpoint/source/sink unions + coreness + stable_k + fr_reproduced,
    or status=missing if the subject has no anchoring JSON (14/18 have it)."""
    f = os.path.join(ANCHOR_DIR, f"yuquan_{subject}.json")
    if not os.path.exists(f):
        return {"status": "missing", "exit_reason": "no_template_anchoring_json"}
    d = json.load(open(f))
    per_t = d.get("per_template", [])
    if not per_t:
        return {"status": "missing", "exit_reason": "empty_per_template"}
    endpoint_sets = [normset(pt.get("endpoint", [])) for pt in per_t]
    source_sets = [normset(pt.get("source", [])) for pt in per_t]
    sink_sets = [normset(pt.get("sink", [])) for pt in per_t]
    coreness_sets = [normset(pt.get("endpoint", [])) for pt in d.get("per_template_coreness", [])]
    audit = d.get("audit", {})
    return {
        "status": "ok",
        "exit_reason": "",
        "endpoint_sets": endpoint_sets,
        "source_sets": source_sets,
        "sink_sets": sink_sets,
        "coreness_sets": coreness_sets,
        "stable_k": audit.get("stable_k"),
        "fr_reproduced": bool(audit.get("forward_reverse_reproduced")),
        "union_endpoint": set().union(*endpoint_sets),
        "union_source": set().union(*source_sets),
        "union_sink": set().union(*sink_sets),
        "union_coreness": set().union(*coreness_sets) if coreness_sets else set(),
    }


def load_swap_class(subject):
    f = os.path.join(RANKDISP_DIR, f"yuquan_{subject}.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    pairs = d.get("pairs", [])
    if not pairs:
        return None
    return pairs[0].get("swap_sweep", {}).get("swap_class")


def load_hfo_core(subject, soz_size):
    """HFO-rate top-k(SOZ-size) core, NA-safe. Returns (core_norm, n_avail, status, exit)."""
    f = os.path.join(PERCHAN_DIR, f"{subject}_perchannel.json")
    if not os.path.exists(f):
        return set(), None, "missing", "no_perchannel_json"
    d = json.load(open(f))
    cm = d.get("channel_metrics", [])
    core, n_avail = hfo_rate_core(cm, soz_size=soz_size, min_ch_events=MIN_CH_EVENTS)
    return normset(core), n_avail, "ok", ""


def main():
    os.makedirs(OUT, exist_ok=True)
    soz_map = json.load(open(SOZ_JSON))
    net_map = json.load(open(NET_JSON))

    # cohort = doc-available subjects among propagation subjects (re-derived, not hardcoded)
    prop_subjects = sorted(
        os.path.basename(p)[len("yuquan_"):-len(".json")]
        for p in [
            os.path.join(PROP_DIR, fn) for fn in os.listdir(PROP_DIR)
            if fn.startswith("yuquan_") and fn.endswith(".json")
        ]
    )
    rows = []
    no_doc = []
    for subj in prop_subjects:
        doc = doc_for(subj)
        if not doc:
            no_doc.append(subj)
            continue

        ablated = ablated_contacts(doc)[0]          # already norm+upper
        clinical_origin = onset_contacts(doc)       # already norm+upper
        multi_session = detect_multi_session(doc)
        soz = normset(soz_map.get(subj, []))
        net = normset(net_map.get(subj, {}).get("network", []))

        # baselines
        clinical_soz_cov = coverage(soz, ablated)
        clinical_net_cov = coverage(net, ablated)
        hfo_core, n_hfo_avail, hfo_status, hfo_exit = load_hfo_core(subj, len(soz))
        hfo_cov = coverage(hfo_core, ablated) if hfo_status == "ok" else None

        # template metrics (NA-safe)
        a = load_anchoring(subj)
        if a["status"] == "ok":
            template_endpoint_cov = coverage(a["union_endpoint"], ablated)
            early_end_cov = coverage(a["union_source"], ablated)
            late_end_cov = coverage(a["union_sink"], ablated)
            coreness_cov = coverage(a["union_coreness"], ablated) if a["union_coreness"] else None
            # shared_endpoint_core gate = stable_k>=2 (spec §4.1)
            if (a["stable_k"] or 0) >= 2:
                core = shared_endpoint_core(a["endpoint_sets"])  # None if <2 templates
            else:
                core = None
            shared_cov = coverage(core, ablated) if core is not None else None
            n_endpoint = len(a["union_endpoint"])
            n_source = len(a["union_source"])
            n_shared = "" if core is None else len(core)
            template_source = a["union_source"]
            has_ep = True
        else:
            template_endpoint_cov = early_end_cov = late_end_cov = None
            coreness_cov = shared_cov = None
            n_endpoint = n_source = ""
            n_shared = ""
            template_source = set()
            has_ep = False

        swap_class = load_swap_class(subj)

        disc = discordant_candidate(
            early_end_cov, template_endpoint_cov, clinical_net_cov,
            template_source, clinical_origin, multi_session,
        )

        rows.append({
            "subject": subj,
            "template_endpoint_coverage": round_or_blank(template_endpoint_cov),
            "n_endpoint": n_endpoint,
            "early_end_coverage": round_or_blank(early_end_cov),
            "n_source": n_source,
            "shared_endpoint_core_coverage": round_or_blank(shared_cov),
            "n_shared_endpoint": n_shared,
            "clinical_soz_coverage": round_or_blank(clinical_soz_cov),
            "n_soz": len(soz),
            "hfo_rate_topk_sozsize_coverage": round_or_blank(hfo_cov),
            # n_hfo_rate_core = size-matched top-k core size (= coverage denominator),
            # NOT the available pool (that is n_hfo_rate_available).
            "n_hfo_rate_core": "" if hfo_status != "ok" else len(hfo_core),
            "n_hfo_rate_available": "" if n_hfo_avail is None else n_hfo_avail,
            "clinical_network_coverage": round_or_blank(clinical_net_cov),
            "n_clinical_network": len(net),
            "template_coreness_endpoint_coverage": round_or_blank(coreness_cov),
            "late_end_coverage": round_or_blank(late_end_cov),
            "n_ablated": len(ablated),
            "n_resected": "",                       # NA — image-only, NEVER 0 (spec §3)
            "resected_status": "image_only_unavailable",
            "n_treated_total": len(ablated),        # = ablated only (resection unknown)
            "treated_total_status": "ablation_text_only",
            "multi_session": multi_session,
            "swap_class": swap_class if swap_class is not None else "",
            "forward_reverse_reproduced": a.get("fr_reproduced", "") if a["status"] == "ok" else "",
            "surgery_type": "",                     # from outcome labels (user fills)
            "template_anchor_status": a["status"],
            "template_anchor_exit_reason": a["exit_reason"],
            "has_template_endpoint_metric": has_ep,
            "hfo_rate_core_status": hfo_status,
            "hfo_rate_core_exit_reason": hfo_exit,
            "metric_status": "ok",
            "exit_reason": "",
            "discordant_candidate": disc,
            "notes_prior_case_hint": PRIOR_HINTS.get(subj, ""),
            "notes": "",
        })

    # ---- write metrics CSV ----
    metrics_csv = os.path.join(OUT, "yuquan_template_resection_metrics.csv")
    with open(metrics_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=METRIC_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in METRIC_COLS})

    # ---- frozen empty outcome schema (subject filled, labels empty) ----
    labels_csv = os.path.join(OUT, "yuquan_outcome_labels.csv")
    with open(labels_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OUTCOME_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({"subject": r["subject"], "source": "yuquan_doc",
                        **{c: "" for c in OUTCOME_COLS if c not in ("subject", "source")}})

    # ---- merged: metrics + empty outcome cols + outcome_status column (no top metadata) ----
    merged_csv = os.path.join(OUT, "yuquan_outcome_merged_metrics.csv")
    outcome_extra = [c for c in OUTCOME_COLS if c != "subject"]
    merged_cols = METRIC_COLS + [f"outcome_{c}" for c in outcome_extra] + ["outcome_status"]
    with open(merged_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=merged_cols)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k, "") for k in METRIC_COLS}
            for c in outcome_extra:
                row[f"outcome_{c}"] = "yuquan_doc" if c == "source" else ""
            row["outcome_status"] = "labels_missing_not_analyzed"
            w.writerow(row)

    # ---- cohort summary ----
    def med(key):
        vals = [r[key] for r in rows if r[key] != ""]
        return round(float(np.median(vals)), 3) if vals else None

    swap_dist = {}
    for r in rows:
        swap_dist[r["swap_class"] or "missing"] = swap_dist.get(r["swap_class"] or "missing", 0) + 1

    summary = {
        "track": "E1_yuquan_template_resection_outcome",
        "outcome_status": "labels_missing_not_analyzed",
        "cohort_layering": {
            "coverage_ok": len(rows),
            "has_template_endpoint_metric": sum(r["has_template_endpoint_metric"] for r in rows),
            "has_hfo_rate_core": sum(r["hfo_rate_core_status"] == "ok" for r in rows),
            "has_swap_class": sum(bool(r["swap_class"]) for r in rows),
            "no_doc_excluded": no_doc,
        },
        "swap_class_distribution": swap_dist,
        "forward_reverse_reproduced_n": sum(r["forward_reverse_reproduced"] is True for r in rows),
        "coverage_medians": {
            "template_endpoint_coverage": med("template_endpoint_coverage"),
            "early_end_coverage": med("early_end_coverage"),
            "shared_endpoint_core_coverage": med("shared_endpoint_core_coverage"),
            "clinical_soz_coverage": med("clinical_soz_coverage"),
            "hfo_rate_topk_sozsize_coverage": med("hfo_rate_topk_sozsize_coverage"),
            "clinical_network_coverage": med("clinical_network_coverage"),
        },
        "discordant_candidates": [r["subject"] for r in rows if r["discordant_candidate"]],
        "hfo_rate_core_def": (
            "n_hfo_rate_core = size-matched top-k core (k=min(|SOZ|, available)) = "
            "coverage denominator; n_hfo_rate_available = full bridged HFO-active pool"
        ),
        "note": "Predictors only; outcome association NOT run (labels absent). All exploratory.",
    }
    with open(os.path.join(OUT, "cohort_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    # console
    print(f"cohort coverage-ok = {len(rows)}; no_doc excluded = {no_doc}")
    print(f"swap_class = {swap_dist}; fr_reproduced = {summary['forward_reverse_reproduced_n']}")
    print(f"anchoring = {summary['cohort_layering']['has_template_endpoint_metric']}, "
          f"hfo = {summary['cohort_layering']['has_hfo_rate_core']}")
    print(f"discordant candidates = {summary['discordant_candidates']}")
    print(f"coverage medians = {summary['coverage_medians']}")
    print(f"wrote {OUT}/")


if __name__ == "__main__":
    main()
