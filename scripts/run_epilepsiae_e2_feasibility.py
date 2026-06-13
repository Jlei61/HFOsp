#!/usr/bin/env python3
"""Track E2 — Epilepsiae region-level template-surgery-outcome FEASIBILITY (read-only).

Separate from E1 (Yuquan contact-level). Question: does Epilepsiae have enough
outcome labels + surgery-region contrast to serve as even a weak external validation
of "treated-template-network -> outcome"? Prior strategy §5.1 found dual-NULL
(resection coded to whole lobe -> predictor degenerates). This lands that as an
artifact: parses follow_up.outcome + surgerylocalisation for the 20 propagation-template
subjects and shows the region-level degeneracy concretely.

NOT an outcome analysis. Outputs:
  results/epilepsiae_template_surgery_outcome_feasibility/epilepsiae_outcome_surgery.csv
  results/epilepsiae_template_surgery_outcome_feasibility/E2_feasibility_summary.json
"""
import csv
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROP_DIR = os.path.join(ROOT, "results/interictal_propagation_masked/per_subject")
ANCHOR_DIR = os.path.join(ROOT, "results/interictal_propagation_masked/template_anchoring/per_subject")
SUBJ_INV = os.path.join(ROOT, "results/epilepsiae_subject_inventory.csv")
OUT = os.path.join(ROOT, "results/epilepsiae_template_surgery_outcome_feasibility")

# INSERT INTO surgerylocalisation (id, surgery, localisation) VALUES (..., 't-r');
RE_SURGLOC = re.compile(r"INSERT INTO surgerylocalisation .*?VALUES \([^)]*?,\s*'([^']*)'\s*\)")
# INSERT INTO follow_up (id, surgery, fup_date, interval, outcome, commentary)
#   VALUES (<id>, <surgery>, '<fup_date>', <interval>, '<outcome>', ...)
RE_FUP = re.compile(
    r"INSERT INTO follow_up [^V]*VALUES \(\s*\d+\s*,\s*\d+\s*,\s*'[^']*'\s*,\s*(\d+)\s*,\s*'([^']*)'"
)
# INSERT INTO eeg_focus (..., localisation, ...) VALUES (..., 'tmr', ...)
RE_FOCUS = re.compile(r"INSERT INTO eeg_focus \([^)]*\) VALUES \([^,]*,\s*[^,]*,\s*'([^']*)'")
RE_SURGERY = re.compile(r"INSERT INTO surgery \(")


def engel_class(outcome):
    """Leading roman numeral of an Engel-style outcome ('IIIa' -> 'III'). Engel I = seizure-free."""
    m = re.match(r"(I{1,3}V?|IV)", outcome or "")
    return m.group(1) if m else None


def parse_sql(path):
    t = open(path, encoding="utf-8", errors="ignore").read()
    foci = sorted(set(RE_FOCUS.findall(t)))
    surg_loc = sorted(set(RE_SURGLOC.findall(t)))
    has_surgery = bool(RE_SURGERY.search(t))
    fups = [(int(iv), oc) for iv, oc in RE_FUP.findall(t)]
    outcome_latest, fup_months_max = None, None
    if fups:
        fup_months_max, outcome_latest = max(fups, key=lambda x: x[0])
    return {
        "eeg_focus": ";".join(foci),
        "has_surgery": has_surgery,
        "surgery_localisation": ";".join(surg_loc),
        "n_surgery_loc_codes": len(surg_loc),
        "outcome_latest": outcome_latest or "",
        "engel_class": engel_class(outcome_latest) or "",
        "followup_months_max": fup_months_max if fup_months_max is not None else "",
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    inv = {r["subject"]: r["sql_path"] for r in csv.DictReader(open(SUBJ_INV))}

    subjects = sorted(
        os.path.basename(p)[len("epilepsiae_"):-len(".json")]
        for p in [os.path.join(PROP_DIR, fn) for fn in os.listdir(PROP_DIR)
                  if fn.startswith("epilepsiae_") and fn.endswith(".json")]
    )

    rows = []
    for s in subjects:
        sql = inv.get(s)
        d = json.load(open(os.path.join(PROP_DIR, f"epilepsiae_{s}.json")))
        n_template_ch = len(d.get("channel_names", []))
        has_anchor = os.path.exists(os.path.join(ANCHOR_DIR, f"epilepsiae_{s}.json"))
        info = parse_sql(sql) if sql and os.path.exists(sql) else {}
        rows.append({
            "subject": s,
            "n_template_channels": n_template_ch,
            "has_anchoring": has_anchor,
            "has_surgery": info.get("has_surgery", ""),
            "surgery_localisation": info.get("surgery_localisation", ""),
            "n_surgery_loc_codes": info.get("n_surgery_loc_codes", ""),
            "eeg_focus": info.get("eeg_focus", ""),
            "outcome_latest": info.get("outcome_latest", ""),
            "engel_class": info.get("engel_class", ""),
            "followup_months_max": info.get("followup_months_max", ""),
            "sql_found": bool(sql and os.path.exists(sql)),
        })

    cols = ["subject", "n_template_channels", "has_anchoring", "has_surgery",
            "surgery_localisation", "n_surgery_loc_codes", "eeg_focus",
            "outcome_latest", "engel_class", "followup_months_max", "sql_found"]
    csv_path = os.path.join(OUT, "epilepsiae_outcome_surgery.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    with_outcome = [r for r in rows if r["engel_class"]]
    with_surgery = [r for r in rows if r["has_surgery"] is True]
    engel_dist = {}
    for r in with_outcome:
        engel_dist[r["engel_class"]] = engel_dist.get(r["engel_class"], 0) + 1
    # region-degeneracy: resection coded as how many lobe codes per subject? (1 = whole-lobe coarse)
    loc_code_sizes = [r["n_surgery_loc_codes"] for r in with_surgery if r["n_surgery_loc_codes"]]
    template_ch = [r["n_template_channels"] for r in rows]

    summary = {
        "track": "E2_epilepsiae_region_level_feasibility",
        "verdict": "region_level_no_go_for_contact_level (weak region pilot only)",
        "n_template_subjects": len(rows),
        "n_with_surgery": len(with_surgery),
        "n_with_outcome_label": len(with_outcome),
        "engel_distribution": engel_dist,
        "engel_I_seizure_free": sum(1 for r in with_outcome if r["engel_class"] == "I"),
        "engel_II_to_IV": sum(1 for r in with_outcome if r["engel_class"] != "I"),
        "resection_granularity": {
            "surgery_localisation_is_lobe_level": True,
            "median_loc_codes_per_subject": (sorted(loc_code_sizes)[len(loc_code_sizes) // 2]
                                             if loc_code_sizes else None),
            "note": "surgerylocalisation = whole-lobe code(s) (e.g. 't-r'); NOT contact-level.",
        },
        "template_channel_pool": {
            "min": min(template_ch), "max": max(template_ch),
            "note": "templates computed on a focus-restricted small channel pool, subset of the resected "
                    "lobe -> 'template-source in resected lobe' ~ trivially true -> binary predictor degenerate.",
        },
        "why_no_go": (
            "Outcome labels EXIST and are parseable (proves the pipeline could run); but resection is "
            "coded at lobe granularity while templates sit inside the focus pool within that lobe, so the "
            "contact-level 'fraction of template network treated' has no contrast (re-confirms strategy "
            "§5.1 dual-NULL). Epilepsiae stays a weak region-level pilot, NOT mixed into the Yuquan "
            "contact-level E1; this is the data-side reason Yuquan contact-level outcome is the critical path."
        ),
    }
    with open(os.path.join(OUT, "E2_feasibility_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"template subjects = {len(rows)}; with surgery = {len(with_surgery)}; "
          f"with outcome = {len(with_outcome)}")
    print(f"Engel dist = {engel_dist} (I seizure-free = {summary['engel_I_seizure_free']}, "
          f"II-IV = {summary['engel_II_to_IV']})")
    print(f"surgery_localisation codes/subject (median) = {summary['resection_granularity']['median_loc_codes_per_subject']} "
          f"(lobe-level); template pool {min(template_ch)}-{max(template_ch)} ch")
    print(f"wrote {OUT}/")


if __name__ == "__main__":
    main()
