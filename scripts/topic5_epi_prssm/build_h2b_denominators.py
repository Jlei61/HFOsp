#!/usr/bin/env python3
"""Task A: the 34 -> 27 -> 361 -> 203 denominator flow, stated so that no number
can be mistaken for another.

The specific hazard this exists to prevent: 203 is the count of seizures whose
pre-ictal window was actually *observed*, and it is not the number of seizures in
the cohort, nor the number analysed.  Writing "203 seizures" without the chain
above it silently converts an observability filter into a study size.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from _common import (  # noqa: E402
    OUTPUT_ROOT, atomic_write_csv, atomic_write_json, code_revision, package_hash,
    resolve_cohort,
)

H2B = OUTPUT_ROOT / "seizure_link_preictal"
CROSSWALK = OUTPUT_ROOT / "seizure_crosswalk"
OUT = OUTPUT_ROOT / "h2b_denominators"

STRATA = ["none", "1to4", "5to19", "ge20"]
STRATA_LABEL = {"none": "0 IED", "1to4": "1-4 IED", "5to19": "5-19 IED", "ge20": ">=20 IED"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", default="linear_graph_recurrent")
    parser.add_argument("--leads", nargs="+", default=["lead5m", "lead15m", "lead30m",
                                                        "lead60m"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cohort = list(resolve_cohort("all34"))
    rows, flows = [], []
    for lead in args.leads:
        path = H2B / f"preictal_effects__{args.layer}__{lead}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        premise = frame.preictal_observation_premise_met.astype(bool)
        denominators = json.loads(
            (H2B / f"preictal_denominators__{args.layer}__{lead}.csv").read_text()
        ) if False else None    # denominators live in the evidence card, read below

        flow = {
            "lead": lead,
            "step_1_cohort_patients": len(cohort),
            "step_2_patients_with_any_analysable_seizure": int(frame.subject.nunique()),
            "step_3_seizures_eligible_all": int(len(frame)),
            "step_4_seizures_meeting_observation_premise": int(premise.sum()),
            "step_4_patients_meeting_observation_premise": int(
                frame[premise].subject.nunique()),
            "share_of_eligible_that_were_observed": float(premise.mean()),
        }
        for name in STRATA:
            flow[f"stratum_{name}"] = int((frame.lookback_stratum == name).sum())
            flow[f"stratum_{name}_premise_met"] = int(
                ((frame.lookback_stratum == name) & premise).sum())
        flows.append(flow)

        for _, r in frame.iterrows():
            rows.append({"lead": lead, "subject": r.subject, "dataset": r.dataset,
                         "seizure_id": r.seizure_id,
                         "lookback_stratum": r.lookback_stratum,
                         "n_events_in_lookback_2h": r.n_events_in_lookback_2h,
                         "premise_met": bool(r.preictal_observation_premise_met),
                         "anchor_gap_to_cutoff_seconds": r.anchor_gap_to_cutoff_seconds,
                         "coverage": r.get("nuisance_coverage", np.nan)})

    table = pd.DataFrame(flows)
    atomic_write_csv(OUT / f"denominator_flow__{args.layer}.csv", table)
    atomic_write_csv(OUT / f"per_seizure__{args.layer}.csv", pd.DataFrame(rows))

    crosswalk_summary = {}
    cw = CROSSWALK / f"CROSSWALK_SUMMARY__{args.layer}__lead30m.json"
    if cw.exists():
        crosswalk_summary = json.loads(cw.read_text())

    card = {
        "contract": "topic5_epi_prssm_h2b_denominators",
        "layer": args.layer,
        "reading_order": [
            "34 patients entered the cohort",
            "27 have at least one seizure inside the recorded span with an analysable "
            "pre-ictal window",
            "361 seizures are eligible at the primary lead -- this is the population "
            "layer and the denominator for the main analysis",
            "203 of those additionally met the observation premise, meaning the "
            "observer actually saw enough discharges before the cut-off -- this is a "
            "high-observability SENSITIVITY layer, not the study size",
        ],
        "do_not_write": [
            "203 seizures were analysed",
            "the cohort had 203 seizures",
            "203/34 patients",
        ],
        "flow_by_lead": flows,
        "strata_definition": {
            "counted_over": "IED observed in the 2 h look-back before the cut-off",
            "labels": STRATA_LABEL,
        },
        "crosswalk": {k: crosswalk_summary.get(k) for k in (
            "n_h2b_seizures", "n_matched_and_verified", "n_unmatched", "n_ambiguous",
            "n_matched_but_timestamp_off", "by_route", "by_dataset",
            "onset_difference_seconds", "subtype_coverage")},
        "code_revision": code_revision(), "package_hash": package_hash(),
    }
    atomic_write_json(OUT / f"H2B_DENOMINATORS__{args.layer}.json", card)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
