#!/usr/bin/env python3
"""Does the relational message need this patient's own topology?

The H1 ladder showed that adding a graph message beats the leaky baseline.  That
alone is compatible with "any relational message helps".  Each null here keeps the
generator, adapter and parameter count fixed and changes only what the edges
connect, so the real graph has to beat them for the topology claim to stand.
"""
from __future__ import annotations

import argparse
import json

import collections

import numpy as np
import pandas as pd

from _common import (  # noqa: E402
    OUTPUT_ROOT, atomic_write_csv, atomic_write_json, code_revision, package_hash,
)
from src.topic5_epi_prssm.stats import holm, paired_effect  # noqa: E402

OUT = OUTPUT_ROOT / "graph_null"
ENDPOINTS = ("event_nll", "order_nll", "selection_nll", "participation_nll", "stop_nll")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="all34")
    parser.add_argument("--package", default="", help="12-char package hash to "
                        "pin; defaults to the current package")
    args = parser.parse_args()

    # A contrast between arms fitted by different code packages is not a contrast:
    # an audit found five package hashes across these arms and 14 duplicated
    # (arm, seed) pairs, with only the path-decomposition arms on the current
    # package. Pin one package, deduplicate, and say what was dropped.
    from src.topic5_epi_prssm.contracts import package_hash
    target_package = args.package or package_hash()[:12]

    candidates, dropped_package, seen = [], collections.Counter(), {}
    for path in sorted((OUT / "runs").glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("cohort") != args.cohort or not record.get("evaluation"):
            continue
        pkg = (record.get("package_hash") or "?")[:12]
        if pkg != target_package:
            dropped_package[f"{record['arm']}::{pkg}"] += 1
            continue
        key = (record["arm"], record["seed"])
        if key in seen:                      # same arm, same seed, same package
            continue
        seen[key] = record["job_id"]
        candidates.append(record)
    runs = candidates
    provenance = {
        "package_pinned_to": target_package,
        "n_runs_kept": len(runs),
        "n_runs_dropped_wrong_package": int(sum(dropped_package.values())),
        "dropped_by_arm_and_package": dict(dropped_package),
        "deduplicated_to_one_run_per": "arm x seed x package",
        "why": "arms fitted by different packages are not comparable, and the same "
               "arm and seed appearing twice would be silently averaged",
    }
    if not runs:
        atomic_write_json(OUT / "GRAPH_NULL_EVIDENCE_CARD.json",
                          {"status": "NO_COMPLETED_RUN"})
        print("no completed graph-null run")
        return

    rows = []
    for record in runs:
        for subject, values in record["evaluation"]["filtered"].items():
            for endpoint in ENDPOINTS:
                if endpoint in values:
                    rows.append({"arm": record["arm"], "seed": record["seed"],
                                 "subject": subject, "endpoint": endpoint,
                                 "value": float(values[endpoint])})
    frame = pd.DataFrame(rows)
    atomic_write_csv(OUT / "graph_null_per_patient.csv", frame)

    # seeds are pooled inside a patient first, then patients are the unit
    per = (frame.groupby(["endpoint", "arm", "subject"]).value.median()
           .reset_index())
    contrasts, family = {}, {}
    for endpoint in sorted(per.endpoint.unique()):
        block = per[per.endpoint == endpoint]
        by_arm = {arm: dict(zip(g.subject, g.value)) for arm, g in block.groupby("arm")}
        if "real" not in by_arm:
            continue
        for arm, values in sorted(by_arm.items()):
            if arm == "real":
                continue
            shared = sorted(set(values) & set(by_arm["real"]))
            if len(shared) < 5:
                continue
            effect = paired_effect({s: by_arm["real"][s] for s in shared},
                                   {s: values[s] for s in shared},
                                   label=f"{endpoint}::real-vs-{arm}")
            contrasts[f"{endpoint}::real-vs-{arm}"] = effect.as_dict()
            if endpoint == "event_nll":
                family[f"real-vs-{arm}"] = effect.sign_test_p

    # the 2x2: the real graph feeds both the slow generator and the within-event
    # decoder, so a wholesale shuffle cannot say which path needed it
    by_path = {}
    for record in runs:
        by_path.setdefault(record.get("graph_path", "both"), set()).add(record["arm"])
    paths_present = {k: sorted(v) for k, v in by_path.items()}

    budgets = {r["arm"]: r.get("edge_budget") for r in runs}
    exact = [a for a, b in budgets.items()
             if b and budgets.get("real") and
             abs(b["total_forward_edges"] - budgets["real"]["total_forward_edges"]) < 1e-6]
    card = {
        "contract": "topic5_epi_prssm_v0_1_graph_null_card",
        "question": "does the relational message need this patient's own topology, or "
                    "does any relational message of the same size do as well?",
        "status": "EXPLORATORY_DEVELOPMENT",
        "primary_endpoint": "event_nll",
        "contrasts": contrasts,
        "holm_corrected_event_nll": holm(family),
        "run_provenance": provenance,
        "edge_budget_by_arm": budgets,
        "arms_by_graph_path": paths_present,
        "path_decomposition_note":
            "an arm labelled <null>@generator shuffled only the graph the slow state "
            "propagates along and left the decoder's spatial prior intact; @decoder is "
            "the reverse; an unlabelled arm shuffled both and therefore cannot "
            "attribute the loss to either path",
        "graph_density_caveat":
            "491 of 492 contacts already have out-degree N-1, so these graphs are "
            "nearly complete and a degree-preserving rewire permutes edge weights "
            "rather than changing binary wiring. The supported claim is about "
            "patient-specific weighted spatial relations with correct contact-identity "
            "alignment, not about physical topology.",
        "edge_budget_matched_arms": sorted(exact),
        "reading": "a negative median means the real graph predicted better than the null. "
                   "Only the arms listed in edge_budget_matched_arms carry the same number "
                   "of edges as the real graph; the others differ in budget as well as in "
                   "topology and are weaker evidence.",
        "denominators": {"n_runs": len(runs),
                         "arms": sorted({r["arm"] for r in runs}),
                         "seeds": sorted({r["seed"] for r in runs}),
                         "n_patients": int(per.subject.nunique())},
        "claim_boundary": [
            "beating a null shows topology specificity, not that the edges are synapses",
            "development-partition result; no untouched-test claim is made here",
        ],
        "code_revision": code_revision(), "package_hash": package_hash(),
    }
    atomic_write_json(OUT / "GRAPH_NULL_EVIDENCE_CARD.json", card)
    print(json.dumps({k: {kk: v[kk] for kk in ("median_delta", "n_favourable",
                                               "n_patients", "sign_test_p")}
                      for k, v in contrasts.items() if k.startswith("event_nll")},
                     indent=1)[:1200])


if __name__ == "__main__":
    main()
