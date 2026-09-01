#!/usr/bin/env python3
"""Is patient-conditioned candidate choice load-bearing, or is one field enough?

Comparing each subject's own best candidate against a fixed one *on the split
that chose it* is circular: the minimum over twenty-four candidates beats a
fixed candidate by construction, even when every candidate is exchangeable
noise. This runs the comparison on the held-out split instead, reusing
simulations the confirmation stage already produced, so no extra compute and no
circularity.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "config/topic4_data_driven_snn_cohort_formal_v1.json"


def _aggregate_module():
    path = ROOT / "scripts/aggregate_topic4_data_driven_snn_cohort_formal.py"
    spec = importlib.util.spec_from_file_location("topic4_formal_agg", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["topic4_formal_agg"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--shared-candidate", default=None,
        help="defaults to the candidate the most subjects selected",
    )
    args = parser.parse_args()
    aggregate = _aggregate_module()
    config = json.loads(args.config.read_text())
    cohort = aggregate.Cohort(config)
    result = json.loads((cohort.output_root / "cohort_result.json").read_text())
    commit = aggregate.subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    counts = result["selected_candidate_counts"]
    shared = args.shared_candidate or max(counts, key=lambda key: (counts[key], key))
    seeds = [int(seed) for seed in config["search"]["confirmation_network_seeds"]]

    own = {row["subject_id"]: row for row in result["canonical_subjects"]}
    rows = []
    for index, subject in enumerate(cohort.subjects):
        subject_id = subject["subject_id"]
        forced = aggregate._confirm_layout(
            cohort, subject, index, shared, seeds, "canonical", commit,
        )
        mine = own[subject_id]
        rows.append({
            "subject_id": subject_id,
            "selected_candidate": mine["candidate_id"],
            "selected_delta": mine["delta_null_median_minus_observed"],
            "selected_pass": bool(mine["subject_endpoint_pass"]),
            "shared_delta": forced["delta_null_median_minus_observed"],
            "shared_pass": bool(forced["subject_endpoint_pass"]),
            "conditioning_gain": (
                mine["delta_null_median_minus_observed"]
                - forced["delta_null_median_minus_observed"]
            ),
            "already_the_shared_candidate": mine["candidate_id"] == shared,
        })

    gains = np.asarray([row["conditioning_gain"] for row in rows], float)
    free = [row for row in rows if not row["already_the_shared_candidate"]]
    free_gains = np.asarray([row["conditioning_gain"] for row in free], float)
    payload = {
        "schema_version": "topic4_cohort_conditioning_value_v1",
        "question": (
            "on held-out data, does each subject's own selected candidate beat "
            "one shared candidate applied to everybody"
        ),
        "why_not_the_training_split": (
            "the minimum over twenty-four candidates beats a fixed candidate on "
            "the split that chose it even under exchangeable noise, so that "
            "comparison cannot support a conditioning claim"
        ),
        "shared_candidate": shared,
        "n_subjects": len(rows),
        "n_subjects_whose_own_choice_is_the_shared_one": len(rows) - len(free),
        "held_out_gain": {
            "median": float(np.median(gains)),
            "n_positive": int(np.sum(gains > 0.0)),
            "wilcoxon_p": float(wilcoxon(gains).pvalue) if np.any(gains != 0) else None,
        },
        "held_out_gain_excluding_subjects_that_chose_the_shared_candidate": {
            "n": len(free),
            "median": float(np.median(free_gains)) if len(free_gains) else None,
            "n_positive": int(np.sum(free_gains > 0.0)),
            "wilcoxon_p": (
                float(wilcoxon(free_gains).pvalue)
                if len(free_gains) >= 6 and np.any(free_gains != 0) else None
            ),
        },
        "pass_fraction_selected": float(np.mean([row["selected_pass"] for row in rows])),
        "pass_fraction_shared": float(np.mean([row["shared_pass"] for row in rows])),
        "subjects": rows,
        "expected_commit": commit,
    }
    output = cohort.output_root / "conditioning_value_audit.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({
        key: payload[key] for key in (
            "shared_candidate", "held_out_gain",
            "held_out_gain_excluding_subjects_that_chose_the_shared_candidate",
            "pass_fraction_selected", "pass_fraction_shared",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
