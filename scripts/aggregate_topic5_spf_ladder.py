#!/usr/bin/env python3
"""Fail-closed patient-first aggregation of SPF ladder shards.

The aggregator refuses incomplete Cartesian products, mixed configurations,
unfinished run states, duplicate subject/seed/model rows, or leakage flags.
Seeds are folded within patient before model comparisons.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_shared_propagation_field import (  # noqa: E402
    CONTRACT_NAME,
    sha256_file,
)

LADDER = [
    "m0_static",
    "m1_markov",
    "m1_markov_phase",
    "m2_markov_mixture",
    "m2_markov_mixture_phase",
    "m3_template",
    "m4_field",
    "m4_field_phase",
]
ADEQUATE = {"CONVERGED", "NO_FREE_PARAMETERS"}


def _expected(config_path: Path) -> tuple[list[str], list[int], str]:
    config = yaml.safe_load(config_path.read_text())
    pilot_path = ROOT / config["outputs"]["phase0"] / "pilot_subjects_target_blind.csv"
    subjects = [
        line.split(",")[0]
        for line in pilot_path.read_text().strip().splitlines()[1:]
        if line.strip()
    ]
    return subjects, [int(seed) for seed in config["training"]["seeds"]], sha256_file(
        config_path
    )


def collect(
    root: Path,
    *,
    config_path: Path,
    allow_partial: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected_subjects, expected_seeds, config_sha = _expected(config_path)
    summaries = sorted(root.glob("*/summary.json"))
    if not summaries:
        raise SystemExit(f"no ladder shards under {root}")
    seen_runs: set[tuple[str, int]] = set()
    expected_source_fingerprint: str | None = None
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text())
        state_path = summary_path.with_name("run_state.json")
        if not state_path.exists() or json.loads(state_path.read_text()).get(
            "status"
        ) != "COMPLETE":
            raise SystemExit(f"incomplete shard: {summary_path.parent}")
        if summary.get("config_sha256") != config_sha:
            raise SystemExit(f"mixed/stale config: {summary_path}")
        source_fingerprint = json.dumps(
            summary.get("source_provenance", {}).get("source_sha256", {}),
            sort_keys=True,
        )
        if source_fingerprint == "{}":
            raise SystemExit(f"missing source fingerprint: {summary_path}")
        if expected_source_fingerprint is None:
            expected_source_fingerprint = source_fingerprint
        elif source_fingerprint != expected_source_fingerprint:
            raise SystemExit(f"mixed source versions: {summary_path}")
        if any(
            bool(summary.get(flag, True))
            for flag in (
                "old_heldout20_scored",
                "ictal_target_read",
                "ab_or_axis_label_read",
                "geometry_input_read",
            )
        ):
            raise SystemExit(f"leakage flag failed: {summary_path}")
        run_key = (str(summary["subject"]), int(summary["seed"]))
        if run_key in seen_runs:
            raise SystemExit(f"duplicate subject/seed shard: {run_key}")
        seen_runs.add(run_key)
        if set(summary["models"]) != set(LADDER):
            raise SystemExit(f"incomplete model ladder: {summary_path}")
        for name in LADDER:
            payload = summary["models"][name]
            rows.append(
                {
                    "subject": summary["subject"],
                    "dataset": summary["dataset"],
                    "seed": summary["seed"],
                    "model": name,
                    "scientific_role": payload["scientific_role"],
                    "n_contacts": summary["n_contacts"],
                    "n_inner_train_events": summary["n_inner_train_events"],
                    "n_monitor_validation_events": summary[
                        "n_monitor_validation_events"
                    ],
                    "n_development_test_events": summary[
                        "n_development_test_events"
                    ],
                    "development_test_mean_decisions": summary[
                        "development_test_mean_decisions"
                    ],
                    "n_trainable_parameters": payload["n_trainable_parameters"],
                    "likelihood_estimator": payload[
                        "complete_event_likelihood_estimator"
                    ],
                    "test_nll_per_event": payload[
                        "development_test_nll_per_event"
                    ],
                    "test_nll_per_event_mc_sd": payload[
                        "development_test_nll_per_event_mc_sd"
                    ],
                    "test_nll_per_decision": payload[
                        "development_test_nll_per_decision"
                    ],
                    "test_nll_per_decision_mc_sd": payload[
                        "development_test_nll_per_decision_mc_sd"
                    ],
                    "prior_predictive_nll_per_decision": payload[
                        "prior_predictive_nll_per_decision"
                    ],
                    "adequacy": payload["training_adequacy"]["verdict"],
                    "best_epoch": payload["training_adequacy"]["best_epoch"],
                    "n_epochs": payload["training_adequacy"]["n_epochs"],
                    "relative_improvement": payload["training_adequacy"].get(
                        "relative_improvement_from_initial", np.nan
                    ),
                    "rescue_used": bool(
                        payload["training_adequacy"].get("rescue_used", False)
                    ),
                    "precedence_correlation": payload["repertoire"][
                        "precedence_correlation"
                    ],
                    "precedence_correlation_rollout_sd": payload[
                        "repertoire_rollout_sd"
                    ]["precedence_correlation"],
                    "precedence_mae": payload["repertoire"]["precedence_mae"],
                    "participation_mae": payload["repertoire"][
                        "participation_mae"
                    ],
                    "rank_wasserstein": payload["repertoire"][
                        "rank_wasserstein"
                    ],
                    "step_nll_json": json.dumps(
                        payload["step_nll_per_decision_diagnostic"]
                    ),
                    "source_sha256_json": source_fingerprint,
                }
            )
    expected_runs = {
        (subject, seed)
        for subject in expected_subjects
        for seed in expected_seeds
    }
    missing = sorted(expected_runs.difference(seen_runs))
    unexpected = sorted(seen_runs.difference(expected_runs))
    if not allow_partial and (missing or unexpected):
        raise SystemExit(
            f"shard Cartesian product mismatch; missing={missing}, "
            f"unexpected={unexpected}"
        )
    frame = pd.DataFrame(rows)
    if frame.duplicated(["subject", "seed", "model"]).any():
        raise SystemExit("duplicate subject/seed/model rows")
    return frame


def _comparison(
    wide: pd.DataFrame,
    usable_subjects: pd.Index,
    left: str,
    right: str,
) -> dict[str, Any]:
    delta_all = wide[left] - wide[right]
    delta = delta_all.reindex(usable_subjects).dropna()
    return {
        "contrast": f"{left}_minus_{right}",
        "negative_means_left_better": True,
        "n_patients_all": int(delta_all.notna().sum()),
        "n_patients_adequate": int(delta.notna().sum()),
        "median_delta_nll_per_decision_adequate": (
            float(np.median(delta)) if len(delta) else None
        ),
        "n_patients_left_lower_adequate": int((delta < 0).sum()),
        "per_patient_delta": {
            str(subject): (None if np.isnan(value) else float(value))
            for subject, value in delta_all.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate SPF ladder shards")
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT
        / "results/topic5_shared_propagation_field/development/ladder_pilot_v0_4",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_shared_propagation_field_v0_1.yaml",
    )
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    root = args.root
    runs = collect(
        root,
        config_path=args.config,
        allow_partial=bool(args.allow_partial),
    )
    runs.to_csv(root / "ladder_runs.csv", index=False)

    per_patient = (
        runs.groupby(["subject", "dataset", "model"], as_index=False)
        .agg(
            test_nll_per_event=("test_nll_per_event", "mean"),
            test_nll_per_decision=("test_nll_per_decision", "mean"),
            nll_seed_sd=("test_nll_per_decision", "std"),
            mc_sd_mean=("test_nll_per_decision_mc_sd", "mean"),
            precedence_correlation=("precedence_correlation", "mean"),
            precedence_rollout_sd=(
                "precedence_correlation_rollout_sd",
                "mean",
            ),
            precedence_mae=("precedence_mae", "mean"),
            n_seeds=("seed", "nunique"),
            n_trainable_parameters=("n_trainable_parameters", "max"),
            n_inadequate=("adequacy", lambda values: int(sum(v not in ADEQUATE for v in values))),
            n_rescued=("rescue_used", "sum"),
            mean_decisions=("development_test_mean_decisions", "mean"),
        )
        .sort_values(["subject", "model"])
    )
    per_patient.to_csv(root / "ladder_per_patient.csv", index=False)

    wide = per_patient.pivot(
        index="subject", columns="model", values="test_nll_per_decision"
    )
    inadequate = per_patient.groupby("subject")["n_inadequate"].sum() > 0
    usable_subjects = inadequate.index[~inadequate]
    contrasts = [
        ("m4_field", "m0_static"),
        ("m4_field", "m1_markov"),
        ("m4_field", "m1_markov_phase"),
        ("m4_field", "m2_markov_mixture"),
        ("m4_field", "m2_markov_mixture_phase"),
        ("m4_field", "m3_template"),
        ("m4_field_phase", "m3_template"),
    ]
    comparisons = {
        f"{left}_minus_{right}": _comparison(
            wide, usable_subjects, left, right
        )
        for left, right in contrasts
    }

    gap = (wide["m3_template"] - wide["m4_field"]).dropna()
    decision_length = (
        per_patient[per_patient["model"] == "m4_field"]
        .set_index("subject")["mean_decisions"]
        .reindex(gap.index)
    )
    clock_probe = {
        "metric": "development_test_nll_per_decision",
        "per_patient": {
            str(subject): {
                "mean_suffix_decisions": float(decision_length.loc[subject]),
                "m3_minus_m4_nll_per_decision": float(gap.loc[subject]),
            }
            for subject in gap.index
        },
        "spearman_decisions_vs_gap": (
            float(decision_length.corr(gap, method="spearman"))
            if len(gap) >= 3
            else None
        ),
        "resolution": (
            "Use the explicit phase-matched M1/M2 controls and the "
            "nonautonomous m4_field_phase diagnostic; correlation alone is "
            "not a mechanism verdict."
        ),
    }

    payload = {
        "contract": CONTRACT_NAME,
        "status": "DEVELOPMENT_PILOT_NO_GATE_VERDICT",
        "score_partition": "untouched_development_test_within_old_train80",
        "unit": "patient_after_seed_folding",
        "models": LADDER,
        "n_subjects": int(runs["subject"].nunique()),
        "n_seeds": int(runs["seed"].nunique()),
        "source_sha256": json.loads(runs["source_sha256_json"].iloc[0]),
        "adequacy_counts": runs.groupby(["model", "adequacy"])
        .size()
        .unstack(fill_value=0)
        .to_dict(orient="index"),
        "rescue_counts": {
            str(model): int(count)
            for model, count in runs.groupby("model")["rescue_used"].sum().items()
        },
        "n_subjects_all_models_adequate": int((~inadequate).sum()),
        "subjects_with_an_inadequate_fit": sorted(
            inadequate[inadequate].index.tolist()
        ),
        "nll_per_decision_by_patient": {
            str(subject): {
                str(model): float(value)
                for model, value in row.items()
                if not np.isnan(value)
            }
            for subject, row in wide.iterrows()
        },
        "comparisons": comparisons,
        "m3_clock_confound_probe": clock_probe,
        "claim_boundary": (
            "m4_field_phase is a nonautonomous clock diagnostic. Its success "
            "cannot support the autonomous shared-field claim."
        ),
        "gate_status": {
            "g0_snn_identifiability": "EXISTING_ARTIFACT_REUSE_AUDIT",
            "g1_full_event_generation": "PILOT_ONLY_NOT_JUDGED",
            "g2_stable_structure": "LOCKED_NOT_RUN",
            "g3_one_structure_many_trajectories": "LOCKED_NOT_RUN",
        },
    }
    (root / "ladder_cohort_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    (root / "LADDER_PILOT_STATE.json").write_text(
        json.dumps(
            {
                "contract": CONTRACT_NAME,
                "status": "COMPLETE",
                "n_subjects": payload["n_subjects"],
                "n_seeds": payload["n_seeds"],
                "n_models": len(LADDER),
                "config_sha256": sha256_file(args.config),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(payload["adequacy_counts"], indent=2))
    print(wide.round(4).to_string())
    print(f"wrote {root / 'ladder_cohort_summary.json'}")


if __name__ == "__main__":
    main()
