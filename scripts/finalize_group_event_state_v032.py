#!/usr/bin/env python3
"""Aggregate the v0.3.2 development runs without changing scientific denominators.

The script is deliberately read-only with respect to model artifacts.  It
combines seeds within patient, preserves pre-model eligibility, and emits the
payload consumed by the established Group-Event State figure producer.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


SUBJECTS = ("epilepsiae_1146", "yuquan_pengzihang", "yuquan_zhangkexuan")
SEEDS = (20260902, 20260903, 20260904)
ALIASES = {subject: f"P{i + 1}" for i, subject in enumerate(SUBJECTS)}
HORIZONS = (("300s", 5), ("1800s", 30), ("7200s", 120))
H2A_ENDPOINTS = (
    ("continue", "continue"),
    ("positive_size", "positive_size"),
    ("subset_identity", "subset"),
    ("later_continuation", "later_continuation"),
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _finite(values: list[Any]) -> list[float]:
    out = []
    for value in values:
        if value is None:
            continue
        value = float(value)
        if np.isfinite(value):
            out.append(value)
    return out


def _mean(values: list[Any]) -> float | None:
    values = _finite(values)
    return float(np.mean(values)) if values else None


def _range(values: list[Any]) -> list[float] | None:
    values = _finite(values)
    return [float(min(values)), float(max(values))] if values else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _eligibility_for(patient: dict[str, Any], minutes: int) -> dict[str, Any]:
    key = {
        5: "count_5min_short_range_only",
        30: "count_30min_primary",
        120: "count_120min_secondary",
    }[minutes]
    return patient["eligibility"][key]


def _h1_summary(data_root: Path, eligibility: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    figure_rows: list[dict[str, Any]] = []
    per_subject: dict[str, Any] = {}
    for subject in SUBJECTS:
        subject_rows = []
        for horizon_key, minutes in HORIZONS:
            seed_rows = []
            for seed in SEEDS:
                path = data_root / "evaluation/h1" / subject / f"h1_result_seed_{seed}.json"
                if not path.exists():
                    continue
                result = _load(path)
                horizon = result["horizons"].get(horizon_key, {})
                if horizon.get("status") != "ok":
                    continue
                variant = horizon["variants"]["H_strong"]
                paired = variant["shared_H_alpha"]["paired"]["dev_test"]
                if paired.get("status") != "ok":
                    continue
                pairs = paired["pairs"]
                sensitivity = variant["per_arm"]["paired"]["dev_test"]["pairs"]
                seed_rows.append(
                    {
                        "seed": seed,
                        "residual_gain_over_history": pairs["H+S_correct_vs_H"]["mean_gain"],
                        "correct_time_gain_over_shifted": pairs["H+S_correct_vs_H+S_shifted_mean"]["mean_gain"],
                        "dynamic_gain_over_mean": pairs["H+S_correct_vs_H+S_mean"]["mean_gain"],
                        "n_score_blocks": paired["n_blocks"],
                        "per_arm_dispersion_sensitivity": {
                            "residual_gain_over_history": sensitivity["H+S_correct_vs_H"]["mean_gain"],
                            "correct_time_gain_over_shifted": sensitivity["H+S_correct_vs_H+S_shifted_mean"]["mean_gain"],
                            "dynamic_gain_over_mean": sensitivity["H+S_correct_vs_H+S_mean"]["mean_gain"],
                        },
                    }
                )
            eligibility_row = _eligibility_for(eligibility["subjects"][subject], minutes)
            row = {
                "subject": subject,
                "alias": ALIASES[subject],
                "horizon_minutes": minutes,
                "eligible": bool(eligibility_row["eligible"]),
                "eligibility_reasons": eligibility_row.get("reasons", []),
                "n_seeds_scored": len(seed_rows),
                "residual_gain_over_history": _mean([r["residual_gain_over_history"] for r in seed_rows]),
                "correct_time_gain_over_shifted": _mean([r["correct_time_gain_over_shifted"] for r in seed_rows]),
                "dynamic_gain_over_mean": _mean([r["dynamic_gain_over_mean"] for r in seed_rows]),
                "n_score_blocks": int(np.median([r["n_score_blocks"] for r in seed_rows])) if seed_rows else 0,
                "seed_ranges": {
                    key: _range([r[key] for r in seed_rows])
                    for key in (
                        "residual_gain_over_history",
                        "correct_time_gain_over_shifted",
                        "dynamic_gain_over_mean",
                    )
                },
                "seed_rows": seed_rows,
            }
            figure_rows.append({k: v for k, v in row.items() if k != "seed_rows"})
            subject_rows.append(row)
        per_subject[subject] = subject_rows
    return figure_rows, per_subject


def _h2a_summary(data_root: Path, eligibility: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    figure_rows: list[dict[str, Any]] = []
    per_subject: dict[str, Any] = {}
    for subject in SUBJECTS:
        seed_results = []
        for seed in SEEDS:
            path = data_root / "evaluation/h2a" / subject / f"h2a_result_seed_{seed}.json"
            if path.exists():
                seed_results.append(_load(path))
        endpoint_rows = []
        for result_name, figure_name in H2A_ENDPOINTS:
            seed_rows = []
            for result in seed_results:
                pairs = result["paired"]["dev_test"][result_name]
                history = pairs["H+S_correct_vs_H"]["mean_gain"]
                mean_state = pairs["H+S_correct_vs_H+S_mean"]["mean_gain"]
                shifted_keys = [f"H+S_correct_vs_H+S_shifted:{i}" for i in range(1, 6)]
                shifted = _mean([pairs[key]["mean_gain"] for key in shifted_keys])
                seed_rows.append(
                    {
                        "seed": int(result["seed"]),
                        "gain_over_history": history,
                        "gain_over_shifted": shifted,
                        "gain_over_mean": mean_state,
                        "gain_over_best_control": min(history, shifted, mean_state),
                        "n_score_blocks": min(
                            pairs["H+S_correct_vs_H"]["n_blocks"],
                            pairs["H+S_correct_vs_H+S_mean"]["n_blocks"],
                            *(pairs[key]["n_blocks"] for key in shifted_keys),
                        ),
                    }
                )
            eligible = bool(
                eligibility["subjects"][subject]["eligibility"]["h2a_positive_k_prefix"]["eligible"]
            )
            row = {
                "subject": subject,
                "alias": ALIASES[subject],
                "horizon_minutes": 30,
                "endpoint": figure_name,
                "eligible": eligible,
                "n_seeds_scored": len(seed_rows),
                "gain_over_history": _mean([r["gain_over_history"] for r in seed_rows]),
                "gain_over_shifted": _mean([r["gain_over_shifted"] for r in seed_rows]),
                "gain_over_mean": _mean([r["gain_over_mean"] for r in seed_rows]),
                "gain_over_best_control": _mean([r["gain_over_best_control"] for r in seed_rows]),
                "n_score_blocks": int(np.median([r["n_score_blocks"] for r in seed_rows])) if seed_rows else 0,
                "seed_ranges": {
                    key: _range([r[key] for r in seed_rows])
                    for key in ("gain_over_history", "gain_over_shifted", "gain_over_mean", "gain_over_best_control")
                },
                "seed_rows": seed_rows,
            }
            endpoint_rows.append(row)
            if figure_name in {"continue", "positive_size", "subset"}:
                figure_rows.append({k: v for k, v in row.items() if k != "seed_rows"})

        budget_rows = []
        for result in seed_results:
            arms = result["fit"]
            budget_rows.append(
                {
                    "seed": int(result["seed"]),
                    "n_arms": len(arms),
                    "n_selected_at_budget_edge": sum(bool(v["selected_at_budget_edge"]) for v in arms.values()),
                    "selected_epochs": {arm: int(v["selected_epoch"]) for arm, v in arms.items()},
                }
            )
        per_subject[subject] = {"endpoints": endpoint_rows, "optimisation": budget_rows}
    return figure_rows, per_subject


def _model_summary(data_root: Path) -> dict[str, Any]:
    per_subject: dict[str, Any] = {}
    for subject in SUBJECTS:
        seed_rows = []
        for seed in SEEDS:
            run_dir = data_root / "model/runs/leaky_bank" / subject / f"seed_{seed}"
            evaluation_path = run_dir / "evaluation.json"
            result_path = run_dir / "result.json"
            if not evaluation_path.exists() or not result_path.exists():
                continue
            evaluation = _load(evaluation_path)
            result = _load(result_path)
            contrasts = evaluation["phases"]["dev_test"]["contrasts"]
            seed_rows.append(
                {
                    "seed": seed,
                    "selected_step": result["selected_step"],
                    "selected_at_budget_edge": result["selected_at_budget_edge"],
                    "selected_first_validation": result["selected_first_validation"],
                    "peak_gpu_memory_bytes": result["peak_gpu_memory_bytes"],
                    "H_minus_correct": contrasts["h_minus_correct"]["mean"],
                    "intercept_minus_correct": contrasts["intercept_minus_correct"]["mean"],
                    "shifted_minus_correct": contrasts["shifted_minus_correct"]["mean"],
                    "mean_minus_correct": contrasts["mean_minus_correct"]["mean"],
                    "random_minus_correct": contrasts["random_minus_correct"]["mean"],
                }
            )
        per_subject[subject] = {
            "n_seeds": len(seed_rows),
            "seed_rows": seed_rows,
            "mean_contrasts": {
                key: _mean([row[key] for row in seed_rows])
                for key in (
                    "H_minus_correct",
                    "intercept_minus_correct",
                    "shifted_minus_correct",
                    "mean_minus_correct",
                    "random_minus_correct",
                )
            },
            "n_selected_at_budget_edge": sum(row["selected_at_budget_edge"] for row in seed_rows),
            "n_selected_first_validation": sum(row["selected_first_validation"] for row in seed_rows),
        }
    return per_subject


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_2"))
    parser.add_argument("--output-root", type=Path, default=Path("results/group_event_state/v0_3_2"))
    parser.add_argument("--allow-incomplete-h2a", action="store_true")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    args.output_root.mkdir(parents=True, exist_ok=True)
    eligibility_path = args.data_root / "measurement/endpoint_eligibility.json"
    eligibility = _load(eligibility_path)
    h1_rows, h1 = _h1_summary(args.data_root, eligibility)
    h2a_rows, h2a = _h2a_summary(args.data_root, eligibility)
    h2a_runs = sum(len(v["optimisation"]) for v in h2a.values())
    if h2a_runs != len(SUBJECTS) * len(SEEDS) and not args.allow_incomplete_h2a:
        raise RuntimeError(f"H2a incomplete: found {h2a_runs}/9 patient-seed results")

    positive_path = args.data_root / "model/synthetic/leaky_bank/epilepsiae_1146/positive/judgement.json"
    null_path = args.data_root / "model/synthetic/leaky_bank/epilepsiae_1146/null/judgement.json"
    positive = _load(positive_path)
    null = _load(null_path)
    ladder_path = (
        args.data_root
        / "model/synthetic_sensitivity/leaky_bank/epilepsiae_1146/summary.json"
    )
    ladder = _load(ladder_path) if ladder_path.exists() else None
    binary_recovery_nonmonotonic = False
    continuous_gain_monotonic = None
    if ladder is not None:
        ordered = sorted(ladder["rows"], key=lambda row: row["beta"])
        binary_recovery_nonmonotonic = any(
            low["pass"] and not high["pass"]
            for i, low in enumerate(ordered)
            for high in ordered[i + 1 :]
        )
        gains = [float(row["median_gain_nats"]) for row in ordered]
        continuous_gain_monotonic = all(high >= low for low, high in zip(gains, gains[1:]))
    registry_path = args.data_root / "shared/frozen_state_registry.json"
    registry = _load(registry_path)
    source_commit = _git_head(repo_root)
    summary = {
        "format": "group_event_state_v0_3_2_closeout_summary",
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "V0_3_2_PIPELINE_ACCEPTED_ASSAY_POWER_UNCALIBRATED_CLOSEOUT",
        "scientific_status": {
            "pipeline": "accepted",
            "assay_power": "uncalibrated",
            "h1": "inconclusive_n1",
            "h2a": "inconclusive_objective_mismatch",
            "h2b": "not_run",
            "h3": "not_run",
        },
        "source_commit": source_commit,
        "sealed_partition_opened": False,
        "subjects": list(SUBJECTS),
        "seeds": list(SEEDS),
        "measurement": {
            "n_patients": eligibility["n_patients"],
            "eligibility_sha256": _sha256(eligibility_path),
            "count_30min_eligible_subjects": [
                s for s in SUBJECTS
                if eligibility["subjects"][s]["eligibility"]["count_30min_primary"]["eligible"]
            ],
            "h2a_eligible_subjects": [
                s for s in SUBJECTS
                if eligibility["subjects"][s]["eligibility"]["h2a_positive_k_prefix"]["eligible"]
            ],
        },
        "instrument": {
            "positive": positive,
            "null": null,
            "effect_size_ladder": ladder,
            "binary_recovery_pass_count_nonmonotonic": binary_recovery_nonmonotonic,
            "continuous_median_gain_monotonic": continuous_gain_monotonic,
            "admissible_for_human_state_claim": bool(positive["pass"] and null["pass"]),
            "decision": (
                "the pipeline is complete, but positive-recovery power is uncalibrated: "
                "continuous median gain rises with beta while a three-replicate CI-based pass "
                "count fluctuates; human results are development diagnostics, not evidence for "
                "presence or absence of a residual state"
            ),
        },
        "state_registry": {
            "path": str(registry_path),
            "sha256": _sha256(registry_path),
            "status": registry["status"],
            "n_complete_entries": registry["n_complete_entries"],
        },
        "model_side": _model_summary(args.data_root),
        "h1": {
            "primary_horizon_minutes": 30,
            "primary_history": "H_strong",
            "primary_dispersion_mode": "shared_H_alpha",
            "patient_first_aggregation": "mean within patient across three seeds",
            "per_subject": h1,
            "decision": (
                "inconclusive (N=1 eligible): the only pre-eligible 30-min patient favours H "
                "over the current count-trained H+S representation"
            ),
        },
        "h2a": {
            "primary_endpoint": "subset_identity_given_K_and_prefix",
            "patient_first_aggregation": "mean within patient across three seeds",
            "per_subject": h2a,
            "decision": (
                "inconclusive objective mismatch: a state trained only on future count did not "
                "transfer stably to grammar; best-control is sensitivity only, while H and shifted "
                "comparisons remain the interpretable contrasts"
            ),
        },
        "h2b_h3": {
            "status": "not_run_by_v0_3_2_contract",
            "interpretation": "not negative and not a missing implementation in this closeout",
        },
        "claim_boundary": {
            "allowed": [
                "the v0.3.2 paired residual-state pipeline runs end to end on three development patients",
                "no false positive was observed in six null sanity-check replicates",
                "positive-recovery power is not calibrated by the current small synthetic experiment",
                "the current count-trained representation is unsupported in the only eligible 30-min patient",
            ],
            "forbidden": [
                "a slow physiological state was found",
                "a slow physiological state was excluded",
                "the frozen state transfers to seizure risk",
                "IEDs causally shape the state",
            ],
        },
    }
    summary_path = args.output_root / "v0_3_2_closeout_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))

    figure_payload = {
        "format": "group_event_state_core_evidence_v2",
        "status": "v0_3_2_pipeline_accepted_assay_power_uncalibrated",
        "source": {
            "summary_format": summary["format"],
            "source_commit": source_commit,
            "sealed_partition_opened": False,
            "model_layer_nested": True,
            "measurement_layer_nested": True,
        },
        "horizons_minutes": [5, 30, 120],
        "training": [
            {
                "subject": subject,
                "alias": ALIASES[subject],
                "optimization_status": "development_diagnostic",
                "selected_epochs": [],
                "n_seeds": len(h2a[subject]["optimisation"]),
            }
            for subject in SUBJECTS
        ],
        "v0_3_1_diagnostics": {"status": "archival_not_primary_estimand", "count_rows": [], "mark_rows": []},
        "h1_future_block": {
            "status": "inconclusive_assay_power_uncalibrated_n1",
            "rows": h1_rows,
            "gain_definition": "control NB NLL minus H+S_correct NLL; positive favours residual state",
            "required_fields": [
                "subject", "horizon_minutes", "residual_gain_over_history",
                "correct_time_gain_over_shifted", "dynamic_gain_over_mean", "n_score_blocks",
            ],
        },
        "h2a_repertoire": {
            "status": "inconclusive_objective_mismatch_count_trained_state",
            "rows": h2a_rows,
            "gain_definition": (
                "control NLL minus H+S_correct NLL; H and shifted-state contrasts are primary "
                "interpretive comparisons, best-control is adversarial sensitivity only"
            ),
            "required_fields": [
                "subject", "horizon_minutes", "endpoint", "gain_over_best_control",
                "gain_over_history", "gain_over_shifted", "gain_over_mean", "n_score_blocks",
            ],
            "same_prefix": {"status": "reported_as_later_continuation_in_closeout_json", "rows": []},
        },
        "h2b_transfer": {"status": "not_run", "risk_rows": [], "field_rows": []},
        "h3_feedback": {"status": "not_run", "model_rows": [], "impulse_rows": []},
        "claim_boundary": summary["claim_boundary"],
    }
    payload_path = args.output_root / "core_evidence_payload_v0_3_2.json"
    payload_path.write_text(json.dumps(figure_payload, indent=2, sort_keys=True, ensure_ascii=False))

    machine_root = args.data_root / "final"
    machine_root.mkdir(parents=True, exist_ok=True)
    machine_summary = machine_root / summary_path.name
    machine_payload = machine_root / payload_path.name
    machine_summary.write_bytes(summary_path.read_bytes())
    machine_payload.write_bytes(payload_path.read_bytes())
    print(json.dumps({"summary": str(summary_path), "payload": str(payload_path), "h2a_runs": h2a_runs}, indent=2))


if __name__ == "__main__":
    main()
