#!/usr/bin/env python3
"""Frozen 34-patient exploratory human test for Topic 5 v3.0."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_topic5_event_innovation_v3_0_local_response as local  # noqa: E402
from scripts import run_topic5_event_innovation_v3_0_cumulative_response as cumulative  # noqa: E402
from scripts.run_topic5_event_innovation_v3_0_observer import sequence_metadata  # noqa: E402
from scripts.run_topic5_event_innovation_v3_0_phase1_measurement import (  # noqa: E402
    _prepare,
    unit_balanced_dense_fields,
)
from src.topic5_event_innovation_data import (  # noqa: E402
    build_cumulative_anchor_splits,
    build_single_event_anchors,
    build_single_event_anchor_splits,
)
from src.topic5_event_innovation_test_v3_0 import (  # noqa: E402
    combine_cumulative_rows,
    combine_response_rows,
    fit_final_test_innovations,
)
from src.topic5_event_innovation_v3_0 import fit_rank_state_basis  # noqa: E402
from src.topic5_resource_guard import atomic_write_json, pin_thread_environment  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic5_event_innovation_v3_0.yaml"
OUTPUT_ROOT = ROOT / "results/topic5_event_innovation_impulse_response/v3_0/human_exploratory"
RELEASE_STATE = ROOT / "results/topic5_event_innovation_impulse_response/v3_0/HUMAN_TEST_RELEASE_STATE.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def verify_release(config_path: Path) -> dict:
    if not RELEASE_STATE.exists():
        raise RuntimeError("human test release state is missing")
    state = json.loads(RELEASE_STATE.read_text(encoding="utf-8"))
    if state.get("status") != "HUMAN_TEST_RELEASED":
        raise RuntimeError("human test is not released")
    if state.get("config_sha256") != sha256(config_path):
        raise RuntimeError("human test release/config hash mismatch")
    if state.get("human_test_outcomes_read") is not False:
        raise RuntimeError("release state is not pre-outcome")
    return state


def _load_common(subject: str, config: Mapping[str, Any]):
    phase0_root = ROOT / str(config["output_root"])
    innovation_root = ROOT / str(config["innovation_output_root"])
    observer_path = innovation_root / "per_subject" / f"{subject}.json"
    observer = json.loads(observer_path.read_text(encoding="utf-8"))
    if observer.get("status") != "INNOVATION_VALID":
        return observer, None
    raw, split_indices, sequences, phase0_path = _prepare(
        subject, config, phase0_root
    )
    selected = observer["observer_selection"]
    dimension = int(selected["dimension"])
    dense_fields, _, dense_weight = unit_balanced_dense_fields(
        raw["rank"], raw["participation"], sequences["train"],
        window=int(config["primary_horizon"]),
    )
    basis = fit_rank_state_basis(
        dense_fields, dimension, sample_weight=dense_weight
    )
    with np.load(Path(observer["crossfit_artifact"]), allow_pickle=False) as data:
        train_innovations = local.innovation_lookup(
            data["event_index"], data["rank_residual"], data["rank_valid"]
        )
    validation_innovations = local.fit_final_observer_innovations(
        raw, split_indices, sequences, basis, selected, config
    )
    fitting_innovations = {**train_innovations, **validation_innovations}
    test_innovations = fit_final_test_innovations(
        raw, sequences, basis, selected, config
    )
    nuisance = {
        split: sequence_metadata(sequences[split], len(raw["rank"]))[2]
        for split in ("train", "validation", "test")
    }
    return observer, {
        "raw": raw, "sequences": sequences, "basis": basis,
        "train_innovations": train_innovations,
        "validation_innovations": validation_innovations,
        "fitting_innovations": fitting_innovations,
        "test_innovations": test_innovations,
        "nuisance": nuisance, "phase0_path": phase0_path,
        "observer_path": observer_path, "dimension": dimension,
    }


def run_local_subject(subject: str, config: Mapping[str, Any]) -> dict:
    observer, common = _load_common(subject, config)
    if common is None:
        return {"subject": subject, "status": observer.get("status"), "eligible": False}
    validation_path = ROOT / str(config["local_response_output_root"]) / "per_subject" / f"{subject}.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    raw, sequences, basis = common["raw"], common["sequences"], common["basis"]
    horizons = {}
    for horizon in map(int, config["horizons"]):
        try:
            anchors = build_single_event_anchor_splits(
                sequences,
                pre_events=int(config["primary_pre_events"]),
                horizon=horizon,
            )
            fitting = combine_response_rows([
                local.build_response_rows(
                    raw, anchors.train, sequences["train"], basis,
                    common["train_innovations"], common["nuisance"]["train"],
                ),
                local.build_response_rows(
                    raw, anchors.validation, sequences["validation"], basis,
                    common["validation_innovations"], common["nuisance"]["validation"],
                ),
            ])
            test = local.build_response_rows(
                raw, anchors.test, sequences["test"], basis,
                common["test_innovations"], common["nuisance"]["test"],
            )
            dense_anchors = build_single_event_anchors(
                sequences["test"],
                pre_events=int(config["primary_pre_events"]),
                horizon=horizon, stride=1,
            )
            dense_test = local.build_response_rows(
                raw, dense_anchors, sequences["test"], basis,
                common["test_innovations"], common["nuisance"]["test"],
            )
            alpha = float(validation["horizons"][str(horizon)]["selected_alpha"])
            result = local.evaluate_horizon(
                raw, basis, fitting, test, dense_test, [alpha],
                donor_draws=int(config.get("local_response_null_draws", 100)),
                donor_seed=int(config.get("local_response_null_seed", 7401)) + horizon,
                block_sizes=config.get("local_response_block_sizes", [1, 2, 5, 10, 20, 40]),
                safe_shift_multipliers=config.get(
                    "local_response_safe_shift_multipliers", [2, 3, 4]
                ),
            )
            result["alpha_frozen_from_validation"] = alpha
            result["single_alpha_no_test_selection"] = True
            result["n_fitting_anchors"] = result["n_train_anchors"]
            result["n_test_anchors"] = result["n_validation_anchors"]
            result["test_state_mse_descriptive"] = result[
                "validation_state_mse"
            ]
            horizons[str(horizon)] = result
        except (ValueError, KeyError) as exc:
            horizons[str(horizon)] = {
                "status": "INSUFFICIENT_SUPPORT", "reason": str(exc)
            }
    primary = horizons[str(int(config["primary_horizon"]))]
    eligible = bool(
        "observable" in primary and primary.get("n_validation_anchors", 0) >= 20
    )
    return {
        "contract": str(config["contract"]), "route": "goal2_local_response",
        "subject": subject,
        "status": "HUMAN_TEST_LOCAL_COMPLETE" if eligible else "HUMAN_TEST_LOCAL_PRIMARY_UNAVAILABLE",
        "eligible": eligible, "dimension": common["dimension"], "horizons": horizons,
        "observer_record_sha256": sha256(common["observer_path"]),
        "validation_selection_sha256": sha256(validation_path),
        "phase0_sha256": sha256(common["phase0_path"]),
        "human_test_outcomes_read": True,
        "one_step_is_one_complete_event": True,
        "within_event_next_rank_model_fit": False,
    }


def run_cumulative_subject(subject: str, config: Mapping[str, Any]) -> dict:
    observer, common = _load_common(subject, config)
    if common is None:
        return {"subject": subject, "status": observer.get("status"), "eligible": False}
    validation_path = ROOT / str(config["cumulative_output_root"]) / "per_subject" / f"{subject}.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    raw, sequences, basis = common["raw"], common["sequences"], common["basis"]
    fitting_projected = cumulative.project_innovation_lookup(
        common["fitting_innovations"], basis
    )
    test_projected = cumulative.project_innovation_lookup(
        common["test_innovations"], basis
    )
    combinations = {}
    for exposure in map(int, config["cumulative_events"]):
        combinations[str(exposure)] = {}
        for horizon in map(int, config["horizons"]):
            try:
                anchors = build_cumulative_anchor_splits(
                    sequences,
                    pre_events=int(config["primary_pre_events"]),
                    exposure_events=exposure, horizon=horizon,
                )
                fitting = combine_cumulative_rows([
                    cumulative.build_cumulative_rows(
                        raw, anchors.train, sequences["train"], basis,
                        common["train_innovations"], common["nuisance"]["train"],
                        projected_innovations=fitting_projected,
                    ),
                    cumulative.build_cumulative_rows(
                        raw, anchors.validation, sequences["validation"], basis,
                        common["validation_innovations"], common["nuisance"]["validation"],
                        projected_innovations=fitting_projected,
                    ),
                ])
                test = cumulative.build_cumulative_rows(
                    raw, anchors.test, sequences["test"], basis,
                    common["test_innovations"], common["nuisance"]["test"],
                    projected_innovations=test_projected,
                )
                alpha = float(validation["combinations"][str(exposure)][str(horizon)]["selected_alpha"])
                result = cumulative.evaluate_cumulative(
                    raw, basis, fitting, test, [alpha],
                    null_draws=int(config.get("cumulative_null_draws", 100)),
                    null_seed=int(config.get("cumulative_null_seed", 7501)) + 100 * exposure + horizon,
                )
                result["alpha_frozen_from_validation"] = alpha
                result["single_alpha_no_test_selection"] = True
                result["n_fitting_anchors"] = result["n_train_anchors"]
                result["n_test_anchors"] = result["n_validation_anchors"]
                combinations[str(exposure)][str(horizon)] = result
            except (ValueError, KeyError) as exc:
                combinations[str(exposure)][str(horizon)] = {
                    "status": "INSUFFICIENT_SUPPORT", "reason": str(exc)
                }
    primary_exposure = int(config["primary_cumulative_events"])
    primary_horizon = int(config["primary_horizon"])
    primary = combinations[str(primary_exposure)][str(primary_horizon)]
    eligible = bool(
        "observable" in primary and primary.get("n_validation_anchors", 0) >= 20
    )
    iei_sensitivity = {}
    primary_anchors = build_cumulative_anchor_splits(
        sequences,
        pre_events=int(config["primary_pre_events"]),
        exposure_events=primary_exposure,
        horizon=primary_horizon,
    )
    for tau in map(float, config.get("iei_decay_tau_seconds", [])):
        try:
            fitting = combine_cumulative_rows([
                cumulative.build_cumulative_rows(
                    raw, primary_anchors.train, sequences["train"], basis,
                    common["train_innovations"], common["nuisance"]["train"],
                    tau_seconds=tau,
                    projected_innovations=fitting_projected,
                ),
                cumulative.build_cumulative_rows(
                    raw, primary_anchors.validation, sequences["validation"], basis,
                    common["validation_innovations"], common["nuisance"]["validation"],
                    tau_seconds=tau,
                    projected_innovations=fitting_projected,
                ),
            ])
            test = cumulative.build_cumulative_rows(
                raw, primary_anchors.test, sequences["test"], basis,
                common["test_innovations"], common["nuisance"]["test"],
                tau_seconds=tau,
                projected_innovations=test_projected,
            )
            alpha = float(
                validation["iei_decay_sensitivity"][str(tau)]["selected_alpha"]
            )
            result = cumulative.evaluate_cumulative(
                raw, basis, fitting, test, [alpha],
                null_draws=int(config.get("cumulative_null_draws", 100)),
                null_seed=int(config.get("cumulative_null_seed", 7501)) + int(tau),
            )
            result["alpha_frozen_from_validation"] = alpha
            result["single_alpha_no_test_selection"] = True
            result["n_fitting_anchors"] = result["n_train_anchors"]
            result["n_test_anchors"] = result["n_validation_anchors"]
            iei_sensitivity[str(tau)] = result
        except (ValueError, KeyError) as exc:
            iei_sensitivity[str(tau)] = {
                "status": "INSUFFICIENT_SUPPORT", "reason": str(exc)
            }
    return {
        "contract": str(config["contract"]), "route": "goal3_cumulative_response",
        "subject": subject,
        "status": "HUMAN_TEST_CUMULATIVE_COMPLETE" if eligible else "HUMAN_TEST_CUMULATIVE_PRIMARY_UNAVAILABLE",
        "eligible": eligible, "dimension": common["dimension"],
        "primary_exposure_events": primary_exposure,
        "primary_horizon": primary_horizon, "combinations": combinations,
        "iei_decay_sensitivity": iei_sensitivity,
        "observer_record_sha256": sha256(common["observer_path"]),
        "validation_selection_sha256": sha256(validation_path),
        "phase0_sha256": sha256(common["phase0_path"]),
        "human_test_outcomes_read": True,
        "one_step_is_one_complete_event": True,
        "within_event_next_rank_model_fit": False,
    }


def patient_inference(values) -> dict:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0}
    rng = np.random.default_rng(20260803)
    bootstrap = np.median(
        rng.choice(array, (10000, len(array)), replace=True), axis=1
    )
    nonzero = array[array != 0]
    try:
        p = float(wilcoxon(array, alternative="two-sided").pvalue)
    except ValueError:
        p = None
    return {
        "n": int(len(array)), "median": float(np.median(array)),
        "bootstrap_median_ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "n_positive": int(np.sum(array > 0)),
        "wilcoxon_two_sided_p": p,
        "sign_test_two_sided_p": (
            float(binomtest(int(np.sum(nonzero > 0)), len(nonzero), 0.5).pvalue)
            if len(nonzero) else None
        ),
    }


def _route_values(rows, kind):
    selected = [row for row in rows if row.get("eligible")]
    if kind == "local":
        primary = [row["horizons"]["20"] for row in selected]
        return {
            "propagation_gain": [x["observable"]["propagation_gain_standardized"] for x in primary],
            "true_minus_matched": [x["true_minus_state_matched_null_gain"] for x in primary],
            "future_minus_past": [x["future_minus_past_state_gain"] for x in primary],
        }
    primary = [
        row["combinations"][str(row["primary_exposure_events"])][str(row["primary_horizon"])]
        for row in selected
    ]
    return {
        "cumulative_gain": [x["observable"]["propagation_gain_standardized"] for x in primary],
        "true_minus_matched": [x["true_minus_matched_cumulative_null_gain"] for x in primary],
        "alignment": [x["dose_alignment"]["alignment_coefficient"] for x in primary],
    }


def aggregate(kind: str, config_path: Path, config: dict, output: Path) -> dict:
    innovation = json.loads(
        (ROOT / str(config["innovation_output_root"]) / "innovation_validity.json").read_text()
    )
    rows, failures = [], []
    for subject in [row["subject"] for row in innovation["patients"]]:
        path = output / kind / "per_subject" / f"{subject}.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            failures.append({"subject": subject, "error": "MissingArtifact"})
    values = _route_values(rows, kind)
    inference = {name: patient_inference(value) for name, value in values.items()}
    dataset = {}
    for prefix in ("epilepsiae", "yuquan"):
        subset = [row for row in rows if row["subject"].startswith(prefix + "_")]
        dataset[prefix] = {
            name: patient_inference(value)
            for name, value in _route_values(subset, kind).items()
        }
    state = {
        "contract": str(config["contract"]), "route": kind,
        "status": "HUMAN_TEST_ROUTE_COMPLETE" if len(rows) == 34 and not failures else "INCOMPLETE",
        "n_completed": len(rows), "n_failed": len(failures),
        "n_eligible": int(sum(row.get("eligible", False) for row in rows)),
        "cohort_inference": inference, "dataset_specific": dataset,
        "patients": rows, "failures": failures,
        "config_sha256": sha256(config_path),
        "release_state_sha256": sha256(RELEASE_STATE),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "human_test_outcomes_read": True,
        "test_dependent_selection": False,
        "one_step_is_one_complete_event": True,
        "within_event_next_rank_model_fit": False,
    }
    atomic_write_json(output / kind / f"{kind.upper()}_TEST_STATE.json", jsonable(state))
    summary = []
    for row in rows:
        value = _route_values([row], kind)
        summary.append({
            "subject": row["subject"], "eligible": row.get("eligible", False),
            **{name: (item[0] if item else np.nan) for name, item in value.items()},
        })
    pd.DataFrame(summary).to_csv(output / kind / "patient_summary.csv", index=False)
    if failures:
        raise SystemExit(1)
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--kind", choices=("local", "cumulative"), required=True)
    parser.add_argument("--phase", choices=("patients", "aggregate"), required=True)
    parser.add_argument("--subjects", nargs="*")
    args = parser.parse_args()
    pin_thread_environment(1, disable_cuda=True)
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    verify_release(config_path)
    output = OUTPUT_ROOT
    if args.phase == "aggregate":
        state = aggregate(args.kind, config_path, config, output)
        print(json.dumps({"status": state["status"], "route": args.kind}, indent=2))
        return
    if not args.subjects:
        raise ValueError("patients phase requires explicit subjects")
    runner = run_local_subject if args.kind == "local" else run_cumulative_subject
    for subject in args.subjects:
        row = runner(subject, config)
        atomic_write_json(
            output / args.kind / "per_subject" / f"{subject}.json", jsonable(row)
        )
        print(subject, row["status"], flush=True)


if __name__ == "__main__":
    main()
