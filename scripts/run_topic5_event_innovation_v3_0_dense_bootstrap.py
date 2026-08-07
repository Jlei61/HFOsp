#!/usr/bin/env python3
"""Frozen dense-test moving-block bootstrap sensitivity for Topic 5 V3.0."""
from __future__ import annotations

import argparse
import hashlib
import json
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

from scripts import run_topic5_event_innovation_v3_0_human_test as human  # noqa: E402
from scripts import run_topic5_event_innovation_v3_0_local_response as local  # noqa: E402
from scripts import run_topic5_event_innovation_v3_0_cumulative_response as cumulative  # noqa: E402
from src.topic5_event_innovation_bootstrap_v3_0 import (  # noqa: E402
    moving_block_resamples,
    observable_gain_sufficient_statistics,
    standardized_propagation_gain,
)
from src.topic5_event_innovation_data import (  # noqa: E402
    build_cumulative_anchors,
    build_cumulative_anchor_splits,
    build_single_event_anchors,
    build_single_event_anchor_splits,
)
from src.topic5_event_innovation_response_v3_0 import (  # noqa: E402
    fit_weighted_local_projection,
    future_precedence_brier,
    masked_rank_field_mse,
)
from src.topic5_resource_guard import atomic_write_json, pin_thread_environment  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic5_event_innovation_v3_0.yaml"
OUTPUT_ROOT = human.OUTPUT_ROOT / "dense_bootstrap"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scales(raw, basis, rows) -> tuple[float, float]:
    backbone = np.broadcast_to(basis.backbone, rows.observed_future_field.shape)
    return (
        masked_rank_field_mse(
            backbone, rows.observed_future_field, rows.future_support
        ),
        future_precedence_brier(
            backbone,
            rows.future_windows,
            raw["rank"],
            raw["participation"],
            raw["rank"],
        ),
    )


def bootstrap_effect(raw, basis, fitting, dense, alpha, *, block_length, draws, seed):
    innovation = (
        dense.innovation_state
        if hasattr(dense, "innovation_state")
        else dense.cumulative_innovation
    )
    fitting_innovation = (
        fitting.innovation_state
        if hasattr(fitting, "innovation_state")
        else fitting.cumulative_innovation
    )
    fit = fit_weighted_local_projection(
        fitting.pre_state,
        fitting.future_state,
        fitting_innovation,
        nuisance=fitting.nuisance,
        alpha=float(alpha),
        sample_weight=local.group_balanced_weights(fitting.group),
    )
    driven = fit.predict(dense.pre_state, innovation, dense.nuisance)
    automatic = fit.predict(dense.pre_state, np.zeros_like(innovation), dense.nuisance)
    statistics = observable_gain_sufficient_statistics(
        basis,
        dense.observed_future_field,
        dense.future_support,
        dense.future_windows,
        raw["rank"],
        raw["participation"],
        raw["rank"],
        automatic,
        driven,
    )
    rank_scale, pair_scale = scales(raw, basis, fitting)
    event_index = (
        dense.event_index if hasattr(dense, "event_index") else dense.anchor_event
    )
    full = standardized_propagation_gain(
        statistics,
        np.arange(len(event_index)),
        rank_scale=rank_scale,
        pair_scale=pair_scale,
    )
    values = np.asarray([
        standardized_propagation_gain(
            statistics,
            selected,
            rank_scale=rank_scale,
            pair_scale=pair_scale,
        )
        for selected in moving_block_resamples(
            dense.group,
            event_index,
            block_length=int(block_length),
            draws=int(draws),
            seed=int(seed),
        )
    ])
    values = values[np.isfinite(values)]
    return {
        "status": "DENSE_MOVING_BLOCK_BOOTSTRAP_COMPLETE",
        "n_dense_anchors": int(len(event_index)),
        "block_length_anchors": int(block_length),
        "n_draws": int(len(values)),
        "full_dense_propagation_gain": float(full),
        "bootstrap_median": float(np.median(values)),
        "bootstrap_ci95": np.quantile(values, [0.025, 0.975]).tolist(),
    }


def run_subject(subject: str, kind: str, config: Mapping[str, Any]) -> dict:
    observer, common = human._load_common(subject, config)
    if common is None:
        return {
            "subject": subject,
            "status": observer.get("status"),
            "eligible": False,
            "runner_sha256": sha256(Path(__file__).resolve()),
            "sensitivity_only": True,
            "one_step_is_one_complete_event": True,
            "within_event_next_rank_model_fit": False,
        }
    raw, sequences, basis = common["raw"], common["sequences"], common["basis"]
    horizon = int(config["primary_horizon"])
    pre_events = int(config["primary_pre_events"])
    if kind == "local":
        selection_path = ROOT / str(config["local_response_output_root"]) / "per_subject" / f"{subject}.json"
        selected = json.loads(selection_path.read_text(encoding="utf-8"))
        alpha = float(selected["horizons"][str(horizon)]["selected_alpha"])
        anchors = build_single_event_anchor_splits(
            sequences, pre_events=pre_events, horizon=horizon
        )
        fitting = human.combine_response_rows([
            local.build_response_rows(raw, anchors.train, sequences["train"], basis, common["train_innovations"], common["nuisance"]["train"]),
            local.build_response_rows(raw, anchors.validation, sequences["validation"], basis, common["validation_innovations"], common["nuisance"]["validation"]),
        ])
        dense_anchors = build_single_event_anchors(
            sequences["test"], pre_events=pre_events, horizon=horizon, stride=1
        )
        dense = local.build_response_rows(
            raw, dense_anchors, sequences["test"], basis,
            common["test_innovations"], common["nuisance"]["test"],
        )
    else:
        exposure = int(config["primary_cumulative_events"])
        selection_path = ROOT / str(config["cumulative_output_root"]) / "per_subject" / f"{subject}.json"
        selected = json.loads(selection_path.read_text(encoding="utf-8"))
        alpha = float(selected["combinations"][str(exposure)][str(horizon)]["selected_alpha"])
        anchors = build_cumulative_anchor_splits(
            sequences, pre_events=pre_events,
            exposure_events=exposure, horizon=horizon,
        )
        fitting_projected = cumulative.project_innovation_lookup(
            common["fitting_innovations"], basis
        )
        test_projected = cumulative.project_innovation_lookup(
            common["test_innovations"], basis
        )
        fitting = human.combine_cumulative_rows([
            cumulative.build_cumulative_rows(raw, anchors.train, sequences["train"], basis, common["train_innovations"], common["nuisance"]["train"], projected_innovations=fitting_projected),
            cumulative.build_cumulative_rows(raw, anchors.validation, sequences["validation"], basis, common["validation_innovations"], common["nuisance"]["validation"], projected_innovations=fitting_projected),
        ])
        dense_anchors = build_cumulative_anchors(
            sequences["test"], pre_events=pre_events,
            exposure_events=exposure, horizon=horizon, stride=1,
        )
        dense = cumulative.build_cumulative_rows(
            raw, dense_anchors, sequences["test"], basis,
            common["test_innovations"], common["nuisance"]["test"],
            projected_innovations=test_projected,
        )
    result = bootstrap_effect(
        raw, basis, fitting, dense, alpha,
        block_length=horizon,
        draws=int(config.get("dense_bootstrap_draws", 500)),
        seed=int(config.get("dense_bootstrap_seed", 20260803)) + (0 if kind == "local" else 100000),
    )
    return {
        "contract": str(config["contract"]),
        "route": kind,
        "subject": subject,
        "status": result["status"],
        "eligible": True,
        "result": result,
        "selection_sha256": sha256(selection_path),
        "release_state_sha256": sha256(human.RELEASE_STATE),
        "human_test_outcomes_read": True,
        "sensitivity_only": True,
        "one_step_is_one_complete_event": True,
        "within_event_next_rank_model_fit": False,
        "runner_sha256": sha256(Path(__file__).resolve()),
    }


def patient_inference(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0}
    nonzero = array[array != 0]
    rng = np.random.default_rng(20260803)
    boot = np.median(rng.choice(array, (10000, len(array)), replace=True), axis=1)
    try:
        p = float(wilcoxon(array, alternative="two-sided").pvalue)
    except ValueError:
        p = None
    return {
        "n": int(len(array)),
        "median": float(np.median(array)),
        "bootstrap_median_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "n_positive": int(np.sum(array > 0)),
        "wilcoxon_two_sided_p": p,
        "sign_test_two_sided_p": (
            float(binomtest(int(np.sum(nonzero > 0)), len(nonzero), 0.5).pvalue)
            if len(nonzero) else None
        ),
    }


def aggregate(kind: str, config: Mapping[str, Any]) -> dict:
    innovation = json.loads((ROOT / str(config["innovation_output_root"]) / "innovation_validity.json").read_text())
    subjects = [row["subject"] for row in innovation["patients"]]
    paths = [OUTPUT_ROOT / kind / "per_subject" / f"{subject}.json" for subject in subjects]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"dense bootstrap artifacts missing: {len(missing)}")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    current_runner = sha256(Path(__file__).resolve())
    stale = [row["subject"] for row in rows if row.get("runner_sha256") != current_runner]
    if stale:
        raise RuntimeError(f"dense bootstrap artifacts use stale runner: {stale}")
    eligible = [row for row in rows if row.get("eligible")]
    values = [row["result"]["full_dense_propagation_gain"] for row in eligible]
    state = {
        "contract": str(config["contract"]),
        "route": kind,
        "status": "DENSE_BOOTSTRAP_ROUTE_COMPLETE",
        "n_patients": len(rows),
        "n_eligible": len(eligible),
        "cohort_inference": patient_inference(values),
        "patients": rows,
        "sensitivity_only": True,
        "human_test_outcomes_read": True,
        "one_step_is_one_complete_event": True,
        "within_event_next_rank_model_fit": False,
        "runner_sha256": current_runner,
    }
    atomic_write_json(OUTPUT_ROOT / kind / f"{kind.upper()}_DENSE_BOOTSTRAP_STATE.json", state)
    pd.DataFrame([
        {
            "subject": row["subject"],
            "eligible": row.get("eligible", False),
            **(row.get("result") or {}),
        }
        for row in rows
    ]).to_csv(OUTPUT_ROOT / kind / "patient_summary.csv", index=False)
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
    human.verify_release(config_path)
    if args.phase == "aggregate":
        print(json.dumps(aggregate(args.kind, config), indent=2, sort_keys=True))
        return
    if not args.subjects:
        raise ValueError("patients phase requires explicit subjects")
    for subject in args.subjects:
        row = run_subject(subject, args.kind, config)
        atomic_write_json(OUTPUT_ROOT / args.kind / "per_subject" / f"{subject}.json", row)
        print(subject, row["status"], flush=True)


if __name__ == "__main__":
    main()
