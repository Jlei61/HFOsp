#!/usr/bin/env python3
"""Formal three-seed test for the repair-only Topic 5 v2.7 model.

The data, scores, controls and patient-first inference are inherited from the
frozen v2.6 implementation.  Only fitting and checkpoint provenance are bound
to v2.7.  Validation selection must already be frozen; this module never
changes a profile or training budget.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_topic5_stateful_event_rnn_v2_6 as parent  # noqa: E402
from scripts import run_topic5_stateful_event_rnn_v2_7 as selection  # noqa: E402
from src.topic5_stateful_event_rnn_v2_7 import trace_to_dict  # noqa: E402


DEFAULT_CONFIG = selection.DEFAULT_CONFIG
jsonable = selection.jsonable
prepare_subject = selection.prepare_subject
score_dict = parent.score_dict
sha256 = selection.sha256


def fit_profile(subject, profile, datasets, encoder, config, scales, seed):
    """Expose the frozen v2.6 call signature with repaired v2.7 fitting."""

    del subject
    return selection.fit_profile(
        profile, datasets, encoder, config, scales, seed
    )


def save_checkpoint(path, fitted, encoder, ewma, subject, seed, training_budget):
    """Write a checkpoint with explicit v2.7 contract and training lineage."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "contract": "topic5_stateful_event_sequence_rnn_v2_7",
            "subject": subject,
            "seed": int(seed),
            "trained_model_state_dict": fitted.trained_model.state_dict(),
            "nested_model_state_dict": fitted.nested_model.state_dict(),
            "feature_mean": fitted.feature_mean,
            "feature_scale": fitted.feature_scale,
            "profile": asdict(fitted.profile),
            "training_budget": {
                key: int(value) for key, value in training_budget.items()
            },
            "trace": trace_to_dict(fitted.trace),
            "encoder": {
                "centers": encoder.centers,
                "feature_mean": encoder.feature_mean,
                "feature_scale": encoder.feature_scale,
                "rank_prior": encoder.rank_prior,
                "n_modes": encoder.n_modes,
            },
            "ewma": {
                "decay": ewma.decay,
                "alpha": ewma.alpha,
                "feature_mean": ewma.feature_mean,
                "feature_scale": ewma.feature_scale,
                "ridge_coef": ewma.ridge.coef_,
                "ridge_intercept": ewma.ridge.intercept_,
            },
            "parent_v2_6": selection.provenance_manifest(DEFAULT_CONFIG)[
                "parent_v2_6"
            ],
        },
        path,
    )


def run_subject(subject, config, output):
    """Run the frozen v2.6 evaluation with v2.7 fitting and provenance."""

    previous = (parent.prepare_subject, parent.fit_profile, parent.save_checkpoint)
    parent.prepare_subject = prepare_subject
    parent.fit_profile = fit_profile
    parent.save_checkpoint = save_checkpoint
    try:
        result = parent.run_subject(subject, config, output)
    finally:
        parent.prepare_subject, parent.fit_profile, parent.save_checkpoint = previous
    if result.get("contract") != "topic5_stateful_event_sequence_rnn_v2_7":
        raise RuntimeError("formal result was not written under the v2.7 contract")
    return result


def verify_frozen(config_path: Path, output: Path) -> dict:
    """Fail closed unless all 34 repaired validation profiles are frozen."""

    state_path = output / "validation_screen/FROZEN_VALIDATION_STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    expected = {
        "config_sha256": sha256(config_path),
        "module_sha256": sha256(ROOT / "src/topic5_stateful_event_rnn_v2_7.py"),
        "runner_sha256": sha256(
            ROOT / "scripts/run_topic5_stateful_event_rnn_v2_7.py"
        ),
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise RuntimeError(f"v2.7 frozen validation hash mismatch: {key}")
    if state.get("status") != "ALL_PATIENT_VALIDATION_PROFILES_FROZEN":
        raise RuntimeError("v2.7 validation profiles are not fully frozen")
    if state.get("test_results_read_during_selection") is not False:
        raise RuntimeError("v2.7 selection provenance does not exclude test read")
    return state


def aggregate(results, failures, config, config_path: Path, output: Path):
    """Write the v2.7 patient-first formal endpoint without v2.6 hash drift."""

    rows = [
        {
            "subject": item["subject"],
            "dataset": item["dataset"],
            "n_events_train80": item["n_events_train80"],
            "n_contacts": item["n_contacts"],
            "n_formal_test_targets": item["n_formal_test_targets"],
            "cell": item["selected_profile"]["cell"],
            "hidden_size": item["selected_profile"]["hidden_size"],
            "tbptt_length": item["selected_profile"]["tbptt_length"],
            "optimizer": item["selected_profile"]["optimizer"],
            "learning_rate": item["selected_profile"]["learning_rate"],
            "trained_rnn_minus_ewma_propagation": item[
                "trained_rnn_minus_ewma"
            ]["propagation"],
            "trained_rnn_minus_ewma_recruitment": item[
                "trained_rnn_minus_ewma"
            ]["recruitment"],
            "nested_rnn_minus_ewma_propagation": item[
                "nested_rnn_minus_ewma"
            ]["propagation"],
        }
        for item in results
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "patient_summary.csv", index=False)
    pd.DataFrame(
        failures, columns=("subject", "error_type", "reason")
    ).to_csv(output / "failures.csv", index=False)
    state = {
        "contract": config["contract"],
        "status": (
            "STATEFUL_34_PATIENT_TEST_COMPLETE"
            if len(results) == 34 and not failures
            else "INCOMPLETE"
        ),
        "n_attempted": 34,
        "n_completed": len(results),
        "n_failed": len(failures),
        "trained_primary_propagation": (
            parent.patient_inference(frame["trained_rnn_minus_ewma_propagation"])
            if len(frame)
            else {}
        ),
        "nested_secondary_propagation": (
            parent.patient_inference(frame["nested_rnn_minus_ewma_propagation"])
            if len(frame)
            else {}
        ),
        "trained_secondary_recruitment": (
            parent.patient_inference(frame["trained_rnn_minus_ewma_recruitment"])
            if len(frame)
            else {}
        ),
        "selected_cell_counts": (
            frame["cell"].value_counts().to_dict() if len(frame) else {}
        ),
        "selected_tbptt_counts": (
            frame["tbptt_length"].value_counts().to_dict() if len(frame) else {}
        ),
        "config_sha256": sha256(config_path),
        "module_sha256": sha256(ROOT / "src/topic5_stateful_event_rnn_v2_7.py"),
        "validation_runner_sha256": sha256(
            ROOT / "scripts/run_topic5_stateful_event_rnn_v2_7.py"
        ),
        "formal_runner_sha256": sha256(Path(__file__).resolve()),
        "frozen_validation_state_sha256": sha256(
            output / "validation_screen/FROZEN_VALIDATION_STATE.json"
        ),
        "old_heldout20_entered": False,
        "one_step_is_one_complete_event": True,
        "within_event_next_rank_model_fit": False,
    }
    destination = output / "STATEFUL_TEST_STATE.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(jsonable(state), indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(destination)
    print(json.dumps(jsonable(state), indent=2, sort_keys=True))
    return state

