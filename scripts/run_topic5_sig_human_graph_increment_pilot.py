#!/usr/bin/env python3
"""Run the six-patient SIG0-vs-SIG1 graph-increment development screen."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_rank_distribution import distribution_errors  # noqa: E402
from src.topic5_shared_propagation_field import (  # noqa: E402
    fit_static_scaffold_ml,
    load_subject_rank_events,
    sha256_file,
)
from src.topic5_stable_interaction_graph import (  # noqa: E402
    SyntheticGraphDataset,
    cardinality_schedule,
    fit_synthetic_sig,
    uniform_provenance,
)


DEFAULT_CONFIG = ROOT / "config/topic5_stable_interaction_graph_v2.yaml"
DEFAULT_OUTPUT = (
    ROOT
    / "results/topic5_stable_interaction_graph/development"
    / "human_graph_increment_pilot_v0_2_training_adequacy"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n"
    )


def _source_sha256() -> dict[str, str]:
    """Provenance of every file that decides what the fits are."""
    return {
        "runner": sha256_file(Path(__file__)),
        "model": sha256_file(ROOT / "src/topic5_stable_interaction_graph.py"),
        "shared_propagation_field": sha256_file(
            ROOT / "src/topic5_shared_propagation_field.py"
        ),
    }


def _subsample(indices: np.ndarray, limit: int) -> np.ndarray:
    values = np.asarray(indices, dtype=int)
    if len(values) <= int(limit):
        return values
    return values[
        np.linspace(0, len(values) - 1, int(limit)).astype(int)
    ]


def _dataset(record, indices: np.ndarray) -> SyntheticGraphDataset:
    groups = record.group_ids[indices].copy()
    counts = record.group_count[indices].copy()
    first = groups == 0
    starts = np.argmax(first, axis=1).astype(np.int16)
    return SyntheticGraphDataset(groups, counts, starts)


def _score_model(model, dataset: SyntheticGraphDataset) -> float:
    groups, counts = dataset.torch()
    model.eval()
    with torch.no_grad():
        return float(model.nll_per_decision(groups, counts))


def _repertoire(model, dataset, *, seed: int, repeats: int) -> dict[str, Any]:
    groups, counts = dataset.torch()
    first = groups == 0
    schedule = torch.as_tensor(
        cardinality_schedule(dataset.group_ids, dataset.group_count),
        dtype=torch.long,
    )
    rows = []
    first_generated = None
    for repeat in range(int(repeats)):
        generated = model.rollout(
            first,
            counts,
            schedule,
            generator=torch.Generator().manual_seed(
                int(seed) + repeat * 1009
            ),
        ).cpu().numpy().astype(np.int16)
        if first_generated is None:
            first_generated = generated
        rows.append(
            distribution_errors(
                generated,
                dataset.group_count,
                dataset.group_ids,
                dataset.group_count,
            )
        )
    return {
        "mean": {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        },
        "sd": {
            key: float(np.std([row[key] for row in rows], ddof=1))
            for key in rows[0]
        },
        "first_generated": first_generated,
    }


def _run_subject(
    subject: str,
    *,
    config_path: str,
    output_dir: str,
) -> dict[str, Any]:
    torch.set_num_threads(2)
    config_path_value = Path(config_path)
    config = yaml.safe_load(config_path_value.read_text())
    data = config["data"]
    training = config["training"]
    model_config = config["model"]
    evaluation = config["evaluation"]
    dataset_dir = ROOT / data["dataset_dir"]
    record = load_subject_rank_events(dataset_dir, subject)
    train_index, validation_index, test_index = record.development_split(
        float(data["validation_fraction"]),
        float(data["test_fraction"]),
    )
    train_index = _subsample(train_index, int(data["max_train_events"]))
    validation_index = _subsample(
        validation_index, int(data["max_validation_events"])
    )
    test_index = _subsample(test_index, int(data["max_test_events"]))
    all_used = np.r_[train_index, validation_index, test_index]
    if np.intersect1d(all_used, record.old_heldout20_indices).size:
        raise RuntimeError(f"{subject}: old heldout20 leakage")
    if (
        np.intersect1d(train_index, validation_index).size
        or np.intersect1d(train_index, test_index).size
        or np.intersect1d(validation_index, test_index).size
    ):
        raise RuntimeError(f"{subject}: development partitions overlap")
    scaffold = fit_static_scaffold_ml(
        record.group_ids,
        record.group_count,
        train_index,
        steps=int(training["static_scaffold_steps"]),
        learning_rate=float(training["static_scaffold_learning_rate"]),
        seed=int(training["static_scaffold_seed"]),
        device="cpu",
    )
    train = _dataset(record, train_index)
    validation = _dataset(record, validation_index)
    # The test values are materialized only after both checkpoints are selected.
    subject_rows = []
    output = Path(output_dir)
    for fit_seed in map(int, training["fit_seeds"]):
        fitted = {}
        for name, learn_graph in (
            ("sig0_nograph", False),
            ("sig1_feedback_graph", True),
        ):
            result = fit_synthetic_sig(
                train,
                validation,
                seed=fit_seed,
                learn_graph=learn_graph,
                max_epochs=int(training["epochs"]),
                patience=int(training["patience"]),
                learning_rate=float(training["learning_rate"]),
                l1_weight=float(model_config["graph_l1_weight"]),
                batch_size=int(training["batch_events"]),
                static_bias=scaffold,
                convergence_tolerance=float(
                    training["convergence_tolerance"]
                ),
                minimum_relative_improvement=float(
                    training["minimum_relative_improvement"]
                ),
                minimum_training_epochs=int(
                    training["minimum_training_epochs"]
                ),
                minimum_best_epoch=int(training["minimum_best_epoch"]),
                maximum_recovery_depth=int(
                    training["maximum_recovery_depth"]
                ),
            )
            fitted[name] = result

        test = _dataset(record, test_index)
        seed_payload = {
            "contract": config["contract"]["name"],
            "subject": subject,
            "dataset": record.dataset,
            "fit_seed": fit_seed,
            "n_contacts": len(record.contact_names),
            "n_train_events": len(train_index),
            "n_validation_events": len(validation_index),
            "n_test_events": len(test_index),
            "n_train_suffix_decisions": int(
                np.sum(record.group_count[train_index] - 1)
            ),
            "models": {},
            "input_sha256": record.input_sha256,
            "config_sha256": sha256_file(config_path_value),
            "source_sha256": _source_sha256(),
            "leakage_flags": {
                "old_heldout20": False,
                "ab_or_axis": False,
                "soz_or_ictal": False,
                "geometry_or_snn": False,
            },
        }
        run_dir = output / "per_run" / subject / f"seed_{fit_seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        generation_arrays = {}
        for offset, (name, result) in enumerate(fitted.items()):
            nll = _score_model(result.model, test)
            repertoire = _repertoire(
                result.model,
                test,
                seed=fit_seed + 500 + offset * 100_000,
                repeats=int(evaluation["rollout_repeats"]),
            )
            generation_arrays[name] = repertoire.pop("first_generated")
            seed_payload["models"][name] = {
                "nll_per_decision": nll,
                "best_validation_nll": result.best_validation_nll,
                "best_epoch": result.best_epoch,
                "learning_rate": result.learning_rate,
                "recovery_depth": result.recovery_depth,
                "n_trainable_parameters": int(
                    sum(
                        parameter.numel()
                        for parameter in result.model.parameters()
                        if parameter.requires_grad
                    )
                ),
                "repertoire": repertoire,
                "training_adequacy": result.adequacy,
            }
        torch.save(
            {
                "contract": config["contract"]["name"],
                "subject": subject,
                "fit_seed": fit_seed,
                "scaffold": scaffold,
                "sig0_state_dict": fitted["sig0_nograph"].model.state_dict(),
                "sig1_state_dict": fitted[
                    "sig1_feedback_graph"
                ].model.state_dict(),
                "sig0_optimizer_state": fitted[
                    "sig0_nograph"
                ].best_optimizer_state,
                "sig1_optimizer_state": fitted[
                    "sig1_feedback_graph"
                ].best_optimizer_state,
            },
            run_dir / "checkpoint.pt",
        )
        np.savez_compressed(
            run_dir / "conditioned_generation.npz",
            observed_group_ids=test.group_ids,
            observed_group_count=test.group_count,
            **generation_arrays,
        )
        _write(
            run_dir / "sig0_history.json",
            fitted["sig0_nograph"].history,
        )
        _write(
            run_dir / "sig1_history.json",
            fitted["sig1_feedback_graph"].history,
        )
        _write(run_dir / "summary.json", seed_payload)
        subject_rows.append(seed_payload)
    state = {
        "status": "COMPLETE",
        "subject": subject,
        "n_seeds": len(subject_rows),
        "all_training_adequate": bool(
            all(
                value["models"][name]["training_adequacy"]["converged"]
                for value in subject_rows
                for name in ("sig0_nograph", "sig1_feedback_graph")
            )
        ),
        "old_heldout20_scored": False,
        "snn_inputs_read": False,
    }
    _write(output / "per_run" / subject / "SUBJECT_STATE.json", state)
    return state


def _aggregate(config_path: Path, output: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    subjects = list(map(str, config["pilot"]["subjects"]))
    seeds = list(map(int, config["training"]["fit_seeds"]))
    run_rows = []
    payloads = []
    for subject in subjects:
        for seed in seeds:
            path = output / "per_run" / subject / f"seed_{seed}/summary.json"
            if not path.exists():
                raise RuntimeError(f"missing human SIG run: {path}")
            payload = json.loads(path.read_text())
            if any(payload["leakage_flags"].values()):
                raise RuntimeError(f"leakage flag in {path}")
            payloads.append(payload)
            row = {
                "subject": subject,
                "dataset": payload["dataset"],
                "fit_seed": seed,
            }
            for model, prefix in (
                ("sig0_nograph", "sig0"),
                ("sig1_feedback_graph", "sig1"),
            ):
                value = payload["models"][model]
                if not value["training_adequacy"]["converged"]:
                    raise RuntimeError(
                        f"inadequate training in {path}: {model}"
                    )
                row[f"{prefix}_nll"] = value["nll_per_decision"]
                row[f"{prefix}_precedence_mae"] = value["repertoire"]["mean"][
                    "precedence_mae"
                ]
                row[f"{prefix}_precedence_correlation"] = value["repertoire"][
                    "mean"
                ]["precedence_correlation"]
                row[f"{prefix}_participation_mae"] = value["repertoire"][
                    "mean"
                ]["participation_mae"]
                row[f"{prefix}_rank_wasserstein"] = value["repertoire"]["mean"][
                    "rank_wasserstein"
                ]
            run_rows.append(row)
    # Fail closed before any number is aggregated: a mixed or missing
    # source/config makes the reported increment unattributable to one model.
    fit_provenance = uniform_provenance(
        payloads,
        ("config_sha256", "source_sha256"),
        current_source_sha256=_source_sha256(),
    )
    # The dataset hash is per subject, so it is checked within subject only.
    input_sha256 = {}
    for subject in subjects:
        rows = [row for row in payloads if row["subject"] == subject]
        input_sha256[subject] = uniform_provenance(rows, ("input_sha256",))[
            "input_sha256"
        ]
    patient_rows = []
    for subject in subjects:
        values = [row for row in run_rows if row["subject"] == subject]
        patient = {
            "subject": subject,
            "dataset": values[0]["dataset"],
        }
        for key in values[0]:
            if key in ("subject", "dataset", "fit_seed"):
                continue
            patient[key] = float(np.median([row[key] for row in values]))
        patient["nll_gain_sig1"] = patient["sig0_nll"] - patient["sig1_nll"]
        patient["precedence_mae_gain_sig1"] = (
            patient["sig0_precedence_mae"]
            - patient["sig1_precedence_mae"]
        )
        patient["both_primary_improved"] = bool(
            patient["nll_gain_sig1"] > 0
            and patient["precedence_mae_gain_sig1"] > 0
        )
        patient_rows.append(patient)
    thresholds = config["evaluation"][
        "continue_to_full_baseline_ladder_requires"
    ]
    counts = {
        "n_patients_nll_better": int(
            sum(row["nll_gain_sig1"] > 0 for row in patient_rows)
        ),
        "n_patients_precedence_mae_better": int(
            sum(row["precedence_mae_gain_sig1"] > 0 for row in patient_rows)
        ),
        "n_patients_both_better": int(
            sum(row["both_primary_improved"] for row in patient_rows)
        ),
    }
    passed = bool(
        all(counts[key] >= int(thresholds[key]) for key in thresholds)
    )
    payload = {
        "contract": config["contract"]["name"],
        "status": "COMPLETE",
        "screen": "SIG0_vs_SIG1_graph_increment_only",
        "runner_revision": "v0_3_provenance_contract",
        "decision": (
            "CONTINUE_TO_PHASE_MATCHED_BASELINE_LADDER"
            if passed
            else "STOP_HUMAN_SIG_LINE"
        ),
        "not_g1": True,
        "n_subjects": len(subjects),
        "n_fit_seeds": len(seeds),
        "n_model_fits": len(run_rows) * 2,
        "all_training_adequate": True,
        "counts": counts,
        "thresholds_frozen_before_run": thresholds,
        "patient_rows": patient_rows,
        "run_rows": run_rows,
        "old_heldout20_scored": False,
        "forbidden_inputs_read": False,
        "snn_inputs_read": False,
        "claim": (
            "This screen tests whether emitted-contact feedback adds value over "
            "a phase-matched no-graph model. It does not compare against the "
            "full baseline ladder and cannot establish G1 or stable structure."
        ),
        "fit_time_source_sha256": fit_provenance["source_sha256"],
        "fit_time_config_sha256": fit_provenance["config_sha256"],
        "input_sha256": input_sha256,
        "aggregation_source_sha256": sha256_file(Path(__file__)),
        "aggregation_config_sha256": sha256_file(config_path),
        "provenance_contract": (
            "Every run artifact carries the same config and source hashes, and "
            "those hashes equal the aggregating source. A mixed or missing "
            "hash raises before any count is computed."
        ),
    }
    _write(output / "HUMAN_GRAPH_INCREMENT_PILOT.json", payload)
    _write(
        output / "PILOT_STATE.json",
        {
            "status": "COMPLETE",
            "decision": payload["decision"],
            "not_g1": True,
            "missing_runs": 0,
            "failed_training_runs": 0,
        },
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    subjects = list(map(str, config["pilot"]["subjects"]))
    failures = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        futures = {
            pool.submit(
                _run_subject,
                subject,
                config_path=str(args.config),
                output_dir=str(args.output_dir),
            ): subject
            for subject in subjects
        }
        for future in as_completed(futures):
            subject = futures[future]
            try:
                state = future.result()
                print(json.dumps(state))
            except Exception as exc:
                failures.append({"subject": subject, "error": repr(exc)})
    if failures:
        _write(
            args.output_dir / "PILOT_STATE.json",
            {
                "status": "FAIL_CLOSED",
                "failures": failures,
            },
        )
        raise RuntimeError(f"human SIG pilot failed: {failures}")
    payload = _aggregate(args.config, args.output_dir)
    print(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
