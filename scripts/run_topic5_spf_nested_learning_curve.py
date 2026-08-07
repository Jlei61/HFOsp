#!/usr/bin/env python3
"""Run the target-blind nested event-count learning curve for SPF-RNN.

This is a bounded development experiment. It uses only the old train80 pool,
keeps the monitor and development-test partitions fixed, and constructs one
deterministic target-blind ordering of the inner-training events per patient.
Every fraction is a prefix of that same ordering, so the curve is genuinely
nested. The old outer heldout20 is never scored.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required; use the cuda_env environment") from exc

from scripts.run_topic5_spf_model_ladder import (  # noqa: E402
    _build,
    _model_seed,
    _score_repeated,
    _seed_everything,
    _subsample_chronological,
    _train_one,
)
from src.topic5_shared_propagation_field import (  # noqa: E402
    fit_static_scaffold_ml,
    load_subject_rank_events,
    sha256_file,
)

CONFIG_PATH = ROOT / "config/topic5_shared_propagation_field_v0_1.yaml"
PILOT_ROOT = (
    ROOT
    / "results/topic5_shared_propagation_field/development/ladder_pilot_v0_4"
)
OUTPUT_ROOT = (
    ROOT
    / "results/topic5_shared_propagation_field/development"
    / "multiround_review_2026-07-31/round3_nested_learning_curve"
)
MODELS = (
    "m2_markov_mixture_phase",
    "m3_template",
    "m4_field",
    "m4_field_phase",
)
SUBSET_SEED = 2026073101


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, ensure_ascii=False) + "\n")


def _indices_sha256(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def _pilot_subjects_seeds() -> tuple[list[str], list[int]]:
    rows = []
    for path in sorted(PILOT_ROOT.glob("*_seed*/summary.json")):
        payload = json.loads(path.read_text())
        rows.append((str(payload["subject"]), int(payload["seed"])))
    subjects = sorted({row[0] for row in rows})
    seeds = sorted({row[1] for row in rows})
    if len(subjects) != 6 or len(seeds) != 3:
        raise RuntimeError("frozen six-patient/three-seed pilot is incomplete")
    return subjects, seeds


def _nested_order(subject: str, train_indices: np.ndarray, cap: int) -> np.ndarray:
    subject_offset = int.from_bytes(
        hashlib.sha256(subject.encode("utf-8")).digest()[:8], "little"
    )
    rng = np.random.default_rng(SUBSET_SEED + subject_offset % 1_000_000_000)
    order = rng.permutation(np.asarray(train_indices, dtype=int))
    return order[: min(int(cap), len(order))]


def _fit_model(
    name: str,
    model: torch.nn.Module,
    groups: np.ndarray,
    counts: np.ndarray,
    train_indices: np.ndarray,
    monitor_indices: np.ndarray,
    *,
    device: torch.device,
    training: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    seed: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    initial_state = copy.deepcopy(model.state_dict())
    fitted = _train_one(
        name,
        model,
        groups,
        counts,
        train_indices,
        monitor_indices,
        device=device,
        training=training,
        evaluation=evaluation,
        seed=seed,
    )
    attempts = [
        {
            "label": "primary",
            "learning_rate": float(training["learning_rate"]),
            "adequacy": copy.deepcopy(fitted["adequacy"]),
        }
    ]
    if fitted["adequacy"]["verdict"] == "EARLY_OPTIMUM_UNVERIFIED":
        rescue = dict(training)
        rescue["learning_rate"] = float(training["learning_rate"]) * float(
            training["learning_rate_rescue_factor"]
        )
        model.load_state_dict(initial_state)
        _seed_everything(seed)
        fitted = _train_one(
            name,
            model,
            groups,
            counts,
            train_indices,
            monitor_indices,
            device=device,
            training=rescue,
            evaluation=evaluation,
            seed=seed,
        )
        attempts.append(
            {
                "label": "lower_learning_rate_rescue",
                "learning_rate": float(rescue["learning_rate"]),
                "adequacy": copy.deepcopy(fitted["adequacy"]),
            }
        )
    fitted["adequacy"] = {
        **fitted["adequacy"],
        "rescue_used": len(attempts) > 1,
        "n_training_attempts": len(attempts),
        "primary_attempt_verdict": attempts[0]["adequacy"]["verdict"],
    }
    fitted["training_attempts_compact"] = attempts
    model.load_state_dict(fitted["best_state"])
    model.eval()
    return model, fitted


def _worker(subject: str, seed: int, fraction: float) -> dict[str, Any]:
    torch.set_num_threads(1)
    config = yaml.safe_load(CONFIG_PATH.read_text())
    record = load_subject_rank_events(
        ROOT / config["data"]["dataset_dir"], subject
    )
    train_pool, monitor, test = record.development_split(
        float(config["data"]["inner_validation_fraction"]),
        float(config["data"]["inner_test_fraction"]),
    )
    ladder = config["ladder"]
    nested = _nested_order(
        subject, train_pool, int(ladder["max_train_events"])
    )
    n_train = max(1, int(round(float(fraction) * len(nested))))
    train = nested[:n_train]
    monitor = _subsample_chronological(
        monitor, int(ladder["max_validation_events"])
    )
    test = _subsample_chronological(test, int(ladder["max_test_events"]))
    if (
        np.intersect1d(train, monitor).size
        or np.intersect1d(train, test).size
        or np.intersect1d(monitor, test).size
    ):
        raise RuntimeError("nested learning-curve partitions overlap")
    if np.intersect1d(
        np.r_[train, monitor, test], record.old_heldout20_indices
    ).size:
        raise RuntimeError("old heldout20 entered nested learning curve")
    device = torch.device("cpu")
    _seed_everything(seed)
    scaffold = fit_static_scaffold_ml(
        record.group_ids,
        record.group_count,
        train,
        steps=int(ladder["scaffold_steps"]),
        learning_rate=float(ladder["scaffold_learning_rate"]),
        seed=seed,
        device=device,
    )
    model_config = dict(config["model"])
    model_config["mixture_components"] = int(ladder["mixture_components"])
    training = dict(config["training"])
    evaluation = dict(config["evaluation"])
    trained: dict[str, tuple[torch.nn.Module, dict[str, Any], float]] = {}
    for name in MODELS:
        model_seed = _model_seed(seed, name)
        _seed_everything(model_seed)
        model = _build(
            name, len(record.contact_names), scaffold, model_config
        ).to(device)
        started = time.time()
        model, fitted = _fit_model(
            name,
            model,
            record.group_ids,
            record.group_count,
            train,
            monitor,
            device=device,
            training=training,
            evaluation=evaluation,
            seed=model_seed,
        )
        trained[name] = (model, fitted, time.time() - started)
    test_groups = torch.as_tensor(record.group_ids[test], dtype=torch.long)
    test_counts = torch.as_tensor(record.group_count[test], dtype=torch.long)
    results: dict[str, Any] = {}
    for name, (model, fitted, elapsed) in trained.items():
        score = _score_repeated(
            model,
            test_groups,
            test_counts,
            prior_samples=64,
            importance_samples=64,
            repeats=2,
            seed=seed + 211,
        )
        results[name] = {
            "nll_per_decision": score["nll_per_decision"],
            "nll_per_decision_mc_sd": score["nll_per_decision_mc_sd"],
            "prior_predictive_nll_per_decision": score[
                "prior_predictive_nll_per_decision"
            ],
            "prior_predictive_nll_per_decision_mc_sd": score[
                "prior_predictive_nll_per_decision_mc_sd"
            ],
            "estimator": score["estimator"],
            "training_adequacy": fitted["adequacy"],
            "training_attempts": fitted["training_attempts_compact"],
            "training_elapsed_seconds": elapsed,
        }
    output = {
        "status": "COMPLETE",
        "contract": "topic5_spf_nested_learning_curve_v0_1",
        "subject": subject,
        "seed": int(seed),
        "fraction": float(fraction),
        "n_train_events": int(len(train)),
        "n_nested_max_events": int(len(nested)),
        "nested_subset_seed": SUBSET_SEED,
        "nested_max_indices_sha256": _indices_sha256(nested),
        "train_indices_sha256": _indices_sha256(train),
        "monitor_indices_sha256": _indices_sha256(monitor),
        "development_test_indices_sha256": _indices_sha256(test),
        "old_heldout20_scored": False,
        "config_sha256": sha256_file(CONFIG_PATH),
        "input_sha256": record.input_sha256,
        "models": results,
    }
    label = f"{subject}_seed{seed}_fraction{fraction:.2f}".replace(".", "p")
    _write_json(OUTPUT_ROOT / "per_run" / f"{label}.json", output)
    return output


def _aggregate(outputs: list[dict[str, Any]], fractions: list[float]) -> None:
    run_rows: list[dict[str, Any]] = []
    for output in outputs:
        for name, model in output["models"].items():
            run_rows.append(
                {
                    "subject": output["subject"],
                    "seed": output["seed"],
                    "fraction": output["fraction"],
                    "n_train_events": output["n_train_events"],
                    "model": name,
                    "nll_per_decision": model["nll_per_decision"],
                    "prior_predictive_nll_per_decision": model[
                        "prior_predictive_nll_per_decision"
                    ],
                    "training_verdict": model["training_adequacy"]["verdict"],
                    "converged": model["training_adequacy"]["converged"],
                    "rescue_used": model["training_adequacy"]["rescue_used"],
                    "training_elapsed_seconds": model[
                        "training_elapsed_seconds"
                    ],
                }
            )
    with (OUTPUT_ROOT / "learning_curve_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(run_rows[0]))
        writer.writeheader()
        writer.writerows(run_rows)

    patient_rows: list[dict[str, Any]] = []
    grouping = lambda row: (row["subject"], row["fraction"], row["model"])
    for key, values in itertools.groupby(
        sorted(run_rows, key=grouping), key=grouping
    ):
        rows = list(values)
        patient_rows.append(
            {
                "subject": key[0],
                "fraction": key[1],
                "n_train_events": int(
                    np.median([row["n_train_events"] for row in rows])
                ),
                "model": key[2],
                "nll_per_decision_mean_across_seeds": float(
                    np.mean([row["nll_per_decision"] for row in rows])
                ),
                "prior_predictive_nll_per_decision_mean_across_seeds": float(
                    np.mean(
                        [
                            row["prior_predictive_nll_per_decision"]
                            for row in rows
                        ]
                    )
                ),
                "n_converged": int(np.sum([row["converged"] for row in rows])),
                "n_seeds": len(rows),
            }
        )
    with (OUTPUT_ROOT / "learning_curve_patient.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(patient_rows[0]))
        writer.writeheader()
        writer.writerows(patient_rows)

    lookup = {
        (row["subject"], row["fraction"], row["model"]): row
        for row in patient_rows
    }
    contrast_rows = []
    subjects = sorted({row["subject"] for row in patient_rows})
    for subject in subjects:
        for fraction in fractions:
            for left, right, label in (
                ("m4_field", "m3_template", "m4_minus_m3"),
                (
                    "m4_field_phase",
                    "m3_template",
                    "m4phase_minus_m3",
                ),
                (
                    "m4_field",
                    "m2_markov_mixture_phase",
                    "m4_minus_m2phase",
                ),
            ):
                left_row = lookup[(subject, fraction, left)]
                right_row = lookup[(subject, fraction, right)]
                contrast_rows.append(
                    {
                        "subject": subject,
                        "fraction": fraction,
                        "n_train_events": left_row["n_train_events"],
                        "comparison": label,
                        "delta_nll_per_decision": (
                            left_row["nll_per_decision_mean_across_seeds"]
                            - right_row[
                                "nll_per_decision_mean_across_seeds"
                            ]
                        ),
                        "delta_prior_predictive_nll_per_decision": (
                            left_row[
                                "prior_predictive_nll_per_decision_mean_across_seeds"
                            ]
                            - right_row[
                                "prior_predictive_nll_per_decision_mean_across_seeds"
                            ]
                        ),
                    }
                )
    with (OUTPUT_ROOT / "learning_curve_contrasts.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(contrast_rows[0]))
        writer.writeheader()
        writer.writerows(contrast_rows)
    _write_json(
        OUTPUT_ROOT / "ROUND_STATE.json",
        {
            "status": "COMPLETE",
            "round": 3,
            "question": (
                "Does an autonomous-field inductive bias help at low event "
                "counts under a genuinely nested target-blind design?"
            ),
            "fractions": fractions,
            "n_jobs": len(outputs),
            "n_models_per_job": len(MODELS),
            "nested_subset_seed": SUBSET_SEED,
            "old_heldout20_scored": False,
            "selection_rule": (
                "checkpoint selected on the fixed monitor partition; "
                "development test opened only after all four models fit"
            ),
            "source_sha256": sha256_file(Path(__file__)),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=(0.10, 0.20, 0.40, 0.60, 0.80, 1.00),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fractions = sorted({float(value) for value in args.fractions})
    if not fractions or fractions[0] <= 0.0 or fractions[-1] > 1.0:
        raise ValueError("fractions must lie in (0, 1]")
    subjects, seeds = _pilot_subjects_seeds()
    jobs = list(itertools.product(subjects, seeds, fractions))
    outputs = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        futures = {
            pool.submit(_worker, subject, seed, fraction): (
                subject,
                seed,
                fraction,
            )
            for subject, seed, fraction in jobs
        }
        for future in as_completed(futures):
            output = future.result()
            outputs.append(output)
            print(
                f"complete {output['subject']} seed={output['seed']} "
                f"fraction={output['fraction']:.2f}",
                flush=True,
            )
    _aggregate(outputs, fractions)


if __name__ == "__main__":
    main()
