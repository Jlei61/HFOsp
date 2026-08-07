#!/usr/bin/env python3
"""D0 patient-matched sensitivity and specificity for current SIG inference."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_topic5_spf_model_ladder import _model_seed  # noqa: E402
from src.topic5_shared_propagation_field import (  # noqa: E402
    fit_static_scaffold_ml,
    load_subject_rank_events,
    sample_conditional_k_subset,
    sha256_file,
)
from src.topic5_stable_interaction_graph import (  # noqa: E402
    StableInteractionGraph,
    SyntheticGraphDataset,
    cardinality_schedule,
    fit_synthetic_sig,
    phase_basis,
)


CONFIG = ROOT / "config/topic5_stable_interaction_identifiability_v2_1_d0.yaml"
OUTPUT = (
    ROOT
    / "results/topic5_stable_interaction_graph/development"
    / "v2_1_patient_matched_identifiability"
)
CONDITIONS = (
    "shared_graph_positive",
    "phase_template_negative",
    "mixture_zero_mean_backbone_negative",
    "event_random_graph_negative",
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
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n")


def _source_sha256() -> dict[str, str]:
    return {
        "runner": sha256_file(Path(__file__)),
        "sig_model": sha256_file(ROOT / "src/topic5_stable_interaction_graph.py"),
        "spf_model": sha256_file(ROOT / "src/topic5_shared_propagation_field.py"),
    }


def _subsample(indices: np.ndarray, limit: int) -> np.ndarray:
    values = np.asarray(indices, dtype=int)
    if len(values) <= int(limit):
        return values
    return values[np.linspace(0, len(values) - 1, int(limit)).astype(int)]


def _dataset(groups: np.ndarray, counts: np.ndarray, indices: np.ndarray) -> SyntheticGraphDataset:
    value = groups[indices].copy()
    return SyntheticGraphDataset(
        value,
        counts[indices].copy(),
        np.argmax(value == 0, axis=1).astype(np.int16),
    )


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float).ravel()
    b = np.asarray(right, dtype=float).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    if np.sum(valid) < 10 or np.std(a[valid]) == 0 or np.std(b[valid]) == 0:
        return float("nan")
    return float(spearmanr(a[valid], b[valid]).statistic)


def _truth(subject: str, contacts: int, scaffold: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    seed = int.from_bytes(subject.encode("utf-8"), "little") % (2**32 - 1)
    rng = np.random.default_rng(seed)
    positive = float(config["model"]["positive_edge_weight"])
    background = float(config["model"]["background_weight"])
    base = np.full((contacts, contacts), background, dtype=np.float32)
    np.fill_diagonal(base, 0.0)
    for source in range(contacts):
        targets = {(source + 1) % contacts, (source + max(2, contacts // 3)) % contacts}
        targets.discard(source)
        for target in targets:
            base[target, source] = positive
    phase = rng.normal(0.0, 0.12, size=(contacts, 3)).astype(np.float32)
    phase[:, 0] += np.linspace(-0.25, 0.25, contacts, dtype=np.float32)
    permutations = [rng.permutation(contacts) for _ in range(3)]
    mixture = np.stack([base[p][:, p] for p in permutations])
    mixture -= mixture.mean(axis=0, keepdims=True)
    maximum = float(np.max(np.abs(mixture)))
    if maximum > 2.7:
        mixture *= 2.7 / maximum
    return {
        "base_weight": base,
        "mixture_weight": mixture.astype(np.float32),
        "phase_loading": phase,
        "static_bias": np.asarray(scaffold, dtype=np.float32),
        "leak": float(config["model"]["leak"]),
        "seed": seed,
    }


def _simulate_condition(
    conditioning_groups: np.ndarray,
    counts: np.ndarray,
    truth: dict[str, Any],
    condition: str,
    *,
    seed: int,
) -> np.ndarray:
    contacts = conditioning_groups.shape[1]
    schedules = cardinality_schedule(conditioning_groups, counts)
    output = np.full(conditioning_groups.shape, -1, dtype=np.int16)
    first = conditioning_groups == 0
    output[first] = 0
    rng = np.random.default_rng(int(seed))
    torch_generator = torch.Generator().manual_seed(int(seed) + 101)
    base = np.asarray(truth["base_weight"], dtype=np.float32)
    mixture = np.asarray(truth["mixture_weight"], dtype=np.float32)
    for event_index, count in enumerate(counts):
        if condition == "shared_graph_positive":
            weight = base
        elif condition == "phase_template_negative":
            weight = np.zeros_like(base)
        elif condition == "mixture_zero_mean_backbone_negative":
            weight = mixture[int(rng.integers(0, 3))]
        elif condition == "event_random_graph_negative":
            order = rng.permutation(contacts)
            weight = base[order][:, order]
        else:
            raise ValueError(condition)
        state = np.zeros(contacts, dtype=np.float32)
        previous = first[event_index].astype(np.float32)
        recruited = first[event_index].copy()
        for step in range(1, int(count)):
            state = float(truth["leak"]) * state + np.tanh(weight @ previous)
            phi = torch.tensor([step / max(int(count) - 1, 1)], dtype=torch.float32)
            basis = phase_basis(phi)[0].numpy()
            logits = (
                np.asarray(truth["static_bias"])
                + state
                + np.asarray(truth["phase_loading"]) @ basis
            )
            cardinality = int(schedules[event_index, step - 1])
            selected = sample_conditional_k_subset(
                torch.as_tensor(logits[None, :], dtype=torch.float32),
                torch.as_tensor((~recruited)[None, :], dtype=torch.bool),
                torch.tensor([cardinality], dtype=torch.long),
                generator=torch_generator,
            )[0].numpy()
            output[event_index, selected] = step
            recruited |= selected
            previous = selected.astype(np.float32)
    return output


def _truth_model(truth: dict[str, Any]) -> StableInteractionGraph:
    contacts = len(truth["static_bias"])
    model = StableInteractionGraph(
        contacts,
        static_bias=truth["static_bias"],
        learn_graph=True,
        initial_leak=float(truth["leak"]),
    )
    with torch.no_grad():
        model.raw_weight.copy_(
            torch.atanh(
                torch.as_tensor(truth["base_weight"] / model.max_weight).clamp(
                    -0.999, 0.999
                )
            )
        )
        model.phase_loading.copy_(torch.as_tensor(truth["phase_loading"]))
    model.eval()
    return model


def _run_subject(subject: str, *, output_dir: str) -> dict[str, Any]:
    torch.set_num_threads(2)
    config = yaml.safe_load(CONFIG.read_text())
    data = config["data"]
    training = config["training"]
    record = load_subject_rank_events(ROOT / data["dataset_dir"], subject)
    inner_train, probe, _ = record.development_split(
        float(data["validation_fraction"]), float(data["test_fraction"])
    )
    midpoint = len(inner_train) // 2
    partitions = {}
    for half, values in (
        ("early", inner_train[:midpoint]), ("late", inner_train[midpoint:])
    ):
        cut = max(1, int(np.floor(len(values) * 0.85)))
        partitions[f"{half}_train"] = _subsample(
            values[:cut], int(data["max_train_events_per_half"])
        )
        partitions[f"{half}_monitor"] = _subsample(
            values[cut:], int(data["max_monitor_events_per_half"])
        )
    probe = _subsample(probe, int(data["max_probe_events"]))
    allowed = np.unique(np.r_[*partitions.values(), probe])
    if np.intersect1d(allowed, record.old_heldout20_indices).size:
        raise RuntimeError(f"{subject}: old heldout20 leakage")
    real_scaffold = fit_static_scaffold_ml(
        record.group_ids,
        record.group_count,
        partitions["early_train"],
        steps=int(training["static_scaffold_steps"]),
        learning_rate=float(training["static_scaffold_learning_rate"]),
        seed=int(training["static_scaffold_seed"]),
        device="cpu",
    )
    truth = _truth(subject, len(record.contact_names), real_scaffold, config)
    condition_groups = {}
    for ordinal, condition in enumerate(CONDITIONS):
        groups = record.group_ids.copy()
        groups[allowed] = _simulate_condition(
            record.group_ids[allowed],
            record.group_count[allowed],
            truth,
            condition,
            seed=20260821 + ordinal * 1009 + int(truth["seed"] % 1000),
        )
        condition_groups[condition] = groups

    operators: dict[str, dict[str, dict[int, np.ndarray]]] = {
        condition: {"early": {}, "late": {}} for condition in CONDITIONS
    }
    run_rows = []
    for condition in CONDITIONS:
        groups = condition_groups[condition]
        for half in ("early", "late"):
            train_index = partitions[f"{half}_train"]
            monitor_index = partitions[f"{half}_monitor"]
            scaffold = fit_static_scaffold_ml(
                groups,
                record.group_count,
                train_index,
                steps=int(training["static_scaffold_steps"]),
                learning_rate=float(training["static_scaffold_learning_rate"]),
                seed=int(training["static_scaffold_seed"]),
                device="cpu",
            )
            for fit_seed in map(int, training["fit_seeds"]):
                fitted = fit_synthetic_sig(
                    _dataset(groups, record.group_count, train_index),
                    _dataset(groups, record.group_count, monitor_index),
                    seed=_model_seed(fit_seed, f"v2_1_d0_{condition}_{half}"),
                    learn_graph=True,
                    max_epochs=int(training["epochs"]),
                    patience=int(training["patience"]),
                    learning_rate=float(training["learning_rate"]),
                    l1_weight=float(config["model"]["graph_l1_weight"]),
                    batch_size=int(training["batch_events"]),
                    static_bias=scaffold,
                    convergence_tolerance=float(training["convergence_tolerance"]),
                    minimum_relative_improvement=float(training["minimum_relative_improvement"]),
                    minimum_training_epochs=int(training["minimum_training_epochs"]),
                    minimum_best_epoch=int(training["minimum_best_epoch"]),
                    maximum_recovery_depth=int(training["maximum_recovery_depth"]),
                )
                if not fitted.adequacy["converged"]:
                    raise RuntimeError(
                        f"{subject} {condition} {half} {fit_seed}: inadequate fit"
                    )
                operator, support = fitted.model.empirical_marginal_intervention_matrix(
                    torch.as_tensor(groups[probe], dtype=torch.long),
                    torch.as_tensor(record.group_count[probe], dtype=torch.long),
                    return_support=True,
                )
                operators[condition][half][fit_seed] = operator.numpy()
                run_rows.append(
                    {
                        "condition": condition,
                        "half": half,
                        "fit_seed": fit_seed,
                        "best_validation_nll": fitted.best_validation_nll,
                        "training_adequacy": fitted.adequacy,
                        "supported_pair_fraction": float(np.mean(support.numpy() > 0)),
                    }
                )

    truth_operator = _truth_model(truth).empirical_marginal_intervention_matrix(
        torch.as_tensor(
            condition_groups["shared_graph_positive"][probe], dtype=torch.long
        ),
        torch.as_tensor(record.group_count[probe], dtype=torch.long),
    ).numpy()
    condition_rows = {}
    for condition in CONDITIONS:
        stability = [
            _spearman(
                operators[condition]["early"][left],
                operators[condition]["late"][right],
            )
            for left, right in itertools.product(
                map(int, training["fit_seeds"]), repeat=2
            )
        ]
        row = {
            "cross_seed_half_stability_median": float(np.nanmedian(stability)),
            "cross_seed_half_stability_min": float(np.nanmin(stability)),
        }
        if condition == "shared_graph_positive":
            recovery = [
                _spearman(operator, truth_operator)
                for half in ("early", "late")
                for operator in operators[condition][half].values()
            ]
            row["truth_recovery_spearman_median"] = float(np.nanmedian(recovery))
            row["truth_recovery_spearman_min"] = float(np.nanmin(recovery))
        condition_rows[condition] = row

    positive = condition_rows["shared_graph_positive"]
    strongest_negative = max(
        condition_rows[name]["cross_seed_half_stability_median"]
        for name in CONDITIONS if name != "shared_graph_positive"
    )
    margin = positive["cross_seed_half_stability_median"] - strongest_negative
    frozen = config["frozen_decision"]
    gates = {
        "positive_truth_recovery": (
            positive["truth_recovery_spearman_median"]
            >= float(frozen["minimum_positive_recovery_spearman"])
        ),
        "positive_split_stability": (
            positive["cross_seed_half_stability_median"]
            >= float(frozen["minimum_positive_split_stability"])
        ),
        "specificity_margin": (
            margin
            >= float(
                frozen["minimum_positive_minus_strongest_negative_stability"]
            )
        ),
    }
    payload = {
        "contract": config["contract"]["name"],
        "subject": subject,
        "dataset": record.dataset,
        "n_contacts": len(record.contact_names),
        "partitions": {name: len(value) for name, value in partitions.items()},
        "n_probe_events": len(probe),
        "conditions": condition_rows,
        "positive_minus_strongest_negative_stability": margin,
        "frozen_gates": gates,
        "status": "PASS" if all(gates.values()) else "NOT_PASSED",
        "run_rows": run_rows,
        "input_sha256": record.input_sha256,
        "config_sha256": sha256_file(CONFIG),
        "source_sha256": _source_sha256(),
        "old_heldout20_scored": False,
        "forbidden_inputs_read": False,
        "snn_inputs_read": False,
    }
    subject_dir = Path(output_dir) / "per_subject" / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        subject_dir / "observable_operators.npz",
        truth_positive=truth_operator,
        **{
            f"{condition}_{half}_{seed}": value
            for condition in CONDITIONS
            for half in ("early", "late")
            for seed, value in operators[condition][half].items()
        },
    )
    _write(subject_dir / "summary.json", payload)
    return {"status": payload["status"], "subject": subject}


def _aggregate(output: Path) -> dict[str, Any]:
    config = yaml.safe_load(CONFIG.read_text())
    patients = [
        json.loads((output / "per_subject" / subject / "summary.json").read_text())
        for subject in config["pilot"]["subjects"]
    ]
    payload = {
        "contract": config["contract"]["name"],
        "status": "COMPLETE_PATIENT_MATCHED_IDENTIFIABILITY",
        "n_subjects": len(patients),
        "n_pass": sum(row["status"] == "PASS" for row in patients),
        "n_model_fits": sum(len(row["run_rows"]) for row in patients),
        "all_training_adequate": all(
            run["training_adequacy"]["converged"]
            for row in patients for run in row["run_rows"]
        ),
        "decision_boundary": (
            "A patient is structurally interpretable only if sensitivity, "
            "positive split stability, and the pre-frozen specificity margin "
            "all pass at that patient's observed C/N/T/k/start support."
        ),
        "patients": patients,
        "old_heldout20_scored": False,
        "forbidden_inputs_read": False,
        "snn_inputs_read": False,
        "source_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(CONFIG),
    }
    _write(output / "D0_PATIENT_MATCHED_IDENTIFIABILITY.json", payload)
    _write(output / "D0_STATE.json", {"status": payload["status"], "n_pass": payload["n_pass"], "n_subjects": payload["n_subjects"]})
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text())
    failures = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        futures = {
            pool.submit(_run_subject, subject, output_dir=str(args.output_dir)): subject
            for subject in config["pilot"]["subjects"]
        }
        for future in as_completed(futures):
            subject = futures[future]
            try:
                print(json.dumps(future.result()))
            except Exception as exc:
                failures.append({"subject": subject, "error": repr(exc)})
                print(json.dumps(failures[-1]))
    if failures:
        _write(args.output_dir / "D0_STATE.json", {"status": "FAIL_CLOSED", "failures": failures})
        raise RuntimeError(f"patient-matched D0 failures: {failures}")
    payload = _aggregate(args.output_dir)
    print(json.dumps({"n_pass": payload["n_pass"], "n_subjects": payload["n_subjects"]}, indent=2))


if __name__ == "__main__":
    main()
