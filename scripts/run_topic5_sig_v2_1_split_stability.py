#!/usr/bin/env python3
"""Run D3 chronological SIG influence stability against matched surrogates."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
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

from scripts.run_topic5_sig_matched_baseline_ladder import _build as _build_baseline  # noqa: E402
from scripts.run_topic5_spf_model_ladder import _model_seed  # noqa: E402
from src.topic5_shared_propagation_field import (  # noqa: E402
    fit_static_scaffold_ml,
    load_subject_rank_events,
    sha256_file,
)
from src.topic5_stable_interaction_graph import (  # noqa: E402
    StableInteractionGraph,
    SyntheticGraphDataset,
    fit_synthetic_sig,
)


CONFIG = ROOT / "config/topic5_stable_interaction_identifiability_v2_1.yaml"
LADDER_ROOT = (
    ROOT
    / "results/topic5_stable_interaction_graph/development"
    / "human_matched_baseline_ladder_v0_2_training_adequacy"
)
OUTPUT = (
    ROOT
    / "results/topic5_stable_interaction_graph/development"
    / "v2_1_split_stability"
)
CONDITIONS = ("real", "m1_phase_surrogate", "m3_template_surrogate")


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


def _load_null_generator(subject: str, model_name: str, contacts: int, seed: int):
    checkpoint = torch.load(
        LADDER_ROOT / "per_run" / subject / f"seed_{seed}/checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint["models"][model_name]["state_dict"]
    model = _build_baseline(model_name, contacts, state["static_bias"].numpy())
    model.load_state_dict(state)
    model.eval()
    return model


def _generate_surrogate(model, groups: np.ndarray, counts: np.ndarray, *, seed: int) -> np.ndarray:
    tensor_groups = torch.as_tensor(groups, dtype=torch.long)
    tensor_counts = torch.as_tensor(counts, dtype=torch.long)
    with torch.no_grad():
        return model.generate_conditioned(
            tensor_groups, tensor_counts, seed=int(seed)
        ).numpy().astype(np.int16)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float).ravel()
    b = np.asarray(right, dtype=float).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    if np.sum(valid) < 10 or np.std(a[valid]) == 0 or np.std(b[valid]) == 0:
        return float("nan")
    return float(spearmanr(a[valid], b[valid]).statistic)


def _label_permutation_distribution(
    early: np.ndarray, late: np.ndarray, *, n: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    values = []
    for _ in range(int(n)):
        order = rng.permutation(late.shape[0])
        values.append(_spearman(early, late[order][:, order]))
    return np.asarray(values, dtype=float)


def _half_partitions(indices: np.ndarray, monitor_fraction: float) -> dict[str, np.ndarray]:
    midpoint = len(indices) // 2
    output = {}
    for name, values in (
        ("early", indices[:midpoint]), ("late", indices[midpoint:])
    ):
        cut = max(1, int(np.floor(len(values) * (1.0 - monitor_fraction))))
        output[f"{name}_train"] = values[:cut]
        output[f"{name}_monitor"] = values[cut:]
    return output


def _run_subject(subject: str, *, output_dir: str) -> dict[str, Any]:
    torch.set_num_threads(2)
    config = yaml.safe_load(CONFIG.read_text())
    data = config["data"]
    rules = config["split_stability"]
    training = config["training"]
    record = load_subject_rank_events(ROOT / data["dataset_dir"], subject)
    inner_train, common_probe, _ = record.development_split(
        float(data["validation_fraction"]), float(data["test_fraction"])
    )
    partitions = _half_partitions(
        inner_train, float(rules["within_half_monitor_fraction"])
    )
    for half in ("early", "late"):
        partitions[f"{half}_train"] = _subsample(
            partitions[f"{half}_train"], int(rules["max_train_events_per_half"])
        )
        partitions[f"{half}_monitor"] = _subsample(
            partitions[f"{half}_monitor"], int(rules["max_monitor_events_per_half"])
        )
    common_probe = _subsample(common_probe, int(rules["max_common_probe_events"]))
    all_used = np.concatenate([*partitions.values(), common_probe])
    if np.intersect1d(all_used, record.old_heldout20_indices).size:
        raise RuntimeError(f"{subject}: old heldout20 leakage")

    null_seed = int(rules["null_generator_fit_seed"])
    m1 = _load_null_generator(
        subject, "m1_markov_matched_phase", len(record.contact_names), null_seed
    )
    m3 = _load_null_generator(
        subject, "m3_latent_template", len(record.contact_names), null_seed
    )
    # Generate only the development indices used by D3. The historical outer
    # heldout20 may be present in the frozen file, but it must not influence a
    # surrogate even through conditioning or RNG order.
    allowed = np.unique(np.r_[inner_train, common_probe])
    m1_groups = record.group_ids.copy()
    m3_groups = record.group_ids.copy()
    m1_groups[allowed] = _generate_surrogate(
        m1,
        record.group_ids[allowed],
        record.group_count[allowed],
        seed=20260811,
    )
    m3_groups[allowed] = _generate_surrogate(
        m3,
        record.group_ids[allowed],
        record.group_count[allowed],
        seed=20260812,
    )
    condition_groups = {
        "real": record.group_ids.copy(),
        "m1_phase_surrogate": m1_groups,
        "m3_template_surrogate": m3_groups,
    }
    output = Path(output_dir)
    operators: dict[str, dict[str, dict[int, np.ndarray]]] = {
        condition: {"early": {}, "late": {}} for condition in CONDITIONS
    }
    run_rows = []
    checkpoints = {}
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
                model_seed = _model_seed(
                    fit_seed, f"v2_1_split_{condition}_{half}"
                )
                fitted = fit_synthetic_sig(
                    _dataset(groups, record.group_count, train_index),
                    _dataset(groups, record.group_count, monitor_index),
                    seed=model_seed,
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
                probe_groups = torch.as_tensor(groups[common_probe], dtype=torch.long)
                probe_counts = torch.as_tensor(
                    record.group_count[common_probe], dtype=torch.long
                )
                operator, support = fitted.model.empirical_marginal_intervention_matrix(
                    probe_groups, probe_counts, return_support=True
                )
                operators[condition][half][fit_seed] = operator.numpy()
                run_rows.append(
                    {
                        "condition": condition,
                        "half": half,
                        "fit_seed": fit_seed,
                        "best_validation_nll": fitted.best_validation_nll,
                        "training_adequacy": fitted.adequacy,
                        "supported_pair_fraction": float(
                            np.mean(support.numpy() > 0)
                        ),
                    }
                )
                checkpoints[f"{condition}/{half}/{fit_seed}"] = {
                    "state_dict": fitted.model.state_dict(),
                    "optimizer_state": fitted.best_optimizer_state,
                }

    condition_rows = {}
    for condition in CONDITIONS:
        correlations = []
        same_seed = []
        permutation_values = []
        for early_seed, late_seed in itertools.product(
            map(int, training["fit_seeds"]), repeat=2
        ):
            rho = _spearman(
                operators[condition]["early"][early_seed],
                operators[condition]["late"][late_seed],
            )
            correlations.append(rho)
            if early_seed == late_seed:
                same_seed.append(rho)
        for fit_seed in map(int, training["fit_seeds"]):
            permutation_values.extend(
                _label_permutation_distribution(
                    operators[condition]["early"][fit_seed],
                    operators[condition]["late"][fit_seed],
                    n=int(rules["contact_label_permutations"]),
                    seed=fit_seed + 71,
                ).tolist()
            )
        condition_rows[condition] = {
            "cross_seed_half_stability_median": float(np.nanmedian(correlations)),
            "cross_seed_half_stability_min": float(np.nanmin(correlations)),
            "same_seed_half_stability_median": float(np.nanmedian(same_seed)),
            "contact_permutation_q95": float(
                np.nanquantile(permutation_values, 0.95)
            ),
            "n_cross_half_comparisons": len(correlations),
        }
    strongest_null = max(
        condition_rows["m1_phase_surrogate"]["cross_seed_half_stability_median"],
        condition_rows["m3_template_surrogate"]["cross_seed_half_stability_median"],
    )
    payload = {
        "contract": config["contract"]["name"],
        "subject": subject,
        "dataset": record.dataset,
        "n_contacts": len(record.contact_names),
        "partitions": {name: len(value) for name, value in partitions.items()},
        "n_common_probe_events": len(common_probe),
        "conditions": condition_rows,
        "real_minus_strongest_matched_null_stability": (
            condition_rows["real"]["cross_seed_half_stability_median"]
            - strongest_null
        ),
        "operator": "SUPPORTED_MARGINAL_SENDER_RESPONSE_ON_COMMON_PROBE",
        "static_phase_response": "REMOVED_BY_WITHIN_CONTEXT_SENDER_CONTROL",
        "run_rows": run_rows,
        "input_sha256": record.input_sha256,
        "config_sha256": sha256_file(CONFIG),
        "source_sha256": _source_sha256(),
        "old_heldout20_scored": False,
        "forbidden_inputs_read": False,
        "snn_inputs_read": False,
    }
    subject_dir = output / "per_subject" / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"contract": payload["contract"], "subject": subject, "models": checkpoints},
        subject_dir / "checkpoints.pt",
    )
    np.savez_compressed(
        subject_dir / "observable_operators.npz",
        **{
            f"{condition}_{half}_{seed}": value
            for condition in CONDITIONS
            for half in ("early", "late")
            for seed, value in operators[condition][half].items()
        },
    )
    _write(subject_dir / "summary.json", payload)
    return {"status": "COMPLETE", "subject": subject}


def _aggregate(output: Path) -> dict[str, Any]:
    config = yaml.safe_load(CONFIG.read_text())
    subjects = list(map(str, config["pilot"]["subjects"]))
    patients = [
        json.loads((output / "per_subject" / subject / "summary.json").read_text())
        for subject in subjects
    ]
    deltas = np.asarray(
        [row["real_minus_strongest_matched_null_stability"] for row in patients],
        dtype=float,
    )
    payload = {
        "contract": config["contract"]["name"],
        "status": "COMPLETE_SPLIT_STABILITY_DEVELOPMENT",
        "n_subjects": len(patients),
        "n_model_fits": sum(len(row["run_rows"]) for row in patients),
        "all_training_adequate": all(
            run["training_adequacy"]["converged"]
            for row in patients for run in row["run_rows"]
        ),
        "real_minus_strongest_null": {
            "median": float(np.nanmedian(deltas)),
            "n_positive": int(np.sum(deltas > 0)),
            "values": deltas.tolist(),
        },
        "interpretation_boundary": (
            "Development structural signal requires real-minus-null stability; "
            "absolute real stability or seed reproducibility alone is not sufficient."
        ),
        "patients": patients,
        "old_heldout20_scored": False,
        "forbidden_inputs_read": False,
        "snn_inputs_read": False,
        "source_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(CONFIG),
    }
    _write(output / "D3_SPLIT_STABILITY.json", payload)
    _write(output / "D3_STATE.json", {"status": payload["status"], "real_minus_strongest_null": payload["real_minus_strongest_null"]})
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text())
    subjects = list(map(str, config["pilot"]["subjects"]))
    failures = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        futures = {
            pool.submit(_run_subject, subject, output_dir=str(args.output_dir)): subject
            for subject in subjects
        }
        for future in as_completed(futures):
            subject = futures[future]
            try:
                print(json.dumps(future.result()))
            except Exception as exc:
                failures.append({"subject": subject, "error": repr(exc)})
                print(json.dumps(failures[-1]))
    if failures:
        _write(args.output_dir / "D3_STATE.json", {"status": "FAIL_CLOSED", "failures": failures})
        raise RuntimeError(f"split stability failures: {failures}")
    payload = _aggregate(args.output_dir)
    print(json.dumps(payload["real_minus_strongest_null"], indent=2))


if __name__ == "__main__":
    main()
