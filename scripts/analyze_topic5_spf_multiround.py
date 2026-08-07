#!/usr/bin/env python3
"""Post-hoc diagnostics for the frozen SPF-RNN v0.4 development pilot.

The script only re-scores saved checkpoints on the already frozen development
test split. It never selects or scores an old outer-heldout20 event and never
retrains a model.  The frozen subject NPZ co-locates both split arrays, so this
is an analysis-use guarantee rather than a byte-level file-read claim.

Rounds implemented here:

1. likelihood-estimator convergence and importance-weight diagnostics;
2. event-length and suffix-step decomposition of the model gaps;
4. seed stability of observable, first-rank-conditioned response summaries.

Rounds 3, 5, and 6 require new fits and live in separate runners so their
provenance cannot be confused with checkpoint-only analyses.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required; use the cuda_env environment") from exc

from src.topic5_shared_propagation_field import (  # noqa: E402
    LatentTemplateModel,
    MarkovMixtureModel,
    PhaseConditionedPropagationFieldRNN,
    SharedPropagationFieldRNN,
    diagonal_gaussian_log_prob,
    load_subject_rank_events,
    sha256_file,
)

PILOT_ROOT = (
    ROOT
    / "results/topic5_shared_propagation_field/development/ladder_pilot_v0_4"
)
OUTPUT_ROOT = (
    ROOT
    / "results/topic5_shared_propagation_field/development"
    / "multiround_review_2026-07-31"
)
CONFIG_PATH = ROOT / "config/topic5_shared_propagation_field_v0_1.yaml"
LATENT_MODELS = ("m3_template", "m4_field", "m4_field_phase")
RESPONSE_MODELS = (
    "m2_markov_mixture_phase",
    "m3_template",
    "m4_field",
    "m4_field_phase",
)


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _indices_sha256(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def _subsample_chronological(indices: np.ndarray, limit: int | None) -> np.ndarray:
    values = np.asarray(indices, dtype=int)
    if limit is None or len(values) <= int(limit):
        return values
    return values[np.linspace(0, len(values) - 1, int(limit)).astype(int)]


def _run_dirs() -> list[Path]:
    values = [
        path
        for path in PILOT_ROOT.iterdir()
        if path.is_dir()
        and (path / "checkpoint.pt").exists()
        and (path / "summary.json").exists()
    ]
    if len(values) != 18:
        raise RuntimeError(f"expected 18 complete pilot runs, found {len(values)}")
    return sorted(values)


def _build(
    name: str,
    n_contacts: int,
    scaffold: np.ndarray,
    model_config: Mapping[str, Any],
) -> torch.nn.Module:
    common = {
        "latent_dim": int(model_config["latent_dim"]),
        "encoder_hidden": int(model_config["encoder_hidden"]),
    }
    if name == "m2_markov_mixture_phase":
        return MarkovMixtureModel(
            n_contacts,
            scaffold,
            n_components=int(model_config["mixture_components"]),
            phase_order=int(model_config["phase_order"]),
        )
    if name == "m3_template":
        return LatentTemplateModel(n_contacts, scaffold, **common)
    if name == "m4_field":
        return SharedPropagationFieldRNN(
            n_contacts,
            scaffold,
            jacobian_soft_cap=float(model_config["jacobian_soft_cap"]),
            **common,
        )
    if name == "m4_field_phase":
        return PhaseConditionedPropagationFieldRNN(
            n_contacts,
            scaffold,
            jacobian_soft_cap=float(model_config["jacobian_soft_cap"]),
            phase_order=int(model_config["phase_order"]),
            **common,
        )
    raise ValueError(name)


def _load_context(run_dir: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    config = yaml.safe_load(CONFIG_PATH.read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    checkpoint = torch.load(
        run_dir / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    if checkpoint["config_sha256"] != sha256_file(CONFIG_PATH):
        raise RuntimeError("saved checkpoint does not match the frozen config")
    source_hash = checkpoint["source_provenance"]["source_sha256"]
    for relative, expected in source_hash.items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"saved checkpoint source drift: {relative}")
    dataset_dir = ROOT / config["data"]["dataset_dir"]
    record = load_subject_rank_events(dataset_dir, checkpoint["subject"])
    if record.input_sha256 != checkpoint["input_sha256"]:
        raise RuntimeError("saved checkpoint input fingerprint drift")
    train, monitor, test = record.development_split(
        float(config["data"]["inner_validation_fraction"]),
        float(config["data"]["inner_test_fraction"]),
    )
    ladder = config["ladder"]
    train = _subsample_chronological(train, int(ladder["max_train_events"]))
    monitor = _subsample_chronological(
        monitor, int(ladder["max_validation_events"])
    )
    test = _subsample_chronological(test, int(ladder["max_test_events"]))
    expected_split = checkpoint["split_provenance"]
    checks = {
        "train_indices_sha256": train,
        "monitor_validation_indices_sha256": monitor,
        "development_test_indices_sha256": test,
        "old_heldout20_indices_sha256": record.old_heldout20_indices,
    }
    for key, indices in checks.items():
        if _indices_sha256(indices) != expected_split[key]:
            raise RuntimeError(f"saved split fingerprint drift: {key}")
    if np.intersect1d(test, record.old_heldout20_indices).size:
        raise RuntimeError("old heldout20 entered checkpoint re-scoring")
    groups = torch.as_tensor(record.group_ids[test], dtype=torch.long)
    counts = torch.as_tensor(record.group_count[test], dtype=torch.long)
    model_config = dict(config["model"])
    model_config["mixture_components"] = int(ladder["mixture_components"])
    return {
        "config": config,
        "summary": summary,
        "checkpoint": checkpoint,
        "record": record,
        "train": train,
        "monitor": monitor,
        "test": test,
        "groups": groups,
        "counts": counts,
        "model_config": model_config,
    }


def _load_model(context: Mapping[str, Any], name: str) -> torch.nn.Module:
    checkpoint = context["checkpoint"]
    record = context["record"]
    scaffold = checkpoint["static_scaffold_ml"].detach().cpu().numpy()
    model = _build(
        name, len(record.contact_names), scaffold, context["model_config"]
    )
    model.load_state_dict(checkpoint["models"][name]["model_state"])
    model.eval()
    return model


@torch.no_grad()
def _latent_log_samples(
    model: torch.nn.Module,
    groups: torch.Tensor,
    counts: torch.Tensor,
    *,
    n_samples: int,
    seed: int,
    proposal: str,
    keep_steps: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    generator = torch.Generator(device=groups.device)
    generator.manual_seed(int(seed))
    first = groups == 0
    p_mean, p_log_variance = model.prior_parameters(first)
    if proposal == "importance":
        q_mean, q_log_variance = model.posterior_parameters(groups, counts)
        sample_mean, sample_log_variance = q_mean, q_log_variance
    elif proposal == "prior":
        q_mean = q_log_variance = None
        sample_mean, sample_log_variance = p_mean, p_log_variance
    else:
        raise ValueError(proposal)
    weights: list[torch.Tensor] = []
    steps: list[torch.Tensor] = []
    active: torch.Tensor | None = None
    for _ in range(int(n_samples)):
        initial = model._sample_gaussian(
            sample_mean, sample_log_variance, generator=generator
        )
        likelihood = model.conditional_log_likelihood(initial, groups, counts)
        value = likelihood["event_log_probability"]
        if proposal == "importance":
            value = (
                value
                + diagonal_gaussian_log_prob(initial, p_mean, p_log_variance)
                - diagonal_gaussian_log_prob(
                    initial, q_mean, q_log_variance
                )
            )
        weights.append(value)
        if keep_steps:
            steps.append(likelihood["step_log_probability"])
            active = likelihood["step_active"]
    return (
        torch.stack(weights, dim=0),
        torch.stack(steps, dim=0) if keep_steps else None,
        active,
    )


def _logmeanexp(value: torch.Tensor, n: int) -> torch.Tensor:
    return torch.logsumexp(value[: int(n)], dim=0) - math.log(float(n))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 3:
        return float("nan")
    def average_rank(value: np.ndarray) -> np.ndarray:
        order = np.argsort(value, kind="stable")
        sorted_value = value[order]
        ranked = np.empty(len(value), dtype=float)
        start = 0
        while start < len(value):
            stop = start + 1
            while stop < len(value) and sorted_value[stop] == sorted_value[start]:
                stop += 1
            ranked[order[start:stop]] = 0.5 * (start + stop - 1)
            start = stop
        return ranked

    xv = x[valid]
    yv = y[valid]
    xr = average_rank(xv)
    yr = average_rank(yv)
    if np.std(xr) == 0.0 or np.std(yr) == 0.0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def _checkpoint_worker(
    run_dir_text: str,
    sample_sizes: tuple[int, ...],
) -> dict[str, Any]:
    run_dir = Path(run_dir_text)
    context = _load_context(run_dir)
    groups = context["groups"]
    counts = context["counts"]
    decisions = (counts - 1).clamp_min(0)
    max_samples = int(max(sample_sizes))
    calibration: list[dict[str, Any]] = []
    event_logp: dict[str, np.ndarray] = {}
    step_rows: list[dict[str, Any]] = []
    for model_index, name in enumerate(LATENT_MODELS):
        model = _load_model(context, name)
        for proposal_index, proposal in enumerate(("prior", "importance")):
            weights, step_samples, active = _latent_log_samples(
                model,
                groups,
                counts,
                n_samples=max_samples,
                seed=int(context["summary"]["seed"])
                + 9001
                + model_index * 10007
                + proposal_index * 100003,
                proposal=proposal,
                keep_steps=proposal == "prior",
            )
            for samples in sample_sizes:
                logp = _logmeanexp(weights, samples)
                row = {
                    "subject": context["summary"]["subject"],
                    "seed": int(context["summary"]["seed"]),
                    "model": name,
                    "proposal": proposal,
                    "samples": int(samples),
                    "n_events": int(len(counts)),
                    "nll_per_event": float(-logp.mean()),
                    "nll_per_decision": float(
                        -logp.sum() / decisions.sum().clamp_min(1)
                    ),
                }
                if proposal == "importance" and samples == max_samples:
                    normalized = torch.softmax(weights, dim=0)
                    ess = 1.0 / normalized.square().sum(0)
                    max_weight = normalized.max(0).values
                    row.update(
                        {
                            "ess_fraction_median": float(
                                torch.median(ess / float(samples))
                            ),
                            "ess_fraction_q10": float(
                                torch.quantile(ess / float(samples), 0.10)
                            ),
                            "max_weight_median": float(
                                torch.median(max_weight)
                            ),
                            "max_weight_q90": float(
                                torch.quantile(max_weight, 0.90)
                            ),
                        }
                    )
                else:
                    row.update(
                        {
                            "ess_fraction_median": float("nan"),
                            "ess_fraction_q10": float("nan"),
                            "max_weight_median": float("nan"),
                            "max_weight_q90": float("nan"),
                        }
                    )
                calibration.append(row)
            final = _logmeanexp(weights, max_samples)
            event_logp[f"{name}:{proposal}"] = final.cpu().numpy()
            if proposal == "prior":
                assert step_samples is not None and active is not None
                marginal_step = _logmeanexp(step_samples, max_samples)
                for step in range(marginal_step.shape[1]):
                    selected = active[:, step]
                    if not bool(selected.any()):
                        continue
                    step_rows.append(
                        {
                            "subject": context["summary"]["subject"],
                            "seed": int(context["summary"]["seed"]),
                            "model": name,
                            "step": int(step + 1),
                            "n_active_events": int(selected.sum()),
                            "prior_step_nll": float(
                                -marginal_step[selected, step].mean()
                            ),
                        }
                    )

    mixture = _load_model(context, "m2_markov_mixture_phase")
    with torch.no_grad():
        exact = mixture.conditional_nll(groups, counts)
    event_logp["m2_markov_mixture_phase:exact"] = (
        exact["event_log_probability"].cpu().numpy()
    )
    for step in range(exact["step_active"].shape[1]):
        selected = exact["step_active"][:, step]
        if not bool(selected.any()):
            continue
        step_rows.append(
            {
                "subject": context["summary"]["subject"],
                "seed": int(context["summary"]["seed"]),
                "model": "m2_markov_mixture_phase",
                "step": int(step + 1),
                "n_active_events": int(selected.sum()),
                "prior_step_nll": float(
                    -exact["step_log_probability_diagnostic"][selected, step].mean()
                ),
            }
        )

    count_values = counts.cpu().numpy()
    length_rows: list[dict[str, Any]] = []
    for key, logp in event_logp.items():
        name, estimator = key.split(":")
        per_decision = -logp / np.maximum(count_values - 1, 1)
        for group_count in sorted(np.unique(count_values)):
            selected = count_values == group_count
            length_rows.append(
                {
                    "subject": context["summary"]["subject"],
                    "seed": int(context["summary"]["seed"]),
                    "model": name,
                    "estimator": estimator,
                    "group_count": int(group_count),
                    "suffix_decisions": int(max(group_count - 1, 0)),
                    "n_events": int(np.sum(selected)),
                    "mean_nll_per_event": float(np.mean(-logp[selected])),
                    "mean_nll_per_decision_within_event": float(
                        np.mean(per_decision[selected])
                    ),
                }
            )
    gap_rows: list[dict[str, Any]] = []
    comparisons = (
        ("m4_field:prior", "m3_template:prior", "m4_minus_m3_prior"),
        (
            "m4_field_phase:prior",
            "m3_template:prior",
            "m4phase_minus_m3_prior",
        ),
        (
            "m4_field:importance",
            "m3_template:importance",
            "m4_minus_m3_importance",
        ),
        (
            "m4_field_phase:importance",
            "m3_template:importance",
            "m4phase_minus_m3_importance",
        ),
        (
            "m4_field:prior",
            "m2_markov_mixture_phase:exact",
            "m4_prior_minus_m2phase",
        ),
    )
    for left, right, label in comparisons:
        delta = (
            -event_logp[left] / np.maximum(count_values - 1, 1)
            + event_logp[right] / np.maximum(count_values - 1, 1)
        )
        gap_rows.append(
            {
                "subject": context["summary"]["subject"],
                "seed": int(context["summary"]["seed"]),
                "comparison": label,
                "n_events": int(len(delta)),
                "mean_delta_nll_per_decision": float(np.mean(delta)),
                "median_delta_nll_per_decision": float(np.median(delta)),
                "spearman_delta_vs_group_count": _spearman(
                    count_values, delta
                ),
            }
        )
    return {
        "calibration": calibration,
        "length": length_rows,
        "steps": step_rows,
        "gaps": gap_rows,
        "run": run_dir.name,
    }


def _first_key(row: np.ndarray) -> bytes:
    return np.packbits(np.asarray(row) == 0).tobytes()


def _response_vector(groups: np.ndarray) -> tuple[np.ndarray, float]:
    participants = groups >= 1
    participation = participants.mean(0)
    precedence: list[float] = []
    entropy: list[float] = []
    for left, right in itertools.combinations(range(groups.shape[1]), 2):
        valid = participants[:, left] & participants[:, right]
        if np.sum(valid) < 2:
            precedence.append(float("nan"))
            continue
        delta = groups[valid, left] - groups[valid, right]
        probability = float(
            np.mean((delta < 0).astype(float) + 0.5 * (delta == 0))
        )
        precedence.append(probability)
        clipped = np.clip(probability, 1e-8, 1.0 - 1e-8)
        entropy.append(
            float(
                -clipped * np.log2(clipped)
                - (1.0 - clipped) * np.log2(1.0 - clipped)
            )
        )
    return np.r_[participation, np.asarray(precedence)], (
        float(np.mean(entropy)) if entropy else float("nan")
    )


def _corr(left: np.ndarray, right: np.ndarray) -> tuple[float, int]:
    valid = np.isfinite(left) & np.isfinite(right)
    if np.sum(valid) < 3:
        return float("nan"), int(np.sum(valid))
    if np.std(left[valid]) == 0.0 or np.std(right[valid]) == 0.0:
        return float("nan"), int(np.sum(valid))
    return float(np.corrcoef(left[valid], right[valid])[0, 1]), int(
        np.sum(valid)
    )


def _stability_worker(
    run_dir_text: str,
    rollout_repeats: int,
    minimum_train_support: int,
) -> dict[str, Any]:
    context = _load_context(Path(run_dir_text))
    record = context["record"]
    test = context["test"]
    train = context["train"]
    test_groups = record.group_ids[test]
    test_counts = record.group_count[test]
    train_keys: dict[bytes, int] = {}
    for row in record.group_ids[train]:
        key = _first_key(row)
        train_keys[key] = train_keys.get(key, 0) + 1
    test_by_key: dict[bytes, list[int]] = {}
    for index, row in enumerate(test_groups):
        key = _first_key(row)
        if train_keys.get(key, 0) >= int(minimum_train_support):
            test_by_key.setdefault(key, []).append(index)
    # Avoid unstable one-off response estimates even if training support is high.
    test_by_key = {
        key: value for key, value in test_by_key.items() if len(value) >= 5
    }
    if not test_by_key:
        raise RuntimeError(f"{record.subject}: no supported first-rank stratum")
    groups_tensor = context["groups"]
    counts_tensor = context["counts"]
    rows: list[dict[str, Any]] = []
    vectors: dict[str, dict[str, np.ndarray]] = {}
    for model_index, name in enumerate(RESPONSE_MODELS):
        model = _load_model(context, name)
        generated = []
        for repeat in range(int(rollout_repeats)):
            with torch.no_grad():
                value = model.generate_conditioned(
                    groups_tensor,
                    counts_tensor,
                    seed=int(context["summary"]["seed"])
                    + 12001
                    + model_index * 10007
                    + repeat * 1009,
                )
            generated.append(value.cpu().numpy())
        model_vectors: dict[str, np.ndarray] = {}
        for stratum_index, (key, indices) in enumerate(
            sorted(test_by_key.items(), key=lambda item: item[0])
        ):
            index = np.asarray(indices, dtype=int)
            observed_vector, observed_entropy = _response_vector(
                test_groups[index]
            )
            generated_vector, generated_entropy = _response_vector(
                np.concatenate([value[index] for value in generated], axis=0)
            )
            correlation, n_features = _corr(
                observed_vector, generated_vector
            )
            mae = float(
                np.nanmean(np.abs(observed_vector - generated_vector))
            )
            label = f"stratum_{stratum_index:03d}"
            model_vectors[label] = generated_vector
            rows.append(
                {
                    "subject": record.subject,
                    "seed": int(context["summary"]["seed"]),
                    "model": name,
                    "stratum": label,
                    "train_support": int(train_keys[key]),
                    "test_events": int(len(index)),
                    "response_correlation_to_observed": correlation,
                    "response_mae_to_observed": mae,
                    "n_response_features": n_features,
                    "observed_precedence_entropy": observed_entropy,
                    "generated_precedence_entropy": generated_entropy,
                    "entropy_ratio_generated_to_observed": float(
                        generated_entropy / observed_entropy
                    )
                    if observed_entropy > 0
                    else float("nan"),
                }
            )
        vectors[name] = model_vectors
    return {
        "subject": record.subject,
        "seed": int(context["summary"]["seed"]),
        "rows": rows,
        "vectors": vectors,
        "n_supported_strata": len(test_by_key),
    }


def _parallel(callable_, arguments: Iterable[tuple[Any, ...]], workers: int):
    values = list(arguments)
    if int(workers) <= 1:
        return [callable_(*args) for args in values]
    output = []
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        futures = {pool.submit(callable_, *args): args for args in values}
        for future in as_completed(futures):
            output.append(future.result())
    return output


def run_checkpoint_rounds(
    sample_sizes: tuple[int, ...],
    workers: int,
) -> None:
    outputs = _parallel(
        _checkpoint_worker,
        [(str(path), sample_sizes) for path in _run_dirs()],
        workers,
    )
    calibration = [row for item in outputs for row in item["calibration"]]
    length = [row for item in outputs for row in item["length"]]
    steps = [row for item in outputs for row in item["steps"]]
    gaps = [row for item in outputs for row in item["gaps"]]
    round1 = OUTPUT_ROOT / "round1_likelihood_calibration"
    round2 = OUTPUT_ROOT / "round2_length_progress_decomposition"
    _write_csv(round1 / "likelihood_calibration_runs.csv", calibration)
    _write_csv(round2 / "event_length_strata.csv", length)
    _write_csv(round2 / "suffix_step_diagnostics.csv", steps)
    _write_csv(round2 / "event_length_gap_correlations.csv", gaps)

    max_samples = max(sample_sizes)
    max_rows = [
        row
        for row in calibration
        if row["samples"] == max_samples
    ]
    patient = []
    for key, values in itertools.groupby(
        sorted(
            max_rows,
            key=lambda row: (
                row["subject"],
                row["model"],
                row["proposal"],
            ),
        ),
        key=lambda row: (row["subject"], row["model"], row["proposal"]),
    ):
        rows = list(values)
        patient.append(
            {
                "subject": key[0],
                "model": key[1],
                "proposal": key[2],
                "samples": max_samples,
                "nll_per_decision_mean_across_seeds": float(
                    np.mean([row["nll_per_decision"] for row in rows])
                ),
                "nll_per_decision_sd_across_seeds": float(
                    np.std(
                        [row["nll_per_decision"] for row in rows], ddof=1
                    )
                ),
                "ess_fraction_median_across_seeds": float(
                    np.nanmedian(
                        [row["ess_fraction_median"] for row in rows]
                    )
                )
                if key[2] == "importance"
                else float("nan"),
                "ess_fraction_q10_min_across_seeds": float(
                    np.nanmin([row["ess_fraction_q10"] for row in rows])
                )
                if key[2] == "importance"
                else float("nan"),
            }
        )
    _write_csv(round1 / "likelihood_calibration_patient.csv", patient)
    _write_json(
        round1 / "ROUND_STATE.json",
        {
            "status": "COMPLETE",
            "round": 1,
            "question": (
                "Does the M3/M4 ranking depend on posterior-proposal IWAE "
                "or insufficient Monte-Carlo samples?"
            ),
            "sample_sizes": sample_sizes,
            "n_runs": len(outputs),
            "old_heldout20_scored": False,
            "full_event_posterior_role": (
                "importance_sampling_proposal_only; never used for rollout"
            ),
            "source_sha256": sha256_file(Path(__file__)),
        },
    )
    _write_json(
        round2 / "ROUND_STATE.json",
        {
            "status": "COMPLETE",
            "round": 2,
            "question": (
                "Where along event length and suffix progress do the "
                "template-field gaps arise?"
            ),
            "likelihood_samples": max_samples,
            "n_runs": len(outputs),
            "old_heldout20_scored": False,
            "step_metric_role": (
                "diagnostic marginal step score, not the co-primary joint "
                "complete-event likelihood"
            ),
            "source_sha256": sha256_file(Path(__file__)),
        },
    )


def run_stability_round(
    workers: int,
    rollout_repeats: int,
    minimum_train_support: int,
) -> None:
    outputs = _parallel(
        _stability_worker,
        [
            (str(path), rollout_repeats, minimum_train_support)
            for path in _run_dirs()
        ],
        workers,
    )
    rows = [row for item in outputs for row in item["rows"]]
    pair_rows: list[dict[str, Any]] = []
    by_subject: dict[str, list[dict[str, Any]]] = {}
    for item in outputs:
        by_subject.setdefault(item["subject"], []).append(item)
    for subject, subject_runs in sorted(by_subject.items()):
        for name in RESPONSE_MODELS:
            for left, right in itertools.combinations(subject_runs, 2):
                correlations = []
                feature_counts = []
                common = sorted(
                    set(left["vectors"][name]).intersection(
                        right["vectors"][name]
                    )
                )
                for stratum in common:
                    value, count = _corr(
                        left["vectors"][name][stratum],
                        right["vectors"][name][stratum],
                    )
                    if np.isfinite(value):
                        correlations.append(value)
                        feature_counts.append(count)
                pair_rows.append(
                    {
                        "subject": subject,
                        "model": name,
                        "seed_left": int(left["seed"]),
                        "seed_right": int(right["seed"]),
                        "n_common_strata": len(common),
                        "mean_observable_response_correlation": float(
                            np.mean(correlations)
                        )
                        if correlations
                        else float("nan"),
                        "min_observable_response_correlation": float(
                            np.min(correlations)
                        )
                        if correlations
                        else float("nan"),
                        "median_features_per_stratum": float(
                            np.median(feature_counts)
                        )
                        if feature_counts
                        else float("nan"),
                    }
                )
    target = OUTPUT_ROOT / "round4_observable_seed_stability"
    _write_csv(target / "response_fidelity_by_stratum.csv", rows)
    _write_csv(target / "response_seed_pair_stability.csv", pair_rows)
    _write_json(
        target / "ROUND_STATE.json",
        {
            "status": "COMPLETE",
            "round": 4,
            "question": (
                "Do different optimization seeds recover the same observable "
                "first-rank-conditioned response distributions?"
            ),
            "rollout_repeats": int(rollout_repeats),
            "minimum_train_support_per_first_rank": int(
                minimum_train_support
            ),
            "n_runs": len(outputs),
            "old_heldout20_scored": False,
            "interpretation_limit": (
                "checkpoint reproducibility diagnostic only; the matched "
                "Markov-surrogate null required for formal G2 was not run"
            ),
            "source_sha256": sha256_file(Path(__file__)),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--round",
        choices=("checkpoint", "stability", "all"),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--sample-sizes", type=int, nargs="+", default=(8, 32, 128, 256)
    )
    parser.add_argument("--rollout-repeats", type=int, default=12)
    parser.add_argument("--minimum-train-support", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_sizes = tuple(sorted({int(value) for value in args.sample_sizes}))
    if min(sample_sizes) < 1:
        raise ValueError("sample sizes must be positive")
    if args.round in ("checkpoint", "all"):
        run_checkpoint_rounds(sample_sizes, args.workers)
    if args.round in ("stability", "all"):
        run_stability_round(
            args.workers,
            args.rollout_repeats,
            args.minimum_train_support,
        )


if __name__ == "__main__":
    main()
