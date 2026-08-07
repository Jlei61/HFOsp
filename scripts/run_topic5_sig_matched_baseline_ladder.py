#!/usr/bin/env python3
"""Fit phase-matched M1/M2/M3 controls for the six-patient SIG pilot."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
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

from scripts.run_topic5_spf_model_ladder import (  # noqa: E402
    _batch,
    _model_seed,
    _score_repeated,
    _seed_everything,
    _train_one,
)
from src.topic5_rank_distribution import distribution_errors  # noqa: E402
from src.topic5_shared_propagation_field import (  # noqa: E402
    LatentTemplateModel,
    load_subject_rank_events,
    sha256_file,
)
from src.topic5_stable_interaction_graph import (  # noqa: E402
    MatchedPhaseMarkovMixtureModel,
    uniform_provenance,
)


SIG_CONFIG = ROOT / "config/topic5_stable_interaction_graph_v2.yaml"
SPF_CONFIG = ROOT / "config/topic5_shared_propagation_field_v0_1.yaml"
SIG_ROOT = (
    ROOT
    / "results/topic5_stable_interaction_graph/development"
    / "human_graph_increment_pilot_v0_2_training_adequacy"
)
OUTPUT = (
    ROOT
    / "results/topic5_stable_interaction_graph/development"
    / "human_matched_baseline_ladder_v0_2_training_adequacy"
)
MODELS = (
    "m1_markov_matched_phase",
    "m2_mixture_matched_phase",
    "m3_latent_template",
)


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _source_sha256() -> dict[str, str]:
    """Provenance of every file that decides what the baseline fits are."""
    return {
        "runner": sha256_file(Path(__file__)),
        "model": sha256_file(ROOT / "src/topic5_stable_interaction_graph.py"),
        "shared_propagation_field": sha256_file(
            ROOT / "src/topic5_shared_propagation_field.py"
        ),
        "training_loop": sha256_file(
            ROOT / "scripts/run_topic5_spf_model_ladder.py"
        ),
    }


def _config_sha256() -> dict[str, str]:
    return {
        "sig": sha256_file(SIG_CONFIG),
        "spf_training": sha256_file(SPF_CONFIG),
    }


def _subsample(indices: np.ndarray, limit: int) -> np.ndarray:
    values = np.asarray(indices, dtype=int)
    if len(values) <= int(limit):
        return values
    return values[
        np.linspace(0, len(values) - 1, int(limit)).astype(int)
    ]


def _load_scaffold(subject: str, seed: int, sig_root: Path) -> np.ndarray:
    path = sig_root / "per_run" / subject / f"seed_{seed}/checkpoint.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return np.asarray(checkpoint["scaffold"], dtype=np.float32)


def _build(name: str, contacts: int, scaffold: np.ndarray):
    if name == "m1_markov_matched_phase":
        return MatchedPhaseMarkovMixtureModel(
            contacts, scaffold, n_components=1
        )
    if name == "m2_mixture_matched_phase":
        return MatchedPhaseMarkovMixtureModel(
            contacts, scaffold, n_components=3
        )
    if name == "m3_latent_template":
        return LatentTemplateModel(
            contacts,
            scaffold,
            latent_dim=4,
            encoder_hidden=32,
            template_hidden=32,
        )
    raise ValueError(name)


def _fit_with_rescue(
    name,
    model,
    record,
    train_index,
    validation_index,
    *,
    training,
    evaluation,
    seed,
):
    initial_state = copy.deepcopy(model.state_dict())
    fitted = _train_one(
        name,
        model,
        record.group_ids,
        record.group_count,
        train_index,
        validation_index,
        device=torch.device("cpu"),
        training=training,
        evaluation=evaluation,
        seed=seed,
    )
    attempts = [fitted["adequacy"]["verdict"]]
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
            record.group_ids,
            record.group_count,
            train_index,
            validation_index,
            device=torch.device("cpu"),
            training=rescue,
            evaluation=evaluation,
            seed=seed,
        )
        attempts.append(fitted["adequacy"]["verdict"])
    if not fitted["adequacy"]["converged"]:
        raise RuntimeError(
            f"{name}: inadequate training after attempts {attempts}"
        )
    model.load_state_dict(fitted["best_state"])
    model.eval()
    return fitted


def _run_subject(
    subject: str, *, sig_root: str = str(SIG_ROOT), output_dir: str = str(OUTPUT)
) -> dict[str, Any]:
    torch.set_num_threads(2)
    sig_root_path = Path(sig_root)
    output = Path(output_dir)
    sig_config = yaml.safe_load(SIG_CONFIG.read_text())
    spf_config = yaml.safe_load(SPF_CONFIG.read_text())
    data = sig_config["data"]
    training = spf_config["training"]
    evaluation = spf_config["evaluation"]
    record = load_subject_rank_events(ROOT / data["dataset_dir"], subject)
    train, validation, test = record.development_split(
        float(data["validation_fraction"]),
        float(data["test_fraction"]),
    )
    train = _subsample(train, int(data["max_train_events"]))
    validation = _subsample(
        validation, int(data["max_validation_events"])
    )
    test = _subsample(test, int(data["max_test_events"]))
    if np.intersect1d(np.r_[train, validation, test], record.old_heldout20_indices).size:
        raise RuntimeError(f"{subject}: outer heldout20 leakage")
    seeds = list(map(int, sig_config["training"]["fit_seeds"]))
    rows = []
    for seed in seeds:
        scaffold = _load_scaffold(subject, seed, sig_root_path)
        fitted_models = {}
        for name in MODELS:
            model_seed = _model_seed(seed, f"sigv2_{name}")
            _seed_everything(model_seed)
            model = _build(name, len(record.contact_names), scaffold)
            fitted = _fit_with_rescue(
                name,
                model,
                record,
                train,
                validation,
                training=training,
                evaluation=evaluation,
                seed=model_seed,
            )
            fitted_models[name] = (model, fitted)

        test_groups, test_counts = _batch(
            record.group_ids,
            record.group_count,
            test,
            torch.device("cpu"),
        )
        run = {
            "contract": "topic5_sig_matched_baseline_ladder_v0_2",
            "subject": subject,
            "dataset": record.dataset,
            "fit_seed": seed,
            "n_train_events": len(train),
            "n_validation_events": len(validation),
            "n_test_events": len(test),
            "models": {},
            "input_sha256": record.input_sha256,
            "split_sha256": hashlib.sha256(
                np.asarray(np.r_[train, validation, test], dtype=np.int64).tobytes()
            ).hexdigest(),
            "config_sha256": _config_sha256(),
            "source_sha256": _source_sha256(),
            "scaffold_source": str(
                sig_root_path / "per_run" / subject / f"seed_{seed}"
                / "checkpoint.pt"
            ),
            "old_heldout20_scored": False,
            "forbidden_inputs_read": False,
            "snn_inputs_read": False,
        }
        checkpoints = {}
        for ordinal, (name, (model, fitted)) in enumerate(fitted_models.items()):
            formal = _score_repeated(
                model,
                test_groups,
                test_counts,
                prior_samples=int(evaluation["prior_predictive_samples"]),
                importance_samples=int(evaluation["importance_samples"]),
                repeats=int(evaluation["prior_predictive_repeats"]),
                seed=seed + 211,
            )
            rollout = []
            first_generated = None
            for repeat in range(int(evaluation["rollout_repeats"])):
                generated = model.generate_conditioned(
                    test_groups,
                    test_counts,
                    seed=seed + 307 + repeat * 1013 + ordinal * 100_000,
                )
                generated_np = generated.cpu().numpy().astype(np.int16)
                if first_generated is None:
                    first_generated = generated_np
                rollout.append(
                    distribution_errors(
                        generated_np,
                        test_counts.cpu().numpy(),
                        test_groups.cpu().numpy(),
                        test_counts.cpu().numpy(),
                    )
                )
            run["models"][name] = {
                "nll_per_event": formal["nll_per_event"],
                "nll_per_decision": formal["nll_per_decision"],
                "nll_mc_sd": formal["nll_per_decision_mc_sd"],
                "likelihood_estimator": formal["estimator"],
                "training_adequacy": fitted["adequacy"],
                "repertoire": {
                    key: float(np.mean([value[key] for value in rollout]))
                    for key in rollout[0]
                },
                "repertoire_sd": {
                    key: float(np.std([value[key] for value in rollout], ddof=1))
                    for key in rollout[0]
                },
            }
            checkpoints[name] = {
                "state_dict": fitted["best_state"],
                "optimizer_state": fitted["best_optimizer_state"],
                "history": fitted["history"],
                "first_generated": first_generated,
            }
        run_dir = output / "per_run" / subject / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "contract": run["contract"],
                "subject": subject,
                "fit_seed": seed,
                "models": {
                    name: {
                        key: value
                        for key, value in checkpoint.items()
                        if key != "first_generated"
                    }
                    for name, checkpoint in checkpoints.items()
                },
            },
            run_dir / "checkpoint.pt",
        )
        np.savez_compressed(
            run_dir / "conditioned_generation.npz",
            observed_group_ids=test_groups.cpu().numpy(),
            observed_group_count=test_counts.cpu().numpy(),
            **{
                name: checkpoint["first_generated"]
                for name, checkpoint in checkpoints.items()
            },
        )
        _write(run_dir / "summary.json", run)
        rows.append(run)
    state = {
        "status": "COMPLETE",
        "subject": subject,
        "n_seeds": len(rows),
        "models": list(MODELS),
    }
    _write(output / "per_run" / subject / "SUBJECT_STATE.json", state)
    return state


def _aggregate(
    *, sig_root: Path = SIG_ROOT, output: Path = OUTPUT
) -> dict[str, Any]:
    config = yaml.safe_load(SIG_CONFIG.read_text())
    subjects = list(map(str, config["pilot"]["subjects"]))
    seeds = list(map(int, config["training"]["fit_seeds"]))
    sig = json.loads((sig_root / "HUMAN_GRAPH_INCREMENT_PILOT.json").read_text())
    run_rows = []
    baselines = []
    for subject in subjects:
        for seed in seeds:
            baseline = json.loads(
                (
                    output
                    / "per_run"
                    / subject
                    / f"seed_{seed}/summary.json"
                ).read_text()
            )
            baselines.append(baseline)
            sig_run = next(
                row
                for row in sig["run_rows"]
                if row["subject"] == subject and row["fit_seed"] == seed
            )
            row = {
                "subject": subject,
                "fit_seed": seed,
                "sig1_nll": sig_run["sig1_nll"],
                "sig1_precedence_mae": sig_run["sig1_precedence_mae"],
            }
            for name in MODELS:
                value = baseline["models"][name]
                if not value["training_adequacy"]["converged"]:
                    raise RuntimeError(
                        f"inadequate matched-baseline training: "
                        f"{subject} seed={seed} model={name}"
                    )
                row[f"{name}_nll"] = value["nll_per_decision"]
                row[f"{name}_precedence_mae"] = value["repertoire"][
                    "precedence_mae"
                ]
            estimator_contract = {
                name: baseline["models"][name]["likelihood_estimator"]
                for name in MODELS
            }
            if estimator_contract != {
                "m1_markov_matched_phase": "exact",
                "m2_mixture_matched_phase": "exact",
                "m3_latent_template": (
                    "importance_weighted_posterior_proposal"
                ),
            }:
                raise RuntimeError(
                    f"likelihood estimator drift: {subject} seed={seed} "
                    f"{estimator_contract}"
                )
            checkpoint = torch.load(
                output
                / "per_run"
                / subject
                / f"seed_{seed}/checkpoint.pt",
                map_location="cpu",
                weights_only=False,
            )
            mixture_state = checkpoint["models"][
                "m2_mixture_matched_phase"
            ]["state_dict"]
            transition = mixture_state["transition"].reshape(3, -1)
            bias = mixture_state["bias_offset"].reshape(3, -1)
            component = torch.cat((transition, bias), dim=1)
            pairwise = [
                float(torch.linalg.vector_norm(component[left] - component[right]))
                for left in range(3)
                for right in range(left + 1, 3)
            ]
            row["m2_min_component_parameter_distance"] = min(pairwise)
            if row["m2_min_component_parameter_distance"] <= 1e-6:
                raise RuntimeError(
                    f"collapsed identical M2 components: {subject} seed={seed}"
                )
            run_rows.append(row)
    # Fail closed before any count: a mixed or missing source/config makes the
    # SIG-versus-baseline gap unattributable to one implementation.
    fit_provenance = uniform_provenance(
        baselines,
        ("config_sha256", "source_sha256"),
        current_source_sha256=_source_sha256(),
    )
    input_sha256 = {}
    for subject in subjects:
        rows = [row for row in baselines if row["subject"] == subject]
        input_sha256[subject] = uniform_provenance(rows, ("input_sha256",))[
            "input_sha256"
        ]
    # SIG1 is scored on the same events by the same dataset revision, so the
    # screen it is quoted from must carry the same per-subject input hash.
    for subject, value in input_sha256.items():
        screen_value = sig["input_sha256"][subject]
        if screen_value != value:
            raise RuntimeError(
                f"{subject}: baseline input {value} does not match the "
                f"SIG screen input {screen_value}"
            )
    patients = []
    for subject in subjects:
        values = [row for row in run_rows if row["subject"] == subject]
        patient = {"subject": subject}
        for key in values[0]:
            if key in ("subject", "fit_seed"):
                continue
            patient[key] = float(np.median([row[key] for row in values]))
        baseline_names = list(MODELS)
        patient["best_baseline_nll"] = min(
            patient[f"{name}_nll"] for name in baseline_names
        )
        patient["best_baseline_precedence_mae"] = min(
            patient[f"{name}_precedence_mae"] for name in baseline_names
        )
        patient["sig1_nll_gain_vs_best"] = (
            patient["best_baseline_nll"] - patient["sig1_nll"]
        )
        patient["sig1_precedence_gain_vs_best"] = (
            patient["best_baseline_precedence_mae"]
            - patient["sig1_precedence_mae"]
        )
        patient["sig1_both_better_than_best"] = bool(
            patient["sig1_nll_gain_vs_best"] > 0
            and patient["sig1_precedence_gain_vs_best"] > 0
        )
        patients.append(patient)
    counts = {
        "sig1_nll_better_than_all_baselines": int(
            sum(row["sig1_nll_gain_vs_best"] > 0 for row in patients)
        ),
        "sig1_precedence_better_than_all_baselines": int(
            sum(row["sig1_precedence_gain_vs_best"] > 0 for row in patients)
        ),
        "sig1_both_better_than_all_baselines": int(
            sum(row["sig1_both_better_than_best"] for row in patients)
        ),
    }
    threshold = int(
        config["evaluation"]["continue_to_full_baseline_ladder_requires"][
            "n_patients_both_better"
        ]
    )
    passed = counts["sig1_both_better_than_all_baselines"] >= threshold
    max_possible_both_given_nll = counts[
        "sig1_nll_better_than_all_baselines"
    ]
    payload = {
        "contract": "topic5_sig_matched_baseline_ladder_v0_2",
        "runner_revision": "v0_3_provenance_contract",
        "status": "COMPLETE",
        "decision": (
            "OPEN_G1_REPERTOIRE_AND_STABILITY_DIAGNOSTICS"
            if passed
            else "STOP_BEFORE_STRUCTURE_CLAIM"
        ),
        "g1_status": (
            "NOT_YET_PASSED" if passed else "NOT_PASSED_DEVELOPMENT"
        ),
        "n_subjects": len(subjects),
        "n_fit_seeds": len(seeds),
        "n_model_fits": len(run_rows) * len(MODELS),
        "all_training_adequate": True,
        "likelihood_estimator_contract_valid": True,
        "m2_components_separated": True,
        "m2_min_component_parameter_distance": float(
            min(
                row["m2_min_component_parameter_distance"]
                for row in run_rows
            )
        ),
        "estimator_boundary": (
            "M1/M2 use exact likelihood; M3 uses a finite-sample IWAE "
            "marginal-likelihood estimate and future-blind prior rollout. "
            "M3 is therefore not given an optimistic exact-score advantage."
        ),
        "models": ["sig1_feedback_graph", *MODELS],
        "counts": counts,
        "continue_threshold_both": threshold,
        "decision_rule_executed": (
            "Per patient, seed-median SIG1 must be strictly lower than the "
            "minimum over M1-phase/M2-phase/M3 on both held-out NLL per "
            "decision and free-rollout precedence MAE; at least "
            f"{threshold} of {len(subjects)} patients must satisfy both."
        ),
        "decision_rule_provenance": (
            "The count threshold is the SIG0-vs-SIG1 screen continue rule from "
            "spec section 6.1, reused here as the ladder stop rule. Spec "
            "section 8 states G1 as non-inferior NLL plus rollout distance "
            "better than M2-phase and M3, with no count. The executed rule is "
            "strictly harder on every axis, so it can stop the line but cannot "
            "pass it."
        ),
        "g1_clauses_not_evaluated": [
            "no mode collapse or obvious over-dispersion (spec section 8): no "
            "within-start dispersion diagnostic was computed in this contract"
        ],
        "max_possible_both_given_nll": max_possible_both_given_nll,
        "stop_robust_to_rollout_resampling": bool(
            max_possible_both_given_nll < threshold
        ),
        "patient_rows": patients,
        "run_rows": run_rows,
        "old_heldout20_scored": False,
        "forbidden_inputs_read": False,
        "snn_inputs_read": False,
        "claim": (
            "The graph must beat the strongest phase-matched Markov mixture "
            "or latent template on both likelihood and free-rollout precedence "
            "before structure stability can be opened."
        ),
        "fit_time_source_sha256": fit_provenance["source_sha256"],
        "fit_time_config_sha256": fit_provenance["config_sha256"],
        "input_sha256": input_sha256,
        "sig_screen_root": str(sig_root),
        "aggregation_source_sha256": sha256_file(Path(__file__)),
        "aggregation_config_sha256": _config_sha256(),
        "provenance_contract": (
            "Every run artifact carries the same config and source hashes, and "
            "those hashes equal the aggregating source. A mixed or missing "
            "hash raises before any count is computed."
        ),
    }
    _write(output / "MATCHED_BASELINE_LADDER.json", payload)
    _write(
        output / "LADDER_STATE.json",
        {
            "status": "COMPLETE",
            "decision": payload["decision"],
            "g1_status": payload["g1_status"],
            "missing_runs": 0,
        },
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--sig-root", type=Path, default=SIG_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(SIG_CONFIG.read_text())
    subjects = list(map(str, config["pilot"]["subjects"]))
    failures = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        futures = {
            pool.submit(
                _run_subject,
                subject,
                sig_root=str(args.sig_root),
                output_dir=str(args.output_dir),
            ): subject
            for subject in subjects
        }
        for future in as_completed(futures):
            subject = futures[future]
            try:
                print(json.dumps(future.result()))
            except Exception as exc:
                failures.append({"subject": subject, "error": repr(exc)})
    if failures:
        _write(
            args.output_dir / "LADDER_STATE.json",
            {"status": "FAIL_CLOSED", "failures": failures},
        )
        raise RuntimeError(f"matched baseline ladder failed: {failures}")
    print(
        json.dumps(
            _aggregate(sig_root=args.sig_root, output=args.output_dir), indent=2
        )
    )


if __name__ == "__main__":
    main()
