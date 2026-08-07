#!/usr/bin/env python3
"""Run D4 unseen-start compositional generalization with current model families."""
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

from scripts.run_topic5_sig_matched_baseline_ladder import (  # noqa: E402
    _build as _build_baseline,
    _fit_with_rescue,
)
from scripts.run_topic5_spf_model_ladder import (  # noqa: E402
    _batch,
    _model_seed,
    _score_repeated,
    _seed_everything,
)
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
)


CONFIG = ROOT / "config/topic5_stable_interaction_identifiability_v2_1.yaml"
SPF_CONFIG = ROOT / "config/topic5_shared_propagation_field_v0_1.yaml"
OUTPUT = (
    ROOT
    / "results/topic5_stable_interaction_graph/development"
    / "v2_1_unseen_start"
)
BASELINES = (
    "m1_markov_matched_phase",
    "m2_mixture_matched_phase",
    "m3_latent_template",
)
SIG_MODELS = {
    "sig0_phase_envelope": False,
    "sig1_feedback_graph": True,
}


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
        "baseline_runner": sha256_file(
            ROOT / "scripts/run_topic5_sig_matched_baseline_ladder.py"
        ),
        "training_loop": sha256_file(
            ROOT / "scripts/run_topic5_spf_model_ladder.py"
        ),
    }


def _subsample(indices: np.ndarray, limit: int) -> np.ndarray:
    values = np.asarray(indices, dtype=int)
    if len(values) <= int(limit):
        return values
    return values[np.linspace(0, len(values) - 1, int(limit)).astype(int)]


def _first_key(row: np.ndarray) -> str:
    packed = np.packbits(np.asarray(row == 0, dtype=np.uint8))
    return hashlib.sha1(packed.tobytes()).hexdigest()[:16]


def _select_holdout(record, train: np.ndarray, validation: np.ndarray, test: np.ndarray, rules: dict[str, Any]) -> dict[str, Any]:
    keys = np.asarray([_first_key(row) for row in record.group_ids])
    candidates = []
    for key in sorted(set(keys[train])):
        train_holdout = train[keys[train] == key]
        validation_holdout = validation[keys[validation] == key]
        test_holdout = test[keys[test] == key]
        start_contacts = np.flatnonzero(record.group_ids[train_holdout[0]] == 0)
        remaining = train[keys[train] != key]
        intermediate = np.sum(record.group_ids[remaining][:, start_contacts] > 0, axis=0)
        # Selection is train-only. Validation/test counts are recorded for the
        # selected key but never decide which key wins; if they are inadequate
        # the subject fails closed instead of switching to a test-favoured key.
        qualifies = (
            len(train_holdout) >= int(rules["minimum_train_events"])
            and int(np.min(intermediate))
            >= int(rules["minimum_intermediate_sender_support_after_holdout"])
        )
        if qualifies:
            candidates.append(
                {
                    "key": key,
                    "n_train": len(train_holdout),
                    "n_validation": len(validation_holdout),
                    "n_test": len(test_holdout),
                    "start_contacts": start_contacts,
                    "intermediate_support_min": int(np.min(intermediate)),
                }
            )
    if not candidates:
        raise RuntimeError(f"{record.subject}: no prequalified unseen-start group")
    candidates.sort(key=lambda row: (-row["n_train"], row["key"]))
    selected = candidates[0]
    return {**selected, "all_keys": keys, "n_qualifying_candidates": len(candidates)}


def _dataset(record, indices: np.ndarray) -> SyntheticGraphDataset:
    groups = record.group_ids[indices].copy()
    counts = record.group_count[indices].copy()
    starts = np.argmax(groups == 0, axis=1).astype(np.int16)
    return SyntheticGraphDataset(groups, counts, starts)


def _sig_score(model, dataset: SyntheticGraphDataset) -> float:
    groups, counts = dataset.torch()
    with torch.no_grad():
        return float(model.nll_per_decision(groups, counts))


def _sig_repertoire(model, dataset: SyntheticGraphDataset, *, seed: int, repeats: int) -> tuple[dict[str, Any], np.ndarray]:
    groups, counts = dataset.torch()
    schedule = torch.as_tensor(
        cardinality_schedule(dataset.group_ids, dataset.group_count), dtype=torch.long
    )
    rows = []
    first_generated = None
    for repeat in range(int(repeats)):
        generated = model.rollout(
            groups == 0,
            counts,
            schedule,
            generator=torch.Generator().manual_seed(seed + repeat * 1009),
        ).numpy().astype(np.int16)
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
    assert first_generated is not None
    return (
        {
            "mean": {key: float(np.mean([row[key] for row in rows])) for key in rows[0]},
            "sd": {key: float(np.std([row[key] for row in rows], ddof=1)) for key in rows[0]},
        },
        first_generated,
    )


def _baseline_score(model, dataset: SyntheticGraphDataset, *, evaluation: dict[str, Any], seed: int) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    groups, counts = dataset.torch()
    formal = _score_repeated(
        model,
        groups,
        counts,
        prior_samples=int(evaluation["prior_predictive_samples"]),
        importance_samples=int(evaluation["importance_samples"]),
        repeats=int(evaluation["prior_predictive_repeats"]),
        seed=seed,
    )
    rows = []
    first_generated = None
    for repeat in range(int(evaluation["rollout_repeats"])):
        generated = model.generate_conditioned(
            groups, counts, seed=seed + 500 + repeat * 1009
        ).numpy().astype(np.int16)
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
    assert first_generated is not None
    repertoire = {
        "mean": {key: float(np.mean([row[key] for row in rows])) for key in rows[0]},
        "sd": {key: float(np.std([row[key] for row in rows], ddof=1)) for key in rows[0]},
    }
    return formal, repertoire, first_generated


def _run_subject(subject: str, *, output_dir: str) -> dict[str, Any]:
    torch.set_num_threads(2)
    config = yaml.safe_load(CONFIG.read_text())
    spf = yaml.safe_load(SPF_CONFIG.read_text())
    data = config["data"]
    rules = config["unseen_start"]
    training = config["training"]
    evaluation = config["evaluation"]
    record = load_subject_rank_events(ROOT / data["dataset_dir"], subject)
    train, validation, test = record.development_split(
        float(data["validation_fraction"]), float(data["test_fraction"])
    )
    selected = _select_holdout(record, train, validation, test, rules)
    keys = selected.pop("all_keys")
    heldout_key = selected["key"]
    if (
        selected["n_validation"] < int(rules["minimum_validation_events"])
        or selected["n_test"] < int(rules["minimum_test_events"])
    ):
        raise RuntimeError(
            f"{subject}: train-selected start lacks untouched evaluation support"
        )
    train = train[keys[train] != heldout_key]
    validation = validation[keys[validation] != heldout_key]
    unseen_test = test[keys[test] == heldout_key]
    seen_test = test[keys[test] != heldout_key]
    train = _subsample(train, int(data["max_train_events"]))
    validation = _subsample(validation, int(data["max_validation_events"]))
    unseen_test = _subsample(unseen_test, int(data["max_test_events"]))
    seen_test = _subsample(seen_test, int(data["max_test_events"]))
    all_used = np.r_[train, validation, unseen_test, seen_test]
    if np.intersect1d(all_used, record.old_heldout20_indices).size:
        raise RuntimeError(f"{subject}: old heldout20 leakage")
    if not len(unseen_test):
        raise RuntimeError(f"{subject}: no untouched unseen-start test events")

    scaffold = fit_static_scaffold_ml(
        record.group_ids,
        record.group_count,
        train,
        steps=int(training["static_scaffold_steps"]),
        learning_rate=float(training["static_scaffold_learning_rate"]),
        seed=int(training["static_scaffold_seed"]),
        device="cpu",
    )
    train_dataset = _dataset(record, train)
    validation_dataset = _dataset(record, validation)
    unseen_dataset = _dataset(record, unseen_test)
    seen_dataset = _dataset(record, seen_test)
    output = Path(output_dir)
    rows = []
    for fit_seed in map(int, training["fit_seeds"]):
        fitted: dict[str, Any] = {}
        for name, learn_graph in SIG_MODELS.items():
            model_seed = _model_seed(fit_seed, f"v2_1_unseen_{name}")
            fitted[name] = fit_synthetic_sig(
                train_dataset,
                validation_dataset,
                seed=model_seed,
                learn_graph=learn_graph,
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
        for name in BASELINES:
            model_seed = _model_seed(fit_seed, f"v2_1_unseen_{name}")
            _seed_everything(model_seed)
            model = _build_baseline(name, len(record.contact_names), scaffold)
            baseline_fit = _fit_with_rescue(
                name,
                model,
                record,
                train,
                validation,
                training=spf["training"],
                evaluation=spf["evaluation"],
                seed=model_seed,
            )
            fitted[name] = {"model": model, **baseline_fit}

        run = {
            "contract": config["contract"]["name"],
            "subject": subject,
            "dataset": record.dataset,
            "fit_seed": fit_seed,
            "heldout_start_key": heldout_key,
            "heldout_start_contacts": record.contact_names[selected["start_contacts"]].tolist(),
            "holdout_selection": selected,
            "n_train_events": len(train),
            "n_validation_events": len(validation),
            "n_unseen_test_events": len(unseen_test),
            "n_seen_test_events": len(seen_test),
            "models": {},
            "input_sha256": record.input_sha256,
            "config_sha256": sha256_file(CONFIG),
            "source_sha256": _source_sha256(),
            "split_sha256": hashlib.sha256(
                np.asarray(np.r_[train, validation, unseen_test, seen_test], dtype=np.int64).tobytes()
            ).hexdigest(),
            "old_heldout20_scored": False,
            "forbidden_inputs_read": False,
            "snn_inputs_read": False,
        }
        checkpoints = {}
        generated_arrays = {}
        for ordinal, name in enumerate([*SIG_MODELS, *BASELINES]):
            value = fitted[name]
            if name in SIG_MODELS:
                model = value.model
                if not value.adequacy["converged"]:
                    raise RuntimeError(f"{subject} {fit_seed} {name}: inadequate fit")
                unseen_nll = _sig_score(model, unseen_dataset)
                seen_nll = _sig_score(model, seen_dataset)
                unseen_repertoire, generated = _sig_repertoire(
                    model,
                    unseen_dataset,
                    seed=fit_seed + ordinal * 100_000,
                    repeats=int(evaluation["rollout_repeats"]),
                )
                seen_repertoire, _ = _sig_repertoire(
                    model,
                    seen_dataset,
                    seed=fit_seed + ordinal * 100_000 + 17,
                    repeats=int(evaluation["rollout_repeats"]),
                )
                model_payload = {
                    "best_validation_nll": value.best_validation_nll,
                    "training_adequacy": value.adequacy,
                    "unseen_nll_per_decision": unseen_nll,
                    "seen_nll_per_decision": seen_nll,
                    "unseen_repertoire": unseen_repertoire,
                    "seen_repertoire": seen_repertoire,
                }
                checkpoints[name] = {
                    "state_dict": value.model.state_dict(),
                    "optimizer_state": value.best_optimizer_state,
                    "history": value.history,
                }
            else:
                model = value["model"]
                if not value["adequacy"]["converged"]:
                    raise RuntimeError(f"{subject} {fit_seed} {name}: inadequate fit")
                unseen_formal, unseen_repertoire, generated = _baseline_score(
                    model,
                    unseen_dataset,
                    evaluation=evaluation,
                    seed=fit_seed + ordinal * 100_000,
                )
                seen_formal, seen_repertoire, _ = _baseline_score(
                    model,
                    seen_dataset,
                    evaluation=evaluation,
                    seed=fit_seed + ordinal * 100_000 + 17,
                )
                model_payload = {
                    "best_validation_nll": value["adequacy"]["best_validation_nll"],
                    "training_adequacy": value["adequacy"],
                    "unseen_nll_per_decision": unseen_formal["nll_per_decision"],
                    "unseen_nll_mc_sd": unseen_formal["nll_per_decision_mc_sd"],
                    "seen_nll_per_decision": seen_formal["nll_per_decision"],
                    "seen_nll_mc_sd": seen_formal["nll_per_decision_mc_sd"],
                    "unseen_repertoire": unseen_repertoire,
                    "seen_repertoire": seen_repertoire,
                }
                checkpoints[name] = {
                    "state_dict": value["best_state"],
                    "optimizer_state": value["best_optimizer_state"],
                    "history": value["history"],
                }
            run["models"][name] = model_payload
            generated_arrays[name] = generated

        run_dir = output / "per_run" / subject / f"seed_{fit_seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "contract": run["contract"],
                "subject": subject,
                "fit_seed": fit_seed,
                "scaffold": scaffold,
                "models": checkpoints,
            },
            run_dir / "checkpoint.pt",
        )
        np.savez_compressed(
            run_dir / "unseen_start_generation.npz",
            observed_group_ids=unseen_dataset.group_ids,
            observed_group_count=unseen_dataset.group_count,
            **generated_arrays,
        )
        _write(run_dir / "summary.json", run)
        rows.append(run)
    _write(
        output / "per_run" / subject / "SUBJECT_STATE.json",
        {"status": "COMPLETE", "subject": subject, "n_seeds": len(rows)},
    )
    return {"status": "COMPLETE", "subject": subject, "n_seeds": len(rows)}


def _aggregate(output: Path) -> dict[str, Any]:
    config = yaml.safe_load(CONFIG.read_text())
    subjects = list(map(str, config["pilot"]["subjects"]))
    seeds = list(map(int, config["training"]["fit_seeds"]))
    patients = []
    run_rows = []
    for subject in subjects:
        runs = []
        for seed in seeds:
            path = output / "per_run" / subject / f"seed_{seed}/summary.json"
            if not path.is_file():
                raise RuntimeError(f"missing unseen-start run: {path}")
            run = json.loads(path.read_text())
            runs.append(run)
            run_rows.append(run)
        patient = {
            "subject": subject,
            "heldout_start_key": runs[0]["heldout_start_key"],
            "heldout_start_contacts": runs[0]["heldout_start_contacts"],
            "n_unseen_test_events": runs[0]["n_unseen_test_events"],
            "models": {},
        }
        for name in [*SIG_MODELS, *BASELINES]:
            patient["models"][name] = {
                metric: float(np.median([
                    run["models"][name][metric] for run in runs
                ]))
                for metric in (
                    "best_validation_nll",
                    "unseen_nll_per_decision",
                    "seen_nll_per_decision",
                )
            }
            for split in ("unseen", "seen"):
                patient["models"][name][f"{split}_precedence_mae"] = float(
                    np.median([
                        run["models"][name][f"{split}_repertoire"]["mean"]["precedence_mae"]
                        for run in runs
                    ])
                )
        selected = min(
            BASELINES,
            key=lambda name: patient["models"][name]["best_validation_nll"],
        )
        patient["validation_selected_baseline"] = selected
        patient["sig1_gains"] = {
            "unseen_nll": (
                patient["models"][selected]["unseen_nll_per_decision"]
                - patient["models"]["sig1_feedback_graph"]["unseen_nll_per_decision"]
            ),
            "unseen_precedence": (
                patient["models"][selected]["unseen_precedence_mae"]
                - patient["models"]["sig1_feedback_graph"]["unseen_precedence_mae"]
            ),
            "seen_nll": (
                patient["models"][selected]["seen_nll_per_decision"]
                - patient["models"]["sig1_feedback_graph"]["seen_nll_per_decision"]
            ),
            "seen_precedence": (
                patient["models"][selected]["seen_precedence_mae"]
                - patient["models"]["sig1_feedback_graph"]["seen_precedence_mae"]
            ),
        }
        patients.append(patient)
    counts = {
        "sig1_unseen_nll_better": sum(row["sig1_gains"]["unseen_nll"] > 0 for row in patients),
        "sig1_unseen_precedence_better": sum(row["sig1_gains"]["unseen_precedence"] > 0 for row in patients),
        "sig1_unseen_both_better": sum(
            row["sig1_gains"]["unseen_nll"] > 0
            and row["sig1_gains"]["unseen_precedence"] > 0
            for row in patients
        ),
    }
    payload = {
        "contract": config["contract"]["name"],
        "status": "COMPLETE_UNSEEN_START_DEVELOPMENT",
        "selection_rule": config["unseen_start"]["selection_rule"],
        "baseline_selection": (
            "one family per patient by seed-median heldout-start-free inner "
            "validation NLL; same family used for both untouched OOD endpoints"
        ),
        "counts": counts,
        "patients": patients,
        "n_model_fits": len(run_rows) * (len(SIG_MODELS) + len(BASELINES)),
        "all_training_adequate": all(
            run["models"][name]["training_adequacy"]["converged"]
            for run in run_rows for name in [*SIG_MODELS, *BASELINES]
        ),
        "old_heldout20_scored": False,
        "forbidden_inputs_read": False,
        "snn_inputs_read": False,
        "source_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(CONFIG),
    }
    _write(output / "D4_UNSEEN_START.json", payload)
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
        _write(args.output_dir / "D4_STATE.json", {"status": "FAIL_CLOSED", "failures": failures})
        raise RuntimeError(f"unseen-start failures: {failures}")
    payload = _aggregate(args.output_dir)
    _write(args.output_dir / "D4_STATE.json", {"status": payload["status"], "counts": payload["counts"]})
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()
