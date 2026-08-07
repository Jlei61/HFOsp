#!/usr/bin/env python3
"""Run the SNN-independent G0-A Stable Interaction Graph benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_shared_propagation_field import sha256_file  # noqa: E402
from src.topic5_stable_interaction_graph import (  # noqa: E402
    CONTRACT_NAME,
    StableInteractionGraph,
    cardinality_schedule,
    fit_synthetic_sig,
    frozen_synthetic_graph,
    precedence_distance,
    precedence_matrix,
    rank_spearman,
    simulate_synthetic_events,
    top_positive_overlap,
)


FIT_SEEDS = (20260731, 20260732, 20260733)
MIN_NLL_GAIN = 0.02
MIN_INFLUENCE_SPEARMAN_EACH = 0.60
MIN_INFLUENCE_SPEARMAN_MEDIAN = 0.75
MIN_TOP_OVERLAP = 0.50


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n"
    )


def _truth_model() -> StableInteractionGraph:
    truth = frozen_synthetic_graph()
    weight = torch.as_tensor(np.asarray(truth["weight"]))
    model = StableInteractionGraph(
        weight.shape[0],
        static_bias=np.asarray(truth["static_bias"]),
        learn_graph=True,
        max_weight=3.0,
        initial_leak=float(truth["leak"]),
    )
    with torch.no_grad():
        ratio = (weight / model.max_weight).clamp(-0.999, 0.999)
        model.raw_weight.copy_(torch.atanh(ratio))
        model.phase_loading.copy_(
            torch.as_tensor(np.asarray(truth["phase_loading"]))
        )
        model.leak_logit.copy_(
            torch.tensor(
                np.log(float(truth["leak"]) / (1.0 - float(truth["leak"]))),
                dtype=torch.float32,
            )
        )
    model.eval()
    return model


def _nll(model, dataset, *, weight_override=None) -> float:
    groups, counts = dataset.torch()
    model.eval()
    with torch.no_grad():
        return float(
            model.nll_per_decision(
                groups, counts, weight_override=weight_override
            )
        )


def _rollout_precedence_distance(
    model,
    dataset,
    *,
    seed: int,
    weight_override=None,
) -> float:
    groups, counts = dataset.torch()
    first = groups == 0
    schedule = torch.as_tensor(
        cardinality_schedule(dataset.group_ids, dataset.group_count),
        dtype=torch.long,
    )
    generated = model.rollout(
        first,
        counts,
        schedule,
        generator=torch.Generator().manual_seed(int(seed)),
        weight_override=weight_override,
    )
    return precedence_distance(
        precedence_matrix(dataset.group_ids),
        precedence_matrix(generated.cpu().numpy()),
    )


def _graph_perturbations(weight: torch.Tensor, seed: int) -> dict[str, torch.Tensor]:
    contacts = weight.shape[0]
    rng = np.random.default_rng(int(seed))
    permutation = torch.as_tensor(rng.permutation(contacts), dtype=torch.long)
    shuffled = weight[:, permutation].clone()
    shuffled.fill_diagonal_(0.0)

    lesion = weight.clone()
    mask = ~torch.eye(contacts, dtype=torch.bool)
    values = lesion[mask]
    threshold = torch.quantile(values, 0.80)
    lesion[(lesion >= threshold) & mask] = 0.0
    return {"shuffle": shuffled, "top20_positive_lesion": lesion}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results/topic5_stable_interaction_graph/development"
        / "synthetic_g0a",
    )
    parser.add_argument("--max-epochs", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir
    train = simulate_synthetic_events(
        2400, starts=(0, 3, 6), seed=91001
    )
    validation = simulate_synthetic_events(
        600, starts=(0, 3, 6), seed=91002
    )
    test = simulate_synthetic_events(
        900, starts=(0, 3, 6), seed=91003
    )
    unseen = simulate_synthetic_events(
        600, starts=(9,), seed=91004
    )
    truth = _truth_model()
    truth_influence = truth.one_step_intervention_matrix().cpu().numpy()
    off_diagonal = ~np.eye(truth_influence.shape[0], dtype=bool)

    runs = []
    for fit_seed in FIT_SEEDS:
        no_graph = fit_synthetic_sig(
            train,
            validation,
            seed=fit_seed,
            learn_graph=False,
            max_epochs=args.max_epochs,
        )
        graph = fit_synthetic_sig(
            train,
            validation,
            seed=fit_seed,
            learn_graph=True,
            max_epochs=args.max_epochs,
        )
        learned_weight = graph.model.effective_weight().detach().cpu()
        learned_influence = (
            graph.model.one_step_intervention_matrix().cpu().numpy()
        )
        test_nll_no_graph = _nll(no_graph.model, test)
        test_nll_graph = _nll(graph.model, test)
        unseen_nll_no_graph = _nll(no_graph.model, unseen)
        unseen_nll_graph = _nll(graph.model, unseen)
        test_precedence_no_graph = _rollout_precedence_distance(
            no_graph.model, test, seed=fit_seed + 300
        )
        test_precedence_graph = _rollout_precedence_distance(
            graph.model, test, seed=fit_seed + 300
        )
        unseen_precedence_no_graph = _rollout_precedence_distance(
            no_graph.model, unseen, seed=fit_seed + 400
        )
        unseen_precedence_graph = _rollout_precedence_distance(
            graph.model, unseen, seed=fit_seed + 400
        )
        perturbation = {}
        for name, weight in _graph_perturbations(
            learned_weight, fit_seed
        ).items():
            nll = _nll(graph.model, test, weight_override=weight)
            precedence = _rollout_precedence_distance(
                graph.model,
                test,
                seed=fit_seed + 300,
                weight_override=weight,
            )
            perturbation[name] = {
                "test_nll_per_decision": nll,
                "test_precedence_distance": precedence,
                "worse_than_learned": bool(
                    nll > test_nll_graph
                    or precedence > test_precedence_graph
                ),
            }
        row = {
            "fit_seed": int(fit_seed),
            "n_train_events": int(len(train.group_ids)),
            "n_validation_events": int(len(validation.group_ids)),
            "n_test_events": int(len(test.group_ids)),
            "n_unseen_start_events": int(len(unseen.group_ids)),
            "sig0_best_epoch": int(no_graph.best_epoch),
            "sig1_best_epoch": int(graph.best_epoch),
            "sig0_validation_nll": float(no_graph.best_validation_nll),
            "sig1_validation_nll": float(graph.best_validation_nll),
            "sig0_training_adequacy": no_graph.adequacy,
            "sig1_training_adequacy": graph.adequacy,
            "test_nll_sig0": test_nll_no_graph,
            "test_nll_sig1": test_nll_graph,
            "test_nll_gain_sig1": test_nll_no_graph - test_nll_graph,
            "test_precedence_distance_sig0": test_precedence_no_graph,
            "test_precedence_distance_sig1": test_precedence_graph,
            "unseen_nll_sig0": unseen_nll_no_graph,
            "unseen_nll_sig1": unseen_nll_graph,
            "unseen_nll_gain_sig1": unseen_nll_no_graph - unseen_nll_graph,
            "unseen_precedence_distance_sig0": unseen_precedence_no_graph,
            "unseen_precedence_distance_sig1": unseen_precedence_graph,
            "influence_spearman": rank_spearman(
                truth_influence[off_diagonal],
                learned_influence[off_diagonal],
            ),
            "top20_positive_overlap": top_positive_overlap(
                truth_influence, learned_influence
            ),
            "graph_perturbations": perturbation,
            "all_perturbations_worse": bool(
                all(value["worse_than_learned"] for value in perturbation.values())
            ),
            "training_finite": bool(
                all(
                    np.isfinite(point["validation_nll_per_decision"])
                    for point in no_graph.history + graph.history
                )
            ),
            "forbidden_inputs_read": False,
            "snn_inputs_read": False,
        }
        run_dir = output / "per_run" / f"seed_{fit_seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "contract": CONTRACT_NAME,
                "fit_seed": fit_seed,
                "sig0_state_dict": no_graph.model.state_dict(),
                "sig1_state_dict": graph.model.state_dict(),
                "sig0_optimizer_state": no_graph.best_optimizer_state,
                "sig1_optimizer_state": graph.best_optimizer_state,
                "truth_weight": frozen_synthetic_graph()["weight"],
            },
            run_dir / "checkpoint.pt",
        )
        _write_json(run_dir / "summary.json", row)
        _write_json(run_dir / "sig0_history.json", no_graph.history)
        _write_json(run_dir / "sig1_history.json", graph.history)
        runs.append(row)

    influence_values = [row["influence_spearman"] for row in runs]
    checks = {
        "three_complete_finite_fits": bool(
            len(runs) == 3
            and all(row["training_finite"] for row in runs)
            and all(
                row["sig0_training_adequacy"]["converged"]
                and row["sig1_training_adequacy"]["converged"]
                for row in runs
            )
        ),
        "sig1_nll_gain_each_ge_0_02": bool(
            all(row["test_nll_gain_sig1"] >= MIN_NLL_GAIN for row in runs)
        ),
        "influence_spearman_each_ge_0_60": bool(
            all(
                value >= MIN_INFLUENCE_SPEARMAN_EACH
                for value in influence_values
            )
        ),
        "influence_spearman_median_ge_0_75": bool(
            np.median(influence_values) >= MIN_INFLUENCE_SPEARMAN_MEDIAN
        ),
        "top20_overlap_each_ge_0_50": bool(
            all(
                row["top20_positive_overlap"] >= MIN_TOP_OVERLAP
                for row in runs
            )
        ),
        "shuffle_and_lesion_worse_each_seed": bool(
            all(row["all_perturbations_worse"] for row in runs)
        ),
        "unseen_start_nll_and_precedence_better_each_seed": bool(
            all(
                row["unseen_nll_sig1"] < row["unseen_nll_sig0"]
                and row["unseen_precedence_distance_sig1"]
                < row["unseen_precedence_distance_sig0"]
                for row in runs
            )
        ),
    }
    passed = bool(all(checks.values()))
    aggregate = {
        "contract": CONTRACT_NAME,
        "benchmark": "generic_synthetic_feedback_graph_g0a",
        "status": "PASS" if passed else "FAIL_CLOSED",
        "human_pilot_decision": (
            "START_HUMAN_PILOT" if passed else "BLOCK_HUMAN_PILOT"
        ),
        "fit_seeds": list(FIT_SEEDS),
        "thresholds_frozen_before_run": {
            "min_nll_gain_each": MIN_NLL_GAIN,
            "min_influence_spearman_each": MIN_INFLUENCE_SPEARMAN_EACH,
            "min_influence_spearman_median": MIN_INFLUENCE_SPEARMAN_MEDIAN,
            "min_top20_overlap_each": MIN_TOP_OVERLAP,
        },
        "checks": checks,
        "runs": runs,
        "influence_spearman_median": float(np.median(influence_values)),
        "test_nll_gain_median": float(
            np.median([row["test_nll_gain_sig1"] for row in runs])
        ),
        "unseen_nll_gain_median": float(
            np.median([row["unseen_nll_gain_sig1"] for row in runs])
        ),
        "source_sha256": {
            str(Path(__file__).relative_to(ROOT)): sha256_file(Path(__file__)),
            "src/topic5_stable_interaction_graph.py": sha256_file(
                ROOT / "src/topic5_stable_interaction_graph.py"
            ),
        },
        "snn_inputs_read": False,
        "claim": (
            "This is an engineering and idealized identifiability calibration, "
            "not evidence about human or SNN mechanisms."
        ),
    }
    _write_json(output / "G0A_BENCHMARK.json", aggregate)
    _write_json(
        output / "G0A_STATE.json",
        {
            "status": aggregate["status"],
            "human_pilot_decision": aggregate["human_pilot_decision"],
            "failed_checks": [
                name for name, value in checks.items() if not value
            ],
            "snn_inputs_read": False,
        },
    )
    print(json.dumps(_jsonable(aggregate), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
