#!/usr/bin/env python3
"""Run the one allowed independent G0-A2 SIG engineering confirmation."""
from __future__ import annotations

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
    StableInteractionGraph,
    cardinality_schedule,
    fit_synthetic_sig,
    independent_synthetic_graph,
    precedence_distance,
    precedence_matrix,
    rank_spearman,
    simulate_synthetic_events,
    top_positive_overlap,
)


FIT_SEEDS = (20260731, 20260732, 20260733)
OUTPUT = (
    ROOT
    / "results/topic5_stable_interaction_graph/development"
    / "synthetic_g0a2_independent_confirmation_v0_2_training_adequacy"
)


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _truth_model(graph) -> StableInteractionGraph:
    model = StableInteractionGraph(
        12,
        static_bias=np.asarray(graph["static_bias"]),
        learn_graph=True,
        initial_leak=float(graph["leak"]),
    )
    with torch.no_grad():
        model.raw_weight.copy_(
            torch.atanh(
                torch.as_tensor(np.asarray(graph["weight"]) / 3.0).clamp(
                    -0.999, 0.999
                )
            )
        )
        model.phase_loading.copy_(
            torch.as_tensor(np.asarray(graph["phase_loading"]))
        )
    model.eval()
    return model


def _nll(model, dataset, weight=None) -> float:
    groups, counts = dataset.torch()
    with torch.no_grad():
        return float(
            model.nll_per_decision(
                groups, counts, weight_override=weight
            )
        )


def _precedence(model, dataset, seed: int, weight=None) -> float:
    groups, counts = dataset.torch()
    schedule = torch.as_tensor(
        cardinality_schedule(dataset.group_ids, dataset.group_count),
        dtype=torch.long,
    )
    generated = model.rollout(
        groups == 0,
        counts,
        schedule,
        generator=torch.Generator().manual_seed(int(seed)),
        weight_override=weight,
    )
    return precedence_distance(
        precedence_matrix(dataset.group_ids),
        precedence_matrix(generated.cpu().numpy()),
    )


def _perturb(weight: torch.Tensor, seed: int) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(int(seed))
    permutation = torch.as_tensor(rng.permutation(len(weight)))
    shuffled = weight[:, permutation].clone()
    shuffled.fill_diagonal_(0)
    lesion = weight.clone()
    mask = ~torch.eye(len(weight), dtype=torch.bool)
    threshold = torch.quantile(lesion[mask], 0.80)
    lesion[(lesion >= threshold) & mask] = 0.0
    return {"shuffle": shuffled, "top20_positive_lesion": lesion}


def main() -> None:
    curve = json.loads(
        (
            ROOT
            / "results/topic5_stable_interaction_graph/development"
            / "synthetic_g0a_diagnostics/round2_nested_event_count"
            / "nested_event_count.json"
        ).read_text()
    )
    if curve["first_zero_state_threshold_count"] != 9600:
        raise RuntimeError("G0-A2 was not authorized by the frozen learning curve")
    graph = independent_synthetic_graph(20260801)
    train = simulate_synthetic_events(
        9600,
        starts=(0, 4, 8),
        seed=93001,
        graph=graph,
    )
    validation = simulate_synthetic_events(
        1200,
        starts=(0, 4, 8),
        seed=93002,
        graph=graph,
    )
    test = simulate_synthetic_events(
        1200,
        starts=(0, 4, 8),
        seed=93003,
        graph=graph,
    )
    unseen = simulate_synthetic_events(
        1200,
        starts=(10,),
        seed=93004,
        graph=graph,
    )
    if int(np.sum(train.group_ids[:, 10] > 0)) < 20:
        raise RuntimeError("held-out start lacks intermediate-node support")
    truth = _truth_model(graph)
    truth_influence = truth.one_step_intervention_matrix().cpu().numpy()
    off = ~np.eye(12, dtype=bool)
    rows = []
    for fit_seed in FIT_SEEDS:
        sig0 = fit_synthetic_sig(
            train,
            validation,
            seed=fit_seed,
            learn_graph=False,
            static_bias=np.asarray(graph["static_bias"]),
        )
        sig1 = fit_synthetic_sig(
            train,
            validation,
            seed=fit_seed,
            learn_graph=True,
            static_bias=np.asarray(graph["static_bias"]),
        )
        test_nll0 = _nll(sig0.model, test)
        test_nll1 = _nll(sig1.model, test)
        unseen_nll0 = _nll(sig0.model, unseen)
        unseen_nll1 = _nll(sig1.model, unseen)
        test_precedence0 = _precedence(sig0.model, test, fit_seed + 100)
        test_precedence1 = _precedence(sig1.model, test, fit_seed + 100)
        unseen_precedence0 = _precedence(
            sig0.model, unseen, fit_seed + 200
        )
        unseen_precedence1 = _precedence(
            sig1.model, unseen, fit_seed + 200
        )
        weight = sig1.model.effective_weight().detach().cpu()
        perturbations = {}
        for name, altered in _perturb(weight, fit_seed).items():
            changed_nll = _nll(sig1.model, test, altered)
            changed_precedence = _precedence(
                sig1.model, test, fit_seed + 100, altered
            )
            perturbations[name] = {
                "test_nll": changed_nll,
                "test_precedence_distance": changed_precedence,
                "worse": bool(
                    changed_nll > test_nll1
                    or changed_precedence > test_precedence1
                ),
            }
        estimate = sig1.model.one_step_intervention_matrix().cpu().numpy()
        row = {
            "fit_seed": fit_seed,
            "sig0_best_epoch": sig0.best_epoch,
            "sig1_best_epoch": sig1.best_epoch,
            "sig0_learning_rate": sig0.learning_rate,
            "sig1_learning_rate": sig1.learning_rate,
            "sig0_training_adequacy": sig0.adequacy,
            "sig1_training_adequacy": sig1.adequacy,
            "test_nll_sig0": test_nll0,
            "test_nll_sig1": test_nll1,
            "test_nll_gain": test_nll0 - test_nll1,
            "test_precedence_sig0": test_precedence0,
            "test_precedence_sig1": test_precedence1,
            "unseen_nll_sig0": unseen_nll0,
            "unseen_nll_sig1": unseen_nll1,
            "unseen_precedence_sig0": unseen_precedence0,
            "unseen_precedence_sig1": unseen_precedence1,
            "influence_spearman": rank_spearman(
                truth_influence[off], estimate[off]
            ),
            "top20_positive_overlap": top_positive_overlap(
                truth_influence, estimate
            ),
            "perturbations": perturbations,
            "all_perturbations_worse": bool(
                all(value["worse"] for value in perturbations.values())
            ),
            "training_finite": bool(
                all(
                    np.isfinite(point["validation_nll_per_decision"])
                    for point in sig0.history + sig1.history
                )
            ),
        }
        run_dir = OUTPUT / "per_run" / f"seed_{fit_seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "fit_seed": fit_seed,
                "sig0_state_dict": sig0.model.state_dict(),
                "sig1_state_dict": sig1.model.state_dict(),
                "sig0_optimizer_state": sig0.best_optimizer_state,
                "sig1_optimizer_state": sig1.best_optimizer_state,
                "independent_graph_seed": 20260801,
            },
            run_dir / "checkpoint.pt",
        )
        _write(run_dir / "summary.json", row)
        rows.append(row)

    influence = [row["influence_spearman"] for row in rows]
    checks = {
        "three_complete_finite_fits": bool(
            len(rows) == 3
            and all(row["training_finite"] for row in rows)
            and all(
                row["sig0_training_adequacy"]["converged"]
                and row["sig1_training_adequacy"]["converged"]
                for row in rows
            )
        ),
        "sig1_nll_gain_each_ge_0_02": bool(
            all(row["test_nll_gain"] >= 0.02 for row in rows)
        ),
        "influence_spearman_each_ge_0_60": bool(
            all(value >= 0.60 for value in influence)
        ),
        "influence_spearman_median_ge_0_75": bool(
            np.median(influence) >= 0.75
        ),
        "top20_overlap_each_ge_0_50": bool(
            all(row["top20_positive_overlap"] >= 0.50 for row in rows)
        ),
        "shuffle_and_lesion_worse_each_seed": bool(
            all(row["all_perturbations_worse"] for row in rows)
        ),
        "unseen_start_nll_and_precedence_better_each_seed": bool(
            all(
                row["unseen_nll_sig1"] < row["unseen_nll_sig0"]
                and row["unseen_precedence_sig1"]
                < row["unseen_precedence_sig0"]
                for row in rows
            )
        ),
    }
    passed = bool(all(checks.values()))
    payload = {
        "contract": "topic5_sig_synthetic_g0a2_independent_confirmation",
        "runner_revision": "v0_2_unified_training_adequacy",
        "status": "PASS" if passed else "FAIL_CLOSED",
        "human_pilot_decision": (
            "START_HUMAN_PILOT" if passed else "BLOCK_HUMAN_PILOT"
        ),
        "original_g0a_status_unchanged": "FAIL_CLOSED",
        "calibration_label": (
            "ENGINEERING_CALIBRATION_PASS_AT_N_MIN_9600"
            if passed
            else "INDEPENDENT_CONFIRMATION_FAILED"
        ),
        "independent_graph_seed": 20260801,
        "event_seeds": [93001, 93002, 93003, 93004],
        "n_train_events": 9600,
        "checks": checks,
        "runs": rows,
        "influence_spearman_median": float(np.median(influence)),
        "test_nll_gain_median": float(
            np.median([row["test_nll_gain"] for row in rows])
        ),
        "source_sha256": {
            str(Path(__file__).relative_to(ROOT)): sha256_file(Path(__file__)),
            "src/topic5_stable_interaction_graph.py": sha256_file(
                ROOT / "src/topic5_stable_interaction_graph.py"
            ),
        },
        "snn_inputs_read": False,
        "claim": (
            "Independent generic-graph engineering calibration only; not a "
            "human or SNN mechanism result."
        ),
    }
    _write(OUTPUT / "G0A2_CONFIRMATION.json", payload)
    _write(
        OUTPUT / "G0A2_STATE.json",
        {
            "status": payload["status"],
            "human_pilot_decision": payload["human_pilot_decision"],
            "failed_checks": [
                name for name, value in checks.items() if not value
            ],
            "original_g0a_status_unchanged": "FAIL_CLOSED",
        },
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
