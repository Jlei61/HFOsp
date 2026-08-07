#!/usr/bin/env python3
"""Nested event-count diagnostic after the failed SIG G0-A benchmark."""
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
    SyntheticGraphDataset,
    fit_synthetic_sig,
    frozen_synthetic_graph,
    rank_spearman,
    simulate_synthetic_events,
    top_positive_overlap,
)


EVENT_COUNTS = (600, 1200, 2400, 4800, 9600)
FIT_SEEDS = (20260731, 20260732, 20260733)
OUTPUT = (
    ROOT
    / "results/topic5_stable_interaction_graph/development"
    / "synthetic_g0a_diagnostics/round2_nested_event_count"
)


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _slice(data: SyntheticGraphDataset, count: int) -> SyntheticGraphDataset:
    return SyntheticGraphDataset(
        group_ids=data.group_ids[:count].copy(),
        group_count=data.group_count[:count].copy(),
        start_contact=data.start_contact[:count].copy(),
    )


def _truth_model() -> StableInteractionGraph:
    truth = frozen_synthetic_graph()
    model = StableInteractionGraph(
        12,
        static_bias=np.asarray(truth["static_bias"]),
        learn_graph=True,
        initial_leak=float(truth["leak"]),
    )
    with torch.no_grad():
        model.raw_weight.copy_(
            torch.atanh(
                torch.as_tensor(np.asarray(truth["weight"]) / 3.0).clamp(
                    -0.999, 0.999
                )
            )
        )
        model.phase_loading.copy_(
            torch.as_tensor(np.asarray(truth["phase_loading"]))
        )
    model.eval()
    return model


def main() -> None:
    original = json.loads(
        (
            ROOT
            / "results/topic5_stable_interaction_graph/development"
            / "synthetic_g0a/G0A_BENCHMARK.json"
        ).read_text()
    )
    if original["status"] != "FAIL_CLOSED":
        raise RuntimeError("learning curve is conditioned on frozen G0-A failure")
    master = simulate_synthetic_events(
        max(EVENT_COUNTS), starts=(0, 3, 6), seed=92001
    )
    validation = simulate_synthetic_events(
        1200, starts=(0, 3, 6), seed=92002
    )
    test = simulate_synthetic_events(
        1200, starts=(0, 3, 6), seed=92003
    )
    test_groups, test_counts = test.torch()
    truth_model = _truth_model()
    zero_truth = truth_model.one_step_intervention_matrix().cpu().numpy()
    empirical_truth = (
        truth_model.empirical_one_step_intervention_matrix(
            test_groups, test_counts
        )
        .cpu()
        .numpy()
    )
    off = ~np.eye(12, dtype=bool)
    truth_test_nll = float(
        truth_model.nll_per_decision(test_groups, test_counts)
    )
    rows = []
    for event_count in EVENT_COUNTS:
        train = _slice(master, event_count)
        for fit_seed in FIT_SEEDS:
            fit = fit_synthetic_sig(
                train,
                validation,
                seed=fit_seed,
                learn_graph=True,
                max_epochs=300,
            )
            model = fit.model
            test_nll = float(
                model.nll_per_decision(test_groups, test_counts)
            )
            zero = model.one_step_intervention_matrix().cpu().numpy()
            empirical = (
                model.empirical_one_step_intervention_matrix(
                    test_groups, test_counts
                )
                .cpu()
                .numpy()
            )
            valid_empirical = (
                off
                & np.isfinite(empirical_truth)
                & np.isfinite(empirical)
            )
            row = {
                "n_train_events": int(event_count),
                "fit_seed": int(fit_seed),
                "best_epoch": int(fit.best_epoch),
                "learning_rate": float(fit.learning_rate),
                "recovery_depth": int(fit.recovery_depth),
                "training_adequacy": fit.adequacy,
                "validation_nll": float(fit.best_validation_nll),
                "test_nll": test_nll,
                "truth_test_nll": truth_test_nll,
                "excess_test_nll": test_nll - truth_test_nll,
                "zero_state_influence_spearman": rank_spearman(
                    zero_truth[off], zero[off]
                ),
                "empirical_influence_spearman": rank_spearman(
                    empirical_truth[valid_empirical],
                    empirical[valid_empirical],
                ),
                "top20_positive_overlap": top_positive_overlap(
                    zero_truth, zero
                ),
            }
            run_dir = OUTPUT / "per_run" / (
                f"n{event_count}_seed{fit_seed}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "n_train_events": event_count,
                    "fit_seed": fit_seed,
                    "state_dict": model.state_dict(),
                    "optimizer_state": fit.best_optimizer_state,
                },
                run_dir / "checkpoint.pt",
            )
            _write(run_dir / "summary.json", row)
            rows.append(row)

    by_count = {}
    for event_count in EVENT_COUNTS:
        subset = [
            row for row in rows if row["n_train_events"] == event_count
        ]
        by_count[str(event_count)] = {
            name: {
                "min": float(np.min([row[name] for row in subset])),
                "median": float(np.median([row[name] for row in subset])),
                "max": float(np.max([row[name] for row in subset])),
            }
            for name in (
                "excess_test_nll",
                "zero_state_influence_spearman",
                "empirical_influence_spearman",
                "top20_positive_overlap",
            )
        }
    first_zero_threshold = next(
        (
            count
            for count in EVENT_COUNTS
            if by_count[str(count)]["zero_state_influence_spearman"]["min"]
            >= 0.60
            and by_count[str(count)]["zero_state_influence_spearman"]["median"]
            >= 0.75
        ),
        None,
    )
    first_empirical_threshold = next(
        (
            count
            for count in EVENT_COUNTS
            if by_count[str(count)]["empirical_influence_spearman"]["min"]
            >= 0.60
            and by_count[str(count)]["empirical_influence_spearman"]["median"]
            >= 0.75
        ),
        None,
    )
    payload = {
        "status": "COMPLETE_DIAGNOSTIC_G0A_REMAINS_FAILED",
        "question": (
            "Does all-pair observable influence recovery cross the frozen "
            "threshold as nested event count increases?"
        ),
        "g0a_status_unchanged": "FAIL_CLOSED",
        "event_counts": list(EVENT_COUNTS),
        "fit_seeds": list(FIT_SEEDS),
        "nested_master_seed": 92001,
        "by_event_count": by_count,
        "first_zero_state_threshold_count": first_zero_threshold,
        "first_empirical_threshold_count": first_empirical_threshold,
        "rows": rows,
        "interpretation": (
            "This curve estimates sample dependence on one already-viewed "
            "synthetic graph. Any crossing can design a future independent "
            "calibration, but cannot retroactively pass G0-A or select a human "
            "sample threshold."
        ),
        "source_sha256": {
            str(Path(__file__).relative_to(ROOT)): sha256_file(Path(__file__)),
            "src/topic5_stable_interaction_graph.py": sha256_file(
                ROOT / "src/topic5_stable_interaction_graph.py"
            ),
        },
        "snn_inputs_read": False,
    }
    _write(OUTPUT / "nested_event_count.json", payload)
    _write(
        OUTPUT / "ROUND_STATE.json",
        {
            "status": "COMPLETE",
            "g0a_status_unchanged": "FAIL_CLOSED",
            "first_zero_state_threshold_count": first_zero_threshold,
            "first_empirical_threshold_count": first_empirical_threshold,
        },
    )
    print(json.dumps(payload["by_event_count"], indent=2))


if __name__ == "__main__":
    main()
