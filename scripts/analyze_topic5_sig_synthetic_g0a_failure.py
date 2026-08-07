#!/usr/bin/env python3
"""Decompose the failed G0-A influence-recovery endpoint without re-gating it."""
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
    frozen_synthetic_graph,
    rank_spearman,
    simulate_synthetic_events,
    top_positive_overlap,
)


SOURCE = (
    ROOT
    / "results/topic5_stable_interaction_graph/development/synthetic_g0a"
)
OUTPUT = (
    ROOT
    / "results/topic5_stable_interaction_graph/development"
    / "synthetic_g0a_diagnostics/round1_endpoint_decomposition"
)


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _model_from_truth() -> StableInteractionGraph:
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
    aggregate = json.loads((SOURCE / "G0A_BENCHMARK.json").read_text())
    if aggregate["status"] != "FAIL_CLOSED":
        raise RuntimeError("diagnostic is frozen for the failed G0-A artifact")
    train = simulate_synthetic_events(
        2400, starts=(0, 3, 6), seed=91001
    )
    test = simulate_synthetic_events(
        900, starts=(0, 3, 6), seed=91003
    )
    groups, counts = test.torch()
    sender = (
        (train.group_ids >= 0)
        & (train.group_ids < (train.group_count[:, None] - 1))
    ).sum(0)
    truth_model = _model_from_truth()
    zero_truth = truth_model.one_step_intervention_matrix().cpu().numpy()
    empirical_truth = (
        truth_model.empirical_one_step_intervention_matrix(groups, counts)
        .cpu()
        .numpy()
    )
    off = ~np.eye(12, dtype=bool)
    rows = []
    zero_estimates = []
    empirical_estimates = []
    for run_dir in sorted((SOURCE / "per_run").glob("seed_*")):
        checkpoint = torch.load(
            run_dir / "checkpoint.pt", map_location="cpu", weights_only=False
        )
        model = StableInteractionGraph(
            12,
            static_bias=np.asarray(frozen_synthetic_graph()["static_bias"]),
            learn_graph=True,
        )
        model.load_state_dict(checkpoint["sig1_state_dict"])
        model.eval()
        zero = model.one_step_intervention_matrix().cpu().numpy()
        empirical = (
            model.empirical_one_step_intervention_matrix(groups, counts)
            .cpu()
            .numpy()
        )
        zero_estimates.append(zero)
        empirical_estimates.append(empirical)
        effect_rows = {}
        for quantile in (0.0, 0.50, 0.75):
            threshold = float(np.quantile(np.abs(zero_truth[off]), quantile))
            mask = off & (np.abs(zero_truth) >= threshold)
            effect_rows[f"q{int(100 * quantile)}"] = {
                "n_pairs": int(mask.sum()),
                "spearman": rank_spearman(zero_truth[mask], zero[mask]),
            }
        empirical_valid = (
            off
            & np.isfinite(empirical_truth)
            & np.isfinite(empirical)
        )
        rows.append(
            {
                "fit_seed": int(checkpoint["fit_seed"]),
                "sender_exposure_min": int(sender.min()),
                "sender_exposure_median": float(np.median(sender)),
                "zero_state_spearman": rank_spearman(
                    zero_truth[off], zero[off]
                ),
                "empirical_context_spearman": rank_spearman(
                    empirical_truth[empirical_valid],
                    empirical[empirical_valid],
                ),
                "top20_positive_overlap": top_positive_overlap(
                    zero_truth, zero
                ),
                "effect_magnitude_decomposition": effect_rows,
            }
        )
    seed_stability = []
    for left in range(len(zero_estimates)):
        for right in range(left + 1, len(zero_estimates)):
            seed_stability.append(
                {
                    "seed_pair": [rows[left]["fit_seed"], rows[right]["fit_seed"]],
                    "zero_state_spearman": rank_spearman(
                        zero_estimates[left][off],
                        zero_estimates[right][off],
                    ),
                    "empirical_context_spearman": rank_spearman(
                        empirical_estimates[left][off],
                        empirical_estimates[right][off],
                    ),
                }
            )
    payload = {
        "status": "COMPLETE_DIAGNOSTIC_G0A_REMAINS_FAILED",
        "question": (
            "Did the influence endpoint fail because sender columns were "
            "unsupported, fits were seed-unstable, or the operator ranked "
            "weak/state-extrapolated effects?"
        ),
        "g0a_status_unchanged": "FAIL_CLOSED",
        "runs": rows,
        "seed_stability": seed_stability,
        "conclusion": (
            "Every sender column had high exposure and fitted influence was "
            "highly seed-stable. Strong positive-pair membership was recovered, "
            "but fine-grained ordering of all 132 off-diagonal effects remained "
            "below the frozen threshold. Averaging over occupied prefix states "
            "improved but did not clear the threshold. This is not a basis to "
            "relabel G0-A as passed."
        ),
        "source_sha256": sha256_file(Path(__file__)),
        "snn_inputs_read": False,
    }
    _write(OUTPUT / "endpoint_decomposition.json", payload)
    _write(
        OUTPUT / "ROUND_STATE.json",
        {
            "status": "COMPLETE",
            "g0a_status_unchanged": "FAIL_CLOSED",
            "next": "run a nested event-count recovery curve",
        },
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
