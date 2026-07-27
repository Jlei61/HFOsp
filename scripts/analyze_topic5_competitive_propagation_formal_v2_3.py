#!/usr/bin/env python3
"""Patient-first formal v2.3 Claims A-D and Markov benefit recovery."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT / "results/topic5_symmetric_axis_competitive_propagation_v2_3"
)
FORMAL = BASE / "formal"
VARIANTS = (
    "local_isotropic_two_state",
    "axis_one_state_no_competition",
    "axis_two_state_no_source",
    "axis_instantaneous_no_history",
    "axis_two_state_source_full",
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    draws = np.median(
        generator.choice(
            values, size=(20_000, len(values)), replace=True
        ),
        axis=1,
    )
    return tuple(map(float, np.quantile(draws, [0.025, 0.975])))


def bh_fdr(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def comparison_table(
    patient_metrics: pd.DataFrame,
) -> pd.DataFrame:
    definitions = (
        (
            "claim_A_predictive_adequacy",
            "node_bias_categorical",
            "axis_two_state_source_full",
            True,
        ),
        (
            "claim_B_history_vs_instantaneous",
            "axis_instantaneous_no_history",
            "axis_two_state_source_full",
            True,
        ),
        (
            "claim_B_competition_vs_one_state",
            "axis_one_state_no_competition",
            "axis_two_state_source_full",
            True,
        ),
        (
            "claim_C_axis_bundle_vs_local",
            "local_isotropic_two_state",
            "axis_two_state_source_full",
            True,
        ),
        (
            "claim_C_matched_axis_no_source_vs_local",
            "local_isotropic_two_state",
            "axis_two_state_no_source",
            True,
        ),
        (
            "claim_D_source_conditioned_direction",
            "axis_two_state_no_source",
            "axis_two_state_source_full",
            True,
        ),
        (
            "benchmark_full_vs_ordered_markov",
            "empirical_ordered_history_markov",
            "axis_two_state_source_full",
            False,
        ),
    )
    pivot = patient_metrics.pivot(
        index="subject", columns="model", values="heldout_categorical_nll"
    )
    rows = []
    for index, (name, baseline, model, registered) in enumerate(definitions):
        benefit = (pivot[baseline] - pivot[model]).to_numpy(float)
        pvalue = (
            1.0
            if np.allclose(benefit, 0.0)
            else float(
                wilcoxon(
                    benefit,
                    alternative="greater",
                    zero_method="wilcox",
                    method="auto",
                ).pvalue
            )
        )
        low, high = bootstrap_ci(benefit, 20260727 + index)
        rows.append(
            {
                "comparison": name,
                "baseline": baseline,
                "model": model,
                "registered_claim_family": registered,
                "n_patients": len(benefit),
                "median_benefit": float(np.median(benefit)),
                "median_ci95_low": low,
                "median_ci95_high": high,
                "n_positive": int(np.sum(benefit > 0)),
                "fraction_positive": float(np.mean(benefit > 0)),
                "wilcoxon_one_sided_p": pvalue,
            }
        )
    table = pd.DataFrame(rows)
    registered = table.registered_claim_family.to_numpy(bool)
    table["bh_fdr_q"] = np.nan
    table.loc[registered, "bh_fdr_q"] = bh_fdr(
        table.loc[registered, "wilcoxon_one_sided_p"].to_numpy(float)
    )
    table["pass"] = (
        table.registered_claim_family
        & (table.median_benefit > 0)
        & (table.median_ci95_low > 0)
        & (table.n_positive > table.n_patients / 2)
        & (table.bh_fdr_q < 0.05)
    )
    return table


def main() -> None:
    launcher = json.loads(
        (FORMAL / "LAUNCHER_STATE.json").read_text(encoding="utf-8")
    )
    markov_state = json.loads(
        (FORMAL / "MARKOV_BENCHMARK_STATE.json").read_text(encoding="utf-8")
    )
    if launcher.get("status") != "COMPLETE" or launcher.get("n_tasks_failed"):
        raise SystemExit("formal trainer is not complete")
    if markov_state.get("status") != "COMPLETE":
        raise SystemExit("categorical Markov benchmark is not complete")

    run_rows: list[dict[str, Any]] = []
    resolved_hashes: set[tuple[str, str]] = set()
    for metrics_path in sorted((FORMAL / "runs").glob("*/*/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        resolved = json.loads(
            (metrics_path.parent / "resolved_config.json").read_text(
                encoding="utf-8"
            )
        )
        if payload.get("target_values_read") or resolved.get(
            "target_values_read"
        ):
            raise SystemExit(f"target leak in {metrics_path}")
        if resolved.get("heldout_used_for_training_or_epoch_selection"):
            raise SystemExit(f"heldout selection leak in {metrics_path}")
        resolved_hashes.add(
            (resolved["core_sha256"], resolved["trainer_sha256"])
        )
        node_values = []
        for variant in VARIANTS:
            item = payload["variants"][variant]
            heldout = item["metrics"]["heldout20_sealed"]
            if not heldout["finite"]:
                raise SystemExit(f"non-finite heldout metric in {metrics_path}")
            node_values.append(float(heldout["node_categorical_nll"]))
            run_rows.append(
                {
                    "subject": payload["subject"],
                    "seed": payload["seed"],
                    "model": variant,
                    "heldout_categorical_nll": heldout[
                        "full_categorical_nll"
                    ],
                    "heldout_node_nll": heldout["node_categorical_nll"],
                    "best_epoch": item["best_epoch"],
                    "epochs_completed": item["epochs_completed"],
                    "early_stopped": item["early_stopped"],
                    "gamma": item["parameters"]["gamma"],
                    "gain_propagation": item["parameters"][
                        "gain_propagation"
                    ],
                    "gain_competition": item["parameters"][
                        "gain_competition"
                    ],
                    "source_beta": item["parameters"]["source_beta"],
                    "rho_propagation": item["parameters"][
                        "rho_propagation"
                    ],
                    "rho_competition": item["parameters"][
                        "rho_competition"
                    ],
                    "target_values_read": False,
                    "metrics_path": str(metrics_path.relative_to(ROOT)),
                }
            )
        if max(node_values) - min(node_values) > 1.0e-10:
            raise SystemExit(f"node baseline drift within {metrics_path}")
    runs = pd.DataFrame(run_rows)
    if (
        len(runs) != 22 * 3 * len(VARIANTS)
        or runs.groupby(["subject", "model"]).seed.nunique().min() != 3
        or len(resolved_hashes) != 1
    ):
        raise SystemExit("formal run count/seed/code fingerprint drifted")
    runs.to_csv(FORMAL / "formal_run_inventory.csv", index=False)

    patient = (
        runs.groupby(["subject", "model"], as_index=False)
        .agg(
            heldout_categorical_nll=("heldout_categorical_nll", "mean"),
            seed_sd=("heldout_categorical_nll", "std"),
            median_best_epoch=("best_epoch", "median"),
            gamma=("gamma", "mean"),
            gain_propagation=("gain_propagation", "mean"),
            gain_competition=("gain_competition", "mean"),
            source_beta=("source_beta", "mean"),
        )
    )
    node = (
        runs.groupby("subject", as_index=False)
        .heldout_node_nll.mean()
        .rename(columns={"heldout_node_nll": "heldout_categorical_nll"})
    )
    node["model"] = "node_bias_categorical"
    node["seed_sd"] = 0.0
    node["median_best_epoch"] = np.nan
    for column in ("gamma", "gain_propagation", "gain_competition", "source_beta"):
        node[column] = np.nan
    markov = pd.read_csv(FORMAL / "markov_benchmarks.csv").rename(
        columns={"heldout_categorical_nll": "heldout_categorical_nll"}
    )
    markov = markov[
        ["subject", "model", "heldout_categorical_nll"]
    ].copy()
    markov["seed_sd"] = 0.0
    markov["median_best_epoch"] = np.nan
    for column in ("gamma", "gain_propagation", "gain_competition", "source_beta"):
        markov[column] = np.nan
    # The independently recomputed node row must agree with the model-attached
    # fixed node baseline before it is dropped from the Markov table.
    markov_node = markov[markov.model == "node_bias_categorical"].set_index(
        "subject"
    )
    attached_node = node.set_index("subject")
    error = np.max(
        np.abs(
            markov_node.heldout_categorical_nll
            - attached_node.heldout_categorical_nll
        )
    )
    if error > 1.0e-10:
        raise SystemExit(f"independent node baseline mismatch: {error}")
    markov = markov[markov.model != "node_bias_categorical"]
    patient = pd.concat([patient, node, markov], ignore_index=True)
    patient.to_csv(FORMAL / "patient_model_metrics.csv", index=False)

    comparisons = comparison_table(patient)
    comparisons.to_csv(FORMAL / "claim_comparisons.csv", index=False)
    lookup = comparisons.set_index("comparison")
    claim_a = bool(lookup.loc["claim_A_predictive_adequacy", "pass"])
    claim_b_history = bool(
        lookup.loc["claim_B_history_vs_instantaneous", "pass"]
    )
    claim_b_competition = bool(
        lookup.loc["claim_B_competition_vs_one_state", "pass"]
    )
    claim_b = claim_b_history and claim_b_competition
    claim_c = bool(
        lookup.loc["claim_C_axis_bundle_vs_local", "pass"]
    )
    matched_axis = bool(
        lookup.loc["claim_C_matched_axis_no_source_vs_local", "pass"]
    )
    claim_d = bool(
        lookup.loc["claim_D_source_conditioned_direction", "pass"]
    )

    pivot = patient.pivot(
        index="subject", columns="model", values="heldout_categorical_nll"
    )
    recovery = pd.DataFrame(
        {
            "subject": pivot.index,
            "full_over_node_benefit": (
                pivot["node_bias_categorical"]
                - pivot["axis_two_state_source_full"]
            ),
            "last_rank_markov_over_node_benefit": (
                pivot["node_bias_categorical"]
                - pivot["empirical_last_rank_markov"]
            ),
            "ordered_markov_over_node_benefit": (
                pivot["node_bias_categorical"]
                - pivot["empirical_ordered_history_markov"]
            ),
        }
    ).reset_index(drop=True)
    recovery["full_fraction_of_ordered_markov_benefit"] = np.where(
        recovery.ordered_markov_over_node_benefit > 0,
        recovery.full_over_node_benefit
        / recovery.ordered_markov_over_node_benefit,
        np.nan,
    )
    recovery.to_csv(FORMAL / "benefit_recovery.csv", index=False)
    median_full = float(np.median(recovery.full_over_node_benefit))
    median_markov = float(
        np.median(recovery.ordered_markov_over_node_benefit)
    )
    status = {
        "contract": "topic5_symmetric_axis_competitive_propagation_v2_3",
        "status": "COMPLETE",
        "n_physical_axis_patients": 22,
        "n_seeds": 3,
        "claim_A_predictive_adequacy": (
            "PASS" if claim_a else "FAIL"
        ),
        "claim_B_history_state_necessary": (
            "PASS" if claim_b else "FAIL"
        ),
        "claim_B_history_vs_instantaneous": (
            "PASS" if claim_b_history else "FAIL"
        ),
        "claim_B_competition_vs_one_state": (
            "PASS" if claim_b_competition else "FAIL"
        ),
        "claim_C_axis_bundle_increment": (
            "PASS" if claim_c else "FAIL"
        ),
        "claim_C_matched_axis_increment": (
            "PASS" if matched_axis else "FAIL"
        ),
        "claim_D_source_conditioned_direction": (
            "PASS" if claim_d else "FAIL"
        ),
        "latent_state_analysis_allowed": bool(
            claim_a and claim_b and claim_c and matched_axis
        ),
        "physical_axis_claim_allowed": bool(claim_c and matched_axis),
        "source_reversal_analysis_allowed": claim_d,
        "median_full_over_node_benefit": median_full,
        "median_ordered_markov_over_node_benefit": median_markov,
        "median_benefit_recovery_ratio": (
            median_full / median_markov if median_markov > 0 else None
        ),
        "matched_axis_safeguard": (
            "Required in addition to the preregistered full-vs-local "
            "comparison because the latter also differs in its source term."
        ),
        "early_ictal_transfer": (
            "BLOCKED_PENDING_INTERICTAL_GATES_AND_EXACT_SOURCE_METADATA"
        ),
        "heldout_used_for_training_or_epoch_selection": False,
        "target_values_read": False,
        "core_and_trainer_fingerprint_unique": True,
        "max_independent_node_nll_error": float(error),
    }
    atomic_json(FORMAL / "FORMAL_GATE_STATUS.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
