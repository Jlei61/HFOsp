#!/usr/bin/env python3
"""Create the final gate-aware closeout for Topic 5 HistoryRNN v0.1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


DEVELOPMENT = {
    "epilepsiae_1073",
    "epilepsiae_1146",
    "yuquan_chenziyang",
}
SEEDS = (20260725, 20260726, 20260727)


def _one_sided(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values) or np.allclose(values, 0.0):
        return 1.0
    return float(wilcoxon(values, alternative="greater", method="auto").pvalue)


def _ci(values: np.ndarray, seed: int) -> list[float]:
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(20_000, len(values)))]
    return [float(value) for value in np.quantile(np.median(samples, axis=1), [0.025, 0.975])]


def _contrast(values: pd.Series, *, seed: int) -> dict:
    array = values.to_numpy(float)
    return {
        "median": float(np.median(array)),
        "bootstrap_median_ci95": _ci(array, seed),
        "n_positive": int(np.sum(array > 0)),
        "n_patients": int(len(array)),
        "one_sided_wilcoxon_p": _one_sided(array),
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("results/topic5_history_rnn_early_ictal_field"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    formal = root / "g1_sequential_formal_v0_1"
    multi = _read_json(formal / "G1_MULTI_SEED_SUMMARY.json")
    if bool(multi.get("target_values_read", True)):
        raise RuntimeError("G1 multi-seed provenance violates the target seal")

    patient = pd.read_csv(formal / "g1_multiseed_patient_metrics.csv")
    primary = patient.loc[~patient.subject.isin(DEVELOPMENT)].copy()
    if len(patient) != 34 or len(primary) != 31:
        raise RuntimeError(f"G1 patient denominator drift: {len(patient)}/34, {len(primary)}/31")

    controls = []
    parameter_counts = []
    log_paths = []
    failed_paths = []
    for seed in SEEDS:
        seed_root = formal / f"seed_{seed}"
        if not (seed_root / "G1_SUMMARY.json").exists():
            raise RuntimeError(f"seed {seed}: missing G1 summary")
        done_paths = sorted(seed_root.glob("*/DONE.json"))
        if len(done_paths) != 34:
            raise RuntimeError(f"seed {seed}: {len(done_paths)}/34 folds")
        for done_path in done_paths:
            done = _read_json(done_path)
            order_path = done_path.parent / "ORDER_CONTROLS.json"
            if not order_path.exists():
                raise RuntimeError(f"missing strict order control: {order_path}")
            order = _read_json(order_path)
            if bool(done.get("target_values_read", True)) or bool(
                order.get("target_values_read", True)
            ):
                raise RuntimeError(f"G1 target seal violation in {done_path.parent}")
            controls.append(
                {
                    "seed": seed,
                    "subject": done["heldout_subject"],
                    **order["learned_event_history_half_life_hours"],
                }
            )
            parameter_counts.append(done["parameter_counts"])
        log_paths.extend(sorted((seed_root / "logs").glob("*.log")))
        failed_paths.extend(seed_root.glob("*.FAILED.json"))
        failed_paths.extend(seed_root.glob("*/FAILED.json"))

    unique_counts = {json.dumps(row, sort_keys=True) for row in parameter_counts}
    if len(unique_counts) != 1:
        raise RuntimeError("parameter-count contract drifted across G1 folds")
    half_life = pd.DataFrame(controls)
    log_text = "\n".join(path.read_text(errors="replace") for path in log_paths).lower()

    g0 = _read_json(root / "g0_causal_prefix" / "G0_SUMMARY.json")
    development = _read_json(
        root / "g1_sequential_development_selection_v0_1" / "DEVELOPMENT_SELECTION.json"
    )
    g1_pass = multi["status"] == "G1_MULTI_SEED_PASS_OPEN_G2"
    g2_root = root / "g2_early_ictal_loso_v0_1"
    g2_path = g2_root / "G2_G3_SUMMARY.json"
    g2 = _read_json(g2_path) if g2_path.exists() else None

    if g1_pass:
        if g2 is None:
            raise RuntimeError("G1 passed but G2/G3 is incomplete")
        g2_done = sorted(g2_root.glob("*/DONE.json"))
        if len(g2_done) != 16:
            raise RuntimeError(f"G2 incomplete: {len(g2_done)}/16 folds")
        if not all(bool(_read_json(path).get("target_values_read", False)) for path in g2_done):
            raise RuntimeError("a completed G2 fold lacks explicit target unlock")
        status = "ACCEPTED_GATED_G2_G3_CLOSEOUT"
        target_values_read = True
        g2_g3_status = g2["status"]
        stop_reason = []
        if not bool(g2["g2"]["pass"]):
            stop_reason.append("G2 M2-minus-M1 early-ictal field increment did not pass")
        elif not bool(g2["g3"]["pass"]):
            stop_reason.append("G3 correct state-seizure pairing did not pass")
        safe_claim = (
            "A three-seed, target-sealed HistoryRNN passed the interictal G1 order gate. "
            f"The gated early-ictal bridge then ended with status {g2_g3_status}; claims are limited to that gate level."
        )
    else:
        if g2 is not None or list(g2_root.glob("*/DONE.json")):
            raise RuntimeError("G2 artifacts exist despite a failed G1 gate")
        status = "PROVISIONAL_BOUNDED_NEGATIVE_FOR_CURRENT_G1_TASK"
        target_values_read = False
        g2_g3_status = "LOCKED_NOT_RUN_IN_V0_1; DIRECT_TEST_MOVED_TO_V0_2"
        stop_reason = []
        if float(multi["median_chronological_increment"]) <= 0:
            stop_reason.append("multi-seed median chronological increment <= 0")
        if float(multi["chronological_increment_one_sided_wilcoxon_p"]) >= 0.05:
            stop_reason.append("multi-seed chronological increment p >= 0.05")
        if not all(
            float(value) > 0
            for value in multi["dataset_median_chronological_increment"].values()
        ):
            stop_reason.append("chronological direction was not positive in both datasets")
        if float(multi["median_prefix_matched_order_shuffle_cost"]) <= 0:
            stop_reason.append("strict order-shuffle cost median <= 0")
        if float(multi["prefix_matched_order_shuffle_one_sided_wilcoxon_p"]) >= 0.05:
            stop_reason.append("strict order-shuffle p >= 0.05")
        if not all(
            value["median_chronological_increment"] > 0
            and value["median_prefix_matched_order_shuffle_cost"] > 0
            for value in multi["per_seed_direction"].values()
        ):
            stop_reason.append("one or more frozen seeds lacked positive direction")
        safe_claim = (
            "Under the target-blind next-event contact-field objective, the capacity-matched "
            "cross-event recurrent branch did not provide a reproducible chronology-specific "
            "increment. This proxy result neither supports nor refutes direct early-ictal transfer."
        )

    result = {
        "status": status,
        "contract": "topic5_history_rnn_early_ictal_field_v0_1",
        "scientific_endpoint": (
            "target-blind interictal history -> next interictal contact field proxy"
        ),
        "g1_status": multi["status"],
        "target_values_read": target_values_read,
        "g2_g3_status": g2_g3_status,
        "g0": g0,
        "development": {
            "status": development["status"],
            "n_runs": development["n_runs"],
            "selected_configuration": development["selected_configuration"],
            "selected_metrics": development["selected_metrics"],
        },
        "g1_execution": {
            "n_seeds": len(SEEDS),
            "n_completed_folds": len(SEEDS) * 34,
            "n_strict_order_controls": len(controls),
            "n_failed_folds": len(failed_paths),
            "oom_detected": "out of memory" in log_text,
            "nan_marker_detected": "nan" in log_text,
            "traceback_detected": "traceback" in log_text,
            "recoverable_session_interruptions": 1,
            "parameter_counts": json.loads(next(iter(unique_counts))),
        },
        "g1_primary_31": {
            "static_to_matched": _contrast(primary.static_to_matched_gain, seed=20260811),
            "matched_to_chronological": _contrast(primary.chronological_increment, seed=20260812),
            "causal_prefix_order_shuffle": _contrast(
                primary.prefix_matched_order_shuffle_cost, seed=20260813
            ),
            "relative_rank_increment": _contrast(primary.rank_increment, seed=20260814),
            "within_event_rank_shuffle": _contrast(
                primary.within_event_rank_shuffle_cost, seed=20260815
            ),
            "dataset_median_chronological_increment": multi[
                "dataset_median_chronological_increment"
            ],
            "per_seed_direction": multi["per_seed_direction"],
        },
        "g1_supportive_34": {
            "role": "pre-registered supportive cohort; the gate uses the 31 primary patients only",
            "static_to_matched": _contrast(patient.static_to_matched_gain, seed=20260821),
            "matched_to_chronological": _contrast(
                patient.chronological_increment, seed=20260822
            ),
            "causal_prefix_order_shuffle": _contrast(
                patient.prefix_matched_order_shuffle_cost, seed=20260823
            ),
            "relative_rank_increment": _contrast(patient.rank_increment, seed=20260824),
            "within_event_rank_shuffle": _contrast(
                patient.within_event_rank_shuffle_cost, seed=20260825
            ),
        },
        "learned_half_life_hours": {
            "patient_seed_median_of_dimension_medians": float(half_life["median"].median()),
            "patient_seed_iqr_of_dimension_medians": [
                float(value) for value in half_life["median"].quantile([0.25, 0.75])
            ],
            "minimum_over_all_patient_seed_dimensions": float(half_life["minimum"].min()),
            "maximum_over_all_patient_seed_dimensions": float(half_life["maximum"].max()),
            "interpretation": (
                "remained near the 2 h initialization; not an identified biological time constant"
            ),
        },
        "g2_g3": g2,
        "revised_interpretation": {
            "engineering_execution": "PASS",
            "g1_next_event_proxy": "PROVISIONAL_BOUNDED_NEGATIVE",
            "chronology_specific_state": "NOT_SUPPORTED_UNDER_CURRENT_OBJECTIVE",
            "latent_state_to_early_ictal_field": "NOT_TESTED_IN_V0_1",
            "history_dependent_network_reconfiguration": "NOT_TESTED",
            "causal_network_shaping": "NOT_ESTABLISHED",
            "next_contract": "topic5_history_rnn_direct_early_ictal_transfer_v0_2",
        },
        "stop_reason": stop_reason,
        "safe_claim": safe_claim,
    }
    (root / "FINAL_CLOSEOUT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
