#!/usr/bin/env python3
"""Patient-first summary, audit and claim adjudication for necessity v0.1."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
)
from src.topic5_shared_functional_necessity_v0_1 import (  # noqa: E402
    CONTROL_FAMILIES,
    LESION_DOSES,
    REAL_ARMS,
    dose_auc,
    holm_adjust,
)
from scripts.freeze_topic5_latent_reference_states_v0_2 import reference_dir  # noqa: E402
from scripts.run_topic5_latent_pass1_v0_2 import OUT, PARENT  # noqa: E402
from scripts.run_topic5_shared_functional_necessity_v0_1 import (  # noqa: E402
    DIRECTIONS,
    FLOAT32_REFERENCE_REPLAY_TOLERANCE,
    FUTURE_TAU,
    LESION,
    LESION_REVISION,
    NECESSITY,
    TRAIN_OPERATOR,
    TRAIN_OPERATOR_REVISION,
    direction_dir,
    lesion_dir,
    train_operator_dir,
)


SUMMARY_REVISION = "PATIENT_FIRST_PAIRWISE_COMMON_SUPPORT_R1_TARGET_FREE_PHASE_CENTER"
PAIR_METRICS = (
    "SHARED",
    "SHARED_MINUS_ORTHOGONAL",
    "SHARED_MINUS_C_SUFFIX",
    "SHARED_MINUS_PCA",
)
PRIMARY_METRICS = PAIR_METRICS[:3]
BOOTSTRAP_SAMPLES = 20000
FLOAT32_BASELINE_TOLERANCE = 2.5e-6
SUPERSEDED_V0_1 = OUT / "shared_functional_computation_necessity_v0_1"


def json_safe(value: object) -> object:
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def bootstrap_median_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if not len(data):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(BOOTSTRAP_SAMPLES, len(data)), replace=True)
    median = np.median(samples, axis=1)
    return float(np.quantile(median, 0.025)), float(np.quantile(median, 0.975))


def greater_p(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if not len(data) or np.allclose(data, 0.0):
        return 1.0
    return float(wilcoxon(data, alternative="greater", zero_method="wilcox").pvalue)


def positive_counts(values: np.ndarray) -> tuple[int, int, int]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    return int(np.count_nonzero(data > 0)), int(np.count_nonzero(data < 0)), int(np.count_nonzero(data == 0))


def state_weighted_effect(
    left: np.ndarray,
    left_valid: np.ndarray,
    right: np.ndarray | None,
    right_valid: np.ndarray | None,
    state_selector: np.ndarray,
) -> tuple[float, int, int]:
    use = np.asarray(left_valid, bool).copy()
    if right_valid is not None:
        use &= np.asarray(right_valid, bool)
    use &= np.asarray(state_selector, bool)[:, None]
    difference = np.asarray(left, float) if right is None else np.asarray(left, float) - np.asarray(right, float)
    state_count = use.sum(axis=1)
    state_mean = np.divide(
        np.where(use, difference, 0.0).sum(axis=1),
        state_count,
        out=np.full(len(use), np.nan, dtype=np.float64),
        where=state_count > 0,
    )
    finite = np.isfinite(state_mean)
    return (
        float(np.mean(state_mean[finite])) if finite.any() else float("nan"),
        int(finite.sum()),
        int(use.sum()),
    )


def extract_cell_rows(row: pd.Series) -> tuple[list[dict[str, object]], dict[str, object]]:
    with np.load(lesion_dir(row) / "lesion_response.npz", allow_pickle=False) as source:
        z = {name: np.asarray(source[name]) for name in source.files}
    family = {str(name): index for index, name in enumerate(z["family_names"].tolist())}
    delta = z["delta_nll"].astype(float)
    valid = z["valid"].astype(bool)
    phase = z["phase_target"].astype(float)
    delayed = np.asarray(FUTURE_TAU, int)
    selectors = {"ALL": np.ones(len(phase), dtype=bool)}
    for phase_target in np.sort(np.unique(phase)):
        selectors[f"{phase_target:.2f}"] = np.isclose(phase, phase_target)
    comparisons = {
        "SHARED": ("SHARED", None),
        "SHARED_MINUS_ORTHOGONAL": ("SHARED", "ORTHOGONAL"),
        "SHARED_MINUS_C_SUFFIX": ("SHARED", "C_SUFFIX"),
        "SHARED_MINUS_PCA": ("SHARED", "PCA"),
    }
    rows = []
    for dose_index, dose in enumerate(z["doses"].astype(float)):
        for phase_name, selector in selectors.items():
            for metric, (left_name, right_name) in comparisons.items():
                left_index = family[left_name]
                right_index = family[right_name] if right_name is not None else None
                left_values = np.take(delta[:, left_index, dose_index, :], delayed, axis=-1)
                left_flags = np.take(valid[:, left_index, dose_index, :], delayed, axis=-1)
                right_values = None if right_index is None else np.take(
                    delta[:, right_index, dose_index, :], delayed, axis=-1
                )
                right_flags = None if right_index is None else np.take(
                    valid[:, right_index, dose_index, :], delayed, axis=-1
                )
                effect, n_states, n_decisions = state_weighted_effect(
                    left_values,
                    left_flags,
                    right_values,
                    right_flags,
                    selector,
                )
                rows.append({
                    "patient": str(row.patient),
                    "fit_id": str(row.fit_id),
                    "public_arm": str(row.public_arm),
                    "seed": int(row.seed),
                    "phase": phase_name,
                    "dose": float(dose),
                    "metric": metric,
                    "effect_nll_per_decision": effect,
                    "n_reference_states": n_states,
                    "n_delayed_decisions": n_decisions,
                })
    # The baseline branch is identical by construction.  Check it only where two
    # families are both valid, so NaN padding never enters the audit.
    baseline = z["base_nll"].astype(float)
    selected_valid = np.take(valid, delayed, axis=-1)
    selected_baseline = np.take(baseline, delayed, axis=-1)
    common = selected_valid.all(axis=1)
    max_baseline_difference = 0.0
    for family_index in range(1, len(CONTROL_FAMILIES)):
        use = common
        difference = np.abs(
            selected_baseline[:, 0] - selected_baseline[:, family_index]
        )
        if use.any():
            max_baseline_difference = max(max_baseline_difference, float(np.nanmax(difference[use])))
    direction_path = direction_dir(str(row.fit_id), str(row.public_arm)) / "direction_contract.npz"
    done = json.loads((lesion_dir(row) / "DONE.json").read_text())
    with np.load(PARENT / "cache" / str(row.fit_id) / "events.npz", allow_pickle=False) as source:
        split = np.asarray(source["split"])
    test_split_ok = bool(np.all(split[z["event_index"].astype(int)] == 2))
    audit = {
        "fit_id": str(row.fit_id),
        "public_arm": str(row.public_arm),
        "seed": int(row.seed),
        "max_baseline_difference": max_baseline_difference,
        "max_displacement_norm_error": float(np.nanmax(np.abs(
            z["actual_displacement_norm"] - z["displacement_norm"][:, None, :]
        ))),
        "valid_values_finite": bool(np.isfinite(delta[valid]).all()),
        "direction_hash_matches": done["direction_contract_sha256"] == sha256_file(direction_path),
        "heldout_reference_split_is_test": test_split_ok,
        "has_primary_common_support": bool(any(
            item["phase"] == "ALL" and item["metric"] in PRIMARY_METRICS
            and item["n_delayed_decisions"] > 0 for item in rows
        )),
    }
    return rows, audit


def aggregate_levels(cell: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["phase", "dose", "metric"]
    fit_arm = (
        cell.groupby(["patient", "fit_id", "public_arm", *keys], as_index=False)
        .agg(
            effect_nll_per_decision=("effect_nll_per_decision", "mean"),
            n_seeds=("seed", "nunique"),
            n_reference_states=("n_reference_states", "sum"),
            n_delayed_decisions=("n_delayed_decisions", "sum"),
        )
    )
    fit = (
        fit_arm.groupby(["patient", "fit_id", *keys], as_index=False)
        .agg(
            effect_nll_per_decision=("effect_nll_per_decision", "mean"),
            n_arms=("public_arm", "nunique"),
            n_reference_states=("n_reference_states", "sum"),
            n_delayed_decisions=("n_delayed_decisions", "sum"),
        )
    )
    patient = (
        fit.groupby(["patient", *keys], as_index=False)
        .agg(
            effect_nll_per_decision=("effect_nll_per_decision", "mean"),
            n_fits=("fit_id", "nunique"),
            n_reference_states=("n_reference_states", "sum"),
            n_delayed_decisions=("n_delayed_decisions", "sum"),
        )
    )
    return fit_arm, fit, patient


def patient_auc_table(patient: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subject, phase, metric), group in patient.groupby(["patient", "phase", "metric"], sort=True):
        values = group.set_index("dose")["effect_nll_per_decision"]
        effects = np.asarray([values.get(float(dose), np.nan) for dose in LESION_DOSES], float)
        rows.append({
            "patient": subject,
            "phase": phase,
            "metric": metric,
            "dose_auc_nll": dose_auc(LESION_DOSES, effects),
            **{f"effect_dose_{dose:.2f}": value for dose, value in zip(LESION_DOSES, effects)},
        })
    return pd.DataFrame(rows)


def arm_auc_table(fit_arm: pd.DataFrame) -> pd.DataFrame:
    patient_arm = (
        fit_arm.groupby(["patient", "public_arm", "phase", "dose", "metric"], as_index=False)
        .agg(effect_nll_per_decision=("effect_nll_per_decision", "mean"))
    )
    rows = []
    for keys, group in patient_arm.groupby(["patient", "public_arm", "phase", "metric"], sort=True):
        values = group.set_index("dose")["effect_nll_per_decision"]
        effects = np.asarray([values.get(float(dose), np.nan) for dose in LESION_DOSES], float)
        rows.append({
            "patient": keys[0], "public_arm": keys[1], "phase": keys[2], "metric": keys[3],
            "dose_auc_nll": dose_auc(LESION_DOSES, effects),
        })
    return pd.DataFrame(rows)


def inference_table(auc: pd.DataFrame, phase: str = "ALL") -> pd.DataFrame:
    rows = []
    for metric in PAIR_METRICS:
        values = auc[(auc.phase == phase) & (auc.metric == metric)]["dose_auc_nll"].to_numpy(float)
        values = values[np.isfinite(values)]
        low, high = bootstrap_median_ci(values, seed=1701 + PAIR_METRICS.index(metric))
        positive, negative, zero = positive_counts(values)
        rows.append({
            "metric": metric,
            "n_patients": int(len(values)),
            "median_dose_auc_nll": float(np.median(values)) if len(values) else float("nan"),
            "ci95_low": low,
            "ci95_high": high,
            "positive": positive,
            "negative": negative,
            "zero": zero,
            "p_greater": greater_p(values),
        })
    frame = pd.DataFrame(rows)
    primary_index = [frame.index[frame.metric.eq(metric)][0] for metric in PRIMARY_METRICS]
    frame["p_holm_primary"] = np.nan
    frame.loc[primary_index, "p_holm_primary"] = holm_adjust(frame.loc[primary_index, "p_greater"])
    return frame


def arm_summary(arm_auc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    use = arm_auc[arm_auc.phase.eq("ALL")]
    for arm in REAL_ARMS:
        for metric in PRIMARY_METRICS:
            values = use[(use.public_arm == arm) & (use.metric == metric)]["dose_auc_nll"].to_numpy(float)
            values = values[np.isfinite(values)]
            low, high = bootstrap_median_ci(values, seed=2710 + REAL_ARMS.index(arm) * 10 + PRIMARY_METRICS.index(metric))
            positive, negative, zero = positive_counts(values)
            rows.append({
                "public_arm": arm,
                "metric": metric,
                "n_patients": int(len(values)),
                "median_dose_auc_nll": float(np.median(values)) if len(values) else float("nan"),
                "ci95_low": low,
                "ci95_high": high,
                "positive": positive,
                "negative": negative,
                "zero": zero,
                "p_greater": greater_p(values),
            })
    return pd.DataFrame(rows)


def train_split_audit(manifest: pd.DataFrame) -> tuple[bool, list[str]]:
    failures = []
    for item in manifest.itertuples(index=False):
        row = pd.Series(item._asdict())
        with np.load(train_operator_dir(row) / "train_operator.npz", allow_pickle=False) as source:
            event = np.asarray(source["train_reference_event_index"], int)
        with np.load(PARENT / "cache" / str(row.fit_id) / "events.npz", allow_pickle=False) as source:
            split = np.asarray(source["split"])
        if not np.all(split[event] == 0):
            failures.append(f"{row.fit_id}/{row.public_arm}/seed{row.seed}")
    return not failures, failures[:20]


def write_p0_repair_provenance() -> dict[str, object]:
    similarities: list[float] = []
    for fit_dir in sorted((DIRECTIONS / "per_fit").glob("*")):
        for heldout_dir in sorted(fit_dir.glob("*")):
            new_path = heldout_dir / "direction_contract.npz"
            old_path = (
                SUPERSEDED_V0_1 / "direction_freeze" / "per_fit"
                / fit_dir.name / heldout_dir.name / "direction_contract.npz"
            )
            if not new_path.is_file() or not old_path.is_file():
                continue
            with np.load(new_path, allow_pickle=False) as source:
                new = np.asarray(source["shared_hidden_components"], float)[0]
            with np.load(old_path, allow_pickle=False) as source:
                old = np.asarray(source["shared_hidden_components"], float)[0]
            similarities.append(float(abs(np.dot(new, old)) / (np.linalg.norm(new) * np.linalg.norm(old))))
    values = np.asarray(similarities, float)
    payload = {
        "contract": "topic5_shared_necessity_p0_repair_provenance_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "superseded_result_root": str(SUPERSEDED_V0_1.relative_to(ROOT)),
        "final_result_root": str(NECESSITY.relative_to(ROOT)),
        "p0": (
            "v0.1 used a completed-heldout-event future-field coordinate in the lesion "
            "centre and support gate; v0.2 uses the train-fitted phase curve only."
        ),
        "v0_1_numbers_eligible_for_scientific_reporting": False,
        "direction_comparisons": int(len(values)),
        "first_component_absolute_cosine": {
            "median": float(np.median(values)) if len(values) else None,
            "mean": float(np.mean(values)) if len(values) else None,
            "minimum": float(np.min(values)) if len(values) else None,
        },
        "interpretation": (
            "The repair changed the lesion centre and empirical support, not the "
            "leave-one-network shared direction definition."
        ),
    }
    atomic_write_json(NECESSITY / "P0_REPAIR_PROVENANCE.json", payload)
    return payload


def main() -> None:
    manifest = pd.read_csv(OUT / "CHECKPOINT_MANIFEST.csv")
    real_manifest = manifest[manifest.public_arm.isin(REAL_ARMS)].copy()
    cell_rows, cell_audits = [], []
    for _, row in real_manifest.iterrows():
        rows, audit = extract_cell_rows(row)
        cell_rows.extend(rows)
        cell_audits.append(audit)
    cell = pd.DataFrame(cell_rows)
    audit_cells = pd.DataFrame(cell_audits)
    fit_arm, fit, patient = aggregate_levels(cell)
    auc = patient_auc_table(patient)
    arm_auc = arm_auc_table(fit_arm)
    inference = inference_table(auc)
    by_arm = arm_summary(arm_auc)
    atomic_write_csv(NECESSITY / "CELL_PAIR_DOSE_EFFECTS.csv", cell)
    atomic_write_csv(NECESSITY / "FIT_ARM_PAIR_DOSE_EFFECTS.csv", fit_arm)
    atomic_write_csv(NECESSITY / "FIT_PAIR_DOSE_EFFECTS.csv", fit)
    atomic_write_csv(NECESSITY / "PATIENT_PAIR_DOSE_EFFECTS.csv", patient)
    atomic_write_csv(NECESSITY / "PATIENT_AUC_EFFECTS.csv", auc)
    atomic_write_csv(NECESSITY / "PATIENT_ARM_AUC_EFFECTS.csv", arm_auc)
    atomic_write_csv(NECESSITY / "PRIMARY_INFERENCE.csv", inference)
    atomic_write_csv(NECESSITY / "HELDOUT_ARM_INFERENCE.csv", by_arm)
    atomic_write_csv(NECESSITY / "LESION_CELL_AUDIT.csv", audit_cells)

    train_status = json.loads((TRAIN_OPERATOR / "TRAIN_OPERATOR_STATUS.json").read_text())
    direction_status = json.loads((DIRECTIONS / "DIRECTION_FREEZE_STATUS.json").read_text())
    direction_seal = json.loads((DIRECTIONS / "DIRECTION_FREEZE_SEAL.json").read_text())
    lesion_status = json.loads((LESION / "LESION_EXECUTION_STATUS.json").read_text())
    lesion_summary = pd.read_csv(LESION / "LESION_CELL_SUMMARY.csv")
    direction_summary = pd.read_csv(DIRECTIONS / "DIRECTION_SUMMARY.csv")
    train_split_ok, train_split_failures = train_split_audit(manifest)
    p0_provenance = write_p0_repair_provenance()
    primary = inference.set_index("metric").loc[list(PRIMARY_METRICS)]
    primary_stats_pass = bool(
        (primary["median_dose_auc_nll"] > 0).all()
        and (primary["p_holm_primary"] < 0.05).all()
    )
    arm_pivot = by_arm.pivot(index="public_arm", columns="metric", values="median_dose_auc_nll")
    arm_consistent = (
        (arm_pivot[list(PRIMARY_METRICS)] > 0).all(axis=1)
    )
    n_consistent_arms = int(arm_consistent.sum())
    supported = bool(primary_stats_pass and n_consistent_arms >= 3)
    if supported:
        verdict = "SUPPORTED"
        allowed_claim = (
            "Different frozen recurrent connectivity designs depend on a cross-topology "
            "functional component for heldout future-contact prediction."
        )
    elif (
        primary.loc["SHARED", "median_dose_auc_nll"] > 0
        and primary.loc["SHARED", "p_holm_primary"] < 0.05
        and primary.loc["SHARED_MINUS_ORTHOGONAL", "median_dose_auc_nll"] > 0
        and primary.loc["SHARED_MINUS_ORTHOGONAL", "p_holm_primary"] < 0.05
    ):
        verdict = "SHARED_ARCHITECTURE_TASK_SENSITIVITY_ORDER_SPECIFICITY_UNSUPPORTED"
        allowed_claim = (
            "The shared component is more functionally important than matched orthogonal "
            "directions, but the shuffled-ending control does not establish order specificity."
        )
    elif primary.loc["SHARED", "median_dose_auc_nll"] > 0 and primary.loc["SHARED", "p_holm_primary"] < 0.05:
        verdict = "GENERIC_PERTURBATION_SENSITIVITY"
        allowed_claim = "Deleting the component worsens prediction, but not selectively versus matched controls."
    else:
        verdict = "NECESSITY_UNSUPPORTED"
        allowed_claim = (
            "Different connectivity designs produce convergent finite-time perturbation "
            "responses aligned with heldout interictal contact-following structure; removing "
            "the target-free leave-one-network shared component did not selectively impair prediction."
        )
    claim = {
        "contract": "topic5_shared_functional_computation_necessity_claim_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "revision": SUMMARY_REVISION,
        "verdict": verdict,
        "figure6_main_panel_eligible": supported,
        "primary_stats_pass": primary_stats_pass,
        "consistent_heldout_arms": n_consistent_arms,
        "required_consistent_heldout_arms": 3,
        "primary_tests": primary.reset_index().to_dict(orient="records"),
        "heldout_arm_direction_consistency": {str(arm): bool(value) for arm, value in arm_consistent.items()},
        "allowed_claim": allowed_claim,
        "forbidden_claims": [
            "unique recurrent topology identified",
            "anatomical pathway identified",
            "epilepsy-specific mechanism established",
            "cross-state or SNN convergence established by this experiment",
            "a low-rank shared component is necessary for future-contact prediction",
        ],
    }
    atomic_write_json(NECESSITY / "CLAIM_ADJUDICATION.json", claim)

    checks = {
        "train_operator_630_pass": train_status.get("status") == "PASS" and train_status.get("completed_cells") == 630,
        "train_operator_target_free_revision": train_status.get("revision") == TRAIN_OPERATOR_REVISION,
        "direction_168_sealed": direction_status.get("status") == "PASS" and direction_status.get("directions") == 168 and direction_seal.get("sealed") is True,
        "lesion_504_pass": lesion_status.get("status") == "PASS" and lesion_status.get("completed_cells") == 504,
        "lesion_target_free_revision": (
            lesion_status.get("revision") == LESION_REVISION
            and bool((lesion_summary.revision == LESION_REVISION).all())
        ),
        "target_free_center_contract": (
            lesion_status.get("state_center_definition") == "TRAIN_FITTED_PHASE_CURVE_GAMMA"
            and lesion_status.get("heldout_future_field_used_in_state_center") is False
            and lesion_status.get("heldout_future_field_used_in_support_gate") is False
            and lesion_status.get("heldout_outcome_keys_dropped_before_lesion") is True
        ),
        "reference_replay_within_float32_tolerance": bool(
            lesion_summary.max_reference_replay_error.max()
            <= FLOAT32_REFERENCE_REPLAY_TOLERANCE
        ),
        "train_references_all_axis_train": train_split_ok,
        "heldout_references_all_test": bool(audit_cells.heldout_reference_split_is_test.all()),
        "heldout_arm_never_read_for_direction": bool((~direction_summary.heldout_arm_operator_read.astype(bool)).all()),
        "direction_hashes_match": bool(audit_cells.direction_hash_matches.all()),
        "valid_nll_values_finite": bool(audit_cells.valid_values_finite.all()),
        "baseline_identical_across_branches": bool(
            audit_cells.max_baseline_difference.max() <= FLOAT32_BASELINE_TOLERANCE
        ),
        "control_displacement_norm_exact": bool(audit_cells.max_displacement_norm_error.max() <= 1e-6),
        "all_cells_have_primary_support": bool(audit_cells.has_primary_common_support.all()),
        "model_hashes_504_unchanged": int(lesion_summary.model_hash_unchanged.sum()) == 504,
        "decoder_hashes_504_unchanged": int(lesion_summary.decoder_hash_unchanged.sum()) == 504,
        "patient_first_denominator_28": int(auc[auc.phase.eq("ALL")].patient.nunique()) == 28,
        "all_four_heldout_arms_present": set(arm_auc.public_arm.unique()) == set(REAL_ARMS),
        "p0_direction_comparison_168": p0_provenance["direction_comparisons"] == 168,
    }
    audit = {
        "contract": "topic5_shared_functional_computation_necessity_audit_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "revision": SUMMARY_REVISION,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "train_split_failures_first20": train_split_failures,
        "max_baseline_difference": float(audit_cells.max_baseline_difference.max()),
        "float32_baseline_tolerance": FLOAT32_BASELINE_TOLERANCE,
        "max_control_displacement_norm_error": float(audit_cells.max_displacement_norm_error.max()),
        "max_reference_replay_error": float(lesion_summary.max_reference_replay_error.max()),
        "reference_replay_tolerance": FLOAT32_REFERENCE_REPLAY_TOLERANCE,
        "n_patients": int(auc[auc.phase.eq("ALL")].patient.nunique()),
        "n_fits": int(real_manifest.fit_id.nunique()),
        "n_real_cells": int(len(real_manifest)),
        "train_operator_status_sha256": sha256_file(TRAIN_OPERATOR / "TRAIN_OPERATOR_STATUS.json"),
        "direction_seal_sha256": sha256_file(DIRECTIONS / "DIRECTION_FREEZE_SEAL.json"),
        "lesion_status_sha256": sha256_file(LESION / "LESION_EXECUTION_STATUS.json"),
        "claim_sha256": sha256_file(NECESSITY / "CLAIM_ADJUDICATION.json"),
        "p0_repair_provenance_sha256": sha256_file(NECESSITY / "P0_REPAIR_PROVENANCE.json"),
    }
    atomic_write_json(NECESSITY / "FINAL_AUDIT.json", audit)
    summary = {
        "contract": "topic5_shared_functional_computation_necessity_summary_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "revision": SUMMARY_REVISION,
        "status": "PASS" if audit["status"] == "PASS" else "AUDIT_FAILED",
        "claim_verdict": verdict,
        "figure6_main_panel_eligible": supported,
        "cohort": {"patients": 28, "fits": 42, "real_cells": 504, "direction_cells": 168},
        "execution": {
            "train_operator_cells": 630,
            "lesion_cells": 504,
            "valid_delayed_decisions": int(lesion_status["delayed_valid_decisions"]),
        },
        "primary_inference": json_safe(inference.to_dict(orient="records")),
        "heldout_arm_inference": json_safe(by_arm.to_dict(orient="records")),
        "allowed_claim": allowed_claim,
    }
    atomic_write_json(NECESSITY / "NECESSITY_SUMMARY.json", summary)
    print(json.dumps({"audit": audit, "claim": claim, "summary": summary}, indent=2))
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
