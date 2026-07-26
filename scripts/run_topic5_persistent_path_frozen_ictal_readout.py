#!/usr/bin/env python3
"""Run the conditional frozen interictal-to-clinical-onset static readout."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_topic5_persistent_path_formal import benjamini_hochberg
from scripts.run_topic5_clinical_onset_gradient_field_cohort_stat import (
    load_phenotype_selector_map,
)


FEATURE_COLUMNS = [
    "nonparticipation_probability",
    *[f"joint_rank_bin_{index}" for index in range(10)],
]
CONDITIONS = (
    "empirical_train80",
    "intact",
    "no_history",
    "graph_lesion",
    "mode_collapse_lesion",
)
BB150_CACHE = (
    ROOT / "results/topic5_ictal_recruitment/t0_feature_cache_bb150_1_150"
)


def _stable_seed(subject: str, seed: int) -> int:
    return int(
        hashlib.sha256(f"{subject}:{seed}:all-contact".encode()).hexdigest()[:8],
        16,
    )


def _within_patient_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError("constant clinical-onset contact field")
    return (values - center) / scale


def _load_targets(feature_subjects: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    selectors = load_phenotype_selector_map()
    contact_rows = []
    inventory = []
    for subject in sorted(feature_subjects):
        dataset = subject.split("_", 1)[0]
        if dataset != "epilepsiae":
            inventory.append(
                {
                    "subject": subject,
                    "status": "excluded",
                    "reason": "clinical_onset_unavailable",
                }
            )
            continue
        meta_path = BB150_CACHE / f"{subject}.json"
        npz_path = BB150_CACHE / f"{subject}.npz"
        if not meta_path.exists() or not npz_path.exists():
            inventory.append(
                {
                    "subject": subject,
                    "status": "excluded",
                    "reason": "missing_bb150_cache",
                }
            )
            continue
        meta = json.loads(meta_path.read_text())
        if not meta.get("line_noise_masked_1_150", False):
            inventory.append(
                {
                    "subject": subject,
                    "status": "excluded",
                    "reason": "bb150_contract_mismatch",
                }
            )
            continue
        eligible = set(int(value) for value in meta.get("eligible_idxs", []))
        strict = set(
            selectors.get(subject, {}).get("strict_broadband", set())
        )
        seizure_indices = sorted(eligible & strict)
        if not seizure_indices:
            inventory.append(
                {
                    "subject": subject,
                    "status": "excluded",
                    "reason": "no_strict_clinical_bb150_seizure",
                }
            )
            continue
        with np.load(npz_path, allow_pickle=False) as cache:
            names = np.asarray(meta.get("channels", cache["channels"])).astype(
                str
            )
            fields = []
            used = []
            for seizure_index in seizure_indices:
                key = f"bb150_auc__{seizure_index}"
                if key not in cache.files:
                    continue
                values = np.asarray(cache[key], float)
                if values.shape != (len(names),):
                    raise RuntimeError(
                        f"{subject}:{key}: target/contact shape mismatch"
                    )
                fields.append(values)
                used.append(seizure_index)
        if not fields:
            inventory.append(
                {
                    "subject": subject,
                    "status": "excluded",
                    "reason": "strict_bb150_arrays_missing",
                }
            )
            continue
        median_field = np.nanmedian(np.row_stack(fields), axis=0)
        finite = np.isfinite(median_field)
        if int(finite.sum()) < 6:
            inventory.append(
                {
                    "subject": subject,
                    "status": "excluded",
                    "reason": "fewer_than_6_finite_target_contacts",
                }
            )
            continue
        for name, raw in zip(names[finite], median_field[finite]):
            contact_rows.append(
                {
                    "subject": subject,
                    "dataset": dataset,
                    "contact_name": str(name),
                    "clinical_bb150_raw": float(raw),
                    "n_seizures": int(len(used)),
                    "seizure_indices": ";".join(str(value) for value in used),
                    "time_reference": "clinical_onset",
                    "window_start_sec": 0.0,
                    "window_end_sec": 10.0,
                    "band_hz": "1-150",
                }
            )
        inventory.append(
            {
                "subject": subject,
                "status": "eligible",
                "reason": "",
                "n_seizures": int(len(used)),
                "n_finite_target_contacts": int(finite.sum()),
            }
        )
    return pd.DataFrame(contact_rows), pd.DataFrame(inventory)


def _collapse_feature_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    empirical = frame[frame.condition.eq("empirical_train80")].copy()
    generated = frame[~frame.condition.eq("empirical_train80")].copy()
    keys = ["subject", "dataset", "condition", "contact_name"]
    generated = generated.groupby(keys, as_index=False)[FEATURE_COLUMNS].median()
    empirical = empirical[keys + FEATURE_COLUMNS]
    return pd.concat([empirical, generated], ignore_index=True)


def _weighted_standardize(
    values: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.average(values, axis=0, weights=weights)
    variance = np.average((values - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def _all_contact_null(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    subject: str,
    n_perm: int,
    seed: int,
) -> np.ndarray:
    prediction_rank = rankdata(np.asarray(prediction, float))
    target_rank = rankdata(np.asarray(target, float))
    prediction_rank -= prediction_rank.mean()
    target_rank -= target_rank.mean()
    denominator = float(
        np.sqrt(np.sum(prediction_rank**2) * np.sum(target_rank**2))
    )
    if denominator <= 0:
        return np.full(int(n_perm), np.nan)
    rng = np.random.default_rng(_stable_seed(subject, seed))
    permutations = np.argsort(
        rng.random((int(n_perm), len(target_rank))), axis=1
    )
    return (target_rank[permutations] @ prediction_rank) / denominator


def _fit_condition_loso(
    frame: pd.DataFrame,
    *,
    condition: str,
    n_perm: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = frame[frame.condition.eq(condition)].copy()
    subjects = sorted(current.subject.unique())
    prediction_rows = []
    subject_rows = []
    for heldout in subjects:
        train = current[~current.subject.eq(heldout)].copy()
        test = current[current.subject.eq(heldout)].copy()
        counts = train.groupby("subject").size().to_dict()
        weights = train.subject.map(lambda value: 1.0 / counts[value]).to_numpy(
            float
        )
        x_train = train[FEATURE_COLUMNS].to_numpy(float)
        x_test = test[FEATURE_COLUMNS].to_numpy(float)
        mean, scale = _weighted_standardize(x_train, weights)
        model = Ridge(alpha=1.0, fit_intercept=True)
        model.fit(
            (x_train - mean) / scale,
            train.clinical_bb150_scaled.to_numpy(float),
            sample_weight=weights,
        )
        prediction = model.predict((x_test - mean) / scale)
        target = test.clinical_bb150_scaled.to_numpy(float)
        rho = float(spearmanr(prediction, target).statistic)
        null = _all_contact_null(
            prediction,
            target,
            subject=heldout,
            n_perm=int(n_perm),
            seed=int(seed),
        )
        if not np.isfinite(rho) or not np.isfinite(null).all():
            raise RuntimeError(f"{heldout}:{condition}: nonfinite readout")
        null_median = float(np.median(null))
        subject_rows.append(
            {
                "subject": heldout,
                "condition": condition,
                "n_contacts": int(len(test)),
                "n_train_subjects": int(len(subjects) - 1),
                "rho_data": rho,
                "rho_channel_shuffle_median": null_median,
                "rho_channel_shuffle_p95": float(np.percentile(null, 95)),
                "margin": rho - null_median,
                "subject_empirical_one_sided_p": float(
                    (1 + np.sum(null >= rho - 1e-15)) / (len(null) + 1)
                ),
            }
        )
        for row, predicted in zip(test.itertuples(index=False), prediction):
            prediction_rows.append(
                {
                    "subject": heldout,
                    "condition": condition,
                    "contact_name": row.contact_name,
                    "clinical_bb150_raw": row.clinical_bb150_raw,
                    "clinical_bb150_scaled": row.clinical_bb150_scaled,
                    "predicted_bb150_scaled": float(predicted),
                }
            )
    return pd.DataFrame(prediction_rows), pd.DataFrame(subject_rows)


def _directional_wilcoxon(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    if not len(values) or np.allclose(values, 0.0):
        return 1.0
    return float(
        wilcoxon(values, alternative="greater", zero_method="wilcox").pvalue
    )


def _cohort_statistics(subjects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, frame in subjects.groupby("condition"):
        margin = frame.margin.to_numpy(float)
        rows.append(
            {
                "condition": condition,
                "n_patients": int(len(frame)),
                "median_rho": float(np.median(frame.rho_data)),
                "median_channel_shuffle": float(
                    np.median(frame.rho_channel_shuffle_median)
                ),
                "median_margin": float(np.median(margin)),
                "n_positive_margin": int(np.sum(margin > 0)),
                "wilcoxon_p_margin_greater_zero": _directional_wilcoxon(
                    margin
                ),
            }
        )
    return pd.DataFrame(rows)


def _condition_comparisons(subjects: pd.DataFrame) -> pd.DataFrame:
    intact = subjects[subjects.condition.eq("intact")].set_index("subject")
    rows = []
    for condition in (
        "no_history",
        "graph_lesion",
        "mode_collapse_lesion",
        "empirical_train80",
    ):
        other = subjects[subjects.condition.eq(condition)].set_index("subject")
        left, right = intact.rho_data.align(other.rho_data, join="inner")
        benefit = left - right
        rows.append(
            {
                "comparison": f"intact_minus_{condition}",
                "n_patients": int(len(benefit)),
                "median_rho_benefit": float(np.median(benefit)),
                "n_intact_better": int(np.sum(benefit > 0)),
                "wilcoxon_p_greater": _directional_wilcoxon(
                    benefit.to_numpy(float)
                ),
            }
        )
    frame = pd.DataFrame(rows)
    inferential = ~frame.comparison.eq("intact_minus_empirical_train80")
    frame["wilcoxon_q_bh"] = np.nan
    frame.loc[inferential, "wilcoxon_q_bh"] = benjamini_hochberg(
        frame.loc[inferential, "wilcoxon_p_greater"].to_numpy(float)
    )
    return frame


def _formal_readout_gate(
    cohort: pd.DataFrame, comparisons: pd.DataFrame
) -> dict:
    intact = cohort[cohort.condition.eq("intact")].iloc[0]
    no_history = comparisons[
        comparisons.comparison.eq("intact_minus_no_history")
    ].iloc[0]
    lesion_checks = {}
    for lesion in ("graph_lesion", "mode_collapse_lesion"):
        row = comparisons[
            comparisons.comparison.eq(f"intact_minus_{lesion}")
        ].iloc[0]
        lesion_checks[lesion] = bool(
            row.median_rho_benefit > 0
            and row.n_intact_better > row.n_patients / 2
            and row.wilcoxon_q_bh < 0.05
        )
    checks = {
        "n_patients_at_least_8": int(intact.n_patients) >= 8,
        "median_channel_shuffle_margin_positive": float(
            intact.median_margin
        )
        > 0,
        "majority_positive_channel_shuffle_margin": int(
            intact.n_positive_margin
        )
        > int(intact.n_patients) / 2,
        "channel_shuffle_patient_wilcoxon_p_below_0p05": float(
            intact.wilcoxon_p_margin_greater_zero
        )
        < 0.05,
        "intact_outperforms_no_history": bool(
            no_history.median_rho_benefit > 0
            and no_history.n_intact_better > no_history.n_patients / 2
            and no_history.wilcoxon_q_bh < 0.05
        ),
        "graph_or_mode_structure_required": any(lesion_checks.values()),
    }
    return {
        "cross_state_positive": bool(all(checks.values())),
        "checks": checks,
        "lesion_checks": lesion_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()

    root = args.root.resolve()
    analysis = root / "analysis"
    gate_path = analysis / "formal_gate_summary.json"
    if not gate_path.exists():
        raise RuntimeError("formal interictal analysis is incomplete")
    interictal_gate = json.loads(gate_path.read_text())
    if interictal_gate.get("formal_interictal_gate_pass") is not True:
        raise RuntimeError(
            "interictal gate failed; clinical-onset target remains sealed"
        )
    feature_path = analysis / "cross_state_frozen_node_rank_features.csv"
    if not feature_path.exists():
        raise RuntimeError("frozen cross-state features are missing")

    # No ictal file is listed or opened before the formal gate checks above.
    features = _collapse_feature_seeds(pd.read_csv(feature_path))
    targets, inventory = _load_targets(set(features.subject))
    inventory.to_csv(analysis / "ictal_readout_target_inventory.csv", index=False)
    if targets.empty:
        raise RuntimeError("no clinical-onset BB150 targets are eligible")
    merged = features.merge(
        targets,
        on=["subject", "dataset", "contact_name"],
        how="inner",
        validate="many_to_one",
    )
    contact_counts = (
        merged.groupby(["subject", "condition"])
        .size()
        .unstack("condition")
    )
    complete = contact_counts.reindex(columns=CONDITIONS)
    eligible_subjects = sorted(
        complete.index[
            complete.notna().all(axis=1) & complete.min(axis=1).ge(6)
        ].astype(str)
    )
    merged = merged[merged.subject.isin(eligible_subjects)].copy()
    if len(eligible_subjects) < 8:
        raise RuntimeError(
            f"fewer than 8 clinical-onset readout patients: {len(eligible_subjects)}"
        )
    common_targets = merged[
        ["subject", "contact_name", "clinical_bb150_raw"]
    ].drop_duplicates()
    if common_targets.duplicated(["subject", "contact_name"]).any():
        raise RuntimeError("clinical target contact names are not unique")
    common_targets["clinical_bb150_scaled"] = (
        common_targets.groupby("subject", group_keys=False)[
            "clinical_bb150_raw"
        ].transform(lambda values: _within_patient_scale(values.to_numpy(float)))
    )
    merged = merged.drop(columns=["clinical_bb150_scaled"], errors="ignore").merge(
        common_targets,
        on=["subject", "contact_name", "clinical_bb150_raw"],
        how="left",
        validate="many_to_one",
    )
    merged.to_csv(
        analysis / "ictal_readout_aligned_contact_fields.csv", index=False
    )

    prediction_frames = []
    subject_frames = []
    for condition in CONDITIONS:
        prediction, subject = _fit_condition_loso(
            merged,
            condition=condition,
            n_perm=int(args.n_perm),
            seed=int(args.seed),
        )
        prediction_frames.append(prediction)
        subject_frames.append(subject)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    subjects = pd.concat(subject_frames, ignore_index=True)
    cohort = _cohort_statistics(subjects)
    comparisons = _condition_comparisons(subjects)
    verdict = _formal_readout_gate(cohort, comparisons)

    predictions.to_csv(
        analysis / "ictal_readout_contact_predictions.csv", index=False
    )
    subjects.to_csv(
        analysis / "ictal_readout_patient_statistics.csv", index=False
    )
    cohort.to_csv(
        analysis / "ictal_readout_cohort_statistics.csv", index=False
    )
    comparisons.to_csv(
        analysis / "ictal_readout_condition_comparisons.csv", index=False
    )
    summary = {
        "status": "complete",
        "contract": "topic5_rnn_frozen_ictal_static_readout_v1_0",
        "interictal_gate_pass": True,
        "n_patients": int(len(eligible_subjects)),
        "patients": eligible_subjects,
        "target": {
            "time_reference": "clinical_onset",
            "window_sec": [0.0, 10.0],
            "band_hz": [1.0, 150.0],
            "aggregation": "contact-wise seizure median within patient",
        },
        "readout": {
            "outer_split": "leave_one_patient_out",
            "ridge_alpha": 1.0,
            "patient_weighting": "equal_total_weight_per_training_patient",
            "n_channel_shuffle_draws": int(args.n_perm),
        },
        **verdict,
        "ictal_target_read": True,
    }
    (analysis / "ictal_readout_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
