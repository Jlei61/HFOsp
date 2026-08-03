#!/usr/bin/env python3
"""Patient-first formal summary for Topic 5 field refinement v0.4."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = [
    "M0_STATIC_AB",
    "M1_FROZEN_HISTORY_HEAD",
    "M2_TIME_AWARE_NONRECURRENT",
    "M3_JOINT_RNN",
]
MODEL_COLUMNS = {
    "M0_STATIC_AB": "m0_static_ab",
    "M1_FROZEN_HISTORY_HEAD": "m1_frozen_history_head",
    "M2_TIME_AWARE_NONRECURRENT": "m2_time_aware_nonrecurrent",
    "M3_JOINT_RNN": "m3_joint_rnn",
}
TIE_TOLERANCE = 1e-9


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16) % (2**32)


def _standard_rank(values: np.ndarray) -> np.ndarray | None:
    values = np.asarray(values, dtype=float)
    if len(values) < 3 or not np.all(np.isfinite(values)):
        return None
    ranked = rankdata(values, method="average")
    ranked -= ranked.mean()
    norm = np.linalg.norm(ranked)
    if norm <= 0:
        return None
    return ranked / norm


def _maxab(candidate_a: np.ndarray, candidate_b: np.ndarray, target: np.ndarray) -> float:
    target_rank = _standard_rank(target)
    a_rank = _standard_rank(candidate_a)
    b_rank = _standard_rank(candidate_b)
    if target_rank is None or a_rank is None or b_rank is None:
        return float("nan")
    return float(max(abs(a_rank @ target_rank), abs(b_rank @ target_rank)))


def _exact_signed_rank(differences: np.ndarray, tolerance: float = TIE_TOLERANCE) -> dict:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    positive = int(np.sum(values > tolerance))
    negative = int(np.sum(values < -tolerance))
    ties = int(len(values) - positive - negative)
    nonzero = values[np.abs(values) > tolerance]
    if len(nonzero) == 0:
        return {
            "p_two_sided_exact": 1.0,
            "n_positive": positive,
            "n_negative": negative,
            "n_tie": ties,
            "n_nonzero": 0,
        }
    ranks = rankdata(np.abs(nonzero), method="average")
    observed = float(np.sum(np.sign(nonzero) * ranks))
    extreme = 0
    total = 2 ** len(nonzero)
    threshold = abs(observed) - 1e-12
    for bits in itertools.product((-1.0, 1.0), repeat=len(nonzero)):
        statistic = float(np.dot(np.asarray(bits), ranks))
        extreme += abs(statistic) >= threshold
    return {
        "p_two_sided_exact": float(extreme / total),
        "n_positive": positive,
        "n_negative": negative,
        "n_tie": ties,
        "n_nonzero": int(len(nonzero)),
    }


def _bootstrap_median_ci(values: np.ndarray, seed: int, draws: int = 50_000) -> list[float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    medians = np.median(values[indices], axis=1)
    return [float(x) for x in np.quantile(medians, [0.025, 0.975])]


def _comparison(table: pd.DataFrame, left: str, right: str, label: str) -> dict:
    eligible = table[[left, right]].dropna()
    delta = (eligible[left] - eligible[right]).to_numpy(float)
    result = {
        "label": label,
        "left": left,
        "right": right,
        "n_patients": int(len(delta)),
        "median_delta": float(np.median(delta)) if len(delta) else float("nan"),
        "mean_delta": float(np.mean(delta)) if len(delta) else float("nan"),
        "bootstrap_95ci_median": _bootstrap_median_ci(
            delta, _stable_seed(f"v0.4-bootstrap:{label}")
        ),
    }
    result.update(_exact_signed_rank(delta))
    return result


def _score_group(group: pd.DataFrame, target_column: str) -> float:
    group = group.sort_values("contact")
    return _maxab(
        group.prediction_a.to_numpy(float),
        group.prediction_b.to_numpy(float),
        group[target_column].to_numpy(float),
    )


def _ensemble_true(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    true = raw.loc[(raw.draw == -1) & raw.model.isin(MODEL_ORDER)].copy()
    keys = ["subject", "seizure_id", "seizure_idx", "contact", "model"]
    checks = true.groupby(keys, sort=False).agg(
        n_seed=("seed", "nunique"),
        target45_min=("target_1_45", "min"),
        target45_max=("target_1_45", "max"),
        target150_min=("target_1_150", "min"),
        target150_max=("target_1_150", "max"),
    )
    if not (checks.n_seed == 3).all():
        raise RuntimeError("not every true prediction has all three seeds")
    if not np.allclose(checks.target45_min, checks.target45_max, atol=0, rtol=0):
        raise RuntimeError("1-45 Hz target differs across seeds")
    if not np.allclose(checks.target150_min, checks.target150_max, atol=0, rtol=0):
        raise RuntimeError("1-150 Hz target differs across seeds")
    ensemble = true.groupby(keys, as_index=False, sort=False).agg(
        prediction_a=("prediction_a", "mean"),
        prediction_b=("prediction_b", "mean"),
        target_1_45=("target_1_45", "first"),
        target_1_150=("target_1_150", "first"),
    )
    rows = []
    for keys_value, group in ensemble.groupby(["subject", "seizure_id", "model"], sort=False):
        subject, seizure, model = keys_value
        rows.append(
            {
                "subject": subject,
                "seizure_id": seizure,
                "model": model,
                "n_contacts": int(len(group)),
                "maxab_1_45": _score_group(group, "target_1_45"),
                "maxab_1_150_no_retrain": _score_group(group, "target_1_150"),
            }
        )
    return ensemble, pd.DataFrame(rows)


def _score_order_control(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    control = raw.loc[raw.model == "M3_ORDER_SHUFFLE_FULL_HISTORY"].copy()
    seizure_rows = []
    for keys, group in control.groupby(["subject", "seizure_id", "seed", "draw"], sort=False):
        subject, seizure, seed, draw = keys
        seizure_rows.append(
            {
                "subject": subject,
                "seizure_id": seizure,
                "seed": int(seed),
                "draw": int(draw),
                "maxab_1_45": _score_group(group, "target_1_45"),
            }
        )
    seizure_table = pd.DataFrame(seizure_rows)
    draws = seizure_table.groupby(["subject", "seed", "draw"], as_index=False).agg(
        patient_median_maxab_1_45=("maxab_1_45", "median")
    )
    patient = draws.groupby("subject", as_index=False).agg(
        m3_order_shuffle_mean_1_45=("patient_median_maxab_1_45", "mean"),
        m3_order_shuffle_p05_1_45=("patient_median_maxab_1_45", lambda x: np.quantile(x, 0.05)),
        m3_order_shuffle_p95_1_45=("patient_median_maxab_1_45", lambda x: np.quantile(x, 0.95)),
        n_order_controls=("patient_median_maxab_1_45", "size"),
    )
    return draws, patient


def _score_swap_control(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    swap = raw.loc[raw.model == "M3_WITHIN_PATIENT_HISTORY_SWAP"].copy()
    if swap.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["subject", "m3_history_swap_median_1_45"])
    keys = ["subject", "seizure_id", "seizure_idx", "contact", "donor_seizure_id"]
    ensemble = swap.groupby(keys, as_index=False, sort=False).agg(
        prediction_a=("prediction_a", "mean"),
        prediction_b=("prediction_b", "mean"),
        target_1_45=("target_1_45", "first"),
    )
    rows = []
    for keys_value, group in ensemble.groupby(
        ["subject", "seizure_id", "donor_seizure_id"], sort=False
    ):
        subject, seizure, donor = keys_value
        rows.append(
            {
                "subject": subject,
                "seizure_id": seizure,
                "donor_seizure_id": donor,
                "maxab_1_45": _score_group(group, "target_1_45"),
            }
        )
    donor_table = pd.DataFrame(rows)
    per_target = donor_table.groupby(["subject", "seizure_id"], as_index=False).agg(
        swapped_history_median_maxab_1_45=("maxab_1_45", "median"),
        n_donors=("donor_seizure_id", "nunique"),
    )
    patient = per_target.groupby("subject", as_index=False).agg(
        m3_history_swap_median_1_45=("swapped_history_median_maxab_1_45", "median"),
        n_swap_targets=("seizure_id", "nunique"),
    )
    return per_target, patient


def _channel_null(
    ensemble: pd.DataFrame,
    subjects: list[str],
    draws: int,
) -> pd.DataFrame:
    prepared: dict[str, list[dict]] = {}
    for subject in subjects:
        seizures = []
        subset = ensemble.loc[ensemble.subject == subject]
        for seizure, seizure_group in subset.groupby("seizure_id", sort=True):
            reference = seizure_group.loc[seizure_group.model == MODEL_ORDER[0]].sort_values("contact")
            target = _standard_rank(reference.target_1_45.to_numpy(float))
            if target is None:
                continue
            candidates = {}
            for model in MODEL_ORDER:
                group = seizure_group.loc[seizure_group.model == model].sort_values("contact")
                candidates[model] = (
                    _standard_rank(group.prediction_a.to_numpy(float)),
                    _standard_rank(group.prediction_b.to_numpy(float)),
                )
            seizures.append({"seizure_id": seizure, "target": target, "candidates": candidates})
        prepared[subject] = seizures

    rows = []
    for subject in subjects:
        rng = np.random.default_rng(_stable_seed(f"v0.4-channel-null:{subject}"))
        seizures = prepared[subject]
        for draw in range(draws):
            scores = {model: [] for model in MODEL_ORDER}
            for seizure in seizures:
                permuted = seizure["target"][rng.permutation(len(seizure["target"]))]
                for model in MODEL_ORDER:
                    candidate_a, candidate_b = seizure["candidates"][model]
                    scores[model].append(
                        max(abs(float(candidate_a @ permuted)), abs(float(candidate_b @ permuted)))
                    )
            for model in MODEL_ORDER:
                rows.append(
                    {
                        "subject": subject,
                        "draw": draw,
                        "model": model,
                        "patient_median_maxab_1_45": float(np.median(scores[model])),
                    }
                )
    return pd.DataFrame(rows)


def _environment_manifest(root: Path, files: list[Path]) -> dict:
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        git_status = subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ).splitlines()
    except Exception:
        git_head, git_status = "unavailable", []
    return {
        "contract": "topic5_history_conditioned_field_refinement_v0_4",
        "git_head": git_head,
        "worktree_dirty_entries": git_status,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "files": {
            str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
            if path.exists()
        },
        "root": str(root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "results/topic5_history_conditioned_field_refinement_v0_4",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "INPUT_MANIFEST.json").read_text())
    subjects = list(manifest["cohort"]["primary_subjects"])
    seeds = [11, 29, 47]
    missing = []
    raw_frames = []
    done_payloads = []
    for seed in seeds:
        for subject in subjects:
            directory = root / "per_subject" / f"seed_{seed}" / subject
            if not (directory / "DONE.json").exists():
                missing.append(f"seed_{seed}/{subject}")
                continue
            done = json.loads((directory / "DONE.json").read_text())
            if done.get("heldout_target_used_for_training") is not False:
                raise RuntimeError(f"heldout target contract failed for {directory}")
            done_payloads.append(done)
            raw_frames.append(pd.read_csv(directory / "heldout_candidate_predictions.csv.gz"))
    if missing:
        raise RuntimeError(f"formal run incomplete ({len(missing)}/45 missing): {missing[:8]}")
    raw = pd.concat(raw_frames, ignore_index=True)
    numeric = raw[["prediction_a", "prediction_b", "target_1_45", "target_1_150"]].to_numpy()
    if not np.all(np.isfinite(numeric)):
        raise RuntimeError("non-finite heldout prediction or target")

    ensemble, seizure_metrics = _ensemble_true(raw)
    seed_metrics = []
    true_raw = raw.loc[(raw.draw == -1) & raw.model.isin(MODEL_ORDER)]
    for keys, group in true_raw.groupby(["subject", "seizure_id", "seed", "model"], sort=False):
        subject, seizure, seed, model = keys
        seed_metrics.append(
            {
                "subject": subject,
                "seizure_id": seizure,
                "seed": int(seed),
                "model": model,
                "maxab_1_45": _score_group(group, "target_1_45"),
            }
        )
    seed_metrics = pd.DataFrame(seed_metrics)

    patient_long = seizure_metrics.groupby(["subject", "model"], as_index=False).agg(
        patient_median_maxab_1_45=("maxab_1_45", "median"),
        patient_median_maxab_1_150_no_retrain=("maxab_1_150_no_retrain", "median"),
        n_seizures=("seizure_id", "nunique"),
        min_contacts=("n_contacts", "min"),
        max_contacts=("n_contacts", "max"),
    )
    wide45 = patient_long.pivot(index="subject", columns="model", values="patient_median_maxab_1_45")
    wide150 = patient_long.pivot(
        index="subject", columns="model", values="patient_median_maxab_1_150_no_retrain"
    )
    patient = pd.DataFrame(index=subjects)
    for model in MODEL_ORDER:
        base = MODEL_COLUMNS[model]
        patient[f"{base}_1_45"] = wide45[model]
        patient[f"{base}_1_150_no_retrain"] = wide150[model]
    denominator = patient_long.groupby("subject").first()[["n_seizures", "min_contacts", "max_contacts"]]
    patient = patient.join(denominator)

    order_draws, order_patient = _score_order_control(raw)
    swap_targets, swap_patient = _score_swap_control(raw)
    patient = patient.reset_index(names="subject")
    patient = patient.merge(order_patient, on="subject", how="left")
    patient = patient.merge(swap_patient, on="subject", how="left")
    patient["delta_m3_minus_m0_1_45"] = patient.m3_joint_rnn_1_45 - patient.m0_static_ab_1_45
    patient["delta_m3_minus_m1_1_45"] = patient.m3_joint_rnn_1_45 - patient.m1_frozen_history_head_1_45
    patient["delta_m3_minus_m2_1_45"] = patient.m3_joint_rnn_1_45 - patient.m2_time_aware_nonrecurrent_1_45
    patient["delta_m1_minus_m0_1_45"] = patient.m1_frozen_history_head_1_45 - patient.m0_static_ab_1_45
    patient["delta_m2_minus_m0_1_45"] = patient.m2_time_aware_nonrecurrent_1_45 - patient.m0_static_ab_1_45
    patient["delta_true_minus_order_shuffle_1_45"] = (
        patient.m3_joint_rnn_1_45 - patient.m3_order_shuffle_mean_1_45
    )
    patient["delta_correct_minus_history_swap_1_45"] = (
        patient.m3_joint_rnn_1_45 - patient.m3_history_swap_median_1_45
    )

    null_draws = int(json.loads((ROOT / "config/topic5_history_conditioned_field_refinement_v0_4.json").read_text())["channel_null_draws"])
    channel_null = _channel_null(ensemble, subjects, null_draws)
    null_summary = channel_null.groupby(["subject", "model"], as_index=False).agg(
        channel_null_median=("patient_median_maxab_1_45", "median"),
        channel_null_p95=("patient_median_maxab_1_45", lambda x: np.quantile(x, 0.95)),
    )
    observed_long = patient_long[["subject", "model", "patient_median_maxab_1_45"]]
    null_summary = null_summary.merge(observed_long, on=["subject", "model"], how="left")
    null_summary["observed_minus_channel_null_median"] = (
        null_summary.patient_median_maxab_1_45 - null_summary.channel_null_median
    )
    p_values = []
    for row in null_summary.itertuples(index=False):
        values = channel_null.loc[
            (channel_null.subject == row.subject) & (channel_null.model == row.model),
            "patient_median_maxab_1_45",
        ].to_numpy()
        p_values.append(float((1 + np.sum(values >= row.patient_median_maxab_1_45)) / (1 + len(values))))
    null_summary["channel_null_p_one_sided"] = p_values
    null_summary["above_channel_null_p95"] = (
        null_summary.patient_median_maxab_1_45 > null_summary.channel_null_p95
    )
    for model in MODEL_ORDER:
        label = MODEL_COLUMNS[model]
        subset = null_summary.loc[null_summary.model == model].set_index("subject")
        patient[f"{label}_minus_channel_null_median_1_45"] = patient.subject.map(
            subset.observed_minus_channel_null_median
        )
        patient[f"{label}_channel_null_p_one_sided"] = patient.subject.map(
            subset.channel_null_p_one_sided
        )

    comparisons = {
        "primary_m3_minus_m0": _comparison(
            patient, "m3_joint_rnn_1_45", "m0_static_ab_1_45", "M3-M0"
        ),
        "m3_minus_m1": _comparison(
            patient, "m3_joint_rnn_1_45", "m1_frozen_history_head_1_45", "M3-M1"
        ),
        "m3_minus_m2": _comparison(
            patient, "m3_joint_rnn_1_45", "m2_time_aware_nonrecurrent_1_45", "M3-M2"
        ),
        "m1_minus_m0": _comparison(
            patient, "m1_frozen_history_head_1_45", "m0_static_ab_1_45", "M1-M0"
        ),
        "m2_minus_m0": _comparison(
            patient, "m2_time_aware_nonrecurrent_1_45", "m0_static_ab_1_45", "M2-M0"
        ),
        "true_minus_order_shuffle": _comparison(
            patient, "m3_joint_rnn_1_45", "m3_order_shuffle_mean_1_45", "M3 true-order - full-history shuffle"
        ),
        "correct_minus_history_swap": _comparison(
            patient, "m3_joint_rnn_1_45", "m3_history_swap_median_1_45", "M3 correct-history - within-patient swap"
        ),
        "sensitivity_1_150_m3_minus_m0": _comparison(
            patient,
            "m3_joint_rnn_1_150_no_retrain",
            "m0_static_ab_1_150_no_retrain",
            "M3-M0 1-150Hz no-retrain",
        ),
    }
    channel_null_cohort = {}
    for model in MODEL_ORDER:
        subset = null_summary.loc[null_summary.model == model]
        deltas = subset.observed_minus_channel_null_median.to_numpy()
        channel_null_cohort[model] = {
            "n_patients": int(len(subset)),
            "median_observed": float(subset.patient_median_maxab_1_45.median()),
            "median_null": float(subset.channel_null_median.median()),
            "median_margin": float(np.median(deltas)),
            "n_above_null_median": int(np.sum(deltas > TIE_TOLERANCE)),
            "n_above_null_p95": int(subset.above_channel_null_p95.sum()),
            **_exact_signed_rank(deltas),
        }

    gains = {
        model: [done["final_gains"][model] for done in done_payloads]
        for model in ("m1", "m2", "m3")
    }
    diagnostic_table = pd.read_csv(root / "history_conditioned_field_state_diagnostics_summary.csv")
    diagnostic_summary = {}
    for model, group in diagnostic_table.groupby("model", sort=True):
        diagnostic_summary[model] = {
            "state_norm_median": float(group.state_norm_median.median()),
            "gain_a_median": float(group.gain_a.median()),
            "gain_b_median": float(group.gain_b.median()),
            "candidate_angle_a_median_degrees": float(
                group.candidate_angle_a_median_degrees.median()
            ),
            "candidate_angle_b_median_degrees": float(
                group.candidate_angle_b_median_degrees.median()
            ),
        }
    static_anchor = pd.read_csv(root / "static_anchor_patient_metrics.csv")
    static_margin = static_anchor.observed_minus_null_median.to_numpy(float)
    static_anchor_summary = {
        "n_patients": int(len(static_anchor)),
        "patient_median_maxab": float(
            static_anchor.observed_patient_median_maxab_1_45.median()
        ),
        "patient_median_channel_null": float(static_anchor.channel_null_median.median()),
        "patient_median_margin": float(np.median(static_margin)),
        "bootstrap_95ci_median_margin": _bootstrap_median_ci(
            static_margin, _stable_seed("v0.4-static-anchor")
        ),
        "n_above_individual_p95": int(static_anchor.pass_null_p95.sum()),
        **_exact_signed_rank(static_margin),
    }
    summary = {
        "status": "COMPLETE",
        "contract": "topic5_history_conditioned_field_refinement_v0_4",
        "scientific_question": "Does causal interictal history improve frozen static A/B early-ictal field correspondence?",
        "primary_endpoint": "clinical_onset_[0,10]s_1-45Hz_contact_energy",
        "sensitivity_endpoint": "1-150Hz_no_retrain",
        "aggregation": "seed-wise candidate-field mean -> seizure maxAB -> patient median -> cohort",
        "cohort": {
            "n_patients": len(subjects),
            "n_seizures": int(seizure_metrics.loc[seizure_metrics.model == MODEL_ORDER[0], "seizure_id"].nunique()),
            "patients": subjects,
            "contact_denominator_min_median_max": [
                int(patient.min_contacts.min()),
                float(patient.min_contacts.median()),
                int(patient.max_contacts.max()),
            ],
        },
        "completeness": {
            "formal_units_complete": len(done_payloads),
            "formal_units_expected": 45,
            "failed_units": 0,
            "seeds": seeds,
        },
        "static_boundary": manifest["static_ab_boundary"],
        "static_anchor_reproduction": static_anchor_summary,
        "comparisons": comparisons,
        "matched_channel_null": channel_null_cohort,
        "history_controls": {
            "order_shuffle": "32 full-history permutations per seed; time slots retained; patient control is the mean over 96 seed-draw realizations",
            "within_patient_swap_n": int(patient.m3_history_swap_median_1_45.notna().sum()),
        },
        "training_diagnostics": {
            "peak_gpu_memory_mb_max": float(max(done["peak_gpu_memory_mb"] for done in done_payloads)),
            "elapsed_seconds_median": float(np.median([done["elapsed_seconds"] for done in done_payloads])),
            "m3_half_life_hours_median": float(np.median([done["final_m3_half_life_hours"] for done in done_payloads])),
            "final_gains_median": {
                model: np.median(np.asarray(values), axis=0).tolist() for model, values in gains.items()
            },
            "heldout_state_and_field_change": diagnostic_summary,
        },
        "interpretation_policy": "No composite gate; each contrast is reported independently.",
        "prohibited_claims": [
            "unique signed seizure-field prediction",
            "seizure-time forecasting",
            "causal shaping of seizure recruitment",
            "cellular E/I interpretation of hidden units",
            "fully prospective model",
        ],
    }

    paths = {
        "seizure": root / "history_conditioned_field_seizure_metrics.csv",
        "patient": root / "history_conditioned_field_patient_metrics.csv",
        "seed": root / "history_conditioned_field_seed_metrics.csv",
        "order": root / "history_conditioned_field_order_shuffle.csv.gz",
        "swap": root / "history_conditioned_field_history_swap.csv",
        "null": root / "history_conditioned_field_channel_null.csv",
        "null_summary": root / "history_conditioned_field_channel_null_summary.csv",
        "summary": root / "HISTORY_CONDITIONED_FIELD_SUMMARY.json",
    }
    seizure_metrics.to_csv(paths["seizure"], index=False)
    patient.to_csv(paths["patient"], index=False)
    seed_metrics.to_csv(paths["seed"], index=False)
    order_draws.to_csv(paths["order"], index=False, compression="gzip")
    swap_targets.to_csv(paths["swap"], index=False)
    channel_null.to_csv(paths["null"], index=False)
    null_summary.to_csv(paths["null_summary"], index=False)
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    reproducibility_files = [
        root / "INPUT_MANIFEST.json",
        ROOT / "config/topic5_history_conditioned_field_refinement_v0_4.json",
        ROOT / "src/topic5_static_anchored_history_residual.py",
        ROOT / "src/topic5_history_rnn.py",
        ROOT / "src/topic5_history_bridge.py",
        ROOT / "src/topic5_rank_distribution.py",
        ROOT / "scripts/build_topic5_history_conditioned_field_cache_v0_4.py",
        ROOT / "scripts/run_topic5_history_conditioned_field_fold_v0_4.py",
        ROOT / "scripts/extract_topic5_history_conditioned_field_diagnostics_v0_4.py",
        ROOT / "scripts/accept_topic5_history_conditioned_field_v0_4.py",
        ROOT / "scripts/plot_topic5_history_conditioned_field_v0_4.py",
        ROOT / "scripts/report_topic5_history_conditioned_field_v0_4.py",
        ROOT / "scripts/run_topic5_history_conditioned_field_v0_4.sh",
        Path(__file__).resolve(),
        root / "ACCEPTANCE.json",
        root / "ACCEPTANCE_UNIT_TABLE.csv",
        root / "history_conditioned_field_state_diagnostics.csv.gz",
        root / "history_conditioned_field_state_diagnostics_summary.csv",
        *paths.values(),
    ]
    reproduction = _environment_manifest(root, reproducibility_files)
    (root / "REPRODUCIBILITY_MANIFEST.json").write_text(
        json.dumps(reproduction, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
