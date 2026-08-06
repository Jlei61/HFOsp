#!/usr/bin/env python3
"""Summarize interictal units, then run isolated same-patient ictal readout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_topic5_history_conditioned_field_v0_4 import (  # noqa: E402
    _exact_signed_rank,
)
from src.topic5_patient_specific_rnn_bridge import (  # noqa: E402
    permutation_indices,
)


MODEL_ORDER = (
    "full_history_gru",
    "linear_state",
    "rank_shuffle_gru",
    "static_fit60",
    "empirical_test20",
)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def fields_from_reference(data, prefix: str) -> dict[str, np.ndarray]:
    participation = np.asarray(data[f"{prefix}_participation"], dtype=float)
    mean_rank = np.asarray(data[f"{prefix}_mean_rank"], dtype=float)
    histogram = np.asarray(data[f"{prefix}_rank_histogram"], dtype=float)
    early = participation * np.sum(histogram[:, :3], axis=1)
    late = participation * np.sum(histogram[:, -3:], axis=1)
    return {
        "participation": participation,
        "early_joint_mass": early,
        "late_joint_mass": late,
        "endpoint_joint_mass": early + late,
        "weighted_earliness": participation * (1.0 - np.where(np.isfinite(mean_rank), mean_rank, 0.5)),
    }


def centered_ranks(values: np.ndarray) -> np.ndarray:
    ranked = rankdata(np.asarray(values, dtype=float), method="average")
    return ranked - np.mean(ranked)


def score_one_target(
    fields: dict[str, np.ndarray],
    target: np.ndarray,
    candidate_order: list[str],
    permutations: np.ndarray,
) -> tuple[float, np.ndarray, str]:
    target_rank = centered_ranks(target)
    field_rank = np.column_stack([centered_ranks(fields[name]) for name in candidate_order])
    field_norm = np.linalg.norm(field_rank, axis=0)
    target_norm = np.linalg.norm(target_rank)
    valid = field_norm > 1e-12
    if target_norm <= 1e-12 or not np.any(valid):
        raise ValueError("constant target or all candidate fields")
    correlation = np.full(len(candidate_order), np.nan)
    correlation[valid] = (target_rank @ field_rank[:, valid]) / (target_norm * field_norm[valid])
    selected = int(np.nanargmax(np.abs(correlation)))
    permuted = target_rank[permutations]
    null_correlation = np.full((len(permutations), len(candidate_order)), np.nan)
    null_correlation[:, valid] = (
        permuted @ field_rank[:, valid]
    ) / (target_norm * field_norm[valid][None, :])
    null = np.nanmax(np.abs(null_correlation), axis=1)
    return float(abs(correlation[selected])), null, candidate_order[selected]


def load_model_fields(unit: Path, candidate_order: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(unit / "free_rollouts.npz", allow_pickle=False) as data:
        names = np.asarray(data["contact_names"]).astype(str)
        fields = {name: np.asarray(data[f"field__{name}"], dtype=float) for name in candidate_order}
    return names, fields


def load_reference_fields(unit: Path, prefix: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(unit / "empirical_references.npz", allow_pickle=False) as data:
        names = np.asarray(data["contact_names"]).astype(str)
        fields = fields_from_reference(data, prefix)
    return names, fields


def target_files(cache_root: Path, subject: str) -> list[Path]:
    directory = cache_root / f"outer_{subject}"
    return sorted(directory.glob(f"{subject}__*.npz"))


def summarize_interictal(output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for done_path in sorted((output / "units").glob("*/*/seed_*/DONE.json")):
        payload = json.loads(done_path.read_text())
        if payload.get("status") != "COMPLETE":
            continue
        rows.append({
            "subject": payload["subject"], "model": payload["model"], "seed": payload["seed"],
            "n_contacts": payload["n_contacts"],
            "fit_events": payload["n_events"]["fit60"],
            "validation_events": payload["n_events"]["validation20"],
            "test_events": payload["n_events"]["test20"],
            "validation_nll": payload["validation"]["heldout_event_nll"],
            "test_nll": payload["test"]["heldout_event_nll"],
            "top1": payload["test"]["top1_next_set_accuracy"],
            "participation_mae": payload["rollout_errors"]["participation_mae"],
            "rank_wasserstein": payload["rollout_errors"]["rank_wasserstein"],
            "precedence_correlation": payload["rollout_errors"]["precedence_correlation"],
            "runtime_seconds": payload["runtime_seconds"],
            "peak_gpu_memory_mb": payload["peak_gpu_memory_mb"],
        })
    seed_frame = pd.DataFrame(rows).sort_values(["subject", "model", "seed"])
    patient = seed_frame.groupby(["subject", "model"], as_index=False).median(numeric_only=True)
    seed_frame.to_csv(output / "interictal_seed_metrics.csv", index=False)
    patient.to_csv(output / "interictal_patient_metrics.csv", index=False)
    return seed_frame, patient


def run_ictal_readout(config: dict, output: Path, subjects: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_order = list(config["readout"]["candidate_fields"])
    cache_root = ROOT / config["target_cache_root"]
    seeds = list(map(int, config["training"]["seeds"]))
    ictal = config["ictal_transfer"]
    band_targets = {
        "1_150": str(ictal["primary_target_key"]),
        "1_45": str(ictal["sensitivity_target_key"]),
    }
    seizure_rows = []
    patient_rows = []
    for subject_index, subject in enumerate(subjects):
        units = output / "units" / subject
        model_seed_fields: dict[str, list[tuple[int, np.ndarray, dict[str, np.ndarray]]]] = {}
        for model in ("full_history_gru", "linear_state", "rank_shuffle_gru"):
            model_seed_fields[model] = []
            for seed in seeds:
                names, fields = load_model_fields(units / model / f"seed_{seed}", candidate_order)
                model_seed_fields[model].append((seed, names, fields))
        reference_unit = units / "full_history_gru" / f"seed_{seeds[0]}"
        names, fit_fields = load_reference_fields(reference_unit, "fit")
        _, test_fields = load_reference_fields(reference_unit, "test")
        model_seed_fields["static_fit60"] = [(seeds[0], names, fit_fields)]
        model_seed_fields["empirical_test20"] = [(seeds[0], names, test_fields)]

        files = target_files(cache_root, subject)
        if not files:
            raise RuntimeError(f"{subject}: no own clinical-onset target files")
        for band, target_key in band_targets.items():
            for model, seed_fields in model_seed_fields.items():
                observed_by_seed = []
                all_null_by_seed = []
                shaft_null_by_seed = []
                for seed, model_names, fields_full in seed_fields:
                    lookup = {name: index for index, name in enumerate(model_names)}
                    observed_seizure = []
                    all_null_seizure = []
                    shaft_null_seizure = []
                    for seizure_index, path in enumerate(files):
                        with np.load(path, allow_pickle=False) as data:
                            joined_names = np.asarray(data["contact_names"]).astype(str)
                            target = np.asarray(data[target_key], dtype=float)
                        if not all(name in lookup for name in joined_names):
                            raise RuntimeError(f"{subject}: target/model contact mismatch")
                        keep = np.asarray([lookup[name] for name in joined_names], dtype=np.int64)
                        finite = np.isfinite(target)
                        keep = keep[finite]
                        joined_names = joined_names[finite]
                        target = target[finite]
                        if len(target) < 6:
                            continue
                        fields = {name: values[keep] for name, values in fields_full.items()}
                        all_perm = permutation_indices(
                            joined_names, n_draws=int(ictal["all_contact_permutations"]),
                            seed=2026080300 + subject_index * 1000 + seizure_index, within_shaft=False,
                        )
                        shaft_perm = permutation_indices(
                            joined_names, n_draws=int(ictal["within_shaft_permutations"]),
                            seed=2026080400 + subject_index * 1000 + seizure_index, within_shaft=True,
                        )
                        observed, all_null, selected = score_one_target(
                            fields, target, candidate_order, all_perm
                        )
                        _, shaft_null, _ = score_one_target(fields, target, candidate_order, shaft_perm)
                        observed_seizure.append(observed)
                        all_null_seizure.append(all_null)
                        shaft_null_seizure.append(shaft_null)
                        seizure_rows.append({
                            "subject": subject, "model": model, "seed": seed, "band": band,
                            "seizure_file": path.name, "n_contacts": len(target),
                            "observed_max_abs_rho": observed, "selected_field": selected,
                            "all_contact_null_median": float(np.median(all_null)),
                            "within_shaft_null_median": float(np.median(shaft_null)),
                        })
                    if not observed_seizure:
                        raise RuntimeError(f"{subject}/{band}: no scoreable seizures")
                    observed_by_seed.append(float(np.median(observed_seizure)))
                    all_null_by_seed.append(np.median(np.stack(all_null_seizure), axis=0))
                    shaft_null_by_seed.append(np.median(np.stack(shaft_null_seizure), axis=0))
                observed = float(np.median(observed_by_seed))
                all_null = np.median(np.stack(all_null_by_seed), axis=0)
                shaft_null = np.median(np.stack(shaft_null_by_seed), axis=0)
                patient_rows.append({
                    "subject": subject, "model": model, "band": band,
                    "n_seizures": len(files),
                    "n_seeds": len(seed_fields),
                    "observed_max_abs_rho": observed,
                    "all_contact_null_median": float(np.median(all_null)),
                    "all_contact_margin": float(observed - np.median(all_null)),
                    "all_contact_p": float((1 + np.count_nonzero(all_null >= observed)) / (len(all_null) + 1)),
                    "within_shaft_null_median": float(np.median(shaft_null)),
                    "within_shaft_margin": float(observed - np.median(shaft_null)),
                    "within_shaft_p": float((1 + np.count_nonzero(shaft_null >= observed)) / (len(shaft_null) + 1)),
                    "development_supportive": subject == ictal["development_subject"],
                })
            print(f"ictal {subject_index + 1}/{len(subjects)} {subject} {band}", flush=True)
    seizures = pd.DataFrame(seizure_rows)
    patients = pd.DataFrame(patient_rows)
    seizures.to_csv(output / "early_ictal_seizure_metrics.csv", index=False)
    patients.to_csv(output / "early_ictal_patient_metrics.csv", index=False)
    return seizures, patients


def inference_summary(patients: pd.DataFrame) -> dict:
    summary = {}
    for band in sorted(patients.band.unique()):
        summary[band] = {}
        primary = patients.loc[(patients.band == band) & ~patients.development_supportive]
        for model in MODEL_ORDER:
            group = primary.loc[primary.model == model]
            values = group.all_contact_margin.to_numpy(float)
            shaft_values = group.within_shaft_margin.to_numpy(float)
            summary[band][model] = {
                "n": int(len(group)),
                "median_absolute_similarity": finite_or_none(group.observed_max_abs_rho.median()),
                "median_all_contact_margin": finite_or_none(np.median(values)),
                "n_positive_margin": int(np.sum(values > 1e-9)),
                "n_negative_margin": int(np.sum(values < -1e-9)),
                "n_tied_margin": int(np.sum(np.abs(values) <= 1e-9)),
                "margin_vs_zero": _exact_signed_rank(values, tolerance=1e-9),
                "median_within_shaft_margin": finite_or_none(np.median(shaft_values)),
                "within_shaft_margin_vs_zero": _exact_signed_rank(
                    shaft_values, tolerance=1e-9
                ),
            }
        wide = primary.pivot(index="subject", columns="model", values="all_contact_margin")
        shaft_wide = primary.pivot(
            index="subject", columns="model", values="within_shaft_margin"
        )
        comparisons = {}
        for left, right in (
            ("full_history_gru", "static_fit60"),
            ("full_history_gru", "rank_shuffle_gru"),
            ("full_history_gru", "linear_state"),
            ("full_history_gru", "empirical_test20"),
        ):
            values = (wide[left] - wide[right]).dropna().to_numpy(float)
            comparisons[f"{left}_minus_{right}"] = {
                "n": int(len(values)), "median": finite_or_none(np.median(values)),
                "test": _exact_signed_rank(values, tolerance=1e-9),
                "within_shaft_median": finite_or_none(
                    np.median((shaft_wide[left] - shaft_wide[right]).dropna().to_numpy(float))
                ),
                "within_shaft_test": _exact_signed_rank(
                    (shaft_wide[left] - shaft_wide[right]).dropna().to_numpy(float),
                    tolerance=1e-9,
                ),
            }
        summary[band]["paired_comparisons"] = comparisons
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = ROOT / config["output_root"]
    state = json.loads((output / "watchers/launcher_state.json").read_text())
    if state.get("status") != "COMPLETE" or state.get("n_failed") != 0:
        raise SystemExit("target-free units are not completely frozen")
    subjects = list(map(str, state["subjects"]))
    seed_frame, interictal_patient = summarize_interictal(output)
    expected = len(subjects) * len(config["training"]["seeds"]) * 3
    if len(seed_frame) != expected:
        raise RuntimeError(f"expected {expected} units, found {len(seed_frame)}")
    seizures, patients = run_ictal_readout(config, output, subjects)
    summary = {
        "status": "COMPLETE",
        "contract": config["contract"],
        "n_subjects": len(subjects),
        "n_primary_subjects": len(subjects) - 1,
        "n_supportive_development_subjects": 1,
        "n_units": len(seed_frame),
        "n_failed_units": 0,
        "other_patient_events_used": False,
        "empirical_ab_used": False,
        "ictal_target_used_for_training": False,
        "interictal": {
            model: {
                "median_test_nll": finite_or_none(group.test_nll.median()),
                "median_precedence_correlation": finite_or_none(group.precedence_correlation.median()),
                "median_rank_wasserstein": finite_or_none(group.rank_wasserstein.median()),
            }
            for model, group in interictal_patient.groupby("model")
        },
        "early_ictal": inference_summary(patients),
    }
    primary_subjects = [
        subject for subject in subjects
        if subject != str(config["ictal_transfer"]["development_subject"])
    ]
    primary_interictal = interictal_patient.loc[
        interictal_patient.subject.isin(primary_subjects)
    ]
    nll_wide = primary_interictal.pivot(
        index="subject", columns="model", values="test_nll"
    )
    order_gain = (
        nll_wide["rank_shuffle_gru"] - nll_wide["full_history_gru"]
    ).dropna().to_numpy(float)
    summary["interictal_primary_inference"] = {
        "n": int(len(order_gain)),
        "rank_shuffle_minus_full_nll_median": finite_or_none(np.median(order_gain)),
        "rank_shuffle_minus_full_nll_test": _exact_signed_rank(
            order_gain, tolerance=1e-9
        ),
    }
    atomic_json(output / "PATIENT_SPECIFIC_RNN_BRIDGE_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
