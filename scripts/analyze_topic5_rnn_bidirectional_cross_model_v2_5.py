#!/usr/bin/env python3
"""A/B-independent bidirectionality and cross-model static-transfer audit v2.5."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_topic5_competitive_propagation_development_v2_3 import (  # noqa: E402
    load_subject,
)
from scripts.train_topic5_competitive_propagation_formal_v2_3 import (  # noqa: E402
    axis_for_subject,
    build_model,
)
from src.topic5_axis_positive_static_transfer_v2_4 import (  # noqa: E402
    paired_rollout_design,
    rollout_model_distribution,
)
from src.topic5_transition_decomposition_v0_1 import (  # noqa: E402
    estimate_node_hazard,
    fibonacci_axes,
    logit,
)


V23 = ROOT / "results/topic5_symmetric_axis_competitive_propagation_v2_3"
V24 = ROOT / "results/topic5_rnn_axis_positive_static_transfer_v2_4"
RANK_ROOT = (
    ROOT
    / "results/topic5_interictal_rank_distribution/runs/"
    "formal_multiseed_20260725_v1"
)
LOW_RANK_ROOT = (
    ROOT
    / "results/topic5_low_rank_dynamics/runs/"
    "low_rank_leaky_multiseed_20260725_v1"
)
DATASET = ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4"
TARGET_ROOT = ROOT / "results/topic5_ictal_recruitment/t0_feature_cache_bb150_1_150"
FIT1 = (
    ROOT
    / "results/topic5_state_conditioned_predictor/fit12_clinical_bb150/"
    "fit1/fig6_fit1_clinical_onset_scaffold_event.csv"
)
RELATION_ROOT = (
    ROOT
    / "results/topic5_state_conditioned_predictor/fit2_prefix_scaffold/"
    "per_subject"
)
OUT = ROOT / "results/topic5_rnn_bidirectional_cross_model_audit_v2_5"
SEEDS_V23 = (17, 29, 43)
SEED_DIRS = ("seed_20260725", "seed_20260726", "seed_20260727")
N_PERM = 5000
FIELDS = (
    "participation",
    "early_joint_mass",
    "late_joint_mass",
    "endpoint_joint_mass",
    "weighted_earliness",
)
ORDINARY_CONTROLS = (
    "empirical_rank_distribution",
    "full_history_gru",
    "static_contact_hazard",
    "unordered_prefix",
    "last_set_first_order",
    "rank_shuffle_gru",
)
STRUCTURED_CONTROLS = (
    "structured_empirical_train80",
    "structured_full",
    "structured_no_history",
    "structured_local_isotropic",
    "structured_axis_no_source",
    "structured_node_only",
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def bootstrap_median_ci(values: np.ndarray, seed: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample = rng.choice(values, size=(20_000, len(values)), replace=True)
    return np.quantile(np.median(sample, axis=1), [0.025, 0.975]).tolist()


def cohort_summary(values: np.ndarray, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    p = (
        1.0
        if not len(values) or np.allclose(values, 0.0)
        else float(wilcoxon(values, alternative="greater").pvalue)
    )
    return {
        "n": int(len(values)),
        "median": float(np.median(values)) if len(values) else None,
        "bootstrap_ci95": bootstrap_median_ci(values, seed) if len(values) else [],
        "n_positive": int(np.count_nonzero(values > 0)),
        "wilcoxon_greater_p": p,
    }


def bh_fdr(values: list[float]) -> list[float]:
    p = np.asarray(values, dtype=np.float64)
    order = np.argsort(p)
    adjusted = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result.tolist()


def relation_for_subject(subject: str) -> tuple[str, float]:
    path = RELATION_ROOT / f"{subject}.json"
    if not path.exists():
        return "unavailable", float("nan")
    payload = json.loads(path.read_text(encoding="utf-8"))
    relation = payload.get("axis_pair", {}).get("relation", {})
    if isinstance(relation, dict):
        return str(relation.get("relation", "unavailable")), float(
            relation.get("abs_cosine", np.nan)
        )
    return str(relation), float("nan")


def source_side_and_displacement(
    groups: np.ndarray,
    indices: np.ndarray,
    coords: np.ndarray,
    axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    projection = (coords - coords.mean(axis=0, keepdims=True)) @ axis
    span = float(np.ptp(projection))
    if span <= 0:
        raise ValueError("degenerate contact projection")
    sides = np.zeros(len(indices), dtype=np.int8)
    displacement = np.full(len(indices), np.nan, dtype=np.float64)
    for row, event_index in enumerate(indices):
        event = groups[event_index]
        source = event == 0
        later = event > 0
        source_position = float(np.mean(projection[source]))
        if source_position == 0 or not np.any(later):
            continue
        sides[row] = -1 if source_position < 0 else 1
        later_position = float(np.mean(projection[later]))
        displacement[row] = (
            -np.sign(source_position) * (later_position - source_position) / span
        )
    return sides, displacement


def load_v23_model(
    subject: str,
    seed: int,
    variant: str,
    record: dict[str, Any],
) -> torch.nn.Module:
    freeze = json.loads(
        (V23 / "development/DEVELOPMENT_FREEZE.json").read_text(encoding="utf-8")
    )
    node_logit = logit(estimate_node_hazard(record["groups"], record["train80"]))
    model = build_model(
        variant=variant,
        coords=record["coords"],
        axis=axis_for_subject(subject),
        node_logit=node_logit,
        rho_propagation=float(freeze["rho_propagation"]),
        rho_competition=float(freeze["rho_competition"]),
        device=torch.device("cpu"),
    )
    checkpoint = (
        V23
        / "formal/runs"
        / subject
        / f"seed_{seed}"
        / variant
        / "best.pt"
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def subset_nll(
    model: torch.nn.Module,
    groups: np.ndarray,
    counts: np.ndarray,
    indices: np.ndarray,
) -> float:
    if not len(indices):
        return float("nan")
    losses = []
    with torch.no_grad():
        for start in range(0, len(indices), 2048):
            batch = indices[start : start + 2048]
            result = model.forward_batch(groups[batch], counts[batch])
            losses.append(result.event_losses.detach().cpu().numpy())
    return float(np.mean(np.concatenate(losses)))


def run_bidirectional_audit() -> tuple[pd.DataFrame, dict[str, Any]]:
    status = json.loads(
        (V23 / "input_audit/INPUT_AUDIT_STATUS.json").read_text(encoding="utf-8")
    )
    patients = list(map(str, status["physical_axis_formal_patients"]))
    candidates = fibonacci_axes(32)
    rows: list[dict[str, Any]] = []
    for patient_index, subject in enumerate(patients):
        record = load_subject(subject)
        heldout = np.asarray(
            record["partitions"]["heldout20_sealed"], dtype=np.int64
        )
        selected_axis = axis_for_subject(subject)
        sides, displacement = source_side_and_displacement(
            record["groups"], heldout, record["coords"], selected_axis
        )
        side_indices = {
            side: heldout[(sides == side) & np.isfinite(displacement)]
            for side in (-1, 1)
        }
        side_disp = {
            side: float(np.nanmedian(displacement[sides == side]))
            for side in (-1, 1)
        }
        candidate_balanced = []
        for candidate in candidates:
            candidate_side, candidate_displacement = source_side_and_displacement(
                record["groups"], heldout, record["coords"], candidate
            )
            medians = []
            for side in (-1, 1):
                values = candidate_displacement[candidate_side == side]
                medians.append(
                    float(np.nanmedian(values)) if len(values) else np.nan
                )
            candidate_balanced.append(float(np.nanmin(medians)))
        selected_balanced_displacement = float(min(side_disp.values()))
        candidate_balanced = np.asarray(candidate_balanced, dtype=np.float64)

        per_seed = []
        for seed in SEEDS_V23:
            models = {
                "isotropic": load_v23_model(
                    subject, seed, "local_isotropic_two_state", record
                ),
                "axis_no_source": load_v23_model(
                    subject, seed, "axis_two_state_no_source", record
                ),
                "full": load_v23_model(
                    subject, seed, "axis_two_state_source_full", record
                ),
            }
            seed_row: dict[str, float] = {}
            for side in (-1, 1):
                indices = side_indices[side]
                nll = {
                    name: subset_nll(
                        model,
                        record["groups"],
                        record["counts"],
                        indices,
                    )
                    for name, model in models.items()
                }
                seed_row[f"axis_benefit_{side}"] = (
                    nll["isotropic"] - nll["axis_no_source"]
                )
                seed_row[f"source_benefit_{side}"] = (
                    nll["axis_no_source"] - nll["full"]
                )
                seed_row[f"full_over_isotropic_{side}"] = (
                    nll["isotropic"] - nll["full"]
                )
            per_seed.append(seed_row)
        relation, abs_cosine = relation_for_subject(subject)
        collapsed = {
            key: float(np.median([row[key] for row in per_seed]))
            for key in per_seed[0]
        }
        rows.append(
            {
                "subject": subject,
                "relation_descriptive": relation,
                "ab_abs_cosine_descriptive": abs_cosine,
                "n_heldout": int(len(heldout)),
                "n_source_negative": int(len(side_indices[-1])),
                "n_source_positive": int(len(side_indices[1])),
                "bilateral_min20_eligible": bool(
                    len(side_indices[-1]) >= 20 and len(side_indices[1]) >= 20
                ),
                "negative_source_inward_displacement": side_disp[-1],
                "positive_source_inward_displacement": side_disp[1],
                "balanced_inward_displacement": selected_balanced_displacement,
                "candidate_axis_median_balanced_displacement": float(
                    np.nanmedian(candidate_balanced)
                ),
                "selected_axis_displacement_margin": float(
                    selected_balanced_displacement
                    - np.nanmedian(candidate_balanced)
                ),
                "selected_axis_candidate_percentile": float(
                    np.nanmean(candidate_balanced <= selected_balanced_displacement)
                ),
                **collapsed,
                "balanced_axis_benefit": float(
                    min(
                        collapsed["axis_benefit_-1"],
                        collapsed["axis_benefit_1"],
                    )
                ),
                "balanced_source_benefit": float(
                    min(
                        collapsed["source_benefit_-1"],
                        collapsed["source_benefit_1"],
                    )
                ),
                "balanced_full_over_isotropic": float(
                    min(
                        collapsed["full_over_isotropic_-1"],
                        collapsed["full_over_isotropic_1"],
                    )
                ),
            }
        )
        print(
            f"bidirectional {patient_index + 1}/{len(patients)} {subject}",
            flush=True,
        )
    frame = pd.DataFrame(rows).sort_values("subject")
    eligible = frame.loc[frame.bilateral_min20_eligible].copy()
    metrics = {}
    for offset, key in enumerate(
        (
            "balanced_inward_displacement",
            "selected_axis_displacement_margin",
            "balanced_axis_benefit",
            "balanced_source_benefit",
            "balanced_full_over_isotropic",
        )
    ):
        metrics[key] = cohort_summary(
            eligible[key].to_numpy(float), 2026072801 + offset
        )
    result = {
        "contract": "topic5_ab_independent_bidirectional_axis_audit_v2_5",
        "status": "COMPLETE",
        "n_physical_axis_patients": len(frame),
        "n_bilateral_min20_eligible": len(eligible),
        "axis_definition": (
            "train80 transition-residual selection among 32 sign-free candidates"
        ),
        "ab_used_for_selection_or_primary_test": False,
        "primary_logic": (
            "both source sides must be informative; patient score is the weaker side"
        ),
        "metrics": metrics,
    }
    return frame, result


def fields_from_conditional_table(
    frame: pd.DataFrame, prefix: str
) -> dict[str, np.ndarray]:
    participation = frame[f"{prefix}_participation"].to_numpy(float)
    mean_rank = frame[f"{prefix}_mean_rank"].to_numpy(float)
    bins = np.column_stack(
        [frame[f"{prefix}_rank_bin_{index}"].to_numpy(float) for index in range(10)]
    )
    early = participation * np.sum(bins[:, :3], axis=1)
    late = participation * np.sum(bins[:, -3:], axis=1)
    return {
        "participation": participation,
        "early_joint_mass": early,
        "late_joint_mass": late,
        "endpoint_joint_mass": early + late,
        "weighted_earliness": participation * (1.0 - mean_rank),
    }


def fields_from_joint_distribution(distribution: np.ndarray) -> dict[str, np.ndarray]:
    distribution = np.asarray(distribution, dtype=np.float64)
    participation = 1.0 - distribution[:, 0]
    bins = distribution[:, 1:]
    centers = (np.arange(10, dtype=np.float64) + 0.5) / 10.0
    mean_joint_rank = bins @ centers
    early = np.sum(bins[:, :3], axis=1)
    late = np.sum(bins[:, -3:], axis=1)
    return {
        "participation": participation,
        "early_joint_mass": early,
        "late_joint_mass": late,
        "endpoint_joint_mass": early + late,
        "weighted_earliness": participation - mean_joint_rank,
    }


def strict_clinical_inventory() -> dict[str, list[int]]:
    frame = pd.read_csv(FIT1)
    strict = frame.loc[
        (frame.group_id == "strict_broadband")
        & (frame.time_reference == "clinical_onset")
    ]
    inventory = {
        str(subject): sorted(group.seizure_idx.astype(int).unique().tolist())
        for subject, group in strict.groupby("subject")
    }
    if len(inventory) != 16 or sum(map(len, inventory.values())) != 106:
        raise RuntimeError("strict clinical-onset 16/106 inventory drifted")
    return inventory


def load_target(
    subject: str, seizure_indices: list[int], model_names: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    with np.load(TARGET_ROOT / f"{subject}.npz", allow_pickle=False) as data:
        target_names = np.asarray(data["channels"]).astype(str)
        lookup = {name: index for index, name in enumerate(target_names)}
        keep_model = np.asarray(
            [index for index, name in enumerate(model_names) if name in lookup],
            dtype=np.int64,
        )
        keep_target = np.asarray(
            [lookup[model_names[index]] for index in keep_model], dtype=np.int64
        )
        targets = []
        used = []
        for seizure_index in seizure_indices:
            key = f"bb150_auc__{seizure_index}"
            if key not in data.files:
                continue
            values = np.asarray(data[key], dtype=np.float64)[keep_target]
            targets.append(values)
            used.append(seizure_index)
    matrix = np.stack(targets)
    finite = np.all(np.isfinite(matrix), axis=0)
    if np.count_nonzero(finite) < 6:
        raise ValueError(f"{subject}: fewer than six finite exact-joined contacts")
    return keep_model[finite], matrix[:, finite], used


def ordinary_model_fields(
    subject: str,
) -> tuple[np.ndarray, dict[str, list[dict[str, np.ndarray]]]]:
    by_model: dict[str, list[dict[str, np.ndarray]]] = {
        model: [] for model in ORDINARY_CONTROLS
    }
    names = None
    for seed_dir in SEED_DIRS:
        frame = pd.read_csv(
            RANK_ROOT / seed_dir / subject / "contact_rank_distributions.csv"
        )
        for model in ORDINARY_CONTROLS:
            subset = frame.loc[frame.control == model].sort_values("contact_index")
            current_names = subset.contact_name.astype(str).to_numpy()
            if names is None:
                names = current_names
            elif not np.array_equal(names, current_names):
                raise RuntimeError(f"{subject}: ordinary contact ordering drifted")
            prefix = "observed" if model == "empirical_rank_distribution" else "predicted"
            by_model[model].append(fields_from_conditional_table(subset, prefix))
    if names is None:
        raise RuntimeError(f"{subject}: no ordinary fields")
    return names, by_model


def low_rank_zero_fields(
    subject: str,
) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
    rows = []
    names = None
    for seed_dir in SEED_DIRS:
        frame = pd.read_csv(
            LOW_RANK_ROOT
            / seed_dir
            / "rank_0"
            / subject
            / "contact_rank_distributions.csv"
        ).sort_values("contact_index")
        current = frame.contact_name.astype(str).to_numpy()
        if names is None:
            names = current
        elif not np.array_equal(names, current):
            raise RuntimeError(f"{subject}: low-rank contact ordering drifted")
        rows.append(fields_from_conditional_table(frame, "predicted"))
    return names, rows


def structured_model_fields(
    subject: str,
) -> tuple[np.ndarray, dict[str, list[dict[str, np.ndarray]]]]:
    by_model: dict[str, list[dict[str, np.ndarray]]] = {
        model: [] for model in STRUCTURED_CONTROLS
    }
    record = load_subject(subject)
    names = None
    generated_root = OUT / "structured_axis_no_source_representations"
    generated_root.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS_V23:
        path = V24 / "representations/per_seed" / f"{subject}_seed{seed}.npz"
        with np.load(path, allow_pickle=False) as data:
            current_names = np.asarray(data["contact_names"]).astype(str)
            arrays = {
                "structured_empirical_train80": np.asarray(
                    data["empirical_train80"], dtype=np.float64
                ),
                "structured_full": np.asarray(
                    data["full_fixed_axis"], dtype=np.float64
                ),
                "structured_no_history": np.asarray(
                    data["no_history"], dtype=np.float64
                ),
                "structured_local_isotropic": np.asarray(
                    data["local_isotropic"], dtype=np.float64
                ),
                "structured_node_only": np.asarray(
                    data["node_only"], dtype=np.float64
                ),
            }
        if names is None:
            names = current_names
        elif not np.array_equal(names, current_names):
            raise RuntimeError(f"{subject}: structured contact ordering drifted")
        no_source_path = generated_root / f"{subject}_seed{seed}.npz"
        if no_source_path.exists():
            with np.load(no_source_path, allow_pickle=False) as data:
                no_source = np.asarray(data["distribution"], dtype=np.float64)
        else:
            model = load_v23_model(
                subject, seed, "axis_two_state_no_source", record
            )
            sampled, uniforms = paired_rollout_design(
                record["groups"],
                record["train80"],
                n_rollouts=5000,
                seed=240000 + seed,
            )
            no_source = rollout_model_distribution(
                model, record["groups"], sampled, uniforms
            )
            np.savez_compressed(
                no_source_path,
                contact_names=current_names,
                distribution=no_source.astype(np.float32),
            )
        arrays["structured_axis_no_source"] = no_source
        for model, distribution in arrays.items():
            by_model[model].append(fields_from_joint_distribution(distribution))
    if names is None:
        raise RuntimeError(f"{subject}: no structured fields")
    return names, by_model


def centered_rank(values: np.ndarray) -> np.ndarray:
    ranked = rankdata(values).astype(np.float64)
    return ranked - ranked.mean()


def score_model(
    field_seeds: list[dict[str, np.ndarray]],
    target: np.ndarray,
    permutations: np.ndarray,
) -> tuple[dict[str, float], float, np.ndarray]:
    target_rank = np.vstack([centered_rank(row) for row in target])
    target_norm = np.linalg.norm(target_rank, axis=1)
    observed_by_seed = []
    signed_by_seed: dict[str, list[float]] = {field: [] for field in FIELDS}
    for seed_fields in field_seeds:
        field_rank = np.column_stack(
            [centered_rank(seed_fields[field]) for field in FIELDS]
        )
        denominator = target_norm[:, None] * np.linalg.norm(field_rank, axis=0)[None, :]
        observed = (target_rank @ field_rank) / denominator
        for field_index, field in enumerate(FIELDS):
            signed_by_seed[field].append(float(np.median(observed[:, field_index])))
        observed_by_seed.append(float(np.median(np.max(np.abs(observed), axis=1))))
    observed_omnibus = float(np.median(observed_by_seed))
    field_summary = {
        field: float(np.median(values)) for field, values in signed_by_seed.items()
    }

    null_by_seed = []
    for seed_fields in field_seeds:
        field_rank = np.column_stack(
            [centered_rank(seed_fields[field]) for field in FIELDS]
        )
        field_norm = np.linalg.norm(field_rank, axis=0)
        per_seizure = []
        for seizure_index in range(len(target)):
            permuted_target = target_rank[seizure_index][permutations]
            correlations = (
                permuted_target @ field_rank
            ) / (target_norm[seizure_index] * field_norm[None, :])
            per_seizure.append(np.max(np.abs(correlations), axis=1))
        null_by_seed.append(np.median(np.column_stack(per_seizure), axis=1))
    null = np.median(np.column_stack(null_by_seed), axis=1)
    return field_summary, observed_omnibus, null


def run_static_transfer() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    inventory = strict_clinical_inventory()
    structured_subjects = {
        path.stem for path in (V24 / "representations/per_subject").glob("*.npz")
    }
    metric_rows = []
    field_rows = []
    for patient_index, (subject, seizure_indices) in enumerate(inventory.items()):
        ordinary_names, ordinary = ordinary_model_fields(subject)
        keep, target, used = load_target(subject, seizure_indices, ordinary_names)
        relation, abs_cosine = relation_for_subject(subject)
        models: dict[str, list[dict[str, np.ndarray]]] = {
            model: [
                {field: seed[field][keep] for field in FIELDS}
                for seed in seed_fields
            ]
            for model, seed_fields in ordinary.items()
        }
        if subject in structured_subjects:
            structured_names, structured = structured_model_fields(subject)
            if not np.array_equal(ordinary_names, structured_names):
                raise RuntimeError(f"{subject}: ordinary/structured contact mismatch")
            for model, seed_fields in structured.items():
                models[model] = [
                    {field: seed[field][keep] for field in FIELDS}
                    for seed in seed_fields
                ]
        rng = np.random.default_rng(2026072800 + patient_index)
        permutations = np.stack(
            [rng.permutation(len(keep)) for _ in range(N_PERM)]
        )
        for model, field_seeds in models.items():
            field_summary, observed, null = score_model(
                field_seeds, target, permutations
            )
            metric_rows.append(
                {
                    "subject": subject,
                    "model": model,
                    "relation_descriptive": relation,
                    "ab_abs_cosine_descriptive": abs_cosine,
                    "n_contacts": int(len(keep)),
                    "n_seizures": int(len(used)),
                    "observed_max_abs_rho": observed,
                    "all_contact_null_median": float(np.median(null)),
                    "all_contact_margin": float(observed - np.median(null)),
                    "all_contact_empirical_p": float(
                        (1 + np.count_nonzero(null >= observed)) / (N_PERM + 1)
                    ),
                }
            )
            for field, signed in field_summary.items():
                field_rows.append(
                    {
                        "subject": subject,
                        "model": model,
                        "field": field,
                        "signed_rho": signed,
                        "absolute_rho": abs(signed),
                        "n_contacts": int(len(keep)),
                        "n_seizures": int(len(used)),
                    }
                )
        print(
            f"static {patient_index + 1}/{len(inventory)} {subject}",
            flush=True,
        )
    metrics = pd.DataFrame(metric_rows).sort_values(["model", "subject"])
    fields = pd.DataFrame(field_rows).sort_values(["model", "field", "subject"])
    summaries = {}
    for model, group in metrics.groupby("model"):
        summaries[model] = {
            "absolute_similarity": cohort_summary(
                group.observed_max_abs_rho.to_numpy(float),
                2026072900 + len(summaries) * 2,
            ),
            "all_contact_margin": cohort_summary(
                group.all_contact_margin.to_numpy(float),
                2026072901 + len(summaries) * 2,
            ),
        }
    common = metrics.loc[
        metrics.model.isin(
            (
                "empirical_rank_distribution",
                "full_history_gru",
                "structured_full",
            )
        )
    ].pivot(index="subject", columns="model", values="observed_max_abs_rho")
    common = common.dropna()
    paired = {}
    if len(common):
        for left, right, key in (
            (
                "structured_full",
                "full_history_gru",
                "structured_minus_full_history",
            ),
            (
                "full_history_gru",
                "empirical_rank_distribution",
                "full_history_minus_empirical",
            ),
            (
                "structured_full",
                "empirical_rank_distribution",
                "structured_minus_empirical",
            ),
        ):
            paired[key] = cohort_summary(
                (common[left] - common[right]).to_numpy(float),
                2026073000 + len(paired),
            )
    wide_observed = metrics.pivot(
        index="subject", columns="model", values="observed_max_abs_rho"
    )
    wide_margin = metrics.pivot(
        index="subject", columns="model", values="all_contact_margin"
    )
    comparison_pairs = (
        ("full_history_gru", "static_contact_hazard"),
        ("full_history_gru", "unordered_prefix"),
        ("full_history_gru", "last_set_first_order"),
        ("full_history_gru", "rank_shuffle_gru"),
        ("full_history_gru", "empirical_rank_distribution"),
        ("structured_full", "structured_no_history"),
        ("structured_axis_no_source", "structured_local_isotropic"),
        ("structured_full", "structured_axis_no_source"),
        ("structured_full", "structured_node_only"),
    )
    comparison_rows = []
    for metric_name, wide in (
        ("observed_max_abs_rho", wide_observed),
        ("all_contact_margin", wide_margin),
    ):
        family_start = len(comparison_rows)
        for left, right in comparison_pairs:
            difference = (wide[left] - wide[right]).dropna()
            summary = cohort_summary(
                difference.to_numpy(float),
                2026073100 + len(comparison_rows),
            )
            comparison_rows.append(
                {
                    "metric": metric_name,
                    "left": left,
                    "right": right,
                    **summary,
                }
            )
        ordinary_indices = list(range(family_start, family_start + 5))
        q_values = bh_fdr(
            [comparison_rows[index]["wilcoxon_greater_p"] for index in ordinary_indices]
        )
        for index, q_value in zip(ordinary_indices, q_values):
            comparison_rows[index]["ordinary_family_bh_fdr_q"] = q_value
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(OUT / "static_transfer_paired_comparisons.csv", index=False)
    result = {
        "contract": "topic5_cross_model_static_transfer_v2_5",
        "status": "COMPLETE",
        "strict_clinical_onset_cohort": {
            "n_patients": len(inventory),
            "n_seizures": int(sum(map(len, inventory.values()))),
            "dataset": "epilepsiae",
            "time_reference": "clinical_onset",
            "window_sec": [0, 10],
            "band_hz": [1, 150],
        },
        "structured_common_cohort_n": int(len(common)),
        "score": (
            "per-seizure maximum absolute Spearman across five frozen rank fields; "
            "seed median then patient-first"
        ),
        "primary_null": (
            "5000 coherent within-patient all-contact permutations with field "
            "reselection in every draw"
        ),
        "fields": list(FIELDS),
        "model_summaries": summaries,
        "paired_common_cohort": paired,
        "paired_model_comparisons": comparison_rows,
        "yuquan_eeg_onset_mixed_into_primary": False,
        "dynamic_source_conditioned_transfer": (
            "BLOCKED_MISSING_EXACT_PER_SEIZURE_CLINICAL_ONSET_CONTACT_SETS"
        ),
    }
    return metrics, fields, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section",
        choices=("all", "bidirectional", "static"),
        default="all",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    bidirectional_summary = None
    static_summary = None
    if args.section in ("all", "bidirectional"):
        bidirectional_frame, bidirectional_summary = run_bidirectional_audit()
        bidirectional_frame.to_csv(
            OUT / "bidirectional_patient_metrics.csv", index=False
        )
        atomic_json(OUT / "BIDIRECTIONAL_SUMMARY.json", bidirectional_summary)
    if args.section in ("all", "static"):
        static_metrics, static_fields, static_summary = run_static_transfer()
        static_metrics.to_csv(
            OUT / "static_transfer_patient_metrics.csv", index=False
        )
        static_fields.to_csv(
            OUT / "static_transfer_field_metrics.csv", index=False
        )
        atomic_json(OUT / "STATIC_TRANSFER_SUMMARY.json", static_summary)
    if args.section == "all":
        atomic_json(
            OUT / "RUN_STATUS.json",
            {
                "contract": "topic5_rnn_bidirectional_cross_model_audit_v2_5",
                "status": "COMPLETE",
                "bidirectional": bidirectional_summary,
                "static_transfer": static_summary,
            },
        )
    print(
        json.dumps(
            {
                "bidirectional": bidirectional_summary,
                "static": static_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
