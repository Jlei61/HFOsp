#!/usr/bin/env python3
"""Full-cohort patient-within RNN field transfer on the canonical Fig. 3 target.

The recurrent models are never trained here.  Stage ``freeze`` reads the 34 x 3
converged interictal-only rollouts, freezes two model-generated propagation
fields for every Fig. 3 patient, and writes a manifest before any ictal cache is
opened.  Stage ``score`` then applies the canonical Fig. 3 event inventory and
all-contact channel-shuffle scorer to all 17 patients / 167 seizures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_constructive_readback import (  # noqa: E402
    evaluate_mode_readback,
    fit_train_mode_readback,
)
from src.topic5_t0_features import window_activation  # noqa: E402
from src.topic5_template_axis_field import make_field_scorer, z_earliness  # noqa: E402
from src.topic5_tspectral_field_concordance import (  # noqa: E402
    exact_name_align_matrix,
    fold_seizure_null_draws,
    make_contact_permutations,
    paired_sign_flip_p,
    score_observed_bundle,
    score_permutation_matrix,
)


CONTRACT = "topic5_rnn_full_cohort_field_transfer_v0_1"
SEEDS = (20260725, 20260726, 20260727)
N_PERM = 1000
BASE_SEED = 20260811


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(dict(payload)), ensure_ascii=False, indent=2) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_seed(text: str, base_seed: int = BASE_SEED) -> int:
    digest = hashlib.sha256(f"{base_seed}:{text}".encode()).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def _safe_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    a, b = np.asarray(left, float), np.asarray(right, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return float("nan")
    return float(spearmanr(a[ok], b[ok]).statistic)


def _required_paths(canonical_root: Path) -> dict[str, Path]:
    return {
        "dataset": canonical_root / "results/topic5_interictal_rank_distribution/dataset_v0_4/per_subject",
        "rollouts": canonical_root / "results/topic5_rnn_training_sufficiency_v0_1/formal/converged_teacher_forced",
        "metrics": canonical_root / "results/topic5_rnn_training_sufficiency_v0_1/analysis/d_patient_metrics.csv",
        "fields": canonical_root / "results/interictal_propagation_masked/template_gradient_fields/per_subject",
        "fig3_events": canonical_root / "results/topic5_ictal_recruitment/tspectral_field_concordance/clinical_onset_gradient_field_cohort_stat_event.csv",
        "fig3_subjects": canonical_root / "results/topic5_ictal_recruitment/tspectral_field_concordance/clinical_onset_gradient_field_cohort_stat_subject.csv",
        "bb": canonical_root / "results/topic5_ictal_recruitment/t0_feature_cache_bb150_1_150",
        "gamma": canonical_root / "results/topic5_ictal_recruitment/v2_band_scan/cache",
    }


def audit_inputs(canonical_root: Path) -> dict[str, object]:
    paths = _required_paths(canonical_root)
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing_{label}:{path}")
    subjects = sorted(path.stem for path in paths["dataset"].glob("*.npz"))
    if len(subjects) != 34:
        raise RuntimeError(f"interictal_subjects_must_equal_34:{len(subjects)}")
    missing = []
    for subject in subjects:
        for seed in SEEDS:
            path = paths["rollouts"] / f"seed_{seed}" / subject / "rollouts.npz"
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"missing_converged_rollouts:{len(missing)}:{missing[:3]}")
    events = pd.read_csv(paths["fig3_events"])
    pooled = events[events.group_id == "all_phenotype_matched"].copy()
    if pooled.subject.nunique() != 17 or len(pooled) != 167:
        raise RuntimeError(
            f"fig3_primary_must_equal_17_167:{pooled.subject.nunique()}_{len(pooled)}"
        )
    if "outer_" in str(paths["fig3_events"]):
        raise RuntimeError("legacy_outer_cache_is_forbidden")
    return {
        "interictal_subjects": subjects,
        "n_interictal_subjects": 34,
        "n_model_seeds_per_subject": 3,
        "primary_ictal_subjects": sorted(pooled.subject.unique().tolist()),
        "n_primary_ictal_subjects": 17,
        "n_primary_ictal_seizures": 167,
        "legacy_outer_cache_reads": 0,
        "target_used_for_training_or_selection": False,
        "source_paths": {key: str(value) for key, value in paths.items()},
        "source_hashes": {
            "fig3_event_inventory": _sha256(paths["fig3_events"]),
            "fig3_subject_inventory": _sha256(paths["fig3_subjects"]),
            "interictal_metric_table": _sha256(paths["metrics"]),
        },
    }


def _mode_to_ab_mapping(
    train_templates: np.ndarray,
    dataset_names: np.ndarray,
    field_record: Mapping[str, object],
) -> tuple[dict[str, int], np.ndarray]:
    field = field_record["interictal_field"]
    field_names = [str(v) for v in field["contact_order"]]
    index = {str(name): i for i, name in enumerate(dataset_names)}
    if any(name not in index for name in field_names):
        raise ValueError("frozen_field_contacts_missing_from_rnn_dataset")
    take = np.asarray([index[name] for name in field_names], int)
    empirical = (np.asarray(field["rank_a"], float), np.asarray(field["rank_b"], float))
    correlation = np.asarray([
        [_safe_spearman(train_templates[mode, take], empirical[target]) for target in range(2)]
        for mode in range(2)
    ])
    row, col = linear_sum_assignment(-np.nan_to_num(correlation, nan=-2.0))
    target_to_mode = {int(target): int(mode) for mode, target in zip(row, col)}
    if set(target_to_mode) != {0, 1}:
        raise RuntimeError("mode_to_ab_assignment_failed")
    return {"a": target_to_mode[0], "b": target_to_mode[1]}, correlation


def _support_by_mode(groups: np.ndarray, labels: np.ndarray, mode: int) -> np.ndarray:
    chosen = np.asarray(groups)[np.asarray(labels) == int(mode)]
    if chosen.shape[0] == 0:
        return np.zeros(groups.shape[1], float)
    return np.mean(chosen >= 0, axis=0)


def _freeze_subject_field(
    subject: str,
    canonical_root: Path,
    out_dir: Path,
) -> dict[str, object]:
    paths = _required_paths(canonical_root)
    dataset_path = paths["dataset"] / f"{subject}.npz"
    field_path = paths["fields"] / f"{subject}.json"
    record = json.loads(field_path.read_text())
    field = record.get("interictal_field") or {}
    if field.get("status") != "ok":
        raise ValueError(f"empirical_field_unavailable:{subject}:{field.get('status')}")
    with np.load(dataset_path, allow_pickle=True) as dataset:
        groups = np.asarray(dataset["event_group_ids"], int)
        split = np.asarray(dataset["event_split"], int)
        names = np.asarray(dataset["contact_names"], str)
    readback = fit_train_mode_readback(groups[split == 0], random_state=0)
    mapping, match_correlation = _mode_to_ab_mapping(
        readback.templates, names, record
    )
    field_names = np.asarray(field["contact_order"], str)
    name_index = {str(name): i for i, name in enumerate(names)}
    take = np.asarray([name_index[str(name)] for name in field_names], int)
    rank_by_label: dict[str, list[np.ndarray]] = {"a": [], "b": []}
    support_by_label: dict[str, list[np.ndarray]] = {"a": [], "b": []}
    seed_rows = []
    for seed in SEEDS:
        rollout_path = paths["rollouts"] / f"seed_{seed}" / subject / "rollouts.npz"
        with np.load(rollout_path, allow_pickle=True) as rollout:
            native = np.asarray(rollout["native_model__event_group_ids"], int)
        evaluated = evaluate_mode_readback(readback, native)
        labels = np.asarray(evaluated["labels"], int)
        templates = np.asarray(evaluated["templates"], float)
        for label in ("a", "b"):
            mode = int(mapping[label])
            rank_by_label[label].append(templates[mode, take])
            support_by_label[label].append(_support_by_mode(native, labels, mode)[take])
        seed_rows.append({
            "seed": seed,
            "n_rollout_events": int(len(native)),
            "mode1_fraction": float(evaluated["mode1_fraction"]),
            "template_match_to_train": float(evaluated["template_match_to_train"]),
            "rollout_sha256": _sha256(rollout_path),
        })
    with np.errstate(invalid="ignore"):
        rank_a = np.nanmean(np.stack(rank_by_label["a"]), axis=0)
        rank_b = np.nanmean(np.stack(rank_by_label["b"]), axis=0)
    support_a = np.mean(np.stack(support_by_label["a"]), axis=0)
    support_b = np.mean(np.stack(support_by_label["b"]), axis=0)
    models = field.get("field_models") or {}
    planes = field.get("planes") or {}
    if all(key in models for key in ("shared_a", "shared_b")):
        route, score_key = "shared", "shared_maxab"
        plane_a = plane_b = planes["shared"]
        scorer_keys = ("shared_a", "shared_b")
    else:
        route, score_key = "own_fallback", "own_maxab"
        plane_a, plane_b = planes["own_a"], planes["own_b"]
        scorer_keys = ("own_a", "own_b")
    scorer_a = make_field_scorer(
        z_earliness(rank_a), np.asarray(plane_a["points"], float), support_a,
        float(plane_a["sigma"]),
    )
    scorer_b = make_field_scorer(
        z_earliness(rank_b), np.asarray(plane_b["points"], float), support_b,
        float(plane_b["sigma"]),
    )
    target = out_dir / "model_fields" / f"{subject}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        contact_order=field_names,
        rank_a=rank_a,
        rank_b=rank_b,
        support_a=support_a,
        support_b=support_b,
        route=np.asarray(route),
        score_key=np.asarray(score_key),
        scorer_key_a=np.asarray(scorer_keys[0]),
        scorer_key_b=np.asarray(scorer_keys[1]),
        template_field_a=np.asarray(scorer_a["template_field"], float),
        template_field_b=np.asarray(scorer_b["template_field"], float),
        points_a=np.asarray(scorer_a["points"], float),
        points_b=np.asarray(scorer_b["points"], float),
        support_scorer_a=np.asarray(scorer_a["support"], float),
        support_scorer_b=np.asarray(scorer_b["support"], float),
        sigma_a=np.asarray(float(scorer_a["sigma"])),
        sigma_b=np.asarray(float(scorer_b["sigma"])),
        weight_id_a=np.asarray(scorer_a["weight_id"], float),
        weight_id_b=np.asarray(scorer_b["weight_id"], float),
        weight_mirror_a=np.asarray(scorer_a["weight_mirror"], float),
        weight_mirror_b=np.asarray(scorer_b["weight_mirror"], float),
    )
    return {
        "subject": subject,
        "dataset": subject.split("_", 1)[0],
        "route": route,
        "score_key": score_key,
        "n_contacts": int(len(field_names)),
        "readback_reliable_diagnostic": bool(readback.reliable),
        "readback_silhouette": float(readback.silhouette),
        "readback_cross_half_ari": float(readback.cross_half_ari),
        "readback_minimum_cluster_fraction": float(readback.minimum_cluster_fraction),
        "mode_to_ab": mapping,
        "mode_empirical_rank_correlation": match_correlation,
        "seed_rows": seed_rows,
        "field_npz": str(target),
        "field_npz_sha256": _sha256(target),
        "empirical_field_sha256": _sha256(field_path),
        "target_values_read": False,
    }


def _interictal_summary(canonical_root: Path, out_dir: Path) -> dict[str, object]:
    metrics = pd.read_csv(_required_paths(canonical_root)["metrics"])
    frame = metrics[
        (metrics.condition == "converged_teacher_forced")
        & (metrics.rollout_condition.isin(["native_model", "static_only"]))
        & (metrics.endpoint == "transition_correlation")
    ]
    wide = frame.pivot(index=["subject", "dataset"], columns="rollout_condition", values="value").reset_index()
    if len(wide) != 34 or wide[["native_model", "static_only"]].isna().any().any():
        raise RuntimeError("interictal_metric_denominator_must_equal_34")
    wide["native_minus_static"] = wide.native_model - wide.static_only
    wide.to_csv(out_dir / "interictal_patient_statistics.csv", index=False)
    diff = wide.native_minus_static.to_numpy(float)
    return {
        "n_subjects": 34,
        "endpoint": "heldout native-model transition correlation",
        "native_median": float(np.median(wide.native_model)),
        "static_median": float(np.median(wide.static_only)),
        "paired_difference_median": float(np.median(diff)),
        "n_native_gt_static": int(np.sum(diff > 0)),
        "wilcoxon_one_sided_native_gt_static_p": float(
            wilcoxon(wide.native_model, wide.static_only, alternative="greater").pvalue
        ),
    }


def freeze(canonical_root: Path, out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_inputs(canonical_root)
    _write_json(out_dir / "INPUT_AUDIT.json", audit)
    rows = [
        _freeze_subject_field(subject, canonical_root, out_dir)
        for subject in audit["primary_ictal_subjects"]
    ]
    if len(rows) != 17 or any(row["target_values_read"] for row in rows):
        raise RuntimeError("model_field_manifest_must_cover_17_before_target_access")
    interictal = _interictal_summary(canonical_root, out_dir)
    manifest = {
        "contract": CONTRACT,
        "stage": "target_free_model_field_freeze",
        "target_access_count": 0,
        "model_field_subjects_before_target_read": len(rows),
        "interictal": interictal,
        "subjects": rows,
        "assertions": {
            "interictal_subjects_eq_34": audit["n_interictal_subjects"] == 34,
            "model_seeds_per_subject_eq_3": audit["n_model_seeds_per_subject"] == 3,
            "model_field_subjects_eq_17": len(rows) == 17,
            "legacy_outer_cache_reads_eq_0": audit["legacy_outer_cache_reads"] == 0,
            "target_used_for_training_or_selection": False,
        },
    }
    _write_json(out_dir / "MODEL_FIELD_MANIFEST.json", manifest)
    _write_json(out_dir / "FREEZE_DONE.json", {"status": "complete", **manifest["assertions"]})
    return manifest


def _load_model_field(path: Path) -> tuple[dict, dict, str, str]:
    with np.load(path, allow_pickle=True) as z:
        order = np.asarray(z["contact_order"], str)
        key_a, key_b = str(z["scorer_key_a"]), str(z["scorer_key_b"])
        scorer_a = {
            "template_field": np.asarray(z["template_field_a"], float),
            "points": np.asarray(z["points_a"], float),
            "support": np.asarray(z["support_scorer_a"], float),
            "sigma": float(z["sigma_a"]),
            "weight_id": np.asarray(z["weight_id_a"], float),
            "weight_mirror": np.asarray(z["weight_mirror_a"], float),
        }
        scorer_b = {
            "template_field": np.asarray(z["template_field_b"], float),
            "points": np.asarray(z["points_b"], float),
            "support": np.asarray(z["support_scorer_b"], float),
            "sigma": float(z["sigma_b"]),
            "weight_id": np.asarray(z["weight_id_b"], float),
            "weight_mirror": np.asarray(z["weight_mirror_b"], float),
        }
        score_key = str(z["score_key"])
    record = {"interictal_field": {"contact_order": order.tolist()}}
    route = "shared" if score_key.startswith("shared_") else "own_fallback"
    return record, {key_a: scorer_a, key_b: scorer_b}, score_key, route


def _load_activation(
    canonical_root: Path,
    subject: str,
    seizure_idx: int,
    phenotype: str,
) -> tuple[list[str], np.ndarray]:
    paths = _required_paths(canonical_root)
    if phenotype == "strict_broadband":
        meta_path, npz_path = paths["bb"] / f"{subject}.json", paths["bb"] / f"{subject}.npz"
        meta = json.loads(meta_path.read_text())
        with np.load(npz_path, allow_pickle=True) as cache:
            names = [str(v) for v in meta.get("channels", cache["channels"].tolist())]
            activation = np.asarray(cache[f"bb150_auc__{seizure_idx}"], float)
        return names, activation
    if phenotype == "gamma_nonbroadband":
        meta_path, npz_path = paths["gamma"] / f"{subject}.json", paths["gamma"] / f"{subject}.npz"
        meta = json.loads(meta_path.read_text())
        with np.load(npz_path, allow_pickle=True) as cache:
            names = [str(v) for v in meta.get("channels", cache["channels"].tolist())]
            activation = window_activation(
                np.asarray(cache[f"gamma_LVFA__zt__{seizure_idx}"], float),
                np.asarray(cache[f"gamma_LVFA__relt__{seizure_idx}"], float),
                0.0, 10.0,
            )
        return names, np.asarray(activation, float)
    raise ValueError(f"unknown_phenotype:{phenotype}")


def _fold_subject_rows(group: pd.DataFrame, nulls: Mapping[tuple, np.ndarray]) -> dict[str, object]:
    observed = group.observed.to_numpy(float)
    arrays = [nulls[(str(row.subject), int(row.seizure_idx), str(row.band))][:, None]
              for row in group.itertuples()]
    folded = fold_seizure_null_draws(arrays)[:, 0]
    data = float(np.median(observed))
    null = float(np.median(folded))
    first = group.iloc[0]
    return {
        "dataset": first.dataset,
        "subject": first.subject,
        "group_id": first.group_id,
        "data": data,
        "channel_null_median": null,
        "channel_null_p95": float(np.percentile(folded, 95)),
        "margin": data - null,
        "n_seizures": int(len(group)),
        "seizure_idxs": ";".join(map(str, sorted(group.seizure_idx.astype(int)))),
        "n_channel_shuffle_draws": int(len(folded)),
        "field_plane": first.field_plane,
        "score_key": first.score_key,
        "subject_empirical_one_sided_p": float((1 + np.sum(folded >= data - 1e-15)) / (len(folded) + 1)),
    }


def score(canonical_root: Path, out_dir: Path, n_perm: int = N_PERM) -> dict[str, object]:
    manifest_path = out_dir / "MODEL_FIELD_MANIFEST.json"
    if not manifest_path.exists():
        raise RuntimeError("target_free_model_field_manifest_missing")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("target_access_count") != 0 or manifest.get("model_field_subjects_before_target_read") != 17:
        raise RuntimeError("invalid_pre_target_manifest")
    canonical_events = pd.read_csv(_required_paths(canonical_root)["fig3_events"])
    pooled = canonical_events[canonical_events.group_id == "all_phenotype_matched"]
    if pooled.subject.nunique() != 17 or len(pooled) != 167:
        raise RuntimeError("canonical_target_denominator_changed")
    event_rows, nulls = [], {}
    e1146_aligned = []
    e1146_order: np.ndarray | None = None
    for subject in sorted(pooled.subject.unique()):
        field_path = out_dir / "model_fields" / f"{subject}.npz"
        record, scorers, score_key, route_prefix = _load_model_field(field_path)
        target_names = [str(v) for v in record["interictal_field"]["contact_order"]]
        if subject == "epilepsiae_1146":
            e1146_order = np.asarray(target_names, str)
        subject_events = canonical_events[canonical_events.subject == subject]
        unique_events = subject_events.drop_duplicates(["subject", "seizure_idx", "band"])
        score_cache = {}
        for row in unique_events.itertuples():
            names, activation = _load_activation(
                canonical_root, subject, int(row.seizure_idx), str(row.phenotype)
            )
            aligned = exact_name_align_matrix(
                record, names, np.asarray(activation, float)[:, None]
            )["values"][:, 0]
            finite = np.isfinite(aligned)
            if int(finite.sum()) < 6:
                raise RuntimeError(f"fewer_than_6_exact_contacts:{subject}:{row.seizure_idx}")
            observed_bundle = score_observed_bundle(scorers, aligned)
            observed = float(observed_bundle[score_key])
            # Reuse the exact Fig. 3 event seed, so every model sees the same
            # all-contact permutations as the empirical field.
            matching = pooled[(pooled.subject == subject) & (pooled.seizure_idx == int(row.seizure_idx))]
            if len(matching) != 1:
                raise RuntimeError(f"missing_unique_primary_event_seed:{subject}:{row.seizure_idx}")
            perm_seed = int(matching.iloc[0].permutation_seed)
            permutations = make_contact_permutations(
                target_names, finite, n_perm, perm_seed, mode="all_contact"
            )
            null = score_permutation_matrix(
                scorers, aligned[None, :], permutations, chunk_draws=100
            )[score_key][:, 0]
            cache_key = (subject, int(row.seizure_idx), str(row.band))
            score_cache[cache_key] = (observed, null, aligned)
            nulls[cache_key] = null
            if subject == "epilepsiae_1146":
                e1146_aligned.append(aligned)
        for row in subject_events.itertuples():
            key = (subject, int(row.seizure_idx), str(row.band))
            observed, null, _ = score_cache[key]
            event_rows.append({
                "dataset": row.dataset,
                "subject": subject,
                "seizure_idx": int(row.seizure_idx),
                "group_id": row.group_id,
                "phenotype": row.phenotype,
                "band": row.band,
                "time_reference": row.time_reference,
                "field_plane": route_prefix,
                "score_key": score_key,
                "observed": observed,
                "null_median": float(np.median(null)),
                "null_p95": float(np.percentile(null, 95)),
                "margin": observed - float(np.median(null)),
                "permutation_seed": int(matching.iloc[0].permutation_seed) if False else int(
                    pooled[(pooled.subject == subject) & (pooled.seizure_idx == int(row.seizure_idx))].iloc[0].permutation_seed
                ),
            })
    events = pd.DataFrame(event_rows).sort_values(["group_id", "subject", "seizure_idx"])
    subjects = pd.DataFrame([
        _fold_subject_rows(group, nulls)
        for (_, _), group in events.groupby(["group_id", "subject"], sort=True)
    ])
    cohort_rows = []
    for group_id, frame in subjects.groupby("group_id", sort=True):
        data = frame.data.to_numpy(float)
        null = frame.channel_null_median.to_numpy(float)
        margin = data - null
        cohort_rows.append({
            "group_id": group_id,
            "n_subjects": int(len(frame)),
            "n_seizures": int(frame.n_seizures.sum()),
            "data_median": float(np.median(data)),
            "null_median": float(np.median(null)),
            "margin_median": float(np.median(margin)),
            "n_data_gt_null": int(np.sum(margin > 0)),
            "wilcoxon_one_sided_data_gt_null_p": float(
                wilcoxon(data, null, alternative="greater").pvalue
            ),
            "two_sided_subject_sign_flip_p": float(
                paired_sign_flip_p(margin, n_perm=100000, seed=_stable_seed(group_id))
            ),
        })
    cohort = pd.DataFrame(cohort_rows)
    primary = cohort[cohort.group_id == "all_phenotype_matched"]
    if len(primary) != 1 or int(primary.iloc[0].n_subjects) != 17 or int(primary.iloc[0].n_seizures) != 167:
        raise RuntimeError("scored_primary_denominator_must_equal_17_167")
    events.to_csv(out_dir / "ictal_event_statistics.csv", index=False)
    subjects.to_csv(out_dir / "ictal_patient_statistics.csv", index=False)
    cohort.to_csv(out_dir / "ictal_cohort_statistics.csv", index=False)
    if len(e1146_aligned) != 17 or e1146_order is None:
        raise RuntimeError(f"e1146_primary_events_must_equal_17:{len(e1146_aligned)}")
    np.savez_compressed(
        out_dir / "e1146_early_ictal_activation.npz",
        contact_order=e1146_order,
        activation=np.nanmedian(np.stack(e1146_aligned), axis=0),
        n_seizures=np.asarray(len(e1146_aligned)),
    )
    summary = {
        "contract": CONTRACT,
        "stage": "canonical_fig3_target_score",
        "target_used_for_training_or_selection": False,
        "target_unsealed_after_model_field_manifest": True,
        "null": "1000 synchronized all-contact channel shuffles; mirror and maxAB reselected per draw",
        "primary": primary.iloc[0].to_dict(),
        "all_groups": cohort.to_dict("records"),
    }
    _write_json(out_dir / "SCORE_SUMMARY.json", summary)
    _write_json(out_dir / "SCORE_DONE.json", {"status": "complete", "n_primary_subjects": 17, "n_primary_seizures": 167})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", type=Path, default=Path("/home/honglab/leijiaxin/HFOsp"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/topic5_rnn_full_cohort_field_transfer_v0_1")
    parser.add_argument("--stage", choices=("audit", "freeze", "score", "all"), default="all")
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    args = parser.parse_args()
    if args.n_perm < N_PERM:
        raise ValueError(f"n_perm_must_be_at_least_{N_PERM}")
    if args.stage == "audit":
        _write_json(args.output_dir / "INPUT_AUDIT.json", audit_inputs(args.canonical_root))
    if args.stage in {"freeze", "all"}:
        freeze(args.canonical_root, args.output_dir)
    if args.stage in {"score", "all"}:
        score(args.canonical_root, args.output_dir, args.n_perm)


if __name__ == "__main__":
    main()
