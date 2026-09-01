#!/usr/bin/env python3
"""Freeze full-data Timing+Space interictal templates for Figure 3 refresh.

The held-out spatial-information analysis establishes whether geometry adds
directional information.  This producer then refits the same hybrid KMeans on
the complete deterministic QC-clean interictal event sample and freezes the
resulting labels, rank templates, supports, axes and fields.  It never reads
seizure, onset or ictal-energy data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.interictal_propagation import (  # noqa: E402
    assign_events_to_templates,
    load_subject_propagation_events,
)
from src.interictal_spatial_information_gain import (  # noqa: E402
    fit_full_spatial_template_model,
    fit_full_temporal_template_model,
)
from src.lagpat_rank_audit import mask_phantom_ranks  # noqa: E402
from src.propagation_skeleton_geometry import parse_shaft  # noqa: E402
from src.seeg_coord_loader import load_subject_coords  # noqa: E402
from src.topic5_interictal_direction_rose import (  # noqa: E402
    assess_event_direction_qc,
    fit_event_directions_3d,
)
from src.topic5_template_axis_field import (  # noqa: E402
    INTERICTAL_FIELD_CONTRACT,
    INTERICTAL_FIELD_FINGERPRINT_ALGORITHM,
    TEMPLATE_AXIS_DEFINITION,
    TEMPLATE_AXIS_DIRECTION,
    build_interictal_template_field_record,
    interictal_field_quality_tier,
)


DEFAULT_REFERENCE_ROOT = (
    ROOT / "results/interictal_propagation_masked/template_gradient_fields/per_subject"
)
DEFAULT_OUT = (
    ROOT
    / "results/interictal_propagation_masked/"
    "template_gradient_fields_timing_plus_space"
)
YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path(
    "/mnt/epilepsia_data/interilca_inter_results/all_data_lns"
)
DEFAULT_MAX_EVENTS = 5000
DEFAULT_MIN_CLUSTER_EVENTS = 20
DEFAULT_SEED = 20260825


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _subject_seed(subject_id: str, seed: int, suffix: str) -> int:
    token = f"{subject_id}|{suffix}".encode("utf-8")
    return int((zlib.crc32(token) + int(seed)) % (2**32 - 1))


def _subject_dir(dataset: str, subject: str) -> Path:
    if dataset == "yuquan":
        return YUQUAN_ROOT / subject
    if dataset == "epilepsiae":
        return EPILEPSIAE_ROOT / subject / "all_recs"
    raise ValueError(f"unsupported dataset: {dataset}")


def _select_events(
    n_events: int, *, max_events: int, subject_id: str, seed: int
) -> np.ndarray:
    if max_events <= 0 or n_events <= max_events:
        return np.arange(n_events, dtype=int)
    rng = np.random.default_rng(_subject_seed(subject_id, seed, "event_sample"))
    return np.sort(rng.choice(n_events, size=max_events, replace=False))


def _old_templates_in_event_order(
    reference: Mapping[str, object], event_names: Sequence[str]
) -> np.ndarray:
    field = reference.get("interictal_field") or {}
    old_names = [str(name) for name in field.get("contact_order", [])]
    if not old_names:
        old_names = [str(name) for name in reference.get("names", [])]
        rank_a = np.asarray(reference.get("rank_a", []), float)
        rank_b = np.asarray(reference.get("rank_b", []), float)
    else:
        rank_a = np.asarray(field.get("rank_a", []), float)
        rank_b = np.asarray(field.get("rank_b", []), float)
    if not (len(old_names) == len(rank_a) == len(rank_b)):
        raise ValueError("reference field template/contact shape mismatch")
    index = {name: i for i, name in enumerate(event_names)}
    templates = np.full((2, len(event_names)), np.nan, float)
    for source_index, name in enumerate(old_names):
        if name in index:
            templates[0, index[name]] = rank_a[source_index]
            templates[1, index[name]] = rank_b[source_index]
    return templates


def _line_angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    left = np.asarray(a, float)
    right = np.asarray(b, float)
    cosine = float(np.clip(left @ right, -1.0, 1.0))
    return float(np.degrees(np.arccos(abs(cosine))))


def _failed_record(
    subject_id: str, status: str, *, reference_path: Path, error: str | None = None
) -> dict[str, object]:
    dataset, subject = subject_id.split("_", 1)
    return {
        "contract": INTERICTAL_FIELD_CONTRACT,
        "subject_id": subject_id,
        "dataset": dataset,
        "subject": subject,
        "stable_k": 2,
        "template_labels": {"a": "TA", "b": "TB"},
        "axis_definition": TEMPLATE_AXIS_DEFINITION,
        "axis_direction_convention": TEMPLATE_AXIS_DIRECTION,
        "status": status,
        "error": error,
        "template_discovery": {
            "method": "timing_plus_space_full_fit_v1",
            "fit_role": "full interictal fit after independent cross-fit validation",
        },
        "direction_validity": {
            "ta": {"estimable": False, "reason_codes": [status]},
            "tb": {"estimable": False, "reason_codes": [status]},
            "pair": {
                "axis_pair_estimable": False,
                "geometry_2d_supported": False,
                "strict_stability_pass": False,
            },
        },
        "interictal_field": {"status": "axis_not_available"},
        "source": {
            "reference_timing_field": str(reference_path),
            "reference_timing_field_sha256": (
                _sha256(reference_path) if reference_path.exists() else None
            ),
        },
    }


def build_subject(
    subject_id: str,
    *,
    reference_root: Path,
    max_events: int,
    min_cluster_events: int,
    seed: int,
) -> dict[str, object]:
    reference_path = reference_root / f"{subject_id}.json"
    if not reference_path.exists():
        return _failed_record(
            subject_id, "missing_reference_coordinate_record", reference_path=reference_path
        )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    dataset, subject = subject_id.split("_", 1)
    try:
        events = load_subject_propagation_events(_subject_dir(dataset, subject))
        event_names = [str(name) for name in events["channel_names"]]
        coord_record = load_subject_coords(dataset, subject, event_names)
        coords_all = np.asarray(
            coord_record.coords_array_in_requested_order, float
        )
        mapped = np.asarray(
            coord_record.mapped_mask_in_requested_order, bool
        ) & np.isfinite(coords_all).all(axis=1)
        if int(mapped.sum()) < 6:
            raise ValueError(f"only {int(mapped.sum())} mapped contacts; need at least 6")

        names = [event_names[index] for index in np.flatnonzero(mapped)]
        coords = coords_all[mapped]
        shafts = [parse_shaft(name)[0] for name in names]
        ranks_all = np.asarray(events["ranks"], float)[mapped]
        bools_all = np.asarray(events["bools"], bool)[mapped]
        blocks_all = np.asarray(events["block_ids"], int)
        selection = _select_events(
            ranks_all.shape[1],
            max_events=max_events,
            subject_id=subject_id,
            seed=seed,
        )
        ranks = ranks_all[:, selection]
        bools = bools_all[:, selection]
        blocks = blocks_all[selection]
        masked = mask_phantom_ranks(ranks, bools, normalize=True)
        direction_fit = fit_event_directions_3d(masked, coords, min_contacts=3)
        direction_qc = assess_event_direction_qc(
            masked,
            coords,
            shafts,
            directions=direction_fit["directions"],
            n_valid_contacts=direction_fit["n_valid_contacts"],
            effective_rank=direction_fit["effective_rank"],
        )
        qc_pass = np.asarray(direction_qc["passes"], bool)
        qc_indices = np.flatnonzero(qc_pass)
        if qc_indices.size < 2 * min_cluster_events:
            raise ValueError(
                f"only {qc_indices.size} QC-clean events; need at least "
                f"{2 * min_cluster_events} for full fit"
            )

        fitted = fit_full_spatial_template_model(
            ranks[:, qc_indices],
            bools[:, qc_indices],
            np.asarray(direction_fit["directions"], float)[qc_indices],
            coords,
            min_cluster_events=min_cluster_events,
            random_state=0,
        )
        temporal_fitted = fit_full_temporal_template_model(
            ranks[:, qc_indices],
            bools[:, qc_indices],
            coords,
            min_cluster_events=min_cluster_events,
            random_state=0,
        )
        labels = np.asarray(fitted["labels"], int)
        supports = np.asarray(fitted["supports"], float)
        templates = np.asarray(fitted["templates"], float)
        field_valid = (
            np.isfinite(templates).all(axis=0)
            & np.isfinite(supports).all(axis=0)
            & (supports[0] > 0.0)
            & (supports[1] > 0.0)
        )
        if int(field_valid.sum()) < 6:
            raise ValueError(
                f"only {int(field_valid.sum())} joint-supported contacts; need at least 6"
            )

        def dense_rank(values: np.ndarray) -> np.ndarray:
            order = np.argsort(np.asarray(values, float), kind="mergesort")
            ranked = np.empty(len(order), dtype=float)
            ranked[order] = np.arange(len(order), dtype=float)
            return ranked

        field_names = [names[index] for index in np.flatnonzero(field_valid)]
        field_shafts = [shafts[index] for index in np.flatnonzero(field_valid)]
        field_templates = np.vstack([
            dense_rank(templates[0, field_valid]),
            dense_rank(templates[1, field_valid]),
        ])
        record = build_interictal_template_field_record(
            subject_id=subject_id,
            dataset=dataset,
            subject=subject,
            stable_k=2,
            names=field_names,
            coords=coords[field_valid],
            rank_ta=field_templates[0],
            rank_tb=field_templates[1],
            shafts=field_shafts,
            support_ta=supports[0, field_valid],
            support_tb=supports[1, field_valid],
            support_source="same-label Timing+Space QC-clean full-fit events",
            template_event_counts={
                "a": int(np.sum(labels == 0)),
                "b": int(np.sum(labels == 1)),
                "unassigned": 0,
            },
            n_axis_boot=200,
            n_pair_boot=500,
            line_threshold=0.50,
            seed=_subject_seed(subject_id, seed, "axis_field"),
        )

        temporal_supports = np.asarray(temporal_fitted["supports"], float)
        temporal_templates = np.asarray(temporal_fitted["templates"], float)
        temporal_field_valid = (
            np.isfinite(temporal_templates).all(axis=0)
            & np.isfinite(temporal_supports).all(axis=0)
            & (temporal_supports[0] > 0.0)
            & (temporal_supports[1] > 0.0)
        )
        if int(temporal_field_valid.sum()) < 6:
            raise ValueError(
                f"Timing-only has only {int(temporal_field_valid.sum())} "
                "joint-supported contacts; need at least 6"
            )
        temporal_field_names = [
            names[index] for index in np.flatnonzero(temporal_field_valid)
        ]
        temporal_field_shafts = [
            shafts[index] for index in np.flatnonzero(temporal_field_valid)
        ]
        temporal_dense_templates = np.vstack([
            dense_rank(temporal_templates[0, temporal_field_valid]),
            dense_rank(temporal_templates[1, temporal_field_valid]),
        ])
        matched_timing_record = build_interictal_template_field_record(
            subject_id=subject_id,
            dataset=dataset,
            subject=subject,
            stable_k=2,
            names=temporal_field_names,
            coords=coords[temporal_field_valid],
            rank_ta=temporal_dense_templates[0],
            rank_tb=temporal_dense_templates[1],
            shafts=temporal_field_shafts,
            support_ta=temporal_supports[0, temporal_field_valid],
            support_tb=temporal_supports[1, temporal_field_valid],
            support_source="same-label matched Timing-only QC-clean full-fit events",
            template_event_counts={
                "a": int(np.sum(np.asarray(temporal_fitted["labels"], int) == 0)),
                "b": int(np.sum(np.asarray(temporal_fitted["labels"], int) == 1)),
                "unassigned": 0,
            },
            n_axis_boot=200,
            n_pair_boot=500,
            line_threshold=0.50,
            seed=_subject_seed(subject_id, seed, "timing_only_axis_field"),
        )
        matched_timing_record["template_discovery"] = {
            "method": "matched_timing_only_full_fit_v1",
            "fit_role": temporal_fitted["fit_role"],
            "template_label_rule": temporal_fitted["template_label_rule"],
            "sample_contract": "identical QC-clean events as Timing+Space full fit",
            "n_qc_clean_events": int(qc_indices.size),
            "qc_original_event_indices": selection[qc_indices],
            "qc_event_labels": np.asarray(temporal_fitted["labels"], int),
            "joint_supported_contact_mask": temporal_field_valid,
            "joint_supported_contact_names": temporal_field_names,
            "n_joint_supported_contacts": int(temporal_field_valid.sum()),
            "raw_mean_rank_templates": temporal_templates,
            "field_dense_rank_templates": temporal_dense_templates,
        }
        matched_timing_record["source"] = {
            "reference_timing_field": str(reference_path),
            "event_artifact_root": str(_subject_dir(dataset, subject)),
            "coordinate_loader": "seeg_coord_loader_v3p1",
            "ictal_input": "none",
        }
        record["_matched_timing_only_field_record"] = matched_timing_record

        try:
            old_templates = _old_templates_in_event_order(
                reference, event_names
            )[:, mapped]
            old_labels = assign_events_to_templates(
                ranks[:, qc_indices],
                bools[:, qc_indices],
                old_templates,
                min_shared_channels=3,
            )
            comparable = old_labels >= 0
            contingency = np.zeros((2, 2), int)
            if np.any(comparable):
                for old_label in (0, 1):
                    for new_label in (0, 1):
                        contingency[old_label, new_label] = int(np.sum(
                            (old_labels[comparable] == old_label)
                            & (labels[comparable] == new_label)
                        ))
                label_ami = float(adjusted_mutual_info_score(
                    old_labels[comparable], labels[comparable]
                ))
                direct_overlap = float(
                    np.mean(old_labels[comparable] == labels[comparable])
                )
            else:
                label_ami = float("nan")
                direct_overlap = float("nan")
            old_comparison_error = None
        except Exception as exc:
            comparable = np.zeros(len(labels), bool)
            contingency = np.zeros((2, 2), int)
            label_ami = float("nan")
            direct_overlap = float("nan")
            old_comparison_error = f"{type(exc).__name__}: {exc}"

        record["template_discovery"] = {
            "method": "timing_plus_space_full_fit_v1",
            "validated_by": "two-way alternating recording-block cross-fit direction score",
            "fit_role": fitted["fit_role"],
            "template_label_rule": fitted["template_label_rule"],
            "temporal_view": "canonical masked lagPat ranks with event-median imputation",
            "spatial_view": "unit 3D early-to-late event gradient on real coordinates",
            "view_weight": "equal total variance on the full-fit QC-clean sample",
            "spatial_scale": float(fitted["spatial_scale"]),
            "n_events_total": int(ranks_all.shape[1]),
            "n_events_sampled": int(selection.size),
            "n_qc_clean_events": int(qc_indices.size),
            "qc_retention": float(qc_indices.size / max(1, selection.size)),
            "n_blocks_qc_clean": int(np.unique(blocks[qc_indices]).size),
            "cluster_counts": np.asarray(fitted["cluster_counts"], int),
            "joint_supported_contact_mask": field_valid,
            "joint_supported_contact_names": field_names,
            "n_joint_supported_contacts": int(field_valid.sum()),
            "raw_mean_rank_templates": templates,
            "field_dense_rank_templates": field_templates,
            "sampled_event_indices": selection,
            "qc_sample_indices": qc_indices,
            "qc_original_event_indices": selection[qc_indices],
            "qc_block_ids": blocks[qc_indices],
            "qc_event_labels": labels,
            "cluster_order_from_raw_kmeans": fitted[
                "cluster_order_from_raw_kmeans"
            ],
            "direction_qc_contract": {
                "minimum_mapped_participating_contacts": 6,
                "minimum_participating_shafts": 2,
                "minimum_effective_coordinate_rank": 2,
                "minimum_loco_valid_fraction": 0.8,
                "minimum_loco_median_signed_cosine": 0.8,
            },
        }
        temporal_labels = np.asarray(temporal_fitted["labels"], int)
        timing_space_contingency = np.zeros((2, 2), int)
        for timing_label in (0, 1):
            for spatial_label in (0, 1):
                timing_space_contingency[timing_label, spatial_label] = int(
                    np.sum(
                        (temporal_labels == timing_label)
                        & (labels == spatial_label)
                    )
                )
        identity_overlap = float(np.mean(temporal_labels == labels))
        flipped_overlap = float(np.mean(temporal_labels == (1 - labels)))
        record["matched_timing_only_full_fit"] = {
            "fit_role": temporal_fitted["fit_role"],
            "template_label_rule": temporal_fitted["template_label_rule"],
            "cluster_counts": temporal_fitted["cluster_counts"],
            "event_labels": temporal_labels,
            "templates": temporal_fitted["templates"],
            "supports": temporal_fitted["supports"],
            "axes": temporal_fitted["axes"],
            "timing_space_label_ami": float(
                adjusted_mutual_info_score(temporal_labels, labels)
            ),
            "timing_space_direct_ab_overlap": identity_overlap,
            "timing_space_best_overlap": max(identity_overlap, flipped_overlap),
            "timing_space_best_mapping": (
                "identity" if identity_overlap >= flipped_overlap else "swap"
            ),
            "contingency_timing_rows_space_columns": timing_space_contingency,
        }
        record["old_timing_comparison"] = {
            "n_comparable_qc_events": int(comparable.sum()),
            "label_ami": label_ami,
            "direct_ab_overlap": direct_overlap,
            "contingency_old_rows_new_columns": contingency,
            "error": old_comparison_error,
        }
        old_pair = reference.get("axis_pair") or {}
        new_pair = record.get("axis_pair") or {}
        if old_pair.get("status") == "ok" and new_pair.get("status") == "ok":
            for key in ("axis_a", "axis_b"):
                old_u = np.asarray(old_pair[key]["u"], float)
                new_u = np.asarray(new_pair[key]["u"], float)
                record["old_timing_comparison"][f"{key}_signed_cosine"] = float(
                    old_u @ new_u
                )
                record["old_timing_comparison"][f"{key}_line_angle_deg"] = (
                    _line_angle_deg(old_u, new_u)
                )
        record["source"] = {
            "reference_timing_field": str(reference_path),
            "reference_timing_field_sha256": _sha256(reference_path),
            "event_artifact_root": str(_subject_dir(dataset, subject)),
            "coordinate_loader": "seeg_coord_loader_v3p1",
            "ictal_input": "none",
        }
        return record
    except Exception as exc:
        return _failed_record(
            subject_id,
            "timing_plus_space_full_fit_failed",
            reference_path=reference_path,
            error=f"{type(exc).__name__}: {exc}",
        )


def _cohort_row(
    record: Mapping[str, object], reference: Mapping[str, object] | None
) -> dict[str, object]:
    field = record.get("interictal_field") or {}
    pair = record.get("axis_pair") or {}
    old_pair = (reference or {}).get("axis_pair") or {}
    old_field = (reference or {}).get("interictal_field") or {}
    comparison = record.get("old_timing_comparison") or {}
    row = {
        "subject_id": record.get("subject_id"),
        "dataset": record.get("dataset"),
        "subject": record.get("subject"),
        "status": record.get("status"),
        "error": record.get("error"),
        "n_qc_clean_events": (record.get("template_discovery") or {}).get(
            "n_qc_clean_events"
        ),
        "cluster_a_events": (record.get("template_event_counts") or {}).get("a"),
        "cluster_b_events": (record.get("template_event_counts") or {}).get("b"),
        "interictal_field_status": field.get("status"),
        "axis_quality_tier": interictal_field_quality_tier(record),
        "axis_pair_estimable": pair.get("axis_pair_estimable"),
        "geometry_2d_supported": pair.get("geometry_2d_supported"),
        "strict_stability_pass": pair.get("strict_stability_pass"),
        "relation": (pair.get("relation") or {}).get("relation"),
        "line_angle_deg": (pair.get("relation") or {}).get("line_angle_deg"),
        "shared_field_available": "shared_a" in (field.get("field_models") or {}),
        "old_geometry_2d_supported": old_pair.get("geometry_2d_supported"),
        "old_relation": (old_pair.get("relation") or {}).get("relation"),
        "old_shared_field_available": "shared_a" in (
            old_field.get("field_models") or {}
        ),
        "label_ami_old_vs_new": comparison.get("label_ami"),
        "direct_ab_overlap_old_vs_new": comparison.get("direct_ab_overlap"),
        "axis_a_line_angle_change_deg": comparison.get("axis_a_line_angle_deg"),
        "axis_b_line_angle_change_deg": comparison.get("axis_b_line_angle_deg"),
        "field_fingerprint_sha256": field.get("fingerprint_sha256"),
        "fingerprint_algorithm": field.get("fingerprint_algorithm"),
    }
    return row


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    ready = [row for row in rows if row.get("interictal_field_status") == "ok"]
    geometry = [row for row in ready if row.get("geometry_2d_supported") is True]
    shared = [row for row in geometry if row.get("shared_field_available") is True]
    old_geometry = [
        row for row in rows if row.get("old_geometry_2d_supported") is True
    ]
    old_shared = [
        row for row in old_geometry if row.get("old_shared_field_available") is True
    ]
    old_shared_ids = {str(row["subject_id"]) for row in old_shared}
    new_shared_ids = {str(row["subject_id"]) for row in shared}
    old_geometry_ids = {str(row["subject_id"]) for row in old_geometry}
    new_geometry_ids = {str(row["subject_id"]) for row in geometry}
    return {
        "contract": "topic5_timing_plus_space_interictal_template_fields_v1",
        "base_field_contract": INTERICTAL_FIELD_CONTRACT,
        "axis_definition": TEMPLATE_AXIS_DEFINITION,
        "axis_direction_convention": TEMPLATE_AXIS_DIRECTION,
        "fingerprint_algorithm": INTERICTAL_FIELD_FINGERPRINT_ALGORITHM,
        "ictal_independence": "no seizure/onset/subtype/energy input is read",
        "denominators": {
            "requested_geometry_reference_subjects": len(rows),
            "full_fit_fields_ready": len(ready),
            "geometry_2d_supported": len(geometry),
            "shared_fields_ready": len(shared),
            "old_geometry_2d_supported_same_subjects": len(old_geometry),
            "old_shared_fields_same_subjects": len(old_shared),
        },
        "geometry_change": {
            "gained": sorted(new_geometry_ids - old_geometry_ids),
            "lost": sorted(old_geometry_ids - new_geometry_ids),
            "unchanged": sorted(old_geometry_ids & new_geometry_ids),
        },
        "shared_field_change": {
            "gained": sorted(new_shared_ids - old_shared_ids),
            "lost": sorted(old_shared_ids - new_shared_ids),
            "unchanged": sorted(old_shared_ids & new_shared_ids),
        },
        "claim_boundary": (
            "cross-fit supports added spatial information; this all-interictal "
            "refit freezes final templates and is not an additional held-out test"
        ),
    }


README = """# Timing+Space template gradient fields

这里保存经 held-out direction-score 验证后，在完整确定性间期样本上重新拟合的
Timing+Space TA/TB 模板、传播轴与二维 field。producer 不读取发作、onset、subtype 或
发作能量数据。

- `per_subject/<dataset>_<subject>.json`：冻结 hybrid event labels、rank templates、
  同一 labels 得到的 support、真实坐标传播轴、own/shared field 及 fingerprint。
- `axis_cohort.csv`：新旧模板的逐患者 label overlap、轴角度变化、二维几何与 shared-field
  gained/lost 状态。
- `cohort_summary.json`：新旧分母和 gained/lost 患者清单。

`template_discovery.fit_role` 明确区分两件事：cross-fit 负责证明空间信息有 held-out 增益；
full fit 负责冻结 Figure 3 所需的最终间期模板。下游不得把 full-fit Figure 3 读出再次写成
独立 held-out 证明。
"""


def _default_subjects(reference_root: Path) -> list[str]:
    subjects = []
    for path in sorted(reference_root.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("stable_k") == 2:
            subjects.append(path.stem)
    if not subjects:
        raise FileNotFoundError(
            f"no stable-k=2 reference subjects under {reference_root}"
        )
    return subjects


def run(
    subjects: Sequence[str],
    *,
    reference_root: Path,
    out_dir: Path,
    matched_timing_out_dir: Path,
    max_events: int,
    min_cluster_events: int,
    seed: int,
    workers: int,
) -> dict[str, object]:
    if workers < 1:
        raise ValueError("workers must be positive")
    per_subject = out_dir / "per_subject"
    timing_per_subject = matched_timing_out_dir / "per_subject"
    per_subject.mkdir(parents=True, exist_ok=True)
    timing_per_subject.mkdir(parents=True, exist_ok=True)

    def job(subject_id: str) -> dict[str, object]:
        return build_subject(
            subject_id,
            reference_root=reference_root,
            max_events=max_events,
            min_cluster_events=min_cluster_events,
            seed=seed,
        )

    completed: dict[str, dict[str, object]] = {}
    if workers == 1:
        for subject_id in subjects:
            completed[subject_id] = job(subject_id)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(job, subject_id): subject_id for subject_id in subjects}
            for future in as_completed(futures):
                completed[futures[future]] = future.result()

    rows = []
    timing_rows = []
    for index, subject_id in enumerate(subjects, 1):
        record = completed[subject_id]
        timing_record = record.pop("_matched_timing_only_field_record", None)
        output = per_subject / f"{subject_id}.json"
        output.write_text(
            json.dumps(_jsonable(record), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        reference_path = reference_root / f"{subject_id}.json"
        reference = (
            json.loads(reference_path.read_text(encoding="utf-8"))
            if reference_path.exists()
            else None
        )
        row = _cohort_row(record, reference)
        rows.append(row)
        if timing_record is None:
            timing_record = _failed_record(
                subject_id,
                "matched_timing_only_full_fit_failed",
                reference_path=reference_path,
                error=record.get("error"),
            )
        (timing_per_subject / f"{subject_id}.json").write_text(
            json.dumps(_jsonable(timing_record), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        timing_rows.append(_cohort_row(timing_record, reference))
        print(
            f"[{index:02d}/{len(subjects)}] {subject_id}: "
            f"status={row['status']} 2d={row['geometry_2d_supported']} "
            f"shared={row['shared_field_available']}",
            flush=True,
        )

    columns = sorted({key for row in rows for key in row})
    with (out_dir / "axis_cohort.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_jsonable(rows))
    summary = _summary(rows)
    summary["run"] = {
        "max_events_per_subject": int(max_events),
        "min_cluster_events": int(min_cluster_events),
        "seed": int(seed),
        "workers": int(workers),
        "reference_root": str(reference_root),
    }
    (out_dir / "cohort_summary.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(README, encoding="utf-8")

    timing_columns = sorted({key for row in timing_rows for key in row})
    with (matched_timing_out_dir / "axis_cohort.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=timing_columns, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(_jsonable(timing_rows))
    timing_summary = _summary(timing_rows)
    timing_summary["contract"] = "topic5_matched_timing_only_interictal_template_fields_v1"
    timing_summary["comparison_role"] = (
        "same QC-clean events and field construction as Timing+Space; spatial view omitted"
    )
    timing_summary["run"] = summary["run"]
    (matched_timing_out_dir / "cohort_summary.json").write_text(
        json.dumps(_jsonable(timing_summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (matched_timing_out_dir / "README.md").write_text(
        "# Matched Timing-only template gradient fields\n\n"
        "与 Timing+Space 使用完全相同的 QC-clean 间期事件、共同参与触点筛选、"
        "稠密序位化和 field producer；唯一差异是聚类时不输入事件空间方向。\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument(
        "--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--matched-timing-out-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--max-events", type=int, default=DEFAULT_MAX_EVENTS)
    parser.add_argument(
        "--min-cluster-events", type=int, default=DEFAULT_MIN_CLUSTER_EVENTS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    subjects = args.subjects or _default_subjects(args.reference_root)
    matched_timing_out_dir = (
        args.matched_timing_out_dir
        if args.matched_timing_out_dir is not None
        else args.out_dir.parent / f"{args.out_dir.name}_matched_timing_only"
    )
    summary = run(
        subjects,
        reference_root=args.reference_root,
        out_dir=args.out_dir,
        matched_timing_out_dir=matched_timing_out_dir,
        max_events=args.max_events,
        min_cluster_events=args.min_cluster_events,
        seed=args.seed,
        workers=args.workers,
    )
    print(json.dumps(summary["denominators"], ensure_ascii=False, indent=2))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
