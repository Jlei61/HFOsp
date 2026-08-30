#!/usr/bin/env python3
"""Freeze all-event Timing+Space templates, axes and fields.

Every interictal event contributes to template discovery and support.  Events
with a finite single-event direction add an optional spatial view; events with
no mathematical direction retain their timing view and are never discarded.
No seizure, onset, ictal-energy or phenotype input is read.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_topic5_timing_plus_space_template_fields import (  # noqa: E402
    DEFAULT_MIN_CLUSTER_EVENTS,
    DEFAULT_REFERENCE_ROOT,
    DEFAULT_SEED,
    _failed_record,
    _jsonable,
    _line_angle_deg,
    _old_templates_in_event_order,
    _select_events,
    _sha256,
    _subject_dir,
    _subject_seed,
)
from src.interictal_propagation import (  # noqa: E402
    assign_events_to_templates,
    load_subject_propagation_events,
)
from src.interictal_spatial_information_gain import (  # noqa: E402
    fit_full_all_event_spatial_template_model,
    fit_full_temporal_template_model,
)
from src.lagpat_rank_audit import mask_phantom_ranks  # noqa: E402
from src.propagation_skeleton_geometry import parse_shaft  # noqa: E402
from src.seeg_coord_loader import load_subject_coords  # noqa: E402
from src.topic5_interictal_direction_rose import fit_event_directions_3d  # noqa: E402
from src.topic5_template_axis_field import (  # noqa: E402
    INTERICTAL_FIELD_CONTRACT,
    INTERICTAL_FIELD_FINGERPRINT_ALGORITHM,
    TEMPLATE_AXIS_DEFINITION,
    TEMPLATE_AXIS_DIRECTION,
    build_interictal_template_field_record,
    interictal_field_quality_tier,
)


DEFAULT_OUT = (
    ROOT / "results/interictal_propagation_masked/"
    "template_gradient_fields_all_events_timing_plus_space"
)
DEFAULT_TIMING_OUT = (
    ROOT / "results/interictal_propagation_masked/"
    "template_gradient_fields_all_events_matched_timing_only"
)
DEFAULT_MAX_EVENTS = 0


def _dense_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(values, float), kind="mergesort")
    ranked = np.empty(len(order), dtype=float)
    ranked[order] = np.arange(len(order), dtype=float)
    return ranked


def _field_record(
    *,
    subject_id: str,
    dataset: str,
    subject: str,
    names: Sequence[str],
    shafts: Sequence[str],
    coords: np.ndarray,
    mapped: np.ndarray,
    fitted: Mapping[str, Any],
    support_source: str,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    templates = np.asarray(fitted["templates"], float)
    supports = np.asarray(fitted["supports"], float)
    field_valid = (
        np.asarray(mapped, bool)
        & np.isfinite(templates).all(axis=0)
        & np.isfinite(supports).all(axis=0)
        & (supports[0] > 0.0)
        & (supports[1] > 0.0)
    )
    if int(field_valid.sum()) < 6:
        raise ValueError(
            f"only {int(field_valid.sum())} mapped joint-supported contacts"
        )
    field_names = [names[index] for index in np.flatnonzero(field_valid)]
    field_shafts = [shafts[index] for index in np.flatnonzero(field_valid)]
    field_templates = np.vstack([
        _dense_rank(templates[0, field_valid]),
        _dense_rank(templates[1, field_valid]),
    ])
    labels = np.asarray(fitted["labels"], int)
    record = build_interictal_template_field_record(
        subject_id=subject_id,
        dataset=dataset,
        subject=subject,
        stable_k=2,
        names=field_names,
        coords=np.asarray(coords, float)[field_valid],
        rank_ta=field_templates[0],
        rank_tb=field_templates[1],
        shafts=field_shafts,
        support_ta=supports[0, field_valid],
        support_tb=supports[1, field_valid],
        support_source=support_source,
        template_event_counts={
            "a": int(np.sum(labels == 0)),
            "b": int(np.sum(labels == 1)),
            "unassigned": 0,
        },
        n_axis_boot=200,
        n_pair_boot=500,
        line_threshold=0.50,
        seed=seed,
    )
    return record, field_valid, field_templates


def build_subject(
    subject_id: str,
    *,
    reference_root: Path,
    max_events: int,
    min_cluster_events: int,
    seed: int,
) -> dict[str, Any]:
    reference_path = reference_root / f"{subject_id}.json"
    if not reference_path.exists():
        return _failed_record(
            subject_id,
            "missing_reference_coordinate_record",
            reference_path=reference_path,
        )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    dataset, subject = subject_id.split("_", 1)
    try:
        events = load_subject_propagation_events(_subject_dir(dataset, subject))
        names = [str(name) for name in events["channel_names"]]
        ranks_all = np.asarray(events["ranks"], float)
        bools_all = np.asarray(events["bools"], bool)
        blocks_all = np.asarray(events["block_ids"], int)
        coordinate_record = load_subject_coords(dataset, subject, names)
        coords = np.asarray(
            coordinate_record.coords_array_in_requested_order, float
        )
        mapped = np.asarray(
            coordinate_record.mapped_mask_in_requested_order, bool
        ) & np.isfinite(coords).all(axis=1)
        if int(mapped.sum()) < 6:
            raise ValueError(f"only {int(mapped.sum())} mapped contacts")
        shafts = [parse_shaft(name)[0] for name in names]
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
        directions = np.asarray(direction_fit["directions"], float)
        direction_estimable = np.isfinite(directions).all(axis=1)

        fitted = fit_full_all_event_spatial_template_model(
            ranks,
            bools,
            directions,
            coords,
            min_cluster_events=min_cluster_events,
            random_state=0,
        )
        temporal_fitted = fit_full_temporal_template_model(
            ranks,
            bools,
            coords,
            min_cluster_events=min_cluster_events,
            random_state=0,
        )
        record, field_valid, field_templates = _field_record(
            subject_id=subject_id,
            dataset=dataset,
            subject=subject,
            names=names,
            shafts=shafts,
            coords=coords,
            mapped=mapped,
            fitted=fitted,
            support_source="all interictal events assigned by optional-spatial-view KMeans",
            seed=_subject_seed(subject_id, seed, "all_event_axis_field"),
        )
        timing_record, timing_field_valid, timing_field_templates = _field_record(
            subject_id=subject_id,
            dataset=dataset,
            subject=subject,
            names=names,
            shafts=shafts,
            coords=coords,
            mapped=mapped,
            fitted=temporal_fitted,
            support_source="same all interictal events assigned by Timing-only KMeans",
            seed=_subject_seed(subject_id, seed, "all_event_timing_axis_field"),
        )

        labels = np.asarray(fitted["labels"], int)
        temporal_labels = np.asarray(temporal_fitted["labels"], int)
        timing_contingency = np.zeros((2, 2), int)
        for timing_label in (0, 1):
            for spatial_label in (0, 1):
                timing_contingency[timing_label, spatial_label] = int(np.sum(
                    (temporal_labels == timing_label) & (labels == spatial_label)
                ))
        direct = float(np.mean(temporal_labels == labels))
        flipped = float(np.mean(temporal_labels == (1 - labels)))

        try:
            old_templates = _old_templates_in_event_order(reference, names)
            old_labels = assign_events_to_templates(
                ranks,
                bools,
                old_templates,
                min_shared_channels=3,
            )
            comparable = old_labels >= 0
            old_ami = float(adjusted_mutual_info_score(
                old_labels[comparable], labels[comparable]
            )) if np.any(comparable) else float("nan")
            old_overlap = float(np.mean(
                old_labels[comparable] == labels[comparable]
            )) if np.any(comparable) else float("nan")
            old_error = None
        except Exception as exc:
            comparable = np.zeros(len(labels), bool)
            old_ami = float("nan")
            old_overlap = float("nan")
            old_error = f"{type(exc).__name__}: {exc}"

        record["template_discovery"] = {
            "method": "timing_plus_space_all_events_missing_view_v1",
            "validated_by": "all-event alternating-block held-out direction score",
            "fit_role": fitted["fit_role"],
            "template_label_rule": fitted["template_label_rule"],
            "training_event_universe": "all selected interictal events",
            "hard_geometry_loco_qc_used": False,
            "temporal_channel_universe": "all event-artifact channels",
            "spatial_view": "unit 3D direction when minimally estimable; otherwise missing",
            "missing_spatial_view": "masked from spatial distance and centroid; event retained through timing",
            "view_weight": "equal total variance on events with both views",
            "n_events_total": int(ranks_all.shape[1]),
            "n_events_used_for_clustering": int(selection.size),
            "n_direction_estimable": int(direction_estimable.sum()),
            "n_direction_missing": int((~direction_estimable).sum()),
            "direction_estimable_fraction": float(direction_estimable.mean()),
            "n_blocks": int(np.unique(blocks).size),
            "cluster_counts": np.asarray(fitted["cluster_counts"], int),
            "cluster_spatial_counts": np.asarray(
                fitted["cluster_spatial_counts"], int
            ),
            "spatial_scale": float(fitted["spatial_scale"]),
            "event_labels": labels,
            "sampled_event_indices": selection,
            "joint_supported_contact_mask": field_valid,
            "joint_supported_contact_names": [
                names[index] for index in np.flatnonzero(field_valid)
            ],
            "raw_mean_rank_templates": fitted["templates"],
            "field_dense_rank_templates": field_templates,
        }
        record["matched_timing_only_full_fit"] = {
            "method": "timing_only_all_events_v1",
            "cluster_counts": temporal_fitted["cluster_counts"],
            "event_labels": temporal_labels,
            "templates": temporal_fitted["templates"],
            "supports": temporal_fitted["supports"],
            "timing_space_label_ami": float(adjusted_mutual_info_score(
                temporal_labels, labels
            )),
            "timing_space_direct_ab_overlap": direct,
            "timing_space_best_overlap": max(direct, flipped),
            "timing_space_best_mapping": "identity" if direct >= flipped else "swap",
            "contingency_timing_rows_space_columns": timing_contingency,
        }
        record["old_timing_comparison"] = {
            "n_comparable_events": int(comparable.sum()),
            "label_ami": old_ami,
            "direct_ab_overlap": old_overlap,
            "error": old_error,
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

        timing_record["template_discovery"] = {
            "method": "matched_timing_only_all_events_v1",
            "training_event_universe": "same all events as Timing+Space",
            "hard_geometry_loco_qc_used": False,
            "n_events_total": int(ranks_all.shape[1]),
            "n_events_used_for_clustering": int(selection.size),
            "cluster_counts": temporal_fitted["cluster_counts"],
            "event_labels": temporal_labels,
            "sampled_event_indices": selection,
            "joint_supported_contact_mask": timing_field_valid,
            "raw_mean_rank_templates": temporal_fitted["templates"],
            "field_dense_rank_templates": timing_field_templates,
        }
        timing_record["source"] = record["source"]
        record["_matched_timing_only_field_record"] = timing_record
        return record
    except Exception as exc:
        return _failed_record(
            subject_id,
            "all_event_timing_plus_space_full_fit_failed",
            reference_path=reference_path,
            error=f"{type(exc).__name__}: {exc}",
        )


def _cohort_row(
    record: Mapping[str, Any],
    reference: Mapping[str, Any] | None,
) -> dict[str, Any]:
    field = record.get("interictal_field") or {}
    pair = record.get("axis_pair") or {}
    old_pair = (reference or {}).get("axis_pair") or {}
    old_field = (reference or {}).get("interictal_field") or {}
    discovery = record.get("template_discovery") or {}
    comparison = record.get("old_timing_comparison") or {}
    return {
        "subject_id": record.get("subject_id"),
        "dataset": record.get("dataset"),
        "subject": record.get("subject"),
        "status": record.get("status"),
        "error": record.get("error"),
        "n_events_used_for_clustering": discovery.get("n_events_used_for_clustering"),
        "n_direction_estimable": discovery.get("n_direction_estimable"),
        "direction_estimable_fraction": discovery.get("direction_estimable_fraction"),
        "cluster_a_events": (record.get("template_event_counts") or {}).get("a"),
        "cluster_b_events": (record.get("template_event_counts") or {}).get("b"),
        "n_field_contacts": field.get("n_contacts"),
        "interictal_field_status": field.get("status"),
        "axis_quality_tier": interictal_field_quality_tier(record),
        "axis_pair_estimable": pair.get("axis_pair_estimable"),
        "geometry_2d_supported": pair.get("geometry_2d_supported"),
        "strict_stability_pass": pair.get("strict_stability_pass"),
        "relation": (pair.get("relation") or {}).get("relation"),
        "line_angle_deg": (pair.get("relation") or {}).get("line_angle_deg"),
        "shared_field_available": "shared_a" in (field.get("field_models") or {}),
        "old_geometry_2d_supported": old_pair.get("geometry_2d_supported"),
        "old_shared_field_available": "shared_a" in (old_field.get("field_models") or {}),
        "label_ami_old_vs_new": comparison.get("label_ami"),
        "direct_ab_overlap_old_vs_new": comparison.get("direct_ab_overlap"),
        "axis_a_line_angle_change_deg": comparison.get("axis_a_line_angle_deg"),
        "axis_b_line_angle_change_deg": comparison.get("axis_b_line_angle_deg"),
        "field_fingerprint_sha256": field.get("fingerprint_sha256"),
        "fingerprint_algorithm": field.get("fingerprint_algorithm"),
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ready = [row for row in rows if row.get("interictal_field_status") == "ok"]
    geometry = [row for row in ready if row.get("geometry_2d_supported") is True]
    shared = [row for row in geometry if row.get("shared_field_available") is True]
    old_geometry = [row for row in rows if row.get("old_geometry_2d_supported") is True]
    old_shared = [row for row in old_geometry if row.get("old_shared_field_available") is True]
    new_geometry_ids = {str(row["subject_id"]) for row in geometry}
    old_geometry_ids = {str(row["subject_id"]) for row in old_geometry}
    new_shared_ids = {str(row["subject_id"]) for row in shared}
    old_shared_ids = {str(row["subject_id"]) for row in old_shared}
    return {
        "contract": "topic5_all_event_timing_plus_space_interictal_template_fields_v1",
        "base_field_contract": INTERICTAL_FIELD_CONTRACT,
        "axis_definition": TEMPLATE_AXIS_DEFINITION,
        "axis_direction_convention": TEMPLATE_AXIS_DIRECTION,
        "fingerprint_algorithm": INTERICTAL_FIELD_FINGERPRINT_ALGORITHM,
        "ictal_independence": "no seizure/onset/subtype/energy input is read",
        "event_policy": "all events contribute timing; finite directions add an optional spatial view",
        "denominators": {
            "requested_subjects": len(rows),
            "full_fit_fields_ready": len(ready),
            "geometry_2d_supported": len(geometry),
            "shared_fields_ready": len(shared),
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
    }


def _default_subjects(reference_root: Path) -> list[str]:
    subjects = []
    for path in sorted(reference_root.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("stable_k") == 2:
            subjects.append(path.stem)
    if not subjects:
        raise FileNotFoundError(f"no stable-k=2 subjects under {reference_root}")
    return subjects


README = """# All-event Timing+Space template gradient fields

全部间期事件都通过 timing view 进入 K=2 聚类。能够由至少三个映射参与触点得到非退化
梯度的事件额外提供空间方向；其余事件的空间视图记为缺失，不删除事件，也不补零向量。
TA/TB rank template、参与 support、传播轴和二维场均由这套全事件标签重新计算。

**关注点**：producer 不读取发作数据；方向不可估事件仍影响最终 timing template 与 support。
"""


def run(
    subjects: Sequence[str],
    *,
    reference_root: Path,
    out_dir: Path,
    timing_out_dir: Path,
    max_events: int,
    min_cluster_events: int,
    seed: int,
    workers: int,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    per_subject = out_dir / "per_subject"
    timing_per_subject = timing_out_dir / "per_subject"
    per_subject.mkdir(parents=True, exist_ok=True)
    timing_per_subject.mkdir(parents=True, exist_ok=True)

    def job(subject_id: str) -> dict[str, Any]:
        return build_subject(
            subject_id,
            reference_root=reference_root,
            max_events=max_events,
            min_cluster_events=min_cluster_events,
            seed=seed,
        )

    completed: dict[str, dict[str, Any]] = {}
    if workers == 1:
        for subject_id in subjects:
            completed[subject_id] = job(subject_id)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(job, subject): subject for subject in subjects}
            for future in as_completed(futures):
                completed[futures[future]] = future.result()

    rows = []
    timing_rows = []
    for index, subject_id in enumerate(subjects, 1):
        record = completed[subject_id]
        timing_record = record.pop("_matched_timing_only_field_record", None)
        reference_path = reference_root / f"{subject_id}.json"
        reference = json.loads(reference_path.read_text()) if reference_path.exists() else None
        (per_subject / f"{subject_id}.json").write_text(
            json.dumps(_jsonable(record), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        rows.append(_cohort_row(record, reference))
        if timing_record is None:
            timing_record = _failed_record(
                subject_id,
                "matched_timing_only_all_event_fit_failed",
                reference_path=reference_path,
                error=record.get("error"),
            )
        (timing_per_subject / f"{subject_id}.json").write_text(
            json.dumps(_jsonable(timing_record), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        timing_rows.append(_cohort_row(timing_record, reference))
        row = rows[-1]
        print(
            f"[{index:02d}/{len(subjects)}] {subject_id}: "
            f"status={row['status']} events={row['n_events_used_for_clustering']} "
            f"2d={row['geometry_2d_supported']} shared={row['shared_field_available']}",
            flush=True,
        )

    for target, target_rows, role in (
        (out_dir, rows, "Timing+Space"),
        (timing_out_dir, timing_rows, "matched Timing-only"),
    ):
        columns = sorted({key for row in target_rows for key in row})
        with (target / "axis_cohort.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(_jsonable(target_rows))
        summary = _summary(target_rows)
        summary["comparison_role"] = role
        summary["run"] = {
            "max_events_per_subject": int(max_events),
            "min_cluster_events": int(min_cluster_events),
            "seed": int(seed),
            "workers": int(workers),
            "reference_root": str(reference_root),
        }
        (target / "cohort_summary.json").write_text(
            json.dumps(_jsonable(summary), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (target / "README.md").write_text(README, encoding="utf-8")
    return _summary(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--matched-timing-out-dir", type=Path, default=DEFAULT_TIMING_OUT)
    parser.add_argument("--max-events", type=int, default=DEFAULT_MAX_EVENTS)
    parser.add_argument("--min-cluster-events", type=int, default=DEFAULT_MIN_CLUSTER_EVENTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    subjects = args.subjects or _default_subjects(args.reference_root)
    summary = run(
        subjects,
        reference_root=args.reference_root,
        out_dir=args.out_dir,
        timing_out_dir=args.matched_timing_out_dir,
        max_events=args.max_events,
        min_cluster_events=args.min_cluster_events,
        seed=args.seed,
        workers=args.workers,
    )
    print(json.dumps(summary["denominators"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
