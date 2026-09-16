#!/usr/bin/env python3
"""Cross-fitted contact-rank read-back for all-event Timing+Space templates.

Every interictal event contributes its masked temporal rank view. Events with
an estimable three-dimensional direction additionally contribute a spatial
view during training. For each alternating-recording-block fold, held-out
events are assigned from the frozen training rank templates only. The outcome
is the Spearman correlation between a training-template early-to-late axis and
the held-out mean contact rank for the corresponding template.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import spearmanr, wilcoxon
from sklearn.cluster import KMeans


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_topic5_interictal_template_direction_rose import _subject_dir  # noqa: E402
from src.interictal_propagation import (  # noqa: E402
    assign_events_to_templates,
    build_cluster_templates,
    load_subject_propagation_events,
)
from src.lagpat_rank_audit import (  # noqa: E402
    build_masked_kmeans_features,
    mask_phantom_ranks,
)
from src.seeg_coord_loader import load_subject_coords  # noqa: E402
from src.topic5_interictal_direction_rose import fit_event_directions_3d  # noqa: E402


SOURCE_ROOT = (
    ROOT / "results/interictal_propagation_masked/spatial_information_gain_all_events"
)
OUT_ROOT = (
    ROOT / "results/interictal_propagation_masked/timing_plus_space_rank_readback"
)
MIN_CLUSTER_EVENTS = 20
MIN_SPATIAL_CLUSTER_EVENTS = 3
MIN_HELDOUT_CLUSTER_EVENTS = 3
MIN_SHARED_CHANNELS = 3
SEED = 20260825


def _alternating_block_folds(block_ids: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    blocks = np.asarray(block_ids)
    unique = np.unique(blocks)
    if unique.size < 2:
        raise ValueError("at least two recording blocks are required")
    even = np.flatnonzero(np.isin(blocks, unique[0::2]))
    odd = np.flatnonzero(np.isin(blocks, unique[1::2]))
    if not even.size or not odd.size:
        raise ValueError("alternating block split produced an empty fold")
    return [(even, odd), (odd, even)]


def _equal_view_scale(temporal: np.ndarray, spatial: np.ndarray) -> float:
    temporal_variance = float(np.var(temporal, axis=0).sum())
    spatial_variance = float(np.var(spatial, axis=0).sum())
    if temporal_variance <= 1e-12 or spatial_variance <= 1e-12:
        raise ValueError("both feature views must have non-zero variance")
    return float(np.sqrt(temporal_variance / spatial_variance))


def _fit_missing_view_labels(
    temporal: np.ndarray,
    directions: np.ndarray,
    *,
    random_state: int,
    n_init: int = 10,
    max_iter: int = 300,
) -> dict[str, Any]:
    """K=2 masked-view fit; missing spatial directions are not zero-filled."""
    temporal = np.asarray(temporal, float)
    spatial = np.asarray(directions, float)
    spatial_valid = np.isfinite(spatial).all(axis=1)
    if temporal.ndim != 2 or spatial.shape != (temporal.shape[0], 3):
        raise ValueError("temporal and spatial views do not align")
    if not np.isfinite(temporal).all() or int(spatial_valid.sum()) < 2:
        raise ValueError("insufficient finite training views")
    scale = _equal_view_scale(temporal[spatial_valid], spatial[spatial_valid])
    scaled_spatial = np.full_like(spatial, np.nan)
    scaled_spatial[spatial_valid] = scale * spatial[spatial_valid]
    best: dict[str, Any] | None = None
    for init in range(n_init):
        labels = KMeans(
            n_clusters=2, n_init=1, random_state=random_state + init
        ).fit_predict(temporal)
        iterations = 0
        for iterations in range(1, max_iter + 1):
            counts = np.bincount(labels, minlength=2)
            spatial_counts = np.asarray([
                np.sum((labels == cluster) & spatial_valid)
                for cluster in (0, 1)
            ], int)
            if int(counts.min()) == 0 or int(spatial_counts.min()) == 0:
                break
            temporal_centers = np.vstack([
                temporal[labels == cluster].mean(axis=0)
                for cluster in (0, 1)
            ])
            spatial_centers = np.vstack([
                scaled_spatial[(labels == cluster) & spatial_valid].mean(axis=0)
                for cluster in (0, 1)
            ])
            distances = np.stack([
                np.sum((temporal - temporal_centers[cluster]) ** 2, axis=1)
                for cluster in (0, 1)
            ], axis=1)
            for cluster in (0, 1):
                distances[spatial_valid, cluster] += np.sum(
                    (scaled_spatial[spatial_valid] - spatial_centers[cluster]) ** 2,
                    axis=1,
                )
            updated = np.argmin(distances, axis=1)
            if np.array_equal(updated, labels):
                break
            labels = updated
        counts = np.bincount(labels, minlength=2)
        spatial_counts = np.asarray([
            np.sum((labels == cluster) & spatial_valid)
            for cluster in (0, 1)
        ], int)
        if int(counts.min()) == 0 or int(spatial_counts.min()) == 0:
            continue
        temporal_centers = np.vstack([
            temporal[labels == cluster].mean(axis=0) for cluster in (0, 1)
        ])
        spatial_centers = np.vstack([
            scaled_spatial[(labels == cluster) & spatial_valid].mean(axis=0)
            for cluster in (0, 1)
        ])
        distances = np.stack([
            np.sum((temporal - temporal_centers[cluster]) ** 2, axis=1)
            for cluster in (0, 1)
        ], axis=1)
        for cluster in (0, 1):
            distances[spatial_valid, cluster] += np.sum(
                (scaled_spatial[spatial_valid] - spatial_centers[cluster]) ** 2,
                axis=1,
            )
        objective = float(np.sum(distances[np.arange(len(labels)), labels]))
        candidate = {
            "labels": labels.copy(),
            "counts": counts,
            "spatial_counts": spatial_counts,
            "spatial_scale": scale,
            "objective": objective,
            "iterations": iterations,
        }
        if best is None or objective < float(best["objective"]) - 1e-12:
            best = candidate
    if best is None:
        raise ValueError("masked-view KMeans failed to form two clusters")
    return best


def _fit_fold_model(
    ranks: np.ndarray,
    bools: np.ndarray,
    directions: np.ndarray,
    coords: np.ndarray,
    train: np.ndarray,
) -> dict[str, Any]:
    temporal = build_masked_kmeans_features(
        ranks[:, train], bools[:, train], impute="event_median"
    )
    fitted = _fit_missing_view_labels(
        temporal, directions[train], random_state=0
    )
    counts = np.asarray(fitted["counts"], int)
    spatial_counts = np.asarray(fitted["spatial_counts"], int)
    if int(counts.min()) < MIN_CLUSTER_EVENTS:
        raise ValueError(f"train cluster support below {MIN_CLUSTER_EVENTS}: {counts.tolist()}")
    if int(spatial_counts.min()) < MIN_SPATIAL_CLUSTER_EVENTS:
        raise ValueError(
            "train spatial support below "
            f"{MIN_SPATIAL_CLUSTER_EVENTS}: {spatial_counts.tolist()}"
        )
    labels = np.asarray(fitted["labels"], int)
    templates = build_cluster_templates(
        ranks[:, train], bools[:, train], labels, n_clusters=2
    )
    axes = np.asarray(
        fit_event_directions_3d(templates.T, coords, min_contacts=3)["directions"],
        float,
    )
    if axes.shape != (2, 3) or not np.isfinite(axes).all():
        raise ValueError("training template axis is not estimable")
    return {
        "templates": templates,
        "axes": axes,
        "counts": counts,
        "spatial_counts": spatial_counts,
        "spatial_scale": float(fitted["spatial_scale"]),
    }


def _readback_fold(
    model: dict[str, Any],
    masked_ranks: np.ndarray,
    bools: np.ndarray,
    coords: np.ndarray,
    test: np.ndarray,
) -> dict[str, Any]:
    assignments = assign_events_to_templates(
        masked_ranks[:, test],
        bools[:, test],
        np.asarray(model["templates"], float),
        min_shared_channels=MIN_SHARED_CHANNELS,
    )
    cluster_rows = []
    for cluster in (0, 1):
        selected = assignments == cluster
        if int(selected.sum()) < MIN_HELDOUT_CLUSTER_EVENTS:
            raise ValueError(
                f"held-out cluster support below {MIN_HELDOUT_CLUSTER_EVENTS}: "
                f"{np.bincount(assignments[assignments >= 0], minlength=2).tolist()}"
            )
        held = masked_ranks[:, test][:, selected]
        with np.errstate(invalid="ignore"):
            mean_rank = np.asarray([
                np.nanmean(row) if np.any(np.isfinite(row)) else np.nan
                for row in held
            ])
        axis = np.asarray(model["axes"], float)[cluster]
        mapped = np.isfinite(coords).all(axis=1)
        centered = coords - np.nanmean(coords[mapped], axis=0, keepdims=True)
        along = centered @ axis
        valid = mapped & np.isfinite(mean_rank) & np.isfinite(along)
        if int(valid.sum()) < 3:
            raise ValueError("fewer than three held-out read-back contacts")
        rho = float(spearmanr(along[valid], mean_rank[valid]).correlation)
        if not np.isfinite(rho):
            raise ValueError("held-out rank read-back is degenerate")
        cluster_rows.append({
            "cluster": int(cluster),
            "rho": rho,
            "n_contacts": int(valid.sum()),
            "n_events": int(selected.sum()),
        })
    return {
        "cluster_rows": cluster_rows,
        "equal_template_rho": float(np.mean([row["rho"] for row in cluster_rows])),
        "assignment_coverage": float(np.mean(assignments >= 0)),
    }


def process_subject(subject_id: str) -> dict[str, Any]:
    dataset, subject = subject_id.split("_", 1)
    events = load_subject_propagation_events(_subject_dir(dataset, subject))
    names = [str(name) for name in events["channel_names"]]
    ranks = np.asarray(events["ranks"], float)
    bools = np.asarray(events["bools"], bool)
    blocks = np.asarray(events["block_ids"], int)
    coords_record = load_subject_coords(dataset, subject, names)
    coords = np.asarray(coords_record.coords_array_in_requested_order, float)
    masked = mask_phantom_ranks(ranks, bools, normalize=True)
    directions = np.asarray(
        fit_event_directions_3d(masked, coords, min_contacts=3)["directions"], float
    )
    fold_rows = []
    for fold_index, (train, test) in enumerate(_alternating_block_folds(blocks)):
        model = _fit_fold_model(ranks, bools, directions, coords, train)
        readback = _readback_fold(model, masked, bools, coords, test)
        fold_rows.append({
            "fold": fold_index,
            "n_train_events": int(train.size),
            "n_test_events": int(test.size),
            "train_cluster_counts": model["counts"].tolist(),
            "train_cluster_spatial_counts": model["spatial_counts"].tolist(),
            **readback,
        })
    rhos = [
        cluster["rho"]
        for fold in fold_rows
        for cluster in fold["cluster_rows"]
    ]
    return {
        "subject_id": subject_id,
        "dataset": dataset,
        "subject": subject,
        "status": "ok",
        "n_events": int(ranks.shape[1]),
        "n_contacts": int(ranks.shape[0]),
        "n_direction_estimable": int(np.isfinite(directions).all(axis=1).sum()),
        "heldout_rho": float(np.mean(rhos)),
        "folds": fold_rows,
    }


def _subjects() -> list[str]:
    with (SOURCE_ROOT / "subject_spatial_information_gain.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    subjects = [row["subject_id"] for row in rows if row.get("status") == "ok"]
    if len(subjects) != 26:
        raise ValueError(f"expected 26 all-event Timing+Space subjects, found {len(subjects)}")
    return subjects


def _dataset_summary(records: Sequence[dict[str, Any]], dataset: str | None) -> dict[str, Any]:
    values = np.asarray([
        record["heldout_rho"]
        for record in records
        if dataset is None or record["dataset"] == dataset
    ], float)
    q25, q75 = np.percentile(values, [25, 75])
    test = wilcoxon(values, alternative="greater")
    return {
        "n": int(values.size),
        "median_spearman_rho": float(np.median(values)),
        "iqr_spearman_rho": [float(q25), float(q75)],
        "n_positive": int(np.sum(values > 0)),
        "one_sided_wilcoxon_greater_p": float(test.pvalue),
    }


def run(*, workers: int) -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    per_subject = OUT_ROOT / "per_subject"
    per_subject.mkdir(parents=True, exist_ok=True)
    records = []
    failures = []
    def job(subject: str) -> dict[str, Any]:
        cached = per_subject / f"{subject}.json"
        if cached.exists():
            record = json.loads(cached.read_text(encoding="utf-8"))
            if record.get("status") == "ok":
                return record
        return process_subject(subject)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(job, subject): subject for subject in _subjects()}
        for future in as_completed(future_map):
            subject = future_map[future]
            try:
                record = future.result()
                records.append(record)
                (per_subject / f"{subject}.json").write_text(
                    json.dumps(record, indent=2) + "\n", encoding="utf-8"
                )
                print(f"OK {subject}: rho={record['heldout_rho']:.3f}", flush=True)
            except Exception as exc:
                failures.append({"subject_id": subject, "error": f"{type(exc).__name__}: {exc}"})
                print(f"FAIL {subject}: {exc}", flush=True)
    records.sort(key=lambda row: row["subject_id"])
    if failures:
        raise RuntimeError(f"rank read-back failed for {len(failures)} subjects: {failures}")
    with (OUT_ROOT / "subject_rank_readback.csv").open("w", newline="") as handle:
        columns = ["subject_id", "dataset", "subject", "heldout_rho", "n_events", "n_contacts", "n_direction_estimable"]
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in columns})
    summary = {
        "contract": "all_event_timing_plus_space_crossfit_rank_readback_v1",
        "statistical_unit": "patient",
        "template_definition": "all training-fold interictal events enter timing; finite event directions add a masked spatial view",
        "split": "two-way alternating recording-block cross-fit",
        "heldout_assignment": "rank-template distance only",
        "outcome": "equal-fold equal-template Spearman rho between training-template early-to-late axis coordinate and held-out mean contact rank",
        "ictal_input": "none",
        "source_cohort": str((SOURCE_ROOT / "subject_spatial_information_gain.csv").relative_to(ROOT)),
        "summary": {
            "yuquan": _dataset_summary(records, "yuquan"),
            "epilepsiae": _dataset_summary(records, "epilepsiae"),
            "cohort": _dataset_summary(records, None),
        },
        "failures": failures,
    }
    (OUT_ROOT / "cohort_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    summary = run(workers=args.workers)
    print(json.dumps(summary["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
