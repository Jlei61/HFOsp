#!/usr/bin/env python3
"""Test Timing+Space clustering while retaining every interictal event.

All events contribute their masked timing view.  A mathematically estimable
single-event direction contributes an additional spatial view; a missing
direction is masked from spatial distance rather than deleted or filled with a
zero vector.  Held-out directions are opened only after rank-template
assignment and are scored on one common denominator for both arms.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_topic5_interictal_template_direction_rose import (  # noqa: E402
    _pretty_subject,
    _subject_dir,
)
from scripts.run_interictal_spatial_information_gain import (  # noqa: E402
    DEFAULT_COHORT,
    DEFAULT_MIN_CLUSTER_EVENTS,
    DEFAULT_N_NULL,
    DEFAULT_SEED,
    _cohort_statistics,
    _jsonable,
    load_default_subjects,
)
from src.interictal_propagation import load_subject_propagation_events  # noqa: E402
from src.interictal_spatial_information_gain import (  # noqa: E402
    compute_crossfit_all_event_spatial_information_gain,
)
from src.lagpat_rank_audit import mask_phantom_ranks  # noqa: E402
from src.seeg_coord_loader import load_subject_coords  # noqa: E402
from src.topic5_interictal_direction_rose import fit_event_directions_3d  # noqa: E402


DEFAULT_OUT = (
    ROOT / "results/interictal_propagation_masked/"
    "spatial_information_gain_all_events"
)
DEFAULT_MAX_EVENTS = 0
DEFAULT_N_TRAIN_SPATIAL_NULL = 50


def _subject_seed(subject_id: str, seed: int, suffix: str) -> int:
    token = f"{subject_id}|{suffix}".encode("utf-8")
    return int((zlib.crc32(token) + int(seed)) % (2**32 - 1))


def _select_events(
    n_events: int,
    *,
    max_events: int,
    subject_id: str,
    seed: int,
) -> np.ndarray:
    if max_events <= 0 or n_events <= max_events:
        return np.arange(n_events, dtype=int)
    rng = np.random.default_rng(_subject_seed(subject_id, seed, "event_sample"))
    return np.sort(rng.choice(n_events, size=max_events, replace=False))


def load_all_event_inputs(
    subject_id: str,
    *,
    max_events: int,
    seed: int,
) -> dict[str, Any]:
    dataset, subject = subject_id.split("_", 1)
    event_record = load_subject_propagation_events(_subject_dir(dataset, subject))
    names = [str(name) for name in event_record["channel_names"]]
    ranks_all = np.asarray(event_record["ranks"], float)
    bools_all = np.asarray(event_record["bools"], bool)
    blocks_all = np.asarray(event_record["block_ids"], int)
    if ranks_all.shape != bools_all.shape or ranks_all.shape[0] != len(names):
        raise ValueError("event channel/rank/participation shapes do not align")
    if blocks_all.shape != (ranks_all.shape[1],):
        raise ValueError("recording blocks do not align with events")

    coordinate_record = load_subject_coords(dataset, subject, names)
    coords = np.asarray(
        coordinate_record.coords_array_in_requested_order, float
    )
    mapped = np.asarray(
        coordinate_record.mapped_mask_in_requested_order, bool
    ) & np.isfinite(coords).all(axis=1)
    if int(mapped.sum()) < 3:
        raise ValueError(f"only {int(mapped.sum())} mapped contacts")

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
    return {
        "subject_id": subject_id,
        "dataset": dataset,
        "subject": subject,
        "channel_names": names,
        "n_channels": len(names),
        "n_mapped_channels": int(mapped.sum()),
        "n_events_total": int(ranks_all.shape[1]),
        "selection": selection,
        "ranks": ranks,
        "bools": bools,
        "masked_ranks": masked,
        "blocks": blocks,
        "coords": coords,
        "directions": directions,
        "direction_estimable": direction_estimable,
        "n_valid_contacts_per_event": np.asarray(
            direction_fit["n_valid_contacts"], int
        ),
        "effective_rank_per_event": np.asarray(
            direction_fit["effective_rank"], int
        ),
    }


def process_subject(
    subject_id: str,
    *,
    max_events: int,
    min_cluster_events: int,
    n_null: int,
    n_train_spatial_null: int,
    seed: int,
) -> dict[str, Any]:
    inputs = load_all_event_inputs(
        subject_id,
        max_events=max_events,
        seed=seed,
    )
    analysis = compute_crossfit_all_event_spatial_information_gain(
        np.asarray(inputs["ranks"], float),
        np.asarray(inputs["bools"], bool),
        np.asarray(inputs["directions"], float),
        np.asarray(inputs["blocks"], int),
        np.asarray(inputs["coords"], float),
        min_cluster_events=min_cluster_events,
        n_null=n_null,
        n_train_spatial_null=n_train_spatial_null,
        seed=_subject_seed(subject_id, seed, "all_event_heldout_direction_null"),
    )
    n_sampled = int(len(inputs["selection"]))
    n_direction = int(np.asarray(inputs["direction_estimable"], bool).sum())
    return {
        "subject_id": subject_id,
        "pretty_subject": _pretty_subject(subject_id),
        "dataset": inputs["dataset"],
        "subject": inputs["subject"],
        "status": "ok",
        "drop_reason": "",
        "n_channels": int(inputs["n_channels"]),
        "n_mapped_channels": int(inputs["n_mapped_channels"]),
        "n_events_total": int(inputs["n_events_total"]),
        "n_events_sampled": n_sampled,
        "n_events_used_for_clustering": n_sampled,
        "n_direction_estimable": n_direction,
        "direction_estimable_fraction": float(n_direction / max(1, n_sampled)),
        "n_blocks": int(np.unique(inputs["blocks"]).size),
        "timing_only_score": analysis["timing_only_score"],
        "timing_plus_space_score": analysis["timing_plus_space_score"],
        "spatial_information_gain": analysis["spatial_information_gain"],
        "event_weighted_timing_only_score": analysis[
            "event_weighted_timing_only_score"
        ],
        "event_weighted_timing_plus_space_score": analysis[
            "event_weighted_timing_plus_space_score"
        ],
        "event_weighted_spatial_information_gain": analysis[
            "event_weighted_spatial_information_gain"
        ],
        "mean_train_label_ami": analysis["mean_train_label_ami"],
        "train_spatial_shuffle_real_hybrid_score": analysis[
            "timing_plus_space_score"
        ],
        "train_spatial_shuffle_null_hybrid_median": float(np.median(
            analysis["train_spatial_shuffle_null_timing_plus_space_score"]
        )),
        "train_spatial_shuffle_real_minus_null": float(
            analysis["timing_plus_space_score"] - np.median(
                analysis["train_spatial_shuffle_null_timing_plus_space_score"]
            )
        ),
        "event_weighted_train_spatial_shuffle_real_hybrid_score": analysis[
            "event_weighted_timing_plus_space_score"
        ],
        "event_weighted_train_spatial_shuffle_null_hybrid_median": float(
            np.median(
                analysis[
                    "event_weighted_train_spatial_shuffle_null_timing_plus_space_score"
                ]
            )
        ),
        "event_weighted_train_spatial_shuffle_real_minus_null": float(
            analysis["event_weighted_timing_plus_space_score"] - np.median(
                analysis[
                    "event_weighted_train_spatial_shuffle_null_timing_plus_space_score"
                ]
            )
        ),
        "folds": analysis["folds"],
        "analysis_contract": analysis["contract"],
        "direction_shuffle_null_timing_only_score": analysis[
            "direction_shuffle_null_timing_only_score"
        ],
        "direction_shuffle_null_timing_plus_space_score": analysis[
            "direction_shuffle_null_timing_plus_space_score"
        ],
        "direction_shuffle_null_gain": analysis[
            "direction_shuffle_null_gain"
        ],
        "event_weighted_direction_shuffle_null_timing_only_score": analysis[
            "event_weighted_direction_shuffle_null_timing_only_score"
        ],
        "event_weighted_direction_shuffle_null_timing_plus_space_score": analysis[
            "event_weighted_direction_shuffle_null_timing_plus_space_score"
        ],
        "event_weighted_direction_shuffle_null_gain": analysis[
            "event_weighted_direction_shuffle_null_gain"
        ],
        "train_spatial_shuffle_null_timing_plus_space_score": analysis[
            "train_spatial_shuffle_null_timing_plus_space_score"
        ],
        "train_spatial_shuffle_null_gain_vs_timing": analysis[
            "train_spatial_shuffle_null_gain_vs_timing"
        ],
        "event_weighted_train_spatial_shuffle_null_timing_plus_space_score": analysis[
            "event_weighted_train_spatial_shuffle_null_timing_plus_space_score"
        ],
        "event_weighted_train_spatial_shuffle_null_gain_vs_timing": analysis[
            "event_weighted_train_spatial_shuffle_null_gain_vs_timing"
        ],
        "source": {
            "event_artifact_root": _subject_dir(
                str(inputs["dataset"]), str(inputs["subject"])
            ),
            "coordinate_loader": "seeg_coord_loader_v3p1",
            "event_selection": "all events" if max_events <= 0 else "deterministic sample",
            "max_events": int(max_events),
            "sample_seed": _subject_seed(subject_id, seed, "event_sample"),
        },
    }


CSV_COLUMNS = [
    "subject_id",
    "pretty_subject",
    "dataset",
    "subject",
    "status",
    "drop_reason",
    "n_channels",
    "n_mapped_channels",
    "n_events_total",
    "n_events_sampled",
    "n_events_used_for_clustering",
    "n_direction_estimable",
    "direction_estimable_fraction",
    "n_blocks",
    "timing_only_score",
    "timing_plus_space_score",
    "spatial_information_gain",
    "event_weighted_timing_only_score",
    "event_weighted_timing_plus_space_score",
    "event_weighted_spatial_information_gain",
    "mean_train_label_ami",
    "train_spatial_shuffle_real_hybrid_score",
    "train_spatial_shuffle_null_hybrid_median",
    "train_spatial_shuffle_real_minus_null",
    "event_weighted_train_spatial_shuffle_real_hybrid_score",
    "event_weighted_train_spatial_shuffle_null_hybrid_median",
    "event_weighted_train_spatial_shuffle_real_minus_null",
]


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=CSV_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in CSV_COLUMNS})


def _train_spatial_shuffle_statistics(
    rows: Sequence[Mapping[str, Any]],
    *,
    event_weighted: bool = False,
) -> dict[str, Any]:
    prefix = "event_weighted_" if event_weighted else ""
    real = np.asarray([
        row[f"{prefix}train_spatial_shuffle_real_hybrid_score"] for row in rows
    ], float)
    null = np.asarray([
        row[f"{prefix}train_spatial_shuffle_null_hybrid_median"] for row in rows
    ], float)
    difference = real - null
    return {
        "n": int(len(rows)),
        "real_hybrid_median": float(np.median(real)),
        "patient_null_median": float(np.median(null)),
        "real_minus_patient_null_median": float(np.median(difference)),
        "n_real_greater_than_null": int(np.sum(difference > 0)),
        "paired_wilcoxon_real_greater_p": float(
            wilcoxon(real, null, alternative="greater").pvalue
        ),
    }


def run(
    subjects: Sequence[str],
    *,
    out_dir: Path,
    cohort_path: Path,
    max_events: int,
    min_cluster_events: int,
    n_null: int,
    n_train_spatial_null: int,
    seed: int,
    workers: int,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if n_train_spatial_null < 1:
        raise ValueError("n_train_spatial_null must be positive")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_subject = out_dir / "per_subject"
    per_subject.mkdir(parents=True, exist_ok=True)

    def job(subject_id: str) -> dict[str, Any]:
        return process_subject(
            subject_id,
            max_events=max_events,
            min_cluster_events=min_cluster_events,
            n_null=n_null,
            n_train_spatial_null=n_train_spatial_null,
            seed=seed,
        )

    completed: dict[str, dict[str, Any]] = {}
    failures: dict[str, Exception] = {}
    if workers == 1:
        for subject_id in subjects:
            try:
                completed[subject_id] = job(subject_id)
            except Exception as exc:  # pragma: no cover - cohort I/O boundary
                failures[subject_id] = exc
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(job, subject_id): subject_id for subject_id in subjects}
            for future in as_completed(futures):
                subject_id = futures[future]
                try:
                    completed[subject_id] = future.result()
                except Exception as exc:  # pragma: no cover - cohort I/O boundary
                    failures[subject_id] = exc

    rows = []
    eligible = []
    primary_null_gain = []
    primary_null_timing = []
    primary_null_hybrid = []
    event_null_gain = []
    event_null_timing = []
    event_null_hybrid = []
    train_spatial_null_hybrid = []
    train_spatial_null_gain = []
    event_train_spatial_null_hybrid = []
    event_train_spatial_null_gain = []
    for index, subject_id in enumerate(subjects, 1):
        if subject_id in completed:
            result = completed[subject_id]
            primary_null_gain.append(np.asarray(
                result.pop("direction_shuffle_null_gain"), float
            ))
            primary_null_timing.append(np.asarray(
                result.pop("direction_shuffle_null_timing_only_score"), float
            ))
            primary_null_hybrid.append(np.asarray(
                result.pop("direction_shuffle_null_timing_plus_space_score"), float
            ))
            event_null_gain.append(np.asarray(
                result.pop("event_weighted_direction_shuffle_null_gain"), float
            ))
            event_null_timing.append(np.asarray(
                result.pop("event_weighted_direction_shuffle_null_timing_only_score"), float
            ))
            event_null_hybrid.append(np.asarray(
                result.pop("event_weighted_direction_shuffle_null_timing_plus_space_score"), float
            ))
            train_spatial_null_hybrid.append(np.asarray(
                result.pop("train_spatial_shuffle_null_timing_plus_space_score"), float
            ))
            train_spatial_null_gain.append(np.asarray(
                result.pop("train_spatial_shuffle_null_gain_vs_timing"), float
            ))
            event_train_spatial_null_hybrid.append(np.asarray(
                result.pop(
                    "event_weighted_train_spatial_shuffle_null_timing_plus_space_score"
                ), float
            ))
            event_train_spatial_null_gain.append(np.asarray(
                result.pop(
                    "event_weighted_train_spatial_shuffle_null_gain_vs_timing"
                ), float
            ))
            row = result
            eligible.append(result)
            print(
                f"[{index:02d}/{len(subjects)}] {subject_id}: "
                f"all={result['n_events_used_for_clustering']} "
                f"direction={result['n_direction_estimable']} "
                f"timing={result['timing_only_score']:.4f} "
                f"+space={result['timing_plus_space_score']:.4f} "
                f"gain={result['spatial_information_gain']:+.4f}",
                flush=True,
            )
        else:
            exc = failures[subject_id]
            row = {
                "subject_id": subject_id,
                "pretty_subject": _pretty_subject(subject_id),
                "dataset": subject_id.split("_", 1)[0],
                "subject": subject_id.split("_", 1)[1],
                "status": "skip",
                "drop_reason": f"{type(exc).__name__}: {exc}"[:500],
            }
            print(
                f"[{index:02d}/{len(subjects)}] {subject_id}: skip ({exc})",
                flush=True,
            )
        rows.append(row)
        (per_subject / f"{subject_id}.json").write_text(
            json.dumps(_jsonable(row), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if not eligible:
        raise RuntimeError("no subject passed the all-event spatial-information contract")
    cohort_primary_null = np.median(np.vstack(primary_null_gain), axis=0)
    primary_statistics = _cohort_statistics(
        eligible,
        cohort_primary_null,
        seed=seed,
        cohort_null_hybrid_score=np.median(
            np.vstack(primary_null_hybrid), axis=0
        ),
    )
    event_rows = [
        {
            **row,
            "timing_only_score": row["event_weighted_timing_only_score"],
            "timing_plus_space_score": row[
                "event_weighted_timing_plus_space_score"
            ],
            "spatial_information_gain": row[
                "event_weighted_spatial_information_gain"
            ],
        }
        for row in eligible
    ]
    cohort_event_null = np.median(np.vstack(event_null_gain), axis=0)
    event_statistics = _cohort_statistics(
        event_rows,
        cohort_event_null,
        seed=seed + 31,
        cohort_null_hybrid_score=np.median(
            np.vstack(event_null_hybrid), axis=0
        ),
    )
    summary = {
        "analysis": "interictal_spatial_information_gain_all_events",
        "status": "complete",
        "contract": {
            "scientific_question": "Does real-coordinate spatial direction improve held-out direction consistency when every interictal event remains in template discovery?",
            "statistical_unit": "patient",
            "training_event_universe": "all available interictal events in both arms",
            "temporal_channel_universe": "all event-artifact channels; unmapped channels remain in timing features",
            "spatial_missingness": "direction unavailable means a masked spatial view, not event deletion or zero-vector imputation",
            "hard_event_qc_used": False,
            "heldout_rule": "alternating recording-block cross-fit; rank-template assignment only",
            "direction_score_domain": "common minimally direction-estimable and rank-assignable held-out events after both models freeze",
            "primary_metric": "same equal-fold equal-cluster signed-cosine score as Figure 2B",
            "sensitivity_metric": "held-out-event-weighted signed cosine",
            "primary_test": "one-sided paired Wilcoxon across patients",
            "ictal_input": "none",
            "max_events_per_subject": int(max_events),
            "min_train_cluster_events": int(min_cluster_events),
            "n_direction_shuffle_draws": int(n_null),
            "n_training_spatial_shuffle_draws": int(n_train_spatial_null),
            "seed": int(seed),
            "claim_boundary": "tests added held-out directional organization without hard event QC; events with mathematically undefined direction still affect clustering but cannot be a direction-score outcome",
        },
        "cohort_flow": {
            "n_requested": int(len(subjects)),
            "n_eligible": int(len(eligible)),
            "n_skipped": int(len(subjects) - len(eligible)),
            "skipped": [
                {"subject_id": row["subject_id"], "reason": row["drop_reason"]}
                for row in rows if row["status"] != "ok"
            ],
        },
        "cohort_statistics": primary_statistics,
        "event_weighted_sensitivity": event_statistics,
        "training_spatial_shuffle_control": {
            "null_hypothesis": "conditional on patient, recording block and direction availability, event timing patterns are unrelated to finite spatial direction vectors",
            "permutation": "training-fold finite directions permuted within recording block; missingness and held-out data unchanged",
            "n_draws_per_patient": int(n_train_spatial_null),
            "primary": _train_spatial_shuffle_statistics(eligible),
            "event_weighted_sensitivity": _train_spatial_shuffle_statistics(
                eligible, event_weighted=True
            ),
        },
    }
    _write_csv(eligible, out_dir / "subject_spatial_information_gain.csv")
    _write_csv(rows, out_dir / "run_status.csv")
    np.savez_compressed(
        out_dir / "cohort_direction_shuffle_null.npz",
        subject_ids=np.asarray([row["subject_id"] for row in eligible]),
        patient_null_gain=np.vstack(primary_null_gain),
        patient_null_timing_only_score=np.vstack(primary_null_timing),
        patient_null_timing_plus_space_score=np.vstack(primary_null_hybrid),
        cohort_median_null_gain=cohort_primary_null,
        patient_event_weighted_null_gain=np.vstack(event_null_gain),
        patient_event_weighted_null_timing_only_score=np.vstack(event_null_timing),
        patient_event_weighted_null_timing_plus_space_score=np.vstack(event_null_hybrid),
        cohort_event_weighted_median_null_gain=cohort_event_null,
        n_null=np.asarray(n_null),
        seed=np.asarray(seed),
    )
    np.savez_compressed(
        out_dir / "cohort_training_spatial_shuffle_null.npz",
        subject_ids=np.asarray([row["subject_id"] for row in eligible]),
        patient_real_hybrid_score=np.asarray([
            row["train_spatial_shuffle_real_hybrid_score"] for row in eligible
        ], float),
        patient_null_hybrid_score=np.vstack(train_spatial_null_hybrid),
        patient_null_gain_vs_timing=np.vstack(train_spatial_null_gain),
        patient_null_hybrid_median=np.asarray([
            row["train_spatial_shuffle_null_hybrid_median"] for row in eligible
        ], float),
        patient_real_minus_null=np.asarray([
            row["train_spatial_shuffle_real_minus_null"] for row in eligible
        ], float),
        patient_event_weighted_real_hybrid_score=np.asarray([
            row["event_weighted_train_spatial_shuffle_real_hybrid_score"]
            for row in eligible
        ], float),
        patient_event_weighted_null_hybrid_score=np.vstack(
            event_train_spatial_null_hybrid
        ),
        patient_event_weighted_null_gain_vs_timing=np.vstack(
            event_train_spatial_null_gain
        ),
        patient_event_weighted_null_hybrid_median=np.asarray([
            row["event_weighted_train_spatial_shuffle_null_hybrid_median"]
            for row in eligible
        ], float),
        patient_event_weighted_real_minus_null=np.asarray([
            row["event_weighted_train_spatial_shuffle_real_minus_null"]
            for row in eligible
        ], float),
        n_train_spatial_null=np.asarray(n_train_spatial_null),
        seed=np.asarray(seed),
    )
    (out_dir / "spatial_information_gain_summary.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--cohort-csv", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-events", type=int, default=DEFAULT_MAX_EVENTS)
    parser.add_argument(
        "--min-cluster-events", type=int, default=DEFAULT_MIN_CLUSTER_EVENTS
    )
    parser.add_argument("--n-null", type=int, default=DEFAULT_N_NULL)
    parser.add_argument(
        "--n-train-spatial-null",
        type=int,
        default=DEFAULT_N_TRAIN_SPATIAL_NULL,
        help="training-fold within-block spatial-direction permutations",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    subjects = args.subjects or load_default_subjects(args.cohort_csv)
    summary = run(
        subjects,
        out_dir=args.out_dir,
        cohort_path=args.cohort_csv,
        max_events=args.max_events,
        min_cluster_events=args.min_cluster_events,
        n_null=args.n_null,
        n_train_spatial_null=args.n_train_spatial_null,
        seed=args.seed,
        workers=args.workers,
    )
    stats = summary["cohort_statistics"]
    print(
        f"cohort n={stats['n']} median_gain={stats['median_gain']:.4f} "
        f"paired_p={stats['paired_wilcoxon_greater_p']:.4g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
