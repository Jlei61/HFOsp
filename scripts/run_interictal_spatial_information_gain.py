#!/usr/bin/env python3
"""Run the held-out spatial-information gain analysis for interictal events."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_topic5_interictal_template_direction_rose import (  # noqa: E402
    FROZEN_ROOT,
    _load_frozen_record,
    _pretty_subject,
    _subject_dir,
)
from src.interictal_propagation import load_subject_propagation_events  # noqa: E402
from src.interictal_spatial_information_gain import (  # noqa: E402
    compute_crossfit_spatial_information_gain,
)
from src.lagpat_rank_audit import mask_phantom_ranks  # noqa: E402
from src.topic5_interictal_direction_rose import (  # noqa: E402
    assess_event_direction_qc,
    fit_event_directions_3d,
)
from src.topic5_tspectral_field_concordance import bootstrap_median_ci  # noqa: E402


DEFAULT_OUT = (
    ROOT / "results/interictal_propagation_masked/spatial_information_gain"
)
DEFAULT_COHORT = (
    ROOT
    / "results/interictal_propagation_masked/axis_representativeness/"
    "subject_folded_axis_representativeness.csv"
)
DEFAULT_MAX_EVENTS = 5000
DEFAULT_MIN_CLUSTER_EVENTS = 20
DEFAULT_N_NULL = 1000
DEFAULT_SEED = 20260825


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.relative_to(ROOT)) if value.is_relative_to(ROOT) else str(value)
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


def _subject_seed(subject_id: str, seed: int, suffix: str) -> int:
    token = f"{subject_id}|{suffix}".encode("utf-8")
    return int((zlib.crc32(token) + int(seed)) % (2**32 - 1))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_default_subjects(cohort_path: Path) -> list[str]:
    with cohort_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    subject_ids = [
        str(row["subject_id"])
        for row in rows
        if row.get("subject_id") and row.get("mean_signed_cosine", "") != ""
    ]
    if len(subject_ids) != len(set(subject_ids)):
        raise ValueError("cohort source contains duplicate subject IDs")
    if not subject_ids:
        raise ValueError("cohort source contains no eligible subjects")
    return subject_ids


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


def load_subject_analysis_inputs(
    subject_id: str,
    *,
    max_events: int,
    seed: int,
) -> Dict[str, Any]:
    """Load the exact sampled, aligned, geometry-QC event pool for one patient."""
    frozen = _load_frozen_record(subject_id)
    dataset, subject = subject_id.split("_", 1)
    event_record = load_subject_propagation_events(_subject_dir(dataset, subject))

    event_names = [str(value) for value in event_record["channel_names"]]
    frozen_names = [str(value) for value in frozen["names"]]
    event_index = {name: index for index, name in enumerate(event_names)}
    missing = [name for name in frozen_names if name not in event_index]
    if missing:
        raise ValueError(f"coordinate contacts absent from event data: {missing[:5]}")
    contact_indices = np.asarray([event_index[name] for name in frozen_names], int)

    all_ranks = np.asarray(event_record["ranks"], float)[contact_indices]
    all_bools = np.asarray(event_record["bools"], bool)[contact_indices]
    all_blocks = np.asarray(event_record["block_ids"], int)
    selection = _select_events(
        all_ranks.shape[1],
        max_events=max_events,
        subject_id=subject_id,
        seed=seed,
    )
    ranks = all_ranks[:, selection]
    bools = all_bools[:, selection]
    blocks = all_blocks[selection]
    masked = mask_phantom_ranks(ranks, bools, normalize=True)
    coords = np.asarray(frozen["coords"], float)

    direction_fit = fit_event_directions_3d(masked, coords, min_contacts=3)
    direction_qc = assess_event_direction_qc(
        masked,
        coords,
        frozen["shafts"],
        directions=direction_fit["directions"],
        n_valid_contacts=direction_fit["n_valid_contacts"],
        effective_rank=direction_fit["effective_rank"],
    )
    qc_pass = np.asarray(direction_qc["passes"], bool)
    qc_indices = np.flatnonzero(qc_pass)
    return {
        "subject_id": subject_id,
        "dataset": dataset,
        "subject": subject,
        "frozen": frozen,
        "all_ranks": all_ranks,
        "selection": selection,
        "ranks": ranks,
        "bools": bools,
        "blocks": blocks,
        "masked_ranks": masked,
        "coords": coords,
        "directions": np.asarray(direction_fit["directions"], float),
        "direction_qc": direction_qc,
        "qc_indices": qc_indices,
    }


def process_subject(
    subject_id: str,
    *,
    max_events: int,
    min_cluster_events: int,
    n_null: int,
    seed: int,
) -> Dict[str, Any]:
    inputs = load_subject_analysis_inputs(
        subject_id,
        max_events=max_events,
        seed=seed,
    )
    dataset = str(inputs["dataset"])
    subject = str(inputs["subject"])
    all_ranks = np.asarray(inputs["all_ranks"], float)
    selection = np.asarray(inputs["selection"], int)
    ranks = np.asarray(inputs["ranks"], float)
    bools = np.asarray(inputs["bools"], bool)
    blocks = np.asarray(inputs["blocks"], int)
    coords = np.asarray(inputs["coords"], float)
    directions = np.asarray(inputs["directions"], float)
    qc_indices = np.asarray(inputs["qc_indices"], int)
    if qc_indices.size < 4 * min_cluster_events:
        raise ValueError(
            f"only {qc_indices.size} QC-clean events; need at least "
            f"{4 * min_cluster_events} before block cross-fit"
        )

    analysis = compute_crossfit_spatial_information_gain(
        ranks[:, qc_indices],
        bools[:, qc_indices],
        directions[qc_indices],
        blocks[qc_indices],
        coords,
        min_cluster_events=min_cluster_events,
        n_null=n_null,
        seed=_subject_seed(subject_id, seed, "heldout_direction_null"),
    )
    frozen_path = FROZEN_ROOT / "per_subject" / f"{subject_id}.json"
    return {
        "subject_id": subject_id,
        "pretty_subject": _pretty_subject(subject_id),
        "dataset": dataset,
        "subject": subject,
        "status": "ok",
        "drop_reason": "",
        "n_events_total": int(all_ranks.shape[1]),
        "n_events_sampled": int(selection.size),
        "n_qc_clean_events": int(qc_indices.size),
        "qc_retention": float(qc_indices.size / max(1, selection.size)),
        "n_blocks_sampled": int(np.unique(blocks).size),
        "n_blocks_qc_clean": int(np.unique(blocks[qc_indices]).size),
        "timing_only_score": analysis["timing_only_score"],
        "timing_plus_space_score": analysis["timing_plus_space_score"],
        "spatial_information_gain": analysis["spatial_information_gain"],
        "mean_train_label_ami": analysis["mean_train_label_ami"],
        "folds": analysis["folds"],
        "direction_shuffle_null_timing_only_score": analysis[
            "direction_shuffle_null_timing_only_score"
        ],
        "direction_shuffle_null_timing_plus_space_score": analysis[
            "direction_shuffle_null_timing_plus_space_score"
        ],
        "direction_shuffle_null_gain": analysis["direction_shuffle_null_gain"],
        "qc_contract": {
            "minimum_mapped_participating_contacts": 6,
            "minimum_participating_shafts": 2,
            "minimum_effective_coordinate_rank": 2,
            "minimum_loco_valid_fraction": 0.8,
            "minimum_loco_median_signed_cosine": 0.8,
            "template_angle_used_for_inclusion": False,
        },
        "analysis_contract": analysis["contract"],
        "source": {
            "frozen_coordinate_record": frozen_path,
            "frozen_coordinate_record_sha256": _sha256(frozen_path),
            "event_artifact_root": _subject_dir(dataset, subject),
            "sample_seed": _subject_seed(subject_id, seed, "event_sample"),
            "selection_count": int(selection.size),
        },
    }


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    columns = [
        "subject_id",
        "pretty_subject",
        "dataset",
        "subject",
        "status",
        "drop_reason",
        "n_events_total",
        "n_events_sampled",
        "n_qc_clean_events",
        "qc_retention",
        "n_blocks_sampled",
        "n_blocks_qc_clean",
        "timing_only_score",
        "timing_plus_space_score",
        "spatial_information_gain",
        "mean_train_label_ami",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in columns})


def _cohort_statistics(
    eligible: Sequence[Mapping[str, Any]],
    cohort_null: np.ndarray,
    *,
    seed: int,
    cohort_null_hybrid_score: np.ndarray | None = None,
) -> Dict[str, Any]:
    timing = np.asarray([row["timing_only_score"] for row in eligible], float)
    hybrid = np.asarray([row["timing_plus_space_score"] for row in eligible], float)
    gain = hybrid - timing
    ci_lo, ci_hi = bootstrap_median_ci(gain, n_boot=10000, seed=seed)
    paired_p = (
        float(wilcoxon(hybrid, timing, alternative="greater").pvalue)
        if np.any(hybrid != timing)
        else 1.0
    )
    observed_median = float(np.median(gain))
    null_p = float(
        (1 + np.sum(cohort_null >= observed_median - 1e-15))
        / (len(cohort_null) + 1)
    )

    subgroup = {}
    for dataset in sorted({str(row["dataset"]) for row in eligible}):
        values = np.asarray([
            row["spatial_information_gain"]
            for row in eligible
            if row["dataset"] == dataset
        ], float)
        subgroup[dataset] = {
            "n": int(values.size),
            "median_gain": float(np.median(values)),
            "n_positive": int(np.sum(values > 0)),
        }
    output = {
        "n": int(gain.size),
        "timing_only_median": float(np.median(timing)),
        "timing_plus_space_median": float(np.median(hybrid)),
        "median_gain": observed_median,
        "median_gain_bootstrap_ci95": [float(ci_lo), float(ci_hi)],
        "n_positive_gain": int(np.sum(gain > 0)),
        "n_negative_gain": int(np.sum(gain < 0)),
        "paired_wilcoxon_greater_p": paired_p,
        "direction_shuffle_null": {
            "n_draws": int(len(cohort_null)),
            "cohort_median_null_median": float(np.median(cohort_null)),
            "cohort_median_null_ci95": [
                float(np.percentile(cohort_null, 2.5)),
                float(np.percentile(cohort_null, 97.5)),
            ],
            "empirical_p_observed_median_greater": null_p,
        },
        "dataset_sensitivity": subgroup,
    }
    if cohort_null_hybrid_score is not None:
        hybrid_ci_lo, hybrid_ci_hi = bootstrap_median_ci(
            hybrid, n_boot=10000, seed=seed + 1
        )
        timing_ci_lo, timing_ci_hi = bootstrap_median_ci(
            timing, n_boot=10000, seed=seed + 2
        )
        absolute_null = np.asarray(cohort_null_hybrid_score, float)
        hybrid_median = float(np.median(hybrid))
        output["absolute_direction_scores"] = {
            "timing_only_median": float(np.median(timing)),
            "timing_only_bootstrap_ci95": [
                float(timing_ci_lo), float(timing_ci_hi)
            ],
            "timing_plus_space_median": hybrid_median,
            "timing_plus_space_bootstrap_ci95": [
                float(hybrid_ci_lo), float(hybrid_ci_hi)
            ],
            "hybrid_direction_shuffle_null_median": float(
                np.median(absolute_null)
            ),
            "hybrid_direction_shuffle_null_ci95": [
                float(np.percentile(absolute_null, 2.5)),
                float(np.percentile(absolute_null, 97.5)),
            ],
            "hybrid_observed_vs_null_empirical_p": float(
                (1 + np.sum(absolute_null >= hybrid_median - 1e-15))
                / (len(absolute_null) + 1)
            ),
        }
    return output


def run(
    subjects: Sequence[str],
    *,
    out_dir: Path,
    cohort_path: Path,
    max_events: int,
    min_cluster_events: int,
    n_null: int,
    seed: int,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_subject = out_dir / "per_subject"
    per_subject.mkdir(parents=True, exist_ok=True)
    rows = []
    eligible = []
    null_rows = []
    null_timing_rows = []
    null_hybrid_rows = []
    for index, subject_id in enumerate(subjects, 1):
        try:
            result = process_subject(
                subject_id,
                max_events=max_events,
                min_cluster_events=min_cluster_events,
                n_null=n_null,
                seed=seed,
            )
            null_values = np.asarray(result.pop("direction_shuffle_null_gain"), float)
            null_timing = np.asarray(
                result.pop("direction_shuffle_null_timing_only_score"), float
            )
            null_hybrid = np.asarray(
                result.pop("direction_shuffle_null_timing_plus_space_score"), float
            )
            null_rows.append(null_values)
            null_timing_rows.append(null_timing)
            null_hybrid_rows.append(null_hybrid)
            eligible.append(result)
            row = result
            print(
                f"[{index:02d}/{len(subjects)}] {subject_id}: "
                f"timing={result['timing_only_score']:.4f} "
                f"+space={result['timing_plus_space_score']:.4f} "
                f"gain={result['spatial_information_gain']:+.4f}",
                flush=True,
            )
        except Exception as exc:
            row = {
                "subject_id": subject_id,
                "pretty_subject": _pretty_subject(subject_id),
                "dataset": subject_id.split("_", 1)[0],
                "subject": subject_id.split("_", 1)[1],
                "status": "skip",
                "drop_reason": str(exc)[:500],
            }
            print(
                f"[{index:02d}/{len(subjects)}] {subject_id}: skip ({exc})",
                flush=True,
            )
        rows.append(row)
        (per_subject / f"{subject_id}.json").write_text(
            json.dumps(_jsonable(row), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if not eligible:
        raise RuntimeError("no subject passed the spatial-information gain contract")
    cohort_null = np.median(np.vstack(null_rows), axis=0)
    cohort_null_timing = np.median(np.vstack(null_timing_rows), axis=0)
    cohort_null_hybrid = np.median(np.vstack(null_hybrid_rows), axis=0)
    statistics = _cohort_statistics(
        eligible,
        cohort_null,
        seed=seed,
        cohort_null_hybrid_score=cohort_null_hybrid,
    )
    summary = {
        "analysis": "interictal_spatial_information_gain",
        "status": "complete",
        "contract": {
            "scientific_question": "Does patient-specific electrode geometry add information beyond within-event relative timing when discovering recurring interictal propagation modes?",
            "statistical_unit": "patient",
            "input_cohort": "same 26-patient two-dimensional direction-score cohort used by current Figure 2B",
            "event_subset": "same deterministic maximum-size sample in both arms; independent geometry+LOCO QC before splitting",
            "temporal_arm": "KMeans k=2 on canonical masked-rank features",
            "hybrid_arm": "same temporal features plus train-fold 3D event directions, with equal total train variance per view",
            "heldout_rule": "two-way alternating recording-block cross-fit; held-out assignment uses rank templates only",
            "primary_metric": "patient-level timing-plus-space minus timing-only held-out direction score",
            "direction_score": "equal-fold, equal-cluster mean signed cosine between held-out event direction and its assigned train-template gradient axis",
            "primary_test": "one-sided paired Wilcoxon across patients",
            "null_sensitivity": "shuffle held-out event directions within recording block after both models are frozen",
            "claim_boundary": "supports added held-out directional information in the QC-clean, geometry-estimable event subset; not unseen-patient prediction, tissue trajectory, speed, or mechanism/causality",
            "ictal_input": "none",
            "spatial_coordinates_pooled_across_patients": False,
            "k": 2,
            "max_events_per_subject": int(max_events),
            "min_train_and_test_events_per_cluster": int(min_cluster_events),
            "n_direction_shuffle_draws": int(n_null),
            "seed": int(seed),
        },
        "cohort_flow": {
            "n_requested": int(len(subjects)),
            "n_eligible": int(len(eligible)),
            "n_skipped": int(len(subjects) - len(eligible)),
            "skipped": [
                {"subject_id": row["subject_id"], "reason": row["drop_reason"]}
                for row in rows
                if row["status"] != "ok"
            ],
        },
        "cohort_statistics": statistics,
        "source": {
            "cohort_csv": cohort_path,
            "cohort_csv_sha256": _sha256(cohort_path),
            "frozen_coordinate_root": FROZEN_ROOT,
            "event_roots": {
                "yuquan": "/mnt/yuquan_data/yuquan_24h_edf",
                "epilepsiae": "/mnt/epilepsia_data/interilca_inter_results/all_data_lns",
            },
        },
        "outputs": {
            "subject_csv": out_dir / "subject_spatial_information_gain.csv",
            "run_status_csv": out_dir / "run_status.csv",
            "cohort_null_npz": out_dir / "cohort_direction_shuffle_null.npz",
        },
    }
    _write_csv(eligible, out_dir / "subject_spatial_information_gain.csv")
    _write_csv(rows, out_dir / "run_status.csv")
    np.savez_compressed(
        out_dir / "cohort_direction_shuffle_null.npz",
        subject_ids=np.asarray([row["subject_id"] for row in eligible]),
        patient_null_gain=np.vstack(null_rows),
        patient_null_timing_only_score=np.vstack(null_timing_rows),
        patient_null_timing_plus_space_score=np.vstack(null_hybrid_rows),
        cohort_median_null_gain=cohort_null,
        cohort_median_null_timing_only_score=cohort_null_timing,
        cohort_median_null_timing_plus_space_score=cohort_null_hybrid,
        n_null=np.asarray(n_null),
        seed=np.asarray(seed),
    )
    (out_dir / "spatial_information_gain_summary.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--cohort-csv", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-events", type=int, default=DEFAULT_MAX_EVENTS)
    parser.add_argument(
        "--min-cluster-events", type=int, default=DEFAULT_MIN_CLUSTER_EVENTS
    )
    parser.add_argument("--n-null", type=int, default=DEFAULT_N_NULL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = args.subjects or load_default_subjects(args.cohort_csv)
    result = run(
        subjects,
        out_dir=args.out_dir,
        cohort_path=args.cohort_csv,
        max_events=args.max_events,
        min_cluster_events=args.min_cluster_events,
        n_null=args.n_null,
        seed=args.seed,
    )
    stats = result["cohort_statistics"]
    print(
        "cohort: "
        f"n={stats['n']} median_gain={stats['median_gain']:.4f} "
        f"paired_p={stats['paired_wilcoxon_greater_p']:.4g}"
    )


if __name__ == "__main__":
    main()
