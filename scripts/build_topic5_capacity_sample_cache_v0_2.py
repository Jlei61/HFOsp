#!/usr/bin/env python3
"""Phase A2 + B2: parent parity audit and the immutable prefix/horizon cache.

Reads the frozen parent caches, re-derives every quantity the models need and
writes one npz per (patient, prefix length).  It also produces the input census,
the split-hash audit, the contact-geometry audit, the horizon denominator census
and the coverage descriptors that the heterogeneity discussion requires.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_strict_history_data_v0_2 import (  # noqa: E402
    AVAILABILITY_CONTRACT,
    DATA_FRACTIONS,
    HORIZONS,
    PRIMARY_PREFIX_LEN,
    SENSITIVITY_PREFIX_LEN,
    SUFFIX_MASK_CONTRACT,
    build_sample_set,
    count_rank_sets,
    horizon_census,
    load_ecog_patient,
    load_seeg_patient,
    save_sample_set,
    sha256_array,
)

FRAME_ROOT = ROOT / "results/topic5_dynamical_motif_rnn_v0_1/frame_cache/GEOMETRY_ONLY_PCA2"
ECOG_ROOT = ROOT / "results/topic5_ecog_physical_neighborhood_rnn_v0_1/cache"
RESULT_ROOT = ROOT / "results/topic5_capacity_constrained_history_motif_v0_2"
SOZ_ROOT = Path("/home/honglab/leijiaxin/HFOsp/results")


def soz_lookup() -> dict[str, list[str]]:
    table: dict[str, list[str]] = {}
    for dataset in ("epilepsiae", "yuquan"):
        path = SOZ_ROOT / f"{dataset}_soz_core_channels.json"
        if not path.exists():
            continue
        for subject, channels in json.loads(path.read_text()).items():
            table[f"{dataset}_{subject}"] = [str(v) for v in channels]
    return table


def _contact_geometry_row(patient, soz: dict[str, list[str]]) -> dict:
    coords = patient.coords_3d_mm
    centred = coords - coords.mean(axis=0)
    singular = np.linalg.svd(centred, full_matrices=False)[1]
    plane_reconstruction = float("nan")
    if patient.dataset == "SEEG":
        plane = np.load(FRAME_ROOT / patient.patient / "plane.npz", allow_pickle=False)
        origin = np.asarray(plane["basis_origin"], dtype=np.float64)
        basis = np.column_stack([plane["basis_u"], plane["basis_w"]]).astype(np.float64)
        projected = (coords - origin) @ basis
        plane_reconstruction = float(np.abs(projected - patient.contacts_xy_mm).max())
    annotated = soz.get(patient.patient, [])
    prefixes = {name[:2] for name in annotated}
    recorded = [name for name in patient.contact_names if name in annotated]
    recorded_shaft = [name for name in patient.contact_names if name[:2] in prefixes]
    return {
        "patient": patient.patient,
        "dataset": patient.dataset,
        "n_contacts": patient.n_contacts,
        "n_shafts": len(set(patient.shafts)),
        "singular_1": float(singular[0]),
        "singular_2": float(singular[1]),
        "singular_3": float(singular[2]) if singular.size > 2 else 0.0,
        "ratio_second_to_first": float(singular[1] / singular[0]) if singular[0] > 0 else float("nan"),
        "ratio_third_to_first": float(singular[2] / singular[0]) if singular.size > 2 and singular[0] > 0 else 0.0,
        "geometry_effective_dimension": float((singular.sum() ** 2) / (singular ** 2).sum()),
        "plane_reconstruction_max_error_mm": plane_reconstruction,
        "recorded_SOZ_annotation_available": bool(annotated),
        "recorded_SOZ_annotation_fraction": (
            len(recorded) / patient.n_contacts if annotated else float("nan")
        ),
        "recorded_SOZ_shaft_fraction": (
            len(recorded_shaft) / patient.n_contacts if annotated else float("nan")
        ),
        "n_recording_blocks": int(np.unique(patient.recording_block).size),
    }


def process_patient(job: tuple[str, str]) -> dict:
    dataset, name = job
    if dataset == "SEEG":
        patient = load_seeg_patient(FRAME_ROOT, name)
    else:
        patient = load_ecog_patient(ECOG_ROOT, name)
    soz = soz_lookup()
    out: dict = {
        "patient": patient.patient,
        "dataset": patient.dataset,
        "geometry": _contact_geometry_row(patient, soz),
        "prefix": {},
    }
    rank_set_count = count_rank_sets(patient.ranks)
    split = patient.split
    out["input_census"] = {
        "patient": patient.patient,
        "dataset": patient.dataset,
        "n_contacts": patient.n_contacts,
        "n_events": patient.n_events,
        "n_train": int((split == 0).sum()),
        "n_calibration": int((split == 1).sum()),
        "n_development_test": int((split == 2).sum()),
        "n_model_unseen": int((split == -1).sum()),
        "rank_sets_median": float(np.median(rank_set_count)),
        "rank_sets_min": int(rank_set_count.min()),
        "rank_sets_max": int(rank_set_count.max()),
        "tied_rank_set_event_fraction": float(
            ((patient.ranks >= 0).sum(axis=1) != rank_set_count).mean()
        ),
        "n_recording_blocks": int(np.unique(patient.recording_block).size),
    }
    out["split_audit"] = {
        "patient": patient.patient,
        "dataset": patient.dataset,
        "split_sha256": sha256_array(split.astype(np.int8)),
        "ranks_sha256": sha256_array(patient.ranks.astype(np.int16)),
        "contact_names_sha256": sha256_array(np.asarray(patient.contact_names, dtype="<U32")),
        "parent_events_sha256": patient.provenance.get("events_sha256"),
        "parent_split_sha256": (
            patient.provenance.get("split_audit", {}).get("split_sha256")
            if patient.dataset == "SEEG" else None
        ),
        "parent_split_matches": (
            sha256_array(split.astype(np.int8))
            == patient.provenance.get("split_audit", {}).get("split_sha256")
            if patient.dataset == "SEEG" else None
        ),
        "model_unseen_equals_parent_heldout": (
            patient.provenance.get("split_audit", {}).get("model_unseen_equals_parent_heldout")
            if patient.dataset == "SEEG" else None
        ),
    }

    prefix_lengths = (PRIMARY_PREFIX_LEN, SENSITIVITY_PREFIX_LEN)
    census_rows = []
    for prefix_len in prefix_lengths:
        samples = build_sample_set(patient, prefix_len)
        target = RESULT_ROOT / "sample_cache" / f"prefix{prefix_len}" / f"{patient.patient}.npz"
        digest = save_sample_set(samples, target)
        fractions = {}
        for fraction in DATA_FRACTIONS:
            mask = samples.fraction_mask(fraction)
            fractions[str(fraction)] = int(mask.sum())
        nested_ok = bool(
            np.all(samples.subset_25[samples.split == 0] <= samples.subset_50[samples.split == 0])
        )
        out["prefix"][str(prefix_len)] = {
            "n_samples": samples.n_samples,
            "max_cardinality": samples.max_cardinality,
            "sha256": digest,
            "train_fraction_counts": fractions,
            "nested_subsets": nested_ok,
            "event_ids_sha256": sha256_array(samples.event_index),
        }
        census_rows.extend(horizon_census(samples))
    out["horizon_census"] = census_rows
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--datasets", default="SEEG,ECOG")
    arguments = parser.parse_args()

    wanted = set(arguments.datasets.split(","))
    jobs: list[tuple[str, str]] = []
    if "SEEG" in wanted:
        jobs += [("SEEG", path.name) for path in sorted(FRAME_ROOT.iterdir()) if path.is_dir()]
    if "ECOG" in wanted:
        jobs += [("ECOG", name) for name in ("958", "1084")]

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        results = list(pool.map(process_patient, jobs))

    pd.DataFrame([row["input_census"] for row in results]).to_csv(
        RESULT_ROOT / "INPUT_CENSUS.csv", index=False
    )
    pd.DataFrame([row["geometry"] for row in results]).to_csv(
        RESULT_ROOT / "CONTACT_GEOMETRY_AUDIT.csv", index=False
    )
    pd.DataFrame([row["geometry"] for row in results]).to_csv(
        RESULT_ROOT / "PER_PATIENT_COVERAGE_DESCRIPTORS.csv", index=False
    )
    census = pd.DataFrame([entry for row in results for entry in row["horizon_census"]])
    (RESULT_ROOT / "basis").mkdir(parents=True, exist_ok=True)
    census.to_csv(RESULT_ROOT / "basis" / "HORIZON_DENOMINATOR_CENSUS.csv", index=False)

    split_audit = {
        "contract": "topic5_capacity_constrained_history_motif_v0_2_split_audit",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "availability_contract": AVAILABILITY_CONTRACT,
        "suffix_mask_contract": SUFFIX_MASK_CONTRACT,
        "horizons": list(HORIZONS),
        "prefix_lengths": [PRIMARY_PREFIX_LEN, SENSITIVITY_PREFIX_LEN],
        "seeg_split_parity_all_pass": all(
            row["split_audit"]["parent_split_matches"] for row in results
            if row["dataset"] == "SEEG"
        ),
        "seeg_model_unseen_equals_parent_heldout": all(
            row["split_audit"]["model_unseen_equals_parent_heldout"] for row in results
            if row["dataset"] == "SEEG"
        ),
        "nested_subsets_all_pass": all(
            entry["nested_subsets"] for row in results for entry in row["prefix"].values()
        ),
        "per_patient": {row["patient"]: {**row["split_audit"], "prefix": row["prefix"]} for row in results},
    }
    (RESULT_ROOT / "SPLIT_HASH_AUDIT.json").write_text(json.dumps(split_audit, indent=2) + "\n")

    print(f"patients processed: {len(results)}")
    print(f"  split parity all pass: {split_audit['seeg_split_parity_all_pass']}")
    print(f"  nested subsets pass  : {split_audit['nested_subsets_all_pass']}")
    for row in results:
        primary = row["prefix"][str(PRIMARY_PREFIX_LEN)]
        print(
            f"  {row['patient']:22s} samples(prefix3)={primary['n_samples']:7d} "
            f"kmax={primary['max_cardinality']:2d} "
            f"train25/50/100={primary['train_fraction_counts']['25']}/"
            f"{primary['train_fraction_counts']['50']}/{primary['train_fraction_counts']['100']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
