#!/usr/bin/env python3
"""Freeze Stage-F artifacts and synchronized null maps before target access."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
FIELD_ROOT = Path(
    "/home/honglab/leijiaxin/HFOsp/results/"
    "interictal_propagation_masked/template_gradient_fields/per_subject"
)
FROZEN_FILES = (
    "RUN_CONTRACT.json", "INPUT_CACHE_MANIFEST.json", "FORMAL_TRAINING_SCHEDULE.csv",
    "STAGE_E_TRAINING_COMPLETE.json", "INTERICTAL_V0_5_SUMMARY.json",
    "PREFIX_TEMPLATE_SUMMARY.json", "STAGE_F_RUN_SNAPSHOT.json",
    "STAGE_F_TARGET_FREE_COMPLETE.json", "EARLY_ICTAL_ROUTING_METADATA.csv",
    "CROSSFIT_NONLOCALITY_PATIENT_SUMMARY.csv", "J_ESTIMAND_PREFREEZE_REPAIR.json",
    "MODEL_FIELD_MANIFEST.csv", "TEMPLATE_FIELD_MANIFEST.csv", "MODEL_FIELDS_FROZEN.json",
    "ATTENUATED_FIELD_MANIFEST.csv", "ATTENUATED_FIELDS_FROZEN.json",
    "GAIN_ADJUSTED_FIELD_MANIFEST.csv", "GAIN_ADJUSTED_SENSITIVITY_COMPLETE.json",
    "mechanism/MECHANISM_PER_PATIENT.csv", "mechanism/MODE_FLOW_BUNDLE_MANIFEST.csv",
    "mechanism/MODE_FLOW_ATTENUATION_PER_PATIENT.csv",
    "mechanism/MODE_FLOW_ATTENUATION_SUMMARY.json", "MODE_FLOW_ATTENUATION_COMPLETE.json",
    "MECHANISM_ANALYSIS_COMPLETE.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def permutations(n: int, shafts: list[str], draws: int, seed: int,
                 within_shaft: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.arange(n, dtype=np.int16)
    output = np.empty((draws, n), dtype=np.int16)
    groups = ([np.flatnonzero(np.asarray(shafts) == shaft) for shaft in sorted(set(shafts))]
              if within_shaft else [np.arange(n)])
    for draw in range(draws):
        row = base.copy()
        for group in groups:
            row[group] = rng.permutation(group)
        output[draw] = row
    return output


def grouped_permutations(n: int, groups: list[np.ndarray], draws: int, seed: int) -> np.ndarray:
    """Freeze label permutations inside prespecified spatial groups only."""
    rng = np.random.default_rng(seed)
    base = np.arange(n, dtype=np.int16)
    output = np.tile(base, (draws, 1))
    for draw in range(draws):
        for group in groups:
            group = np.asarray(group, dtype=int)
            if len(group) > 1:
                output[draw, group] = rng.permutation(group)
    return output


def distance_bin_groups(xy: np.ndarray) -> tuple[list[np.ndarray], dict]:
    """Equal-count montage-eccentricity bins for a coarse spatial null."""
    xy = np.asarray(xy, float)
    n = len(xy)
    geometry_rank = int(np.linalg.matrix_rank(xy - xy.mean(0))) if n else 0
    if n < 9 or geometry_rank < 2:
        return [], {"eligible": False, "reason": "REQUIRES_2D_AND_AT_LEAST_9_CONTACTS"}
    distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    eccentricity = distance.sum(1) / max(n - 1, 1)
    groups = [np.asarray(group, dtype=int) for group in np.array_split(
        np.argsort(eccentricity, kind="stable"), 3
    )]
    if min(map(len, groups)) < 3:
        return [], {"eligible": False, "reason": "DISTANCE_BIN_TOO_SMALL"}
    return groups, {
        "eligible": True,
        "contract": "THREE_EQUAL_COUNT_BINS_BY_MEAN_PAIRWISE_DISTANCE_TO_MONTAGE",
        "group_sizes": list(map(len, groups)),
        "eccentricity_mm": eccentricity.tolist(),
    }


def laplacian_basis(xy: np.ndarray) -> tuple[np.ndarray, dict]:
    """Contact-graph Laplacian eigenbasis frozen without target values."""
    xy = np.asarray(xy, float)
    n = len(xy)
    geometry_rank = int(np.linalg.matrix_rank(xy - xy.mean(0))) if n else 0
    if n < 8 or geometry_rank < 2:
        return np.empty((0, 0)), {
            "eligible": False, "reason": "REQUIRES_2D_AND_AT_LEAST_8_CONTACTS",
        }
    distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    k = min(3, n - 1)
    adjacency = np.zeros((n, n), float)
    for source in range(n):
        neighbors = np.argsort(distance[source], kind="stable")[1:k + 1]
        adjacency[source, neighbors] = 1.0
    adjacency = np.maximum(adjacency, adjacency.T)
    active_lengths = distance[adjacency > 0]
    scale = max(float(np.median(active_lengths)), 1e-6)
    adjacency *= np.exp(-distance / scale)
    laplacian = np.diag(adjacency.sum(1)) - adjacency
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    connected = bool(eigenvalues[1] > 1e-10)
    return (eigenvectors if connected else np.empty((0, 0))), {
        "eligible": connected,
        "reason": "PASS" if connected else "CONTACT_GRAPH_DISCONNECTED",
        "contract": "SYMMETRIZED_3NN_EXPONENTIAL_WEIGHT_LAPLACIAN",
        "eigenvalues": eigenvalues.tolist(),
        "knn": k,
        "distance_scale_mm": scale,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-perm", type=int, default=5000)
    args = parser.parse_args()
    if args.n_perm < 5000:
        raise ValueError("primary synchronized null requires at least 5000 draws")
    out = args.out_root.resolve()
    for marker in ("STAGE_E_INTERICTAL_ANALYSIS_COMPLETE.json", "STAGE_F_TARGET_FREE_COMPLETE.json",
                   "MODEL_FIELDS_FROZEN.json",
                   "ATTENUATED_FIELDS_FROZEN.json", "MECHANISM_ANALYSIS_COMPLETE.json"):
        if not (out / marker).exists():
            raise RuntimeError(f"cannot authorize target before {marker}")
    for marker in ("GAIN_ADJUSTED_SENSITIVITY_COMPLETE.json",
                   "MODE_FLOW_ATTENUATION_COMPLETE.json"):
        if not (out / marker).exists():
            raise RuntimeError(f"cannot authorize target before {marker}")
    hashes = {}
    for relative in FROZEN_FILES:
        path = out / relative
        if not path.exists():
            raise FileNotFoundError(path)
        hashes[relative] = sha256_file(path)
    routing = pd.read_csv(out / "EARLY_ICTAL_ROUTING_METADATA.csv")
    if routing.subject.nunique() != 17 or len(routing) != 167:
        raise RuntimeError("early-ictal routing denominator must remain 17 patients / 167 seizures")
    census = pd.read_csv(out / "FULL_PARENT_FIT_CENSUS.csv")
    plane_fit_by_subject = census.groupby("subject", sort=False).first().fit_id.to_dict()
    null_rows = []
    null_root = out / "null_maps"; null_root.mkdir(exist_ok=True)
    for event in routing.itertuples():
        intact = out / "model_fields/intact/per_patient" / event.subject / "L3_LOCAL_PLUS_LEARNED_LR.npz"
        with np.load(intact, allow_pickle=False) as data:
            contacts = data["contacts"].astype(str).tolist()
        empirical = json.loads((FIELD_ROOT / f"{event.subject}.json").read_text())["interictal_field"]
        order = [str(value) for value in empirical["contact_order"]]
        shaft_lookup = dict(zip(order, map(str, empirical["shafts"])))
        shafts = [shaft_lookup[name] for name in contacts]
        seed = int(event.permutation_seed)
        shaft_counts = {shaft: shafts.count(shaft) for shaft in sorted(set(shafts))}
        shaft_groups = [
            np.flatnonzero(np.asarray(shafts) == shaft)
            for shaft in sorted(set(shafts)) if shaft_counts[shaft] >= 4
        ]
        within_shaft_eligible = bool(shaft_groups)
        within_shaft = (
            grouped_permutations(len(contacts), shaft_groups, args.n_perm, seed + 104729)
            if within_shaft_eligible else np.empty((0, len(contacts)), dtype=np.int16)
        )
        plane_fit = str(plane_fit_by_subject[event.subject])
        with np.load(out / "cache" / plane_fit / "plane.npz", allow_pickle=False) as plane:
            xy = np.asarray(plane["contacts_xy_mm"], float)
        provenance = json.loads((out / "cache" / plane_fit / "provenance.json").read_text())
        if list(map(str, provenance["joint_contacts"])) != contacts:
            raise RuntimeError(f"null geometry/contact mismatch: {event.subject}")
        distance_groups, distance_meta = distance_bin_groups(xy)
        distance_bin = (
            grouped_permutations(len(contacts), distance_groups, args.n_perm, seed + 209759)
            if distance_meta["eligible"] else np.empty((0, len(contacts)), dtype=np.int16)
        )
        eigenvectors, spectral_meta = laplacian_basis(xy)
        spectral_rng = np.random.default_rng(seed + 314159)
        spectral_signs = (
            spectral_rng.choice(np.asarray([-1, 1], dtype=np.int8),
                                size=(args.n_perm, len(contacts)))
            if spectral_meta["eligible"] else np.empty((0, len(contacts)), dtype=np.int8)
        )
        if len(spectral_signs):
            spectral_signs[:, 0] = 1
        variogram_eligible = bool(
            len(contacts) >= 12 and int(np.linalg.matrix_rank(xy - xy.mean(0))) >= 2
        )
        variogram_normals = (
            np.random.default_rng(seed + 433494437).standard_normal(
                (args.n_perm, len(contacts)), dtype=np.float32
            ) if variogram_eligible else np.empty((0, len(contacts)), dtype=np.float32)
        )
        destination = null_root / f"{event.subject}__seizure{int(event.seizure_idx)}.npz"
        np.savez_compressed(
            destination, contacts=np.asarray(contacts, dtype="U64"),
            all_contact=permutations(len(contacts), shafts, args.n_perm, seed, False),
            within_shaft=within_shaft,
            within_shaft_eligible=np.asarray(within_shaft_eligible),
            distance_bin=distance_bin,
            distance_bin_eligible=np.asarray(bool(distance_meta["eligible"])),
            contact_xy_mm=xy.astype(np.float32),
            spectral_eigenvectors=eigenvectors.astype(np.float32),
            spectral_signs=spectral_signs,
            spectral_eligible=np.asarray(bool(spectral_meta["eligible"])),
            variogram_normals=variogram_normals,
            variogram_eligible=np.asarray(variogram_eligible),
            permutation_seed=np.asarray(seed), n_permutations=np.asarray(args.n_perm),
        )
        null_rows.append({
            "subject": event.subject, "seizure_idx": int(event.seizure_idx),
            "path": str(destination), "sha256": sha256_file(destination),
            "n_contacts": len(contacts), "n_permutations": args.n_perm,
            "within_shaft_eligible": within_shaft_eligible,
            "within_shaft_contract": "PURE_WITHIN_SHAFT_MIN_GROUP_4_NO_FALLBACK",
            "within_shaft_permutable_groups": len(shaft_groups),
            "within_shaft_permutable_contacts": int(sum(map(len, shaft_groups))),
            "minimum_shaft_contacts": min(shaft_counts.values()) if shaft_counts else 0,
            "distance_bin_eligible": bool(distance_meta["eligible"]),
            "distance_bin_contract": distance_meta.get("contract", distance_meta.get("reason")),
            "spectral_eligible": bool(spectral_meta["eligible"]),
            "spectral_contract": spectral_meta.get("contract", spectral_meta.get("reason")),
            "variogram_eligible": variogram_eligible,
            "variogram_contract": "TARGET_RANK_VARIOGRAM_RANGE_FIT_WITH_FROZEN_GAUSSIAN_DRAWS",
            "target_values_read": False,
        })
    null_manifest = out / "NULL_INDEX_MAP_MANIFEST.csv"
    pd.DataFrame(null_rows).to_csv(null_manifest, index=False)
    scorer = ROOT / "scripts/score_topic5_multiscale_early_ictal_v0_5.py"
    if not scorer.exists():
        raise FileNotFoundError(scorer)
    authorization = {
        "contract": "topic5_multiscale_target_unseal_v0_5",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authorized": True, "target_values_read": False,
        "status": "LOCKED_INTERNAL_MECHANISTIC_FOLLOWUP_NOT_INDEPENDENT_CONFIRMATION",
        "target": "clinical onset 0-10 s, 1-150 Hz broadband energy",
        "patients": 17, "seizures": 167,
        "frozen_hashes": hashes,
        "null_manifest_sha256": sha256_file(null_manifest),
        "scorer_sha256": sha256_file(scorer),
        "primary_endpoint": "signed_best_mode_spearman_oracle_repertoire_coverage",
        "primary_null": (
            "joint_patient_label_permutation_and_synchronized_all_contact_"
            "spatial_null_interaction;_both_must_pass"
        ),
        "primary_interaction": "spearman(J, raw_signed_oracle_L3_minus_L2m)>0",
        "robustness_nulls": {
            "shaft_preserving": "pure within-shaft permutation for shaft groups with >=4 contacts",
            "distance_bin": "permutation within three equal-count montage-eccentricity bins",
            "graph_laplacian_spectral": "frozen random sign rotations in a symmetrized-3NN Laplacian basis",
            "variogram_matched": "target-rank range fit with frozen Gaussian innovations and marginal rank restoration",
        },
        "model_or_field_generation_after_unseal_forbidden": True,
    }
    (out / "TARGET_UNSEAL_AUTHORIZATION.json").write_text(json.dumps(authorization, indent=2) + "\n")
    print(json.dumps(authorization, indent=2))


if __name__ == "__main__":
    main()
