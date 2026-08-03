#!/usr/bin/env python3
"""Freeze v0.4 input provenance and reproduce the 1-45 Hz static A/B anchor."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_static_ab import load_frozen_static_scaffold  # noqa: E402


DEVELOPMENT_SUBJECT = "epilepsiae_1146"
TIE_TOL = 1e-9


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _maxab(field_a: np.ndarray, field_b: np.ndarray, target: np.ndarray) -> float:
    values = []
    for candidate in (field_a, field_b):
        if len(candidate) < 3 or np.nanstd(candidate) <= 0 or np.nanstd(target) <= 0:
            values.append(np.nan)
            continue
        rho = spearmanr(candidate, target).statistic
        values.append(abs(float(rho)) if np.isfinite(rho) else np.nan)
    return float(np.nanmax(values))


def _target_metadata(root: Path, subject: str) -> dict:
    path = root / f"{subject}.json"
    payload = json.loads(path.read_text())
    return {"path": str(path), "sha256": _sha256(path), "payload": payload}


def _load_subject_rows(
    subject: str,
    inventory: pd.DataFrame,
    *,
    artifact_root: Path,
    g0_root: Path,
    target45_root: Path,
    target150_root: Path,
) -> list[dict]:
    dataset_path = (
        artifact_root
        / "results/topic5_interictal_rank_distribution/dataset_v0_4/per_subject"
        / f"{subject}.npz"
    )
    with np.load(dataset_path, allow_pickle=False) as data:
        contact_names = np.asarray(data["contact_names"]).astype(str)
    timeline_path = g0_root / "timeline" / f"{subject}.npz"
    with np.load(timeline_path, allow_pickle=False) as data:
        target_contact_index = np.asarray(data["target_contact_index"], np.int64)
    scaffold = load_frozen_static_scaffold(artifact_root, subject, contact_names)
    valid = np.asarray(scaffold.pop("scaffold_valid"), bool)
    field_a = np.asarray(scaffold["scaffold_field_a"], float)[valid]
    field_b = np.asarray(scaffold["scaffold_field_b"], float)[valid]
    joined_names = contact_names[valid]
    target45_path = target45_root / f"{subject}.npz"
    target150_path = target150_root / f"{subject}.npz"
    rows = []
    with (
        np.load(target45_path, allow_pickle=False) as target45,
        np.load(target150_path, allow_pickle=False) as target150,
    ):
        for seizure in inventory.itertuples(index=False):
            index = int(seizure.seizure_idx)
            energy45 = np.asarray(target45[f"bb_auc__{index}"], float).squeeze()[
                target_contact_index
            ][valid]
            energy150 = np.asarray(target150[f"bb150_auc__{index}"], float).squeeze()[
                target_contact_index
            ][valid]
            rows.append(
                {
                    "subject": subject,
                    "seizure_id": str(seizure.seizure_id),
                    "seizure_idx": index,
                    "n_contacts": int(valid.sum()),
                    "contact_names": joined_names.tolist(),
                    "field_a": field_a,
                    "field_b": field_b,
                    "target45": energy45,
                    "target150": energy150,
                    "maxab_1_45": _maxab(field_a, field_b, energy45),
                    "maxab_1_150": _maxab(field_a, field_b, energy150),
                    "target45_rank": rankdata(energy45, method="average").tolist(),
                    "dataset_path": str(dataset_path),
                    "timeline_path": str(timeline_path),
                    "target45_path": str(target45_path),
                    "target150_path": str(target150_path),
                }
            )
    return rows


def _patient_null(rows: list[dict], draws: int) -> tuple[dict, pd.DataFrame]:
    subject = str(rows[0]["subject"])
    observed = float(np.median([row["maxab_1_45"] for row in rows]))
    rng = np.random.default_rng(_stable_seed(f"v0.4-static-null:{subject}"))
    seizure_null = []
    for row in rows:
        target_rank = rankdata(np.asarray(row["target45"], float), method="average")
        target_rank -= target_rank.mean()
        rank_a = rankdata(np.asarray(row["field_a"], float), method="average")
        rank_b = rankdata(np.asarray(row["field_b"], float), method="average")
        rank_a -= rank_a.mean()
        rank_b -= rank_b.mean()
        target_norm = float(np.linalg.norm(target_rank))
        norm_a = float(np.linalg.norm(rank_a))
        norm_b = float(np.linalg.norm(rank_b))
        permutations = np.vstack(
            [rng.permutation(len(target_rank)) for _ in range(int(draws))]
        )
        permuted = target_rank[permutations]
        rho_a = np.abs(permuted @ rank_a / max(target_norm * norm_a, 1e-12))
        rho_b = np.abs(permuted @ rank_b / max(target_norm * norm_b, 1e-12))
        seizure_null.append(np.maximum(rho_a, rho_b))
    null = np.median(np.row_stack(seizure_null), axis=0)
    summary = {
        "subject": subject,
        "n_seizures": len(rows),
        "n_contacts": int(rows[0]["n_contacts"]),
        "observed_patient_median_maxab_1_45": observed,
        "channel_null_median": float(np.median(null)),
        "channel_null_p95": float(np.percentile(null, 95)),
        "observed_minus_null_median": float(observed - np.median(null)),
        "permutation_p_one_sided": float((1 + np.sum(null >= observed)) / (len(null) + 1)),
        "pass_null_p95": bool(observed > np.percentile(null, 95)),
        "n_perm": int(draws),
    }
    frame = pd.DataFrame({"subject": subject, "draw": np.arange(draws), "null_maxab": null})
    return summary, frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("/home/honglab/leijiaxin/HFOsp"))
    parser.add_argument(
        "--g0-root",
        type=Path,
        default=ROOT / "results/topic5_history_rnn_early_ictal_field/g0_causal_prefix",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/topic5_history_conditioned_field_refinement_v0_4",
    )
    parser.add_argument("--n-perm", type=int, default=5000)
    args = parser.parse_args()
    artifact = args.artifact_root.resolve()
    g0 = args.g0_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    target45 = artifact / "results/topic5_ictal_recruitment/t0_feature_cache"
    target150 = artifact / "results/topic5_ictal_recruitment/t0_feature_cache_bb150_1_150"
    inventory_path = g0 / "seizure_causal_history_inventory.csv"
    inventory = pd.read_csv(inventory_path, dtype={"subject": str, "seizure_id": str})
    inventory = inventory.loc[inventory.g2_metadata_eligible.astype(bool)].copy()
    primary = sorted(set(inventory.subject.astype(str)) - {DEVELOPMENT_SUBJECT})
    if len(primary) != 15:
        raise RuntimeError(f"primary cohort drift: expected 15, got {len(primary)}")
    meta45 = _target_metadata(target45, primary[0])
    meta150 = _target_metadata(target150, primary[0])
    if list(map(float, meta45["payload"].get("band_broad", []))) != [1.0, 45.0]:
        raise RuntimeError("primary target sidecar is not 1-45 Hz")
    if list(map(float, meta150["payload"].get("band_broad_1_150", []))) != [1.0, 150.0]:
        raise RuntimeError("sensitivity target sidecar is not 1-150 Hz")

    all_rows: list[dict] = []
    subject_summaries = []
    null_frames = []
    source_files: dict[str, dict] = {}
    for subject in primary:
        subject_inventory = inventory.loc[inventory.subject == subject].copy()
        rows = _load_subject_rows(
            subject,
            subject_inventory,
            artifact_root=artifact,
            g0_root=g0,
            target45_root=target45,
            target150_root=target150,
        )
        all_rows.extend(rows)
        summary, null = _patient_null(rows, args.n_perm)
        subject_summaries.append(summary)
        null_frames.append(null)
        for label, path in {
            "dataset": Path(rows[0]["dataset_path"]),
            "timeline": Path(rows[0]["timeline_path"]),
            "target45": Path(rows[0]["target45_path"]),
            "target45_sidecar": target45 / f"{subject}.json",
            "target150": Path(rows[0]["target150_path"]),
            "target150_sidecar": target150 / f"{subject}.json",
            "static_ab": artifact / "results/interictal_propagation_masked/template_gradient_fields/per_subject" / f"{subject}.json",
        }.items():
            source_files[f"{subject}:{label}"] = {"path": str(path), "sha256": _sha256(path)}

    seizure_table = pd.DataFrame(
        [
            {key: value for key, value in row.items() if key not in {"field_a", "field_b", "target45", "target150", "target45_rank"}}
            for row in all_rows
        ]
    )
    patient_table = pd.DataFrame(subject_summaries)
    null_table = pd.concat(null_frames, ignore_index=True)
    seizure_table.to_csv(output / "static_anchor_seizure_metrics.csv", index=False)
    patient_table.to_csv(output / "static_anchor_patient_metrics.csv", index=False)
    null_table.to_csv(output / "static_anchor_channel_null_draws.csv.gz", index=False, compression="gzip")
    manifest = {
        "status": "INPUT_CONTRACT_CONFIRMED",
        "contract": "topic5_history_conditioned_field_refinement_v0_4",
        "primary_endpoint": {
            "alignment": "clinical_onset",
            "window_seconds": [0.0, 10.0],
            "band_hz": [1.0, 45.0],
            "npz_key": "bb_auc__<seizure_idx>",
            "source_root": str(target45),
        },
        "sensitivity_endpoint": {
            "band_hz": [1.0, 150.0],
            "npz_key": "bb150_auc__<seizure_idx>",
            "source_root": str(target150),
            "training": "NO_RETRAIN",
        },
        "cohort": {
            "development_engineering_only": DEVELOPMENT_SUBJECT,
            "primary_subjects": primary,
            "n_primary_subjects": len(primary),
            "n_primary_seizures": int(len(seizure_table)),
            "contact_denominator_min": int(patient_table.n_contacts.min()),
            "contact_denominator_median": float(patient_table.n_contacts.median()),
            "contact_denominator_max": int(patient_table.n_contacts.max()),
        },
        "static_ab_boundary": {
            "target_blind": True,
            "retrospective_full_record": True,
            "may_include_post_target_interictal_events": True,
            "prospective_claim_allowed": False,
        },
        "static_anchor_1_45": {
            "patient_median_maxab": float(patient_table.observed_patient_median_maxab_1_45.median()),
            "patient_median_margin_vs_channel_null": float(patient_table.observed_minus_null_median.median()),
            "n_positive_margin": int(np.sum(patient_table.observed_minus_null_median > TIE_TOL)),
            "n_tied_margin": int(np.sum(np.abs(patient_table.observed_minus_null_median) <= TIE_TOL)),
            "n_pass_individual_p95": int(patient_table.pass_null_p95.sum()),
        },
        "source_files": source_files,
        "inventory": {"path": str(inventory_path), "sha256": _sha256(inventory_path)},
    }
    (output / "INPUT_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
