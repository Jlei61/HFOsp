#!/usr/bin/env python3
"""Unlock and score frozen interictal fields against early-ictal energy."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_patient_specific_rnn_bridge import chronological_60_20_20  # noqa: E402
from scripts.train_topic5_interictal_rank_distribution import load_records  # noqa: E402
from src.topic5_shared_scaffold_field_readout import (  # noqa: E402
    contact_label_permutations,
    paired_model_patient_statistics,
    score_frozen_field_against_ictal,
    seizure_first_patient_first_summary,
    validate_frozen_field_manifest,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _centered_rank(values: Sequence[float]) -> np.ndarray:
    ranked = rankdata(np.asarray(values, dtype=float), method="average")
    return ranked - np.mean(ranked)


def score_single_field(
    *,
    field: Sequence[float],
    target: Sequence[float],
    contact_names: Sequence[str],
    n_draws: int,
    all_seed: int,
    shaft_seed: int,
) -> dict[str, Any]:
    """Absolute Spearman and matched nulls for the unique static field."""

    field = np.asarray(field, dtype=float)
    target = np.asarray(target, dtype=float)
    names = np.asarray(contact_names).astype(str)
    finite = np.isfinite(field) & np.isfinite(target)
    field, target, names = field[finite], target[finite], names[finite]
    field_rank, target_rank = _centered_rank(field), _centered_rank(target)
    denominator = float(np.linalg.norm(field_rank) * np.linalg.norm(target_rank))
    if len(field) < 6 or denominator <= 1.0e-12:
        raise ValueError("static field score needs at least six nonconstant contacts")
    observed = float(abs(target_rank @ field_rank / denominator))
    result: dict[str, Any] = {
        "n_contacts": int(len(field)),
        "observed_max_abs_rho": observed,
        "minus_signed_rho": float(target_rank @ field_rank / denominator),
        "plus_signed_rho": np.nan,
        "selected_direction": "single_static_field",
        "n_directions": 1,
        "n_null_draws": int(n_draws),
    }
    for mode, seed in (("all_contact", all_seed), ("within_shaft", shaft_seed)):
        permutation = contact_label_permutations(
            names, n_draws=n_draws, seed=seed, mode=mode
        )
        null = np.abs(target_rank[permutation] @ field_rank / denominator)
        median = float(np.median(null))
        result[f"{mode}_null"] = null
        result[f"{mode}_null_median"] = median
        result[f"{mode}_null_p95"] = float(np.percentile(null, 95))
        result[f"{mode}_margin"] = observed - median
        result[f"{mode}_empirical_p"] = float(
            (1 + np.count_nonzero(null >= observed - 1.0e-15)) / (len(null) + 1)
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--readout-config",
        type=Path,
        default=ROOT / "config/topic5_shared_scaffold_ictal_readout_v0_2.yaml",
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=ROOT / "config/topic5_shared_scaffold_propagation_rnn_v0_2.yaml",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--source-pool-rule",
        choices=("learned_axis", "normalized_laplacian"),
        default="learned_axis",
    )
    args = parser.parse_args()
    freeze_dir = (
        "field_freeze"
        if args.source_pool_rule == "learned_axis"
        else "field_freeze_diffusion_graph_sensitivity"
    )
    readout = yaml.safe_load(args.readout_config.resolve().read_text())
    training = yaml.safe_load(args.training_config.resolve().read_text())
    output = (
        args.output_root.resolve()
        if args.output_root
        else ROOT / readout["output_root"]
    )
    manifest_path = output / freeze_dir / "FROZEN_FIELD_MANIFEST.json"
    target_seal_path = output / "target_audit" / "TARGET_SEAL.json"
    manifest = json.loads(manifest_path.read_text())
    target_seal = json.loads(target_seal_path.read_text())
    validate_frozen_field_manifest(manifest)
    if target_seal.get("target_values_read") is not False or target_seal.get("target_values_sealed") is not True:
        raise RuntimeError("target metadata seal is absent")
    expected_subjects = list(map(str, readout["primary_subjects"])) + [
        str(readout["supportive_subject"])
    ]
    record_lookup = {
        (str(record["subject_id"]), str(record["model_name"])): record
        for record in manifest["records"]
    }
    missing = [
        (subject, model)
        for subject in expected_subjects
        for model in ("structured", "ordinary_gru")
        if (subject, model) not in record_lookup
    ]
    if missing:
        raise RuntimeError(f"frozen manifest misses required fields: {missing}")

    early_root = output / (
        "early_ictal"
        if args.source_pool_rule == "learned_axis"
        else "early_ictal_diffusion_graph_sensitivity"
    )
    unlock = {
        "contract": readout["contract"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_values_read_before_this_record": False,
        "target_values_unlocked_after_field_freeze": True,
        "field_manifest_sha256": str(manifest["manifest_sha256"]),
        "field_manifest_file_sha256": sha256_file(manifest_path),
        "target_seal_file_sha256": sha256_file(target_seal_path),
        "scoring_code_sha256": sha256_file(Path(__file__).resolve()),
        "target_key": str(readout["target_key"]),
    }
    atomic_json(early_root / "TARGET_UNLOCK_RECORD.json", unlock)

    dataset_root = Path(training["dataset_artifact_root"]).resolve() / training["dataset_root"]
    records = load_records(dataset_root)
    target_root = Path(readout["target_cache_root"]).resolve()
    n_draws = int(readout["all_contact_permutations"])
    seizure_scores: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for subject in expected_subjects:
        target_files = sorted((target_root / f"outer_{subject}").glob(f"{subject}__*.npz"))
        record = records[subject]
        fit60, _, _ = chronological_60_20_20(record)
        static_field = np.mean(np.asarray(record.group_ids)[fit60] >= 0, axis=0)
        for seizure_index, path in enumerate(target_files):
            with np.load(path, allow_pickle=False) as data:
                target_names = np.asarray(data["contact_names"]).astype(str)
                target_values = np.asarray(data[str(readout["target_key"])], dtype=float)
            seizure_id = path.stem.split("__", 1)[-1]
            for model in ("structured", "ordinary_gru"):
                all_seed = stable_seed(subject, seizure_id, model, "all_contact")
                shaft_seed = stable_seed(subject, seizure_id, model, "within_shaft")
                result = score_frozen_field_against_ictal(
                    record_lookup[(subject, model)],
                    seizure_id=seizure_id,
                    target_contact_names=target_names,
                    target_values=target_values,
                    n_draws=n_draws,
                    all_contact_seed=all_seed,
                    within_shaft_seed=shaft_seed,
                    min_contacts=int(readout["minimum_exact_joined_contacts"]),
                )
                result["n_directions"] = 2
                seizure_scores.append(result)
            model_lookup = {name: index for index, name in enumerate(record.contact_names)}
            target_lookup = {name: index for index, name in enumerate(target_names)}
            joined_names = [name for name in record.contact_names if name in target_lookup]
            static_result = score_single_field(
                field=[static_field[model_lookup[name]] for name in joined_names],
                target=[target_values[target_lookup[name]] for name in joined_names],
                contact_names=joined_names,
                n_draws=n_draws,
                all_seed=stable_seed(subject, seizure_id, "static", "all_contact"),
                shaft_seed=stable_seed(subject, seizure_id, "static", "within_shaft"),
            )
            static_result.update(
                subject=subject,
                model="static",
                seizure_id=seizure_id,
                field_fingerprint_sha256="fit60_participation_probability",
                matched_contact_names=joined_names,
            )
            seizure_scores.append(static_result)

    for row in seizure_scores:
        csv_rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"all_contact_null", "within_shaft_null", "matched_contact_names"}
            }
        )
    seizure_frame = pd.DataFrame(csv_rows).sort_values(["subject", "model", "seizure_id"])
    seizure_frame.to_csv(early_root / "seizure_scores.csv", index=False)
    summary = seizure_first_patient_first_summary(
        seizure_scores,
        supportive_subject=str(readout["supportive_subject"]),
        n_boot=5000,
        bootstrap_seed=771_003,
    )
    patient_rows = summary["patients"]
    patient_csv = [
        {key: value for key, value in row.items() if key not in {"all_contact_null", "within_shaft_null"}}
        for row in patient_rows
    ]
    pd.DataFrame(patient_csv).sort_values(["subject", "model"]).to_csv(
        early_root / "patient_scores.csv", index=False
    )
    comparisons = {
        "structured_vs_ordinary_all_contact": paired_model_patient_statistics(
            patient_rows, model_a="structured", model_b="ordinary_gru", null_mode="all_contact",
            supportive_subject=str(readout["supportive_subject"]), bootstrap_seed=811_001,
        ),
        "structured_vs_static_all_contact": paired_model_patient_statistics(
            patient_rows, model_a="structured", model_b="static", null_mode="all_contact",
            supportive_subject=str(readout["supportive_subject"]), bootstrap_seed=811_002,
        ),
        "structured_vs_ordinary_within_shaft": paired_model_patient_statistics(
            patient_rows, model_a="structured", model_b="ordinary_gru", null_mode="within_shaft",
            supportive_subject=str(readout["supportive_subject"]), bootstrap_seed=811_003,
        ),
    }
    cohort = {
        "contract": readout["contract"],
        "field_manifest_sha256": str(manifest["manifest_sha256"]),
        "target_values_read": True,
        "target_unlock_record_sha256": sha256_file(early_root / "TARGET_UNLOCK_RECORD.json"),
        "n_primary_subjects": len(readout["primary_subjects"]),
        "supportive_subject": str(readout["supportive_subject"]),
        "model_statistics": summary["cohort"],
        "paired_comparisons": comparisons,
    }
    atomic_json(early_root / "cohort_statistics.json", cohort)
    atomic_json(
        early_root / "permutation_manifest.json",
        {
            "n_draws": n_draws,
            "modes": ["all_contact", "within_shaft"],
            "absolute_and_two_direction_max_recomputed_inside_each_draw": True,
            "static_uses_one_direction_only": True,
            "seed_rule": "sha256(subject|seizure|model|null_mode)",
        },
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "n_seizures": int(seizure_frame[["subject", "seizure_id"]].drop_duplicates().shape[0]),
                "n_primary_subjects": len(readout["primary_subjects"]),
                "target_values_read": True,
            }
        )
    )


if __name__ == "__main__":
    main()
