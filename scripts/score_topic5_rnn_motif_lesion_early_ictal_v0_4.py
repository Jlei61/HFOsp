#!/usr/bin/env python3
"""Secondary early-ictal readout of target-free intact and lesioned RNN fields."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from score_topic5_rnn_motif_early_ictal_v0_4 import (  # noqa: E402
    build_scorer,
    locked_target_artifacts,
    paired_summary,
    permutation_indices,
    score_one,
    stable_seed,
)


def read_records(out_root: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted(
        (out_root / "matched_lesions").glob("**/LESION_DONE.json")
    )]


def patient_fields(records: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Resolve shared-fit or own-a/own-b fields without averaging candidates before maxAB."""
    grouped: dict[tuple[str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for record in records:
        for lesion, payload in record["lesions"].items():
            if payload.get("status") == "motif_not_estimable" or "targeted_fields" not in payload:
                continue
            grouped.setdefault((record["subject"], record["model"], lesion), []).append((record, payload))
    output: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, values in grouped.items():
        resolved: dict[str, Any] = {}
        resolved_status: dict[str, dict[str, bool]] = {}
        for condition in ("baseline", "targeted"):
            candidates = {}
            contacts = {}
            producers = {}
            matched_available = {}
            for record, payload in values:
                for template, field in payload[f"{condition}_fields"].items():
                    template = str(template).upper()
                    preferred = record["scope"] == "shared" or record["scope"] == f"own_{template.lower()}"
                    if template not in candidates or preferred:
                        candidates[template] = np.asarray(field, float)
                        contacts[template] = [str(value) for value in payload["field_contacts"]]
                        producers[template] = record["fit_id"]
                        matched_available[template] = payload.get("status") == "inference_available"
            if set(candidates) != {"A", "B"}:
                continue
            if contacts["A"] != contacts["B"]:
                raise RuntimeError(f"lesion A/B contact mismatch: {key} {condition}")
            resolved[condition] = {
                "A": candidates["A"], "B": candidates["B"],
                "contacts": contacts["A"], "producers": producers,
            }
            resolved_status[condition] = matched_available
        if set(resolved) == {"baseline", "targeted"}:
            resolved["matched_inference_available"] = all(
                resolved_status[condition].get(template, False)
                for condition in ("baseline", "targeted") for template in ("A", "B")
            )
            output[key] = resolved
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty lesion readout: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def bounded_secondary_summary(values: list[float], *, seed: int) -> dict[str, Any]:
    """Keep very small cross-state lesion denominators descriptive only."""
    result = paired_summary(values, seed=seed)
    eligible = int(result.get("n", 0)) >= 5
    result["cohort_inference_eligible"] = eligible
    result["minimum_patients_for_inference"] = 5
    if not eligible:
        result["wilcoxon_p"] = None
        result["bootstrap_95ci"] = [None, None]
        result["inference_status"] = "descriptive_only_small_patient_denominator"
    else:
        result["inference_status"] = "eligible_secondary_inference"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--target-cache-root", type=Path, required=True)
    parser.add_argument("--n-perm", type=int, default=5000)
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    target_root = args.target_cache_root.resolve()
    access = json.loads((out_root / "target_access_audit.json").read_text())
    if not access.get("target_values_read"):
        raise RuntimeError("primary target unseal/scoring must finish before lesion readout")
    metadata = json.loads((out_root / "EARLY_ICTAL_METADATA_INVENTORY.json").read_text())
    manifest = json.loads((out_root / "MODEL_FIELD_MANIFEST.json").read_text())
    primary = list(metadata["actual_primary_join"])
    supportive = str(metadata["supportive_subject"])
    subjects = primary + ([supportive] if metadata["supportive_available"] else [])
    target_files_by_subject = locked_target_artifacts(out_root, target_root, metadata)
    fields = patient_fields(read_records(out_root))
    rows: list[dict[str, Any]] = []
    for subject in subjects:
        subject_fields = {key: value for key, value in fields.items() if key[0] == subject}
        if not subject_fields:
            continue
        record = json.loads(Path(manifest["patient_geometry"][subject]["empirical_record"]).read_text())
        field_record = record["interictal_field"]
        order = [str(value) for value in field_record["contact_order"]]
        shafts = [str(value) for value in field_record["shafts"]]
        aligned = {}
        for key, conditions in subject_fields.items():
            aligned[key] = {
                "matched_inference_available": conditions["matched_inference_available"]
            }
            for condition in ("baseline", "targeted"):
                value = conditions[condition]
                lookup_a = dict(zip(value["contacts"], value["A"]))
                lookup_b = dict(zip(value["contacts"], value["B"]))
                aligned[key][condition] = (
                    np.asarray([lookup_a.get(name, np.nan) for name in order], float),
                    np.asarray([lookup_b.get(name, np.nan) for name in order], float),
                )
        common_finite = np.ones(len(order), bool)
        for conditions in aligned.values():
            for condition in ("baseline", "targeted"):
                a, b = conditions[condition]
                common_finite &= np.isfinite(a) & np.isfinite(b)
        for target_path in target_files_by_subject[subject]:
            with np.load(target_path, allow_pickle=False) as data:
                names = np.asarray(data["contact_names"]).astype(str).tolist()
                values = np.asarray(data["target_1_150"], float)
            target_lookup = dict(zip(names, values))
            target = np.asarray([target_lookup.get(name, np.nan) for name in order], float)
            finite = common_finite & np.isfinite(target)
            eligible = np.flatnonzero(finite)
            if len(eligible) < 6:
                continue
            seizure = target_path.stem.split("__", 1)[-1]
            permutations = permutation_indices(
                len(order), eligible, shafts, args.n_perm,
                stable_seed(subject, seizure, "canonical_full", "all_contact"), False,
            )
            for (_, model, lesion), conditions in aligned.items():
                scored = {}
                for condition in ("baseline", "targeted"):
                    a, b = conditions[condition]
                    maxab = score_one(build_scorer(record, a, b, finite), target, permutations)
                    common = 0.5 * (a + b)
                    common_score = score_one(
                        build_scorer(record, common, common, finite), target, permutations
                    )
                    scored[condition] = (maxab, common_score)
                intact, intact_common = scored["baseline"]
                lesioned, lesioned_common = scored["targeted"]
                rows.append({
                    "subject": subject, "primary": subject in primary,
                    "supportive": subject == supportive, "seizure_id": seizure,
                    "model": model, "cell": "rnn", "lesion": lesion,
                    "matched_inference_available": conditions["matched_inference_available"],
                    "n_contacts": int(len(eligible)),
                    "intact_maxab": intact["observed"], "lesioned_maxab": lesioned["observed"],
                    "intact_margin": intact["margin"], "lesioned_margin": lesioned["margin"],
                    "damage_maxab": intact["observed"] - lesioned["observed"],
                    "damage_margin": intact["margin"] - lesioned["margin"],
                    "intact_common": intact_common["observed"],
                    "lesioned_common": lesioned_common["observed"],
                    "damage_common": intact_common["observed"] - lesioned_common["observed"],
                })
    write_csv(out_root / "lesion_early_ictal_per_seizure.csv", rows)
    keys = sorted({(row["subject"], row["model"], row["lesion"]) for row in rows})
    patient_rows = []
    for subject, model, lesion in keys:
        selected = [row for row in rows if (row["subject"], row["model"], row["lesion"])
                    == (subject, model, lesion)]
        patient_rows.append({
            "subject": subject, "primary": subject in primary, "supportive": subject == supportive,
            "model": model, "cell": "rnn", "lesion": lesion, "n_seizures": len(selected),
            "matched_inference_available": all(
                row["matched_inference_available"] for row in selected
            ),
            **{metric: float(np.nanmedian([row[metric] for row in selected]))
               for metric in ("intact_maxab", "lesioned_maxab", "intact_margin", "lesioned_margin",
                              "damage_maxab", "damage_margin", "intact_common", "lesioned_common",
                              "damage_common")},
        })
    write_csv(out_root / "lesion_early_ictal_per_patient.csv", patient_rows)
    statistics = {}
    for model, lesion in sorted({(row["model"], row["lesion"]) for row in patient_rows}):
        selected = [row for row in patient_rows if row["primary"] and row["model"] == model
                    and row["lesion"] == lesion and row["matched_inference_available"]]
        for metric in ("damage_maxab", "damage_margin", "damage_common"):
            statistics[f"{model}|{lesion}|{metric}"] = bounded_secondary_summary(
                [row[metric] for row in selected],
                seed=stable_seed(model, lesion, metric, "lesion_early"),
            )
    (out_root / "LESION_EARLY_ICTAL_SUMMARY.json").write_text(json.dumps({
        "contract": "topic5_rnn_motif_lesion_early_ictal_v0_4",
        "status": "SECONDARY_TARGET_READOUT_COMPLETE",
        "target_selection_used_for_lesions": False,
        "fields_generated_from_all_heldout_interictal_events": True,
        "n_primary_subjects": len({row["subject"] for row in patient_rows if row["primary"]}),
        "n_primary_subjects_with_matched_inference": len({
            row["subject"] for row in patient_rows
            if row["primary"] and row["matched_inference_available"]
        }),
        "inference_rule": (
            "all targeted lesion fields are retained descriptively; cohort statistics use only "
            "patient/model/lesion rows whose A and B producers each had at least 200 matched controls; "
            "inferential p-values additionally require at least 5 unique primary patients"
        ),
        "statistics": statistics,
    }, indent=2))
    print(json.dumps({"status": "COMPLETE", "n_rows": len(rows),
                      "n_patient_rows": len(patient_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
