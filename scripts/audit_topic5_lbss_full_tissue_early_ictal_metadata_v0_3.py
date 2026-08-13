#!/usr/bin/env python3
"""Target-value-free Figure 3 join audit for full-tissue LBSS v0.3.

Only inventory columns and JSON channel-name arrays are read.  No broadband or
gamma activation array and no precomputed observed/null field score is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


CANONICAL_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_OUT = Path("results/topic5_lbss_full_tissue_rnn_v0_3")
EVENT_COLUMNS = (
    "dataset", "subject", "seizure_idx", "group_id", "phenotype", "band",
    "permutation_seed",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def target_metadata_path(root: Path, subject: str, phenotype: str) -> Path:
    if phenotype == "strict_broadband":
        return root / "results/topic5_ictal_recruitment/t0_feature_cache_bb150_1_150" / f"{subject}.json"
    if phenotype == "gamma_nonbroadband":
        return root / "results/topic5_ictal_recruitment/v2_band_scan/cache" / f"{subject}.json"
    raise ValueError(f"unknown Figure 3 phenotype: {phenotype}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--canonical-root", type=Path, default=CANONICAL_ROOT)
    args = parser.parse_args()
    out = args.out_root.resolve()
    canonical = args.canonical_root.resolve()

    model_manifest_path = out / "INPUT_MANIFEST.json"
    model_manifest = json.loads(model_manifest_path.read_text())
    model_subjects = sorted({str(item["subject"]) for item in model_manifest["fits"]})
    if len(model_subjects) != 21:
        raise RuntimeError(f"full-tissue spatial cohort changed: {len(model_subjects)} != 21")

    event_path = canonical / (
        "results/topic5_ictal_recruitment/tspectral_field_concordance/"
        "clinical_onset_gradient_field_cohort_stat_event.csv"
    )
    events = pd.read_csv(event_path, usecols=list(EVENT_COLUMNS))
    events = events[events.group_id.eq("all_phenotype_matched")].copy()
    if events.subject.nunique() != 17 or len(events) != 167:
        raise RuntimeError(
            f"canonical Figure 3 denominator changed: {events.subject.nunique()}/17 patients, "
            f"{len(events)}/167 seizures"
        )
    figure3_subjects = sorted(events.subject.unique().tolist())
    joined_subjects = sorted(set(model_subjects) & set(figure3_subjects))
    missing_model = sorted(set(figure3_subjects) - set(model_subjects))
    model_without_ictal = sorted(set(model_subjects) - set(figure3_subjects))

    field_root = canonical / "results/interictal_propagation_masked/template_gradient_fields/per_subject"
    rows = []
    for row in events[events.subject.isin(joined_subjects)].itertuples():
        field_path = field_root / f"{row.subject}.json"
        field = json.loads(field_path.read_text())["interictal_field"]
        order = [str(value) for value in field["contact_order"]]
        metadata_path = target_metadata_path(canonical, str(row.subject), str(row.phenotype))
        metadata = json.loads(metadata_path.read_text())
        target_contacts = [str(value) for value in metadata["channels"]]
        exact = sorted(set(order) & set(target_contacts))
        rows.append({
            "dataset": row.dataset,
            "subject": row.subject,
            "seizure_idx": int(row.seizure_idx),
            "phenotype": row.phenotype,
            "band": row.band,
            "permutation_seed": int(row.permutation_seed),
            "n_model_field_contacts": len(order),
            "n_target_metadata_contacts": len(target_contacts),
            "n_exact_join_contacts": len(exact),
            "exact_join_eligible": len(exact) >= 6,
            "field_metadata_sha256": sha256(field_path),
            "target_channel_metadata_sha256": sha256(metadata_path),
            "target_numeric_values_read": False,
        })
    inventory = pd.DataFrame(rows).sort_values(["subject", "seizure_idx"])
    if inventory.subject.nunique() != 12 or len(inventory) != 141:
        raise RuntimeError(
            f"spatial/Figure3 intersection changed: {inventory.subject.nunique()}/12 patients, "
            f"{len(inventory)}/141 seizures"
        )
    if not inventory.exact_join_eligible.all():
        bad = inventory.loc[~inventory.exact_join_eligible, ["subject", "seizure_idx", "n_exact_join_contacts"]]
        raise RuntimeError(f"Figure 3 exact join below six contacts:\n{bad.to_string(index=False)}")
    inventory_path = out / "EARLY_ICTAL_METADATA_INVENTORY.csv"
    inventory.to_csv(inventory_path, index=False)

    attrition = pd.DataFrame({
        "subject": figure3_subjects,
        "in_full_tissue_spatial_cohort": [subject in model_subjects for subject in figure3_subjects],
        "n_figure3_seizures": [int((events.subject == subject).sum()) for subject in figure3_subjects],
        "status": [
            "SPATIAL_MODEL_AND_EXACT_JOIN" if subject in joined_subjects
            else "NO_FULL_TISSUE_SPATIAL_MODEL"
            for subject in figure3_subjects
        ],
    })
    attrition_path = out / "EARLY_ICTAL_COHORT_ATTRITION.csv"
    attrition.to_csv(attrition_path, index=False)

    payload = {
        "contract": "topic5_lbss_full_tissue_early_ictal_metadata_v0_3",
        "figure3_parent_patients": 17,
        "figure3_parent_seizures": 167,
        "full_tissue_spatial_patients": 21,
        "actual_spatial_join_patients": 12,
        "actual_spatial_join_seizures": 141,
        "actual_spatial_join": joined_subjects,
        "missing_full_tissue_model_from_figure3": missing_model,
        "full_tissue_model_without_figure3_target": model_without_ictal,
        "exact_join_contacts_min": int(inventory.n_exact_join_contacts.min()),
        "exact_join_contacts_median": float(inventory.n_exact_join_contacts.median()),
        "exact_join_contacts_max": int(inventory.n_exact_join_contacts.max()),
        "event_inventory_sha256": sha256(event_path),
        "model_input_manifest_sha256": sha256(model_manifest_path),
        "inventory_csv_sha256": sha256(inventory_path),
        "attrition_csv_sha256": sha256(attrition_path),
        "event_columns_deserialized": list(EVENT_COLUMNS),
        "target_numeric_columns_deserialized": False,
        "target_activation_arrays_deserialized": False,
        "target_values_read": False,
        "interpretation": (
            "The 17-patient/167-seizure contact-space result remains the full-cohort cross-state endpoint. "
            "The geometry-dependent full-tissue mechanism analysis has a prespecified exact intersection "
            "of 12 patients/141 seizures; five Figure 3 patients lack a full-tissue spatial model and are "
            "reported rather than silently excluded."
        ),
    }
    write_json(out / "EARLY_ICTAL_METADATA_INVENTORY.json", payload)
    write_json(out / "EARLY_ICTAL_METADATA_AUDIT_COMPLETE.json", {
        "status": "PASS",
        "n_patients": 12,
        "n_seizures": 141,
        "target_values_read": False,
    })


if __name__ == "__main__":
    main()
