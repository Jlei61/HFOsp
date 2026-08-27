#!/usr/bin/env python3
"""Build the read-only E384 Phase-0 inventory for H2b cross-task transfer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.topic5_continuous_marked_state_h2b.contract import (
    LEAD_MINUTES,
    PRIMARY_LEAD_MINUTES,
    R1_6_MACHINE_AUDIT,
    R1_6_ROOT,
    RESULT_ROOT,
    atomic_csv,
    atomic_json,
)
from src.topic5_continuous_marked_state_h2b.inventory import (
    E384_STABLE_SEEDS,
    E384_SUBJECT,
    exclusion_funnel_payload,
    load_epilepsiae_sql_seizures,
    load_r16_checkpoint_inventory,
    load_state_support_arrays,
    summarise_seizure_support,
    target_inventory,
)


DEFAULT_SOURCE_REPO = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_E384_SQL = Path(
    "/mnt/epilepsia_data/all_data_sqls/pat_38402_2012-12-20.sql"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default=E384_SUBJECT, choices=[E384_SUBJECT])
    parser.add_argument("--source-repo-root", type=Path, default=DEFAULT_SOURCE_REPO)
    parser.add_argument("--r1-6-root", type=Path, default=R1_6_ROOT)
    parser.add_argument("--r1-6-machine-audit", type=Path, default=R1_6_MACHINE_AUDIT)
    parser.add_argument("--epilepsiae-sql", type=Path, default=DEFAULT_E384_SQL)
    args = parser.parse_args()

    manifest_root = RESULT_ROOT / "manifests"
    checkpoints = load_r16_checkpoint_inventory(
        audit_path=args.r1_6_machine_audit,
        result_root=args.r1_6_root,
        source_repo_root=args.source_repo_root,
        subject=args.subject,
        seeds=E384_STABLE_SEEDS,
    )
    crosswalk = load_epilepsiae_sql_seizures(
        args.epilepsiae_sql, subject=args.subject
    )
    arrays = load_state_support_arrays(args.source_repo_root, subject=args.subject)
    coverage = arrays["coverage"]
    for row in crosswalk:
        row["in_development_partition"] = bool(
            float(row["onset_epoch"]) < float(coverage.dev_end_epoch)
        )
    support, funnel, detail = summarise_seizure_support(
        crosswalk,
        coverage=coverage,
        training_anchor_time=arrays["training_anchor_time"],
        training_anchor_session=arrays["training_anchor_session"],
        inference_anchor_time=arrays["inference_anchor_time"],
        inference_anchor_segment=arrays["inference_anchor_segment"],
        leads=LEAD_MINUTES,
    )
    targets = target_inventory(args.source_repo_root, subject=args.subject)
    for row in support:
        row.update({
            "n_frozen_checkpoints": checkpoints["n_checkpoints"],
            "checkpoint_seeds": "|".join(map(str, checkpoints["stable_seeds"])),
            "state_revision": checkpoints["entries"][0]["state_revision"],
            "seizure_metadata_source": str(args.epilepsiae_sql.resolve()),
            "seizure_metadata_truth": "Epilepsiae SQL recording/block/seizure",
            "preexisting_seizure_subtype_available": (
                targets["preexisting_seizure_subtype_available"]
            ),
            "preexisting_early_recruitment_available": (
                targets["preexisting_early_recruitment_available"]
            ),
            "evidence_layer_checkpoint_available": True,
            "evidence_layer_h1_stable": True,
        })
    primary = next(row for row in support if row["lead_minutes"] == PRIMARY_LEAD_MINUTES)
    checkpoints["subject_support"] = {
        "n_seizures_total": len(crosswalk),
        "n_seizures_development": sum(
            bool(row["in_development_partition"]) for row in crosswalk
        ),
        "primary_lead_minutes": PRIMARY_LEAD_MINUTES,
        "primary_n_eligible_seizures": primary["n_eligible_seizures"],
        "primary_support_tier": primary["support_tier"],
        **targets,
    }
    funnel_payload = exclusion_funnel_payload(
        checkpoint_inventory=checkpoints,
        support_funnel=funnel,
        targets=targets,
        source_arrays=arrays,
    )
    funnel_payload["per_seizure_support"] = detail
    funnel_payload["seizure_crosswalk"] = {
        "metadata_source": str(args.epilepsiae_sql.resolve()),
        "metadata_source_sha256": crosswalk[0]["metadata_source_sha256"],
        "n_rows": len(crosswalk),
        "n_matched": sum(bool(row["matched"]) for row in crosswalk),
        "n_exact_onset_delta_zero": sum(
            bool(row["onset_exact_match"]) for row in crosswalk
        ),
        "n_unmatched": sum(not bool(row["matched"]) for row in crosswalk),
        "n_ambiguous": sum(bool(row["ambiguous"]) for row in crosswalk),
        "yuquan_policy": (
            "Yuquan uses explicit recording-code plus within-record onset order; "
            "a match is accepted only when onset delta is exactly 0 seconds"
        ),
    }

    outputs = {
        "state_checkpoint_inventory": manifest_root / "state_checkpoint_inventory.json",
        "seizure_crosswalk": manifest_root / "seizure_crosswalk.csv",
        "seizure_support_by_lead": manifest_root / "seizure_support_by_lead.csv",
        "exclusion_funnel": manifest_root / "exclusion_funnel.json",
    }
    atomic_json(outputs["state_checkpoint_inventory"], checkpoints)
    atomic_csv(outputs["seizure_crosswalk"], crosswalk)
    atomic_csv(outputs["seizure_support_by_lead"], support)
    atomic_json(outputs["exclusion_funnel"], funnel_payload)
    print(json.dumps({
        "status": "COMPLETE",
        "subject": args.subject,
        "n_checkpoints": checkpoints["n_checkpoints"],
        "n_seizures_total": len(crosswalk),
        "n_seizures_development": checkpoints["subject_support"]["n_seizures_development"],
        "primary_lead_minutes": PRIMARY_LEAD_MINUTES,
        "primary_n_eligible_seizures": primary["n_eligible_seizures"],
        "primary_support_tier": primary["support_tier"],
        "outputs": {key: str(value) for key, value in outputs.items()},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
