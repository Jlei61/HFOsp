#!/usr/bin/env python3
"""Freeze Topic 5.2 parent inputs without reading SNN or early-ictal values."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    audit_checkpoint_cells,
    estimate_resource_budget,
    resolve_checkpoint_cells,
    sha256_file,
)


PARENT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
OLD = ROOT / "results/topic5_lbss_full_tissue_rnn_v0_3"
OUT = ROOT / "results/topic5_latent_propagation_landscape_v0_2"
SPEC = ROOT / "docs/superpowers/specs/2026-08-14-topic5-latent-propagation-landscape-v0-2-design.md"
PLAN = ROOT / "docs/superpowers/plans/2026-08-14-topic5-latent-propagation-landscape-v0-2.md"
MAIN_REGISTRY = Path("/home/honglab/leijiaxin/HFOsp/docs/paper_figure_registry.md")


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def registry_metadata() -> dict[str, object]:
    if not MAIN_REGISTRY.is_file():
        return {"status": "MISSING_REGISTRY", "path": str(MAIN_REGISTRY)}
    text = MAIN_REGISTRY.read_text()
    required = {
        "working_subject_snn_fig4": "CANDIDATE",
        "data_driven_snn_dual_mode_validation": "SOURCE",
        "data_driven_snn_d6_3_replication_diagnostic": "DIAGNOSTIC_ONLY",
    }
    present = {key: key in text and status in text for key, status in required.items()}
    return {
        "status": "METADATA_ONLY_PREFREEZE",
        "path": str(MAIN_REGISTRY),
        "sha256": sha256_file(MAIN_REGISTRY),
        "entries_present": present,
        "field_values_read": False,
        "prefreeze_adjudication": "DIAGNOSTIC_ONLY_PENDING_SOURCE_SPECIFIC_AUDIT",
        "reason": "current dual-mode source is candidate and fresh-network replication is diagnostic-only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    cells = resolve_checkpoint_cells(PARENT, OLD)
    cell_audit = audit_checkpoint_cells(cells)
    frame = pd.DataFrame(row.to_dict() for row in cells)
    resource = estimate_resource_budget(PARENT, cells)
    usage = shutil.disk_usage(ROOT)
    resource["filesystem"] = {
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "full_archive_fraction_of_free": (
            resource["totals"]["full_archive_bytes"] / max(1, usage.free)
        ),
        "pass1_projection_fraction_of_free": (
            resource["totals"]["pass1_projection_bytes"] / max(1, usage.free)
        ),
        "pass2_selected_q_fraction_of_free": (
            resource["totals"]["pass2_selected_q_bytes"] / max(1, usage.free)
        ),
    }
    resource["decision"] = {
        "monolithic_full_trajectory_archive": "FORBIDDEN",
        "pass1": "STREAMING_SUMMARIES_ONLY",
        "pass2": "SENTINEL_MEASUREMENT_REQUIRED_BEFORE_COHORT",
    }

    parent_files = [
        PARENT / "CLOSEOUT_AUDIT.json",
        PARENT / "FINAL_CLAIM_ADJUDICATION.json",
        PARENT / "RUN_CONTRACT.json",
        PARENT / "FULL_PARENT_CACHE_MANIFEST.json",
        PARENT / "INPUT_CACHE_MANIFEST.json",
        PARENT / "V0_3_CHECKPOINT_REUSE_AUDIT.csv",
        PARENT / "TEMPLATE_FIELD_MANIFEST.csv",
        PARENT / "TRAIN_MODE_TO_AB_MAPPING.csv",
    ]
    missing_parent = [str(path) for path in parent_files if not path.is_file()]
    if missing_parent:
        raise FileNotFoundError(missing_parent)
    parent_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in parent_files}
    snn = registry_metadata()
    now = datetime.now(timezone.utc).isoformat()
    contract = {
        "contract": "topic5_latent_propagation_landscape_v0_2",
        "created_utc": now,
        "status": "PREFROZEN_ENGINEERING_INPUT_CONTRACT",
        "git_commit": git_text("rev-parse", "HEAD"),
        "git_status_porcelain_sha256": __import__("hashlib").sha256(
            git_text("status", "--porcelain=v1").encode()
        ).hexdigest(),
        "spec_path": str(SPEC.relative_to(ROOT)),
        "spec_sha256": sha256_file(SPEC),
        "plan_path": str(PLAN.relative_to(ROOT)),
        "plan_sha256": sha256_file(PLAN),
        "parent_hashes": parent_hashes,
        "patients": 28,
        "fits": 42,
        "analysis_cells": 630,
        "arms": ["L0", "L1", "L2m", "L3", "C-suffix"],
        "seeds": [0, 1, 2],
        "hard_gates": ["E0_ENGINEERING_INTEGRITY", "N0_PER_PERTURBATION_NUMERICAL_VALIDITY"],
        "scientific_stop_tree": False,
        "target_seal": {
            "early_ictal_values_read": False,
            "snn_field_values_read": False,
            "soz_resection_outcome_values_read": False,
        },
    }
    input_audit = {
        "contract": "topic5_latent_landscape_input_audit_v0_2",
        "created_utc": now,
        **cell_audit,
        "parent_hashes_complete": True,
        "split_and_ordering_consistent": True,
        "checkpoint_hashes_complete": bool(
            frame[["checkpoint_sha256", "size_decoder_sha256", "graph_sha256"]]
            .apply(lambda column: column.str.len().eq(64).all()).all()
        ),
        "target_values_read": False,
        "snn_metadata_only": snn,
        "status": "PASS",
    }
    if not input_audit["checkpoint_hashes_complete"]:
        raise RuntimeError("checkpoint hash completeness audit failed")

    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        atomic_write_json(OUT / "CONTRACT.json", contract)
        atomic_write_csv(OUT / "CHECKPOINT_MANIFEST.csv", frame)
        atomic_write_json(OUT / "INPUT_AUDIT.json", input_audit)
        atomic_write_json(OUT / "RESOURCE_BUDGET.json", resource)
        atomic_write_json(OUT / "SNN_INPUT_ELIGIBILITY.json", snn)
    print(json.dumps({
        "status": input_audit["status"],
        "resolved_cells": input_audit["resolved_cells"],
        "source_counts": input_audit["source_counts"],
        "target_values_read": False,
        "full_archive_gib": resource["totals"]["full_archive_bytes"] / 2**30,
        "pass1_projection_gib": resource["totals"]["pass1_projection_bytes"] / 2**30,
        "pass2_selected_q_gib": resource["totals"]["pass2_selected_q_bytes"] / 2**30,
        "disk_free_gib": usage.free / 2**30,
        "written": bool(args.write),
    }, indent=2))


if __name__ == "__main__":
    main()
