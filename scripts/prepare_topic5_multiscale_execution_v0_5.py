#!/usr/bin/env python3
"""Freeze the v0.5 target-free execution contract and 531-unit schedule."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
SOURCE_PATHS = (
    "src/topic5_wiring_economy_rnn.py",
    "src/topic5_lbss_rnn_v0_2.py",
    "src/topic5_multiscale_scaffold_v0_5.py",
    "src/topic5_rnn_motif_v0_4.py",
    "scripts/train_topic5_lbss_unit_v0_2.py",
    "scripts/train_topic5_multiscale_scaffold_unit_v0_5.py",
    "scripts/build_topic5_l2m_graph_controls_v0_5.py",
    "scripts/run_topic5_multiscale_training_v0_5.py",
    "scripts/run_topic5_v0_5_target_free.py",
)
CACHE_INPUTS = (
    "events.npz",
    "events_raw.npz",
    "events_suffix_null_seed0.npz",
    "events_suffix_null_seed1.npz",
    "events_suffix_null_seed2.npz",
    "plane.npz",
    "provenance.json",
    "train_only_modes.npz",
)
STAGE_EVIDENCE = (
    "FULL_PARENT_CACHE_MANIFEST.json",
    "STAGE_A_COMPLETE.json",
    "STAGE_B_COMPLETE.json",
    "STAGE_C_GRAPH_CONTROL_COMPLETE.json",
    "STAGE_D_J_COMPLETE.json",
    "J_ESTIMAND_PREFREEZE_REPAIR.json",
    "CROSSFIT_NONLOCALITY_FIT_SUMMARY.csv",
    "CROSSFIT_NONLOCALITY_PATIENT_SUMMARY.csv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    for marker in ("STAGE_A_COMPLETE.json", "STAGE_B_COMPLETE.json",
                   "STAGE_C_GRAPH_CONTROL_COMPLETE.json", "STAGE_D_J_COMPLETE.json"):
        if not (OUT_ROOT / marker).exists():
            raise RuntimeError(f"cannot freeze execution before {marker}")
    if not (OUT_ROOT / "TARGET_PHYSICAL_EMBARGO_ACTIVE.json").exists():
        raise RuntimeError("target physical embargo is not active")
    census = pd.read_csv(OUT_ROOT / "FULL_PARENT_FIT_CENSUS.csv")
    reuse = pd.read_csv(OUT_ROOT / "V0_3_CHECKPOINT_REUSE_AUDIT.csv")
    reused = set(reuse.loc[reuse.checkpoint_reuse_eligible.astype(bool), "fit_id"])
    phase1 = []
    for fit_id in census.fit_id:
        arms = ["C-suffix"] if fit_id in reused else ["L0", "L1", "L3", "C-suffix"]
        for arm in arms:
            for seed in range(3):
                phase1.append({"phase": 1, "fit_id": fit_id, "arm": arm, "seed": seed})
    phase2 = [
        {"phase": 2, "fit_id": fit_id, "arm": "L2m", "seed": seed}
        for fit_id in census.fit_id for seed in range(3)
    ]
    schedule = pd.DataFrame(phase1 + phase2)
    if len(schedule) != 531 or len(phase1) != 405 or len(phase2) != 126:
        raise RuntimeError(f"formal schedule mismatch: {len(phase1)} + {len(phase2)}")
    schedule.to_csv(OUT_ROOT / "FORMAL_TRAINING_SCHEDULE.csv", index=False)

    # Freeze the exact bytes consumed by every formal unit.  The Stage-A
    # manifest predates train-only modes and suffix-null construction, so
    # copying it here would leave the scientifically important Stage-B inputs
    # outside the immutable execution contract.
    cache_records = []
    for fit_id in sorted(census.fit_id.astype(str)):
        fit_root = OUT_ROOT / "cache" / fit_id
        record = {"fit_id": fit_id, "files": {}}
        for name in CACHE_INPUTS:
            path = fit_root / name
            if not path.exists():
                raise FileNotFoundError(path)
            record["files"][name] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        cache_records.append(record)
    evidence_hashes = {}
    for name in STAGE_EVIDENCE:
        path = OUT_ROOT / name
        if not path.exists():
            raise FileNotFoundError(path)
        evidence_hashes[name] = sha256_file(path)
    input_manifest = {
        "contract": "topic5_multiscale_scaffold_formal_input_manifest_v0_5",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_values_read": False,
        "fits": len(cache_records),
        "required_files_per_fit": list(CACHE_INPUTS),
        "stage_evidence_hashes": evidence_hashes,
        "cache_records": cache_records,
    }
    input_manifest_path = OUT_ROOT / "INPUT_CACHE_MANIFEST.json"
    input_manifest_path.write_text(json.dumps(input_manifest, indent=2) + "\n")
    snapshot = OUT_ROOT / "run_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for relative in SOURCE_PATHS:
        source = ROOT / relative
        if not source.exists():
            raise FileNotFoundError(source)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[relative] = sha256_file(source)
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, text=True
    ).splitlines()
    contract = {
        "contract": "topic5_multiscale_scaffold_execution_v0_5",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_source_files": [path for path in untracked if path in SOURCE_PATHS],
        "source_hashes": hashes,
        "source_mutation_after_freeze_forbidden": True,
        "target_physical_embargo_required": True,
        "target_values_read": False,
        "formal_units": 531,
        "phase1_units": 405,
        "phase2_units": 126,
        "patients": int(census.subject.nunique()),
        "fits": int(census.fit_id.nunique()),
        "exact_reuse_fits": len(reused),
        "full_retrain_fits": int(census.fit_id.nunique() - len(reused)),
        "model": "state_dim_1_leaky_full_tissue_rnn",
        "arms": ["L0", "L1", "L2m", "L3", "C-suffix"],
        "seeds": [0, 1, 2],
        "checkpoint_rule": "BEST_VALIDATION_EPOCH_AT_OR_AFTER_MASK_FREEZE",
        "phase_dependency": "L3_FINAL_MASK_THEN_EXACT_L2M_GRAPH_THEN_L2M_REFIT",
        "primary_interictal_test": "spearman(J, distal_metric_L3_minus_L2m)>0",
        "target_status": "PHYSICALLY_EMBARGOED",
        "schedule_sha256": sha256_file(OUT_ROOT / "FORMAL_TRAINING_SCHEDULE.csv"),
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "stage_evidence_hashes": evidence_hashes,
        "formal_input_files_hashed": len(cache_records) * len(CACHE_INPUTS),
    }
    temporary = OUT_ROOT / "RUN_CONTRACT.json.tmp"
    temporary.write_text(json.dumps(contract, indent=2))
    temporary.replace(OUT_ROOT / "RUN_CONTRACT.json")
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
