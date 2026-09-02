#!/usr/bin/env python3
"""Fail-closed pre-unseal guard for the detached v0.5 pipeline.

This process never imports a target reader.  It waits for the target-free
analysis-metric freeze, verifies every external and worktree input that the
locked scorer or final claim/figure machinery will consume, records the exact
bytes it accepted, and only then resumes the paused posttraining process.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def verify_hash(path: Path, expected: str, label: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"hash mismatch for {label}: {path}")
    return {"path": str(path), "sha256": actual}


def verify_prefreeze(out: Path) -> dict:
    if (out / "TARGET_UNSEAL_AUTHORIZATION.json").exists():
        raise RuntimeError("target authorization exists before resume guard")

    closeout_manifest_path = out / "CLOSEOUT_TOOLING_PREFREEZE_MANIFEST.json"
    closeout_manifest = load_json(closeout_manifest_path)
    closeout_files = []
    if not (
        closeout_manifest.get("status") == "PASS_TARGET_FREE"
        and closeout_manifest.get("target_values_read") is False
        and closeout_manifest.get("source_count")
        == len(closeout_manifest.get("sources", {}))
        and closeout_manifest.get("source_count") == 8
    ):
        raise RuntimeError("closeout-tooling prefreeze manifest is invalid")
    for relative, expected in closeout_manifest.get("sources", {}).items():
        closeout_files.append(
            verify_hash(ROOT / relative, expected, f"closeout_source:{relative}")
        )
    closeout_producer = Path(str(closeout_manifest.get("producer", "")))
    if not closeout_producer.is_file() or sha256_file(closeout_producer) != str(
        closeout_manifest.get("producer_sha256", "")
    ):
        raise RuntimeError("closeout-tooling freezer provenance changed")

    hotfill_active = out / "ATTENUATION_HOTFILL_ACTIVE.json"
    hotfill_complete_path = out / "ATTENUATION_HOTFILL_COMPLETE.json"
    hotfill_parity_path = out / "ATTENUATION_HOTFILL_EXACT_PARITY.json"
    hotfill_evidence: dict = {"status": "NOT_USED"}
    if hotfill_active.exists():
        raise RuntimeError("attenuation hotfill is still active before target unseal")
    if hotfill_complete_path.exists() or hotfill_parity_path.exists():
        if not hotfill_complete_path.exists() or not hotfill_parity_path.exists():
            raise RuntimeError("attenuation hotfill provenance is incomplete")
        hotfill = load_json(hotfill_complete_path)
        parity = load_json(hotfill_parity_path)
        producer = Path(str(hotfill.get("producer_script", "")))
        if not (
            hotfill.get("status") == "PASS_TARGET_FREE"
            and hotfill.get("target_values_read") is False
            and parity.get("status") == "PASS_TARGET_FREE"
            and parity.get("target_values_read") is False
            and parity.get("events") == 1492
            and parity.get("mismatches") == 0
            and producer.exists()
            and sha256_file(producer) == hotfill.get("producer_script_sha256")
            and hotfill.get("exact_parity_sha256") == sha256_file(hotfill_parity_path)
        ):
            raise RuntimeError("attenuation hotfill contract or parity evidence is invalid")
        annotated = []
        for cache in (out / "attenuation/unit_cache").glob("**/*.json.gz"):
            with gzip.open(cache, "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
            if "rollout_dedup_contract" not in payload:
                continue
            if not (
                payload.get("rollout_dedup_contract")
                == "DETERMINISTIC_SAME_MODEL_SAME_FIRST_RANK_EXACT_EXPANSION"
                and payload.get("rollout_dedup_producer_sha256")
                == hotfill.get("producer_script_sha256")
                and payload.get("target_values_read") is False
            ):
                raise RuntimeError(f"invalid hotfilled attenuation cache: {cache}")
            annotated.append(cache)
        if len(annotated) != int(hotfill.get("hotfilled", -1)):
            raise RuntimeError("hotfill completion count does not match annotated caches")
        hotfill_evidence = {
            "status": "PASS_TARGET_FREE",
            "complete": verify_hash(
                hotfill_complete_path, sha256_file(hotfill_complete_path), "hotfill_complete"
            ),
            "exact_parity": verify_hash(
                hotfill_parity_path, hotfill["exact_parity_sha256"], "hotfill_parity"
            ),
            "annotated_unit_targets": len(annotated),
            "producer": verify_hash(
                producer, hotfill["producer_script_sha256"], "hotfill_producer"
            ),
        }

    mixture_repair_path = out / "TRAIN_PREVALENCE_MIXTURE_REPAIR_COMPLETE.json"
    mixture_repair = load_json(mixture_repair_path)
    if not (
        mixture_repair.get("status") == "PASS_TARGET_FREE"
        and mixture_repair.get("target_values_read") is False
        and mixture_repair.get("oracle_ab_vectors_changed") is False
        and mixture_repair.get("changed_patient_arm_fields") == 70
    ):
        raise RuntimeError("train-prevalence mixture repair is not target-free PASS")
    repair_source = Path(str(mixture_repair.get("producer_script", "")))
    if (
        not repair_source.exists()
        or sha256_file(repair_source) != mixture_repair.get("producer_script_sha256")
    ):
        raise RuntimeError("train-prevalence mixture repair source hash changed")

    metric_complete = load_json(out / "PREUNSEAL_ANALYSIS_METRIC_FREEZE_COMPLETE.json")
    metric_manifest_path = out / "PREUNSEAL_ANALYSIS_METRIC_MANIFEST.json"
    if (
        metric_complete.get("status") != "PASS_TARGET_FREE"
        or metric_complete.get("target_values_read") is not False
        or metric_complete.get("manifest_sha256") != sha256_file(metric_manifest_path)
    ):
        raise RuntimeError("analysis-metric freeze marker is invalid")
    metric_manifest = load_json(metric_manifest_path)
    if (
        metric_manifest.get("status") != "PASS_TARGET_FREE"
        or metric_manifest.get("target_values_read") is not False
    ):
        raise RuntimeError("analysis-metric manifest is not target-free PASS")
    metric_files = []
    for relative, evidence in metric_manifest.get("files", {}).items():
        metric_files.append(verify_hash(out / relative, evidence["sha256"], relative))

    empirical_path = out / "EMPIRICAL_FIELD_INPUT_PREFREEZE_MANIFEST.json"
    empirical = load_json(empirical_path)
    if not (
        empirical.get("status") == "PASS"
        and empirical.get("target_values_read") is False
        and empirical.get("fit_rows") == 42
        and empirical.get("spatial_patients") == 28
        and empirical.get("early_patients_covered") == 17
    ):
        raise RuntimeError("empirical-field prefreeze manifest is invalid")
    empirical_files = [
        verify_hash(Path(row["path"]), row["expected_sha256"], "empirical_field")
        for row in empirical.get("fields", [])
    ]

    run_contract = load_json(out / "RUN_CONTRACT.json")
    formal_sources = [
        verify_hash(ROOT / relative, expected, f"formal_source:{relative}")
        for relative, expected in run_contract.get("source_hashes", {}).items()
    ]
    stage_f = load_json(out / "STAGE_F_RUN_SNAPSHOT.json")
    stage_f_sources = [
        verify_hash(ROOT / relative, expected, f"stage_f_source:{relative}")
        for relative, expected in stage_f.get("source_hashes", {}).items()
    ]
    posttraining = load_json(out / "POSTTRAINING_PIPELINE_SNAPSHOT.json")
    posttraining_paths = {
        "driver": ROOT / "scripts/run_topic5_multiscale_posttraining_v0_5.py",
        "embargo": ROOT / "scripts/run_topic5_v0_5_target_free.py",
        "interictal": ROOT / "scripts/analyse_topic5_multiscale_interictal_v0_5.py",
        "stage_f": ROOT / "scripts/run_topic5_multiscale_stage_f_v0_5.py",
        "authorize": ROOT / "scripts/prepare_topic5_multiscale_target_unseal_v0_5.py",
        "score": ROOT / "scripts/score_topic5_multiscale_early_ictal_v0_5.py",
        "figure": ROOT / "scripts/paper_figures/plot_topic5_figure6_multiscale_scaffold_v0_5.py",
    }
    posttraining_sources = [
        verify_hash(posttraining_paths[key], expected, f"posttraining_source:{key}")
        for key, expected in posttraining.get("source_hashes", {}).items()
    ]
    scorer_repair_path = out / "SCORER_CONTRACT_PREFREEZE_REPAIR.json"
    scorer_repair = load_json(scorer_repair_path)
    if not (
        scorer_repair.get("status") == "PASS_TARGET_FREE"
        and scorer_repair.get("target_values_read") is False
        and scorer_repair.get("target_authorization_absent") is True
        and scorer_repair.get("snapshot_sha256")
        == sha256_file(out / "POSTTRAINING_PIPELINE_SNAPSHOT.json")
        and scorer_repair.get("scorer_sha256") == posttraining["source_hashes"]["score"]
        and scorer_repair.get("authorizer_sha256")
        == posttraining["source_hashes"]["authorize"]
    ):
        raise RuntimeError("target-free scorer-contract repair evidence is invalid")

    finalizer_manifest_path = out / "FIGURE6_FINALIZER_R2_PREFREEZE_MANIFEST.json"
    finalizer = load_json(finalizer_manifest_path)
    if finalizer.get("target_values_read") is not False:
        raise RuntimeError("Figure-6 finalizer contract is not target-free")
    finalizer_checks = {
        "panel_c_decision_sha256": out / "FIGURE6_PREUNSEAL_PANEL_C_DECISION.json",
        "panel_e_decision_sha256": out / "FIGURE6_PREUNSEAL_PANEL_E_DECISION.json",
        "panel_i_decision_sha256": out / "FIGURE6_PREUNSEAL_PANEL_I_DECISION.json",
        "finalizer_script_sha256": ROOT / "scripts/finalize_topic5_figure6_multiscale_scaffold_v0_5_r2.py",
    }
    finalizer_files = [
        verify_hash(path, finalizer[key], f"figure_prefreeze:{key}")
        for key, path in finalizer_checks.items()
    ]

    adjudicator_manifest_path = out / "FINAL_CLAIM_ADJUDICATOR_PREFREEZE_MANIFEST.json"
    adjudicator = load_json(adjudicator_manifest_path)
    if adjudicator.get("target_values_read") is not False:
        raise RuntimeError("claim adjudicator contract is not target-free")
    adjudicator_file = verify_hash(
        ROOT / "scripts/adjudicate_topic5_multiscale_claims_v0_5.py",
        adjudicator["script_sha256"], "claim_adjudicator",
    )

    return {
        "closeout_tooling_manifest": verify_hash(
            closeout_manifest_path,
            sha256_file(closeout_manifest_path),
            "closeout_tooling_manifest",
        ),
        "closeout_tooling_sources": closeout_files,
        "attenuation_hotfill": hotfill_evidence,
        "metric_manifest": verify_hash(
            metric_manifest_path, metric_complete["manifest_sha256"], "metric_manifest"
        ),
        "train_prevalence_mixture_repair": {
            "path": str(mixture_repair_path),
            "sha256": sha256_file(mixture_repair_path),
        },
        "metric_files": metric_files,
        "empirical_manifest": {
            "path": str(empirical_path), "sha256": sha256_file(empirical_path),
        },
        "empirical_files": empirical_files,
        "formal_sources": formal_sources,
        "stage_f_sources": stage_f_sources,
        "posttraining_sources": posttraining_sources,
        "scorer_contract_prefreeze_repair": verify_hash(
            scorer_repair_path, sha256_file(scorer_repair_path), "scorer_contract_repair"
        ),
        "figure_prefreeze_files": finalizer_files,
        "claim_adjudicator": adjudicator_file,
    }


def terminate_group(main_pid: int) -> None:
    try:
        os.killpg(main_pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--main-pid", type=int, required=True)
    parser.add_argument("--freezer-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=2)
    parser.add_argument("--timeout-hours", type=float, default=72.0)
    args = parser.parse_args()
    out = args.out_root.resolve()
    complete = out / "PREUNSEAL_ANALYSIS_METRIC_FREEZE_COMPLETE.json"
    begin = time.monotonic()
    try:
        while not complete.exists():
            if (out / "TARGET_UNSEAL_AUTHORIZATION.json").exists():
                raise RuntimeError("target authorization preceded resume guard")
            if (out / "STAGE_F_TARGET_FREE_FAILED.json").exists():
                raise RuntimeError("Stage F failed before resume guard")
            try:
                os.kill(args.freezer_pid, 0)
            except ProcessLookupError as error:
                raise RuntimeError("analysis-metric freezer exited without marker") from error
            if time.monotonic() - begin > args.timeout_hours * 3600:
                raise TimeoutError("timed out waiting for analysis-metric freeze")
            time.sleep(max(1, int(args.poll_seconds)))
        evidence = verify_prefreeze(out)
        created = datetime.now(timezone.utc).isoformat()
        write_json(out / "PREUNSEAL_RESUME_GUARD_COMPLETE.json", {
            "contract": "topic5_preunseal_resume_guard_v0_5",
            "status": "PASS_TARGET_FREE",
            "created_utc": created,
            "target_values_read": False,
            "main_pid": args.main_pid,
            "evidence": evidence,
        })
        os.kill(args.main_pid, signal.SIGCONT)
    except Exception as error:
        write_json(out / "PREUNSEAL_RESUME_GUARD_FAILED.json", {
            "status": "FAIL_CLOSED",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "target_values_read": False,
            "error": repr(error),
        })
        terminate_group(args.main_pid)
        raise


if __name__ == "__main__":
    main()
