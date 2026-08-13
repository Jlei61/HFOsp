#!/usr/bin/env python3
"""Wait for frozen interictal artifacts and resolve the spatial-search branch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import pandas as pd


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def target_sealed(out: Path) -> None:
    forbidden = (
        "TARGET_UNSEAL_AUTHORIZATION.json",
        "TARGET_ACCESS_AUDIT.json",
        "EARLY_ICTAL_SCORING_COMPLETE.json",
    )
    present = [name for name in forbidden if (out / name).exists()]
    if present:
        raise RuntimeError(f"spatial decision requires sealed target; found {present}")


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write(f"\n[{now()}] {' '.join(command)}\n")
        stream.flush()
        process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"command failed ({process.returncode}); see {log}")


def current_contract_is_selective(out: Path) -> tuple[bool, dict[str, dict[str, float]]]:
    """Apply the same matched-arm rule used by searched configurations.

    Distal gain alone is not sufficient: a spatial configuration that obtains a
    tiny distal improvement by degrading overall likelihood or free generation
    should enter the target-free search rather than being retained.
    """
    patient = pd.read_csv(out / "interictal_per_patient.csv")
    pivot = patient.pivot(index="subject", columns="arm")
    l3 = "L3_LOCAL_PLUS_LEARNED_LR"
    comparisons: dict[str, dict[str, float]] = {}
    for comparator in (
        "L0_LOCAL_ONLY",
        "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
        "L2_LOCAL_PLUS_RANDOM_LR",
        "C_L3_ORDER_SHUFFLED",
    ):
        comparisons[comparator] = {
            "median_overall_gain": float(
                (pivot["test_contact_nll"][comparator] - pivot["test_contact_nll"][l3]).median()
            ),
            "median_distal_gain": float(
                (pivot["distal_contact_nll"][comparator] - pivot["distal_contact_nll"][l3]).median()
            ),
            "median_rollout_gain": float(
                (pivot["rollout_spearman"][l3] - pivot["rollout_spearman"][comparator]).median()
            ),
        }
    matched = [comparisons[arm] for arm in (
        "L0_LOCAL_ONLY",
        "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
        "L2_LOCAL_PLUS_RANDOM_LR",
    )]
    retain = bool(
        all(item["median_distal_gain"] > 0 for item in matched)
        and all(item["median_overall_gain"] >= -0.01 for item in matched)
        and all(item["median_rollout_gain"] >= -0.02 for item in matched)
        and comparisons["C_L3_ORDER_SHUFFLED"]["median_distal_gain"] > 0
    )
    return retain, comparisons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--search-snapshot", type=Path, required=True)
    parser.add_argument("--postprocess-snapshot", type=Path, required=True)
    parser.add_argument("--python", default="/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--selected-root", type=Path)
    args = parser.parse_args()
    out = args.out_root.resolve()
    search_snapshot = args.search_snapshot.resolve()
    postprocess_snapshot = args.postprocess_snapshot.resolve()
    selected_root = (
        args.selected_root.resolve() if args.selected_root is not None
        else out.parent / "topic5_lbss_full_tissue_rnn_v0_4_selected"
    )
    search_name = "development_spatial_search_v0_4"
    search = out / search_name
    status_path = out / "SPATIAL_DECISION_WATCH_STATUS.json"
    log_root = out / "run_logs" / "spatial_decision_v0_4"
    try:
        pretarget_marker = out / "INTERICTAL_POSTPROCESS_PRETARGET_COMPLETE.json"
        screen_marker = search / "SCREEN_COMPLETE.json"
        # Spatial selection is target-free and independent of the longer
        # attenuation post-process.  Resolve it as soon as the frozen screen
        # finishes; only the irreversible target unseal below must wait for
        # every pre-target field.
        while not screen_marker.exists():
            target_sealed(out)
            if (out / "FORMAL_TRAINING_FAILED.json").exists():
                raise RuntimeError("formal training failed before spatial decision")
            if (out / "PIPELINE_FAILED.json").exists():
                failure = json.loads((out / "PIPELINE_FAILED.json").read_text())
                raise RuntimeError(f"pretarget postprocess failed: {failure}")
            complete = len(list((out / "per_fit").glob("*/*/seed*/DONE.json")))
            atomic(status_path, {
                "status": "WAITING_FOR_SPATIAL_SCREEN",
                "formal_complete_units": complete,
                "scheduled_units": 465,
                "pretarget_postprocess_complete": pretarget_marker.exists(),
                "spatial_screen_complete": screen_marker.exists(),
                "pid": os.getpid(),
                "updated_at": now(),
                "target_values_read": False,
            })
            time.sleep(min(30, max(5, args.poll_seconds)))

        target_sealed(out)
        retain, comparisons = current_contract_is_selective(out)
        trigger = {
            "contract": "topic5_lbss_spatial_search_trigger_v0_4",
            "current_matched_comparisons": comparisons,
            "current_contract_selective": retain,
            "action": "RETAIN_CURRENT_AND_UNSEAL" if retain else "RUN_TARGET_FREE_DEVELOPMENT_SEARCH",
            "target_values_read": False,
            "created_at": now(),
        }
        atomic(out / "SPATIAL_SEARCH_TRIGGER_DECISION.json", trigger)

        runner = search_snapshot / "scripts/run_topic5_lbss_spatial_search_v0_4.py"
        trainer = search_snapshot / "scripts/train_topic5_lbss_unit_v0_2.py"
        if not retain:
            base = [
                args.python, str(runner), "--out-root", str(out),
                "--search-name", search_name, "--trainer", str(trainer),
                "--workers", str(args.workers), "--device", "cuda:0",
            ]
            # The screen launcher already froze SEARCH_CONTRACT.json and all
            # 117 units.  Re-initializing here would overwrite the original
            # contract timestamp/hash provenance.  Reuse those immutable units
            # and let this newer snapshot perform only patient-first summary
            # and downstream confirmation.
            stages = ("summarize-screen",)
            for stage in stages:
                atomic(status_path, {
                    "status": "RUNNING_TARGET_FREE_SEARCH", "stage": stage,
                    "pid": os.getpid(), "updated_at": now(), "target_values_read": False,
                })
                run([*base, "--stage", stage], log_root / f"{stage}.log")
            screen = json.loads((search / "SCREEN_DECISION.json").read_text())
            if screen["joint_config_id"] != "base":
                run([*base, "--stage", "joint"], log_root / "joint.log")
            run([*base, "--stage", "select-confirm"], log_root / "select-confirm.log")
            run([*base, "--stage", "confirm"], log_root / "confirm.log")
            run([*base, "--stage", "summarize-confirm"], log_root / "summarize-confirm.log")
            decision = json.loads((search / "SPATIAL_MODEL_DECISION.json").read_text())
            if decision["selected_config_id"] is not None:
                atomic(status_path, {
                    "status": "RUNNING_FULL_COHORT_SELECTED_CONFIG_CONFIRMATION",
                    "selected_config_id": decision["selected_config_id"],
                    "pid": os.getpid(), "updated_at": now(), "target_values_read": False,
                })
                run([*base, "--stage", "formal-selected"], log_root / "formal-selected.log")
                run([*base, "--stage", "summarize-formal-selected"],
                    log_root / "summarize-formal-selected.log")
                formal = json.loads((search / "FORMAL_SELECTED_DECISION.json").read_text())
                if formal["verdict"] == "FULL_COHORT_SELECTIVE_NONLOCAL_CONFIRMED":
                    atomic(status_path, {
                        "status": "PREPARING_SELECTED_CONFIG_PRIMARY_ROOT",
                        "selected_config_id": decision["selected_config_id"],
                        "pid": os.getpid(), "updated_at": now(), "target_values_read": False,
                    })
                    selected_decision = {
                        **formal,
                        "search_snapshot": str(search_snapshot),
                        "selected_artifact_root": str(selected_root),
                        "target_values_read": False,
                    }
                    atomic(out / "SPATIAL_DECISION_REQUIRES_SELECTED_FIELD_FREEZE.json", selected_decision)
                    preparer = search_snapshot / "scripts/prepare_topic5_lbss_selected_primary_root_v0_4.py"
                    run([
                        args.python, str(preparer),
                        "--source-out-root", str(out),
                        "--selected-root", str(selected_root),
                        "--search-name", search_name,
                    ], log_root / "prepare-selected-root.log")
                    atomic(out / "PRIMARY_ARTIFACT_POINTER.json", {
                        "contract": "topic5_lbss_primary_artifact_pointer_v0_4",
                        "artifact_root": str(selected_root),
                        "selected_config_id": decision["selected_config_id"],
                        "reason": "FULL_COHORT_SELECTIVE_NONLOCAL_CONFIRMED",
                        "target_values_read": False,
                        "created_at": now(),
                    })
                    atomic(status_path, {
                        "status": "RUNNING_SELECTED_CONFIG_FROZEN_BENCHMARK",
                        "selected_config_id": decision["selected_config_id"],
                        "artifact_root": str(selected_root),
                        "pid": os.getpid(), "updated_at": now(), "target_values_read": False,
                    })
                    postprocess = postprocess_snapshot / "scripts/run_topic5_lbss_full_tissue_postprocess_v0_3.py"
                    run([
                        args.python, str(postprocess), "--out-root", str(selected_root),
                        "--through-target",
                    ], log_root / "selected-through-target.log")
                    atomic(out / "PRIMARY_ARTIFACT_POINTER.json", {
                        "contract": "topic5_lbss_primary_artifact_pointer_v0_4",
                        "artifact_root": str(selected_root),
                        "selected_config_id": decision["selected_config_id"],
                        "reason": "FULL_COHORT_SELECTIVE_NONLOCAL_CONFIRMED",
                        "target_values_read": True,
                        "created_at": now(),
                    })
                    atomic(out / "SPATIAL_DECISION_COMPLETE.json", {
                        **selected_decision,
                        "selected_contract": "FULL_COHORT_SELECTED_SPATIAL_CONFIG",
                        "primary_artifact_root": str(selected_root),
                        "target_values_read": True,
                    })
                    atomic(status_path, {
                        "status": "COMPLETE", "pid": os.getpid(), "updated_at": now(),
                        "artifact_root": str(selected_root), "target_values_read": True,
                    })
                    return
                atomic(out / "SPATIAL_DECISION_COMPLETE.json", {
                    **formal,
                    "selected_contract": "CURRENT_V0_3_DEVELOPMENT_GAIN_NOT_CONFIRMED",
                    "target_values_read": False,
                })
            else:
                atomic(out / "SPATIAL_DECISION_COMPLETE.json", {
                    **decision,
                    "selected_contract": "CURRENT_V0_3_NO_BETTER_CONFIGURATION_FOUND",
                    "target_values_read": False,
                })
        else:
            atomic(out / "SPATIAL_DECISION_COMPLETE.json", {
                **trigger,
                "selected_contract": "CURRENT_V0_3",
                "target_values_read": False,
            })

        # A completed target-free spatial decision is necessary but not
        # sufficient to unlock the scorer.  The base attenuation pipeline can
        # finish concurrently with the search/confirmation above; wait here
        # until every model-derived field is frozen.
        while not pretarget_marker.exists():
            target_sealed(out)
            if (out / "PIPELINE_FAILED.json").exists():
                failure = json.loads((out / "PIPELINE_FAILED.json").read_text())
                raise RuntimeError(f"pretarget postprocess failed: {failure}")
            atomic(status_path, {
                "status": "WAITING_FOR_PRETARGET_POSTPROCESS",
                "pretarget_postprocess_complete": False,
                "spatial_decision_complete": True,
                "pid": os.getpid(),
                "updated_at": now(),
                "target_values_read": False,
            })
            time.sleep(min(30, max(5, args.poll_seconds)))

        # The pretarget watcher writes its marker just before returning and
        # releases the pipeline lock in ``finally``.  Do not race that cleanup.
        while (out / "POSTPROCESS_PIPELINE.lock").exists():
            time.sleep(1)

        atomic(status_path, {
            "status": "ADVANCING_TO_FROZEN_EARLY_ICTAL_BENCHMARK",
            "pid": os.getpid(), "updated_at": now(), "target_values_read": False,
        })
        postprocess = postprocess_snapshot / "scripts/run_topic5_lbss_full_tissue_postprocess_v0_3.py"
        run([
            args.python, str(postprocess), "--out-root", str(out), "--through-target",
        ], log_root / "through-target.log")
        atomic(status_path, {
            "status": "COMPLETE", "pid": os.getpid(), "updated_at": now(),
            "target_values_read": True,
        })
    except Exception as error:
        atomic(out / "SPATIAL_DECISION_FAILED.json", {
            "status": "FAILED", "error": repr(error), "pid": os.getpid(),
            "updated_at": now(), "target_values_read": False,
        })
        raise


if __name__ == "__main__":
    main()
