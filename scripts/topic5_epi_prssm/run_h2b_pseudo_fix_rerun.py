#!/usr/bin/env python3
"""One-time corrected H2b producer rerun without touching paper-ready figures."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

from src.topic5_epi_prssm.contracts import (
    OUTPUT_ROOT, atomic_write_json, package_hash, sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
PLAN = OUTPUT_ROOT / "manifests/plans/goal3b_caliper.json"
STATUS = OUTPUT_ROOT / "manifests/H2B_PSEUDO_FIX_RERUN_STATUS.json"
ARCHIVE = OUTPUT_ROOT / "h2b_pseudo_fix_archive/pre_fix_2026-08-27"
REPORT = OUTPUT_ROOT / "h2b_pseudo_fix_rerun_2026-08-27"
LAYERS = (
    "linear_graph_recurrent", "leaky_state",
    "resource_anchored_on_best_family",
)
LEADS = (30, 60, 15, 5)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment() -> dict[str, str]:
    value = os.environ.copy()
    conda_lib = str(PYTHON.parent.parent / "lib")
    inherited_library_path = value.get("LD_LIBRARY_PATH", "")
    value.update({
        "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
        # A detached process otherwise inherits only the CUDA library path on
        # this host and pandas resolves the older system libstdc++.
        "LD_LIBRARY_PATH": (
            conda_lib
            if not inherited_library_path
            else f"{conda_lib}:{inherited_library_path}"
        ),
    })
    return value


def write_status(stage: str, **extra) -> None:
    old = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    atomic_write_json(STATUS, {
        **old, "status": "RUNNING" if stage != "complete" else "COMPLETE",
        "stage": stage, "updated_at": now(), "package_hash": package_hash(),
        "formal_test_partition_opened": False, "sealed_opened": False,
        **extra,
    })


def archive_old() -> dict:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    source = OUTPUT_ROOT / "seizure_link_preictal"
    copied = []
    patterns = (
        "H2B_PRIMARY_EVIDENCE_CARD__*.json",
        "preictal_effects__*.csv", "preictal_denominators__*.csv",
        "CALIPER_VERIFICATION.json", "DOWNSTREAM_FRESHNESS.json",
    )
    for pattern in patterns:
        for path in source.glob(pattern):
            target = ARCHIVE / path.name
            if not target.exists():
                shutil.copy2(path, target)
            copied.append(str(target))
    for directory in ("h2b_sensitivity", "h2b_denominators", "seizure_crosswalk"):
        source_dir = OUTPUT_ROOT / directory
        target_dir = ARCHIVE / directory
        if source_dir.exists() and not target_dir.exists():
            shutil.copytree(source_dir, target_dir)
        if target_dir.exists():
            copied.extend(str(path) for path in target_dir.rglob("*") if path.is_file())
    manifest = {
        "status": "COMPLETE", "archived_at": now(),
        "reason": "preserve pre-pseudo-onset-fix H2b summaries before the one-time rerun",
        "files": sorted(set(copied)),
    }
    atomic_write_json(ARCHIVE / "MANIFEST.json", manifest)
    return manifest


def run_task(task: dict, attempt: int) -> dict:
    key = task["label"].replace(":", "__")
    log = REPORT / "logs" / f"{key}.attempt{attempt}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [str(PYTHON), str(ROOT / task["script"]), *map(str, task["args"])]
    with log.open("a") as handle:
        handle.write(f"[{now()}] {' '.join(command)}\n")
        process = subprocess.run(
            command, cwd=HERE, env=environment(), stdout=handle,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True,
            start_new_session=True,
        )
    return {
        "label": task["label"], "attempt": attempt,
        "returncode": int(process.returncode), "log": str(log),
    }


def run_producer(tasks: list[dict], workers: int) -> list[dict]:
    pending = list(tasks)
    final = []
    for attempt in (1, 2):
        rows = []
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            futures = {pool.submit(run_task, task, attempt): task for task in pending}
            for number, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if number % 12 == 0 or number == len(futures):
                    write_status(
                        "producer", attempt=attempt,
                        completed_this_attempt=number,
                        tasks_this_attempt=len(futures),
                        failed_this_attempt=sum(r["returncode"] != 0 for r in rows),
                    )
        failed_labels = {row["label"] for row in rows if row["returncode"] != 0}
        final.extend(row for row in rows if row["returncode"] == 0)
        if not failed_labels:
            return final
        pending = [task for task in tasks if task["label"] in failed_labels]
    raise RuntimeError(f"H2b producer failures after retry: {[t['label'] for t in pending]}")


def step(
    label: str,
    script: str,
    *arguments: str,
    allowed_returncodes: tuple[int, ...] = (0,),
) -> dict:
    log = REPORT / "logs/downstream" / f"{label}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [str(PYTHON), str(HERE / script), *map(str, arguments)]
    with log.open("a") as handle:
        handle.write(f"[{now()}] {' '.join(command)}\n")
        process = subprocess.run(
            command, cwd=HERE, env=environment(), stdout=handle,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True,
        )
    if process.returncode not in allowed_returncodes:
        raise RuntimeError(f"H2b downstream step failed: {label}; see {log}")
    return {
        "label": label, "log": str(log),
        "returncode": int(process.returncode),
    }


def verify_caliper_step() -> dict:
    """Persist a partial-balance result without treating it as a crash.

    The standalone verifier intentionally exits non-zero unless at least 90% of
    seizures are hard-caliper matched.  That behaviour is useful as a strict
    gate for callers that require a fully balanced set.  This one-time H2b
    sensitivity rerun, however, must still rebuild the population and
    high-observability strata when the result is ``CALIPER_PARTIAL``.  The
    partial verdict is therefore recorded as a scientific limitation rather
    than converted into an engineering failure.
    """
    row = step(
        "verify_caliper", "verify_caliper_applied.py",
        allowed_returncodes=(0, 1),
    )
    path = OUTPUT_ROOT / "seizure_link_preictal/CALIPER_VERIFICATION.json"
    if not path.exists():
        raise RuntimeError("caliper verifier did not persist its evidence file")
    value = json.loads(path.read_text())
    verdict = value.get("verdict")
    if verdict not in {"CALIPER_APPLIED_AND_BALANCED", "CALIPER_PARTIAL"}:
        raise RuntimeError(
            f"H2b caliper instrument is not usable: verdict={verdict!r}"
        )
    row.update({
        "scientific_status": verdict,
        "share_with_caliper_applied": value.get("share_with_caliper_applied"),
        "evidence": str(path),
    })
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--downstream-only", action="store_true",
        help="resume after a verified 408/408 producer completion",
    )
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    tasks = plan["tasks"]
    if len(tasks) != 408 or any("--overwrite" not in task["args"] for task in tasks):
        raise ValueError("H2b corrected plan is not the frozen 408-task overwrite plan")
    REPORT.mkdir(parents=True, exist_ok=True)
    if args.downstream_only:
        prior = json.loads(STATUS.read_text()) if STATUS.exists() else {}
        if not (
            prior.get("producer_complete") == len(tasks)
            and prior.get("completed_this_attempt") == len(tasks)
            and prior.get("failed_this_attempt") == 0
        ):
            raise RuntimeError(
                "--downstream-only requires a recorded 408/408 zero-failure producer run"
            )
        archive_path = ARCHIVE / "MANIFEST.json"
        if not archive_path.exists():
            raise RuntimeError("pre-fix archive manifest is missing")
        archive = json.loads(archive_path.read_text())
        producer_count = len(tasks)
    else:
        write_status("archive", expected_producer_tasks=len(tasks))
        archive = archive_old()
        write_status("producer", archived_files=len(archive["files"]))
        producer = run_producer(tasks, args.workers)
        producer_count = len(producer)
    write_status("verify_caliper", producer_complete=producer_count)
    downstream = [verify_caliper_step()]
    write_status("aggregate_12")
    for layer in LAYERS:
        for lead in LEADS:
            downstream.append(step(
                f"aggregate_{layer}_{lead}", "aggregate_goal3b.py",
                "--layer", layer, "--lead-minutes", str(lead),
            ))
    downstream.append(step("verify_fresh", "verify_downstream_fresh.py"))
    for lead in LEADS:
        downstream.append(step(
            f"crosswalk_{lead}", "build_seizure_crosswalk.py",
            "--layer", "linear_graph_recurrent", "--lead", f"lead{lead}m",
        ))
    downstream.append(step("denominators", "build_h2b_denominators.py"))
    downstream.append(step("sensitivity", "run_h2b_sensitivity.py"))
    cards = sorted((OUTPUT_ROOT / "seizure_link_preictal").glob(
        "H2B_PRIMARY_EVIDENCE_CARD__*.json"
    ))
    summary = {
        "status": "COMPLETE", "revision": "h2b_pseudo_onset_fix_rerun_v1",
        "producer_tasks": producer_count, "aggregate_cards": len(cards),
        "cards": [{"path": str(path), "sha256": sha256_file(path)} for path in cards],
        "downstream": downstream,
        "archive_manifest": str(ARCHIVE / "MANIFEST.json"),
        "paper_ready_figures_touched": False,
        "h3b_run": False,
        "formal_test_partition_opened": False, "sealed_opened": False,
    }
    atomic_write_json(REPORT / "SUMMARY.json", summary)
    write_status("complete", summary=str(REPORT / "SUMMARY.json"), aggregate_cards=len(cards))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
