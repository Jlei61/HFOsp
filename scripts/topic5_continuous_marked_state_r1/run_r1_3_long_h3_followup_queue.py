#!/usr/bin/env python3
"""Wait for long-subject T1 triage, then run only support-qualified H3 arms."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.coverage import CoverageTable


PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
REVISION = "r1_3_long_t1_qualified_boxcar_h3_v2_full_contrast_support"
CANDIDATE_N = (1000, 2000, 3000, 4000, 5000, 10000, 15000)
MIN_NONOVERLAP = 3
DELAY_EVENTS = 1000


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        value = load(path)
    except Exception:
        return False
    return value.get("status") == "COMPLETE" and value.get("sealed_opened") is False


def atomic_status(path: Path, payload: dict) -> None:
    contract.atomic_json(path, {
        **payload,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "updated_at": now(),
    })


def wait_for_t1(root: Path, status_path: Path) -> dict:
    while True:
        if status_path.exists():
            status = load(status_path)
            if status.get("status") != "RUNNING":
                return status
        time.sleep(30.0)


def greedy_nonoverlap(time_value: np.ndarray, start: np.ndarray,
                      end: np.ndarray, rows: np.ndarray) -> int:
    order = rows[np.argsort(time_value[end[rows]], kind="stable")]
    last = -np.inf
    count = 0
    for row in order:
        if time_value[start[row]] >= last:
            count += 1
            last = float(time_value[end[row]])
    return count


def support_for(subject: str) -> dict:
    """Independent-window budget on the partition the H3 design actually uses.

    ``build_long_window_design`` groups events by *recorded coverage segment*
    (``_event_coverage_segment``), not by ``event_session``.  A session is a
    continuity label that survives ordinary metadata gaps, so it is strictly
    coarser: epilepsiae_922 has 49 segments inside 11 sessions, 958 has 135
    inside 9.  Counting candidate windows per session therefore admits windows
    that straddle unrecorded time and that the design will never build, which
    inflates the non-overlap budget and can qualify a patient whose real design
    yields no window at that N at all.
    """
    path = contract.RESULT_ROOT / "r1_2/cache" / subject / "full_design.npz"
    value = np.load(path)
    event_time = np.asarray(value["event_time"], dtype=np.float64)
    event_split = np.asarray(value["event_split"], dtype=np.int8)
    coverage = CoverageTable.load(
        contract.RESULT_ROOT / "r1_2/coverage" / f"{subject}.npz"
    )
    event_segment = np.searchsorted(
        coverage.stop, event_time, side="right"
    ).astype(np.int64)
    n_session = int(len(np.unique(np.asarray(value["event_session"]))))
    rows = []
    for scale in CANDIDATE_N:
        real_start, support_start, end = [], [], []
        for label in np.unique(event_segment):
            index = np.flatnonzero(event_segment == label)
            for local in range(int(scale) + DELAY_EVENTS, len(index)):
                real_start.append(int(index[local - int(scale)]))
                support_start.append(
                    int(index[local - int(scale) - DELAY_EVENTS])
                )
                end.append(int(index[local]))
        if not real_start:
            continue
        real_start_array = np.asarray(real_start, dtype=np.int64)
        support_start_array = np.asarray(support_start, dtype=np.int64)
        end_array = np.asarray(end, dtype=np.int64)
        split = event_split[end_array]
        row = {
            "scale_events": int(scale),
            "causal_delay_events": DELAY_EVENTS,
            "full_instrument_support_events": int(scale) + DELAY_EVENTS,
        }
        for name, code in (("train", 0), ("validation", 1)):
            take = np.flatnonzero(split == code)
            row[name] = {
                "windows": int(len(take)),
                "nonoverlapping_real_exposure_windows": greedy_nonoverlap(
                    event_time, real_start_array, end_array, take
                ),
                "nonoverlapping_full_windows": greedy_nonoverlap(
                    event_time, support_start_array, end_array, take
                ),
                "median_real_exposure_hours": (
                    float(np.median(
                        (event_time[end_array[take]]
                         - event_time[real_start_array[take]])
                        / 3600.0
                    )) if len(take) else None
                ),
                "median_full_instrument_hours": (
                    float(np.median(
                        (event_time[end_array[take]]
                         - event_time[support_start_array[take]])
                        / 3600.0
                    )) if len(take) else None
                ),
            }
        rows.append(row)
    qualified = [
        row for row in rows
        if row["train"]["nonoverlapping_full_windows"] >= MIN_NONOVERLAP
        and row["validation"]["nonoverlapping_full_windows"] >= MIN_NONOVERLAP
    ]
    chosen = max(qualified, key=lambda row: row["scale_events"]) if qualified else None
    return {
        "subject": subject,
        "partition": "recorded_coverage_segment",
        "n_recorded_segments": int(len(np.unique(event_segment))),
        "n_event_sessions": n_session,
        "partition_note": (
            "windows are counted on the recorded coverage segments the H3 "
            "design builds on; event_session is coarser and would admit "
            "windows that cross unrecorded time"
        ),
        "candidate_windows": rows,
        "minimum_nonoverlapping_each_split": MIN_NONOVERLAP,
        "chosen": chosen,
        "design": str(path),
        "design_sha256": contract.sha256_file(path),
    }


def available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0 / 1024.0
    return 0.0


def gpu_free_mib() -> float:
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ], text=True)
        return min(float(row.strip()) for row in output.splitlines())
    except Exception:
        return 0.0


def run_h3(subject: str, seed: int, scale: int, source: str,
           t1_root: Path, root: Path) -> dict:
    output = root / "human" / subject / f"event_count_{scale}"
    if source == "repertoire":
        output = output / source
    output = output / f"seed_{seed}" / "result.json"
    if complete(output):
        return {"subject": subject, "seed": seed, "source": source,
                "status": "COMPLETE", "skipped": True, "output": str(output)}
    while available_gib() < 48.0 or gpu_free_mib() < 9000.0:
        time.sleep(20.0)
    log = root / "logs" / f"{subject}_n{scale}_{source}_seed_{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": str(contract.REPO_ROOT),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "CUDA_MODULE_LOADING": "LAZY",
        "LD_LIBRARY_PATH": (
            "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib:"
            + environment.get("LD_LIBRARY_PATH", "")
        ),
    })
    command = [
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_r1/run_t2_long_total_human.py",
        "--subject", subject, "--seed", str(seed),
        "--window", f"event_count_{scale}",
        "--t1-source", "r1_3", "--t1-root", str(t1_root),
        "--exposure-memory", "boxcar", "--exposure-source", source,
        "--output-root", str(root / "human"), "--device", "cuda",
    ]
    started = now()
    with log.open("a") as handle:
        handle.write(f"\n[{started}] {' '.join(command)}\n")
        handle.flush()
        process = subprocess.run(
            command, cwd=contract.REPO_ROOT, env=environment,
            stdout=handle, stderr=subprocess.STDOUT, text=True,
            start_new_session=True,
        )
    return {
        "subject": subject, "seed": seed, "source": source,
        "scale": scale, "output": str(output), "log": str(log),
        "started": started, "finished": now(),
        "returncode": int(process.returncode),
        "status": (
            "COMPLETE" if process.returncode == 0 and complete(output) else "FAIL"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--t1-root", type=Path,
        default=contract.RESULT_ROOT / "r1_3_long_t1_triage",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=contract.RESULT_ROOT / "r1_3_long_h3_followup",
    )
    args = parser.parse_args()
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    atomic_status(root / "STATUS.json", {"status": "RUNNING", "stage": "wait_t1"})
    t1_status = wait_for_t1(root, args.t1_root / "STATUS.json")
    if t1_status.get("status") != "COMPLETE":
        atomic_status(root / "STATUS.json", {
            "status": "COMPLETE", "stage": "no_h3_t1_incomplete",
            "t1_status": t1_status,
        })
        return
    t1_summary = load(args.t1_root / "summary.json")
    support = {
        subject: support_for(subject) for subject in t1_summary["subjects"]
    }
    contract.atomic_json(root / "support_audit.json", {
        "revision": REVISION,
        "support": support,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
    })
    tasks = []
    for subject in t1_summary["subjects"]:
        chosen = support[subject]["chosen"]
        if chosen is None:
            continue
        # All three pre-registered T1 criteria, not two.  The published
        # eligibility rule is "most seeds selected a target-aligned epoch, most
        # seeds show persistent beats memoryless, and most seeds show correct
        # time beats matched wrong time"; dropping the third made the
        # implemented gate looser than the one the plan states.
        eligible = [
            row for row in t1_summary["rows"]
            if row["subject"] == subject
            and row["target_alignment_selected"]
            and row["persistent_memory_supported"]
            and row["time_specific_supported"]
        ]
        if len(eligible) < 2:
            continue
        for row in eligible:
            for source in ("load", "repertoire"):
                tasks.append((
                    subject, int(row["seed"]), int(chosen["scale_events"]),
                    source, args.t1_root, root,
                ))
    atomic_status(root / "STATUS.json", {
        "status": "RUNNING", "stage": "h3",
        "scheduled_h3_jobs": len(tasks),
    })
    jobs = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(run_h3, *task): task for task in tasks}
        for future in as_completed(futures):
            try:
                jobs.append(future.result())
            except Exception as error:
                jobs.append({
                    "task": list(map(str, futures[future])), "status": "FAIL",
                    "error": repr(error),
                })
    results = []
    for job in jobs:
        path = Path(job.get("output", ""))
        if job.get("status") != "COMPLETE" or not path.exists():
            continue
        value = load(path)
        results.append({
            "subject": value["subject"], "seed": value["seed"],
            "window": value["window_kind"],
            "exposure_source": value["exposure_source"],
            "admissible": value["instrument_admissibility"][
                "human_biological_contrasts_admissible"
            ],
            "real_minus_intercept": value["contrasts"][
                "real_minus_intercept_matched"
            ]["decoder_total_equal_block_mse"],
            "real_minus_delayed": value["contrasts"][
                "real_minus_causal_delayed"
            ]["decoder_total_equal_block_mse"],
            "nonoverlap": value["whole_window_nonoverlap_support"],
            "result": str(path),
        })
    status = "COMPLETE" if all(
        row.get("status") == "COMPLETE" for row in jobs
    ) else "COMPLETE_WITH_FAILURES"
    summary = {
        "status": status, "revision": REVISION,
        "t1_summary": str(args.t1_root / "summary.json"),
        "support_audit": str(root / "support_audit.json"),
        "minimum_nonoverlap_each_split": MIN_NONOVERLAP,
        "scheduled_jobs": len(tasks), "jobs": jobs, "results": results,
        "formal_test_partition_opened": False, "sealed_opened": False,
        "claim_boundary": (
            "development support-qualified H3 exploration; ordinary negative "
            "results do not exclude longer or different exposure mechanisms"
        ),
    }
    contract.atomic_json(root / "summary.json", summary)
    atomic_status(root / "STATUS.json", {
        "status": status, "stage": "complete",
        "scheduled_h3_jobs": len(tasks),
        "completed_h3_jobs": len(results),
        "summary": str(root / "summary.json"),
    })
    print(json.dumps({
        "status": status, "scheduled": len(tasks),
        "completed": len(results), "root": str(root),
    }, indent=2))


if __name__ == "__main__":
    main()
