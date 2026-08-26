#!/usr/bin/env python3
"""Recoverable, memory-bounded queue for the support-selected very-long H3 screen."""
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


PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
REVISION = "t2_very_long_support_selected_multiseed_v1"
SUBJECT_WINDOWS = {
    "yuquan_chengshuai": (
        "event_count_10000", "event_count_15000", "physical_6h",
    ),
    "yuquan_pengzihang": ("event_count_5000",),
    "epilepsiae_922": ("event_count_3000",),
    "yuquan_chenziyang": (
        "event_count_3000", "event_count_4000", "physical_6h",
    ),
    "yuquan_hanyuxuan": ("event_count_2000", "physical_6h"),
}
SEEDS = tuple(range(7))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text())
    except Exception:
        return False
    return value.get("status") == "COMPLETE" and value.get("sealed_opened") is False


def t1_scientifically_usable(value: dict) -> bool:
    contrasts = value.get("contrasts", {})
    return bool(
        int(value.get("selected_epochs", 0)) > 0
        and contrasts.get("filtered_minus_no_state_joint_nll", 0.0) < 0.0
        and contrasts.get(
            "filtered_minus_validation_correction_off_joint_nll", 0.0
        ) < 0.0
    )


def available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0 / 1024.0
    return 0.0


def gpu_free_mib() -> float:
    command = [
        "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
    ]
    try:
        values = subprocess.check_output(command, text=True).splitlines()
        return min(float(value.strip()) for value in values)
    except Exception:
        return 0.0


def wait_for_resources(*, min_ram_gib: float, min_gpu_mib: float) -> None:
    while available_gib() < min_ram_gib or gpu_free_mib() < min_gpu_mib:
        time.sleep(20.0)


def run(command: list[str], log: Path, *, min_ram_gib: float,
        min_gpu_mib: float) -> dict:
    wait_for_resources(min_ram_gib=min_ram_gib, min_gpu_mib=min_gpu_mib)
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": str(contract.REPO_ROOT),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUDA_MODULE_LOADING": "LAZY",
        "LD_LIBRARY_PATH": (
            "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib:"
            + environment.get("LD_LIBRARY_PATH", "")
        ),
    })
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
        "command": command, "log": str(log), "started": started,
        "finished": now(), "returncode": int(process.returncode),
    }


def prepare_subject(subject: str, root: Path) -> dict:
    log_root = root / "logs/preparation"
    r1_2 = contract.RESULT_ROOT / "r1_2"
    tasks = [
        (
            r1_2 / "baselines" / subject / "seed_0/result.json",
            [str(PYTHON), "scripts/topic5_continuous_marked_state_r1/run_r1_2_baseline.py",
             "--subject", subject, "--seed", "0", "--device", "cuda",
             "--mark-batch-size", "512"],
            log_root / f"{subject}_baseline.log",
        ),
        (
            r1_2 / "bridge_e1" / subject / "seed_0/result.json",
            [str(PYTHON), "scripts/topic5_continuous_marked_state_r1/run_r1_2_bridge.py",
             "--subject", subject, "--seed", "0", "--device", "cuda",
             "--anchor-batch-size", "2", "--max-train-anchors", "64",
             "--max-validation-anchors", "32"],
            log_root / f"{subject}_bridge.log",
        ),
        (
            r1_2 / "cache" / subject / "manifest.json",
            [str(PYTHON), "scripts/topic5_continuous_marked_state_r1/run_r1_2_cache.py",
             "--subject", subject, "--device", "cuda",
             "--anchor-batch-size", "4"],
            log_root / f"{subject}_cache.log",
        ),
    ]
    history = []
    for output, command, log in tasks:
        if complete(output):
            history.append({"output": str(output), "skipped_complete": True})
            continue
        value = run(command, log, min_ram_gib=72.0, min_gpu_mib=6000.0)
        value["output"] = str(output)
        history.append(value)
        if value["returncode"] != 0 or not complete(output):
            return {"subject": subject, "status": "FAIL", "history": history}
    return {"subject": subject, "status": "COMPLETE", "history": history}


def fit_t1(subject: str, seed: int, root: Path) -> dict:
    output = (
        contract.RESULT_ROOT / "r1_2/t1_full" / subject
        / f"explicit_d8_seed_{seed}/result.json"
    )
    if complete(output):
        return {
            "subject": subject, "seed": seed, "status": "COMPLETE",
            "skipped_complete": True, "output": str(output),
        }
    command = [
        str(PYTHON), "scripts/topic5_continuous_marked_state_r1/run_r1_2_t1.py",
        "--subject", subject, "--arm", "explicit", "--seed", str(seed),
        "--device", "cuda", "--epochs", "4", "--chunk-anchors", "128",
    ]
    value = run(
        command, root / "logs/t1" / f"{subject}_seed_{seed}.log",
        min_ram_gib=64.0, min_gpu_mib=4500.0,
    )
    value.update({"subject": subject, "seed": seed, "output": str(output)})
    value["status"] = (
        "COMPLETE" if value["returncode"] == 0 and complete(output) else "FAIL"
    )
    return value


def fit_h3(subject: str, seed: int, window: str, root: Path,
           exposure_memory: str) -> dict:
    output = root / "human" / subject / window / f"seed_{seed}/result.json"
    if complete(output):
        return {
            "subject": subject, "seed": seed, "window": window,
            "status": "COMPLETE", "skipped_complete": True,
            "output": str(output),
        }
    command = [
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_r1/run_t2_long_total_human.py",
        "--subject", subject, "--seed", str(seed), "--window", window,
        "--t1-source", "r1_2", "--device", "cuda",
        "--exposure-memory", exposure_memory,
        "--t1-root", str(contract.RESULT_ROOT / "r1_2"),
        "--output-root", str(root / "human"),
    ]
    value = run(
        command, root / "logs/h3" / f"{subject}_{window}_seed_{seed}.log",
        min_ram_gib=64.0, min_gpu_mib=4000.0,
    )
    value.update({
        "subject": subject, "seed": seed, "window": window,
        "exposure_memory": exposure_memory,
        "output": str(output),
    })
    value["status"] = (
        "COMPLETE" if value["returncode"] == 0 and complete(output) else "FAIL"
    )
    return value


def parallel(function, tasks: list[tuple], workers: int) -> list[dict]:
    rows = []
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        future = {pool.submit(function, *task): task for task in tasks}
        for item in as_completed(future):
            try:
                rows.append(item.result())
            except Exception as error:
                rows.append({"task": list(future[item]), "status": "FAIL",
                             "error": repr(error)})
    return rows


def summarise(root: Path, preparation: list[dict], t1_jobs: list[dict],
              h3_jobs: list[dict], *, exposure_memory: str) -> dict:
    t1 = []
    for subject in SUBJECT_WINDOWS:
        for seed in SEEDS:
            path = (
                contract.RESULT_ROOT / "r1_2/t1_full" / subject
                / f"explicit_d8_seed_{seed}/result.json"
            )
            if not complete(path):
                continue
            value = json.loads(path.read_text())
            t1.append({
                "subject": subject, "seed": seed,
                "selected_epochs": int(value["selected_epochs"]),
                "filtered_minus_no_state_joint_nll": value["contrasts"][
                    "filtered_minus_no_state_joint_nll"
                ],
                "matched_correct_minus_wrong_joint_nll": value["contrasts"][
                    "matched_filtered_minus_wrong_time_joint_nll"
                ],
                "admissible_for_h3": t1_scientifically_usable(value),
                "trained_above_epoch_zero": bool(int(value["selected_epochs"]) > 0),
                "result": str(path),
            })
    h3 = []
    for subject, windows in SUBJECT_WINDOWS.items():
        for window in windows:
            for seed in SEEDS:
                path = root / "human" / subject / window / f"seed_{seed}/result.json"
                if not complete(path):
                    continue
                value = json.loads(path.read_text())
                # Re-derive against the current T1 rule instead of trusting the
                # flag the H3 artifact was written with.  Artifacts produced
                # before the rule was tightened carry a looser flag, and a
                # superseded "admissible" sitting next to a favourable number is
                # exactly how a withdrawn result gets read as a finding.
                t1_path = (
                    contract.RESULT_ROOT / "r1_2/t1_full" / subject
                    / f"explicit_d8_seed_{seed}/result.json"
                )
                recorded = value["instrument_admissibility"][
                    "human_biological_contrasts_admissible"
                ]
                admissible = bool(
                    recorded
                    and complete(t1_path)
                    and t1_scientifically_usable(json.loads(t1_path.read_text()))
                )
                contrast = value["contrasts"]["real_minus_intercept_matched"][
                    "decoder_total_equal_block_mse"
                ]
                delayed = value["contrasts"]["real_minus_causal_delayed"][
                    "decoder_total_equal_block_mse"
                ]
                scale = value["effective_exposure_time_scale"]
                h3.append({
                    "subject": subject, "window": window, "seed": seed,
                    "admissible": bool(admissible),
                    "admissible_flag_in_artifact": bool(recorded),
                    "artifact_revision": value.get("revision"),
                    "real_minus_intercept_decoder_mse": float(contrast),
                    "real_minus_delayed_decoder_mse": float(delayed),
                    "median_window_hours": value["denominators"][
                        "median_window_hours_validation"
                    ],
                    "median_events": value["denominators"][
                        "median_events_per_window_validation"
                    ],
                    "slowest_generator_minutes": scale[
                        "slowest_mode_time_constant_minutes"
                    ],
                    "hours_holding_ninety_percent_weight": scale[
                        "median_hours_holding_ninety_percent_weight"
                    ],
                    "result": str(path),
                })
    status = (
        "COMPLETE" if all(row.get("status") == "COMPLETE" for row in (
            preparation + t1_jobs + h3_jobs
        )) else "COMPLETE_WITH_EXPLORATORY_FAILURES"
    )
    payload = {
        "status": status, "revision": REVISION, "generated_at": now(),
        "exposure_memory": exposure_memory,
        "subjects": list(SUBJECT_WINDOWS), "seeds": list(SEEDS),
        "subject_windows": {key: list(value) for key, value in SUBJECT_WINDOWS.items()},
        "resource_policy": {
            "preparation_workers": 2, "t1_workers": 4, "h3_workers": 3,
            "threads_per_worker": 1, "minimum_available_ram_gib": 64,
            "resume_by_complete_result": True,
        },
        "preparation_jobs": preparation, "t1_jobs": t1_jobs,
        "h3_jobs": h3_jobs, "t1_results": t1, "h3_results": h3,
        "formal_test_partition_opened": False, "sealed_opened": False,
        "claim_boundary": (
            "support-selected multi-seed development exploration; no pooled "
            "patient p-value and no causal network-shaping claim"
        ),
    }
    contract.atomic_json(root / "summary.json", payload)
    lines = [
        "# 超长尺度 H3 探索报告", "", f"状态：{status}", "",
        "这轮只问：在同一段连续记录里，几千到两万次 IED 的累计历史，是否能在一个已经学动的前状态模型上解释后续状态变化。患者按运行前的长序列支持度选择，未打开正式检验分区。",
        "", "## 前状态仪器", "",
    ]
    for subject in SUBJECT_WINDOWS:
        rows = [row for row in t1 if row["subject"] == subject]
        moved = sum(row["trained_above_epoch_zero"] for row in rows)
        valid = sum(row["admissible_for_h3"] for row in rows)
        lines.append(
            f"- {subject}: {moved}/{len(rows)} 个 seed 训练动了（离开 epoch 0），"
            f"其中 {valid}/{len(rows)} 个同时在外层验证上有预测力且跨窗口记忆有利。"
        )
    lines += ["", "## 长尺度结果", ""]
    for subject, windows in SUBJECT_WINDOWS.items():
        for window in windows:
            rows = [
                row for row in h3
                if row["subject"] == subject and row["window"] == window
                and row["admissible"]
            ]
            if not rows:
                lines.append(f"- {subject} / {window}: 没有合格前状态 seed，不能作生物学判断。")
                continue
            primary = np.asarray(
                [row["real_minus_intercept_decoder_mse"] for row in rows]
            )
            delayed = np.asarray(
                [row["real_minus_delayed_decoder_mse"] for row in rows]
            )
            effective = np.asarray(
                [row["hours_holding_ninety_percent_weight"] for row in rows]
            )
            lines.append(
                f"- {subject} / {window}: {len(rows)} 个合格 seed；真实累计相对拟合截距对照的中位差 "
                f"{np.median(primary):+.4g}，相对延迟对照 {np.median(delayed):+.4g}；"
                f"90% 实际权重落在最近约 {np.median(effective):.2f} 小时。"
            )
    lines += [
        "", "## 解释边界", "",
        "负数表示真实累计历史的预测误差更低。重叠窗口不当作独立样本；多 seed 是稳定性检查，不是患者数。名义上的 20,000 次只有在生成器实际保留了足够久的权重时才算真正长尺度。",
    ]
    (root / "REPORT_PLAIN.md").write_text("\n".join(lines) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path,
        default=contract.RESULT_ROOT / "t2_very_long_discovery",
    )
    parser.add_argument("--prep-workers", type=int, default=2)
    parser.add_argument("--t1-workers", type=int, default=4)
    parser.add_argument("--h3-workers", type=int, default=3)
    parser.add_argument(
        "--exposure-memory", choices=("generator_weighted", "boxcar"),
        default="generator_weighted",
    )
    parser.add_argument("--wait-for-status", type=Path, default=None)
    args = parser.parse_args()
    if args.wait_for_status is not None:
        while True:
            try:
                upstream = json.loads(args.wait_for_status.read_text())
            except Exception:
                upstream = {"status": "RUNNING"}
            if upstream.get("status") != "RUNNING":
                break
            time.sleep(30.0)
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    contract.atomic_json(root / "STATUS.json", {
        "status": "RUNNING", "revision": REVISION, "started_at": now(),
        "subjects": list(SUBJECT_WINDOWS), "seeds": list(SEEDS),
        "exposure_memory": args.exposure_memory,
        "formal_test_partition_opened": False, "sealed_opened": False,
    })
    preparation = parallel(
        prepare_subject,
        [(subject, root) for subject in SUBJECT_WINDOWS],
        max(1, min(int(args.prep_workers), 2)),
    )
    prepared = {
        row.get("subject") for row in preparation if row.get("status") == "COMPLETE"
    }
    t1_jobs = parallel(
        fit_t1,
        [(subject, seed, root) for subject in SUBJECT_WINDOWS for seed in SEEDS
         if subject in prepared],
        max(1, min(int(args.t1_workers), 4)),
    )
    valid = []
    for subject in SUBJECT_WINDOWS:
        for seed in SEEDS:
            path = (
                contract.RESULT_ROOT / "r1_2/t1_full" / subject
                / f"explicit_d8_seed_{seed}/result.json"
            )
            if not complete(path):
                continue
            value = json.loads(path.read_text())
            if t1_scientifically_usable(value):
                valid.append((subject, seed))
    h3_jobs = parallel(
        fit_h3,
        [(subject, seed, window, root, args.exposure_memory)
         for subject, seed in valid
         for window in SUBJECT_WINDOWS[subject]],
        max(1, min(int(args.h3_workers), 3)),
    )
    payload = summarise(
        root, preparation, t1_jobs, h3_jobs,
        exposure_memory=args.exposure_memory,
    )
    contract.atomic_json(root / "STATUS.json", {
        "status": payload["status"], "revision": REVISION,
        "finished_at": now(), "summary": str(root / "summary.json"),
        "report": str(root / "REPORT_PLAIN.md"),
        "exposure_memory": args.exposure_memory,
        "formal_test_partition_opened": False, "sealed_opened": False,
    })


if __name__ == "__main__":
    main()
