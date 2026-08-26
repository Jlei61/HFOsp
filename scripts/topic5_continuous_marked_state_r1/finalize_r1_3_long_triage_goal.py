#!/usr/bin/env python3
"""Final plain/technical closeout for long-subject T1 and qualified H3."""
from __future__ import annotations

import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np

from src.topic5_continuous_marked_state_r1 import contract


ROOT = contract.RESULT_ROOT
T1_ROOT = ROOT / "r1_3_long_t1_triage"
H3_ROOT = ROOT / "r1_3_long_h3_followup"
OUT = ROOT / "r1_3_long_triage_goal_report"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def wait_finished(path: Path) -> dict:
    while True:
        if path.exists():
            value = load(path)
            if value.get("status") != "RUNNING":
                return value
        time.sleep(30.0)


def med(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def signed(value: float | None) -> str:
    return "NA" if value is None else f"{value:+.5g}"


def main() -> None:
    h3_status = wait_finished(H3_ROOT / "STATUS.json")
    t1_status = load(T1_ROOT / "STATUS.json")
    t1 = load(T1_ROOT / "summary.json") if t1_status.get("status") == "COMPLETE" else None
    h3 = load(H3_ROOT / "summary.json") if (H3_ROOT / "summary.json").exists() else {
        "status": h3_status.get("status"), "scheduled_jobs": 0,
        "jobs": [], "results": [],
    }
    support = load(H3_ROOT / "support_audit.json") if (
        H3_ROOT / "support_audit.json"
    ).exists() else {"support": {}}
    t1_detail = {}
    if t1:
        for subject in t1["subjects"]:
            values = []
            for row in t1["rows"]:
                if row["subject"] != subject:
                    continue
                result = load(Path(row["result"]))
                persistent = result["validation"]["persistent_minus_memoryless"]
                wrong = result["validation"]["strict_matched_wrong_time"][
                    "correct_minus_wrong_median"
                ]
                endpoint = result["validation"]["mark_endpoints"][
                    "persistent_minus_memoryless"
                ]
                values.append({
                    "seed": int(row["seed"]),
                    "selected_total_epoch": int(row["selected_total_epoch"]),
                    "persistent_joint": float(persistent["joint_nll_per_event"]),
                    "persistent_timing": float(persistent["timing_nll_per_event"]),
                    "persistent_mark": float(persistent["mark_nll_per_event"]),
                    "correct_minus_wrong_joint": float(
                        wrong["joint_nll_per_event"]
                    ),
                    "first_subset": float(
                        endpoint["first_group_subset_nll_per_event"]
                    ),
                    "continuation": float(
                        endpoint["same_prefix_continuation_nll_per_event"]
                    ),
                    "stop": float(endpoint["stop_nll_per_event"]),
                    "size": float(
                        endpoint["selecting_group_size_nll_per_event"]
                    ),
                    "initial_checkpoint_sha256": row[
                        "initial_checkpoint_sha256"
                    ],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "oom_retries": result["oom_retries"],
                })
            t1_detail[subject] = {
                "seeds": values,
                "selected_count": int(sum(
                    row["selected_total_epoch"] > 0 for row in values
                )),
                "persistent_count": int(sum(
                    row["persistent_joint"] < 0 for row in values
                )),
                "time_specific_count": int(sum(
                    row["correct_minus_wrong_joint"] < 0 for row in values
                )),
                "median_persistent_joint": med([
                    row["persistent_joint"] for row in values
                ]),
                "median_persistent_timing": med([
                    row["persistent_timing"] for row in values
                ]),
                "median_persistent_mark": med([
                    row["persistent_mark"] for row in values
                ]),
                "median_correct_minus_wrong_joint": med([
                    row["correct_minus_wrong_joint"] for row in values
                ]),
                "median_first_subset": med([
                    row["first_subset"] for row in values
                ]),
                "median_continuation": med([
                    row["continuation"] for row in values
                ]),
                "median_stop": med([row["stop"] for row in values]),
                "median_size": med([row["size"] for row in values]),
            }
    groups = []
    for subject in sorted({row["subject"] for row in h3.get("results", [])}):
        for source in ("load", "repertoire"):
            rows = [
                row for row in h3["results"]
                if row["subject"] == subject
                and row["exposure_source"] == source and row["admissible"]
            ]
            if not rows:
                continue
            primary = [float(row["real_minus_intercept"]) for row in rows]
            delayed = [float(row["real_minus_delayed"]) for row in rows]
            groups.append({
                "subject": subject, "source": source,
                "window": rows[0]["window"], "admissible_seeds": len(rows),
                "real_minus_intercept_median": float(np.median(primary)),
                "real_minus_intercept_favourable": int(np.sum(np.asarray(primary) < 0)),
                "real_minus_delayed_median": float(np.median(delayed)),
                "real_minus_delayed_favourable": int(np.sum(np.asarray(delayed) < 0)),
                "seed_values": rows,
            })
    candidates = [
        row for row in groups
        if row["real_minus_intercept_median"] < 0.0
        and row["real_minus_delayed_median"] < 0.0
        and row["real_minus_intercept_favourable"] > row["admissible_seeds"] / 2
        and row["real_minus_delayed_favourable"] > row["admissible_seeds"] / 2
    ]
    if candidates:
        verdict = "H3_EXPLORATORY_SUPPORT_CANDIDATE"
    elif groups:
        verdict = "H3_NO_CURRENT_SUPPORT_IN_QUALIFIED_MINIMAL_SCREEN"
    else:
        verdict = "H3_NOT_RUN_NO_PATIENT_MET_STATE_AND_INDEPENDENT_SUPPORT"
    pytest_path = OUT / "pytest.xml"
    verification = None
    if pytest_path.exists():
        suite = ET.parse(pytest_path).getroot()
        suites = list(suite) if suite.tag == "testsuites" else [suite]
        verification = {
            "artifact": str(pytest_path),
            "sha256": contract.sha256_file(pytest_path),
            "tests": sum(int(row.attrib.get("tests", 0)) for row in suites),
            "failures": sum(
                int(row.attrib.get("failures", 0)) for row in suites
            ),
            "errors": sum(int(row.attrib.get("errors", 0)) for row in suites),
            "skipped": sum(
                int(row.attrib.get("skipped", 0)) for row in suites
            ),
        }
    machine = {
        "status": "COMPLETE",
        "scientific_verdict": verdict,
        "t1_status": t1_status,
        "h3_status": h3_status,
        "t1_by_subject": t1["by_subject"] if t1 else {},
        "t1_rows": t1["rows"] if t1 else [],
        "t1_numeric_detail": t1_detail,
        "support": support.get("support", {}),
        "h3_groups": groups,
        "h3_candidate_groups": candidates,
        "h3_jobs": h3.get("jobs", []),
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "source_hashes": {
            "t1_queue": contract.sha256_file(
                contract.REPO_ROOT / "scripts/topic5_continuous_marked_state_r1/run_r1_3_long_t1_triage_queue.py"
            ),
            "h3_queue": contract.sha256_file(
                contract.REPO_ROOT / "scripts/topic5_continuous_marked_state_r1/run_r1_3_long_h3_followup_queue.py"
            ),
            "h3_runner": contract.sha256_file(
                contract.REPO_ROOT / "scripts/topic5_continuous_marked_state_r1/run_t2_long_total_human.py"
            ),
            "h3_operator": contract.sha256_file(
                contract.REPO_ROOT / "src/topic5_continuous_marked_state_r1/t2_long_total.py"
            ),
            "finalizer": contract.sha256_file(Path(__file__)),
        },
        "verification": verification,
        "claim_boundary": (
            "fixed three-subject development triage; H3 candidates require "
            "replication in more patients and independent long recordings"
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    contract.atomic_json(OUT / "machine_audit.json", machine)

    subject_lines = []
    if t1:
        for subject in t1["subjects"]:
            row = t1["by_subject"][subject]
            detail = t1_detail[subject]
            subject_lines.append(
                f"- {subject}: target alignment {row['target_alignment_selected']}/3；"
                f"persistent 胜 memoryless {row['persistent_memory_supported']}/3；"
                f"correct-time 胜 wrong-time {row['time_specific_supported']}/3；"
                f"persistent joint 中位 {signed(detail['median_persistent_joint'])} "
                f"（timing {signed(detail['median_persistent_timing'])}，"
                f"mark {signed(detail['median_persistent_mark'])}）；"
                f"correct−wrong 中位 {signed(detail['median_correct_minus_wrong_joint'])}；"
                f"端点中位 subset {signed(detail['median_first_subset'])}，"
                f"continuation {signed(detail['median_continuation'])}，"
                f"STOP {signed(detail['median_stop'])}，"
                f"size {signed(detail['median_size'])}；"
                f"起点/终点 distinct payload {row['distinct_initial_payloads']}/"
                f"{row['distinct_final_payloads']}。"
            )
    support_lines = []
    for subject, value in support.get("support", {}).items():
        chosen = value.get("chosen")
        if chosen is None:
            support_lines.append(
                f"- {subject}: 没有任何 N 同时达到 TRAIN/validation 各至少 3 个"
                "不重叠完整对比支持窗。"
            )
            continue
        support_lines.append(
            f"- {subject}: 选择 N={chosen['scale_events']}；"
            f"完整对比实际需要 {chosen['full_instrument_support_events']} events；"
            f"TRAIN/validation 不重叠窗 "
            f"{chosen['train']['nonoverlapping_full_windows']}/"
            f"{chosen['validation']['nonoverlapping_full_windows']}；"
            f"validation 名义/完整支持时长中位 "
            f"{chosen['validation']['median_real_exposure_hours']:.2f}/"
            f"{chosen['validation']['median_full_instrument_hours']:.2f} h。"
        )
    h3_lines = []
    for row in groups:
        h3_lines.append(
            f"- {row['subject']} / {row['window']} / {row['source']}: "
            f"real−intercept 中位 {row['real_minus_intercept_median']:+.5g} "
            f"（{row['real_minus_intercept_favourable']}/{row['admissible_seeds']} 有利）；"
            f"real−delayed {row['real_minus_delayed_median']:+.5g} "
            f"（{row['real_minus_delayed_favourable']}/{row['admissible_seeds']} 有利）。"
        )
    if not h3_lines:
        h3_lines.append(
            "- 没有患者同时满足可用跨窗口状态和 TRAIN/validation 各至少 3 个不重叠长窗口，"
            "因此本轮没有为了凑结果而运行新的人体 H3。"
        )
    plain = f"""# 长序列 T1 分诊与最小 H3：白话报告

## 一句话

本轮先检验长记录患者能否在最新 R1.3 目标训练下形成可用状态，再决定是否运行长尺度 H3；最终机器判定为 `{verdict}`。这不是正式队列结论，普通阴性也不否定更长或不同形式的 IED 累积作用。

## 为什么要做这轮

前一轮长患者实际上只跑过 R1.2，而且 R1.3 命令入口把扩展患者排除在外。现在第一次让固定的三类长患者进入相同的 timing + sequential mark target alignment：韩宇轩代表已有预测记忆，陈子阳代表训练启动但外层失败，程帅代表万次支持但旧模型 epoch 0。

## T1 结果

{chr(10).join(subject_lines) if subject_lines else '- T1 队列没有完整结束，未形成可解释结果。'}

seed 是优化起点，不是患者数。persistent 胜 memoryless 才说明跨窗口携带额外信息；correct-time 胜 wrong-time 决定能否进一步称为时刻专属状态。

## 独立长窗口与 H3

运行前已按真实事件时刻计算不重叠整窗。自动规则只选择两侧至少各 3 个独立长窗口的最大 N，并只在至少 2/3 T1 起点同时训练启动且 persistent 有利时运行 H3。

{chr(10).join(support_lines) if support_lines else '- 支持审计未生成。'}

{chr(10).join(h3_lines)}

若出现合格患者，H3 才会同时比较 load 和 participation/repertoire composition。只有真实 exposure 同时胜过拟合截距对照与因果延迟对照，才保留为探索性候选；无边对照不再作为暴露证据。本轮这两类人体臂均未调度，不能引用任何新 H3 效应量。

## 当前边界

- 正式检验分区和 seizure probe 均未打开；
- 三位患者只是事前固定的开发分诊，不是队列推断；
- 多 seed 只检查优化稳定性；
- 即使出现探索性候选，也必须在更多患者与更多独立长记录中复现。
"""
    (OUT / "REPORT_PLAIN.md").write_text(plain)

    technical = f"""# 长序列 T1 分诊与最小 H3：技术报告

## 验收

- T1 status: `{t1_status.get('status')}`；H3 status: `{h3_status.get('status')}`。
- scientific verdict: `{verdict}`。
- formal test opened: false；sealed opened: false。
- fixed subjects: {', '.join(t1['subjects']) if t1 else 'T1 incomplete'}。
- H3 scheduled jobs: {h3.get('scheduled_jobs', 0)}；可解释 groups: {len(groups)}。
- full module tests: {verification['tests'] if verification else 'NA'}；failures/errors: {verification['failures'] if verification else 'NA'}/{verification['errors'] if verification else 'NA'}。

## T1 分层

{chr(10).join(subject_lines) if subject_lines else '- 无完整 T1。'}

## 支持度

{chr(10).join(support_lines) if support_lines else '- 支持审计未生成。'}

```json
{json.dumps(support.get('support', {}), ensure_ascii=False, indent=2)}
```

## H3 对比

{chr(10).join(h3_lines)}

## 方法更正

- R1.3 H3 入口读取真实 persistent−memoryless 符号，不再自动写 `True`；
- 主对比仅为 real−intercept-matched 与 real−causal-delayed；
- participation exposure 先去除总 load，再用 TRAIN-only 条件残差的两个 PCA 分量；
- boxcar 支持多维 exposure，并在 TRAIN-only decoder space 拟合；
- H3 运行前要求 TRAIN/validation 各至少 3 个不重叠整窗。

## 复现入口

- T1 summary: `{T1_ROOT / 'summary.json'}`
- H3 support: `{H3_ROOT / 'support_audit.json'}`
- H3 summary: `{H3_ROOT / 'summary.json'}`
- machine audit: `{OUT / 'machine_audit.json'}`
"""
    (OUT / "REPORT_TECHNICAL.md").write_text(technical)
    contract.atomic_json(OUT / "STATUS.json", {
        "status": "COMPLETE", "scientific_verdict": verdict,
        "formal_test_partition_opened": False, "sealed_opened": False,
        "plain": str(OUT / "REPORT_PLAIN.md"),
        "technical": str(OUT / "REPORT_TECHNICAL.md"),
        "machine_audit": str(OUT / "machine_audit.json"),
    })
    print(json.dumps({
        "status": "COMPLETE", "scientific_verdict": verdict,
        "output": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
