#!/usr/bin/env python3
"""Fail-closed machine audit plus collaborator-facing R1.5/H3-long reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.r1_2 import load_full_design
from src.topic5_continuous_marked_state_r1.h3_long import (
    H3_LONG_REVISION,
    H3_LONG_SUPPORT_REVISION,
    SOURCES,
)
from src.topic5_continuous_marked_state_r1.h3_long_human import R1_5_REVISION
from scripts.topic5_continuous_marked_state_r1.run_h3_long_queue import (
    SEEDS,
    instrument_ready,
)
from scripts.topic5_continuous_marked_state_r1 import aggregate_r1_5 as r1_agg
from scripts.topic5_continuous_marked_state_r1 import aggregate_h3_long as h3_agg


DATE = "2026-08-27"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=contract.REPO_ROOT, text=True
    ).strip()


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.{digits}f}"


def fmt_count(value) -> str:
    """Format a count without the directional sign used for contrasts."""
    if value is None:
        return "NA"
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.1f}"


def r1_table(summary: dict) -> str:
    rows = [
        "| 患者 | 身份 | 已更新 seed | persistent 有利 | correct-time 有利 | first subset 有利 | continuation 有利 | 联合稳定 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    by_subject = summary["by_subject"]
    for subject in summary["subjects"]:
        value = by_subject[subject]
        rows.append(
            f"| {subject} | {value['subject_role']} | "
            f"{value['updated_seeds']}/5 | "
            f"{value['persistent_favourable_seeds']}/{value['persistent_estimable_seeds']} | "
            f"{value['time_specific_favourable_seeds']}/{value['time_specific_estimable_seeds']} | "
            f"{value['first_subset_favourable_seeds']}/{value['first_subset_estimable_seeds']} | "
            f"{value['continuation_favourable_seeds']}/{value['continuation_estimable_seeds']} | "
            f"{value['joint_stable_seeds']}/{value['joint_stable_distinct_checkpoints']} distinct |"
        )
    return "\n".join(rows)


def h3_table(summary: dict) -> str:
    rows = [
        "| 患者 | source | N | 支持层 | 边可估 seed | 独立 validation 单元 | full 阳性 | boundary 支持 | H5 | H10 | real-state | real-time-trend |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for value in summary["patient_scale_source"]:
        rows.append(
            f"| {value['subject']} | {value['source']} | "
            f"{value['scale_events']} | {value['support_role']} | "
            f"{value['edge_estimable_seeds']}/5 | "
            f"{fmt_count(value['validation_independent_units_final_common_median'])} | "
            f"{value['primary_full_control_increment_distinct_payloads']}/5 | "
            f"{value['supportive_boundary_increment_distinct_payloads']}/5 | "
            f"{value['H5_persistence_distinct_payloads']}/5 | "
            f"{value['H10_persistence_distinct_payloads']}/5 | "
            f"{fmt(value['next_real_minus_state_joint'])} | "
            f"{fmt(value['next_real_minus_chronological_joint'])} |"
        )
    return "\n".join(rows)


def audit_development_times(r1_root: Path, subjects: list[str]) -> dict:
    rows = []
    for subject in subjects:
        fit = read(
            r1_root / "fits" / subject / "explicit_seed_0/result.json"
        )
        manifest = read(Path(fit["observation_cache_manifest"]))
        design = load_full_design(Path(manifest["design"]))
        split = np.asarray(design.event_split, dtype=np.int8)
        contract.assert_development_times(
            subject, design.event_time[split == 0], "train"
        )
        contract.assert_development_times(
            subject, design.event_time[split == 1], "validation"
        )
        rows.append({
            "subject": subject,
            "train_events": int((split == 0).sum()),
            "validation_events": int((split == 1).sum()),
            "maximum_event_time": float(np.max(design.event_time)),
            "dev_end_epoch": contract.load_split(subject)[1],
        })
    return {"all_pass": True, "subjects": rows}


def audit_result_packages(r1_root: Path, h3_root: Path,
                          support: dict) -> dict:
    """Re-open every result with the aggregation-time fail-closed loaders."""
    n_r1 = 0
    for subject in contract.R1_5_EXTENSION_SUBJECTS:
        for seed in SEEDS:
            path = r1_root / "fits" / subject / f"explicit_seed_{seed}/result.json"
            value = r1_agg.load_result(path)
            checkpoint = Path(value["checkpoint"])
            if (
                value.get("subject") != subject
                or value.get("seed") != seed
                or value.get("arm") != "explicit"
                or value.get("full_recorded_support") is not True
                or not checkpoint.is_file()
                or contract.sha256_file(checkpoint)
                != value.get("checkpoint_sha256")
            ):
                raise RuntimeError(
                    f"R1.5 result identity/checkpoint mismatch: {path}"
                )
            n_r1 += 1
    n_h3 = 0
    support_path = h3_root / "support/summary.json"
    for cell in support["scheduled_cells"]:
        for source in SOURCES:
            for seed in SEEDS:
                expected = {
                    "subject": cell["subject"],
                    "seed": seed,
                    "source": source,
                    "scale_events": int(cell["scale_events"]),
                    "support_role": cell["role"],
                }
                path = (
                    h3_root / "human" / cell["subject"] / source
                    / f"seed_{seed}_n_{int(cell['scale_events'])}/result.json"
                )
                h3_agg.load(
                    path, expected=expected, support_path=support_path,
                    r1_5_root=r1_root,
                )
                n_h3 += 1
    return {
        "all_pass": True,
        "r1_5_packages": n_r1,
        "h3_long_packages": n_h3,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--r1-root", type=Path, default=contract.RESULT_ROOT / "r1_5"
    )
    parser.add_argument(
        "--h3-root", type=Path,
        default=contract.RESULT_ROOT / "r1_5_h3_long",
    )
    parser.add_argument(
        "--docs-root", type=Path,
        default=contract.REPO_ROOT / "docs/archive/topic5",
    )
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    r1_status = read(args.r1_root / "STATUS.json")
    h3_status = read(args.h3_root / "STATUS.json")
    r1 = read(args.r1_root / "reports/r1_5_summary.json")
    h3 = read(args.h3_root / "reports/h3_long_summary.json")
    support = read(args.h3_root / "support/summary.json")
    synthetic = read(args.h3_root / "synthetic/synthetic_recovery.json")
    ready, instrument_checks = instrument_ready(args.h3_root)
    if not (
        r1_status.get("status") == "COMPLETE"
        and r1_status.get("revision") == R1_5_REVISION
        and h3_status.get("status") == "COMPLETE"
        and h3_status.get("revision") == H3_LONG_REVISION
        and r1.get("status") == "COMPLETE"
        and r1.get("revision") == R1_5_REVISION
        and r1.get("target_observer_runner_sha256")
        == r1_status.get("target_observer_runner_sha256")
        and h3.get("status") == "COMPLETE"
        and h3.get("revision") == H3_LONG_REVISION
        and support.get("revision") == H3_LONG_SUPPORT_REVISION
        and ready
    ):
        raise RuntimeError("R1.5/H3-long is not complete under the frozen package")
    flags = [
        r1_status.get("sealed_opened"),
        r1_status.get("formal_test_partition_opened"),
        h3_status.get("sealed_opened"),
        h3_status.get("formal_test_partition_opened"),
        r1.get("sealed_opened"), r1.get("formal_test_partition_opened"),
        h3.get("sealed_opened"), h3.get("formal_test_partition_opened"),
        support.get("sealed_opened"), support.get("formal_test_partition_opened"),
        synthetic.get("sealed_opened"),
        synthetic.get("formal_test_partition_opened"),
    ]
    if any(value is not False for value in flags):
        raise RuntimeError("formal/sealed audit did not remain explicitly false")
    r1_paths = sorted((args.r1_root / "fits").glob("*/explicit_seed_*/result.json"))
    h3_paths = sorted((args.h3_root / "human").glob("*/*/seed_*_n_*/result.json"))
    expected_h3 = len(support["scheduled_cells"]) * len(SEEDS) * len(SOURCES)
    if len(r1_paths) != len(r1["subjects"]) * len(SEEDS):
        raise RuntimeError("R1.5 fit denominator mismatch")
    if len(h3_paths) != expected_h3:
        raise RuntimeError("H3-long cell denominator mismatch")
    package_audit = audit_result_packages(
        args.r1_root, args.h3_root, support
    )
    time_audit = audit_development_times(args.r1_root, r1["subjects"])
    test = {"skipped": bool(args.skip_tests), "returncode": None, "output": None}
    if not args.skip_tests:
        process = subprocess.run(
            [
                "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python",
                "-m", "pytest", "-q", "tests/topic5_continuous_marked_state_r1",
            ], cwd=contract.REPO_ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        test.update({"returncode": process.returncode, "output": process.stdout})
        if process.returncode:
            raise RuntimeError("final Topic 5 test suite failed")
    base = "dda5f2a5^"
    changed = git("diff", "--name-only", f"{base}..HEAD").splitlines()
    forbidden_changes = [
        path for path in changed if (
            path.startswith("results/paper-ready-figure/")
            or path.startswith("scripts/paper_figures/")
            or "seizure_probe" in path
        )
    ]
    if forbidden_changes:
        raise RuntimeError(f"out-of-scope committed changes: {forbidden_changes}")
    h3_rows = h3["patient_scale_source"]
    estimability_counts = {
        key: int(sum(row[key] for row in h3_rows))
        for key in (
            "edge_estimable_seeds",
            "zero_selected_seeds",
            "rank_degenerate_seeds",
            "zero_gradient_seeds",
            "nonfinite_gradient_seeds",
            "not_estimable_seeds",
        )
    }
    n_full_rows = int(sum(
        row["support_role"] == "full_control" for row in h3_rows
    ))
    n_boundary_rows = len(h3_rows) - n_full_rows
    n_full_positive_rows = int(sum(
        row["support_role"] == "full_control"
        and row["primary_full_control_increment_distinct_payloads"] >= 3
        for row in h3_rows
    ))
    n_boundary_supportive_rows = int(sum(
        row["support_role"] == "boundary_incomplete_control"
        and row["supportive_boundary_increment_distinct_payloads"] >= 3
        for row in h3_rows
    ))
    audit = {
        "status": "COMPLETE",
        "r1_5_revision": R1_5_REVISION,
        "h3_long_revision": H3_LONG_REVISION,
        "support_revision": H3_LONG_SUPPORT_REVISION,
        "r1_5_fit_files": len(r1_paths),
        "h3_long_cell_files": len(h3_paths),
        "expected_h3_long_cells": expected_h3,
        "instrument_checks": instrument_checks,
        "synthetic_cells": len(synthetic["rows"]),
        "synthetic_all_pass": synthetic["all_cells_pass"],
        "result_package_audit": package_audit,
        "development_time_audit": time_audit,
        "h3_estimability_seed_cells": estimability_counts,
        "h3_patient_scale_source_rows": {
            "full_control": n_full_rows,
            "full_control_positive": n_full_positive_rows,
            "boundary": n_boundary_rows,
            "boundary_supportive": n_boundary_supportive_rows,
        },
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "paper_ready_or_seizure_probe_committed_changes": forbidden_changes,
        "goal_commit_range": f"{base}..HEAD",
        "goal_changed_paths": changed,
        "audited_code_head_before_report_commit": git("rev-parse", "HEAD"),
        "tests": test,
        "claim_boundary": (
            "development predictive evidence; H3 is an exact-N antecedent "
            "screen and teacher-forced latent-correction diagnostic, not causality"
        ),
    }
    final_root = args.h3_root / "final_reports"
    final_root.mkdir(parents=True, exist_ok=True)
    machine_audit_repo_path = (
        final_root.relative_to(contract.REPO_ROOT) / "machine_audit.json"
    )
    contract.atomic_json(final_root / "machine_audit.json", audit)
    args.docs_root.mkdir(parents=True, exist_ok=True)
    docs_audit_path = (
        args.docs_root
        / f"continuous_marked_state_r1_5_h3_long_machine_audit_{DATE}.json"
    )
    docs_audit_repo_path = Path("docs/archive/topic5") / docs_audit_path.name
    contract.atomic_json(docs_audit_path, audit)

    n_new = r1["n_stable_independent_extension_subjects"]
    n_calibration = len([
        subject for subject in r1["stable_all_subjects"]
        if subject in r1["calibration_subjects"]
    ])
    plain = f"""# R1.5 / H3-long 阶段报告：白话版

## 一句话

R1.5 在 3 位真正新增患者中有 **{n_new}/3** 位找到跨窗口且正确时刻特异的候选状态；3 位旧长记录校准患者中有 **{n_calibration}/3** 位满足同一条件。H3-long 的 **{n_full_positive_rows}/{n_full_rows}** 个完整对照患者-source-尺度组合达到患者级阳性，**{n_boundary_supportive_rows}/{n_boundary_rows}** 个边界组合达到支持标准；本轮没有证据表明 1,000--10,000 次 IED 历史稳定增加下一事件或未来状态预测。

## H1 / H2a：连续背景是否形成有用状态

{r1_table(r1)}

这里 5 seeds 是优化稳定性检查，不是 5 个独立患者。`0/0` 表示没有 seed 真正更新，不能读成生物学阴性。只有同一 seed 同时满足模型确实更新、persistent 胜 memoryless、正确时刻胜 matched wrong-time，且患者至少有 3 个不同 checkpoint，才允许进入后面的状态持续性分析。

具体来说，张克轩是唯一达到联合稳定标准的新增患者：5/5 seeds 均支持跨窗口持续和正确时刻特异，first subset 也为 5/5，但 continuation 为 0/5。最安全的解释是候选状态帮助预测下一事件最先募集哪些触点，尚未证明它控制整个后续传播路径。E384 的正确时刻为 5/5，但 persistent 只有 2/5，更像时刻相关 observation code；E1096 没有任何 seed 更新，不能作为阴性。

## H3：很长一段 IED 历史是否还有增量

{h3_table(h3)}

读表时负数表示真实累积历史比对照预测误差更低。只有真实历史同时胜过 state-matched、当前单次事件、时间趋势、拟合截距和可用的前一整块，并且最终共同支持至少有 3 个互不重叠 validation 单元，才记作患者级 development 阳性。其余普通阴性、epoch-0、匹配失败和支持不足分别保留。

130 个预定 seed-cells 中，只有 **{estimability_counts['edge_estimable_seeds']}** 个学到了可估的非零边；**{estimability_counts['zero_gradient_seeds']}** 个是零梯度，**{estimability_counts['zero_selected_seeds']}** 个选择了零边，**{estimability_counts['not_estimable_seeds']}** 个在最终共同支持上不可估。唯一稳定 T1 患者张克轩在 N=1,000 只有 1 个独立 validation 单元，N=3,000 又没有完整 TRAIN 支持。因此 H3 的 0 个阳性既不是仪器全面看清后的生物学否定，也没有留下可升级的稳定正信号。

## 能说与不能说

- 能说：development 数据中有 1 位新增患者形成了稳定、正确时刻特异的候选状态，主要增量落在下一事件的 first subset，而不是 continuation。
- 能说：本轮 exact-N H3 没有患者级完整对照支持；个别 seed 的有利点估计没有跨 seeds、完整对照和独立单元共同成立。
- H5/H10 没有任何组合通过。即使未来通过，也只能称 exposure-conditioned latent correction 在使用真实未来事件历史时仍有预测持续性。
- 不能说：IED 已因果塑造了真实慢状态、生成器或癫痫网络。当前并非逐事件递推的自主生成模型。
- 也不能说：长尺度 IED 对状态没有作用。唯一稳定 T1 患者的最终独立支持不足以承担这种否定。
- formal/sealed 分区、seizure probe 与 paper-ready 图均未打开或修改。
"""
    technical = f"""# R1.5 / H3-long 阶段报告：技术版

## 1. 冻结版本与分母

- R1.5 revision：`{R1_5_REVISION}`；{len(r1_paths)} 个 fits。
- H3 revision：`{H3_LONG_REVISION}`；{len(h3_paths)}/{expected_h3} 个预定 cells。
- support revision：`{H3_LONG_SUPPORT_REVISION}`；34 人 corrected recorded-segment audit。
- synthetic：{len(synthetic['rows'])} cells，all-pass={synthetic['all_cells_pass']}。
- 所有 event time 已逐 subject 重算满足 TRAIN/validation < dev_end；formal/sealed=false。

## 2. R1.5 patient-first 结果

{r1_table(r1)}

R1.5 的正式描述层要求每个 seed 同时满足 selected epoch>0、persistent−memoryless<0、correct−matched-wrong<0；epoch-0 不进入方向分母。患者层要求至少 3 个 stable seeds 且至少 3 个 distinct checkpoint hashes。

独立扩展层只有张克轩通过（1/3）：persistent、correct-time 与 first subset 均为 5/5，continuation 为 0/5。E384 的 correct-time 为 5/5，但 persistent 仅 2/5；E1096 为 5/5 epoch-0 no-update。校准层没有患者达到 3 个 stable checkpoints。

## 3. H3-long 设计

Exposure 是 TRAIN-only cross-fitted load 或 participation innovation 的 exact last-N boxcar sum，按 recorded coverage segment 重置。所有 trainable arms 有拟合 intercept；主对照为 state-matched non-overlap、current-event-only、chronological-trend、intercept-only，以及 full-control cell 中严格不重叠的 causal previous-N block。每个 cell 绑定 support、split、代码、R1.5 result/checkpoint 与身份 fingerprint。

130 个预定 seed-cells 的最终分类为：edge-estimable={estimability_counts['edge_estimable_seeds']}，ZERO_GRADIENT={estimability_counts['zero_gradient_seeds']}，ZERO_SELECTED={estimability_counts['zero_selected_seeds']}，RANK_DEGENERATE={estimability_counts['rank_degenerate_seeds']}，NONFINITE_GRADIENT={estimability_counts['nonfinite_gradient_seeds']}，SUPPORT_NOT_ESTIMABLE={estimability_counts['not_estimable_seeds']}。{n_full_positive_rows}/{n_full_rows} 个 full-control 患者-source-尺度组合达到患者级阳性，{n_boundary_supportive_rows}/{n_boundary_rows} 个 boundary 组合达到支持标准。

{h3_table(h3)}

full-control 独立单元宽度为 2N，boundary 为 N；上表独立单元数在 state matching 后的最终共同支持上计算。`primary_full_control_increment` 还要求独立单元中位 contrast 同向，且至少 3 个 validation 单元。重复 seed payload 不重复计稳定性。

张克轩是唯一 stable-T1 患者，但 N=1,000 最终只有 1 个独立 validation 单元；load 的非零边 3/5，且 0/3 胜 causal previous block，participation 虽 5/5 可估但 0/5 胜 state-matched/current-event/causal block。N=3,000 在最终共同支持上不可估。故本轮 H3 无支持，但不能解释成充分检出力下的生物学阴性。

## 4. H5/H10 解释

H5/H10 仅在当前 seed 的 T1 合格时运行；真实累积必须同时胜过 state-matched、current-event、chronological-trend、intercept 和可用 causal block，并有非零传播位移。它关闭新的 raw correction 和后续 H3 jumps，但使用真实 future event history，因此是 teacher-forced one-shot persistence，不是 autonomous rollout。本轮 H5 和 H10 均为 0 个患者通过。

## 5. 验收边界

- ordinary negative 不触发停跑。
- `ZERO_SELECTED`、`RANK_DEGENERATE`、`ZERO_GRADIENT`、`NONFINITE_GRADIENT`、`SUPPORT_NOT_ESTIMABLE` 分开统计。
- 少于 3 个最终 validation 独立单元只作描述。
- 本轮最多支持 development predictive association；不能升级为 IED→state 因果机制。

机器审计（结果树）：`{machine_audit_repo_path}`。

机器审计（随报告提交的快照）：`{docs_audit_repo_path}`。
"""
    plain_path = args.docs_root / f"continuous_marked_state_r1_5_h3_long_plain_{DATE}.md"
    technical_path = args.docs_root / f"continuous_marked_state_r1_5_h3_long_technical_{DATE}.md"
    plain_repo_path = Path("docs/archive/topic5") / plain_path.name
    technical_repo_path = Path("docs/archive/topic5") / technical_path.name
    plain_path.write_text(plain)
    technical_path.write_text(technical)
    contract.atomic_json(final_root / "report_manifest.json", {
        "status": "COMPLETE",
        "plain": str(plain_repo_path),
        "plain_sha256": contract.sha256_file(plain_path),
        "technical": str(technical_repo_path),
        "technical_sha256": contract.sha256_file(technical_path),
        "machine_audit": str(machine_audit_repo_path),
        "machine_audit_sha256": contract.sha256_file(
            final_root / "machine_audit.json"
        ),
        "submitted_machine_audit": str(docs_audit_repo_path),
        "submitted_machine_audit_sha256": contract.sha256_file(
            docs_audit_path
        ),
        "formal_test_partition_opened": False,
        "sealed_opened": False,
    })
    print(json.dumps({
        "status": "COMPLETE", "plain": str(plain_repo_path),
        "technical": str(technical_repo_path),
        "machine_audit": str(machine_audit_repo_path),
        "submitted_machine_audit": str(docs_audit_repo_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
