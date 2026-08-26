#!/usr/bin/env python3
"""Patient-first aggregation and two-level report for R1.2."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.r1_2 import R1_2_REVISION


def _load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("status") != "COMPLETE":
        raise ValueError(f"incomplete R1.2 artifact: {path}")
    if value.get("sealed_opened") is not False:
        raise ValueError(f"sealed flag is not false: {path}")
    return value


def _sign_p(n_favourable: int, n_total: int) -> float:
    if n_total <= 0:
        return float("nan")
    tail = sum(math.comb(n_total, k) for k in range(0, min(n_favourable, n_total - n_favourable) + 1))
    return min(1.0, 2.0 * tail / (2 ** n_total))


def _summary(values: list[float | None]) -> dict:
    finite = np.asarray([value for value in values if value is not None and np.isfinite(value)])
    return {
        "n": int(len(finite)),
        "median": float(np.median(finite)) if len(finite) else None,
        "n_favourable_negative": int(np.sum(finite < 0)),
        "sign_test_two_sided_p": (
            _sign_p(int(np.sum(finite < 0)), len(finite)) if len(finite) else None
        ),
    }


def main() -> None:
    root = contract.RESULT_ROOT / "r1_2"
    rows = []
    for subject in contract.PILOT_SUBJECTS:
        baseline = _load(
            root / "baselines" / subject / "seed_0/result.json"
        )
        bridge = _load(
            root / "bridge_e1" / subject / "seed_0/result.json"
        )
        cache = _load(root / "cache" / subject / "manifest.json")
        arm = {}
        for name in ("explicit", "explicit_raw"):
            arm[name] = _load(
                root / "t1_full" / subject / f"{name}_d8_seed_0/result.json"
            )
            if arm[name].get("r1_2_revision") != R1_2_REVISION:
                raise ValueError(f"{subject}/{name}: R1.2 revision mismatch")
        explicit = arm["explicit"]
        raw = arm["explicit_raw"]
        row = {
            "subject": subject,
            "train_events": cache["n_train_events_full_recorded_support"],
            "validation_events": cache["n_validation_events_full_recorded_support"],
            "train_anchors": cache["n_train_anchors"],
            "validation_anchors": cache["n_validation_anchors"],
            "history_timing_minus_static": baseline["contrasts"]["timing_history_minus_static_validation_nll"],
            "history_mark_minus_static": baseline["contrasts"]["mark_history_minus_static_validation_nll"],
            "history_mark_minus_shuffle": baseline["contrasts"]["mark_history_minus_shuffled_validation_nll"],
            "bridge_raw_minus_explicit": bridge["contrasts_raw_minus_explicit"]["joint_nll_per_event"],
            "explicit_selected_epochs": explicit["selected_epochs"],
            "raw_selected_epochs": raw["selected_epochs"],
            "explicit_filtered_minus_no_state": explicit["contrasts"]["filtered_minus_no_state_joint_nll"],
            "explicit_filtered_minus_validation_off": explicit["contrasts"]["filtered_minus_validation_correction_off_joint_nll"],
            "explicit_timing_minus_validation_off": explicit["contrasts"]["filtered_minus_validation_correction_off_timing_nll"],
            "explicit_mark_minus_validation_off": explicit["contrasts"]["filtered_minus_validation_correction_off_mark_nll"],
            "explicit_matched_filtered_minus_wrong_time": explicit["contrasts"]["matched_filtered_minus_wrong_time_joint_nll"],
            "explicit_matched_timing_minus_wrong_time": explicit["contrasts"]["matched_filtered_minus_wrong_time_timing_nll"],
            "explicit_matched_mark_minus_wrong_time": explicit["contrasts"]["matched_filtered_minus_wrong_time_mark_nll"],
            "raw_filtered_minus_no_state": raw["contrasts"]["filtered_minus_no_state_joint_nll"],
            "raw_filtered_minus_validation_off": raw["contrasts"]["filtered_minus_validation_correction_off_joint_nll"],
            "raw_timing_minus_validation_off": raw["contrasts"]["filtered_minus_validation_correction_off_timing_nll"],
            "raw_mark_minus_validation_off": raw["contrasts"]["filtered_minus_validation_correction_off_mark_nll"],
            "raw_matched_filtered_minus_wrong_time": raw["contrasts"]["matched_filtered_minus_wrong_time_joint_nll"],
            "raw_matched_timing_minus_wrong_time": raw["contrasts"]["matched_filtered_minus_wrong_time_timing_nll"],
            "raw_matched_mark_minus_wrong_time": raw["contrasts"]["matched_filtered_minus_wrong_time_mark_nll"],
            "raw_minus_explicit_filtered_nll": (
                raw["final_validation"]["filtered"]["joint_nll_per_event"]
                - explicit["final_validation"]["filtered"]["joint_nll_per_event"]
            ),
            "raw_minus_explicit_filtered_timing_nll": (
                raw["final_validation"]["filtered"]["timing_nll_per_event"]
                - explicit["final_validation"]["filtered"]["timing_nll_per_event"]
            ),
            "raw_minus_explicit_filtered_mark_nll": (
                raw["final_validation"]["filtered"]["mark_nll_per_event"]
                - explicit["final_validation"]["filtered"]["mark_nll_per_event"]
            ),
            "explicit_matched_events": explicit["wrong_time_match"]["matched_support_events"],
            "raw_matched_events": raw["wrong_time_match"]["matched_support_events"],
        }
        rows.append(row)
    output = root / "reports"
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "r1_2_patient_first.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    endpoints = [
        "history_timing_minus_static",
        "history_mark_minus_static",
        "history_mark_minus_shuffle",
        "bridge_raw_minus_explicit",
        "explicit_filtered_minus_no_state",
        "explicit_filtered_minus_validation_off",
        "explicit_timing_minus_validation_off",
        "explicit_mark_minus_validation_off",
        "explicit_matched_filtered_minus_wrong_time",
        "explicit_matched_timing_minus_wrong_time",
        "explicit_matched_mark_minus_wrong_time",
        "raw_filtered_minus_no_state",
        "raw_filtered_minus_validation_off",
        "raw_timing_minus_validation_off",
        "raw_mark_minus_validation_off",
        "raw_matched_filtered_minus_wrong_time",
        "raw_matched_timing_minus_wrong_time",
        "raw_matched_mark_minus_wrong_time",
        "raw_minus_explicit_filtered_nll",
        "raw_minus_explicit_filtered_timing_nll",
        "raw_minus_explicit_filtered_mark_nll",
    ]
    summary = {
        "status": "COMPLETE",
        "contract": contract.REVISION,
        "r1_2_revision": R1_2_REVISION,
        "n_subjects": len(rows),
        "patient_first": {key: _summary([row[key] for row in rows]) for key in endpoints},
        "rows": rows,
        "direction": "negative values favour the first-named arm",
        "ordinary_negative_results_are_not_gates": True,
        "sealed_opened": False,
    }
    contract.atomic_json(output / "r1_2_summary.json", summary)

    def fmt(value) -> str:
        return "NA" if value is None else f"{float(value):+.6f}"

    table = [
        "| 患者 | Exp: filtered−no-state | Exp: filtered−off | Exp: filtered−wrong | Raw−Exp filtered |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            f"| {row['subject']} | {fmt(row['explicit_filtered_minus_no_state'])} | "
            f"{fmt(row['explicit_filtered_minus_validation_off'])} | "
            f"{fmt(row['explicit_matched_filtered_minus_wrong_time'])} | "
            f"{fmt(row['raw_minus_explicit_filtered_nll'])} |"
        )
    mark_table = [
        "| 患者 | Exp mark: filtered−off | Exp mark: filtered−wrong | Raw mark: filtered−off | Raw−Exp filtered mark |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        mark_table.append(
            f"| {row['subject']} | {fmt(row['explicit_mark_minus_validation_off'])} | "
            f"{fmt(row['explicit_matched_mark_minus_wrong_time'])} | "
            f"{fmt(row['raw_mark_minus_validation_off'])} | "
            f"{fmt(row['raw_minus_explicit_filtered_mark_nll'])} |"
        )
    baseline_table = [
        "| 患者 | history timing−static | history mark−static | history mark−shuffle | Bridge raw−explicit |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        baseline_table.append(
            f"| {row['subject']} | {fmt(row['history_timing_minus_static'])} | "
            f"{fmt(row['history_mark_minus_static'])} | "
            f"{fmt(row['history_mark_minus_shuffle'])} | "
            f"{fmt(row['bridge_raw_minus_explicit'])} |"
        )
    patient = summary["patient_first"]
    explicit_epoch_zero = sum(int(row["explicit_selected_epochs"]) == 0 for row in rows)
    raw_epoch_zero = sum(int(row["raw_selected_epochs"]) == 0 for row in rows)
    plain = f"""# R1.2 六人全锚点白话报告

## 一句话

R1.2 核心 T1 已在六位患者全部可读 development anchors 和完整 recorded support 上完成。过去 IED history 对下一事件发生时刻有稳定信息，但 background-observed persistent state 在三把主尺子上都只有 2/6 患者有利、中位数为 0；因此这批六人 pilot **不支持 H1 的时刻特异持续状态，也没有把 H2a 的 recruitment-state 关系复现出来**。这不是 34 人队列结论，也不检验 H2b/H3。

## 分母

六位患者均使用各自全部 admissible、可读 background anchors；ictal 与发作后 2 小时明确排除并单列 attrition，preictal 事件保留。事件与 survival integral 使用完整 TRAIN/validation recorded intervals，raw 暂不可用的区间没有被删掉。发作/postictal 禁用区之后重置状态 session；observer 在 cache 前冻结，state 训练没有重复读 raw 或重新选择 observer。

## History baseline 与 raw observer

负值有利于表头左侧的 arm。

{chr(10).join(baseline_table)}

## 逐患者结果

负值有利于表头左侧的 arm。

{chr(10).join(table)}

## H2a：完整 recruitment mark

这里把 timing 从 joint likelihood 中拆掉，只看 tied-group recruitment mark。负值仍有利于
filtered/raw arm；因此本表直接回答 persistent state 是否改善下一次 IED 的 participation、
group-size/STOP 与 contact identity 的合计概率，而不是让 timing 阳性替 mark 代答。

{chr(10).join(mark_table)}

## 患者优先汇总

- explicit filtered−no-state：中位 {fmt(patient['explicit_filtered_minus_no_state']['median'])}，{patient['explicit_filtered_minus_no_state']['n_favourable_negative']}/6 有利；
- explicit filtered−validation-off：中位 {fmt(patient['explicit_filtered_minus_validation_off']['median'])}，{patient['explicit_filtered_minus_validation_off']['n_favourable_negative']}/6 有利；
- explicit matched filtered−wrong-time：中位 {fmt(patient['explicit_matched_filtered_minus_wrong_time']['median'])}，{patient['explicit_matched_filtered_minus_wrong_time']['n_favourable_negative']}/6 有利；
- explicit mark filtered−validation-off：中位 {fmt(patient['explicit_mark_minus_validation_off']['median'])}，{patient['explicit_mark_minus_validation_off']['n_favourable_negative']}/6 有利；
- explicit mark matched filtered−wrong-time：中位 {fmt(patient['explicit_matched_mark_minus_wrong_time']['median'])}，{patient['explicit_matched_mark_minus_wrong_time']['n_favourable_negative']}/6 有利；
- raw−explicit filtered：中位 {fmt(patient['raw_minus_explicit_filtered_nll']['median'])}，{patient['raw_minus_explicit_filtered_nll']['n_favourable_negative']}/6 有利。

## 当前假设判定

- **H1：本轮不支持。** explicit 与 explicit+raw 各有 {explicit_epoch_zero}/6 和 {raw_epoch_zero}/6 患者选择 epoch 0；三把主尺子均仅 2/6 有利。黄瀚文有 filtered gain，但错误时刻状态略好；韩雨轩三把尺子同向但 wrong-time 差只有约 2e-5；139 的 joint likelihood 反而变差。
- **H2a-history：只支持 timing，mark 有明显患者异质性。** history timing−static 为 6/6 有利；history mark−static 只有 3/6、mark−shuffle 4/6。
- **H2a-state：本轮不支持。** persistent state 的 mark-only correction-off 与 wrong-time 对比都只有 2/6 有利、中位 0；139 timing 改善但 mark 恶化，黄瀚文 mark 改善但 wrong-time 更好，不能拼成一致的 state-dependent recruitment 证据。
- **raw residual：没有新增可见信息。** Bridge raw−explicit 只有 1/6 略有利，最终 T1 raw−explicit 也只有 1/6、量级约 5e-6。
- **H2b/H3：未检验。** 本轮不允许用这个阴性替它们作答。

## 读法

只有 filtered 同时优于 no-state、validation-off 和 matched wrong-time，才能把结果称为 time-specific predictive state estimate。H2a 必须另外看 mark-only 行，不能由 timing-only 改善代替。raw−explicit 只回答原始 SEEG 是否增加了显式背景特征之外的信息。普通阴性会缩窄结论，但不使已完成实验失效；H2b seizure probe、H3 event-to-state edge、34 人扩展和 sealed partition 本轮均未运行。
"""
    (output / "plain_report_2026-08-25.md").write_text(plain)

    technical = f"""# Continuous marked-state R1.2 技术报告

## 1. 合同

- revision: `{R1_2_REVISION}`
- observer: per-patient Bridge-E1 selected checkpoint, frozen before cache
- anchors: all readable 30 s development anchors
- likelihood support: all TRAIN/validation recorded intervals, including intervals without a new raw correction
- state: one 8-dimensional stable continuous state; no event-triggered jump
- continuity: reset after every ictal/postictal exclusion; no unmodelled propagation through a seizure
- event scoring: strict pre-event state `z(t-)`; an equal-time raw anchor updates only after the event
- optimisation: time-ordered truncated BPTT, 256 anchors per chunk; epoch-zero inner-TRAIN selection
- sealed partition: unopened

## 2. 逐患者主表

### 2.1 History baseline 与 Bridge observer

{chr(10).join(baseline_table)}

### 2.2 Persistent-state T1

{chr(10).join(table)}

### 2.3 H2a exact recruitment mark decomposition

{chr(10).join(mark_table)}

## 3. Patient-first summaries

```json
{json.dumps(summary['patient_first'], indent=2, sort_keys=True)}
```

## 4. Scientific interpretation

- H1 time-specific persistent-state evidence is not supported in this six-subject pilot: the three joint contrasts each favour explicit T1 in only 2/6 subjects with median zero; {explicit_epoch_zero}/6 explicit arms select epoch zero.
- H2a history-level timing is supported (6/6), but history-level marks are heterogeneous and persistent-state mark contrasts are only 2/6 favourable with median zero.
- Raw residual information beyond the explicit background features is not detected (Bridge and final T1 each 1/6 favourable, negligible magnitude).
- This run does not execute H2b, T2/H3, anchor-specific H5/H10/H20 correction-off, or a fully generative rollout.

## 5. 结论边界

`filtered−no-state` 检验状态层总体预测增量；`filtered−validation-off` 检验 validation 中持续背景校正；matched `filtered−wrong-time` 检验状态的时刻特异性；mark-only contrasts 单独检验 H2a；`raw−explicit` 检验 raw residual 增量。任一单独为负不足以升级为 controlled/autonomous state。本轮未运行 anchor-specific H5/H10/H20 correction-off、fully-generative rollout、H2b 或 H3；结果只属于六人 development pilot。
"""
    (output / "technical_report_2026-08-25.md").write_text(technical)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
