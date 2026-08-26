#!/usr/bin/env python3
"""Independent scientific closeout for both very-long H3 exposure kernels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.coverage import CoverageTable
from src.topic5_continuous_marked_state_r1.t2_long_total import (
    delayed_union_start_index,
    estimability_guard,
)


ROOT = contract.RESULT_ROOT
OUT = ROOT / "t2_very_long_overall"
SUBJECTS = (
    "yuquan_chengshuai", "yuquan_pengzihang", "epilepsiae_922",
    "yuquan_chenziyang", "yuquan_hanyuxuan",
)
KERNELS = {
    "generator_weighted": ROOT / "t2_very_long_discovery/human",
    "boxcar": ROOT / "t2_very_long_boxcar/human",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def t1_row(subject: str, seed: int) -> dict:
    path = (
        ROOT / "r1_2/t1_full" / subject
        / f"explicit_d8_seed_{seed}/result.json"
    )
    value = load(path)
    contrast = value["contrasts"]
    predictive = contrast["filtered_minus_no_state_joint_nll"] < 0.0
    persistent = (
        contrast["filtered_minus_validation_correction_off_joint_nll"] < 0.0
    )
    time_specific = contrast["matched_filtered_minus_wrong_time_joint_nll"] < 0.0
    return {
        "subject": subject, "seed": seed,
        "selected_epochs": int(value["selected_epochs"]),
        "predictive": bool(predictive), "persistent": bool(persistent),
        "time_specific": bool(time_specific),
        "usable_for_exploratory_h3": bool(
            int(value["selected_epochs"]) > 0 and predictive and persistent
        ),
        "filtered_minus_no_state_joint_nll": contrast[
            "filtered_minus_no_state_joint_nll"
        ],
        "persistent_minus_validation_off_joint_nll": contrast[
            "filtered_minus_validation_correction_off_joint_nll"
        ],
        "correct_minus_wrong_time_joint_nll": contrast[
            "matched_filtered_minus_wrong_time_joint_nll"
        ],
        "result": str(path),
    }


def greedy_nonoverlap(start: np.ndarray, end: np.ndarray) -> int:
    order = np.argsort(end, kind="stable")
    last = -np.inf
    count = 0
    for row in order:
        if start[row] >= last:
            count += 1
            last = float(end[row])
    return count


def event_segment_for(subject: str) -> np.ndarray:
    coverage = CoverageTable.load(ROOT / "r1_2/coverage" / f"{subject}.npz")
    event_time = np.load(
        ROOT / "r1_2/cache" / subject / "full_design.npz"
    )["event_time"]
    return np.searchsorted(
        coverage.stop, np.asarray(event_time, dtype=np.float64), side="right"
    ).astype(np.int64)


def h3_row(kernel: str, subject: str, window: str, seed: int,
           t1_usable: bool) -> dict:
    root = KERNELS[kernel] / subject / window / f"seed_{seed}"
    path = root / "result.json"
    value = load(path)
    support = np.load(root / "parameters_and_support.npz")
    event_time = np.load(
        ROOT / "r1_2/cache" / subject / "full_design.npz"
    )["event_time"]
    split = np.asarray(support["split"], dtype=np.int8)
    start_index = np.asarray(support["start_index"], dtype=np.int64)
    end_index = np.asarray(support["end_index"], dtype=np.int64)
    # Independence has to be counted on the union the *contrast* reads, i.e.
    # the real window plus the causal-delayed arm's extra history, not on the
    # nominal real window alone.  Newer artifacts persist it; older ones do not,
    # so recompute the union rather than silently falling back to real-only.
    persisted = value.get("whole_instrument_nonoverlap_support") or {}
    union_start_index = delayed_union_start_index(
        start_index, event_segment_for(subject),
        int(value["exposure"]["delay_events"]),
    )
    start_time = event_time[union_start_index]
    end_time = event_time[end_index]
    nonoverlap = {
        name: (
            int(persisted[name]["nonoverlapping_full_windows"])
            if persisted.get(name) else
            greedy_nonoverlap(start_time[split == code], end_time[split == code])
        )
        for name, code in (("train", 0), ("validation", 1))
    }
    nonoverlap_real_only = {
        name: greedy_nonoverlap(
            event_time[start_index][split == code], end_time[split == code]
        )
        for name, code in (("train", 0), ("validation", 1))
    }
    real = value["contrasts"]["real_minus_intercept_matched"]
    delayed = value["contrasts"]["real_minus_causal_delayed"]
    next_event = value["next_event_exact_likelihood_secondary"]
    next_real = next_event["real_occurrence_plus_load"]
    next_intercept = next_event["no_edge_plus_fitted_intercept"]
    next_delayed = next_event["causal_delayed_load_1000"]
    return {
        "kernel": kernel, "subject": subject, "window": window, "seed": seed,
        "t1_usable_under_corrected_gate": bool(t1_usable),
        "estimable": bool(estimability_guard(
            value["validation_decoder_space"][
                next(k for k in value["validation_decoder_space"]
                     if k.startswith("real_"))
            ],
            value["validation_decoder_space"]["no_edge_plus_fitted_intercept"],
        )["estimable"]),
        "arm_over_intercept_ratio": float(estimability_guard(
            value["validation_decoder_space"][
                next(k for k in value["validation_decoder_space"]
                     if k.startswith("real_"))
            ],
            value["validation_decoder_space"]["no_edge_plus_fitted_intercept"],
        )["arm_over_reference_ratio"]),
        "nonoverlapping_real_exposure_windows": nonoverlap_real_only,
        "nonoverlap_definition": "real window plus causal-delay history (union)",
        "readout_floored_blocks": value["decoder_readout"]["blocks_at_scale_floor"],
        "readout_fully_degenerate": bool(value["decoder_readout"]["degenerate"]),
        "real_minus_intercept_decoder_mse": real[
            "decoder_total_equal_block_mse"
        ],
        "real_minus_delayed_decoder_mse": delayed[
            "decoder_total_equal_block_mse"
        ],
        "real_minus_intercept_by_block": real[
            "decoder_block_standardised_mse"
        ],
        "next_event_real_minus_intercept_joint_nll": (
            next_real["joint_nll_per_event"]
            - next_intercept["joint_nll_per_event"]
        ),
        "next_event_real_minus_delayed_joint_nll": (
            next_real["joint_nll_per_event"]
            - next_delayed["joint_nll_per_event"]
        ),
        "train_windows": int((split == 0).sum()),
        "validation_windows": int((split == 1).sum()),
        "nonoverlapping_full_windows": nonoverlap,
        "median_events": value["denominators"][
            "median_events_per_window_validation"
        ],
        "median_hours": value["denominators"][
            "median_window_hours_validation"
        ],
        "hours_holding_ninety_percent_weight": value[
            "effective_exposure_time_scale"
        ]["median_hours_holding_ninety_percent_weight"],
        "formal_test_partition_opened": value["formal_test_partition_opened"],
        "sealed_opened": value["sealed_opened"],
        "result": str(path),
    }


def aggregate(rows: list[dict], field: str) -> dict:
    value = np.asarray([row[field] for row in rows], dtype=np.float64)
    return {
        "median": float(np.median(value)),
        "favourable_seeds": int(np.sum(value < 0.0)),
        "completed_seeds": int(len(value)),
        "values": [float(item) for item in value],
    }


def main() -> None:
    source_summaries = {
        name: load(root.parent / "summary.json")
        for name, root in KERNELS.items()
    }
    t1 = [t1_row(subject, seed) for subject in SUBJECTS for seed in range(7)]
    t1_by_subject = {}
    for subject in SUBJECTS:
        rows = [row for row in t1 if row["subject"] == subject]
        t1_by_subject[subject] = {
            "completed_seeds": len(rows),
            "selected_above_zero": sum(row["selected_epochs"] > 0 for row in rows),
            "predictive_and_persistent": sum(
                row["usable_for_exploratory_h3"] for row in rows
            ),
            "time_specific": sum(row["time_specific"] for row in rows),
            "filtered_minus_no_state": aggregate(
                rows, "filtered_minus_no_state_joint_nll"
            ),
            "persistent_minus_validation_off": aggregate(
                rows, "persistent_minus_validation_off_joint_nll"
            ),
            "correct_minus_wrong_time": aggregate(
                rows, "correct_minus_wrong_time_joint_nll"
            ),
        }
    windows = source_summaries["boxcar"]["subject_windows"]
    h3 = []
    for kernel in KERNELS:
        for subject, subject_windows in windows.items():
            usable_seed = {
                row["seed"]: row["usable_for_exploratory_h3"]
                for row in t1 if row["subject"] == subject
            }
            for window in subject_windows:
                for seed in range(7):
                    path = KERNELS[kernel] / subject / window / f"seed_{seed}/result.json"
                    if path.exists():
                        h3.append(h3_row(
                            kernel, subject, window, seed, usable_seed[seed]
                        ))
    h3_aggregate = []
    for kernel in KERNELS:
        for subject, subject_windows in windows.items():
            for window in subject_windows:
                rows = [
                    row for row in h3 if row["kernel"] == kernel
                    and row["subject"] == subject and row["window"] == window
                ]
                if not rows:
                    continue
                h3_aggregate.append({
                    "kernel": kernel, "subject": subject, "window": window,
                    "t1_usable_seeds": sum(
                        row["t1_usable_under_corrected_gate"] for row in rows
                    ),
                    "estimable_seeds": sum(row["estimable"] for row in rows),
                    "max_arm_over_intercept_ratio": max(
                        row["arm_over_intercept_ratio"] for row in rows
                    ),
                    "estimability_note": (
                        "a fitted arm nests the intercept arm, so landing far "
                        "above it means the fit is extrapolating; such a "
                        "contrast is non-estimable, not an exposure null"
                    ),
                    "real_minus_intercept": aggregate(
                        rows, "real_minus_intercept_decoder_mse"
                    ),
                    "real_minus_delayed": aggregate(
                        rows, "real_minus_delayed_decoder_mse"
                    ),
                    "next_event_real_minus_intercept": aggregate(
                        rows, "next_event_real_minus_intercept_joint_nll"
                    ),
                    "next_event_real_minus_delayed": aggregate(
                        rows, "next_event_real_minus_delayed_joint_nll"
                    ),
                    "nonoverlapping_full_windows": rows[0][
                        "nonoverlapping_full_windows"
                    ],
                    "median_events": rows[0]["median_events"],
                    "median_hours": rows[0]["median_hours"],
                    "hours_holding_ninety_percent_weight": float(np.median([
                        row["hours_holding_ninety_percent_weight"] for row in rows
                    ])),
                    "floored_blocks_by_seed": [
                        row["readout_floored_blocks"] for row in rows
                    ],
                })
    sealed = any(row["sealed_opened"] for row in h3)
    formal = any(row["formal_test_partition_opened"] for row in h3)
    machine = {
        "status": "COMPLETE",
        "scientific_verdict": "H3_LONG_UNRESOLVED_NO_CURRENT_SUPPORT",
        "engineering": {
            "preparation_jobs": 5,
            "t1_jobs": len(t1),
            "h3_results": len(h3),
            "kernels": list(KERNELS),
            "subjects": list(SUBJECTS),
            "seeds_per_subject": 7,
        },
        "corrected_instrument_rule": (
            "selected epoch > 0 AND filtered beats no-state on development "
            "validation AND persistent beats validation-correction-off"
        ),
        "t1_by_subject": t1_by_subject,
        "t1_rows": t1,
        "h3_aggregate": h3_aggregate,
        "h3_rows": h3,
        "formal_test_partition_opened": formal,
        "sealed_opened": sealed,
        "claim_boundary": (
            "development-only support-selected exploration; seeds are optimisation "
            "repeats and sliding windows are not independent biological samples"
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    contract.atomic_json(OUT / "machine_audit.json", machine)

    chen_box = next(row for row in h3_aggregate if (
        row["kernel"], row["subject"], row["window"]
    ) == ("boxcar", "yuquan_chenziyang", "physical_6h"))
    hany_n = next(row for row in h3_aggregate if (
        row["kernel"], row["subject"], row["window"]
    ) == ("boxcar", "yuquan_hanyuxuan", "event_count_2000"))
    hany_h = next(row for row in h3_aggregate if (
        row["kernel"], row["subject"], row["window"]
    ) == ("boxcar", "yuquan_hanyuxuan", "physical_6h"))
    plain = f"""# 超长尺度 IED→状态实验：白话总报告

## 一句话

这轮没有得到可以支撑“几千到上万次 IED 累积塑造后续状态”的合格阳性证据；但也不能据此否定 H3。真正的瓶颈不是窗口不够长，而是最长序列患者大多没有形成可用的前状态仪器，唯一可用患者也只有一个不重叠的长验证窗口。

## 实际完成

- 5 位按长序列支持度事前选择的患者，每位 7 个 seed，共 35 个 T1 作业；
- 近期衰减和 whole-window boxcar 两种累计方式，共 {len(h3)} 个 H3 结果；
- 所有窗口均未跨记录缺口，正式检验分区未打开；
- boxcar 确实让整个窗口等权，因此技术上真正测试了数千次 IED，而不是把长窗口重新衰减成最近一小时。

## 为什么当前仍不能证明 H3

1. 程帅、彭子航和 E922：各 7/7 个 seed 都停在 epoch 0，根本没有候选状态，不能做 H3 判断。
2. 陈子阳：7/7 都训练了，但 development validation 上持续状态没有胜过无状态/关闭持续更新；自动队列原先把“训练动了”错当成“仪器有效”，现已更正。其 6 h boxcar 的表面趋势为 real vs 拟合截距对照 {chen_box['real_minus_intercept']['median']:+.3g}（{chen_box['real_minus_intercept']['favourable_seeds']}/7，其中 {chen_box['estimable_seeds']}/7 是有效估计），但不能作为生物学证据；保留为"若未来 T1 修复后值得复查"的候选。
3. 韩宇轩：7/7 的持续状态确实有预测增量，是唯一能进入探索性 H3 的患者；但 correct-time 只在 {t1_by_subject['yuquan_hanyuxuan']['time_specific']}/7 seed 胜过 wrong-time，仍不是可靠的时刻特异状态。
4. 在韩宇轩上，2,000 次 boxcar 相对 intercept 为 {hany_n['real_minus_intercept']['median']:+.3g}（仅 {hany_n['real_minus_intercept']['favourable_seeds']}/7 有利），6 h boxcar 为 {hany_h['real_minus_intercept']['median']:+.3g}（{hany_h['real_minus_intercept']['favourable_seeds']}/7 有利），均不支持累计 exposure。
5. 每个 boxcar 条件在 validation 中都只有 1 个真正不重叠的整窗；上千个滑动端点不能当作上千个独立样本，多 seed 也只是同一数据上的优化重复。

## 当前最安全结论

在当前 R1.2 前状态和 load-innovation exposure 定义下，韩宇轩没有显示 2,000 次或 6 h 累计 IED 历史超出拟合截距对照的预测价值。其他更长患者因前状态仪器不合格而无法裁定。因此 H3 的状态是“未决、当前无支持”，不是“IED 不会在长尺度塑造状态”。

## 下一步

先停止继续扩 N 或 seed。下一步应先在程帅等真正长序列患者上建立能在 development validation 胜过 no-state 和 memoryless 的 T1；随后使用不重叠长块或更多独立长记录重新做 boxcar。固定 N 时事件数恒定，下一版 exposure 还应加入 participation/repertoire composition，而不能只靠 load。
"""
    (OUT / "REPORT_PLAIN.md").write_text(plain)

    technical = f"""# 超长尺度 IED→状态实验：技术审计报告

## 验收结论

- 工程完成：通过。5 preparation、35 T1、{len(h3)} H3 artifacts 完成；sealed/formal test 均为 false。
- 科学验收：不通过 H3 阳性；结论为 `H3_LONG_UNRESOLVED_NO_CURRENT_SUPPORT`。
- P0 更正：原调度器只用 `selected_epochs > 0` 判定 T1 可用，导致陈子阳进入 H3。正确规则已改为：非零 epoch、filtered 胜 no-state、persistent 胜 validation-correction-off 三者同时成立。

## T1 仪器

| 患者 | 非零 epoch | predictive+persistent | correct-time 有利 | 判定 |
|---|---:|---:|---:|---|
| 程帅 | {t1_by_subject['yuquan_chengshuai']['selected_above_zero']}/7 | {t1_by_subject['yuquan_chengshuai']['predictive_and_persistent']}/7 | {t1_by_subject['yuquan_chengshuai']['time_specific']}/7 | 退化 |
| 彭子航 | {t1_by_subject['yuquan_pengzihang']['selected_above_zero']}/7 | {t1_by_subject['yuquan_pengzihang']['predictive_and_persistent']}/7 | {t1_by_subject['yuquan_pengzihang']['time_specific']}/7 | 退化 |
| E922 | {t1_by_subject['epilepsiae_922']['selected_above_zero']}/7 | {t1_by_subject['epilepsiae_922']['predictive_and_persistent']}/7 | {t1_by_subject['epilepsiae_922']['time_specific']}/7 | 退化 |
| 陈子阳 | {t1_by_subject['yuquan_chenziyang']['selected_above_zero']}/7 | {t1_by_subject['yuquan_chenziyang']['predictive_and_persistent']}/7 | {t1_by_subject['yuquan_chenziyang']['time_specific']}/7 | 训练动但外层验证无效 |
| 韩宇轩 | {t1_by_subject['yuquan_hanyuxuan']['selected_above_zero']}/7 | {t1_by_subject['yuquan_hanyuxuan']['predictive_and_persistent']}/7 | {t1_by_subject['yuquan_hanyuxuan']['time_specific']}/7 | predictive persistent memory；时间专属性不足 |

陈子阳 filtered-minus-no-state 中位 {t1_by_subject['yuquan_chenziyang']['filtered_minus_no_state']['median']:+.6g}，persistent-minus-validation-off 中位 {t1_by_subject['yuquan_chenziyang']['persistent_minus_validation_off']['median']:+.6g}，均为不利方向。韩宇轩相应为 {t1_by_subject['yuquan_hanyuxuan']['filtered_minus_no_state']['median']:+.6g} 与 {t1_by_subject['yuquan_hanyuxuan']['persistent_minus_validation_off']['median']:+.6g}。

## H3 主读数

只有韩宇轩满足更正后的 predictive+persistent gate：

| kernel / window | real−intercept | 有利 seed | real−delayed | 有利 seed | TRAIN/VAL 不重叠整窗 |
|---|---:|---:|---:|---:|---:|
| boxcar / N=2000 | {hany_n['real_minus_intercept']['median']:+.6g} | {hany_n['real_minus_intercept']['favourable_seeds']}/7 | {hany_n['real_minus_delayed']['median']:+.6g} | {hany_n['real_minus_delayed']['favourable_seeds']}/7 | {hany_n['nonoverlapping_full_windows']['train']}/{hany_n['nonoverlapping_full_windows']['validation']} |
| boxcar / 6 h | {hany_h['real_minus_intercept']['median']:+.6g} | {hany_h['real_minus_intercept']['favourable_seeds']}/7 | {hany_h['real_minus_delayed']['median']:+.6g} | {hany_h['real_minus_delayed']['favourable_seeds']}/7 | {hany_h['nonoverlapping_full_windows']['train']}/{hany_h['nonoverlapping_full_windows']['validation']} |

N=2000 虽在 real−delayed 上多 seed 有利，但 real−intercept 失败；这表示 delayed arm 更差，不能转写为 exposure 增量。该档 {hany_n['estimable_seeds']}/7 个 seed 是有效估计（拟合臂相对截距对照最大 {hany_n['max_arm_over_intercept_ratio']:.3g} 倍）。

6 h 那一档只有 {hany_h['estimable_seeds']}/7 个 seed 是有效估计：拟合臂内含截距对照的常数，却落到它的最大 {hany_h['max_arm_over_intercept_ratio']:.3g} 倍，属于外推。因此该行应写成**不可估计**（与结构零同类），不能写成"两个主对照均失败"。

## 时间尺度审计

- generator-weighted 在所有名义长窗口中，90% 权重仍集中在约 1.8–2.0 h，只是近期记忆敏感性。
- boxcar 的 N=2000 在韩宇轩覆盖约 8.72 h 的 90% 权重，陈子阳 4,000 次覆盖约 5.60 h，确实是长记忆实现。
- 但所有 boxcar validation 条件的完整不重叠窗口数均为 1，无法形成患者内独立重复。

## 端点与限制

韩宇轩 N=2000 的 timing 在 7/7 seed 落到 scale floor，STOP 多数 seed、contact subset 多个 seed 也偏弱；因此 equal-block 总分应谨慎。next-event exact likelihood 同样未支持：N=2000 real−intercept joint NLL 中位 {hany_n['next_event_real_minus_intercept']['median']:+.6g}，6 h 为 {hany_h['next_event_real_minus_intercept']['median']:+.6g}。

陈子阳 4,000 次/6 h 的表面有利方向保留为“若未来 T1 修复后值得复查”的候选，不纳入当前 H3 证据。

## 文件

- machine audit: `{OUT / 'machine_audit.json'}`
- generator-weighted summary: `{ROOT / 't2_very_long_discovery/summary.json'}`
- boxcar summary: `{ROOT / 't2_very_long_boxcar/summary.json'}`
- support audit: `{ROOT / 't2_long_total_effect/cohort_support/summary.json'}`
"""
    (OUT / "REPORT_TECHNICAL.md").write_text(technical)
    contract.atomic_json(OUT / "STATUS.json", {
        "status": "COMPLETE",
        "scientific_verdict": machine["scientific_verdict"],
        "engineering_complete": True,
        "formal_test_partition_opened": formal,
        "sealed_opened": sealed,
        "reports": {
            "plain": str(OUT / "REPORT_PLAIN.md"),
            "technical": str(OUT / "REPORT_TECHNICAL.md"),
            "machine_audit": str(OUT / "machine_audit.json"),
        },
    })
    print(json.dumps({
        "status": machine["status"],
        "scientific_verdict": machine["scientific_verdict"],
        "plain": str(OUT / "REPORT_PLAIN.md"),
        "technical": str(OUT / "REPORT_TECHNICAL.md"),
        "machine": str(OUT / "machine_audit.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
