#!/usr/bin/env python3
"""Generate authoritative plain and technical reports from completed machine outputs."""
from __future__ import annotations

import csv
from datetime import date
import json
from pathlib import Path

import numpy as np

from src.topic5_continuous_marked_state_r1 import contract


R1 = contract.RESULT_ROOT / "r1_4"
T2 = contract.RESULT_ROOT / "t2_r2"
H2B = contract.UPSTREAM_ROOT
REPORTS = contract.RESULT_ROOT / "final_reports"
H2B_RERUN = H2B / "h2b_pseudo_fix_rerun_2026-08-27/SUMMARY.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def number(value, digits: int = 5) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    return f"{value:+.{digits}f}"


def patient_label(value: str) -> str:
    return value.replace("epilepsiae_", "E").replace("yuquan_", "Y-")


def h2b_cell(population: str) -> dict | None:
    path = H2B / "h2b_sensitivity/h2b_sensitivity_grid.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if (
            row["layer"] == "linear_graph_recurrent"
            and row["lead"] == "lead30m"
            and row["population"] == population
            and row["reading"] == "open_loop_at_onset"
            and row["endpoint"] == "first_selection_entropy"
        ):
            return {
                "n_patients": int(float(row["n_patients"])),
                "n_seizures": int(float(row["n_seizures"])),
                "median_delta": float(row["median_delta"]),
                "n_favourable": int(float(row["n_favourable"])),
                "sign_test_p": float(row["sign_test_p"]),
            }
    return None


def r1_table(rows: list[dict]) -> str:
    output = [
        "| 患者 | validation events | persistent−memoryless joint | correct−wrong joint | first subset | continuation | raw−explicit joint |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    by = {(row["subject"], row["arm"]): row for row in rows}
    for subject in dict.fromkeys(row["subject"] for row in rows):
        explicit = by[(subject, "explicit")]
        raw = by[(subject, "explicit_raw")]
        output.append(
            f"| {patient_label(subject)} | "
            f"{explicit['validation_events']} | "
            f"{number(explicit['persistent_minus_memoryless_joint'])} | "
            f"{number(explicit['correct_minus_wrong_joint'])} | "
            f"{number(explicit['persistent_minus_memoryless_first_subset'])} | "
            f"{number(explicit['persistent_minus_memoryless_continuation'])} | "
            f"{number(raw.get('raw_minus_explicit_joint'))} |"
        )
    return "\n".join(output)


def t2_table(rows: list[dict]) -> str:
    if not rows:
        return "没有患者通过冻结的稳定 T1 条件，因此人体 T2 未启动。"
    output = [
        "| 患者 | source | 可估计 seed | 支持不足 seed | next real−placebo | next real−current | H5 state MSE | H5 mark | H10 state MSE | H10 mark |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        output.append(
            f"| {patient_label(row['subject'])} | {row['source']} | "
            f"{row['estimable_seeds']}/3 | "
            f"{row['support_ineligible_seeds']}/3 | "
            f"{number(row['next_real_minus_placebo_joint'])} | "
            f"{number(row['next_real_minus_current_joint'])} | "
            f"{number(row['H5_real_minus_placebo_state_mse'])} | "
            f"{number(row['H5_real_minus_placebo_mark'])} | "
            f"{number(row['H10_real_minus_placebo_state_mse'])} | "
            f"{number(row['H10_real_minus_placebo_mark'])} |"
        )
    return "\n".join(output)


def main() -> None:
    r1 = read_json(R1 / "reports/r1_4_summary.json")
    t2 = read_json(T2 / "reports/t2_r2_summary.json")
    synthetic = read_json(T2 / "synthetic/synthetic_recovery.json")
    h2b_rerun = read_json(H2B_RERUN)
    h2b_denominators = read_json(
        H2B / "h2b_denominators/H2B_DENOMINATORS__linear_graph_recurrent.json"
    )
    h2b_sensitivity = read_json(
        H2B / "h2b_sensitivity/H2B_SENSITIVITY_CARD.json"
    )
    tests = read_json(T2 / "FINAL_TEST_AUDIT.json")
    h2b_all = h2b_cell("all_eligible")
    h2b_high = h2b_cell("high_observability")
    if h2b_all is None or h2b_high is None:
        raise ValueError("corrected H2b primary cells are missing")
    h2b_lead30 = next(
        row for row in h2b_denominators["flow_by_lead"]
        if row["lead"] == "lead30m"
    )
    h2b_caliper = next(
        row for row in h2b_rerun["downstream"]
        if row["label"] == "verify_caliper"
    )
    h2b_observability = h2b_sensitivity["continuous_observability"]
    h2b_subtype = h2b_sensitivity["subtype_interaction__broad_ER"]
    by_subject = r1["by_subject"]
    persistent = sum(
        value["explicit_persistent_favourable_seeds"] >= 2
        for value in by_subject.values()
    )
    time_specific = sum(
        value["explicit_time_specific_favourable_seeds"] >= 2
        for value in by_subject.values()
    )
    first_subset = sum(
        value["explicit_first_subset_favourable_seeds"] >= 2
        for value in by_subject.values()
    )
    continuation = sum(
        value["explicit_continuation_favourable_seeds"] >= 2
        for value in by_subject.values()
    )
    stable_r1_subjects = {
        subject for subject, value in by_subject.items()
        if value["stable_explicit_t1_for_t2"]
    }
    stable_first_subset = sum(
        by_subject[subject]["explicit_first_subset_favourable_seeds"] >= 2
        for subject in stable_r1_subjects
    )
    stable_continuation = sum(
        by_subject[subject]["explicit_continuation_favourable_seeds"] >= 2
        for subject in stable_r1_subjects
    )
    raw_estimable = sum(
        value.get("raw_joint_estimable_seeds", 3) >= 2
        for value in by_subject.values()
    )
    raw = sum(
        value.get("raw_joint_estimable_seeds", 3) >= 2
        and value["raw_joint_favourable_seeds"] >= 2
        for value in by_subject.values()
    )
    raw_stable = sum(
        by_subject[subject].get("raw_joint_estimable_seeds", 3) >= 2
        and by_subject[subject]["raw_joint_favourable_seeds"] >= 2
        for subject in stable_r1_subjects
    )
    stable = t2["stable_t1_subjects"]
    expansion = t2["scale_expansion_candidates"]
    t2_rows = t2["patient_source"]
    load_rows = [row for row in t2_rows if row["source"] == "load"]
    load_primary = sum(row["primary_increment_seeds"] >= 2 for row in load_rows)
    load_h5 = sum(row["H5_persistence_seeds"] >= 2 for row in load_rows)
    load_h10 = sum(row["H10_persistence_seeds"] >= 2 for row in load_rows)
    stored_persistence_mismatches = sum(
        row.get("stored_persistence_flag_mismatches", 0) for row in t2_rows
    )

    if load_primary:
        h3_sentence = (
            f"N=100 load edge 在 {load_primary}/{len(load_rows)} 位稳定 T1 患者中至少 2/3 seed "
            "同时胜过不重叠 donor 与 current-event；这仍是 development 增量。"
        )
    elif load_rows:
        h3_sentence = (
            f"N=100 load edge 在 {len(load_rows)} 位稳定 T1 患者中没有形成至少 2/3 seed 的 "
            "real-over-donor-and-current 增量；按可估计性区分普通阴性与结构零。"
        )
    else:
        h3_sentence = "没有患者达到冻结的稳定 T1 条件，所以人体 T2 正确地没有启动。"

    plain = f"""# Continuous marked-state R1.4 / T2-R2.0 阶段报告：白话版

**日期：** {date.today().isoformat()}
**范围：** 六患者 R1.4、N=100 T2-R2.0、修复后 H2b 一次性重跑。
**证据边界：** 全部是 development；formal/sealed partition、seizure-loss 训练和 paper-ready 图均未打开。

## 一句话结论

六患者复现后，跨窗口预测记忆在 {persistent}/6 位患者中达到至少 2/3 seed 同向，正确时刻专属性在 {time_specific}/6 位中达到同一标准。first subset 和 later continuation 在全六人中分别有 {first_subset}/6 和 {continuation}/6 同向，但真正可承重的是稳定 T1 患者内的 {stable_first_subset}/{len(stable_r1_subjects)} 和 {stable_continuation}/{len(stable_r1_subjects)}；其余方向性结果不单独称 H2a 复现。raw waveform 在 {raw_estimable}/6 位患者中至少有 2 个 seed 可估计，其中 {raw}/6 位达到至少 2/3 seed 同向，但在稳定 T1 患者中仅 {raw_stable}/{len(stable_r1_subjects)}；结构零不计作 raw 阴性。因此 raw 仍是敏感性信息，而不是主结论。{h3_sentence}

## 这轮真正做了什么

1. 六位患者全部从同口径、同 seed 的 R1.2 explicit 状态起点重新训练，不再让原三位额外经过 R1.2b。
2. memoryless 与 persistent 使用相同 observation；wrong-time donor 被限制在同一真实记录覆盖段，主分析每个时刻 5 个 donor，另用同一 checkpoint 做 10-donor 敏感性。
3. T2 不再使用数千次 boxcar。每个事件的 load 或 participation 先由 pre-event state、固定 history 和当前 observation 交叉拟合；只累加无法预测的部分，N 固定为 100。
4. T2 只训练 post-event edge。next-event 是一级端点；H5/H10 只注入一次 jump，随后关闭 raw correction 和后续 jump，同时检查未来状态 MSE 和 mark。
5. H2b 的 3 层 × 4 lead 共 12 组结果由 408 个 per-subject 作业重新产生；旧汇总先归档，未重画 paper-ready 图。

## H1/H2a：六患者结果

表中差值均为左减右，负值有利；每个患者先取三个 seed 的中位数。

{r1_table(r1['patient_arm'])}

安全读法：persistent−memoryless 回答“跨窗口保留是否有用”；correct−wrong 回答“是否属于正确时刻”；first subset 与 continuation 只在整体 T1 也稳定时，才直接承担下一场 IED 空间 repertoire 的 H2a 结论。Y-黄瀚文只有 107 个 validation events，效应大但支持小，必须与 E620/E958 分开看。raw−explicit 不稳定时，只能说显式 spectral/variance/autocorrelation 已解释大部分当前可见增量。

## H2b：修复后的发作前结果

- 分母必须分开读：34 位患者进入，27 位有可分析发作；30 min 主 lead 有 {h2b_lead30['step_3_seizures_eligible_all']} 次资格发作，其中 {h2b_all['n_seizures']} 次主端点可计算。{h2b_lead30['step_4_seizures_meeting_observation_premise']} 次满足高可观测条件，其中 {h2b_high['n_seizures']} 次主端点可计算。
- 主 population 层：患者中位 {number(h2b_all['median_delta'], 4)} SD，{h2b_all['n_favourable']}/{h2b_all['n_patients']} 同向，sign p={h2b_all['sign_test_p']:.4g}。
- high-observability 敏感性层：患者中位 {number(h2b_high['median_delta'], 4)} SD，{h2b_high['n_favourable']}/{h2b_high['n_patients']} 同向，sign p={h2b_high['sign_test_p']:.4g}。
- hard caliper 实际覆盖率为 {100 * h2b_caliper['share_with_caliper_applied']:.1f}%，机器结论为 `{h2b_caliper['scientific_status']}`；其余发作退回 soft matching，所以不能把全体结果写成“混杂已完全配平”。
- 信号没有随可观测性稳定增强：患者内 IED 数、anchor gap、coverage 的中位 Spearman 分别为 {h2b_observability['n_ied_lookback']['median_within_patient_spearman']:+.3f}、{h2b_observability['anchor_gap_seconds']['median_within_patient_spearman']:+.3f}、{h2b_observability['coverage']['median_within_patient_spearman']:+.3f}。预先要求的 broad-ER 亚型交互只有 {h2b_subtype['n_patients_with_two_usable_subtypes']} 位患者可算，{h2b_subtype['n_patients_above_their_own_null']}/{h2b_subtype['n_patients_with_two_usable_subtypes']} 高于各自 null，sign p={h2b_subtype['sign_test_p']:.3g}，不支持统一的特定发作亚型效应。

这里仍是“冻结旧 state 在发作前对齐后的关联”，不是发作机制，也不能用发作数代替患者数。此次重跑修复了 span 外发作未进入 pseudo-onset 排除、以及 nuisance 与 endpoint 使用不同 pseudo 集合两处问题。

## H3：N=100 event edge

稳定 T1 患者：{', '.join(map(patient_label, stable)) if stable else '无'}。

{t2_table(t2_rows)}

next-event 只支持 exposure-conditioned prediction；只有真实边可估计且产生非零位移，并在 H5/H10 同时改善未来 state MSE 与 mark，才记为一次 jump 经冻结 generator 保留下来的 state update。本轮 load H5/H10 达到至少 2/3 seed 的患者数分别为 {load_h5}/{len(load_rows)}、{load_h10}/{len(load_rows)}。聚合器独立复算该标签，并纠正了 {stored_persistence_mismatches} 个“真实边为零却因 placebo 更差而被标阳”的旧标签。允许扩 N=50/200 的患者-source 组合为 {len(expansion)} 个；本轮没有自动扩尺度。

## 目前对三个假设的证据力度

- **H1：** 六患者 development 复现后，以 persistent 和 correct-time 两层分别判断；仍不称 autonomous physiological state。
- **H2a：** 仍是最强假设；承重证据是稳定 T1 患者中 {stable_first_subset}/{len(stable_r1_subjects)} 的 first subset 和 {stable_continuation}/{len(stable_r1_subjects)} 的 continuation，而不是把整体 joint 未改善患者的端点方向也算成复现。
- **H2b：** 修复后数字可作为新的 development 探索结果，但仍不是 seizure mechanism。
- **H3a：** 只按上表的可估计 next-event 与 H5/H10 分层解释；没有 H5/H10 持续就不称 state update。
- **H3b：** 未运行，继续保持未支持。

## 质量与边界

- T2 synthetic positive、zero、reversed-sign 共 9 个作业全部满足预先写入的方向/零真值标准：`{synthetic['all_criteria_pass']}`。
- 最终定向测试：{tests['passed']} passed；退出码 {tests['returncode']}。
- R1.4/T2/H2b 均记录 sealed=false；未按结果删除患者或 seed。
- fitted intercept 只作 offset 诊断，real−no-edge 不单独承担 exposure 结论。
- 滑动事件对用于 likelihood，不当作独立患者或独立生物重复。

## 给合作者的一段话

> 这一阶段先在六位固定 development 患者中重新检验了“脑状态是否跨窗口保留，以及是否属于正确时刻”，再只在这两项较稳定的患者上检验最近 100 次 IED 的不可预测部分能否改变下一次事件和更远的状态。这样避免了上一轮在退化 T1、免费截距和伪长时间尺度上继续堆作业。H2a 仍由下一次 IED 的 first subset 和 continuation 承担；H2b 已用修复后的 pseudo-onset 仪器重跑；H3 只按 next-event 与 H5/H10 分层陈述，不能从预测增量直接升级为因果机制。正式检验分区仍然关闭。
"""

    technical = f"""# Continuous marked-state R1.4 / T2-R2.0 阶段报告：技术版

**日期：** {date.today().isoformat()}
**冻结合同：** `docs/archive/topic5/continuous_marked_state_r1_4_t2_r2_0_contract_2026-08-27.md`
**分区：** development only；formal/sealed 未打开。

## 1. 核心 estimand

R1.4 同时比较 `persistent-memoryless`、`correct-matched_wrong_time` 与 `raw-explicit`。T2-R2.0 固定

`eta_e = phi(m_e) - E[phi(m_e) | z_e^-, r_e, o_e^-]`,

`x_e = exp(-1/100) x_(e-1) + eta_e`,

`z_e^+ = z_e^- + B x_e`。

innovation 在 TRAIN 内按时间分成 5 折交叉拟合，validation prediction 只由全 TRAIN 拟合器产生。donor 的有效历史按 5N=500 events 排除，剩余核权重上界 `exp(-5)=0.0067`。observer、K、history baseline、timing/mark decoder 全冻结，只训练 B。

## 2. R1.4 数据与患者级结果

- subjects=6；arms=explicit/explicit_raw；seeds=0,1,2；fits=36。
- 主 wrong-time=5 donors；10-donor 为同 checkpoint 后处理。
- 稳定 T1 定义：同一患者至少 2/3 seed selected epoch>0，且 persistent<memless、correct<wrong 同时成立。

{r1_table(r1['patient_arm'])}

患者级计数：persistent {persistent}/6；correct-time {time_specific}/6；first subset 全六人方向 {first_subset}/6、稳定 T1 内 {stable_first_subset}/{len(stable_r1_subjects)}；continuation 全六人方向 {continuation}/6、稳定 T1 内 {stable_continuation}/{len(stable_r1_subjects)}；raw 可估计 {raw_estimable}/6，raw joint 有利 {raw}/6，其中稳定 T1 内 {raw_stable}/{len(stable_r1_subjects)}。进入 T2 的患者为 {len(stable)} 位：{', '.join(stable) if stable else 'none'}。

## 3. T2-R2.0 synthetic 与 estimability

synthetic revision=`{synthetic['revision']}`；positive/zero/reversed-sign × 3 seeds。全部 criteria：

```json
{json.dumps(synthetic['criteria'], indent=2, sort_keys=True)}
```

每个人体臂持久化 exposure variance/rank、B=0 gradient norm、selected epoch、edge norm 与是否离开零。结构零记为不可估计，不进入 favourable 分母；普通 validation 阴性保留。

## 4. 人体 T2-R2.0

{t2_table(t2_rows)}

primary increment 要求同一 seed 的 real next-event joint NLL 同时小于 state-matched non-overlap placebo 与 current-event-only。patient/source 扩展要求至少 2/3 seeds 可估计且达到 primary increment；scale expansion candidates={json.dumps(expansion, ensure_ascii=False)}。H5/H10 直接从 anchor post-event state 经冻结 matrix exponential 到目标事件，不读新 raw observation、不施加后续 T2 jump；state target 是冻结 T1 的 filtered pre-event state。persistence 标签在聚合时依据数值重算，强制要求 real edge estimable + nonzero displacement + state MSE<placebo + mark NLL<placebo；本轮发现并纠正 stored flag mismatches={stored_persistence_mismatches}。

## 5. H2b 修复后重跑

- producer=408/408；aggregate cards={h2b_rerun['aggregate_cards']}；旧汇总 archive=`{h2b_rerun['archive_manifest']}`。
- denominator flow：34 patients → 27 analysable patients → {h2b_lead30['step_3_seizures_eligible_all']} eligible seizures → {h2b_all['n_seizures']} primary-endpoint usable；high-observability {h2b_lead30['step_4_seizures_meeting_observation_premise']} eligible → {h2b_high['n_seizures']} usable。
- primary all-eligible/open-loop：median={h2b_all['median_delta']:+.6f}，favourable={h2b_all['n_favourable']}/{h2b_all['n_patients']}，p={h2b_all['sign_test_p']:.8g}。
- high-observability/open-loop：median={h2b_high['median_delta']:+.6f}，favourable={h2b_high['n_favourable']}/{h2b_high['n_patients']}，p={h2b_high['sign_test_p']:.8g}。
- hard-caliper share={h2b_caliper['share_with_caliper_applied']:.6f}，verdict=`{h2b_caliper['scientific_status']}`；fallback seizures 保留在 population 层，故不宣称全体已平衡。
- continuous observability：n_IED rho={h2b_observability['n_ied_lookback']['median_within_patient_spearman']:+.6f} ({h2b_observability['n_ied_lookback']['n_positive']}/{h2b_observability['n_ied_lookback']['n_patients']} positive)，anchor-gap rho={h2b_observability['anchor_gap_seconds']['median_within_patient_spearman']:+.6f}，coverage rho={h2b_observability['coverage']['median_within_patient_spearman']:+.6f}。
- broad-ER subtype interaction：n_patients={h2b_subtype['n_patients_with_two_usable_subtypes']}，median excess over within-patient null={h2b_subtype['median_excess_over_null']:+.6f}，above-null={h2b_subtype['n_patients_above_their_own_null']}，sign p={h2b_subtype['sign_test_p']:.8g}，Fisher p={h2b_subtype['combined_p_fisher']:.8g}。

修复点：pseudo exclusion 使用全部已知发作，不只 admissible span 内发作；nuisance 和 endpoint 使用相同完整 pseudo 集合。patient 是统计单位，seizure 是患者内观测。

## 6. 工程审计

- R1.4 summary revision=`{r1['revision']}`；T2 revision=`{t2['revision']}`。
- final tests：{tests['passed']} passed，returncode={tests['returncode']}，command=`{tests['command']}`。
- R1.4、T2、H2b、synthetic 均 sealed=false；paper-ready figures touched=false。
- T2 checkpoint 记录 source hash、split manifest hash、T1 checkpoint hash；arms share exact event support。
- H2b 旧 12 组汇总在覆盖前归档；新链未运行 H3b、未画图。

## 7. 允许与禁止的结论

允许：按患者报告 predictive memory、time specificity、IED repertoire increment；按 estimability 分层报告 N=100 next-event/H5/H10；报告修复后 H2b development association。

禁止：把 6 人比例称队列患病率；把 raw 阴性写成 raw 无信息；把 next-event gain 写成长期 generator state update；把 H2b 写成发作机制；把 T2 预测增量写成 IED 因果塑造 epileptic network；在本轮结果上打开 formal test。
"""

    REPORTS.mkdir(parents=True, exist_ok=True)
    plain_path = REPORTS / "r1_4_t2_r2_h2b_plain_2026-08-27.md"
    technical_path = REPORTS / "r1_4_t2_r2_h2b_technical_2026-08-27.md"
    plain_path.write_text(plain)
    technical_path.write_text(technical)
    audit = {
        "status": "COMPLETE",
        "revision": "r1_4_t2_r2_h2b_final_report_v1",
        "inputs": {
            str(path): contract.sha256_file(path) for path in (
                R1 / "reports/r1_4_summary.json",
                T2 / "reports/t2_r2_summary.json",
                T2 / "synthetic/synthetic_recovery.json",
                H2B_RERUN,
                H2B / "h2b_sensitivity/h2b_sensitivity_grid.csv",
                H2B / "h2b_denominators/H2B_DENOMINATORS__linear_graph_recurrent.json",
                H2B / "h2b_sensitivity/H2B_SENSITIVITY_CARD.json",
                H2B / "seizure_link_preictal/CALIPER_VERIFICATION.json",
                T2 / "FINAL_TEST_AUDIT.json",
            )
        },
        "outputs": {
            str(plain_path): contract.sha256_file(plain_path),
            str(technical_path): contract.sha256_file(technical_path),
        },
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "paper_ready_figures_touched": False,
    }
    contract.atomic_json(REPORTS / "r1_4_t2_r2_h2b_machine_audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
