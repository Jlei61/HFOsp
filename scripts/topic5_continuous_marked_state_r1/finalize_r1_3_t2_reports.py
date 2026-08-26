#!/usr/bin/env python3
"""Write the plain, technical, audit and handoff closeout for R1.2b–T2-S1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.r1_3 import R1_3_REVISION
from src.topic5_continuous_marked_state_r1.t2_human import T2_HUMAN_REVISION


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("status") != "COMPLETE":
        raise ValueError(f"incomplete input: {path}")
    if value.get("sealed_opened") is not False:
        raise ValueError(f"sealed partition opened: {path}")
    return value


def number(value: float | None) -> str:
    if value is None:
        return "NA"
    value = float(value)
    return f"{value:.4g}"


def markdown_table(rows: list[dict], fields: list[tuple[str, str]]) -> str:
    line = ["| " + " | ".join(label for label, _ in fields) + " |"]
    line.append("|" + "|".join("---" for _ in fields) + "|")
    for row in rows:
        line.append("| " + " | ".join(
            number(row.get(key)) if isinstance(row.get(key), (int, float))
            else str(row.get(key, ""))
            for _, key in fields
        ) + " |")
    return "\n".join(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite-handoff", action="store_true")
    overwrite_handoff = bool(parser.parse_args().overwrite_handoff)
    r12b_path = contract.RESULT_ROOT / "r1_2b/reports/r1_2b_persistent_diagnostics_summary.json"
    r13_path = contract.RESULT_ROOT / "r1_3/reports/r1_3_summary.json"
    observability_path = contract.RESULT_ROOT / "t2_s1_long_scale/long_scale_observability.json"
    synthetic_path = contract.RESULT_ROOT / "t2_s1_long_scale/synthetic/synthetic_recovery.json"
    t2_path = contract.RESULT_ROOT / "t2_s1_long_scale/reports/t2_s1_summary.json"
    tests_path = contract.RESULT_ROOT / "r1_3/VALIDATION_TESTS.json"
    route_audit_path = (
        contract.RESULT_ROOT
        / "r1_2b/reports/combined_route_audit_plain_2026-08-25.md"
    )
    r0_acceptance_path = (
        contract.REPO_ROOT
        / "docs/archive/topic5/continuous_marked_state_r0_1_acceptance_2026-08-24.md"
    )
    r12b = load(r12b_path)
    r13 = load(r13_path)
    observability = load(observability_path)
    synthetic = load(synthetic_path)
    t2 = load(t2_path)
    tests = load(tests_path)
    if tests.get("failed") != 0 or tests.get("passed", 0) < 46:
        raise ValueError("continuous marked-state validation tests are incomplete")
    if not r13.get("all_raw_selection_gradients_nonzero"):
        raise ValueError("formal R1.3 raw gradient coverage failed")
    if not r13.get("all_raw_common_parameter_updates_exact_zero"):
        raise ValueError("formal R1.3 raw/common isolation failed")
    if not all(synthetic["acceptance"].values()):
        raise ValueError("T2-S1 synthetic recovery failed")

    explicit = [
        row for row in r13["patient_arm"] if row["arm"] == "explicit"
    ]
    raw = r13["raw_paired_patient"]
    n_persistent = sum(
        row["persistent_minus_memoryless_joint"] < 0 for row in explicit
    )
    n_specific = sum(row["correct_minus_wrong_joint"] < 0 for row in explicit)
    n_raw = sum(row["raw_minus_explicit_joint"] < 0 for row in raw)
    n_first = sum(
        row["persistent_minus_memoryless_first_subset"] < 0 for row in explicit
    )
    n_continuation = sum(
        row["persistent_minus_memoryless_continuation"] < 0 for row in explicit
    )
    t2_primary = [row for row in t2["rows"] if row["scale_events"] == 1000]
    n_t2_no_edge = sum(
        row["real_minus_no_edge_joint_nll_per_event"] < 0 for row in t2_primary
    )
    n_t2_placebo = sum(
        row["real_minus_state_matched_placebo_joint_nll_per_event"] < 0
        for row in t2_primary
    )
    n_t2_both = sum(
        row["real_minus_no_edge_joint_nll_per_event"] < 0
        and row["real_minus_state_matched_placebo_joint_nll_per_event"] < 0
        for row in t2_primary
    )
    if n_persistent and n_t2_both == 0:
        system_interpretation = (
            "当前最一致的是单向预测结构：背景/持续记忆影响下一 IED 表达，"
            "但在 N=1000 内尚未见当前状态之外的 IED 累积残余边；"
            "这既不检验已经由当前状态中介的总效应，也不排除更弱或 N=10000 效应。"
        )
    elif n_persistent and n_t2_both:
        system_interpretation = (
            "当前同时出现 state→IED 与累积 exposure→下一事件的预测增量，"
            "构成双向 development 线索，但后者仍不是因果生理机制证明。"
        )
    else:
        system_interpretation = (
            "当前 persistent-state 与 exposure-edge 证据均不稳定，不能建立闭环状态模型。"
        )

    report_root = contract.RESULT_ROOT / "final_reports"
    report_root.mkdir(parents=True, exist_ok=True)
    plain_path = report_root / "r1_2b_r1_3_t2_s1_plain_2026-08-25.md"
    technical_path = report_root / "r1_2b_r1_3_t2_s1_technical_2026-08-25.md"
    audit_path = report_root / "r1_2b_r1_3_t2_s1_machine_audit.json"
    handoff_path = contract.RESULT_ROOT / "CURRENT_HANDOFF.md"

    plain = f"""# Continuous marked-state：R1.2b、R1.3 与长尺度 H3 白话报告

## 一句话

我们现在能把“当前 30 秒脑电的即时提示”和“跨多个窗口保留下来的预测记忆”分开；
R1.3 中 {n_persistent}/3 位患者的 persistent 模型优于 memoryless，但只有
{n_specific}/3 位在严格 matched wrong-time 下由正确时刻胜出。完整 raw 分支确实接受了
IED 目标梯度，且共同读出没有多训练，但 raw 相对 explicit 只在 {n_raw}/3 位有利。
因此 H1 最安全的叫法仍是 **三位均有跨窗口预测记忆，但尚未稳定证明时刻专属生理状态**。

## 这几轮有没有走偏

没有。R0.1 先停止了“用未来频谱预测代答 H1–H3”的错误工具，并把旧 H2a/H3-S0 限定为
development predictive evidence；R1.2 首次把完整 recorded-time timing likelihood 和
sequential mark 放进同一个目标，history timing 在 6/6 患者胜 static，但 frozen observer
下 persistent 只有 2/6 有利；R1.2b 只对齐 spatial tail，2/3 出现 mark filter 增量但
0/3 通过 wrong-time；本轮 R1.3 才第一次让完整 explicit/raw observer 对正确 IED 目标学习，
并把 persistent、time specificity、raw increment 分开验收。H3 同时保持独立，不被 raw
普通阴性 gate。路线是在逐步接近原始问题，不是在旧频谱支线上继续调参。

## H1：有没有跨时间持续的内部状态

- R1.2b 先验收出 2/3 位患者的跨窗口记忆主要帮助 mark，而不是 timing；
- R1.3 端到端重新训练 explicit/raw observer 后，persistent 胜 memoryless 为
  {n_persistent}/3；correct-time 胜 matched wrong-time 为 {n_specific}/3；
- raw 全栈的 tokenizer 和两层 temporal Transformer 在所有 9 个 raw fit 中梯度非零，
  共同 spatial/state/readout 更新量严格为 0；raw 真正被训练过，但只在 {n_raw}/3 位
  胜过配对 explicit。另有 {r13['raw_selected_final_budget_fits']}/9 个 raw fit 把最佳点选在
  当前 4 epoch 对齐预算的最后一轮，因此这是“当前预算下未见稳定 raw 增量”，不是 raw 生物学阴性。

结论：H1 有 **development 级跨窗口预测记忆证据**；时刻专属性和 raw 独立增量按实际计数
分别报告，不能升级成自主生理慢状态。

## H2a：这个状态会不会改变下一场 IED 的表达

精确 sequential mark 里，persistent 对 first subset 有利 {n_first}/3，对 later
continuation 有利 {n_continuation}/3。这里预测的是谁参与、后续如何延续和何时 STOP，
而不是只看一个参与度总数。

结论：若改善集中在 subset/continuation，可称 **state-dependent repertoire prediction**；
若只在 STOP/size，则只能称 extent/termination memory。仍是预测关系，不是因果机制。

## H3：大量 IED 会不会反过来塑造后续状态

你提出的长尺度判断被纳入主设计：100 次只是参照，1000 次是当前主探索，10000 次是
高事件量目标。严格不跨记录缺口时，固定 R1.3 三人无人能做 10000 次；620 和 958
可做 1000 次，其历史覆盖中位分别约 3.28 小时和 1.22 小时。张家齐有 5,715 个
validation 候选下一事件能做 10000 次，历史覆盖中位约 6.04 小时，但必须先有同合同
target-trained T1，不能拿旧状态混用。事件数不是统一的物理时间常数，最终必须把两条轴并报。

仪器先在合成数据上通过：真边 3/3 seeds 恢复正确方向，联合误差中位改善
{number(synthetic['recovery']['truth_joint_gain_median'])}；打乱暴露中位只有
{number(synthetic['recovery']['truth_shuffled_gain_median'])}。
按约 770 个 validation 下一事件做的小效应阶梯显示：edge=0.02 基本不可见，0.05
在 2/3 seeds 可见，0.10 起才稳定 3/3。因此人体零结果还受小效应灵敏度限制。

人体 N=1000 中，真实累积暴露胜 no-edge 为 {n_t2_no_edge}/2 位，胜严格 state-matched
placebo 为 {n_t2_placebo}/2 位。无论结果方向如何，这只是条件于当前 T1 的 one-step
残余边筛查：阳性支持开发级 H3a，阴性不能排除 10000 次尺度。更重要的是，如果历史 IED
的作用已经被当前状态吸收，先条件于当前状态会把这条中介路径一起扣掉；所以本轮零结果也不能
排除“IED 已经塑造了当前状态”。

下一阶段的万次实验不只是把 `N` 调大：应先在张家齐训练同合同 T1，再把“未来状态相对
无 IED 自然演化的偏移”设为直接结局，并保留下一事件 likelihood 为辅助结局。具体从累积窗
开始前的共同状态出发，比较真实 exposure 序列、no-edge natural flow 和反事实 exposure
序列对窗口末端 observation-inferred state 的预测；主误差在冻结 T1 decoder 的 timing、
STOP/size、subset 和 continuation 读出空间计算，raw latent norm 只作敏感性。这样检验的才是长期塑形，而不只是某个
长历史特征能否改善下一场 IED 的预测。由于张家齐的 10000 次
中位只覆盖约 6.04 小时，另加一个固定约 6 小时的同定义辅助臂，区分 event-count accumulation
和 physical-time integration；不扩成大时间网格。

**整体系统读法：** {system_interpretation}

## 当前三条假设的证据等级

- **H1：有限支持。** 有跨窗口预测记忆；时刻专属性是否稳定由 strict swap 的实际结果限制。
- **H2a：按精确端点分层支持。** subset/continuation 才是 repertoire，STOP/size 只是范围。
- **H3a：长尺度 development 探索。** 合成仪器合格；人体证据只限两位 N=1000 one-step。
- **H2b/H3b：本轮未检验。** 没有打开 seizure probe，也不从间期预测增量外推发作机制。

正式检验分区保持关闭；所有结论都是 development 级。paper-ready Fig1–Fig4 未触碰。
"""
    plain_path.write_text(plain)

    r13_table = markdown_table(explicit, [
        ("患者", "subject"),
        ("persistent-memoryless joint", "persistent_minus_memoryless_joint"),
        ("timing", "persistent_minus_memoryless_timing"),
        ("mark", "persistent_minus_memoryless_mark"),
        ("first subset", "persistent_minus_memoryless_first_subset"),
        ("continuation", "persistent_minus_memoryless_continuation"),
        ("correct-wrong joint", "correct_minus_wrong_joint"),
    ])
    raw_table = markdown_table(raw, [
        ("患者", "subject"),
        ("raw-exp joint", "raw_minus_explicit_joint"),
        ("timing", "raw_minus_explicit_timing"),
        ("mark", "raw_minus_explicit_mark"),
        ("group size", "raw_minus_explicit_group_size"),
        ("subset", "raw_minus_explicit_subset"),
    ])
    t2_table = markdown_table(t2["rows"], [
        ("患者", "subject"), ("N", "scale_events"),
        ("train pairs", "train_pairs"), ("validation pairs", "validation_pairs"),
        ("real-no edge joint", "real_minus_no_edge_joint_nll_per_event"),
        ("real-placebo joint", "real_minus_state_matched_placebo_joint_nll_per_event"),
        ("current-no edge joint", "current_event_minus_no_edge_joint_nll_per_event"),
        ("real-no edge mark", "real_minus_no_edge_mark_nll_per_event"),
    ])
    technical = f"""# R1.2b–R1.3–T2-S1 技术报告

## 1. 冻结 revision

- R1.3: `{R1_3_REVISION}`；3 patients × 2 arms × 3 seeds = 18 fits。
- T2-S1: `{T2_HUMAN_REVISION}`；2 patients × 2 scales × 3 seeds = 12 fits。
- exact recorded-support timing likelihood + exact tied-group sequential mark likelihood。
- sealed partition opened: false；patient-first aggregation: true。

## 1.1 与前两轮 goal 的连续性

- R0.1 限定验收：H2a strongest development prediction；H1 仅 predictive filter；
  H3-S0 为约 25–200 event 的 STOP/extent screen，不是 generator edge。
- R1.2：6 人 full recorded support exact timing+mark；history timing 6/6 胜 static，
  persistent/filter/wrong-time 只有 2/6 有利且 patient median 0。
- R1.2b：3 人 limited spatial-tail target alignment；filtered 增量 2/3，strict
  wrong-time 0/3，raw 上游未训练。
- R1.3/T2-S1：完整 observer target gradient、exact H2a、长尺度 residual edge；
  不改变 H1–H3 定义，不打开 seizure 或正式分区。

## 2. R1.2b 后处理

原 18 checkpoint 上增加 memoryless anchor baseline、5-donor strict matched wrong-time
和 mark endpoint decomposition。joint-explicit 的 patient median persistent-memoryless joint
为 {number(r12b['arm_summary']['joint_explicit']['persistent_minus_memoryless_joint']['patient_median'])}，
有利 {r12b['arm_summary']['joint_explicit']['persistent_minus_memoryless_joint']['n_favourable_negative']}/3；
strict correct-wrong joint 有利
{r12b['arm_summary']['joint_explicit']['correct_minus_wrong_joint']['n_favourable_negative']}/3。

## 3. Formal R1.3 explicit/H2a

{r13_table}

所有差值均为 left minus right，负数有利。epoch 0 保留为 no-update，不作患者生物学阴性。

## 4. Formal R1.3 raw increment

{raw_table}

- raw selection gradient nonzero: `{r13['all_raw_selection_gradients_nonzero']}`；
- raw common parameter update exact zero: `{r13['all_raw_common_parameter_updates_exact_zero']}`；
- raw selected at final 4-epoch budget: `{r13['raw_selected_final_budget_fits']}/9`；
- raw 从同 seed completed explicit checkpoint 初始化，只有 raw tokenizer、position/valid
  projection、两层 temporal Transformer、norm/gate 可训练。

## 5. H3 长尺度可观测性

- 620 N=1000：779 个 validation 候选，历史时长中位 3.276 h；
- 958 N=1000：1,426 个 validation 候选，历史时长中位 1.217 h；
- 固定三人 N=10000：0 个；
- 张家齐 N=10000：5,715 个，历史时长中位 6.036 h；但同合同 T1 尚未训练，本轮不运行。
- 下一阶段将 `N=10000` 作为主臂，并加同暴露定义的约 6 h 固定时间辅助臂；主结局均为
  从窗口开始前共同状态预测窗口末端 observation-inferred state，相对 no-edge natural flow
  和反事实 exposure 的增量；主评分在 frozen decoder readout space，latent norm 仅为敏感性，
  不以 next-event likelihood 代替长期塑形。

## 6. T2-S1 仪器与人体结果

Synthetic truth gain median:
{number(synthetic['recovery']['truth_joint_gain_median'])}；shuffled:
{number(synthetic['recovery']['truth_shuffled_gain_median'])}；negative truth:
{number(synthetic['recovery']['negative_joint_gain_median'])}；truth direction 3/3。

小效应阶梯（n=2,200，validation=770）：edge 0.02 的 gain median
{number(synthetic['small_edge_sensitivity_ladder']['0.02']['real_gain_median'])}、方向恢复
{synthetic['small_edge_sensitivity_ladder']['0.02']['n_direction_recovered']}/3；0.05 为
{number(synthetic['small_edge_sensitivity_ladder']['0.05']['real_gain_median'])}、
{synthetic['small_edge_sensitivity_ladder']['0.05']['n_direction_recovered']}/3；0.10 为
{number(synthetic['small_edge_sensitivity_ladder']['0.1']['real_gain_median'])}、
{synthetic['small_edge_sensitivity_ladder']['0.1']['n_direction_recovered']}/3。

{t2_table}

Exposure 是 TRAIN-only load expectation 后的 signed residual，在一个无缺口 coverage segment
内滚动累积最近 N 次并除以 sqrt(N)。四臂逐元素共享 current/next event 和 quadrature support。
validation donor 的 state-matched placebo 只来自 TRAIN。edge 只在当前事件后施加一次，随后关闭
observation correction 预测下一事件；没有递归 teacher forcing。

## 7. 结论边界

- engineering completion 不等于 H1/H2/H3 科学验收；
- prediction increment 不等于 IED 因果改变生理状态；
- 当前 T2-S1 条件于窗口末端 pre-event state，只测 residual edge，可能控制掉已进入 state
  的长期中介效应；
- event count 不等于统一物理时间；
- 两位 N=1000 阴性不能排除 N=10000；
- H2b/H3b、seizure subtype、34 人扩展与正式分区均未运行。
"""
    technical_path.write_text(technical)

    artifacts = {
        str(path): contract.sha256_file(path)
        for path in (
            r12b_path, r13_path, observability_path, synthetic_path, t2_path,
            tests_path, route_audit_path, r0_acceptance_path,
            plain_path, technical_path,
        )
    }
    audit = {
        "status": "COMPLETE",
        "r1_3_revision": R1_3_REVISION,
        "t2_revision": T2_HUMAN_REVISION,
        "r1_3_expected_fits": 18,
        "r1_3_completed_fits": r13["n_fits"],
        "t2_expected_fits": 12,
        "t2_completed_fits": t2["n_fits"],
        "raw_gradients_nonzero": r13["all_raw_selection_gradients_nonzero"],
        "raw_common_parameters_isolated": r13[
            "all_raw_common_parameter_updates_exact_zero"
        ],
        "raw_selected_final_budget_fits": r13[
            "raw_selected_final_budget_fits"
        ],
        "synthetic_acceptance": synthetic["acceptance"],
        "validation_tests_passed": tests["passed"],
        "validation_tests_failed": tests["failed"],
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "patient_first": True,
        "paper_ready_figures_touched_by_this_work_package": False,
        "artifacts_sha256": artifacts,
    }
    contract.atomic_json(audit_path, audit)
    handoff = f"""# Continuous marked-state 当前接手说明

状态：R1.2b 收口、formal R1.3 18/18、T2-S1 synthetic 与人体 12/12 均完成。

## 权威入口

- 白话报告：`{plain_path}`
- 技术报告：`{technical_path}`
- 机器审计：`{audit_path}`
- R1.3 汇总：`{r13_path}`
- T2-S1 汇总：`{t2_path}`

## 边界

- 正式检验分区未打开；全部是 development 结果。
- `N=10000` 未在人体运行；张家齐有数据支持，但先要训练同合同 T1。
- H2b/H3b、seizure probe、34 人扩展与 paper-ready 图均未触碰。
- 工作树原有无关修改全部保留，不要为本项目清扫或回滚。

## 下一工作包：H3 长尺度总效应

1. 先为张家齐训练并验收同合同 target-trained T1，不混用旧 latent/checkpoint；
2. 主臂为 `N=10000`，辅助臂为约 6 h 固定物理时间，不扩成大尺度网格；
3. 从累积窗开始前的共同状态出发，沿真实 exposure、no-edge natural flow 和反事实
   exposure 序列推进，而不是在窗口末端给当前状态加一次汇总 exposure；
4. 预测窗口末端 observation-inferred state，主评分使用 frozen timing/mark decoder readout，
   raw latent norm 仅作敏感性；next-event likelihood 仅作辅助；
5. 这一步检验长期总效应；当前 N=100/1000 结果只检验条件于当前状态后的 residual edge。

## 复现

```bash
PYTHONPATH=. /home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/topic5_continuous_marked_state_r1/aggregate_r1_3.py
PYTHONPATH=. /home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/topic5_continuous_marked_state_r1/aggregate_t2_s1_human.py
PYTHONPATH=. /home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/topic5_continuous_marked_state_r1/finalize_r1_3_t2_reports.py
```
"""
    # This script is the automated writer of the 2026-08-25 T2-S1 closeout.  A
    # later stage (the long total-effect round) hand-wrote a newer handoff into
    # the same path, and `results/` is untracked, so an unguarded rewrite here
    # would silently destroy it with no way back.  Only overwrite a handoff this
    # script itself wrote, unless the caller says otherwise.
    marker = "T2-S1 synthetic 与人体 12/12 均完成"
    if handoff_path.exists() and marker not in handoff_path.read_text():
        audit["handoff_rewritten"] = False
        audit["handoff_skipped_reason"] = (
            f"{handoff_path} was written by a later stage; rerun with "
            "--overwrite-handoff only if you mean to revert it"
        )
        if not overwrite_handoff:
            print(json.dumps(audit, indent=2, sort_keys=True))
            return
    handoff_path.write_text(handoff)
    audit["handoff_rewritten"] = True
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
