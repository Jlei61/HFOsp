#!/usr/bin/env python3
"""Build machine-readable and Chinese final acceptance for the RNN audit."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/topic5_ordered_history_architecture_audit"
ANALYSIS = BASE / "analysis"
FIGURE = (
    ROOT
    / "results/paper-ready-figure/"
    "fig6_ordered_history_architecture_audit/figures"
)
HISTORY_NECESSITY = (
    ROOT
    / "results/topic5_interictal_scaffold_reliability_history_necessity/"
    "history_runs_v0_1/history_necessity_summary.json"
)
HISTORY3_SHUFFLE = (
    ROOT
    / "results/topic5_interictal_scaffold_reliability_history_necessity/"
    "history3_rank_shuffle_runs_v0_1/history3_rank_shuffle_summary.json"
)
INTERNAL_STATE = (
    ROOT / "results/topic5_rnn_internal_state_reduction/INTERICTAL_SUMMARY.json"
)
DOC = (
    ROOT
    / "docs/archive/topic5/"
    "ordered_history_architecture_audit_v0_1_report_2026-07-29.md"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fmt(value: float) -> str:
    return f"{float(value):.4f}"


def evidence_label(summary: dict) -> str:
    inferential_p = summary.get(
        "selection_corrected_maxT_p", summary["wilcoxon_greater_p"]
    )
    if summary["median_gain"] > 0 and inferential_p < 0.05:
        return "SUPPORTED"
    if summary["median_gain"] > 0:
        return "POSITIVE_DIRECTION_NOT_GROUP_SIGNIFICANT"
    return "NO_POSITIVE_INCREMENT"


def resource_audit(path: Path) -> dict:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty resource log: {path}")
    return {
        "n_samples": len(rows),
        "minimum_memory_available_gb": min(
            float(row["mem_available_gb"]) for row in rows
        ),
        "maximum_gpu_memory_used_mb": max(
            float(row["gpu_memory_used_mb"]) for row in rows
        ),
        "maximum_gpu_utilization_percent": max(
            float(row["gpu_utilization_percent"]) for row in rows
        ),
        "maximum_gpu_temperature_c": max(
            float(row["gpu_temperature_c"]) for row in rows
        ),
    }


def main() -> None:
    pairing_path = BASE / "input_audit/PAIRING_AUDIT.json"
    architecture_path = ANALYSIS / "ARCHITECTURE_SUMMARY.json"
    intervention_path = ANALYSIS / "HISTORY_INTERVENTION_SUMMARY.json"
    early_path = ANALYSIS / "EARLY_ICTAL_CONDITIONAL_SUMMARY.json"
    parameter_matched_path = ANALYSIS / "PARAMETER_MATCHED_SENSITIVITY.json"
    test_audit_path = BASE / "TEST_AUDIT.json"
    formal_launcher = (
        BASE
        / "formal/architecture_controls_formal_20260729/LAUNCHER_DONE.json"
    )
    shuffle_launcher = (
        BASE
        / "rank_shuffle/selected_architecture_rank_shuffle_20260729/"
        "LAUNCHER_DONE.json"
    )
    intervention_launcher = (
        BASE
        / "interventions/selected_history_interventions_20260729/"
        "LAUNCHER_DONE.json"
    )
    parameter_launcher = (
        BASE
        / "parameter_matched/parameter_matched_formal_20260729/"
        "LAUNCHER_DONE.json"
    )
    resource_logs = {
        "fixed_hidden_architectures": (
            BASE
            / "formal/architecture_controls_formal_20260729/resource_log.csv"
        ),
        "selected_matched_rank_shuffle": (
            BASE
            / "rank_shuffle/selected_architecture_rank_shuffle_20260729/"
            "resource_log.csv"
        ),
        "parameter_matched_sensitivity": (
            BASE
            / "parameter_matched/parameter_matched_formal_20260729/"
            "resource_log.csv"
        ),
        "history_interventions": (
            BASE
            / "interventions/selected_history_interventions_20260729/"
            "resource_log.csv"
        ),
    }
    figure_path = FIGURE / "fig6_ordered_history_architecture_audit.png"
    required = [
        pairing_path,
        architecture_path,
        intervention_path,
        early_path,
        parameter_matched_path,
        test_audit_path,
        figure_path,
        HISTORY_NECESSITY,
        HISTORY3_SHUFFLE,
        INTERNAL_STATE,
        formal_launcher,
        shuffle_launcher,
        intervention_launcher,
        parameter_launcher,
        *resource_logs.values(),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"acceptance inputs missing: {missing}")
    pairing = json.loads(pairing_path.read_text())
    architecture = json.loads(architecture_path.read_text())
    intervention = json.loads(intervention_path.read_text())
    early = json.loads(early_path.read_text())
    history_necessity = json.loads(HISTORY_NECESSITY.read_text())
    history3_shuffle = json.loads(HISTORY3_SHUFFLE.read_text())
    internal_state = json.loads(INTERNAL_STATE.read_text())
    parameter_matched = json.loads(parameter_matched_path.read_text())
    launcher_payloads = {
        "fixed_hidden_architectures": json.loads(formal_launcher.read_text()),
        "selected_matched_rank_shuffle": json.loads(shuffle_launcher.read_text()),
        "history_interventions": json.loads(intervention_launcher.read_text()),
        "parameter_matched_sensitivity": json.loads(
            parameter_launcher.read_text()
        ),
    }
    for label, payload in launcher_payloads.items():
        if payload is not None and payload.get("status") != "COMPLETE":
            raise RuntimeError(f"{label} launcher is not complete: {payload}")
    selected = architecture["target_blind_best_non_gru"]["control"]
    comparisons = architecture["comparisons"]
    selected_vs_static = comparisons[
        f"{selected}_vs_static_contact_hazard"
    ]
    selected_vs_last_set = comparisons[
        f"{selected}_vs_last_set_first_order"
    ]
    selected_vs_unordered = comparisons[f"{selected}_vs_unordered_prefix"]
    selected_vs_shuffle = comparisons[
        f"{selected}_vs_matched_within_event_rank_shuffle"
    ]
    gru_vs_unordered = comparisons["full_history_gru_vs_unordered_prefix"]
    gru_vs_shuffle = comparisons["full_history_gru_vs_rank_shuffle_gru"]
    recurrent_families = [
        "linear_state",
        "vanilla_rnn",
        "full_history_gru",
        *[f"low_rank_r{rank}" for rank in (0, 1, 2, 4)],
    ]
    family_evidence = {
        family: evidence_label(
            comparisons[f"{family}_vs_unordered_prefix"]
        )
        for family in recurrent_families
    }
    n_supported_families = sum(
        value == "SUPPORTED" for value in family_evidence.values()
    )
    cross_architecture_status = (
        "SUPPORTED"
        if n_supported_families >= 2
        else "NOT_ESTABLISHED"
    )
    early_primary = early["cohort_summaries"].get(
        "selected_ordered__static_plus_unordered__absolute_margin", {}
    )
    paired_early = next(
        row
        for row in early["paired_comparisons"]
        if row["conditioning"] == "static_plus_unordered"
        and row["metric"] == "absolute_rho"
        and row["right"] == "selected_rank_shuffle"
    )
    status = "COMPLETE_LAYERED_ORDER_AND_CROSS_STATE_AUDIT"
    final = {
        "contract": "topic5_ordered_history_architecture_audit_v0_1",
        "status": status,
        "overall_scientific_verdict": (
            "WITHIN_EVENT_ORDER_SUPPORTED_IN_SELECTED_LINEAR_STATE_BUT_"
            "CROSS_ARCHITECTURE_STABILITY_AND_EARLY_ICTAL_INCREMENT_"
            "NOT_ESTABLISHED"
        ),
        "execution": {
            "temporal_pairing_audit_complete": True,
            "interictal_patients": 34,
            "architecture_seeds": 3,
            "new_architecture_cells": 204,
            "selected_shuffle_cells": 102,
            "intervention_cells": 102,
            "parameter_matched_sensitivity_cells": 204,
            "strict_early_ictal_patients": 16,
            "strict_early_ictal_seizures": 106,
            "distinct_causal_preseizure_histories": pairing[
                "strict_clinical_onset_metadata"
            ]["n_distinct_causal_histories"],
            "early_ictal_target_reused": True,
            "launcher_status": launcher_payloads,
            "resource_audit": {
                label: resource_audit(path)
                for label, path in resource_logs.items()
            },
        },
        "scientific_acceptance": {
            "within_event_ordered_history": (
                {
                    "selected_vs_unordered": evidence_label(
                        selected_vs_unordered
                    ),
                    "selected_vs_matched_rank_shuffle": evidence_label(
                        selected_vs_shuffle
                    ),
                    "interpretation": (
                        "supported for the target-blind selected linear-state "
                        "model; architecture-dependent because the "
                        "preregistered cross-architecture requirement was not met"
                    ),
                }
            ),
            "cross_architecture_stability": {
                "n_supported_recurrent_families": n_supported_families,
                "n_tested_recurrent_families": len(recurrent_families),
                "preregistered_requirement": (
                    "at least two recurrent families must survive family-wise "
                    "inference and the selected model must survive the matched "
                    "within-event order null"
                ),
                "status": cross_architecture_status,
                "per_family": family_evidence,
            },
            "selected_non_gru_architecture": selected,
            "selected_vs_static": selected_vs_static,
            "selected_vs_last_set": selected_vs_last_set,
            "selected_vs_unordered": selected_vs_unordered,
            "selected_vs_matched_rank_shuffle": selected_vs_shuffle,
            "gru_vs_unordered": gru_vs_unordered,
            "gru_vs_rank_shuffle": gru_vs_shuffle,
            "early_ictal_conditional_increment": early[
                "conditional_early_ictal_increment_status"
            ],
            "early_ictal_primary_effects": {
                "ordered_absolute_margin_beyond_static_plus_unordered": (
                    early_primary
                ),
                "ordered_minus_matched_rank_shuffle_absolute_partial_r": (
                    paired_early
                ),
            },
            "bounded_history_depth": {
                "history2_over_history1": history_necessity["contrasts"][
                    "gain_history2_over_history1"
                ],
                "history3_over_history2": history_necessity["contrasts"][
                    "gain_history3_over_history2"
                ],
                "full_over_history3": history_necessity["contrasts"][
                    "gain_full_over_history3"
                ],
                "history3_ordered_over_matched_shuffle": history3_shuffle,
                "interpretation": (
                    "the useful ordered memory is concentrated in the latest "
                    "two to three within-event rank sets"
                ),
            },
            "hidden_dimension_boundary": {
                "ordered_gru_effective_rank_median": internal_state[
                    "cohort_metrics"
                ]["full_history_gru__effective_rank"]["median"],
                "rank_shuffle_gru_effective_rank_median": internal_state[
                    "cohort_metrics"
                ]["rank_shuffle_gru__effective_rank"]["median"],
                "interpretation": (
                    "low effective dimension is shared by the order-shuffled "
                    "control and is therefore descriptive, not evidence for a "
                    "biological low-dimensional manifold"
                ),
            },
            "readout_relevant_local_memory": intervention[
                "readout_relevant_local_memory"
            ],
            "across_event_preseizure_dynamics": (
                "NOT_PRIMARY_DUE_TO_PSEUDOREPLICATION_AND_SPARSE_DISTINCT_HISTORIES"
            ),
            "parameter_matched_sensitivity": parameter_matched["comparisons"],
        },
        "claim": (
            "The audit quantifies, separately for each recurrent family, "
            "whether ordered recruitment steps improve held-out prediction "
            "over an unordered prefix and a matched order shuffle. Evidence "
            "is reported as patient-level effect sizes and uncertainty rather "
            "than compressed into a global go/no-go gate."
        ),
        "cross_state_claim": (
            "The frozen ordered-history residual is evaluated for incremental "
            "sign-free early-ictal contact-field correspondence beyond static "
            "and unordered fields within the reused clinical-onset cohort; "
            "its effect size and uncertainty are retained whether or not the "
            "group threshold is crossed."
        ),
        "forbidden_overinterpretations": [
            "two-dimensional biological seizure manifold",
            "cell-level E/I identity of hidden units",
            "continuous-time biological slow variable or recovery constant",
            "seizure-specific forecasting from a patient-static field",
            "independent validation using the reused early-ictal target",
        ],
        "paper_position": (
            "supplementary data-driven sequence-identification and boundary "
            "result between empirical interictal propagation and the separate "
            "SNN mechanism layer"
        ),
        "artifacts": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
                *required,
            ]
        },
    }
    (BASE / "FINAL_ACCEPTANCE.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2) + "\n"
    )
    producer_paths = [
        ROOT
        / "docs/superpowers/specs/"
        "2026-07-29-topic5-ordered-history-architecture-audit-v0_1.md",
        ROOT
        / "docs/superpowers/plans/"
        "2026-07-29-topic5-ordered-history-architecture-audit-v0_1.md",
        ROOT / "scripts/audit_topic5_ordered_history_pairing_v0_1.py",
        ROOT / "scripts/train_topic5_architecture_control_v0_1.py",
        ROOT / "scripts/summarize_topic5_ordered_history_architecture_v0_1.py",
        ROOT / "scripts/run_topic5_selected_history_interventions_v0_1.py",
        ROOT
        / "scripts/summarize_topic5_selected_history_interventions_v0_1.py",
        ROOT
        / "scripts/summarize_topic5_parameter_matched_architecture_v0_1.py",
        ROOT
        / "scripts/analyze_topic5_ordered_history_early_ictal_increment_v0_1.py",
        ROOT
        / "scripts/paper_figures/"
        "plot_topic5_ordered_history_architecture_audit_v0_1.py",
        ROOT / "scripts/build_topic5_ordered_history_final_acceptance_v0_1.py",
        ROOT / "scripts/run_topic5_architecture_controls_v0_1.sh",
        ROOT / "scripts/run_topic5_selected_architecture_shuffle_v0_1.sh",
        ROOT
        / "scripts/run_topic5_parameter_matched_architecture_sensitivity_v0_1.sh",
        ROOT
        / "scripts/run_topic5_selected_history_interventions_v0_1.sh",
        ROOT / "src/topic5_rank_distribution.py",
        ROOT / "src/topic5_rnn_internal_state.py",
        ROOT / "config/topic5_interictal_rank_distribution_v0_4.yaml",
        ROOT
        / "results/topic5_interictal_rank_distribution/dataset_v0_4/"
        "dataset_manifest.json",
    ]
    manifest = {
        "contract": final["contract"],
        "status": "REPRODUCIBLE_RUN_MANIFEST",
        "launcher_status": launcher_payloads,
        "target_seal": {
            "architecture_selection_target_values_read": architecture[
                "target_values_read"
            ],
            "history_intervention_target_values_read": intervention[
                "target_values_read"
            ],
            "early_ictal_target_reused": True,
        },
        "producer_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in producer_paths
        },
        "accepted_artifact_sha256": final["artifacts"],
    }
    (BASE / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )

    family_labels = {
        "linear_state": "Linear state",
        "vanilla_rnn": "Vanilla RNN",
        "full_history_gru": "GRU",
        **{
            f"low_rank_r{rank}": f"Low-rank r={rank}"
            for rank in (0, 1, 2, 4)
        },
    }
    architecture_table = "\n".join(
        [
            "| 架构 | 中位 NLL gain vs unordered | 正向患者 | 名义 P | maxT P |",
            "|---|---:|---:|---:|---:|",
            *[
                (
                    f"| {family_labels[family]} | "
                    f"{fmt(comparisons[f'{family}_vs_unordered_prefix']['median_gain'])} | "
                    f"{comparisons[f'{family}_vs_unordered_prefix']['n_positive']}/34 | "
                    f"{comparisons[f'{family}_vs_unordered_prefix']['wilcoxon_greater_p']:.3g} | "
                    f"{comparisons[f'{family}_vs_unordered_prefix']['selection_corrected_maxT_p']:.3g} |"
                )
                for family in recurrent_families
            ],
        ]
    )
    parameter_table = "\n".join(
        [
            "| 敏感性模型 | hidden | 参数量 | 中位 NLL gain vs unordered | 正向患者 | 名义 P | Holm P |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *[
                (
                    f"| {control} | {row['hidden_size']} | "
                    f"{row['n_parameters']} | "
                    f"{fmt(row['median_nll_gain_vs_unordered'])} | "
                    f"{row['n_positive_of_34']}/34 | "
                    f"{row['wilcoxon_greater_p']:.3g} | "
                    f"{row['holm_p']:.3g} |"
                )
                for control, row in parameter_matched["comparisons"].items()
            ],
        ]
    )
    intervention_labels = [
        "reverse_prefix",
        "drop_earliest",
        "reset_after_rank_0",
        "reset_after_rank_1",
        "reset_after_rank_2",
    ]
    intervention_table_rows = []
    for label in intervention_labels:
        key = (
            f"selected_ordered__{label}__heldout_event_balanced_nll"
        )
        row = intervention["summaries"][key]
        intervention_table_rows.append(
            f"| {label} | {fmt(row['median_nll_cost'])} | "
            f"{row['n_positive']}/34 | {row['wilcoxon_greater_p']:.3g} |"
        )
    intervention_table = "\n".join(
        [
            "| 干预 | 中位 NLL 代价 | 正向患者 | one-sided P |",
            "|---|---:|---:|---:|",
            *intervention_table_rows,
        ]
    )
    local_memory = intervention["readout_relevant_local_memory"]
    doc = f"""# Topic 5 有序间期历史与架构控制：综合验收

## 一句话结论

本轮验收状态为 **{status}**。当前 RNN 线检验的是单个间期群体事件内部的 rank-step history，而不是跨小时的发作倒计时。target-blind 选出的 `{selected}` 支持顺序增量，但 7 个预注册递归家族中只有这一个通过 family-wise inference，故跨架构稳定性和 early-ictal 条件增量均未建立。

## 1. 实际做了什么

1. 审计 34 人、864,163 个合格间期群体事件的绝对时间，以及 16 人 106 次 clinical-onset target 的配对关系。
2. 在完全不读取发作 target 的阶段，比较 static、unordered prefix、first-order、linear state、vanilla RNN、GRU 和预注册的 low-rank r0/r1/r2/r4。
3. 对最佳非 GRU 模型补做同架构 within-event rank shuffle，并对冻结状态执行 reverse、drop-first 和 rank 后 reset。
4. 模型和表示冻结后，才读取既有 `[0,10] s`、`1–150 Hz` early-ictal contact energy，检验 ordered field 是否在 static participation 与 unordered-prefix field 之外仍有增量。

## 2. 时间语义审计

- 现有模型每一步是同一个群体事件内的 recruitment rank set；每个事件开始时状态清零。
- 106 次发作只有 {pairing["strict_clinical_onset_metadata"]["n_distinct_causal_histories"]} 条不同的 causal pre-seizure histories，仅 {pairing["strict_clinical_onset_metadata"]["n_patients_with_at_least_3_distinct_histories"]} 位患者至少有 3 条不同历史。
- 因此 across-event history–seizure 分支没有被包装成 106 个独立样本；它不是本轮 primary。

## 3. 间期自监督架构结果

最佳非 GRU `{selected}` 相对 static、last-set 和 unordered-prefix 的患者中位 NLL 增益依次为 {fmt(selected_vs_static["median_gain"])}、{fmt(selected_vs_last_set["median_gain"])} 和 {fmt(selected_vs_unordered["median_gain"])}。其中相对 unordered-prefix 有 {selected_vs_unordered["n_positive"]}/34 患者方向为正，名义 Wilcoxon p={selected_vs_unordered["wilcoxon_greater_p"]:.3g}；正式解释使用下表跨 7 个预注册递归家族的 maxT 校正 P。

相对同架构 rank-shuffle，其真实顺序增益为 {fmt(selected_vs_shuffle["median_gain"])}，{selected_vs_shuffle["n_positive"]}/34 为正，名义 p={selected_vs_shuffle["wilcoxon_greater_p"]:.3g}；由于该架构先按 unordered 对照被挑中，这一项是 selection-aware sensitivity，不作为独立确认。GRU 相对 unordered 的中位增益为 {fmt(gru_vs_unordered["median_gain"])}，相对 rank-shuffle 为 {fmt(gru_vs_shuffle["median_gain"])}。

安全解释是：**事件内部真实顺序可被一个简单线性递归状态利用，而且不依赖 GRU 门控；但该增量尚未跨至少两个递归家族稳定复现。** 因此结果是 architecture-dependent 的序列证据，不是 hidden manifold 的独立脑相似性验证，也不能写成跨架构普遍规律。

{architecture_table}

## 4. 容量公平性敏感性

为避免把较小参数量误当成较弱的架构，本轮另以 GRU(h=32) 的 11,246 个参数为参照，补跑 linear-state(h=64) 与 vanilla-RNN(h=48)；两者参数量均在参照的 10% 以内。该分析只检查固定 hidden-size 结果是否受模型容量驱动，不参与 target-blind 模型选择。

{parameter_table}

## 5. 有效历史深度与历史干预

前一轮冻结的同架构历史窗口实验已经给出历史深度上限：H2 相对 H1 的中位 NLL 增益为 {fmt(history_necessity["contrasts"]["gain_history2_over_history1"]["median"])}（{history_necessity["contrasts"]["gain_history2_over_history1"]["n_positive"]}/34），H3 相对 H2 为 {fmt(history_necessity["contrasts"]["gain_history3_over_history2"]["median"])}（{history_necessity["contrasts"]["gain_history3_over_history2"]["n_positive"]}/34），但 full history 相对 H3 为 {fmt(history_necessity["contrasts"]["gain_full_over_history3"]["median"])}（P={history_necessity["contrasts"]["gain_full_over_history3"]["wilcoxon_two_sided_p"]:.3g}）。匹配的 ordered H3 相对 H3 rank-shuffle 增益为 {fmt(history3_shuffle["median_ordered_gain"])}（{history3_shuffle["n_positive"]}/34，P={history3_shuffle["wilcoxon_two_sided_p"]:.3g}）。

因此当前数据支持的是最近 2–3 个 rank set 的 bounded short history，而不是无界 full-history memory。

已在 34 人、3 seeds 上比较 ordered、reverse prefix、drop earliest，以及在第 1/2/3 个 rank set 后 reset（代码索引 0/1/2）。所有 eligible contact mask 始终由完整真实 prefix 决定，因此干预只改变进入 recurrent state 的历史，不会把已出现触点错误放回候选集。

这些结果回答“模型是否真的使用该段历史”，不能解释为生物恢复时间常数。

{intervention_table}

在 selected model 的真实 contact-logit readout 方向上，局部一步 retention 中位数为 {fmt(local_memory["readout_retention_median"]["median"])}，readout alignment 为 {fmt(local_memory["readout_alignment_median"]["median"])}，局部 Jacobian spectral radius 为 {fmt(local_memory["local_spectral_radius_median"]["median"])}。这些只是 rank-step 上的输出相关记忆诊断；不同架构的 hidden 坐标不可直接逐单元比较。

## 6. 低维结果的解释边界

既有 frozen hidden-state audit 中，ordered GRU 的 effective rank 中位数为 {fmt(internal_state["cohort_metrics"]["full_history_gru__effective_rank"]["median"])}，但 rank-shuffle GRU 同样只有 {fmt(internal_state["cohort_metrics"]["rank_shuffle_gru__effective_rank"]["median"])}。因此低维性不能单独支持“二维癫痫状态流形”；本轮把主要证据放在 heldout 顺序增量和显式历史干预上。

## 7. early-ictal 条件增量

在 static participation 与 unordered-prefix field 条件下，ordered field 的 absolute partial-r margin 患者中位数为 {fmt(early_primary.get("median", float("nan")))}，n={early_primary.get("n_patients", 0)}，p={early_primary.get("wilcoxon_greater_p", float("nan")):.3g}。

相对 matched rank-shuffle，ordered field 的 absolute partial-r 差值中位数为 {fmt(paired_early.get("median", float("nan")))}，{paired_early.get("n_positive", 0)}/{paired_early.get("n_patients", 0)} 为正，p={paired_early.get("wilcoxon_greater_p", float("nan")):.3g}。

两项都没有通过冻结门，因此 **ordered-history 对 early-ictal field 的条件增量未建立**。该 target 已在前序工作中读取，故即使结果为阳性也只能称为 reused-target internal validation。输出仍是患者固定的 contact field，不是逐次发作预测器。

## 8. 与论文核心目标的关系

- **未偏移**：输入仍为原始 SEEG 简化后的 contact-rank event；主问题仍是间期有序传播信息是否与发作早期静态能量招募场共享。
- **主动收窄**：不再把低 effective dimension、本身的二维轨迹或 hidden PC 解释为真实脑流形。
- **与 SNN 分工清楚**：RNN 只负责数据驱动的 history-state identification；SNN 单独检验局部抑制/慢变量机制能否生成相似的状态转移。
- **未新开 IEI 主线**：绝对时间只用于因果配对和伪重复审计，不作为预测输入。

## 9. 图与产物

- Paper-ready candidate：`results/paper-ready-figure/fig6_ordered_history_architecture_audit/figures/fig6_ordered_history_architecture_audit.png`
- 机器验收：`results/topic5_ordered_history_architecture_audit/FINAL_ACCEPTANCE.json`
- 测试与独立复算审计：`results/topic5_ordered_history_architecture_audit/TEST_AUDIT.json`
- 架构表、干预表与 early-ictal 条件统计：`results/topic5_ordered_history_architecture_audit/analysis/`
- 四个正式阶段均保存逐 20 秒资源日志；最终验收 JSON 汇总其最低可用内存、峰值显存、GPU 利用率和温度。

## 10. 最终用语边界

允许写：

> 在 target-blind 选出的线性递归模型中，有序间期 recruitment history 对 heldout next-contact prediction 提供了超越静态结构、last-set 和无序前缀的增量；该证据具有架构依赖性，对 early-ictal contact field 的条件增量未建立。

禁止写：

> RNN 发现了二维癫痫脑状态流形、恢复了真实 E/I 回路，或学得了连续时间发作倒计时。

## 11. 审阅批注与下一步

- **信息控制通过，但结论分层**：linear-state 同时超过 static、last-set、unordered 和同架构 rank-shuffle，说明单个事件内部的 recruitment order 不是普通参与频率的重述；但 1/7 家族通过意味着它尚不是跨架构稳定规律。
- **架构结论不是“需要 GRU”**：参数匹配的 linear-state 仍阳性，vanilla RNN 和 GRU 均未通过相同的 family-wise 标准。当前最简、最诚实的表示是 bounded linear event-indexed state，而不是更深的门控网络或强制 low-rank/Dale 约束。
- **early-ictal 主桥没有得到新增支持**：冻结 ordered residual 在 static + unordered 之外不超过 contact shuffle，也不超过 matched rank-shuffle。论文中可以保留既有 static contact-field correspondence，但不能把本轮序列状态写成其新增解释来源。
- **没有偏到 IEI 或发作倒计时**：所有训练状态都在事件边界清零；绝对时间只用于因果配对与伪重复审计。现有 106 次发作只有 46 条不同 causal histories，不足以把 across-event 分支当作 106 个独立预测样本。
- **停止继续刷当前模型**：不再用更多 seeds、hidden size、low-rank rank 或 loss 权重追 early-ictal 阳性。只有独立 clinical-onset cohort，或每位患者至少三条真正不同且可因果配对的 pre-seizure histories，才值得重新开放跨事件/跨状态训练。
"""
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(doc, encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
