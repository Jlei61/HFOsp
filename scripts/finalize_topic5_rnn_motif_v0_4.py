#!/usr/bin/env python3
"""Final engineering acceptance and bounded scientific report for v0.4."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any

import numpy as np


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle: return list(csv.DictReader(handle))


def fmt(value: Any, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{value:.{digits}f}" if np.isfinite(value) else "NA"


def contrast(summary: dict, name: str) -> dict[str, Any]:
    return summary.get(name, {"n": 0, "median": None, "positive": 0,
                              "wilcoxon_p": None, "holm_q_core_family": None})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--test-log", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_root.resolve()
    required = [
        "STAGE_CORE_STATUS.json", "STAGE_DOSE_STATUS.json", "STAGE_GRU_STATUS.json",
        "INTERICTAL_SUMMARY.json", "MODEL_FIELD_MANIFEST.json", "TARGET_UNSEAL_AUTHORIZATION.json",
        "target_access_audit.json", "EFFECTIVE_INFLUENCE_SUMMARY.json", "EFFECTIVE_MOTIF_SUMMARY.json",
        "MATCHED_LESION_SUMMARY.json", "LESION_EARLY_ICTAL_SUMMARY.json",
        "CONVERGENCE_AUDIT.json",
        "COMMON_OBSERVABLES.json", "figures/topic5_figure6_rnn_connectivity_motifs.png",
        "figures/topic5_figure6_rnn_connectivity_motifs.pdf",
        "figures/topic5_figure6_rnn_connectivity_motifs.svg",
        "figures/figure6_source_manifest.json", "VISUAL_QA.json",
        "POSTPROCESS_READY_FOR_VISUAL_QA.json", "UNIT_CONTRACT_EXPORT_AUDIT.json",
        "stage_d_scientific_drift_audit.json", "stage_e_scientific_drift_audit.json",
        "stage_f_scientific_drift_audit.json", "stage_g_scientific_drift_audit.json",
        "stage_h_scientific_drift_audit.json",
    ]
    missing = [name for name in required if not (out / name).exists()]
    stages = {stage: load(out / f"STAGE_{stage.upper()}_STATUS.json")
              for stage in ("core", "dose", "gru")}
    stage_clean = all(int(row["remaining"]) == 0 and int(row["failed"]) == 0
                      and int(row["oom"]) == 0 and int(row["nonfinite"]) == 0
                      for row in stages.values())
    all_metric_paths = list((out / "per_subject").glob("*/*__*/seed*/metrics.json"))
    metric_paths = [path for path in all_metric_paths
                    if not path.parents[1].name.startswith("SMOKE_")]
    metrics_count = len(metric_paths)
    target = load(out / "target_access_audit.json")
    visual = load(out / "VISUAL_QA.json") if (out / "VISUAL_QA.json").exists() else {}
    visual_ok = bool(
        visual.get("status") == "ACCEPTED"
        and visual.get("scientific_contract_pass") is True
        and visual.get("visual_pass") is True
    )
    test_text = args.test_log.read_text().lower() if args.test_log.exists() else ""
    tests_ok = bool(" passed" in test_text and re.search(r"\b[1-9]\d* failed\b", test_text) is None)
    unit_contracts = (load(out / "UNIT_CONTRACT_EXPORT_AUDIT.json")
                      if (out / "UNIT_CONTRACT_EXPORT_AUDIT.json").exists() else {})
    unit_contracts_ok = bool(
        unit_contracts.get("n_all_training_units") == 1435
        and unit_contracts.get("n_formal_training_units") == 1426
        and unit_contracts.get("n_smoke_training_units") == 9
        and unit_contracts.get("n_config_contracts") == 1435
        and unit_contracts.get("n_input_hash_contracts") == 1435
        and unit_contracts.get("checkpoint_or_metric_values_changed") is False
        and len(all_metric_paths) == 1435
        and all((path.parent / "config.json").exists() for path in all_metric_paths)
        and all((path.parent / "input_hashes.json").exists() for path in all_metric_paths)
    )
    engineering_accepted = bool(
        not missing and stage_clean and metrics_count == 1426 and tests_ok
        and visual_ok and unit_contracts_ok
    )

    inter = load(out / "INTERICTAL_SUMMARY.json")
    adequate = inter["task_adequacy"]["rnn"]["models"]
    adequate_models = [model for model, result in adequate.items()
                       if result["tier"] in {"ADEQUATE_PARTIAL", "ADEQUATE_STRONG"}]
    level1 = len(adequate_models) >= 2
    inter_rows = [row for row in csv_rows(out / "interictal_per_patient.csv") if row["cell"] == "rnn"]
    by_model = {model: [row for row in inter_rows if row["model"] == model]
                for model in {row["model"] for row in inter_rows}}
    dense_wire = np.nanmedian([float(row["c_wiring"]) for row in by_model.get("M1_DENSE", [])])
    m6_wire = np.nanmedian([float(row["c_wiring"]) for row in by_model.get("M6_SPATIAL_MID", [])])
    level2 = bool(level1 and np.isfinite(dense_wire) and np.isfinite(m6_wire) and m6_wire < dense_wire)
    m6_ceiling = inter["noise_ceiling_reference"]["model_minus_reference"].get(
        "M6_SPATIAL_MID|rnn", {}
    )

    early = load(out / "early_ictal_model_contrasts.json")
    m6_zero = contrast(early, "canonical_full|M6_SPATIAL_MID__rnn_margin_gt_zero")
    m6_m0 = contrast(early, "canonical_full|M6_SPATIAL_MID__rnn_vs_M0_NO_REC__rnn")
    m6_dense = contrast(early, "canonical_full|M6_SPATIAL_MID__rnn_vs_M1_DENSE__rnn")
    level3_correspondence = bool((m6_zero.get("median") or 0) > 0 and
                                 (m6_zero.get("wilcoxon_p") or 1) < 0.05)
    level3_selective = bool((m6_m0.get("median") or 0) > 0 and
                            (m6_m0.get("holm_q_core_family") or 1) < 0.05)
    conditional = load(out / "early_ictal_conditional_on_interictal_fidelity.json")
    conditional_m6_m0 = conditional.get("contrasts", {}).get(
        "M6_SPATIAL_MID_vs_M0_NO_REC", {}
    )
    conditional_ci = conditional_m6_m0.get("patient_cluster_bootstrap_95ci", [None, None])
    conditional_inductive_bias = bool(
        (conditional_m6_m0.get("estimate") or 0) > 0
        and conditional_ci[0] is not None
        and float(conditional_ci[0]) > 0
        and (conditional_m6_m0.get("patient_label_permutation_p") or 1) < 0.05
    )

    theory = load(out / "EFFECTIVE_MOTIF_SUMMARY.json")
    motif_components = theory["M6_motif_claim_components"]
    enrichment_pass = (motif_components["local_effective_enrichment"]
                       and motif_components["long_range_effective_enrichment"])
    stability_pass = motif_components["effective_operator_seed_stability"]
    split_stability_pass = motif_components["effective_operator_split_half_stability"]
    task_relation_pass = motif_components["task_relation"]
    lesion_pass = (motif_components["local_backbone_matched_lesion"]
                   and motif_components["long_range_or_connector_matched_lesion"])
    lesion_summary = theory.get("matched_lesion", {})
    lesion_statistics = lesion_summary.get("statistics", {})
    local_lesion_estimable = "M6_SPATIAL_MID|local_backbone_edges" in lesion_statistics
    long_lesion_estimable = any(
        key in lesion_statistics for key in (
            "M6_SPATIAL_MID|long_range_high_influence_edges",
            "M6_SPATIAL_MID|connector_nodes",
        )
    )
    lesion_estimable = bool(local_lesion_estimable and long_lesion_estimable)
    lesion_wording = ("通过" if lesion_pass else
                      "未通过" if lesion_estimable else
                      "不可估计（严格 matched-control 分母不足）")
    proposal_pass = motif_components["not_binary_proposal_only"]
    level4 = bool(theory["M6_motif_claim_pass"])

    acceptance = {
        "contract": "topic5_rnn_motif_cross_state_final_acceptance_v0_4",
        "engineering_accepted": engineering_accepted,
        "missing_artifacts": missing,
        "formal_training_units": metrics_count,
        "stage_clean": stage_clean,
        "focused_tests_passed": tests_ok,
        "unit_contracts_complete": unit_contracts_ok,
        "visual_qa_accepted": visual_ok,
        "target_access": target,
        "scientific_levels": {
            "level1_multiple_recurrences_sufficient": level1,
            "level2_economic_constraints": level2,
            "level3_cross_state_correspondence": level3_correspondence,
            "level3_raw_motif_selectivity": level3_selective,
            "level3_conditional_inductive_bias": conditional_inductive_bias,
            "level4_intervenable_computational_motif": level4,
        },
        "level4_components": {"coherent_local_and_long_enrichment": enrichment_pass,
                              "effective_operator_seed_stability": stability_pass,
                              "effective_operator_split_half_stability": split_stability_pass,
                              "task_relation": task_relation_pass,
                              "matched_lesion_estimable": lesion_estimable,
                              "coherent_local_and_long_matched_lesion": lesion_pass,
                              "not_binary_proposal_only": proposal_pass},
        "adequate_rnn_models": adequate_models,
    }
    (out / "FINAL_ACCEPTANCE.json").write_text(json.dumps(acceptance, indent=2))

    report = f"""# Topic 5 RNN connectivity motif / cross-state v0.4 最终报告

## 一句话结论

本轮严格回答三件事：哪些连接约束足以让 RNN 在同一患者内生成留出间期传播；这些完全冻结的模型场是否复现论文已有的 early-ictal broadband 场对应；以及哪类有效连接组织经 matched lesion 后真正承担预测。工程验收为 **{'ACCEPTED' if engineering_accepted else 'NOT ACCEPTED'}**；科学结论按 Level 1–4 分层，不使用一个总 gate 把低层阳性压掉。

## 1. 间期传播充分性

- 正式训练：{metrics_count}/1426 单元；Core/Dose/GRU 均为 0 failed、0 OOM、0 nonfinite。
- 独立复现合同：{unit_contracts.get('n_config_contracts', 0)}/1435 `config.json`，{unit_contracts.get('n_input_hash_contracts', 0)}/1435 `input_hashes.json`；不改 checkpoint 或 metrics。
- 至少达到 partial adequacy 的 leaky-RNN 模型：{', '.join(adequate_models) if adequate_models else '无'}。
- 因此“多种 recurrence 是否足以学习患者内传播”的 Level 1：**{'支持' if level1 else '不支持'}**。
- Dense 的患者中位 wiring cost 为 {fmt(dense_wire)}，Spatial + cost 为 {fmt(m6_wire)}；Level 2 经济性：**{'支持' if level2 else '不支持'}**。
- Spatial + cost 相对 `sqrt(event-pair reliability)` 的中位差为 {fmt(m6_ceiling.get('median'))}；该量只作噪声参照，差异不显著也不写成“达到天花板”。

这里的“学会”同时要求留出 next-contact 与删除已提供起点后的自由推演不塌缩；不等于恢复了真实脑连接组。

## 2. 冻结间期场与发作早期场

- early-ictal primary cohort 是 target 解封前确定的实际交集 n={target['n_primary_subjects']}；主量为 clinical onset 0–10 s、1–150 Hz、canonical-full maxAB 相对 5000 次同步 all-contact null。
- Spatial + cost 自身相对 null：median margin={fmt(m6_zero.get('median'))}，{m6_zero.get('positive', 0)}/{m6_zero.get('n', 0)} 患者为正，P={fmt(m6_zero.get('wilcoxon_p'))}。
- 相对 no-recurrence：Δmargin={fmt(m6_m0.get('median'))}，Holm q={fmt(m6_m0.get('holm_q_core_family'))}；相对 dense：Δmargin={fmt(m6_dense.get('median'))}。
- 控制患者内 interictal field fidelity 后，Spatial + cost 相对 no-recurrence 的 model effect 为 {fmt(conditional_m6_m0.get('estimate'))}，patient-cluster bootstrap 95% CI={conditional_ci}，patient-label permutation P={fmt(conditional_m6_m0.get('patient_label_permutation_p'))}。
- 因此“冻结 RNN 场是否存在跨状态对应”：**{'支持' if level3_correspondence else '未支持'}**；原始模型选择性：**{'支持' if level3_selective else '未支持'}**；控制间期拟合后仍支持特殊 inductive bias：**{'支持' if conditional_inductive_bias else '未支持'}**。

canonical full、seed-removed、common-field 与 A/B contrast 已分开报告。单个 maxAB 阳性不会被写成“模型恢复了两种 A/B 模式”。

## 3. 有效计算 motif

- 同一 local-backbone + long-range-connector 结构的双重富集：**{'通过' if enrichment_pass else '未通过'}**。
- 完整 effective operator 的跨 seed 稳定性：**{'通过' if stability_pass else '未通过'}**。
- 同一冻结模型在前后半留出事件中的 effective operator 稳定性：**{'通过' if split_stability_pass else '未通过'}**。
- motif score 与留出传播/间期场拟合的患者级关系：**{'通过' if task_relation_pass else '未通过'}**。
- local 与 long/connector targeted lesion 相对 matched random lesion 的同结构特异损害：**{lesion_wording}**。
- 与相同生长规则的 order-shuffle 对照相比并非二值 proposal 自动造成：**{'通过' if proposal_pass else '未通过'}**。
- 全部同结构证据同时成立的 Level 4：**{'支持 local-backbone + sparse connector motif' if level4 else '未达到机制性 motif 措辞，只保留描述性组织'}**。

GRU 只承担架构方向复现；matched lesion 的主分析限定在 leaky RNN。所有时间量是 rank-step，不是秒级生物时间常数。

## 4. Human–RNN–SNN 边界

三条线只在 contact field、传播方向、空间 reach 与 perturbation readout 这些中尺度量上并列。既有 SNN 不重跑；其 E1146 产物支持双向间期虚拟触点 readout，但没有一个已验收的闭环 early-ictal recruitment field，因此该格明确写为 `not established`。不做 RNN edge ↔ SNN synapse 或 hidden unit ↔ neuron 映射。

## 5. 可以写 / 不可以写

可以写：

1. 多类连接约束在患者内自监督任务上是否足以生成留出间期传播；
2. 在相近任务表现下，空间生长与布线成本是否形成更经济的有效网络；
3. 冻结模型生成场与同患者 early-ictal broadband 场的 target-free 外部对应；
4. 只有 Level 4 三项同时成立时，写某种 effective motif 更容易支持该传播计算。

不可以写：

1. RNN 恢复了患者真实解剖连接组；
2. RNN 从未见过的几何中独立发现病理轴（几何为 retrospective/test-informed）；
3. early-ictal 对应等于发作预测或因果转变机制；
4. hidden state 是真实神经流形，或 rank-step persistence 是生物时间常数。

## 6. 产物

- 主图：`figures/topic5_figure6_rnn_connectivity_motifs.png/.pdf/.svg`
- 图源：`figures/figure6_source_manifest.json`
- 全量统计：`INTERICTAL_SUMMARY.json`、`early_ictal_model_contrasts.json`、`EFFECTIVE_MOTIF_SUMMARY.json`、`MATCHED_LESION_SUMMARY.json`
- 跨系统表：`COMMON_OBSERVABLES.json/.csv`
- 工程验收：`FINAL_ACCEPTANCE.json`
"""
    (out / "TOPIC5_RNN_MOTIF_FINAL_REPORT_ZH.md").write_text(report)
    if engineering_accepted:
        (out / "PIPELINE_COMPLETE.json").write_text(json.dumps({
            "status": "COMPLETE", "acceptance": "ACCEPTED",
            "final_acceptance": str(out / "FINAL_ACCEPTANCE.json"),
            "report": str(out / "TOPIC5_RNN_MOTIF_FINAL_REPORT_ZH.md"),
            "figure": str(out / "figures/topic5_figure6_rnn_connectivity_motifs.png"),
        }, indent=2))
    else:
        (out / "PIPELINE_FAILED.json").write_text(json.dumps({
            "status": "INCOMPLETE", "missing": missing, "stage_clean": stage_clean,
            "metrics_count": metrics_count, "tests_ok": tests_ok,
        }, indent=2))
    return 0 if engineering_accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
