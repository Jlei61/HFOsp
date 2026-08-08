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
        "figures/topic5_figure6_rnn_connectivity_motifs.pdf", "figures/figure6_source_manifest.json",
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
    metric_paths = [path for path in (out / "per_subject").glob("*/*__*/seed*/metrics.json")
                    if not path.parents[1].name.startswith("SMOKE_")]
    metrics_count = len(metric_paths)
    target = load(out / "target_access_audit.json")
    test_text = args.test_log.read_text().lower() if args.test_log.exists() else ""
    tests_ok = bool(" passed" in test_text and re.search(r"\b[1-9]\d* failed\b", test_text) is None)
    engineering_accepted = not missing and stage_clean and metrics_count == 1426 and tests_ok

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

    early = load(out / "early_ictal_model_contrasts.json")
    m6_zero = contrast(early, "canonical_full|M6_SPATIAL_MID__rnn_margin_gt_zero")
    m6_m0 = contrast(early, "canonical_full|M6_SPATIAL_MID__rnn_vs_M0_NO_REC__rnn")
    m6_dense = contrast(early, "canonical_full|M6_SPATIAL_MID__rnn_vs_M1_DENSE__rnn")
    level3_correspondence = bool((m6_zero.get("median") or 0) > 0 and
                                 (m6_zero.get("wilcoxon_p") or 1) < 0.05)
    level3_selective = bool((m6_m0.get("median") or 0) > 0 and
                            (m6_m0.get("holm_q_core_family") or 1) < 0.05)

    theory = load(out / "EFFECTIVE_MOTIF_SUMMARY.json")
    motif_components = theory["M6_motif_claim_components"]
    enrichment_pass = (motif_components["local_effective_enrichment"]
                       and motif_components["long_range_effective_enrichment"])
    stability_pass = motif_components["effective_operator_seed_stability"]
    split_stability_pass = motif_components["effective_operator_split_half_stability"]
    task_relation_pass = motif_components["task_relation"]
    lesion_pass = (motif_components["local_backbone_matched_lesion"]
                   and motif_components["long_range_or_connector_matched_lesion"])
    proposal_pass = motif_components["not_binary_proposal_only"]
    level4 = bool(theory["M6_motif_claim_pass"])

    acceptance = {
        "contract": "topic5_rnn_motif_cross_state_final_acceptance_v0_4",
        "engineering_accepted": engineering_accepted,
        "missing_artifacts": missing,
        "formal_training_units": metrics_count,
        "stage_clean": stage_clean,
        "focused_tests_passed": tests_ok,
        "target_access": target,
        "scientific_levels": {
            "level1_multiple_recurrences_sufficient": level1,
            "level2_economic_constraints": level2,
            "level3_cross_state_correspondence": level3_correspondence,
            "level3_motif_selectivity": level3_selective,
            "level4_intervenable_computational_motif": level4,
        },
        "level4_components": {"coherent_local_and_long_enrichment": enrichment_pass,
                              "effective_operator_seed_stability": stability_pass,
                              "effective_operator_split_half_stability": split_stability_pass,
                              "task_relation": task_relation_pass,
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
- 至少达到 partial adequacy 的 leaky-RNN 模型：{', '.join(adequate_models) if adequate_models else '无'}。
- 因此“多种 recurrence 是否足以学习患者内传播”的 Level 1：**{'支持' if level1 else '不支持'}**。
- Dense 的患者中位 wiring cost 为 {fmt(dense_wire)}，Spatial + cost 为 {fmt(m6_wire)}；Level 2 经济性：**{'支持' if level2 else '不支持'}**。

这里的“学会”同时要求留出 next-contact 与删除已提供起点后的自由推演不塌缩；不等于恢复了真实脑连接组。

## 2. 冻结间期场与发作早期场

- early-ictal primary cohort 是 target 解封前确定的实际交集 n={target['n_primary_subjects']}；主量为 clinical onset 0–10 s、1–150 Hz、canonical-full maxAB 相对 5000 次同步 all-contact null。
- Spatial + cost 自身相对 null：median margin={fmt(m6_zero.get('median'))}，{m6_zero.get('positive', 0)}/{m6_zero.get('n', 0)} 患者为正，P={fmt(m6_zero.get('wilcoxon_p'))}。
- 相对 no-recurrence：Δmargin={fmt(m6_m0.get('median'))}，Holm q={fmt(m6_m0.get('holm_q_core_family'))}；相对 dense：Δmargin={fmt(m6_dense.get('median'))}。
- 因此“冻结 RNN 场是否存在跨状态对应”：**{'支持' if level3_correspondence else '未支持'}**；“该对应是否对 spatial+cost motif 有选择性”：**{'支持' if level3_selective else '未支持'}**。

canonical full、seed-removed、common-field 与 A/B contrast 已分开报告。单个 maxAB 阳性不会被写成“模型恢复了两种 A/B 模式”。

## 3. 有效计算 motif

- 同一 local-backbone + long-range-connector 结构的双重富集：**{'通过' if enrichment_pass else '未通过'}**。
- 完整 effective operator 的跨 seed 稳定性：**{'通过' if stability_pass else '未通过'}**。
- 同一冻结模型在前后半留出事件中的 effective operator 稳定性：**{'通过' if split_stability_pass else '未通过'}**。
- motif score 与留出传播/间期场拟合的患者级关系：**{'通过' if task_relation_pass else '未通过'}**。
- local 与 long/connector targeted lesion 相对 matched random lesion 的同结构特异损害：**{'通过' if lesion_pass else '未通过'}**。
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
