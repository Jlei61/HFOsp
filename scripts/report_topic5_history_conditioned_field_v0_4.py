#!/usr/bin/env python3
"""Write the plain-language scientific closeout report for Topic 5 v0.4."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _comparison_line(name: str, result: dict) -> str:
    low, high = result["bootstrap_95ci_median"]
    return (
        f"- **{name}**：患者中位差 {result['median_delta']:+.4f}，"
        f"bootstrap 95% CI [{low:+.4f}, {high:+.4f}]；"
        f"{result['n_positive']} 正 / {result['n_negative']} 负 / {result['n_tie']} 并列，"
        f"exact P={result['p_two_sided_exact']:.4g}（n={result['n_patients']}）。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "results/topic5_history_conditioned_field_refinement_v0_4",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    summary = json.loads((root / "HISTORY_CONDITIONED_FIELD_SUMMARY.json").read_text())
    acceptance = json.loads((root / "ACCEPTANCE.json").read_text())
    patient = pd.read_csv(root / "history_conditioned_field_patient_metrics.csv")
    static = pd.read_csv(root / "static_anchor_patient_metrics.csv")
    comparisons = summary["comparisons"]
    primary = comparisons["primary_m3_minus_m0"]
    m3_m1 = comparisons["m3_minus_m1"]
    m3_m2 = comparisons["m3_minus_m2"]
    order = comparisons["true_minus_order_shuffle"]
    swap = comparisons["correct_minus_history_swap"]

    if primary["median_delta"] <= 0:
        plain_conclusion = "联合 RNN 残差没有在患者级改善已经有效的静态 A/B 场。"
    elif m3_m1["median_delta"] <= 0:
        plain_conclusion = "历史分支带来改善，但冻结的 target-blind state 已经足够，联合微调 recurrent dynamics 没有额外收益。"
    elif m3_m2["median_delta"] <= 0:
        plain_conclusion = "历史分支带来改善，但固定时间汇总已经足够，当前结果不要求 recurrence。"
    elif swap["median_delta"] <= 0:
        plain_conclusion = "联合 RNN 改善了静态 A/B 场，但尚不能证明该增量匹配到具体一次发作的历史。"
    elif order["median_delta"] <= 0:
        plain_conclusion = "联合 RNN 的改善与具体发作历史匹配，但主要来自历史内容或负荷，而不是事件的真实先后顺序。"
    else:
        plain_conclusion = "联合 RNN 在静态 A/B 场之外利用了与具体发作匹配、且依赖真实顺序的间期历史信息。"

    diagnostic = summary["training_diagnostics"]["heldout_state_and_field_change"]
    static_median = float(static.observed_patient_median_maxab_1_45.median())
    static_margin = float(static.observed_minus_null_median.median())
    static_positive = int(np.sum(static.observed_minus_null_median > 1e-9))
    static_group = summary["static_anchor_reproduction"]
    m3_null = summary["matched_channel_null"]["M3_JOINT_RNN"]
    sensitivity = comparisons["sensitivity_1_150_m3_minus_m0"]
    report = f"""# Topic 5：history-conditioned early-ictal field refinement v0.4 结果报告

## 一句话结论

{plain_conclusion}

这句话只描述本轮 15 位 strict clinical-onset 患者的回顾性 LOSO 结果，不把它升级成唯一发作方向、发作时间预警或因果机制。

## 1. 这轮实际测了什么

本轮先冻结论文已经有效的患者特异静态 A/B 间期病理场，再问：发作前、同一连续记录段内的间期事件历史，能否通过一个受限残差支路，把这两个静态候选场修正得更接近该次发作 clinical onset 后 0–10 s 的 **1–45 Hz contact-energy field**。正式评分仍为两个候选中较高的绝对 Spearman（maxAB），不要求模型选择唯一 A/B 或正负号。

训练和评价严格按患者外层留一：每个 fold 的 event encoder 与 HistoryGRU 均未见过 heldout patient；14 位训练患者提供 early-ictal supervision，heldout patient target 只在最终评分时读取。三个固定 seed 的 A/B candidate fields 先逐 contact 平均，再计算 seizure maxAB，随后按 patient median 做队列统计。

## 2. 静态 A/B 锚点

- 15 位患者、31 次发作；每位患者的评分场覆盖 6–16 个触点，中位 9 个。
- M0 的患者中位 maxAB 为 **{static_median:.4f}**。
- 相对每患者 5000 次 matched all-contact channel shuffle，患者中位 margin 为 **{static_margin:+.4f}**；{static_positive}/15 位高于各自 null median，{int(static.pass_null_p95.sum())}/15 位超过各自 p95；患者级 exact P={static_group['p_two_sided_exact']:.4g}。
- 因此，本轮不是让 RNN 从零生成发作场，而是在一个已存在信息的静态 A/B 基底上检验历史增量。

## 3. 四个模型

- **M0**：冻结 static A/B，不训练。
- **M1**：冻结 target-blind HistoryGRU，只训练 contact-query residual heads 与两个共享 gain；回答已有状态是否已经可读。
- **M2**：不用 recurrence，只使用固定 2 h EWMA、历史 mean/max、last event、event count 和 history span；回答简单历史内容/负荷是否足够。
- **M3**：从与 M1 完全相同的 30-epoch head checkpoint 分叉，再联合微调 HistoryGRU/decay 与 head 30 epochs；回答 early-ictal supervision 是否需要改变 recurrent dynamics。

所有模型共用同一静态 A/B、contact denominator、1–45 Hz target、patient-first loss 与 LOSO split。没有 architecture sweep、best-seed 选择或根据 heldout target early stopping。

## 4. Primary 和模型解释对比

{_comparison_line('Primary，M3−M0', primary)}
{_comparison_line('M3−M1', m3_m1)}
{_comparison_line('M3−M2', m3_m2)}
{_comparison_line('M1−M0', comparisons['m1_minus_m0'])}
{_comparison_line('M2−M0', comparisons['m2_minus_m0'])}

这些对比分开报告，没有复合 hard gate。M3−M0 只回答“联合 history residual 是否改善静态场”；M3−M1 和 M3−M2 才决定这种改善是否需要改变 recurrent dynamics、以及是否超过简单时间汇总。

## 5. 历史是否真有特异性

{_comparison_line('M3 true order−完整历史顺序打乱', order)}
{_comparison_line('M3 correct history−同患者其他发作 history swap', swap)}

顺序对照对整段 causal history 做事件身份置换并保留原时间槽，每个 seed 32 次；不是只洗最近 64 个事件。History-swap 保持患者、静态 A/B、contact set 和 target 不变，只替换成同患者另一场发作的历史；只有一场合格发作的患者不进入这一对比。

## 6. 绝对信息和频带敏感性

- M3 的患者中位 observed maxAB 为 **{m3_null['median_observed']:.4f}**，matched channel-null 中位为 **{m3_null['median_null']:.4f}**，中位 margin **{m3_null['median_margin']:+.4f}**；{m3_null['n_above_null_median']}/15 位高于 null median，{m3_null['n_above_null_p95']}/15 位超过 p95。
- 1–150 Hz 只做 no-retrain sensitivity：M3−M0 患者中位差 **{sensitivity['median_delta']:+.4f}**，{sensitivity['n_positive']} 正 / {sensitivity['n_negative']} 负 / {sensitivity['n_tie']} 并列。它没有参与模型、seed 或超参数选择。

## 7. 模型内部实际改了多少

- M1 heldout 状态范数中位 {diagnostic['M1_FROZEN_HISTORY_HEAD']['state_norm_median']:.4f}；candidate A/B 相对 static 的夹角中位分别为 {diagnostic['M1_FROZEN_HISTORY_HEAD']['candidate_angle_a_median_degrees']:.2f}° / {diagnostic['M1_FROZEN_HISTORY_HEAD']['candidate_angle_b_median_degrees']:.2f}°。
- M2 candidate A/B 相对 static 的夹角中位分别为 {diagnostic['M2_TIME_AWARE_NONRECURRENT']['candidate_angle_a_median_degrees']:.2f}° / {diagnostic['M2_TIME_AWARE_NONRECURRENT']['candidate_angle_b_median_degrees']:.2f}°。
- M3 heldout 状态范数中位 {diagnostic['M3_JOINT_RNN']['state_norm_median']:.4f}；candidate A/B 相对 static 的夹角中位分别为 {diagnostic['M3_JOINT_RNN']['candidate_angle_a_median_degrees']:.2f}° / {diagnostic['M3_JOINT_RNN']['candidate_angle_b_median_degrees']:.2f}°。
- M3 学到的 rank-step/clock-time decay 在这里只是模型记忆参数，不解释为细胞级 E/I 时间常数。

这些量用于确认模型做的是“受限修正”还是重写静态场，不把 raw hidden coordinate 当作唯一神经流形。

## 8. 工程验收和资源

- 工程验收：**{acceptance['status']}**；{acceptance['formal_units_complete']}/45 单元完成，失败 {acceptance['failed_units']}。
- 训练日志固定为每单元 150 行：common head 30、M1 continuation 30、M3 joint 30、M2 60。
- 最大单进程显存 {acceptance['resource_summary']['peak_gpu_memory_mb_max']:.1f} MB；单元中位运行时间 {acceptance['resource_summary']['elapsed_seconds_median']/60:.1f} min。
- 验收只检查 leakage、outer-fold 坐标、训练预算、有限数值、控制分母和 artifact 完整性；科学效应大小不是工程 gate。

## 9. 科学边界

1. 静态 A/B 不读取 early-ictal target，但由患者全记录间期事件回顾性估计，可能包含目标发作之后的事件。因此 residual-history 支路严格 causal，整体模型却不是完全前瞻预测器。
2. 输出是 A/B 两候选的无符号集合，不是唯一发作方向。
3. 该任务预测发作早期空间场，不预测发作何时发生。
4. 只有 correct-history 超过 within-patient swap，才把增量称为 seizure-matched；只有 true-order 超过整段 shuffle，才称为顺序特异。
5. 6–16 个触点的评分分辨率较粗，患者级精确并列是预期现象，统计已使用 1e-9 tie band 和 exact sign-rank null。

## 10. 产物

- 六联图：`figures/history_conditioned_field_refinement_six_panel.png` 与同名 PDF。
- 中位效应代表病例：`figures/representative_history_refinement.png` 与同名 PDF。
- 患者级结果：`history_conditioned_field_patient_metrics.csv`。
- 5000-draw null：`history_conditioned_field_channel_null.csv`。
- raw state/residual：`history_conditioned_field_state_diagnostics.csv.gz`。
- 正式汇总：`HISTORY_CONDITIONED_FIELD_SUMMARY.json`。
- 工程验收：`ACCEPTANCE.json`。
- 可复现清单：`REPRODUCIBILITY_MANIFEST.json`。
"""
    output = root / "HISTORY_CONDITIONED_FIELD_REPORT.md"
    output.write_text(report, encoding="utf-8")
    reproduction_path = root / "REPRODUCIBILITY_MANIFEST.json"
    if reproduction_path.exists():
        reproduction = json.loads(reproduction_path.read_text())
        reproduction["files"][str(output.relative_to(ROOT))] = {
            "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
        reproduction_path.write_text(
            json.dumps(reproduction, ensure_ascii=False, indent=2) + "\n"
        )
    print(output)


if __name__ == "__main__":
    main()
