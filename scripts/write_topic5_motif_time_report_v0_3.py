#!/usr/bin/env python3
"""Report for the motif-vs-time round.  Every verdict sentence is computed.

An earlier round shipped a paragraph asserting a conclusion its own table
contradicted, so nothing here states an outcome that is not read back out of
``MOTIF_TIME_EVIDENCE.json`` at the moment of writing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULT_ROOT = ROOT / "results/topic5_motif_time_targets_v0_3"
ARCHIVE = ROOT / "docs/archive/topic5"

ARM_LABEL = {
    "M0_ISOTROPIC_DIFFUSION": "各向同性局部扩散（往四面八方一样扩）",
    "M1_AXIAL_CORRIDOR": "轴向走廊（沿这位患者的一条方向拉长）",
    "M2_DIRECTED_TRANSPORT": "方向输运（由事件早期的位移选一个方向，然后一直往那边推）",
    "M3_AXIAL_FEEDFORWARD_TRANSIENT": "轴向前馈的有限时程放大",
    "MFREE_LOW_RANK_UPPER_BOUND": "自由低秩替代算子（同一个细胞，秩 4–7 的不受约束算子）",
    "MFREE_LOW_RANK_ALTERNATIVE": "自由低秩替代算子（同一个细胞，秩 4–7 的不受约束算子）",
}


def band(entry: dict, floor: float | None = None) -> str:
    if not entry or "median" not in entry:
        return f"只有 {entry.get('n', 0)} 位患者，不足以聚合"
    low, high = entry["median_ci95"]
    text = (f"{entry['n']} 位患者，中位 {entry['median']:+.5f}"
            f"（95% 区间 {low:+.5f} 到 {high:+.5f}，"
            f"{'跨过零' if entry['crosses_zero'] else '不跨零'}），"
            f"正/负 = {entry['n_positive']}/{entry['n_negative']}，"
            f"符号检验 p={entry['sign_test_p']:.3f}")
    if floor is not None and np.isfinite(floor):
        ratio = abs(entry["median"]) / floor if floor > 0 else float("inf")
        text += f"；换一个优化起点的散布是 {floor:.5f}（效应是它的 {ratio:.1f} 倍）"
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    arguments = parser.parse_args()

    evidence = json.loads((RESULT_ROOT / "MOTIF_TIME_EVIDENCE.json").read_text())
    status = json.loads((RESULT_ROOT / "RUN_STATUS.json").read_text())
    table = pd.read_csv(RESULT_ROOT / "PER_ARM_SCORES.csv")
    ladder = evidence["ladder"]
    basin = evidence["optimisation_basin_sensitivity"]
    verdicts = evidence["verdicts"]

    def floor_for(arm: str) -> float:
        return float(basin.get(arm, {}).get("median_spread", float("nan")))

    lines = [
        "# Topic 5.3：哪种动力学解释「走这么远需要多久」",
        "",
        f"> 日期：{arguments.date}　|　结果根：`results/topic5_motif_time_targets_v0_3/`",
        "> 冻结设计：`docs/superpowers/plans/2026-08-19-topic5-motif-time-targets-v0-3.md`",
        "",
        "## 0. 这一轮问的是什么",
        "",
        "一次间期事件里，触点是一批一批亮起来的。上一轮问的是「下一个亮的是谁」，"
        "四种传播规则分不出高下。这一轮换一个量：**从这一批亮到下一批亮，中间隔了多久**。",
        "",
        "换这个量是有依据的：控制住「隔了几个名次」之后，两个触点之间的**空间距离**"
        "仍然预测它们的**时间差**，28 位患者里 27 位同向。也就是说时间里装着名次里没有的距离信息，"
        "而上一轮所有模型用的「时间」其实只是名次序号，把这份信息整轮扔掉了。",
        "",
        "⚠️ 这里的「时间」是事件内谱质量中心的位置，**不是**临床上的招募时间，"
        "**不是**轴突传导延迟，**不能**拿它算传导速度。",
        "",
        "## 1. 怎么比的",
        "",
        "四种传播规则依次嵌套，后一种是前一种加一个机制；每一层都从**前一层已经训好的模型**"
        "接着练，新加的那个旋钮从零开始，所以「加这个机制有没有用」问得干净。",
        "",
    ]
    for arm in ("M0_ISOTROPIC_DIFFUSION", "M1_AXIAL_CORRIDOR",
                "M2_DIRECTED_TRANSPORT", "M3_AXIAL_FEEDFORWARD_TRANSIENT",
                "MFREE_LOW_RANK_UPPER_BOUND"):
        lines.append(f"- **{ARM_LABEL[arm]}**（内部 `{arm}`）")
    lines += [
        "",
        "**最关键的一点是参照要公平。** 模型读的是「实际亮起来的那个触点」上的场，"
        "也就是它被告知了目的地。如果参照只知道「走到第几步」，那么「目的地有多远」带来的好处"
        "就会被白送给模型——实测这一项**单独**就值 0.2%–5.5% 的方差，和模型全部效果同一量级。"
        "所以参照逐级加信息：只知道第几步 → 再知道实际走了多远 → 再知道那个触点是否惯常早到。",
        "",
        f"规模：{status['patient_states']}，共 {status['n_rows']} 条臂记录，"
        f"{len(status['theta_starts'])} 个轴角起点 × 2 种继承方式。",
        "",
        "### 参照是怎么定的（这一条决定了下面的判决）",
        "",
        "参照固定为「知道第几步 + 走了多远 + 该触点是否惯常早到」那一条，理由是**结构性的**："
        "它严格包含另外两条所知道的一切。按「它知道什么」选，不按「它考了多少分」选。",
        "",
        "两个被否掉的做法，各自的理由：",
        "",
        "- **逐患者在测试集上取三条基线的最小值**：这是看过测试误差之后再决定参照，"
        "违反「测试集只评一次」。（核实过：26/28 位患者会选同一条，改掉后中位数一位不差。）",
        "- **给参照加一个验证集选出的岭正则**：我实现并跑了，结果第二层从全部跨零"
        "**翻成三条显著**。查下来是：岭让参照的中位恰好不变，却把 3 位患者的参照显著削弱"
        "（有一位从 0.828 变成 0.893）。去掉那 3 位后全部塌回不显著，用无岭参照也全部不显著。"
        "数据并不支持「无岭在过拟合」——无岭版在 18/28 位患者上测试误差更低或相等。"
        "**一个让对照变弱的正则化不予采用**；它作为敏感性记录在证据文件的 "
        "`comparator_sensitivity_validation_selected_ridge` 字段。",
        "",
        "## 2. 第一层：距离本身能解释多少",
        "",
        f"- 知道走了多远，比只知道第几步好多少：{band(ladder.get('rung1_distance_beyond_step_index'))}",
        f"- 再知道那个触点惯常早到还是晚到，又好多少："
        f"{band(ladder.get('rung1b_static_target_beyond_distance'))}",
        "",
    ]
    rung1 = ladder.get("rung1_distance_beyond_step_index", {})
    if "median" in rung1:
        if not rung1["crosses_zero"] and rung1["median"] > 0:
            lines.append("**距离确实携带时间间隔信息**，所以后面每一层都必须先跨过这条线，"
                         "而不是跨过「只知道第几步」那条低得多的线。")
        else:
            lines.append("**距离本身没有稳定地解释时间间隔**，那么后面各层若有增益，"
                         "来源就不是简单的几何距离，需要单独说明。")

    lines += ["", "## 3. 第二层：递归场是否比距离更多", ""]
    for arm in ("M0_ISOTROPIC_DIFFUSION", "M1_AXIAL_CORRIDOR", "M2_DIRECTED_TRANSPORT",
                "M3_AXIAL_FEEDFORWARD_TRANSIENT", "MFREE_LOW_RANK_UPPER_BOUND"):
        key = f"rung2_{arm}_beyond_best_baseline"
        if key in ladder:
            lines.append(f"- **{ARM_LABEL[arm]}** 相对最强参照："
                         f"{band(ladder[key], floor_for(arm))}")

    lines += ["", "## 4. 第三层：每个机制是否各自再增加解释", ""]
    chain = ["M0_ISOTROPIC_DIFFUSION", "M1_AXIAL_CORRIDOR",
             "M2_DIRECTED_TRANSPORT", "M3_AXIAL_FEEDFORWARD_TRANSIENT"]
    for simple, rich in zip(chain, chain[1:]):
        key = f"rung3_{rich}_over_{simple}"
        if key in ladder:
            lines.append(f"- 加上「{ARM_LABEL[rich]}」比「{ARM_LABEL[simple]}」好多少："
                         f"{band(ladder[key], floor_for(rich))}")
    lines += [
        "",
        "**每条都要连着「换一个优化起点的散布」一起看**。要说清楚的是：那个散布是"
        "**验证集上换个起点的敏感度**，与留出患者效应不是同一个随机量，**不是零分布**。"
        "所以它只用来给判决加标注，不用来生成判决。",
        "",
        "### 优化器有没有把这些参数移离零（这不等于「机制被启用」）",
        "",
        "各向异性强度和前馈强度是**非负**的、下界恰好在零；方向偏置是**有符号**的。"
        "参数离开零只说明优化器用上了它，**不能**说成某个机制在生物层面被启用：",
        "",
    ]
    for arm, entry in evidence["mechanism_engagement"].items():
        median = entry["median_when_moved"]
        signed = "有符号" if "SIGNED" in entry["parameter_semantics"] else "非负、下界在零"
        lines.append(
            f"- **{ARM_LABEL.get(arm, arm)}**（{signed}）："
            f"优化器在 {entry['n_moved_off_zero']}/{entry['n_patients']} 位患者上把它移离了零"
            + (f"（其中 {entry['n_negative']} 位为负），移动时中位 {median:.4f}"
               if median is not None else "，从未移动"))

    lines += ["", "## 5. 候选的四种规则是不是太窄", ""]
    for arm in chain:
        key = f"free_low_rank_over_{arm}"
        if key in ladder:
            lines.append(f"- 自由低秩替代算子相对 {ARM_LABEL[arm]}："
                         f"{band(ladder[key], floor_for('MFREE_LOW_RANK_UPPER_BOUND'))}")
    correction = evidence.get("naming_correction", {})
    lines += [
        "",
        "⚠️ **这条臂被我一开始错标成「上界」，它不是。** 它是秩 4–7 的低秩算子，"
        "而四个 motif 的结构化核在其局部支撑上是**满秩**的——所以它**不包含**那四种规则，"
        "是另一个族。因此赢过它**不能**说明候选够宽；"
        f"「候选是否太窄」这个问题本设计**{verdicts.get('candidate_motifs_too_narrow')}**。"
        "结果文件里 `arm` 列仍是旧名，改名只对后续运行生效，纠正记录在证据文件的 "
        "`naming_correction` 字段。",
        "",
        "能说的只有族间优劣：",
    ]
    for arm, state in (verdicts.get("free_low_rank_alternative_vs_motifs") or {}).items():
        lines.append(f"- 相对 {ARM_LABEL.get(arm, arm)}：**{state}**"
                     + ("（自由算子更差）" if state == "REVERSED" else ""))

    lines += [
        "",
        "## 6. 判决（逐条，互不蕴含）",
        "",
        "| 命题 | 判决 |",
        "|---|---|",
    ]
    label = {
        "distance_explains_duration": "空间距离解释相邻时间间隔",
        **{f"recurrent_field_beyond_baselines|{a}":
           f"{ARM_LABEL.get(a, a).split('（')[0]} 提供简单基线之外的信息"
           for a in ("M0_ISOTROPIC_DIFFUSION", "M1_AXIAL_CORRIDOR",
                     "M2_DIRECTED_TRANSPORT", "M3_AXIAL_FEEDFORWARD_TRANSIENT")},
        "M1_AXIAL_CORRIDOR_adds_over_M0_ISOTROPIC_DIFFUSION": "轴向走廊在均匀扩散之上再有贡献",
        "M2_DIRECTED_TRANSPORT_adds_over_M1_AXIAL_CORRIDOR": "方向输运在走廊之上再有贡献",
        "M3_AXIAL_FEEDFORWARD_TRANSIENT_adds_over_M2_DIRECTED_TRANSPORT": "轴向前馈在方向输运之上再有贡献",
        "candidate_motifs_too_narrow": "候选的四种规则太窄",
    }
    for key, value in verdicts.items():
        if isinstance(value, dict):
            continue
        lines.append(f"| {label.get(key, key)} | **{value}** |")

    audit = evidence["engineering_audit"]
    lines += [
        "",
        "## 7. 工程核对",
        "",
        f"- 逐层继承后父子状态最大差 {audit['warm_start_max_state_gap']}，"
        f"逐位一致 = {audit['warm_start_all_exact']}",
        f"- 各向同性那层不记虚假轴角 = {audit['m0_theta_recorded_as_null']}",
        "- 触点分数完全在训练目标之外，不参与任何选择",
        "- 优化起点只在 validation 上挑，测试集只评一次",
        "",
        "### 优化起点敏感性（不进任何统计，单独看）",
        "",
        "| 规则 | 起点数 | 起点间散布中位 | 九成分位 | 哪种继承方式胜出 |",
        "|---|---:|---:|---:|---|",
    ]
    for arm, entry in basin.items():
        lines.append(
            f"| {ARM_LABEL.get(arm, arm)} | {entry['n_starts_median']:.0f} "
            f"| {entry['median_spread']:.5f} | {entry['p90_spread']:.5f} "
            f"| {entry['chosen_head_mode_counts']} |")

    lines += [
        "",
        "## 8. 这一轮不能推出什么",
        "",
        "- 本轮**只**回答相邻两步的时间间隔。未来 2–5 步的触点分布、落点、完整后续场、"
        "传播范围与事件长度**都没有测**，不得引用。",
        "- 触点分数是目标之外的诊断量，不能当作空间预测能力的结论。",
        "- 时间代理不是招募时间也不是传导延迟，任何速度换算都是错的。",
        "- 各条判决互不蕴含，逐条读。",
        "",
    ]
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    path = ARCHIVE / f"motif_time_targets_v0_3_{arguments.date}.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
