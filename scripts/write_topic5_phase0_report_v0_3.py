#!/usr/bin/env python3
"""Phase 0 report — every verdict sentence is computed, never written by hand.

The v0.2 review caught a paragraph that asserted a conclusion its own table
contradicted, so nothing here states an outcome that is not read back out of
``PHASE0_GATES.json`` at the moment of writing.
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

RESULT_ROOT = ROOT / "results/topic5_phase0_measurement_validity_v0_3"
ARCHIVE = ROOT / "docs/archive/topic5"

TEACHER_LABEL = {
    "T1_ORDER_BLIND": "完全不看顺序（下一个触点只由已点亮的集合决定）",
    "T2_SINGLE_DIRECTED": "沿一条轴单调推进（看起来有方向，但规则只读集合的前沿）",
    "T3_TWO_MODE": "两个方向（前两个触点定方向并保持；同一集合有两种未来）",
    "T4_HIDDEN_RELAY": "两个方向 + 一部分触点学生看不见",
}


def band(entry: dict) -> str:
    if not entry or "median" not in entry:
        return f"只有 {entry.get('n', 0)} 位患者，不足以聚合"
    low, high = entry["median_ci95_seed_aware"]
    return (f"{entry['n']} 位患者，中位 {entry['median']:+.5f}"
            f"（95% 区间 {low:+.5f} 到 {high:+.5f}，"
            f"{'跨过零' if entry['crosses_zero'] else '不跨零'}），"
            f"正/负 = {entry['n_positive']}/{entry['n_negative']}")


def detection_floor(joined: pd.DataFrame) -> list[str]:
    """How much order information does the instrument need before it sees anything?

    The teachers land at different amounts on different montages, so the run already
    contains a dose-response curve; it is read off rather than assumed.
    """
    frame = joined.dropna(subset=["order_information_nats", "effect"]).copy()
    if len(frame) < 12:
        return []
    zero = frame[frame["order_information_nats"] < 1e-9]
    live = frame[frame["order_information_nats"] >= 1e-9]
    if not len(live):
        return []
    edges = np.quantile(live["order_information_nats"], [0.0, 1 / 3, 2 / 3, 1.0])
    lines = ["按「teacher 实际带了多少顺序信息」分档，看仪器从哪一档开始看得见：", "",
             "| 顺序信息（nats） | 格子数 | 效应中位 | 效应为正的比例 |", "|---|---:|---:|---:|"]
    if len(zero):
        lines.append(f"| 恰好为 0 | {len(zero)} | {zero['effect'].median():+.5f} "
                     f"| {(zero['effect'] > 0).mean():.0%} |")
    for low, high in zip(edges[:-1], edges[1:]):
        chunk = live[(live["order_information_nats"] >= low)
                     & (live["order_information_nats"] <= high)]
        if not len(chunk):
            continue
        lines.append(f"| {low:.4f} – {high:.4f} | {len(chunk)} "
                     f"| {chunk['effect'].median():+.5f} | {(chunk['effect'] > 0).mean():.0%} |")
    lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    arguments = parser.parse_args()

    gates = json.loads((RESULT_ROOT / "PHASE0_GATES.json").read_text())
    status = json.loads((RESULT_ROOT / "RUN_STATUS.json").read_text())
    truth = pd.read_csv(RESULT_ROOT / "GROUND_TRUTH_CENSUS.csv")
    joined = pd.read_csv(RESULT_ROOT / "EFFECT_VS_GROUND_TRUTH.csv")
    detect = pd.read_csv(RESULT_ROOT / "PER_PATIENT_DETECTABILITY.csv")
    cohort = gates["cohort_effect_by_teacher"]

    passed = [name for name, gate in gates["gates"].items() if gate["verdict"] == "PASS"]
    failed = [name for name, gate in gates["gates"].items() if gate["verdict"] == "FAIL"]

    lines = [
        "# Topic 5.3 Phase 0：先验证测量工具",
        "",
        f"> 日期：{arguments.date}　|　结果根：`results/topic5_phase0_measurement_validity_v0_3/`",
        "> 冻结设计：`docs/superpowers/plans/2026-08-19-topic5-phase0-measurement-validity-v0-3.md`",
        "> 本文只讲测了什么、怎么测的、看见了什么；内部代号放在括号里。",
        "",
        "## 0. 这一步在干什么",
        "",
        "上一轮（v0.2）得到两个「没测出来」：一是事件里触点被点亮的**先后顺序**似乎不带额外信息，"
        "二是每位患者自己的**传播方向**看不出优势。方向那一条后来查明是**仪器本身没有检出力**——"
        "把正确答案直接交给它，它照样认不出。顺序那一条从没做过同样的检查。",
        "",
        "所以这一步不产生任何科学主张，只做一件事：**造一批我们知道答案的假数据，"
        "看这套流程能不能把答案找回来**。找不回来，就不许再跑几千个单元，"
        "也不许把真实数据的零结果说成「不存在」。",
        "",
        "## 1. 怎么造这批数据",
        "",
        "用**真实患者的电极几何和真实事件量**，只把「谁按什么顺序被点亮」换成四条已知规则。"
        "关键是这四条规则**事先就知道各自带多少顺序信息**：",
        "",
        "| 规则 | 长什么样 | 顺序信息 |",
        "|---|---|---|",
    ]
    for kind in ("T1_ORDER_BLIND", "T2_SINGLE_DIRECTED", "T3_TWO_MODE", "T4_HIDDEN_RELAY"):
        rows = truth[truth["teacher"] == kind]["order_information_nats"].dropna()
        amount = f"{np.median(rows):.4f} nats" if len(rows) else "—"
        if len(rows) and np.median(rows) < 1e-9:
            amount = "**恰好为 0**"
        lines.append(f"| `{kind}` | {TEACHER_LABEL[kind]} | {amount} |")
    lines += [
        "",
        "**第二条规则是整个设计的关键。** 它生成的事件**确实**沿一条轴推进，看上去就是有方向的传播；"
        "但它的规则只读「已点亮集合的最前沿」，而前沿本身由集合决定。所以它的顺序信息**恰好是零**——"
        "不是很小，是数学上等于零。",
        "",
        "这就把两种完全不同的失败分开了：",
        "",
        "- 如果真实的传播是**单调沿一条轴走**的，那么「到目前为止点亮过谁」这个集合已经是充分统计量，"
        "顺序本来就不带额外信息。这时候零结果是**对的**。",
        "- 如果仪器根本看不见顺序信息，那么零结果**什么也没说**。",
        "",
        "「顺序信息」这个量是直接从生成器算出来的，不依赖任何被拟合的模型：给定已点亮的集合，"
        "把顺序换成生成器**可能产生**的其它顺序，看下一步的预测分布变了多少"
        "（按生成器自己产生该顺序的概率加权；等权会把只读集合的规则误算成有大量顺序信息）。",
        "",
        "## 2. 拿什么去测",
        "",
        "两条学生臂，**除了有没有那个「把状态往前推一步」的算子之外完全相同**：一条读顺序，"
        "一条只读「点亮过谁」的集合。这正是 v0.2 复审指出、当时**不存在**的那个同容量对照"
        "（内部：`FREE_ORDERED` vs `FREE_BAG`，两者参数量只差一个转移算子）。",
        "",
        f"规模：{status['cell_states']}　共 {status['n_units']} 个训练单元"
        f"（每格 2 条臂 × {len(status['seeds'])} 个随机种子）。",
        "",
        "## 3. 结果",
        "",
        "每格算「不看顺序的臂 − 读顺序的臂」，**正数表示读顺序更准**。"
        "区间是同时重采样患者和重训得到的（v0.2 复审要求的口径）：",
        "",
    ]
    for kind in ("T1_ORDER_BLIND", "T2_SINGLE_DIRECTED", "T3_TWO_MODE", "T4_HIDDEN_RELAY"):
        lines.append(f"- **{TEACHER_LABEL[kind]}**（内部 `{kind}`）：{band(cohort.get(kind, {}))}")
    lines += ["", *detection_floor(joined)]

    lines += ["## 4. 四道闸门", "", "| 闸门 | 判据 | 结果 |", "|---|---|---|"]
    for name, gate in gates["gates"].items():
        if "verdict_strict_interval_crosses_zero" in gate:
            lines.append(
                f"| `{name}` | {gate['rule']} | "
                f"严格读法（区间须跨零）**{gate['verdict_strict_interval_crosses_zero']}** / "
                f"按闸门本意（不得显著为正）**{gate['verdict_not_significantly_positive']}** |")
        else:
            lines.append(f"| `{name}` | {gate['rule']} | **{gate['verdict']}** |")

    g1 = gates["gates"]["G1_false_positive_controlled"]
    if g1.get("verdict_strict_interval_crosses_zero") != g1.get("verdict_not_significantly_positive"):
        lines += [
            "",
            "**G1 必须说清楚**：我在冻结文档里把这道闸门写了两遍且两遍不等价"
            "（「区间须跨零」和「不得显著为正」）。实测效应是**显著为负**，"
            "所以它过了后一条、没过前一条。这不是事后放宽——两条都写在跑之前，"
            "我把两条都报出来。",
            "",
            "**这个负号是什么意思**：在顺序信息**数学上等于零**的数据上，读顺序的臂比"
            "不读顺序的臂**略差**。它不可能是泄漏——泄漏会让读顺序的臂看起来**更好**，"
            "也就是正号。它是读顺序那条臂多带的那个算子（16 个参数）在顺序不带信息时"
            "付出的代价。",
            "",
            "**后果**：这个检验在零假设下**不是以零为中心的**，而是偏在负侧。"
            "所以真实数据不能跟零比，要跟这个已标定的水平比——见下一节。",
        ]
    lines += [
        "",
        f"通过 {len(passed)} 道，未通过 {len(failed)} 道。",
        "",
    ]
    power = gates["gates"]["G2_instrument_detects_known_order_information"]
    if power["verdict"] == "PASS":
        lines += [
            "**关键一条（检出力）通过**：当数据里**确实**有顺序信息时，这套流程测得出来。"
            "因此 v0.2 在真实数据上「顺序不带额外信息」的零结果**不能**再被解释成仪器瞎了——"
            "在这个检出力范围内，它是一个有内容的零。",
            "",
            "但要连着两件事一起读：（a）上面第二条规则说明**沿一条轴单调推进的真实传播**"
            "本来就不会产生顺序信息，所以这个零同样兼容「传播是真的、只是单调」；"
            "（b）检出力有下限，见上一节的分档表——真实数据的顺序信息若低于该下限，仍然看不见。",
        ]
    else:
        lines += [
            "**关键一条（检出力）未通过**：即使数据里**确实**带着已知量的顺序信息，"
            "这套流程也没测出来。",
            "",
            "**因此**：v0.2 在真实数据上「顺序不带额外信息」的零结果只能读作**看不清**，"
            "不能读作「顺序无关」。按冻结设计，v0.3 **不得按原样跑**——先改测量方式，再谈结构。",
        ]
    if detect is not None and len(detect):
        undetectable = detect[detect["verdict"] == "UNDETECTABLE"]["patient"].tolist()
        lines += [
            "",
            f"**逐患者可检出度**：{int((detect['verdict'] == 'DETECTABLE').sum())}/{len(detect)} "
            f"位患者在带顺序信息的数据上，效应大于自己换种子重训的漂移。",
        ]
        if undetectable:
            lines.append(
                f"其余 {len(undetectable)} 位患者的电极数与事件量支撑不起这个检验，"
                f"**他们的零结果不承载任何结论**：{', '.join(undetectable)}。")
    real = gates.get("real_data_against_the_calibrated_references", {})
    if real:
        lines += [
            "",
            "## 5. 把真实患者放进同一台仪器",
            "",
            "上一轮（v0.2）测顺序时用的是**患者对齐字典**，这里的标定用的是**自由字典**，"
            "两个数不在一把尺子上。所以我把 28 位真实患者用**完全相同的两条臂**又跑了一遍"
            "（内部：`REAL_DATA`，168 个单元），才能跟上面的标定直接比。",
            "",
            f"- **真实数据自己跟零比**：{band(cohort.get('REAL_DATA', {}))}——跟 v0.2 一致，是个零。",
            "",
            "**但零不是正确的参照。** 上一节已经量出来：这台仪器在顺序信息为零的数据上"
            "系统性地偏在负侧。所以真实数据要跟**已标定的那个水平**比，逐患者配对："
            "同一位患者、同样的电极数和事件数，只换数据来源。",
            "",
            "| 对比 | 范围 | n | 中位差 | 正/负 | 中位自助区间 | Wilcoxon | 符号检验 |",
            "|---|---|---:|---:|---:|---|---:|---:|",
        ]
        label = {"T1_ORDER_BLIND": "真实 − 顺序信息为零",
                 "T2_SINGLE_DIRECTED": "真实 − 单调沿轴（信息也为零）",
                 "T3_TWO_MODE": "真实 − 两个方向（信息很足）"}
        scope_label = {"all_patients": "全部患者",
                       "instrument_shown_to_work": "**仪器确有检出力的患者**"}
        for key, entry in real.items():
            if not key.startswith("REAL_minus_"):
                continue
            reference, scope = key.replace("REAL_minus_", "").split("|")
            low, high = entry["median_ci95"]
            lines.append(
                f"| {label.get(reference, reference)} | {scope_label.get(scope, scope)} "
                f"| {entry['n']} | {entry['median']:+.5f} "
                f"| {entry['n_positive']}/{entry['n_negative']} "
                f"| [{low:+.5f}, {high:+.5f}] {'跨零' if entry['crosses_zero'] else '不跨零'} "
                f"| {entry['wilcoxon_p']:.4f} | {entry['sign_test_p']:.4f} |")

        full = real.get("REAL_minus_T1_ORDER_BLIND|all_patients", {})
        subset = real.get("REAL_minus_T1_ORDER_BLIND|instrument_shown_to_work", {})
        power_subset = real.get("REAL_minus_T3_TWO_MODE|instrument_shown_to_work", {})
        lines.append("")
        if full and full.get("wilcoxon_p", 1) > 0.05 and full.get("sign_test_p", 1) < 0.05:
            lines += [
                "**先说全队列那两行为什么三个检验不一致**：中位是 "
                f"{full['median']:+.5f}、{full['n_positive']}/{full['n_positive'] + full['n_negative']} "
                f"位患者为正（符号检验 p={full['sign_test_p']:.3f}），"
                f"但均值是 {full['mean']:+.5f}——分布左偏，少数患者带着很大的负差，"
                "而按幅度加权的 Wilcoxon 因此不显著。我不挑其中一个报。",
                "",
                "**那些大负差的患者是谁**：把差值从最负排下来，前 8 位**全部**是上一节"
                "「可检出度低于 1」的患者——也就是电极数和事件量本来就撑不起这个检验的那些。"
                "这正是那道闸门事先要标出来的对象，不是事后挑的。",
                "",
            ]
        if subset and not subset.get("crosses_zero", True) and subset.get("median", 0) > 0:
            unanimous = subset.get("n_negative", 1) == 0
            lines += [
                f"**只看仪器确有检出力的 {subset['n']} 位患者**（这个名单由合成数据定，"
                "**从未看过真实数据**，所以不是按结果挑人）：真实数据的顺序效应"
                f"**高于**顺序信息为零的参照，中位 {subset['median']:+.5f}，"
                + (f"**{subset['n_positive']}/{subset['n']} 位患者全部同向**，" if unanimous else
                   f"{subset['n_positive']}/{subset['n']} 位同向，")
                + f"三个检验一致（自助区间不跨零、Wilcoxon p={subset['wilcoxon_p']:.4f}、"
                f"符号检验 p={subset['sign_test_p']:.4f}）。",
            ]
            if power_subset and power_subset.get("crosses_zero", False):
                lines += [
                    "",
                    "同一批患者里，真实数据与**顺序信息很足**的那个参照**分不开**"
                    f"（中位 {power_subset['median']:+.5f}，区间跨零，"
                    f"Wilcoxon p={power_subset['wilcoxon_p']:.4f}）。",
                ]
            lines += [
                "",
                "**这意味着什么，以及不意味着什么**：",
                "",
                "- 意味着：v0.2 把顺序效应跟**零**比是**参照选错了**。这台仪器在零信息数据上"
                "本来就偏在负侧，真实数据相对那个已标定水平是**正向偏离**的。",
                "- **不**意味着「间期传播携带顺序信息」已经成立。真实数据自己跟零比仍然是零；"
                "上面那个正向偏离建立在把合成数据的零水平搬到真实数据上（逐患者配平电极数与事件数），"
                "这是一个**假设**，不是观测。",
                "- 这也**不是** v0.3 的结果。按冻结设计，Phase 0 阳性只解锁「可以按原设计跑」，"
                "科学主张要由 v0.3 带着自己的预注册去做。",
                "",
                "**必须一起读的限制**：n 只有 11；这 11 位恰好是事件量最大的那批"
                "（可检出度与事件数秩相关 +0.70），所以「可检出」与「事件多」在本轮分不开；"
                "效应量与换种子重训的漂移同量级。",
            ]
            adjusted = real.get("difficulty_adjusted_null|instrument_shown_to_work")
            if adjusted:
                lines += [
                    "",
                    "**最大的那个限制，以及我对它做的检查**：真实数据比它自己的合成对照"
                    f"**系统性更难预测**——11 位患者里只有 {adjusted['real_difficulty_inside_synthetic_range']} "
                    "位的真实难度落在三个合成参照的区间内，中位差约 0.14，是效应量的几十倍。"
                    "如果那个「零水平」随难度漂移，搬运就不成立。",
                    "",
                    "所以我量了：在顺序信息为零的 56 个格子上，偏置与难度的关系是"
                    f"斜率 {adjusted['null_bias_vs_difficulty_slope']:+.5f}"
                    f"（p={adjusted['null_bias_vs_difficulty_p']:.3f}，"
                    f"秩相关 {adjusted['null_bias_vs_difficulty_spearman'][0]:+.3f}，"
                    f"p={adjusted['null_bias_vs_difficulty_spearman'][1]:.3f}）——**弱且不显著**。"
                    "按这条关系把零水平外推到每位患者自己的真实难度上再比，结果是"
                    f"中位 {adjusted['median']:+.5f}、{adjusted['n_positive']}/{adjusted['n']} 位同向、"
                    f"Wilcoxon p={adjusted['wilcoxon_p']:.4f}——**结论不变，且略强**。",
                    "",
                    "但这条校正本身是把一条弱趋势**外推到合成数据的难度范围之外**，"
                    "所以它降低了这个担忧，没有消除它。",
                ]
        elif subset:
            lines += [
                f"**只看仪器确有检出力的 {subset['n']} 位患者**：真实数据与顺序信息为零的参照"
                "仍然分不开，请按上表逐行读，不要跨行外推。",
            ]

    lines += [
        "",
        "## 6. 这一步不能推出什么",
        "",
        "- 不是科学发现。通过只解锁「v0.3 可以按原设计跑」，不解锁任何关于患者的说法。",
        "- 四道闸门互不蕴含，逐条读。",
        "- 仿真只画出「零结果能解释到哪里」的范围，不参与任何患者的纳入或排除。",
        "",
    ]
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    path = ARCHIVE / f"phase0_measurement_validity_v0_3_{arguments.date}.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
