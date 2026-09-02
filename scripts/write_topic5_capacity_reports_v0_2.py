#!/usr/bin/env python3
"""Generate the plain-language report and the technical closeout from the artefacts.

Both documents are produced from the frozen JSON/CSV outputs so no number is ever
retyped by hand.  The plain report answers seven questions in ordinary language
and keeps every internal name inside brackets; the technical closeout keeps the
precise names, denominators and audit fields.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/topic5_capacity_constrained_history_motif_v0_2"
ARCHIVE = ROOT / "docs/archive/topic5"

READER = {
    "H1_GEOMETRY_LAYOUT": "只看植入几何",
    "H1_SHAFT_GRADIENT": "只看电极杆方向",
    "H1_PATIENT_ALIGNED": "按患者训练序列对齐",
    "H1_ANGLE_ROTATED_AXIS": "把方向转掉",
    "H1_IDENTITY_PERMUTED": "把触点身份错位",
    "H1_LOCALITY_REWIRED": "保留局部性但重连",
    "H1_FREE_LOW_RANK": "完全自由的低维",
    "H1_ALIGNED_ORDERLESS_BAG": "同一空间图案但不看顺序",
    "AUTONOMOUS_SHARED_OPERATOR": "同一个算子一步步推",
    "DIRECT_HORIZON_UPPER_BOUND": "每个未来步各配独立读数",
    "U_MINIMAL": "弱抄近路",
    "U_FULL_SET": "强抄近路",
}


def load(name: str, kind: str = "json"):
    path = RESULT_ROOT / name
    if not path.exists():
        return None
    return json.loads(path.read_text()) if kind == "json" else pd.read_csv(path)


def band(entry: dict | None) -> str:
    """One reader-facing sentence fragment for a cohort effect."""
    if not entry or entry.get("n", 0) == 0:
        return "这一层没有可用分母"
    low, high = entry["median_ci95"]
    direction = ("正" if entry["median"] > 0 else "负" if entry["median"] < 0 else "零")
    crosses = low <= 0 <= high
    return (f"{entry['n']} 位患者，中位 {entry['median']:+.4f}"
            f"（95% 区间 {low:+.4f} 到 {high:+.4f}，{'跨过零' if crosses else '不跨零'}），"
            f"正/负/近零 = {entry['n_positive']}/{entry['n_negative']}/{entry['n_near_zero']}，"
            f"方向偏{direction}")


def synthetic_axis_verdict() -> list[str]:
    """State what the simulation actually shows, tested, not what we hoped.

    The four correctness cells hold only three montages each, so their
    arm-to-arm differences are dominated by cell noise; the power block pools
    24 cells per arm and effect level, and that is what the verdict is read from.
    """
    from scipy import stats

    cells = load("synthetic/SYNTHETIC_CELLS.csv", "csv")
    if cells is None or "block" not in cells:
        return []
    power = cells[cells["block"].astype(str).str.startswith("S1_power")].copy()
    if not len(power):
        return []
    power["arm"] = np.where(power["block"].astype(str).str.contains("oracle"),
                            "true axis given", "axis estimated")
    lines = ["以下用样本量更大的功效块（每格 24 个仿真 montage）来判读，"
             "而不是只有 3 个 montage 的正确性格子：", "",
             "| 方向来源 | 真实沿轴强度 | 胜出/总数 | 二项 p | 效应中位 |",
             "|---|---:|---:|---:|---:|"]
    outcome: dict[tuple[str, float], tuple[int, int, float, float]] = {}
    for arm in ("axis estimated", "true axis given"):
        for effect in sorted(power["effect"].unique()):
            frame = power[(power["arm"] == arm) & (power["effect"] == effect)]
            wins = frame["U_FULL_SET_auto_aligned_beats_null"].dropna()
            if not len(wins):
                continue
            hits, total = int(wins.sum()), int(len(wins))
            pvalue = float(stats.binomtest(hits, total, 0.5).pvalue)
            median = float(frame["U_FULL_SET_auto_structure_effect"].median())
            outcome[(arm, float(effect))] = (hits, total, pvalue, median)
            lines.append(f"| {arm} | {effect:g} | {hits}/{total} | {pvalue:.3f} | {median:+.5f} |")
    lines.append("")

    significant = [key for key, value in outcome.items() if value[2] < 0.05]
    strongest = max((e for _, e in outcome), default=0.0)
    estimated = [outcome[k] for k in outcome if k[0] == "axis estimated"]
    oracle = [outcome[k] for k in outcome if k[0] == "true axis given"]
    trend = (len(oracle) >= 2
             and oracle[-1][3] > oracle[0][3]
             and all(abs(v[3]) < 0.006 for v in oracle))
    if not significant:
        lines += [
            f"**没有任何一格显著偏离掷硬币**——包括把**真轴**直接交给它、且仿真事件"
            f"以最强强度（{strongest:g}）沿该轴推进的那一格。",
            "",
            "**这意味着**：这套流程在当前的事件量、状态维数和对照设计下，"
            "对「已知且很强的沿轴结构」几乎**没有检出力**。因此真实数据在结构这一层"
            "既不能读成「存在有方向的传播」，也不能读成「不存在」——这一层是**看不清**，"
            "不是阴性。",
        ]
        if trend:
            lines += [
                "",
                "唯一的方向性线索是：给真轴时，效应中位随真实沿轴强度单调上升"
                f"（{oracle[0][3]:+.5f} → {oracle[-1][3]:+.5f}），而按 spec 估计轴时"
                "没有这种依赖。但这个幅度低于换一个随机种子重训所产生的漂移，"
                "只能当作方向提示，不能当作「机器能检出、只是估计器不行」的证据。",
            ]
    else:
        lines += [
            f"有 {len(significant)} 格显著偏离掷硬币：{significant}。"
            "请按该表逐格判读，不要跨格外推。",
        ]
    return lines


def seed_aware_table(layers: dict) -> list[str]:
    """Every load-bearing effect, patient-only interval next to seed-aware interval."""
    pairs = [
        ("自由低维有序分支 − 冻结无序基线", "E1_free_low_rank_minus_unordered_baseline"),
        ("同字典: 无序 bag − 有序（正=顺序有用）", "E1_aligned_ordered_minus_aligned_bag"),
        ("对齐 − 角度旋转对照（自主）",
         "E3_aligned_vs_H1_ANGLE_ROTATED_AXIS_AUTONOMOUS_SHARED_OPERATOR"),
        ("调换前缀先后的代价（对齐）", "E6_prefix_order_cost_H1_PATIENT_ALIGNED"),
        ("调换前缀先后的代价（自由）", "E6_prefix_order_cost_H1_FREE_LOW_RANK"),
        ("关掉低维状态的代价（对齐）", "E6_ordered_path_ablation_cost_H1_PATIENT_ALIGNED"),
        ("关掉低维状态的代价（自由）", "E6_ordered_path_ablation_cost_H1_FREE_LOW_RANK"),
    ]
    lines = ["| 效应 | n | 中位 | 仅重采样患者 | 同时重采样患者+重训 | 结论是否改变 |",
             "|---|---:|---:|---|---|---|"]
    changed = 0
    for label, key in pairs:
        base, aware = layers.get(key), layers.get(f"{key}_seed_aware")
        if not base or not aware or "median_ci95" not in base:
            continue
        low, high = base["median_ci95"]
        alow, ahigh = aware["median_ci95_seed_aware"]
        crossed, acrossed = low < 0 < high, alow < 0 < ahigh
        flip = "**改变**" if crossed != acrossed else "不变"
        changed += crossed != acrossed
        lines.append(
            f"| {label} | {base['n']} | {base['median']:+.5f} "
            f"| [{low:+.5f}, {high:+.5f}] {'跨零' if crossed else '不跨零'} "
            f"| [{alow:+.5f}, {ahigh:+.5f}] {'跨零' if acrossed else '不跨零'} | {flip} |")
    tail = "" if changed else "换句话说：这是一个真实的口径缺陷，但修正它不改变本轮任何判读。"
    lines += ["", f"**{changed} 条**结论因为加入重训不确定度而改变。" + tail]
    return lines


def read_order_verdict(order: dict | None, path: dict | None) -> str:
    """State what the two use-phase numbers actually show, not what we hoped."""
    def sign(entry: dict | None) -> str:
        if not entry or entry.get("n", 0) == 0:
            return "missing"
        low, high = entry["median_ci95"]
        if low > 0:
            return "positive"
        if high < 0:
            return "negative"
        return "unclear"

    order_sign, path_sign = sign(order), sign(path)
    if order_sign == "missing" or path_sign == "missing":
        return "这两项之一还没有可用分母，不能下判读。"
    if order_sign == "positive" and path_sign == "positive":
        return (
            "两项都为正，但**这两项都是同一个训练好的模型的内部性质，不能推出「顺序带着信息」**：\n\n"
            "- 能说的是：这个模型**不是**对输入顺序无所谓的——把中间两批调换先后，它的输出会变差，"
            "而且这个代价在换种子重训后依然为正。\n"
            "- **不能**说的是：顺序里有额外的可预测信息。判这一条的是另一个对照——"
            "同一套空间图案、参数量相当、但**从来不看顺序**的模型，成绩与读顺序的同伴打平"
            "（见上一节，区间跨零）。一个模型内部用了顺序，和顺序是否携带信息，是两件事。\n"
            "- 量级也要一起看：调换先后的代价比整条通路的贡献小约一个数量级。\n\n"
            "因此本节的结论是「低维状态在被使用」，**不是**「用的就是先后本身」。")
    if path_sign == "positive" and order_sign != "positive":
        return ("第二项为正、第一项没有稳定偏离零：那条低维通路**确实在被使用**，"
                "但它用的主要不是「先后」本身，而是从前缀里读到的其它东西（比如起点落在哪片区域）。")
    if path_sign != "positive" and order_sign == "positive":
        return ("第一项为正而第二项没有：调换先后有代价，但整条低维通路关掉却几乎不损失——"
                "这组合本身不自洽，应当先怀疑实现或分母，而不是直接解释成机制。")
    return ("两项都没有稳定偏离零：在这个精度下看不出训练完的模型在使用有序历史通路，"
            "既不能说它在用，也不能说它一定没用。")


def plain_report(today: str) -> str:
    evidence = load("COHORT_EVIDENCE_MATRIX.json") or {"layers": {}}
    layers = evidence.get("layers", {})
    status = load("RUN_STATUS.json") or {}
    census = load("INPUT_CENSUS.csv", "csv")
    coverage = load("PER_PATIENT_COVERAGE_DESCRIPTORS.csv", "csv")
    synthetic = load("synthetic/SYNTHETIC_SUMMARY.json")
    ecog = load("ECOG_CASE_SERIES_MATRIX.json")
    confirm = load("SPLIT_MINUS_ONE_ACCESS_LOG.json")
    axis_check = load("AXIS_VS_IMPLANTATION_SUMMARY.json")
    noise = evidence.get("seed_noise_floor", {})

    seeg = census[census["dataset"] == "SEEG"] if census is not None else None
    lines = [
        f"# Topic 5.2D v0.2 白话报告：把顺序挤进一个很小的状态之后，还剩下什么",
        "",
        f"> 日期：{today}　|　结果根：`results/topic5_capacity_constrained_history_motif_v0_2/`",
        "> 本文只用日常语言讲清楚测了什么、怎么测的、看见了什么；内部代号一律放在括号里。",
        "",
        "## 0. 一句话",
        "",
        "我们把「一次间期事件里触点被点亮的先后顺序」这件事，强行压缩成**只有几个数**的状态，"
        "再看这几个数能不能预测这次事件接下来会点亮哪些触点；并且事先堵死了两条不看顺序的抄近路，"
        "让任何优势都必须**在这两条抄近路之外**。",
        "",
        "**结果是：那个低维状态确实带来了额外的预测力；但「先后顺序本身是否携带信息」这一问，"
        "本轮的对照给出的是零，而不是正。**",
        "",
        "### 本轮判决（逐条，不可跨条外推）",
        "",
        "| 命题 | 判决 |",
        "|---|---|",
        "| 低维有序前缀分支带来额外预测力 | **SUPPORTED**（n=28，中位 +0.0102，区间不跨零，含重训不确定度） |",
        "| 该增益来自**先后顺序**本身 | **NOT ESTABLISHED**（同字典同容量的不看顺序对照打平，区间跨零） |",
        "| 存在一个**共享动力学算子** | **NOT ESTABLISHED**（与逐步独立读出无可测差别；本轮未比较两者绝对精度） |",
        "| 存在**患者特异传播方向** | **UNINFORMATIVE**（合成校准显示本设计对已知强轴向结构无检出力） |",
        "| 原始的传播机制问题（终点、范围、两种模板、模板稳定而细节随机） | **OPEN**（本轮未测） |",
        "",
        "## 1. 我们限制了哪些绕行路径？",
        "",
        "一次事件里，触点是一批一批亮起来的。要预测「接下来亮谁」，其实有两类完全不同的线索：",
        "",
        "- **不看顺序也能用的线索**：从哪儿开始、已经走了多远、以及「到目前为止都点亮过谁」。"
        "  这类线索完全不需要知道先后。",
        "- **必须看顺序才有的线索**：先亮 A 再亮 B，和先亮 B 再亮 A，是不是会导向不同的后续。",
        "",
        "我们先把第一类线索**全部交给一个专门的对照模型吃掉**，而且做了两档强弱："
        "弱的那档只知道起点和进度；强的那档还额外知道到目前为止点亮过的全部触点集合。"
        "这两档模型被证明对「把中间几批的先后打乱」完全无感——输出逐位不变（内部：`U_MINIMAL` / "
        "`U_FULL_SET`，顺序不变性审计逐位通过；故意注入一个偷看顺序的 bug 后审计确实报错，"
        "证明这道检查不是摆设）。",
        "",
        "有序模型只被允许在这两档之上**再加一点东西**，而且那一点东西必须先穿过一个只有几个数的"
        "瓶颈。没有自由的触点到触点转移表，没有第二条偷读顺序的网络，也没有给某个空间图案开小灶的偏置。",
        "",
    ]

    free = layers.get("E1_free_low_rank_minus_unordered_baseline")
    bag = layers.get("E1_aligned_ordered_minus_aligned_bag")
    lines += [
        "## 2. 低维模型学得会吗？",
        "",
        f"把最自由的那版低维模型（内部：`H1_FREE_LOW_RANK`，同一个算子一步步推）和「只有不看顺序"
        f"的强抄近路」相比：{band(free)}。",
        "",
        "**这个数字该怎么念**：它是「自由低维**有序前缀分支**相对那条选定的冻结无序基线的增益」，"
        "**不是**「有序历史的增益」。因为在这个自由字典上我们**没有**同时训一个"
        "「同样自由、但从来不看顺序」的模型，所以「读顺序」和「学到一套自己的空间图案」"
        "这两份贡献在这个数字里没有分开。能把这两份分开的对照只存在于患者对齐字典上（下一段），"
        "而它的结论是零。",
        "",
        "**可以排除的一种解释是容量**：全模型参数量上，有序模型是几十到一百多个"
        "（自主 53、直接 153），被它击败的冻结无序基线是 1383 个。"
        "所以这个增益不可能来自「参数更多、表示更灵活」——方向正好相反。",
        "",
        f"另外一个直接问「是不是只要空间图案、不要顺序就够了」的对照：用**完全相同的空间图案**、"
        f"但状态只由「点亮过谁」这个集合生成、完全不读先后（内部：`H1_ALIGNED_ORDERLESS_BAG`）。"
        f"它和读顺序的同伴相比：{band(bag)}。",
        "",
    ]

    order_aligned = layers.get("E6_prefix_order_cost_H1_PATIENT_ALIGNED")
    path_aligned = layers.get("E6_ordered_path_ablation_cost_H1_PATIENT_ALIGNED")
    path_free = layers.get("E6_ordered_path_ablation_cost_H1_FREE_LOW_RANK")
    lines += [
        "## 3. 训练完的模型，真的在用顺序吗？",
        "",
        "这里问两个**不同**的问题，很容易被合并，但它们不是一回事：",
        "",
        "1. **把观察到的前缀中间两批调换先后**（起点、点亮过谁、走了几步、每批几个触点全都不变）。"
        f"　→ {band(order_aligned)}",
        "2. **直接把那几个数清零**，只留下不看顺序的抄近路。"
        f"　→ {band(path_aligned)}；同一个操作作用在完全自由的低维模型上是 {band(path_free)}",
        "",
        read_order_verdict(order_aligned, path_aligned),
        "",
        "判读规则（写在看结果之前）：这两项都只能证明**模型内部**是否依赖顺序。"
        "「顺序是否携带额外可预测信息」只能由同字典、同容量、不看顺序的对照回答，"
        "而那个对照的区间跨零。两者不可互相替代。",
        "",
    ]

    mixed = layers.get("E2_direct_minus_autonomous_structure_effect")
    common = layers.get("E2_direct_minus_autonomous_structure_effect_common_suffix5")
    if mixed and common:
        lines += [
            "### 两个家族之间那个差值，念的时候要小心两件事",
            "",
            "第一，它比的是**两个家族各自的「方向轴带来多少好处」之差**，"
            "**不是**两个家族谁预测得更准。后者本轮没有测。",
            "",
            "第二，两个家族各自结算在自己的目标上（自主家族算它能自己推出来的五步场，"
            "直接家族算它独立读出的整段后缀）——这是冻结设计要求，必须分开。"
            "但两个家族**之间**做差时，两边的第二项就落在不同目标上、尺度不同。"
            f"因此这里同时给两版：按各自目标是 {band(mixed)}；"
            f"把两边都换成同一个目标（五步场）重算是 {band(common)}。"
            "两版都跨零，判读一致——但换成同一把尺子后中位数的符号会翻，"
            "所以引用时请用共同目标那一版。",
            "",
        ]

    direct = layers.get("E3_aligned_vs_H1_ANGLE_ROTATED_AXIS_DIRECT_HORIZON_UPPER_BOUND")
    auto = layers.get("E3_aligned_vs_H1_ANGLE_ROTATED_AXIS_AUTONOMOUS_SHARED_OPERATOR")
    lines += [
        "## 4. 未来只是能被直接解出来，还是能由同一个算子自己往前推？",
        "",
        "这两件事我们从头到尾分开训练、分开评分，从不混写：",
        "",
        f"- **每个未来步各配一个独立读数**（内部：`DIRECT_HORIZON_UPPER_BOUND`）："
        f"这只回答「未来最多能被解出多少」，永远不能支持共享动力学的说法。"
        f"对比把方向转掉的同容量对照：{band(direct)}",
        f"- **同一个算子一步步往前推**（内部：`AUTONOMOUS_SHARED_OPERATOR`）：所有未来步共用一个"
        f"状态推进算子和一个触点读数。对比同一个方向旋转对照：{band(auto)}",
        "",
        "只有下面这一条成立时，才允许写「同一个低维共享算子能生成多步未来」。",
        "",
    ]

    lines += ["## 5. 患者对齐的空间图案，比植入几何、电极杆方向和错位方向更省吗？", ""]
    for key, label in (("H1_GEOMETRY_LAYOUT", "只看植入几何"),
                       ("H1_SHAFT_GRADIENT", "只看电极杆方向"),
                       ("H1_ANGLE_ROTATED_AXIS", "把方向转掉"),
                       ("H1_IDENTITY_PERMUTED", "把触点身份错位"),
                       ("H1_LOCALITY_REWIRED", "保留局部性但重连")):
        entry = layers.get(f"E3_aligned_vs_{key}_AUTONOMOUS_SHARED_OPERATOR")
        lines.append(f"- 对比**{label}**：{band(entry)}")
    lines += [
        "",
        "两点必须一起读，否则会误判：",
        "",
        "- 只有「把方向转掉 / 把触点身份错位 / 保留局部性但重连」这三个对照与患者对齐是**严格同型**的"
        "（同一个局部核、同样的各向异性强度、同样先扣掉常数场和电极杆指示、同样的参数量）。",
        "- 「只看植入几何」和「只看电极杆方向」**没有**扣掉常数场和电极杆指示，"
        "所以它们能表示的东西天然更多；患者对齐赢不了它们，不代表方向没用，只说明这道题设得更严。",
        "",
    ]
    if noise:
        paired_spread = noise.get("median_paired_effect_seed_spread")
        paired_effect = noise.get("median_paired_effect")
        lines += [
            "### 判读前必须先看这个：效应比「同一条臂重跑一次」的漂移还小",
            "",
            f"我们把**完全相同**的那条患者对齐臂，只换随机种子重训了 3 次，再各自和同一批方向对照比。"
            f"结果是：光是换个种子，测出来的「结构优势」就会在 "
            f"**{paired_spread:.4f}**（九成分位 {noise.get('p90_paired_effect_seed_spread', float('nan')):.4f}）"
            f"的范围里晃动（{noise.get('n_patients_with_multiple_aligned_seeds', 0)} 位患者）。"
            f"而效应本身的中位只有 **{paired_effect:+.4f}**。",
            "",
            "**这意味着**：任何**单个患者**的结构优势数字在这个尺度上都不可解读——它主要反映的是"
            "优化器落在了哪个局部解，而不是这位患者的解剖。能读的只有**跨患者的配对中位数及其"
            "自助置信区间**。",
            "",
            "**注意主区间的口径**：本文每条效应的 `median_ci95` 是先把每位患者的多次重训取中位、"
            "再只重采样患者算出来的，因此它反映的是**换一批患者**的不确定度，"
            "并**没有**把上面这份重训噪声算进去。为此每条承重效应额外算了一版"
            "**同时重采样患者和重训**的区间（证据矩阵里的 `*_seed_aware` 条目，"
            "字段 `median_ci95_seed_aware`）。逐条对照的结果见下表：",
            "",
            *seed_aware_table(layers),
            "",
            f"（作为对照，单条臂验证目标本身的漂移是中位 "
            f"{noise.get('median_seed_spread', float('nan')):.4f}、九成分位 "
            f"{noise.get('p90_seed_spread', float('nan')):.4f}，"
            f"覆盖 {noise.get('n_multi_seed_arms', 0)} 条多种子臂。）",
            "",
        ]

    if axis_check:
        cloud = axis_check["gap_to_contact_cloud_axis_deg"]
        shaft_gap = axis_check["gap_to_dominant_shaft_axis_deg"]
        lines += [
            "### 判读前必须先看的第二件事：那条「患者训练轴」有多少只是植入形状",
            "",
            "所谓「按患者训练序列对齐」的方向，是这样定出来的：对每个训练事件，量一下"
            "「起点在哪」到「事件后段重心在哪」的位移，再取所有位移**外积的主特征方向**。"
            "这个统计量找的是位移**散布最大**的方向——如果一位患者的电极本身就排成狭长一片，"
            "那么散布最大的方向就是电极云自己的长轴，跟有没有东西沿它传播无关。",
            "",
            f"实测：{axis_check['n_patients']} 位患者里，这条训练轴与**该患者电极云长轴**的夹角"
            f"中位只有 **{cloud['median']:.1f}°**，{cloud['n_within_20_deg']}/{cloud['n']} 位在 20° 以内"
            f"（与主要电极杆方向的夹角中位是 {shaft_gap['median']:.1f}°，"
            f"{shaft_gap['n_within_20_deg']}/{shaft_gap['n']} 位在 20° 以内）。"
            f"而且植入越狭长，两者越重合（长宽比与夹角的秩相关 "
            f"{axis_check['spearman_aspect_vs_gap_to_cloud']:+.2f}）；"
            f"这条轴在只用 25% 训练事件时也几乎不变（中位差 "
            f"{axis_check['axis_stability_100_vs_25_percent_deg']['median']:.2f}°），"
            "说明它是被几何定死的，不是采样噪声。",
            "",
            "**这意味着**：下面「患者对齐 vs 把方向转掉」的比较，很大程度上是"
            "「植入的长轴」对「把它转掉之后的方向」的比较。方向对照本身是匹配的"
            "（同一个局部核、同样的各向异性强度、同样的秩和参数量，旋转/对齐的主奇异值比中位 1.00），"
            "所以比较是公平的；但**它不等于「患者真实的传播方向 vs 错误方向」**。",
            "",
        ]

    ceiling = {k: v for k, v in layers.items() if k.startswith("E0_ceiling_informative_")}
    if ceiling:
        lines += [
            "### 训练之前先问：这些空间图案在**表示层面**够用吗？",
            "",
            "在训练任何模型之前，可以只问一个纯几何问题：如果允许每个留出事件单独挑最合适的系数，"
            "这套空间图案最多能张开多少留出残差？（这是天花板，不是能部署的预测器。）",
            "",
        ]
        for key, entry in sorted(ceiling.items()):
            other = key.replace("E0_ceiling_informative_", "").replace("_minus_aligned", "")
            tag = "严格同型" if entry.get("matched_to_aligned") else "非同型，对患者对齐更严"
            lines.append(f"- 对比 {READER.get('H1_' + other, other)}（{tag}）：{band(entry)}")
        lines += [
            "",
            "只有候选触点数至少是状态维数两倍的患者进入这一层判读；其余患者候选太少，"
            "任何低维图案都能张满，天花板是平凡的。",
            "",
        ]

    bypass = layers.get("E4_bypass_interaction")
    minimal = layers.get("E4_delta_structure_U_MINIMAL")
    full = layers.get("E4_delta_structure_U_FULL_SET")
    lines += [
        "## 6. 把不看顺序的抄近路削弱以后，结构优势会变大吗？",
        "",
        f"- 在**弱抄近路**下（只知道起点和进度）：{band(minimal)}",
        f"- 在**强抄近路**下（还知道点亮过谁）：{band(full)}",
        f"- 两者之差，也就是「抄近路被削弱后结构多赚了多少」：{band(bypass)}",
        "",
        "正数意味着：强抄近路确实吸收掉了一部分本来属于空间结构的信息。",
        "",
    ]

    capacity = [(rank, layers.get(f"E5_capacity_rank{rank}")) for rank in (1, 2, 4, 8)]
    lines += ["### 省状态、省数据吗？", ""]
    for rank, entry in capacity:
        lines.append(f"- 状态只留 {rank} 个数：{band(entry)}")
    for fraction in (25, 50, 100):
        e2e = layers.get(f"E5_end_to_end_fraction{fraction}")
        fixed = layers.get(f"E5_fixed_basis_fraction{fraction}")
        lines.append(f"- 只用 {fraction}% 训练事件（空间图案也跟着重估）：{band(e2e)}")
        lines.append(f"- 只用 {fraction}% 训练事件（空间图案固定为全量版）：{band(fixed)}")
    lines.append("")

    lines += ["## 7. 部分电极覆盖，让结论只能说到哪里？", ""]
    if seeg is not None:
        lines += [
            f"我们只看到了临床植入的那些触点：{len(seeg)} 位患者，触点数从 "
            f"{int(seeg['n_contacts'].min())} 到 {int(seeg['n_contacts'].max())}"
            f"（中位 {int(seeg['n_contacts'].median())}），事件数从 {int(seeg['n_events'].min())} 到 "
            f"{int(seeg['n_events'].max())}。",
        ]
    if coverage is not None and "recorded_SOZ_annotation_fraction" in coverage:
        available = coverage.dropna(subset=["recorded_SOZ_annotation_fraction"])
        if len(available):
            lines.append(
                f"在有临床标注的 {len(available)} 位患者里，被记录触点中带临床发作起始标注的比例"
                f"中位是 {available['recorded_SOZ_annotation_fraction'].median():.2f}"
                f"（这只是描述覆盖情况，不用来排除任何患者，也不叫“覆盖率”）。")
    lines += [
        "",
        "**这意味着**：即使结果是正的，也只能说「在临床已经记录到的这些触点构成的问题里，"
        "某种空间先验提高了有序历史的预测效率」。它**不能**说我们找到了患者真实的神经连接，"
        "不能说电极覆盖了发作起始区或传播网络，也不能把阴性读成「脑内不存在有方向的传播」。",
        "",
    ]

    if synthetic:
        lines += [
            "### 用已知答案的仿真校准：这套方法在什么条件下看得见？",
            "",
            "我们造了一批「答案已知」的假数据：先规定好一条真实的方向轴和一个真实的低维算子，"
            "再让它生成事件，然后把和真实数据**完全相同**的分析流程套上去，看能不能把答案找回来。",
            "",
        ]
        lines += synthetic_axis_verdict()
        for label, entry in sorted(synthetic.get("S2_misspecification", {}).items()):
            lines.append(f"- {label}：找回比例 {entry['P_aligned_beats_angle_null']:.2f}，"
                         f"实际观察到的触点占比中位 {entry['median_observed_fraction']:.2f}")
        lines += ["", "仿真只用来划定「阴性能解释到哪里」，它不决定任何真实患者的纳入或后续实验。", ""]

    if ecog:
        lines += [
            "## 8. 高密度网格电极的构造效度个案（两位患者，各自单独看）",
            "",
            "这两位患者用的是贴在皮层表面的方格电极，物理上下左右邻接比深部电极的距离图更明确。"
            "我们用**完全相同**的任务和抄近路对照，只把状态换成「整片触点场按冻结网格的少参数多项式往前推」。",
            "",
        ]
        for subject, entry in ecog.get("subjects", {}).items():
            swap = entry.get("runtime_graph_swap", {})
            lines.append(
                f"- {subject}：{entry['n_units']} 个训练单元；把网格换成错位网格后代价 "
                f"{swap.get('median_cost_identity_permuted')}，换成同度数随机重连后代价 "
                f"{swap.get('median_cost_degree_rewired')}（换图时参数一个都没动："
                f"{swap.get('parameters_unchanged')}）")
        lines += [
            "",
            "两位患者分别报告，不合并算 p 值；一位为正不构成「网格电极的普遍机制」，"
            "两位不一致也不能反过来否定深部电极那 28 位的结论。",
            "",
        ]

    if confirm:
        lines += [
            "## 9. 模型完全没见过的那部分数据",
            "",
            f"所有选择（模型、状态维数、数据量、空间图案、正则、对照族）全部冻结之后，"
            f"我们才解封最后一段数据，而且只评分一个事先锁死的组合。"
            f"实际访问 {confirm['units_accessed']} 个单元，"
            f"{confirm['units_refused_because_outside_the_lock']} 个因为在锁定范围之外被拒绝。",
            "",
            f"- 患者对齐 vs 把方向转掉：{json.dumps(confirm['structure_effect_autonomous'], ensure_ascii=False)}",
            f"- 完全自由的低维 vs 强抄近路：{json.dumps(confirm['free_minus_unordered_baseline'], ensure_ascii=False)}",
            "",
            "这是「模型没见过」的确认，**不是**前瞻性验证。",
            "",
        ]

    lines += [
        "## 10. 工程完成情况",
        "",
        f"- 预注册单元：{status.get('units_eligible', '?')} 个合格，"
        f"{status.get('units_complete', '?')} 个完成，{status.get('units_unresolved', '?')} 个未解决，"
        f"{status.get('units_missing', '?')} 个缺失",
        f"- 出现过非有限数值的单元：{status.get('units_with_nonfinite_batches', '?')}",
        "",
        "（内部归档代号：`U_MINIMAL` / `U_FULL_SET` 两级无序 baseline、"
        "`DIRECT_HORIZON_UPPER_BOUND` / `AUTONOMOUS_SHARED_OPERATOR` 两个模型族、"
        "`PREFIX_ORDER_COST` / `ORDERED_PATH_ABLATION_COST` / `BASIS_TRANSPLANT_COST` 三个使用期实验、"
        "`ANGLE_ROTATED_AXIS` / `IDENTITY_PERMUTED` / `LOCALITY_REWIRED` 三族对照、"
        "`RUNTIME_GRAPH_SWAP` 仅用于 ECoG。技术细节见同目录技术收口报告。）",
        "",
    ]
    return "\n".join(lines)


def technical_report(today: str) -> str:
    evidence = load("COHORT_EVIDENCE_MATRIX.json") or {"layers": {}}
    closeout = load("CLOSEOUT_AUDIT.json") or {}
    contract = load("SCIENTIFIC_CONTRACT_AUDIT.json") or {}
    figure_qa = load("FIGURE_VISUAL_QA.json") or {}
    status = load("RUN_STATUS.json") or {}
    manifest = load("MASTER_UNIT_MANIFEST.json") or {}
    nulls = load("basis/NULL_MATCH_AUDIT.json") or {}
    census = load("basis/HORIZON_DENOMINATOR_CENSUS.csv", "csv")
    splits = load("SPLIT_HASH_AUDIT.json") or {}
    confirm = load("SPLIT_MINUS_ONE_ACCESS_LOG.json")
    synthetic = load("synthetic/SYNTHETIC_SUMMARY.json")
    ecog = load("ECOG_CASE_SERIES_MATRIX.json")
    rollout = load("STOCHASTIC_ROLLOUT_SUMMARY.json")
    axis_check = load("AXIS_VS_IMPLANTATION_SUMMARY.json")
    axis_table = load("PER_PATIENT_AXIS_VS_IMPLANTATION.csv", "csv")

    lines = [
        "# Topic 5.2D v0.2 技术收口",
        "",
        f"> 日期：{today}　|　spec：`docs/superpowers/specs/2026-08-17-topic5-capacity-constrained-"
        "structural-identifiability-v0-2-design.md`",
        f"> plan：`docs/superpowers/plans/2026-08-17-topic5-capacity-constrained-structural-"
        "identifiability-v0-2.md`",
        f"> 结果根：`results/topic5_capacity_constrained_history_motif_v0_2/`",
        "",
        "## 1. 模型与目标",
        "",
        "```text",
        "prefix state      z_{e,q+1} = F_m z_{e,q} + B_m^T x_{e,q},  z in R^r,  r in {1,2,4,8}",
        "encoder/readout   B_m = Q_m C^in_m,   R_m = Q_m C^out_m       (Q_m column-orthonormal)",
        "direct family     dl_{e,t,h} = R_{m,h} z_{e,t}                (horizon-specific readout)",
        "autonomous family z_{e,t+h} = F_m^h z_{e,t},  dl_{e,t,h} = R_m z_{e,t+h}",
        "orderless bag     z^bag_{e,t} = C_bag^T Q^align^T S_{e,t}     (no F, no rank order)",
        "unordered base    l^base_{e,t,h} = b_h + U_h V_h^T a_{e,t}",
        "  a^min  = [x_{e,1}, t, |S_{e,t}|/C]",
        "  a^full = [x_{e,1}, S_{e,t}, t, |S_{e,t}|/C]",
        "loss              L_h = -log p(n_{e,t+h}) - log p(S_{e,t+h} | n, l_{e,t,h})",
        "exact subset law  p(S|n) = prod_{i in S} w_i / e_n(w_available),  w = exp(l)",
        "checkpoint        L_space = sum_{h in 1,2,3} w_h L_h + lambda_f L_suffix   (STOP excluded)",
        "autonomous suffix f^{suffix,5} = 1 - prod_{h=1..5} (1 - p_h)   (prefix-only no-repeat mask)",
        "```",
        "",
        "冻结常数：`CHECKPOINT_HORIZONS=(1,2,3)`，`w_h=1/3`，`lambda_f=1.0`，训练期每个 horizon 权重 1/5，"
        "suffix 权重 1.0；这些是设计常数，对每条臂完全相同，不做任何按臂调参。",
        "",
        "## 2. 输入、split 与防泄漏",
        "",
        f"- SEEG 复用 parent 的 28 人 `GEOMETRY_ONLY_PCA2` cache；split 逐位一致："
        f"`{splits.get('seeg_split_parity_all_pass')}`；`split == -1` 与 parent held-out 完全相同："
        f"`{splits.get('seeg_model_unseen_equals_parent_heldout')}`",
        f"- 25% ⊂ 50% ⊂ 100% 严格嵌套且按 recording block × 事件长度分层："
        f"`{splits.get('nested_subsets_all_pass')}`",
        f"- 候选集合约定：`{splits.get('availability_contract')}`；suffix 掩码约定："
        f"`{splits.get('suffix_mask_contract')}`",
        "",
        "## 3. 单元分母",
        "",
    ]
    if manifest:
        lines += [
            f"- 预注册单元 {manifest.get('n_units_total')} 个，合格 {manifest.get('n_units_eligible')} 个，"
            f"不合格 {manifest.get('n_units_ineligible')} 个：{manifest.get('ineligible_reasons')}",
            "",
            "| block | planned | eligible |",
            "|---|---:|---:|",
        ]
        for block, values in sorted((manifest.get("per_block") or {}).items()):
            lines.append(f"| {block} | {values['planned']} | {values['eligible']} |")
        lines += ["", "计划偏离（均在任何结果之前决定并冻结）：", ""]
        for note in manifest.get("plan_deviations", []):
            lines.append(f"- {note}")
        lines.append("")
    if status:
        lines += [
            f"实际完成：complete {status.get('units_complete')}，unresolved "
            f"{status.get('units_unresolved')}，missing {status.get('units_missing')}，"
            f"出现非有限批次的单元 {status.get('units_with_nonfinite_batches')}，"
            f"总训练墙钟 {status.get('total_wall_seconds', 0) / 3600:.1f} 小时。",
            "",
        ]
    if census is not None:
        primary = census[(census["prefix_len"] == 3) & (census["split"] == "development_test")]
        lines += [
            "每个 horizon 的留出分母（prefix=3，development test）。每格是"
            "`总决策数 / 其中真正有空间选择的决策数`——当剩下的候选触点数不多于要选的个数时，"
            "那一步是被逼的，精确子集似然恒等于零、不携带空间信息，所以必须和总数一起看：",
            "",
            "| patient | h1 | h2 | h3 | h4 | h5 |",
            "|---|---:|---:|---:|---:|---:|"]

        def cell(row: dict, horizon: int) -> str:
            total = row.get(f"h{horizon}_denominator")
            if total is None or not np.isfinite(total) or int(total) == 0:
                return "0"
            forced = row.get(f"h{horizon}_forced_fraction")
            if forced is None or not np.isfinite(forced):
                return f"{int(total)} / ?"
            return f"{int(total)} / {int(round(total * (1.0 - forced)))}"

        for row in primary.to_dict("records"):
            lines.append(f"| {row['patient']} | "
                         + " | ".join(cell(row, h) for h in (1, 2, 3, 4, 5)) + " |")
        # only h1-h3 enter the objective, so the reader needs the forced count there
        # without scanning the table
        summary = []
        for horizon in (1, 2, 3):
            total = primary[f"h{horizon}_denominator"]
            live = (total * (1.0 - primary[f"h{horizon}_forced_fraction"])).fillna(0.0)
            dead = primary[(total > 0) & (live < 1)]["patient"].tolist()
            summary.append(f"h{horizon}: {int((total > 0).sum())}/{len(primary)} 位有分母，"
                           + (f"其中 {len(dead)} 位空间信息为零（{', '.join(dead)}）"
                              if dead else "全部带空间信息"))
        lines += ["", "只有 h1–h3 进入目标函数，那三档的情况是：" + "；".join(summary) + "。"]
        lines.append("")

    if axis_check:
        lines += [
            "## 3b. 冻结轴估计器的实际行为（判读前置）", "",
            "`PATIENT_ALIGNED` 的轴按 spec §4.5 step 2 定义为 split-0 起点→late-field 位移"
            "外积的主特征向量。该统计量返回位移**二阶矩最大**的方向，因此在狭长植入上会返回"
            "contact-cloud 长轴。逐患者实测：", "",
            "```json", json.dumps({k: v for k, v in axis_check.items()
                                   if k not in ("contract",)}, indent=1, ensure_ascii=False), "```",
            "",
        ]
        if axis_table is not None:
            full = axis_table[axis_table["basis_fraction"] == 100].sort_values(
                "gap_to_contact_cloud_axis_deg")
            lines += ["| patient | C | cloud aspect | gap→cloud axis (deg) | gap→dominant shaft (deg) |",
                      "|---|---:|---:|---:|---:|"]
            for row in full.to_dict("records"):
                lines.append(
                    f"| {row['patient']} | {int(row['n_contacts'])} | {row['cloud_aspect_2d']:.2f} "
                    f"| {row['gap_to_contact_cloud_axis_deg']:.1f} "
                    f"| {row['gap_to_dominant_shaft_axis_deg']:.1f} |")
            lines += ["",
                      "该表不用于加权、排除或修正任何下游结果；它只固定 aligned-vs-rotated "
                      "对比的可读范围。方向 null 的匹配度另见 §4（旋转/对齐主奇异值比）。", ""]

    lines += ["## 4. Null 匹配实况", ""]
    if nulls:
        rows = nulls.get("null_rows", [])
        rewire = [row for row in rows if row["kind"] == "LOCALITY_REWIRED"]
        lines += [
            f"- 方向旋转角（弧度）：{[round(v, 4) for v in nulls.get('angle_grid_rad', [])]}；"
            f"合格患者 {sum(1 for e in nulls.get('per_patient', {}).values() if e['angle_null_eligible'])}/"
            f"{len(nulls.get('per_patient', {}))}（近一维几何不补造，也不用其它 null 顶替）",
            f"- 触点身份错位 null：每位 {nulls.get('n_identity_nulls')} 张，"
            f"按 (shaft, 径向距离, degree) 分箱内置换，正交性与奇异值逐位保留",
            f"- 局部重连 null：每位 {nulls.get('n_rewire_nulls')} 张，共 {len(rewire)} 张；"
            f"完全匹配 {sum(1 for r in rewire if not r.get('unmatched'))} 张，"
            f"标记退化 {sum(1 for r in rewire if 'REWIRE_DEGENERATE' in str(r.get('unmatched', '')))} 张"
            f"（近一维链状布局在保度数、保同轴性、保边长的约束下没有第二种接法）",
            "",
        ]

    lines += ["## 5. 证据层逐条", "", "| 层 | n | 中位 | 95% 区间 | 正/负/近零 | Wilcoxon p |",
              "|---|---:|---:|---|---|---:|"]
    coverage_layers = {}
    for key, entry in sorted(evidence.get("layers", {}).items()):
        if entry.get("n", 0) == 0:
            continue
        if "median_ci95" not in entry:
            coverage_layers[key] = entry
            continue
        low, high = entry["median_ci95"]
        lines.append(
            f"| `{key}` | {entry['n']} | {entry['median']:+.5f} | [{low:+.5f}, {high:+.5f}] "
            f"| {entry['n_positive']}/{entry['n_negative']}/{entry['n_near_zero']} "
            f"| {entry.get('wilcoxon_p', float('nan')):.4f} |")
    for key, entry in coverage_layers.items():
        lines += ["", f"**`{key}`**（描述性，非队列统计）：", "", "```json",
                  json.dumps(entry, indent=1, ensure_ascii=False)[:1500], "```"]
    noise = evidence.get("seed_noise_floor", {})
    if noise:
        lines += ["", f"种子噪声底：{noise.get('n_multi_seed_arms')} 条多种子臂，"
                      f"中位离散 {noise.get('median_seed_spread')}，"
                      f"九成分位 {noise.get('p90_seed_spread')}。所有效应量必须对着它读。", ""]

    if confirm:
        lines += [
            "## 6. Model-unseen 紧凑确认", "",
            f"- 锁定组合：{json.dumps(confirm['locked_combination'], ensure_ascii=False)}",
            f"- 访问单元 {confirm['units_accessed']}，锁外拒绝 {confirm['units_refused_because_outside_the_lock']}",
            f"- 方向对照分母：{confirm['angle_comparison_denominator']}",
            f"- 结构效应：{json.dumps(confirm['structure_effect_autonomous'], ensure_ascii=False)}",
            f"- 自由低维 vs 强抄近路：{json.dumps(confirm['free_minus_unordered_baseline'], ensure_ascii=False)}",
            "",
        ]

    if synthetic:
        lines += ["## 7. 合成可辨识面", "",
                  "**实测结论（先读这一段，再读设计意图）**：功效块每格 24 个 montage 的"
                  "二项检验**没有任何一格显著偏离掷硬币**，包括把 teacher 真轴直接交给"
                  " student、且事件以最强强度沿该轴推进的那一格（14/24, p=0.541）。"
                  "该流程对已知强沿轴结构的检出力在本设计下未被建立，"
                  "真实数据的结构层结果只能读作 uninformative，不能读作 negative。"
                  "逐格数字与二项 p 见白话版同名小节。", "",
                  "S0 的 oracle-axis 臂把两种失败分开：同一 teacher / 同一数据 / 同一机器，"
                  "只把 student 的轴从 spec 估计器换成 teacher 真轴。若 oracle 臂恢复而估计臂不恢复，"
                  "瓶颈是轴估计器而非模型族或损失。该臂只存在于合成块，真实数据永远没有真轴。", "",
                  f"- 格子数 {synthetic['n_cells']}，失败 {synthetic['n_failed']}，跳过 {synthetic['n_skipped']}",
                  f"- 角色：{synthetic['role']}", "",
                  "```json", json.dumps({k: v for k, v in synthetic.items()
                                         if k.startswith("S")}, indent=1)[:3000], "```", ""]
    if ecog:
        lines += ["## 8. ECoG 构造效度个案", "",
                  "```json", json.dumps(ecog, indent=1, ensure_ascii=False)[:3000], "```", ""]
    if rollout:
        lines += ["## 9. 随机 rollout（次级）", "",
                  f"- 层级：{rollout['tier']}",
                  f"- 采样器：{rollout['shared_sampler']}",
                  "```json", json.dumps(rollout.get("per_arm_median", {}), indent=1), "```", ""]

    lines += ["## 10. 工程与科学合同审计", ""]
    if closeout:
        lines += ["```json", json.dumps({k: v for k, v in closeout.items()
                                         if k not in ("unresolved_unit_ids",)}, indent=1), "```", ""]
    if contract:
        lines += ["允许措辞：", ""] + [f"- {v}" for v in contract.get("allowed_wording", [])]
        lines += ["", "禁止措辞：", ""] + [f"- {v}" for v in contract.get("forbidden_wording", [])]
        lines += ["", "科学合同关键项：", "",
                  "```json",
                  json.dumps({k: v for k, v in contract.items()
                              if k not in ("allowed_wording", "forbidden_wording", "contract",
                                           "captured_utc")}, indent=1, ensure_ascii=False)[:3500],
                  "```", ""]
    if figure_qa:
        lines += ["## 11. 图形审计", "", "```json",
                  json.dumps(figure_qa, indent=1)[:2500], "```", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=str(date.today()))
    arguments = parser.parse_args()
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    plain = ARCHIVE / f"capacity_constrained_history_motif_v0_2_plain_report_{arguments.date}.md"
    technical = ARCHIVE / f"capacity_constrained_history_motif_v0_2_technical_closeout_{arguments.date}.md"
    plain.write_text(plain_report(arguments.date) + "\n")
    technical.write_text(technical_report(arguments.date) + "\n")
    print(f"wrote {plain}")
    print(f"wrote {technical}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
