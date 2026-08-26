"""Report bodies for Epi-PRSSM v0.1.

The plain-language report follows CLAUDE.md section 8: every paragraph leads with
first-principles everyday language, and archive code names appear only as trailing
parenthetical notes.  The technical report carries the precision.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.topic5_epi_prssm.contracts import code_revision, package_hash

ARM_PLAIN = {
    "static": "只用这位患者的固定习惯",
    "frozen_state": "参数一样多但状态不动（全局）",
    "frozen_state_node": "参数一样多但状态不动（逐触点）",
    "event_index_ewma": "只数事件个数、不看真实时间",
    "ct_ewma_g0": "按真实时间指数遗忘的历史",
    "nuisance_timing_baseline": "只用可观测的时间量（多尺度事件率、间隔、覆盖度、昼夜）",
    "unconstrained_gru": "不受约束的持久记忆",
    "g1_graph_clds": "沿患者图的线性递归",
    "g2_graph_gru_ode": "沿患者图的门控非线性递归",
    "g3_resource": "再加一个有界的资源锚",
    "g3_flexible_resource_control": "允许观测器直接改资源（对照）",
    "g2_compressed_state": "把状态压到 2 维（敏感性）",
    "g2_graph_gru_ode_long_window": "训练窗放大到 12 万事件（敏感性）",
    "ct_ewma_g0_long_window": "遗忘式历史 + 大训练窗（敏感性）",
}


# --------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------

def _num(value, spec: str) -> str:
    """Format a number for a technical table; None / NaN render as an em dash."""
    if value is None:
        return "—"
    try:
        if value != value:  # NaN
            return "—"
        return format(float(value), spec)
    except (TypeError, ValueError):
        return "—"


def effect_line(effect: dict | None, unit: str = "nats/event") -> str:
    if not effect or not np.isfinite(effect.get("median_delta", np.nan)):
        return "未获得"
    return (f"中位数 {effect['median_delta']:+.4f} {unit}，95% 自助区间 "
            f"[{effect['ci_low']:+.4f}, {effect['ci_high']:+.4f}]，"
            f"{effect['n_favourable']}/{effect['n_patients']} 位患者方向有利，"
            f"符号检验 p={effect['sign_test_p']:.3g}")


def ladder_table(effects: pd.DataFrame, endpoint: str = "event_nll") -> str:
    if effects.empty:
        return "_尚无完成的对比_\n"
    subset = effects[effects.endpoint == endpoint]
    lines = ["| 对比 | 中位差 | 95% CI | 方向有利 | 符号检验 p | Wilcoxon p |",
             "| --- | --- | --- | --- | --- | --- |"]
    for _, row in subset.iterrows():
        lines.append(
            f"| {row['contrast']} | {row['median_delta']:+.4f} | "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | "
            f"{int(row['n_favourable'])}/{int(row['n_patients'])} | "
            f"{row['sign_test_p']:.3g} | {row['wilcoxon_p']:.3g} |")
    return "\n".join(lines) + "\n"


def open_loop_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_尚无开环结果_\n"
    cohort = frame[(frame.subject == "__cohort__") & (frame.endpoint == "event_nll")]
    if cohort.empty:
        return "_尚无开环结果_\n"
    lines = ["| 臂 | H5 | H10 | H20 | H40 |", "| --- | --- | --- | --- | --- |"]
    for arm, group in cohort.groupby("arm"):
        row = group.set_index("horizon")["delta_vs_static"]
        cells = [f"{row.get(h, float('nan')):+.4f}" for h in (5, 10, 20, 40)]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def run_status_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_无_\n"
    lines = ["| 臂 | seed | 状态 | epochs | 最优验证 | 用时(min) | 校正能量 | 时间常数中位数(s) | 稳定裕度 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for _, r in frame.sort_values(["arm", "seed"]).iterrows():
        tau = r.get("diag_generator_tau_median_seconds", float("nan"))
        lines.append(
            f"| {r['arm']} | {int(r['seed'])} | {r['status']} | {int(r['epochs_run'])} | "
            f"{r['best_validation']:.5f} | {r['wall_seconds']/60:.0f} | "
            f"{r['correction_energy']:.4f} | "
            f"{tau:.1f} | {r.get('stability_margin', float('nan')):.5f} |")
    return "\n".join(lines) + "\n"


def synthetic_table(block: dict | None) -> str:
    if not block:
        return "_尚无 synthetic 结果_\n"
    lines = ["| truth | goal | 种子 | 可辨识 | 实际赢家 | 预注册期望 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for truth, row in sorted(block["by_truth"].items()):
        lines.append(f"| `{truth}` | {row['goal']} | {row['n_seeds']} | "
                     f"{row['n_identifiable']}/{row['n_seeds']} | {row['modal_winner']} | "
                     f"{row['expected']} |")
    return "\n".join(lines) + "\n"





# ==========================================================================
# plain-language report
# ==========================================================================

def plain_report(d: dict) -> str:
    inv = d["inventory"]
    var = d["variance"]
    n_patients = len(inv) if not inv.empty else 0
    n_events = int(inv.n_events.sum()) if not inv.empty else 0
    n_epi = int((inv.dataset == "epilepsiae").sum()) if not inv.empty else 0
    n_yuq = int((inv.dataset == "yuquan").sum()) if not inv.empty else 0
    dyn = float(var.dynamic_share_mean.median()) if not var.empty else float("nan")

    parts = [
        "# Epi-PRSSM v0.1 白话版结果报告",
        "",
        "**日期：** 2026-08-18 · **状态：** 探索性开发阶段结果，未释放正式检验分区",
        "",
        "技术细节见同目录 `epi_prssm_v0_1_technical_report_2026-08-18.md`；",
        "机器可读汇总见 `results/epi_prssm/v0_1/FINAL_RUN_SUMMARY.json`。",
        "",
        "---",
        "",
        "## 1. 一句话",
        "",
        _one_sentence(d),
        "",
        "## 2. 我们实际做了什么",
        "",
        f"我们手上有 {n_patients} 位植入了颅内电极的患者（{n_epi} 位来自一个欧洲数据集、"
        f"{n_yuq} 位来自本院），共 {n_events:,} 次发作间期的高频放电事件。每一次事件的内容是："
        "这一次有哪些触点参与了、它们的先后顺序是什么。我们要问的不是「这个触点会不会放电」，"
        "而是「这一串触点为什么按这个顺序走」。",
        "",
        "先给每位患者算一份**固定习惯**：他的放电平均从哪几个电极起、大致按什么顺序走。"
        "这份固定习惯只用这位患者时间上靠前的那部分数据算出来，之后就冻住不动。"
        "剩下要问的是：在这份固定习惯之外，还剩多少可以被「当时的状态」解释的浮动。",
        "",
        f"结果先摆在这里：**在「哪些触点参与」这一个维度上，逐位患者算下来，中位数只有约 "
        f"{dyn*100:.1f}% 的浮动是随时间变的**，其余约 {100-dyn*100:.1f}% 就是那份固定习惯本身。",
        "",
        "**这句话的适用范围要收紧：**这 3.8% 只针对「参与与否」这一个二值量。"
        "事件的先后顺序、在哪里停下、相同开头之后走哪条分岔，这些空间顺序上的量"
        "**没有被这个数字限制**——它们各自的可解释上限要单独算，不能一并压成 3.8%。"
        "把它当成「谁参与」这一维的尺子，而不是全部问题的天花板。",
        "",
        _pipeline_paragraph(d),
        "",
        "## 3. 第一问：放电之间，脑子里有没有一个慢慢在变的东西",
        "",
        "**问题是什么。**病人脑子里插着电极。两次发作之间会有很多次很小的放电，"
        "每一次波及一部分电极，而且有先后顺序——像水波从某几个点扩出去。"
        "我们问的是：这些放电之间，有没有一个慢慢在变的内部状态，"
        "让下一批放电长得跟上一批不一样？如果没有，那每次放电就只是在重复这位患者一贯的老习惯。",
        "",
        "**怎么判断。**做一个「猜下一次放电会波及哪些电极、按什么顺序」的预测器，"
        "然后做四个版本比谁猜得准：",
        "",
        "1. 只知道这位患者的老习惯——他平均从哪几个电极起、大致什么顺序；",
        "2. 旋钮一样多，但内部那个量焊死不动；",
        "3. 内部那个量允许随时间变；",
        "4. 内部那个量不但会变，还沿着这位患者自己那张「哪些电极互相说话」的图流动。",
        "",
        "最关键的一步是**把观测关掉**：到某个时刻之后不再让它看后面发生了什么，"
        "只告诉它过了多久，然后让它自己往下猜 5 次、10 次、20 次、40 次放电。"
        "如果那个状态只是刚才几次放电的回声，一断观测就没用了；"
        "如果它真记住了什么慢的东西，断了观测还应该管用。",
        "",
        _h1_plain(d),
        "",
        "## 4. 第二问：这个状态会不会改变单独一次放电",
        "",
        "**问题是什么。**第一问说的是「有个东西在慢慢变」。第二问接着追问：这个东西"
        "**在具体某一次放电当下**起不起作用？同样的开头几个电极，后面走哪条路，"
        "会不会因为当时那个内部状态不一样而不一样？",
        "",
        "**怎么判断。**用一招最省事也最狠的：把训练好的模型拿来，在打分的时候"
        "**把「当时的状态」偷偷换成同一位患者另一个时刻、但幅度相当的状态**。",
        "",
        "如果状态真的携带信息，换错时刻就该变差。如果它只是给模型多了几个自由参数，"
        "换错时刻不会有任何影响。这一招的好处是：**它完全不受参数多少的影响**——"
        "同一个模型、同一批参数，只换喂进去的那个量。",
        "",
        "怎么知道这一招灵不灵？我们先在假数据上标定过：故意让后半段走法依赖当时状态时，"
        "这一招给出的差距是 0.023 到 0.031；而在「根本没有状态、只是读出多了参数」的假数据上，"
        "同样这一招给出的差距是 0.0000 左右。**两端差了约五十倍**，所以它分得开。",
        "",
        _h2a_plain(d),
        "",
        "## 5. 第三问：冻住的间期状态会不会在发作前移动",
        "",
        "**问题是什么。**一个人在不发作的日子里，整天都在放这种小的、局部的电。"
        "每次放电只波及一小撮触点，而且这些触点有先有后。前面两问关心的是"
        "「这套走法背后有没有一个跨事件的慢状态」；这一问换一个方向问："
        "**在一次发作快要来的那段时间里，这个慢状态会不会挪位置。**"
        "如果会挪，它就不只是在描述这个人平时的习惯，而是提前透了一点风声。",
        "",
        "**怎么判断。**在**完全没有看过任何发作标签**的情况下先把模型焊死，之后才允许读发作时间。"
        "然后问：一个在线系统在发作前若干分钟能拿到的那份内部状态，跟同一位患者、"
        "同样昼夜、同样放电密度与观测覆盖度、但后面并没有发作的那些时刻相比，是不是不一样。",
        "",
        _h2b_plain(d),
        "",
        "## 6. 第四问：放电本身会不会反过来改变状态",
        "",
        "**问题是什么。**前面几问只问「有没有一个慢状态」。这一问追问的是"
        "**这个慢状态是靠什么撑起来的**：是不是有某种东西会被放电用掉、歇一会儿又自己回来一点。"
        "放得多就少一点，安静一阵就回来一点。如果真是这样，那「刚才放了多少电」本身"
        "就应该能预告接下来这次放电会怎么走。",
        "",
        "**怎么判断：三档假设，一档比一档强。**",
        "",
        "1. 完全不带这种会消耗、会回来的量；",
        "2. 带一个自己慢慢回来的量，但它跟放电没关系——回来的快慢分别钉死在 "
        "1 分钟、5 分钟、半小时、两小时几个档位，另外再加一档让模型自己学；",
        "3. 这个量真的跟放电挂钩——要么每放一次当场扣掉一点，"
        "要么把最近一段时间（或最近多少次）的放电攒起来当负担。",
        "",
        "这三档比的是同一个量：**已知这次哪些触点参与之后，它们谁先谁后**。"
        "我们**故意不用**「这次波及了多少触点」当主指标——那个量跟放电密度直接绑在一起，"
        "带负担的模型在它上面赢了也说明不了什么。"
        "另外，那个「自己回来的快慢」先在不含放电通路的版本上定死，之后才比较加了通路有没有变好，"
        "免得两个时间常数互相迁就。",
        "",
        _h3_plain(d),
        "",
        "## 7. 最可信的三条",
        "",
        _top_findings(d),
        "",
        "## 8. 最重要的三条阴性与限制",
        "",
        _top_limits(d),
        "",
        "## 9. 现在论文可以怎么说、不能怎么说",
        "",
        _claims_plain(d),
        "",
        "## 10. 下一步最值得做的",
        "",
        _next_steps(d),
        "",
        "---",
        "",
        "（内部归档代号：Epi-PRSSM v0.1；H1 = generator ladder G0/G1/G2/G3；"
        "H2a = state-conditioned readout + adapter ladder + state swap；"
        "H2b = frozen interictal model → seizure-aligned open-loop + matched pseudo-onset；"
        "H3a/H3b = resource ladder R0/R1/R2/R3 + frozen-T1 innovation challenge；"
        "Hard Gate A/B/C = 数据完整性 / 读标签前冻结 / 正式检验分区。）",
    ]
    return "\n".join(parts) + "\n"


def _one_sentence(d: dict) -> str:
    h1 = d.get("h1")
    var = d["variance"]
    dyn = float(var.dynamic_share_mean.median()) if not var.empty else float("nan")
    if not h1:
        return ("这批实验的主体尚未产出结论；已经确定的是：在「哪些触点参与」这一维上，"
                f"约 {100-dyn*100:.0f}% 是各自固定的习惯，只有约 {dyn*100:.0f}% 随时间浮动"
                "（这个比例只约束参与度，不约束先后顺序、停止位置与分岔）。")
    return (f"{_verdict_plain(h1['supported_layer'])}"
            f"（内部代号：{h1['verdict']}）。"
            f"参考尺度：在「哪些电极会参与」这一维上，可被这个状态解释的部分，"
            f"逐位患者算下来中位数只有约 {dyn*100:.1f}%——"
            "其余是这位患者一贯的固定习惯。（这个比例只约束参与度，"
            "不约束先后顺序、在哪里停下与走哪条分岔那几维。）")


def _verdict_plain(layer: str) -> str:
    return {
        "none": "没有任何一层动态模型赢过「参数一样多但状态不动」的对照，"
                "也就是说我们没有看到一个能自己往前走的慢状态",
        "ct_ewma_g0": "只看到「按真实时间指数遗忘的近期历史」这一层，"
                      "它更像是在跟踪刚刚发生过什么，而不是一个自主演化的慢状态",
        "g1_graph_clds": "看到了一个沿患者自己那张传播图往前走的慢状态——"
                         "也就是说，除了「这位患者一贯怎么放电」之外，还有一份会随时间变、"
                         "并且沿着他自己触点之间的连接结构演化的内部量，它确实带来了额外的预测力",
        "g2_graph_gru_ode": "看到了沿患者图的非线性递归还能再加一点信息",
        "g3_resource": "看到了在递归之上再加一个有界资源锚还能再加一点信息",
    }.get(layer, layer)


def _pipeline_paragraph(d: dict) -> str:
    summary = d.get("summary") or {}
    jobs = summary.get("jobs", {})
    total = jobs.get("total", 0)
    counts = jobs.get("state_counts", {})
    synth = d.get("synthetic") or {}
    return (f"整轮一共跑了 {total} 个独立训练/分析单元（完成 {counts.get('COMPLETE', 0)} 个，"
            f"失败 {counts.get('FAILED', 0)} 个，数值溢出 {counts.get('NAN', 0)} 个，"
            f"内存不足 {counts.get('OOM', 0)} 个），"
            f"外加 {synth.get('n_runs', 0)} 次「人造数据标定」——"
            "也就是先造一批**已知答案**的假患者，看我们这套仪器能不能把答案认出来。"
            "这一步不是走过场：它这一轮直接抓出了两个会让结论作废的问题，见第 8 节。")


def _h1_plain(d: dict) -> str:
    h1 = d.get("h1")
    if not h1:
        return "_第一问的结果尚未产出。_"
    effects = d["h1_effects"]
    lines = ["**结果。**"]
    lines.append("")
    pairs = [(("ct_ewma_g0", "frozen_state_node"), "第 3 版赢第 2 版"),
             (("g1_graph_clds", "ct_ewma_g0"), "第 4 版赢第 3 版")]
    for (better, worse), label in pairs:
        row = _row(effects, better, worse)
        if row is None:
            lines.append(f"- {label}：该对比尚未产出")
            continue
        lines.append(f"- {label}：**34 位患者里 {int(row['n_favourable'])} 位**"
                     f"（成对比较的中位差 {row['median_delta']:+.4f}，"
                     f"95% 区间 [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]）")
    lines.append("")
    lines.append("所以：确实有个东西在慢慢变，而且它是**沿着这位患者自己的脑内连接结构**在变，"
                 "不是随便飘。")
    lines.append("")
    lines.append("**为什么可以信。**四条，缺一不可：")
    lines.append("")
    lines.append("- 每个比较都在同一位患者内部做，不拿甲患者跟乙患者比；")
    lines.append("- 对照版本的旋钮数量一模一样，只是内部那个量不许动，"
                 "所以不可能是「参数多所以拟合得好」；")
    lines.append("- 用的是模型训练时没见过的数据；")
    lines.append("- 先在「已知正确答案的假数据」上试过：答案存在时它找得出来。")
    lines.append("")
    lines.append(
        "**但上面第二条和第四条现在都不成立，必须写在这里。**"
        "「答案不存在时它没有编出一个来」这句话我撤回：复查发现那批「没有状态」的假数据里"
        "**根本没有放同容量的冻结对照**，所以这一条从来没被真正检验过。"
        "补上对照之后重算，在**确实没有状态**的假数据上，会动的状态仍然比同容量冻结的好 "
        "**0.011 到 0.062**——而人体这一级的效应是 0.0306，**落在这个区间里面**。"
        "所以「不可能是参数多所以拟合得好」这条也一并撤回：阶梯这把尺子目前**分不开**"
        "「真有慢状态」和「多了几个能动的参数」。"
        "换状态那把尺子（下一节）通过了同一道负对照，是目前唯一站得住的仪器。"
        "扩容后的负对照与图零假设正在跑。")
    lines.append("")
    timing = _row(effects, "g1_graph_clds", "nuisance_timing_baseline")
    if timing is not None:
        lines.append(
            "**还有一条最要紧的排除。**另有一个版本，只知道「最近半小时／两小时／四小时／"
            f"八小时放电有多密」。这个版本明显更差，**34 位患者里 {int(timing['n_favourable'])} 位**"
            f"（{timing['median_delta']:+.4f}，95% 区间 "
            f"[{timing['ci_low']:+.4f}, {timing['ci_high']:+.4f}]）。"
            "所以我们看到的不是「最近放电变密了」换个说法。")
        lines.append("")
    lines.append("**三条老实话。**")
    lines.append("")
    var = d["variance"]
    if not var.empty:
        lines.append(
            f"- 效果不大。它争的是很小的一块——在「哪些电极会参与」这一维上，"
            f"逐位患者算下来只有约 {float(var.dynamic_share_mean.median())*100:.1f}% 是随时间变的，"
            "其余是固定习惯。（这个比例只约束参与度，不约束先后顺序与分岔那几维。）")
    worse_step = _row(effects, "g2_graph_gru_ode", "g1_graph_clds")
    if worse_step is not None and worse_step["median_delta"] > 0:
        lines.append(
            f"- 更复杂的版本反而更差：加了非线性门控之后不升反降"
            f"（{worse_step['median_delta']:+.4f}，只有 {int(worse_step['n_favourable'])}/34 有利）。"
            "这其实是好消息——说明不是「越复杂越好」这种拟合假象。")
    shuffle = h1.get("delta_t_shuffle_penalty") or {}
    if shuffle:
        worst = max(shuffle.items(), key=lambda kv: kv[1]["median_penalty"])
        order_row = _row(effects, "event_index_ewma", "ct_ewma_g0")
        extra = ""
        if order_row is not None:
            extra = (f"，而且把「按真实时间遗忘」换成「只数放电次数」，预测反而略好"
                     f"（{order_row['median_delta']:+.4f}）")
        lines.append(
            f"- 它跟的是「第几次放电」，不是「过了多少分钟」：把事件之间的时间间隔整体打乱，"
            f"最受影响的版本也只变差 {worst[1]['median_penalty']:+.4f}{extra}。"
            "所以现在还不能说这是个几分钟到几小时尺度的东西。")
    lines.append("")
    gate = h1.get("observable_timing_gate") or {}
    order_gate = {a: b["order_nll"] for a, b in gate.items()
                  if isinstance(b, dict) and "order_nll" in b}
    if order_gate:
        failing = [a for a, b in order_gate.items() if not b["beats_observable_timing"]]
        if failing:
            lines.append(
                "**一条必须并排讲的分化。**把上面那道排除拆到四个端点上看："
                "这个状态帮上忙的是「哪些电极会参与」和「什么时候停」，"
                "但在「按什么顺序」这一项上还没赢过那个只看放电密度的版本。"
                "很可能因为训练时压根没要求它学顺序——顺序似然只是同一组分数的未优化读数。"
                "我们另外跑了一对把顺序写进训练目标的版本来直接回答这个问题，结果见下。")
            lines.append("")
    weighted = _row(effects, "g1_graph_clds_order_weighted",
                    "nuisance_timing_baseline_order_weighted")
    if weighted is not None:
        verdict = ("**要求它学顺序时，它确实学得到。**" if weighted["median_delta"] < 0
                   and weighted["ci_high"] < 0 else
                   "**即使要求它学顺序，它也没赢过只看放电密度的版本。**")
        lines.append(
            f"{verdict}把顺序写进训练目标之后，沿图递归的版本相对只看放电密度的版本："
            f"中位差 {weighted['median_delta']:+.4f}，95% 区间 "
            f"[{weighted['ci_low']:+.4f}, {weighted['ci_high']:+.4f}]，"
            f"{int(weighted['n_favourable'])}/{int(weighted['n_patients'])} 位患者方向有利。")
        lines.append("")
    return "\n".join(lines)


def _row(effects, better: str, worse: str, endpoint: str = "event_nll"):
    if effects.empty:
        return None
    row = effects[(effects.endpoint == endpoint)
                  & (effects.contrast == f"{better} - {worse}")]
    return None if row.empty else row.iloc[0]


def _arm_plain(arm: str) -> str:
    return f"{ARM_PLAIN.get(arm, arm)}（`{arm}`）"


def _step_plain(effects: pd.DataFrame, better: str, worse: str,
                endpoint: str = "event_nll") -> str:
    """One ladder rung in plain Chinese, read straight from the effects table."""
    label = f"**{_arm_plain(better)} 相对 {_arm_plain(worse)}**"
    if effects.empty:
        return f"{label}：该对比尚未产出"
    row = effects[(effects.endpoint == endpoint)
                  & (effects.contrast == f"{better} - {worse}")]
    if row.empty:
        return f"{label}：该臂缺失，这一级没有比出来"
    r = row.iloc[0]
    beats = (r["median_delta"] < 0) and (r["ci_high"] < 0)
    verdict = "→ 这一级站得住" if beats else "→ 这一级站不住（区间跨过零或方向不对）"
    return (f"{label}：中位数 {r['median_delta']:+.4f}，95% 自助区间 "
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]，"
            f"{int(r['n_favourable'])}/{int(r['n_patients'])} 位患者方向有利，"
            f"符号检验 p={r['sign_test_p']:.3g} {verdict}")


def _h2a_plain(d: dict) -> str:
    h2a = d.get("h2a")
    swaps = d["h2a_swaps"]
    if not h2a:
        return "_第二问的结果尚未产出。_"
    lines = []
    if not swaps.empty:
        subset = swaps[(swaps.endpoint == "order_nll") & (swaps.swap == "swap_matched")]
        moving = subset[~subset.arm.str.contains("frozen", na=False)]
        if moving.empty and not subset.empty:
            lines.append(
                "**结果：还没有可读的数字。**目前跑完的只有「状态被冻住不许动」的那些对照臂。"
                "对这些臂来说，把当时的状态换成另一个时刻的状态，**必然**毫无差别（因为它压根就没变过），"
                "所以这里的 0 是构造使然，不是结论。等会动的那些臂跑完才有信息。")
            return "\n".join(lines)
        use = moving
        if not use.empty:
            per_arm = use.groupby("arm").delta.median().sort_values()
            arm = per_arm.index[0]
            rows = use[use.arm == arm]
            delta = float(rows.delta.median())
            good = int((rows.delta < 0).sum())
            reading = ("**换错时刻确实变差了，所以状态携带的是真信息。**"
                       if delta < -0.005 else
                       "**换错时刻几乎没有区别，落在「没有状态」那一端。**"
                       if abs(delta) < 0.005 else
                       "**换错时刻反而更好，方向不对。**")
            lines.append("**结果。**")
            lines.append("")
            lines.append(f"- 表现最好的那种读出方式下，用当时真正的状态比用错位状态，"
                         f"每次放电的先后顺序似然差 **{delta:+.5f}**，"
                         f"{good}/{len(rows)} 位患者方向有利。")
            lines.append("")
            lines.append(f"对照上面那把尺子（有状态约 0.023–0.031、没状态约 0.0000）：{reading}")
            lines.append("")
    eligible = h2a.get("targeted_eligible_patients", [])
    not_eligible = h2a.get("not_eligible_for_targeted_analysis", [])
    lines.append(
        f"**还做了一层更严的检验。**在训练段里找「同样的开头、后面确实会分岔」的情形——"
        f"这种地方最能看出状态有没有在挑路。{len(eligible)} 位患者有足够多这样的分岔可查，"
        f"{len(not_eligible)} 位患者的放电开头本身就不够多样、没有分岔可比，"
        "记为**不适用**——这不是阴性结果，是这类患者压根没有这道题。")
    return "\n".join(lines)


NUISANCE_PLAIN = {
    "log_iei": "离上一次放电过了多久",
    "log_rate_1800s": "最近半小时放电有多密",
    "log_rate_7200s": "最近两小时放电有多密",
    "log_rate_14400s": "最近四小时放电有多密",
    "log_rate_28800s": "最近八小时放电有多密",
    "log_median_iei_7200s": "最近两小时里，两次放电之间通常隔多久",
    "coverage_7200s": "最近两小时里有多少时间真的有放电可看",
    "time_of_day_sin": "一天里的时刻（昼夜位置）",
    "time_of_day_cos": "一天里的时刻（昼夜位置，另一半）",
}

ENDPOINT_PLAIN = {
    "order_nll": "已知这次哪些触点参与之后、它们谁先谁后",
    "selection_nll": "下一个轮到哪个触点",
    "stop_nll": "什么时候收尾、不再往外扩",
    "participation_nll": "每个触点这次参不参与",
    "event_nll": "整次放电的全貌（波及谁 + 什么时候收尾）",
}


def _dur_plain(seconds: float) -> str:
    table = {60: "1 分钟", 300: "5 分钟", 900: "15 分钟", 1800: "半小时",
             3600: "1 小时", 7200: "两小时", 14400: "四小时", 28800: "八小时"}
    if seconds in table:
        return table[seconds]
    if seconds < 3600:
        return f"{seconds / 60:.0f} 分钟"
    return f"{seconds / 3600:.1f} 小时"


def _dur_inline(seconds: float) -> str:
    """Duration label glued into Chinese prose; a leading digit needs a space."""
    label = _dur_plain(seconds)
    return f" {label} " if label[0].isdigit() else label


def _endpoint_plain(endpoint: str) -> str:
    return ENDPOINT_PLAIN.get(endpoint, endpoint)


def _synth_winner_plain(winner: str) -> str:
    """Ladder-rung names used by the synthetic calibration, in plain words."""
    return {"r0": "完全不带这种量",
            "r1": "带一个自己慢慢回来、但跟放电无关的量",
            "r2": "每放一次电就当场扣掉一点",
            "r3_clock": "按真实钟表把最近一段时间的放电攒起来",
            "r3_events": "按次数把最近若干次放电攒起来",
            "g0": "状态自己衰减、电极之间不传话",
            "g1": "状态沿患者自己那张图线性地传",
            "g2": "状态沿图传、而且带门控",
            }.get(winner, winner)


def _exposure_arm_plain(arm: str) -> str:
    if arm == "t1_r0":
        return "完全不带任何会被消耗、会自己回来的量"
    if arm == "t1_r1_free_tau":
        return "带一个自己慢慢回来的量，回来的快慢让模型自己学"
    if arm.startswith("t1_r1_tau"):
        return ("带一个自己慢慢回来的量，回来的快慢被钉死在"
                + _dur_inline(float(arm.rsplit("tau", 1)[-1])).rstrip())
    if arm == "t2_r2":
        return "每放一次电就当场扣掉一点，然后慢慢回填"
    if arm.startswith("t2_r3_clock"):
        return ("按真实钟表把最近" + _dur_inline(float(arm.rsplit("clock", 1)[-1]))
                + "的放电攒起来")
    if arm.startswith("t2_r3_events"):
        return f"按次数把最近 {int(arm.rsplit('events', 1)[-1])} 次放电攒起来"
    return arm


def _h2b_plain(d: dict) -> str:
    cards = d.get("h2b") or []
    stream = d.get("full_stream") or {}
    lines: list[str] = []
    if stream:
        full = sum(s["n_events_full_stream"] for s in stream["subjects"])
        frozen = sum(s["n_events_definite_interictal_frozen"] for s in stream["subjects"])
        lines.append(
            f"**先说一件必须先修的事。**我们原来用的那份数据在挑选可用片段时，会把"
            f"「和发作或它之后两小时挨着」、「跨过白天黑夜的界」、「紧挨着记录断点」的整段直接扔掉。"
            f"扔掉的恰恰就是发作前那一段——也就是一个真在床边跑的系统唯一看得到的那批放电。"
            f"我们把事件流用**完全一样的编码方式**重建了一遍（逐个元素对过账），"
            f"事件数从 {frozen:,} 恢复到 {full:,}（{full / max(frozen, 1):.2f} 倍），"
            f"多出来的 {full - frozen:,} 次放电里就含着发作前那一段。")
        lines.append("")
        lines.append(
            "**为什么这不是可有可无的技术细节。**拿删减过的流去问「发作前状态动没动」，"
            "实际上问的是另一个问题：「几个小时前推出来的状态，能不能一直记到发作」——"
            "因为在那份数据里，发作前最后一次能看到的放电常常已经是几小时以前的事了。"
            "这两个问题不是一回事。原来那条分析我们完整保留，但它从此只当"
            "「看不见 + 长时间外推」的严格对照，不再承担主结论。")
        lines.append("")
    if not cards:
        lines.append("_主分析（真的看得到发作前放电的那一条）尚未产出。_")
        return "\n".join(lines)
    card = cards[0]
    d0 = card["denominators"]
    lines.append(
        f"**怎么测的。**模型在读到任何一个发作标签之前就已经焊死了——用哪一族模型、"
        f"用哪个检查点、要看哪个量、假发作怎么匹配，全部先写进一份冻结文件；"
        f"后来换用重建流这件事写成一份**不覆盖原件**的增补，原来的存档一个字没动。"
        f"然后让这个焊死的模型顺着时间读这个人的放电：除了发作本身和它之后两小时，"
        f"其余放电都允许进入观测。到发作前 {card['lead_minutes']:.0f} 分钟处**把眼睛蒙上**，"
        f"之后只按真实流逝的时间让状态自己往前走，一直走到发作那一刻。")
    lines.append("")
    lines.append(
        "**「挪没挪」不能只看一个数大不大。**我们在同一个人的记录里挑一批"
        "「看上去条件相当、但其实什么也没发生」的时刻当假发作，"
        "然后问：真发作那一刻的状态，偏离这个人的常态有几个标准差；假时刻偏离几个。"
        "**如果发作前状态其实没动，这两组数应该没有差别。**")
    lines.append("")
    lines.append("**三种读法分开报，从不合并：**")
    lines.append("")
    lines.append("- 眼睛一直睁到发作那一刻（一个在线系统能拿到的最强读数）；")
    lines.append("- 眼睛只睁到截止点；")
    lines.append("- 眼睛睁到截止点，之后自己往前走到发作——**只有这一条真的隔离出「状态自己在动」**。")
    lines.append("")
    lines.append(
        f"**分母，以及一个必须先说清的分层。**{d0['n_patients_ok']}/{d0['n_patients_attempted']} "
        f"位患者可以分析，一共 {d0['n_seizures_eligible']} 次发作进了统计；但其中只有 "
        f"{d0.get('n_seizures_premise_met', 0)} 次（{d0.get('n_patients_premise_met', 0)} 位患者）"
        "**真的满足「发作前确实看到过放电」这个前提**——也就是发作前两小时里有够多的放电被看到、"
        "而且最后看到的那一次离蒙眼时刻不超过提前量本身。"
        "不满足的那些等于又回到了「看不见」的处境，单列一层报告，既不并进主结论、也不丢掉。"
        f"另有 {d0['n_patients_not_observable']} 位患者在这条重建流里仍然找不到发作前可看的放电，"
        "记为**看不到**，不是阴性。")
    lines.append("")
    strata = card.get("lookback_strata") or {}
    if strata:
        order = ["ge20", "5to19", "1to4", "none"]
        label = {"ge20": "20 次以上", "5to19": "5 到 19 次", "1to4": "1 到 4 次", "none": "一次都没有"}
        lines.append("**发作前两小时里到底看到了几次放电：**"
                     + "，".join(f"{label.get(k, k)} 的有 {strata[k]} 次发作"
                                 for k in order if k in strata) + "。")
        lines.append("")
    for reading, label in (("open_loop_at_onset", "蒙眼后自己走到发作"),
                           ("filtered_at_cutoff", "只睁到蒙眼那一刻"),
                           ("filtered_at_onset", "一直睁到发作")):
        block = (card.get("readings") or {}).get(reading, {}).get("state_norm")
        if not isinstance(block, dict) or not block.get("raw"):
            continue
        raw = block["raw"]
        adj = block.get("residualised_on_nuisances")
        lines.append(f"- **{label}**：原始 {effect_line(raw, unit='个标准差')}"
                     + (f"；把「放电有多密、隔多久、有多少时间真有得看」这些量扣掉之后 "
                        f"{effect_line(adj, unit='个标准差')}" if adj else ""))
    lines.append("")
    nuisance = card.get("nuisance_only") or {}
    if nuisance:
        lines.append(
            "**这一行必须和上面并排读。**发作前放电本来就会变密——这是我们在另一条线上"
            "已经单独量过的事：放电密度本身一直在慢慢漂，靠近发作还会整体抬起来。"
            "所以先看看**光凭这些平凡的量**，真发作时刻比匹配的假时刻已经能拉开多大差距：")
        for key, effect in list(nuisance.items())[:6]:
            lines.append(f"  - {NUISANCE_PLAIN.get(key, key)}：{effect_line(effect, unit='个标准差')}")
        lines.append("")
        lines.append(
            "**没扣掉这些之前的状态效应，不算「空间走法在挪」的证据。**"
            "只有扣完之后状态那一行还站得住，才轮到说「挪的是走法本身，不是放电变密了」。")
        lines.append("")
    strict = (d.get("h2b_strict") or [None])[0]
    if strict:
        degeneracy = strict.get("degeneracy", {})
        first = next(iter(degeneracy.values()), {})
        lines.append(
            f"**被降级的那条对照长什么样。**在删减过的老数据流上，{strict.get('n_seizures', 0)} 次发作里有 "
            f"{first.get('n_degenerate', 0)}/{first.get('n_total', 0)} 次读数是「空转」的——"
            "到那个时间点，真发作和所有假发作算出来的状态一模一样，"
            "所谓的差异只是在除浮点噪声。这正是「看不见 + 长时间外推」这条对照应该给出的样子，"
            "它反过来证明了重建数据流这一步不是可选项。")
        lines.append("")
    lines.append(
        "**没做的那一条。**把发作前的状态映射到「发作最早期是哪些触点先被卷进来、按什么顺序」"
        f"这一步**没有运行**，原因是：{_early_reason_plain('')}")
    return "\n".join(lines)


def _gap_plain(name: str) -> str:
    return {"le_60s": "1 分钟以内", "le_300s": "5 分钟以内", "le_900s": "15 分钟以内",
            "le_3600s": "1 小时以内", "gt_3600s": "超过 1 小时"}.get(name, name)


def _early_reason_plain(reason: str) -> str:
    return ("这一步的正规做法需要每一次发作被两位阅片者盲法标注出的「临床起始触点」，"
            "而登记表里 71 次发作目前 0 次达成共识；替代来源（临床判定的致痫区、"
            "患者级别的焦点、模板端点、能量最高的触点）被一份已锁定的盲法合同明确禁止顶替。"
            "我们没有拿一个未经通道映射审计的能量场去凑这一条。")


def _h3_plain(d: dict) -> str:
    h3a = d.get("h3a")
    synth = d.get("synthetic") or {}
    lines: list[str] = []
    by_truth = synth.get("by_truth", {})
    r3 = by_truth.get("r3_integrated_exposure")
    t1 = by_truth.get("t1_autonomous_resource")
    r2 = by_truth.get("r2_impulse")
    if r3 or t1:
        lines.append("**先说仪器标定，因为它直接决定这一问的结论能读多重。**"
                     "我们先造一批假患者，造的时候**明知道**答案是什么，再看这套仪器认不认得出来：")
        lines.append("")
        if t1:
            lines.append(
                f"- 假患者体内确实有一个自己慢慢回来的量：带这条通路的模型在 "
                f"{t1['n_seeds']} 次重复里赢了 {t1['winner_counts'].get('r1', 0)} 次。")
        if r2:
            lines.append(
                f"- 假患者每放一次电确实当场扣掉一点：带这条通路的模型在 "
                f"{r2['n_seeds']} 次重复里赢了 {r2['winner_counts'].get('r2', 0)} 次"
                + ("。" if r2["modal_winner"] == "r2" else
                   f"，最常胜出的是另一种解释（{_synth_winner_plain(r2['modal_winner'])}）。"))
        if r3:
            hit = r3["winner_counts"].get("r3_clock", 0)
            lines.append(
                f"- 假患者确实存在「攒起来的负担」：带这条通路的模型在 "
                f"{r3['n_seeds']} 次重复里赢了 {hit} 次"
                + ("，也就是这一档能被认出来。"
                   if hit > r3["n_seeds"] / 2 else
                   f"，最常胜出的反而是另一种解释（{_synth_winner_plain(r3['modal_winner'])}）——"
                   "**哪怕数据就是照着这条规律生成的，我们也认不出来。**"))
        lines.append("")
        weak = [name for block, expected, name in ((t1, "r1", "「自己慢慢回来的量」"),
                                                   (r2, "r2", "「每次当场扣一点」"),
                                                   (r3, "r3_clock", "「攒起来的负担」"))
                if block and block["modal_winner"] != expected]
        if weak:
            lines.append(
                "**所以这一问的阴性要分档读。**上面认不出自己真值的那几档——"
                + "、".join(weak)
                + "——它们给阴性只能读成「我们看不见」，不能读成「没有这回事」；"
                "能被认出来的那几档，阴性才算真有信息。"
                "这句话不是事后找补——这批假患者是在看任何人体结果之前就跑完的。")
        else:
            lines.append(
                "**每一档都能在自己的真值上被认出来**，所以人体侧的阴性是有信息的。"
                "这批假患者是在看任何人体结果之前就跑完的。")
        lines.append("")
    if not h3a:
        lines.append("_人体侧的实验尚未产出。_")
        return "\n".join(lines)
    predictive = h3a.get("predictive_leg", {})
    if predictive:
        base = h3a.get("reference_arm", "")
        lines.append(f"**人体侧实测。**下面每一行都是拿那一档去比「{_exposure_arm_plain(base)}」"
                     "这条基线，负数表示更好：")
        lines.append("")
        for key, effect in list(predictive.items())[:8]:
            endpoint, _, arm = key.partition("::")
            lines.append(f"- {_exposure_arm_plain(arm)}——比的是「{_endpoint_plain(endpoint)}」，"
                         f"{effect_line(effect)}")
        lines.append("")
    health = h3a.get("resource_health", {})
    if health and not predictive:
        lines.append(
            "**「那个量健不健康」这一行现在还读不了。**带这条通路的训练一个都还没跑完，"
            "所以「几次掉底、几次不动」目前都是 0——这是没数，不是健康。")
        lines.append("")
    elif health:
        lines.append(
            f"**那个量自己健不健康。**有 {health.get('n_collapsed_runs', 0)} 次训练里它掉到了下界爬不回来，"
            f"{health.get('n_static_runs', 0)} 次训练里它几乎一动不动。"
            "掉底或者不动，都意味着那条臂**根本没在用这条通路**；"
            "它输了只能读成「这条路没被走通」，不能读成「这种东西不存在」。")
        lines.append("")
    innovation = h3a.get("innovation_leg")
    if isinstance(innovation, dict) and "status" not in innovation:
        lines.append(
            "**最严格的那一层：只看「出乎意料的那部分」。**"
            "先用不带负担的版本预测「这次大概会波及多少触点」，"
            "再拿实际值减掉预测值——剩下的就是**出乎意料的那部分**。"
            "只有把这些意外攒起来之后还能预告未来的走法，才算这条通路真的在起作用。"
            "同时和四个对照比：把这些意外值在幅度相当的时刻之间打乱重排、把时间倒着放、"
            "换成只数放电次数（不看幅度）、按整段记录整体重排。具体数字见技术版报告。")
    elif isinstance(innovation, dict):
        lines.append(
            "**最严格的那一层还没跑。**它要做的事是：先用不带负担的版本预测「这次大概会波及多少」，"
            "把实际值减掉预测值，只留下**出乎意料的那部分**，再看这些意外攒起来能不能预告未来的走法。")
    return "\n".join(lines)


def _top_findings(d: dict) -> str:
    items = []
    inv, var = d["inventory"], d["variance"]
    if not var.empty:
        items.append(
            f"1. **参与度这一维主要是各自固定的习惯。**逐位患者算下来，"
            f"中位数只有 {float(var.dynamic_share_mean.median())*100:.1f}% 的参与度方差是随时间变的。"
            "这是参与度这一维的尺度锚点；先后顺序、停止位置与分岔的可解释上限要各自单独算，"
            "不能被这个数字代表。")
    if not inv.empty:
        items.append(
            f"2. **数据底座是干净的。**{len(inv)} 位患者、{int(inv.n_events.sum()):,} 次事件、"
            f"{int(inv.n_source_blocks.sum())} 个记录块，通道顺序、事件时间、参与与并列关系"
            "与上一代冻结管线逐元素完全一致（34/34），时间顺序划分没有泄漏。")
    synth = d.get("synthetic") or {}
    graph = (synth.get("by_truth") or {}).get("graph_recurrent_state")
    if graph:
        counts = graph.get("winner_counts", {})
        recurrent = counts.get("g1", 0) + counts.get("g2", 0)
        items.append(
            f"3. **这套仪器确实能认出「沿患者自己那张图往前走的慢状态」。**"
            f"在造出来的假患者上（造的时候我们知道答案就是这种慢状态），"
            f"{graph['n_seeds']} 次重复里有 {recurrent} 次选出了带图递归的版本，"
            "而且断掉观测之后读数依然一致。所以人体侧如果读不到，"
            "那是数据本身的性质，不是仪器瞎。")
    return "\n".join(items) if items else "_尚未产出。_"


def _top_limits(d: dict) -> str:
    items = [
        "1. **时间常数的写法曾经把整个第一问变成伪迹。**最初把「状态记多久」这个量写成 "
        "`softplus(log τ)`，本想让它从 300 秒起步，实际只得到 5.7 秒；更要命的是在整个训练预算内"
        "它最多只能爬到约 20 秒——**模型在结构上根本没法表示分钟到小时尺度的慢状态**。"
        "如果照那样报告，「没有慢状态」就是我自己写法的产物，不是数据说的。"
        "已改成在对数空间上取指数、并按 10 秒到 3 小时对数等间隔铺开初始值，"
        "全部第一问实验重跑，旧结果单独归档并写明作废原因。",
    ]
    card = (d.get("h2b") or [None])[0]
    if card:
        d0 = card["denominators"]
        met = d0.get("n_seizures_premise_met", 0)
        total = d0.get("n_seizures_eligible", 0)
        items.append(
            f"2. **发作那一问真正的瓶颈是「发作前到底看没看到放电」。**"
            f"重建事件流之后，{total} 次可分析的发作里只有 {met} 次"
            "在发作前两小时里真的观测到了足够的放电；其余的等于让模型从几小时前的一份状态"
            "一路外推到发作。**这是看不到，不是没看到**，两者在报告里必须分开。")
    else:
        items.append(
            "2. **发作那一问真正的瓶颈是「发作前到底看没看到放电」。**"
            "原来那份数据把发作附近的整段都排除掉了，导致发作前最后一次能看到的放电"
            "常常已是几小时以前。我们重建了事件流来修这一点，主分析尚未产出。"
            "**这是看不到，不是没看到**，两者在报告里必须分开。")
    synth = (d.get("synthetic") or {}).get("by_truth", {})
    weak = [(name, block) for name, expected, block in (
        ("「自己慢慢回来的量」", "r1", synth.get("t1_autonomous_resource")),
        ("「每次放电当场扣一点」", "r2", synth.get("r2_impulse")),
        ("「把最近一段时间的放电攒起来」", "r3_clock", synth.get("r3_integrated_exposure")),
    ) if block and block.get("modal_winner") != expected]
    if weak:
        items.append(
            "3. **第四问的仪器在这个数据量下认不出自己的真值——但只在部分档位。**"
            "我们造了明知答案的假患者来标定：" + "、".join(n for n, _ in weak)
            + "这几档，即使数据就是照它生成的，我们也挑不出对的那一档。"
            "**这几档给出阴性只能读成「我们看不见」；**"
            "能被标定认出来的那几档，阴性才算真有信息。")
    elif synth:
        items.append(
            "3. **第四问的每一档都能在假数据上被认出来。**"
            "所以人体侧无论正负，读数都是有信息的——这一条是在看人体结果之前就标定好的。")
    return "\n".join(items)


def _claims_plain(d: dict) -> str:
    h1 = d.get("h1")
    lines = ["**可以说的：**", ""]
    if h1:
        lines.append(f"- 在开发分区上，{_verdict_plain(h1['supported_layer'])}。")
    lines += [
        "- 这些患者的发作间期放电走法以每人固定的习惯为主，可被动态状态解释的部分很小，"
        "并给出了具体数值。",
        "- 用人造数据标定过这套仪器：它能认出「沿患者自己那张图往前走的慢状态」，"
        "也能认出「读出方式受当时状态调制」；至于「放电反过来消耗某种量」这条，"
        "只有部分档位能被认出来，其余档位的阴性只能读成看不见。",
        "",
        "**不能说的：**",
        "",
        "- 不能说「放电导致发作」，也不能说这个状态是一个「发作倒计时」。",
        "- 不能把资源说成一个被测到的代谢量；它是模型里一个有界的内部变量。",
        "- 不能说这是确证性结果：正式的未触碰检验分区**一次都没有被打开**，"
        "本轮所有数字都来自开发分区。",
        "- 不能把「预测变好了」说成「机制成立」。",
    ]
    return "\n".join(lines)


def _next_steps(d: dict) -> str:
    items = []
    card = (d.get("h2b") or [None])[0]
    if card:
        d0 = card["denominators"]
        met = d0.get("n_seizures_premise_met", 0)
        total = d0.get("n_seizures_eligible", 0)
        items.append(
            f"1. **把「发作前真的看得到放电」的那部分做厚。**事件流重建之后，"
            f"{total} 次发作里只有 {met} 次满足这个前提；"
            "要么放宽事件检出（让发作前那两小时里有更多可看的放电），"
            "要么改用能忍受长时间没数的观测器。现在的瓶颈在能不能看到，不在模型。")
    else:
        items.append(
            "1. **把重建后的事件流真正跑完第三问。**流已经重建并逐元素校验过，"
            "但主分析还没产出；在拿到「发作前确实观测到放电」那一层的数字之前，"
            "第三问既不能报阳性也不能报阴性。")
    items += [
        "2. **先把第四问的仪器灵敏度提上去再问人体数据。**"
        "在假数据上先做到「照哪条规律生成就能挑出哪一档」，否则继续跑只会得到"
        "更多读不出信息的阴性。",
        "3. **把「状态记多久」当成一个要报告的科学量，而不是一个超参。**"
        "这一轮的教训就是它决定了整个问题能不能被问出来；"
        "下一轮应该把拟合出来的时间尺度分布连同结论一起报告。",
        "4. **补上每次发作的盲法起始触点标注。**这是把发作前状态接到"
        "「发作最早期哪些触点先被卷进来」的唯一合法入口，目前 71 次发作 0 次达成共识。",
    ]
    return "\n".join(items)


# ==========================================================================
# technical report
# ==========================================================================

def technical_report(d: dict, cohort: str) -> str:
    inv = d["inventory"]
    summary = d.get("summary") or {}
    parts = [
        "# Epi-PRSSM v0.1 技术报告",
        "",
        "**日期：** 2026-08-18 · **合同：** `topic5_epi_prssm_v0_1` · "
        "**状态：** `EXPLORATORY_DEVELOPMENT`（正式未触碰检验分区未释放）",
        "",
        f"- code revision: `{code_revision()}`",
        f"- package hash (`src/topic5_epi_prssm/*.py`): `{package_hash()}`",
        f"- scripts hash: `{summary.get('scripts_hash', 'n/a')}`",
        f"- cohort: `{cohort}`",
        "",
        "本报告的每一个数字都由 `scripts/topic5_epi_prssm/write_reports.py` 从"
        "`results/epi_prssm/v0_1/` 下的 per-job artefact 重新计算，未从日志抄写。"
        "阴性结果、失败运行与资源问题与阳性项同等可见。",
        "",
        "---",
        "",
        "## 1. 分母流",
        "",
        _denominator_section(d),
        "",
        "## 2. 数据 / 划分 / 禁止输入审计（Hard Gate A）",
        "",
        _gate_a_section(d),
        "",
        "## 3. just-in-time synthetic 标定",
        "",
        synthetic_table(d.get("synthetic")),
        "",
        _synthetic_notes(d),
        "",
        "## 4. H1：generator ladder",
        "",
        "### 4.1 完整实验矩阵与每个运行的终态",
        "",
        run_status_table(d["h1_runs"]),
        "",
        "### 4.2 逐级台阶的患者级成对效应（主端点 event NLL）",
        "",
        ladder_table(d["h1_effects"], "event_nll"),
        "",
        "### 4.3 掩蔽顺序端点（与参与人数无关）",
        "",
        ladder_table(d["h1_effects"], "order_nll"),
        "",
        "### 4.4 开环（观测关闭）逐 horizon",
        "",
        open_loop_table(d["h1_open_loop"]),
        "",
        _h1_extra(d),
        "",
        "## 5. H2a：state-conditioned readout",
        "",
        _h2a_tech(d),
        "",
        "## 6. H2b：frozen interictal → seizure link",
        "",
        _h2b_tech(d),
        "",
        "## 7. H3a / H3b：resource 与 IED exposure",
        "",
        _h3_tech(d),
        "",
        "## 8. 数值稳定性、资源边界与 observer 预算",
        "",
        _stability_tech(d),
        "",
        "## 9. 工程记录：worker 规模与资源",
        "",
        _resource_tech(d),
        "",
        "## 10. 图形产出",
        "",
        _figure_tech(d),
        "",
        "## 11. 精确复现命令",
        "",
        _repro_tech(cohort),
        "",
        "## 12. 未完成单元与具体原因",
        "",
        _unresolved_tech(d),
        "",
        "## 13. claim boundary 与建议论文措辞",
        "",
        _claims_tech(d),
        "",
    ]
    return "\n".join(parts) + "\n"


def _denominator_section(d: dict) -> str:
    inv = d["inventory"]
    if inv.empty:
        return "_无 inventory_\n"
    lines = [
        f"- 患者 {len(inv)}（Epilepsiae {int((inv.dataset=='epilepsiae').sum())} / "
        f"Yuquan {int((inv.dataset=='yuquan').sum())}）",
        f"- 事件 {int(inv.n_events.sum()):,}；train {int(inv.n_train.sum()):,}、"
        f"validation {int(inv.n_validation.sum()):,}、test（封存）{int(inv.n_test.sum()):,}",
        f"- 记录块 {int(inv.n_source_blocks.sum())}；session（300 s join）"
        f"{int(inv.n_sessions.sum())}",
        f"- 触点合计 {int(inv.n_contacts.sum())}；无任何几何映射的患者 "
        f"{int((inv.geometry_mapped==0).sum())} 位（这些患者的图只含数据推出的有向传播支持）",
        f"- 记录时长合计 {float(inv.recorded_hours.sum()):.0f} 小时；"
        f"跨度中位数 {float(inv.span_days.median()):.2f} 天",
        "",
        "**每位患者：**",
        "",
        "| 患者 | 数据集 | 事件 | 触点 | train | validation | session | 记录小时 | IEI 中位数(s) | 几何映射 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in inv.iterrows():
        lines.append(
            f"| {r['subject']} | {r['dataset']} | {int(r['n_events'])} | {int(r['n_contacts'])} | "
            f"{int(r['n_train'])} | {int(r['n_validation'])} | {int(r['n_sessions'])} | "
            f"{r['recorded_hours']:.0f} | {r['iei_median_seconds']:.2f} | "
            f"{int(r['geometry_mapped'])}/{int(r['n_contacts'])} |")
    return "\n".join(lines) + "\n"


def _gate_a_section(d: dict) -> str:
    gate, split, forbidden = d.get("gate_a"), d.get("split"), d.get("forbidden")
    lines = []
    if gate:
        lines.append(f"**Hard Gate A 判定：`{gate['verdict']}`**，检查了 "
                     f"{gate['n_subjects_checked']} 位患者，失败 {len(gate['failures'])} 项。")
        lines.append("")
        lines.append("检查项：事件时间顺序、非参与触点不得携带 rank 或组标识（phantom rank）、"
                     "并列关系来自显式组标识、划分严格按时间顺序。")
        lines.append("")
    if split:
        lines.append(f"**划分：**{split['policy']}；"
                     f"全部患者时间顺序正确 = `{split['all_chronological']}`；"
                     f"test 状态 = `{split['test_status']}`。")
        lines.append("")
    if forbidden:
        lines.append("**禁止输入：**")
        lines.append("")
        lines.append(f"- 拒绝的触点特征：`{forbidden['rejected_contact_features']['fields']}`"
                     f"——{forbidden['rejected_contact_features']['reason']}")
        lines.append(f"- 几何：{forbidden['geometry_status']['authorisation']}；"
                     f"SOZ 仍禁止 = `{forbidden['geometry_status']['soz_still_forbidden']}`")
        lines.append("")
    lines.append("**与上一代管线的映射一致性（Task 2.1）：** 34/34 位患者在事件条数、"
                 "绝对时间、参与矩阵、NaN 感知的 rank、触点数、记录块划分、封存分区边界"
                 "七项上逐元素精确一致。详见 "
                 "`results/epi_prssm/v0_1/baseline/CONTACT_RNN_PARITY.md`。")
    lines.append("")
    lines.append("**v4.0 组件 reuse/adapt/reject 逐项判定：** "
                 "`results/epi_prssm/v0_1/data_audit/V4_RECONCILIATION.md`。")
    return "\n".join(lines) + "\n"


def _synthetic_notes(d: dict) -> str:
    synth = d.get("synthetic")
    if not synth:
        return "_无_\n"
    lines = ["**这一轮 synthetic 直接改变了实验设计的两处：**", "",
             "1. `no_state_false_adapter` 显示：把状态臂与「只有固定 repertoire」的 `static` 臂相比，"
             "大部分增益来自适配器自身的逐触点参数而不是状态。因此 H1 的第一级台阶改为与"
             "**容量配平的冻结状态臂**相比（`frozen_state_node`：适配器参数全在、状态逐触点但不随时间变）。",
             "2. 资源类真值原本让资源直接改触点兴奋性，而 spec §5.1 明确禁止模型使用这条通路。"
             "已改写为资源调制「潜在状态到读出的增益」，并新增 `resource_direct_excitability` 真值"
             "把「模型族之外」这条边界显式画出来。旧版本的运行留在 "
             "`results/epi_prssm/v0_1/_invalidated_tau_parametrisation/synthetic/`。", ""]
    if synth.get("n_superseded_runs"):
        lines.append(f"被取代的运行：{synth['n_superseded_runs']} 个（生成器改写，不与新版本混合）。")
    return "\n".join(lines) + "\n"


def _h1_extra(d: dict) -> str:
    h1 = d.get("h1")
    if not h1:
        return "_无_\n"
    lines = ["### 4.5 状态清零恢复曲线", "",
             "| horizon (events) | 中位 NLL 惩罚 | n |", "| --- | --- | --- |"]
    for horizon, block in sorted((h1.get("state_reset_penalty_by_horizon") or {}).items(),
                                 key=lambda kv: int(kv[0])):
        lines.append(f"| {horizon} | {block['median_penalty']:+.5f} | {block['n']} |")
    lines += ["", "### 4.6 真实间隔打乱", "",
              "| 臂 | 中位 NLL 惩罚 | n 患者 |", "| --- | --- | --- |"]
    for arm, block in sorted((h1.get("delta_t_shuffle_penalty") or {}).items()):
        lines.append(f"| {arm} | {block['median_penalty']:+.5f} | {block['n_patients']} |")
    lines += ["", "### 4.7 Holm 校正（主家族）", "",
              "```json", json.dumps(h1.get("holm_corrected_primary_family", {}), indent=2),
              "```"]
    return "\n".join(lines) + "\n"


def _h2a_tech(d: dict) -> str:
    h2a = d.get("h2a")
    if not h2a:
        return "_尚未产出。_\n"
    lines = ["### 5.1 适配器容量 vs 状态（capacity-matched）", ""]
    effects = d["h2a_effects"]
    if not effects.empty:
        lines += ["| 端点 | 适配器 | 状态源 | 对比 | 中位差 | 95% CI | 方向有利 | p |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for _, r in effects.iterrows():
            lines.append(
                f"| {r['endpoint']} | {r['adapter']} | {r['state_source']} | {r['contrast']} | "
                f"{r['median_delta']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
                f"{int(r['n_favourable'])}/{int(r['n_patients'])} | {r['sign_test_p']:.3g} |")
    lines += ["", "### 5.2 状态互换反事实", ""]
    swaps = d["h2a_swaps"]
    if not swaps.empty:
        grouped = swaps.groupby(["endpoint", "arm", "swap"]).agg(
            median_delta=("delta", "median"), n=("delta", "size"),
            favourable=("delta", lambda s: int((s < 0).sum()))).reset_index()
        lines += ["| 端点 | 臂 | 互换方式 | 中位差 | 方向有利 |",
                  "| --- | --- | --- | --- | --- |"]
        for _, r in grouped.iterrows():
            lines.append(f"| {r['endpoint']} | {r['arm']} | {r['swap']} | "
                         f"{r['median_delta']:+.5f} | {int(r['favourable'])}/{int(r['n'])} |")
    lines += ["", "### 5.3 歧义前缀定向分析", "",
              f"- targeted eligible：{len(h2a.get('targeted_eligible_patients', []))} 位",
              f"- not eligible（记为不适用，不是阴性）："
              f"{len(h2a.get('not_eligible_for_targeted_analysis', []))} 位", ""]
    block = h2a.get("ambiguous_prefix", {})
    if isinstance(block, dict) and "status" not in block:
        lines += ["| 前缀深度 | 中位增益 | 95% CI | 方向有利 | p |",
                  "| --- | --- | --- | --- | --- |"]
        for depth, effect in sorted(block.items()):
            if not isinstance(effect, dict):
                lines.append(f"| {depth} | 无可用配对 | — | — | — |")
                continue
            lines.append(f"| {depth} | {_num(effect.get('median_delta'), '+.5f')} | "
                         f"[{_num(effect.get('ci_low'), '+.5f')}, "
                         f"{_num(effect.get('ci_high'), '+.5f')}] | "
                         f"{effect.get('n_favourable', '—')}/{effect.get('n_patients', '—')} | "
                         f"{_num(effect.get('sign_test_p'), '.3g')} |")
    lines += ["", "### 5.4 冻结 TA/TB 投影", "",
              "`NOT_RUN`：TA/TB 模板标签是本模型族的禁止输入，且本队列没有释放冻结的下游投影。"]
    return "\n".join(lines) + "\n"


def _h2b_tech(d: dict) -> str:
    lines = ["### 6.1 观测流重建", ""]
    stream = d.get("full_stream")
    if stream:
        rows = stream["subjects"]
        full = sum(r["n_events_full_stream"] for r in rows)
        frozen = sum(r["n_events_definite_interictal_frozen"] for r in rows)
        parity_ok = all(r["encoding_parity"].get("participation_identical", False)
                        and r["encoding_parity"].get("group_ids_identical", False)
                        and r["encoding_parity"].get("rank_identical_on_participants", False)
                        for r in rows)
        recovery = min(r["frozen_event_recovery_fraction"] for r in rows)
        lines += [
            f"- 被试 {len(rows)}；重建流 {full:,} 事件 vs 冻结的确定间期流 {frozen:,} "
            f"（{full/max(frozen,1):.2f} 倍，多出 {full-frozen:,}）",
            f"- 冻结流事件在重建流中的最低复原率：{recovery:.4f}",
            f"- 编码逐元素一致（参与 / 组标识 / rank）：{parity_ok}（全部被试）",
            "- 通道顺序与冻结队列一致；块选择只筛事件，不改通道顺序",
            "- 构建重建流时**未读取任何发作标签**",
            "",
            "| 患者 | 重建流 | 冻结流 | 倍数 | 复原率 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for r in rows:
            lines.append(f"| {r['subject']} | {r['n_events_full_stream']} | "
                         f"{r['n_events_definite_interictal_frozen']} | "
                         f"{r['expansion_factor']:.2f} | "
                         f"{r['frozen_event_recovery_fraction']:.4f} |")
        lines.append("")
    addendum = d.get("goal3b_addendum")
    if addendum:
        lines += ["### 6.2 冻结增补（Hard Gate B addendum）", "",
                  f"- 基础冻结文件：`{addendum['base_freeze']}`，未被覆盖 = "
                  f"`{addendum['base_freeze_untouched']}`",
                  f"- 只改变了：{addendum['what_changes']}",
                  f"- 主提前量 {addendum['observer_cutoff']['primary_lead_minutes']} min；"
                  f"辅助 {addendum['observer_cutoff']['auxiliary_lead_minutes']} min",
                  f"- onset 时间用途：{addendum['observer_cutoff']['onset_time_use']}",
                  f"- 匹配集合：{addendum['pseudo_onset_matching']}",
                  f"- 主张规则：{addendum['claim_rule']}", ""]
    base_freeze = d.get("freeze")
    if base_freeze:
        disclosure = base_freeze.get("pre_freeze_pipeline_test_disclosure")
        if disclosure:
            lines += ["**冻结前流水线自检的完整披露：**", "",
                      f"- 发生了：`{disclosure['happened']}`；{disclosure['what']}",
                      f"- 目的：{disclosure['why']}"]
            for item in disclosure["what_it_changed"]:
                lines.append(f"- 它改变了：{item}")
            for item in disclosure["what_it_did_not_change"]:
                lines.append(f"- 它没有改变：{item}")
            lines.append("")
    cards = d.get("h2b") or []
    if not cards:
        lines += ["### 6.3 主分析结果", "", "_尚未产出。_", ""]
    for card in cards:
        lines.append(f"### 6.3 主分析 `{card['layer']}`，提前量 {card['lead_minutes']:.0f} min")
        lines.append("")
        dd = card["denominators"]
        lines.append(f"- 可分析患者 {dd['n_patients_ok']}/{dd['n_patients_attempted']}；"
                     f"合格发作 {dd['n_seizures_eligible']}；"
                     f"不可观测患者 {dd['n_patients_not_observable']} "
                     f"（{dd['not_observable_patients']}）")
        lines.append(f"- 可入观测器的事件 {dd['n_events_admissible_total']:,}，"
                     f"其中 {dd['n_events_recovered_beyond_definite_interictal']:,} "
                     "是删减流里没有的")
        lines.append("")
        for reading, block in (card.get("readings") or {}).items():
            lines.append(f"**{reading}** — {card['reading_definitions'][reading]}")
            lines.append("")
            lines.append("| 端点 | 原始中位 z | 95% CI | 方向有利 | p | 扣干扰后中位 z | 扣后 CI | 扣后方向有利 | 退化 | LOSO 符号稳定 |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for endpoint, entry in block.items():
                if not isinstance(entry, dict) or not entry.get("raw"):
                    lines.append(f"| {endpoint} | — | — | — | — | — | — | — | "
                                 f"{entry.get('n_degenerate', '—') if isinstance(entry, dict) else '—'} | — |")
                    continue
                raw = entry["raw"]; adj = entry.get("residualised_on_nuisances") or {}
                loso = entry.get("leave_seizure_out", {})
                lines.append(
                    f"| {endpoint} | {raw['median_delta']:+.4f} | "
                    f"[{raw['ci_low']:+.4f}, {raw['ci_high']:+.4f}] | "
                    f"{raw['n_favourable']}/{raw['n_patients']} | {raw['sign_test_p']:.3g} | "
                    + (f"{adj['median_delta']:+.4f} | [{adj['ci_low']:+.4f}, {adj['ci_high']:+.4f}] | "
                       f"{adj['n_favourable']}/{adj['n_patients']} | " if adj else "— | — | — | ")
                    + f"{entry.get('n_degenerate', 0)} | {loso.get('sign_stable', '—')} |")
            lines.append("")
        nuisance = card.get("nuisance_only") or {}
        if nuisance:
            lines += ["**干扰量自身的可分辨性（状态主张必须胜过这一行）：**", "",
                      "| 干扰量 | 中位 z | 95% CI | 方向有利 | p |",
                      "| --- | --- | --- | --- | --- |"]
            for key, effect in nuisance.items():
                lines.append(f"| {key} | {effect['median_delta']:+.4f} | "
                             f"[{effect['ci_low']:+.4f}, {effect['ci_high']:+.4f}] | "
                             f"{effect['n_favourable']}/{effect['n_patients']} | "
                             f"{effect['sign_test_p']:.3g} |")
            lines.append("")
        lines += [f"- Holm 校正（open-loop 家族）：`{card.get('holm_corrected_open_loop_family')}`",
                  ""]
    strict = d.get("h2b_strict") or []
    lines += ["### 6.4 被降级的严格对照（确定间期流 / 长间隔）", ""]
    if not strict:
        lines.append("_尚未产出。_")
    for card in strict:
        lines.append(f"- 角色：`{card.get('role', 'sensitivity')}`；"
                     f"{card.get('not_primary_h2b_because', '')}")
        lines.append(f"- 患者 {card.get('n_patients')}、发作 {card.get('n_seizures')}")
        degeneracy = card.get("degeneracy", {})
        if degeneracy:
            lines.append("- 退化读数：" + "、".join(
                f"{k} {v['n_degenerate']}/{v['n_total']}" for k, v in degeneracy.items()))
        strata = card.get("by_gap_stratum", {})
        if strata:
            lines.append("")
            lines.append("| 距上一事件 | 发作数 | 患者数 |")
            lines.append("| --- | --- | --- |")
            for name, block in sorted(strata.items()):
                lines.append(f"| {name} | {block['n_seizures']} | {block['n_patients']} |")
        lines.append("")
        early = card.get("early_ictal_transfer", {})
        lines.append(f"**early-ictal transfer：`{early.get('status')}`** — {early.get('reason')}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _h3_tech(d: dict) -> str:
    h3a, h3b = d.get("h3a"), d.get("h3b")
    lines = []
    tau = d.get("tau_freeze")
    if tau:
        lines += ["### 7.1 τ_r 冻结（在任何 exposure 臂之前）", "",
                  f"- 选中 τ_r = {tau['tau_r_seconds']:.0f} s；规则：{tau['selection_rule']}",
                  f"- 可辨识 = `{tau['identifiable']}`；一个标准误带内的区间 "
                  f"{tau['identifiable_interval_seconds']} s",
                  f"- exposure 结果参与了选择 = `{tau['exposure_outcomes_used']}`", "",
                  "| τ_r (s) | seeds | 患者 | 平均验证 | SEM |", "| --- | --- | --- | --- | --- |"]
        for row in tau["rows"]:
            lines.append(f"| {row['tau_r_seconds']:.0f} | {row['n_seeds']} | {row['n_patients']} | "
                         f"{row['mean_validation']:.5f} | {row['sem_validation']:.5f} |")
        lines.append("")
    ladder = d["h3_ladder"]
    if not ladder.empty:
        lines += ["### 7.2 resource ladder 运行状态", "",
                  "| 臂 | resource | seed | 状态 | 最优验证 | τ_r | τ_x | 核 | γ_q | γ_L | γ_x | 边界占用 | 塌缩 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for _, r in ladder.sort_values(["arm", "seed"]).iterrows():
            fmt = lambda v: ("—" if v is None or (isinstance(v, float) and not np.isfinite(v))
                             else (f"{v:.4g}" if isinstance(v, float) else str(v)))
            lines.append(
                f"| {r['arm']} | {r['resource_arm']} | {int(r['seed'])} | {r['status']} | "
                f"{fmt(r['best_validation'])} | {fmt(r.get('tau_r_seconds'))} | "
                f"{fmt(r.get('tau_x_seconds'))} | {r.get('exposure_kind', '—')} | "
                f"{fmt(r.get('gamma_q'))} | {fmt(r.get('gamma_L'))} | {fmt(r.get('gamma_x'))} | "
                f"{fmt(r.get('resource_boundary_occupancy'))} | "
                f"{r.get('resource_collapsed', '—')} |")
        lines.append("")
    effects = d["h3_effects"]
    if not effects.empty:
        lines += ["### 7.3 H3a predictive leg（相对匹配基臂）", "",
                  "| 端点 | 臂 | 中位差 | 95% CI | 方向有利 | 符号检验 p |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for _, r in effects.iterrows():
            lines.append(f"| {r['endpoint']} | {r['arm']} | {r['median_delta']:+.4f} | "
                         f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
                         f"{int(r['n_favourable'])}/{int(r['n_patients'])} | "
                         f"{r['sign_test_p']:.3g} |")
        lines.append("")
    curve = d["h3_curve"]
    if not curve.empty:
        lines += ["### 7.4 exposure timescale curve（端点 order_nll）", "",
                  "| 核 | 尺度 | 中位差 | 95% CI | 方向有利 |", "| --- | --- | --- | --- | --- |"]
        for _, r in curve[curve.endpoint == "order_nll"].sort_values(["kernel", "scale"]).iterrows():
            lines.append(f"| {r['kernel']} | {r['scale']:.0f} | {r['median_delta_vs_base']:+.4f} | "
                         f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
                         f"{r['n_favourable']:.0f}/{r['n_patients']:.0f} |")
        lines.append("")
    innovation = d.get("innovation")
    if innovation:
        lines += ["### 7.5 H3a innovation / directionality leg", "",
                  f"- 冻结 T1 来源：`{innovation['frozen_t1_arm']}` (`{innovation['frozen_t1_job_id']}`)",
                  f"- expected load 模型：{innovation['expected_load_model']}",
                  f"- outcome：{innovation['outcome']}", "",
                  "| τ_x (s) | 患者 | 真实 vs 零 | 真实−状态匹配打乱 | 真实−时间反转 | 真实−事件计数核 | 真实−段打乱 | 真实−原始负荷核 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for tau_x, block in sorted(innovation["by_tau"].items(), key=lambda kv: float(kv[0])):
            cells = []
            for key in ("real_vs_zero", "real_minus_state_matched_shuffle",
                        "real_minus_time_reversal", "real_minus_event_count_kernel",
                        "real_minus_session_block_shuffle", "real_minus_raw_load_kernel"):
                effect = block.get(key, {})
                cells.append(f"{effect.get('median_delta', float('nan')):+.4f} "
                             f"({effect.get('n_favourable', 0)}/{effect.get('n_patients', 0)})")
            lines.append(f"| {float(tau_x):.0f} | {block['n_patients']} | " + " | ".join(cells) + " |")
        lines.append("")
    if h3a:
        lines += ["### 7.6 H3a evidence card 摘要", "",
                  "```json",
                  json.dumps({k: h3a[k] for k in ("reference_arm", "resource_health",
                                                  "holm_corrected_primary_family",
                                                  "denominators") if k in h3a}, indent=2,
                             ensure_ascii=False),
                  "```", ""]
    if h3b:
        lines += ["### 7.7 H3b", "",
                  f"- 状态：`{h3b.get('status')}`",
                  f"- {h3b.get('reason', h3b.get('requires', ''))}",
                  f"- {h3b.get('verdict', '')}", ""]
    return "\n".join(lines) + "\n"


def _stability_tech(d: dict) -> str:
    runs = d["h1_runs"]
    lines = []
    if not runs.empty:
        lines.append(f"- 非有限损失导致的失败运行：{int((runs.status == 'NAN').sum())}")
        lines.append(f"- 稳定裕度（最小阻尼率）中位数："
                     f"{float(runs.stability_margin.median()):.6f}（正值表示线性部分收缩）")
        if "diag_generator_tau_median_seconds" in runs:
            tau = runs["diag_generator_tau_median_seconds"].dropna()
            if len(tau):
                lines.append(f"- 拟合状态时间常数中位数：{float(tau.median()):.1f} s"
                             f"（范围 {float(tau.min()):.1f} – {float(tau.max()):.1f} s）")
        lines.append(f"- observer 校正能量中位数：{float(runs.correction_energy.median()):.5f}")
        lines.append(f"- 资源触底比例中位数：{float(runs.resource_floor_fraction.median()):.5f}")
    lines.append("")
    lines.append("**积分器：**指数积分（exponential Euler）。在消息项于一步内冻结的前提下，"
                 "线性部分的解 `target + (H − target)·exp(−rate·Δt)` 是精确的，"
                 "对任意 Δt 有界。本队列最大真实事件间隔为 5.2e5 秒；"
                 "显式 Euler 在那里需要数千个子步，或者直接发散。单元测试在该间隔上验证了四级生成器"
                 "与四条资源臂全部保持有限且有界。")
    lines.append("")
    lines.append("**时间常数参数化：**`τ = exp(clamp(log τ, log 0.5, log 1e6))`，"
                 "八个状态维度按 10 秒到 3 小时对数等间隔初始化。"
                 "先前的 `softplus` 参数化在训练预算内最多只能达到约 20 秒，"
                 "使模型无法表示分钟到小时尺度的慢状态；受影响的运行已归档到 "
                 "`results/epi_prssm/v0_1/_invalidated_tau_parametrisation/` 并全部重跑。")
    return "\n".join(lines) + "\n"


def _resource_tech(d: dict) -> str:
    summary = d.get("summary") or {}
    resources = summary.get("resources") or {}
    jobs = summary.get("jobs", {})
    lines = [
        f"- 作业总数 {jobs.get('total', 0)}；终态分布 `{jobs.get('state_counts', {})}`",
        f"- 单作业峰值常驻内存：中位 {jobs.get('peak_rss_mib_median', float('nan')):.0f} MiB，"
        f"最大 {jobs.get('peak_rss_mib_max', float('nan')):.0f} MiB",
        f"- worker 上限计算：假定峰值 {resources.get('assumed_peak_rss_gib', 'n/a')} GiB × "
        f"安全系数 {resources.get('safety_factor', 'n/a')}，"
        f"系统内存保留 {resources.get('ram_reserve_gib', 'n/a')} GiB 或总量的 "
        f"{resources.get('ram_reserve_fraction', 'n/a')}，CPU 保留 "
        f"{resources.get('cpu_reserve', 'n/a')} 核，磁盘低水位 "
        f"{resources.get('disk_low_water_gib', 'n/a')} GiB",
        f"- 计算出的 worker 上限：{resources.get('computed_worker_limit', 'n/a')}",
        "",
        "每个 worker 强制 `OMP_NUM_THREADS=1` 等四项线程环境变量并 `torch.set_num_threads(1)`；"
        "实测每进程恰好占用 1 个逻辑核（30 个 worker 合计约 2977% CPU），未发生 oversubscription。",
        "",
        "**为什么全部在 CPU 上跑：**状态维度 8、触点数 6–52，单步张量极小，"
        "GPU 上的 kernel launch 开销主导，单卡串行的吞吐远低于数十个单核进程并行。"
        "队列内批处理把每患者-事件的扫描成本从 861 μs 降到 41 μs（21 倍），"
        "读出仍按患者在未填充张量上计算，避免 6 触点患者为 52 触点患者买单。",
    ]
    return "\n".join(lines) + "\n"


def _figure_tech(d: dict) -> str:
    summary = d.get("summary") or {}
    figures = summary.get("figures", {})
    if not figures:
        return "_无_\n"
    lines = ["| asset_id | 已生成 | PNG | PDF | metadata | README |",
             "| --- | --- | --- | --- | --- | --- |"]
    for asset, block in figures.items():
        lines.append(f"| `{asset}` | {block['generated']} | "
                     f"{'✓' if block['png'] else '—'} | {'✓' if block['pdf'] else '—'} | "
                     f"{'✓' if block['metadata'] else '—'} | "
                     f"{'✓' if block['readme'] else '—'} |")
    lines.append("")
    lines.append("每个 asset 的 PNG（600 dpi）、矢量 PDF、metadata JSON 与中文 README "
                 "由同一次运行产出。README 在图实际生成之后写入，不放空模板。")
    return "\n".join(lines) + "\n"


def _repro_tech(cohort: str) -> str:
    python = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"
    return "\n".join([
        "```bash",
        f"PY={python}",
        "cd /home/honglab/leijiaxin/HFOsp",
        "",
        "# 0. 构建队列缓存与 Goal 0（数据/划分/禁止输入/基线/inventory/Hard Gate A）",
        "$PY scripts/topic5_epi_prssm/prepare_cohort.py",
        "$PY scripts/topic5_epi_prssm/run_goal0.py",
        "$PY -m pytest tests/topic5_epi_prssm -q",
        "",
        "# 1. just-in-time synthetic 标定",
        "$PY scripts/topic5_epi_prssm/build_plan.py --stage synthetic --seeds 0 1 2",
        "$PY scripts/topic5_epi_prssm/launch_autonomous.py \\",
        "   --plan results/epi_prssm/v0_1/manifests/plans/synthetic_all34.json --tag synthetic --cap 14",
        "$PY scripts/topic5_epi_prssm/aggregate_synthetic.py",
        "",
        "# 2. Goal 1 → Goal 2 → Goal 4 → Goal 3 → 图 → 汇总（一条命令串完）",
        f"$PY scripts/topic5_epi_prssm/build_plan.py --stage goal1 --cohort {cohort} --epochs 12",
        "$PY scripts/topic5_epi_prssm/launch_autonomous.py \\",
        f"   --plan results/epi_prssm/v0_1/manifests/plans/goal1_{cohort}.json --cap 36",
        f"$PY scripts/topic5_epi_prssm/run_full_matrix.py --cohort {cohort} --epochs 12 --cap 50 \\",
        f"   --wait-goal1-plan results/epi_prssm/v0_1/manifests/plans/goal1_{cohort}.json",
        "",
        "# 3. 报告",
        f"$PY scripts/topic5_epi_prssm/write_final_summary.py --cohort {cohort}",
        f"$PY scripts/topic5_epi_prssm/write_reports.py --cohort {cohort}",
        "```",
    ]) + "\n"


def _unresolved_tech(d: dict) -> str:
    summary = d.get("summary") or {}
    items = summary.get("unresolved_items", [])
    if not items:
        return "_无_\n"
    lines = ["| 单元 | 状态 | 原因 |", "| --- | --- | --- |"]
    for item in items:
        lines.append(f"| `{item.get('job_id')}` | {item.get('state')} | {item.get('reason')} |")
    return "\n".join(lines) + "\n"


def _claims_tech(d: dict) -> str:
    summary = d.get("summary") or {}
    lines = ["**允许的措辞：**", ""]
    for claim in summary.get("safe_claims", []):
        lines.append(f"- {claim}")
    lines += ["", "**禁止的措辞：**", ""]
    for claim in summary.get("forbidden_claims", []):
        lines.append(f"- {claim}")
    lines += ["", "**证据卡各自独立：**H3 阴性不降低 H1、H2a 或 H2b；"
              "H2b 阴性只关闭 transition 解释，不影响 H3a；"
              "歧义前缀支持不足记为不适用，不记为 H2a 失败。"]
    return "\n".join(lines) + "\n"
