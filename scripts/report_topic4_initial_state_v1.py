#!/usr/bin/env python3
"""Write figures/README.md (zh) and scientific_report.md for initial-state v1.

Reads only the analysis outputs; re-runnable. Figure text stays candidate /
pending human visual review.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import topic4_initial_state_runtime as rt  # noqa: E402

ARM_ZH = {"B0": "B0（共同基线：全体 V_reset）", "B1": "B1（core A 的 733 个 E 细胞 +1 mV）", "B2": "B2（core B 的 733 个 E 细胞 +1 mV）"}


def fmt(v, d=1):
    return "—" if v is None else f"{v:.{d}f}"


def pct(v):
    return "—" if v is None else f"{100 * v:.1f}%"


def stage_paragraph(name, e):
    if e is None:
        return f"- {name}：未执行。"
    pc = e["primary_contrast"]
    est = pc["n_estimable"]
    if pc["status"] != "ESTIMABLE":
        bounds = pc["full_design_mean_bounds_missing_at_plus_minus_one"]
        return (f"- {name}：{est}/{pc['n_design']} 对可估计，主对比 NOT_ESTIMABLE；已观测对的平均差 "
                f"{fmt(100 * pc['observed_mean_of_estimable_pairs'] if pc['observed_mean_of_estimable_pairs'] is not None else None)} 个百分点，"
                f"缺失对取 ±1 时全 12 对均值界限 [{fmt(100 * bounds[0])}, {fmt(100 * bounds[1])}] 个百分点。判定 {e['verdict']}。")
    eff = pc["effect"]
    inst = eff["instability"]
    flags = [k for k in ("single_pair_dominates", "leave_one_out_sign_flip", "bootstrap_exchange_disagree") if inst[k]]
    near_zero = abs(eff["mean"]) < 0.02
    note = ""
    if inst["leave_one_out_sign_flip"] and near_zero:
        note = "（去一对后均值变号只是均值≈0 时的必然现象，不改变'区间落在 ±10 点内'的判定）"
    return (f"- {name}：12/12 对可估计。B1−B2 的 12–24 s M0 比例差 Δ = {fmt(e['delta_percentage_points'])} 个百分点，"
            f"95% 配对 bootstrap 区间 [{fmt(e['ci95_percentage_points'][0])}, {fmt(e['ci95_percentage_points'][1])}]，"
            f"双侧配对交换 p = {e['exchange_p']:.4f}（4096 种符号分配）；预注册不稳标志：{('、'.join(flags) if flags else '无')}{note}；"
            f"最大单对贡献 {pct(inst['max_single_pair_contribution'])}。判定 {e['verdict']}。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, default=rt.DESIGN_PATH)
    args = parser.parse_args()
    design = rt.load_design(args.design)
    out = rt.output_root(design)
    final = rt.read(out / "primary_effect.json")
    quality = rt.read(out / "conditional_propagation.json")
    qual = rt.read(out / "qualification.json")
    frozen = rt.read(out / "frozen_manifest.json")
    status = rt.read(out / "status.json")
    f = final["final"]
    screen, replication = final["screen"], final.get("replication")
    stages = [s for s in ("screen", "replication") if final.get(s)]
    rows = []
    with open(out / "run_summary.csv") as stream:
        import csv
        rows = list(csv.DictReader(stream))
    n_runs = len(rows)
    n_runaway = sum(r["physical_status"] != "COMPLETE_NO_RUNAWAY_BY_EXISTING_GATE" for r in rows)
    wall = [float(r["wall_seconds"]) for r in rows if r.get("wall_seconds")]

    # ---------------- figures/README.md ----------------
    lines = ["# 初态条件传播 v1：诊断图说明（候选版，待用户人工目视审阅）", "",
             "所有图由 `scripts/analyze_topic4_initial_state_v1.py` 与 `scripts/render_topic4_initial_state_events_v1.py` 生成；",
             "这是比较诊断图，不套机制主图的四列布局。臂配色固定：B0 灰、B1（core A）橙、B2（core B）绿；模式配色 M0 蓝、M1 红。", ""]
    for s in stages:
        e = final[s]
        topo = e["topology_seed"]
        seed = min(design["stages"][s]["dynamics_seeds"])
        tag = "筛查图" if s == "screen" else "重复图"
        lines += [f"### fig1_late_window_pairs_{s}.png / .pdf",
                  f"{tag}（拓扑 {topo}）12 个噪声配对在 12–24 s 窗内的 M0 比例：左图每条细线连接同一噪声实现下 B1 与 B2 的比例，灰短横为 B0；"
                  f"右图是每对 B1−B2 的差值、全体均值及 95% 配对 bootstrap 区间，灰带是预注册的 ±10 个百分点优先级阈值。"
                  f"本图判定：{e['verdict']}；Δ = {fmt(e['delta_percentage_points'])} 个百分点，p = {fmt(e['exchange_p'], 4)}。",
                  "**关注点**：差值是否被单个配对拉动、区间是否跨 0 与是否落在 ±10 点内；所有 12 对是否都可估计。", "",
                  f"### fig2_window_proportions_{s}.png / .pdf",
                  f"{tag} 三臂在 0.5–6、6–12、12–18、18–24 与 0.5–24 s 五个窗口的逐运行 M0 比例（左）、可分类 primary 事件数（中）"
                  f"以及被排除窗口的原因分解与提前终止次数（右）。",
                  "**关注点**：早期窗口与晚期窗口的差是否一致；事件量是否随臂变化（比例差是否伴随低产）；排除原因是否在三臂间失衡。", "",
                  f"### fig3_conditional_propagation_{s}.png / .pdf",
                  f"{tag} 按患者冻结分类器的模式分层（上 M0、下 M1）看传播质量：逐触点参与率（患者 FIT 黑线 vs 三臂）、成对有符号中位时差"
                  f"（横轴患者、纵轴模型，越贴对角线越像）、以及 run 级条件 D_off（点=每次运行，粗线=均值与 run 级 bootstrap 区间；"
                  f"灰带=按运行事件量匹配的患者 FIT 块重采样 2.5–97.5% 区间，黑点=患者各记录块的自然 D_off）。",
                  "**关注点**：初态是否只改标签占比而不改善参与/时差结构；模型 D_off 是否落进患者匹配区间；负 D_off 是否来自小 N 减项。", "",
                  f"### fig4_core_state_timeline_{s}_seed{seed}.png / .pdf",
                  f"{tag} 最小 seed {seed} 的三臂全程（0–24 s）core A/B 平均膜电位（1 ms 记录、20 ms 平滑）与全体 E 放电率，"
                  f"竖线是每个检测窗口的时间（实线=primary，点线=排除；颜色=模式）。",
                  "**关注点**：两核状态是否在事件后回到同一水平、B1/B2 与 B0 的轨迹何时分离、后期事件标签是否与某核状态对应。", "",
                  f"### events_seed{seed}/<arm>/all_event_contact_maps.png / native_chronological_page_XX.png / all_chronological_events.gif",
                  f"最小 seed {seed} 三臂的全部检测窗口按时间顺序展示：触点图（相对质心 0–100 ms，空心=不参与）、原生场抽帧（首质心 −16…+64 ms）"
                  f"和逐窗口动画（原生场 + 触点读出 + 全程时间轴 + 核状态）。未按相似度挑选，也未跨臂对齐事件号。",
                  "**关注点**：同一噪声下不同初态的事件是否走不同路线；M 标签是否对应可见的传播差异；排除窗口的形态。", "",
                  "### native_frames/<run>/",
                  "其余 seed 每次运行的原生场抽帧页与逐事件表（event_table.csv）。",
                  "**关注点**：仅作逐运行核查导航；主张以统计文件为准。", ""]
    (out / "figures" / "README.md").write_text("\n".join(lines))

    # ---------------- scientific_report.md ----------------
    qs = quality["final"]
    fit = quality[stages[0]]["patient_FIT_reference"]
    rep = ["# 同一结构下的初态条件传播 v1：科学报告（候选版）", "",
           f"状态：`{status.get('status')}`；生成时间 {time.strftime('%Y-%m-%d %H:%M')}。工作树快照 `{frozen['git_commit'][:12]}`（{frozen['snapshot_worktree']}）。", "",
           "## 1. 先回答问题", "",
           f"- **初态效应**：`{f['initial_state_effect']}`。",
           f"- **患者传播结构是否改善**：`{f['patient_conditional_propagation']}`。", "",
           "## 2. 测了什么、怎么测的", "",
           "把同一张固定网络（连接、细胞位置、两个病灶核、逐细胞阈值、所有时间常数、外源随机输入规律全部不变）从三种起点各跑 24 秒：",
           f"{ARM_ZH['B0']}；{ARM_ZH['B1']}；{ARM_ZH['B2']}。同一噪声 seed 的三臂收到逐步完全相同的全局 OU、空间 OU 与 Poisson 输入"
           "（用每秒一段的流式摘要逐段核验，不是只凭同一个整数 seed）。",
           "然后用冻结的患者观察器（0.5 s burn-in 不变）检测事件，用冻结的患者分类器给每个 primary 事件贴 M0/M1 标签；",
           "主要读出只有一个：12–24 s 窗内 M0 占比的 B1−B2 差，以 12 个噪声配对为统计单位，等权平均，不池化事件。", "",
           "## 3. 资格检查（I0）", "",
           f"`qualification.json`：{qual['status']}，{sum(qual['checks'].values())}/{len(qual['checks'])} 项通过；"
           f"B1/B2 与 B0 轨迹分离时刻 {qual['divergence_ms']}；短程单元估算正式单元约 "
           f"{qual['resources']['projected_formal_unit_seconds'] / 60:.0f} 分钟、进程峰值 {fmt(qual['resources']['peak_rss_gib_this_process'])} GiB。", "",
           "## 4. 主要结果", ""]
    rep.append(stage_paragraph("I1 筛查（拓扑 2511，dynamics 820101–820112）", screen))
    rep.append(stage_paragraph("I2 重复（拓扑 6251，dynamics 820201–820212）", replication) if replication else
               f"- I2 重复：未触发（I1 判定 {screen['verdict']} 未达到持续效应门槛）。")
    for s in stages:
        e = final[s]
        rep.append("")
        rep.append(f"次要窗口（{s}）B1−B2 平均差（百分点，95% 区间；描述性）：")
        for key, w in e["secondary_windows"].items():
            c = w["B1-B2"]
            if c["effect"]:
                rep.append(f"  - {w['window_name']}: {fmt(100 * c['effect']['mean'])} [{fmt(100 * c['effect']['bootstrap']['ci'][0])}, {fmt(100 * c['effect']['bootstrap']['ci'][1])}]，{c['n_estimable']}/{c['n_design']} 对")
            else:
                rep.append(f"  - {w['window_name']}: 可估计对不足（{c['n_estimable']}/{c['n_design']}）")
        b = e["baseline_contrasts_late_window_holm"]
        rep.append("晚期窗口相对 B0（Holm 校正两项）：" + "；".join(
            f"{k}: {fmt(100 * v['effect']['mean']) if v['effect'] else '—'} 点，Holm p = {fmt(v.get('holm_adjusted_p'), 4)}" for k, v in b.items()))
    rep += ["", "## 5. 模式条件传播质量（独立结果层）", "",
            f"患者 FIT 参考：M0 {fit['M0']['n_events']} 事件、M1 {fit['M1']['n_events']} 事件；各块自然 D_off 中位数 M0 {fmt(fit['M0']['per_block_D_off_median'], 4)}、M1 {fmt(fit['M1']['per_block_D_off_median'], 4)}。", ""]
    for s in stages:
        rep.append(f"{s}：")
        for key, v in qs[s]["arm_mode"].items():
            band = v["patient_matched_band"]
            rep.append(f"  - {key}: 可估计运行 {v['runs_estimable']}，run 级 D_off 均值 {fmt(v['D_off_run_mean'], 4)}"
                       f"（区间 {v['D_off_run_ci95']}），患者匹配带 {None if band is None else [round(x, 4) for x in band]}，"
                       f"参与率 MAE {fmt(v['participation_mae_vs_FIT'], 3)}，成对时差残差 MAE {fmt(v['pair_signed_median_residual_mae_ms'])} ms，"
                       f"顺序 TV {fmt(v['order_TV_at_2ms_mean'], 3)}，支持/OOD 比例 {pct(v['supported_fraction_pooled'])}/{pct(v['unsupported_fraction_pooled'])}")
    rep += ["", "相对共同基线 B0 的描述性比较（不是检验）："]
    for s, rows in f["propagation_comparison_vs_B0"].items():
        for r in rows:
            rep.append(f"  - {s} {r['arm']} M{r['mode']}: D_off {fmt(r['D_off_run_mean'], 4)} vs B0 {fmt(r['B0_D_off_run_mean'], 4)}"
                       f"（run 级区间{'不' if r['D_off_run_level_intervals_separated_from_B0'] else ''}重叠），"
                       f"参与率{'更近' if r['participation_closer_than_B0_descriptive'] else '未更近'}，"
                       f"成对时差{'更近' if r['timing_closer_than_B0_descriptive'] else '未更近'}，"
                       f"落入患者匹配带 {r['inside_patient_band']}")
    rep += ["", "## 6. 完成数与工程状态", "",
            f"正式 24 s 运行 {n_runs} 次（设计 {36 * len(stages)}），提前终止（runaway 门）{n_runaway} 次，"
            f"单次墙钟中位 {fmt(sorted(wall)[len(wall) // 2] / 60 if wall else None)} 分钟。",
            "跨臂输入摘要与静态数组身份核查：" + "；".join(
                f"{s}: 摘要全等 {f['crosscheck'][s]['all_external_input_digests_equal_across_arms']}，"
                f"静态身份全等 {f['crosscheck'][s]['static_identity_equal_across_all_runs']}，"
                f"与主线历史 worker 一致 {f['crosscheck'][s]['static_identity_matches_historical_mainline_worker']}"
                for s in stages), "",
            "## 7. 限制与边界", ""]
    rep += [f"- {b}" for b in f["boundaries"]]
    rep += ["- 纯 V 初态只覆盖状态空间的一小部分；阴性只限制这个工作点上的 1 mV 冷启动探针。",
            "- 12 对是有限筛查预算，没有独立初态方差估计，不承诺检出 10 个百分点。",
            "- 所有图仍是候选版，需用户亲自目视检查后才算验收。", "",
            "## 8. 下一步（不在本轮执行）", "",
            "- 若初态效应持续且可重复：按方案第 9 节检验自主访问与状态记忆（自然静息期完整 checkpoint 延续、原位扰动/恢复）。",
            "- 若阴性或仅瞬态：隐状态作为新的结构假设（固定标量 s 调制阈值），必须与 s=0 及单峰慢漂移模型对照。",
            "- 若状态只改标签占比而不改善参与/时差结构：停止用状态选择解决传播质量缺口。"]
    (out / "scientific_report.md").write_text("\n".join(rep))
    print({"readme": str(out / "figures" / "README.md"), "report": str(out / "scientific_report.md")})


if __name__ == "__main__":
    main()
