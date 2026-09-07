# 第一轮 XY 搜索自动接续 Fig.5（2026-09-05）

用户授权：本轮搜索完成后，直接接上后续 Fig.5 分析。接续代码独立于正在运行的搜索；不修改其源文件、候选、执行合同和排名。

## 入口和产物

工作树：`/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix`。

- 控制器：`scripts/chain_topic4_xy_round1_fig5.py`
- 原生 SNN worker：`scripts/run_topic4_xy_fig5_worker.py`
- 判据与记录器：`src/topic4_xy_fig5_followup.py`
- 预先固定的方案：`config/topic4_xy_round1_fig5_followup.json`
- 图生成器：`scripts/paper_figures/plot_topic4_xy_fig5_followup.py`
- 输出：`results/topic4_sef_hfo/vth_dual_core_xy_research/fig5_followup/`
- 主仓库中的同名结果目录通过已有 symlink 可直接访问。

后台单元名：`codex-t4-xy-round1-fig5-followup-20260905.service`。它等待 `codex-t4-xy-relative-round1-20260905.service`；状态独立写在 `fig5_followup/status.json`，不会覆盖上游搜索状态。

```bash
systemctl --user status codex-t4-xy-round1-fig5-followup-20260905.service
journalctl --user -u codex-t4-xy-round1-fig5-followup-20260905.service -n 40 --no-pager
```

## 接续条件和模型身份

必须等全局、局部 refinement、selection、六种子 confirmation 及上游最终作图全部完成。只看到 `final_search_report.json` 尚不足以启动，因为该文件可能早于最终 completion 写出。

候选仅按 confirmation 之前的 `selection.J_round1` 选：取 whole-sheet/interior 中合格最低分，排除历史 control。它必须在预先送去确认的名单中，并且六种子齐全、指标可估计、每种子的方向得分有限、无早期 runaway。失败时记录“不合格”，不改用确认表现更好的第二名，也不回退 s39。

`development_substrate.json` 固定新候选的 XY、VTH 描述和映射、修复连接图的 SHA256、虚拟电极、确认结果与源代码。此处是允许下游分析的开发参数锁；`final_substrate_frozen=false`、`author_acceptance=false`。确认后的开发扫描不等于证明全局最优或作者认可最终底物。

底物仍为 VTH-only：学习得到的 EE/EtoI 系数均为零；全片 EE 原有各向异性连接、真实延迟、异质阈值和 OU 输入保持原有定义。采用新 XY 对应的实际 core A/B/surround 划分。没有继承旧 s39 + Joint=1.25 的 fold、逃逸阈值或 tonic 工作点。

## 自动执行顺序

1. **原生 Z/M 扫描**：18 个配置 × 3 组拓扑/动力学种子，共 54 条、每条上限 20 s。拓扑复用确认中的 2731–2733，动力学为新种子 2831–2833。12 点扫描 `I_th_EI × tau_z`，另有 eta_M、tau_M 的四个敏感性点，以及 slow-off、Z=1 两种对照。参数顺序在 selection 完成前已固定。
2. **独立噪声复验**：按预定配置顺序选第一个至少 2/3 种子合格的活动 Z/M 点。合格要求至少 2 s 准备期、准备期至少两个自终止群体活动片段，以及末尾 1 s tonic plateau。此单点及两种匹配对照在新动力学 2851–2853 上复验（9 条）；不会再按复验表现换参数。
3. **配对空间扰动**：活动点至少 2/3 复验轨迹合格，且三条 slow-off 复验保持有界反复事件，才进入。每条合格轨迹完整重放，保存 1 s 基线和 onset−200 ms 两个 checkpoint。先验证重放的原始速率和 LFP checksum 与母轨迹相同，再在 3×3 网格每点强制 16 个最近 E 神经元发一个脉冲；每次续跑 300 ms。sham 和每个 probe 恢复完全相同的电压、电流、延迟环、Z/M 和全部 RNG 状态。
4. **诊断图与表**：参数扫描/状态图、同一轨迹的虚拟接触点/速率/Z/M 图、空间响应差异图。输出 PNG/PDF/SVG、中文 `figures/README.md`、JSON 表、输入和图文件哈希。若转变不合格，仍保留参数扫描及其诊断图，并明确空间分析不可估计。

Z/M 方程直接继承当前 `MZSlowVars`，新增 subclass 只记录 core A/B/surround。参考值 `I_th=95.1985, tau_z=5000 ms, tau_M=500 ms, eta_M=0.00745159` 来自旧机制配置，但在新图上全部重新检验。降低 `I_th` 是改变局部抑制电流的耗竭触发条件；不能把它直接叫作患者的抑制强度。

## 判据与结论边界

Tonic runaway 的末尾 1 s：群体均值 ≥300 Hz、core A/B/surround 均 ≥250 Hz、群体前后半窗差绝对值 ≤5 Hz。onset 是进入相应高态连续片段的时间。20 s 内未达到判据是右删失；若引擎提前停止却未通过区域 tonic 判据，则记 unresolved，不计为未发作。潜伏期图显示共同 20 s 截断下的限制平均值，不能写成所有种子的平均发作时间。

准备期事件沿用当前 active-fraction detector；这是用于 Fig.5 的群体活动诊断，不等于训练目标中的因果家族或两种传播模板，也不用于重新选择 XY。第一次 200 ms 的初始化片段不计数，返回判据只使用 onset−200 ms 之前的数据。

空间主要读数是 300 ms 的 `probe−sham` 有符号额外脉冲数/E 神经元，以及首末 50 ms 响应。报告脉冲与自然放电的碰撞、实际脉冲半径、正负响应和全部 seed。跨状态配对同一 site，但只有同一状态内才使用完全相同的噪声续接；pre-onset 窗口可能横跨自然跃迁，因此必须减去对应 sham。

这些是新底物的开发分析，不是临床发作复现、患者机制鉴定或分叉类型判定。Z/M 是随轨迹变化的状态；tau 与 I_th 等是外部扫描参数。空间图的长轴趋势也不足以证明各向异性连接的因果作用，后续仍需专门的几何 null。

## 运行、恢复和资源约束

每 worker 一个数值线程，地址空间上限 25 GiB；系统留 40 GiB。准入不仅检查 `MemAvailable`，还为已启动但尚未达到上限的 worker 预留增长空间，最多 24 个，内存不足时等待而非强行启动一个。文件系统剩余不足 30 GiB 停止派新任务，已运行任务自然排空。地址空间上限触发的分配失败有日志并停止链路，不伪装为科学阴性。

重复启动同一控制器使用文件锁防止重复派发。已完成 worker 必须有一致的 job、handoff、runtime 和数组哈希才跳过。正常异常会先排空本控制器的子进程；未开启自动无限失败重试。若输入漂移或上游失败，独立状态文件记录原因。

结果输出保留在工作树，依赖当前路径。机器重启或手动停止后可用相同命令恢复；该 transient systemd 服务用于脱离对话继续运行，不保证跨系统重启自动恢复。

## 验证记录

单元与原生引擎测试覆盖：确认前选择与禁止重新选择、缺失/重复种子和非有限分数、机制漂移、区域 tonic 判据/右删失、瞬时高态后返回、速率单位、内存增长预留、记录器不改方程、OU + 延迟环 + Z/M checkpoint 与 sham/固定脉冲逐点一致、上游未完成时不启动、实际 local-first 配置路径和完成包的哈希核验。

实际 40,000 神经元、修复图 seed=2511 的 100 ms 工程短跑放 `fig5_engineering_qa/`。该目录仅检查 worker 能加载真实底物和输出完整数据，不是被选中的科学候选，也不参与筛选或最终参数扫描；短跑峰值内存不代表 20 s 长跑的峰值。
