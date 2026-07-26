# FCXR-HEO 机制线最终验收（RC1 → Stage D → LC1 → HEO1/2/3）

日期：2026-07-26

分支：`codex/topic4-mz-fcxr-heo1`

状态：**ACCEPT AS BOUNDED MECHANISTIC INTERMEDIATE；NOT A SEIZURE-LIFECYCLE PASS**

## 0. 最终判决

这条线完成了一个重要但有限的机制推进：

> 原始 ZM 的 terminal runaway 被改造成了一个数值安全、振幅有界、可持续的高活动振荡分支；但该分支仍由单一主导空间模态承载，跨区域相位关系高度锁定，频谱窄带，且没有完成间期—发作—爆后—统计恢复的生命周期。

因此，本线最终采用以下名称：

**bounded coherent common-mode oscillatory branch**

中文：**有界、相干、公共模主导的振荡分支**。

若强调其稳定相位梯度，可补充为 **phase-locked traveling-wave-like oscillatory branch**。在正式计算 continuation / Floquet multiplier 前，不把它写成已证明的 limit cycle、Hopf 分支或稳定周期吸引子。

本验收取代 HEO1–H3.1b 归档中较宽松的“高能量发作候选”“接近真实发作态”“方向性空间机制已出现”等过渡表述。旧文保留作审计记录，但后续主文、spec 与 agent handoff 均以本文为准。

## 1. 与原始 ZM 相比，方程层面做对了什么

### 1.1 快系统不再用一条无界的 recurrent additive current

最终保留的 RC1 substrate 是：

- 外源 / feedforward AMPA 仍是 additive current，避免 full-conductance 对间期工作点的破坏；
- recurrent E→E 改为 reversal-aware conductance；
- recurrent conductance 单独经过平滑饱和
  `g_rec_eff = g_sat·tanh(g_rec_raw/g_sat)`；
- GABA 与已有慢变量通路保持 reversal-aware；
- 所有新增通路默认关闭时保持旧引擎 parity。

这个拆分解决了 full-conductance 初版“外源电导造成过活跃、recurrent 电导造成局部撞顶”的混淆。只饱和 recurrent 路后，seed1/seed3 的间期工作点和 `g_sat ±20%` 均通过，且不再依赖 hard cap。

### 1.2 合作门只改变 recurrent 中高活动增益，饱和仍负责有界

HEO1 在 recurrent 原始电导进入 `tanh` 前加入 Hill 型合作增益：

```text
H        = relu(u-u_c)^n / (K_c^n + relu(u-u_c)^n)
u_tilde  = u · (1 + A_c · H)
gErec    = g_sat · tanh(u_tilde / g_sat)
```

合作门在中高活动区提供再生正反馈；`tanh` 在大输入处限制振幅。这一组合确实把 Stage D 的“磨损越深、事件越密”连续斜坡推进成可持续的高活动振荡分支，同时保持 48/48 网格数值有界、无 clip / runaway。

它解决的是 **amplitude boundedness + persistent oscillation**，不是 lifecycle、真实波形或空间多模态。

### 1.3 逐细胞 adaptation `m_i(t)` 暴露了重要的异质性机制

HEO2/3 的动态适应不是纯粹的恒定负荷：

- 均值匹配的 static-K 对照仍保持同步窄带；
- 动态逐细胞 `m_i(t)` 能部分拓宽频谱并降低锁相；
- 把每个细胞的 `m_i(t)` 换成负荷匹配的群体均值 `m̄(t)` 后，宽带窗与去同步窗均降为 0。

因此可以验收的机制结论是：

> 当前有限的频谱拓宽依赖活动历史产生的逐细胞恢复状态差异，而不是一个群体平均的一阶慢变量。

这要求下一版继续保留 per-cell load / recovery state，不能用单一 global scalar 取代。

## 2. 这条线实际跑了什么

| 阶段 | 核心问题 | 结果 |
|---|---|---|
| RC1 / workpoint | recurrent-only conductance + smooth saturation 能否保住间期统计 | seed1/3 与 `g_sat ±20%` 通过，零撞顶 |
| Stage D frozen-Z | 抑制失效轴上是否出现独立 finite-high branch | 没有；同一自终止事件族随磨损连续加密，密端亚稳 |
| LC1 dynamic Z/X | 是否可自然进入并由 relay X 终止、再恢复 | X 可在两 seed 终止持续高活动；但 Z 不恢复、无统计闭环 |
| HEO1 cooperative gate | 是否得到有界持续高能量振荡 | 得到约 15.6 Hz 的高相干公共模振荡；严格门 0/48 |
| HEO1 real calibration | HEO1 的“持续 30–150 Hz 平台”是否对应 E1146 | 不对应；真实 E1146 是约 3 Hz 尖波，1–80 Hz 抬升，80–150 Hz 不抬 |
| HEO2 adaptation | 均匀逐细胞适应能否把 16 Hz 态改成真实宽带尖波 | 部分拓宽/降相干，但表现为爆发—静默交替，不能持住目标状态 |
| HEO2.1 de-conflation | 招募、宽带、相干是否被一个指标混在一起 | 已拆开；source/sensor 招募不是首要缺口，缺的是同窗持续与波形 |
| HEO3 joint-window/source-space | 四个目标能否在同一窗共存、是否只是共同投影 | 0/8 联合目标窗；约六成 E 细胞参与，非单核投影，但公共包络仍主导 |
| H3.1/H3.1b spatial `tau_m` | 外加恢复时间空间排列能否维持宽带去同步 | 原几何有 P0；纠正后空间摆放确实改变权衡，但所有 arm 联合窗仍为 0 |

## 3. 三张关键图的最终读法

### 3.1 `real_vs_model_band_dB.png`

路径：
`results/topic4_sef_hfo/mz_full_conductance_spatial_relay/high_energy_oscillatory_branch/figures/real_vs_model_band_dB.png`

- 真实 E1146：`1–4 / 4–8 / 8–13 / 13–30 / 30–80 Hz` 均抬升，低频最强；`80–150 Hz` 约为基线或略降。
- 模型高态：`13–30 Hz` 主峰，`30–80 Hz` 仅中度上升，低频明显受压。

所以模型不是“还差一点就像真实发作”，而是处在另一种波形族：**持续窄带相干振荡** 对 **间歇、尖锐、1–80 Hz 宽带发作**。

### 3.2 `stage0_joint_windows.png`

路径：
`results/topic4_sef_hfo/mz_full_conductance_spatial_relay/heo3/figures/stage0_joint_windows.png`

在修正 250 ms 频谱泄漏和错误同步指标后，招募、宽带、去同步、高能量四项从未在同一个 1 s 窗共存。动态适应产生的拓宽主要发生在仍相位锁定的活动窗；相干下降主要发生在活动崩塌时。

因此，不能把“整段平均每项都改善”写成“出现了目标发作态”。

### 3.3 `stage1b_geometry_fix.png`

路径：
`results/topic4_sef_hfo/mz_full_conductance_spatial_relay/heo3/figures/stage1b_geometry_fix.png`

条带平移后，源核整核快、汇核整核慢，空间摆放确实提高招募、去同步和高能量占比，并优于其 shuffled 对照。但：

- centred arm 的 broadband 并不优于 shuffled；
- swapped arm 的 broadband 更高但去同步 / 高能量更低；
- 所有 arm 的联合目标窗仍为 0；
- 两核原始率相关仍为 `+0.956`。

安全结论是 **空间摆放会改变各指标之间的权衡**，不是“source-fast / sink-slow 是唯一或稳定的方向性机制”。H3.1b 只有 seed1、单锚点、5 s，不能承载方向性主张。

## 4. 动力学上的最终解释

当前最简解释是：合作增益使一个主导复模态获得净增益，平滑饱和限制其振幅，网络进入一个窄带、相位关系稳定的高活动周期样分支。15/15 电极参与和约 168° 的相位跨度说明它不是零相位同步，也不是局限在单核；但跨区域高相干、两核同起同落和联合目标窗为 0 表明它仍是 **common-mode dominated**。

HEO1 的 48 格与高分支判读仍以 seed1 为主，未完成同一锚点的 seed3 吸引子身份确认；因此本验收是“可复现工程底座上的受控机制中间态”，不是跨 seed 的高分支 robustness claim。下一 spec 的 P0 必须补这一门。

这三个问题必须分开：

1. **生命周期问题**：怎样从间期统计邻域进入、有限持续、终止、爆后抑制，再回到原有不规则 IED 分布；
2. **波形问题**：怎样从 16 Hz 窄带波变成约 3 Hz 尖锐、1–80 Hz 宽带的 burst morphology；
3. **空间模态问题**：怎样让公共模不再压倒 transverse / core-differential / axial modes，同时保留广泛招募。

在正式 Jacobian-vector / Floquet 计算前，以上是结构性推断，不写成已计算出的本征值或分岔类型。

## 5. 允许与禁止的科学表述

### 5.1 允许

- recurrent conductance + smooth saturation 保住了间期工作点并消除了 hard-cap / runaway；
- cooperative recurrent gain 产生了数值安全、持续的高活动振荡分支；
- 该分支约 15.6 Hz、跨电极高度相干、存在稳定相位梯度；
- 逐细胞恢复状态差异承载了部分频谱拓宽；
- 正确摆放的空间恢复异质性会改变招募、相干和能量之间的权衡；
- 当前结果是下一步 lifecycle / waveform / spatial-mode 机制的受控中间底座。

### 5.2 禁止

- 已得到真实发作态、完整 seizure lifecycle、双稳态、Hopf 或稳定 limit cycle；
- “全平台 30–150 Hz”是 E1146 的真实目标；
- 当前 16 Hz 分支与 E1146 发作一致或只差小幅调参；
- source-fast / sink-slow 已被证明是唯一方向；
- 两核 `+0.956` 足以否定所有爆发内相位错开；
- 外加空间异质性已被全局否证；
- 继续通过合作增益、`tau_adp` 或条带位置的大网格可以补齐生命周期。

## 6. 线级验收与冻结决定

### 6.1 工程验收

- 最终验收复跑 FCXR workpoint、Stage D/LC1、HEO1–3 与 slow-variable suites：**216 passed**；
- 6 个 blessed engine 文件保持未改；
- off-by-default 与 byte-parity 合同保留；
- 长运行有 setsid/nohup、PID/sentinel、swap 与 worker 限制，未出现 OOM；
- H3.1 的几何 P0 和区域交替指标 bug 均有回归测试。

### 6.2 科学验收

**验收为 bounded mechanistic intermediate，不验收为 seizure model closure。**

本线到此冻结：

- 不再扫 `A_c / gate_quantile / g_sat / tau_adp / stripe position` 来追正结果；
- 不把 H3.1b 扩成大规模方向性网格；
- 不在同一 sprint 同时打开 X、M、pump 和新的 global brake。

下一阶段以
`docs/superpowers/specs/2026-07-26-topic4-mz-fcxr-pump-lifecycle-design.md`
为唯一设计入口：先形式化诊断当前公共模周期样分支与真实读出，再只加入逐细胞 activity / Na-like load → pump-equivalent recovery，`X=1`、旧 `M` 关闭，先解决 termination 与 postictal memory。

## 7. 证据索引

- HEO1：`docs/archive/topic4/sef_hfo/mz_fcxr_heo1_2026-07-24.md`
- HEO2/2.1：`docs/archive/topic4/sef_hfo/mz_fcxr_heo2_2026-07-24.md`
- HEO3 source/joint-window：`docs/archive/topic4/sef_hfo/mz_fcxr_heo3_stage0_2026-07-25.md`
- HEO3 H3.1/H3.1b：`docs/archive/topic4/sef_hfo/mz_fcxr_heo3_stage1_2026-07-25.md`
- Stage D：`docs/archive/topic4/sef_hfo/mz_fcxr_stage_d_branch_map_2026-07-22.md`
- LC1：`docs/archive/topic4/sef_hfo/mz_fcxr_lc1_bounded_negative_2026-07-23.md`
