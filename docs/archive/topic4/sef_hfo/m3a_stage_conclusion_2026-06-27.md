# M3A 慢变量阶段性结论（2026-06-27）

> Scope: 主树合并后的阶段性收口。证据来自 `topic4-m3a-a2` worktree 已执行的 A1/A1b/A1c/A2 screen 和
> `results/topic4_sef_hfo/m3a_slowvars/`。这是机制 screen，不是癫痫发作机制 validation。

## 一句话判断

M3A 不是从零开始。我们已经扫过单变量冻结慢状态、静态 local-global 状态地形、动态全局反馈、以及 Abbott-style
区域抑制资源 `q` + `g_K` 恢复项。当前最稳妥结论是：

> 同一 E->E 轴向 scaffold 上，慢变量确实能通过局部资源耗竭提高网络许可度，并把小的局部事件推成更大的轴向相干招募波；
> 但“大范围仍沿轴传播”只能算 expanded axial recruitment，不能直接叫发作。M3A 还没有达到稳健的“间期短事件 -> 离轴/全局发作样招募 -> 恢复”机制 PASS。

## 科学目标重述

M3A 要回答的不是单纯“能不能从短事件变成长事件”，而是三件事：

1. **为什么推**：活动越多，局部抑制资源越耗竭，局部许可度升高。
2. **怎么推**：慢状态把系统从 core-localized axial propagation 推向更大范围的 recruitment；但如果仍沿同一轴推进，只能叫轴向扩大。
3. **怎么回来**：需要恢复项，例如 `g_K` / sAHP 或阈值适应，把高许可度状态拉回低许可度状态。

对应相图坐标建议保留为模型坐标：

- `X = 局部化易激性优势`：局部 core 易激性相对全局背景易激性的优势。高 `X` 更像局部间期轴向事件；低 `X` 表示全局背景追上来。
- `Y = 有效 E/I 推动量`：例如 `rho = lgr / (q_core * q_global)`。`q` 耗竭时 `Y` 上升，网络更接近大范围招募。

这两个坐标不能直接写成生理量；它们是把 SNN 慢变量投到 rate / 相图的可解释读数。

## 已经跑过什么

### A1: 冻结单变量慢状态

在均匀衬底上扫 `z`、`e_GABA`、`phi`、`g_K`。结论是有界负面：均匀衬底没有干净间期基线，OFF 状态基本纯 R0，所以不能用它证明“冻结慢状态把间期推向发作”。次级观察是不同慢变量会给不同事件形状，但不是可写的状态迁移证据。

归档：`docs/archive/topic4/sef_hfo/m3a_quasistatic_slowvars_recap_2026-06-24.md`

### A1b: 静态 local-loop x global-restraint 状态地形

在 Stage-3 two-core 衬底上扫局部 loop 和全局反馈抑制。这个结果支持“局部化易激性优势 / 全局抑制约束”作为 M3A 坐标：弱全局 restraint 容易 runaway；强 restraint + 弱 local loop 静默；对角中段出现“大范围但仍可回静”的高招募状态。

口径必须收紧：A1b 的 `seizure_like` 是旧 screen 标签，不是科学结论。若源空间活动仍沿 E->E 长轴推进，这类状态应改称 `expanded axial recruitment` / `preictal-like axial expansion`，不能作为发作态相图成立的证据。

归档：`docs/archive/topic4/sef_hfo/m3a_a1b_state_topography_2026-06-25.md`

### A1c: 动态全局反馈

动态全局反馈能改变时序和终止，但空间均匀反馈本身不能干净解释 core-focused runaway 的终止。它支持后续 A2 走“局部使用依赖资源”的路线，而不是只靠一个全局刹车。

具体说，“要压住核，就会过压周边/弱态”指的是：全局反馈是一个统一刹车，强到能压住 core runaway 时，也会把 surround 或较弱活动态压成 silent/suppressed；弱到能保住周边/弱态时，又压不住强核心。A1c 因此是一个负面机制筛查，不是干净终止工具。

归档：`docs/archive/topic4/sef_hfo/m3a_a1c_pilot_recap_2026-06-25.md`

### A2: Abbott-style 区域抑制资源 `q`

只靠 `q` 耗竭的结果很清楚：要么 stay，要么 runaway，没有稳定的“升上去再回落”中间窗。这回答了“为什么推”：局部活动耗竭抑制资源，`q_core` 下降，`rho` 上升；但也暴露了问题：这个正反馈没有反向力，不能自己回来。

加入 `g_K` 后出现窄窗口慢-快脉冲候选：`q` 把系统推上高许可度，`g_K` 把它压回去。这回答了“怎么回来”的最低机制要求。但该窗口种子脆弱，且最初的传播 gate 经历了粗读数误判；最新源空间逐细胞 onset map 显示高许可度事件不是同步爆发，而是沿两核轴的相干招募波。这个 gate 因此重新打开，但还不能改写为 PASS。

所以 A2 当前不能写成“完全同步”或“已出现发作样全局招募”。它能写的是：局部资源耗竭 + 恢复项可以把小局部事件推成更大的轴向相干招募波，并可能回落；是否能破开轴向 scaffold 仍未证明。

归档：`docs/archive/topic4/sef_hfo/m3a_a2_abbott_lg_pilot_recap_2026-06-26.md`

### M3A-v2: 空间慢变量场 closed-loop screen（M3A-V2-1，2026-06-28）

A2 的两个**标量**油箱（`q_core`/`q_global`）没有空间历史——"轴向疲劳的同时周边许可度上升 → 破轴"这件事结构上承载不了。M3A-v2 把它们升级成**空间场** `q_I(x,t)`（抑制资源，宽核 σ_q）+ `g_K(x,t)`（疲劳/恢复，窄核 σ_K），实现到 green，然后做了一条四步 closed-loop screen。**结论：一致 NEGATIVE——载体（field 层）正，但当前 SNN 闭环触发不了它。**

按本 doc 的"三件事"对账：

- **为什么推（成立）**：field-only pilot 正。给一段持续 prescribed 活动，`σ_q>σ_K` 的场确实在地图层面造出"旁边追上主轴"的离轴易激性优势，剂量可控。**局部资源耗竭这个方向对。**
- **怎么推（未达成）**：把场接回 SNN 闭环，**没有一次推到"受控离轴 / 全局招募"**。
  - **Step 1 衬底鉴定**：先确认稳健局部沿轴自限事件存在（AR=4/nu=0.46，8/8 seed）——解开闭环死结。
  - **Step 2 q_I only**：事件本就 corridor-saturated（`axis_reach=1.0`），q_I 推大就直接 runaway，不是离轴。
  - **Step 3 q_I+g_K**：g_K 是刹车（疲劳正确压轴），只**终止 / 缩小**轴向事件、不重定向离轴；离轴 q 许可度始终没形成（`q_off` 从未 < 0.7）。唯一出现 off-axis（F=0.635）是在 runaway。
  - **Step 4 低-q（fork A，补"先把 q 耗低再 probe"）**：采样的 kq 网格里 q **没有稳定中间低-q 带（sharp transition）**——要么浅耗竭无效（q~0.9）、要么 crash（q ~0.015–0.18）；够 `q<0.7` 的只有 crash 态，probe 出 off-axis 但 runaway、刹不回。**small 0/24、finer 0/12、合计 0/36 success。**
- **怎么回来（方式不对）**：g_K **能**把活动压回来，但走的是 **suppress**（把轴向事件刹短 / 刹死），**不是**"完成发作样离轴招募之后再恢复"。

**M3A-v2 能写**：空间慢变量场是一个 sound 的 field-level 载体（sanity 正）；闭环不闭合是**当前 SNN 衬底（全或无 / 全场事件、采样网格内无稳定中间低-q 态）**的问题，**不是**"慢变量机制总体失败"。
**M3A-v2 不能写**：① "已破轴 / 已出现发作样离轴招募"（off-axis 只在 runaway）；② "证明存在 saddle / 双稳态结构"（Step 4 只是采样网格上的 sharp-transition 观察）；③ "g_K 把 runaway rescue 成发作样招募后恢复"（它只 suppress）；④ "慢变量机制被证伪"（field-only 仍正，且 full-state preload 等变体未测）。
**下一步（新方向、非当前 spec、待用户定，本轮不开始）**：`D_EE(x,t)` 削轴向 relay scaffold 优势让离轴能竞争（前提仍要先解决衬底给不出局部可部分填充事件）；或事件协议 / 衬底重做，让系统先产生更长、更局部、可部分填充的 preictal-like activity。

归档：`docs/archive/topic4/m3a_v2_{field_pilot,substrate_qualification,step2_qI,step3_qI_gK,step4_lowq}_2026-06-28.md`（各 archive 顶部带可复现命令 + JSON）；进度链见 `docs/topic4_m3_stage.md §6`。

## 当前能写什么

可以写：

- 数据侧：间期事件和发作共享患者特异传播 field / skeleton，间期传播分类可帮助估计患者特异传播轴。
- SNN 侧：带 E->E 各向异性 scaffold 的 E/I SNN 能在无外部 kick 下产生自发轴向传播模板。
- M3A 侧：慢变量提供一个状态迁移机制假说：局部资源耗竭提高有效许可度，恢复项提供回落力，使事件有可能从小局部轴向传播变成更大轴向相干招募波；这仍低于“发作样离轴/全局招募”。
- 相图侧：`局部化易激性优势` 和 `有效 E/I 推动量` 是合理模型坐标；`rho` 是诊断坐标，不是生理量。

## 现在不能写什么

不能写：

- “已经证明慢变量导致癫痫发作”。
- “A2 已经通过间期-发作两态验证”。
- “A1b 局部:全局易激性状态图已经证明发作态成立”。
- “大范围但仍沿轴的 recruitment wave 等于发作”。
- “网络连接骨架从各向异性变成各向同性”。更安全说法是：连接 scaffold 不变，慢变量改变主导活动模式和表观传播场。
- “只靠局部抑制资源耗竭就能完成发作与恢复”。当前 `q-only` 是 stay/runaway；回落需要第二个恢复过程。
- “双核 collision 是发作必要读数”。源空间复核提示单源有向全场招募波可能是本机制的关键读数，正确 instrument 应该是 source-space onset gradient。

## 下一步最小路线

1. 先冻结 phenotype gate：小局部轴向事件、大轴向招募波、离轴/全局发作样候选、runaway 四类必须分开。
2. 把 `source-space onset gradient`、axis score、perpendicular spread / isotropy、low-k/globality 作为 A2-P canonical readout，重跑/复核低许可度 vs 高许可度事件。
3. 在 `q + g_K` 候选窗做多 seed，明确小局部事件、大轴向招募波、runaway、quiet 四类边界。
4. 加 rate-/mass-matched control：同等放电量下，高许可度是否仍更广、更离轴/全局，而不是只更大或更轴向。
5. 分开两个 gate：Gate A 是可投到 M3B 相图的慢状态轨迹；Gate B 才是 seizure-like phenotype。Gate A 可以先过，不能自动升级成 Gate B。
6. 只有 A2 的 trajectory/export schema 过测试后，才把 slow-state trajectory 交给 M3B 做 frozen rate / eigenmode phase-map overlay。

## 合并范围

本次主树合并包括：

- A1/A1b/A1c/A2 archive docs；
- A2 runner/analyzer/plot scripts；
- `src/snn_engine/slow_vars.py` 的 `RegionalResource` / `g_K` slow path；
- M3A calibration/export/helper modules；
- M3A/A2 单元测试；
- ignored 结果目录 `results/topic4_sef_hfo/m3a_slowvars/`。

没有做整分支 `git merge topic4-m3a-a2`，因为该分支相对主树包含大量无关旧线差异和删除；本次只选择性并入 M3A-A2 证据链。
