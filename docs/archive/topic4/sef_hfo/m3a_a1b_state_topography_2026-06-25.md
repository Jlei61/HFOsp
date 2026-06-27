# M3A-A1b state topography — local-loop × global-restraint on the Stage-3 two-focus core (2026-06-25)

> Scope: A1b only. STATIC structural knobs (frozen), state-topography MAP — NOT a dynamic causal chain.
> Substrate = Stage-3 `twoend_equal` core (cm-spontaneous readout). Results
> `results/topic4_sef_hfo/m3a_slowvars/a1b_grid/` (status_a1b.json + figures/a1b_state_surface.png).

## 朴素话（测什么 / 怎么测 / 揭示什么）

**测什么**：Stage-3 衬底 = 两个易兴奋的"病灶核"埋在安静脑片里，背景噪声下各自自发点火（间期）。这次固定住
两个**结构**旋钮——**局部病灶 loop 强度**（核内抑制弱一点 + 核内 recurrent E 强一点 = 局部更容易点着）和
**全局反馈抑制强度**（把所有 E 细胞收到的抑制统一缩放）——看网络落在哪个状态：保护静默 / 间期样沿轴自限传播 /
发作样大范围同步招募 / 失控。只问"当前局部:全局的比例落在状态空间哪个区"，**不问慢变量怎么随时间动**（那是 A2）。

**怎么测**：3 档局部 loop × 4 档全局抑制 × 3 个网络 = 36 个组合（+ e_GABA=16 阳性对照），跑 8 秒长记录。每个组合
不只看事件多不多，还看一整套活动读数：全网平均放电率、核内 vs 核外放电率、有没有持续不回静（tonic 占空比）、
事件空间多大（r95）、两个核同不同步（回静事件里两核 30ms 内共点火的比例）、事件回不回静。用这些**一起**把四态分开。

**揭示什么（诚实）**：
1. **全局反馈抑制是主轴**：抑制最弱（global_ei_scale=0.7）时，**不管局部 loop 多强、哪个网络，全部失控**（9/9
   超时跑不完）；抑制中等（1.0）网络活跃；抑制最强（1.6）+ 局部弱 → **几乎静默**（全网放电率 0.06 Hz = 保护态）。
2. **局部 loop 在固定抑制下把网络往更兴奋推**：抑制 1.0 时，局部弱（l0）= 间期样（双向平衡、回静）；局部强（l2）
   = 失控（核内放电率飙到 ~413 Hz、几乎不回静 0.02）。
3. **发作样态（大范围同步但仍回静）出现在对角线中段**：l1_g1.0 和 l2_g1.3——两核同步度升到 0.36–0.40、事件大
   （r95 9–10 mm）、但**还回静**（return 0.89–0.96）。
4. **状态地形图沿对角线**：把"局部:全局比例"从小往大调，网络走 **静默 → 间期样 → 发作样 → 失控**——正是预期四态。
5. **结构旋钮 vs e_GABA（阳性对照）的机制差别**：e_GABA=16 把两核同步推得更狠（同步 0.72）但**冲过头到失控**
   （只有 12% 回静、核内 ~216 Hz）；结构性的局部:全局旋钮同步度低一些（~0.4）却**停在更干净的"发作样且仍回静"**态。
   两条路都能到发作样，机制不同（膜机制 vs 兴奋-抑制结构平衡）。

**边界**：这是**状态地形图，不是动态因果链**——旋钮是冻结定值；3 网络 = screen 级；`collision_rate_returned_sidecar`
= sidecar 里回静事件中两核 30ms 共点火的比例（同时报 ambiguous_rate / n_sidecar / n_total）；global=0.7 列全超时 =
按 runaway 记（不是测到的回静态）；四态阈值是按本网格标定的**描述性**分界（原始指标全在 status_a1b.json）；
`local_global_ratio = (core_ee_gain/core_ei_scale)/global_ei_scale` 是**模型地形坐标，不是真实生理量**。

（内部归档代号：M3A-A1b, twoend_equal core, core_ei_scale/core_ee_gain/global_ei_scale, local_global_ratio,
collision_rate_returned_sidecar, activity readouts global/core/surround_E_rate_mean_hz/tonic_fraction,
posctrl e_GABA16, build_connectivity_rot local_scale_EI/w_EE_gain_core, src/sef_hfo_a1b.py）

## 1. 网格与旋钮

- 局部 loop level → (core_ei_scale, core_ee_gain)：`0=(1.0,1.0) 1=(0.85,1.15) 2=(0.70,1.30)`
  （核内 I→E 权重 × core_ei_scale；核内 both-in-core E→E × core_ee_gain）。
- 全局 restraint：`global_ei_scale ∈ {0.7,1.0,1.3,1.6}`，缩放**所有** E target 的 GABA 输入；核内 E 额外 ×core_ei_scale
  （E-target GABA scale = global_ei_scale；core E-target = global_ei_scale·core_ei_scale）。
- 实现：复用 `build_connectivity_rot(local_scale_EI, w_EE_gain_core, core_mask_E)` —— **不改引擎**，默认 1.0 = bit-parity。
- 这是**静态**全局抑制；真动态全局反馈 `I_global(t)=feedback_gain·filtered_global_E_rate(t)` 是 A1c/A2 的 engine hook。

## 2. 状态面（status_a1b.json + figures/a1b_state_surface.png）

```
            g0.7        g1.0           g1.3           g1.6
local0:     runaway     interictal     interictal     silent
local1:     runaway     seizure_like   interictal     interictal
local2:     runaway     runaway        seizure_like   interictal
posctrl e_GABA16: runaway (collision 0.72 但 return 0.12 = 冲过发作样到失控)
```
关键单元活动读数（均值 over 3 seeds）：l0_g1.6 全网 0.06 Hz（静默）；l2_g1.0 核内 413 Hz / return 0.02（失控）；
l1_g1.0 同步 0.36 / r95 10.3 / return 0.96（发作样）；l2_g1.3 同步 0.40 / r95 9.2 / return 0.89（发作样）。

## 3. 对 Abbott / A2 的衔接

A1b 的全局反馈抑制是 Abbott（Liou et al. 2020）"global feedback inhibition + local:global ratio"的**静态版**。
A1b 给出靶点：**发作样态在 local:global 比例的对角中段（global≈1.0–1.3 + 局部 loop 适中）**，失控在低 restraint /
高局部 loop，静默在高 restraint / 低局部 loop。下一步：
- **A1c**：把全局抑制做成真动态反馈（随全网 E 率滤波）。
- **A2**：把 z / e_GABA 做成动态用进废退（Abbott 的 `τ_z ż=H(g_th−g_I)−z` / Cl⁻ 累积）——这才是 Abbott 的完整机制；
  A1b 只是地形图，告诉动态版该往哪个区演化。

关联：[[m3_slowvar_mechanism_handoff_2026-06-24]]、[[project_topic4_sef_hfo_m3a_stage3_core]]、
`docs/paper/abbott_model.md`（Liou,Schevon,Abbott 2020 eLife）。
