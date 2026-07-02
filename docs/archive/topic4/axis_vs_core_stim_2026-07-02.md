# 挡轴 vs 打灶（固定电极预算）+ 自发难点图 — 归档

**日期**：2026-07-02
**分支**：`topic5-v2-phase1`（Topic 4 工作续在此分支，同 `20d4cd8`/`0766b0a`）
**状态**：两张 paper-grade 图已出、验收门通过（含一条诚实的压力测试不成立）。模型内单轨迹 + 小扫描的机制/效率示意，**非临床证明**。
**Spec / Plan**：`docs/superpowers/specs/2026-07-02-topic4-axis-vs-core-stim-difficulty-design.md` / `docs/superpowers/plans/2026-07-02-topic4-axis-vs-core-stim-difficulty.md`

## Abstract（第一性原理）

**要讲的一句临床话**：一块会发作的脑组织有"灶"（发作起始的源）和"轴"（活动往外传的必经窄通道）。我们用模型说明——**在固定电极预算下，把刺激挡在"轴"上，效果至少不弱于打在"灶"上**；但这有个前提：得**有一条真正的窄通道可堵**。

**为什么先要一张铺垫图（图 A）**：读者会问"那为什么不直接让一个自发的灶自己发作、就地做刺激？"我们让一块很容易兴奋的脑组织只靠背景噪声自己放电（不外部戳），看它会怎样。测三种底物：一个大而均匀的热核、一个小而均匀的热核、以及外部按节奏戳出来的两灶。**结果**：大核在 23 毫秒就**快速冲进失控、全片被招募**（点火时刻**有从核到边缘的展布**——是有梯度地快速铺开，不是"整片同时闪"）；小核从中心往外扩、前锋一路**铺满到整片**（也是一次事件），只是慢一点；只有**外部戳的两灶**能先冒出**3 次分开的小事件**（一个"抑制油箱"分 3 档一级级耗干）、磨到 757 毫秒才塌进失控。**揭示**：自发单灶（无论大核小核）都只有"快速冲进失控、全片被招募"这一种命运（一次事件铺满、停不下来，只是快慢不同），**出不了**"一串分开的小事件慢慢累积"那条路——所以刺激实验的事件串必须**外部戳**出来。根子是一个矛盾：要它自己点着就得"易燃到能自燃"，可一旦易燃到能自燃，点着后就铺满停不下来。

**主图（图 B）怎么测"轴 ≥ 灶"**：两种刺激用**同样多的 4 个电极**，比谁把失控推得更晚。**两灶+中段走廊**（两灶之间有唯一的窄通道）：灶通常比预算大盖不全，打端点只掐掉一个灶、剩下的照样磨油箱，把失控推后 **+414 毫秒**；把同样 4 个电极压在**中段走廊**上，两灶的事件都传不过去，推后 **+834 毫秒**——**挡轴 ≥ 打灶**。**单个中心核**（新跑的诚实检验）：打灶把失控从 42.8 毫秒推后 **+37 毫秒**，挡轴只推 **+8 毫秒**——**这里打灶反而更狠**，因为中心核往**四面八方**径向漏、没有单一咽喉，挡轴只堵住了那条轴的两个方向、其余方向照样漏。

**揭示了什么**：**"挡轴 ≥ 打灶"是有条件的**——成立在**有单一咽喉**的多源几何里，在**单个中心核**这种没有咽喉的几何里如实报告为**不成立**（这是预先说好的压力测试，不是失败）。临床含义：挡传播通道能省电极、甚至更管用，但**前提是那条通道确实是必经的窄口**。

（内部归档代号：M3A-v2.2 `SpatialSlowField` 的 `q_I`（跨事件抑制资源耗竭）/`g_K`（每事件疲劳）慢变量；`_build_stage4_patch`（单核自发）/`_build_subject1146`（E1146 两灶）；`intervention_vth_at_time` 的 V_th clamp 做刺激；Stage-4 v2 `classify_workpoint` 的 `one_shot_burst` 12/12 + 6/6；E1146 `fig_m3a_v2_2_qI_stim_site_compare` endpoint/middle=+413.8/+834.4）

## 跑了什么

- **底物 = canonical Stage-4 自发 runner** 值（`run_sef_hfo_snn_cm_spontaneous_readout.py:520-525`）：`g=3.6, AR=2.0, theta=45°, density=100, L=20, drive=0.6`。E→E 各向异性（AR=2 沿 45°）使扩散沿主轴拉长——这正是"轴"有意义的原因。
- **图 A 三行**：`big`（`_build_stage4_patch(core_radius=6)`，T=200）/`small`（`core_radius=3`，T=200）/`kick`（`_build_subject1146`，T=1000，跨过引用基线的 757 ms 失控）。
- **图 B 小核三臂**：`no_stim` / `core_stim`（打灶：核内 4 触点近端，留残余源）/ `axis_stim`（挡轴：下游 4 触点两侧对称），`T=600`，刺激窗 `[0, 300)`。E1146 那行**引用**已提交的 `fig_m3a_v2_2_qI_stim_site_compare`（不重跑）。

## 结果

### 图 A（regime 门，全部通过）
| 行 | n_events | runaway | 铺满 frac_ever | 判读 |
|----|----------|---------|----------------|------|
| `big` r=6 | 1 | 23.0 ms | 1.00 | 快速冲进 runaway、全片被招募（G-A1：1 事件 & <60 ms ✓） |
| `small` r=3 | 1 | 42.8 ms | 1.00 | 前锋铺满整片（G-A2：1 事件 & frac_ever>0.5 & 有 runaway ✓） |
| `kick` 两灶 | 3 | 757.5 ms | 0.22（单事件 contained） | 外部戳出 3 次分开事件（G-A3：≥3 ✓）→ 失控 |

- **frac_ever 说明**：`max_active_frac`（每 1 ms 窗活跃比例）因 `tau_ref_E=2 ms` 饱和在 ~0.5，不能测"铺满"；改用**累计发放过的 E 细胞比例** `frac_ever_fired`（big/small=1.0 铺满、kick 单事件=0.22 contained）。
- **kick 事件计数说明**：5 次戳在全场活跃比例里并成 3 个可分辨的鼓包（t≈190/457/695 ms，油箱 `q_I` 对应 3 级台阶）；因鼓包近基线（~3% 活跃），用敏感门（0.15×峰值，非标准 0.5×，后者会漏掉最小的早鼓包读成 2）。计数逻辑抽成 `src.topic4_axis_vs_core.count_events_pre_runaway` 并**有单测锁死**（train=3 vs record-peak=1、immediate-burst=1）。
- **同步性说明（回应 review）**：不写"整片同步爆/无梯度"——col1 的点火时刻**确有从核到边缘的展布**。改报两个量化指标（写入 `figure_metadata.json` + col1 标题）：`onset_spread_ms`（代表事件里第一次放电时刻的**范围**，越大越有梯度）与 `core_sync10`（核内 E 触点在首个核放电后 10 ms 内点亮比例，big/small）。judgment 只写"快速走向全片招募、出不了分离事件"，不写同步强主张。

### 图 B（固定 footprint=4）
| 情况 | 打灶 delay | 挡轴 delay | 判读 |
|------|-----------|-----------|------|
| 两灶+中段走廊（有咽喉，引用 E1146） | +413.8 ms | +834.4 ms | **挡轴 ≥ 打灶** |
| 单中心核 r=3（无咽喉，新跑） | **+37.1 ms** | **+8.1 ms** | **打灶 > 挡轴（单核无咽喉）** — 诚实压力测试不成立 |

小核 no_stim 基线 runaway=42.8 ms；打灶→79.9 ms（残余源仍点火，n_stim_E=2132）；挡轴→50.9 ms（径向漏出，n_stim_E=1875）。

- **诚实边界①（预防式 clamp，回应 review P1-3）**：小核三臂的刺激窗是 `[0, 300)`、`stim_on=0`——即**从头就开的"预防式"固定窗压制**，**不是"等事件传播起来之后再挡"的传播阻断**。对"单核无咽喉→挡轴输"这个**负结论**它偏保守、可以接受；但**不能读成"传播后阻断轴"**。要回答真正的传播阻断，需要**事件相对窗**（核点火后、前锋到达轴/边界前才开）——留作后续（见 off-axis surround plan）。
- **诚实边界②（E 细胞数不等，回应 review eng）**：打灶 clamp 2132 个 E 细胞、挡轴 1875 个（边界细胞归了打灶臂，见下"接触盘重叠处理"）。**公平单位是触点数（各 4 个），不是细胞数**；细胞数不等（挡轴更少）对"挡轴 ≥ 打灶"主张**偏保守**。两个 `n_stim_E` 都写进 `small_core_stim.json` 备查。

## 公平合同（验收门）
- **G-B1 公平**：两臂触点数相等 `core=axis=4=N`（代码 assert；不等则 raise）。✓
- **G-B2 灶盖不全**：`N=4 < n_source_contacts=5`（留 1 个残余源触点，图上灰点在核圈内）。✓
- **G-B3 parity**：刺激只经 parity-tested `intervention_vth_at_time` 改 V_th 比较、无额外 RNG；`stim_on=0` 时两臂共用同一随机流、只差 clamp。✓
- **G-B4 都有效**：`core=+37.1>0` 且 `axis=+8.1>0`（都做了事）。✓
- **G-B5 主张（报告，不强求为真）**：`axis ≥ core - 10`？两灶 ✓（+834≥+414）；单核 ✗（+8 < +37-10）→ 如实报告"打灶 > 挡轴"。这是科学结果（径向核无咽喉），非 bug。
- **接触盘重叠处理**：`r_stim=2.0` > 半 pitch(1.2)，核边界与最近下游 clamp 盘相交；给核臂优先（`axis_mask &= ~core_mask`），挡轴臂只 clamp 下游细胞——保证公平单位是**触点数**、对"挡轴 ≥ 打灶"主张保守。

## 声明范围（红线）
visual diagnostic；within-model 单轨迹 + 小扫描；**runaway / tonic 不是 ictal 事件**。"挡轴 ≥ 打灶"是模型内固定预算的**效率示意**——在多源/咽喉几何（E1146）里成立、在单中心核里如实检验为不成立。**禁**："证明发作机制"、"电刺激治发作"、"闭环/recovery"。

## Provenance / 复现
- 代码（本 plan）：`src/topic4_axis_vs_core.py`（几何/footprint/onset/delay 纯函数）、`scripts/run_stage4_axis_vs_core_stim.py`（小核三臂 runner）、`scripts/paper_figures/plot_fig_stage4_axis_vs_core_difficulty.py`（两图渲染 + `simulate_row`）。测试 `tests/test_topic4_axis_vs_core.py`（8 过）。
- 数据：图 B `results/topic4_sef_hfo/axis_vs_core/small_core_stim.json`；图 A 诊断 `.../fig_stage4_axis_vs_core_difficulty/figures/_figA_af_traces.npz`。
- 背景（为什么自发失败）：`docs/archive/topic4/stage4_v2_workpoint_2026-07-02.md` + `results/topic4_sef_hfo/stage4_v2_workpoint/{screen_fast,scan_small_core}.json`（12/12 + 6/6 `one_shot_burst`）。
- 引用基线：`results/paper-ready-figure/fig_m3a_v2_2_qI_stim_site_compare_epilepsiae_1146/`（no_stim 757.5、endpoint 1171.3=+413.8、middle 1591.9=+834.4，footprint=4）。
- 复现：`python scripts/run_stage4_axis_vs_core_stim.py` → JSON；`python scripts/paper_figures/plot_fig_stage4_axis_vs_core_difficulty.py --figure both`。
