# FCXR-HYB2：事件尺度、有界幅度的自主招募层

日期：2026-07-31 · 状态：**DESIGN LOCK CANDIDATE**（已过一轮审阅；六处订正见正文 ⚠️ 块）
Plan：`docs/superpowers/plans/2026-07-31-topic4-fcxr-hyb2.md`

> **spec 与 plan 冲突 = 执行 blocker。** 不设"谁优先"——发现冲突必须**先同步修订两份文件**再执行。
> （原先写的"以 plan 为准"会让两份合同悄悄分叉，已撤回。）

> 升 `DESIGN LOCK` 的条件 = plan 附录 A 的待回填项全部落定。
> 目前已锁两项：**`T_event,guard` = 22.0 ms**（LC1 24 s 冻结 bar，两 seed 事件时长**最大值**；
> 旧名 `T_event,90` 是错的——22.0 不是 q90）、
> `I_R,max = 4.134151260609386`（逐位复现）。其余待 §5.1 那条 calibration run。

前代与证据：

- `docs/archive/topic4/sef_hfo/fcxr_hyb1_baseline_disturbed_2026-07-31.md`
- `docs/archive/topic4/sef_hfo/fcxr_ion_B2_1_calibration_repair_2026-07-28.md`
- `docs/archive/topic4/sef_hfo/mz_fcxr_lc1_bounded_negative_2026-07-23.md`
- `docs/archive/topic4/sef_hfo/mz_fcxr_heo2_2026-07-24.md`（以 §7 HEO2.1 订正为准）

---

## 0. 核心科学目标不变

在固定的 E1146 40k RC1 substrate 上得到一次**无 kick、自发、有限、可恢复、具有空间结构的
seizure-like excursion**：

1. onset 前至少 8 s，系统位于能够按原有统计规律产生稀疏、不规则 IED 的邻域；
2. 自发进入 1–5 s 有界、高能量、振荡性高态；
3. 高态具有多电极招募与非同时的 source-space onset structure；
4. `X` 在 onset 后启动并终止高态；
5. 至少 8 s post 窗重新落回原来的间期统计邻域，而不是永久静默、快速复燃或固定周期轨道；
6. 生命周期成立后，才允许用固定的 `M` 配置改善 3–8 Hz、1–80 Hz broadband 与 active-window
   去同步。

本设计仍采用职责分离：

| 变量 | 唯一职责 |
|---|---|
| `Z` | onset susceptibility |
| `R_evt` | 单次高负荷事件内的空间招募与维持 |
| `X` | sustained-high termination |
| `M` | 生命周期成立后的 waveform shaping |

---

## 1. HYB1 到底否定了什么

HYB1 的 activity-excess K 层为

`excess source → τ_K=0.6546 s 的浓度积累 → K-reversal current`。

在真实间期工作点上，IED 每约 0.4–0.6 s 出现一次，与 `τ_K` 同量级。结果：

- sensor-only 层已经在间期跨事件累积；
- `P_{t,v}(δK>0.05 mM)=0.298–0.364`，远高于 0.01 门；
- feedback-on 后升至 0.632–0.698；
- 事件间地板在 8 s 内增加 4–6 倍；
- 两 seed 均改变 IEI CV，并改变部分 duration/participation。

因此 HYB1 否定的是：

> **用 0.65 s 浓度记忆承载招募、同时要求它对 0.4–0.6 s 间隔的 IED 不产生跨事件影响。**

它**没有**否定 B2.1 的 matched causal result：高活动下的动态 recruitment feedback 能在不提高
同一热点峰值的情况下增加参与细胞、RMS 半径与 occupied voxels。

---

## 2. 为什么不能“事件结束时脚本归零”

不得调用离线 `detect_events()`，也不得由 runner 在事件结束时把状态强制清零。那会：

- 把恢复变成外部脚本，而不是模型动力学；
- 用最终判决器反过来控制方程；
- 人工切断持续高态，制造伪生命周期；
- 使 onset/termination 因果归因失效。

HYB2 必须使用**因果、局部、自主的连续状态**。它可以在活动回落后快速自然衰减，但不能读取未来、
不能读取离线事件标签、不能 hard reset。

---

## 3. 新招募层：event-limited recruitment（ELR）

### 3.1 命名与解释边界

新状态记为 `R_evt`，名称为：

> **event-limited recruitment layer（事件尺度有界招募层）**

它是从 B2.1 动态 K recruitment effect 抽象出的**现象学执行器**，不是细胞外钾浓度，不使用 mM
作为状态变量，也不得称为 ion-homeostasis、potassium accumulation 或患者离子机制。

允许复用 B2.1 的膜电流**幅度锚**，但不得因此把 `R_evt` 重新解释成真实 `K_o`。

### 3.2 因果局部活动包络

沿用 32×32 体素与逐体素占据归一化。令 `s_v(t)` 为当前离子步内的 E+I spike-load density，
`b_v` 为独立 sensor-only calibration 窗的逐体素间期上包络。

先形成背景以上的局部活动：

$$
e_v(t)=R_\varepsilon(s_v(t)-b_v),
$$

其中 `R_ε` 必须：

- 在 `u≤0` 时严格为 0；
- 在 0 点至少 C¹；
- 参数由 calibration artifact 决定，不能从 lifecycle 结果反选。

> **⚠️ 审阅订正 P0-1：上面只给了 `R_ε` 的函数**类**，没有唯一确定方程。** `b_v` 的取法、`R_ε` 的
> 精确式、`ε` 的数值规则、spike-load 的聚合步长、`q_v` 的步长与积分方法，**每一个都会改变 `q_v`、
> `Q_on` 和 A0 的激活时刻**——它们是隐形自由参数。全部在 plan §2 写死，摘要：
>
> | 项 | 锁定 |
> |---|---|
> | 聚合步长 | `dt_R = 0.5 ms`（沿用已验收的离子步长）；`s_v` = 该块内体素 E+I spike 数 ÷ 体素细胞数 ÷ `dt_R` |
> | 空体素 | `n_v = 0` 是**采样空缺**：不产源、不进任何统计（沿用 B2.1 契约） |
> | `b_v` | `b_v = Q99_t[s_v]`，**只**用 seed1 calibration 前 12 s，仅占据体素 |
> | `R_ε` | `R_ε(u) = u²/(u+ε)`（`u>0`），否则**严格** 0（沿用 HYB1 已测实现） |
> | `ε_s` | `:= 0.1 · median_v(b_v)`（沿用 HYB1 规则） |
> | `ε_q` | `:= 0.1 · Q_on`（同一 10% 约定，零自由度） |
> | 积分 | **精确指数更新** `q ← q·e^{−dt_R/τ_R} + e·(1−e^{−dt_R/τ_R})`（对分块常值 `e` 精确、无条件稳定） |
> | 两遍离线 | pass1 由记录的 load 算 `b_v`；pass2 **重放同一条记录**算 `q_v` → `Q_on`。**全程离线，无新仿真** |
> | 硬门 | `Q_on > 0` 且 `Q_scale > 0`，否则判 `CALIBRATION_INVALID` |

再由一个事件尺度 leaky state 积分：

$$
\tau_R\dot q_v=-q_v+e_v,\qquad q_v(0)=0.
$$

`q_v` 是**当前局部事件包络**，不是跨事件浓度。它的记忆时间必须同时满足：

$$
T_{\mathrm{event,guard}}<\tau_R
\quad\text{且}\quad
\exp(-GAP_{05}/\tau_R)\le 0.01,
\qquad GAP_k := t_{\mathrm{on},k+1}-t_{\mathrm{off},k}.
$$

> **⚠️ 审阅订正 P0-2（两处）。**
> **(a) 分母必须是事件间空白 `GAP`，不是 onset-to-onset 的 `IEI`。** 事件**结束之前** `q_v` 一直
> 在被驱动，真正能用来衰减的只有 `t_off,k` 到 `t_on,k+1` 这段。用 `IEI` 等于多算了约一个事件时长，
> **系统性低估残留**。
> 保守界（有效）：`GAP_k = IEI_k − dur_k` 逐点成立且 `dur_k ≤ T_max`，所以
> `GAP_k ≥ IEI_k − T_max` 逐点成立 ⟹ **分位数在这个平移下保持** ⟹ `GAP_05 ≥ IEI_05 − T_max`。
> 代入预审值：`GAP_05 ≥ 169.5 − 22.0 = 147.5 ms` ⟹ 上界从 **36.81 → 32.03 ms**，
> 区间 `[22.0, 32.03]`（仍非空），几何中点 **26.55 ms**。
> **余量只剩 10 ms**：`T_event,guard` 只要涨到 32.0 ms 区间就闭合。
> **(b) 命名订正**：`22.0 ms` 是**两 seed 的事件时长最大值**，不是 q90。改名 **`T_event,guard`**。
> （per-seed q90 分别是 19.7 / 14.0 ms，pooled q90 15.0 ms——都不是这个数。）

解释：

- `τ_R` 必须长到能看见一个完整的较长间期事件；
- 又必须短到在最短一档常见 IEI 内残留 <1%。

> **⚠️ 审阅补充：这条规则只覆盖 95% 的间隔，剩下 5% 的残留必须一并报告，不得藏起来。**
> **（按 P0-2 换成 GAP 后重算）** 取几何中点 `τ_R = 26.55 ms` 时：`GAP` = 147.5 ms 残留 **0.0039**、
> 100 ms 残留 **0.0231**、60 ms 残留 **0.1043**、40 ms 残留 **0.2216**。
> plan 必须把 `GAP_01` 与**实测最短 `GAP`** 处的残留一并写进 manifest，让最短那一段尾巴可见。

`T_event,guard` 与 `GAP_05` 只从**既有、未开 ELR 的 accepted baseline artifacts**估计。

> **⚠️ 审阅订正：必须指名唯一来源，并用冻结的 event bar。** HYB1 自己那条 8 s 基线在 seed3 上
> 给出 duration 中位 **39.0 ms**，而 LC1 的 24 s 基线在同一配置下给出 **11.0 ms**——差 3.5 倍，
> 因为短窗会**重新推导** event bar。**锁 LC1 的 24 s `baseline_contract_seed{1,3}.json`
> （冻结 bar）作为唯一来源**，不得用 HYB1 的 8 s 数字。
>
> 已核（零仿真）：seed1 q90 = **19.7 ms**、seed3 q90 = **14.0 ms**、两 seed 最大值 = **22.0 ms**、
> pooled q90 = **15.0 ms**。取**最保守**的 `T_event,guard = 22.0 ms`，并按 P0-2 用 **GAP** 作分母，
> 区间是 **`[22.0, 32.03]`（非空）**，几何中点 **26.55 ms**——**不是**旧稿的 `[22.0, 36.8]`。
> 所以 `DESIGN_BLOCKED_EVENT_TIMESCALE` 目前不会触发，但**余量只剩 10 ms**，
> 且 `GAP_05` 的下界依赖的 `IEI_05 = 169.5 ms` 仍是下采样近似，必须从 canonical onsets/offsets 重算。
若上述可行区间为空，判 `DESIGN_BLOCKED_EVENT_TIMESCALE`，不得调 baseline 或跳到 40k lifecycle。

`τ_R` 的唯一选择规则必须在 plan 中预注册。建议取可行区间的几何中点；不得扫描 `τ_R` 找好结果。

**零仿真可行性预审（非最终锁值）**：现有 LC1 baseline contract 给出 pooled
pooled 事件时长 q90 = 15 ms；用已保存的 6 ms rate trace 按原 event bar 重建得到近似
`IEI_05 = 169.5 ms`。

> **⚠️ 这段预审已被两处订正取代（见上文 P0-2）**：
> (a) 时长应取**两 seed 最大值 `T_event,guard = 22.0 ms`**，不是 pooled q90 的 15 ms；
> (b) 分母应是 **event gap** 而非 IEI，保守界 `GAP_05 ≥ 169.5 − 22.0 = 147.5 ms`。
> 修正后的可行区间是 **`[22.0, 32.03] ms`，几何中点 `26.55 ms`**（不是 `15–36.8`／`23.5`）。
> 这说明新结构不是空想，但**余量只剩 10 ms**；最终 plan 仍须从 canonical event onsets **与
> offsets** 重算，不得把下采样近似冒充锁值。

### 3.3 间期不触发阈值

ELR 只在 `q_v` 超过间期事件自身的上界后启动：

$$
u_v=R_{\varepsilon_q}(q_v-Q_{\mathrm{on}}).
$$

`Q_on` 的锁定必须使用 calibration/validation 分离：

1. 用 sensor-only baseline 的 calibration half 估计 `q_v` 的逐事件/逐体素峰值分布；
2. 按预注册规则锁定 `Q_on`（建议 calibration maximum 加固定 10% margin；样本充分时可用更高
   分位，但规则必须先写死）；
3. 用未参与定阈值的 validation half 和另一 seed 检查 baseline false activation；
4. 后续不得因 lifecycle 结果降低 `Q_on`。

HYB1 只保存了 `b_v/load_mean`，没有保存完整逐体素 load time series。因而允许在 selection rule
先锁死后，新增**一条无反馈、无 Z/X/M、无 kick 的 sensor-only calibration run**来落盘所需流式
峰值统计；它不是 actuator/lifecycle run，不得据其结果修改 selection rule。

这不是“把所有 IED 都增强”；它只允许**Z 已经把一次事件推到超出 accepted interictal envelope**
之后，ELR 才介入招募。

### 3.4 有界幅度与膜执行器

输出必须平滑、有界：

$$
R_{\mathrm{evt},v}
= I_{R,\max}\,
\tanh\!\left(\frac{u_v}{Q_{\mathrm{scale}}}\right).
$$

- `R_evt=0`：所有 accepted interictal load below `Q_on`；
- `R_evt≤I_R,max`：任何持续活动都不能让执行器无限累积；
- 活动下降后 `q_v` 按 `τ_R` 自主衰减，`R_evt` 随之关闭；
- 不需要事件标签或 reset；
- sustained ictal activity 可维持一个**有界平台**，而不是每个 spike 继续往上垫浓度。

`I_R,max` 不得自由扫参。首轮锚定为 B2.1 matched closed arm 的 hotspot `δK=0.6715 mM`
所对应的 `ΔE_K` 膜驱动幅度（`g=1`）；这是**force anchor**，不是浓度解释。
按当前已验收 `E_K` 实现解析换算为 **`I_R,max=4.134151260609386` engine-drive units**；
plan 只需核 sha/单位链，不得重新拟合。

> **⚠️ 审阅订正：`Q_scale` 不可能从间期 calibration 得到。** 按 §3.3，`Q_on` 被锁成 calibration
> **最大值 + 10%**，因此在整个 calibration 数据上 `q_v − Q_on ≤ 0` 恒成立——**没有任何一个样本
> 落在 `Q_on` 之上**，也就无从"解析出 `q_v−Q_on` 的尺度"。
>
> plan 必须改从**唯一且不在间期数据里**的来源取，二选一并先写死：
> - **(a) 解析规则（首选）**：`Q_scale := Q_on`。即执行器在超出间期上界"一倍"时达到
>   `tanh(1)=0.762·I_R,max`。这是一条无自由度的规则，只依赖已锁的 `Q_on`；
> - (b) 从 §4.2 Gate A0 的 supra-interictal artifact 取 `q_v−Q_on` 的中位数——但那条 artifact
>   **同时**是 A0 的被测输入，用它定标会让 A0 的 ≥10% 判据部分自证，**因此不采用**。
>
> 锁 (a)。`Q_scale` 与 `I_R,max` 都不得扫。

E 与 I 都接收同一附加 current；virtual-SEEG 仍为 E-only readout。六个 blessed engine 文件
不得修改。

### 3.5 空间结构

首轮不增加新的空间扩散参数。`R_evt` 按逐体素活动局部产生，空间扩张只能来自：

1. 固定 RC1 recurrent scaffold；
2. 邻近体素被网络活动因果招募后各自启动 ELR。

这样可检验“执行器是否借原有网络扩大招募”，避免用一个新扩散系数直接画出传播。

若没有招募扩张，不得通过扫描 diffusion、connectivity 或 geometry 抢救；另行审阅。

> **⚠️ 审阅补充：必须写明 A0 到底在测什么。** 去掉扩散项之后，`R_evt` 在一个体素里只取决于
> **该体素自己**的负荷，所以执行器**只能放大已经在活动的组织，不能点亮一个安静的体素**。
> 招募范围仍可能扩大，但因果路径变成"被放大的体素经 recurrent scaffold 驱动邻居"——
> **一步突触，而不是一个扩散场**。
>
> 这同时是 A0 的**主要风险**：B2.1 那 +24% 半径是在 `τ_K = 654 ms` 的记忆 **加** `D_K` 扩散
> **加** 200 ms 窗内累积下测到的；HYB2 把记忆缩短约 28 倍、把扩散删掉。
> **A0 很可能失败，而那会是一个干净的结果**：它把"B2.1 的空间招募是否依赖跨事件浓度记忆"
> 这个问题回答成"是"。**不得为过 A0 恢复扩散项或加长 `τ_R`。**

---

## 4. 两道前置门

### 4.1 Gate B0：baseline invisibility

在任何 Z/X/lifecycle screen 前，ELR off/on、无 kick、两个 baseline seed，必须同时满足：

1. validation 数据中的 `R_evt` active occupancy ≤1%；
2. **下一事件 onset 前 2 ms 窗**内 `q_v` 的 q99（跨事件×占据体素）≤ `0.01·Q_on`；
3. 无单调抬升：`(q_floor,末段 − q_floor,首段)/Q_on ≤ 0.01`（**差值**，不是比值）；
4. returning event rate、IEI CV、duration、participation、silent fraction 落在原
   baseline contract；
5. zero clip、finite、`tau_eff_min≥2dt`。

失败即 `STOP_ELR_BASELINE_VISIBLE`。禁止调 drive、连接、`Q_on`、`τ_R` 或 `I_R,max` 抢救。

> **⚠️ 审阅订正 P0-3（三处）。**
> **(a) 原第 2 条按定义必败。** "整个 gap 内 `q_v` 的 q99"把**事件刚结束那一刻**也算进去，
> 那时 `q_v` 本来就高——这条几乎不可能通过，而且它测的不是"残留"而是"衰减起点"。
> 改为**下一事件 onset 前 2 ms 窗**（用小窗而不是单个采样点，压采样噪声）。
> **(b) 原第 3 条的比值在首段接近零时数值不稳**（0/0 或除以极小数）。改为**用 `Q_on` 归一化的差值**。
> **(c) 我上一轮写的"第 3 条按构造通过"是错的**：`R_evt ≡ 0` 确实是构造保证（第 1 条），
> 但 **`q_v` 不发生棘轮完全不是构造保证**——那正是 HYB1 栽的地方。第 3 条是 B0 的**主判据**。
>
> **仍然成立的部分**：`Q_on` = calibration 最大值 × 1.1 ⟹ **第 1 条**在 calibration 半段上
> 按构造必然通过。
> 真正有信息量的只有 **validation 半段 + 另一个 seed** 上的第 1、2 条，以及第 4 条
> （事件统计不变）——后者仍可能因执行器在 validation 上偶发启动而失败。
>
> 因此：**B0 是必要条件，不是成功指标**；报告时不得写成"事件尺度执行器已被证明不打扰基线"，
> 只能写"在 validation 半段与第二个 seed 上未观察到基线扰动"。**风险全部集中在 A0。**
>
> 另：第 3 条"pre-event floor 比值 ≤1.10"**必须定义在 `q_v` 上，不是 `R_evt` 上**（`R_evt` 在
> 间期恒为 0，比值无定义）。plan 需写死这一点。

### 4.2 Gate A0：招募执行器仍然有效

baseline invisibility 通过后，才运行一个 matched actuator test：

- 同一 substrate、seed、初值与 supra-interictal high-load input；
- ELR off/on 两臂在 ELR 第一次启动前逐比特一致；
- 输入来自既有 q50/Stage-D high-load artifact 或预注册的固定 causal replay；
- 该测试允许 kick/replay，因为它只验证 actuator，不承担最终 spontaneous claim。

最低通过条件：

1. ELR on 确实越过 `Q_on`，off 不施加执行器；
2. 参与细胞、RMS radius、occupied voxels 中至少两个相对 off 增加 ≥10%；
3. 局部执行器幅度不超过 `I_R,max`；
4. zero clip、finite；
5. 不得把该结果写成 propagation；未测 onset delay/gradient/contact order 时只能写
   recruitment extent。

失败即 `NO_GO_EVENT_LIMITED_ACTUATOR`。禁止为了得到正结果扫 `I_R,max`。

> **⚠️ 审阅订正 P0-4：现在这版分不开"执行器无效"和"输入无效"。**
> 原文断言 q50"一定"越过新的 `Q_on`——**但那个阈值此刻还不存在**，不能提前保证。
> LC1 实测 q50 约 3 s 才进 dense、约 6 s 才进 ictal 窗、末段升到约 452.8 Hz，所以固定 6 s 有
> **两个方向相反**的风险：要么还没攒够"激活后 1 s"窗口，要么 off 臂已贴近全局招募天花板、
> ELR 根本没有 ≥10% 的余量可加。
>
> **改为三分判决（先写死）**：
>
> | 标签 | 条件 |
> |---|---|
> | `A0_INPUT_INSUFFICIENT` | `max_v q_v` 全程未越 `Q_on`，或首次越阈后剩余 < 1000 ms |
> | `A0_CEILING_CONFOUNDED` | off 臂在该 1 s 窗内 participants ≥ 90% E 细胞、或 occupied voxels ≥ 90% 占据体素、或触发 300 Hz early-stop |
> | `A0_ELIGIBLE` | 以上都不成立 |
>
> **只有 `A0_ELIGIBLE` 之后仍达不到 ≥10%，才允许判 `NO_GO_EVENT_LIMITED_ACTUATOR`。**
>
> **`t_gate` 必须由两臂共同追踪的 counterfactual sensor 定义**：两臂都完整跑 `q_v`（同一代码路径），
> **off 臂只把膜电流置零，不停掉 sensor 状态**。`t_gate := 首次 max_v q_v > Q_on`，
> 两臂应**逐位相同**——这既是 `t_gate` 的定义，也是结构性匹配的 sanity（沿用 B2.1 §2.3 的形态）。

---

## 5. Z 轴也要随本轮一起修

HYB1 的 `h_Z=dD_Z/dt|0` 在数学上成立，但执行结果显示：

- 正确统计量下 q75↔q50 初始 hazard 只差 1.64×；
- `H_LO` 实际落在 q75 边界；
- q75/q50 真正分开的重要部分是随后耗竭是否自限，而不是单个 `t=0` 导数。

> **⚠️ 审阅订正（2026-07-31，在写 plan 前）：这一段的理由站不住，但改法仍然可取。**
>
> "自限" = `z↓ → GABA 电导被缩小 → 细胞受到的 I_I 下降 → 落回阈下 → z_inf 翻 1 → 恢复"。
> 那是一条**闭环**通路。**在冻结的 slow-off 负荷轨迹上离线 replay，`I_I(t)` 按定义不受 `z` 影响，
> 所以 replay 结构上不可能表现出自限**——它复现的只是开环耗竭。
>
> 更强的算术：如果一个细胞在窗口内的"阈上/阈下"状态**不变**，那么所有阈上细胞走同一条
> `z(t)=exp(−t/τ)`，于是 `S_Z(I_th) = a_p(I_th)·C(T_cal, τ)` —— **和 `h_Z` 严格成正比，不含新信息**。
> `C(T)` 实测：`T_cal` = 3 / 5 / 15 / 24 s → 0.248 / 0.368 / 0.683 / 0.793。
>
> **`S_Z` 真正多出来的东西只有一样**：那些在窗口内**跨越**阈值的细胞——即它衡量的是
> **阈上时间占比**而不是某一瞬间的阈上比例。这确实比 `h_Z` 稳健（不受单一时刻抖动支配），
> 也确实把 1.64 倍窄区间上的三点分得更开，**但它不是"整段耗竭响应"，更不是自限的度量**。
>
> 因此：**保留 `S_Z`，但把它的名字与说法改为"开环累积耗竭坐标（阈上时间占比加权）"**，
> 并写死两条本来缺失的要求：
> - **`T_cal ≥ 3·τ_Z_down = 15 s`**。低于这个长度 `C(T)` 还在近线性段，`S_Z` 退化成 `h_Z` 的常数倍，
>   换轴等于什么也没换（`T_cal=3 s` 时 `C=0.248`，几乎纯线性）。
> - replay 需要**逐细胞** `I_I(t)` 时间序列。**现有 artifact 里没有**（HYB1 的 zaxis probe 只落盘了
>   生存曲线与分位数，没存逐细胞样本）。必须由 §3.3 授权的那条 sensor-only calibration run 一并落盘。
>
> **三点是否真的能分开闭环行为，是 12 格短屏回答的问题，不是这条轴能保证的。**

因此 HYB2 不再把三个点称为“初始 hazard 内点”，改用**标准化负荷下的开环累积耗竭坐标**：

$$
S_Z(I_{th})=
\frac{1}{T_{cal}}\int_0^{T_{cal}}D_Z(t;I_{th},\tau_{Z,down},\tau_{Z,up})\,dt,
\qquad T_{cal}\ge 3\,\tau_{Z,down}=15\ \mathrm{s}.
$$

锁定规则：

1. 使用同一条已保存、slow-off 的 inhibitory-load trace，离线 replay Z；
2. 固定 `tau_Z_down=5000 ms`、`tau_Z_up=20000 ms`；
3. 用 `I_th=q75` 与 `I_th=q50` 在**同一 replay、同一 tau**下定义弱/强端点；
4. 验证 `S_Z(I_th)` 单调；
5. 在 `S_Z` 坐标的 25% / 50% / 75% 处反解三个 `I_th`；
6. 三个值在新 40k 结果出现前写入 plan/manifest，此后不得移动。

若 `S_Z` 不单调或 replay trace 不足以覆盖端点，判 `DESIGN_BLOCKED_Z_RESPONSE_AXIS`。
不得回到 `I_th×tau_z` 二维网格。

---

## 6. 生命周期结构与执行顺序

Gate B0、Gate A0、Z response axis 均通过后，才允许复用 12-cell screen：

`3 Z points × ELR{off,on} × X{off,on}`，M off，no kick。

严格顺序：

1. baseline invisibility；
2. actuator efficacy；
3. Z response-axis lock；
4. 12-cell short screen；
5. ≤2 survivor 的长窗七门；
6. 生命周期 candidate 出现后固定一档 M；
7. 最后 seed3 与 unseen noise confirmatory。

任一上游门失败，不扩参数网格。

---

## 7. 七门保持不变，但恢复统计补充为正式合同

生命周期七门沿用 HYB1 已锁版本：

1. no kick spontaneous onset；
2. pre interictal ≥8 s；
3. bounded high-energy bout 1–5 s；
4. ≥12/15 contacts 且 source-space onset 非同时；方向只作描述，不声称 E1146 双向传播；
5. `X` 在 onset 后 ≥100 ms 启动；
6. post 连续 ≥8 s 且回到 pre/baseline 的 IED rate、IEI CV、duration、participation、
   silent-fraction 统计邻域；永久静默、快速复燃、固定周期串都失败；
7. zero clip / no runaway。

坏数据回归仍为：

- q75 必须挂 sustained-high gate；
- q50 必须挂 pre 与 recovery；
- q50 without X 必须挂 runaway。

ELR 不能替代这些门，也不能用七门结果反调 B0/A0 参数。

---

## 8. 与最终 paper figure 的接口

只有完整 lifecycle candidate 出现后才生成 candidate figure，并保存：

1. virtual-SEEG：interictal → ictal → interictal；
2. `D_Z–R_evt–D_X` 相轨迹；
3. pre / early / ictal / post 的 energy field；
4. source-space onset、contact recruitment、active radius、occupied voxels；
5. 后续 eigenmode/stimulation 所需的四段状态快照与固定 ordering。

未得到 candidate 时只画 gate/diagnostic figure，不画成功生命周期占位图。

---

## 9. 允许 / 禁止

允许：

- “HYB1 发现浓度记忆与 IED 间隔同量级，导致跨事件棘轮”；
- “HYB2 检验一个从 B2.1 recruitment effect 抽象出的事件尺度有界执行器”；
- 若 B0/A0 通过，可分别称 baseline-invisible / recruitment-effective；
- A0 **不过**时**只能**写："**当前这个短记忆、无扩散、阈值化的 ELR 未能保留 B2.1 的招募 extent 效应**"。
  **不得**写成"B2.1 的招募依赖跨事件浓度记忆"——HYB2 同时改了**至少三件事**（记忆 654 ms → 数十 ms、
  删掉扩散、新增 deadband/阈值/饱和曲线），单臂失败无法把原因归到其中任何一件。
  真要单独归因记忆，需要 memory × diffusion 的对照；**本轮不扩成 2×2，只收紧措辞**。

禁止：

- 把 `R_evt` 称为 extracellular potassium concentration；
- 脚本 event reset；
- 为过 baseline 门调 `Q_on`，或为过 actuator 门调 `I_R,max`；
- 在七门前称 seizure lifecycle；
- 未经动力学分析称 limit cycle / Hopf / bistability；
- 未测 onset gradient/latency 时称 propagation；
- source/sink 成功措辞。

---

## 10. 进入 implementation plan 前必须锁定的五项

本文件升为 `DESIGN LOCK` 前，plan 必须从既有 artifact 解析并写死：

1. 从 canonical event onsets **与 offsets** 锁定 `T_event,guard`、`GAP_05/01/min`、`tau_R`
   的唯一数值与可行性（**GAP，不是 IEI**）；
2. calibration/validation 划分、`Q_on`、`Q_scale`；
3. 核验 B2.1 force anchor `I_R,max=4.134151260609386` 的单位链与代码 sha；
4. `S_Z` 三个目标点与反解后的三个 `I_th_EI`；
5. matched actuator input 的唯一 provenance、窗口和 ≥10% 判据实现。

任何一项无法从现有 artifact 得到时，状态保持 `DESIGN LOCK CANDIDATE`，不得进入 40k。
