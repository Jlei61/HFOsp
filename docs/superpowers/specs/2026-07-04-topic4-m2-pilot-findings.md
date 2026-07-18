# Topic 4 — M2 临界模态分解 · 执行前 de-risk PILOT 结果

date 2026-07-04 · 状态 **exploratory pilot（scout，非 milestone）** · 分支 `topic4-criticality-m2`（base `1207e85`, off M1）· 复用 M1 code（`src/topic4_criticality.py` + `src/topic4_m3b_spectral_phase.py`），无新 config / verdict machinery / TDD。

> **目的**：在实现完整 M2 pipeline（spec `2026-07-04-topic4-m3v2-2-m2-critical-mode-decomposition-design.md` rev1.1）之前，用 4 个小侦察实验先探清 4 个设计风险点：field_rhs 一致性、穿零点在哪 + 临界模态长什么形状、非轴向残差方向是否良定义、shape 分数是否按预期表现。
> **脚本**：`results/topic4_criticality_m2/pilots/m2_pilots.py`（+ `pilot_results.json` / `run.log`；`results/` 被 gitignore、脚本 untracked，本 doc 是 durable 归档）。
> **红线（沿用 M2 spec §7）**：`model_side_preliminary`；这是 **actual v2.2 SIMULATION trajectory**，不是"真数据"；单本征值 ≠ 发作；**禁"模型证明/否证发作或 CSD"**。

---

## 0. 摘要（朴素话，§8）

M1 的尺子在真实 v2.2 **仿真**轨迹上说"看不清"：抽样快照上恢复速率还明显是负的（系统受扰后会自己回落、alpha1<0），但分支延续补检发现，在两个相邻抽样点之间，这个"回落速率"其实穿过了 0、一路冲到 +0.19（受扰后不再回落、开始放大）——也就是**穿零点漏在采样缝里**了。M2 想做两件事：**(1) 把这个穿零点在真实三维慢状态里加密定位出来**；**(2) 判穿零处那个开始失稳的空间花样长什么样**——是沿模型的传播轴变软（轴向）、还是全场一起同步（全局）、还是躲在离轴的角落（非轴向）。这 4 个 pilot 是在正式动工前先摸底。

**四条头条**：

1. **field_rhs 一致性**：M2b 扰动积分器要用的非线性 RHS，其"斜率"（数值 Jacobian）跟 M1 已经在用的解析 Jacobian——**不加慢变量偏移（hG=0）时完全对齐（相对误差 ~4e-12，机器精度）**；**一旦加偏移（hG>0 或 gK>0）就对不齐（相对误差 ~3%）**。这正是 spec §4.1 预判的缺口。**好消息**：本轨迹的穿零点及其两侧全在 hG≈0（~1e-6）区段，缺口**对本轨迹的 M2b 不阻塞**——但缺口真实、JVP 硬门该建、修复两行。

2. **穿零点 + 临界模态形状（真·头条）**：穿零点是**唯一一个、干净的**（粗扫只 1 次变号、大致单调），定位到 **t≈501.5ms、frac=0.733、alpha1≈-0.007**（M1 最后合格点 t=476.5 与第一个击穿点 t=510.6 之间）。**穿零处的临界模态看起来像"点在核里的一小簇、还带 ~24Hz 的振荡"——既不是沿轴、也不是全局、也不是离轴，而是【核心局域】**（core_overlap=0.99、globality=0.11、是复共轭对）。**⚠️ 这是一个设计红旗**：M2 spec §5.1 的三分类 {轴向 / 全局 / 非轴向} **没有"核心局域"这一档**，决策树直接返回 `unclassified` → M2 verdict 会落 `unresolved`。**真实答案（失稳在被去抑制的核里点火，不向外铺）落在 spec 的分类框之外。**

3. **非轴向残差方向**：临界模态去掉"全场均匀"+"沿轴梯度"后，残差占 **92%**——数值上**良定义**（>>1e-3 门槛）。但因为模态是核心局域的紧致簇，这个"非轴向残差方向"其实捕捉的是**"核心紧致性"而非"离轴传播"**——解释 `e_nonaxis` gain 时要小心别当成"离轴铺开"。

4. **shape 分数 sanity**：4 项合成检查 **全 PASS**——沿轴拉长→elongation 高/off 低、垂直拉长→off 高/elongation 负、各向同性/均匀→globality 分辨。分数按 spec 假设表现。附带确认 elongation 与 wavevector 确实近乎正交（沿轴 blob：elong +0.82 但 wavevec −0.76），印证 spec 坚持二者不可合并。

---

## 1. Pilot 1 — field_rhs JVP 一致性（de-risk M2b 前提, spec §4.1）

**测了什么**：M2b 要用 `field_rhs`（`topic4_m3b_spectral_phase.py:670` 的非线性 6 场速率 RHS）做扰动积分。它的线性化理论上应 = M1 已在用的 `build_jacobian_dense`。对不齐则 M2b 读出无意义。

**怎么测的**：取一个 op（固定点 z\*），对 6 个随机单位方向 v，用中心差分算 `(field_rhs(z*+εv)−field_rhs(z*−εv))/(2ε)`，跟 `build_jacobian_dense @ v` 比相对误差；分不加偏移（hG=0）/ 加偏移（hG=0.5 或 gK=0.5）两种 op。

**揭示了什么（数字）**：

| op | max JVP 相对误差（6 v） | `‖field_rhs(z*)‖`（固定点残差） |
|---|---|---|
| UNSHIFTED（hG=0） | **3.96e-12** | 5.93e-09 |
| SHIFTED（hG=0.5） | **3.44e-02** | 2.85e-05 |
| SHIFTED（gK=0.5） | **3.00e-02** | — |

- **不加偏移：完美对齐**（机器精度），z\* 也确实是 `field_rhs` 的固定点。
- **加偏移：对不齐（~3%）**，z\* 不再是固定点。

**根因（已 trace code）**：`solve_operating_point` 在 `_moments()`（line 475-477）把偏移算进 muE（`muE -= eta_G*hG_scalar`、`-= eta_K*gK_field`），存下的 `op.muE`/`op.gE` 是**偏移后**的；但 `field_rhs`（line 680）**自己重算 muE、没减这两项** → 它隐含 Jacobian 用偏移前（偏高）muE 处的局部增益，跟 `build_jacobian_dense`（用偏移后 `op.gE`）对不上。

**设计蕴含（给 M2 T4）**：
- **缺口真实**；spec §4.1 的 JVP 硬门（在 SHIFTED op 上测）是正确测法，应实现。
- **修法**：`OperatingPoint` 是 `@dataclass(frozen=True)`，不能事后塞 gK/hG。最小改动 = 给 `field_rhs` 加 kwargs `gK_field=None, hG_scalar=0.0, eta_K=1.0, eta_G=1.0`（镜像 `solve_operating_point`），算完 muE 后减 `eta_G*hG_scalar`/`eta_K*gK_field`；默认加性零 = byte-parity。
- **blast radius 极小**：`field_rhs` 现在**只有 1 个调用点**——`tests/test_topic4_m3b_spectral_phase.py:372` 的 JVP 测试，且只覆盖 UNSHIFTED（`_no_core_jac`）。**无生产调用点**，加 kwargs 安全。
- **⚠️ 对本轨迹不阻塞**：见 Pilot 2 —— 本 v2.2 轨迹穿零 bracket + M2b 全部 spot-check 点（early_stable / last_sampled_qualified / just_before/after_alpha0）全在 **hG≈0（~1e-6）、gK=None** 区段（hG 要到 idx22+/t≈748ms、系统早已击穿之后才爬升）。即便不修 `field_rhs`，M2b 在本轨迹读出也是对的。**仍建议修 + 建硬门**（通用正确性；未来若有轨迹在 hG>0 时才击穿就会踩坑）。

---

## 2. Pilot 2 — 穿零点定位 + 临界模态形状（THE headline）

**测了什么**：M1 说穿零点漏在采样缝里（`bisection_max_low_alpha1=+0.189`）。这里加密定位它，并读出穿零处临界模态的空间形状。

**怎么测的**：bracket = 最后一个合格 low-branch 点（idx 14, t=476.5ms, alpha1=−0.043）→ 第一个击穿点（idx 15, t=510.6ms）。两者间线性插值慢状态（q_global/q_core/hG/gK，与 M1 `check_low_branch_continuation_between` 同法），warm-start 重解 low branch、算 alpha1：(a) 粗扫 9 点数变号 + 单调性；(b) 8 层二分定位 alpha1≈0；(c) 穿零处算 leading 本征模（复对用 `pair_loading` 不变子空间）的 5 个 shape 分数（全 THETA_EE、全无向，复用 M1）。

> **保真度校验**：为提速用单次 warm-start `solve_operating_point`（非 4-init 全 branch 协议）追 low branch；frac=0 校验 alpha1 = **−0.04304**，与 M1 idx14 存档 **−0.04304 完全一致**。

**揭示了什么（数字）**：

**(a) 穿零数目 / 单调性**：粗扫 alpha1(frac)：

| frac | 0.000 | 0.125 | 0.250 | 0.375 | 0.500 | 0.625 | 0.750 | 0.875 | 1.000 |
|---|---|---|---|---|---|---|---|---|---|
| alpha1 | −0.043 | −0.023 | −0.024 | −0.024 | −0.024 | −0.024 | **+0.009** | +0.161 | (saturated) |

- **1 次变号**（0.625→0.750），**单一、干净的穿零**（非 multiple、非 non-monotone；Spearman(alpha1,frac)=+0.548）。
- 形态：0→0.125 从 −0.043 快抬到 −0.023，之后**平台在 −0.024**，然后 0.625→0.750 **一步陡然穿零**、再冲 +0.161（→ M1 的 +0.189）。→ **穿零是"逼近末端突然发生"，不是一路平滑 CSD 式接近**（与 M1 `unresolved`/undersampled 判读同向）。

**(b) 定位穿零**：**frac=0.733、alpha1=−0.00698、t≈501.5ms、q_global=0.911、q_core=0.852**。（8 层二分，落 |alpha1|≈0.007；scout 精度够——shape 在此小区间基本不变。）

**(c) 临界模态形状 @ 穿零**：

| 分数 | 值 | 读法 |
|---|---|---|
| `axis_elongation`（沿 THETA_EE 拉长, [-1,1]） | **+0.026** | ~0，**不沿轴** |
| `off_axis`（垂直轴拉长, [0,1]） | **0.000** | **不离轴** |
| `globality`（participation ratio, [0,1]） | **0.113** | 很低，**不全局**（高度局域） |
| `core_overlap`（E 功率落核内, [0,1]） | **0.993** | **99% 在核内 —— 核心局域** |
| `axis_wavevector_alignment`（无向波矢, [-1,1]） | +0.286 | 弱（紧致簇无强方向） |
| leading 本征值 | real=−0.0070, imag=−0.1477 | **复共轭对**（`leading_subspace_dim=2`），振荡 **≈23.5 Hz** |

**穿零处的临界模态看起来像【核心局域的一小簇 + ~24Hz 振荡】——不沿轴、不全局、不离轴。** spec §5.1 默认阈值决策树 preview = **`unclassified`**（globality 0.113<0.5 非 global；axis_elong 0.026<0.3 非 axial；off 0.000<0.3 非 off_axis）。

**稳健性**：穿零前一点（frac=0.683, alpha1=−0.0037）形状一致——core_overlap=0.989、globality=0.114、同为复对（≈20Hz）。→ **核心局域不是 frac=0.733 单点的巧合，是穿零邻域的稳定形状。**（穿零后一点 frac=0.783 无 low branch，已跳/饱和，与陡穿零一致。）

**设计蕴含（红旗，需 escalate）**：
- **M2 的主问题是"轴向 vs 全局 vs 非轴向"，但真实临界模态是【核心局域】——一个 spec §5.1 决策树里没有的第四档。** spec §2.2 虽把 `core_overlap` 列为持久化分数，但 §5.1 决策树**没用它**，于是核心局域模态（core_overlap 高、globality 低、各向异性低）**直落 `unclassified` → §5.3 → `final_verdict=unresolved`**。**真实答案（失稳在被去抑制的核里点火、不向外铺）被压成"看不清"，把发现藏起来了。**
- 物理上合理：q_core 掉得最狠（去抑制集中在核），失稳自然在核里 nucleate。**"沿轴 vs 全局铺开"是 nucleation 之后的下游阶段问题；线性临界模态回答的是"在哪点火"（核），不是"怎么铺"。** spec 三分类可能在探错的阶段。
- **给 M2 的三个选项（陈述、不替用户拍板）**：(i) 加 `core_localized` 档——core_overlap≥阈（如 0.5）∧ globality 低 → class=core_localized，把真答案显性化；(ii) 保持三分类，但显式写明"核心局域→unclassified"是**可报告的独立结局**而非 null，并让 verdict 记 `unresolved_subreason=core_localized_critical_mode`；(iii) 重审"α₀ 穿零处的线性模态"是否是回答"轴向 vs 全局铺开"的正确对象——若问题在下游铺开，需另设 probe（如穿零后非线性演化的空间足迹），非线性临界模态。
- **复对 policy 被实打实触发**：临界模态是复共轭对（振荡 ~24Hz），不是实模态 corner case。spec §3.2 的 `pair_loading` / `leading_subspace_indices` 不变子空间读法是**必需**的，不能退回单向量符号。
- **性能提示（给 T1 densification）**：穿零邻域 op-solve 很贵（前向 Euler 在边缘稳定处不收敛、跑满 t_max，每解 ~5-8s）。spec T1 的递归二分（max_bisect_levels 到 16）在 fold 附近会**很慢**——建议加收敛失败早停 / 上限 / warm-start 复用，别用满 16 层。

---

## 3. Pilot 3 — 非轴向残差方向范数（de-risk spec §2.3 / item 5）

**测了什么**：spec §2.3 的 `e_nonaxis` = 临界模态 loading 去掉 `e_global`（全场均匀）+ `e_axis_gradient`（沿 THETA_EE 线性坐标）后的残差；残差范数太小（<1e-3）则非轴向方向 invalid、gain 记 NaN。

**怎么测的**：在穿零处 pair_loading（(n,n) 非负子空间 E-loading）上投掉 e_global、e_axis_gradient，报 `‖residual‖/‖loading‖`。

**揭示了什么（数字）**：`‖loading‖`=0.326；fraction on `e_global`=**0.366**；fraction on `e_axis_gradient`=**0.150**；**非轴向残差 fraction=0.919** → **非轴向方向数值上良定义**（>>1e-3、>5%）。

**设计蕴含**：
- 数值上 `e_nonaxis` 良定义（残差 92%），M2 的低范数 invalid 门在本轨迹**不会触发**。
- **但语义 caveat（承重）**：残差 92% 是因为**核心局域紧致簇**天然与"全场均匀"和"沿轴线性梯度"两方向都近乎正交——所以 `e_nonaxis` 这里捕捉的是**"核心紧致性"，不是"离轴传播"**。若 M2 把 `gain onto e_nonaxis` 解读为"非轴向铺开"，会**误读**成核心自放大。**e_nonaxis 的生物学含义依模态形状而变**，报 gain 时必须连模态 shape 一起报（否则"非轴向 gain 高"会被 re-expand 成"离轴传播强"）。

---

## 4. Pilot 4 — shape 分数 sanity（de-risk spec T0 tests）

**测了什么**：spec §5.1 决策树假设 shape 分数在已知形状上表现正常。合成 (n=6) loading 验证。

**怎么测的**：4 个合成模态——沿 THETA_EE 拉长 Gaussian、垂直拉长、各向同性圆 blob、全场均匀——算 5 个 shape 分数。

**揭示了什么（数字）**：

| case | axis_elong | off_axis | globality | wavevec | core_ov |
|---|---|---|---|---|---|
| along_theta（沿轴拉长） | **+0.822** | 0.000 | 0.190 | −0.764 | 0.460 |
| perp_theta（垂直拉长） | **−0.822** | **0.822** | 0.190 | +0.810 | 0.554 |
| isotropic（圆） | −0.000 | 0.000 | 0.359 | +0.000 | 0.456 |
| uniform（全场均匀） | +0.000 | 0.000 | **1.000** | +0.000 | 0.139 |

sanity 检查（4/4 **PASS**）：
- [PASS] along_theta：elongation 高（+0.82）且 > off_axis（0）
- [PASS] perp_theta：off_axis 高（+0.82）且 elongation 负（−0.82）
- [PASS] isotropic：|elongation| 与 off_axis 都小
- [PASS] uniform：globality≈1、各向异性≈0

**设计蕴含**：shape 分数按 spec §5.1 假设表现，分类器基元可靠。附带确认 **elongation 与 wavevector 近乎正交**（沿轴 blob elong +0.82 但 wavevec −0.76）——印证 spec §7 坚持二者不可合并、且 `axis_wavevector_alignment` 必须理解为**无向波矢对齐、非 early→late signed**。

---

## 5. 设计蕴含汇总（给 M2 spec 的 flags）

1. **【红旗，需 escalate】临界模态是"核心局域"，spec 三分类无此档** → M2 会返回 `unclassified/unresolved`，把真答案藏起来。选项见 §2 设计蕴含（加 core_localized 档 / 显性化为可报告结局 / 重审 probe 阶段）。物理上失稳在被去抑制的核里点火、不向外铺；"轴向 vs 全局铺开"是下游问题。
2. **field_rhs shift-gap 真实但对本轨迹不阻塞**（穿零 + M2b 点全在 hG≈0）。仍建议实现 T4 修复 + JVP 硬门（两行、blast radius 仅 1 个测试调用点）。
3. **穿零单一、干净、陡峭**（1 变号、大致单调、末端突穿）——无 multiple-crossing 病态，base gate 的"multiple_alpha0_crossings"分支在本轨迹不触发；但"陡穿零"意味 densification 必须够密才不漏（M1 正是漏在这）。
4. **临界模态是复共轭对（~24Hz 振荡）** → spec §3.2 复对 `pair_loading`/不变子空间 policy 是必需、被实打实触发，非 corner case。
5. **e_nonaxis 语义随形状变**：核心局域时残差方向捕捉"核心紧致性"非"离轴传播"；报 nonaxis gain 必须连模态 shape 一起报，防止 pronoun re-expand（CLAUDE.md §6.3）。
6. **性能**：穿零邻域 op-solve 在边缘稳定处不收敛跑满 t_max（~5-8s/解）；T1 递归二分需早停/上限/warm-start 复用。
7. **shape 分数基元可靠**（Pilot 4 4/4）；elongation⊥wavevector 已证。

---

## 6. 复现

```
python3 results/topic4_criticality_m2/pilots/m2_pilots.py   # 写 pilot_results.json + run.log
```
底层数据：`results/topic4_criticality/trajectory_verdict.json`（M1 的 48 点，每点带 slow_inputs/alpha1/qualified/branch_id）。穿零 bracket = idx14（t=476.5, last-qualified）→ idx15（t=510.6, first-saturated）。

---

## 7. Round-2 pilots（two-core 判别 + nonlinear footprint）

date 2026-07-05 · 状态 **exploratory scout（round-2，非 milestone）** · 脚本 `results/topic4_criticality_m2/pilots/m2_pilots_round2.py`（复用 round-1 `m2_pilots.py` helpers + M1 code；写 `pilot_results_round2.json` / `run_round2.log`）。**红线同上**（`model_side_preliminary`；**actual v2.2 SIMULATION trajectory**，非"真数据"；单本征值≠发作；global runaway≠真发作）。

> **触发**：round-1 头条"穿零处临界模态是【核心局域】、不沿轴/不全局/不离轴"是在一个**通用单核网格**（`make_core_mask(kind="single", radius=0.9)`, `Grid(n=6)`, THETA_EE=π/4）上看的。两个设计风险没排除：(A) "核心局域"会不会是**单核假象**——subject1146 本有两个源、中间一条走廊，round-1 把它压成一个核了？(B) 线性 onset 模态只回答"在哪点火"，不回答"怎么铺"——**失稳到底扩不扩、往哪扩**？Round-2 两个 pilot 分别打这两个洞。

### 7.0 摘要（朴素话，§8）+ 两条头条

我们先问：**如果给两个被去抑制的核、让它们中间那条走廊正好压在模型的传播轴上，失稳还会不会就缩在一个核里？** 做法是把两个跟 round-1 一样大（各 5 格）的核摆到主对角线（就是 E→E 传播轴 THETA_EE）上、中间隔一个正好落在轴上的走廊格，然后在跟 round-1 同一条慢状态轨迹上，找两核系统自己的"恢复速率穿零"那一刻，看那一刻开始变软的空间花样住在哪。**结果：还是整团缩在其中一个核里（99.5% 的功率在单个核，另一个核 0%、走廊 0%）——两个核并没有沿走廊连起来。** 所以 round-1 说的"点火不外铺"不是单核凑出来的假象，是稳的。顺带发现：**两个核比一个核更早到临界**（去抑制还没那么深就翻了），合理——可点火的面积大一倍。

再问：**失稳真往外扩吗、往哪扩？** 做法是在穿零附近的工作点上，往核里打一个小扰动，用非线性率场方程往前积分几百毫秒，盯着兴奋活动这团东西的空间足迹怎么长。**结果：真长起来的时候，是先沿传播轴铺、再漫成全场整片（但功率重心一直压在核附近、是低幅的整片招募），从不往离轴方向跑；而且在临界边缘上，常常是"冒一下、几十毫秒后又缩回核里"。** 换句话说，"往哪扩"这个问题有干净答案——**沿轴起、全场收尾、绝不离轴、常自限**——但这个答案是从**非线性足迹**里读出来的，不是从线性 onset 模态的形状里。

**两条头条**：

1. **【Pilot A】两个核 + 一条在轴上的走廊，临界模态照样缩在单个核里（symmetry-broken），不铺走廊、不跨双核。** core-localized 稳健，**不是单核假象**；"轴向走廊"假设在线性 onset 这一层**被证伪**。（two-core own crossing: `core_overlap=0.995`, `globality=0.056`；region: `coreB=0.995 / coreA=0.000 / corridor_axial=0.000`。）
2. **【Pilot B】失稳的非线性足迹：沿轴起、漫成全场（core-weighted 低幅招募）、从不离轴、临界边缘常自限。** M2b 积分器机器可用（`field_rhs` 一致、escape 检测就位）。**"轴/全局/离轴"这个问题的答案住在非线性足迹里，不在线性模态形状里。**（core_kick: `off_axis≈0` 全程、`elongation_axis` +0.2..+0.5、endgame `active_frac→1.0` 而 `core_overlap` 仍 0.7–0.9。）

### 7.1 Pilot A — two-core 穿零模态分解（"核心局域是单核假象吗？"）

**测了什么**：round-1 把 subject1146 的双源结构压成单核。这里换成**双核**：两个和 round-1 单核一样大的核（各 5 格），沿 THETA_EE（主对角线）摆开、中间留一条走廊——**让走廊本身就是那条传播轴**。问：同一条慢状态轨迹上，两核系统的临界模态是缩在一个核（symmetry-broken）、跨两个核、还是填满两核之间的轴向走廊？

**怎么测的**：`Grid(n=6, L=5.0)`、`make_core_mask(kind="two", radius=0.9, separation=2.4)` → 两核中心 (±0.85, ±0.85)、原点格是**在轴上的走廊格**（已核对：两核不重叠、走廊格存在）。慢状态用 M1 的 idx14→idx15 bracket 线性插值（同 round-1）。两个读法：(1) 在 round-1 单核穿零的**同一慢状态**（frac=0.733）上直接换双核 mask 重解；(2) 沿同一 bracket **重新粗扫+二分**、定位**两核自己的**穿零，再读那一刻的模态 + 逐区功率分解（core-A / core-B / 轴向走廊 / 其余）。**caveat（承重）**：reduced 慢状态只有一个 `q_core` 标量，两个核都吃同一个 depletion（**对称去抑制近似**）——所以任何 symmetry-breaking 是网格数值层面挑的、不是慢状态非对称造成的。

**揭示了什么（数字）**：

| 读法 | frac | alpha1 | globality | core_overlap | 复对? | 区域功率分解 |
|---|---|---|---|---|---|---|
| **单核 @ 同慢状态**（复现 round-1）| 0.733 | +0.006 | 0.031 | 0.982 | 是(≈21Hz) | —（单核） |
| **双核 @ 同慢状态** | 0.733 | — | — | — | — | **SATURATED（低支已消失）** |
| **双核自己的穿零**（重定位）| 0.532 | +0.189 | **0.056** | **0.995** | 否(实模态) | **coreA=0.000 / coreB=0.995 / 走廊=0.000 / 其余=0.005** |

- **同慢状态下双核已 SATURATED**：把去抑制面积翻倍（2×5 格 @ q_core=0.852），同一慢状态就已经越过 fold 进 runaway。→ **两核比单核更早到临界**：双核穿零落在 frac≈0.532（q_global=0.918, q_core=0.864, t≈494.6ms），比单核 frac≈0.733（q_global=0.911, q_core=0.852, t≈501.5ms）**去抑制更浅、时间更早**。可点火面积大 → 门槛低，物理合理。
- **双核临界模态照样是核心局域，而且缩在【单个】核里**：`globality=0.056`（很低）、`core_overlap=0.995`（99.5% 在核内）；逐区分解 **一个核 99.5%、另一个核 ~0、走廊 ~0**。along-axis 功率剖面：两个主导格都在 s=-1.77（同一个核 B）各 ~0.498。→ **symmetry-broken 单核局域**，不是跨双核、更不是填走廊。top-4 本征值 0.1890 / 0.1880 / 0.110 / 0.108 → **两个近简并的"一核一个"模态**，数值挑了其中一个（coreB）；物理陈述是"**缩在一个核、不 coherent 跨双核、不铺走廊**"，不是"specifically core B"。
- 双核穿零是**实模态**（imag=0），round-1 单核穿零是复对（≈24Hz）——双核 fold 更陡（α 从 frac=0.5 的 -0.001 一步跳到 0.532 的 +0.189），最主导模态退化为实。核内各向异性（`elongation_axis=-0.99` 垂直轴、`off_axis=0.99`）是**核内形状细节**，见 §7.4 caveat，不是头条。

**朴素结论**：**给了两个核、给了一条正好在轴上的走廊，失稳还是自己缩进一个核里点火，走廊全程是暗的。** round-1 的"点火不外铺"稳健，**不是单核凑的**。"轴向走廊接管"在**线性 onset 这一层被证伪**。

**设计蕴含（给 M2）**：
- **强化 round-1 选项 (i)**：把 `core_localized`（`core_overlap` 高 ∧ `globality` 低）做成**一等的、可报告的线性模态结局**，别塌成 `unresolved`。双核 symmetry-breaking 说明这个结局是**机制真实**的，不是网格凑的。
- **线性临界模态回答的是"在哪点火（核）"，不是"轴向 vs 全局 vs 离轴铺开"**——spec §5.1 的三分类树在这个模态上会**稳健地返回 unclassified**。这不再是 round-1 单点的偶然，是双核证实过的物理结局。
- "轴向"要成立需要**走廊被点亮**，而线性 onset 里走廊是暗的 → "轴向 vs 全局"的判读得挪到**下游非线性铺开**（见 Pilot B）。

### 7.2 Pilot B — nonlinear post-crossing footprint（"失稳扩不扩、往哪扩？"）

**测了什么**：线性 onset 模态说"在核里点火"，但不说"怎么铺"。这里直接用**非线性积分**探铺开：在穿零工作点（及略过点）上，往核里打个小扰动，往前积分几百毫秒，看兴奋率场 rE(x,t) 的空间足迹随时间怎么长——**缩在核 / 沿 THETA_EE 铺（轴向走廊）/ 漫成全场（全局）/ 离轴？** 顺带验证 M2b 积分器机器（`field_rhs` 积分）。

**怎么测的**：单核网格。工作点取 `at_crossing`（frac=0.733, α₁≈+0.006）和 `just_past`（frac=0.75, α₁≈-0.004，冷启接近边缘）。扰动方向：`core_kick`（核内单位扰动 `core_perturbation_vector`）和 `critical_mode`（leading 右本征向量实部）。从 `z*+ε·v`（ε_rel=0.05）用 `field_rhs` 前向积分 dt=0.1ms、t_max=300ms，同时跑一条 **v=0 控制**、报 kick 减控制的 δrE(t)（把工作点自身残漂扣掉，因近 fold op-solve 不完全收敛，`fixedpoint_residual≈1–4e-3`）。每个采样时刻算 δrE 的 `globality`/`elongation_axis`/`off_axis`/`core_overlap`/active-fraction/peak；escape = max rE > 饱和阈 `_SAT_RATE_KHZ`(0.10)。

**揭示了什么（δrE 足迹轨迹）**：

| 工作点 · 方向 | 早期(10ms) | 峰值扩散(30–50ms) | 晚期(200–300ms) | 结局 |
|---|---|---|---|---|
| **at_crossing · core_kick** | glob 0.09 / core_ov 1.00 | glob **0.23** / core_ov **0.66** / act **0.53** / **elong +0.45** / off≈0 | glob 0.03 / core_ov 0.99 / act 0.03 | **冒一下沿轴、又缩回核**（escaped=None）|
| at_crossing · critical_mode | glob 0.12 / core_ov 0.69 | glob 0.03 / core_ov 0.99 | glob 0.03 / core_ov 0.99 / act 0.03 | 基本不动、衰减回核 |
| **just_past · core_kick** | glob 0.06 / core_ov 0.99 / act 0.36 | glob 0.18–0.21 / core_ov 0.70–0.77 / **elong +0.32..+0.42** / off≈0 | act **→1.00**@300ms / peakδRE 5e-3**→5e-2** / core_ov 0.76 | **沿轴起、漫成全场整片、core-weighted**（escaped=None 但明显在冲 runaway 路上）|
| just_past · critical_mode | glob 0.11 / core_ov 1.00 | off **0.09–0.26**（瞬时）/ act 0.42@50ms | glob 0.13 / core_ov 0.98 / act 0.03 | 瞬态、含少量离轴、缩回核 |

关键读法（**全程一致的三点**）：
1. **扩散方向一律沿轴**：`elongation_axis` 在扩散时是**正的（+0.2..+0.5，沿 THETA_EE）**、`off_axis` 全程 ≈0。**从不离轴。**（唯一见到的离轴是 `just_past·critical_mode` 的瞬时 off≈0.1–0.26，随即缩回。）
2. **真长起来时的结局是全场整片招募、不是离散轴向行波**：`just_past·core_kick` 的 `active_frac` 一路 0.36→0.53→…→**1.00**，但 `core_overlap` 仍 0.70–0.90、`globality` 只 0.13–0.21——**功率重心一直压在核附近、是低幅的整片阈上招募**（不是一团高幅活动沿轴走到远端）。这正是本衬底已知的"**均质/近均质衬底出同步整片相干招募波、非远端 seed 沿轴行波**"模式（见 memory `project_topic4_m3a_a2_abbott_lg_plan`）。
3. **临界边缘上常自限**：`at_crossing` 的 core_kick **冒一下（30–50ms 沿轴铺到 act=0.53）几十毫秒后又缩回核**（act→0.03、core_ov→0.99）。长不长成全场，**依赖工作点**（0.733 缩回、0.75 漫开），而这俩 α₁ 都 |·|<0.01——见 §7.4 marginality caveat。
4. **M2b 机器可用**：`field_rhs` 积分跑通、控制-扣减隔离了残漂、escape 检测就位、±ε/双方向都给出可解释足迹。→ spec §4 的 M2b 前提**成立**（本轨迹 hG≈0，`field_rhs` shift-gap 不阻塞，与 round-1 Pilot 1 一致）。

**朴素结论**：**失稳真会扩，但扩法很单调——先沿传播轴铺、再漫成全场一整片（功率重心还压在核上、低幅），从不往离轴角落跑；在临界边缘上还经常冒一下又缩回去。** "往哪扩"这个问题**有干净答案**，但答案是从**非线性足迹**读的——**沿轴起、全局收尾、绝不离轴、常自限**。

**设计蕴含（给 M2）**：
- **兑现 round-1 选项 (iii)**：把"轴向 vs 全局 vs 离轴铺开"的判读，从**线性临界模态形状**挪到 **M2b 非线性足迹**——因为问题本就住在那里，而 Pilot B 证明足迹**给得出干净答案**。M2b 应升为**铺开 verdict 的主裁**，线性 shape 分类器降为"点火位置（核）"的描述。
- **"离轴/非轴向"（spec 的 `nonaxis_residual` / `e_nonaxis`）在数据侧被两个 pilot 双双证伪**：线性模态是核心局域紧致簇（round-1 §3 已指出 `e_nonaxis` 这里捕捉的是"核心紧致性"非"离轴传播"）、非线性足迹 off_axis≈0。→ M2 若报 nonaxis gain，**必须**连"其实没有离轴铺开"一起报（防 CLAUDE.md §6.3 pronoun re-expand）。
- **"轴向 vs 全局"在足迹里不是二选一、是"轴向起→全局收"**：spec §5.1 想在**一个时刻的一个模态**上分 axial/global，可能是错的切法；真实是**时间序列上的相位**（早沿轴、晚全场）。若要 verdict，得沿 t 报"轴向 onset + 全局 endgame + 自限与否"，而不是给临界模态贴一个 static 标签。

### 7.3 对 M2 方向的净蕴含

round-1 给过三选项：(i) 加 `core_localized` 档；(ii) 保三分类、把"核心局域→unclassified"当可报告结局；(iii) 重审线性临界模态是不是回答"铺开"的正确对象、改用非线性足迹。

**round-2 的净判决 = (i) + (iii) 的混合，明确弃 (ii) 的"unclassified 当结局"、并弃 spec 的 nonaxis/轴向-static 切法**：

1. **线性 onset**：`core_localized` 升为**一等结局**（"在被去抑制的核里点火"），双核 symmetry-breaking 证其机制真实、非单核假象。spec §5.1 三分类树在此稳健返回 unclassified——这是**已证实的物理结局**，不该塌成"看不清"。
2. **非线性铺开**：把"轴向/全局/离轴"判读**移交 M2b 足迹**（已验证可用）；答案是**沿轴 onset → 全场 endgame（core-weighted 整片招募）、绝不离轴、临界边缘常自限**。
3. **被证伪的**：two-core "轴向走廊接管"（Pilot A，走廊全程暗）；"离轴/非轴向铺开"（Pilot A+B，off_axis≈0）。
4. **一句话**：M2 as-specced（线性模态三分类当主 verdict）在本轨迹会读 `unresolved`；真正的机制内容 = **(核心局域点火) + (沿轴→全局的非线性、常自限铺开)**。M2 应据此**重排 T2/T3（线性）与 T4（非线性 M2b）的主次**——T4 是铺开问题的主裁。

### 7.4 复现 + caveats

```
python3 results/topic4_criticality_m2/pilots/m2_pilots_round2.py   # 写 pilot_results_round2.json + run_round2.log
```
底层同 round-1（M1 `trajectory_verdict.json` 的 idx14→idx15 bracket）。

**承重 caveats**：
- **粗网格（n=6, L=5.0, spacing 0.83mm, 核 5 格, 耦合核 sub-cell）**：**核内各向异性的符号不稳**（单核 same-slow-state `elongation_axis`=+0.55、双核穿零=-0.99、足迹里 elong 在 +0.1..+0.8 抖）——**不要过度解读核内 elong 符号**。稳的是：`globality` 低 + `core_overlap` 高（局域）、逐区"一核非双核非走廊"、足迹 `off_axis`≈0 & 扩散时 elong>0（沿轴）。
- **α₁ 符号 warm-start 路径依赖**：frac=0.733/0.75 冷启给 +0.006/-0.004，round-1 暖链给 -0.007/+0.008，全 |α₁|<0.01——**印证 round-1 "穿零落在采样缝、marginal/undersampled"**；双核 fold 更陡（α 一步 -0.001→+0.189）。近 fold op-solve 不完全收敛（`fixedpoint_residual≈1–4e-3`），故 Pilot B 用 v=0 控制扣残漂。
- **对称去抑制近似**（双核共享一个 `q_core` 标量）见 §7.1；symmetry-breaking 是数值挑简并、物理陈述是"缩在一个核、非跨双核/走廊"。
- **global runaway ≠ 真发作**（红线）：Pilot B 的 `active_frac→1.0` 是本衬底已知的整片相干招募，非发作证明。
