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
