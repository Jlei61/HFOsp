# FCXR-LC5：逐细胞 episode-load / pump 自主退出与恢复设计

日期：2026-08-10

状态：**DESIGN LOCK rev2 — 审阅修复后授权 U0–U2；U3–U4 仅在 gate 通过后解锁。**

上游收口：

- `docs/archive/topic4/sef_hfo/fcxr_lc4ef_review_closeout_2026-08-10.md`
- `docs/archive/topic4/sef_hfo/fcxr_lc4f_x_depth_negative_2026-08-10.md`
- `docs/archive/topic4/sef_hfo/mz_fcxr_pump_lifecycle_gate_Ia_2026-07-27.md`

## 0. 唯一科学目标

在不引入任何空间 mask、global seizure sensor 或 recruited-area field 的条件下，回答：

> 一个始终在线、由每个 E 细胞自身 spike history 驱动、在活动下降后仍保留数秒记忆的慢外向恢复电流，能否让当前 `Z→H` 自然进入的有界高态自主退出，并为 Z 恢复留下 postictal 窗口？

最终目标仍是：

```text
稀疏不规则 returning IED
→ repeated IED 累积 D/Z entry coordinate
→ 无 kick 局部 onset
→ H-supported bounded carrier
→ autonomous offset
→ postictal protection
→ Z 恢复
→ returning IED 统计邻域
```

U0–U2 只回答 **termination authority**，不声称完整 lifecycle。U3 才检验自然闭环；U4 才把快 `M_i` 加回去做 ictal morphology。

## 1. 冻结资产与禁止项

### 1.1 冻结

- E1146 40k geometry、两个低阈值 core、患者特异各向异性 E→E 连接；
- RC1：外源 additive current、recurrent E→E conductance、recurrent-only smooth saturation；
- 已验收的动态 `Z_i` entry；
- 当前 local activity-supported `H_i` carrier；
- current-based virtual-SEEG 与现有 event ledger；
- connection/noise seed 分层、exact-state 与 baseline 合同。

### 1.2 U0–U2 禁止

- 不使用 spatially shared X、Gaussian field、core/axis/off-axis 权重；
- 不使用 population rate、seizure classifier、onset detector 驱动机制；
- 不使用 potassium diffusion、HYB2 event field、recruited-area integrator；
- 不改 E→E 拓扑、core mask、Z/H 参数或 tonic drive；
- 不开 `M_i`；不让 `M` 与 `U` 同时承担退出；
- 不通过 kick 产生接受用 lifecycle。kick 只可用于独立 basin 诊断，本 sprint 不需要；
- 不把新状态称为真实 Na concentration、ATP 或完整 pump biophysics。

## 2. 变量职责

| 变量 | 本设计职责 | 明确不负责 |
|---|---|---|
| `D_i=1-z_i` | repeated-IED entry coordinate | episode termination |
| `H_i` | activity-supported bounded carrier | postictal memory |
| `M_i` | U4 才用于数百毫秒 burst/clonic morphology | U0–U3 的 offset |
| `U_i` | 数秒 episode integration、offset、postictal protection | onset detection、空间传播 |
| `X_i` | U0–U4 固定 1；仅保留历史诊断 | lifecycle 主退出 |

所有 E 细胞使用同一组 `a_U, tau_U, h_U, g_U`。空间差异只来自连接和各自 spike history。

## 3. 锁定方程

### 3.1 逐细胞 load

`u_i≥0` 是无量纲、Na/pump-inspired 的 activity load：

\[
\Phi(u)=\frac{u^{h_U}}{1+u^{h_U}},\qquad h_U=3.
\]

连续记法：

\[
\dot u_i=a_U S_i(t)-\frac{1}{\tau_U}\Phi(u_i).
\]

离散因果顺序沿用已测试的 pump plugin：

\[
u_i^{n+1}=\max\left[0,
u_i^n+a_U N_i^n-\frac{\Delta t}{\tau_U}\Phi(u_i^n)
\right].
\]

膜在第 n 步只看到 `u_i^n`；该步 spike 从下一膜步才起作用。spike jump 不乘 `dt`，clearance 乘 `dt/tau_U`。

### 3.2 基线以上的额外外向电流

旧 pump sprint 使用 signed baseline-centered current：

\[
I_{old}=I_{max}[\Phi(u_i)-p_{0,i}],
\]

它在 `Phi<p0` 时提供相对去极化补偿，并已实测改变间期事件时长和源核份额。LC5 **不得原样复活**。

LC5 使用：

\[
I_{U,i}=I_{max}[\Phi(u_i)-p_{0,i}]_+.
\]

其中 `p0_i` 是同一细胞在无机制间期参考窗中的基线 activation，经预锁的 rate-decile shrinkage 得到；它是 baseline state reference，不是空间标签或 seizure threshold。该项只解释为 **baseline 以上的额外 load-activated recovery current**，不解释为 total physiological pump current。

膜方程加入 E-only additive outward current：

\[
\tau_m\dot V_i=F_i^{RC1+Z+H}-I_{U,i}.
\]

I 细胞第一版不加 U；这是模型边界，必须在产物中明示。若未来加入 I-cell load，属于另一个 spec。

### 3.3 不扫描半激活位置

`h_U=3`、无量纲 half activation `u=1` 固定。`K_U` 不作为自由参数，避免与 `a_U/tau_U` 不可辨识。

对每个 `tau_U`，由 pump-off 高态参考率解析选择 `a_U`：

\[
a_U(\tau_U)=
\frac{0.5}{r_{hi,ref}\tau_U},
\]

其中率用 ms 单位一致换算。该规则使参考高态的时间平均质量平衡满足 median `Phi(u*)=0.5`，于是 `tau_U` 主要改变积累/释放时间，而不是偷偷改变高态稳态激活目标。

`r_hi,ref` 必须来自 U1a 的 canonical no-U-actuator 轨迹，且必须先落盘并锁定，之后才能生成任何 formal `u_i`、`p0_i` 或 U-augmented snapshot。禁止先用 provisional `a_U` 积分状态，再用 fresh `r_hi,ref` 改写 `a_U`；这会让源状态与 candidate lock 循环依赖。

定义逐细胞质量平衡量：

\[
q_i^*=a_U r_i\tau_U.
\]

三个 `tau_U` 共用同一个 rate-distribution admissibility gate：`q0.99(q_i*)<0.90`，且所有 eligible E cells 满足 `q_i*<1`。后者的失败计为 divergent-cell fraction > 0；不得把 q99 与 max 写成两个可以互相替代的门。

### 3.4 高态剂量轴

不用裸 `Imax` 比较，使用当前 SNN 自己的 recurrent E drive 归一化：

\[
\Gamma_U=
\frac{\operatorname{median}_i I_{U,i}}
{\operatorname{median}_i I_{EE,i}^{force}},
\qquad
I_{EE,i}^{force}=g_{EE,i}^{eff}(E_E-V_{match}).
\]

在固定的 pump-off high-state reference 与相应 `u_i,p0_i` 上解析反推全局 `Imax`。`Gamma_U` 的 numerator 与 denominator 必须使用同一个锁定样本域 `E_hi={(i,t): i 为 eligible E cell，t 位于 U1a 锁定的 high-reference window}`、同一个时间权重和同一个细胞支持集。`I_EE_force` 使用 RC1 锁定的 `V_match`；零/非有限 force 样本按预锁规则排除并报告分母保留率，不得看结果后换 support。分母或 excess activation 中位数若非正，calibration fail，不得换统计量救活。

## 4. 锁定探索面

第一阶段只扫两个轴：

\[
\tau_U\in\{3,8,15\}\ \mathrm{s},
\qquad
\Gamma_U\in\{0.10,0.25,0.40\}.
\]

共 9 格，外加一个完全匹配的 `Imax=0` control。不得增加第三个轴、扩大范围或根据图临时加点。

旧 per-cell separation artifact 只提供可行性旁证：5 s raw load 在 `h=3` 下 interictal population activation 约 0.0089、ictal 约 0.811；它没有检验 closed-loop termination，不能代替本 screen。

用该 artifact 的 high-rate median 66.32 Hz 做零仿真 sanity，可得暂定 `a_U` 为 0.002513 / 0.000942 / 0.000503（tau 3/8/15 s）；同一归一化下 interictal rate median/max 对应 activation 约 0.023/0.207，high-state rate median/max 约 0.498/0.810。它说明解析尺度在数值上可行，但这些数不进入 candidate lock；T3 fresh capture 必须重算。

## 5. H 语义合同

当前代码中的 H source 是经过 presynaptic X relay 和 recurrent path 后、进入 RC1 平滑饱和前的 recurrent conductance `gErec_raw`；H 以 `tau_H` 低通该 source，因此它按实现是 activity-supported，而不是外部 latch。`gErec_raw` 不得称为 effective conductance。U1/U2 必须同时记录 `gErec_raw`、饱和后的 `gErec_eff`、`I_EE_force` 与 H。LC5 不增加 `U→H` hard reset。

必须验证因果链：

\[
I_U\uparrow
\Rightarrow r_E\downarrow
\Rightarrow I_{EE}^{eff}\downarrow
\Rightarrow H\downarrow
\Rightarrow carrier\ collapse.
\]

若 `I_U` 已达到 `Gamma=0.40`、活动明显下降，但 H 不下降或 high branch 仍由独立 `g_H` 支撑，则判 `H_BYPASS_OR_CARRIER_MISALIGNMENT`，停止加大 U。

## 6. U0：LC4 收口与 instrument audit

U0 不跑长仿真：

1. 固化 LC4e/f 的收窄口径和修订图；
2. 核对 existing pump plugin 的方程、因果顺序、snapshot、intervention 与 off-by-default parity；
3. 新增 `rectified_excess` 模式，保留旧 `signed_centered` 仅为历史重现，禁止静默改旧语义；
4. 核对 H source 确实来自有效 recurrent drive；
5. 建立 atomic artifact transaction 与 JSON sanitizer。

U0 通过只表示 instrument 可用，不表示 pump 有效。

## 7. U1：先捕获 canonical 轨迹，再离线生成 formal U state

### 7.1 U1a：canonical no-U-actuator capture

只运行一条 LC5 nominal trajectory：`Z dynamic`、`H dynamic`、`X≡1 from t=0`、`M≡0 from t=0`、U membrane current off、no kick/reset/parameter step。LC4f 的 dynamic-X post-onset snapshot 只能作 prefix/provenance 旁证，禁止作为 U2 source substrate。

U1a 保存：

- 至少 8 s pre-onset baseline；
- onset、onset+1 s、+4 s、late high-state 的原始 exact SNN states；
- membrane、synapse、delay ring、refractory、Z/H、RNG/counter-based input state；
- 从 baseline 到 late 的完整 per-cell sparse spike stream；
- fresh `r_hi,ref`、逐细胞 baseline/high rate fields；
- `gErec_raw`、`gErec_eff`、`I_EE_force`、H、rate、event ledger、virtual-SEEG traces；
- 完整 canonical config/code/input hash。

U1a 不生成 formal `u_i/p0_i`；任何为了在线监控而计算的 provisional observer state 都不得进入 candidate lock。若 22 s 内没有自然进入，记 `U1_ENTRY_NOT_REPRODUCED`，停止；不允许 kick 代替。

### 7.2 U1b：锁尺度后的 deterministic offline replay

U1a 发布后，先用 fresh `r_hi,ref` 锁定三套 `a_U`，再对同一条 sparse spike stream 离线重放 `tau_U={3,8,15}s`，生成 formal `u_i`、rate-decile shrinkage `p0_i` 与 onset/+1 s/+4 s/late U-augmented source states。三套 load 必须共享完全相同的 spike history，不得为三个 `tau_U` 重跑三条噪声轨迹。

若 sparse stream 无法无损重放，只允许在 `a_U` 锁定后对 canonical no-U-actuator trajectory 做一次 deterministic replay；必须证明 main-state/input hashes 与 U1a 一致。任何在 `a_U` 锁定前生成的 load state 都不得进入 U2。

## 8. U2：exact high-state fork 的 3×3 authority screen

所有 arm 从同一个 onset+1 s exact snapshot 出发：

- `D/Z` 冻结在 snapshot 值，避免 entry coordinate 继续把终止面推远；
- H 保持动态，必须随 recurrent activity 自然回落；
- `X=1`；`M=0`；
- 相同预生成 external input；
- 每条 8 s；40k 严格单 worker；
- pump-off control 使用同一 load state并继续 sensor evolution，只把 `Imax=0`。

这一步是 **counterfactual actuator-on authority test**：只问当前 carrier 是否存在 cell-autonomous U termination authority，不测自然 entry、自然 offset latency 或 recovery。U2 的 offset 时间是 step-on sufficiency readout，不能写成模型的自然 seizure duration。

### 8.1 轨迹标签

- `HIGH_EQUILIBRIUM`：8 s 内持续高态或只连续降 rate、无稳定低态；
- `BOUNDED_OFFSET`：启动后 0.5–6.0 s 内退出，并在同一 8 s fork 内连续观察 ≥2 s 保持低于 pre-locked interictal upper band；
- `LATE_DECLINE_UNRESOLVED`：6–8 s 才出现持续下降或候选退出，剩余窗口不足 2 s，不能算 offset；
- `OVER_SUPPRESSED`：<0.5 s 直接塌陷、数值虽稳定但没有有限高态驻留；
- `BURST_SILENCE_LOOP`：反复熄灭/再点火，未形成 episode-level offset；
- `NUMERICAL_FAIL`：非有限、clip/cap、异常早停。

### 8.2 U2 正门

只有同时满足才可进入 U3：

1. pump-off control 8 s 持续当前 high carrier；
2. 至少一个格点为 `BOUNDED_OFFSET`，且同一 `tau_U` 或同一 `Gamma_U` 邻接方向呈 coherent dose ordering（例如 `HIGH_EQUILIBRIUM → BOUNDED_OFFSET → OVER_SUPPRESSED`，或相邻格也为 bounded），没有数值失败；两个相邻 positive cells 作为宽参数窗证据单独报告，不是 U3 的硬前提；
3. offset 前 high-state dwell ≥0.5 s；
4. offset 伴随 `I_EE_force` 和 H 同向下降，不是 classifier trough；
5. offset 后 `I_U` 在 ≥2 s 内仍明显高于 pre-ictal baseline，具备 postictal memory；
6. onset+4 s 的 U2b replay 仍为 `BOUNDED_OFFSET`；
7. U2c baseline-compatibility replay 通过；
8. finite、zero hard clip、无 refractory plateau。

通过时按最小 `Gamma_U`、再取最接近 8 s 的 `tau_U` 选 primary；最多再保留一个相邻 sensitivity。不得按“图最好看”选。

### 8.3 U2 停机分支

- 9 格全 `HIGH_EQUILIBRIUM`：`U_AUTHORITY_NO_GO_IN_CURRENT_H`；检查 H bypass，不继续加剂量。
- 9 格全 `OVER_SUPPRESSED` 或从 low/high 两边无相邻正窗：`NO_ROBUST_U_WINDOW`。
- 有 offset 但 H 不降：`H_BYPASS_OR_CARRIER_MISALIGNMENT`。
- 只有 burst-silence：`FAST_ADAPTATION_LIKE_NOT_EPISODE_OFFSET`。

上述均是当前 H substrate + locked U family 的机制结论，不外推到完整离子模型。

### 8.4 U2b 移动慢状态鲁棒性

U2 初筛产生的最多两个候选，还必须在 U1 的 onset+4 s 与 late exact snapshots 上各做一次 8 s fork；每次冻结各自的 D/Z，H 保持动态。这样直接检验 LC4f 暴露的移动终止面问题，而不把 9 格乘成 27 格。

onset+1 s 与 +4 s 是授权所需的 hard states。late replay 是 stress test，只报告 `late-established-carrier rescue: pass/fail`；它失败不能单独封锁 U3，但必须在 U3 风险表中显式携带，禁止包装为“对全部 visited slow states 稳健”。

### 8.5 U2c 间期兼容性 replay

对最多两个 U2b 候选，从同一个 pre-onset interictal exact snapshot 各做一对 6–8 s replay：`Imax=0` 与 candidate；D/Z 冻结在间期工作点，H 动态，外源输入严格相同。比较 returning event count、duration、participation、peak、core-A/core-B share、virtual-SEEG energy 与 mean `I_U`。

因为 `[Phi-p0]+` 是凸整流，`p0_i=mean Phi` 不能保证间期 mean current 为零。候选只有在上述统计落入 canonical baseline 的预锁 block-to-block variability 才算 baseline compatible；否则即使高态可终止也不得进入 U3。

## 9. U3：最多两个候选进入完整自然闭环

U2、onset+4 s U2b 与 U2c 都过门后才解锁。每条至少 70 s，机制从 t=0 始终在线，无 kick/reset/parameter step，Z/H 动态，X=1，M=0。

U3 的初始 load 必须唯一确定：对每个 E cell 使用 `u_i(0)=Phi^{-1}(p0_i)`，使 `I_U(0)=0`，并记录初值 hash；不得把 onset/late load 状态带回 t=0。该初始化只保证执行电流基线一致，不预判未来是否 onset。

必须同时满足：

1. ≥8 s 稀疏、不规则 returning IED；
2. repeated IED 后自然 onset；
3. 1–5 s 有界 high carrier；
4. autonomous offset；
5. postictal suppression 中 `U` 保持、D 持续下降；
6. all-E mean `D=mean_i(1-z_i)` 回到 canonical quiet-stable reference band（历史中心参考约 0.047，执行时须读取完整 band 而非只抄单点）；该条件只作 supportive coordinate，不替代低态稳定性、returning-IED 分布与 no-rebound；
7. `T_D,recover < T_U,release`；
8. 最后 ≥8 s 的 event rate、IEI、duration、participation、空间模板落回 pre-ictal reference distribution；
9. returning IED 实际出现，不以 mean rate 代替。

开发 seed 通过后才运行独立 connection seed；confirmation seed 不参与参数选择。跨 seed 锁定 `tau_U,h_U,Gamma_U` 与目标 `Phi(u_hi*)=0.5`；每个 seed 只可从 outcome-blind nominal reference 用同一解析公式重算 nuisance `p0_i,a_U,Imax`，不得根据 lifecycle 结果回调。

## 10. U4：快 M 只塑造 morphology

只有 U3 已有完整 lifecycle scaffold 才解锁。锁住 U candidate，打开一小面 `M_i` fast adaptation，用于 3–8 Hz envelope、宽带和错相；必须保住 U3 的 entry/offset/recovery。mean-field M 是负对照。U4 不属于本次初始实施授权。

## 11. 必须输出的图与主图边界

### U2 诊断图

1. 3×3 label map；
2. representative rate / H / `I_EE_force` / `I_U`；
3. per-cell U activation distribution（interictal vs high vs post-offset）；
4. `D` frozen 标识与 exact-state/noise hash。

### U3 lifecycle 图

只有 U3 全过才生成 paper-candidate：

- virtual-SEEG：interictal → ictal → postictal → returning interictal；
- `(mean D, mean Phi(U))` 慢路径，注明只是 readout 坐标，模型内部仍为 per-cell states；
- early-ictal energy field 与 interictal scaffold alignment；
- interictal / early-ictal / pre-offset / postictal / recovered response mode。

U2 图不得伪装成完整 lifecycle 图。

## 12. 工程合同

- 六个 blessed engine 文件不改；`mz_slow_vars.py` 若改，必须加入 mechanism hash；
- new mode off-by-default；pump off 必须 byte-parity；历史 `signed_centered` fixture 不变；
- 40k 严格 1 worker，OMP/BLAS/NUMEXPR=1；提交前检查 sibling 与 swap；
- 所有长任务 `setsid nohup` + stage flock + PID + RUNNING/DONE/FAILED sentinel；
- swap 增量 ≥256 MiB 停止新提交，≥512 MiB 终止最新 worker；
- 每 1–2 s 仿真时间原子覆盖一个 rolling exact checkpoint，并流式追加 event-ledger/summary checksum；长期仅保留 pre-onset、onset、early、pre-offset、post-offset、recovered landmarks，不永久保存全部滚动状态；
- 外源输入使用 counter-based deterministic stream 或分块 replay，不生成巨大的 `N×T` 输入数组；
- NPZ/JSON/ledger/state 全部写临时文件，schema/hash 全过后 atomic rename；
- 失败路径清理 RUNNING sentinel，但保留 FAILED 与日志；
- 不触碰 sibling worktree，不 push/merge，除非用户另行授权；
- U1a 完成后实测 `c_wall=T_wall/T_sim`；后续 wall-kill 用 `1.5*c_wall*T_target + I/O margin` 计算，并受 12 h hard safety cap 约束。禁止用固定 4 h 冒充 70 s lifecycle 的机器实测预算。

## 13. 允许的最终措辞

U2 若通过：

> 在冻结 entry coordinate 的 exact high-state fork 中，逐细胞 episode-load current 具有终止当前 H-supported carrier 的局部动力学权威。

U3 若通过：

> 在一个 development/confirmation seed 集合上，逐细胞 load–recovery coordinate 与 Z/H 形成了无 kick 的 bounded lifecycle scaffold，并回到 returning-IED 统计邻域。

未通过 U3 前禁止：seizure lifecycle、recovery established、patient mechanism、真实 Na/K pump reconstruction。
