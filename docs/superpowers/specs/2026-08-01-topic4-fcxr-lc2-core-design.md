# FCXR-LC2-Core — local hysteretic carrier, relay-mediated offset, and recovery

日期：2026-08-01

状态：**DESIGN LOCK CANDIDATE — REVIEW REQUIRED BEFORE EXECUTION**

中文名：**FCXR-LC2-Core：局部迟滞高态—乘性 relay 终止—间期恢复**

Plan：`docs/superpowers/plans/2026-08-01-topic4-fcxr-lc2-core.md`

> 本文件取代同日的 LC2 大 spec 作为下一轮唯一候选执行合同。旧文件已标记
> `REVISION REQUIRED — DO NOT EXECUTE`，不得混用其 T1–T12、M gate 或七门顺序。

---

## 0. 唯一科学问题

在已验收的 40k RC1 fast substrate 上，局部 recurrent state H 能否产生一个**有限、非饱和的高态
basin**；现有 presynaptic E→E relay X 能否通过同一 recurrent 通路消灭该 basin；现有 Z/X 能否在
同一条无 kick SNN 轨迹中完成进入、offset、postictal protection 和稀疏不规则 IED 的恢复？

本代只验证：

\[
\boxed{Z\text{ entry}\rightarrow H\text{ carrier}\rightarrow X\text{ offset}
\rightarrow Z\text{ recovery}.}
\]

`M=0`。3–8 Hz、宽带、E1146 dynamotype、精细空间招募和论文图均不属于 Core 成败条件。

---

## 1. 范围与继承资产

### 1.1 In scope

- RC1：feedforward additive、recurrent E→E conductance、`g_sat*tanh` smooth saturation；
- 原 40k、L=20、各向异性 E→E、双端低阈值 core；
- 新 local H；
- LC1 已有 X relay；
- 已有 Z；
- 一个 development seed，Core candidate 后再跑两个预锁 replication seeds；
- basin probe、frozen-state fork 和一条动态无 kick lifecycle。

### 1.2 Out of scope

- per-cell/mean-field M；
- E1146 真实窗、dynamotype、3–8 Hz、1–80 Hz、15-contact morphology；
- detailed first-passage/front/axis 结论；
- `{4k,8k,16k}` scaling；
- hidden confirmation seeds；
- K/Na、HYB1/HYB2/ELR、recruited-area A；
- eigenmode、完整因果矩阵、最终论文图。

上述内容只在 `CORE_LIFECYCLE_REPLICATED` 后进入 deferred Phenotype spec。

### 1.3 继续冻结的资产

1. RC1 boundedness 与 interictal workpoint；
2. 原连接、延迟、权重、病理轴和 core；
3. LC1 X 的 causal presynaptic relay 实现；
4. 现有 Z 方程和非对称恢复能力；
5. current-based virtual SEEG 只作诊断；
6. kick/reset/parameter step 不计入 lifecycle，只能作 basin/causality probe。

---

## 2. 真实 SNN 数据流：已核实、必须同构

当前代码路径已经核实：

```text
spike(t)
  -> slow.step snapshots x_relay(t-) as ee_relay_send
  -> presynaptic E->E edge weights multiplied by ee_relay_send
  -> delay ring / recurrent synaptic filter
  -> I_E_rec
  -> gA_raw = c_E * I_E_rec / (E_E - V_match)
  -> RC1 tanh saturation
  -> conductance membrane
```

因此 `I_E_rec/gA_raw` **已经是 post-X recurrent drive**。`mz_slow_vars.py::membrane_terms` 同时能看到
该量并控制 tanh 之前的 recurrent conductance；LC2-Core 不需要为 H 修改 blessed engine。

工程硬原则是：

> **H-off 必须与 RC1 逐位一致。**

“blessed 文件永远不可修改”不是科学原则；但当前接口已足够，本版不应为同一功能扩大改动面。

---

## 3. H/X 同构方程

### 3.1 坐标

- `z`：inhibitory availability；`d=1-z` 是 depletion；
- `x_relay,j`：presynaptic E→E availability；
- reduced termination load：

\[
\ell_X=1-\overline{x_{relay}},\qquad 0\leq\ell_X\leq1.
\]

禁止把 `x_relay` 本身称为上升的 termination load。

### 3.2 post-X fast recurrent drive

\[
g_A(r_E,d,\ell_X)=(1-\ell_X)G_A^{RC1}(r_E,d).
\]

`G_A^{RC1}` 包含实际 LIF/RC1 recurrent input-output relation；d 通过 E/I operating point 改变响应，
不另造 additive excitation。

### 3.3 local H

\[
\tau_H\dot h=-h+g_A(r_E,d,\ell_X).
\]

定义零基线平滑 gate：

\[
S_0=\sigma(-\theta_H/k_H),\qquad
\widetilde S_H(h)=\frac{\sigma[(h-\theta_H)/k_H]-S_0}{1-S_0}.
\]

因为 `h>=0`，该式在 `h=0` 严格为零且保持平滑。

最终 recurrent conductance：

\[
g_{rec}^{eff}=g_{sat}\tanh\left[
\frac{g_A+\rho_H\widetilde S_H(h)}{g_{sat}}
\right].
\]

### 3.4 X 终止路径

Core reduced model 中**删除** `-g_X x` 加性负电流。唯一 offset 路径为：

\[
\ell_X\uparrow\Rightarrow g_A\downarrow\Rightarrow
h\text{ 不再获得负荷}\Rightarrow h\downarrow\Rightarrow
g_{rec}^{eff}\downarrow\Rightarrow\text{high basin disappears}.
\]

已有 H conductance 可以在 X 上升后按 `tau_H` 短暂残留，但 X 必须最终对 high basin 有 authority。
如果 H 单独维持高态、任意可达 relay load 都无法关闭，判 `X_HAS_NO_OFFSET_AUTHORITY`。

### 3.5 SNN 离散因果顺序

在 step n：

1. membrane 使用 `h(t_n^-)` 产生 `gH_n`；
2. 本步 post-X `gA_raw,n` 只更新下一步状态：

\[
h_{n+1}=h_n e^{-dt/\tau_H}+(1-e^{-dt/\tau_H})g_{A,n}^{raw};
\]

3. 当前 spike 的 E→E send 仍使用 `x_relay(t_n^-)`。

不得让同一步输入立刻通过 H 反馈自己。H 是共享原 W_EE 拓扑的 slow-conductance proxy，不声称重建
真实 NMDA 生化动力学；本版 `B(V)=1`，保留 RC1 已有 reversal dependence。

---

## 4. tau_H 只由时间可分离性决定

在 equilibrium：

\[
h^*=g_A(r_E^*,d,\ell_X),
\]

因此 `tau_H` 不决定 fixed-point branch 是否存在。禁止再扫
`tau_H × rho_H` 来寻找 equilibrium hysteresis。

### 4.1 所需 sensor traces

现有归档大多只保存 rate/LFP 或 pooled histogram，没有足够的逐细胞 `gA_raw(t)`。因此必须运行
**同配置 sensor-only replay**，H 输出严格为零，并验证 rate/LFP/事件摘要复现原 artifact：

1. accepted RC1 returning-IED baseline；
2. LC1 q75 dense event train；
3. HEO1 sustained 16 Hz state；
4. HEO2 fast-10% intermittent/clonic state。

这些旧状态只提供 sensor input，不作为目标 phenotype。

### 4.2 可分离性

对 sensor trace 数值求解一段宽 tau 区间的 LPF；这只是离线求可行区，不是 SNN 参数 sweep。对每个
tau 定义：

\[
L_\tau=Q_{0.999}(h_{IED}),\qquad
U_\tau=\min_{s\in\{HEO1,HEO2\}}Q_{0.10}(h_{s,established}).
\]

两类 `h_established` 都排除 onset/offset；Q0.10 自然测量 high-state trough。LC1 q75 dense train 只作
过渡输入诊断：它可以部分激活 H，不参与 L/U 硬门，避免把 onset 前的事件加密误当作必须关闭。

要求 bootstrap 后仍有：

\[
L_\tau^{upper95}<U_\tau^{lower95}.
\]

同时报告：下一次 IED onset 前的 residual、最短 IED gap residual、high trough residual。若不存在非空
连通 tau 区间，判 `H_SENSOR_NOT_SEPARABLE`，不做 continuation、不启用 H current。

若存在，锁：

\[
\tau_H=\sqrt{\tau_{min}\tau_{max}},\qquad
\theta_H=(L_{\tau_H}+U_{\tau_H})/2,
\]

\[
k_H=(U_{\tau_H}-L_{\tau_H})/(2\ln 9).
\]

这使未归一化 sigmoid 在 L/U 附近约为 0.1/0.9。选择规则在 sensor replay 前锁定，不按下游 branch
结果移动。

---

## 5. targeted reduced geometry

### 5.1 只测三个 operating regions

不先建立完整二维 E/I transfer 工程。只在实际 RC1/LIF 上测：

1. baseline low region；
2. transition region；
3. existing finite tonic/high region。

每区测 local slope、recurrent saturation slope、refractory distance 和噪声重复性。解析 LIF 只作 sanity。

### 5.2 rho_H 的选择

固定 R1 的 `tau/theta/k`。用 measured slopes 计算 H loop gain：

\[
\mathcal L_H(r)=
\frac{\partial\Phi_E}{\partial g_{rec}^{eff}}
\frac{\partial g_{rec}^{eff}}{\partial(g_A+\rho_H\widetilde S_H)}
\left[
\frac{\partial g_A}{\partial r_E}
+\rho_H\widetilde S_H'(h^*)\frac{\partial h^*}{\partial r_E}
\right].
\]

目标：low `<1`，transition `>1`，high 因 RC1 saturation 再 `<1`。先解出使 transition 最大 loop
gain 为 1 的 `rho_crit`，只测试：

\[
\rho_H\in\{1.05\rho_{crit},\;1.25\rho_{crit}\}.
\]

多个通过取较小 rho；禁止扩盒。

### 5.3 先候选、后正式 continuation

先用 multi-start equilibrium、forward/backward slow sweep、low/high initial conditions 判断是否存在三交点/
双 basin。只有出现候选，才用 pseudo-arclength、Jacobian eigenvalues、最小奇异值和 step/tolerance
sensitivity 正式确认 fold。

Core 只要求有限 low/high basin 与非零 hysteresis margin，不要求 Hopf、3–8 Hz 或患者形态。

### 5.4 frozen X geometry

在同一 reduced equation 中增加 frozen `ell_X`，寻找 high branch 消失的
`ell_X,off` 和再次可达的 `ell_X,release`。不能另加 outward current。若 `[0,1-x_min]` 的可达范围内
high basin始终存在，判 `X_HAS_NO_OFFSET_AUTHORITY`。

### 5.5 reduced model 不生成 IED

Reduced model 只证明：

- 给定 d 的 low/high basin；
- 给定 ell_X 的 offset geometry；
- 使用 SNN event statistics 标定的平均 d drift 时，存在几何上可闭合的 d–ell_X 路径。

它不需要也不允许新增 point process/event generator 来伪造 repeated IED。真正的无 kick IED-driven
onset 只在 dynamic SNN 验收。

---

## 6. 40k frozen SNN basin forks

不做三种小网络 scaling。现有计时和内存表明 40k 的 2–5 s fork 比重建三个等价小网络更直接；Core
固定使用原 RC1 40k connection seed 1。

从 exact snapshot 分叉四组（healthy 组含两个 H 初值）。状态场映射在 fork 结果出现前锁死：

- `d` 使用既有 Stage-D 个体失效图 `p_i` 乘 reduced scalar amplitude；
- `h_low_E` 使用 R1 baseline post-X gA 的时间均值；
- `h_high_E` 使用 R1 established-high post-X gA 的时间均值，并只作一个解析缩放，使其加权均值等于
  reduced high equilibrium；不得改空间排序；
- frozen X load 使用 LC1 established-high `x_relay` 空间模板，解析缩放到目标 `ell_X`；若原 artifact
  缺该 snapshot，运行同配置 sensor-only replay 生成，不用 uniform field 替代。

| 组 | d | h init | X load | 预期 |
|---|---|---|---|---|
| A-low/A-high | healthy | low / reduced-high | 0 | 两者均回 low |
| B | susceptible coexistence | low | 0 | 保持 low |
| C | susceptible coexistence | high | 0 | 保持有限 high |
| D | 同 C | high | `ell_X > ell_X,off` | 回 low，且 matched low 仍 low |

B/C 共同证明 basin coexistence；D 证明 X authority。high initialization 是合法 basin probe，不计
lifecycle。每条 2–5 s，判读使用末段持续性、数值有界性、refractory-ceiling fraction、local gain 和
low/high separation；不使用 morphology classifier。

---

## 7. X protection 时间由几何解析选择

由 reduced/frozen fork 得到 offset 后 `ell_X,0`、high 再次可达的 `ell_X,release`，并用 X-clamped-low
recovery fork 测 Z 回到 `d_safe` 的 `T_Z`。若
`ell_X,0 <= ell_X,release`，没有保护区，判 recovery negative。

指数恢复：

\[
\ell_X(t)=\ell_{X,0}e^{-t/\tau_{X,up}}.
\]

用不确定性保守端计算：

\[
\tau_{X,up}^{min}=
\frac{T_Z^{upper95}}
{\ln(\ell_{X,0}^{lower95}/\ell_{X,release}^{upper95})}.
\]

锁 `tau_X,up = 1.10 * tau_X,up^min`；不机械扫 5/10 s。若分母不为正或数值超出预注册最长保护窗，
判 `OFFSET_POSITIVE_RECOVERY_NEGATIVE`。

---

## 8. Dynamic Z/H/X Core lifecycle

### 8.1 Z entry 标定

从 H-off baseline 的 pre-Z inhibitory sensor replay 和 reduced `d_on` 解析求 `I_th_EI`：预测前 8 s
不得跨 d_on，锁定最长开发窗内应跨越。最多允许两个验证点；无解即 Core lifecycle 失败，不开阈值网格。

### 8.2 Core gates（M=0）

| Gate | 问题 | 验收 |
|---|---|---|
| C0 baseline | H/X/Z 是否破坏间期 | 前 8 s 保留稀疏、不规则 returning IED；无 sustained high |
| C1 onset | 是否无 kick 进入 | dynamic Z onset；matched Z-frozen 不进入 |
| C2 high | 是否有限 carrier | 非 runaway、非 refractory plateau，持续到 X offset |
| C3 offset | X 是否因果终止 | X-on offset；同 snapshot X-off 明显延长或到 cap |
| C4 recovery | 是否回间期统计邻域 | postictal protection 后重新出现稀疏不规则 IED |

不要求3–8 Hz、宽带、15 contacts 或漂亮空间传播。

### 8.3 Recovery 分层

Primary observable recovery：

\[
D_{obs}=D[\text{event rate, IEI, duration, participation, vSEEG energy}].
\]

post 与 pre 的 `D_obs` 必须落入 pre-vs-pre bootstrap band。方向/轴只作 Core secondary diagnostic，
不阻塞 offset/recovery。

Latent diagnostics 分开报告：

```text
h < h_off
ell_X < ell_release
z > z_safe
```

禁止把 observable 与 latent variables 标准化后混成一个总距离。永久静默、固定节律或快速复燃均不是
recovery。

### 8.4 Replication

development connection seed 1 通过 C0–C4 后，才运行两个**提前写入 manifest**的 replication seeds。
它们不隐藏，但在 candidate 前不运行、不参与调参。两个均复现才称 `CORE_LIFECYCLE_REPLICATED`。

---

## 9. 允许的科学标签

```text
H_SENSOR_NOT_SEPARABLE
H_LOOP_NO_BISTABILITY
H_HIGH_BRANCH_SATURATED
X_HAS_NO_OFFSET_AUTHORITY
OFFSET_POSITIVE_RECOVERY_NEGATIVE
CORE_LIFECYCLE_CANDIDATE
CORE_LIFECYCLE_REPLICATED
```

工程故障、artifact 缺失或资源中断只扣留判决，不发明新的科学标签。

结果层级：

- 前五个标签：对应机制位置的 bounded negative；
- `CORE_LIFECYCLE_CANDIDATE`：单 development seed、M=0 的核心闭环候选；
- `CORE_LIFECYCLE_REPLICATED`：两个预锁 replication seeds 也复现；
- 即便 replicated，也不能称患者表型、宽带、空间传播或最终 E1146 lifecycle 已完成。

---

## 10. 停机规则

1. sensor tau 可行区为空：停在 `H_SENSOR_NOT_SEPARABLE`；
2. measured-slope rho 两点均无 bistability：`H_LOOP_NO_BISTABILITY`；
3. high basin只靠 refractory ceiling：`H_HIGH_BRANCH_SATURATED`；
4. X 可达 relay load 不能消灭 high basin：`X_HAS_NO_OFFSET_AUTHORITY`；
5. X offset 成立但保护时间与 Z recovery 无相容区：`OFFSET_POSITIVE_RECOVERY_NEGATIVE`；
6. 任一 Core 成功需要 kick、hard reset、parameter step：不计 lifecycle；
7. Core 失败不允许加入 M、K、ELR、A 或患者阈值抢救。

---

## 11. 工程合同

- H off-by-default；`rho_H=0` 对 RC1 raster/trace 逐位一致；
- sensor-only H 可以演化内部 state，但不得改变任何膜项；
- H 使用 post-X `gA_raw`，one-step causal exact exponential update；
- `coop_A=0`；旧 HEO1 cooperative gate 只作为 sensor input/bad-state provenance；
- snapshot/restart 保存 H，fork 后逐位可复现；
- blessed hashes 每 stage 核验；当前实现不修改 blessed 文件；
- 40k 长轨迹 single worker；短 fork 是否双 worker 由单实例 RSS 决定；
- OMP/BLAS/NUMEXPR=1，setsid nohup、PID、flock、RUNNING/DONE/FAILED、resource log；
- 不触碰 sibling worktree/process；
- 结果根 `results/topic4_sef_hfo/fcxr_lc2_core/`；实际生成图后才写中文 `figures/README.md`；
- spec/plan 任一冲突都是 blocker。

---

## 12. Phenotype 解锁

只有 `CORE_LIFECYCLE_REPLICATED` 才解锁：

`docs/superpowers/specs/2026-08-01-topic4-fcxr-lc2-phenotype-deferred-design.md`

届时冻结 H/X/Z，不回到 Core 调参；M_i、E1146 morphology 和空间 recruitment 的失败不得反写成
Core lifecycle 失败。
