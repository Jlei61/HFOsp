# Topic 4 MZ-FCXR：E-cell full-conductance + persistence-gated spatial relay resource 设计

日期：2026-07-20

状态：**DESIGN LOCK CANDIDATE — 先审阅，不启动 40k SNN**

分支：`codex/topic4-mz-conductance`

## 0. 一句话设计

下一版不直接重跑普通 E→E STD。先把 E-cell 上仍为 additive current 的 AMPA 输入改成真正的 conductance，检验固定慢状态下是否出现有限高活动支；只有该 fast-topology gate 通过，才加入一个**只识别持续局部招募、对普通短 IED 基本不响应**的 presynaptic E→E relay availability `x_j`，让 recruited region 后方形成空间 refractory wake。`Z` 继续负责 event-locked onset push；旧线性 `M` 主实验关闭；快速阈值适应 `phi` 只在有限高支已经存在但没有 within-bout oscillation 时作为第二阶段波形机制。

本设计取代 `mz_conductance_stage_handoff_2026-07-20.md §6` 中“直接做普通 `U_x × tau_x` 网格”的执行建议。旧段保留为思考记录，但不再是可执行合同。

## 1. 为什么必须修订上一版 `x_j` 设想

### 1.1 普通 STD 已经做过，不能当成新机制

仓库中的 M1/M4-2 已实现 per-presynaptic-cell E→E availability：每个 E spike 后乘性耗竭、指数恢复。它在两个不同问题上已经给出有界 negative：

- 能让事件时间自限，但可读事件仍随网格扩大到边界；
- 在已有 bounded-persistent M4 工作点上，把动力学推成 `persist → fragment → suppress`，没有 clean re-triggerable termination。

因此本线不能把同一个 `ee_std_u × tau` 网格接到 conductance 分支后重新命名。

### 1.2 普通 STD 与当前线性 M 有同一个时序漏洞

它们都从第一个间期 spike 开始积累。当前 L=20 基线的普通事件只有约 30–55 ms，但会重复出现；一个对瞬时 spike/rate 直接响应的负反馈会在 Z 到达 onset corridor 前先压住正常 IED，结果仍是 prevention、fragment 或静默。

真正缺的是“短而高的 IED 峰值”与“持续 recruited shoulder”的分离，而不是另一档负反馈强度。

### 1.3 当前 conductance 仍不是 full conductance

已验收版本只把 E-cell 的 GABA/M 项变成 reversal-aware conductance；`I_E` 仍作为 additive voltage drive，I cells 仍走旧 current membrane。该版本能说明 additive-global GABA 只移动 runaway/prevention 边界，但不能否定完整 AMPA/GABA conductance fast subsystem 是否存在有限高支。

## 2. 与并行线的独立边界

本线只拥有以下组合：

```text
E-cell full AMPA/GABA membrane conductance
+ per-cell Z inhibitory-efficacy depletion
+ local persistence sensor y_j
+ presynaptic E→E relay availability x_j
+ optional phi only after a finite high branch exists
```

明确不做：

- 不使用 `1/(1+alpha_G S_G)` recurrent divisor；这是 `topic4-mz-divisive-lifecycle` 的对象；
- 不增加 dynamic global inhibition pool 或 slow-gated global divisor；
- 不重做 current-based MZ slow-fast transition、direct spatial modes 或 early-field bridge；
- 不改变 E1146 scaffold、`l_EE/C_EE`、外部 drive、双核位置或电极 montage 来抢救结果；
- 不把另一分支的未验收参数复制到本分支。

共同接口只有 lifecycle gate、registered E1146 masks 和最终状态定义。不同线的正/负结果在综合阶段比较，不在方程中混合。

## 3. 下一版方程

### 3.1 E-cell full-conductance fast membrane

E 细胞使用完整 AMPA/GABA exact exponential conductance update：

\[
\tau_{m,E}\dot V_i=
-V_i
+g^{ff}_{E,i}(E_E-V_i)
+g^{rec}_{E,i}(E_E-V_i)
+g_{I,i}(E_I-V_i).
\]

I 细胞保留已验收的 literal current membrane：

\[
\tau_{m,I}\dot V_i=-V_i+I_i^E-I_i^I.
\]

其中外源/前馈 AMPA 与 recurrent AMPA 必须分开保存；只有 E→E recurrent 分量受 `x_j` 调制。电流 proxy 到 leak-relative conductance 的映射在固定参考电位做 force matching：

\[
g_E=\frac{c_E I_E}{E_E-V_{match}},\qquad
g_I=\frac{c_I I_I}{V_{match}-E_I}.
\]

第一版固定：

- `E_E = 58 mV`（Abbott/Liou 的 `E_E=0, E_L=-58 mV` 平移到本引擎 `E_L=0` 坐标）；
- `V_match = 18 mV`；
- E-cell `E_I=0`、`gaba_gain=1.125`、protected additive-global `beta=1/12` 沿用已验收 conductance anchor；
- `z_scope=local_only`：Z 的耗竭 sensor 与乘法只作用于 local received GABA，protected global 项既不进入 sensor，也不随 Z 耗竭；本轮不得把它悄悄改回 `total`；
- I-cell 不带 Z/global/X 变换，整个首轮保持原 current 路径；将 I-cell 也改成 conductance 只可在 lifecycle candidate 出现后作 sensitivity；
- `c_E=1` 是 force-match anchor，只允许用 slow-off 工作点合同做一个预注册小 bracket，不用 lifecycle 结果选值。

exact step：

\[
V_i(t+dt)=V_{\infty,i}+[V_i(t)-V_{\infty,i}]
\exp[-dt/\tau_{eff,i}],
\]

\[
\tau_{eff,i}=\frac{\tau_{m,i}}
{1+g^{ff}_{E,i}+g^{rec}_{E,i}+g_{I,i}}.
\]

这一改动的动力学目的不是“更生物”，而是让 recurrent excitation 的 driving force 随 `V` 上升而饱和，从 fast subsystem 本身创造有限高支的可能。若 full conductance 仍只有 low 与 operational-runaway 两端，后续 `x_j` 不启动。

### 3.2 Z：保留为局部 onset push

E 细胞：

\[
\tau_z\dot z_i=H(I^{EI}_{th}-I^{EI}_{sensor,i})-z_i,
\qquad 0\le z_i\le1.
\]

\[
g_{I,i}^{E}=z_i g^{local}_{I,i}
+\beta\langle g_I^{local}\rangle_E.
\]

`Z` 只缩放 local received GABA；protected global 项不随 local Z 耗竭。主锚仍是 `q75 / tau_z=2.5 s`，seed 3 confirmation；`tau_z=5 s` 只作已注册 sensitivity。

### 3.3 `y_j`：区分短 IED 与持续招募

每个 E neuron 用自身 spike train 建一个 Hz 单位的局部 persistence sensor：

\[
\tau_y\dot y_j=-y_j+1000\sum_k\delta(t-t_j^k),
\]

即每个 spike 令 `y_j += 1000/tau_y`。第一版 `tau_y=120 ms`，只允许 `80/160 ms` 两侧 sensitivity；它不是全局 rate，也不做空间平均。

阈值只从 slow-off 基线定：

```text
y_gate = pooled seed-1/3 slow-off y_j 的 event-window Q99.9
```

锁定后再看 early-runaway。若普通 IED 中 `y>y_gate` 的细胞比例超过 1%，或 early-runaway 200 ms 内跨阈细胞比例不足 10%，说明 sensor 没有分开两类时间结构，本机制停止，不移动阈值。

### 3.4 `x_j`：高状态门控的 presynaptic relay availability

\[
x_\infty(y_j)=1-(1-x_{min})
\frac{[y_j-y_{gate}]_+^n}
{K_y^n+[y_j-y_{gate}]_+^n},
\]

\[
\tau_x\dot x_j=x_\infty(y_j)-x_j,
\qquad 0\le x_j\le1.
\]

来自 presynaptic E neuron `j`、投向 E target 的 recurrent conductance为：

\[
g^{EE,eff}_{ij}(t)=x_j(t^-)g^{EE}_{ij}(t).
\]

E→I、I→E、I→I、external E 均不受 `x_j` 影响。当前 spike 使用 `x_j(t^-)` 发送，随后才更新 `y_j/x_j`；这样单次 spike 不会即时削弱自己，只有持续放电才逐步留下 outgoing-relay wake。

第一版固定 `n=4`。observer gate 通过后一次性固定：

\[
K_y=\max\{5\ \mathrm{Hz},\
\operatorname{median}(y_{j,early-high}\mid j\text{ recruited})-y_{gate}\}.
\]

若 recruited-cell median 不高于 `y_gate`，sensor gate 直接失败；不做 `K_y × n` 网格。

### 3.5 旧 M 与可选 phi

- 当前线性 `M` 在主实验中 `off`；它只保留为“postsynaptic slow conductance” ablation，不能和 `x` 同时调参。
- `phi` 在 full-conductance + Z + X 已出现 finite high branch 和 clean recovery 后才可开启：

\[
\tau_\phi\dot\phi_i=-\phi_i+\Delta_\phi r_i.
\]

`phi` 的职责仅是把 finite tonic/high plateau 转成 within-bout oscillation；它不承担 onset 或最终 termination。

## 4. 假设性动力学分析

### 4.1 frozen fast subsystem 为什么可能出现有限高支

一个保留 conductance driving force 的归一化 mean-field 是：

\[
V_\infty(r;z,x)=
\frac{[g_0+Jxr]E_E+[g_{I0}+(K_Lz+K_G)r]E_I}
{1+g_0+Jxr+g_{I0}+(K_Lz+K_G)r},
\]

\[
\tau_r\dot r=-r+\Phi(V_\infty-\theta).
\]

固定点满足 `F(r;z,x)=0`；fold 同时满足 `partial F / partial r=0`。因为分母随 excitatory conductance 一起增加，高 `r` 下的正反馈不再按 additive current 无界增长，有限 high branch 或 S-shaped bistable branch 在结构上是允许的。

本轮做了一个**未按 SNN 标定的 normalized hypothesis probe**，只检查结构可能性：

| frozen state | 归一化快系统 |
|---|---|
| `z=1, x=1` | low branch only |
| `z=0.9, x=1` | low + unstable separator + finite high branch |
| `z=0.9, x≈0.65` | high branch disappeared, low branch only |

在同一个示例里，high branch 的存在边界随 `z` 单调移动：Z 越耗竭，越小的 x 仍可支持高态。这个结果不证明 SNN 有同一分岔，但证明“Z 打开高支、X 关闭高支”的符号组合不是数学矛盾。

### 4.2 最可能的 entry/exit 图景

本设计的首选预测不是“Z 自己发生 Hopf”，而是：

```text
stable excitable low branch
  -- repeated IEDs ratchet Z downward -->
lower finite-amplitude ignition threshold / bistable corridor
  -- endogenous fluctuation escapes -->
finite recruited high branch
  -- sustained activity raises y and lowers local x -->
high branch crosses its exit fold
  --> low/postictal branch
  -- z and x recover -->
original interictal regime
```

这是一条 noise-triggered slow passage through distinct entry/exit boundaries；它可以是闭合 excursion，而不必是永久 deterministic limit cycle。

### 4.3 为什么会出现 fragment、单次闭合和 postictal suppression 三个区

高态退出后，近似恢复为：

\[
z(t)=1-(1-z_{off})e^{-t/\tau_z},\qquad
x(t)=1-(1-x_{off})e^{-t/\tau_x}.
\]

令 `x_on(z)` 为 frozen fast map 上重新允许 high branch/低 ignition threshold 的边界。恢复路径必须在 Z 恢复前保持在安全侧：

\[
x(t)<x_{on}[z(t)].
\]

预测：

- `tau_x` 太短：x 在 z 仍低时先恢复，轨迹重新进入 high-permitted 区 → fragment/recurrent bursts，可能形成 noise-sustained relaxation cycle；
- `tau_x` 中等：x 保持低到 z 恢复，得到单次 bounded bout + refractory gap + interictal recovery；
- `tau_x` 太长或 `x_min` 太低：高态能终止，但随后长时间 suppression，无法恢复 returning event。

归一化 probe 的恢复路径在 `tau_z=2.5 s` 时也显示这一分层：短 `tau_x` 重新穿过 high-permitted 边界，中/长 `tau_x` 留在安全侧。它解释了旧 M4-2 为什么容易落到 fragment/suppress 两端，也指出候选不是靠更密的 `U×tau` 盲扫，而应由 frozen fold geometry 反推。

### 4.4 Hopf / limit cycle 的实际可能性

`Z+X` 首先更可能给 fold-mediated relaxation，而不是小振幅 Hopf。若把固定 z 下的 `(r,y,x)` 线性化：

\[
\dot{\delta r}=a\delta r+c\delta x,\quad
\dot{\delta y}=p\delta r-q\delta y,\quad
\dot{\delta x}=d\delta y-e\delta x,
\]

其中 `c>0, d<0`，特征多项式为：

\[
(\lambda-a)(\lambda+q)(\lambda+e)-cpd=0.
\]

延迟负反馈足够强时可以满足三阶系统的 Hopf 条件，但 high Hill gate 更倾向于大振幅 relaxation/fragmentation。真正要一个稳定的 within-ictal oscillatory orbit，更清楚的路线是：先让 full conductance 建立 finite high branch，再让快速 `phi` 在该支上过 Hopf，最后由更慢的 X 把 orbit 推过 fold-of-cycles/exit fold。

展开特征多项式：

\[
\lambda^3+A_1\lambda^2+A_2\lambda+A_3=0,
\]

\[
A_1=q+e-a,\quad
A_2=qe-a(q+e),\quad
A_3=-aqe-cpd.
\]

高支稳定时需要 `A1,A2,A3>0` 且 `A1*A2>A3`；候选 Hopf 边界是 `A1*A2=A3`。这也暴露了设计漏洞：如果 high-state 下 Hill gate 已完全饱和，`d=partial xdot/partial y` 接近 0，延迟反馈只能缓慢搬动 fixed point，不能制造 Hopf。`K_y` 因而锁在 early-high recruited-cell median，而不是设在远离数据的饱和区；即使如此，没有 frozen-state 多周期收敛也不得称 Hopf。

因此预期的分岔 itinerary 是：

```text
finite-amplitude basin escape or entry fold
    -> optional phi-mediated Hopf / bounded high-state oscillation
    -> X-mediated exit fold or fold of cycles
    -> recovery to the low branch
```

在 frozen slow state 下没有稳定多周期 orbit，就只能称 transient burst train，不能称 limit cycle。

### 4.5 设计前的先验排序

以下是结构判断，不是仿真成功率：

| 目标 | 先验判断 | 最可能的失败方式 |
|---|---|---|
| full E-cell conductance 保留 returning-event 工作点 | 中高 | AMPA force matching 改变事件 participation 或频率 |
| frozen fast subsystem 出现低支 + 有限高支 | 中等 | 仍是 low 与 refractory-ceiling saturation 的陡跳 |
| `Z+X` 形成单次可恢复 high excursion | 条件性中等 | `tau_x` 太短为 fragment，太长为 suppression |
| `Z+X` 单独形成真正稳定的 high-state limit cycle | 低 | Hill gate 更容易产生 relaxation transient 而非小振幅 Hopf |
| 加 phi 后形成 bounded within-bout oscillation | 条件性中等 | 只有波形振荡，释放慢变量后仍不能完整返回 |
| 形成不依赖边界的空间 front–wake | 中低、且是最大风险 | `y/x` 在波前到达 L=20 边界后才启动，只能终止时间、不能限制空间 |

因此这个版本最现实的首个阳性终点是“有限空间招募的单次闭合 excursion”，不是永久周期发作。若 Stage 2 latency gate 失败，即使全局 rate 可以漂亮地返回，也必须判为本空间机制 NO-GO。

## 5. 执行阶梯

### Stage 0：full-conductance engineering + slow-off 工作点

1. 新分支完全 off 时与 `9e0fa4f` byte parity；
2. 引擎把 E-cell external/forward AMPA、recurrent E→E 与其他 excitatory input 分开累计；只转换 E-cell received AMPA，已验收的 E-cell GABA conductance及 I-cell current 路径保持不变；
3. E-cell AMPA/GABA conductance exact-step 单元测试、reversal sign、`tau_eff`、finite/clip fail-fast；I-cell current parity；
4. `L=20, T=8 s, seed=1` slow-off，随后 seed 3；
5. `c_E` 只允许 `[0.85, 1.00, 1.15]` force-match bracket，按旧 slow-off event count/duration/participation/rate 的距离选择，不看 Z/X lifecycle；
6. 若三点都不能保留稳定 returning-event train，停止 full-conductance route。

### Stage 1：先找 fast high branch，不加 X

先跑 full-conductance Z-only 自然轨迹；若存在 transition，注册 baseline、pre-onset、early-high 三个 state。若 12 s 内只有 returning events，则使用已经锁定的 onset-depletion spatial pattern，做 controlled `D=[0,0.05,0.075,0.10,0.125,0.15]` frozen field，不因自然轨迹缺少 transition 就调 Z。冻结 Z/global terms，用同一 RNG checkpoint 做两类 initial condition：native low state与固定 deterministic probe 产生的 recruited state；若自然 early-high 存在，它替代 probe high initial condition。

对每个 state 只测：

```text
x_clamp = [1.00, 0.85, 0.70, 0.55, 0.40]
window = 1000 ms
M/phi/y dynamics off
```

本阶段禁用 `120 Hz / 100 ms` phenotype early-stop；它是旧 operational-runaway 标签，不是数值发散。只保留 nonfinite、conductance clipping、`tau_eff` 与内存安全 stop。要区分：low only、finite high only、bistable、bounded orbit、持续贴近 refractory ceiling 的 saturation。只有出现不依赖 early-stop 的 finite high branch/orbit，才进入 Stage 2。若仍是 low↔ceiling-saturation cliff，停止；负反馈不能替代缺失的 fast-state topology。

### Stage 2：persistence sensor 分离 gate

只跑 observer，不改变膜或突触：slow-off returning events 与 Z early-high 各一条。按 §3.3 锁 `y_gate`，检查 IED false activation 与 recruited sensitivity。

空间 latency 也是硬 gate：在 early-high 回放中，source/core recruited cells 的 `y` crossing 必须早于 registered first-boundary recruitment 至少 30 ms，并且 crossing-time surface 应位于活动前沿之后、不是全片同时出现。若 `y` 只在活动已经铺满 L=20 后才上升，它最多是时间 terminator，不是 spatial relay mechanism；本线停止，不把 `tau_y` 缩到违反 IED false-activation gate。

### Stage 3：由 fold 反推 X，不做笛卡尔网格

从 Stage 1 得到 `x_off(z)`，固定：

```text
x_min = first registered x clamp below the high-branch exit boundary
tau_x candidates = 使解析 T_term 约为 [0.5, 1.0, 2.0] s 的三个值
```

高态近似的 termination time：

\[
T_{term}\approx\tau_x
\log\frac{1-x_\infty^{high}}{x_{off}-x_\infty^{high}}.
\]

seed 1、`T=12 s`、单 worker；三格与 X-off control 共 4 cells。只要没有 bounded bout，停止，不增加第二参数轴。

### Stage 4：自然 L=20 lifecycle confirmation

候选才跑 seeds 1/3、`T=20 s`、每次最多 2 workers。要求同一自主轨迹：

1. 至少 3 个正常 returning interictal events；
2. 随后进入持续高招募 bout；
3. bout 不命中 numerical safety stop；它可以跨过旧 `120 Hz / 100 ms` operational-runaway 阈值，但必须随后形成有限 envelope 并自主返回；
4. 自主终止并连续回到 same-seed baseline band至少 2 s；
5. 后期出现 returning interictal event，或 registered fixed probe 恢复到 baseline response band；
6. seed 1/3 同方向。

seed 4 只作 stress，不混 primary denominator。

### Stage 5：只有 temporal lifecycle 通过才做空间与 phi

- 先做空间 front/wake gate；
- 若 finite high bout 是 tonic plateau、没有 oscillation，再固定 X 参数只开 `phi`，用 Abbott-scale `tau_phi=[80,120,160] ms` 三格；
- `delta_phi` 只能按“slow-off IED unchanged + frozen high branch出现有限周期”的双合同选，不按最终图好不好看选；
- phi 关、X 关、Z 关和 clamped-X 均是必要 ablation。

## 6. 空间验收合同

本线的承重对象不是全局 rate，而是 source-space recruitment 与 local resource wake。

固定 masks：source core、sink core、axis corridor、off-axis field、core-excluded field、15-contact E1146 readout。每 1 ms 在线累计 coarse spatial spikes，不保存完整 `T×NE` raster；只在 registered event windows 保存 per-cell first onset、`y_j` 与 `x_j` snapshot。

候选必须同时满足：

- duration 与 recruited fraction 超过 same-seed interictal P99，但不是一帧全片同步；
- onset surface 有有限传播时间，不能把 near-simultaneous remote activation写成 wavefront；
- `x` 低值区位于已招募区/前沿之后，形成可重复的 front–wake separation；
- wake 内 matched local probe 的 propagation gain 低于同 Z、未耗竭前方区域；
- bout 结束不依赖 L=20 边界：primary 用 registered edge-margin gate；候选后只做一个更大 L diagnostic，事件尺度不能随 L 线性长大；
- recovery 后原有 source→sink 或 reverse interictal template重新出现，而不是永久换成另一张静态图。

结果分类：

```text
bounded_front_with_wake
bounded_global_sync_no_front
boundary_limited_spread
fragment_train
prevention_or_suppression
delayed_runaway
recovered_no_template
full_lifecycle_candidate
```

只有最后一类能交给 ecomode、early-ictal bridge 和电气相变三条下游 workflow。

## 7. 动力学验收合同

### 7.1 fold / bistability

需要 frozen Z/X state forks、low/high 两类 initial condition 和上下扫；不能从单条自然轨迹宣称 bifurcation。至少报告：

- fixed-point/plateau/cycle rate 与 spatial extent；
- entry 与 exit 的不同边界；
- hysteresis width；
- zero-probe 是否自发逃逸；
- finite-amplitude ignition threshold `epsilon_c(z,x)`。

### 7.2 Hopf / limit cycle

要称 Hopf candidate，至少需要 frozen state 下：

- 高支失稳前后振幅从小到大连续出现；
- 主频非零且跨窗口稳定；
- 多个 initial condition 收敛到同一周期/振幅；
- slow variables frozen 后仍持续至少 10 个周期；
- 释放冻结后轨迹沿 X 方向离开该 orbit 并终止。

否则只称 oscillatory transient。SNN hybrid dynamics 不直接声称解析 Hopf；rate reduction/continuation 与 frozen-SNN empirical orbit 必须方向一致。

## 8. 关键 ablation

按最小顺序：

1. accepted partial-conductance Z-only anchor；
2. full conductance, X off；
3. full conductance + naive instantaneous/per-spike STD control；
4. full conductance + persistence-gated X；
5. candidate with X clamped at 1；
6. candidate with Z blocked；
7. candidate with spatial X snapshot shuffled across E cells；
8. optional phi off/on。

第 3 项只需一个 registered control，用来证明新结果不是普通 STD 重跑；不重新开放旧 M4-2 网格。

## 9. 已知漏洞与对应证伪

| 漏洞 | 为什么严重 | 本设计如何 fail-closed |
|---|---|---|
| high-state gate 可能只是把目标标签写回方程 | positive 只能证明一个抽象 persistence-selective brake 可工作，不能证明具体生理机制 | `y_gate` 只由 slow-off 定且一次锁定；结果只叫 phenomenological relay-fatigue screen |
| full E conductance 与 X 同时加入会无法归因 | fast branch 与 termination 可能来自不同改动 | Stage 0/1 先只做 conductance，finite high branch 通过后才实现动态 X |
| X 对首次进入新鲜组织的 wavefront 天生较弱 | wake 能压住重复接力，却不一定限制 first-pass extent | Stage 2 latency gate + Stage 4 front/wake + larger-L diagnostic；失败不缩短 sensor 抢救 |
| 局部 X 可能退化成一张近全局均匀场 | 时间生命周期可能成立但没有空间机制 | spatial-shuffle ablation、front–wake separation 和 wake-vs-ahead matched probe 是承重证据 |
| 短 tau_x 容易造 fragment train | 把多次小爆误写成 ictal oscillation | frozen fold recovery-path 预测 + frozen-slow 10-cycle orbit gate；不凭 rate trace 命名 limit cycle |
| optional phi 会成为第三个可调刹车 | 容易用复杂度掩盖前两阶段失败 | temporal+spatial lifecycle 未通过前 phi 不可开启；phi 只允许改变高态波形 |

## 10. 资源与产物

- L=20、40k SNN；Stage 0–3 单 worker，Stage 4 最多 2 workers；BLAS threads=1；
- 先跑 0.5–1 s 单 worker RSS smoke；20 s worker 的调度预算取 `max(12 GiB, 1.5 × measured_peak_extrapolation)`，启动前至少保留 `2 × worker_budget + 32 GiB` available；swap 增长即停止新 wave；
- network 每 seed 只 build 一次，checkpoint fork/COW；atomic per-cell JSON；支持 `--resume`；
- 不保存 full I raster；E raster 仅在 engine 仍强制需要时保留，结果写出前压缩为 online spatial bins + registered windows；
- 新结果根建议：`results/topic4_sef_hfo/mz_full_conductance_spatial_relay/`；figures 子目录必须有中文 README。

## 11. 停止条件与安全结论

任一条件触发即停止当前机制，不加新旋钮：

- full conductance 无法复现 slow-off returning-event workpoint；
- frozen fast map 没有有限 high branch/orbit；
- persistence sensor 不能分开普通 IED 与 sustained recruitment；
- persistence sensor 通过幅值分离但启动晚于 boundary recruitment；
- 三个 fold-derived `tau_x` 全落入 fragment/prevention/runaway；
- temporal lifecycle 成立但空间尺度仍随 L 增长或没有 front–wake；
- seed 1/3 方向相反。

设计阶段的安全预测是：**该结构有能力把 current model 的 terminal runaway 改造成 finite high excursion，并且比普通 STD 更可能保留 onset；但更可能先得到 fold-mediated、noise-triggered closed excursion，而不是干净的自主 Hopf limit cycle。真正的 high-state oscillation 若存在，预计需要在 finite high branch 上由快速 phi 产生，再由更慢的空间 X 终止。**

在 Stage 4/5 通过前，禁止写“已得到发作态、极限环、Hopf、双稳态或完整 seizure lifecycle”。
