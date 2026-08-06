# Topic 4：local divisive-recovery spatial lifecycle（superseded conditional design v0.2）

日期：2026-07-20

状态：**2026-07-20 已被 entry/exit nullcline 审计取代，不再作为主实现合同。** `D_R` 再次作用于 recurrent E gain，与既有 `S_G/T_G` fast topology 以及并行 E→E relay 线机制重叠；它仍可作为历史 counterfactual 保留，但不得继续实现或扫描。新的主合同见 `docs/superpowers/specs/2026-07-20-topic4-mz-persistence-gated-additive-spatial-lifecycle-design.md`。Stage 0F v1 已归档为 engineering-invalid，v1.1 权威运行仍固定为 `STAGE0F_NUMERICAL_UNRESOLVED`；本文以下内容仅作被取代方案的审计记录。

本线 worktree：`.worktrees/topic4-mz-divisive-lifecycle`

与并行 conductance 线的边界：本线只使用 current/rate moment closure、recurrent-E delayed divisor、local Z/p/r 和 finite-range field；不改 reversal potential、总膜电导、`tau_eff`、GABA conductance 或 presynaptic relay。

## 1. 目标与当前证据

最小目标不是先造一条永久 seizure limit cycle，而是在同一组参数、无 reset 的轨迹中实现：

> returning interictal events -> local low-state loss -> finite period-1 ictal orbit -> local cycle loss -> same-basin return -> early/late retrigger recovery。

Stage 0E 已证明 `z=0.85, alpha_G={15,16}` 存在约 605--609 ms 的闭合周期，并有强 finite-perturbation return。但当前根继续更支持下列 onset 预测：

- `z=0.88` 尚有 stable low root 和 intermediate unstable root；
- `z=0.87` 两者消失，只剩 unstable finite high root；
- Stage 0E 周期包围的是 unstable high focus。

因此当前首选图景是 **low-state saddle-node loss 后跳入已存在的大振幅周期**，而不是预设 supercritical Hopf 从 low state 平滑长出小振荡。周期在何处生成、退出是 Hopf、fold of cycles、SNIC 还是 homoclinic，必须由 continuation 判断。

一次不进 repo 的 frozen-gain cheap probe 只用来锁窄 Stage 1A 范围，不作正式结论：

- `w_ee_mult=1.10 -> 1.045` 时仍为有限振幅周期，周期约 `1.69 s`；
- `1.0446/1.0444` 时周期增至约 `2.27/3.93 s`，且 `dt/2` 一致；
- `<=1.0442` 时 exact-orbit fork 回到约 `2.49 Hz` low state；
- 等效只需 `D_R ~= 1.053`，即 recurrent-E 分数增益下降约 `5.1%`；
- 固定 `w_ee_mult=1.10`时，`z=.85/.86/.87` 仍为周期，`z=.88` 已回 low。

这预测一条具体的二维回程：`r` 先在 `z~.85` 暂时打断周期，再保持安静窗让 Z 至少恢复到 `z>=.88`，最后 `r` 才能退去。周期在边界附近出现 finite-amplitude/infinite-period slowing，但 `[1.0442,1.045]` 仍是 unresolved transition strip，不允许据此预先命名 SNIC 或 homoclinic。

## 2. 快变量：保留 Stage 0E 九维对象

局部快状态不重写：

\[
X=(e,i,s_{EE},s_{EI},s_{IE},s_{II},\bar e_D,\mu_D,S_D).
\]

四个突触滤波保持 Stage 0B/0C 合同；delayed divisor 保持：

\[
\tau_s\partial_t\bar e_D=-\bar e_D+K_{D0}*e,
\]

\[
\tau_\mu\partial_t\mu_D=-\mu_D+\Psi(\bar e_D),
\qquad
\tau_S\partial_tS_D=-S_D+S_{\max}\mu_D.
\]

0D 时 `K_{D0}*e=e`，必须 bitwise/容差内复刻 Stage 0F 的周期。

## 3. 主更新：recovery 改变 recurrent loop gain，不再加第三股慢电流

主 arm 让慢 recovery `r in [0,1]` 成为独立的分数增益因子，而不是加到快池的绝对分母上：

\[
D_G(x,t)=1+\alpha_GS_D(x,t),
\qquad
D_R(x,t)=1+\kappa_Rr(x,t),
\qquad
D_E=D_GD_R.
\]

\[
\mu_E=\tau_{mE}\left[
C_{EE}\frac{w_{EE}}{D_E}s_{EE}
-C_{EI}z(x,t)w_{EI}s_{EI}
\right]+\tau_{mE}J_{XE}\nu_{ext}+\mu_{core},
\]

\[
\sigma_E^2=\tau_{mE}\left[
C_{EE}\left(\frac{w_{EE}}{D_E}\right)^2s_{EE}
+C_{EI}[z(x,t)w_{EI}]^2s_{EI}
\right]+\tau_{mE}J_{XE}^2\nu_{ext}.
\]

它只改 recurrent E->E，不改 feed-forward drive、I population 或膜时间常数。乘法形式使 `r` 在整个周期中提供近似固定的分数增益下降，不会因 `S_D` 相位而改变实际效力。这个更新的动力学目的是让 slow recovery 移动快系统 Jacobian/cycle-loss surface，而不是在高态上抵消一个 signed current。第一个锁定范围覆盖完整 `D_R in [1,1.06]`；只有这一整段都找不到 `B_C` 才判 leverage no-go，不扩大网格。

预注册 matched comparator 仍保留加性路线：

\[
\mu_E\mapsto\mu_E-\eta_Rr.
\]

`eta_R` 只能在 Stage 0E 周期上按 phase-averaged recurrent-mean reduction 匹配，不能单独调优。cheap probe 中 `0.325 mV` 左右的常数偏置也能使轨迹突然回 low，但更弱偏置仍保留 `>100 Hz` 峰值，呈现的是较陡的 drive cliff。因此即使主 arm 更平滑，也只能比较 peak control、low-baseline preservation 与 cycle-loss geometry；如果两臂都终止，只能支持“慢 recovery 降低 E-loop gain/drive 可退出”，不支持 divisive specificity。

## 4. Z、p 与非对称 recovery kinetics

Z 保留 postsynaptic inhibitory-efficacy 语义：

\[
I_I^{raw}=\tau_{mE}C_{EI}w_{EI}s_{EI},
\qquad
\tau_z\partial_t z=z_\infty(I_I^{raw})-z,
\]

\[
z_\infty=1-S_\kappa(I_I^{raw}-I_z).
\]

不用平均 rate 直接驱动 persistence。Stage 0E 周期是短高峰，普通 IED 也可有高峰，因此 `p` 积分高活动占用：

\[
\tau_p\partial_t p=H_\kappa(e-e_p)-p.
\]

recovery 使用同一个状态、非对称 build/decay：

\[
\partial_t r=
k_\uparrow H_\kappa(p-p_r)(1-r)
-k_\downarrow[1-H_\kappa(p-p_r)]r.
\]

`p` 是 persistence sensor，`r` 是 recovery effector；相较单一 recovery 状态，这里明确增加两级状态，用于分离 sensing 和 asymmetric refractory kinetics。`build` 必须能在 seizure 占满空间前推过 cycle-loss，`decay` 又必须等 Z 恢复后才解除 refractory protection。

### 4.1 时间尺度锁

Stage 0E `T_C ~= 0.605 s`，current-stage 单次 returning IED 的活动宽度
`T_{IED,width} ~= 0.083 s`；这不是 inter-event interval。`tau_p` 不做宽网格，先锁：

\[
T_{IED,width}\ll\tau_p,\qquad 2T_C\le\tau_p\le5T_C.
\]

`e_p/p_r` 由锁定的完整 pre-onset history 和 Stage 0F 周期各自计算。定义：

\[
P_{pre}^{max}=
\sup_{h\in\mathcal H_{locked},\,t<t_{on}}p_h(t),
\qquad
P_{pre}^{max}+\delta_p<p_r<p_C^{min}-\delta_p.
\]

这个不等式是可识别性 gate，不是可调参愿望。`H_locked` 必须同时包括完整 repeated-IED replay、interval jitter、单事件和 low-root noise，不能把 `82.6 ms` 事件宽度误当成 IED 周期。若不存在同一 `e_p,tau_p,p_r` 的非空间隔，当前单一 occupancy sensor 结构 no-go，不用手挑事件救它。

recovery 至少允许 3--5 个周期先建立。参数锁定顺序不得交换：`e_p` 由 IED/cycle occupancy 分离锁，`tau_p` 由时标锁，`p_r` 从非空分离区间取，`k_up` 由到达正式 `B_C` 的时间锁，`k_down` 最后由 Z recovery 和 early/late retrigger 锁：

\[
t_{r,cross}-t_{on}\ge3T_C.
\]

offset 后的 no-reentry tube 不用模糊的 `B_C^-` 表示。定义 `D_guard(z)` 为“使锁定 early-retrigger probe 仍回 low 的最小恢复强度”，要求：

\[
D_R(t)\ge D_{guard}(z(t))+\delta_D
\quad\text{for }z<z_{safe},
\qquad
D_R\rightarrow1\text{ only after }z\ge z_{safe}+\delta_z.
\]

`z_safe` 必须同时满足：`D_R=1` 时无 stable ictal cycle，early-retrigger probe 回 low，且 exact low root 可连续回 `(z,D_R)=(1,1)`。cheap probe 的 `.88` 只是最低数值先验；Stage 1A 必须正式重新锁定。

## 5. Stage 1A：先解析四类几何对象

不先跑完整慢轨迹。固定 `alpha_G={15,16}`，对可识别的快系统坐标 `(z,D_R)` 做 root + periodic-orbit pseudo-arclength continuation；`kappa_R` 与 `r` 只在 Stage 1B 慢动力学中分解，不在拓扑层任意缩放。分开报告：

1. `B_L(z,D_R)=0`：low fixed point 失稳/消失线；
2. `B_C(z,D_R)=0`：periodic orbit 失稳/消失线；
3. `Sigma(z,D_R)=0`：low/cycle basin separatrix；
4. `D_sep(z)`：特定 state-fork 的操作边界。

`D_sep` 不能代替 `B_C`。参考 ictal state 被突然放到某个 `(z,D_R)` 后回 low，只说明该 probe 在 low basin；如果 stable cycle 仍存在，真实慢轨迹会继续绝热跟随 cycle。真正自终止必须穿过 `B_C`。所有 low-state fork 必须从同一 `(z,D_R)` 的 exact low root 启动；普通 `(e,i)=(0.5,2) Hz` 等任意低率初值会先产生强瞬变并可能跨 separatrix，不是 low-basin 证据。

### Stage 1A 验收

- `z=1,r=0` 保留 stable low；
- `z=0.85,r=0` 保留 Stage 0F 稳定周期；
- `z=.85` 的第一个 continuation 窄窗只覆盖 `D_R=1.00...1.06`，并包含 cheap probe 的 `[1.0526,1.0534]` 转换带；
- 有限 `D_R in (1,1.06]` 下周期失稳或消失，并有 stable low 可承接；
- 定位 onset/offset 类型，不预设 Hopf；
- 两种 `dt` 与 smooth/exact transfer 一致，全程无 `>=100 Hz`、support 或 natural-bound 违反。
- periodic branch 使用双向 pseudo-arclength + phase condition，同时保存 Floquet multipliers、period、振幅、closure 和邻近 saddle 特征值；至少继续到 `max(10*T_C0,8 s)`。达到 period cap 仍未闭合 global bifurcation 时判 `unresolved_long_period_boundary`，不写 cycle disappeared。

offset 类型按下列数值特征区分：

- homoclinic：`T ~ -c log|r-r_c|`，振幅保持有限；
- SNIC：`T ~ c/sqrt(|r-r_c|)`；
- supercritical Hopf：振幅趋零、频率保持有限；
- fold of cycles：有限振幅/频率的周期突然消失。

只有 low root 重现而 stable cycle 未消失，判 prevention/basin ambiguity，不开放 Stage 1B。

## 6. Stage 1B--1C：周期平均慢流与 0D lifecycle

### Stage 1B：不积分完整慢系统的预测

分别在 low branch 和 periodic orbit 上计算周期平均：

\[
\langle\dot z\rangle,
\quad
\langle\dot p\rangle,
\quad
\langle\dot r\rangle.
\]

慢向量必须：

- low/interictal 时 r 不 build；
- 进入 cycle 后先维持至少 3 周期，再r向 `B_C` 方向移动；
- offset 后先向高 Z 方向返回，在 `z>=z_safe`前 `D_R` 不得跨回 cycle side；随后 `r` 才衰减到 0。

沿整条 continued cycle branch 计算一周期慢映射：

\[
\Delta r_C(z,D_R;p,r)=r(t+T_C)-r(t).
\]

在 adiabatic-valid 区域内，向量与 `B_C` 相切、指向周期内部、`1+kappa_Rr_max` 达不到 `B_C`，或 `Delta r_C` 在 `B_C` 前变号，都直接判当前 occupancy-recovery 结构 no-go，不用时间常数网格补救。最后一种失败预测的是 permanent slow bursting：周期变长使 `p/r` 自行熄火，而不是 termination。

但 cheap probe 已显示越靠近退出带周期越长，因而周期平均不能一直外推到 `B_C`。以 `(z,D_R)` 的锁定无量纲尺度计算到 `B_C` 的 signed normal distance `d_C` 和慢向量法向速度 `v_n`，定义：

\[
\epsilon_{geom}=\frac{T_C|v_n|}{|d_C|},
\qquad
\epsilon_{time}=\max\!\left(
\frac{T_C}{\tau_z},\frac{T_C}{\tau_p},
k_\uparrow T_C,k_\downarrow T_C\right),
\qquad
\epsilon_{ad}=\max(\epsilon_{geom},\epsilon_{time}).
\]

- `epsilon_ad<=0.1`：允许使用周期平均慢流；
- `0.1<epsilon_ad<0.2`：周期平均只作 sensitivity；
- `epsilon_ad>=0.2` 或无法定义周期：进入 terminal layer，必须从最后一个通过点直接积分完整慢快方程；
- terminal layer 的 offset 需要 base/half `dt` 一致、回到 exact-low basin 且无 reset，不能用外推的 orbit average 代替。

### Stage 1C：一条完整慢快轨迹

reduced low fixed point 自身 inhibitory drive 很低，Z 会恢复到 1，不会无缘无故走向 onset。因此不人为加一个“静息时 Z 恒定下降”的 seizure clock。Stage 1C 使用 current-stage 中机器锁定的 returning-IED pulse train 作为受控 bridge：

- pulse 形状/间隔从已保存轨迹取值，不按结果调制；
- 单个 IED 不启动 r；
- repeated IED load 可通过 Z 把系统推过 `B_L`；
- cycle 建立 3--5 周期后 r 把它推过 `B_C`；
- 无 reset 回同一 low basin；
- early retrigger 失败，late retrigger 恢复。

这一层只支持 controlled event-load lifecycle。“自发 onset”必须在映射回保留 endogenous IED/noise 的完整 SNN 后另行验证。

## 7. Stage 2：从局部周期到 tissue-state front

### 7.1 空间 delayed-divisor field

不把 Stage 0E pool 保持为全局平均。小 core 在 global mean 中会被面积稀释，局部周期可能直接消失。主核改为固定物理宽度：

\[
K_D=(1-\gamma_D)G_{\sigma_E}+\gamma_DG_{\sigma_D},
\qquad
\sigma_E<\sigma_D\ll L.
\]

两个 Gaussian 都归一化，所以 homogeneous limit 对所有 `gamma_D` 完全复刻 0D。不先使用 `U=L^-1`的 uniform kernel；它会让同样大小的 core 在更大模拟域中反馈更弱，产生 domain-fraction 假 stall。

### 7.2 local recovery 的能力边界

local r 能形成 wake，但一般只生成恒宽 traveling pulse：

\[
t_{off}(x)\approx t_{on}(x)+T_{ictal},
\qquad
W_{recruited}\approx c_{front}T_{ictal}.
\]

因此 `gamma_D=0` 的预注册预测是 **wake PASS / containment likely FAIL**，不得因为图像有限宽就改判成 self-termination。

### 7.3 leading/trailing front 与 intrinsic annihilation 门

将历史招募范围和当前活动范围分开：

\[
A_{hist}(t)=|\{x:t_{on}(x)\le t\}|,
\qquad
A_{act}(t)=|\{x:t_{on}(x)\le t<t_{off}(x)\}|.
\]

`c_on>=0` 是 observed recruitment-front speed，`c_off>=0` 是 observed recovery-front speed，两者都是历史速度，不允许写成负值。可以正负变号的对象是 clamped feedback 下的 frozen signed invasion velocity `c_inv(Q)`。

宽程分母的 zero-speed surface 必须写成：

\[
c_{inv}(Q)<0, =0, >0
\]

并且 dynamic trajectory 在到边界前满足 `c_on->0`、`c_off>0`、`A_act->0`，同时 `A_hist` 达到有限平台，才称 contained self-termination。分类固定为：

- `c_on~=c_off>0`：constant-width traveling pulse；
- `c_on=0,c_off=0,A_act>0`：stationary persistent bump；
- `c_on->0,c_off>0,A_act->0`：contained self-termination；
- 从未出现 `t_on`：prevention。

在启动 finite-range 动力学扫描前，先计算 edge-load curve：

\[
Q_{edge}(W)=(K_D*\mathbf 1_{[0,W]})(x_{edge}).
\]

如果 `Q_edge` 已饱和而 frozen `c_inv` 仍为正，finite-range arm 直接判 no extent brake，不扩大空间模拟。

`gamma_D=0` 只用于验证 local nucleation/front/wake classifier。只在它产生 traveling pulse 时，才开放一个小型 `gamma_D x sigma_D` 有界 screen，寻找上述 zero-speed/edge-collision surface。若无有限窗口，本线 spatial containment no-go；不加 global GABA 与 conductance 线重复。

### 7.4 conditional Stage 2C：连续 activity-load 低秩 divisor

归一化 finite-range kernel 在 active width 超过 `sigma_D` 后会饱和；它可以改变 front，但不是 Abbott/Liou 那种按已招募范围继续增强的 spatial integrator。因此只在 Stage 2B 给出干净 traveling-pulse no-go 时，条件开放一个仍属本 current/rate 线的 rank-one recurrent-E gain arm：

\[
Q_A(t)=\frac{1}{A_{ref}}
\int_\Omega P_\epsilon\!\left[S_D(x,t)-S_{D,rest}-\delta_D\right]dx,
\]

\[
D_A(t)=1+\kappa_AQ_A(t),
\qquad
D_E=D_GD_RD_A.
\]

`P_epsilon` 是平滑正部；`S_D_rest/delta_D` 由 low/IED 负对照预锁，不按结果调。这里反馈的是已有连续快池 `S_D`，不是分类后的 recruited area；不新增时间常数。`A_ref` 用固定物理 core/核宽预锁，积分含 `dx`、不除以 domain size，因而增加 inactive padding 不得改变 `Q_A`。

Stage 2C 是新动力学节点，不可强称 high homogeneous parity。它的 fail-closed 门是：

1. `kappa_A=0` 与 low state 精确复刻 Stage 1/2B；
2. 对 clamped `Q_A` 重做 `B_L/B_C` atlas；
3. 初始 local core 产生的 `Q_A` 不得消灭局部周期，否则是 prevention；
4. clamped `Q_A` 的 `c_inv(Q_A)` 必须真正穿零；dynamic `Q_A` 上升领先于 `c_on` 衰减，且 `c_off>c_on` 使两前缘相遇；
5. domain doubling/buffer extension 不改变临界物理宽度。

它不是 global GABA conductance，不改 reversal potential 或 `tau_eff`，因而与并行 conductance 线仍是独立机制检验。

### 7.5 tissue-state classifier

front 不用首次 spike 或 50-ms activity crossing 定义。空间点的未来依赖 `K_D*e` 与邻域状态，因而只冻结局部 `(z,D_R)` 的 0D fork 不是 ground truth。正式 classifier 必须：

1. 冻结完整 `z(x),D_R(x)` 慢场，保留完整 fast spatial coupling；
2. 从实际状态和 low-reference 两套初值积分全场，或至少宽度 `>3*sigma_D` 的窗口；
3. low/ictal membership 由这一 spatial frozen-slow fork 判定；
4. persistent switch 需 ictal membership 持续至少 2--3 个局部周期，onset/offset 使用分开 hysteresis。

在大仿真中可以用索引 `(z,D_R,S_D,K_D*e)` 的 local atlas 作在线 surrogate，但必须对分层抽样 `(x,t)` 做真实 full-field frozen fork 复核。

### 7.6 Z/penumbra 边界

当前九维 current/rate 模型只有一个 `s_EI`，无法不改快维度地实现“Z 只作用 local GABA、broad GABA 受保护”。因此这不是本线 Stage 2 必做 arm；它正式移交并行 conductance 线。本线只报告 raw inhibitory barrage、Z 与 front-ahead activity 的相序，不拆 local/global GABA 方程。

## 8. 预注册失败模式

- `p_r` 太低或 `k_up` 太大：r 在 onset 前上升，prevention；
- `1+kappa_R r_max` 达不到 `B_C` 或慢向量与 `B_C` 相切：permanent oscillation；
- `tau_p<T_C`：每个 burst 都拉高 r，chattering/假 clonic；
- `k_down` 太大：Z 仍低时 r 已消失，immediate retrigger；
- low branch 上无 endogenous IED/noise：Z 回到 1，reduced system 永不 onset；
- 静息时人为让 Z 恒定下降：得到漂亮但机械的 seizure clock；
- global mean divisor：local core 被面积稀释，周期消失或重新 runaway；
- local r 无 broad brake：恒宽 pulse 撞边界；
- uniform normalized global pool：stall 位置随 domain length 缩放；
- `D_r>0` 太大：recovery 跑到 front 前方，wake 变 prevention。主版锁 `D_r=0`；如开放，必须 `sqrt(D_r/k_down)<<sigma_E`。

## 9. Stage 2 验收

1. homogeneous field 复刻 Stage 1 的 low/cycle/exit boundaries；
2. local nucleation，非全轴近同时点燃；
3. tissue-state front 与 fast event wave 分离，`c_tissue/c_event << 1`；
4. r 在 local recruitment 后滞后上升，形成 refractory wake；
5. `gamma_D=0` 的 traveling-pulse negative 不改判；
6. broad finite `K_D` 若通过，必须显示 clamped `c_inv` 穿零、dynamic `c_on->0`、`c_off>0` 且 `A_act->0`；
7. domain doubling、buffer extension 和 boundary-condition sensitivity 下 `A_hist` 有限平台/物理位置基本不变；
8. front 在到达边界前 stall/回缩，全域无 reset 回 low；
9. local/global GABA-Z sensitivity 移交 conductance 线，本线不为了该对照静默增加快维度；
10. 所有空间 claim 在 full-field frozen-slow membership 层成立，不只是 rate/LFP 图形。

## 10. 映射回完整 SNN

只有 Stage 2 通过才开放。不建 all-to-all 新状态；使用固定 2D coarse field（例如 24x24）：

1. 每 1 ms 从 E spikes 构建局部 rate field；
2. 在 coarse grid 上更新 `bar_e_D/mu_D/S_D/p/r`、做 finite-range convolution；
3. 按神经元位置插值 `D_E(x_i,t)`；
4. 只对已分离的 recurrent E current 作除法，feed-forward/external E 不变；
5. Z 保持逐神经元/局部抑制耗竭语义。

映射前必做三个 parity：

- homogeneous coarse field 复刻 reduced 0D 周期频率/波形；
- local patch 不因全网面积稀释而丢失 divisor；
- matched additive comparator 保持相同平均刹车强度。

完整 SNN 验收仍包括 slow-off returning events、无 kick 自发 onset、same-basin recovery、early/late retrigger、局部 persistent front/wake/stall，以及 P99 高率尾部和 moment envelope 约束。

## 11. 计算资源合同

- Stage 1A/1B：单点单线程，最多 8--12 个并行 continuation shards，每个 RSS < 2 GiB；
- Stage 1C：先 2 个工作点 x 3 个 event-load histories，不开大网格；
- Stage 2：先 1D 128/256 点、单线程 BLAS，再并行 8 个以下物理参数 shards；
- 始终保留至少 25% 主机内存和 4 个 CPU cores；
- Stage 2 通过前不启动 40k SNN，不写完整 spike bool，仅保存派生 field/抽样 state forks。

## 12. 会合语言

本线只能主张：

> delayed recurrent-E gain control 可以创建有限快周期；local persistent gain recovery 是否能使慢轨迹穿过真正 cycle-loss surface，finite-range divisor field 是否能产生 domain-invariant zero-speed front，由 Stage 1/2 判定。

并行 conductance 线只能主张：

> reversal-aware membrane conductance、local/global GABA、sAHP 和 relay 是否能在完整 spiking network 中独立产生同类 branch/cycle/front/return 拓扑。

两线只比较输出对象：low/cycle boundaries、onset/offset type、front speed、wake、zero-speed surface 与 retrigger。各自 gate 通过前不拼方程。
