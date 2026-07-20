# Topic 4：topology-first spatial slow–fast field 路线（review-ready v0.6）

> 日期：2026-07-20
> 状态：**review-ready v0.6；Stage 0B 已触发 fast-equation stop rule；Stage 0C 打开 dynamic-divisor 振荡线索；Stage 0D v1.1 未通过窗口式 open-basin 门；Stage 0E 已直接解出两条闭合 period-1 orbit 和强有限扰动回归，但 Floquet 导数平台未解析；仅 Stage 0F 数值证书开放，slow/field 仍关闭**
> 本路线 owner：`.worktrees/topic4-mz-divisive-lifecycle`
> 与并行 FCXR 的边界：本路线只做 reduced rate/field 的动力学拓扑与空间前沿；不改 LIF 膜电导，不实现 E→E presynaptic relay `x_j`，不编辑 `.worktrees/topic4-mz-conductance`。

## 1. 为什么另开这一条线

当前 current-based Z–`T_G`–M 结果已经说明：

- Z 能逐步提高事件幅度/占空比并跨过 recruited-state 操作阈值；
- 高状态选择性的 `T_G` 能把 delayed runaway 改成有限窗 bursting；
- 固定慢状态后，原始六变量块只有 low/saturation cliff；加入 delayed mean divisive pool 后已解出有限闭合周期，但其导数级稳定性证书及 entry/exit 拓扑仍未闭合；
- `T_G` 是全场标量，线性 M 从 `t=0` 起作用，结果是改变进入或 prevention，不是建立高态后再退出；
- 当前空间结构主要来自固定连接底物，而不是一个能形成 front、penumbra 与 refractory wake 的动态空间状态。

因此，本路线不再盲扫第三个全局慢刹车。它先在低成本 reduced model 中回答一个更基础的问题：

> 能否先构造并定位“低固定点 + 有限 ictal oscillation/branch + entry boundary + exit boundary”，再把这些局部动力学嵌入患者特异的空间连接，使发作以局部前沿招募、在已招募区留下恢复尾迹并自行退出？

这与 FCXR 的 bottom-up 问题互补：FCXR 问 full-conductance SNN 是否自然开出该拓扑；本路线从 top-down 的最小动力学对象出发，先证明什么拓扑和空间反馈是必要的。

## 2. 设计原则

1. **fast object first**：slow variables 关闭/冻结时，先找到明确的 low fixed point 与 finite high branch/orbit。没有这个对象就不允许靠 slow feedback 伪造“bounded”。
2. **entry、maintenance、exit 分开**：Z 负责靠近/跨入，fast E–I subsystem 负责 ictal 段，局部 recovery field 负责跨出。
3. **空间变量必须是场，不是全局标量**：局部招募历史必须保留下来，才能定义 wavefront 前方、已招募核心和 wavefront 后方的 refractory wake。
4. **患者 scaffold 只做异质耦合，不替代动力学**：先在 homogeneous 0D/1D 找到拓扑，再加入 E1146 长轴 `W_AB`；不能靠固定异质性把全场同步误写成传播。
5. **cheap first**：0D continuation → 1D front → 2D E1146 field；只有三层都通过，才讨论映射回 40k SNN。

## 3. 主方程：spatial E–I fast field + local Z/recovery loop

第一版用可继续分析的 E–I rate field，而不是再改 LIF 电流。fast block 不另造任意 sigmoid；
复用 `src/topic4_m3b_spectral_phase.py` 已有的 LIF transfer、E/I rate 与
`sEE/sEI/sIE/sII` 四个 synaptic filters。完整输入合同锁为：

\[
\tau_A\partial_t s_{EE}=K_E*e-s_{EE},\qquad
\tau_G\partial_t s_{EI}=K_I*i-s_{EI},
\]

\[
\tau_A\partial_t s_{IE}=K_{IE}*e-s_{IE},\qquad
\tau_G\partial_t s_{II}=K_{II}*i-s_{II}.
\]

Z 延续当前 SNN 的 **postsynaptic inhibitory-efficacy** 语义，而不是改成 presynaptic resource：

\[
\mu_E=\tau_{mE}\!\left(C_{EE}w_{EE}s_{EE}
-C_{EI}\,z(x,t)w_{EI}s_{EI}\right)
+\tau_{mE}J_{XE}\nu_{ext}+\mu_{core}-r(x,t),
\]

\[
\sigma_E^2=\tau_{mE}\!\left(C_{EE}w_{EE}^2s_{EE}
+C_{EI}[z(x,t)w_{EI}]^2s_{EI}\right)
+\tau_{mE}J_{XE}^2\nu_{ext},
\]

\[
\mu_I=\tau_{mI}(C_{IE}w_{IE}s_{IE}-C_{II}w_{II}s_{II})
+\tau_{mI}J_{XI}\nu_{ext},
\]

\[
\sigma_I^2=\tau_{mI}(C_{IE}w_{IE}^2s_{IE}+C_{II}w_{II}^2s_{II})
+\tau_{mI}J_{XI}^2\nu_{ext},
\]

\[
\tau_E\partial_t e=-e+\Phi_E(\mu_E,\sigma_E;V_{th,E}+\phi),\qquad
\tau_I\partial_t i=-i+\Phi_I(\mu_I,\sigma_I;V_{th,I}).
\]

其中：

- `e/i` 与四个 synaptic filters 是快变量；`Phi_E/Phi_I` 使用既有、有界且单位已审计的 LIF
  population transfer，输出有明确 rate ceiling；
- `K_E` 较窄且可沿 E1146 长轴各向异性；`K_I` 较宽，负责前沿外的 feedforward-inhibition penumbra；
- `W_AB` 是患者特异固定 scaffold，只在 homogeneous/1D gate 通过并锁定 producer、单位、归一化与
  subject fingerprint 后加入；
- Z 在抑制均值中乘一次、在抑制方差中平方一次；禁止写成 `K_I*(z*i)`，因为那会静默改成
  presynaptic resource；
- recovery `r(x,t)` 单位为 mV，只从 `mu_E` 减一次；dynamic threshold `phi` 只进入
  `V_th,E`，禁止把 `r` 同时写进 threshold。`r` 不是 FCXR 的 presynaptic E->E resource `x_j`。

Z 的 sensor 也延续当前 SNN：它读取 **未被 Z 缩放前**的 postsynaptic inhibitory drive：

\[
I_I^{raw}(x,t)=\tau_{mE}C_{EI}w_{EI}s_{EI}(x,t),
\]

\[
\tau_z\partial_t z=z_\infty(I_I^{raw})-z,\qquad
z_\infty=1-S_\epsilon(I_I^{raw}-I_{th}),\qquad 0\le z\le1.
\]

`S_epsilon` 是为 continuation 使用的平滑 Heaviside；`epsilon -> 0` 必须作为 current-SNN strict-threshold
sensitivity。recovery loop 写成：

\[
\tau_p\partial_t p=-p+e,
\]

\[
\tau_r\partial_t r=-r+r_{\max}
\frac{[p-\vartheta_r]_+^{n_r}}
{p_{50}^{n_r}+[p-\vartheta_r]_+^{n_r}}
+D_r\nabla^2r.
\]

解释：

- 普通短 IED 可消耗少量 Z，但 `p` 达不到 `vartheta_r`，所以 recovery gate 不启动；
- 持续局部 recruitment 使 `p` 越阈，`r` 在该处累积并把 E-nullcline 推向退出边界；
- `r` 在 wavefront 后方保留有限时间，形成 refractory wake，使刚退出的组织不会立刻被低 Z 再点燃；
- `D_r=0` 是主版本起点，弱扩散只作 sensitivity。第一版不引入第三个全局 slow scalar。

若 frozen fast block 只得到 bounded tonic high branch、没有可读的 ictal oscillation，则只开放一个
预注册扩展：加入 80--150 ms 量级的 dynamic threshold `phi(x,t)`，使其只承担 within-bout rhythm：

\[
\tau_\phi\partial_t\phi=-\phi+\Delta_\phi e,
\qquad \Phi_E(\cdot;V_{th,E})\rightarrow\Phi_E(\cdot;V_{th,E}+\phi).
\]

`phi` 不承担秒级最终 termination；`r` 不承担 onset 前的一般 IED 抑制。两者不能合成一个从
`t=0` 起工作的 M。

### 3.1 两个明确的非候选对照

- **current-control**：把 `r` 改成 uniform additive subtraction；用于复刻当前失败，不能成为主模型。
- **global-control**：把 `r(x,t)` 替换为全场均值；用于检验“有界率”是否以丢失 front/wake 为代价。

## 4. 关键差别不在“additive / non-additive”标签

必须明确：`r` 从 `mu_E` 中减去，在代数上仍是 activation input 中的 additive
subtraction。把它叫 threshold、current 或 control parameter，本身都不会自动产生 seizure dynamics。
current-based 系统只要 fast phase portrait 合适，同样可以有 bistability/Hopf/limit cycle；conductance
也不自动保证这些对象存在。

当前模型固定 Z/`T_G`/M 后，fast subsystem 基本仍是原 current-LIF，只是输入偏置或 recurrent gain 改了；有限窗 bursting 可以来自慢变量的瞬时抵消，而不要求存在新的 fast attractor。

新路线真正新增的是三个合同，而不是一种神奇的代数形式：

1. fast E–I block 必须先独立通过 fixed-point/branch/orbit/state-fork；
2. recovery 必须由 established local recruitment 持续门控，不能从第一颗 IED spike 就工作；
3. recovery 必须是局部场，在 front 后方保存空间历史。

目标不是“多加一条负电流”，而是先锁定 fast phase portrait，再让慢控制参数按可验证次序跨越它：

\[
\text{low fixed point}
\xrightarrow[Z\downarrow]{entry}
\text{finite oscillatory branch}
\xrightarrow[r\uparrow]{exit}
\text{low fixed point}.
\]

空间卷积随后用于检验局部 branch switching 能否形成 moving front；局部 `r` 用于检验招募历史能否形成
wake。两者都不是预设成功。没有这些局部状态与对照，固定 scaffold 只能决定哪里更容易点火，不能自动提供传播和退出。

### 4.1 慢路径必须在 `(z,r)` 平面闭环

先用 frozen state forks 在 `(z,r)` 平面求出：

- low/high basin 与 onset/offset surfaces；
- `r_sep(z)`：给定 Z 时，使注册的 high-state initial condition 回到 low basin 所需的最小 recovery；
- `z_safe`：在 `r=0` 时，该 high-state probe 也只能回到 low basin 的最小恢复 Z。

真实慢轨迹必须满足：

\[
\text{cross onset surface}\rightarrow\text{establish high state}
\rightarrow\text{cross offset surface}\rightarrow\text{return low},
\]

并在 offset 后、Z 尚未恢复到 `z_safe` 的整个窗口内满足：

\[
r(t)-r_{sep}(z(t))\ge \delta_r>0.
\]

这条 safety margin 防止 `r` 先衰减、低 Z 立即重新点燃。`delta_r`、early probe 与 late probe
必须在看结果前锁定；只看一条 rate trace 不算闭环证明。

## 5. 分阶段执行门

### Stage 0A：0D topology oracle（不作机制 claim）

先用同一分析代码跑一个已知具有 excitable fixed point / oscillatory branch 的二变量 canonical normal form，验证：

- continuation/state-fork 能正确找到 entry/exit；
- classifier 不会把 rate ceiling 或长 transient 当 limit cycle；
- onset/offset、retrigger 和 slow-loop 指标实现正确。

这是分析链 sanity control，不进论文主张。

### Stage 0B：0D E–I fast topology

冻结 `z,r`，不加 noise、空间和患者 scaffold。先复用 M3B 的六变量
`rE/rI/sEE/sEI/sIE/sII` homogeneous operator；只有 bounded tonic branch 而无节律时，才启用上面唯一
预注册的 `phi` arm：

1. continuation 或密集 initial-condition state fork；
2. 找 low fixed point；
3. 找 finite high fixed point 或 bounded oscillatory orbit；
4. 定位 entry/exit boundary；
5. 排除 refractory ceiling、数值 clipping 与只有一条饱和支的情况。

Stage 0B 只扫两个轴：固定 `ratio=1.0`，`w_ee_mult={1.0,1.1,...,1.5}`，
`q_I={1.00,0.99,...,0.80}`；`q_I` 只缩放 I->E，I->I 不变。工作点和大振幅轨迹必须在同一六变量
RHS 中自洽更新 `mu/sigma`，禁止用只适合局部线性的 frozen-sigma `field_rhs` 做 state fork。

先做 root/bidirectional continuation；只在多根区与边界做 low/intermediate/high initial-condition fork。
初筛 `dt=0.25 ms, T=6 s`；只有 `<100 Hz` 且无 LUT clipping/ceiling occupancy 的 bounded high candidate
才用 `dt=0.125 ms, T=12 s` 和至少两个 high initial conditions 确认。`phi` 不能成为第三条扫描轴；只有
无 `phi` 时先找到 `<100 Hz` bounded tonic high branch，才开放固定 `tau_phi=100 ms` 的三个
`Delta_phi` 值。

**停止规则**：若只得到 silent/prevention 与 `rE>100 Hz` saturation cliff、只有 unstable middle root、
或“周期”随记录时长漂移，则 clean no-go；不接 slow loop，不进空间层。

### Stage 0C：0D augmented fast divisive topology

Stage 0B 的原始六变量 E/I fast equation 若触发上述 stop rule，不回去扩大 Z/M 网格，而只开放一个沿既有
M4 证据的最小 fast-topology surgery。把 M4 的完整 `rE_fast -> mu_G -> S_G` 回路并入 fast block，状态为：

\[
X=(e,i,s_{EE},s_{EI},s_{IE},s_{II},\bar e_G,\mu_G,S_G),
\qquad D_G=1+\alpha_GS_G.
\]

只对 recurrent E->E 输入作除法；外部 E drive、I->E 抑制与 I population 不动。reduced moment closure
按 realized recurrent current 被缩放的语义，同时缩放均值和方差：

\[
\mu_E=\tau_{mE}\left[
C_{EE}\frac{w_{EE}w_{mult}}{D_G}s_{EE}
-C_{EI}z w_{EI}s_{EI}\right]+\tau_{mE}J_{XE}\nu_{ext}-r,
\]

\[
\sigma_E^2=\tau_{mE}\left[
C_{EE}\left(\frac{w_{EE}w_{mult}}{D_G}\right)^2s_{EE}
+C_{EI}(z w_{EI})^2s_{EI}\right]+\tau_{mE}J_{XE}^2\nu_{ext}.
\]

快池为：

\[
\tau_s\dot{\bar e}_G=-\bar e_G+e,
\qquad
\Psi_G(\bar e_G)=
\frac{[\bar e_G-e_0]_+^2}{e_{50}^2+[\bar e_G-e_0]_+^2},
\]

\[
\tau_\mu\dot\mu_G=-\mu_G+\Psi_G(\bar e_G),
\qquad
\tau_S\dot S_G=-S_G+S_{max}\mu_G.
\]

锁定 `tau_s=15 ms, tau_mu=30 ms, tau_S=80 ms, S_max=1, e0=0.005 kHz,
e50=0.015 kHz, n_Psi=2, beta_SG=0`。`mu_G/S_G` 的连续 ODE 不用数值 clip 制造假分支。
SNN 的 `r50_psi=0.4` 输入是每步网格 spike-field 量，不是 kHz；禁止直接搬入 reduced equation。
0D homogeneous 时 p-norm 满足 `A_G=Psi_G`，所以 `p_pool` 严格消掉，不能成为 Stage 0C 扫描轴。

固定 `ratio=1, w_ee_mult=1.1, r=0`，只扫：

- `z={1.00,0.99,...,0.80}`；
- `alpha_G={0,1,2,4,8,12,16,24,32}`。

`w_ee_mult=1.1` 是 Stage 0B 预先看到 low/separator/saturation 重叠最宽的诊断截面，不按 Stage 0C
结果挑选。`alpha_G=0` 必须复刻 Stage 0B；固定点可降成二维求根，但稳定性必须用完整 9x9 Jacobian。
state forks 同时覆盖平衡流形和 AMPA/GABA phase-mismatched 6D 初态。初筛/确认仍分别为
`dt=0.25 ms,T=6 s` 与 `dt=0.125 ms,T=12 s`。

若出现 `<100 Hz` 候选，必做 `full dynamic / instantaneous / matched-clamped /
matched-subtractive / mean-only-divisor` 对照。只有 full dynamic 有有限 orbit 才支持 delayed-feedback
解释；三种 pool 都有同一 tonic state 只支持 nonlinear gain compression；静态池只压回 low 是 prevention。

**Stage 0C pass** 至少要求：`z=1` 保留 `<5 Hz` stable low；有限 stable high root/orbit 跨相邻两个格、
由至少两个非 exact-root 初态进入；tail mean/peak 均 `<100 Hz`，无 LUT clipping、ceiling 或 drift；
`alpha_G=0` 同点没有该对象。通过后才沿 `z down` 定位 entry，并以冻结 `r in (0,7) mV` 定位 exit。
只有 exact separator、initialized-high pool、prevention、`>100 Hz` 或 window-dependent transient 均 no-go。

这个 pool 的安全名称是 **activity-dependent recurrent-E divisive normalization** 或 M4 rank-one fast
gain-control loop。它不是 GABA conductance，不改变 reversal potential、总膜电导或 `tau_eff`，也没有
presynaptic relay；因此不能称 Abbott/Liou model replication。它只负责把 fast high object 变有限，不能替代
local `r(x,t)` 的秒级 termination 与 refractory wake。

### Stage 0D--0F：周期复现与稳定性证书

Stage 0D 使用固定尾窗 FFT/振幅分类器做 post-discovery prospective replication；它已完成，但
180 条历史仅 5 条在同一 phase/邻点通过，因而没有通过预注册 open-basin 门。该失败受
尾窗相位和 spectral-ratio 门影响，只能写成 `NO_REPLICATION_WITH_UNRESOLVED_TRAJECTORIES`，不能写成
不存在周期。

Stage 0E 固定 `z=0.85, alpha_G={15,16}`、`S_G=0.15` 向上 Poincare 截面，不再使用
FFT。其 shooting、base/half-`dt` 波形一致性和 4 phases x (2 fast + 2 pool directions) x 8
returns 已支持两条闭合且对有限扰动强收缩的 period-1 orbit。但 bilinear-LUT Poincare
Jacobian 的三档 epsilon 之间没有形成相对差平台，所以按锁定门仍是
`STAGE0E_NUMERICAL_UNRESOLVED`，不能称 stable Floquet attractor。

Stage 0F 是当前唯一开放节点，只解析上述导数级不确定性：

1. 参数点、Poincare 截面、base/half `dt` 和 Stage 0E orbit 全部固定；
2. 用 smooth/exact transfer 加 discrete variational/event projection，或与数值积分器一致的 tangent map；
3. 若使用 orbit-local differentiable surrogate，必须先通过 exact transfer value/derivative 误差门与 orbit parity；
4. 同时报告 absolute operator error、base/half-`dt` 一致性和与 Stage 0E nonlinear return battery 的一致性；
5. 不缩小 epsilon 到机器底噪进行事后救援，不用 FFT 替代 Floquet，不扫新生物参数。

Stage 0F 只有在两种步长和两种导数构造中均给出与 nonlinear battery 一致、且对单位圆有预注册余量的稳定半径，才允许进入冻结 `r` continuation。若得到可靠 `rho>=1`，关闭 dynamic-divisor 节点；若仍不可辨，停在 numerical unresolved，不加慢变量“救”它。

### Stage 1：0D closed slow loop

只在 Stage 0F 完成稳定性证书，且冻结 continuation 找到 low/cycle 与 `r_sep(z)` 边界后打开 Z、`p/r`：

- 短事件能返回且不启动 `r`；
- Z 漂移可使系统自主或由有限短刺激进入 fast high branch；
- `r` 必须在 established recruitment 后上升，并使 frozen fast system 穿过 exit；
- 无 reset 回到同一 low basin；
- offset 后到 `z>=z_safe` 期间始终满足 `r-r_sep(z)>=delta_r`；
- early retrigger 被抑制、late retrigger 恢复。

必须做 `r free / r=0 / matched-clamped r` state fork，不能只比较两条从 `t=0` 就不同的轨迹。

### Stage 2：1D source→sink field

只沿 E1146 注册长轴布点，加入 `K_E/K_I`，暂不加完整二维异质性。验收：

- 局部 onset，不是全轴同一时间跨阈；
- local ictal state 以冻结 branch/orbit membership 和至少 2--3 个周期的 persistent dwell 定义，并使用分开的 onset/offset hysteresis；
- front position 取每个位置首次 persistent state switch，禁止用首次 spike 或 50-ms activity crossing 代替；
- 有可测 front position、front velocity 和 recruited width；
- front 前有高 synaptic/inhibitory drive、低 local E firing 的 penumbra；
- front 后 `r` 滞后于 recruitment，形成 wake；
- larger-L 后速度/宽度不随边界伪变。

局部 wake **不等于** containment。Stage 2 必须把结果分成：

1. `intrinsic_stall_or_annihilation`：front 在到达边界前停下/相消，并全轴返回，才允许继续；
2. `constant_speed_traveling_pulse`：wake 成立但 front 一直跑到边界，只能判 wake PASS、containment FAIL；
3. `near_synchronous_global_ignition`：空间 gate FAIL。

只有第 2 类才开放一个预注册的 reduced-field sensitivity：

\[
K_I=(1-\gamma)G_{\sigma_I}+\gamma U,
\]

其中 `U` 是归一化均匀核，表示 fast extent-dependent nonlocal inhibition，不是第三个 global slow
scalar。若不存在同时避开 prevention、全场同步和 boundary termination 的有限 `gamma` 窗口，则停止，不能用患者
异质性或边界制造假 stall。

### Stage 3：2D E1146 scaffold

Stage 3 启动前先锁定 `W_AB` 的 canonical producer/artifact、subject fingerprint、矩阵方向、单位、
稀疏性与归一化。`W_AB` 和 `K_E` 必须用显式 blend（例如 row-normalized 后由唯一 `kappa_W` 混合），禁止两次
计算同一 scaffold coupling；`kappa_W=0` 是必做对照。该数据合同当前尚未锁定，因此 Stage 3 仍有一个明确 P0。

随后固定 Stage 1/2 的 fast/slow 参数，只允许一个总耦合尺度校准，再加入：

- 各向异性 `K_E`；
- 较宽 `K_I`；
- 患者特异 `W_AB` 与双 source heterogeneity；
- 正式 virtual-SEEG readout。

目标是同一轨迹中出现：returning interictal event → 局部 ictal recruitment → patterned spread/stall → spatially resolved termination → recovered interictal event。不得按图形好看逐患者重调参数。

### Stage 4：与 SNN 两线会合

只有 reduced field 和 FCXR 分别过各自 gate 后，才比较：

- 哪个 fast topology、entry/exit surface 和 spatial wake observable 一致；
- 哪些 reduced parameters 可映射为 SNN conductance、threshold 或 synaptic resource；
- 是否值得合并。未通过前不拼方程。

## 6. 预注册指标

### 时间动力学

- frozen low/high branch existence 与 basin；
- finite high-state mean rate、modulation、dominant frequency；
- entry/exit boundary 与 slow trajectory crossing order；
- post-offset return distance to same low fixed point；
- early/late retrigger ratio；
- state-fork causal verdict。

### 空间动力学

- nucleation area fraction；
- front position/velocity/width；
- recruitment-time vs source-distance monotonicity；
- penumbra gap：input/inhibitory barrage ahead of front minus local firing；
- wake lag：`t_peak(r)-t_recruit`；
- stall/annihilation location；
- spatial offset dispersion；
- recovered event template similarity。

### 数值与边界检查

- dt/space-grid convergence；
- larger-domain control；
- periodic vs no-flux boundary sensitivity；
- rate ceiling occupancy；
- seed/noise sensitivity；
- homogeneous/isotropic/shuffled-scaffold controls。

## 7. 必做 ablation

- `z off`：应失去慢性进入或 recruitment susceptibility；
- `r off`：应失去 exit/refractory wake；
- `p gate off`（令所有活动驱动 r）：应退化为 IED prevention；
- `K_I` 不比 `K_E` 宽：检验 penumbra/containment 是否消失；
- global `r`：检验是否只剩全场同步退出；
- `W_AB` shuffle/isotropic：区分动态前沿与固定 scaffold 继承；
- matched-clamped `r`：区分动态 timing 与相同 DC 抑制量。

## 8. 资源策略

- Stage 0A/0B/0C：CPU 1–4，内存 <4 GiB；先 continuation/state forks，不上大网格。
- Stage 1：每组轨迹独立并行，但总内存保留至少 96 GiB。
- Stage 2：1D field 优先，最多使用可用 CPU 的 1/2；每个进程先做 RSS smoke。
- Stage 3：二维域先 1 seed、短窗，再跨 seed；不得与 40k SNN 同时按峰值内存满配。
- 任一 stage 的科学 gate 失败，停止下游，不用更多并行掩盖结构性 no-go。

## 9. 文献定位与边界

- Jirsa et al. 的 Epileptor 强调 fast/intermediate objects 与 slow permittivity 的分时标结构，并把 onset/offset 定义为不同动力学边界：[Brain 2014](https://academic.oup.com/brain/article/137/8/2210/2847958)。
- Proix et al. 把局部 Epileptor 扩展为空间 field，区分慢 ictal wavefront 与快 SWD，并展示空间传播/终止来自 excitable media、coupled oscillators 与多时标相互作用：[Nature Communications 2018](https://www.nature.com/articles/s41467-018-02973-y)。
- Schevon et al. 的人类记录支持 ictal core、wavefront 与被强突触输入影响但未局部招募的 inhibitory penumbra 分离：[Nature Communications 2012](https://www.nature.com/articles/ncomms2056)。
- Liou et al. 的 biophysical network 用 reversal-aware conductances、局部与广域 inhibition、sAHP 等解释 recruitment/termination；它支持并行 FCXR 的膜方程方向，但不等于本 reduced-field 路线：[eLife 2020](https://elifesciences.org/articles/50927)。

本 spec 借的是这些工作的**动力学组织原则**，不是直接复制数值或宣称当前 HFOsp 模型已复现它们。

## 10. 本轮 definition of done

1. 当前 v2/v3 轨迹 paper-ready diagnostic 与真实空间 capture 完成；
2. Stage 0A topology oracle 的 continuation/state-fork 工具通过 toy sanity；
3. Stage 0B 只做原始 0D fast E–I branch screen；若触发结构性 no-go，仅开放锁定的 Stage 0C 动态除法快池；Stage 0C 的单轨迹只允许 Stage 0D 复现、Stage 0E Poincare/Floquet 与 Stage 0F variational certificate 这一条逐级 fail-closed 链；
4. 在明确找到 finite high branch 与 entry/exit boundary 前，不启动 1D/2D 大扫描；
5. 三条旧 workflow 继续冻结为 downstream readout。

### 2026-07-20 执行状态

- Stage 0A 已 PASS：数值恢复 canonical normal form 的解析 entry/exit boundaries、bistable state fork
  与 finite cycle，并正确拒绝 ceiling/long transient；constructed closed toy 的 entry/exit/return/retrigger
  只作为 analyzer sanity。
- 结果：`results/topic4_sef_hfo/spatial_slowfast_topology/stage0a_oracle/stage0a_oracle_summary.json`。
- Stage 0B 已锁为限定范围 `CLEAN_NO_GO_LOW_OR_SATURATION_CLIFF_ONLY`：126 个参数点共 200 个 roots；
  排除 200 条 exact-root 初态后的 1786 条 dynamical forks 为 450 low + 1336 `>100 Hz`，504 条
  off-manifold probes 为 123 low + 381 `>100 Hz`，0 finite candidate/confirm。exact-Siegert sensitivity
  200/200 收敛；37/37 个 sub-100-Hz unstable roots 仍不稳定，111/111 个 stable high roots 仍稳定且
  全部 `>100 Hz`。该结论关闭直接进入 Stage 1--3，但不声称穷尽所有六维 basin 或所有 current-based 模型。
- Stage 0C coarse screen 的 23 条疑似 oscillatory candidates 全部越过原 transfer LUT 支持域，因而只能判 `INCONCLUSIVE_NO_CONFIRMED_FINITE_FAST_OBJECT`。
- 独立 transfer-support v1.1 用经 exact-Siegert 验证的 extra-fine、no-clip/no-extrapolation transfer 重放 6 点、102 histories：17 条 `>100 Hz`，84 条 unresolved，仅 `z=0.85, alpha_G=16, root_0_plus` 一条轨迹在 12 s confirm 与 `dt/2` 下保留约 1.665 Hz 振荡。同点只有 1/17 histories 存活，0 个 supported parameter points，因而当前 verdict 是 `EXTRA_FINE_VALID_NO_SUPPORTED_OBJECT_WITH_UNRESOLVED_TRANSIENTS`，不是 pass 也不是 clean no-go。
- Stage 0D v1.1 已完成：180 条锁定 histories 得到 175 `numerical_unresolved` + 5 `candidate_survives`；5 条全部集中在 `z=0.85, alpha_G=15, phase_050`，中心点无 survivor，无 open basin/相容 Manhattan 邻点。v1-->v1.1 为 0/180 分类改变，`scientific_result_changed=false`；verdict 为 `STAGE0D_NO_REPLICATION_WITH_UNRESOLVED_TRAJECTORIES`。
- Stage 0E 已完成：`alpha_G=15/16` 均有约 605--609 ms 的 period-1 shooting closure，base/half-`dt` 波形一致性通过，每点 16 条多相位 fast/pool 扰动均在 8 returns 内收缩到数值底噪。两点 physical/support audit 通过，但 Floquet Jacobian epsilon 平台失败，因而 verdict 为 `STAGE0E_NUMERICAL_UNRESOLVED`，不是 stable-attractor PASS。
- Stage 0F smooth/variational Floquet certificate 已锁为唯一开放节点；它不扫生物参数、不加 slow/space、不修改 conductance 线。
- Stage 1--3 继续关闭。Stage 0A PASS、Stage 0C engineering completion、Stage 0D survivor 或 Stage 0E 闭合周期都尚不构成 HFOsp Z/recovery 时空机制证据。
