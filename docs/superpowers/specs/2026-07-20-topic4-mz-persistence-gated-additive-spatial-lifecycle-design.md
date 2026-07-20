# Topic 4：persistence-gated additive Z–M spatial lifecycle（executed design v0.5）

日期：2026-07-20

状态：formal stable-cycle boundary、dual-sensor Stage B 与 seed-1 spatial Gate 0 sentinel 已执行；memoryless 0D gate 为 clean no-go，single-seed operational decomposition 支持 spatial recruitment 是缺失状态，但 formal multiseed Gate 0 仍开放。下一执行节点为 P=1/uniform parity。

本线 worktree：`.worktrees/topic4-mz-divisive-lifecycle`

并行线边界：本线固定 `W_EE`、E→E kernel/delay、长轴各向异性和 presynaptic relay；不实现 conductance membrane，不增加第二个 recurrent-E divisor。另一条 `.worktrees/topic4-mz-conductance` 继续拥有 E→E conductance/relay 路线。

## 0. 2026-07-20 执行判定

### 0.1 Fast boundary：机制方向成立，正式命名仍开放

在 `z=.85, alpha_G=15` 上，event-restarted Poincaré continuation 得到：

- stable connected branch 最后 accepted：`A=.31645 mV`；
- 首个无 `12 s` 内 return：`A=.31648 mV`；
- fixed-point fold：`A_SN=.3165099207 mV`；
- 周期由 `604.8 ms` 增至 `11.35 s`，轨道到 fold state 的 scaled distance 缩小约 `8445` 倍；
- 对 `A>=.28 mV`，`T=c0+c1/sqrt(A_SN-A)` 的 `R²=.999987`；
- 横向 multiplier 没有趋近 `+1`，因此不支持 finite-period limit-cycle fold；但 `.31/.316 mV` 的 finite-difference matrix 平台未通过，且 exact transfer 在 `.3164 mV` 未于 `12 s` 内返回。

因此当前最安全标签是 **strong SNIC-like candidate, formal label open**。它确认 additive current 能把既有 fast cycle 推回 low-state bottleneck，但只追踪了与 `A=0` 连通的 attracting branch，未排除 unstable/disconnected cycle。

### 0.2 Stage B：原 memoryless persistence gate 不可进入完整 0D

同一组参数同时检验：

1. `recorded_SNN_UTG_replay`：保留真实空间招募历史；
2. `endogenous_phase_sensor`：由 alpha-15 fast cycle 的 `rE_fast` 重建 raw `A_G`，再走完全相同的 `slow_gate_drive`。

primary `tau_p=750 ms,p_r=.0722287` 下：

- SNN gate 到 `4.56` 个 A=0 baseline-period equivalents 才首次打开；memoryless `Amax=1.6 mV,tau_up=125/100 ms` 到第 5 个 equivalent 的 margin 仍为 `-0.257/-0.132 mV`，均为 insufficient leverage；
- 四个 homogeneous phase offsets 在 `0.03–0.81` 个 baseline-period equivalents 已开门/越界，全部是 prevention risk；
- `tau_p={500,750,1000} ms` 与全部五个 `Amax/tau_up` arms 没有共同可行组合。

按 Stage B stop rule，**不执行完整 0D 或空间/SNN**。这不是 additive exit leverage 的否定，而是原先“一个 memoryless scalar p 同时表达空间招募与 established-state memory”的假设被否定。

### 0.3 Post-no-go latch diagnostic：定位出两个缺失状态

主结果失败后追加、且不能改判 primary 的乐观诊断：p 首次跨阈后把 build gate 锁存到未来 `z_safe` release。此时 SNN replay 的 `Amax=1.6 mV,tau_up=125/100 ms` 分别在 `4.958/4.864` 个 baseline-period equivalents 追上 exit oracle；但 homogeneous phase 仍在第一个 equivalent 内 prevention。这些不是 dynamic slow trajectory 的真实 Poincaré return counts。

所以缺的不是单纯更大电流，而是两个彼此独立的状态：

1. **spatial recruitment coordinate**：区分“一个局部成熟 oscillator”与“组织范围正在扩大”；
2. **post-detection latch/hysteresis**：一旦 established ictal state 被确认，effector 继续 build，直到 Z 回到安全侧才释放。

下一版不再强行把空间信息压成 homogeneous scalar；优先进入 coarse multi-patch rate field，而不是继续调 `p_r/Amax`。

### 0.4 Gate 0 sentinel：空间坐标方向成立，瞬时 rho 阈值不成立

对锁定 seed-1 capture 做 artifact replay。当前 `p_pool=1`，因此：

\[
A_G(t)=\max_x\Psi(r_{E,fast}(x,t))\,\rho_{eff}(t),
\qquad
\rho_{eff}=\frac{\langle\Psi\rangle_x}{\max_x\Psi}.
\]

重建 `U_TG` 最大误差 `1.74e-7`，重建 `T_G` 最大误差 `7.86e-9`。同一 `tau_p=750 ms,p_r=.0722287` 下，local-intensity-only p 在记录 `421.7 ms` 已由普通 IED 开门，area-weighted p 则在 causal macro onset 后 `2758.5 ms` 开门。

以 causal frame end 对齐的 onset-seed `[0,.25 s)` 和 recruited `[1,3 s)` window 比较：local raw peak 仅变 `+2.64%`，`rho_eff` 增 `.0994`，独立 24×24 movie area-PR 增 `.1139`；在 `|Delta Psi_peak|<=.02` 的 10 对 frame 中，两者 median increase 仍为 `.0562/.0623`，且 8 个连续 250-ms block 方向一致。post-onset `rho_eff` 与 movie PR descriptive Spearman 为 `.942`；不对 dependent frames 报 iid p-value。48-bin axial PR 只作 longitudinal-span sensitivity，不进入二维主判定。

但完整 pre-onset `rho_eff` 最大 `.3000`，高于 onset+2 s 后 established Q25 `.1579`，所以一个 instantaneous `rho>rho_r` 仍会被 IED 误触发。当前只有 seed 1 且没有 core-specific numerator / same-field `sum(z_G^2)`，因此状态锁为：

> `single_seed_operational_spatial_decomposition_supported_persistence_AND_latch_required`

这允许进入 P=1 parity 和 P=2 cheap oracle，但不允许声称 formal Gate 0 pass，也不允许从该 trace 标定 memoryless rho threshold。

## 1. 这次反思修正了什么

### 1.1 旧 QI/JK 成功对象被混淆了

旧 `q_I+g_K` screen 没有得到可控 ictal state；真正打开 bounded oscillatory third state 的是：

\[
I_E^{net}=I_E^{ff}+\frac{I_E^{rec}}{1+\alpha_GS_G}-q_II_I,
\]

而该锚点中 `g_K` 明确关闭。当前 Stage 0C–0F 复用了同一个 delayed recurrent-E divisor，因此它是已经建立的 **fast inner-cycle generator**，不是本线可以再次包装成“新 recovery”的机制。

### 1.2 上一版 `D_R` 应退出

若再写：

\[
I_E^{rec}\mapsto\frac{I_E^{rec}}{(1+\alpha_GS_G)(1+\kappa_Rr)},
\]

那么冻结快系统时，`r` 只会塌缩成另一个 effective E→E gain。它同时重复旧 M4/current divisor、当前 `T_G` 和并行 E→E relay 线，不能提供独立的 entry/exit 方向。因此 v0.2 `D_R` 方案已标记 superseded。

### 1.3 “additive M 一定无效”也不正确

在锁定的九维 rate system 中加入冻结负 E-current：

\[
\mu_E\mapsto\mu_E-A,
\]

会真实移动 E-nullcline 与 low-state saddle-node，不只是压低平均 rate。精确结果为：

\[
z_{SN}(0)=0.87447467,
\qquad
(r_E,r_I)=(2.0264,7.0559)\,\mathrm{Hz}.
\]

在 `z=0.85` 时：

\[
A_{SN}=0.3165099\ \mathrm{mV}.
\]

从已建立周期出发，`A=0.31 mV` 仍振荡、`A=0.32 mV` 回到低固定点，base/half `dt` 一致。旧线性 M 的主要失败不是没有 exit leverage，而是每个 spike 都使 M 从 `t=0` 累积，先移动 entry corridor，因而表现为 prevention。

### 1.4 但 `0.3165 mV` 只是冻结 Z 的乐观下界

发作中 Z 仍下降。用现有 20 s capture 的平均 Z 恒等式反推 representative depletion occupancy，并让 reduced Z 在没有 recovery 时漂移：

| A=0 baseline-period equivalents | 未抵消的 Z 估计 | 追上 fold 所需 A |
|---:|---:|---:|
| 1 cycle | 约 0.849 | 约 0.33 mV |
| 2 cycles | 约 0.825 | 约 0.62 mV |
| 3 cycles | 约 0.803 | 约 0.89 mV |
| 4 cycles | 约 0.781 | 约 1.14 mV |
| 5 cycles | 约 0.761 | 约 1.37 mV |

现有 `T_G` persistence trace 的 pilot threshold 约在 causal onset 后 `2.76 s` 才跨越；若此前 Z 不受抵消，所需 A 已约 `1.28 mV`。这些数是 **cross-model stress-test oracle**，不是校准后的慢轨迹；它们的作用是锁定真正的困难：M 必须足够晚才不 prevention，又必须足够早、足够强才能追上持续下降的 Z。

## 2. 已解析的 fast topology

保留九维快状态：

\[
X=(r_E,r_I,s_{EE},s_{EI},s_{IE},s_{II},\bar r_E,\mu_G,S_G).
\]

冻结 `(z,A)` 后：

\[
\tau_E\dot r_E=-r_E+\Phi_E(\mu_E-A,\sigma_E),
\]

\[
\tau_I\dot r_I=-r_I+\Phi_I(\mu_I,\sigma_I),
\]

\[
\mu_E=\tau_E\left[
\frac{C_{EE}w_{EE}s_{EE}}{1+\alpha_GS_G}
-C_{EI}zW_{EI}s_{EI}
\right]+\mu_X.
\]

平衡时 `S_G=Psi(r_E)`。由于 fold 处 `r_E<5 Hz`、`Psi=0`，`alpha_G` 不移动 low-state entry fold；它只在较高 rate 把 E-nullcline向回弯，将原来的 saturation branch 变成 unstable focus 外围的大振幅周期。

周期从 `z=.85` 的约 `605 ms` 延长到 `z=.87445` 的约 `5.5 s`，满足：

\[
T\approx500.1+\frac{24.45}{\sqrt{z_{SN}-z}}\ \mathrm{ms},
\qquad R^2=0.9993.
\]

这是很强的 SNIC-like 证据，但在 periodic-orbit shooting continuation 完成前不正式命名分岔。稳定 torus 不是目标；需要的是轨迹沿 fast-cycle family 做有限次旋转，再跨回 low branch 的 transient slow passage。

## 3. 下一版最小 0D 方程

> **执行后状态：memoryless scalar 0D no-go。**本节保留被检验方程与失败对象，不能再作为下一执行节点；替代结构见 §6 的 spatial recruitment AND-set / Z-safe-reset latch。

### 3.1 Z 保持原 entry 语义

第一版不改 Z 方程：

\[
\tau_z\dot z=z_\infty(I_I^{raw})-z,
\qquad
z_\infty=H(I_{th}-I_I^{raw}).
\]

Z 只承担 inhibitory exhaustion / entry permission；不要求它自己在 ictal state 中反向。

### 3.2 把现有慢 gate 从 E→E 分母上拆下，改成纯 persistence sensor

定义因果、无未来泄漏的局部 macrostate drive：

\[
h(x,t)=H_{\kappa_e}\!\left(\bar r_E^{causal}(x,t)-r_p\right).
\]

0D 时：

\[
\tau_p\dot p=-p+h.
\]

`p` 只做“已经持续进入 ictal family”的证据，不直接改任何 current。按 causal onset 和完整 pre-onset history 重算，seed 1 在 `tau_p=750 ms` 时 pre-onset 最大约 `0.0611`，onset+2 s 后下四分位约 `0.0834`，midpoint pilot 为 `p_r≈0.0722`；正式锁定必须补齐 primary seeds 的 `A_G/U_TG` history，不能把该单 seed 数值直接当参数。

### 3.3 M 改为有界、状态门控的 additive recovery effector

第一版 gate 必须有 literal-zero dead zone，不能使用在 `p=0` 仍泄漏的普通 logistic：

\[
G_p(p)=
\begin{cases}
0,&p\le p_0,\\
3u^2-2u^3,&p_0<p<p_1,\\
1,&p\ge p_1,
\end{cases}
\qquad u=\frac{p-p_0}{p_1-p_0}.
\]

primary 可先取 hard-gate limit `p_0=p_1=p_r`；有限宽度只作预注册 sensitivity。

\[
\dot m=
k_\uparrow G_p(1-m)
-k_\downarrow(1-G_p)L(z)m,
\qquad m\in[0,1],
\]

\[
L(z)=\epsilon_m+(1-\epsilon_m)
H_{\kappa_z}(z-z_{safe}),
\qquad 0<\epsilon_m\ll1.
\]

膜/率方程只增加：

\[
A(x,t)=A_{max}m(x,t),
\qquad
I_E^{net}=I_E^{ff}
+\frac{I_E^{rec}}{1+\alpha_GS_G}
-zI_I-A.
\]

这与旧 `m_i += 1 per spike` 有三个本质区别：

1. `m` 有界，不再是无上界 spike count；
2. `p>p_r` 后才 build，普通 IED 不直接产生 recovery current；
3. offset 后只在 Z 回到安全侧才快速 decay，形成 early-retrigger protection。

`epsilon_m` 必须大于零，避免 unsafe 区域形成非双曲 memory continuum。

## 4. 三个慢 nullcline 与 exit 曲线

在 low/cycle sheet 上分别周期平均。必须保留 Z 在 burst silent phase 中的恢复占空比：

\[
Q_a(z,m)=\left\langle H(I_{th}-I_I^{raw})\right\rangle_a,
\qquad
\bar h_a(z,m)=\langle h\rangle_a,
\qquad a\in\{L,C\}.
\]

完整的三个 slow nullcline 为：

\[
z^*=Q_a(z,m),
\]

\[
p^*=\bar h_a(z,m),
\]

\[
m^*_a=
\frac{k_\uparrow\bar G_{p,a}}
{k_\uparrow\bar G_{p,a}+k_\downarrow(1-\bar G_{p,a})L(z)},
\qquad a\in\{L,C\}.
\]

快系统 fold surface 给出 slow-plane exit curve：

\[
z=z_{SN}(A_{max}m),
\]

或反解为：

\[
m_{exit}(z)=\frac{A_{SN}(z)}{A_{max}}.
\]

目标轨迹为：

\[
z\downarrow,\ m\simeq0
\rightarrow \text{entry}
\rightarrow p\uparrow
\rightarrow m\uparrow
\rightarrow A_{max}m>A_{SN}(z)
\rightarrow \text{low return}
\rightarrow z\uparrow
\rightarrow m\downarrow.
\]

cycle sheet 内若 `dot z=dot p=dot m=0` 的共同交点位于 exit curve 之前，则预测 permanent bursting，应判当前参数 no-go。只画 `p/m` 两条 nullcline 会漏掉 silent phase 造成的 `Q_C>0`，不得作为执行合同。

定义 boundary distance：

\[
D(z,m)=z-z_{B_C}(A_{max}m).
\]

entry 必须在 `D=0` 时满足 `dot D<0`；exit 必须在 `D=0` 时满足 `dot D>0`。在 formal cycle boundary 尚未完成前，可用 `z_SN(A)` 作 oracle，但不能用它替代最终 `B_C`。

## 5. 0D cheap-first 执行合同

### Stage A：formal boundary

**执行状态：完成 attracting connected branch；unstable branch / formal label 仍开放。**本轮 directed Poincaré continuation 已把 stable-cycle strip 压到 `.31645–.31648 mV`，并与 fixed-point fold 对齐；因为证据指向 infinite-period collision 而非 finite-period `+1` multiplier，未触发 cycle-fold pseudo-arclength。transition-near derivative ladder 的 matrix platform 未全过，不能把本节点写成正式 SNIC 证明。

1. 对 `(z,A)` 做 fixed-point pseudo-arclength continuation，复核 `z_SN(A)`；
2. 对 fast cycle 做双向 shooting continuation，区分真正 `B_C` 与 state-fork/basin boundary；
3. 保存 period、amplitude、Floquet multipliers、low/saddle eigenvalues；
4. base/half `dt`，smooth/exact transfer 一致；
5. `>=100 Hz` peak/occupancy 单独报告，不允许只看 mean。

### Stage B：slow nullcline feasibility

**执行状态：primary clean no-go。**下列三档与 `tau_p={500,750,1000} ms` 已按 dual-sensor 合同执行；memoryless gate 无任何共同可行 arm。完整结果见 `results/topic4_sef_hfo/mz_persistence_feasibility/`，不得继续到 Stage C。

不做宽网格。先计算 `(t_gate,tau_up,A_max)` race feasibility，再决定哪些组合值得积分。三档 `A_max={0.9,1.3,1.6} mV` 的角色不等价：

- `0.9 mV`：3-cycle frozen-Z lower bound，预注册 negative-leverage anchor；
- `1.3 mV`：当前约 2.76-s sensor delay 下只剩约 `0.023 mV` headroom，合理慢尺度预计失败；
- `1.6 mV`：当前唯一可行主候选；若要求第 5 周期前退出，oracle 约要求 `tau_up<=135 ms`。

`tau_p/p_r` 先由 locked IED history 与 established-state history 的 separation interval 决定；`k_up` 再锁定为至少保留 3 个 cycle、但在 5 个 cycle 前横穿 exit curve；`k_down/z_safe` 最后由 early/late retrigger 决定。不得按最终成功轨迹反向挑 threshold。

当前 persistence 数据只有 seed 1 的完整 `A_G/U_TG/T_G`；seeds 3/4 只有 population-rate/event history。因此 `tau_p=750 ms,p_r≈0.0722` 只能作为 seed-1 pilot，不能写成 multiseed lock。0D 必须并列运行两个 bracket：

1. `endogenous_phase_sensor`：检查 homogeneous mature cycle 是否第一圈就过早触发；
2. `recorded_SNN_UTG_replay`：检查空间招募导致的晚 gate 下是否仍有 exit leverage。

两个 bracket 必须使用同一组 `tau_p,p_r,tau_up,A_max`。只在一个 bracket 成功应判 sensor-transfer mismatch，不得挑成功的一侧继续。旧 `T_G` 已经是 persistence integrator；新 `p` 是替代它的状态，不得对 `T_G` 再叠加第二层低通。

### Stage C：完整 0D lifecycle

**当前关闭。**原 0D homogeneous sensor 把 spatial recruitment 压没，导致 mature-cycle bracket 在第一周期内 prevention；在 coarse spatial coordinate 建立前，不再执行或调参抢救这一版。

使用冻结的 returning-IED drive/noise history：

1. slow-off 仍有 returning IED；
2. `m≈0` 时 Z 跨 low fold；
3. 建立至少 3 个 fast cycle；
4. 无 reset 横穿正式 `B_C` 并回 exact low basin；
5. Z 回到 `z_safe` 前 m 保持；
6. early retrigger 失败、late retrigger 恢复；
7. 返回 `(z,p,m)≈(1,0,0)`。

## 6. 下一节点：recruitment-set / Z-safe-reset spatial latch

### 6.1 为什么不再继续原 homogeneous 0D

当前 dual-sensor no-go 已证明 homogeneous mature cycle 和逐步扩大 recruited area 不是同一个 autonomous nullcline。下一步先用 `0D+rho replay` 检验 recruited fraction 是否真能解释晚 gate，但它只允许作为 necessary-condition oracle；外加一个 logistic `rho(t)` 不能算空间模型。

真正的最小动力学节点是 **core–surround 两区**。实现可写成通用 P-patch 模块，但执行顺序必须为 `P=1 parity → P=2 core–surround → P=32/64 coarse field`，不能直接用 field 大网格同时调 sensor、coupling 和 termination。

### 6.2 P-patch 状态与共享 pool

每个 patch 只保留七个 local fast states：

\[
X_j=(r_{E,j},r_{I,j},s_{EE,j},s_{EI,j},s_{IE,j},s_{II,j},\bar r_{E,j}),
\]

加上 local `z_j,p_j,m_j`。全域只能有一对共享：

\[
(\mu_G,S_G),
\]

因此连续 state shape 为 `10P+2`，另有 P 个 hybrid latch bits。不能把 Stage 0C 的 batch rows 直接当 patches，因为那会给每个 patch 偷偷复制一套 local `mu_G/S_G` divisor，人工制造 containment。

spatial synaptic targets 只复用既有固定 normalized kernels：

\[
s_{EE}\leftarrow K_{EE}*r_E,
\quad
s_{EI}\leftarrow K_I*r_I,
\quad
s_{IE}\leftarrow K_I*r_E,
\quad
s_{II}\leftarrow K_I*r_I.
\]

`W_EE`、kernel width/direction、delay 和 synaptic tau 全部冻结，不允许通过重调 coupling 抢救本机制。uniform P-patch limit 必须复刻原九维 Stage 0C period/boundary。

共享 recruitment pool 为：

\[
A_G=\left[\frac{1}{P}\sum_j\Psi(\bar r_{E,j})^{p_G}\right]^{1/p_G},
\]

并沿用锁定的 `tau_mu/tau_S` 更新唯一 `mu_G/S_G`。

### 6.3 Recruitment AND gate 与 latch

定义 local ictal occupancy：

\[
y_j=H_\kappa(\bar r_{E,j}-r_{ict}),
\]

以及不进入 recurrent current 的 neighborhood recruitment readout：

\[
\rho_j=\sum_k K^P_{jk}f_k y_k.
\]

local persistence：

\[
\tau_p\dot p_j=-p_j+y_j.
\]

latch set 必须是 local persistence 与 spatial recruitment 的 AND gate：

\[
G_{on,j}=G_p(p_j)G_\rho(\rho_j).
\]

core-only 状态必须保持 `G_rho=0`，从而 entry 后前几圈严格 `dot m=0`；只有 surround/neighbor 被实际招募后才允许 set。latch 一旦 set，不能随 bursting trough memoryless 关闭；reset 必须同时满足 local low-branch membership、`z_j>=z_safe` 和 `p_j<=p_off`。

连续 effector 可写成：

\[
\dot m_j=k_\uparrow\ell_j(1-m_j)
-k_\downarrow(1-\ell_j)L(z_j)m_j,
\qquad
A_j=A_{max}m_j,
\]

其中 `ell_j` 是上述 set/reset 合同的 hybrid state。memoryless `ell_j=G_on,j` 必须保留为负 baseline，不能删除。

### 6.4 Core–surround frozen sheets

两区先只检查 `LL/CL/CC/LC` 四个 sheet。邻区状态会移动本区边界，因此需要：

\[
z_{B,j}=z_{B,j}(A_j,X_k^{L/C}),
\]

\[
D_j=z_j-z_{B,j}(A_{max}m_j,X_k).
\]

目标顺序：

1. `LL→CL`：core `D_c=0,dot D_c<0`，同时 `dot m_c=0`；
2. `CL→CC`：冻结原 coupling 使 surround `dot D_s<0`，并至少晚一个真实 section return；
3. `CC→low`：rho gate set 后至少一个区 `D_j=0,dot D_j>0`，随后另一域退出；
4. low sheet：Z 恢复、M 短期保持后 reset，回到原 basin。

`CL/CC` sheet 内若三个 slow nullcline在 exit boundary 前形成共同稳定点，判 permanent-bursting trap。

### 6.5 Cheap gates

1. **Gate 0，artifact replay**：seed-1 operational sentinel 已完成。area-weighted local signal 能解释 recorded late gate，并与独立二维 movie extent 一致；但当前 artifact不能精确提取 core occupancy、没有 primary-seed common interval，formal Gate 0 继续开放。新增 capture 最小字段为每步 `sum(z_G),sum(z_G^2),max(z_G)` 与 core/surround compact numerator，不保存 full field/raster。
2. **Gate 1，P=1/uniform parity**：RHS、period、boundary、constant-preserving kernels 必须复刻 Stage 0C；失败即工程 no-go。
3. **Gate 2，P=2 core–surround**：只跑 memoryless/latch、rho off/on、m off、cross-zone coupling off；`Amax=1.6,tau_p=750,tau_up=125 ms` 起步，不做宽网格。
4. **Gate 3，coarse 1D field**：两区通过后原参数直接移植；先 `gamma_p=0` 看 wake，再只加一个预注册 broad arm（可用 `1/6`）看 stall/annihilation。

必须用真实 Poincaré returns 计 cycle。near-boundary period 已达到 `1.6–11.3 s`，不能再用固定 `604.8 ms × cycle count` 代替真实快慢时标。

空间验收保存：`t_on/off(j)`、section returns、latch set/reset、`D_j`、leading/trailing front speed、active/recruited width、`z/p/m` fields、peak rate 与 `>=100 Hz` occupancy。不能再用 population rate 或 global scalar 冒充 tissue-state front。

## 7. Stop rules

以下任一出现就停止当前结构，不用大网格挽救：

1. formal cycle continuation 显示 `A<=1.6 mV` 时 stable cycle 仍存在，state fork 只是换 basin；
2. area-weighted local occupancy 不能解释 `U_TG` 的 late spatial gate，或正式 SNN 确认阶段 primary seeds 没有共同 spatiotemporal separation；当前 seed-1 replay 支持前者，但 instantaneous rho interval 明确失败，所以 P=2 必须使用 persistence AND recruitment，不能 memoryless set；
3. P=1/uniform-P 不能复刻原 Stage 0C RHS、period 或 boundary；
4. 每个 patch 拥有自己的 `mu_G/S_G`，而不是全域唯一 shared pool；
5. core-only 时 latch 在第 3 个真实 return 前 set，重新造成 prevention；
6. recruitment 后 `A_max m(t)` 仍追不上随 Z 下降而上升的 exit boundary；
7. crossing 距 3–5 return window 边缘小于 `.25 return`，或 current margin 小于 `.1 mV`/预注册 uncertainty band；当前 `4.958 baseline-period equivalents,.034 mV` latch diagnostic 因此不能升格；
8. slow vector 在 `CL/CC` exit curve 前形成 stable cycle-sheet fixed point；
9. transition-near orbit 的 `>=100 Hz` violation 在 base/half `dt` 均持续；
10. offset 后 m 过早下降导致立即 re-entry，或过晚下降导致永久 suppression；
11. cross-zone coupling off 后仍凭 imposed rho/latch“招募”，说明空间动力学是外加脚本；
12. coarse field 的 `gamma_p=0` 无 refractory wake；
13. broad arm 只造成全场同步 prevention，没有局部 onset、front、stall/annihilation；
14. 任何实现修改 E→E relay、weight、kernel 或 delay——该结果归并行线，不归本线。

## 8. 当前允许与禁止的结论

允许写：

> Additive current 的 attracting fast-cycle branch 在 fixed-point fold 前出现 inverse-square-root period divergence，并在 `A=.31645–.31648 mV` 的窄带失去有限 return，支持 strong SNIC-like boundary candidate。原 memoryless scalar persistence gate 在真实空间 SNN history 中过晚、在 homogeneous mature-cycle history 中过早，未得到共同 3–5 baseline-period-equivalent exit arm；post-no-go latch 只恢复了 SNN timing leverage，说明下一版必须显式加入 spatial recruitment coordinate 与 post-detection hysteresis。

禁止写：

- 已证明 SNIC；
- 已证明 stable physiological ictal limit cycle；
- `0.32 mV` 足以终止真实动态 Z 轨迹；
- 已形成 torus；
- 已有 spatial seizure front、containment 或 recovery；
- 已完成完整 SNN lifecycle。
- latched diagnostic 已通过生命周期验收；
- memoryless no-go 可以靠继续扩大 `Amax` 直接抢救；
- homogeneous phase sensor 可以代表局部起始后逐步扩大的空间招募。
