# Topic 4：persistence-gated additive Z–M spatial lifecycle（conditional design v0.3）

日期：2026-07-20

状态：entry/exit cheap geometry 已完成；0D slow lifecycle 尚未执行；空间层和完整 SNN 仍关闭。

本线 worktree：`.worktrees/topic4-mz-divisive-lifecycle`

并行线边界：本线固定 `W_EE`、E→E kernel/delay、长轴各向异性和 presynaptic relay；不实现 conductance membrane，不增加第二个 recurrent-E divisor。另一条 `.worktrees/topic4-mz-conductance` 继续拥有 E→E conductance/relay 路线。

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

| 已维持 fast cycle | 未抵消的 Z 估计 | 追上 fold 所需 A |
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

`p` 只做“已经持续进入 ictal family”的证据，不直接改任何 current。现有 capture 中 `T_G` 在 onset 前最大约 `0.0600`，onset 后 2 s 的下四分位约 `0.0834`，因此单 seed pilot 有一个 `p_r≈0.0717` 的非空区间；正式锁定必须用全 pre-onset history 和 primary seeds 重算，不能把该单 seed 数值直接当参数。

### 3.3 M 改为有界、状态门控的 additive recovery effector

\[
G_p=H_{\kappa_p}(p-p_r),
\]

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

## 4. 两个慢 nullcline 与 exit 曲线

在 low/cycle sheet 上分别周期平均。`p` 与 `m` nullcline 为：

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

cycle sheet 内若 `dot p=dot m=0` 在 exit curve 之前形成稳定交点，则预测 permanent bursting，应判当前参数 no-go。

## 5. 0D cheap-first 执行合同

### Stage A：formal boundary

1. 对 `(z,A)` 做 fixed-point pseudo-arclength continuation，复核 `z_SN(A)`；
2. 对 fast cycle 做双向 shooting continuation，区分真正 `B_C` 与 state-fork/basin boundary；
3. 保存 period、amplitude、Floquet multipliers、low/saddle eigenvalues；
4. base/half `dt`，smooth/exact transfer 一致；
5. `>=100 Hz` peak/occupancy 单独报告，不允许只看 mean。

### Stage B：slow nullcline feasibility

不做宽网格，先用三档 geometry-informed `A_max={0.9,1.3,1.6} mV`：

- `0.9 mV` 约对应 3-cycle lower bound；
- `1.3 mV` 约对应当前 sensor-delay stress test；
- `1.6 mV` 只作 leverage upper sensitivity。

`tau_p/p_r` 先由 locked IED history 与 fast-cycle history 的 separation interval 决定；`k_up` 再锁定为至少保留 3 个 cycle、但在 5 个 cycle 前横穿 exit curve；`k_down/z_safe` 最后由 early/late retrigger 决定。不得按最终成功轨迹反向挑 threshold。

### Stage C：完整 0D lifecycle

使用冻结的 returning-IED drive/noise history：

1. slow-off 仍有 returning IED；
2. `m≈0` 时 Z 跨 low fold；
3. 建立至少 3 个 fast cycle；
4. 无 reset 横穿正式 `B_C` 并回 exact low basin；
5. Z 回到 `z_safe` 前 m 保持；
6. early retrigger 失败、late retrigger 恢复；
7. 返回 `(z,p,m)≈(1,0,0)`。

## 6. 空间版本：同一 additive 机制生成 wake 与 containment

0D 通过后才定义 spatial persistence field：

\[
u_p(x,t)=K_p*h,
\]

\[
K_p=(1-\gamma_p)G_{\sigma_L}
+\gamma_pG_{\sigma_B},
\qquad \sigma_L<\sigma_B\ll L.
\]

两个核均归一化，保证 homogeneous limit 精确复刻 0D；第一版不使用 domain-uniform `U`，避免反馈强度随模拟域面积变化。

- local component：被招募区域后方 build `m(x)`，形成 refractory wake / trailing front；
- broad component：招募面积扩大时，在前沿前方提高 `p/m`，使 leading front 的 local fold 下移并可能 stall；
- `gamma_p=0` 预期只能产生 traveling pulse/wake，通常不能全域 containment；
- `gamma_p>0` 才检验 front stall/annihilation。

空间验收必须分别保存：`t_on(x)`、`t_off(x)`、leading/trailing front speed、当前活动面积、历史招募面积、`z/p/m` 场与 local fold-distance map。不能再用 population rate 或 global scalar 冒充 tissue-state front。

## 7. Stop rules

以下任一出现就停止当前结构，不用大网格挽救：

1. formal cycle continuation 显示 `A<=1.6 mV` 时 stable cycle 仍存在，state fork 只是换 basin；
2. primary seeds 的 locked pre-onset history 与 cycle history 没有共同的 `p_r` separation interval；
3. 保留至少 3 cycle 后，`A_max m(t)` 始终追不上随 Z 下降而上升的 `A_SN[z(t)]`；
4. slow vector 在 exit curve 前形成 stable cycle-sheet fixed point；
5. transition-near orbit 的 `>=100 Hz` 违反在 base/half `dt` 均持续；
6. additive recovery 在 gate 泄漏时只表现为 prevention；
7. offset 后 m 过早下降导致立即 re-entry，或过晚下降导致永久 suppression；
8. 空间 `gamma_p=0` 无 refractory wake；
9. `gamma_p>0` 只造成全场同步 prevention，没有局部 onset、front、stall/annihilation；
10. 任何实现修改 E→E relay、weight、kernel 或 delay——该结果归并行线，不归本线。

## 8. 当前允许与禁止的结论

允许写：

> 当前 fast system 已有 low saddle-node 与大振幅周期的共同边界骨架；additive M 能定量移动该 fold，并在冻结 state fork 中产生回低 leverage。旧线性 M 的失败主要是 entry/exit gating 混叠。下一步应测试 persistence-gated bounded M 是否能在 Z 持续漂移下及时横穿正式 cycle boundary。

禁止写：

- 已证明 SNIC；
- 已证明 stable physiological ictal limit cycle；
- `0.32 mV` 足以终止真实动态 Z 轨迹；
- 已形成 torus；
- 已有 spatial seizure front、containment 或 recovery；
- 已完成完整 SNN lifecycle。
