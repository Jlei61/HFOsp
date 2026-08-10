# Topic 4 rev9：data-driven node-edge substrate factorization 设计

**状态：** 设计冻结候选，尚未执行 rev9 仿真  
**日期：** 2026-08-10  
**上游：** rev8.1 frozen field，最终 blind verdict 为 `RIGID_TEMPLATE_MATCH_NOT_BEATEN`

## 1. 科学问题

rev8.1 已经证明：自由连续场能在同一网络中恢复稳定的双模式符号结构，并改善全局事件剖面距离；但最弱模式
与患者不一致，且没有胜过 rigid filament / hand dual-core 的模式匹配。下一轮不能继续把所有变化都塞回阈值场，
而要回答一个更窄、更可证伪的问题：

> 同一个冻结空间场 `h(x,y)` 所定义的局部易感底物，能否由逐神经元阈值调制（node）与局部 E->E
> source-target 权重重分配（edge）分别实现；两者单独或联合时，哪一层控制点火位置、模式比例与传播形态？

rev9 是**静态底物的机制分解**，不是发作生命周期机制，也不是重新恢复传播轴。只有 node/edge 的局部响应等效、
因果消融和事件读出都完成后，才允许把冻结底物交给相图与 lifecycle 线路。

## 2. 冻结输入与结论边界

- `h`、K=3、17 个场参数、预算投影、神经元位置顺序和 rev8.1 candidate hash 全部冻结，不在 rev9-A/B 重拟合。
- node 的异质性深度 `d_i`、quantile seed 和符号保留；不得把它简写成“core 内统一降阈值”。
- E->E 结构轴仍来自患者 rank-derived connectivity；rev9 不独立发现轴。
- patient train 可用于开发指标。rev8.1 patient held-out 已经打开，**不得再次称为 blind**。
- 若以后执行患者驱动的 edge 优化，必须先提供新冻结、从未参与 rev8/rev9 的 recording blocks 或独立数据单元；
  否则只能报告 development result。
- 所有长仿真由 `/usr/bin/nohup` 启动，写 `.log/.status`，结束时 `notify-send`。

## 3. 三轮设计反思

### 3.1 数学与代码方向审计

仓库中 `ampa_by_delay[d]` 的矩阵方向为**行=target，列=excitatory source**。矩阵权重已经包含
`tau_m,target / tau_r,AMPA` 的 jump factor；放电后按 source 列展开，再把权重加到 target 的 delay ring。
因此 rev9 所有公式固定写 `t=target, s=source`，禁止继续用含糊的 `i,j`。

现有 `1 + alpha h_t h_s` 本身是非可分离 pair interaction，并不会被 per-target normalization 完全抵消。
真正会抵消的是 `a(h_s)b(h_t)`：对固定 target，`b(h_t)` 是公因子。rev9 仍改用指数形式保证权重严格为正，
但不能把“改成指数”误写成修复了一个现存 bug。

### 3.2 可辨识性与预算审计

node 参数单位是 mV，edge 参数是无量纲权重重分配；二者没有可直接比较的“同预算”。正式比较改为
**matched-local-response**：先用 paired small-kick/sham 测出 node-only 的局部响应曲线，再在守恒约束内选择 edge 参数。
若 edge 在预注册的权重畸变范围内无法匹配 node，就判定当前 edge family 不足，不扩大参数直到碰到结果。

edge primary 只共享学得的非负空间 envelope `h`，不把 `d` 偷渡进连接权重。`d` 是冻结的、位置独立的 node
阈值抽样，而不是患者学得的空间变量；因此本轮问的是“同一 h 几何能否由 edge redistribution 实现”，不是要求 edge
逐神经元复制 signed `Delta Vtheta=-hd`。若二者不等效，这是机制分解结果，不是必须通过的工程 gate。

per-target incoming-E 权重和守恒只排除了静态总增益变化，不能守恒实际递归电流，因为 source firing rate、相关性和
时序会改变。实际响应必须由 SNN paired perturbation 测量，不能由权重和替代。

### 3.3 统计与机制边界审计

rev8.1 的平均匹配被 mode B 掩盖，因此 rev9 的连续目标必须保护最弱模式。正对角/负交叉继续作 fail-closed gate，
但不再充当主要连续目标。每个模式要分别比较 recruitment、precedence、mean profile 和完整事件分布。

静态 edge core 只决定 substrate / entry susceptibility。它不提供 slow exit、postictal protection 或 recovery。
任何相图与有限发作表述仍须满足 entry、bounded carrier、exit、postictal protection、return/recovery 五项合同。

`d` 的 spatial shuffle 只检验结果是否依赖这一次冻结随机实现及其与 `h` 相乘后的局部组成。因为 `d` 原本就独立于
位置抽样，shuffle 结果不得表述为发现或破坏了患者来源的生物学空间配准。

## 4. 与真实 SNN 方程的对应

对 delay bin `delta`，原始 E->E 权重写作 `W^(delta)_{t,s}`。它已经是引擎实际注入 synaptic gate 的权重：

```text
s_E,t <- exp(-dt/tau_r,AMPA) s_E,t
         + sum_s W^(delta)_{t,s} spike_s(time-delta)
I_E,t <- s_E,t + exp(-dt/tau_d,AMPA) (I_E,t-s_E,t)
V_t   <- I_net,t + (V_t-I_net,t) exp(-dt/tau_m,t)
spike_t = 1[V_t >= Vtheta,t]
```

### 4.1 Node-only

```text
d_t = Vtheta,0 - Vtheta,core,t
Vtheta,t = Vtheta,0 - gamma_node h_t d_t,     t in E
Vtheta,t = Vtheta,0,                          t in I
```

rev8.1 node-only 固定 `gamma_node=1`。`d_t>0` 降阈值，`d_t<0` 升阈值。

### 4.2 Edge-only primary family

先定义原始 target incoming-E 总量和 source 分布：

```text
S_t = sum_(s,delta) W^(delta)_{t,s}
p^(delta)_{t,s} = W^(delta)_{t,s} / S_t
M_{t,s}(alpha) = exp(alpha h_t h_s)
Z_t = sum_(s,delta) p^(delta)_{t,s} M_{t,s}
W'^(delta)_{t,s} = W^(delta)_{t,s} M_{t,s} / Z_t
```

由此 `sum_(s,delta) W' = S_t` 精确成立。只修改 `t<E, s<E`：E->I、GABA、edge index、delay bin、
神经元位置和外部输入均不变。修改后必须删除旧 `ampa_flat`，让引擎按新权重重建 source-indexed cache。

`alpha>=0`。当 `h_t=0` 时该 target 不受调制；高 h target 会在固定 incoming-E 总量下偏向高 h source。

### 4.3 Secondary geometry term

只有 primary family 不能匹配 node 的空间响应形状时，才允许增加一个参数 `beta>=0`：

```text
kappa_{t,s} = exp[-r_(t,s)^2 / (2 l_EE^2)]
kappa_tilde_{t,s} = (kappa_{t,s} - E_p[kappa|t]) / (SD_p[kappa|t] + eps)
M_{t,s}(alpha,beta)
  = exp[alpha h_t h_s + beta h_t h_s kappa_tilde_{t,s}]
```

`kappa_tilde` 按每个 target 的原始 incoming-E 分布中心化，减少 `alpha` 与纯距离偏置的混淆。`beta` 不得单独作为
全网络短程增益，因为不乘 `h_t h_s` 会改变全片几何而不是形成 core。primary 通过时禁止再开 `beta`。

## 5. Edge 畸变预算与硬工程门

对每个 target 报告：

```text
KL_t = sum_(s,delta) p' log(p'/p)
ratio_(t,s,delta) = W'/W
ESS_t = 1 / sum p'^2
```

formal calibration 只允许候选同时满足：

- every E target incoming-E absolute error `<=1e-9`；
- topology hash、delay-bin hash、E->I hash、GABA hash 不变；
- 所有非零 E->E edge 的 `0.25 <= W'/W <= 4.0`；
- median `KL_t <=0.05` nats 且 p99 `KL_t <=0.20` nats；
- `alpha=beta=0` 与原网络逐值一致；无 NaN/Inf/负权重；
- baseline 不 runaway，不因 edge map 单独进入 tonic plateau。

这些上限是防止“守恒总和但把一个 target 的权重压到少数 source”造成伪 core。若无候选在该范围内匹配 node，
结论是 family insufficiency，不得看结果后放宽上限。

## 6. 实验臂

四个主臂在同一 network seed、OU/Poisson seed、位置、h 和 readout 下配对：

| arm | threshold | E->E edge |
|---|---|---|
| Null | uniform `Vtheta,0` | original |
| Node-only | `Vtheta,0-hd` | original |
| Edge-only | uniform `Vtheta,0` | `W'(alpha*, beta=0)` |
| Node+Edge | `Vtheta,0-hd` | 同一个 `W'(alpha*, beta=0)` |

`alpha*` 只由 §7 local-response calibration 选择。Node+Edge 不重新调 alpha；它检验相加、冗余或非线性交互。
“同预算”统一改写为“edge-only 对 node-only 做局部响应匹配”。

## 7. Matched-local-response calibration

### 7.1 Pulse preflight

- `T=400 ms`、`t_kick=100 ms`、`r_kick=1.0 mm`；kick 与 sham 使用相同 seed。
- 位置固定为三个 Gaussian center 加三个 matched off-field control。
- 候选幅度为 `{0.25, 0.5, 1.0} * nu_theta`。
- 选择最小的 quasi-linear 幅度区间：source 与 downstream response 对幅度单调，二点斜率比在 `[0.7,1.3]`。
- calibration window 内若 kick 或 sham 任一出现 detector-qualified population event，该 paired unit 从线性响应估计中
  预注册排除；每个位置至少保留 `5/6` seeds，否则 instrument calibration 失败。
- 主 response window 从 kick 结束后 `{0--10, 10--20, 20--40} ms` 中选择 first downstream peak 所在窗；
  只看 Node-only canary seeds 冻结，之后不变。

### 7.2 响应向量

每个 seed `r`、位置 `q` 和幅度 `a` 计算 kick-sham：

- `g_source`：source disk 内 E spike-mass slope；
- `g_downstream`：source disk 外 E spike-mass slope；
- `r50` / `r90`：正响应质量的空间半径；
- `axis_ratio`：沿冻结 connectivity axis / transverse 的响应二阶矩比；
- `response_map`：coarse-bin 未归一正响应分布；
- sham baseline rate、event count、runaway/tonic status。

### 7.3 校准损失与选择

`alpha` primary coarse grid `{0,0.25,0.5,1,2,4}`，先应用 §5 畸变门，再对最优相邻区间做两轮 midpoint refinement。

```text
J_cal(alpha) = median_(r,q) [
    z2(log1p g_source_edge - log1p g_source_node)
  + z2(log1p g_downstream_edge - log1p g_downstream_node)
  + z2(r90_edge-r90_node)
  + z2(log axis_ratio_edge-log axis_ratio_node)
] + P_control + P_baseline
```

每个 `z2(delta)=delta^2/s^2`，`s=max(Node-only seed IQR, 0.1*|Node median|, numerical floor)`，并在公式中对应的
同一变换坐标上计算，只由 Node-only calibration seeds 定义。`P_control` 惩罚 edge 相对 node 在 off-field control
出现额外 response。`P_baseline` 来自独立
`T=2000 ms` no-kick paired runs：baseline E-rate 或 active-fraction p95 改变 >25%，以及任何 runaway/tonic，均设
fail-closed 大罚；短 calibration run 的 event count 只报告，不用不稳定的小计数作 25% gate。

匹配通过还需：source 与 downstream slope ratio 均在 `[0.8,1.25]`，`|delta r90|<=1 mm`，response-map
Spearman `>=0.80`，且至少 `5/6` calibration network seeds 同方向满足。否则 edge family 不等效。

## 8. 冻结候选的三个因果诊断

### 8.1 Mode-component onset density

对 rev8.1 全部 50 final events，定义 earliest set 为 event 内 relative onset 的最早 1% E neurons。每个 event 的
二维 histogram 先归一到 1，再在 mode 内平均；source centroid 分配给最近 Gaussian center，报告
`P(component | mode)`。该分析只说明 mode 与起点的关联，不单独证明 component 是 causal core。

### 8.2 Component lesion 与 matched relocation

对 component `c` 的 raw contribution `q_c`：

- direct lesion：`q_minus=max(q-q_c,eps)`，保持原 projection lambda，不补回预算；测总贡献；
- matched relocation：删去 `q_c`，把同 covariance/weight 分量移动到保持 sheet-center radius 和边界距离的冻结
  control location；primary control 是绕 sheet center 旋转 `+90/-90` 度后与原场重叠最小的一侧，component orientation
  同步旋转。对替换后的 raw `q` 重新调用 `project_to_budget(q,target_count)`，只重新求 projection `lambda`，不改变任何
  Gaussian 的 weight/covariance；测位置特异性；
- 每个 component、control location 和 seed 配对；禁止只做最显眼的两个主分量。

报告 event rate、mode proportion、`D_global`、每个 `D_k`、earliest component、duration、size、return。

### 8.3 `d_i` interaction audit

固定 h，至少比较：原始 d、全局 spatial shuffle、1 mm spatial-bin neighborhood shuffle、positive-only、negative-only。
shuffle 保持 d marginal；positive/negative-only 同时报 raw 与 `sum h|d|` matched 版本。这样分开 h 几何、d 异质性和
特定随机实现的局部组成；不得把后者写成患者来源的空间配准。

## 9. 新的 patient-training 连续目标

该目标只在有新 blind unit 时进入 formal optimization；此前只能作 development read-back。

对 mode `k in {A,B}` 定义四个 patient-training-only 距离：

- `D_rec,k`：mode-conditioned contact recruitment probability RMSE；
- `D_prec,k`：共同参与 contact pairs 的 precedence probability weighted RMSE；
- `D_prof,k`：31-point mean normalized-rank profile RMSE；
- `D_dist,k`：frozen embedding 中 mode-conditioned event-cloud sliced Wasserstein。

患者 mode 标签和 prototype 只在 patient train 上冻结。每个模型候选用固定 `K=2, n_init=100, random_state=0` 对其
31-point normalized-rank curves 聚类；两模型簇用 2x2 Hungarian assignment 最小化 patient-train prototype 的
`D_prof` 后命名 A/B。该 assignment、两个簇的支持数和 KMeans inertia 全部入 artifact。若重复 `random_state=0--9`
的 consensus AMI 中位数 `<0.8`，候选按 mode-unstable fail，不允许靠某一次标签排列进入连续目标。

`D_dist` 的 embedding 只在 patient train 的 31-point event curves 上拟合：标准化参数和 PCA basis 冻结，保留达到
95% variance 的最小维数并封顶 8 维；sliced Wasserstein 使用 64 个 hash-locked unit directions。模型只能 transform，
不得 refit。`D_JS` 使用自然对数和每个 mode `0.5` 的固定 Jeffreys pseudocount；它自己的 floor 同样来自 patient-train
recording-block split-half。

每一项都用 recording-block split-half patient floor 转成 excess-noise 单位：

```text
E(D) = max(0, (D - floor_median) / (floor_p95 - floor_median + eps))
D_k = 1/4 * (E_rec,k + E_prec,k + E_prof,k + E_dist,k)
```

mode proportion 用带固定 pseudocount 的 `D_JS(pi_model, pi_patient)`。最弱模式保护为：

```text
LSE_tau(D_A,D_B) = tau log[(exp(D_A/tau)+exp(D_B/tau))/2]
tau = 0.25

J_rev9 = E(D_global)
       + 2.0 * LSE_0.25(D_A,D_B)
       + 0.25 * E(D_JS)
       + 0.10 * R_edge
```

`R_edge` 是 median/p99 target KL 的冻结归一组合。全局目标仍固定 20 events；mode 目标固定 32 events，且每簇至少
10 events。候选不足时先走 usable count / minority support feasibility key，不给一个看似很好的连续分数。

hard gate 继续要求：两个 matched cells >0、两个 crossed cells <0、每簇支持、无 simulation error、无 runaway/tonic。
这些 gate 不替代 `J_rev9`，也不因某个一般性反向模式而算成功。

## 10. 优化层级与 seed 合同

1. **rev9-A engineering calibration：** seeds `901--906`，只选择 `alpha*`，不读取 patient held-out。
2. **rev9-B causal factorization：** seeds `911--922`，四主臂加 lesion/d audits；参数冻结，不再选择。
3. **rev9-C patient development：** 仅 patient train，可评估 §9 目标但不能声称 blind。
4. **rev9-D formal patient optimization：** 只有新 blind unit 存在才开启；新的 fit/selection/final seeds 和 patient
   recording units 必须在运行前写入 hash-locked config。

一到两个 edge 参数优先用可穷举、可配对的 deterministic grid，不使用 CMA-ES。只有参数维度扩展且 grid 不再可行时，
才另立 optimizer spec。

## 11. 与相图和有限发作方程的接口

rev9 输出的 `h_i + alpha* (+ beta*) + connectivity transform` 是静态 substrate bundle。它不替换现有 slow equations：

- `D_i=1-Z_i` 仍是 entry/permissivity slow field；
- presynaptic `a_X` / 已有 adaptation 候选仍负责 relay、offset 与 termination；
- `h/alpha` 不得改名为 Z、M、q 或 adaptation；
- 后续相图可把 `alpha_edge` 当 substrate coordinate，把既有 slow coordinate 当 dynamic coordinate，但正式点必须再跑
  finite-pulse lifecycle，并分别标记 runaway、tonic、brief dip、bounded carrier 和 recovered return。

handoff bundle 必须含：field/reference/config SHA256、NE 顺序、h、alpha/beta、每个 delay bin 的 topology/delay hash、
incoming-E conservation、KL/ratio/ESS、paired local-response calibration 和 node/edge arm结果。

## 12. 接受门、停止规则和允许表述

### 可进入 lifecycle handoff

- edge engineering hard gates 全过；
- edge-only matched-local-response 通过；
- Node/Edge/Node+Edge 在 paired seeds 上有可解释、稳定的差异；
- component lesion/relocation 与 d audit 能区分几何、异质性和配准；
- 无证据来自重新使用旧 held-out 选参。

### 停止

- primary edge family 在畸变上限内无法匹配 node：记录 family insufficiency，禁止直接开 beta 以外的高维搜索；
- beta 与 alpha 在 paired response 上不可辨识：退回 single-parameter family；
- edge 只增加 rate/event count，不保持局部 response、模式或 return：关闭 edge-core 路线；
- lesion/relocation 对 mode 与 onset 无位置特异性：不得称 Gaussian peaks 为 mode-specific cores；
- 没有新 blind unit：不得给 rev9 patient generalization claim。

### 允许写

- “同一冻结 field 的 node 与 edge 实现达到/未达到局部响应等效”；
- “模式起点与某分量相关，且 lesion/relocation 支持/不支持位置特异性”；
- “edge redistribution 改变了 mode-specific readout，同时保持 topology、delay 和 target incoming-E 总量”。

### 禁止写

- “连接性恢复了真实病人 core”而没有新 blind validation；
- “守恒 incoming-E 就证明动力学等效”；
- “静态 edge core 产生了完整发作生命周期”；
- “rev9 独立重新发现了患者传播轴”。
