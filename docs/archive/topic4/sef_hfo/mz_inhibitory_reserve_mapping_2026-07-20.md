# 审阅结论：MZ inhibitory-reserve mapping 与 exact periodic hold

日期：2026-07-20

工作线：`.worktrees/topic4-mz-divisive-lifecycle`

## 1. 一句话判断

**R0 的二维 fixed-q corridor 是真阳性，原线性 reserve 的 hold 也是可解的；但同一个 reserve law 无法满足锁定 sparse-event schedule 的 last-only entry，因此当前 autonomous lifecycle 必须保持 NO-GO。**问题已经从“有没有 bounded ictal state”进一步收缩为“event-to-entry eligibility 的慢流结构不对”。

## 2. 完成程度

> **当前 inhibitory-reserve 诊断完成度：86/100；完整 lifecycle 仍未通过**

已完成：

- `.835-.845` 五节点 fixed-q safe corridor 的双 dt/四 phase formal confirm；
- 完整 8-return `Ubar_CCO` 测量；
- `q_hold -> q_res,tau_D,d` 的唯一/单调 root mapping；
- 锁定六事件 replay 的 fail-closed entry-ordering 检查；
- 24 个 `q_hold x phase x dt` exact periodic q hold oracle；
- 线性 H-lowpass 与 `U H` 两个直接补丁的解析/数值关闭；
- thresholded eligibility 的 3x3 scalar screen 与严格 no-root/JSON 审计；
- 原线性 reserve 的 recovery-timescale 事件映射 pilot，以及 entry/exit handoff 的新 R2 锁定设计；
- 全过程未触碰并行线的 E→E conductance、saturation、relay 或 connection geometry。

尚未完成：

- state-dependent fast+q closed loop；
- recovery-timescale R2 formal scalar continuation 与短 coupled arm；
- M 打开后的 `(q,m)` return-map Jacobian、termination、reset/retrigger；
- bath mask 移除后的 local field recruitment/containment；
- full SNN migration。

## 3. P0 / P1 关键问题

### P0：`tau_r=20 s` 的原线性 reserve 不能同时完成 entry 与 hold

锁定方程为：

\[
\dot q={q_0-q\over\tau_r}-{(q-q_{res})U\over\tau_d}.
\]

三个 registered hold nodes 均有唯一、单调、物理合法且局部吸引的 mapping root：

| q_hold | Ubar_CCO | tau_D,d (ms) | q_res | pre-last minimum q |
|---:|---:|---:|---:|---:|
| .8400 | .19946 | 246.008 | .836300 | .854568 |
| .8425 | .19075 | 218.560 | .839206 | .854325 |
| .8450 | .18097 | 189.815 | .842116 | .854052 |

entry fold 为 `.8558315843`。三个 replay 都在第 5 个事件后约 `7.60 s` 已越 fold，而最后一个锁定事件在 `10.915 s`。因此 mapping status 是：

```text
RESERVE_MAPPING_CLEAN_NO_GO_LOCKED_EVENT_ORDERING_CONFLICT
```

禁止通过换 seed、改 final target、放宽 pre-last margin 或选另一个 q_hold 抢救。这个
no-go 的严格边界是**已注册的 `tau_r=20 s` 节点**；它不是对所有恢复时间常数的
全局不存在证明。

### P0：q_res 不是安全下界

`q_res` 是参数 floor；系统安全性必须由真实 `min_t q(t)`、periodic ripple 与 `(q,A)` path 验收。`.8275` 还是 source-unresolved，不能写成 confirmed safe point。formal fixed-q strip 仍是 `.835-.845`；`.825/.830` 只是 failure/safe anchors。

### P1：当前 sensor 仍是 frozen replay

event 与 CCO 的 `U(t)` 都在 q/M frozen 时生成。它适合做 cheap scalar falsification，却不能证明 q 下降后 fast event、U 与空间范围仍不变。即使 scalar gate 通过，也必须再做 closed-loop state-fork。

### P1：机制含义必须锁定

当前 `q/D_I` 最多表示突触前可释放 GABA reserve 的粗粒化代理或 effective inhibitory capacity。它不能同时代表 chloride accumulation 或 KCC2 recovery：前者改变 GABA conductance amplitude，后两者主要改变 `E_GABA`。若未来加入 chloride，必须另设离子状态而不是继续乘一个 q。

## 4. 科学性问题与动力学反思

### 4.1 什么做对了

R0b 证明失败边界确实是二维 `(q,A)` 几何，而不是一维 q support threshold。`.835-.845` 内，原 225-ms occupancy-gated M smooth ramp 在 40/40 formal arms 中回 LLL，且 `.025 mV` registered margin、parameter restoration、双 dt/四 phase、zero fail-closed violations 全通过。

这说明 additive M 的符号和 exit 方向没有错。旧 autonomous latch 失败，是因为 q 继续下降使 `A_exit(q)` 移动；固定 q 后 M 可以穿过 exit fiber。

### 4.2 exact periodic oracle 排除了哪个替代解释

对 hash-locked frozen CCO sensor 做 exact piecewise-constant q update，24/24 组合全部收敛：

```text
global q range = .839443-.845664
max |period mean - q_hold| = 4.74e-5
max q-direction rho per period = .63910
max stroboscopic error = 5.55e-14
```

所以 current reserve law 不是“无法维持 CCO”；它能形成稳定 q hold。真正失败的是 sparse-event entry ordering。这个区分很重要：继续调 hold nullcline不会修复 entry。

### 4.3 为什么简单再加一个慢变量仍不够

若只加：

\[
\tau_H\dot H=U-H,
\qquad
\dot q={q_0-q\over\tau_r}-{(q-q_{res})H\over\tau_d},
\]

则 `H*=Ubar(q)`，平均 q-nullcline与原式相同。其二维平均 Jacobian trace 恒负；在当前 measured `U_q` 下 determinant 为正，且 divergence 恒负，因此这个 H 不提供新 branch 或外层周期轨道。`tau_H=1 ms-10^7 ms` 的 locked replay 也 0 个可行点。

若把 depletion 改为 `U H`，只有 `tau_D,d<=4.5 ms` 才能通过 ordering；这已不是慢变量，并在大 tau_H 下退化为 `tau_H tau_D,d` 的不可辨识缩放。该臂关闭。

### 4.4 thresholded eligibility 已完成：有选择性，但没有鲁棒区域

执行的候选为 thresholded event eligibility：

\[
\tau_H\dot H=U-H,
\]

\[
g_H={1\over2}\left[1+\tanh{H-\theta_H\over .002}\right],
\]

\[
\dot q={q_0-q\over\tau_r}
-g_H(H){(q-q_{res})U\over\tau_d}.
\]

它改变的不是 fast E→E gain，而是“event use 何时有资格耗竭 reserve”。中心
`(tau_H=10 s,theta_H=.020)` 确实得到唯一 nested root
`tau_d=110.141 ms,q_res=.840759`，并通过 nominal 第六事件 entry、periodic hold、
dense/sparse/isolated schedule probes 与 U=0 recovery。但 `theta=.021` 已变为
`no_entry`；8 个有 root 的 cells 中 theta sensitivity 为 `0/8`，safe cells 为 0。
右下 cell 是注册物理域内单调但无零点，不是数值 unresolved。修补后 status 为：

```text
THRESHOLDED_INHIBITORY_ELIGIBILITY_SCALAR_CLEAN_NO_GO_REGISTERED_ROBUSTNESS_GATES
```

因此该臂正式关闭，不进入 coupled/autonomous 仿真。它证明 eligibility 可以制造
interval selectivity，但当前 sigmoid 把转换压在过窄的参数窗口内。

### 4.5 反思后的关键漏洞：20-s recovery 把事件顺序翻转

锁定六事件中，event 5 到 event 6 的间隔最长（`3.384 s`）。原设计把
`tau_r=20 s` 当作固定常数，导致这段恢复足以使第 5 次事件的 q 谷值比第 6 次更深。
一个不落盘 pilot 保持原方程、每个节点重新由 CCO hold 和 unchanged `.855` endpoint
解 `q_res/tau_d`，只延长 `tau_r`，得到：

| tau_r | nominal ordering | pre-last min q | approximate q=.885 wait from q_hold |
|---:|---|---:|---:|
| 20 s | event 5 premature | .85433 | 26.9 s |
| 40 s | event 6, margin不足 | .85638 | 53.8 s |
| 60 s | target event 6 | .85711 | 80.6 s |
| 80 s | target event 6 | .85748 | 107.5 s |
| 90 s | target event 6 | .85761 | 120.9 s |
| 100 s | target event 6 | .85771 | 134.4 s |

代码复核后，handoff 不是“退出后 M 立刻按 12 s 衰减”。现有 state-defined latch
在 low occupancy、reset 前令 `dM/dt=0`；只有 low+p-off+`q>=.885` 后才 reset 并
释放 M。因此 entry 下界来自最长 quiet gap，而 operational 上界来自既有 120-s
recovery/retrigger horizon。pilot 预示 `60-80 s` 可能是有限宽度 corridor。这仍是
pilot-informed hypothesis，不是正式结果；其正式轴和 stop rules 已锁在
`docs/superpowers/specs/2026-07-20-topic4-mz-inhibitory-reserve-recovery-corridor-design.md`。

动力学上，延长 `tau_r` 不会创造新 q fixed point 或 Hopf。它改变的是有限事件映射；
若 coupled arm 成功，完整图景应是 fast CCO 小环嵌在 `(q,M)` 慢 relaxation loop 中，
而不是宣称 q 自己产生极限环。

### 4.5 文献边界

- [Zhang et al., 2012](https://doi.org/10.1523/JNEUROSCI.4247-11.2012) 直接支持 preictal presynaptic GABA release exhaustion 及秒级恢复，但不支持当前具体线性 law 或固定 q_res。
- [Lillis et al., 2012](https://doi.org/10.1016/j.nbd.2012.05.016) 支持 GABA barrage 驱动 chloride accumulation；这应移动 `E_GABA`，不能用当前 q 乘法冒充。
- [Buchin et al., 2016](https://doi.org/10.1523/JNEUROSCI.4228-15.2016) 支持快放电轨道嵌在二维慢离子边界中的图景，与当前 `(q,A)` corridor 相符，但其 Cl/K/KCC2 模型不能压成一个 q 后仍声称复刻。
- [Krishnan & Bazhenov, 2011](https://doi.org/10.1523/JNEUROSCI.6200-10.2011) 支持“大慢环包住快 burst 小环”和独立 termination variable；它不证明 inhibitory reserve 本身能终止 seizure。

## 5. 工程性问题

核心产物：

- mapping：`results/topic4_sef_hfo/mz_inhibitory_reserve_mapping/`
- exact periodic oracle：`results/topic4_sef_hfo/mz_inhibitory_reserve_periodic_oracle/`
- mapping 主图：`figures/mz_inhibitory_reserve_mapping.png`
- periodic 主图：`figures/mz_inhibitory_reserve_periodic_oracle.png`

mapping runner 已锁 R0b summary/sentinel hash、accepted q interval、完整 Cartesian product、唯一 root/单调性、support/bound/nonfinite 与图前中间产物落盘。canonical run 用时 `9:02`，峰值 RSS 约 `286 MB`，单进程/单 BLAS 线程、0 swap、无 OOM。

当前相关 reserve/corridor/mapping/periodic/latch tests 至少 30 个通过；最终提交前仍需跑一次合并测试集和 `git diff --check`。

## 6. 最小修改路线

1. 保留 `tau_r=20 s` linear-reserve 与 thresholded-eligibility 两个 clean no-go；
2. 按锁定轴运行 `tau_r=[20,40,50,60,70,80,90,100,120,160] s` exact scalar continuation；
3. 每个 `(tau_r,q_hold)` 仍只由 CCO hold 与 locked final target 解 `q_res/tau_d`，不能用 entry/handoff gate 反调；
4. 同时检查 complete phase/dt periodic orbit、base/half event ordering 与 R0b low-fold + `.025 mV` postictal handoff；
5. 只有至少 3 个连续 tau nodes 在全部 3 个 q-hold 节点通过，且包含 80 s，才运行 `[70,80,90] s` 短 coupled arm；
6. coupled arm 后仍需真正的 termination、same-basin recovery、early/late retrigger 和动态 local field gates，才能恢复三条下游 workflow。

## 7. 下一步建议

**thresholded eligibility 已 CLEAN NO-GO；GO 到 recovery-timescale R2 scalar continuation；NO-GO 到 autonomous lifecycle、field 和 SNN。**

当前最安全的核心结论是：

> MZ fast system 已有 bounded localized CCO 与可恢复 exit corridor；`tau_r=20 s` 的原线性 inhibitory reserve 能 hold cycle，但因最长事件间隔内恢复过快而在第 5 次事件提前越界。硬阈值 eligibility 能修正 nominal ordering，却没有鲁棒参数区域。下一最小节点先检验有限的 q-recovery timescale corridor，而不是增加 E→E gain、另一个电流或继续微调 sigmoid threshold。
