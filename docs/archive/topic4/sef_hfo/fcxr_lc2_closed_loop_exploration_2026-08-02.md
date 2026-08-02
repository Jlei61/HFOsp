# FCXR-LC2 closed-loop mechanism exploration：完整收口

日期：2026-08-02  
分支：`codex/topic4-fcxr-lc2`  
最终层级：`COMPLETE_BOUNDED_NEGATIVE`

## 0. 结论先行

本轮完整跑完了 R1 重新表征、90 格 H 高初值 screen，以及两组正式候选的 40k frozen H/X forks。最终判决为：

```text
H_FROZEN_GEOMETRY_NO_GO_HEALTHY_BASELINE_DISTURBED
```

这不是说 H 不能产生高态。相反，测试的局部 H 正反馈很容易产生**数值有界、非硬截断的持续高态**。问题是它在健康抑制状态 `D=0`、从 `H=0` 出发也会被正常间期事件逐步点燃；因此没有得到“健康时保留间期、抑制耗竭时才出现低/高双 basin”的选择性几何。两档来自 LC1 的冻结 X 负荷也只轻度降率，没有把这个高态送回间期工作点。

所以 dynamic Z/H/X pilot 没有解锁。当前没有测试、也不能声称已经得到无 kick 的 onset–offset–returning-IED lifecycle。

## 1. 本轮真正问的问题

本轮只测试一条机制链：

```text
局部 recurrent-drive persistence H
  -> RC1 上的有限低/高 basin
  -> 已验收 X 负荷消灭高 basin
  -> dynamic Z/H/X 无 kick 进入、退出并回到 IED 邻域
```

锁出范围的机制包括 `M/K/A/ELR`、新 E→E 边、全局 seizure label、kick、hard reset 和参数 step。1 秒 screen 只负责找延长候选，不能直接给双稳态或发作结论。

## 2. R1：旧 HEO2 窗口被正确拆开

旧 HEO2 的 1500–4500 ms 不是一段连续高态，而是两个 active bouts，中间隔着一个 1066 ms 的低活动 gap。去掉两端各 50 ms 后：

- gap 的 50 ms population-rate median 为 `0.02125 Hz`，低于 RC1 inter-event q95 `20.7927 Hz`；
- 15/15 个虚拟触点都回到低参考的 `±3 dB` 内；
- 因此该 gap 合法标为 `rest_like`；
- HEO2 active bouts 中 recurrent-drive support 为 `528/4096 = 12.890625%` 的采样 E 细胞，覆盖 `13/16` 个空间块。

完整离线表征得到 100 个传感器点、31 个 Pareto 点，并按锁定角色选择 6 个候选。允许的结论是：持续 HEO1 控制与 HEO2 active bouts 都提供了可分离的 H 时间尺度；不允许再把完整 HEO2 窗口当作连续高态。没有一个单一时间常数能同时零 false latch 并桥接完整长 gap。

## 3. E3：90 格 screen 发现的是有界限幅，不是 basin

对 6 个 R1 候选，扫描：

```text
k_H/theta_H = {0.05, 0.10, 0.20}
rho_H/g_sat = {0.10, 0.20, 0.35, 0.50, 0.70}
```

在 frozen `D=0.15`、统一 `H(0)=2theta` 下，90/90 全部完成：

| 标签 | 数量 |
|---|---:|
| `screen_survivor` | 52 |
| `saturated_tonic` | 38 |
| `decay_low` | 0 |
| `unresolved_1s` | 0 |
| `numerical_failure` | 0 |

尾段活动主要由 `rho_H/g_sat` 排序：弱档约 101 Hz，中档约 185 Hz，随后约 231、257、275 Hz。最窄 gate (`k/theta=0.05`) 在统一高初值下对六个传感器家族给出相同的饱和 H 驱动力；因此该 screen 对 tau/theta 的分辨力很弱。52 个 survivor 只说明 RC1 平滑饱和能把 H 正反馈限制成有限活动，不是独立高 basin 的证据。

## 4. E4：frozen geometry 的失败点在健康低态

按预注册排序和实际 40k 吞吐，只取两个正式候选：

- `H6_k05_r10`；
- `H6_k10_r10`。

每个候选运行 A-low/A-high/B/C/D1/D2 六臂；canonical 标签使用已验收 Stage-D 的 300 ms rolling-rate workpoint classifier，而不是任意 20 Hz 尾段阈值。

### 4.1 两个候选的共同结果

| arm | 科学问题 | 结果 |
|---|---|---|
| A-low：`D=0, H(0)=0` | 健康低态能否保留 | 两候选均变成 `FINITE_HIGH_ORBIT`；尾段 78.2 / 84.2 Hz |
| A-high：`D=0, H(0)=2theta` | 健康底座能否拒绝高初值 | 均保持 finite high |
| B：`D=0.15, H(0)=0` | 易感低 basin 是否存在 | 均变成约 101–103 Hz 的 finite high |
| C：`D=0.15, H(0)=2theta` | 易感高 basin 是否存在 | 均为约 102 Hz 的 finite high |
| D1：C + `D_X=0.128` | 已观察 X 负荷能否消灭高态 | 尾段约 97.6 Hz，未回间期 |
| D2：C + `D_X=0.214` | 更强已观察 X 负荷能否消灭高态 | 尾段约 94.9 Hz，未回间期 |

所有 12 条正式轨迹数值有限、零 hard clip、零 numerical failure，refractory-ceiling fraction 为零。因此这不是 runaway 或数值天花板造成的假阳性；它是一个真实的有界高活动解，但不是我们需要的 susceptibility-selective carrier。

提前运行、后续保留为 developmental corroboration 的 H1 两点也一致：A-low 不再是接受的间期工作点，而是 `ELEVATED_EVENT_TRAIN`；其余高初值/易感/X 臂均保持 finite high。它们不参与 canonical 判决，但支持失败并非 H6 单点特例。

### 4.2 机制解释

当前实现是：

```text
recurrent gA -> local EMA h -> sigmoid gate -> positive recurrent conductance
```

H actuator 本身不依赖 Z/depletion。正常 IED 已足以反复抬高 h；一旦跨过 sigmoid 区域，H 正反馈就继续增强，并在 RC1 tanh saturation 下落到有限高态。Z 因而没有真正成为“只负责 entry 的 onset coordinate”：即使 `D=0`，H 也能自行点燃。

这解释了为什么 B/C 没有显示双 basin：不是高态造不出来，而是低态先丢了。也解释了为什么不能直接把 dynamic Z/H/X 接回去——那只会测试一个健康底座也自燃的系统。

## 5. X 结论的边界

在这两个测试 H 分支上，冻结 `D_X=0.128/0.214` 只能把尾段率从约 102 Hz 降到 95–98 Hz，没有返回间期工作点。因此允许写：

> 已验收 LC1 量级的两档冻结 X 负荷，不足以消灭本轮测试的 H 高分支。

不能写成“X 没有终止 authority”。LC1 已经在另一种持续高态上验证过 X 的终止能力；本轮只是表明当前 H 正反馈相对该 X 量级过强，而且因为健康低态已经失败，X 结果不是一个完整、选择性 lifecycle 上的 offset 测试。

## 6. dynamic pilot 为什么没有运行

E5 的合法前置是：A-low/A-high/B 都回到间期工作点，C 保持有限高态，且 D1/D2 至少一臂回到间期。正式两个候选都在第一条 A-low 就失败，也没有任何 X-return arm。因此：

```text
dynamic_pilot_manifest.status = NOT_UNLOCKED
noise 402 = unopened
dynamic Z/H/X lifecycle = untested
```

这不是保守停机，而是避免用一个错误的健康基线去制造看似完整的 onset/offset 曲线。

## 7. 工程与资源验收

- E3：90/90；最多 4 workers；实测 peak RSS `6.793 GiB`。
- E4：2 finalists × 6 arms；最多 2 workers；实测 peak RSS `7.214 GiB`。
- 没有任何 `T>=20 s` 生产运行，因为 E5 未解锁。
- E3 watchdog 全程约 5.76 h，chain 到 E4 约 8.05 h；swap 相对启动最多只增加约 `0.5 MiB`。
- 所有长任务使用 `setsid nohup`、exact PID、stage flock、DONE/FAILED sentinel 和 wall/resource watchdog；任务结束后无残留仿真进程。
- 6 个 blessed engine 文件 sha256 与 `execution_lock.json` 逐条一致。
- 相关回归：`171 passed`。
- `M/K/A/ELR`、新边、kick、hidden confirmation seed 均未打开。

冻结 fork 的 canonical 结果由运行时脚本字节产生；该脚本在结果生成前已固定，随后原样提交为 `d8122f1c`。最终 `candidate_verdict.json` 同时记录脚本 sha256 和 blessed engine hashes。

## 8. 允许和禁止的科学口径

### 允许

- R1 active-bout/support/false-latch/gap-bridge 表征完成；
- H 高初值 screen 完成，但 survivor 只是开发标签；
- 测试的 H loop 能产生数值有界的 finite-high 解；
- 两个锁定正式候选都破坏健康 `D=0` 的间期工作点；
- 所测 frozen X 量级不能把该 H 分支送回间期；
- 这条具体 H architecture 在本轮锁定参数与候选上是 bounded-negative。

### 禁止

- “H 永远不能产生双稳态”；
- “X 普遍没有终止能力”；
- “已经测试或得到 Z/H/X lifecycle”；
- “已经得到 seizure carrier、limit cycle、bistability 或 E1146 phenotype”；
- 用 52 个 `screen_survivor` 代替 basin 证据。

## 9. 下一步唯一机制建议

不要继续沿同一个 `rho_H` 强度轴补 40k 点。下一版应先让 H 的**闭环增益对 Z/depletion 有选择性**：健康 `D=0` 时 loop gain 必须小于 1，易感 `D≈0.15` 时才允许大于 1。最小可检验形式是给 H actuator 加一个平滑、局部的 susceptibility gate（例如由 local inhibitory depletion 或其膜去极化后果驱动），而不是全局 seizure label 或硬开关：

```text
g_H,i = rho_H * S_H(h_i) * S_D(D_i)
```

下一轮先在 frozen `D={0,0.15}` 的低成本两臂 continuation 上证明“健康低态保留 + 易感高态出现”，再测试 X。若这一步仍不能形成选择性窗口，才关闭当前 sigmoid-EMA H family，转 short-term facilitation 或 intrinsic conditional bistability；不要再把时间预算花在同一架构的 40k 强度网格上。

## 10. 产物

结果根：`results/topic4_sef_hfo/fcxr_lc2_core/closed_loop_exploration/`

- `execution_lock.json`
- `r1_resegmentation_summary.json`
- `r1_sensor_pareto.csv`
- `r1_sensor_support_map.npz`
- `h_loop_screen.json`
- `frozen_fork_map.json`
- `candidate_verdict.json`
- `STATUS.md`
- `figures/r1_sensor_characterization.png`
- `figures/r1_sensor_pareto.png`
- `figures/h_loop_screen.png`
- `figures/frozen_fork_map.png`
- `figures/failure_taxonomy.png`
- `figures/README.md`

`dynamic_pilot.png` 未生成，因为没有真实 dynamic 输入；画占位图会谎报执行状态。
