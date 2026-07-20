# 审阅结论：MZ spatial P=3 bounded CCO 与 fast hand-off

日期：2026-07-20

工作线：`.worktrees/topic4-mz-divisive-lifecycle`

并行边界：本节点沿 current-based additive Z–M 路线，固定 Stage 0C shared divisor 与 canonical M3B 空间核；没有修改 E→E weight/kernel/delay，没有加入 recurrent conductance saturation 或 presynaptic relay。并行 `topic4-mz-conductance` 线的 recurrent-only conductance/saturation 结果不在本节点复用或调参。

## 1. 一句话判断

**P=3 找到了数学上有界、空间局限的 `CCO` fast cycle，但没有找到单核到邻环的 fast hand-off。**预置 core+annulus cycle 的四个相位在 base/half-dt 下均闭合；相反，`CLL` 与 `LCL` 四相位全部回到 `LLL`。所以缺口已从“没有目标周期态”收缩为“目标态存在，但单 patch 快耦合进不去”。

## 2. 完成程度

> **完成度：100/100（仅针对 mass-balanced P=3 frozen-fast gate）**

已经完成：

- exact `core / equal-area annulus / far bath` 三分区，far bath 不删除、不重新归一化；
- `K1=1` 与 `f^T K=f^T` 质量守恒；
- `LLL/CLL/LCL/CCL/CCC`、四个 relative phases、反对称扰动；
- fixed M3B coupling 与 `K=I` 对照；
- base/half-dt、20 s directed returns、Poincaré closure、period/peak drift、ceiling occupancy、transfer support 与 state bounds；
- canonical PNG/PDF、CSV/JSON/NPZ、中文 README 与单元测试。

这里的 100 分不表示完整 lifecycle 完成。`z/p/m` 仍冻结，区域 Z 入口、additive M 退出与 autonomous slow loop 均不属于本 gate。

## 3. P0 / P1 关键问题

### P0 已修复：瞬时 `peak>=120 Hz` 不能等同于 runaway

旧 classifier 把 core 瞬时峰值约 `136 Hz` 直接标成 `unbounded_or_saturation`。但这条轨迹有稳定 directed returns、极低 period CV、闭合 Poincaré section，且 `>120 Hz` 只占任一 100-ms 窗的最多约 `14%`，不是持续 refractory ceiling。

**修复**：动态有界性与生理 ceiling 拆开：

- cycle 至少 6 returns，丢弃前 2 个；
- recent period CV `<=1%`；
- recent peak drift `<=1%`；
- scaled Poincaré closure `<=2e-5`；
- half-dt closure 必须优于 base-dt；
- sustained ceiling 单独定义为任一 100-ms 窗中 `>120 Hz` 至少 80 ms。

`2e-5` 不是放宽科学门，而是覆盖 registered Euler + linear-section interpolation 的实测数值底噪：base-dt closure 为 `0.48–1.45e-5`，half-dt 为 `0.29–3.50e-6`，并随步长减半继续下降。

### P1 仍开放：fast spatial entry path 缺失

fixed K 下四个 `CLL` 与四个 `LCL` 相位全部回到 `LLL`；annulus 没有在 core 退出前建立 returns。`K=I` 下这些局部 sheet 越出 transfer support，不能当作生理 recruitment。

**怎么改**：不调 E→E。只允许最后一个 frozen oracle：让 core 与 annulus 的 Z 一起作为 regional control coordinate，检验 low branch 是否通过真实 fold 消失并自然进入已存在的 `CCO` basin；随后只在 recruited core+annulus 打开 additive A，检验 established cycle 能否退出。

## 4. 科学性问题

### 4.1 空间约化是质量守恒的

离散面积为 `113/112/2079` cells，即：

\[
f=(0.049045,\ 0.048611,\ 0.902344).
\]

固定矩阵为：

\[
K_{EE}=\begin{bmatrix}
.779680&.195566&.024754\\
.197312&.515382&.287305\\
.001345&.015478&.983177
\end{bmatrix},
\]

\[
K_I=\begin{bmatrix}
.867570&.131499&.000931\\
.132673&.682993&.184333\\
.000051&.009930&.990019
\end{bmatrix}.
\]

row-sum 与 stationarity 最大误差分别为 `4.44e-16` 与 `2.22e-16`。因此邻环耦合不是截取后人为 row-normalize 出来的。

### 4.2 预置 CCL 是 bounded localized fast-cycle candidate

四相位、两步长合并范围：

| 指标 | 结果 |
|---|---:|
| core / annulus returns | `29–30 / 29–30` |
| period | `670.48–670.82 ms` |
| core period CV | `3.4e-8–2.4e-7` |
| recent core peak drift | `0.116–0.231%` |
| core / annulus peak | `135.89–136.29 / 88.49–88.86 Hz` |
| bath peak / returns | `8.45–8.59 Hz / 0` |
| Poincaré closure | `2.9e-7–1.45e-5` |
| sustained ceiling | `0/8` forks |
| support / bound failures | `0/8` forks |

因此安全标签是 **bounded localized `CCO` fast-cycle candidate**，不是 seizure lifecycle，也不是 physiological amplitude acceptance。

### 4.3 有状态不等于能进入该状态

`CCL` 是 imposed initial sheet；它证明 attractor-like fast orbit 存在。`CLL→LLL` 则证明单核 cycle 不能靠当前快耦合把邻环接力起来。两者必须同时报告：

\[
\text{target fast orbit exists}\quad\neq\quad
\text{endogenous fast recruitment exists}.
\]

这排除了继续盲扫 `Amax/tau_p/tau_up` 的合理性，但保留 regional-Z slow entry 这一条不同机制的检验空间。

## 5. 工程性问题

- 16 forks 向量化，单进程、单 BLAS 线程；
- 两个 spatial arms × base/half-dt × 20 s，总 wall `6:49.6`；
- peak RSS `266528 kB`，0 swap；
- return crossing 同时保存线性插值的完整 continuous state，用于 Poincaré closure；
- transfer support、natural state bounds、finite 状态逐 patch 审计；
- 上游三个输入 SHA-256 fail-closed；
- `K=I` 只删除 cross-zone synaptic coupling，shared pool 与真实面积保持不变；
- 图已目视检查；三块区域、四类 seed 与对照表均可读。

## 6. 最小修改路线

1. 冻结 P3：bounded CCO existence = positive，fast `CLL→CCL` hand-off = negative。
2. 在同一固定 K 上求 `z=(z_R,z_R,.90)` 的 6D rate nullcline 与去掉冻结慢变量后的 23D fast Jacobian。
3. 用增广方程 `F=0, D_rFv=0, ||v||=1` 区分 real fold、Hopf 与 basin switch。
4. 若 low history 跨边界进入 bounded CCO，再做 event-locked `A=(A_R,A_R,0)` delayed exit；必须保留 matched `A=0` twin。
5. 只有 entry、established-cycle exit、参数恢复后同一 LLL basin 三者均过，才允许一个最小 autonomous Z/M slow latch。

## 7. 下一步建议

**GO 到 regional-Z entry/exit frozen oracle；NO-GO 到 slow-time-constant 网格、coarse field 和 SNN 移植。**这一 oracle 与并行 conductance/relay 线互补：本线不改 E→E，只问原 current-based fast topology 是否已具有可被 Z/M 慢变量穿越的入口和退出边界。

## 8. 产物

- 图：`results/topic4_sef_hfo/mz_spatial_p3_front_handoff/figures/mz_spatial_p3_front_handoff.png`
- summary：`results/topic4_sef_hfo/mz_spatial_p3_front_handoff/p3_front_handoff_summary.json`
- outcomes：`results/topic4_sef_hfo/mz_spatial_p3_front_handoff/p3_front_handoff_outcomes.csv`
- traces：`results/topic4_sef_hfo/mz_spatial_p3_front_handoff/p3_*_traces.npz`
- config：`config/topic4_mz_spatial_p3_front_handoff.yaml`
