# Stage 0B 均匀 E/I 快拓扑筛查简报

## 一句话结论

**CLEAN NO-GO：当前复用 M3B/LIF 的均匀六变量 E/I 快子系统，确实存在低态与高态的双稳区，但高态是 179.35--304.32 Hz 的过高率/饱和支；没有找到可由有限初值进入、低于 100 Hz、并在长窗确认中保持有界的 ictal 高态对象。** 因此按预注册停止规则，Stage 1--3 保持关闭，不接 Z、recovery 或空间场。

## 锁定合同

- 参数轴：`w_ee_mult = 1.0, 1.1, ..., 1.5`；`q = 1.00, 0.99, ..., 0.80`；共 126 点；`ratio=1.0`。
- 无 noise、无 slow variables、无 spatial coupling、无 dynamic threshold `phi`。
- 状态为 `rE/rI/sEE/sEI/sIE/sII`；复用 M3B/LIF 常数和 transfer。
- 与既有局部线性 `field_rhs` 不同，本筛查在每次 RHS 调用中由当前 synaptic states 自洽重算 `muE/sigmaE/muI/sigmaI`。
- dense multistart root search 沿 q 单方向 warm continuation，并复用相邻 `w_ee_mult` 的同-q roots；它不是 bidirectional continuation。state forks 包含低/边界/高平衡流形初值、每个 root 的双侧扰动，以及 4 类 synaptic-history off-manifold 初值；初筛 `dt=0.25 ms, T=6 s`。
- 4 类 off-manifold 初值为 E-synapse-loaded/I-low、I-loaded/E-low、rate-high/synapse-low、rate-low/synapse-high；全部在 126 点的初帧处于 LUT 支持内。
- 只有非 exact 初值产生、尾窗低于 100 Hz 的 bounded candidate 才能进入 `dt=0.125 ms, T=12 s` confirm；本轮没有任何对象满足，因此 confirm 数为 0，`phi` arm 也不开放。

## 主要结果

### 1. Root topology

共找到 200 个自洽 root：

- 51 个 stable low roots，E 率 0.248--2.516 Hz；
- 37 个低于 100 Hz 的 unstable separators，E 率 1.017--87.889 Hz；其中 19 个位于 5--100 Hz，虽然数值上是“finite-high root”，但全部为实特征值不稳定，不能作为 ictal attractor；
- 111 个 stable over-100-Hz roots，E 率 179.348--304.323 Hz，中位 303.525 Hz；
- 另有 1 个 over-100-Hz unstable root。

按 `w_ee_mult` 分层的 q 边界为：

| `w_ee_mult` | stable low q | unstable separator q | stable >100-Hz q |
|---:|:---:|:---:|:---:|
| 1.0 | 0.83--1.00 | 0.83--0.88 | 0.80--0.88 |
| 1.1 | 0.88--1.00 | 0.88--0.98 | 0.80--0.97 |
| 1.2 | 0.92--1.00 | 0.92--1.00 | 0.80--1.00 |
| 1.3 | 0.95--1.00 | 0.95--1.00 | 0.80--1.00 |
| 1.4 | 0.97--1.00 | 0.97--1.00 | 0.80--1.00 |
| 1.5 | 1.00 | 1.00 | 0.80--1.00 |

这说明问题不是“系统没有第二个 basin”，而是**第二个 basin 的目标态错误**：随着 recurrent E 增强或 inhibition efficacy 降低，低态跨过不稳定 separator 后直接落到过高率支，而不是有限 ictal branch/orbit。

### 2. State forks

共运行 1986 条轨迹，其中 200 条 exact-root 轨迹只用于方程/Jacobian parity，不参与 basin verdict。其余 1786 条动态 probes/root perturbations/off-manifold probes：

- 450 条回到 low fixed point；
- 1336 条进入 over-100-Hz branch；
- 0 条 bounded tonic candidate；
- 0 条 bounded oscillatory candidate；
- 0 条 long transient、bounded-indeterminate 或 numerical divergence。

其中 504 条 off-manifold probes 单独计数为 123 low + 381 over-100-Hz，四类初值均没有打开隐藏的有限 orbit/basin。

exact 初始化在不稳定 separator 上可能因浮点残差短时看似 tonic，因此正式 verdict 明确排除 `initial_kind=exact_root`，只结合非 exact state forks 与 root stability 判定。

### 3. LUT clipping 审计

M3B `_phi_field` 会把 `mu/sigma` 裁剪到 LUT 支持范围。当前 runner 对每个保存帧和尾窗输出 clipping occupancy 及四个 moment 的极值，并规定任何依赖 clipping 的 finite candidate 必须判 invalid。

- 所有 low-tail 的 clipping occupancy 均为 0；
- 所有 37 个低于 100 Hz 的 unstable separators 均位于 LUT 支持范围内；
- 111 个 stable over-100-Hz roots 中，9 个仍完全位于 LUT 支持内，102 个超出 LUT 支持；
- 任一 putative finite candidate 只要在任一 **audited saved frame** 发生 clipping 就必须 fail；本轮没有 candidate 被 clipping 掩盖。这里的审计采样间隔为 5 ms，不能写成连续时间的“ever touched”。

为排除高支定量依赖 LUT 的歧义，又对全部 200 个 LUT-discovered roots 做了未裁剪 exact Siegert 局部 refinement：200/200 收敛；37/37 个 source sub-100-Hz unstable roots 仍不稳定；111/111 个 source stable over-100-Hz roots 仍稳定且 >100 Hz；0 个 stable finite-high root。exact 高支 E 率为 179.355--460.119 Hz（中位 427.153 Hz）。该审计是对已发现 roots 的局部复核，不是另一轮 dense exact continuation。

因此 LUT clipping 会影响多数极高率平台的精确数值，不能把 LUT 平台的 303 Hz 当作生物物理定量值；但未裁剪复核反而把多数高支推得更接近 refractory ceiling，且没有制造有限高态，不改变停止结论。

## 验收与资源

- 定向合同测试：12/12 passed，包括 root/RHS 同方程、全 sigma Jacobian directional parity、ceiling/long-drift 拒绝、exact-root 排除、4 类 off-manifold 覆盖与全网格初帧 LUT-valid、LUT-clipped candidate fail-closed、exact audit fail-closed、CLI 显式确认门。
- 正式运行：单进程、BLAS 单线程；wall time 5 分 16.74 秒；峰值 RSS 0.221 GiB，低于 4 GiB 合同；无 swap 增量。
- 主摘要：`stage0b_summary.json`。
- root 证据：`root_continuation.json`、`root_table.csv`。
- exact sensitivity：`exact_siegert_root_audit.json`、`exact_siegert_root_audit.csv`。
- state-fork 证据：`state_fork_screen.json`、`state_fork_screen.csv`；confirm 文件为空数组，表示没有合法候选进入确认，不是漏跑。
- 本 Stage0B 子任务没有生成图，因此本目录不创建空的 `figures/` 或占位 `figures/README.md`；主线程的 current-stage paper-ready figure 是另一项独立产物。

## 下一步边界

本结果只是否定当前参数轴内、当前 homogeneous M3B/LIF fast equation 的有限 ictal object；它不否定其他 E/I transfer、显式 voltage/reversal dependence、额外 fast/intermediate feedback 或并行 FCXR conductance 线。当前 topology-first 路线不应继续接 slow loop 或空间层；下一轮若更新 fast equation，首先必须改变高支拓扑，再从 Stage 0B 重新验收。
