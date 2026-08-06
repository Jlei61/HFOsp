# 审阅结论：MZ spatial P=2 frozen fast sheets

> **2026-07-20 P3 后续修正**：P=2 的 whole-complement `CL` negative 不能回流成“没有任何局部 bounded sheet”。质量守恒 P=3 已找到预置 core+annulus 的 bounded `CCO` fast cycle，但 `CLL/LCL` 仍不能 fast hand-off；最新判定见 `mz_spatial_p3_front_handoff_2026-07-20.md`。

日期：2026-07-20

工作线：`.worktrees/topic4-mz-divisive-lifecycle`

并行边界：本节点只使用锁定的 additive Z–M current line、Stage 0C shared divisor 与 canonical M3B `.54/.27/.25 mm` 空间核；没有修改 E→E weight/kernel/delay，没有加入 conductance membrane 或 presynaptic relay。

## 1. 一句话判断

**P=2 fast scaffold 本身正确，但当前结构没有可维持的 focal ictal sheet。**真实 coupling 下四个 `CL` 相位全部被压回 `LL`；拿掉跨区突触 coupling 后四个 `CL` 又全部越出 transfer support。也就是说，小范围 core 的 shared pool 太弱，而把完整低态 surround 平均进 core 的局部 coupling 又过强；当前两区约化在“失控”和“过度压制”之间没有中间 focal attractor。

## 2. 完成程度

> **完成度：90/100（仅针对 P=2 whole-sheet frozen-fast gate）**

已经完成：

- 从 canonical M3B API 固定导出 single-core/full-complement `K_EE/K_I`；
- 使用真实离散面积权重，不再把 4.90% core 当成 50%；
- 新增 `f^T K=f^T` 质量守恒 gate，避免 source-area double count；
- product-history lift：spatial K 先作用于 synaptic histories，全域只有一对面积加权 `mu_G/S_G`；
- `LL/CL/LC/CC` 四类 sheet、四个 cycle phases、反对称扰动、`K=I` 对照；
- base/half-dt (`.125/.0625 ms`) 20 s 积分；
- local directed `rE_fast=20 Hz` returns、transfer support、state bounds、`>100 Hz` occupancy；
- 图、CSV、JSON、NPZ、中文 README 和测试。

扣 10 分：full complement 把 95% tissue 压成一个均匀变量，不能解析紧邻 core 的 recruitment front；conditional entry/exit sheets 与 P=3 core–annulus–bath 尚未完成。

## 3. P0 / P1 关键问题

### P0 已关闭：P=2 不能默认 equal weights

canonical `48×48, L=12 mm` grid 中，single core 只有 `113/2304=.049045`。若用默认 `[.5,.5]`，会把 focal activity 对 shared pool 的贡献放大约 10.2 倍，直接改变 core 是否被 global divisor containment。

**修复**：正式矩阵显式保存 `f=[.049045,.950955]`；`PatchKernels.validate()` 现在同时要求 row sum=1 与 `f^T K=f^T`。

### P0 已关闭：不能把 adjacent front 截成 2×2 后重新归一化

若只保留 core 与等面积邻环，再把流向 far bath 的 kernel mass 重新塞回两列，会人工放大 cross coupling。正确最小局部 front 是：

1. core disk；
2. equal-area annulus；
3. far bath；

并保留完整 3×3 operator。当前 whole-surround P=2 因此只能作 exact patch-constant diagnostic，不能单独否定 wavefront。

### P1 开放：当前结构缺少 focal C sheet

固定 K 时 `CL→LL`；`K=I` 时 `CL→physical/numerical failure`。这不是“空间 coupling 完全无效”，而是两种作用方向没有重叠出可维持窗口。

**怎么改**：下一步不调 E→E，先把 surround 拆成 near annulus + far bath，检验 near patch 能否在 core 熄灭前获得 bounded returns。只有 P=3 仍无 local hand-off，才把 fixed current scaffold 判为 spatial recruitment no-go。

## 4. 科学性问题

### 4.1 锁定空间算子

精确离散约化为：

\[
K_{EE}=\begin{bmatrix}
.779680&.220320\\
.011363&.988637
\end{bmatrix},\qquad
K_I=\begin{bmatrix}
.867570&.132430\\
.006830&.993170
\end{bmatrix}.
\]

最大 row-sum/stationarity error 为 `2.22e-16/1.11e-16`。矩阵的强不对称不是手调参数，而是 core 与 full complement 面积相差约 19.4 倍后的质量守恒结果。

### 4.2 四类 frozen sheets

primary `dt=.125 ms`，half-dt 标签完全一致：

| 初态 | fixed M3B K | K=I（shared pool 保留） |
|---|---|---|
| `LL` | `LL` | `LL` |
| `CL`，4 phases | `LL`，4/4 | support failure，4/4 |
| `LC`，4 phases | `CC`，4/4 | `LC`，4/4 |
| `CC`，4 relative phases | `CC`，4/4 | `CC`，4/4 |

同步 `CC` 保持约 `33` 个 local returns/20 s，peak `98.15 Hz`；反对称扰动不破坏它。`LL` 保持 stable low root（`rE=.8109 Hz,rI=5.6786 Hz`）。所以 P=1 cycle 与 low basin 都没有因重写空间 RHS 而消失。

`CL` 的解释最关键：

- fixed K：core 的低态 surround 输入削弱 active core 的 recurrent history，四个 phase 均未完成一个 local return，收敛到 heterogeneous low（core `1.84 Hz`，surround `.835 Hz`）；
- K=I：core 不再被低态 surround 稀释，但它只占全域 4.9%，`p_pool=1` 下 shared divisor 太弱，四个 phase 均越出 transfer support；
- 因而当前没有“core bounded C、surround L”的中间 attractor。

`LC→CC` 不能当作局部 recruitment：这里的 `C` 已经占 95.1% tissue，既主导 shared pool，又通过 `K_core,surround=.2203` 驱动小 core。这只是约化算子的反向一致性检查。

### 4.3 对 nullcline / entry–exit 的含义

本结果已经给出一个直接的 basin 约束：在 `z_core=.85,z_surround=.90,A=0` 时，cycle-seeded focal core 位于 fixed-K low basin，而非 focal cycle basin；去掉 cross-zone coupling 又没有 bounded basin。因此 additive M 还没有开始 build 之前，现结构已无法提供 latch 所需的 `CL` 驻留窗口。

这也解释了为什么“再加一个 additive current”不够：M 的 exit 方向本身是对的，但 `dot m=0` 的 core-only 阶段必须先存活并招募邻区。当前失败发生在这个更早的 fast spatial step，不是 exit current 大小不足。

## 5. 工程性问题

- hot RHS 把 validate/prepare 移出循环，仍实时读取 local `z/m` 与唯一 shared pool；相对安全 oracle 的最大差异 `3.47e-18`；
- 17 个本节点 unit tests 通过；
- 15 forks 向量化，单进程、单线程；
- 四组 20 s base/half-dt run 总 wall `6:00.2`，peak RSS `256788 kB`，0 swap、无 OOM；
- 所有上游输入 SHA-256 fail-closed；
- invalid `K=I/CL` 分支逐 fork 记 support violation，不被误分类为 low；
- 图已目视检查，PNG/PDF 与中文 README 齐全。

## 6. 最小修改路线

1. 冻结本 P=2 结果；不把 `CL→LL` 写成 full spatial no-go。
2. 用同一个 fixed kernel 构建 `core / equal-area annulus / far bath` 3×3 mass-balanced operator，不重新归一化、不调 E→E。
3. 先跑 `CLL→CCL/CCC` 与 `K=I`；判断 annulus 是否在 core 首次退出前完成至少三个 bounded returns。
4. 同时求 P=3 autonomous low roots、fast Jacobian 与 cycle-seeded basin boundary，区分 nullcline shift、transverse instability和单纯 transient。
5. 只有 local hand-off 存在，才加入 Z/p/M：persistence AND near-recruitment set、Z-safe reset；否则本 additive line 停止。

## 7. 下一步建议

**GO 到 P=3 core–annulus–bath cheap gate；NO-GO 到任何 slow-latch 参数扫描。**当前应回答“邻环能否在 focal core 被低态背景压回前接力”，而不是先调 `Amax/tau_p/tau_up`。若 P=3 仍是 core extinguish 或 support runaway，说明当前 additive/shared-pool architecture 没有空间可维持的 ictal basin，应把该负结论交给并行 conductance/relay 线比较。

## 8. 产物

- 图：`results/topic4_sef_hfo/mz_spatial_p2_frozen_sheets/figures/mz_spatial_p2_frozen_sheets.png`
- summary：`results/topic4_sef_hfo/mz_spatial_p2_frozen_sheets/p2_frozen_sheet_summary.json`
- outcomes：`results/topic4_sef_hfo/mz_spatial_p2_frozen_sheets/p2_frozen_sheet_outcomes.csv`
- traces：`results/topic4_sef_hfo/mz_spatial_p2_frozen_sheets/p2_*_traces.npz`
- config：`config/topic4_mz_spatial_p2_frozen_sheets.yaml`

设计与执行合同：`docs/superpowers/specs/2026-07-20-topic4-mz-persistence-gated-additive-spatial-lifecycle-design.md`。
