# 审阅结论：MZ spatial P-patch P=1 parity

日期：2026-07-20

工作线：`.worktrees/topic4-mz-divisive-lifecycle`

并行边界：本节点只把既有 Stage 0C fast scaffold 嵌入通用 P-patch state；没有改变 E→E weight/kernel/delay，没有实现 conductance/relay，也没有运行 P=2 或 SNN。

> **后续执行修正**：P=2 whole-complement 会把 local boundary wake 稀释，故本文第 6/7 节“P=2 不招募即停止”的旧预案已由 `mz_spatial_p2_frozen_sheets_2026-07-20.md` supersede。当前 stop gate 是质量守恒 P=3 core–annulus–bath 仍无 local hand-off；不调 E→E。

## 1. 一句话判断

**P=1/uniform-P parity 完整通过，可以进入 P=2。**新的 `10P+2` scaffold 在逐点 RHS、完整 Euler trace、directed return、`.31645/.31648` cycle boundary 和 fold surface 上都与锁定 Stage 0C additive system 为机器级 exact identity；这一步没有产生任何新的空间动力学结论。

## 2. 完成程度

> **完成度：95/100（仅针对 P=1 identity gate）**

已经完成：

- 锁定 field-major state：每 patch 7 fast + `z/p/m`，全域唯一 `mu_G/S_G`，总长 `10P+2`；
- 显式 frozen-slow mode：`dot z=dot p=dot m=0`，不是用超大时间常数近似；
- additive `A=Amax*m` 只减 `mu_E`，不改 variance；
- Z 同时进入 I→E mean 的一次方与 variance 的平方，不作用 I→I；
- spatial synapse target 使用 row-normalized `K@rate`，moments 不二次卷积；
- shared pool 使用显式 patch-area weights，只更新一对 `mu_G/S_G`；
- P=1、uniform P=4、off-manifold、cycle section、fold surface 与 base/half-dt return 全部验收；
- 图、CSV、JSON、NPZ、中文 README 与测试完成。

扣 5 分：当前 P=4 operator 只是 constant-preserving probe；P=2 的 fixed geometry block average、dynamic Z/p/latch/M 与 frozen sheets 尚未实现。

## 3. P0 / P1 关键问题

### P0 已关闭：没有把 Stage 0C batch 当空间 patch

新 state 只在末尾保存一个 `mu_G/S_G`。为了复用 moments 临时生成的 `(P,9)` rows 只复制同一 shared pool 做只读计算；不消费其局部 pool RHS。

**验收**：P=1 state size=`12`，P=4 state size=`42`；uniform-P 的 local RHS 全部等于 P=1，global RHS 只出现一次且没有随 P 放大。

### P0 已关闭：冻结 additive parity 没有偷偷 decay M

在 parity mode 中 `m=A/Amax` 且 `dot m=0`。如果让 latch-off M 自然衰减，就不再是上游 frozen-A vector field。

**验收**：所有 probe 的 extra `z/p/m` RHS 精确为 `0.0`。

### P1 开放：当前安全 oracle 不适合直接跑 P=2 长轨迹

本轮每个 RHS step 都重新验证 local Z 并构造 prepared moment rows，保证身份清楚，但 10 个 boundary returns 用时 `6:55.7`。

**怎么改**：P=2 hot path 写同一 vectorized moment algebra，并继续用本 P=1 oracle 做随机 state regression；不能为了速度改变方程顺序或复制 local pool。

## 4. 科学性问题

### 精确等价结果

所有最大误差均为 `0.0`：

- P=1 pointwise RHS；
- uniform P=4 RHS；
- base/half-dt return period；
- crossing state；
- 每一个 Euler state 的完整 trace；
- 上游 `A=0–1.5 mV` fold-surface RHS。

关键周期/边界逐项复刻：

| A (mV) | dt (ms) | Stage 0C / P=1 |
|---:|---:|---:|
| 0 | .125 | `604.8153 ms` clean return |
| .30 | .125 | `1229.4156 ms` clean return |
| .316 | .125 | `4288.8806 ms` clean return |
| .31645 | .125 | `11345.3606 ms` clean return |
| .31648 | .125/.0625 | both no return within `12 s` |

### 这一步允许和不允许的解释

允许：

> 后续 P=2 与 Stage 0C 的差异可以归因于 fixed spatial coupling、local slow state 与 shared recruitment，而不是 P=1 方程重写漂移。

不允许：

- 已有 core→surround recruitment；
- 已有 local onset/containment/termination；
- P=4 probe 的矩阵就是正式空间 kernel；
- 已经证明空间模型能回到 low basin。

## 5. 工程性问题

- 四个上游输入 SHA-256 fail-closed；
- formal smooth no-clip transfer 与上游 additive continuation 完全相同；
- `PatchKernels` 强制 finite、nonnegative、row sum=1；
- patch-area weights 强制 positive、sum=1；
- 新增 9 个 patch unit tests；
- 相关 Gate0/persistence/pool + patch tests 将在提交前合并复跑；
- runner 单进程、BLAS 单线程，peak RSS `318324 kB`，无 swap/OOM；
- 图已目视检查，结果目录含中文 README。

## 6. 最小修改路线

1. 冻结 P=1 identity commit，不在此节点加入 slow equations。
2. 从既有 M3B geometry 固定构建 P=2 block-averaged `K_EE/K_I`；宽度、gain、delay 不扫。
3. 先做 `LL/CL/CC/LC` frozen sheets，求邻区状态如何移动每区的 fold/return boundary。
4. 写 vectorized P=2 hot RHS；每次变更以本 P=1 exact oracle 回归。
5. 先测 fixed coupling 是否能产生 `CL→CC`；若不能，不调 E→E，本线停止并交并行线。
6. 只有 recruitment 成立，才加入 local persistence + neighborhood recruitment AND-set、Z-safe reset 与 bounded additive M。
7. P=2 完整 ODE 只跑 memoryless/latch、rho off/on、m off、coupling off；通过后才开 P=32/64。

## 7. 下一步建议

**GO 到 P=2 frozen-sheet / fixed-coupling oracle。**下一问题是邻区处于 low 或 cycle 时，本区的 nullcline/fold/return boundary 如何移动；不是先调 latch 参数。若 fixed current scaffold 本身不能让 core 招募 surround，当前 additive line 应诚实停止，而不是借修改 E→E 抢救。

## 8. 产物

- 图：`results/topic4_sef_hfo/mz_spatial_patch_p1_parity/figures/mz_spatial_patch_p1_parity.png`
- summary：`results/topic4_sef_hfo/mz_spatial_patch_p1_parity/p1_parity_summary.json`
- RHS：`results/topic4_sef_hfo/mz_spatial_patch_p1_parity/p1_rhs_parity.csv`
- returns：`results/topic4_sef_hfo/mz_spatial_patch_p1_parity/p1_return_parity.csv`
- folds：`results/topic4_sef_hfo/mz_spatial_patch_p1_parity/p1_fold_surface_parity.csv`
- traces：`results/topic4_sef_hfo/mz_spatial_patch_p1_parity/p1_parity_traces.npz`
- config：`config/topic4_mz_spatial_patch_p1_parity.yaml`

设计与执行合同：`docs/superpowers/specs/2026-07-20-topic4-mz-persistence-gated-additive-spatial-lifecycle-design.md`。
