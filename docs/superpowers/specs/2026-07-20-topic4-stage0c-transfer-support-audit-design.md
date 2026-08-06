# Topic 4 Stage 0C transfer-support audit（LOCKED）

**状态**：v1.0，2026-07-20 锁定。
**目的**：只裁决 Stage 0C coarse screen 中 23 条 `audit_invalid_candidate`
是否由原始 LIF transfer LUT 的低端裁剪伪造。它不是新参数搜索，也不开放
`phi`、slow variables、noise 或 spatial coupling。

**provenance note**：v1 数值产物完成于 05:17，而本 spec 的误差门文字与当前
source 收口于 05:20；v1 artifact 只记录 module hash，没有记录 runner hash。因此
v1 必须按 `validation implementation/spec provenance mismatch -> unresolved` 保留，
不能把 05:20 的文字追溯称为运行前预注册合同，也不能用后来口径改写 v1 verdict。

## 1. 被审计对象与停止边界

只重放 primary Stage 0C 已经暴露的六个唯一参数点：

| frozen `z` | `alpha_G` |
|---:|---:|
| 0.80 | 12 |
| 0.85 | 16 |
| 0.81 | 16 |
| 0.84 | 24 |
| 0.82 | 24 |
| 0.84 | 32 |

每点从 primary `root_continuation.json` 重建相同的 17 个 non-exact
state forks；exact roots 不进入动力学重放。六点共 102 条初态。禁止因为结果不理想
而加点、换初态或调 pool 参数。

## 2. 数值 transfer 合同

原 LUT 的支持域为 `mu=[-40,120] mV`、`sigma=[0.5,30] mV`，而六点原轨迹
的保存态已达到约 `muE=-1573 mV`、`muI=-569 mV`、`sigmaI=41.1 mV`。
因此本审计不允许把 `mu<-40` 再夹到 -40，也不允许把极低 `mu` 任意置零。

参考值使用同一 Siegert integral，但在 log domain 计算：

\[
\log f(x)=\log\!\left[\operatorname{erfcx}(-x)\right],\quad
\log I=\log\!\int_{y_r}^{y_\theta}e^{\log f(x)}\,dx .
\]

当上端点很大时，以该端点归一化并作 endpoint-scaled quadrature；endpoint 坐标
采用固定 80 的 log-distance cutoff，其误差不作未经验证的解析上界声明，而由原域
parity、算法接缝与 direct-exact 抽点共同验收。rate 只在最后一步按 IEEE-754 表示发生自然下溢，
同时保留有限的 `log(rate)`，不能把下溢当作模型裁剪。

extended transfer 使用不规则 `mu` 轴与 `log-integral` 插值，覆盖
`mu=[-2500,120] mV`、`sigma=[0.5,50] mV`；越界返回 NaN 并将该 fork 判为
`numerical_unresolved`，绝不 clip 或外推。两档分辨率锁定为：

- coarse：核心区 `mu=[-250,120]` 步长 0.5 mV，低端 64 个几何节点；
  `sigma` 步长 0.25 mV；
- fine：核心区步长 0.25 mV，低端 128 个几何节点；`sigma` 步长 0.125 mV。

必须完成三类验证：

1. 在原 LUT 重叠域与 canonical `lif_rate` 的有限值逐点 parity；
2. stable exact reference 在极低 `mu` 的连续性、随 `mu` 单调性和 branch 接缝；
3. coarse/fine LUT 对 deterministic direct-exact 抽点的误差，以及两种分辨率的
   轨迹分类一致性。

以下阈值是 v1 产物之后补齐的审计解释，不是可追溯的 v1 preregistration：原域随机
抽点的最大绝对误差在 coarse/fine 分别必须 `<=0.5/0.25 Hz`；
meaningful-rate 抽点的相对误差 P99 分别必须 `<=5%/2%`。候选轨迹上的
fine direct-exact 门同样要求最大绝对误差 `<=0.25 Hz` 且相对误差 P99 `<=2%`。

## 3. 动力学重放与逐 Euler 审计

- screen：coarse 与 fine 均用 `dt=0.25 ms, T=6 s`；
- confirm：只对 screen survivor，用 coarse 与 fine 的
  `dt=0.125 ms, T=12 s`；
- 每一个 Euler state 审计 transfer support、finite、rate/synapse/pool bounds、
  refractory occupancy 与 `rE>=100 Hz`；保存步不能代替逐步审计；
- direct-exact 抽点包括每条 fine 轨迹 tail 的固定分位时点和 moment 极值邻域。

每条 fork 的结论只能是：

- `candidate_survives`：coarse/fine 均为同类 bounded candidate，12 s confirm
  仍成立、逐步审计 clean、direct-exact 误差门通过；
- `collapses_low`：extended transfer 下回到 `<=5 Hz` low fixed point；
- `becomes_over_100`：tail 任一步达到或超过 100 Hz；
- `numerical_unresolved`：其余长瞬态、分辨率分歧、越界、非有限或误差门失败。

一个参数点至少要有两个不同 non-exact histories 收敛到相同对象，才算 point-level
support；单条 survivor 不算 basin。

## 4. survivor 后置门

只有 12 s confirm survivor 才运行：

1. fine transfer，`dt=0.0625 ms, T=12 s` 的时间步敏感性；
2. 五臂机制消融：`dynamic`、`instantaneous`、`clamped`、
   `matched_subtractive`、`mean_only`。

通过这些门仍只说明均匀冻结快系统存在 transfer-supported finite object；不得直接
启动 Stage 1，也不得写成 seizure、termination、retrigger 或 spatial pattern。

## 5. 工程与产物合同

- 单进程、BLAS threads=1、数组预计算，peak RSS `<4 GiB`；
- primary Stage 0C 的代码、config 与结果全部只读；
- 独立结果目录：
  `results/topic4_sef_hfo/spatial_slowfast_topology/stage0c_transfer_support_audit/`；
- 必须输出 JSON/CSV、transfer parity/误差审计、candidate trace 图、
  `figures/README.md` 与 `STATUS.md`。
