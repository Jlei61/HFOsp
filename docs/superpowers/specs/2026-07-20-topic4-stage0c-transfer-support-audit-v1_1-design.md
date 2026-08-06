# Topic 4 Stage 0C transfer-support audit v1.1（LOCKED numerical repair）

**状态**：v1.1，2026-07-20 锁定。
**父审计**：`2026-07-20-topic4-stage0c-transfer-support-audit-design.md`。
**触发原因**：v1 的 stable exact reference 与候选轨迹 direct-exact 均通过，但
v1 artifact 没有 runner hash，且数值门 spec 晚于产物，形成 implementation/spec
provenance mismatch；所以 v1 必须保留为 unresolved。v1.1 在查看任何 v1 动力学
分类之前，新锁定一个更保守的 absolute 0.25 Hz 数值修复门。

本修复只增加一个预先固定的 `extra_fine` transfer，不能用 v1 的动力学分类来调轴、
选参数或换初态；v1 产物不得覆盖。

## 1. 唯一工程修改

`extra_fine` 与 v1 使用同一 exact-Siegert reference、同一不规则支持域
`mu=[-2500,120] mV`、`sigma=[0.5,50] mV`、同一 log-integral 插值；只把
分辨率锁定为：

- dense-core `mu` step = 0.125 mV；
- `sigma` step = 0.0625 mV；
- low-mu geometric tail = 256 nodes。

v1.1 新锁定的 conservative deterministic audit 要求最大绝对误差 `<=0.25 Hz`，
meaningful-rate 相对误差 P99 `<=2%`。若 extra-fine 不通过，立即停止并保留
`numerical_unresolved`。

## 2. 重放合同

- 六点、每点 17 个固定 non-exact forks、共 102 条；参数、初态和顺序与 v1 完全相同；
- screen：extra-fine `dt=0.25 ms, T=6 s`；现有 v1 fine 只作为独立分辨率诊断；
- authoritative extra-fine 必须同时通过 overlap parity、候选轨迹 direct-exact、
  every-Euler support/state/refractory audit；
- extra-fine 与 fine 的 classification 必须相同，tail mean rate 差不超过
  `max(1 Hz,10%)`；oscillation frequency 差不超过 `max(0.5 Hz,15%)`；
- coarse 只保留为 v1 诊断，不参加 v1.1 survivor 判定。

只有 screen survivor 才重跑 fine/extra-fine `dt=0.125 ms,T=12 s` confirm。
confirm survivor 仍须运行 extra-fine `dt=0.0625 ms,T=12 s` 和原五臂 ablation。
point-level support 仍要求至少两条不同 non-exact histories 收敛到同一对象。

## 3. 输出与边界

独立输出：
`results/topic4_sef_hfo/spatial_slowfast_topology/stage0c_transfer_support_audit_v1_1/`。
schema 为 `topic4_stage0c_transfer_support_audit.v1_1`，必须链接 v1 summary/hash。
peak RSS `<4 GiB`。无论结果如何，都不得直接开放 Stage 1；本审计仍只裁决
均匀、冻结 z、无噪声快系统中的有限对象。
