# FCXR-LC2-Core R1 验收与闭环探索交接（2026-08-02）

## 0. 正式判决

本阶段工程实现验收，科学判决收窄为：

```text
R1_IMPLEMENTATION_ACCEPTED
R1_SUSTAINED_CONTROL_SEPARABLE
R1_ACTIVE_BOUT_REANALYSIS_REQUIRED
R1_LONG_REST_GAP_NOT_BRIDGED
R1_CLOSED_LOOP_H_GEOMETRY_UNTESTED
R2_CLOSED_LOOP_EXPLORATION_AUTHORIZED
```

原始统计文件中的 `H_SENSOR_NOT_SEPARABLE` 只表示：单一绝对阈值无法同时压过 returning IED 的
极端峰值，并在包含一次完整 offset-like gap 的 HEO2 全窗口内始终保持开启。它是 strict
full-window / long-gap stress test，不再具有关闭 LC2 或阻止 closed-loop H 实验的权限。

## 1. 实现了什么

新增的 `h_lc2` 是每个 E 细胞的 local recurrent-drive persistence state：

\[
h_{n+1}=h_n e^{-\Delta t/\tau_H}+
(1-e^{-\Delta t/\tau_H})g^{raw}_{A,n}.
\]

输入位于 X relay 之后、RC1 recurrent saturation 之前；膜在第 n 步只读取 `h(t-)`，本步
`gA_raw` 在膜更新后才进入下一步 H，因此没有同一步代数自激。H 关闭或 `rho_H=0` 时 RC1 路径逐位
一致。本阶段 H 只作 sensor，尚未打开 H current，也未测试 H basin、X offset 或 dynamic Z/H/X
lifecycle。

## 2. R0 工程验收

- post-X / pre-tanh 数据流合同通过；
- `rho_H=0` membrane/raster/RNG parity 通过；
- snapshot/restart 通过；
- 500 ms active-H smoke 有限、确定、零 conductance clip；
- 六个 blessed engine 文件未修改。

## 3. 四条 40k sensor-only replay

四条轨迹均使用 connection seed 1、固定 4096 个 E cells、1 ms block-average，单 worker 运行，并逐点
复现锁定 artifact：

| state | T | mean rate | events | 解释 |
|---|---:|---:|---:|---|
| RC1 baseline | 8 s | 3.838 Hz | 9 | returning IED negative reference |
| LC1 q75 | 5 s | 5.347 Hz | 12 | dense-train diagnostic only |
| HEO1 | 4 s | 130.592 Hz | 1 | sustained-load positive control |
| HEO2 fast-10% | 5 s | 62.174 Hz | 3 | intermittent morphology / gap stress reference |

HEO2 第一段活动在 2796 ms 结束，第二段在 3862 ms 才开始，中间 rest-like gap 为 1066 ms。

首次 replay 曾因 float64 结果与 float32 归档存在 `1.4e-14` 表示残差而被误报失败；比较改为归档 dtype
后四条 exact-prefix 全过，旧失败保存在 `superseded/`，没有删除。

## 4. R1 原始严格统计与重判

原始 gate 比较 returning IED 的 `Q99.9` bootstrap upper 95% 与 HEO1/HEO2 全窗口 trough 的
bootstrap lower 95%。该统计保留，但拆开解释：

- HEO1-only 在 `tau_H = 248.866–915.442 ms` 存在可分区；
- 最佳 HEO1-only 点为 `tau_H=419.017 ms`，margin `+0.30695`；
- HEO2 full-window 没有可分区；最佳点在 `tau_H=2000 ms`，margin `-0.70520`；
- 因此旧 joint-min gate 全程为负，但负值由 HEO2 的完整 rest-like gap 承重。

这证明 local recurrent-drive EMA 含有持续性信息，也证明延长时间常数不能同时解决 IED carryover 与
1066 ms long-gap bridge。它不回答 closed-loop positive feedback 是否能建立并维持新的 high basin。

## 5. 工程与资源

- R1/H path/RC1/MZ slow-variable 相关回归：102 tests passed；
- replay peak self RSS：10.46 GiB；
- swap：820.46 MiB → 816.21 MiB，无增长；
- 四条 40k replay 严格 single worker、线程钉死为 1；
- detached `setsid nohup` 运行，按 PID/sentinel 等待；当前无本任务残留进程；
- blessed hashes：`kick_probe febba300...`、`params c2036d1e...`、`model e77aad35...`、
  `connectivity 5ad75ec5...`、`connectivity_rot 2aa44262...`、`lfp 7f079081...`。

## 6. 产物

结果根：

```text
results/topic4_sef_hfo/fcxr_lc2_core/
```

关键文件：

- `r0_vertical_slice/{dataflow_contract,parity,smoke,DONE}.json`；
- `r1_sensor/artifact_preflight.json`；
- `r1_sensor/{baseline,q75,heo1,heo2}_{gA_sensor.npz,replay.json}`；
- `r1_sensor/h_sensor_separability.json`（原始 strict diagnostic）；
- `r1_sensor/r1_stage_acceptance.json`（canonical scoped adjudication）；
- `candidate_verdict.json`；
- `figures/h_sensor_separability.png` 与中文 `figures/README.md`。

## 7. 允许与禁止的结论

允许：H 竖向实现可信；HEO1 sustained control 与 IED 在有限 tau 区间可分；HEO2 full-window long gap
不能由同一 strict threshold 干净桥接；R2 closed-loop geometry 可以继续。

禁止：不得写成 H 无双稳态、LC2 失败、X 无终止权限、已得到发作生命周期，或 HEO2 是未来 H carrier
必须逐点复现的 phenotype。

## 8. 下一阶段

下一阶段先零新增 40k 仿真完成 active-bout / spatial-support / false-latch / bridge Pareto 表征；随后直接
测试 state-dependent H return map、40k frozen low/high/X forks，并仅在 frozen geometry 成立后运行
dynamic Z/H/X 无 kick lifecycle pilot。R1 strict full-window statistic 不再是 hard stop。
