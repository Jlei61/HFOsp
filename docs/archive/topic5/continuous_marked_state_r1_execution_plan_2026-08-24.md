# Continuous marked-state R1 Execution Plan

**版本：** R1.0–R1.3
**日期：** 2026-08-24
**原则：** 先完成正确 likelihood，再做 observer，再做 T1；不以普通阴性为 gate。

## 1. 目录与所有权

新代码与旧版隔离：

```text
src/topic5_continuous_marked_state_r1/
scripts/topic5_continuous_marked_state_r1/
tests/topic5_continuous_marked_state_r1/
results/epi_prssm/continuous_marked_state/r1/
```

R0.1 只读。所有 R1 JSON 必须携带 contract revision、producer SHA、split/sealed 状态、输入 artifact hash、seed 和完成状态。临时文件用同目录 `.tmp` 后原子替换。

## 2. Phase A：R0.1 封板

交付：

- R0.1 正式验收记录；
- `R0_1_ACCEPTANCE.json`；
- 本地重跑 tests 与 package audit；
- 锁定 H3-S0/R0.2/大 T2 停止项。

该阶段只确认上一轮交付，不追加科学实验。

## 3. Phase B：R1.0 likelihood kernel

### B1 recorded coverage

1. 从 `event_marks` 与 `sessions.build_sessions` 构建每位患者真实 recorded block intervals；
2. 合并相接/重叠区间，保留 gap；
3. 将 TRAIN/validation boundary 切入 coverage segments；
4. 对现有 `recorded_intervals/*.npz` 做逐事件累计 recorded-duration parity；
5. 写 per-subject NPZ 与 manifest。

关键测试：不重叠、有序、正长度、gap 不进积分、event 必须落在 coverage、sealed 时间不可见。

### B2 timing likelihood

实现：

- log-intensity callback；
- recorded-segment deterministic quadrature；
- event log term 与 survival term；
- patient/event/hour normalization；
- time-rescaling diagnostic。

合成测试：常强度 Poisson 有闭式值；带 gap 的积分等于分段积分和；未来 gap perturbation 不改变更早 loss。

### B3 exact tied subset

实现 log elementary symmetric polynomial DP、conditional subset log-prob、group-size/STOP likelihood 和完整 event likelihood。

穷举测试：小 `N` 下与所有组合直接枚举逐元素一致；合法 subsets 概率和为 1；tie permutation 不变；recruited contact 的概率为 0；非法 group fail closed。

## 4. Phase C：event-only strong baseline

1. 建 deterministic history features；
2. TRAIN-only fit/freeze timing baseline；
3. 建 static/history sequential mark decoder；
4. 用 exact likelihood 重现“历史有预测信息”的最基本 H2a 层；
5. 输出 timing、group-size/STOP、identity、full mark 和 prefix-depth 指标。

本阶段不需要重现旧模型的所有效应量；目的是证明新尺子能在真实事件流上训练、校准和区分 history-shuffle。

## 5. Phase D：Bridge-E1 正确目标上的 raw 检验

只比较：

- `B_explicit`；
- `B_explicit_raw`。

二者共享 exact timing/mark head。raw residual `alpha_raw=0` 初始化；不运行 raw-only、Conformer、wide-depth 或频谱 horizon 网格。

先在 3 位预定患者、seed 0 运行：`epilepsiae_620`、`epilepsiae_958`、`yuquan_huanghanwen`。无论结果正负都形成有效读数。若 raw 分支数值失败，修数值；若只是无增量，记录为科学阴性并继续 T1 explicit 版本。

Raw observer 的 IED core 处理必须保持因果：anchor 后事件不能改变过去窗口；anchor 前 IED core 只用于背景插补，mask 缺口不能作为额外 event-history 输入。该规则属于 H1 目标隔离，不是结果阳性闸门。

## 6. Phase E：R1.2 T1 六人 pilot

执行顺序：

1. 缓存 frozen observation embedding；
2. 训练 `T0_no_state`、`T1_explicit`、`T1_explicit_raw`；
3. patient-first 汇总 filtered prediction；
4. matched wrong-time state swap；
5. event-observed raw-correction-off H1/H5/H10/H20；
6. fully generative supportive smoke；
7. state_dim16 一次敏感性。

主比较只使用 seed 0；数值接近噪声时，对主 cell 加 seed 1/2，不扩展结构网格。

## 7. 资源与恢复

- 数据构建使用独立 CPU workers，每个 worker 只写自己的 subject 目录；
- 训练按可用 GPU 一设备一作业，先按 `batch × contacts × patches` 估算，OOM 时依次减 batch、开 gradient checkpoint、减 sequence chunk；不改科学输入；
- 所有长作业使用 `nohup`/`setsid` 或 tmux，stdout/stderr 独立日志；
- `OMP_NUM_THREADS=1`，数据 workers 不与 GPU 作业争磁盘；
- manifest 驱动断点续跑；只有 `status=COMPLETE` 且 artifact hash 匹配的结果可聚合；
- 网络波动不得影响本地作业，运行期不依赖下载或远程服务。

## 8. 实验推进规则

普通负结果不是 blocker，也不触发停止整条路线。继续执行后续对照，并缩窄结论。

只在以下情况暂停对应分支：

- 关键 raw/cache/inventory 未挂载或 subject/contact 映射不唯一；
- event 落在 recorded coverage 外；
- TRAIN/validation/sealed 边界违规；
- current-event/future-mark 泄漏；
- exact subset likelihood 不归一；
- revision/package 混用；
- 持续 OOM 且 batch/chunk/checkpoint 降级后仍不能运行。

不得因为某患者结果难看而剔除；不得在看到 validation 数字后改变主 endpoint 或患者集合。

## 9. 验证与审计

每个阶段至少生成：

- unit tests；
- synthetic recovery；
- one-subject end-to-end smoke；
- manifest/hash audit；
- sealed-partition scan；
- denominator table；
- plain-language 与 technical report。

测试证明实现符合合同，不等于科学结论成立。科学结论按 patient-first development evidence 单独写。

## 10. 交付与接手

R1 根目录最终包含：

```text
CURRENT_HANDOFF.md
RUN_STATUS.json
manifests/R1_CONTRACT.json
manifests/FINAL_PACKAGE_AUDIT.json
reports/plain_report_<date>.md
reports/technical_report_<date>.md
coverage/
baselines/
bridge_e1/
t1_pilot/
figures/README.md
```

只有实际生成图后才创建 `figures/README.md`。图未生成前不写占位 README。

## 11. R1.2 之后的决策

- 若 exact instrument 与 synthetic recovery 合格，且真实数据可训练：进入六人 T1，无需等待阳性；
- 若 explicit+raw 无增量：保留 explicit observer 继续 T1，并将 raw 阴性如实报告；
- 若 T1 只有 filtered 增量：命名 predictive filter，仍可进入小规模 R2.0 探索，但不得称 autonomous state；
- 若 exact mark identity/continuation 有增量：H2a 得到新模型复现；
- 只有 R1.2 的数据流、likelihood、rollout semantics 和 checkpoint 冻结后，才讨论 R1.3 队列扩展或 R2.0 `N=100` T2。
