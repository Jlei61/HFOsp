# Continuous Marked-State T2 长尺度总效应合同（2026-08-26）

## 1. 科学问题

本工作包不再问“最近 100 或 1000 次 IED 是否在当前状态之外预测下一次 IED”，而问：

> 从同一个累积窗起点状态出发，随后真实发生的数千到一万次 IED，是否能解释累积窗终点状态相对自然演化产生的偏移？

它测量的是长期 `IED exposure sequence -> future state` 的总效应候选，不是短尺度 residual edge。阳性仍是 development-level 预测证据，不等于因果干预或癫痫网络被永久重塑。

## 2. 数据与封条

- 患者：`yuquan_zhangjiaqi`，因为其连续记录段内可观测到一万次事件历史。
- T1：与 R1.3 相同的 exact timing + sequential tied-group mark 目标，explicit observation，state dimension 8，三 seed。
- TRAIN endpoint 用于拟合；development-validation endpoint 只评分。
- validation 窗口可以包含 endpoint 之前的 TRAIN 历史，这是因果历史，不是 outcome 泄漏。
- endpoint 必须逐一通过 `assert_development_times`；正式 test 分区保持封存。
- 窗口不跨未记录区间、发作或 post-ictal reset。

## 3. 两个预先固定的长尺度

### 3.1 主尺度：固定 10,000 次事件

每个 endpoint 取同一连续记录段内之前恰好 10,000 次 IED。该患者此前审计显示，此尺度的 validation 历时中位约 6.0 小时；因此它是“事件数尺度”，不能自动称为跨日尺度。

### 3.2 辅助尺度：固定约 6 小时

每个 endpoint 从同一连续记录段内最接近 `endpoint - 6 h`、且不晚于该时刻后的第一个 IED 起算。它保留窗口内事件数的自然变化，用于区分事件数与真实时间。

两种尺度独立报告；不要求 AND gate，不因其中一个普通阴性而停止另一个。

## 4. exposure 与状态演化

每次事件提供两个低参数输入：

1. occurrence impulse：该 IED 是否发生；
2. load innovation：实际参与负荷减去仅用 TRAIN 拟合的 `pre-event state + deterministic history` 条件均值，再除以 TRAIN residual SD。

冻结 T1 generator `K` 和均值 `mu`。窗口起点为 observation-informed T1 pre-event state。真实 exposure 臂在每次 IED 后施加同一个有正有负的二维线性 jump：

```text
z(t_j+) = z(t_j-) + B_occurrence + B_load * innovation_j
```

随后按冻结 `K` 演化到下一事件。`B` 只有 `2 x 8 = 16` 个参数。

## 5. 三个比较臂

1. `no_edge_natural_flow`：只从共同起点按冻结 K 演化；
2. `real_occurrence_plus_load`：真实 occurrence + load innovation 序列；
3. `causal_delayed_load_1000`：occurrence 完全相同，但第 j 次事件使用同一连续段内第 j-1000 次的 load innovation。

第三臂不读取未来 exposure，且与真实臂参数量相同。它用于判断收益是一般事件发生效应，还是依赖真实 load 的时间对齐。若真实和 delayed 都优于 no-edge、但彼此相同，只允许说 occurrence-like cumulative exposure 有信号；不能说正确 load 序列被识别。

## 6. 拟合与评分

- 在 TRAIN 内按时间顺序 80/20 划 inner selection；ridge 仅从固定小网格 `{1e-4, 1e-2, 1, 100}` 选择，然后在全部 TRAIN endpoint 重拟合。
- 不用 development-validation 选择尺度、ridge、seed 或 endpoint。
- 主评分在冻结 T1 decoder readout 空间完成，四块等权：
  - timing log-rate state contribution；
  - STOP logit；
  - non-zero group-size logits；
  - contact/subset logits（同一读出同时进入 first group 与 continuation）。
- 每块只用 TRAIN target variation 定标，避免 contact 数量较多而自动主导总分。
- 敏感性：latent-state MSE。
- 次要结果：从预测的窗口终点 state 开始，在不再读取新 observation 的情况下，对下一 IED 计算 exact one-step timing + sequential mark likelihood，并分解 STOP、first-subset、continuation-subset。

报告 `real - no_edge` 与 `real - delayed`；损失越低越好，因此负值有利。没有单项结果作为其他探索的 blocker。

## 7. synthetic recovery 先于人体解释

至少覆盖：

- true long edge：真实序列应优于 no-edge 和 delayed，并恢复 B 的方向；
- null edge：real 不应系统优于 no-edge；
- occurrence-only truth：real 与 delayed 可共同优于 no-edge，验证解释规则不是强制 real 胜 delayed。

synthetic 失败属于仪器问题，必须修复后才能解释人体数字；普通人体阴性不触发停工。

## 8. 允许和禁止的结论

允许：

- “在这一位高事件量 development 患者中，长 IED exposure 序列对约 6 小时后冻结 decoder 所见状态有/无增量预测信息。”
- “收益主要与 occurrence 累积一致”或“正确 load 时序额外有信息”，前提是对应比较成立。

禁止：

- 把一位患者写成队列结论；
- 把预测增量写成 IED 的因果塑形；
- 把 `N=10000` 直接写成跨日慢状态；
- 用 latent norm 单独承重；
- 把 total-effect 阴性当作不存在任何更慢、非线性或亚型特异 H3；
- 打开正式 test 分区。

## 9. 运行与交付

- GPU 作业串行；CPU worker 设 `OMP_NUM_THREADS=1`；OOM 时降低 batch/chunk，不删减患者或 endpoint。
- 所有长作业用 `setsid/nohup`、原子结果和可重入状态文件。
- 交付：三 seed T1、synthetic truth/null、两尺度三臂人体结果、白话报告、技术报告、机器审计与 `CURRENT_HANDOFF.md`。
