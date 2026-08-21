# Topic 4 rev12-ND：历史 Node 连续场零仿真重评分

## 一句话裁定

当前冻结 Node 仍是相容历史库中最低的最弱模式损失，但完整患者事件分布的 held-out `R2=-0.563`；没有历史 field 同时改善完整分布和弱模式，因此历史库只能提供新的连续场搜索方向，不能直接替换 Node。

## 为什么重评分

旧 Fig.4 只显示 returned、双杆且位于冻结患者支持域内的 `formal-clean` events。这适合展示可辨认模式，却会删除拟合必须承担的单杆和 OOD events。rev12-ND 冻结两个估计量：

- `complete_returned`：所有 returned events，新拟合主目标；
- `formal_clean`：旧 Fig.4 支持域内事件，仅作历史诊断。

四类模式距离不再直接平均原始数值。每种患者模式从不同 recording blocks 各取 6 个事件，块间距离中位数定义为 0、q95 定义为 1；raw 和 normalized 距离同时保存。这样 recruitment、precedence、profile 和 event cloud 都表示“超出患者自身块间波动多少”。

## 工程 parity

当前冻结 Node 的 rev11 `formal_clean` 记录精确复现既有审计：

| 指标 | 旧审计 | rev12 重算 | 最大绝对误差 |
|---|---:|---:|---:|
| held-out event representation `R2` | -0.2171863872 | -0.2171863872 | 2.22e-16 |
| scale-calibrated contrast `R2` | 0.4471590290 | 0.4471590290 | 2.22e-16 |

`D6.1 edge_noop`、`D6.2 edge_noop`、`D6.3 edge_noop` 和 rev11 `node_baseline` 的 field hash 完全相同。此前 record-level 数值差异来自不同 seed pools，不是不同 field。正式重评分因此在相同 spatial-OU stratum 内按 field hash 合并，并让每个 network seed 只计一次；相同 seed 有多条记录时保留最长轨迹。

## 主要结果

82 条历史 candidate records 合并为 66 个唯一 fields，其中 9 个满足当前 spatial-OU Node-only 合同。关键结果为：

| field | networks | objective | mode 1 loss | mode 2 loss | held-out `R2` | scaled contrast `R2` |
|---|---:|---:|---:|---:|---:|---:|
| 当前冻结 Node | 36 | 0.645 | 0.469 | 0.688 | -0.563 | 0.316 |
| 最低目标的不同场 `f05_sin_m0p8` | 12 | 0.705 | 0.494 | 0.761 | -0.610 | 0.244 |
| 几何 Pareto 场 `d62_a0p5_b1p0` | 6 | 0.736 | 0.351 | 0.868 | -0.560 | 0.268 |

三点需要一起读：

1. 当前 Node 对两个模式都不是患者 floor 内匹配，mode 2 是稳定的弱项；
2. `f05_sin_m0p8` 没有改善当前 Node，只提供一条不同的 whole-sheet residual 方向；
3. `d62_a0p5_b1p0` 略改善整体 held-out 几何和 mode 1，却明显损害 mode 2，证明平均几何可以掩盖弱模式。

所有相容历史 field 的完整 held-out `R2` 仍为负。自然双簇在这些网络中普遍存在，但“双簇存在”没有转化为“完整患者事件分布被解释”。

## 科学边界

历史 worker 没有保存逐事件 sheet-level source onset maps，因此不能判断两个模式是否具有可重复、不同的神经元级起始拓扑，也不能区分单一传播源与时间耦合的多处点火。历史 Pareto 只用于初始化，不满足 rev12 的 source-topology 或同 checkpoint 干预端点。

## 下一步

1. 用当前 Node、最低目标的不同场和几何 Pareto 场跑同一组 Node-only canary seeds；EE、E-to-I、Z/M 全关。
2. 同时保存每个 returned event 的 1 mm source-onset map，分别审计患者标签和自然 KMeans。
3. 若历史方向都不改善，则在当前 Node 周围启动不使用 contact 坐标的 4x4 whole-sheet continuous residual search；不是增加 core 数量。
4. 只有完整 held-out 分布、弱模式和源拓扑共同改善后，才做同 checkpoint 干预并冻结 Node。

图见 `results/topic4_sef_hfo/data_driven_node_dualmode_rev12/historical_rescore/figures/node_historical_rescore.png`。
