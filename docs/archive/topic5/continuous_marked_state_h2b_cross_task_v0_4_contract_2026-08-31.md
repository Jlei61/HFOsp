# Continuous Marked State H2b Cross-task Transfer v0.4 冻结合同

> 合同 revision v10：final assay 连续验收修复了 conditional risk set、非合同 gate、支持分层、control 硬匹配、假 route、弱正则化和过严的阴性解释门。v9 补入独立 `B_history`；逐问 closeout 又发现，若只分别汇总 `B_observation-B_history` 和主效应，不能可靠恢复患者层面的 `B_route_state-B_history`。v10 因此仅保留这一直接派生 contrast；模型、risk set、gate、route 算法和主 estimand 均不改变。所有旧 cell/assay 输出按 source hash 作废并重跑。

## 一句话科学问题

同一患者的不同发作不必由同一条状态轨迹进入。v0.4 检验的是：一个完全由间期背景与 IED 任务训练、随后冻结的状态，在允许最多两条患者内 seizure-entry route 后，能否在真正更晚的发作上增加风险与进入几何信息。

## 为什么放松 v0.3 gate

v0.3 把“多维 persistent state 资格”和“半合成联合 T+M+lag power”设为整个下游的启动门。该策略保护了阴性解释，但也隐含了一个过强前提：同一患者的不同发作必须共享一套足够稳定、单一的状态—风险映射。若不同发作靠近不同状态区或沿不同方向进入，单一线性读出和单一平均 entry direction 会互相抵消。

v0.4 因此只把工程与防泄漏边界保留为硬门。v0.3 的 `all_frozen`、`scalar_slow_axis`、`multidimensional_state`、`collapsed_or_instrument_weak` 变为并列报告的事前分层，不再阻止运行。半合成 assay 决定真实阴性能否解释，不决定队列是否执行。

## 不能放松的硬边界

- 状态仍只由连续背景和 IED timing/mark 训练，seizure loss 不得更新任何状态组件；
- checkpoint 与 full-grid state cache 必须逐个通过 v0.3 最终机器审计中的 SHA256；
- 每个 anchor 只使用 `t <= anchor` 的信息，跨 coverage gap 必须 reset；
- outer test 发作必须晚于 probe、PCA 和 route prototype 使用的全部发作；
- held-out 发作不得定义自己的 route；
- 所有比较臂使用完全相同的 fold rows；
- 统计支持单位是患者与 OOF 发作，seed、control 与连续 grid row 都不是患者重复；
- formal、sealed、H3、T2、physical clock 和 paper-ready figures 继续关闭。

## 异质 route 的固定实现

每个 outer fold 只使用更早发作的 `onset - lead` state。状态空间标准化与 PCA 只在 outer TRAIN 时间行拟合，最多保留四个分量。若已有至少四次过去发作，deterministic two-means 的两组各至少两次，且两中心在 TRAIN coverage 中相隔至少一个 route-distance bandwidth，则使用两条 route；否则自动退回一条 route。最多两条，不按真实 H2b 结果增减。

route ID 不跨 fold、seed 或患者比较。模型只使用到一条或两条 TRAIN route 的距离，因此 label switching 不影响结果。held-out 发作只能被投影和评分，不能重算 PCA、中心、带宽或 route 数。

## 风险模型与主 estimand

5 min full recorded grid 只用于产生候选 anchor；评分在患者内 rolling-origin/prequential conditional risk sets 上完成。至少 10 次支持发作时，前 `floor(60%)` 用于初始化并依次预测后 40%；5–9 次时以前两次初始化作 rolling sensitivity；3–4 次仅作描述。每个 OOF 发作形成一个 risk set：一个 `onset-lead` case 加五个同患者、有效记录且完整 seizure-free horizon 的 controls。同 coverage segment 优先但不设为硬门。controls 通过与 history、observation、state 全部独立的确定性整数 key 抽样；history、segment/session position 在所有 probe 臂显式调整，避免六维硬配平把慢状态一并消掉。outer TEST controls 严格晚于上一发作 cutoff，不能复用 TRAIN 时间点。每个 OOF 发作贡献一个等权 conditional loss。wrong-time state donor 仍必须同 coverage segment。

30 min 为唯一主 horizon，5、15、60、120 min 全部作为固定 sensitivity。比较：

1. `B_history`：IED history 与严格因果的 seizure/session context；
2. `B_observation`：`B_history` 加当前 explicit observation 和 observation-route 距离；
3. `B_linear_state`：同一 base 加原始 frozen persistent state，作为直接线性描述对照；
4. `B_single_axis_state`：同一 base 加仅由过去发作定义的单一有向慢轴；
5. `B_route_state`：同一 base 加最多双 route state 距离；
6. `B_route_memoryless`：以同容量 memoryless route 距离替代 persistent route；
7. `B_route_wrong_time`：保持训练模型不变，只在 test fold 替换 matched wrong-time state。

主 estimand 是等发作权 30-min conditional risk-set `logloss(B_route_state) - logloss(B_observation)`。负值表示 frozen state 在 history 与当前 observation 之外有增量。异质性专属 estimand 只在 TRAIN 已成功定义两条 route 的 folds 上计算，为 `B_route_state - B_single_axis_state`；单中心距离不作为对照，因为它本身可通过系数反号表示两个远端。`B_route_state - B_linear_state` 只作不同低容量表示的描述比较，不能单独证明异质性。

## 流形—流场模块

只在 outer TRAIN 的 clean interictal decoder output 上拟合投影。过去发作的进入 route 同时按终点位置和单位进入方向聚类，避免不同方向被平均抵消。held-out 发作只作 OOS projection，并与同患者、同 coverage 支持的过去 controls 比较 route-specific basin gating、directed approach 与 abrupt off-manifold transition。

双 route 几何必须同时报告单 route 结果。UMAP 不作证据，MARBLE 本轮不解锁。

## 半合成 assay 与解释边界

在真实 coverage、状态自相关、缺失和发作分母上运行 null、observation-only、single-route persistent、two-route persistent 与 clock-confounded worlds。校准集与评价集分开。方向 gate 要求 persistent worlds 的 primary、memoryless、time-specificity 以及 two-route heterogeneity 至少 70% replicates 为有利方向，同时三个负对照的 primary directional rate 不超过 65%，独立 null 5% 阈值错误率不超过 10%。严格 80% 单-replicate power 单独报告。若方向 gate 不过，或真实结果为 null/反方向而严格 power 未过，只能写 `NO_BIOLOGICAL_NEGATIVE`；assay 不阻止 all-frozen development 队列执行。

双 route 优于单 route，只能说明患者内异质 route 的低容量读出更合适，不能证明不同发作具有不同病因。只有 route-state 胜 observation、胜 memoryless，并出现正确时刻和 OOS 几何一致性，才允许写“支持 time-specific persistent state 向发作迁移”。全部结果仍是 development。
