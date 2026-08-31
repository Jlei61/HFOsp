# H2b Cross-task Transfer v0.3 Scientific Spec

> 2026-08-31 执行附录：保持本 spec 的假设、估计量与 claim ladder，但将 A1/A2/T/M/D 改为 claim-specific evidence tiers，不再作为整个 v0.3 的串行总 gate。详见 `continuous_marked_state_h2b_cross_task_v0_3_exploration_addendum_2026-08-31.md`。

## 0. 版本定位

版本名：**Frozen Interictal State, Incremental Hazard, and Seizure-entry Geometry**。

本版是看过 v0.2 development 结果后的测量重设计，不是独立 confirmation。它不修改上游 state，不打开 formal/sealed，不进入 H3/T2，也不把发作标签反向训练进 state。

一句话科学论点：

> 一个只从连续背景和间期事件中学习、随后冻结的状态，如果真是可迁移的生理状态，应当同时保留当前窗口之外的历史、在正确时间尺度上增加未来发作风险信息，并在 held-out 发作前进入可重复的状态空间组织。

机器合同：`config/topic5_continuous_marked_state_h2b_v0_3.json`。解释冲突时，以机器合同和本 spec 为准；执行细节见同日 plan。

## 1. 与最初目标的关系

最初目标不是做一个普通 seizure classifier，而是检验“从间期任务中发现的状态能否跨任务”。它包含三个递进层级：

1. **susceptibility transfer**：间期学到的表示是否含未来发作风险信息；
2. **dynamical continuity**：这种信息是否来自跨窗口持续、时间正确的同一状态；
3. **organisational continuity**：该状态是否连接间期事件的表达方式与即将发生的 ictal recruitment。

v0.2 只完成了第一层的一次低容量 probe，而且测试分母很小。v0.3 的任务是把三层重新接起来，而不是仅提高分类分数。

本版与其他假设的边界：

- H1 是前提：上游必须先证明候选量具有跨窗口状态性质；
- H2a 提供 IED timing/mark decoder 和可解释距离；
- 本版只检验 H2b 的跨任务迁移；
- IED 是否反过来改变状态属于 H3，预测关联不能替代 H3。

## 2. 三个冻结假设

### H2b-T：真正的增量迁移

在 nuisance、近期事件历史和当前 observation 已进入模型后，persistent state 仍改善未来 30 min 发作风险的 held-out log loss。

### H2b-M：跨窗口记忆

persistent state 中不能由同一当前窗口、memoryless code、nuisance 和近期历史解释的残余部分，仍改善 held-out 发作风险。

### H2b-D：时间与动力学专属性

在纯间期数据估计的状态时间尺度内，held-out 发作前轨迹表现为以下至少一种可重复结构：

- 进入并停留于高风险 basin；
- 沿流场逐渐逼近 seizure-entry region；
- onset 附近突然离开 clean interictal manifold。

T、M、D 同时成立，才允许称 `transferable persistent latent state`。只有 T 成立时，只称 `transferable representation`；只有 5 min 时刻信号而无 M 时，只称 `acute preictal encoder`。

## 3. 数据与不可变边界

### 3.1 输入数据

- 上游：R1.7B development checkpoint、observer、generator、IED timing/mark decoder；
- 当前已知 attrition：85 个训练单元，75 个 checkpoint 可读，46 个 state-cache cells，10 位患者进入 v0.2 支持审查；
- outcome：development 分区内、经 crosswalk 和 coverage segment 验证的 lead seizures；
- 时间：绝对时间保持 float64；任何 gap 处重置，不跨未记录区间传播。

这些数字只是 A0 起点，不自动等于 v0.3 的可估计分母。每个 contrast 必须重新报告实际 OOF lead seizures 和患者数。

### 3.2 永久禁止

- seizure loss 更新 observer、generator、state 或 IED decoder；
- 用 seizure performance 选 state dimension、checkpoint、seed 或流形参数；
- 把 optimizer seed、5 min anchors 或重叠窗口当独立患者/发作；
- 在全数据或 held-out seizure 上拟合流形后再声称 OOS geometry；
- 用 UMAP/CEBRA 的颜色分离承担证据；
- 重新聚类 seizure subtype 挽救 phenotype；
- 把预测结果写成 IED→state 因果或临床预测器。

## 4. 变量和嵌套估计量

每个 5 min anchor 只使用该时刻及以前的信息：

- `C`：time of day、recording day、距上次发作、coverage segment，以及真实可用的 sleep/medication/stimulation；
- `H`：近期 IED count/rate/trend、距上次 IED、last/recent load、STOP/extent、冻结传播摘要；
- `O`：当前 explicit observation，加上将 persistent state 重置后由当前窗口独立得到的 memoryless observer code；
- `ZP`：正常因果传播和 observation correction 得到的 persistent state；
- `ZM`：在同一 anchor 重置历史后，仅由当前 observation 得到的 memoryless state；
- `R`：只在 outer-training fold 内，用低容量模型从 `ZP` 中扣除 `ZM+O+C+H` 后的 persistent-history residual。

冻结模型：

```text
M0 = C + H
M1 = C + H + O
M2 = C + H + O + ZP
M3 = C + H + O + ZM
M4 = C + H + O + ZM + R
```

主迁移量：

```text
Delta_T = held-out logloss(M2) - held-out logloss(M1)
```

记忆量：

```text
Delta_M = held-out logloss(M4) - held-out logloss(M3)
```

两者均为负值有利。主 endpoint 是完整 5 min grid 上未来 30 min 的离散 hazard log loss；5/15/60 min 从同一模型导出为 secondary。v0.2 的 exact onset-minus-lead risk set 只作为 bridge sensitivity。

## 5. State instrument qualification

资格只用 interictal development 数据，不读 seizure outcome。

|检查|实际问题|合格解释|
|---|---|---|
|Q1 non-collapse|状态是否只是常数、缺失标志或一条轴|decoder metric 下有可重复有效秩和方差|
|Q2 cross-window information|是否真的保留当前窗口以前的信息|held-out interictal target 上 persistent 胜 memoryless|
|Q3 generator contribution|历史是否每次都被 observer 覆盖|generator/open-loop 带来可测信息，不是全靠即时 correction|
|Q4 time scale|是否有可辨识的慢度|`tau_z` 长于 observation window，且在连续记录段内能观察到衰减；区间删失记为不可辨识|
|Q5 seed stability|是否只是任意 latent 坐标|decoder outputs、距离矩阵或对齐轨迹在至少三 seed 稳定|
|Q6 not-only-clock|是否只是昼夜/住院日/距上次发作|调整所有已验证可用 nuisance 后，间期增量仍在|

输出两个不混用的人群：

- `all_frozen`：完整展示生产流程和失败原因；
- `state_qualified`：承担 persistent-state 科学主张。

缺少经过验证的 sleep/medication 标签只记为限制，不伪装成“状态失败”；缺少 Q1–Q5 的承重证据则不能进入 state-qualified。

## 6. Assay qualification

半合成世界必须保留真实 coverage、缺失、发作数与聚集、时钟分布、state autocorrelation 和 control sampling：

1. null；
2. observation-only；
3. persistent-state；
4. clock-confounded；
5. basin gating；
6. directed approach；
7. abrupt transition。

先用 100 replicates 做实现 smoke test；冻结所有设置后只对最终仪器运行 1000 replicates。最小相关效应固定为相对 held-out log loss 改善 5%。最终要求 type-I error 不超过 0.05、其 95% 上界不超过 0.075，power 不低于 0.80、其 95% 下界不低于 0.75。

未达到灵敏度时，结果是 `ASSAY_NOT_SENSITIVE`，不是 biological negative。该规则只停止受影响的结论分支，不阻断上游 instrument 修复或已预定的 all-frozen 描述。

## 7. Causal evaluation without wasting seizures

按每位患者的 lead seizure 时间做 rolling-origin/prequential evaluation：用前 `K` 次初始化，只用过去数据预测下一次，再把已发生发作加入训练。`K` 从 2/3/4/5 中由半合成 assay 一次冻结，不能按患者结果调整。

科学支持单位依次为：

1. 真正 OOF 的 lead seizure；
2. recording segment；
3. patient。

患者是 cohort inference 的一级单位；seed 先在患者内汇总。大量 grid anchors 只能增加时间覆盖，不能虚增样本量。

主 probe 是 ridge pooled logistic/discrete hazard；低自由度 spline/GAM 仅作 sensitivity；大型非线性模型只作 development oracle ceiling。

## 8. 时间专属性

先在 clean interictal trajectory 的 decoder metric 中估计 `tau_z`，再比较：

```text
Z(t), Z(t-0.5*tau_z), Z(t-tau_z), Z(t-2*tau_z), Z(t-4*tau_z)
```

每个 case 使用多个过去 donor，并匹配同患者、coverage segment、可用 sleep、time of day、recording day、距上次发作和 observation availability。未来状态只能作明确标记的 acausal falsification。

慢状态不要求逐分钟唯一；预期是超过自身时间尺度后增量或几何一致性衰减。

## 9. OOS 流形—流场

几何只用 outer-training clean interictal 连续轨迹拟合，预先排除 preictal、ictal、postictal。held-out preictal/ictal 点只能 OOS projection。

距离优先级：

1. 模型原生生理参数；
2. IED contact participation、STOP/extent、order/propagation 等 decoder outputs；
3. decoder-induced metric；
4. robust whitening 后的 diffusion/kNN geodesic。

预注册三个互不强求同时出现的 family：basin occupancy/dwell、directed approach/flow alignment、abrupt off-manifold exit。UMAP 仅作可视化；MARBLE 仅在轨迹够密、邻域跨多个独立 segment、局部 flow bootstrap 稳定且半合成可恢复后解锁。

流形模块回答 D，不能补救 T 或 M 的失败。

## 10. IED-specific 与 ictal organisation 扩展

只有出现稳定 T 且至少一个 M/D 分量后，才运行完全匹配的上游三臂：`full`、`background-only`、`IED-shuffled`。三臂架构、宽度、训练量和 qualification 相同，且都不见 seizure label。只有 full 胜两个对照，才允许称 IED-specific transfer。

organizational extension 使用不看 state 预先冻结的连续 target：IED–ictal reuse、early recruitment extent/entropy、propagation speed、laterality。核心问题是发作前 state/IED decoder 是否预测同一次发作如何复用该患者的冻结 IED propagation template。它仍是 H2b，不是 H3。

## 11. 推断与 claim ladder

主推断使用 patient-first hierarchical bootstrap、随机效应或相应校准置换；sign test 只作方向补充。完整 grid 才允许解释 Brier/calibration；sampled risk set 不解释为绝对概率。

|结果|允许结论|
|---|---|
|T−且 assay power 不足|不可估计|
|T−且 assay power 充分|当前 checkpoint 无可用 transfer evidence|
|T+、M−、D−|transferable representation，不是持续状态|
|T+、M−、仅 5 min D+|acute preictal encoder|
|T+、M+、D−|slow contextual state，未证明 seizure-entry specificity|
|T+、M+、D+|persistent interictal-to-seizure state transfer|
|再有 full 胜两个上游对照|IED-specific transfer|
|再有连续 phenotype bridge|状态与 ictal recruitment organization 相连|

## 12. 科学路线偏移审计

每个阶段开始和结束必须回答以下七问，并写入机器状态：

1. 被测试的 state 是否完全由背景+IED 学习，并在读发作标签前冻结？
2. 主比较是否在同一 `C+H+O` 上增量加入 state？
3. 是否把跨窗口记忆与当前窗口即时滤波分开测量？
4. 时间与几何是否按纯间期估计的时间尺度、在 held-out 发作上 OOS 评价？
5. 科学分母是否是患者和 OOF lead seizure，而不是 seed、anchor 或 fit 数？
6. 当前实验能否区分“可迁移表示”“持续状态”“发作组织”三个层级？
7. IED-specific 或因果措辞是否有独立 ablation/干预支持？

任一答案为否，状态写为 `SCIENTIFIC_ROUTE_DRIFT`，只停止受影响的 claim branch 并回到最近的正确模块；不得用工程 PASS、更多 head 或更漂亮的流形图继续沿错误问题扩展。

## 13. 术语冻结

- `observation`：当前窗口输入，不叫 state；
- `memoryless code`：状态重置后由当前窗口得到的代码；
- `persistent state`：跨窗口因果传播并经 observer 更新的候选状态；
- `transferable representation`：只通过 T；
- `persistent latent state`：T、M、D 均通过；
- `seizure-entry geometry`：OOS 动力学组织，不等于因果机制；
- `IED-specific transfer`：还需 full 胜 background-only 与 IED-shuffled；
- `H3`：IED 对未来状态的作用，本 spec 不检验。
