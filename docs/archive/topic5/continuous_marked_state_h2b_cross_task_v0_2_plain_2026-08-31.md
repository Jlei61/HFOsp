# H2b Cross-task Transfer v0.2 白话收口

## 一句话

**本轮没有建立 H2b 跨任务状态证据。**唯一达到主层支持的患者中，冻结状态相对当前背景的增量很小、落在置换零带内，而且持续记忆没有胜过只看当前窗口；低支持患者有一些方向有利的读数，但正确时刻状态没有优于匹配的错误时刻。

## 这一步真正问了什么

模型先只用连续背景和间期事件的 timing/完整 contact mark 学状态。读入发作标签之前，observer、state update、generator 和 IED decoder 全部冻结。发作任务只训练一个很小的 ridge risk probe，因此发作标签不能回头改造状态。

问题是：这个从间期任务中学到的冻结状态，能否在“近期 IED 历史”和“当前背景观察”之外，帮助区分真正的发作前时刻和同患者、同记录段中的普通时刻。

## 数据规模

- R1.7 冻结清单：17 位患者、85 个训练单元，其中 75 个 checkpoint 可读。
- R1.7B 是 exploratory development extension：consumer-side audit 已逐单元重算哈希，但它不满足旧 v0.1 的 50-fit formal release gate，因此本轮不能升级为正式 H2b confirmation。
- 124 条 development 发作进入 crosswalk；正式和 sealed 分区没有打开。
- 10 位患者完成原始背景读取和发作支持审查，生成 46 个 checkpoint-seed 状态缓存。
- 30 分钟支持：1 位 primary、3 位 LOSO sensitivity、5 位 descriptive、1 位 not estimable。
- 9 位患者进入主 risk probe；15 个 primary/wrong-time 分析中，13 个完成 100 次置换，2 个在 30 分钟主端点明确记为不可估计。

## 核心结果

唯一 primary 患者 `epilepsiae_548` 有 10 次合格发作，但 chronological TEST 最终只有 2 个 held-out risk sets：

- `state - observation = -0.0212`，表面上方向有利，但在置换零带 `[-0.0862, +0.0501]` 内；
- `persistent - memoryless = +0.0071`，不支持跨窗口持续记忆；
- donor-valid wrong-time 子集降到 LOSO 层，`correct - wrong = +0.2016`，错误时刻反而更好。

因此，主层结果不能支持“间期任务学到的持续状态能跨任务预测发作接近”。

低支持层只能作为探索：

- LOSO 三位患者的 30 分钟 `state - observation` 为 2/3 方向有利，中位数 `-0.0011`，几乎为零；
- descriptive 四位可评分患者为 4/4 方向有利，中位数 `-0.1215`，但只有 4 人，双侧符号检验 `p=0.125`；
- 30 分钟正确时刻状态没有稳定胜过错误时刻：LOSO 为 0/1，descriptive 为 1/4。

这些读数最多是扩大样本后值得复查的线索，不能提升为 H2b 阳性。

## 发作类型二级探索

只连接分析前已经存在并冻结的 `broad_ER` / `gamma_ER` 标签，没有重新聚类，也没有事后创造 early-recruitment 指标。6 位患者执行了 phenotype probe，但 18 个患者×目标单元中只有 2 个给出有限结果，而且两者最终都只评分 2 次发作：

- `epilepsiae_253 / broad_ER`：描述层，`state - observation = -0.1663`；
- `epilepsiae_548 / broad_ER`：主层，`state - observation = -0.4592`。

其余 16/18 个单元因目标缺失、类别不足或样本不足不可估计，因此没有跨患者亚型迁移证据。

## 旧 E384 如何收口

v0.1 的 E384 产物在 30 分钟实际只有 4 次而不是旧文字中的 5 次合格发作；`state - observation = +0.1560`，方向不利，只保留为单患者描述性阴性。

## 当前安全结论

当前应写成：

> 在 development 数据中，当前冻结 R1.7 间期状态尚未显示可靠的跨任务发作前增量。一个低支持描述层出现方向一致的探索性信号，但唯一主层患者未通过置换、持续性和时刻专属性三项解释检验。因此 H2b 未建立，也未被证明不存在。

不能写成发作因果机制、临床预测性能、cohort confirmation 或 H3 的 IED→state 证据。

## 权威产物

- 结果根：`results/epi_prssm/continuous_marked_state/h2b_cross_task/v0_2/`
- 白话报告：`reports/h2b_cross_task_v0_2_plain.md`
- 技术报告：`reports/h2b_cross_task_v0_2_technical.md`
- 机器审计：`reports/machine_audit.json`，`PASS_COMPLETE`
- patient-first 表：`reports/per_patient_lead_results.csv`
- 队列完成标记：`COHORT_RUN_COMPLETE.json`
- 接手说明：`CURRENT_HANDOFF.md`
