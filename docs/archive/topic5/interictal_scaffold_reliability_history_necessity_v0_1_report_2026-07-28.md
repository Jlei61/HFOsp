# Topic 5 间期静态 scaffold 可靠性与短历史必要性 v0.1 收口报告

日期：2026-07-28

## 1. 一句话结论

34 名患者的间期群体事件中存在两个可重复的对象：

1. 每个触点参与间期事件的患者特异静态概率场；
2. 对下一 rank set 有用的、主要集中在最近 2–3 个 rank set 内的顺序信息。

这支持继续保留“间期 rank 事件自监督学习”这条线，但把模型对象收窄为
**静态 contact scaffold + 有限短历史**。它不支持把 full-history GRU 解释成无界 latent
dynamics，也没有测试病理轴、source 自动恢复或发作早期迁移。

## 2. 执行状态

> **科学合同执行完成度：100/100**

- cohort：34 人，18 名 Epilepsiae、16 名 Yuquan；
- split：冻结的 chronological train80 / heldout20；
- seeds：`20260725`、`20260726`、`20260727`；
- 新训练：
  - 34 人 × 3 seeds × H1/H2/H3 = 306 个有限历史模型；
  - 34 人 × 3 seeds × matched H3 rank-shuffle = 102 个模型；
  - 合计 408 个新模型；
- 推断单位：先在患者内合并 seeds，再做患者水平统计；
- target seal：所有 static、history 和 matched-shuffle summary 均为
  `ictal_target_read=false`；
- 运行状态：static、main history、matched shuffle 的 `DONE.json` 均存在；
- 资源：最大 GPU allocated memory 为 445,306,880 bytes，未出现 OOM；
- 日志：204 个 formal fold log，加 launcher/summary 等共 209 个 log；未检出
  Traceback、OOM、CUDA error 或 NaN；
- 测试：
  - base 环境：3 passed、1 skipped（无 PyTorch 时正确跳过模型测试）；
  - `cuda_env`：18 passed。

## 3. 实际测了什么

### 3.1 静态 contact participation scaffold

对每名患者、每个触点定义：

\[
q_i(D)=\frac{1}{|D|}\sum_{e\in D}\mathbf 1(i\text{ 参与事件 }e).
\]

主检验比较 train80 和 heldout20 的 contact-wise Spearman 相关。它测的是：

> 大量间期群体事件汇总后，哪些触点经常参与这一患者特异 contact topography 是否能在
> 未见事件中复现。

它不是 A/B 分类、完整 rank distribution、物理轴或传播方向。

对照包括：

- train80 chronological first-half vs second-half；
- train80 odd vs even events；
- within-shaft circular permutation null；
- 25–2000 个事件的 deterministic subsampling saturation。

### 3.2 有限历史 next-set 自监督任务

输入为一场间期群体事件已经观察到的 contact rank-set prefix，输出为：

- 下一 rank set；
- 或 STOP。

H1、H2、H3 模型使用相同 contact encoder、decoder、likelihood、优化器和评分分母；
唯一差别是预测下一步前，只重放最近 1、2 或 3 个 rank set。所有模型仍对完整已观察
prefix 做 candidate mask，不会重新选择已参与触点，也不会读取最终事件长度。

主指标为 heldout20 event-balanced next-set/STOP NLL，越低越好。

matched H3 rank-shuffle 使用完全相同的 H3 架构和覆盖，只在训练与 heldout calibration
中打乱参与触点的 rank；最终仍在原始顺序的 heldout20 上评分。这一对照用于判断增益是否
来自真实顺序，而不是模型容量或参与集合。

## 4. 结果

### 4.1 静态参与场高度可重复

| 指标 | 结果 |
|---|---:|
| train80 vs heldout20 Spearman 中位数 | 0.893 |
| 95% bootstrap CI | [0.868, 0.936] |
| 正值患者 | 34/34 |
| one-sided Wilcoxon P | \(1.82\times10^{-7}\) |
| chronological half Spearman 中位数 | 0.907 |
| odd-even Spearman 中位数 | 0.997 |

在 33 名可做 structured null 的患者中：

| 指标 | 结果 |
|---|---:|
| observed − within-shaft null 中位数 | 0.685 |
| 95% bootstrap CI | [0.536, 0.727] |
| 正值患者 | 33/33 |
| patient-level null P<0.05 | 31/33 |
| one-sided Wilcoxon P | \(1.16\times10^{-10}\) |

因此，结果不是简单由同一 shaft 上相邻触点的位置排列解释。

### 4.2 大约数百个事件即可稳定估计静态场

| train80 事件数 | 可用患者 | 中位 Spearman | 与 full train80 相差不超过 0.05 |
|---:|---:|---:|---:|
| 25 | 34 | 0.799 | 32.4% |
| 50 | 34 | 0.853 | 44.1% |
| 100 | 34 | 0.858 | 67.6% |
| 200 | 34 | 0.882 | 88.2% |
| 500 | 32 | 0.898 | 96.9% |

安全表述是：

> 约 200 个事件时，30/34 名患者已经落在 full train80 估计的 Spearman 0.05 范围内；
> 500 个事件时为 31/32。

1000/2000 事件只在事件量足够的患者中可计算，存在 cohort attrition，不能用其更高相关
证明所有患者都需要或都达到这一事件数。

### 4.3 最近 2–3 个 rank set 含有稳定增量

所有 gain 定义为较简单模型 NLL 减去较丰富模型 NLL，因此正值表示后者更好。

| 比较 | 中位 NLL gain | 95% CI | 正值患者 | two-sided P |
|---|---:|---:|---:|---:|
| H2 over H1 | 0.0172 | [0.0117, 0.0240] | 32/34 | \(1.01\times10^{-6}\) |
| H3 over H2 | 0.0113 | [0.0089, 0.0154] | 29/34 | \(2.95\times10^{-8}\) |
| Full over H3 | -0.0010 | [-0.0051, 0.0035] | 16/34 | 0.436 |
| Full over first-order | 0.0358 | [0.0246, 0.0495] | 30/34 | \(5.14\times10^{-6}\) |

这给出清楚的层级结论：

- 不止最后一个 rank set 有信息；
- 最近 2–3 个 rank set 提供稳定增量；
- 使用整场全部历史没有超过 H3；
- 因此数据支持 bounded short memory，不支持 unbounded full-history necessity。

### 4.4 顺序增量不是 GRU 容量造成的

架构匹配的 H3 对照结果：

| 比较 | 中位 NLL gain | 95% CI | 正值患者 | two-sided P |
|---|---:|---:|---:|---:|
| ordered H3 over H3 rank-shuffle | 0.0261 | [0.0168, 0.0348] | 27/34 | 0.00361 |

因此，短历史增益至少部分依赖 rank set 的真实顺序，而不只是哪些触点最终参与。

## 5. 旧比较器标签错误及修正

旧报告曾把 `ordered_history_nll_gain` 写成 full GRU vs rank-shuffle。该字段实际是：

```text
strongest_nonrecurrent_nll - full_history_gru_nll
```

从原始 heldout NLL 重新计算真正的 full GRU vs 独立训练 rank-shuffle GRU 后：

- 中位 gain = 0.0408；
- 27/34 为正；
- one-sided \(P=2.51\times10^{-4}\)。

但 full GRU vs strongest nonrecurrent prefix model 只有：

- 中位 gain = 0.0010；
- 17/34 为正；
- one-sided \(P=0.440\)。

因此正确结论是：

> 真实事件顺序相对 rank-shuffle 有 heldout 泛化信息，但 full-history recurrence 没有超过
> 最强 nonrecurrent prefix model。

对应 audit、旧 Figure 6 producer、归档报告和 manuscript-facing draft 已统一修正。

## 6. 与核心科学目标的关系

### 没有偏移的部分

- 输入仍是 34 人的原始 contact rank-set 事件，不是 IEI；
- 任务仍是间期事件内 next-set 自监督，不是 A/B 聚类；
- 模型学习的是接触点参与和局部顺序，不把 A/B 当 label；
- 所有统计先按患者汇总；
- 没有读取 early-ictal target；
- 没有把预测性能写成病理轴、source 或生物 latent state。

### 对论文主张的增量

这轮结果补充支持：

1. 间期病理网络不只体现在少量聚类模板中，还体现在可重复的患者特异 contact
   participation scaffold；
2. 事件内部顺序不是随机排列，最近 2–3 个 rank set 对下一步有独立信息；
3. 适合论文的模型对象是“稳定 contact scaffold 上的有限短时序”，而不是无界 GRU
   hidden state。

### 仍然不能声称

- RNN 自动读出了患者病理轴；
- RNN 自动识别了 A/B source；
- 间期传播轨迹在发作期被逐步 replay；
- 当前实验预测了 clinical onset 后的 early-ictal energy；
- 2–3 步历史等同于细胞级兴奋/抑制状态或真实生物时间常数。

## 7. 对 RNN 架构的验收意见

当前 GRU 作为**有限历史必要性的判别器**是合理的，因为 H1/H2/H3 只改变可见历史深度，
其余训练和评分合同保持一致。

它不应作为最终机制模型直接进入主张。下一版若继续结构化建模，建议：

- 把 H3 作为性能参考和历史上限；
- 可解释模型只需表达最近 2–3 个 rank set 的 state；
- 任何 structured/low-rank 模型都必须与 H3 而不是 full GRU 比较；
- 不再为了提高 AUC 增加无界 hidden capacity；
- early-ictal bridge 继续使用独立冻结的静态 contact field 合同，不能从本轮结果偷渡。

## 8. 图的科学含义

综合图包含四块：

- **A**：train80→heldout20 的静态 participation field 相对 within-shaft null；
- **B**：参与场估计随事件数增加的饱和过程；
- **C**：first-order、H1、H2、H3、H3-shuffle 和 full-history 的 heldout NLL；
- **D**：H2/H1、H3/H2、full/H3 和 ordered-H3/shuffled-H3 的患者水平配对增量。

这张图适合作为 Supplementary 或 Figure 6 中间期自监督模块的候选，不应单独冒充六块
Figure 6 的完整跨状态结论。early-ictal bridge 应由其自身冻结 target 和独立 panel 承担。

## 9. 主要产物

- 规范：
  `docs/superpowers/specs/2026-07-28-topic5-static-scaffold-reliability-history-necessity-v0_1.md`
- 执行计划：
  `docs/superpowers/plans/2026-07-28-topic5-static-scaffold-reliability-history-necessity-v0_1.md`
- 静态结果：
  `results/topic5_interictal_scaffold_reliability_history_necessity/static_reliability_v0_1/`
- 有限历史结果：
  `results/topic5_interictal_scaffold_reliability_history_necessity/history_runs_v0_1/`
- matched H3 shuffle：
  `results/topic5_interictal_scaffold_reliability_history_necessity/history3_rank_shuffle_runs_v0_1/`
- paper-ready PNG/PDF：
  `results/topic5_interictal_scaffold_reliability_history_necessity/figures/`

## 10. 最终验收

### P0

无。

### P1

无未关闭的执行问题。科学边界必须继续保留：

- static participation field 不等于完整 rank distribution；
- short-history prediction 不等于 latent mechanism；
- target sealed experiment 不等于 early-ictal transfer。

### 结论

```text
PASS_BOUNDED_STATIC_SCAFFOLD_AND_SHORT_HISTORY
```

RNN 线可以继续，但应以 H3 bounded-history 作为上限与参考；当前没有依据继续扩大 full-GRU
历史或重新追求自动 axis/source 恢复。
