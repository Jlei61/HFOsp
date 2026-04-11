# 论文总论与文档索引

> 状态：当前正式入口
> 目的：给人和 Agent 一个稳定的总索引，先回答“这篇论文现在到底讲哪 3 个 topic、各自结论是什么、应该先读哪里”。

---

## 1. 论文现在的 3 个 topic

### Topic 1：间期事件内部时序结构

关注单个群体事件内部的时序组织，而不是事件与事件之间的间隔。

- 正式入口：`docs/topic1_within_event_dynamics.md`
- 核心数据对象：`lagPatRank`、`eventsBool`、`chnNames`、event-level synchrony rows
- 核心问题：
  - 单个群体事件内部的传播顺序是否刻板、是否多模态、是否与 SOZ 有关
  - 单个事件内部/事件级同步性是否支持“发作前后重置”叙事

### Topic 2：间期群体事件之间的时序分析

关注群体事件作为一个点过程，在事件与事件之间表现出什么时间结构。

- 正式入口：`docs/topic2_between_event_dynamics.md`
- 核心数据对象：group-event timestamps、IEI、PSD、rate trace、`n_participating`
- 核心问题：
  - `~2 Hz` 峰是不是内禀振荡
  - IEI 是否 power-law
  - 慢时间尺度调制是否存在、发生在什么时间尺度、是否与发作邻近有关

### Topic 3：Where / SOZ 空间归因

关注慢调制和时序差异在空间上发生在哪里，尤其是 SOZ / non-SOZ 的分离。

- 正式入口：`docs/topic3_spatial_soz_modulation.md`
- 核心数据对象：per-channel relaxed-refine events、SOZ labels、i/l/e labels
- 核心问题：
  - lagPat 框架为什么回答不好 where
  - per-channel 框架下 SOZ 与 non-SOZ 是否真的不同
  - 哪部分是全局调制，哪部分是 SOZ 的局部短程记忆

---

## 2. 一句话总论

### Topic 1

间期群体事件内部存在稳定但多模态的传播结构；cluster-aware 分析显示刻板性真实存在，但 SOZ 优势目前仍偏探索性。事件级同步性在线队列水平总体为 null，仅 extra-focal phase synchrony 出现探索性 `pre > post`。

### Topic 2

`~2 Hz` 群体事件峰不是内禀振荡器证据；现有证据支持“带不应期的兴奋性点过程 + 多时间尺度慢调制”。`21/21` 有 specparam 峰的 subject 已被 refractory renewal + slow modulation 解释。

### Topic 3

lagPat 群体事件框架的 SOZ / non-SOZ 对比受结构性选择偏差污染。per-channel relaxed-refine 分析显示：原始 serial correlation 没有 SOZ 差异，但去趋势后 SOZ 更像保留了额外的局部短程记忆。

---

## 3. 先读哪份文档

### 如果你只想知道当前正式结论

1. `docs/paper_overview.md`
2. `docs/topic1_within_event_dynamics.md`
3. `docs/topic2_between_event_dynamics.md`
4. `docs/topic3_spatial_soz_modulation.md`

### 如果你要看历史证据链或审阅意见

- Topic 1 历史来源（`docs/archive/topic1/`）：
  - `docs/archive/topic1/interictal_group_event_internal_propagation.md`
  - `docs/archive/topic1/interictal_synchrony_preliminary_report_2026-04-03.md`
- Topic 2 历史来源（`docs/archive/topic2/`）：
  - `docs/archive/topic2/event_periodicity_analysis.md`
  - `docs/archive/topic2/interictal_population_event_methodological_review.md`
  - `docs/archive/topic2/event_periodicity_phase2_review_2026-04-05.md`
- Topic 3 历史来源（`docs/archive/topic3/`）：
  - `docs/archive/topic3/spatial_modulation_soz_analysis.md`

这些历史文档保留事实、审阅和阶段性推理，但不再是首选入口。

---

## 4. 结果与代码入口

### Topic 1

- 结果：`results/interictal_propagation/`
- 代码：`src/interictal_propagation.py`
- 脚本：`scripts/run_interictal_propagation.py`、`scripts/plot_interictal_propagation.py`

### Topic 2

- 结果：`results/event_periodicity/`、`results/event_periodicity/phase2/`
- 代码：`src/event_periodicity.py`
- 脚本：`scripts/run_event_periodicity.py`、`scripts/run_periodicity_phase2.py`、`scripts/plot_periodicity_phase2.py`

### Topic 3

- 结果：`results/spatial_modulation/`、`results/refine_soz_validation/`
- 代码：`src/event_periodicity.py` 中 per-channel / SOZ helpers，`src/group_event_analysis.py`
- 脚本：`scripts/audit_gpu_npz.py`、`scripts/run_spatial_modulation.py`、`scripts/plot_spatial_modulation.py`

---

## 5. 当前最稳的科学结论

- Topic 1：内部传播不是单一模板，而是多模态但稳定的病理网络传播路径；legacy MI 可复现，cluster-aware τ 明显高于整体 τ。
- Topic 1：interictal synchrony 在 cohort level 没有支持“post-ictal reset / pre-ictal resynchronization”；唯一值得继续追的是 extra-focal `phase_e` 的 `pre > post`。
- Topic 2：`~2 Hz` peak 不是 oscillator；IEI 是 lognormal，不是 power-law。
- Topic 2：IEI 相邻正相关是硬结果，支持慢率漂移；去趋势后仍保留短程依赖。
- Topic 2：rate trace 存在 seizure-centered broad elevation，但现在还不能诚实地叫作 pre-ictal biomarker。
- Topic 3：SOZ / non-SOZ 的 raw serial correlation 差异在 per-channel 框架下消失，说明旧 lagPat 结果部分混入了事件率与通道选择偏差。
- Topic 3：SOZ 更像是“全局调制之上叠加局部短程记忆”，而不是简单地“整体更同步”或“整体更周期”。

---

## 6. 规则入口

- Topic 1 rule：`.cursor/rules/topic1-within-event-dynamics.mdc`
- Topic 2 rule：`.cursor/rules/topic2-between-event-dynamics.mdc`
- Topic 3 rule：`.cursor/rules/topic3-spatial-soz-modulation.mdc`

旧 rule：

- `.cursor/rules/interictal-propagation-pr-plan.mdc`
- `.cursor/rules/event-periodicity-pr-plan.mdc`

目前保留为过渡入口，防止旧引用失效。
