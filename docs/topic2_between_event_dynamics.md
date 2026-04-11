# Topic 2：间期群体事件之间的时序分析

> 状态：当前正式入口
> 范围：只讨论群体事件作为一个点过程，在事件与事件之间表现出的时间结构。

---

## 1. 这个 topic 只回答什么问题

本 topic 只回答：

1. 群体事件的 `~2 Hz` PSD 峰是不是真正的内禀振荡。
2. IEI 分布是什么，老论文的 power-law 叙事是否站得住。
3. 事件之间是否存在慢时间尺度调制，发生在什么尺度，是否与发作邻近有关。

它**不**回答：

- 单个事件内部传播是否刻板：那是 `docs/topic1_within_event_dynamics.md`
- 慢调制在空间上发生在哪里：那是 `docs/topic3_spatial_soz_modulation.md`

---

## 2. 一句话当前结论

`~2 Hz` 群体事件峰不是 oscillator。现有最稳的解释是：**带不应期的兴奋性点过程 + 多时间尺度慢率调制**。IEI 是 lognormal，不是 power-law；`21/21` 有 specparam 峰的 subject 已全部被 refractory renewal + slow modulation 解释。

---

## 3. 核心证据链

### 3.1 老论文最关键的三条叙事

老论文原本的主张是：

1. 群体事件 PSD 在 `~2 Hz` 有稳定峰
2. IEI 服从 power-law
3. 这个峰反映内禀周期性

当前结论是：

- 第 1 条现象本身存在，但解释错了
- 第 2 条被推翻
- 第 3 条基本不成立

### 3.2 为什么 `~2 Hz` 不是 oscillator

最硬的证据链如下：

- Gamma renewal surrogate：`15/21` 有峰 subject 可由平稳 renewal null 直接解释
- 解析 renewal PSD overlay：`16/21` 的解析峰频与经验峰频 `|Δf| < 1 Hz`
- 两条路径互补后，`19/21 (90%)` 已被解释
- PR-2.5 回填 1084/1096 后，剩余 `2/21` 的 escape subjects 去趋势后峰完全消失

所以最终结论不是 `19/21`，而是更强的：

- **`21/21` specparam-peak subjects 全部已被 refractory renewal + slow modulation 解释**

### 3.3 IEI 分布不是 power-law

用 MLE + LLR 比较后：

- `30/30` subjects 都是 lognormal 显著优于 power-law
- `0/30` 支持 power-law 更优

这条线已经可以封板。老论文 Fig S7 的“幂律”叙事不该再沿用。

### 3.4 IEI 相邻正相关是硬结果

- `30/30` subjects 的 lag-1 方向都为正
- subject-level sign test：`p = 9.31e-10`
- lag-1 `r` 中位数约 `0.299`

这与稳定振荡器驱动矛盾，却与慢率漂移完全一致。

### 3.5 慢调制不是单一时间尺度

PR-2 与 PR-2.5 显示：

- 去趋势分数中位 `0.720`：约 `72%` 的 serial correlation 来自慢漂移
- 去趋势后残差仍为正：约 `28%` 是短程依赖
- 多尺度 `Δ_frac` 近似平坦：慢调制是宽频段、近 `1/f` 型
- IEI 与 `n_participating` 的衰减曲线互相关中位 `0.742`

这说明“慢调制存在”是硬结果，但“单一全局状态变量完全解释一切”说得过头。PR-2.7 的 coherence 修正后只剩中位 `0.358`，因此更稳的说法是：**部分共享驱动 + 局部独立成分**。

### 3.6 发作邻近结果该怎么说

PR-2.7 的 seizure-triggered rate average：

- `[-6h,-1h]` vs `[-12h,-6h]`：`p = 0.019`，`16/21` 同方向
- 但 `post > pre`：`p = 0.016`

所以当前能 defend 的不是“已经发现 pre-ictal biomarker”，而是：

- **存在 seizure-centered broad rate elevation**

这值得继续做，但现在不能吹成纯 pre-ictal buildup。

---

## 4. 当前最可信的结果

- `~2 Hz` peak 不是内禀振荡，而是 dead-time + slow modulation 的结果
- IEI 是 lognormal，不是 power-law
- IEI serial correlation 全部为正，排除 oscillator 叙事
- 慢调制是多时间尺度、宽频段的，不是单一节律
- 宏观 rate trace 的慢漂移在连续时间上可延伸到多小时
- seizure-centered broad elevation 是这条线里第一个真正的 population-level seizure-linked 信号

---

## 5. 仍未解决的问题 / 风险点

- PR-2.7 的 seizure-centered effect 还没拆出多少是 pre、多少是 post、多少是 cluster/circadian baseline。
- coherence 修正后变弱，意味着“单一全局状态变量”这句话必须降级，不能再写满。
- centroid bypass 只是在 legacy 映射框架内说明“窗口内锚点不是主要来源”，不等于完全摆脱 packing 链路。
- hazard 图目前只适合定性，不该过度做参数推断。

---

## 6. 代码与结果入口

- 主文档：`docs/archive/topic2/event_periodicity_analysis.md`
- 方法学审阅：`docs/archive/topic2/event_periodicity_phase2_review_2026-04-05.md`
- 合作者叙事版：`docs/archive/topic2/interictal_population_event_methodological_review.md`
- 代码：`src/event_periodicity.py`
- 脚本：`scripts/run_event_periodicity.py`、`scripts/run_periodicity_phase2.py`、`scripts/plot_periodicity_phase2.py`
- 结果：`results/event_periodicity/`、`results/event_periodicity/phase2/`

---

## 7. 与其他 topic 的边界

- 如果问题是“传播 stereotype 是否真实”，去 `docs/topic1_within_event_dynamics.md`
- 如果问题是“SOZ 与 non-SOZ 的差异发生在哪里”，去 `docs/topic3_spatial_soz_modulation.md`
- Topic 2 可以引用 SOZ dead-time 或 SOZ serial corr 作为现象，但一旦问题变成空间归因，就应该切去 topic 3

---

## 8. 历史文档索引

- `docs/archive/topic2/event_periodicity_analysis.md`
  - 当前最完整的结果与代码地图
- `docs/archive/topic2/interictal_population_event_methodological_review.md`
  - 面向合作者的叙事版历史总文档
- `docs/archive/topic2/event_periodicity_phase2_review_2026-04-05.md`
  - 审阅与可信度边界

这些文档保留详细历史推理；当前正式口径以本文件为准。
