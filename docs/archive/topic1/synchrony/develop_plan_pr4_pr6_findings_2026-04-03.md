# PR4–PR6 间期同步性：event-level 全链路实跑与科学发现（Archived 2026-05-22）

> **归档说明**：这是 `docs/DEVELOP_PLAN.md` §4 旧 L364–453 PR4–PR6 章节的完整原文。
>
> **数字过期警告**：本归档定格于 2026-04-03，cohort 当时只覆盖 Epilepsiae 16 subjects；后续 topic1 主 doc 已扩展到 **29 subjects（Epilepsiae + Yuquan 联合）**，并补出 region-stratified `e` p=0.012 这一探索性发现。引用任何具体数字时，请以 `docs/topic1_within_event_dynamics.md` + `docs/archive/topic1/synchrony/interictal_synchrony_preliminary_report_2026-04-03.md` 为准，不要从本归档复制。
>
> 本归档只保留当时的方法学链路 + 6 大科学发现 + 指标层级判定 + 对后续工作约束，供历史溯源使用。

---

#### PR4–PR6 间期同步性分析：event-level 全链路实跑与科学发现（2026-04-03）

**背景**：PR4–PR6 原计划以 block-level 指标为分析主语。实施过程中发现 block mean 抹掉了事件内部的时间结构，对 within-interval trajectory 的统计力极低。已重构为 **event-level 主链**：PR4 输出每个 HFO 群体事件的三指标行（`sync_legacy_global`, `sync_phase_global`, `sync_span_global`），PR5 按 seizure interval / day-night / phase 对事件行做标注与排除，PR6 消费标注后的事件表跑统计与图。

**实现主链（已落地）**

- `src/interictal_synchrony.py`：`build_event_rows_from_result()` → `event_sync_v1` schema
- `src/interictal_synchrony_aggregation.py`：`_annotate_sync_events_against_intervals()` → event-level interval/window 标注；day/night 直接由事件 epoch 推导，不继承 block label
- `src/interictal_synchrony_analysis.py`：`assign_fixed_window_positions()`, `compute_normalized_trajectory()`, `paired_window_test()`, `within_interval_trend_test()`, `run_pr6_analysis()`
- `scripts/pr6_interictal_sync_figures.py`：Figure A–E 全链 CLI
- `scripts/run_epilepsiae_interictal_synchrony.py`：Epilepsiae event CSV 产出
- `scripts/run_yuquan_interictal_synchrony.py`：Yuquan event CSV 产出
- `scripts/aggregate_epilepsiae_interictal_synchrony.py`：Epilepsiae event-level 聚合
- `scripts/aggregate_yuquan_interictal_synchrony.py`：Yuquan event-level 聚合
- `scripts/interictal_sync_analysis.py`：Yuquan 侧分析入口
- `tests/test_interictal_synchrony_analysis.py`（23 tests）+ `test_interictal_synchrony.py` + `test_interictal_synchrony_aggregation.py`：sync 三件套合计 38 tests pass

**数据规模（Epilepsiae `ready_full_artifacts`）**

- 16 subjects / 2962 source blocks / **~1,280,824** event rows
- 232 intervals（≥3 events each）可用于 within-interval trend 检验
- 128 Post–Pre pairs 可用于固定窗口配对检验

**Yuquan**

- 156 source blocks（T1–T3 有 lagPat 的患者）/ event CSV ~72.6 MB
- 4 个 T1 subjects（gaolan, litengsheng, sunyuanxin, xuxinyi）有手动发作标注可做 interval 分析

##### 关键科学发现

**发现 1：队列水平无显著 Post→Pre 同步性上升**

固定窗口配对 Wilcoxon（interval-level，n_pairs = 128）：

| 指标 | p | effect_size_r | 方向 |
|---|---|---|---|
| legacy | 0.529 | 0.064 | 微弱，无一致方向 |
| phase | 0.380 | 0.089 | 微弱，无一致方向 |
| span | 0.947 | 0.007 | 无效应 |

**结论**：在 Epilepsiae 16 subject 队列上，固定 Post/Pre 1h 窗口的 paired 比较不支持"发作前同步性高于发作后"的假设。

**发现 2：Within-interval trajectory 同样为 null**

每个 interval 内 Spearman ρ（事件 `norm_t` vs 指标值），再对 ρ 的分布做 one-sample Wilcoxon：

| 指标 | n_intervals | median_rho | p |
|---|---|---|---|
| legacy | 232 | −0.0026 | 0.290 |
| phase | 232 | +0.0011 | 0.933 |
| span | 232 | −0.0079 | 0.053 |

**结论**：即使用 event-level 粒度做 within-interval trajectory 分析，三个指标均不显示一致的再同步趋势。span 的 p≈0.053 方向为负（与假设相反）。

**发现 3：个体差异远大于队列效应**

Subject-level direction counts（phase pre−post）：11 负 / 5 正。Adaptive-Kuramoto 双判据筛选（phase pre>post AND within-interval ρ>0）仅 3/16 弱符合（subject 548, 1125, 958）。

**几个有代表性的个体**：
- Subject 916（52 seizures, 435 blocks）：within-interval legacy ρ = +0.175, p=0.001（**最强正趋势**）
- Subject 1073（最大数据量，>216h）：legacy ρ = −0.674（**强负趋势**）
- Subject 548（=论文 E14）：legacy ρ = +0.133（正，与论文一致）

**发现 4：Legacy 指标的 "0.6 wall" 是数学伪影**

Subject 1073 的 194,521 个事件中，73,896 个 `n_participating = 3`，其 `sync_legacy_global` 模态为 ~0.5918。理论推导：3 通道均匀 lag 模式的 legacy 指标数学极限 ≈ 0.5917517。这不是生物学信号，而是低通道数的离散化几何效应。

**发现 5：Core/Global 在 Epilepsiae 上不可区分**

Epilepsiae 的 lagPat 通道来自 legacy `avgPickChns`（基于 refine count 阈值选出的"高事件率"通道），不含临床 SOZ 标注。实测 1,280,824 行中 `n_core_channels == n_channels` 占 100%。SQL `electrode.focus_rel` 字段含 `i`（in-focus）/ `e`（extra-focal）/ `l`（疑似边界/不确定），但当前管线未消费该字段。

**发现 6：论文 Figure 7B/C 数值可复现**

在 subject 548（= E14）上按论文同口径（event-level, 发作前 1h, Pearson）复算：r = 0.147317, p = 3.201e−14, n_seizures = 14, n_events = 2628。与论文报告数字完全一致。**但这是单 subject event-level Pearson**，不能简单外推到队列水平的 interval-level 分析。

**指标层级判定**

| 指标 | 角色 | 理由 |
|---|---|---|
| **phase**（exp pairwise） | 主科学指标 | 无低通道离散化伪影；对通道数增长最稳定 |
| **legacy**（pdelay） | legacy 可比性 | 受 `n_participating` 几何效应严重污染，但老论文用此指标 |
| **span**（pairwise coincidence） | 敏感性附录 | 对窗口几何敏感；方向不稳定 |

##### 对后续工作的约束

- PR6 统计结论已出：**队列水平 null**。继续在相同数据上换统计方法不会改变结论实质。
- 破局只能来自：(a) 真实 SOZ label 重建 core/penumbra 对比，(b) `n_participating` 作协变量的条件分析，(c) subject 分层描述而非全队列假设检验，(d) 更细时间粒度（event timestamp 而非 1h block epoch）
- Yuquan 侧 PR6 仅部分跑通（event export 完成，聚合与图尚需补 Yuquan seizure interval inventory = PR3 依赖）
- 不得把"pooled million-event Spearman p 极小"当成生物学发现——那是 N 的暴力，效应量 ρ < 0.04
