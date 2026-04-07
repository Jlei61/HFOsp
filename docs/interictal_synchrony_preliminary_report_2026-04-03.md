# 间期同步性分析初步结果（经审阅修订 v2）

## 文档目的

本文件是 Epilepsiae + Yuquan 双队列间期同步性分析的**统计报告**。所有正式统计以 seizure interval 为统计单位，event-level 结果仅作探索性参考。

---

## 一、数据概览

### 1.1 队列

| 项目 | Epilepsiae | Yuquan |
|---|---|---|
| 有 lagPat 的 subject 数 | 16 | 13 |
| 总事件行 | 1,280,824 | 187,956 |
| 被分配到 seizure interval 的事件 | 1,045,658（两队列合计） | — |
| 合格 seizure interval 数 | 253（合计） | — |
| 纳入 fixed-window 比较的 interval | 141 | — |
| 纳入 trajectory 分析的 interval | 249–253 | — |

### 1.2 SOZ / 区域定义

- **Epilepsiae**：`electrode.focus_rel` 取值 `i`（in-focus / SOZ）、`l`（lesion）、`e`（extra-focal），从 SQL 元数据提取（`results/epilepsiae_electrode_focus_rel.json`）。`core_channels` = `i`-labeled channels（同 `results/epilepsiae_soz_core_channels.json`）。
- **Yuquan**：SOZ 来自 `p16_subs_info.py` 手工标注（`results/yuquan_soz_core_channels.json`，20 个 subject 有非空 SOZ）。13 个有 lagPat 的 subject 中，8 个有 meaningful core/penumbra 对照（SOZ 通道与 lagPat 通道有重叠且 penumbra > 0）。

### 1.3 同步性指标

- **phase**（主指标）：half-circle phase-order，对 lag_raw 绝对值做 $[0, \pi]$ 相位映射后取 Kuramoto order parameter，值域 $[0,1]$
- **legacy**（兼容指标）：$1 - \text{std}(\text{lag}) / \text{range}(\text{lag})$，低通道数时离散化严重
- **span**（附录指标）：$1 - \text{range}(\text{lag}) / \text{window\_duration}$，混入窗口长度效应

所有指标使用 `lag_raw`（绝对值，秒），**不是 rank**。

---

## 二、Cohort-level 正式结果

> 统计来源：`results/interictal_synchrony/analysis/combined/pr6_statistics_summary.json`
> 统计单位：seizure interval（interval-first）
> 队列：Epilepsiae + Yuquan 合并

### 2.1 Phase 同步性：固定窗口 Post vs Pre

| 指标 | n_pairs | p | effect r | median post | median pre |
|---|---|---|---|---|---|
| phase_all | 138 | 0.279 | 0.106 | 0.488 | 0.486 |
| phase_core (SOZ-only) | 137 | 0.967 | −0.004 | — | — |

**结论**：全通道和 SOZ-only 均未显示 cohort-level post-ictal reset 或 pre-ictal resynchronization。

### 2.2 Phase 同步性：Within-Interval Trajectory

| 指标 | n_intervals | median ρ | Wilcoxon p | n_pos / n_neg |
|---|---|---|---|---|
| phase_all | 249 | 0.004 | 0.589 | 131 / 118 |
| phase_core | 247 | 0.004 | 0.643 | 129 / 118 |

**结论**：interval 内部无系统性时间趋势。

### 2.3 Legacy 和 Span（验证性）

| 指标 | fixed-window p | trajectory p |
|---|---|---|
| legacy_all | 0.426 | 0.290 |
| legacy_core | 0.516 | 0.297 |
| span_all | — | — |
| span_core | — | — |

**结论**：`legacy` 与 `phase` 一致，不改变 null 结论。`span` 降级至附录。

---

## 三、区域分层分析（i / l / e）

> 仅 Epilepsiae 子集有 `focus_rel` 标注

### 3.1 固定窗口 Post vs Pre

| 区域 | n_pairs | p | effect r | median post | median pre |
|---|---|---|---|---|---|
| phase_i (SOZ) | 94 | 0.646 | −0.058 | 0.437 | 0.431 |
| phase_l (lesion) | 40 | 0.543 | −0.123 | 0.521 | 0.515 |
| **phase_e (extra-focal)** | **89** | **0.012** | **0.313** | **0.392** | **0.456** |

### 3.2 Within-Interval Trajectory

| 区域 | n_intervals | median ρ | p |
|---|---|---|---|
| phase_i | 153 | 0.005 | 0.940 |
| phase_l | 58 | −0.010 | 0.612 |
| phase_e | 133 | 0.014 | 0.129 |

### 3.3 解读

**`phase_e`（extra-focal）是本轮唯一的 nominally significant result**：

- Fixed-window Post vs Pre p = 0.012, r = 0.31（medium effect）
- 方向：pre-ictal synchrony **高于** post-ictal（median pre = 0.456 > median post = 0.392）
- Bonferroni 校正（3 regions）后 p = 0.037，仍保留在 α = 0.05 水平
- Trajectory 分析 p = 0.129，方向一致但不显著

**解读**：

1. 这说明上一版报告中"global-core 差异提示 extra-focal 驱动"的假设**部分得到验证**。
2. 具体方向为：extra-focal 区域的同步性在发作后较低、在下一次发作前较高——这与"发作打断非 SOZ 区域的远程同步"的叙事一致。
3. SOZ 内部（`i`）和 lesion 区域（`l`）**没有此效应**，说明这不是全脑统一模式。
4. n = 89 pairs / 12 subjects 的样本量有限，结论应标注为 **exploratory-significant**。

> **当前正式表述**：Extra-focal phase synchrony 在固定窗口分析中显示出 nominally significant 的 pre > post 趋势（p = 0.012, r = 0.31），Bonferroni 校正后勉强保留。SOZ 内部和 lesion 区域无此效应。该发现方向与"发作暂时打断远程同步"一致，但需更大样本验证。

---

## 四、事件频率（Event Rate）

| 比较 | n_pairs | p | median post (events/hr) | median pre/mid (events/hr) |
|---|---|---|---|---|
| post vs pre | 138 | 0.361 | 318.0 | 290.5 |
| post vs mid | 136 | 0.461 | 318.0 | 266.0 |

**结论**：间期群体事件的发生频率在 post/mid/pre 窗口间**无显著差异**。发作前后 HFO 群体事件的密度没有系统性改变。

---

## 五、Yuquan SOZ in/out 结果

### 5.1 队列信息

13 个 subject 有 lagPat，其中 8 个有 meaningful SOZ core/penumbra 对照：

| subject | n_core (SOZ) | n_penumbra | events |
|---|---|---|---|
| chenziyang | 4 | 6 | 9,609 |
| hanyuxuan | 7 | 15 | 5,468 |
| huanghanwen | 5 | 5 | 484 |
| litengsheng | 15 | 9 | 2,070 |
| liyouran | 12 | 5 | 2,346 |
| wangyiyang | 21 | 1 | 1,919 |
| xuxinyi | 13 | 2 | 9,646 |
| zhangjinhan | 4 | 1 | 6,196 |

（chengshuai, dongyiming, gaolan, huangwanling, sunyuanxin 全通道 = core，无 penumbra 对照）

### 5.2 结果

Yuquan subjects 纳入合并 PR6 分析。由于 Yuquan event CSV 不含 i/l/e 列（仅有 core/penumbra），区域分层仅来自 Epilepsiae 子集。Yuquan 的贡献体现在 cohort-level phase_all 和 phase_core 检验中。

合并后 cohort phase_all p = 0.279（单独 Epilepsiae 时 p = 0.380），方向一致、结论不变。

---

## 六、三类指标结论级别（更新）

### 主指标：`phase`

- 保留为主指标
- Cohort-level null 不变
- **Extra-focal 子区域** 出现 nominally significant pre > post（需后续验证）

### 验证性指标：`legacy`

- 降级。与 phase 一致，不改变任何结论方向

### 附录指标：`span`

- 降级至附录。物理语义混杂

---

## 七、更新后的正式摘要

> 在 Epilepsiae（16 subjects）+ Yuquan（13 subjects）的合并间期同步性分析中，以 seizure interval 为统计单位的 fixed-window 和 within-interval trajectory 检验均未发现 cohort-level 的 phase-order synchronization post-ictal reset 或 pre-ictal resynchronization（phase_all p = 0.279, phase_core p = 0.967）。事件频率（events/hour）在 post/mid/pre 窗口间无显著差异（p = 0.361）。
>
> 区域分层分析揭示了一个探索性发现：**extra-focal 区域**（Epilepsiae `focus_rel == 'e'`）的 phase synchrony 在 pre-ictal 窗口显著高于 post-ictal 窗口（p = 0.012, r = 0.31, Bonferroni-corrected p = 0.037），而 SOZ（`i`）和 lesion（`l`）区域无此效应。该方向提示发作可能暂时打断远程（非 SOZ）区域的同步性，但样本量有限（89 interval pairs / 12 subjects），需更大队列验证。

---

## 八、下一步建议

1. **验证 phase_e 效应**：增加样本量或用 bootstrap confidence interval 量化 effect size 的稳定性
2. **Subject-level case series**：围绕 phase_e 效应做 per-subject 拆解，看是否由少数 subject 驱动
3. **Event-timestamp 精细化**：从 1h block mean 细化到 event-level 时间序列分析
4. **Prediction framing**：测试 pre-ictal synchrony buildup 是否有 seizure prediction 价值

---

## 九、产出物清单

| 产出物 | 路径 |
|---|---|
| Epilepsiae focus_rel JSON | `results/epilepsiae_electrode_focus_rel.json` |
| Epilepsiae SOZ core JSON | `results/epilepsiae_soz_core_channels.json` |
| Yuquan SOZ core JSON | `results/yuquan_soz_core_channels.json` |
| Epilepsiae region-stratified event CSV | `results/interictal_synchrony/epilepsiae_ready_full_artifacts/epilepsiae_region_stratified_events.csv` |
| Epilepsiae aggregated + regions CSV | `results/interictal_synchrony/epilepsiae_ready_full_artifacts/aggregated/epilepsiae_sync_event_annotations_with_regions.csv` |
| Yuquan SOZ event CSV | `results/interictal_synchrony/yuquan_soz/yuquan_soz_interictal_sync_events.csv` |
| Yuquan SOZ aggregated CSV | `results/interictal_synchrony/yuquan_soz/aggregated/yuquan_sync_event_annotations.csv` |
| PR6 统计 JSON | `results/interictal_synchrony/analysis/combined/pr6_statistics_summary.json` |
| PR6 Figures A–F | `results/interictal_synchrony/analysis/combined/figures/` |
| Figure F: event rate boxplot | `results/interictal_synchrony/analysis/combined/figures/figure_f_event_rate.png` |

---

## 十、文档状态

- 状态：初步结果 v2
- 日期：2026-04-04
- 适用范围：Epilepsiae + Yuquan 双队列 interictal synchrony PR4/PR5/PR6
- 统计口径：seizure_interval（interval-first）
- 关键更新：(1) 加入 Yuquan SOZ 分层，(2) i/l/e 区域分层，(3) 事件频率分析，(4) phase_e exploratory-significant finding
