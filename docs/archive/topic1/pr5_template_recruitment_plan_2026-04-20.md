# PR-5 计划：Template Recruitment Around Seizures，附 Novel-Template 证伪 gate

> 性质：archive / 计划文档（待执行）
> 触发：PR-4C P0 已封板（2026-04-19），geometry seizure-proximity 调制 cohort-level null；唯一稳健剩余信号是 `rate_by_template`。用户要求把这条信号变成 PR-5 的正式分析，同时把方向 1（KONWAC v2）放到 PR-6。
> 上游入口：`docs/topic1_within_event_dynamics.md` §7.6 / §7.8 / §7.9 / §10
> 配套规则：`.cursor/rules/topic1-within-event-dynamics.mdc` PR Status 表

---

## 1. 一句话定位

PR-5 不是再去翻案"模板几何受发作邻近调制"，而是：

1. **先用 falsification gate 钉死前提**：peri-ictal 事件是否仍然能由全局稳定模板库低残差解释；
2. **如前提成立，正式做主分析**：哪个固定模板在 baseline / pre / post 之间被招募的频率发生显著变化（即 PR-4C `rate_by_template` 信号的正式 inferential 化）。

本 PR **不**做以下事情：

- 不发明新主 metric；不画新主图。
- 不重新讨论 `raw_tau / centered_tau / lag_span / pearson_r / dominant_cluster_fraction` 五指标几何变化（PR-4C 已封板 null）。
- 不引入 KONWAC v2 / manifold / persistent homology；这些进入 PR-6/PR-7 的占位（本文 §9 略写）。

---

## 2. 假设、备择与 falsification 合同

### 2.1 主假设 H_recruit

- **H_recruit**：发作邻近时，固定模板库中 dominant template 的事件率 (`events/hour`) 显著高于 baseline，方向：`post > baseline`（强信号），可能伴 `pre > baseline`（弱信号）。
- 模板库本身不变（continuation of PR-2/2.5 stable templates + PR-4D fixed-template projection）。
- 这是对 PR-4C `rate_by_template` post_ictal vs baseline 主 p=0.0009 / 辅 p=0.0067 信号的正式化。

### 2.2 备择假设 H_alt（必须先证伪，否则 H_recruit 解读失效）

- **H_OOD**：peri-ictal 事件并非全局模板库的 in-distribution 样本，而是来自一个被全 cohort 训练 dominate 的"新模板族"。如果 H_OOD 成立，`rate_by_template` 的"主导模板被招募更多"实际上可能是"新模板族被错误投到老 dominant 模板上"。
- **H_assignment_drift**：`assign_events_to_templates` 在 peri-ictal 段的归属置信度系统性下降（更高 reconstruction error / 更低 best-template gap），即使 cluster 不变，归属也未必稳定。
- 这两个备择假设由 PR-5-A（gate）证伪；只有 gate PASS，才能进入 PR-5-B 主分析。

### 2.3 失败合同（fail-fast）

| 阶段 | 必须满足 | 不满足 → 行为 |
|---|---|---|
| Gate 加载 | 复用 PR-4C P0 修复后的 `compute_seizure_proximity_coupling` 输出，**不**重跑 PR-4C 主分析 | 缺 `pr1_cohort_summary.json` 或 `pr4c_seizure_proximity*.json` → 立即停，要求先跑 PR-4C |
| Gate 样本 | per-subject 每 state ≥ `min_state_events_for_gate = 30`（pool across seizures） | 不达标 subject 直接 ineligible，cohort n 缩水必须报告，不能补救 |
| Gate 判定 | best-template Spearman r 与重建残差 distribution 在 `peri_ictal_states` 与 `baseline` 之间不出现系统漂移（subject-level paired Wilcoxon p ≥ 0.05 + cohort 级 |median delta| < 预设阈值） | gate FAIL → **不**进入 PR-5-B；改为输出"H_OOD 不能排除"的归档，并在 Topic 1 主文档明确 PR-5 主分析未启动 |
| 主分析样本 | 主分析 cohort = `26` 个有 PR-4C usable seizure-proximity 输出的 subjects 中，通过 PR-5-A gate 且进入同一 gate-eligible 事件池的 retained subset；统计单元 = subject。每次复跑必须同时报告 `n_subjects_retained` 与 `n_windows_retained`（main / aux 分开） | 任一 config `n_subjects_retained < 20` → 标记为 underpowered，不上 cohort-level 强结论；`n_windows_retained` 只作结果报告，不再拿 PR-4C 全 cohort 的原始 360 / 370 当硬门槛 |
| 主分析复制 | 主 + 辅助两套 `SEIZURE_PROXIMITY_CONFIGS` 读数方向必须一致；任一显著、另一 p > 0.1 且方向相反 → 视为 sensitivity FAIL | sensitivity FAIL → 降级为 descriptive，不写 inferential conclusion |
| 暴露分母 | 必须走 PR-4C 修复后的 gap-aware coverage（`coverage_ranges` 透传 `block_time_ranges`） | 不走 → 拒绝合并结果 |

---

## 3. PR-5-A：Novel-Template Falsification Gate

### 3.1 科学问题

> 对于落入 `baseline / pre_ictal / post_ictal` 的事件，全局稳定模板库 `T_global = {τ_1, ..., τ_K}`（来自 Topic 1 已验收的 `compute_adaptive_cluster_stereotypy`）能否以与 baseline 同质的 best-template 相关与 reconstruction error 解释？

### 3.2 数据合同

- **模板来源**：每 subject 的 `T_global` = `build_cluster_templates(v_ranks, v_bools, labels=adaptive_labels, n_clusters=stable_k)`，与 `PR-4D` 完全一致。**不**对 peri-ictal 子集重新聚类（重新聚类是 H_OOD 的*探索性诊断*，不是 gate 的主读数）。
- **窗口来源**：直接读 `results/interictal_propagation/pr4c_seizure_proximity.json` (主) + `pr4c_seizure_proximity_auxiliary.json` (辅) 中每个 usable window 的 `state_event_indices`；不重做 `_build_seizure_proximity_windows`，避免引入第二份口径。
- **事件向量**：`v_rel = _compute_relative_lag_matrix(lag_raw[:, valid_event_indices], v_bools)`，与 PR-4C 同。L3-eligible 事件门槛 `min_participating_l3 = 5` 与 PR-4C 同。

### 3.3 度量定义（每事件 i × 每模板 k）

1. **best-template Spearman r**：
   `r_i = max_k spearman(v_rel[:, i] on participating, τ_k on same channels)`。
2. **reconstruction error (rank-space, 与聚类目标一致)**：
   `e_i = min_k mean_squared_error_masked(v_ranks[:, i], τ_k)`，复用 `assign_events_to_templates` 的 mask + euclidean 逻辑（`min_shared_channels = 3`）。
3. **assignment gap**：`gap_i = e_i^{2nd} - e_i^{best}`，反映归属置信度。

每 state（baseline / pre / post）pool across seizures，得到三组分布 `{r}`, `{e}`, `{gap}`。

### 3.4 主统计读数

- **subject-level paired Wilcoxon**（每 subject 一对配对样本）：
  - `median(r_post) − median(r_baseline)`
  - `median(e_post) − median(e_baseline)`
  - `median(gap_post) − median(gap_baseline)`
  - 同样做 `pre vs baseline`
- **cohort-level**：subject-level deltas 上做 sign test + Wilcoxon。

### 3.5 PASS / FAIL 判定（写死阈值，避免事后调）

Gate **PASS** 需要同时满足：

| 维度 | 阈值 |
|---|---|
| `r` 漂移 | cohort `|median Δ_r| ≤ 0.05` 且 paired Wilcoxon p ≥ 0.05（任一 state vs baseline） |
| `e` 漂移 | cohort `|median Δ_e / median(e_baseline)| ≤ 0.10` 且 paired Wilcoxon p ≥ 0.05 |
| `gap` 漂移 | cohort `Δ_gap` 不出现"peri-ictal 显著低于 baseline"的方向（Wilcoxon p ≥ 0.05 或方向反向） |
| Sensitivity | 上面三条在主 + 辅助两套配置下结论一致 |

任何一条不满足 → gate **FAIL**。

### 3.6 探索性诊断（不影响 gate 主判定）

- `peri_pool_recluster`：把 baseline + post 池化后跑一次 `compute_adaptive_cluster_stereotypy`，与 `T_global` 做 Hungarian + Spearman 匹配；若出现 best-match `|r| < 0.5` 的新簇，记为 *exploratory* OOD candidate。**这部分只进 archive，不进主结论。**
- 不做 cohort-level inferential；样本不够。

### 3.7 输出

- `results/interictal_propagation/pr5a_novel_template_gate.json`
  - per-subject: `subject_id, dataset, n_events_by_state, r/e/gap_state_medians, deltas, n_l3_eligible`
  - cohort: `cohort_n, paired_wilcoxon_p, median_deltas, gate_pass_main, gate_pass_auxiliary, overall_pass`
- `results/interictal_propagation/per_subject/pr5a/<subject>.json`（探索性 reclustering 诊断）

---

## 4. PR-5-B：Template Recruitment Shift（gate PASS 后启动）

### 4.1 科学问题

> 在固定模板库下，dominant template 的 absolute event rate (`events/hour`, gap-aware exposure) 在 baseline / pre_ictal / post_ictal 三段之间是否存在系统性差异？

PR-4C `rate_by_template` 已经看到主 + 辅两配置都给出 `post > baseline` 显著（主 p=0.0009 / 辅 p=0.0067）。PR-5-B 的任务是：在通过 PR-5-A gate 的前提下，检验这条信号**能否**升级为 Topic 1 × Topic 2 桥接的正式结论，而不再只是 PR-4C 的次级描述层。

**主结论范围严格收敛在 absolute recruitment 一条线上**：dominant template 的 events/hour 是否在 baseline → peri_ictal 间系统抬升。其他派生口径（占比、非主导模板事件率、模板组成漂移）一律不进入主结论判定，只能以 §4.5 secondary composition diagnostic 的固定形式出现。

### 4.2 数据合同

- **复用 PR-4C P0 修复后的 window ownership / pair usability / gap-aware coverage 合同，但 PR-5-B 不直接吃 PR-4C 原始 `rate_by_template` 数值。** 原因很简单：PR-5-A gate 只验证 `min_participating_l3 >= 5` 的 gate-eligible 事件；为了保持“先 gate、再主分析”的闭环，PR-5-B 必须在**同一事件池**上重算 per-window `counts_by_template` 与 `rate_by_template_per_hour`。否则 gate 审的是一批事件，主分析吃的是另一批事件，合同直接失效。
- **事件池**：PR-5-B 仅纳入与 PR-5-A 相同的 gate-eligible 事件，即 `min_participating_l3 = 5`。实现上：复用 PR-4C 的 `state_event_indices`、`pair_usability`、`state_covered_hours` 与 `coverage_ranges` 逻辑，但在每个 window / state 内先筛掉 `n_participating < 5` 的事件，再在剩余事件上重算 `counts_by_template` / `rate_by_template_per_hour`。
- **不重写 window 分配；不重定义模板。**
- 暴露分母**必须**走 `coverage_ranges` (gap-aware)；脚本入口已经在 PR-4C P0 修复中接好。
- 统计单元 = subject。
- 主导模板的定义保留**两条并行路径**，结论以两条路径的结合判定（见 §4.4 sensitivity gate）：
  - **`dominant_global`**（与 PR-4D 一致）：在该 subject 的 `T_global` 中，**全程 occupancy 最高**的那个簇。捕获 cross-window 稳定的"该 subject 的主旋律"；与 PR-4D 的描述层口径直接对齐，跨 subject 单元一致。
  - **`dominant_per_window`**：在每个 usable window 的 baseline state 内，事件计数最高的那个簇。捕获"window 内重新洗牌"的可能（例如某次发作前后 dominant 临时换手）；与 `dominant_global` 不必相等。
- 每个 subject 同时输出两套读数（见 §4.3）；两条路径不允许相互替代，也不允许中途裁剪。

### 4.3 主读数

| 读数族 | dominant 选择 | 说明 |
|---|---|---|
| `dom_global_rate_{pair}` | `dominant_global` | per-subject 主导模板事件率差（**主结论候选 A**） |
| `dom_window_rate_{pair}` | `dominant_per_window` | per-window 各自选 dominant 后再 across-window 加权汇总（**主结论候选 B**） |
| `nondom_global_rate_{pair}` | `dominant_global` 取补集 | 次级：非主导模板合计事件率差（不进任何 Bonferroni 池，仅作为 §4.5 composition diagnostic 的补充） |
| `dom_global_share_{pair}` | `dominant_global` | **Secondary composition diagnostic**：dominant 占总事件率的比例差。详细统计合同见 §4.5 |
| `dom_agreement` | — | 描述：`dominant_per_window` 与 `dominant_global` 在该 subject 的 usable windows 上的一致比例 |

`{pair} ∈ {pre_minus_baseline, post_minus_baseline, post_minus_pre}`。每 subject 的 per-state rate 通过对其 usable windows 加权平均（权重 = `state_covered_hours`），避免短 window 与长 window 等权。

### 4.4 主统计

- **subject-level paired Wilcoxon**（按 subject 配对），两条候选**分别**跑：
  - 候选 A（`dominant_global`）：3 个 pair × 2 套窗口配置
  - 候选 B（`dominant_per_window`）：同上
- **次级**：`nondom_global_rate_*` 与 `dom_global_share_*` 走 §4.5 secondary composition diagnostic 的独立合同，**不进**主结论 Bonferroni 池，**不**参与 §4.4 sensitivity gate 的 strong/medium/descriptive 判定。
- **多重校正**：每条候选 3 pair × 2 config = 6 主比较 → 候选内 Bonferroni alpha = 0.0083；两条候选之间**不**再二次 Bonferroni，因为它们是同一现象的不同算子选择，不是独立假设家族。
- **sensitivity gate**（合并判定）：
  - 强结论（"recruitment shift 在 cohort level 成立"）需要：候选 A 的 `post_vs_baseline` 在主 + 辅两配置下方向一致，且至少一个配置 Bonferroni 通过、另一个 nominal 0.05 通过；**且**候选 B 的 `post_vs_baseline` 在主配置下至少 nominal 0.05 通过且方向一致。
  - 中等结论（"recruitment shift 在 dominant_global 口径下成立，per-window 口径未独立支持"）：候选 A 满足上面，候选 B 不满足 → 报告时显式写"per-window dominant 重选后效应未达 nominal 0.05；可能因为 dominant 在 window 间洗牌从而稀释了固定单簇的 rate signal"，**不**升级为强结论。
  - 候选 A 都过不了 → 整体降级为 descriptive，不写 inferential 主结论。
- **bootstrap CI**：subject-level cluster bootstrap (n_boot = 2000) 给出 `median Δ` 95% CI；两条候选都报；不替代 Wilcoxon，只补强 effect size 报告。

### 4.5 Secondary composition diagnostic（写死合同）

**动机来源**：`scripts/plot_topic1_pr4_ppt.py` 生成的 PR-4 PPT panel d（`fig2_pr4a_dominant_template.png` 第 4 panel）显示，cohort 26 个 subject 的 per-bin Spearman ρ(dominant template share, |Δt to nearest seizure|) 中位数为 +0.077（16/26 > 0），cohort Wilcoxon p = 0.041。方向是：seizure 越近，dominant template **占比** 越低。该信号与 PR-4C `rate_by_template` 主信号正交：peri-ictal 阶段所有模板 absolute rate 都被抬，但 non-dominant 模板被抬得更多，使 dominant 占比反而下降。

**升格判定**：该信号在 PPT 层只能算"补充现象"，正式化只走 §4.5 这一条。**不**升级为 PR-5-B 主结论，**不**进入主分析 Bonferroni 池，**不**触发 strong/medium/descriptive sensitivity gate。

**为什么用离散三段窗口而不是 panel d 的连续 |Δt| 框架**：与 PR-5-B 主分析、PR-5-A gate 共用同一事件池、同一 window/state、同一 gap-aware exposure，避免在 PR-5 内部并存第二份连续时间合同。panel d 的连续 ρ 与本节读数回答的是同一现象的两种切片，本节统计单元为 subject、配对 Wilcoxon，比 panel d 的 cohort 26-个-ρ-Wilcoxon 更干净。

#### 4.5.1 度量定义

对每 subject、每 usable window、每 state ∈ {baseline, pre_ictal, post_ictal}，在与 §4.2 相同的 gate-eligible 事件池上：

- `share_state = sum(counts_by_template[dominant_global_id]) / sum(counts_by_template)`，跨该 subject 的 usable windows 按 `state_covered_hours` 加权平均（与主分析 `dom_global_rate_state` 加权口径完全一致，禁止换权重）。
- 若某 window 在该 state 内总事件数为 0，则该 window 在该 state 上 ineligible，不进入加权平均；该 subject 任一 state 上 eligible window 数 < 1 → 该 subject 在该 pair 上整体 ineligible，记入 `n_subjects_ineligible_share`。
- **只用 `dominant_global`**，不做 `dominant_per_window` 版本。理由：composition diagnostic 的科学问题是"全程 dominant 模板占比是否在 peri-ictal 下降"，per-window 重选会把"占比"变成"被重选模板占自身的比例"，破坏问题语义。

#### 4.5.2 主统计

- **subject-level paired Wilcoxon**，三个 pair × 2 套窗口配置 = 6 比较：
  - `share_pre_minus_baseline`
  - `share_post_minus_baseline`
  - `share_post_minus_pre`
- **不做 Bonferroni 校正**：本读数是 secondary diagnostic，单独家族，校正只会在 archive 内额外报告 nominal alpha = 0.05 与 alpha = 0.0083 两套结果以便对照。
- **方向先验**：panel d 给出的方向是 seizure 越近 share 越低 → 期待 `share_pre_minus_baseline ≤ 0` 与 `share_post_minus_baseline ≤ 0`。报告时显式写出方向，不允许只报双侧 p。
- **bootstrap CI**：subject-level cluster bootstrap (n_boot = 2000) 给出 `median Δ_share` 95% CI；与主分析 bootstrap 同口径。

#### 4.5.3 报告与失败合同

- **报告位置**：`pr5b_recruitment_shift.json` 内单独字段 `composition_diagnostic`，与主分析 `dominant_rate` 字段同级但物理隔离；archive 复跑节也单独成段，不与主结论混写。
- **不与主结论联动**：composition diagnostic 显著或不显著，**都不**改变 §4.4 sensitivity gate 输出。
- **解读边界（写死）**：
  - 若主分析 strong + composition diagnostic 在两配置下方向一致显著（`share_post_minus_baseline < 0`，nominal p < 0.05）→ archive 写"absolute recruitment ↑ 与 dominant share ↓ 同时成立，与 panel d 方向一致；机制层面非 PR-5 工作量"。
  - 若主分析 strong + composition diagnostic null → archive 写"absolute recruitment ↑ 成立，dominant share 未观察到 cohort-level 漂移；panel d 信号未被本口径复制"。
  - 若主分析 descriptive + composition diagnostic 显著 → 不允许据此回头救主结论；archive 显式标注"主分析未通过 sensitivity gate；composition diagnostic 单独显著仅作描述层记录，不作为 Topic 1 × Topic 2 桥接结论"。
- **禁止**在 PR-5-B 内基于 composition diagnostic 推断 "novel template emergence" — 那是 PR-5-A gate 的工作，且 PR-5-A 已 PASS（`overall_pass = True`）；composition shift 只能解释为"in-distribution 模板库内部招募权重变化"。

### 4.6 输出

- `results/interictal_propagation/pr5b_recruitment_shift.json`
  - per-subject: `dom_global_id, dom_window_ids_per_window, dom_agreement, dom_global_rate_state, dom_window_rate_state, nondom_global_rate_state, dom_global_share_state, deltas, n_windows_used, share_state_n_eligible_windows`
  - cohort.dominant_rate: 主 + 辅两配置下每个 pair × 每条候选的 `n, median_delta, wilcoxon_p, bonferroni_pass, ci95`，以及合并 sensitivity gate 的 `overall_strong / overall_medium / overall_descriptive` 三选一标志
  - cohort.composition_diagnostic: 主 + 辅两配置下每个 pair 的 `n, n_ineligible, median_delta_share, wilcoxon_p, direction_consistent_count, ci95`；**不**写 `bonferroni_pass`，**不**写 sensitivity 标志
- 不画新主图。在已有 `PR-4D` per-subject `rate×type` 图旁附一栏 inset 标注 baseline / pre / post 区间，并在 inset 标题区注明该 subject 的 `dom_global_id`、`dom_agreement` 与 `share_baseline → share_post`（仅文本，不再加新 panel）。inset 由 `scripts/plot_interictal_propagation.py` 的 `--pr4d` 路径增量加，不开新图族。

---

## 5. 实现合同与代码入口

### 5.1 复用层（**不重写**）

- `src/interictal_propagation.py`
  - `build_cluster_templates`、`assign_events_to_templates`
  - `compute_seizure_proximity_coupling`（PR-4C P0 修复版）
  - `_build_seizure_proximity_windows`、`_rate_by_template_for_window`、`_intersect_seconds`
  - `_compute_relative_lag_matrix`、`_valid_event_indices`
  - `SEIZURE_PROXIMITY_CONFIGS["main"|"auxiliary"]`
  - `load_subject_propagation_events`（含 `block_time_ranges` → 透传 `coverage_ranges`）

### 5.2 新增层（PR-5 内部）

- `src/interictal_propagation.py` 末尾新增：
  - `compute_novel_template_gate(... ) -> Dict[str, Any]`
    - 输入：`T_global`, `v_ranks`, `v_rel`, `v_bools`, `proximity["windows"]` (来自 PR-4C 输出 reload)
    - 输出：每 state 的 `r / e / gap` distribution + per-subject deltas
  - `compute_template_recruitment_shift(... ) -> Dict[str, Any]`
    - 输入：每 window 在**同一 gate-eligible 事件池 (`min_participating_l3 = 5`)**上重算的 `rate_by_template_per_hour` + `counts_by_template` + `state_covered_hours`，**以及该 subject 的 `dominant_global_id`**（由调用方按 PR-4D 同口径在该 subject 的全程 occupancy 上计算并传入；本函数内部不重算）
    - 内部为每 usable window 同时算 `dominant_per_window`（取该 window baseline state `counts_by_template` argmax；baseline 空则取该 window 总 counts argmax；总计数为 0 则该 window 在候选 B 上 ineligible）
    - 输出：每 subject 的两条候选 deltas、`dom_agreement`、per-window `dom_window_id` 列表，以及 cohort 统计
- `scripts/run_interictal_propagation.py` 新增：
  - `--pr5-gate`：跑 PR-5-A（main + auxiliary 同时跑）
  - `--pr5-recruitment`：跑 PR-5-B（要求 PR-5-A `overall_pass=True`，否则报错退出）
- 不新增独立 plotting 脚本；inset 走 `--pr4d` 增量。

### 5.3 测试合同（TDD 红绿）

`tests/test_interictal_propagation.py` 新增 9 项：

1. `test_pr5_gate_passes_when_states_are_resampled_baseline`
   - 用模拟数据，把 baseline 事件随机重抽样填入 pre / post → 期待 gate PASS。
2. `test_pr5_gate_fails_when_post_events_drawn_from_orthogonal_template`
   - 注入 baseline-orthogonal 模板事件到 post → 期待 gate FAIL（`r` 显著降，`e` 显著升）。
3. `test_pr5_gate_uses_only_l3_eligible_events`
   - n_part=3-4 事件不应进入 `r/e/gap` 计算。
4. `test_pr5_recruitment_uses_same_l3_eligible_event_pool_as_gate`
   - 构造混合事件池：同一 window / state 内同时含 `n_part>=5` 与 `n_part<5` 事件；PR-5-B 重算的 `counts_by_template` / `rate_by_template_per_hour` 必须只由前者贡献，且与 PR-5-A gate 的 eligible mask 完全一致。
5. `test_pr5_recruitment_uses_gap_aware_coverage`
   - 在 baseline 段制造一个 50% gap → dominant rate 应按真实 covered hours 计算，而不是 nominal window width。
6. `test_pr5_recruitment_weights_per_window_by_covered_hours`
   - 同一 subject 多 window，rate aggregation 必须以 `state_covered_hours` 为权重。
7. `test_pr5_recruitment_aborts_if_gate_not_passed`
   - 在脚本层面：`--pr5-recruitment` 在 gate JSON 不存在或 `overall_pass=False` 时抛 `SystemExit`，不写结果。
8. `test_pr5_recruitment_dual_dominant_definitions`
   - 构造两 window 数据：window A baseline 中 cluster 0 计数最高、window B baseline 中 cluster 1 计数最高，但全程 occupancy cluster 0 最高 → `dominant_global_id == 0`、`dom_window_ids == [0, 1]`、`dom_agreement == 0.5`；候选 A 与候选 B 的 `dom_*_rate_*` 数值不应相等（否则两条路退化），并且该 subject 在两条候选下都各产出一组 deltas。
9. `test_pr5_recruitment_composition_diagnostic_isolated_from_main_family`
   - 构造数据：主分析 `dominant_rate` 在 post_vs_baseline 显著（候选 A Bonferroni pass），同时 share 在 post_vs_baseline 方向一致显著。
   - 断言 1：`composition_diagnostic` 字段独立存在，含 `wilcoxon_p / median_delta_share / ci95 / direction_consistent_count`，**不**含 `bonferroni_pass`。
   - 断言 2：sensitivity gate 的 `overall_strong / overall_medium / overall_descriptive` 计算路径只读 `dominant_rate`，对 `composition_diagnostic` 字段做任何替换都不会改变 sensitivity 标志（用 monkey-patch 验证）。
   - 断言 3：share 加权用的 `state_covered_hours` 与主分析 `dom_global_rate_state` 的权重完全一致（同函数同输入 → 同权重）。
   - 断言 4：某 window 在某 state 上事件数 = 0 时，该 window 不进 share 加权平均；该 subject 任一 state eligible window 数 = 0 时，subject 在该 pair 上 ineligible 并计入 `n_ineligible`。

测试运行环境：

- Python 3.10+ + 仓库现有 venv（`requirements.txt` 不新增依赖；`scipy.stats.wilcoxon` / `scipy.optimize.linear_sum_assignment` 已在）。
- `pytest tests/test_interictal_propagation.py -v`
- 期望：原 36 项 + 新 9 项 = 45 项，全部 pass；原有 PR-4C 测试不许 break。

### 5.4 数据测试环境

- 主复跑命令：
  - `python scripts/run_interictal_propagation.py --pr5-gate`
  - 在 `--pr5-gate` 输出 `overall_pass=True` 后：
    - `python scripts/run_interictal_propagation.py --pr5-recruitment`
- 数据集口径分三层写死：
  - 总 cohort = 30 subjects（Topic 1 propagation 全 cohort）
  - 有 PR-4C usable seizure-proximity 输出的 cohort = 26 subjects（其余 subject 无法进入 seizure-proximity 正式分析）
  - 预计 PR-5 gate PASS 后的可分析 cohort ≈ 20 subjects（因每 state ≥ 30 events + `min_participating_l3 = 5` 双门槛进一步缩水）
- 平台：Linux 5.15 + 工作站 GPU 不强制（PR-5 全部走 CPU；`scipy + numpy` 即可）。
- 单 subject smoke：Epilepsiae `548`，确保 `n_usable_windows ≥ 20` 且 `state_covered_hours` 与 PR-4C 报告一致。

### 5.5 复跑产物落位

- `results/interictal_propagation/pr5a_novel_template_gate.json`
- `results/interictal_propagation/pr5b_recruitment_shift.json`
- `results/interictal_propagation/per_subject/pr5a/<subject>.json`
- `results/interictal_propagation/pr1_cohort_summary.json` 增字段 `novel_template_gate` 与 `template_recruitment_shift`，与 `seizure_proximity_analysis*` 同级，便于后续 plotting 读。

---

## 6. 与 PR-4C / PR-4D 的边界

- PR-4C 五指标几何 cohort null **保持封板**，PR-5 不重开。
- PR-4D `rate×type` 描述层保持原状；PR-5-B 是把其中"哪个模板被招募更多"的次级信号尝试正式化为 Topic 1 × Topic 2 桥接结论，不替代 PR-4D 的描述图。
- PR-5-B 的 rate summary **不直接复用 PR-4C 原始 all-event `rate_by_template` 数值**；它只复用 PR-4C 的 windows / usability / coverage 合同，并在与 PR-5-A 相同的 gate-eligible 事件池上重算。
- PR-5 不涉及 SOZ 解剖锚定（那条线属于 Topic 3 §7，独立 P1 候选，与本 PR 并行）。
- PR-4 PPT 图（`scripts/plot_topic1_pr4_ppt.py` 输出 + `docs/topic1_pr4_ppt_figures.md` 注释）只保留为 §4.5 secondary composition diagnostic 的**历史 motivation / 补充现象**层；panel d 的 cohort 26 个 ρ 与 Wilcoxon p=0.041 不再被引用为正式结论，正式口径以 §4.5 离散三段窗口的 subject-level paired Wilcoxon 为准。任何后续涉及 dominant share 的论述必须引用 §4.5 的 `composition_diagnostic` 字段，禁止跨文件二次解读 panel d 数值。

---

## 7. 风险与诚实评估

1. **gate 设计的功效问题**：`r / e / gap` 漂移阈值（`|Δr| ≤ 0.05`、`|Δe| / e_baseline ≤ 0.10`）是基于已观测的 PR-4B Step 0 中 dominant cluster Pearson r cohort median = 0.601 的合理范围拟定的；首次跑完后允许在 archive 内复盘是否过松，但**不在主分析回头改阈值**。
2. **gate FAIL 的处理**：FAIL 不等于阴性结论，等于"H_OOD 不能排除"。此时 PR-5 主分析不启动，Topic 1 主文档应明确写"PR-5 在当前数据上未通过 gate，PR-4C `rate_by_template` 维持描述层口径"。这避免了在仍有 OOD 风险的前提下硬讲 recruitment shift。
3. **subject pool 缩水**：总 cohort 是 30 subjects，但只有 26 subjects 有 PR-4C usable seizure-proximity 输出；PR-5-A 再叠加 `per-state >= 30 events` 与 `min_participating_l3 = 5` 两道门槛后，预计可分析 cohort 可能掉到 ~20。允许，必须如实报告，不允许补救（如降低 `min_state_events_for_gate` 或把 `n_part < 5` 事件塞回 PR-5-B）。
4. **dominant template 选择**：保留 `dominant_global`（与 PR-4D 一致，全程 occupancy 最高）与 `dominant_per_window`（每 window baseline 内重选）两条并行路径。代价：(a) 两条路径在 forward/reverse 高对称 subject 上可能给出明显不同的 dominant id（`dom_agreement` 接近 0.5），表面上是两条结论；这种情况下按 §4.4 sensitivity gate 走"中等结论"路径，**不**回头挑一条更好看的当主。 (b) 候选 B 的功效天然低于候选 A（每 window 重选 dominant 在 across-window 累加时会稀释固定单簇的 rate signal）；候选 B 不显著时不能回头改其定义（例如改成 "per-seizure" 或 "per-day" 重选）。
5. **方向性陷阱**：`post > baseline` 在 PR-4C 已经稳定；但 `pre > baseline` 在 PR-4C 主配置只到 p=0.042，不能写成强结论，必须按 Bonferroni 后的判定走。
6. **composition diagnostic 的过度解读风险**：§4.5 secondary composition diagnostic 与主分析在同一事件池上跑，结果在数值上**必然**与主分析存在数学耦合（dominant share = dominant_count / total_count，而 dominant_rate 与 total_rate 都是被相同 gate-eligible mask 过滤的）。这意味着 share ↓ 的方向同时受 dominant 与 non-dominant 两端事件率变化驱动，单看 share Wilcoxon 显著并不能区分"dominant 没怎么变 + non-dominant 显著抬"与"dominant 抬一点 + non-dominant 抬很多"两种机制。因此 §4.5 输出**必须**与主分析 `dominant_rate` 与次级 `nondom_global_rate_*` 同时报告，禁止单独引用 share 数值做机制层结论。
7. **panel d 的 selection bias**：panel d 的 cohort n=26 是 PR-4 PPT 作图时 timeline 内有 ≥1 个 seizure 的 subject 子集；§4.5 的 cohort 由 PR-4C usable seizure-proximity 输出（26 subjects）叠加 PR-5-A gate retained subset（main n=23 / aux n=22）决定。两套子集的 cohort 不是同一组人，跑出来的方向若一致是加分，方向若反向也不能反过来质疑 panel d；只能记录"两套 cohort 不同，方向反向是已知边界"。

---

## 8. 工作量与时序

| 阶段 | 工作 | 估时 |
|---|---|---|
| Day 0–1 | PR-5-A `compute_novel_template_gate` 实现 + 单元测试 1–3 红→绿 | 1.5 天 |
| Day 2 | runner `--pr5-gate` 接入 + 全量复跑 + cohort 汇总 | 1 天 |
| Day 3 | gate 结果审阅；PASS / FAIL 决策；archive 中间报告 | 0.5 天 |
| Day 4–5 | （仅 gate PASS）PR-5-B `compute_template_recruitment_shift`（双口径 dominant_global / dominant_per_window + share composition diagnostic + 与 gate 同事件池） + 测试 4–9 | 2 天 |
| Day 6 | runner `--pr5-recruitment` + 主 + 辅两配置全量跑 + bootstrap（两条候选 + composition diagnostic 并行） | 1 天 |
| Day 7 | inset 图增量（标注 dom_global_id + dom_agreement + share_baseline → share_post 文本）；Topic 1 主文档 §7.8 改写为正式结论；archive 更新 | 1 天 |
| Day 8 | sensitivity check（两候选合并判定 strong/medium/descriptive） / Bonferroni / composition diagnostic 独立报告与解读边界自审 / 文档自审 / 36+9 测试套绿灯 | 1 天 |

合计 7–8 工作日，约 1.5 周。这个估算建立在"复用 PR-4C 输出"的前提上；若需要回头修 PR-4C，工时另算。

---

## 9. PR-6（KONWAC v2 进阶建模）—— 略写占位

> 性质：占位 / 待定。本文件只锁 scope 与启动条件，不展开实现。

### 9.1 科学定位

- PR-6 不是再做一次老论文的 KONWAC 演示。它必须把 **PR-4 已钉死的事实** 与 **PR-5 若通过 gate / sensitivity 后才能成立的结果** 当成约束反过来约束模型：
  - 多模板（30/30 stable_k 分布：27×k=2、2×k=4、1×k=6）
  - forward/reverse 对称（12/30 候选对，8/9 跨时间复现）
  - within-cluster identity-bias = 86%
  - geometry stable（PR-4C 阴性，已封板）+ recruitment signal 是否能升级为 formal conclusion（取决于 PR-5-A / PR-5-B）
- 文献定位：2024 KONWAC（老）、2024 pEI-Kuramoto（Kuroki & Mizuseki）、2024 plastic neural mass model（Köksal-Ersöz et al., PLoS Comput Biol）、2024 Brain wave dynamics in Hopfield-Kuramoto。再做"框架演示型"工作没有空间，必须给出可证伪预测。

### 9.2 PR-6 启动条件（硬前置）

- PR-5-A gate PASS；
- PR-5-B 在主 + 辅两配置下 `post_vs_baseline` 通过 Bonferroni；
- 否则 PR-6 待定，Topic 1 内不再排期。

### 9.3 略写：候选可证伪问题

1. 多模板 Hebbian 编码在 30 subject 实测 `stable_k` 分布下的容量 / 干扰；`zhangjinhan`（n_ch=5, k=6）作为 stress test。
2. 慢 `α(t)` 调制能否同时复现"几何稳定 + 候选 recruitment 偏移"的 PR-4C/PR-5 联合约束；若 PR-5 只到 `descriptive`，这里必须同步降级。
3. forward/reverse 对在多模板叠加下是否天然出现。
4. `86%` identity-bias 用作 noise / coupling 比例的硬约束，而不是后验解释。

### 9.4 PR-6 工作量与依赖

- 估时 6–10 周（建模 + 个体化推断），单独立 PR。
- 工程入口：暂未建。计划放 `src/interictal_propagation_model/`（与分析层隔离，避免污染 PR-1..PR-5 codepath）。

### 9.5 几何 / manifold / persistent homology 方向

- 不进 PR-5/PR-6 主线。在 PR-6 启动后，如果模型与数据都已稳定，可作为 PR-7 的可选 discussion 层引入。
- 当前不立项；本 archive §9 只记录文献口径，不展开。

---

## 10. 与 Topic 1 主文档的关系

- 本 archive 是 PR-5 的完整科学合同与实现合同；Topic 1 主文档 `docs/topic1_within_event_dynamics.md` 在 §7.8 / §7.9 / §10 仅保留指针与最小摘要。
- 数字 / 阈值 / 文件路径 / 测试列表的最终来源以本 archive 为准。
- 一旦 PR-5 跑完，本 archive 末尾应增加 `## 11. 复跑结论` 一节，保留计划原文不动，新结论以追加形式写入。

---

## 11. 复跑结论（2026-04-20）

> 性质：实跑结果追加，**不**修改前面 §1–§10 的合同。命令链：
> `python scripts/run_interictal_propagation.py --pr5-gate` → PR-5-A overall_pass=True →
> `python scripts/run_interictal_propagation.py --pr5-recruitment`
> 测试套：`pytest tests/test_interictal_propagation.py` → 49/49 PASS（41 旧 + 8 项 PR-5-B / 修复回归测试 = 49；spec §5.3 写的 9 项含 PR-5-A 已存在 3 项、PR-5-B 原 6 项，本轮另补 2 项回归测试以锁死 retained-subset 与无先验方向合同）。

### 11.1 PR-5-A gate（前置，未变）

`results/interictal_propagation/pr5a_novel_template_gate.json`：
- main：`gate_pass = True`，n_subjects_eligible = 23
- auxiliary：`gate_pass = True`，n_subjects_eligible = 22
- `overall_pass = True` → PR-5-B 启动条件满足

### 11.2 PR-5-B 主分析（candidate A = `dominant_global`，§4.4 sensitivity gate）

`results/interictal_propagation/pr5b_recruitment_shift.json`：**cohort summary 严格限定在 PR-5-A retained subset**（main n_subjects = 23 / auxiliary n_subjects = 22；per-subject JSON 仍为 26 个有 PR-4C usable seizure-proximity 输出的 subject，但仅 retained subset 进入 cohort inferential 层）。`dom_agreement < 0.5` 的高对称 subject 数：main 7 / aux 5，已透明记录。

| config | 候选 | pair | n | median Δ (events/h) | 95% CI | Wilcoxon p | Bonferroni (α=0.0083) |
|---|---|---|---|---|---|---|---|
| main | A (`dominant_global`) | post − baseline | 23 | **+65.46** | [+12.10, +143.90] | **0.00128** | ✅ |
| main | A | pre − baseline | 23 | +10.66 | [−17.00, +40.11] | 0.160 | ❌ |
| main | A | post − pre | 23 | +5.87 | [−6.46, +54.67] | 0.142 | ❌ |
| main | B (`dominant_per_window`) | post − baseline | 23 | **+65.46** | [+6.26, +139.29] | **0.00214** | ✅ |
| aux | A | post − baseline | 22 | **+42.43** | [+1.41, +93.71] | **0.0115** (nominal) | ❌（>0.0083 边界） |
| aux | A | pre − baseline | 22 | +16.03 | [+4.38, +66.33] | 0.0425 (nominal) | ❌ |
| aux | A | post − pre | 22 | −9.32 | [−36.80, +35.55] | 0.702 | ❌ |
| aux | B | post − baseline | 22 | +42.95 | [−2.52, +75.54] | 0.0229 (nominal) | ❌ |

§4.4 sensitivity gate 判定：
- 候选 A `post_minus_baseline` 在 main 与 aux 方向一致（两边 sign = +1）；main Bonferroni pass，aux nominal pass → `candidate_a_strong = True`。
- 候选 B `post_minus_baseline` 在 main 配置 nominal pass 且方向一致 → `candidate_b_supports = True`。
- → `overall_strong = True`，`overall_medium = False`，`overall_descriptive = False`。

**结论**：PR-4C `rate_by_template` post-ictal 信号在 PR-5-A gate-eligible 事件池上正式化为 cohort-level inferential 结论：dominant_global template 的 absolute event rate 在 post_ictal 相对 baseline 系统抬升，主配置 Bonferroni 通过，辅配置同方向 nominal 通过；per-window 重选 dominant 的候选 B 在主配置同方向 nominal 通过 → **strong** sensitivity。

`pre_minus_baseline` 与 `post_minus_pre` 在主配置都未通过 Bonferroni，按 §7 风险 #5 维持次级描述层，不写强结论。

### 11.3 §4.5 secondary composition diagnostic

| config | pair | n | n_ineligible | median Δ share | 95% CI | Wilcoxon p | direction-consistent (a priori share ↓, where defined) |
|---|---|---|---|---|---|---|---|
| main | post − baseline | 23 | 0 | **+0.0156** | [+0.002, +0.040] | 0.0149 | 6 / 23 |
| main | pre − baseline | 23 | 0 | −0.0023 | [−0.042, +0.023] | 0.601 | 13 / 23 |
| main | post − pre | 23 | 0 | **+0.0447** | [+0.004, +0.059] | 0.0135 | — |
| aux | post − baseline | 22 | 0 | **+0.0328** | [+0.011, +0.049] | 0.0301 | 5 / 22 |
| aux | pre − baseline | 22 | 0 | +0.0115 | [−0.010, +0.046] | 0.388 | 11 / 22 |
| aux | post − pre | 22 | 0 | +0.0099 | [−0.010, +0.027] | 0.337 | — |

按 §4.5.3 解读边界（写死的判定路径）：
- 主分析 = strong；composition diagnostic 的 `share_post_minus_baseline` 在两配置下都呈 **panel d 反方向** 的 nominal-positive：main `+0.0156`（p = 0.0149），aux `+0.0328`（p = 0.0301）；但 a priori 方向要求的是 `< 0`，direction_consistent_count 仅 `6/23` 与 `5/22`，因此它**不能**被解释为对 panel d 的复制，反而表明在 PR-5 合同下拿到的是反方向结果。
- 因此走 §4.5.3 第二条判定：
  > "若主分析 strong + composition diagnostic null → archive 写'absolute recruitment ↑ 成立，dominant share 未观察到 cohort-level 漂移；panel d 信号未被本口径复制'"
- 按 §7 风险 #7（panel d selection bias），这是已写死的"两套 cohort 不同，方向反向是已知边界"；因此**不**回头质疑 panel d 的 cohort-26 ρ 数值，也**不**据此推翻 strong 主结论。
- `post_minus_pre` 在 §4.5 中**没有** a priori 方向先验，因此 `direction_consistent_count = null / —` 是合同要求，不再伪造一个"应当下降"的方向计数。main 配置 `post_minus_pre`（p = 0.0135，median +0.0447）仍只记为描述层，不联动主结论。
- 机制解读边界（§7 风险 #6）：share 与 absolute rate 数学耦合 → share + 与 dom_global rate ↑ 同时成立的可能机制是"dominant 抬得比 non-dominant 多"。本 PR 不展开机制层，留给 PR-6。

### 11.4 测试结果

```
$ pytest tests/test_interictal_propagation.py
49 passed in 6.52s
```

新增 8 项 PR-5-B / 修复回归测试全部 PASS：
1. `test_pr5_recruitment_uses_same_l3_eligible_event_pool_as_gate` — gate-eligible 事件池一致性
2. `test_pr5_recruitment_uses_gap_aware_coverage` — gap-aware exposure
3. `test_pr5_recruitment_weights_per_window_by_covered_hours` — 跨 window 权重
4. `test_pr5_recruitment_aborts_if_gate_not_passed` — 缺 gate 或 gate FAIL → SystemExit(2) 且不写 artifact
5. `test_pr5_recruitment_filters_to_gate_retained_subset_per_config` — PR-5-B cohort 必须严格继承 PR-5-A retained subset，且 main / auxiliary 分配置过滤
6. `test_pr5_recruitment_dual_dominant_definitions` — 双口径 dom 数值差异 + dom_agreement
7. `test_pr5_recruitment_composition_diagnostic_isolated_from_main_family` — composition diagnostic schema 独立、无 `bonferroni_pass` 字段、monkey-patch 不影响 sensitivity gate
8. `test_pr5_recruitment_share_post_minus_pre_has_no_a_priori_direction` — `post_minus_pre` 无 panel-d 方向先验，`direction_consistent_count` 必须为 `null`

### 11.5 落位文件

- `results/interictal_propagation/pr5b_recruitment_shift.json`（per-subject + cohort）
- `results/interictal_propagation/per_subject/pr5b/<dataset>_<subject>.json` × 26
- `results/interictal_propagation/pr1_cohort_summary.json` 增字段 `template_recruitment_shift`（与 `novel_template_gate` 同级）

### 11.6 不做的事（自审）

- 不重画 PR-4D 主图；§4.6 inset 文本只标注 `dom_global_id`、`dom_agreement`、`share_baseline → share_post`，**未**作为本次复跑要求实现，inset 增量留给 plotting 单 PR。
- 不在本次复跑里基于 strong 主结论改 `min_state_events_for_gate` 或 `min_participating_l3` 阈值（§7 风险 #1）。
- 不基于 §4.5 的 `post_minus_pre` 单一显著 pair 推任何机制层结论（§4.5.3 写死禁止）。
- 不重新定义 `dominant_per_window`（候选 B 在主配置 nominal pass 已支持 strong；不需要换 per-seizure / per-day 形式回头救），符合 §7 风险 #4.b。

### 11.7 验收脚注（2026-04-20）

| 子 PR | 验收判定 | 锚点 |
|---|---|---|
| PR-5-A | **PASS / ACCEPTED** | §11.1；`overall_pass=True`；retained main n=23 / aux n=22；六条 gate 对比全部 within §3.5 写死阈值；49/49 测试 PASS |
| PR-5-B | **STRONG / ACCEPTED** | §11.2 + §4.4 sensitivity gate；`overall_strong=True`；候选 A `post_minus_baseline` main Bonferroni-pass + aux 同向 nominal-pass + 候选 B 主配置同向 nominal-pass；cohort filtering 按 retained subset per config 严格执行（§5.3 regression test #5） |
| §4.5 composition diagnostic | **不复制 panel d** | §11.3；`share_post_minus_baseline` 两配置都是 panel d 反方向的 nominal-positive；按 §4.5.3 第二条 + §7 风险 #7 写死路径，不为主结论背书；`post_minus_pre` 无 a priori 方向（§5.3 regression test #8） |
| PR-6 | **DEFERRED**（硬前置未达成） | §9.2；aux `post_vs_baseline` p=0.0115 > 0.0083 |

**单句承上结论**（与 Topic 1 主文档 §2 / §7.8 一致）：在 PR-5-A 已剔除 novel-template / assignment-drift 假象的前提下，dominant template 的**绝对**招募率（events/h）在 post-ictal 相对 baseline 出现 cohort-level 系统升高，把 PR-4C `rate_by_template` 的描述层信号正式升级为推断结论；其**相对占比**变化在 PR-5 合同下不复制 panel d，因此这是"绝对招募增多、模板间相对权重不偏"的画面，而非 "non-dominant template emergence"。
