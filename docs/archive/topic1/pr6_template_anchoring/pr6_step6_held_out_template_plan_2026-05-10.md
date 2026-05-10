# PR-6 Step 6 — Held-out Time Template Stability（plan-of-record）

> 性质：PR-6 后续稳健性，**Robustness / sensitivity tier，不开新 cohort claim**。
> 母 plan：`pr6_template_endpoint_anchoring_plan_2026-04-25.md` §15 Step 6（"Held-out time validation"，列为 5 项下一步验证之 2）。
> 主文档：`docs/topic1_within_event_dynamics.md` §7.10 加一行回链。

## 1. 一句话核心

**用前半时间定义传播模板和 endpoint，用后半时间只做验证，不重新发明 endpoint。**
（user 明确指令 2026-05-10。这里强调严格 train/test 不对称，区别于 PR-2.5 split-half 的对称双重聚类 + Hungarian 匹配。）

## 2. Context — 为什么做

PR-6 主线（Step 2 / 4 / 4b / 5a / 5b / 3）所有结果均在**全数据**上同时定义 endpoint 与检验 endpoint anchoring，存在 PR-2 cluster 与 PR-6 endpoint 共用 events 的潜在 double-dipping。Step 6 把 cohort 上的 H1 NULL（pooled n=21 Wilcoxon p=0.42）与 §8 swap_class dual-tier 都搬到 train/test 框架下，回答两个问题：

(a) **Is endpoint set ε_first overfit?** 如果 ε_first 在第二半数据里漂移到中段位置，则全数据 H1 / §8 / §9 的所有结果都需要重读为 "single-fold pattern, may not generalize"。
(b) **Is template geometry within-recording stationary?** 也就是 H_drift 在论文记录窗口（24h Yuquan / 多日 Epilepsiae）内是否检出。**这不是 H_drift 全盘否定**——长于记录窗口的 drift 不在本 Step 验证范围。

## 3. 假设卡片

| 字段 | 值 |
|---|---|
| **Tier** | Robustness / sensitivity；**不**进 paper α，不升级既有结论 tier |
| **可证伪条件**（任一即不通过） | (a) cohort `template_spearman` 中位数 < 0.5；(b) cohort `endpoint_position_recall` 中位数 < 0.5；(c) cohort `swap_class_concordant` 比例 < 50% |
| **可结** | 在论文记录时间窗口内，PR-6 主线 endpoint 与 §8 swap_class 是否对 PR-2 + PR-6 same-events double-dipping 稳健 |
| **不能结** | 论文记录窗口外（week-month）的 drift；H_drift 全盘否定；endpoint 定义本身（top-3 ∪ bottom-3）的合理性 |

## 4. Cohort

- **主 cohort = PR-6 §15 audit-derived n=21**（13 epilepsiae + 8 yuquan，stable_k=2 ∩ endpoint_defined ∩ matched_SOZ ≥ 1）。直接复用 `results/interictal_propagation/template_anchoring/cohort_summary.json` 的 inclusion list。
- **Sensitivity cohort = supplementary v14 全 n=35**（stable_k=2 ∩ rank_displacement universe，含 PR-2 sentinel fallback 5 例）。仅 descriptive 报告，不进 cohort acceptance。
- **不预筛 swap_class**；分层报告 strict / candidate / none 在两个 cohort 上的子分布。

## 5. Inputs（不重算，全部读已有 artifact）

| 资源 | 路径 | 用途 |
|---|---|---|
| PR-2 per-subject | `results/interictal_propagation/per_subject/<subject>.json` | adaptive_cluster (k, valid_mask, channel_names, template_rank)，event 元数据 |
| lagPat raw bools / ranks | `results/lagPat/<subject>_lagPat_withFreqCent.npz` | 每 event 重算 rank 矩阵；event_abs_times；block_ids；start_t |
| PR-6 endpoint baseline | `results/interictal_propagation/template_anchoring/per_subject/<subject>.json` | endpoint set ε_full（top-3 ∪ bottom-3 per cluster），用于和 ε_first 一致性 sanity check |
| §8 swap baseline | `results/interictal_propagation/template_anchoring/cohort_summary.json` | swap_class_full + decision_k_full + p_fw_full |
| Clinical SOZ | `results/{epilepsiae,yuquan}_soz_core_channels.json` | H1 anchoring delta 报告（fixed channel set，跨半段不变） |

## 6. 方法（严格 train/test，单向）

### 6.1 时间划分

事件按 `start_t` 排序；median 切分 first / second 两段。同时记录两段 day/night 比例（confound 报告字段，不参与筛选）。
- 切分前严格使用 `block_time_ranges` 排除跨 block 边界事件（CLAUDE.md §6 boundary 合同）。

### 6.2 Train（仅 first half）

复用既有 helper，不新写聚类逻辑：

1. KMeans (k=stable_k=2，n_init=10，random_state=0) on first-half rank features → `labels_first`
2. `build_cluster_templates(v_ranks_first, v_bools_first, labels_first, k=2)` → `templates_first[A]`, `templates_first[B]`（rank 向量长度 = n_valid）
3. **Endpoint 抽取（写死 top-3 ∪ bottom-3，与 PR-6 主定义一致）**：对每 cluster k 的 template_first[k]，取 top-3 channels 与 bottom-3 channels 之并 → `endpoint_first[k]`（**channel 名集合，不是 rank 位置**）。`n_valid ≥ 6` 是 endpoint_defined 前提（PR-6 §4 contract）。
4. **§8 swap_class 抽取**：对 (template_first[A], template_first[B]) 对跑 §8 variable-k max-null 分类器（k 扫描 + 1000 perm，seed=0）→ `swap_class_first ∈ {strict, candidate, none}`，`decision_k_first`，`p_fw_first`。

### 6.3 Test（仅 second half；不在 second half 上 recluster；endpoint 集合不重新发明）

1. **Cluster 投射**：`assign_events_to_templates(v_ranks_second, v_bools_second, templates_first, min_shared_channels=3)` → `assigned_second`（每 second-half event 分配到 templates_first 的 nearest cluster；不可分配的 event 报 -1 并计 `n_unassignable_second`）
2. **第二半投射模板**：对每 cluster k，取 `assigned_second == k` 的 events，重算 mean rank → `template_second_projected[k]`（rank 向量长度 = n_valid，与 templates_first[k] 同 channel ordering）
3. **不做**：在 second half 上跑 KMeans / 重新算 endpoint / 重新跑 §8 stable_k 决策。

### 6.4 Held-out validation metrics（每 subject 4 个数）

| 量 | 定义 | 阈值 |
|---|---|---|
| `template_spearman` | `mean_k(ρ_Spearman(template_first[k], template_second_projected[k]))`，跨 valid channels | strong > 0.7（PR-2.5 阈值复用） |
| `endpoint_position_recall` | 对每 cluster k，`#{c ∈ endpoint_first[k] : c 在 template_second_projected[k] 中仍处于 top-3（c 原本在 top）或 bottom-3（c 原本在 bottom）} / |endpoint_first[k]|`，再 mean over k | strong > 0.6；NULL ≈ 6/n_valid（n_valid=10 → 0.6 即 baseline） |
| `assignment_coverage` | `#{e ∈ second-half events : assigned_second[e] ≠ -1} / n_events_second`；**descriptive only，不进 tier 规则**。⚠️ 原 plan 名为 `cluster_assignment_purity` 并要求 nearest vs second-nearest gap ≥ 0.1——实现中**未做** gap 计算（user review 2026-05-10 指出）；改名 `assignment_coverage` 与实际语义一致；后续若需 gap-margin purity 应补 `assignment_purity` 作 stricter sibling，不替换 coverage。 | 报告分布，不设阈值 |
| `swap_class_concordant` | `swap_class_first == swap_class_second_projected`（categorical 三值），同时报告 (strict↔candidate, candidate↔none, strict↔none) 三类不一致频次 | strong = concordant; flip = fail |

**注**：H1 anchoring delta 本身在 ε_first 固定 + clinical SOZ 固定下是同一个数，跨半段不变。Step 6 不"重跑 H1"——held-out 的 burden 全部压在 endpoint 几何稳定性 + swap_class concordance 上。如果这两项都稳，则 PR-6 全数据 H1 NULL 的 robustness 得到支持；如果两项都漂，则全数据 H1 / §8 / §9 一并需要 caveat。

### 6.5 Day/night confound 控制（敏感性，不替换主分析）

可选 `--day-night-balanced` 切分模式：取每段内 day events 与 night events 各 50% 重组成 first / second，重跑 4 个量。仅当主切分中两段 day/night 比例差 > 20% 时报告该 sensitivity。

## 7. 统计与 acceptance

### 7.1 Per-subject 4-tier（仿 PR-2.5 reproducibility tier）

| Tier | 条件 |
|---|---|
| **strong** | template_spearman > 0.7 AND endpoint_position_recall > 0.6 AND swap_class_concordant |
| **moderate** | 上面任一项不达，但其他两项达标 |
| **weak** | 仅 1 项达标 |
| **fail** | 0 项达标 |

### 7.2 Cohort 报告（描述层 + 结构性检验）

- **n=21 主**：4 个量的 cohort 中位数 + IQR，sign test on `template_spearman > 0.5`、`endpoint_position_recall > random_baseline (= 6/n_valid)`，sign test on `swap_class_concordant`。
- **n=35 sensitivity**：同上，descriptive only。
- **PR-6 主线 H1 NULL 的 caveat 升级判定**：如果主 cohort `endpoint_position_recall` median < 0.5 OR `swap_class_concordant` 比例 < 50% → 在 §7.10 加 framing："PR-6 主线 H1 NULL 与 §8 dual-tier 在 within-recording train/test 上不稳健，需进一步功率检验"。

### 7.3 失败合同（halt execution）

- 若 pilot 三个 sentinel subject 上 `endpoint_position_recall` 的 SE > 0.20 → **停止 cohort run**，先做 power analysis 决定是否减小 endpoint set（top-2 ∪ bottom-2）或放弃 endpoint 几何稳健性主量、改报 template_spearman + swap concordance 二元主量。
- 若 second-half 不可分配事件比 (`n_unassignable_second / n_events_second`) > 0.30 → 该 subject 标 `unassignable_dominant`，excluded from cohort acceptance（保留 per-subject report）。
- 若两半 day/night 比例差 > 20% AND 主 cohort 主量 sign test border (p ∈ [0.04, 0.10]) → 必须跑 §6.5 day/night-balanced 敏感性，archive doc 同时报告主切分与 day/night-balanced 切分两套结果。

## 8. Outputs

```
results/interictal_propagation/pr6_step6_held_out_template/
├── per_subject/<subject>.json           # 4 量 + 半段 raw template/endpoint/swap_class + day/night
├── pilot_3subjects.json                 # chengshuai / 818 / E14 三个 sentinel power 检查
├── cohort_summary.json                  # n=21 主 + n=35 sensitivity + 4-tier 计数 + sign test
└── figures/
    ├── README.md                                  # 中文，按 AGENTS.md 规范，图实际生成后写
    ├── tier_distribution_bar.{png,pdf}            # strong/moderate/weak/fail 计数 (n=21 + n=35 双 panel)
    ├── template_spearman_jaccard_box.{png,pdf}    # template_spearman + endpoint_position_recall box，按 swap_class_full 分 hue
    ├── endpoint_position_recall_scatter.{png,pdf} # 每 subject endpoint channel 在 first vs second_projected 的 rank-position 散点 + identity line
    └── swap_class_transitions.{png,pdf}           # 9-cell concordance matrix (first × second_projected swap_class)
```

`figures/README.md` 必须在图实际生成后写，**不预先占位空 README**（CLAUDE.md / AGENTS.md 规范）。

## 9. 代码改动（最小化 + 大量复用）

### 9.1 新增：`compute_held_out_endpoint_validation` in `src/interictal_propagation.py`

**位置**：紧邻 `compute_time_split_reproducibility`（line ~1754）下方，**不修改**既有 helper。

**签名**：
```python
def compute_held_out_endpoint_validation(
    ranks: np.ndarray,
    bools: np.ndarray,
    event_abs_times: np.ndarray,
    block_ids: np.ndarray,
    block_time_ranges: np.ndarray,           # 必填，无默认；CLAUDE.md §6 boundary 合同
    chosen_k: int,                           # 通常 = 2
    valid_event_indices: np.ndarray,
    channel_names: List[str],
    soz_channels: Set[str],
    min_shared_channels: int = 3,
    endpoint_top_n: int = 3,                 # 与 PR-6 主定义一致；不开放为运行时调参
    *,
    balance_day_night: bool = False,
) -> Dict[str, Any]:
    """Train on first-half time, test on second-half time. Asymmetric.

    Returns:
      - first_half: {labels, templates, endpoints, swap_class, decision_k, p_fw, day_night_ratio}
      - second_half: {assigned, template_projected, swap_class_projected, decision_k_projected, p_fw_projected, day_night_ratio, n_unassignable}
      - validation: {template_spearman, endpoint_position_recall, assignment_coverage, swap_class_concordant, tier}
      - confound: {day_night_imbalance}
    """
```

**禁止**：`endpoint_top_n` 不写默认值兜底 path——如果传 None 应 raise；如果 n_valid < 2 * endpoint_top_n 应 raise（与 PR-6 endpoint_defined 合同一致）。

### 9.2 新增 CLI：`scripts/run_pr6_step6.py`

**职责**：
- `--pilot`：跑 chengshuai (yuquan, 高 stereotypy) / 818 (yuquan, k=4 case-series 转 stable_k=2 fallback skip) / E14 (即 epilepsiae 548, paper-level)；写 `pilot_3subjects.json`；终端打印 SE 与 tier。
- `--cohort`：跑全 n=21 主 + n=35 sensitivity；写 `per_subject/*.json` + `cohort_summary.json`。
- `--day-night-balanced`：开启 §6.5 切分模式。

**复用**：`load_subject_propagation_events`（已 sort by start_t、含 block_time_ranges）；§8 swap classifier 调用复用 `src/rank_displacement.py` 既有 helper（不再 reimplement）。

### 9.3 新增 figure script：`scripts/plot_pr6_step6.py`

四张图 + `figures/README.md`，使用 `src/plot_style.py` Morandi 调色板（与 Topic 2 PPT 一致）。

### 9.4 不动

- `compute_time_split_reproducibility`（PR-2.5 contract，不能改签名）
- PR-6 既有 `template_anatomical_anchoring.py`（PR-6 主线代码不动）
- §8 variable-k swap classifier 实现（仅 import）
- PR-2 KMeans 与 stable_k 决策（不重做）

## 10. TDD 任务分解（bite-sized）

每步 commit；CLAUDE.md §6 stub 合同：未实现 path 写 `raise NotImplementedError(...)`，**不返回 plausible 默认值**。

### Task 1 — 时间切分 helper

- **Files**: create `tests/test_held_out_template.py`; modify `src/interictal_propagation.py`
- [ ] **Step 1.1**: 写 failing test `test_split_events_by_time_respects_block_boundaries` —— 5 events 跨 2 blocks，验证切分点不落在 block boundary 上、两段 day/night 比例正确返回。
- [ ] **Step 1.2**: 运行 `pytest tests/test_held_out_template.py::test_split_events_by_time_respects_block_boundaries -v` → expect FAIL (NotImplementedError)
- [ ] **Step 1.3**: 实现 `_split_events_by_time(times, block_time_ranges, balance_day_night=False) -> (idx_first, idx_second, day_night_ratio_dict)`
- [ ] **Step 1.4**: 运行测试 → expect PASS
- [ ] **Step 1.5**: commit "feat(pr6 step6): time-split helper with block-boundary respect"

### Task 2 — Asymmetric train/test pipeline

- **Files**: `tests/test_held_out_template.py`, `src/interictal_propagation.py`
- [ ] **Step 2.1**: 写 toy test `test_compute_held_out_endpoint_validation_recovers_train_template` —— 构造 30 events 完美 stereotyped (k=2)，second half 与 first half 完全相同 → 4 个量都达 strong。
- [ ] **Step 2.2**: 运行 → FAIL
- [ ] **Step 2.3**: 实现 `compute_held_out_endpoint_validation`（§9.1 签名），内部串联：split → KMeans first → build_cluster_templates → endpoint 抽取 → assign_events_to_templates → mean rank per cluster → 4 量计算
- [ ] **Step 2.4**: 测试 → PASS

### Task 3 — Endpoint geometric stability 边界

- [ ] **Step 3.1**: 写 test `test_endpoint_position_recall_drift_to_middle` —— 构造 second half template 与 first half template Spearman = 0（彻底 drift），endpoint_position_recall ≈ baseline (= 6/n_valid)。
- [ ] **Step 3.2**: 运行 → 应已 PASS（实现已覆盖此 path）；如 FAIL，修 §9.1 endpoint position 检查逻辑。

### Task 4 — Stub / contract 合同

- [ ] **Step 4.1**: 写 test `test_held_out_validation_raises_on_missing_block_time_ranges` —— 不传 `block_time_ranges` 应 raise ValueError。
- [ ] **Step 4.2**: 写 test `test_held_out_validation_raises_on_n_valid_below_endpoint_threshold` —— n_valid=5 (< 2 * endpoint_top_n) 应 raise。
- [ ] **Step 4.3**: 实现这两个守卫；运行 → PASS。

### Task 5 — swap_class concordance 集成

- [ ] **Step 5.1**: 写 test `test_swap_class_concordance_uses_variable_k_classifier` —— 第二半 template_projected 与 first half template 几何相似 → swap_class_concordant=True；几何反向 → False。
- [ ] **Step 5.2**: 实现 swap_class 调用（import §8 helper）；运行 → PASS。

### Task 6 — CLI pilot

- [ ] **Step 6.1**: 写 `scripts/run_pr6_step6.py --pilot`，跑 chengshuai / E14 / 一个 §8 strict subject (e.g., epi_1146)。
- [ ] **Step 6.2**: 检查输出 `pilot_3subjects.json` 中 endpoint_position_recall 的 SE。**如果 SE > 0.20 → halt，按 §7.3 决策**。
- [ ] **Step 6.3**: commit "feat(pr6 step6): CLI + pilot run"
- [ ] **Step 6.4**: **hand back to user**：pilot 通过则继续 Task 7；不通过则停下重新决策。

### Task 7 — Cohort run（user 绿灯后）

- [ ] **Step 7.1**: `python scripts/run_pr6_step6.py --cohort` 跑 n=21 主。
- [ ] **Step 7.2**: 跑 n=35 sensitivity。
- [ ] **Step 7.3**: 写 `cohort_summary.json` 4-tier 计数 + sign test。
- [ ] **Step 7.4**: commit "feat(pr6 step6): cohort run + summary"

### Task 8 — Figures + README

- [ ] **Step 8.1**: 写 `scripts/plot_pr6_step6.py` 四张图。
- [ ] **Step 8.2**: 生成图。
- [ ] **Step 8.3**: 写 `figures/README.md` 中文逐图说明（每图 2-4 句 + "**关注点**：" 行）。
- [ ] **Step 8.4**: commit "feat(pr6 step6): figures + README"

### Task 9 — Archive results doc + 主文档回链

- [ ] **Step 9.1**: 写 `docs/archive/topic1/pr6_template_anchoring/pr6_step6_held_out_template_results_2026-05-10.md`（results-only，不重述 plan）。
- [ ] **Step 9.2**: 主文档 §7.10 加 1-2 行回链（user-facing 摘要 + archive doc 路径）。
- [ ] **Step 9.3**: commit "docs(pr6 step6): archive results + main doc backlink"

**预算**：Task 1-5（实现 + TDD）≈ 0.4 d；Task 6 pilot ≈ 0.1 d；Task 7-8 cohort + figures ≈ 0.4 d；Task 9 docs ≈ 0.3 d。总 ≈ 1.2 d。

## 11. 显式不做的事（防漂移）

- **不**重做 PR-2 KMeans / stable_k 决策
- **不**重做 PR-6 endpoint 定义（top-3 ∪ bottom-3 锁死）
- **不**重做 §8 variable-k 决策规则（strict/candidate/none 阈值锁死）
- **不**做 multi-fold cross-validation（5-fold 或 k-fold，留给独立后续）
- **不**做 forward-only / forward+reverse 拆分（PR-2.5 已做）
- **不**在 second half 上 recluster（这就是"不重新发明 endpoint"的字面含义）
- **不**改 H1 / H1b / H2 / H3 阈值或定义
- **不**把 Step 6 结果当作 H1 / H_drift 的全盘判定——只是 robustness 一层

## 12. Critical files to touch

```
src/interictal_propagation.py             # 加 _split_events_by_time + compute_held_out_endpoint_validation
tests/test_held_out_template.py           # 新文件，5 个测试 (Tasks 1-5)
scripts/run_pr6_step6.py                  # 新文件，CLI
scripts/plot_pr6_step6.py                 # 新文件，4 图
docs/archive/topic1/pr6_template_anchoring/pr6_step6_held_out_template_results_2026-05-10.md  # results doc，Task 9
docs/topic1_within_event_dynamics.md      # §7.10 加回链 1-2 行
```

不动：`src/template_anatomical_anchoring.py`、`src/rank_displacement.py`（仅 import）、`compute_time_split_reproducibility`、PR-2 / §8 既有 codepath。

## 13. 与 PR-6-sup1 的边界（防混淆）

- **Step 6** 防 double-dipping：train/test 数据切分；问的是 endpoint 是否 generalize。
- **PR-6-sup1**（`pr6_supplementary_rank_entropy_plan_2026-05-10.md`）防 symmetry-breaking 漏检：rank position entropy 描述；问的是 endpoint position 是否比 middle position 更不稳定（confluence-point preflight）。

两者**数据流分开**：Step 6 切时间，sup1 不切时间；Step 6 用 endpoint 集合，sup1 不依赖 endpoint 集合定义。**不共享 helper、不共享 output 目录、不在同一图里出现**。
