# PR-6-sup1 — First-rank Entropy / Symmetry-breaking Diagnostic（plan-of-record）

> 性质：**Topic 4 / SBA mechanism preflight，descriptive only，不进 paper α，不开 cohort claim**。
> 母 plan：`pr6_template_endpoint_anchoring_plan_2026-04-25.md`；user 明确 2026-05-10 决定"作为 PR-6-sup1，不塞进 PR-6 主线"。
> 主文档：`docs/topic1_within_event_dynamics.md` §7.10 加一行 "Topic 4 preflight" 回链。

## 1. 一句话核心

**iHGE 事件的 rank vector 在 endpoint position（最快 + 最慢通道位置）上是否比 middle position 更不稳定？**
若 Liou-Abbott 风 noise-driven confluence-point 模型成立，则起始位点应 highly variable，但传播 backbone 应 stable —— 这就是 cohort symmetry-breaking 的 sharp single-subject prediction。

## 2. Context — 为什么做、为什么不混进 PR-6 主线

PR-6 主线问 "endpoint 是否 anatomically anchor SOZ"（H1）。该问题的两个 prerequisite 假设之一是：**endpoint position 是 mechanistically 特殊的**，不只是 rank 排序的两个端点。如果 rank vector 整段 entropy 平台、中段与端点同样 unstable / 同样 stable，那 PR-6 主线把 endpoint 单独抽出做 anchoring 检验的科学动机被削弱。

但这是 mechanism 维度的预检，不是 anatomy 维度的检验，**与 PR-6 H1/H2/H3 的 falsification 路径无交叉**。混进 PR-6 主线会：
- 把"H1 SOZ NULL"与"endpoint vs middle entropy"两套互不冲突的发现错误纠缠
- 让 §8 swap_class dual-tier 的 strict/candidate/none 标签被 entropy 量耦合
- 给 reviewer "PR-6 在 cohort NULL 之后又改假设" 的把柄

因此 PR-6-sup1 独立成档，**用 PR-6 §8 swap_class 仅作 stratifier label**（不作筛选门，不作 cohort 拆分依据），**不引用 PR-6 H1 / endpoint 集合本身**。

## 3. 假设卡片

| 字段 | 值 |
|---|---|
| **Tier** | **Topic 4 mechanism preflight，descriptive only**；不进 paper α，不写 cohort PASS/NULL 判定 |
| **可证伪条件**（任一即说"confluence model 不被支持"） | (a) cohort `Δ_subject` 中位数 ≤ 0；(b) 与 N0 random-rank null 不可分（cohort 上 1000-perm p > 0.10） |
| **Sharp prediction** | (a) `H_p_norm` 曲线在 p ∈ {1, n_valid} 高、p ∈ {2, ..., n_valid-1} 低；(b) `Δ_subject = mean(H_endpoint) − mean(H_middle) > 0` 多数 subject；(c) §8 strict swap subject 的 Δ 系统高于 none subject（descriptive，不调 α） |
| **可结** | iHGE 事件 rank vector 在 endpoint vs middle 上的 entropy 结构（descriptive）；与 §8 swap_class 是否分层 |
| **不能结** | Liou-Abbott 是机制（只能说 consistent with）；ictal recruitment 与 iHGE 同构；cohort 上"iHGE 来自 noise-driven confluence" |

## 4. Cohort

- **唯一 cohort = supplementary v14 全 n=35 subject**（stable_k=2 ∩ rank_displacement universe，含 PR-2 sentinel fallback 5 例）。
- **Stratifier = §8 swap_class label**（strict n=10 / candidate n=8 / none n=17），仅 descriptive 分组报告，**不作筛选门**。
- **不**预筛 endpoint_defined / matched_SOZ / SOZ-non-empty——sup1 完全不依赖 SOZ。

## 5. Inputs

| 资源 | 路径 | 用途 |
|---|---|---|
| PR-2 per-subject | `results/interictal_propagation/per_subject/<subject>.json` | adaptive_cluster (k, valid_mask, channel_names)，cluster assignment per event |
| lagPat raw | `results/lagPat/<subject>_lagPat_withFreqCent.npz` | 每 event 的 rank vector 重算（必要时由 lagPatRaw → per-event min-subtracted argsort-of-argsort） |
| §8 cohort_summary | `results/interictal_propagation/template_anchoring/cohort_summary.json` | swap_class label（strict/candidate/none） |

**不**读：clinical SOZ JSON / focus_rel JSON / PR-6 endpoint set / PR-4 / PR-5 / PR-7 输出。

## 6. 方法（操作化重写——rank 1 是类别量，不直接用 var）

### 6.1 Per-event rank matrix

每 subject 每 cluster k：
- `valid_mask` 由 PR-6 优先 / PR-2 sentinel fallback 抽取（与 rank_displacement v14 同 provenance；CLAUDE.md "valid_mask semantics" 合同）
- `n_valid = sum(valid_mask)`
- 对每属于 cluster k 的 event e：
  - 取 valid channels 的 lagPat 时间，per-event min-subtraction 后做 argsort-of-argsort → `r_e ∈ {1, ..., n_valid}^{n_valid}`（整数 rank 向量）
- 拼成 `R_k` 矩阵：shape `(n_events_k, n_valid)`

### 6.2 逐 rank position 香农熵（核心量）

对每 rank position p ∈ {1, ..., n_valid}：
1. `count(c, p) = #{events : channel c 在 R_k 中 rank position p}`
2. `P(c | rank=p) = count(c, p) / n_events_k`
3. **Shannon entropy**：`H_p = -Σ_c P(c | rank=p) · log_2 P(c | rank=p)`（log 0 := 0）
4. **归一化**（跨 n_valid 不同的 cluster 可比）：`H_p_norm = H_p / log_2(n_valid)` ∈ [0, 1]

**全随机 rank baseline**：每 event 内独立 random shuffle → `H_p_norm → 1.0` ∀p。
**完全 deterministic baseline**：每 event 同样 rank → `H_p_norm → 0` ∀p。

### 6.3 Endpoint vs middle entropy 差

| 量 | 定义 |
|---|---|
| endpoint positions | `{1, n_valid}`（**不**取 top-3 ∪ bottom-3，避免与 PR-6 endpoint 集合双源） |
| middle positions | `{2, ..., n_valid - 1}` |
| `Δ_subject_k` | `mean(H_1_norm, H_{n_valid}_norm) − mean(H_p_norm for p ∈ middle)` |
| `Δ_subject` | `mean_k(Δ_subject_k)` over the 2 clusters |

预期：confluence model 成立 → `Δ > 0`；rank 全随机 → `Δ ≈ 0`；只有 endpoint 是 deterministic 而 middle 随机（反向情形）→ `Δ < 0`。

### 6.4 Source vs sink 对称性（descriptive 副量）

H_1_norm（最快通道位置）vs H_{n_valid}_norm（最慢通道位置）的差：
- `asymmetry_subject_k = H_1_norm − H_{n_valid}_norm`
- 若 confluence 是真正对称的（source 与 sink 各自 noise-driven）→ |asymmetry| ≈ 0
- 若有方向偏置（始终某 channel 先发）→ |asymmetry| > 0

**仅按 swap_class 分层报告**，不作主结论。

## 7. Surrogate / Null（CLAUDE.md §6 surrogate construction 是合同，必须写死）

### 7.1 N0 — Per-event independent random rank permutation

- 对每 event e，独立 `np.random.RandomState(seed).permutation(n_valid)` 重赋 r_e
- 重算 `H_p_norm` 与 `Δ_subject`
- 1000 surrogates，**`base_seed = 0`，第 i 个 surrogate 用 `np.random.RandomState(0 + i)`**
- **预期效果**：H_p_norm 期望均 ≈ 1（满熵），`Δ_null ≈ 0`

### 7.2 N1 — Channel-marginal-preserving null（更严格）

- 保 channel `c` 在 `R_k` 中的整体 rank 分布不变（边际频率），但打散 rank 与 event 的对应
- 实现：bipartite shuffle，逐 rank position 内独立打乱 events 的 channel 分配，使每 channel 的 marginal `Σ_p count(c, p) / (n_events × n_valid)` 不变
- 1000 surrogates，**`base_seed = 1`，第 i 个 surrogate 用 `np.random.RandomState(1000 + i)`**（与 N0 seed 不重叠）
- **目的**：控制"某 channel 始终偏向某 rank position"的先验偏置；如 N1 下 Δ 仍显著 > 0，则 endpoint vs middle 的非平凡熵结构存在

### 7.3 Per-subject p-value

- N0 p-value：`p_N0 = (1 + #{Δ_null_N0 ≥ Δ_subject}) / 1001`
- N1 p-value：`p_N1 = (1 + #{Δ_null_N1 ≥ Δ_subject}) / 1001`
- Cohort sign test 在 N1 显著 (p_N1 < 0.05) subject 子集上做（descriptive，不调 α）

## 8. 统计与 stratifier

### 8.1 Per-subject 描述（不设 PASS 阈值）

每 subject 报告：`(Δ_subject, p_N0, p_N1, H_1_norm, H_{n_valid}_norm, mean_H_middle_norm, asymmetry_subject)`。**不**给 strong/moderate/weak label——这是 descriptive，不是 robustness tier。

### 8.2 Cohort stratification by §8 swap_class

| 组 | n | 描述指标 |
|---|---|---|
| strict (§8 swap_class) | 10 | `Δ_subject` median + IQR；fraction with p_N1 < 0.05 |
| candidate | 8 | 同上 |
| none | 17 | 同上 |

**Kruskal-Wallis 三组比较**`Δ_subject` 分布；若 H 统计显著 → post-hoc Mann-Whitney U（pairwise，**不**调 α，descriptive only）。

**预期描述方向**：strict ≥ candidate ≥ none on Δ_subject median。但**不写"swap subject 来自 confluence 机制"** —— swap geometry 与 endpoint entropy 是独立维度，相关 ≠ 因果。

### 8.3 显式不做

- **不**做 `Δ_subject` vs PR-6 H1 anchoring delta 的相关 / 回归
- **不**做 `Δ_subject` vs §8 swap_score 连续值 Pearson / Spearman（避免 reviewer 当 cohort claim 升级）
- **不**算 cohort-aggregated p-value 进 paper α
- **不**用 H_endpoint vs H_middle 给 PR-6 endpoint 定义反向 sanity check（违反"两件事分开"原则）

## 9. Outputs

```
results/interictal_propagation/pr6_sup1_rank_entropy/
├── per_subject/<subject>.json           # H_p_norm 全曲线 (per cluster) + Δ_subject + p_N0 + p_N1 + asymmetry
├── cohort_summary.json                  # 35 subject 汇总 + swap_class 三组分布 + Kruskal-Wallis
└── figures/
    ├── README.md                                       # 中文，按规范，图实际生成后写
    ├── H_p_curves_by_swapclass.{png,pdf}              # 35 subject H_p_norm vs p，按 swap_class 三色叠加
    ├── delta_by_swapclass_box.{png,pdf}               # Δ_subject 按 strict/candidate/none box + 散点 + N1 95% CI 灰带
    └── delta_obs_vs_null_scatter.{png,pdf}            # x = Δ_obs, y = N1 95% CI 上界，identity 红线；点大小 = n_events
```

## 10. 代码改动（最小化 + 不污染 PR-6 主线）

### 10.1 新增 helper：`src/rank_displacement.py`

紧邻既有 `compute_signed_rank_displacement` 等 helper，加 3 个函数：

```python
def compute_rank_position_entropy(
    R_cluster: np.ndarray,            # shape (n_events, n_valid), int rank 1..n_valid
    n_valid: int,
) -> np.ndarray:
    """Return H_p_norm of shape (n_valid,)."""

def compute_endpoint_middle_entropy_delta(
    H_p_norm: np.ndarray,             # shape (n_valid,)
) -> Tuple[float, float]:
    """Return (delta, asymmetry) per cluster."""

def rank_entropy_null_sample(
    R_cluster: np.ndarray,
    n_valid: int,
    null_kind: str,                   # 'N0' | 'N1', no default
    n_perm: int = 1000,
    base_seed: int = 0,
) -> np.ndarray:
    """Return null delta distribution of length n_perm. Raises on null_kind not in {N0, N1}."""
```

**Stub 合同**：`null_kind not in {'N0', 'N1'}` 必须 `raise ValueError`，不返回 plausible 默认值。

### 10.2 新增 CLI：`scripts/run_pr6_sup1_rank_entropy.py`

- `--pilot`：跑 chengshuai (none) / 一个 strict (e.g., epi_1146) / 一个 candidate (e.g., epi_1125) → 写 `pilot_3subjects.json`，终端打印 Δ + p_N1。**hand back to user 后再跑 cohort**。
- `--cohort`：跑全 n=35。

### 10.3 新增 figure script：`scripts/plot_pr6_sup1_rank_entropy.py`

3 张图 + `figures/README.md`。

### 10.4 不动

- PR-6 主线代码（`template_anatomical_anchoring.py`、§8 swap classifier）
- PR-2 / PR-2.5 / PR-4 / PR-5 / PR-7 既有 codepath
- `compute_time_split_reproducibility`
- Step 6 新增的 `compute_held_out_endpoint_validation`（Step 6 与 sup1 不共享 helper）

## 11. TDD 任务分解

### Task 1 — entropy primitive

- **Files**: create `tests/test_rank_entropy.py`; modify `src/rank_displacement.py`
- [ ] **Step 1.1**: 写 failing test `test_rank_position_entropy_uniform_baseline` —— 全随机 rank 矩阵 (n_events=10000, n_valid=10) → `H_p_norm` 接近 1.0 (容差 0.05)
- [ ] **Step 1.2**: 写 failing test `test_rank_position_entropy_deterministic_baseline` —— 全相同 rank 向量 → `H_p_norm` 全 0
- [ ] **Step 1.3**: 实现 `compute_rank_position_entropy`
- [ ] **Step 1.4**: 测试 → PASS
- [ ] **Step 1.5**: commit "feat(pr6 sup1): rank position entropy primitive + tests"

### Task 2 — endpoint vs middle delta

- [ ] **Step 2.1**: 写 test `test_endpoint_middle_delta_confluence_toy` —— 构造 toy：endpoint 位置每 event 随机 channel，middle 位置 deterministic → Δ 显著 > 0
- [ ] **Step 2.2**: 写 test `test_endpoint_middle_delta_zero_when_uniform` —— 全随机 → Δ ≈ 0
- [ ] **Step 2.3**: 实现 `compute_endpoint_middle_entropy_delta`
- [ ] **Step 2.4**: 测试 → PASS

### Task 3 — Surrogate construction

- [ ] **Step 3.1**: 写 test `test_null_N0_returns_zero_mean_delta` —— N0 1000 perm → null Δ mean 接近 0 (容差 0.02)
- [ ] **Step 3.2**: 写 test `test_null_N1_preserves_channel_marginal` —— N1 1 perm 后每 channel 的 Σ_p count(c,p) 不变
- [ ] **Step 3.3**: 写 test `test_null_kind_invalid_raises` —— `null_kind='N2'` 应 raise ValueError
- [ ] **Step 3.4**: 实现 N0 + N1（reproducibility：两次跑 base_seed=0 给完全相同的 null 分布）
- [ ] **Step 3.5**: 测试 → PASS

### Task 4 — Per-subject pipeline integration

- [ ] **Step 4.1**: 写 integration test `test_per_subject_pipeline_on_synthetic_subject` —— 合成 stable_k=2 subject (n_events=2000, 各 cluster 1000 events)，cluster A endpoint random + middle deterministic（高 Δ），cluster B 全随机（Δ≈0）→ 验证 per_subject JSON schema 与 Δ 数值方向。
- [ ] **Step 4.2**: 实现 `run_subject_rank_entropy(subject_data) -> dict` 在 `src/rank_displacement.py`
- [ ] **Step 4.3**: 测试 → PASS

### Task 5 — Pilot CLI

- [ ] **Step 5.1**: 写 `scripts/run_pr6_sup1_rank_entropy.py --pilot`
- [ ] **Step 5.2**: 跑 chengshuai / epi_1146 / epi_1125 → 写 `pilot_3subjects.json`
- [ ] **Step 5.3**: commit "feat(pr6 sup1): pilot CLI"
- [ ] **Step 5.4**: **hand back to user**：审阅 pilot Δ 与 p_N1 的方向 / 量级；user 给绿灯后才进 Task 6

### Task 6 — Cohort run

- [ ] **Step 6.1**: `--cohort` 跑全 n=35
- [ ] **Step 6.2**: 写 `cohort_summary.json` (Kruskal-Wallis + 分组 median + IQR)
- [ ] **Step 6.3**: commit "feat(pr6 sup1): cohort run"

### Task 7 — Figures + README

- [ ] **Step 7.1**: 写 `scripts/plot_pr6_sup1_rank_entropy.py` 3 张图
- [ ] **Step 7.2**: 生成图
- [ ] **Step 7.3**: 写 `figures/README.md` 中文
- [ ] **Step 7.4**: commit "feat(pr6 sup1): figures + README"

### Task 8 — Archive results doc + 主文档回链

- [ ] **Step 8.1**: 写 `docs/archive/topic1/pr6_template_anchoring/pr6_supplementary_rank_entropy_results_2026-05-10.md`
- [ ] **Step 8.2**: 主文档 §7.10 加一行 "Topic 4 mechanism preflight, descriptive only" 回链
- [ ] **Step 8.3**: commit "docs(pr6 sup1): archive results + main doc backlink"

**预算**：Task 1-4 ≈ 0.4 d；Task 5 pilot ≈ 0.1 d；Task 6-7 cohort + figures ≈ 0.3 d；Task 8 docs ≈ 0.2 d。总 ≈ 1.0 d。

## 12. Caveats（必须写进 archive doc）

### 12.1 不能升级的边界

- 这是 Topic 4 / SBA preflight，**不**进 Topic 1 paper main figure
- swap_class 仅作 stratifier，**不**说 "swap subject 来自 confluence 机制"
- Liou-Abbott 是**类比映射**，不是同构（iHGE = sub-second pre-ictal HFO group events，不是 ictal recruitment 流形）

### 12.2 Channel-selection circular caveat（与 PR-6 §9 同构）

lagPat / `*_lagPat_withFreqCent.npz` 通道集已是 high-HI / high-HFO-rate selected，本量描述的是 **selected channel 集内部** 的 rank-symmetry 结构，**不是**全脑信号。任何 paper-level 表述必须紧邻数字带这条 caveat。

### 12.3 n_valid 异质性

cohort n_valid ∈ [6, 10]。归一化 `H_p / log_2(n_valid)` 后跨 subject 可比，但 sample-noise 在小 n_valid（=6）subject 上更大。Figures README 必须用 n_valid 作 size legend。

### 12.4 Cluster 数与 stable_k

**仅在 stable_k=2** 的 35 subject 上做；不重做 PR-2 stable_k 决策；不在 stable_k=4 / 6 case-series subject 上跑（这些 subject 已在 PR-2 走 case-series tier）。

## 13. 显式不做的事（防漂移）

- **不**重做 PR-2 KMeans / valid_mask 决策
- **不**改 PR-6 endpoint 定义
- **不**与 PR-6 H1 / H1b / H2 / H3 / §8 / §9 任一统计量做 correlation / regression
- **不**与 Step 6 share helper / output 目录 / 图（user 明确指令"别把这两个混起来"）
- **不**做 cluster k > 2 / k 扫描
- **不**做时间切分（与 Step 6 数据流分开）
- **不**做 Topic 4 主线 mechanism modeling（这是 preflight，不是 modeling）

## 14. Critical files to touch

```
src/rank_displacement.py                         # 加 compute_rank_position_entropy + delta + null + run_subject_rank_entropy
tests/test_rank_entropy.py                       # 新文件，~6 个测试 (Tasks 1-4)
scripts/run_pr6_sup1_rank_entropy.py             # 新文件，CLI
scripts/plot_pr6_sup1_rank_entropy.py            # 新文件，3 图
docs/archive/topic1/pr6_template_anchoring/pr6_supplementary_rank_entropy_results_2026-05-10.md  # results doc，Task 8
docs/topic1_within_event_dynamics.md             # §7.10 加回链 1 行
```

不动：`src/interictal_propagation.py`（Step 6 拥有那里）、PR-6 主线 endpoint helper、§8 swap classifier、PR-2 / PR-2.5 / PR-4 / PR-5 / PR-7 既有 codepath。

## 15. 与 PR-6 Step 6 的边界（防混淆，再写一次）

| 维度 | PR-6 Step 6 | PR-6-sup1 |
|---|---|---|
| 防什么 | PR-2 + PR-6 same-events double-dipping | endpoint position 的 mechanism preflight |
| 数据流 | 时间切分 (first / second half) | **不**切时间，全 events |
| 依赖 endpoint 集合 | 是（fixed ε_first） | **否**，只用 rank position {1, n_valid} |
| 依赖 SOZ | 是（H1 报告） | **否** |
| Output 目录 | `pr6_step6_held_out_template/` | `pr6_sup1_rank_entropy/` |
| 共享 helper | 无 | 无 |
| 共享图 | 无 | 无 |
| Tier | Robustness | Topic 4 mechanism preflight (descriptive) |

**两份 plan、两份 results doc、两组 commit、两段 §7.10 回链。不交叉。**
