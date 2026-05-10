# PR-6-sup1 — First-rank Entropy / Symmetry-breaking Diagnostic（plan-of-record）

> 性质：**Topic 4 / SBA mechanism preflight，descriptive only，不进 paper α，不开 cohort claim**。
> 母 plan：`pr6_template_endpoint_anchoring_plan_2026-04-25.md`；user 明确 2026-05-10 决定"作为 PR-6-sup1，不塞进 PR-6 主线"。
> 主文档：`docs/topic1_within_event_dynamics.md` §7.10 加一行 "Topic 4 preflight" 回链。

> **Plan revision log**：
> - **v1（2026-05-10 初稿）**：N1 = channel-marginal-preserving null；rank matrix 直接用 valid_mask 排 rank。
> - **v2（2026-05-10 user review 修订）**：
>   1. **N1 改为 pseudo-endpoint position null**（§7.2）。原 v1 N1 在数学上退化（保留 row-marginal `Σ_p count(c, p)` ⇒ 保留 `count(c, p)` 全表 ⇒ Δ_null ≡ Δ_obs，p-value ≡ 1）。修订后 N1 保留观测 H_p_norm 整曲线，exact 枚举 `C(n_valid, 2)` 个 pseudo-endpoint 位置对计算 Δ_null。
>   2. **新增 §6.0 participation handling rule**：lagPatRank 含 non-participating channels 的 sentinel 值，必须按 `bools[c, e]` 过滤。锁定 Option B = all-valid-participating events filter，逐 cluster 报告 `drop_rate`、设 eligibility flag (`ok` / `high_drop_rate_warning` / `excluded_low_kept_events`)。
>   3. 相应更新 §7.3 双重显著、§10.1 helper 拆为两函数 (`rank_entropy_null_N0` + `rank_entropy_null_N1_pseudo_endpoint`)、Task 3 + Task 4 测试合同、§12.4 caveat。
>
> **TDD 仍未起**——v2 plan 落盘后等 user 审阅再起 Task 1。

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

### 6.0 Non-participating channels — Participation handling rule（user 2026-05-10 review，写死）

> ⚠️ **Topic 1 既有规则**：lagPatRank 包含 non-participating channels 的 sentinel 值（NaN / 0 / 等）；不能把 sentinel 当真实传播 rank 平均进去。如果按 valid_mask 固定通道集对每 event 做 argsort，会把 "channel 没参与该 event" 与 "channel 参与且排在低位" 混为同一种 rank=1，**污染 endpoint position 的 entropy 估计**。`bools[c, e]` 才是 ground truth。

**锁定规则（Option B：all-valid-participating events filter）**：

每 subject 每 cluster k，**只保留 valid_mask 内全部 channel 都参与的 events**：

```
keep_event(e) = all(bools[c, e] for c in valid_mask)
R_k = stack(r_e for e in cluster_k if keep_event(e))
```

具体步骤：

1. 取 `valid_mask`（PR-6 优先 / PR-2 sentinel fallback）→ `valid_idx = where(valid_mask)`
2. 对每属于 cluster k 的 event e：
   - 计算 `n_part_e = #{c ∈ valid_idx : bools[c, e] = True}`
   - **若 `n_part_e < n_valid` → 丢弃该 event**（`drop_partial_participation`）
   - 否则取 valid channels 的 lagPat 时间，per-event min-subtraction → argsort-of-argsort → `r_e ∈ {1..n_valid}^{n_valid}`
3. 拼 `R_k`：shape `(n_events_kept_k, n_valid)`

**Drop rate 报告（必须）**：每 subject 每 cluster 输出：

- `n_events_total_k`（cluster 原始事件数）
- `n_events_kept_k`（all-valid-participating 通过的事件数）
- `drop_rate_k = 1 − n_events_kept_k / n_events_total_k`

**Cluster-level eligibility gate（写死）**：

- `n_events_kept_k < 50` → 该 cluster 标 `excluded_low_kept_events`，subject-level `Δ_subject` 用剩余 cluster 算（若两 cluster 都被排除 → subject 整体 `excluded_no_participating_events`）
- `drop_rate_k > 0.5` → 该 cluster 标 `high_drop_rate_warning`，仍参与计算但在 cohort 输出加红旗 flag（**不**自动剔除——降低主观介入空间，drop rate 留给读者判断）

**Sensitivity follow-up（写死，本批不做）**：

- 若 cohort 中 `high_drop_rate_warning` cluster 占比 > 30% → 后续做 X=80% 阈值的 partial-participation rule（保留 ≥80% 参与的 events，缺失 channel rank 用 `rank_among_participating` reformulation）。**本批锁 X=100%（all-valid-participating），任何 X<100% 的 sensitivity 必须独立 follow-up plan，不修当前 N0/N1。**

**为什么 Option A / C 不选**：

- **Option A**（per-event 只用参与通道排 rank，n_valid 跟事件变）→ position p 在跨 events 上不可比，`H_p_norm` 失定义。
- **Option C**（variable n_part dense-rank entropy + position 归一化）→ 实现复杂、引入额外 normalization 自由度（按 n_part 还是按 valid_mask？）、reviewer 容易抓 cohort claim 漏洞。仅作 follow-up 候选。

### 6.1 Per-event rank matrix

每 subject 每 cluster k（在 §6.0 过滤后）：
- `valid_mask` 由 PR-6 优先 / PR-2 sentinel fallback 抽取（与 rank_displacement v14 同 provenance；CLAUDE.md "valid_mask semantics" 合同）
- `n_valid = sum(valid_mask)`
- `R_k`：shape `(n_events_kept_k, n_valid)`，每行是 all-valid-participating event 的整数 rank 向量 ∈ `{1, ..., n_valid}^{n_valid}`

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

### 7.2 N1 — Pseudo-endpoint position null（user 2026-05-10 review 修订；replaces 退化 N1）

> ⚠️ **原 plan 的 N1 = "channel-marginal-preserving" 在数学上退化**：`Δ_subject` 由 `count(c, p)` 算出；保留 row-marginal `Σ_p count(c, p)`（即每 channel 的 rank 分布）→ 实际上保留了 `count(c, p)` 全表（因为每 event 内 rank 是全排列，row sum 已唯一确定 row 内分布）→ `H_p_norm` 与 `Δ_null` 与观测同 → p-value ≡ 1。该 null 不能用。

**修订 N1（pseudo-endpoint position null）**：保留观测 `H_p_norm` 整条曲线**不变**，**随机抽两个 rank positions 当 pseudo-endpoint** 计算 `Δ_null`：

- 对每对 `(p1, p2) ∈ {1, ..., n_valid}` (`p1 < p2`)：
  - `Δ_pair = mean(H_{p1}, H_{p2}) − mean(H_q for q ∉ {p1, p2})`
- **Exact null（推荐）**：枚举全部 `C(n_valid, 2)` 对（n_valid=10 → 45 对；n_valid=15 → 105 对）。Δ_obs = `Δ_(1, n_valid)`。
- **p-value**：`p_N1 = #{(p1, p2) : Δ_pair ≥ Δ_obs} / C(n_valid, 2)`
- **检验问题**："`{1, n_valid}` 这对位置是否真的比任意其他两 rank position 的 entropy 差更高？"

**与 N0 的分工**：

| Null | 保留 | 检验 |
|---|---|---|
| N0 (per-event rank shuffle) | nothing | `Δ_obs > 0` 是否真信号 vs random rank baseline |
| N1 (pseudo-endpoint position) | 整条 `H_p_norm` 曲线 | 端点位置 `{1, n_valid}` 是否**特殊**（vs 任意位置对） |

N0 与 N1 互补：N0 否决"全随机"零假设；N1 否决"任意位置对都行"零假设。两个都通过 → 才能 descriptive 报告"端点位置确实更高熵"。

**确定性**：N0 用 `np.random.RandomState(seed=base_seed_N0=0)`；N1 是 exact 不需要 seed。

### 7.3 Per-subject p-value

- N0 p-value：`p_N0 = (1 + #{Δ_null_N0 ≥ Δ_subject}) / (n_perm_N0 + 1)`，n_perm_N0 = 1000
- N1 p-value（exact）：`p_N1 = #{(p1, p2) : Δ_pair ≥ Δ_obs} / C(n_valid, 2)`
- Cohort sign test 在 (`p_N0 < 0.05` AND `p_N1 < 0.05`) 双重显著 subject 子集上做（descriptive，不调 α）

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
├── per_subject/<subject>.json           # 每 cluster: H_p_norm 全曲线 + Δ + p_N0 + p_N1 + asymmetry
                                          #            + n_events_total_k + n_events_kept_k + drop_rate_k
                                          #            + eligibility_flag ∈ {ok, high_drop_rate_warning, excluded_low_kept_events}
├── cohort_summary.json                  # 35 subject 汇总 + swap_class 三组分布 + Kruskal-Wallis
                                          # + cohort_drop_rate_summary (median, max, n with high_drop_rate_warning)
└── figures/
    ├── README.md                                       # 中文，按规范，图实际生成后写
    ├── H_p_curves_by_swapclass.{png,pdf}              # 35 subject H_p_norm vs p，按 swap_class 三色叠加；high_drop_rate_warning 用虚线
    ├── delta_by_swapclass_box.{png,pdf}               # Δ_subject 按 strict/candidate/none box + 散点；点大小 = n_events_kept
    └── delta_obs_vs_null_scatter.{png,pdf}            # x = Δ_obs, y = N1 p_N1（exact, 0-1），identity 红线；颜色 = swap_class
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

def rank_entropy_null_N0(
    R_cluster: np.ndarray,            # shape (n_events, n_valid)
    n_valid: int,
    n_perm: int = 1000,
    base_seed: int = 0,
) -> np.ndarray:
    """Per-event rank shuffle resampling null.
    Returns Δ_null distribution of length n_perm. Determinism via seed.
    """

def rank_entropy_null_N1_pseudo_endpoint(
    H_p_norm: np.ndarray,             # shape (n_valid,) — observed curve
) -> Tuple[float, np.ndarray]:
    """Exact pseudo-endpoint position null.
    Returns (delta_obs, delta_pair_distribution) where delta_pair_distribution
    has length C(n_valid, 2). Caller computes p_N1 = #{Δ_pair ≥ Δ_obs} / C(n_valid, 2).
    No randomness — exact enumeration over all unordered position pairs.
    Raises ValueError on n_valid < 4 (need ≥ 2 endpoint + ≥ 2 middle).
    """
```

**Stub 合同**：两函数都 `raise ValueError` on 输入退化（`n_valid < 4`、空矩阵、NaN 输入），不返回 plausible 默认值。两函数**分开**——避免单一 `null_kind` 入口因 N0/N1 不同的 seed/perm 语义而产生隐性 bug。

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

### Task 3 — Surrogate construction（修订 2026-05-10：N1 改 pseudo-endpoint exact）

- [ ] **Step 3.1**: 写 test `test_null_N0_returns_zero_mean_delta` —— N0 1000 perm（per-event rank shuffle）→ null Δ mean 接近 0（容差 0.02）；两次跑 base_seed=0 给同 null 分布
- [ ] **Step 3.2**: 写 test `test_null_N1_endpoint_pair_in_distribution` —— exact null 包含 `Δ_(1, n_valid)` 自身（`n_valid=10` 时 distribution 长 = 45）；验证 `Δ_(1, n_valid) == delta_obs` 出现在分布中
- [ ] **Step 3.3**: 写 test `test_null_N1_uniform_curve_yields_zero_delta_everywhere` —— 输入 `H_p_norm` 全等于 0.5 → 所有 pairs 的 Δ_pair = 0 → p_N1 = 1.0
- [ ] **Step 3.4**: 写 test `test_null_N1_endpoint_special_under_confluence_curve` —— 构造 `H_p_norm = [0.9, 0.3, 0.3, 0.3, 0.9]`（端点高、中段低）→ Δ_(1,5) 在分布最大值 → p_N1 ≤ 1/C(5,2) = 0.1
- [ ] **Step 3.5**: 写 test `test_null_N0_n_valid_below_4_raises` 与 `test_null_N1_n_valid_below_4_raises`
- [ ] **Step 3.6**: 实现 `rank_entropy_null_N0`（per-event rank shuffle）+ `rank_entropy_null_N1_pseudo_endpoint`（exact enumeration via `itertools.combinations`）
- [ ] **Step 3.7**: 测试 → PASS

### Task 4 — Per-subject pipeline integration（含 §6.0 participation 过滤）

- [ ] **Step 4.1**: 写 test `test_partial_participation_events_are_dropped` —— 合成 cluster：1000 events 全参与 + 500 events 只 70% 参与 → 过滤后 R_k.shape = (1000, n_valid)；`drop_rate_k = 500/1500 = 0.333`
- [ ] **Step 4.2**: 写 test `test_low_kept_events_triggers_excluded_flag` —— 合成 cluster 仅 30 events 全参与 → eligibility_flag = `excluded_low_kept_events`（< 50 阈值）
- [ ] **Step 4.3**: 写 test `test_high_drop_rate_triggers_warning_not_exclude` —— 合成 cluster 100 全参与 + 200 partial → drop_rate=0.667 > 0.5 → flag = `high_drop_rate_warning`，**仍参与计算**
- [ ] **Step 4.4**: 写 integration test `test_per_subject_pipeline_on_synthetic_subject` —— 合成 stable_k=2 subject，cluster A endpoint random + middle deterministic（高 Δ），cluster B 全随机（Δ≈0），各 1000 全参与 events → 验证 per_subject JSON schema (含 `n_events_total_k`, `n_events_kept_k`, `drop_rate_k`, `eligibility_flag`) + Δ 数值方向
- [ ] **Step 4.5**: 实现 `run_subject_rank_entropy(subject_data) -> dict` 在 `src/rank_displacement.py`，含 §6.0 过滤 + §7.2 N0+N1 双 null 调用
- [ ] **Step 4.6**: 测试 → PASS

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

### 12.4 Partial-participation drop（user 2026-05-10 review，§6.0 锁死）

本 plan 用 **Option B = all-valid-participating events filter** 处理 lagPatRank 中的 non-participating channel sentinel。Caveats：

- `H_p_norm` 是在过滤后的子集 events 上算的；若 `drop_rate_k` 高（plan 阈值 > 0.5 → `high_drop_rate_warning`），结论描述的是"参与最完整的 events" 子集，**不是** cluster 的全部 events。
- Cohort `Δ` 中位数 / Kruskal-Wallis 必须配 `cohort_drop_rate_summary`（drop rate median + max + n with warning）一起读，**单看 Δ 不足以判定**。
- 若 cohort 内 `high_drop_rate_warning` 占比 > 30% → 触发 follow-up（X=80% 阈值的 Option A/C 变体），**本批不做**，写进 archive results "Next steps"。
- **不对** `drop_rate` 与 `Δ_subject` 做相关 / 回归（避免 reviewer 当 cohort claim 升级；descriptive only）。

### 12.5 Cluster 数与 stable_k

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
