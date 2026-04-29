# PR-7 Step 3.5：Burst-Level Template Run Diagnostic（plan-of-record）

> 状态：plan-of-record，2026-04-29（v1）
> 性质：**post-hoc exploratory diagnostic**——解释 H1 cohort NULL 的 etiology，不改判 H1 verdict，不进任何 PASS / FAIL 判据
> 范围：仅在 PR-7 H1 primary cohort（n=6 fwd/rev reproduced）上跑；H2 推广留作后续条件性决定
> 上游：`docs/archive/topic1/pr7_template_antagonistic_pairing_plan_2026-04-28.md`（PR-7 主 plan-of-record）
> 上游结果：`docs/archive/topic1/pr7_template_pairing_results_2026-04-29.md`（H1 NULL 已封）

---

## 1. 这个 step 只回答什么问题

**核心问题**：H1 cohort 在 event-level 10s/30s reciprocal coupling 上 NULL（Wilcoxon p=0.844, sign 3/6, median 轻微负），但**事件 mark 序列是逐事件独立抽样的，还是组织成 same-template burst / run，burst 之间才切换？**

这是 archive results §11 marked-point-process taxonomy 5 种 time-coupling 形式中的 form **(2) persistence** vs form **(5) independence** 区分。

**它不**回答：
- 是否存在 short-window cross-excitation（H1 已封 NULL）
- 是否存在 latent-state coupling（form 4，归 PR-4A/PR-5）
- 是否要重启 Ping-Pong 因果叙事（**禁止**）

---

## 2. 锁死的三条不变量（pre-registered，写死）

1. **不改写 H1 verdict**：H1 NULL 是 Step 3 已封板结论。Step 3.5 任何结果都不能回头修改 H1 PASS / NULL 判据
2. **不引入 PASS / FAIL 判据**：Step 3.5 仅产出**描述性统计 + 与 null 比较的 effect direction**；不写 cohort 主 p-value 阈值；论文里只能作为"H1 NULL 的解释机制"出现
3. **范围限 H1 n=6 cohort**：H2 negative cohort 推广是 Step 4 / 后续决定的事；本 step 不预跑 H2

---

## 3. Run / Burst 定义（最少新参数）

**核心规则**：连续相同 `cluster_label` 的最长事件 run，**block-aware**：跨 `block_time_ranges` 边界自动切断。

**形式化**：events 按 absolute time 排序后，
```
run = maximal consecutive subsequence (i, i+1, ..., i+k-1) such that:
    1. all events i .. i+k-1 belong to same block (block_s identical)
    2. all labels[i] == labels[i+1] == ... == labels[i+k-1]
```

**显式不引入**：
- ISI 阈值切断（没有 "ISI > X 强制结束 burst" 这种 ad hoc 参数）
- 多 cluster (k > 2) 的特殊处理（H1 cohort 全 stable_k=2，固定二值）
- 任何空间或几何 filter（仅按 mark 同质性切分）

---

## 4. Metrics（去掉用户提议中的逻辑陷阱）

### 4.1 用户原始提议的逻辑陷阱

> ❌ `p_switch_after_run = #run transitions A→B or B→A / #run transitions`

按定义，run 边界本身**就是** label 转换（A-run 后必接 B-run，否则两者会被合并成同一 run）。所以这个比例**恒为 1.0**，不携带任何信息。Step 3.5 **不**采用此 metric，避免 plan 内置数学错误。

等价的非平凡 event-level 测量是 `P(label[i+1] != label[i])`——已在 Step 3 的 `compute_transition_odds` 中报告（与 baseline_odds 对比）。**Step 3.5 不重复报告**。

### 4.2 实际采用的 metrics（per subject）

**A. Run-length 统计**（主诊断 — form 2 persistence 测验）：
- `n_runs_total`、`n_runs_per_block` 分布
- `mean_run_length` / `median_run_length` / `p95_run_length`
- 同 label 拆分：`mean_run_length_a` / `mean_run_length_b`（label 0 / label 1 各自）

**B. Run 时间结构**：
- `run_duration_seconds`：每 run 内最后事件 − 第一事件的时间跨度（中位 + p95）
- `within_run_IEI_median`：同 run 内连续两事件的 ISI 中位
- `between_run_gap_median`：相邻 run 之间（前一 run 末尾事件到后一 run 起始事件）的 ISI 中位
- 关键比值 `gap_to_iei_ratio = between_run_gap_median / within_run_IEI_median`：> 1 表示 run 之间有显著时间间隙（temporal burst clustering），≈ 1 表示无显著聚集

**C. 全局 burst 占比**：
- `event_fraction_in_long_runs`：长度 ≥ 3 的 run 中事件数占总事件数比例
- `event_fraction_in_singleton_runs`：长度 = 1 的 run 中事件数占比

**D. lag-1 same-label excess（user 2026-04-30 加入）**：
- `lag1_same_empirical = P(label[i+1] == label[i] | both in same block)`
- `lag1_same_null_mean = mean over n_perm of P(label[i+1] == label[i]) under null shuffle`
- `lag1_same_excess = lag1_same_empirical − lag1_same_null_mean`
- **理由**：run length 是 lag-1 same-label dependence 的累积表现；`lag1_same_excess` 更直接、更稳定，且与 Step 3 `compute_transition_odds` 的 `1 − p_next_opposite` 数学上对应（差别仅在 null 来源：Step 3 用 i.i.d. 理论 baseline，本指标用 N2 / N1 null shuffle baseline）。提供"single-number diagnostic"，避免所有 form (2) vs (5) 解释都压在 run length 一个量上

### 4.3 Null hierarchy（user 2026-04-30 修正：N2 primary + N1 sanity）

**主 null = N2 local-window shuffle**（Step 1 `shuffle_labels_local_window`，30 min window，50% overlap，first-covering，per-block 独立）：
- 保持 30 min 局部 rate burst 与 block 隔离
- 与 Step 3 H1 主 null 一致 → Step 3.5 诊断与 Step 3 H1 NULL 在同一 null 框架内对话

**Sanity null = N1 block-aware shuffle**（Step 1 `shuffle_labels_block_aware`）：
- 仅 block 内全局 shuffle，作为 ceiling-style sanity
- 若 N2 lift 与 N1 lift 方向一致 → 持久性结构不依赖于具体 30 min 窗口选择

**显式不用**：N0 global / N3 circular shift / N4 — Step 3.5 是 post-hoc 诊断，单 metric × 双 null 已足；多 null 只会让结果矩阵无边界。

**Per-subject lift**（每 metric × 每 null）：
- `run_length_lift(N) = mean_run_length_empirical / mean_run_length_null(N)_mean` （> 1 = persistence）
- `gap_to_iei_lift(N) = gap_to_iei_ratio_empirical / gap_to_iei_ratio_null(N)_mean`
- `lag1_same_excess(N)` 直接用上面 §4.2 D 的差值定义（不取比值，因为可能跨过 0）

**Permutation budget**：n_perm = 500（每 subject × 每 null 单独 500；总 6 × 2 × 500 = 6000 perms，每 perm 仅几个 O(N) pass，不进 PASS gate；500 给 mean ± SE 足够）

---

## 5. Cohort decision tree（user 2026-04-30 三种收口）

cohort 判读基于 6 个 subject 的 N2 主 null 下 effect direction，**不**做 cohort-level p-value 主推断；N1 sanity 仅检查方向是否一致。

按 user 锁死的三类收口：

### 5.1 N2 `run_length_lift > 1`（cohort 多数）

**判读**：模板选择有 same-template persistence；H1 NULL **不是无时间结构**，而是时间结构方向**与 Ping-Pong 相反**（同模板成簇，而不是反向接力）。

**论文写法**：
> "Forward/reverse template marks exhibit same-template persistence (mean run length above N2 local-window shuffle null in {k}/6 subjects). The event-level reciprocal-coupling NULL (Step 3) is therefore not the absence of temporal structure but a structure pointing **opposite** to the Ping-Pong prediction: 10s windows fall mostly inside same-template runs rather than across switch boundaries. The two templates organize as a burst-level alternation pattern, not as event-by-event reciprocal triggering."

如同时 `gap_to_iei_lift > 1` → 加一句"burst 在时间上也成簇出现（gap >> within-run IEI），event-level 窗口尤其容易掉进 same-template burst 内部"。

### 5.2 N2 `run_length_lift ≈ 1`（cohort 多数）

**判读**：在 event / run 尺度上未见模板 mark dependence；`lag1_same_excess` 也接近 0；mark 序列在该尺度上像独立抽样。

**论文写法**：
> "At the event/run scale tested here, mark sequences are statistically indistinguishable from N2 local-window null shuffles ({k}/6 `run_length_lift` ≈ 1, |lag1_same_excess| < {ε}). We cannot detect mark dependence at this scale. Slower-state coupling (rate / vigilance / seizure proximity, archive §11 form 4) is left to a separate analysis (Topic 1 PR-4A / PR-5 framework)."

### 5.3 548 outlier 单独路径

**判读**：548 event-level excess(10s) = −0.20 是 cohort 中唯一极端值；如其 `run_length_lift` 显著高于 cohort 其他 5 个（e.g. > 2 vs cohort median ≈ 1.1），把 548 单独以 case-series 形式呈现：

**论文写法**：
> "Subject 548 shows extreme same-template burst clustering (run_length_lift = {x:.2f}, well above cohort median {m:.2f}); its event-level reciprocal-coupling negativity is fully consistent with this burst structure rather than indicating cluster-level reverse coupling."

**严禁**：用 548 升级为 cohort claim、用 548 推导 cohort 主结论方向。548 是 case-series 例外，cohort verdict 仍按其余 5 个 subject 的 majority direction 收口。

### 5.4 cohort 异质（3:3 split）

per-subject 描述为主，不做 cohort verdict；archive results doc 列出 6 subject 的所有 8 个数字，论文对应 framing 用"per-subject heterogeneous"。

---

## 6. Scope-creep 守门（**关键**）

| 试图做的事 | 是否允许 | 理由 |
|---|---|---|
| 把 Step 3.5 升级为 PR-7 新 primary endpoint | ❌ | H1 已封；Step 3.5 是 post-hoc，不能改写 |
| 用 burst-level 结果"恢复" Ping-Pong 叙事 | ❌ | 即使 cohort 显示 form (2)+(3) 都强，叙事仅是"双模板组织成 burst-level switching"，**不**是"反向接力" |
| 在 Step 3.5 内引入 ISI 阈值或新参数 | ❌ | 任何新参数会让 cohort 多重检验空间无边界，违反 plan §10 风险表 |
| 在 Step 3.5 跑 H2 cohort | ❌ | 留 Step 4 / 后续条件性决定；现在不预跑 |
| 改 H1 主图 1（fig1_cohort_excess_curve）| ❌ | 主图 1 已封；Step 3.5 出独立 fig5_burst_diagnostic.png |

---

## 7. 失败合同

| 场景 | 应对 |
|---|---|
| 6 subject 全部 `run_length_lift ≈ 1` | 报告"在无 ISI 阈值的 same-label run 定义下未见 persistence；数据与 mark-independent sampling 在已测 metric 上 compatible（最简洁解释，**不等于证明独立**）"；论文层面：H1 NULL + 在已测尺度上 compatible with form (5)；不能写"已证明独立" |
| 6 subject 全部 `run_length_lift > 1` | 报告 form (2) persistence；论文层面写：H1 NULL 是因为 fixed-window event-level 跨进 same-template burst 内部 |
| cohort 异质（3:3 split）| 单 subject case-series 描述；不做 cohort claim |
| 548 与 cohort 其他 5 个分离 | 报告 548 outlier 在 burst 维度上的具体特征；不升级为 cohort claim |

---

## 8. TDD（6 项）

测试文件加到现有 `tests/test_pr7_template_pairing.py`：

```
T_burst_1. test_compute_runs_basic:
    labels = [0,0,0,1,1,0,1,1,1]; single block
    -> n_runs=4, run_lengths=[3,2,1,3], run_labels=[0,1,0,1]

T_burst_2. test_compute_runs_block_isolation:
    block A: events with labels [0,0,0]; block B: [0,1,1]
    Even though both blocks start with label 0, the runs MUST split at block boundary
    -> 3 runs total: [0,0,0] in A, [0] in B, [1,1] in B

T_burst_3. test_compute_runs_alternating:
    labels = [0,1,0,1,0,1,0,1]
    -> n_runs=8, every run length=1 (singletons)
    -> mean_run_length = 1.0

T_burst_4. test_compute_run_metrics_constant_label:
    labels = [0,0,0,0,0,0,0,0,0,0]
    -> 1 run of length 10
    -> mean_run_length = 10, run_lengths_per_label = {0: [10], 1: []}

T_burst_5 (sanity). test_run_length_lift_under_independence:
    Generate i.i.d. labels with p=0.5 over 5000 events; single block
    Empirical mean_run_length should ≈ null mean_run_length (both ≈ 2)
    -> run_length_lift ≈ 1.0 (within ±0.1)

T_burst_6. test_lag1_same_excess_alternating_vs_clustered:
    alternating (0,1,0,1,...) -> lag1_same_empirical = 0
    clustered ([0]*N + [1]*N) -> lag1_same_empirical ≈ (N-1)/(2N-1) ≈ 0.5
    Under N2 shuffle on clustered: null mean ≈ 0.5 (because shuffle within
    sufficiently large window doesn't preserve global clustering — 但 N2
    是 30 min local-window shuffle，若 clustering 跨多个窗口，shuffle 后
    会破坏 large-scale clustering，null 接近 baseline ≈ 0.5)
    断言 alternating case 下 lag1_same_excess < -0.4
    断言 clustered case 下 (with 30min window covering everything in this test
    setup) lag1_same_excess close to 0 — sanity not rigorous
```

不引入 N4 / 不 touch 已有 16 项 PR-7 测试（Step 1+3）。

---

## 9. Step 拆分

| Sub-step | 名称 | 交付 | 预算 |
|---|---|---|---|
| **3.5a** | 代码层 + TDD | `compute_runs` / `compute_run_metrics` / `compute_run_metrics_with_null` 加到 `src/template_temporal_pairing.py`；T_burst_1..5 全绿 | 0.3 d |
| **3.5b** | H1 cohort 跑数 | 复用 H1 audit pass subject + per-subject JSON 已有 labels；新增 `--burst-diagnostic` 模式输出 `per_subject/<id>_burst.json` | 0.1 d |
| **3.5c** | Cohort 聚合 + 决策树判读 | 加到 `cohort_summary.json` 的 `burst_diagnostic` block；按 §5 表格输出 cohort verdict | 0.1 d |
| **3.5d** | 出图 | `scripts/plot_pr7_template_pairing.py` 加 `fig5_burst_diagnostic.png`：左 panel = 6 subject `run_length_lift` 散点，右 panel = `gap_to_iei_lift` 散点 | 0.2 d |
| **3.5e** | archive results doc 追加 §14 | `pr7_template_pairing_results_2026-04-29.md` 新加 §14 "Step 3.5 burst diagnostic" 段；§1 / §11 链回 §14；不改 §3-§8 H1 NULL 主结论 | 0.1 d |
| **总预算** | | | **~0.8 d** |

---

## 10. 与 PR-7 主 plan / 后续 PR 的边界

| PR / Step | 关系 |
|---|---|
| **PR-7 Step 3** (H1 NULL 已封) | Step 3.5 解释其 etiology，**不**修改其 verdict |
| **PR-7 Step 4** (H2 negative control) | 优先级仍低；待 Step 3.5 完成后**条件性**决定是否跑（仅当 H1 cohort 在 burst 维度上有方向时，可考虑 H2 是否也有同样 burst 结构）|
| **PR-7 Step 5/6/7** | Step 3.5 不影响后续 robustness / figures / final wrap |
| **未来独立 PR**：history-dependent regression model | Step 3.5 是单一 metric (run length) 的简单诊断；history model 是更全的 framework，不在本 plan 内 |

---

## 11. 论文层面预期影响

按本 plan 完成后，论文级 framing 的可能升级：

✅ **如 form (2) persistence 显著**：
> "Forward/reverse template marks exhibit same-template persistence (mean run length significantly above label-shuffle null in 6/6 subjects, including the 548 burst outlier). This explains the event-level reciprocal-coupling NULL: ten-second windows mostly fall inside same-template runs rather than spanning a switch boundary. The two templates organize as a burst-level alternation pattern, not as event-by-event reciprocal triggering."

✅ **如 form (5) independence**：
> "Mark sequences are statistically indistinguishable from independent draws given block-conditional label fractions; the two templates can be modelled as two independently-sampled propagation geometries on a shared network."

❌ **禁止写**：
- "Burst-level reciprocal coupling restores Ping-Pong"（不论 form 2 多强，run 边界恒为 switch 是 trivial 后果，不是新发现）
- 任何把 Step 3.5 升级为 cohort-level inferential 主结论的措辞

---

## 12. 历史

- **2026-04-29**：本 plan 落盘（PR-7 Step 3.5 plan-of-record，v0）；待 user 批准
- **2026-04-30 (v1)**：user 两点修正——(a) §4.2 加入 `lag1_same_excess`（与 Step 3 transition_odds 数学对应，single-number diagnostic 避免所有解释压在 run_length 一个量上）；(b) §4.3 null hierarchy 改为 N2 primary + N1 sanity（不只用 block-aware）。§5 cohort decision tree 按 user 三种收口路径重写。§8 TDD 加 T_burst_6（lag1_same_excess 锐度检查）。批准实施 §9 sub-steps
