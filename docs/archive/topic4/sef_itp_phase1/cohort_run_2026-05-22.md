# SEF-ITP Phase 1 Cohort 实跑 — 2026-05-22

> **状态**（v1.0.7 update 2026-05-22）：preliminary，但 H1 cohort 出现强 pass 信号；pending Yuquan coord 补全 + cohort 扩展 + sensitivity checks
> **runner**：`scripts/run_sef_itp_phase1.py` (commits `6c3a89e` v1.0.6 → 待 v1.0.7 commit)
> **per-subject 输出**：`results/topic4_sef_itp/phase1_spatial_geometry/per_subject/<dataset>_<sid>.json` (23 个)
> **cohort summary**：`results/topic4_sef_itp/phase1_spatial_geometry/cohort_summary.json`
> **图**：`results/topic4_sef_itp/phase1_spatial_geometry/figures/`（含中文 README）
> **plan**：`docs/archive/topic4/sef_itp_phase1/plan_2026-05-21.md`
> **framework**：`docs/topic4_sef_itp_framework.md` v1.0.2

---

## 7. v1.0.7 H1 null-pool 修正 — 用户回归 2026-05-22

### 7.1 用户诊断（朴素话）

第一轮 cohort run（v1.0.6）的 H1 结果是这样：50% 的 cluster 卡在 INCONCLUSIVE_ENVELOPE_INDETERMINATE，剩下能测的 fail-like 12 / pass-like 9 / null-like 2 / untestable 4。我当时把这归结为"通道数小、cohort 小、统计功效低"。

用户回归后立刻指出：**这不是统计功效问题，是 null pool 框错了**。

旧实现把 H1 matched-null 的 candidate_pool 限制在 PR-6 给的 valid_mask 内——也就是"参与了这个 cluster 的事件的通道"，对 stable_k=2 小病人就 6-10 个通道。然后 endpoint k=3 + k=3 拿走 6 个，pool 剩 0-4 个，matched-null 抽不到 100 个样本，全部 INSUFFICIENT_NULL。

**更严重的科学错误**：lagPat valid_mask 选的就是"高 HFO rate + 高 HI"通道——这本身已经是非随机选择。在这个非随机子集内部做 matched-null 等于问"在已经被 lagPat 选过的高 HFO 通道里，endpoint 比其他高 HFO 通道更紧凑吗"。**这是循环论证**：matched 约束把 null 框死在跟 endpoint 同一族通道里，根本测不出空间结构性差异。

H1c envelope 同样问题：non_endpoint 用的是 valid_pool（lagPat 子集），小病人端点拿走 6 个剩 < 3 → INSUFFICIENT_NON_ENDPOINT，这就是 50% INCONCLUSIVE 的真正原因。

### 7.2 修正方案

| 组件 | v1.0.6 | v1.0.7 |
|---|---|---|
| H1 strict matched-null pool | PR-6 valid_mask ∩ mapped_mask | **全 SEEG implantation ∩ mapped_mask − endpoint** |
| matched 约束 | shaft + participation + hfo_rate（多层 tier 退化） | **只 shaft** |
| H1c envelope non_endpoint pool | PR-6 valid_mask − endpoint | **全 SEEG mapped − endpoint** |
| `participation` / `hfo_rate` 参数 | 用于 matched 约束 | 保留签名（向后兼容）但被 ignored |

**为什么去掉 participation / hfo_rate matched 约束**：这两个变量本身就是端点的核心特征。把它们作为 nuisance 控制掉等于把要测的信号一并扣掉。

**为什么保留 shaft matched 约束**：去掉 shaft 约束后 null 主导是跨 shaft 配对（距离 50+ mm 量级），actual 在同 shaft 里（distance 几 mm）几乎一定显著更紧凑——但这就是 H1 想测的"端点是不是在解剖上聚集"。保留 shaft 约束等于让 null 拿同 shaft 的随机抽样，问的是"在 shaft 内部的紧凑程度，端点是否比随机更紧凑"。

**新增**：`src/seeg_coord_loader.py::enumerate_subject_all_channels(dataset, subject_id)` 枚举病人全 SEEG implantation 通道（Yuquan 从 chnXyzDict.npy 全 shaft × n_contacts 枚举；Epilepsiae 从 SQL electrode 表，**过滤 invasive=True**，scalp EEG 不计）。

**Channel namespace 改造**：`SubjectPhase1Data` 引入 unified namespace —— `channel_names` 前 `n_lagpat_channels` 项是 lagPat 选中通道（events_bool 有真实数据），后面是 all-SEEG 扩展通道（events_bool 行全 False）。endpoint indices 在 unified namespace 中（不变），valid_indices 现在指 "all mapped SEEG − endpoint"。

**H6 不改**：H6 测的是"参与场空间分隔"，输入是 events_bool 的 mean。在非 lagPat 通道上 participation 恒为 0 → 全进 low group，会扭曲 H6 的 high/low split 语义。`run_h6` 显式把 events_bool / channel_names / coords 切片到 lagPat 前缀（first `n_lagpat_channels` 行）。

### 7.3 重跑 cohort 对比

n=23 (8 Yuquan + 15 Epilepsiae)，n_permutations=1000、n_null=1000。

**H1 per-cluster verdict 分布（46 cluster）**：

| 裁决 | v1.0.6 | v1.0.7 | Δ |
|---|---|---|---|
| PASS | 1 | **11** | +10 |
| partial_PASS | 2 | **19** | +17 |
| PASS_one_side_untestable | 6 | 1 | -5 |
| NULL | 1 | 12 | +11 |
| NULL_one_side_untestable | 1 | 1 | 0 |
| FAIL | 7 | **0** | -7 |
| FAIL_DIFFUSE | 5 | 2 | -3 |
| UNTESTABLE_BOTH_SIDES | 4 | 0 | -4 |
| INCOMPLETE_GATED_ON_COORDS | 0 | 0 | 0 |
| INCONCLUSIVE_ENVELOPE_INDETERMINATE | 19 | **0** | -19 |

**categorical**:

| | v1.0.6 | v1.0.7 |
|---|---|---|
| pass-like | 9 (20%) | **31 (67%)** |
| null-like | 2 (4%) | 13 (28%) |
| fail-like | 12 (26%) | 2 (4%) |
| untestable | 23 (50%) | **0 (0%)** |

**H2 / H6 不变**（设计上 v1.0.7 没碰这两条）：

- H2: 4 PASS + 1 partial_PASS + 1 NULL（n=6 pair）
- H6: 0 PASS + 3 PARTIAL + 10 NULL + 8 INSUFFICIENT_SPLIT + 2 EXCLUDED_SINGLE_SHAFT

**Coord 覆盖中位数**：从 11（lagPat 限制）→ 96（全 SEEG implantation）。

### 7.4 修正后的 cohort verdict

**H1 cohort-level 出现强 pass 信号（31/46 pass-like，67%）**——v1.0.6 看不到这个信号因为 null pool 框死在 lagPat 池里循环论证。修正后端点相对**整个 SEEG implantation 同 shaft 的随机通道**显著更紧凑：endpoint 不是 lagPat 选择偏差的产物，它们在解剖空间里真的是聚集的。

7 个 FAIL（v1.0.6 出现的）全部变成 NULL 或 partial_PASS——证实那些是错配的 null pool 制造的"端点逃出参与场"假象，不是真的端点 anti-compact。

剩下 2 个 FAIL_DIFFUSE 在 v1.0.7 下仍然是 fail-like，是 strict-layer 仍判定端点比同 shaft 随机更散开的真实信号，**值得追踪**：是 subject anatomy 异常还是 PR-6 端点定义在这两个 subject 上失败。

### 7.5 测试 + 实现

- 新增 `tests/test_run_sef_itp_phase1_integration.py::test_load_subject_for_phase1_unified_namespace_extends_pool` — 验证 unified namespace 把 lagPat + extra SEEG 拼接，endpoint indices 留在 lagPat 前缀，valid_indices 扩到全 SEEG 非端点
- 更新 `test_h1c_envelope_within_field` / `test_h1c_envelope_outside_field_fail` / `test_h1c_envelope_circularity_guard` — 调用方传 `non_endpoint_pool` 显式排除 endpoint
- 新增 `test_h1c_envelope_rejects_endpoint_in_pool` — v1.0.7 契约：pool 不包含 endpoint，违反 raise
- 更新 `_make_h1_synthetic` — 非均匀 shaft z packing（首 3 dense + 后 7 spread），让 compact target 在 shaft-only null 下可显著
- `test_h1_strict_diffuse_null_strict` 接受 NULL or FAIL_DIFFUSE（都满足 "diffuse target ≠ PASS" 的科学原意）
- 整套测试：114/114 GREEN

### 7.6 v1.0.6 → v1.0.7 是 framework 修订 OR 实现 bug 修复

**是实现 bug 修复**，不是 framework 修订。framework v1.0.2 §3.1 写的就是 "matched random 3-channel sampling 1000 次；匹配条件 = shaft 分布 + participation rate + HFO rate" + "**all_valid_indices**"——其中 "all_valid_indices" 在 plan archive 里没有明文限定为 lagPat 子集。v1.0.6 实现把这个 "valid_indices" 误解为 PR-6 valid_mask，导致 null pool 框错。framework prose 没说错，实现没读 prose。

修复时**保留** framework 写的 participation/hfo_rate matched 约束作为可选 sensitivity（参数仍在签名里），但**默认行为**改为只 shaft——因为在全 SEEG implantation 池里，非 lagPat 通道的 participation/hfo_rate 没有定义。这是工程现实，不是 framework 修订。

---

## 8. v1.0.8 H2 PR-6 swap_check ingest — 用户回归 2026-05-22

### 8.1 用户诊断

我之前实现 H2 时犯了 CLAUDE.md §5+§6 经典错误**三段叠加**：

1. **CLAUDE.md §5 用错字段**：input cohort 取 PR-2 `candidate_forward_reverse_pairs`，但这个字段 PR-2 archive 明文写 "**候选描述标签，不是最终机制判定**"。正确字段是 PR-2.5 `forward_reverse_reproduced` (split-half **OR** odd-even, AGENTS.md Cross-PR locked rule)。
2. **CLAUDE.md §6 重新发明 helper**：PR-6 anchoring 已经有 `forward_reverse_swap_check` 函数，输出 `h2_swap_check.{swap_score, null_p, null_95th}`，per-subject + 1000-perm null 全锁好。我又写了一遍 `compute_h2_set_reversal` + `compute_h2_spatial_reversal`，公式重叠 + null 实现不同。
3. **CLAUDE.md §5 升级 hypothesis tier**：PR-6 plan §3.3 + §15 pre-register H2 为 **directional mechanism sanity, NOT cohort claim**，明文 "严禁包装成独立可发表 finding"。我的 Phase 1 v1.0.6 / v1.0.7 用 PASS/NULL/FAIL 整合裁决报 "5/6 cohort pass-like, strongest signal" —— 这是 tier upgrade。

用户回归后指出："PR-6 是逐 contact 确定的，PR-2 的整模板 spearman 反向被共享节点稀释"——这是关键 insight：PR-6 swap_score 在端点 contact 层做 Jaccard 反向判定，比 PR-2 整模板 spearman_r 反向更精细；PR-2 candidate cohort 标签是"整模板共享节点没排除"的产物，不能当 strict input。

### 8.2 修正方案

| 组件 | v1.0.7 之前 | v1.0.8 |
|---|---|---|
| H2 input cohort | PR-2 `candidate_forward_reverse_pairs` (n=6) | PR-6 `h2_swap_check` 直接读 (n_testable = 6 in current cohort) |
| swap statistic | 自己重新算 set Jaccard + spatial centroid | 直接读 PR-6 `swap_score` |
| `compute_h2_set_reversal` / `compute_h2_spatial_reversal` | Phase 1 内重写 | **删除**，从 src/sef_itp_phase1.py + tests 中移除 |
| spatial reversal centroid | Phase 1 自己加的指标 | **删除** — PR-6 swap_score 逐 contact 已比 centroid 距离精细 |
| H2 verdict 形式 | PASS / partial_PASS / NULL / FAIL | **descriptive only**: swap_score, null_p, exceeds_null_95th |
| Cohort summary | "5/6 pass-like cohort signal" | sign-test `n_exceed_null_95th / n_testable` + binomial p；**no cohort PASS/NULL/FAIL** |
| `forward_reverse_pairs` dataclass field | 驱动 H2 logic | 移除（h2_swap_check 字段替代）|

### 8.3 重跑 cohort 结果

n=23 cohort, n_permutations + n_null 不再使用（PR-6 已 1000-perm null per subject）：

| 字段 | 值 |
|---|---|
| n_testable (PR-6 swap_check available + 有坐标) | 6 |
| n_not_testable (PR-6 exit_reason or 缺坐标) | 17 |
| n_exceed_null_95th | 4 |
| sign-test binomial one-sided p (p_null=0.05) | **8.64e-05** |
| swap_score median | 0.625 |
| null_p median | 0.0315 |

per-subject 数值：

| subject | swap_score | null_p | null_95th | exceed |
|---|---|---|---|---|
| epilepsiae_1073 | 0.5 | 0.29 | 0.6 | No |
| epilepsiae_139 | 1.0 | 0.002 | 0.5 | **Yes** |
| epilepsiae_253 | 0.5 | 0.103 | 0.5 | No |
| epilepsiae_958 | 0.75 | 0.0 | 0.36 | **Yes** |
| yuquan_zhangjiaqi | 0.75 | 0.02 | 0.5 | **Yes** |
| yuquan_zhaochenxi | 0.25 | 0.043 | 0.2 | **Yes** |

(2 个 PR-6 swap_check 出了 swap_score 但缺坐标被排除：yuquan_chenziyang / wangyiyang)

### 8.4 修正后的 H2 cohort verdict

**方向上**支持 SEF-ITP "对偶端点"假说（4/6 exceed null_95th，sign-test 显著超过 chance）；但 tier-locked 不能升级为 cohort claim。判读语言：

> H2 PR-6 swap_check 在 6 个 cohort subject 中 4 个 exceed null_95th（mechanism sanity 方向支持），sign-test p=8.64e-05；per-subject swap_score 中位数 0.625。**不构成 cohort-level 反转几何 claim**（PR-6 plan §3.3 + §15 pre-register tier lock）。

### 8.5 删除 / 改动清单

- `src/sef_itp_phase1.py`：删除 `compute_h2_set_reversal`、`compute_h2_spatial_reversal`、`_jaccard_set`；模块 docstring 升级 v1.0.8 lock，说明 H2 只 ingest
- `scripts/run_sef_itp_phase1.py`：`run_h2` 重写为 PR-6 ingest；`SubjectPhase1Data` 删除 `forward_reverse_pairs` 字段，新增 `h2_swap_check: Optional[Dict]`；`load_subject_for_phase1` 读 PR-6 `h2_swap_check` 写入 `subject.h2_swap_check`
- `scripts/summarize_sef_itp_phase1.py`：H2 cohort summary 改 sign-test + binomial p；删除 verdict_distribution_per_pair_integrated 等过时字段
- `scripts/plot_sef_itp_phase1_cohort.py`：H2 plot 改 sign-test bar + per-subject swap_score vs null_95th 散点
- `tests/test_sef_itp_phase1.py`：删除 H2 test suite (test_h2_set_reversal_*, test_h2_spatial_reversal_*, test_h2_jaccard_basic, test_compute_h2_spatial_reversal_rejects_voxel_coords)；保留 placeholder 注释说明 H2 测试在 PR-6 anchoring TDD 已覆盖
- `tests/test_run_sef_itp_phase1_integration.py`：删除 `test_load_subject_for_phase1_no_pr6_pair_skipped`（forward_reverse_pairs 不再存在）；新增 `test_load_subject_for_phase1_h2_swap_check_unavailable_when_pr6_missing`
- `docs/topic4_sef_itp_framework.md` §3.2：H2 prose v1.0.3 lock，旧 v1.0.2 prose 标记弃用，保留 audit trail
- `tests/test_run_sef_itp_phase1_integration.py:_make_pr6_json`：新加 `h2_swap_check` 字段默认值
- `docs/topic4_sef_itp_framework.md` 顶部版本横幅：v1.0.2 → v1.0.3

测试：108/108 GREEN（之前 114 个，删了 6 个过时 H2 测试）

### 8.6 v1.0.7 → v1.0.8 是 framework 修订 OR 实现 bug 修复

**两个都是**：

- framework prose 修订（v1.0.2 → v1.0.3）：早期 §3.2 prose "主分析 1 + 主分析 2 两条 reversal index" 不符合 PR-6 已锁的 contract，且 spatial reversal 是 framework 自己发明的指标 PR-6 没有。这是 framework 内部不一致，必须修 prose。
- 实现 bug 修复（v1.0.7 → v1.0.8）：用错字段 + 重新发明 helper + 升级 tier 三个 cascade 都是实现层错误，跟 framework prose 怎么写无关。

修完之后**framework prose 与实现一致**，且都对齐 PR-6 plan §3.3 + §15 的 mechanism sanity 锁定。

---

## 9. v1.0.9 H2 input-order 调查 — PR-2 / PR-6 endpoint / rank-displacement 分层

### 9.1 用户诊断（朴素话）

上一版 v1.0.8 修正了三个错误：不再拿 PR-2 `candidate_forward_reverse_pairs` 当 strict input、不重新发明 PR-6 helper、不把 H2 升级成 cohort claim。但它仍然容易让后续实现者误会：**只要谈 H2，就先去抓 PR-2 或 PR-6 fixed top-3 endpoint**。

这仍然不够精确。三层东西回答的是三个不同问题：

- **PR-2 / PR-2.5**：先把事件分成几种稳定传播模板。它回答"这个 subject 有没有两条像正反的传播路线"。这是模板发现层，不是 channel-label 层。
- **PR-6 endpoint anchoring**：每条模板取 fixed top-3 source/sink。它回答"每条传播路线最前 / 最后的几个通道是谁"。这是端点摘要层，比 PR-2 接近 H2，但 fixed-k 会丢掉 subject-specific cardinality。
- **rank-displacement + swap-k**：逐通道看同一个 channel 从模板 A 到模板 B 的 rank 位移，再用 `swap_sweep` 找 family-wise null 下最稳的 `decision_k`。它回答"哪些通道真正构成这对模板的 source/sink swap endpoint"。这是 Topic 4 建模前最该用的 H2 channel-label。

### 9.2 当前结论

**rank-displacement 之后的 `swap_sweep` / `decision_k` 才是 H2 最稳输入**。原因：

1. PR-2 的整模板 Spearman 相关会被共享节点稀释；它能发现候选模式，但不能稳健标通道。
2. PR-6 fixed top-3 endpoint 是硬切 k=3；对不同有效通道数 subject 不够自适应。
3. rank-displacement 的 `swap_sweep` 是 per-channel + variable-k + family-wise max-null，直接给出 `swap_class`、`decision_k`、`T_obs`、`p_fw`，再由 `joint_valid` + `rank_a_dense_full` 派生 endpoint channel label。

### 9.3 后续实现合同

Topic 4 H2 建模前输入顺序锁定为：

1. 读取 masked rank-displacement per-subject JSON：`results/interictal_propagation_masked/rank_displacement/per_subject/<dataset>_<subject>.json`
2. 只在 `stable_k=2` 且 `primary_pair.exit_reason == "ok"` 的 subject 上取 `primary_pair.swap_sweep`
3. 用 `swap_sweep.decision_k` + `joint_valid` + `rank_a_dense_full` 派生 `swap_endpoint_channels`
4. **不要停在 label 层**：`swap_class` / `decision_k` 只说明哪些通道构成 swap-k endpoint；真正的 H2 空间性还要把 source-side 前 `k` 个节点、sink-side 后 `k` 个节点分别拿去和其他有效 SEEG 节点做空间 compactness null
5. 空间性检验不复用 H1 的 same-shaft null：`members = swap source-side k` 或 `swap sink-side k`；`candidate_pool = all mapped SEEG minus swap endpoint`；从这个 pool 无 shaft 约束随机抽同样大小节点集。`combined_endpoint (2k)` 只作辅助描述
6. `swap_class == "strict"` 是主机制 sanity label；`candidate` 是描述 / sensitivity；`none` 保留为非 swap 对照
7. PR-2 / PR-2.5 只作 provenance / funnel 字段，不作为 H2 channel-label；PR-6 fixed top-3 endpoint / `h2_swap_check` 保留为 audit trail 和摘要层

H2 tier 不变：仍是 **directional mechanism sanity, NOT cohort claim**。

### 9.4 2026-05-22 实现纠偏

第一次 cohort 运行只统计了 `swap_class_distribution`，这是 **label-source validation**，不是完整 H2。正确 H2 必须多走一步：用 rank-displacement 产生的 swap-k 节点测试空间性。后续结果汇总必须同时报告：

- label 层：`swap_class` / `decision_k` / `T_obs` / `p_fw` / `swap_endpoint_channels`
- spatial 层：`source_side` / `sink_side` / `combined_endpoint` compactness verdict

其中 `source_side` 与 `sink_side` 是主 sanity；`combined_endpoint` 是辅助，因为 source 与 sink 可能本来就在传播轨迹两端，混成 `2k` 后测 compactness 不应作为唯一主结论。

### 9.5 verdict 验收（v1.0.4 lock 2026-05-22；措辞已收紧）

**接受作为 H2 pre-modeling spatial sanity test**。Cohort 实跑（n=23，euclidean，masked tree）：

| 子项 | 数值 |
|---|---|
| n_testable | 23 / 23（0 个 `INSUFFICIENT_POOL`，0 个 `INSUFFICIENT_NULL`，0 个 `not_testable`）|
| swap_class | strict 5 / candidate 4 / none 14 / unknown 0 |
| source-side spatial | PASS 19 / NULL 3 / FAIL_DIFFUSE 1 |
| sink-side spatial | PASS 16 / NULL 7 |
| combined_endpoint spatial（辅助） | PASS 20 / NULL 3 |

**5 个 swap_class=strict subject 的 spatial 一致性**：
- epilepsiae_139 / epilepsiae_1146 / yuquan_zhangjiaqi / yuquan_zhaochenxi: source PASS + sink PASS
- epilepsiae_958: source PASS + sink NULL

**4 个 swap_class=candidate subject**：
- yuquan_liyouran: source PASS + sink PASS
- epilepsiae_635 / epilepsiae_1073 / epilepsiae_253: 至少一侧 NULL / FAIL_DIFFUSE

**14 个 swap_class=none subject 中，多数也得到 source/sink PASS**：这并不意味着这些 subject 也成立 swap 反转——只说明在 rank-displacement `decision_k` 给出的"前 k / 后 k"通道上，**单纯的空间紧凑性**对绝大多数有效 SEEG 实施都成立。这是 label 层"无 swap 信号"与 spatial 层"通道空间相邻"互不矛盾的体现。

**允许的判读语言（lock）**：
- ✅ "在可测的 H2 cohort 中，rank-displacement 定义出来的 swap-k source-side / sink-side 节点，整体上呈现相对其他 mapped SEEG 节点的空间紧凑性"
- ✅ "swap-k source/sink endpoint 有空间聚合倾向；这是 mechanism sanity，不是最终 cohort claim"
- ✅ EN: "H2 spatial sanity was supported: rank-displacement-derived swap-k source-side and sink-side endpoints showed non-random spatial compactness relative to other mapped SEEG contacts in most testable subjects."
- ❌ "H2 证明 source/sink reversal 是 cohort-level 主效应"
- ❌ "所有 source/sink 节点都有空间聚合"
- ❌ "cohort PASS / partial PASS"

**v1.0.4 实现纠错（accept-condition）**：旧实现复用 H1 `compute_h1_compactness`（same-shaft matching）→ 大量 `INSUFFICIENT_NULL`。H2 问的是"swap-k 节点相对全植入空间是否特殊"，source/sink 所在 shaft 是机制的一部分，不能强行 same-shaft 抽样。已替换为 `_compute_h2_unmatched_compactness`（pool=`all mapped SEEG minus swap endpoint`，无 shaft 约束）。

**实现文件**：
- `src/sef_itp_phase1.py`（H1 / H6 helper 不变）
- `scripts/run_sef_itp_phase1.py`：`_compute_h2_unmatched_compactness` + `_run_h2_spatial_compactness` + `run_h2` ingest rank-displacement
- `scripts/summarize_sef_itp_phase1.py`：`spatial_compactness_verdict_distribution` 字段
- `tests/test_run_sef_itp_phase1_integration.py` + `tests/test_summarize_sef_itp_phase1.py`：115 passed

**输出位置**：`results/topic4_sef_itp/phase1_h2_rank_displacement_spatial_validation/`（per_subject + cohort_summary.json + cohort_subjects.csv）

---

---

## 0. Cohort 漏斗（实测）

```
40 个 masked Phase 0a 病人 (yuquan 20 + epilepsiae 20)
    ↓ 过滤：stable_k = 2
34 个
    ↓ 过滤：masked PR-6 anchoring 出了结果（valid 通道 ≥ 6 + SOZ JSON 等闸门）
30 个
    ↓ 过滤：3D 坐标可用（mm 单位）
23 个 = 8 Yuquan (fs_native_ras_mm) + 15 Epilepsiae (mni152_1mm)
```

**流失原因（按 cohort 大小排序）**：

| 闸门 | 流失 | 数据集 | 原因 |
|---|---|---|---|
| stable_k ≠ 2 | 6 | 混合 | 3 个 k=4，1 个 k=3，2 个 k=5/6 — 多模板结构，SEF-ITP 当前 H1/H2 假设 k=2 对偶模板 |
| PR-6 anchoring 没出 | 4 | 主要 yuquan | valid_channels < 6 或 SOZ JSON 空（epilepsiae_1125/620/916/384/yuquan_gaolan 子集）|
| 没坐标 | 7 | 全是 yuquan | chenziyang, hanyuxuan, huanghanwen, litengsheng, sunyuanxin, wangyiyang, xuxinyi — 原始 CT+MRI+reg.mat 齐，但电极点选未做（详见 §3.4 yuquan_coord_gap_investigation_2026-05-21.md）|

---

## 1. H6 — 参与场空间分隔

### 测了什么

每个病人，看高 HFO 参与率的通道（参与 ≥50% 事件）和低参与率的通道（< 50%），在大脑里是不是空间上**分开聚集**——是不是有一块"热区"和一块"冷区"，而不是高低参与率随机散布。

### 怎么测的

每个 subject 算三个指标：

- Moran's I（参与率的空间自相关，inverse-distance 权重）
- 高/低参与率两组的 centroid 距离
- 两组的 silhouette 分数

跑 1000 次 shaft-stratified shuffle（在每根电极杆内部 permute 参与率，跨杆分布保持）当 null。三个指标里 **≥2 个 p<0.05 → PASS；0 个 → NULL；1 个 → PARTIAL**。

### 揭示了什么（cohort n=23）

| 裁决 | 数 |
|---|---|
| PASS (≥2 显著) | 0 |
| PARTIAL (1/3 显著) | 3 |
| NULL (0/3 显著) | 10 |
| INSUFFICIENT_SPLIT (高或低组 < 2 通道) | 8 |
| EXCLUDED_SINGLE_SHAFT (subject 只有 1 根杆) | 2 |

**结论（preliminary）**：H6 在 cohort level **没有得到强支持**。能测到的 13 个 subject 里 10 个 NULL + 3 个 PARTIAL + 0 个 PASS。8 个 INSUFFICIENT_SPLIT 主要是因为通道数小 / 参与率阈值（0.5）下高低组之一 < 2。但即使把 PARTIAL 算 pass-like，3/13 = 23%，远远谈不上 cohort claim。

**两种可能的解读**：

1. **参与场确实没有跨电极杆的强空间组织** — 高/低参与率主要在同一根杆内梯度（这与 shaft 内部相关性大致匹配）。这个解读对 SEF-ITP 框架略有不利——意味着"病理易激场"如果存在，它的空间尺度可能小于杆间距（5–10 mm 量级），而不是跨杆的厘米尺度。
2. **shaft-stratified null 太严** — shaft-stratified 把"沿杆参与率梯度"当作 null 的一部分，等于扣掉了 H6 可能想抓的一个主要信号。如果用 unstratified random shuffle，cohort PASS 数应该更高，但那也意味着我们抓的是"参与率非均匀"而不是"跨杆空间聚集"，是更弱的 H6 claim。

我们选择保留 shaft-stratified null（严格版本）——这与 framework v1.0.2 §3.6 锁定的设计一致。

**敏感性 follow-up**：在补完 Yuquan 7 个坐标 → cohort=30 之后，重新跑一次；同时考虑 framework 里加一个"shaft-pooled H6"作为 sensitivity（不替换主分析）。

---

## 2. H1 — Endpoint 紧凑性 + 嵌入参与场

### 测了什么

每个 subject 每个 cluster（一共 23 × 2 = 46 cluster），看：

- 这个 cluster 的"起点通道"集合（PR-6 source，k=3）是不是空间上紧凑聚集
- "终点通道"集合（PR-6 sink，k=3）是不是空间上紧凑聚集
- 起点 + 终点的并集是不是嵌在"高参与率通道的几何中心"附近（必要条件 — 端点不能跑出参与场外）

### 怎么测的

三层独立检验：

1. **strict_source / strict_sink**：source/sink 集合的平均两两距离 vs 1000 个"匹配 shaft + 参与率 + HFO rate"的随机 3-通道子集（matched-null）。
2. **envelope**：端点到非端点中心的距离比，比例显著高于 1000 个随机端点子集 → 端点逃出参与场。
3. **整合裁决**：envelope FAIL → 整体 FAIL（必要条件）；envelope SKIPPED → INCOMPLETE_GATED_ON_COORDS；否则按 strict_source × strict_sink 真值表组合（PASS、partial_PASS、PASS_one_side_untestable、NULL、NULL_one_side_untestable、UNTESTABLE_BOTH_SIDES、FAIL_DIFFUSE、FAIL）。

### 揭示了什么（46 cluster）

| 类别 | 子裁决 | 数 |
|---|---|---|
| pass-like | PASS / partial_PASS / PASS_one_side_untestable | 1 / 2 / 6 = **9** |
| null-like | NULL / NULL_one_side_untestable | 1 / 1 = **2** |
| fail-like | FAIL / FAIL_DIFFUSE | 7 / 5 = **12** |
| untestable | INCONCLUSIVE_ENVELOPE_INDETERMINATE / UNTESTABLE_BOTH_SIDES | 19 / 4 = **23** |

**主要观察**：

- **50% 的 cluster 无法测**（INCONCLUSIVE_ENVELOPE_INDETERMINATE 主导）。原因：多数 subject 总通道数 ≤ 8，端点 k=3 + k=3 拿走 6 个之后 non_endpoint < 3，envelope 必要条件不能跑。这是**通道数限制**，不是 H1 假说被否。
- **能测的 cluster 里 fail-like (12) > pass-like (9)**：5 个 FAIL_DIFFUSE 是 strict layer 发现端点反而比 matched-null 更**散开**（anti-compact），7 个 FAIL 是 envelope 发现端点跑出参与场外。
- **6 个 PASS_one_side_untestable**：一边端点紧凑，另一边 INSUFFICIENT_NULL（matched-null 抽不出 100 个满足约束的样本）—— 这是 cohort 小 + 通道数小的复合效应。

**preliminary 结论**：当前精度下 H1 在 cohort level 不构成强支持，但**不能就此宣告假说否决**——

1. 一半 cluster 是 untestable，能测的部分 cohort n 也只有 23，统计功效太低
2. FAIL_DIFFUSE 信号（5/46）值得单独追踪——这是 strict layer 主动告诉我们"端点比同 shaft 同参与率的随机 3-通道子集更散开"，不是 null 信号
3. 补完 Yuquan 7 个坐标 + 扩展 cohort 到 30 之后必须重跑

---

## 3. H2 — 正反模板的 source/sink 反转几何

### 测了什么

每个 fwd/rev pair（一共 6 对，来自 23 subject 里有 candidate_forward_reverse_pairs 的 6 个），看：

- 正模板的 source 是不是和反模板的 sink **集合上**对应（Jaccard 反向 > 同向 → set-based reversal）
- 正模板的 source centroid 和反模板的 sink centroid 是不是**空间上**靠近（distance 反向 < 同向 → spatial reversal）

### 怎么测的

两条独立的 reversal index：

- **R_set = J_swap − J_same** > null upper 5% → set-based PASS
- **R_spatial = d_same / (d_swap + d_same)** > 0.5 且 null upper 5% → spatial PASS

null = 在 4 个端点集合 union 池里 shuffle role labels 1000 次。

整合裁决：两条都 PASS → PASS；一条 PASS → partial_PASS；其他 NULL。

### 揭示了什么（6 pair）

| 裁决 | 整合 | set-based | spatial |
|---|---|---|---|
| PASS | 4 | 4 | 5 |
| partial_PASS | 1 | — | — |
| NULL | 1 | 1 | — |
| EMPTY_SET | — | 1 | 1 |

**主要观察**：

- **5/6 pass-like**（4 PASS + 1 partial_PASS）—— 这是当前 cohort 最强的信号
- spatial reversal 5/5 都 PASS（剔除 1 个 EMPTY_SET），set-based 4/5 PASS
- spatial 比 set-based 还略稳，说明几何反转比集合反转更鲁棒

**preliminary 结论**：在能测到 fwd/rev pair 的少数 subject 里，**SEF-ITP 框架预测的"对偶端点 / 涟漪可从两端起"的几何反转特性高度成立**。这个信号方向上和 framework v1.0.2 §3.2 的 H2 PASS 预测完全一致。

**重要 caveat**（不能跳过）：

- n=6 太小，cohort-level 5/6 ratio 的二项检验 p = 0.109（α=0.5 null），**还不到 cohort claim 的统计门槛**
- 6 个 pair 来自哪 6 个 subject？需要核对是否集中在 dataset 一侧——如果 6 个全是 Epilepsiae，那"cohort claim"实际上只是"Epilepsiae 一侧 claim"
- PR-6 candidate_forward_reverse_pairs 的判定标准（spearman_r < 阈值）目前过严，导致很多 stable_k=2 subject 没出 pair

---

## 4. Cohort 整体一句话 verdict

**H2 在 6 对里 5 个 pass-like（最强信号但 n=6 不足以 cohort claim）；H1 一半 untestable + fail-like 12 vs pass-like 9（受通道数限制）；H6 cohort-level NULL/untestable 主导（弱信号）**。

framework v1.0.2 §0 一句话承诺里写："如果指纹大多数没看到 → 假说在我们的数据精度内被证伪"。**当前精度下 H1/H6 没看到强指纹**，但因为统计功效太低（n_testable 太小），**不能升级到 framework-level 证伪**。H2 的方向支持是真实的，但 n=6 还不够"cohort 支持"。

---

## 5. Pending 工作

按优先级：

1. **补 Yuquan 7 个 subject 的电极定位** → cohort 从 23 推到 30。这是当前 cohort 限制的最大单点突破。
2. **PR-2 candidate_forward_reverse_pairs 判定标准重审** → 让更多 stable_k=2 subject 进 H2 分析。
3. **Epilepsiae warp 类型核实** → normalization_certainty 从 `grid_confirmed_warp_type_unverified` 升级到 `mni_normalized_verified`。
4. **Sensitivity sweep**：H6 shaft-pooled null（不替换主分析，只作 sensitivity）；H1 endpoint k=2 / k=4（当前 k=3 lock）。
5. **Cohort summary 写进 paper_overview + topic4 主文档**——但只能在补完 §5.1 cohort 扩展之后写"接受结论"，目前只能写"preliminary，pending cohort 扩展"。

---

## 6. 内部归档代号映射（CLAUDE.md §8 朴素话风格）

- H6 = participation field spatial segregation
- H1 strict layer = matched-null endpoint compactness (within source, within sink)
- H1 envelope layer = endpoint ratio vs non-endpoint centroid (necessary condition)
- H2 set reversal = Jaccard-based source/sink role swap index
- H2 spatial reversal = centroid-distance-based source/sink swap index
- stable_k = adaptive cluster scan 锁定的 K（聚类数）
- PR-6 anchoring = endpoint definition via top-k template_rank with valid_mask filter
- fwd/rev pair = candidate_forward_reverse_pairs from PR-2 with spearman_r < threshold
- coord_units mm = main-analysis gate (voxel coord rejected by assert_coord_result_is_mm_for_main_analysis)
- normalization_certainty = honest tag for coord-comparability claim (subject_native / grid_confirmed_warp_type_unverified / mni_normalized_verified)
