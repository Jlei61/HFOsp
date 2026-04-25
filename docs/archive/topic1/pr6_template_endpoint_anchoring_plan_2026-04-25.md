# PR-6 重启计划：Stable Template Endpoint Anatomical Anchoring

> 状态：plan-of-record，2026-04-25（v2）
> 范围：PR-6 主线从“ictal-onset anchor + 命名”转向“stable template centroid rank 的 endpoint 通道（source ∪ sink）是否解剖锚定病理网络”。
> 替代：原 PR-6-A multi-anchor consensus / ictal alignment 主线全部冻结归档。
> 上游：`docs/topic1_within_event_dynamics.md` §7（PR-6 占位），`docs/topic3_spatial_soz_modulation.md` §7（source/sink stub 已存在）。

---

## 1. Context — 为什么 pivot

旧 PR-6-A 把 “stable interictal template 命名” 绑在 “每次 seizure 之间稳定的 ictal channel-wise anchor” 上。三份 archive 已经把这条路证伪：

- `docs/archive/topic1/pr6a_template_ictal_alignment_plan_2026-04-21.md` — single ictal anchor 主线
- `docs/archive/topic1/pr6a_step0-2_step3preview_review_2026-04-23.md` — Step3 `t_ER_onset` 在 sentinel 上 cross-seizure top10 overlap=0、cross-band ρ=−0.21；正式冻结为 preview-only
- `docs/archive/topic1/pr6a-1.md` — multi-anchor consensus 用 5 个弱 anchor 救一个不稳的 anchor，方法学辩论盖过生物问题
- `docs/archive/topic1/pr6_direction_brainstorm_2026-04-25.md` — Obs 3（identity-bias 86% 来自少数 hub）+ Topic 3 §7 source/sink stub 指向更干净入口

文献：Smith 2022 / Korzeniewska 2014 给 spatial recapitulation 的强先验；Schroeder 2020 / Wenzel 2017 / Pinto 2023 / Bailey 2021 一起说 “稳定 ictal anchor 在领域里已知不 work”。

**论文核心问题**：间期刻板 HFO group event 时序是否刻画/影响癫痫病理网络。
**最直接的检验**：把每个 stable template 的 **rank 极端通道（endpoint = source ∪ sink）** 直接对到 SOZ / focus_rel 标签上，看 endpoint 是否相对 middle 通道显著富集 SOZ。这条路不需要任何 ictal anchor。

---

## 2. 编号与归档决定

- **新主线**：PR-6 重启，主线名 = **PR-6 Stable Template Endpoint Anatomical Anchoring**
- **老 PR-6-A 全部冻结归档**：三份 doc 保留在 `docs/archive/topic1/` 原位，**顶部加 `> SUPERSEDED 2026-04-25` 块**指向新计划，不删除内容
- **brainstorm 文档保留**：`pr6_direction_brainstorm_2026-04-25.md` 保留作为 pivot rationale
- **主文档回写**：`docs/topic1_within_event_dynamics.md` §7 / `docs/topic3_spatial_soz_modulation.md` §7 各加一句话指向新计划

---

## 3. 假设与统计合同（核心）

### 3.1 H1 primary — Endpoint anchoring

> 关键设计：source > sink 不是 H1。Forward/reverse subject 上 SOZ 可能在 source 也可能在 sink，两个方向相消会得到 false negative。**正确主问题是 endpoint vs middle**。

**定义**（per template）：
- `endpoint = source ∪ sink = top-3 smallest rank channels ∪ top-3 largest rank channels`（n_ch ≥ 6 保证不重叠，共 6 个 endpoint）
- `middle = channel_names \ endpoint`（n_ch − 6 个）

**Per-subject 主统计量（subject-level，锁死）**：
```
delta_subject = mean_{k ∈ clusters_of_subject} ( frac_SOZ_endpoint_k − frac_SOZ_middle_k )
```
其中 `frac_SOZ_X = |X ∩ SOZ| / |X|`。对 stable k=2 subject，是两个 template 上的 average；这把“2 templates per subject”的依赖结构正确地折叠进 subject 单元。

**Cohort 主检验**：
- **Pooled (Yuquan + Epilepsiae)** Wilcoxon signed-rank test on `{delta_subject}` against 0
- 二级：sign test（`#subjects with delta > 0` / total），二项检验
- α = 0.05（单一主检验，不分 Bonferroni；dataset-specific 在 §3.5 sensitivity）

**PASS 判据**：Wilcoxon p < 0.05 **AND** sign test p < 0.05 **AND** cohort delta 中位数 > 0。

**H0 / null 接受**：Wilcoxon p > 0.10 → 诚实公开 cohort null：stable template endpoint 不锚定 SOZ。Topic 1 §7 一句话结论改写。论文 framing 转向 “stereotypy independent of clinical SOZ”，仍可发表。

### 3.2 H1b secondary — Polarity (source vs sink)

仅作方向性描述：`frac_SOZ(source)` vs `frac_SOZ(sink)` per template。

**显式声明**：在 8/9 reproduced forward/reverse subject 上，T0 与 T1 的极性相反，per-subject mean over k 会相消。所以这个 secondary 只在 **non-forward/reverse subset**（cohort 减去 forward/reverse 8 个）上跑 paired Wilcoxon，作为 supplementary 描述，**不进 H1 α 池**。

### 3.3 H2 — Forward/reverse 机制 sanity（8/9 subset）

对 PR-2.5 已 reproduce 的 8 个 forward/reverse subject（`time_split_reproducibility.splits.first_half_second_half.forward_reverse_reproduced == True` ∩ `inter_cluster_corr_matrix` 含 < −0.5 entry）：
- `swap_score_subject = mean( Jaccard(source_T0, sink_T1), Jaccard(source_T1, sink_T0) )`
- Permutation null：保留 endpoint set size，从 channel_names 随机抽 3-channel set 做 1000 次重抽，得 swap_null 分布
- 报告：subject-level swap_score vs null；cohort sign test (`#subjects with swap_score > null_95th` / 8)

H2 仅作 mechanism sanity，不进 H1 α 池。判读语言用方向性表达。

### 3.4 H3 — Focus_rel 三级 sensitivity（仅 Epilepsiae）

复用 H1 框架，但把 `frac_SOZ` 拆成三套：
- `frac_i_endpoint − frac_i_middle`（核心病理）→ 主预期：endpoint 富集 i
- `frac_l_endpoint − frac_l_middle`（lesion 解剖结构）→ 次预期：可能 endpoint 富集 l 或不富集
- `frac_e_endpoint − frac_e_middle`（extra-focal）→ negative control：应 ≈ 0

每个 label 独立 cohort Wilcoxon（仅 Epilepsiae subject），三个测试间不分 α，作为 H1 的解剖结构精细化补充。

### 3.5 Sensitivity — Dataset-specific

- **不**作为 co-primary（避免与 §3.1 pooled 主检验矛盾）
- 对 Yuquan、Epilepsiae 各自跑一次 Wilcoxon，仅作 robustness 描述
- 若两个 dataset 方向冲突（一正一负且 |delta_median| > 0.1）→ 报告 dataset divergence，不强行 spin pooled 结果

---

## 4. Cohort 定义（audit-derived，不预写 N，2-tier eligibility）

**关键设计**：source/sink 解剖锚定不需要 PR-5-A retained 作为硬前置。Cohort 从实际数据 audit 出。

**审计输出两个正交 flag（Step 1 review 修订，2026-04-26）**：
- `endpoint_defined`：source ∪ sink 可以提取（`n_ch >= 6`）。可作 case-series 与可视化。
- `h1_primary_eligible`：middle 非空（`n_ch >= 7`），`frac_SOZ_middle` 才有定义。**这是 H1 主检验的硬门槛**。
- `pass = h1_primary_eligible`。`n_ch == 6` 的 subject 标 `endpoint_defined=True / h1_primary_eligible=False / exit_reason='middle_empty'`，**不进 H1 paired Wilcoxon**，但保留在 case-series 报告。

**入选条件检查顺序（首次失败决定 exit_reason）**：
1. 在 `results/interictal_propagation/per_subject/` 下有有效 adaptive cluster JSON
2. `adaptive_cluster.stable_k == 2` → `'k!=2'`
3. SOZ JSON 中有非空 entry → `'empty_soz'`
4. `n_ch >= 6` → `'n_ch<6'`（endpoint 都无法提取）
5. 通道对齐后 `|matched_SOZ_channels| >= 1` → `'no_matched_soz'`
6. centroid rank 有 polarity（任一 cluster `std > 1e-12`） → `'no_polarity'`
7. `n_ch >= 7` → `'middle_empty'`（endpoint_defined=True 但 H1 ineligible）

**Step 2 第一动作**：跑 audit 脚本，输出 `cohort_audit.csv`，一行一个 candidate subject，列出每条入选条件 pass/fail + 退出原因 + 两 flag 值。**绝不预写 cohort size**。审稿人能在 audit 表里独立验证 inclusion logic。

**Case series**：
- k=4 (`818`)、k=6 (`zhangjinhan`)：单独跑同样 endpoint vs middle 表，不进 cohort 推断
- `n_ch == 6` 的 subject（`endpoint_defined=True / h1_primary_eligible=False`）：单独走 case-series，作为 main cohort 的 representativeness caveat
- 因 SOZ empty 退出的 subject：列在 audit 表，作为 cohort representativeness 的 caveat

**Forward/reverse 8/9 subset**：从 audit-derived cohort 内交集挑出，预期 ≤ 8。

---

## 5. Source/Sink / Endpoint / Middle 定义

### 5.1 主定义（唯一进 H1 主检验）

输入：`adaptive_cluster.clusters[k].template_rank`（已是 argsort-of-argsort 整数 rank，`src/interictal_propagation.py:1536`）

```
source_channels = channels with smallest 3 template_rank values
sink_channels   = channels with largest  3 template_rank values
endpoint        = source ∪ sink   (6 channels, n_ch ≥ 6 guarantees disjoint)
middle          = channel_names \ endpoint
```

Top-3 / bottom-3 选择理由：
- 简单可解释，文献对接干净（Smith 2022 propagation onset 通道、Korzeniewska 2014 SdDTF 早期节点）
- 6 个 endpoint 相对 cohort 内典型 n_ch ≈ 10–20，占 30–60%，给 enrichment 留出对比空间
- 不发明新参数，不被审稿人质疑“为什么 N=3”超出“常见 onset 通道数”的范围

### 5.2 Appendix robustness（不进主检验）— Coreness composite

仅作 sensitivity，不并列报告：

```
coreness_ch = (1/(IQR(ranks_ch)+1)) × |median_rank_ch − (n_ch-1)/2| / ((n_ch-1)/2) × mean(bools_ch)
```

按 coreness 取 top 20% 重做 endpoint。若 H1 PASS 在主定义和 coreness 上同向 → archive 一句“两定义同向”；若分歧 → 优先报主定义并讨论。

### 5.3 通道对齐（reuse 已建 helper，不新建）

**复用** `src/event_periodicity.py::match_bipolar_soz`（line 3153）和 `match_bipolar_focus_rel`（line 3164）。两者已实现：
- Normalize（strip + uppercase + 去 `EEG ` 前缀）
- Bipolar `X-Y` split：任一 contact ∈ SOZ set → 'soz'
- Epilepsiae CAR 单 part 名通过同样路径正常工作（split 后只有一个 part）

`align_template_rank_to_soz` helper 内部直接调这两个函数：
```
for ch_name in source_channels:
    label = match_bipolar_soz(ch_name, soz_set)  # or match_bipolar_focus_rel
```

Audit 输出：每个 subject 列出 `unmatched_soz_channels`（在 SOZ JSON 但不在 channel_names 任何 part 中），允许审稿人独立检查匹配率。

---

## 6. Split-half robustness（描述性，不 gate H1，Step 1 review 修订 2026-04-26）

复用 `compute_time_split_reproducibility`（`src/interictal_propagation.py:1734`）。**扩展该函数**：每个 split 内对每个 cluster 存储四个字段：

- `cluster_rank_a` / `cluster_rank_b`：argsort-of-argsort 整数 rank，**仅对 valid（finite）通道排序**；invalid 通道置 `-1` sentinel
- `cluster_valid_mask_a` / `cluster_valid_mask_b`：bool list per cluster，True 表示该 cluster 中该通道至少参与一次（`isfinite(template_row)`）
- `cluster_rank_b_matched_to_a` / `cluster_valid_mask_b_matched_to_a`：按 `mapping_a_to_b` 重排的版本；`matched_b[a_id] == raw_b[mapping_a_to_b[a_id]]`，避免 KMeans label 任意性导致的同 index 跨 cluster 比较错误

**为什么三件都必要（Step 1 review pushback 教训）**：
- 旧实现（`np.where(isfinite, t, nanmean)`）会把非参与通道硬塞成中间 rank，让 robustness 图看起来比真实情况更稳；用 `-1` sentinel 强制 downstream 检 mask
- KMeans 在每个 split 上独立 fit，cluster id 任意；同 index 直接 Jaccard 会跨 cluster 乱配；matched_to_a 版本是审稿人查 robustness 时唯一可用的对齐表
- `extract_endpoint_middle` 的 `valid_mask` 参数把这一约束 push 到 helper 层，`middle` 也只取 valid 通道（不被非参与通道污染）

报告：`Jaccard(source_split_a, source_split_b_matched)` 中位数 + cohort 分布（**只在两边都 valid 的通道上算**）；同样对 endpoint 整体集合做 Jaccard。

**不**用作 H1 gating（H1 仍在 full-data centroid rank 上跑，无 double-dipping 因为 SOZ 标签是外部独立信号）。仅作 “endpoint 通道在时间上是否稳定” 的 robustness 描述。

---

## 7. 代码改动（最小化 + 大量复用）

### 7.1 修改：`src/interictal_propagation.py`
- **扩展** `compute_time_split_reproducibility`（line 1801–1850）：每个 split 的 `split_results[split_name]` 追加 `cluster_rank_a: list[list[int]]` 和 `cluster_rank_b: list[list[int]]`（约 +20 行）

### 7.2 新增：`src/template_anatomical_anchoring.py`
独立模块，纯统计层，导入 `match_bipolar_soz` / `match_bipolar_focus_rel` 复用：

```python
def extract_endpoint_middle(channel_names, template_rank, n=3):
    """Return {'source': [...], 'sink': [...], 'endpoint': [...], 'middle': [...]}."""

def compute_template_anchoring(
    channel_names, template_rank, soz_channels, focus_rel_dict=None
):
    """One template -> {
        'frac_SOZ_endpoint', 'frac_SOZ_middle', 'frac_SOZ_source', 'frac_SOZ_sink',
        ['frac_i_endpoint', 'frac_l_endpoint', 'frac_e_endpoint', ... if focus_rel],
        'matched_soz_channels', 'unmatched_soz_channels',
    }."""

def compute_subject_delta(per_template_records):
    """Average over k templates: {'delta_endpoint_vs_middle', 'delta_source_vs_sink', ...}."""

def cohort_wilcoxon(deltas, alternative="greater"): ...
def cohort_sign_test(deltas): ...

def forward_reverse_swap_check(
    t0_source, t0_sink, t1_source, t1_sink, channel_names, n_perm=1000
):
    """Return {'swap_score', 'null_p', 'null_95th', ...}."""

def compute_template_coreness(ranks, bools, labels, n_clusters):
    """Sensitivity, appendix."""
```

### 7.3 新增：`scripts/run_pr6_template_anchoring.py`
- `--audit`：跑 cohort audit，输出 `cohort_audit.csv`
- `--per-subject`：对 audit-pass subject 跑 endpoint/middle 计算，输出 `results/interictal_propagation/template_anchoring/per_subject/<subject>.json`
- `--cohort`：H1 / H2 / H3 cohort 统计 + dataset-specific sensitivity，输出 `cohort_summary.json`
- `--coreness`（optional flag）：跑 sensitivity coreness 平行表

### 7.4 新增：`scripts/plot_pr6_template_anchoring.py`
- 主图 1：cohort `delta_subject` paired scatter（endpoint vs middle SOZ frac），对角线 + Wilcoxon p
- 主图 2：H3 三联（i / l / e × Epilepsiae cohort）
- 主图 3：H2 8-subset T0/T1 source/sink × SOZ heatmap（per-subject）
- Appendix：H1b polarity scatter（non-forward/reverse subset），coreness 平行 scatter
- 输出 `results/interictal_propagation/template_anchoring/figures/` + `README.md`（中文，AGENTS.md 规范）

### 7.5 不动
- `compute_adaptive_cluster_stereotypy`、`build_cluster_templates`、`assign_events_to_templates` 完全不动
- 所有 PR-2 / PR-2.5 / PR-3 / PR-4A/B / PR-5-A 输出完全不重跑
- 不引入任何 ER / CUSUM / Page-Hinkley / ictal annotation 代码

---

## 8. Step 拆分

| Step | 名称 | 交付 | 预算 |
|---|---|---|---|
| **PR-6 Step 0** | 老文档冻结 + 主文档 §7 回写 | 三份老 doc 顶部加 SUPERSEDED 块；topic1 §7 / topic3 §7 一句话回写 | 0.5 d |
| **PR-6 Step 1** | 代码层 + TDD（§9） | `extract_endpoint_middle` / `compute_template_anchoring` / split-half 扩展 / `forward_reverse_swap_check`；T1–T8 全绿 | 1 d |
| **PR-6 Step 2** | Cohort audit + per-subject 跑 | `cohort_audit.csv` + audit-pass subject 全部 per-subject JSON + audit 表 review | 0.5 d |
| **PR-6 Step 3** | H1 cohort + H1b polarity + 主图 | pooled Wilcoxon + sign test + non-forward/reverse polarity；主图 1 + appendix polarity | 0.5 d |
| **PR-6 Step 4** | H2 forward/reverse swap + 8-subset 图 | swap_score + permutation null + 8-subset 4-cell heatmap | 0.5 d |
| **PR-6 Step 5** | H3 i/l/e + dataset-specific sensitivity + coreness appendix | Epilepsiae 三联 + Yuquan/Epilepsiae 各 Wilcoxon + coreness 平行表 | 0.5 d |
| **PR-6 Step 6** | 归档 + 主文档结论 | `docs/archive/topic1/pr6_template_anchoring_results_2026-05-xx.md` + topic1 §7 一句话结论 + topic3 §7 一句话结论 | 0.5 d |
| **总预算** | | | **4.0 d** |

---

## 9. TDD 测试合同（锁 8 项）

测试文件 `tests/test_pr6_template_anchoring.py`：

```
T1. test_extract_endpoint_middle_basic:
    channel_names=['A','B','C','D','E','F','G','H']
    template_rank=[7,6,5,4,3,2,1,0]   # H 最早，A 最晚
    -> source=['H','G','F'], sink=['A','B','C']
    -> endpoint={'A','B','C','F','G','H'}, middle=['D','E']

T2. test_extract_endpoint_middle_min_n_ch:
    n_ch=6: endpoint=全部 6 个, middle=[]; compute_subject_delta 必须能 graceful 处理 frac_middle 未定义
    n_ch=5: helper 返回 exit_reason='n_ch<6'

T3. test_align_yuquan_bipolar:
    channel_names=['D13-D14','D14-D15','E1-E2','F1-F2','F2-F3','F3-F4']
    soz=['D13','D14','E1','E2']
    复用 match_bipolar_soz -> 'D13-D14' 与 'D14-D15' 与 'E1-E2' = SOZ
    frac_SOZ_endpoint / frac_SOZ_middle 与手算一致

T4. test_align_epilepsiae_focus_rel_3level:
    focus_rel={'i':['HRA1'], 'l':['BFLA1','BFLA2'], 'e':[]}
    channel_names=['HRA1','HRB1','BFLA1','BFLA2','BFLA3','BFLA4']
    复用 match_bipolar_focus_rel
    -> frac_i / frac_l / frac_e 与手算一致

T5. test_subject_delta_averages_over_k:
    构造 2 templates:
      T0: frac_SOZ_endpoint=0.5, frac_SOZ_middle=0.1
      T1: frac_SOZ_endpoint=0.3, frac_SOZ_middle=0.2
    -> delta_subject = mean(0.4, 0.1) = 0.25
    断言 delta 是 subject-level scalar，不是 template-level

T6. test_split_half_centroid_rank_storage:
    跑 compute_time_split_reproducibility，断言
    splits['first_half_second_half']['cluster_rank_a'] / ['cluster_rank_b'] 都存在
    且 shape=(k, n_ch)

T7. test_forward_reverse_swap_score:
    T0_source=['A','B','C'], T1_sink=['B','C','A']
    T0_sink=['F','G','H'], T1_source=['G','H','F']
    -> swap_score Jaccard 都 = 1.0
    permutation null (n_perm=200) -> p < 0.05

T8. test_cohort_exit_audit:
    构造 5 个 candidate subject:
    - A: stable_k=2, soz=['A','B'], n_ch=10 -> PASS
    - B: stable_k=4 -> EXIT 'k!=2'
    - C: stable_k=2, soz=[] -> EXIT 'empty_soz'
    - D: stable_k=2, soz=['Z'], n_ch=10 (no match) -> EXIT 'no_matched_soz'
    - E: stable_k=2, soz=['A'], n_ch=5 -> EXIT 'n_ch<6'
    audit 表 5 行，PASS 列只 A=True
```

---

## 10. 失败合同

| 触发 | 响应 |
|---|---|
| H1 pooled Wilcoxon p > 0.10 且 sign test p > 0.10 | **诚实公开 cohort null**：endpoint 不锚定 SOZ。Topic 1 §7 一句话结论：“stable template endpoint 不显著富集 SOZ”。论文 framing 转向 “interictal stereotypy reflects network organization independent of clinical SOZ annotation”，仍可发表 |
| H1 pooled PASS 但 dataset-specific 一正一负 (`|median delta| > 0.1`) | 报告 dataset divergence，讨论可能原因（Yuquan 临床标注 vs Epilepsiae focus_rel 标注差异；电极覆盖密度），不强行 spin pooled |
| H1 PASS 但 H3 i/l/e 全部不显著 | 报告 “endpoint 富集 SOZ 但不分焦点级别”，讨论 SOZ 二值化损失 |
| H1 PASS 但 H1b polarity (non-forward/reverse subset) Wilcoxon p > 0.05 | endpoint 锚定但 source vs sink 无系统方向，可能 stereotypy 是双向行波双方向各占一半；记录为 mechanism-uninformative，不影响 H1 |
| H2 swap_score 8-subset sign test 不显著 | forward/reverse 几何不构成 “双方向行波 swap” 的强证据；H2 仅描述，不影响 H1 |
| Coreness 与主定义反向 | 报告 “endpoint 定义不 robust”，archive 加 caveat |
| Audit-derived cohort `n < 10` | 报告 underpowered；优先扩 Yuquan SOZ JSON 完整性，其次 Epilepsiae focus_rel coverage |
| Split-half source 通道 Jaccard 中位数 < 0.4 | 报告 “endpoint 通道时间稳定性不足”，作为 H1 解读 caveat |

---

## 11. 验证方法（end-to-end）

1. `pytest tests/test_pr6_template_anchoring.py -v` — T1–T8 全绿
2. `python scripts/run_pr6_template_anchoring.py --audit` — 输出 `cohort_audit.csv`，手动检查每条入选 / 退出原因
3. `python scripts/run_pr6_template_anchoring.py --per-subject` — audit-pass subject 全跑
4. `python scripts/run_pr6_template_anchoring.py --cohort` — H1/H2/H3 + sensitivity，输出 `cohort_summary.json`
5. `python scripts/plot_pr6_template_anchoring.py --all` — 主图 1/2/3 + appendix + figures/README.md
6. **手动 audit**：抽 2 个 subject（一 Yuquan，一 Epilepsiae）：
   - 打印 channel_names、template_rank、source/sink/endpoint/middle 列表
   - 打印 SOZ list、matched/unmatched 通道
   - 用计算器手算 frac_SOZ_endpoint / frac_SOZ_middle，与脚本输出对一致
7. 写 archive：`docs/archive/topic1/pr6_template_anchoring_results_2026-05-xx.md`，回写 `docs/topic1_within_event_dynamics.md` §7 + `docs/topic3_spatial_soz_modulation.md` §7

---

## 12. 显式不做的事（防止漂移）

- 不重启任何 ictal anchor / ER / CUSUM / Page-Hinkley pipeline
- 不引入 multi-anchor consensus / voting / naming label
- 不重跑 PR-2 / PR-2.5 / PR-3 / PR-4A/B / PR-5-A
- 不做 π embedding（brainstorm PR-6-B），作为后续独立 PR 与本计划解耦
- 不在同一份数据上先挑 hub 再重跑 PR-4C（避免 brainstorm Obs 3 警告的 double-dipping）— 留给独立 “PR-6 后续 held-out replay” 计划
- 不预写 cohort size — 必须从 §4 audit 推导

---

## 13. Critical files to touch

| 文件 | 动作 |
|---|---|
| `src/interictal_propagation.py` | 扩展 `compute_time_split_reproducibility` (~line 1801–1850) 存 split-level `cluster_rank_a/b` |
| `src/template_anatomical_anchoring.py` | 新建（统计层 + endpoint/middle helper + swap check） |
| `scripts/run_pr6_template_anchoring.py` | 新建，支持 `--audit / --per-subject / --cohort / --coreness` |
| `scripts/plot_pr6_template_anchoring.py` | 新建，输出主图 + figures/README.md |
| `tests/test_pr6_template_anchoring.py` | 新建（T1–T8） |
| `docs/archive/topic1/pr6a-1.md` | 顶部加 SUPERSEDED 块 |
| `docs/archive/topic1/pr6a_template_ictal_alignment_plan_2026-04-21.md` | 顶部加 SUPERSEDED 块 |
| `docs/archive/topic1/pr6a_step0-2_step3preview_review_2026-04-23.md` | 顶部加 SUPERSEDED 块 |
| `docs/topic1_within_event_dynamics.md` §7 | PR-6 条目改写为新主线一句话 + 链接 |
| `docs/topic3_spatial_soz_modulation.md` §7 | source/sink stub 改写为 “正式实现于 PR-6 新计划” |
| `results/interictal_propagation/template_anchoring/` | 新建（含 `cohort_audit.csv` / `per_subject/` / `figures/README.md` / `cohort_summary.json`） |

---

## 14. 一句话核心

PR-6 的论文核心问题是“间期刻板 HFO 时序是否刻画病理网络”。最干净的检验方式是 stable template centroid rank 的 **endpoint (source ∪ sink) vs middle** 的 SOZ 富集差异，subject-level delta + cohort Wilcoxon。Forward/reverse subject 的 polarity 抵消问题被 endpoint 框架自动消除；source vs sink 极性只作为方向性 secondary。Pass 是 Topic 1 × Topic 3 的 capstone；Null 也是干净可发表的结果。复用 `match_bipolar_soz` / `match_bipolar_focus_rel` 让通道对齐不脆。Cohort 从 audit 推导，不预写 N。

---

## 15. Step 完成记录

### Step 1 — 代码层 + TDD（2026-04-26 ACCEPTED）

**交付**：
- `src/template_anatomical_anchoring.py`（新建，~370 行）：`extract_endpoint_middle` / `compute_template_anchoring` / `compute_subject_delta` / `cohort_wilcoxon` / `cohort_sign_test` / `forward_reverse_swap_check` / `audit_subject_eligibility`
- `src/interictal_propagation.py`：`compute_time_split_reproducibility` 扩展为存储 6 个 PR-6 robustness 字段（`cluster_rank_a/b`、`cluster_valid_mask_a/b`、`cluster_rank_b_matched_to_a`、`cluster_valid_mask_b_matched_to_a`）
- `tests/test_pr6_template_anchoring.py`（新建，13 项测试 — 覆盖 §9 锁定的 T1–T8 全部 + 5 项 review 后追加的 edge-case 测试）

**测试结果**：13 PR-6 + 49 既有 interictal_propagation = **62/62 全绿**

**Step 1 review 触发的合同修订**（用户 2026-04-26 pushback）：

1. **Audit 拆为 2-tier eligibility**（§4 修订）
   - 旧：`pass = (n_ch >= 6) AND ...`
   - 新：`endpoint_defined`（n_ch ≥ 6）vs `h1_primary_eligible`（n_ch ≥ 7），`pass = h1_primary_eligible`
   - **触发原因**：`n_ch == 6` 时 endpoint 覆盖全部通道、`middle = []`、`frac_SOZ_middle = NaN`，`_safe_mean` 跳 NaN 会让 subject 通过 audit 但悄悄不进 H1 delta，cohort 大小被偷偷缩水
   - **新行为**：n_ch=6 标记为 `exit_reason='middle_empty'`，进 case-series 不进 H1 paired Wilcoxon

2. **Split-half cluster_rank_b 按 mapping 对齐**（§6 修订）
   - 旧：仅存 `cluster_rank_a/b`；KMeans label 任意，同 index 直接 Jaccard 跨 cluster 乱配
   - 新：额外存 `cluster_rank_b_matched_to_a`（按 `mapping_a_to_b` 重排）和 `cluster_valid_mask_b_matched_to_a`
   - **触发原因**：同 index 跨 split 比较是 robustness 图最容易被审稿人挑出的 bug

3. **Split-half NaN 处理改 `-1` sentinel + valid_mask**（§6 修订）
   - 旧：`np.where(isfinite, t, nanmean)` 把非参与通道塞中间 rank
   - 新：仅对 finite 通道排序，invalid 置 `-1`；`extract_endpoint_middle` / `compute_template_anchoring` 接受 `valid_mask`，`middle` 也只取 valid 通道
   - **触发原因**：旧实现会让非参与通道获得 mid-rank，artificially 抬高 robustness 稳定性

**关键设计语义锁死（Step 2+ 必须遵守）**：
- Audit 表必须含 `endpoint_defined` 与 `h1_primary_eligible` 双 flag
- Split-half 任何 Jaccard 计算用 `cluster_rank_b_matched_to_a` + `cluster_valid_mask_*`，**禁止**对 raw `cluster_rank_b` 做同 index 比较
- `extract_endpoint_middle` 默认把 rank `< 0` 视为 invalid（split-half sentinel 兼容），调用方可显式传 `valid_mask` 覆盖

**Step 2 启动条件**：本 ACCEPTED 即解锁。下一步交付 `scripts/run_pr6_template_anchoring.py` 的 `--audit / --per-subject / --cohort` 三个 mode + `cohort_audit.csv`。

### Step 2 — Cohort audit + per-subject + cohort statistics（2026-04-26 跑完）

**交付**：
- `scripts/run_pr6_template_anchoring.py`（新建 ~470 行）：`--audit / --per-subject / --cohort / --all` 四个 mode；复用 §7.2 helper；Yuquan/Epilepsiae 数据集自动分流；reuses `match_bipolar_soz` / `match_bipolar_focus_rel`
- `results/interictal_propagation/template_anchoring/cohort_audit.csv`（30 subject 候选，13 字段）
- `results/interictal_propagation/template_anchoring/per_subject/<dataset>_<subject>.json`（23 个 endpoint_defined subject）
- `results/interictal_propagation/template_anchoring/cohort_summary.json`（H1/H1b/H2/H3 + dataset-specific sensitivity）

**Cohort audit 结果**（30 candidate subject）：

| 类别 | n | 说明 |
|---|---|---|
| H1-eligible（pass=h1_primary_eligible） | **21** | 13 epilepsiae + 8 yuquan，进 H1 主检验 |
| Endpoint-defined-only（n_ch=6，case-series） | 2 | epilepsiae 1073 / 1077，标 `middle_empty` |
| EXIT `empty_soz` | 4 | epilepsiae 1125 / 384 / 620 / 916（SOZ JSON 缺失）|
| EXIT `k!=2` | 3 | epilepsiae 818（k=4）、yuquan huangwanling（k=4）、yuquan zhangjinhan（k=6）|
| EXIT `n_ch<6` / `no_matched_soz` / `no_polarity` | 0 | 全无 |

**H2 forward/reverse 子集 audit-derived n=5**（不是 PR-2.5 8/9，因 SOZ 缺失 + n_ch=6 限制）：epilepsiae 1073 / 139 / 635 / 958 + yuquan chenziyang。

**Cohort 统计结果（pre-Step 5 sensitivity）**：

| 检验 | n | median | p-value（greater）| sign-test | 判读 |
|---|---|---|---|---|---|
| **H1 pooled** | 21 | 0.0000 | **0.42** | 9pos/8neg/4tie，p=1.0 | **NULL（plan §10 第一行触发）**|
| H1 dataset Yuquan | 8 | +0.098 | 0.38 | 4pos/2neg | NULL，方向略偏正 |
| H1 dataset Epilepsiae | 13 | 0.000 | 0.54 | 5pos/6neg | NULL |
| H1b polarity (non-fwdrev, n=17) | 17 | 0.0 (two-sided) | 0.62 | 4pos/4neg | NULL（plan §10 第四行触发，mechanism-uninformative）|
| **H2 forward/reverse swap** | 5 | swap_score 0.50–1.00 | **5/5 exceed null_95th** | per-subject null_p ∈ [0.000, 0.050] | **方向性强信号** |
| H3 focus_rel `i` | 10 | 0.0 | 0.50 | 4pos/4neg | NULL |
| H3 focus_rel `l` | 10 | 0.0 | 0.67 | 1pos/1neg/8tie | NULL |
| H3 focus_rel `e` | 10 | 0.0 | 0.40 | 4pos/3neg | NULL |

**初步科学口径（待 Step 5 sensitivity / Step 6 archive 之前不写主文档）**：

1. **H1 cohort-level NULL 已成立**：stable template endpoint vs middle 在 cohort 上不显著富集 SOZ。Plan §10 “诚实公开 cohort null” 的 framing 触发：论文叙事将转向 “interictal stereotypy reflects network organization independent of clinical SOZ annotation”。
2. **H2 5/5 swap above null_95th 是个独立、可发表的 mechanism finding**：在 forward/reverse reproduced subjects 上 T0 source ↔ T1 sink 的几何 swap 是真实的（per-subject null_p ≤ 0.05），与 Smith 2022 bidirectional traveling wave prediction 完全对应。但 cohort 功效极有限（n=5 binomial 5/5 = p=0.0625），按 plan §3.3 H2 仅作 mechanism sanity，不作 cohort-level 主结论。
3. **H1b mechanism-uninformative**：non-forward/reverse subset polarity (source vs sink) 也是 null（4/4），与 plan §10 第四行预测一致 — 表示 stable templates 即使在 non-fwdrev 子集上也是双向 / 无系统极化。
4. **H3 三级 focus_rel 全 null** 印证了 H1 — endpoint 富集既不出现在 SOZ 严格定义（i）也不出现在 lesion（l）或 extra-focal（e），与 Topic 1 “stereotypy 不依赖 clinical SOZ annotation” 叙事一致。

**Step 2 验收标准**（plan §11 #1–#4）：
- ✅ 13 PR-6 + 49 既有测试 = 62/62 全绿
- ✅ `cohort_audit.csv` 落盘，2-tier flag 正常工作
- ✅ 23 per-subject JSON 落盘（21 H1 + 2 case-series）
- ✅ `cohort_summary.json` 落盘，含 H1/H1b/H2/H3 + dataset-specific
- ⏳ Step 3 plotting 仍待补（主图 1/2/3 + figures/README.md）
- ⏳ Step 5 coreness sensitivity + split-half robustness Jaccard 仍待补
- ⏳ Step 6 archive results doc + 主文档结论回写

**Step 3 启动条件**：plotting 与 Step 5 sensitivity 都基于本 Step 2 落盘的 cohort_summary.json，可并行启动。
