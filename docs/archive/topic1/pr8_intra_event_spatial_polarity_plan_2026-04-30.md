# PR-8 计划：Intra-Event Spatial Polarity（SOZ-first / SOZ-last）

> **状态：DEFERRED pending PR-T3-1**（2026-04-30 决定）。原因：本 PR 核心因变量"SOZ-first / SOZ-last"完全依赖 SOZ 标签质量；在 `docs/archive/topic3/pr_t3_1_data_driven_soz_audit_plan_2026-04-30.md`（data-driven ictal-onset SOZ audit）完成、产出 `results/spatial_modulation/data_driven_soz/<dataset>_<subject>.json` 之前，本 plan 的 H1 / H1' 检验在 clinical SOZ 单一定义下都不具备 publishable 价值。
>
> v1（本文件）保留**作为历史记录**，**不执行**。PR-T3-1 完成后回写 v2，把 §3.4 H2 sensitivity 重构为 multi-source SOZ 协同检验（clinical + M1 + M2 并列报告），并合并 §16 列出的 9 条修订点。
>
> v1 范围：在 PR-2.5 forward/reverse-reproduced cohort 上，以**单 event 内的空间结构**（参与通道的相对 lag）为对象，检验 forward / reverse template 在 SOZ-内外是否呈极性。这是 Ping-Pong 假说 L4 层（intra-event 空间极性）的可证伪检验，**不**承担机制层（兴奋-抑制）论断，**不**承担 PR-7 已封的短窗时间耦合层。
> 上游：`docs/archive/topic1/ping_pong_hypothesis_review_2026-04-28.md` §11.4 + §12.4；`docs/archive/topic1/pr7_template_pairing_results_2026-04-29.md` §17（locked NULL）；PR-2.5 forward/reverse 8/9；PR-6 endpoint anchoring per-template source/sink。
> 下游：本 PR 只验证 L4 空间极性。无论 PASS / NULL，均**不**回开 PR-7 已封的 L2 时间耦合层。

---

## 1. Context — 为什么这是下一个 PR

PR-7 在 stable_k=2 forward/reverse cohort 上完成了 H1 NULL（cohort excess(10s) Wilcoxon p > 0.10 across N1/N2/N3），并且 N2 window sweep + burst-level 复算均没有把 NULL 翻过来。结论锁定为：在已测试的时间尺度（1s–1h）上 forward/reverse template 的事件流**与 mark-independent sampling 一致**。

但 PR-7 的检验对象只是 event timestamp + cluster 标签，**完全没用到** event 内部的通道级 lag 结构。用户原始直觉的 L4 层（"SOZ 兴奋诱发 inhibitory wall 反弹"）的最直接可证伪签名应当出现在 **单 event 内空间极性**：

- forward template 是不是把 SOZ 当作"早出现"的源头；
- reverse template 是不是把 SOZ 当作"晚出现"的尾巴；
- 两者是否表现出 polarity reversal（一个负、一个正）。

PR-7 NULL **不**包含这一层。本 PR 直接用 `lagPatRaw` 检验该极性是否存在。

**重要边界**：HFO 80–250 Hz 不区分 E/I。即使 PR-8 PASS，结论也只是"forward/reverse template 在空间极性上呈反向"，**不**等价于"兴奋-抑制反弹"。机制层归因留给将来 LFP / 单元活动的并行实验。

---

## 2. 编号与归档决定

- **新 PR**：PR-8 = **Intra-Event Spatial Polarity**
- **不打包**：history-dependent marked point process（review §12.1）/ peri-seizure template ratio（§12.2）/ subject 548 single-subject case-study（§12.3）**不**进 PR-8
- **范围声明（写死）**：本 PR 只检验 L4 空间极性，**不**包含 L2 时间耦合（已封）/ L3 长时动力学（未来 PR）/ 任何机制层论断
- **主文档回写**：完成后在 `docs/topic1_within_event_dynamics.md` §7 加一行 PR-8 verdict + 归档链接

---

## 3. 假设与统计合同（核心）

### 3.1 数据对象

每个 forward/reverse-reproduced subject 的输入：

- `lag_pat_raw[ch, e]: np.ndarray[float]` — 来自 `*_lagPat_withFreqCent.npz`（**10ch 全集**，与 PR-2 cluster JSON 对齐；**禁止**用 `*_lagPat.npz` 7ch 旧切片）
- `bools[ch, e]: np.ndarray[bool]` — 同 npz 内的参与通道掩码（True = 该 event 该通道参与）
- `cluster_labels: np.ndarray[int, N]` — `adaptive_cluster.labels`（PR-2 per-subject JSON）
- `channel_names: list[str]` — npz / PR-2 JSON / PR-6 JSON 三方共享的通道顺序
- `soz_channels: set[str]` — 来自 `results/{epilepsiae,yuquan}_soz_core_channels.json`
- `per_template_anchoring`：PR-6 `template_anchoring/per_subject/<subject>.json` 的 `per_template[k]` 含 `cluster_id / source / sink / frac_SOZ_source / frac_SOZ_sink / valid_mask`

**频道顺序合同**：`channel_names`、`bools` 行序、`lag_pat_raw` 行序、`per_template[k].valid_mask` 必须严格按同一通道顺序。代码必须先验证再用，不一致则 `raise ValueError`，**禁止**默认对齐。

### 3.2 主指标 — `Δ_event` 与 `median_Δ_template`

对每个事件 `e`：

```
participating_e = { ch : bools[ch, e] == True }
SOZ_p           = participating_e ∩ SOZ
nonSOZ_p        = participating_e ∩ (channels \ SOZ)

eligible_e      = (|SOZ_p| ≥ 1) ∧ (|nonSOZ_p| ≥ 1) ∧ (|participating_e| ≥ 3)

if not eligible_e:
    Δ_event(e) = NaN          # do NOT impute 0; do NOT include in median

# event-internal anchor at 0 over participating channels only:
ref_e          = min_{ch ∈ participating_e} lag_pat_raw[ch, e]
rel_lag[ch, e] = lag_pat_raw[ch, e] − ref_e   # only defined for ch ∈ participating_e

Δ_event(e)     = mean_{ch ∈ SOZ_p}    rel_lag[ch, e]
               − mean_{ch ∈ nonSOZ_p} rel_lag[ch, e]
```

约定：`Δ_event(e) < 0` ⇒ 该 event 中 SOZ 通道平均比 nonSOZ 通道**更早**到达；`Δ_event(e) > 0` ⇒ SOZ 比 nonSOZ **更晚**到达。

Per-template 聚合（subject-level）：

```
events_in_template_k = { e : cluster_labels[e] == k ∧ eligible_e }
median_Δ_k           = median_{e ∈ events_in_template_k} Δ_event(e)
n_eligible_k         = |events_in_template_k|
```

如果 `n_eligible_k < N_MIN_TEMPLATE`（默认 30），该 template 退出 H1 / H1' 主检验，归入 audit appendix。

### 3.3 Template 极性指派（Ta / Tb）

PR-6 anchoring 已经给出每个 cluster 的 `frac_SOZ_source`（source 端通道中 SOZ 的占比）。**指派规则**：

```
Ta = argmax_k frac_SOZ_source[k]      # SOZ-leading 候选
Tb = argmin_k frac_SOZ_source[k]      # SOZ-trailing 候选
```

排除条件（subject 退出 H1 directional 检验）：

- 两个 cluster 的 `frac_SOZ_source` 完全相等（Δ ≤ 1e-9）⇒ 无法指派
- 至少一个 cluster `n_valid` （PR-6 endpoint 通道数）< 6 ⇒ endpoint 不可信
- subject 不属于 PR-2.5 forward/reverse-reproduced（OR over `first_half_second_half` / `odd_even_block`，参考 CLAUDE.md cross-PR contract lookups）

排除的 subject 仍进入 H1' polarity_reversal flag 统计（不依赖 Ta/Tb 指派）。

### 3.4 Hypothesis tier（pre-registered）

| Tier | 名称 | 假设 | 统计量 | 拒绝条件 |
|---|---|---|---|---|
| **H1 primary** | Directional polarity | forward template SOZ-first（Δ_Ta < 0），reverse template SOZ-last（Δ_Tb > 0），且差异显著 | Subject-level Wilcoxon signed-rank `(Δ_Tb − Δ_Ta)` > 0 (one-sided) | p < 0.05 + sign test p < 0.05 + 中位数 (Δ_Tb − Δ_Ta) > 0 |
| **H1' sanity** | Polarity reversal flag | sign(Δ_Ta) ≠ sign(Δ_Tb) 的 subject 比例显著 > 0.5 | Binomial test on flag fraction, two-sided | p < 0.05 |
| **H2 sensitivity** | Alt SOZ definitions | 用 per-seizure onset 通道 / PR-1 ER-leading 通道替代 SOZ JSON 后 H1 / H1' 不翻号 | 重跑 H1 + H1'，比较 (Δ_Tb − Δ_Ta) 中位数符号 | 中位数符号一致即通过 |

**严禁升级 H1' 为 primary**——它只是 polarity 存在性的方向无关 sanity。
**严禁升级 H2 为 mechanism claim**——它只是 SOZ 标签鲁棒性 sensitivity。

### 3.5 PASS / NULL / FAIL 判据

| 条件 | 解读 |
|---|---|
| H1 primary triple gate 全部满足 | **PASS**：forward/reverse template 在空间极性上呈 SOZ-first / SOZ-last 反向；可以写入 topic1 §7。机制层仍**不**得论断 |
| H1 primary Wilcoxon p > 0.10 但 H1' polarity_reversal flag 显著 | **partial**：极性反转存在但方向不一致（不一定 SOZ-first vs SOZ-last）；只在 archive，不进 main doc |
| H1 + H1' 都 NULL | **L4 NULL**：写入 archive；topic1 §7 只加一行"PR-8 NULL，L4 空间极性签名亦不成立"；**严禁**写"机制不存在"——SOZ 标注是粗标签，且 HFO 80–250 Hz 不区分 E/I |
| Cohort 可用 subject < 5 (H1 directional) 且 < 6 (H1') | **abort**：报告样本量不足，归 archive，不出主文档结论 |

### 3.6 禁止性表述（写入 plan）

主文档 / archive results 中**严禁**出现以下措辞，无论 verdict：

- "证明 forward template 是兴奋驱动 / reverse template 是抑制反弹"——HFO 频段无法区分 E/I
- "PR-8 NULL 说明 Ping-Pong 机制不存在"——SOZ 标注是粗标签；NULL 只覆盖"以 SOZ JSON 定义的 SOZ-内外极性"
- "polarity_reversal 比例显著 > 0.5 即支持 Ping-Pong"——polarity_reversal 不依赖方向，是 H1' sanity，不能替代 H1 directional
- "subject XXX 个体显著即支持 cohort 假说"——cohort claim 必须 cohort-level；subject-level 极端值只入 case-series，不进 §7
- "PR-8 PASS 闭环了 ping-pong 全部假说"——L1/L2/L3 都没合上；只能 claim L4 空间极性

允许的表述（PASS）：

- "在 PR-2.5 forward/reverse cohort 上，PR-6 anchoring-assigned forward template 表现为 SOZ-leading（Δ_Ta < 0），reverse template 表现为 SOZ-trailing（Δ_Tb > 0），subject-level Wilcoxon p = X"
- "Polarity reversal flag rate = Y/N，binomial p = Z（H1' sanity）"
- "用 per-seizure onset 通道复测后 directional 中位数符号一致（H2 sensitivity）"

允许的表述（NULL）：

- "在 PR-2.5 forward/reverse cohort 上，未观察到 forward/reverse template 在 SOZ-内外空间极性上的显著差异（Wilcoxon p = X，n = Y）；该结果在替换 SOZ 定义后保持一致"
- "L4 层（intra-event spatial polarity）签名不成立；机制层归因仍开放，需 LFP / 单元活动并行实验"

---

## 4. Surrogate / null hierarchy

主统计是 subject-level Wilcoxon，**不**需要 cohort-level permutation。但 per-subject 内部需要一个 sanity null 来回答"个体水平上 |median_Δ_Ta − median_Δ_Tb| 是否大于 cluster 标签随机化的基线"。

### 4.1 N1 — per-subject label shuffle（per-subject sanity）

```
For perm in 1..1000:
    label_perm = random permutation of cluster_labels (preserving counts)
    median_Δ_k_perm = median Δ_event over events with label_perm == k
    diff_perm       = median_Δ_Ta_perm − median_Δ_Tb_perm
record diff_obs vs distribution of diff_perm
report two-sided p_perm
```

**用途**：per-subject 表格里加一列 `n1_p_perm`，仅供解读单 subject 是否极端，**不**进 cohort PASS / NULL 判据。

### 4.2 N2 — circular shift on cluster_labels with block-aware boundary（备选 sanity）

如果 N1 多数 subject p_perm < 0.05 但 H1 cohort NULL，需要 N2 排除"cluster 标签与 event 序号有时间相关"导致的 block-aware 偏差。N2 在 block 内做 circular shift 而非全局打乱。

**默认不跑**；仅在 N1 出现 cohort 不一致时启用。

### 4.3 不需要 ISI 重采样

PR-8 是空间极性测试，不涉及 inter-event interval；不构造 N3 / N4。`resample_isi_per_cluster` 等帮手在本 PR 不出现，**不**复用 PR-7 已 raise 的 stub。

---

## 5. Eligibility / Failure / Abort Modes

### 5.1 Subject 级 eligibility

| 条件 | 出 | 进 H1 | 进 H1' |
|---|---|---|---|
| 不在 PR-2.5 fwd/rev cohort（OR rule） | drop | ✗ | ✗ |
| stable_k ≠ 2 | drop | ✗ | ✗ |
| 任一 cluster `n_eligible_k < 30` | flag low-n | ✗ | ✗ |
| 任一 cluster PR-6 `n_valid` < 6 | flag low-n_endpoint | ✗ | ✗ |
| `frac_SOZ_source[Ta]` − `frac_SOZ_source[Tb]` ≤ 1e-9 | tie | ✗（无法指派） | ✓ |
| 上述全部通过 | OK | ✓ | ✓ |

### 5.2 Cohort failure → abort

如发生以下任一，写 abort 报告，不出 §7 conclusion：

1. H1 cohort N（passing eligibility）< 5
2. H1' cohort N（含 tie subject）< 6
3. 全 cohort 上 event eligibility rate（per-subject 中位数）< 50% — SOZ JSON 与 participating set 切片太狠，单 event 几乎全是 SOZ 或全是 nonSOZ
4. Ta/Tb 之间 `n_eligible` 差异中位数 > 3× — 指派引入了样本量不平衡 confound

### 5.3 Confound profile

| Profile | 触发条件 | 解读 |
|---|---|---|
| A | H1 PASS 但仅 1–2 个 subject 主导 | 单 subject driven，不 cohort claim；只入 case-series |
| B | H1 NULL 且 |median(Δ_Ta)| 与 |median(Δ_Tb)| 都接近 0（< 0.1× event-level σ_Δ） | 两 template 在 SOZ-内外极性上都没结构 |
| C | H1 NULL 但 H1' PASS | polarity reverse 存在但方向不一致；可能 Ta/Tb 指派规则失效，触发 sensitivity 检查（用 frac_SOZ_sink 或两端综合指标重指派） |

---

## 6. 实现合同检查清单（CLAUDE.md §5/§6 对应）

每条都必须在代码 review 时勾掉，否则不接受 commit。

- [ ] **Channel ordering 严验**：npz `channel_names`、PR-2 JSON `channel_names`、PR-6 JSON `node_anatomy.channel_names_order` 三方比对，不一致 raise ValueError；不允许 default-by-index alignment
- [ ] **Participating mask 来自 raw `bools`，非 PR-2 `template_valid_mask`**：后者是 cluster-level，不是 event-level；混用会把 fallback rank 当成参与
- [ ] **Eligibility 检查**：`compute_event_delta` 在 SOZ_p 或 nonSOZ_p 为空、或 |participating_e| < 3 时 return NaN，不返回 0；caller 用 `np.nanmedian` 聚合；如全 NaN 则 raise（不静默给 0）
- [ ] **统计单元 = subject**：cohort Wilcoxon 必须接受按 subject 名 sort 后的对齐数组；entrance assertion: `assert sorted(d_a.keys()) == sorted(d_b.keys())`，不一致 raise；**禁止** `list(d.values())`
- [ ] **`forward_reverse_reproduced` OR rule**：同时读 `splits.first_half_second_half` 和 `splits.odd_even_block` 两个 boolean，OR；只读其一会少算
- [ ] **PR-6 anchoring `valid_mask` 必须显式传入**：full-data Δ_event 计算时如果调用 anchoring helper，必须从 raw bools 派生 per-cluster `valid_mask` 显式传入；不允许默认 `valid_mask=None`
- [ ] **Stub raise**：H2 sensitivity 中如果 per-seizure onset 通道尚未对接，对应 helper 必须 `raise NotImplementedError("per-seizure onset channel resolver not yet implemented")`，不给 fallback "best-effort"
- [ ] **lagPat 文件类型**：必须 `*_lagPat_withFreqCent.npz`；脚本里硬编码扩展名，不在 try/except 里 fallback 到 `*_lagPat.npz`
- [ ] **forbid pooled-event p-value**：source code grep 不能出现 `wilcoxon(np.concatenate(per_event_delta_list))` 之类的 cohort-level pooled 调用
- [ ] **NaN propagation 测试**：TDD 必须包含 SOZ_p 空 / nonSOZ_p 空 / participating < 3 三种情况的 NaN 测试
- [ ] **Forbidden-phrase grep**：results doc 写完后跑 `grep -niE 'excitation|inhibitory rebound|证明|机制成立'` 在 archive 里，命中即修

---

## 7. 文件 / 模块 map

新增：

- `src/intra_event_spatial.py` — 核心估计器（rel_lag、Δ_event、template polarity 指派、N1 shuffle）
- `tests/test_intra_event_spatial.py` — TDD 测试 12 项（见 §9）
- `scripts/run_intra_event_spatial.py` — CLI runner（`--audit` / `--per-subject` / `--cohort-stats` / `--sensitivity-alt-soz`）
- `scripts/plot_intra_event_spatial.py` — 4 张图（见 §10）
- `docs/archive/topic1/pr8_intra_event_spatial_results_<date>.md` — 完成时写

新增结果目录（参考 AGENTS.md 命名规则）：

```
results/intra_event_spatial/
├── audit.csv
├── cohort_h1.json
├── cohort_h1_prime_polarity_flag.json
├── sensitivity_alt_soz.json          # 进入 Step 5 才生成
├── per_subject/
│   ├── epilepsiae_<sid>.json
│   └── yuquan_<sid>.json
└── figures/
    ├── README.md                      # 中文，每图 2-4 句
    ├── fig1_cohort_directional.png    # subject-level Δ_Ta vs Δ_Tb scatter + paired diff box
    ├── fig2_polarity_reversal.png     # H1' flag 计数条形图
    ├── fig3_subject_exemplar_548.png  # 548 (largest n) per-event Δ histogram per cluster
    └── fig4_audit_table.png           # 每 subject 的 n_eligible / frac_SOZ_source / 是否进 H1
```

不新增 / 不动：`src/interictal_propagation.py`（PR-1..PR-6 已固化，本 PR 只读其输出 JSON）、`src/template_temporal_pairing.py`（PR-7 已封）。

---

## 8. Step breakdown

每步 2–5 分钟一个 task，TDD-first。每步独立 commit。

### Step 0 — Audit + cohort 名单

- [ ] **0.1** 写 `scripts/run_intra_event_spatial.py --audit`：枚举 PR-2.5 fwd/rev cohort（OR rule）；对每个 subject 读 `*_lagPat_withFreqCent.npz` + PR-2 per-subject JSON + PR-6 anchoring + SOZ JSON，计算 per-cluster `n_eligible`、`frac_SOZ_source`、是否能指派 Ta/Tb。
- [ ] **0.2** 输出 `results/intra_event_spatial/audit.csv`，列：`dataset, subject, cluster_id, n_events_total, n_eligible, frac_SOZ_source, frac_SOZ_sink, n_valid_endpoint, fwd_rev_split_half, fwd_rev_odd_even, fwd_rev_or, in_h1_directional, in_h1_polarity_flag`。
- [ ] **0.3** 跑 audit，人眼复核：cohort 计数 vs PR-2.5 8/9 / PR-6 fwd/rev cohort；如果数字对不上，停下追源头，不要直接进 Step 1。
- [ ] **0.4** Commit：`feat(pr8): Step 0 — audit cohort eligibility for intra-event spatial polarity`

### Step 1 — 核心 spatial estimator + TDD

新文件 `src/intra_event_spatial.py`，签名：

```python
def compute_relative_lag(
    lag_pat_raw_event: np.ndarray,        # shape (n_channels,), float, NaN 允许
    valid_mask_event: np.ndarray,         # shape (n_channels,), bool
) -> np.ndarray:
    """Return rel_lag with NaN where valid_mask==False."""

def compute_event_delta(
    rel_lag_event: np.ndarray,
    valid_mask_event: np.ndarray,
    soz_mask: np.ndarray,                 # shape (n_channels,), bool, fixed per subject
    nonsoz_mask: np.ndarray,
    min_participating: int = 3,
) -> float:
    """Return Δ_event or NaN if ineligible (any side empty / |participating|<min)."""

def compute_template_delta(
    delta_events: np.ndarray,             # 1D, may contain NaN
    min_n_eligible: int = 30,
) -> tuple[float, int]:
    """Return (median_Δ, n_eligible). median_Δ=NaN if n<min."""

def assign_template_polarity(
    per_template_anchoring: list[dict],
    min_endpoint: int = 6,
    eps_tie: float = 1e-9,
) -> dict | None:
    """Return {'Ta_cluster_id': k, 'Tb_cluster_id': j, 'frac_SOZ_source_a': float, 'frac_SOZ_source_b': float} or None if cannot assign."""

def shuffle_label_null_per_subject(
    delta_events_per_event: np.ndarray,   # full N events (NaN for ineligible)
    cluster_labels: np.ndarray,
    n_perm: int = 1000,
    rng_seed: int = 0,
) -> dict:
    """Return {'diff_obs', 'diff_perm_distribution', 'p_two_sided'}; permute labels preserving counts."""
```

- [ ] **1.1** TDD test T1：`compute_relative_lag` 在全 valid 输入下，rel_lag.min() == 0；shift 全部 lag 一个常数，rel_lag 不变。
- [ ] **1.2** T2：`compute_relative_lag` 在 valid_mask=False 处 return NaN。
- [ ] **1.3** T3：`compute_event_delta` SOZ 早 nonSOZ 晚 → Δ < 0；反之 Δ > 0；对称构造。
- [ ] **1.4** T4：`compute_event_delta` SOZ_p 空 → return NaN（不 return 0）。
- [ ] **1.5** T5：`compute_event_delta` nonSOZ_p 空 → return NaN。
- [ ] **1.6** T6：`compute_event_delta` |participating|=2 → return NaN（< min_participating=3）。
- [ ] **1.7** T7：`compute_template_delta` n_eligible < 30 → return (NaN, n)。
- [ ] **1.8** T8：`compute_template_delta` 全 NaN 输入 → return (NaN, 0)，不 raise；caller 决定如何处理。
- [ ] **1.9** T9：`assign_template_polarity` 两 cluster frac_SOZ_source 相等 → return None（tie）。
- [ ] **1.10** T10：`assign_template_polarity` 一 cluster `n_valid < 6` → return None。
- [ ] **1.11** T11：`assign_template_polarity` 正常 case：frac_SOZ_source = [0.8, 0.0] → Ta=cluster 0, Tb=cluster 1。
- [ ] **1.12** T12：`shuffle_label_null_per_subject` 输入 NaN-free 完全可分离的 Δ → diff_obs 极端 → p_two_sided < 0.01。
- [ ] **1.13** 跑 `pytest tests/test_intra_event_spatial.py -v`，预期 12/12 PASS。
- [ ] **1.14** Commit：`feat(pr8): Step 1 — core intra-event spatial estimator + TDD (12/12)`

### Step 2 — Per-subject runner + 输出 JSON

- [ ] **2.1** 在 `scripts/run_intra_event_spatial.py` 加 `--per-subject` 模式；接受 `--subject`（单跑）或 `--all`（cohort）。
- [ ] **2.2** Per-subject 流程：
  ```
  load lagPat_withFreqCent.npz  # 必须是 _withFreqCent
  load PR-2 cluster JSON         # cluster.labels
  load PR-6 anchoring JSON       # per_template
  load SOZ JSON                  # soz_channels
  verify channel_names alignment 三方
  build SOZ mask, nonSOZ mask
  for e in events:
      Δ_event[e] = compute_event_delta(...)
  for k in {0, 1}:
      median_Δ_k, n_eligible_k = compute_template_delta(Δ_event[labels==k])
  polarity = assign_template_polarity(per_template_anchoring)
  if polarity is not None:
      Δ_Ta = median_Δ_k @ polarity['Ta_cluster_id']
      Δ_Tb = median_Δ_k @ polarity['Tb_cluster_id']
  N1_null = shuffle_label_null_per_subject(Δ_event, labels, n_perm=1000)
  write per_subject/<dataset>_<subject>.json
  ```
- [ ] **2.3** Per-subject JSON schema（locked at plan time）：
  ```json
  {
    "dataset": "...",
    "subject": "...",
    "n_events_total": int,
    "n_events_eligible": int,
    "eligibility_rate": float,
    "fwd_rev_split_half": bool,
    "fwd_rev_odd_even": bool,
    "fwd_rev_or": bool,
    "stable_k": int,
    "per_cluster": [
      {"cluster_id": 0, "n_eligible": int, "median_delta": float, "frac_SOZ_source": float, "frac_SOZ_sink": float, "n_valid_endpoint": int}
    ],
    "polarity_assignment": {
      "Ta_cluster_id": int_or_null,
      "Tb_cluster_id": int_or_null,
      "tie": bool,
      "exit_reason": str_or_null
    },
    "subject_delta": {
      "delta_Ta": float_or_null,
      "delta_Tb": float_or_null,
      "diff_Tb_minus_Ta": float_or_null,
      "polarity_reversal_flag": bool_or_null
    },
    "n1_label_shuffle_null": {
      "diff_obs": float,
      "diff_perm_median": float,
      "diff_perm_p2sided": float,
      "n_perm": int,
      "rng_seed": int
    },
    "in_h1_directional": bool,
    "in_h1_polarity_flag": bool
  }
  ```
- [ ] **2.4** 跑全 cohort（fwd/rev OR-cohort，约 8/9 subjects）；rng_seed=0 保证可复现。
- [ ] **2.5** Commit：`feat(pr8): Step 2 — per-subject intra-event spatial Δ + N1 shuffle`

### Step 3 — Cohort H1 + H1' 检验

- [ ] **3.1** 加 `--cohort-stats` 模式：聚合 per_subject JSON。
- [ ] **3.2** Cohort 流程（**严格按 §3.4 contract**）：
  ```
  in_h1_dir = [s for s in per_subject if s.in_h1_directional]
  in_h1_pol = [s for s in per_subject if s.in_h1_polarity_flag]

  # 严格 key match
  d_Ta = {s.subject: s.subject_delta.delta_Ta for s in in_h1_dir}
  d_Tb = {s.subject: s.subject_delta.delta_Tb for s in in_h1_dir}
  assert sorted(d_Ta.keys()) == sorted(d_Tb.keys())  # raise if not

  keys = sorted(d_Ta.keys())
  arr_Ta = np.array([d_Ta[k] for k in keys])
  arr_Tb = np.array([d_Tb[k] for k in keys])

  diff = arr_Tb - arr_Ta
  median_diff = np.median(diff)
  wilcoxon_p_one_sided = wilcoxon(diff, alternative="greater").pvalue
  wilcoxon_p_two_sided = wilcoxon(diff, alternative="two-sided").pvalue
  sign_p = binom_test( sum(diff > 0), n=len(diff), p=0.5)

  # H1' polarity flag
  flags = [s.subject_delta.polarity_reversal_flag for s in in_h1_pol]
  k_pos = sum(flags)
  binom_p = binom_test(k_pos, n=len(flags), p=0.5, alternative="two-sided")
  ```
- [ ] **3.3** PASS 判据 §3.5 三条全部满足 → `verdict = "PASS"`；任一不满足 → `"NULL"` / `"abort"` / `"partial"`。
- [ ] **3.4** 输出 `results/intra_event_spatial/cohort_h1.json` 与 `cohort_h1_prime_polarity_flag.json`，包含 `n_subjects_in_h1`, `median_diff`, `wilcoxon_p_one_sided`, `wilcoxon_p_two_sided`, `sign_p`, `n_polarity_reversal`, `n_total`, `binom_p`, `verdict`。
- [ ] **3.5** Commit：`feat(pr8): Step 3 — cohort H1 directional + H1' polarity_reversal`

### Step 4 — N1 per-subject sanity 表格

- [ ] **4.1** 把 per-subject `n1_label_shuffle_null.diff_perm_p2sided` 整理成 table；标注哪些 subject p < 0.05。
- [ ] **4.2** 报告："多少 subject 的 |diff_obs| 在 label-shuffle null 之外"——这是 sanity，不是 PASS gate。如果 cohort H1 NULL 但 N1 多数显著，触发 N2 confound discussion（不在本 PR 实施 N2，留为下游）。
- [ ] **4.3** 写入 cohort_h1.json 的 `n1_sanity` 字段。
- [ ] **4.4** Commit：`feat(pr8): Step 4 — N1 per-subject label-shuffle sanity`

### Step 5 — Sensitivity: alternative SOZ definitions

- [ ] **5.1** 列三种替代 SOZ 来源，**逐一**对接，不存在的 raise NotImplementedError：
  - **Alt-1**: PR-1 spatial_modulation 已对接的 ER-leading 通道（仅 Yuquan，9 valid pairs，参见 `docs/archive/topic3/spatial_modulation_soz_analysis.md`）
  - **Alt-2**: per-seizure onset 通道（来自 `src/preprocessing.py::detect_seizure_by_spatial_extent` 的 onset detection；如果当前 cohort 的 per-seizure onset 通道尚未导出 JSON，stub `raise NotImplementedError`）
  - **Alt-3**: Epilepsiae focus_rel = 'i' + 'l'（合并 SOZ 与 lesion，看是否扩大 SOZ 集合后 Δ 仍同号）
- [ ] **5.2** 对每种 alt SOZ 重跑 §3.2–§3.4；输出 `results/intra_event_spatial/sensitivity_alt_soz.json`，含 `alt1_yuquan_only / alt2_per_seizure_onset / alt3_focus_rel_il` 三个字段，每个字段含 `n_subjects, median_diff, wilcoxon_p_one_sided, sign_p, verdict_consistency_with_main`。
- [ ] **5.3** **不**把 sensitivity 改成 primary。即使 alt 比 main 更显著，也只在 results doc §sensitivity 报告，main verdict 仍由 §3.5 main SOZ 决定。
- [ ] **5.4** Commit：`feat(pr8): Step 5 — alt SOZ sensitivity (Alt-1 ER-leading / Alt-2 stub / Alt-3 focus_rel)`

### Step 6 — 可视化

- [ ] **6.1** Fig 1: cohort directional scatter—— per subject `(Δ_Ta, Δ_Tb)` 点 + paired diff box（`Δ_Tb − Δ_Ta`）。pass 时希望落在 `y > x` 线上方（或 `Δ_Ta < 0 < Δ_Tb` 象限）。
- [ ] **6.2** Fig 2: H1' polarity_reversal 条形图——`同号 / 异号` 计数 vs 0.5 baseline。
- [ ] **6.3** Fig 3: subject-548 per-cluster Δ_event 直方图（最大 n，最具单 subject 解读价值）；如果 548 不在 fwd/rev cohort，换最大 n_eligible 的 subject。
- [ ] **6.4** Fig 4: audit table heatmap——每 subject × 每 cluster 的 `frac_SOZ_source` + `n_eligible`，标注 `in_h1_directional`。
- [ ] **6.5** 生成 `results/intra_event_spatial/figures/README.md`，每图 2–4 句中文说明 + "**关注点**"行（参考 AGENTS.md "Results Directory Standards"）。
- [ ] **6.6** Commit：`feat(pr8): Step 6 — visualization (4 figs + Chinese README)`

### Step 7 — Doc closeout

- [ ] **7.1** 写 `docs/archive/topic1/pr8_intra_event_spatial_results_<commit-date>.md`，结构：
  - §1 Cohort & eligibility
  - §2 Main verdict（H1 directional + H1' polarity flag）
  - §3 N1 per-subject sanity（不当 PASS gate）
  - §4 Sensitivity（Alt SOZ）
  - §5 Allowed / forbidden phrasings actually used
  - §6 Figure inventory
  - §7 Locked final conclusion
- [ ] **7.2** 在 `docs/topic1_within_event_dynamics.md` §2（"PR 状态一句话总览"）和 §7 末尾加 PR-8 verdict 单行 + archive 链接；**严格按 §3.5 / §3.6 措辞**。
- [ ] **7.3** 在 `docs/archive/topic1/ping_pong_hypothesis_review_2026-04-28.md` §11.4（L4）和 §12.4 末尾追加 "PR-8 verdict" 一段，引用 archive results doc。
- [ ] **7.4** Forbidden-phrase grep（自动）：在 archive results doc 跑 `grep -niE 'excitation|inhibitory rebound|证明|机制成立|mechanism proven'`；命中即修。
- [ ] **7.5** Commit：`docs(pr8): Step 7 — close out PR-8 intra-event spatial polarity`

---

## 9. TDD 测试列表（locked at plan time，12 项）

| ID | 函数 | 关注的 invariant |
|---|---|---|
| T1 | `compute_relative_lag` | `rel_lag.min() == 0`；常数 shift 不变 |
| T2 | `compute_relative_lag` | valid_mask=False 处 NaN |
| T3 | `compute_event_delta` | SOZ 早 nonSOZ 晚 ⇒ Δ < 0；反向构造对称 |
| T4 | `compute_event_delta` | SOZ_p 空 ⇒ NaN |
| T5 | `compute_event_delta` | nonSOZ_p 空 ⇒ NaN |
| T6 | `compute_event_delta` | participating < 3 ⇒ NaN |
| T7 | `compute_template_delta` | n_eligible < 30 ⇒ NaN |
| T8 | `compute_template_delta` | 全 NaN 输入 ⇒ (NaN, 0)，不 raise |
| T9 | `assign_template_polarity` | frac_SOZ_source tie ⇒ None |
| T10 | `assign_template_polarity` | n_valid < 6 ⇒ None |
| T11 | `assign_template_polarity` | 正常 [0.8, 0.0] ⇒ Ta=0, Tb=1 |
| T12 | `shuffle_label_null_per_subject` | 完全可分离 Δ ⇒ p_two_sided < 0.01 |

`tests/test_intra_event_spatial.py` 一文件全收。所有测试用 `np.testing.assert_*` + `pytest.raises`，不依赖外部数据。

---

## 10. 可视化 spec（locked）

| Fig | 内容 | size | 命名 |
|---|---|---|---|
| 1 | cohort scatter + paired diff box | 12 × 5 | `fig1_cohort_directional.png` |
| 2 | polarity reversal flag bar | 6 × 4 | `fig2_polarity_reversal.png` |
| 3 | subject exemplar per-cluster Δ histogram | 10 × 4 | `fig3_subject_exemplar_<sid>.png` |
| 4 | audit heatmap | 8 × n_subjects×0.4 | `fig4_audit_table.png` |

颜色复用 `src/plot_style.py` Morandi 调色（与 PR-7 figs 一致）。

---

## 11. 成功 / 失败的判读语言（再强调一次）

不论 verdict 如何，写 results doc / topic1 时严格按下表挑选措辞：

| Verdict | 允许说 | 严禁说 |
|---|---|---|
| **PASS** | "PR-2.5 fwd/rev cohort 上 forward template 表现为 SOZ-leading（Δ_Ta < 0），reverse template 表现为 SOZ-trailing（Δ_Tb > 0），subject-level Wilcoxon p=X" | "证明 ping-pong 机制 / forward = excitation / reverse = inhibitory rebound" |
| **PARTIAL** | "polarity_reversal 在 N/M 个 subject 中观察到（H1' p=X），但 directional 检验未达显著（H1 p=Y）" | "支持 ping-pong" |
| **NULL** | "未观察到 forward/reverse template 在 SOZ-内外空间极性上的显著差异（p=X，n=Y）；alt SOZ 替换后保持一致" | "机制不存在 / SOZ 内外没有结构 / ping-pong 被证伪" |
| **ABORT** | "样本量不足以做 cohort claim（H1 n<5 / H1' n<6）" | 任何带方向的结论 |

---

## 12. Out of scope（写明）

- L1 phenomenology（PR-2 已完成，不重做）
- L2 短窗时间耦合（PR-7 已封 NULL，不重开）
- L3 长时动力学（未来 PR）
- 机制层 E/I 归因（HFO 频段不可分，需 LFP / 单元活动并行实验）
- History-dependent marked point process model（review §12.1，独立 PR）
- Peri-seizure template ratio + variance trajectory（review §12.2，独立 PR）
- Subject 548 single-subject case-study（review §12.3，独立 PR）
- Cohort-level permutation test（subject-level Wilcoxon 已是 cohort 推断，不需）

---

## 13. 自检清单（写完 plan 必须自查）

- [x] 每个 step 有具体 file path（§7）和 commit message
- [x] TDD 12 项每项 invariant 清晰，无 placeholder
- [x] 失败模式（§5.2）写明 abort 条件，不是软警告
- [x] 禁止性表述（§3.6 / §11）锁定，跨文档一致
- [x] CLAUDE.md §6 五条 contract 全部对应到 §6 检查清单
- [x] 不复用 PR-7 已 raise 的 stub（resample_isi_per_cluster）
- [x] Channel ordering 三方对齐（npz / PR-2 JSON / PR-6 JSON）写入 §6
- [x] `forward_reverse_reproduced` OR-rule（CLAUDE.md cross-PR contract）写入 §6
- [x] PR-6 anchoring `valid_mask` 显式传入要求写入 §6
- [x] Stub raise NotImplementedError（H2 sensitivity Alt-2）写入 §8 Step 5
- [x] Visualization README.md 中文 + "关注点" 行（AGENTS.md 规范）写入 §8 Step 6
- [x] Out of scope 明列，不让 Plan 蔓延

---

## 14. 一句话承诺

PR-8 只检验"在 PR-2.5 fwd/rev cohort 上，forward template 是否 SOZ-leading、reverse template 是否 SOZ-trailing"。无论 PASS / NULL，都不论及机制层，不复活 L2，不替代 L1/L3。SOZ 标注是粗标签——NULL 时只写"以 SOZ JSON 定义的极性签名不成立"，**不**写"机制不存在"。

---

## 15. Deferred 状态说明（2026-04-30 决定）

本 v1 plan **不执行**。PR-8 的核心因变量"SOZ-first / SOZ-last"完全依赖 SOZ 标签的质量；clinical 'i' / 'l' / 'e' 在 cohort 上的可靠性是**未量化的**前置变量。在没有数据驱动 SOZ 作为对照的前提下：

- v1 H1 / H1' 在 clinical SOZ 单一定义下 NULL → 杀伤力弱（"以临床主观标签未观察到极性"，不能推及机制层）
- v1 H1 / H1' PASS → 也可能是 PR-6 anchoring 的 rank-based 结论被 raw-time-based 重新包装（partial circularity，§16.1）

**先做 PR-T3-1**（`docs/archive/topic3/pr_t3_1_data_driven_soz_audit_plan_2026-04-30.md`）：用 HFO-onset rate (M1) + ER-ratio (M2) 派生数据驱动 SOZ，量化 clinical SOZ 在当前 cohort 上的可靠性。PR-T3-1 完成后回写 PR-8 v2，§3.4 H2 sensitivity 重构为 multi-source 协同检验（clinical + M1 + M2 并列；多源同向才支持 strong claim，分歧降级为 source-dependent）。

回写 v2 时**必须**合并下面 §16 的 9 条修订点。

---

## 16. v2 修订点（待 PR-T3-1 完成后并入）

下面 9 条由 2026-04-30 review 提出，v1 在没看到 cohort 数据前不重写；v2 必须全部应用。

### 16.1（致命）H1 directional 与 PR-6 anchoring 的 partial circularity

**v1 问题**：§3.3 用 `frac_SOZ_source` 最大的 cluster 定义 Ta，再检验 Ta 是否 SOZ-first。但 `template_rank` 由 per-event `argsort(lagPatRaw)` 派生，且 SOZ 已参与 Ta/Tb 指派 → H1 可被解读为"PR-6 endpoint geometry 在 event-level 的重新包装"，而非全新机制检验。

**v2 修法（二选一）**：

- **A. Held-out orientation（推荐）**：把 events 按时间 50/50 切两半。前半数据用来跑 PR-2 cluster + PR-6 anchoring → 决定 Ta / Tb 指派；后半数据**不参与指派**，只用于计算 Δ_event 并跑 H1 directional。再翻过来跑一次（后半指派 + 前半检验），取 cohort 一致方向。Held-out 检验是真正独立的。
- **B. 降级 framing**：保留 v1 设计，但把 H1 verdict 在 results doc 中明确写为"event-level validation of PR-6 SOZ-source assignment"（PR-6 endpoint anchoring 是否在 raw-lag 层 generalize），而**不**写"全新机制 / 独立检验 / novel finding"。

v2 默认采用 A；若 cohort 太小不支持 50/50 切分（每半 < 30 eligible events / cluster），fall back 到 B 并显式标注。

### 16.2（致命）`shuffle_label_null_per_subject` 缺 Ta/Tb 参数

**v1 问题**：§7 / §8 Step 1.12 函数签名只接受 `cluster_labels`，但 null 要算 `Δ_Tb − Δ_Ta`，必须知道哪个 label 是 Ta、哪个是 Tb。否则默认按 cluster id 0/1 当作 Ta/Tb 会在 KMeans label 任意性上出错。

**v2 修法**：

```python
def shuffle_label_null_per_subject(
    delta_events_per_event: np.ndarray,
    cluster_labels: np.ndarray,
    ta_cluster_id: int,
    tb_cluster_id: int,
    block_time_ranges: list[tuple[float, float]] | None = None,   # 见 16.3
    event_abs_times: np.ndarray | None = None,                     # 见 16.3
    n_perm: int = 1000,
    rng_seed: int = 0,
) -> dict:
    """
    Compute observed diff = median(Δ | label==ta) − median(Δ | label==tb).
    Shuffle labels, recompute, return null distribution + p_two_sided.
    """
```

TDD 增加：参数缺失 raise；ta == tb raise；ta / tb 不在 cluster_labels 唯一值集 raise。

### 16.3（致命）N1 label shuffle 默认应 block-aware

**v1 问题**：§4.1 是全局 label shuffle。Δ_event 很可能随 block / vigilance / rate state 漂移；全局 shuffle 打散这些慢结构，per-subject p 偏乐观。

**v2 修法**：

- 默认 N1 改为 **within-block label shuffle**（在每个 `block_time_range` 内独立 permute label，跨 block 不混合）
- 全局 shuffle 降级为 weak sanity，仅在 per-subject JSON 中作为附加字段 `n1_global_shuffle_for_reference`
- Per-subject JSON `n1_label_shuffle_null` 字段加 `mode = "within_block" | "global"`
- TDD：构造 within-block 与跨-block Δ 分布显著不同的合成数据，验证 within-block null 给出比 global null 更保守的 p 值

### 16.4（致命）H1' 统计合同不一致

**v1 问题**：§3.4 表格写"H1' 比例显著 > 0.5"，但 §8 Step 3 用 `binom_test(..., alternative="two-sided")`。两边方向不一致。

**v2 修法（二选一，必须挑一个）**：

- **A**：H1' 写 one-sided greater：`binom_test(k_pos, n, p=0.5, alternative="greater")`，与"polarity_reversal 比随机更多"的预期对齐
- **B**：保留 two-sided，并把 §3.4 表格改成"polarity_reversal flag rate 显著偏离 0.5"（不预设方向）

v2 默认采用 A（与"forward / reverse template 反向极性"假设的方向一致）。

### 16.5（致命）主文档结论应等 sensitivity 后再写

**v1 问题**：§8 Step 7.2 在 Step 3 完成后就准备回写 `docs/topic1_within_event_dynamics.md`，但 H2 alt SOZ sensitivity 在 Step 5。按 CLAUDE.md §5"sensitivity gates are pre-conditions for main-doc conclusions"，这个顺序违反规则。

**v2 修法**：

- Step 3 后只写 archive preliminary（`docs/archive/topic1/pr8_intra_event_spatial_results_<date>.md`），verdict 标注"preliminary, pending multi-source SOZ sensitivity"
- Step 5（v2 改为 multi-source SOZ 协同检验，clinical + M1 + M2）完成后才允许回写 topic1 §2 / §7
- Step 7 拆成 Step 7a（archive preliminary）与 Step 7b（topic1 主文档回写，门控在 Step 5 通过后）

### 16.6（中等）Δ_event 用均值对少量 SOZ 通道敏感

**v1 问题**：§3.2 `Δ_event = mean(rel_lag[SOZ_p]) − mean(rel_lag[nonSOZ_p])`；当 SOZ_p 只有 1–2 个通道时均值受单通道极端值左右。

**v2 修法**：appendix sensitivity，在 per-subject JSON 加 `subject_delta_median_variant`：

```
Δ_event_median(e) = median(rel_lag[SOZ_p]) − median(rel_lag[nonSOZ_p])
```

results doc §appendix 报告 main vs median variant 的 cohort verdict 是否一致。不替换 main metric。

### 16.7（中等）"forward / reverse template" 命名不稳

**v1 问题**：v1 把 Ta = SOZ-source-enriched cluster 直接叫"forward template"，但 PR-6 / PR-2 里"forward / reverse"是 rank-based 几何标签，与"SOZ-source-enriched"不严格等价。直接互换会引入语义漂移。

**v2 修法**：

- 文档统一把 Ta 写成 **"SOZ-source-enriched template (Ta)"**，Tb 写成 **"SOZ-source-depleted template (Tb)"**
- 仅在引用 PR-6 几何时使用 "forward / reverse"，并显式说明"在本 cohort 上 SOZ-source-enriched 与 forward 几何高度重合（cohort 一致性 = X/N）"作为可验证 footnote
- topic1 §7 / archive results doc / 图标题 / README 全部按这套命名

### 16.8（中等）H1 p ∈ [0.05, 0.10] 没定义

**v1 问题**：§3.5 PASS 判据三条全过；NULL 判据 p > 0.10。p ∈ (0.05, 0.10] 没有专门档位。

**v2 修法**：增加 **inconclusive / trend** 档位：

| 条件 | verdict |
|---|---|
| H1 三条全过 | PASS |
| Wilcoxon p ∈ (0.05, 0.10] 且 sign / 中位数方向一致 | **trend, archive only** |
| Wilcoxon p > 0.10 | NULL |
| 任一方向反向 | NULL（带方向 confound 注脚） |

trend 档**不**进 topic1 §7，仅 archive；results doc verdict 字段允许值扩展为 `{"PASS", "TREND", "NULL", "PARTIAL", "ABORT"}`。

### 16.9（中等）Step 5 Alt-2 stub raise

**v1 已写**：§8 Step 5.1 Alt-2 在 per-seizure onset 通道未导出时 `raise NotImplementedError`。**v2 保留**，不改。

v2 中 Step 5 整体被 PR-T3-1 输出取代（multi-source SOZ 协同检验，consume `results/spatial_modulation/data_driven_soz/`），Alt-1 / Alt-3 也相应重写；但"未导出的功能必须 raise"这条原则跨 v1 / v2 保留。

### 16.10（追加）多源 SOZ 协同 verdict 设计（PR-T3-1 完成后填）

**待 PR-T3-1 verdict 决定后填**：

- 若 PR-T3-1 verdict = "broadly consistent"：PR-8 v2 用 clinical + M1 + M2 三源并列，三源同向 → strong PASS / NULL；任一源分歧 → source-dependent
- 若 PR-T3-1 verdict = "partially consistent"：PR-8 v2 以 M1 / M2 数据驱动 SOZ 为主，clinical 作 audit baseline；多源同向才能写 topic1 §7
- 若 PR-T3-1 verdict = "unreliable"：PR-8 v2 完全不依赖 clinical SOZ，只用 M1 / M2 派生 SOZ；clinical 仅在 archive 报对照

回写 v2 时根据 PR-T3-1 verdict 把这条具体化。
