# Topic 1：间期事件内部时序结构

> 状态：当前正式入口
> 范围：只讨论单个间期群体事件内部的时序组织，包括传播刻板性与事件级同步性。
> **Paper 1 架构性 framework**：`docs/paper1_framework_sba.md`（SBA framework：单核心假设 + 5 sharp predictions + BHPN-toy/fit + 5 dumb baselines + 失败模式）。本 topic 的 PR-2 / PR-2.5 / PR-6 / PR-7 / 待立 PR-T4-1/T4-2/9 全部受该框架统辖；任何与 framework 中已 lock 的 prediction 判据冲突的修改必须先改 framework。

---

## 1. 这个 topic 只回答什么问题

本 topic 只回答两个问题：

1. 单个群体事件内部，不同通道的激活顺序是否稳定、是否刻板、是否存在多种主要传播模式。
2. 单个事件内部的同步性指标在发作前后是否表现出系统性变化。

它**不**回答：

- 事件与事件之间的 IEI / PSD / rate modulation：那是 `docs/topic2_between_event_dynamics.md`
- 慢调制发生在 SOZ 还是 non-SOZ：那是 `docs/topic3_spatial_soz_modulation.md`

---

## 2. 一句话当前结论

- **Paper 1 framework（SBA, v1.1.2 lock 2026-04-30 / PR-7 addendum 2026-05-01）**：单核心假设 + 5 sharp predictions（P1 时间稳定 / P2 共享几何骨架 / P3 短窗 mark-independent cohort-level TOST equivalence / P4 解剖锚定 / P5 间期→发作 directionality）；**P1 + P2 PASS**（PR-2.5 / PR-6 Step 4b），**P3 INCONCLUSIVE-locked**（PR-7 addendum 完成：1800s window + lag1_same_excess null-relative 干净 PASS；10/30/60s + run_length_lift cohort CI underpowered at n=6 with structural outliers；SBA 不被 falsified；写法限定 "compatible with mark-independent within tested precision"，禁止写 PASS）；P4–P5 待执行（PR-T3-1 → PR-8 v2 / 新立 PR-9，directional predictor `D_ij = sin(φ_j*−φ_i*)`，**不**用 cos-based A_ij）。Toy model BHPN-toy（s_rate(t) 与 ε_id 两过程统计独立；T3 验证走 large-N simulation 而非 PR-7 anchor）+ fitted BHPN-fit（仅 aggregate + conditional predictions）+ 5 dumb baseline 设计 lock 在 `docs/paper1_framework_sba.md`。详见 `docs/archive/topic1/pr7_template_pairing/pr7_addendum_p3_equivalence_2026-05-01.md`。
- **传播刻板性**：内部传播真实存在但不是单一模板；`k=2` 是主导压缩，少数 subject 有 `k=4`/`k=6` 多模态；模板在 split-half / blockwise 上 `23/30 strong` + `7/30 moderate` + `0 weak`。Identity-bias 在簇内高达 86%，必须并列报告 raw 与 centered。
- **慢调制（PR-4B）**：模板混合（L1）与模板内顺序一致性（L2）cohort 全 null；模板内相对时延结构（L3）在全 cohort 上证据不足，仅在 8 个高置信子集（`dom_r > 0.7`）的 Pearson r 上探索性显著（p=0.016, 7/8）。详见 `docs/archive/topic1/propagation/interictal_group_event_internal_propagation.md`。
- **发作邻近（PR-4C）**：propagation pattern 五指标 cohort Wilcoxon 在主+辅两配置下均 null（主 1/15 / 辅 1/15 名义显著且跨配置不一致）→ **模板内部几何无稳健发作邻近调制，正式封板为阴性**。唯一稳健信号在 `rate_by_template`（post_ictal vs baseline 主 p=0.0009、辅 p=0.0067）。详见 `docs/archive/topic1/propagation/pr4c_seizure_proximity_review_2026-04-17.md` §9。
- **模板招募（PR-5）— 核心科学结论（已验收 2026-04-20）**：在 PR-5-A novel-template gate 已 PASS（main n=23 / aux n=22，未观察到 `H_OOD` 或 `H_assignment_drift` 的 cohort-level 证据）的前提下，PR-5-B 把 PR-4C `rate_by_template` 的描述层信号正式升级为推断结论：**dominant template 的绝对事件率（events/h）在 post-ictal 相对 baseline 出现 cohort-level 系统升高**（候选 A `dominant_global` `post_minus_baseline` median main `+65.46` / aux `+42.43` events/h；main p=0.00128 Bonferroni-pass α=0.0083，aux p=0.0115 nominal-pass，方向一致；候选 B main p=0.00214 同向支持 → §4.4 sensitivity gate `overall_strong=True`）。**§4.5 composition diagnostic 在 PR-5 合同下未复制 panel d**：`share_post_minus_baseline` 两配置都是 nominal-positive 但**与 panel d 预期方向相反**（main `+0.0156`, p=0.0149，direction-consistent 6/23；aux `+0.0328`, p=0.0301，direction-consistent 5/22）→ panel d 信号不在 PR-5 cohort 复现，且不为主结论背书。验收口径：PR-5-A PASS / PR-5-B STRONG；`pre_minus_baseline`、`post_minus_pre` 仅留为次级描述层。详见 `docs/archive/topic1/pr5_template_recruitment/pr5_template_recruitment_plan_2026-04-20.md` §11。
- **PR-6（2026-04-25 重启）**：原 PR-6-A multi-anchor consensus / ictal-onset alignment 主线**全部冻结归档**（sentinel `548/916` 已经把“稳定 ictal onset rank”证伪：cross-seizure top10 overlap=0、cross-band ρ=−0.21、early channels 大量落在 `other`），文献亦指明该方向在领域内高风险（Schroeder 2020 / Wenzel 2017 / Pinto 2023 / Bailey 2021）。**新主线 = stable template endpoint (source ∪ sink) anatomical anchoring**：H1 检验 `frac_SOZ(endpoint) − frac_SOZ(middle)` 的 subject-level cohort Wilcoxon，subset polarity（H1b）+ forward/reverse swap（H2）+ Epilepsiae focus_rel i/l/e（H3）作为方向性 / 机制 / 解剖三类 sensitivity。复用 `match_bipolar_soz` / `match_bipolar_focus_rel` 与已有 `template_rank`，cohort 从 audit 推导不预写 N。详见正式 plan-of-record：`docs/archive/topic1/pr6_template_anchoring/pr6_template_endpoint_anchoring_plan_2026-04-25.md`；老 PR-6-A 三份 doc 顶部已加 SUPERSEDED 块。
- **PR-7 Template Antagonistic Temporal Pairing（已验收 2026-04-30）**：检验 forward/reverse template 的时间耦合签名。**核心结论 = 几何上相关，已测试时间尺度上未见 mark dependence**：H1 主检验三条 metric 全部 NULL——event-level Δt ∈ [10s, 30s] opposite-template excess（N2 主 null Wilcoxon p=0.844，sign 3/6, median(30s)=−0.015）；N3 robustness 一致 NULL（p=0.891）；N2 window sweep {10/30/60 min} 三个尺度全部 NULL（p ∈ [0.78, 0.89]）。Step 3.5 burst diagnostic 在无 ISI 阈值 same-label run 定义下未见 persistence（cohort run_length_lift median=0.977）。**精确 framing**：在已测试的 event-level fixed-window + lag-1 + run-based 三类 metric 上数据 **compatible with mark-independent sampling**（最简洁描述，**不等于证明独立**）。PR-6 已建立的 fwd/rev 几何相关性**保留**；bouncing-back / 短时接力版本 Ping-Pong 撤回。**未测**：alternative burst definitions、rate-state / seizure-proximity switching、form (4) latent-state coupling、history-dependent regression。详见 `docs/archive/topic1/pr7_template_pairing/pr7_template_pairing_results_2026-04-29.md` §17。
- **未来模型层（§7.9）** 维持冻结：当前不绑 PR 编号。
- **同步性**：cohort-level interictal synchrony 总体为 null。唯一探索性信号是 extra-focal `phase_e` 的 `pre > post`（p=0.012, r=0.31）。

---

## 3. 核心证据链

> ### Cohort tier 注解（2026-05-07 起强制双轨/三层口径）
>
> 本 topic 现在共有 **三个 cohort 层**，下面 §3.1–§3.2 任何具体数字读到时**必须先识别它属于哪一层**：
>
> | 层 | n | 谱系 | 来源 | 适用范围 |
> |---|---|---|---|---|
> | **Tier 0 — 原 30-subject** | 30 | 21 年 cusignal vintage 全链 | PR-1/PR-2/PR-2.5/PR-3/PR-4*/PR-5/PR-6/PR-7 全部主线（pre-2026-05-06） | §3.1 / §3.1b 所有原始 PASS 比例（如 `23/30 strong`、`27 × k=2`、`30/30 MI permutation`、PR-4A `3/30 < 0.8`、PR-4B 的 `dom_r > 0.7` 集等）；PR-4C / PR-5 / PR-7 全部统计；§4 同步性 cohort 统计 |
> | **Tier 1 — n=33 primary** | 33 | 30 vintage + 3 lineage-adjacent (cuda_env pack) | Slice A1 (2026-05-06) + Slice A2 (2026-05-07) 把 PR-1/PR-2/PR-2.5/PR-6 cohort 重算后落 `pr1_cohort_summary.json` default、`template_anchoring/cohort_summary_n33.json`；A1/A2 follow-up 又把 PR-3 cohort 6-panel `cohort_propagation_summary_n33.png` 与 PR-7 cohort summary 也重算到 n=9 h1_primary | **新增**：从 2026-05-07 起 PR-1/PR-2 mixture/τ/bias/stable_k 主表、PR-2.5 forward_reverse 主表、PR-6 H1/H2/H3 主表、**PR-3 cohort 6-panel n=33 主图**、**PR-7 mark-independence n=9 主表**都用此层。PR-4* / PR-5 **未**重算 |
> | **Tier 2 — n=40 extended** | 40 | Tier 1 33 + 7 legacy variant（仅 `_lagPat.npz`，不可考 pack 参数） | Slice A2 双轨第二轨；落 `pr1_cohort_summary_n40.json`、`template_anchoring/cohort_summary_n40.json`、**`cohort_propagation_summary_n40.png`** | **sensitivity / case-extended only**。任何 cohort 主张写到 paper 不允许只引此层。Tier 1 与 Tier 2 数字差异本身是 lineage robustness 检验，不是新 cohort 信号。PR-7 不分 Tier 2（per_subject 仍按 cohort 流程汇总到一份 cohort_summary，n=9 / n=21 已含 path-D）|
>
> **重要**：Tier 0 ↔ Tier 1 ↔ Tier 2 之间的数字差异要么由"cohort 增加 / 谱系变化"驱动，要么由"重新 aggregate 时函数版本变化"驱动。如果 Tier 0 与 Tier 1 在某个共同 subject 上数字不一致，先怀疑 aggregator 版本，不要默认 cohort 大小是唯一变量。一个 already-known mass：`epilepsiae/1096` 在 Tier 1/Tier 2 的 PR-6 H1 pooled 都有 `valid_mask_source=fallback_all_valid` 的 pre-existing contamination，inherits 进两轨 H1 p 值；要 fix 需要单独 sensitivity PR。
>
> **写作合同**：本 topic 的 paper-level 论述按 priority **Tier 1 > Tier 0 > Tier 2** 引用（Tier 1 是当前主表，Tier 0 是历史 framework lock 时刻的 cohort，Tier 2 仅作 lineage robustness 注脚）。任何 mixed-tier 引用必须显式标注 "(Tier 0)" / "(Tier 1)" / "(Tier 2)"，不允许裸写 `30/30` 或 `28/40` 让读者猜。

### 3.1 内部传播刻板性（**Tier 0 = 原 30-subject framework lock**）

来自 `lagPatRank + eventsBool + chnNames` 的 cluster-aware 分析。**这一节的全部比例都是 Tier 0 cohort（n=30）**——SBA framework P1/P2 的 lock 时刻引用的就是这些数字，**不要替换成 Tier 1**：

- `30/30` subject 的 pairwise Kendall `τ` 分布呈多模态
- KMeans(`k=2`) 后，簇内 `τ` 中位数 `0.250` 显著高于整体 `τ = 0.089`
- `29/30` subject within-cluster `τ > overall τ`
- `30/30` legacy MI permutation 显著，复现老论文结论

合理口径：**一个 subject 常常有多条主要传播路径，每条路径内部仍然刻板。**

### 3.1b 数据合同、聚类稳定性与跨时间复现（PR-2a/2b/2.5，**Tier 0**）

**全部 Tier 0（n=30）数字**——PR-2.5 的"strong/moderate/weak" 分级合同与 SBA P1 PASS 判据都钉在这一层：

- 全量 `30/30` subject 都找到 `stable_k`，零 fallback；`stable_k` 分布 `27 × k=2`、`2 × k=4`、`1 × k=6`
- Adaptive within-cluster `τ` 中位数 `0.252`，相对整体 uplift 中位数 `+0.100`
- PR-2.5 时间切片复现：`23/30 strong`、`7/30 moderate`、`0 weak`；split-half 中位模板相关 `0.899`，odd/even block 中位 `0.985`
- `9` 个 k=2 subject 带 `candidate_forward_reverse` 对（inter-cluster `r < -0.5`），其中 `8/9` 跨时间切片可复现互逆关系

完整数值表与算法合同见 archive `interictal_group_event_internal_propagation.md`。

#### 3.1b.1 Tier 1 / Tier 2 当前 cohort 数字（2026-05-07 起 PR-1/PR-2/PR-2.5/PR-6 主表）

| 指标 | Tier 1 (n=33 primary) | Tier 2 (n=40 extended) |
|---|---|---|
| `n_strict_mixture` | 30 | **30**（不变，7 个 path-D 全是 possible mixture） |
| `n_possible_mixture` | 3 | 10 |
| `mean_tau_median` | 0.0884 | 0.0845 |
| `bias_fraction_median` | 0.6568 | 0.7110 |
| `stable_k_distribution` | `{2:30, 4:2, 6:1}` | `{2:35, 4:2, 5:2, 6:1}`（+2 个 stable_k=5 是 zhaojinrui/zhourongxuan 单电极 4ch case） |
| PR-2.5 grade | strong=26, moderate=7 | strong=31, moderate=9 |
| PR-2.5 forward_reverse `n_with_pairs / n_reproduced` | 14 / 13 | 17 / 16 |
| PR-6 H1 pooled wilcoxon_greater p | 0.388 (n=23 eligible) | 0.223 (n=28 eligible) |
| PR-6 H1 Yuquan-only p | 0.344 (n=10) | **0.107 (n=15)**（5 个 path-D 把 median 从 0.031 推到 0.0625, marginal trend not α=0.05） |

**关键叙事**：从 Tier 0 → Tier 1，3 个 Slice A1 subject 全是 possible mixture 不是 strict；从 Tier 1 → Tier 2，7 个 path-D 也全是 possible mixture，**主流 stable_k=2 仍占 35/40**。`mean_tau_median` 略降反映 path-D 中两个 4ch outlier；`bias_fraction_median` 提升反映 path-D 多数高 bias。**Tier 1 → Tier 2 的 PR-6 H1 Yuquan p 从 0.344 降到 0.107 是 lineage robustness 提示而不是新发现**——5 个 path-D 把 median delta 从 0.031 翻到 0.0625，方向上跟 framework P4 假设一致但 effect size 仍未达 α=0.05。

历史与详细数值见两个 archive（按时间序）：
- Slice A1（lineage-adjacent，3 subject）：[`docs/archive/topic1/propagation/cohort_slice_a1_2026-05-06.md`](archive/topic1/propagation/cohort_slice_a1_2026-05-06.md)
- Slice A2（legacy variant，7 subject + 双轨报告）：[`docs/archive/topic1/propagation/cohort_slice_a2_legacy_variant_2026-05-07.md`](archive/topic1/propagation/cohort_slice_a2_legacy_variant_2026-05-07.md)

### 3.1c PR-3 / PR-4A：固定模板可视化与 occupancy 漂移（**Tier 0** — 未重算）

**这一节多数数字仍是 Tier 0 (n=30)**——PR-4A occupancy / PR-4B 全 cohort / PR-4C / PR-5 在 Slice A1/A2 后**没有重算**，要扩到 Tier 1 / Tier 2 必须各自单独发 PR。**例外**：PR-3 cohort 6-panel 与 PR-7 cohort summary 已经在 2026-05-07 follow-up 中跟着 Slice A2 一起重做（PR-3 双轨 figures、PR-7 cohort 增至 n=9 h1_primary / n=21 h2_negative），但下面三个 PR-4A 数字仍然是 framework lock 时的 Tier 0：

- PR-3 论文级 6-panel cohort 图已固定（n=30 framework lock 版）；新增簇内 identity-bias 计算（median = 86%）。**2026-05-07 follow-up**：双轨重画 `cohort_propagation_summary_n33.png` (Tier 1) + `cohort_propagation_summary_n40.png` (Tier 2)；default `cohort_propagation_summary.png` 留 Tier 1。10 个新 subject 全部出 `per_subject/<sub>_propagation.png` + `per_subject_mi/<sub>_mi_distribution.png`
- PR-4A 在固定模板投射前提下做 day/night occupancy timeline：dominant fraction Wilcoxon `p=0.124`、entropy `p=0.245`、TV distance median `0.019`（n=30）
- 模板投射 agreement 中位数 `0.888`；只有 `3/30`（`chengshuai`、`253`、`818`）低于 `0.8`
- **结论口径**：模板稳定，但占比的昼夜漂移整体较弱。**这是描述层结果，不是强机制结论**
- occupancy 在低 rate 时段天然高方差，不适合直接承担 PR-4C 的主统计读数 → PR-4D 已把这层补强成 `rate×type`

### 3.1d Cluster geometry 可视化（trilateration plane + bimodality audit，2026-05-06）

PR-2 / PR-2.5 cluster decomposition 的描述性补强。每 subject 一张 trilateration 图——把每事件按它到两个最反向模板 T_a / T_b 的距离做 2D 三边定位（T_a→(0,0)、T_b→(d_ab,0)、event 解 ‖event−T_a‖=d_a 且 ‖event−T_b‖=d_b 的 (x, y)）。每 cluster 用 2D KDE 等密度填充加散点叠加，模板钉在 cluster 中心；上方加 marginal-x 密度面板做 bimodality 直接检验。

- **入选 cohort = 20 Epilepsiae subjects**；**18 Yuquan 排除**（PR-2 saved labels 与当前 lagPat valid_events 数量不对齐 / 缺 adaptive_cluster JSON）—— **data freshness 问题不在本 PR 范围**，列为 P0 follow-up：在最新 lagPat 上重跑 PR-2 / PR-2.5
- **关键 cohort 数字**：silhouette median = **0.460**（range 0.182–0.671，5/20 < 0.3），KMeans-vs-template-matching agreement median = **0.892**（range 0.769–0.955，8/20 < 0.85），与用户先前 audit 完全一致
- **Metric drift 集中在低 n_participating 事件**：boundary fraction by n_part bin 单调下降 `0.135 → 0.097 → 0.046`
- **新审阅发现 — Marginal-x bimodality**（archive §3.5）：cohort 在 cluster discreteness 上**不是同质的**——dip + GMM BIC 给出 **11/20 BIMODAL（清晰双模）/ 8/20 AMBIGUOUS（dip 与 BIC 矛盾）/ 1/20 UNIMODAL（`442`：KMeans 在切单峰连续分布）**。PR-2 archive 的"30/30 multimodal via pairwise τ dip"是必要不充分条件——pairwise τ dip 不能区分"双 cluster"和"1D 连续谱"（连续谱端点对端点的 τ 也会双峰）。**保留 KMeans 2-cluster 主线作论文叙事**，但 cohort 异质性必须并列报告，特别 `442`。**continuous-spectrum reframe（principal-curve / 1D manifold）留给后续论文，本论文不动 KMeans 主线**
- **Showcase**：`958`（BIMODAL k=2 forward/reverse）/ `818`（BIMODAL k=4）/ `253`（AMBIGUOUS k=2 弱 cluster）/ **`442`（UNIMODAL counter-example，必须报告）**
- **Silhouette × agreement Spearman ρ = 0.889**——consistency check / sanity，**不是独立 finding**（两者都派生自同一组 d_within / d_min_other 距离差）
- **不推翻**任何 PR-2/2.5/3/4/5/6/7 主结论
- **不进 SBA framework P1–P5**；纯描述层
- 详细数字、bimodality 检验机制、failure 合同、follow-up：
  - Plan：`docs/archive/topic1/propagation/cluster_geometry_viz_plan_2026-05-06.md`
  - Results：`docs/archive/topic1/propagation/cluster_geometry_viz_results_2026-05-06.md`

### 3.2 Identity bias 不是小问题，在簇内水平更高

| 层 | raw τ | centered τ | bias fraction |
|---|---|---|---|
| 整体 | 0.089 | 0.023 | 0.652 |
| 簇内 | 0.252 | ≈0.03 | **0.86** |

簇内 86% bias 意味着每个传播模式内部约 86% 通道排序一致性来自固有激活位置（identity ordering），仅 14% 是事件特异性传播动力学。**stereotypy 主要由结构性通道排序驱动**，不否定传播结构（identity 本身反映网络拓扑约束），但量化口径必须更新。

### 3.3 Event-level synchrony 的正式统计口径

PR4–PR6 以 seizure interval 为统计单位，主指标是 `phase`：

- `phase_all` post vs pre：`p = 0.279`
- `phase_core` post vs pre：`p = 0.967`
- within-interval trajectory：`phase_all p = 0.589`，`phase_core p = 0.643`
- event rate：`p = 0.361`

cohort level **不**支持"发作后去同步重置"或"发作前同步性恢复"。

### 3.4 Topic 1 中唯一值得继续追的 synchrony 信号

Epilepsiae 的区域分层分析中：

- `phase_i`：`p = 0.646`
- `phase_l`：`p = 0.543`
- `phase_e`：`p = 0.012`，`r = 0.31`，方向 `pre > post`，Bonferroni 校正后仍勉强保留

这是 synchrony 线中唯一可称为 exploratory-significant 的结果。

---

## 4. 当前最可信的结果

### 4.1 传播刻板性

- 多模态是普遍现象但主要压缩仍是 `k=2`
- 模板跨时间切片总体稳定（`23/30 strong`，余 `7/30 moderate`，无 weak）
- PR-3 / PR-4A 固定模板可视化已稳定，day/night occupancy 漂移弱
- forward/reverse 双模式可复现（`8/9` k=2 subject）
- legacy MI 全部显著，老论文最硬的结果站得住
- 真正可信的定量指标应是 cluster-aware `τ` 与 raw/centered 并列报告

### 4.2 同步性

- 主结论是 population-level null
- `phase` 是主指标，`legacy` 仅作兼容，`span` 仅作附录
- `phase_e` 的 `pre > post` 是唯一需要继续追的分层信号

---

## 5. 仍未解决的问题 / 风险点

- SOZ > non-SOZ 的传播优势仍偏弱，更像探索性趋势，不该写成定论
- centered rank 可能过度校正；`soz_source_erased` 仅 `3/30`，仍必须和 raw 结果并列报告
- PR-4A occupancy 已给出 day/night 描述性答案，但不能升级为正式发作邻近主结论
- `candidate_forward_reverse` 仅是描述标签（`inter-cluster Spearman r < -0.5`），不是生理机制
- 高 k subject 中 `818` 与 `zhangjinhan` 仍需 `n_participating` 匹配子样本验证
- 固定模板投射 agreement 整体够高，但 `chengshuai`、`253`、`818` 这 3 个 subject 解释时间轨迹细节时需谨慎
- synchrony 线最大风险是"把 null 写得太花"；最诚实说法是 **总体 null，局部 extra-focal 线索待验证**
- propagation 与 synchrony 是不同统计对象，文档里必须并列而非混写
- **PR-4B 高置信子集 H2 探索性支持的功效极有限**：n=8 Wilcoxon 最小可能 p = 0.004，当前 p=0.016（W=1）；不能据此做 population-level 结论。`huangwanling` 在 L3 读数上完全 ineligible（n=29）
- **PR-4C 阴性已封板**（2026-04-19，三处合同 P0 修复后复跑两套配置）。原本怀疑被合同 bug 掩盖的 `pre_ictal vs baseline raw_tau aux` 在修复后也消失（p=0.0005 → p=0.141），印证旧版那条信号是 bug 制品
- **PR-5 已完成**：完整阈值 / 失败合同 / sensitivity gate 三态以 archive `pr5_template_recruitment_plan_2026-04-20.md` 为单一来源；本文档只保留判定摘要。已知边界：PR-5-B cohort 必须严格限定在 PR-5-A retained subset（main n=23 / aux n=22）；`pre_minus_baseline` / `post_minus_pre` 主配置未通过 Bonferroni（次级描述层）；§4.5 composition diagnostic `share_post_minus_baseline` 在两配置下都 nominal-significant 但方向与 panel d 预期反向 → panel d 未在 PR-5 合同下复制，且不能为主结论背书
- **PR-6 已 pivot（2026-04-25）**：原 PR-6-A multi-anchor consensus / ictal-onset alignment 三份 doc（`pr6a-1.md`、`pr6a_template_ictal_alignment_plan_2026-04-21.md`、`pr6a_step0-2_step3preview_review_2026-04-23.md`）已冻结归档；不再继续推 ER pipeline / CUSUM / `t_ER_onset` 封板。新主线见 §7.10 + `docs/archive/topic1/pr6_template_anchoring/pr6_template_endpoint_anchoring_plan_2026-04-25.md`。新主线的已知风险：(a) cohort audit 后实际 n 可能 < 10（Yuquan SOZ JSON coverage 限）→ 需要先看 audit 再判 PASS/null；(b) Epilepsiae focus_rel `e` 在多 subject 上 list 为空 → H3 的 negative control 可能 underpowered；(c) split-half 通道稳定性 Jaccard < 0.4 时 H1 解读必须加 caveat。
- **未来模型层尚未启动**：硬前置见 §7.9；当前数据发现序列尚未给出已封板的几何/招募一致性结论，模型层维持冻结，不进主线工作量

---

## 6. 传播模板受慢调控的三类读数框架

PR-4 系列的核心问题：**固定传播模板受到什么慢调控？**

| 读数类别 | 科学问题 | 因变量 | 核心指标 | 数据来源 | 当前状态 |
|---|---|---|---|---|---|
| **模板混合 / 模式选择** | 慢调制改变了哪个模板更常出现 / 总 rate 由哪个模板贡献？ | 占比或固定模板分解后绝对事件率 | occupancy fraction（PR-4A）、`rate×type` envelope + stacked count（PR-4D） | `lagPatRank` cluster labels | PR-4A / PR-4D 完成 |
| **模板内顺序一致性** | 慢调制改变了模板内部 rank 一致性？ | within-cluster `τ` | Pairwise Kendall `τ` (high vs low rate) | `lagPatRank` | PR-4B 完成（null） |
| **模板内相对时延结构** | 慢调制改变了模板内部相对时延几何？ | within-cluster lag span / Pearson `r` | fixed-template 内的 relative-lag 统计 | `lagPatRaw` | PR-4B 完成（HC 子集探索性显著） |

三类读数的**数据合同**（per-event min-subtraction、`min_participating ≥ 5` 的 Pearson r 门槛、为何不能用 MI、为何不重新聚类、L3 读数的灵敏度论证）已写入：

- `.cursor/rules/topic1-within-event-dynamics.mdc` "L1/L2/L3 three-layer modulation guardrails"
- `docs/archive/topic1/propagation/interictal_group_event_internal_propagation.md`

主文档不再重述。

---

## 7. 推荐的下一步验证

### 7.1 PR-4B：Rate state × stereotype coupling

**状态**：DONE（2026-04-14）。

**结论摘要**：
- L1（模板混合）null：dominant cluster ρ median = −0.083，13/30 正方向
- L2（模板内顺序一致性）null：raw τ p=0.349、centered τ p=0.221（86% identity-bias 限制灵敏度）
- L3 lag span 全 cohort（n=30）：p=0.135，方向一致；高置信子集（n=8）p=0.055
- L3 Pearson r 高置信子集（n=8, dom_r>0.7）：p=0.016, 7/8（**唯一显著**，但功效极有限）
- 跨指标同向性：Spearman(lag_span_Δ, pearson_r_Δ) = 0.628（p=0.0003），24/29 同号
- 综合：H2 在高置信子集**探索性支持**，全 cohort **证据不足**

完整 Step 0 / Step 1 / Step 2-3 数值表与决策合同：archive `interictal_group_event_internal_propagation.md`。

### 7.2 PR-4C：Seizure proximity 双轨口径

**状态**：DONE / CLOSED（2026-04-19）。

**结论摘要**：主 + 辅两配置全量已跑两轮。第一轮（2026-04-17）识别三处合同问题（pair-wise window usability / 候选枚举式事件归属 / gap-aware rate denominator）；第二轮（2026-04-19）完成 P0 TDD 修复后复跑（n_usable_windows 主 187→360、辅 245→370）。

- **传播模式五指标**：cohort Wilcoxon 主 1/15、辅 1/15 名义显著且跨配置不一致 → **模板内部几何无稳健发作邻近调制，正式封板为阴性**
- **唯一稳健信号**：`rate_by_template` post_ictal vs baseline 主 p=0.0009、辅 p=0.0067，方向一致

完整数值、合同问题分析、P0 修复细节：archive `pr4c_seizure_proximity_review_2026-04-17.md` §9。该信号已被 §7.8 PR-5 正式化。

### 7.3 PR-4D：Template-rate decomposition

**状态**：DONE / ACCEPTED（2026-04-16）。PR-4A 的描述层补强。

**结论摘要**：
- 不再"平滑 occupancy"。正式读数只有一个：**固定模板分解后绝对事件率**（`rate×type`）
- 每个 subject 一张图：上面板 smoothed rate envelope（events/hour），下面板同色离散计数堆叠柱
- 暴露分母按真实 `coverage_ranges` 计算；gap 必须留白
- cohort dominant template rate fraction 中位数 `0.584`（range `0.262–0.866`）；`25/30` subject 至少出现一次主导模板交叉，`17/30` 在中高 rate 区间出现，`6/30` 反复交叉
- **PR-4D 是描述层，不能单独升级为推断结论**

图契约与 guardrails 见规则文件 "PR-4D gap-aware rate×type guardrails"。

### 7.4 高 k subject 鲁棒性复核

- 对 `818`、`zhangjinhan` 做 `n_participating` 匹配子样本、raw/centered 双版本模板比较
- 与 PR-4B / PR-5 并行，不阻塞

### 7.5 优先级

1. ~~**PR-4B**（P0）~~ — DONE（2026-04-14）
2. ~~**PR-4C**（P0）~~ — DONE / CLOSED（2026-04-19）
3. ~~**PR-4D**（P1）~~ — DONE / ACCEPTED（2026-04-16）
4. **高 k 复核**（P1）：可并行
5. ~~**PR-5**（P0）~~ — DONE（2026-04-20，PR-5-A PASS + PR-5-B STRONG）；详见 §7.8
6. **PR-6 Stable Template Endpoint Anatomical Anchoring**（**P0，2026-04-25 重启**）：原 PR-6-A 三份 doc 冻结归档；新主线问 "stable template endpoint (source ∪ sink) 是否解剖锚定 SOZ / focus_rel-i"；详见 §7.10 + `docs/archive/topic1/pr6_template_anchoring/pr6_template_endpoint_anchoring_plan_2026-04-25.md`
7. **PR-7 Template Antagonistic Temporal Pairing**（**P0，2026-04-28 立项**）：检验 forward/reverse template 在短时间尺度上是否构成"拮抗性配对"（Ping-Pong 假说的功能耦合层）。Pre-registered triple-gate PASS（10s Wilcoxon + 10s sign + 30s 同方向）；主 null = N2 local-window 30 min；详见 §7.11 + `docs/archive/topic1/pr7_template_pairing/pr7_template_antagonistic_pairing_plan_2026-04-28.md` + `docs/archive/topic1/ping_pong_hypothesis_review_2026-04-28.md`
8. **§7.6 / §7.7 可选方向**：§7.6 已被 PR-5 吸收为正式分析，§7.7 仍维持 exploratory 子集
9. **未来模型层（§7.9）**：硬前置未达成，维持冻结；不绑 PR 编号

---

### 7.6 后续可选方向：模板招募频率（Topic 1 × Topic 2，已升级为 PR-5 正式分析）

> 状态：**已升级为 PR-5 正式分析**（2026-04-20）。本节保留科学问题陈述与边界，正式合同与判定见 §7.8 + archive `pr5_template_recruitment_plan_2026-04-20.md`。

**科学问题**：PR-4C 唯一稳健发作邻近信号在 `rate_by_template` 层（主 p=0.0009 / 辅 p=0.0067，方向一致）。这指向"哪个模板被招募的频率随发作邻近变化"，而非"模板内部几何如何变形"。

**判定边界**：本方向**不**回答模板内部几何（PR-4C 主分析回答 → 几何已封板 null）；只回答 recruitment frequency。

### 7.7 后续可选方向：高置信子集的窄 exploratory 分支（Topic 1 内部）

> 状态：可选 / 后续；PR-4C P0 已 CLOSED。本方向是**全 cohort 阴性已封板**之后的 exploratory 子集分析，仅辅助叙事，不作为主结论。

**科学问题**：在 8/30 dominant_r > 0.7 高置信 subject 与 8/9 forward/reverse 跨时间复现 subject 上做条件性发作邻近分析，作为机制性 case-series 呈现。

**最小工作合同**：
- subset 定义：(a) `dominant_r > 0.7`（n=8）；(b) `inter-cluster r < -0.5` 且跨时间复现的 forward/reverse 对（n=8/9）
- 复用 PR-4C 修复后主读数（`lag_span` + `pearson_r`），不引入新 metric
- 单独报告 subject-level paired 比较 + 个案展示，不做 cohort-level inferential 论述
- 必须写明 selection criterion 是 Step 0 / PR-2.5 数据质量门槛，避免 cherry-picking
- 不替代 PR-4C 主分析的 cohort 阴性结论

### 7.8 PR-5：Template Recruitment Around Seizures

> 完整合同（数据/假设/失败合同/代码入口/测试合同/工作量）+ §11 复跑结论：`docs/archive/topic1/pr5_template_recruitment/pr5_template_recruitment_plan_2026-04-20.md`
> PR-5-A gate 中间报告：`docs/archive/topic1/pr5_template_recruitment/pr5a_novel_template_gate_2026-04-20.md`
> 性质：正式入口。本节只保留**判定摘要**，不重述阈值与 metric 定义（避免与 archive 双源漂移）。

**当前状态（2026-04-20 复跑）**：

| 子 PR | 结论 | 关键数 |
|---|---|---|
| PR-5-A novel-template gate | `overall_pass=True` | retained main n=23 / aux n=22；max \|Δr\|=0.0088、max \|Δe/e_baseline\|=0.0169；六条对比全部满足 archive §3.5 写死阈值 |
| PR-5-B recruitment shift | `overall_strong=True` | retained main n=23 / aux n=22；候选 A `dominant_global` `post_minus_baseline` median main `+65.46` / aux `+42.43` events/h（main p=0.00128 Bonferroni-pass α=0.0083；aux p=0.0115 nominal-pass，方向一致）；候选 B `dominant_per_window` main p=0.00214 nominal-pass 同向 → §4.4 sensitivity gate 三态判定 **strong** |
| §4.5 composition diagnostic | 不复制 panel d | `share_post_minus_baseline` main p=0.0149 / aux p=0.0301，且方向**与 panel d 预期反向**（中位数 `+0.0156 / +0.0328`；direction-consistent 仅 `6/23` 与 `5/22`）→ panel d 未在 PR-5 合同下复制，且不能联动主结论 |

**已知边界（按 archive §7 风险写死）**：
- `pre_minus_baseline` 与 `post_minus_pre` 主配置未通过 Bonferroni → 维持次级描述层
- 高对称 subject `dom_agreement < 0.5`（main 7 / aux 5）按 sensitivity gate `medium` 判读路径，不剔除
- share 与 absolute rate 数学耦合 → 机制层不展开，留未来模型层（§7.9）

**与 PR-4C / PR-4D 的边界**：PR-4C 五指标几何 cohort null 保持封板；PR-4D `rate×type` 描述层保持原状。PR-5 不涉及 SOZ 解剖锚定（属于 Topic 3 §7 独立 P1 候选）。PR-4 PPT panel d（`scripts/plot_topic1_pr4_ppt.py` fig 2d）已在 `docs/archive/topic1/propagation/topic1_pr4_ppt_figures.md` 与 archive plan §6 同步降级为"历史 motivation / 描述层"，正式归属一律指向 §4.5。

**验收意见（2026-04-20）**：

| 子 PR | 验收 | 关键证据 |
|---|---|---|
| **PR-5-A** | ✅ PASS / ACCEPTED | `overall_pass=True`；六条 gate 对比全部满足 archive §3.5 写死阈值（max \|Δr\|=0.0088，max \|Δe/e_baseline\|=0.0169，所有 Wilcoxon p ≥ 0.21）；retained subset 透明记录；49/49 测试通过 |
| **PR-5-B** | ✅ STRONG / ACCEPTED | `overall_strong=True`；候选 A 主配置 Bonferroni-pass + 辅配置同向 nominal-pass；候选 B 主配置同向 nominal-pass；composition diagnostic 反向结果按 §4.5.3 第二条诚实记录、不联动主结论；cohort filtering 按 retained subset per config 严格执行（regression test 锁死） |

**核心科学结论一句话**：剔除 novel-template / assignment-drift 假象后，dominant template 在 post-ictal 相对 baseline 的**绝对招募率**系统抬升（main +65 ev/h Bonferroni-pass，aux +42 ev/h nominal-pass 同向）；其相对**占比**变化在 PR-5 合同下不复制 panel d，所以这是"绝对招募整体增多、模板间相对权重不偏向 non-dominant"的画面，而**不是** "non-dominant template emergence"。

### 7.9 未来模型层（不绑 PR 编号）

> 性质：未来候选模型方向，不占任何 PR 编号空间，不进数据发现序列主线。本节只锁定模型层启动条件与排除清单；与数据发现序列的边界由 §7.5 优先级与 §7.10 起的数据发现条目共同界定。

**硬启动条件（模型层尚未启动）**：
- 数据发现序列必须先在主文档形成至少一条已封板的几何/招募一致性结论；
- 数据合同冻结后才允许讨论"框架是否能解释现象"；
- 在数据层证据未就位之前，模型层维持冻结，不进任何主线工作量与排期。

**为什么不在数据发现序列内做**：老一代多模板 Hebbian 框架（2024）已经把"框架能解释现象"做完。继续沿同一框架重跑没有论文价值边际。要做下一代框架，必须以 §7.1–§7.8 已钉死的事实 + 数据发现序列通过后的结果做强约束。

**几何 / manifold / persistent homology 方向**：不进数据发现序列主线；当前不立项，仅留作模型层启动后再评估的可选讨论层。

---

### 7.10 PR-6：Stable Template Endpoint Anatomical Anchoring（Topic 1 × Topic 3 桥）

> 完整合同（数据/假设/失败合同/代码入口/测试合同/工作量）+ §11 复跑结论：`docs/archive/topic1/pr6_template_anchoring/pr6_template_endpoint_anchoring_plan_2026-04-25.md`
> 性质：正式入口。本节只保留**判定摘要 + pivot 来源**，不重述阈值与 metric 定义（避免与 archive 双源漂移）。

**Pivot 来源（2026-04-25）**：原 PR-6-A multi-anchor consensus / ictal-onset alignment 已冻结归档：
- `docs/archive/topic1/pr6_template_anchoring/pr6a-1.md` — multi-anchor consensus probe 计划（5 anchor voting）
- `docs/archive/topic1/pr6_template_anchoring/pr6a_template_ictal_alignment_plan_2026-04-21.md` — single ictal anchor 主线
- `docs/archive/topic1/pr6_template_anchoring/pr6a_step0-2_step3preview_review_2026-04-23.md` — Step3 `t_ER_onset` preview-only 审阅
- `docs/archive/topic1/pr6_template_anchoring/pr6_direction_brainstorm_2026-04-25.md` — pivot 决策的 brainstorm（Obs 1–4 + 文献分类 + Topic 1 × Topic 3 桥的提出）

三份 PR-6-A doc 顶部已加 `> SUPERSEDED 2026-04-25` 块。Pivot 的实证基础：sentinel `548/916` 跨 seizure 顶部通道 overlap=0、cross-band ρ=−0.21、early channels 被 `other` 主导；文献基础：Schroeder 2020 / Wenzel 2017 / Pinto 2023 / Bailey 2021 一起说 “稳定 ictal anchor 在领域里已知不 work”。

**新主线一句话**：把每个 stable template centroid rank 的 **endpoint (source ∪ sink, top-3 + bottom-3)** vs **middle (其余通道)** 的 SOZ / focus_rel 富集差异折叠成 subject-level delta，跨 cohort 跑 Wilcoxon。Forward/reverse subject 上的 polarity 抵消问题被 endpoint 框架自动消除；source vs sink 极性只作 H1b secondary 描述。

**假设结构（archive §3）**：
- **H1 primary**：`delta_subject = mean_k(frac_SOZ_endpoint − frac_SOZ_middle)`，pooled (Yuquan + Epilepsiae) Wilcoxon vs 0，α=0.05；PASS = Wilcoxon p<0.05 + sign test p<0.05 + cohort delta median > 0
- **H1b secondary**：source vs sink polarity，**仅在 non-forward/reverse subset** 上跑（forward/reverse subject 极性相消，会假阴）
- **H2 mechanism sanity**：8/9 forward/reverse subject 上 `Jaccard(source_T0, sink_T1)` + permutation null
- **H3 sensitivity**（仅 Epilepsiae）：focus_rel i / l / e 三套 endpoint vs middle，i 主预期、l 次预期、e 应 ≈0
- **Dataset-specific**：Yuquan、Epilepsiae 各自 Wilcoxon，仅作 robustness，不进 H1 α 池

**Cohort 处理**：audit-derived，不预写 N。入选条件 = stable_k=2 ∩ SOZ JSON 非空 ∩ n_ch ≥ 6 ∩ centroid 有 polarity ∩ matched_SOZ ≥ 1。Step 2 第一动作是出 `cohort_audit.csv`。`818` (k=4) 与 `zhangjinhan` (k=6) 走 case-series。

**复用基础设施**：
- `adaptive_cluster.clusters[k].template_rank`（`src/interictal_propagation.py:1536`）— argsort-of-argsort 整数 rank 已计算好
- `match_bipolar_soz` / `match_bipolar_focus_rel`（`src/event_periodicity.py:3153,3164`）— Yuquan bipolar endpoint matching + Epilepsiae CAR 直接匹配
- `compute_time_split_reproducibility`（仅扩展 ~20 行存 `cluster_rank_a/b`）

**显式不做**：不重启 ER pipeline / CUSUM / Page-Hinkley / `t_ER_onset`；不引入 multi-anchor voting / naming label；不重跑 PR-2 / PR-2.5 / PR-3 / PR-4A/B / PR-5；不做 π embedding；不做"先挑 hub 再重跑 PR-4C"的 double-dipping replay（留给独立后续 PR）。

**当前状态（2026-04-27 综合更新）**：plan-of-record 已落盘；Step 1 ACCEPTED（2026-04-26，78 测试全绿）；Step 2 / 4 / 4b / 5a / 5b / 3 全部 preliminary 验收。综合科学叙事：

> Stable interictal HFO group-event template pairs show **swap-leaning node geometry detectable cohort-wide on Wilcoxon (h1-eligible n=21, swap_node count − same_side_node count median +2, p=0.012) but failing the more conservative sign-test (p=0.12)**. The cohort-wide effect is largely driven by the forward/reverse-reproduced subset (n=6, 6/6 positive, sign-test p=0.031), translating PR-2.5 cluster-rank anticorrelation into node-level swap counts. The non-forward/reverse subset alone (n=16) does not reach cohort significance (Wilcoxon p=0.18) and spans a heterogeneous spectrum from no-swap (subject 590, all template-specific endpoints) to partial local swap. **The swap-vs-same geometry is not explained by subject-level SOZ enrichment** (h1-eligible n=19, SOZ frac in swap-nodes minus SOZ frac in template-specific endpoints: median Δ=0, Wilcoxon-greater p=0.19, sign-test 9p/4n/6z); pooled SOZ fractions are channel-count-weighted aggregates, not valid cohort claims. Endpoint geometry is time-stable within a fixed metric (Step 5b: split-half median Jaccard 0.71, odd-even 0.93), but H1 SOZ-anchoring direction is not robust under endpoint metric change (Step 5a: 7/20 direction-discordant + 1 one-is-zero between top-3 vs coreness-top-20%). **Single-metric H1 cannot determine whether stable templates anchor clinical SOZ; the more robust observation is the structured node-level pair geometry, which is itself not explained by clinical SOZ annotation.**

**Cohort 与方法层数字**：
- Audit-derived 主 cohort **n=21（13 epilepsiae + 8 yuquan）**；2 个 n_ch=6 case-series；4 个 SOZ 缺失退出；3 个 k≠2 退出
- H1 pooled n=21 Wilcoxon p=0.42（NULL）；coreness top-20% sensitivity (n=20) Wilcoxon p=0.140（同 NULL，但 7 subject direction-discordant、1 subject one-is-zero）→ endpoint 主定义不 robust
- Step 4b node-level：4 分层（all_endpoint_defined / h1_eligible / forward_reverse_reproduced / non_forward_reverse_h1_eligible）的 subject-level swap−same paired Wilcoxon + sign-test 都已落盘，结果如上
- Step 5b time-stability：split-half endpoint Jaccard median 0.71，odd-even 0.93；source/sink 各自分别报告；稳定性是在 rank-position endpoint 这一定义内部成立，仅限 lagPat/high-HI 覆盖到的节点集合
- Step 3 figures：6-panel 主图 + 3 张 supp 已生成（`results/interictal_propagation/template_anchoring/figures/`），主图中心是 template-pair geometry 而非 H1 SOZ null

**论文级 framing 收紧**（2026-04-27）：
- ✅ 可以说："stereotypy 内部存在结构化的 pair-level 几何（fwd/rev = 干净 swap，non-fwdrev = 弱混合 spectrum）；这套结构在 cohort 上 partially significant，且不能由 subject-level SOZ 富集解释"
- ❌ 不能说："stereotyped HFO timing 锚定 SOZ"（H1 cohort NULL）；"swap geometry 与 SOZ 无关"（仅检验了 swap vs template-specific）；"non-fwdrev = 独立网络"（实际是渐变 spectrum）
- 与 paper 主论点"间期事件刻板时序能否成为癫痫病理网络指示器"的差距：当前能描述 stereotypy 的几何结构，但不能直接断言它锚定 clinical SOZ；论文 framing 应转向"stereotypy 是稳定可测的 pair-level 几何，但与 clinical SOZ 标签不简单同构 → 需要更精细的 anatomy 检验（onset 通道 / ictal 传播路径）"

**最值得做的下一步验证（plan §15 Step 6 详细列表）**：
1. **Endpoint vs per-seizure onset channel set** Jaccard / hit-rate（subject-level Wilcoxon）— 用真实 onset 通道而非 SOZ JSON 二值化，预算 0.5 d（最高优先）
2. **Held-out time validation**：前 50% 时间训练 template、后 50% 跑 endpoint anchoring 检验，避免 PR-2 + PR-6 同数据 double-dipping，预算 0.5 d
3. **PR-4B HC subset (n=8) × PR-6 endpoint** 联动：高放电期 dominant cluster 的 endpoint 是否更靠 SOZ，预算 1 d
4. **fwd/rev subject 的 ictal propagation 直接对比**：reuse Topic 3 PR-2.5 onset estimation，把 T0 source / T1 sink 与真实 ictal LFP 传播方向比对，预算 1.5 d 起步
5. **pre-ictal vs baseline endpoint anatomy**：reuse PR-2.7 seizure-triggered window，把 endpoint 集合在 pre-ictal vs baseline 各自建 template，看通道集合是否偏移

**显式不做**：不在当前数据上重新调 endpoint 定义（top-3 / coreness / median 之外）；不重做 PR-2/3/4/4B 核心 cluster pipeline；不引入 ictal anchor / ER / CUSUM。

**Step 6 / archive results doc 推迟**：等 §1 或 §2 中的下一步验证有结果后再写正式归档 doc 与主文档一句话回写；当前 PR-6 主线（Step 1–5b + 4 + 4b + 3）所有结果归档到 `docs/archive/topic1/pr6_template_anchoring/pr6_template_endpoint_anchoring_plan_2026-04-25.md` §15。

**Step 6 plan-of-record 落盘（2026-05-10）**：上述下一步验证清单的第 2 项"Held-out time validation"独立成档为 `docs/archive/topic1/pr6_template_anchoring/pr6_step6_held_out_template_plan_2026-05-10.md`。**严格 train/test 不对称**：first half 定 cluster + endpoint channels（具体集合，不只是规则），second half 只投射 + 验证，不重新发明 endpoint。Held-out burden 压在 endpoint geometric stability + §8 swap_class concordance 上（H1 在 ε_first 固定下结构上不变）。Tier = robustness / sensitivity，不开新 cohort claim。

**Step 6 cohort 结果（2026-05-10）**：n=35 stable_k=2 subject（PR-6 main 23 ∪ rank_displacement v14 sensitivity n=35 去重）跑完。Tier 分布：strong 20 / moderate 13 / weak 2 / fail 0；cohort 中位数 `template_spearman` = 0.922、`endpoint_position_recall` = 0.833、`cluster_assignment_purity` = 1.000、swap_class concordant = 24/35 = 68.6%。**Null calibration（advisor 2026-05-10 post-hoc，epilepsiae_548 per-event 二半 rank-shuffle 50 trials）**揭示 pipeline 有结构性 selection-bias 底：`template_spearman` null median = **0.747**，`endpoint_position_recall` null median = **0.667**，`swap_class_concordant` null = **36%**（18/50）。重新表述：cohort 信号是 spearman gain +0.17、recall gain +0.17、**swap_class concordance gain +33 pp**——`swap_class_concordant` 实际上是三量里最 discriminative 的判定（与 raw count 直觉相反），对应 §8 dual-tier swap geometry 在 hold-out 下的真实辨识力。**结论仍然成立但措辞收紧**：PR-6 主线 H1 cohort NULL 与 §8 dual-tier swap_class 在论文记录窗口内 above pipeline null floor；不是 PR-2 + PR-6 same-events double-dipping artifact，但**也不是"几乎完美 hold-out"**。Plan §7.1 的 strong-tier 阈值 (spearman > 0.7、recall > 0.6) 跨在 null 分布之上，"strong 20"计数高估实际信号——tier 计数保留作 plan-of-record 一致性，但 paper-level 表述应基于 calibrated gains。strong tier 中 12/20 (60%) 是 §8 swap_class = none —— hold-out 稳健性不依赖 swap geometry，与 H1 cohort NULL 同向。weak 2 subject (`yuquan_litengsheng`、`yuquan_zhaochenxi`) 走 case-series；`zhaochenxi` 的 §8 strict 标签在 hold-out 下推翻为 none，提示 single-fold pattern。**Scope caveats**：(a) 本 Step 只覆盖论文记录窗口内（Yuquan 24h / Epilepsiae 数日）within-recording stationarity，**不是** H_drift 全盘否定；(b) null calibration 仅一个 subject，per-subject 异质性未覆盖；严格 tier rebalancing 需 per-subject null。详见 `docs/archive/topic1/pr6_template_anchoring/pr6_step6_held_out_template_results_2026-05-10.md` §4.6。

**PR-6-sup1 plan-of-record 落盘（2026-05-10，Topic 4 mechanism preflight）**：First-rank entropy / symmetry-breaking diagnostic 独立成档为 `docs/archive/topic1/pr6_template_anchoring/pr6_supplementary_rank_entropy_plan_2026-05-10.md`。问 "iHGE 事件 rank vector 在 endpoint position 是否比 middle position 更不稳定"（Liou-Abbott 风 noise-driven confluence-point sharp prediction）。**与 Step 6 数据流分开**：sup1 不切时间、不依赖 endpoint 集合 / SOZ；§8 swap_class 仅作 stratifier。Tier = Topic 4 preflight，descriptive only，不进 paper α，不进 PR-6 主线。

**Continuous-version supplementary（2026-05-06）**：把 PR-6 Step 4b 离散 swap_node count 升级到逐通道 signed rank displacement（Δr = rank_T_b − rank_T_a），加 Spearman footrule + Diaconis-Graham 归一化 + Kendall τ + baseline-corrected SOZ contribution split。**不**立独立 PR、**不**开 cohort gate。**Cohort = 全部 27 个 stable_k=2 subject**（PR-2 stable_k 分布 `{2:27, 4:2, 6:1}`）。valid_mask 优先用 PR-6 `valid_mask`（23 subject），PR-6 缺失时 fallback 到 PR-2 `(template_rank ≠ −1)` sentinel（4 subject：`1125/384/620/916`，PR-6 SOZ-空过滤掉的）。两 provenance 在共有 subject 上完全一致。**Paper-level deliverable 是单张 composite supplementary figure**：cohort heatmap 主体（行按 F_norm 降序，列按 rank_T_a_dense，SOZ 黑框，x-axis 在上方）+ 一条 F_norm summary track（带 2/3 ref）+ 主热图正下方水平 Δr colorbar + colorbar 旁的 SOZ channel mini-legend。Kendall τ 与 SOZ contribution_excess **均不进 paper figure**：τ 与 F_norm 在本 cohort Spearman ρ = −0.92 共线，track 冗余；SOZ 在 lagPat 通道集上覆盖与 i/l/e 边界尚未稳定。τ 与 SOZ 数字均保留在 archive 作 descriptive cross-check。结论：(a) cohort 在 F_norm 范围 [0.39, 1.00] 上呈连续谱，不是离散二分（n=27，median F_norm=0.81）；(b) 与 PR-6 离散 swap_node 同向（PR-2.5 fwd/rev-reproduced 8 subject 的 Kendall τ 全部 < 0；其中 PR-6 cohort n=6 sign-test p=0.031）。新加入的 4 个 subject 中 `epi_620` (F=1.00, τ=−0.78) 与 `epi_1125` (F=0.88, τ=−0.71) 都是 PR-2.5 fwd/rev-reproduced subject——之前因 PR-6 SOZ-空过滤错失。详见 `docs/archive/topic1/pr6_supplementary_rank_displacement_plan_2026-05-06.md` 与 `pr6_supplementary_rank_displacement_results_2026-05-06.md`。

**Variable-k swap classifier（2026-05-07，supplementary §8，v3 dual-tier）**：在 continuous-version 之上加一个二级 deliverable，把 PR-6 H2 swap_score 公式从 hard-coded `k=3` 改成数据驱动扫描 `k ∈ {2 .. ⌊n_valid/2⌋}`，用 family-wise max-null（1000 perm，seed=0）做单一统计量检验：`T_obs = max_k swap_score(k)`。**双轨 decision rule**：`swap_class = strict iff T_obs ≥ 0.5 AND p_fw < 0.05`（n=10/35），`candidate iff T_obs ≥ 0.5 AND 0.05 ≤ p_fw < 0.20`（n=8/35），`none` 其他（n=17/35）。strict ∪ candidate = 18/35 ≈ 51% subject 进入 label tier。**仍是 mechanism-sanity tier，不开新 cohort claim**；channel-level label 只 strict tier 允许且必须 split-half 验证（详见 §8.7 channel-label 合同）。strict 10 个里 8 个落在 PR-2.5 fwd/rev-reproduced，2 个 PR-2.5 非候选通过（`epi_1146 dk=7, yuq_hanyuxuan dk=6`）；candidate 8 个里 3 个是 PR-2.5 fwd/rev-reproduced 但 FW 力不足（`epi_1125, epi_548, yuq_chenziyang`）。user motivation "2 节点 swap" 在 strict tier 无显著支持，在 candidate tier 有 2 例 exploratory 信号（`epi_1125 dk=2, epi_384 dk=2`）。**Variable-k swap markers 已整合进主图 `cohort_displacement_heatmap.{png,pdf}`**：每行在 column = decision_k − 1 与 column = n_valid − decision_k 两个 cell 上分别画 `>` 与 `<` 三角（`> ... <` = inward swap）；strict = 实心黑三角，candidate = 空心灰三角；读者用"离边几格"读 decision_k。Legend 在 suptitle 与 heatmap 之间的顶部条带，2/3 random-null 红虚线加粗。Supplementary 详图 `swap_cardinality_heatmap.{png,pdf}` 给出 subject × k swap_score 全图。**PR-2.5 fwd/rev-reproduced 不是 ground truth**，分歧不应解读为 classifier "validation"。详见 archive doc §8。

**Topic 1 论文第一部分空间收束 — Swap × Clinical SOZ Set-Relationship（2026-05-08，supplementary §9）**：在 §8 dual-tier swap_class 之上做 lagPat universe 内的 set-relationship 描述，把 swap_endpoint 与 clinical SOZ 的几何关系作为"间期刻板时序是癫痫病理网络指示器"这条主线的最后一块**空间证据**。**主统计形态从"cohort p-value"改为"cohort typology distribution + sign test on enrichment_over_lagPat"**，避开 advisor 抓到的 paired Δ n_middle 结构性退化（strict tier `decision_k ≈ ⌊n_valid/2⌋`，coverage→1 → middle 挤空）。Per-pair schema：`precision = |E ∩ S| / |E|, recall_within_lagPat = |E ∩ S| / |S|, coverage = |E| / |L|, lagpat_baseline = |S| / |L|, enrichment_over_lagPat = precision − lagpat_baseline, typology ∈ {E_subset_S, S_subset_E, partial, disjoint, degenerate}`。informative gate `0 < n_S < n_L AND n_E < n_L`。**Primary cohort = strict ∩ informative (n=5)**：sign test p=0.500，median enrichment +0.042，bootstrap 95% CI [−0.071, +0.098] —— **NULL，与 PR-6 H1 hard-coded `k=3` Wilcoxon p=0.19 NULL prior 一致**，refined dual-tier swap_class 没把 NULL 翻号。**Sensitivity = candidate ∩ informative (n=5)**：4/5 positive，median +0.127，CI [0.000, +0.167] —— underpowered 但方向上正向。**Typology descriptive 主读数（不算 cohort claim）**：`E_subset_S` (swap ⊊ SOZ, "refined SOZ candidate") **全 cohort = 0**；`S_subset_E` (swap ⊋ SOZ, "extended SOZ candidate") strict 1 + candidate 3 = 4 个；`partial` 多数。方向上 swap_endpoint **倾向"覆盖且包含"clinical SOZ 而非"被包含于"**，与 user 设计阶段 "swap 范围比临床更大可能是补充信息" 直觉一致。**Channel-selection circular caveat**：lagPat 已对 SOZ 富集（high-HI / high-HFO-rate gate），本 §9 量化的是 high-HI 区内 swap 是否进一步富集 SOZ，**不是**全脑关系；任何 paper-level 表述必须紧邻数字带 caveat。Paper-level deliverable: `swap_clinical_soz_set_relation.{png,pdf}` 3-panel（A precision×recall scatter + B enrichment×coverage scatter + C typology stacked bar）。M1=HFO-onset rate / M2=ER-rank multi-source sensitivity 等 PR-T3-1 producer 重设计后单独补 §9.5；outcome × partition 检验 (HFO rate / detrend_fraction / cluster participation × 4-cell) 属于 Topic 1 Part 2 演化故事 PR，不在本 §9。详见 `docs/archive/topic1/pr6_supplementary_swap_clinical_soz_plan_2026-05-08.md` 与 `pr6_supplementary_rank_displacement_results_2026-05-06.md` §9。

---

### 7.11 PR-7：Template Antagonistic Temporal Pairing（Ping-Pong 功能耦合层）

> 完整合同（数据/假设/失败合同/代码入口/测试合同/可视化方案/工作量）：`docs/archive/topic1/pr7_template_pairing/pr7_template_antagonistic_pairing_plan_2026-04-28.md`
> 整体假说审阅与 PR roadmap：`docs/archive/topic1/ping_pong_hypothesis_review_2026-04-28.md`
> 性质：正式入口，**plan-of-record 已落盘 2026-04-28**。本节只保留**判定摘要**与 pivot 来源，不重述阈值与 metric 定义（避免与 archive 双源漂移）。

**科学问题**：PR-2.5 + PR-6 Step 4 已经稳健建立 forward/reverse template 的现象学（n=6 fwd/rev reproduced subject，节点级 source/sink swap geometry sign-test p=0.031）。这建立了**现象学层 (A)**，但没有回答**功能耦合层 (B)**：T_a 与 T_b 是不是在时间上配对出现？如果它们时间上独立，"两类反向模板"只是同一群体事件流的统计压缩，**Ping-Pong 因果叙事不成立**。

PR-7 是 Ping-Pong 假说能否成立的最直接可证伪检验。失败（lift ≈ 1 across all Δt）即关闭机制叙事；通过则功能耦合层得到支持，可向 inhibitory restraint / rebound 文献借用机制语言（**仅作文献一致性，HFO 80–250 Hz 不区分 E/I**）。

**假设结构**（archive §3）：
- **H1 primary triple gate**：`excess(10s)` Wilcoxon p<0.05 **AND** sign p<0.05 **AND** `excess(30s)` 中位数 > 0
- **H1b**：direction symmetry（T_a→T_b 与 T_b→T_a 应对称）
- **H2 negative control**：non-fwdrev cohort `excess(10s)` Wilcoxon p > 0.10
- **Secondary**：next-event transition odds + time-to-next（描述层，不进 α 池）

**Cohort（pre-registered）**：endpoint_defined ∩ forward_reverse_reproduced ∩ n_events≥300 ∩ min_cluster_n≥75 ∩ (n_blocks≥3 OR coverage≥6h)；预期 4–6（PR-6 §15 H2 cohort 6 是上界）；**门槛不放宽**，n<4 走 case-series。

**Surrogate hierarchy**：N2 local-window shuffle (30 min) 主 null；N3 circular shift robustness；N0/N1 sanity；N4 rate-matched ISI 仅在 N2/N3 不一致时 conditional follow-up。

**显式不做**：不重做 cluster 算法（KMeans k=2 锁死）；不做节点级 signed displacement（→ PR-8 candidate）；不做 subject typology × PR-5 split（→ PR-9 candidate）；不做发作邻近窗口（PR-4C 已封板 null）。

**当前状态（2026-04-30，Step 0–6 收口）**：

**最终结论（locked）**：**几何上相关，已测试时间尺度上未见 mark dependence。**

PR-6 已建立 forward/reverse template 共享同一网络几何（n=6 sign-test p=0.031，source/sink swap）；PR-7 在三类 metric 上检验 mark dependence：(i) event-level fixed-window opposite-template excess at Δt ∈ [10s, 30s]（Step 3）；(ii) event-level direction asymmetry + next-event transition odds（Step 3 secondary）；(iii) run-based persistence（无 ISI 阈值 same-label run + lag-1 + gap-to-IEI；Step 3.5 post-hoc）。三类 metric **全部 NULL**：

| 检验 | 数字 | 结果 |
|---|---|---|
| Step 3 H1 N2 主 null | Wilcoxon(10s, greater) p=0.844, sign 3/6, median(30s)=−0.015 | NULL |
| Step 3 H1 N3 robustness | Wilcoxon p=0.891, sign 2/6, median(30s)=−0.012 | NULL |
| Step 3.5 burst diagnostic（cohort N2）| run_length_lift median=0.977, lag1_same_excess median=−0.013, 2/6 above null | **未见 persistence** |
| Step 5 N2 window sweep {10/30/60 min} | Wilcoxon ∈ [0.78, 0.89], median(30s) ∈ [−0.029, −0.002], 全部 NULL | **cohort verdict robust** across window |

Step 5 注意：cohort verdict 跨 window 稳健（**不**应写"三条曲线高度重合"）；magnitude 由 548 outlier 主导，548 magnitude 跨 window 1× → 2× → 3× 单调放大（−0.10 → −0.20 → −0.30）。

**精确 framing**：在已测试 metric 上数据 **compatible with mark-independent sampling**（最简洁描述，**不等于证明独立**）。**未测**：alternative burst definitions、rate-state / seizure-proximity switching、form (4) latent-state coupling、history-dependent regression。

**禁止措辞**：
- ❌ "Two templates are time-independent / no causal coupling"
- ❌ "Mark sequences are mark-independent"
- ❌ 把 548 outlier 升级为 cohort claim
- ❌ "Burst-level reciprocal coupling restores Ping-Pong"
- ❌ 删除 PR-6 几何 narrative

详见 `docs/archive/topic1/pr7_template_pairing/pr7_template_pairing_results_2026-04-29.md` §1–§17。完整图集：`results/interictal_propagation/template_pairing/figures/`（fig1–5 + appendix 1/3）。

**下一步**（**不在 PR-7 内**）：
- 独立 follow-up PR：history-dependent marked point process model（`next_label ~ previous_label + recent_rate + time_since_last + block / state`），可一并测 form (1) + (2) + (4) 不依赖 fixed-window metric
- H2 negative control（n=17）：完整性补强；优先级低，不会改变 H1 NULL verdict

---

## 8. 代码与结果入口

### 内部传播

- 文档：`docs/archive/topic1/propagation/interictal_group_event_internal_propagation.md`
- 代码：`src/interictal_propagation.py`
- 脚本：`scripts/run_interictal_propagation.py`、`scripts/plot_interictal_propagation.py`、`scripts/plot_topic1_pr4_ppt.py`
- 结果：`results/interictal_propagation/`

### 事件级同步性

- 文档：`docs/archive/topic1/synchrony/interictal_synchrony_preliminary_report_2026-04-03.md`
- 代码：`src/interictal_synchrony.py`、`src/interictal_synchrony_aggregation.py`、`src/interictal_synchrony_analysis.py`
- 脚本：`scripts/pr6_interictal_sync_figures.py`
- 结果：`results/interictal_synchrony/analysis/combined/`

---

## 9. 与其他 topic 的边界

- "`~2 Hz` 峰是不是真的"或"IEI serial correlation 说明什么" → `docs/topic2_between_event_dynamics.md`
- "SOZ 和 non-SOZ 到底差在哪里" → `docs/topic3_spatial_soz_modulation.md`
- 同时涉及"传播是否真实"和"慢调制是否发生在 SOZ" → 先分别读 topic 1 和 topic 3，不要混成一个问题

---

## 10. 历史文档索引

- `docs/paper1_framework_sba.md` — **Paper 1 架构性 framework（最高优先级 pre-registration）**：SBA 单核心假设 + 5 sharp predictions（P1–P5）+ BHPN-toy/fit 数学合同 + 5 dumb baseline + 失败模式表 + 命名/范围/Out of scope。本 topic 的 PR-2 / PR-2.5 / PR-6 / PR-7 / 待立 PR-T4-1/T4-2 / PR-9 全部受其统辖。
- `docs/archive/topic1/pr7_template_pairing/pr7_addendum_p3_equivalence_2026-05-01.md` — PR-7 addendum：P3 cohort-level equivalence test (TOST + bootstrap CI + leave-one-out + leave-548-out)。verdict = INCONCLUSIVE；1800s + lag1_same_excess PASS，短窗 + run_length_lift CI underpowered。SBA 不被 falsified。
- `docs/archive/topic4/pr_t4_1_bhpn_toy/pr_t4_1_bhpn_toy_plan_2026-05-01.md` — **PR-T4-1 BHPN-toy plan-of-record**（继承 framework v1.1.2）：toy model 实现 + 14 项 TDD + T1–T7 内部 sanity（large-N simulation, **不**以 PR-7 实证 cohort 为 anchor）+ TT14 toy-validate framework δ_excess scientific 推理。下游 PR-T4-2 BHPN-fit 等 PR-T4-1 + PR-T3-1 完成后启动。
- `docs/archive/topic1/propagation/interictal_group_event_internal_propagation.md` — 内部传播线的详细结果与合同文档
- `docs/archive/topic1/synchrony/interictal_synchrony_preliminary_report_2026-04-03.md` — PR4–PR6 的统计报告
- `docs/archive/topic1/propagation/pr4c_seizure_proximity_review_2026-04-17.md` — PR-4C 主+辅助配置全量审阅。§1-§8 是 2026-04-17 第一轮审阅（cohort 数值表 / 三处实现合同问题 / P0/P1/P2/P3 路线）；**§9 是 2026-04-19 P0 修复完成后的复跑数值与正式封板结论**。Topic 1 §3.1c / §5 / §7.2 / §7.6 / §7.7 都引用本文件。
- `docs/archive/topic1/pr5_template_recruitment/pr5_template_recruitment_plan_2026-04-20.md` — PR-5 完整计划合同：科学问题 / 主+备择假设 / 失败合同 / PR-5-A novel-template gate / PR-5-B recruitment shift（含 §4.5 secondary composition diagnostic 独立合同）/ 9 项 TDD 测试合同 / §9 未来模型层占位（不绑 PR 编号，对应主文档 §7.9）/ §11 复跑结论。Topic 1 §5 / §7.5 / §7.6 / §7.8 / §7.9 / §10 都引用本文件。
- `docs/archive/topic1/pr5_template_recruitment/pr5a_novel_template_gate_2026-04-20.md` — PR-5-A gate 全 cohort 跑数与判定中间报告。
- `docs/archive/topic1/pr6_template_anchoring/pr6a_step0-2_step3preview_review_2026-04-23.md` — **SUPERSEDED 2026-04-25**：PR-6A Step0-2 / Step3-preview 阶段性审阅；保留作为 pivot 决策的实证证据（sentinel `548/916` 跨 seizure top10 overlap=0 / cross-band ρ=−0.21）。当前正式入口：`pr6_template_endpoint_anchoring_plan_2026-04-25.md`。
- `docs/archive/topic1/pr6_template_anchoring/pr6a_template_ictal_alignment_plan_2026-04-21.md` — **SUPERSEDED 2026-04-25**：原 PR-6-A single ictal anchor + Smith 2022 重现的合同级计划；保留作为科学背景与方法学讨论。
- `docs/archive/topic1/pr6_template_anchoring/pr6a-1.md` — **SUPERSEDED 2026-04-25**：原 PR-6-A multi-anchor consensus probe 计划（5 anchor voting）；保留作为 pivot 决策证据。
- `docs/archive/topic1/pr6_template_anchoring/pr6_direction_brainstorm_2026-04-25.md` — PR-6 pivot 决策的 brainstorm 文档：Obs 1–4 分类、文献整理、Topic 1 × Topic 3 桥的提出；驱动了 PR-6 主线从 ictal alignment 转向 endpoint anatomical anchoring。
- `docs/archive/topic1/pr6_template_anchoring/pr6_template_endpoint_anchoring_plan_2026-04-25.md` — **PR-6 正式入口（plan-of-record）**：H1 endpoint vs middle / H1b polarity / H2 forward-reverse swap / H3 i-l-e sensitivity / cohort audit-derived / 8 项 TDD / 失败合同。Topic 1 §2 / §5 / §7.5 / §7.10 / §10 都引用本文件。
- `docs/archive/topic1/ping_pong_hypothesis_review_2026-04-28.md` — Ping-Pong 假说整体审阅：三层分离（现象学 A / 功能耦合 B / 机制 C）+ user 提议实验逐项对账 + PR roadmap（PR-7 → PR-8 candidate → PR-9 candidate）。Topic 1 §7.5 / §7.11 / §10 都引用本文件。
- `docs/archive/topic1/pr6_supplementary_rank_displacement_plan_2026-05-06.md` — PR-6 supplementary plan：把离散 swap_node count 升级到逐通道 signed rank displacement + Diaconis-Graham 归一化 footrule + Kendall τ + baseline-corrected SOZ split。Pre-registered 反 sorting bias（列按 rank_T_a_dense 排）+ sign anchor 合同（Δr 仅 subject 内部有效）+ n_available 不预承诺。Topic 1 §7.10 / §10 引用本文件。
- `docs/archive/topic1/pr6_supplementary_rank_displacement_results_2026-05-06.md` — PR-6 supplementary results：cohort n=23（stable_k=2 ∩ PR-6 endpoint-defined）；reproduced n=6 (Kendall τ median = −0.495, F_norm = 0.964) vs not reproduced n=17 (τ = −0.048, F_norm = 0.688)。Reproduced 6/6 subject τ < 0，与 PR-6 离散 swap geometry 同向。SOZ contribution_excess ≈0（descriptive only，无 enrichment claim）。**§8 (2026-05-07) variable-k swap classifier dual-tier**（strict/candidate/none）+ **§9 (2026-05-08) swap × clinical SOZ set-relationship**（precision/recall/coverage/enrichment_over_lagPat + typology；strict primary NULL, candidate sensitivity directional, S_subset_E 4/35）继续在本 doc 内更新。Topic 1 §7.10 / §10 引用本文件。
- `docs/archive/topic1/pr6_supplementary_swap_clinical_soz_plan_2026-05-08.md` — §9 plan: swap × clinical SOZ set-relationship within lagPat universe (n=35)；pre-registered informative gate / dual-tier cohort / sign test on enrichment_over_lagPat + bootstrap CI / 11 TDD / 3-panel figure spec / channel-selection circular caveat 写死。Topic 1 §7.10 / §10 引用本文件。
- `docs/archive/topic1/pr7_template_pairing/pr7_template_antagonistic_pairing_plan_2026-04-28.md` — **PR-7 正式入口（plan-of-record）**：H1 triple gate (10s primary + 30s sensitivity + sign test) / H1b direction symmetry / H2 non-fwdrev negative control / N0–N4 surrogate hierarchy（N2 主 null + N3 robustness + N4 conditional）/ cohort 5 条 pre-registered 入选门槛 / 10 项 TDD / §6.5 可视化方案 / 7 类失败合同。Topic 1 §7.5 / §7.11 / §10 都引用本文件。
- `docs/archive/topic1/propagation/pr4_ppt_per_subject_iteration_summary_2026-04-20.md` — PR-4 PPT/per-subject 综合图的对话迭代记录：版式收敛、关键病例池、以及 SBCI/TRIS 新 metric 需求定义。
- `docs/archive/topic1/propagation/cluster_geometry_viz_plan_2026-05-06.md` — Cluster geometry 可视化 plan-of-record（template-matching metric / 数据合同 / 失败合同 / panel layout / TDD 合同）。Topic 1 §3.1d / §10 引用本文件。
- `docs/archive/topic1/propagation/cluster_geometry_viz_results_2026-05-06.md` — Cluster geometry 验收 results（cohort 数字 / showcase 叙事 / KMeans-vs-template-matching audit / follow-up）。Topic 1 §3.1d / §10 引用本文件。
- `docs/topic5_seizure_subtyping.md` — Topic 1 × Topic 5 cross-link：本 topic 关注 inter-ictal 群体事件内部的 within-event 时序结构；Topic 5 关注 ictal seizure 自身的 within-subject 异质性（v2.3 ER atlas + z-ER 张量子型聚类）。两 topic 的 propagation pattern / template 几何在下游 PR 可做联合分析，但当前主文档之间无强制依赖。

这些文档保留为历史事实来源；当前正式口径以本文件为准。

---

## 11. 文档整理里程碑

- **2026-04-20**：主文档大幅瘦身（442 行 → ~270 行），按"主文档只放正式口径，过程性细节回链 archive"原则重写。所有 §-锚点保留；PR-4B / PR-4C / PR-5 的完整数值表、阈值、metric 公式、测试合同、复跑过程已下沉到对应 archive。同步把 `docs/plans/` 下已完成的 yuquan_lagpat 系列（6 个文件）归档到 `docs/archive/yuquan_lagpat/`，event_periodicity Phase 2 plan 归档到 `docs/archive/topic2/plans/`；`interictal_synchrony_analysis_v4.plan.md` 仍有 pending 项，2026-05-05 整体迁入 `docs/archive/topic1/plans/`。
