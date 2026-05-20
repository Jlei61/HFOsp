# 中期 PPT 大纲 — Topic 1 主线（间期刻板时序 → swap → SOZ 互补）

> 日期：2026-05-10
> 用途：60 min 中期组会 / 委员会报告，~40 页（含已完成的 ~10 页背景）
> 结构原则：段连段无封面页；每页 1 张主图 carry 1 个科学问题；stat 收 backup；caveat 在自然位置摆出来不藏。
> Take-home 由用户撰写；本文档不写具体数字结论。

---

## 0. 背景（用户已完成，~10 页）

逻辑链：癫痫是什么 → 间期事件 vs 发作期信息（为什么探索间期结构）→ 间期事件分类 → HFO →
HFO 群体事件结构与诊断信息（high-HI 与 SOZ 的一致性，两个数据集 AUC）。

**主图入口**：
- `results/refine_soz_validation/figures/yuquan_refine_soz_cohort.png` —— SOZ AUC 验证

下游 PPT 主线即从"群体事件层面已经看到 high-HI ≈ SOZ"过渡到"那么群体事件**内部**呢？"。

---

## Part A：间期事件存在刻板时序（3-4 页）

> 这一段只回答：单事件内通道激活顺序是不是稳定的？

### A.1 概念页（schematic）

- 内容：N 通道 LFP/HFO trace → 每通道 onset → rank vector；定义 lagPatRank + n_participating ≥ k_min。
- 图源：用 `scripts/plot_topic1_pr4_ppt.py::_toy_template / _toy_event_from_template` 已有 toy-rank 函数生成 schematic（如新画一张 1×3：trace → onset → rank vector）。
- 一句话：把事件压成 channel rank，是这一整段的 atomic 数据对象。

### A.2 单 subject 视觉证据

- 主图：`results/interictal_propagation/figures/per_subject/epilepsiae_958_propagation.png`（cluster geometry showcase 中干净的 BIMODAL k=2 fwd/rev 案例）
- 一句话：数百个事件的 rank 在视觉上呈现块状一致——"刻板"不是 p-hacking。

### A.3 Cohort MI + identity-bias（合 1 页）

- 主图：`results/interictal_propagation/figures/cohort_propagation_summary.png` 的 **panel a**（MI vs permutation null）+ **panel f**（identity-bias 散点：raw τ vs centered τ）
- 双 panel 同图：左 a 证明 stereotypy 真实，右 f 主动暴露 86% bias 来自 hub identity ordering。
- 一句话（诚实口径）：stereotypy 真实存在，但其中绝大部分来自 hub 通道的固定先后激活；这是后面分析的合同。

> 提前摆 caveat 避免审稿人质疑"你做的就是 hub identity"。

---

## Part B：k=2 是间期刻板时序的主导特征（5 页）

> 这一段只回答：刻板时序是单一模板还是多模态？答：多模态，k=2 是主导压缩。

### B.1 多模态来源 — pairwise τ bimodality

- 主图：`cohort_propagation_summary.png` **panel b**（within-cluster vs between-cluster pairwise τ，dip test）
- 一句话：单模板预期单峰；现在 cohort 全员双峰 → 至少两种主要模式。

### B.2 KMeans k=2 的 within-cluster τ uplift

- 主图：`cohort_propagation_summary.png` **panel c**（散点：within τ vs overall τ，全员 above diagonal）
- 一句话：分成两簇后每簇 stereotypy 显著高于 overall——k=2 切得有意义。

### B.3 k=2 的视觉本质 — cluster geometry trilateration

- 主图：`results/interictal_propagation/cluster_geometry/figures/per_subject/` 4 张拼 2×2：
  - `epilepsiae_958`（BIMODAL k=2 fwd/rev，干净）
  - `epilepsiae_818`（BIMODAL k=4，多模态非 k=2 反例）
  - `epilepsiae_253`（AMBIGUOUS k=2 弱簇）
  - `epilepsiae_442`（UNIMODAL，1D 连续谱反例）
- 一段：多数 subject 双峰干净；少数 4 簇；极少数 1D 连续谱——k=2 是主流但 cohort 不同质，先把异质性说出来。

### B.4 k=2 两簇是 anti-correlated

- 主图：`cohort_propagation_summary.png` **panel e**（inter-cluster Spearman r 直方图，左偏 + 显著负簇）
- 一句话：k=2 的两簇在通道 rank 上整体反向；其中一部分 subject 反向程度 ≤ −0.5，是 "forward/reverse" 候选。

### B.5 模板时间稳定性 + 数值小表

- 主图：`cohort_propagation_summary.png` **panel d**（split-half + odd-even match correlation）
- 紧贴一张 4 行小表（不堆 stat）：stable_k 分布 / PR-2.5 grade 分布 / fwd-rev reproduced 比例 / within-cluster τ median vs overall。
- 一句话：模板跨时间切片稳定——这些簇不是采样偶然。

---

## Part C：两种时序由 swap 节点主导（5-6 页）

> 这一段只回答：k=2 两个模板的反向是哪些通道在驱动？答：少数 channel-level swap 节点在两端 source/sink 互换；多数通道是共用 hub。

### C.1 概念过渡（schematic）

- 内容：T_a rank vector vs T_b rank vector；signed Δr = rank_T_b − rank_T_a 的定义示意。
- 一句话：上一段说两簇反向；现在 zoom in 到通道级——是哪几个通道在做这个反转？

### C.2 Cohort displacement heatmap（论文级主图）

- 主图：`results/interictal_propagation/rank_displacement/figures/cohort_displacement_heatmap.png`
- 行按 F_norm 降序、列按 rank_T_a_dense；红 = 在 T_b 中更靠后，蓝 = 更靠前。SOZ 黑框这一页**先不强调**。
- 一句话：对每个 subject，反转主要发生在通道集两端——中间通道相对不动；cohort 呈连续谱，最上几行近乎完全反向，最下几行落在随机 null 区。

### C.3 Variable-k swap classifier 拆 cohort 异质性

- 主图：同上 `cohort_displacement_heatmap.png`，narrative 切到 swap markers（实心黑 = strict / 空心灰 = candidate / 无标记 = none）。
- 一段：约一半 subject 通过 family-wise max-null gate（strict ∪ candidate）；其余是弱反向/弱簇/unimodal——cohort 不同质，**不能写成"所有 subject 都有清晰 swap"**。

### C.4 Forward/reverse 子集上 swap geometry 验证

- 主图：`results/interictal_propagation/template_anchoring/figures/pr6_template_pair_geometry_main.png`
- 一句话：在 PR-2.5 跨时间复现的 forward/reverse 子集上，swap geometry 是 mechanism-sanity tier 通过的（subject-level 一致）；这是 "swap 真实可重复" 的最强子集证据，但**不是 cohort-wide claim**。

### C.5 单 subject swap exemplar

- 主图：`results/interictal_propagation/template_anchoring/figures/pr6_supp_fwdrev_small_multiples.png`（任选一个 subject 抠出）
  - 备选：`results/interictal_propagation/template_pairing/figures/fig4_exemplars.png`
- 一句话：给一个 subject 的解剖直觉——T_a 的 source channel set ≈ T_b 的 sink channel set。

### C.6 Part C 主张总结（纯文本/概念图，1 页）

- 核心 claim（写一行大字）：**"k=2 的两条时序不是独立模板，而是同一网络几何上 swap 节点的两种 source/sink 分配；背景上有共用 hub，前景上有少数节点反转。"**
- 配一张 schematic：两个模板叠在同一通道集上 → 高亮 swap 节点反向 + hub 节点共享。
- 这是整个 PPT 的 climax——把 Part B 的"反向"upgrade 到机制层。

---

## Part D：Swap 节点与临床 SOZ 的关系——诚实 + 互补（4-5 页）

> 这一段只回答：swap 节点是不是临床 SOZ？答：不是简单等同；H1 NULL，但 swap 倾向"覆盖并扩展" SOZ。

### D.1 H1 cohort NULL — 主动摆出来

- 主图：从 `pr6_template_pair_geometry_main.png` 抠出 H1 panel（subject-level delta_subject = mean_k(frac_SOZ_endpoint − frac_SOZ_middle) violin/scatter vs 0）
- 关键文本（不能含糊）：
  - 如果 swap 就是 SOZ，应该看到 endpoint > middle 的 SOZ 富集
  - Cohort Wilcoxon NULL
  - Single-metric H1 不能断言 swap 锚定 SOZ
- 一句话：不藏 NULL；下面我们看 swap 与 SOZ 究竟是什么集合关系。

### D.2 Swap geometry 真实 — 重用 displacement heatmap，开 SOZ 黑框

- 主图：`cohort_displacement_heatmap.png`（与 C.2 同图，但 narrative 切到看 SOZ 黑框位置）
- 一段：SOZ 不是均匀分布在 swap endpoints；也不全部落在 swap endpoints 之内——partial overlap → 那 swap 与 SOZ 是什么集合关系？

### D.3 Set-relationship typology（paper-grade supplementary）

- 主图：`results/interictal_propagation/rank_displacement/figures/swap_clinical_soz_set_relation.png`（3-panel：precision×recall / enrichment×coverage / typology stacked bar）
- 三条 takeaway（不堆 stat）：
  - 0/35 subject 是 E_subset_S（swap 不是 SOZ 的子集）
  - 4/35 subject 是 S_subset_E（swap **覆盖且扩展** SOZ）
  - 多数 partial overlap
- 一句话核心 claim：**"swap 倾向于覆盖 SOZ 并扩展——它给我们的不是 SOZ 验证，而是 SOZ 的 hypothesis-generating extension。"**

### D.4 Channel-selection caveat（必须放）

- 一段（**不能省**，是 Part E.1 PR-T3-1 的桥）：
  - lagPat 通道集已被 high-HI / high-HFO-rate gate 富集 SOZ
  - 本节量化的是 high-HI 区**之内** swap 是否进一步富集 SOZ——不是全脑关系
  - 换通道选择策略后所有数字会变 → 直接接到 Part E.1
- 这一页不放新图，是文字 + 概念图（lagPat universe vs 全脑 universe 的 Venn 示意）。

---

## Part E：未做或漏掉的部分（4 页，无机制建模）

> 删去 Topic 4 BHPN / E.4 mechanism modeling（用户决定先不放）。
> 保留 4 页：PR-T3-1 / Topic 5 z-ER cross-link / Topic 1 Part 2 / Roadmap。

### E.1 PR-T3-1 数据驱动多源 SOZ audit（1 页）

- 问题：临床 SOZ 标注本身有 noise（channel coverage / annotation bias / focus_rel 缺标）；H1 NULL 可能源于此而不是 swap 真不锚定。
- 计划：用 ER-rank / HFO onset rate / multi-source proxy 重建 SOZ candidate set，再回测 PR-6 H1 + swap-vs-SOZ typology。
- 主图：schematic（multi-source SOZ Venn / fusion 示意，新画）。
- 这是 D.4 caveat 的天然下游接口。

### E.2 Topic 5 z-ER × interictal template cross-link（1 页）

- 问题：如果间期 k=2 templates 与 ictal seizure subtype 一一对应，就是 strong 内部交叉验证。
- 现状：z-ER PR-1 已 audit-rerun，cohort 上多数 subject-band 找到 ≥2 ictal subtypes；442/548 sentinel 视觉过关。
- 计划：每 subject 把 ictal subtype label 与 interictal cluster label 对齐，看 ictal pathway diversity 是否 = interictal k；下游必须 per-subtype 不 per-subject。
- 主图：concept 拼图——`cohort_displacement_heatmap.png` 选一行 + 同 subject 的 z-ER ictal subtype heatmap 并排。

### E.3 Topic 1 Part 2 — outcome × partition / pre-ictal rate（1 页）

- 已做：PR-5 dominant template 在 post-ictal 相对 baseline 的绝对招募率系统抬升（Bonferroni-pass），PR-4D 描述层 rate × type 已落盘。
- 计划：(a) pre-ictal 时模板招募率是否 reorganize；(b) surgical outcome × swap_class 的差异；(c) extra-focal phase synchrony pre>post（exploratory）扩 cohort。
- 主图：`results/interictal_propagation/figures/ppt/fig5_pr4d_template_rate.png`（选一个 subject 单图做 motivation）。

### E.4 Roadmap（1 页）

- 时间线（季度粒度）：
  - **Near-term**：PR-T3-1 multi-source SOZ → swap × SOZ revisit
  - **Mid-term**：Topic 5 z-ER × interictal cross-link，interictal × ictal subtype 对齐
  - **Mid-term**：Topic 1 Part 2（pre-ictal rate / outcome × partition）
  - **Long-term**：机制建模层（**本次报告不展开**，作为 placeholder 出现）
- 配一张依赖关系图：哪个 PR 解锁哪个。

---

## Take-home（1-2 页，用户撰写）

> 本文档不预写 take-home；下面是已盖章的 qualitative 结论候选清单，供用户挑选/重组/重写。

### 一定可写（多重 audit 通过）

1. 间期 HFO 群体事件内部存在刻板的通道激活顺序——legacy paper 核心发现在新 pipeline 下整体复现。
2. 这种 stereotypy 的主要构成是结构性的：少数 hub 通道在每个事件里都先激活，是 ordering 的主要来源；事件特异性 dynamics 占比较小。
3. Stereotypy 不是单一模板。多数 subject 能被低维 (k=2) 压缩主导描述；少数需要更高 k 或表现为 1D 连续谱。
4. k=2 的两个模板在通道排序上整体反向，并且在数据切前后两半 / 奇偶 block 上稳定复现——它们不是采样偶然。
5. 这种反向不是均匀全通道反转——由少数 channel-level swap 节点在两端 source/sink 之间互换驱动，多数通道是共用 hub。
6. swap geometry 不等同于临床 SOZ：swap 不构成 SOZ 的子集；更倾向于覆盖 SOZ 并向其周围扩展。

### 可写但需配 caveat（exploratory tier）

7. cohort 内有一个 FW-显著 + 跨时间复现的 forward/reverse 子集——swap geometry 在这个子集上是 mechanism-sanity 通过的，但不是 cohort-wide claim。
8. lagPat 通道集已被 high-HI / high-HFO-rate 选过——目前看到的 swap-vs-SOZ 关系是 high-HI 区**之内**的关系，不是全脑关系。

### 避免在 take-home 出现的措辞（与文档 §7.10 "不能说" 清单一致）

- ❌ "stereotyped HFO timing 锚定 SOZ"
- ❌ "swap geometry 与 SOZ 无关"（只检验了 swap vs template-specific endpoint）
- ❌ "interictal 模板就是 ictal subtype"（cross-link 还没做）
- ❌ "swap 是 SOZ 的精炼候选"（typology 上 0/35 是 E_subset_S）

### 一句话整体框架建议

> "Interictal HFO group events 携带刻板的内部传播模式；这些模式有低维结构（主要是 k=2），且两个模式之间的反向由少数通道级 swap 节点驱动；swap 几何在 cohort 上真实可测，但**不**等同于临床 SOZ——它倾向于覆盖并扩展 SOZ，提示间期刻板时序所刻画的病理网络比临床 onset 标注更广。"

---

## 关键设计原则（贯穿全 PPT）

1. **每页 1 张主图回答 1 个问题**——多 panel 仅在已是论文级 6-panel 主图时使用（cohort_propagation_summary）。
2. **Stat 收 backup**——主页只放 1-2 个最关键的数；完整表搬到 backup slides。
3. **诚实摆 caveat 在自然位置**——A.3 identity-bias / B.3 442 unimodal / C.3 swap classifier 49% none / D.4 channel-selection。caveat 不藏在最后。
4. **不重复用图但允许 narrative 切换**——cohort_propagation_summary 切片 5 次；displacement heatmap 出现 2 次（C.2 不带 SOZ 框 / D.2 带 SOZ 框）。
5. **段连段，无章节封面**——靠每段第 1 页的 schematic + 一句话过渡完成衔接。

---

## 已落盘资产清单（可直接调用，不需要新画）

| 段 | 主图路径 | 该图回答的问题 |
|---|---|---|
| 背景 | `results/refine_soz_validation/figures/yuquan_refine_soz_cohort.png` | high-HI ≈ SOZ |
| A.2 | `results/interictal_propagation/figures/per_subject/<sub>_propagation.png`（推荐 epilepsiae_958） | 单 subject 视觉证据 |
| A.3 | `results/interictal_propagation/figures/cohort_propagation_summary.png` panel a + f | cohort MI + identity bias |
| B.1-B.5 | 同上 panel b / c / e / d + 4 行小表 | bimodality / uplift / anti-corr / 时间稳定 |
| B.3 | `results/interictal_propagation/cluster_geometry/figures/per_subject/{958,818,253,442}*.png` | k=2 视觉本质 + 异质性 |
| C.2 / C.3 | `results/interictal_propagation/rank_displacement/figures/cohort_displacement_heatmap.png` | swap geometry 主图 |
| C.4 / D.1 | `results/interictal_propagation/template_anchoring/figures/pr6_template_pair_geometry_main.png` | fwd/rev subset 验证 / H1 NULL |
| C.5 | `results/interictal_propagation/template_anchoring/figures/pr6_supp_fwdrev_small_multiples.png` | swap exemplar |
| D.3 | `results/interictal_propagation/rank_displacement/figures/swap_clinical_soz_set_relation.png` | typology |
| E.3 | `results/interictal_propagation/figures/ppt/fig5_pr4d_template_rate.png` | rate × type motivation |

需新画的 schematic：A.1（rank 概念）/ C.1（Δr 定义）/ C.6（swap 主张总结）/ D.4（channel-selection Venn）/ E.1（multi-source SOZ Venn）/ E.2（cross-link 拼图）/ E.4（roadmap 依赖图）。共 7 张，单页量级。
