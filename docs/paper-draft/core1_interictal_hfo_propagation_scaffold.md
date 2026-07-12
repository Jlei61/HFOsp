# Core 1：间期 HFO 群体事件反复采样患者特异性传播骨架

> 状态：paper claim package v0.2（2026-07-12）  
> 投稿 cohort 已锁：40 subjects（Yuquan n=20；Epilepsiae n=20）。masked 时序分析以这 40 例为总体 cohort；三维传播几何及其他下游检验按各自 eligibility 报告实际分母。  
> 临床队列及补充表：[`cohort_contract_and_supplementary_tables.md`](cohort_contract_and_supplementary_tables.md)。

## 1. 一句话论点

在两个独立 SEEG 队列中，间期 HFO 群体事件在患者内反复呈现稳定而多模态的通道激活顺序；在具有可复现相反排序模板的患者子集中，两类模板可投影为同一三维 SEEG 接触点传播轴的相反读取，支持这些事件反复采样患者特异性病理传播骨架。

边界：分析发生在 SOZ-enriched HFO network 中，但不能写成“证明了局限于 clinical SOZ 内部的传播轴”。

## 2. 当前证据快照

### 时间组织，primary cohort n=40

| 指标 | 当前结果 | 解释 |
|---|---:|---|
| MI > permutation null | 40/40 | 共同参与触点上患者内固定 rank structure 高于随机置换（masked shared-participant 主度量，cohort median 0.228；纳入 phantom 的全通道 legacy 版同为 40/40、median 0.188，作 unmasked sensitivity）|
| Overall Kendall tau | median 0.084 | 多模板混合后的低总体一致性 |
| Within-template Kendall tau | median 0.289 | 分模板后时序一致性提高 |
| Within-template uplift | median +0.184 | 支持少数 recurrent temporal templates |
| Stable k | k=2: 34/40；其余 k=3–6 | 双模板主导但不是普遍真相 |
| Temporal reproducibility | strong 26；moderate 12；weak 2 | 模板总体可跨时间复现 |
| Opposing-template funnel | candidate 16；reproduced 15 | 正反关系属于定义明确的患者子集 |

Artifact：`results/interictal_propagation_masked/pr1_cohort_summary.json`。

### 空间组织，analysis-specific subset

| 指标 | 当前结果 | 解释 |
|---|---:|---|
| Geometry records | 30 ok；primary 23；fallback 3；descriptive 4 | 不与总体 n=40 混写 |
| Held-out axis validation | n=26；median Spearman rho=0.752 | 接触点轴可预测 held-out rank |
| Paired axes | n=10 | 可严格比较两模板方向的 subset |
| Strong reversed shared axis | 7/10；median cosine=-0.977 | 支持同轴相反读取 |

Artifact：`results/spatial_modulation/propagation_geometry/cohort_summary.json`。

### SOZ 边界

- Refined HFO rate 可区分 clinical SOZ contacts；
- endpoint 相对 middle channels 未显示额外 cohort-level SOZ enrichment；
- 因此安全主张是 `patient-specific propagation scaffold within an SOZ-enriched HFO network`，不是 `axis confined to the clinical SOZ`。

## 3. Figure 1

Figure 1 已指定为本核心主图。逐 panel 合同见 [`figure1_interictal_hfo_temporal_scaffold.md`](figure1_interictal_hfo_temporal_scaffold.md)。

当前定稿候选可承担：群体事件现象、SOZ clinical anchor、患者内多模板结构、MI/null 与 within-template uplift。（跨时间复现是 cohort 结果，写入正文 / Table S3，不再作为 Fig 1 面板。）

当前 Figure 1 明确不承担三维同轴正反传播；contact-space map、held-out axis 和 paired-axis cosine 进入下一张 spatial 主图。因而 Figure 1 的图题和图注只写 temporal organization，不单独宣称 shared 3D axis。

## 4. Supplementary tables

### Table S1：Yuquan patient characteristics

20 例临床与记录信息。当前缺口是 Y19 介入/结局/随访、Y20 长期结局/随访，以及两例 implantation sheet。

### Table S2：Epilepsiae patient characteristics

20 例临床与记录信息。`Duration` 和 `Contacts` 必须给出明确表注定义。

### Table S3：Per-patient temporal evidence

`patient, dataset, artifact_lineage, n_events, n_core_channels, masked_features, stable_k, overall_tau, within_tau, uplift, MI_p, reproducibility_grade, opposing_pair_candidate, opposing_pair_reproduced`

### Table S4：Per-patient spatial evidence

`patient, coordinate_available, coordinate_space, geometry_tier, n_mapped_contacts, heldout_axis_rho, paired_axis_cosine, endpoint_compactness, spatial_eligible, exclusion_reason`

## 5. Cohort denominator contract

投稿总体 cohort 已锁定为 n=40（Yuquan n=20；Epilepsiae n=20），40 例均进入当前 masked 时序 primary analysis。旧 n=33 不再作为正文主分母，但保留为 same-lineage sensitivity；7 例 legacy-variant 的来源差异继续在 Table S3/方法学审计中记录。

总体 cohort n=40 不自动传递给空间、SOZ、发作或模型分析。每个下游检验继续报告自己的 eligible n 和 exclusion reason。

## 6. 尚未闭合的 paper gates

1. Y19 介入/长期 outcome/follow-up、Y20 长期 outcome/follow-up 及两例 implantation sheet 尚待医院来源；
2. TA/TB 四端点 compactness 缺 dedicated export；
3. `皮层源空间` 必须统一改为三维 SEEG 接触点空间；
4. 离散双模态措辞限定为 subset/多模板组织，不能用 40/40 dip test 宣称所有患者均有两个离散状态。
