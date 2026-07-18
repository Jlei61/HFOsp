# Cohort contract 与 Supplementary Tables S1–S4

> 状态：working clinical table v0.2（2026-07-12）  
> 作者决定：论文总体 cohort = 40 subjects（Yuquan n=20；Epilepsiae n=20）。  
> 分母纪律：masked temporal analysis 以 n=40 为 primary cohort；空间轴、SOZ、发作和模型分析按各自 eligibility 报告实际 n。  
> 隐私纪律：投稿表仅使用 Y1–Y20 / E1–E20；临床身份与 artifact folder 的 crosswalk 不进入公开补充表。

## 1. Manuscript cohort 口径

**English**

The study included 40 patients who underwent intracranial EEG monitoring, comprising an institutional Yuquan cohort (n = 20) and an independent Epilepsiae cohort (n = 20). All 40 patients were included in the primary masked analysis of interictal HFO temporal organization. Subsequent spatial and clinical analyses were restricted to patients meeting the prespecified data-availability and analysis-specific eligibility criteria, and the corresponding denominators are reported for each analysis.

旧 n=33 作为 same-lineage sensitivity 保留，不再作为正文主分母。7 例 legacy-variant 的 artifact lineage 继续进入 Table S3/方法学审计。

## 2. Table S1 | Yuquan patient characteristics

Y1–Y18 沿用旧临床表的正式顺序；新增两例已在不入库的 private crosswalk 中锁定为 Y19/Y20。公开表不写 artifact folder 或真实身份。当前 40-subject masked primary artifact 恰好包含这 20 例 Yuquan 患者；第 21 个 detector artifact 不在投稿队列。

| Patient | Sex | Age at SEEG (y) | Analyzed duration (h) | Fs (Hz) | Implanted SEEG contacts | Seizures captured | Etiology | Intervention | Outcome / follow-up |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| Y1 | F | 0–5 | 24 | 2000 | 102 | 3 | TSC | Laser ablation | No seizures for 3 years |
| Y2 | M | 0–5 | 24 | 2000 | 148 | 6 | TSC | RF thermocoagulation | No seizures for 1 year |
| Y3 | M | 16–20 | 24 | 2000 | 140 | 3 | FCD | Surgery | No seizures for 4 years |
| Y4 | F | 6–10 | 24 | 2000 | 158 | 2 | TSC | Laser ablation | No seizures for 3 years |
| Y5 | F | 0–5 | 24 | 2000 | 102 | 2 | TSC | Laser ablation | No seizures for 2.5 years |
| Y6 | M | 0–5 | 24 | 2000 | 206 | 17 | TSC | Laser surgery | No seizures for 1 year and 9 months |
| Y7 | F | 0–5 | 22 | 2000 | 172 | 1 | FCD | Surgery | No seizures for 4 years |
| Y8 | F | 6–10 | 26 | 2000 | 142 | 3 | TSC | Surgery | No seizures for 3 years and 5 months |
| Y9 | F | 6–10 | 24 | 2000 | 114 | 3 | FCD | RF thermocoagulation | No seizures for 4 years |
| Y10 | F | 16–20 | 26 | 2000 | 144 | 4 | ganglioglioma | Surgery | No seizures for 3.5 years |
| Y11 | M | 0–5 | 24 | 2000 | 118 | 2 | TSC | Laser ablation | Occasional minor seizures (gelastic) at 3 years |
| Y12 | M | 11–15 | 24 | 2000 | 112 | 1 | TSC | RF thermocoagulation | Occasional minor seizures at 4 years |
| Y13 | M | 21–25 | 26 | 2000 | 148 | 1 | FCD | Laser ablation | Seizures reduced but still present at 3 years |
| Y14 | M | 6–10 | 24 | 2000 | 94 | 1 | FCD | Surgery | No seizures for 4 years and 9 months |
| Y15 | M | 11–15 | **31** | 2000 | 142 | 8 | FCD | Surgery | No seizures for 2 years |
| Y16 | F | 6–10 | 26 | 2000 | 130 | 3 | TSC | Laser ablation | No seizures for 1 year |
| Y17 | F | 0–5 | 26 | 2000 | 104 | 8 | FCD | Laser ablation | No seizures for 1 year and 1 month |
| Y18 | M | 0–5 | 24 | 2000 | 128 | 8 | TSC | Laser ablation | No seizures for 2.5 years |
| Y19 | F | 0–5 | 24 | 2000 | 142§ | 4 | TSC | Not available in mounted source | Not available in mounted source |
| Y20 | M | 11–15 | 24 | 2000 | 134§ | 0‡ | FCD | RF thermocoagulation | Long-term outcome/follow-up not available in case document |

**Table S1 字段合同**：`Implanted SEEG contacts` 指临床 implantation sheet 中的颅内 SEEG 植入触点数，不是 EDF signals、不是双极重参考后通道数，也不包含 ECG/EMG/annotation 等辅助信号。Y1–Y18 的 contact counts 采用作者临床表数值，不用“EDF 减六”反推。病例 doc 支持 Y3 年龄档为 `16–20`、Y15 为 `11–15`。§ Y19/Y20 尚未找到可文本核验的 implantation sheet，当前 `142/134` 是从 EDF 中排除六个明确非触点信号后的 recorded-contact 计数，只能作为临时值。‡ `0` 表示 analyzed recording 中未捕获发作，不表示没有癫痫诊断。

**Y15 duration 合同**：当前 mounted EDF headers 的实际时长求和为 `30.88 h`，表内按整小时报为 `31 h`。

**Outcome 来源边界**：Y1–Y18 的 outcome/follow-up 文本沿用作者临床表，不是从当前病例 doc 重建；当前 doc 多数只到围手术期。原表所写 `data cut: 2021-07-18` 与部分多年随访时长无法同时成立，因此在找到原始随访表前不在列名中声称该 data-cut date。

### Table S1 剩余来源缺口

1. Y19 在当前 `/mnt/yuquan_data/yuquan_24h_bingli` 中没有病例 doc；介入、长期 outcome 和 follow-up 不能从 EDF/MRI/CT 推断。
2. Y20 病例 doc 只到围手术期，不能支持长期 outcome/follow-up。
3. Y19/Y20 尚缺可核验的 implantation sheet；`142/134` 暂时保留为带脚注的 recorded-contact 计数，不声称为正式 implanted-contact 数。

## 3. Table S2 | Epilepsiae patient characteristics

| Patient | Sex | Age at SEEG (y) | Analyzed duration (h) | Fs (Hz) | Contacts | Seizures captured | Etiology | Engel outcome (follow-up) |
|---|---|---:|---:|---:|---:|---:|---|---|
| E1 | F | 31–35 | 162.6 | 1024 | 100 | 9 | malformation; hippocampal sclerosis | Ia (3 m) |
| E2 | F | 46–50 | 244.7 | 1024 | 117 | 94 | malformation | Ia (3 m); IIb (11 m) |
| E3 | F | 11–15 | 219.1 | 1024 | 121 | 16 | cryptogenic | Ia (3 m) |
| E4 | M | 36–40 | 110.6 | 1024 | 107 | 30 | cryptogenic | Ia (3–24 m) |
| E5 | M | 16–20 | 245.2 | 1024 | 122 | 13 | malformation | Ia (3–24 m) |
| E6 | F | 31–35 | 151.6 | 1024 | 125 | 9 | cryptogenic | IVb (3–6 m) |
| E7 | M | 21–25 | 170.6 | 1024 | 95 | 22 | genetic risk | Ia (3–12 m) |
| E8 | F | 46–50 | 224.0 | 1024 | 96 | 20 | hippocampal sclerosis | IIIa (3–12 m) |
| E9 | F | 36–40 | 260.1 | 512 | 58 | 7 | hippocampal sclerosis | IIb (3 m); Ia (48 m) |
| E10 | M | 21–25 | 113.2 | 1024 | 116 | 26 | vascular hypoxia | Not offered |
| E11 | M | 21–25 | 424.1 | 1024 | 109 | 52 | malformation | Ib (3–24 m) |
| E12 | M | 41–45 | 252.4 | 1024 | 63 | 7 | malformation | Ia (3 m) |
| E13 | F | 21–25 | 63.0 | 1024 | 65 | 23 | malformation; tumor | Ia (3 m); IIa (36 m) |
| E14 | M | 16–20 | 142.0 | 1024 | 109 | 31 | malformation; hippocampal sclerosis | Ia (3–24 m) |
| E15 | F | 46–50 | 65.0 | 1024 | 119 | 15 | hippocampal sclerosis | IIb (3–6 m) |
| E16 | F | 51–55 | 130.0 | 512 | 63 | 6 | hippocampal sclerosis | IVb (3 m); Ia (60 m) |
| E17 | F | 11–15 | 155.0 | 1024 | 87 | 14 | malformation; hippocampal sclerosis | Ia (3 m) |
| E18 | F | 26–30 | 183.1 | 1024 | 122 | 9 | malformation; hippocampal sclerosis | Ia (3 m) |
| E19 | F | 26–30 | 248.7 | 1024 | 71 | 9 | hippocampal sclerosis | Ib (3 m); IIIa (24 m) |
| E20 | F | 61–65 | 118.9 | 1024 | 82 | 21 | tumor | Not offered |

### Table S2 口径风险

- `Duration` 应明确为 analyzed/selected recording duration。E13、E15、E16 的旧表值小于当前 SQL 中全部 recording 合计时长，不能称 total available duration；
- `Contacts` 在同一患者不同 recording 间可变化，旧表值不能与 median intracranial channels 或预处理后 bipolar channels 混用；
- `Not offered` 表示未提供/未接受手术，因此 Engel outcome 不适用，不是普通 follow-up missing。

## 4. 我建议表内补充什么

### S1/S2 直接增加的临床列

1. **Clinical SOZ localization and laterality**：本论文以 clinical SOZ 为锚，患者表必须交代其解剖位置；
2. **Intervention**：resection、laser ablation、RF thermocoagulation 或 not offered；
3. **Standardized outcome and follow-up duration**：优先拆成 outcome class 与 months；Yuquan 的自由文本不能由分析人员自行映射 Engel/ILAE；
4. 明确列名为 **Analyzed recording duration**、**Implanted/recorded contacts**、**Seizures captured during analyzed recording**；
5. 若 outcome 有多个时间点，建议拆成 first available 与 last available outcome，或明确只报 last available。

### 放入 S3/S4，而不是继续加宽人口学表

- **Table S3 temporal evidence**：`patient, dataset, artifact_lineage, n_events, n_core_channels, masked_features, stable_k, overall_tau, within_tau, uplift, MI_p, reproducibility_grade, opposing_pair_candidate, opposing_pair_reproduced`；
- **Table S4 spatial evidence**：`patient, n_implanted_shafts, coordinate_available, coordinate_space, geometry_tier, n_mapped_contacts, heldout_axis_rho, paired_axis_cosine, endpoint_compactness, spatial_eligible, exclusion_reason`。

这样 S1/S2 回答“患者是谁、记录了什么”，S3/S4 回答“每位患者支持了哪一层论文结论”。
