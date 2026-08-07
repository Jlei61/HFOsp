# Handoff — Topic 5 cohort-parent-matched 多频带 field concordance

date: 2026-07-18  
status: **SUPERSEDED — DO NOT RUN**  
analysis role: Fig3 cohort field-concordance 的频带条件分解；不是旧 F2 的继续筛选  
run form: `docs/archive/topic5/field_concordance_multiband_erin_run_form_2026-07-18.md`

> **作废说明（2026-07-18）**：本合同锁定的是 R2 contact-evaluated smoothed similarity，已不能满足 Figure 3 统一使用 R3 dense-grid field similarity 的要求。请改执行 `docs/archive/topic5/fig3_ictal_gradient_r3_full_recompute_handoff_2026-07-18.md`；本文件仅保留方法演变 provenance。

> 给 Erin 的一句话任务：在当前 paper-facing n=17 / 167 seizures 的母分析上，只把 ictal activation 展开成七个固定频带；保持同一 gradient field、同一 `[0,10] s`、同一 subject-first fold，并以跨频带共用的 all-contact permutation 作为 primary null。纯 within-shaft 只作第二层敏感性，不做 fallback 混合。

---

## 0. 先锁定科学问题

### 0.1 母分析已经回答什么

母分析比较的是：

- 固定的 **interictal HFO propagation timing field**（TA/TB earliness field）；
- clinical/EEG onset 后 `[0,10] s` 的 early-ictal energy field；
- sign-free、mirror-invariant、A/B `maxAB` spatial concordance；
- 与 patient-specific all-contact channel-label shuffle 比较。

当前 paper-facing 结果来自：

`results/topic5_ictal_recruitment/tspectral_field_concordance/clinical_onset_gradient_field_cohort_stat_summary.json`

当前数值仅作 provenance，不是新运行必须命中的结果门：

| group | n subject / seizure | data median | null median | margin median | data>null | one-sided p |
|---|---:|---:|---:|---:|---:|---:|
| Pooled phenotype-matched | 17 / 167 | 0.842 | 0.797 | 0.052 | 12/17 | 0.0153 |
| Strict broadband | 16 / 106 | 0.784 | 0.780 | 0.048 | 12/16 | 0.0193 |
| Gamma non-BB | 11 / 61 | 0.842 | 0.824 | 0.043 | 6/11 | 0.350 |

安全表述是 **cohort-level shift with subject heterogeneity**，不是“17/17 普遍阳性”，也不是逐点 replay、传播方向或因果机制。

### 0.2 新分析必须回答两个不同问题

1. **Band inheritance**：哪些频带的 field concordance 高于该频带自己的 all-contact null？
2. **Band specificity**：不同频带的 excess concordance 是否真的彼此不同？

“一个频带显著、另一个不显著”不能证明两个频带之间不同。第二问必须直接检验 band 间的 paired margin。

### 0.3 Null 的角色

| null | role | 可以支持什么 | 不能支持什么 |
|---|---|---|---|
| all-contact channel shuffle | **primary；继承母分析 estimand** | patient-specific coarse scaffold concordance | 不能声称已控制 shaft identity / local smoothness |
| pure within-shaft shuffle | **secondary anatomical sensitivity** | concordance 是否细于 shaft-level topology | 不足时不得退化成 distance/subject-wide 后仍叫 within-shaft |

选择 all-contact 是因为它匹配母分析问题，不是因为它更容易显著。

---

## 1. 为什么现有 n=17 F2 不能直接交稿

当前 n=17 F2 只是把旧 n=19 fixed-sigma F2 按 subject 名单截成 17 人，没有按母分析事件重算。

| contract | paper-facing parent | current n=17 F2 |
|---|---|---|
| subjects | 17 | 17 |
| seizures | 167 phenotype-matched | 184 old-F2 seizures |
| common seizures | — | 108 |
| F2-only / parent-only | — | 76 / 59 |
| time | exact onset `[0,10] s` | five overlapping windows, `[-5,5]` to `[15,25] s` |
| null | all-contact | within-shaft → distance → subject-wide mixed fallback |
| smoothing | frozen per-model in current parent | subject-fixed A sigma |
| inference | subject-paired cohort statistic | cohort-median permutation + seven-band maxT |

因此旧图保留为 sensitivity provenance；**不得把它重命名成 parent-matched frequency decomposition，也不得覆盖旧 artifact。**

---

## 2. Locked cohort and event contract

### 2.1 唯一事件清单

从下列文件读取，不重新按当前结果筛选：

`results/topic5_ictal_recruitment/tspectral_field_concordance/clinical_onset_gradient_field_cohort_stat_event.csv`

固定过滤：

```text
group_id == "all_phenotype_matched"
unique key = (subject, seizure_idx)
expected = 17 subjects / 167 unique seizures
```

其中：

- `strict_broadband`: 106 seizures；
- `gamma_nonbroadband`: 61 seizures；
- 16 名 Epilepsiae 使用 clinical onset；
- `yuquan_xuxinyi` 的 1 次发作只有 EEG onset，必须保留 `time_reference=eeg_onset_only`，不得写成 clinical onset。

### 2.2 固定 subject 列表

```text
epilepsiae_1077
epilepsiae_1084
epilepsiae_1096
epilepsiae_1125
epilepsiae_1146
epilepsiae_1150
epilepsiae_139
epilepsiae_253
epilepsiae_384
epilepsiae_442
epilepsiae_548
epilepsiae_590
epilepsiae_620
epilepsiae_635
epilepsiae_922
epilepsiae_958
yuquan_xuxinyi
```

这份 n=17 是 phenotype-matched cohort，不是“删除两名单杆患者”：E139 与 E1146 明确保留。E583 和 `yuquan_zhangkexuan` 因不在 parent `all_phenotype_matched` group 而不进入本分析。

### 2.3 当前已完成的只读 coverage audit

- 七个 primary bands 的 167 个 event cache key 全部存在；
- 每个 event 各带对 frozen field contact order 均有 `n_finite_contacts >= 6`；
- 七带共同 finite-contact intersection 在全部 167 events 上均 `>=6`；
- common-seven-band contact count：min 6，median 11，max 16；
- 与 historical `bb150_auc` 再取交集仍是全部 167 events、min 6 contacts。

Erin 必须在 runner 中重新 fail-closed 验证这些数字；不能只相信 handoff。

---

## 3. Locked inputs

| role | canonical input |
|---|---|
| parent event/subject truth | `results/topic5_ictal_recruitment/tspectral_field_concordance/clinical_onset_gradient_field_cohort_stat_{event,subject,summary}.{csv,json}` |
| frozen gradient fields | `results/interictal_propagation_masked/template_gradient_fields/per_subject/<subject>.json` |
| seven-band activation cache | `results/topic5_ictal_recruitment/v2_band_scan/cache/<subject>.{json,npz}` |
| historical parent BB 1–150 anchor | `results/topic5_ictal_recruitment/t0_feature_cache_bb150_1_150/<subject>.{json,npz}` |
| band definitions | `config/topic5_v2_phase1.yaml::bands.primary` |
| parent implementation reference | `scripts/run_topic5_clinical_onset_gradient_field_cohort_stat.py` |
| fixed-sigma helper reference | `scripts/run_topic5_gradient_multiband_original_f2.py::rebuild_subject_fixed_scorers` |
| scorer/null primitives | `src/topic5_template_axis_field.py`, `src/topic5_tspectral_field_concordance.py` |

Do not mutate any input NPZ/JSON. Hash inputs before and after the run.

---

## 4. Locked field/scorer contract

### 4.1 Field routing

Per subject, outcome-independent routing only:

```text
complete shared_a + shared_b available -> shared maxAB
otherwise                            -> own maxAB
```

Expected routing on n=17：7 shared / 10 own fallback。不得根据 ictal correlation、p 值或频带重新选择 shared/own。

### 4.2 Subject-fixed smoothing

本次统一采用此前已决定的旧 endpoint smoothing 口径：

1. 从 selected A plane 的 median nearest-neighbour spacing 估计一次 subject sigma；
2. TA、TB、七个频带、全部 seizures、observed 与所有 null draws 共用该 sigma；
3. A/B 各自保留冻结 field 中的 earliness 与 support；只共享 sigma，不把 B support 偷换成 A support；
4. parent anchor 也必须用同一 subject-fixed scorer 重跑。当前 frozen-sigma parent 数值只能作 provenance。

### 4.3 Field statistic

For each event and band:

```text
r_A = max(abs(corr_identity_A), abs(corr_mirror_A))
r_B = max(abs(corr_identity_B), abs(corr_mirror_B))
r_maxAB = max(r_A, r_B)
```

必须使用 corrected abs-max mirror rule，不得回到 legacy `abs(max(signed correlations))`。

每个 null draw 从 contact activation 开始，重新：

```text
permute activation -> smooth -> choose mirror -> score A/B -> maxAB
```

### 4.4 Common contact support

对每个 `(subject, seizure_idx)`，建立一次七带共同 finite-contact mask：

```text
frozen field contact_order
∩ cache channel names
∩ analysis_channels
∩ finite in all seven bands over [0,10] s
```

同一个 event 的七个频带、parent anchor 和所有 null draws 必须使用同一个 mask；`n_common_contacts < 6` 时停止并报告，不得按频带静默改变 denominator。

---

## 5. Locked frequency contract

七带顺序、边界和标签必须原样来自 `config/topic5_v2_phase1.yaml`：

| key | interval, half-open |
|---|---:|
| `delta_HYP_slow` | `[1,4)` Hz |
| `theta_preictal_PAC` | `[4,8)` Hz |
| `alpha_sharp_leq13` | `[8,13)` Hz |
| `beta_LVFA_low` | `[13,30)` Hz |
| `gamma_LVFA` | `[30,80)` Hz |
| `hg_low_ripple` | `[80,150)` Hz |
| `ripple_high` | `[150,250)` Hz |

Primary family 仍为这七带，不增加 composite、不删除“结果不好”的 band。

`epilepsiae_139` 与 `epilepsiae_253` 的 `ripple_high` 有 `fs_edge_flag=True`。主表如实保留，另做预设 sensitivity：

- 排除 `fs_edge_flag=True` subject；或
- 使用 `ripple_safe_80_220` 作为 sidecar。

不得用该 sensitivity 取代七带主 family。

---

## 6. Locked null and permutation contract

### 6.1 Primary all-contact null

- `n_perm = 1000`；
- `seed = 20260718`；
- 每个 `(subject, seizure_idx, draw)` 生成一套 physical-contact permutation；
- **同一 permutation 必须跨七频带复用**；
- 如同时计算 historical parent anchor，同一 mapping 也应用到 anchor；
- unmatched contacts 保持 missing，不移动进分析分母；
- 输出 permutation hash audit，证明同一 draw 在七带使用相同 mapping。

优先复用：

- `src.topic5_tspectral_field_concordance.make_contact_permutations(..., mode="all_contact")`
- `score_permutation_matrix(...)`
- `fold_seizure_null_draws(...)`

不得复用旧 F2 “相同 perm_id、但每 band/window 重新抽 permutation”的行为。

### 6.2 Secondary pure within-shaft null

- 不使用 distance-bin 或 subject-wide fallback；
- 保留 `min_group_for_shaft=4` 的 strong 口径；
- 只有所有进入分析的 finite contacts 都能被合法 within-shaft groups 覆盖时，才标为 `within_shaft_strong`；
- 其余 event/subject 标为 unavailable，并输出 shaft-size inventory；
- within-shaft cohort 的 n 必须单独报告，不能沿用 n=17 标签。

若 all-contact 有 evidence、within-shaft 没有，结论仍可到 coarse scaffold，但不能写 fine within-shaft specificity。

---

## 7. Folding and inference

### 7.1 Statistical unit

Subject 是唯一 cohort statistical unit：

```text
event score -> median within subject -> cohort over subjects
```

E1084 的 56 次发作不得比单发作 subject 获得更大 cohort 权重。

For each subject `s` and band `b`:

```text
data[s,b] = median_event(r_obs)
null[s,b,k] = median_event(r_perm[k])
delta[s,b] = data[s,b] - median_k(null[s,b,k])
```

### 7.2 Band inheritance inference

必须同时输出：

- per-band data/null/margin distribution；
- one-sided paired Wilcoxon `data > subject_null_median`；
- seven-band family-wise p，使用保留跨带依赖的 subject-level sign-flip maxT 或等价的 coherent maxT；
- median margin、IQR、`n_delta_positive/n_subjects`；
- per-subject empirical p 只作 sidecar，不当 cohort unit。

### 7.3 Band specificity inference

Mandatory omnibus：直接检验七带 `delta[s,b]` 是否存在 band effect。

若 omnibus 没有 evidence，结论为 band-generic；不得根据单带星号挑 winner。

若报告 pairwise contrasts：

- 必须预先列出或完整报告全部 pairwise；
- subject-paired；
- 多重比较校正；
- 主比较对象是 `delta`，不是 raw observed correlation，也不是不同 band 各自 p 值。

### 7.4 Phenotype decomposition

Primary frequency board 使用同一 167 events。另输出：

- strict-broadband 106-event stratum；
- gamma-nonbroadband 61-event stratum。

这两层是条件分解，不把 seizure 当独立 cohort 样本，也不把 phenotype difference 写成机制。

---

## 8. Required implementation and outputs

### 8.1 Suggested new runner

```text
scripts/run_topic5_parent_matched_multiband_field_concordance.py
```

Suggested CLI:

```bash
python scripts/run_topic5_parent_matched_multiband_field_concordance.py \
  --n-perm 1000 \
  --seed 20260718 \
  --outdir results/topic5_ictal_recruitment/field_concordance_multiband_parent_matched
```

这是待 Erin 实现的目标接口，不是当前已存在命令。

### 8.2 Output root

```text
results/topic5_ictal_recruitment/field_concordance_multiband_parent_matched/
```

禁止覆盖：

- `results/topic5_ictal_recruitment/gradient_multiband_original_f2_fixed_sigma*`
- `results/topic5_ictal_recruitment/tspectral_field_concordance/clinical_onset_gradient_field_cohort_stat_*`
- 任何现有 paper-ready figure。

### 8.3 Required artifacts

```text
contract_manifest.json
cohort_event_inventory.csv
field_routing_and_sigma.csv
common_contact_inventory.csv
permutation_mapping_audit.csv
parent_anchor_event.csv
parent_anchor_subject.csv
parent_anchor_cohort.csv
multiband_event.csv
multiband_subject.csv
multiband_cohort.csv
multiband_subject_null_draws.parquet
multiband_band_omnibus.json
multiband_band_contrasts.csv
within_shaft_event_inventory.csv
within_shaft_subject.csv
within_shaft_cohort.csv
drop_inventory.csv
summary.json
figures/README.md
figures/parent_anchor_data_vs_null.{png,pdf}
figures/multiband_all_contact_margin.{png,pdf}
figures/multiband_within_shaft_sensitivity.{png,pdf}
```

`figures/README.md` 按 repo 规范逐图写 2–4 句中文说明并以 `**关注点**：` 收尾。

### 8.4 Required tests

至少覆盖：

1. 17 subjects / 167 unique event keys 精确锁定；
2. 七带 common-contact mask 完全相同且 `>=6`；
3. shared/own routing 不随 band/outcome 改变；
4. one subject-fixed sigma 贯穿 A/B/bands/null；
5. corrected abs-max mirror；
6. 同一 event/draw 的 permutation hash 在七带相同；
7. null 每 draw 重做 smoothing/mirror/maxAB；
8. seizure→subject fold，不 pool seizures；
9. seven-band family correction 使用 coherent cross-band dependence；
10. pure within-shaft 无 fallback；
11. input hashes before/after 一致；
12. deterministic rerun byte/numeric parity。

Suggested test file:

```text
tests/test_topic5_parent_matched_multiband_field_concordance.py
```

---

## 9. Stop conditions

以下是 contract blockers；遇到即停止并回报，不自行修分母：

- parent event list 不是 17 subjects / 167 unique seizures；
- frozen field fingerprint mismatch；
- 任一 event 的 seven-band common mask `<6`；
- 任一 band 静默丢失 event/contact，导致 band denominators 不同；
- 无法保证同一 permutation mapping 跨 band 复用；
- input artifact 被改写；
- within-shaft 需要 fallback 才能运行；
- cohort inference 把 seizures 当独立样本。

以下不是 stop condition：

- parent anchor 数值因 fixed sigma 有变化；
- 某个或全部频带不显著；
- band omnibus 为 null；
- within-shaft sensitivity 不支持；
- 没有“winner band”。

所有结果都必须如实输出，不设 outcome-based PASS/FAIL 或 winner selection。

---

## 10. Allowed and forbidden claims

### Allowed, depending on results

- “Early-ictal energy in band X shows cohort-level concordance with the frozen interictal HFO timing field above an all-contact channel-label null.”
- “The effect is distributed across bands / shows a direct band contrast.”
- “The association is supported at a coarse patient-specific topology level.”
- 如果 pure within-shaft 也有 evidence，才可补充“survives shaft-preserving relabeling”。

### Forbidden

- “所有患者普遍显著”；
- “某带显著、另一带不显著，所以前者更强”；
- “all-contact null 已控制空间平滑”；
- “endpoint 的旧 null 可以给 gradient 复用”；
- “发作沿间期 HFO 顺序逐点 replay”；
- “方向、极性或因果传播已证明”；
- “HFO/ripple-specific mechanism” without direct band contrast and required controls；
- 按结果排 subject、event、band 或 shared/own route。

---

## 11. Deprecated copy-ready message

```text
DO NOT SEND OR RUN THIS OLD CONTRACT.
Use instead:
docs/archive/topic5/fig3_ictal_gradient_r3_full_recompute_handoff_2026-07-18.md
```
