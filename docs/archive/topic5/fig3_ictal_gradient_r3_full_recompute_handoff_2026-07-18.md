# Handoff v2 — Figure 3 发作相关 gradient R3 field concordance 全量重算

date: 2026-07-18  
status: **READY FOR AGENT IMPLEMENTATION / RUN**  
primary estimand: **R3 dense-grid field similarity**  
paired sensitivity: **R2 contact-evaluated smoothed similarity**  
run form: `docs/archive/topic5/fig3_ictal_gradient_r3_full_recompute_run_form_2026-07-18.md`  
supersedes: `field_concordance_multiband_unified_handoff_2026-07-18.md` and its Erin form

> 给执行 agent 的一句话任务：以正式的 n=17 / 167 seizures 为唯一母清单，把 Figure 3 中所有仍承载定量 field-similarity 的发作相关结果统一迁移到 gradient-axis R3 dense-grid scorer；先完整重算 cohort Data-vs-Null 与七频带统计，再把 Fig3-B 的 score provenance 和仍保留在 Figure 3 的 Fig3-C 轨迹/null 同步迁移。R2 必须用完全相同的输入重算，只作配对敏感性，不得与旧文件直接比较。

---

## 0. 这次重算的拍板

### 0.1 为什么旧结果不能直接继续用

目前 Figure 3 混有三套不完全相同的统计口径：

| analysis | geometry | similarity | cohort/window | 主要问题 |
|---|---|---|---|---|
| 旧 paper-ready endpoint cohort | endpoint/source–sink | R3，81×81 grid | 旧 n=20、旧 seizure set | 不是 gradient；legacy mirror 为 signed-max 后取 abs；TB 曾借用 TA plane/support |
| 当前 gradient parent | gradient shared-else-own | R2，触点位置核平滑后相关 | n=17 / 167，onset `[0,10] s` | 不是 dense-grid field similarity |
| 当前 n=17 multiband 图 | gradient shared-else-own | R2 | 从旧 n=19 F2 截 subject，实际 184 seizures/重叠窗/mixed null | 不是 parent-matched 重算 |

目标不是让 gradient 复制旧 endpoint 的低 null，也不是选择更容易显著的方法。目标是：

```text
同一 frozen gradient geometry
+ 同一 cohort/event/contact mask
+ 同一 sigma
+ 同一 permutation
+ 同一 seizure→subject fold
```

只把最终 evaluation layer 成对比较为：

```text
R3 primary = dense-grid support-gated field correlation
R2 sensitivity = contact-evaluated smoothed-field correlation
```

### 0.2 科学问题

母问题：患者特异的间期 HFO propagation timing field，是否与发作早期 `[0,10] s` energy field 的空间分布，比患者内 contact-label shuffle 更一致？

频带问题必须拆成两问：

1. **Band inheritance**：每个频带的 R3 concordance 是否高于自己的 coherent all-contact null？
2. **Band specificity**：七个频带的 null-relative concordance 是否存在直接的 paired band effect？

“一个频带显著、另一个不显著”不等于两个频带彼此不同。

### 0.3 统计量的含义边界

Primary statistic 是 sign-free、transverse-mirror-invariant、A/B `maxAB` 的空间场一致性。它可以支持 patient-specific coarse field/scaffold concordance；不能单独支持：

- 发作按间期 timing order 逐触点 replay；
- 传播方向或极性一致；
- 因果传播机制；
- HFO/ripple-specific mechanism；
- 所有 subject 都有个体显著效应。

---

## 1. “整个 Figure 3”在本 handoff 中的范围

| Figure 3 component | 当前性质 | 本次动作 | 完成层级 |
|---|---|---|---|
| Fig3-A raw SEEG/TFR/band context | signal-context display，无 field statistic | 保留；只核 input checksum 与链接，不因 R3 重算 | 不阻塞 |
| Fig3-B E1146 seizure 2 field exemplar | 图面是连续 field；metadata/checkpoint 仍引用 R2 score | 保留已锁定案例与视觉构图；重算 R3 score provenance，R2 作 sidecar；禁止重新选 seizure | Stage 2 mandatory |
| cohort `field_concordance_cohort_stat` | 旧正式图是 endpoint R3；当前 gradient parent 是 R2 | 以 n=17 / 167 重新生成 gradient R3 Data-vs-Null 图 | Stage 1 mandatory |
| 七频带统计图 | 当前 n=17 图并非 167-event parent-matched 重算 | 以同一 167 events、共同 mask/permutation 重新生成 | Stage 1 mandatory |
| Fig3-C peri-onset similarity | 纵轴和 spatial null 都是 R2，不是普通 illustration | 若继续留在 Figure 3，7 名 shared-only 轨迹和全部空间 null 必须迁移到 R3 | Stage 2 mandatory |
| 旧 Fig3-Sup1 / old F2 | exploratory historical contracts | 保留 provenance；不得重命名为新结果 | archive only |

交付措辞必须按阶段：

- 只完成 Stage 1：只能写“cohort 与 frequency statistics 已完成 R3 重算”；
- Stage 1 + Stage 2 都完成：才可写“Figure 3 发作相关 field-similarity package 已统一为 R3”。

---

## 2. Locked cohort、事件与输入

### 2.1 唯一母事件表

```text
results/topic5_ictal_recruitment/tspectral_field_concordance/
clinical_onset_gradient_field_cohort_stat_event.csv
```

固定过滤：

```text
group_id == "all_phenotype_matched"
unique key = (subject, seizure_idx)
expected = 17 subjects / 167 seizures
```

固定分层：

| group | subjects | seizures | role |
|---|---:|---:|---|
| Pooled phenotype-matched | 17 | 167 | primary parent cohort |
| Strict broadband | 16 | 106 | phenotype decomposition |
| Gamma non-BB | 11 | 61 | phenotype decomposition |

Subject 是唯一 cohort unit。E1084 的 56 次发作不能比单次发作 subject 获得更高 cohort 权重。

### 2.2 固定 subject 名单

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

这不是“删掉两名单杆患者”得到的名单。E139 与 E1146 明确保留；E583 与 `yuquan_zhangkexuan` 是因为不在 parent `all_phenotype_matched` event contract 中而不进入。

### 2.3 时间与 activation

- Epilepsiae：clinical onset `[0,10] s`；
- `yuquan_xuxinyi`：没有 clinical onset，保留真实 EEG onset `[0,10] s`，metadata 必须写 `eeg_onset_only`；
- activation 为每 contact、每 seizure 的 baseline-normalized log band power robust-z 后，在 `[0,10] s` 取均值；
- 统计前不得对 contacts 做 min–max、rank、spatial z-score、sign flip 或 clipping；
- display-only min–max 必须与 statistical values 分开保存。

### 2.4 Canonical inputs

| role | path |
|---|---|
| parent event/subject truth | `results/topic5_ictal_recruitment/tspectral_field_concordance/clinical_onset_gradient_field_cohort_stat_{event,subject,summary}.{csv,json}` |
| frozen gradient fields | `results/interictal_propagation_masked/template_gradient_fields/per_subject/<subject>.json` |
| seven-band activation | `results/topic5_ictal_recruitment/v2_band_scan/cache/<subject>.{json,npz}` |
| parent BB 1–150 anchor | `results/topic5_ictal_recruitment/t0_feature_cache_bb150_1_150/<subject>.{json,npz}` |
| fixed bands | `config/topic5_v2_phase1.yaml::bands.primary` |
| current R2 parent reference | `scripts/run_topic5_clinical_onset_gradient_field_cohort_stat.py` |
| current frozen scorer | `src/topic5_template_axis_field.py` |
| historical R3 primitives | `src/propagation_contact_plane_readout.py` |
| coherent null/fold helpers | `src/topic5_tspectral_field_concordance.py` |

所有输入在运行前后做 SHA-256。不得修改或重写 frozen JSON/NPZ。

### 2.5 七个 primary bands

频带顺序与 half-open 边界必须原样读取 config：

| key | interval |
|---|---:|
| `delta_HYP_slow` | `[1,4)` Hz |
| `theta_preictal_PAC` | `[4,8)` Hz |
| `alpha_sharp_leq13` | `[8,13)` Hz |
| `beta_LVFA_low` | `[13,30)` Hz |
| `gamma_LVFA` | `[30,80)` Hz |
| `hg_low_ripple` | `[80,150)` Hz |
| `ripple_high` | `[150,250)` Hz |

E139/E253 的 `ripple_high` 有 `fs_edge_flag`。七带主 family 仍保留它们；另做预设 sidecar：排除这两名 subject，或计算 `ripple_safe_80_220`。不得用 sidecar 替换主 family。

### 2.6 Common contact mask

每个 `(subject, seizure_idx)` 只建立一次：

```text
frozen field contact_order
∩ analysis/cache channel names
∩ finite in all seven primary bands over [0,10] s
∩ finite in the parent BB150 anchor
```

该 mask 同时用于 parent anchor、七带、R2、R3、observed 与全部 null draws。

已完成的只读 audit 预期：167/167 events 七带齐全；共同触点 min=6、median=11、max=16。Runner 必须重新 fail-closed 核验。任一 event `<6` 时停止，不得按频带静默更改 denominator。

---

## 3. Locked gradient R3 dense-grid scorer

建议新增纯函数模块：

```text
src/topic5_gradient_grid_field.py
```

不要改变 frozen-field producer 或旧 scorer 的既有语义。

### 3.1 Outcome-independent routing

```text
complete shared_a + shared_b + shared plane -> shared
otherwise complete own_a + own_b           -> own_fallback
otherwise                                  -> fail closed
```

n=17 的预期 routing 为 7 shared / 10 own fallback。不得根据 ictal score、band、p 值或哪条 route 更高来选择。

### 3.2 一个 subject-fixed sigma

```python
plane_a = "shared" if route == "shared" else "own_a"
sigma_common = field["planes"][plane_a]["sigma"]
```

`sigma_common` 必须用于：

- TA 与 TB；
- parent anchor 与七频带；
- observed 与全部 null draws；
- R2 与 R3；
- Stage 2 中同一 subject 的 Fig3-B/Fig3-C statistical scorer。

TA/TB 仍分别保留自己的 `earliness_a/b` 与 `support_a/b`。own fallback 时，A/B 使用各自 `own_a/own_b` plane points；绝不能把 TB 放到 TA plane/support 上。

### 3.3 81×81 adaptive grid

保留旧 endpoint 的 81×81 dense-grid 思路，但禁止直接复制旧固定域：

```text
X=[-0.5,1.5], Y=[-1,1]
```

只读几何审计显示：n=17 实际使用的 27 个 plane（7 shared + 10×A/B own）中，22/27 至少有一个 gradient contact 落在旧域外，共 55 个 contact-plane 位置。固定旧域会系统漏掉这些触点周围的 field evaluation area。

每个实际 plane 的 grid 只能由 frozen interictal geometry、template support 与 `sigma_common` 预先生成：

```python
S_budget = max(sum(support_vector) for support_vector sharing_this_grid)
r_support = sigma_common * (
    sqrt(2 * log(max(S_budget / 0.15, 1.0))) + 1.0
)

x_lo = min(points[:, 0]) - r_support
x_hi = max(points[:, 0]) + r_support
y_ext = max(abs(points[:, 1])) + r_support

x = linspace(x_lo, x_hi, 81)
y = linspace(-y_ext, y_ext, 81)
Y, X = meshgrid(y, x, indexing="ij")
```

规则：

- shared A/B：同一 points、同一 grid；`S_budget=max(sum(support_a),sum(support_b))`；
- own fallback：A/B plane 不同，各建自己的 grid；各自使用对应 support；
- y 必须严格关于 0 对称，`flip(axis=0)` 才精确等于 transverse mirror；
- 输出每个 grid 的 model、bounds、spacing、shape、sigma、support budget 与 SHA-256；
- 构造后断言 `S>=0.15` 的 support region 不接触任何 grid 边界；否则停止，不得边跑边扩大网格。

### 3.4 Field construction

对每个模板 `T∈{A,B}`：

```text
K[g,i] = exp(-||grid_g - point_i||² / (2*sigma_common²))

S_inter[g] = Σ_i K[g,i] * support_T[i]
F_inter[g] = Σ_i K[g,i] * support_T[i] * earliness_T[i] / S_inter[g]

S_ictal[g] = Σ_i K[g,i] * support_T[i] * finite_i
F_ictal[g] = Σ_i K[g,i] * support_T[i] * finite_i * activation_i / S_ictal[g]
```

- `F_inter` 使用完整 frozen interictal field；
- `F_ictal` 使用该 event 的 locked common finite-contact mask；
- null 只移动 activation value，geometry/support/contact availability 不移动；
- support 使用模板自己的 participation support；
- 不把 support 当 activation value 再算一次，避免 double-count participation。

### 3.5 Support gate、mirror 与 maxAB

Primary constants：

```text
S_THRESH = 0.15
OVERLAP_MIN at N=81 = 25 pixels
```

Identity 与 mirror 必须各自独立做 support-overlap gate：

```python
M_id = (S_inter >= .15) & (S_ictal >= .15) & finite(F_inter) & finite(F_ictal)
M_mir = (S_inter >= .15) & (flip(S_ictal,0) >= .15) \
        & finite(F_inter) & finite(flip(F_ictal,0))
```

然后：

```text
r_T = eligible candidate with maximum abs(r)
score_T = abs(r_T)
score_maxAB = max(score_A, score_B)
```

必须保存：

- `r_identity` / `r_mirror`；
- 两个候选各自的 overlap pixels/fraction；
- `mirror_choice`；
- chosen signed r 与 abs r；
- A/B scores 与 `best_template`。

禁止调用旧 `corr_pair_mirror_invariant()` 后再取 abs。应复用或等价实现 `corr_pair_mirror_invariant_signed()` 的 abs-max 选择。

若 A/B 任一模板的 identity 与 mirror 都不合格，固定 167-event primary run 必须停止并报告，不能静默删 event 或降为单模板。

### 3.6 R2 paired sensitivity

R2 必须在本次运行内用同一输入重算，不能读取旧 R2 subject summary。它与 R3 必须共享：

- event keys、common mask；
- routing、A/B points/support；
- `sigma_common`；
- activation vectors；
- permutation mappings；
- corrected mirror/maxAB；
- seizure→subject fold。

唯一差别：R2 在 frozen contact positions 评估核场；R3 在 dense grid pixels 评估。

输出 subject-level `R3−R2` data、null 与 margin differences。没有预先定义并通过 equivalence test 时，不得写“两种方法等价”。

### 3.7 Resolution sensitivity

Primary 固定 `N=81`；同时在完全相同 bounds 上计算 `N=161`：

```text
overlap_min(N) = ceil((25 / 81²) * N²)
```

预设工程验收：

- observed event×band `|r81-r161|` 的 95th percentile `<=0.02`；
- 任一 subject-level data median 或 null median 差 `<=0.02`；
- support mask 不接触 grid 边界；
- finite/nonfinite eligibility 不改变。

不满足时不得交付 81 版；先检查 bounds/spacing，或在报告偏差后统一提升分辨率并完整重跑。禁止按哪个分辨率更显著来选择。

---

## 4. Locked null 与 fold

### 4.1 Primary all-contact null

```text
n_perm = 1000
seed = 20260718
```

每个 `(subject, seizure_idx, draw)` 只生成一套 physical-contact permutation：

- seed/hash 不得依赖 band、R2/R3、A/B 或 phenotype group；
- 只在 common finite contacts 内洗牌；missing contacts 保持 missing；
- 同一 mapping 同时用于 A、B、parent anchor、七带、R2、R3；
- pooled/phenotype views 从同一已评分 event 读出，不重新抽 null。

每个 draw 必须完整重做：

```text
permute activation
→ rebuild ictal grid field/support
→ identity/mirror candidate-specific overlap
→ abs-max mirror choice
→ A/B maxAB
```

输出 permutation mapping hash audit；随机 seed 只能依赖 subject + seizure + draw，不能依赖 band。

### 4.2 Secondary pure within-shaft null

- 只作 anatomical sensitivity；
- `min_group_for_shaft=4`；
- 不允许 distance-bin 或 subject-wide fallback；
- 不能合法覆盖所有 common finite contacts的 event 标 `unavailable`；
- 单独报告 eligible subject/event denominator；不得沿用图上的 n=17；
- 若 all-contact 有 evidence 而 pure within-shaft 无 evidence，claim ceiling 仍是 coarse patient-specific scaffold。

### 4.3 Subject-first folding

For each subject `s`, readout `b`, draw `k`：

```text
D[s,b]    = median_event(observed score)
N[s,b,k]  = median_event(null score at draw k)
Nmed[s,b] = median_k N[s,b,k]
Delta[s,b]= D[s,b] - Nmed[s,b]
```

禁止先 pool seizures 再做 cohort test。

---

## 5. Stage 1 — parent cohort 与七频带重算

### 5.1 Suggested runner

```text
scripts/run_topic5_figure3_ictal_grid_rebuild.py
tests/test_topic5_parent_matched_grid_field_concordance.py
```

目标接口：

```bash
python scripts/run_topic5_figure3_ictal_grid_rebuild.py --validate-only

python scripts/run_topic5_figure3_ictal_grid_rebuild.py \
  --n-perm 20 --seed 20260718 \
  --outdir /tmp/topic5_fig3_r3_smoke

python scripts/run_topic5_figure3_ictal_grid_rebuild.py \
  --n-perm 1000 --seed 20260718 \
  --outdir results/topic5_ictal_recruitment/field_concordance_grid_parent_matched

python scripts/run_topic5_figure3_ictal_grid_rebuild.py \
  --verify-only \
  --outdir results/topic5_ictal_recruitment/field_concordance_grid_parent_matched
```

这些是待实现接口，不是假定当前已经存在的命令。

### 5.2 Parent cohort statistic

必须产出三个 paired Data-vs-Null groups：

| group | expected n subject / event | activation |
|---|---:|---|
| Pooled | 17 / 167 | strict events 用 BB150；gamma-nonBB events 用 30–80 |
| Broadband | 16 / 106 | BB 1–150 |
| Gamma | 11 / 61 | 30–80 |

每组同时输出：data/null median + IQR、subject margin、positive count、one-sided paired Wilcoxon、coherent cohort spatial-null p、two-sided subject sign-flip sidecar。

### 5.3 Band inheritance

每个 band 输出：

- `D`、`Nmed`、`Delta`；
- subject Delta median、IQR、positive count；
- paired one-sided Wilcoxon；
- synchronized spatial-null cohort permutation p；
- coherent seven-band maxT pFWER。

MaxT 使用完全同步的 cross-band draws：

```text
Cobs[b]    = median_subject D[s,b]
Cnull[b,k] = median_subject N[s,b,k]
Zobs[b]    = Cobs[b] - median_k Cnull[b,k]
Znull[b,k] = Cnull[b,k] - median_k Cnull[b,k]
M[k]       = max_b Znull[b,k]
pFWER[b]   = (1 + #{M[k] >= Zobs[b]}) / (K + 1)
```

### 5.4 Direct band specificity

基于完整 `17×7 Delta` matrix：

- primary omnibus：Friedman rank statistic；
- 用至少 100,000 次 subject 内 band-label permutation 校准；
- 报 Kendall's W；
- 全部 21 个 paired contrasts 均输出；
- pairwise two-sided Wilcoxon + Holm；
- 每个 contrast 报 paired median difference、IQR 与 Holm p。

Omnibus 没有 evidence 时，解释必须是 band-generic；不得根据单带星号挑 winner。

### 5.5 Required paper-ready staging figures

计算层输出先进入 parallel root：

```text
results/topic5_ictal_recruitment/field_concordance_grid_parent_matched/
```

图先进入 staging，不覆盖现有 paper-ready：

```text
results/paper-ready-figure/fig3_ictal_field_concordance_grid_rebuild/
```

必须包含：

```text
figures/field_concordance_cohort_stat.{png,pdf}
figures/field_concordance_cohort_stat_metadata.json
figures/multiband_field_concordance_stat.{png,pdf}
figures/multiband_field_concordance_stat_metadata.json
figures/r2_vs_r3_sensitivity.{png,pdf}
figures/multiband_within_shaft_sensitivity.{png,pdf}
figures/README.md
```

图形合同：

- cohort 图复用现有 Data-vs-Null violin/box/paired subject point grammar；横轴 Pooled/Broadband/Gamma；
- multiband 图 y 轴为 `R3 grid-field concordance − all-contact null median`；每点一名 subject；黑横杠为 cohort Delta；
- 星号只对应 seven-band pFWER；
- 标注真实 n，不出现旧 F2、184 seizures 或 mixed-null wording；
- README 逐图 2–4 句中文，末行 `**关注点**：`。

---

## 6. Stage 2 — Figure 3 其余定量 field panels

### 6.1 Fig3-B score provenance

- 固定 `epilepsiae_1146`, seizure 2；
- 保留现有 morphology gate、candidate provenance 与视觉选择；
- 禁止依据新 R3 score 重新挑 seizure；
- 使用 Stage 1 同一 R3 engine、shared grid、sigma、support 与 `[0,10] s` BB150 activation；
- metadata 明确分开：`statistical_r3`、`paired_r2_sensitivity`、`display_field_6mm`；
- 现有 6 mm physical-mm display smoothing 不得冒充 statistical sigma；
- staging 中重画/核验，用户目检前不覆盖 locked figure。

### 6.2 Fig3-C peri-onset R3 migration

只要 Fig3-C 继续留在 Figure 3，就必须同步迁移，不能只替换曲线而沿用旧 R2 null/p 值。

固定 shared-only subjects：

```text
epilepsiae_1084
epilepsiae_1146
epilepsiae_384
epilepsiae_548
epilepsiae_583
epilepsiae_590
epilepsiae_958
```

合同：

- 不扩成 n=17，不回退 own；
- 沿用当前 canonical manifest 的 successful seizure set 与 denominator flow；
- time span `[-120,+20] s`，window=10 s，step=2 s，66 window centers；
- R3 `maxAB |r|` 与 signed A/B 全部重算；
- 每个 subject 使用其 frozen shared grid 与同一 `sigma_common`；
- 每个 `seizure×draw` 只抽一次 spatial mapping，贯穿全部 66 windows；
- window missing 值保持 missing，mapping/contact universe 不按时间窗重抽；
- all-contact 与 pure within-shaft 均重跑；
- maxT/cluster correction 必须从新的 R3 null matrices 重算；
- 保存每窗 support overlap、finite contacts、mirror choice 与 best template。

Stage 2 required artifacts：

```text
fig3b_r3_score_provenance.json
peri_onset_r3/subject_index.csv
peri_onset_r3/per_seizure_window.csv
peri_onset_r3/subject_summary.csv
peri_onset_r3/spatial_null_stats.csv
peri_onset_r3/spatial_null_matrices.npz
peri_onset_r3/run_manifest.json
```

---

## 7. Required artifact inventory

```text
contract_manifest.json
input_hashes_before_after.json
cohort_event_inventory.csv
field_routing_sigma_grid_inventory.csv
common_contact_inventory.csv
support_overlap_inventory.csv
permutation_mapping_audit.parquet
permutation_mapping_audit_summary.csv

parent_anchor_event.csv
parent_anchor_subject.csv
parent_anchor_subject_null_draws.parquet
parent_anchor_cohort.csv

multiband_event.csv
multiband_subject.csv
multiband_subject_null_draws.parquet
multiband_cohort.csv
multiband_band_omnibus.json
multiband_band_contrasts.csv
multiband_phenotype_subject.csv
multiband_phenotype_cohort.csv

r2_sensitivity_event.csv
r2_sensitivity_subject.csv
r2_sensitivity_cohort.csv
r2_r3_subject_comparison.csv
r2_r3_cohort_comparison.csv
r2_r3_grid_convergence.csv

within_shaft_event_inventory.csv
within_shaft_subject.csv
within_shaft_cohort.csv
within_shaft_subject_null_draws.parquet

fs_edge_sensitivity.csv
drop_inventory.csv
summary.json
figures/README.md
```

`contract_manifest.json` 至少记录：git commit、Python/NumPy/SciPy/pandas、seed、n_perm、event-list hash、input hashes、R3 formula version、grid fingerprints、support/overlap constants、routing、sigma rule、band definitions、null modes、fold 与所有 output paths。

---

## 8. Required tests and validation

至少覆盖：

1. 17 subjects / 167 unique event keys 精确锁定；
2. 106 strict + 61 gamma，且互斥并集为 167；
3. 七带 + parent anchor common mask 完全相同且每 event `>=6`；
4. frozen field fingerprint fail-closed；
5. routing 固定 7 shared / 10 own，不随结果变化；
6. y-grid 对称且 `flip(axis=0)` 精确实现 mirror；
7. adaptive bounds 覆盖全部 `S>=0.15` region 且不触边；
8. adversarial negative-mirror case 选择 `max(abs(r_id),abs(r_mir))`；
9. A/B plane 与 support 分离；
10. `sigma_common` 跨 A/B、bands、anchor、R2/R3、obs/null 一致；
11. template full support 与 event finite activation support 的公式正确；
12. candidate-specific overlap gate；
13. 同一 event/draw permutation hash 跨 bands/anchor/R2/R3 完全相同；
14. 每个 draw 重算 smoothing/support/mirror/maxAB；
15. seizure→subject fold，不 pool seizures；
16. coherent seven-band maxT；
17. direct band omnibus 与 21 contrasts；
18. pure within-shaft 无 fallback；
19. 81/161 convergence；
20. 新 R3 engine 在旧 endpoint fixed-grid synthetic/reference input 上复现旧 `smooth_field`，只允许 corrected-mirror adversarial case 出现预期差异；
21. deterministic rerun parity；
22. input hashes before/after 一致；
23. Fig3-B 不重新选 exemplar；
24. Fig3-C mapping 跨 66 windows 固定，R3 null matrices 与图中 p/cluster 一致。

建议测试：

```bash
pytest -q \
  tests/test_topic5_parent_matched_grid_field_concordance.py \
  tests/test_topic5_contact_similarity.py \
  tests/test_topic5_axis_alignment.py \
  tests/test_topic5_gradient_multiband_significance.py
```

最后运行：

```bash
git diff --check
```

并人工核对 PNG/PDF、metadata 与 CSV 中数值一致。

---

## 9. Stop conditions

以下任一项出现即停止并回报，不得自行缩分母或改方法：

- event list 不是 17 subjects / 167 unique seizures；
- strict/gamma 不是 106/61 或不互斥；
- frozen fingerprint mismatch；
- 任一 event common contacts `<6`；
- 任一 band/event 静默缺失；
- shared/own route 根据 outcome 变化；
- A/B 不能分别使用正确 plane/support；
- support region 接触 grid boundary；
- R3 candidate overlap 不合格；
- 81/161 数值稳定性不满足预设门；
- 同一 mapping 不能跨 bands/anchor/R2/R3 复用；
- null draws 不完整或含非预期 NaN；
- within-shaft 必须 fallback 才能运行；
- cohort inference pool seizures；
- input artifacts 被改写；
- Fig3-C 只更新 observed curve 而没有同步更新 null/correction。

以下不是 stop condition：

- R3 null 比预期高；
- R3 与 R2 结论不同；
- parent anchor 不显著；
- 某个或全部频带不显著；
- band omnibus 为 null；
- within-shaft sensitivity 不支持；
- 没有 winner band；
- Fig3-C 不出现 onset-specific increase。

所有结果无论正负均完整落盘，不设 outcome-based PASS/FAIL。

---

## 10. Allowed / forbidden reporting

### Allowed, depending on results

- “Early-ictal energy fields show subject-level concordance with frozen interictal HFO timing fields above an all-contact label-shuffle null.”
- “The effect is distributed across bands”或“direct paired band test supports a band effect”。
- “The association is detectable at a coarse patient-specific field/scaffold level.”
- 只有 pure within-shaft 也有 evidence 时，才能写“survives shaft-preserving relabeling”。

### Forbidden

- “17/17 subjects universally significant”；
- “band A significant、band B non-significant，所以 A>B”；
- “all-contact null controls shaft identity/local smoothness”；
- “R3 显著而 R2 不显著，所以场是真实、触点是假象”；
- “endpoint 旧 null 可直接迁移到 gradient”；
- “发作按间期顺序逐点 replay”；
- “方向、极性或因果机制已证明”；
- “HFO/ripple-specific mechanism” without direct band evidence and required controls；
- 按结果排 subject/event/band、改 grid、改 route 或挑 resolution。

---

## 11. Copy-ready message to the execution agent

```text
请严格执行：
docs/archive/topic5/fig3_ictal_gradient_r3_full_recompute_handoff_2026-07-18.md

目标是把 Figure 3 中承载定量 field-similarity 的发作相关部分统一迁移到 gradient R3 dense-grid scorer，而不是续跑旧 F2，也不是复用旧 endpoint n=20 数值。

Stage 1：以 clinical_onset_gradient_field_cohort_stat_event.csv 中 group_id=all_phenotype_matched 的 17 subjects / 167 seizures 为唯一母清单，重算 parent cohort Data-vs-Null 和七频带统计。Primary 固定为 R3；R2 必须在同一次运行中用相同 event、mask、routing、sigma、activation、permutation 和 fold 重算，只作 paired sensitivity。Primary null 为 coherent all-contact shuffle；同一 subject×seizure×draw mapping 跨 A/B、七带、parent、R2/R3 复用。Pure within-shaft 无 fallback，只作 secondary。

Stage 2：保留 E1146 seizure 2，不重新选 Fig3-B exemplar，只更新 R3 score provenance；若 Fig3-C 继续留在 Figure 3，则对现有 7 名 shared-only、66-window trajectory 及其 all-contact/within-shaft spatial null、maxT/cluster correction 全部重算为 R3。

禁止覆盖旧结果。先写 parallel calculation root 与 paper-ready staging root。无论结果正负都完整落盘，并回填：
docs/archive/topic5/fig3_ictal_gradient_r3_full_recompute_run_form_2026-07-18.md

交付时只报告：Stage 1/2 完成状态、实现与测试、真实 cohort/event/contact/grid inventory、parent R3、七带 vs null、direct band test、R2–R3 paired diagnostic、pure within-shaft、Fig3-B/Fig3-C 状态、deviations 与新 artifact paths。不要修改 manuscript claim wording，不要预选 winner。
```
