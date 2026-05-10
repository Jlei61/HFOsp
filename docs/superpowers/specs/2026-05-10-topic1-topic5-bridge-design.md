# Topic 1 × Topic 5 Bridge — Design Spec (2026-05-10)

> **Tier**：exploratory（"hammer-meets-nail"），cohort N=10，**不开 cohort α 主张**。
> **Status**：design spec，待 user 审阅 → 进 writing-plans。
> **Topic**：把 Topic 1 PR-2 内部传播 stable_k=2 "两模板" 与 Topic 5 PR-1 z-ER ictal seizure subtypes 在 within-subject 层面对齐，验证间期模板状态是否在 pre-ictal 窗口内随 ictal subtype 系统改变。
> **Cohort 限制**：Topic 5 = 16 epilepsiae（yuquan 未建 v2.3 atlas）；Topic 1 stable_k=2 全 cohort 覆盖；交集 audit 已落 `results/topic1_topic5_bridge_audit.csv`。

---

## 1. 一句话主张（locked framing）

在一个 N=10 的 epilepsiae 探索性 cohort 内，每 ictal seizure 在 [-30, -1] min pre-ictal 窗口内的 interictal HFO group-event template (T0/T1) 指纹**条件依赖**于该 seizure 的 ictal subtype label。Cohort 判定 = "≥2/3 sensitivity 窗口在 binomial count + dual-gate 下 PER-WINDOW-PASS"，**禁止**任何"predicts subtype" / cross-subject claim。

## 2. 假设结构

### 2.1 Q1 (primary)

**H1**：`per_seizure_fingerprint × ictal_subtype` 在 per-subject 内 statistical association（exploratory α/3 内任一过 → subject positive）。

| feature | 类型 | per-subject 检验 | effect size |
|---|---|---|---|
| `frac_T0` | continuous | KW (k_s≥3) / Mann-Whitney (k_s=2) | ε² (KW) / rank-biserial r (MW) |
| `switch_rate` | continuous | 同上 | 同上 |
| `last_template` | categorical {T0,T1} | Fisher exact / χ² | Cramér V |

**Pre-registered T0/T1 sign convention**：`T0 = template with larger mean assignment fraction across the entire recording`（ties broken by template index）。在 `bridge_setup.json` 落盘，per-subject 独立 frozen，整个分析期间不变。

**Sensitivity battery**：3 个 pre-ictal 窗口 `[-15,-1] / [-30,-1] / [-60,-1] min` 全部并跑，独立报。**Primary cohort judgement = ≥2/3 窗口 cohort 阳（H2）**；任一窗口阳作为 H1 启动条件。

### 2.2 Q1b sentinel — 442 binary-outlier

非 cohort claim。442 (γ_gamma=1, 1 主子型 + 1 outlier sz=9, swap_class=none, silhouette=0.255) 上跑：

> sz=9 vs 其他 16 sz 的三维指纹 exact Mann-Whitney + Fisher exact（per-feature）。

**用途**：descriptive case study。
- 阳性：方法在 topic1 几何弱的 subject 上仍能识别 topic5 outlier seizure，方法层 robustness。
- 阴性：indeterminate（不是 "negative control passed"）。

报告独立段落，不进 cohort statistic。

### 2.3 Q2 — **deferred, NOT in first implementation**

Q2 (z-ER channel-onset rank × interictal template rank correspondence) 有两个 blocker：
1. **Schema blocker**：14/16 topic5 binned JSON 没暴露 `channels_used`，需要 schema patch
2. **Cohort 太小**：`swap_class ∈ {strict, candidate} ∩ γ_gamma ≥ 2` = 5 subject，sign-test 功效差到一定程度

**决定**：Q2 整体推到 follow-up（§8 项目 1）。**第一轮实现仅 Q1 + Q1b + Q3**。Q2 hypothesis、step、code 均不在本 spec 实施范围；当 Q1 出 PASS / NULL / INDETERMINATE 后再评估是否值得做 Q2 schema unblock + 实施。

### 2.4 Q3 (stratifier, descriptive only)

Q1 cohort effect-size distribution 按两个 stratifier 二分作 2×2 散点：
- `swap_class ∈ {strict, candidate}` vs `none`
- `silhouette_median > 0.5` vs `≤ 0.5`

**不开 α**。仅作图层，看 Q1 信号是否集中于 topic1 几何"真"的子集。

## 3. Cohort gating

### 3.1 Q1 cohort（gamma band 主）

**Inclusion**（必须全部满足）：
1. Topic 5 PR-1 `gamma_ER` status = `ok`
2. Topic 5 PR-1 `gamma_n_subtypes ≥ 2`（含 outlier-only 不算）
3. Topic 1 `adaptive_cluster.stable_k = 2`
4. `n_seizures_with_topic5_label ≥ 4`（per-subject KW power floor）
5. `n_pre_ictal_events_per_seizure_median ≥ 5` for [-30,-1] min 窗口
6. Topic 5 audit-rerun 完成（消费 audit-rerun labels，不消费 pre-audit）

**Audit-derived cohort（N=10，basis = pre-audit labels，audit-rerun 完成后 re-audit）**：

```
1073 1096 1146 253 548 590 635 916 922 958
```

**Excluded with reason**：
- `1077` (γ=0 全 insufficient_n)
- `442` (γ=1, → Q1b sentinel)
- `1084` (γ_gamma=1; γ_broad=2 但 broad 是 sensitivity 不主跑)
- `139, 1150` (γ_gamma=1)
- `583` (median pre-ictal events = 0; topic1 block coverage 仅 2/22 sz)

**Sensitivity / sentinel cohort**：
- `442` → Q1b
- `1084` → broad-band 双轨 sensitivity（broad γ=2 跑同套统计，独立报）

### 3.2 Q1b sentinel cohort

`442` 单 subject。

### 3.3 Q2 cohort（**deferred, 不在本 spec 实施**）

Q2 整体推迟到 follow-up（§2.3 / §8 项目 1）。Cohort 估算如下（仅供 follow-up 评估时参考，**不**做 first-round 实现 gate）：

`{swap_class ∈ {strict, candidate}} ∩ γ_gamma ≥ 2 ∩ Q1 cohort` =
- strict: `1073, 1146, 635, 958`（n=4）
- candidate: `548`（n=1）
- 总 N_Q2 = 5

前置 unblock = topic5 PR-1 schema patch (`channels_used`)。

### 3.4 Q3 stratifier

应用于 Q1 cohort（N=10）：
- swap-real (strict ∪ candidate) ∩ Q1: `1073, 1146, 548, 635, 958`（n=5）
- swap-none ∩ Q1: `1096, 253, 590, 916, 922`（n=5）

5×5 平衡，便于 stratified 比较。

### 3.5 Per-window eligibility & no-event seizure 合同（locked）

**每个 window 独立判定 subject + seizure eligibility**。不允许实现时静默回填默认值。

**Per-seizure eligibility for window W ∈ {[-15,-1], [-30,-1], [-60,-1]}**：

| 条件 | 处理 |
|---|---|
| `n_events_in_W ≥ 3` | seizure 计入 per-subject 检验 |
| `1 ≤ n_events_in_W ≤ 2` | seizure 计入但 fingerprint 仅 `last_template` 有定义；`switch_rate` = NaN（n=1 时 0 个 transition）；`frac_T0` 仍可算但权重低 |
| `n_events_in_W = 0` | seizure **drop**，**不**伪造 fingerprint（不写 `frac_T0=0.5` / `switch_rate=0` 之类默认值）；记录到 `per_subject/<sid>__bridge.json[window].dropped_seizures` |
| seizure clinical onset 缺失（`clin_onset_epoch` NaN 且 `eeg_onset_epoch` NaN） | seizure **drop** + 记录 `dropped_reason="no_onset"` |

**Per-subject eligibility for window W**：

| 条件 | 处理 |
|---|---|
| `n_seizures_with_subtype_label_and_W_eligible ≥ 4` | subject 计入该 window 的 cohort denominator |
| 否则 | subject **从该 window 退出**；`per_window_status[W] = "ineligible"`；不计入分子，**也不计入分母** |

**Per-window cohort denominator**：

```
denom_W = | { subject : n_eligible_seizures_in_W ≥ 4 } |
```

`denom_W` 可能 < 10 if 某 window 上某 subject 事件不足。Cohort PER-WINDOW-PASS 阈值要求 `count_positive_W / denom_W` 与 binomial(`denom_W`, p_null) 比，**不是**写死 `count ≥ 3/10`。

**Locked rule**：

```
PER-WINDOW-PASS_W iff binomial_one_sided_p(count_positive_W, denom_W, p_null=0.049) < 0.05
```

`p_null = 0.049` 来自 dual gate 在独立 null 下的近似（3 features 取 OR + α/3 单 feature gate；ε² > 0.10 在 null 下概率 ≈ 0 但保守取 1，使 p_null 上限 = 1 − (1 − α/3)³）。

**预期 denominator**：根据 audit `n_pre_ictal_events_per_sz_min` 列：
- [-30,-1] min: 全 N=10 都 eligible（min event count ≥ 5 across all cohort subjects）
- [-15,-1] min: 期望 `denom_15 ∈ [9, 10]`（548 min=2 在 30min 内，缩到 15min 可能 1-2 sz drop，但 n_seizures ≥ 4 仍可能保住）
- [-60,-1] min: 全 N=10 期望 eligible

实际 `denom_W` 在 audit re-run（spec implementation 第一步）落盘到 `cohort_audit.csv`。

**禁止**：
- 不准用 0-event seizure 的"默认指纹"补 cohort
- 不准在某 window 上 denom < 10 时偷换为"分子的实际 N"
- 不准跨 window pool seizure（每 window 独立 cohort）

## 4. 统计 contract

### 4.1 Per-subject (dual gate, **same-feature AND**)

**Per-subject Q1-positive iff**：

```
∃ feature f ∈ {frac_T0, switch_rate, last_template} :
    p_f < α/3 = 0.0167  AND  |effect_f| > 0.10
```

**关键合同**：dual gate 必须是**同一 feature 内**的 (p, effect) 同时通过；**不允许** feature A 提供 p、feature B 提供 effect 拼凑出 positive。Implementation 上等价于：先 per-feature 计算 (p, effect) 二元组，per-feature 判定通过/不通过，再对 3 个 feature 取 OR。

**双 gate 理由**：单纯 p-value 偏向高 event-count subject（922 ≈1300 events，任 1% 漂移即 reject）；单纯 effect-size 在低 n 无 type-I 控制。同一 feature 内 AND 同时控制 type-I 与 effect threshold；3 features 取 OR 是"任一非平凡指纹存在即 subject positive"的合理化表达。

### 4.2 Cohort

**Per-window 判定（PER-WINDOW-PASS）**：
- 每 window 独立 denominator `denom_W`（详见 §3.5；可能 < 10 if 某 subject n_eligible_sz < 4）
- count(per-subject Q1-positive) per window，与 binomial(`denom_W`, p_null=0.049) one-sided
- **PER-WINDOW-PASS iff** `binomial_one_sided_p(count_positive_W, denom_W, 0.049) < 0.05`
- 等价描述（reference）：当 `denom_W = 10` 时阈值 = `count ≥ 3`（p≈0.012）；`denom_W = 9` 时阈值 = `count ≥ 3`（p≈0.018）；`denom_W = 8` 时 `count ≥ 3`（p≈0.025）
- 三个窗口独立报；**不**做窗口间 multiple-testing 校正（sensitivity battery，不是独立 claim）

**Cohort 判定（locked, primary）**：
- **COHORT-EXPLORATORY-PASS** = ≥ 2/3 windows 通过 PER-WINDOW-PASS
- **NULL-locked** = 0/3 windows 通过 AND 所有 windows count ≤ 1/10
- **INDETERMINATE** = 其他

**Secondary descriptive (k_s=2 subset)**：
- 限定 `gamma_n_subtypes = 2` 的 subject（cohort 内：`1073, 1096, 1146, 253, 590, 635, 916, 922` = 8 subject）
- 主 feature = `frac_T0`
- per-subject 一侧 z-score: `z_s = sign(ε(subtype 0)) × Φ⁻¹(1 − p_s/2)`
  - sign 由 "subtype 0 中 frac_T0 mean − subtype 1 中 frac_T0 mean" 决定
  - subtype 编号继承 topic5 PR-1（按 size 降序）
- Stouffer combined Z = `Σ z_s / √n`，one-sided p
- 仅 [-30,-1] min 主窗跑；不进多窗 sensitivity battery

### 4.3 Effect-size distribution（必报）

- per-subject feature-winner ε²/V 的 cohort 分布（median, IQR, 全 list）
- per-subject feature-winner identity 的频次（哪个 feature 最常 winning）

### 4.4 3-state failure contract

| 状态 | 主条件（必须全满足） |
|---|---|
| **COHORT-EXPLORATORY-PASS** | ≥ 2/3 windows 通过 PER-WINDOW-PASS |
| **NULL-locked** | 0/3 windows pass AND median ε²/V ≤ 0.05 AND Stouffer Z one-sided p > 0.5 |
| **INDETERMINATE** | 其他 |

任何 paper-level 表述只能在 **COHORT-EXPLORATORY-PASS** 下提及，且必须带 "n=10, exploratory" caveat。
Per-subject Q1-positive 中 ε²/V > 0.10 已是 dual gate 的必要条件，cohort 层不再叠加 effect-size 阈值（避免 vacuous 复述）。

### 4.5 Sentinel verdicts (descriptive, **no hard gates**)

- **548**：高信息量 case（γ=5 给最大 KW degrees-of-freedom；swap_class=candidate；split=strong；silhouette=0.627）。Spec **不**给出数值 gate（如 "top-3 ε²"）。548 minority_jaccard soft failure (`gamma=0.25` / `broad=0.43`) 已说明 user-marked outlier subset 在 cluster 内 recall 不完美；ε² 排名不可作 hard expected outcome。报告时仅作 case description：列出 548 在每 window 的 (p, effect) 与 PER-WINDOW-PASS 状态，不与 cohort 平均比较。
- **442**：Q1b sentinel，独立段落（§2.2）。
- **1073/1146/635/958 ("strict swap × γ≥2" 脊梁)**：Q3 stratifier 视图主舞台（描述层）。

## 5. 代码 / 数据架构

### 5.1 复用基础设施

| 来源 | 接口 | 提供 |
|---|---|---|
| `src/interictal_propagation.py::load_subject_propagation_events` | 现有 | per-event timestamps + block_time_ranges + lagPatRank |
| `src/interictal_propagation.py` | `adaptive_cluster.clusters[2].assignments` | per-event template_id (Topic1) |
| `results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/per_subject/<sid>__zer_binned.json` | 直接读 | per-seizure subtype_label, ictal_clinical_onset |
| `results/epilepsiae_seizure_inventory.csv` | 直接读 | clin_onset_epoch (fallback eeg_onset_epoch) |
| `src/event_periodicity.py::match_bipolar_soz / match_bipolar_focus_rel` | 现有 | （Q2 deferred；本 spec 不消费） |

### 5.2 新代码

**新 module**：`src/topic1_topic5_bridge.py`（仅 Q1 + Q1b + Q3 范围）
- `compute_pre_ictal_fingerprint(event_times, event_template_ids, seizure_clinical_onsets, window_min_max_min)` → DataFrame[seizure_idx, n_events, frac_T0, switch_rate, last_template]；`n_events=0` seizure drop with reason
- `register_t0_t1_convention(adaptive_cluster_assignments)` → dict[subject → template_id_for_T0]，落 `bridge_setup.json` 永久 freeze（idempotent，重跑不变）
- `q1_per_subject_test(fingerprint_df, subtype_labels, alpha_within=0.0167, effect_min=0.10)` → per-feature (p, effect, passed) tuple；同 feature AND；3 features 取 OR；返回 subject_positive bool
- `q1_cohort_per_window(per_subject_results, denom_W)` → PER-WINDOW-PASS 判定 + binomial p
- `q1b_sentinel_442(...)` → 独立 path（非 cohort）
- `q3_stratifier_summary(q1_results, swap_class_map, silhouette_map)` → 4-cell stratified table + matplotlib helper

**Q2 不实现**（§2.3 deferred）。

**新 driver**：`scripts/run_topic1_topic5_bridge.py`
- subcommands: `setup` (T0/T1 freeze + cohort_audit re-run) / `per-subject` / `cohort` / `sentinel-442` / `figures`
- 启动时 assert `cohort_zer_audit_*.log` 已有 `[cohort] cohort_summary.csv → ...` line（audit-rerun done marker），不在则 abort
- **不**包含 `q2` subcommand

**测试**：`tests/test_topic1_topic5_bridge.py`
- T0/T1 freeze 在重跑 idempotent（同样 input → 同样 output）
- dual gate 同 feature AND 在伪数据上正确（feature A 高 p / feature B 高 effect 拼凑 → 必须 NOT positive）
- 3-feature OR 在伪数据上正确（任一 feature 双 gate pass → subject positive）
- 0-event seizure 被 drop 且 dropped_reason 写入 per-subject JSON（不伪造 fingerprint）
- 1-event seizure switch_rate=NaN，不参与 frac_T0 KW（或参与但权重显式标注）
- pre-ictal window cut 用 event-level timestamps（hand-crafted 测试 input：3 events 落在 [-30, -1] min 真值，验证函数返回 3 个）
- per-window denominator 计算正确（hand-crafted：subject A 在 [-30,-1] 5 sz eligible，[-15,-1] 3 sz eligible → denom_30=入，denom_15=不入）

### 5.3 输出目录

```
results/topic1_topic5_bridge/
├── bridge_setup.json                 # T0/T1 convention freeze + audit-rerun marker
├── cohort_audit.csv                  # final inclusion/exclusion + per-window denom_W
├── per_subject/
│   └── <sid>__bridge.json            # 10 个 cohort + 442 + 1084(broad sensitivity)
│                                     # schema: { window: { fingerprint_df, dropped_seizures, q1_test_per_feature, subject_positive } }
├── cohort_summary.json               # PER-WINDOW-PASS / COHORT-EXPLORATORY-PASS / Stouffer + 3-state verdict
├── q1b_442_sentinel.json             # 442 sz=9 vs 其他 16 sz exact tests
├── figures/
│   ├── README.md                     # 中文，必须存在
│   ├── q1_cohort_count_x_window.png  # 主图：3 windows × (count / denom)
│   ├── q1_effect_distribution.png    # 主图：per-subject feature-winner effect
│   ├── q1_stratified_swap_silhouette.png  # Q3 2×2 stratifier (descriptive)
│   ├── q1b_442_sentinel.png          # 442 sz=9 vs 其他 fingerprint
│   └── per_subject/<sid>_fingerprint.png  # cohort 10 + 442 + 1084 strip plot
```

**Q2 outputs 不在本目录第一轮交付**（`q2_pattern_match.json` 等仅在 follow-up 实施时新增）。

## 6. Caveats & explicit exclusions

### Caveats

1. **Cohort N=10 是探索性极限**：≥3/10 是 binomial p≈0.012；4/10 是 p≈0.0010。再低 cohort 不写。
2. **Topic 1 cohort tier mixed**：16 epilepsiae 中部分属于 Tier 0（vintage），部分 Slice A1/A2。Tier 分类在 `cohort_audit.csv` 单列；mixed tier 在 spec §6 显式注明，paper framing 必须 "Tier 0/1 mixed" qualifier。
3. **Topic 5 audit-rerun 已完成（2026-05-10 16:21）**：cohort summary post-audit 与 pre-audit 在 status / n_subtypes / n_seizures 等 cohort gating 字段上 byte-identical（median Δgap_perm −0.0007, 0 over_split flips）。Spec §3.1 cohort N=10 在 audit-rerun 后保持。**Sentinel soft failures**：`548/gamma_ER` minority_jaccard=0.25、`548/broad_ER`=0.43、`916/gamma_ER`=0.00（详见 `_GATE_FAILED.txt`）。548 仍入 cohort（subtype 数目不变，仅 user-marked outlier set 在 cluster 内 recall 不完美）；**§4.5 中已去掉 "548 应 top-3 ε²" 数值 gate**——548 仅作 high-information descriptive case 报告，不参 cohort hard verdict。916 不是 spec sentinel，不受影响。
4. **Imputation_warning 不污染 Q1**：`cluster_geometry.py:319` 已确认 imputed_fraction 是 MDS pairwise-distance imputation（trilateration plane viz 用），不是 per-event template 伪造。Q1 消费的 `adaptive_cluster.assignments` 是 KMeans 真标签。但 5 个 imputation_warning subject (1073, 253, 916 在 cohort 内) 的 §3.1d 几何叙述需 caveat。
5. **Block-resolution proxy 不可用**：spec 明令 per-event timestamps 必须从 `load_subject_propagation_events` 拿，不准用 JSON `block_boundaries` proxy。
6. **Pre-ictal event count 跨 subject 17→1309 倍差**：通过 dual gate (ε² + p) 解决，不通过 subsampling（subsampling 922 到 17 events 是 99% signal 浪费）。
7. **442 不是 negative control**：是 binary-outlier sentinel；阴性 = indeterminate，不构成"方法学验证"。

### 显式 NOT-DO

- 不重新跑 PR-2 cluster pipeline；直接消费 `adaptive_cluster.assignments`
- 不做跨 subject template 对齐
- 不试图把 Q1 升级为 cohort α=0.05 paper-level claim
- 不在本 spec 内做 yuquan v2.3 atlas 建设（前置依赖，独立 PR）
- 不在本 spec 内做 topic5 PR-1 schema patch（Q2 unblock 独立 PR）
- 不做 "predicts subtype" / "causes subtype" framing；仅 association
- 不引入 ML model（random forest / SVM 之类）；仅非参检验

## 7. 工作量估算

| 任务 | 估时 |
|---|---|
| Step 1 数据合同测试（subtype label / event-level timestamp / T0-T1 freeze idempotent） | 0.25 d |
| Step 2 Q1 fingerprint extraction + 2-3 subject sanity | 0.5 d |
| Step 3 per-subject statistics（dual gate same-feature AND）| 0.25 d |
| Step 4 Q1 cohort N=10 gamma + per-window denominator | 0.25 d |
| Step 5 sentinels (442 / 548 case) + Q3 stratifier | 0.5 d |
| Step 6 minimum 5 figures + figures/README.md | 0.5 d |
| Step 7 archive doc + bridge §2 主结论回写 | 0.5 d |
| **小计** | **~2.75 d 实施工作** |

Q2 deferred（§2.3 / §8 项目 1）。Audit-rerun 已完成，无需等待。

## 8. Open follow-ups（不在本 spec）

| 项 | 描述 | 触发条件 |
|---|---|---|
| 1 | Topic 5 PR-1 binned JSON schema patch 暴露 `channels_used` | Q1 EXPLORATORY-PASS 后才值得做 Q2 |
| 2 | Yuquan v2.3 atlas 建设 → 扩 Q1 cohort | 独立 topic5 PR-0 follow-up |
| 3 | broad-band sensitivity（broad γ≥2 cohort，独立 cohort summary） | 与 gamma 主结果同期 |
| 4 | 子型 outcome × pre-ictal fingerprint 的 mediator analysis | 仅在 outcome 数据可用且 Q1 PASS 后 |
| 5 | Pre-ictal template fingerprint 与 PR-5 dominant-template recruitment 的关系 | Topic 1 内部 follow-up，不绑 Topic 5 |

## 9. 来源文档

- `docs/topic1_within_event_dynamics.md` §3.1 / §3.1b / §3.1d / §7.10 / §8 (rank displacement)
- `docs/topic5_seizure_subtyping.md` §2 / §3.1 / §3.2 / §6
- `docs/archive/topic1/propagation/interictal_group_event_internal_propagation.md`
- `docs/archive/topic1/pr6_supplementary_rank_displacement_results_2026-05-06.md` §8
- `docs/archive/topic5/pr1_seizure_clustering/pr1_zer_cohort_2026-05-10.md`
- `results/topic1_topic5_bridge_audit.csv` (本 spec audit)
- `results/run_logs/cohort_zer_audit_20260510_1045.log` (audit-rerun completed 2026-05-10 16:21)

---

**Spec self-review checklist** (执行于 spec 写完后):
- [ ] 所有 placeholder / TBD / TODO 移除
- [ ] §2 hypothesis 与 §4 statistical contract 互不矛盾
- [ ] §3 cohort N 与 §4 alpha threshold 数学上自洽
- [ ] §5 复用接口路径全部存在（grep verify）
- [ ] §6 caveats 覆盖 advisor 三个 blocking 与所有非 blocking
- [ ] 仅 association，无 causal/predicts framing
