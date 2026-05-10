# Topic 5 PR-2+ Sub-type Seizure Onset Feature Menu (2026-05-10)

> **形态**：候选菜单，**不是 PR spec**。下游 PR 从此菜单选 ≈ 5–10 个 feature 单独立 spec。
> **范围**：(A) ictal onset 形态 + (B) pre-ictal interictal 状态。
> **目的**：(I) per-subtype 独立指纹描述 + (II) discriminator effect-size ranking。
> **Cohort**：基线 = topic5 PR-1 cohort 16 epilepsiae（28 ok bands 中 20 ≥2 subtypes）。**禁止 outcome 耦合**（不接 Engel 等临床变量；那是更下游 PR-3+）。
> **Bridge 关系**：B1a/B1b/B1c (frac_T0 / switch_rate / last_template) 是 bridge spec primary，菜单仅引用，不重定义。其他 9 个 feature 均为菜单新增。

## 1. 菜单选 feature 的硬约束

1. **Non-circular**：feature 不能直接来自 topic5 z-ER tensor 5-bin × channel 本身（subtype 由它定义）。允许从 atlas_v2_3 的 channel_onsets / per-channel z-ER trajectory 派生但与 5-bin 不同的物理量。
2. **Subject-internal aggregation**：每 feature 在 per-subject 层面独立计算，不跨 subject 平均。Cohort 是 per-subject effect size 的分布。
3. **Pre-ictal scope**：(B) 类 feature 默认窗口 `[-30, -1] min`，可 sensitivity 到 `[-15, -1]` / `[-60, -1]`。
4. **Subtype size ≥ 3 才参 KW**：cohort 中很多 subtype size = 2，处理规则 = `pool_with_annotation` 或 drop（菜单不绑定，落 PR 时决定）。
5. **No supervised model in this menu**：discriminator ranking 仅用非参检验 effect size，不进 random forest / SVM。
6. **Two separate rankings, NOT one** (locked)：ictal morphology features (A 类) 来源接近 subtype 定义本身（atlas_v2_3 的同套 z-ER trajectory），排高是预期不是外部验证；pre-ictal features (B 类) 才是 bridge 的独立证据。**A 类与 B 类不能合并到同一张 ranking 表**——这会把 sanity check 与外部验证视觉混淆。详见 §4 / §5。

---

## 2. (A) Ictal onset 形态 candidate

### Tier 1（高 ROI，复用 atlas_v2_3，~1 d/feature）

| # | feature | 简要定义 | 数据来源 | prior 期望 | 与 subtype 区分度 |
|---|---|---|---|---|---|
| **A1** | `t_soz_first` | 在 [-120, +30]s 内，clinical SOZ 通道首个 z-ER > 1 的最早时刻；subject 内 SOZ list 取 min；无则 NaN | `atlas_v2_3/per_subject/<sid>/<sid>_<sz>_layer_a_v2_3_timing.json` + `results/epilepsiae_soz_core_channels.json` | SOZ-first vs non-SOZ-first 双峰；subtype 可能在此 dichotomy 上区分 | **强**（已有文献支持） |
| **A2** | `onset_spread_duration` | 从最早 t_onset 到 50% 通道 onset 的时间差（秒） | atlas_v2_3 `channel_onsets[ch].t_onset_sec` | "diffuse vs focal" 对应 spread 长 vs 短 | **中** |
| **A3** | `peak_zer_amplitude` | max(z-ER) across (channel × frame) within [0, +30]s（band-specific） | atlas_v2_3 z-ER tensor | subtype 可能在 "high-amplitude vs low-amplitude" 上分；circular risk 低（amplitude 不是 5-bin temporal 维度） | 中 |
| **A4** | `soz_recruitment_frac_5s` | clinical SOZ ch 中在 [0, +5]s 内 z-ER > 1 的比例 | A1 + soz_core_channels | 早期 SOZ 招募比例 直接反映 onset 几何；与 t_soz_first 互补 | **强** |

### Tier 2（中 ROI，需补 spectral / metadata，~2 d/feature）

| # | feature | 简要定义 | 数据来源 | prior |
|---|---|---|---|---|
| **A5** | `gamma_broad_ER_ratio` | peak gamma_ER / peak broad_ER (per seizure) | atlas_v2_3 双 band | subtype 可能反映 "纯 gamma" vs "broad-coupled"；中度 prior |
| **A6** | `frequency_drift` | spectrogram peak freq 在 [0, +30]s 的线性 slope (Hz/s) | 重新算 spectrogram（atlas 不直接有） | DC drift 是 ictal 经典 feature；subtype 区分中度 prior |
| **A7** | `lateralization_index` | (right_onset_count − left_onset_count) / total（基于电极 metadata） | A1 + 电极 LR 标签 | per subject 通常单侧，cross-subject 才有意义；用作 cohort tier 校验，不主推 |

### Tier 3（低 priority，high cost，留 future）

- Peak frequency at t_onset
- Rise rate from baseline (slope of z-ER from -10s to onset)
- Polyspike presence (binary, requires raw EDF)

成本理由：返回 raw EDF / SQL，每 seizure 需重算 spectrogram；与 z-ER tensor 高相关，边际信息有限。

---

## 3. (B) Pre-ictal interictal 状态 candidate

### Tier 1（bridge primary，菜单仅引用）

| # | feature | bridge 关系 |
|---|---|---|
| **B1a** | `frac_T0`（T0/T1 fraction） | bridge primary — 见 `2026-05-10-topic1-topic5-bridge-design.md` §2.1 |
| **B1b** | `switch_rate` | bridge primary |
| **B1c** | `last_template` | bridge primary |

### Tier 2（菜单新增，~1 d/feature）

| # | feature | 简要定义 | 数据来源 | prior |
|---|---|---|---|---|
| **B2** | `event_rate_pre_ictal` | events / min in [-30, -1] min | `load_subject_propagation_events` event_abs_times | PR-2.7 已示 seizure-triggered rate 抬升；pre-ictal rate 可能 subtype-specific |
| **B3** | `event_rate_per_template` | T0 rate (ev/min), T1 rate (ev/min) 单独 | B2 + adaptive_cluster.assignments | bridge 看 fraction，B3 看绝对率；互补 |
| **B4** | `cluster_stereotypy_pre_ictal` | within-cluster Kendall τ on pre-ictal subset | `compute_within_cluster_centered_tau` (existing) | 高 stereotypy = 模板"咬定"；subtype 可能在此区分 |
| **B5** | `n_participating_mean_pre_ictal` | mean events 参与通道数 across pre-ictal events | `event_metadata` | 大 event size 与 subtype 关联曾被旧文献提及 |
| **B6** | `lag_k_serial_corr_pre_ictal` | lag-1 IEI auto-correlation 在 pre-ictal 子集 | PR-2 exp7 helper | PR-2 全 cohort 30/30 positive；subtype-conditional 是否有差异未知 |

### Tier 3（high cost，留 future）

- `sync_phase_global_pre_ictal` / `sync_phase_e_pre_ictal`（PR-4–6 phase metrics，需要重算 phase 估计）
- `rate_state_coupling_pre_ictal`（PR-4B framework）
- `rate_trace_psd_beta_pre_ictal`（PR-2.7 spectral）

成本理由：每 feature 都依赖独立的 PR-4* / PR-2.7 helper，pre-ictal 子集重算需要把现有 cohort summary 反向切到 per-window，工作量 ≈ 0.5 d/feature 但与现有 PR 已算的 cohort summary 不复用。

---

## 4. 统一统计 recipe（两张 ranking 共用）

每 feature × subject：

```
per-seizure feature value f_k (k=1..K_s)
ictal subtype label y_k ∈ {0, ..., k_s - 1}  from topic5 PR-1

if k_s == 2:
    test = Mann-Whitney U
    effect = rank-biserial r
elif k_s >= 3:
    test = Kruskal-Wallis
    effect = ε² (epsilon-squared)
else:
    skip (k_s = 1, no discrimination possible)

per-subject Q-positive: ∃ same feature: p < 0.05 AND |effect| > 0.10  (same-feature AND, 与 bridge 一致)
```

每 feature 跨 cohort（ranking 行单元）：

| col | 计算 |
|---|---|
| feature_name | A1 / A2 / ... 或 B1a / B2 / ... |
| feature_class | `ictal_morphology` (A) 或 `pre_ictal_state` (B) |
| n_subjects_eligible | 至少 2 个 subtype 且 subtype size ≥3 的 subject 数 |
| n_subjects_q_positive | 通过 dual gate 的 subject 数 |
| cohort_effect_median | per-subject \|effect\| 的 median |
| cohort_effect_iqr | 同上 IQR |
| per_subtype_summary | 嵌套 dict: subtype 0 → median/IQR, subtype 1 → median/IQR, ...（subject pool；用于 (I) 描述层） |

## 5. **Two separate rankings (locked)**

### 5.1 Ranking #1：Ictal morphology sanity (A 类)

| 范围 | A1 / A2 / A3 / A4 / A5 / A6 / A7 |
|---|---|
| 解读层级 | **Sanity check**——这些 feature 来源接近 z-ER atlas 本身，排高是预期。Top-3 = "subtype 在 ictal 形态层的最直接表型"，**不是**bridge 的外部验证 |
| 用途 | (a) 确认 subtype label 在 atlas 形态上自洽；(b) 描述每 subtype 的 ictal 形态指纹；(c) 与 (B) ranking 对比看 ictal vs pre-ictal 的相对 effect 量级 |
| Paper-level 表述 | "Subtypes differ in ictal onset morphology (A4 ε²=…, A1 ε²=…), as expected from the clustering input" |
| **禁止 framing** | ❌ "ictal morphology features confirm bridge result"；❌ "validation of pre-ictal predictor" |

### 5.2 Ranking #2：Pre-ictal discriminator (B 类)

| 范围 | B1a / B1b / B1c / B2 / B3 / B4 / B5 / B6 |
|---|---|
| 解读层级 | **External evidence**——pre-ictal 状态特征独立于 ictal 形态聚类输入，排高是 bridge 假设的**支持证据** |
| 用途 | (a) 检验 bridge 3 features 的 effect 量级是否处在 cohort 上限；(b) 找有没有更强的 pre-ictal discriminator；(c) per-subtype B 类指纹独立描述 |
| Paper-level 表述 | "Among pre-ictal interictal-state features, B2 (event rate) and B1a (template T0 fraction) discriminate subtypes (cohort effect median ε²=…); other pre-ictal features (B5, B6) do not" |
| **关键判读** | bridge 3 features (B1a/B1b/B1c) 在 ranking #2 中靠前 → bridge 主张被 pre-ictal cohort 内自洽；靠后 → bridge 选 feature 不是最强，需要回炉 |

### 5.3 两张 ranking 的禁止合并理由

合并到一张 ranking 会出现："A4 (SOZ recruitment) ε²=0.4, B2 (event rate) ε²=0.15, B1a (frac_T0) ε²=0.10" 这种排序，让读者误以为 A4 是"最强 discriminator"。但 A4 的高 effect 是**循环结论**（z-ER atlas 给 subtype，再用 atlas 派生 feature 验证 subtype），与 B 类的"bridge 假设是否成立"是**两个量级不同的 epistemic claim**。两张表分开，paper-level framing 上各自封闭。

---

## 6. 描述层与 ranking 层的输出形态

每张 ranking（§5.1 和 §5.2）独立产出两份表：

**(I) Per-subtype 描述表**：一行 / (subject, subtype) 二元组；列 = 该 ranking 范围内的 feature × {median, IQR}。**Pool 跨 subject 是合法的**因为这层不做主张，只是描述每 subtype 在每 feature 上的分布。subtype 编号 (0, 1, ...) **不**跨 subject 对齐——subtype 0 只是 subject 内 size 降序排第一名，跨 subject "subtype 0 vs subtype 1" 列不构成统一含义。

**(II) Discriminator ranking 表**：一行 / feature；列 = `n_q_positive`、`cohort_effect_median`、`cohort_effect_iqr`、`bridge_owner` (Y/N for B 类，全 N for A 类)。按 `cohort_effect_median` 降序排。

两层独立产出，不互相替代。Paper-level framing 只能引用 (II) ranking + per-subject case series；(I) 描述层是 supplement。

每个 PR-2+ 子 PR 对自己的 ranking #1 或 #2（或两者）出独立 figure，不允许把两 ranking 合并到同一图。

---

## 7. 菜单 → PR 触发条件

下游 PR-2 应在以下任一触发后启动：

1. **Bridge spec 完成**（无论 PASS / NULL / INDETERMINATE）：菜单 9 个新 feature 与 bridge 3 features 同台 ranking，看是否有更强 discriminator
2. **Yuquan v2.3 atlas 建设完成**（topic5 PR-0 follow-up）：cohort 扩到 16 + 9 = 25，菜单 ranking 在更大 cohort 上有意义
3. **Topic5 PR-1 cohort sensitivity 完成**：`min_subtype_size=3` / `intersection-only mask` / bootstrap stability 三件 sensitivity 全跑过，subtype label 进入 publication-grade 状态

任何**单一**触发条件即可启动。下游 PR 选 feature 时建议 Tier 1 全选 (A1–A4 + B2–B6 = 9 个) 优先实施，Tier 2 在 Tier 1 ranking 完成后再追加。

---

## 8. 显式不做

- 不在本菜单内做 outcome (Engel) / 临床变量耦合 — 那是 topic5 PR-3+ 范围
- 不引入 supervised ML 模型 — 仅非参检验
- 不重新跑 topic5 PR-1 cluster pipeline — 直接消费 audit-rerun 的 subtype_label
- 不跨 subject 对齐 subtype 编号 — subtype 编号仅 subject 内 size-rank 含义
- 不假设 yuquan 必须先建 — 16 epilepsiae 已可启动 ranking
- 不绑死 Tier 1 必须全实施 — 下游 PR 可选 5-10 个 feature 即可

## 9. 来源 / 关联文档

- `docs/topic5_seizure_subtyping.md` §2 / §3 / §6
- `docs/superpowers/specs/2026-05-10-topic1-topic5-bridge-design.md`（B1a/b/c owner）
- `docs/topic1_within_event_dynamics.md` §6 (PR-4 三类 readout 框架，B-tier feature 灵感)
- `docs/topic2_between_event_dynamics.md`（B6 / B-tier 3 PR-2.7 spectral 入口）
- `docs/archive/topic5/pr1_seizure_clustering/pr1_zer_cohort_2026-05-10.md`
- `results/data_driven_soz/layer_a_ictal_er_rank/atlas_v2_3/`（atlas_v2_3 数据契约）
- `results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/per_subject/<sid>__zer_binned.json`（subtype label 输入）

## 10. Open questions for menu→PR

- Subtype size < 3 的处理是 pool / drop / per-subtype CI？（菜单留开）
- A6 frequency_drift 是否需要 EDF raw recompute 还是 atlas 内已有 spectrogram? （需要 inspect atlas v2.3 schema 才能确定）
- A7 lateralization 是否值得 ~1 d 实施成本？（per-subject 通常单侧，cohort tier 校验成 binary 用法可能价值较低）
- B6 lag_k_serial_corr 的 pre-ictal 子集计算与 PR-2 exp7 helper 接口是否兼容？需 `compute_lag_k_serial_correlation` 是否支持 t-mask 参数？
