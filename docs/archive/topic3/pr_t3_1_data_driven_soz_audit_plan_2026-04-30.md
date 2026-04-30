# PR-T3-1 计划：Data-driven Ictal-Onset SOZ Audit

> 状态：plan-of-record，2026-04-30（**v1.1**，整合 2026-04-30 review 8 条修正：size-matched k / 取消三档 verdict / M1 rate-normalized / M2 log-ratio + Nyquist guard / 修正 null abort / canonical channel matcher / 修正数据路径 / "两条 proxy 而非独立通道" framing）
> 范围：在 Yuquan + Epilepsiae cohort 上，用 HFO-onset rate (M1) 与 ER-ratio (M2) 两条**频段 / 时间对齐高度相关**的数据驱动 proxy，从 SEEG 信号本身派生 ictal-onset SOZ 通道集合，并与现有临床 SOZ 标注做 overlap audit。**v1.1 不出 EI**（推迟 PR-T3-2）；**不替换** `*_soz_core_channels.json`；**不出三档 qualitative verdict**，只出预注册的 cohort 数值表 + null-corrected enrichment。本 PR 的目标**不是**判定 clinical SOZ 谁对谁错，而是回答下游 SOZ-dependent 分析（PR-8 v2 / topic2 SOZ-stratified）是否必须改成 multi-source SOZ 报告。
> 上游：`docs/topic3_spatial_soz_modulation.md`；`docs/archive/topic3/spatial_modulation_soz_analysis.md`（PR-1 spatial_modulation Yuquan 9/11 valid pairs；§"双极通道的 SOZ 匹配（关键）：必须使用 alias_bipolar_to_any 逻辑"，本 PR 必须遵循）；`results/{epilepsiae,yuquan}_soz_core_channels.json`；`results/epilepsiae_seizure_inventory.csv`；`results/hfo_detection/<subject>/*_gpu.npz`（new pipeline 输出）。
> 下游：本 PR 输出 `results/spatial_modulation/data_driven_soz/<dataset>_<subject>.json`，被 topic1 PR-8 v2（**deferred pending PR-T3-1**）与 topic2 SOZ-stratified PR 共同消费。

---

## 1. Context — 为什么这是 PR-8 的先决条件

PR-8 plan v1 已 deferred（`docs/archive/topic1/pr8_intra_event_spatial_polarity_plan_2026-04-30.md` §15）。其核心因变量"SOZ-first / SOZ-last"完全依赖 SOZ 标签的质量；clinical 'i' / 'l' / 'e' 在当前 cohort 上的可靠性是**未量化的**前置变量。

**重要的 framing 校正（本 v1.1 比 v1 收紧）**：

- 本 PR **不**论断 clinical SOZ 是错的，也**不**论断数据驱动 SOZ 是真实 epileptogenic zone
- M1 与 M2 都来自同一频段（80–250 Hz）+ 同一 seizure-onset 对齐，**不是**两条独立证据，而是**两条频段 / 时间锚点高度相关的 proxy**。任何 framing 都不得说"two independent channels"
- 低 overlap 的可能解释包括：(a) clinical SOZ 不准；(b) M1 / M2 proxy 太粗；(c) ictal HFO onset 晚于真实电生理起始（Bartolomei detection time 概念）；(d) HFO event-rate confound（参见 topic3 PR-2 已确认的 SOZ × event-rate 相关）；(e) montage 差异（CAR vs bipolar）；(f) 多灶 / 跨 seizure 变异。本 PR**不**单选其中一种作为解释
- 本 PR 唯一能强主张的事情：**下游 SOZ-dependent 分析（PR-8 v2 / topic2 SOZ-stratified）是否必须改成 multi-source SOZ 报告**

---

## 2. 编号与归档决定

- **新 PR**：PR-T3-1 = **Data-driven Ictal-Onset SOZ Audit**
- **不打包**（推到 PR-T3-2）：EI 完整实现 / threshold 自动优化 / 30–100 Hz 与 12–127 Hz band sensitivity / clinical 'i'/'l'/'e' 三层比较 / clinical 'l' 单独审计 / CAR vs bipolar montage sensitivity
- **范围声明（写死）**：本 PR 只输出 audit 数值表 + reusable JSON。**不出 qualitative verdict 标签**（broadly / partially / unreliable 这种三档定性已删除）；下游 PR 的 SOZ 定义策略由人类基于本 PR 的数值表决定，不由本 PR 自动判读
- **主文档回写**：完成后在 `docs/topic3_spatial_soz_modulation.md` 加 PR-T3-1 数值摘要 + archive 链接；不替换或下线已有 SOZ JSON

---

## 3. 假设与统计合同（核心）

### 3.1 数据对象（数据路径已 verify 2026-04-30）

每个 subject 的输入：

- `seizure_onsets: list[float]` — 每次 seizure 的绝对 onset 时间（sec since epoch）
  - Yuquan：来自 `src/preprocessing.py::detect_seizure_by_spatial_extent` 输出，与 `results/seizure_detection/yuquan/<subject>.json` 对齐
  - Epilepsiae：`results/epilepsiae_seizure_inventory.csv`（绝对 ts = `block.start_ts + onset_seconds`；Step 0 audit 必须验证字段名）
- `hfo_event_times_per_channel: dict[str, np.ndarray]` — 每通道 HFO 检测时间戳（绝对秒）
  - 来源：`results/hfo_detection/<subject>/*_gpu.npz`（new pipeline 输出；不再用 `/mnt/epilepsia_data/.../all_recs/`）
  - 多 block 的 npz 跨 block 拼接为单一 dict
  - 文件大小验证：每个 npz > 1KB（Epilepsiae 历史 stub 是 216B，必须先确认重检测完成）
- `signal_loader: callable(t_start, t_end, channels) -> (signal[T, C], sfreq)` — 用于 ER-ratio 的原始信号载入器
  - Yuquan：包装 `src/preprocessing.py` 现有 EDF 路径
  - Epilepsiae：包装 `src.preprocessing.load_epilepsiae_block`（CAR 默认）
- `clinical_soz: set[str]` — 临床 SOZ 通道（**raw 单极**或 dataset native 命名）
  - Epilepsiae：`results/epilepsiae_soz_core_channels.json`（focus_rel='i' only，不合并 l/e）
  - Yuquan：`results/yuquan_soz_core_channels.json`
- `analysis_channel_set: list[str]` — HFO npz / signal loader 共享的通道顺序

### 3.2 Channel matching 合同（v1.1 重写 — 软匹配 clinical，硬对齐 HFO/signal）

- **HFO npz channel_names ↔ signal loader channel_set**：**严格对齐**（顺序与名字一致），不一致 raise ValueError；不允许 default-by-index alignment
- **Clinical SOZ → analysis_channel_set 的标注**：用 canonical matcher，**逐通道**标注 `SOZ / nonSOZ / unknown` 三态：
  - 调用 `src/event_periodicity.py::match_bipolar_soz` 或等价逻辑：对双极通道 `X-Y`，X ∈ clinical_soz 或 Y ∈ clinical_soz → SOZ；都不在 → nonSOZ；任一端为名字残缺 / 空 → unknown
  - **禁止**用 `alias_bipolar_to_left` 作为最终 matcher（参见 `docs/archive/topic3/spatial_modulation_soz_analysis.md` §"双极通道的 SOZ 匹配（关键）"）
  - **禁止**做 set-equality 严格对齐（即"clinical_soz set == analysis_channel_set 子集"），因 clinical SOZ 通常是单极而 analysis_channel_set 是双极 / CAR
- **Audit 透明性要求**：每 subject 报告 `n_clinical_soz_total`（原始 clinical SOZ 通道数）、`n_clinical_matched`（成功标注为 SOZ 的 analysis channels 数）、`n_clinical_unmatched`（找不到匹配的原始 clinical SOZ）。`unmatched_count > 0` 不 raise，但写入 audit.csv 供人眼复核

### 3.3 M1 — HFO-onset rate（v1.1：加 baseline rate + Poisson z-score sensitivity）

每次 seizure `s` 的 onset `t_s`，对每个通道 `ch`：

```
window_pre  = [t_s − W_pre,  t_s)             # W_pre  = 30s baseline
window_post = (t_s,         t_s + W_post]     # W_post = 10s primary

n_pre[s, ch]  = #HFO events of ch in window_pre
n_post[s, ch] = #HFO events of ch in window_post

rate_pre[s, ch]  = n_pre[s, ch]  / W_pre
rate_post[s, ch] = n_post[s, ch] / W_post
```

**Per-seizure 指标三种并列**（M1 内部三个 variant，不是三个 method）：

```
M1_raw[s, ch]    = rate_post[s, ch] − rate_pre[s, ch]                 # primary
M1_log[s, ch]    = log(n_post[s, ch] + 1) − log(n_pre[s, ch] + 1)
                   − log(W_post / W_pre)                              # rate-normalized
M1_pois[s, ch]   = (n_post[s, ch] − μ_pre[s, ch]) / sqrt(μ_pre[s, ch] + 1)
                   where μ_pre[s, ch] = rate_pre[s, ch] × W_post      # Poisson z
```

`M1_log` 处理低 baseline 通道（避免 0 → 0 减法导致排序不稳）；`M1_pois` 把 enrichment 放到泊松 expected variance 上做归一化。

**主报告**：`M1_pois`（rate-normalized），其他两个 variant 作 sensitivity。`M1_raw` 是 v1.0 主指标；v1.1 降为 sensitivity，因为 topic3 PR-2 已确认 SOZ × event-rate confound（高 baseline rate 通道天然 dominate raw 减法）。

Per-subject JSON 必须报告 `baseline_rate_per_channel` 字典（每通道平均 `rate_pre` 跨 seizure），下游可手工诊断 confound。

Per-seizure ranking + 跨 seizure 聚合：见 §3.5 / §3.6。

### 3.4 M2 — ER-ratio（v1.1：log-ratio + Nyquist guard + 噪声底 eps）

**Nyquist guard（前置硬门禁）**：

```
band = (80.0, 250.0) Hz                       # primary
if sfreq / 2 < band[1] * 1.05:                # 5% safety margin
    raise NyquistGuardError(f"sfreq={sfreq} too low for band {band}; need sfreq >= {band[1]*2.1}")
```

实践含义：`sfreq < 525 Hz` 的 subject / block 在 M2 主指标下被 drop。Epilepsiae 部分 subject 是 256 Hz / 1024 Hz，需 Step 0 audit 报告 sfreq 分布；256 Hz subject 在 M2 主指标下 ineligible。Yuquan 通常 1024 Hz，全 cohort 通过。

**Filter 合同**：Butter order 4（IIR），`scipy.signal.filtfilt` 零相位；`padlen = max(default, int(1.5 * sfreq / band[0]))` 保证最低 cutoff 有足够 padding；padlen 不足时 raise FilterPaddingError。

**Per-channel 噪声底 eps**：不写死 1e-12。对每个通道 `ch`，eps 取该 subject 跨所有 seizure 的 `power_pre[s, ch]` 的 1st percentile（最低噪声底估计）；如果某通道全 cohort 都 0，drop 该通道（写入 ineligible_channels）。

```
edge_buffer = 2.0 s                           # 避开 onset 时刻附近 filter ringing
W_pre        = 30 s
W_post       = 10 s

bandpass    = filtfilt(butter(4, band, fs=sfreq), signal, padlen=...)
power_pre[s, ch]  = mean( bandpass[t_s − W_pre  − edge : t_s − edge, ch]^2 )
power_post[s, ch] = mean( bandpass[t_s + edge   : t_s + edge + W_post, ch]^2 )

eps_ch       = max(percentile_1st(power_pre[:, ch] across s), 1e-18)
M2_logratio[s, ch] = log(power_post[s, ch] + eps_ch) − log(power_pre[s, ch] + eps_ch)
```

**Primary M2 指标**：`M2_logratio`。v1.0 的 `er_ratio = post / max(pre, eps)` 已弃用——低 pre-power 通道天然 explosion 是 v1.0 的科学漏洞。

Per-seizure ranking + 跨 seizure 聚合：见 §3.5 / §3.6。

### 3.5 跨 seizure 聚合（v1.1：预注册 primary，不做 winner's curse）

对每个 method（M1_pois / M2_logratio）：

- **Consensus rule**：channel 必须在 ≥ 50% seizures 的 top-k 中出现 → `consensus_topk[ch]`
- **Median-rank rule**：channel 跨 seizure 的 rank 中位数（NaN 占满 rank 的最大值），取 top-k → `medianrank_topk[ch]`

**预注册 primary aggregation = `medianrank`**。理由：consensus 在 seizure 数 < 4 时阈值粒度过粗（< 4 时 50% 规则等价 ≥ 2 个 seizure 即入选）；medianrank 在 sparse seizure 下更稳。Consensus 仅作 sensitivity，不进 cohort headline。

### 3.6 Top-k：primary = size-matched k = |clinical_matched|

**v1.0 漏洞**：固定 k = 3 / 5 / 10 与 clinical SOZ 集合大小不匹配（cohort 内 clinical SOZ 通道数从 1 到 14+ 不等），低 Jaccard 可能纯粹是集合大小算术，不是 SOZ 标签问题。

**v1.1 修法**：

- **Primary k**：`k = |clinical_matched|`（per subject，size-matched）。这是 cohort 比较的主轴
- **Sensitivity k**：3 / 5 / 10（固定档位）+ |clinical_matched| ± 2（局部敏感性）
- 每个 (method × aggregation × k) 组合都计算并写入 per-subject JSON，但 **headline 只看 size-matched primary**

### 3.7 Overlap audit + null-corrected enrichment（v1.1：核心）

对每个 (method × aggregation × k)：

```
A = clinical_matched                          # set, per §3.2
B = method_aggregation_topk                   # set, |B| = k

jaccard(A, B)   = |A ∩ B| / |A ∪ B|
precision(A, B) = |A ∩ B| / |B|
recall(A, B)    = |A ∩ B| / |A|
f1(A, B)        = 2 P R / (P + R)
```

**Random-expected baseline（v1.1 新增）**：

```
n_total = |analysis_channel_set|
random_expected_intersection(|A|, |B|, n_total) = |A| * |B| / n_total      # 期望命中数
random_expected_jaccard(|A|, |B|, n_total)      = computed analytically OR by 1000-iter MC

enrichment[method × agg × k] = observed_intersection / random_expected_intersection
  # > 1 表示 overlap 高于随机水平；= 1 表示 noise；> 2 表示 substantial enrichment
```

**Cohort 报告主轴（per subject 三个 headline 数值）**：

```
H_M1 = enrichment(M1_pois, medianrank, k=|clinical_matched|)
H_M2 = enrichment(M2_logratio, medianrank, k=|clinical_matched|)
H_concord = enrichment( M1_pois ∩ M2_logratio at k_concord, clinical )
            其中 k_concord = |M1_topk ∩ M2_topk|（concordance set；可能小于 k）
```

cohort_overlap_summary 报告 `H_M1`、`H_M2`、`H_concord` 三列的 cohort 中位数 + IQR + 直方图，**不做 cohort scalar p-value**（这是 audit，不是假设检验）。

### 3.8 Per-seizure consistency（同 v1.0）

对每 (subject, method, k=|clinical_matched|)：

```
consistency = median over (s_i, s_j) pairs of jaccard( topk(s_i), topk(s_j) )
```

低 consistency（< 0.3）说明该 subject 即使在数据驱动定义下"哪些通道是 SOZ"跨 seizure 也不稳定。这种 subject 进 audit 但**不**进下游 PR-8 v2 / topic2 SOZ-stratified 的 default cohort。

### 3.9 NO three-tier verdict（v1.1：删除 v1.0 §3.8 的定性判读）

v1.0 的"broadly consistent / partially consistent / unreliable"三档 verdict 已删除。理由：

- 这是 audit，不是假设检验，不应有自动 verdict
- v1.0 的"按 best method 较好的一档定 verdict"是 winner's curse
- 多种因素都能拉低 overlap（§1 列了 6 条），单看 overlap 数值不能判读 clinical SOZ 谁对谁错

**v1.1 报告形式**：

- cohort_overlap_summary 直接写 `H_M1 / H_M2 / H_concord` 中位数 / IQR / 分布直方图 + per-subject 表
- results doc §结论部分只描述数值，**不**贴标签
- PR-8 v2 / topic2 SOZ-stratified PR 的设计决策由人类阅读 cohort 数值后做出，**不**由本 PR 自动驱动

### 3.10 本 PR 能 / 不能说明什么（v1.1 显式写入）

| 能说明 | 不能说明 |
|---|---|
| 下游 PR-8 v2 / topic2 SOZ-stratified 是否必须改 multi-source SOZ 报告（如果 cohort 中位数 enrichment 低 / 跨 method 高分散，则必须） | clinical SOZ 是错的 / 数据驱动 SOZ 是真的 |
| Clinical SOZ 在当前 cohort 上的 multi-method overlap 数值范围 | 哪些 subject 的 clinical SOZ 不可靠（需要外部金标准对比，本 PR 没有） |
| HFO event-rate confound 与 SOZ 标注的相关程度（通过 baseline_rate × overlap 散点） | 真实 epileptogenic zone 在哪里 |
| 跨 seizure SOZ 候选稳定性（per-seizure consistency）| 多灶 epilepsy vs 跨 seizure 漂移谁主导 low consistency |

---

## 4. 不做的部分（v1.1 out of scope）

- **EI（Epileptogenicity Index）完整实现** → PR-T3-2
- **HF band sensitivity beyond 80–250 Hz**（30–100 Hz / Bartolomei 12–127 Hz） → PR-T3-2
- **Threshold 自动优化（ROC / threshold sweep）** → PR-T3-2
- **i / l / e 三层比较** → PR-T3-2
- **CAR vs bipolar montage sensitivity** → PR-T3-2
- **替换 `*_soz_core_channels.json`** → 永远不做
- **Qualitative SOZ-reliability verdict（"clinical SOZ 不可靠"等定性结论）** → 不在本 PR / PR-T3-2 / 任何后续 PR 做；本 repo 永远不做这种判读
- **机制层论断** → HFO 频段不可分 E/I，不在本 repo 做
- **CAR / bipolar montage sensitivity** → PR-T3-2
- **PR-8 v2 / topic2 PR 重启** → 等本 PR 完成后再启动

---

## 5. Surrogate / null（v1.1：替换错误的 abort）

### 5.1 Time-shifted null

每个 seizure，把 `t_s` 随机平移到该 block 内非 ictal 区间（避开真 seizure ± 5 min）；重算 M1_pois 与 M2_logratio 排名 + 聚合 → 记录 shifted 版本的 `H_M1_shifted`、`H_M2_shifted`、`H_concord_shifted`。

### 5.2 v1.1 新主指标：true-vs-shifted enrichment

```
enrichment_true_over_shift[method] = H_method_observed / H_method_shifted_median
```

- > 1 表示 真 onset 在 SOZ 标注上有 enrichment 超过随机时间点
- ≈ 1 表示 真 onset 与随机时间点等价（M1 / M2 没抓到 onset-specific 信号）
- < 1 不应该出现（出现则提示 onset 时间错位）

**v1.1 删除 v1.0 错误 abort**：v1.0 写"`null_jaccard_median > 0.4` 触发 abort"。这是错的——若某 subject 全天 HFO-rich 通道集中在固定几条电极，shifted top-k 与 true top-k 自然高重叠，这只是 spatial autocorrelation，不是 onset 错位。

**v1.1 正确 sanity**：

- 报告 `enrichment_true_over_shift[M1]` 与 `[M2]` 的 cohort 分布
- 如果 `enrichment_true_over_shift < 1` 的 subject 数 > 30%，提示系统性 onset 时间问题（停下追源）
- 默认 `n_iter = 200, rng_seed = 0`

---

## 6. Eligibility / Failure / Abort modes（v1.1）

### 6.1 Subject 级 eligibility

| 条件 | 出 | 备注 |
|---|---|---|
| Clinical SOZ 通道数 < 1 / `n_clinical_matched` < 1 | drop | 没有 audit baseline |
| Subject seizures < 2 | drop | per-seizure consistency 无法定义 |
| HFO npz 缺失 / corrupt（< 1KB） | drop | M1 不可用 |
| `sfreq < 525 Hz` | drop **M2 only**（M1 仍可跑） | Nyquist guard |
| 信号 loader 在 onset window 失败 | drop **该 seizure** | per-seizure 排除 |
| seizure onset window 内 < 2 个非零通道 | drop **该 seizure** | 数据稀疏 |

### 6.2 Cohort failure → abort（v1.1：唯一 abort 条件）

- 全 cohort 通过 eligibility 的 subject < 5 → abort，输出 audit 表但不出 cohort summary
- 全 cohort `enrichment_true_over_shift[M1] < 1` 的 subject 比例 > 30% → abort，停下追源（onset 时间错位 / channel matcher 错 / npz 损坏）

**v1.0 错误 abort 已删除**：`null_jaccard_median > 0.4` 不再是 abort 条件。

### 6.3 已知数据缺陷（前置依赖）

CLAUDE.md fast path：**Epilepsiae 旧 `*_gpu.npz` 是 216-byte stub**。新 pipeline 写入 `results/hfo_detection/<subject>/`。

**Step 0 前置检查**：

```
find results/hfo_detection -name "*_gpu.npz" -size +1k | wc -l
对齐 results/epilepsiae_block_inventory.csv 行数（每 block 一个 npz）
```

不齐 → 先 `python scripts/run_hfo_detection.py --dataset epilepsiae --all` 重检测，再进 Step 1。

---

## 7. 实现合同检查清单（v1.1）

每条 review 时勾掉，否则不接受 commit：

- [ ] **HFO npz / signal loader channel 严格对齐**：raise on mismatch；不允许 default-by-index alignment
- [ ] **Clinical SOZ 用 `match_bipolar_soz` canonical matcher**：对双极 `X-Y`，X∈SOZ 或 Y∈SOZ → SOZ；都不在 → nonSOZ；缺名 → unknown。**禁止**用 `alias_bipolar_to_left`
- [ ] **报告 `n_clinical_unmatched`**：unmatched > 0 不 raise，但 audit.csv 必须列出
- [ ] **HFO npz 大小验证**：每 block npz > 1KB
- [ ] **HFO npz 路径 = `results/hfo_detection/<subject>/*_gpu.npz`**：禁止用 `/mnt/epilepsia_data/.../all_recs/`
- [ ] **Inventory 路径 = `results/epilepsiae_seizure_inventory.csv`**：顶级，非 `dataset_inventory/`
- [ ] **Onset 时间合同**：绝对秒（sec since epoch）；Epilepsiae = `block.start_ts + onset_seconds`
- [ ] **Nyquist guard**：`sfreq/2 < band[1] * 1.05` raise NyquistGuardError；M2 在该 subject ineligible（M1 仍可跑）
- [ ] **Filter padding 合同**：`padlen >= int(1.5 * sfreq / band[0])`，不足 raise FilterPaddingError
- [ ] **Per-channel eps_ch 噪声底估计**：用 1st percentile of `power_pre` across seizures；不写死 1e-12
- [ ] **M1 报告 `M1_pois` (primary), `M1_log` (sensitivity), `M1_raw` (legacy)**：三个 variant 全在 per-subject JSON
- [ ] **M2 用 log-ratio**：`log(power_post + eps_ch) − log(power_pre + eps_ch)`；不用 `post / max(pre, eps)`
- [ ] **窗口边界缓冲**：ER-ratio 用 [edge_buffer, W_post + edge_buffer]
- [ ] **每 seizure 独立排名 → 跨 seizure 聚合**：禁止 cohort 内池化排名
- [ ] **Per-seizure consistency 用 median**：抗离群
- [ ] **Primary aggregation = `medianrank`**：consensus 仅 sensitivity
- [ ] **Primary k = `|clinical_matched|` (size-matched)**：3 / 5 / 10 仅 sensitivity
- [ ] **Random-expected enrichment 必须报告**：每 subject 每 method × k 都算
- [ ] **NO qualitative verdict**：cohort_overlap_summary 不写 "broadly_consistent" 等标签
- [ ] **True-vs-shifted enrichment 是 sanity 主指标**：不再用 top-k 自相似度做 abort
- [ ] **Stub raise**：EI / threshold sweep / band sensitivity / 'l' 'e' 分层 / CAR-bipolar 必须 `raise NotImplementedError`
- [ ] **Forbidden-phrase grep**：`grep -niE '金标准|临床.*错|true.*soz|ground.*truth|epileptogenic.*zone|EZ\b|broadly_consistent|unreliable|partially_consistent'`
- [ ] **不删除 / 不改 `*_soz_core_channels.json`**
- [ ] **rng_seed = 0 写死**

---

## 8. 文件 / 模块 map（v1.1：已校正路径）

新增：

- `src/data_driven_soz.py` — 核心估计器（M1 三 variant / M2 log-ratio / per-seizure aggregation / overlap metrics / random-expected enrichment / time-shifted null + true-vs-shifted enrichment / canonical channel matcher 包装）
- `tests/test_data_driven_soz.py` — TDD 测试 18 项
- `scripts/run_data_driven_soz.py` — CLI（`--audit` / `--per-subject` / `--cohort-overlap`）
- `scripts/plot_data_driven_soz.py` — 5 张图
- `docs/archive/topic3/pr_t3_1_data_driven_soz_audit_results_<commit-date>.md`

新增结果目录：

```
results/spatial_modulation/data_driven_soz/
├── audit.csv                                  # Step 0 前置审计
├── cohort_overlap_summary.json                # Step 4 cohort 数值（无 verdict 标签）
├── per_subject/
│   ├── epilepsiae_<sid>.json
│   └── yuquan_<sid>.json
└── figures/
    ├── README.md                              # 中文，每图 2-4 句
    ├── fig1_cohort_enrichment_distribution.png   # H_M1 / H_M2 / H_concord 直方图
    ├── fig2_size_matched_overlap_scatter.png     # observed vs random-expected per subject
    ├── fig3_per_seizure_consistency.png
    ├── fig4_subject_exemplar_high_low.png        # 高/低 enrichment exemplar 对比
    └── fig5_audit_heatmap.png                    # subject × (method × k) Jaccard
```

不动：`src/interictal_propagation.py` / `src/template_temporal_pairing.py` / `src/intra_event_spatial.py`；不动 `results/{epilepsiae,yuquan}_soz_core_channels.json`。

---

## 9. Step breakdown（v1.1）

每步独立 commit。

### Step 0 — Audit 前置依赖

- [ ] **0.1** 验证 HFO 重检测：`find results/hfo_detection -name "*_gpu.npz" -size +1k | wc -l`，与 `results/epilepsiae_block_inventory.csv` 行数对齐；不齐则停下重跑 `run_hfo_detection.py`
- [ ] **0.2** Yuquan HFO npz 路径 verify：Step 0 必须 explicitly 找到 Yuquan `*_gpu.npz` 实际位置（可能在 `results/hfo_detection/yuquan/<subject>/` 或 `/mnt/yuquan_data/.../`），写入 audit.csv
- [ ] **0.3** Sfreq audit：每 subject 报告 sfreq；标注 `m2_eligible = (sfreq >= 525)`
- [ ] **0.4** 写 `scripts/run_data_driven_soz.py --audit`：枚举 cohort，对每 subject 跑 channel matcher，报告 `n_clinical_total / n_clinical_matched / n_clinical_unmatched / n_seizures / hfo_npz_ok / sfreq / m2_eligible`
- [ ] **0.5** 输出 `results/spatial_modulation/data_driven_soz/audit.csv`
- [ ] **0.6** Commit：`feat(pr-t3-1): Step 0 — audit cohort eligibility (paths fixed, Nyquist + matcher)`

### Step 1 — M1 三 variant + TDD

新文件 `src/data_driven_soz.py`，第一组函数：

```python
def compute_hfo_onset_metrics(
    hfo_event_times_per_channel: dict[str, np.ndarray],
    seizure_onset: float,
    w_pre: float = 30.0,
    w_post: float = 10.0,
) -> dict[str, dict[str, float]]:
    """
    Return {ch: {'rate_pre': float, 'rate_post': float,
                 'M1_raw': float, 'M1_log': float, 'M1_pois': float}}.
    """

def rank_top_k_per_seizure(
    per_channel_score: dict[str, float],
    k: int,
    nan_handling: str = "rank_last",
) -> list[str]:
    """Top-k channels by score; deterministic tie-break by channel name ascending; NaN → bottom."""

def aggregate_consensus(
    per_seizure_topk: list[list[str]],
    min_seizure_fraction: float = 0.5,
) -> set[str]:
    """Channel must appear in top-k of >= min_seizure_fraction of seizures."""

def aggregate_median_rank(
    per_seizure_ranks: list[dict[str, int]],
    k: int,
) -> set[str]:
    """Median rank per channel; channels not ranked get rank=n_channels; top-k of medians."""
```

- [ ] **1.1** TDD T1：`compute_hfo_onset_metrics` M1_raw 简单算术 (post=5/10s, pre=1/30s → 0.467)
- [ ] **1.2** T2：M1_log 公式：post=5, pre=1 → log(6) − log(2) − log(10/30) = log(3) + log(3) ≈ 2.197
- [ ] **1.3** T3：M1_pois 公式：post=5, pre=1, μ=1*10/30 ≈ 0.333 → (5−0.333)/sqrt(1.333) ≈ 4.04
- [ ] **1.4** T4：通道全无 events → 三 variant 全 0（M1_pois 用 sqrt(1) 兜底 → 0）
- [ ] **1.5** T5：`rank_top_k_per_seizure` 5 通道 → top 3 deterministic
- [ ] **1.6** T6：`rank_top_k_per_seizure` NaN 通道排到底
- [ ] **1.7** T7：平局 deterministic 按通道名升序
- [ ] **1.8** T8：`aggregate_consensus` 4 seizure，A 在 3 个里 → 入选
- [ ] **1.9** T9：`aggregate_consensus` 4 seizure，A 在 1 个里 → 不入选
- [ ] **1.10** T10：`aggregate_median_rank` 4 seizure，A median rank=2 → 入选 top-3
- [ ] **1.11** 跑 `pytest tests/test_data_driven_soz.py::test_m1_* -v`，预期 10/10 PASS
- [ ] **1.12** Commit：`feat(pr-t3-1): Step 1 — M1 three variants (raw/log/pois) + aggregation TDD (10/10)`

### Step 2 — M2 log-ratio + Nyquist + per-channel eps + TDD

第二组函数：

```python
def compute_er_logratio(
    signal_loader,
    channels: list[str],
    seizure_onset: float,
    eps_per_channel: dict[str, float],            # 来自 §3.4 per-channel noise floor
    w_pre: float = 30.0,
    w_post: float = 10.0,
    edge_buffer: float = 2.0,
    band: tuple[float, float] = (80.0, 250.0),
) -> dict[str, float]:
    """log(power_post + eps_ch) − log(power_pre + eps_ch); raises NyquistGuardError / FilterPaddingError."""

def estimate_per_channel_eps(
    power_pre_matrix: np.ndarray,                 # shape (n_seizures, n_channels)
    floor: float = 1e-18,
) -> np.ndarray:
    """1st percentile across seizures per channel; bounded below by floor."""

def _bandpass_power(signal_2d, sfreq, band) -> np.ndarray:
    """Butter order 4, filtfilt zero-phase, padlen guard."""

class NyquistGuardError(ValueError): ...
class FilterPaddingError(ValueError): ...
```

- [ ] **2.1** T11：`_bandpass_power` 带内 sin 高功率 / 带外慢波 ~0
- [ ] **2.2** T12：`compute_er_logratio` mock：post 注入 burst，pre 无 → logratio > log(10) ≈ 2.3
- [ ] **2.3** T13：`compute_er_logratio` 同分布噪声 → logratio ≈ 0（容差 ± 0.5）
- [ ] **2.4** T14：边界缓冲 honored（onset ±1s spike 不进入 power）
- [ ] **2.5** T15：`compute_er_logratio` `power_pre[ch] = 0` → 用 eps_ch 兜底，不返回 inf / NaN；与 power_pre = eps_ch 等价
- [ ] **2.6** T16：Nyquist guard：sfreq=512Hz, band=(80,250) → raise NyquistGuardError
- [ ] **2.7** T17：Filter padding：极短信号 → raise FilterPaddingError
- [ ] **2.8** T18：`estimate_per_channel_eps` 1st percentile 计算 + floor 兜底
- [ ] **2.9** 跑全部 M2 测试，8/8 PASS（T11–T18）
- [ ] **2.10** Commit：`feat(pr-t3-1): Step 2 — M2 log-ratio + Nyquist guard + per-channel eps (8/8)`

### Step 3 — Per-subject runner + JSON 输出

- [ ] **3.1** 在 `scripts/run_data_driven_soz.py` 加 `--per-subject` 模式
- [ ] **3.2** Per-subject 流程：
  ```
  load seizure_onsets, hfo_npz, signal_loader, clinical_soz, analysis_channel_set
  verify HFO/signal channel 严格对齐 → raise on mismatch
  annotate clinical_soz via match_bipolar_soz → analysis-channel-level SOZ/nonSOZ/unknown 三态
  report n_clinical_total, n_clinical_matched, n_clinical_unmatched
  if subject m2_eligible:
      pre-compute power_pre_matrix across all seizures
      eps_per_channel = estimate_per_channel_eps(power_pre_matrix)
  for s in seizures:
      m1_metrics[s] = compute_hfo_onset_metrics(...)        # 三 variant
      if m2_eligible: m2_metrics[s] = compute_er_logratio(...)
  k_primary = n_clinical_matched
  for k in {3, 5, 10, k_primary, max(1, k_primary-2), k_primary+2}:
      for method in {M1_pois (primary), M1_log, M1_raw, M2_logratio (primary)}:
          for agg in {medianrank (primary), consensus}:
              top_k = aggregate(per_seizure_topk(method, k), agg)
              jaccard, precision, recall, f1 = compute_overlap(clinical_matched, top_k)
              expected_intersection = |clinical_matched| * k / |analysis_channel_set|
              random_expected_jaccard = analytical(|clinical_matched|, k, |analysis_channel_set|)
              enrichment = observed_intersection / max(expected_intersection, 0.5)
  consistency = per_seizure_topk_jaccard_median(M1_pois & M2_logratio at k_primary)
  shifted_null = time_shifted_null(M1_pois, M2_logratio, n_iter=200, rng_seed=0)
  enrichment_true_over_shift = ...
  write per_subject/<dataset>_<subject>.json
  ```
- [ ] **3.3** Per-subject JSON schema（locked）：
  ```json
  {
    "dataset": "...",
    "subject": "...",
    "n_seizures_used": int,
    "n_channels_total": int,
    "sfreq": float,
    "m2_eligible": bool,
    "channel_matching": {
      "n_clinical_total": int,
      "n_clinical_matched": int,
      "n_clinical_unmatched": int,
      "unmatched_clinical_names": [str, ...]
    },
    "baseline_rate_per_channel": {ch: float},
    "k_primary_size_matched": int,
    "results": {
      "M1_pois": {
        "consensus":  {"k_primary": [...], "k3": [...], "k5": [...], "k10": [...], ...},
        "medianrank": {"k_primary": [...], ...}
      },
      "M1_log":  {...},
      "M1_raw":  {...},
      "M2_logratio": {...}
    },
    "overlap_with_clinical": {
      "M1_pois_medianrank_kPrimary":  {"jaccard": float, "precision": float, "recall": float, "f1": float,
                                       "observed_intersection": int, "random_expected_intersection": float,
                                       "enrichment": float, "random_expected_jaccard": float},
      ...
    },
    "headline_primary": {
      "H_M1_pois_medianrank_size_matched":   float,
      "H_M2_logratio_medianrank_size_matched": float,
      "H_concord_M1_M2_size_matched":          float
    },
    "per_seizure_consistency": {"M1_pois_kPrimary": float, "M2_logratio_kPrimary": float},
    "time_shifted_null": {
      "n_iter": 200,
      "rng_seed": 0,
      "enrichment_true_over_shift_M1_pois":   float,
      "enrichment_true_over_shift_M2_logratio": float
    }
  }
  ```
- [ ] **3.4** 跑全 cohort（按 Step 0 audit eligible 名单，rng_seed=0）
- [ ] **3.5** Commit：`feat(pr-t3-1): Step 3 — per-subject metrics + size-matched primary + enrichment`

### Step 4 — Cohort overlap summary（**NO verdict labels**）

- [ ] **4.1** 加 `--cohort-overlap` 模式
- [ ] **4.2** 计算 cohort 表格：
  - per (method × aggregation × k) 的 Jaccard / Precision / Recall / F1 / enrichment 中位数 + IQR + n
  - headline 三个数 `H_M1_pois / H_M2_logratio / H_concord` 的 cohort 中位数 + IQR + 直方图分箱
  - true-vs-shifted enrichment 的 cohort 分布
- [ ] **4.3** **NO qualitative verdict**：JSON 字段不含 `verdict`、不含 `broadly_consistent` 等标签；只含数值
- [ ] **4.4** 输出 `results/spatial_modulation/data_driven_soz/cohort_overlap_summary.json`：
  ```json
  {
    "n_subjects_in_audit": int,
    "n_subjects_m2_eligible": int,
    "headline": {
      "H_M1_pois_medianrank":   {"median": float, "iqr": [lo, hi], "n": int, "n_above_1": int, "n_above_2": int},
      "H_M2_logratio_medianrank": {...},
      "H_concord_M1_M2":          {...}
    },
    "true_vs_shifted_enrichment": {
      "M1_pois":   {"median": float, "iqr": [lo, hi], "n_below_1": int},
      "M2_logratio": {...}
    },
    "per_method_k_full_table": {
      "M1_pois_medianrank_kPrimary":     {...},
      ...  // 完整 (method × agg × k) 网格
    },
    "per_seizure_consistency_summary": {...},
    "downstream_decision_inputs": {
      "n_subjects_with_H_M2_below_1.5":   int,
      "n_subjects_with_concord_below_1":  int,
      "n_subjects_with_consistency_below_0.3": int
    }
  }
  ```
- [ ] **4.5** Commit：`feat(pr-t3-1): Step 4 — cohort enrichment summary (no verdict labels)`

### Step 5 — Visualization

- [ ] **5.1** Fig 1：cohort enrichment 直方图（H_M1 / H_M2 / H_concord 三条），加随机水平 x=1 与 substantial x=2 参考线
- [ ] **5.2** Fig 2：observed intersection vs random-expected intersection 散点（per subject，按 |clinical_matched| 大小着色，验证 size effect）
- [ ] **5.3** Fig 3：per-seizure consistency vs subject seizure count 散点（看 sparse seizure 是否拖低 consistency）
- [ ] **5.4** Fig 4：高/低 enrichment exemplar 对比（每个 subject 画 channel-by-seizure top-k heatmap，并叠加 clinical SOZ 与 baseline rate）
- [ ] **5.5** Fig 5：subject × (method × k) Jaccard heatmap（含 size-matched k_primary 列高亮）
- [ ] **5.6** 生成 `results/spatial_modulation/data_driven_soz/figures/README.md`，每图 2–4 句中文 + "**关注点**"行
- [ ] **5.7** Commit：`feat(pr-t3-1): Step 5 — visualization (enrichment-centric, no verdict)`

### Step 6 — Doc closeout

- [ ] **6.1** 写 `docs/archive/topic3/pr_t3_1_data_driven_soz_audit_results_<commit-date>.md`：
  - §1 Cohort & eligibility（paths verify + sfreq distribution + clinical match counts）
  - §2 M1 三 variant headline 数值（特别看 M1_pois vs M1_raw 是否一致）
  - §3 M2 headline 数值（仅 m2_eligible cohort）
  - §4 M1 ∩ M2 concordance（headline H_concord）
  - §5 Random-expected enrichment 与 size effect
  - §6 Per-seizure consistency
  - §7 True-vs-shifted enrichment sanity
  - §8 本 PR **能 / 不能** 说明什么（§3.10 显式列表）
  - §9 对 PR-8 v2 / topic2 SOZ-stratified PR 的 inputs（不做决策，只列数值）
  - §10 Allowed / forbidden phrasings actually used
  - §11 Figure inventory
- [ ] **6.2** 在 `docs/topic3_spatial_soz_modulation.md` 加 PR-T3-1 数值摘要 + archive 链接（**不**贴 qualitative verdict）
- [ ] **6.3** Forbidden-phrase grep：`grep -niE '金标准|临床.*错|true.*soz|ground.*truth|epileptogenic.*zone|EZ\b|broadly_consistent|unreliable|partially_consistent'`
- [ ] **6.4** Commit：`docs(pr-t3-1): Step 6 — close out PR-T3-1 (numerical-only, no verdict)`

---

## 10. TDD 测试列表（v1.1，18 项）

| ID | 函数 | 关注 invariant |
|---|---|---|
| T1 | `compute_hfo_onset_metrics` | M1_raw 算术 |
| T2 | `compute_hfo_onset_metrics` | M1_log 算术 + log(W_post/W_pre) 修正 |
| T3 | `compute_hfo_onset_metrics` | M1_pois Poisson z 算术 |
| T4 | `compute_hfo_onset_metrics` | 全无 events → 三 variant 全 0 |
| T5 | `rank_top_k_per_seizure` | k=3 + 5 通道 → top 3 |
| T6 | `rank_top_k_per_seizure` | NaN 通道排到底 |
| T7 | `rank_top_k_per_seizure` | 平局 deterministic 按通道名 |
| T8 | `aggregate_consensus` | 50% 阈值正例 |
| T9 | `aggregate_consensus` | 50% 阈值反例 |
| T10 | `aggregate_median_rank` | median rank + tie 处理 |
| T11 | `_bandpass_power` | 带内 sin 高功率 / 带外慢波 ~0 |
| T12 | `compute_er_logratio` | post burst → logratio > 2.3 |
| T13 | `compute_er_logratio` | 同分布噪声 → logratio ≈ 0 |
| T14 | `compute_er_logratio` | 边界缓冲 honored |
| T15 | `compute_er_logratio` | power_pre=0 → eps_ch 兜底 |
| T16 | `compute_er_logratio` | Nyquist guard sfreq=512 raise |
| T17 | `compute_er_logratio` | filter padding 不足 raise |
| T18 | `estimate_per_channel_eps` | 1st percentile + floor |

`tests/test_data_driven_soz.py` 一文件全收。所有测试用 `np.testing.assert_*` + `pytest.raises`；不依赖外部数据；信号合成用 numpy 直接生成。

---

## 11. Visualization spec（v1.1，enrichment-centric，无 verdict 标签）

| Fig | 内容 | size | 命名 |
|---|---|---|---|
| 1 | cohort enrichment 直方图（H_M1/H_M2/H_concord）| 12 × 5 | `fig1_cohort_enrichment_distribution.png` |
| 2 | observed vs random-expected intersection 散点 | 8 × 6 | `fig2_size_matched_overlap_scatter.png` |
| 3 | per-seizure consistency vs n_seizures | 8 × 5 | `fig3_per_seizure_consistency.png` |
| 4 | 高/低 enrichment exemplar 对比 | 14 × 8 | `fig4_subject_exemplar_high_low.png` |
| 5 | subject × (method × k) Jaccard heatmap | 12 × n_subjects×0.4 | `fig5_audit_heatmap.png` |

Fig 1 需画 x=1（random）和 x=2（substantial enrichment）参考线。颜色复用 `src/plot_style.py` Morandi。

---

## 12. 成功 / 失败的判读语言（v1.1：仅数值，不贴标签）

| 内容 | 允许说 | 严禁说 |
|---|---|---|
| Cohort headline | "Cohort 上 H_M1_pois_medianrank_size_matched 中位数 = X.XX (IQR [Y, Z])，N/M subject 高于 random expected (>1.0)，K/M 高于 substantial (>2.0)" | "broadly_consistent / partially_consistent / unreliable"（三档定性已删） |
| Clinical SOZ 评估 | "Cohort 上数据驱动 proxy 与 clinical SOZ overlap 在某档数值范围；下游 SOZ-dependent 检验在 H_concord < 1.5 cohort 比例为 X% 时建议改 multi-source 报告" | "证明 clinical SOZ 是错的 / 是金标准 / 必须废弃" |
| 数据驱动 SOZ 评估 | "M1_pois 与 M2_logratio 在 N/M subject 上互相 enrichment ≥ 1.5；两条 proxy 在 K/M subject 上 disagreement" | "数据驱动 SOZ 是真实 epileptogenic zone / EZ / true SOZ" |
| True-vs-shifted | "enrichment_true_over_shift 中位数 = X，N/M subject < 1（提示 onset 时间或 channel matcher 问题，需要逐 subject 复核）" | "shifted null 失败说明 M1/M2 不工作" |

无论数值如何，主文档 / archive 都**严禁**：

- "金标准 / true SOZ / ground truth" 描述任何 SOZ 定义
- "EZ / epileptogenic zone" 命名数据驱动 SOZ
- 三档 qualitative verdict 标签
- 删除 / 重写 `*_soz_core_channels.json`
- 单 subject 极端数值推及 cohort 论断
- 把 M1 / M2 称作"两条独立证据 / two independent channels"——它们是**两条 proxy**，频段与 onset 锚点高度相关

---

## 13. Out of scope（写明）

- EI（Bartolomei detection time + tonicity gating） → PR-T3-2
- 30–100 Hz / 12–127 Hz band sensitivity → PR-T3-2
- Threshold 自动优化 → PR-T3-2
- i / l / e 三层 clinical 分层比较 → PR-T3-2
- CAR / bipolar montage sensitivity → PR-T3-2
- 替换 `*_soz_core_channels.json` → 永远不做
- Qualitative SOZ-reliability verdict → 永远不做（v1.1 / v2 / v3 都不做）
- topic1 PR-8 v2 重启 → 等本 PR 完成
- topic2 SOZ-stratified PR 重启 → 等本 PR 完成

---

## 14. 自检清单（v1.1）

- [x] 路径已校正：`results/epilepsiae_seizure_inventory.csv`（顶级）+ `results/hfo_detection/<subject>/*_gpu.npz`
- [x] Channel matching 用 canonical bipolar-to-any matcher，**不**用 alias_bipolar_to_left；clinical SOZ 软标注（SOZ/nonSOZ/unknown），**不**做 set-equality 严格对齐
- [x] M1 三 variant（raw / log / pois），primary = `M1_pois`；报告 baseline_rate_per_channel
- [x] M2 改 log-ratio + per-channel eps + Nyquist guard + filter padding 合同
- [x] Top-k primary = `|clinical_matched|`（size-matched）；3/5/10 仅 sensitivity
- [x] Random-expected enrichment 必须报告（每 subject 每组合）
- [x] **NO three-tier qualitative verdict**：cohort_overlap_summary 无 `verdict` 字段
- [x] True-vs-shifted enrichment 替换 v1.0 错误的 top-k 自相似 abort
- [x] §3.10 / §12 / §13 显式写明本 PR 能/不能说明什么
- [x] M1 / M2 framing 改成"两条 proxy"，不再写"独立证据"
- [x] 14 项 → 18 项 TDD（加 M1_log / M1_pois / Nyquist / filter padding / per-channel eps）
- [x] §7 实现合同 19 → 24 条
- [x] forbidden-phrase grep 加 `broadly_consistent|unreliable|partially_consistent`
- [x] **不**改 / **不**覆盖 `*_soz_core_channels.json`

---

## 15. 一句话承诺

PR-T3-1 v1.1 是 audit 工程，不是判决工程。输出 reusable JSON 与 cohort enrichment 数值表格，让下游 PR（topic1 PR-8 v2 / topic2 SOZ-stratified）能基于 multi-source SOZ 定义做更稳健的分析。**verdict 的最强陈述只是"clinical SOZ 与 data-driven proxy 在 cohort 上的 enrichment 数值范围"——不论谁对，不改 `*_soz_core_channels.json`，不出 qualitative verdict 标签**。下游 PR 的 SOZ 策略由人类读数值后决定，不由本 PR 自动驱动。
