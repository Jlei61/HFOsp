# PR-T3-1 Pivot — Layer A ictal ER-rank producer + Layer B data-driven SOZ label & overlap audit

> 状态：plan-of-record（**v2.1 pivot, post-review**），2026-05-03
> 取代：`docs/archive/topic3/pr_t3_1_data_driven_soz_audit_plan_2026-04-30.md` (v1.1)
> v2.0 → v2.1 修订（基于 2026-05-03 用户审阅）：
> - 明确 PR-6A **superseded**，仅 Step 0-2 落定为 preview 经验；本 PR **不**继承 PR-6A H1/H1' 主线，**不**重做 Smith 2022 template-ictal alignment。
> - 把 PR-T3-1 v2 拆成两层：**Layer A**（新建的、范围受限的 ictal ER-rank producer）+ **Layer B**（label consumer + audit）。
> - v1.1 HFO-rate 资产**分阶段**归档（先 banner + 改目录名 + 标 obsolete；等 v2 跑通才考虑迁出代码），不直接删除。
> - Step 1 阻塞对象改成"Layer A 通过 stability gate"，不是"PR-6A Step 3-5 完成"。
> - 把 `broad_ER` 设为 **independence sensitivity 更关键的一档**，因为 `gamma_ER` (60–100 Hz) 仍与 HFO band (80–250 Hz) 部分重叠。
> - 加 honest-failure mode：如果 Layer A 在 cohort 上 stability 不达标，Layer B 不运行，PR 诚实归档"目前没有可靠第二套 data-driven SOZ 标签"。
>
> 范围：在 topic3 §1 "where" 主线上，提供与 clinical SOZ 并报的第二套 SOZ 标签来源；该来源必须**与 HFO 事件检测无关**以打破循环验证。
>
> 本 PR **依然不**：出 EI、出三档 verdict、替换 `*_soz_core_channels.json`、对 clinical SOZ 做 "对/错" 判读、复活 PR-6A H1/H1'。

---

## 0. 为什么 v1.1 必须 pivot（结构性问题，不是工程问题）

v1.1 跑完全 cohort 24 个 subject、两轮 review 后，3 条结构性缺陷暴露：

### 0.1 M1 ≡ M2，不是两条独立 proxy

v1.1 §3.3 / §3.4：

- **M1** = HFO 事件计数在 onset 前后的 enrichment（rate-based）。
- **M2** = 80–250 Hz 带功率在 onset 前后的 log-ratio（continuous-signal-based）。

但 HFO 检测本身就是 80–250 Hz 包络阈值过线，**M2 是 M1 的"未阈值化"版本**。两者高度相关——cohort 数字佐证：多数 subject 上 H_M1 / H_M2 方向一致，仅在阈值边缘漂移。"两条独立 proxy 一致 → 数据驱动 SOZ 可信" 在 v1.1 里是同源信号自我对齐。

### 0.2 M2 的 ER-ratio 定义已经从 PR-6A 漂移

PR-6A (`docs/archive/topic1/pr6a_template_ictal_alignment_plan_2026-04-21.md`) 已经在 Step 0-2 落定并 sentinel 验证过 ER 提取层（gamma 60–100 / 4–20 主，broad 12–127 / 4–20 sensitivity，4–250 Hz bandpass + 1 s 滑窗 + log ratio + per-channel z-score against pre-ictal baseline）；Step 3+ (Page-Hinkley CUSUM, n_d, r_sz, stability gate) 在 sentinel 暴露跨 seizure 稳定性问题后**未进入正式 H1/H1' 主线**，PR-6A 整体 superseded。

v1.1 的 M2 完全没有这套 z-score / CUSUM / stability gate，只是简单 80–250 Hz 带功率 post/pre log-ratio。这个量没有被任何工作验证作为 SOZ 排序信号。

### 0.3 循环论证风险（最严重）

topic3 下游要验证的命题（topic1 §6 共识候选 + topic3 §3）：

- 刻板时序**反转节点**（PR-7 antagonistic pairing）↔ SOZ 关系
- 刻板时序**共用节点**（PR-5/6 template recruitment / endpoint anchoring）↔ SOZ 关系

如果 data-driven SOZ 也从 HFO rate / HFO band-power 派生：

- 反转节点本身从 HFO template 推出来
- data-driven SOZ 也从 HFO 信号推出来
- 两者都共享 HFO 检测器的偏差和噪声

"反转节点和高 HFO-rate 通道相关" 部分是定义重叠，**不是独立验证**。要打破循环，data-driven SOZ 必须用与 HFO 事件检测无关的连续信号特征产出。**Ictal continuous-signal ER ranking 是当前最务实的候选**：直接消费原始 .data / .edf 信号，不依赖 HFO 检测器的过阈门限。但**这个 producer 还不存在**——本 PR 必须自己建（Layer A），不能假装它已经验证。

### 0.4 v1.1 的 cohort 已跑结果归档为"反例证据"

v1.1 跑出来的 24 份 per-subject JSON 不能服务于 "避免循环论证的 SOZ 标签" 目的，但有诊断价值——它们证明 HFO-rate-based 和 HFO-band-power-based SOZ **数值相似**（M1 / M2 方向一致率高）。归档到 `results/spatial_modulation/data_driven_soz/per_subject_hfo_rate_obsolete_v1_1/` 保留，作为"为什么 PR-T3-1 v1.1 不能产出独立 SOZ 标签"的实证支持，**不**作为 topic3 下游 SOZ 标签来源。

---

## 1. Context — 在 topic3 / 整体路线图里的位置（v2.1）

PR-T3-1 v2.1 是 topic3 下游 SOZ-stratified 分析的**第二根 SOZ 标签产出 PR**。第一根（已有）是 clinical SOZ：`results/epilepsiae_soz_core_channels.json` + `results/yuquan_soz_core_channels.json`。第二根（本 PR 产出）是 data-driven SOZ from a **scope-restricted, newly built ictal ER-rank producer**（Layer A），消费成本最小化（Layer B）。

下游消费者（topic3 PR-8 v2 / topic2 SOZ-stratified / topic1 §6 SOZ 关联）必须**两套并报**：

```
∀ 下游 SOZ-dependent 检验：
    对 clinical_SOZ_label 跑一次  → 得 (effect, p)_clinical
    对 data_driven_SOZ_label 跑一次 → 得 (effect, p)_data_driven
    并列报告两组数；不做 cohort-level 加权融合
```

如果两套结论一致 → 该结论稳健。如果只在 clinical 上成立 → 暴露 clinical 标注的影响；如果只在 data-driven 上成立 → 暴露数据驱动定义本身的偏差。两种偏差都比"用单一标签报告""结论稳健""更诚实。

PR-T3-1 v2.1 **不**判定 clinical 和 data-driven 谁对谁错；只产出两套标签 + 它们之间的 overlap audit 数值。

### 1.1 与 PR-6A 的明确边界

| 项 | PR-6A (superseded) | PR-T3-1 v2.1 Layer A |
|---|---|---|
| ER 提取层（gamma_ER / broad_ER, 4-250 Hz, 1 s 滑窗, log ratio, baseline z-score） | 已落定 Step 0-2，**保留并复用其源代码 + 接口** | 复用 PR-6A `src/ictal_onset_extraction.py` 的 Step 0-2 helpers（`compute_er`, `baseline_zscore_er`, `extract_seizure_window`, `resolve_baseline_window`） |
| Page-Hinkley CUSUM + n_d 检测 | Sentinel preview 暴露稳定性问题，**未进入正式 H1/H1' 主线** | **新实现**，但只服务 r_sz 输出，不服务 template alignment |
| `r_sz` 跨 seizure 中位 rank 构造 | superseded | **新实现**，scope-restricted（不附带 H1 / H1' 推断） |
| stability gate（s_sz ≥ 0.5 strict / 0.3 relaxed） | 设计已锁定但未跑 cohort | **新实现 + 跑 cohort**；阈值复用 PR-6A 设计 |
| Smith 2022 template alignment / H1 / H1' / r_template_corr | superseded | **不做**（明确 out of scope）|
| Sanity gate (focal vs non-focal z-ER 分离) | sentinel 已 PASS（548 + 916） | 复用 sentinel 结果作为已知 baseline；本 PR 自己 cohort 内 sanity 不与 H1 联动 |

**核心**：PR-6A 的 Step 0-2 ER + baseline z-score 是已验收的工程基础（src module 与 sentinel result 都在），但 r_sz / stability gate / 上层假设检验 **从未跑通到 cohort**。本 PR Layer A 正是"在已验收的 Step 0-2 之上，**只为 SOZ label 这一目的**新建 r_sz + stability gate"，不复活 H1/H1'。

---

## 2. 编号与归档决定

- **PR 编号**：保留 PR-T3-1（topic3 §1 "where" 主线第 1 个 PR）。两层用 `Step A.x` / `Step B.x` 区分。
- **plan-of-record**：本文件。`pr_t3_1_data_driven_soz_audit_plan_2026-04-30.md` (v1.1) 标记为 superseded（顶部加 banner，指向本文件）。
- **代码命名**：
  - Layer A 复用 `src/ictal_onset_extraction.py`（已有 Step 0-2 helpers），扩展新模块 `src/ictal_er_rank.py`（CUSUM + n_d + r_sz + stability，scope-restricted）。
  - Layer B 用新模块 `src/data_driven_soz_pivot.py`（label builder + audit），与 v1.1 `src/data_driven_soz.py` 暂时并存（v1.1 标 obsolete 但不删；§9 Step 0 详述）。
  - CLI：`scripts/run_ictal_er_rank.py`（Layer A）+ `scripts/run_data_driven_soz.py` 复用为 Layer B（v1.1 模式标 obsolete）。

---

## 3. 假设与统计合同（核心）

### 3.1 数据对象 — 两层 producer/consumer 接口

**Layer A — ER-rank producer**（新建，scope-restricted）：

```
input  : seizure 中心切片信号 (continuous .data / .edf)
output : per-subject JSON, schema §3.5
contract:
  - 复用 PR-6A Step 0-2 已验收的 ER 提取 + baseline z-score
  - 新加 Page-Hinkley CUSUM + n_d 第一报警时刻 + r_sz 中位 rank
  - 新加 stability gate (s_sz ≥ 0.5 strict / 0.3 relaxed / < 0.3 dropped)
  - 不做 template alignment / H1 / H1'
  - 不做 per-subject 后验切换 ER config
```

**Layer B — label consumer + audit**：

```
input  : Layer A 输出 + clinical SOZ JSON + audit.csv (v1.1 已有)
output : data_driven_soz_core_channels_<er>_<stability>.json (4 份)
         + cohort_overlap_summary.json
contract:
  - 严格按 audit_eligible 24 与 Layer A cohort 求交集
  - 取 size-matched k = |clinical_matched|
  - overlap audit 数值，无 verdict label
  - 两套 ER × 两套 stability cohort = 4 套独立标签，禁止融合
```

### 3.2 Channel matching 合同（保留 v1.1 §3.2，commit `d0fae20` 修复后版本）

- **Layer A signal channel ↔ analysis channel**：严格对齐（Layer A 的 channel set 即 analysis set，无 alias），不一致 raise ValueError
- **Clinical SOZ → analysis_channel_set 标注**：用 canonical 3-state matcher `annotate_clinical_soz`：
  - 双极 `X-Y`：X 或 Y 在 clinical_soz → `SOZ`；都不在 → `nonSOZ`；任一端为名字残缺 / 空 → `unknown`
  - **逐端点 normalize**（commit `d0fae20` 修复，handle "EEG A1-EEG A2" 等 dual-prefix）
  - 禁止 `alias_bipolar_to_left` / 禁止 set-equality 严格对齐
- **Audit 透明性**：每 subject 报 `n_clinical_total` / `n_clinical_matched` / `n_clinical_unmatched` / `unmatched_clinical_names`（v1.1 audit.csv schema 保留）

### 3.3 Layer A — ER-rank producer 合同（新建，scope-restricted）

继承 PR-6A Step 0-2 已落定参数，**禁止运行时第三套自定义 band**：

| key | E_fast | E_slow | 定位（v2.1）|
|---|---|---|---|
| `gamma_ER` | `∫_{60}^{100} \|S(f,n)\|² df` | `∫_{4}^{20} \|S(f,n)\|² df` | 主（HFO-centered） |
| `broad_ER` | `∫_{12}^{127} \|S(f,n)\|² df` | `∫_{4}^{20} \|S(f,n)\|² df` | **independence sensitivity（v2.1 升格）**，与 HFO 80-250 Hz 重叠最少，最适合做"独立于 HFO" 标签 |

通用约束（两套共享，复用 PR-6A 已落定）：

- bandpass 4–250 Hz（Epilepsiae 主流 1024 Hz / Yuquan 2000 Hz 都支持）
- `ER[n] = log(E_fast[n] / E_slow[n])`
- 滑窗 1 s，步长 100 ms
- baseline window：`[-300 s, baseline_end_sec]` 相对 clinical onset，`baseline_end_sec = min(0, eeg_onset_rel_sec) - 60 s`；baseline-invalid (有效长度 < 60 s) → seizure 整体 drop
- per-channel z-score against baseline（identity-bias 抵消层）
- 通道 baseline-valid 样本 < 60 s → 该 seizure 该通道 drop（不进 rank）

Layer A 新加层（v2.1 自己实现，禁止从 PR-6A H1/H1' 主线借）：

- Page-Hinkley CUSUM (clamped, upward-change form)：`U[n] = max(0, U[n-1] + z_ER[n] - bias)`，`bias = 0.5`，alarm `n_d` = first n where `U[n] >= λ`（修订自 2026-05-03 implementation：原 plan 草稿写的 `M[n] - U[n] >= λ` 形式对持续上行 step 不会触发——step 期间 U 单调非减、M = U、M - U = 0；`U >= λ` 是 PR-6A `detect_er_onset_preview` 已经使用的可工作形式，对 unclamped `S = sum(z - bias)` 等价于 `S - min(S[k≤n]) >= λ`）
- λ 通过 per-subject **baseline re-armed CUSUM budget** 校准（修订自 2026-05-03：原 "permutation" 描述与实际实现不符）：在 baseline window 上对每通道跑 clamped CUSUM，每次 `U >= λ` 触发即计 1 次假阳报警并将 `U` 重置为 0（re-armed），对所有通道求和；λ 取 grid 上最小满足 `pooled_alarms ≤ fpr_target_per_hour × baseline_hours` 的值。Default fpr_target = 1/hour（plan §3.3 与 v1.1 一致）。Pooled across-channel 合同：subject-level FPR 对齐 cohort 报告口径，per-channel FPR 不写入接口
- detection window：`[clinical_onset - 5 s, clinical_onset + 30 s]`
- Tie-handling：fractional rank for ties (Δt < 50 ms)；`>60%` channels detected within `[onset, onset + 1 s]` → seizure 标 `onset_tied`，剔除主分析；`<30%` channels valid n_d → seizure 标 `onset_unreached`，剔除
- per-subject `r_sz` = median rank per channel across qualifying seizures
- stability：`s_sz` = median pairwise Spearman ρ on r over all seizure pairs；阈值 0.5 strict / 0.3 relaxed / <0.3 drop

**强 invariant（合同级，不可松动）**：

- 两套 ER 配置 (`gamma_ER`, `broad_ER`) 全程并行跑、独立产出独立的 `r_sz` / `s_sz` / 独立 cohort 决定。**不允许** per-subject 后验切换。
- 主配置切换只在 cohort-level sanity gate 后允许（详见 §6.2）。
- Layer A **不输出** template alignment / Smith 2022 H1 / H1' 相关字段。

### 3.4 Layer A — sentinel sanity（复用 PR-6A）

继承 PR-6A 已锁定的 sentinel：

| key | subject | 入选理由 |
|---|---|---|
| `sentinel_A` | `epilepsiae/548` | 9-subset 中 seizure 数最多 |
| `sentinel_B` | `epilepsiae/916` | 普通 k=2、reproducibility=strong、不在 9-subset |

Layer A 在跑 cohort 前先在 sentinel 上目视检查：

- z-ER trace 图（focal vs non-focal pre30s / post30s 中位 max）已由 PR-6A Step 2 产出，路径 `results/interictal_propagation/ictal_alignment/_sentinel_step2/<subject>_<seizure_idx>_<er>.png` —— **复用作为已知 baseline**
- v2.1 新加：sentinel 上的 CUSUM 报警时刻 `n_d` per channel + per-subject `r_sz` + `s_sz` —— 由 v2.1 Layer A 新跑

如果 sentinel `r_sz` 上 focal 通道的 rank 中位数 不显著早于 non-focal （Wilcoxon p > 0.1），Layer A pipeline 标可疑，写 archive sentinel report，**不进 cohort run**。

### 3.5 Layer A — per-subject JSON schema（v2.1 锁定）

`results/data_driven_soz/layer_a_ictal_er_rank/per_subject/<dataset>_<subject>.json`：

```json
{
  "schema_version": "pr_t3_1_layer_a_v2_1",
  "dataset": str,
  "subject": str,
  "n_seizures_total": int,
  "n_seizures_qualifying": int,
  "channel_names": [str, ...],
  "stability": {
    "s_sz_gamma_ER": float,
    "s_sz_broad_ER": float,
    "cohort_assignment_gamma": "strict" | "relaxed" | "dropped",
    "cohort_assignment_broad":  "strict" | "relaxed" | "dropped"
  },
  "r_sz": {
    "gamma_ER": {"<channel>": float, ...},
    "broad_ER": {"<channel>": float, ...}
  },
  "n_d_per_seizure": {                # detection times for downstream sanity / re-rank
    "gamma_ER": [{"seizure_id": str, "<channel>": float|null, ...}, ...],
    "broad_ER": [...]
  },
  "seizure_status": [
    {"seizure_id": str, "status": "ok"|"onset_tied"|"onset_unreached"|"baseline_invalid", ...},
    ...
  ],
  "diagnostics": {
    "lambda_per_subject": {"gamma_ER": float, "broad_ER": float},
    "baseline_window_sec": [float, float],
    "detection_window_sec": [float, float]
  },
  "provenance": {
    "pr6a_module_used": "src/ictal_onset_extraction.py",
    "layer_a_module": "src/ictal_er_rank.py",
    "plan_doc": "docs/archive/topic3/pr_t3_1_pivot_to_pr6a_er_ranking_2026-05-03.md"
  }
}
```

### 3.6 Layer A — honest-failure 合同（v2.1 新加）

如果 cohort 上：

- 两套 ER 配置都没有 ≥ 5 subjects 通过 strict gate，**且**没有 ≥ 10 subjects 通过 relaxed gate → Layer A 整体 fail，PR-T3-1 在 archive 中明确："当前 cohort 上 ictal ER ranking 不能产生 cohort-scale 可靠的第二套 SOZ 标签"。Layer B 不运行，**不**输出 fake label JSON。
- 仅 broad_ER 通过、gamma_ER 没通过 → broad_ER 单独进入 Layer B；archive 注明 gamma_ER 失败。
- 仅 gamma_ER 通过、broad_ER 没通过 → 写入 archive，但 Layer B **不运行 broad_ER 之外的标签**：因为 gamma_ER 与 HFO 80-250 Hz 仍部分重叠，单独 gamma 标签不能完全打破循环验证。需要在主文档明确告知"目前 data-driven SOZ 的 HFO-independence 不充分"。

honest-failure 优于 fake-success；任何 "硬造一套不可靠的 data-driven SOZ" 都是合同 violation。

### 3.7 Layer B — top-k：size-matched k = |clinical_matched|（保留 v1.1 §3.6）

每 subject × 每套 ER 配置 × 每个 stability cohort：

```
k = max(|clinical_matched|, 1)
data_driven_top_k = sorted(channels by r_sz)[:k]   # r_sz 升序
```

主报告 k = |clinical_matched|；sensitivity k ∈ {3, 5, 10, k_primary - 2, k_primary + 2}。

### 3.8 Layer B — SOZ 标签 JSON 输出（v2.1 锁定）

`results/data_driven_soz/layer_b_labels/data_driven_soz_core_channels_<er_config>_<stability_cohort>.json`：

```json
{
  "schema_version": "pr_t3_1_layer_b_v2_1",
  "er_config": "gamma_ER" | "broad_ER",
  "stability_cohort": "strict" | "relaxed",
  "k_rule": "size_matched_clinical",
  "n_subjects_in_label": int,
  "subjects": {
    "<dataset>_<subject>": {
      "clinical_matched_channels": [str, ...],
      "data_driven_soz_channels": [str, ...],
      "k_primary": int,
      "rsz_full_rank": {"<channel>": float, ...},
      "stability_score": float
    },
    ...
  },
  "provenance": {
    "layer_a_per_subject_dir": "results/data_driven_soz/layer_a_ictal_er_rank/per_subject",
    "layer_a_sanity_passed": bool,
    "audit_csv": "results/spatial_modulation/data_driven_soz/audit.csv",
    "clinical_soz_source": {
      "epilepsiae": "results/epilepsiae_soz_core_channels.json",
      "yuquan": "results/yuquan_soz_core_channels.json"
    },
    "layer_b_module": "src/data_driven_soz_pivot.py",
    "plan_doc": "docs/archive/topic3/pr_t3_1_pivot_to_pr6a_er_ranking_2026-05-03.md"
  }
}
```

`rsz_full_rank` 是关键字段：保留全 channel 的 r_sz 数值，让下游 PR 能在不重跑 Layer A 的前提下做不同 k / 不同阈值的 sensitivity。

### 3.9 Layer B — overlap audit（保留 v1.1 §3.7）

每 (ER_config × stability_cohort × k)：

```
A = clinical_matched
B = data_driven_soz_channels (top-k by r_sz)

jaccard(A, B)   = |A ∩ B| / |A ∪ B|
precision(A, B) = |A ∩ B| / |B|
recall(A, B)    = |A ∩ B| / |A|
f1(A, B)        = 2 P R / (P + R)

random_expected_intersection(|A|, |B|, n_total) = |A| * |B| / n_total
random_expected_jaccard(|A|, |B|, n_total)      = analytical or 1000-iter MC

enrichment[ER × stability × k] = observed_intersection / max(expected_intersection, 0.5)
```

cohort 报告：`enrichment` 中位数 + IQR + 直方图 per (ER_config × stability_cohort × k)。**不做** cohort scalar p-value。**不做** verdict label。

### 3.10 Layer B — gamma vs broad 必须并报（v2.1 强化）

每个 audit 数值在 `gamma_ER` 与 `broad_ER` 上各跑一遍，并列报告。

**v2.1 新加 independence audit**：在主文档 + archive 中显式比较：

- `gamma_ER` SOZ 标签 vs 现有 HFO-rate-based ranking（v1.1 M1 cohort 数据）的 jaccard 中位数
- `broad_ER` SOZ 标签 vs 现有 HFO-rate-based ranking 的 jaccard 中位数

**预期**：`broad_ER` 与 HFO-rate 重叠更低（跨度更宽 + 包含 < 80 Hz）→ 适合作为下游"避免循环验证" 的主标签；`gamma_ER` 重叠更高 → 作为 sensitivity 报，不作 HFO-related 下游分析的主标签。如果实际数据显示反过来，写入 archive 作为意外发现。

### 3.11 NO verdict（保留 v1.1 §3.9）

- audit 不出 "broadly_consistent / partially_consistent / unreliable" 三档定性
- `data_driven_soz_core_channels_*.json` 只列通道，不带可信度等级
- 主文档 §结论 只描述数值

### 3.12 本 PR 能 / 不能说明什么（v2.1）

| 能说明 | 不能说明 |
|---|---|
| 提供两套 SOZ 标签（clinical + data-driven from new Layer A）供 topic3 下游 cross-validate | clinical SOZ 是错的 / data-driven SOZ 是真的 |
| Clinical SOZ 与 newly-built ictal ER ranking 的 overlap 数值 | 哪些 subject 的 clinical SOZ 不可靠 |
| `gamma_ER` / `broad_ER` 两种 ER 定义对 SOZ ranking 的敏感度 | 真实 epileptogenic zone 在哪里 |
| stability gate (strict/relaxed) 在 cohort 上的覆盖率 | PR-6A H1/H1' 的状态（明确不复活）|
| Honest 报告："目前 data-driven SOZ 的 HFO-independence 是否充分" | LVFA / non-LVFA onset pattern 的机制 |

---

## 4. 不做的部分（v2.1 out of scope）

- **PR-6A H1/H1' 主线**（template-ictal alignment / Smith 2022 双向行波 sanity / r_template_corr）→ 永远不做（PR-6A superseded）
- **EI 完整实现** → PR-T3-2
- **30–100 Hz / 12–127 Hz 之外的 HF band sensitivity** → PR-T3-2
- **Threshold 自动优化（ROC sweep）** → PR-T3-2
- **i / l / e 三层比较** → PR-T3-2
- **CAR vs bipolar montage sensitivity** → PR-T3-2
- **替换 `*_soz_core_channels.json`** → 永远不做
- **Qualitative SOZ-reliability verdict** → 永远不做
- **机制层论断** → 不在本 repo 做
- **HFO-rate-based SOZ proxy（v1.1 M1）** → 已经做过（topic3 PR-1 / PR-2），不重做
- **HFO-band-power log-ratio proxy（v1.1 M2）** → 与 M1 同源，归档于 obsolete
- **Layer A 上跑 SOZ 之外的假设检验**（如 H1 / H1'） → Layer A 是 SOZ-purpose only

---

## 5. Surrogate / null（v2.1 简化）

### 5.1 Time-shifted r_sz null（Layer A 内）

每 subject × 每套 ER 配置：把 seizure onset 在 baseline window 内随机平移 `n_iter = 100` 次（避开真 seizure ± 5 min），重跑 ER + z-score + CUSUM + n_d，得到 shifted r_sz。

`enrichment_true_over_shift_<er> = enrichment(true_r_sz, clinical) / median(enrichment(shifted_r_sz_iter, clinical))`

写入 Layer A per-subject JSON `null_surrogate` 字段。

### 5.2 v2.1 报告口径

- > 1：真 onset r_sz 在 SOZ 标注上 enrichment 超过随机时点
- ≈ 1：真 onset 与随机时点等价 → ER pipeline 没抓到 onset-specific 信号
- < 1：异常 → 写警告，但不 abort（cohort 中位 < 1 的 subject 比例 > 30% 触发 §6.2 cohort failure）

---

## 6. Eligibility / Failure / Abort modes（v2.1）

### 6.1 Subject 级 eligibility

| 条件 | 出 | 备注 |
|---|---|---|
| Subject 不在 audit_eligible (v1.1 已锁) | drop | 缺前置 |
| `n_clinical_matched` < 1 | drop | 无 audit baseline |
| Subject seizures 通过 Layer A 标 `onset_tied` 或 `onset_unreached` 比例 > 70% | drop | r_sz 不可靠 |
| `s_sz_gamma_ER` < 0.3 AND `s_sz_broad_ER` < 0.3 | drop | 两套 ER stability 都不达 |
| `s_sz_gamma_ER` ≥ 0.3 OR `s_sz_broad_ER` ≥ 0.3 | 进 relaxed cohort（per ER config 独立判定） | 至少一套有用 |

### 6.2 Cohort failure → abort（v2.1）

- 两套 ER 配置都没有 ≥ 5 subjects 通过 strict gate (s_sz ≥ 0.5) AND 都没有 ≥ 10 通过 relaxed (s_sz ≥ 0.3) → **Layer A fail，Layer B 不运行**，PR archive 写明
- `enrichment_true_over_shift < 1` 的 subject 比例 > 30% → archive 警告但不 abort（除非同时触发上一条）
- 仅 broad 通过、gamma 不通过 → broad 单独进 Layer B；archive 注明
- 仅 gamma 通过、broad 不通过 → archive 警告 "data-driven SOZ HFO-independence 不充分"，但 Layer B 仍可输出 gamma 标签供下游评估（下游必须知情）

### 6.3 Layer B 启动 prerequisites（v2.1 关键）

Layer B Step B.1 之前必须 check：

- [ ] Layer A 跑完 cohort（all audit_eligible 24 + sentinel）
- [ ] Layer A sentinel sanity PASS（focal channels 显著早于 non-focal at p < 0.1）
- [ ] §6.2 abort 条件未触发
- [ ] Layer A 输出的 per-subject JSON schema 通过 §3.5 schema 校验

如果 Layer A 没 ready / 失败，Layer B Step B.x 全部阻塞；写 archive note 说明阻塞原因，**不**伪造 label JSON。

---

## 7. 实现合同检查清单（v2.1）

- [ ] Layer A 复用 PR-6A `src/ictal_onset_extraction.py` Step 0-2 helpers，**不**改动其源代码
- [ ] Layer A 在新模块 `src/ictal_er_rank.py` 实现 CUSUM + n_d + r_sz + stability + null surrogate
- [ ] Layer A 不引用 / 不调用 PR-6A H1/H1' 相关模块（template alignment / r_template_corr）
- [ ] 两套 ER 配置 (gamma + broad) 全程并行，独立产出独立 r_sz / s_sz / cohort
- [ ] 禁止按 subject 后验切换 ER config 或 stability cohort
- [ ] Layer B 严格按 audit_eligible ∩ Layer A cohort 求交集
- [ ] Channel matching 仍走 v1.1 commit `d0fae20` 的 dual-prefix-aware `annotate_clinical_soz`
- [ ] Top-k 取 size-matched k = |clinical_matched|
- [ ] Overlap audit 用 `len(B)` 计算 random_expected_intersection（v1.1 commit `d0fae20` 的 P1.2 修复合同）
- [ ] 主体不在 Layer A cohort → Layer B 不为该 subject 输出条目（**不**写 None entry，不写 fake label）
- [ ] Layer A / Layer B per-subject / per-cohort JSON schema 锁定（§3.5 / §3.8）
- [ ] 每张 audit 表 / 图都同时出 gamma + broad 两列 + strict + relaxed 两套
- [ ] 不出 verdict 字段（grep `verdict|broadly_consistent|partially_consistent|unreliable|true_soz|ground_truth` 必须返回空）
- [ ] §6.2 honest-failure 真的 honest：abort 触发时 Layer B 不运行、不 fake、不 silent

---

## 8. 文件 / 模块 map（v2.1）

### 8.1 v1.1 资产分阶段处置（**不直接删除**）

| 阶段 | 动作 | 触发条件 |
|---|---|---|
| Phase 1（本 PR Step 0）| v1.1 plan 顶部加 superseded banner；24 JSON 目录改名 obsolete | 立即 |
| Phase 2（本 PR Step 0） | v1.1 helpers 在源码顶部加 obsolete docstring；不在主线调用，但模块文件不动 | 立即 |
| Phase 3（v2.1 跑通后）| v1.1 测试标 `pytest.mark.skip(reason="v1.1 obsolete, kept for evidence")` | Layer B 完成后再做 |
| Phase 4（远期）| 评估是否真的迁出代码 / 删除测试 | 至少经过一次 cohort 复跑验证 v2.1 stable 之后；**本 PR 不做** |

理由：v1.1 资产是"为什么 HFO-rate SOZ 不能做"的实证证据 + 可复查路径。直接删除会切断这条证据链。

### 8.2 保留并复用

- `src/data_driven_soz.py` 中保留：
  - `annotate_clinical_soz`、`matched_clinical_contacts`、`check_channel_schema_consistency`
  - `SOZ_LABEL` / `NON_SOZ_LABEL` / `UNKNOWN_LABEL` 常量
  - `compute_overlap`、`random_expected_jaccard`
  - 标 obsolete（不在主线调用）：`compute_hfo_onset_metrics`、`rank_top_k_per_seizure`、`aggregate_consensus`、`aggregate_median_rank`、`compute_er_logratio`、`estimate_per_channel_eps`、`select_m2_eligible_channels`、`time_shifted_seizure_onsets`、`compute_per_subject_audit`、`_bandpass_power`、`PerChannelEps`、`prefilter_seizures_by_block_window`
- `src/ictal_onset_extraction.py`（PR-6A Step 0-2）保留并被 Layer A 复用
- `scripts/run_data_driven_soz.py` 保留 `--audit` 模式；旧 `--per-subject` / `--cohort-overlap` / `--build-data-driven-soz-labels` 标 obsolete（仍可运行但 print warning，主线走新 CLI）

### 8.3 新增（Layer A）

- `src/ictal_er_rank.py`（新建）：
  - `compute_cusum_n_d(z_er: np.ndarray, lambda_thresh: float, bias: float = 0.5) -> int | None`
  - `calibrate_lambda_per_subject(z_er_baseline: np.ndarray, fpr_target: float = 1/3600) -> float`
  - `rank_channels_by_n_d(n_d_per_channel: Mapping[str, float | None]) -> Dict[str, float]`
  - `compute_seizure_status(n_d_dict, n_channels, detection_window) -> str`  # ok/onset_tied/onset_unreached
  - `compute_per_subject_r_sz(per_seizure_rank_vectors) -> Dict[str, float]`
  - `compute_stability_s_sz(per_seizure_rank_vectors) -> float`
  - `compute_time_shifted_r_sz_null(...) -> Dict[str, np.ndarray]`
  - `compute_per_subject_layer_a(...)` orchestrator → Layer A schema §3.5
- `scripts/run_ictal_er_rank.py`（新建）：
  - `--sentinel`：跑 sentinel sanity (548 + 916)，输出 sanity report
  - `--per-subject`：跑 Layer A cohort，输出 per_subject JSON
  - `--cohort-summary`：写 Layer A cohort_summary.json（per-config stability 分布、abort 检查）

### 8.4 新增（Layer B）

- `src/data_driven_soz_pivot.py`（新建）：
  - `load_layer_a_per_subject(subject, dataset, layer_a_dir) -> Dict`
  - `select_data_driven_soz_topk(rsz: Mapping[str, float], k: int) -> List[str]`
  - `build_subject_label_record(...) -> Dict`
  - `build_data_driven_soz_label_file(...) -> Dict`
  - `compute_per_subject_overlap_table(...) -> Dict`
- `scripts/run_data_driven_soz.py` 加 mode：
  - `--build-labels-from-layer-a`：跑 §3.8 输出 4 份 JSON
  - `--cohort-overlap-v2`：跑 §3.9 audit 数值

### 8.5 新增（下游消费 helper）

- `src/soz_label_loader.py`（新建）：
  - `load_clinical_soz_label(dataset) -> Dict[str, set[str]]`
  - `load_data_driven_soz_label(dataset, er_config, stability_cohort) -> Dict[str, set[str]]`
  - `dual_label_iter(...) -> Iterator[(subject, clinical_set, data_driven_set)]`

---

## 9. Step breakdown（v2.1）

### Step 0 — v1.1 资产 Phase 1+2 归档 + plan 切换

- [ ] **0.1** `mv results/spatial_modulation/data_driven_soz/per_subject results/spatial_modulation/data_driven_soz/per_subject_hfo_rate_obsolete_v1_1`
- [ ] **0.2** `mv results/spatial_modulation/data_driven_soz/per_subject_legacy_full_block results/spatial_modulation/data_driven_soz/per_subject_hfo_rate_obsolete_v1_1_legacy_full_block`
- [ ] **0.3** 在 `docs/archive/topic3/pr_t3_1_data_driven_soz_audit_plan_2026-04-30.md` 顶部加 superseded banner
- [ ] **0.4** 在 v1.1 obsolete helpers (`src/data_driven_soz.py` 内的 M1/M2/loader 一组) 函数 docstring 顶加 `OBSOLETE: superseded by Layer A in PR-T3-1 v2.1`，模块顶部加 banner
- [ ] **0.5** 主线 `scripts/run_data_driven_soz.py` 的 `--per-subject` / `--cohort-overlap` / `--build-data-driven-soz-labels` mode 在 stdout 加 deprecation warning（仍可跑，但提示用 v2.1 新 CLI）
- [ ] **0.6** Commit：`chore(pr-t3-1): v1.1 phase-1 archive — banners, dir rename, obsolete docstrings (no code deletion)`

### Step A — Layer A: ictal ER-rank producer（新建，scope-restricted）

#### Step A.1 — 模块骨架 + TDD（CUSUM + n_d）

- [ ] **A.1.1** 新建 `src/ictal_er_rank.py`，import PR-6A Step 0-2 helpers
- [ ] **A.1.2** TDD：CUSUM 在 known step input 上正确报警（synthetic z-ER 在 t=10 加 +5 step → n_d ≈ 10）
- [ ] **A.1.3** TDD：null input (random noise) → false-positive rate ≤ fpr_target；λ 校准函数收敛
- [ ] **A.1.4** TDD：`rank_channels_by_n_d` 处理 None / NaN tail 同 v1.1 `rank_top_k_per_seizure` 合同
- [ ] **A.1.5** TDD：`compute_seizure_status` 三态判定（ok / onset_tied / onset_unreached）
- [ ] **A.1.6** Commit：`feat(pr-t3-1 layer-a): Step A.1 — CUSUM + n_d + seizure status (TDD N tests)`

#### Step A.2 — r_sz + stability + null surrogate

- [ ] **A.2.1** TDD：`compute_per_subject_r_sz` median rank across qualifying seizures
- [ ] **A.2.2** TDD：`compute_stability_s_sz` median pairwise Spearman on r vectors，处理 NaN tail
- [ ] **A.2.3** TDD：`compute_time_shifted_r_sz_null` shifted onset 在 baseline window 内、避开真 seizure ± 5 min
- [ ] **A.2.4** Commit：`feat(pr-t3-1 layer-a): Step A.2 — r_sz + s_sz + time-shifted null (TDD N tests)`

#### Step A.3 — Sentinel sanity

- [ ] **A.3.1** 在 `scripts/run_ictal_er_rank.py` 写 `--sentinel` mode
- [ ] **A.3.2** 跑 `epilepsiae/548` + `epilepsiae/916`，输出 per-sentinel JSON + sanity report
- [ ] **A.3.3** Sanity 判据：focal channels 的 r_sz 中位数显著早于 non-focal (Wilcoxon p < 0.1)，per ER config
- [ ] **A.3.4** 如果 sanity FAIL：写 archive sentinel report，**不进 Step A.4**
- [ ] **A.3.5** Commit：`feat(pr-t3-1 layer-a): Step A.3 — sentinel sanity report (548 + 916)`

#### Step A.4 — Cohort run + abort gate

- [ ] **A.4.1** 在 `scripts/run_ictal_er_rank.py` 写 `--per-subject`（cohort）
- [ ] **A.4.2** 跑 cohort（audit_eligible 24），写 per_subject JSON
- [ ] **A.4.3** 在 `--cohort-summary` 中检查 §6.2 abort 条件
- [ ] **A.4.4** 如果 abort 触发：archive note + Layer B 永久阻塞
- [ ] **A.4.5** Commit：`feat(pr-t3-1 layer-a): Step A.4 — cohort run + per-subject JSON + abort gate check`

### Step B — Layer B: data-driven SOZ label + overlap audit

**前置（必须先满足）**：Step A.4 cohort 跑完且 §6.2 abort 未触发

#### Step B.1 — Label builder

- [ ] **B.1.1** 新建 `src/data_driven_soz_pivot.py`
- [ ] **B.1.2** TDD：`load_layer_a_per_subject` 字段缺失 raise；非 Layer A 文件 raise
- [ ] **B.1.3** TDD：`select_data_driven_soz_topk` size-matched k + NaN tail
- [ ] **B.1.4** TDD：`build_subject_label_record` schema lock §3.8
- [ ] **B.1.5** TDD：`build_data_driven_soz_label_file` provenance lock + er_config + stability_cohort 字段
- [ ] **B.1.6** TDD：subject 不在 Layer A cohort → 不进 label file（不写 None entry）
- [ ] **B.1.7** TDD：grep forbidden phrase 在 label JSON 中不出现
- [ ] **B.1.8** Commit：`feat(pr-t3-1 layer-b): Step B.1 — label builder (TDD N tests)`

#### Step B.2 — `--build-labels-from-layer-a` CLI

- [ ] **B.2.1** 在 `scripts/run_data_driven_soz.py` 加 `--build-labels-from-layer-a` mode
- [ ] **B.2.2** 4 份 JSON 落盘：gamma_strict / gamma_relaxed / broad_strict / broad_relaxed（前提是各 cohort 有 subject）
- [ ] **B.2.3** Commit：`feat(pr-t3-1 layer-b): Step B.2 — build labels from Layer A r_sz`

#### Step B.3 — `--cohort-overlap-v2` audit + independence audit

- [ ] **B.3.1** 在 `scripts/run_data_driven_soz.py` 加 `--cohort-overlap-v2` mode
- [ ] **B.3.2** Per (ER × stability × k) 计算 overlap 表 + null-corrected enrichment（用 len(B)）
- [ ] **B.3.3** §3.10 independence audit：与 v1.1 obsolete cohort 求 jaccard（gamma vs HFO-rate, broad vs HFO-rate）
- [ ] **B.3.4** 写 cohort_overlap_summary_v2.json
- [ ] **B.3.5** Commit：`feat(pr-t3-1 layer-b): Step B.3 — cohort overlap audit + independence audit (gamma + broad × strict + relaxed)`

### Step C — Visualization + 文档

- [ ] **C.1** 5 张图（每张拆 gamma + broad 两列，叠加 strict / relaxed）：
  1. `H_clinical_vs_data_driven` 散点（per subject，颜色 = stability cohort）
  2. enrichment 直方图（per ER × stability）
  3. clinical vs data_driven top-k Venn 平均
  4. gamma vs broad 一致性（jaccard between two configs' top-k per subject）
  5. independence audit：gamma vs HFO-rate jaccard, broad vs HFO-rate jaccard, true-vs-shifted enrichment
- [ ] **C.2** 写 `docs/archive/topic3/pr_t3_1_pivot_results_<YYYY-MM-DD>.md` 详细数值表 + 图说明
- [ ] **C.3** 在 `docs/topic3_spatial_soz_modulation.md` §7 / §8 加 1-2 段：两套 SOZ 标签现已落盘 + independence audit 结论 + 下游用法
- [ ] **C.4** 在 `docs/topic3_spatial_soz_modulation.md` 历史索引添加 archive 链接
- [ ] **C.5** Commit：`docs(pr-t3-1): Step C — pivot results + visualization + topic3 main doc update`

### Step D — 下游 PR 接口落定

- [ ] **D.1** 新建 `src/soz_label_loader.py`：dual-label loader（§8.5）
- [ ] **D.2** TDD：clinical + data_driven 同 subject 都返回 set[channel]
- [ ] **D.3** TDD：data_driven 文件缺失 → 明确 FileNotFoundError
- [ ] **D.4** 在 `AGENTS.md` "Fast Path" 加："SOZ 标签该用哪个？" 答："两套并报；clinical = `*_soz_core_channels.json`；data-driven = `data_driven_soz_core_channels_<er>_<stability>.json`；避免循环验证用 broad_ER + strict（如 cohort 通过）"
- [ ] **D.5** Commit：`feat(soz_label): dual SOZ label loader for downstream PRs`

### Step E — Doc closeout + v1.1 Phase 3 评估

- [ ] **E.1** Phase 3：v1.1 obsolete tests 标 `pytest.mark.skip`（不删）
- [ ] **E.2** 主文档 §结论 update：当前 cohort 上 PR-T3-1 v2.1 提供的两套 SOZ 标签状态
- [ ] **E.3** v1.1 plan-of-record archive 标"closed by v2.1"
- [ ] **E.4** Commit：`docs(pr-t3-1): closeout v2.1 — pivot complete, dual SOZ label contract documented`

---

## 10. TDD 测试列表（v2.1）

### Layer A 测试（`tests/test_ictal_er_rank.py`）

| # | 测试 | 验证内容 |
|---|---|---|
| A1 | `test_cusum_alarm_on_known_step_input` | synthetic z-ER, t=10 加 +5 step → n_d ∈ [10, 12]；bias=0.5 |
| A2 | `test_cusum_no_false_alarm_on_white_noise` | 30 s baseline z-ER (zero-mean unit-variance) → no alarm with calibrated λ |
| A3 | `test_calibrate_lambda_converges` | calibrate_lambda_per_subject 在足够长 baseline 上收敛；fpr_target = 1/hour |
| A4 | `test_rank_channels_by_n_d_nan_tail` | None / NaN n_d 排在末尾，alphabetical tie-break |
| A5 | `test_seizure_status_ok` | n_d 分散 → "ok" |
| A6 | `test_seizure_status_tied` | >60% 通道 n_d 在 [onset, onset+1s] → "onset_tied" |
| A7 | `test_seizure_status_unreached` | <30% 通道有 valid n_d → "onset_unreached" |
| A8 | `test_compute_per_subject_r_sz_median_rank` | 3 seizures × 4 channels：r_sz = median rank per channel |
| A8b | `test_compute_per_subject_r_sz_excludes_baseline_invalid_seizures` | baseline-invalid seizure（baseline_end - baseline_start < 60 s 或 EEG-aware clip 后 < 60 s 有效样本）必须从 median rank 输入中预先剔除；只剩 ok / onset_tied / onset_unreached 进 r_sz 计算（onset_tied / onset_unreached 仍按 v1.1 plan §3.4 的规则单独剔除）|
| A9 | `test_compute_stability_s_sz_pairwise_spearman` | 高一致性 ranks → s_sz 接近 1；随机 ranks → s_sz 接近 0 |
| A9b | `test_compute_stability_s_sz_excludes_baseline_invalid_seizures` | baseline-invalid seizure 不进 pairwise Spearman 计算（与 A8b 合同对齐）|
| A10 | `test_time_shifted_r_sz_avoids_real_seizures` | shifted onset 与真 seizure 距离 ≥ 5 min |
| A11 | `test_time_shifted_r_sz_inside_baseline_window` | shifted onset 在 baseline 内（不漂出 [-300s, baseline_end]）|
| A12 | `test_layer_a_orchestrator_schema_lock` | per-subject JSON 含全部 §3.5 字段 |
| A13 | `test_layer_a_no_template_alignment_fields` | grep `template_alignment` / `r_template_corr` / `winning_template_id` 不在 Layer A JSON 出现 |
| A14 | `test_layer_a_dual_er_independent` | gamma_ER cohort 决定不影响 broad_ER cohort 决定 |
| A15 | `test_layer_a_drop_below_stability_threshold` | s_sz_gamma=0.29 AND s_sz_broad=0.29 → cohort_assignment=dropped |
| A16 | `test_layer_a_relaxed_cohort_threshold_0_3` | s_sz=0.3 → relaxed； s_sz=0.49 → relaxed； s_sz=0.5 → strict |
| A17 | `test_layer_a_seizure_drop_propagates_to_r_sz` | onset_tied / onset_unreached / baseline_invalid 的 seizure 不进 median rank |

### Layer B 测试（`tests/test_data_driven_soz_pivot.py`）

| # | 测试 | 验证内容 |
|---|---|---|
| B1 | `test_load_layer_a_per_subject_returns_required_fields` | fixture JSON → dict 含 r_sz / stability / channel_names |
| B2 | `test_load_layer_a_per_subject_missing_subject_raises` | 文件不存在 → FileNotFoundError，不要 silent None |
| B3 | `test_load_layer_a_per_subject_missing_required_field_raises` | 字段缺失 → KeyError with field name |
| B4 | `test_select_data_driven_soz_topk_smallest_first` | r_sz = {a:0, b:1, c:2}, k=2 → ["a","b"] |
| B5 | `test_select_data_driven_soz_topk_handles_nan_tail` | r_sz 含 None → 末尾 alphabetical |
| B6 | `test_build_subject_label_record_schema_lock` | 含 (clinical_matched, data_driven, k_primary, rsz_full_rank, stability) |
| B7 | `test_build_data_driven_soz_label_file_provenance` | provenance 含全部 6 字段（layer_a_per_subject_dir / layer_a_sanity_passed / audit_csv / clinical_soz_source / layer_b_module / plan_doc） |
| B8 | `test_subject_not_in_layer_a_cohort_excluded_from_label` | label JSON 不含该 subject 的条目（**不**写 None entry） |
| B9 | `test_no_verdict_keys_in_label_json` | grep forbidden phrase 在 label JSON 中不出现 |
| B10 | `test_overlap_uses_len_B_not_k` (regression of v1.1 P1.2) | 复用 v1.1 `compute_overlap` 测试合同 |
| B11 | `test_gamma_and_broad_label_files_independent` | 两套 ER 配置的 SOZ 标签 set 不被错误共享 |
| B12 | `test_strict_and_relaxed_cohorts_independent` | 同 ER config 下 strict 与 relaxed 的 subject 集合独立判定 |

### 下游 loader 测试（`tests/test_soz_label_loader.py`）

| # | 测试 | 验证内容 |
|---|---|---|
| L1 | `test_dual_label_loader_returns_both_sets` | clinical + data_driven 同 subject 都返回 set[channel] |
| L2 | `test_dual_label_loader_handles_missing_data_driven_label` | data_driven 文件缺失 → FileNotFoundError，提示先跑 PR-T3-1 v2.1 |
| L3 | `test_dual_label_loader_handles_missing_clinical_label` | clinical 文件缺失 → FileNotFoundError |
| L4 | `test_dual_label_iter_yields_intersection_subjects_only` | 只 yield 两套都有标签的 subject |

---

## 11. Out of scope（v2.1 显式写明）

- **PR-6A H1/H1' 主线**（template alignment / Smith 2022 双向行波 sanity / r_template_corr）→ 永远不复活
- **HFO-rate-based SOZ proxy**（v1.1 M1）→ 已经做过，不重做
- **HFO-band-power log-ratio proxy**（v1.1 M2）→ 与 M1 同源，归档于 obsolete
- **EI 完整实现** → PR-T3-2
- **除 gamma + broad 之外的第三套 ER band** → 合同禁止运行时第三套
- **Per-subject 后验切换 ER config 或 stability cohort** → 合同禁止
- **替换 `*_soz_core_channels.json` 文件** → 永远不做
- **三档 verdict** → 永远不做
- **机制层结论** → 不在本 repo 做
- **topic3 下游 SOZ-stratified 分析本身** → topic3 PR-8 v2 / 下游 PR 自己跑
- **Layer A 跑 SOZ 之外的假设检验**（H1 / H1') → 不做

---

## 12. 自检清单（v2.1）

- [ ] §0 四条结构性问题在 v2.1 设计里都有显式应对
- [ ] §1.1 与 PR-6A 的边界明确，**不复活 H1/H1'**
- [ ] §3.3 Layer A 合同明确，scope-restricted
- [ ] §3.6 Layer A honest-failure 明确（不伪造 label）
- [ ] §3.10 independence audit：broad_ER 设为更关键的 sensitivity
- [ ] §6.3 Layer B prerequisites 强制 check Layer A 完成
- [ ] §8.1 v1.1 资产**分阶段处置**，不直接删除
- [ ] §9 Step A / Step B 清晰分离，B 阻塞 A 完成
- [ ] §10 TDD 列表覆盖每层每个新 helper + 主要合同 invariant + forbidden phrase 检查
- [ ] §11 out-of-scope 明确包含 PR-6A H1/H1'

---

## 13. 一句话承诺

PR-T3-1 v2.1 = **新建一个范围受限的 ictal ER-rank producer (Layer A)，复用 PR-6A Step 0-2 已验收的 ER 提取层但不复活其 H1/H1' 主线；产出 r_sz + stability gate 后由 Layer B 消费成 data-driven SOZ 标签 (gamma + broad × strict + relaxed = 4 套)，与 clinical SOZ 做 overlap audit，并诚实报告 data-driven SOZ 与 HFO-rate 的 independence 程度；如果 cohort stability 不达标，Layer B 不运行，PR 诚实归档"目前没有可靠第二套 data-driven SOZ 标签"，绝不伪造。**
