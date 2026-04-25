> **SUPERSEDED 2026-04-25**：本审阅记录已不再驱动 PR-6 主线。PR-6 已 pivot 到“stable template endpoint (source ∪ sink) 解剖锚定 SOZ”，不再继续推进 ER pipeline / CUSUM / `t_ER_onset` 的封板路径。本文档保留作为 pivot 决策的关键证据 — `548` 跨 seizure top10 overlap=0、`916` cross-band ρ=−0.21、early channels 大量落在 `other` 这三条数值是触发 pivot 的实证基础。**正式 plan-of-record**：[`docs/archive/topic1/pr6_template_endpoint_anchoring_plan_2026-04-25.md`](pr6_template_endpoint_anchoring_plan_2026-04-25.md)。Step3-preview 工具层（`detect_er_onset_preview`）维持 preview-only，不被新主线消费。

---

# PR-6A Step0-2 / Step3-preview 审阅与验收记录（2026-04-23）

> 性质：archive / 阶段性审阅与验收记录
> 触发：完成 PR-6A baseline-fix、sentinel Step2 重绘、以及 Step3-preview `t_ER_onset` 预览后，用户要求先审阅、验收、记录科学结论与关键问题，再决定后续主线
> 上游入口：`docs/topic1_within_event_dynamics.md` §2 / §5 / §7
> 相关合同：`docs/archive/topic1/pr6a_template_ictal_alignment_plan_2026-04-21.md`

---

## 1. 一句话结论

- **Step0-2：有条件验收。** EEG-aware baseline clip、`<60s -> baseline-invalid` 不回退、sentinel 图重构这三件事都做对了；`548` 的 baseline 污染确实下降，图也比旧版可读。
- **Step3-preview：只接受为 preview，不接受为正式结果层。** 它现在足以回答“ER 能不能粗略给出 clinical-onset 前的招募顺序预览”，但还不够资格进入 H1/H1' 或 sanity 主叙事。

---

## 2. 本次验收范围

### 2.1 Step0-2（已审阅）

- `src/ictal_onset_extraction.py`
  - EEG-aware baseline：`baseline_end_sec = min(0, eeg_onset_rel_sec) - 60s`
  - `<60s` baseline-invalid，不回退 legacy `[−300, −60]`
- `tests/test_pr6a_ictal_onset.py`
  - baseline clip / invalid / preview 原语单测
- `scripts/sentinel_pr6a_step2.py`
  - raw panel + 单一 `cluster1` 风格 ER panel + heatmap
  - `[-200, 200]s` 可视化
  - `High-HI index` / `High-HI ∩ ictal` 语义层
- `results/interictal_propagation/ictal_alignment/_sentinel_step2/`
  - 548/916 sentinel 图与 `sentinel_step2_summary.json`

### 2.2 Step3-preview（已审阅）

- `src/ictal_onset_extraction.py`
  - `resolve_detection_window()`
  - `detect_er_onset_preview()`
  - `preview_threshold_from_baseline()`
- `scripts/sentinel_t_er_onset_preview.py`
  - 对 `548/916`、`gamma_ER+broad_ER` 导出 per-channel `t_ER_onset`
- `results/seizure_onset/er_onset_preview/`
  - `sentinel_t_er_onset_preview.csv`
  - `sentinel_t_er_onset_preview_summary.json`

---

## 3. Step0-2 审阅意见与验收

### 3.1 通过点

1. **baseline 污染修复是实质改进，不是 cosmetic。**
   - `epilepsiae/548` seizure 1：baseline end 从旧版固定 `-60s` 前移到 `-106.9s`
   - `epilepsiae/548` seizure 2：baseline end 前移到 `-146.2s`
   - 这两次都明确避开了 EEG onset 前 60s 的污染区，符合用户确认的合同

2. **548 的 z-ER 可读性确实改善。**
   - `548 / seizure 1 / gamma_ER`：
     - `focal_zER_pre30s_max_median = -0.03`
     - `focal_zER_post30s_max_median = 0.60`
   - 这至少说明 baseline 没再把 focal pre30 段整体抬高成“伪发作态”

3. **新图层次是合理的。**
   - raw waveform panel 解决“临床上看不见 onset 形态”的问题
   - 单一 `cluster1` 风格 ER panel 避免 `cluster0/cluster1` 双 panel 干扰
   - heatmap 分层改成 `High-HI ∩ ictal / High-HI index / ictal only / other`，语义比旧版清楚

### 3.2 保留问题（未阻断 Step0-2 验收）

**`IED peak exclusion` 还没真正落地。**

`baseline_zscore_er()` 虽然有 `exclude_peaks` 接口，但当前 `sentinel_pr6a_step2.py` 与 `sentinel_t_er_onset_preview.py` 都没有传任何 peak mask。  
这意味着当前 Step0-2 只能被称为：

> “EEG-aware baseline clip + no-fallback enforcement 已实现”

不能被称为：

> “§3.2 baseline contract 完整实现”

因为 §3.2 里写的 baseline 内已知 IED 排除，目前还只是接口，不是行为。

### 3.3 Step0-2 验收结论

**结论：`有条件通过 / ACCEPT WITH CAVEAT`**

- 接受的部分：
  - EEG-aware baseline contract（动态 baseline end）
  - baseline-invalid 不回退
  - sentinel Step2 新图
- 未完成但不阻断当前验收的部分：
  - baseline 内 IED peak exclusion

---

## 4. Step3-preview 数值结果与科学判断

### 4.1 这版 preview 修掉了什么

第一版 preview 用固定阈值 `5.0`，结果几乎 `84/84` 通道都被检出，纯属垃圾。  
这次改成：

- 每通道自己的 baseline peak CUSUM 作为阈值基线
- preview threshold = `baseline_peak + margin`

之后结果至少不再是全通道假阳性：

| subject | seizure | gamma preclinical detected | broad preclinical detected |
|---|---:|---:|---:|
| `epilepsiae/548` | 1 | 23 | 34 |
| `epilepsiae/548` | 2 | 27 | 42 |
| `epilepsiae/916` | 0 | 16 | 29 |
| `epilepsiae/916` | 1 | 42 | 31 |

### 4.2 548：有中等一致性，但远没到“稳健”

#### 548 / seizure 1

- `gamma_ER` preclinical top5：
  - `TBLA3 (-100.0, ictal)`
  - `HL2 (-90.5, other)`
  - `HL3 (-88.9, other)`
  - `TBRC2 (-58.8, other)`
  - `HL4 (-50.9, other)`
- `broad_ER` preclinical top5：
  - `HL9 (-100.0, high_hi_ictal)`
  - `HL3 (-95.1, other)`
  - `GC6 (-94.3, other)`
  - `GC4 (-91.5, other)`
  - `GC5 (-91.3, other)`
- 同一次 seizure 内，gamma vs broad：
  - `n_common_detected = 20`
  - `spearman_rho = 0.74`

#### 548 / seizure 2

- `gamma_ER` preclinical top5：
  - `GD2 (-111.2, high_hi_index)`
  - `GC6 (-108.4, other)`
  - `GD5 (-105.9, other)`
  - `GB4 (-83.8, high_hi_index)`
  - `TLRA4 (-73.3, other)`
- `broad_ER` preclinical top5：
  - `GC6 (-114.6, other)`
  - `GB6 (-111.5, other)`
  - `TBLC2 (-110.6, other)`
  - `GA6 (-109.3, other)`
  - `GD2 (-107.8, high_hi_index)`
- 同一次 seizure 内，gamma vs broad：
  - `n_common_detected = 24`
  - `spearman_rho = 0.67`

#### 对 548 的科学判断

- **优点**：同一次 seizure 内，gamma/broad 排序有中等一致性（`rho ~ 0.67–0.74`）
- **缺点**：跨 seizure 的 top10 overlap 极弱
  - `gamma_ER` 两次 seizure top10 overlap = `[]`
  - `broad_ER` 两次 overlap 也只剩 `GB6`, `GC6`

**结论**：`548` 说明“ER 也许能给出某种 clinical 前招募顺序”，但还不能说“这个顺序在同一 subject 跨 seizure 稳定”。**

### 4.3 916：比 548 更不稳

#### 916 / seizure 0

- gamma vs broad：
  - `n_common_detected = 8`
  - `spearman_rho = 0.64`
- 但最早 top5 通道几乎全是 `other`

#### 916 / seizure 1

- gamma vs broad：
  - `n_common_detected = 23`
  - `spearman_rho = -0.21`
- 这已经不是“稍有差异”，而是**同一次 seizure 的双 band 顺序本身互相打架**

#### 对 916 的科学判断

`916` 的 preview 更像：

> ER 在这个 subject 上能扫出很多 preclinical crossing，但这些 crossing 更像背景/非核心通道漂移，而不是一个可信的招募几何。

### 4.4 这版 preview 的核心问题

最早被检出的通道里，`other` 比例仍然过高。

例如 `548`：

- `548 / seizure 1 / gamma_ER` preclinical 检出 23 个通道，其中
  - `ictal = 1`
  - `high_hi_ictal = 1`
  - `high_hi_index = 2`
  - `other = 19`
- `548 / seizure 2 / gamma_ER`
  - `high_hi_index = 2`
  - `other = 25`

这直接说明：**当前 preview 表不能被当作“正式 onset rank”读。**

### 4.5 Step3-preview 验收结论

**结论：`BLOCKED AS FORMAL RESULT / ACCEPT AS PREVIEW-ONLY`**

接受的部分：
- 它现在足以回答一个工程问题：  
  **“ER 在 sentinel 上能不能产出一个不完全荒谬的 clinical 前时序预览？”**
- 这次答案是：**可以，尤其在 548 上有一定信号。**

不接受为正式结果层的原因：
1. 最早通道过多落在 `other`
2. 916 的跨 band 顺序不稳（甚至反相关）
3. 还没有 tie/unreached 正式 gate
4. 还没有 per-subject permutation-calibrated `λ`
5. 还没有 baseline IED exclusion

---

## 5. 当前最诚实的科学结论

1. **ER 值得继续用来追“clinical onset 前的 electrographic recruitment 顺序”，但目前只配叫 preview，不配叫正式 onset rank。**
2. **`gamma_ER` 与 `broad_ER` 都该保留。**
   - `broad_ER` 作为 Bartolomei 2008 经典 EI 风格的 sensitivity arm，应尊重
   - `gamma_ER` 更贴近我们当前想看的 HFO/LVFA recruitment 问题
3. **`548` 比 `916` 更值得继续做 Step 3。**
   - `548` 同一次 seizure 内 gamma/broad 相关中等偏高
   - `916` 的排序稳定性不足，不能拿来做强叙事
4. **这次 baseline fix 真修到了一个真实问题，但没修到“正式发作时序已可提取”。**
   - 它解决了 baseline contamination
   - 但没解决“哪些 early crossings 才是真正可信的早起 recruitment”

---

## 6. 后续阻塞点 / 推荐下一步

### 6.1 P0（阻塞正式 Step 3）

1. **把 baseline 内 IED peak exclusion 真接上**
2. **正式 Step 3 的 `λ` 校准要按合同实现**
   - baseline false-alarm `< 1/hour`
3. **给 preview / Step 3 都加防误读护栏**
   - earliest detections 若主要落在 `other`
   - 或 gamma/broad 顺序强冲突
   - summary 必须显式标 `warning/fail`

### 6.2 P1（最值得先做）

先做一版**受限通道表**：

- 只保留 `high_hi_ictal / high_hi_index / ictal`
- 先不让 `other` 参与排序主表

原因很简单：  
如果把 `other` 继续混进主表，我们得到的会更像“谁先漂起来”，不是“谁先招募”。

---

## 7. 本次审阅后的正式口径

### 7.1 Step0-2

**正式口径：**

> PR-6A Step0-2 已完成 EEG-aware baseline clip、baseline-invalid no-fallback、以及 sentinel Step2 图重构；可作为后续 Step 3 的数据准备层验收通过。  
> 但 baseline 内 IED exclusion 仍未接通，因此当前实现是 §3.2 的“主体通过版”，不是“完全闭环版”。

### 7.2 Step3-preview

**正式口径：**

> PR-6A Step3-preview 目前只接受为 preview-only 工具层：它足以帮助判断 ER 是否值得继续用于 clinical 前招募顺序，但不进入正式 H1/H1' / sanity 叙事，也不能被视为已验收的 onset-rank 提取层。

---

## 8. 配套结果路径

- Step2 sentinel 图：
  - `results/interictal_propagation/ictal_alignment/_sentinel_step2/`
- Step3-preview 表：
  - `results/seizure_onset/er_onset_preview/sentinel_t_er_onset_preview.csv`
  - `results/seizure_onset/er_onset_preview/sentinel_t_er_onset_preview_summary.json`

