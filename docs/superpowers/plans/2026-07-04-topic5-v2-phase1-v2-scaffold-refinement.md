# Topic 5 V2 Phase-1-v2 — Candidate Scaffold Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 回答 Phase-1 已验收 candidate scaffold **能否被科学解释**的三个必要问题——**survive?**（Gate B/C 后是否仍在；是 broadband/common-field 还是 residual frequency-specific layer）· **who?**（哪些 subject 真有；cohort effect 是否少数 subject 驱动）· **when?**（preictal 已存在还是 onset 后才出现）。**这不是 optional「补充探索」，是解释 Phase-1 结果的必要下一步**——Phase-1 目前最薄弱三处：时间锚不稳（只 onset 后 0–20s、clinical onset 可能不准）、subject 异质不清（cohort 6/7 是聚合、narrow 中位 2/7）、broadband/1-f 混杂未排（Gate B/C 未跑）。

**Architecture:** 3 workstream，大量复用 Phase-1 infra（residual cache builders、cohort-perm null/gate、resumable+setsid runner、alignment `--feature`、band cache 已覆盖 onset−130s）。W2 纯后处理（不跑新 null），W1/W3 跑数据。

**Tech Stack:** Python / numpy / pandas / pyarrow / matplotlib；复用 `src/topic5_v2_band_scan.py` + `scripts/run_topic5_v2_{alignment,nulls,gates}.py` + `scripts/build_topic5_v2_{common_resid_cache,aperiodic_cache,confound_maps}.py`。

## Global Constraints（每个 task 隐含包含）

- **Tier LOCK（pre-registered）**：exploratory **candidate scaffold refinement**（**不是 closure**——formal within-shaft Gate A 仍 unresolved，本 phase 不解 SEEG 杆几何 2/20 硬限制）。**可升级的只是修饰语**：`raw-only` → `residual-robust`（W1）→ `preictal-present` / `onset-triggered`（W3）→ `subject-heterogeneous`（W2）candidate scaffold。**不能**升级成：formal spatial-null positive · HFO-/LVFA-/ripple-specific · timing-order replay · criticality-proven · 机制。
- **统计单位 = subject**（永不把 window 当独立样本；window→seizure→subject→cohort）。承重定性主张 → 数值阈值 gate（feedback_acceptance_gate）。
- **null n_perm**：dev 100 → final 1000；**必须用 Phase-1 resumable + setsid 基础设施**（`_partial_{feature}/` checkpoint + `setsid bash launcher`，抗 teardown）。
- **队列**：Phase-1 全 20（narrow）/ 17（broad）；yuquan `anchor=eeg_onset` 保留标注。
- **统计口径**：复用 §2 subject-level cohort permutation of the median + max-over-bands（**FWER family = 7 primary bands**）。

## Workstream 优先级 + 执行顺序

| WS | 优先级 | 回答 |
|---|---|---|
| **W1 Gate B/C** | **P0** | scaffold 是 broadband/common-field 还是 residual frequency-specific layer |
| **W2 subject phenotype** | **P0** | cohort effect 是否少数 subject 驱动、有无频带梯度 phenotype |
| **W3 peri-ictal timing** | **P0/P1** | scaffold 是 static/preictal 还是 onset-triggered |

**工程执行顺序（资源有限时）**：`W2 → W3-dev100/raw → W1-full1000 → W3-final`（W2 最快暴露异质、不跑新 null；W3-raw 快答 preictal 是否已有；W1 最重最承重）。**科学写作里三者并列**，不因执行顺序改重要性。

---

## Workstream 1 — Gate B/C（P0：scaffold 表达层是 broadband 还是 residual frequency-specific）

**为什么**：Phase-1 band-generic + ripple_high 不过 FWER，最自然的替代解释 = shared broadband recruitment / 1-f aperiodic shift / common spatial field / HFO-rate·SOZ·contact-density confound。**Gate B/C 不是锦上添花，是决定 scaffold 该怎么命名的承重结果。**

**判读框架（LOCK；Gate B/C 意义 = 表达层归属，非简单真/假）**：

| Gate B/C 结果 | 正确解释（accepted wording） |
|---|---|
| raw 有、common_resid 后消失 | scaffold 的表达层 = **broadband/common-field recruitment**（"consistent with shared broadband/common-field account"；**不写**"证明就是 broadband"）。**scaffold 不假**，只是表达层是共享宽带招募、非频带特异 residual |
| raw 有、common_resid 后某些 band 存活 | scaffold 上有 **frequency-specific residual layer** |
| aperiodic residual 后 ripple 存活 | **才能讨论** ripple/HFO-specific residual evidence（且须先过 §Task 1.4 sanity） |
| aperiodic residual 后 ripple 不存活 | 不支持 HFO-frequency-specific expression |

### Task 1.1: 建 20 队列 common_resid + aperiodic residual cache（+ aperiodic QC）

**Files:** Run `scripts/build_topic5_v2_common_resid_cache.py`, `scripts/build_topic5_v2_aperiodic_cache.py`（复用；喂全 20）。Output: `results/.../v2_band_scan/{common_resid_cache,aperiodic_resid_cache}/*.npz`。

- [ ] **Step 1**: 确认两 builder 能喂全 20（含 yuquan）；显式传 narrow-20 / broad-17 list。
- [ ] **Step 2**: `build_topic5_v2_common_resid_cache.py --subjects <20>`（setsid）。
- [ ] **Step 3（aperiodic QC，新增）**: aperiodic builder 每 (subject,band,seizure) 输出 QC 字段 `{fit_r2, fit_error, n_valid_freq_bins, line_noise_bins_excluded, fraction_failed_fits, residual_variance_by_band}`，汇总到 `aperiodic_qc.json`。
- [ ] **Step 4**: `build_topic5_v2_aperiodic_cache.py --subjects <20>`（setsid，慢，resumable 化）。
- [ ] **Step 5**: 验证两 cache 各 20、keys 与 band cache 一致。

**判据（LOCK）**：`fraction_failed_fits < 0.2`（cohort）；`median_fit_r2` 必报；line-noise 谐波已排除；residual cache shape == raw cache。**若 1/f fit failure 高（≥0.2）→ Gate C 只能 descriptive，不能作强否定**。

### Task 1.2: 残差 feature alignment + cohort-perm null（narrow+broad，1000）

**Files:** Run `run_topic5_v2_alignment.py --feature {common_resid,aperiodic_resid}` + `run_topic5_v2_nulls.py --feature ... --n-perm 1000`（复用，resumable）。New: `scripts/run_topic5_v2_residual_chain.sh`（setsid launcher）。

- [ ] **Step 1**: 写 launcher：`feature ∈ {common_resid, aperiodic_resid} × substrate ∈ {narrow, broad}` 各 `alignment → nulls --n-perm 1000`。
- [ ] **Step 2**: `setsid bash run_topic5_v2_residual_chain.sh > log 2>&1 < /dev/null &`（resumable）。
- [ ] **Step 3**: 监控 checkpoint；4 组跑完。

**判据**：4 组 residual null summary + perm-long 齐（n_perm=1000）。

### Task 1.3: Gate B/C decision（按上表框架）

- [ ] **Step 1**: 重跑 narrow+broad gate（residual summary 存在，gate_B/C 不再 NaN）。
- [ ] **Step 2**: 统计 `n_primary_bands_surviving_common_resid_FWER` + ripple 带 aperiodic residual 是否存活。
- [ ] **Step 3（accepted wording，LOCK，语言已降级）**：
  - **Outcome A（最可能）**：common_resid 后 0–1/7 primary 存活 → 写 **"The Phase-1 band-generic scaffold is not robust to common-field / broadband residualization, consistent with a shared broadband recruitment or common spatial field account."**（tier → `broadband/common-field candidate scaffold`；**不写**"证明主要是 broadband"）。
  - **Outcome B**：≥2/7 primary 以某模式存活 → "shared scaffold + frequency-specific residual layer"（`residual-robust candidate scaffold`）。
  - **Outcome C**：ripple 带 aperiodic residual 后仍显著 → **先过 Task 1.4 sanity**，才可讨论 ripple/HFO-specific residual evidence（与 Phase-1 "ripple_high 最弱" 矛盾，需交叉核对）。
- [ ] **Step 4**: 写进 archive doc（"Gate B/C closure" 段 + tier 更新，不越 candidate scaffold refinement）。

### Task 1.4: Outcome C sanity（若 ripple 残差后突然显著）+ 单-covariate confound

- [ ] **Step 1（Outcome C sanity gate，LOCK）**：若 ripple 残差后显著，必查 `{raw vs residual sign, effect size, line-noise, aperiodic fit QC (Task 1.1), subject drivers, 512Hz subjects handling}`；任一异常 → 只 descriptive，不写 HFO-specific。
- [ ] **Step 2**: `build_topic5_v2_confound_maps.py --subjects <20>`（HFO-rate / baseline-power / shaft-position）。
- [ ] **Step 3**: `run_topic5_v2_nulls.py --confound-null`（**单-covariate 为主，不塞 combined 大模型**，n contacts 小）；写 descriptive。

---

## Workstream 2 — Subject-level phenotyping（P0：cohort effect 是否少数驱动 + 频带梯度 phenotype）

**为什么**：cohort 6/7 是聚合，per-subject 弱（narrow 中位 2/7、≥5/7 仅 3/20）。**cohort-level band-generic positivity ≠ 每个 subject 都有稳定多频带 scaffold。** 纯读现有 artifact，不跑 null/仿真。

### Task 2.1: 连续 subject profile artifact（不止 n_sig）

**Files:** Create `scripts/analyze_topic5_v2_subject_phenotype.py`. Output `results/.../v2_band_scan/phase1_v2_subject_phenotype.csv`. Test `tests/test_topic5_v2_phase1_v2.py`.

**为什么要连续 profile**：某些 subject 只 2/7 显著但 effect size 大 / 很多 band δ>0 但 seizure 少导致 p 不显著 / 低频强高频弱 / HFA 强低频弱 / 所有 band 弱但方向一致。**pass/fail 单标签会漏掉这些。**

- [ ] **Step 1（TDD）**: 写测试 `test_subject_profile_features`：合成 subject（7 band delta 已知）→ 断言 `low_band_score / LVFA_band_score / HFA_ripple_score / HF_minus_low / band_genericity_index / ripple_rank / n_positive_delta_7bands` 计算正确。
- [ ] **Step 2**: 跑 → FAIL。
- [ ] **Step 3**: 实现 `subject_profile(null_df, align_df, primary_bands)` 输出（每 subject × substrate）：

  ```python
  SUBJECT_PROFILE_FEATURES = [
      "mean_delta_7bands", "median_delta_7bands",
      "n_positive_delta_7bands",          # δ>0 的带数 (of 7)
      "n_sig_7bands",                     # δ>0 & self-null p<0.05 的带数
      "band_genericity_index",            # = n_positive_delta_7bands / 7
      "low_band_score",                   # median(delta for 1-13Hz: δ θ α)
      "LVFA_band_score",                  # median(delta for 13-80Hz: β γ)
      "HFA_ripple_score",                 # median(delta for 80-250Hz: hg R)
      "HF_minus_low",                     # HFA_ripple_score - low_band_score
      "ripple_rank",                      # rank of ripple_high delta among 7 bands
      "profile_entropy",                  # 7-band delta 分布的 Shannon entropy(归一化正部分)
      "within_subject_seizure_consistency",  # per-seizure maxAB 的 1 - IQR/median (稳定性)
      # + n_sz, n_contacts, maxab_primary, spatial_strength, order_strength, anchor
  ]
  ```
- [ ] **Step 4**: 跑 → PASS。
- [ ] **Step 5**: CLI 输出 phenotype csv（narrow+broad）。
- [ ] **Step 6**: Commit + 写进 archive doc（连续 profile 表 + "cohort 6/7 是聚合、per-subject 弱一致"）。

**判据（LOCK）**：报告必须写 `median_n_sig` 与 `frac_ge4of7`；**禁**"每个 subject 都稳定"。bad-data 回归基线：narrow median 2/7、≥4/7=6/20；broad median 3/7、≥4/7=7/17。

### Task 2.2: 三档 subject label（不止 ≥4/7）

**为什么**：subject-level p 很受 seizure 数影响（2–3 次发作难有多带 p<0.05，但 7/7 δ>0 也说明方向一致）。

- [ ] **Step 1**: 三档 label（LOCK）：

  ```python
  strong subject:      n_sig >= 4/7
  directional subject: n_positive_delta >= 5/7  AND  n_sig < 4/7
  weak/absent subject: n_positive_delta < 5/7
  ```
- [ ] **Step 2**: 报每档名单 + 特征（strength / n_sz / n_con / maxAB / anchor / SOZ 覆盖若可得）。
- [ ] **Step 3**: 跨两池都稳的 subject（Phase-1 初查 = 1146/1150/384）+ 唯一 within_shaft_strong+多带阳（1146）作个例，**n=1 非 cohort**。
- [ ] **Step 4**: Commit。

### Task 2.3: Phenotype hunt + 频带 profile subgroup（descriptive）

- [ ] **Step 1**: n_sig（或 band_genericity_index）vs 各 feature 算 Spearman；**判据**：`|r|>0.4 & p<0.05` 才写 "该 feature 预测 multi-band positivity"，否则 **"no single clean phenotype"**（预期）。
- [ ] **Step 2**: 按 low/LVFA/HFA score 描述 subgroup（low-frequency scaffold / LVFA-HFA-leaning / flat band-generic / weak-absent）——**只描述性，不做 KMeans 硬 subtype 显著性主张**（n 小，feedback_silhouette_threshold）。
- [ ] **Step 3**: Commit + 图（遵 §图规范）。

---

## Workstream 3 — Peri-ictal timing（P0/P1：static/preictal 还是 onset-triggered；EEG onset 主锚）

**为什么**：Phase-1 只 onset 后 0–20s + clinical onset 可能不准 → 无法区分 scaffold 是发作点着后才现还是 pre-ictal 已存在。数据已覆盖 onset−130s（band cache pre=130）。

**⚠️ 锚点（LOCK，主锚 = EEG onset，非 clinical）**：

```python
PRIMARY_ANCHOR     = "eeg_onset"    # 临界性/preictal/early-ictal 是电生理问题
SENSITIVITY_ANCHOR = "clin_onset"   # 临床表现时间；可能滞后/模糊真正电生理起始
```
理由：若某些 seizure 的 electrographic onset 早于 clinical onset，用 clinical 对齐会让 "pre-ictal −100s" 已含真正 early ictal、"onset+0–20s" 已是 spread/clinical expression。**报告主结论锚 EEG onset；clinical onset 只作 sensitivity。若两锚结论不一致，以 EEG-onset 作生理主结论。**

### Task 3.1: Peri-ictal epoch grid（−100→+20s）+ 硬 pre-window gate

**Files:** Create `config/topic5_v2_phase1_v2_periictal.yaml`（`epoch.main_rel = [-100, 20]`）。Modify alignment `_epoch_grid` / window loop。

**为什么**：pre-ictal 窗 ictal_fraction=0 是合法 pre-ictal，**不能被 ictal_fraction_min 全滤掉**（否则隐蔽 bug：以为跑了 −100→+20，实际所有 pre 窗被过滤没了）。

- [ ] **Step 1（TDD，硬 gate）**: 写测试 `test_periictal_grid_pre_windows_survive`：epoch main_rel=[-100,20]、window=10、step=5 → 断言：

  ```python
  assert n_pre_windows_per_seizure > 0
  assert min(win_center_rel) <= -90
  assert all(w.epoch_region == "pre" for w in pre_windows)   # 有 epoch_region 标签
  assert pre_windows_not_filtered_by_ictal_fraction          # pre 窗豁免 ictal_fraction_min
  ```
- [ ] **Step 2**: 跑 → FAIL。
- [ ] **Step 3**: 改 `_epoch_grid`/loop：pre-ictal 窗（win_end_rel≤0）**豁免 ictal_fraction_min**（本就在发作前），只对 onset 后窗要求 ictal_fraction≥0.5；每窗加 `epoch_region ∈ {far_pre, mid_pre, near_pre, peri_onset, early_post}` 标签。
- [ ] **Step 4**: 跑 → PASS。
- [ ] **Step 5**: 全 20 跑 window-level alignment（−100→+20s，raw feature，**EEG onset 锚**），输出 win_center_rel + epoch_region。

**time bins（LOCK）**：

```python
TIME_BINS = {"far_pre": (-100,-60), "mid_pre": (-60,-30), "near_pre": (-30,-10),
             "peri_onset": (-10,10), "early_post": (0,20)}
```

### Task 3.2: band-generic scaffold-score 轨迹 + subject-level 检验

**为什么主 endpoint 用 band-generic score**：Phase-1 核心结果已是 band-generic 非 ripple-specific；W3 逐频带做 7 endpoint 会把问题搞散。

**Files:** Create `scripts/run_topic5_v2_trajectory.py`. Output `phase1_v2_alignment_trajectory.csv` + 图.

- [ ] **Step 1**: primary endpoint（LOCK）：

  ```python
  band_generic_scaffold_score = median alignment across the 7 primary bands   # per window
  ```
  per-band 轨迹作 **secondary/descriptive**。
- [ ] **Step 2（统计单位 = subject，LOCK）**：

  ```python
  for subject:
      for seizure:
          bin_score[bin] = median(band_generic_score over windows in bin)   # window→seizure
      subject_bin[bin] = median over seizures(bin_score[bin])               # seizure→subject
  cohort_bin[bin] = median over subjects(subject_bin[bin])
  ```
- [ ] **Step 3（TDD）**: 写测试 `test_trajectory_contrasts`：合成轨迹（pre≈0.3、post≈0.7）→ 断言 `post_minus_far_pre / near_pre_minus_far_pre / post_minus_near_pre` 计算与方向正确。
- [ ] **Step 4**: 实现 3 个 contrast（per subject）：

  ```python
  near_pre_minus_far_pre = subject_bin[near_pre] - subject_bin[far_pre]
  post_minus_far_pre     = subject_bin[early_post] - subject_bin[far_pre]
  post_minus_near_pre    = subject_bin[early_post] - subject_bin[near_pre]
  ```
- [ ] **Step 5（主检验 = subject-level sign-flip，LOCK）**：

  ```python
  p = subject_level_signflip_test(subject_effect)   # 对 subject 配对差做 sign-flip permutation
  ```
  **不用 window-label shuffle 作主 p**（window 强自相关 → p 虚高）；window shuffle 只 descriptive。
- [ ] **Step 6（解释表，LOCK）**：

  | 轨迹形状 | 解释 |
  |---|---|
  | far_pre 低、near_pre 升、post 最高 | preictal loading / 接近临界阈值动态 |
  | far_pre ≈ near_pre ≈ post 都高 | 静态病理 scaffold / anatomy-like susceptibility field |
  | pre 低、post 突然高 | onset-triggered recruitment scaffold |
  | pre 高、post 不升/下降 | scaffold 是 preictal state，不一定是发作招募 |
  | clinical 有 rise、EEG 无 rise | clinical onset 滞后导致假 rise |
  | EEG 有 rise、clinical 无 rise | 真动力学变化在电生理 onset 附近、clinical anchor 被模糊 |
- [ ] **Step 7（双锚 sensitivity）**: epilepsiae 用 EEG onset（主）vs clin onset（sensitivity）各跑；**两锚不一致 → 以 EEG-onset 作生理主结论**，clinical 只解释临床时间对齐敏感性。
- [ ] **Step 8**: Commit + 主图（EEG onset 相对时间 × band_generic_score，cohort median ± subject IQR；per-band 轨迹放 supplementary）。

---

## Supported conclusions（phase1-v2 完成后，最多支持这几种口径）

- **情况 A（Gate B/C 后消失 + preictal 已有 + subject 异质，最可能）**：
  > interictal HFO geometry identifies a broad, trait-like pathological spatial scaffold that overlaps with broadband/common-field recruitment, but the effect is heterogeneous and not frequency-specific.
- **情况 B（Gate B/C 后部分频带存活 + preictal 有上升）**：
  > interictal HFO geometry identifies a candidate susceptibility scaffold with a residual frequency-specific layer and possible preictal loading.
- **情况 C（Gate B/C 后存活 + subject 清晰 subgroup + pre→onset 稳定 rise）**：
  > HFO-derived scaffold is subject-specific and state-dependent, making it a plausible candidate pathological mode for later model–data criticality tests.
  （**即便 C 也不能说"临界模态已证明"**——那要 Phase-2 criticality：variance/AR1/tau/lambda_max/leading mode/branching ratio 沿 G_HFO 加载。）

## Non-goals（明确禁止）

- 不追 formal within-shaft Gate A（SEEG 杆几何 2/20 硬限制，本 phase 不解）。
- 不做 HFO/LVFA/ripple 机制、timing-order replay、criticality 证明。
- 不做 KMeans 硬 subtype 显著性主张（n 小；只描述性 subgroup）。
- 不建新仿真 / 不碰 Phase-2 criticality 或 V3a/V3p（各自独立 plan）。
- 不把 min3 sensitivity 升为 primary。
