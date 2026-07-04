# Topic 5 V2 Phase-1-v2 — Scaffold closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 把 Phase-1 已验收的 **cohort-level candidate early-ictal spatial recruitment scaffold**（间期 HFO geometry ↔ 发作 onset+0–20s 多频带能量场对齐，6/7 primary 过弱空间 null + order null）往前推一格——判它 (1) 是否只是宽带招募 / 1-f 伪影、(2) 是哪些病人有、(3) 是 onset 后才现还是 pre-ictal 已有——**但保持 exploratory tier，不升机制、不宣称 formal Gate A**。

**Architecture:** 三条 workstream，**大量复用 Phase-1 已建 infra**（residual cache builders、cohort-perm null/gate、resumable+setsid runner、alignment `--feature`）。W1/W3 跑数据（复用管线），W2 纯 artifact 后处理（不跑新仿真/null）。

**Tech Stack:** Python / numpy / pandas / pyarrow；复用 `src/topic5_v2_band_scan.py` 纯函数 + `scripts/run_topic5_v2_{alignment,nulls,gates}.py` + `scripts/build_topic5_v2_{common_resid_cache,aperiodic_cache,confound_maps}.py`。

## Global Constraints（每个 task 隐含包含）

- **Tier LOCK（pre-registered，不可在结果里升级）**：exploratory candidate-scaffold。phase1-v2 只做 **refine**（survive / who / when），**NOT** 升级到 formal within-shaft Gate A 或任何机制。**即使 Gate B/C 全 survive，tier 仍是 "candidate scaffold, residual-robust"**——因 formal within-shaft Gate A 仍被 SEEG 杆几何卡住（2/20 within_shaft_strong，Phase-1 已确认，非本 phase 能解）。
- **统计口径**：复用 §2 subject-level cohort permutation of the median + max-over-bands（**FWER family = 7 primary bands**；composite descriptive）。primary null 强度仍 weakest-wins（弱 null → 结论带 "likely inflated" caveat）。
- **null n_perm**：dev 100 → final 1000。**必须用 Phase-1 的 resumable + setsid 基础设施**（`run_topic5_v2_nulls.py` 的 `_partial_{feature}/` checkpoint + `setsid bash launcher`），否则长跑会被 session teardown 杀（Phase-1 被杀 3 次的教训）。
- **队列**：Phase-1 全 20（narrow）/ 17（broad）；yuquan（xuxinyi/zhangkexuan）保留 `anchor=eeg_onset` 标注，报告里两版中位数（20-mixed / 18-clean）并列。
- **禁止措辞（§EXP，spec `2026-07-02-topic5-v2-phase1b-gate-closure-spec.md`）**：HFO-/LVFA-/ripple-specific · timing-order replay · formal Gate A passed · 超过任何空间随机场 · 任何机制主张。
- **每个承重定性主张 → 数值阈值 gate**（CLAUDE.md feedback_acceptance_gate）：本 plan 每个 Task 的"判据"就是该 gate，spec 正文同步锁参数。

## File Structure

**复用（不重写）：**
- `scripts/build_topic5_v2_common_resid_cache.py` — LOBO common-field residual cache（Task 10b，只 smoke 过 139，未跑 20）。
- `scripts/build_topic5_v2_aperiodic_cache.py` — aperiodic/1-f residual cache（Task 11b，只 smoke 过 139）。
- `scripts/build_topic5_v2_confound_maps.py` — HFO-rate / baseline-power / shaft-position covariate maps。
- `scripts/run_topic5_v2_alignment.py` — 已支持 `--feature {raw,common_resid,aperiodic_resid}` + `--feature-cache-dir`。
- `scripts/run_topic5_v2_nulls.py` — 已支持 `--feature`、resumable checkpoint、`--min-group`、confound。
- `scripts/run_topic5_v2_gates.py` — cohort-perm + primary-family FWER + Gate B/C 判据（`gate_pass_flags` 读 common_resid/aperiodic summary）。

**新建：**
- `scripts/run_topic5_v2_residual_chain.sh` — W1 的 setsid launcher（build residual cache → alignment → null → gate，narrow+broad，resumable）。
- `scripts/analyze_topic5_v2_subject_phenotype.py` — W2 纯后处理（读现有 artifact，输出 per-subject 稳定性 + phenotype + 频带 profile）。
- `config/topic5_v2_phase1_v2_periictal.yaml` — W3 peri-ictal epoch（−100→+20s）覆盖 config。
- `scripts/run_topic5_v2_trajectory.py` — W3 window-level alignment 轨迹。
- `tests/test_topic5_v2_phase1_v2.py` — 新纯函数测试（phenotype 判据、trajectory 判据、peri-ictal epoch grid）。

---

## Workstream 1 — Gate B/C closure（P1，最高 ROI）

**为什么**：Phase-1 的 6/7 band-generic 正面**没排除"这就是一张共享宽带招募场"或"只是 1-f 背景"**。W1 判它是不是伪影。这是把 candidate scaffold 往前推的**最小必要科学门**。

### Task 1.1: 建 20 队列 common_resid + aperiodic residual cache

**Files:**
- Run: `scripts/build_topic5_v2_common_resid_cache.py`, `scripts/build_topic5_v2_aperiodic_cache.py`（复用；只需喂全 20 subjects）
- Output: `results/topic5_ictal_recruitment/v2_band_scan/{common_resid_cache,aperiodic_resid_cache}/*.npz`（应各 20）

**Interfaces:**
- Consumes: Phase-1 的 `v2_band_scan/cache/*.npz`（20 已建）。
- Produces: 残差 cache，keys 与 band cache 同结构（alignment `--feature` 直接读）。

- [ ] **Step 1**: 确认两 builder 的 subject 入口（`--subjects` 或 `SUBJECTS_BY_SUB`）能喂全 20（含 yuquan）；若默认只 13，显式传 20 list（同 Phase-1 alignment 的 narrow-20 / broad-17）。
- [ ] **Step 2**: `python scripts/build_topic5_v2_common_resid_cache.py --subjects <20>`（后台 setsid，~分钟级，非 perm 无需 1000）。
- [ ] **Step 3**: `python scripts/build_topic5_v2_aperiodic_cache.py --subjects <20>`（aperiodic 拟合 fit-once-per-(c,tt)，较慢，setsid+可 resumable 化）。
- [ ] **Step 4**: 验证 `ls .../common_resid_cache/*.npz | wc -l == 20` 且 `aperiodic_resid_cache` 同；抽 1 subject 检查 npz keys 与 band cache 一致（`{band}__zt__{idx}`）。
- [ ] **Step 5**: Commit（只 tracked 的 runner 改动，如加 `--subjects`；results 不 commit）。

**判据（gate）**：两残差 cache 各 20 subject 齐、keys 结构正确。否则不进 Task 1.2。

### Task 1.2: 残差 feature 的 alignment + cohort-perm null（narrow+broad，1000）

**Files:**
- Run: `scripts/run_topic5_v2_alignment.py --feature {common_resid,aperiodic_resid}` + `run_topic5_v2_nulls.py --feature ... --n-perm 1000`（复用，resumable）
- New: `scripts/run_topic5_v2_residual_chain.sh`（setsid launcher，见 Global Constraints）

**Interfaces:**
- Consumes: Task 1.1 残差 cache；Phase-1 geometry（load_context）。
- Produces: `phase1_null_{common_resid,aperiodic_resid}_subject_summary.csv` + `_perm_subject_long.parquet`（narrow+broad）。

- [ ] **Step 1**: 写 `run_topic5_v2_residual_chain.sh`：对 `feature ∈ {common_resid, aperiodic_resid}` × `substrate ∈ {narrow, broad}`，`alignment --feature f` → `nulls --feature f --n-perm 1000` → 无独立 gate（gate 一次读全 feature）。
- [ ] **Step 2**: `setsid bash run_topic5_v2_residual_chain.sh > log 2>&1 < /dev/null &`（Phase-1 launcher 模板；resumable 保证被杀可续）。
- [ ] **Step 3**: 监控 checkpoint（`_partial_{feature}/*.marker.json`）；4 组（2 feature × 2 substrate）跑完。
- [ ] **Step 4**: Commit runner + launcher。

**判据**：4 组 residual null summary + perm-long 齐（n_perm=1000）。

### Task 1.3: Gate B/C decision（3 个 pre-written outcome，先锁）

**Files:**
- Run: `scripts/run_topic5_v2_gates.py --substrate {narrow,broad}`（复用；自动读 common_resid/aperiodic summary → gate_B/C 列）
- Output: `phase1_gate_summary.csv`（gate_B_frequency_specific_pass / gate_C_HFO_specific_pass 列刷新）

- [ ] **Step 1**: 重跑 narrow + broad gate（现在 residual summary 存在，gate_B/C 不再全 NaN）。
- [ ] **Step 2**: 统计 `n_primary_bands_surviving_common_resid_FWER`（gate_B 判据：common_resid_delta>0 且 max_over_bands_p<alpha）。
- [ ] **Step 3**: 统计 ripple 带的 aperiodic residual 是否存活（gate_C）。
- [ ] **Step 4**: 按**运行前锁定的 3 个结局**写结论（spec §5）：

  **判据（LOCK，encode 结论）**：
  - **Outcome A（最可能，broadband-recruitment）**：common_resid 后 **0–1/7 primary 存活** → 写 "band-generic alignment 主要是 shared broadband recruitment；G_HFO 预测的是共享招募场，非频带特异"。**candidate scaffold 降级为 broadband-recruitment readout，但仍是有效 exploratory 正结果**。
  - **Outcome B（frequency layer）**：common_resid 后 **≥2/7 primary 以某频带模式存活** → 写 "shared scaffold + frequency-specific residual layer"（仍不写机制）。
  - **Outcome C（HFO/ripple 存活）**：**ripple 带在 aperiodic residual 后仍显著** → 会与 Phase-1 "ripple_high 最弱" 矛盾，需交叉核对（大概率 Outcome A/B，不是 C）。
- [ ] **Step 5**: Commit gate 结果解读进 archive doc（新增 "Gate B/C closure" 段，标 outcome + tier）。

**判据**：明确落到 A/B/C 之一 + tier 相应更新（不越过 candidate-scaffold/broadband-recruitment）。

### Task 1.4: 单-covariate confound sensitivity（补充）

- [ ] **Step 1**: `build_topic5_v2_confound_maps.py --subjects <20>` 建 HFO-rate / baseline-power / shaft-position maps。
- [ ] **Step 2**: `run_topic5_v2_nulls.py --confound-null`（单 covariate 残差后重测对齐）。
- [ ] **Step 3**: 判据：对齐在扣掉各单 covariate 后是否仍 >0（**单-covariate 为主，不塞 combined 大模型**，n contacts 小）。写进 doc（descriptive）。

---

## Workstream 2 — Subject-level phenotyping（P0 认识论，无新重算）

**为什么**：Phase-1 review 暴露 cohort 6/7 是**聚合**、per-subject 弱（narrow 中位 2/7）。W2 把这条写实，并找"哪些病人有 scaffold"。**纯读现有 artifact，不跑 null/仿真。**

### Task 2.1: 正式 per-subject 稳定性 artifact + doc

**Files:**
- Create: `scripts/analyze_topic5_v2_subject_phenotype.py`
- Output: `results/topic5_ictal_recruitment/v2_band_scan/phase1_v2_subject_phenotype.csv`
- Test: `tests/test_topic5_v2_phase1_v2.py`

**Interfaces:**
- Consumes: `phase1_null_raw_subject_summary.csv`（per-subject spatial delta/p/strength）+ `phase1_alignment_raw_subject_summary.csv`（n_seizures/n_contacts/maxAB）。
- Produces: per-subject `{n_sig, n_deltapos, strength, order, n_sz, n_contacts, maxab, HF_minus_low, anchor}`（narrow+broad）。

- [ ] **Step 1（TDD）**: 写测试 `test_subject_stability_count`：给合成 subject_summary（primary 7 带，3 带 delta>0&p<.05）→ 断言 `n_sig==3`。
- [ ] **Step 2**: 跑测试 → FAIL（函数未定义）。
- [ ] **Step 3**: 实现 `per_subject_stability(null_df, align_df, primary_bands)` → 返回 DataFrame（复现 Phase-1 review 的口径：n_sig = #{delta>0 & empirical_p<0.05}）。
- [ ] **Step 4**: 跑测试 → PASS。
- [ ] **Step 5**: CLI 输出 narrow+broad 的 phenotype csv + 打印 ≥4/7、≥5/7 名单。
- [ ] **Step 6**: Commit + 写进 archive doc（per-subject 表 + "cohort 6/7 是聚合、per-subject 弱一致"定性）。

**判据（LOCK）**：报告必须写 `median_n_sig` 与 `frac_ge4of7`；**禁止**写 "每个 subject 都稳定"。当前值（Phase-1 review 已算）：narrow median 2/7、≥4/7=6/20；broad median 3/7、≥4/7=7/17——作 bad-data 回归基线。

### Task 2.2: Phenotype hunt（feature → n_sig）

- [ ] **Step 1**: 对 n_sig vs 每个 candidate feature（n_contacts / n_sz / spatial_strength tier / raw maxAB / dataset / anchor / SOZ 覆盖若可得）算 Spearman r（pooled + per-substrate）。
- [ ] **Step 2**: **判据（LOCK）**：`|r|>0.4 且 p<0.05` 才写 "该 feature 预测 multi-band positivity"；否则写 **"no single clean phenotype"**（Phase-1 review 初查即无单一 feature；预期 no clean phenotype）。
- [ ] **Step 3**: 报告跨两池都稳的 subject（Phase-1 初查 = 1146/1150/384）+ 唯一 within_shaft_strong+多带阳（1146）作个例，**明确 n=1 非 cohort**。
- [ ] **Step 4**: Commit。

### Task 2.3: Per-subject frequency profile

- [ ] **Step 1**: 每 subject 的 7-band delta 曲线（矩阵：subject × band）。
- [ ] **Step 2**: 描述性判断有无 subgroup（如 low-freq-scaffold vs HF-scaffold）；**judged descriptively，不做 KMeans 硬聚类主张**（n 小、feedback_silhouette_threshold）。
- [ ] **Step 3**: **判据**：只报 "存在/不存在肉眼可辨的 band-profile subgroup" + 每 subgroup subject 名单；不写 cluster 显著性。
- [ ] **Step 4**: Commit + 图（若画，遵 docs/figure_style_guide.md + 生成后 README + 目视）。

---

## Workstream 3 — Peri-ictal 时间扩展（P1，新分析）

**为什么**：Phase-1 只 onset 后 0–20s，无法区分 scaffold 是**发作点着后才现**还是 **pre-ictal 已存在**。数据其实已覆盖 onset−130s（`iter_subject_seizure_windows` pre=130）——**只需放开 epoch grid + 看 window-level 轨迹**。

### Task 3.1: Peri-ictal epoch grid（−100→+20s）+ window-level alignment

**Files:**
- Create: `config/topic5_v2_phase1_v2_periictal.yaml`（覆盖 `epoch.main_rel = [-100, 20]`；其余同 Phase-1）
- Run: `run_topic5_v2_alignment.py`（若 epoch 从 config 读则加 `--config` 覆盖；否则加参）

**Interfaces:**
- Consumes: Phase-1 band cache（已覆盖 onset−130s；**无需重建 cache**）。
- Produces: window-level `phase1_v2_periictal_window_long.csv`（win_center_rel ∈ [−100,20]）。

- [ ] **Step 1（TDD）**: 写测试 `test_periictal_epoch_grid_covers_preictal`：epoch main_rel=[-100,20]、window=10、step=5 → 断言 grid 含 win_center_rel<0（pre-ictal）窗且 ictal_fraction 仅对 post 窗要求（pre 窗 ictal_fraction=0 是合法 pre-ictal，**不能被 ictal_fraction_min 全滤掉**）。
- [ ] **Step 2**: 跑 → FAIL。
- [ ] **Step 3**: 改 `_epoch_grid` / alignment：pre-ictal 窗（win_end_rel≤0）**豁免 ictal_fraction_min**（它们本就在发作前），只对 onset 后窗要求 ictal_fraction≥0.5。加 `epoch_region` 标签（pre/onset/post）。
- [ ] **Step 4**: 跑 → PASS。
- [ ] **Step 5**: 全 20 跑 window-level alignment（−100→+20s，raw feature），输出带 win_center_rel + epoch_region。
- [ ] **Step 6**: Commit。

**判据**：window_long 覆盖 [−100,20]、pre 窗未被误滤、每窗有 epoch_region 标签。

### Task 3.2: Alignment 轨迹（onset vs pre-ictal）+ 双锚 sensitivity

**Files:**
- Create: `scripts/run_topic5_v2_trajectory.py`
- Output: `phase1_v2_alignment_trajectory.csv`（per subject × time-bin 的 cohort median alignment）+ 图

- [ ] **Step 1（TDD）**: 写测试 `test_onset_vs_preictal_metric`：给合成轨迹（pre 段 alignment≈0.3、post 段≈0.7）→ 断言 `onset_rise = median(post) − median(pre)` 计算正确、方向为正。
- [ ] **Step 2**: 跑 → FAIL。
- [ ] **Step 3**: 实现 `onset_rise` = median(alignment in [0,20]) − median(alignment in [−100,−20])，per subject → cohort median + 一个 within-subject 时间置换 null（打乱 window 的时间标签）判 rise 是否显著。
- [ ] **Step 4**: 跑 → PASS。
- [ ] **Step 5**: **判据（LOCK，encode 结论）**：
  - `onset_rise` cohort median **显著 >0**（时间置换 null）→ "scaffold 是 **onset-triggered**（发作点着后才对齐）"。
  - `onset_rise ≈ 0`（pre ≈ post）→ "scaffold **pre-ictal 已存在**（静态解剖底座，非发作特异动态）"——这会呼应 Topic 5 A-line "distal pre-onset 不弱于 ictal" 的先验。
  - 报告 window-level 轨迹图（−100→+20s，cohort median ± IQR）。
- [ ] **Step 6**: 双锚 sensitivity：epilepsiae 用 clin_onset vs eeg_onset 各跑一遍 `onset_rise`，判 clinical onset 不准对结论的影响（**若两锚结论一致 → 稳；不一致 → onset 定义敏感，标注**）。
- [ ] **Step 7**: Commit + 图（遵 figure_style_guide + README + 目视）。

---

## Phase go/no-go（phase1-v2 完成判据）

phase1-v2 **完成**（可再验收）当且仅当：
1. **W1 Gate B/C**：落到锁定的 Outcome A/B/C 之一，tier 相应更新（不越 candidate-scaffold / broadband-recruitment）。
2. **W2 phenotyping**：per-subject 弱一致写实 + phenotype 有/无明确（|r|>0.4 阈）；跨池稳 subject 名单出。
3. **W3 time**：`onset_rise` 落到 onset-triggered / pre-ictal-present 之一 + 双锚一致性标注。

**tier 天花板不变**：即便三条全"正面"，最高仍写 **"residual-robust, subject-heterogeneous candidate scaffold；onset-triggered/pre-ictal（据 W3）"**——**不**写 formal Gate A / 机制。

## Non-goals（明确禁止）

- 不追 formal within-shaft Gate A（SEEG 杆几何 2/20 硬限制，非本 phase 能解；除非未来换更密电极队列）。
- 不做 HFO/LVFA/ripple 机制、不做 timing-order replay。
- 不做 KMeans 硬 subtype 显著性主张（n 小；只描述性 subgroup）。
- 不建新仿真 / 不碰 Phase-2 criticality 或 V3a mode-transition（各自独立 plan）。
- 不把 min3 sensitivity 升为 primary（§1 spec 已锁 sensitivity-only）。
