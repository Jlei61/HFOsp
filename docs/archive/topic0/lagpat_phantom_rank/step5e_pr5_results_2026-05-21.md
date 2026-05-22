# Step 5e — PR-5 / PR-5-B 修过版重跑结果（2026-05-21）

> 状态：Step 5e 完成。3 个 aux script 加 `--masked-features` flag；3 项新 TDD（path-routing smoke）+ 既有 `tests/test_interictal_propagation.py` 51 项全绿（54/54 PASS）。**PR-5-A gate 在 masked 下 overall_pass=True (main 27/4, aux 26/5)**；**PR-5-B retained-cohort 主结论 `candidate_a_global.post_minus_baseline = +65.66 events/h, Wilcoxon p=0.0004, sign 21+/6−, bonferroni PASS`**（orig 同口径 +65.46 events/h, p=0.0013, 19+/4−, bonferroni PASS）——**plan §4 pre-registered primary cohort claim 方向保持且 magnitude 几乎相同**。**2 条 plan §4.5 pre-registered secondary composition diagnostic 翻转 (share)** + **1 条 PR-7 §17 补丁 secondary diagnostic 翻转 (transition lift)**：fig_a `composition_diagnostic.share post_minus_baseline` 原 p=0.015 sig → 修过 p=0.86 NULL（plan §11.3 已记录原版是"panel d 反方向" + direction_consistent 6/23 不达，本身不算 cohort claim；mask 把 nominal Wilcoxon 也推回 NULL）；fig_a extended Δ post−base 同向翻转 p=0.006→0.82；fig_b `transition_lift.post_minus_baseline` 原 p=0.29 NULL → 修过 p=0.022 Wilcoxon-only sig（sign p=0.076 边缘，不达 cohort claim 门槛）。三条翻转都通过 intersection n=25/26 like-for-like 复核确认非 cohort 替换造成。**PR-5 plan §4.5 明确**："**主结论范围严格收敛在 absolute recruitment 一条线上**" (§3 第 7 节)，share 与 transition lift 都是 secondary composition diagnostic / PR-7 补丁，**不进 §4 主 Bonferroni 池**，**不参与 sensitivity gate 判定**——这些翻转不构成 framework-level revision，**亦不构成 Checkpoint B 重新打开的理由**（advisor 2026-05-21 已 sign off Checkpoint B；该报告 §3.3 明确"5e 后 SEF-ITP H4 evidence base 整体瓦解风险"指向的是 PR-5-B 主信号翻转的情境，本档 primary 主信号没翻）。
> 主入口：`docs/topic0_methodology_audits.md`
> 上游：`./step5a_pr2_results_2026-05-20.md` / `./step5b_pr25_results_2026-05-20.md` / `./step5c_pr3_results_2026-05-20.md` / `./step5f_pr6_results_2026-05-21.md`（5f 已先做）
> 路线图：`./rerun_roadmap_2026-05-20.md` §5e
> 修过版结果：
> - `results/interictal_propagation_masked/pr5a_novel_template_gate.json`
> - `results/interictal_propagation_masked/pr5b_recruitment_shift.json`
> - `results/interictal_propagation_masked/pr5b_recruitment_shift_extended.json/.csv`
> - `results/interictal_propagation_masked/pr5_transition_windows.json/.csv`
> - `results/interictal_propagation_masked/template_share_switching/figures/{fig_a, fig_b}.{png,pdf}, README.md`

---

## 1. 三段式朴素话

**测了什么** —— PR-5 整套 "间期模板在发作邻近窗口（baseline / pre / post）的招募与切换" 的检验，用 5a 修过版的 PR-2 cluster labels 重跑：
- **PR-5-A** = "新模板 falsification gate"（pre/post 窗口里出现的 lagPat 是不是仍能被已有 K 个 cluster templates 解释——如果不能，gate FAIL，意味着发作邻近期出现了新模板，PR-5-B 主张失效）
- **PR-5-B** = "招募率移位"（pre/post 窗口里 dominant cluster 的事件/小时是否相对 baseline 抬升），PR-5 archive 核心数字 "post-ictal +65.46 events/h, Wilcoxon p=0.00128"
- **PR-5-B extended** = 把 PR-5-B 跑到全 stable_k=2 cohort（绕过 PR-5-A gate；fig_a 描述层用）
- **PR-5 transition windows** = 模板切换次数 odds 的发作邻近窗口分解（PR-7 §17 补丁；fig_b 用）

**怎么测的** ——
1. 输入：5a 写好的 `results/interictal_propagation_masked/per_subject/<sid>.json`（masked PR-2 cluster labels）
2. **PR-5-A / PR-5-B 自身不需要 KMeans 调用**——它们读取 per-subject JSON 里 `adaptive_cluster.labels` 字段，labels 已是 masked 输出；所以**无 `use_masked_features=True` kwarg 需要透传**，只需 path routing。`scripts/run_interictal_propagation.py` 的 `RESULTS_DIR` 已经在 5a/5b/5c/5d 加 `--masked-features` 时 auto-route，PR-5-A/B 直接用，**无新代码改动**。
3. 给 3 个 auxiliary script 加 `_apply_masked_paths()` + `--masked-features` flag：
   - `scripts/run_pr5b_share_extended.py`：路径重路由 `RESULTS_DIR / PR1_SUBJECT_SUMMARY / OUT_JSON / OUT_CSV`。**关键 filename trap**：masked 树的 PR-1 summary 是 `pr1_subject_summary.json`（**无 `_n40` 后缀**），机械只 swap 目录会 FileNotFound；必须连 filename 也 swap（advisor 提醒，已经 capture 在 test）。
   - `scripts/run_pr5_transition_windows.py`：同上 4 项 path swap + filename trap。
   - `scripts/plot_template_share_switching.py`：路径重路由 input JSON + figures 输出目录。
4. 3 项新 TDD（`tests/test_pr5_masked_path_routing.py`，全部 PASS）+ 既有 `tests/test_interictal_propagation.py` 51 项无回归（54/54 PASS）。
5. 跑序：
   - `python scripts/run_interictal_propagation.py --pr5-gate --masked-features` → 写 `pr5a_novel_template_gate.json`（~23 min wall）
   - gate PASS → `python scripts/run_interictal_propagation.py --pr5-recruitment --masked-features` → 写 `pr5b_recruitment_shift.json`（~30 s，hard fail-fast：gate FAIL 时 `SystemExit(2)`）
   - `python scripts/run_pr5b_share_extended.py --masked-features`（绕过 gate，给 fig_a 用，~30 s）
   - `python scripts/run_pr5_transition_windows.py --masked-features`（给 fig_b 用，~30 s）
   - `python scripts/plot_template_share_switching.py --masked-features`（写 fig_a/b + 新 README）

**揭示了什么** ——

- **PR-5-A gate 仍 PASS**（masked overall_pass=True，main 6/6 axis pass, aux 6/6 axis pass；orig 同 6/6+6/6）。Cohort 从 main 23/22 (orig) 涨到 27/26 (mask) —— **mask 后更多 subject 在 gate 内**（mask 5a 没把任何 stable_k=2 subject 翻出去；新增 cohort 主要是 mask 5a 修过 cluster label 后 `n_events_by_state ≥ min_state_events_for_gate=30` 的条件更容易满足）。结论：**发作邻近期没有出现需要超出已有 cluster templates 才能解释的"新模板"**，PR-5-B 的招募率 framing 在 masked 下仍 valid。

- **PR-5-B retained-cohort 主结论 `candidate_a_global.post_minus_baseline` 几乎完全不变**：orig **+65.46 events/h, n=23, Wilcoxon p=0.0013, sign 19+/4− (p=0.003), bonferroni PASS**；修过版 **+65.66 events/h, n=27, Wilcoxon p=0.0004, sign 21+/6− (p=0.006), bonferroni PASS**。Magnitude 几乎完全相同（Δ +0.20 events/h on median），p 值因 n 增加略改善，方向 100% 保持。**PR-5 主统计指标在修过版不需要修正**。`candidate_b_window`（窗口分母）也保持：orig +65.46/p=0.002 → mask +65.66/p=0.0011，都 bonferroni PASS。

- **⚠ secondary 指标 fig_a `composition_diagnostic.share.post_minus_baseline` 显著 → NULL 翻转**：orig n=23 median +0.0156, Wilcoxon p=0.0149 (sig), sign p=0.035 (sig), 17+/6− → mask n=27 median +0.0021, Wilcoxon p=0.859 (NULL), sign p=1.00 (NULL), 14+/13−。**这是 fig_a "dominant 模板 share 在 post-ictal 抬升"的核心 metric 翻转**。Intersection n=25 like-for-like 复核：orig median +0.0295, p=0.0061, 18+/7− → mask median +0.0021, p=0.958, 13+/12−。**翻转不是 cohort 替换造成的，是 metric 本身的修过版重算结果**。Extended cohort 同样翻：orig n=26 median +0.0253, p=0.006, 19+/7− → mask n=26 median +0.0054, p=0.822, 14+/12−。
  - **解读**：原版 share 抬升被 phantom 加进 dominant cluster 的混淆事件人为放大。修过版后 dominant cluster 不再吸收错位的 phantom-noisy 事件，share 比率回归 ~0。但 dominant 模板的**绝对事件率**（events/h）仍上升 ~65 events/h——意味着 post-ictal 时段是**整体事件率上升 + dominant 模板等比例招募**，不是 share 比率改变。
  - **论文叙事建议**：fig_a 主张从"dominant share 抬升"改为"dominant 绝对事件率抬升（PR-5-B candidate_a_global），share 维持"。这一改动**收紧但不否定** PR-5 archive 主结论。

- **⚠ secondary 指标 fig_b `transition_lift.post_minus_baseline` NULL → Wilcoxon-only borderline 翻转**：orig n=26 median +0.024, Wilcoxon p=0.291 (NULL), sign p=0.557 (NULL), 15+/11− → mask n=26 median +0.009, Wilcoxon p=0.022 (sig), sign p=0.076 (borderline), 18+/8−。**Wilcoxon 跨过 0.05；sign-test 仍未达 0.05**。Intersection n=26 like-for-like 复核：orig median +0.026, p=0.217 → mask median +0.009, p=0.0151, 15+/11− → 18+/8−。
  - **解读**：修过版后 next-event template-switching 的 lift 在 post-ictal 比 baseline 略增，effect size 小（中位 Δ lift +0.009），方向 18 subject 正向。但 sign-test 不达 cohort 显著门槛——不能升级为 cohort claim。
  - **论文叙事建议**：fig_b 从 "transition 不变" 改为 "transition 在 post-ictal vs baseline 出现 weakly significant 抬升（Wilcoxon p=0.022, sign p=0.076 边缘）；effect size 小"。仍 supports Topic 1 "间期刻板时序几何不变形" 主结论——切换率的微小抬升不改变模板内部几何，只是 dominant 模板更常被启动。

- **PR-5-A axis details 不变**：r 轴 / e 轴 / gap 轴在 6 个 (config, pair) 上都仍 pass；axis-level medians 大致同向。e 轴的 pre-baseline median 从 +0.073 (orig main) 涨到 +0.119 (mask main)，但 Wilcoxon 仍 NULL（p=0.27 → 0.15）；gap 轴的 post-baseline median 从 +0.149 涨到 +0.313，Wilcoxon NULL (p=0.21 → 0.19)。这些是描述性方向数字，gate 决策（pair_pass）全部不变。

- **总体判读**：**PR-5 framework 在修过版下不需要 framework-level revision**——gate 仍 PASS，主统计 `dominant_rate.candidate_a_global.post_minus_baseline = +65.66 events/h, p=0.0004` 保持。**3 条 secondary metric 翻转**（share Wilcoxon 显著→NULL × 两 cohort，transition lift NULL→Wilcoxon-only borderline sig）只影响 fig_a/fig_b 的叙事 framing，不动 PR-5 plan §4 核心 cohort claim。PR-5 plan §4.5 + §11.3 明确："**主结论范围严格收敛在 absolute recruitment 一条线上**"，share 被预先注册为 secondary composition diagnostic、不进主 Bonferroni 池——所以本档翻转**完全在 pre-registered secondary tier 范围内**。**Checkpoint B 已在本档之前完成 advisor sign-off**（`checkpoint_b_report_2026-05-21.md`，基于 5d.2.2 + 5d.3）；该报告 §3.3 触发 "5e 后单独评估" 的预设条件是"PR-5-B post-baseline events/h 主信号也翻转"，**本档 primary 主信号未翻**，不触发 Checkpoint B 重新打开。下一步：5i 收口阶段调整 Topic 1 主文档 fig_a/fig_b 叙事 framing。

代号补注：PR-5-A novel-template gate = `compute_novel_template_gate` → `pr5a_novel_template_gate.json.cohort`；PR-5-B retained recruitment = `compute_template_recruitment_shift` → `pr5b_recruitment_shift.json.cohort.main.dominant_rate.candidate_a_global`；extended = bypass gate 全 stable_k=2 cohort，仅描述层；transition windows = `_accumulate_pairs` + per-state lift = transition_odds / baseline_odds。

---

## 2. 实现层改动（surgical）

| 文件 | 改动 |
|---|---|
| `scripts/run_pr5b_share_extended.py` | 新增 `_apply_masked_paths()` swap `RESULTS_DIR / PR1_SUBJECT_SUMMARY / OUT_JSON / OUT_CSV`；`--masked-features` flag；filename trap 注释（masked 无 `_n40`）。argparse 用 description=`__doc__`。|
| `scripts/run_pr5_transition_windows.py` | 同上 4 项 swap + flag + filename trap 注释。|
| `scripts/plot_template_share_switching.py` | 路径 swap `RESULTS_DIR / EXT_JSON / ORIG_JSON / TRANSITION_JSON / FIGURES_DIR`；`--masked-features` flag。`STRICT_MATCH_SUBJECTS` 保持原版 PR-4D 派生集合（PR-4D 在 masked 下 SKIPPED 待 5i 收口决定是否补）——本图 strict-match 高亮**沿用原版集合**，**不更新**，在 README 中明确标注。|
| `tests/test_pr5_masked_path_routing.py`（新建） | 3 项 smoke test：每个 aux script 一项，验证 `_apply_masked_paths()` 在 reload 后正确 swap 所有 path globals 到 `interictal_propagation_masked/`；额外断言 PR1_SUBJECT_SUMMARY 的 filename 从 `pr1_subject_summary_n40.json` swap 到 `pr1_subject_summary.json`（filename trap regression guard）。|

不动：
- `scripts/run_interictal_propagation.py`（PR-5-A `_run_pr5_gate` + PR-5-B `_run_pr5_recruitment` 已经在 5a/5b/5c 时通过 `RESULTS_DIR` global auto-route 实现 masked path swap，且**它们不调用 KMeans**——直接读 per-subject JSON 里 `adaptive_cluster.labels`，labels 已经是 masked 5a 写好的；无 `use_masked_features` kwarg 需要透传）
- `src/interictal_propagation.py`（`compute_novel_template_gate` / `compute_template_recruitment_shift` 不读 lagPatRank-derived KMeans，只用上游 cluster labels；不需要 mask 改动）
- 任何 PR-2 / PR-2.5 / PR-3 / PR-4 / PR-6 / PR-7 输出

**TDD 验证**：
```
tests/test_pr5_masked_path_routing.py: 3/3 PASS
tests/test_interictal_propagation.py: 51/51 PASS（既有，无回归）
合计 54/54 PASS
```

---

## 3. Cohort 数字对比表

> **Cohort note**: 原版（orig）数据来自 `results/interictal_propagation/pr5a_novel_template_gate.json` 和 `pr5b_recruitment_shift.json`（5a 改造前的 latest），masked 数据来自对应 `interictal_propagation_masked/` 路径。PR-5-A gate 的 like-for-like 比较以"同 cohort 定义"（`min_state_events_for_gate=30`）为准；PR-5-B 主比较锁定 retained-cohort（gate PASS subjects）。**关键 like-for-like 复核**：fig_a 翻转 + fig_b 翻转都在 **intersection cohort n=25 / n=26** 上复核确认非 cohort 替换造成（intersect 是 orig main ∩ mask main 在 `pr5b_recruitment_shift_extended.per_subject.main` 上的 26 个 subject 减 1 NaN = 25）。

### 3.1 PR-5-A novel-template gate cohort & overall_pass

| | orig main | orig aux | mask main | mask aux |
|---|---:|---:|---:|---:|
| n_subjects_eligible | 23 | 22 | **27** | **26** |
| n_subjects_ineligible | 3 | 4 | 4 | 5 |
| gate_pass (per config) | True | True | **True** | **True** |
| **overall_pass** | **True** | — | **True** | — |
| axis_pass (r / e / gap) | T / T / T | T / T / T | T / T / T | T / T / T |

**Cohort 成员变化**：
- orig main ineligible (3): `yuq/huanghanwen, yuq/zhangjinhan, epi/818`
- mask main ineligible (4): `yuq/zhaojinrui, yuq/huanghanwen, yuq/zhangjinhan, epi/818`
- mask 多了 `zhaojinrui` (no_gate_eligible_events)，少了 4 个原本 below_min_state_events 的 subject——**mask 后 cluster label 重算，部分 subject 的 baseline/pre/post 事件计数过 `min_state_events_for_gate=30` 门槛**。
- aux 同理：mask 多 `chenziyang` ineligible 持平，新增 `zhaojinrui`。

**判读**：mask 后 cohort 扩张但 gate 主结论**不动**——overall_pass 仍 True，6 个 axis 全 pass。

### 3.2 PR-5-A 单 axis 检验细节（r / e / gap）

| axis | pair | orig main (median Δ, Wilcoxon p) | mask main | orig aux | mask aux |
|---|---|---|---|---|---|
| **r** | pre−base | (−0.009, 0.243) | (0.000, 0.578) | (0.000, 0.836) | (0.000, 0.717) |
| **r** | post−base | (−0.007, 0.615) | (0.000, 0.277) | (0.000, 0.711) | (0.000, 0.945) |
| **e** | pre−base | (+0.073, 0.273) | (+0.119, 0.151) | (+0.012, 0.799) | (+0.055, 0.635) |
| **e** | post−base | (+0.016, 0.580) | (−0.076, 0.897) | (−0.029, 0.775) | (−0.035, 0.515) |
| **gap** | pre−base | (+0.013, 0.615) | (−0.016, 0.972) | (+0.025, 0.406) | (−0.060, 0.394) |
| **gap** | post−base | (+0.149, 0.211) | (+0.313, 0.186) | (+0.162, 0.425) | (+0.329, 0.105) |

**判读**：所有 axis × pair 都 `pair_pass=True`（数值在 plan §3.5 阈值范围内）。axis-level medians 有些变化（如 e.post−base orig +0.016 → mask −0.076），但都在 NULL 区，方向变化不构成 axis 失败。**gate 判定完全保持**。

### 3.3 PR-5-B retained cohort（PR-5 archive 核心：post-ictal +65.46 events/h, p=0.00128）

> **Like-for-like comparison**：cohort 定义来自各自 PR-5-A retained list（orig main n=23 / aux n=22；mask main n=27 / aux n=26）。orig main 主结论数字直接读 `pr5b_recruitment_shift.json.cohort.main.dominant_rate.candidate_a_global`，mask 同。

#### candidate_a_global（fix-window 全部覆盖；主统计）

| | orig main (n=23) | mask main (n=27) | orig aux (n=22) | mask aux (n=26) |
|---|---|---|---|---|
| pre−base median (events/h) | +10.66 | +11.86 | +16.03 | +17.30 |
| pre−base Wilcoxon p | 0.160 | 0.313 | **0.042** | **0.016** |
| pre−base sign-test p | 0.093 | 0.248 | 0.053 | **0.029** |
| pre−base n+/n− | 16/7 | 17/10 | 16/6 | 19/7 |
| pre−base bonferroni_pass | False | False | False | False |
| **post−base median (events/h)** | **+65.46** | **+65.66** | +42.43 | +42.55 |
| **post−base Wilcoxon p** | **0.0013** | **0.0004** | **0.0115** | **0.0039** |
| **post−base sign-test p** | **0.0026** | **0.0059** | 0.134 | 0.076 |
| **post−base n+/n−** | **19/4** | **21/6** | 15/7 | 18/8 |
| **post−base bonferroni_pass** | **TRUE** | **TRUE** | False | **TRUE** |
| post−pre median (events/h) | +5.87 | +11.60 | −9.32 | −14.96 |
| post−pre Wilcoxon p | 0.142 | 0.186 | 0.702 | 0.532 |
| post−pre sign-test p | 0.405 | 0.701 | 0.524 | 0.169 |
| post−pre n+/n− | 14/9 | 15/12 | 9/13 | 9/17 |
| post−pre bonferroni_pass | False | False | False | False |

#### candidate_b_window（窗口分母版；sensitivity）

| | orig main (n=23) | mask main (n=27) | orig aux (n=22) | mask aux (n=26) |
|---|---|---|---|---|
| post−base median (events/h) | +65.46 | +65.66 | +42.95 | +37.49 |
| post−base Wilcoxon p | **0.0021** | **0.0011** | 0.023 | **0.0067** |
| post−base sign-test p | 0.011 | **0.019** | 0.134 | **0.029** |
| post−base bonferroni_pass | TRUE | TRUE | False | TRUE |

**判读**：
- **main 配置 post-ictal +65 events/h 主结论完全保持**：median 几乎相同 (+65.46 → +65.66)，Wilcoxon p 略 improve (0.0013 → 0.0004)，sign-test 也 sig，bonferroni PASS 两版皆然。**PR-5 archive §4.5 核心数字不需要 update**。
- aux 配置 post-base 在 mask 下从未 bonferroni pass 变成 PASS——nominal sig 加强。
- **方向 100% 保持，无 cohort-level 翻转**。

#### composition_diagnostic.share（fig_a 主 metric — ⚠ 翻转）

| | orig main (n=23) | mask main (n=27) | orig aux (n=22) | mask aux (n=26) |
|---|---|---|---|---|
| pre−base median Δ share | −0.0023 | −0.0181 | n/a | n/a |
| pre−base Wilcoxon p | 0.601 | 0.135 | 0.388 | 0.635 |
| **post−base median Δ share** | **+0.0156** | **+0.0021** | n/a | n/a |
| **post−base Wilcoxon p** | **0.0149 ✓** | **0.859 ⚠** | **0.030 ✓** | **0.394 ⚠** |
| **post−base sign-test p** | **0.035 ✓** | **1.00 ⚠** | n/a | n/a |
| **post−base n+/n−** | **17/6** | **14/13** | n/a | n/a |
| post−pre median Δ share | +0.0447 | +0.0219 | n/a | n/a |
| post−pre Wilcoxon p | **0.0135 ✓** | 0.194 ⚠ | 0.337 | 0.689 |

**⚠ 翻转**：post−base 显著 → NULL（main 同 aux 都翻），post−pre main 也翻。direction (median 仍正)保持但 magnitude collapse。**Intersection n=25 like-for-like 复核**：orig median +0.030, p=0.006, 18+/7− → mask median +0.002, p=0.96, 13+/12−，**翻转不是 cohort 替换造成的**。

### 3.4 PR-5-B extended cohort（绕过 gate 的描述层，fig_a 用）

> cohort definition：原版 cohort = stable_k=2 in `pr1_subject_summary_n40.json`（n=35）；mask cohort = stable_k=2 in masked `pr1_subject_summary.json`（n=34，少 `epilepsiae/916`，因 5a stable_k 翻 2→4）。

| metric | orig (n=26 finite) | mask (n=26 finite) | intersect n=25 orig | intersect n=25 mask |
|---|---|---|---|---|
| Δ post−base median | +0.0253 | **+0.0054** | +0.0295 | **+0.0021** |
| Δ post−base Wilcoxon p | **0.0056 ✓** | **0.822 ⚠** | **0.0061 ✓** | **0.958 ⚠** |
| Δ post−base sign p | **0.029 ✓** | 0.845 ⚠ | **0.043 ✓** | **1.00 ⚠** |
| Δ post−base n+/n− | 19/7 | 14/12 | 18/7 | 13/12 |
| Δ pre−base median | −0.004 | −0.018 | — | — |
| Δ pre−base Wilcoxon p | 0.546 | 0.178 | — | — |
| Δ post−pre median | +0.046 | +0.023 | — | — |
| Δ post−pre Wilcoxon p | **0.0056 ✓** | 0.117 ⚠ | — | — |

**⚠ 翻转**：与 retained-cohort `composition_diagnostic.share` 翻转一致，extended cohort 上 fig_a 主张"dominant share 抬升"翻 NULL。**Intersection 复核也翻转**。

### 3.5 PR-5 transition windows（PR-7 §17 补丁，fig_b 用）

> cohort：stable_k=2 中能成功 compute（≥4 events in baseline+pre+post）的 27 subject。剔除 `yuquan/huanghanwen`（n_pairs<10, lift outlier）后 n=26 用于 figure。

| metric | orig (n=26 figure) | mask (n=26 figure) | intersect n=26 orig | intersect n=26 mask |
|---|---|---|---|---|
| Δ post−base lift median | +0.024 | +0.009 | +0.026 | +0.009 |
| Δ post−base lift Wilcoxon p | 0.291 | **0.022 ⚠** | 0.217 | **0.015 ⚠** |
| Δ post−base lift sign p | 0.557 | 0.076 (边缘) | 0.557 | 0.076 (边缘) |
| Δ post−base lift n+/n− | 15/11 | 18/8 | 15/11 | 18/8 |
| Δ pre−base lift median | +0.007 | +0.001 | — | — |
| Δ pre−base lift Wilcoxon p | 0.437 | 0.803 | — | — |
| Δ post−pre lift median | +0.019 | +0.013 | — | — |
| Δ post−pre lift Wilcoxon p | 0.353 | 0.280 | — | — |

**⚠ 翻转（反方向）**：post−base lift 原 NULL → 修过 Wilcoxon-only sig（sign-test 仍 0.076 NULL/borderline）。Magnitude 反而**变小**（+0.024 → +0.009），但 18/26 subject 同向（+），Wilcoxon 把它推进 0.05。sign-test 没达 cohort 显著门槛——**不能升级 cohort claim**。Intersection n=26 复核同样翻转。

---

## 4. 判读 — Checkpoint 标准对照

按 `rerun_roadmap_2026-05-20.md` 默认 gate（本档独立判读 5e 内部标准；Checkpoint B 现在 5d.3 + 5e 都完成，可正式触发）：

| Gate | 状态 |
|---|---|
| PR-5-A gate overall_pass 反转 (orig PASS → mask FAIL)？ | ❌ NO（mask 仍 PASS, 6/6 axis pass × 2 configs）|
| PR-5-A axis_pass 反转？ | ❌ NO（所有 6 axis 在两版都 pair_pass=True）|
| PR-5-B candidate_a_global post_minus_baseline 方向反转？ | ❌ NO（orig +65.46 → mask +65.66，几乎完全相同正向；19+/4− → 21+/6−）|
| PR-5-B candidate_a_global post_minus_baseline 显著 ↔ NULL 翻转？ | ❌ NO（orig p=0.0013 → mask p=0.0004，都 bonferroni PASS）|
| PR-5-B candidate_b_window post_minus_baseline 方向反转？ | ❌ NO（orig +65.46 → mask +65.66）|
| PR-5-B composition_diagnostic.share post_minus_baseline 显著 ↔ NULL 翻转？ | ⚠ **YES**（orig p=0.0149/sign 0.035 sig → mask p=0.859/sign 1.00 NULL；secondary tier；direction median 仍正但 magnitude collapse；intersection n=25 复核确认）|
| PR-5-B extended Δ post−base 显著 ↔ NULL 翻转？ | ⚠ **YES**（orig p=0.006 sig → mask p=0.822 NULL；secondary tier；与 retained composition_diagnostic.share 翻转一致；intersection n=25 复核确认）|
| PR-5 transition_lift post_minus_baseline 显著 ↔ NULL 翻转？ | ⚠ **YES (Wilcoxon-only borderline)**（orig p=0.291 NULL → mask p=0.022 Wilcoxon sig；但 sign p=0.076 不达 cohort claim 门槛；secondary tier；反方向翻转——orig NULL → mask sig；intersection n=26 复核确认）|
| Primary PR-5 cohort claim 方向反转？ | ❌ NO（所有 primary `dominant_rate` metric 方向 + 显著性保持）|

**Step 5e 整体方向**：⚠ PASS WITH 3 SECONDARY FLIPS — primary cohort claim (`candidate_a_global.post_minus_baseline = +65.66 events/h, p=0.0004, bonferroni PASS`) 完全保持；3 条 secondary metric 翻转（`composition_diagnostic.share` 显著→NULL，`extended Δ post−base` 显著→NULL，`transition_lift post−base` NULL→Wilcoxon-only sig），都在 fig_a/fig_b 描述层，且 intersection like-for-like 复核确认非 cohort 替换造成。**Checkpoint B 已在本档之前 advisor sign-off**（`checkpoint_b_report_2026-05-21.md`，基于 5d.2.2 PR-4B L3 fragility-on-small-n + 5d.3 PR-4C NULL stays NULL）；本档 secondary 翻转**不构成重新打开 Checkpoint B 的理由**——CkB 报告 §3.3 触发 "5e 后单独评估" 的条件是 PR-5-B 主信号翻转，本档 primary 主信号没翻。PR-5 plan §4.5 + §11.3 已 pre-registered share 是 secondary diagnostic、不进主 Bonferroni 池。可进 5g/5h（user 并行 session 已分别完成）+ 5i 收口。

---

## 5. 不再 valid 的旧数字 / 需要主文档更新的位置

### 5.1 Topic 1 主文档（`docs/topic1_within_event_dynamics.md`）

§4.5（若有）"post-ictal +65.46 events/h, p=0.00128" 是 PR-5-B retained-cohort 主结论。**修过版口径：保持原数字**（+65.46 → +65.66, 0.0013 → 0.0004，方向不变、bonferroni 仍 PASS）。**fig_a/fig_b 叙事需要调整**：
- 从 "post-ictal dominant 模板 share 抬升" 改为 "post-ictal dominant 模板**绝对事件率**抬升 ~65 events/h；**share 比率维持**（mask Wilcoxon p=0.86 NULL）"
- 从 "transition lift 不变" 改为 "transition lift 在 post-ictal vs baseline **出现 Wilcoxon-only borderline 抬升**（mask p=0.022, sign p=0.076 边缘，不达 cohort claim 门槛）"

具体改写留到 Step 5i 收口阶段。

### 5.2 `results/interictal_propagation/template_share_switching/figures/README.md`

原 README §fig_a 的核心数字 "post−baseline median +0.021, Wilcoxon p=0.005**, sign 20+/7-（p=0.019）" 用的是 PR-5-B extended cohort (n=27)。修过版 (n=27)：median +0.005, Wilcoxon p=0.822 (NULL), sign 14+/12- (p=0.85)。**本档不直接覆盖原 README**；masked figures 在 `interictal_propagation_masked/template_share_switching/figures/` 下新建 README，引用本 archive doc。

### 5.3 PR-4D STRICT_MATCH_SUBJECTS（plot_template_share_switching.py 高亮集合）

原版 9 subject 来自 PR-4D `_score_rate_cluster_seizure` strict-match。PR-4D 在 masked 下 SKIPPED（`docs/topic0_methodology_audits.md` §5 row 5d.4），masked figure 沿用 orig STRICT_MATCH，技术上**信息不更新**。5i 收口时若需要在 masked cohort 上重做 strict-match，须先补跑 PR-4D。已在 masked figure README 明确标注。

---

## 6. 下一步（按 rerun_roadmap §5g 起步）

- **Checkpoint B 状态**：已于 2026-05-21 advisor sign-off（`checkpoint_b_report_2026-05-21.md`），verdict = **trigger met (soft form), broad re-derivation 继续**——基于 5d.2.2 PR-4B L3 高置信 fragility-on-small-n。**本档 PR-5 翻转不构成重新打开 Checkpoint B 的理由**：
  - Checkpoint B 报告 §3.3 明确 "5e 完成后单独评估，不在本 Checkpoint B 范围"——但前提是"PR-5-B 主信号也翻转"（指 `dominant_rate.candidate_a_global.post_minus_baseline`）；**本档 primary 主信号没翻**（+65.46 → +65.66, p=0.0013 → 0.0004，bonferroni PASS 保持）。
  - 翻转的 3 条都是 plan §4.5 / PR-7 §17 pre-registered secondary，不进 §4 主 Bonferroni 池。
- **5g (PR-7 antagonistic pairing on masked)** — 已完成（user 并行 session，详见 Topic 0 §5 row 5g 与 5h 的同步更新；本档不重复）。P3 INCONCLUSIVE-locked 翻转 = framework-level revision，需先看 5g 报告。
- **5h (Topic 4 attractor on masked)** — 已完成（user 并行 session）。
- **5i (主文档收口)** — pending。本档 §5.1 + §5.2 给出 Topic 1 主文档 fig_a/fig_b 叙事调整建议（"share 抬升" → "absolute rate 抬升、share 维持"；"transition 不变" → "transition Wilcoxon-only borderline 抬升"）；5i 收口时统一执行。

---

## 7. 工件清单

新生成（masked）：
- `results/interictal_propagation_masked/per_subject/pr5a/*.json` × 31（main config 27 eligible + 4 ineligible written；aux 26+5）
- `results/interictal_propagation_masked/per_subject/pr5b/*.json` × 27（main retained）
- `results/interictal_propagation_masked/pr5a_novel_template_gate.json`
- `results/interictal_propagation_masked/pr5b_recruitment_shift.json`
- `results/interictal_propagation_masked/pr5b_recruitment_shift_extended.json/.csv`
- `results/interictal_propagation_masked/pr5_transition_windows.json/.csv`
- `results/interictal_propagation_masked/template_share_switching/figures/fig_a_template_share.{png,pdf}`
- `results/interictal_propagation_masked/template_share_switching/figures/fig_b_template_switching.{png,pdf}`
- `results/interictal_propagation_masked/template_share_switching/figures/README.md`（新写，含翻转判读）

代码（3 项 TDD + 3 个 aux script 改造）：
- `tests/test_pr5_masked_path_routing.py`（新建，3 tests）
- `scripts/run_pr5b_share_extended.py`（`_apply_masked_paths` + `--masked-features` flag + filename trap）
- `scripts/run_pr5_transition_windows.py`（同上）
- `scripts/plot_template_share_switching.py`（同上，纯路径 + figures 目录）

日志：
- `logs/step5e_pr5a_gate_masked.log`（~23 min wall）
- `logs/step5e_pr5b_recruitment_masked.log`（~30 s）
- `logs/step5e_pr5b_extended_masked.log`（~30 s）
- `logs/step5e_pr5_transitions_masked.log`（~10 s）
- `logs/step5e_plot_template_share_switching_masked.log`（~5 s）

---

## 8. 一句话总结

**PR-5 framework 在 phantom rank 修过版下不需要 framework-level revision**——PR-5-A gate 仍 overall_pass=True (6/6 axis × 2 configs)，PR-5-B retained-cohort **主统计** `dominant_rate.candidate_a_global.post_minus_baseline = +65.66 events/h, Wilcoxon p=0.0004, sign 21+/6−, bonferroni PASS` 几乎完全保持（orig +65.46/p=0.0013）。**3 条 secondary metric 翻转**：fig_a `composition_diagnostic.share post−base` 显著→NULL（p=0.015→0.86，direction median 仍正但 magnitude collapse，intersection n=25 复核确认），fig_a extended Δ post−base 同向显著→NULL（p=0.006→0.82），fig_b `transition lift post−base` NULL→Wilcoxon-only borderline sig（p=0.29→0.022, sign p=0.076 边缘）。**论文 fig_a 叙事需调整**：从"dominant share 抬升"改为"dominant 绝对率抬升、share 维持"。**Checkpoint B 现在可触发**（5d.3 + 5e 都完成）；不动 PR-5 plan §4 核心 cohort claim，不动 Topic 1 主结论 "间期刻板时序几何不变形"。可进 5g + 5h。
