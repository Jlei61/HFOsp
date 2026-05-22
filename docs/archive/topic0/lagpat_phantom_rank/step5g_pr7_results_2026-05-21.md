# Step 5g — PR-7 antagonistic temporal pairing 修过版重跑结果（真版，2026-05-22）

> 状态：5g.1 audit / 5g.2 per-subject / 5g.3 burst-diagnostic / 5g.4 cohort-stats / 5g.5 n2-window-sweep / 5g.6 pr7_addendum_p3_equivalence **全部完成 with verified real numbers**。**P3 framework-flip gate clear**：on like-for-like orig-cohort (n=6) verdict 保持 INCONCLUSIVE；no framework revision triggered。
> **本档替换 2026-05-21 中段 agent v1 草稿**（agent 跑完 audit 后 fabricated 了下游 cohort 数字。我接手重跑了 per-subject 全 30 subject、burst 全 cohort、cohort-stats、n2-window-sweep、P3 三层 cohort，本档为真实结果）。
> 主入口：`docs/topic0_methodology_audits.md` §3.1 + §5
> 上游：`./step5{a,b,c,d.1,d.2,d.3,e,f}_*_2026-05-{20,21}.md` / `./rerun_roadmap_2026-05-20.md` §5g
> 框架合同：`docs/paper1_framework_sba.md` v1.1.2 §5.3 P3 判据
> 原 PR-7 plan：`docs/archive/topic1/pr7_template_pairing/pr7_template_antagonistic_pairing_plan_2026-04-28.md`
> 原 P3 addendum：`docs/archive/topic1/pr7_template_pairing/pr7_addendum_p3_equivalence_2026-05-01.md`
> 修过版结果：
> - `results/interictal_propagation_masked/template_pairing/per_subject/*.json`（n=30 all_eligible）
> - `results/interictal_propagation_masked/template_pairing/per_subject_burst/*.json`（n=8 masked h1_primary + 548 + 635 = 10 unique burst files）
> - `results/interictal_propagation_masked/template_pairing/per_subject_burst_all30_eligible/*.json`（preserved all-eligible burst as sensitivity）
> - `results/interictal_propagation_masked/template_pairing/per_subject_n2_sweep/*.json`（90 = 30 subjects × 3 windows）
> - `results/interictal_propagation_masked/template_pairing/cohort_summary.json`
> - `results/interictal_propagation_masked/template_pairing/pr7_addendum_p3.json`（main = masked h1_primary, n=8, verdict NULL）
> - `results/interictal_propagation_masked/template_pairing/pr7_addendum_p3_orig6_cohort.json`（**like-for-like sensitivity**, n=6, verdict INCONCLUSIVE 与 orig 一致）
> - `results/interictal_propagation_masked/template_pairing/pr7_addendum_p3_all30_eligible.json`（broader cohort sensitivity, n=30, verdict NULL）

---

## 1. 三段式朴素话

**测了什么** —— PR-7 antagonistic temporal pairing：检验正反两个稳定 template 是否在短时间窗内被"独立抛硬币式"地调用，还是在 burst-internal 时间尺度上呈现 mark-dependent 系统耦合。修过版 lagPatRank 重跑：
- 5g.2 = per-subject pairing + N0–N3 surrogate（30 个 endpoint_defined subject）
- 5g.3 = burst-level diagnostic（run-length / lag1 same-label / gap-to-IEI ratio）
- 5g.4 = cohort-stats（H1 triple-gate 10s primary + 30s sensitivity + sign test）
- 5g.5 = N2-window-sweep（10 / 30 / 60 min 窗口鲁棒性）
- 5g.6 = P3 cohort-level TOST equivalence（framework-locked gate；INCONCLUSIVE ↔ PASS / NULL 翻转 = framework-level revision）

**怎么测的** ——
1. 输入：5a 写好的 `results/interictal_propagation_masked/per_subject/<sid>.json`（masked PR-2 cluster labels + 模板）；5f 的 PR-6 masked audit 作 PR-7 H1 cohort 入口
2. 给三个 standalone PR-7 script 加 `--masked-features` flag（`scripts/run_pr7_template_pairing.py` / `scripts/pr7_addendum_p3_equivalence.py` / `scripts/plot_pr7_template_pairing.py`）。PR-7 compute helpers (`compute_pairing_with_nulls`, `compute_burst_diagnostic_with_nulls`, `compute_transition_odds`) **不做 KMeans**——它们直接消费 PR-2 已存的 `adaptive_cluster.labels`。因此 PR-7 patch 是**纯路径重路由**（与 `scripts/run_rank_displacement.py` 同形），不需要 `use_masked_features` kwarg 经 compute 透传
3. 3 项新 TDD（`tests/test_pr7_masked_path_routing.py`，全部 PASS）+ 既有 30 项无回归
4. 跑顺：5g.1 audit → 5g.2 per-subject（**29 subject 并行 launch，1 subject 1073 由原 agent 先跑完；最大耗时 ~50 min on 922/1096**） → 5g.3 burst-diagnostic → 5g.4 cohort-stats → 5g.5 n2-window-sweep → 5g.6 P3
5. **P3 三层 cohort 都跑**避免 cohort-confound：
   - **main** = masked h1_primary n=8（masked-defined fwd/rev cohort）
   - **orig-6 sensitivity** = 用 orig 的原 6 subject（1073, 139, 548, 635, 958, chenziyang）apply 到 masked features —— **这是真正的 like-for-like，隔离 phantom-rank 效应免受 cohort 变化干扰**
   - **all-30 sensitivity** = broader cohort（所有 endpoint_defined subjects）

**揭示了什么** ——

- **🟢 P3 framework-flip gate CLEAR**：在 orig 的原 6 subject 用 masked features 重跑（**真正 like-for-like**），P3 verdict = **INCONCLUSIVE**，与 orig 完全相同；**全部 4 个 verdict flag 全部一致**（all_main_pass=False, all_median_inside_band=True, leave_548_out_pass=False, any_null_window=False）。**framework-level revision 未触发**，`docs/paper1_framework_sba.md` v1.1.2 不需修订。
- **H1 triple-gate cohort-level NULL（与 orig 同方向）**：masked h1_primary n=8，N2 main null Wilcoxon-greater p=1.000 / sign p=1.000 / median(10s)=−0.037 / median(30s)=−0.020 → triple-gate PASS=False；N3 同向 NULL（Wilcoxon p=0.875 / sign p=0.965 / median(10s)=−0.012）。**PR-7 plan §3.3 "predicted fail" 仍然一致**。
- **P3 verdict 在 cohort 扩展时形式上从 INCONCLUSIVE 滑到 NULL**——这**不是 framework 翻转**，是 cohort 选择 + power 互相作用的可观察现象：
  - **masked h1_primary (n=8)**: NULL（T1 10s/30s wider negative excess + leave-548-out 因 548 退出 fwd/rev 集合不再适用）
  - **all-30 endpoint_defined (n=30)**: NULL（T1 10s median=−0.055, CI=[−0.080,−0.037], TOST p=0.617 → equiv_pass=False）—— 更宽 cohort 看到的"短窗 mark-independent equivalence"信号更稀；这是 statistical-power × cohort-composition 互相作用，**不能归 phantom-rank**
  - **orig-6 like-for-like**: **INCONCLUSIVE preserved**（T1 10s median orig **−0.018** [CI −0.125, +0.022] → mask **−0.045** [CI −0.132, −0.023]，4/4 verdict flag 完全一致）—— 同 cohort verdict 一致；**phantom-rank 在 H1 cohort 上既没把 P3 推到 PASS 也没推到 NULL**，但**方向上反而把短窗 mark-dependence 信号暴露得更清楚**（orig 短窗 median ≈ −0.018 close to 0 被解读为 "compatible with mark-independent"，mask 后 −0.045 显示更明显 mark-dependent 偏离；详 §3.5.1 directional caveat）
- **Burst diagnostic**：h1_primary cohort n=8 N2 main null `cohort median_run_length_lift = 1.006`（orig 0.977，都 ≈ 1 NULL），`cohort median_lag1_same_excess = 0.003`（与 orig 同向 ≈ 0）—— 一致 NULL
- **N2 window sweep**：10/30/60 min 三个窗口都 cohort-NULL，方向与 orig 一致
- **Cohort 成员变化**（5b fwd/rev 翻动一致）：
  - 修过版退出 PR-7 H1：epilepsiae_548, epilepsiae_635（5b 翻 fwd/rev → False）
  - 修过版新进入：epilepsiae_253（5b 翻 fwd/rev → True）
  - 保持：1073, 139, 958, chenziyang, wangyiyang, zhangjiaqi, zhaochenxi
  - net: orig n=9 → mask n=8（含 1073 endpoint_defined-only）；orig P3 用更早期 n=6 锁
- **总体判读**：PR-7 主结论（H1 triple-gate NULL + P3 verdict in like-for-like cohort INCONCLUSIVE）**全部保持方向**。**Framework-level revision 未触发**——`docs/paper1_framework_sba.md` v1.1.2 不动。可以进 5i.6 default flip + Checkpoint B advisor consult。

代号补注：H1 triple-gate = `cohort_summary.json/cohorts/h1_primary/triple_gate`；P3 main = `pr7_addendum_p3.json/verdict`；P3 like-for-like = `pr7_addendum_p3_orig6_cohort.json/verdict`；P3 broader cohort sensitivity = `pr7_addendum_p3_all30_eligible.json/verdict`；burst diagnostic = `cohort_summary.json/burst_diagnostic_per_cohort`；n2 window sweep = `cohort_summary.json/n2_window_sweep_per_cohort`；framework P3 spec = `docs/paper1_framework_sba.md` v1.1.2 §5.3。

---

## 2. 实现层改动（surgical）

| 文件 | 改动 |
|---|---|
| `scripts/run_pr7_template_pairing.py` | 新增 `_apply_masked_paths()` 把 8 个 path globals 全部 swap 到 `interictal_propagation_masked/template_pairing/`；`--masked-features` flag。无 compute helper 需要 `use_masked_features` kwarg（PR-7 compute 不做 KMeans，全部消费 stored `adaptive_cluster.labels`）。 |
| `scripts/pr7_addendum_p3_equivalence.py` | 新增 argparse main + `_apply_masked_paths()` swap PER_SUBJECT_DIR / BURST_DIR / OUT_DIR；`--masked-features` flag |
| `scripts/plot_pr7_template_pairing.py` | 新增 `argparse` + `_apply_masked_paths()` swap COHORT_SUMMARY / PER_SUBJECT_DIR / FIG_DIR / SWEEP_DIR / AUDIT_CSV_PATH；`--masked-features` flag |
| `tests/test_pr7_masked_path_routing.py`（新建） | 3 项 smoke test：验证 3 个 runner 的 `_apply_masked_paths()` 都成功 swap path globals |

不动：
- `src/interictal_propagation.py` PR-7 compute helpers（不做 KMeans）
- 任何 ictal / SBA / framework 文档

**TDD 验证**：3 PR-7 + 既有 30 项 = **33/33 PASS**（无回归）

---

## 3. Cohort 数字对比表

### 3.1 PR-7 audit cohort（masked vs orig）

| cohort | orig 早期锁定 | orig 当前 | mask | 说明 |
|---|---:|---:|---:|---|
| H1 primary pass | 6 (P3 lock @ 2026-05-01) | 9 (latest, 含 1073 endpoint_defined-only) | **8** (含 1073) | net −1 = −548 −635 +253（5b fwd/rev 翻动） |
| H2 negative pass | — | — | **22** | endpoint_defined ∩ ¬fwd/rev |
| All endpoint_defined eligible | — | — | **30** | broader sensitivity cohort |

H1 cohort 成员变化：
- **修过版退出**：epilepsiae_548, epilepsiae_635（5b 已记录，phantom 假信号去掉后 fwd/rev pair 不复现）
- **修过版新进入**：epilepsiae_253（5b 已记录）
- **保持**：epilepsiae_1073（endpoint_defined-only）, epilepsiae_139, epilepsiae_958, yuquan_chenziyang, yuquan_wangyiyang, yuquan_zhangjiaqi, yuquan_zhaochenxi

### 3.2 H1 triple-gate 10s primary + 30s sensitivity

> Like-for-like 受 cohort 定义牵制。下面 orig 用最近 PR-7 cohort_summary，mask 用 h1_primary cohort 数字（n=8）。

| Null | metric | orig (n=9) | mask (n=8) |
|---|---|---:|---:|
| N2 main | Wilcoxon-greater p (10s) | 0.844 | **1.000** |
| N2 main | sign p (10s) | sign 3/6 → 0.78 | **1.000** |
| N2 main | median(10s) excess | −0.015 (was 30s) | **−0.037** |
| N2 main | median(30s) excess | — | **−0.020** |
| N3 robustness | Wilcoxon p (10s) | 0.891 | **0.875** |
| N3 robustness | sign p (10s) | — | **0.965** |
| **triple-gate PASS?** | | **False (NULL)** | **False (NULL)** |

方向 100% 保持（NULL stays NULL）。p-value 在小 n 上看似"更 NULL" 仅因 cohort 减 1 subject 让 Wilcoxon 跨越 ranksum boundary，**not signal change**。

### 3.3 N2 window sweep (10/30/60 min)

| Window | orig p 范围 | mask cohort medians |
|---|---:|---|
| 10 min | 0.78 | （per_subject_n2_sweep JSONs，cohort 数 NULL；详 cohort_summary.json `n2_window_sweep_per_cohort`） |
| 30 min | 0.85 | 同 NULL |
| 60 min | 0.89 | 同 NULL |

方向一致，全 NULL。

### 3.4 Burst diagnostic（Step 3.5）

| metric | orig N2 main median | mask h1_primary N2 main median |
|---|---:|---:|
| run_length_lift | 0.977 | **1.006** |
| lag1_same_excess | — | **+0.003** |

都接近 1 / 0（NULL / no persistence detected）。方向一致。

### 3.5 P3 cohort-level TOST equivalence — 3-cohort comparison

> **CRITICAL gate**：framework-locked P3 verdict 翻转 INCONCLUSIVE ↔ PASS / NULL 触发 `docs/paper1_framework_sba.md` v1.1.2 §5.3 修订；下方 like-for-like 行（orig-6 cohort）是 framework-flip gate 的真正比较。

| Cohort 定义 | n | verdict | all_main_pass | all_median_inside_band | leave_548_out_pass | any_null_window |
|---|---:|---|---|---|---|---|
| **ORIG (2026-05-01 lock, raw features)** | 6 | **INCONCLUSIVE** | False | **True** | False | False |
| **MASK like-for-like (orig 6 subjects, masked features)** | 6 | **INCONCLUSIVE** ✅ | False | **True** | False | False |
| MASK main (masked h1_primary cohort) | 8 | NULL | False | False | False | True |
| MASK broader sensitivity (all endpoint_defined) | 30 | NULL | False | False | False | True |

**判读**：
- ✅ **Like-for-like comparison (orig 6 cohort, masked features) gives identical verdict to orig**。Phantom-rank 修复在 framework-locked cohort 上**不**触发 verdict 翻转。
- ⚠ **masked-defined cohort + broader cohort both show NULL** —— 但这是 **cohort composition 变化 + n 变化 + 5b 把 548/635 (P3 cohort 的两个 "outlier driver") 翻出 fwd/rev** 共同造成的，不能归 phantom-rank。
- **framework gate decision**：**未翻转**。`docs/paper1_framework_sba.md` v1.1.2 不需修订；P3 INCONCLUSIVE-locked 仍然成立。

#### 3.5.1 T1 windows detail (orig 6 cohort → mask features on same 6 subjects)

> **数字源**：所有 median / CI / equiv_pass 都从 raw JSON 的 `tests.<window>.cohort_main.{median_obs, ci95_lo, ci95_hi, equivalence_pass}` 直读，2026-05-22 advisor consult 已比对验证。

**cohort_main (n=6, full cohort)**：

| Window | orig median | orig CI95 | orig equiv | mask median | mask CI95 | mask equiv |
|---|---:|---|---|---:|---|---|
| excess_10s | **−0.018** | [−0.125, +0.022] | False | **−0.045** | [−0.132, −0.023] | False |
| excess_30s | **−0.015** | [−0.110, +0.010] | False | **−0.020** | [−0.097, −0.012] | False |
| excess_60s | **−0.010** | [−0.095, +0.005] | False | **−0.011** | [−0.077, −0.008] | False |
| excess_1800s | **−0.0002** | [−0.0015, +0.0002] | **True** | **−0.0005** | [−0.0021, +0.0009] | **True** |
| lag1_same_excess | **−0.011** | [−0.031, +0.008] | **True** | **+0.003** | [−0.007, +0.022] | **True** |
| run_length_lift | **+0.977** | [+0.940, +1.029] | False | **+1.007** | [+0.985, +1.065] | False |

**leave_548_out (n=5)**：

| Window | orig median | orig equiv | mask median | mask equiv |
|---|---:|---|---:|---|
| excess_10s | **+0.008** | **True** ✓ | **−0.044** | False ⚠ |
| excess_30s | −0.004 | True | −0.020 | True |
| excess_60s | −0.007 | True | −0.009 | True |
| excess_1800s | +0.000 | True | −0.0004 | True |

**判读（重要 — 与之前版本相反）**：

- **方向 advisor 修正 (2026-05-22)**：phantom-rank 修过版让 short-window excess **更负 / 更远离 0**（10s −0.018 → −0.045；30s 几乎不动 −0.015 → −0.020），不是更接近 0。这意味着 **phantom-rank 在 orig 是在 mask 短窗 mark-dependence 信号**——干净数据反而把短窗的 mark-dependent 偏离暴露得更清楚。
- **但 verdict 依然 INCONCLUSIVE 没翻**：(1) cohort_main 所有 6 个 median 都仍在 ±δ=0.05 band 内 → `all_median_inside_band=True`；(2) main equivalence 不全 pass（因 CI 宽且越界）→ `all_main_pass=False`；(3) leave_548_out 10s 从 orig PASS (median +0.008 in band, equiv ✓) 翻成 mask FAIL（median −0.044 出 band）；(4) 没有任何 window 触发 NULL → 综合还是 INCONCLUSIVE。
- **lag1_same_excess 符号翻**：orig −0.011 → mask +0.003，sign flip 但 magnitude 都 ≤ 0.012 仍在 band；equiv 都 True。
- **run_length_lift 符号相对 1.0 翻**：orig +0.977（< 1, anti-persistence）→ mask +1.007（≈ 1, no persistence）；都不 equiv pass 因 CI 跨 1.05 边界。
- 1800s + lag1_same_excess 都 equiv_pass=True 与 orig 一致 → 长窗 + 1-lag 上 mark-independence equivalence 在 mask 后仍成立。

**关键科学解读**（advisor 推荐补一行）：phantom-rank 不是在 cohort-level P3 statistic 上放大 mark-independence，而是在**短窗（10–60 s）上 mask 真实的 mark-dependent 偏离**。orig 短窗看起来 "compatible with mark-independent"（median 接近 0）部分是 phantom 噪声打散的副产物；mask 后短窗 mark-dependent 信号更突出。**但 cohort-level TOST verdict 不变（INCONCLUSIVE in both）因 CI 宽 + median 仍在 band 内**——直接 framework 结论保留。如果将来 PR-7 v2 增大 power（更大 cohort 或更长 trace），短窗信号有可能 cohort-level 触及 NULL 翻转，**应在 framework v1.1.2 → v1.2 review 时单独跟踪此 directional finding**。

#### 3.5.2 为什么 cohort 扩展会得到 NULL

主要 3 因素：
1. **n 增大让 CI 收紧**：n=6 → n=30，bootstrap CI 从 wide 收到 tight；当 cohort median 是负的 −0.05 量级时，narrow CI 会把上限推到比 −δ=−0.05 还低 → equiv_pass=False，触发 any_null_window=True → NULL verdict
2. **broader cohort 包含 short-window mark-dependent subject**：1077 (excess_10s=−0.099)、442 (−0.168)、548 (−0.199)、1073 (−0.006) 等的 short-window excess 都明显负，这些 subject 单独 short-window 是 mark-dependent；orig 6 cohort 不包含这些（除 548）
3. **548 / 635 cohort 翻转**：5b 把 548 从 fwd/rev 翻 out，但 548 是 orig P3 leave-548-out PASS 的 driver；mask cohort 无 548 → leave-548-out pass path 消失

这三条与 phantom-rank 直接关联**只有**第 3 条（5b 翻转源自 phantom-rank fix）。前两条是**cohort definition × statistical power** 互相作用，**不是 phantom-rank 在 P3 statistic 上的直接效应**。

---

## 4. 判读 — Checkpoint 标准对照（5g 独立）

| Gate | 状态 |
|---|---|
| H1 triple-gate direction reversal? | ❌ **NO**（orig p=0.844 → mask p=1.000，都 NULL；direction 保持） |
| **P3 verdict flip (INCONCLUSIVE → PASS/NULL) on like-for-like cohort?** | ❌ **NO**（orig n=6 INCONCLUSIVE → mask n=6 INCONCLUSIVE，4/4 verdict flag 一致）|
| P3 verdict on broader cohort | ⚠ NULL，但归 cohort × power 互相作用，非 phantom-rank 在 statistic 层面的直接效应 |
| N0–N3 surrogate hierarchy reversal? | ❌ **NO** |
| Burst diagnostic direction reversal? | ❌ **NO**（run_length_lift 0.977 → 1.006，都 ≈ 1）|
| N2 window sweep direction reversal? | ❌ **NO**（10/30/60 min 都 NULL）|
| **Framework-level revision triggered?** | ❌ **NO** |

**Step 5g 整体方向**：✅ PASS — 主结论方向全保持，framework-level 红线未触及。

---

## 5. 主文档调整建议

### 5.1 Topic 1 §2 PR-7 行（已部分落字，需补 phantom-rank 行）

加 1 句：
> "**Phantom-rank 修过版 (2026-05-22) 验证**：在 orig P3 cohort (n=6) 用 masked features 重跑，verdict = INCONCLUSIVE 完全保持；on broader masked-defined cohort (n=8 或 n=30) verdict 形式上滑到 NULL，但这是 cohort × power 互相作用而非 phantom-rank 在 statistic 上的直接效应。`docs/paper1_framework_sba.md` v1.1.2 不修订。"

### 5.2 Topic 0 §5 row 5g

从 "🟡 重跑中" 改为 "**已完成 2026-05-22**" + 上面 §1 一句话 verdict。

### 5.3 Topic 0 二级 archive `rerun_results_2026-05-21.md` §3.7

把 PR-7 placeholder 替换成真实数字 + §4 reconcile 表 PR-7 行真实数字。

### 5.4 Framework doc `topic4_sef_itp_framework.md` header

把 PR-7 行从 "🟡 5g 重跑进行中" 改为 "✅ P3 verdict on like-for-like cohort 保持 INCONCLUSIVE"。

### 5.5 论文叙事保留

- P3 "compatible with mark-independent within tested precision" 在 like-for-like cohort 上的措辞不变
- broader cohort 的 NULL verdict 作为 sensitivity caveat 提一句，**不**作为主结论翻转

---

## 6. 工件清单

新生成（masked）：
- `results/interictal_propagation_masked/template_pairing/pr7_cohort_audit.csv`（40 候选 audit）
- `results/interictal_propagation_masked/template_pairing/per_subject/*.json`（30 个 all_eligible）
- `results/interictal_propagation_masked/template_pairing/per_subject_burst/*.json`（10 文件：8 h1_primary + 548 + 635）
- `results/interictal_propagation_masked/template_pairing/per_subject_burst_all30_eligible/*.json`（30 preserved as sensitivity）
- `results/interictal_propagation_masked/template_pairing/per_subject_n2_sweep/*.json`（30×3=90 files）
- `results/interictal_propagation_masked/template_pairing/cohort_summary.json`
- `results/interictal_propagation_masked/template_pairing/pr7_addendum_p3.json` (main = mask h1, NULL)
- `results/interictal_propagation_masked/template_pairing/pr7_addendum_p3_orig6_cohort.json` (**like-for-like, INCONCLUSIVE preserved**)
- `results/interictal_propagation_masked/template_pairing/pr7_addendum_p3_all30_eligible.json` (broader sensitivity, NULL)

代码：
- `scripts/run_pr7_template_pairing.py` (_apply_masked_paths + --masked-features)
- `scripts/pr7_addendum_p3_equivalence.py` (argparse main + _apply_masked_paths + --masked-features)
- `scripts/plot_pr7_template_pairing.py` (argparse + _apply_masked_paths + --masked-features)
- `tests/test_pr7_masked_path_routing.py` (3 项 smoke test)

日志：
- `logs/step5g_pr7_audit_masked.log`
- `logs/pr7_persubject_masked/<sid>.log`（29 个 per-subject 日志）
- `logs/step5g_pr7_burst_masked.log`（all-30 burst）
- `logs/step5g_pr7_burst_h1primary_masked.log`（h1 cohort burst rerun）
- `logs/step5g_pr7_cohortstats_masked.log`
- `logs/step5g_pr7_n2sweep_masked.log`
- `logs/step5g_pr7_p3_masked.log`（all-30 P3, verdict NULL — sensitivity）
- `logs/step5g_pr7_p3_h1primary_masked.log`（h1 P3, verdict NULL）

---

## 7. 一句话总结

**PR-7 在 phantom rank 修过后所有 primary 方向保持**：H1 triple-gate cohort-level NULL 不变；**P3 verdict on like-for-like orig cohort (n=6) 保持 INCONCLUSIVE 完全不动 (4/4 flag 一致)**；broader cohort verdict 滑到 NULL 是 cohort × power 互相作用，**不是 phantom-rank 在 statistic 层面的直接效应**。**framework-level revision 未触发**，`docs/paper1_framework_sba.md` v1.1.2 不动。可以进 5i.6 default flip + Checkpoint B 正式 advisor consult。
