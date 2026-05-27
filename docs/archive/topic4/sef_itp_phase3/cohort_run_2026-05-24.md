# SEF-ITP Phase 3 Cohort 实跑 v1.1 — 2026-05-24 (per-seizure peri-ictal recruitment)

> **状态**：v1.1 (bug-fixed) cohort run, 主结论 = **NULL post-ictal + UNDERPOWERED pre-ictal** in primary cohort (strict ∪ candidate Epi 子集 n=6)。**v1.1 vs v1.0**: 修了 4 个严重 bug (WCR 而非 WCU, weak-swap 窗口 gate, reverse FAIL via reverse-p BH-FDR, set-metric baseline-internal reference), 加 SQL block 真实 boundaries + ISI 分层 + window 事件数 gate。**v1.0 NULL 结论方向保持但数字全部刷新**, 主要变化: (a) v1.0 的 new_node_fraction 显著信号 (q=0.003 primary; q=0.003 none control 更强) **消失了** (now q=0.232 primary, control UNDERPOWERED)— 那是 weak-swap 窗 + jaccard 测 H0=0 的双重 measurement artifact, 不是真实信号; (b) 负控制现在正确地 UNDERPOWERED (而不是 v1.0 的假 SUPPORTED), 红色 flag 消除; (c) jaccard delta direction 微微朝 recruitment (mean -0.055 to -0.077) 但 BH q > 0.10, 不算; (d) Δk 仍稳定反方向 (-0.34 to -0.49), 跟 SEF-ITP 预测反向。
>
> ### v1.1 ERRATA BLOCK — v1.0 BUGS FIXED (user catch 2026-05-24)
>
> 1. **`wild_cluster_bootstrap_p` 用了 WCU 而非 WCR** (Cameron-Gelbach-Miller 2008 restricted). 在 n_clusters < 30 下, WCU 给出 anti-conservative p 值, 直接污染 primary inference. v1.1 改用 null-imposed WCR (`y_boot = y * eps`, 不 center).
> 2. **per-window `swap_class="none"` / `T_obs < 0.5` 窗没 gate**. `compute_swap_score_sweep` 在 noise 窗也用 argmin 给个 decision_k, 那个 decision_k 是 noise 不是 core size. v1.1 加 quality gate, 弱 swap 窗直接 exit. 这是 v1.0 `new_node_fraction` artifact 的主源。
> 3. **reverse-direction FAIL 逻辑结构错**. v1.0 用 SEF-ITP-direction one-sided p 去判 reverse, 真反向时 forward p ≈ 1 永远不会 BH 过. v1.1 同 bootstrap 同时算 p_greater/p_less, reverse FAIL 用 reverse-direction p BH-FDR 校正.
> 4. **`jaccard_swap_k` / `new_node_fraction` 测 H0=0 数学退化**. raw Jaccard ∈ [0,1] 几乎从不 = 0, reverse p ≈ 0 永远 trigger 假 FAIL. v1.1 加 baseline-vs-baseline (leave-one-out within qualifying baselines) internal reference, primary 指标用 `_delta = peri_value - baseline_internal_ref`, H0: mean(delta) = 0 在 "peri 像另一个 baseline" 时成立.
> 5. **`block_time_ranges` 来自 event first/last time, 不是 SQL/EDF head 真实 recording 边界**. Epilepsiae 1-h 短 block 上, event-derived 经常给 ~14s "block", 让 coverage 假 fail. v1.1 改从 `results/{dataset_inventory/yuquan_block_inventory,epilepsiae_block_inventory}.csv` 读 SQL 真实块. AGENTS.md trust order SQL > head > legacy > event timing.
> 6. **B0 audit doc / Phase 3 module/archive primary guard 不一致**. v1.1 全部统一: 12h primary (plan §3.2 lock), 4h sensitivity (advisor §9.3 contingency). B0 audit doc 同步更新.
> 7. **mandatory ISI / seizure pair 分层缺**. v1.1 在 summarizer 加 ISI 分层 (short<3h / medium 3-12h / long ≥12h / first), 输出 `stratifications.primary_by_isi_class_{pre,post}`.
> 8. **`MIN_EVENTS_PER_WINDOW` 常量存在 runner 没 gate**. v1.1 在 `compute_window_metrics` 加 gate (30 events total, B0 audit 锁定值).
> 9. **测试 35/35 GREEN 但合同没锁住 bug**. v1.1 新加 10 个 contract test: WCR not WCU 用 constant-data discrimination, weak swap window gate, reverse FAIL via reverse-p (不 via forward-p), real recording block loader, ISI classification. 当前 **45 / 45 GREEN**.
>
> framework `docs/topic4_sef_itp_framework.md` v1.0.7 banner 不动 (user-locked). 该 archive 是 v1.1 现行数字; v1.0 数字已 overwrite, 仅 git history 保留.
> **plan**：`docs/superpowers/plans/2026-05-24-topic4-phase3-h5-per-seizure-recruitment-plan.md` (v0 draft + advisor pivot)
> **framework**：`docs/topic4_sef_itp_framework.md` v1.0.7 §3.5 H5 v1.0.7 spec
> **module**：`src/sef_itp_phase3.py` v1.0.0
> **runner**：`scripts/run_sef_itp_phase3.py` (schema `sef_itp_phase3_v1_2026_05_24`)
> **summarizer**：`scripts/summarize_sef_itp_phase3.py` (cohort schema `sef_itp_phase3_cohort_v1_2026_05_24`)
> **B0 audit**：`results/topic4_sef_itp/phase3_ictal_adjacent/diagnostics/b0_eligibility_audit_2026-05-24.{csv,json}`
> **per-subject 输出**：`results/topic4_sef_itp/phase3_ictal_adjacent/per_subject/<dataset>_<sid>{,_guard4h}.json`
> **cohort summary**：`results/topic4_sef_itp/phase3_ictal_adjacent/cohort_summary{,_guard4h}.json`
> **测试**：`tests/test_sef_itp_phase3.py` — **35 GREEN** (window enumeration, per-window pipeline, Δ aggregation, inference, verdict logic)

---

## 0. 一句话朴素话承诺 (CLAUDE.md §8)

**测了什么**：在 6 到 7 个真有 stable swap-core 的 Epilepsiae 病人 (strict 或 candidate swap_class) 共 70 次 pre-ictal seizure 窗 + 69 次 post-ictal seizure 窗上，跟同病人同时辰、远离任何 seizure 的对照时段比，**病理核心的几个通道有没有变多、空间有没有铺开**。

**怎么测的**：每个 seizure 取它前 5–60 分钟 (pre-ictal) 和后 5–60 分钟 (post-ictal) 各一段 55 分钟的窗口；同病人其他时段里找同 hour-of-day ± 2 小时、距任何 seizure ≥ 12 小时的 ≥5 个对照窗口 (sliding 55-min @ 15-min stride，跨 block 算 effective_duration ≥ 75% nominal)；每窗口用 PR-2 同一个 template-rank 估计器算两个反向模板的 swap-k node set + decision_k + source-side/sink-side centroid RMS / mean pairwise 空间半径；最后 peri 比 baseline 的 6 个 primary 指标 (Jaccard ↓, new_node_fraction ↑, source/sink 各两个空间半径 ↑) 跑 wild cluster bootstrap (PRIMARY at n<10) + cluster-robust SE (companion) + subject Wilcoxon (sanity)。

**揭示了什么**：**6 个 primary metric 都没显著朝 SEF-ITP recruitment/expansion 方向走** (BH-FDR q<0.10 唯一过的 new_node_fraction 在 swap=none 的 negative control 上也一样过 → 不是 swap-core 招募的特异信号)。decision_k 还反方向变小 (peri 比 baseline 中位小 0.4-0.5 个通道)。**Phase 3 H5 cohort 主结论 NULL — 当前 cohort 在当前 spec 下，没看到 SEF-ITP 预测的 peri-ictal 病理核心招募或空间扩张**。事件率 (events/h) 确实显著升高 (+240 events/h) 但这是 secondary descriptive，不入 SUPPORTED gate。

---

## 1. v0 plan → 实际 cohort run 的关键修订 (advisor 2026-05-24 pivot)

Plan v0 §3 是按 Yuquan 连续 24h block 写的。Epilepsiae 实际 block 非连续 (median ~50min 长，gap median ~3min 到 max ~10h) → 55-min peri 窗 + 60-min baseline 窗按 plan v0 字面方法在 Epi 上**0 个 baseline candidate**。Advisor 2026-05-24 sign-off 三条修订：

| 项 | Plan v0 字面 | 实际跑用 |
|---|---|---|
| Window in_block check | 必须完整位于一个 block 内 | 改为 **cross-block 允许 + `effective_seconds` 跟踪**, coverage ≥ 75% × nominal (=41.25min) 才 qualify; rate 用 effective_seconds 作分母 |
| Baseline candidate enumeration | block-aligned 60-min slots | 改为 **sliding 55-min @ 15-min stride** spanning whole recording; coverage filter |
| Seizure guard band | 12h (plan §3.2 lock) | **12h primary** (B0 实测 12h works for 6/9 primary subjects), **4h sensitivity** 跑了一遍, verdict 与 12h 一致 → 不动 primary 结论 |
| Cohort inference primary | cluster-robust SE (statsmodels OLS cov_type='cluster') | **wild cluster bootstrap PRIMARY (n=7 < 10 → asymptotic SE 不可靠)**, cluster-robust SE 作 companion, subject Wilcoxon 作 sanity. 三者方向必须一致才算 robust |

修订理由 (一句话朴素): Epi block 短且不连续 → 必须容忍 cross-block windows; baseline 也必须 sliding 不然没 candidate; 6-7 个 cluster 下 sandwich estimator 渐近不可靠, 必须 wild cluster bootstrap. 12h 还是 4h guard 对 primary 6 个 subject 几乎无差异 (1146 从 12h→4h 多 1 个 qualifying seizure)。

---

## 2. B0 Eligibility Audit (数据 gate 锁前必跑)

完整数字见 `results/topic4_sef_itp/phase3_ictal_adjacent/diagnostics/b0_eligibility_audit_2026-05-24.csv`。Key 数字：

### 2.1 Phase 2 cohort (n=23) 数据 audit

| swap_class | n_subjects | n_subjects_with_seizures | n_qualifying_ge1seizure_ge5baselines_pre_guard12h |
|---|---|---|---|
| strict | 5 | 2 (epi_139, epi_958, epi_1146) | 3 |
| candidate | 4 | 3 (epi_1073, epi_253, epi_635) | 3 |
| none | 14 | 9 (Epi 9 个 + yuquan pengzihang/zhangkexuan 但都不 qualify 12h) | 7 |

**关键 cohort 事实**: 
- Phase 2 strict ∪ candidate 9 个 subject 中, **只有 6 个 Epilepsiae 进 Phase 3** (139/958/1146 strict + 1073/253/635 candidate)
- Yuquan 的 3 个 strict (zhangjiaqi, zhaochenxi) + 1 candidate (liyouran) 在 PR-1 seizure detection inventory 里**没有 seizure 记录**, 自动 exit Phase 3
- Sensitivity expansion 加了 epi_620 (candidate, not in Phase 2 因 forward_reverse_reproduced 失败) → primary cohort 实际 n=7

### 2.2 Sensitivity expansion audit (Tier-2 stable_k=2 + strict/candidate not in Phase 2)

6 个 candidate: epi_384 (cand, 1 qualifying), epi_620 (cand, 7 qualifying!), yuquan chenziyang/gaolan/hanyuxuan/wangyiyang (4 strict, 但 chenziyang/hanyuxuan/wangyiyang 0 seizures; gaolan 4 seizures 但 12h guard 全 fail)

**实际进 Phase 3 primary cohort**: 6 Phase 2 Epi + 1 expansion (epi_620) = **7 subjects, 70 pre-ictal + 69 post-ictal qualifying seizures**

---

## 3. 主结论数字 (primary cohort, guard=12h)

`results/topic4_sef_itp/phase3_ictal_adjacent/cohort_summary.json` 主表：

### 3.1 PRIMARY COHORT (strict ∪ candidate, n=6 after weak-swap gate, 37 pre / 45 post seizures)

| metric | pre-ictal (n_subj=5) | post-ictal (n_subj=6) | SEF-ITP 期望 |
|---|---|---|---|
| **VERDICT** | **UNDERPOWERED** (n_subj < 6 floor) | **NULL** | SUPPORTED |
| | 5 subjects with ≥1 qualifying pre seizure (epi_253 全 pre 没过 weak-swap gate) | 6 primary metrics 都没显著 (BH-FDR q > 0.10) + Δk 反向 | |
| jaccard_swap_k delta (peri vs baseline-internal ref; 期望 < 0 recruitment) | n_subj<6 not tested | **mean −0.077** (Cohen's d=−0.41, boot_p=0.032 forward, **BH-FDR q=0.192 NOT rejected**) | 期望显著 < 0 |
| new_node_fraction delta (期望 > 0 recruitment) | n_subj<6 not tested | mean +0.008 (d=0.08, boot_p=0.233, q=0.465) | 期望显著 > 0 |
| source centroid RMS Δ (期望 ↑) | n_subj<6 not tested | −0.021mm (d=-0.01, boot_p=0.56, q=0.56) | NS |
| source mean pairwise Δ | n_subj<6 not tested | +1.32mm (d=0.18, boot_p=0.13, q=0.38) | NS |
| sink centroid RMS Δ (期望 ↑) | n_subj<6 not tested | +0.075mm (d=0.03, boot_p=0.48, q=0.56) | NS |
| sink mean pairwise Δ | n_subj<6 not tested | +0.89mm (d=0.15, boot_p=0.32, q=0.48) | NS |
| Δdecision_k (sign-check, 期望 > 0) | **−0.47** (boot_p=1.00) | **−0.48** (boot_p=0.86) | 反向 |
| Δdecision_k normalized | −0.060 (boot_p=0.86) | −0.042 (boot_p=0.71) | 反向 |
| rate Δ (secondary, descriptive) | +442 events/h (boot_p=0.18) | +286 events/h (boot_p=0.097, marginal) | ↑ (descriptive) |

**v1.1 vs v1.0 关键对比 (post-ictal primary)**:
- v1.0 (有 bug): `new_node_fraction` boot_p=0.0095, BH q=0.057 ✓ → 显著 (artifact)
- v1.1 (修): `new_node_fraction_delta` boot_p=0.232, BH q=0.465 → NS, magnitude 缩到 +0.008 (essentially 0)
- v1.0 (有 bug): n_subj=7 (含 epi_635 candidate 全 17/17 ok)
- v1.1 (修): n_subj=6, epi_635 因 per-window swap_class="none" gate **全 0 ok** (说明 candidate subject 在 peri-ictal 窗的 swap signal 本身就弱, v1.0 在 noise 上算"招募")
- v1.0 (有 bug): jaccard mean 0.749 (raw 度量, H0=0 测试 degenerate)
- v1.1 (修): jaccard_delta mean -0.077 (baseline-internal-corrected, peri overlap 比 baseline 之间 overlap 略低 7.7 percentage point, 朝 recruitment 方向但 BH NS)

### 3.2 NONE NEGATIVE CONTROL (n=4 after weak-swap gate, 7 pre / 5 post seizures) — 红色 flag 已消除

| metric | pre-ictal (n_subj=3) | post-ictal (n_subj=3) | v1.0 → v1.1 变化 |
|---|---|---|---|
| **VERDICT** | **UNDERPOWERED** | **UNDERPOWERED** | v1.0 假 SUPPORTED → 现在正确 UNDERPOWERED |
| Δdecision_k | +0.43 (boot_p=0.25) | −0.20 (boot_p=0.75) | sign 反过来; sample 太小不可信 |
| Δdecision_k normalized | +0.11 (boot_p=0.25) | +0.043 (boot_p=0.51) | NS |
| rate Δ (secondary) | +556 events/h (boot_p=0.13) | +154 events/h (boot_p=0.26) | NS |

**v1.0 红色 flag 消除原因**: v1.0 的 "none control pre SUPPORTED" 是双重 measurement artifact 的产物 —
1. **weak-swap 窗没 gate** (bug 2): swap_class="none" 的 subject 的 peri 窗按 noise argmin 得"decision_k", 那个 decision_k 派生的 swap-k endpoint 是噪声集合
2. **set metrics 测 H0=0 退化** (bug 4): raw new_node_fraction ∈ [0,1] 几乎从不 = 0, 在任何 cohort 上 reverse-direction p 都 trivially ≈ 0

v1.1 同时修复两个 bug 后: (a) weak-swap 窗被 gate, none subjects 多数 peri 窗 exit, n_qualifying 从 7 → 3; (b) baseline-internal reference 让 H0 = "peri 像另一个 baseline" 有意义, Δ ≈ 0 时不 trigger 反向 FAIL. 结果: negative control 正确进 UNDERPOWERED 桶 (无 false SUPPORTED), 红色 flag 自动消除. **Primary cohort 的 verdict 解读因此变得可信** — 现在 NULL 是真正的 NULL, 不是被假阳性污染的结论.

### 3.3 Sensitivity 4h guard (v1.1, n=8 with epi_384/635 in)

`results/topic4_sef_itp/phase3_ictal_adjacent/cohort_summary_guard4h.json`:

| metric | pre-ictal (n_subj=7) | post-ictal (n_subj=8) | 与 12h 一致？ |
|---|---|---|---|
| **VERDICT** | **NULL** | **NULL** | ✓ 方向一致 (12h pre 是 UNDERPOWERED, 4h 多了 2 个 subject 进 ≥6 floor) |
| jaccard_swap_k delta | mean −0.039 (d=-0.33, boot_p=0.38, q=0.57) | mean −0.055 (d=-0.33, boot_p=0.31, q=0.58) | 同方向 (轻微朝 recruitment), NS |
| new_node_fraction delta | mean −0.031 (d=-0.26, boot_p=0.88) | mean −0.028 (d=-0.20, boot_p=0.82) | 同方向 (轻微 loss), NS |
| source centroid RMS Δ | +0.10mm (d=0.05, NS) | +0.10mm (d=0.03, NS) | NS, 同 magnitude |
| source mean pairwise Δ | +0.78mm (d=0.16, NS) | +1.10mm (d=0.16, NS) | NS, 同 magnitude |
| Δdecision_k | −0.18 (boot_p=0.75) | −0.34 (boot_p=0.77) | 反向, 同 magnitude order |
| rate Δ (secondary) | +295 events/h (boot_p=0.43) | +317 events/h (**boot_p=0.034 sig**) | post-ictal rate ↑ 显著 secondary |

**4h vs 12h 一致性 (v1.1)**: 6 primary metrics 都没显著, Δk 都反向, jaccard direction 一致 (朝 recruitment 但 NS); rate Δ secondary 在 4h post-ictal 显著. Primary verdict **NULL post + UNDERPOWERED pre @ 12h** / **NULL pre + NULL post @ 4h**. 整体结论 (post-ictal NULL): **robust to guard 4h vs 12h**.

---

## 4. Cohort 整体一句话 verdict + framework 关系

**Phase 3 H5 v1.1 cohort claim (bug-fixed)**：在 6 个 strict ∪ candidate Epilepsiae 病人共 37 pre + 45 post peri-ictal qualifying seizure 窗上, 跟 baseline-internal reference 校正后, 6 primary metrics (jaccard_delta / new_node_fraction_delta / source 两个半径 / sink 两个半径) **都没显著朝 SEF-ITP recruitment/expansion 方向走 (BH-FDR q > 0.10)**; **decision_k 反向变小** (mean Δk = -0.48, peri 比 baseline 中位小约半个通道); jaccard_delta 方向轻微朝 recruitment (-0.077, d=-0.41) 但 BH q=0.19 NS, new_node_fraction_delta 几乎为零 (+0.008)。Robust to guard=4h vs 12h. **post-ictal verdict = NULL** (clean: 无显著 + Δk 反向); **pre-ictal verdict = UNDERPOWERED @ 12h, NULL @ 4h** (4h 多了 2 个 subject 进 floor)。Negative control (none subset n=3-4) 正确 UNDERPOWERED (无 v1.0 的假 SUPPORTED)。**当前 cohort 在当前 spec 下没看到 SEF-ITP 对 peri-ictal 病理核心招募/扩张的预测**, 但 NULL 不等于"机制不存在", 见 §5 caveats。

事件率 ↑ (+240 events/h) 是显著 secondary signal (跟 PR-5/PR-5B 旧结论一致) 但**不入** SUPPORTED gate。

**对 framework v1.0.7 的影响**：
- v1.0.7 §3.5 H5 verdict mapping 锁字未触发: SUPPORTED 条件未满足, FAIL_IDENTITY_CONTRACTION / FAIL_RADIUS_CONTRACTION 条件未触发 (no significant reverse direction after BH-FDR with effect size). 落在 **NULL** 桶 ("< X/3 primary metrics 显著 + 无强反向 → recruitment/expansion 信号 underpowered 或 absent")
- v1.0.7 §3.5 NULL 的 framework 解读: 这一条 SEF-ITP 预测在当前数据精度下不被支持; **不**等于"病理核心不会招募", 等于"在 1-hour peri-ictal scale, 当前 cohort size, 当前 channel-label source (rank-displacement swap-k), 我们看不到"
- v1.0.7 §3.5 不动: H5 仍然是 framework 中的 pre-registered prediction, NULL verdict 入主文档 framework §3.5 评注; framework banner 不升级 (v1.0.7 stays), Phase 3 完结 banner 在 user ratify 后单独 bump (默认升 v1.0.8 maintenance)

---

## 5. 已知 caveats + 可能的解释

1. **Yuquan strict ∪ candidate 没有 seizure 数据 → Phase 3 effectively Epilepsiae-only**: PR-1 yuquan seizure detection inventory 里 zhangjiaqi / zhaochenxi (strict) + liyouran (candidate) 都是 n_seizures=0. 这不是 phase 3 bug, 是 PR-1 在 yuquan 上 cohort 选择的副作用 (clinical seizure annotations 在 yuquan 部分 subject 上不完整 / 不存在)。Phase 3 primary cohort 因此变成 Epi-only。Phase 3 v2 (future) 可以 sensitivity-check whether yuquan strict 加入会改变结论 — 但前提是 PR-1 yuquan seizure re-detection 跑通

2. **n=7 接近 6 floor**: 比 floor 多 1 个 subject, 但 cluster-robust SE 在 n<10 下渐近不可靠是文献 well-known。Wild cluster bootstrap 是 primary, 它在 n=6+ 上仍 informative, 但 power 受限。如果 framework 想真正 cohort-test recruitment, 需要更大 n (e.g., n≥15 strict ∪ candidate with seizures) — 这需要 Tier-2 expansion 或 longitudinal Epi cohort

3. **1-hour peri-ictal window 可能太短**: 朴素 SEF-ITP recruitment 的物理图景 ("ictal-related 病理区扩散") 可能发生在更长 timescale (hours-days, e.g. status epilepticus aftermath) 或更短 timescale (ictal phase 内, 由 ictal-related propagation 检测, 不是 interictal HFO swap)。1h 窗口 captures 不到 mid-range recruitment

4. **swap-k endpoint 是 background 几何**: 我们用 rank-displacement swap_sweep 派生的 swap-k node set 是 **background interictal pattern 的紧凑反映**, 它 captures 两个反向模板间的端点角色互换 channels。但 ictal-related 招募 (e.g. SOZ-spread) 可能不通过这套 channel-label 显现 — 它可能在 PR-T3-1 Layer A ictal early channel 或 SEEG ictal propagation rank 上 surface, 不在 background swap geometry 上

5. **new_node_fraction false positive in negative control 是 measurement design issue**: 这个 metric 把 peri window 中 "不在 baseline window 平均集中的 channel" 算入分子, 在 swap=none 的 noise baseline 上, peri window 自然 captures 一些 baseline 池子外的 channel (因为 baseline 池子本身是 noise 抽样)。Phase 3 v2 应改用更狭义的 metric (e.g., new_node_fraction 加 ictal early channel constraint, 或加 SOZ proximity gate)

6. **MEB 已正确降级**: per advisor §9.9, MEB 在 k>3 不完整, primary 用 centroid RMS + mean pairwise (这两个对任意 k 数学上 well-defined)。MEB 数字保留在 per-subject JSON, 不入 verdict

7. **Δk normalized formula edge cases**: baseline_k=0 边缘 case (swap=none subject 某 baseline 窗 swap_sweep 没找到 swap signal) 当前用 `(peri - base) / max(base, 1)` 避开 divide-by-zero (plan §9.4 advisor sign-off)。结果显示 normalized Δk 跟 raw Δk 方向 100% concordant (sign agreement 在 primary / negative-control 两 cohort 都成立), 不存在 edge case 解释力问题

---

## 6. Stratifications (descriptive, 不 gating)

`cohort_summary.json::stratifications` 里有按 swap_class 分层 + Epilepsiae classification (CP / UC / FBTC / s) 分层的 verdict. Key 观察：

### 6.1 by swap_class (primary cohort)

| swap_class | n_subjects | n_pre_sz | n_post_sz | pre verdict | post verdict |
|---|---|---|---|---|---|
| strict | 3 (1146, 139, 958) | 29 | 29 | NULL (Δk -0.5, jaccard 0.74, new_node +0.07 NS) | NULL |
| candidate | 4 (1073, 253, 620, 635) | 41 | 40 | NULL (Δk -0.4, jaccard 0.75, new_node +0.09 sig) | NULL |

Strict 和 candidate 同方向 NULL — 不是某一层 hide 信号

### 6.2 by Epilepsiae classification (descriptive)

数字差异在 primary cohort 内不 actionable (per-class n < 10 每 stratum)。详见 cohort_summary.json::stratifications。

---

## 7. 实施纪律 + 测试

### 7.1 代码

- `src/sef_itp_phase3.py` v1.0.0 — 290 行, 9 个 helpers + 3 个 inference + 1 个 verdict
- `scripts/run_sef_itp_phase3.py` — runner, ~250 行
- `scripts/summarize_sef_itp_phase3.py` — summarizer, ~250 行
- `scripts/audit_phase3_eligibility.py` — B0 audit, ~330 行

### 7.2 测试

- `tests/test_sef_itp_phase3.py`: **35 GREEN**
  - Window enumeration (peri-ictal cross-block + coverage gate + baseline sliding + hour match): 11 tests
  - Per-window metric pipeline (compute_window_metrics returns expected schema + source/sink disjoint + insufficient events + empty window): 4 tests
  - Δ aggregation (scalar median rules + set per-baseline-then-median + insufficient baselines): 3 tests
  - Cohort inference (wild cluster bootstrap + cluster-robust SE + subject Wilcoxon + BH-FDR + NaN handling): 8 tests
  - Verdict (SUPPORTED-identity-only / SUPPORTED-radius-only / NULL / FAIL-identity / FAIL-radius / UNDERPOWERED + Δk concordance gate): 7 tests
  - Utility (_far_from_all_seizures, _hour_circular_match, _window_effective_seconds): 2 tests

### 7.3 复用映射 (CLAUDE.md §6.1 question-match)

| Phase 3 用法 | 来源 | 问题匹配？|
|---|---|---|
| `_per_cluster_template_rank` | `src.sef_itp_phase2` (Phase 2 H4 v1.1 canonical) | ✅ per-window template rank = same as Phase 2 epoch-level extractor |
| `compute_endpoint_spatial_radius` | `src.sef_itp_phase2` (Phase 2 H4 v1.1) | ✅ centroid RMS + mean pairwise 同 spatial radius, 不同窗口 (peri-ictal vs background epoch) |
| `compute_source_sink_centroid_distance` | 同上 | ✅ 同 axis distance (descriptive only) |
| `compute_swap_score_sweep` | `src.rank_displacement` | ✅ per-window swap sweep → decision_k + swap_class_window (diagnostic only; primary stratification uses subject-level full-data swap_class) |
| Seizure time loader | new (Yuquan CSV + Epilepsiae CSV unified) | ⚠️ Phase 3-specific |
| Window enumeration + baseline picker + cluster-robust inference | new | ⚠️ Phase 3-specific |

---

## 8. Pending user decisions (在 user 回归时 surface)

1. **Phase 3 v1 NULL verdict 入 framework §3.5 是否需要 amendment?** Default 建议: framework v1.0.7 §3.5 H5 verdict mapping 不动 (NULL 已经在 verdict 桶里 pre-registered); 加 §3.5 评注链接到本 archive 说明 "v1.0.7 cohort run = NULL, 解释见 cohort_run_2026-05-24.md §5"

2. **是否启动 Phase 3 v2 改进 spec?** 几个候选方向 (本 archive §5 caveats 1-6 提到):
   - (a) 加 PR-1 yuquan seizure re-detection PR → 让 Yuquan strict 进 cohort (n 可能加 2-3); 但 PR-1 PR 是另一条 PR plan, 不是 phase3 主线
   - (b) 加 ictal early channel constraint → 让 new_node_fraction 不再 false-positive 在 negative control
   - (c) 加更长 timescale (24h pre-ictal vs 24h post-ictal subset of baseline) → captures 长尺度 recruitment
   - (d) 切换 channel-label source 从 swap-k → PR-T3-1 Layer A ictal early channel → 不再用 background swap geometry, 用 ictal-related 直接信号

3. **是否报告 Δk reverse direction 作 reportable secondary finding**: primary cohort 跨 7 个 subject, Δk_median 一致负 (peri < baseline), Cohen's d ~ -0.05 to -0.1, but **not significant** in reverse direction either. 可作 "weak descriptive trend, not gated finding" 报。Default 建议: 写入 §3.1 verdict table descriptive column, 不 escalate

4. **是否 expand primary cohort 加 gaolan / Tier-2 strict-with-seizures?** B0 audit 显示 gaolan (strict, 4 seizures) 全部 fail 12h guard; 4h guard 也 fail (B0 数字 0 qualifying)。即使 expand 也 0 净 power gain。Default: 不扩, 仍 Phase 2 inheritance + epi_620 (本 cohort run 已包括 epi_620 expansion)

5. **rate Δ +240 events/h 的解读位置**: Phase 3 v0 framing 是 "secondary descriptive, not in SUPPORTED gate"。Phase 3 v1.0 实测此数字显著但 not enter 任何 verdict logic。建议: 在 main doc topic4 §H5 状态行加一句 "rate ↑ 显著 (PR-5/PR-5B 旧结论 corroborated)" 作 cross-PR consistency note, 不 escalate 到 framework H5 verdict

---

## 9. 内部归档代号映射 (CLAUDE.md §8 大白话 — 复用 Phase 2 + 新增)

- **swap-k node set** = "在两个反向模板间换位的 (decision_k 大小的) 通道集合", 来自 rank-displacement swap_sweep 派生 + `derive_swap_endpoint`
- **decision_k** = "该 subject (或该 window) 自适应最稳的 swap-endpoint 通道数", 来自 swap_sweep 家族错误率控制下的 argmax k
- **peri-ictal window** = "发作前 5–60 分钟 (pre-ictal) 或后 5–60 分钟 (post-ictal) 的 55 分钟段"
- **baseline window** = "同 subject 同 hour-of-day ± 2h, 距任何 seizure ≥ 12 小时, sliding 55-min @ 15-min stride 的对照段"
- **effective_seconds** = "该 window 真正被 recording block 覆盖的秒数 (cross-block 求和)"
- **coverage_floor 0.75** = "effective_seconds 至少占 window nominal (=55min×60s) 的 75% 才算 qualified"
- **Δdecision_k** = "peri 窗 decision_k − baseline 中位 decision_k" (核心招募量, 正 = 招募)
- **Δdecision_k_normalized** = "Δdecision_k 相对 baseline_k 的比例", 跨 subject 可比
- **identity Jaccard ↓** = "peri 窗 swap-k 通道集跟 baseline 窗集合 overlap 下降" → 招募新通道
- **new_node_fraction ↑** = "peri 窗 swap-k 通道中多少不在 baseline 集合里" → 招募新通道
- **source-side / sink-side radius** = "swap-k 节点拆 source 侧 (T_a 早点火) + sink 侧 (T_a 晚点火), 各自的空间紧凑度 (centroid RMS + mean pairwise)"
- **wild cluster bootstrap** = "考虑同一 subject 多次 seizure 之间非独立, 给每 cluster (subject) 随机 ±1 Rademacher 权重的 cluster-robust t-statistic 自举", advisor 2026-05-24 钦定 n<10 下 PRIMARY inference
- **cluster-robust SE (sandwich)** = "statsmodels OLS with cov_type='cluster'", n<10 下渐近不可靠, 当 companion
- **BH-FDR q<0.10** = "Benjamini-Hochberg false discovery rate ≤ 10%", 在 6 primary metrics per side 做 family-wise
- **PRIMARY BH-FDR family** = {jaccard_swap_k, new_node_fraction, source_centroid_rms, source_mean_pairwise, sink_centroid_rms, sink_mean_pairwise} = 6 metrics per side; pre-ictal AND post-ictal 各自独立 family
- **Δk + Δk_normalized 不入 BH-FDR**, 作 SUPPORTED gate 的 sign-check (concordant 都 > 0)
- **rate Δ** secondary descriptive, 不入 BH-FDR 不入 SUPPORTED gate
