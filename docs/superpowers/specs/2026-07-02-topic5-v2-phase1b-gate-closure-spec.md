# Topic 5 V2 Phase 1b — Gate-closure + broad↔narrow ablation spec (LOCK BEFORE RUN)

date 2026-07-02 · 状态：spec（运行前锁定，待 user sign-off）· 前身：Phase-1 backbone（`docs/archive/topic5/v2_phase1_band_scan_backbone_2026-07-02.md`）

> **目的**：把 Phase-1 dev 结果从"weak/global-null 下描述性为正、formal Gate A 未可评估"推进到**可判读的形式化结论**——但**只在锁定的 null hierarchy + 正确 cohort 推断下**，且**不升级 tier**（保持 candidate scaffold / broadband-recruitment；禁 timing-order mechanism / critical mode）。核心科学问题：**narrow/core HFO 几何是否标记一个跨频带共享的 early-ictal recruitment scaffold，且在合理空间 null + order null + common-field/1-f 控制后仍站得住？**

> **非目标（明确禁止）**：(a) 让 Gate A "通过"作为目标——目标是"在合理 null 下能不能站住"，可以是干净的 negative；(b) 把 min_group=3 直接升级成 formal primary；(c) 把 Phase 2 criticality 当主线；(d) 任何 HFO-specific / timing-order-mechanism / critical-mode 措辞。

---

## §0 Cohort lock（2026-07-02 user sign-off：补齐 field-similarity 全 20 队列）

Phase-1 backbone 沿用了发作内场动力学脚本手挑的 13 被试（broad 9 / narrow 7），**欠采样**于 field-similarity（`axis_alignment_*_max_ab_B1000.json`，per_subject n=20 = 18 epilepsiae + 2 yuquan: xuxinyi, zhangkexuan）。**P1b 队列锁定为补齐全 20。**

可行性已核实（`t0_eligibility_audit.csv` + `ICTAL_REFERENCE` 含 yuquan）——7 个新被试全部有 analysis_eligible 发作、`iter_subject_seizure_windows` 可跑通，无需新建原始数据：
`epilepsiae_1084`(72 elig)、`epilepsiae_548`(26)、`epilepsiae_583`(22)、`epilepsiae_590`(12)、`epilepsiae_922`(28)、`yuquan_xuxinyi`(3)、`yuquan_zhangkexuan`(3)。

映射到两套几何（`observation_readout/real_subjects/*_t_a.json` 已存在）：
- **narrow（核心）：7 → 20**（全 20 都有 narrow 几何）。
- **broad：9 → 17**（442/548/958 只有 narrow 几何、无 broad 几何 → 不入 broad）。

构建：`build_topic5_v2_band_cache.py --subjects <7 new>`（cache substrate-independent，建一次两 geometry 共用）；随后 alignment/nulls/gates 在 narrow / broad 上重跑。**cohort 是描述性事实层的锁，不改任何 §1–§6 的 null/tier 口径。**

> **⚠️ 2026-07-02 执行发现（待 user 决策，暂未最终锁）**：18 个 epilepsiae（1084/548/583/590/922 + 已有 13）全部干净可建。**2 个 yuquan（xuxinyi/zhangkexuan）被真实数据模型不兼容阻塞**：yuquan 发作 inventory 只有 `eeg_onset_epoch`、**无 `clin_onset_epoch`**，而整条 field-dynamics 长窗/早窗管线以临床起始为 0 点 → `iter_subject_seizure_windows` 全 drop；且这 2 人各仅 1–2 个真发作（余为零时长标记）。**当前可干净达成 = narrow 18 / broad 15**（epilepsiae-only、统一临床起始锚、统一数据集）。达 20 需改 loader 让 yuquan 改锚 `eeg_onset`（+"早期"0 点定义在 2 人上与其余 18 不同的 caveat + 薄 n）。默认推荐 18-clean，yuquan 作为可选增量。

## §1 空间-null strength hierarchy（LOCK；取代“单一 within_shaft_strong 才算”）

运行前冻结四档 + 各自允许的主张层级：

```text
primary-preferred:     within_shaft, min_group=4   → 可携带 FORMAL Gate A（若可评估）
feasibility-sensitivity: within_shaft, min_group=3 → EXACT/enumerated 排列（见下）；SENSITIVITY 层，不自动升为 primary
weak fallback:         distance_bin                 → 仅描述性
weakest descriptive:   subject_wide                 → 仅描述性，且标注 anti-conservative（不保杆内局部自相关 → obs 偏显著）
```

**每被试必报**（不只 delta/p）：`null_strength`、`n_effectively_permutable_contacts`、`n_singleton_or_small_groups`、`unique_permutation_capacity`（组内可区分排列数）。
**min_group=3 用 exact/enumerated within-shaft 排列**（组合数少，普通 1000 次随机置换只是重复抽极少排列 → p 失真）；若某杆 enumerated 空间过大再退回随机并记录。
**判读规则**：`formal Gate A` 只在 `null_strength ∈ {within_shaft_min4}`（或修订后显式纳入 min3）且 cohort 推断（§2）显著时才写"passed"；min3 结果写"sensitivity-tier supportive/not"，不写 formal pass，除非本 spec 修订版把 min3 纳入 primary（需再 sign-off）。

## §2 Cohort 推断（LOCK；退役 median-of-per-subject-p）

主推断改为 **subject-level cohort permutation of the median statistic + max-over-bands**（perm-long 已存所需 subject×band×perm 值）：
```python
for perm:
    cohort_stat_perm[band,perm] = median_over_subjects(subject_stat_perm[:,band,perm])   # 用 delta 或 null-centered delta
maxT_perm[perm] = max_over_bands(cohort_stat_perm[:,perm])
cohort_p[band]  = (1+#{cohort_stat_perm[band] >= obs_cohort_stat[band]})/(1+N)           # 每 band cohort perm p
gateA_fwer_p    = (1+#{maxT_perm >= max_band obs_cohort_stat})/(1+N)                      # family-wise
```
`cohort_empirical_p`(median-of-p) 降为附属诊断列，不再进 gate。**不预判方向**（cohort perm 可能比 median-of-p 强也可能弱）。

## §3 Order-null 闭合（narrow 是核心，必须闭合，不能只到 strong-子集描述性）

- **T13 anti-conservatism 显式化**：每被试/模板记录 **unshuffled-event-rebuild vs producer typical_rank 的 corr**（现只用它分 strong/weak_downgrade at 0.90）作可审计诊断列；报告 order-p 时并列该 corr，读者可见 obs(producer)-vs-null(rebuild) 的 gap。
- **B-only-template 对称性**（review 边角）：若某被试 producer 有 F_a+F_b 但 event-rebuild-B <4 有限 → observed maxes over {A,B}、null 只 over {A}（额外 anti-conservative）。修：null 与 observed **max over 同一模板集**（用 observed 的 `F_inter_b is None` 决定）。
- **闭合口径**：narrow order-null 结论 = strong 子集（≥0.90）的 cohort perm（§2）+ 上述对称化 + final n_perm。若 strong 子集 n 太小 → 明确降级为 `weak_order_null / rank-stratified only, strongest timing-geometry claim disabled`（rev2 设计的关键防线）。**narrow 若 order-null 不闭合 → narrow 主张只能写"aligns with HFO-derived core geometry/topography"，不写"timing order"**。
- **weak-disable 阈值（LOCK，2026-07-02 user sign-off）**：order closure = **`strong`(evaluable) iff `n_strong >= ceil(0.5 × n_order_evaluable)`**，其中 `n_order_evaluable` = order_strength ∈ {strong, weak_downgrade} 的被试数（即非 `missing`）；否则 `weak_downgrade`(disabled)。**这取代 order 强度的 weakest-wins**：dev-100 暴露一个 `missing` 被试（epilepsiae_916，无 interictal 事件可重建）在 weakest-wins 下把整个 cohort order 掉成 `missing`→order_p 全 NaN，错误地丢掉 13 个 strong 被试的顺序信号。full-20 现值：strong 13、order_evaluable 19（strong+weak）、missing 1 → 13 ≥ ceil(0.5×19)=10 → **order closure evaluable**（strong 子集 cohort perm）。spatial 侧不受影响（§1 仍 weakest-wins：formal Gate A 要求全体 within_shaft_strong，min3 是独立 sensitivity）。

## §4 broad↔narrow：用三个 QC 钉死（现在是 inference，不是结果）

**先证差异是"通道池/几何"而非"实现路径"，再解释频带特异来源。broad/narrow 有 3 个 shared 被试（253/1096/1125）→ 不能当独立队列比，必须 paired。**

- **§4.1 Axis geometry audit**（每 shared 被试）：`n_broad_channels, n_narrow_channels, overlap_count, broad_only_count, narrow_only_count, rank_corr_broad_vs_narrow_on_overlap, field_corr_broad_vs_narrow, HFO_rate_corr_with_rank_{broad,narrow}`。目的：坐实 broad=扩展池 / narrow=核心池，且量化两几何差多少。
- **§4.2 Paired subject frequency profile**（shared 被试 paired diff，不看 cohort median）：`alignment_narrow[band] − alignment_broad[band]`、`delta_null_narrow[band] − delta_null_broad[band]`。
- **§4.3 Channel-pool gradient ablation**（LOCK 规则）：从 broad 几何出发，**按 pre-defined broad-only/peripheral 成员**逐步移除（**不按 observed effect 大小移除**）；每步重建 G_HFO；每步跑**同一个** §1 null；输出每步 `overall_alignment` + **band-specificity index（运行前锁定定义）**：
  ```python
  HF_minus_low = median(delta[13-250Hz bands]) - median(delta[1-13Hz bands])
  ```
  **判据（先锁）**：若移除 broad-only 外围通道后 `HF_minus_low` 单调下降**且** band-generic overall alignment 保留 → 支持"频带特异来自外围通道、narrow core 是宽带共享底座"。若 overall alignment 随核心通道移除单调塌陷 → 承重在少数核心通道。
- **§4.4 narrow leave-one-channel/endpoint**：仅作 **influence/stability**，**不作正式显著性**（核心通道少，single-channel leverage 高）。

## §5 Gate B/C final residuals（先 LOBO + aperiodic）

先跑最关键两项，用 §2 cohort 推断 + final n_perm：
- **LOBO common-field residual**：判 narrow band-generic 是否只是 shared broadband recruitment（大概率）。
- **aperiodic / 1-f residual**：判高频 alignment 是否只是 broadband offset/slope。
再跑 HFO-rate / baseline-power / shaft-position residual。**confound 用单-covariate sensitivity 为主，不塞大 combined 模型**（n contacts 小）。**三种结局**（先写好措辞）：A 残差后消失 → "G_HFO predicts broadband/common recruitment"（最可能、最干净，broadband-recruitment tier）；B 某 band 残差后存活 → shared scaffold + frequency-specific layer；C 80-250 残差后存活 → HFO/ripple-specific（现 narrow band-generic 形态下 C 不最可能，仍跑完）。

## §6 判读语言 tiers（LOCK；配合口径修正）

- **最稳可写**：`HFO-derived core geometry may mark a shared, band-generic early-ictal recruitment scaffold`（narrow）。
- **标签**：`legacy: PASS` · `weak-null: positive (likely inflated, see §1)` · `formal within-shaft Gate A: <pass/unresolved>` · `order-null (narrow): <closed-positive/weak-disabled>` · `Gate B/C: <tier>`。
- **禁**：timing-order mechanism / critical mode / "沿路径传播"（除非 Phase 3 temporal recruitment）。

## §7 n_perm + 运行顺序

`dev 100 → final 1000（→ 5000 if p 触底需要）`。顺序：**P1（§1+§2+§3）先跑**（判 scaffold 在合理 null 下能否站住）→ **P2（§5 LOBO+aperiodic）** → **P3（§4 ablation）**。全量 null 慢（E916 44 发作 ≈ 2h @1000）→ 后台排队 + 先 narrow（核心、被试少、~1-1.5h/feature @1000）。

## §8 Merge / 报告

- backbone **code** 可 merge（legacy 逐位复现、cache 不破坏旧 bb、subject-unit 聚合、结果目录隔离、31/31 测过、全分支复审 READY）——**但 merge message 必写**：`dev n_perm=100; formal Gate A not passed (null-strength gated); Gate B/C pending; NO HFO-specific / critical-mode claim`。
- **不 merge** 当前中文报告作为 final scientific report，除非按"口径修正"降级。
- 正式 docs 报告标题（建议）：`HFO-derived core geometry shows band-generic early-ictal recruitment alignment under weak spatial nulls`（不写 "Gate A passed" / "HFO-specific"）。
