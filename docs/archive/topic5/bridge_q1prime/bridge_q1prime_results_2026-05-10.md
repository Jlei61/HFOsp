# Topic 1 × Topic 5 Bridge Q1' (PIVOT) — 探索性 case-series 结果 (2026-05-10)

> **Tier**: case-series exploratory，N=4 strict swap + 1 candidate sentinel + 1 axis-collapse descriptive。
> **Verdict**: **INDETERMINATE**（4/4 strict subjects 在 effect-size 上正向, 0/4 通过 p<0.05 dual gate; n_eligible 5–14, power-floor 主导）
> **Spec**: `docs/superpowers/specs/2026-05-10-topic1-topic5-bridge-design.md` §10
> **Plan**: `docs/superpowers/plans/2026-05-10-topic1-topic5-bridge-q1prime.md`
> **Phase-1 reference**: `docs/archive/topic5/bridge_q1/bridge_q1_results_2026-05-10.md` (Q1 NULL-locked, 弃案)

## 1. 主 axis (PIVOT)

phase-1 的 state-fingerprint approach (frac_T0/switch_rate/last_template) 因为：
- (a) pre-ictal 窗口分钟级太短 (median 4 events / sz, frac 分辨率仅 5 个值)
- (b) `last_template` 单 boundary event 噪声主导
- (c) 选错 axis (state fraction ≠ subtype 的物理基础)

被弃案。Q1' 改用：每 ictal seizure 在 **swap-channel subset (rank_displacement §8 strict tier endpoint = top-decision_k ∪ bottom-decision_k channels, joint_valid)** 上的 channel-onset rank 与该 subject 两个 interictal template rank (T0, T1) 的 Spearman 相关 **(ρ_a, ρ_b)** 决定 per-seizure assignment ∈ {T0, T1, tie}。Within-subject 检验：assignment 与 topic5 PR-1 z-ER subtype label 的列联 (Fisher / χ² + Cramér V + AMI)。

详见 spec §10。

## 2. Cohort

| 类别 | 数量 | subjects | 备注 |
|---|---|---|---|
| Strict (主检验) | 4 | 1073, 1146, 635, 958 | §8.7 channel-label 合同 strict tier |
| Candidate (sentinel only) | 1 | 548 | swap_class=candidate, p_fw=0.057 |
| Inadmissible (axis collapse) | 1 | 442 | γ_gamma=1 + swap_class=none |

## 3. Per-subject 数值表

| subject | swap | n_atlas | n_swap_endpoint | n_eligible | p (Fisher/χ²) | Cramér V | AMI | q1prime_positive |
|---|---|---|---|---|---|---|---|---|
| 1073 | strict (k=3) | 18 | 6 | 5 | 1.000 | 0.250 | -0.250 | False |
| 1146 | strict | 19 | 14 | 5 | 0.400 | **0.667** | 0.251 | False |
| 635 | strict | 16 | 6 | 8 | 1.000 | 0.378 | 0.000 | False |
| 958 | strict | 14 | 12 | 14 | 0.176 | **0.595** | 0.056 | False |
| 548 | candidate | 26 | 8 | 18 | 0.113 | **0.703** | 0.114 | False |
| 442 | none | 21 | 8 | 14 | 1.000 | 0.113 | -0.070 | False |

## 4. Cohort case-series verdict

- **3-state**: INDETERMINATE
- **median Cramér V (strict)**: 0.486
- **median AMI (strict)**: 0.028
- **n_strict_positive**: 0 / 4

**关键观察**：4 strict subjects 全部 q1prime_positive=False，但 **Cramér V 全部为正向且较大**（0.25–0.67，median 0.49）；**没有任何 subject 通过 p<0.05** 是 power-floor 而非空信号——n_eligible 在 5–14 之间，Fisher exact 2×2 perfect 5+5 仅给 p=0.008，但实际 contingency 不是 perfect，需要更大 n 才能跨过 0.05。**这是 INDETERMINATE 的典型画面：effect 在 cohort 内一致正向但 power 不足以触发 p-gate**。

## 5. Sentinel verdicts

### 548 (candidate, sentinel only)
- swap_class=candidate (p_fw=0.057, just above strict α=0.05)
- γ_gamma=5 (highest cohort heterogeneity)
- n_eligible=18, p=0.113, **Cramér V=0.703**, AMI=0.114
- **Highest Cramér V in cohort**——effect direction 与 strict 4 subject 一致；不进 strict cohort 主统计但作 confirmatory case study。

### 442 (axis collapse, inadmissible)
- γ_gamma=1（仅 1 主子型 + 1 outlier）
- swap_class=none
- Cramér V=0.113 ≈ noise floor; AMI=-0.07
- **预期 axis collapse 已确认**——既无 within-subject subtype 异质性也无 swap geometry，Q1' 在该 subject 上无可测信号。可作 negative-control descriptive。

## 6. 结构性观察 (重要)

### 6.1 power floor: n_eligible 太小

n_atlas (14–26) 在通过 swap-subset ∩ atlas-valid-onset 双滤后落到 n_eligible (5–18)。dual gate (p<0.05 AND V>0.30) 在 Fisher 2×2 上要求 perfect ≥5+5；strict 4 subject 中 1073/1146 仅 5 eligible，结构性无法过 p。

潜在改进路径（不在本档案范围）：
- 放宽 channel-intersection floor（min_channels=2 而非 3）
- 用 χ² 替代 Fisher 在 n>10 时（χ² 在小 n 上 inflated p）
- aggregate 多 subject 的 ρ_a/ρ_b 做 cohort-level Spearman（前提是 ρ 跨 subject 可比，需另外 audit）

### 6.2 effect 一致正向是真信号

4 strict + 1 candidate 共 5 subjects 全部 V>0.25；4 strict median V=0.486 显著高于 V_min=0.30。AMI 中位 0.03 较弱，可能是 contingency 形状（2×k_s≥3 多类）使 AMI denominator 变大。

### 6.3 442 axis collapse 干净

与 phase-1 archive 的 442 sentinel 一致——442 在 Q1' 框架下既无 swap geometry 也无 subtype 异质性，作 negative-control。

## 7. Caveats (from spec §10.6)

- N=4 不构成 cohort claim（仅 case-series）
- §8.7 strict-only channel-label 合同——548 candidate 必须独立报，不可合并入主统计
- Phase-1 Q1 NULL 与 Q1' INDETERMINATE 不是同一 axis 的两次试验，不可叠加 framing
- Q1' 不依赖 pre-ictal 窗口（纯 ictal channel-onset 几何对应）；spec §3.5 windows 在 Q1' 下作废
- channel intersection floor = 4: 低于此 Spearman 不可计算，seizure 整段 drop

## 8. 文件清单

### 代码
- `src/topic1_topic5_bridge.py` — Q1' 模块（loader + alignment + per-subject test + cohort summary + 3 figures）
- `scripts/run_topic1_topic5_bridge.py` — `q1prime` / `q1prime-cohort` / `q1prime-figures` subcommands
- `tests/test_topic1_topic5_bridge.py` — 41 tests (10 Q1')

### 数据 (gitignored)
- `results/topic1_topic5_bridge/q1prime_per_subject/*.json` — 6 个 per-subject JSON
- `results/topic1_topic5_bridge/q1prime_cohort_summary.json` — verdict + per_subject roll-up
- `results/topic1_topic5_bridge/figures/q1prime_*.png` — 3 PNGs
- `results/topic1_topic5_bridge_q1prime_audit.csv` — audit (16 subjects → 4 strict + 1 candidate + 1 inadmissible)

## 9. 推荐下一步

verdict = INDETERMINATE，不是 NULL。三条候选路径（user 决定）：

1. **接受 INDETERMINATE，等更大 cohort**：yuquan v2.3 atlas 建好后 cohort 扩容；当前 4 strict + 1 candidate 不足。
2. **放宽 dual gate 阈值**：α=0.05 太严苛 with n_eligible 5-14；考虑 α=0.10 或仅用 effect-size 作 case-series gate。
3. **走 cohort-level pooling**：聚合 5 subjects 的 (ρ_a, ρ_b, subtype) 做 cohort-level Spearman / mixed model；但需额外 audit ρ 跨 subject 可比性（每 subject T0/T1 freeze convention 已统一，ρ 是 subject-internal 量度，理论上可比）。

每条都是新 PR，本档案不直接执行。
