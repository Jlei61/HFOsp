# Group-Event State v0.3.3 Trainability Closeout（技术版）

<!-- REVISION_2026-09-03_EVENING -->
> **2026-09-03 evening interpretation revision (current authority).** The PM audit numbers and controls remain valid, but “no extra predictive information” was too strong. A selection-period constant reproducing the gain in 5/9 subjects is positive evidence for period-level or ultra-slow predictive context relative to the old baseline. It is not target leakage; because early anchors use state inputs from the rest of the selection period, it is a noncausal input-side period oracle rather than a deployable causal state estimator. Separately, learned dynamics beating random dynamics in 3/9 subjects remains subject-level evidence for candidate event-history information. Current evidence tiers: L1 moderate-to-strong; L2 weak-to-moderate and heterogeneous; L3 not established; L4/L5 not yet validly tested by a trained multi-view producer. The revised contract is `group_event_state_v0_3_4_multiview_predictive_state_{spec,plan}_2026-09-03.md`.

<!-- REVISION_2026-09-03_PM -->
> **2026-09-03 PM review revision (supersedes the conclusions below).** Reviewer audit of code + artefacts after the morning closeout. All numbers below are rendered from `incremental_summary.json` (schema 9) and the per-subject `review.json` files; nothing is typed by hand. The original text is kept below for provenance.

## R0. Findings (severity, status)

| # | Severity | Finding | Status |
|---|---|---|---|
| P0-1 | scientific | The headline STATE_SELECTION contrast `H_mark − learned` is reproduced by replacing the state with one constant vector (its inner-val mean) in 5/9 subjects (E1096, E1146, E548, E916, E922); `beyond_period_offset` CI_low > 0 only in E1125, E384, whose total-gain CI crosses 0. H_mark under-predicts counts in STATE_SELECTION (count/μ ratio up to 2–3×). A same-segment block-circular shift preserves the period mean and cannot detect this. | control added to `multi_seed_card_diagnostics` (`period_offset_control`), monitor schema 9, label `increment_explained_by_constant_period_offset`; offset audit script `scripts/audit_group_event_state_v033_offset_drift_control.py` |
| P0-2 | scientific | A crude causal 3 h count recalibration of H_mark (no model) recovers a comparable or larger STATE_SELECTION gain where the drift is large (E1096 +1.08; E1146 +0.45; E384 +0.30; E583 +2.11) but over-corrects sparse subjects (E1125 -1.02; E253 -0.24; E548 -1.51) — a probe, not a baseline. Retraining the selected recipes on it leaves `H_recal − learned` CI_low>0 only in E253 (of which E253 only repairs the damaged baseline); none of E1096/E1146/E548 retains a supported increment; `shifted − correct` > 0 in 4/9, `learned < random` in 1/9. | `scripts/run_group_event_state_v033_recalibrated_baseline_arms.py` (DIAGNOSTIC; TRAIN+STATE_SELECTION only) |
| P0-3 | reporting | Two already-consumed one-time development scores on the same subjects (pilot requests) were omitted: E916 STATE_SELECTION +0.049 → development −0.017; E253 +0.010 → development +0.488 with random ≈ H. | now read (read-only) into the monitor and both reports |
| P1-1 | design | Recipe selection (48 configs → 24 → 12 → 3×5 seeds), checkpoint step selection and the reported CI all use the same STATE_SELECTION anchors (optimistic bias). In E922 100% and E1146 77% of freshly sampled rung-0 configs already beat H. | rung-0 context added to the monitor; documented; nested selection split recommended for the next round |
| P1-2 | engineering | Selected-recipe tiny-overfit re-run used a fixed 300-step budget regardless of the recipe LR. | re-run at the recipe's own 900-step budget: only E583 flips to pass; E548 diverges (gap −0.31); others unchanged → not a budget artefact |
| P1-3 | engineering (S_G) | Human S_G O2 S1: 6/6 cells select step 1 (gain ~1e-5). The transplanted "frozen O1 recipe" scales the pilot LR by 0.1 → encoder LR 1.7e-6, adapter 4.2e-6; inner-NLL selection among LR scales picks the do-nothing scale. | reported; S_G needs its own LR search after synthetic D3 Level-2 recovery |
| P2 | reporting | `selection_metric_is_canonical=false` on all 9 cards (canonical NB function is used but no evaluator-hash registration/parity run); `search.n_batches=1, stop_reason=max_batches` for all 7 broad subjects (T5 two-batch stop rule never exercised); E916/E253 best step 10 (warm-up) in 4/5 seeds. | documented |

## R1. Period-offset control (frozen checkpoints, 5 seeds, seed-median then within-target-segment moving-block bootstrap)

| 患者 | 总增益 H−learned | 常数臂 H−period_mean | 常数解释不了的部分 | oracle 常数上界 | H_mark 选择期 实际/预测 (0–5/5–15/15–30 min) | 训练期 实际/预测 | 独立块 |
|---|---:|---:|---:|---:|---|---|---:|
| E1096 | +0.715 [+0.384, +1.099] | +0.677 | +0.038 [-0.230, +0.315] | +1.946 | 2.00/2.08/2.22 | 0.95/0.99/1.18 | 26 |
| E1125 | +0.129 [-0.006, +0.256] | +0.046 | +0.084 [+0.003, +0.169] | +0.059 | 1.02/0.94/0.82 | 1.08/1.05/0.99 | 28 |
| E1146 | +0.847 [+0.065, +1.942] | +0.757 | +0.090 [-0.054, +0.274] | +1.170 | 1.43/1.45/1.29 | 1.07/1.11/1.25 | 10 |
| E253 | +0.014 [-0.013, +0.042] | +0.001 | +0.013 [-0.014, +0.039] | +0.062 | 1.18/1.30/1.27 | 1.03/1.09/1.08 | 50 |
| E384 | +0.069 [-0.043, +0.208] | -0.047 | +0.116 [+0.066, +0.178] | +0.059 | 0.94/0.90/0.84 | 1.06/1.03/1.01 | 10 |
| E548 | +0.301 [+0.052, +0.590] | +0.684 | -0.383 [-0.746, -0.110] | +1.875 | 1.82/2.85/3.34 | 1.04/1.11/1.11 | 18 |
| E583 | +0.042 [-0.156, +0.223] | -0.195 | +0.236 [-0.031, +0.526] | +0.542 | 0.92/0.68/0.97 | 1.02/1.01/1.02 | 11 |
| E916 | +0.014 [+0.006, +0.024] | +0.015 | -0.000 [-0.008, +0.006] | +0.099 | 1.10/1.08/1.17 | 1.10/1.00/1.01 | 70 |
| E922 | +0.161 [+0.020, +0.301] | +0.144 | +0.017 [-0.032, +0.066] | +0.153 | 0.73/0.79/0.84 | 1.16/1.16/1.14 | 12 |

Identity per anchor: `H − learned = (H − period_mean) + (period_mean − learned)`. `oracle` = per-bin constant fitted on the STATE_SELECTION targets (upper bound of any constant story; not a model).

## R2. Causally recalibrated baseline, same recipe retrained (learned + frozen-random encoder arms)

`log μ_recal(a) = log μ_H(a) + clip(log((Σ y + 1)/(Σ μ_H + 1)), ±1.5)` over exposed anchors of the same carry segment whose target window ended before `a` and within a trailing 3 h window.

| 患者 | 因果重标定本身 H_mark−H_recal | 重训后 H_recal−learned | 常数解释不了 | 错时代价 | 学到−随机 |
|---|---:|---:|---:|---:|---:|
| E1096 | +1.079 [+0.345, +1.812] | +0.021 [-0.055, +0.103] | +0.016 [-0.049, +0.089] | -0.012 [-0.056, +0.033] | -0.022 [-0.106, +0.053] |
| E1125 | -1.021 [-3.311, +0.372] | +0.788 [-0.068, +2.082] | +0.889 [-0.087, +2.397] | +3.782 [+0.509, +9.339] | -0.144 [-0.665, +0.221] |
| E1146 | +0.450 [-0.031, +1.575] | +0.503 [-0.296, +1.663] | -0.054 [-0.243, +0.178] | -0.227 [-0.418, -0.032] | -0.027 [-0.225, +0.156] |
| E253 | -0.238 [-0.881, +0.266] | +0.155 [+0.032, +0.329] | +0.162 [+0.030, +0.346] | +0.265 [+0.105, +0.454] | -0.156 [-0.324, -0.031] |
| E384 | +0.301 [-0.072, +0.603] | +0.019 [-0.008, +0.053] | +0.021 [-0.003, +0.049] | +0.038 [+0.005, +0.089] | -0.019 [-0.054, +0.009] |
| E548 | -1.507 [-6.671, +1.395] | +0.459 [-0.159, +1.537] | +0.155 [-0.239, +0.751] | +0.371 [-0.315, +1.163] | -0.432 [-1.602, +0.205] |
| E583 | +2.109 [-0.200, +5.593] | -0.018 [-0.081, +0.046] | -0.006 [-0.055, +0.043] | +0.011 [-0.016, +0.042] | +0.018 [-0.003, +0.041] |
| E916 | -0.013 [-0.145, +0.116] | +0.012 [-0.005, +0.030] | -0.002 [-0.017, +0.013] | +0.001 [-0.024, +0.026] | -0.011 [-0.025, +0.001] |
| E922 | +0.062 [-0.112, +0.197] | +0.006 [-0.043, +0.051] | -0.006 [-0.031, +0.023] | +0.025 [+0.002, +0.051] | +0.008 [-0.004, +0.023] |

## R3. Slow-bank arms (τ = 6/12/24 h) — "could it be a slower state?"

| 患者 | 慢库 on H_recal：H_recal−learned | 学到−随机 | 慢库 on H_mark：H_mark−learned | 学到−随机 | 原配方 on H_mark：H_mark−learned（训练卡） |
|---|---:|---:|---:|---:|---:|
| E1096 | +0.021 [-0.023, +0.064] | -0.023 [-0.062, +0.018] | +0.479 [+0.187, +0.773] | -0.470 [-0.766, -0.179] | +0.715 [+0.384, +1.099] |
| E1125 | +0.788 [-0.287, +2.315] | -0.259 [-0.864, +0.304] | 未完成 |  | +0.129 [-0.006, +0.256] |
| E1146 | +0.061 [-0.015, +0.189] | +0.030 [-0.012, +0.090] | +0.070 [-0.007, +0.193] | -0.070 [-0.197, +0.010] | +0.847 [+0.065, +1.942] |
| E253 | +0.220 [-0.136, +0.688] | -0.220 [-0.686, +0.136] | 未完成 |  | +0.014 [-0.013, +0.042] |
| E384 | +0.000 [-0.005, +0.005] | +0.000 [-0.002, +0.004] | 未完成 |  | +0.069 [-0.043, +0.208] |
| E548 | +0.063 [+0.001, +0.138] | -0.065 [-0.136, -0.002] | +0.002 [-0.010, +0.012] | -0.008 [-0.017, +0.000] | +0.301 [+0.052, +0.590] |
| E583 | -0.049 [-0.087, -0.015] | +0.014 [-0.006, +0.035] | 未完成 |  | +0.042 [-0.156, +0.223] |
| E916 | +0.005 [-0.007, +0.016] | +0.001 [-0.001, +0.003] | 未完成 |  | +0.014 [+0.006, +0.024] |
| E922 | +0.011 [-0.051, +0.061] | -0.004 [-0.047, +0.047] | +0.055 [+0.026, +0.091] | +0.092 [+0.007, +0.180] | +0.161 [+0.020, +0.301] |

Tally: on H_recal gain 1/9, learned<random 1/9; on H_mark gain 2/4, learned<random 1/4. A slow *mark-dependent* state requires the learned slow bank to beat its random twin on the recalibrated baseline; a tie means a rate level only.

## R3b. Multi-shift null (32 offsets) and linear-trend arm

| 患者 | 32 个错位里比正确时刻差的比例 | 错位代价 中位 | 错位变化的伤害 | 正确时刻超出常数的帮助 | 线性趋势臂增益 H−trend | 超出线性趋势 trend−learned |
|---|---:|---:|---:|---:|---:|---:|
| E1096 | 97% | +0.309 | +0.175 | +0.082 | +0.815 [+0.476, +1.215] | -0.100 [-0.157, -0.048] |
| E1125 | 94% | +0.132 | +0.066 | +0.076 | +0.013 [-0.174, +0.156] | +0.116 [+0.048, +0.208] |
| E1146 | 75% | +0.138 | +0.156 | +0.022 | +0.785 [+0.057, +1.788] | +0.062 [-0.086, +0.228] |
| E253 | 97% | +0.017 | +0.007 | +0.013 | +0.001 [-0.001, +0.004] | +0.013 [-0.014, +0.039] |
| E384 | 94% | +0.037 | -0.019 | +0.082 | +0.092 [-0.035, +0.254] | -0.023 [-0.060, +0.015] |
| E548 | 56% | +0.055 | +0.397 | -0.393 | +0.796 [-0.053, +2.071] | -0.495 [-1.651, +0.202] |
| E583 | 34% | -0.008 | -0.172 | +0.157 | +0.048 [-0.144, +0.223] | -0.006 [-0.035, +0.021] |
| E916 | 66% | +0.001 | +0.001 | -0.000 | +0.019 [+0.008, +0.032] | -0.004 [-0.015, +0.003] |
| E922 | 100% | +0.152 | +0.110 | +0.040 | +0.171 [+0.018, +0.321] | -0.010 [-0.027, +0.004] |

`shift delta = (shift − period_mean) + (period_mean − learned)` per anchor; "trend arm" replaces the state by its per-target-segment least-squares linear ramp in time (input only). Correct time in the favourable tail (≥90% of offsets worse): E1096, E1125, E253, E384, E922; trend arm ≥ learned: E1096, E384, E548, E583, E916, E922; beyond-trend CI_low>0: E1125. Script: `scripts/audit_group_event_state_v033_multi_shift_null.py`.

## R4. Selection context and adequacy re-check

| 患者 | 第一轮 48 个随机配置里已胜过 H 的比例 | 其中位增益 | 搜索批次 | 最终配方 300 步 gap | 最终配方 900 步 gap |
|---|---:|---:|---:|---:|---:|
| E1096 | 50% (n=48) | +0.001 | 1 (max_batches) | 0.06 | 0.12 |
| E1125 | 88% (n=48) | +0.014 | 1 (max_batches) | 0.24 | 0.25 |
| E1146 | 77% (n=48) | +0.029 | 1 (max_batches) | 0.57 | 0.59 |
| E253 | 0% (n=18) | -0.024 | 2 (stable_plateau) | NA | NA |
| E384 | 81% (n=48) | +0.014 | 1 (max_batches) | 0.40 | 0.40 |
| E548 | 46% (n=48) | -0.004 | 1 (max_batches) | 0.22 | -0.31 |
| E583 | 69% (n=48) | +0.023 | 1 (max_batches) | 0.40 | 0.66 |
| E916 | 100% (n=18) | +0.026 | 2 (stable_plateau) | NA | NA |
| E922 | 100% (n=48) | +0.042 | 1 (max_batches) | 0.29 | 0.57 |

## R5. Previously consumed development scores (same subjects, earlier requests; read-only)

- E253，旧请求 `human-sn-r0-253-pilot-v1`（冻结时训练标签 DIAGNOSTIC）：选择期增量 +0.010 [-0.004, +0.027] → development 段 +0.488 [+0.371, +0.622]；development 错时代价 +0.086 [-0.034, +0.217]；随机−学到 +0.470 [+0.355, +0.600]；development 独立块 97。
- E916，旧请求 `human-sn-r0-916-pilot-v1`（冻结时训练标签 TRAINING-ADEQUATE）：选择期增量 +0.049 [+0.010, +0.088] → development 段 -0.017 [-0.030, -0.006]；development 错时代价 -0.006 [-0.025, +0.012]；随机−学到 -0.020 [-0.033, -0.009]；development 独立块 137。

## R6. Revised allowed statement

> The current S_N results do not provide evidence that group interictal event history carries future-burden information beyond recent statistics. The STATE_SELECTION increments are reproduced by a constant period offset that corrects a level miscalibration of H_mark in later recording time; this is compatible with a slower-than-model variable, but that variable is a rate level and belongs to the baseline ladder (H_rate at longer windows), not to S. Whether a mark-dependent slow state exists is untested until the baseline self-calibrates in time and the learned slow bank is compared with a random slow bank on that baseline.

## R7. Code changes (uncommitted, this worktree)

- `src/topic5_group_event_state/v033_training_lab/diagnostics.py`: `merge_seed_anchor_diagnostics(..., period_mean_nll=)`, `multi_seed_card_diagnostics` computes the period-mean arm per seed → `period_offset_control` + per-seed fields.
- `src/topic5_group_event_state/v033_training_lab/card.py`, `queue.py`: card field `period_offset_control` (CARD_FIELDS).
- `tests/test_group_event_state_v033_training_lab_period_offset.py` (2 tests; decomposition identity + card carriage). Training-lab + canonical suites: 97 + 27 passed.
- `scripts/monitor_group_event_state_v033_trainability.py`: schema 9 (offset control, rung-0 context, 900-step re-check, recalibrated arms, prior development scores, new label).
- New scripts: `scripts/audit_group_event_state_v033_offset_drift_control.py`, `scripts/run_group_event_state_v033_recalibrated_baseline_arms.py` (`--baseline`, `--taus`).
- Artefacts: `supervisor_reports/trainability_incremental/{offset_drift_control, recalibrated_baseline_arms, slow_bank_arms_on_recal, slow_bank_arms_on_mark, selected_recipe_tiny_overfit_900steps}/`.

<!-- REVISION_2026-09-03_PM -->

---

> Original 2026-09-03 AM text follows; its §3 interpretation and §6/§7 are superseded by R0–R7 above. §2 (execution integrity) remains valid.

**日期：** 2026-09-03  
**结论层级：** optimization / identifiability diagnostic；不是 H1/H2/H3 efficacy evaluation。

## 1. 冻结范围

- 科学目标：`S_N` 对三段 future count profile 的 residual prediction；bins 为 `[0,300)`, `[300,900)`, `[900,1800)` 秒。
- 基线：`H_mark`，已含 `H_rate` 及 extent/STOP、contact/repertoire occupancy、multiband EMA。
- 数据层：`STATE_TRAIN` 拟合，chronological `STATE_SELECTION` 选配方与形成卡片。
- 禁止层：development evaluation、seizure outcome、sealed/formal partition、human H3。
- broad-search subjects：E1096、E1125、E1146、E384、E548、E583、E922；另纳入先前 O1a 的 E253、E916 形成 9 人汇总。
- 每个 broad request：48 configs；successive-halving `100→300→900` steps；最终 3 configs、5 seeds；单 request 最多 2 workers。

## 2. 执行完整性

- expansion controller commit：`2fada35abb6615ac5a012241fa00533f7ec40b80`。
- expansion units：791 `COMPLETE`，0 pending，0 running；7/7 request `COMPLETE`。
- 资源：2×RTX 3090；全局最多 4 workers，每 request 最多 2；未见 OOM、NaN、Traceback。
- 监控闭环：9/9 training cards、7/7 selected-recipe tiny-overfit reviews、8/8 O1b replications；`read_errors=[]`。
- 最终 monitor：`all_expected_cards_complete=true`、`all_selected_recipe_audits_complete=true`、`all_replications_complete=true`、`all_outputs_complete=true`。
- 结果 JSON SHA-256：`2cd3254a62fe8082c69fb584beee1db4d3a8caa5cd5229ec1c4e2f831821b854`。

## 3. 主要对比

定义：

- `gain = NLL(H_mark) − NLL(H_mark + learned state)`；正值有利。
- `shift = NLL(block-shifted state) − NLL(correct-time state)`；正值有利。
- `random = NLL(learned state) − NLL(equal-capacity frozen random reservoir)`；负值有利。
- 区间：先逐 anchor 对 5 seeds 取中位，再在 target segment 内做 moving-block bootstrap。

| Subject | gain [95% CI] | shift [95% CI] | learned−random [95% CI] | independent windows | gain/time/random support |
|---|---:|---:|---:|---:|---|
| E1096 | +0.7146 [+0.3844,+1.0990] | +0.3365 [−0.1043,+0.7259] | −0.6985 [−1.0778,−0.3718] | 26 | 1/0/1 |
| E1125 | +0.1294 [−0.0057,+0.2564] | +0.2298 [+0.0715,+0.4127] | −0.0471 [−0.1423,+0.0539] | 28 | 0/1/0 |
| E1146 | +0.8473 [+0.0652,+1.9416] | +0.0896 [−0.0308,+0.2063] | −0.8053 [−1.8534,−0.0612] | 10 | 1/0/1 |
| E253 | +0.0144 [−0.0132,+0.0419] | +0.0249 [−0.0088,+0.0551] | −0.0109 [−0.0365,+0.0172] | 50 | 0/0/0 |
| E384 | +0.0687 [−0.0434,+0.2084] | +0.0399 [−0.0408,+0.1394] | −0.0678 [−0.2062,+0.0438] | 10 | 0/0/0 |
| E548 | +0.3009 [+0.0516,+0.5904] | +0.0764 [−0.3325,+0.4760] | −0.3032 [−0.5966,−0.0484] | 18 | 1/0/1 |
| E583 | +0.0417 [−0.1562,+0.2231] | +0.0059 [−0.0960,+0.0973] | +0.0032 [−0.1227,+0.1378] | 11 | 0/0/0 |
| E916 | +0.0144 [+0.0063,+0.0236] | +0.0042 [−0.0040,+0.0129] | −0.0019 [−0.0041,+0.0001] | 70 | 1/0/0 |
| E922 | +0.1610 [+0.0202,+0.3007] | +0.1725 [+0.0737,+0.3070] | −0.0305 [−0.0990,+0.0161] | 12 | 1/1/0 |

Cohort tally：gain 5/9，correct-time 2/9，learned-better-random 3/9；gain∩time∩random 为 0/9。患者数是推断单位，seed 不当作患者复制。

## 4. 最终配方训练充分性复核

原卡片的 tiny-overfit 在 broad search 前用默认 recipe 执行。由于最终选择会改变 optimizer、LR、width/depth、normalization 与 initialization，原 T0 不能证明最终 recipe 可训练。为此，对 7 位 broad-search subjects 用最终入选 recipe 在同一 12-anchor TRAIN slice 上重跑 300 steps；card/input/split/producer SHA 均闭合。

| Subject | optimizer | schedule | width/depth | state dim | activation/norm | selected-recipe gap closed | pass |
|---|---|---|---:|---:|---|---:|---|
| E1096 | RMSprop | constant | 64/1 | 12 | ReLU/LayerNorm | 0.055 | no |
| E1125 | RMSprop | constant | 32/2 | 6 | SiLU/none | 0.243 | no |
| E1146 | AdamW | plateau | 128/3 | 24 | SiLU/LayerNorm | 0.568 | yes |
| E384 | Adam | constant | 128/1 | 24 | ReLU/none | 0.399 | no |
| E548 | AdamW | plateau | 128/2 | 12 | SiLU/LayerNorm | 0.215 | no |
| E583 | AdamW | constant | 64/2 | 24 | SiLU/none | 0.401 | no |
| E922 | AdamW | cosine | 128/3 | 24 | GELU/none | 0.289 | no |

只有 E1146 通过 `gap_closed ≥ 0.5`。E1146 仍因 synthetic recovery 失败且部分 seed 在 warm-up 选中而不构成完整 training adequacy。E922 原卡片为 `TRAINING-ADEQUATE`，但最终 recipe tiny-overfit 仅 0.289，因此聚合器已按最终 recipe 撤回该标签。最终有效 training adequacy 为 0/9。

## 5. Seed 与表示诊断

- 所有患者均为 5 个独立 final seeds；learned 与 random checkpoint hash 各自 5/5 唯一。
- E1096/E1146/E548 的 gain 和 learned−random 同时过线，但 shift 未过线。
- E922 的 gain 和 shift 同时过线，但 learned−random 未过线。
- E1125 只有 shift 过线；E253/E384/E583 三项均未过；E916 只有量级很小的 gain 过线。
- state dimension 为 6/12/24，TRAIN-state participation ratio 约 1.26–5.45；这是表示秩诊断，不是生理维度估计。E922 的 24 维状态 top-1 解释 88.6%，提示高度塌缩，但不能单独解释其预测增量。
- O1b 在 E253、E916 各 4 seeds 完成；固定配方 gain 在两患者均出现 seed 方向翻转，说明 optimization/initialization 对小效应读数有实质影响。

## 6. 科学边界

本轮唯一允许的科学陈述是：

> 在部分患者中，过去群体间期事件携带超出预定义多尺度近期统计、并可用于预测未来 5–30 分钟事件负荷的信息；该信息对网络配置敏感，尚未稳定识别为时刻特异的慢状态。

禁止由本轮推出：

- 已建立 H1 physiological slow state；
- 已复现 H2a contact-sequence/repertoire modulation；
- 状态可预测 seizure distance 或 early ictal field；
- IED 对生理状态存在 event-feedback causality；
- 训练充分性测试阴性等于人体没有状态。

## 7. 下一实验接口

1. 将当前 checkpoints 登记为 `S_N` candidates，不作为唯一 state producer。
2. 训练独立的 `S_G` producer，使 future conditional repertoire、participation field、STOP/extent、delay 与 multiband expression 直接进入 objective。
3. 在固定物理时间 anchor 上，用同一 evaluator 比较 `H_mark`、`H_mark+S_N`、`H_mark+S_G` 与 block-shifted state。
4. 冻结 state 后运行 H2a exact contact sequence/same-prefix continuation 与 H2b survival + early ictal spatial field；不以当前结果作为 gate。
5. E1096/E1146/E548/E922 作为 development candidate；E583 作为 count-vs-spatial dissociation case；E1073/E1077 保持 untouched replication。

## 8. 权威产物

- 白话增量报告：`/data/hfosp_group_event_state_v0_3_3/supervisor_reports/trainability_incremental/incremental_plain.md`
- 技术增量报告：`/data/hfosp_group_event_state_v0_3_3/supervisor_reports/trainability_incremental/incremental_technical.md`
- 机器汇总：`/data/hfosp_group_event_state_v0_3_3/supervisor_reports/trainability_incremental/incremental_summary.json`
- 逐患者最终配方复核：`/data/hfosp_group_event_state_v0_3_3/supervisor_reports/trainability_incremental/selected_recipe_tiny_overfit/<subject>/selected_recipe_tiny_overfit_review.json`
- 监控终态：`/data/hfosp_group_event_state_v0_3_3/supervisor_reports/trainability_incremental/monitor_state.json`
