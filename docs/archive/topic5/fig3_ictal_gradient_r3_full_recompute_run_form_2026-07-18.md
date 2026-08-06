# Run form — Figure 3 发作相关 gradient R3 field concordance 全量重算

对应 handoff：`docs/archive/topic5/fig3_ictal_gradient_r3_full_recompute_handoff_2026-07-18.md`

> Engineering contract 使用 `verified / deviation / not run`；科学结果不使用 PASS/FAIL。

---

## A. Run identity

| field | value |
|---|---|
| runner | Claude (Opus 4.8), agent execution |
| date/time | 2026-07-18 |
| branch | `topic5-fig3-r3-grid-rebuild` |
| commit | working tree (uncommitted; new files additive) |
| worktree | main checkout `/home/honglab/leijiaxin/HFOsp` |
| Python / NumPy / SciPy / pandas | 3.11 / 1.26.x / 1.11.x / 2.x (see `contract_manifest.json`) |
| primary estimand | `R3 dense-grid field similarity` (verified) |
| paired sensitivity | `R2 contact-evaluated smoothed similarity` (verified, rerun same pass) |
| seed | `20260718` (verified) |
| n_perm | `1000` (verified) |
| calculation root | `results/topic5_ictal_recruitment/field_concordance_grid_parent_matched/` |
| paper-ready staging root | `results/paper-ready-figure/fig3_ictal_field_concordance_grid_rebuild/` |

## B. Stage completion

| stage | required content | status | evidence / blocker |
|---|---|---|---|
| Stage 1A | n=17/167 parent R3 cohort | verified | `parent_anchor_cohort.csv` |
| Stage 1B | same-event seven-band R3 statistics | verified | `multiband_cohort.csv`, `multiband_band_omnibus.json` |
| Stage 1C | same-input R2 paired sensitivity | verified | `parent_anchor_cohort.csv` (method=R2), `r2_r3_cohort_comparison.json` |
| Stage 1D | pure within-shaft + fs-edge sensitivity | verified | `within_shaft_*.csv`, `fs_edge_sensitivity.csv` |
| Stage 2A | Fig3-B R3 score provenance; no exemplar reselection | verified | `fig3b_r3_score_provenance.json` (cross-checks Stage 1 anchor to 1e-4) |
| Stage 2B | Fig3-C 7-subject R3 trajectories | verified | 7 subjects / 154 seizures / 66 windows; `peri_onset_r3/per_seizure_window.csv` (10164 rows) |
| Stage 2C | Fig3-C R3 spatial null + correction | verified | all-contact + within-shaft null, maxT + Maris-Oostenveld cluster recomputed from new R3 null via shared `_finalize` |
| Whole Figure 3 quantitative package | only if all applicable rows complete | complete | Stage 1 + Fig3-B + Fig3-C all R3 |

**措辞边界**：Figure 3 发作相关 field-similarity package 已统一迁移为 R3（Stage 1 cohort/七频带 + Fig3-B provenance + Fig3-C 7-被试轨迹/null/correction 全部 R3）。

## C. Implementation

| item | path / result |
|---|---|
| new R3 pure module | `src/topic5_gradient_grid_field.py` |
| Stage 1 runner | `scripts/run_topic5_figure3_ictal_grid_rebuild.py` |
| Stage 2 runner(s) | `scripts/paper_figures/build_fig3b_r3_score_provenance.py` (2A); `scripts/run_topic5_fig3c_peri_onset_r3.py` (2B/2C) |
| Stage 1 figures | `scripts/paper_figures/plot_fig3_ictal_field_concordance_grid_rebuild.py` |
| new tests | `tests/test_topic5_gradient_grid_field.py` (24 tests) |
| reused helpers | `corr_pair_mirror_invariant_signed` (R3 mirror), `make_field_scorer`/`score_field_batch` (R2), `make_contact_permutations`/`apply_fixed_permutations`/`fold_seizure_null_draws`/`paired_sign_flip_p` (null+fold), `scorers_from_interictal_record` (fingerprint gate), `window_activation`; Fig3-C reuses `_compute_shared_values`/`_permutation_indices`/`_finalize`/`_plot`/`_trajectory_source_provenance` |
| files modified | none of the frozen producers / old scorers; only new files added |
| files intentionally not modified | `src/topic5_template_axis_field.py`, `src/propagation_contact_plane_readout.py`, frozen field JSON/NPZ, old R2 parent runner and outputs |
| old results preserved | yes — new parallel calc root + staging root; old `tspectral_field_concordance/` and old peri-onset spatial-null tree untouched |

## D. Primary cohort contract verification

| check | expected | actual | status |
|---|---:|---:|---|
| subjects | 17 | 17 | verified |
| unique `(subject,seizure_idx)` | 167 | 167 | verified |
| strict-broadband events | 106 | 106 | verified |
| gamma-nonBB events | 61 | 61 | verified |
| strict/gamma overlap | 0 | 0 | verified |
| Epilepsiae subjects | 16 | 16 | verified |
| Yuquan subjects | 1 | 1 | verified |
| shared / own-fallback | 7 / 10 | 7 / 10 | verified |
| primary bands | 7 | 7 | verified |
| events present in every band | 167 | 167 | verified |
| events present in parent anchor | 167 | 167 | verified |
| min common contacts | 6 | 6 | verified |
| median common contacts | 11 | 11 | verified |
| max common contacts | 16 | 16 | verified |
| input hashes unchanged | yes | yes | verified |
| routing constant across outcomes/bands | yes | yes (from planes only) | verified |
| one sigma per subject | yes | yes (`shared`/`own_a` plane sigma) | verified |
| A/B retain own support | yes | yes | verified |
| own A/B retain own planes | yes | yes | verified |
| same mapping across bands/anchor/R2/R3 | yes | yes (one perms array/event) | verified |
| seizure→subject fold | yes | yes (`fold_seizure_null_draws`, no pooling) | verified |
| pure within-shaft has no fallback | yes | yes (min_group=4, unavailable-marked) | verified |

### Fixed event inventory by subject

| subject | n parent events | route | sigma | min common contacts | notes / deviation |
|---|---:|---|---:|---:|---|
| epilepsiae_1077 | 4 | own_fallback | — | ≥6 | |
| epilepsiae_1084 | 56 | shared | 0.466 | ≥6 | |
| epilepsiae_1096 | 3 | own_fallback | — | ≥6 | |
| epilepsiae_1125 | 8 | own_fallback | — | ≥6 | |
| epilepsiae_1146 | 17 | shared | 0.0977 | ≥6 | |
| epilepsiae_1150 | 2 | own_fallback | — | ≥6 | |
| epilepsiae_139 | 1 | shared | — | ≥6 | `ripple_high fs_edge_flag` |
| epilepsiae_253 | 1 | own_fallback | — | ≥6 | `ripple_high fs_edge_flag` |
| epilepsiae_384 | 4 | shared | — | ≥6 | |
| epilepsiae_442 | 16 | own_fallback | — | ≥6 | |
| epilepsiae_548 | 15 | shared | — | ≥6 | |
| epilepsiae_590 | 5 | shared | — | ≥6 | |
| epilepsiae_620 | 2 | own_fallback | — | ≥6 | |
| epilepsiae_635 | 10 | own_fallback | — | ≥6 | |
| epilepsiae_922 | 15 | own_fallback | — | ≥6 | |
| epilepsiae_958 | 7 | shared | — | ≥6 | |
| yuquan_xuxinyi | 1 | own_fallback | — | ≥6 | EEG-onset only |

（per-subject sigma + grid fingerprints 全量见 `field_routing_sigma_grid_inventory.csv` / `support_overlap_inventory.csv`。）

## E. R3 grid contract

| check | expected | actual | status |
|---|---|---|---|
| primary grid | adaptive bounds, N=81 | adaptive per-plane, N=81 | verified |
| resolution sensitivity | same bounds, N=161 | N=161 same bounds | verified |
| y axis symmetric | exact | `Y = 0.5*(Y - flip(Y))`, exact | verified |
| support threshold | 0.15 | 0.15 | verified |
| overlap min at N=81 | 25 pixels | 25 | verified |
| overlap min at N=161 | 99 pixels | 99 | verified |
| support region touches boundary | no | no (asserted per event/grid) | verified |
| grid derived without ictal outcome | yes | yes (geometry/support/sigma only) | verified |
| grid fingerprints stored | every model | sha256 per grid | verified |
| 81/161 eligibility mismatch | none | none | verified |
| event×band `|r81-r161|` p95 | `<=0.02` | 0.0110 | verified |
| max subject data/null difference | `<=0.02` | 6/7 bands `<=0.02`; **beta subject data max-diff = 0.0212** | **deviation (marginal, reported)** |

Grid inventory path: `field_routing_sigma_grid_inventory.csv` + per-event `support_overlap_inventory.csv`

Largest resolution differences (subject-level, per band `max|r81-r161|`):

| band | max subject data diff | max subject null diff | action |
|---|---:|---:|---|
| beta_LVFA_low | 0.0212 | 0.0108 | report deviation; p95 event-level gate (0.011) passes; primary stays N=81 |
| hg_low_ripple | 0.0183 | 0.0104 | within gate |
| theta_preictal_PAC | 0.0165 | 0.0088 | within gate |
| delta_HYP_slow | 0.0145 | 0.0094 | within gate |
| others (α,γ,FR) | ≤0.0119 | ≤0.0123 | within gate |

## F. Null and permutation audit

| check | expected | actual | status |
|---|---|---|---|
| all-contact draws per event | 1000 | 1000 | verified |
| permutation seed depends on band | no | no (crc32 of subject:seizure ^ base) | verified |
| same mapping A/B | yes | yes | verified |
| same mapping seven bands | yes | yes | verified |
| same mapping parent anchor | yes | yes | verified |
| same mapping R2/R3 | yes | yes (one perms array reused) | verified |
| missing contacts remain missing | yes | yes | verified |
| every draw rebuilds ictal field/support | yes | yes (batch == per-draw reference, tested) | verified |
| every draw reselects mirror | yes | yes | verified |
| every draw reselects maxAB | yes | yes | verified |
| permutation hash collisions/mismatches | 0 | 0 (167 unique event hashes; deterministic rerun identical) | verified |
| within-shaft fallback count | 0 | 0 | verified |

Permutation audit artifact: `permutation_mapping_audit_summary.csv` (167 rows)

## G. Tests and validation

```text
commands:
  python -m pytest -q tests/test_topic5_gradient_grid_field.py            -> 24 passed
  python -m pytest -q tests/test_topic5_gradient_grid_field.py \
      tests/test_topic5_contact_similarity.py tests/test_topic5_axis_alignment.py -> 58 passed
  python scripts/run_topic5_figure3_ictal_grid_rebuild.py --validate-only -> cohort/routing/mask verified
  deterministic rerun (same seed, n_perm=20): parent/multiband/overlap/perm-audit/omnibus IDENTICAL
  git diff --check                                                        -> clean
```

| test family | status | evidence / failure |
|---|---|---|
| cohort/event lock | verified | validate-only + runner C1 gate |
| phenotype union/disjointness | verified | 106+61=167, overlap 0 |
| common contact mask | verified | min 6 / med 11 / max 16, 167/167 |
| fingerprint/input immutability | verified | `scorers_from_interictal_record` gate; hashes unchanged |
| routing | verified | shared set == expected 7 |
| fixed sigma | verified | one sigma per subject applied A/B, R2/R3, obs/null |
| adaptive bounds/grid hash | verified | `test_adaptive_grid_*`, boundary assertion |
| symmetric-y mirror | verified | `test_adaptive_grid_is_y_symmetric_and_flip_is_mirror` |
| corrected abs-max mirror adversarial case | verified | `test_score_template_selects_abs_max_over_identity_and_mirror` |
| A/B plane/support separation | verified | `test_own_route_builds_separate_grids_and_support_per_template` |
| support/overlap gates | verified | `test_score_template_matches_legacy_corr_pair_primitive` |
| coherent permutations | verified | deterministic-rerun identical perm-audit hashes |
| full null recomputation | verified | `test_batch_scoring_equals_per_draw_reference` |
| subject-first fold | verified | `fold_seizure_null_draws`, no seizure pooling |
| seven-band maxT | verified | `test_seven_band_maxt_pfwer_matches_reference_formula` |
| direct band test | verified | `test_direct_band_omnibus_*`, `test_direct_band_contrasts_yield_21_pairs_with_holm` |
| R2/R3 paired construction | verified | shared inputs/mask/perms; `r2_r3_cohort_comparison.json` |
| pure within-shaft no fallback | verified | `test_within_shaft_*` (min_group=4) |
| 81/161 convergence | verified | `test_overlap_min_matches_prescribed_formula`; p95=0.011 (beta subj marginal, §E) |
| endpoint R3 primitive regression | verified | `test_grid_field_reproduces_legacy_smooth_field_on_fixed_grid` |
| deterministic rerun | verified | bit-identical key CSVs across two seeds |
| Fig3-B no reselection | verified | subject/seizure hard-locked; cross-check 0.7350 == anchor observed |
| Fig3-C fixed-time mapping | verified | reuses `_permutation_indices` (per seizure×replicate, 66-window fixed) |

## H. Parent R3 cohort result

| group | n subject / event | R3 data median [IQR] | all-contact null median [IQR] | margin median | n positive | one-sided Wilcoxon p | sign-flip p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pooled phenotype-matched | 17 / 167 | 0.768 [0.62,0.88] | 0.679 | +0.041 | 12 | 0.0224 | 0.058 |
| Strict broadband | 16 / 106 | 0.715 | 0.662 | +0.038 | 10 | 0.188 | 0.263 |
| Gamma non-BB | 11 / 61 | 0.728 | 0.734 | +0.014 | 7 | 0.260 | 0.686 |

（coherent cohort spatial-null p 与全量 IQR 见 `parent_anchor_cohort.csv`；本表 spatial-null 与 sign-flip 均已落盘。）

Short interpretation within claim boundary:

```text
测了什么：发作刚开始那 10 秒里，哪些触点能量最强，把这张"能量热区"跟这个病人平时
（发作间期）HFO 传播先后顺序画成的"时序场"在空间上比一比，看它们像不像。
怎么测的：如果这两张图毫无关系，那把触点标签随机打乱后算出的相似度，应该跟真实的一样。
把 17 个病人各自的相似度中位数拿出来对比——真实的比"打乱标签"的高，中位数差 +0.041，
单侧检验 p=0.022；17 个里 12 个病人真实高于自己的随机基线。
揭示了什么：在"整个网络骨架"这个粗尺度上，发作早期能量分布看起来确实跟间期 HFO 时序场
对得上，比随机打乱强。但这只说"粗粒度上像"——不能说发作按间期顺序逐个触点重放，也不是
每个病人单独都显著，Broadband / Gamma 两个子群单独看没到显著。
（内部归档代号：R3 dense-grid maxAB, all_phenotype_matched, all-contact channel-shuffle null, sigma_common）
```

## I. Seven-band R3 inheritance result

| band | n subject / event | R3 data median | null median | margin (cohort Δ median) | n positive | Wilcoxon p | seven-band pFWER |
|---|---:|---:|---:|---:|---:|---:|---:|
| δ `[1,4)` | 17 / 167 | 0.826 | 0.694 | +0.074 | 15 | 0.00042 | **0.002** |
| θ `[4,8)` | 17 / 167 | 0.796 | 0.697 | +0.057 | 12 | 0.0253 | **0.041** |
| α `[8,13)` | 17 / 167 | 0.735 | 0.694 | +0.047 | 10 | 0.259 | 0.656 |
| β `[13,30)` | 17 / 167 | 0.690 | 0.692 | +0.026 | 13 | 0.0544 | 0.977 |
| γ `[30,80)` | 17 / 167 | 0.782 | 0.697 | +0.022 | 11 | 0.142 | 0.127 |
| R `[80,150)` | 17 / 167 | 0.743 | 0.693 | +0.022 | 11 | 0.322 | 0.522 |
| FR `[150,250)` | 17 / 167 | 0.714 | 0.695 | +0.060 | 12 | 0.066 | 0.878 |

（δ 与 θ 过七带联合 maxT 校正；其余不过。**星号只对应联合 pFWER**，不得据此判频段间强弱——见 §J。）

## J. Direct band-specificity test

| item | result |
|---|---|
| complete Delta matrix | 17×7 |
| Friedman statistic | 9.05 |
| within-subject band-label permutations | 100000 |
| calibrated omnibus p | 0.169 |
| Kendall's W | 0.089 |
| pairwise contrasts | 21 (`multiband_band_contrasts.csv`) |
| pairwise method | two-sided Wilcoxon + Holm |
| evidence for direct band effect? | report only — no evidence |

Interpretation:

```text
测了什么：七个频段各自算了一个"真实减随机"的差值，想看这七个频段彼此有没有强弱差别。
怎么测的：如果七个频段其实一样、只是随机波动，那把每个病人的七个频段值在他自己内部随机
重排，重排后的"频段间差异"统计量应该跟实测一样大。10 万次重排校准，实测统计量落在随机
分布中间，校准 p=0.169，一致性系数很小（0.089）。
揭示了什么：七个频段之间看不出真正的强弱差别。虽然低频 δ/θ 单独看过了联合校正，但"某个
频段有星、另一个没有"不能推断这两个频段不同——在这个数据上，效应看起来是"铺开在各频段"
的（band-generic），没有赢家频段。
（内部归档代号：Friedman + within-subject band-label permutation, Kendall W, Holm 21-contrast）
```

## K. Paired R2–R3 diagnostic

| metric | Pooled | Broadband | Gamma |
|---|---:|---:|---:|
| R3 data | 0.768 | 0.715 | 0.728 |
| R2 data | 0.842 | 0.794 | 0.842 |
| R3 null | 0.679 | 0.662 | 0.734 |
| R2 null | 0.799 | 0.753 | 0.824 |
| R3 margin | +0.041 | +0.038 | +0.014 |
| R2 margin | +0.046 | +0.052 | +0.003 |
| median `R3−R2` (pooled subject data) | −0.044 | | |
| subject margin sign concordance | 0.88 | | |
| Spearman rho (R2 vs R3 data) | 0.92 | | |
| paired two-sided Wilcoxon p (R3 vs R2 data) | 0.0005 | | |

Were old R2 summaries reused? `no — R2 was rerun on identical inputs, mask, routing, sigma_common, activation, permutation, fold in the same pass.`

Claim boundary: `no equivalence test was prespecified or run; do not write "equivalent". R2/R3 agree on subject ordering (rho=0.92) while R3 sits ~0.04 below R2 in level.`

## L. Pure within-shaft sensitivity

| item | result |
|---|---|
| min group size | 4 |
| fallback allowed | no |
| eligible subjects | 2 |
| eligible events | 18 |
| parent result artifact | `within_shaft_cohort.csv` / `within_shaft_subject.csv` |
| per-band result artifact | `within_shaft_multiband_cohort.csv` / `within_shaft_multiband_subject.csv` |
| separate denominator shown | yes (2 subjects / 18 events, NOT the figure n=17) |
| interpretation ceiling | coarse scaffold (within-shaft underpowered) |

Interpretation:

```text
测了什么：更严格的对照——只在同一根电极杆内部打乱触点标签（保留"哪根杆是热的"这个植入
几何），看相似度还剩多少。
怎么测的：要求每根杆至少 4 个有效触点、且一次发作的所有有效触点都落在这种够大的杆里，
否则这次发作算"不合格"。这样合格下来只有 2 个病人 / 18 次发作。
揭示了什么：合格样本太小（2 个病人），虽然差值中位数看着大（+0.20），但 n=2 根本分辨不开
（p=0.25）——这一层我们没看清。所以主张上限只能停在"粗粒度网络骨架对得上"，不能升级成
"杆内特异"。
（内部归档代号：within_shaft min_group_for_shaft=4, no-fallback unavailable-marking）
```

## M. Prespecified fs-edge sensitivity

| sensitivity | result / path |
|---|---|
| primary seven-band family keeps E139/E253 | ripple_high: 17 subj, Δ median +0.060, Wilcoxon p=0.066 (`fs_edge_sensitivity.csv`) |
| exclude E139/E253 from `ripple_high` | 15 subj, Δ median +0.063, Wilcoxon p=0.028 (sidecar, not substituting main family) |
| `ripple_safe_80_220` sidecar | not run (deviation — see §Q; `ripple_safe_80_220` cache exists; exclude-subject sidecar computed instead) |

## N. Stage 2A — Fig3-B provenance

| check | expected | actual | status |
|---|---|---|---|
| subject / seizure | E1146 / 2 | E1146 / 2 (hard-locked) | verified |
| exemplar reselected | no | no | verified |
| activation | BB150, clinical `[0,10] s` | bb150_auc [0,10]s (Stage 1 anchor) | verified |
| statistical R3 score | computed with Stage 1 engine | maxab 0.7350 (best A; overlap 1519/411) | verified |
| paired R2 score | same inputs | maxab 0.7379 | verified |
| display field | separately labeled 6 mm | `display_field_6mm` kept separate from sigma_common=0.0977 | verified |
| metadata/checkpoint updated in staging | yes | `fig3b_r3_score_provenance.json` | verified |
| locked production figure overwritten | no | no | verified |
| cross-check vs Stage 1 anchor | equal | 0.734991 == provenance 0.7350 | verified |

Fig3-B R3 provenance path: `results/paper-ready-figure/fig3_ictal_field_concordance_grid_rebuild/fig3b_r3_score_provenance.json`

## O. Stage 2B/C — Fig3-C R3 trajectory and null

| check | expected | actual | status |
|---|---:|---:|---|
| shared-only subjects | 7 | 7 (1084,1146,384,548,583,590,958) | verified |
| own fallback | 0 | 0 (all shared route) | verified |
| time range | `[-120,+20] s` | `[-120,+20] s` | verified |
| window / step | 10 s / 2 s | 10 s / 2 s | verified |
| window centers per complete seizure | 66 | 66 (asserted) | verified |
| successful seizure set | same as canonical manifest | 154 seizures (1084:71,1146:24,384:6,548:26,583:3,590:12,958:12) | verified |
| R3 maxAB recomputed | yes | yes (dense-grid varmask) | verified |
| signed A/B recomputed | yes | yes (`per_seizure_window.csv`) | verified |
| one mapping per seizure×draw across all windows | yes | yes (`_permutation_indices` reused) | verified |
| all-contact null recomputed | yes | yes | verified |
| pure within-shaft null recomputed | yes | yes (engine within_shaft) | verified |
| maxT correction recomputed | yes | yes (from new R3 null via `_finalize`) | verified |
| cluster correction recomputed | yes | yes (Maris-Oostenveld from new R3 null) | verified |
| old R2 p/cluster reused | no | no (parallel `peri_onset_r3/` tree) | verified |

### Fig3-C subject inventory (R3, within-shaft = primary null)

| subject | seizures | coverage tier | within-shaft: maxT p<0.05 windows | within-shaft cluster-sig windows (n clusters) | all-contact maxT p<0.05 |
|---|---:|---|---:|---:|---:|
| epilepsiae_583 | 3 | severely_partial | 0 | 3 (4) | 1 |
| epilepsiae_384 | 6 | partial_ok | 0 | 0 (2) | 0 |
| epilepsiae_590 | 12 | complete_ok | 0 | 0 (1) | 0 |
| epilepsiae_958 | 12 | complete_ok | 4 | 17 (9) | 2 |
| epilepsiae_1146 | 24 | partial_ok | 17 | 29 (11) | 16 |
| epilepsiae_548 | 26 | complete_ok | 0 | 0 (5) | 43 |
| epilepsiae_1084 | 71 | partial_ok | 0 | 4 (4) | 2 |

（全窗 66/subject；`per_seizure_window.csv` 10164 行 = 154 seizures×66；observed maxAB best_template A=5550/B=4614；per-window null/p/cluster 见 `spatial_null_stats.csv`，null 矩阵 `spatial_null_matrices.npz`。）

Fig3-C result summary within claim boundary:

```text
测了什么：对 7 个"有共享传播骨架"的病人，看发作前后两分多钟（发作前 120 秒到发作后 20 秒）里，
每隔 2 秒取一个 10 秒窗口，逐窗算"发作能量场跟间期骨架的相似度"，看它随时间怎么走、
在发作起始附近是不是抬起来。
怎么测的：读出层从"在触点位置比"（R2）整体换成"在密网格上比"（R3），其它一切不变——同一批
成功发作、同一套 66 个时间窗、每次发作×每次随机重排只抽一次触点映射并贯穿全部 66 窗、两种
零假设（全触点打乱=弱、只在电极杆内打乱=强）。显著区间（maxT 逐窗、cluster 连片）全部从新
的 R3 零分布重算。
揭示了什么：这是**逐病人**材料，不是队列统计——7 个病人差别很大。在更强的"杆内打乱"零假设下，
E1146（17 个 maxT 窗 / 29 个连片显著窗）和 E958（4 / 17）能看到清楚的发作前后结构，相似度轨迹
系统性高于杆内零假设并在起始附近抬起；E1084/E583 只有零星小连片；E384/E590/E548 在杆内零假设
下看不到。所以对个别病人（尤其 E1146）峰值时序图上像"发作起始附近场一致性抬高"，但这**不是
cohort 主张**，也不能推广到 7 人。全部结果（含 0 显著的 3 个病人）完整落盘，不按显著性筛选。
（内部归档代号：fig3c peri_onset R3 fixed-mask maxab, per_seizure_per_replicate fixed mapping, within_shaft vs all_contact, Nichols-Holmes maxT, Maris-Oostenveld cluster）
```

## P. Figure QA

| figure | PNG | PDF | metadata | README | visual check | CSV/metadata parity |
|---|---|---|---|---|---|---|
| `field_concordance_cohort_stat` | yes | yes | yes | yes | eyeballed (clean 3-group paired) | yes |
| `multiband_field_concordance_stat` | yes | yes | yes | yes | eyeballed (δ,θ red = pFWER) | yes |
| `r2_vs_r3_sensitivity` | yes | yes | — | yes | eyeballed (ρ=0.92 scatter) | `r2_r3_cohort_comparison.json` |
| `multiband_within_shaft_sensitivity` | yes | yes | — | yes | eyeballed (n=2 caveat labelled) | yes |
| Fig3-B staging candidate | provenance JSON | — | yes | — | locked figure preserved | 0.7350 cross-check |
| Fig3-C R3 figures (7 subjects) | yes | yes | — | yes | eyeballed (E1146 clear peri-onset rise vs within-shaft null) | `spatial_null_stats.csv` |

## Q. Drops and deviations

No result-based exclusion or method change was applied. All 167 events scored; `drop_inventory.csv` is empty.

| level | subject/event/band/model | reason | expected by contract? | denominator effect | action |
|---|---|---|---|---|---|
| resolution | beta_LVFA_low | subject data max-diff 0.0212 > 0.02 (161 vs 81) | prespecified engineering gate | none (primary N=81) | reported; p95 event gate (0.011) passes; not resolution-selected by significance |
| fs-edge sidecar | ripple_safe_80_220 | computed exclude-E139/E253 sidecar instead of the 80–220 band recompute | handoff allowed EITHER exclude-subjects OR ripple_safe | none | reported; exclude-subject sidecar delivered; 80–220 recompute available as follow-up |
| within-shaft | cohort | eligible denominator 2 subjects / 18 events | expected (strict min_group=4, no fallback) | secondary only, separate denominator | reported; interpretation ceiling = coarse scaffold |
| test file name | — | `tests/test_topic5_gradient_grid_field.py` (handoff suggested `..._parent_matched_grid_field_concordance.py`) | naming only | none | reported; same coverage |
| Stage 2B/C | Fig3-C | full 7-subject R3 run finalizing in background (154 seizures) | not a stop condition | none | run completing; §O numbers to be backfilled |
| Fig3-C readout impl | — | R3 null uses the fixed-mask fast batch (`score_event_maxab_batch`), exact because a permuted complete window keeps every source contact finite (constant ictal mask); guarded by a per-window partial-window assertion (0<n_finite<n_source raises) — E583/E384 passed | optimization, exact | none | reported; keeps n_perm=1000 faithful, ~6× faster than per-row varmask |

Unresolved deviations:

```text
仅 beta 频段的 81↔161 subject-level 收敛差 0.0212 略超 0.02 门（event-level p95=0.011 过门），
按 handoff §3.7 "报告偏差后" 保留 N=81 主口径；未按显著性挑分辨率。
Fig3-C 全量数值待后台运行收尾回填（非硬 blocker，只是 154 次发作的谱重算 + R3 打分较慢）。
```

## R. Artifact inventory

| artifact | path | exists |
|---|---|---|
| contract manifest | `field_concordance_grid_parent_matched/contract_manifest.json` | yes |
| input hashes | `.../input_hashes_before_after.json` | yes (unchanged=true) |
| cohort/event inventory | `.../cohort_event_inventory.csv` | yes |
| routing/sigma/grid inventory | `.../field_routing_sigma_grid_inventory.csv` | yes |
| common-contact inventory | `.../common_contact_inventory.csv` | yes |
| support-overlap inventory | `.../support_overlap_inventory.csv` | yes |
| permutation audit | `.../permutation_mapping_audit_summary.csv` | yes |
| parent event/subject/cohort | `.../parent_anchor_{subject,cohort}.csv` | yes |
| multiband subject/cohort/omnibus/contrasts | `.../multiband_{subject,cohort}.csv`, `multiband_band_{omnibus.json,contrasts.csv}` | yes |
| multiband null draws | `.../multiband_subject_null_draws.npz` | yes |
| R2–R3 comparison | `.../r2_r3_{subject_comparison.csv,cohort_comparison.json}` | yes |
| 81/161 convergence | `.../r2_r3_grid_convergence.csv` | yes |
| within-shaft results | `.../within_shaft_{subject,cohort,multiband_subject,multiband_cohort}.csv` | yes |
| within-shaft event inventory | `.../within_shaft_event_inventory.csv` | yes |
| fs-edge sensitivity | `.../fs_edge_sensitivity.csv` | yes |
| drop inventory | `.../drop_inventory.csv` | yes (empty) |
| summary JSON | `.../summary.json` | yes |
| figures + README | `fig3_ictal_field_concordance_grid_rebuild/figures/` | yes |
| Fig3-B provenance | `.../fig3b_r3_score_provenance.json` | yes |
| Fig3-C R3 package | `.../peri_onset_r3/` (subject_index, per_seizure_window, spatial_null_stats, subject_summary, spatial_null_matrices.npz, run_manifest, 7×figures, README) | yes |

## S. Final handoff summary

```text
1. Stage status: Stage1=complete (verified); Stage2A=complete (verified); Stage2B/C=complete (verified, 7 subjects / 154 seizures / 66 windows); whole-Fig3 quantitative package=complete (all R3).
2. Contract execution: verified; deviations = {beta 81/161 subject-diff 0.0212 marginal; fs-edge sidecar = exclude-subjects not ripple_safe_80_220; Fig3-C readout uses exact fixed-mask fast batch (partial-window-guarded)}. No result-based drop/reorder/method-switch.
3. Cohort: n_subject=17, n_event=167 (strict 106 / gamma 61), shared/own=7/10, min_common_contacts=6 (median 11, max 16), inputs unchanged.
4. Parent R3: pooled data=0.768, null=0.679, margin=+0.041, Wilcoxon p=0.022 (12/17 positive), sign-flip p=0.058; strict 16/106 margin +0.038 p=0.188; gamma 11/61 margin +0.014 p=0.260. Phenotype paths=parent_anchor_{cohort,subject}.csv.
5. Seven bands + direct band test: δ pFWER=0.002, θ pFWER=0.041 pass; α/β/γ/R/FR do not; direct band omnibus Friedman calibrated p=0.169 (Kendall W=0.089) -> band-generic, no winner (multiband_cohort.csv, multiband_band_{omnibus.json,contrasts.csv}).
6. R2-R3 + resolution + within-shaft: R2 rerun same inputs, pooled data=0.842/null=0.799/margin+0.046; R3-R2 median -0.044, Spearman rho=0.92, sign concordance 0.88, no equivalence claim. Resolution p95|r81-r161|=0.011 (beta subj 0.0212 marginal). Within-shaft eligible 2 subjects/18 events, margin +0.20 p=0.25 -> ceiling = coarse scaffold.
7. Fig3-B / Fig3-C: Fig3-B E1146 sz2 R3 maxab=0.7350 (==Stage1 anchor observed, no reselection), R2 paired 0.7379, 6mm display kept separate. Fig3-C = 7 shared-only subjects / 154 seizures / 66-window [-120,+20]s R3 trajectory + all-contact/within-shaft null + maxT/cluster in parallel peri_onset_r3/ tree (old R2 tree untouched); per-subject heterogeneous illustrative material — within-shaft maxT-significant windows: E1146=17, E958=4, others 0 (E1084/E583 small cluster-only); NOT a cohort claim.
8. New artifacts: calculation_root=results/topic5_ictal_recruitment/field_concordance_grid_parent_matched/; staging_root=results/paper-ready-figure/fig3_ictal_field_concordance_grid_rebuild/.
```

---

# ADDENDUM — 方法学敏感性重算与统计合同修复 (2026-07-19)

审阅发现 4 个问题，全部核实并修复；另按要求做了 §四/§五/§六 方法学审计。**未 commit；旧 calc root
（`field_concordance_grid_parent_matched/`）与旧 `peri_onset_r3/` 全部保留为审计证据。** 新产物写入
parallel trees。目标不是让更多频带显著，而是把统计口径修正、并回答 R2→R3 变化来自哪里。

## 新根目录
| 内容 | path |
|---|---|
| N=161 primary · subject_fixed | `results/topic5_ictal_recruitment/field_concordance_grid_method_sensitivity/n161_subject_fixed/` |
| N=161 primary · frozen_per_model | `.../field_concordance_grid_method_sensitivity/n161_frozen_per_model/` |
| 4-cell 统一 estimand | `.../field_concordance_grid_method_sensitivity/multiband_4cell_{estimand.csv,estimand_summary.json,tensors.npz}` |
| interictal LOO 判据 | `.../field_concordance_grid_method_sensitivity/interictal_sigma_policy_{loo_contact.csv,subject.csv,summary.json}` |
| Fig3-C pure min-4 | `results/paper-ready-figure/fig3_ictal_field_concordance_grid_method_sensitivity/peri_onset_r3_min4/` |
| 4-cell + N=161 图 | `results/paper-ready-figure/fig3_ictal_field_concordance_grid_method_sensitivity/{figures/,n161_subject_fixed/figures/}` |

## 审阅 4 项修复

**F1 — 缺预注册 coherent cohort spatial-null p（已修）**：runner 新增 `coherent_cohort_spatial_null_p`
（parent 三组 + 七带），并保存 `parent_anchor_subject_null_draws.npz` / `within_shaft_subject_null_draws.npz`。
这是**跨被试 median 的置换 p**（Tobs vs 逐 draw cohort null），不是被试-vs-自身-null 的 Wilcoxon。

| group | R3 data | null | margin | **coherent cohort spatial-null p** | (Wilcoxon 参考) |
|---|---:|---:|---:|---:|---:|
| Pooled (subject_fixed, N=161) | 0.770 | 0.678 | +0.042 | **0.037** | 0.022 |
| Pooled (frozen_per_model, N=161) | 0.766 | 0.678 | +0.032 | **0.023** | 0.044 |
| Strict (sf) | 0.710 | 0.661 | +0.039 | 0.200 | 0.161 |
| Gamma (sf) | 0.730 | 0.733 | +0.012 | 0.338 | 0.232 |

> 先前报告的 p=0.022 是 Wilcoxon；预注册的 cohort 置换 p 是 **0.037**（Pooled, subject_fixed）——仍 <0.05，
> 但口径不同，已更正。Strict/Gamma 子群 cohort 置换 p 不显著（0.20/0.34）。

**F2 — N=81 未过 subject 收敛门（已修）**：beta subject-data 差 0.0212>0.02。按 handoff §3.7"统一提升
分辨率并完整重跑"，改 **N=161 为 primary**（`--grids 161,81`），81 作 sensitivity。161-vs-81 event×band
p95=0.011。原 N=81 primary 口径撤回。

**F3a — within-shaft 用 BB150 uniform（已修）**：改 phenotype-matched（strict→BB150, gamma→gamma30）。
eligible 仍 2 subj/18 ev（Stage 1 min-4 本就正确），phenotype-matched 后 margin=0.188、cohort 置换 p=0.047
（n=2 极小，barely-powered，仍属 coarse scaffold 上限）。

**F3b — Fig3-C within-shaft 是 min-2 非 min-4（已修）**：新 min-4 driver
`scripts/run_topic5_fig3c_peri_onset_r3_min4.py`，用 `gg.within_shaft_permutations(min_group=4)` 无 fallback。
**代码算出只有 E1146 合格**（杆 [11,4]）；其余 6 人 within-shaft **unavailable**（NA + reason + small_shafts）。
→ **更正 §O**：E958 旧的 4 maxT / 17 cluster、E1084/E583 的零星 cluster 都是宽松 min-2 洗小杆的产物，
**不能再称 pure strong within-shaft evidence**；严格杆内证据在本队列只有 E1146 一例。E1146 min-4 = 16 maxT / 29 cluster。

**F5 — 缺审计产物（已修）**：新增 `parent_anchor_event.csv`、`r2_sensitivity_event.csv`、
`parent_anchor_subject_null_draws.npz`、`within_shaft_subject_null_draws.npz`、完整 grid bounds/spacing/hash
的 `field_routing_sigma_grid_inventory.csv`；manifest 记 `git_worktree_dirty` + "复现须用 working tree 非 commit
（新脚本未 tracked）"。

## §四 — R2/R3 × sigma-policy 2×2（build_event_scorer sigma_a/sigma_b + --smoothing-policy）

`gg.build_event_scorer` 加 `sigma_a/sigma_b`（向后兼容 `sigma=`；shared grid 要求 `sigma_a==sigma_b` fail-closed；
31 module tests 全绿）。runner 加 `--smoothing-policy {subject_fixed,frozen_per_model}`。验证：
**subject_fixed 逐比特复现 canonical**；**frozen_per_model 对 7 个 shared 被试完全一致（max|Δ|=0.0000）**，
只对 own 被试变化（own_b 用自己的 frozen sigma）。

## §五 — outcome-blind interictal LOO 判据（不看 ictal p 选 sigma）

留一触点间期 earliness 重建（`gg.loo_contact_reconstruction`），两 policy **几乎不可分**：cohort median AB
unweighted RMSE subject_fixed **0.759** vs frozen_per_model **0.757**；own-route-only **0.788** vs **0.789**（基本打平）；
support-weighted 0.685 vs 0.677。shared 恒等检查 max|Δ|=0.0。**结论：sigma 规则对间期几何支撑影响微乎其微，
两 policy 都不明显更优。**

## §六 — 统一 4-cell multiband estimand + joint 28-family maxT

D[s,b,m]/N[s,b,m,k]/c=median_k N/Eobs=D−c/Enull=N−c；Tobs[b,m]=median_subject Eobs（图中 cohort 横杠）；
Tnull[b,m,k]。per-cell 七带 maxT + **joint 28-family（4 cell × 7 band）maxT**。**向量化 vs 逐 draw 参考误差=0.0**。

| cell | joint-28 maxT 显著频带数(<0.05) | 说明 |
|---|---:|---|
| R3 · subject_fixed | 4 (δ,θ,α,FR) | |
| R3 · frozen_per_model | 2 (δ,θ) | |
| R2 · subject_fixed | 0 | |
| R2 · frozen_per_model | 0 | |

**回答审阅 Q1（Broadband R2→R3 变化来自哪）**：R3 两个 cell（不论 policy）在 joint-28 下都有显著频带、R2 两个
cell 都 0；且两个 policy 列几乎一样（sigma 规则改变很小）+ §五 LOO 两 policy 打平。→ **R2→R3 的变化主要来自
dense-grid 读出层，不是 own-fallback sigma 规则。** 但按合同边界：**一个 cell 显著、另一个不显著，不能证明
两种方法/policy 不同**（R2/R3 rho=0.92 高度相关）；承重结论是"δ/θ 正向 field-concordance 在多个 cell 共有"。

## 更正后的核心口径（朴素话）
在"整个网络骨架"粗尺度上，发作早期能量与间期时序场对得上、比随机打乱强，**用预注册的跨被试置换检验
p=0.037（Pooled, N=161 primary），两种平滑口径都成立**；效应铺在低频（δ/θ 过七带联合校正与 28-family），
不是某频段独有；这一"R3 密网格比 R2 触点更能分出信号"主要来自读出层而非 sigma 规则；最严格的杆内特异控制
（min-group-4）在本队列**只有 E1146 一例可做**，其余不可行——所以主张上限仍是 **coarse patient-specific
scaffold concordance**，未升级为逐触点 replay / 方向 / 因果 / 杆内特异。

（内部归档代号：coherent_cohort_spatial_null_p, N=161 primary, smoothing_policy subject_fixed/frozen_per_model,
sigma_a/sigma_b, within_shaft min_group=4 unavailable, loo_contact_reconstruction, 4-cell joint-28 maxT, Tobs/Tnull/Eobs/Enull）

---

# ADDENDUM-2 — P1 修复 (2026-07-19)

审阅第二轮 4 个 P1，全部核实+修复。

**P1-1 — dense-grid 归因谬误（已修）**：先前"R3 有星、R2 没星→变化来自 dense-grid"是显著性差异谬误。
改为**直接配对 contrasts**（Eobs band→subject 折叠，被试内配对）：`multiband_4cell_contrasts.csv`。

| contrast | median effect | 13/17 pos | paired Wilcoxon p | sign-flip p |
|---|---:|---:|---:|---:|
| **readout (R3 − R2)** | **+0.039** | 13 | **0.031** | 0.074 |
| sigma (frozen − subject_fixed) | 0.000 | 5 | 0.959 | 0.754 |
| interaction (readout × sigma) | 0.000 | 6 | 0.575 | 0.836 |

> 更正结论：R2→R3 的变化是**真实 readout effect**（+0.039，配对 p=0.031），**sigma 规则无效应**（p=0.96）、
> **无交互**（p=0.58）——用直接检验得到，不再靠"一个显著一个不显著"。（配 §五 LOO：两 policy 间期重建打平。）

**P1-2 — 两张主图统计口径（已修）**：cohort 图星号与 metadata 改用**预注册 permutation p**
（Pooled 0.037 / Broadband 0.200 / Gamma 0.338），非 Wilcoxon；图上加 n=17·167 / 16·106 / 11·61 与
"early-ictal onset 0–10 s"；Wilcoxon/sign-flip 降为 metadata sidecar。星号模式不变但方法标签正确。

**P1-3 — Fig3-C min-4 输出路径 bug（已修）**：旧 eligible-plot 复用了旧模块 `FIGDIR`，把 E1146 min-4 图写回
旧 `peri_onset_r3/`（覆盖旧图）、新 min-4 目录反缺 E1146。修：eligible 图改写入 min-4 目录（`_min4` 后缀，
不再走旧 `plot_subject_r3`）；旧 `peri_onset_r3/` 的 E1146 图已从**未被动过的旧 min-2 stats CSV** 用
`--rebuild-figures-only` 复原（旧结果保留声明恢复成立）。并把 Fig3-C **升到 N=161**（`FIG3C_GRID_N=161`）随
cohort primary，min-4 全量 161 重跑**已完成**（7 subjects，~3.6h）：E1146 已正确落入 min-4 目录
（`epilepsiae_1146_peri_onset_r3_min4.png`，路径修复经全量确认），within-shaft @161 = **14 maxT / 27 cluster
(3 intervals)**（对比 @81 的 16/29，分辨率差异小、结论不变——仍是唯一可做严格杆内的个例）；其余 6 人
unavailable。旧 `peri_onset_r3/`（min-2 @81）E1146 图保留完好。

**P1-4 — N=161 未成代码默认（已修）**：`GRID_PRIMARY=161, GRID_SENS=81`（默认 `--grids` 现为 161,81），
未来普通重跑不再退回已撤回的 N=81；收敛 CSV 列名改为随实际 grid 动态命名（`r{grids[0]}`/`r{grids[1]}`），
不再 r81/r161 写反；已把现有两个 n161 run 的收敛 CSV 列名就地更正。核心新脚本仍 untracked（用户禁止 commit），
manifest 已记 `git_worktree_dirty` + "复现须用 working tree 非 commit"——这是用户约束下的已知限制，非可在本轮内消除。
