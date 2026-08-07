# Erin run form — Topic 5 parent-matched multiband field concordance

对应 handoff：`docs/archive/topic5/field_concordance_multiband_unified_handoff_2026-07-18.md`

> **SUPERSEDED — DO NOT USE**：对应 handoff 已被 R3 whole-Figure-3 重算合同取代。请使用 `docs/archive/topic5/fig3_ictal_gradient_r3_full_recompute_run_form_2026-07-18.md`。

> 请复制本表为运行报告后填写。Engineering contract 用 `verified / deviation / not run`；科学结果不使用 PASS/FAIL。

---

## A. Run identity

| field | value |
|---|---|
| runner | Erin |
| date/time | |
| branch | |
| commit | |
| worktree | |
| Python / NumPy / SciPy / pandas | |
| seed | expected `20260718` |
| n_perm | expected `1000` |
| output root | expected `results/topic5_ictal_recruitment/field_concordance_multiband_parent_matched/` |

## B. Implementation

| item | path / result |
|---|---|
| new runner | |
| new tests | |
| reused scorer helpers | |
| reused permutation/fold helpers | |
| files modified | |
| files intentionally not modified | |

## C. Contract verification

| check | expected | actual | status |
|---|---:|---:|---|
| subjects | 17 | | |
| unique `(subject,seizure_idx)` | 167 | | |
| strict-broadband events | 106 | | |
| gamma-nonbroadband events | 61 | | |
| Epilepsiae subjects | 16 | | |
| Yuquan subjects | 1 | | |
| shared / own-fallback subjects | 7 / 10 | | |
| bands | 7 | | |
| events present in every band | 167 | | |
| min seven-band common contacts | 6 | | |
| input hashes unchanged | yes | | |
| routing constant across bands | yes | | |
| one sigma per subject | yes | | |
| same mapping across bands | yes | | |
| pure within-shaft has no fallback | yes | | |
| seizure→subject fold | yes | | |

### Fixed cohort inventory

| subject | n parent events | field plane | sigma | min common contacts | fs-edge notes | deviation |
|---|---:|---|---:|---:|---|---|
| epilepsiae_1077 | 4 | | | | | |
| epilepsiae_1084 | 56 | | | | | |
| epilepsiae_1096 | 3 | | | | | |
| epilepsiae_1125 | 8 | | | | | |
| epilepsiae_1146 | 17 | | | | | |
| epilepsiae_1150 | 2 | | | | | |
| epilepsiae_139 | 1 | | | | `ripple_high fs_edge_flag` | |
| epilepsiae_253 | 1 | | | | `ripple_high fs_edge_flag` | |
| epilepsiae_384 | 4 | | | | | |
| epilepsiae_442 | 16 | | | | | |
| epilepsiae_548 | 15 | | | | | |
| epilepsiae_590 | 5 | | | | | |
| epilepsiae_620 | 2 | | | | | |
| epilepsiae_635 | 10 | | | | | |
| epilepsiae_922 | 15 | | | | | |
| epilepsiae_958 | 7 | | | | | |
| yuquan_xuxinyi | 1 | | | | EEG-onset only | |

## D. Tests

```text
command:

result:

```

| test family | status | evidence / failure |
|---|---|---|
| cohort/event lock | | |
| common contact mask | | |
| field routing/fingerprint | | |
| fixed sigma | | |
| corrected mirror + maxAB | | |
| coherent cross-band permutations | | |
| subject-first fold | | |
| seven-band family correction | | |
| pure within-shaft | | |
| deterministic rerun | | |
| input immutability | | |

## E. Parent anchor under the new fixed-sigma scorer

| group | n subject / event | data median | null median | margin median | n positive | one-sided p | sign-flip p |
|---|---:|---:|---:|---:|---:|---:|---:|
| pooled phenotype-matched | | | | | | | |
| strict broadband | | | | | | | |
| gamma non-BB | | | | | | | |

Difference from frozen-sigma provenance:

```text

```

## F. Seven-band primary all-contact results

| band | n subject / event | data median | null median | margin median [IQR] | n positive | raw p | seven-band pFWER |
|---|---:|---:|---:|---:|---:|---:|---:|
| δ `[1,4)` | | | | | | | |
| θ `[4,8)` | | | | | | | |
| α `[8,13)` | | | | | | | |
| β `[13,30)` | | | | | | | |
| γ `[30,80)` | | | | | | | |
| R `[80,150)` | | | | | | | |
| FR `[150,250)` | | | | | | | |

## G. Direct band-specificity test

| item | result |
|---|---|
| omnibus method | |
| omnibus statistic | |
| omnibus p | |
| multiple-comparison method | |
| evidence for band differences? | report only; no PASS/FAIL |

Pairwise/contrast artifact path:

```text

```

Short interpretation:

```text
If omnibus has no evidence, explicitly write band-generic even if individual bands have different stars.
```

## H. Pure within-shaft sensitivity

| item | result |
|---|---|
| eligible subjects | |
| eligible events | |
| shaft-size rule | expected min group 4, no fallback |
| per-band result artifact | |
| cohort denominator shown separately | |
| interpretation ceiling | coarse vs within-shaft |

## I. Prespecified fs-edge sensitivity

| sensitivity | result / path |
|---|---|
| exclude E139/E253 from `ripple_high` | |
| `ripple_safe_80_220` sidecar | |

## J. Drops and deviations

No-result-based exclusion is allowed.

| level | subject/event/band | reason | expected by contract? | effect on denominator | action |
|---|---|---|---|---|---|
| | | | | | |

Unresolved deviations:

```text

```

## K. Artifact inventory

| artifact | path | exists | notes |
|---|---|---|---|
| contract manifest | | | |
| cohort/event inventory | | | |
| field routing/sigma | | | |
| common-contact inventory | | | |
| permutation audit | | | |
| parent anchor tables | | | |
| multiband event table | | | |
| multiband subject table | | | |
| multiband cohort table | | | |
| null draws parquet | | | |
| band omnibus | | | |
| band contrasts | | | |
| within-shaft tables | | | |
| drop inventory | | | |
| summary JSON | | | |
| figures + README | | | |

## L. Final handoff summary

请严格按以下六行返回：

```text
1. Contract execution: verified / deviations listed
2. Cohort: n_subject=?, n_event=?, min_common_contacts=?
3. Parent anchor: data=?, null=?, margin=?, p=?
4. Bands vs all-contact: [完整七带简表或 artifact 路径]
5. Direct band test + pure within-shaft sensitivity: [结果与路径]
6. New artifacts: [canonical output root]
```
