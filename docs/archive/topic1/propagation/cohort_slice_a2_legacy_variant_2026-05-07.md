# Slice A2 cohort 扩容：7 个 Yuquan legacy-variant subject 加入扩展 cohort

> 日期：2026-05-07
> 范围：**仅 PR-1 / PR-2 cluster + PR-2.5 reproducibility + PR-6 anchoring 的 cohort 统计重算**。零 PR-3 / PR-4* / PR-5 重算。
> 双轨范围：本次同时维护 **n=33 primary cohort（withFreqCent uniform）** 与 **n=40 extended cohort（含 7 个 legacy variant）**。任何 cohort-level p 值 / Wilcoxon / Spearman / 占比都必须双轨披露。
> 上游主文档：`docs/topic1_within_event_dynamics.md`，`docs/archive/topic1/propagation/interictal_group_event_internal_propagation.md`，`docs/archive/topic1/propagation/cohort_slice_a1_2026-05-06.md`

## 1. 背景

[`cohort_slice_a1_2026-05-06.md`](cohort_slice_a1_2026-05-06.md) §1 列了 7 个 silent-failure subject (`zhangkexuan, pengzihang, songzishuo, zhangbichen, zhaochenxi, zhaojinrui, zhourongxuan`)，因为缺 `_gpu.npz` / `_refineGpu.npz` / `_lagPat_withFreqCent.npz`，被 Yuquan runner gate 排除。Slice A1 当时把它们留作"等 v2 cohort rebuild 完成后整批重做"。

复查（2026-05-07）：v2 cohort rebuild plan 当前**只覆盖 Epilepsiae**（Task 3.2 step 2: "Yuquan 路径的 use_gpu 默认仍 False；GPU 路径只在 Epilepsiae 用"；Task 5.1 注释明确 Yuquan-specific naming `_lagPat_withFreqCent` 不在 backfill 输出范围）。Yuquan 没有现成的 v2 路径。如果继续等，这 7 个 subject 的科学价值会被无限期 hold。

同时复查这 7 个的实际产物：所有 7 个**都有完整的 `_lagPat.npz` + `_packedTimes.npy` 配对**（11–13 blocks each, 100% paired），且 schema 完整（`lagPatRaw / lagPatRank / eventsBool / chnNames / start_t` 全有）；只缺 `lagPatFreq`。loader 的 fallback 分支已经在 Slice A1 修过——`load_subject_propagation_events` 在没有 `_lagPat_withFreqCent.npz` 时自动 glob `_lagPat.npz`。**唯一阻挡的是 runner gate**。

→ Slice A2 决定走 Path D：**接受这 7 个进入 cohort 作为独立 lineage stratum**，双轨报告 n=33 / n=40。后续 v2 Yuquan rebuild 完成后，n=33 primary 也会切到 v2 谱系，那时这 7 个再用 v2 重新接入。

## 2. Lineage 与 cross-PR 合同

**这 7 个 subject 的 lineage 与现有 33 个 cohort 不可比 1:1。** 关键差异：

| 方面 | n=33 primary（withFreqCent uniform） | n=7 legacy variant（Slice A2） |
|---|---|---|
| pack 脚本 | `p16_packGroupEvents..._withFreqCenter.py` | `p16_packGroupEvents..._refine_bool.py`（无 freqCenter 后缀） |
| 输出 | `*_lagPat_withFreqCent.npz` + `*_packedTimes_withFreqCent.npy` | `*_lagPat.npz` + `*_packedTimes.npy` |
| `lagPatFreq` 字段 | ✓（频率质心可用） | ✗ |
| `pickChn_thresh` / `packWinLen` | per-subject 在 legacy `sub_pickT_list_new` 字典里 | per-subject **不在** 字典里 → 当年另一脚本路径生成，**参数实际值不可考** |
| Detect / refine 谱系 | 21 年 cusignal vintage（10 Yuquan 原始 + 20 Epilepsiae）+ Slice A1 的 cuda_env pack | 21 年 cusignal vintage detect/refine + 21 年 vintage pack |

**chengshuai sanity check (2026-05-07，证明两份不等价)**：

| 变体 | n_ch (1st block) | n_events (1st block) | chnNames |
|---|---|---|---|
| `_lagPat.npz` | 6 | 2,965 | K5, K6, K7, K8, K9, K10 |
| `_lagPat_withFreqCent.npz` | 8 | 2,601 | E11, K3, K5, K6, K7, K8, K9, K10 |

→ 两份变体**不是同一组 group event**：channel 集合差（E11、K3 进出），event 总数差 12%。Slice A1 之所以投入工程做 cuda_env pack，就是为了把 chengshuai 这种变体差异 fix 掉，让 n=33 cohort 数值统一。Slice A2 的 7 个 subject **没有 withFreqCent 路径可走**——选项不是"老 vs 新"，是"老 vs 完全没有"。

引用 cross-PR 合同（`docs/archive/topic1/propagation/interictal_group_event_internal_propagation.md`）：

> Use `*_lagPat_withFreqCent.npz` (10ch full set), not `*_lagPat.npz` (older 7ch legacy slice).

→ **Slice A2 是这条合同的明确例外**。Runner gate 用显式 allow-list (`YUQUAN_LEGACY_VARIANT_SUBJECTS`) 而不是简单 fallback，避免后续 silent 污染。

## 3. 7 个 subject 的输入特征

数据来源：直接 `np.load` 各 subject `_lagPat.npz` 第一 block；SOZ 来自 `results/yuquan_soz_core_channels.json`（手工标注）。

| Subject | n_blocks | n_ch (1st block) | total events (~) | SOZ ch | SOZ ∩ lagPat | 备注 |
|---|---|---|---|---|---|---|
| zhangkexuan | 12 | 26 | 18,190 | 17 | 10/17 | 多电极 SOZ (A/C/E/F)，覆盖广 |
| pengzihang | 12 | 12 | 46,055 | 18 | 10/18 | 高 event 密度 |
| songzishuo | 12 | 38 | **447** | 21 | 12/21 | **极低 event 数**（38ch 但只 ~37 ev/block）；下游 PR-4 / PR-6 eligibility 大概率失败 |
| zhangbichen | 11 | 52 | 8,371 | 18 | 8/18 | 通道数最多 |
| zhaochenxi | 12 | 26 | 3,814 | 24 | 21/24 | SOZ 覆盖度高 (88%) |
| zhaojinrui | 13 | **4** | **81,141** | 2 | 2/2 | **极端 case**：只 4ch + SOZ=2，全部 lagPat 通道都在 SOZ 内（同 Slice A1 zhangjiaqi 模式） |
| zhourongxuan | 12 | **4** | 23,142 | 3 | 3/3 | **极端 case**：只 4ch + SOZ=3，全部 lagPat 通道都在 SOZ 内 |

**对下游 PR-4* / PR-6 SOZ-stratified 分析的提醒**（同 Slice A1 §9 zhangjiaqi 警告）：
- `zhaojinrui` 和 `zhourongxuan` 是"SOZ-only lagPat"模式 —— 没有 non-SOZ 对照通道。下游做 SOZ vs non-SOZ 比较时这两个 subject 必须排除或单独 case-study。
- `songzishuo` event 数过少（~447），PR-2 propagation_stereotypy 的 bootstrap 采样可能不稳定；PR-2.5 split-half / odd-even 也容易失败。

## 4. 工程改动

### 4.1 Runner gate

`scripts/run_interictal_propagation.py::_has_propagation_inputs` 改成显式 allow-list 模式：

```python
YUQUAN_LEGACY_VARIANT_SUBJECTS = frozenset({
    "zhangkexuan", "pengzihang", "songzishuo", "zhangbichen",
    "zhaochenxi", "zhaojinrui", "zhourongxuan",
})

def _has_propagation_inputs(dataset: str, subject_dir: Path) -> bool:
    if not subject_dir.exists():
        return False
    if dataset == "yuquan":
        if list(subject_dir.glob("*_lagPat_withFreqCent.npz")):
            return True
        if subject_dir.name in YUQUAN_LEGACY_VARIANT_SUBJECTS:
            return bool(list(subject_dir.glob("*_lagPat.npz")))
        return False
    return bool(list(subject_dir.glob("*_lagPat.npz")))
```

→ 7 个 allowlist subject 通过，原有 33 个 primary cohort gate 行为不变（仍要求 withFreqCent），其他未来潜在的 silent-failure subject（如未来再发现）默认仍被拒。

### 4.2 cohort manifest 双轨

| Manifest | 范围 | 输出 |
|---|---|---|
| `cohort_manifest_n33_2026-05-06.txt` | 33 primary withFreqCent | `pr1_cohort_summary.json`（默认路径，主表） |
| `cohort_manifest_n40_2026-05-07.txt` | 33 primary + 7 legacy variant | `pr1_cohort_summary_n40.json`（扩展表） |

aggregator 命令：

```bash
# Primary (n=33)
conda run -n cuda_env --no-capture-output python scripts/aggregate_propagation_cohort.py \
    --manifest results/interictal_propagation/cohort_manifest_n33_2026-05-06.txt

# Extended (n=40)
conda run -n cuda_env --no-capture-output python scripts/aggregate_propagation_cohort.py \
    --manifest results/interictal_propagation/cohort_manifest_n40_2026-05-07.txt \
    --out-summary results/interictal_propagation/pr1_subject_summary_n40.json \
    --out-cohort  results/interictal_propagation/pr1_cohort_summary_n40.json
```

主表用 default path（PR-2.5 / PR-6 driver 默认从这里读），扩展表用 `_n40` 后缀。这条规则**不要改**——下游 PR 都默认主表是 n=33。

### 4.3 PR-6 双轨

PR-6 driver 先一次跑全部 40 subject 的 per-subject anchoring。然后**先**复制 n=40 cohort summary，**再**通过物理移开 7 个 path-D per-subject anchoring 输出、重跑 `--cohort` 拿 n=33 baseline：

```bash
# Step 1: full run with all 40 subjects -> writes default cohort_summary.json (n=40 状态)
conda run -n cuda_env --no-capture-output python scripts/run_pr6_template_anchoring.py --all

# Step 2: snapshot n=40 cohort summary (在切到 n=33 之前)
cp results/interictal_propagation/template_anchoring/cohort_summary.json \
   results/interictal_propagation/template_anchoring/cohort_summary_n40.json

# Step 3: stash 7 path-D per-subject anchoring outputs
mkdir -p /tmp/pr6_pathd_stash
mv results/interictal_propagation/template_anchoring/per_subject/yuquan_{zhangkexuan,pengzihang,songzishuo,zhangbichen,zhaochenxi,zhaojinrui,zhourongxuan}.json \
   /tmp/pr6_pathd_stash/

# Step 4: re-run cohort step only -> default cohort_summary.json 切到 n=33 状态
conda run -n cuda_env --no-capture-output python scripts/run_pr6_template_anchoring.py --cohort
cp results/interictal_propagation/template_anchoring/cohort_summary.json \
   results/interictal_propagation/template_anchoring/cohort_summary_n33.json

# Step 5: restore stash so per_subject 目录回到 n=40 状态
mv /tmp/pr6_pathd_stash/*.json \
   results/interictal_propagation/template_anchoring/per_subject/
# 注意：default cohort_summary.json 此时仍然停在 n=33——这是有意为之，跟
# pr1_cohort_summary.json default 的 n=33 主表约定保持一致。要切到 n=40
# default 必须再跑一次 `--cohort`；本流程不做这一步。
```

**最终落盘约定**（**与 §4.2 manifest 双轨严格平行**）：
- `template_anchoring/cohort_summary.json` = **n=33 primary**（default，与 `pr1_cohort_summary.json` default 一致）
- `template_anchoring/cohort_summary_n33.json` = n=33 显式归档（与 default 同内容）
- `template_anchoring/cohort_summary_n40.json` = n=40 extended 显式归档

下游引用 PR-6 cohort 数字时**必须显式选 `_n33` 或 `_n40`**，不要直接读 `cohort_summary.json` 然后假设它代表哪一轨——尽管 default 是 n=33，但跑过 `--all` 之后短暂会变 n=40，依靠 default 文件名读 cohort 是 race condition。

### 4.4 PR-2.5 双轨

PR-2.5 是 per-subject 字段（`time_split_reproducibility`），不直接出 cohort summary。Cohort-level forward/reverse 数 / split-half 通过率从 `pr1_cohort_summary*.json` 间接得到（aggregator 跑两次）。

```bash
# Step 1: produce per-subject time_split_reproducibility for the 10 new subjects
conda run -n cuda_env --no-capture-output python scripts/run_interictal_propagation.py \
    --pr25 --dataset yuquan \
    --subjects zhangjiaqi gaolan wangyiyang \
               zhangkexuan pengzihang songzishuo zhangbichen zhaochenxi zhaojinrui zhourongxuan

# Step 2: re-aggregate cohort summaries (n=33 + n=40)
[见 4.2]
```

## 5. PR-1/PR-2 + PR-2.5 + PR-6 cohort 数值（2026-05-07 跑完填入）

### 5.1 7 个 Path D subject 的 PR-1 / PR-2 结果

| Subject | n_ch | n_events | n_blocks | mixture (strict / possible) | bias_fraction | mean_tau (all) | stable_k | PR-2.5 grade | FR (split-half) | FR (odd-even) |
|---|---|---|---|---|---|---|---|---|---|---|
| zhangkexuan | 26 | 18,190 | 12 | False / True | 0.893 | 0.283 | 2 | moderate | None | None |
| pengzihang | 12 | 46,055 | 12 | False / True | 0.925 | 0.155 | 2 | strong | None | None |
| songzishuo | 38 | 447 | 12 | False / True | 0.944 | 0.204 | 2 | moderate | None | None |
| zhangbichen | 52 | 8,371 | 11 | False / True | 0.982 | 0.080 | 2 | strong | None | None |
| zhaochenxi | 26 | 3,814 | 12 | False / True | 0.920 | 0.024 | 2 | strong | False | True |
| zhaojinrui | **4** | 81,141 | 13 | False / True | **0.000** | 0.029 | **5** | strong | True | True |
| zhourongxuan | **4** | 23,142 | 12 | False / True | **0.000** | 0.018 | **5** | strong | True | True |

**两个特殊 case 跟 Slice A1 的 zhangjiaqi 同模式**（单电极相邻触点）：
- `zhaojinrui` (4ch F5/F6/F7/F8 全在 SOZ)：`bias_fraction=0`、`stable_k=5`、`mean_tau=0.029`
- `zhourongxuan` (4ch G7/G8/G9/G10 全在 SOZ)：`bias_fraction=0`、`stable_k=5`、`mean_tau=0.018`

→ 这两个在 PR-6 audit 中 `endpoint_defined=False`（n_ch < 6 + k != 2），不进 H1 分子。其他 5 个 Path D subject 全部 `stable_k=2`、PR-6 `h1_primary_eligible=True`。

### 5.2 PR-1 / PR-2 cohort summary 对比

| 指标 | n=33 primary | n=40 extended | Δ |
|---|---|---|---|
| `n_strict_mixture` | 30 | **30**（**不变**） | 0 |
| `n_possible_mixture` | 3 | **10** | +7 |
| `mean_tau_median` | 0.0884 | 0.0845 | -0.0039 |
| `bias_fraction_median` | 0.6568 | 0.7110 | +0.0542 |
| `stable_k_distribution` | `{2:30, 4:2, 6:1}` | `{2:35, 4:2, 5:2, 6:1}` | +5 k=2, +2 k=5 |

**关键叙事不变**：cohort 主体仍然 `stable_k=2`（35/40），strict mixture 仍然 0，possible mixture 全部新加（7 个 Path D 都是 possible，跟 Slice A1 一致）。`mean_tau_median` 略降（path-D 多个低 tau case），`bias_fraction_median` 提升（path-D 大多数高 bias）。**两个 stable_k=5 的 outlier 是单电极相邻触点案例，从 cohort 主流叙事的角度等效于"4ch 不可分模式"**。

### 5.3 PR-2.5 双轨

| 指标 | n=33 primary | n=40 extended |
|---|---|---|
| Total subjects | 33 | 40 |
| `grade_distribution` | strong=26, moderate=7 | strong=31, moderate=9 |
| `split_half.median_match_corr` | 0.9074 | 0.8988 |
| `split_half.median_agreement` | 0.8871 | 0.8765 |
| `odd_even_block.median_match_corr` | 0.9825 | 0.9705 |
| `odd_even_block.median_agreement` | 0.8983 | 0.8822 |
| `forward_reverse.n_subjects_with_pairs` | 14 | 17 |
| `forward_reverse.n_reproduced` | 13 | 16 |

**Path D 贡献**：3 个新增 forward/reverse pair（zhaochenxi OR-only、zhaojinrui AND、zhourongxuan AND），其中 zhaochenxi 仅 odd-even 复现 → forward_reverse 复现率维持 13/14 → 16/17（同 ~93%）。`grade` 主流（strong）从 79% 提到 78%，整体稳定。

### 5.4 PR-6 双轨

| 指标 | n=33 primary | n=40 extended |
|---|---|---|
| `endpoint_defined` 通过 | 23/33 | 30/40（+5 path-D pass; +2 path-D fail [k!=2, n_ch=4]） |
| `h1_primary_eligible` 通过 | 23/33 | 28/40 |
| **H1 pooled** wilcoxon_greater n / median / p | 23 / 0.000 / 0.388 | 28 / 0.010 / 0.223 |
| H1 sign_test n+/N / p | (10+ / total) / 0.815 | (14+ / total) / 0.286 |
| H1 Yuquan-only median / p | 10 / 0.031 / 0.344 | 15 / 0.0625 / 0.107 |
| H1 Epilepsiae-only median / p | 13 / 0.000 / 0.551 | 13 / 0.000 / 0.551 |

**关键观察**：path-D 的 5 个 H1-eligible 把 **Yuquan-only median 从 0.031 提到 0.0625（×2）**，Wilcoxon p 从 0.344 降到 0.107（marginal）。Pooled p 从 0.388 降到 0.223。**整体趋势更接近"模板端点向 SOZ 富集"，但仍未达 α=0.05**。**双轨披露要求**：n=33 H1 pooled 是 null（accept-the-null），n=40 也是 null 但 effect size 增加。下游引用 PR-6 H1 时必须说明引用的是哪个 cohort。

H2 forward/reverse swap：两轨结构相同，per-subject swap_score 计算覆盖 PR-2.5 forward/reverse-pair subject。具体每 subject 数值见 `cohort_summary_{n33,n40}.json::h2_forward_reverse_swap`。

### 5.5 valid_mask_source contamination 审计（cross-PR 合同）

`scripts/_audit_pr6_valid_mask_source.py` 跑过：
- 5 个 path-D h1_eligible subject 全部 `valid_mask_source=raw_bools`（合同合规）
- 2 个 path-D 不在 audit 表内（endpoint_defined=False，未进 PR-6 anchoring 计算）
- **0 个 path-D 落入 `fallback_all_valid` 路径** — Path D 对 cross-PR 合同的"`valid_mask=None` 默认导致 buggy fallback 路径"风险**完全规避**。

**唯一 fallback case = `epilepsiae/1096`**，pre-existing 状态（Slice A1 / A2 都没引入），跟 Path D 无关；本 slice 不在 fix 范围内。**注意**：`1096` 在 n=33 与 n=40 H1 pooled 中都被纳入 eligible 集（`h1_primary_eligible=True`），所以 §5.4 报的 H1 p 值在两轨里都继承了这个 pre-existing fallback 污染——它的 endpoint/middle 走的是"`valid_mask=None` → all-channel 默认"的 buggy 路径，不是 raw_bools 路径。下游若要排除 1096 重跑 H1，需要专门 sensitivity PR。

### 5.6 PR-3 viz（A1 + A2 follow-up，2026-05-07）

PR-2 主跑后立刻 stage 的可视化，**不**在主 PR-2/PR-2.5/PR-6 cohort 重算口径之内，仅作 per-subject 临床可解释性 + paper-level cohort figure 准备：

- **A1 — per-subject viz × 10**：10/10 propagation heatmap + 10/10 MI distribution 落盘。脚本 `scripts/plot_interictal_propagation.py --pr3 --mi --dataset yuquan --subjects ...`。注意 `--pr3` 内部 `return` 早，所以 PR-3 propagation viz 与 MI viz 必须**分两次跑**（已分别落 `path_d_pr3_perpatient_2026-05-07.log` 与 `path_d_pr3_mi_2026-05-07.log`）。
- **A2 — cohort 6-panel × 2 tier**：n=33 默认表 + n=40 显式表分别落 `cohort_propagation_summary_n33.png` 与 `cohort_propagation_summary_n40.png`；default `cohort_propagation_summary.png` 留在 Tier 1 (n=33) 状态，与 §4.3 PR-6 default 约定一致。

**输出位置**：
- `results/interictal_propagation/figures/per_subject/yuquan_<sub>_propagation.png` × 10
- `results/interictal_propagation/figures/per_subject_mi/yuquan_<sub>_mi_distribution.png` × 10
- `results/interictal_propagation/figures/cohort_propagation_summary{,_n33,_n40}.png`

### 5.7 PR-7 directional template pairing（A3 follow-up，2026-05-07）

PR-7 的 cohort 来自 PR-6 audit 的 `forward_reverse_reproduced=True AND h1_primary_eligible=True`（h1_primary）/ `False AND True`（h2_negative）。Path D 因为只有 `_lagPat.npz`，需要 `run_pr7_template_pairing.py` 也加 fallback 才能跑动；这次 commit 加了 `PR7_LEGACY_VARIANT_ALLOWLIST`，与 §4.1 runner gate / PR-6 driver 同款 allowlist。

**Cohort 增长（PR-6 audit n=40 → PR-7 audit）**：

| 类别 | 原 (n=30 era) | 新 (n=40 audit) | Δ | 新加入 subject |
|---|---|---|---|---|
| h1_primary_pass (h1_eligible + fwdrev_repro) | 6 | 9 | +3 | zhangjiaqi (Slice A1, AND), wangyiyang (Slice A1, AND), zhaochenxi (Path D, OR-only) |
| h2_negative_pass (h1_eligible + ¬fwdrev_repro) | 17 | 21 | +4 | pengzihang, songzishuo, zhangbichen, zhangkexuan (4 Path D, h1_eligible+fwdrev_repro=False) |
| 不进 PR-7 cohort | — | 2 | — | zhaojinrui, zhourongxuan（fwdrev=True 但 n_ch=4 + k=5 → h1_eligible=False）；gaolan（fwdrev=False + h1_eligible=False） |

**N2 mark-independence main test (h1_primary, n=9) — INCONCLUSIVE-locked verdict 不变**：

| Δt | median_excess | wilcoxon_greater_p | sign_test_greater_p |
|---|---|---|---|
| 1 s | +0.0052 | 0.326 | 0.500 |
| 5 s | -0.0232 | 0.820 | 0.910 |
| **10 s (main)** | **-0.0061** | **0.898** | **0.910** |
| **30 s** | **-0.0085** | **0.935** | **0.910** |
| 60 s | -0.0071 | 0.898 | 0.746 |
| 300 s | -0.0056 | 0.963 | 0.910 |

**N3 (h1_primary)**：wilc(10s) p=0.936、sign(10s) p=0.980，median(10s)=−0.0164、median(30s)=−0.0086。**所有 wilcoxon p 远大于 0.05，所有 median 在 0 附近 ±0.01 内**——n=6→n=9 cohort 增长**没有翻盘 PR-7 的 INCONCLUSIVE 锁**。h2_negative cohort (n=21) 同向 NULL。

写作合同（与 PR-7 archive `pr7_template_pairing_results_2026-04-29.md` §17 一致）：仍然写 "compatible with mark-independent within tested precision (event-level Δt ∈ [10s, 30s] window, n=9 h1_primary)"，**不**写 PASS/FAIL，也**不**写"more evidence for mark-independence with n=9"——n=6→n=9 的统计功效提升微小，verdict 不变是预期内的。

**输出位置**：
- `results/interictal_propagation/template_pairing/pr7_cohort_audit.csv` (40 rows)
- `results/interictal_propagation/template_pairing/per_subject/yuquan_<7 new>.json`
- `results/interictal_propagation/template_pairing/cohort_summary.json` (n=9 / n=21)

## 6. 文件清单

新增：
- `results/interictal_propagation/cohort_manifest_n40_2026-05-07.txt`
- `results/interictal_propagation/pr1_cohort_summary_n40.json`（待生成）
- `results/interictal_propagation/pr1_subject_summary_n40.json`（待生成）
- `results/interictal_propagation/template_anchoring/cohort_summary_n33.json`（待生成）
- `results/interictal_propagation/template_anchoring/cohort_summary_n40.json`（待生成）
- 7 个 `results/interictal_propagation/per_subject/yuquan_<name>.json`（runner 输出）
- 7 个 `results/interictal_propagation/template_anchoring/per_subject/yuquan_<name>.json`（PR-6 输出）

代码改动：
- `scripts/run_interictal_propagation.py`：(a) `YUQUAN_LEGACY_VARIANT_SUBJECTS` 显式 allow-list；(b) `_has_propagation_inputs` 改成 primary + allowlist 双层 gate

跑日志：
- `results/run_logs/path_d_pr2_2026-05-07.log` (PR-1+PR-2)
- `results/run_logs/path_d_pr25_2026-05-07.log` (PR-2.5, 10 subjects)
- `results/run_logs/path_d_pr6_2026-05-07.log` (PR-6)

## 7. 引用规范

任何 cohort 数字必须双轨披露。例如：

✅ "Adaptive cluster `stable_k=2` 在 primary cohort (n=33) 上 30/33 (91%)；在 extended cohort (n=40) 上 X/40 (Y%)"
✅ "PR-2.5 split-half OR odd-even forward/reverse 复现 = 8/9 (primary n=33) / Z/W (extended n=40)"
❌ "整个 cohort `stable_k=2` 占 91%"（缺 n 标识，无法判断是 33 还是 40）
❌ "n=40 cohort 的 mean_tau median = ..."（不交代 lineage 混杂，下游会以为 40 是 uniform）

## 8. v2 完工后的迁移

当 v2 cohort rebuild 扩展到 Yuquan 后：

1. 整批 Yuquan（含这 7 个）按 v2 detector 重新 detect → 重新 refine → 重新 pack
2. 新输出落 `results/hfo_detector_v2/yuquan_propagation/`（参考 v2 Epilepsiae propagation 隔离规则）
3. v2 Yuquan cohort summary 取代当前 n=40 extended 作为新 primary
4. **本 archive (slice_a2) 不会失效** —— 它仍然是"在 v1/legacy 谱系下 7 个 silent-failure subject 怎么进 cohort、双轨怎么报"的凭证。但 §5 的具体数字会被新 cohort 取代，应在 v2 完工后加 deprecation banner。

## 9. 历史索引回链

- 主文档（精简口径）：`docs/topic1_within_event_dynamics.md` §3.1c
- 上游详细 archive：`docs/archive/topic1/propagation/interictal_group_event_internal_propagation.md`
- Slice A1 archive（3 lineage-adjacent subject）：`docs/archive/topic1/propagation/cohort_slice_a1_2026-05-06.md`
- v2 detector 主路径：`docs/archive/hfo_detector_v2/v2_specification.md`
- v2 cohort rebuild plan：`docs/archive/hfo_detector_v2/v2_cohort_rebuild_plan_2026-05-05.md`
- PR-6 plan：`docs/archive/topic1/pr6_template_anchoring/pr6_template_endpoint_anchoring_plan_2026-04-25.md`
