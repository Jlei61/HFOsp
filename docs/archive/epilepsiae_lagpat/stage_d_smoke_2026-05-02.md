# Stage D Smoke — 12 stable subjects, new lagPat vs legacy

> 计划：[`epilepsiae_lagpat_backfill_plan_2026-04-29.md`](./epilepsiae_lagpat_backfill_plan_2026-04-29.md) §5 D.1 + D.2
> Stage C decision = `enter_smoke`（修复 Stage B 后），cohort 12 stable subjects = `548, 1077, 1084, 442, 818, 916, 922, 958, 583, 590, 620, 1125`
> 目的：验证下游 PR-1 propagation 与 PR-6 synchrony **接口能否吃新 lagPat**，并做**方向比对**。**不是 strict parity，也不主张新 lagPat 替代老 lagPat**——这是 secondary sensitivity，不是主结果重跑。

## Stage D 的本质

Stage A/B/C 已生成新 pipeline lagPat 并完成结构对齐审计。Stage D 接着问的现实问题是：

> 如果 Topic 1 下游分析不用 legacy lagPat，而改吃 Stage B 生成的新 lagPat，**关键结论的方向**还稳不稳？

具体只验证两件事：

1. PR-1 / PR-2 propagation 的 `stable_k` 分布——尤其 Topic 1 "stable_k 偏向 k=2" 这个口径是否依赖 legacy lagPat。
2. PR-6 synchrony 的 `mean_sync_phase_global`——是否与 legacy 方向一致，避免 PR-6 的 null/方向结论只是 legacy artifact。

Stage D 不重新主张 Topic 1 的全部科学结论；smoke 不出 cohort p-value。

## 工程改动（已落代码）

- `scripts/run_interictal_propagation.py`：新增 `--epilepsiae-root` 与 `--output-root` CLI flag；新增 `_epilepsiae_subject_dir(root, subject)` helper（先 probe `<root>/<subj>/all_recs/`，否则 fallback 到 `<root>/<subj>/`）；11 处 `subject_dir = root / subject if dataset == "yuquan" else root / subject / "all_recs"` 全部替换为 `_subject_dir(dataset, root, subject)`；`RESULTS_DIR` 在 `main()` 入口可被 `--output-root` 覆盖。
- `src/interictal_synchrony.py`：`run_epilepsiae_interictal_synchrony_from_manifest` 改为 layout-tolerant（先 probe `all_recs/`，否则 fallback flat）。
- `scripts/run_epilepsiae_interictal_synchrony.py`：新增 `--lagpat-root`、`--manifest`、`--output-dir` CLI flag。
- 测试：新增 `tests/test_interictal_propagation_root_override.py`（7 tests）、`tests/test_interictal_synchrony.py` 增 2 个 layout 测试（flat layout fallback、legacy precedence）。
- 全套相关 test suite 运行结果：**72 passed, 1 failed**（`test_select_core_penumbra_mask_rejects_empty_overlap` 在 HEAD 已失败，与本次 Stage D 改动无关，已 verify）。显式 deselect 该 pre-existing failure 后：**72 passed, 1 deselected**。

## 数据落地

- 新 lagPat 通过 Stage B 修复后的 cohort 重跑产出（commit `170624e`），3788 records，n_failed=0。
- 本次 smoke 输出：
  - `results/interictal_propagation/sensitivity_new_lagpat_smoke/per_subject/epilepsiae_<subj>.json` × 12
  - `results/interictal_propagation/sensitivity_new_lagpat_smoke/pr1_subject_summary.json`
  - `results/interictal_propagation/sensitivity_new_lagpat_smoke/pr1_cohort_summary.json`
  - `results/interictal_propagation/sensitivity_new_lagpat_smoke/_direction_comparison.json`
  - `results/interictal_synchrony/sensitivity_new_lagpat_smoke/_stable12_manifest.csv`（12-subject 子集，从 cohort manifest 过滤而来）
  - `results/interictal_synchrony/sensitivity_new_lagpat_smoke/epilepsiae_ready_full_artifacts_interictal_sync_summary.csv`（1843 blocks）
  - `results/interictal_synchrony/sensitivity_new_lagpat_smoke/epilepsiae_ready_full_artifacts_interictal_sync_events.csv`

Stage D 只覆盖 Stage C 的 12 stable subjects；Synchrony 只覆盖其中 9 个 `ready_full_artifacts` tier 的 subject（这是 PR-6 主流水线 pre-existing 的 tier filter，本次 smoke 没动）。**这是 smoke，不是全 cohort sensitivity**。

## D.1 Propagation 方向比对（12 stable subjects）

12/12 跑通，无 crash。stable_k 比对：

| subject | new_k | leg_k | match | 备注 |
|---------|-------|-------|-------|------|
| 548 | 2 | 2 | ✓ |  |
| 1077 | 6 | 2 | ✗ | new k 显著更高 |
| 1084 | 2 | 2 | ✓ |  |
| 442 | 2 | 2 | ✓ |  |
| 818 | 5 | 4 | ✗ | new k 略高 |
| 916 | 3 | 2 | ✗ | new k 略高 |
| 922 | 2 | 2 | ✓ |  |
| 958 | 2 | 2 | ✓ |  |
| 583 | 4 | 2 | ✗ | new k 显著更高 |
| 590 | 3 | 2 | ✗ | new k 略高 |
| 620 | 2 | 2 | ✓ |  |
| 1125 | 2 | 2 | ✓ |  |

**stable_k**：7/12 = 58% 完全一致；5/12 不一致**全部是 new_k > legacy_k**——单向漂移，不是噪声。

**漂移的可能原因（暂未定位）**：5 个 mismatch subject 的 new vs legacy 通道数对比是 1077 (5 vs 6)、818 (5 vs 5)、916 (6 vs 6)、583 (5 vs 7)、590 (16 vs 16)——通道数本身**不支持**"通道更多导致 cluster 更多"的简单解释。说明新 lagPat 在改变了**事件/lag pattern 的表达**（detector + refine + packing 联合产出），使 adaptive clustering 更容易把 patterns 拆成更多 cluster。**根因尚未定位**——可能是事件 timing 微差、lagPatRank 的 -1 sentinel 比例、或者 packing window 内的 lag 分布形状。下一步如要定位需做 per-cluster diagnostic。

**MI permutation 维度**：本次 smoke 中两侧 `mi_significance.p_value` 字段都是 `None`——这是 PR-1 默认行为（不跑 MI permutation），不是匹配。该维度本次 **not_evaluated**，要补需单独跑 PR-3 相关的脚本。**之前 _direction_comparison.json 里写的 `mi_significance_match_fraction = 1.0` 是 None==None 的伪匹配，不能算通过。已在解读时纠正口径**。

按 plan §5 enter_smoke 标准（"方向同号即可"）：propagation 的 7/12 完全一致 + 0/12 反向 + 0 crash 满足该标准；但 5/12 单向漂移本身就是 sensitivity audit 的一个重要 finding——不应被 enter_smoke pass 遮蔽。

## D.2 Synchrony 方向比对（9 stable + ready_full subjects）

9/12 stable subjects 在 `ready_full_artifacts` tier；442、818、1084 是 `ready_partial_artifacts`，PR-6 主流水线天然过滤。本次没改 tier filter——保持与 legacy PR-6 同样的 subject 范围。

block-level paired comparison（按 `(subject, block_stem)` join）：

- new rows: 1843，legacy rows: 1830，shared blocks: **1830** (only_new=13 是 stem-mismatch，已经被 join 排除)
- 4 个 block 是 NaN（pre-existing 边界条件），过滤后 1826 个 paired blocks 可用
- **`mean_sync_phase_global` Pearson r(new, legacy) = 0.916**
- new median = 0.4790；legacy median = 0.4809（差 < 0.5%）
- block-level same-side-of-cohort-median agreement: **1598/1826 = 87.5%**

按 plan §5 enter_smoke 标准：1830 paired blocks 上 r=0.916 + 中位数差 <0.5% + 87.5% block-level 同侧——**强方向一致**。

## 结论（口径修正版）

> Stage D smoke passed as an **interface and directionality check**.
>
> - **Synchrony is strongly concordant**: 1826 paired blocks, Pearson r=0.916 on `mean_sync_phase_global`, 中位数差 <0.5%, 87.5% same-side agreement. PR-6 synchrony 的方向结论**不是 legacy artifact**。
> - **Propagation shows one-sided sensitivity drift** toward higher `stable_k`: 7/12 exact match, 5/12 都是 new_k > legacy_k（单向漂移）。Topic 1 PR-2 "stable_k=2 cohort claim" 在新 lagPat 上**只能部分迁移**——必须以 sensitivity caveat 报告。

**不能写**（口径过满，证据不足）：

- ❌ "更多通道导致更多 cluster"（通道数不支持这个解释）
- ❌ "MI significance matched"（两侧 None，是 not_evaluated 不是 match）
- ❌ "Stage D 全面通过 / 全部通过"（只是 smoke，不是 full sensitivity）
- ❌ "新 lagPat 可替代 legacy lagPat"（plan §0 明确说不主张）

## 已知 gap（不阻塞 smoke，但需 follow-up）

1. **propagation k drift 根因**：5/12 单向漂移到更高 k，原因尚未定位。下一步如要定位需做 per-cluster diagnostic（lag pattern 形状、cluster sizes、AMI vs k 曲线对比）。**不应通过调算法把 mismatch "修掉"——那会污染 sensitivity audit**。
2. **MI permutation 维度未跑**：plan §5 D.1 step 4 表格中的 "MI significant fraction" 行 not_evaluated。如果 final sensitivity 报告需要这一项，要单独跑 `--pr3` 相关流水线。
3. **`ready_partial_artifacts` 三个 subject 没进 synchrony**：442、818、1084。如果以后要 partial-tier 的 sensitivity，需要单独改 manifest tier filter 或新流水线。
4. **Stage C outlier 没在本次 smoke**：253、139、1073、1150（large_drift）+ 1146（moderate_drift）——还在等单独 audit（数据对齐 / 实现错误，不调全局阈值）。

## 推荐下一步（按优先级）

1. **commit 工程 wiring**：CLI flag、helper、layout fallback、test——这些有独立价值，不应被 outlier 或 propagation drift 阻塞。建议 commit message: `feat(epilepsiae_lagpat): Stage D smoke wiring (--epilepsiae-root, --lagpat-root, layout fallback) + sensitivity comparison (72/72)`。
2. **outlier audit (Stage C)**：先审 253（jaccard 0.091, 仅 1 shared chn）和 1150（count_ratio 6.6）——只查数据对齐 / 实现错误，**不调全局阈值**。
3. **propagation drift 诊断**：5 subjects 漂向 k>2 的根因。Per-subject diagnostic：AMI vs k 曲线、cluster size 分布、lagPatRank -1 sentinel 比例。**只诊断不优化**，避免事后调算法污染 sensitivity audit。
4. **MI permutation 是否补**：等用户决定。如果要写 final sensitivity 报告"MI significance"维度，需要单独跑。

## 文件级 commit 建议

代码与测试在 working tree，**未 commit**：

修改：
- `scripts/run_interictal_propagation.py`
- `scripts/run_epilepsiae_interictal_synchrony.py`
- `src/interictal_synchrony.py`
- `tests/test_interictal_synchrony.py`

新增：
- `tests/test_interictal_propagation_root_override.py`
- `docs/archive/epilepsiae_lagpat/stage_d_smoke_2026-05-02.md`（本文）

Results 目录由 `.gitignore` 排除，不进 commit。
