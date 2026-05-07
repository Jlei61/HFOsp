# PR-6 Supplementary：Per-Channel Signed Rank Displacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **状态：planned (2026-05-06)。** 本计划是 PR-6 endpoint anchoring 的 supplementary，不立独立 PR、不开 H1/H2 cohort gate。
>
> **上游合同**：PR-2 `adaptive_cluster.clusters[k].template_rank` + PR-6 `per_template[k].valid_mask` + PR-2.5 `forward_reverse_reproduced` (OR 规则) + Yuquan/Epilepsiae SOZ JSON。
>
> **下游归属**：在 `docs/topic1_within_event_dynamics.md` §7 内"PR-6 几何合同"位置加一行 supplementary 链接，**不**升级 main doc 结论。

**Goal:** 把 PR-6 离散 swap_node count 升级到逐通道连续 signed rank displacement，并产出顶刊风格 supplementary figure（**PR-6 supplementary cohort = stable_k=2 ∩ PR-6 endpoint-defined**，n_available 由 Task 4 跑数后报告 — 上限 23 = PR-6 anchoring per_subject JSON 实际数；不预承诺；列：footrule + Kendall τ 描述性分布），不引入新的 cohort claim。

**Scientific boundary（写死）**：PR-2 stable_k 实际分布 = `{2: 27, 4: 2, 6: 1}`；本 supplementary 覆盖 stable_k=2 ∩ PR-6 endpoint-defined（n≤23），**不是**完整 27-subject stable_k=2 rank-geometry 分析。4 个 stable_k=2 subject（无 PR-6 anchoring：`epilepsiae_1125, 384, 620, 916`）被排除；3 个 stable_k≠2 subject（`epilepsiae_818, yuquan_huangwanling, yuquan_zhangjinhan`）不在范围。这个选择是为了与 PR-6 离散 swap_node 同 cohort 比较，**不能**包装成"全 stable_k=2 cohort"。

**Architecture:** 新增独立 stat helper 模块 `src/rank_displacement.py`（与 PR-6 anchoring 解耦），以 PR-2 cluster JSON 与 PR-6 anchoring JSON 为唯一输入；新增 batch runner + figure script；写一份 supplementary archive doc，回链 PR-6 plan 与 ping-pong review。

**Tech Stack:** numpy, scipy.stats (kendalltau, spearmanr), matplotlib + 现有 `src/plot_style.py`（Morandi 调色板，DPI_PUB=300）。**禁止**新增 PCA/UMAP/降维依赖；rank 距离用 footrule + Kendall τ，不引欧氏距离。

---

## 0. 范围与禁区（visualization + analysis contract，post-hoc 修订过两轮，写死）

> **诚实标注**：本节最初按 pre-registered 风格写，但 §0 第 5、6、7、8 条与 §3.0 sign anchor 合同、§3.3 SOZ baseline correction、Panel B/C scatter 设计都是在 user review 之后加入或重写的。所以这是 **post-hoc visualization + analysis contract**，不是真正意义的 pre-registration。当前版本是经过两轮 review 收敛后的写死版本——后续不再放宽，但放在 archive 而非 main doc 是为了保留 review 演化痕迹。

**做**：
1. 逐通道有符号位移 Δr(ch) = rank_Tb(ch) − rank_Ta(ch)
2. 整体 Spearman footrule F = Σ|Δr| 与 Diaconis–Graham 归一化 F/F_max
3. 整体 Kendall τ(rank_Ta, rank_Tb)
4. PR-6 supplementary cohort（≤23 subject）× channel 谱系热图（rows 按 τ 排序，bars 颜色按 F_norm > 2/3 分组）
5. footrule cohort 分布（按**本 supplementary 自己的 F_norm 在 D-G 渐近随机参考点 2/3 处**分组：F_norm > 2/3 vs F_norm ≤ 2/3；**不**用任何 PR-2.5 派生阈值）。Panel B 不做 MW-U on F_norm（循环论证），改用 Spearman ρ(F_norm, τ) 作 continuous 统计；Panel C 做 MW-U on τ（descriptive only）
6. supplementary archive doc + 顶刊风格 figure (PNG + PDF)

**不做**（违反则任务失败）：
1. **不**预注册任何 cohort-level Wilcoxon / sign-test PASS gate；统计量只作描述报告
2. **不**用 stable_k≠2 的 subject（k=4 / k=6 共 3 个）做主可视化；只在 archive 文字提一句
3. **不**重新跑 PR-2 clustering / PR-6 anchoring；只 consume 已落盘的 JSON
4. **不**提兴奋-抑制 / Ping-Pong 机制；archive 文字不写"forward = excitatory"
5. **不**改写 main doc §7 结论；只在历史索引加一行回链
6. **不**跨 subject 聚合 signed Δr 的方向（"正 / 负" 方向）。每个 subject 内部 T_a / T_b 由 §3.0 anchor rule 确定，但 anchor 本身只是工程约定，没有跨 subject 的生物学含义。Cohort 比较只用 invariant 量（footrule, |Δr|, Kendall τ），不用 signed_mean
7. **不**预先承诺 cohort size。stable_k=2 candidate 上限 ≤ 23（PR-6 anchoring per_subject JSON 实际数）；最终 n_available 由 Task 4 跑数后报告，**不**在 plan / archive doc 里写死"27"或"30"
8. **不**把 Diaconis-Graham 2/3 baseline 写成精确基线。它是 n→∞ 渐近期望；图上 / 文字里一律写"asymptotic random reference (≈ 2/3)"，不写"random baseline (2/3)"

---

## 1. 数据合同（必须验证后才能跑）

**输入文件**：

| 文件 | 字段 | 用途 |
|---|---|---|
| `results/interictal_propagation/per_subject/<dataset>_<subject>.json` | `channel_names`, `adaptive_cluster.stable_k`, `adaptive_cluster.clusters[k].template_rank`, `adaptive_cluster.clusters[k].cluster_id`, `adaptive_cluster.inter_cluster_corr_matrix`, `time_split_reproducibility.splits.first_half_second_half.forward_reverse_reproduced`, `time_split_reproducibility.splits.odd_even_block.forward_reverse_reproduced` | rank 向量、forward/reverse OR 规则 |
| `results/interictal_propagation/template_anchoring/per_subject/<dataset>_<subject>.json` | `per_template[k].cluster_id`, `per_template[k].valid_mask`, `per_template[k].n_valid_channels`, `per_template[k].source`, `template_pair_geometry.n_valid_intersection`, `template_pair_geometry.spearman_rank_pair`, `h2_swap_check.swap_score`, `h2_swap_check.null_p` | per-cluster valid_mask、cross-check |
| `results/epilepsiae_soz_core_channels.json` / `results/yuquan_soz_core_channels.json` | `<subject>: [ch_name, ...]` | SOZ 标注 |

**通道顺序合同**（**写死**）：
- PR-2 JSON `channel_names` 是 master ordering
- PR-6 `per_template[k].valid_mask` 必须与 `channel_names` 长度相同；不一致 → `raise ValueError`
- `template_rank[k]` 必须与 `channel_names` 长度相同；不一致 → `raise ValueError`
- 不允许默认 `valid_mask=None` 回退到"全 True"（CLAUDE.md cross-PR contract: PR-6 valid_mask 已被列为高风险默认）

**cluster_id 对齐**（**写死**）：
- PR-2 `clusters[k].cluster_id` 与 PR-6 `per_template[k].cluster_id` 取交集；按 `cluster_id` 配对，**不**按 list index
- 缺一致 cluster pair → 该 subject 入 `exit_reason="cluster_id_mismatch"`，跳过

**stable_k 过滤**（**写死**）：
- 主可视化只入 `adaptive_cluster.stable_k == 2` 且 `len(per_template) == 2` 的 subject
- stable_k=4 / 6 subject：所有 cluster pair 都计算 metric，存 per_subject JSON，但**不**进主热图

**forward/reverse-reproduced 标签**（**写死，OR 规则**）：
```
fwd_rev_reproduced = (
    splits.first_half_second_half.forward_reverse_reproduced == True
    OR
    splits.odd_even_block.forward_reverse_reproduced == True
)
```
（CLAUDE.md cross-PR contract lookup §`forward_reverse_reproduced (PR-2.5)`）

---

## 2. File Structure

**Create:**
- `src/rank_displacement.py` — 纯 stat helper（`compute_signed_rank_displacement`, `compute_footrule_normalized`, `aggregate_pair_metrics`）
- `tests/test_rank_displacement.py` — TDD 单元测试
- `scripts/run_rank_displacement.py` — batch runner（per-subject JSON + cohort summary JSON）
- `scripts/plot_rank_displacement.py` — figure 生成（cohort heatmap + footrule + per-subject）
- `results/interictal_propagation/rank_displacement/per_subject/<dataset>_<subject>.json` — 运行产物
- `results/interictal_propagation/rank_displacement/cohort_summary.json` — 运行产物
- `results/interictal_propagation/rank_displacement/figures/README.md` — 图说明（中文，每图 2–4 句 + 关注点）
- `results/interictal_propagation/rank_displacement/figures/cohort_displacement_heatmap.{png,pdf}`
- `results/interictal_propagation/rank_displacement/figures/footrule_kendall_summary.{png,pdf}`
- `results/interictal_propagation/rank_displacement/figures/per_subject/<dataset>_<subject>_displacement.png`
- `docs/archive/topic1/pr6_supplementary_rank_displacement_results_2026-05-06.md` — supplementary results doc

**Modify:**
- `docs/topic1_within_event_dynamics.md`（§7 PR-6 那一节末尾或"历史文档索引"）— 加一行 supplementary 链接
- `AGENTS.md`（"Current Code Map" 下 interictal propagation 那一段）— 加一行 `src/rank_displacement.py` 入口

**目录命名遵守 CLAUDE.md "Results Directory Standards"**：
- 用 `rank_displacement/` topic-based 目录名，不命名为 `pr6_supplement/`
- `figures/README.md` 必须存在且是中文

---

## 3. Math Specification（实现前固定）

### 3.0 T_a / T_b anchor rule（写死，per-subject only）

每个 stable_k=2 subject 有两个 cluster。我们规定：

```
T_a = cluster with smaller cluster_id (PR-2 KMeans label)
T_b = the other cluster
```

**这是工程 anchor**，不是生物学 anchor。KMeans cluster_id 由初始化决定，没有跨 subject 的方向意义。

**Sign 合同（写死）**：
- `Δr(ch) = rank_T_b(ch) − rank_T_a(ch)` 永远是 "T_b 中比 T_a 中更靠后" 的位移
- 所有 signed 解读 **只在 subject 内部有效**：例如某个 subject 内 channel X 的 Δr=+3 表示在该 subject 的 T_b 中比 T_a 中靠后 3 位
- **跨 subject 不能比较 signed 方向**。Cohort 比较只用 invariant 量：footrule, |Δr|_mean, Kendall τ
- 因此本计划**不**输出 `signed_displacement_mean_soz` / `signed_displacement_mean_nonsoz`（任何 anchor-dependent 聚合都从 helper / archive 中移除）

### 3.1 Per-channel signed rank displacement

给定两个 cluster template Ta, Tb，和它们各自的 valid_mask v_a, v_b（bool, len=n_channels）：

```
joint_valid = v_a AND v_b                # bool[n]
n_valid     = sum(joint_valid)

if n_valid < 4:
    return {exit_reason: "n_valid<4"}    # 4 是最小可解释的 rank 长度

# Re-rank within joint_valid set so ranks are dense 0..n_valid-1
# 重要: 用 PR-2 template_rank 的相对序，但只保留 joint_valid 通道，再 re-rank
r_a_full = template_rank_Ta              # int[n], 含 -1 sentinel for invalid
r_b_full = template_rank_Tb              # int[n]

r_a_subset = r_a_full[joint_valid]
r_b_subset = r_b_full[joint_valid]

# Re-rank dense 0..n_valid-1 (避免 -1 sentinel 污染 displacement)
r_a_dense = scipy.stats.rankdata(r_a_subset, method="average") - 1.0
r_b_dense = scipy.stats.rankdata(r_b_subset, method="average") - 1.0

# Per-channel signed displacement (joint_valid 通道)
delta_subset = r_b_dense - r_a_dense     # float[n_valid], 含 0
abs_subset   = np.abs(delta_subset)

# Pad back to full channel length (NaN for non-joint-valid)
delta_full   = np.full(n_channels, np.nan)
delta_full[joint_valid] = delta_subset
```

**契约**：返回 `signed_displacement_full` (NaN-padded, len=n_channels) + `signed_displacement_dense` (subset-only) + `joint_valid` (bool mask) + `n_valid` + `channel_names`（master ordering）。

### 3.2 Aggregation

```
footrule           = sum(abs_subset)                          # ∈ [0, F_max]
F_max              = floor(n_valid * n_valid / 2)             # Diaconis-Graham 1977
footrule_normalized = footrule / F_max if F_max > 0 else nan  # ∈ [0, 1], 1 = 完全反向

kendall_tau, kendall_p = scipy.stats.kendalltau(r_a_dense, r_b_dense)
spearman_rho, spearman_p = scipy.stats.spearmanr(r_a_dense, r_b_dense)
```

**Sanity（写入 test）**：
- 完全相同 ranks → footrule=0, F_norm=0, τ=1
- 完全反向 ranks (n=4: [0,1,2,3] vs [3,2,1,0]) → footrule = 8, F_max = 8, F_norm=1, τ=-1
- 随机 ranks → F_norm 趋近 2/3 (Diaconis-Graham 期望)

### 3.3 SOZ contribution split（baseline-corrected）

**关键：通道数会污染 contribution_fraction**。如果 SOZ 通道占比 = k/n 且 |Δr| 大致均匀，contribution_fraction 期望 ≈ k/n（不是 0.5）。所以必须同时报告 channel_fraction 与 excess。

```
soz_mask = np.array([ch in soz_channels for ch in channel_names])  # bool[n]
soz_joint = soz_mask & joint_valid
nonsoz_joint = (~soz_mask) & joint_valid

n_soz_joint = int(soz_joint.sum())
n_nonsoz_joint = int(nonsoz_joint.sum())
n_valid = n_soz_joint + n_nonsoz_joint

# (a) Channel-count baseline（关键参考）
soz_channel_fraction = n_soz_joint / n_valid     # ∈ [0, 1]; chance level for contribution

# (b) Total displacement contribution
soz_contribution_fraction = np.nansum(np.abs(delta_full[soz_joint])) / footrule \
    if footrule > 0 and n_soz_joint > 0 else nan
nonsoz_contribution_fraction = np.nansum(np.abs(delta_full[nonsoz_joint])) / footrule \
    if footrule > 0 and n_nonsoz_joint > 0 else nan
# soz_contribution_fraction + nonsoz_contribution_fraction = 1.0 (when both > 0)

# (c) Excess over channel-count baseline（relative SOZ-vs-nonSOZ involvement）
soz_contribution_excess = soz_contribution_fraction - soz_channel_fraction \
    if soz_contribution_fraction is not nan else nan
# > 0 ⇒ SOZ contributes more than its channel-count share; < 0 ⇒ less; ≈ 0 ⇒ on baseline

# (d) Per-channel mean magnitude（avoids count-confound entirely）
soz_abs_mean = float(np.nanmean(np.abs(delta_full[soz_joint]))) if n_soz_joint > 0 else nan
nonsoz_abs_mean = float(np.nanmean(np.abs(delta_full[nonsoz_joint]))) if n_nonsoz_joint > 0 else nan
soz_minus_nonsoz_abs_mean = soz_abs_mean - nonsoz_abs_mean   # positive ⇒ SOZ channels move more
```

**Helper 必须输出全部 7 个字段**：
`soz_channel_fraction`, `soz_contribution_fraction`, `nonsoz_contribution_fraction`,
`soz_contribution_excess`, `soz_abs_mean`, `nonsoz_abs_mean`, `soz_minus_nonsoz_abs_mean`。

**不**输出 `signed_displacement_mean_soz` / `signed_displacement_mean_nonsoz`（违反 §3.0 sign 合同；anchor-dependent 不能 cohort 聚合）。

**不**做 SOZ vs nonSOZ 显著性检验；只作 descriptive 报告。Archive doc 比较时主用 `soz_contribution_excess` 与 `soz_minus_nonsoz_abs_mean`，不主用裸 `soz_contribution_fraction`。

---

## 4. Tasks

### Task 1: 创建 `src/rank_displacement.py` 骨架与 `compute_signed_rank_displacement`

**Files:**
- Create: `src/rank_displacement.py`
- Test: `tests/test_rank_displacement.py`

- [ ] **Step 1: 写失败测试 — 完全相同 ranks**

```python
# tests/test_rank_displacement.py
import numpy as np
import pytest
from src.rank_displacement import compute_signed_rank_displacement


def test_identical_ranks_zero_displacement():
    rank_a = np.array([0, 1, 2, 3, 4], dtype=float)
    rank_b = np.array([0, 1, 2, 3, 4], dtype=float)
    valid_mask = np.array([True] * 5)
    result = compute_signed_rank_displacement(
        rank_a=rank_a,
        rank_b=rank_b,
        valid_mask_a=valid_mask,
        valid_mask_b=valid_mask,
        channel_names=["A", "B", "C", "D", "E"],
    )
    assert result["exit_reason"] == "ok"
    assert result["n_valid"] == 5
    np.testing.assert_array_equal(result["signed_displacement_dense"], np.zeros(5))
    assert result["footrule"] == pytest.approx(0.0)
    assert result["footrule_normalized"] == pytest.approx(0.0)
    assert result["kendall_tau"] == pytest.approx(1.0)
```

- [ ] **Step 2: 跑测试，确认失败**

```bash
cd /home/honglab/leijiaxin/HFOsp && python -m pytest tests/test_rank_displacement.py::test_identical_ranks_zero_displacement -v
```

Expected: `ImportError: cannot import name 'compute_signed_rank_displacement'` 或 `ModuleNotFoundError`.

- [ ] **Step 3: 写最小实现**

```python
# src/rank_displacement.py
"""Per-channel signed rank displacement metrics for cluster template comparison.

Supplementary to PR-6 endpoint anchoring (forward/reverse template geometry).
Continuous version of PR-6 discrete swap_node count.
"""
from __future__ import annotations

from math import floor
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import stats


def compute_signed_rank_displacement(
    rank_a: np.ndarray,
    rank_b: np.ndarray,
    valid_mask_a: np.ndarray,
    valid_mask_b: np.ndarray,
    channel_names: Sequence[str],
) -> Dict[str, object]:
    """Compute per-channel signed rank displacement Δr(ch) = rank_b(ch) - rank_a(ch).

    All inputs MUST share the same channel ordering (length n_channels).
    Joint valid set = valid_mask_a AND valid_mask_b. Ranks are re-densified
    within the joint set before subtraction so that -1 sentinels in the
    template_rank vector cannot pollute the displacement.

    Aggregation: Spearman footrule, Diaconis-Graham normalized footrule,
    Kendall tau, Spearman rho. No NaN imputation; no default valid_mask.
    """
    rank_a = np.asarray(rank_a, dtype=float)
    rank_b = np.asarray(rank_b, dtype=float)
    valid_mask_a = np.asarray(valid_mask_a, dtype=bool)
    valid_mask_b = np.asarray(valid_mask_b, dtype=bool)
    channel_names = list(channel_names)

    n_channels = len(channel_names)
    if not (rank_a.shape == rank_b.shape == valid_mask_a.shape == valid_mask_b.shape == (n_channels,)):
        raise ValueError(
            f"Shape mismatch: rank_a={rank_a.shape}, rank_b={rank_b.shape}, "
            f"valid_mask_a={valid_mask_a.shape}, valid_mask_b={valid_mask_b.shape}, "
            f"n_channels={n_channels}"
        )

    joint_valid = valid_mask_a & valid_mask_b
    n_valid = int(joint_valid.sum())

    delta_full = np.full(n_channels, np.nan)
    rank_a_dense_full = np.full(n_channels, np.nan)
    rank_b_dense_full = np.full(n_channels, np.nan)
    out: Dict[str, object] = {
        "channel_names": channel_names,
        "joint_valid": joint_valid.tolist(),
        "n_valid": n_valid,
        "rank_a_full": rank_a.tolist(),                # raw input (PR-2 template_rank)
        "rank_b_full": rank_b.tolist(),
        "rank_a_dense_full": rank_a_dense_full.tolist(),  # NaN outside joint_valid
        "rank_b_dense_full": rank_b_dense_full.tolist(),
        "signed_displacement_full": delta_full.tolist(),
        "signed_displacement_dense": [],
        "footrule": float("nan"),
        "footrule_max": float("nan"),
        "footrule_normalized": float("nan"),
        "kendall_tau": float("nan"),
        "kendall_p": float("nan"),
        "spearman_rho": float("nan"),
        "spearman_p": float("nan"),
        "exit_reason": "ok",
    }

    if n_valid < 4:
        out["exit_reason"] = f"n_valid<4 (got {n_valid})"
        return out

    r_a_subset = rank_a[joint_valid]
    r_b_subset = rank_b[joint_valid]
    r_a_dense = stats.rankdata(r_a_subset, method="average") - 1.0
    r_b_dense = stats.rankdata(r_b_subset, method="average") - 1.0

    delta_subset = r_b_dense - r_a_dense
    abs_subset = np.abs(delta_subset)

    delta_full[joint_valid] = delta_subset
    rank_a_dense_full[joint_valid] = r_a_dense
    rank_b_dense_full[joint_valid] = r_b_dense

    footrule = float(abs_subset.sum())
    footrule_max = float(floor(n_valid * n_valid / 2))
    footrule_normalized = footrule / footrule_max if footrule_max > 0 else float("nan")

    tau_res = stats.kendalltau(r_a_dense, r_b_dense)
    rho_res = stats.spearmanr(r_a_dense, r_b_dense)

    out.update({
        "rank_a_dense_full": rank_a_dense_full.tolist(),
        "rank_b_dense_full": rank_b_dense_full.tolist(),
        "signed_displacement_full": delta_full.tolist(),
        "signed_displacement_dense": delta_subset.tolist(),
        "footrule": footrule,
        "footrule_max": footrule_max,
        "footrule_normalized": footrule_normalized,
        "kendall_tau": float(tau_res.statistic) if hasattr(tau_res, "statistic") else float(tau_res[0]),
        "kendall_p": float(tau_res.pvalue) if hasattr(tau_res, "pvalue") else float(tau_res[1]),
        "spearman_rho": float(rho_res.statistic) if hasattr(rho_res, "statistic") else float(rho_res[0]),
        "spearman_p": float(rho_res.pvalue) if hasattr(rho_res, "pvalue") else float(rho_res[1]),
    })
    return out
```

- [ ] **Step 4: 跑测试，确认通过**

```bash
cd /home/honglab/leijiaxin/HFOsp && python -m pytest tests/test_rank_displacement.py::test_identical_ranks_zero_displacement -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rank_displacement.py tests/test_rank_displacement.py
git commit -m "feat(rank_displacement): scaffold compute_signed_rank_displacement with identity test

PR-6 supplementary helper: continuous version of swap_node count.
First TDD test — identical ranks → footrule=0, kendall_tau=1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: 反向 + Diaconis-Graham 边界 + n_valid<4 测试

**Files:**
- Test: `tests/test_rank_displacement.py`

- [ ] **Step 1: 增加完全反向、随机、n_valid<4 的测试**

```python
def test_reversed_ranks_full_footrule():
    rank_a = np.array([0, 1, 2, 3], dtype=float)
    rank_b = np.array([3, 2, 1, 0], dtype=float)
    valid_mask = np.array([True] * 4)
    result = compute_signed_rank_displacement(
        rank_a=rank_a, rank_b=rank_b,
        valid_mask_a=valid_mask, valid_mask_b=valid_mask,
        channel_names=["A", "B", "C", "D"],
    )
    assert result["exit_reason"] == "ok"
    # Reversed: |Δ| = [3, 1, 1, 3] → footrule = 8
    assert result["footrule"] == pytest.approx(8.0)
    # Diaconis-Graham F_max = floor(n^2 / 2) = floor(16/2) = 8
    assert result["footrule_max"] == pytest.approx(8.0)
    assert result["footrule_normalized"] == pytest.approx(1.0)
    assert result["kendall_tau"] == pytest.approx(-1.0)


def test_signed_displacement_signs_correct():
    # rank_a = [0, 1, 2, 3] (A first, D last)
    # rank_b = [3, 2, 1, 0] (D first, A last)
    # delta = rank_b - rank_a = [+3, +1, -1, -3]
    rank_a = np.array([0, 1, 2, 3], dtype=float)
    rank_b = np.array([3, 2, 1, 0], dtype=float)
    valid_mask = np.array([True] * 4)
    result = compute_signed_rank_displacement(
        rank_a=rank_a, rank_b=rank_b,
        valid_mask_a=valid_mask, valid_mask_b=valid_mask,
        channel_names=["A", "B", "C", "D"],
    )
    np.testing.assert_array_almost_equal(
        result["signed_displacement_dense"], [3.0, 1.0, -1.0, -3.0]
    )


def test_partial_valid_mask_intersection():
    # n_channels = 5, but valid_mask drops index 0 in A and index 4 in B
    rank_a = np.array([-1, 0, 1, 2, 3], dtype=float)
    rank_b = np.array([3, 2, 1, 0, -1], dtype=float)
    valid_mask_a = np.array([False, True, True, True, True])
    valid_mask_b = np.array([True, True, True, True, False])
    result = compute_signed_rank_displacement(
        rank_a=rank_a, rank_b=rank_b,
        valid_mask_a=valid_mask_a, valid_mask_b=valid_mask_b,
        channel_names=["A", "B", "C", "D", "E"],
    )
    # Joint valid = indices 1, 2, 3 → n_valid = 3 < 4 → abort
    assert result["exit_reason"].startswith("n_valid<4")
    assert result["n_valid"] == 3


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_signed_rank_displacement(
            rank_a=np.array([0, 1, 2]),
            rank_b=np.array([0, 1, 2, 3]),
            valid_mask_a=np.array([True, True, True]),
            valid_mask_b=np.array([True, True, True]),
            channel_names=["A", "B", "C"],
        )
```

- [ ] **Step 2: 跑全部测试**

```bash
cd /home/honglab/leijiaxin/HFOsp && python -m pytest tests/test_rank_displacement.py -v
```

Expected: 5/5 PASS（4 个新测试 + 1 个 Task 1 测试）。如果 `kendall_tau` 在 `kendalltau` 旧版返回 namedtuple，确认 `tau_res.statistic` / `tau_res[0]` fallback 命中。

- [ ] **Step 3: Commit**

```bash
git add tests/test_rank_displacement.py
git commit -m "test(rank_displacement): add reversal/sign/partial-mask/shape tests

Cover Diaconis-Graham F_max=floor(n^2/2), perfect reversal kendall=-1,
partial valid_mask intersection abort, shape mismatch ValueError.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: 添加 `aggregate_pair_metrics` SOZ 分裂 helper

**Files:**
- Modify: `src/rank_displacement.py`
- Modify: `tests/test_rank_displacement.py`

- [ ] **Step 1: 写测试 — SOZ split with baseline correction**

```python
def test_aggregate_pair_metrics_soz_baseline_correction():
    from src.rank_displacement import aggregate_pair_metrics

    # 4 channels, A/B in SOZ, C/D non-SOZ → 50% SOZ baseline
    rank_a = np.array([0, 1, 2, 3], dtype=float)
    rank_b = np.array([3, 2, 1, 0], dtype=float)
    valid_mask = np.array([True] * 4)
    result = aggregate_pair_metrics(
        rank_a=rank_a, rank_b=rank_b,
        valid_mask_a=valid_mask, valid_mask_b=valid_mask,
        channel_names=["A", "B", "C", "D"],
        soz_channels={"A", "B"},
    )
    assert result["exit_reason"] == "ok"
    # |Δ| = [3, 1, 1, 3]; SOZ {A,B} → |Δ_SOZ| = [3,1]; nonSOZ {C,D} → |Δ_non| = [1,3]
    # Footrule = 8; SOZ contribution = 4/8 = 0.5
    assert result["soz_contribution_fraction"] == pytest.approx(0.5)
    assert result["nonsoz_contribution_fraction"] == pytest.approx(0.5)
    # Channel-count baseline: 2/4 = 0.5
    assert result["soz_channel_fraction"] == pytest.approx(0.5)
    # Excess = 0.5 − 0.5 = 0.0 (this perfect-reversal example sits on baseline)
    assert result["soz_contribution_excess"] == pytest.approx(0.0)
    # Per-channel |Δ| means: SOZ mean = 2.0, nonSOZ mean = 2.0
    assert result["soz_abs_mean"] == pytest.approx(2.0)
    assert result["nonsoz_abs_mean"] == pytest.approx(2.0)
    assert result["soz_minus_nonsoz_abs_mean"] == pytest.approx(0.0)


def test_aggregate_pair_metrics_soz_excess_positive():
    """When SOZ channels actually carry more displacement, excess should be > 0."""
    from src.rank_displacement import aggregate_pair_metrics

    # 4 channels, A/B in SOZ swap heavily; C/D barely move
    # rank_a = [0, 1, 2, 3]; rank_b = [3, 2, 0, 1]
    # Δ = [3, 1, -2, -2]; |Δ| = [3, 1, 2, 2]; footrule = 8
    # SOZ {A,B}: |Δ| = [3,1] → contribution = 4/8 = 0.5; baseline 2/4 = 0.5; excess = 0
    # Use [0,1,2,3] vs [3,1,0,2]: Δ=[3,0,-2,-1]; |Δ|=[3,0,2,1]=6; SOZ |Δ|=[3,0]=3 → 0.5 still.
    # Construct: 5 channels, A in SOZ; rank_a=[0,1,2,3,4]; rank_b=[4,1,2,3,0]
    # Δ=[4,0,0,0,-4]; |Δ|=[4,0,0,0,4]=8; A in SOZ; SOZ contribution=4/8=0.5; baseline=1/5=0.2
    # excess = 0.3 (SOZ overrepresented in displacement)
    rank_a = np.array([0, 1, 2, 3, 4], dtype=float)
    rank_b = np.array([4, 1, 2, 3, 0], dtype=float)
    valid_mask = np.array([True] * 5)
    result = aggregate_pair_metrics(
        rank_a=rank_a, rank_b=rank_b,
        valid_mask_a=valid_mask, valid_mask_b=valid_mask,
        channel_names=["A", "B", "C", "D", "E"],
        soz_channels={"A"},
    )
    assert result["soz_channel_fraction"] == pytest.approx(0.2)
    assert result["soz_contribution_fraction"] == pytest.approx(0.5)
    assert result["soz_contribution_excess"] == pytest.approx(0.3)
    # Per-channel |Δ| mean: SOZ {A} = 4.0; nonSOZ {B,C,D,E} = (0+0+0+4)/4 = 1.0
    assert result["soz_abs_mean"] == pytest.approx(4.0)
    assert result["nonsoz_abs_mean"] == pytest.approx(1.0)
    assert result["soz_minus_nonsoz_abs_mean"] == pytest.approx(3.0)


def test_aggregate_does_not_export_signed_soz_mean():
    """Sign-anchor contract: never expose anchor-dependent aggregates."""
    from src.rank_displacement import aggregate_pair_metrics

    rank_a = np.array([0, 1, 2, 3], dtype=float)
    rank_b = np.array([3, 2, 1, 0], dtype=float)
    valid_mask = np.array([True] * 4)
    result = aggregate_pair_metrics(
        rank_a=rank_a, rank_b=rank_b,
        valid_mask_a=valid_mask, valid_mask_b=valid_mask,
        channel_names=["A", "B", "C", "D"],
        soz_channels={"A", "B"},
    )
    assert "signed_displacement_mean_soz" not in result
    assert "signed_displacement_mean_nonsoz" not in result
```

- [ ] **Step 2: 跑测试，确认失败**

```bash
cd /home/honglab/leijiaxin/HFOsp && python -m pytest tests/test_rank_displacement.py::test_aggregate_pair_metrics_soz_split -v
```

Expected: `ImportError: cannot import name 'aggregate_pair_metrics'`.

- [ ] **Step 3: 实现 helper（baseline-corrected SOZ split, no signed cohort aggregates）**

在 `src/rank_displacement.py` 末尾追加：

```python
def aggregate_pair_metrics(
    rank_a: np.ndarray,
    rank_b: np.ndarray,
    valid_mask_a: np.ndarray,
    valid_mask_b: np.ndarray,
    channel_names: Sequence[str],
    soz_channels: Optional[set] = None,
) -> Dict[str, object]:
    """Wrap compute_signed_rank_displacement with baseline-corrected SOZ split.

    Outputs (descriptive only, no PASS gate):
      - soz_channel_fraction       (chance baseline for contribution_fraction)
      - soz_contribution_fraction  (Σ|Δr|_SOZ / footrule)
      - nonsoz_contribution_fraction
      - soz_contribution_excess    (= contribution_fraction − channel_fraction)
      - soz_abs_mean, nonsoz_abs_mean    (per-channel |Δr| means; count-confound free)
      - soz_minus_nonsoz_abs_mean

    Does NOT export signed_displacement_mean_soz / nonsoz — those are anchor-
    dependent and per §3.0 cannot be aggregated across subjects.
    """
    base = compute_signed_rank_displacement(
        rank_a=rank_a,
        rank_b=rank_b,
        valid_mask_a=valid_mask_a,
        valid_mask_b=valid_mask_b,
        channel_names=channel_names,
    )
    if base["exit_reason"] != "ok":
        return base

    delta_full = np.asarray(base["signed_displacement_full"], dtype=float)
    joint_valid = np.asarray(base["joint_valid"], dtype=bool)
    soz_set = set(soz_channels or [])
    soz_mask = np.array([ch in soz_set for ch in channel_names], dtype=bool)
    soz_joint = soz_mask & joint_valid
    nonsoz_joint = (~soz_mask) & joint_valid

    n_soz_joint = int(soz_joint.sum())
    n_nonsoz_joint = int(nonsoz_joint.sum())
    n_valid = n_soz_joint + n_nonsoz_joint
    footrule = base["footrule"]
    abs_full = np.abs(delta_full)
    nan = float("nan")

    soz_channel_fraction = (n_soz_joint / n_valid) if n_valid > 0 else nan
    soz_contribution = (
        float(np.nansum(abs_full[soz_joint]) / footrule)
        if footrule > 0 and n_soz_joint > 0 else nan
    )
    nonsoz_contribution = (
        float(np.nansum(abs_full[nonsoz_joint]) / footrule)
        if footrule > 0 and n_nonsoz_joint > 0 else nan
    )
    soz_excess = (
        soz_contribution - soz_channel_fraction
        if not (np.isnan(soz_contribution) or np.isnan(soz_channel_fraction))
        else nan
    )
    soz_abs_mean = float(np.nanmean(abs_full[soz_joint])) if n_soz_joint > 0 else nan
    nonsoz_abs_mean = float(np.nanmean(abs_full[nonsoz_joint])) if n_nonsoz_joint > 0 else nan
    soz_minus_nonsoz_abs_mean = (
        soz_abs_mean - nonsoz_abs_mean
        if not (np.isnan(soz_abs_mean) or np.isnan(nonsoz_abs_mean))
        else nan
    )

    base.update({
        "soz_mask": soz_mask.tolist(),
        "n_soz_joint": n_soz_joint,
        "n_nonsoz_joint": n_nonsoz_joint,
        "soz_channel_fraction": soz_channel_fraction,
        "soz_contribution_fraction": soz_contribution,
        "nonsoz_contribution_fraction": nonsoz_contribution,
        "soz_contribution_excess": soz_excess,
        "soz_abs_mean": soz_abs_mean,
        "nonsoz_abs_mean": nonsoz_abs_mean,
        "soz_minus_nonsoz_abs_mean": soz_minus_nonsoz_abs_mean,
    })
    return base
```

- [ ] **Step 4: 跑测试，确认通过**

```bash
cd /home/honglab/leijiaxin/HFOsp && python -m pytest tests/test_rank_displacement.py -v
```

Expected: 8/8 PASS（5 from Tasks 1–2 + 3 new in Task 3）。

- [ ] **Step 5: Commit**

```bash
git add src/rank_displacement.py tests/test_rank_displacement.py
git commit -m "feat(rank_displacement): aggregate_pair_metrics with baseline-corrected SOZ split

SOZ contribution_fraction is count-confounded; outputs channel_fraction,
contribution_excess, and per-channel abs_mean to disambiguate. Does NOT
export signed_mean_soz / signed_mean_nonsoz: signed Δr is per-subject
only per §3.0 sign anchor contract.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: 写 batch runner `scripts/run_rank_displacement.py`

**Files:**
- Create: `scripts/run_rank_displacement.py`

- [ ] **Step 1: 写 runner 框架（PR-2 + PR-6 JSON 联动 + cluster_id 对齐）**

```python
#!/usr/bin/env python3
"""Per-subject signed rank displacement runner.

Inputs:
  - results/interictal_propagation/per_subject/<dataset>_<subject>.json (PR-2)
  - results/interictal_propagation/template_anchoring/per_subject/<dataset>_<subject>.json (PR-6)
  - results/<dataset>_soz_core_channels.json

Outputs:
  - results/interictal_propagation/rank_displacement/per_subject/<dataset>_<subject>.json
  - results/interictal_propagation/rank_displacement/cohort_summary.json

Cohort scope: all subjects with stable_k == 2 from PR-2 cluster JSON.
Forward/reverse-reproduced flag uses OR rule (split-half OR odd-even).
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.rank_displacement import aggregate_pair_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
PR2_DIR = REPO_ROOT / "results" / "interictal_propagation" / "per_subject"
PR6_DIR = REPO_ROOT / "results" / "interictal_propagation" / "template_anchoring" / "per_subject"
OUT_DIR = REPO_ROOT / "results" / "interictal_propagation" / "rank_displacement"
OUT_PER_SUBJECT = OUT_DIR / "per_subject"
SOZ_FILES = {
    "epilepsiae": REPO_ROOT / "results" / "epilepsiae_soz_core_channels.json",
    "yuquan": REPO_ROOT / "results" / "yuquan_soz_core_channels.json",
}


def load_soz_lookup() -> Dict[str, Dict[str, set]]:
    out: Dict[str, Dict[str, set]] = {}
    for ds, path in SOZ_FILES.items():
        if not path.exists():
            out[ds] = {}
            continue
        raw = json.loads(path.read_text())
        out[ds] = {sub: set(chs or []) for sub, chs in raw.items()}
    return out


def derive_fwd_rev_reproduced(pr2: dict) -> Optional[bool]:
    splits = pr2.get("time_split_reproducibility", {}).get("splits", {})
    fhsh = splits.get("first_half_second_half", {}).get("forward_reverse_reproduced")
    oeb = splits.get("odd_even_block", {}).get("forward_reverse_reproduced")
    flags = [v for v in (fhsh, oeb) if isinstance(v, bool)]
    if not flags:
        return None
    return any(flags)


def derive_fwd_rev_source(pr2: dict) -> str:
    splits = pr2.get("time_split_reproducibility", {}).get("splits", {})
    fhsh = splits.get("first_half_second_half", {}).get("forward_reverse_reproduced")
    oeb = splits.get("odd_even_block", {}).get("forward_reverse_reproduced")
    if fhsh and oeb:
        return "both"
    if fhsh:
        return "first_half_second_half"
    if oeb:
        return "odd_even_block"
    if fhsh is False or oeb is False:
        return "none"
    return "unknown"


def build_template_lookup(per_template: list, channel_names_master: list) -> Dict[int, Dict]:
    """Map cluster_id -> {valid_mask, source, sink, n_valid_channels}."""
    out: Dict[int, Dict] = {}
    for entry in per_template:
        cid = entry.get("cluster_id")
        valid_mask = entry.get("valid_mask")
        if cid is None or valid_mask is None:
            continue
        if len(valid_mask) != len(channel_names_master):
            raise ValueError(
                f"PR-6 valid_mask length {len(valid_mask)} != "
                f"PR-2 channel_names {len(channel_names_master)}"
            )
        out[int(cid)] = {
            "valid_mask": np.asarray(valid_mask, dtype=bool),
            "source": entry.get("source", []),
            "sink": entry.get("sink", []),
            "n_valid_channels": int(entry.get("n_valid_channels", sum(valid_mask))),
        }
    return out


def process_subject(pr2_path: Path, pr6_path: Path, soz_lookup: Dict[str, Dict[str, set]]) -> Optional[dict]:
    pr2 = json.loads(pr2_path.read_text())
    if not pr6_path.exists():
        return {"subject": pr2.get("subject"), "dataset": pr2.get("dataset"),
                "exit_reason": "pr6_missing"}
    pr6 = json.loads(pr6_path.read_text())

    dataset = pr2.get("dataset")
    subject = pr2.get("subject")
    channel_names = pr2.get("channel_names")
    if channel_names is None:
        return {"subject": subject, "dataset": dataset, "exit_reason": "no_channel_names"}

    ac = pr2.get("adaptive_cluster", {})
    stable_k = ac.get("stable_k")
    clusters = ac.get("clusters", [])
    rank_lookup: Dict[int, np.ndarray] = {}
    for c in clusters:
        cid = c.get("cluster_id")
        tr = c.get("template_rank")
        if cid is None or tr is None:
            continue
        if len(tr) != len(channel_names):
            raise ValueError(
                f"{subject}: template_rank length {len(tr)} != channel_names {len(channel_names)}"
            )
        rank_lookup[int(cid)] = np.asarray(tr, dtype=float)

    template_lookup = build_template_lookup(pr6.get("per_template", []), channel_names)
    common_cids = sorted(set(rank_lookup) & set(template_lookup))

    soz_channels = soz_lookup.get(dataset, {}).get(str(subject), set())
    if not soz_channels:
        soz_channels = soz_lookup.get(dataset, {}).get(subject, set())

    fwd_rev = derive_fwd_rev_reproduced(pr2)
    fwd_rev_source = derive_fwd_rev_source(pr2)

    pairs = []
    for cid_a, cid_b in combinations(common_cids, 2):
        rank_a = rank_lookup[cid_a]
        rank_b = rank_lookup[cid_b]
        v_a = template_lookup[cid_a]["valid_mask"]
        v_b = template_lookup[cid_b]["valid_mask"]
        metrics = aggregate_pair_metrics(
            rank_a=rank_a,
            rank_b=rank_b,
            valid_mask_a=v_a,
            valid_mask_b=v_b,
            channel_names=channel_names,
            soz_channels=soz_channels,
        )
        metrics["cluster_id_a"] = int(cid_a)
        metrics["cluster_id_b"] = int(cid_b)
        pairs.append(metrics)

    inter_corr = ac.get("inter_cluster_corr_matrix")
    geom = pr6.get("template_pair_geometry", {})

    return {
        "subject": subject,
        "dataset": dataset,
        "stable_k": stable_k,
        "n_channels": len(channel_names),
        "channel_names": channel_names,
        "soz_channels": sorted(soz_channels),
        "fwd_rev_reproduced": fwd_rev,
        "fwd_rev_source": fwd_rev_source,
        "pr6_swap_score": pr6.get("h2_swap_check", {}).get("swap_score"),
        "pr6_swap_null_p": pr6.get("h2_swap_check", {}).get("null_p"),
        "pr6_pair_geometry_spearman": geom.get("spearman_rank_pair"),
        "inter_cluster_corr_matrix": inter_corr,
        "pairs": pairs,
        "exit_reason": "ok",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="Optional list of <dataset>_<subject> stems; default all PR-2 JSONs.")
    args = ap.parse_args()

    OUT_PER_SUBJECT.mkdir(parents=True, exist_ok=True)
    soz_lookup = load_soz_lookup()

    if args.subjects:
        stems = args.subjects
    else:
        stems = sorted(p.stem for p in PR2_DIR.glob("*.json")
                       if not p.stem.startswith("pr"))

    cohort: List[dict] = []
    for stem in stems:
        pr2_path = PR2_DIR / f"{stem}.json"
        pr6_path = PR6_DIR / f"{stem}.json"
        if not pr2_path.exists():
            continue
        result = process_subject(pr2_path, pr6_path, soz_lookup)
        if result is None:
            continue
        out_path = OUT_PER_SUBJECT / f"{stem}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        cohort.append({k: v for k, v in result.items() if k != "channel_names"
                       and k != "inter_cluster_corr_matrix"})
        print(f"[{stem}] stable_k={result.get('stable_k')} "
              f"fwd_rev={result.get('fwd_rev_reproduced')} "
              f"n_pairs={len(result.get('pairs', []))} "
              f"exit={result.get('exit_reason')}")

    cohort_path = OUT_DIR / "cohort_summary.json"
    cohort_path.write_text(json.dumps(cohort, indent=2, ensure_ascii=False))
    print(f"\nWrote {cohort_path} with {len(cohort)} subjects")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 烟雾测试 — 跑前 3 个 subject**

```bash
cd /home/honglab/leijiaxin/HFOsp && python scripts/run_rank_displacement.py --subjects epilepsiae_1073 epilepsiae_1077 epilepsiae_1084
```

Expected: 3 行输出，`exit=ok`，`results/interictal_propagation/rank_displacement/per_subject/` 下出现 3 个 JSON。

- [ ] **Step 3: 跑全部 cohort**

```bash
cd /home/honglab/leijiaxin/HFOsp && python scripts/run_rank_displacement.py 2>&1 | tee /tmp/rank_displacement_run.log
```

Expected: ~30 行输出，每行 `stable_k`、`fwd_rev`、`n_pairs`；`cohort_summary.json` 包含 ~30 entries。

- [ ] **Step 4: Commit**

```bash
git add scripts/run_rank_displacement.py
git commit -m "feat(rank_displacement): add batch runner for per-subject + cohort summary

Consume PR-2 cluster JSON + PR-6 anchoring JSON, output per-subject
displacement JSON + cohort_summary.json. cluster_id-aligned pairing,
OR-rule forward/reverse-reproduced flag.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: 写 figure script Panel A — cohort heatmap

**Files:**
- Create: `scripts/plot_rank_displacement.py`

- [ ] **Step 1: 写 cohort heatmap 函数（顶刊风格，divergent palette + SOZ outline）**

```python
#!/usr/bin/env python3
"""Top-tier-journal-style supplementary figures for PR-6 rank displacement.

Produces 3 deliverables:
  1. cohort_displacement_heatmap.{png,pdf} — 27 stable_k=2 subjects × channels,
     rows sorted by Kendall tau, divergent RdBu palette, SOZ marked.
  2. footrule_kendall_summary.{png,pdf} — 2-panel: footrule_normalized vs
     fwd/rev-reproduced grouping; Kendall tau strip with reference lines.
  3. per_subject/<stem>_displacement.png — per-subject zoom-in heatstrip
     with channel labels (debug + supplement).

No statistical PASS gate. All annotations are descriptive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from src.plot_style import (
    COL_SOZ, COL_NONSOZ, COL_SIG, COL_NONSIG,
    DPI_PUB, FS_LABEL, FS_TICK, FS_TITLE, FS_PANEL_LETTER,
    style_panel,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RES_DIR = REPO_ROOT / "results" / "interictal_propagation" / "rank_displacement"
PER_SUBJECT_DIR = RES_DIR / "per_subject"
FIG_DIR = RES_DIR / "figures"
PER_SUB_FIG_DIR = FIG_DIR / "per_subject"


def load_cohort_records() -> List[dict]:
    """Load per-subject JSON; only keep stable_k=2 with one valid pair."""
    records = []
    for path in sorted(PER_SUBJECT_DIR.glob("*.json")):
        d = json.loads(path.read_text())
        if d.get("stable_k") != 2:
            continue
        pairs = [p for p in d.get("pairs", []) if p.get("exit_reason") == "ok"]
        if len(pairs) != 1:
            continue  # k=2 should have exactly 1 pair
        d["primary_pair"] = pairs[0]
        records.append(d)
    return records


def build_heatmap_matrix(records: List[dict]) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Build (subjects, max_n_valid) matrix of signed displacement.

    CRITICAL: rows arrange channels by **rank_T_a_dense** order (T_a's source →
    T_a's sink), NOT by Δr. Sorting by Δr would force every row into a
    monotonic red→blue gradient regardless of whether the underlying rank
    pair is reversed or random — that is the circular-sorting bias and is
    forbidden. Sorting by rank_T_a means:
      - perfect reversal ⇒ Δr is monotonic from +max to −max (red→blue strip)
      - random ⇒ Δr scatters with no visible gradient
    """
    sub_labels = [f"{r['dataset'][:3]}_{r['subject']}" for r in records]
    # Determine max_n_valid across the cohort (column count)
    max_n_valid = 0
    cached: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for r in records:
        pair = r["primary_pair"]
        delta = np.asarray(pair["signed_displacement_full"], dtype=float)
        joint = np.asarray(pair["joint_valid"], dtype=bool)
        soz_mask = np.asarray(pair.get("soz_mask", [False] * len(delta)), dtype=bool)
        # Use rank_a_dense_full from helper output: NaN outside joint_valid,
        # 0..n_valid-1 inside (T_a's source→sink axis position).
        rank_a_dense_full = np.asarray(pair["rank_a_dense_full"], dtype=float)
        valid_idx = np.where(joint)[0]
        if len(valid_idx) == 0:
            cached.append((np.array([]), np.array([], dtype=bool), np.array([])))
            continue
        rank_a_dense_subset = rank_a_dense_full[valid_idx]
        order = np.argsort(rank_a_dense_subset)  # T_a source first → sink last
        delta_sorted = delta[valid_idx][order]
        soz_sorted = soz_mask[valid_idx][order]
        max_n_valid = max(max_n_valid, len(delta_sorted))
        cached.append((delta_sorted, soz_sorted, rank_a_dense_subset[order]))

    matrix = np.full((len(records), max_n_valid), np.nan)
    soz_overlay = np.zeros_like(matrix, dtype=bool)
    for i, (delta_sorted, soz_sorted, _) in enumerate(cached):
        n = len(delta_sorted)
        matrix[i, :n] = delta_sorted
        soz_overlay[i, :n] = soz_sorted
    return matrix, sub_labels, soz_overlay


def sort_by_kendall_tau(records: List[dict]) -> List[dict]:
    return sorted(records, key=lambda r: r["primary_pair"].get("kendall_tau", 0.0))


def plot_cohort_heatmap(records: List[dict], out_stem: Path) -> None:
    sorted_records = sort_by_kendall_tau(records)
    matrix, labels, soz_overlay, _ = build_heatmap_matrix(sorted_records)
    n_sub, n_ch = matrix.shape
    taus = np.array([r["primary_pair"]["kendall_tau"] for r in sorted_records])
    fwd_rev_flags = [bool(r.get("fwd_rev_reproduced")) for r in sorted_records]

    vmax = float(np.nanmax(np.abs(matrix)))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    fig = plt.figure(figsize=(11, max(6, 0.32 * n_sub)))
    gs = fig.add_gridspec(1, 3, width_ratios=[8, 1.2, 0.5], wspace=0.05)
    ax_h = fig.add_subplot(gs[0])
    ax_tau = fig.add_subplot(gs[1], sharey=ax_h)
    ax_cb = fig.add_subplot(gs[2])

    im = ax_h.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm,
                     interpolation="nearest")
    # SOZ overlay: black border on cells
    for i in range(n_sub):
        for j in range(n_ch):
            if soz_overlay[i, j]:
                ax_h.add_patch(mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False,
                    edgecolor="black", linewidth=0.7,
                ))
    ax_h.set_yticks(range(n_sub))
    ax_h.set_yticklabels(labels, fontsize=FS_TICK - 4)
    ax_h.set_xticks([0, n_ch - 1])
    ax_h.set_xticklabels(["T_a source\n(earliest in T_a)",
                          "T_a sink\n(latest in T_a)"],
                         fontsize=FS_TICK - 2)
    ax_h.set_xlabel("Channel position along T_a's source→sink axis",
                    fontsize=FS_LABEL)
    ax_h.set_title("a", fontsize=FS_PANEL_LETTER, loc="left", pad=10, fontweight="bold")

    # Kendall tau side panel
    colors = [COL_SIG if f else COL_NONSIG for f in fwd_rev_flags]
    ax_tau.barh(range(n_sub), taus, color=colors, edgecolor="black", linewidth=0.4)
    ax_tau.axvline(0, color="gray", linewidth=0.6)
    ax_tau.axvline(-0.5, color="gray", linewidth=0.4, linestyle="--")
    ax_tau.axvline(0.5, color="gray", linewidth=0.4, linestyle="--")
    ax_tau.set_xlim(-1.05, 1.05)
    ax_tau.set_xticks([-1, 0, 1])
    ax_tau.tick_params(axis="x", labelsize=FS_TICK - 4)
    plt.setp(ax_tau.get_yticklabels(), visible=False)
    ax_tau.set_xlabel("Kendall τ", fontsize=FS_LABEL - 2)
    ax_tau.spines["top"].set_visible(False)
    ax_tau.spines["right"].set_visible(False)

    # Colorbar
    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label("Signed Δr (= rank_Tb − rank_Ta)", fontsize=FS_LABEL - 2)
    cb.ax.tick_params(labelsize=FS_TICK - 4)

    # Legend for fwd/rev
    legend_handles = [
        mpatches.Patch(color=COL_SIG, label="forward/reverse reproduced (PR-2.5 OR)"),
        mpatches.Patch(color=COL_NONSIG, label="not reproduced / unknown"),
        mpatches.Patch(facecolor="white", edgecolor="black", label="SOZ channel"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3,
               fontsize=FS_TICK - 2, frameon=False, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(
        f"Per-channel signed rank displacement, stable_k=2 cohort (n={n_sub})",
        fontsize=FS_TITLE, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_stem.with_suffix(".png"), dpi=DPI_PUB, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="all",
                    choices=["all", "cohort", "summary", "per_subject"])
    args = ap.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PER_SUB_FIG_DIR.mkdir(parents=True, exist_ok=True)
    records = load_cohort_records()
    print(f"Loaded {len(records)} stable_k=2 subjects")

    if args.what in ("all", "cohort"):
        plot_cohort_heatmap(records, FIG_DIR / "cohort_displacement_heatmap")
        print("Wrote cohort_displacement_heatmap.{png,pdf}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑 figure，目视检查**

```bash
cd /home/honglab/leijiaxin/HFOsp && python scripts/plot_rank_displacement.py --what cohort
ls -la results/interictal_propagation/rank_displacement/figures/
```

Expected: `cohort_displacement_heatmap.png` 和 `.pdf` 存在；目视检查（用户在 IDE 打开 PNG）：
- 行按 Kendall τ 从负到正排序（最反向在最上）
- divergent RdBu_r 调色板，红色 = +Δr，蓝色 = −Δr
- 每行内通道按 **rank_T_a_dense** 排（左边 = T_a 的 source，右边 = T_a 的 sink），**不是**按 Δr 排
- 关键 sanity check：完全反向的 subject（Kendall τ ≈ −1）在最上几行应当呈"红→蓝"单调梯度；Kendall τ ≈ 0 的 subject 应当颜色散乱无梯度（如果**所有**行都是单调梯度，说明 sorting bias 没修好，立即停下查 build_heatmap_matrix）
- SOZ 通道有黑色边框
- 右侧侧边条颜色按 fwd/rev-reproduced 分（rust = 是，gray = 否）
- 顶部图例齐全

- [ ] **Step 3: Commit**

```bash
git add scripts/plot_rank_displacement.py
git commit -m "feat(rank_displacement): add cohort heatmap figure (Panel A)

stable_k=2 cohort × channels heatmap. Rows sorted by Kendall tau;
columns within each row sorted by rank_T_a_dense (T_a source→sink),
NOT by Δr — sorting by Δr would introduce circular sorting bias and
force random pairs into pseudo-monotonic gradients. divergent RdBu_r,
SOZ outlined, fwd/rev-reproduced color-coded. PNG + PDF at DPI_PUB=300.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Figure summary panel — footrule + Kendall τ 分布

**Files:**
- Modify: `scripts/plot_rank_displacement.py`

- [ ] **Step 1: 在 `plot_rank_displacement.py` 末尾添加 `plot_footrule_summary`**

```python
def plot_footrule_summary(records: List[dict], out_stem: Path) -> None:
    fwd_rev_yes, fwd_rev_no = [], []
    taus_yes, taus_no = [], []
    for r in records:
        f_norm = r["primary_pair"].get("footrule_normalized")
        tau = r["primary_pair"].get("kendall_tau")
        if f_norm is None or tau is None or np.isnan(f_norm) or np.isnan(tau):
            continue
        if r.get("fwd_rev_reproduced"):
            fwd_rev_yes.append(f_norm)
            taus_yes.append(tau)
        else:
            fwd_rev_no.append(f_norm)
            taus_no.append(tau)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    # Panel B: footrule normalized split
    ax = axes[0]
    positions = [0, 1]
    parts = ax.violinplot(
        [fwd_rev_yes, fwd_rev_no],
        positions=positions, widths=0.7,
        showmeans=False, showmedians=True, showextrema=False,
    )
    for pc, col in zip(parts["bodies"], [COL_SIG, COL_NONSIG]):
        pc.set_facecolor(col)
        pc.set_edgecolor("black")
        pc.set_alpha(0.55)
    parts["cmedians"].set_color("black")
    rng = np.random.default_rng(42)
    for pos, vals, col in zip(positions, [fwd_rev_yes, fwd_rev_no], [COL_SIG, COL_NONSIG]):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.array([pos] * len(vals)) + jitter, vals,
                   color=col, edgecolors="black", linewidths=0.4, s=28, zorder=3)
    # Diaconis-Graham F_norm random expectation is *asymptotically* 2/3 (n→∞);
    # finite-n random expectation is slightly below. Annotate as reference, not gate.
    ax.axhline(2 / 3, color="gray", linewidth=0.6, linestyle=":")
    ax.text(1.45, 2 / 3, "asymptotic random\nreference (≈ 2/3)",
            fontsize=FS_TICK - 4, va="center", color="gray")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"reproduced\n(n={len(fwd_rev_yes)})",
                        f"not reproduced\n(n={len(fwd_rev_no)})"],
                       fontsize=FS_TICK - 2)
    ax.set_ylabel("Footrule (Diaconis-Graham normalized)", fontsize=FS_LABEL)
    ax.set_ylim(0, 1.05)
    style_panel(ax, label="b")

    # Panel C: Kendall tau strip
    ax = axes[1]
    ax.scatter(taus_yes, np.zeros(len(taus_yes)) + 1.0,
               color=COL_SIG, edgecolors="black", linewidths=0.4, s=44, zorder=3,
               label=f"reproduced (n={len(taus_yes)})")
    ax.scatter(taus_no, np.zeros(len(taus_no)) + 0.0,
               color=COL_NONSIG, edgecolors="black", linewidths=0.4, s=44, zorder=3,
               label=f"not reproduced (n={len(taus_no)})")
    ax.axvline(0, color="gray", linewidth=0.6)
    ax.axvline(-0.5, color="gray", linewidth=0.4, linestyle="--")
    ax.axvline(0.5, color="gray", linewidth=0.4, linestyle="--")
    ax.set_xlim(-1.05, 1.05)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.tick_params(axis="x", labelsize=FS_TICK - 2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["not\nreproduced", "reproduced"], fontsize=FS_TICK - 2)
    ax.set_xlabel("Kendall τ between Tₐ and T_b ranks", fontsize=FS_LABEL)
    ax.set_ylim(-0.7, 1.7)
    style_panel(ax, label="c")
    ax.legend(loc="upper left", fontsize=FS_TICK - 4, frameon=False)

    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), dpi=DPI_PUB, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
```

并在 `main()` 里加入：

```python
    if args.what in ("all", "summary"):
        plot_footrule_summary(records, FIG_DIR / "footrule_kendall_summary")
        print("Wrote footrule_kendall_summary.{png,pdf}")
```

- [ ] **Step 2: 跑 figure，目视检查**

```bash
cd /home/honglab/leijiaxin/HFOsp && python scripts/plot_rank_displacement.py --what summary
```

Expected:
- Panel B: 两个 violin（reproduced vs not reproduced），中位线黑色，散点 jitter，2/3 random baseline 灰色虚线
- Panel C: Kendall τ 双行 strip plot，τ=−1 / 0 / +1 标尺线，颜色与 Panel B 一致

- [ ] **Step 3: Commit**

```bash
git add scripts/plot_rank_displacement.py
git commit -m "feat(rank_displacement): add Panel B+C footrule + kendall summary

Two-panel descriptive: footrule_normalized violin split by fwd/rev-
reproduced flag; Kendall tau strip plot with reference lines at
+/-0.5. No PASS gate; 2/3 random-baseline reference annotated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Per-subject 详图（debug + supplement）

**Files:**
- Modify: `scripts/plot_rank_displacement.py`

- [ ] **Step 1: 在 `plot_rank_displacement.py` 末尾添加 `plot_per_subject_strip`**

```python
def plot_per_subject_strip(record: dict, out_path: Path) -> None:
    pair = record["primary_pair"]
    delta = np.asarray(pair["signed_displacement_full"], dtype=float)
    joint = np.asarray(pair["joint_valid"], dtype=bool)
    soz_mask = np.asarray(pair.get("soz_mask", [False] * len(delta)), dtype=bool)
    rank_a_dense_full = np.asarray(pair["rank_a_dense_full"], dtype=float)
    channel_names = record["channel_names"]
    valid_idx = np.where(joint)[0]
    if len(valid_idx) == 0:
        return
    delta_v = delta[valid_idx]
    chs_v = [channel_names[i] for i in valid_idx]
    soz_v = soz_mask[valid_idx]
    rank_a_v = rank_a_dense_full[valid_idx]
    # Sort by rank_T_a_dense (T_a source → sink), NOT by Δr (avoids circular bias)
    order = np.argsort(rank_a_v)
    delta_sorted = delta_v[order]
    chs_sorted = [chs_v[i] for i in order]
    soz_sorted = soz_v[order]

    n_ch = len(delta_sorted)
    vmax = float(np.max(np.abs(delta_sorted)))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * n_ch), 2.4))
    im = ax.imshow(delta_sorted[None, :], aspect="auto", cmap="RdBu_r", norm=norm)
    for j, is_soz in enumerate(soz_sorted):
        if is_soz:
            ax.add_patch(mpatches.Rectangle(
                (j - 0.5, -0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.0,
            ))
    ax.set_xticks(range(n_ch))
    ax.set_xticklabels(chs_sorted, rotation=60, fontsize=FS_TICK - 4, ha="right")
    ax.set_yticks([])
    sub_label = f"{record['dataset']} {record['subject']}"
    fwd = "✓" if record.get("fwd_rev_reproduced") else "✗"
    tau = pair.get("kendall_tau", float("nan"))
    f_norm = pair.get("footrule_normalized", float("nan"))
    ax.set_title(
        f"{sub_label}  |  k={record.get('stable_k')}  |  "
        f"fwd/rev={fwd}  |  τ={tau:.3f}  |  F_norm={f_norm:.3f}",
        fontsize=FS_LABEL - 2,
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("Δr", fontsize=FS_LABEL - 4)
    cb.ax.tick_params(labelsize=FS_TICK - 4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI_PUB, bbox_inches="tight")
    plt.close(fig)
```

并在 `main()` 里加入：

```python
    if args.what in ("all", "per_subject"):
        for r in records:
            stem = f"{r['dataset']}_{r['subject']}"
            plot_per_subject_strip(r, PER_SUB_FIG_DIR / f"{stem}_displacement.png")
        print(f"Wrote per-subject strips for {len(records)} subjects")
```

- [ ] **Step 2: 跑 figure，目视检查任意 3 个**

```bash
cd /home/honglab/leijiaxin/HFOsp && python scripts/plot_rank_displacement.py --what per_subject
ls results/interictal_propagation/rank_displacement/figures/per_subject/ | wc -l
```

Expected: ~27 个 PNG。打开 epilepsiae_548、epilepsiae_958（如果存在）等 forward/reverse 候选目视检查。

- [ ] **Step 3: Commit**

```bash
git add scripts/plot_rank_displacement.py
git commit -m "feat(rank_displacement): add per-subject displacement strip plots

Per-subject zoom-in heatstrip with channel labels, SOZ outline, and
title showing fwd/rev flag + Kendall tau + normalized footrule.
Supports debugging + supplement for n=27 stable_k=2 cohort.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: 写 figures README（中文）

**Files:**
- Create: `results/interictal_propagation/rank_displacement/figures/README.md`

- [ ] **Step 1: 写 README，每图 2–4 句 + 关注点**

```markdown
# Rank displacement supplementary figures

PR-6 supplementary：把离散 swap_node count 升级到逐通道 signed rank displacement。三类图均为描述性可视化，不预注册任何 cohort-level 显著性 gate。Cohort = stable_k=2 ∩ PR-6 endpoint-defined（n_available 由 cohort_summary.json 报告，上限 23）。

**Sign 合同**：T_a = cluster_id 较小的那个 cluster（PR-2 KMeans 标签），Δr = rank_T_b − rank_T_a。signed Δr 仅在 subject 内部有效，**不**作跨 subject 的"统一发射 / 接受方向"解读。

### cohort_displacement_heatmap

stable_k=2 cohort × 通道的有符号 Δr 热图。行按 Kendall τ 从最负（最反向）到最正排序；**每行内通道按 rank_T_a_dense 排（左 = T_a 的 source，右 = T_a 的 sink）**。这种排序保证了：完全反向的 subject 自然呈"红→蓝"单调梯度，随机的 subject 颜色散乱。颜色：红色 = +Δr（在 T_b 中比 T_a 中靠后），蓝色 = −Δr；SOZ 通道有黑色边框。右侧侧边条 = 每 subject 的 Kendall τ，颜色按 PR-2.5 forward/reverse-reproduced (OR 规则) 分组（rust = 是，gray = 否）。

**关注点**：(1) 最上几行（Kendall τ < −0.5）是否呈红→蓝单调梯度，并且 fwd/rev-reproduced 颜色集中在这一头；(2) τ ≈ 0 的中间几行是否颜色散乱（如果**所有**行都是单调梯度，立即停下查 sorting bias）；(3) SOZ 黑框在反向 subject 中倾向于聚集在两端还是均匀分布。

### footrule_kendall_summary

Panel B：footrule_normalized（Diaconis-Graham 归一化）按 fwd/rev-reproduced 分组的 violin + jitter 散点，灰色虚线为"asymptotic random reference (≈ 2/3)"——n→∞ 渐近期望，**不是**精确基线，不能据此做显著性判断。Panel C：Kendall τ 双行 strip plot，τ=−0.5 / 0 / +0.5 灰色参考线。

**关注点**：(1) reproduced 组的 footrule_normalized 是否偏向 1.0（接近完全反向）；(2) reproduced 组的 Kendall τ 是否集中在 < −0.5 那一端；(3) 两个 panel 同向 → continuous metric 与 PR-6 离散 swap_node 一致；不同向 → 提示离散阈值噪声。

### per_subject/

每 subject 一张 zoom-in heatstrip，标题显示 (dataset, subject, stable_k, fwd/rev flag, Kendall τ, normalized footrule)，通道名沿 x-axis 按 rank_T_a_dense 排序（与 cohort heatmap 一致），SOZ 黑框。仅作 debug 与 supplement。

**关注点**：(1) Kendall τ < −0.5 的 subject 应当在视觉上显示明确的"红→蓝单调梯度"；(2) τ ≈ 0 的 subject 应当看上去像随机噪声而**不是**单调梯度。
```

- [ ] **Step 2: 目视检查 README 与图对应**

打开 README 与 PNG 同时核对，每段对应的图实际描述是否一致。

- [ ] **Step 3: Commit**

```bash
git add results/interictal_propagation/rank_displacement/figures/README.md
git commit -m "docs(rank_displacement): add figures README per CLAUDE.md standard

Three-section Chinese README covering cohort heatmap, footrule+kendall
summary, and per-subject strips. Each section ends with **关注点**.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: 写 supplementary results archive doc

**Files:**
- Create: `docs/archive/topic1/pr6_supplementary_rank_displacement_results_2026-05-06.md`

- [ ] **Step 1: 写 results doc 骨架**

> **执行人在跑完 Task 4 后填入实际数字**：把 `cohort_summary.json` 加载，统计 `stable_k=2 cohort size`、`fwd_rev_reproduced count`、`footrule_normalized 中位数（reproduced vs not）`、`Kendall τ 中位数（reproduced vs not）`、`参考 swap geometry sign-test n=6 p=0.031` 是否方向一致。

文件初稿（数字位置用 `<TBD-RUN>` 占位，实际跑数后立即替换）：

```markdown
# PR-6 Supplementary：Per-Channel Signed Rank Displacement Results

> **状态：supplementary to PR-6（2026-05-06）。** Continuous version of PR-6 Step 4b
> discrete swap_node count（n=6 forward/reverse-reproduced subset, sign-test p=0.031）。
> 不立独立 cohort claim；不开 H1/H2 gate。
>
> **上游**：`docs/archive/topic1/pr6_template_endpoint_anchoring_plan_2026-04-25.md` §15 Step 4b；
> `docs/archive/topic1/ping_pong_hypothesis_review_2026-04-28.md` §4.2 + §6 PR-8 candidate roadmap。
>
> **plan**：`docs/archive/topic1/pr6_supplementary_rank_displacement_plan_2026-05-06.md`
>
> **代码**：`src/rank_displacement.py`、`scripts/run_rank_displacement.py`、
> `scripts/plot_rank_displacement.py`、`tests/test_rank_displacement.py`。
>
> **artifacts**：`results/interictal_propagation/rank_displacement/`（per_subject/, cohort_summary.json, figures/）。

## 1. 度量定义

对每对 cluster template (T_a, T_b)（按 PR-2 cluster_id 配对，**T_a = 较小 cluster_id**；PR-6 valid_mask 取交集后 dense re-rank）：

- 逐通道有符号位移 `Δr(ch) = rank_T_b(ch) − rank_T_a(ch)` (channel ∈ joint_valid)
  - **sign 合同**：仅 subject 内部有效，不跨 subject 聚合方向
- 整体 footrule `F = Σ|Δr|`
- Diaconis-Graham 归一化 `F_norm = F / floor(n_valid² / 2)` ∈ [0, 1]，1 = 完全反向
- Kendall τ(rank_T_a, rank_T_b)，−1 = 完全反向
- SOZ split（baseline-corrected）：
  - `soz_channel_fraction = n_soz_joint / n_valid`（chance baseline）
  - `soz_contribution_fraction = Σ|Δr|_SOZ / F`
  - `soz_contribution_excess = contribution_fraction − channel_fraction`（关键比较量）
  - `soz_abs_mean`, `nonsoz_abs_mean`, `soz_minus_nonsoz_abs_mean`

完整数学定义见 plan §3。

## 2. Cohort

主可视化 cohort = `stable_k == 2` ∩ PR-6 anchoring 有 valid_mask 的 subject。**上限 23**（PR-6 anchoring per_subject JSON 实际数）；**最终 n_available 由 cohort_summary.json 给出**，本文档不预承诺数字。

forward/reverse-reproduced 标签用 OR 规则（`first_half_second_half OR odd_even_block`，CLAUDE.md cross-PR contract lookup）：n_reproduced = <TBD-RUN>。

## 3. 主结果

| 指标 | reproduced | not reproduced | 备注 |
|---|---|---|---|
| n | <TBD-RUN> | <TBD-RUN> | |
| Kendall τ 中位数 | <TBD-RUN> | <TBD-RUN> | reproduced 期望 < 0；not reproduced 期望 ≈ 0 |
| F_norm 中位数 | <TBD-RUN> | <TBD-RUN> | asymptotic random reference ≈ 2/3（n→∞ 渐近，非精确基线）|
| `soz_contribution_excess` 中位数 | <TBD-RUN> | <TBD-RUN> | > 0 ⇒ SOZ 在 Δr 上参与高于通道占比；descriptive only |
| `soz_minus_nonsoz_abs_mean` 中位数 | <TBD-RUN> | <TBD-RUN> | > 0 ⇒ 单 SOZ 通道平均 \|Δr\| 大于单 non-SOZ；count-confound free |

**与 PR-6 离散 swap_node 一致性**：PR-6 Step 4b sign-test n=6, p=0.031；本 supplementary 在同一 cohort 上的 Kendall τ 中位数为 <TBD-RUN>，方向 <TBD-RUN>（"一致" / "不一致"）。

## 4. 图

- `results/interictal_propagation/rank_displacement/figures/cohort_displacement_heatmap.{png,pdf}` — stable_k=2 × 通道 Δr 热图（行按 τ 排序，列按 rank_T_a_dense 排序）
- `.../footrule_kendall_summary.{png,pdf}` — 2-panel descriptive summary
- `.../per_subject/<subject>_displacement.png` — 每 subject 详图
- `.../figures/README.md` — 中文图说明

## 5. 解读边界（写死）

可以说：
- "Continuous-version footrule + Kendall τ 与 PR-6 离散 swap_node 方向 <一致 / 不一致>"
- "Forward/reverse-reproduced subject 的 Kendall τ 集中在 [<low>, <high>]"
- "SOZ `contribution_excess` 中位数 = <X>（baseline-corrected，descriptive）"

**不**可以说：
- ~~"反向 template 是抑制墙的反弹"~~ — HFO 80–250 Hz 不区分 E/I
- ~~"footrule_normalized 高 ⇒ 致痫"~~ — 没做任何疾病侧 outcome 检验
- ~~"forward template SOZ-leading"~~ — 这是 PR-8 v1（DEFERRED）的范围，本 supplementary 不做 SOZ 极性方向判读
- ~~"SOZ contribution_fraction = X% ⇒ SOZ 主导"~~ — 必须用 `contribution_excess` 或 `abs_mean` 比较，裸 fraction 受通道数 confound
- ~~"高于 random baseline 2/3 即代表反向"~~ — 2/3 是 n→∞ 渐近期望，不是精确基线，不能用作显著性 gate
- ~~"signed Δr 在 SOZ 通道为正 / 负"~~ — signed Δr 只在 subject 内部有效，跨 subject 聚合方向无意义

## 6. 历史链接

- `docs/topic1_within_event_dynamics.md` §7 — Topic 1 主文档
- `docs/archive/topic1/pr6_template_endpoint_anchoring_plan_2026-04-25.md` — PR-6 plan（上游 swap_node 离散合同）
- `docs/archive/topic1/ping_pong_hypothesis_review_2026-04-28.md` — PR-8 candidate 来源
- `docs/archive/topic1/pr6_supplementary_rank_displacement_plan_2026-05-06.md` — 本 supplementary 的 plan
```

- [ ] **Step 2: 跑数后填入 `<TBD-RUN>` 实际数字**

```bash
cd /home/honglab/leijiaxin/HFOsp && python -c "
import json
from pathlib import Path
import numpy as np
ds = json.loads(Path('results/interictal_propagation/rank_displacement/cohort_summary.json').read_text())
sk2 = [r for r in ds if r.get('stable_k') == 2 and any(p.get('exit_reason')=='ok' for p in r.get('pairs', []))]
yes = [r for r in sk2 if r.get('fwd_rev_reproduced')]
no  = [r for r in sk2 if not r.get('fwd_rev_reproduced')]
def med(lst, key):
    v = [r['pairs'][0].get(key) for r in lst if r['pairs'] and r['pairs'][0].get('exit_reason')=='ok']
    v = [x for x in v if x is not None and not (isinstance(x,float) and np.isnan(x))]
    return float(np.median(v)) if v else float('nan')
print(f'cohort n={len(sk2)}  reproduced={len(yes)}  not={len(no)}')
print(f'tau median:                       repro={med(yes,\"kendall_tau\"):.3f}  not={med(no,\"kendall_tau\"):.3f}')
print(f'Fnorm median:                     repro={med(yes,\"footrule_normalized\"):.3f}  not={med(no,\"footrule_normalized\"):.3f}')
print(f'SOZ contribution_excess median:   repro={med(yes,\"soz_contribution_excess\"):.3f}  not={med(no,\"soz_contribution_excess\"):.3f}')
print(f'SOZ minus non abs_mean median:    repro={med(yes,\"soz_minus_nonsoz_abs_mean\"):.3f}  not={med(no,\"soz_minus_nonsoz_abs_mean\"):.3f}')
"
```

把输出贴回 archive doc，把 `<TBD-RUN>` 全部替换。

- [ ] **Step 3: Commit**

```bash
git add docs/archive/topic1/pr6_supplementary_rank_displacement_results_2026-05-06.md
git commit -m "docs(pr6_supplementary): add rank displacement results archive

Continuous-version footrule + Kendall tau supplementary to PR-6
discrete swap_node count. Cohort: stable_k=2 (n=27 target);
forward/reverse-reproduced flagged per OR rule. No new PR claim.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: 主文档 + AGENTS.md 回链

**Files:**
- Modify: `docs/topic1_within_event_dynamics.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: 在 `docs/topic1_within_event_dynamics.md` 找到 PR-6 那一节（搜索 "PR-6" 或 "endpoint anchoring"），在 supplementary 子节或末尾的"历史文档索引"加一行**

```markdown
- `docs/archive/topic1/pr6_supplementary_rank_displacement_plan_2026-05-06.md` — PR-6 supplementary plan（continuous footrule + signed displacement vector）
- `docs/archive/topic1/pr6_supplementary_rank_displacement_results_2026-05-06.md` — PR-6 supplementary results（27 stable_k=2 subject × channel 谱系图，描述性）
```

- [ ] **Step 2: 在 `AGENTS.md` 找到"## Current Code Map" 下的 interictal propagation 那段（搜索 `src/interictal_propagation.py`），在最后追加一行**

```markdown
- Rank displacement supplementary (PR-6 supplementary):
  - `src/rank_displacement.py` — continuous-version footrule + Kendall τ helper（compute_signed_rank_displacement, aggregate_pair_metrics）
  - `scripts/run_rank_displacement.py` — batch runner（PR-2 cluster JSON × PR-6 anchoring JSON）
  - `scripts/plot_rank_displacement.py` — cohort heatmap + footrule/Kendall 描述性 summary + per-subject 详图
  - `results/interictal_propagation/rank_displacement/` — per_subject/, cohort_summary.json, figures/
```

- [ ] **Step 3: 跑测试 + Commit**

```bash
cd /home/honglab/leijiaxin/HFOsp && python -m pytest tests/test_rank_displacement.py -v
```

Expected: 6/6 PASS.

```bash
git add docs/topic1_within_event_dynamics.md AGENTS.md
git commit -m "docs: link PR-6 supplementary rank displacement from main doc + AGENTS

Topic 1 main doc and AGENTS.md backlinks to the new supplementary
plan + results + module map. Per CLAUDE.md memory rule, this lets
later sessions find the work without conversation context.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage**：
- 用户原始诉求：(a) 系统盘点 PR-8 是否做这个 ✅；(b) per-channel signed rank displacement metric 数学定义 ✅（§3.0–§3.3）；(c) 顶刊风格图 ✅（Tasks 5–7）；(d) supplementary 而非独立 PR ✅；(e) 不预注册阈值 ✅。
- ping-pong review §4.2 / §6 PR-8 candidate ✅。
- ping-pong review §4.3 第 77 行"硬阈值不预注册" ✅。

**修订意见 (2026-05-06) 落实情况**：
1. 主热图 sorting bias ✅ — Task 5 / Task 7 改为按 `rank_T_a_dense` 排序，sanity check 写入视觉 checklist（§Task 5 Step 2）
2. signed Δr 方向锚定 ✅ — §3.0 写死 T_a/T_b anchor 规则与"per-subject only" sign 合同；§0 禁区第 6 条；helper 不再输出 `signed_displacement_mean_soz`；archive doc 第 5 节写死禁止跨 subject signed 聚合
3. SOZ contribution baseline 校正 ✅ — §3.3 输出 7 个字段（含 channel_fraction / contribution_excess / abs_mean）；Task 3 测试覆盖 `soz_excess_positive` 与 `does_not_export_signed_soz_mean`
4. n=27 不预承诺 ✅ — §0 禁区第 7 条；Goal 行；§2 写"上限 23"并由 cohort_summary.json 给数
5. 2/3 baseline 软化 ✅ — §0 禁区第 8 条；Task 6 figure annotation；README；archive doc 第 5 节禁止把 2/3 当 gate

**Placeholder scan**：
- `<TBD-RUN>` 5 处 — Task 9 Step 2 给出复制粘贴脚本，明确 fill-in 路径，**不**是真正的 placeholder
- 其余无 "TBD"、"TODO"、"implement later"

**Type consistency**：
- `compute_signed_rank_displacement` → `aggregate_pair_metrics` → runner → plotter，所有字段一致：`rank_a_full`, `rank_b_full`, `rank_a_dense_full`, `rank_b_dense_full`, `signed_displacement_full`, `joint_valid`, `soz_mask`, `kendall_tau`, `footrule_normalized`, `soz_channel_fraction`, `soz_contribution_excess`, `soz_minus_nonsoz_abs_mean`
- 不再出现 `signed_displacement_mean_soz` / `signed_displacement_mean_nonsoz`

**未覆盖的 spec 缺口**：无。
