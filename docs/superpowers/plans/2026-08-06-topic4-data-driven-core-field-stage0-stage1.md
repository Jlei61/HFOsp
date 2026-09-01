# Topic 4 数据驱动病理场 — Stage 0 + Stage 1 实施计划（rev2，第三轮审阅后）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建成"病理场参数化 + 冻结打分 + 网络缓存 + 纯函数报告器"四件基础设施，跑完 Stage 0 的 parity 复现与三参照、Stage 1 的 96 次配对探针，产出一份**描述性报告**供人判断。

**Architecture:** 四个纯计算模块 + 三个脚本。仿真部分**只调用**现有引擎与读出链路，不改任何引擎代码。**姿态是探索性的**（spec §0.4b）：只有数据完整性会 fail-closed，科学结论一律交人判断。

**Tech Stack:** Python 3, NumPy, SciPy (`truncnorm`, `spearmanr`, `binomtest`, `special.expit`), Matplotlib, pytest, `multiprocessing`。SNN 引擎在 `src/snn_engine/`（纯 NumPy CPU）。

**Spec:** `docs/superpowers/specs/2026-08-06-topic4-axis-constrained-data-driven-core-field-design.md`（rev3）。每个 Task 开工前**重读对应 §**。

## Global Constraints

- 被试 `epilepsiae_1146`，montage `narrow`，placement `gradient_shared`。共享轴 `u_C` 冻结（`theta_deg = -22.8`），任何代码不得重估。
- **不修改** `src/snn_engine/`、`scripts/run_sef_hfo_subject_snn.py`、`scripts/paper_figures/plot_fig_subject_snn_realvsmodel.py`。
- 引擎常数：`V_base=18.0`、`V_reset=11.0`、`core_mean=17.5`、`core_std=1.0`、`core_r=1.5`、`L=20.0`、`density=100.0`、`AR=2.0`、`g=3.6`、`dt=0.1`、`k_dir=2`、`N=40000`、`N_E=32000`。
- 场 config：`M=9`、`EPS=1e-3`、`TAU_H=0.25`、`A0=B0=1.5`、`SIGMA_S_FACTOR=1.2`、`AXIAL_MARGIN=2.0`、`SHIFT_MM=3.0`、`DELTA_EQ=0.05`。
- **唯一预算约束 `Σ h_i = N_core^manual`**；`d_i` 保留符号。横向自由度是**定面积长宽比 `ρ`**。
- **事件门只有一个：`n_part ≥ 2·k_dir+1 = 5`。`gate=4` 不存在**（signed 事件 `n_part` 最小值就是 5）。
- **`S_rank` 只在 `n_dir` 相同的种子内作差**；跨档位记 direction-tier win/loss。`n_dir=0 ⇒ S_rank=NaN`。
- **实现与测试必须先提交，再跑 Stage 0/1**；运行前断言工作树对本 plan 涉及文件干净。
- 全部产出落 `results/topic4_sef_hfo/data_driven_core_field/`。含图目录**必须**有中文 `figures/README.md`（`### filename` + 2–4 句 + 一行 `**关注点**：`），**图看过之后**写。
- 仿真相关测试标 `@pytest.mark.integration` / `@pytest.mark.slow`；纯计算测试必须秒级。

## 文件结构

| 文件 | 职责 |
|---|---|
| `src/topic4_core_field.py` | 坐标、partition-of-unity 基、八个臂的 `h`、latent-quantile 抽样、预算投影、`V_th`、非空性 pre-flight |
| `src/topic4_core_field_scoring.py` | 冻结支撑打分：患者模板、模型模板（逐方向 coverage）、2×2 Spearman、balanced pair score、`axis_only`、字典序键、对抗诊断 |
| `src/topic4_core_field_report.py` | Stage 1 报告器（纯函数；仅完整性 fail-closed） |
| `src/topic4_core_field_runner.py` | 网络缓存（完整配置哈希 + 原子写）、provenance、单臂运行 |
| `scripts/run_topic4_core_field_stage0.py` | parity 复现、三参照、冻结 config |
| `scripts/run_topic4_core_field_stage1.py` | pre-flight + **按网络种子并行** 8 臂 × 12 种子 |
| `scripts/analyze_topic4_core_field_stage1.py` | 对比表、concordance、报告、图 + README |
| `tests/test_topic4_core_field.py` | Task 1–3 |
| `tests/test_topic4_core_field_scoring.py` | Task 4–5 |
| `tests/test_topic4_core_field_cache.py` | Task 6 |
| `tests/test_topic4_core_field_crn.py` | Task 7（integration） |
| `tests/test_topic4_core_field_report.py` | Task 8 |

---

### Task 1: 轴向坐标与 partition-of-unity 基

**重读 spec：** §4.1、§4.4

**Files:** Create `src/topic4_core_field.py`；Test `tests/test_topic4_core_field.py`

**Interfaces:**
- Produces: `axis_coords(pos, center, u_axis) -> (s, r)`；`axial_basis_centers(s_support, M) -> kappa`；`partition_of_unity(s, kappa, sigma_s) -> Phi`（`(len(s), M)`，行和为 1）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_topic4_core_field.py
import numpy as np
import pytest
from src.topic4_core_field import axis_coords, axial_basis_centers, partition_of_unity


def test_axis_coords_projects_onto_axis_and_perpendicular():
    pos = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    s, r = axis_coords(pos, np.array([0.0, 0.0]), np.array([1.0, 0.0]))
    assert np.allclose(s, [1.0, 0.0, 2.0])
    assert np.allclose(np.abs(r), [0.0, 1.0, 2.0])


def test_axis_coords_axis_flip_negates_s_and_preserves_abs_r():
    rng = np.random.default_rng(0)
    pos = rng.uniform(-5, 5, size=(50, 2))
    center, u = np.array([0.3, -0.2]), np.array([0.6, 0.8])
    s1, r1 = axis_coords(pos, center, u)
    s2, r2 = axis_coords(pos, center, -u)
    assert np.allclose(s2, -s1)
    assert np.allclose(np.abs(r2), np.abs(r1))


def test_partition_of_unity_rows_sum_to_one():
    kappa = axial_basis_centers((-8.0, 8.0), M=9)
    s = np.linspace(-8.0, 8.0, 200)
    Phi = partition_of_unity(s, kappa, 1.2 * (kappa[1] - kappa[0]))
    assert Phi.shape == (200, 9)
    assert np.allclose(Phi.sum(axis=1), 1.0, atol=1e-12)


def test_uniform_weights_give_a_flat_axial_profile():
    """Why partition-of-unity is required: unnormalised Gaussians sag where fewer
    bases overlap, which would make `uniform_axial` a broad peak, not a corridor."""
    kappa = axial_basis_centers((-8.0, 8.0), M=9)
    s = np.linspace(-8.0, 8.0, 200)
    profile = partition_of_unity(s, kappa, 1.2 * (kappa[1] - kappa[0])) @ np.full(9, 1 / 9)
    assert (profile.max() - profile.min()) / profile.mean() < 1e-6
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_topic4_core_field.py -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.topic4_core_field'`

- [ ] **Step 3: 最小实现**

```python
# src/topic4_core_field.py
"""Topic 4 axis-constrained data-driven pathology field (spec rev3).

Pure computation: no simulation, no engine import.
"""
from __future__ import annotations

import numpy as np

M_DEFAULT = 9
EPS = 1e-3
TAU_H = 0.25
A0 = 1.5
B0 = 1.5
AXIAL_MARGIN = 2.0
SIGMA_S_FACTOR = 1.2
SHIFT_MM = 3.0


def axis_coords(pos, center, u_axis):
    """Axial (s) and transverse (r) coordinates. u_axis is undirected: flipping
    its sign negates both, which every score must be invariant to."""
    pos = np.asarray(pos, float)
    u = np.asarray(u_axis, float)
    u = u / np.linalg.norm(u)
    u_perp = np.array([-u[1], u[0]])
    d = pos - np.asarray(center, float)[None, :]
    return d @ u, d @ u_perp


def axial_basis_centers(s_support, M=M_DEFAULT):
    return np.linspace(float(s_support[0]), float(s_support[1]), int(M))


def partition_of_unity(s, kappa, sigma_s):
    """Normalised Gaussian bases: rows sum to exactly 1 (spec 4.1)."""
    s = np.asarray(s, float)
    kappa = np.asarray(kappa, float)
    logw = -((s[:, None] - kappa[None, :]) ** 2) / (2.0 * float(sigma_s) ** 2)
    logw -= logw.max(axis=1, keepdims=True)
    w = np.exp(logw)
    return w / w.sum(axis=1, keepdims=True)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_topic4_core_field.py -x -q`
Expected: 4 passed

- [ ] **Step 5: 提交**

```bash
git add src/topic4_core_field.py tests/test_topic4_core_field.py
git commit -m "feat(topic4-core-field): axial coordinates and a partition-of-unity basis

Uniform weights on unnormalised Gaussians sag where fewer bases overlap, which
would have made the uniform-axial control a broad peak rather than the flat
corridor every comparison is measured against."
```

---

### Task 2: latent-quantile 抽样与预算投影

**重读 spec：** §4.3.1、§4.3.2、§2.5

**Files:** Modify `src/topic4_core_field.py`；Test `tests/test_topic4_core_field.py`

**Interfaces:**
- Produces: `sample_core_quantiles(n_E, seed) -> u`；`core_thresholds(u, core_mean=17.5, core_std=1.0, v_reset=11.0) -> V_core`；`signed_depth(V_core, v_base=18.0) -> d`；`project_to_budget(q, target_count, tau_h=TAU_H) -> (h, lam)`；`build_vth(h, d, n_total, n_E, v_base=18.0) -> vth`

- [ ] **Step 1: 写失败测试**

```python
# append to tests/test_topic4_core_field.py
from src.topic4_core_field import (
    EPS, TAU_H, build_vth, core_thresholds, project_to_budget,
    sample_core_quantiles, signed_depth,
)


def test_core_thresholds_match_the_truncated_normal_moments():
    v = core_thresholds(sample_core_quantiles(200_000, seed=7))
    assert v.min() >= 11.0
    assert abs(v.mean() - 17.5) < 0.02
    assert abs(v.std() - 1.0) < 0.02


def test_signed_depth_keeps_the_negative_third():
    """About 31% of 'core' neurons sit ABOVE baseline; max(0,.) would drop them
    and break parity with the accepted manual core (spec C1)."""
    d = signed_depth(core_thresholds(sample_core_quantiles(200_000, seed=7)))
    assert 0.28 < (d < 0).mean() < 0.34
    assert abs(d.mean() - 0.5) < 0.02


def test_budget_projection_hits_the_target_count():
    q = np.random.default_rng(1).uniform(EPS, 1.0, size=32_000)
    h, lam = project_to_budget(q, target_count=1131.0)
    assert np.isfinite(lam) and (h >= 0).all() and (h <= 1).all()
    assert abs(h.sum() - 1131.0) / 1131.0 < 1e-6


def test_budget_projection_is_monotone_in_lambda():
    """Strictly decreasing, so the root is unique. Budgeting on sum(h*d) would
    NOT be monotone once a third of d is negative."""
    from scipy.special import expit
    q = np.random.default_rng(2).uniform(EPS, 1.0, size=5_000)
    lq = np.log(q + EPS)
    totals = [expit((lq - lam) / TAU_H).sum() for lam in np.linspace(-8, 2, 25)]
    assert all(b < a for a, b in zip(totals, totals[1:]))


@pytest.mark.parametrize("target", [0.0, -1.0, 32_000.0, 40_000.0])
def test_budget_projection_rejects_an_out_of_range_target(target):
    q = np.random.default_rng(3).uniform(EPS, 1.0, size=32_000)
    with pytest.raises(ValueError):
        project_to_budget(q, target_count=target)


def test_budget_projection_rejects_non_finite_input():
    q = np.random.default_rng(4).uniform(EPS, 1.0, size=100)
    q[3] = np.nan
    with pytest.raises(ValueError):
        project_to_budget(q, target_count=10.0)


def test_budget_projection_does_not_overflow_on_an_extreme_field():
    """expit, not 1/(1+exp(-x)): the naive form overflows for large |x|."""
    q = np.concatenate([np.full(500, 1e-12), np.full(500, 1e6)])
    with np.errstate(over="raise"):
        h, _ = project_to_budget(q, target_count=500.0)
    assert np.isfinite(h).all()


def test_build_vth_places_baseline_outside_and_core_distribution_inside():
    n_E, n_total = 1000, 1250
    d = signed_depth(core_thresholds(sample_core_quantiles(n_E, seed=3)))
    h = np.zeros(n_E); h[:100] = 1.0
    vth = build_vth(h, d, n_total=n_total, n_E=n_E)
    assert vth.shape == (n_total,)
    assert np.allclose(vth[100:], 18.0)
    assert np.allclose(vth[:100], 18.0 - d[:100])
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_topic4_core_field.py -x -q`
Expected: FAIL — `ImportError: cannot import name 'sample_core_quantiles'`

- [ ] **Step 3: 最小实现**

```python
# append to src/topic4_core_field.py
from scipy.special import expit
from scipy.stats import truncnorm

V_BASE = 18.0
V_RESET = 11.0
CORE_MEAN = 17.5
CORE_STD = 1.0


def sample_core_quantiles(n_E, seed):
    """One uniform quantile per E neuron, drawn once and frozen. Position- and
    field-independent, so every arm shares the same latent draw."""
    return np.random.default_rng(int(seed)).uniform(0.0, 1.0, size=int(n_E))


def core_thresholds(u, core_mean=CORE_MEAN, core_std=CORE_STD, v_reset=V_RESET):
    """Truncated-normal inverse transform: same distribution as the engine's
    rejection sampler, but deterministic per neuron. Bitwise reproduction of the
    legacy draw is impossible -- rejection makes its stream position data-dependent."""
    a = (float(v_reset) - float(core_mean)) / float(core_std)
    return truncnorm.ppf(np.asarray(u, float), a=a, b=np.inf,
                         loc=float(core_mean), scale=float(core_std))


def signed_depth(v_core, v_base=V_BASE):
    return float(v_base) - np.asarray(v_core, float)


def project_to_budget(q, target_count, tau_h=TAU_H, eps=EPS, max_iter=200):
    """Bisect lambda so that sum_i h_i == target_count.

    h is strictly decreasing in lambda, so the root is unique. This is a
    LEVEL-SET operation: the region's size is pinned by the budget and q only
    sets its shape (spec 4.4).
    """
    q = np.asarray(q, float)
    if not np.isfinite(q).all():
        raise ValueError("project_to_budget: q contains non-finite values")
    if (q + eps <= 0).any():
        raise ValueError("project_to_budget: q + eps must be positive")
    target = float(target_count)
    if not np.isfinite(target) or not (0.0 < target < q.size):
        raise ValueError(
            f"project_to_budget: target_count must lie in (0, {q.size}), got {target}")

    lq = np.log(q + eps)
    lo, hi = lq.min() - 20.0, lq.max() + 20.0
    for _ in range(max_iter):
        lam = 0.5 * (lo + hi)
        if expit((lq - lam) / tau_h).sum() > target:
            lo = lam
        else:
            hi = lam
    lam = 0.5 * (lo + hi)
    return expit((lq - lam) / tau_h), lam


def build_vth(h, d, n_total, n_E, v_base=V_BASE):
    """Per-neuron threshold vector for the engine. I neurons keep baseline."""
    vth = np.full(int(n_total), float(v_base))
    vth[:int(n_E)] = float(v_base) - np.asarray(h, float) * np.asarray(d, float)
    return vth
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_topic4_core_field.py -x -q`
Expected: 13 passed

- [ ] **Step 5: 提交**

```bash
git add src/topic4_core_field.py tests/test_topic4_core_field.py
git commit -m "feat(topic4-core-field): latent-quantile sampling and effective-count budget

Budgeting on sum(h) keeps the bisection strictly monotone, which sum(h*d) is not
once a third of d is negative. expit rather than the naive logistic, and the
target is range-checked against the population size."
```

---

### Task 3: 八个臂与逐对比的非空性 pre-flight

**重读 spec：** §7.2、§4.4

**Files:** Modify `src/topic4_core_field.py`；Test `tests/test_topic4_core_field.py`

**Interfaces:**
- Consumes: Task 1、Task 2
- Produces: `ARM_NAMES`（8 个）；`SHAPE_CHECKS`；`two_core_q(s, r, sep, rho, a0, b0, r_shift, eps)`；`uniform_axial_q(...)`；`manual_mask(pos, src_xy, snk_xy, core_r) -> bool[]`；`arm_h(name, s, r, geom, target_count, manual_mask_E=None) -> h`；`shape_metrics(h, s, r) -> dict`；`preflight_shape(h_by_arm, s, r, target_count, checks=None) -> dict`

**关键（第三轮 P0-1）**：`manual_projected` **就是 legacy 的两个圆盘 hard mask**，只换 threshold 抽样；平滑参考是**另一个臂** `manual_smooth`。B 组形状对比的基准臂是 `manual_smooth`。

- [ ] **Step 1: 写失败测试**

```python
# append to tests/test_topic4_core_field.py
from src.topic4_core_field import (
    ARM_NAMES, arm_h, manual_mask, preflight_shape, shape_metrics,
)

SEP = 6.0


def _mock_sheet(n=32_000, L=20.0, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, L, size=(n, 2))
    return pos, pos[:, 0] - L / 2.0, pos[:, 1] - L / 2.0


def _geom():
    return dict(sep=SEP, s_support=(-8.0, 8.0), M=9, sigma_perp=1.5, shift_mm=3.0)


def _mask(s, r, core_r=1.5):
    """Cores sit at +-sep/2, matching two_core_q."""
    return (np.minimum((s - SEP / 2) ** 2, (s + SEP / 2) ** 2) + r ** 2) <= core_r ** 2


def test_there_are_eight_arms_including_both_manual_variants():
    assert len(ARM_NAMES) == 8
    for name in ("manual_hard", "manual_projected", "manual_smooth"):
        assert name in ARM_NAMES


def test_manual_projected_is_exactly_the_hard_mask():
    """spec 4.3.1: manual_projected changes the DRAWS, not the mask. If it were a
    smoothed field, comparison A would move three things at once."""
    _, s, r = _mock_sheet()
    m = _mask(s, r)
    h = arm_h("manual_projected", s, r, _geom(), float(m.sum()), manual_mask_E=m)
    assert np.array_equal(h, m.astype(float))


def test_manual_smooth_is_close_to_but_not_identical_to_the_hard_mask():
    _, s, r = _mock_sheet()
    m = _mask(s, r)
    h = arm_h("manual_smooth", s, r, _geom(), float(m.sum()), manual_mask_E=m)
    assert np.corrcoef(h, m.astype(float))[0, 1] >= 0.9
    assert not np.array_equal(h, m.astype(float))


def test_all_arms_hit_the_same_budget():
    _, s, r = _mock_sheet()
    m = _mask(s, r)
    for name in ARM_NAMES:
        if name == "manual_hard":
            continue
        h = arm_h(name, s, r, _geom(), float(m.sum()), manual_mask_E=m)
        assert abs(h.sum() - m.sum()) / m.sum() < 1e-6, name


def test_width_arms_reshape_rather_than_blur():
    """spec 4.4: bare sigma_perp only blurs the edge because the budget pins the
    area; rho reshapes it at fixed a*b."""
    _, s, r = _mock_sheet()
    m = _mask(s, r); target = float(m.sum())
    wide = arm_h("width_wide", s, r, _geom(), target, manual_mask_E=m)
    narrow = arm_h("width_narrow", s, r, _geom(), target, manual_mask_E=m)
    assert shape_metrics(wide, s, r)["rms_transverse"] > \
           2.5 * shape_metrics(narrow, s, r)["rms_transverse"]


def test_transverse_arms_are_mirror_images():
    _, s, r = _mock_sheet()
    m = _mask(s, r); target = float(m.sum())
    plus = arm_h("transverse_plus", s, r, _geom(), target, manual_mask_E=m)
    minus = arm_h("transverse_minus", s, -r, _geom(), target, manual_mask_E=_mask(s, -r))
    assert np.corrcoef(plus, minus)[0, 1] > 0.999


def test_preflight_covers_shape_comparisons_only_and_excludes_the_equivalence_arms():
    """P0-2: manual_hard and manual_projected SHOULD be near-identical. An
    all-pairs correlation gate would reject the correct implementation."""
    _, s, r = _mock_sheet()
    m = _mask(s, r); target = float(m.sum())
    h_by_arm = {n: arm_h(n, s, r, _geom(), target, manual_mask_E=m)
                for n in ARM_NAMES if n != "manual_hard"}
    h_by_arm["manual_hard"] = m.astype(float)
    rep = preflight_shape(h_by_arm, s, r, target)
    assert rep["ok"] is True
    assert set(rep["checks"]) == {"B1", "B2", "B3", "B4"}


def test_preflight_fails_when_a_shape_arm_collapses_onto_the_baseline():
    _, s, r = _mock_sheet()
    m = _mask(s, r); target = float(m.sum())
    h_by_arm = {n: arm_h(n, s, r, _geom(), target, manual_mask_E=m)
                for n in ARM_NAMES if n != "manual_hard"}
    h_by_arm["manual_hard"] = m.astype(float)
    h_by_arm["uniform_axial"] = h_by_arm["manual_smooth"].copy()   # collapse B1
    rep = preflight_shape(h_by_arm, s, r, target)
    assert rep["ok"] is False
    assert rep["checks"]["B1"]["ok"] is False
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_topic4_core_field.py -x -q`
Expected: FAIL — `ImportError: cannot import name 'ARM_NAMES'`

- [ ] **Step 3: 最小实现**

```python
# append to src/topic4_core_field.py

ARM_NAMES = (
    "manual_hard",        # legacy engine path (rejection sampler + np.minimum)
    "manual_projected",   # SAME hard mask, latent-quantile draws  -> comparison A
    "manual_smooth",      # smoothed two-core through the budget   -> baseline for B1-B4
    "uniform_axial",
    "width_wide",
    "width_narrow",
    "transverse_plus",
    "transverse_minus",
)

# Shape comparisons and the quantity each one must actually move (spec 4.4).
SHAPE_CHECKS = {
    "B1": dict(a="manual_smooth", b="uniform_axial",
               metric="rms_axial", kind="rel", threshold=0.20),
    "B2": dict(a="manual_smooth", b="width_wide",
               metric="aspect", kind="rel", threshold=0.50),
    "B3": dict(a="manual_smooth", b="width_narrow",
               metric="aspect", kind="rel", threshold=0.50),
    "B4": dict(a="manual_smooth", b="transverse_plus",
               metric="centroid_transverse", kind="abs", threshold=1.5),
}


def manual_mask(pos, src_xy, snk_xy, core_r):
    """The legacy two-disk core mask, in sheet coordinates."""
    pos = np.asarray(pos, float)
    d = np.minimum(((pos - np.asarray(src_xy, float)) ** 2).sum(1),
                   ((pos - np.asarray(snk_xy, float)) ** 2).sum(1))
    return d <= float(core_r) ** 2


def two_core_q(s, r, sep, rho=1.0, a0=A0, b0=B0, r_shift=0.0, eps=EPS):
    """Two elliptical cores at s = +-sep/2, transverse offset r_shift.

    rho is the FIXED-AREA aspect ratio (a = a0*rho, b = b0/rho), so it reshapes
    the region instead of merely blurring its edge (spec 4.4).
    """
    a, b = float(a0) * float(rho), float(b0) / float(rho)
    s = np.asarray(s, float)
    rr = np.asarray(r, float) - float(r_shift)
    q = np.zeros_like(s)
    for c in (-float(sep) / 2.0, float(sep) / 2.0):
        q = np.maximum(q, np.exp(-((s - c) ** 2 / (2 * a ** 2) + rr ** 2 / (2 * b ** 2))))
    return q + eps


def uniform_axial_q(s, r, kappa, sigma_s, sigma_perp, eps=EPS):
    """Flat axial profile: pi_m == 1/M on a partition-of-unity basis."""
    M = len(kappa)
    profile = partition_of_unity(np.asarray(s, float), kappa, sigma_s) @ np.full(M, 1.0 / M)
    return profile * np.exp(-np.asarray(r, float) ** 2 / (2 * float(sigma_perp) ** 2)) + eps


def arm_h(name, s, r, geom, target_count, manual_mask_E=None):
    """h field for one Stage 1 arm.

    manual_hard is not built here (legacy engine path). manual_projected is the
    hard mask verbatim -- it changes the DRAWS, not the geometry (spec 4.3.1).
    """
    if name == "manual_projected":
        if manual_mask_E is None:
            raise ValueError("manual_projected requires manual_mask_E")
        return np.asarray(manual_mask_E, bool).astype(float)
    sep = geom["sep"]
    if name == "manual_smooth":
        q = two_core_q(s, r, sep, rho=1.0)
    elif name == "uniform_axial":
        kappa = axial_basis_centers(geom["s_support"], geom["M"])
        q = uniform_axial_q(s, r, kappa, SIGMA_S_FACTOR * (kappa[1] - kappa[0]),
                            geom["sigma_perp"])
    elif name == "width_wide":
        q = two_core_q(s, r, sep, rho=0.5)
    elif name == "width_narrow":
        q = two_core_q(s, r, sep, rho=2.0)
    elif name == "transverse_plus":
        q = two_core_q(s, r, sep, rho=1.0, r_shift=+geom["shift_mm"])
    elif name == "transverse_minus":
        q = two_core_q(s, r, sep, rho=1.0, r_shift=-geom["shift_mm"])
    else:
        raise ValueError(f"arm_h does not build {name!r}")
    h, _ = project_to_budget(q, target_count)
    return h


def shape_metrics(h, s, r):
    """h-weighted geometry. Uses h, never h*d -- d is signed and h*d is not a
    non-negative mass (spec 9 / P0-7)."""
    h = np.asarray(h, float)
    w = h.sum()
    rms_ax = float(np.sqrt((h * np.asarray(s, float) ** 2).sum() / w))
    rms_tr = float(np.sqrt((h * np.asarray(r, float) ** 2).sum() / w))
    return dict(rms_axial=rms_ax, rms_transverse=rms_tr,
                aspect=rms_tr / rms_ax if rms_ax > 0 else np.inf,
                centroid_transverse=float((h * np.asarray(r, float)).sum() / w),
                centroid_axial=float((h * np.asarray(s, float)).sum() / w),
                budget=float(w))


def preflight_shape(h_by_arm, s, r, target_count, checks=None):
    """Refuse to launch 96 simulations on a vacuous shape comparison.

    Only the B comparisons are checked. manual_hard and manual_projected SHOULD
    be near-identical -- an all-pairs correlation gate would reject the correct
    implementation (third-review P0-2). Correlation is reported as a diagnostic,
    never as the gate.
    """
    checks = checks or SHAPE_CHECKS
    metrics = {name: shape_metrics(h, s, r) for name, h in h_by_arm.items()}
    out, ok_all = {}, True
    for key, c in checks.items():
        ma, mb = metrics[c["a"]][c["metric"]], metrics[c["b"]][c["metric"]]
        if c["kind"] == "rel":
            observed = abs(ma - mb) / max(abs(ma), abs(mb), 1e-12)
        else:
            observed = abs(ma - mb)
        ok = bool(observed >= c["threshold"])
        ok_all &= ok
        out[key] = dict(ok=ok, a=c["a"], b=c["b"], metric=c["metric"],
                        observed=float(observed), threshold=c["threshold"],
                        correlation=float(np.corrcoef(h_by_arm[c["a"]],
                                                      h_by_arm[c["b"]])[0, 1]))
    budget_err = {n: abs(m["budget"] - target_count) / target_count
                  for n, m in metrics.items()}
    worst = max(budget_err.values())
    ok_all &= bool(worst < 1e-6)
    return dict(ok=bool(ok_all), checks=out, metrics=metrics,
                worst_budget_error=float(worst))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_topic4_core_field.py -x -q`
Expected: 21 passed

- [ ] **Step 5: 提交**

```bash
git add src/topic4_core_field.py tests/test_topic4_core_field.py
git commit -m "feat(topic4-core-field): eight arms, with manual_projected as the hard mask

An earlier draft made manual_projected a smoothed field, so comparing it with
manual_hard moved the sampling contract, the hard-versus-smooth distinction and
the edge treatment together. It is now the mask verbatim, and a separate
manual_smooth arm is the baseline the shape comparisons run against.

The pre-flight checks the quantity each comparison is about rather than demanding
that every pair of arms differ, which would have rejected the equivalence arm."
```

---

### Task 4: 冻结支撑打分（逐方向 coverage + balanced pair score）

**重读 spec：** §5.1、§5.2、§5.2a、§5.4、§5.5

**Files:** Create `src/topic4_core_field_scoring.py`；Test `tests/test_topic4_core_field_scoring.py`

**Interfaces:**
- Produces: `PART_MIN = 5`；`load_patient_templates(subject, source, root=".")`；`model_templates(events, support, part_min=PART_MIN)`（含 `forward` / `reverse` / `n_dir` / `coverage_forward` / `coverage_reverse` / `coverage_union` / `mean_n_part`）；`sim_matrix(model, target, support, missing_rule)`；`assignment_invariant_S(M)`；`balanced_pair_score(model, target, support)`；`axis_only_templates(names, coords, center, u_axis)`；`adversarial_gain(model, target, support, missing_rule)`

**事件门只有 `part_min = 5`。** `gate=4` 不存在。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_topic4_core_field_scoring.py
import numpy as np
import pytest
from src.topic4_core_field_scoring import (
    adversarial_gain, assignment_invariant_S, axis_only_templates,
    balanced_pair_score, model_templates, sim_matrix,
)

SUPPORT = ["c1", "c2", "c3", "c4", "c5", "c6"]
TARGET = {"t_a": {n: float(i) for i, n in enumerate(SUPPORT)},
          "t_b": {n: float(-i) for i, n in enumerate(SUPPORT)}}


def _ev(sign, ranks):
    return {"sign": sign, "n_part": len(ranks), "ranks": ranks}


def _both(names_fwd, names_rev):
    return [_ev(1.0, {n: float(i) for i, n in enumerate(names_fwd)}),
            _ev(-1.0, {n: float(-i) for i, n in enumerate(names_rev)})]


def test_model_templates_never_widen_beyond_the_frozen_support():
    m = model_templates([_ev(1.0, {"c1": 0.0, "c2": 1.0, "c3": 2.0, "c4": 3.0,
                                   "c5": 4.0, "zzz": 9.0})], SUPPORT, part_min=5)
    assert set(m["forward"]) <= set(SUPPORT)


def test_events_below_the_participation_floor_are_dropped():
    m = model_templates([_ev(1.0, {"c1": 0.0, "c2": 1.0})], SUPPORT, part_min=5)
    assert m["n_dir"] == 0


def test_coverage_is_reported_per_direction_not_only_as_a_union():
    """P0-4: a union hides 'each direction covers a different small patch'."""
    m = model_templates(_both(SUPPORT[:5], SUPPORT[1:6]), SUPPORT, part_min=5)
    assert m["coverage_forward"] == pytest.approx(5 / 6)
    assert m["coverage_reverse"] == pytest.approx(5 / 6)
    assert m["coverage_union"] == pytest.approx(1.0)
    assert m["coverage_union"] > min(m["coverage_forward"], m["coverage_reverse"])


def test_score_is_invariant_to_swapping_the_two_patient_templates():
    m = model_templates(_both(SUPPORT, SUPPORT), SUPPORT, part_min=5)
    s1 = assignment_invariant_S(sim_matrix(m, TARGET, SUPPORT, "mean_rank"))
    s2 = assignment_invariant_S(
        sim_matrix(m, {"t_a": TARGET["t_b"], "t_b": TARGET["t_a"]}, SUPPORT, "mean_rank"))
    assert s1 == pytest.approx(s2)


def test_a_single_direction_model_yields_nan_not_a_best_cell():
    """spec 5.3: S_rank must never be a number that can be differenced against a
    two-direction score. One direction -> undefined."""
    m = model_templates([_ev(1.0, {n: float(i) for i, n in enumerate(SUPPORT)})],
                        SUPPORT, part_min=5)
    assert m["n_dir"] == 1
    assert np.isnan(assignment_invariant_S(sim_matrix(m, TARGET, SUPPORT, "mean_rank")))


def test_balanced_pair_score_is_assignment_invariant_and_bidirectional():
    m = model_templates(_both(SUPPORT, SUPPORT), SUPPORT, part_min=5)
    s1 = balanced_pair_score(m, TARGET, SUPPORT)
    s2 = balanced_pair_score(m, {"t_a": TARGET["t_b"], "t_b": TARGET["t_a"]}, SUPPORT)
    assert s1 == pytest.approx(s2)
    one = model_templates([_ev(1.0, {n: float(i) for i, n in enumerate(SUPPORT)})],
                          SUPPORT, part_min=5)
    assert np.isnan(balanced_pair_score(one, TARGET, SUPPORT))


def test_balanced_pair_score_uses_a_fixed_denominator():
    full = model_templates(_both(SUPPORT, SUPPORT), SUPPORT, part_min=5)
    half = model_templates(_both(SUPPORT[:5], SUPPORT[:5]), SUPPORT, part_min=5)
    assert balanced_pair_score(half, TARGET, SUPPORT) < \
           balanced_pair_score(full, TARGET, SUPPORT)


def test_axis_only_templates_are_exact_mirrors():
    names = ["c1", "c2", "c3"]
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    t = axis_only_templates(names, coords, np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    assert np.allclose([t["forward"][n] for n in names], [-t["reverse"][n] for n in names])


def test_adversarial_gain_reports_how_much_dropping_one_contact_can_help():
    """Diagnostic, not an assertion: under mean-rank filling a badly ranked
    contact CAN be worth dropping, and we need to see by how much."""
    ranks = {n: float(i) for i, n in enumerate(SUPPORT)}
    ranks["c3"] = 99.0
    m = model_templates([_ev(1.0, ranks), _ev(-1.0, {n: -v for n, v in ranks.items()})],
                        SUPPORT, part_min=5)
    g = adversarial_gain(m, TARGET, SUPPORT, "mean_rank")
    assert g["worst_contact"] == "c3"
    assert g["gain"] > 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_topic4_core_field_scoring.py -x -q`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 最小实现**

```python
# src/topic4_core_field_scoring.py
"""Frozen-support scoring for the data-driven core field (spec section 5).

Reimplements the definitions in
scripts/paper_figures/plot_fig_subject_snn_realvsmodel.py (templates from event
sign; 2x2 Spearman against the patient's two templates) and adds the frozen
scoring support that file cannot provide. That file is NOT modified -- it carries
published numbers.
"""
from __future__ import annotations

import itertools
import json
import os

import numpy as np
from scipy.stats import spearmanr

GRADIENT_ROOT = "results/interictal_propagation_masked/template_gradient_fields/per_subject"
GEOMETRY_ROOT = "results/spatial_modulation/propagation_geometry/observation_readout/real_subjects"
PART_MIN = 5   # 2*k_dir+1 with k_dir=2; endpoint_centroid_axis returns None below this,
               # so a post-hoc gate of 4 admits nothing and is not an axis.


def load_patient_templates(subject, source, root="."):
    """Patient TA/TB contact ranks. Two non-identical sources, both reported."""
    if source == "gradient":
        field = json.load(open(os.path.join(
            root, GRADIENT_ROOT, f"{subject}.json")))["interictal_field"]
        names = [str(x) for x in field["contact_order"]]
        return {tpl: {n: float(v) for n, v in zip(names, np.asarray(field[key], float))
                      if np.isfinite(v)}
                for key, tpl in (("rank_a", "t_a"), ("rank_b", "t_b"))}
    if source == "geometry":
        out = {}
        for tpl in ("t_a", "t_b"):
            g = json.load(open(os.path.join(root, GEOMETRY_ROOT, f"{subject}_{tpl}.json")))
            out[tpl] = {c["name"]: float(c["typical_rank"]) for c in g["channels"]
                        if c.get("typical_rank") is not None}
        return out
    raise ValueError(f"unknown template source {source!r}")


def model_templates(events, support, part_min=PART_MIN):
    """Forward/reverse mean within-event rank on the frozen support.

    Templates come from event SIGN, not cluster labels: for a one-direction
    readout a cluster->direction mapping would be fabricated.
    """
    support = list(support)
    idx = {n: i for i, n in enumerate(support)}
    usable = [e for e in events
              if e.get("sign") is not None and int(e.get("n_part", 0)) >= int(part_min)]
    acc = {+1: [[] for _ in support], -1: [[] for _ in support]}
    for e in usable:
        key = +1 if e["sign"] > 0 else -1
        for name, v in (e.get("ranks") or {}).items():
            if v is not None and name in idx:
                acc[key][idx[name]].append(float(v))
    out = {}
    for key, label in ((+1, "forward"), (-1, "reverse")):
        tpl = {support[i]: float(np.mean(vals)) for i, vals in enumerate(acc[key]) if vals}
        out[label] = tpl
        out[f"coverage_{label}"] = len(tpl) / len(support) if support else 0.0
    out["n_dir"] = int(bool(out["forward"])) + int(bool(out["reverse"]))
    union = set(out["forward"]) | set(out["reverse"])
    out["coverage_union"] = len(union) / len(support) if support else 0.0
    out["mean_n_part"] = float(np.mean([e["n_part"] for e in usable])) if usable else 0.0
    return out


def _aligned(model_tpl, target_tpl, support, missing_rule):
    """Vectors on the FULL frozen support.

    'mean_rank' fills a contact the candidate never recruited with that
    direction's mean -- an explicit modelling assumption, which is why
    balanced_pair_score and adversarial_gain are reported alongside.
    'common_only' is the legacy candidate-dependent support: regression and
    sensitivity only, never load-bearing.
    """
    names = [n for n in support if n in target_tpl]
    if missing_rule == "common_only":
        names = [n for n in names if n in model_tpl]
        if len(names) < 4:
            return None, None
        return (np.array([model_tpl[n] for n in names]),
                np.array([target_tpl[n] for n in names]))
    if missing_rule != "mean_rank":
        raise ValueError(missing_rule)
    if not model_tpl or len(names) < 4:
        return None, None
    fill = float(np.mean(list(model_tpl.values())))
    return (np.array([model_tpl.get(n, fill) for n in names]),
            np.array([target_tpl[n] for n in names]))


def sim_matrix(model, target, support, missing_rule):
    """2x2 Spearman: rows model forward/reverse, cols patient t_a/t_b."""
    M = np.full((2, 2), np.nan)
    for i, row in enumerate(("forward", "reverse")):
        for j, col in enumerate(("t_a", "t_b")):
            a, b = _aligned(model.get(row, {}), target[col], support, missing_rule)
            if a is None or np.ptp(a) == 0 or np.ptp(b) == 0:
                continue
            M[i, j] = float(spearmanr(a, b).correlation)
    return M


def assignment_invariant_S(M):
    """max over the two TA/TB assignments of the diagonal mean.

    NaN when no full assignment exists. Deliberately NOT a best-single-cell
    fallback: such a value would invite differencing a one-direction arm against
    a two-direction one, which spec 5.3 forbids.
    """
    opts = [0.5 * (M[i, j] + M[k, l])
            for (i, j), (k, l) in (((0, 0), (1, 1)), ((0, 1), (1, 0)))
            if np.isfinite(M[i, j]) and np.isfinite(M[k, l])]
    return float(max(opts)) if opts else float("nan")


def _directed_pair(model_tpl, target_tpl, support):
    """Pairwise concordance with a FIXED denominator: every pair of the frozen
    support counts, and a pair touching an unrecruited contact contributes 0."""
    pairs = list(itertools.combinations(support, 2))
    if not pairs:
        return float("nan")
    tot = sum(np.sign((model_tpl[a] - model_tpl[b]) * (target_tpl[a] - target_tpl[b]))
              for a, b in pairs
              if a in model_tpl and b in model_tpl and a in target_tpl and b in target_tpl)
    return float(tot / len(pairs))


def balanced_pair_score(model, target, support):
    """Bidirectional, assignment-invariant, fixed-denominator pair score.

    NaN unless both directions exist, for the same reason as
    assignment_invariant_S.
    """
    support = list(support)
    if model.get("n_dir", 0) < 2:
        return float("nan")
    opts = [0.5 * (_directed_pair(model["forward"], target[a_col], support)
                   + _directed_pair(model["reverse"], target[b_col], support))
            for a_col, b_col in (("t_a", "t_b"), ("t_b", "t_a"))]
    return float(max(opts))


def axis_only_templates(names, coords, center, u_axis):
    """A model with NO pathology field: contacts ordered by axial projection.

    Pure geometry already scores 0.696 against this patient's templates, so this
    is the reference every claim has to beat (spec 2.4 / 5.4).
    """
    u = np.asarray(u_axis, float)
    u = u / np.linalg.norm(u)
    proj = (np.asarray(coords, float) - np.asarray(center, float)[None, :]) @ u
    return {"forward": {n: float(p) for n, p in zip(names, proj)},
            "reverse": {n: float(-p) for n, p in zip(names, proj)},
            "n_dir": 2, "coverage_forward": 1.0, "coverage_reverse": 1.0,
            "coverage_union": 1.0, "mean_n_part": float(len(names))}


def adversarial_gain(model, target, support, missing_rule):
    """How much could this candidate gain by dropping its worst-matching contact?

    Reported, not asserted. Under mean-rank filling a badly ranked contact can be
    worth dropping, so the size of that incentive has to be visible rather than
    assumed away (third-review P0-4).
    """
    base = assignment_invariant_S(sim_matrix(model, target, support, missing_rule))
    best_gain, worst = 0.0, None
    for name in support:
        trimmed = dict(model)
        trimmed["forward"] = {k: v for k, v in model.get("forward", {}).items() if k != name}
        trimmed["reverse"] = {k: v for k, v in model.get("reverse", {}).items() if k != name}
        s = assignment_invariant_S(sim_matrix(trimmed, target, support, missing_rule))
        if np.isfinite(s) and np.isfinite(base) and s - base > best_gain:
            best_gain, worst = float(s - base), name
    return dict(base=float(base), gain=float(best_gain), worst_contact=worst)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_topic4_core_field_scoring.py -x -q`
Expected: 10 passed

- [ ] **Step 5: 加两条 integration 测试（回归对齐 + 门的正当性）**

```python
# append to tests/test_topic4_core_field_scoring.py
import glob, json, sys
from pathlib import Path

RUN = Path("results/topic4_sef_hfo/field_swap_subject_snn")


@pytest.mark.integration
def test_common_only_mode_reproduces_the_published_sim_matrix():
    """Our reimplementation must agree with the file carrying the published
    Figure 4 numbers, in the mode that file implements."""
    sys.path.insert(0, ".")
    from scripts.paper_figures.plot_fig_subject_snn_realvsmodel import (
        _model_templates, _real_templates, _sim_matrix)
    tags = sorted(glob.glob(str(RUN / "readout_epilepsiae_1146_paired_tsrc_highn_s*_20260721.json")))
    if not tags:
        pytest.skip("paired_tsrc_highn artifacts not present")
    tag = Path(tags[0]).stem.removeprefix("readout_")
    real = _real_templates("epilepsiae_1146", "narrow")
    ref_M, _ = _sim_matrix(_model_templates(tag), real, B=1, seed=0)
    ro = json.load(open(RUN / f"readout_{tag}.json"))
    support = sorted(set(real["t_a"]) | set(real["t_b"]))
    # The published file admits n_part >= 2*k_dir; match it here so the regression
    # compares SCORING, not the participation floor.
    ours = model_templates(ro["events"], support, part_min=2 * int(ro.get("k_dir", 2)))
    our_M = sim_matrix(ours, {"t_a": real["t_a"], "t_b": real["t_b"]}, support, "common_only")
    assert np.allclose(ref_M, our_M, atol=1e-9, equal_nan=True)


@pytest.mark.integration
def test_no_signed_event_falls_below_the_participation_floor():
    """Justifies deleting gate=4: endpoint_centroid_axis returns None below
    2*k_dir+1, so a post-hoc gate of 4 admits nothing."""
    below = 0
    for path in glob.glob(str(RUN / "readout_epilepsiae_1146_*.json")):
        d = json.load(open(path))
        if d.get("k_dir") != 2:
            continue
        below += sum(1 for e in d.get("events", [])
                     if e.get("sign") is not None and e.get("n_part", 0) < 5)
    assert below == 0
```

- [ ] **Step 6: 跑全部打分测试**

Run: `python -m pytest tests/test_topic4_core_field_scoring.py -q`
Expected: 12 passed（artifact 缺失时 skip 一条）

- [ ] **Step 7: 提交**

```bash
git add src/topic4_core_field_scoring.py tests/test_topic4_core_field_scoring.py
git commit -m "feat(topic4-core-field): frozen-support scoring, per-direction coverage

The published scorer computes Spearman on whatever contacts a candidate recruits,
so recruiting fewer and easier ones raises the score; support is now frozen.
Coverage is recorded per direction because a union hides a candidate whose two
directions each cover a different small patch. A one-direction model scores NaN
rather than a best single cell, so it cannot be differenced against a
two-direction one. How much a candidate could gain by dropping its worst contact
is measured and reported rather than assumed away."
```

---

### Task 5: 字典序候选键

**重读 spec：** §5.3

**Files:** Modify `src/topic4_core_field_scoring.py`；Test `tests/test_topic4_core_field_scoring.py`

**Interfaces:** Produces `candidate_key(n_dir, s_rank) -> tuple`（可直接 `sorted`，大者优）

- [ ] **Step 1: 写失败测试**

```python
# append to tests/test_topic4_core_field_scoring.py
from src.topic4_core_field_scoring import candidate_key


def test_two_directions_always_outrank_one_even_when_S_is_lower():
    """The counterexample that killed scalar grading: one direction matching a
    template perfectly scores 0.5; two directions whose best assignment is +1 and
    -1 score 0."""
    assert candidate_key(2, 0.0) > candidate_key(1, 0.5)


def test_within_a_tier_the_better_match_ranks_higher():
    assert candidate_key(2, 0.8) > candidate_key(2, 0.3)


def test_no_direction_ranks_last_and_tolerates_nan():
    assert candidate_key(0, float("nan")) < candidate_key(1, -0.9)
    assert candidate_key(0, float("nan")) == candidate_key(0, float("nan"))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_topic4_core_field_scoring.py -x -q -k candidate_key`
Expected: FAIL — `ImportError: cannot import name 'candidate_key'`

- [ ] **Step 3: 最小实现**

```python
# append to src/topic4_core_field_scoring.py
def candidate_key(n_dir, s_rank):
    """Lexicographic fitness key: (n_dir, S_rank), larger is better.

    CMA-ES consumes candidate ORDER, so the tiers separate without inventing a
    rate-loss weight. Never compare S_rank across tiers (spec 5.3).
    """
    s = float(s_rank)
    return (int(n_dir), s if np.isfinite(s) else -np.inf)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_topic4_core_field_scoring.py -q`
Expected: 15 passed

- [ ] **Step 5: 提交**

```bash
git add src/topic4_core_field_scoring.py tests/test_topic4_core_field_scoring.py
git commit -m "feat(topic4-core-field): rank candidates lexicographically by (n_dir, S)"
```

---

### Task 6: 网络缓存（完整配置哈希 + 原子写）

**重读 spec：** §6.2

**Files:** Create `src/topic4_core_field_runner.py`；Test `tests/test_topic4_core_field_cache.py`

**Interfaces:** Produces `CONNECTIVITY_FIELDS`；`TRACKED_MODULES`；`canonical_checksum(obj, drop=("checksum",))`；`provenance()`；`connectivity_config(p, theta_deg, ar)`；`cache_key(config)`；`get_network(p, theta_deg, ar, cache_dir) -> (net, NE, NI, from_cache)`；`atomic_write_json(obj, path)`（Task 10 会用）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_topic4_core_field_cache.py
import numpy as np
import pytest
from src.topic4_core_field_runner import (
    canonical_checksum, cache_key, connectivity_config,
)


class _P:
    L = 20.0; density = 100.0; f_E = 0.8; seed = 1; g = 3.6
    C_EE = 800; C_IE = 800; C_EI = 200; C_II = 200
    l_EE = 0.380; l_IE = 0.250; l_EI = 0.250; l_II = 0.250
    rho_EE = 0.6; rho_IE = 0.0; rho_EI = 0.0; rho_II = 0.0
    tau0 = 0.1; v_axon = 0.3; delay_dt = 0.1


def test_cache_key_is_stable_for_an_unchanged_config():
    assert cache_key(connectivity_config(_P(), -22.8, 2.0)) == \
           cache_key(connectivity_config(_P(), -22.8, 2.0))


@pytest.mark.parametrize("field,value", [
    ("L", 21.0), ("density", 120.0), ("f_E", 0.75), ("seed", 2),
    ("C_EE", 700), ("C_IE", 700), ("C_EI", 150), ("C_II", 150),
    ("l_EE", 0.40), ("l_IE", 0.26), ("l_EI", 0.26), ("l_II", 0.26),
    ("rho_EE", 0.5), ("rho_IE", 0.1), ("rho_EI", 0.1), ("rho_II", 0.1),
    ("tau0", 0.2), ("v_axon", 0.4), ("delay_dt", 0.2),
])
def test_perturbing_any_connectivity_field_changes_the_key(field, value):
    base = cache_key(connectivity_config(_P(), -22.8, 2.0))
    p = _P(); setattr(p, field, value)
    assert cache_key(connectivity_config(p, -22.8, 2.0)) != base, field


@pytest.mark.parametrize("theta,ar", [(-20.0, 2.0), (-22.8, 1.5)])
def test_theta_and_aspect_ratio_are_in_the_key(theta, ar):
    assert cache_key(connectivity_config(_P(), theta, ar)) != \
           cache_key(connectivity_config(_P(), -22.8, 2.0))


def test_canonical_checksum_ignores_the_checksum_field_itself():
    """P0-7: the config stores its own checksum, so verification must recompute
    from the config MINUS that field, not compare a string with itself."""
    cfg = {"a": 1, "b": [1, 2]}
    c = canonical_checksum(cfg)
    assert canonical_checksum({**cfg, "checksum": c}) == c


def test_canonical_checksum_detects_a_changed_field():
    assert canonical_checksum({"a": 1, "b": [1, 3]}) != canonical_checksum({"a": 1, "b": [1, 2]})
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_topic4_core_field_cache.py -x -q`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 最小实现**

```python
# src/topic4_core_field_runner.py
"""Simulation glue: network cache, provenance, and one arm run.

Calls the blessed engine and the existing read-out chain; changes neither.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import tempfile

import numpy as np

CONNECTIVITY_FIELDS = (
    "L", "density", "f_E", "seed", "g",
    "C_EE", "C_IE", "C_EI", "C_II",
    "l_EE", "l_IE", "l_EI", "l_II",
    "rho_EE", "rho_IE", "rho_EI", "rho_II",
    "tau0", "v_axon", "delay_dt",
)
TRACKED_MODULES = (
    "src/topic4_core_field.py",
    "src/topic4_core_field_scoring.py",
    "src/topic4_core_field_report.py",
    "src/topic4_core_field_runner.py",
)


def _git(*args, default="unknown"):
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return default


def canonical_checksum(obj, drop=("checksum",)):
    """SHA256 of the canonical JSON with `drop` fields removed.

    Verification must recompute from content, never compare a stored string with
    itself (third-review P0-7).
    """
    if isinstance(obj, dict):
        obj = {k: v for k, v in obj.items() if k not in drop}
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def provenance():
    """What code actually produced an artifact."""
    dirty = _git("status", "--porcelain", *TRACKED_MODULES, default="?")
    return dict(
        git_commit=_git("rev-parse", "HEAD"),
        tracked_modules_dirty=(bool(dirty.strip()) if dirty != "?" else None),
        module_sha256={m: hashlib.sha256(open(m, "rb").read()).hexdigest()
                       for m in TRACKED_MODULES if os.path.exists(m)},
        numpy_version=np.__version__,
    )


def connectivity_config(p, theta_deg, ar):
    """Every field that can change the connectivity graph. Keying on
    (seed, theta, L, density, AR) alone would silently hit a stale cache."""
    cfg = {f: getattr(p, f) for f in CONNECTIVITY_FIELDS}
    cfg["theta_EE_deg"] = float(theta_deg)
    cfg["AR"] = float(ar)
    cfg["numpy_version"] = np.__version__
    cfg["rng_bit_generator"] = "PCG64"
    cfg["git_commit"] = _git("rev-parse", "HEAD")
    return cfg


def cache_key(config):
    return canonical_checksum(config, drop=())


def get_network(p, theta_deg, ar, cache_dir):
    """Build or load the connectivity graph.

    Field-independent, so ONE build per (seed, theta) serves every arm. Written
    via a temp file plus atomic rename: Stage 1 parallelises over seeds precisely
    so two workers never race here, and the rename makes a partial file
    impossible even if that assumption is ever broken.
    """
    import sys
    eng = os.path.join("src", "snn_engine")
    for path in (eng, os.getcwd()):
        if path not in sys.path:
            sys.path.insert(0, path)
    from connectivity import place_neurons
    from connectivity_rot import build_connectivity_rot

    cfg = connectivity_config(p, theta_deg, ar)
    key = cache_key(cfg)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        return payload["net"], payload["NE"], payload["NI"], True

    rng = np.random.default_rng(p.seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng,
                                 theta_EE=np.deg2rad(theta_deg), AR=ar, verbose=False)
    fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
    with os.fdopen(fd, "wb") as fh:
        pickle.dump({"net": net, "NE": NE, "NI": NI, "config": cfg},
                    fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return net, NE, NI, False


def atomic_write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_topic4_core_field_cache.py -q`
Expected: 25 passed

- [ ] **Step 5: 加 bit-parity 测试**

```python
# append to tests/test_topic4_core_field_cache.py
import sys


@pytest.mark.integration
@pytest.mark.slow
def test_cache_hit_reproduces_the_built_network_bitwise(tmp_path):
    sys.path.insert(0, "src/snn_engine"); sys.path.insert(0, ".")
    from params import Params
    from src.topic4_core_field_runner import get_network
    p = Params(g=3.6, L=6.0, density=40.0, T=100.0, dt=0.1, nu_ext_ratio=1.0, seed=11)
    a, NE_a, NI_a, hit_a = get_network(p, -22.8, 2.0, str(tmp_path))
    b, NE_b, NI_b, hit_b = get_network(p, -22.8, 2.0, str(tmp_path))
    assert hit_a is False and hit_b is True
    assert (NE_a, NI_a) == (NE_b, NI_b)
    assert np.array_equal(a["pos"], b["pos"]) and np.array_equal(a["labels"], b["labels"])
    assert a["max_delay_steps"] == b["max_delay_steps"]
    for key in ("ampa_by_delay", "gaba_by_delay"):
        assert len(a[key]) == len(b[key])
        for Wa, Wb in zip(a[key], b[key]):
            assert np.array_equal(Wa.toarray(), Wb.toarray())
    assert not [f for f in tmp_path.iterdir() if f.suffix == ".tmp"]
```

- [ ] **Step 6: 跑全部缓存测试**

Run: `python -m pytest tests/test_topic4_core_field_cache.py -q`
Expected: 26 passed

- [ ] **Step 7: 提交**

```bash
git add src/topic4_core_field_runner.py tests/test_topic4_core_field_cache.py
git commit -m "feat(topic4-core-field): cache networks under a full connectivity hash

The graph does not depend on the pathology field, so one build serves every arm
at a seed. Keying on seed and theta alone would return a stale network after any
other connectivity parameter moved, and the checksum is recomputed from content
rather than compared with the copy stored beside it."
```

---

### Task 7: common random numbers 回归锁（比较 OU 轨迹）

**重读 spec：** §6.3a

**Files:** Test `tests/test_topic4_core_field_crn.py`

已核验引擎主循环内只有 `rng.standard_normal()`（每步 1 个）与 `rng.poisson(..., size=N)`（每步 N 个）两处无条件定长调用。**调用次数相同不够 —— 必须比较实际噪声轨迹**（第三轮末条）。

- [ ] **Step 1: 写测试**

```python
# tests/test_topic4_core_field_crn.py
"""Locks the premise Stage 1's paired design rests on.

If anyone introduces a spike-dependent RNG call, changing the threshold field
would desynchronise the noise between arms and the paired probe would stop
meaning anything. Equal call counts would still permit a divergent stream, so the
driving normals themselves are recorded and compared.
"""
import sys
import numpy as np
import pytest

sys.path.insert(0, "src/snn_engine")
sys.path.insert(0, ".")


class RecordingGenerator(np.random.Generator):
    """Generator attributes are read-only, so recording needs a subclass."""

    def __init__(self, bit_generator):
        super().__init__(bit_generator)
        self.normals = []
        self.poisson_calls = 0

    def standard_normal(self, *args, **kwargs):
        v = super().standard_normal(*args, **kwargs)
        self.normals.append(float(v) if np.ndim(v) == 0 else float(np.asarray(v).sum()))
        return v

    def poisson(self, *args, **kwargs):
        self.poisson_calls += 1
        return super().poisson(*args, **kwargs)


@pytest.mark.integration
@pytest.mark.slow
def test_changing_the_threshold_field_leaves_the_noise_trajectory_identical(tmp_path):
    from params import Params
    from kick_probe import simulate_kick
    from src.topic4_core_field_runner import get_network

    p = Params(g=3.6, L=6.0, density=40.0, T=200.0, dt=0.1, nu_ext_ratio=1.0, seed=5)
    net, NE, NI, _ = get_network(p, -22.8, 2.0, str(tmp_path))
    N = NE + NI

    def run(vth):
        gen = RecordingGenerator(np.random.PCG64(5))
        net["rng"] = gen
        simulate_kick(p, net, KICK_BOOST=0.0, kick_center=[3.0, 3.0], r_kick=1.0,
                      t_kick=1e9, V_th_per_neuron=vth)
        return gen

    flat = run(np.full(N, 18.0))
    lowered_vth = np.full(N, 18.0); lowered_vth[: NE // 4] = 16.0
    lowered = run(lowered_vth)

    assert flat.poisson_calls == lowered.poisson_calls
    assert len(flat.normals) == len(lowered.normals)
    assert np.array_equal(np.asarray(flat.normals), np.asarray(lowered.normals)), (
        "the OU driving noise diverged between threshold fields: common random "
        "numbers no longer hold and the Stage 1 paired probe is invalid"
    )


@pytest.mark.integration
@pytest.mark.slow
def test_two_arms_at_one_seed_start_from_identical_state(tmp_path):
    """No state is inherited between arms: rebuilding the rng at the same seed
    reproduces the run bit for bit."""
    from params import Params
    from kick_probe import simulate_kick
    from src.topic4_core_field_runner import get_network

    p = Params(g=3.6, L=6.0, density=40.0, T=200.0, dt=0.1, nu_ext_ratio=1.0, seed=5)
    net, NE, NI, _ = get_network(p, -22.8, 2.0, str(tmp_path))
    vth = np.full(NE + NI, 18.0)
    outs = []
    for _ in range(2):
        net["rng"] = np.random.default_rng(5)
        outs.append(simulate_kick(p, net, KICK_BOOST=0.0, kick_center=[3.0, 3.0],
                                  r_kick=1.0, t_kick=1e9,
                                  V_th_per_neuron=vth)["E_spk_bool"])
    assert np.array_equal(outs[0], outs[1])
```

- [ ] **Step 2: 跑测试**

Run: `python -m pytest tests/test_topic4_core_field_crn.py -q`
Expected: 2 passed（约 1–2 分钟）

- [ ] **Step 3: 提交**

```bash
git add tests/test_topic4_core_field_crn.py
git commit -m "test(topic4-core-field): lock the OU trajectory, not just RNG call counts"
```

---

### Task 8: Stage 1 报告器（描述性；仅完整性 fail-closed）

**重读 spec：** §7.4、§7.5

**Files:** Create `src/topic4_core_field_report.py`；Test `tests/test_topic4_core_field_report.py`

**Interfaces:**
- Produces: `SEEDS`、`SIM_ARMS`、`PROJECTED_ARMS`、`SCORE_KEYS`、`PRIMARY_KEY`、`COMPARISONS`、`COVERAGE_MARGIN`；`arm_value(runs, arm, seed, key, field)`；`_arm_n_dir(runs, arm, seed, key)`；`tiered_paired_stats(pairs) -> dict`；`concordance(runs, key) -> float`；`stage1_report(runs, config) -> dict`

`runs` 键：`(arm, seed, source, missing_rule, score_def)`；`score_def ∈ {"spearman","pair"}`。**无 gate 维度。**

- [ ] **Step 1: 写失败测试**

```python
# tests/test_topic4_core_field_report.py
import numpy as np
import pytest
from src.topic4_core_field_report import (
    COMPARISONS, SCORE_KEYS, SEEDS, SIM_ARMS, concordance, stage1_report,
    tiered_paired_stats,
)

CFG = {"checksum": "abc123", "delta_eq": 0.05}


def _runs(separable=True, n_dir=2, cov=0.9):
    out = {}
    for arm in SIM_ARMS:
        for seed in SEEDS:
            base = 0.80 if arm.startswith("manual") else 0.60
            if not separable:
                base = 0.70
            jitter = 0.001 * ((seed * 7 + SIM_ARMS.index(arm)) % 3)
            for key in SCORE_KEYS:
                out[(arm, seed) + key] = dict(
                    n_dir=n_dir, S_rank=base + jitter,
                    coverage_forward=cov, coverage_reverse=cov)
    return out


def test_there_is_no_event_gate_dimension():
    """gate=4 admits zero signed events, so it is not an axis."""
    assert all(len(k) == 3 for k in SCORE_KEYS)
    assert {k[2] for k in SCORE_KEYS} == {"spearman", "pair"}


def test_comparisons_use_manual_smooth_as_the_shape_baseline():
    shape = {c["name"]: c for c in COMPARISONS if c["group"] == "shape"}
    assert set(shape) == {"B1", "B2", "B3", "B4"}
    assert all(c["a"] == "manual_smooth" for c in shape.values())
    assert {c["name"] for c in COMPARISONS if c["group"] == "equivalence"} == {"A", "A2"}


def test_s_rank_is_never_differenced_across_direction_tiers():
    """spec 5.3: only same-tier seeds contribute a numeric difference; the rest
    are counted as direction-tier wins and losses."""
    st = tiered_paired_stats([(2, 0.8, 2, 0.5), (2, 0.9, 1, float("nan")),
                              (0, float("nan"), 2, 0.4)])
    assert st["n_same_tier"] == 1
    assert st["tier_wins"] == 1
    assert st["tier_losses"] == 1
    assert st["mean"] == pytest.approx(0.3)


def test_tiered_stats_report_a_confidence_interval():
    st = tiered_paired_stats([(2, 0.6 + 0.01 * i, 2, 0.5) for i in range(12)])
    assert st["ci_low"] < st["mean"] < st["ci_high"]


def test_report_is_a_pure_function():
    runs = _runs()
    snapshot = {k: dict(v) for k, v in runs.items()}
    stage1_report(runs, CFG)
    assert runs == snapshot


def test_integrity_fails_closed_on_a_missing_cell():
    runs = _runs(); del runs[(SIM_ARMS[0], SEEDS[0]) + SCORE_KEYS[0]]
    assert stage1_report(runs, CFG)["integrity"]["status"] == "FAIL_CLOSED"


def test_integrity_fails_closed_on_a_nan_with_two_directions():
    runs = _runs()
    runs[(SIM_ARMS[0], SEEDS[0]) + SCORE_KEYS[0]]["S_rank"] = float("nan")
    assert stage1_report(runs, CFG)["integrity"]["status"] == "FAIL_CLOSED"


def test_a_nan_with_no_directions_is_legitimate():
    runs = _runs()
    cell = runs[(SIM_ARMS[0], SEEDS[0]) + SCORE_KEYS[0]]
    cell["n_dir"], cell["S_rank"] = 0, float("nan")
    assert stage1_report(runs, CFG)["integrity"]["status"] == "ok"


def test_separable_and_flat_arms_give_different_recommendations():
    sep = stage1_report(_runs(separable=True), CFG)
    flat = stage1_report(_runs(separable=False), CFG)
    assert sep["recommendation"]["shape_separates"] is True
    assert flat["recommendation"]["shape_separates"] is False
    for r in (sep, flat):
        assert r["integrity"]["status"] == "ok"
        assert "verdict" not in r          # exploratory: no automatic gate


def test_low_coverage_is_flagged_but_does_not_stop_anything():
    runs = _runs()
    for seed in SEEDS:
        for key in SCORE_KEYS:
            runs[("uniform_axial", seed) + key]["coverage_forward"] = 0.2
    rep = stage1_report(runs, CFG)
    assert "uniform_axial" in rep["coverage"]["low_coverage_arms"]
    assert rep["integrity"]["status"] == "ok"


def test_transverse_sign_flip_does_not_change_the_report():
    runs = _runs(separable=True)
    swap = {"transverse_plus": "transverse_minus", "transverse_minus": "transverse_plus"}
    flipped = {(swap.get(k[0], k[0]),) + k[1:]: dict(v) for k, v in runs.items()}
    assert stage1_report(flipped, CFG)["recommendation"] == \
           stage1_report(runs, CFG)["recommendation"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_topic4_core_field_report.py -x -q`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 最小实现**

```python
# src/topic4_core_field_report.py
"""Stage 1 report (spec 7.5).

Exploratory posture: this produces NUMBERS and a recommendation, not an automatic
gate. Only data integrity fails closed -- that is a correctness guard, not a
science gate.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import binomtest

SEEDS = tuple(range(1, 13))
SIM_ARMS = ("manual_hard", "manual_projected", "manual_smooth", "uniform_axial",
            "width_wide", "width_narrow", "transverse_plus", "transverse_minus")
PROJECTED_ARMS = ("manual_smooth", "uniform_axial", "width_wide",
                  "width_narrow", "transverse_mean")
# No event-gate axis: gate=4 admits zero signed events.
SCORE_KEYS = tuple(itertools.product(("gradient", "geometry"),
                                     ("mean_rank", "common_only"),
                                     ("spearman", "pair")))
PRIMARY_KEY = ("gradient", "mean_rank", "spearman")

COMPARISONS = (
    dict(name="A", a="manual_projected", b="manual_hard", group="equivalence",
         purpose="sampling contract"),
    dict(name="A2", a="manual_smooth", b="manual_projected", group="equivalence",
         purpose="hard mask vs smoothed field"),
    dict(name="B1", a="manual_smooth", b="uniform_axial", group="shape",
         purpose="longitudinal shape"),
    dict(name="B2", a="manual_smooth", b="width_wide", group="shape",
         purpose="transverse width, flattened"),
    dict(name="B3", a="manual_smooth", b="width_narrow", group="shape",
         purpose="transverse width, elongated"),
    dict(name="B4", a="manual_smooth", b="transverse_mean", group="shape",
         purpose="transverse position"),
    dict(name="C", a="manual_smooth", b="axis_only", group="geometry",
         purpose="pure-geometry reference"),
)
COVERAGE_MARGIN = 0.10


def arm_value(runs, arm, seed, key, field):
    if arm == "transverse_mean":
        return float(np.mean([runs[("transverse_plus", seed) + key][field],
                              runs[("transverse_minus", seed) + key][field]]))
    return runs[(arm, seed) + key][field]


def _arm_n_dir(runs, arm, seed, key):
    if arm == "transverse_mean":
        return min(runs[("transverse_plus", seed) + key]["n_dir"],
                   runs[("transverse_minus", seed) + key]["n_dir"])
    return runs[(arm, seed) + key]["n_dir"]


def tiered_paired_stats(pairs):
    """pairs: iterable of (n_dir_a, S_a, n_dir_b, S_b).

    Only seeds where both arms sit in the SAME direction tier contribute a
    numeric difference. Seeds where the tiers differ are counted as wins and
    losses: differencing across tiers is what spec 5.3 forbids.
    """
    deltas, wins, losses = [], 0, 0
    for nda, sa, ndb, sb in pairs:
        if nda == ndb:
            if np.isfinite(sa) and np.isfinite(sb):
                deltas.append(float(sa) - float(sb))
        elif nda > ndb:
            wins += 1
        else:
            losses += 1
    d = np.asarray(deltas, float)
    out = dict(n_same_tier=int(d.size), tier_wins=int(wins), tier_losses=int(losses),
               mean=float(d.mean()) if d.size else float("nan"),
               sd=float(d.std(ddof=1)) if d.size > 1 else float("nan"))
    if d.size > 1:
        se = out["sd"] / np.sqrt(d.size)
        out["ci_low"], out["ci_high"] = out["mean"] - 1.96 * se, out["mean"] + 1.96 * se
    else:
        out["ci_low"] = out["ci_high"] = float("nan")
    nz = d[d != 0.0]
    if nz.size:
        sign = np.sign(out["mean"]) if out["mean"] != 0 else 1.0
        n_same = int((np.sign(nz) == sign).sum())
        out["n_same"] = n_same
        out["p_uncorrected"] = float(
            binomtest(n_same, int(nz.size), 0.5, alternative="two-sided").pvalue)
    else:
        out["n_same"], out["p_uncorrected"] = 0, float("nan")
    return out


def concordance(runs, key):
    """Cross-seed ordering consistency among the projected arms. Diagnostic only:
    how well a single seed would order candidates for CMA-ES."""
    hits = []
    for a, b in itertools.combinations(PROJECTED_ARMS, 2):
        deltas = {}
        for s in SEEDS:
            if _arm_n_dir(runs, a, s, key) != _arm_n_dir(runs, b, s, key):
                continue
            va, vb = arm_value(runs, a, s, key, "S_rank"), arm_value(runs, b, s, key, "S_rank")
            if np.isfinite(va) and np.isfinite(vb):
                deltas[s] = va - vb
        if len(deltas) < 2:
            continue
        pooled = np.sign(np.mean(list(deltas.values()))) or 1.0
        hits.extend(1.0 if (np.sign(v) or 1.0) == pooled else 0.0 for v in deltas.values())
    return float(np.mean(hits)) if hits else float("nan")


def stage1_report(runs, config):
    # --- integrity: the ONLY fail-closed path -----------------------------
    for arm in SIM_ARMS:
        for seed in SEEDS:
            for key in SCORE_KEYS:
                cell = runs.get((arm, seed) + key)
                if cell is None:
                    return dict(integrity=dict(status="FAIL_CLOSED",
                                               reason=f"missing cell {(arm, seed) + key}"))
                if cell["n_dir"] >= 2 and not np.isfinite(cell["S_rank"]):
                    return dict(integrity=dict(
                        status="FAIL_CLOSED",
                        reason=f"non-finite S_rank with n_dir=2 at {(arm, seed) + key}"))
    integrity = dict(status="ok", checksum=config.get("checksum"))

    uninformative = [s for s in SEEDS
                     if sum(1 for a in SIM_ARMS
                            if runs[(a, s) + PRIMARY_KEY]["n_dir"] == 0) >= 2]

    comparisons = {}
    for key in SCORE_KEYS:
        for comp in COMPARISONS:
            if comp["b"] == "axis_only":
                continue                       # filled by the analysis script
            pairs = [(_arm_n_dir(runs, comp["a"], s, key),
                      arm_value(runs, comp["a"], s, key, "S_rank"),
                      _arm_n_dir(runs, comp["b"], s, key),
                      arm_value(runs, comp["b"], s, key, "S_rank")) for s in SEEDS]
            comparisons[(comp["name"],) + key] = dict(
                group=comp["group"], purpose=comp["purpose"], **tiered_paired_stats(pairs))

    delta_eq = float(config.get("delta_eq", 0.05))
    equivalence = {}
    for name in ("A", "A2"):
        st = comparisons[(name,) + PRIMARY_KEY]
        inside = (np.isfinite(st["ci_low"]) and np.isfinite(st["ci_high"])
                  and st["ci_low"] > -delta_eq and st["ci_high"] < delta_eq)
        equivalence[name] = dict(delta_eq=delta_eq, equivalent=bool(inside), **st)

    shape = {n: comparisons[(n,) + PRIMARY_KEY] for n in ("B1", "B2", "B3", "B4")}
    separates = [n for n, st in shape.items()
                 if np.isfinite(st["p_uncorrected"]) and st["p_uncorrected"] < 0.05]

    cov, low = {}, []
    ref = float(np.mean([min(runs[("manual_smooth", s) + PRIMARY_KEY]["coverage_forward"],
                             runs[("manual_smooth", s) + PRIMARY_KEY]["coverage_reverse"])
                         for s in SEEDS]))
    for arm in SIM_ARMS:
        per_dir = float(np.mean([min(runs[(arm, s) + PRIMARY_KEY]["coverage_forward"],
                                     runs[(arm, s) + PRIMARY_KEY]["coverage_reverse"])
                                 for s in SEEDS]))
        cov[arm] = per_dir
        if per_dir < ref - COVERAGE_MARGIN:
            low.append(arm)

    return dict(
        integrity=integrity,
        scorable=dict(uninformative_seeds=uninformative, n_seeds=len(SEEDS)),
        equivalence=equivalence,
        shape=shape,
        comparisons={"|".join(map(str, k)): v for k, v in comparisons.items()},
        concordance={"|".join(k): concordance(runs, k) for k in SCORE_KEYS},
        coverage=dict(per_arm=cov, reference=ref, low_coverage_arms=low,
                      margin=COVERAGE_MARGIN),
        sensitivities_reported_separately=dict(
            TEMPLATE_SOURCE=sorted({k[0] for k in SCORE_KEYS}),
            SCORER_SENSITIVITY=sorted({k[1] for k in SCORE_KEYS}),
            SCORE_DEFINITION=sorted({k[2] for k in SCORE_KEYS}),
        ),
        recommendation=dict(
            shape_separates=bool(separates),
            separating_dimensions=separates,
            note=("uncorrected p across 4 shape comparisons; exploratory posture, "
                  "no multiplicity correction and no automatic gate"),
        ),
    )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_topic4_core_field_report.py -q`
Expected: 11 passed

- [ ] **Step 5: 提交**

```bash
git add src/topic4_core_field_report.py tests/test_topic4_core_field_report.py
git commit -m "feat(topic4-core-field): descriptive Stage 1 report, integrity-only fail-closed

The user has ruled this stage exploratory, so the nine-exit gate becomes a report
a human reads. S_rank is differenced only within a direction tier, with mixed-tier
seeds counted as wins and losses; template source, scorer and score definition are
reported as separate sensitivities rather than collapsed into one exit; and low
coverage is flagged without stopping anything."
```

---

### Task 9: Stage 0 —— parity 复现、三参照、冻结 config

**重读 spec：** §6.1、§2.4

**Files:** Create `scripts/run_topic4_core_field_stage0.py`

**Interfaces:** Produces `config/stage_config.json`（含 `support` / `quantile_seed` / `N_core_manual` / `D0` / `part_min` / `delta_eq` / `checksum`）、`model_integrity_report.md`、`reference_scores.csv`、`parity_seed5.json`

- [ ] **Step 1: 写脚本**

```python
# scripts/run_topic4_core_field_stage0.py
"""Stage 0: re-run the seed-5 baseline for parity, rescore the three references
with ONE frozen scorer, and freeze the config later stages are checked against.

The three references live in different regimes and are not interchangeable:
  axis_only          -- no pathology field at all
  manual spontaneous -- the two-core network running free; THIS is the baseline
  driven_pooled      -- source-only + sink-only pooled; a READ-OUT UPPER REFERENCE,
                        which its own frozen stats file states in as many words
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from src.topic4_core_field import (  # noqa: E402
    core_thresholds, manual_mask, sample_core_quantiles, signed_depth)
from src.topic4_core_field_runner import canonical_checksum, provenance  # noqa: E402
from src.topic4_core_field_scoring import (  # noqa: E402
    PART_MIN, adversarial_gain, assignment_invariant_S, axis_only_templates,
    balanced_pair_score, load_patient_templates, model_templates, sim_matrix)

OUT = "results/topic4_sef_hfo/data_driven_core_field"
RUN = "results/topic4_sef_hfo/field_swap_subject_snn"
SUBJECT = "epilepsiae_1146"
PARITY_TAG = f"{SUBJECT}_gradient_shared_corefrozen_cr1p5_s5_20260722"
QUANTILE_SEED = 20260806
SOURCES = ("gradient", "geometry")
MISSING_RULES = ("mean_rank", "common_only")


def _score_row(model, targets, support):
    row = {}
    for src in SOURCES:
        for rule in MISSING_RULES:
            row[f"S_{src}_{rule}"] = assignment_invariant_S(
                sim_matrix(model, targets[src], support, rule))
        row[f"pair_{src}"] = balanced_pair_score(model, targets[src], support)
        row[f"advgain_{src}"] = adversarial_gain(
            model, targets[src], support, "mean_rank")["gain"]
    row.update(n_dir=model["n_dir"],
               coverage_forward=model["coverage_forward"],
               coverage_reverse=model["coverage_reverse"])
    return row


def _parity(out_dir):
    """Re-run gradient_shared seed 5 with the CURRENT code and compare to the
    frozen artifact field by field (spec 6.1 step 1)."""
    spec = importlib.util.spec_from_file_location(
        "subrun", os.path.join("scripts", "run_sef_hfo_subject_snn.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fresh, _, _ = mod.subject_run(SUBJECT, "narrow", "twoend_equal", 20.0, 100.0,
                                  mod.cmrun.DRIVE, 8000.0, 17.5, 1.0, 1.5, 5, None, 2,
                                  "gradient_shared", 3, None, None)
    frozen = json.load(open(os.path.join(RUN, f"readout_{PARITY_TAG}.json")))
    fields = ("n_events", "n_directional", "dir_forward", "dir_reverse", "n_clean",
              "clean_forward", "clean_reverse", "valid_contacts", "n_contacts",
              "theta_deg", "inter_core_sheet")
    diffs = {f: dict(frozen=frozen.get(f), fresh=fresh.get(f))
             for f in fields if frozen.get(f) != fresh.get(f)}
    result = dict(tag=PARITY_TAG, identical=not diffs, differing_fields=diffs,
                  provenance=provenance())
    json.dump(result, open(os.path.join(out_dir, "parity_seed5.json"), "w"), indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--skip-parity", action="store_true",
                    help="only for iterating on the scoring path; never for a real Stage 0")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "config"), exist_ok=True)

    parity = dict(skipped=True)
    if not a.skip_parity:
        print("[stage0] re-running gradient_shared seed 5 for parity (~8 min) ...")
        parity = _parity(a.out)
        print(f"[stage0] parity identical={parity['identical']} "
              f"differing={list(parity['differing_fields'])}")

    targets = {s: load_patient_templates(SUBJECT, s) for s in SOURCES}
    support = sorted(set(targets["gradient"]["t_a"]) & set(targets["gradient"]["t_b"])
                     & set(targets["geometry"]["t_a"]) & set(targets["geometry"]["t_b"]))
    print(f"[stage0] frozen scoring support: {len(support)} contacts -> {support}")

    rows = []
    fd = np.load(os.path.join(RUN, f"figdata_{PARITY_TAG}.npz"), allow_pickle=True)
    reg = fd["reg"].item()
    ao = axis_only_templates([str(x) for x in fd["names"]],
                             np.asarray(fd["contacts"], float),
                             np.asarray(reg["center"]), np.asarray(reg["axis_unit"]))
    rows.append(dict(reference="axis_only", tag="-", **_score_row(ao, targets, support)))

    for path in sorted(glob.glob(os.path.join(RUN, "readout_*.json"))):
        ro = json.load(open(path))
        if ro.get("subject") != SUBJECT:
            continue
        lesion, placement = ro.get("lesion"), ro.get("placement")
        if lesion == "twoend_equal" and placement in ("gradient_shared", "template_source"):
            ref = f"spontaneous_two_core_{placement}"
        elif lesion == "driven_pooled":
            ref = "driven_pooled_upper_reference"
        else:
            continue
        m = model_templates(ro["events"], support, part_min=PART_MIN)
        rows.append(dict(reference=ref, tag=os.path.basename(path),
                         **_score_row(m, targets, support)))

    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(a.out, "reference_scores.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"[stage0] wrote reference_scores.csv ({len(rows)} rows)")

    # --- pathology budget from the ACTUAL sheet geometry ------------------
    sys.path.insert(0, os.path.join("src", "snn_engine"))
    from connectivity import place_neurons
    from params import Params
    from src.sef_hfo_subject_placement import (
        gradient_shared_template_foci, register_to_sheet, template_source_foci)
    m_real, _, _, _ = gradient_shared_template_foci(SUBJECT, 3)
    _, src_n, snk_n = template_source_foci(SUBJECT, "narrow", 3)
    regd = register_to_sheet(m_real, src_n, snk_n, L=20.0, target_inter_core_mm=None)
    p = Params(g=3.6, L=20.0, density=100.0, T=100.0, dt=0.1, seed=1)
    pos, _, NE, _ = place_neurons(p, np.random.default_rng(1))
    mask = manual_mask(pos[:NE], regd["source_centroid"], regd["sink_centroid"], 1.5)
    n_core = int(mask.sum())
    d = signed_depth(core_thresholds(sample_core_quantiles(NE, QUANTILE_SEED)))
    D0 = float((mask.astype(float) * d).sum())
    print(f"[stage0] N_core_manual = {n_core}   D0 = {D0:.2f} mV")

    cfg = dict(
        subject=SUBJECT, support=support, quantile_seed=QUANTILE_SEED,
        N_core_manual=n_core, D0=D0, part_min=PART_MIN, delta_eq=0.05,
        sources=list(SOURCES), missing_rules=list(MISSING_RULES),
        score_defs=["spearman", "pair"],
        seeds=list(range(1, 13)), duration_ms=8000.0,
        field=dict(M=9, EPS=1e-3, TAU_H=0.25, A0=1.5, B0=1.5,
                   SIGMA_S_FACTOR=1.2, AXIAL_MARGIN=2.0, SHIFT_MM=3.0),
        engine=dict(L=20.0, density=100.0, AR=2.0, g=3.6, dt=0.1, k_dir=2,
                    core_mean=17.5, core_std=1.0, core_r=1.5, v_base=18.0),
        provenance=provenance(), parity=parity,
    )
    cfg["checksum"] = canonical_checksum(cfg)
    json.dump(cfg, open(os.path.join(a.out, "config", "stage_config.json"), "w"), indent=2)
    print(f"[stage0] froze config checksum={cfg['checksum'][:12]}")

    def summarise(prefix):
        v = [r["S_gradient_mean_rank"] for r in rows
             if r["reference"].startswith(prefix) and r["n_dir"] == 2
             and np.isfinite(r["S_gradient_mean_rank"])]
        if not v:
            return "n/a"
        return f"{np.mean(v):.3f} +/- {np.std(v, ddof=1) if len(v) > 1 else 0:.3f} (n={len(v)})"

    open(os.path.join(a.out, "model_integrity_report.md"), "w").write(f"""# Stage 0 完整性报告 — {SUBJECT}

## 网络规模（从代码读，不从手稿读）
`N = round(density * L^2) = 100 * 20 * 20 = 40000`，`N_E = 32000`、`N_I = 8000`。
手稿若写 `N = 4000`，是手稿错。

## seed 5 parity 复现
`identical = {parity.get('identical')}`；差异字段：`{list(parity.get('differing_fields', {}))}`
（详见 `parity_seed5.json`）

## 冻结的打分支撑集
{len(support)} 个触点：{support}

## 病理预算
`N_core_manual = {n_core}`，`D0 = {D0:.2f}` mV（有符号，约 31% 的 d_i 为负）

## 三个参照（同一冻结 scorer，`S_gradient_mean_rank`，事件门 n_part>=5）
| 参照 | 是什么 | 分数 |
|---|---|---|
| `axis_only` | **完全没有病理场**，只按触点在 u_C 上的投影排序 | {rows[0]['S_gradient_mean_rank']:.3f} |
| 自发双核 gradient_shared | **这才是基线** | {summarise('spontaneous_two_core_gradient_shared')} |
| 自发双核 template_source（旧几何） | 参考 | {summarise('spontaneous_two_core_template_source')} |
| `driven_pooled` | **读出上参照，不是基线** | {summarise('driven_pooled')} |

`driven_pooled` 的冻结统计文件逐字写着
`"independent_unit": "paired network seed (source-only and sink-only arms)"`。

## 事件门
只有 `n_part >= 5`。`gate = 4` 不存在：`endpoint_centroid_axis` 在低于 `2*k_dir+1` 时返回 `None`，
实测 signed 事件 `n_part` 最小值就是 5。

## config
`config/stage_config.json`，checksum `{cfg['checksum']}`
""")
    print("[stage0] wrote model_integrity_report.md")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 先提交脚本（运行之前）**

```bash
git add scripts/run_topic4_core_field_stage0.py
git commit -m "feat(topic4-core-field): Stage 0 parity run, three references, frozen config"
```

- [ ] **Step 3: 断言工作树干净**

Run: `git status --porcelain src/topic4_core_field*.py scripts/run_topic4_core_field_stage0.py`
Expected: 无输出。有输出就先提交再跑 —— 否则产物来自未提交的代码。

- [ ] **Step 4: 跑 Stage 0（约 10 分钟，含 parity 重跑）**

Run: `python scripts/run_topic4_core_field_stage0.py`
Expected: 打印 parity 结果、支撑集（预期 15 个触点）、`N_core_manual`、`D0`、checksum

- [ ] **Step 5: 目视核对报告 —— 人工闸门**

Run: `cat results/topic4_sef_hfo/data_driven_core_field/model_integrity_report.md`

检查三件事，任一不对**停下来查，不要进 Stage 1**：
1. **parity `identical = True`**。False 说明当前代码与冻结 artifact 的产出不一致。
2. **`axis_only` ≈ 0.696**。偏离说明冻结 scorer 与 spec §2.4 的算法有出入。
3. **自发双核应当与 `axis_only` 接近**（这正是 Stage 1 存在的理由）。远高于同样要查。

- [ ] **Step 6: 提交产物**

```bash
git add results/topic4_sef_hfo/data_driven_core_field/config/stage_config.json \
        results/topic4_sef_hfo/data_driven_core_field/model_integrity_report.md \
        results/topic4_sef_hfo/data_driven_core_field/reference_scores.csv \
        results/topic4_sef_hfo/data_driven_core_field/parity_seed5.json
git commit -m "chore(topic4-core-field): Stage 0 artifacts

Separates the spontaneous two-core baseline from the driven-pooled read-out upper
reference, which an earlier draft had conflated."
```

---

### Task 10: Stage 1 —— 96 次配对探针（按网络种子并行）

**重读 spec：** §7.2、§4.4

**Files:** Create `scripts/run_topic4_core_field_stage1.py`；Modify `src/topic4_core_field_runner.py`

**Interfaces:** Produces `_placement(cfg)`；`run_arm_on_network(arm, seed, cfg, net, NE, NI, reg, cmrun)`；产物 `stage1_variance_probe/per_run/<seed>/<arm>.json`（原子写）

**并行单位是网络种子**（第三轮 P0-8）：一个 worker 建/载入一张网，顺序跑 8 个臂。

- [ ] **Step 1: 在 runner 里加运行函数**

```python
# append to src/topic4_core_field_runner.py
def _placement(cfg):
    """Frozen shared-plane montage, core centroids and axis. Never refits."""
    from src.sef_hfo_subject_placement import (
        gradient_shared_template_foci, register_to_sheet, template_source_foci)
    m_real, _, _, _ = gradient_shared_template_foci(cfg["subject"], 3)
    _, src_names, snk_names = template_source_foci(cfg["subject"], "narrow", 3)
    reg = register_to_sheet(m_real, src_names, snk_names,
                            L=cfg["engine"]["L"], target_inter_core_mm=None)
    axis = reg["sink_centroid"] - reg["source_centroid"]
    reg["axis_unit_vec"] = axis / np.linalg.norm(axis)
    return reg


def run_arm_on_network(arm, seed, cfg, net, NE, NI, reg, cmrun):
    """One arm on an ALREADY-BUILT network. The caller owns the network so the
    eight arms at a seed share one build (third-review P0-8)."""
    from kick_probe import simulate_kick
    from lfp import LFPRecorder
    from params import Params
    from src.sef_hfo_events import detect_events
    from src.sef_hfo_heterogeneity import sample_core_field
    from src.sef_hfo_snn_adapter import snn_event_envelope
    from src.topic4_core_field import (
        arm_h, axis_coords, build_vth, core_thresholds, manual_mask,
        sample_core_quantiles, signed_depth)

    e = cfg["engine"]
    msheet = reg["montage_sheet"]
    src_xy, snk_xy = reg["source_centroid"], reg["sink_centroid"]
    axis_unit = reg["axis_unit_vec"]
    posE = net["pos"][:NE]
    is_E = np.zeros(len(net["pos"]), bool); is_E[:NE] = True
    mask = manual_mask(posE, src_xy, snk_xy, e["core_r"])

    if arm == "manual_hard":
        cf1 = sample_core_field(net["pos"], is_E, src_xy, e["core_r"],
                                np.random.default_rng(seed + 7), core_mean=e["core_mean"],
                                core_std=e["core_std"], base_mean=e["v_base"])
        cf2 = sample_core_field(net["pos"], is_E, snk_xy, e["core_r"],
                                np.random.default_rng(seed + 8), core_mean=e["core_mean"],
                                core_std=e["core_std"], base_mean=e["v_base"])
        vth = np.minimum(cf1["vth"], cf2["vth"])
        h_sum = float(mask.sum())
    else:
        s, r = axis_coords(posE, reg["center"], axis_unit)
        geom = dict(sep=float(np.linalg.norm(snk_xy - src_xy)),
                    s_support=(float(s.min()) + cfg["field"]["AXIAL_MARGIN"],
                               float(s.max()) - cfg["field"]["AXIAL_MARGIN"]),
                    M=cfg["field"]["M"], sigma_perp=e["core_r"],
                    shift_mm=cfg["field"]["SHIFT_MM"])
        h = arm_h(arm, s, r, geom, float(cfg["N_core_manual"]), manual_mask_E=mask)
        d = signed_depth(core_thresholds(
            sample_core_quantiles(NE, cfg["quantile_seed"]), e["core_mean"], e["core_std"]),
            e["v_base"])
        vth = build_vth(h, d, n_total=NE + NI, n_E=NE, v_base=e["v_base"])
        h_sum = float(h.sum())

    p = Params(g=e["g"], L=e["L"], density=e["density"], T=cfg["duration_ms"],
               dt=e["dt"], nu_ext_ratio=cmrun.DRIVE, seed=seed)
    k_dir = int(e["k_dir"])
    valid = cmrun.valid_mask(msheet, posE, e["L"], p.Rr)
    rec = LFPRecorder(p, net["pos"], net["labels"], sites=msheet.contacts)
    net["rng"] = np.random.default_rng(seed)
    res = simulate_kick(p, net, KICK_BOOST=0.0, kick_center=list(reg["center"]),
                        r_kick=e["core_r"], t_kick=1e9, V_th_per_neuron=vth,
                        lfp_recorder=rec)
    spk = res["E_spk_bool"]

    af, bin_w = cmrun.active_fraction(spk, e["dt"], cmrun.BIN_MS)
    nb0, nb1 = int(cmrun.BASELINE_MS[0] / bin_w), int(cmrun.BASELINE_MS[1] / bin_w)
    floor = float(np.percentile(af[nb0:nb1], 95)) if nb1 > nb0 else float(af.min())
    bar = floor + cmrun.CAL_FRAC * (float(af.max()) - floor)
    events = detect_events(af, bin_w, event_on_frac=bar)
    env_f, fdt, _ = snn_event_envelope(spk, posE, msheet, e["dt"])

    recs = []
    for ev in events:
        rd = cmrun.read_event(env_f, fdt, msheet, valid, (ev["t_on"], ev["t_off"]),
                              axis_unit, k_dir=k_dir, part_min=2 * k_dir + 1)
        recs.append(dict(n_part=int(rd["n_part"]), sign=rd["sign"], ranks=rd["ranks"]))
    return dict(arm=arm, seed=int(seed), events=recs, n_events=len(recs),
                h_sum=h_sum, config_checksum=cfg["checksum"], provenance=provenance())
```

- [ ] **Step 2: 写 Stage 1 驱动脚本**

```python
# scripts/run_topic4_core_field_stage1.py
"""Stage 1: 8 arms x 12 seeds paired probe, parallelised over NETWORK SEEDS.

One worker owns a seed: it builds or loads that seed's network once and runs all
eight arms on it. Dispatching (arm, seed) instead would have eight workers miss
the cache together, build the same network eight times and overwrite one file.

Refuses to launch if a shape comparison is vacuous -- that check costs
milliseconds and the run costs an hour.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.getcwd())
from src.topic4_core_field import (  # noqa: E402
    ARM_NAMES, arm_h, axis_coords, manual_mask, preflight_shape)
from src.topic4_core_field_runner import (  # noqa: E402
    _placement, atomic_write_json, canonical_checksum, get_network,
    provenance, run_arm_on_network)

OUT = "results/topic4_sef_hfo/data_driven_core_field"


def _load_cmrun():
    spec = importlib.util.spec_from_file_location(
        "cmrun", os.path.join("scripts", "run_sef_hfo_snn_cm_spontaneous_readout.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def preflight(cfg):
    sys.path.insert(0, os.path.join("src", "snn_engine"))
    from connectivity import place_neurons
    from params import Params
    e = cfg["engine"]
    reg = _placement(cfg)
    p = Params(g=e["g"], L=e["L"], density=e["density"], T=100.0, dt=e["dt"], seed=1)
    pos, _, NE, _ = place_neurons(p, np.random.default_rng(1))
    posE = pos[:NE]
    s, r = axis_coords(posE, reg["center"], reg["axis_unit_vec"])
    geom = dict(sep=float(np.linalg.norm(reg["sink_centroid"] - reg["source_centroid"])),
                s_support=(float(s.min()) + cfg["field"]["AXIAL_MARGIN"],
                           float(s.max()) - cfg["field"]["AXIAL_MARGIN"]),
                M=cfg["field"]["M"], sigma_perp=e["core_r"],
                shift_mm=cfg["field"]["SHIFT_MM"])
    mask = manual_mask(posE, reg["source_centroid"], reg["sink_centroid"], e["core_r"])
    target = float(cfg["N_core_manual"])
    h_by_arm = {a: arm_h(a, s, r, geom, target, manual_mask_E=mask)
                for a in ARM_NAMES if a != "manual_hard"}
    h_by_arm["manual_hard"] = mask.astype(float)
    return preflight_shape(h_by_arm, s, r, target)


def _seed_job(args):
    seed, cfg, cache_dir, out_dir = args
    sys.path.insert(0, os.path.join("src", "snn_engine"))
    from params import Params
    try:
        cmrun = _load_cmrun()
        k_dir = int(cfg["engine"]["k_dir"])
        cmrun.KDIR, cmrun.PART_MIN = k_dir, 2 * k_dir + 1
        reg = _placement(cfg)
        e = cfg["engine"]
        p = Params(g=e["g"], L=e["L"], density=e["density"], T=cfg["duration_ms"],
                   dt=e["dt"], nu_ext_ratio=cmrun.DRIVE, seed=seed)
        net, NE, NI, hit = get_network(p, reg["theta_deg"], e["AR"], cache_dir)
        done = []
        for arm in ARM_NAMES:
            path = os.path.join(out_dir, str(seed), f"{arm}.json")
            if os.path.exists(path):
                done.append(f"{arm}:cached"); continue
            rec = run_arm_on_network(arm, seed, cfg, net, NE, NI, reg, cmrun)
            atomic_write_json(rec, path)
            done.append(f"{arm}:{rec['n_events']}")
        return dict(seed=seed, network_cache_hit=hit, arms=done)
    except Exception as exc:
        return dict(seed=seed, error=f"{type(exc).__name__}: {exc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8, help="one worker per network seed")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.out, "config", "stage_config.json")))
    recomputed = canonical_checksum(cfg)
    if recomputed != cfg["checksum"]:
        print(f"[stage1] config checksum mismatch: stored={cfg['checksum'][:12]} "
              f"recomputed={recomputed[:12]} -- config was edited after Stage 0")
        return 1

    probe = os.path.join(a.out, "stage1_variance_probe")
    per_run = os.path.join(probe, "per_run")
    os.makedirs(probe, exist_ok=True)

    rep = preflight(cfg)
    json.dump(dict(preflight=rep, provenance=provenance()),
              open(os.path.join(probe, "preflight.json"), "w"), indent=2, default=str)
    if not rep["ok"]:
        bad = [k for k, v in rep["checks"].items() if not v["ok"]]
        print(f"[stage1] PREFLIGHT FAILED on {bad}; worst budget error "
              f"{rep['worst_budget_error']:.2e}")
        print("[stage1] refusing to launch 96 simulations on a vacuous comparison")
        return 1
    print("[stage1] preflight OK:",
          {k: round(v["observed"], 3) for k, v in rep["checks"].items()})

    todo = [(s, cfg, os.path.join(a.out, "network_cache"), per_run) for s in cfg["seeds"]]
    print(f"[stage1] {len(todo)} seeds x {len(ARM_NAMES)} arms, {a.workers} workers")
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for i, r in enumerate(pool.imap_unordered(_seed_job, todo), 1):
            print(f"[stage1] {i}/{len(todo)} seed {r['seed']} "
                  f"{r.get('error') or r['arms']}", flush=True)

    got = {(int(d), f[:-5]) for d in os.listdir(per_run)
           for f in os.listdir(os.path.join(per_run, d)) if f.endswith(".json")}
    want = {(s, arm) for s in cfg["seeds"] for arm in ARM_NAMES}
    missing = want - got
    print(f"[stage1] {len(got)}/{len(want)} runs present")
    if missing:
        print(f"[stage1] MISSING {sorted(missing)[:10]}{' ...' if len(missing) > 10 else ''}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: 先提交脚本（运行之前）**

```bash
git add scripts/run_topic4_core_field_stage1.py src/topic4_core_field_runner.py
git commit -m "feat(topic4-core-field): Stage 1 probe, parallelised over network seeds

One worker owns a seed and runs all eight arms on a single build. Dispatching by
(arm, seed) would have eight workers miss the cache together and race to
overwrite the same file, which also defeats the one-build-per-seed promise."
```

- [ ] **Step 4: 只跑 pre-flight，不开仿真**

Run:
```bash
python -c "
import json,sys; sys.path.insert(0,'.')
import importlib.util
spec=importlib.util.spec_from_file_location('s1','scripts/run_topic4_core_field_stage1.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
cfg=json.load(open('results/topic4_sef_hfo/data_driven_core_field/config/stage_config.json'))
rep=m.preflight(cfg); print('ok =',rep['ok'])
for k,v in rep['checks'].items():
    print(f\"  {k} {v['metric']:20s} observed={v['observed']:.3f} thr={v['threshold']} corr={v['correlation']:.3f} ok={v['ok']}\")
print('worst budget error =',rep['worst_budget_error'])"
```
Expected: `ok = True`；B1–B4 的 `observed` 都超过各自 `threshold`；`worst budget error < 1e-6`。任一不过 **停下来**修参数化，不要跑仿真。

- [ ] **Step 5: 跑 Stage 1（约 1–1.6 小时）**

Run: `python scripts/run_topic4_core_field_stage1.py --workers 8`
Expected: 12 个 seed 全部完成，末行 `96/96 runs present`。中断可直接重跑（已完成的臂会跳过）。

- [ ] **Step 6: 提交产物**

```bash
git add results/topic4_sef_hfo/data_driven_core_field/stage1_variance_probe/preflight.json \
        results/topic4_sef_hfo/data_driven_core_field/stage1_variance_probe/per_run
git commit -m "chore(topic4-core-field): Stage 1 probe artifacts (96 runs)"
```

---

### Task 11: Stage 1 分析、报告与图

**重读 spec：** §7.3、§7.4、§7.5、§7.6

**Files:** Create `scripts/analyze_topic4_core_field_stage1.py`

**Interfaces:** Produces `per_run.csv`、`prespecified_comparisons.csv`、`concordance.csv`、`axis_only_comparison.csv`、`stage1_report.json`、`figures/*.pdf` + `figures/README.md`

- [ ] **Step 1: 写分析脚本**

```python
# scripts/analyze_topic4_core_field_stage1.py
"""Score every Stage 1 run under all scoring combinations and hand the table to
the descriptive report.

Exploratory posture: this prints numbers and a recommendation. It does not gate.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.getcwd())
from src.topic4_core_field_report import (  # noqa: E402
    COMPARISONS, PRIMARY_KEY, SCORE_KEYS, SEEDS, SIM_ARMS, _arm_n_dir,
    arm_value, concordance, stage1_report, tiered_paired_stats)
from src.topic4_core_field_runner import canonical_checksum  # noqa: E402
from src.topic4_core_field_scoring import (  # noqa: E402
    adversarial_gain, assignment_invariant_S, axis_only_templates,
    balanced_pair_score, load_patient_templates, model_templates, sim_matrix)

OUT = "results/topic4_sef_hfo/data_driven_core_field"
RUN = "results/topic4_sef_hfo/field_swap_subject_snn"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    probe = os.path.join(a.out, "stage1_variance_probe")
    fig_dir = os.path.join(probe, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    cfg = json.load(open(os.path.join(a.out, "config", "stage_config.json")))
    recomputed = canonical_checksum(cfg)
    if recomputed != cfg["checksum"]:
        raise SystemExit(f"config checksum mismatch: stored={cfg['checksum'][:12]} "
                         f"recomputed={recomputed[:12]}")
    support = cfg["support"]
    targets = {s: load_patient_templates(cfg["subject"], s) for s in cfg["sources"]}

    runs, rows = {}, []
    for seed in cfg["seeds"]:
        for arm in SIM_ARMS:
            rec = json.load(open(os.path.join(probe, "per_run", str(seed), f"{arm}.json")))
            if rec.get("config_checksum") != cfg["checksum"]:
                raise SystemExit(f"run {arm}/{seed} was produced under a different config")
            m = model_templates(rec["events"], support, part_min=cfg["part_min"])
            for src in cfg["sources"]:
                for rule in cfg["missing_rules"]:
                    S = assignment_invariant_S(sim_matrix(m, targets[src], support, rule))
                    P = balanced_pair_score(m, targets[src], support)
                    common = dict(n_dir=m["n_dir"],
                                  coverage_forward=m["coverage_forward"],
                                  coverage_reverse=m["coverage_reverse"])
                    runs[(arm, seed, src, rule, "spearman")] = dict(S_rank=S, **common)
                    runs[(arm, seed, src, rule, "pair")] = dict(S_rank=P, **common)
                    rows.append(dict(arm=arm, seed=seed, source=src, missing_rule=rule,
                                     n_dir=m["n_dir"], S_spearman=S, S_pair=P,
                                     coverage_forward=m["coverage_forward"],
                                     coverage_reverse=m["coverage_reverse"],
                                     coverage_union=m["coverage_union"],
                                     mean_n_part=m["mean_n_part"],
                                     adversarial_gain=adversarial_gain(
                                         m, targets[src], support, rule)["gain"],
                                     n_events=rec["n_events"], h_sum=rec["h_sum"]))

    with open(os.path.join(probe, "per_run.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(rows[0])); w.writeheader(); w.writerows(rows)

    comp_rows = []
    for key in SCORE_KEYS:
        for comp in COMPARISONS:
            if comp["b"] == "axis_only":
                continue
            pairs = [(_arm_n_dir(runs, comp["a"], s, key),
                      arm_value(runs, comp["a"], s, key, "S_rank"),
                      _arm_n_dir(runs, comp["b"], s, key),
                      arm_value(runs, comp["b"], s, key, "S_rank")) for s in SEEDS]
            comp_rows.append(dict(comparison=comp["name"], group=comp["group"],
                                  purpose=comp["purpose"], source=key[0],
                                  missing_rule=key[1], score_def=key[2],
                                  **tiered_paired_stats(pairs)))
    with open(os.path.join(probe, "prespecified_comparisons.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(comp_rows[0]))
        w.writeheader(); w.writerows(comp_rows)

    with open(os.path.join(probe, "concordance.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["source", "missing_rule", "score_def", "concordance"])
        for key in SCORE_KEYS:
            w.writerow([*key, concordance(runs, key)])

    # --- C group: every arm against the pure-geometry reference -----------
    fd = np.load(os.path.join(
        RUN, f"figdata_{cfg['subject']}_gradient_shared_corefrozen_cr1p5_s5_20260722.npz"),
        allow_pickle=True)
    reg = fd["reg"].item()
    ao = axis_only_templates([str(x) for x in fd["names"]],
                             np.asarray(fd["contacts"], float),
                             np.asarray(reg["center"]), np.asarray(reg["axis_unit"]))
    with open(os.path.join(probe, "axis_only_comparison.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "source", "missing_rule", "mean_S", "axis_only_S", "delta"])
        for src in cfg["sources"]:
            for rule in cfg["missing_rules"]:
                ao_S = assignment_invariant_S(sim_matrix(ao, targets[src], support, rule))
                for arm in SIM_ARMS:
                    vals = [runs[(arm, s, src, rule, "spearman")]["S_rank"] for s in SEEDS]
                    vals = [v for v in vals if np.isfinite(v)]
                    mean = float(np.mean(vals)) if vals else float("nan")
                    w.writerow([arm, src, rule, mean, ao_S, mean - ao_S])

    report = stage1_report(runs, cfg)
    json.dump(report, open(os.path.join(probe, "stage1_report.json"), "w"),
              indent=2, default=str)
    print(f"[stage1] integrity = {report['integrity']['status']}")
    if report["integrity"]["status"] == "ok":
        print(f"[stage1] shape separates = {report['recommendation']['shape_separates']} "
              f"{report['recommendation']['separating_dimensions']}")
        print(f"[stage1] equivalence A = {report['equivalence']['A']['equivalent']}, "
              f"A2 = {report['equivalence']['A2']['equivalent']}")
        print(f"[stage1] low-coverage arms = {report['coverage']['low_coverage_arms']}")

    key = PRIMARY_KEY
    ao_S = assignment_invariant_S(sim_matrix(ao, targets[key[0]], support, key[1]))
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for i, arm in enumerate(SIM_ARMS):
        vals = [runs[(arm, s) + key]["S_rank"] for s in SEEDS]
        vals = [v for v in vals if np.isfinite(v)]
        ax.scatter(np.full(len(vals), i) + rng.uniform(-.12, .12, len(vals)), vals,
                   s=22, alpha=.75)
        if vals:
            ax.hlines(np.mean(vals), i - .28, i + .28, lw=2.2, color="k")
    ax.axhline(ao_S, color="crimson", ls="--", lw=1.2)
    ax.text(len(SIM_ARMS) - .5, ao_S, " axis-only", color="crimson", va="center", fontsize=8)
    ax.set_xticks(range(len(SIM_ARMS)))
    ax.set_xticklabels(SIM_ARMS, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("assignment-invariant rank match")
    ax.set_title("Stage 1 arms, 12 paired network seeds")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "stage1_arm_scores.pdf")); plt.close(fig)

    shape = [c for c in COMPARISONS if c["group"] == "shape"]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for i, comp in enumerate(shape):
        d = [arm_value(runs, comp["a"], s, key, "S_rank")
             - arm_value(runs, comp["b"], s, key, "S_rank") for s in SEEDS
             if _arm_n_dir(runs, comp["a"], s, key) == _arm_n_dir(runs, comp["b"], s, key)]
        d = [v for v in d if np.isfinite(v)]
        ax.scatter(np.full(len(d), i), d, s=22, alpha=.75)
        if d:
            ax.hlines(np.mean(d), i - .25, i + .25, lw=2.2, color="k")
    ax.axhline(0, color="0.6", lw=.8)
    ax.set_xticks(range(len(shape)))
    ax.set_xticklabels([f"{c['name']}\n{c['purpose']}" for c in shape], fontsize=7.5)
    ax.set_ylabel("same-tier paired difference vs manual_smooth")
    ax.set_title("Pre-registered shape comparisons")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "stage1_shape_deltas.pdf")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    width = 0.38
    for j, d in enumerate(("coverage_forward", "coverage_reverse")):
        vals = [float(np.mean([runs[(arm, s) + key][d] for s in SEEDS])) for arm in SIM_ARMS]
        ax.bar(np.arange(len(SIM_ARMS)) + (j - .5) * width, vals, width,
               label=d.replace("coverage_", ""))
    ax.set_xticks(range(len(SIM_ARMS)))
    ax.set_xticklabels(SIM_ARMS, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("contacts recruited / frozen support")
    ax.legend(frameon=False, fontsize=8); ax.set_title("Per-direction coverage")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "stage1_coverage.pdf")); plt.close(fig)
    print(f"[stage1] wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 先提交脚本**

```bash
git add scripts/analyze_topic4_core_field_stage1.py
git commit -m "feat(topic4-core-field): Stage 1 comparisons, report and figures"
```

- [ ] **Step 3: 跑分析**

Run: `python scripts/analyze_topic4_core_field_stage1.py`
Expected: 打印 integrity / shape separates / equivalence / low-coverage；写出 5 个数据文件 + 3 张图

- [ ] **Step 4: 目视检查三张图**

打开 `stage1_arm_scores.pdf`（八臂散点 + axis-only 红虚线）、`stage1_shape_deltas.pdf`（同档位配对差）、`stage1_coverage.pdf`（逐方向覆盖率）。**图必须自己看过再写 README。**

- [ ] **Step 5: 写 `figures/README.md`（中文，图看过之后写）**

```markdown
# Stage 1 配对方差探针 — 图说明

### stage1_arm_scores.pdf
八个病理场在 12 个**配对网络种子**上的模板匹配得分，每点一个种子，黑横线为均值，
红虚线是 axis-only 参照（完全没有病理场、只按触点在轴上的投影排序）。所有臂共用同一张网、
同一条噪声轨迹，因此臂间差异只来自场本身。

**关注点**：先看有没有任何一个臂稳定高过红虚线 —— 高不过就说明得分可由沿轴几何完全解释；
再看 `manual_hard` 与 `manual_projected` 是否重叠（重叠说明换抽样方式没有偷偷改变工作点）。

### stage1_shape_deltas.pdf
四个预注册形状对比（纵向形状 / 横向摊平 / 横向拉长 / 横向位置）相对 `manual_smooth` 的
逐种子配对差。**只画两臂方向档位相同的种子** —— 档位不同的种子记为方向胜负，不并入数值差。

**关注点**：同一列的点是否稳定落在零线同一侧。哪个臂赢不重要，重要的是有没有任何一个几何维度
分得开；分不开就意味着这 15 个触点的秩次读不出场的形状。

### stage1_coverage.pdf
每个臂两个方向各自招募到的触点比例（分母是冻结的打分支撑集）。

**关注点**：某个臂若靠"少招募几个难匹配的触点"取得高分，会在这里露出来 —— 逐方向看，
不看并集，并集会把"两个方向各覆盖一小块"掩盖掉。
```

- [ ] **Step 6: 提交产物**

```bash
git add results/topic4_sef_hfo/data_driven_core_field/stage1_variance_probe/
git commit -m "chore(topic4-core-field): Stage 1 analysis, report and figures"
```

- [ ] **Step 7: 停下来汇报，不要自动进 Stage 2**

把 `stage1_report.json` 的数值与含义讲给用户，等用户决定。Stage 2 的结局分类（spec §8.1）
与等价最优场协议（§8.2）需要在开跑前冻结 —— 那时才恢复严格预注册。

---

## 自查

**Spec 覆盖：** §4.1→T1；§4.3→T2；§4.4+§7.2→T3；§5.1/5.2/5.2a/5.4/5.5→T4；§5.3→T5+T8；§6.2→T6；§6.3a→T7；§7.4/7.5→T8；§6.1→T9；§7.2 跑→T10；§7.3/7.6→T11。
§6.3 流式包络未实现（spec 明写 Stage 1 不强制）。§8/§9/§10.3 属 Stage 2/3。

**第三轮 8 条 P0 落点：**
P0-1→T3（`manual_projected` = hard mask；新增 `manual_smooth`；测试坐标统一 ±sep/2）；
P0-2→T3（`preflight_shape` 只覆盖 B1–B4，用形状指标，相关只作诊断）；
P0-3→T4+T8（`assignment_invariant_S` 单方向返回 NaN；`tiered_paired_stats` 只在同档位作差，跨档位记胜负）；
P0-4→T4+T8（逐方向 coverage、`balanced_pair_score`、`adversarial_gain`、coverage 进报告）；
P0-5→T8（TOST 式区间 + 冻结 `delta_eq`，不用"p > 0.05"）；
P0-6→T9（真跑 seed 5 parity + 写 `N_core_manual` / `D0`）；
P0-7→T6+T9+T10+T11（`canonical_checksum` 去自身字段重算、先提交后运行、per-run 记 provenance + config checksum）；
P0-8→T10（按种子并行 + 缓存原子 rename + 逐文件原子写 + 96 键验收）。
末条杂项：`expit` + 输入校验→T2；CRN 比较 OU 轨迹→T7；`common_only` 降级为敏感性→T4；`gate=4` 删除→T4+T8；三类敏感性分列→T8；`concordance.csv` 与 axis-only C 组→T11。

**占位符：** 无 TBD/TODO；每个代码步骤都是可运行代码。

**类型一致性：** `arm_h(name, s, r, geom, target_count, manual_mask_E=None)`、`model_templates(events, support, part_min=PART_MIN)`、`sim_matrix(model, target, support, missing_rule)`、`candidate_key(n_dir, s_rank)`、`get_network(p, theta_deg, ar, cache_dir)`、`run_arm_on_network(arm, seed, cfg, net, NE, NI, reg, cmrun)`、`stage1_report(runs, config)` 定义处与调用处一致；`runs` 键统一为 `(arm, seed, source, missing_rule, score_def)`，**无 gate 维度**。
