# Topic 4 数据驱动病理场 — Stage 0 + Stage 1 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建成"病理场参数化 + 冻结打分 + 网络缓存 + 纯函数裁决器"四件基础设施，跑完 Stage 0 的三个参照分数与 Stage 1 的 84 次配对方差探针，得到一个 fail-closed 的闸门裁定。

**Architecture:** 四个纯计算模块（场、打分、裁决、运行器）+ 三个脚本（Stage 0 参照、Stage 1 跑、Stage 1 分析）。仿真部分**只调用**现有引擎与读出链路，不改任何引擎代码。所有科学判据在跑之前写进 config 并落盘校验和。

**Tech Stack:** Python 3, NumPy, SciPy (`truncnorm`, `spearmanr`, `binomtest`), Matplotlib, pytest, `multiprocessing`。SNN 引擎在 `src/snn_engine/`（纯 NumPy CPU）。

**Spec:** `docs/superpowers/specs/2026-08-06-topic4-axis-constrained-data-driven-core-field-design.md`（rev2.2）。每个 Task 开工前**重读对应 §**（CLAUDE.md §5：每个步骤边界都是重读检查点）。

## Global Constraints

- **被试固定 `epilepsiae_1146`，montage 固定 `narrow`，placement 固定 `gradient_shared`。**
- **共享轴 `u_C` 冻结**，来自 `results/interictal_propagation_masked/template_gradient_fields/per_subject/epilepsiae_1146.json`；`theta_deg = -22.8`。任何代码不得重估、旋转、微调它。
- **不修改** `src/snn_engine/` 下任何文件、`scripts/run_sef_hfo_subject_snn.py`、`scripts/paper_figures/plot_fig_subject_snn_realvsmodel.py`。后两者承载已发表数字。
- 引擎常数：`V_base = 18.0`、`V_reset = 11.0`、`core_mean = 17.5`、`core_std = 1.0`、`core_r = 1.5`、`L = 20.0`、`density = 100.0`、`AR = 2.0`、`g = 3.6`、`dt = 0.1`、`k_dir = 2`、`N = 40000`、`N_E = 32000`。
- 场 config 常数（Stage 0 冻结）：`M = 9`、`EPS = 1e-3`、`TAU_H = 0.25`、`A0 = B0 = 1.5`、`SIGMA_S0 = 1.2 × 基间距`、`AXIAL_MARGIN = 2.0`。
- **Σ h_i = N_core^manual** 是唯一预算约束；`d_i` 保留符号（约 31% 为负）。
- **横向自由度是定面积长宽比 `ρ`，不是裸 `σ_⊥`**（spec §4.1 + §4.4）。
- 打分支撑集 `SUPPORT` 在 Stage 1 开跑前冻结；所有臂、所有候选同一支撑集同一缺失规则。
- 全部产出落 `results/topic4_sef_hfo/data_driven_core_field/`。
- 每个含图目录**必须**有中文 `figures/README.md`（`### filename` + 2–4 句 + 一行 `**关注点**：`），图生成后写。
- 测试标记：涉及仿真的用 `@pytest.mark.slow` 或 `@pytest.mark.integration`；纯计算测试不加标记，必须秒级。

## 文件结构

| 文件 | 职责 |
|---|---|
| `src/topic4_core_field.py` | 轴向坐标、partition-of-unity 基、七个臂的 `q` 场、latent-quantile 抽样、预算投影、`V_th` 构造。**纯计算，不含仿真** |
| `src/topic4_core_field_scoring.py` | 冻结支撑打分：患者模板加载（两套源）、模型模板、2×2 Spearman、`S_pair`、交换不变 `S_rank`、`axis_only`、字典序候选键。**纯计算** |
| `src/topic4_core_field_verdict.py` | Stage 1 裁决器（纯函数、fail-closed、9 条出口） |
| `src/topic4_core_field_runner.py` | 网络缓存（完整配置哈希）+ 单次臂运行（调引擎、走现有读出链路） |
| `scripts/run_topic4_core_field_stage0.py` | Stage 0：基线复现、三参照、冻结 config、完整性报告 |
| `scripts/run_topic4_core_field_stage1.py` | Stage 1：pre-flight + 7 臂 × 12 种子并行跑 + checkpoint/resume |
| `scripts/analyze_topic4_core_field_stage1.py` | 预注册对比、concordance、裁决、图 + README |
| `tests/test_topic4_core_field.py` | Task 1–3 |
| `tests/test_topic4_core_field_scoring.py` | Task 4–5 |
| `tests/test_topic4_core_field_cache.py` | Task 6 |
| `tests/test_topic4_core_field_crn.py` | Task 7（integration，慢） |
| `tests/test_topic4_core_field_verdict.py` | Task 8 |

---

### Task 1: 轴向坐标与 partition-of-unity 基

**重读 spec：** §4.1、§4.4

**Files:**
- Create: `src/topic4_core_field.py`
- Test: `tests/test_topic4_core_field.py`

**Interfaces:**
- Produces: `axis_coords(pos, center, u_axis) -> (s, r)`；`axial_basis_centers(s_support, M) -> kappa`；`partition_of_unity(s, kappa, sigma_s) -> Phi`（形状 `(len(s), M)`，每行和为 1）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_topic4_core_field.py
import numpy as np
import pytest
from src.topic4_core_field import axis_coords, axial_basis_centers, partition_of_unity


def test_axis_coords_projects_onto_axis_and_perpendicular():
    pos = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    center = np.array([0.0, 0.0])
    u = np.array([1.0, 0.0])
    s, r = axis_coords(pos, center, u)
    assert np.allclose(s, [1.0, 0.0, 2.0])
    assert np.allclose(np.abs(r), [0.0, 1.0, 2.0])


def test_axis_coords_axis_flip_negates_s_and_preserves_abs_r():
    """u_C is an undirected line: flipping it must only negate s."""
    rng = np.random.default_rng(0)
    pos = rng.uniform(-5, 5, size=(50, 2))
    center = np.array([0.3, -0.2])
    u = np.array([0.6, 0.8])
    s1, r1 = axis_coords(pos, center, u)
    s2, r2 = axis_coords(pos, center, -u)
    assert np.allclose(s2, -s1)
    assert np.allclose(np.abs(r2), np.abs(r1))


def test_partition_of_unity_rows_sum_to_one():
    kappa = axial_basis_centers((-8.0, 8.0), M=9)
    s = np.linspace(-8.0, 8.0, 200)
    sigma_s = 1.2 * (kappa[1] - kappa[0])
    Phi = partition_of_unity(s, kappa, sigma_s)
    assert Phi.shape == (200, 9)
    assert np.allclose(Phi.sum(axis=1), 1.0, atol=1e-12)


def test_uniform_weights_give_flat_axial_profile():
    """P0-6: this is why partition-of-unity is required. Unnormalised Gaussians
    would sag at the ends and make `uniform_axial` a broad peak, not a corridor."""
    kappa = axial_basis_centers((-8.0, 8.0), M=9)
    s = np.linspace(-8.0, 8.0, 200)
    sigma_s = 1.2 * (kappa[1] - kappa[0])
    pi = np.full(9, 1.0 / 9.0)
    profile = partition_of_unity(s, kappa, sigma_s) @ pi
    rel_ripple = (profile.max() - profile.min()) / profile.mean()
    assert rel_ripple < 1e-6
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_topic4_core_field.py -x -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.topic4_core_field'`

- [ ] **Step 3: 最小实现**

```python
# src/topic4_core_field.py
"""Topic 4 axis-constrained data-driven pathology field (spec rev2.2).

Pure computation: no simulation, no engine import. Builds the per-neuron
threshold field V_th from a low-dimensional spatial parameterisation under a
fixed effective-count budget.
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


def axis_coords(pos, center, u_axis):
    """Axial (s) and transverse (r) coordinates relative to `center`.

    `u_axis` is the undirected shared axis; flipping its sign negates s and r
    together, which every downstream score must be invariant to (spec red line 2).
    """
    pos = np.asarray(pos, float)
    u = np.asarray(u_axis, float)
    u = u / np.linalg.norm(u)
    u_perp = np.array([-u[1], u[0]])
    d = pos - np.asarray(center, float)[None, :]
    return d @ u, d @ u_perp


def axial_basis_centers(s_support, M=M_DEFAULT):
    """M evenly spaced basis centres spanning the axial support."""
    lo, hi = float(s_support[0]), float(s_support[1])
    return np.linspace(lo, hi, int(M))


def partition_of_unity(s, kappa, sigma_s):
    """Normalised Gaussian bases: rows sum to exactly 1 (spec §4.1, P0-6)."""
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
corridor every Stage 1 and Stage 2 comparison is measured against."
```

---

### Task 2: latent-quantile 抽样合同与预算投影

**重读 spec：** §4.3.1、§4.3.2、§2.5

**Files:**
- Modify: `src/topic4_core_field.py`
- Test: `tests/test_topic4_core_field.py`

**Interfaces:**
- Consumes: 无（独立于 Task 1）
- Produces: `sample_core_quantiles(n_E, seed) -> u`；`core_thresholds(u, core_mean=17.5, core_std=1.0, v_reset=11.0) -> V_core`；`signed_depth(V_core, v_base=18.0) -> d`；`project_to_budget(q, target_count, tau_h=TAU_H) -> (h, lam)`；`build_vth(h, d, n_total, n_E, v_base=18.0) -> vth`

- [ ] **Step 1: 写失败测试**

```python
# append to tests/test_topic4_core_field.py
from src.topic4_core_field import (
    sample_core_quantiles, core_thresholds, signed_depth,
    project_to_budget, build_vth,
)


def test_core_thresholds_match_the_truncated_normal_moments():
    u = sample_core_quantiles(200_000, seed=7)
    v = core_thresholds(u)
    assert v.min() >= 11.0
    assert abs(v.mean() - 17.5) < 0.02
    assert abs(v.std() - 1.0) < 0.02


def test_signed_depth_keeps_the_negative_third():
    """~31% of 'core' neurons sit ABOVE baseline; max(0,.) would drop them and
    break parity with the accepted manual core (spec C1)."""
    d = signed_depth(core_thresholds(sample_core_quantiles(200_000, seed=7)))
    frac_negative = (d < 0).mean()
    assert 0.28 < frac_negative < 0.34
    assert abs(d.mean() - 0.5) < 0.02


def test_budget_projection_hits_the_target_count():
    rng = np.random.default_rng(1)
    q = rng.uniform(EPS, 1.0, size=32_000)
    h, lam = project_to_budget(q, target_count=1131.0)
    assert np.isfinite(lam)
    assert (h >= 0).all() and (h <= 1).all()
    assert abs(h.sum() - 1131.0) / 1131.0 < 1e-6


def test_budget_projection_is_monotone_in_lambda():
    """Sum h is strictly decreasing in lambda, so the bisection root is unique.
    Budgeting on sum(h*d) instead would NOT be monotone (31% of d is negative)."""
    rng = np.random.default_rng(2)
    q = rng.uniform(EPS, 1.0, size=5_000)
    lq = np.log(q + EPS)
    sums = [1.0 / (1.0 + np.exp(-(lq - lam) / TAU_H)) for lam in np.linspace(-8, 2, 25)]
    totals = [s.sum() for s in sums]
    assert all(b < a for a, b in zip(totals, totals[1:]))


def test_build_vth_places_baseline_outside_and_core_distribution_inside():
    n_E, n_total = 1000, 1250
    u = sample_core_quantiles(n_E, seed=3)
    d = signed_depth(core_thresholds(u))
    h = np.zeros(n_E)
    h[:100] = 1.0
    vth = build_vth(h, d, n_total=n_total, n_E=n_E)
    assert vth.shape == (n_total,)
    assert np.allclose(vth[100:], 18.0)                      # E outside + all I
    assert np.allclose(vth[:100], 18.0 - d[:100])            # h=1 restores V_core
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_topic4_core_field.py -x -q`
Expected: FAIL — `ImportError: cannot import name 'sample_core_quantiles'`

- [ ] **Step 3: 最小实现**

```python
# append to src/topic4_core_field.py
from scipy.stats import truncnorm

V_BASE = 18.0
V_RESET = 11.0
CORE_MEAN = 17.5
CORE_STD = 1.0


def sample_core_quantiles(n_E, seed):
    """One uniform quantile per E neuron, drawn once and frozen (spec §4.3.1).

    Position-independent and field-independent, so every arm and every candidate
    shares the same latent draw and differs only in h.
    """
    return np.random.default_rng(int(seed)).uniform(0.0, 1.0, size=int(n_E))


def core_thresholds(u, core_mean=CORE_MEAN, core_std=CORE_STD, v_reset=V_RESET):
    """Truncated-normal inverse transform.

    Same distribution as the engine's rejection sampler, deterministic per
    neuron. Bitwise reproduction of the legacy draw is impossible -- the engine
    resamples on rejection, so its stream position is data-dependent (spec P0-4).
    """
    a = (float(v_reset) - float(core_mean)) / float(core_std)
    return truncnorm.ppf(np.asarray(u, float), a=a, b=np.inf,
                         loc=float(core_mean), scale=float(core_std))


def signed_depth(v_core, v_base=V_BASE):
    """d_i = V_base - V_core,i, sign preserved (about 31% negative)."""
    return float(v_base) - np.asarray(v_core, float)


def project_to_budget(q, target_count, tau_h=TAU_H, eps=EPS, max_iter=200):
    """Bisect lambda so that sum_i h_i == target_count.

    h_i = sigmoid((log(q_i + eps) - lambda) / tau_h) is strictly decreasing in
    lambda, so the root is unique. This is a LEVEL-SET operation: the region's
    size is pinned by the budget and q only sets its shape (spec §4.4).
    """
    lq = np.log(np.asarray(q, float) + eps)
    lo, hi = lq.min() - 20.0, lq.max() + 20.0
    target = float(target_count)

    def total(lam):
        return 1.0 / (1.0 + np.exp(-(lq - lam) / tau_h))

    for _ in range(max_iter):
        lam = 0.5 * (lo + hi)
        if total(lam).sum() > target:
            lo = lam
        else:
            hi = lam
    lam = 0.5 * (lo + hi)
    return total(lam), lam


def build_vth(h, d, n_total, n_E, v_base=V_BASE):
    """Per-neuron threshold vector for the engine. I neurons keep baseline."""
    vth = np.full(int(n_total), float(v_base))
    vth[:int(n_E)] = float(v_base) - np.asarray(h, float) * np.asarray(d, float)
    return vth
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_topic4_core_field.py -x -q`
Expected: 9 passed

- [ ] **Step 5: 提交**

```bash
git add src/topic4_core_field.py tests/test_topic4_core_field.py
git commit -m "feat(topic4-core-field): latent-quantile sampling and effective-count budget

Budgeting on sum(h) rather than sum(h*d) keeps the bisection strictly monotone,
which sum(h*d) is not once a third of d is negative."
```

---

### Task 3: 七个臂的 `q` 场与非空性 pre-flight

**重读 spec：** §7.2、§4.4

**Files:**
- Modify: `src/topic4_core_field.py`
- Test: `tests/test_topic4_core_field.py`

**Interfaces:**
- Consumes: Task 1 的 `axis_coords` / `partition_of_unity`，Task 2 的 `project_to_budget`
- Produces: `two_core_q(s, r, sep, rho, a0=A0, b0=B0) -> q`；`uniform_axial_q(s, r, kappa, sigma_s, sigma_perp) -> q`；`ARM_NAMES`；`arm_h(name, s, r, geom, target_count) -> h`；`preflight_arm_distinctness(h_by_arm, max_corr=0.95) -> dict`

- [ ] **Step 1: 写失败测试**

```python
# append to tests/test_topic4_core_field.py
from src.topic4_core_field import (
    ARM_NAMES, arm_h, preflight_arm_distinctness, two_core_q,
)


def _mock_sheet(n=32_000, L=20.0, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, L, size=(n, 2))
    return pos[:, 0] - L / 2.0, pos[:, 1] - L / 2.0


def _geom():
    return dict(sep=6.0, s_support=(-8.0, 8.0), M=9, sigma_perp=1.5, shift_mm=3.0)


def test_all_seven_arms_hit_the_same_budget():
    s, r = _mock_sheet()
    for name in ARM_NAMES:
        if name == "manual_hard":
            continue
        h = arm_h(name, s, r, _geom(), target_count=1131.0)
        assert abs(h.sum() - 1131.0) / 1131.0 < 1e-6, name


def test_width_arms_change_shape_not_just_edge_blur():
    """spec 4.4: bare sigma_perp only blurs the edge because the budget pins the
    region area. rho reshapes it at fixed a*b."""
    s, r = _mock_sheet()
    g = _geom()
    wide = arm_h("width_wide", s, r, g, 1131.0)
    narrow = arm_h("width_narrow", s, r, g, 1131.0)
    rms_r = lambda h: np.sqrt((h * r ** 2).sum() / h.sum())
    assert rms_r(wide) > 2.5 * rms_r(narrow)
    assert np.corrcoef(wide, narrow)[0, 1] < 0.6


def test_transverse_arms_are_mirror_images():
    """u_perp's sign is a convention, so +delta and -delta must be symmetric."""
    s, r = _mock_sheet()
    g = _geom()
    plus = arm_h("transverse_plus", s, r, g, 1131.0)
    minus = arm_h("transverse_minus", s, -r, g, 1131.0)
    assert np.corrcoef(plus, minus)[0, 1] > 0.999


def test_manual_projected_reproduces_the_manual_core_geometry():
    s, r = _mock_sheet()
    h = arm_h("manual_projected", s, r, _geom(), 1131.0)
    manual_mask = (np.minimum((s - 6.0) ** 2, (s + 6.0) ** 2) + r ** 2) <= 1.5 ** 2
    assert np.corrcoef(h, manual_mask.astype(float))[0, 1] >= 0.9


def test_preflight_rejects_a_vacuous_arm_pair():
    s, r = _mock_sheet()
    g = _geom()
    h_by_arm = {n: arm_h(n, s, r, g, 1131.0) for n in ARM_NAMES if n != "manual_hard"}
    report = preflight_arm_distinctness(h_by_arm)
    assert report["ok"] is True
    h_by_arm["clone"] = h_by_arm["uniform_axial"].copy()
    bad = preflight_arm_distinctness(h_by_arm)
    assert bad["ok"] is False
    assert any("clone" in pair for pair in bad["violations"])
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_topic4_core_field.py -x -q`
Expected: FAIL — `ImportError: cannot import name 'ARM_NAMES'`

- [ ] **Step 3: 最小实现**

```python
# append to src/topic4_core_field.py

ARM_NAMES = (
    "manual_hard",
    "manual_projected",
    "uniform_axial",
    "width_wide",
    "width_narrow",
    "transverse_plus",
    "transverse_minus",
)


def two_core_q(s, r, sep, rho=1.0, a0=A0, b0=B0, r_shift=0.0, eps=EPS):
    """Two elliptical cores at s = +-sep/2, transverse offset r_shift.

    rho is the FIXED-AREA aspect ratio: a = a0*rho, b = b0/rho, so a*b is
    constant and rho reshapes the region instead of merely blurring its edge
    (spec 4.4).
    """
    a = float(a0) * float(rho)
    b = float(b0) / float(rho)
    rr = np.asarray(r, float) - float(r_shift)
    q = np.zeros_like(np.asarray(s, float))
    for c in (-float(sep) / 2.0, float(sep) / 2.0):
        q = np.maximum(q, np.exp(-((np.asarray(s, float) - c) ** 2 / (2 * a ** 2)
                                   + rr ** 2 / (2 * b ** 2))))
    return q + eps


def uniform_axial_q(s, r, kappa, sigma_s, sigma_perp, eps=EPS):
    """Flat axial profile (pi_m == 1/M on a partition-of-unity basis)."""
    M = len(kappa)
    profile = partition_of_unity(np.asarray(s, float), kappa, sigma_s) @ np.full(M, 1.0 / M)
    return profile * np.exp(-np.asarray(r, float) ** 2 / (2 * float(sigma_perp) ** 2)) + eps


def arm_h(name, s, r, geom, target_count):
    """h field for one Stage 1 arm. `manual_hard` is not built here -- it uses
    the legacy engine path (see topic4_core_field_runner)."""
    sep = geom["sep"]
    if name == "manual_projected":
        q = two_core_q(s, r, sep, rho=1.0)
    elif name == "uniform_axial":
        kappa = axial_basis_centers(geom["s_support"], geom["M"])
        sigma_s = SIGMA_S_FACTOR * (kappa[1] - kappa[0])
        q = uniform_axial_q(s, r, kappa, sigma_s, geom["sigma_perp"])
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


def preflight_arm_distinctness(h_by_arm, max_corr=0.95):
    """Refuse to launch 84 simulations on a comparison that is vacuous.

    Costs milliseconds; catches a parameterisation where two arms collapse to the
    same field (spec 4.4).
    """
    names = sorted(h_by_arm)
    violations, table = [], {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            c = float(np.corrcoef(h_by_arm[a], h_by_arm[b])[0, 1])
            table[f"{a}|{b}"] = c
            if c > max_corr:
                violations.append(f"{a}|{b}")
    return dict(ok=not violations, violations=violations,
                correlations=table, max_corr=max_corr)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_topic4_core_field.py -x -q`
Expected: 14 passed

- [ ] **Step 5: 提交**

```bash
git add src/topic4_core_field.py tests/test_topic4_core_field.py
git commit -m "feat(topic4-core-field): the seven Stage 1 arms and a vacuity pre-flight

Width is parameterised as a fixed-area aspect ratio, because the count budget
pins the region's area and scaling sigma_perp alone would only soften its edge.
The pre-flight refuses to spend 84 simulations when two arms build the same field."
```

---

### Task 4: 冻结支撑打分

**重读 spec：** §5.1、§5.2、§5.2a、§5.4

**Files:**
- Create: `src/topic4_core_field_scoring.py`
- Test: `tests/test_topic4_core_field_scoring.py`

**Interfaces:**
- Produces: `load_patient_templates(subject, source) -> {"t_a": {name: rank}, "t_b": {...}}`（`source ∈ {"gradient", "geometry"}`）；`model_templates(events, support, part_min) -> {"forward": {...}, "reverse": {...}, "n_dir": int, "coverage": float}`；`sim_matrix(model, target, support, missing_rule) -> np.ndarray (2,2)`；`assignment_invariant_S(M) -> float`；`pair_score(model, target, support) -> float`；`axis_only_templates(names, coords, center, u_axis) -> dict`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_topic4_core_field_scoring.py
import numpy as np
import pytest
from src.topic4_core_field_scoring import (
    assignment_invariant_S, axis_only_templates, model_templates,
    pair_score, sim_matrix,
)

SUPPORT = ["c1", "c2", "c3", "c4", "c5", "c6"]


def _events(ranks_by_sign):
    out = []
    for sign, ranks in ranks_by_sign:
        out.append({"sign": sign, "n_part": len(ranks), "ranks": ranks})
    return out


def test_model_templates_never_widen_beyond_the_frozen_support():
    ev = _events([(1.0, {"c1": 0.0, "c2": 1.0, "zzz": 2.0})])
    m = model_templates(ev, SUPPORT, part_min=3)
    assert set(m["forward"]) <= set(SUPPORT)
    assert "zzz" not in m["forward"]


def test_coverage_reports_the_fraction_of_frozen_support_recruited():
    ev = _events([(1.0, {"c1": 0.0, "c2": 1.0, "c3": 2.0})])
    m = model_templates(ev, SUPPORT, part_min=3)
    assert m["coverage"] == pytest.approx(3 / 6)


def test_shrinking_the_recruited_set_cannot_raise_the_score():
    """P0-2: scoring only on a candidate's own contacts lets it win by recruiting
    fewer, harder contacts. The frozen support forbids that."""
    target = {"t_a": {n: float(i) for i, n in enumerate(SUPPORT)},
              "t_b": {n: float(-i) for i, n in enumerate(SUPPORT)}}
    full = model_templates(
        _events([(1.0, {n: float(i) for i, n in enumerate(SUPPORT)}),
                 (-1.0, {n: float(-i) for i, n in enumerate(SUPPORT)})]),
        SUPPORT, part_min=3)
    narrow = model_templates(
        _events([(1.0, {n: float(i) for i, n in enumerate(SUPPORT[:3])}),
                 (-1.0, {n: float(-i) for i, n in enumerate(SUPPORT[:3])})]),
        SUPPORT, part_min=3)
    s_full = assignment_invariant_S(sim_matrix(full, target, SUPPORT, "mean_rank"))
    s_narrow = assignment_invariant_S(sim_matrix(narrow, target, SUPPORT, "mean_rank"))
    assert s_narrow <= s_full + 1e-9


def test_score_is_invariant_to_swapping_the_two_patient_templates():
    target = {"t_a": {n: float(i) for i, n in enumerate(SUPPORT)},
              "t_b": {n: float(-i) for i, n in enumerate(SUPPORT)}}
    m = model_templates(
        _events([(1.0, {n: float(i) for i, n in enumerate(SUPPORT)}),
                 (-1.0, {n: float(-i) for i, n in enumerate(SUPPORT)})]),
        SUPPORT, part_min=3)
    s1 = assignment_invariant_S(sim_matrix(m, target, SUPPORT, "mean_rank"))
    swapped = {"t_a": target["t_b"], "t_b": target["t_a"]}
    s2 = assignment_invariant_S(sim_matrix(m, swapped, SUPPORT, "mean_rank"))
    assert s1 == pytest.approx(s2)


def test_axis_only_templates_are_exact_mirrors():
    names = ["c1", "c2", "c3"]
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    t = axis_only_templates(names, coords, np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    fwd = np.array([t["forward"][n] for n in names])
    rev = np.array([t["reverse"][n] for n in names])
    assert np.allclose(fwd, -rev)


def test_pair_score_penalises_missing_contacts_without_shrinking_support():
    target = {"t_a": {n: float(i) for i, n in enumerate(SUPPORT)},
              "t_b": {n: float(-i) for i, n in enumerate(SUPPORT)}}
    full = model_templates(
        _events([(1.0, {n: float(i) for i, n in enumerate(SUPPORT)})]), SUPPORT, part_min=3)
    half = model_templates(
        _events([(1.0, {n: float(i) for i, n in enumerate(SUPPORT[:3])})]), SUPPORT, part_min=3)
    assert pair_score(half, target, SUPPORT) < pair_score(full, target, SUPPORT)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_topic4_core_field_scoring.py -x -q`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 最小实现**

```python
# src/topic4_core_field_scoring.py
"""Frozen-support scoring for the data-driven core field (spec section 5).

Reimplements the definitions used by
scripts/paper_figures/plot_fig_subject_snn_realvsmodel.py (model templates built
from event sign; 2x2 Spearman against the patient's two templates) and adds the
frozen scoring support that file cannot provide. That file is NOT modified -- it
carries published numbers.
"""
from __future__ import annotations

import itertools
import json
import os

import numpy as np
from scipy.stats import spearmanr

GRADIENT_ROOT = "results/interictal_propagation_masked/template_gradient_fields/per_subject"
GEOMETRY_ROOT = "results/spatial_modulation/propagation_geometry/observation_readout/real_subjects"


def load_patient_templates(subject, source, root="."):
    """Patient TA/TB contact ranks. Two non-identical sources, both reported
    (spec C3): they agree at Spearman 0.959 / 0.943 but are not the same numbers."""
    if source == "gradient":
        path = os.path.join(root, GRADIENT_ROOT, f"{subject}.json")
        field = json.load(open(path))["interictal_field"]
        names = [str(x) for x in field["contact_order"]]
        out = {}
        for key, tpl in (("rank_a", "t_a"), ("rank_b", "t_b")):
            vals = np.asarray(field[key], float)
            out[tpl] = {n: float(v) for n, v in zip(names, vals) if np.isfinite(v)}
        return out
    if source == "geometry":
        out = {}
        for tpl in ("t_a", "t_b"):
            g = json.load(open(os.path.join(root, GEOMETRY_ROOT, f"{subject}_{tpl}.json")))
            out[tpl] = {c["name"]: float(c["typical_rank"]) for c in g["channels"]
                        if c.get("typical_rank") is not None}
        return out
    raise ValueError(f"unknown template source {source!r}")


def model_templates(events, support, part_min):
    """Forward/reverse mean within-event rank, restricted to the frozen support.

    Templates are built from the event SIGN, not from cluster labels -- for a
    one-direction readout a cluster->direction mapping would be fabricated.
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
    out, n_dir, recruited = {}, 0, set()
    for key, label in ((+1, "forward"), (-1, "reverse")):
        tpl = {}
        for i, name in enumerate(support):
            if acc[key][i]:
                tpl[name] = float(np.mean(acc[key][i]))
                recruited.add(name)
        out[label] = tpl
        if tpl:
            n_dir += 1
    out["n_dir"] = n_dir
    out["coverage"] = len(recruited) / len(support) if support else 0.0
    return out


def _aligned(model_tpl, target_tpl, support, missing_rule):
    """Model and target vectors on the FULL frozen support.

    missing_rule 'mean_rank': a contact the candidate never recruited is given
    that direction's mean rank -- an explicit modelling assumption (spec P1-1),
    which is why pair_score is reported alongside.
    missing_rule 'common_only': legacy `_sim_matrix` behaviour, kept as the
    sensitivity arm and as the regression target.
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

    Undefined (nan) when neither assignment has both cells -- the caller ranks
    candidates lexicographically by (n_dir, S), never by S alone (spec P0-5).
    """
    opts = []
    for (i, j), (k, l) in (((0, 0), (1, 1)), ((0, 1), (1, 0))):
        if np.isfinite(M[i, j]) and np.isfinite(M[k, l]):
            opts.append(0.5 * (M[i, j] + M[k, l]))
    if opts:
        return float(max(opts))
    finite = M[np.isfinite(M)]
    return float(finite.max()) if finite.size else float("nan")


def pair_score(model, target, support):
    """Fixed-support pairwise concordance (spec P1-1 sensitivity).

    Denominator is always every pair of the frozen support, so a pair touching a
    contact the candidate never recruited contributes 0 rather than being dropped.
    """
    support = list(support)
    pairs = list(itertools.combinations(range(len(support)), 2))
    if not pairs:
        return float("nan")
    best = -np.inf
    for row, col in (("forward", "t_a"), ("forward", "t_b"),
                     ("reverse", "t_a"), ("reverse", "t_b")):
        m, t = model.get(row, {}), target[col]
        tot = 0.0
        for i, j in pairs:
            ni, nj = support[i], support[j]
            if ni in m and nj in m and ni in t and nj in t:
                tot += np.sign((m[ni] - m[nj]) * (t[ni] - t[nj]))
        best = max(best, tot / len(pairs))
    return float(best)


def axis_only_templates(names, coords, center, u_axis):
    """A model with NO pathology field: contacts ordered by axial projection.

    This is the reference every claim has to beat -- pure geometry already scores
    0.696 against this patient's templates (spec section 2.4).
    """
    u = np.asarray(u_axis, float)
    u = u / np.linalg.norm(u)
    proj = (np.asarray(coords, float) - np.asarray(center, float)[None, :]) @ u
    return {"forward": {n: float(p) for n, p in zip(names, proj)},
            "reverse": {n: float(-p) for n, p in zip(names, proj)},
            "n_dir": 2,
            "coverage": 1.0}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_topic4_core_field_scoring.py -x -q`
Expected: 6 passed

- [ ] **Step 5: 加回归测试 —— 与已发表 `_sim_matrix` 对齐**

```python
# append to tests/test_topic4_core_field_scoring.py
import json, glob
from pathlib import Path

RUN = Path("results/topic4_sef_hfo/field_swap_subject_snn")


@pytest.mark.integration
def test_common_only_mode_reproduces_the_published_sim_matrix():
    """Our reimplementation must agree with the file that carries the published
    Figure 4 numbers, in the mode that file implements."""
    import sys
    sys.path.insert(0, ".")
    from scripts.paper_figures.plot_fig_subject_snn_realvsmodel import (
        _model_templates, _real_templates, _sim_matrix,
    )
    tags = sorted(glob.glob(str(RUN / "readout_epilepsiae_1146_paired_tsrc_highn_s*_20260721.json")))
    if not tags:
        pytest.skip("paired_tsrc_highn artifacts not present")
    tag = Path(tags[0]).stem.removeprefix("readout_")
    ref_model = _model_templates(tag)
    real = _real_templates("epilepsiae_1146", "narrow")
    ref_M, _ = _sim_matrix(ref_model, real, B=1, seed=0)

    ro = json.load(open(RUN / f"readout_{tag}.json"))
    support = sorted(set(real["t_a"]) | set(real["t_b"]))
    ours = model_templates(ro["events"], support, part_min=2 * int(ro.get("k_dir", 2)))
    our_M = sim_matrix(ours, {"t_a": real["t_a"], "t_b": real["t_b"]},
                       support, "common_only")
    assert np.allclose(ref_M, our_M, atol=1e-9, equal_nan=True)
```

- [ ] **Step 6: 跑回归测试**

Run: `python -m pytest tests/test_topic4_core_field_scoring.py -q`
Expected: 7 passed（若 artifact 缺失则 6 passed 1 skipped）

- [ ] **Step 7: 提交**

```bash
git add src/topic4_core_field_scoring.py tests/test_topic4_core_field_scoring.py
git commit -m "feat(topic4-core-field): frozen-support scoring with an axis-only reference

The published scorer computes Spearman on whatever contacts a candidate happens
to recruit, so recruiting fewer and easier ones raises the score. Support is now
frozen before optimisation and a regression test pins the reimplementation to the
published file in the mode that file implements."
```

---

### Task 5: 字典序候选键

**重读 spec：** §5.3（P0-5）

**Files:**
- Modify: `src/topic4_core_field_scoring.py`
- Test: `tests/test_topic4_core_field_scoring.py`

**Interfaces:**
- Produces: `candidate_key(n_dir, s_rank) -> tuple`（可直接用于 `sorted`，大者优）

- [ ] **Step 1: 写失败测试**

```python
# append to tests/test_topic4_core_field_scoring.py
from src.topic4_core_field_scoring import candidate_key


def test_two_directions_always_outrank_one_even_when_S_is_lower():
    """The exact counterexample that killed the scalar grading: a one-direction
    candidate matching one template perfectly scores 0.5, while a two-direction
    candidate whose best assignment is +1 and -1 scores 0."""
    one_direction = candidate_key(n_dir=1, s_rank=0.5)
    two_directions = candidate_key(n_dir=2, s_rank=0.0)
    assert two_directions > one_direction


def test_within_a_tier_the_better_match_ranks_higher():
    assert candidate_key(2, 0.8) > candidate_key(2, 0.3)
    assert candidate_key(1, 0.8) > candidate_key(1, 0.3)


def test_no_directions_ranks_last_and_tolerates_nan():
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

    CMA-ES consumes candidate ORDER, so the two tiers can be separated without
    inventing a rate-loss weight. Never compare S_rank across tiers -- a
    one-direction candidate can out-score a two-direction one (spec P0-5).
    """
    s = float(s_rank)
    if not np.isfinite(s):
        s = -np.inf
    return (int(n_dir), s)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_topic4_core_field_scoring.py -q`
Expected: 10 passed（含 1 integration）

- [ ] **Step 5: 提交**

```bash
git add src/topic4_core_field_scoring.py tests/test_topic4_core_field_scoring.py
git commit -m "feat(topic4-core-field): rank candidates lexicographically by (n_dir, S)

A one-direction candidate matching one template perfectly scores 0.5 and a
two-direction candidate whose best assignment is +1 and -1 scores 0, so no scalar
ordering separates the tiers without an invented weight."
```

---

### Task 6: 网络缓存（完整配置哈希）

**重读 spec：** §6.2（Eng-1）

**Files:**
- Create: `src/topic4_core_field_runner.py`
- Test: `tests/test_topic4_core_field_cache.py`

**Interfaces:**
- Produces: `connectivity_config(p, theta_deg, ar) -> dict`；`cache_key(config) -> str`；`get_network(p, theta_deg, ar, seed, cache_dir) -> (net, NE, NI, from_cache)`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_topic4_core_field_cache.py
import numpy as np
import pytest
from src.topic4_core_field_runner import cache_key, connectivity_config


class _P:
    """Minimal stand-in exposing the engine Params fields the cache must cover."""
    L = 20.0; density = 100.0; f_E = 0.8; seed = 1; g = 3.6
    C_EE = 800; C_IE = 800; C_EI = 200; C_II = 200
    l_EE = 0.380; l_IE = 0.250; l_EI = 0.250; l_II = 0.250
    rho_EE = 0.6; rho_IE = 0.0; rho_EI = 0.0; rho_II = 0.0
    tau0 = 0.1; v_axon = 0.3; delay_dt = 0.1


def test_cache_key_is_stable_for_an_unchanged_config():
    p = _P()
    assert cache_key(connectivity_config(p, -22.8, 2.0)) == \
           cache_key(connectivity_config(p, -22.8, 2.0))


@pytest.mark.parametrize("field,value", [
    ("L", 21.0), ("density", 120.0), ("f_E", 0.75), ("seed", 2),
    ("C_EE", 700), ("C_IE", 700), ("C_EI", 150), ("C_II", 150),
    ("l_EE", 0.40), ("l_IE", 0.26), ("l_EI", 0.26), ("l_II", 0.26),
    ("rho_EE", 0.5), ("rho_IE", 0.1), ("rho_EI", 0.1), ("rho_II", 0.1),
    ("tau0", 0.2), ("v_axon", 0.4), ("delay_dt", 0.2),
])
def test_perturbing_any_connectivity_field_changes_the_key(field, value):
    """A key that misses a field silently returns the wrong cached network."""
    base = cache_key(connectivity_config(_P(), -22.8, 2.0))
    p = _P(); setattr(p, field, value)
    assert cache_key(connectivity_config(p, -22.8, 2.0)) != base, field


@pytest.mark.parametrize("theta,ar", [(-20.0, 2.0), (-22.8, 1.5)])
def test_theta_and_aspect_ratio_are_in_the_key(theta, ar):
    base = cache_key(connectivity_config(_P(), -22.8, 2.0))
    assert cache_key(connectivity_config(_P(), theta, ar)) != base
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_topic4_core_field_cache.py -x -q`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 最小实现**

```python
# src/topic4_core_field_runner.py
"""Simulation glue for the Stage 1 probe: network cache + one arm run.

Calls the blessed engine and the existing read-out chain; changes neither.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess

import numpy as np

CONNECTIVITY_FIELDS = (
    "L", "density", "f_E", "seed", "g",
    "C_EE", "C_IE", "C_EI", "C_II",
    "l_EE", "l_IE", "l_EI", "l_II",
    "rho_EE", "rho_IE", "rho_EI", "rho_II",
    "tau0", "v_axon", "delay_dt",
)


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def connectivity_config(p, theta_deg, ar):
    """Every field that can change the connectivity graph.

    Keying on (seed, theta, L, density, AR) alone would silently hit a stale
    cache after any other connectivity parameter moved (spec Eng-1).
    """
    cfg = {f: getattr(p, f) for f in CONNECTIVITY_FIELDS}
    cfg["theta_EE_deg"] = float(theta_deg)
    cfg["AR"] = float(ar)
    cfg["numpy_version"] = np.__version__
    cfg["rng_bit_generator"] = "PCG64"
    cfg["git_commit"] = _git_commit()
    return cfg


def cache_key(config):
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_topic4_core_field_cache.py -q`
Expected: 23 passed

- [ ] **Step 5: 加 `get_network` 与 bit-parity 测试**

```python
# append to src/topic4_core_field_runner.py
def get_network(p, theta_deg, ar, cache_dir):
    """Build or load the connectivity graph. Field-independent, so one build per
    (seed, theta) serves every arm -- the 120 s build is pure overhead otherwise."""
    import sys
    eng = os.path.join("src", "snn_engine")
    if eng not in sys.path:
        sys.path.insert(0, eng)
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
    with open(path, "wb") as fh:
        pickle.dump({"net": net, "NE": NE, "NI": NI, "config": cfg},
                    fh, protocol=pickle.HIGHEST_PROTOCOL)
    return net, NE, NI, False
```

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
```

- [ ] **Step 6: 跑全部缓存测试**

Run: `python -m pytest tests/test_topic4_core_field_cache.py -q`
Expected: 24 passed

- [ ] **Step 7: 提交**

```bash
git add src/topic4_core_field_runner.py tests/test_topic4_core_field_cache.py
git commit -m "feat(topic4-core-field): cache networks under a full connectivity hash

The graph does not depend on the pathology field, so one 120 s build serves every
arm at a seed. Keying on seed and theta alone would have returned a stale network
after any other connectivity parameter moved."
```

---

### Task 7: common random numbers 回归锁

**重读 spec：** §6.3a（Eng-2）

**Files:**
- Test: `tests/test_topic4_core_field_crn.py`

**Interfaces:**
- Consumes: Task 6 的 `get_network`

这个 Task **只有测试**。已核验引擎主循环内只有 `rng.standard_normal()`（每步 1 个）与 `rng.poisson(..., size=N)`（每步 N 个）两处无条件定长调用，故同 seed 给出同一条噪声实现；本测试把该性质锁住，防止将来引擎改动悄悄破坏 Stage 1 的配对设计。

- [ ] **Step 1: 写测试**

```python
# tests/test_topic4_core_field_crn.py
"""Locks the premise Stage 1's paired design rests on.

Verified in the engine: the loop's only RNG calls are one scalar normal and one
fixed-size Poisson per step, neither conditioned on spikes. If anyone introduces
a spike-dependent RNG call, changing the threshold field would desynchronise the
noise between arms and the paired-variance probe would become meaningless -- this
test must fail then.
"""
import sys
import numpy as np
import pytest

sys.path.insert(0, "src/snn_engine")
sys.path.insert(0, ".")


@pytest.mark.integration
@pytest.mark.slow
def test_changing_the_threshold_field_does_not_change_the_noise_stream(tmp_path):
    from params import Params
    from kick_probe import simulate_kick
    from src.topic4_core_field_runner import get_network

    p = Params(g=3.6, L=6.0, density=40.0, T=200.0, dt=0.1, nu_ext_ratio=1.0, seed=5)
    net, NE, NI, _ = get_network(p, -22.8, 2.0, str(tmp_path))
    N = NE + NI

    class CountingGenerator(np.random.Generator):
        """Generator attributes are read-only, so counting needs a subclass."""

        def __init__(self, bit_generator):
            super().__init__(bit_generator)
            self.calls = {"normal": 0, "poisson": 0}

        def standard_normal(self, *args, **kwargs):
            self.calls["normal"] += 1
            return super().standard_normal(*args, **kwargs)

        def poisson(self, *args, **kwargs):
            self.calls["poisson"] += 1
            return super().poisson(*args, **kwargs)

    draws = {}

    def run(vth, label):
        gen = CountingGenerator(np.random.PCG64(5))
        net["rng"] = gen
        simulate_kick(p, net, KICK_BOOST=0.0, kick_center=[3.0, 3.0], r_kick=1.0,
                      t_kick=1e9, V_th_per_neuron=vth)
        draws[label] = dict(gen.calls)

    run(np.full(N, 18.0), "flat")
    lowered = np.full(N, 18.0)
    lowered[: NE // 4] = 16.0
    run(lowered, "lowered")

    assert draws["flat"] == draws["lowered"], (
        "RNG call counts diverged between threshold fields: common random numbers "
        "no longer hold, so the Stage 1 paired probe is invalid"
    )


@pytest.mark.integration
@pytest.mark.slow
def test_two_arms_at_one_seed_start_from_identical_state(tmp_path):
    """No state is inherited between arms: rebuilding the rng at the same seed
    must reproduce the run bit for bit."""
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
git commit -m "test(topic4-core-field): lock common random numbers across arms

Verified the loop's only RNG calls are unconditional and fixed-size, so the same
seed gives the same noise whatever the threshold field. This test fails the day
that stops being true, which is the day the paired probe stops meaning anything."
```

---

### Task 8: Stage 1 裁决器（纯函数、fail-closed）

**重读 spec：** §7.4、§7.5

**Files:**
- Create: `src/topic4_core_field_verdict.py`
- Test: `tests/test_topic4_core_field_verdict.py`

**Interfaces:**
- Produces: `PRESPECIFIED_COMPARISONS`；`paired_stats(delta) -> dict`；`holm(pvals) -> list[bool]`；`concordance(per_seed_signs) -> float`；`stage1_verdict(runs, config_checksum, expected_checksum) -> dict`

`runs` 结构（Task 10 产出）：`{(arm, seed, source, missing_rule, gate): {"n_dir": int, "S_rank": float, "coverage": float}}`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_topic4_core_field_verdict.py
import numpy as np
import pytest
from src.topic4_core_field_verdict import (
    PRESPECIFIED_COMPARISONS, concordance, holm, paired_stats, stage1_verdict,
)

ARMS = ["manual_hard", "manual_projected", "uniform_axial", "width_wide",
        "width_narrow", "transverse_plus", "transverse_minus"]
SEEDS = list(range(1, 13))
KEYS = [("gradient", "mean_rank", 5), ("gradient", "common_only", 5),
        ("gradient", "mean_rank", 4), ("gradient", "common_only", 4),
        ("geometry", "mean_rank", 5), ("geometry", "common_only", 5),
        ("geometry", "mean_rank", 4), ("geometry", "common_only", 4)]
CHECKSUM = "abc123"


def _runs(separable=True, n_dir=2, coverage=0.9):
    """manual_projected beats every shape arm at every seed when separable."""
    out = {}
    for arm in ARMS:
        for seed in SEEDS:
            base = 0.80 if arm in ("manual_projected", "manual_hard") else 0.60
            if not separable:
                base = 0.70
            # Deterministic jitter: hash(str) varies with PYTHONHASHSEED.
            jitter = 0.001 * ((seed * 7 + ARMS.index(arm)) % 3)
            for key in KEYS:
                out[(arm, seed) + key] = {
                    "n_dir": n_dir, "S_rank": base + jitter, "coverage": coverage}
    return out


def test_paired_stats_reports_sign_agreement_and_a_two_sided_p():
    st = paired_stats(np.array([0.1] * 12))
    assert st["n_same"] == 12
    assert st["p"] < 0.001
    st2 = paired_stats(np.array([0.1] * 6 + [-0.1] * 6))
    assert st2["p"] == pytest.approx(1.0)


def test_holm_is_more_conservative_than_uncorrected():
    flags = holm([0.001, 0.02, 0.04, 0.9])
    assert flags[0] is True
    assert flags[3] is False


def test_prespecified_comparisons_exclude_the_equivalence_and_geometry_pairs():
    """P0-2: identifiability may not be decided by the sampling-contract pair or
    by any axis_only pair."""
    gated = [c for c in PRESPECIFIED_COMPARISONS if c["gates_identifiability"]]
    assert {c["name"] for c in gated} == {"B1", "B2", "B3", "B4"}
    for c in gated:
        assert "manual_hard" not in (c["a"], c["b"])
        assert "axis_only" not in (c["a"], c["b"])


def test_separable_arms_yield_a_go_verdict():
    v = stage1_verdict(_runs(separable=True), CHECKSUM, CHECKSUM)
    assert v["verdict"] in ("GO_SINGLE_SEED", "GO_MULTI_SEED")


def test_indistinguishable_arms_yield_readout_insensitive():
    v = stage1_verdict(_runs(separable=False), CHECKSUM, CHECKSUM)
    assert v["verdict"] == "READOUT_INSENSITIVE"


def test_checksum_mismatch_fails_closed():
    v = stage1_verdict(_runs(), "wrong", CHECKSUM)
    assert v["verdict"] == "FAIL_CLOSED"


def test_a_nan_score_fails_closed():
    runs = _runs()
    runs[("uniform_axial", 3) + KEYS[0]]["S_rank"] = float("nan")
    v = stage1_verdict(runs, CHECKSUM, CHECKSUM)
    assert v["verdict"] == "FAIL_CLOSED"


def test_too_many_uninformative_seeds_stops_the_probe():
    runs = _runs()
    for seed in (1, 2, 3, 4):
        for arm in ("uniform_axial", "width_wide"):
            for key in KEYS:
                runs[(arm, seed) + key]["n_dir"] = 0
    v = stage1_verdict(runs, CHECKSUM, CHECKSUM)
    assert v["verdict"] == "INSUFFICIENT_SCORABLE"


def test_disagreement_between_template_sources_stops_the_probe():
    runs = _runs(separable=True)
    for arm in ARMS:
        for seed in SEEDS:
            runs[(arm, seed, "geometry", "mean_rank", 5)]["S_rank"] = 0.70
    v = stage1_verdict(runs, CHECKSUM, CHECKSUM)
    assert v["verdict"] == "SOURCE_DISAGREEMENT"


def test_verdict_is_a_pure_function():
    runs = _runs()
    snapshot = {k: dict(val) for k, val in runs.items()}
    stage1_verdict(runs, CHECKSUM, CHECKSUM)
    assert runs == snapshot


def test_transverse_sign_flip_does_not_change_the_verdict():
    """u_perp's sign is a convention; swapping the two transverse arms must not
    move the verdict (spec P0-4)."""
    runs = _runs(separable=True)
    flipped = {}
    swap = {"transverse_plus": "transverse_minus", "transverse_minus": "transverse_plus"}
    for k, v in runs.items():
        arm = swap.get(k[0], k[0])
        flipped[(arm,) + k[1:]] = dict(v)
    assert stage1_verdict(flipped, CHECKSUM, CHECKSUM)["verdict"] == \
           stage1_verdict(runs, CHECKSUM, CHECKSUM)["verdict"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_topic4_core_field_verdict.py -x -q`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 最小实现**

```python
# src/topic4_core_field_verdict.py
"""Stage 1 gate. Pure function, fail-closed (spec section 7.5).

Nine exits, evaluated in order. Any NaN, missing cell, or checksum mismatch
returns FAIL_CLOSED rather than continuing on partial evidence.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import binomtest

SEEDS = tuple(range(1, 13))
SIM_ARMS = ("manual_hard", "manual_projected", "uniform_axial", "width_wide",
            "width_narrow", "transverse_plus", "transverse_minus")
# Arms the optimiser will actually navigate: same sampling contract, real noise.
PROJECTED_ARMS = ("manual_projected", "uniform_axial", "width_wide",
                  "width_narrow", "transverse_mean")
KEYS = tuple(itertools.product(("gradient", "geometry"),
                               ("mean_rank", "common_only"), (5, 4)))

PRESPECIFIED_COMPARISONS = (
    dict(name="A", a="manual_projected", b="manual_hard",
         purpose="parameterisation equivalence", gates_identifiability=False),
    dict(name="B1", a="manual_projected", b="uniform_axial",
         purpose="longitudinal shape", gates_identifiability=True),
    dict(name="B2", a="manual_projected", b="width_wide",
         purpose="transverse width (flattened)", gates_identifiability=True),
    dict(name="B3", a="manual_projected", b="width_narrow",
         purpose="transverse width (elongated)", gates_identifiability=True),
    dict(name="B4", a="manual_projected", b="transverse_mean",
         purpose="transverse position", gates_identifiability=True),
)

CONCORDANCE_GO_SINGLE = 0.85
CONCORDANCE_GO_MULTI = 0.65
MAX_UNINFORMATIVE_SEEDS = 3          # >= 4 of 12 stops the probe


def paired_stats(delta):
    delta = np.asarray(delta, float)
    nonzero = delta[delta != 0.0]
    mean = float(delta.mean())
    sign = np.sign(mean) if mean != 0 else 1.0
    n_same = int((np.sign(nonzero) == sign).sum())
    n = int(nonzero.size)
    p = float(binomtest(n_same, n, 0.5, alternative="two-sided").pvalue) if n else 1.0
    return dict(mean=mean, sd=float(delta.std(ddof=1)) if delta.size > 1 else 0.0,
                n_same=n_same, n=n, p=p)


def holm(pvals, alpha=0.05):
    order = np.argsort(pvals)
    m = len(pvals)
    flags = [False] * m
    for rank, idx in enumerate(order):
        if pvals[idx] <= alpha / (m - rank):
            flags[idx] = True
        else:
            break
    return flags


def concordance(per_seed_signs, pooled_signs):
    hits = [1.0 if per_seed_signs[(pair, seed)] == pooled_signs[pair] else 0.0
            for pair in pooled_signs for seed in SEEDS]
    return float(np.mean(hits)) if hits else 0.0


def _arm_value(runs, arm, seed, key, field):
    if arm == "transverse_mean":
        vals = [runs[("transverse_plus", seed) + key][field],
                runs[("transverse_minus", seed) + key][field]]
        return float(np.mean(vals))
    return runs[(arm, seed) + key][field]


def _verdict_for_key(runs, key):
    """One (source, missing_rule, gate) combination -> (identifiability, concordance)."""
    stats = {}
    for comp in PRESPECIFIED_COMPARISONS:
        delta = np.array([_arm_value(runs, comp["a"], s, key, "S_rank")
                          - _arm_value(runs, comp["b"], s, key, "S_rank") for s in SEEDS])
        stats[comp["name"]] = paired_stats(delta)

    gated = [c["name"] for c in PRESPECIFIED_COMPARISONS if c["gates_identifiability"]]
    flags = holm([stats[n]["p"] for n in gated])
    identifiable = any(flags)
    suggestive = any(stats[n]["p"] < 0.05 for n in gated)

    pooled, per_seed = {}, {}
    for a, b in itertools.combinations(PROJECTED_ARMS, 2):
        pair = f"{a}|{b}"
        deltas = {s: _arm_value(runs, a, s, key, "S_rank")
                     - _arm_value(runs, b, s, key, "S_rank") for s in SEEDS}
        m = float(np.mean(list(deltas.values())))
        pooled[pair] = np.sign(m) if m != 0 else 1.0
        for s in SEEDS:
            per_seed[(pair, s)] = np.sign(deltas[s]) if deltas[s] != 0 else 1.0
    conc = concordance(per_seed, pooled)
    return dict(stats=stats, holm=dict(zip(gated, flags)),
                identifiable=identifiable, suggestive=suggestive, concordance=conc)


def stage1_verdict(runs, config_checksum, expected_checksum):
    if config_checksum != expected_checksum:
        return dict(verdict="FAIL_CLOSED", reason="config checksum mismatch")

    for arm in SIM_ARMS:
        for seed in SEEDS:
            for key in KEYS:
                cell = runs.get((arm, seed) + key)
                if cell is None:
                    return dict(verdict="FAIL_CLOSED",
                                reason=f"missing cell {(arm, seed) + key}")
                if cell["n_dir"] > 0 and not np.isfinite(cell["S_rank"]):
                    return dict(verdict="FAIL_CLOSED",
                                reason=f"non-finite S_rank at {(arm, seed) + key}")

    ref = KEYS[0]
    uninformative = sum(
        1 for seed in SEEDS
        if sum(1 for arm in SIM_ARMS if runs[(arm, seed) + ref]["n_dir"] == 0) >= 2)
    if uninformative > MAX_UNINFORMATIVE_SEEDS:
        return dict(verdict="INSUFFICIENT_SCORABLE", uninformative_seeds=uninformative)

    per_key = {key: _verdict_for_key(runs, key) for key in KEYS}

    equiv = per_key[ref]["stats"]["A"]
    if equiv["p"] < 0.05 and abs(equiv["mean"]) > equiv["sd"]:
        return dict(verdict="PARAMETERIZATION_MISMATCH", equivalence=equiv,
                    per_key=per_key)

    ident = {k: v["identifiable"] for k, v in per_key.items()}
    def tier(c):
        return ("GO_SINGLE_SEED" if c >= CONCORDANCE_GO_SINGLE
                else "GO_MULTI_SEED" if c >= CONCORDANCE_GO_MULTI
                else "NO_GO_UNRESOLVABLE")
    tiers = {k: tier(v["concordance"]) for k, v in per_key.items()}
    if len(set(ident.values())) > 1 or len(set(tiers.values())) > 1:
        return dict(verdict="SOURCE_DISAGREEMENT", identifiable=ident, tiers=tiers,
                    per_key=per_key)

    if not per_key[ref]["identifiable"]:
        verdict = ("UNDERPOWERED_PROBE" if per_key[ref]["suggestive"]
                   else "READOUT_INSENSITIVE")
        return dict(verdict=verdict, per_key=per_key)

    return dict(verdict=tiers[ref], concordance=per_key[ref]["concordance"],
                per_key=per_key)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_topic4_core_field_verdict.py -q`
Expected: 11 passed

- [ ] **Step 5: 提交**

```bash
git add src/topic4_core_field_verdict.py tests/test_topic4_core_field_verdict.py
git commit -m "feat(topic4-core-field): fail-closed Stage 1 gate on named comparisons

Identifiability is decided only by the four pre-registered shape comparisons
under Holm, never by the sampling-contract pair or an axis-only pair, and never
by whichever of many pairs happened to separate."
```

---

### Task 9: Stage 0 —— 三个参照与冻结 config

**重读 spec：** §6.1、§2.4、§5.2a

**Files:**
- Create: `scripts/run_topic4_core_field_stage0.py`

**Interfaces:**
- Consumes: Task 4 的打分模块
- Produces: `results/topic4_sef_hfo/data_driven_core_field/config/stage_config.json`（含 `SUPPORT`、`quantile_seed`、`N_core_manual`、`D0`、`checksum`）、`model_integrity_report.md`、`reference_scores.csv`

- [ ] **Step 1: 写脚本**

```python
# scripts/run_topic4_core_field_stage0.py
"""Stage 0: reproduce the baseline, rescore the three references with ONE frozen
scorer, and freeze the config every later stage is checksummed against.

The three references live in different regimes and are not interchangeable:
  axis_only        -- no pathology field at all, contacts ordered by axial projection
  manual spontaneous -- the two-core network running free (this is the baseline)
  driven_pooled    -- source-only + sink-only arms pooled; a READ-OUT UPPER REFERENCE,
                      not a baseline (its frozen stats file says so in as many words)
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from src.topic4_core_field_scoring import (  # noqa: E402
    assignment_invariant_S, axis_only_templates, load_patient_templates,
    model_templates, pair_score, sim_matrix,
)

OUT = "results/topic4_sef_hfo/data_driven_core_field"
RUN = "results/topic4_sef_hfo/field_swap_subject_snn"
SUBJECT = "epilepsiae_1146"
QUANTILE_SEED = 20260806
GATES = (5, 4)
MISSING_RULES = ("mean_rank", "common_only")
SOURCES = ("gradient", "geometry")


def _score(model, targets, support):
    row = {}
    for src in SOURCES:
        for rule in MISSING_RULES:
            M = sim_matrix(model, targets[src], support, rule)
            row[f"S_{src}_{rule}"] = assignment_invariant_S(M)
        row[f"Spair_{src}"] = pair_score(model, targets[src], support)
    row["n_dir"] = model["n_dir"]
    row["coverage"] = model["coverage"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "config"), exist_ok=True)

    targets = {s: load_patient_templates(SUBJECT, s) for s in SOURCES}
    support = sorted(set(targets["gradient"]["t_a"]) & set(targets["gradient"]["t_b"])
                     & set(targets["geometry"]["t_a"]) & set(targets["geometry"]["t_b"]))
    print(f"[stage0] frozen scoring support: {len(support)} contacts -> {support}")

    rows = []

    # --- axis_only: analytic, no simulation -------------------------------
    fd = np.load(os.path.join(
        RUN, f"figdata_{SUBJECT}_gradient_shared_corefrozen_cr1p5_s5_20260722.npz"),
        allow_pickle=True)
    names = [str(x) for x in fd["names"]]
    coords = np.asarray(fd["contacts"], float)
    reg = fd["reg"].item()
    ao = axis_only_templates(names, coords, np.asarray(reg["center"]),
                             np.asarray(reg["axis_unit"]))
    rows.append(dict(reference="axis_only", regime="analytic", tag="-", **_score(ao, targets, support)))

    # --- spontaneous two-core (THE baseline) and driven_pooled ------------
    for path in sorted(glob.glob(os.path.join(RUN, "readout_*.json"))):
        ro = json.load(open(path))
        if ro.get("subject") != SUBJECT:
            continue
        lesion, placement = ro.get("lesion"), ro.get("placement")
        if lesion == "twoend_equal" and placement in ("gradient_shared", "template_source"):
            regime = f"spontaneous_two_core_{placement}"
        elif lesion == "driven_pooled":
            regime = "driven_pooled_upper_reference"
        else:
            continue
        for gate in GATES:
            m = model_templates(ro["events"], support, part_min=gate)
            rows.append(dict(reference=regime, regime=regime, gate=gate,
                             tag=os.path.basename(path), **_score(m, targets, support)))

    import csv
    csv_path = os.path.join(a.out, "reference_scores.csv")
    keys = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[stage0] wrote {csv_path} ({len(rows)} rows)")

    def summarise(prefix, gate=None):
        vals = [r["S_gradient_mean_rank"] for r in rows
                if r["reference"].startswith(prefix)
                and (gate is None or r.get("gate") == gate)
                and r["n_dir"] == 2 and np.isfinite(r["S_gradient_mean_rank"])]
        if not vals:
            return "n/a"
        return f"{np.mean(vals):.3f} +/- {np.std(vals, ddof=1) if len(vals) > 1 else 0:.3f} (n={len(vals)})"

    cfg = dict(
        subject=SUBJECT, support=support, quantile_seed=QUANTILE_SEED,
        gates=list(GATES), missing_rules=list(MISSING_RULES), sources=list(SOURCES),
        seeds=list(range(1, 13)), duration_ms=8000.0,
        field=dict(M=9, EPS=1e-3, TAU_H=0.25, A0=1.5, B0=1.5,
                   SIGMA_S_FACTOR=1.2, AXIAL_MARGIN=2.0),
        engine=dict(L=20.0, density=100.0, AR=2.0, g=3.6, dt=0.1, k_dir=2,
                    core_mean=17.5, core_std=1.0, core_r=1.5, v_base=18.0),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
    )
    blob = json.dumps(cfg, sort_keys=True).encode()
    cfg["checksum"] = hashlib.sha256(blob).hexdigest()
    cfg_path = os.path.join(a.out, "config", "stage_config.json")
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    print(f"[stage0] froze config -> {cfg_path} checksum={cfg['checksum'][:12]}")

    report = f"""# Stage 0 完整性报告 — {SUBJECT}

## 网络规模（从代码读，不从手稿读）
- `N = round(density * L^2) = 100 * 20 * 20 = 40000`，`N_E = 32000`、`N_I = 8000`
- 手稿若写 `N = 4000`，是手稿错

## 冻结的打分支撑集
{len(support)} 个触点：{support}

## 三个参照分数（同一冻结 scorer，`S_gradient_mean_rank`）
| 参照 | 是什么 | 分数 |
|---|---|---|
| `axis_only` | 完全没有病理场，只按触点在 u_C 上的投影排序 | {rows[0]['S_gradient_mean_rank']:.3f} |
| 自发双核（gradient_shared） | **这才是基线** | {summarise('spontaneous_two_core_gradient_shared', 5)} |
| 自发双核（template_source，旧几何） | 参考 | {summarise('spontaneous_two_core_template_source', 5)} |
| `driven_pooled` | **读出上参照，不是基线** | {summarise('driven_pooled', 5)} |

`driven_pooled` 的冻结统计文件逐字写着
`"independent_unit": "paired network seed (source-only and sink-only arms)"`。

## 未提交改动
见 `preexisting_worktree.patch`；含参与门 `2*k_dir` -> `2*k_dir+1` 的行为改动。

## 冻结 config
`config/stage_config.json`，checksum `{cfg['checksum']}`
"""
    open(os.path.join(a.out, "model_integrity_report.md"), "w").write(report)
    print(f"[stage0] wrote model_integrity_report.md")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑 Stage 0**

Run: `python scripts/run_topic4_core_field_stage0.py`
Expected: 打印支撑集触点数（预期 15）、写出 `reference_scores.csv` / `stage_config.json` / `model_integrity_report.md`

- [ ] **Step 3: 目视核对三个参照**

Run: `cat results/topic4_sef_hfo/data_driven_core_field/model_integrity_report.md`
Expected: `axis_only` ≈ 0.696；自发双核与它**接近**（这正是 Stage 1 存在的理由）；`driven_pooled` 明显更高且被标注为上参照。若自发双核**远高于** `axis_only`，说明冻结 scorer 与 §2.4 的算法有出入，**停下来查**，不要进 Stage 1。

- [ ] **Step 4: 提交**

```bash
git add scripts/run_topic4_core_field_stage0.py \
        results/topic4_sef_hfo/data_driven_core_field/config/stage_config.json \
        results/topic4_sef_hfo/data_driven_core_field/model_integrity_report.md \
        results/topic4_sef_hfo/data_driven_core_field/reference_scores.csv
git commit -m "feat(topic4-core-field): Stage 0 rescores the three references under one scorer

Separates the spontaneous two-core baseline from the driven-pooled read-out
upper reference, which an earlier draft had conflated, and freezes the scoring
support and config checksum every later stage is checked against."
```

---

### Task 10: Stage 1 —— 84 次配对探针

**重读 spec：** §7.2、§4.4（pre-flight）

**Files:**
- Create: `scripts/run_topic4_core_field_stage1.py`
- Modify: `src/topic4_core_field_runner.py`（加 `run_arm`）

**Interfaces:**
- Consumes: Task 3 `arm_h` / `preflight_arm_distinctness`、Task 6 `get_network`、Task 4 打分
- Produces: `stage1_variance_probe/per_run.jsonl`（每行一个 (arm, seed) 结果，支持 resume）

- [ ] **Step 1: 在 runner 里加 `run_arm`**

```python
# append to src/topic4_core_field_runner.py
def run_arm(arm, seed, cfg, cache_dir):
    """One (arm, seed) simulation, returning the event table plus counts.

    Reuses the existing read-out chain (envelope -> detect_events -> read_event)
    exactly as scripts/run_sef_hfo_subject_snn.py does; nothing about detection
    or direction reading is redefined here.
    """
    import importlib.util
    import sys
    eng = os.path.join("src", "snn_engine")
    for path in (eng, os.getcwd()):
        if path not in sys.path:
            sys.path.insert(0, path)
    from params import Params
    from kick_probe import simulate_kick
    from lfp import LFPRecorder
    from src.sef_hfo_events import detect_events
    from src.sef_hfo_heterogeneity import sample_core_field
    from src.sef_hfo_snn_adapter import snn_event_envelope
    from src.sef_hfo_subject_placement import (
        gradient_shared_template_foci, register_to_sheet, template_source_foci)
    from src.topic4_core_field import (
        arm_h, axis_coords, build_vth, core_thresholds,
        sample_core_quantiles, signed_depth)

    spec = importlib.util.spec_from_file_location(
        "cmrun", os.path.join("scripts", "run_sef_hfo_snn_cm_spontaneous_readout.py"))
    cmrun = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmrun)
    k_dir = int(cfg["engine"]["k_dir"])
    cmrun.KDIR, cmrun.PART_MIN = k_dir, 2 * k_dir + 1

    m_real, _, _, _ = gradient_shared_template_foci(cfg["subject"], 3)
    _, src_names, snk_names = template_source_foci(cfg["subject"], "narrow", 3)
    reg = register_to_sheet(m_real, src_names, snk_names,
                            L=cfg["engine"]["L"], target_inter_core_mm=None)
    msheet = reg["montage_sheet"]
    src_xy, snk_xy = reg["source_centroid"], reg["sink_centroid"]
    axis_unit = (snk_xy - src_xy) / np.linalg.norm(snk_xy - src_xy)

    e = cfg["engine"]
    p = Params(g=e["g"], L=e["L"], density=e["density"], T=cfg["duration_ms"],
               dt=e["dt"], nu_ext_ratio=cmrun.DRIVE, seed=seed)
    net, NE, NI, _ = get_network(p, reg["theta_deg"], e["AR"], cache_dir)
    posE = net["pos"][:NE]
    is_E = np.zeros(len(net["pos"]), bool); is_E[:NE] = True

    if arm == "manual_hard":
        cf1 = sample_core_field(net["pos"], is_E, src_xy, e["core_r"],
                                np.random.default_rng(seed + 7),
                                core_mean=e["core_mean"], core_std=e["core_std"],
                                base_mean=e["v_base"])
        cf2 = sample_core_field(net["pos"], is_E, snk_xy, e["core_r"],
                                np.random.default_rng(seed + 8),
                                core_mean=e["core_mean"], core_std=e["core_std"],
                                base_mean=e["v_base"])
        vth = np.minimum(cf1["vth"], cf2["vth"])
        h = (cf1["core_mask"] | cf2["core_mask"])[:NE].astype(float)
    else:
        center = reg["center"]
        s, r = axis_coords(posE, center, axis_unit)
        geom = dict(sep=float(np.linalg.norm(snk_xy - src_xy)),
                    s_support=(float(s.min()) + cfg["field"]["AXIAL_MARGIN"],
                               float(s.max()) - cfg["field"]["AXIAL_MARGIN"]),
                    M=cfg["field"]["M"], sigma_perp=e["core_r"], shift_mm=3.0)
        manual_mask = (np.minimum(((posE - src_xy) ** 2).sum(1),
                                  ((posE - snk_xy) ** 2).sum(1)) <= e["core_r"] ** 2)
        h = arm_h(arm, s, r, geom, target_count=float(manual_mask.sum()))
        u = sample_core_quantiles(NE, cfg["quantile_seed"])
        d = signed_depth(core_thresholds(u, e["core_mean"], e["core_std"]), e["v_base"])
        vth = build_vth(h, d, n_total=NE + NI, n_E=NE, v_base=e["v_base"])

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
        recs.append(dict(n_part=rd["n_part"], sign=rd["sign"], ranks=rd["ranks"]))
    return dict(arm=arm, seed=int(seed), events=recs,
                h_sum=float(h.sum()), n_events=len(recs))
```

- [ ] **Step 2: 写 Stage 1 驱动脚本**

```python
# scripts/run_topic4_core_field_stage1.py
"""Stage 1: 7 arms x 12 seeds paired-variance probe.

Refuses to launch if any two arms build the same field -- that check costs
milliseconds and the run costs an hour.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.getcwd())
from src.topic4_core_field import (  # noqa: E402
    ARM_NAMES, arm_h, axis_coords, preflight_arm_distinctness)
from src.topic4_core_field_runner import run_arm  # noqa: E402

OUT = "results/topic4_sef_hfo/data_driven_core_field"


def _preflight(cfg):
    """Build every arm's h on the real sheet geometry and assert distinctness."""
    from src.sef_hfo_subject_placement import (
        gradient_shared_template_foci, register_to_sheet, template_source_foci)
    sys.path.insert(0, os.path.join("src", "snn_engine"))
    from connectivity import place_neurons
    from params import Params

    e = cfg["engine"]
    m_real, _, _, _ = gradient_shared_template_foci(cfg["subject"], 3)
    _, src, snk = template_source_foci(cfg["subject"], "narrow", 3)
    reg = register_to_sheet(m_real, src, snk, L=e["L"], target_inter_core_mm=None)
    p = Params(g=e["g"], L=e["L"], density=e["density"], T=100.0, dt=e["dt"], seed=1)
    pos, _, NE, _ = place_neurons(p, np.random.default_rng(1))
    posE = pos[:NE]
    axis_unit = (reg["sink_centroid"] - reg["source_centroid"])
    axis_unit = axis_unit / np.linalg.norm(axis_unit)
    s, r = axis_coords(posE, reg["center"], axis_unit)
    geom = dict(sep=float(np.linalg.norm(reg["sink_centroid"] - reg["source_centroid"])),
                s_support=(float(s.min()) + cfg["field"]["AXIAL_MARGIN"],
                           float(s.max()) - cfg["field"]["AXIAL_MARGIN"]),
                M=cfg["field"]["M"], sigma_perp=e["core_r"], shift_mm=3.0)
    manual_mask = (np.minimum(((posE - reg["source_centroid"]) ** 2).sum(1),
                              ((posE - reg["sink_centroid"]) ** 2).sum(1)) <= e["core_r"] ** 2)
    target = float(manual_mask.sum())
    h_by_arm = {a: arm_h(a, s, r, geom, target) for a in ARM_NAMES if a != "manual_hard"}
    h_by_arm["manual_hard"] = manual_mask.astype(float)
    return preflight_arm_distinctness(h_by_arm), target


def _job(args):
    arm, seed, cfg, cache_dir = args
    try:
        return run_arm(arm, seed, cfg, cache_dir)
    except Exception as exc:                      # loud, never silent
        return dict(arm=arm, seed=int(seed), error=f"{type(exc).__name__}: {exc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.out, "config", "stage_config.json")))
    probe_dir = os.path.join(a.out, "stage1_variance_probe")
    cache_dir = os.path.join(a.out, "network_cache")
    os.makedirs(probe_dir, exist_ok=True)

    report, target = _preflight(cfg)
    json.dump(dict(preflight=report, target_count=target),
              open(os.path.join(probe_dir, "preflight.json"), "w"), indent=2)
    if not report["ok"]:
        print("[stage1] PREFLIGHT FAILED -- arms are not distinct:", report["violations"])
        print("[stage1] refusing to launch 84 simulations on a vacuous comparison")
        return 1
    print(f"[stage1] preflight OK; budget N_core_manual = {target:.0f}")

    jsonl = os.path.join(probe_dir, "per_run.jsonl")
    done = set()
    if os.path.exists(jsonl):
        for line in open(jsonl):
            rec = json.loads(line)
            if "error" not in rec:
                done.add((rec["arm"], rec["seed"]))
    todo = [(arm, seed, cfg, cache_dir)
            for seed in cfg["seeds"] for arm in ARM_NAMES
            if (arm, seed) not in done]
    print(f"[stage1] {len(done)} done, {len(todo)} to run, {a.workers} workers")

    with open(jsonl, "a") as fh, Pool(a.workers, maxtasksperchild=1) as pool:
        for i, rec in enumerate(pool.imap_unordered(_job, todo), 1):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            tag = rec.get("error", f"events={rec.get('n_events')}")
            print(f"[stage1] {i}/{len(todo)} {rec['arm']} s{rec['seed']} {tag}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: 先只跑 pre-flight（不开仿真）**

Run: `python -c "
import json,sys; sys.path.insert(0,'.')
sys.argv=['x','--workers','1']
import importlib.util
spec=importlib.util.spec_from_file_location('s1','scripts/run_topic4_core_field_stage1.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
cfg=json.load(open('results/topic4_sef_hfo/data_driven_core_field/config/stage_config.json'))
rep,t=m._preflight(cfg); print('ok=',rep['ok']); print('budget=',t)
print(json.dumps(rep['correlations'],indent=1))"`

Expected: `ok= True`；`budget` ≈ 1100–1200；所有两两相关 < 0.95。若有任一对 ≥ 0.95，**停下来**修参数化，不要跑仿真。

- [ ] **Step 4: 跑 Stage 1（约 1 小时）**

Run: `python scripts/run_topic4_core_field_stage1.py --workers 8`
Expected: 84 行写入 `per_run.jsonl`；无 `error` 行。中断可直接重跑，已完成的会跳过。

- [ ] **Step 5: 提交**

```bash
git add scripts/run_topic4_core_field_stage1.py src/topic4_core_field_runner.py
git commit -m "feat(topic4-core-field): Stage 1 paired probe over seven arms and twelve seeds

Every arm at a seed runs on the same cached network and the same noise seed, so
the comparison is paired. A pre-flight refuses to launch when two arms build the
same field."
```

---

### Task 11: Stage 1 分析、裁决与图

**重读 spec：** §7.3、§7.4、§7.6

**Files:**
- Create: `scripts/analyze_topic4_core_field_stage1.py`

**Interfaces:**
- Consumes: Task 10 的 `per_run.jsonl`、Task 8 的 `stage1_verdict`、Task 4 的打分
- Produces: `per_run.csv`、`prespecified_comparisons.csv`、`concordance.csv`、`gate_verdict.json`、`figures/*.pdf` + `figures/README.md`

- [ ] **Step 1: 写分析脚本**

```python
# scripts/analyze_topic4_core_field_stage1.py
"""Score every Stage 1 run under all eight scoring combinations, run the
pre-registered comparisons, and hand the whole table to the fail-closed gate.
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
from src.topic4_core_field_scoring import (  # noqa: E402
    assignment_invariant_S, load_patient_templates, model_templates,
    pair_score, sim_matrix)
from src.topic4_core_field_verdict import (  # noqa: E402
    KEYS, PRESPECIFIED_COMPARISONS, SEEDS, SIM_ARMS, _arm_value,
    paired_stats, stage1_verdict)

OUT = "results/topic4_sef_hfo/data_driven_core_field"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    probe = os.path.join(a.out, "stage1_variance_probe")
    fig_dir = os.path.join(probe, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    cfg = json.load(open(os.path.join(a.out, "config", "stage_config.json")))
    support = cfg["support"]
    targets = {s: load_patient_templates(cfg["subject"], s) for s in cfg["sources"]}

    runs, rows = {}, []
    for line in open(os.path.join(probe, "per_run.jsonl")):
        rec = json.loads(line)
        if "error" in rec:
            raise SystemExit(f"per_run.jsonl contains a failed run: {rec}")
        for src in cfg["sources"]:
            for rule in cfg["missing_rules"]:
                for gate in cfg["gates"]:
                    m = model_templates(rec["events"], support, part_min=gate)
                    S = assignment_invariant_S(sim_matrix(m, targets[src], support, rule))
                    runs[(rec["arm"], rec["seed"], src, rule, gate)] = dict(
                        n_dir=m["n_dir"], S_rank=S, coverage=m["coverage"])
                    rows.append(dict(arm=rec["arm"], seed=rec["seed"], source=src,
                                     missing_rule=rule, gate=gate, n_dir=m["n_dir"],
                                     S_rank=S, coverage=m["coverage"],
                                     S_pair=pair_score(m, targets[src], support),
                                     n_events=rec["n_events"], h_sum=rec["h_sum"]))

    with open(os.path.join(probe, "per_run.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(rows[0]))
        w.writeheader(); w.writerows(rows)

    comp_rows = []
    for key in KEYS:
        for comp in PRESPECIFIED_COMPARISONS:
            delta = np.array([_arm_value(runs, comp["a"], s, key, "S_rank")
                              - _arm_value(runs, comp["b"], s, key, "S_rank") for s in SEEDS])
            st = paired_stats(delta)
            comp_rows.append(dict(comparison=comp["name"], purpose=comp["purpose"],
                                  gates=comp["gates_identifiability"],
                                  source=key[0], missing_rule=key[1], gate=key[2],
                                  **st))
    with open(os.path.join(probe, "prespecified_comparisons.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(comp_rows[0]))
        w.writeheader(); w.writerows(comp_rows)

    verdict = stage1_verdict(runs, cfg["checksum"], cfg["checksum"])
    json.dump(verdict, open(os.path.join(probe, "gate_verdict.json"), "w"),
              indent=2, default=str)
    print(f"[stage1] VERDICT = {verdict['verdict']}")

    ref = KEYS[0]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, arm in enumerate(SIM_ARMS):
        vals = [runs[(arm, s) + ref]["S_rank"] for s in SEEDS]
        ax.scatter(np.full(len(vals), i) + np.random.default_rng(0).uniform(-.12, .12, len(vals)),
                   vals, s=22, alpha=.75)
        ax.hlines(np.nanmean(vals), i - .28, i + .28, lw=2.2, color="k")
    ax.set_xticks(range(len(SIM_ARMS)))
    ax.set_xticklabels(SIM_ARMS, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("assignment-invariant rank match")
    ax.set_title("Stage 1 arms, 12 paired network seeds")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "stage1_arm_scores.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    gated = [c for c in PRESPECIFIED_COMPARISONS if c["gates_identifiability"]]
    for i, comp in enumerate(gated):
        delta = [_arm_value(runs, comp["a"], s, ref, "S_rank")
                 - _arm_value(runs, comp["b"], s, ref, "S_rank") for s in SEEDS]
        ax.scatter(np.full(len(delta), i), delta, s=22, alpha=.75)
        ax.hlines(np.mean(delta), i - .25, i + .25, lw=2.2, color="k")
    ax.axhline(0, color="0.6", lw=.8)
    ax.set_xticks(range(len(gated)))
    ax.set_xticklabels([f"{c['name']}\n{c['purpose']}" for c in gated], fontsize=7.5)
    ax.set_ylabel("paired difference vs manual_projected")
    ax.set_title("Pre-registered shape comparisons")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "stage1_prespecified_deltas.pdf"))
    plt.close(fig)
    print(f"[stage1] wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑分析**

Run: `python scripts/analyze_topic4_core_field_stage1.py`
Expected: 打印 `VERDICT = ...`；写出 4 个数据文件 + 2 张图

- [ ] **Step 3: 目视检查两张图**

打开 `stage1_arm_scores.pdf` 与 `stage1_prespecified_deltas.pdf`。检查：七个臂的散点是否分离；配对差是否跨种子同号。**图必须自己看过再写 README。**

- [ ] **Step 4: 写 `figures/README.md`（中文，图看过之后写）**

```markdown
# Stage 1 配对方差探针 — 图说明

### stage1_arm_scores.pdf
七个病理场（含未做任何改动的人工双核）在 12 个**配对网络种子**上的模板匹配得分，
每点一个种子，横线为均值。所有臂共用同一张网、同一条噪声轨迹，因此臂间差异只来自场本身。

**关注点**：`manual_projected` 与 `manual_hard` 是否重叠（重叠说明新参数族没有偷偷改变工作点）；
七个臂的散点是否互相分离 —— 若挤成一团，说明这 15 个触点的秩次分辨不出场的形状。

### stage1_prespecified_deltas.pdf
四个**预注册**形状对比（纵向形状 / 横向摊平 / 横向拉长 / 横向位置）各自相对 `manual_projected`
的逐种子配对差，每点一个种子，横线为均值。

**关注点**：同一列的点是否**稳定落在零线同一侧** —— 判据是符号一致性（经 Holm 校正），不是均值大小。
哪个臂赢不重要，重要的是有没有任何一个维度分得开。
```

- [ ] **Step 5: 提交**

```bash
git add scripts/analyze_topic4_core_field_stage1.py \
        results/topic4_sef_hfo/data_driven_core_field/stage1_variance_probe/
git commit -m "feat(topic4-core-field): Stage 1 comparisons, gate verdict and figures"
```

- [ ] **Step 6: 停下来汇报，不要自动进 Stage 2**

无论裁定是什么，**都停下来**把 `gate_verdict.json` 的数值与含义讲给用户，等用户决定（spec §7.5，用户 2026-08-06 已裁定此分支）。`GO_*` 也不例外 —— Stage 2 的结局分类（spec §8.1）与等价最优场协议（§8.2）需要在开跑前冻结。

---

## 自查

**Spec 覆盖：** §4.1→T1；§4.3→T2；§4.4+§7.2 臂→T3；§5.1/5.2/5.2a/5.4→T4；§5.3→T5；§6.2→T6；§6.3a→T7；§7.4/7.5→T8；§6.1→T9；§7.2 跑→T10；§7.3/7.6→T11。
§6.3 流式包络**未实现** —— spec 明写"Stage 1 不强制，Stage 2 之前必须完成"，故不在本 plan（8 s 跑 ~9.4 GB/worker，12 worker 装得下）。§8/§9/§10.3 属 Stage 2/3，不在本 plan。

**占位符：** 无 TBD/TODO；每个代码步骤都是可运行代码。

**类型一致性：** `arm_h(name, s, r, geom, target_count)`、`model_templates(events, support, part_min)`、`sim_matrix(model, target, support, missing_rule)`、`candidate_key(n_dir, s_rank)`、`get_network(p, theta_deg, ar, cache_dir)`、`run_arm(arm, seed, cfg, cache_dir)`、`stage1_verdict(runs, config_checksum, expected_checksum)` 在定义处与调用处一致；`runs` 键统一为 `(arm, seed, source, missing_rule, gate)`。
