# Topic 5 发作早期方向无监督两类聚类 ↔ 间期 A/B 模板方向 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 对每个几何干净被试的发作早期激活方向无监督分两类（盲于间期），再描述性比较两类堆内平均方向与间期 A/B 模板方向，全程预锁三道门防止算法制造结论。

**Architecture:** 新纯函数模块 `src/topic5_directional_replay.py`（几何/聚类/null/门，全部 TDD），复用 `src/topic5_axis_direction.py` 的方向与圆周统计；runner 复用 rose 脚本的数据加载，逐被试出 JSON + cohort 表；plot 脚本出每被试玫瑰 + cohort 分档。

**Tech Stack:** Python 3, numpy, scikit-learn（`KMeans` / `silhouette_score` / `adjusted_rand_score`），matplotlib，pytest。

## Global Constraints

- **设计 spec（源）**：`docs/superpowers/specs/2026-06-27-topic5-ictal-direction-clustering-design.md`。每个 task 的口径以 spec 为准。
- **主队列**：6 个 ECoG 被试 `epilepsiae_{442,548,583,1084,384,958}`；SEEG / `coord_aspect<0.15` 仅 caveat 灵敏度层，不进主汇总。
- **激活 band**：`broadband`（`bb_auc`）主；`hfa`（`hfa_auc`）灵敏度。两 band 分别跑、分别报。
- **全局随机种子**：`SEED = 20260627`。
- **P0 两类资格门（三条 AND）**：`n_sz≥6` 且 `min(class size)≥3`；`p_bimodal<0.05`（**二模 null = 主方向+均匀背景散点**，B=2000，统计量=单位向量上 silhouette；纯单峰 null 太弱已弃，见 spec §4.1）；bootstrap 标签稳定性中位 `ARI≥0.5`（B=500，**次级**——对固定散点会假高，反散点主门是二模 null）。未过门 → 只准 "主方向" 措辞，**禁止 "两类"**。
- **P1 对齐 null**：旋转 B=2000，`p_align<0.05` 才算 "对齐显著"。
- **P1 轴质量门**：`Δ_AB≥120°` interpretable / `60–120°` weak_axis / `<60°` diagnostic_only；任一模板 `n_valid<6` → diagnostic_only。
- **报告分档**：几何不干净（SEEG / `coord_aspect<0.15`）→ 直接 `diagnostic_only`；"两类对上 A/B" 当且仅当 `geometry_clean ∧ interpretable ∧ two_class_eligible ∧ p_align<0.05`；否则逐级降。队列层只出描述性表，**无 pooled p、无队列断言、不声称重放**。
- **复用不重造**：圆周统计全部 import 自 `src.topic5_axis_direction`；数据加载 import 自 `scripts.plot_topic5_axis_direction_rose`。
- **输出根**：`results/topic5_ictal_recruitment/directional_clustering/`。
- **每个 commit 结尾加** `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`。
- 层级：**exploratory、描述性、无预设假设**。

---

### Task 1: 几何辅助 — `plane_fit_direction` + `coord_aspect`

**Files:**
- Create: `src/topic5_directional_replay.py`
- Test: `tests/test_topic5_directional_replay.py`

**Interfaces:**
- Consumes: 无（叶子任务）。
- Produces:
  - `plane_fit_direction(x, y, values) -> (angle_rad_or_nan, grad_norm, r2, n_valid)`：对 `values ~ a*x + b*y` 最小二乘，返回值增长方向角 [0,2π)、梯度范数、拟合 R²、有限点数。
  - `coord_aspect(x, y) -> float`：触点云 PCA 次/主奇异值比 ∈ [0,1]（近一维≈0）。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_topic5_directional_replay.py
import numpy as np
import pytest
from src.topic5_directional_replay import plane_fit_direction, coord_aspect


def test_plane_fit_direction_increasing_x():
    x = np.array([0, 1, 2, 0, 1, 2], float)
    y = np.array([0, 0, 0, 1, 1, 1], float)
    vals = x.copy()                       # increases along +x -> angle ~ 0
    ang, gnorm, r2, n = plane_fit_direction(x, y, vals)
    assert n == 6
    assert gnorm > 0
    assert r2 == pytest.approx(1.0, abs=1e-6)
    assert min(abs(ang - 0.0), abs(ang - 2 * np.pi)) < 1e-6


def test_plane_fit_direction_degenerate_constant():
    x = np.array([0, 1, 2], float)
    y = np.array([0, 1, 2], float)
    ang, gnorm, r2, n = plane_fit_direction(x, y, np.array([5.0, 5.0, 5.0]))
    assert np.isnan(ang)
    assert n == 3


def test_plane_fit_direction_too_few_points():
    ang, gnorm, r2, n = plane_fit_direction([0, 1], [0, 1], [1.0, 2.0])
    assert np.isnan(ang)
    assert n == 2


def test_coord_aspect_square_vs_line():
    sq_x = np.array([0, 1, 0, 1], float); sq_y = np.array([0, 0, 1, 1], float)
    assert coord_aspect(sq_x, sq_y) == pytest.approx(1.0, abs=1e-6)
    ln_x = np.array([0, 1, 2, 3], float); ln_y = np.array([0, 0, 0, 0], float)
    assert coord_aspect(ln_x, ln_y) < 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_topic5_directional_replay.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.topic5_directional_replay'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/topic5_directional_replay.py
"""Topic 5 发作早期方向无监督两类聚类 ↔ 间期 A/B 方向（纯函数, TDD）。

设计 spec: docs/superpowers/specs/2026-06-27-topic5-ictal-direction-clustering-design.md
圆周统计复用 src.topic5_axis_direction; 这里只放 geometry / clustering / null / gate。
"""
from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def plane_fit_direction(x, y, values):
    """方向(值增长, [0,2pi)) + 梯度范数 + 拟合 R² + 有限点数, 经最小二乘平面拟合。"""
    x = np.asarray(x, float); y = np.asarray(y, float); v = np.asarray(values, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    n_valid = int(ok.sum())
    if n_valid < 3 or np.nanstd(v[ok]) < 1e-12:
        return (np.nan, 0.0, 0.0, n_valid)
    X = np.column_stack([x[ok] - x[ok].mean(), y[ok] - y[ok].mean()])
    vv = v[ok] - v[ok].mean()
    beta, *_ = np.linalg.lstsq(X, vv, rcond=None)
    grad_norm = float(np.linalg.norm(beta))
    if grad_norm < 1e-12:
        return (np.nan, 0.0, 0.0, n_valid)
    pred = X @ beta
    ss_res = float(np.sum((vv - pred) ** 2)); ss_tot = float(np.sum(vv ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    angle = float(np.mod(np.arctan2(beta[1], beta[0]), TWO_PI))
    return (angle, grad_norm, r2, n_valid)


def coord_aspect(x, y):
    """触点云 PCA 次/主奇异值比 ∈ [0,1]; 近一维≈0。"""
    P = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    P = P[np.isfinite(P).all(1)]
    if len(P) < 3:
        return np.nan
    P = P - P.mean(0)
    ev = np.linalg.svd(P, compute_uv=False)
    return float(ev[1] / ev[0]) if ev[0] > 0 else np.nan
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_topic5_directional_replay.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/topic5_directional_replay.py tests/test_topic5_directional_replay.py
git commit -m "feat(topic5 dir-cluster): plane_fit_direction + coord_aspect geometry helpers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: 聚类 — `cluster_directions_k2` + `silhouette_unit`

**Files:**
- Modify: `src/topic5_directional_replay.py`
- Test: `tests/test_topic5_directional_replay.py`

**Interfaces:**
- Consumes: `src.topic5_axis_direction.{circular_mean, resultant_length, axial_resultant_length}`。
- Produces:
  - `cluster_directions_k2(angles, seed=0) -> dict`：键 `n, R_dir, R_axial, labels(np.ndarray), means([θ0,θ1]), sizes([n0,n1]), class_R([R0,R1]), angles(有限角)`。
  - `silhouette_unit(angles, labels) -> float`：单位向量 `[cosθ,sinθ]` 上的 silhouette；<2 类返回 -1.0。

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_topic5_directional_replay.py
from src.topic5_directional_replay import cluster_directions_k2, silhouette_unit


def test_cluster_directions_k2_two_clear_poles():
    rng = np.random.default_rng(0)
    a = np.concatenate([rng.normal(0.0, 0.1, 10), rng.normal(np.pi, 0.1, 8)])
    res = cluster_directions_k2(a, seed=0)
    assert res["n"] == 18
    assert sorted(res["sizes"]) == [8, 10]
    assert min(res["class_R"]) > 0.9
    # the two class means are ~pi apart
    d = abs(res["means"][0] - res["means"][1]) % TWO_PI
    assert min(d, TWO_PI - d) > 2.5


def test_silhouette_unit_clean_split_high():
    rng = np.random.default_rng(1)
    a = np.concatenate([rng.normal(0.0, 0.1, 10), rng.normal(np.pi, 0.1, 10)])
    res = cluster_directions_k2(a, seed=0)
    assert silhouette_unit(res["angles"], res["labels"]) > 0.5
```

Add `from src.topic5_directional_replay import TWO_PI` to the test imports if not present (use module constant).

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_topic5_directional_replay.py -q -k "cluster or silhouette"`
Expected: FAIL with `ImportError: cannot import name 'cluster_directions_k2'`

- [ ] **Step 3: Write minimal implementation**

```python
# add imports at top of src/topic5_directional_replay.py
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.topic5_axis_direction import (circular_mean, resultant_length,
                                        axial_resultant_length)


def cluster_directions_k2(angles, seed=0):
    """ictal-only k=2 on [cosθ,sinθ]; 返回 labels/means/sizes/class_R + 全体 R_dir/R_axial。"""
    a = np.asarray(angles, float); a = a[np.isfinite(a)]
    out = {"n": int(a.size),
           "R_dir": float(resultant_length(a)) if a.size else np.nan,
           "R_axial": float(axial_resultant_length(a)) if a.size else np.nan,
           "angles": a}
    if a.size < 2:
        out.update(labels=np.zeros(a.size, int), means=[np.nan, np.nan],
                   sizes=[int(a.size), 0], class_R=[np.nan, np.nan])
        return out
    V = np.column_stack([np.cos(a), np.sin(a)])
    labels = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(V)
    means, sizes, class_R = [], [], []
    for c in (0, 1):
        ac = a[labels == c]
        means.append(circular_mean(ac) if ac.size else np.nan)
        sizes.append(int(ac.size))
        class_R.append(float(resultant_length(ac)) if ac.size else np.nan)
    out.update(labels=labels, means=means, sizes=sizes, class_R=class_R)
    return out


def silhouette_unit(angles, labels):
    """silhouette on [cosθ,sinθ]; <2 distinct labels -> -1.0。"""
    a = np.asarray(angles, float); labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2 or a.size < 3:
        return -1.0
    V = np.column_stack([np.cos(a), np.sin(a)])
    return float(silhouette_score(V, labels))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_topic5_directional_replay.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/topic5_directional_replay.py tests/test_topic5_directional_replay.py
git commit -m "feat(topic5 dir-cluster): ictal-only circular k=2 + unit-vector silhouette

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: P0 命脉 — `kappa_from_R` + `unimodal_null_pvalue`

**Files:**
- Modify: `src/topic5_directional_replay.py`
- Test: `tests/test_topic5_directional_replay.py`

**Interfaces:**
- Consumes: `cluster_directions_k2`, `silhouette_unit`, `src.topic5_axis_direction.{circular_mean, resultant_length}`。
- Produces:
  - `kappa_from_R(R) -> float`：Mardia-Jupp `A⁻¹(R)` von Mises 浓度估计。
  - `unimodal_null_pvalue(angles, B=2000, seed=SEED) -> (p_bimodal, S_obs)`：观测 k=2 silhouette 相对 **"主方向+均匀背景散点" null** 的 p（防算法制造两类的核心门；纯单峰 null 太弱已弃，见 spec §4.1）。

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_topic5_directional_replay.py
from src.topic5_directional_replay import kappa_from_R, unimodal_null_pvalue


def test_kappa_from_R_monotone_and_edges():
    assert kappa_from_R(0.0) == pytest.approx(0.0, abs=1e-6)
    assert kappa_from_R(0.3) < kappa_from_R(0.6) < kappa_from_R(0.9)
    assert np.isfinite(kappa_from_R(0.999))


def test_unimodal_null_rejects_single_mode():        # P0 REGRESSION (命脉)
    rng = np.random.default_rng(7)
    a = rng.vonmises(0.6, 4.0, 24)                    # single mode + noise
    p, s = unimodal_null_pvalue(a, B=300, seed=20260627)
    assert p > 0.1                                    # must NOT be called two-class


def test_unimodal_null_passes_true_bimodal():
    rng = np.random.default_rng(8)
    a = np.concatenate([rng.vonmises(1.0, 12, 15), rng.vonmises(1.0 + np.pi, 12, 9)])
    p, s = unimodal_null_pvalue(a, B=300, seed=20260627)
    assert p < 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_topic5_directional_replay.py -q -k "kappa or unimodal"`
Expected: FAIL with `ImportError: cannot import name 'kappa_from_R'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/topic5_directional_replay.py
SEED = 20260627


def kappa_from_R(R):
    """von Mises 浓度 κ = A⁻¹(R) (Mardia & Jupp 1999 分段近似)。"""
    R = float(R)
    if R < 1e-8:
        return 0.0
    if R < 0.53:
        return 2 * R + R ** 3 + 5 * R ** 5 / 6.0
    if R < 0.85:
        return -0.4 + 1.39 * R + 0.43 / (1 - R)
    denom = R ** 3 - 4 * R ** 2 + 3 * R
    return 1.0 / denom if denom > 1e-9 else 1e6


def unimodal_null_pvalue(angles, B=2000, seed=SEED):
    """p_bimodal: H0 = 一个集中主方向 + 均匀背景散点; 信号 = 第二个集中模。

    先 k=2 分多数簇, 对多数簇拟合 von Mises(mu, kappa), 少数比例 f=n_minor/n;
    每次模拟 n 个角度(以概率 1-f 抽 von Mises, f 抽 [0,2pi) 均匀)再 k=2 取 silhouette。
    p 小 = 观测 silhouette 超过 '主方向+散点' 能产生的水平 = 真有第二个集中模。
    纯单峰 null 太弱(主方向+少数散点会被误判两类); 见 spec §4.1 P1 修复 2026-06-27。
    """
    a = np.asarray(angles, float); a = a[np.isfinite(a)]
    n = a.size
    if n < 4:
        return (1.0, np.nan)
    clus = cluster_directions_k2(a, seed=0)
    s_obs = silhouette_unit(a, clus["labels"])
    labels = clus["labels"]
    n0 = int((labels == 0).sum())
    maj = a[labels == (0 if n0 >= n - n0 else 1)]
    mu = circular_mean(maj)
    kappa = max(kappa_from_R(resultant_length(maj)), 1e-6)
    f = 1.0 - maj.size / n
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(B):
        u = rng.random(n)
        sim = np.where(u < f, rng.uniform(0, TWO_PI, n), rng.vonmises(mu, kappa, n))
        s = silhouette_unit(sim, cluster_directions_k2(sim, seed=0)["labels"])
        if s >= s_obs:
            ge += 1
    return ((1 + ge) / (B + 1), float(s_obs))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_topic5_directional_replay.py -q -k "kappa or unimodal"`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/topic5_directional_replay.py tests/test_topic5_directional_replay.py
git commit -m "feat(topic5 dir-cluster): von Mises unimodal null p_bimodal (P0 anti-manufacture gate)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: P0 稳定性 — `bootstrap_label_stability`

**Files:**
- Modify: `src/topic5_directional_replay.py`
- Test: `tests/test_topic5_directional_replay.py`

**Interfaces:**
- Consumes: `cluster_directions_k2`, `sklearn.metrics.adjusted_rand_score`。
- Produces: `bootstrap_label_stability(angles, B=500, seed=SEED) -> float`：有放回重抽 → 重聚类 → 把全部原始点指派到最近重抽质心 → 与原始标签 ARI；返回中位 ARI（n<4 → nan）。

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_topic5_directional_replay.py
from src.topic5_directional_replay import bootstrap_label_stability


def test_bootstrap_stability_high_for_clean_bimodal():
    rng = np.random.default_rng(3)
    a = np.concatenate([rng.normal(0.0, 0.08, 14), rng.normal(np.pi, 0.08, 12)])
    assert bootstrap_label_stability(a, B=200, seed=20260627) > 0.7


def test_bootstrap_stability_nan_for_tiny_n():
    assert np.isnan(bootstrap_label_stability(np.array([0.1, 0.2, 0.3]), B=50))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_topic5_directional_replay.py -q -k "bootstrap"`
Expected: FAIL with `ImportError: cannot import name 'bootstrap_label_stability'`

- [ ] **Step 3: Write minimal implementation**

```python
# add import at top: from sklearn.metrics import silhouette_score, adjusted_rand_score
# (extend the existing sklearn.metrics import line)

def bootstrap_label_stability(angles, B=500, seed=SEED):
    """中位 bootstrap ARI: 重抽→重聚类→全点指派最近质心→与原始标签 ARI。"""
    a = np.asarray(angles, float); a = a[np.isfinite(a)]
    n = a.size
    if n < 4:
        return np.nan
    base = cluster_directions_k2(a, seed=0)["labels"]
    V = np.column_stack([np.cos(a), np.sin(a)])
    rng = np.random.default_rng(seed)
    aris = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        boot = cluster_directions_k2(a[idx], seed=0)
        cents = []
        for c in (0, 1):
            ac = a[idx][boot["labels"] == c]
            if ac.size == 0:
                cents = None; break
            cents.append([np.cos(ac).mean(), np.sin(ac).mean()])
        if cents is None:
            continue
        cents = np.asarray(cents)
        d = np.linalg.norm(V[:, None, :] - cents[None, :, :], axis=2)
        pred = d.argmin(axis=1)
        aris.append(adjusted_rand_score(base, pred))
    return float(np.median(aris)) if aris else np.nan
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_topic5_directional_replay.py -q -k "bootstrap"`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/topic5_directional_replay.py tests/test_topic5_directional_replay.py
git commit -m "feat(topic5 dir-cluster): bootstrap label stability (P0 secondary gate)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: 门 — `two_class_eligible`

**Files:**
- Modify: `src/topic5_directional_replay.py`
- Test: `tests/test_topic5_directional_replay.py`

**Interfaces:**
- Consumes: 无（纯逻辑）。
- Produces: `two_class_eligible(n_sz, sizes, p_bimodal, stability, *, bimodal_alpha=0.05, stab_min=0.5) -> (bool, list[str])`：返回是否过门 + 未过门原因列表。

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_topic5_directional_replay.py
from src.topic5_directional_replay import two_class_eligible


def test_two_class_eligible_all_pass():
    ok, reasons = two_class_eligible(10, [7, 3], 0.01, 0.6)
    assert ok and reasons == []


def test_two_class_eligible_each_failure_reason():
    assert two_class_eligible(5, [3, 2], 0.01, 0.9)[1].count("n_sz<6") == 1
    assert "min_class<3" in two_class_eligible(11, [9, 2], 0.01, 0.9)[1]
    assert "p_bimodal>=alpha" in two_class_eligible(10, [5, 5], 0.2, 0.9)[1]
    assert "stability<min" in two_class_eligible(10, [5, 5], 0.01, 0.3)[1]
    assert two_class_eligible(5, [3, 2], 0.2, 0.3)[0] is False


def test_unimodal_with_scattered_outliers_not_two_class():   # P1 review anti-deception (命脉#2)
    rng = np.random.default_rng(9)
    main = rng.vonmises(0.4, 10, 20)
    scatter = rng.uniform(0, 2 * np.pi, 4)
    a = np.concatenate([main, scatter])                       # one dominant direction + scatter
    p, _ = unimodal_null_pvalue(a, B=500, seed=20260627)      # contaminated null -> p ~ 0.27 (high)
    clus = cluster_directions_k2(a, seed=0)
    stab = bootstrap_label_stability(a, B=200, seed=20260627) # ~1.0 (fixed scatter), so p is the gate
    eligible, _ = two_class_eligible(clus["n"], clus["sizes"], p, stab)
    assert eligible is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_topic5_directional_replay.py -q -k "two_class"`
Expected: FAIL with `ImportError: cannot import name 'two_class_eligible'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/topic5_directional_replay.py
def two_class_eligible(n_sz, sizes, p_bimodal, stability, *,
                       bimodal_alpha=0.05, stab_min=0.5):
    """P0 三条 AND 门; 返回 (eligible, reasons)。未过门只准 '主方向' 措辞。"""
    reasons = []
    if n_sz < 6:
        reasons.append("n_sz<6")
    if min(sizes) < 3:
        reasons.append("min_class<3")
    if not (p_bimodal is not None and np.isfinite(p_bimodal) and p_bimodal < bimodal_alpha):
        reasons.append("p_bimodal>=alpha")
    if not (stability is not None and np.isfinite(stability) and stability >= stab_min):
        reasons.append("stability<min")
    return (len(reasons) == 0, reasons)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_topic5_directional_replay.py -q -k "two_class"`
Expected: PASS (3 passed) — incl. the scattered-outliers anti-deception test

- [ ] **Step 5: Commit**

```bash
git add src/topic5_directional_replay.py tests/test_topic5_directional_replay.py
git commit -m "feat(topic5 dir-cluster): two_class_eligible P0 AND-gate with reasons

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: 门 — `axis_quality_tier`

**Files:**
- Modify: `src/topic5_directional_replay.py`
- Test: `tests/test_topic5_directional_replay.py`

**Interfaces:**
- Consumes: 无。
- Produces: `axis_quality_tier(delta_ab_rad, n_valid_a, n_valid_b, *, interp_min_deg=120, weak_min_deg=60, min_valid=6) -> str`，取值 `'interpretable'|'weak_axis'|'diagnostic_only'`。

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_topic5_directional_replay.py
from src.topic5_directional_replay import axis_quality_tier


def test_axis_quality_tier_boundaries():
    assert axis_quality_tier(np.radians(147), 10, 10) == "interpretable"
    assert axis_quality_tier(np.radians(120), 10, 10) == "interpretable"
    assert axis_quality_tier(np.radians(119), 10, 10) == "weak_axis"
    assert axis_quality_tier(np.radians(60), 10, 10) == "weak_axis"
    assert axis_quality_tier(np.radians(59), 10, 10) == "diagnostic_only"
    assert axis_quality_tier(np.radians(6), 10, 10) == "diagnostic_only"


def test_axis_quality_tier_low_valid_forces_diagnostic():
    assert axis_quality_tier(np.radians(147), 5, 10) == "diagnostic_only"
    assert axis_quality_tier(np.nan, 10, 10) == "diagnostic_only"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_topic5_directional_replay.py -q -k "axis_quality"`
Expected: FAIL with `ImportError: cannot import name 'axis_quality_tier'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/topic5_directional_replay.py
def axis_quality_tier(delta_ab_rad, n_valid_a, n_valid_b, *,
                      interp_min_deg=120, weak_min_deg=60, min_valid=6):
    """间期 A/B 轴质量分档 (按 Δ_AB 度数, 预锁阈值); 模板触点不足强降 diagnostic。"""
    if n_valid_a < min_valid or n_valid_b < min_valid or not np.isfinite(delta_ab_rad):
        return "diagnostic_only"
    deg = float(np.degrees(delta_ab_rad))
    if deg >= interp_min_deg:
        return "interpretable"
    if deg >= weak_min_deg:
        return "weak_axis"
    return "diagnostic_only"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_topic5_directional_replay.py -q -k "axis_quality"`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/topic5_directional_replay.py tests/test_topic5_directional_replay.py
git commit -m "feat(topic5 dir-cluster): axis_quality_tier (P1 pre-locked template-axis gate)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: P1 对齐 — `angular_distance` + `best_pair_residual` + `best_pair_rotation_null`

**Files:**
- Modify: `src/topic5_directional_replay.py`
- Test: `tests/test_topic5_directional_replay.py`

**Interfaces:**
- Consumes: 无（自含）。
- Produces:
  - `angular_distance(a, b) -> float`：全圆角距 ∈ [0,π]。
  - `best_pair_residual(class_means, template_dirs) -> dict|None`：`{sum, mean, pairing('straight'|'crossed'), matched:[d1,d2]}`（全 rad）；任一 NaN → `None`。
  - `best_pair_rotation_null(class_means, template_dirs, B=2000, seed=SEED) -> float`：旋转 null 下 `p_align`（残差 ≤ 观测的比例）。

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_topic5_directional_replay.py
from src.topic5_directional_replay import (angular_distance, best_pair_residual,
                                           best_pair_rotation_null)


def test_best_pair_residual_picks_straight_and_is_exchange_invariant():
    r1 = best_pair_residual([0.1, 3.0], [0.0, 3.1])
    assert r1["pairing"] == "straight" and r1["sum"] == pytest.approx(0.2, abs=1e-6)
    assert sorted(r1["matched"]) == pytest.approx([0.1, 0.1], abs=1e-6)
    r2 = best_pair_residual([3.0, 0.1], [0.0, 3.1])   # swap c1<->c2
    assert r2["sum"] == pytest.approx(r1["sum"], abs=1e-9) and r2["pairing"] == "crossed"


def test_best_pair_residual_none_on_nan():
    assert best_pair_residual([np.nan, 1.0], [0.0, 3.0]) is None


def test_rotation_null_small_when_aligned():
    p = best_pair_rotation_null([0.1, np.pi + 0.1], [0.0, np.pi], B=2000, seed=20260627)
    assert p < 0.1


def test_rotation_null_large_when_orthogonal():
    p = best_pair_rotation_null([np.pi / 2, 3 * np.pi / 2], [0.0, np.pi], B=2000, seed=20260627)
    assert p > 0.8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_topic5_directional_replay.py -q -k "best_pair or rotation"`
Expected: FAIL with `ImportError: cannot import name 'angular_distance'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/topic5_directional_replay.py
def angular_distance(a, b):
    """全圆角距 ∈ [0,pi]; d(0,pi)=pi, d(0,2pi)=0。"""
    d = abs(float(a) - float(b)) % TWO_PI
    return float(min(d, TWO_PI - d))


def best_pair_residual(class_means, template_dirs):
    """{c1,c2}->{A,B} 取 straight/crossed 角距和最小; dict{sum,mean,pairing,matched}(rad); 任一 NaN -> None。"""
    c1, c2 = class_means; A, B = template_dirs
    if not all(np.isfinite(v) for v in (c1, c2, A, B)):
        return None
    straight = (angular_distance(c1, A), angular_distance(c2, B))
    crossed = (angular_distance(c1, B), angular_distance(c2, A))
    matched, pairing = (straight, "straight") if sum(straight) <= sum(crossed) else (crossed, "crossed")
    return {"sum": float(sum(matched)), "mean": float(np.mean(matched)),
            "pairing": pairing, "matched": [float(matched[0]), float(matched[1])]}


def best_pair_rotation_null(class_means, template_dirs, B=2000, seed=SEED):
    """旋转 null: 共同旋转两类均向 φ~U[0,2pi), 对固定模板重做 best-pair; p_align=resid_sum<=obs 比例。"""
    c1, c2 = class_means
    if not all(np.isfinite(v) for v in (c1, c2, *template_dirs)):
        return np.nan
    obs = best_pair_residual(class_means, template_dirs)["sum"]
    rng = np.random.default_rng(seed)
    le = 0
    for _ in range(B):
        phi = rng.uniform(0, TWO_PI)
        r = best_pair_residual([c1 + phi, c2 + phi], template_dirs)
        if r["sum"] <= obs:
            le += 1
    return (1 + le) / (B + 1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_topic5_directional_replay.py -q`
Expected: PASS (all tests, ~15 passed)

- [ ] **Step 5: Commit**

```bash
git add src/topic5_directional_replay.py tests/test_topic5_directional_replay.py
git commit -m "feat(topic5 dir-cluster): best_pair_residual + rotation null p_align (P1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Runner — `run_topic5_directional_replay.py`（逐被试 JSON + cohort 表）

**Files:**
- Create: `scripts/run_topic5_directional_replay.py`
- Test: `tests/test_run_topic5_directional_replay.py`

**Interfaces:**
- Consumes: `src.topic5_directional_replay.*`；`scripts.plot_topic5_axis_direction_rose.{_load_frame, _seizure_angles, _electrode_kind}`；间期模板场 `_t_{a,b}.json`。
- Produces:
  - `PRIMARY_COHORT = ["epilepsiae_442","epilepsiae_548","epilepsiae_583","epilepsiae_1084","epilepsiae_384","epilepsiae_958"]`
  - `template_direction(ds_sid, x, y, names, which) -> (angle, grad_norm, r2, n_valid)`
  - `process_subject(ds_sid, activation, *, n_perm=2000, n_boot=500, seed=SEED) -> dict`（含全部门量 + tier + provenance；`status='ok'|'skip'` + reason）
  - `_report_tier(axis_tier, eligible, p_align, geometry_clean) -> str`（geometry 不干净 → 直接 `diagnostic_only`）
  - `main()`：遍历 cohort、写 `per_subject/*.json` + `cohort_summary_{activation}.{json,csv}`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_topic5_directional_replay.py
import numpy as np
import pytest
from scripts.run_topic5_directional_replay import process_subject, PRIMARY_COHORT, _report_tier


def test_primary_cohort_is_six_ecog():
    assert PRIMARY_COHORT == ["epilepsiae_442", "epilepsiae_548", "epilepsiae_583",
                              "epilepsiae_1084", "epilepsiae_384", "epilepsiae_958"]


def test_process_subject_442_smoke():
    """Integration: needs real cache + frame + template files present."""
    rec = process_subject("epilepsiae_442", "broadband", n_perm=50, n_boot=50, seed=20260627)
    assert rec["status"] == "ok"
    for k in ("n_sz", "sizes", "R_dir", "R_axial", "p_bimodal", "stability",
              "two_class_eligible", "theta_A", "theta_B", "delta_ictal_deg", "delta_ab_deg",
              "axis_tier", "best_pair_resid_sum_deg", "best_pair_resid_each_deg",
              "best_pair_pairing", "p_align", "report_tier", "geometry_clean",
              "electrode_kind", "coord_aspect", "activation"):
        assert k in rec
    assert rec["axis_tier"] in ("interpretable", "weak_axis", "diagnostic_only")
    assert rec["report_tier"] in ("two_class_mapped", "two_class_unmapped",
                                  "single_axis", "diagnostic_only")


def test_report_tier_geometry_unclean_forces_diagnostic():
    assert _report_tier("interpretable", True, 0.001, geometry_clean=False) == "diagnostic_only"
    assert _report_tier("interpretable", True, 0.001, geometry_clean=True) == "two_class_mapped"
    assert _report_tier("interpretable", False, 0.001, geometry_clean=True) == "single_axis"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_topic5_directional_replay.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.run_topic5_directional_replay'`

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""Topic 5 发作早期方向无监督两类聚类 ↔ 间期 A/B 方向 runner。

设计 spec: docs/superpowers/specs/2026-06-27-topic5-ictal-direction-clustering-design.md
口径: ictal-only k=2 (盲于间期) -> 三道预锁门 -> 描述性分档。无锚点/无预设/无队列断言。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.plot_topic5_axis_direction_rose import (_load_frame, _seizure_angles,
                                                     _electrode_kind)
from src.topic5_directional_replay import (
    SEED, plane_fit_direction, coord_aspect, cluster_directions_k2,
    unimodal_null_pvalue, bootstrap_label_stability, two_class_eligible,
    axis_quality_tier, angular_distance, best_pair_residual, best_pair_rotation_null)
from src.topic5_axis_direction import axial_mean, axial_distance

REAL_DIR = _ROOT / "results/spatial_modulation/propagation_geometry/observation_readout/real_subjects"
OUT_DIR = _ROOT / "results/topic5_ictal_recruitment/directional_clustering"
ACTIVATION_KEY = {"broadband": "bb_auc", "hfa": "hfa_auc"}
PRIMARY_COHORT = ["epilepsiae_442", "epilepsiae_548", "epilepsiae_583",
                  "epilepsiae_1084", "epilepsiae_384", "epilepsiae_958"]
ASPECT_MIN = 0.15


def template_direction(ds_sid, x, y, names, which):
    f = REAL_DIR / f"{ds_sid}_t_{which}.json"
    if not f.exists():
        return (np.nan, 0.0, 0.0, 0)
    j = json.loads(f.read_text())
    tr = {c["name"]: c.get("typical_rank") for c in j.get("channels", [])}
    vals = np.array([tr[n] if (n in tr and tr[n] is not None) else np.nan for n in names], float)
    return plane_fit_direction(x, y, vals)


def _report_tier(axis_tier, eligible, p_align, geometry_clean):
    if not geometry_clean:                       # P1 review: unclean geometry -> never two-class
        return "diagnostic_only"
    if axis_tier == "diagnostic_only":
        return "diagnostic_only"
    if not eligible:
        return "single_axis"
    if p_align is not None and np.isfinite(p_align) and p_align < 0.05:
        return "two_class_mapped"
    return "two_class_unmapped"


def process_subject(ds_sid, activation, *, n_perm=2000, n_boot=500, seed=SEED):
    loaded = _load_frame(ds_sid)
    if loaded is None:
        return {"subject": ds_sid, "activation": activation, "status": "skip",
                "reason": "no_frame"}
    rec, x, y, names = loaded
    ds, subj = ds_sid.split("_", 1)
    kind, _ = _electrode_kind(ds, subj, names)
    asp = coord_aspect(x, y)
    sz = _seizure_angles(ds_sid, x, y, names, activation)
    if sz.size < 4:
        return {"subject": ds_sid, "activation": activation, "status": "skip",
                "reason": "too_few_seizures", "n_sz": int(sz.size)}
    clus = cluster_directions_k2(sz, seed=0)
    p_bimodal, s_obs = unimodal_null_pvalue(sz, B=n_perm, seed=seed)
    stability = bootstrap_label_stability(sz, B=n_boot, seed=seed)
    eligible, reasons = two_class_eligible(clus["n"], clus["sizes"], p_bimodal, stability)
    thA, gnA, r2A, nvA = template_direction(ds_sid, x, y, names, "a")
    thB, gnB, r2B, nvB = template_direction(ds_sid, x, y, names, "b")
    delta_ab = angular_distance(thA, thB) if (np.isfinite(thA) and np.isfinite(thB)) else np.nan
    axis_tier = axis_quality_tier(delta_ab, nvA, nvB)
    bp = best_pair_residual(clus["means"], [thA, thB])               # dict or None
    p_align = best_pair_rotation_null(clus["means"], [thA, thB], B=n_perm, seed=seed)
    geom_clean = bool(kind == "ECoG" and np.isfinite(asp) and asp >= ASPECT_MIN)
    report_tier = _report_tier(axis_tier, eligible, p_align, geom_clean)
    m0, m1 = clus["means"]
    delta_ictal = angular_distance(m0, m1) if (np.isfinite(m0) and np.isfinite(m1)) else np.nan
    axis_offset = (axial_distance(axial_mean(clus["angles"]), axial_mean(np.array([thA, thB])))
                   if (np.isfinite(thA) and np.isfinite(thB) and clus["n"] >= 2) else np.nan)
    return {
        "subject": ds_sid, "activation": activation, "status": "ok",
        "geometry_clean": geom_clean,
        "electrode_kind": kind, "coord_aspect": None if not np.isfinite(asp) else float(asp),
        "n_sz": clus["n"], "sizes": clus["sizes"], "class_R": clus["class_R"],
        "means_deg": [None if not np.isfinite(m) else float(np.degrees(m)) for m in clus["means"]],
        "R_dir": clus["R_dir"], "R_axial": clus["R_axial"],
        "delta_ictal_deg": None if not np.isfinite(delta_ictal) else float(np.degrees(delta_ictal)),
        "axis_offset_deg": None if not np.isfinite(axis_offset) else float(np.degrees(axis_offset)),
        "silhouette": s_obs, "p_bimodal": p_bimodal, "stability": stability,
        "two_class_eligible": eligible, "two_class_reasons": reasons,
        "theta_A": None if not np.isfinite(thA) else float(np.degrees(thA)),
        "theta_B": None if not np.isfinite(thB) else float(np.degrees(thB)),
        "template_quality": {"grad_norm_a": gnA, "r2_a": r2A, "n_valid_a": nvA,
                             "grad_norm_b": gnB, "r2_b": r2B, "n_valid_b": nvB},
        "delta_ab_deg": None if not np.isfinite(delta_ab) else float(np.degrees(delta_ab)),
        "axis_tier": axis_tier,
        "best_pair_resid_sum_deg": None if bp is None else float(np.degrees(bp["sum"])),
        "best_pair_resid_each_deg": None if bp is None else [float(np.degrees(d)) for d in bp["matched"]],
        "best_pair_pairing": None if bp is None else bp["pairing"],
        "p_align": None if (p_align is None or not np.isfinite(p_align)) else float(p_align),
        "report_tier": report_tier,
        "provenance": {"spec": "docs/superpowers/specs/2026-06-27-topic5-ictal-direction-clustering-design.md",
                       "n_perm": n_perm, "n_boot": n_boot, "seed": seed},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--activation", choices=list(ACTIVATION_KEY), default="broadband")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    subs = args.subjects or PRIMARY_COHORT
    (OUT_DIR / "per_subject").mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in subs:
        rec = process_subject(sid, args.activation, n_perm=args.n_perm, n_boot=args.n_boot)
        (OUT_DIR / "per_subject" / f"{sid}__dir_cluster_{args.activation}.json").write_text(
            json.dumps(rec, indent=2))
        rows.append(rec)
        print(f"  {sid}: {rec.get('status')} "
              f"{rec.get('report_tier', '')} {rec.get('axis_tier', '')}", flush=True)
    summ = OUT_DIR / f"cohort_summary_{args.activation}.json"
    summ.write_text(json.dumps(rows, indent=2))
    ok = [r for r in rows if r["status"] == "ok"]
    cols = ["subject", "n_sz", "sizes", "R_dir", "R_axial", "p_bimodal", "stability",
            "two_class_eligible", "delta_ictal_deg", "delta_ab_deg", "axis_tier",
            "best_pair_resid_sum_deg", "best_pair_resid_each_deg", "best_pair_pairing",
            "p_align", "report_tier", "geometry_clean", "electrode_kind", "coord_aspect"]
    with open(OUT_DIR / f"cohort_summary_{args.activation}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for r in ok:
            w.writerow([r.get(c) for c in cols])
    print(f"wrote {summ}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_run_topic5_directional_replay.py -q`
Expected: PASS (3 passed). If `test_process_subject_442_smoke` skips/fails on missing data, confirm `results/topic5_ictal_recruitment/t0_feature_cache/epilepsiae_442.npz` and `..._t_a.json/_t_b.json` exist first.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_topic5_directional_replay.py tests/test_run_topic5_directional_replay.py
git commit -m "feat(topic5 dir-cluster): runner with per-subject JSON + cohort summary + report tiers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: 图 + README — `plot_topic5_directional_replay.py`

**Files:**
- Create: `scripts/plot_topic5_directional_replay.py`
- Create: `results/topic5_ictal_recruitment/directional_clustering/figures/README.md`
- Test: `tests/test_plot_topic5_directional_replay.py`

**Interfaces:**
- Consumes: cohort_summary + per_subject JSON（Task 8 产物）；`_load_frame, _seizure_angles`（重算发作角着色）；`cluster_directions_k2`。
- Produces: `plot_subject(ds_sid, activation) -> Path|None`（每被试玫瑰：发作按类着色 + θ_c1/θ_c2 实线 + θ_A/θ_B 虚线 + tier 角注）；`main()` 遍历 + 写图。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_plot_topic5_directional_replay.py
from pathlib import Path
from scripts.plot_topic5_directional_replay import plot_subject


def test_plot_subject_442_writes_png(tmp_path):
    out = plot_subject("epilepsiae_442", "broadband")
    assert out is not None and Path(out).exists() and Path(out).suffix == ".png"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_plot_topic5_directional_replay.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.plot_topic5_directional_replay'`

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""Topic 5 发作方向两类聚类 ↔ 间期 A/B 方向 — 每被试玫瑰图。

黑虚线 = 间期模板 A/B 方向; 彩色实线 = 发作两类堆内平均方向; 彩色 ticks = 逐发作方向(按类着色)。
角注 = report_tier / axis_tier / p_bimodal / p_align。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.plot_topic5_axis_direction_rose import _load_frame, _seizure_angles
from scripts.run_topic5_directional_replay import template_direction, OUT_DIR
from src.topic5_directional_replay import cluster_directions_k2

FIG_DIR = OUT_DIR / "figures"
C1, C2 = "#1b9e77", "#7570b3"


def plot_subject(ds_sid, activation):
    loaded = _load_frame(ds_sid)
    if loaded is None:
        return None
    rec, x, y, names = loaded
    sz = _seizure_angles(ds_sid, x, y, names, activation)
    if sz.size < 4:
        return None
    clus = cluster_directions_k2(sz, seed=0)
    labels, means = clus["labels"], clus["means"]
    thA, *_ = template_direction(ds_sid, x, y, names, "a")
    thB, *_ = template_direction(ds_sid, x, y, names, "b")
    rj = OUT_DIR / "per_subject" / f"{ds_sid}__dir_cluster_{activation}.json"
    meta = json.loads(rj.read_text()) if rj.exists() else {}

    fig = plt.figure(figsize=(7.2, 7.6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="polar")
    for c, col in ((0, C1), (1, C2)):
        for a in clus["angles"][labels == c]:
            ax.plot([a, a], [0, 0.82], color=col, lw=1.1, alpha=0.6, zorder=2)
        if np.isfinite(means[c]):
            ax.plot([means[c], means[c]], [0, 1.05], color=col, lw=3.4, zorder=4,
                    label=f"ictal class {c+1} (n={clus['sizes'][c]}, R={clus['class_R'][c]:.2f})")
    for th, nm in ((thA, "interictal A"), (thB, "interictal B")):
        if np.isfinite(th):
            ax.plot([th, th], [0, 1.12], color="black", lw=2.2, ls="--", zorder=3, label=nm)
    ax.set_theta_zero_location("E"); ax.set_theta_direction(1)
    ax.set_rticks([]); ax.set_rlim(0, 1.2)
    tier = meta.get("report_tier", "?"); axt = meta.get("axis_tier", "?")
    pb = meta.get("p_bimodal"); pa = meta.get("p_align")
    pretty = ds_sid.replace("epilepsiae_", "E").replace("yuquan_", "Y-")
    ax.set_title(f"{pretty} — ictal direction k=2 vs interictal A/B  ({activation})\n"
                 f"report_tier={tier} · axis={axt} · "
                 f"p_bimodal={pb if pb is None else round(pb,3)} · "
                 f"p_align={pa if pa is None else round(pa,3)}", fontsize=11)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=8.6)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{ds_sid}__dir_cluster_{activation}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--activation", default="broadband")
    args = ap.parse_args()
    from scripts.run_topic5_directional_replay import PRIMARY_COHORT
    subs = args.subjects or PRIMARY_COHORT
    for sid in subs:
        out = plot_subject(sid, args.activation)
        print(f"  {'wrote ' + out.name if out else 'skip ' + sid}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes, then render the cohort + write README**

Run: `python -m pytest tests/test_plot_topic5_directional_replay.py -q`
Expected: PASS (1 passed)

Then render both bands and write the mandatory Chinese `figures/README.md`:

Run: `python scripts/run_topic5_directional_replay.py --activation broadband && python scripts/run_topic5_directional_replay.py --activation hfa && python scripts/plot_topic5_directional_replay.py --activation broadband && python scripts/plot_topic5_directional_replay.py --activation hfa`
Expected: per-subject PNG + cohort_summary JSON/CSV written; **用户目视每张图**。

Create `results/topic5_ictal_recruitment/directional_clustering/figures/README.md`:

```markdown
# 发作早期方向两类聚类 ↔ 间期 A/B 方向（图说明）

> 探索性、描述性、无预设假设。口径见 `docs/superpowers/specs/2026-06-27-topic5-ictal-direction-clustering-design.md`。

### epilepsiae_<id>__dir_cluster_broadband.png / _hfa.png
每被试一张极坐标玫瑰。彩色细线 = 该被试逐次发作的早期激活方向（按无监督两类着色，绿=类1/紫=类2）；
两条彩色粗线 = 两类各自的堆内平均方向；两条黑虚线 = 间期两条传播路线（模板 A/B）的方向。
标题角注给出该被试的分档：`report_tier`（two_class_mapped / two_class_unmapped / single_axis / diagnostic_only）、
`axis_tier`（间期 A/B 是否成方向对）、`p_bimodal`（发作方向是否真的分两堆，越小越像真两类）、
`p_align`（两类方向贴间期 A/B 是否超过随机，越小越显著）。

**关注点**：先看 `axis_tier` 是否 interpretable（不是的话两条黑虚线几乎重合、不能比）；再看 `p_bimodal` 是否 <0.05（否则只是把一个主方向硬切两半，只能读"主方向"）；只有三门全绿（report_tier=two_class_mapped）才可读成"发作两类分别对上 A/B"，其余一律按主方向 / 不显著 / 只诊断解读。
```

- [ ] **Step 5: Commit**

```bash
git add scripts/plot_topic5_directional_replay.py tests/test_plot_topic5_directional_replay.py
git add -f results/topic5_ictal_recruitment/directional_clustering/figures/README.md  # results/* is gitignored (.gitignore:247)
git commit -m "feat(topic5 dir-cluster): per-subject rose figure + figures README

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage** (spec §→task):
- §2 队列/band/坐标系/激活场 → Task 8 (`PRIMARY_COHORT`, `ACTIVATION_KEY`, `_load_frame`, `_seizure_angles`)。✓
- §3 Step1 逐发作方向 → Task 8 (`_seizure_angles`，底层 `gradient_angle`)。✓
- §3 Step2 ictal-only k=2 → Task 2。✓
- §3 Step3 间期模板方向+质量 → Task 1 (`plane_fit_direction`) + Task 8 (`template_direction`)。✓
- §3 Step4 关系量 (Δ_ictal/Δ_AB/轴偏移/best-pair) → Task 7 + Task 8。✓ （Δ_ictal = `angular_distance(means[0], means[1])`，在 Task 8 可加；见下补）
- §4.1 P0 两类资格 → Task 3 (p_bimodal) + Task 4 (stability) + Task 5 (gate)。✓
- §4.2 P1 旋转 null → Task 7。✓
- §4.3 P1 轴质量门 → Task 6。✓
- §5 报告分档 → Task 8 (`_report_tier`)。✓
- §6 复用 + 新代码 + 输出目录 → Tasks 1-9。✓
- §7 TDD 合同 11 项 → 各 task 测试覆盖（单峰回归=Task3 test；旋转标定=Task7 test；轴阈边界=Task6 test；best-pair 对换=Task7 test；数量门=Task5 test；κ 单调=Task3 test；plane_fit 退化=Task1 test；复用核对=import 自 topic5_axis_direction）。✓
- §8 验收（图+README+目视+cohort 表）→ Task 9。✓

**Gap found + fix:** §3 Step4 的 `Δ_ictal`（两类均向夹角）和"ictal 轴 vs 间期轴偏移"未写进 `process_subject` 输出。**修复**：在 Task 8 Step 3 的 `process_subject` 返回 dict 中，`axis_tier` 行后补两个字段（实现时加入，不另起 task）：
```python
        "delta_ictal_deg": None if not (np.isfinite(clus["means"][0]) and np.isfinite(clus["means"][1]))
                           else float(np.degrees(angular_distance(clus["means"][0], clus["means"][1]))),
        "axis_offset_deg": None if not (np.isfinite(thA) and np.isfinite(thB) and clus["n"] >= 2)
                           else float(np.degrees(__import__("src.topic5_axis_direction", fromlist=["axial_distance"]).axial_distance(
                               __import__("src.topic5_axis_direction", fromlist=["axial_mean"]).axial_mean(clus["angles"]),
                               __import__("src.topic5_axis_direction", fromlist=["axial_mean"]).axial_mean(np.array([thA, thB]))))),
```
（更干净：在 Task 8 顶部 `from src.topic5_axis_direction import axial_mean, axial_distance`，则两字段写成 `axial_distance(axial_mean(clus["angles"]), axial_mean(np.array([thA,thB])))`；实现时用 import 形式，勿用 `__import__`。）

**2. Placeholder scan:** 无 TBD/TODO；每个 code step 含完整代码；commit 命令完整。✓

**3. Type consistency:**
- `cluster_directions_k2` 返回 `means`/`sizes`/`class_R` 为长度 2 list、`labels` 为 np.ndarray、`angles` 为有限角数组 — Task 2/3/4/7/8/9 一致消费。✓
- `best_pair_residual` 返回 `(float|nan, str|None)`，`best_pair_rotation_null` 返回 `float|nan` — Task 8 消费一致（`p_align` None 化处理）。✓
- `two_class_eligible` 返回 `(bool, list)` — Task 8 解包 `eligible, reasons` 一致。✓
- `axis_quality_tier` 返回 str ∈ 三值 — Task 8 `_report_tier` 分支一致。✓
- `template_direction`/`plane_fit_direction` 四元组 `(angle, grad_norm, r2, n_valid)` — Task 8 解包一致。✓

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-27-topic5-ictal-direction-clustering.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — 每个 task 派新 subagent，task 间我复审，迭代快。

**2. Inline Execution** — 本会话内按 executing-plans 批量执行，带 checkpoint 复审。

**走哪种？**
