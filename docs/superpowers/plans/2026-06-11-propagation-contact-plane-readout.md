# 2D 传播触点平面读出 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把真实传播模板与模型输出都渲染成同一套标准化 2D 触点平面读出（rank 主场 + 连续场 T/S/U），用镜像不变、support-gated 的描述性指标比"像不像"，SOZ 仅作描述性叠加。

**Architecture:** 新模块 `src/propagation_contact_plane_readout.py` 只装新东西（signed-transverse 轴 + normalized 2D 场 + 镜像不变相关 + model-vs-cohort 比较 + SOZ overlay 匹配）；轴框架 / 端点核 / per-channel stereotypy / 数据加载全部复用 `src/propagation_skeleton_geometry.py` + `src/interictal_propagation.py` + `src/lagpat_rank_audit.py` + `src/seeg_coord_loader.py`。模型侧复用已有虚拟 SEEG 观测层写出的真实格式 NPZ，走完全相同的读出。

**Tech Stack:** Python, NumPy, SciPy（spearman/插值），Matplotlib（静态图），pytest（TDD）。

**Spec:** `docs/superpowers/specs/2026-06-11-propagation-contact-plane-readout-design.md`

> **修订 2026-06-11 (pilot audit)**: `x_norm/y_norm` 归一化分母由 `axis_length` 改为参与触点 along 的鲁棒跨度 `norm_scale_mm = p97.5−p2.5`（真实数据 60% 主队列 along/axis_length 爆表出网格）。record 增 `norm_scale_mm` + `out_of_field` 统计。runner 增良性 no-coord 跳过。详见 spec §3 修订段。

---

## File Structure

**新建：**
- `src/propagation_contact_plane_readout.py` — 纯函数库（本 plan 的核心；Task 1–7）
- `scripts/run_contact_plane_readout.py` — 真实 subject runner（Task 8）
- `scripts/run_model_contact_plane_readout.py` — 模型 runner（Task 9）
- `scripts/run_real_vs_model_comparison.py` — 比较 runner（Task 10）
- `scripts/plot_contact_plane_static.py` — 静态图 + README（Task 11）
- `tests/test_propagation_contact_plane_readout.py` — 全部单元测试

**复用（不修改）：**
- `src/propagation_skeleton_geometry.py`：`compute_axis_frame`、`build_endpoint_cores`、`channel_stereotypy_components`、`assign_events_to_templates`、`classify_sampling_geometry`、`parse_shaft`
- `src/lagpat_rank_audit.py`：`mask_phantom_ranks(ranks, bools, normalize=True)`（= spec 的 `rank_norm`，已验证）
- `src/interictal_propagation.py`：`load_subject_propagation_events`
- `src/seeg_coord_loader.py`：`load_subject_coords`
- `src/sef_hfo_soz_localization.py`：`_first_contact`、`classify_montage`（SOZ first-contact alias）
- `src/sef_hfo_observation.py`：`sample_envelopes` 思路（高斯核），`grid_coords` 思路（对称网格）

**输出目录：** `results/spatial_modulation/propagation_geometry/observation_readout/{real_subjects,model_subjects,comparison,figures/{static_maps,animations}}/`

**本 plan 范围（spec §0 第一批）：** 不实现 animation、`reverse_axis_cosine`、`heldout_rho`。

---

## 锁定常量（写在模块顶部，writing-plans 钉死的数值门）

```python
# src/propagation_contact_plane_readout.py 顶部
GRID_N = 81                 # 网格边长（奇数 → 含 y=0 行；flip 行 = y -> -y 精确）
X_LO, X_HI = -0.5, 1.5      # along_norm 范围（含 source 前 / sink 后）
Y_EXT = 1.0                 # signed_transverse_norm 对称半幅 [-Y_EXT, +Y_EXT]
S_THRESH = 0.15             # 支撑显示阈：S(x,y) < S_THRESH 灰掉且不进相关
OVERLAP_MIN = 25            # §9 相关所需最少交集像素数
POOR_PLANARITY_PC1 = 0.80   # transverse_pc1_variance_explained < 此值 -> poor_planarity
MIN_CONTACTS = 6            # 参与+coord-mapped 触点 < 此值 -> low_contact_count
LOW_SUPPORT_FRAC = 0.25     # 中位 support < 此值 -> low_support
```

> 敏感性检查（Task 10 完成后跑一次，记进 archive）：`S_THRESH ∈ {0.10,0.15,0.20}`、`OVERLAP_MIN ∈ {15,25,40}`、`GRID_N ∈ {61,81,101}` 下 §9 的 percentile/z 结论方向不变。

---

### Task 1: signed-transverse 轴 + 对称平面网格

**Files:**
- Create: `src/propagation_contact_plane_readout.py`
- Test: `tests/test_propagation_contact_plane_readout.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_propagation_contact_plane_readout.py
import sys
from pathlib import Path
import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import propagation_contact_plane_readout as R


def _perp_from_coords(coords, src_idx, snk_idx):
    """Helper: replicate compute_axis_frame 的轴外残差 perp_vec (n,3)."""
    src_c = np.nanmean(coords[src_idx], axis=0)
    snk_c = np.nanmean(coords[snk_idx], axis=0)
    u = (snk_c - src_c) / np.linalg.norm(snk_c - src_c)
    rel = coords - src_c
    along = rel @ u
    return rel - np.outer(along, u)


def test_signed_transverse_determinism_and_mirror():
    # 触点沿 x 排开，y 方向左右各半（带符号才能分开）
    coords = np.array([[0,0,0],[1,0,0],[2,0,0],[1,1.0,0],[1,-1.0,0]], float)
    part = np.array([True]*5)
    perp = _perp_from_coords(coords, [0], [2])
    out = R.signed_transverse_axis(perp, part)
    st = out["signed_transverse"]
    # 左右两触点符号相反
    assert np.sign(st[3]) == -np.sign(st[4])
    # 确定性：再跑一次一致
    out2 = R.signed_transverse_axis(perp, part)
    assert np.allclose(out["signed_transverse"], out2["signed_transverse"], equal_nan=True)
    # 镜像输入 -> 整体变号（不是随机翻面）
    coords_m = coords.copy(); coords_m[:, 1] *= -1
    perp_m = _perp_from_coords(coords_m, [0], [2])
    out_m = R.signed_transverse_axis(perp_m, part)
    assert np.allclose(np.abs(out_m["signed_transverse"]), np.abs(st), equal_nan=True)


def test_signed_transverse_pc1_variance_and_degenerate():
    # 近一维残差 -> pc1 方差解释率高
    coords = np.array([[0,0,0],[1,0,0],[2,0,0],[1,1,0],[1,-1,0]], float)
    perp = _perp_from_coords(coords, [0], [2])
    out = R.signed_transverse_axis(perp, np.array([True]*5))
    assert out["pc1_variance_explained"] > 0.95
    # 真二维散布 -> pc1 解释率明显 < 1（残差在 y、z 两个方向等量铺开；
    # 注意：轴 = x，所以横向点必须在 y 和 z 上都有分量，否则残差仍共线）
    coords2 = np.array([[0,0,0],[1,0,0],[2,0,0],
                        [1,1,0],[1,-1,0],[1,0,1.0],[1,0,-1.0]], float)
    perp2 = _perp_from_coords(coords2, [0], [2])
    out2 = R.signed_transverse_axis(perp2, np.array([True]*7))
    assert out2["pc1_variance_explained"] < 0.95
    # <3 参与 -> 全 NaN
    out3 = R.signed_transverse_axis(perp, np.array([True, True, False, False, False]))
    assert np.isnan(out3["signed_transverse"]).all()


def test_plane_grid_symmetry():
    X, Y = R.make_plane_grid()
    # y 对称：flip 行后 Y == -Y
    assert np.allclose(np.flip(Y, axis=0), -Y)
    assert X.shape == Y.shape == (R.GRID_N, R.GRID_N)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_propagation_contact_plane_readout.py -k "signed_transverse or plane_grid" -v`
Expected: FAIL（`module ... has no attribute 'signed_transverse_axis'`）

- [ ] **Step 3: 写最小实现**

```python
# src/propagation_contact_plane_readout.py
"""Subject-specific 2D 传播触点平面读出（real ↔ model 共同语言）。

Spec: docs/superpowers/specs/2026-06-11-propagation-contact-plane-readout-design.md
新增 only: signed-transverse 轴 / normalized 2D 场 / 镜像不变相关 / model-vs-cohort 比较 /
SOZ overlay 匹配。轴框架/端点核/stereotypy/数据加载复用其它模块。
"""
from __future__ import annotations

from typing import Dict, Sequence, Optional, List

import numpy as np

GRID_N = 81
X_LO, X_HI = -0.5, 1.5
Y_EXT = 1.0
S_THRESH = 0.15
OVERLAP_MIN = 25
POOR_PLANARITY_PC1 = 0.80
MIN_CONTACTS = 6
LOW_SUPPORT_FRAC = 0.25


def make_plane_grid(n: int = GRID_N):
    """对称 normalized 平面网格。Y 关于 0 对称（奇数 n 含 y=0 行），
    np.flip(F, axis=0) 精确实现 y -> -y。返回 (X, Y) 各 (n, n)。"""
    x = np.linspace(X_LO, X_HI, n)
    y = np.linspace(-Y_EXT, Y_EXT, n)
    # row index = y, col index = x；flip(axis=0) 翻 y
    Y, X = np.meshgrid(y, x, indexing="ij")
    return X, Y


def signed_transverse_axis(perp_vec: np.ndarray,
                           participating_mask: np.ndarray) -> Dict[str, object]:
    """带符号横向坐标 = 轴外残差在其第一主方向上的投影。

    perp_vec : (n_ch, 3) 轴外残差（= compute_axis_frame 内部 rel - along*u），
               NaN 行允许（非 mapped / 非参与）。
    participating_mask : (n_ch,) bool。

    符号约定（B1，仅供画图）：令参与触点里 |投影| 最大者为正，确定性。
    返回 signed_transverse (n_ch,)（非参与/退化 = NaN）、v_perp (3,)、
    pc1_variance_explained (float)、n_used (int)。
    """
    perp = np.asarray(perp_vec, float)
    part = np.asarray(participating_mask, bool)
    n_ch = perp.shape[0]
    st = np.full(n_ch, np.nan)
    use = part & ~np.isnan(perp).any(axis=1)
    idx = np.where(use)[0]
    if idx.size < 3:
        return {"signed_transverse": st, "v_perp": np.array([np.nan]*3),
                "pc1_variance_explained": float("nan"), "n_used": int(idx.size)}
    P = perp[idx]
    Pc = P - P.mean(axis=0)
    # SVD of centered residuals; 第一主方向 + 方差解释率
    U, S, Vt = np.linalg.svd(Pc, full_matrices=False)
    v1 = Vt[0]
    var_total = float((S ** 2).sum())
    pc1_var = float((S[0] ** 2) / var_total) if var_total > 1e-12 else float("nan")
    proj = P @ v1
    # B1 确定性符号：|proj| 最大触点为正
    anchor = idx[np.argmax(np.abs(proj))]
    sign = 1.0 if (P[np.argmax(np.abs(proj))] @ v1) >= 0 else -1.0
    v1 = v1 * sign
    st[idx] = P @ v1
    return {"signed_transverse": st, "v_perp": v1,
            "pc1_variance_explained": pc1_var, "n_used": int(idx.size)}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_propagation_contact_plane_readout.py -k "signed_transverse or plane_grid" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/propagation_contact_plane_readout.py tests/test_propagation_contact_plane_readout.py
git commit -m "feat(contact-plane): signed-transverse axis + symmetric plane grid (Task 1)"
```

---

### Task 2: 每触点时序聚合（rank 主、time 副，单位不变）

**Files:**
- Modify: `src/propagation_contact_plane_readout.py`
- Test: `tests/test_propagation_contact_plane_readout.py`

- [ ] **Step 1: 写失败测试**

```python
def test_contact_aggregates_rank_event_size_invariant():
    # 同一空间顺序，两组 event size 不同 -> typical_rank 必须一致（不被 size 污染）
    from src.lagpat_rank_audit import mask_phantom_ranks
    # 3-ch event 与 5-ch event 对同 3 个核心触点给相同顺序
    n_ch = 5
    # event A: 触点 0,1,2 参与，顺序 0<1<2
    # event B: 全 5 个参与，顺序 0<1<2<3<4
    bools = np.zeros((n_ch, 2), bool)
    bools[[0,1,2], 0] = True
    bools[:, 1] = True
    ranks = np.full((n_ch, 2), np.nan)
    ranks[[0,1,2], 0] = [0, 1, 2]
    ranks[:, 1] = [0, 1, 2, 3, 4]
    masked = mask_phantom_ranks(ranks, bools, normalize=True)
    agg = R.contact_aggregates(masked, lag_raw=np.where(bools, ranks, np.nan), bools=bools)
    # 触点 0 在两事件里都是"最早"(归一化 rank=0) -> typical_rank≈0
    assert agg["typical_rank"][0] == pytest.approx(0.0, abs=1e-9)
    # support：触点 3 只在 event B 参与 -> 0.5
    assert agg["support"][3] == pytest.approx(0.5)
    assert agg["support"][0] == pytest.approx(1.0)


def test_contact_aggregates_time_unit_invariant():
    # lag_raw 一份秒、一份 ms(×1000) -> typical_time 场一致（事件内归一化消单位）
    n_ch = 4
    bools = np.ones((n_ch, 1), bool)
    masked = np.array([[0.0],[0.333],[0.667],[1.0]])
    lag_sec = np.array([[0.0],[0.010],[0.020],[0.030]])
    lag_ms = lag_sec * 1000.0
    a = R.contact_aggregates(masked, lag_sec, bools)
    b = R.contact_aggregates(masked, lag_ms, bools)
    assert np.allclose(a["typical_time"], b["typical_time"], equal_nan=True)
    # 归一化后 0..1
    assert a["typical_time"][0] == pytest.approx(0.0)
    assert a["typical_time"][3] == pytest.approx(1.0)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_propagation_contact_plane_readout.py -k "contact_aggregates" -v`
Expected: FAIL（`no attribute 'contact_aggregates'`）

- [ ] **Step 3: 写最小实现**

```python
def contact_aggregates(masked: np.ndarray, lag_raw: np.ndarray,
                       bools: np.ndarray) -> Dict[str, np.ndarray]:
    """每触点时序聚合。

    masked : (n_ch, n_ev) 事件内归一化 masked rank（= mask_phantom_ranks(normalize=True)），
             非参与 = NaN。这就是 spec 的 rank_norm，不再二次归一化。
    lag_raw : (n_ch, n_ev) 原始 lag（秒或 ms，单位无关），非参与 = NaN。
    bools : (n_ch, n_ev) bool。

    返回各 (n_ch,) 数组：
      typical_rank   = nanmedian_e masked            （主，进比较）
      typical_time   = nanmedian_e lag_norm          （副，仅画图）
      support        = 参与事件数 / 总事件数
      uncertainty_rank = nan-IQR_e masked
      uncertainty_time = nan-IQR_e lag_norm
    其中 lag_norm(c,e) = (lag-min)/max(lag-min) 在每事件参与触点内（min/max 消单位）。
    """
    masked = np.asarray(masked, float)
    lag_raw = np.asarray(lag_raw, float)
    bools = np.asarray(bools, bool)
    n_ch, n_ev = masked.shape
    # 事件内 lag 归一化（仅参与触点）
    lag_norm = np.full_like(lag_raw, np.nan)
    for e in range(n_ev):
        idx = np.where(bools[:, e])[0]
        if idx.size == 0:
            continue
        v = lag_raw[idx, e]
        vmin = np.nanmin(v)
        rel = v - vmin
        rmax = np.nanmax(rel)
        lag_norm[idx, e] = rel / rmax if rmax > 1e-12 else 0.0

    def _nan_iqr(a, axis):
        with np.errstate(invalid="ignore"):
            q75 = np.nanpercentile(a, 75, axis=axis)
            q25 = np.nanpercentile(a, 25, axis=axis)
        return q75 - q25

    with np.errstate(invalid="ignore"):
        typ_rank = np.nanmedian(masked, axis=1)
        typ_time = np.nanmedian(lag_norm, axis=1)
    support = bools.sum(axis=1).astype(float) / max(n_ev, 1)
    return {
        "typical_rank": typ_rank,
        "typical_time": typ_time,
        "support": support,
        "uncertainty_rank": _nan_iqr(masked, 1),
        "uncertainty_time": _nan_iqr(lag_norm, 1),
    }
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_propagation_contact_plane_readout.py -k "contact_aggregates" -v`
Expected: PASS（`RuntimeWarning: All-NaN slice` 可忽略，已 errstate 包裹）

- [ ] **Step 5: Commit**

```bash
git add src/propagation_contact_plane_readout.py tests/test_propagation_contact_plane_readout.py
git commit -m "feat(contact-plane): per-contact rank/time/support aggregates, unit-invariant (Task 2)"
```

---

### Task 3: 标准化读出 record 组装 + flags + lag_time_unit

**Files:**
- Modify: `src/propagation_contact_plane_readout.py`
- Test: `tests/test_propagation_contact_plane_readout.py`

- [ ] **Step 1: 写失败测试**

```python
def test_build_readout_record_normalized_coords_and_flags():
    # along_axis_mm + axis_length -> x_norm = along/axis_length; signed_transverse -> y_norm
    n_ch = 7
    names = [f"A{i}" for i in range(n_ch)]
    coords = np.array([[i, 0, 0] for i in range(n_ch)], float)
    coords[3, 1] = 1.0; coords[4, 1] = -1.0   # 两侧横向
    along = np.array([float(i) for i in range(n_ch)])     # 0..6 mm
    axis_length = 6.0
    signed_t = np.array([0,0,0, 1.0, -1.0, 0, 0])
    masked = np.tile(np.linspace(0, 1, n_ch)[:, None], (1, 4))
    bools = np.ones((n_ch, 4), bool)
    rec = R.build_readout_record(
        dataset="yuquan", subject="s1", template_id="t0", names=names,
        along_axis_mm=along, axis_length_mm=axis_length, off_axis_mm=np.zeros(n_ch),
        signed_transverse=signed_t, pc1_variance_explained=0.99,
        masked=masked, lag_raw=masked.copy(), bools=bools,
        soz_first_contacts=set(), lag_time_unit="ms",
        one_dimensional_sampling=False)
    ch = {c["name"]: c for c in rec["channels"]}
    assert ch["A6"]["x_norm"] == pytest.approx(1.0)         # along/axis_length
    assert ch["A3"]["y_norm"] == pytest.approx(1.0 / 6.0)   # signed_t/axis_length
    assert ch["A4"]["y_norm"] == pytest.approx(-1.0 / 6.0)
    assert rec["lag_time_unit"] == "ms"
    assert rec["flags"]["poor_planarity"] is False


def test_build_readout_record_poor_planarity_and_low_contact():
    n_ch = 7
    names = [f"A{i}" for i in range(n_ch)]
    along = np.arange(n_ch, dtype=float)
    masked = np.tile(np.linspace(0, 1, n_ch)[:, None], (1, 4))
    bools = np.ones((n_ch, 4), bool)
    rec = R.build_readout_record(
        dataset="yuquan", subject="s2", template_id="t0", names=names,
        along_axis_mm=along, axis_length_mm=6.0, off_axis_mm=np.zeros(n_ch),
        signed_transverse=np.zeros(n_ch),
        pc1_variance_explained=0.5,                 # < POOR_PLANARITY_PC1
        masked=masked, lag_raw=masked.copy(), bools=bools,
        soz_first_contacts=set(), lag_time_unit="s",
        one_dimensional_sampling=False)
    assert rec["flags"]["poor_planarity"] is True
    # 仅 4 参与触点 (< MIN_CONTACTS=6) -> low_contact_count
    bools_few = np.zeros((n_ch, 4), bool); bools_few[:4, :] = True
    rec2 = R.build_readout_record(
        dataset="yuquan", subject="s3", template_id="t0", names=names,
        along_axis_mm=along, axis_length_mm=6.0, off_axis_mm=np.zeros(n_ch),
        signed_transverse=np.zeros(n_ch), pc1_variance_explained=0.99,
        masked=masked, lag_raw=masked.copy(), bools=bools_few,
        soz_first_contacts=set(), lag_time_unit="s",
        one_dimensional_sampling=False)
    assert rec2["flags"]["low_contact_count"] is True
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_propagation_contact_plane_readout.py -k "build_readout_record" -v`
Expected: FAIL（`no attribute 'build_readout_record'`）

- [ ] **Step 3: 写最小实现**

```python
def build_readout_record(
    *, dataset: str, subject: str, template_id: str, names: Sequence[str],
    along_axis_mm: np.ndarray, axis_length_mm: float, off_axis_mm: np.ndarray,
    signed_transverse: np.ndarray, pc1_variance_explained: float,
    masked: np.ndarray, lag_raw: np.ndarray, bools: np.ndarray,
    soz_first_contacts: set, lag_time_unit: str,
    one_dimensional_sampling: bool,
) -> Dict[str, object]:
    """组装一份标准化 readout record（real / model 同构）。

    along_axis_mm / off_axis_mm 来自 compute_axis_frame；signed_transverse 来自
    signed_transverse_axis；masked = mask_phantom_ranks(normalize=True)。
    x_norm = along/axis_length，y_norm = signed_transverse/axis_length（spec §3 双坐标系）。
    每触点一条（仅 along/signed 非 NaN 的参与触点）。flags 见 spec §10。
    SOZ overlay 仅描述性：标 is_soz（first-contact alias 在 soz_first_contacts 内）。
    """
    from src.propagation_skeleton_geometry import parse_shaft
    from src.sef_hfo_soz_localization import _first_contact

    along = np.asarray(along_axis_mm, float)
    st = np.asarray(signed_transverse, float)
    agg = contact_aggregates(masked, lag_raw, bools)
    L = float(axis_length_mm)
    channels: List[dict] = []
    for i, nm in enumerate(names):
        a_i, s_i = float(along[i]), float(st[i])
        if not (np.isfinite(a_i) and np.isfinite(s_i)) or L < 1e-9:
            continue
        channels.append({
            "name": str(nm),
            "shaft": str(parse_shaft(nm)[0]),
            "along_axis_mm": a_i,
            "signed_transverse_mm": s_i,
            "off_axis_mm": float(off_axis_mm[i]),
            "x_norm": a_i / L,
            "y_norm": s_i / L,
            "typical_rank": float(agg["typical_rank"][i]),
            "typical_time": float(agg["typical_time"][i]),
            "support": float(agg["support"][i]),
            "uncertainty_rank": float(agg["uncertainty_rank"][i]),
            "is_soz": _first_contact(str(nm)) in soz_first_contacts,
        })
    med_support = float(np.median([c["support"] for c in channels])) if channels else 0.0
    flags = {
        "one_dimensional_sampling": bool(one_dimensional_sampling),
        "poor_planarity": bool(np.isfinite(pc1_variance_explained)
                               and pc1_variance_explained < POOR_PLANARITY_PC1),
        "low_contact_count": len(channels) < MIN_CONTACTS,
        "low_support": med_support < LOW_SUPPORT_FRAC,
        "weak_axis": L < 1e-9,
    }
    return {
        "dataset": dataset, "subject": subject, "template_id": template_id,
        "axis_length_mm": L,
        "transverse_pc1_variance_explained": float(pc1_variance_explained),
        "lag_time_unit": lag_time_unit,
        "channels": channels,
        "flags": flags,
        "n_channels": len(channels),
    }
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_propagation_contact_plane_readout.py -k "build_readout_record" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/propagation_contact_plane_readout.py tests/test_propagation_contact_plane_readout.py
git commit -m "feat(contact-plane): standardized readout record + flags + lag_time_unit (Task 3)"
```

---

### Task 4: 连续场 T / S / U（support 加权 kernel，gating）

**Files:**
- Modify: `src/propagation_contact_plane_readout.py`
- Test: `tests/test_propagation_contact_plane_readout.py`

- [ ] **Step 1: 写失败测试**

```python
def test_smooth_field_support_weight_and_gate():
    # 两触点 (x_norm,y_norm,typical_rank,support)，sigma 小 -> 各自附近显色
    rec = {"channels": [
        {"x_norm": 0.1, "y_norm": 0.0, "typical_rank": 0.0, "support": 1.0,
         "uncertainty_rank": 0.0},
        {"x_norm": 0.9, "y_norm": 0.0, "typical_rank": 1.0, "support": 1.0,
         "uncertainty_rank": 0.0},
    ]}
    X, Y = R.make_plane_grid()
    fld = R.smooth_field(rec, X, Y, sigma_xy=0.05, scalar="rank")
    assert fld["T"].shape == X.shape
    # S 是支撑权重（>=0），不是事件率
    assert (fld["S"] >= 0).all()
    # 远离两触点的像素 S 很低 -> 被 gate 掉（mask=False）
    far = (np.abs(X - 0.5) < 0.02) & (np.abs(Y) < 0.02)   # 网格中央、无触点
    assert fld["mask"][far].sum() == 0
    # 触点 0 附近 T≈0，触点 1 附近 T≈1
    near0 = np.unravel_index(np.argmin((X-0.1)**2 + (Y-0)**2), X.shape)
    near1 = np.unravel_index(np.argmin((X-0.9)**2 + (Y-0)**2), X.shape)
    assert fld["T"][near0] == pytest.approx(0.0, abs=0.2)
    assert fld["T"][near1] == pytest.approx(1.0, abs=0.2)


def test_smooth_field_sigma_default_nn_spacing():
    rec = {"channels": [
        {"x_norm": 0.0, "y_norm": 0.0, "typical_rank": 0.0, "support": 1.0,
         "uncertainty_rank": 0.0},
        {"x_norm": 0.3, "y_norm": 0.0, "typical_rank": 1.0, "support": 1.0,
         "uncertainty_rank": 0.0},
    ]}
    X, Y = R.make_plane_grid()
    fld = R.smooth_field(rec, X, Y, sigma_xy=None, scalar="rank")
    # 默认 sigma = 最近邻间距中位数 = 0.3
    assert fld["sigma_xy"] == pytest.approx(0.3, abs=1e-9)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_propagation_contact_plane_readout.py -k "smooth_field" -v`
Expected: FAIL（`no attribute 'smooth_field'`）

- [ ] **Step 3: 写最小实现**

```python
def _median_nn_spacing(pts: np.ndarray) -> float:
    """最近邻间距中位数（normalized plane）。"""
    if pts.shape[0] < 2:
        return 0.1
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def smooth_field(record: Dict, X: np.ndarray, Y: np.ndarray,
                 sigma_xy: Optional[float] = None, scalar: str = "rank",
                 s_thresh: float = S_THRESH) -> Dict[str, object]:
    """normalized 平面上的 support 加权 kernel regression。

    每触点提供 (x_norm, y_norm, value, support)，value = typical_rank(scalar='rank')
    或 typical_time('time')。
      w_i(x,y) = support_i * exp(-((x-x_i)^2+(y-y_i)^2)/(2 sigma^2))
      T = Σ w_i value_i / Σ w_i ；S = Σ w_i（支撑权重，NOT 事件率）；
      U = Σ w_i unc_i / Σ w_i ；mask = S >= s_thresh。
    返回 T/S/U/mask（均 (n,n)）+ sigma_xy。
    """
    val_key = "typical_rank" if scalar == "rank" else "typical_time"
    # 过滤：坐标有限 AND 该标量有限 AND support>0。否则 NaN-value 通道仍会把权重
    # 加进 S 分母却不进 T 分子 -> 稀释 T（reviewer P1）。
    chans = [c for c in record["channels"]
             if np.isfinite(c["x_norm"]) and np.isfinite(c["y_norm"])
             and np.isfinite(c.get(val_key, np.nan)) and c.get("support", 0) > 0]
    pts = np.array([[c["x_norm"], c["y_norm"]] for c in chans], float).reshape(-1, 2)
    vals = np.array([c[val_key] for c in chans], float)
    sup = np.array([c["support"] for c in chans], float)
    unc = np.array([c.get("uncertainty_rank", 0.0) for c in chans], float)
    if sigma_xy is None:
        sigma_xy = _median_nn_spacing(pts)
    sig2 = 2.0 * sigma_xy ** 2
    gx = X.ravel(); gy = Y.ravel()
    S = np.zeros(gx.shape); WT = np.zeros(gx.shape); WU = np.zeros(gx.shape)
    for k in range(pts.shape[0]):
        d2 = (gx - pts[k, 0]) ** 2 + (gy - pts[k, 1]) ** 2
        w = sup[k] * np.exp(-d2 / sig2)
        S += w
        if np.isfinite(vals[k]):
            WT += w * vals[k]
        WU += w * unc[k]
    with np.errstate(invalid="ignore", divide="ignore"):
        T = np.where(S > 1e-12, WT / S, np.nan).reshape(X.shape)
        U = np.where(S > 1e-12, WU / S, np.nan).reshape(X.shape)
    S = S.reshape(X.shape)
    mask = S >= s_thresh
    return {"T": T, "S": S, "U": U, "mask": mask, "sigma_xy": float(sigma_xy)}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_propagation_contact_plane_readout.py -k "smooth_field" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/propagation_contact_plane_readout.py tests/test_propagation_contact_plane_readout.py
git commit -m "feat(contact-plane): support-weighted T/S/U field + gating (Task 4)"
```

---

### Task 5: 镜像不变、support-gated 相关（P0 lock）

**Files:**
- Modify: `src/propagation_contact_plane_readout.py`
- Test: `tests/test_propagation_contact_plane_readout.py`

- [ ] **Step 1: 写失败测试**

```python
def test_corr_pair_mirror_invariant():
    X, Y = R.make_plane_grid()
    # 一个沿 +y 偏的场
    F1 = np.exp(-((X-0.5)**2 + (Y-0.4)**2)/0.05)
    S1 = F1.copy()
    # F2 = F1 沿 y 翻面（即镜像副本）
    F2 = np.flip(F1, axis=0)
    S2 = np.flip(S1, axis=0)
    out = R.corr_pair_mirror_invariant(F1, S1, F2, S2, s_thresh=0.1, overlap_min=5)
    # 镜像不变：max-over-mirror 必须把翻面的 F2 对回去 -> corr≈1
    assert out["corr"] == pytest.approx(1.0, abs=1e-6)
    assert out["insufficient_overlap"] is False


def test_corr_pair_low_overlap_flag():
    X, Y = R.make_plane_grid()
    # 两个支撑几乎不交叠的场
    F1 = np.exp(-((X-0.1)**2 + Y**2)/0.01); S1 = F1.copy()
    F2 = np.exp(-((X-1.4)**2 + Y**2)/0.01); S2 = F2.copy()
    out = R.corr_pair_mirror_invariant(F1, S1, F2, S2, s_thresh=0.5, overlap_min=25)
    assert out["insufficient_overlap"] is True
    assert out["corr"] is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_propagation_contact_plane_readout.py -k "corr_pair" -v`
Expected: FAIL（`no attribute 'corr_pair_mirror_invariant'`）

- [ ] **Step 3: 写最小实现**

```python
def _support_corr(F1, F2, S1, S2, s_thresh):
    """两场在双方 support>=thresh 像素交集上的 Pearson 相关 + 交集像素数。"""
    m = (S1 >= s_thresh) & (S2 >= s_thresh) & np.isfinite(F1) & np.isfinite(F2)
    n = int(m.sum())
    if n < 2:
        return float("nan"), n
    a = F1[m]; b = F2[m]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan"), n
    return float(np.corrcoef(a, b)[0, 1]), n


def corr_pair_mirror_invariant(F1, S1, F2, S2, s_thresh: float = S_THRESH,
                               overlap_min: int = OVERLAP_MIN) -> Dict[str, object]:
    """y-reflection invariant、support-gated 场相关（spec §4 P0 lock）。

    signed-y 只是图上稳定坐标，不是解剖左右；所以取
    corr = max(corr(F1,F2), corr(F1, flip_y(F2)))。
    仅在双方 S>=s_thresh 像素交集上算；交集 < overlap_min -> corr=None +
    insufficient_overlap=True。网格须 y 对称（make_plane_grid 保证），故
    np.flip(axis=0) == y -> -y。
    """
    c_id, n_id = _support_corr(F1, F2, S1, S2, s_thresh)
    F2m = np.flip(F2, axis=0); S2m = np.flip(S2, axis=0)
    c_mir, n_mir = _support_corr(F1, F2m, S1, S2m, s_thresh)
    n_overlap = max(n_id, n_mir)
    if n_overlap < overlap_min:
        return {"corr": None, "n_overlap": n_overlap, "insufficient_overlap": True}
    cands = [c for c in (c_id, c_mir) if np.isfinite(c)]
    if not cands:
        return {"corr": None, "n_overlap": n_overlap, "insufficient_overlap": True}
    return {"corr": float(max(cands)), "n_overlap": n_overlap,
            "insufficient_overlap": False}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_propagation_contact_plane_readout.py -k "corr_pair" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/propagation_contact_plane_readout.py tests/test_propagation_contact_plane_readout.py
git commit -m "feat(contact-plane): mirror-invariant support-gated field correlation (Task 5)"
```

---

### Task 6: model-vs-cohort 比较（描述性，subject-first 聚合）

**Files:**
- Modify: `src/propagation_contact_plane_readout.py`
- Test: `tests/test_propagation_contact_plane_readout.py`

- [ ] **Step 1: 写失败测试**

```python
def test_robust_z_and_percentile():
    dist = [0.1, 0.2, 0.3, 0.4, 0.5]
    out = R.placement_in_distribution(0.3, dist)
    assert out["percentile"] == pytest.approx(50.0, abs=15)   # 中位附近
    assert out["robust_z"] == pytest.approx(0.0, abs=1e-9)     # = median


def test_subject_first_folding_no_overweight():
    # subject 'A' 有 3 个 template record，'B' 有 1 个；折叠后各算 1 个 subject 值
    recs = [
        {"dataset": "yuquan", "subject": "A", "scalar": 0.0},
        {"dataset": "yuquan", "subject": "A", "scalar": 0.0},
        {"dataset": "yuquan", "subject": "A", "scalar": 0.0},
        {"dataset": "yuquan", "subject": "B", "scalar": 1.0},
    ]
    folded = R.subject_first_fold(recs, key="scalar")
    # 两个 subject -> 长度 2，不是 4
    assert len(folded) == 2
    assert sorted(folded) == [0.0, 1.0]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_propagation_contact_plane_readout.py -k "robust_z or subject_first" -v`
Expected: FAIL

- [ ] **Step 3: 写最小实现**

```python
def placement_in_distribution(value: float, dist: Sequence[float]) -> Dict[str, float]:
    """value 落在 dist 的 percentile + robust z（median + MAD）。不报 p 值（spec §9）。"""
    d = np.asarray([x for x in dist if np.isfinite(x)], float)
    if d.size == 0 or not np.isfinite(value):
        return {"percentile": float("nan"), "robust_z": float("nan"), "n": int(d.size)}
    pct = float((d < value).mean() * 100.0)
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    rz = float((value - med) / (1.4826 * mad)) if mad > 1e-12 else float("nan")
    return {"percentile": pct, "robust_z": rz, "n": int(d.size)}


def subject_first_fold(records: Sequence[dict], key: str) -> List[float]:
    """多模板 subject 先折叠为一个代表值（同 subject 的 key 取中位），再返回每 subject 一个值。
    防 cohort 汇总里模板多的 subject 被重复计数（spec §9 聚合纪律）。"""
    by_subj: Dict[tuple, list] = {}
    for r in records:
        v = r.get(key)
        if v is None or not np.isfinite(v):
            continue
        by_subj.setdefault((r["dataset"], r["subject"]), []).append(float(v))
    return [float(np.median(vs)) for vs in by_subj.values()]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_propagation_contact_plane_readout.py -k "robust_z or subject_first" -v`
Expected: PASS

- [ ] **Step 5: 写 compare_model_to_cohort（组装层）+ 测试**

```python
# 测试
def test_compare_model_to_cohort_scalar_and_field():
    X, Y = R.make_plane_grid()
    def mk(subj, tid, shift):
        chans = [{"x_norm": x, "y_norm": 0.0, "typical_rank": x, "support": 1.0,
                  "uncertainty_rank": 0.0} for x in np.linspace(0.1, 0.9, 8)]
        return {"dataset": "yuquan", "subject": subj, "template_id": tid,
                "axis_length_mm": 30.0 + shift, "channels": chans,
                "scalars": {"axis_length_mm": 30.0 + shift}}
    reals = [mk(f"s{i}", "t0", i) for i in range(5)]
    model = mk("model", "t0", 2)
    out = R.compare_model_to_cohort(model, reals, X, Y,
                                    sigma_xy=0.1, s_thresh=0.05, overlap_min=5)
    assert "scalar_placement" in out and "axis_length_mm" in out["scalar_placement"]
    assert "field_placement" in out
    # field：model_to_real 中位相关 落在 real_to_real 分布里 -> 有 percentile
    assert np.isfinite(out["field_placement"]["model_to_real_median_corr"])
```

```python
def compare_model_to_cohort(model_record: dict, real_records: Sequence[dict],
                            X: np.ndarray, Y: np.ndarray,
                            sigma_xy: Optional[float] = None,
                            s_thresh: float = S_THRESH,
                            overlap_min: int = OVERLAP_MIN) -> Dict[str, object]:
    """spec §9 描述性 posterior-predictive 比较。

    (a) 标量：对每个 real-vs-model cohort scalar，real subject-first 折叠成分布，
        model 取 placement_in_distribution（percentile + robust z）。
    (b) field：先给每个 record 算 rank 场；real_to_real(i)=median_{j≠i} corr_pair；
        model_to_real=median_i corr_pair(model,i)；报 model 值落在
        {real_to_real} 分布的 percentile/z。镜像不变 + support-gated。
    禁报 p 值。
    """
    SCALARS = ["axis_length_mm", "transverse_width_mm", "early_zone_spread",
               "late_zone_spread", "early_late_centroid_distance_norm",
               "rank_vs_xnorm_spearman"]
    # (a) 标量
    scalar_placement = {}
    for s in SCALARS:
        dist = subject_first_fold(
            [{"dataset": r["dataset"], "subject": r["subject"],
              "scalar": r.get("scalars", {}).get(s)} for r in real_records],
            key="scalar")
        mv = model_record.get("scalars", {}).get(s)
        if mv is not None and dist:
            scalar_placement[s] = placement_in_distribution(float(mv), dist)
    # (b) field —— subject-first（reviewer P1：同 subject 多模板不得自我膨胀/过度加权）
    def fld(rec):
        return R_smooth_rank(rec, X, Y, sigma_xy, s_thresh)
    def _subj(r):
        return (r["dataset"], r["subject"])
    real_flds = [fld(r) for r in real_records]
    model_fld = fld(model_record)
    # real-to-real：同一 (dataset,subject)（含其它模板）跳过；每 record 取对【其他 subject】
    # 的 median，再 subject_first_fold 折成每 subject 一个值
    r2r = []
    for i, fi in enumerate(real_flds):
        cs = []
        for j, fj in enumerate(real_flds):
            if _subj(real_records[i]) == _subj(real_records[j]):
                continue
            c = corr_pair_mirror_invariant(fi["T"], fi["S"], fj["T"], fj["S"],
                                           s_thresh, overlap_min)["corr"]
            if c is not None:
                cs.append(c)
        if cs:
            r2r.append({"dataset": real_records[i]["dataset"],
                        "subject": real_records[i]["subject"],
                        "scalar": float(np.median(cs))})
    r2r_dist = subject_first_fold(r2r, key="scalar")
    # model-to-real：先按 real subject 折叠 corr（同 subject 多模板取 median），再跨 subject median
    m2r_by_subj: Dict[tuple, list] = {}
    for fi, ri in zip(real_flds, real_records):
        c = corr_pair_mirror_invariant(model_fld["T"], model_fld["S"],
                                       fi["T"], fi["S"], s_thresh, overlap_min)["corr"]
        if c is not None:
            m2r_by_subj.setdefault(_subj(ri), []).append(c)
    m2r_subj = [float(np.median(v)) for v in m2r_by_subj.values()]
    m2r_med = float(np.median(m2r_subj)) if m2r_subj else float("nan")
    field_placement = {
        "model_to_real_median_corr": m2r_med,
        "real_to_real_distribution_n": len(r2r_dist),
        "placement": placement_in_distribution(m2r_med, r2r_dist),
    }
    return {"scalar_placement": scalar_placement, "field_placement": field_placement,
            "note": "descriptive posterior-predictive; no p-value; SOZ not a metric"}


def R_smooth_rank(rec, X, Y, sigma_xy, s_thresh):
    """compare 内部用：取 rank 场（薄封装 smooth_field）。"""
    return smooth_field(rec, X, Y, sigma_xy=sigma_xy, scalar="rank", s_thresh=s_thresh)
```

- [ ] **Step 6: 跑测试确认通过**

Run: `pytest tests/test_propagation_contact_plane_readout.py -k "compare_model_to_cohort" -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/propagation_contact_plane_readout.py tests/test_propagation_contact_plane_readout.py
git commit -m "feat(contact-plane): descriptive model-vs-cohort comparison, subject-first (Task 6)"
```

---

### Task 7: SOZ overlay 匹配（first-contact alias lock）+ cohort scalars

**Files:**
- Modify: `src/propagation_contact_plane_readout.py`
- Test: `tests/test_propagation_contact_plane_readout.py`

- [ ] **Step 1: 写失败测试**

```python
def test_soz_first_contact_alias_and_ambiguity():
    # 单极 montage：精确匹配
    out = R.resolve_soz_overlay(["E11", "E12", "A6"], soz_core={"E11", "A6"},
                                montage="single")
    assert out["soz_first_contacts"] == {"E11", "A6"}
    assert out["soz_ambiguous"] == []
    # 双极 montage：first-contact alias（'E11-E12' -> 'E11'）
    out2 = R.resolve_soz_overlay(["E11-E12", "E12-E13", "A6-A7"],
                                 soz_core={"E11"}, montage="bipolar")
    assert "E11" in out2["soz_first_contacts"]
    # 歧义：SOZ contact 映射到 >=2 个 montage 通道 -> 记 ambiguous，不强配
    out3 = R.resolve_soz_overlay(["E11-E12", "E11-E10"], soz_core={"E11"},
                                 montage="bipolar")
    assert "E11" in out3["soz_ambiguous"]


def test_cohort_scalars_rank_vs_xnorm_no_theta_ref():
    # rank_vs_xnorm_spearman 真实数据自身有定义（不需 theta_ref）
    rec = {"channels": [{"x_norm": x, "typical_rank": x, "y_norm": 0.0,
                         "signed_transverse_mm": 0.0, "along_axis_mm": x*30}
                        for x in np.linspace(0, 1, 8)],
           "axis_length_mm": 30.0}
    sc = R.compute_cohort_scalars(rec)
    assert sc["rank_vs_xnorm_spearman"] == pytest.approx(1.0, abs=1e-6)
    assert "axis_angle_error" not in sc          # model-only validation, 不在 cohort scalar
    assert "transverse_width_mm" in sc
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_propagation_contact_plane_readout.py -k "soz_first_contact or cohort_scalars" -v`
Expected: FAIL

- [ ] **Step 3: 写最小实现**

```python
def resolve_soz_overlay(names: Sequence[str], soz_core: set,
                        montage: str) -> Dict[str, object]:
    """SOZ overlay 匹配，政策锁 = first-contact alias（spec §7）。

    single montage：montage 名即 contact，精确匹配 soz_core。
    bipolar montage：每通道取 _first_contact（'E11-E12'->'E11'），匹配 soz_core；
      一个 SOZ contact 映射到 >=2 个通道 -> ambiguous（记录，不强配）。
    返回 soz_first_contacts（用于 record is_soz）+ soz_ambiguous（图注标注）。
    仅描述性，不进任何 metric。
    """
    from src.sef_hfo_soz_localization import _first_contact
    if montage == "single":
        hit = {n for n in names if n in soz_core}
        return {"soz_first_contacts": hit, "soz_ambiguous": []}
    first_map: Dict[str, list] = {}
    for n in names:
        first_map.setdefault(_first_contact(n), []).append(n)
    hit, ambig = set(), []
    for contact in soz_core:
        pairs = first_map.get(contact, [])
        if len(pairs) == 1:
            hit.add(contact)
        elif len(pairs) >= 2:
            ambig.append(contact)
    return {"soz_first_contacts": hit, "soz_ambiguous": sorted(ambig)}


def compute_cohort_scalars(record: dict) -> Dict[str, float]:
    """real-vs-model cohort scalar（spec §9 (a)，均真实数据自身有定义；NO theta_ref）。"""
    from scipy.stats import spearmanr
    chans = record["channels"]
    x = np.array([c["x_norm"] for c in chans], float)
    rk = np.array([c["typical_rank"] for c in chans], float)
    st = np.array([c["signed_transverse_mm"] for c in chans], float)
    ok = np.isfinite(x) & np.isfinite(rk)
    rho = float(spearmanr(x[ok], rk[ok]).correlation) if ok.sum() >= 3 else float("nan")
    # transverse_width：signed_transverse 的稳健展宽（p90-p10），mm
    tw = float(np.nanpercentile(st, 90) - np.nanpercentile(st, 10)) if st.size else float("nan")
    # 早/晚端：按 typical_rank 取前/后 1/3 的 signed_transverse 展宽 + 沿轴中心距
    order = np.argsort(rk)
    k = max(1, len(order) // 3)
    early, late = order[:k], order[-k:]
    along = np.array([c["along_axis_mm"] for c in chans], float)
    early_late_dist = abs(float(np.nanmean(along[late]) - np.nanmean(along[early])))
    L = record.get("axis_length_mm", float("nan"))
    return {
        "axis_length_mm": float(L),
        "transverse_width_mm": tw,
        "early_zone_spread": float(np.nanstd(st[early])) if k else float("nan"),
        "late_zone_spread": float(np.nanstd(st[late])) if k else float("nan"),
        "early_late_centroid_distance_norm":
            early_late_dist / L if L and np.isfinite(L) and L > 1e-9 else float("nan"),
        "rank_vs_xnorm_spearman": rho,
    }
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_propagation_contact_plane_readout.py -k "soz_first_contact or cohort_scalars" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/propagation_contact_plane_readout.py tests/test_propagation_contact_plane_readout.py
git commit -m "feat(contact-plane): SOZ first-contact overlay + cohort scalars (Task 7)"
```

---

### Task 8: 真实 subject runner

**Files:**
- Create: `scripts/run_contact_plane_readout.py`
- Test: `tests/test_propagation_contact_plane_readout.py`（schema smoke，mount-free）

- [ ] **Step 1: 写失败测试（用合成 record，不碰 /mnt）**

```python
def test_runner_build_record_from_arrays():
    # runner 的核心 build_record_from_events 应能从内存数组产出合法 record
    from scripts.run_contact_plane_readout import build_record_from_events
    # classify_sampling_geometry: one_d = (len(shafts)<=1) OR (p90_off < spacing_mm)。
    # 单 shaft 直接判 1D，所以必须给两根 shaft(A/B) + 横向散布 >= spacing_mm，
    # 否则正确实现也会标 1D（reviewer P0）。
    names = [f"A{i}" for i in range(4)] + [f"B{i}" for i in range(4)]
    coords = np.array([
        [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0],
        [1, 2.0, 0], [2, -2.0, 0], [4, 0, 0], [5, 0, 0],
    ], float)
    n_ch = len(names)
    mapped = np.ones(n_ch, bool)
    ranks = np.tile(np.arange(n_ch)[:, None], (1, 6)).astype(float)
    bools = np.ones((n_ch, 6), bool)
    rec = build_record_from_events(
        dataset="yuquan", subject="s1", template_id="t_a",
        names=names, ranks=ranks, bools=bools, lag_raw=ranks.copy(),
        coords=coords, mapped=mapped, soz_core=set(), montage="single",
        lag_time_unit="ms", spacing_mm=1.0)
    assert rec["flags"]["weak_axis"] is False
    assert rec["n_channels"] >= R.MIN_CONTACTS
    assert "scalars" in rec and "rank_vs_xnorm_spearman" in rec["scalars"]
    # 两 shaft + p90 off-axis(≈1.9) >= spacing(1.0) -> 不是 1D
    assert rec["flags"]["one_dimensional_sampling"] is False
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_propagation_contact_plane_readout.py -k "runner_build_record" -v`
Expected: FAIL（`No module named scripts.run_contact_plane_readout`）

- [ ] **Step 3: 写 runner**

```python
#!/usr/bin/env python3
"""真实 subject 2D 传播触点平面读出。
Spec: docs/superpowers/specs/2026-06-11-propagation-contact-plane-readout-design.md
Out:  results/spatial_modulation/propagation_geometry/observation_readout/real_subjects/
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
warnings.filterwarnings("ignore", message="Mean of empty slice")

from src.interictal_propagation import load_subject_propagation_events
from src.lagpat_rank_audit import mask_phantom_ranks
from src.seeg_coord_loader import load_subject_coords
from src.sef_hfo_soz_localization import classify_montage, _first_contact
from src import propagation_skeleton_geometry as G
from src import propagation_contact_plane_readout as R

YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
RANKDISP = _ROOT / "results/interictal_propagation_masked/rank_displacement/per_subject"
SOZ_JSON = {ds: _ROOT / f"results/{ds}_soz_core_channels.json"
            for ds in ("yuquan", "epilepsiae")}
OUT = _ROOT / "results/spatial_modulation/propagation_geometry/observation_readout/real_subjects"
EXCLUDE_BAD_DATA = {("yuquan", "pengzihang")}


def _subject_dir(ds, subj):
    return YUQUAN_ROOT / subj if ds == "yuquan" else EPILEPSIAE_ROOT / subj / "all_recs"


def _soz_set(ds, subj):
    try:
        d = json.loads(SOZ_JSON[ds].read_text())
        entry = d.get(subj, d.get(str(subj)))
        if isinstance(entry, dict):
            entry = entry.get("core_channels", [])
        return set(entry or [])
    except Exception:
        return set()


def _load_accepted_templates(ds, subj, names):
    """复用 skeleton runner 的 accepted-template 加载（rank-displacement pair[0]）。"""
    f = RANKDISP / f"{ds}_{subj}.json"
    if not f.exists():
        raise FileNotFoundError(f"rank-displacement JSON missing for {ds}:{subj} ({f})")
    d = json.loads(f.read_text())
    pair = (d.get("pairs") or [{}])[0]
    rd_names = list(pair.get("channel_names") or [])
    ra = np.asarray(pair.get("rank_a_dense_full"), float)
    rb = np.asarray(pair.get("rank_b_dense_full"), float)
    idx_a = {nm: ra[i] for i, nm in enumerate(rd_names)}
    idx_b = {nm: rb[i] for i, nm in enumerate(rd_names)}
    ta = np.array([idx_a.get(nm, np.nan) for nm in names], float)
    tb = np.array([idx_b.get(nm, np.nan) for nm in names], float)
    swap = (((pair.get("swap_sweep") or {}).get("swap_class")) or "none")
    return ta, tb, swap


def build_record_from_events(*, dataset, subject, template_id, names, ranks, bools,
                             lag_raw, coords, mapped, soz_core, montage,
                             lag_time_unit, spacing_mm, template_axis=None):
    """事件数组 -> 标准化 readout record（mount-free，单测入口）。

    1D 采样判定在 compute_axis_frame 之后，用【真实】fr['off_axis'] + participating
    mask（reviewer P1：不得用全零 off-axis 提前判，会系统性误标 1D）。
    """
    masked = mask_phantom_ranks(ranks, bools, normalize=True)
    if template_axis is None:
        template_axis = np.array(
            [np.nanmean(r) if np.any(~np.isnan(r)) else np.nan for r in masked])
    eligible = (~np.isnan(template_axis)) & np.asarray(mapped, bool)
    cores = G.build_endpoint_cores(template_axis, eligible, k_primary=3)
    if cores["tier"] == "descriptive_only":
        return {"dataset": dataset, "subject": subject, "template_id": template_id,
                "status": "descriptive_only"}
    fr = G.compute_axis_frame(coords, cores["source_idx"], cores["sink_idx"])
    # 轴外残差 perp_vec：rel - along*u
    src_c = np.array(fr["source_centroid"])
    axis = np.array(fr["sink_centroid"]) - src_c
    u = axis / max(np.linalg.norm(axis), 1e-12)
    rel = np.asarray(coords, float) - src_c
    along = np.asarray(fr["along_axis"], float)
    perp_vec = rel - np.outer(np.where(np.isnan(along), 0.0, along), u)
    perp_vec[np.isnan(coords).any(axis=1)] = np.nan
    participating = bools.any(axis=1) & eligible
    st = R.signed_transverse_axis(perp_vec, participating)
    # 1D 判定（真实 off-axis，不是全零）
    samp = G.classify_sampling_geometry(
        names, participating, np.asarray(fr["off_axis"], float), spacing_mm=spacing_mm)
    one_d = samp.get("geometry") == "1D"
    soz = R.resolve_soz_overlay(list(names), soz_core, montage)
    rec = R.build_readout_record(
        dataset=dataset, subject=subject, template_id=template_id, names=names,
        along_axis_mm=along, axis_length_mm=fr["axis_length"],
        off_axis_mm=np.asarray(fr["off_axis"], float),
        signed_transverse=st["signed_transverse"],
        pc1_variance_explained=st["pc1_variance_explained"],
        masked=masked, lag_raw=lag_raw, bools=bools,
        soz_first_contacts=soz["soz_first_contacts"], lag_time_unit=lag_time_unit,
        one_dimensional_sampling=one_d)
    rec["soz_ambiguous"] = soz["soz_ambiguous"]
    rec["sampling_geometry"] = samp.get("geometry")
    rec["scalars"] = R.compute_cohort_scalars(rec)
    return rec


def process_subject(ds, subj):
    ev = load_subject_propagation_events(_subject_dir(ds, subj))
    if not ev["channel_names"] or np.asarray(ev["ranks"]).size == 0:
        return [{"dataset": ds, "subject": subj, "status": "no_events"}]
    names = list(ev["channel_names"])
    ranks = np.asarray(ev["ranks"], float)
    bools = np.asarray(ev["bools"]) > 0
    masked = mask_phantom_ranks(ranks, bools, normalize=True)
    ta, tb, swap = _load_accepted_templates(ds, subj, names)
    labels = G.assign_events_to_templates(masked, ta, tb)
    cr = load_subject_coords(ds, subj, names)
    coords = np.asarray(cr.coords_array_in_requested_order, float)
    mapped = np.asarray(cr.mapped_mask_in_requested_order, bool)
    # montage 类型（single/bipolar）从通道名判定
    montage = "bipolar" if sum("-" in n for n in names) >= len(names) / 2 else "single"
    spacing = 3.5 if ds == "yuquan" else 4.6
    # hard-assert lag_raw 存在（reviewer：不得 silently 用 rank 伪造 time 副图）
    if "lag_raw" not in ev:
        raise KeyError(f"{ds}:{subj} load_subject_propagation_events 缺 lag_raw 键")
    lag_raw = np.where(bools, np.asarray(ev["lag_raw"], float), np.nan)
    out = []
    # 逐 template（A/B 两个 accepted 模板）各出一份 record（spec §8 逐模板处理）。
    # 命名 t_a/t_b：只是 rank-displacement pair 的两支，NOT dominant/minority 的科学事实。
    for tid, lbl in (("t_a", 0), ("t_b", 1)):
        sel = labels == lbl
        if sel.sum() == 0:
            continue
        tmask = masked[:, sel]
        taxis = np.array([np.nanmean(r) if np.any(~np.isnan(r)) else np.nan
                          for r in tmask])
        rec = build_record_from_events(
            dataset=ds, subject=subj, template_id=tid, names=names,
            ranks=ranks[:, sel], bools=bools[:, sel], lag_raw=lag_raw[:, sel],
            coords=coords, mapped=mapped, soz_core=_soz_set(ds, subj),
            montage=montage, lag_time_unit="s", spacing_mm=spacing,
            template_axis=taxis)
        rec["swap_class"] = swap
        out.append(rec)
    return out


def discover_cohort():
    return [tuple(f.stem.split("_", 1)) for f in sorted(RANKDISP.glob("*.json"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cohort = ([tuple(s.split(":", 1)) for s in args.subjects] if args.subjects
              else discover_cohort())
    cohort = [(d, s) for d, s in cohort if (d, s) not in EXCLUDE_BAD_DATA]
    n_ok = 0
    for ds, subj in cohort:
        recs = process_subject(ds, subj)
        for rec in recs:
            tid = rec.get("template_id", "t0")
            (out / f"{ds}_{subj}_{tid}.json").write_text(
                json.dumps(rec, indent=2, default=float))
            if rec.get("status") not in ("no_events", "descriptive_only"):
                n_ok += 1
    if n_ok == 0:
        raise SystemExit("no usable real readout records — refusing vacuous run")
    print(f"wrote real readout records for {len(cohort)} subjects ({n_ok} usable)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_propagation_contact_plane_readout.py -k "runner_build_record" -v`
Expected: PASS

- [ ] **Step 5: 真实 smoke（若 /mnt 挂载）**

Run: `python scripts/run_contact_plane_readout.py --subjects yuquan:chengshuai`
Expected: 写出 `results/.../real_subjects/yuquan_chengshuai_t_a.json` + `..._t_b.json`；若 /mnt 未挂载则跳过（记录在 archive）。

- [ ] **Step 6: Commit**

```bash
git add scripts/run_contact_plane_readout.py tests/test_propagation_contact_plane_readout.py
git commit -m "feat(contact-plane): real-subject readout runner (Task 8)"
```

---

### Task 9: 模型 runner（复用观测层 NPZ，走同一读出）

**Files:**
- Create: `scripts/run_model_contact_plane_readout.py`
- Test: `tests/test_propagation_contact_plane_readout.py`

- [ ] **Step 1: 写失败测试（用已存在的模型 NPZ fixture）**

```python
def test_model_runner_reads_observation_npz():
    from scripts.run_model_contact_plane_readout import build_model_record
    npz = (_ROOT / "results/topic4_sef_hfo/observation_layer/"
           "increment1_toywave/example30_lagPat_withFreqCent.npz")
    if not npz.exists():
        pytest.skip("model observation NPZ fixture absent")
    # 核心：legacy-key 读取 + sidecar montage 不 KeyError/顺序不匹配
    rec = build_model_record(str(npz), model_id="example30", template_id="t0")
    assert rec["dataset"] == "model"
    # 单事件 toy fixture 可能退化成 descriptive_only；只要不抛即算通过
    assert rec.get("status") == "descriptive_only" or ("channels" in rec and "scalars" in rec)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_propagation_contact_plane_readout.py -k "model_runner" -v`
Expected: FAIL

- [ ] **Step 3: 写 runner**

```python
#!/usr/bin/env python3
"""模型 2D 传播触点平面读出 —— 复用观测层写出的真实格式 NPZ，走同一读出链。
Spec §8: 模型 = "多一个 subject/template record"，逐 template 处理。
Out: results/spatial_modulation/propagation_geometry/observation_readout/model_subjects/
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, json, sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import propagation_skeleton_geometry as G
from src import propagation_contact_plane_readout as R
from scripts.run_contact_plane_readout import build_record_from_events

OUT = _ROOT / "results/spatial_modulation/propagation_geometry/observation_readout/model_subjects"


def build_model_record(npz_path, model_id, template_id="t0"):
    """观测层 NPZ -> 模型 readout record。

    on-disk legacy 键（已验证 example30_lagPat_withFreqCent.npz）：
      lagPatRank / eventsBool / lagPatRaw / chnNames / start_t
    坐标 NOT 在 NPZ —— 在同前缀 sidecar `<record>_montage.json`：
      {contact_coords: [N×2], chn_names: [...]}，顺序须等于 NPZ chnNames（断言）。
    """
    npz_path = Path(npz_path)
    z = np.load(npz_path, allow_pickle=True)
    names = [str(n) for n in z["chnNames"]]
    ranks = np.asarray(z["lagPatRank"], float)
    bools = np.asarray(z["eventsBool"]) > 0
    lag_raw = np.asarray(z["lagPatRaw"], float)
    # sidecar montage：把 `_lagPat_withFreqCent.npz` 换成 `_montage.json`
    stem = npz_path.name.replace("_lagPat_withFreqCent.npz", "")
    mont_f = npz_path.parent / f"{stem}_montage.json"
    if not mont_f.exists():
        raise FileNotFoundError(f"montage sidecar missing: {mont_f}")
    mont = json.loads(mont_f.read_text())
    if [str(n) for n in mont["chn_names"]] != names:
        raise ValueError(f"montage chn_names order != NPZ chnNames for {npz_path}")
    coords2d = np.asarray(mont["contact_coords"], float)        # (N, 2)
    coords = np.column_stack([coords2d, np.zeros(coords2d.shape[0])])  # 2D->3D(z=0)
    mapped = np.ones(len(names), bool)
    rec = build_record_from_events(
        dataset="model", subject=model_id, template_id=template_id, names=names,
        ranks=ranks, bools=bools, lag_raw=np.where(bools, lag_raw, np.nan),
        coords=coords, mapped=mapped, soz_core=set(), montage="single",
        lag_time_unit="ms",          # 模型 sim 内部 ms（lag 归一化已消单位）
        spacing_mm=4.0)              # 虚拟杆 pitch ~4mm
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="观测层 *_lagPat_withFreqCent.npz")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--template-id", default="t0")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rec = build_model_record(args.npz, args.model_id, args.template_id)
    (out / f"{args.model_id}_{args.template_id}.json").write_text(
        json.dumps(rec, indent=2, default=float))
    print(f"wrote model readout: {args.model_id}_{args.template_id}.json "
          f"(n_channels={rec.get('n_channels')})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_propagation_contact_plane_readout.py -k "model_runner" -v`
Expected: PASS（若 fixture 在）

- [ ] **Step 5: Commit**

```bash
git add scripts/run_model_contact_plane_readout.py tests/test_propagation_contact_plane_readout.py
git commit -m "feat(contact-plane): model readout runner via observation-layer NPZ (Task 9)"
```

---

### Task 10: 比较 runner + 敏感性检查

**Files:**
- Create: `scripts/run_real_vs_model_comparison.py`
- Test: `tests/test_propagation_contact_plane_readout.py`

- [ ] **Step 1: 写失败测试**

```python
def test_comparison_runner_end_to_end(tmp_path):
    from scripts.run_real_vs_model_comparison import run_comparison
    # 造 3 个 real record + 1 个 model record 到临时目录
    def mk(d, s, tid, shift):
        chans = [{"x_norm": x, "y_norm": 0.0, "typical_rank": x, "support": 1.0,
                  "uncertainty_rank": 0.0, "signed_transverse_mm": 0.0,
                  "along_axis_mm": x*30} for x in np.linspace(0.1, 0.9, 8)]
        return {"dataset": d, "subject": s, "template_id": tid,
                "axis_length_mm": 30.0+shift,
                "scalars": R.compute_cohort_scalars(
                    {"channels": chans, "axis_length_mm": 30.0+shift}),
                "channels": chans}
    rd = tmp_path / "real_subjects"; rd.mkdir()
    md = tmp_path / "model_subjects"; md.mkdir()
    for i in range(3):
        (rd / f"yuquan_s{i}_t0.json").write_text(json.dumps(mk("yuquan", f"s{i}", "t0", i)))
    (md / "example_t0.json").write_text(json.dumps(mk("model", "example", "t0", 1)))
    summary = run_comparison(rd, md, tmp_path / "comparison")
    assert "scalar_placement" in summary
    assert (tmp_path / "comparison" / "real_vs_model_summary.json").exists()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_propagation_contact_plane_readout.py -k "comparison_runner" -v`
Expected: FAIL

- [ ] **Step 3: 写 runner**

```python
#!/usr/bin/env python3
"""real-vs-model 描述性 posterior-predictive 比较（spec §9）。
Out: results/.../observation_readout/comparison/real_vs_model_summary.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src import propagation_contact_plane_readout as R

BASE = _ROOT / "results/spatial_modulation/propagation_geometry/observation_readout"


def _load_records(d):
    recs = []
    for f in sorted(Path(d).glob("*.json")):
        r = json.loads(f.read_text())
        if r.get("status") in ("no_events", "descriptive_only"):
            continue
        if not r.get("channels"):
            continue
        recs.append(r)
    return recs


def run_comparison(real_dir, model_dir, out_dir, s_thresh=R.S_THRESH,
                   overlap_min=R.OVERLAP_MIN):
    reals = _load_records(real_dir)
    models = _load_records(model_dir)
    X, Y = R.make_plane_grid()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    per_model = {}
    for m in models:
        per_model[f"{m['subject']}_{m['template_id']}"] = R.compare_model_to_cohort(
            m, reals, X, Y, sigma_xy=None, s_thresh=s_thresh, overlap_min=overlap_min)
    summary = {
        "n_real_records": len(reals), "n_model_records": len(models),
        "params": {"s_thresh": s_thresh, "overlap_min": overlap_min,
                   "grid_n": R.GRID_N},
        "scalar_placement": (next(iter(per_model.values()))["scalar_placement"]
                             if per_model else {}),
        "per_model": per_model,
        "note": "descriptive posterior-predictive; no p-value; SOZ not a metric",
    }
    (out_dir / "real_vs_model_summary.json").write_text(
        json.dumps(summary, indent=2, default=float))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", default=str(BASE / "real_subjects"))
    ap.add_argument("--model-dir", default=str(BASE / "model_subjects"))
    ap.add_argument("--out-dir", default=str(BASE / "comparison"))
    ap.add_argument("--sensitivity", action="store_true",
                    help="跑 S_THRESH/OVERLAP_MIN/GRID_N 敏感性，验证 placement 方向不变")
    args = ap.parse_args()
    s = run_comparison(args.real_dir, args.model_dir, args.out_dir)
    print(json.dumps({"n_real": s["n_real_records"], "n_model": s["n_model_records"]},
                     indent=2))
    if args.sensitivity:
        grid = []
        for st in (0.10, 0.15, 0.20):
            for om in (15, 25, 40):
                ss = run_comparison(args.real_dir, args.model_dir,
                                    Path(args.out_dir) / f"sens_{st}_{om}",
                                    s_thresh=st, overlap_min=om)
                fp = next(iter(ss["per_model"].values()), {}).get("field_placement", {})
                grid.append({"s_thresh": st, "overlap_min": om,
                             "model_to_real_median_corr":
                                 fp.get("model_to_real_median_corr")})
        (Path(args.out_dir) / "sensitivity.json").write_text(
            json.dumps(grid, indent=2, default=float))
        print("sensitivity grid written")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_propagation_contact_plane_readout.py -k "comparison_runner" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/run_real_vs_model_comparison.py tests/test_propagation_contact_plane_readout.py
git commit -m "feat(contact-plane): real-vs-model comparison runner + sensitivity (Task 10)"
```

---

### Task 11: 静态图 + figures/README.md（中文）

**Files:**
- Create: `scripts/plot_contact_plane_static.py`
- Create: `results/spatial_modulation/propagation_geometry/observation_readout/figures/README.md`（图生成后写）

- [ ] **Step 1: 写绘图脚本**

每个 subject/template 一张静态图，4 panel（守 CLAUDE.md §7 一面板一问题）：
1. 触点散点（x_norm, y_norm；颜色=typical_rank；大小=support）+ SOZ overlay 标记（`is_soz` 的触点描黑边；图注 "SOZ overlay only, not metric input"）
2. 连续 rank 场 T（support-gated，`mask=False` 处灰掉）
3. 支撑场 S（看哪里有采样支撑）
4. 不确定度场 U（早端这团的离散度——守口径锁，标注"早端=轮流当早端的小入口群，非固定点"）

```python
#!/usr/bin/env python3
"""2D 触点平面读出静态图（per subject/template）。
口径锁：早端画成带不确定度的小入口群；禁 "固定稳定早端 / 弹性入口区"。
SOZ：仅描述性叠加，图注固定 "SOZ overlay only, not metric input"。
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src import propagation_contact_plane_readout as R

BASE = _ROOT / "results/spatial_modulation/propagation_geometry/observation_readout"


def plot_record(rec, out_png):
    X, Y = R.make_plane_grid()
    fldT = R.smooth_field(rec, X, Y, sigma_xy=None, scalar="rank")
    fldU = {"U": fldT["U"], "mask": fldT["mask"]}
    chans = rec["channels"]
    xs = [c["x_norm"] for c in chans]; ys = [c["y_norm"] for c in chans]
    rk = [c["typical_rank"] for c in chans]; sp = [c["support"] for c in chans]
    soz = [c.get("is_soz", False) for c in chans]
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    sc = ax[0].scatter(xs, ys, c=rk, s=[40 + 200*s for s in sp], cmap="viridis",
                       vmin=0, vmax=1, edgecolors=["k" if z else "none" for z in soz],
                       linewidths=1.5)
    ax[0].set_title("contacts: color=order, size=support\n(black ring = SOZ overlay)")
    ax[0].set_xlabel("along-axis (norm)"); ax[0].set_ylabel("transverse (norm, display only)")
    plt.colorbar(sc, ax=ax[0], label="typical order (0=early,1=late)")
    for a, key, ttl, cm in [(ax[1], "T", "smoothed order field", "viridis"),
                            (ax[2], "S", "support field", "magma"),
                            (ax[3], "U", "uncertainty field\n(early = small taking-turns group)",
                             "cividis")]:
        F = fldT[key].copy()
        if key != "S":
            F = np.where(fldT["mask"], F, np.nan)
        im = a.imshow(F, origin="lower", extent=[R.X_LO, R.X_HI, -R.Y_EXT, R.Y_EXT],
                      aspect="auto", cmap=cm)
        a.set_title(ttl); plt.colorbar(im, ax=a)
    amb = rec.get("soz_ambiguous", [])
    fig.suptitle(f"{rec['dataset']}:{rec['subject']} {rec['template_id']} | "
                 f"SOZ overlay only, not metric input"
                 + (f" | SOZ ambiguous: {amb}" if amb else ""))
    fig.tight_layout()
    fig.savefig(out_png, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", default=str(BASE / "real_subjects"))
    ap.add_argument("--out", default=str(BASE / "figures/static_maps"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(args.real_dir).glob("*.json")):
        rec = json.loads(f.read_text())
        if not rec.get("channels"):
            continue
        plot_record(rec, out / f"{f.stem}.png")
        print(f"  {f.stem}.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 生成图（若有真实 record）**

Run: `python scripts/plot_contact_plane_static.py`
Expected: `figures/static_maps/<ds>_<subj>_<tid>.png`

- [ ] **Step 3: 目视检查 + 写 figures/README.md（中文，图生成后）**

按 AGENTS.md 格式（`### filename` + 2–4 句 + `**关注点**：`）。必须包含口径锁说明：早端是小入口群、SOZ 仅叠加不进指标。

- [ ] **Step 4: Commit**

```bash
git add scripts/plot_contact_plane_static.py "results/spatial_modulation/propagation_geometry/observation_readout/figures/README.md"
git commit -m "feat(contact-plane): static 4-panel readout figure + README (Task 11)"
```

---

### Task 12: 全量跑 + archive doc

**Files:**
- Create: `docs/archive/topic3/propagation_contact_plane_readout_2026-06-11.md`（或 topic4，按落位）

- [ ] **Step 1: 全 cohort 跑**

```bash
python scripts/run_contact_plane_readout.py
python scripts/run_model_contact_plane_readout.py --npz <model_npz> --model-id <id>
python scripts/run_real_vs_model_comparison.py --sensitivity
python scripts/plot_contact_plane_static.py
```

- [ ] **Step 2: 写 archive doc**

三段式（CLAUDE.md §8）：测了什么 / 怎么测的 / 揭示了什么 + flags 分布 + 敏感性结果 + 口径锁复述。主文档只放摘要 + 链接。

- [ ] **Step 3: 全测试回归**

Run: `pytest tests/test_propagation_contact_plane_readout.py -v`
Expected: 全 PASS

- [ ] **Step 4: Commit**

```bash
git add docs/archive/topic3/propagation_contact_plane_readout_2026-06-11.md docs/topic3_spatial_soz_modulation.md
git commit -m "docs(contact-plane): cohort run results + archive (Task 12)"
```

---

## Self-Review（writing-plans 自检）

**Spec coverage：**
- §3 双坐标系 → Task 3（x_norm/y_norm）+ Task 7（mm scalars）✓
- §4 signed-transverse + P0 镜像 → Task 1 + Task 5 ✓
- §5 rank 主/time 副 + 事件内归一化 → Task 2 ✓
- §6 连续场 T/S/U + gating → Task 4 ✓
- §7 SOZ first-contact alias → Task 7 ✓
- §8 模型复用观测层逐模板 → Task 9 ✓
- §9 描述性 + subject-first + 标量分类 → Task 6 + Task 7 + Task 10 ✓
- §10 flags + 口径锁 → Task 3（flags）+ Task 11（图注口径锁）✓
- §11 输出目录 → Task 8–11 ✓
- §12 TDD 9+2 条 → Task 1–7 测试覆盖：镜像不变(T5)/低support(T4)/rank归一化(T2)/SOZ mismatch(T7)/1D+planarity(T3)/单位不变(T2)/多模板(T6) ✓
- §13 执行顺序（动画 deferred）→ 本 plan 不含 animation ✓

**Placeholder scan：** 无 TBD/TODO；runner 代码完整；数值门集中在顶部常量 + 敏感性 Task 10。

**Type consistency：** record schema（`channels`/`flags`/`scalars`/`axis_length_mm`/`lag_time_unit`）跨 Task 3/7/8/9/10 一致；`smooth_field` 返回 `{T,S,U,mask,sigma_xy}` 跨 Task 4/5/6/11 一致；`corr_pair_mirror_invariant` 返回 `{corr,n_overlap,insufficient_overlap}` 跨 Task 5/6 一致。

**已解（reviewer 第二轮，已在 plan 内核实并修掉）：**
- 观测层 NPZ 是 **legacy 键** `lagPatRank/eventsBool/lagPatRaw/chnNames/start_t`（已用 `example30_lagPat_withFreqCent.npz` 实测确认），坐标在 sidecar `<record>_montage.json`（`contact_coords [N×2]` + `chn_names`）—— Task 9 已按此重写 + 顺序断言。
- `load_subject_propagation_events` **确带 `lag_raw` 键**（src/interictal_propagation.py:406）—— Task 8 已改 hard-assert（缺则 raise，不 silently 用 rank 伪造 time 副图）。
- Task 1 二维 PCA 测试样例已修成真正二维残差（原样例共线，会让正确实现也红）。
- Task 8 `one_dimensional_sampling` 判定已移到轴框架之后用真实 `fr['off_axis']`（原用全零 off-axis 会系统性误标 1D）。
- Task 6 field comparison 已改真正 subject-first（real-to-real 跳过同 `(dataset,subject)`，model-to-real 先按 real subject 折叠）。
- `t_dom/t_min` → `t_a/t_b`（rank-displacement pair 两支，不预设 dominant/minority 科学事实）。

**仍需执行时留意（非阻断）：**
- `classify_sampling_geometry` 的返回键（`geometry` / `measurable`）以模块实际为准，执行 Task 8 第一个真实 subject 时打印一次确认。
- 模型 swap/正反双模板仍是第二批（spec §0）；本 plan 模型侧只逐 template 出 record，不依赖双模板自动聚类。
