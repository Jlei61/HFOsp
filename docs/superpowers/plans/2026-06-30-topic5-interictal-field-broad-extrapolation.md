# Topic 5 间期顺序场 → 发作招募【顺序】外推 — Implementation Plan【CLOSED-NEGATIVE】

> **状态（2026-07-01 restructure）：CLOSED-NEGATIVE。** 本 plan 只装**一个**问题=
> "间期传播顺序场能否预测发作**招募顺序**(z-ER `r_sz`)"。**结果阴性**——发作招募顺序跨发作本就不稳，
> 无可预测的稳定方向（见 `docs/archive/topic5/field_extrapolation_pilot_2026-06-30.md` 第一版段）。
> **能量问题（间期顺序场 → 发作早期能量空间分布）是另一个问题，另开 spec/plan**：
> `docs/superpowers/specs/2026-07-01-topic5-energy-field-extrapolation-design.md` +
> `docs/superpowers/plans/2026-07-01-topic5-energy-field-extrapolation.md`。本 plan 不再扩展，保留作 order-问题的归档记录。

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development 或 executing-plans，逐 task 实现。Steps 用 checkbox 跟踪。

**Goal（CLOSED）：** 检验"用间期传播顺序场，能否预测间期隐身电极（broad∖narrow）在发作时的**招募顺序**"。结果阴性（发作顺序不稳）。

**Architecture:** 复用既有场引擎（`src/propagation_contact_plane_readout.py` 的 support 加权 kernel regression + 带符号场相关）+ 既有 broad 间期轴场（`propagation_geometry_broad`，已含 x_norm/y_norm/typical_rank/support）+ Layer A 发作 z-ER 招募序（`r_sz`）。新建一个薄模块做"场在隐身电极位置的留一预测 + F/C 对照 + null + 半径基线"，pilot runner 跑 1–2 个 Epilepsiae 被试。

**Tech Stack:** Python, numpy, scipy.stats（spearmanr），既有 HFOsp 模块。

**Spec:** `docs/superpowers/specs/2026-06-30-topic5-interictal-field-broad-extrapolation-design.md`

## Global Constraints

- 发作侧 = z-ER 招募顺序（Layer A `per_er.broad_ER.r_sz`，低=早=源），**非** `bb_auc` 激活强度。（用户 2026-06-30 锁）
- 统计量带符号；同向 = 间期早↔发作早 → 正相关。
- 间期 broad rank 是 phantom-masked 的（broad 跑全程 masked）→ 直接用 `propagation_geometry_broad` 的 `typical_rank`，不再二次 mask。
- 坐标单位 mm；broad record 已自带投影坐标，pilot 不另调 coord loader。
- 不主张"发作早期特异"（只说方向延伸进隐身 territory）。
- broad∖narrow = 两 per_subject json `channel_names` 的**精确字符串差集**。
- 产物目录 `results/topic5_ictal_recruitment/field_extrapolation/`（新建，含 `figures/README.md`）。
- **不提交**（当前在 topic4 分支；新文件全 untracked，待用户定干净 base）。

---

## 关键工件路径（已核实）

- 间期 broad 轴场：`results/spatial_modulation/propagation_geometry_broad/observation_readout/real_subjects/{ds}_{sid}_t_a.json`
  - top: `dataset, subject, axis_length_mm, norm_scale_mm, channels[], n_channels, ...`
  - `channels[i]`: `name, shaft, along_axis_mm, signed_transverse_mm, x_norm, y_norm, typical_rank, support, coord_mm, is_soz, ...`
- narrow 池通道：`results/interictal_propagation_masked/per_subject/{ds}_{sid}.json::channel_names`
- broad 池通道：`results/interictal_propagation_masked_broad/per_subject/{ds}_{sid}.json::channel_names`
- 发作 z-ER 序：`results/data_driven_soz/layer_a_ictal_er_rank/per_subject/{ds}_{sid}.json`
  - `per_er.broad_ER.r_sz[ch]`（median fractional rank，低=早），`r_sz_valid_count[ch]`，`seizure_records[]`（含 `channel_onsets[ch]={frame_idx,t_onset_sec}`、`status`）

pilot 候选（broad ∩ Layer A onset，Epilepsiae）：583, 590, 1077, 1096, 1146, 1150（先 583 + 1077）。

---

## File Structure

- Create `src/topic5_field_extrapolation.py` — 纯函数模块（数据加载 + 场预测 + F/C + null + 半径基线）。
- Create `tests/test_topic5_field_extrapolation.py` — TDD。
- Create `scripts/run_topic5_field_extrapolation_pilot.py` — pilot runner（per-subject JSON + 控制台摘要）。
- Create `scripts/plot_topic5_field_extrapolation.py` — 诊断图。
- Create `results/topic5_ictal_recruitment/field_extrapolation/figures/README.md`（图生成后写）。

---

### Task 1: 数据加载 + broad∖narrow + 发作 z-ER 序

**Files:**
- Create: `src/topic5_field_extrapolation.py`
- Test: `tests/test_topic5_field_extrapolation.py`

**Interfaces — Produces:**
- `load_broad_axis_record(ds_sid, axis_dir) -> dict`（直接 json.load）
- `channel_names_from_pool(ds_sid, pool_dir) -> list[str]`
- `broad_minus_narrow(broad_names, narrow_names) -> list[str]`（精确字符串差集，排序）
- `ictal_zer_ranks(ds_sid, layer_a_dir, er_config="broad_ER", min_valid_count=3) -> dict[str,float]`（读 r_sz，按 r_sz_valid_count≥min 过滤；低=早）

- [ ] **Step 1: 写失败测试**

```python
import numpy as np, json, math
from src.topic5_field_extrapolation import (
    broad_minus_narrow, ictal_zer_ranks, channel_names_from_pool, load_broad_axis_record,
)

def test_broad_minus_narrow_exact_string():
    broad = ["TLA1","TLA2","TLB2","TBA2"]; narrow = ["TLA2","TBA2"]
    assert broad_minus_narrow(broad, narrow) == ["TLA1","TLB2"]

def test_broad_minus_narrow_no_basestrip():
    # 不做 base-stripping：'TLA1' 与 'TLA10' 不同
    assert broad_minus_narrow(["TLA1","TLA10"], ["TLA1"]) == ["TLA10"]

def test_ictal_zer_ranks_filters_low_valid_count(tmp_path):
    sid = "epilepsiae_TEST"
    d = {"per_er": {"broad_ER": {
        "r_sz": {"A1": 0.1, "A2": 0.9, "A3": 0.5},
        "r_sz_valid_count": {"A1": 5, "A2": 1, "A3": 4}}}}
    p = tmp_path / f"{sid}.json"; p.write_text(json.dumps(d))
    out = ictal_zer_ranks(sid, str(tmp_path), min_valid_count=3)
    assert set(out) == {"A1","A3"}          # A2 dropped (valid_count 1<3)
    assert out["A1"] == 0.1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_topic5_field_extrapolation.py -k "broad_minus or ictal_zer" -v`
Expected: FAIL（ImportError / not defined）

- [ ] **Step 3: 实现**

```python
"""Topic 5 间期传播场外推到发作隐身 territory。

发作侧 = z-ER 招募顺序 (Layer A r_sz，低=早=源)；间期侧 = broad 轴场 typical_rank (低=早)。
同向 → 正相关。详见 docs/superpowers/specs/2026-06-30-topic5-interictal-field-broad-extrapolation-design.md
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Sequence, Optional
import numpy as np

DEF_AXIS_DIR = "results/spatial_modulation/propagation_geometry_broad/observation_readout/real_subjects"
DEF_BROAD_POOL = "results/interictal_propagation_masked_broad/per_subject"
DEF_NARROW_POOL = "results/interictal_propagation_masked/per_subject"
DEF_LAYER_A = "results/data_driven_soz/layer_a_ictal_er_rank/per_subject"


def load_broad_axis_record(ds_sid: str, axis_dir: str = DEF_AXIS_DIR) -> dict:
    return json.load(open(Path(axis_dir) / f"{ds_sid}_t_a.json"))


def channel_names_from_pool(ds_sid: str, pool_dir: str) -> List[str]:
    d = json.load(open(Path(pool_dir) / f"{ds_sid}.json"))
    return list(d["channel_names"])


def broad_minus_narrow(broad_names: Sequence[str], narrow_names: Sequence[str]) -> List[str]:
    nset = set(narrow_names)
    return sorted(n for n in set(broad_names) if n not in nset)


def ictal_zer_ranks(ds_sid: str, layer_a_dir: str = DEF_LAYER_A,
                    er_config: str = "broad_ER", min_valid_count: int = 3) -> Dict[str, float]:
    d = json.load(open(Path(layer_a_dir) / f"{ds_sid}.json"))
    er = d["per_er"][er_config]
    r_sz = er["r_sz"]; vc = er.get("r_sz_valid_count", {})
    out = {}
    for ch, r in r_sz.items():
        if r is None:
            continue
        if int(vc.get(ch, 0)) < min_valid_count:
            continue
        out[ch] = float(r)
    return out
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_topic5_field_extrapolation.py -k "broad_minus or ictal_zer" -v`
Expected: PASS

---

### Task 2: 场在隐身电极位置的留一预测

**Files:**
- Modify: `src/topic5_field_extrapolation.py`
- Test: `tests/test_topic5_field_extrapolation.py`

**Interfaces — Produces:**
- `field_predict_at_points(record_channels, eval_xy, exclude_name=None, sigma_xy=None) -> np.ndarray`
  - record_channels: list of dict（含 x_norm,y_norm,typical_rank,support,name）
  - eval_xy: (m,2) 待评估位置；返回 (m,) support 加权 kernel 回归的 typical_rank 预测；NaN 若该点总权重≈0。
- `predicted_interictal_order(record, target_names, loo=True, sigma_xy=None) -> dict[str,float]`
  - 对每个 target（隐身电极），用 record 全 broad 通道（loo=True 时排除该 target 本身）做场预测，评估在 target 自己的 (x_norm,y_norm)。

设计要点：support 加权使低发放隐身电极对场贡献小 → 场值由可信核心主导（= "核心补充噪电极"）；loo 排除 target 自身 rank → F 是干净样本外预测。

- [ ] **Step 1: 写失败测试**

```python
from src.topic5_field_extrapolation import field_predict_at_points, predicted_interictal_order

def _chan(name, x, y, rank, support):
    return {"name": name, "x_norm": x, "y_norm": y, "typical_rank": rank, "support": support}

def test_field_predict_matches_nearby_high_support():
    # 高 support 核心点在原点 rank=0.2；远处低 support 点 rank=0.9；
    # 评估在靠近核心处应≈0.2
    chans = [_chan("C1",0,0,0.2,1.0), _chan("C2",1.0,0,0.9,0.05)]
    pred = field_predict_at_points(chans, np.array([[0.05,0.0]]), sigma_xy=0.3)
    assert abs(pred[0] - 0.2) < 0.1

def test_loo_excludes_self():
    # 待测点自己 rank=0.99、support 高，但 loo 排除自己后由邻居 0.2 决定
    chans = [_chan("C1",0,0,0.2,1.0), _chan("SELF",0.05,0,0.99,1.0)]
    rec = {"channels": chans}
    out = predicted_interictal_order(rec, ["SELF"], loo=True, sigma_xy=0.3)
    assert abs(out["SELF"] - 0.2) < 0.15      # 被邻居拉住，不是 0.99
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_topic5_field_extrapolation.py -k "field_predict or loo_excludes" -v`
Expected: FAIL

- [ ] **Step 3: 实现**

```python
def _median_nn(pts: np.ndarray) -> float:
    if pts.shape[0] < 2:
        return 0.1
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(1)))


def field_predict_at_points(record_channels, eval_xy, exclude_name: Optional[str] = None,
                            sigma_xy: Optional[float] = None) -> np.ndarray:
    chans = [c for c in record_channels
             if c["name"] != exclude_name
             and np.isfinite(c["x_norm"]) and np.isfinite(c["y_norm"])
             and np.isfinite(c.get("typical_rank", np.nan)) and c.get("support", 0) > 0]
    pts = np.array([[c["x_norm"], c["y_norm"]] for c in chans], float).reshape(-1, 2)
    vals = np.array([c["typical_rank"] for c in chans], float)
    sup = np.array([c["support"] for c in chans], float)
    if sigma_xy is None:
        sigma_xy = _median_nn(pts)
    sig2 = 2.0 * sigma_xy ** 2
    eval_xy = np.asarray(eval_xy, float).reshape(-1, 2)
    out = np.full(eval_xy.shape[0], np.nan)
    for i, (x, y) in enumerate(eval_xy):
        w = sup * np.exp(-(((pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2) / sig2))
        sw = w.sum()
        if sw > 1e-12:
            out[i] = float((w * vals).sum() / sw)
    return out


def predicted_interictal_order(record, target_names, loo: bool = True,
                               sigma_xy: Optional[float] = None) -> Dict[str, float]:
    chans = record["channels"]
    by_name = {c["name"]: c for c in chans}
    out = {}
    for nm in target_names:
        if nm not in by_name:
            continue
        c = by_name[nm]
        xy = np.array([[c["x_norm"], c["y_norm"]]], float)
        pred = field_predict_at_points(chans, xy,
                                       exclude_name=nm if loo else None, sigma_xy=sigma_xy)
        out[nm] = float(pred[0])
    return out
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_topic5_field_extrapolation.py -k "field_predict or loo_excludes" -v`
Expected: PASS

---

### Task 3: F / C 统计量 + null + 半径基线

**Files:**
- Modify: `src/topic5_field_extrapolation.py`
- Test: `tests/test_topic5_field_extrapolation.py`

**Interfaces — Produces:**
- `signed_spearman(x, y) -> float`（scipy.stats.spearmanr；<2 或退化 → nan）
- `compute_f_c(record, hidden_names, ictal_ranks, loo=True, sigma_xy=None) -> dict`
  - 对齐 hidden∩has(ictal_rank)∩has(broad typical_rank)；
  - F = signed_spearman(predicted_interictal_order, ictal_rank)
  - C = signed_spearman(own broad typical_rank, ictal_rank)
  - 返回 `{F, C, n_hidden, names, predicted, own_rank, ictal}`
- `null_F(record, hidden_names, ictal_ranks, n=2000, seed=0, **kw) -> dict`
  - 打乱 hidden 上的 ictal_rank → F_null 分布；返回 `{p_value(单尾正), p95, null_median}`
- `radial_baseline_corr(record, hidden_names, ictal_ranks) -> float`
  - 用每个 hidden 电极沿轴位置到间期源（along_axis_mm 最小=源）的距离作预测 → 对 ictal_rank 的 signed_spearman（"近的先亮"基线）

- [ ] **Step 1: 写失败测试**

```python
from src.topic5_field_extrapolation import compute_f_c, null_F, signed_spearman

def _rec_line():
    # 5 个核心点沿 x 轴 rank 单调，2 个隐身点在中间，support 低
    ch = [_chan(f"K{i}", i*0.2, 0.0, i*0.2, 1.0) for i in range(5)]
    ch += [_chan("H1", 0.35, 0.05, 0.99, 0.05), _chan("H2", 0.75, 0.05, 0.01, 0.05)]
    return {"channels": ch}

def test_F_uses_field_not_self():
    rec = _rec_line()
    # 隐身电极自己的 rank 是乱的(0.99/0.01)，但发作真顺序跟"位置"一致 → 场预测应对
    ictal = {"H1": 0.30, "H2": 0.70}   # H1 早 H2 晚，与 x 位置一致
    out = compute_f_c(rec, ["H1","H2"], ictal, loo=True, sigma_xy=0.3)
    assert out["n_hidden"] == 2
    assert out["F"] > out["C"]         # 场赢逐通道(自身 rank 是反的)

def test_null_returns_p_and_p95():
    rec = _rec_line()
    ictal = {"H1": 0.30, "H2": 0.70}
    nd = null_F(rec, ["H1","H2"], ictal, n=200, seed=1, sigma_xy=0.3)
    assert "p_value" in nd and "p95" in nd
    assert 0.0 <= nd["p_value"] <= 1.0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_topic5_field_extrapolation.py -k "F_uses_field or null_returns" -v`
Expected: FAIL

- [ ] **Step 3: 实现**

```python
from scipy.stats import spearmanr

def signed_spearman(x, y) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.std(x[m]) < 1e-12 or np.std(y[m]) < 1e-12:
        return float("nan")
    return float(spearmanr(x[m], y[m]).correlation)


def _align(record, hidden_names, ictal_ranks):
    by_name = {c["name"]: c for c in record["channels"]}
    names = [n for n in hidden_names if n in by_name and n in ictal_ranks
             and np.isfinite(by_name[n].get("typical_rank", np.nan))]
    return names, by_name


def compute_f_c(record, hidden_names, ictal_ranks, loo: bool = True, sigma_xy=None) -> dict:
    names, by_name = _align(record, hidden_names, ictal_ranks)
    pred = predicted_interictal_order(record, names, loo=loo, sigma_xy=sigma_xy)
    names = [n for n in names if np.isfinite(pred.get(n, np.nan))]
    p = np.array([pred[n] for n in names])
    own = np.array([by_name[n]["typical_rank"] for n in names])
    ict = np.array([ictal_ranks[n] for n in names])
    return {"F": signed_spearman(p, ict), "C": signed_spearman(own, ict),
            "n_hidden": len(names), "names": names,
            "predicted": p.tolist(), "own_rank": own.tolist(), "ictal": ict.tolist()}


def null_F(record, hidden_names, ictal_ranks, n: int = 2000, seed: int = 0,
           loo: bool = True, sigma_xy=None) -> dict:
    base = compute_f_c(record, hidden_names, ictal_ranks, loo=loo, sigma_xy=sigma_xy)
    names = base["names"]; F_obs = base["F"]
    p = np.array(base["predicted"]); ict = np.array(base["ictal"])
    if len(names) < 3 or not np.isfinite(F_obs):
        return {"F_obs": F_obs, "p_value": float("nan"), "p95": float("nan"),
                "null_median": float("nan"), "n_hidden": len(names)}
    rng = np.random.default_rng(seed)
    null = np.array([signed_spearman(p, rng.permutation(ict)) for _ in range(n)])
    null = null[np.isfinite(null)]
    p_value = float((1 + (null >= F_obs).sum()) / (1 + null.size))
    return {"F_obs": F_obs, "p_value": p_value, "p95": float(np.percentile(null, 95)),
            "null_median": float(np.median(null)), "n_hidden": len(names)}


def radial_baseline_corr(record, hidden_names, ictal_ranks) -> float:
    names, by_name = _align(record, hidden_names, ictal_ranks)
    # 间期源 = along_axis_mm 最小的核心触点；隐身电极沿轴位置作"离源距离"代理
    src = min((c["along_axis_mm"] for c in record["channels"]
               if np.isfinite(c.get("along_axis_mm", np.nan))), default=0.0)
    dist = np.array([by_name[n]["along_axis_mm"] - src for n in names])
    ict = np.array([ictal_ranks[n] for n in names])
    return signed_spearman(dist, ict)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_topic5_field_extrapolation.py -v`
Expected: 全 PASS

---

### Task 4: Pilot runner + 结果表判读

**Files:**
- Create: `scripts/run_topic5_field_extrapolation_pilot.py`

**Interfaces — Consumes:** Task1-3 全部。

逐被试：load broad record → broad/narrow 池名 → broad∖narrow → ictal r_sz → compute_f_c + null_F + radial_baseline → 按四格表分类 → 写
`results/topic5_ictal_recruitment/field_extrapolation/per_subject/{ds_sid}.json` + 控制台摘要。

四格判读（pilot 描述性）：
- F 过 null（p_value<0.05 且 F>0）且 F>radial 且 F>C → `field_wins`（主结果）
- F 过 + C 也过（C>0 且 C>radial）→ `channel_already_enough`（pivot）
- F 不过 + C 不过 → `science_negative`（发作另走他路）
- F 不过 + C 过 → `field_method_misspecified`

- [ ] **Step 1: 写 runner**

```python
import argparse, json
from pathlib import Path
from src.topic5_field_extrapolation import (
    load_broad_axis_record, channel_names_from_pool, broad_minus_narrow,
    ictal_zer_ranks, compute_f_c, null_F, radial_baseline_corr,
    DEF_BROAD_POOL, DEF_NARROW_POOL,
)
OUT = Path("results/topic5_ictal_recruitment/field_extrapolation")

def classify(F, C, p, radial, *, a=0.05):
    f_ok = (p is not None and p < a) and (F is not None and F > 0) and (radial is None or F > radial)
    c_ok = (C is not None and C > 0) and (radial is None or C > radial)
    if f_ok and not c_ok: return "field_wins"
    if f_ok and c_ok:     return "channel_already_enough"
    if (not f_ok) and (not c_ok): return "science_negative"
    return "field_method_misspecified"

def run_subject(ds_sid, *, min_valid=3, n_null=2000, sigma_xy=None):
    rec = load_broad_axis_record(ds_sid)
    broad = channel_names_from_pool(ds_sid, DEF_BROAD_POOL)
    narrow = channel_names_from_pool(ds_sid, DEF_NARROW_POOL)
    hidden = broad_minus_narrow(broad, narrow)
    ictal = ictal_zer_ranks(ds_sid, min_valid_count=min_valid)
    fc = compute_f_c(rec, hidden, ictal, loo=True, sigma_xy=sigma_xy)
    nd = null_F(rec, hidden, ictal, n=n_null, sigma_xy=sigma_xy)
    radial = radial_baseline_corr(rec, hidden, ictal)
    verdict = classify(fc["F"], fc["C"], nd["p_value"], radial)
    return {"subject": ds_sid, "n_broad": len(broad), "n_narrow": len(narrow),
            "n_hidden_total": len(hidden), "n_hidden_eval": fc["n_hidden"],
            "F": fc["F"], "C": fc["C"], "F_p_value": nd["p_value"],
            "F_null_p95": nd["p95"], "radial_baseline": radial,
            "verdict": verdict, "detail": fc}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subjects", nargs="+", help="e.g. epilepsiae_583 epilepsiae_1077")
    ap.add_argument("--n-null", type=int, default=2000)
    ap.add_argument("--min-valid", type=int, default=3)
    ap.add_argument("--sigma-xy", type=float, default=None)
    args = ap.parse_args()
    (OUT / "per_subject").mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in args.subjects:
        r = run_subject(sid, min_valid=args.min_valid, n_null=args.n_null, sigma_xy=args.sigma_xy)
        json.dump(r, open(OUT / "per_subject" / f"{sid}.json", "w"), indent=2)
        rows.append(r)
        print(f"{sid}: n_hidden_eval={r['n_hidden_eval']:>2} | F={r['F']!s:>7} "
              f"C={r['C']!s:>7} | F_p={r['F_p_value']!s:>6} radial={r['radial_baseline']!s:>7} "
              f"| {r['verdict']}")
    json.dump(rows, open(OUT / "pilot_summary.json", "w"), indent=2)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑 pilot（583 先单跑 smoke）**

Run: `python -m scripts.run_topic5_field_extrapolation_pilot epilepsiae_583 --n-null 500`
Expected: 打印一行，n_hidden_eval>0，verdict ∈ 四类之一；写出 per_subject/epilepsiae_583.json。

- [ ] **Step 3: 跑 pilot（583 + 1077，全 null）**

Run: `python -m scripts.run_topic5_field_extrapolation_pilot epilepsiae_583 epilepsiae_1077`
Expected: 两行摘要 + pilot_summary.json。

---

### Task 5: 诊断图 + README

**Files:**
- Create: `scripts/plot_topic5_field_extrapolation.py`
- Create: `results/topic5_ictal_recruitment/field_extrapolation/figures/README.md`

图（per subject 一张，2 panel）：
- Panel A：归一化平面上间期顺序场热图（用 `propagation_contact_plane_readout.make_plane_grid`+`smooth_field`），叠核心触点（圈）+ 隐身电极（方块，按 predicted vs actual ictal 双色边）。
- Panel B：散点 —— x=predicted interictal order（场）/ own broad rank（逐通道），y=ictal z-ER rank；两组点 + 各自 signed corr（F、C）标题；半径基线虚线参考。一眼看 F 是否比 C 更贴对角。

- [ ] **Step 1: 写 plot**（复用既有 make_plane_grid/smooth_field；matplotlib）
- [ ] **Step 2: 跑图** Run: `python -m scripts.plot_topic5_field_extrapolation epilepsiae_583 epilepsiae_1077`
- [ ] **Step 3: 写 figures/README.md**（中文逐图，末尾"**关注点**："，图生成后写）

---

## 能量问题 → 已迁出本 plan

间期顺序场 → 发作早期**能量**空间分布（bb_auc/hfa_auc、C1/C2 基线、cohort null 阶梯）是**另一个问题**，
全部内容迁到新 plan：`docs/superpowers/plans/2026-07-01-topic5-energy-field-extrapolation.md`
（spec `docs/superpowers/specs/2026-07-01-topic5-energy-field-extrapolation-design.md`）。
本 plan 到此为止（order-问题，closed-negative）。

## Self-Review 注记

- Spec 覆盖：F/C（Task3）、broad∖narrow（Task1）、z-ER 发作序（Task1）、留一场预测（Task2）、null+半径基线（Task3）、四格判读（Task4）、图（Task5）。杆向基线/时间对照/cohort → Phase 2（pilot 先验证主信号）。
- 符号约定：间期 typical_rank 低=早，ictal r_sz 低=早 → 同向=正相关，F>0 为对齐（贯穿 Task3/4）。
- 类型一致：`compute_f_c` 返回键（F/C/n_hidden/names/predicted/own_rank/ictal）在 null_F、runner 一致复用。
