# Topic 5 发作内 field 动力学 pilot — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 对 6 个代表性 ECoG subject，逐次发作计算 field 指标随发作进程（onset→offset）与终止前后的变化（轴向中段相对活动 / 非轴向 / 同步 / 场方向漂移 / 场-轴对齐），纯描述性 pilot。

**Architecture:** 三段管线 — (1) 新建长窗 ictal robust-z cache（复用现有 `extract_seizure_window`+`_features_one`，每发作抽到 offset+90s）；(2) 纯数学模块 `src/topic5_ictal_field_dynamics.py`（source-core compact 定位、轴四分区、各指标）+ TDD；(3) 驱动脚本扫窗写 CSV/JSON，绘图脚本两层出图。统计口径全部复用 `run_topic5_axis_alignment` 的 field 构造与 `corr_pair_mirror_invariant`。

**Tech Stack:** Python 3 / numpy / scipy / matplotlib（Agg）；pytest。复用 `src.ictal_onset_extraction`、`src.topic5_ictal_recruitment`、`src.propagation_contact_plane_readout`、`src.topic5_axis_alignment`、`scripts.build_topic5_t0_feature_cache`、`scripts.run_topic5_t0_eligibility`、`scripts.plot_contact_plane_static`、`scripts.plot_topic5_swap_nodes_fields`。

设计依据 spec: `docs/superpowers/specs/2026-06-28-topic5-ictal-field-dynamics-design.md`（已含用户 P0/P1 review 修复）。

## Global Constraints

每个 task 隐含遵守（spec §7 锁定参数，verbatim）：

- substrate = **narrow**；subjects = `epilepsiae_{442,548,583,384,958,1084}`。
- eligibility = `analysis_eligible==True` ∧ `has_complete_eeg_interval==True`；offset 口径 = **eeg**（`eeg_offset_epoch`）。
- baseline = 现有自适应 `[-pre,-60]`(pre≥120)，per-ch median/MAD*1.4826（**复用 `_features_one`，不重算**）。
- band = bb 1–45 Hz primary；hfa 60–100 Hz 存但 secondary。
- cache 窗 = `pre_sec=130`，`post_sec=ceil(eeg_duration_sec)+90`（每发作）。
- source core = 每侧 top2 if maxdist<**15 mm** 否则 top1（单点）；`decision_k` 仅 provenance。
- 分区 = 4 组 `source_core / axis_end_noncore / axial_mid / non_axial`；on-axis 阈 `d ≤ median(d over 非 source_core mapped)`；mid-zone `t∈[0.25,0.75]`。
- 退化轴 = `L < 0.15 * bbox_diag` → `axis_degenerate`。
- 组占比 = `positive_mass_share = Σmax(z,0)_g / Σmax(z,0)_all`（**不用 mean/mean 比值**）。
- 滑窗 = 长 10s/step 5s 到 offset；终止窗 = `[-60,-30],[-30,-10],[-10,0],[0,30]s` rel offset，左缘<onset → `pre_onset_overlap`（排除 summary）。
- alignment = maxAB `|corr_pair_mirror_invariant|`，`S_THRESH=0.15`，`OVERLAP_MIN=25`；活动向量传 raw window-mean z（与 `bb_auc` 同路径，由 `R_smooth_rank` 内部处理，**不预先 rank**）。
- baseline parity = 长 cache `bb_auc[0,10]` vs `t0_feature_cache_v2_windows` 同发作，`max|Δ|<1e-3` 否则 `parity_fail` 剔除。
- per-seizure 详图门槛 = `eeg_duration≥40s` 且非 `parity_fail`。
- 输出根 = `results/topic5_ictal_recruitment/{ictal_field_long_cache, field_dynamics}/`；NOT 触碰现有 `t0_feature_cache*`。

---

## File Structure

- **Create** `scripts/build_topic5_ictal_field_long_cache.py` — 长窗 cache builder（每发作 post_sec=dur+90，存 bb_zt/relt + onset/offset rel）。
- **Create** `src/topic5_ictal_field_dynamics.py` — 纯数学：source_core / axis_partition / 各指标 / parity helper。无 `scripts.*` 依赖。
- **Create** `tests/test_topic5_ictal_field_dynamics.py` — 纯函数 TDD。
- **Create** `scripts/run_topic5_ictal_field_dynamics.py` — loader（rank-disp+geometry+frame+cache，非 swap-gated）+ 扫窗驱动 → CSV/JSON。
- **Create** `scripts/plot_topic5_ictal_field_dynamics.py` — 两层图 + README。
- **Output** `results/topic5_ictal_recruitment/ictal_field_long_cache/<ds>.{npz,json}`。
- **Output** `results/topic5_ictal_recruitment/field_dynamics/{per_seizure_metrics.csv, per_subject/<ds>.json, figures/}`。

---

## Task 1: 长窗 ictal field cache builder

**Files:**
- Create: `scripts/build_topic5_ictal_field_long_cache.py`
- Output: `results/topic5_ictal_recruitment/ictal_field_long_cache/`

**Interfaces:**
- Consumes: `scripts.build_topic5_t0_feature_cache.{_features_one, _pre_target, recruit, STORE_BB_ZT, PRE_FEATURE_SEC}`；`scripts.run_topic5_t0_eligibility.{_inventory_rows, ICTAL_REFERENCE}`；`src.ictal_onset_extraction.extract_seizure_window`。
- Produces: 每 subject 一个 npz（keys: `channels`, 每 eligible idx 的 `bb_zt__{idx}`, `bb_relt__{idx}`, `hfa_zt__{idx}`, `hfa_relt__{idx}`, `bb_auc__{idx}`, `hfa_auc__{idx}`）+ json meta（`eligible_idxs`, `channels`, `fs`, `seizure[str(idx)] = {seizure_id, pre_sec, post_sec, eeg_onset_rel, eeg_offset_rel, eeg_duration_sec}`, `drops`）。

- [ ] **Step 1: 写脚本骨架**（实现，下一步用真实 subject 验证结构）

```python
"""Topic 5 — 长窗 ictal robust-z field cache (onset-130s .. offset+90s) for field-dynamics pilot.

复用 build_topic5_t0_feature_cache._features_one（同 baseline robust-z / band 口径），只把 post_sec
按每次发作 eeg_duration 自适应到 offset+90s（现有 v2_windows 只到 +20s，不够到 offset）。写到 parallel
dir，不动现有 t0_feature_cache*。channels = bipolar_alias_label（与几何/axis record 同名约定）。
"""
from __future__ import annotations
import argparse, csv, json, math, sys, warnings
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="invalid value encountered")

import scripts.build_topic5_t0_feature_cache as cb
from scripts.run_topic5_t0_eligibility import _inventory_rows, ICTAL_REFERENCE
from src.ictal_onset_extraction import extract_seizure_window

AUDIT = _ROOT / "results/topic5_ictal_recruitment/t0_eligibility_audit.csv"
OUT = _ROOT / "results/topic5_ictal_recruitment/ictal_field_long_cache"
SUBJECTS = ["epilepsiae_442", "epilepsiae_548", "epilepsiae_583",
            "epilepsiae_384", "epilepsiae_958", "epilepsiae_1084"]
POST_PAD = 90.0
MAX_ICTAL_SEC = 600.0   # span 上限（疑似 status；亦防 OOM）


def _eligible_complete(ds_sid, inv_rows):
    """analysis_eligible idx (audit) ∩ has_complete_eeg_interval (inventory)."""
    elig = set()
    for r in csv.DictReader(open(AUDIT)):
        if r["subject_id"] == ds_sid and str(r["analysis_eligible"]).strip().lower() in ("true", "1", "yes"):
            elig.add(int(r["seizure_idx"]))
    out = []
    for idx in sorted(elig):
        inv = inv_rows[idx] if idx < len(inv_rows) else {}
        if str(inv.get("has_complete_eeg_interval", "")).strip().lower() in ("true", "1", "yes", "t"):
            out.append(idx)
    return out


def build_subject(ds_sid):
    cb.STORE_BB_ZT = True          # 存完整 broadband z trace
    cb.PRE_FEATURE_SEC = 130.0     # pre floor（与 v2_windows 一致）
    dataset, sid = ds_sid.split("_", 1)
    ref = ICTAL_REFERENCE[dataset]
    inv_rows, _ = _inventory_rows(dataset, sid)
    idxs = _eligible_complete(ds_sid, inv_rows)
    arrays, drops = {}, []
    meta = {"dataset": dataset, "subject": sid, "hop_sec": cb.HOP if hasattr(cb, "HOP") else 0.1,
            "channels": None, "fs": None, "eligible_idxs": [], "seizure": {}, "post_pad": POST_PAD,
            "baseline": {"guard_sec": 60.0,
                         "note": "robust-z baseline=[-pre_sec,-60] adaptive (resolve_baseline_window, "
                                 "eeg-rel clipped); per-seizure pre_sec in seizure[idx]"}}
    for idx in idxs:
        inv = inv_rows[idx]
        try:
            eeg_dur = float(inv["eeg_duration_sec"])
            clin_on = float(inv["clin_onset_epoch"])
            eeg_off_rel = float(inv["eeg_offset_epoch"]) - clin_on
            eeg_on_rel = float(inv["eeg_onset_epoch"]) - clin_on
        except (KeyError, TypeError, ValueError) as e:
            drops.append({"idx": idx, "reason": f"inv_field:{type(e).__name__}"}); continue
        span = max(eeg_off_rel, eeg_dur)   # P1: 覆盖 eeg offset，即使 eeg_onset 晚于 clin_onset(384 ~+36s)
        if span > MAX_ICTAL_SEC:
            drops.append({"idx": idx, "reason": f"duration_too_long_for_pilot:{span:.0f}s"}); continue
        pre = cb._pre_target(dataset, inv)
        post = math.ceil(span) + POST_PAD
        try:
            sw = extract_seizure_window(f"{dataset}/{sid}", idx, pre_sec=pre, post_sec=post, reference=ref)
        except Exception as e:
            drops.append({"idx": idx, "reason": f"extract:{type(e).__name__}"}); continue
        eeg_rel = (sw.eeg_onset_epoch - sw.clin_onset_epoch) if sw.eeg_onset_epoch is not None else None
        try:
            bb_auc, hfa_auc, ramp, hfa_zt, bact, bb_zt, bb_relt, hfa_relt = cb._features_one(sw, eeg_rel)
        except Exception as e:
            drops.append({"idx": idx, "reason": f"features:{type(e).__name__}"}); continue
        ch = [cb.recruit.bipolar_alias_label(c) for c in sw.ch_names]
        if meta["channels"] is None:
            meta["channels"] = ch; meta["fs"] = float(sw.fs)
        elif len(ch) != len(meta["channels"]):
            drops.append({"idx": idx, "reason": f"chan_count:{len(ch)}!={len(meta['channels'])}"}); continue
        arrays[f"bb_zt__{idx}"] = bb_zt; arrays[f"bb_relt__{idx}"] = bb_relt
        arrays[f"hfa_zt__{idx}"] = hfa_zt; arrays[f"hfa_relt__{idx}"] = hfa_relt
        arrays[f"bb_auc__{idx}"] = bb_auc.astype(np.float32)
        arrays[f"hfa_auc__{idx}"] = hfa_auc.astype(np.float32)
        meta["eligible_idxs"].append(idx)
        meta["seizure"][str(idx)] = {"seizure_id": sw.seizure_id, "pre_sec": float(sw.pre_sec),
                                     "post_sec": float(post), "eeg_onset_rel": eeg_on_rel,
                                     "eeg_offset_rel": eeg_off_rel, "eeg_duration_sec": eeg_dur}
        print(f"  [{ds_sid} sz{idx}] cached post={post:.0f}s dur={eeg_dur:.0f}s", flush=True)
    meta["drops"] = drops
    if not meta["eligible_idxs"]:
        print(f"  [{ds_sid}] nothing cached ({len(drops)} drops)", flush=True); return
    arrays["channels"] = np.array(meta["channels"])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{ds_sid}.npz", **arrays)
    json.dump(meta, open(OUT / f"{ds_sid}.json", "w"), indent=2, ensure_ascii=False)
    print(f"  [{ds_sid}] wrote {len(meta['eligible_idxs'])} sz, {len(drops)} drops", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="*", default=SUBJECTS)
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()
    for ds_sid in args.subjects:
        if (OUT / f"{ds_sid}.npz").exists() and not args.restart:
            print(f"[cache] {ds_sid} exists, skip", flush=True); continue
        print(f"[cache] {ds_sid} ...", flush=True)
        try:
            build_subject(ds_sid)
        except Exception as e:
            print(f"  SUBJECT ERROR {type(e).__name__}: {e}", flush=True)
    print("LONG CACHE DONE", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑最小 subject 验证结构**（integration checkpoint；I/O-bound，~分钟级）

Run: `python scripts/build_topic5_ictal_field_long_cache.py --subjects epilepsiae_384`
Expected: 打印若干 `cached post=..s`，末尾 `wrote N sz`；生成 `results/topic5_ictal_recruitment/ictal_field_long_cache/epilepsiae_384.{npz,json}`。

- [ ] **Step 3: 断言 npz/meta 内容 + 长度覆盖到 offset**

Run:
```bash
python - <<'PY'
import json, numpy as np
from pathlib import Path
d = Path("results/topic5_ictal_recruitment/ictal_field_long_cache")
m = json.load(open(d/"epilepsiae_384.json")); z = np.load(d/"epilepsiae_384.npz", allow_pickle=True)
assert m["eligible_idxs"], "no seizures"
for i in m["eligible_idxs"]:                               # P1: 遍历全部，不只第一个
    s = m["seizure"][str(i)]
    assert f"bb_zt__{i}" in z.files and f"bb_relt__{i}" in z.files, f"sz{i} missing bb trace"
    relt = z[f"bb_relt__{i}"]
    assert relt.max() >= s["eeg_offset_rel"] + 60, \
        f"sz{i} trace_max {relt.max():.0f}s < offset+60 {s['eeg_offset_rel']+60:.0f}s (eeg_onset_rel={s['eeg_onset_rel']:.1f})"
    assert relt.min() <= -120, f"sz{i} pre {relt.min():.0f}s not <= -120"
print("OK ALL", len(m["eligible_idxs"]), "seizures cover offset+60; channels", len(z["channels"]))
PY
```
Expected: `OK channels .. sz .. offset_rel .. trace_max ..`，且 trace_max ≥ offset+60。

- [ ] **Step 4: 建全部 6 subject**（detached；1084 有 72 sz，最慢）

Run (background): `python scripts/build_topic5_ictal_field_long_cache.py`
Expected: 6 个 npz/json；`LONG CACHE DONE`。

- [ ] **Step 5: Commit**

```bash
git add scripts/build_topic5_ictal_field_long_cache.py
git commit -m "feat(topic5 field-dynamics): long ictal robust-z cache builder (onset-130s..offset+90s)"
```

---

## Task 2: 纯几何 — source_core + axis_partition（TDD）

**Files:**
- Create: `src/topic5_ictal_field_dynamics.py`
- Test: `tests/test_topic5_ictal_field_dynamics.py`

**Interfaces:**
- Produces: `source_core(order_names, pos, compact_mm=15.0) -> (core_names:list, uncertain:bool, top2_dist:float)`；`axis_partition(names, pos, core_a, core_b, *, mid_band=(0.25,0.75), degen_frac=0.15) -> dict(groups, t, d, P_A, P_B, L, bbox_diag, axis_degenerate, med_d)`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_topic5_ictal_field_dynamics.py
import numpy as np
import pytest
from src import topic5_ictal_field_dynamics as fd


def test_source_core_compact_takes_top2():
    pos = {"a": (0, 0), "b": (5, 0), "c": (40, 0)}
    core, uncertain, dist = fd.source_core(["a", "b", "c"], pos, compact_mm=15.0)
    assert core == ["a", "b"] and uncertain is False and dist == pytest.approx(5.0)


def test_source_core_scattered_falls_back_to_single():
    pos = {"a": (0, 0), "b": (40, 0)}
    core, uncertain, dist = fd.source_core(["a", "b"], pos, compact_mm=15.0)
    assert core == ["a"] and uncertain is True and dist == pytest.approx(40.0)


def test_source_core_single_mapped_is_uncertain():
    core, uncertain, dist = fd.source_core(["a"], {"a": (1, 1)}, compact_mm=15.0)
    assert core == ["a"] and uncertain is True and np.isnan(dist)


def test_axis_partition_mece_four_groups():
    # axis along x from a(0,0) to b(40,0); mid contact on-axis; off-axis contact far in y
    pos = {"a": (0, 0), "b": (40, 0), "mid": (20, 1), "off": (20, 30), "endish": (4, 1)}
    r = fd.axis_partition(["a", "b", "mid", "off", "endish"], pos, ["a"], ["b"])
    g = r["groups"]
    assert g["a"] == "source_core" and g["b"] == "source_core"
    assert g["mid"] == "axial_mid"
    assert g["off"] == "non_axial"
    assert set(g.values()) <= {"source_core", "axis_end_noncore", "axial_mid", "non_axial"}
    assert len(g) == 5  # all mapped covered


def test_axis_partition_degenerate_when_cores_coincide():
    pos = {"a": (10, 0), "b": (10.5, 0), "x": (40, 0), "y": (0, 30)}
    r = fd.axis_partition(["a", "b", "x", "y"], pos, ["a"], ["b"])
    assert r["axis_degenerate"] is True
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_topic5_ictal_field_dynamics.py -x -q`
Expected: FAIL — `ModuleNotFoundError` / `AttributeError: ... source_core`。

- [ ] **Step 3: 实现 source_core + axis_partition**

```python
# src/topic5_ictal_field_dynamics.py
"""Topic 5 发作内 field 动力学 — 纯数学（source-core 定位、轴四分区、各 field 指标）。
设计: docs/superpowers/specs/2026-06-28-topic5-ictal-field-dynamics-design.md。无 scripts.* 依赖。"""
from __future__ import annotations
import numpy as np

GROUPS = ("source_core", "axis_end_noncore", "axial_mid", "non_axial")


def source_core(order_names, pos, compact_mm=15.0):
    """order_names: 触点名 earliest-first（已过滤到 valid）。pos: {name:(x,y)} mm。
    top2 间距 < compact_mm → 双点 core；否则单点 + uncertain。返回 (core, uncertain, top2_dist)。"""
    mapped = [n for n in order_names if n in pos]
    if not mapped:
        return [], True, float("nan")
    if len(mapped) == 1:
        return [mapped[0]], True, float("nan")
    p0, p1 = np.asarray(pos[mapped[0]], float), np.asarray(pos[mapped[1]], float)
    d = float(np.hypot(*(p1 - p0)))
    if d < compact_mm:
        return [mapped[0], mapped[1]], False, d
    return [mapped[0]], True, d


def axis_partition(names, pos, core_a, core_b, *, mid_band=(0.25, 0.75), degen_frac=0.15):
    """names: 全部 mapped 名。core_a/core_b: 两侧 source-core 名。投影到 P_A->P_B 线段，
    按 d(垂距) 中位 + t(沿轴位置) 划 4 组 MECE。"""
    P = {n: np.asarray(pos[n], float) for n in names if n in pos}
    pts = np.array(list(P.values()))
    bbox_diag = float(np.hypot(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))) if len(pts) > 1 else float("nan")
    PA = np.mean([P[n] for n in core_a if n in P], axis=0)
    PB = np.mean([P[n] for n in core_b if n in P], axis=0)
    u = PB - PA
    L = float(np.hypot(*u))
    axis_degenerate = (not np.isfinite(L)) or (np.isfinite(bbox_diag) and L < degen_frac * bbox_diag)
    src = set(core_a) | set(core_b)
    t, d = {}, {}
    for n, p in P.items():
        if L > 0:
            tt = float(np.dot(p - PA, u) / (L * L))
            t[n] = tt
            d[n] = float(np.hypot(*((p - PA) - tt * u)))
        else:
            t[n] = float("nan"); d[n] = float("nan")
    nonsrc = [n for n in P if n not in src]
    med_d = float(np.median([d[n] for n in nonsrc])) if nonsrc else float("nan")
    groups = {}
    for n in P:
        if n in src:
            groups[n] = "source_core"
        elif np.isfinite(d[n]) and d[n] > med_d:
            groups[n] = "non_axial"
        elif np.isfinite(t[n]) and mid_band[0] <= t[n] <= mid_band[1]:
            groups[n] = "axial_mid"
        else:
            groups[n] = "axis_end_noncore"
    return dict(groups=groups, t=t, d=d, P_A=PA.tolist(), P_B=PB.tolist(),
                L=L, bbox_diag=bbox_diag, axis_degenerate=bool(axis_degenerate), med_d=med_d)
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_topic5_ictal_field_dynamics.py -x -q`
Expected: 5 passed。

- [ ] **Step 5: Commit**

```bash
git add src/topic5_ictal_field_dynamics.py tests/test_topic5_ictal_field_dynamics.py
git commit -m "feat(topic5 field-dynamics): source-core compact locator + MECE axis partition (TDD)"
```

---

## Task 3: 纯指标 — positive_mass_share / gradient / synchrony / angle / parity（TDD）

**Files:**
- Modify: `src/topic5_ictal_field_dynamics.py`
- Modify: `tests/test_topic5_ictal_field_dynamics.py`

**Interfaces:**
- Produces: `positive_mass_share(zmean, groups) -> {group:float}`（四组和≈1）；`group_mean(zmean, groups, group) -> float`；`field_gradient(zmean, pos) -> (angle_deg, mag)`；`fold_angle_deg(a,b) -> float[0,90]`；`field_synchrony(ztraces) -> float`；`participation(zmean, thresh=2.0) -> float`；`offset_pre_onset_overlap(win_lo_rel_offset, eeg_offset_rel) -> bool`；`parity_max_abs_diff(a,b) -> float`。

- [ ] **Step 1: 写失败测试（追加）**

```python
def test_positive_mass_share_stable_with_negative_mean():
    # robust-z 可负；rectified mass 永远有界、不反号
    zmean = {"a": 4.0, "b": -3.0, "c": 2.0, "d": -1.0}
    groups = {"a": "axial_mid", "b": "non_axial", "c": "non_axial", "d": "source_core"}
    pms = fd.positive_mass_share(zmean, groups)
    assert pms["axial_mid"] == pytest.approx(4 / 6) and pms["non_axial"] == pytest.approx(2 / 6)
    assert pms["source_core"] == 0.0
    assert sum(pms.values()) == pytest.approx(1.0)


def test_positive_mass_share_all_nonpositive_is_zero():
    pms = fd.positive_mass_share({"a": -1.0, "b": -2.0}, {"a": "axial_mid", "b": "non_axial"})
    assert all(v == 0.0 for v in pms.values())


def test_field_gradient_recovers_known_direction():
    # z 沿 x 增 → 梯度角≈0°
    pos = {f"c{i}": (float(i), float(j)) for i in range(5) for j in range(3)}
    zmean = {n: pos[n][0] for n in pos}
    ang, mag = fd.field_gradient(zmean, pos)
    assert fd.fold_angle_deg(ang, 0.0) < 1.0 and mag > 0


def test_fold_angle_deg_axis_invariant():
    assert fd.fold_angle_deg(170.0, 0.0) == pytest.approx(10.0)
    assert fd.fold_angle_deg(95.0, 0.0) == pytest.approx(85.0)


def test_field_synchrony_identical_traces_is_one():
    base = np.array([0.0, 1.0, 2.0, 3.0, 2.0])
    s = fd.field_synchrony({"a": base, "b": base * 1.0, "c": base + 0.5})
    assert s == pytest.approx(1.0)


def test_participation_fraction():
    assert fd.participation({"a": 3.0, "b": 1.0, "c": 2.5, "d": -1.0}, thresh=2.0) == pytest.approx(0.5)


def test_offset_pre_onset_overlap_short_seizure():
    # 25s 发作，offset_rel≈25；终止窗 [-60,-30] 左缘 rel-onset = 25-60 = -35 < 0 → overlap
    assert fd.offset_pre_onset_overlap(-60.0, 25.0) is True
    assert fd.offset_pre_onset_overlap(-10.0, 25.0) is False


def test_parity_max_abs_diff_ignores_nan():
    a = np.array([1.0, 2.0, np.nan]); b = np.array([1.0, 2.0005, 9.0])
    assert fd.parity_max_abs_diff(a, b) == pytest.approx(0.0005, abs=1e-6)
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_topic5_ictal_field_dynamics.py -x -q`
Expected: FAIL — `AttributeError: ... positive_mass_share`。

- [ ] **Step 3: 实现（追加到 `src/topic5_ictal_field_dynamics.py`）**

```python
def positive_mass_share(zmean, groups):
    rect = {n: max(float(v), 0.0) for n, v in zmean.items() if np.isfinite(v)}
    total = sum(rect.values())
    out = {g: 0.0 for g in GROUPS}
    if total <= 0:
        return out
    for n, r in rect.items():
        g = groups.get(n)
        if g in out:
            out[g] += r / total
    return out


def group_mean(zmean, groups, group):
    vals = [float(v) for n, v in zmean.items() if groups.get(n) == group and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def field_gradient(zmean, pos):
    items = [(pos[n][0], pos[n][1], float(v)) for n, v in zmean.items()
             if n in pos and np.isfinite(v)]
    if len(items) < 3:
        return float("nan"), float("nan")
    A = np.array([[x, y, 1.0] for x, y, _ in items], float)
    z = np.array([v for _, _, v in items], float)
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    return float(np.degrees(np.arctan2(b, a))), float(np.hypot(a, b))


def fold_angle_deg(a_deg, b_deg):
    if not (np.isfinite(a_deg) and np.isfinite(b_deg)):
        return float("nan")
    diff = abs((a_deg - b_deg) % 180.0)
    return float(min(diff, 180.0 - diff))


def field_synchrony(ztraces):
    names = [n for n, tr in ztraces.items()
             if np.isfinite(tr).sum() >= 2 and np.nanstd(tr) > 0]
    corrs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = np.asarray(ztraces[names[i]], float), np.asarray(ztraces[names[j]], float)
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() >= 2 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
                corrs.append(np.corrcoef(a[ok], b[ok])[0, 1])
    return float(np.median(corrs)) if corrs else float("nan")


def participation(zmean, thresh=2.0):
    vals = [float(v) for v in zmean.values() if np.isfinite(v)]
    return float(np.mean([v > thresh for v in vals])) if vals else float("nan")


def offset_pre_onset_overlap(win_lo_rel_offset, eeg_offset_rel):
    return bool((eeg_offset_rel + win_lo_rel_offset) < 0.0)


def parity_max_abs_diff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[m] - b[m]))) if m.any() else float("inf")
```

- [ ] **Step 4: 运行确认全过**

Run: `pytest tests/test_topic5_ictal_field_dynamics.py -q`
Expected: 13 passed。

- [ ] **Step 5: Commit**

```bash
git add src/topic5_ictal_field_dynamics.py tests/test_topic5_ictal_field_dynamics.py
git commit -m "feat(topic5 field-dynamics): positive-mass-share + gradient/synchrony/parity metrics (TDD)"
```

---

## Task 4: Subject loader + interictal 场 + per-window 指标装配

**Files:**
- Create: `scripts/run_topic5_ictal_field_dynamics.py`（本 task 只到 loader + 单窗指标，下一 task 加扫窗/IO）

**Interfaces:**
- Consumes: Task 1 cache；`src.topic5_ictal_field_dynamics`(Task 2/3)；`src.propagation_contact_plane_readout.{make_plane_grid, R_smooth_rank, corr_pair_mirror_invariant, S_THRESH, OVERLAP_MIN}`；`src.topic5_axis_alignment.{matched_channels, make_field_record}`；`scripts.plot_contact_plane_static.{_subject_display_frame, _display_points, _attach_real_coords}`；`scripts.plot_topic5_swap_nodes_fields._arrays`。
- Produces: `load_context(ds_sid) -> dict`（keys: `names_m, pos, soz, matched, names_geo, order_a, order_b, decision_k, X, Y, sigma, F_inter_a, F_inter_b, core_a, core_b, uncert_a, uncert_b, dist_a, dist_b, part, frame, ta`）；`window_maxab(ctx, vals_by_name) -> float`。

- [ ] **Step 1: 写 loader + 单窗 alignment（实现）**

```python
"""Topic 5 发作内 field 动力学 — 驱动：loader（非 swap-gated）+ 扫窗写 CSV/JSON。"""
from __future__ import annotations
import argparse, csv, json, sys, warnings
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered")

from src import topic5_ictal_field_dynamics as fd
from src.propagation_contact_plane_readout import (make_plane_grid, R_smooth_rank,
                                                   corr_pair_mirror_invariant, S_THRESH, OVERLAP_MIN)
from src.topic5_axis_alignment import matched_channels, make_field_record
from scripts.plot_contact_plane_static import _subject_display_frame, _display_points, _attach_real_coords
from scripts.plot_topic5_swap_nodes_fields import _arrays

RD_DIR = _ROOT / "results/interictal_propagation_masked/rank_displacement/per_subject"
GEO_DIR = _ROOT / "results/spatial_modulation/propagation_geometry/observation_readout/real_subjects"
CACHE = _ROOT / "results/topic5_ictal_recruitment/ictal_field_long_cache"
V2REF = _ROOT / "results/topic5_ictal_recruitment/t0_feature_cache_v2_windows"
OUT = _ROOT / "results/topic5_ictal_recruitment/field_dynamics"
SUBJECTS = ["epilepsiae_442", "epilepsiae_548", "epilepsiae_583",
            "epilepsiae_384", "epilepsiae_958", "epilepsiae_1084"]
PARITY_TOL = 1e-3


def _abs_corr(Fi, Fj):
    r = corr_pair_mirror_invariant(Fi["T"], Fi["S"], Fj["T"], Fj["S"], S_THRESH, OVERLAP_MIN)["corr"]
    return abs(r) if r is not None and np.isfinite(r) else np.nan


def load_context(ds_sid):
    rd = json.load(open(RD_DIR / f"{ds_sid}.json"))
    pp = rd.get("primary_pair") or rd["pairs"][0]
    rd_names = pp["channel_names"]
    jv = np.asarray(pp["joint_valid"], bool)
    ra = np.asarray(pp["rank_a_dense_full"], float)
    rb = np.asarray(pp["rank_b_dense_full"], float)
    decision_k = int(pp["swap_sweep"]["decision_k"])
    ta = json.load(open(GEO_DIR / f"{ds_sid}_t_a.json"))
    tb = json.load(open(GEO_DIR / f"{ds_sid}_t_b.json"))
    recs = [ta, tb]
    _attach_real_coords(recs)
    frame = _subject_display_frame(recs)
    names_geo, xs, ys, inter, sup, soz = _arrays(ta, frame)
    pos = {n: (float(x), float(y)) for n, x, y in zip(names_geo, xs, ys)
           if np.isfinite(x) and np.isfinite(y)}
    sozmap = {n: bool(s) for n, s in zip(names_geo, soz)}
    data = np.load(CACHE / f"{ds_sid}.npz", allow_pickle=True)
    cache_names = [str(x) for x in data["channels"]]
    matched = matched_channels(ta, {n: 0.0 for n in cache_names})
    names_m = [c["name"] for c in matched]
    mapped = [n for n in names_m if n in pos]

    def order(rank):
        idx = [i for i in range(len(rd_names))
               if jv[i] and np.isfinite(rank[i]) and rd_names[i] in pos]
        return [rd_names[i] for i in sorted(idx, key=lambda i: rank[i])]

    order_a, order_b = order(ra), order(rb)
    core_a, uncert_a, dist_a = fd.source_core(order_a, pos)
    core_b, uncert_b, dist_b = fd.source_core(order_b, pos)
    part = fd.axis_partition(mapped, pos, core_a, core_b)
    X, Y = make_plane_grid()
    F_inter_a = R_smooth_rank(make_field_record(matched, [float(c["typical_rank"]) for c in matched]),
                              X, Y, None, S_THRESH)
    sigma = F_inter_a["sigma_xy"]
    mb = matched_channels(tb, {n: 0.0 for n in cache_names})
    brank = {c["name"]: float(c["typical_rank"]) for c in mb}
    inter_b = [brank.get(n, np.nan) for n in names_m]
    F_inter_b = (R_smooth_rank(make_field_record(matched, inter_b), X, Y, sigma, S_THRESH)
                 if np.isfinite(inter_b).sum() >= 4 else None)
    return dict(ds_sid=ds_sid, names_m=names_m, mapped=mapped, pos=pos, soz=sozmap, matched=matched,
                order_a=order_a, order_b=order_b, decision_k=decision_k, X=X, Y=Y, sigma=sigma,
                F_inter_a=F_inter_a, F_inter_b=F_inter_b, core_a=core_a, core_b=core_b,
                uncert_a=uncert_a, uncert_b=uncert_b, dist_a=dist_a, dist_b=dist_b,
                part=part, frame=frame, ta=ta)


def window_maxab(ctx, vals_by_name):
    """vals_by_name: {name->raw window-mean z}. 传 raw（与 bb_auc 同路径），R_smooth_rank 内部处理。"""
    vals = [vals_by_name.get(n, np.nan) for n in ctx["names_m"]]
    Fw = R_smooth_rank(make_field_record(ctx["matched"], vals), ctx["X"], ctx["Y"], ctx["sigma"], S_THRESH)
    ra = _abs_corr(ctx["F_inter_a"], Fw)
    if ctx["F_inter_b"] is None:
        return ra
    rb = _abs_corr(ctx["F_inter_b"], Fw)
    v = [x for x in (ra, rb) if np.isfinite(x)]
    return float(max(v)) if v else float("nan")
```

- [ ] **Step 2: 验证 loader（integration，需 Task 1 的 384 cache）**

Run:
```bash
python - <<'PY'
import sys; sys.path.insert(0, ".")
from scripts.run_topic5_ictal_field_dynamics import load_context
for s in ("epilepsiae_384", "epilepsiae_1084"):
    c = load_context(s)
    g = c["part"]["groups"]; from collections import Counter
    print(s, "mapped", len(c["mapped"]), "groups", dict(Counter(g.values())),
          "degen", c["part"]["axis_degenerate"], "uncertA/B", c["uncert_a"], c["uncert_b"],
          "F_inter_b", c["F_inter_b"] is not None)
PY
```
Expected: 384 出四组非空、`degen False`、`uncertA/B True True`（实测两侧散）；**1084 `degen True`**（退化轴负控）。

- [ ] **Step 3: Commit**

```bash
git add scripts/run_topic5_ictal_field_dynamics.py
git commit -m "feat(topic5 field-dynamics): non-gated subject loader + maxAB window alignment"
```

---

## Task 5: 扫窗驱动 — CSV + per_subject JSON（含 parity gate）

**Files:**
- Modify: `scripts/run_topic5_ictal_field_dynamics.py`（加扫窗 + 写出 + parity + main）

**Interfaces:**
- Consumes: Task 4 `load_context` / `window_maxab`；Task 1 cache；`V2REF` parity 参考。
- Produces: `results/topic5_ictal_recruitment/field_dynamics/per_seizure_metrics.csv`（spec §6 列）+ `per_subject/<ds>.json`。

- [ ] **Step 1: 加扫窗 + 写出（实现，追加）**

```python
ONSET_WIN, ONSET_STEP = 10.0, 5.0
OFFSET_WINS = [(-60, -30), (-30, -10), (-10, 0), (0, 30)]


def _slice(zt, relt, lo, hi):
    """返回 (window-mean per ch, ztrace per ch[n_ch x n_bins_in_window])；无 bin 时 (None,None)。"""
    m = (relt >= lo) & (relt <= hi)
    if m.sum() == 0:
        return None, None
    sub = zt[:, m]
    return np.nanmean(sub, axis=1), sub


def _zmean_by_name(zmean_vec, cache_names, mapped):
    idx = {n: i for i, n in enumerate(cache_names)}
    return {n: float(zmean_vec[idx[n]]) for n in mapped if n in idx and np.isfinite(zmean_vec[idx[n]])}


def _ztraces_by_name(sub, cache_names, mapped):
    idx = {n: i for i, n in enumerate(cache_names)}
    return {n: sub[idx[n]] for n in mapped if n in idx}


def _metrics_row(ctx, zmean_by_name, ztr_by_name):
    g = ctx["part"]["groups"]
    pms = fd.positive_mass_share(zmean_by_name, g)
    allmean = fd.group_mean(zmean_by_name, {n: "all" for n in zmean_by_name}, "all")
    sc = fd.group_mean(zmean_by_name, g, "source_core")
    am = fd.group_mean(zmean_by_name, g, "axial_mid")
    na = fd.group_mean(zmean_by_name, g, "non_axial")
    en = fd.group_mean(zmean_by_name, g, "axis_end_noncore")
    ang, mag = fd.field_gradient(zmean_by_name, ctx["pos"])

    def share(grp):
        vals = [v for n, v in zmean_by_name.items() if g.get(n) == grp and np.isfinite(v)]
        return float(np.mean([v > 0 for v in vals])) if vals else float("nan")

    nonax = [v for n, v in zmean_by_name.items() if g.get(n) == "non_axial" and np.isfinite(v)]
    return dict(n_matched=len(zmean_by_name), align_maxab=window_maxab(ctx, zmean_by_name),
                grad_angle=ang, grad_mag=mag, source_core_mean_z=sc, axis_end_noncore_mean_z=en,
                axial_mid_mean_z=am, non_axial_mean_z=na,
                axialmid_minus_nonaxial=(am - na) if np.isfinite(am) and np.isfinite(na) else float("nan"),
                source_core_minus_all=(sc - allmean) if np.isfinite(sc) and np.isfinite(allmean) else float("nan"),
                pms_source_core=pms["source_core"], pms_axis_end_noncore=pms["axis_end_noncore"],
                pms_axial_mid=pms["axial_mid"], pms_non_axial=pms["non_axial"],
                source_core_pos_share=share("source_core"), axial_mid_pos_share=share("axial_mid"),
                non_axial_pos_share=share("non_axial"),
                non_axial_p95_z=float(np.nanpercentile(nonax, 95)) if nonax else float("nan"),
                sync_median_corr=fd.field_synchrony(ztr_by_name),
                participation=fd.participation(zmean_by_name))


def _parity_fail(ds_sid, idx, long_npz):
    """长 cache bb_auc[idx] vs v2_windows bb_auc[idx]，max|Δ|>tol → True。参考缺失 → False(警告)。"""
    ref = V2REF / f"{ds_sid}.npz"
    if not ref.exists():
        return False, "no_ref"
    r = np.load(ref, allow_pickle=True)
    k = f"bb_auc__{idx}"
    if k not in r.files or k not in long_npz.files:
        return False, "no_key"
    diff = fd.parity_max_abs_diff(long_npz[k], r[k])
    return (diff > PARITY_TOL), f"{diff:.2e}"


def run_subject(ds_sid):
    ctx = load_context(ds_sid)
    meta = json.load(open(CACHE / f"{ds_sid}.json"))
    data = np.load(CACHE / f"{ds_sid}.npz", allow_pickle=True)
    cache_names = [str(x) for x in data["channels"]]
    rows, n_long, n_parity_fail = [], 0, 0
    for idx in meta["eligible_idxs"]:
        s = meta["seizure"][str(idx)]
        off = s["eeg_offset_rel"]
        pf, pf_info = _parity_fail(ds_sid, idx, data)
        if pf:
            n_parity_fail += 1
            continue
        zt, relt = data[f"bb_zt__{idx}"], data[f"bb_relt__{idx}"]
        dur = s["eeg_duration_sec"]
        if dur >= 40:
            n_long += 1
        # onset 滑窗
        lo = 0.0
        while lo + ONSET_WIN <= max(off + 1e-6, ONSET_WIN):
            zmv, sub = _slice(zt, relt, lo, lo + ONSET_WIN)
            tc = lo + ONSET_WIN / 2
            if zmv is not None:
                zmn = _zmean_by_name(zmv, cache_names, ctx["mapped"])
                if len(zmn) >= 6:
                    r = dict(ds_sid=ds_sid, subject=ds_sid.split("_", 1)[1], seizure_idx=idx,
                             seizure_id=s["seizure_id"], window_kind="onset_slide",
                             t_center_rel_onset=tc, t_center_rel_offset=tc - off,
                             progress_frac=(tc / off) if off > 0 else float("nan"),
                             pre_onset_overlap=False, parity_fail=False, band="bb",
                             axis_degenerate=ctx["part"]["axis_degenerate"],
                             source_focus_uncertain_a=ctx["uncert_a"], source_focus_uncertain_b=ctx["uncert_b"],
                             n_source_core=sum(v == "source_core" for v in ctx["part"]["groups"].values()),
                             n_axis_end_noncore=sum(v == "axis_end_noncore" for v in ctx["part"]["groups"].values()),
                             n_axial_mid=sum(v == "axial_mid" for v in ctx["part"]["groups"].values()),
                             n_non_axial=sum(v == "non_axial" for v in ctx["part"]["groups"].values()))
                    r.update(_metrics_row(ctx, zmn, _ztraces_by_name(sub, cache_names, ctx["mapped"])))
                    rows.append(r)
            lo += ONSET_STEP
        # offset 终止窗
        for wlo, whi in OFFSET_WINS:
            zmv, sub = _slice(zt, relt, off + wlo, off + whi)
            if zmv is None:
                continue
            zmn = _zmean_by_name(zmv, cache_names, ctx["mapped"])
            if len(zmn) < 6:
                continue
            r = dict(ds_sid=ds_sid, subject=ds_sid.split("_", 1)[1], seizure_idx=idx,
                     seizure_id=s["seizure_id"], window_kind="offset_aligned",
                     t_center_rel_onset=off + (wlo + whi) / 2, t_center_rel_offset=(wlo + whi) / 2,
                     progress_frac=float("nan"),
                     pre_onset_overlap=fd.offset_pre_onset_overlap(wlo, off),
                     parity_fail=False, band="bb", axis_degenerate=ctx["part"]["axis_degenerate"],
                     source_focus_uncertain_a=ctx["uncert_a"], source_focus_uncertain_b=ctx["uncert_b"],
                     n_source_core=sum(v == "source_core" for v in ctx["part"]["groups"].values()),
                     n_axis_end_noncore=sum(v == "axis_end_noncore" for v in ctx["part"]["groups"].values()),
                     n_axial_mid=sum(v == "axial_mid" for v in ctx["part"]["groups"].values()),
                     n_non_axial=sum(v == "non_axial" for v in ctx["part"]["groups"].values()))
            r.update(_metrics_row(ctx, zmn, _ztraces_by_name(sub, cache_names, ctx["mapped"])))
            rows.append(r)
    # drift（每发作各窗 grad_angle - onset 首窗 grad_angle，fold[0,90]）
    by_sz = {}
    for r in rows:
        by_sz.setdefault(r["seizure_idx"], []).append(r)
    for idx, rs in by_sz.items():
        onset_rs = sorted([x for x in rs if x["window_kind"] == "onset_slide"],
                          key=lambda x: x["t_center_rel_onset"])
        ref_ang = onset_rs[0]["grad_angle"] if onset_rs else float("nan")
        for r in rs:
            r["drift_vs_onset"] = fd.fold_angle_deg(r["grad_angle"], ref_ang)
            r["angle_to_interictal_axis"] = fd.fold_angle_deg(r["grad_angle"], 0.0)
    subj = dict(ds_sid=ds_sid, n_eligible=len(meta["eligible_idxs"]),
                n_used=len(by_sz), n_parity_fail=n_parity_fail, n_long_seizures=n_long,
                axis=dict(L=ctx["part"]["L"], bbox_diag=ctx["part"]["bbox_diag"],
                          axis_degenerate=ctx["part"]["axis_degenerate"],
                          source_core_a=ctx["core_a"], source_core_b=ctx["core_b"],
                          source_focus_uncertain_a=ctx["uncert_a"], source_focus_uncertain_b=ctx["uncert_b"],
                          source_top2_dist_a_mm=ctx["dist_a"], source_top2_dist_b_mm=ctx["dist_b"],
                          decision_k_provenance=ctx["decision_k"]))
    return rows, subj
```

- [ ] **Step 2: 加 main（写 CSV + JSON）**

```python
CSV_COLS = ["ds_sid", "subject", "seizure_idx", "seizure_id", "window_kind", "t_center_rel_onset",
            "t_center_rel_offset", "progress_frac", "pre_onset_overlap", "parity_fail", "band",
            "n_matched", "n_source_core", "n_axis_end_noncore", "n_axial_mid", "n_non_axial",
            "axis_degenerate", "source_focus_uncertain_a", "source_focus_uncertain_b", "align_maxab",
            "drift_vs_onset", "angle_to_interictal_axis", "grad_mag", "source_core_mean_z",
            "axis_end_noncore_mean_z", "axial_mid_mean_z", "non_axial_mean_z", "axialmid_minus_nonaxial",
            "source_core_minus_all", "pms_source_core", "pms_axis_end_noncore", "pms_axial_mid",
            "pms_non_axial", "source_core_pos_share", "axial_mid_pos_share", "non_axial_pos_share",
            "non_axial_p95_z", "sync_median_corr", "participation"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="*", default=SUBJECTS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_subject").mkdir(exist_ok=True)
    all_rows = []
    for ds_sid in args.subjects:
        if not (CACHE / f"{ds_sid}.npz").exists():
            print(f"[skip] {ds_sid} no long cache", flush=True); continue
        rows, subj = run_subject(ds_sid)
        all_rows += rows
        json.dump(subj, open(OUT / "per_subject" / f"{ds_sid}.json", "w"), indent=2, ensure_ascii=False)
        print(f"[{ds_sid}] {len(rows)} window-rows, {subj['n_used']} sz, parity_fail={subj['n_parity_fail']}, "
              f"degen={subj['axis']['axis_degenerate']}", flush=True)
    with open(OUT / "per_seizure_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader(); w.writerows(all_rows)
    print(f"[done] {len(all_rows)} rows -> {OUT/'per_seizure_metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 跑 384 + 1084 验证**

Run: `python scripts/run_topic5_ictal_field_dynamics.py --subjects epilepsiae_384 epilepsiae_1084`
Expected: 两行 `[..] N window-rows ..`；384 `degen=False`，**1084 `degen=True`**。

- [ ] **Step 4: 断言 CSV/JSON + source_core sanity + 负控**

Run:
```bash
python - <<'PY'
import csv, json, numpy as np
from pathlib import Path
O = Path("results/topic5_ictal_recruitment/field_dynamics")
rows = list(csv.DictReader(open(O/"per_seizure_metrics.csv")))
assert rows, "empty csv"
need = {"align_maxab","pms_axial_mid","pms_non_axial","source_core_minus_all","sync_median_corr","drift_vs_onset"}
assert need <= set(rows[0]), need - set(rows[0])
# 384 source_core sanity: 多数窗 source_core_minus_all > 0
v = [float(r["source_core_minus_all"]) for r in rows if r["ds_sid"]=="epilepsiae_384"
     and r["source_core_minus_all"] not in ("","nan") and np.isfinite(float(r["source_core_minus_all"]))]
assert v and np.mean([x>0 for x in v]) >= 0.5, f"source_core not consistently hot: {np.mean([x>0 for x in v]):.2f}"
# 1084 负控
j = json.load(open(O/"per_subject"/"epilepsiae_1084.json"))
assert j["axis"]["axis_degenerate"] is True, "1084 should be degenerate"
print("OK rows", len(rows), "384 sc_hot_frac", round(np.mean([x>0 for x in v]),2),
      "1084 degen", j["axis"]["axis_degenerate"])
PY
```
Expected: `OK rows .. 384 sc_hot_frac >=0.5 1084 degen True`。

- [ ] **Step 5: 跑全部 6 + Commit**

Run: `python scripts/run_topic5_ictal_field_dynamics.py`
Then:
```bash
git add scripts/run_topic5_ictal_field_dynamics.py
git commit -m "feat(topic5 field-dynamics): per-window metrics driver (4-group activity/align/drift/sync) + parity gate"
```

---

## Task 6: 两层图 + README + FIGURE_INDEX

**Files:**
- Create: `scripts/plot_topic5_ictal_field_dynamics.py`
- Create: `results/topic5_ictal_recruitment/field_dynamics/figures/README.md`
- Modify: `results/FIGURE_INDEX.md`

**Interfaces:**
- Consumes: Task 5 CSV + per_subject JSON；Task 4 `load_context`；Task 1 cache；`scripts.plot_contact_plane_static._smooth_rank_field_mm`。
- Produces: `figures/per_seizure/<ds>/*.png`（仅 `eeg_duration≥40s` 非 parity_fail）+ `figures/<ds>_progress.png` / `<ds>_offset.png` / `<ds>_seizure_heatmap.png` / `<ds>_geometry_qc.png`。

- [ ] **Step 1: 写绘图脚本（实现）**

```python
"""Topic 5 发作内 field 动力学 — 两层图（per-seizure 详图 + subject-level 聚合）。
复用 field-vs-ictal 的 display frame / 平滑 / 色标逻辑（_smooth_rank_field_mm + viridis）。"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.run_topic5_ictal_field_dynamics import load_context, CACHE, OUT, SUBJECTS, _slice, _zmean_by_name
from scripts.plot_contact_plane_static import _smooth_rank_field_mm

GROUP_COL = {"source_core": "#d62728", "axial_mid": "#ffcc00",
             "non_axial": "#1f77b4", "axis_end_noncore": "#999999"}
PROG_KEYS = [("align_maxab", "场-轴对齐 maxAB"), ("pms_axial_mid", "轴向中段 正质量占比"),
             ("pms_non_axial", "非轴向 正质量占比"), ("sync_median_corr", "同步 median corr")]


def _rank01(v):
    v = np.asarray(v, float); out = np.full(v.shape, np.nan); ok = np.isfinite(v)
    if ok.sum() >= 2:
        out[ok] = np.argsort(np.argsort(v[ok])) / (ok.sum() - 1)
    elif ok.sum() == 1:
        out[ok] = 0.5
    return out


def _field_ax(ax, ctx, zmean_by_name, title):
    frame = ctx["frame"]; xlim, ylim, sigma = frame["xlim"], frame["ylim"], frame["sigma_mm"]
    names = list(ctx["mapped"])
    xs = np.array([ctx["pos"][n][0] for n in names]); ys = np.array([ctx["pos"][n][1] for n in names])
    vals = _rank01([zmean_by_name.get(n, np.nan) for n in names])
    sup = np.ones_like(xs)
    _, _, T, _, _ = _smooth_rank_field_mm(xs, ys, vals, sup, xlim, ylim, sigma)
    ax.imshow(T, origin="lower", extent=[xlim[0], xlim[1], ylim[0], ylim[1]], aspect="equal",
              cmap="viridis", vmin=0, vmax=1)
    g = ctx["part"]["groups"]
    ax.scatter(xs, ys, c=[GROUP_COL.get(g.get(n), "w") for n in names], s=46, edgecolors="k",
               linewidths=0.7, zorder=3)
    PA, PB = ctx["part"]["P_A"], ctx["part"]["P_B"]
    ax.plot([PA[0], PB[0]], [PA[1], PB[1]], "w--", lw=1.6, zorder=4)
    ax.set_title(title, fontsize=10); ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xticks([]); ax.set_yticks([])


def _seizure_windows(ctx, ds_sid, idx, meta, data, cache_names):
    s = meta["seizure"][str(idx)]; off = s["eeg_offset_rel"]
    zt, relt = data[f"bb_zt__{idx}"], data[f"bb_relt__{idx}"]
    snaps = []
    for frac in (0.0, 0.33, 0.66, 1.0):
        lo = max(0.0, frac * off - 5.0)
        zmv, _ = _slice(zt, relt, lo, lo + 10.0)
        if zmv is not None:
            snaps.append((frac, _zmean_by_name(zmv, cache_names, ctx["mapped"])))
    return snaps


def plot_per_seizure(ds_sid, ctx, meta, data, cache_names, rows_by_sz):
    fig_dir = OUT / "figures" / "per_seizure" / ds_sid
    fig_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for idx in meta["eligible_idxs"]:
        s = meta["seizure"][str(idx)]
        if s["eeg_duration_sec"] < 40 or idx not in rows_by_sz:
            continue
        snaps = _seizure_windows(ctx, ds_sid, idx, meta, data, cache_names)
        rs = sorted([r for r in rows_by_sz[idx] if r["window_kind"] == "onset_slide"],
                    key=lambda r: float(r["t_center_rel_onset"]))
        fig, ax = plt.subplots(2, 4, figsize=(17, 8), layout="constrained")
        for j, (frac, zmn) in enumerate(snaps[:4]):
            _field_ax(ax[0, j], ctx, zmn, f"progress {int(frac*100)}%")
        prog = [float(r["progress_frac"]) for r in rs]
        for key, lbl in PROG_KEYS:
            ax[1, 0].plot(prog, [float(r[key]) for r in rs], marker="o", ms=3, label=lbl)
        ax[1, 0].set_title("指标 vs progress"); ax[1, 0].set_xlabel("progress"); ax[1, 0].legend(fontsize=7)
        for key, col in (("pms_source_core", GROUP_COL["source_core"]),
                         ("pms_axial_mid", GROUP_COL["axial_mid"]),
                         ("pms_non_axial", GROUP_COL["non_axial"])):
            ax[1, 1].plot(prog, [float(r[key]) for r in rs], marker="o", ms=3, color=col, label=key)
        ax[1, 1].set_title("正质量占比 vs progress"); ax[1, 1].legend(fontsize=7)
        offs = sorted([r for r in rows_by_sz[idx] if r["window_kind"] == "offset_aligned"
                       and r["pre_onset_overlap"] in ("False", False)],
                      key=lambda r: float(r["t_center_rel_offset"]))
        if offs:
            ax[1, 2].plot([float(r["t_center_rel_offset"]) for r in offs],
                          [float(r["pms_axial_mid"]) for r in offs], "o-", color=GROUP_COL["axial_mid"])
            ax[1, 2].plot([float(r["t_center_rel_offset"]) for r in offs],
                          [float(r["pms_non_axial"]) for r in offs], "o-", color=GROUP_COL["non_axial"])
        ax[1, 2].axvline(0, color="k", ls=":"); ax[1, 2].set_title("offset zoom (rel offset s)")
        ax[1, 3].plot(prog, [float(r["sync_median_corr"]) for r in rs], "o-", color="purple")
        ax[1, 3].set_title("同步 vs progress")
        fig.suptitle(f"{ds_sid} sz{idx} (dur {s['eeg_duration_sec']:.0f}s) — 发作内 field 动力学", fontsize=13)
        fig.savefig(fig_dir / f"{ds_sid}_sz{idx}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
        n += 1
    return n


def plot_subject_level(ds_sid, ctx, rows, subj):
    fdir = OUT / "figures"; fdir.mkdir(parents=True, exist_ok=True)
    onset = [r for r in rows if r["window_kind"] == "onset_slide"]
    by_sz = defaultdict(list)
    for r in onset:
        by_sz[r["seizure_idx"]].append(r)
    # progress summary
    fig, ax = plt.subplots(1, len(PROG_KEYS), figsize=(5 * len(PROG_KEYS), 4), layout="constrained")
    bins = np.linspace(0, 1, 6)
    for k, (key, lbl) in enumerate(PROG_KEYS):
        med_x, med_y = [], []
        for sz, rs in by_sz.items():
            rs = sorted(rs, key=lambda r: float(r["progress_frac"]))
            ax[k].plot([float(r["progress_frac"]) for r in rs], [float(r[key]) for r in rs],
                       color="0.7", lw=0.8)
        for b0, b1 in zip(bins[:-1], bins[1:]):
            vals = [float(r[key]) for r in onset if b0 <= float(r["progress_frac"]) < b1
                    and r[key] not in ("", "nan")]
            if vals:
                med_x.append((b0 + b1) / 2); med_y.append(np.nanmedian(vals))
        ax[k].plot(med_x, med_y, "k-o", lw=2); ax[k].set_title(lbl); ax[k].set_xlabel("progress")
    fig.suptitle(f"{ds_sid} — 发作进程 summary (spaghetti+median; degen={subj['axis']['axis_degenerate']})")
    fig.savefig(fdir / f"{ds_sid}_progress.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    # geometry QC
    fig, axg = plt.subplots(figsize=(7, 6), layout="constrained")
    _field_ax(axg, ctx, {n: 0.0 for n in ctx["mapped"]}, f"{ds_sid} geometry QC")
    axg.set_title(f"{ds_sid} 四分区 + 轴 (uncertA/B={subj['axis']['source_focus_uncertain_a']}/"
                  f"{subj['axis']['source_focus_uncertain_b']}, distA/B="
                  f"{subj['axis']['source_top2_dist_a_mm']:.0f}/{subj['axis']['source_top2_dist_b_mm']:.0f}mm)",
                  fontsize=9)
    fig.savefig(fdir / f"{ds_sid}_geometry_qc.png", dpi=120, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="*", default=SUBJECTS)
    args = ap.parse_args()
    rows_all = list(csv.DictReader(open(OUT / "per_seizure_metrics.csv")))
    for ds_sid in args.subjects:
        if not (CACHE / f"{ds_sid}.npz").exists():
            continue
        ctx = load_context(ds_sid)
        meta = json.load(open(CACHE / f"{ds_sid}.json"))
        data = np.load(CACHE / f"{ds_sid}.npz", allow_pickle=True)
        cache_names = [str(x) for x in data["channels"]]
        subj = json.load(open(OUT / "per_subject" / f"{ds_sid}.json"))
        rows = [r for r in rows_all if r["ds_sid"] == ds_sid]
        rbs = defaultdict(list)
        for r in rows:
            rbs[int(r["seizure_idx"])].append(r)
        npf = plot_per_seizure(ds_sid, ctx, meta, data, cache_names, rbs)
        plot_subject_level(ds_sid, ctx, rows, subj)
        print(f"[fig] {ds_sid}: {npf} per-seizure + subject-level", flush=True)
    print("FIGS DONE", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 生成 384（含 per-seizure，dur≥40s）+ 1084（仅 subject-level）**

Run: `python scripts/plot_topic5_ictal_field_dynamics.py --subjects epilepsiae_384 epilepsiae_1084`
Expected: 打印 `[fig] epilepsiae_384: M per-seizure + subject-level`、`[fig] epilepsiae_1084: 0 per-seizure ...`；生成 `figures/epilepsiae_384_progress.png` 等。

- [ ] **Step 3: 目视检查（用户）+ 断言文件存在**

Run:
```bash
ls -1 results/topic5_ictal_recruitment/field_dynamics/figures/*.png \
      results/topic5_ictal_recruitment/field_dynamics/figures/per_seizure/epilepsiae_384/*.png | head
```
Expected: 存在 `*_progress.png` / `*_geometry_qc.png` 与 per_seizure/epilepsiae_384/ 下若干 png。**用户目视**：场图轴线穿两 source、四分区着色合理、progress 曲线可读。

- [ ] **Step 4: 写 `figures/README.md`（中文，图实际生成后写）**

```markdown
# 发作内 field 动力学 — 图说明

### <ds>_progress.png
每条发作一条灰线、黑线为 median，横轴为发作进程 0→100%。子图：场-轴对齐(maxAB)、轴向中段正质量占比、非轴向正质量占比、同步(median corr)。
**关注点**：轴向中段占比是否随进程下降、非轴向是否上升、同步是否上升、对齐是否塌。

### <ds>_geometry_qc.png
间期态四分区着色（红=source_core，金=轴向中段，蓝=非轴向，灰=贴轴端区）+ 白虚线为 source-A→source-B 轴。
**关注点**：轴线是否穿过两 source、source_core 是否单点（uncertain）、退化轴 subject（1084）轴是否塌缩。

### per_seizure/<ds>/<ds>_szN.png
单次发作：四个进程快照场图（0/33/66/100%）+ 指标随进程 + 正质量占比随进程 + offset zoom + 同步。仅 eeg_duration≥40s 的发作出。
**关注点**：单次发作内 field 是否保持同一轴还是方向漂移、终止前后是否突变。
```

- [ ] **Step 5: 生成全部 6 + append FIGURE_INDEX + Commit**

Run: `python scripts/plot_topic5_ictal_field_dynamics.py`
Then 在 `results/FIGURE_INDEX.md` 追加一行指向 `results/topic5_ictal_recruitment/field_dynamics/figures/`（发作内 field 动力学 pilot，6 ECoG，narrow）。
```bash
git add scripts/plot_topic5_ictal_field_dynamics.py \
        results/topic5_ictal_recruitment/field_dynamics/figures/README.md results/FIGURE_INDEX.md
git commit -m "feat(topic5 field-dynamics): two-layer figures (per-seizure + subject) + README + index"
```

---

## Self-Review（plan vs spec）

- **spec §1 输入/口径** → Task 1（eligibility+complete-interval+offset+baseline 复用）、Task 4（narrow rank-disp+geometry，非 swap-gated）。✓
- **spec §2 长 cache + parity** → Task 1（per-seizure post=dur+90，存 bb_zt/relt+offset_rel）、Task 5 `_parity_fail`。✓
- **spec §3 四分区 source-core compact** → Task 2（`source_core` 15mm gate + 单点 fallback；`axis_partition` 4 组 MECE + 退化）。✓
- **spec §4 指标** → Task 3（positive_mass_share/gradient/synchrony/participation）+ Task 4（maxAB alignment）+ Task 5（装配 + drift fold）。✓
- **spec §5 窗** → Task 5（onset 滑窗 10/5 到 offset + progress + offset 终止窗 + pre_onset_overlap）。✓
- **spec §6 输出** → Task 5（CSV 列 + per_subject JSON）+ Task 6（两层图 + README + FIGURE_INDEX）。✓
- **spec §8 健康检查** → Task 4 Step2（1084 degen）、Task 5 Step4（source_core sanity + 1084 负控 + parity）。✓
- **type 一致性**：`load_context`/`window_maxab`/`_slice`/`_zmean_by_name`/`_metrics_row` 在 Task 4/5 跨步骤签名一致；`GROUPS`/`positive_mass_share` 键四组一致；CSV_COLS ⊇ `_metrics_row` 输出键。✓
- **YAGNI**：无 cohort 统计 / 无 per-window null / 无 clinical offset / 无 Yuquan（spec §9）。✓

> 注（impl 时留意，非 plan 缺陷）：`R_smooth_rank` 接受 raw 活动向量并内部成 rank（与 `bb_auc` 同路径），故 `window_maxab` 传 raw window-mean z，不预先 rank（Global Constraints 已锁）。`_inventory_rows[idx]` 与 `extract_seizure_window` 的 idx 同序（源码 docstring 担保），Task 1 依赖之。

---

## Addendum — P1 review 修复（2026-06-28，实现以本节为准，覆盖上文 Task 1/5/6 代码细节）

用户第二轮 review（4×P1）已应用。Task 2/3（纯模块）不变；Task 1/5/6 按下列修订实现：

**Task 1（已在上文代码内修订）**
- `span = max(eeg_offset_rel, eeg_duration_sec)`；`post_sec = ceil(span)+90`（P1-1：eeg_onset 晚于 clin_onset 时 duration 少抽，384 `eeg_onset_rel≈+36s`）。
- `span > 600s`(`MAX_ICTAL_SEC`) → drop `duration_too_long_for_pilot`（防 status / OOM）。
- meta 写 `baseline={guard_sec:60}`；Step3 验证遍历**全部** eligible_idxs。

**Task 5（band loop + ictal_fraction，覆盖上文 run_subject）**
- `BANDS = ("bb","hfa")`；对每 band 读 `{band}_zt`/`{band}_relt`，每窗 × 每 band 出一行（`band` 列）。P1-4：HFA 真分析非仅缓存。
- 每窗加 `ictal_fraction`（窗内 bin 落 `[0, eeg_offset_rel]` 比例）与 `post_offset_overlap`（窗右缘>offset）。P1-2：短发作 [0,10] 跨 offset。
- `_slice` 仍返回 `(mean, sub)`；新增 `_ictal_fraction(relt, lo, hi, eeg_offset_rel)`。
- drift 按 `(seizure_idx, band)` 分组取各自 onset 首窗参照。
- CSV_COLS 在 `pre_onset_overlap` 后插 `post_offset_overlap, ictal_fraction`。
- Step4 sanity 用 `band=="bb"` ∧ `window_kind=="onset_slide"` ∧ `ictal_fraction>=0.5` 的行；并断言 hfa 行存在。

**Task 6（图，覆盖上文 plot 脚本）**
- per-seizure：**每 seizure 一张 8-panel composite**（合同即如此；trajectory panel 用 `band=="bb"` ∧ `ictal_fraction>=0.5`）。
- subject-level：**4 张** = `progress` / `offset`(rel-offset 窗 median±散点，排除 pre_onset_overlap) / `seizure_heatmap`(行=发作 列=progress bin 值=pms_axial_mid) / `geometry_qc`；删 representative atlas（与 per-seizure snapshots 重复）。
- 所有图默认 `band=="bb"`（`--band hfa` 可选）。README 增 `<ds>_offset.png` / `<ds>_seizure_heatmap.png` 两条。

> 内存/OOM：cache build 单进程顺序跑 6 subject（峰值 ~1 subject arrays + 1 EDF 窗 ≤ ~300MB）；分析/绘图逐 subject 载入；**不并行多 subject 的数据载入**。`MAX_ICTAL_SEC=600` 兜住超长发作。
