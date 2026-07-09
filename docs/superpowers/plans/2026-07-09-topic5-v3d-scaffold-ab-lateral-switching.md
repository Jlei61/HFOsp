# Topic5 V3d — scaffold 上 A/B 侧向切换 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.
> **Plan-of-record spec**: `docs/superpowers/specs/2026-07-09-topic5-v3d-scaffold-ab-lateral-switching-design.md`
> (rev2). 每个 Task 开工前**重读 spec 对应 §**（CLAUDE.md §5 每步边界重读合同）。

**Goal:** 构造固定取向的间期 A/B 对比坐标 `C_AB(t)`，用穷举时间 null 检验发作临近的侧向极化，出三表三图。

**Architecture:** 纯数据侧。核心数值放 `src/topic5_scaffold_ab_contrast.py`（可测函数、raw-contact、无镜像）；一个
producer 批量出表 + npz；三个 plotter 出图。复用现有窗能量/触点匹配/within-shaft-shuffle 基建，不重造。

**Tech Stack:** Python 3, numpy, scipy.stats（binom/zscore/pearsonr），matplotlib（Agg），pytest。

## Global Constraints（每个 Task 隐含包含；数值全部来自 spec §6.1，改需回 spec）

- 窗合同：`START=-120, STOP=+20, WINDOW=10, STEP=2, BAND=(1,150)`；`FAR_PRE=[-120,-60]`,
  `NEAR_ONSET=[-30,+10]`, `NEAR_PRE=[-30,0)`, `EARLY_ICTAL=[0,+10]`。
- **window center 派生（唯一口径）**：`center = window_start + WINDOW/2`；`window_start ∈ {-120,-118,…,+10}`
  （要求 `window_start+WINDOW ≤ STOP`）→ `center ∈ {-115,-113,…,+15}` = `np.arange(-115, 16, 2.0)`，**共 66 窗**
  （`T=66`，非零 shift `T-1=65`）。producer / plotter / **测试**一律从 `window_start + WINDOW/2` 取 center，
  不得手写 `np.arange(-120, 21, 2)`（那是 71 点、错的）。真实 start 来自 `_compute_values` 的 `starts` + `_keep_window`。
- **git staging 纪律**：**禁 `git add -A`**（dirty worktree 会误 stage 别人的改动）、**禁 `git commit -am`**
  （`-a` 漏新建文件）。每 Task commit 前 `git add <该 Task 的 exact 文件>` 再 `git commit -m`。
  `results/` 被 `.gitignore` 忽略 → README 要 track 必须 `git add -f`（已核实：新 README 路径 check-ignore 命中）。
- 门槛：`F_MIN_WIN=0.9, N_JOINT_MIN=6, RHO_DEGEN=0.85, DELTA_SIDE=0.2, ALPHA_PRESENT=0.05,
  N_MULTI_SHAFT_MIN=2, FRAC_SHUFFLE_MIN=0.6, N_PERM=1000, N_VALID_SHIFT_MIN=40, N_VALID_SEIZURE_MIN=3,
  LOCK_ALPHA=0.05`。tier 边界 `rho_reciprocal=-0.5, rho_aligned=0.5, rho_degen=0.85`。
- **rank 极性锁**：`typical_rank` 低=早=源 → earlyness `eA=-zscore(rank_A)`；`C_AB>0`=偏 A 源侧（spec §2.1）。
- **无镜像铁律**：主路径**禁** import 任何 `corr_pair_mirror_invariant*`；`C_AB` 唯一 source of truth =
  `pearson(E_t[finite], D_AB[finite])`（direct，非闭式；spec §2.2/§10）。
- **fail-closed**：`insufficient_joint / hard_degenerate / axis_present_low_dof / insufficient_valid_seizures`
  必须显式落 drop 记录、退出 H1/H2，禁静默默认值（spec §10）。
- **确定性**：所有随机（within-shaft shuffle、subject-level 抽样）走显式 `np.random.default_rng(seed)`。
- 复用（禁重造，CLAUDE.md §6）：`scripts.compute_topic5_signed_broadband_similarity._compute_values/_load_axis`、
  `scripts.plot_topic5_signed_broadband_similarity_timecourse._eligible_idxs/_on_common_grid`、
  `src.topic5_axis_alignment.matched_channels/within_shaft_shuffle`、`src.propagation_skeleton_geometry.parse_shaft`。

---

## File Structure

- Create `src/topic5_scaffold_ab_contrast.py` — 数值核心（Task 1–6 累积）。
- Create `tests/test_topic5_scaffold_ab_contrast.py` — 核心 TDD + §6.4 fixtures（Task 1–7）。
- Create `scripts/run_topic5_scaffold_ab_switching.py` — producer + cohort（Task 8–9）。
- Create `scripts/plot_topic5_scaffold_ab_contrast_timecourse.py` / `..._state_space.py` / `..._cohort_raster.py`
  （Task 10）。
- Create `results/topic5_ictal_recruitment/scaffold_ab_switching/figures/README.md`（Task 11）。
- Modify `docs/topic5_seizure_subtyping.md`（V3p 口径改写，Task 11）。

---

### Task 1: `build_D_AB` + `template_pair_tier`（spec §2.1/§2.3/§4）

**Files:** Create `src/topic5_scaffold_ab_contrast.py`；Test `tests/test_topic5_scaffold_ab_contrast.py`

**Interfaces — Produces:**
- `build_D_AB(rank_a, rank_b) -> dict{eA, eB, D_AB, rho_AB, sd_D_AB}`（arrays + floats）
- `template_pair_tier(rho_AB) -> str` ∈ {`reciprocal`,`oblique`,`aligned`,`hard_degenerate`}

- [ ] **Step 1: 写失败测试**
```python
import numpy as np
from src.topic5_scaffold_ab_contrast import build_D_AB, template_pair_tier

def test_build_D_AB_earlyness_sign():
    # rank 低=早. A 早的触点 (rank 0..) 在 D_AB 上应为正 (eA 大).
    rank_a = np.array([0., 1., 2., 3., 4., 5.])   # contact0 最早 = A 源
    rank_b = rank_a[::-1].copy()                   # B 完全反相
    out = build_D_AB(rank_a, rank_b)
    assert out["D_AB"][0] > 0 and out["D_AB"][-1] < 0        # A 源端 D_AB>0
    assert out["rho_AB"] < -0.99                              # 反相 -> rho≈-1
    # rho_AB 对符号翻转不变: pearson(eA,eB)==pearson(zA,zB)
    zA, zB = -out["eA"], -out["eB"]
    assert abs(out["rho_AB"] - np.corrcoef(zA, zB)[0,1]) < 1e-9

def test_template_pair_tier_boundaries():
    assert template_pair_tier(-0.6) == "reciprocal"
    assert template_pair_tier(-0.5) == "reciprocal"     # <= -0.5
    assert template_pair_tier(0.0)  == "oblique"
    assert template_pair_tier(0.5)  == "aligned"        # [0.5,0.85)
    assert template_pair_tier(0.9)  == "hard_degenerate"
```

- [ ] **Step 2: 跑测试确认失败** — `pytest tests/test_topic5_scaffold_ab_contrast.py -k "build_D_AB or tier" -v`（ImportError）

- [ ] **Step 3: 最小实现**
```python
import numpy as np

RHO_RECIPROCAL, RHO_ALIGNED, RHO_DEGEN = -0.5, 0.5, 0.85

def _zscore(x):
    x = np.asarray(x, float)
    sd = x.std(ddof=0)
    return (x - x.mean()) / sd if sd > 1e-12 else np.zeros_like(x)

def build_D_AB(rank_a, rank_b):
    eA = -_zscore(rank_a)          # 大 = 早 = source-like (rank 低=早)
    eB = -_zscore(rank_b)
    D_AB = eA - eB
    rho_AB = float(np.corrcoef(eA, eB)[0, 1]) if eA.std() > 1e-12 and eB.std() > 1e-12 else 1.0
    return {"eA": eA, "eB": eB, "D_AB": D_AB, "rho_AB": rho_AB, "sd_D_AB": float(D_AB.std(ddof=0))}

def template_pair_tier(rho_AB):
    if rho_AB <= RHO_RECIPROCAL: return "reciprocal"
    if rho_AB < RHO_ALIGNED:     return "oblique"
    if rho_AB < RHO_DEGEN:       return "aligned"
    return "hard_degenerate"
```

- [ ] **Step 4: 跑测试确认通过** — 同 Step 2 命令，Expected PASS

- [ ] **Step 5: 提交** — `git add src/topic5_scaffold_ab_contrast.py tests/test_topic5_scaffold_ab_contrast.py && git commit -m "feat(topic5-v3d): build_D_AB earlyness contrast + pair tier"`

---

### Task 2: `derive_joint_contacts`（spec §2.1 joint set + hard_degenerate 守卫）

**Files:** Modify `src/topic5_scaffold_ab_contrast.py`；Test 同文件

**Interfaces — Consumes:** `build_D_AB`, `template_pair_tier`（Task 1）; `matched`（`matched_channels` 输出：dict 带
`name`/`typical_rank`/`x_norm`/`y_norm`/`support`）; `axis_b`（`_load_axis` 的 t_b JSON）; `window_vals`（(n_win,
n_matched)）。 **Produces:** `derive_joint_contacts(matched, axis_b, window_vals, f_min_win=0.9, n_joint_min=6)
-> dict{status, names, idx, rank_a, rank_b, D_AB, eA, eB, rho_AB, tier, n_joint}`（`status` ∈
`ok|insufficient_joint|hard_degenerate`）。

- [ ] **Step 1: 写失败测试**
```python
from src.topic5_scaffold_ab_contrast import derive_joint_contacts

def _mk_matched(names, ranks):
    return [{"name": n, "typical_rank": r, "x_norm": i*0.1, "y_norm": 0.0, "support": 1.0}
            for i, (n, r) in enumerate(zip(names, ranks))]

def test_joint_requires_finite_in_A_B_and_windows():
    names = [f"A{i}-A{i+1}" for i in range(6)]
    matched = _mk_matched(names, [0,1,2,3,4,5])
    axis_b = {"channels": [{"name": n, "typical_rank": 5-i} for i,n in enumerate(names)]}
    wv = np.random.default_rng(0).normal(size=(10, 6))
    out = derive_joint_contacts(matched, axis_b, wv)
    assert out["status"] == "ok" and out["n_joint"] == 6 and out["tier"] == "reciprocal"

def test_joint_insufficient_when_lt_6():
    names = [f"A{i}-A{i+1}" for i in range(4)]
    matched = _mk_matched(names, [0,1,2,3])
    axis_b = {"channels": [{"name": n, "typical_rank": 3-i} for i,n in enumerate(names)]}
    out = derive_joint_contacts(matched, axis_b, np.zeros((10,4)))
    assert out["status"] == "insufficient_joint"

def test_joint_hard_degenerate_when_templates_identical():
    names = [f"A{i}-A{i+1}" for i in range(6)]
    matched = _mk_matched(names, [0,1,2,3,4,5])
    axis_b = {"channels": [{"name": n, "typical_rank": i} for i,n in enumerate(names)]}  # B==A
    out = derive_joint_contacts(matched, axis_b, np.random.default_rng(1).normal(size=(10,6)))
    assert out["status"] == "hard_degenerate"
```

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 最小实现**（要点：joint = A rank 有限 ∩ B rank 有限 ∩ 窗内 z 有限比例 ≥ `f_min_win`；`n_joint<6`→
  `insufficient_joint`；`tier=='hard_degenerate'`→`hard_degenerate`。用 `name` 对 B rank，禁按顺序对齐——
  spec Cross-PR `channel_names` ordering。）
```python
def derive_joint_contacts(matched, axis_b, window_vals, f_min_win=0.9, n_joint_min=6):
    b_rank = {c["name"]: float(c.get("typical_rank", np.nan)) for c in axis_b.get("channels", [])}
    wv = np.asarray(window_vals, float)
    finite_frac = np.isfinite(wv).mean(axis=0)               # per matched contact
    idx, names, ra, rb = [], [], [], []
    for i, c in enumerate(matched):
        rbi = b_rank.get(c["name"], np.nan)
        if np.isfinite(c["typical_rank"]) and np.isfinite(rbi) and finite_frac[i] >= f_min_win:
            idx.append(i); names.append(c["name"]); ra.append(c["typical_rank"]); rb.append(rbi)
    if len(idx) < n_joint_min:
        return {"status": "insufficient_joint", "n_joint": len(idx), "names": names}
    d = build_D_AB(np.array(ra), np.array(rb)); tier = template_pair_tier(d["rho_AB"])
    status = "hard_degenerate" if tier == "hard_degenerate" else "ok"
    return {"status": status, "names": names, "idx": np.array(idx, int),
            "rank_a": np.array(ra), "rank_b": np.array(rb), **d, "tier": tier, "n_joint": len(idx)}
```

- [ ] **Step 4: 跑测试确认通过**
- [ ] **Step 5: 提交** — `git add src/topic5_scaffold_ab_contrast.py tests/test_topic5_scaffold_ab_contrast.py && git commit -m "feat(topic5-v3d): derive_joint_contacts + fail-closed guards"`

---

### Task 3: `contrast_timecourse`（spec §2.2 direct corr，无镜像）

**Files:** Modify core + test. **Interfaces — Produces:** `contrast_timecourse(window_vals_joint, D_AB, eA, eB)
-> dict{C_AB, r_A, r_B, maxAB}`（各 (n_win,)，每窗在有限触点子集上 direct Pearson；`sd(E_w)<1e-9` 或有限触点
`<3` → NaN）。

- [ ] **Step 1: 写失败测试**（含 P2b：缺触点窗只测 direct，闭式只在 full-finite 测等价）
```python
from src.topic5_scaffold_ab_contrast import contrast_timecourse, build_D_AB

def test_contrast_direct_is_source_of_truth():
    rng = np.random.default_rng(0)
    ranks_a = np.arange(8.0); ranks_b = ranks_a[::-1].copy()
    d = build_D_AB(ranks_a, ranks_b)
    E = rng.normal(size=(5, 8))
    out = contrast_timecourse(E, d["D_AB"], d["eA"], d["eB"])
    for w in range(5):
        assert abs(out["C_AB"][w] - np.corrcoef(E[w], d["D_AB"])[0,1]) < 1e-9   # == direct corr

def test_closed_form_only_on_full_finite():
    ranks_a = np.arange(8.0); ranks_b = ranks_a[::-1].copy()
    d = build_D_AB(ranks_a, ranks_b); E = np.random.default_rng(1).normal(size=(1,8))
    o = contrast_timecourse(E, d["D_AB"], d["eA"], d["eB"])
    rho = d["rho_AB"]; closed = (o["r_A"][0]-o["r_B"][0])/np.sqrt(2*(1-rho))
    assert abs(o["C_AB"][0] - closed) < 1e-9                 # full-finite: 闭式成立

def test_partial_window_uses_direct_not_closed():
    ranks_a = np.arange(8.0); ranks_b = ranks_a[::-1].copy()
    d = build_D_AB(ranks_a, ranks_b); E = np.random.default_rng(2).normal(size=(1,8))
    E[0, 3] = np.nan                                          # 缺一个触点
    o = contrast_timecourse(E, d["D_AB"], d["eA"], d["eB"])
    m = np.isfinite(E[0])
    assert abs(o["C_AB"][0] - np.corrcoef(E[0,m], d["D_AB"][m])[0,1]) < 1e-9
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 最小实现**
```python
def _pear(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3: return np.nan
    x, y = a[m]-a[m].mean(), b[m]-b[m].mean()
    dn = np.sqrt((x*x).sum()*(y*y).sum())
    return float((x*y).sum()/dn) if dn > 1e-12 else np.nan

def contrast_timecourse(window_vals_joint, D_AB, eA, eB):
    E = np.asarray(window_vals_joint, float)
    C = np.array([_pear(E[w], D_AB) for w in range(E.shape[0])])
    rA = np.array([_pear(E[w], eA) for w in range(E.shape[0])])
    rB = np.array([_pear(E[w], eB) for w in range(E.shape[0])])
    return {"C_AB": C, "r_A": rA, "r_B": rB, "maxAB": np.maximum(np.abs(rA), np.abs(rB))}
```

- [ ] **Step 4: 跑测试确认通过**
- [ ] **Step 5: 提交** — `git add src/topic5_scaffold_ab_contrast.py tests/test_topic5_scaffold_ab_contrast.py && git commit -m "feat(topic5-v3d): contrast_timecourse direct corr (no mirror)"`

---

### Task 4: `axis_present`（spec §2.4 within-shaft null + low_dof 降级）

**Files:** Modify core + test. **Interfaces — Consumes:** `within_shaft_shuffle`, `parse_shaft`.
**Produces:** `axis_present(window_vals_joint, names_joint, eA, eB, rng, n_perm=1000, alpha=0.05,
n_multi_shaft_min=2, frac_shuffle_min=0.6) -> dict{present(bool arr), within_shaft_p(arr), testable(bool),
low_dof(bool), qc{n_contacts_shuffled, fraction_contacts_shuffled, n_singleton_contacts, n_shafts}}`。

- [ ] **Step 1: 写失败测试**（真信号过 null；单触点为主 → low_dof）
```python
from src.topic5_scaffold_ab_contrast import axis_present, build_D_AB

def test_axis_present_true_when_energy_matches_template():
    ranks_a = np.arange(8.0); ranks_b = ranks_a[::-1].copy(); d = build_D_AB(ranks_a, ranks_b)
    names = [f"S{i//2}-{i}" for i in range(8)]                # 4 根杆各 2 触点
    E = np.tile(d["eA"], (6,1)) + np.random.default_rng(0).normal(scale=0.05, size=(6,8))
    out = axis_present(E, names, d["eA"], d["eB"], np.random.default_rng(0))
    assert out["testable"] and out["present"].mean() > 0.5

def test_axis_present_low_dof_when_mostly_singletons():
    ranks_a = np.arange(8.0); ranks_b = ranks_a[::-1].copy(); d = build_D_AB(ranks_a, ranks_b)
    names = [f"Q{i}-{i}" for i in range(8)]                   # 8 根单触点杆
    E = np.random.default_rng(1).normal(size=(6,8))
    out = axis_present(E, names, d["eA"], d["eB"], np.random.default_rng(1))
    assert out["low_dof"] and not out["testable"]
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 最小实现**（要点：obs 和 null 都在**全 joint 触点**上算 maxAB（Task 3 的 r_A/r_B 逻辑）；
  null 用 `within_shaft_shuffle(vals, names, rng)` 每窗 `n_perm` 次；pointwise `p=(1+#{null≥obs})/(n_perm+1)`；
  QC: 多触点杆上的触点占比 `< frac_shuffle_min` 或 多触点杆 `< n_multi_shaft_min` → `low_dof=True, testable=False`。）
  单元计算复用 Task 3 的 `_pear`；`present = within_shaft_p < alpha`。

- [ ] **Step 4: 跑测试确认通过**
- [ ] **Step 5: 提交** — `git add src/topic5_scaffold_ab_contrast.py tests/test_topic5_scaffold_ab_contrast.py && git commit -m "feat(topic5-v3d): axis_present within-shaft null + low_dof gate"`

---

### Task 5: `label_sides` + `classify_event` + `locking_statistic`（spec §3 H2 taxonomy / §6.2）

**Files:** Modify core + test. **Interfaces — Produces:**
- `label_sides(C_AB, present, delta_side=0.2) -> np.ndarray[str]`（`A`/`B`/`unlabeled`）
- `locking_statistic(C_AB, present, centers, far_pre, near_onset) -> dict{polar_far, polar_near, locking}`
  （`polar=|mean over present∧window C_AB|`；任一侧 present 窗 `<3` → NaN）
- `classify_event(C_AB, present, centers, far_pre, near_onset, near_pre, early_ictal, delta_side) ->
  dict{far_side, near_side, event_class, polar_near_pre, polar_early_ictal}`
  （`event_class` ∈ selection|switch|persistent|none，按 spec §3 H2）

- [ ] **Step 1: 写失败测试**（承载 spec §6.4 static/ramp 语义）
```python
from src.topic5_scaffold_ab_contrast import locking_statistic, classify_event
C_centers = np.arange(-115, 16, 2.0)              # window_start+WINDOW/2, 66 窗中心（-115..+15）
present = np.ones_like(C_centers, bool)

def test_static_gives_zero_locking():
    C = np.full_like(C_centers, 0.7)               # 恒定偏 A
    out = locking_statistic(C, present, C_centers, (-120,-60), (-30,10))
    assert abs(out["locking"]) < 1e-9              # near-far = 0
    ev = classify_event(C, present, C_centers, (-120,-60), (-30,10), (-30,0), (0,10), 0.2)
    assert ev["event_class"] == "persistent"

def test_ramp_gives_positive_locking_and_selection():
    C = np.clip((C_centers+30)/40*0.8, 0, 0.8)     # far≈0 -> near +0.8
    out = locking_statistic(C, present, C_centers, (-120,-60), (-30,10))
    assert out["locking"] > 0.3
    ev = classify_event(C, present, C_centers, (-120,-60), (-30,10), (-30,0), (0,10), 0.2)
    assert ev["event_class"] == "selection"        # far none -> near A

def test_switch_when_sign_flips():
    C = np.where(C_centers < -30, -0.6, 0.6)
    ev = classify_event(C, present, C_centers, (-120,-60), (-30,10), (-30,0), (0,10), 0.2)
    assert ev["event_class"] == "switch"
```

- [ ] **Step 2/3/4:** 跑失败 → 实现（`polar=abs(mean)`；`far_side=sign(mean_far) if |mean_far|>=delta else none`；
  taxonomy 按 spec §3）→ 跑通过。
- [ ] **Step 5: 提交** — `git add src/topic5_scaffold_ab_contrast.py tests/test_topic5_scaffold_ab_contrast.py && git commit -m "feat(topic5-v3d): side label + event taxonomy + locking stat"`

---

### Task 6: 时间 null —— per-seizure 穷举 + subject-level 抽样（spec §5 P2a / §6.3）

**Files:** Modify core + test. **Interfaces — Produces:**
- `circular_shift_null_seizure(C_AB, present, centers, far_pre, near_onset, n_valid_shift_min=40)
  -> dict{locking_obs, valid_shift_lockings(1d arr), locking_shift_p, n_valid_shift, status}`
  （穷举 `shift∈{1..T-1}`；每个 shift 把 `(C_AB, present)` **同偏移**环移后重算 locking；shift 后任一侧 present
  `<3` → 该 shift invalid；`n_valid_shift<40` → `status='insufficient'`）
- `subject_locking_null(per_seizure(list of dict from above), n_perm=1000, seed=0)
  -> dict{L_obs, L_null_p95, subject_locked, p, n_valid_seizures}`

- [ ] **Step 1: 写失败测试**（穷举计数 + 静态不显著 + 组合抽样）
```python
from src.topic5_scaffold_ab_contrast import circular_shift_null_seizure, subject_locking_null
centers = np.arange(-115, 16, 2.0); present = np.ones_like(centers, bool)   # 66 窗 -> T-1=65

def test_enumeration_count_is_T_minus_1():
    C = np.clip((centers+30)/40*0.8, 0, 0.8)
    out = circular_shift_null_seizure(C, present, centers, (-120,-60), (-30,10))
    assert out["n_valid_shift"] <= centers.size - 1          # 穷举，非抽样；centers.size-1 == 65
    assert out["valid_shift_lockings"].ndim == 1

def test_static_not_significant():
    C = np.full_like(centers, 0.7)
    s = circular_shift_null_seizure(C, present, centers, (-120,-60), (-30,10))
    assert s["locking_shift_p"] > 0.5                        # 恒定 -> 所有 shift 同 locking

def test_subject_null_combines_seizures():
    C = np.clip((centers+30)/40*0.8, 0, 0.8)
    seiz = [circular_shift_null_seizure(C, present, centers, (-120,-60), (-30,10)) for _ in range(3)]
    out = subject_locking_null(seiz, n_perm=1000, seed=0)
    assert out["n_valid_seizures"] == 3 and out["subject_locked"] in (True, False)
```

- [ ] **Step 2/3/4:** 跑失败 → 实现（per-seizure：`np.roll((C,present), shift)`；invalid drop；
  `p=(1+#{valid locking≥obs})/(n_valid+1)`。subject：`L_obs=median(locking_obs)`；`n_perm` 次每 seizure
  `rng.choice(valid_shift_lockings)` 取 median → `L_null`；`subject_locked = L_obs > percentile(L_null,95)`）→ 跑通。
- [ ] **Step 5: 提交** — `git add src/topic5_scaffold_ab_contrast.py tests/test_topic5_scaffold_ab_contrast.py && git commit -m "feat(topic5-v3d): enumerated per-seizure time null + subject combinatorial null"`

---

### Task 7: 五个坏数据回归 fixtures（spec §6.4）——端到端保证结论被编码

**Files:** Modify test only（把 §6.4 五个 fixture 串成端到端 assert，复用 Task 1–6）。

- [ ] **Step 1: 写测试**（`flat_noise`→axis_present 多 False；`static_on_axis`(E=3·eA)→locking≈0 & H1 不显著 &
  event=persistent；`ramp_to_onset`→H1 显著；`degenerate_AB`→`derive_joint_contacts.status=='hard_degenerate'`；
  `mirror_invariance_gone`→触点 y 翻转后 `C_AB` 数值不变）
- [ ] **Step 2: 跑测试** — 应全绿（Task 1–6 已实现）；任何红 = 回对应 Task 修
- [ ] **Step 3: 提交** — `git add tests/test_topic5_scaffold_ab_contrast.py && git commit -m "test(topic5-v3d): §6.4 bad-data regressions encode the conclusions"`

---

### Task 8: producer `run_topic5_scaffold_ab_switching.py`（单被试 → 表1/2/3 + npz；spec §7/§10）

**Files:** Create `scripts/run_topic5_scaffold_ab_switching.py`；输出到
`results/topic5_ictal_recruitment/scaffold_ab_switching/{per_subject/,}`。

**Interfaces — Consumes:** `_compute_values`(逐窗 z + matched + centers)、`_load_axis`、`_eligible_idxs`、
Task 1–6 全部核心函数。**Produces:** `<ds_sid>_scaffold_ab_{per_window.csv, per_seizure.csv, summary.json,
matrices.npz}`（列/字段严格按 spec §7 表1/2/3）。

- [ ] **Step 1:** 写单被试流程：遍历 `_eligible_idxs`，每 seizure 走 `_compute_values`→`derive_joint_contacts`
  （`status!=ok` 落 drop、跳过）→ `contrast_timecourse` + `axis_present` + `label_sides` + `classify_event` +
  `locking_statistic` + `circular_shift_null_seizure`；聚合 `subject_locking_null` + H1 前置门（spec §6.3）+ tier。
- [ ] **Step 2: verify（真实单被试）** — `python scripts/run_topic5_scaffold_ab_switching.py --subject epilepsiae_1146`
  Expected: 三表 + summary 落地；`python -c "import json;d=json.load(open('results/topic5_ictal_recruitment/scaffold_ab_switching/per_subject/epilepsiae_1146_scaffold_ab_summary.json'));print(d['template_pair_tier'],d['H1_eligible'],d['subject_locked'],d['n_valid_seizures'])"`
  逐字段核对 spec §7 表3 存在、`tier` 属四档之一、drop 记录不为静默默认。
- [ ] **Step 3: 提交** — `git add scripts/run_topic5_scaffold_ab_switching.py && git commit -m "feat(topic5-v3d): per-subject producer (tables 1/2/3 + npz)"`

---

### Task 9: cohort 批量 + subject-count 二项（spec §4/§6.3）

**Files:** Modify producer（加 `--all-ok` 复用 `_ok_subjects` 式索引 + `cohort_summary.json`）。

- [ ] **Step 1:** 批量 fail-closed（每被试 try/except 落 drop）；cohort 汇总 `k/m` + `scipy.stats.binomtest(k, m,
  0.05, alternative='greater')` + exact CI；按 §4 两轴分层计数（reciprocal/oblique/aligned/hard_degenerate ×
  fine/coarse-only/untestable/low_dof）+ selection/switch/persistent 计数。
- [ ] **Step 2: verify** — `python scripts/run_topic5_scaffold_ab_switching.py --all-ok`
  Expected: `cohort_summary.json` 有 `k,m,binom_p,binom_ci` + 分层计数；打印 `k/m`。核对 `m` 只含 H1-eligible、
  未把 seizure 当独立样本 pool。
- [ ] **Step 3: 提交** — `git add scripts/run_topic5_scaffold_ab_switching.py && git commit -m "feat(topic5-v3d): cohort batch + subject-count binomial"`

---

### Task 10: 三张图（spec §8；CLAUDE.md §7 一图一问题；render→eyeball→改再 commit）

**Files:** Create `scripts/plot_topic5_scaffold_ab_contrast_timecourse.py` / `..._state_space.py` /
`..._cohort_raster.py`。读 `docs/figure_style_guide.md`。

- [ ] **Step 1:** 实现三图（**图1** per-seizure 细线 + 主侧对齐 median + axis_present 底纹 + null/locking inset，
  spec §8 P9；**图2** x=`C_AB`,y=scaffold-QC，色=时间，标注"描述非独立证据"，spec §8 P8；**图3** cohort raster
  行=被试按分层排序、色=`C_AB`、门控外置灰）。
- [ ] **Step 2: verify（渲染 + 目视）** — 对 **E1146 + E922** 跑图1/2、cohort 跑图3；**人工目视**：固定取向后切换/极化
  形态是否真实（非镜像翻面）、图1 median 未被抵消、图2 未暗示 y⊥x。**目视不过就改，不 commit**（feedback:
  render→eyeball→fix 再 commit）。
- [ ] **Step 3: 提交** — `git add scripts/plot_topic5_scaffold_ab_contrast_timecourse.py scripts/plot_topic5_scaffold_ab_state_space.py scripts/plot_topic5_scaffold_ab_cohort_raster.py && git commit -m "feat(topic5-v3d): C_AB timecourse + state-space + cohort raster figures"`

---

### Task 11: figures/README.md + V3p 口径改写（spec §1.1/§7）

**Files:** Create `results/topic5_ictal_recruitment/scaffold_ab_switching/figures/README.md`（中文，逐图"展示什么/
关注点"，图**实际生成后**写）；Modify `docs/topic5_seizure_subtyping.md`（把 V3p 结论改写成 spec §1.1 口径：
"没有轴向→离轴单调重组；能量多数仍在 scaffold 上，部分患者 A/B 侧向选择/切换"，附 V3d 链接）。
**默认只改 repo docs + README，不动 memory**（memory 仅在用户明确要求时更新；审阅/执行 plan 不等于授权 memory
写入）。若用户明确要求，再单独写 `project_topic5_v3d_...` 的 "executed / 结论 k/m" note。

- [ ] **Step 1:** 写 README（按 AGENTS.md Results Standards 格式：`### filename` + 2–4 句 + `**关注点**：`）。
- [ ] **Step 2: verify** — README 每张图文件真实存在；topic5 主文档 V3p 段已改、未 dangling V3d 链接。
- [ ] **Step 3: 提交** — `git add -f results/topic5_ictal_recruitment/scaffold_ab_switching/figures/README.md && git add docs/topic5_seizure_subtyping.md && git commit -m "docs(topic5-v3d): figures README + V3p 口径改写"`（README 被 `.gitignore` 忽略，必须 `-f`）

---

## Self-Review（spec 覆盖 / 占位 / 类型一致）

- **Spec 覆盖**：§2.1→T1/T2、§2.2→T3、§2.3/§4→T1/T2、§2.4→T4、§3→T5、§5/§6.2/§6.3→T5/T6、§6.4→T7、
  §7→T8/T9、§8→T10、§1.1/README→T11、§10 fail-closed→T2/T8/T9、§11.1 abstract→producer summary caveats(T8)。无缺口。
- **占位**：无 TBD；数值全走 Global Constraints（spec §6.1）。图任务是"spec+render→eyeball"而非盲写像素，符合
  repo figure 纪律（feedback_figure_self_contained）——不是占位，是可执行 verify（渲染 + 目视）。
- **类型一致**：`build_D_AB`(T1) 返回 `eA/eB/D_AB/rho_AB` 被 T2/T3/T4 一致消费；`axis_present.present`(T4) 被
  T5/T6 的 `present` 消费；`circular_shift_null_seizure`(T6) 输出被 `subject_locking_null`(T6) 消费；producer(T8)
  消费全部核心函数名一致。

## Execution Handoff

计划已存 `docs/superpowers/plans/2026-07-09-topic5-v3d-scaffold-ab-lateral-switching.md`。两种执行方式：
1. **Subagent-Driven（推荐）** — 每 Task 派 fresh subagent，Task 间人审，快迭代。
2. **Inline Execution** — 本会话按 executing-plans 批量执行 + checkpoint。

Task 1–7 是可测数值核心（TDD 全绿才进 T8）；T8 起接真实数据 + 目视，是 go/no-go 关口（"去镜像后切换还在不在"）。
