# Stage 3 局部→全局事件层级定量 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 cm-SNN 两端等强病灶单网络自发记录里已经分好型的 621 个事件（local 377 / collision 187 / readable_global 54 / readable_unknown_source 3），从"为什么大多数是局部的"这个角度定量到位 —— 出正式分源图 + 效应量，并对 54 个可读全局事件做一次**描述性**模板复核（不是 Stage 3 主问的复活）。

**Architecture:** 只读已落盘 artifact（`event_types.csv` / `readable_global_events.csv` / `record/<tag>/*_lagPat_withFreqCent.npz` / `sidecar_*.json`），新增两个独立脚本，不改任何现有 runner。模板复核复用 `pool_and_cluster_spontaneous.py` 已验证的 **masked** PR-2 / rank-displacement 机器（`compute_adaptive_cluster_stereotypy(..., use_masked_features=True)` + `compute_swap_score_sweep`），只是把输入从"整条记录"换成"被标成 readable_global 的那几列事件"。

**Tech Stack:** Python / numpy / pandas / matplotlib；`src.interictal_propagation`（masked KMeans 路径）；`src.rank_displacement`（swap sweep）；`src.sef_hfo_stage3.build_sidecar` 的列对齐契约（`event_id == lagPat 列号`）。

---

## §0 朴素导读（先读这段）

**测了什么** —— 一张厘米尺度脉冲网络，两端各放一块"容易自己点着的病灶组织"，只靠背景噪声让它们自发放电。我们已经把 621 次自发群体放电逐个贴了标签：是只在一头点着、传不开的小事件（local），还是传开到够多虚拟电极触点、能读出方向的大事件（readable_global），还是两头同时点着、读不出谁先谁后的事件（collision）。这一步要回答的不是"双向行波存不存在"，而是更基础的一问：**这些事件分成"小局部"和"少数可读大"两档，这两档到底差在哪 —— 差在点火能量，还是差在传得远不远、持续得久不久？**

**怎么测的** —— ① 把"小局部"和"少数可读大"两档放在一起比四件事：持续多久（duration）、点亮几个触点（n_part）、起点那块核点得多旺（source_core_ignite_frac）、读出来的方向跟起点端一不一致（sign）。**关键纪律：必须按"哪头先点着"（neg / pos）分开比** —— 因为合在一起看像"点火能量两档一样"，分开看其实两头不对称（负端起的小事件点得比大事件还旺，正端起的反过来）。② 对那 54 个可读大事件，先核对每个事件能不能精确对回它在记录文件里的那一列，再用**和真实病人完全相同的 masked 聚类**看它们能不能聚成"两套相反模板"—— 但这只当**描述性复核**，不当 Stage 3 主问的答案。

**揭示了什么（预期表述口径）** —— 区分小局部和可读大事件的是**传播（持续时间 + 扩散范围），不是成核能量**；小局部事件是"点得很旺却传不远"的 contained / relay-failure，是机制信息不是失败。**不能说**已经证明"波前先走后停被截断"（那要事件级时空快照）；只能说"点着了但没扩散开"。模板复核结果只分三档写：可复核 / 数据太少 / 被源-方向歧义限制。

（内部归档代号：`twoend_equal` / `propagation_class` ∈ {local, collision, readable_global, readable_unknown_source} / `source` ∈ {neg,pos,both,none} / `source_core_ignite_frac` / `swap_class` / `decision_k` / masked PR-2 = `compute_adaptive_cluster_stereotypy(use_masked_features=True)`）

---

## §A 审阅锁定合同（这一步的硬约束 —— 必须逐条满足）

审阅报告（2026-06-14）核过当前 artifact，下列数字已用 `event_types.csv` 复核一致，作为合同冻结：

- **artifact 真值**：local **377** / collision **187** / readable_global **54** / readable_unknown_source **3**（共 621）。字段名是真实的 `propagation_class` / `source`（**不是** `event_type` / `source_label`）。
- **C1（P1-1，分源是硬要求）**：不得只用 pooled 点火强度支持"local≈global"。分源后是**不对称**的（已复核）：
  - `source=neg`：local source-core median **0.366** (n=152) > global **0.315** (n=35)
  - `source=pos`：local **0.278** (n=195) < global **0.334** (n=19)
  - 允许的结论是"local 不是弱点火"，**禁止**写成"所有源上点火能量完全相等"。Step 2 图**必须按 `source=neg/pos` 分面**，每组报告 median + bootstrap 95% CI + permutation 效应量。
- **C2（P1-2，54 不是一个 homogeneous subject）**：54 个 readable_global 跨 cell / seed / source，且 pos 源方向读出 **10 forward / 9 reverse**（≈抛硬币），neg 源 **31/35 forward**（89%）。模板复核必须三层输出：**(a) 全池化仅作 sanity**；**(b) 按 source 分层**；**(c) 按 top cell 分层**。`swap_class` 只能叫"**描述性模板复核**"，**不能**叫 Stage 3 证据 / 主问的答案。
- **C3（P1-3，Step 4 重跑门要硬）**：可选重跑前设硬标准 —— **每个 source ≥ 20 readable_global 且每个 source 跨 ≥ 2 seeds**。已复核：**当前没有任何 cell 达标**（最好的 `sep0.7/std1.0/m17.5` 也只有 neg 10（3 seeds）/ pos 5（3 seeds））。所以 **Step 4 默认不开**，只有 Step 3 明确显示"事件数不足但形态有信号"才讨论。
- **C4（§4 科学边界）**：可以说 contained / relay-failure（点火强、不扩散）；**不能**说已证明"波前截断 / 先走后停"（需事件级时空快照）。解释里要带 baseline 对照：Stage 2 单灶全局事件、oneend sign calibration、hot/collision 条件。
- **C5（§5 工程）**：Step 3 必须有硬 assert（列契约 + `event_id→lagPat 列`映射 + masked 路径）；归档数字 `within_recovery` **1/377**（不是 doc 里的 0/379）要同步。
- **C6（不做的）**：这一步**不是** formal Stage 3 复活，**不**自动开 Step 4 仿真，**不**改任何现有 runner。

**Round-2 审阅补丁（2026-06-14，已逐条复核，并入合同）：**
- **C7（P1-1 record 定位）**：record 必须经 **sidecar→同级 `dirname(sidecar)/record/<tag>/`** 唯一解析，**不**全局 glob 取第一个。复核：`ROOT/record` 直挂仅 2/21；全局递归命中 21/21 但 2 个 tag（`gs_te_sep0.6_std1.0_m17.0_s1`/`gs_te_sep0.7_std1.0_m17.5_s1`）有重复 record；sidecar→sibling 唯一解析 21/21。见 Task 3。
- **C8（P1-2 duration 不是 stale）**：归档 line 85 的 `local 24 ms` 是 `source∈{neg,pos}, n=347` 的正确值（all-local 才是 23 ms）。Task 1 **只**改 `within_recovery` 0/379→1/377，duration 不动。
- **C9（P1-3 形态多样性闸）**：每个聚类层必须落 `n_unique_masked_patterns`（masked 特征 distinct 行数）。复核：pooled 13/54、neg 6/35、pos 7/19。`diversity_limited`（<10）层的 `stable_k` 不得读成"找到稳定双模板"——这是挡 stable_k 过度解释的硬闸。见 Task 4。

---

## §B 文件结构

- **Modify:** `docs/archive/topic4/sef_hfo/stage3_regime_screen_2026-06-14.md` —— 同步两个 stale 数字（C5）。
- **Create:** `scripts/plot_stage3_local_global_hierarchy.py` —— Step 2 分源层级图 + 效应量（C1）。
- **Create:** `scripts/analyze_stage3_readable_templates.py` —— Step 3 readable-template dry-run + masked 三层聚类 + 三档判定（C2）。
- **Create/Update:** `results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/figures/README.md` —— 新增图的中文逐图说明（AGENTS.md 规范）。
- **Append:** `docs/archive/topic4/sef_hfo/stage3_regime_screen_2026-06-14.md` —— Step 2/3 结果段（三档口径，C2/C4）。

---

## Task 1: 同步归档 stale 数字（C5，先做，最小）

**Files:**
- Modify: `docs/archive/topic4/sef_hfo/stage3_regime_screen_2026-06-14.md`

- [ ] **Step 1: 复核真值（含 duration 分母辨析，P1-2）**

Run:
```bash
python3 - <<'PY'
import pandas as pd
d = pd.read_csv("results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/event_types.csv")
loc = d[d.propagation_class=="local"]
print("local n:", len(loc), "within_recovery True:", int((loc.within_recovery.astype(str).isin(["True","true"])).sum()))
print("all-local duration median:", loc.duration.median())
print("source-in{neg,pos} local duration median:", loc[loc.source.isin(['neg','pos'])].duration.median(),
      " (这是 line 84/85 用的分母, n=347)")
PY
```
Expected: `local n: 377 within_recovery True: 1`；`all-local duration median: 23.0`；`source-in{neg,pos} local duration median: 24.0`。

**P1-2 辨析（已复核）**：归档 line 84 上一句的分母是 `source∈{neg,pos}, n=347`，对应 local duration median = **24 ms** —— 所以 line 85 的 `24 ms` **不是 stale**，是 source-restricted 的正确值（all-local 才是 23 ms）。**因此 duration 一个字都不改。**

- [ ] **Step 2: 改文档（只改一个数字）**

在 `stage3_regime_screen_2026-06-14.md` 把 line 86 的
`0/379 local events fall within 250 ms` 改成 `1/377 local events fall within 250 ms`（结论不变：refractory-shadow 不是主因）。
**只改这一个数字。line 85 的 `24 ms` 保持不动**（如担心歧义，可在其后括注 `(source∈{neg,pos}, n=347; all-local=23 ms)`，但不强制）。

- [ ] **Step 3: Commit**

```bash
git add docs/archive/topic4/sef_hfo/stage3_regime_screen_2026-06-14.md
git commit -m "docs(stage3): sync stale within_recovery 0/379->1/377 (duration 24ms is correct, source-restricted)"
```

---

## Task 2: Step 2 — 分源局部↔全局层级图 + 效应量（C1）

**Files:**
- Create: `scripts/plot_stage3_local_global_hierarchy.py`
- Input: `results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/event_types.csv`
- Output fig: `results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/figures/stage3_local_global_hierarchy.png`
- Output stats: `results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/stage3_hierarchy_effect_sizes.json`

**四个面板 = 四个独立科学问题（CLAUDE.md §7，禁止冗余）：**
- A `duration`：全局事件是不是持续更久？
- B `n_part`：全局事件是不是扩散到更多触点？
- C `source_core_ignite_frac`：点火能量两档一不一样、且是否分源不对称？（**C1 的核心面板**）
- D `sign`：可读模板带不带"起点端→方向"结构？（neg→fwd 31/35；pos→10fwd/9rev）

A/B/C 三个连续量面板都**按 `source∈{neg,pos}` 分面**，每个 (source × class) 报 median + bootstrap 95% CI；每个 source 内做 local-vs-global 的 permutation 效应量（label-shuffle，B≥5000，descriptive p，不是 gated test）。

- [ ] **Step 1: 写脚本骨架 + 硬 assert（列契约 C5）**

```python
"""Stage 3 局部→全局层级图（探索性，分源；review-corrected 2026-06-14 C1）。
四面板各答一个独立问题：duration / n_part / source_core_ignite_frac / sign。
连续量面板按 source=neg/pos 分面，报 median + bootstrap 95% CI + permutation 效应量。
禁止: 不分源的 "local≈global" 单值；不写 "波前截断"（只 contained / relay-failure）。"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous"
RNG = np.random.default_rng(0)

required = {"cell", "seed", "tag", "event_id", "t_on", "source", "sign",
            "n_part", "propagation_class", "source_core_ignite_frac",
            "other_core_ignite_frac", "duration"}

def load():
    d = pd.read_csv(os.path.join(ROOT, "event_types.csv"))
    assert required <= set(d.columns), f"missing cols: {required - set(d.columns)}"
    assert set(d["propagation_class"]) >= {"local", "readable_global"}
    return d

def boot_ci(x, n=5000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan)
    meds = [np.median(RNG.choice(x, len(x), replace=True)) for _ in range(n)]
    return tuple(np.percentile(meds, [2.5, 97.5]))

def perm_effect(a, b, n=5000):
    """median(global) - median(local); two-sided permutation p on the label."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return dict(delta=np.nan, p=np.nan, n_local=len(a), n_global=len(b))
    obs = np.median(b) - np.median(a)
    pool = np.concatenate([a, b]); na = len(a)
    cnt = 0
    for _ in range(n):
        RNG.shuffle(pool)
        d = np.median(pool[na:]) - np.median(pool[:na])
        cnt += abs(d) >= abs(obs)
    return dict(delta=round(float(obs), 4), p=round((cnt + 1) / (n + 1), 4),
                n_local=na, n_global=len(b))
```

- [ ] **Step 2: 算分源效应量并落 JSON**

```python
def effect_table(d):
    out = {}
    for metric in ("duration", "n_part", "source_core_ignite_frac"):
        out[metric] = {}
        for src in ("neg", "pos"):
            loc = d[(d.propagation_class == "local") & (d.source == src)][metric].dropna()
            glo = d[(d.propagation_class == "readable_global") & (d.source == src)][metric].dropna()
            out[metric][src] = dict(
                local_median=round(float(np.median(loc)), 4) if len(loc) else None,
                local_ci=[round(float(c), 4) for c in boot_ci(loc)],
                global_median=round(float(np.median(glo)), 4) if len(glo) else None,
                global_ci=[round(float(c), 4) for c in boot_ci(glo)],
                **perm_effect(loc, glo))
    # D: sign vs source among readable_global
    rg = d[d.propagation_class == "readable_global"]
    out["sign_by_source"] = {
        src: dict(forward=int((rg[rg.source == src].sign == 1.0).sum()),
                  reverse=int((rg[rg.source == src].sign == -1.0).sum()))
        for src in ("neg", "pos")}
    return out
```
保存到 `stage3_hierarchy_effect_sizes.json`。

- [ ] **Step 3: 画四面板图（paper-grade、自包含）**

约束（feedback_figure_self_contained_paper_grade + figure_style_guide）：
- 每面板标题用朴素问句（"持续更久？" / "扩散更广？" / "点火能量（分源）" / "方向读出"）；轴标签是物理量不是变量名（`duration` → "持续时间 (ms)"、`n_part` → "点亮触点数 (/12)"、`source_core_ignite_frac` → "源核点火强度"）。
- A/B/C：分源（neg/pos）的 local vs global 分布（violin 或 box + 散点），median 标 95% CI 误差棒，面板内文字注 `Δ=…, p=…`。
- C 面板必须看得出 neg（local>global）与 pos（local<global）方向相反 —— 这是 C1 的全部意义。
- D：堆叠条 forward/reverse × source，标 31/35、10/9 计数。
- 单一共享图例；颜色锁 figure_style_guide：local vs global 用顺序色（viridis 两端），方向 forward/reverse 用 diverging 红蓝。
- 末尾 caption 一行："contained propagation / relay-failure，未证明波前截断"（C4）。

- [ ] **Step 4: 跑图 + 目视 + 写 figures/README.md**

Run: `python scripts/plot_stage3_local_global_hierarchy.py`
Expected: 生成 png + json，stdout 打印四组效应量；neg `source_core_ignite_frac` delta<0、pos delta>0。
然后**目视检查 png**（用户亲检），确认四面板各自独立、分源方向相反看得清。
在 `figures/README.md` 追加：
```markdown
### stage3_local_global_hierarchy.png
两端等强病灶单网络自发事件的"小局部 vs 少数可读全局"四面板对比，按起点端（neg/pos）分面。
A 持续时间、B 扩散触点数：全局显著更大（这才是两档的真正分界）。C 源核点火强度：**分源方向相反**（负端起的小事件比大事件还旺，正端反过来）——所以不能说"点火能量两档相等"。D 方向读出：负端起几乎全正向，正端起一半一半。
**关注点**：C 面板两个分面的箭头方向必须相反；A/B 的 local↔global 差距远大于 C。
```

- [ ] **Step 5: Commit**

```bash
git add scripts/plot_stage3_local_global_hierarchy.py \
        results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/figures/README.md
git commit -m "feat(stage3): source-faceted local-global hierarchy figure + effect sizes (C1)"
```
（注：`results/` 下 png/json 若被 .gitignore 忽略则不入库，只入脚本 + README —— 与现有约定一致。）

---

## Task 3: Step 3a — readable-template dry-run（映射验证，C5；先于聚类）

**Files:**
- Create: `scripts/analyze_stage3_readable_templates.py`（本 Task 写 dry-run 段；Task 4 加聚类段）
- Input: `readable_global_events.csv` + `record/<tag>/model_<tag>_lagPat_withFreqCent.npz` + `sidecar_*.json`

**契约（`src/sef_hfo_stage3.py:195-198`）：`event_id == lagPat 列号`；sidecar `t_on` 与 record 窗口 onset 同源。** dry-run 在跑聚类前把这个映射焊死。

**P1-1 record 定位（已复核，硬要求）**：`ROOT/record` 直挂只命中 **2/21** tag；全局递归 glob 命中 21/21，但 **2 个 tag 有重复 record**（`gs_te_sep0.6_std1.0_m17.0_s1`、`gs_te_sep0.7_std1.0_m17.5_s1`），取 `[0]` 可能配错列。**正确做法：先递归唯一定位 sidecar（它就是生成 event_types 行的那份），再从 `dirname(sidecar)/record/<tag>/` 取同级 record**（sidecar→sibling 唯一解析 21/21）。绝不全局 glob 后取第一个 —— 否则可能把 `regime_screen/` 的 sidecar 配上 `twoend_sweep/` 的同名 record（事件列不同），静默毁掉 `event_id→列` 映射。

- [ ] **Step 1: 写 dry-run（列契约 + sidecar→同级 record 映射 + montage 一致性 assert）**

```python
"""Stage 3 可读模板复核（描述性，C2）。先 dry-run 验证 54 个 readable_global 能精确对回
record 列，再走 masked 聚类。聚类结果只分三档：可复核 / 数据太少 / 源-方向歧义限制。
swap_class 是描述性模板复核，不是 Stage 3 证据。"""
import os, re, glob, json, argparse
import numpy as np, pandas as pd

ROOT = "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous"

required = {"cell", "seed", "tag", "event_id", "t_on", "source", "sign",
            "n_part", "propagation_class", "source_core_ignite_frac",
            "other_core_ignite_frac", "duration"}

def find_sidecar(tag):
    """递归唯一定位 sidecar（生成 event_types 行的源文件）。非唯一 → None（dry-run 会报）。"""
    hits = glob.glob(os.path.join(ROOT, "**", f"sidecar_{tag}.json"), recursive=True)
    return hits[0] if len(hits) == 1 else None

def find_record(tag):
    """P1-1: record 必须是 sidecar 的同级 record/<tag>/，不能全局 glob 取第一个。"""
    sc = find_sidecar(tag)
    if sc is None:
        return None
    hits = glob.glob(os.path.join(os.path.dirname(sc), "record", tag, "*_lagPat_withFreqCent.npz"))
    return hits[0] if len(hits) == 1 else None   # 要求唯一；重复/缺失 → None

def dry_run():
    rg = pd.read_csv(os.path.join(ROOT, "readable_global_events.csv"))
    assert required <= set(rg.columns), f"missing cols: {required - set(rg.columns)}"
    assert set(rg["propagation_class"]) == {"readable_global"}
    assert rg["source"].isin(["neg", "pos"]).all(), "readable_global source must be neg/pos"
    print(f"[dry-run] {len(rg)} readable_global events; "
          f"source={dict(rg.source.value_counts())}; sign={dict(rg.sign.value_counts())}")

    montage_ref, problems = None, []
    for tag, sub in rg.groupby("tag"):
        sc = find_sidecar(tag)
        if sc is None:
            n = len(glob.glob(os.path.join(ROOT, "**", f"sidecar_{tag}.json"), recursive=True))
            problems.append(f"{tag}: sidecar not unique (found {n})"); continue
        rec = find_record(tag)
        if rec is None:
            n = len(glob.glob(os.path.join(os.path.dirname(sc), "record", tag, "*_lagPat_withFreqCent.npz")))
            problems.append(f"{tag}: sibling record not unique under {os.path.dirname(sc)}/record (found {n})"); continue
        npz = np.load(rec, allow_pickle=True)
        n_events = npz["lagPatRank"].shape[1]
        names = [str(c) for c in npz["chnNames"]]
        if montage_ref is None: montage_ref = names
        elif names != montage_ref: problems.append(f"{tag}: montage mismatch")
        # event_id == lagPat 列号
        for eid in sub.event_id:
            if not (0 <= eid < n_events):
                problems.append(f"{tag}: event_id {eid} out of range {n_events}")
        # t_on 同源核对：sidecar 事件按 t_on 排序后第 event_id 个的 t_on == csv t_on
        sc_ev = sorted(json.load(open(sc)).get("events", []), key=lambda e: e["t_on"])
        for _, r in sub.iterrows():
            if int(r.event_id) < len(sc_ev):
                if abs(sc_ev[int(r.event_id)]["t_on"] - r.t_on) > 1e-6:
                    problems.append(f"{tag}: event_id {r.event_id} t_on misaligned")
    assert not problems, "DRY-RUN FAILED:\n" + "\n".join(problems)
    print(f"[dry-run] OK — all {len(rg)} events map to record columns; montage consistent "
          f"({len(montage_ref)} contacts)")
    return rg, montage_ref
```

- [ ] **Step 2: 跑 dry-run，先看它过不过**

Run: `python scripts/analyze_stage3_readable_templates.py --dry-run`（在 `main()` 里接 `--dry-run` 分支调用 `dry_run()`）
Expected（已复核）: 21 个 tag 全部经 sidecar→同级 record 唯一解析（含两个有重复全局 record 的 tag：`gs_te_sep0.6_std1.0_m17.0_s1`、`gs_te_sep0.7_std1.0_m17.5_s1` —— 同级解析消歧）；打印 54 events、source `{'neg':35,'pos':19}`、sign `{1.0:41,-1.0:13}`，最后 `[dry-run] OK — all 54 events map to record columns; montage consistent (12 contacts)`。
**若任何 assert 失败（sidecar/record 非唯一 / montage 不一致 / t_on 错位）→ 停，dry-run 会精确报是哪个 tag、是 sidecar 还是 sibling record 非唯一，先查再说，不进聚类。**

- [ ] **Step 3: Commit dry-run**

```bash
git add scripts/analyze_stage3_readable_templates.py
git commit -m "feat(stage3): readable-template dry-run — event_id->lagPat column mapping assert (C5)"
```

---

## Task 4: Step 3b — masked 三层聚类 + 三档判定（C2）

**Files:**
- Modify: `scripts/analyze_stage3_readable_templates.py`（加聚类段）
- Output: `results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/stage3_readable_templates_summary.json`

**复用 `pool_and_cluster_spontaneous.py` 已验证的 masked 机器**（同一套 `compute_adaptive_cluster_stereotypy(use_masked_features=True)` + `compute_swap_score_sweep`），只是输入换成**被选中的 readable_global 列**。这是 §6.1 question-match：我们要的是"这 54 个可读事件能不能聚成两套相反模板"，输入必须是这 54 列、masked、跨 tag 同 montage 拼接。

- [ ] **Step 1: 写"按列子集拼记录"helper**

```python
import sys; sys.path.insert(0, os.getcwd())
from src.interictal_propagation import (
    load_subject_propagation_events, compute_adaptive_cluster_stereotypy,
    build_cluster_templates)
from src.rank_displacement import compute_swap_score_sweep
from src.lagpat_rank_audit import build_masked_kmeans_features   # P1-3: distinct-pattern count

def pool_subset(rg_subset, montage_ref):
    """把 rg_subset 里每个 (tag,event_id) 对应的 lagPat 列抽出来，跨 tag 拼成一个
    合成 subject（masked-loader-readable）。montage 必须一致（dry-run 已 assert）。"""
    ranks, bools, raws = [], [], []
    for tag, sub in rg_subset.groupby("tag"):
        npz = np.load(find_record(tag), allow_pickle=True)
        assert [str(c) for c in npz["chnNames"]] == montage_ref
        cols = sub.event_id.astype(int).to_numpy()
        ranks.append(npz["lagPatRank"][:, cols])
        bools.append(npz["eventsBool"][:, cols])
        raws.append(npz["lagPatRaw"][:, cols])
    R = np.concatenate(ranks, axis=1)
    B = np.concatenate(bools, axis=1)
    return R, B, montage_ref
```

- [ ] **Step 2: 写"masked 聚类 + swap"helper（一层）**

```python
def n_unique_patterns(R, B):
    """P1-3: 有效形态多样性 = masked 特征矩阵里 distinct 行数。distinct 行很少时
    stable_k 不能当真（54 个可读事件已复核：pooled=13, neg=6, pos=7）。"""
    f = np.asarray(build_masked_kmeans_features(R, B, impute='event_median'))
    return int(np.unique(np.round(f, 6), axis=0).shape[0])

def cluster_layer(R, B, names, label):
    n_ev = R.shape[1]
    nuq = n_unique_patterns(R, B) if n_ev else 0
    out = dict(layer=label, n_events=int(n_ev), n_unique_masked_patterns=nuq)
    if n_ev < 8:                       # 事件太少（k=2 下限不稳）
        out["verdict"] = "数据太少（事件数<8）"; return out
    pr2 = compute_adaptive_cluster_stereotypy(R, B, names, use_masked_features=True)
    chosen_k = int(pr2["chosen_k"])
    out.update(chosen_k=chosen_k, stable_k=pr2.get("stable_k"),
               diversity_limited=bool(nuq < 10))   # P1-3: <10 distinct -> stable_k 慎读
    if chosen_k == 2 and pr2.get("labels"):
        labels = np.array(pr2["labels"], int)
        t0, t1 = build_cluster_templates(R, B, labels, 2)
        vm0 = np.isfinite(t0) & (t0 >= 0); vm1 = np.isfinite(t1) & (t1 >= 0)
        try:
            swap = compute_swap_score_sweep(t0, t1, vm0, vm1, n_perm=1000, seed=0)
            out["swap_class_descriptive"] = swap.get("swap_class")   # C2: 描述性，非 Stage3 证据
            out["decision_k"] = swap.get("decision_k")
        except Exception as e:
            out["swap_error"] = repr(e)
    return out
```

> **P1-3 纪律**：`n_unique_masked_patterns` 必须进 summary 的每一层。`diversity_limited=True`（distinct<10）时，该层的 `stable_k`/`chosen_k` **不能**被读成"找到了稳定的两套模板"——它只反映了极少的几个 distinct 形状（已复核 neg=6/pos=7）。这是 P1-1/P1-2 之外把 stable_k 过度解释挡住的第三道闸。

- [ ] **Step 3: 三层调用（C2：全池化 sanity / 按 source / 按 top cell）**

```python
def cluster_all(rg, montage_ref):
    layers = []
    # (a) 全池化 —— 仅 sanity
    R, B, names = pool_subset(rg, montage_ref)
    layers.append({**cluster_layer(R, B, names, "pooled_sanity"),
                   "note": "全池化仅 sanity；混了 cell/seed/source，不作模板结论"})
    # (b) 按 source 分层
    for src in ("neg", "pos"):
        sub = rg[rg.source == src]
        R, B, names = pool_subset(sub, montage_ref)
        layers.append(cluster_layer(R, B, names, f"source_{src}"))
    # (c) 按 top cell 分层（事件最多的 cell）
    top_cell = rg.cell.value_counts().idxmax()
    sub = rg[rg.cell == top_cell]
    R, B, names = pool_subset(sub, montage_ref)
    layers.append({**cluster_layer(R, B, names, f"top_cell:{top_cell}"),
                   "n_seeds": int(sub.seed.nunique())})
    return layers
```

- [ ] **Step 4: 三档判定 + 落 JSON**

每层 verdict 只能取三档之一（C2）：
- `可复核`：n_events≥8 + 形态多样性够（n_unique≥10）+ chosen_k==2 + swap 有定义；
- `数据太少`：n_events<8 **或** 形态多样性不足（n_unique<10，P1-3）**或** swap exit 因样本不足；
- `源-方向歧义限制`：聚到了但 source 内方向读出本身≈coin flip（pos 源 10/9），模板分离被歧义吃掉。

```python
def verdict_of(layer):
    if str(layer.get("verdict","")).startswith("数据太少"): return layer["verdict"]
    if layer.get("diversity_limited"):
        return f"数据太少（形态多样性不足: n_unique={layer['n_unique_masked_patterns']}）"  # P1-3
    if "swap_class_descriptive" not in layer: return "数据太少"
    return "可复核"   # "源-方向歧义限制" 由人读 source_pos 层 + sign_by_source 判定，写进归档段
```
保存 `stage3_readable_templates_summary.json`，stdout 打印每层 (n_events, n_unique_masked_patterns, chosen_k, swap_class_descriptive, verdict)。

- [ ] **Step 5: 跑全流程**

Run: `python scripts/analyze_stage3_readable_templates.py`
Expected（已复核 distinct counts）: dry-run OK → 四层输出。`n_unique_masked_patterns`：pooled **13**/54、source_neg **6**/35、source_pos **7**/19 → 两个 source 层 `diversity_limited=True` → verdict 落 **"数据太少（形态多样性不足）"**；pooled 层 distinct=13≥10 可进"可复核"但带 `note=sanity-only`（混了 source，不作模板结论）。**这正是 C2 要的诚实结果：分源模板复核因形态多样性不足做不实，pooled 仅 sanity。不得**把任何一层的 `swap_class` 写成"Stage 3 PASS / 双向模板复活"。

- [ ] **Step 6: 写归档结果段（三档口径 + C4 baseline）**

在 `stage3_regime_screen_2026-06-14.md` 追加一节 "Step 2/3 局部→全局层级定量（2026-06-14）"：
- Step 2 分源效应量（带 CI/p）：duration/n_part 是真正分界；source_core 分源相反（C1）。
- Step 3 三层三档：每层 n_events + verdict；`swap_class` 标注"描述性模板复核，非 Stage 3 证据"（C2）。
- baseline 对照句：相对 Stage 2 单灶干净全局事件 / oneend sign calibration / hot-collision 条件（C4）。
- 边界句：contained / relay-failure，未证明波前截断（C4）。

- [ ] **Step 7: Commit**

```bash
git add scripts/analyze_stage3_readable_templates.py \
        docs/archive/topic4/sef_hfo/stage3_regime_screen_2026-06-14.md
git commit -m "feat(stage3): masked 3-layer readable-template review (descriptive) + archive results (C2/C4)"
```

---

## Task 5: Step 4 GATE —— 默认不开（C3，仅记录门决策）

**Files:** 无新代码。这是一个**判定关卡**，不是仿真任务。

- [ ] **Step 1: 用硬标准查当前是否够格重跑**

Run:
```bash
python3 - <<'PY'
import pandas as pd
rg = pd.read_csv("results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/readable_global_events.csv")
g = rg.groupby(["cell","source"]).agg(n=("event_id","size"),
       seeds=("seed", lambda s: s.nunique())).reset_index()
ok = g[(g.n>=20) & (g.seeds>=2)]
print(g.to_string()); print("\nCELLS PASSING (n>=20 per source AND >=2 seeds):", len(ok))
PY
```
Expected: `CELLS PASSING ... : 0`（已复核：当前 0 个 cell 达标）。

- [ ] **Step 2: 记录门决策（不开仿真）**

在归档段写明："Step 4 重跑门 = 每 source ≥20 readable_global 且跨 ≥2 seeds；当前 0/19 cell 达标 → **不开仿真**。只有 Step 3 明确显示'事件数不足但形态有信号'（某 source 层 chosen_k==2 且 swap 方向清晰但 n<阈值）才重提，且重跑只针对达标候选格、加长 T / 加 seed。"
**禁止**在本计划内自动触发任何长仿真（C6）。

---

## §C 科学表达规则（写结果时逐条对照）

- **允许**：local 不是弱点火（分源后仍成立，方向不对称）；区分 local↔global 的是传播（duration + spread）不是能量；contained / relay-failure；"很多小局部 + 少数可读大"的层级本身比强行平衡双向列车更像真实 HFO；readable 模板在 neg 端带方向、pos 端歧义。
- **禁止**：① 不分源的"点火能量 local≈global"单值结论（C1）；② 把 54 事件当一个 homogeneous subject、把 `swap_class` 当 Stage 3 证据 / 双向模板复活（C2）；③ "波前截断 / 先走后停"（C4，需时空快照）；④ 把这一步说成 Stage 3 主问（单网络标签-时序独立性）的答案 —— 那个测试床从未建成（见 `axisB_propagation_handoff_2026-06-14.md`）。
- **tier 纪律**：Stage 3 主问 = 未检验；本步 = 探索性机制刻画，进 archive、不进建模主文档主结论。

---

## §D Self-Review（写完计划后自查）

- **spec 覆盖**：C1→Task 2；C2→Task 4（三层三档 + swap 描述性）；C3→Task 5（硬门 + 默认不开）；C4→Task 2 caption + Task 4 Step 6 baseline 句；C5→Task 1（doc sync）+ Task 3（asserts）；C6→Task 5 Step 2 + 全程不改 runner。✅
- **placeholder 扫描**：每个 code step 给了真实可跑代码；asserts、helper、reuse 函数名（`compute_adaptive_cluster_stereotypy`/`build_cluster_templates`/`compute_swap_score_sweep`/`load_subject_propagation_events`）均来自已读源文件。✅
- **类型一致**：`pool_subset` 返回 `(R,B,names)` 与 `cluster_layer(R,B,names,label)` 入参一致；`find_record`/`find_sidecar` 在 Task 3 定义、Task 4 复用。✅
- **风险点**：Step 3a 的 tag→record 命名（`conf_*` vs `gs_te_*`）是唯一可能 dry-run 卡住处 —— 已在 Task 3 Step 2 标注，dry-run 会精确报哪个 tag 找不到 record，属设计内的 fail-loud，不是 placeholder。
