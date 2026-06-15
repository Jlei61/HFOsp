# Stage 3 局部→全局 Regime-Map Scout 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用一个**粗网格、短时长**的扫描把 cm-SNN 两端等强病灶的参数空间画成一张"局部 / 中继 / 共点火"三态地图，回答"什么参数让事件停在局部、什么参数让它中继成大传播、什么参数让两端共点火"，并用一个**数值进入门**判断是否存在"两端都能产出足够干净全局事件、碰撞又不高"的工作点——只有该门通过，才进二级的稳定双向模板测试。

**Architecture:** 复用已验证的 spontaneous runner（`--lesion twoend_equal` + `--dump-fullfield`）跑一个 `core_mean × sep_frac` 粗网格（短 T，每格几 seed），新增一个 RAM-safe 驱动 + 一个把每格 sidecar 汇成三态指标 + 进入门 + 地图图的分析脚本。**长仿真不是第一步**——只有 scout 的进入门点亮某个工作点后，长跑才作为该格的确认。

**Tech Stack:** Python / numpy / matplotlib；bash 驱动（RAM 预检并发闸）；已存在的 `scripts/run_sef_hfo_snn_cm_spontaneous_readout.py`（引擎现在 `src/snn_engine/`）+ sidecar 字段（`hidden_source_label` / `clean_for_timing` / `n_part`）。

---

## §0 朴素导读（先读这段）

**测了什么** —— 上一步我们搞清楚了：在那个又冷又稀的候选工作点上，大多数自发放电是"点得旺却传不出去"的小局部事件，少数能传开的大事件两头其实等价。现在不再追"那一个点上的不对称"，而是退一步问一个**地图级**问题：**调哪些旋钮（核兴奋度 core_mean、两灶间距 sep_frac），事件会停在局部 / 会中继成大传播 / 会让两头同时点着撞在一起？** 目的是找一块"两头都能各自传出干净大事件、又不老撞车"的中间地带。

**怎么测的** —— 粗扫一个 3×3 网格（核兴奋度 3 档 × 间距 3 档），每格几个随机种子，每次跑一段**不长**的记录（只够看清这格属于哪种态，不为统计），把每格的事件按四类点票：局部 / 碰撞 / 干净大事件(负端起) / 干净大事件(正端起)。**进入门是一条写死的数值线**：某格只有当"两端各自至少有 K 个干净大事件 + 碰撞率低于阈值 + 不是几乎全局部"时，才算"够格做双向模板测试"。

**揭示了什么（预期产出）** —— 一张三态地图 + 一句话结论：要么"存在中间地带 X 格满足两端中继 + 低碰撞 → 下一步只在 X 格上加长确认"，要么"网格内不存在这样的中间地带（冷格只攒重复局部、热格只升碰撞）→ 这套两灶单网构造在可行参数里给不出可测的稳定双向"。**长仿真只在某格点亮进入门之后作为确认，绝不是第一步。**

（内部归档代号：`twoend_equal` / `core_mean`·`sep_frac` 网格 / regime∈{local-dominated, relay-capable, collision-dominated} / `hidden_source_label`·`clean_for_timing`·`n_part` / advance-gate / post-gate confirmation）

---

## §A Scope + 三态问题 + 进入门（数值锁定）

**事件分桶（互斥，5 桶，按 hidden_source_label 优先；FATAL-1/2 锁定）：**
```
if hidden == "collision":                       -> collision           # 两端共点火（来源级）
elif hidden in {"neg","pos"} and clean_for_timing: -> clean_global_{neg,pos}  # 干净大事件，按端
elif n_part < 7:                                -> local               # 传不开（严格 n_part<7）
else:                                           -> dirty_global        # n_part>=7 但读不清/来源不明
```
- **FATAL-1：`local` 严格 = 非碰撞 ∧ `n_part<7`。** `n_part>=7 但读不清` 归 `dirty_global`，**不进 `local_frac`**（否则把"传不开"和"传开了但读不清"混成冷格）。
- **FATAL-2：干净大事件只看 `hidden_source_label∈{neg,pos} ∧ clean_for_timing`。** 不自造 `readable`/`axis_err` 判据——`clean_for_timing` 已由 `src/sef_hfo_stage3.build_sidecar()` 锁成「单端来源 ∧ 可读 ∧ axis_err<25」（且 `readable` 已含 `n_part>=part_min`，见 `:212-217`）。复用合同，别重造。

**三个并列问题（每个对应一个地图层）：** ① 停在局部 → `local_frac`（=local/total，严格定义如上）；② 中继成大传播 → `clean_global_neg`/`clean_global_pos`（按端）；③ 共点火 → `collision_rate`（hidden=="collision" 占比）。

**进入门（ADVANCE GATE，写死数值，encode 结论；FATAL-3 三类失败）：** 某格（pooled over seeds）verdict ∈：
- `too_cold/undersampled`：`total_events < 10`（短 T 下事件不足，**不判** pass/fail）。
- `fail:collision_dominated`（**热**）：`collision_rate ≥ 0.30`。
- `fail:local_dominated`（**冷**）：`local_frac ≥ 0.90`。
- `PASS:relay_both_ends`：`clean_global_neg ≥ 2` **且** `clean_global_pos ≥ 2`（两端都中继）。
- `fail:one_end_or_no_clean_relay`（**中间**：低碰撞、非局部，但双端干净中继不足/一端独大）：以上都不满足。子字段 `mid_reason ∈ {one_end_dominant, no_clean_relay}` 区分"一端独大"vs"两端都读不出干净大事件（多 dirty_global）"。

> **门语义（acceptance-gate 纪律）**：PASS 不代表"找到双向模板"，只代表"值得花长 T accumulate 事件做模板测试"。**三个失败方向**：热（碰撞主导）/ 冷（局部主导，长跑只攒重复局部）/ 中间（低碰撞非局部但一端独大或读不出干净大事件）——中间类正是 regime-screen 反复撞上的"一端通吃"，是双向问题的核心障碍，**必须单列，不并入冷**。

**FATAL-4 — seed 稳定性（pooled 太松，必须输出）：** 每格除 pooled 计数外，**必输** `n_seeds_neg_clean`（有 ≥1 neg 干净大事件的 seed 数）/ `n_seeds_pos_clean` / `n_seeds_both`（两端都有的 seed 数），否则单 seed 偶然点亮整格无法察觉、Task 3「跨 seed 不稳」档无数据支撑。`dirty_global_count` / `ambiguous` 也入 summary。

**纪律（写死）：**
- **本 scout 用短 T**（`T=3000`，每格 3 seed）。**不在本计划内跑任何 >1 格的长 T 仿真。**
- 长 T 确认**只在某格通过 ADVANCE GATE 后**、且作为该格的二级步骤单独提（不在本 plan 自动触发）。
- 二级"稳定双向模板测试"复用 Stage-2 的 masked pipeline + `analyze_stage3_readable_templates.py`，**不在本 plan 实现**——本 plan 只到"地图 + 进入门判定"。

**Round-2 审阅补丁（2026-06-15，并入合同）：**
- **IMPROVE-5（硬阈值不裸奔）**：`analyze_stage3_regime_map.py` 的 `gate()`/`cell_of()`/事件分桶必须有 **pytest 测试**（`tests/test_stage3_regime_map.py`，合成 sidecar 事件覆盖 5 桶 + 5 个 verdict + cell_of 解析）。阈值是科学合同，先测后跑。
- **IMPROVE-6**：summary 加 `dirty_global_count` / `ambiguous`（=dirty_global）/ per-seed clean 计数（FATAL-4）。地图**仍 3 面板**（local_frac / collision_rate / min(neg,pos) clean global），不加废面板。
- **IMPROVE-7（gitignore 一致）**：`results/.../regime_map/` 被 gitignore。`figures/README.md` 只写盘**本地**（供目视，AGENTS.md），**不 git add**；图的文字说明进 **archive doc**（Task 3）。不要 `git add -f` 跟 gitignore 对打。
- **IMPROVE-8（skip 稳健）**：driver 的 skip 不只看 `sidecar_$tag.json` 存在；要 `sidecar` **和** `readout` 都存在且 JSON 可 load（防半写文件假 skip）。

---

## §B 文件结构

- **Create:** `scripts/run_stage3_regime_map_scout.sh` —— RAM-safe 粗网格驱动（`core_mean×sep_frac×seed`，T=3000，`--dump-fullfield`）。
- **Create:** `scripts/analyze_stage3_regime_map.py` —— 每格三态指标 + ADVANCE GATE 判定 + 三态地图图 + figures/README。
- **Reuse（不改）:** `scripts/run_sef_hfo_snn_cm_spontaneous_readout.py`（runner，已有 `--dump-fullfield`）；sidecar 字段（`hidden_source_label`/`clean_for_timing`/`n_part`，由 `build_sidecar` 在运行时写）。
- **Output dir（gitignored）:** `results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/regime_map/`。

---

## Task 1: Regime-map scout 驱动（粗网格，RAM-safe）

**Files:**
- Create: `scripts/run_stage3_regime_map_scout.sh`

复用 `run_stage3_source_asymmetry_battery.sh` 已验证的 RAM 预检并发闸（cap + free-RAM gate），换成 `core_mean × sep_frac × seed` 网格。

- [ ] **Step 1: 写驱动脚本**

```bash
#!/usr/bin/env bash
# Stage 3 local-global REGIME-MAP scout (2026-06-15). Coarse core_mean x sep_frac grid, SHORT T,
# few seeds — maps local / relay / collision regimes. NOT a long formal run (that is post-gate only).
# RAM-safe: ~13GB/sim, cap concurrency + gate on free RAM (OOM lesson 2026-06-14). Idempotent-ish.
set -u
cd /home/honglab/leijiaxin/HFOsp
OUT=results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/regime_map
mkdir -p "$OUT/logs"
MEANS="${MEANS:-16.5 17.0 17.5}"
SEPS="${SEPS:-0.6 0.7 0.8}"
SEEDS="${SEEDS:-1 2 3}"
MAXJOBS="${MAXJOBS:-5}"            # 5 x ~13GB = 65GB
MIN_FREE_GB="${MIN_FREE_GB:-40}"
T="${T:-3000}"

free_gb(){ free -g | awk '/^Mem:/{print $7}'; }
running(){ pgrep -f "run_sef_hfo_snn_cm_spontaneous_readout.py.*regime_map" | grep -vc grep; }

launch(){  # $1=mean $2=sep $3=seed
  tag="rm_m${1}_sep${2}_s${3}"
  if [ -f "$OUT/sidecar_$tag.json" ]; then echo "[skip] $tag"; return; fi
  while [ "$(running)" -ge "$MAXJOBS" ] || [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; do sleep 30; done
  echo "[$(date +%H:%M:%S)] launch $tag (running=$(running) free=$(free_gb)G)"
  nohup python3 scripts/run_sef_hfo_snn_cm_spontaneous_readout.py \
    --lesion twoend_equal --core-mean "$1" --sep-frac "$2" --core-std 1.0 --seed "$3" \
    --T "$T" --dump-fullfield --tag "$tag" --out "$OUT" > "$OUT/logs/$tag.log" 2>&1 &
  sleep 5
}

echo "SCOUT start $(date +%H:%M:%S): means=[$MEANS] seps=[$SEPS] seeds=[$SEEDS] T=$T"
for m in $MEANS; do for sp in $SEPS; do for s in $SEEDS; do launch "$m" "$sp" "$s"; done; done; done
wait
echo "SCOUT DONE $(date +%H:%M:%S)"
```

- [ ] **Step 2: 设可执行**

Run: `chmod +x scripts/run_stage3_regime_map_scout.sh`

- [ ] **Step 3: 单格 smoke（确认网格 tag/参数接线，短 T）**

Run（前台快验，约 5min，与他人 sim 共存安全）:
```bash
SMK=/tmp/rm_smoke; mkdir -p "$SMK"
timeout 600 python3 scripts/run_sef_hfo_snn_cm_spontaneous_readout.py \
  --lesion twoend_equal --core-mean 17.0 --sep-frac 0.7 --core-std 1.0 --seed 1 \
  --T 400 --dump-fullfield --tag rm_m17.0_sep0.7_s1 --out "$SMK" 2>&1 | tail -3
ls "$SMK"/sidecar_rm_m17.0_sep0.7_s1.json "$SMK"/fullfield_rm_m17.0_sep0.7_s1.json
```
Expected: 两个 json 都写出；stdout 有 `stage3 SOURCE counts`。**若报错 → 停，先查 runner 参数接线。**

- [ ] **Step 4: Commit 驱动**

```bash
git add scripts/run_stage3_regime_map_scout.sh
git commit -m "feat(stage3): regime-map scout driver — coarse core_mean x sep_frac grid, RAM-safe, short T"
```

---

## Task 2: Regime-map 分析 + ADVANCE GATE + 地图图

**Files:**
- Create: `scripts/analyze_stage3_regime_map.py`
- Output: `results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/regime_map/regime_map_summary.json` + `figures/stage3_regime_map.png` + `figures/README.md`

每格指标直接从 sidecar 字段算（`hidden_source_label`/`clean_for_timing`/`n_part`，运行时由 `build_sidecar` 写）——与 `analyze_stage3_event_types.py` 的 `prop_class` 规则一致（n_part≥7 ∧ readable ∧ source∈{neg,pos} = readable_global），但按端拆 clean global。

- [ ] **Step 1: 写分析（每格三态指标 + 进入门）**

```python
"""Stage 3 local-global REGIME MAP (2026-06-15). Aggregate each core_mean x sep_frac cell's
sidecars into 4-way regime metrics + a hard ADVANCE GATE. NOT a pass/fail of the science — the
gate only says 'this cell is worth a longer post-gate run to accumulate enough events for the
bidirectional-template test'. fail is split into collision_dominated (hot) vs local/too_cold."""
import os, re, glob, json
import numpy as np

ROOT = "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/regime_map"
PART_MIN = 7
# ADVANCE GATE thresholds (locked §A)
K_END = 2          # >= this many clean global events PER END
COLL_MAX = 0.30    # collision_rate must be below
LOCAL_MAX = 0.90   # local_frac must be below
MIN_EV = 10        # else too_cold/undersampled -> not judged


def cell_of(tag):                      # rm_m17.0_sep0.7_s3 -> (17.0, 0.7)
    m = re.search(r"_m([\d.]+)_sep([\d.]+)_s", tag)
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def cell_metrics():
    cells = {}
    for sc in sorted(glob.glob(os.path.join(ROOT, "sidecar_rm_*.json"))):
        tag = os.path.basename(sc)[8:-5]
        mean, sep = cell_of(tag)
        ev = json.load(open(sc)).get("events", [])
        d = cells.setdefault((mean, sep), dict(n=0, local=0, collision=0, cg_neg=0, cg_pos=0,
                                               other_global=0, seeds=set()))
        d["seeds"].add(tag.split("_s")[-1])
        for e in ev:
            d["n"] += 1
            lab = e.get("hidden_source_label")
            npart = e.get("n_part") or 0
            readable = npart >= PART_MIN and e.get("axis_err") is not None
            if lab == "collision":
                d["collision"] += 1
            elif readable and e.get("clean_for_timing") and lab == "neg":
                d["cg_neg"] += 1
            elif readable and e.get("clean_for_timing") and lab == "pos":
                d["cg_pos"] += 1
            elif readable:
                d["other_global"] += 1          # global but unclean/ambiguous source
            else:
                d["local"] += 1
    return cells


def gate(d):
    n = d["n"]
    if n < MIN_EV:
        return "too_cold/undersampled"
    local_frac = d["local"] / n
    coll_rate = d["collision"] / n
    if coll_rate >= COLL_MAX:
        return "fail:collision_dominated"
    if local_frac >= LOCAL_MAX:
        return "fail:local_dominated"
    if d["cg_neg"] >= K_END and d["cg_pos"] >= K_END:
        return "PASS:relay_both_ends"
    return "fail:one_or_no_end_relays"


def main():
    cells = cell_metrics()
    rows = []
    for (mean, sep), d in sorted(cells.items()):
        n = max(1, d["n"])
        rows.append(dict(core_mean=mean, sep_frac=sep, n_events=d["n"], n_seeds=len(d["seeds"]),
                         local_frac=round(d["local"] / n, 3), collision_rate=round(d["collision"] / n, 3),
                         clean_global_neg=d["cg_neg"], clean_global_pos=d["cg_pos"],
                         other_global=d["other_global"], verdict=gate(d)))
    passing = [r for r in rows if r["verdict"].startswith("PASS")]
    summary = dict(gate_thresholds=dict(K_END=K_END, COLL_MAX=COLL_MAX, LOCAL_MAX=LOCAL_MAX, MIN_EV=MIN_EV),
                   cells=rows, n_passing=len(passing), passing_cells=[(r["core_mean"], r["sep_frac"]) for r in passing])
    json.dump(summary, open(os.path.join(ROOT, "regime_map_summary.json"), "w"), indent=2)
    print(f"=== REGIME MAP ({len(rows)} cells) ===")
    for r in rows:
        print(f"  m{r['core_mean']}/sep{r['sep_frac']}: n={r['n_events']:>3} "
              f"local={r['local_frac']} coll={r['collision_rate']} "
              f"cg neg{r['clean_global_neg']}/pos{r['clean_global_pos']} -> {r['verdict']}")
    print(f"\nPASSING (relay both ends, low collision): {summary['passing_cells'] or 'NONE'}")
    return summary, rows
```

- [ ] **Step 2: 加三态地图图（3 面板，每板一问 §7）**

在 `main()` 末尾追加（`local_frac` 热图 / `collision_rate` 热图 / `clean_global both-ends` 指示，core_mean × sep_frac 栅格）：

```python
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    means = sorted({r["core_mean"] for r in rows}); seps = sorted({r["sep_frac"] for r in rows})
    def grid(key):
        g = np.full((len(means), len(seps)), np.nan)
        for r in rows:
            g[means.index(r["core_mean"]), seps.index(r["sep_frac"])] = r[key]
        return g
    cg_both = np.full((len(means), len(seps)), np.nan)
    for r in rows:
        cg_both[means.index(r["core_mean"]), seps.index(r["sep_frac"])] = min(r["clean_global_neg"], r["clean_global_pos"])
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, (g, ttl, cm) in zip(ax, [(grid("local_frac"), "事件停在局部的比例", "viridis"),
                                    (grid("collision_rate"), "两端共点火(碰撞)比例", "magma"),
                                    (cg_both, "两端中继下限 min(neg,pos) 干净大事件", "cividis")]):
        im = a.imshow(g, origin="lower", aspect="auto", cmap=cm)
        a.set_xticks(range(len(seps))); a.set_xticklabels([f"sep{s}" for s in seps])
        a.set_yticks(range(len(means))); a.set_yticklabels([f"m{m}" for m in means])
        a.set_title(ttl, fontsize=10); fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle("两端等强病灶 cm-SNN：局部 / 中继 / 共点火 三态地图（短 T scout）", fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "figures", "stage3_regime_map.png"), dpi=150, bbox_inches="tight")
```
（注：中文字体 rcParams 复用 `scripts/plot_stage3_local_global_hierarchy.py` 顶部的 Noto Sans CJK 块——否则中文成方框。）

- [ ] **Step 3: 跑分析 + 写 figures/README.md**

Run: `python scripts/analyze_stage3_regime_map.py`
Expected: 9 格表 + 每格 verdict + PASSING 列表。
**目视检查 png**，然后写 `regime_map/figures/README.md`：
```markdown
### stage3_regime_map.png
两端等强病灶 cm-SNN 在 核兴奋度(core_mean) × 间距(sep_frac) 网格上的三态地图（短 T scout）。
左：事件停在局部的比例（越亮越"传不动"）；中：两端共点火/碰撞比例（越亮越"撞车"）；右：两端中继下限 min(neg,pos) 干净大事件数（越亮=两端都越能传出干净大事件）。
**关注点**：右图有没有亮格（两端都≥2 干净大事件）且同位置左/中都偏暗（低局部+低碰撞）——那才是值得加长 T 确认的"中间地带"。
```

- [ ] **Step 4: Commit**

```bash
git add scripts/analyze_stage3_regime_map.py results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/regime_map/figures/README.md
git commit -m "feat(stage3): regime-map analysis + advance gate + 3-panel map (local/collision/relay-both-ends)"
```

---

## Task 3: 跑 scout、判进入门、归档 + 决定是否进 post-gate

**Files:**
- Modify: `docs/archive/topic4/sef_hfo/stage3_regime_screen_2026-06-14.md`（追加 regime-map scout 结果段）

- [ ] **Step 1: 机器预检（OOM 纪律）**

Run:
```bash
free -g | awk '/^Mem:/{print "available:",$7"G"}'
pgrep -af "run_sef_hfo_snn_cm_spontaneous_readout.py" | grep -v grep | grep -v regime_map | head
```
确认 available ≥ ~80G 且了解别人在跑什么；驱动自带 free<40G 不启动的闸。

- [ ] **Step 2: 后台跑 scout（27 run，cap 5，约 3.5h）**

Run（background）:
```bash
MEANS="16.5 17.0 17.5" SEPS="0.6 0.7 0.8" SEEDS="1 2 3" MAXJOBS=5 MIN_FREE_GB=40 T=3000 \
  bash scripts/run_stage3_regime_map_scout.sh
```
等 `SCOUT DONE`；确认 `sidecar_rm_*.json` 有 27 个、logs 无 Traceback。

- [ ] **Step 3: 出地图 + 判门**

Run: `python scripts/analyze_stage3_regime_map.py`
读 `regime_map_summary.json` 的 `passing_cells`。

- [ ] **Step 4: 归档结果段（三档口径）**

在 `stage3_regime_screen_2026-06-14.md` 追加 "## Local-global regime-map scout（2026-06-15）"，§8 朴素话开头，按结果写三档之一：
- **存在中间地带**：列出 PASS 格（两端中继 + 低碰撞）→ 下一步**只在这些格**加长 T 做二级稳定双向模板测试（单独提，不在本 plan）。
- **不存在**：网格内无 PASS 格；明确两个失败方向各自占哪片（冷格 `local_dominated`/`too_cold`、热格 `collision_dominated`）→ "两灶单网在此可行参数里给不出可测稳定双向"，回落 Stage-2 结构层。
- **边界不清**：PASS 格存在但 n_events 少/跨 seed 不稳 → 标"需更密 seed 或邻格细扫"，仍不直接长跑。
- 末尾锁纪律：长 T 仅 post-gate 确认；冷长跑只攒重复局部、热长跑只升碰撞（已由地图佐证）。

- [ ] **Step 5: Commit**

```bash
git add docs/archive/topic4/sef_hfo/stage3_regime_screen_2026-06-14.md
git commit -m "docs(stage3): local-global regime-map scout results + post-gate decision"
```

---

## §C 纪律 + Self-Review

**纪律（写结果时对照）：**
- 进入门是**预注册数值**（K_END=2 / COLL_MAX=0.30 / LOCAL_MAX=0.90 / MIN_EV=10），pass=值得长跑确认，**不**=找到双向模板。
- fail 必须分 `collision_dominated`(热) vs `local_dominated/too_cold`(冷)——别合并成一个"fail"。
- **长 T 仿真只在某格 pass 后作为该格确认**，不在本 plan 自动触发；本 plan 到"地图+判门"为止。
- 二级稳定双向模板测试复用现有 masked pipeline，**不在本 plan 实现**。

**Self-Review：**
- **spec 覆盖**：三态问题→Task2 三指标 + 三面板；进入门数值→Task2 `gate()`；短 T scout 非长跑→Task1 T=3000 + §A 纪律；冷/热失败分流→`gate()` 两个 fail 子类 + Task3 归档三档；RAM 安全→Task1 free-gate（复用 battery 模式）。✅
- **placeholder 扫描**：driver/分析/图均给真实可跑代码；门阈值写死；图复用已存在的 CJK 字体块。✅
- **类型一致**：`cell_of`→(mean,sep)；`cell_metrics` 返回的 dict 键（n/local/collision/cg_neg/cg_pos/other_global/seeds）在 `gate()`/`main()` 一致使用。✅
- **风险点**：T=3000 在冷格可能 <10 事件 → 落 `too_cold/undersampled`（设计内，不误判 fail）；这本身是地图信息（冷格就是冷）。
