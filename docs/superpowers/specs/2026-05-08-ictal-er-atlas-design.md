# Ictal ER-onset Atlas — Design

> **Topic**：PR-T3-1 v2.2 Layer A 跑完后，给 16 epilepsiae cohort 做"ER-ratio 变化时间 vs clinical/EEG onset"的视觉诊断 atlas，支持 per-subject 深入分析。
>
> **决定 (Brainstorm 2026-05-08)**：per-subject 汇总图 + per-seizure 双 band 改造图都做；t_ER_onset 由 augmented Layer A 持久化；新建 `scripts/plot_ictal_er_atlas.py`，不动 archive。

---

## 1. 背景与动机

PR-T3-1 已经把 16 epilepsiae subject 的 Layer A producer 跑完，结论是 **ER-ratio 单指标不能作为整体 data-driven SOZ label**——主要因为 within-subject seizure-to-seizure 变异性（即同一 subject 不同 seizure 排序差异大）。这个结论已经写进 `docs/archive/topic3/pr_t3_1_data_driven_soz/per_subject_ictal_er_atlas.md`，是文字版。

但 **"ER 变化时间是否和 clinical/EEG onset 一致"** 这个二级问题在现有视觉产物里看不出来：
- `results/interictal_propagation/ictal_alignment/_sentinel_step2/` 里的 80 张 PNG 是 sentinel 历史快照，时间窗 [-200, +200]s 太宽、单 PNG 单 band 没法横扫、**没有任何 t_ER_onset 视觉标记**——光看热图根本判断不出 channel 在 clinical onset 之前还是之后被 CUSUM 触发。
- archive 里的 atlas 是文字 (`r_sz` / Wilcoxon / tags)，但**没有时序信息**。

本设计交付一套新的可视化 atlas，专门回答：每个 channel 的 ER-onset 走在 clinical onset 前还是后？走前的程度如何？这个走前/走后的图样在 same subject 不同 seizure 之间是否一致？

---

## 2. 范围

**In scope**：
- 16 epilepsiae cohort（v2.2 audit_eligible 15 + 916 sentinel-only）
- Per-seizure 双 band 改造图（左 gamma / 右 broad，主时间窗 [-120, +30]s）
- Per-subject 汇总图（channel × seizure 的 t_ER_onset 矩阵，双 band 上下两块）
- Cohort batch 模式
- 新脚本：`scripts/plot_ictal_er_atlas.py`
- 增项：`src/ictal_er_rank.py` + `scripts/run_ictal_er_rank.py` 持久化 t_er_onset_sec
- README.md（中文）+ figures dir

**Out of scope**：
- Yuquan 9 subject（依赖 yuquan 版本 `extract_seizure_window`，独立后续 PR）
- 任何对 Layer A 科学定义的修改（r_sz / s_sz / tags 公式不动；只增加 per-channel onset 持久化字段）
- 任何统计推断（atlas 是视觉诊断工具，不出 p 值）
- 旧 `_sentinel_step2/` 目录的 PNG 保持原样（archive，不重生不删）
- Per-seizure z-ER trace 中间行（已决议删除，仅留 raw + heatmap）

---

## 3. 架构

### 3.1 新增产物根目录

```
results/data_driven_soz/layer_a_ictal_er_rank/atlas/
├── per_seizure/
│   ├── epilepsiae_548_seizure_00.png
│   ├── epilepsiae_548_seizure_01.png
│   ├── ...
├── per_subject/
│   ├── epilepsiae_548.png
│   ├── epilepsiae_916.png
│   └── ...
└── figures/
    └── README.md  ← 必须，中文，符合 AGENTS.md 结果目录标准
```

### 3.2 数据流

```
原始 EDF / .data
       │
       ▼
extract_seizure_window (src/ictal_onset_extraction.py)
       │
       ▼
compute_er → baseline_zscore_er (gamma + broad bands)
       │
       ▼
calibrate_lambda_per_subject (Layer A, 持久化在 per_subject JSON)
       │
       ▼
compute_cusum_n_d_with_time   ← NEW (返回 frame_idx + t_onset_sec)
       │                              ↓
       │                       per_subject JSON
       │                       seizure_records[i].channel_onsets[ch]: t_onset_sec
       │                       (与 r_sz / r_sz_valid_count 同级，atlas 脚本只读)
       ▼
plot_ictal_er_atlas.py
   per-seizure mode → 单 PNG 双 band，从 raw + recomputed z-ER 画
   per-subject mode → 单 PNG，纯读 JSON 里 channel_onsets 矩阵
   cohort mode → 批量 16 subject
```

**关键约束**：per-seizure 模式必须从 raw 重算 z-ER（要画 heatmap 背景）；但 t_ER_onset 数值用 JSON 里持久化的（保证与 r_sz 计算一致）。Per-subject 模式只读 JSON，不碰 raw。

---

## 4. Per-seizure 双 band 图详细规格

### 4.1 布局

```
┌────────── GAMMA_ER (4-100 fast / 4-20 slow) ───────────┬─────────── BROAD_ER (10-200 / 4-20) ──────────┐
│  Row 1: RAW SEEG (focal + High-HI cluster traces)     │  Row 1: RAW SEEG (same channels)              │
│         - clinical onset vertical line at t=0         │                                                │
│         - eeg_onset dashed if exists                  │                                                │
│         - baseline edge (-60s) light gray             │                                                │
│         - y-scale shared with right column            │                                                │
├───────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
│  Row 2: Full-channel z-ER heatmap [-120, +30]s        │  Row 2: Full-channel z-ER heatmap (same window)│
│         - rows grouped by SOZ role                    │  (rows in SAME channel order as left)          │
│           focal-only → focal∩High-HI →                │                                                │
│           High-HI-only → other                        │                                                │
│         - within group sorted by t_ER_onset asc,      │                                                │
│           NaN at end                                  │                                                │
│         - y-tick color: red=SOZ, dark gray=High-HI,   │                                                │
│           gray=other                                  │                                                │
│         - per-row marker: ✦ at t_ER_onset             │                                                │
│           (no marker = onset_unreached)               │                                                │
│         - top-5 focal channel z-ER traces overlaid    │                                                │
│           as semi-transparent lines on top of heatmap │                                                │
│           (top-5 by gamma r_sz asc; broad column      │                                                │
│           overlays the SAME 5 channels)               │                                                │
└───────────────────────────────────────────────────────┴────────────────────────────────────────────────┘
                  [shared horizontal colorbar at bottom: z-ER ±3]
                  [legend at top right]
```

### 4.2 figsize / 几何

- 总幅 ~24 × 12 inch（宽栏一栏 ~11 inch，行高 raw=4 inch + heatmap=8 inch）
- DPI 150
- 左右两栏 channel-y 轴 100% 同步——同一 row index 在 gamma 和 broad 中都是同一物理 channel（关键 invariant，便于横扫双 band 差异）

### 4.3 颜色与符号

| 元素 | 编码 |
|---|---|
| z-ER heatmap | RdBu_r diverging，中点 0，clip ±3 |
| t_ER_onset marker | 白色 ✦ 带黑边描，画在 (t_ER_onset, channel_row) |
| clinical onset line | 黑色实线 lw=1.5 |
| eeg_onset line | 暗红虚线 lw=1.0 |
| baseline edge | 浅灰虚线 lw=0.8 |
| SOZ y-tick | 红色 (#c0392b) |
| High-HI y-tick | 深灰 (#34495e) |
| 普通 y-tick | 浅灰 (#95a5a6) |

### 4.4 输出文件名

`{dataset}_{subject_id}_seizure_{idx:02d}.png`，例 `epilepsiae_548_seizure_00.png`。

---

## 5. Per-subject 汇总图详细规格

### 5.1 布局

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ TITLE: epilepsiae/548  |  n_seizures=31  |  γ: stable=14 ok=15 unreached=11                     │
│                                              β: stable=18 ok=19 unreached=7                     │
├─────────┬─────────────────────────────────────────────────────────────────────────────┬─────────┤
│ ch tick │            t_ER_onset matrix — GAMMA_ER (channels × seizures)               │ cov γ   │
│ (color  │   ┌────────────────────────────────────────────────────────────────────┐    │ (bar    │
│  by SOZ │   │ HL3   c1   c2   _    c3   c4   ...   (cool=pre-clinical t<0)       │    │  to     │
│  role)  │   │ HL2   c5   c6   _    _    c7   ...                                 │    │  scale) │
│         │   │ TBA1  w1   w2   _    w3   w4   ...   (warm=post-clinical t>0)      │    │         │
│         │   │ ...   (sorted by gamma median r_sz asc, NaN at end)                │    │         │
│         │   └────────────────────────────────────────────────────────────────────┘    │         │
├─────────┼─────────────────────────────────────────────────────────────────────────────┼─────────┤
│         │            t_ER_onset matrix — BROAD_ER (same channel order)                │ cov β   │
│         │   (mirror layout, channels in same row order as gamma)                       │         │
├─────────┴─────────────────────────────────────────────────────────────────────────────┴─────────┤
│  seizure_idx →    0    1    2    3    4    ...    30                                            │
│  status strip:    ok   ok   ur   bi   ok   ...    tied   (color-coded: ok=green,                │
│                                                            ur=light gray, bi=black, tied=brown) │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                  [horizontal colorbar: t_ER_onset (-120 .. +30 s, RdBu_r)]
                  [side text annotation: producer_health + clinical_concordance per band]
```

### 5.2 排序规则

- **Channel y-axis**：按 gamma_ER 的 `r_sz` 升序排列（rsz=None 的 channel 排在最下，灰色 tick）
- **Seizure x-axis**：按 `seizure_idx` 升序（按发作时间顺序）
- **双 band 共用同一 channel 顺序**：方便横扫"gamma 早 vs broad 早"差异

### 5.3 颜色与符号

| 元素 | 编码 |
|---|---|
| t_ER_onset 矩阵格 | RdBu_r diverging，vmin=-120, vmax=+30, midpoint=0 |
| 未触发 (onset_unreached) | 浅灰填充 #d5d5d5 |
| seizure status ≠ ok | 整列覆盖斜纹 hatching |
| seizure status strip | ok=绿 #2ecc71 / unreached=浅灰 / tied=褐 #8e44ad / baseline_invalid=黑 / not_loaded=白 / boundary_skip=蓝 |
| cov 条 | 水平 bar，长度 = r_sz_valid_count / n_ok ∈ [0,1]，深绿 #27ae60 |
| y-tick 颜色 | 红 = SOZ ('i')，深灰 = High-HI cluster member，浅灰 = other |
| 矩阵 grid | 浅灰薄线分隔每 cell |

### 5.4 figsize / 几何

- 宽度：`max(12, 8 + 0.3 × n_seizures)` inch，clip 30
- 高度：`max(8, 0.18 × n_channels × 2 + 4)` inch（双 band 各占 0.18 × n_channels），clip 30
- 上下两块共享 x-axis（seizure_idx + status strip 在最底）
- 左侧 ch tick + 右侧 cov bar 双 band 共享

**边界 case：某 band n_ok=0**：不省略该 band 的子图块；仍画全灰矩阵（所有 cell 浅灰），并在子图标题加 `BROAD_ER (n_ok=0, no detection)` 红字标注。这样 16 subject 的 PNG 视觉结构保持一致，便于翻图。

### 5.5 输出文件名

`{dataset}_{subject_id}.png`，例 `epilepsiae_548.png`。

---

## 6. 接口契约

### 6.1 Augmented Layer A — `compute_cusum_n_d_with_time`

```python
@dataclass(frozen=True)
class CusumOnsetResult:
    frame_idx: Optional[int]    # 原 compute_cusum_n_d 输出
    t_onset_sec: Optional[float]  # = frame_idx * hop_sec + win_sec/2 - pre_sec, 或 None

def compute_cusum_n_d_with_time(
    z_er_1d: np.ndarray,
    lambda_thresh: float,
    *,
    bias: float = 0.5,
    detection_idx_window: Optional[Tuple[int, int]] = None,
    hop_sec: float,
    win_sec: float,
    pre_sec: float,
) -> CusumOnsetResult:
    ...
```

**契约要求**：
- 当 `detect_er_onset_preview` 返回 None 时，`frame_idx=None, t_onset_sec=None`
- 当返回有效 idx 时，`t_onset_sec = float(frame_idx) * hop_sec + win_sec / 2.0 - pre_sec`
- 对 NaN 处理逻辑与原 `compute_cusum_n_d` 完全一致（reset CUSUM）

### 6.2 Augmented Layer A persistence

`per_subject` JSON `seizure_records[i]` 新增字段：

```json
{
  "seizure_id": "...",
  "seizure_idx": 0,
  "status": "ok | onset_unreached | onset_tied | baseline_invalid | ...",
  "n_active": 10,
  "n_total": 84,
  "channel_onsets": {           // ← NEW
    "HL3": -34.2,
    "HL2": -22.0,
    "TBA1": +5.5,
    "HRA1": null,                // CUSUM 未触发
    ...
  },
  "schema_version": "v2.2.4"     // ← bumped from "v2.2.3"
}
```

`channel_onsets` 包含**所有** channel（包括 None），确保 channel coverage 与 r_sz 对得上。

### 6.3 atlas 脚本 CLI

```bash
# Single seizure
python scripts/plot_ictal_er_atlas.py per-seizure \
    --subject epilepsiae/548 --seizure-idx 0

# Single subject (all seizures + summary)
python scripts/plot_ictal_er_atlas.py per-subject \
    --subject epilepsiae/548

# Full epilepsiae cohort (16 subjects)
python scripts/plot_ictal_er_atlas.py cohort

# Optional flags
    --pre-sec 300 --post-sec 60   # window override (default matches Layer A)
    --time-window-sec=-120,30     # heatmap display window
    --skip-existing               # skip outputs already present
    --out-dir results/.../atlas
```

---

## 7. 实现拆分（writing-plans 之前的 step roadmap）

| Step | 内容 | 文件 | 行数估 |
|---|---|---|---|
| 1 | TDD: 单元测试 `compute_cusum_n_d_with_time` | `tests/test_ictal_er_rank.py` | +30 |
| 2 | 实现 `compute_cusum_n_d_with_time` | `src/ictal_er_rank.py` | +25 |
| 3 | Augment `_process_one_seizure_all_ers` 写 `channel_onsets` + bump `schema_version="v2.2.4"` | `scripts/run_ictal_er_rank.py` | +20 |
| 4 | 重跑 Layer A on 16 epilepsiae（`--cohort` 模式，`--skip-existing=False`） | bash | 0 |
| 5 | 验证 16 个 JSON 都升级到 v2.2.4 + cohort_summary 重生 | bash | 0 |
| 6 | TDD: 单元测试 atlas 数据装载 + 颜色映射 + sort | `tests/test_plot_ictal_er_atlas.py` | +60 |
| 7 | 新建 `scripts/plot_ictal_er_atlas.py` 框架 + `per-subject` mode | 新文件 | +250 |
| 8 | 新建 `per-seizure` mode（双 band raw + heatmap） | 同 | +200 |
| 9 | 新建 `cohort` 子模式 | 同 | +30 |
| 10 | 写 `results/data_driven_soz/layer_a_ictal_er_rank/atlas/figures/README.md`（中文） | 新文件 | +40 |
| 11 | Smoke-test：跑 epilepsiae/548 + epilepsiae/916 验证目视 | bash | 0 |
| 12 | 跑全 cohort + 视觉抽检 6 个 subject 类型代表（stable+concordant / stable+discordant / unstable+concordant / unstable+discordant / sentinel / 边界 case） | bash | 0 |

---

## 8. 接受门 (acceptance gates)

1. **代码门**：单元测试全过；`compute_cusum_n_d_with_time` 与原 `compute_cusum_n_d` 在所有 seed 下 frame_idx 一致（不允许行为漂移）
2. **JSON schema 门**：16 个 v2.2.4 JSON 全部包含 `channel_onsets` 且 keys 与 r_sz keys 完全相同
3. **Cohort 输出门**：16 个 per-subject PNG + 全部 per-seizure PNG 生成成功（无异常退出）
4. **视觉门**（用户人工检查）：
   - 583（stable+concordant model case）：focal channel 在 per-subject 矩阵上呈一致冷色（pre-clinical），nonfocal 暖色或灰
   - 916（stable+discordant）：focal channel 不集中在矩阵顶部，可视化能直接看出 data-driven vs clinical 的 mismatch
   - 548（unstable+concordant）：同 channel 跨 seizure 颜色高度漂移，验证 within-subject 变异性结论
   - 1084（focus_rel 全空，producer-only 评估）：仍可生成图，标题标明 "no clinical labels"

---

## 9. 风险与回滚

**风险**：
- Layer A 重跑 1-3 小时；若中断会留半完成 JSON。**缓解**：在 `_run_subject_all_ers` 加 `tempfile + atomic rename` 写入路径；现有写法是直接 `open(out_path, "w")`，需小改
- 每张 per-seizure PNG ~500 KB；16 subject × ~20 seizure × 1 PNG = ~160 MB；不进 git，加 `.gitignore` entry
- channel 数 epilepsiae 多在 60-130，per-subject 矩阵高度可能超过 30 inch；clip 到 30 后 channel-row 高度自适应缩

**回滚**：
- Step 1-3 出问题：JSON schema_version 仍是 v2.2.3，Atlas 脚本不读 v2.2.3 → 不会污染既有 archive 结论
- Step 4-5 出问题：删 v2.2.4 JSON 后从备份恢复 v2.2.3（备份在 `per_subject_lambda500/` 已有历史快照）
- Step 6-12 出问题：atlas 脚本独立，删 `atlas/` 目录即可

---

## 10. 决议日志

| 日期 | 决议 | 替代方案 |
|---|---|---|
| 2026-05-08 | per-subject + per-seizure 都做 | 仅做其中一种 |
| 2026-05-08 | per-subject 主体 = t_ER_onset 矩阵热图 | 多记录 z-ER 堆叠热图 / focal-vs-nonfocal 双点图 |
| 2026-05-08 | per-seizure 双 band 同 PNG（左 gamma / 右 broad） | 保持 1 PNG = 1 band |
| 2026-05-08 | per-seizure 删除中间 z-ER trace 行，仅留 raw + heatmap | 全留 / 删 raw / 删 heatmap |
| 2026-05-08 | 主时间窗 [-120, +30]s | [-200, +200]s / 主+inset |
| 2026-05-08 | 新建独立 atlas 脚本，不动 archive | 扩展 archive 脚本 / src/ helper 拆分 |
| 2026-05-08 | t_ER_onset 经 Layer A 持久化 | atlas 现算 / 中间 cache |
| 2026-05-08 | channel 排序按 gamma r_sz 升序，双 band 共用 | broad 主导 / SOZ 优先分组 |

---

## 11. 后续 PR（不在本设计内）

- Yuquan extract_seizure_window 实现 → 9 yuquan subject 补跑 → atlas 自动 pickup
- 若 atlas 视觉检查发现某种 ER-onset 时序模式（例如 focal channel 一致 pre-clinical 但 nonfocal post-clinical），可考虑独立 PR 把这个时序模式做成新的 data-driven SOZ proxy（与 r_sz 互补）
- atlas 上 cohort-level 视觉聚合（如把 16 个 per-subject 矩阵的 SOZ 行 z-stack）若有研究价值则单独 PR
