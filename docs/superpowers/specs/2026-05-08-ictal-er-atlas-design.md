# Ictal ER-onset Timing Atlas — Design (Layer A v2.3)

> **Topic**：PR-T3-1 v2.2 cohort 已确认 ER-rank 单指标无法作为整体 data-driven SOZ label，within-subject seizure-to-seizure 变异性是真问题。新 atlas 把 Layer A 检测窗从 [-5, +30]s **扩到 [-120, +30]s**（schema v2.3），用于回答 "ER-onset 是否走在 clinical / EEG onset 之前"，并交付 per-subject 矩阵 + per-seizure 双 band 改造图。
>
> **决议 (Brainstorm 2026-05-08, post-review revision)**：
> - 这是 **Layer A v2.3 数据合同变更**，不是只加字段；旧 v2.2 JSON 不能与新 v2.3 JSON 混用
> - 全 cohort 重跑；`r_sz / s_sz / producer_health / clinical_concordance` 数值会随检测窗变化
> - Atlas 仅是诊断工具，不升级为新 SOZ proxy
> - **Spec v2 (post-review)**：修正 9 处问题（详 §11 决议日志）

---

## 1. 背景与红线

PR-T3-1 v2.2 已经把 16 epilepsiae cohort 跑完，结论：
- ER-ratio 单指标不能作为整体 SOZ label（within-subject 变异性 → s_sz 低 → label 不稳）
- producer_health 与 clinical_concordance 是**两个独立诊断维度**，不能互替
- band disagreement（gamma vs broad）在 n_ok 边界小时尤其明显，atlas 是在**诊断**这个差异，不是在合并出"真 ER-ratio"

第二级问题：**ER-onset 时间是否走在 clinical / EEG onset 前？**——v2.2 数据回答不了，因为：
- v2.2 detection_window = `[-5s, +30s]`（`scripts/run_ictal_er_rank.py:89-90` 写死）
- CUSUM 从未在 `[-120s, -5s]` 段搜过；任何 Layer A 派生的 t_ER_onset 必然 ≥ -5s
- 在 v2.2 数据上画 `[-120, +30]s` 的 t_ER_onset atlas 是**科学撒谎**：用户会以为我们看到了 -40s 的提前，实际 producer 没搜过那段

⚠️ **红线**：要回答 onset timing，必须把 detection_window 从 `[-5, +30]` 扩到 `[-120, +30]`，**接受 v2.2 所有 r_sz / s_sz / tags 数值会变**。这是 v2.3，不是 v2.2.4。新旧不能混用。

---

## 2. 范围

**In scope**：
- Layer A 检测窗扩展 `[-5, +30] → [-120, +30]`（v2.3 重定义）
- 16 epilepsiae cohort（v2.2 audit_eligible 15 + 916 sentinel-only）全 cohort 重跑
- Per-channel `(frame_idx, t_onset_sec)` 双存进 `seizure_records[i].channel_onsets`
- 顶层 `schema_version: "pr_t3_1_layer_a_v2_3_timing"`
- Atomic write（temp + os.replace）—— **Step 0，先实现再重跑**
- Per-seizure 双 band 改造图（左 gamma / 右 broad，raw + heatmap，[-120, +30]s）
- Per-subject 汇总图（channel × seizure t_ER_onset 矩阵，双 band 上下两块）
- Cohort batch 模式
- 新脚本：`scripts/plot_ictal_er_atlas.py`
- 合成 step 硬测试（`-30s step → t_ER_onset ≈ -30s`，否则 atlas 判失败）

**Out of scope**：
- Yuquan 9 subject（依赖 yuquan 版本 `extract_seizure_window`，独立后续 PR）
- v2.2 JSON 兼容（直接覆盖；如需保留 v2.2 数值，备份到 `per_subject_v2_2/`）
- 任何对 r_sz / s_sz / producer_health 公式的修改（公式不动，**只改输入窗口**）
- 任何统计推断（atlas 是视觉诊断工具，不出 p 值）
- 旧 `_sentinel_step2/` 目录的 PNG 保持原样（archive）
- Per-seizure z-ER trace 中间行（已删，仅留 raw + heatmap）

---

## 3. 架构

### 3.1 输出根目录（修正：PNG 必须在 figures/ 下，符合 AGENTS.md）

```
results/data_driven_soz/layer_a_ictal_er_rank/
├── per_subject/                    ← v2.3 重跑后覆盖（v2.2 备份在 per_subject_v2_2/）
│   ├── epilepsiae_548.json
│   ├── ...
│   └── cohort_summary.json
├── per_subject_v2_2/               ← v2.3 重跑前的 v2.2 备份（rename 自 per_subject/）
│   └── (原 16 个 JSON)
└── atlas_v2_3/
    └── figures/
        ├── README.md               ← 必须，中文，符合 AGENTS.md §"结果目录标准"
        ├── per_subject/
        │   ├── epilepsiae_548.png
        │   ├── epilepsiae_916.png
        │   └── ...
        └── per_seizure/
            ├── epilepsiae_548_seizure_00.png
            └── ...
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
calibrate_lambda_per_subject (per subject; baseline FPR=1/h, λ_max=100)
       │
       ▼
DETECTION_PRE_SEC = 120.0   ← v2.3 (was 5.0 in v2.2)
DETECTION_POST_SEC = 30.0   ← unchanged
       │
       ▼
compute_cusum_n_d_with_time   ← NEW (返回 frame_idx + t_onset_sec, 双存)
       │                              ↓
       │                       per_subject JSON (v2.3 schema)
       │                       seizure_records[i].channel_onsets[ch] = {
       │                         "frame_idx": int|null,
       │                         "t_onset_sec": float|null
       │                       }
       │                       (与 r_sz / r_sz_valid_count 同级)
       ▼
plot_ictal_er_atlas.py
   per-seizure mode → 单 PNG 双 band，从 raw + recomputed z-ER 画
   per-subject mode → 单 PNG，纯读 JSON channel_onsets 矩阵
   cohort mode → 批量 16 subject
```

**关键约束**：
- per-seizure 模式：z-ER heatmap 背景从 raw 重算（无法预存全部帧），但每 channel 的 t_ER_onset 标记必须从 JSON 里取（不允许 atlas 脚本独立重算 CUSUM，否则与 r_sz 不一致）
- per-subject 模式：纯读 JSON，不碰 raw

---

## 4. Per-seizure 双 band 图详细规格

### 4.1 布局（修正：focal overlay 加 guard）

```
┌────────── GAMMA_ER (4-100 fast / 4-20 slow) ───────────┬─────────── BROAD_ER (10-200 / 4-20) ──────────┐
│  Row 1: RAW SEEG (overlay channels)                   │  Row 1: RAW SEEG (same channels)              │
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
│         - top-5 trace overlays (see overlay rule §4.4)│                                                │
└───────────────────────────────────────────────────────┴────────────────────────────────────────────────┘
                  [shared horizontal colorbar at bottom: z-ER ±3]
                  [legend at top right]
```

### 4.2 figsize / 几何

- 总幅 ~24 × 12 inch（宽栏一栏 ~11 inch，行高 raw=4 inch + heatmap=8 inch）
- DPI 150
- 左右两栏 channel-y 轴 100% 同步——同一 row index 在 gamma 和 broad 中都是同一物理 channel

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

### 4.4 Trace overlay 规则（修正：加 guard）

| 条件 | overlay 来源 | 标题标注 |
|---|---|---|
| `focal_channels` 非空 | top-5 focal channels by gamma `r_sz` 升序 | 默认（无特殊标注） |
| `focal_channels` 为空（如 1084） | top-5 producer channels by gamma `r_sz` 升序 | `"(no clinical labels — overlays = producer top-5)"` |
| 双 band 都 unstable / insufficient | producer top-5 by gamma `r_sz`（即便 unreliable） | `"(both bands unreliable — overlays may not represent real onset)"` |

**绝不**在没有 focal label 的 subject 上画 "focal" overlay。

### 4.5 输出文件名

`{dataset}_{subject_id}_seizure_{idx:02d}.png`，例 `epilepsiae_548_seizure_00.png`。

---

## 5. Per-subject 汇总图详细规格

### 5.1 布局（修正：sort_band 规则）

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ TITLE: epilepsiae/548  |  n_seizures=31  |  γ: stable=14 ok=15 unreached=11                     │
│                            β: stable=18 ok=19 unreached=7                                       │
│ sort_band = broad  (rule: stable > moderate > unstable; tie → gamma)                            │
├─────────┬─────────────────────────────────────────────────────────────────────────────┬─────────┤
│ ch tick │            t_ER_onset matrix — GAMMA_ER (channels × seizures)               │ cov γ   │
│ (color  │   ┌────────────────────────────────────────────────────────────────────┐    │ (bar    │
│  by SOZ │   │ HL3   c1   c2   _    c3   c4   ...   (cool=pre-clinical t<0)       │    │  to     │
│  role)  │   │ HL2   c5   c6   _    _    c7   ...                                 │    │  scale) │
│         │   │ TBA1  w1   w2   _    w3   w4   ...   (warm=post-clinical t>0)      │    │         │
│         │   │ ...   (sorted by sort_band median r_sz asc, NaN at end)            │    │         │
│         │   └────────────────────────────────────────────────────────────────────┘    │         │
├─────────┼─────────────────────────────────────────────────────────────────────────────┼─────────┤
│         │            t_ER_onset matrix — BROAD_ER (same channel order)                │ cov β   │
│         │   (mirror layout, channels in same row order as gamma)                       │         │
├─────────┴─────────────────────────────────────────────────────────────────────────────┴─────────┤
│  seizure_idx →    0    1    2    3    4    ...    30                                            │
│  status strip:    ok   ok   ur   bi   ok   ...    tied   (color-coded)                          │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                  [horizontal colorbar: t_ER_onset (-120 .. +30 s, RdBu_r)]
                  [side text: producer_health + clinical_concordance per band]
```

### 5.2 排序规则（修正：sort_band 选择规则）

```
sort_band selection (rank stable > moderate > unstable > insufficient):
  1. If exactly one band ∈ {stable, moderate}: sort_band = that band
  2. If both bands ∈ {stable, moderate}: sort_band = gamma (default, paper-canonical)
  3. If both bands ∈ {unstable, insufficient}:
       sort_band = gamma (fallback)
       title 标注 "(sort band unreliable — both bands {unstable|insufficient})"
  4. If one band insufficient: 仍按 (1) 选另一个
```

- **Channel y-axis**：按 `sort_band` 的 `r_sz` 升序排列（rsz=None 排底，灰 tick）
- **Seizure x-axis**：按 `seizure_idx` 升序（发作时间顺序）
- **双 band 共用同一 channel 顺序**：方便横扫双 band 差异

### 5.3 颜色与符号

| 元素 | 编码 |
|---|---|
| t_ER_onset 矩阵格 | RdBu_r diverging，vmin=-120, vmax=+30, midpoint=0 |
| 未触发 (onset_unreached, t_onset_sec=null) | 浅灰填充 #d5d5d5 |
| seizure status ≠ ok | 整列覆盖斜纹 hatching |
| seizure status strip | ok=绿 #2ecc71 / unreached=浅灰 / tied=褐 #8e44ad / baseline_invalid=黑 / not_loaded=白 / boundary_skip=蓝 |
| cov 条 | 水平 bar，长度 = `r_sz_valid_count / n_ok` ∈ [0,1]，深绿 #27ae60 |
| y-tick 颜色 | 红 = SOZ ('i')，深灰 = High-HI cluster member，浅灰 = other |
| 矩阵 grid | 浅灰薄线分隔每 cell |

### 5.4 figsize / 几何 + 边界 case

- 宽度：`max(12, 8 + 0.3 × n_seizures)` inch，clip 30
- 高度：`max(8, 0.18 × n_channels × 2 + 4)` inch（双 band 各占 0.18 × n_channels），clip 30
- 上下两块共享 x-axis（seizure_idx + status strip 在最底）
- 左侧 ch tick + 右侧 cov bar 双 band 共享

**边界 case：某 band n_ok=0**：仍画全灰矩阵 + 子图标题加 `BROAD_ER (n_ok=0, no detection)` 红字标注。视觉结构跨 16 subject 一致。

### 5.5 输出文件名

`{dataset}_{subject_id}.png`，例 `epilepsiae_548.png`。

---

## 6. 接口契约

### 6.1 v2.3 顶层 schema

```jsonc
{
  "schema_version": "pr_t3_1_layer_a_v2_3_timing",   // ← 顶层，全文件唯一
  "subject": "epilepsiae/548",
  "detection_window_sec": [-120.0, 30.0],            // ← 显式记录窗口（v2.3）
  "n_seizures_total": 31,
  "focal_channels": ["HL7", "HL8", ...],
  "per_er": { "gamma_ER": {...}, "broad_ER": {...} },
  "producer_health": {...},
  "clinical_concordance": {...}
}
```

### 6.2 `compute_cusum_n_d_with_time`（双存 frame_idx + sec）

**实现必须是字面 wrapper**（advisor 建议：no-drift 由结构保证，不靠测试）：

```python
@dataclass(frozen=True)
class CusumOnsetResult:
    frame_idx: Optional[int]      # CUSUM 第一次跨 λ 的帧索引；None=未触发
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
    """字面 wrapper of compute_cusum_n_d + 时间转换. 不复制循环体."""
    idx = compute_cusum_n_d(
        z_er_1d, lambda_thresh,
        bias=bias, detection_idx_window=detection_idx_window,
    )
    if idx is None:
        return CusumOnsetResult(None, None)
    t_sec = float(idx) * float(hop_sec) + float(win_sec) / 2.0 - float(pre_sec)
    return CusumOnsetResult(idx, t_sec)
```

**契约要求**：
- 实现严禁复制 `compute_cusum_n_d` 的循环体；必须直接调用它
- 当 `compute_cusum_n_d` 返回 None 时，`CusumOnsetResult(None, None)`
- 当返回有效 idx 时，`t_onset_sec = float(frame_idx) * hop_sec + win_sec / 2.0 - pre_sec`

### 6.3 `seizure_records[i].channel_onsets` 持久化格式

```jsonc
{
  "seizure_id": "...",
  "seizure_idx": 0,
  "status": "ok | onset_unreached | onset_tied | baseline_invalid | ...",
  "n_active": 10,
  "n_total": 84,
  "channel_onsets": {                     // ← NEW v2.3
    "HL3": { "frame_idx": 1132, "t_onset_sec": -36.7 },
    "HL2": { "frame_idx": 1320, "t_onset_sec": -17.9 },
    "TBA1": { "frame_idx": 1500, "t_onset_sec": +0.1 },
    "HRA1": { "frame_idx": null, "t_onset_sec": null }   // 未触发
  }
}
```

`channel_onsets` 必须包含**所有** channel（与 `r_sz` keys 等同），未触发 channel 显式 null（不允许 key 缺失）。

### 6.4 atomic write（Step 0）

```python
def _write_subject_json_atomic(out_path: Path, payload: dict) -> None:
    """Atomic write: temp file in same dir + os.replace.

    Why same dir: os.replace must be atomic, which requires temp + dest
    on the same filesystem.
    """
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, out_path)
```

替换 `scripts/run_ictal_er_rank.py` 中所有 `with open(out_path, "w")` 写入点。

**Stale .tmp 清理**（advisor 建议）：在 runner 任意 mode 启动时扫描 `output_dir / "*.json.tmp"` 并 unlink，避免上次中断留下的 .tmp 在后续 debug 中混淆视听。约 3 行。

### 6.5 atlas 脚本 CLI（修正：与 Layer A runner 风格对齐）

Layer A runner 用 `--sentinel / --per-subject / --no-skip-existing / --force / --output-dir`。Atlas 脚本采用 git 风格 subcommand（避免与 runner `--per-subject` 冲突）：

```bash
# Single seizure
python scripts/plot_ictal_er_atlas.py per-seizure \
    --subject epilepsiae/548 --seizure-idx 0

# Single subject (per-subject summary; without --include-seizures only the matrix)
python scripts/plot_ictal_er_atlas.py per-subject \
    --subject epilepsiae/548 [--include-seizures]

# Full epilepsiae cohort
python scripts/plot_ictal_er_atlas.py cohort [--include-seizures]

# Shared flags (match Layer A runner conventions where applicable):
    --no-skip-existing    # rerun even if PNG exists  (matches runner)
    --force               # overwrite + ignore lock files (matches runner)
    --output-dir PATH     # default = results/data_driven_soz/.../atlas_v2_3
    --schema-version-required pr_t3_1_layer_a_v2_3_timing
                          # hard-fail if loaded JSON doesn't match
```

`--schema-version-required` 拒绝在 v2.2 JSON 上跑（防 silent 误用旧数据）。

---

## 7. 实现拆分（writing-plans 之前的 step roadmap）

| Step | 内容 | 文件 | 行数估 |
|---|---|---|---|
| **0** | **atomic write 工具 + 替换所有 Layer A 写入点** | `scripts/run_ictal_er_rank.py` | +20 |
| 1 | TDD: 单元测试 `compute_cusum_n_d_with_time`（含合成 step at -30s 必返 ≈-30s 测试） | `tests/test_ictal_er_rank.py` | +60 |
| 2 | 实现 `compute_cusum_n_d_with_time`（薄壳 + dataclass） | `src/ictal_er_rank.py` | +30 |
| 3 | TDD: schema_version 顶层 + detection_window_sec 持久化测试 | `tests/test_ictal_er_rank.py` | +30 |
| 4 | Augment `_process_one_seizure_all_ers` 写 `channel_onsets`（双存）；改 `DETECTION_PRE_SEC = 120.0`；`_run_subject_all_ers` 输出 payload 顶层加 `schema_version` + `detection_window_sec` | `scripts/run_ictal_er_rank.py` | +30 |
| 5 | 一次性手动 rename `per_subject/ → per_subject_v2_2/`（保留 v2.2 备份），再 mkdir 新 `per_subject/` | bash one-liner | 0 |
| 5.5 | **Sentinel canary**：跑 `--sentinel`（548 + 916, ~10min）；硬验证 (a) 顶层 `schema_version` 存在 (b) `detection_window_sec == [-120, 30]` (c) `channel_onsets` 双键 dict (d) **至少一个 channel 的 `t_onset_sec < -5`**（证明窗口真扩展）。任一项失败立即停，不进 Step 6 | bash | 0 |
| 6 | 重跑 Layer A on 16 epilepsiae（`--per-subject`），1-3h | bash | 0 |
| 7 | 验证 16 个 v2.3 JSON 都包含 `schema_version` + `detection_window_sec=[-120,30]` + `channel_onsets` | bash one-liner | 0 |
| 8 | 新建 `scripts/plot_ictal_er_atlas.py` 框架 + 通用 helpers | 新文件 | +150 |
| 9 | TDD: per-subject sort_band 选择规则 + cell 颜色映射 + status strip 单元测试 | `tests/test_plot_ictal_er_atlas.py` | +80 |
| 10 | 实现 `per-subject` mode（双 band 矩阵 + status strip + cov bar） | 同 plot 脚本 | +200 |
| 11 | 实现 `per-seizure` mode（双 band raw + heatmap + onset markers + overlay guard） | 同 | +250 |
| 12 | 实现 `cohort` 子模式（include-seizures flag） | 同 | +30 |
| 13 | 写 `atlas_v2_3/figures/README.md`（中文） | 新文件 | +50 |
| 14 | Smoke: 跑 epilepsiae/548 + epilepsiae/916 视觉抽检 | bash | 0 |
| 15 | 跑全 cohort + 视觉门验证 6 case 类型代表 | bash | 0 |

---

## 8. 接受门 (acceptance gates)

### 8.1 代码门
- 单元测试全过
- `compute_cusum_n_d_with_time` 与 `compute_cusum_n_d` 在所有 seed 下 frame_idx **完全相同**（不允许行为漂移）

### 8.2 合成 step 硬测试 ⚠️ Linus 验收
```python
# 测试 1: step at t=-30s
result = compute_cusum_n_d_with_time(synthetic_z_step_at_neg30, lambda_thresh=10, ...)
assert result.t_onset_sec is not None
assert -32.0 <= result.t_onset_sec <= -28.0   # ≈ -30s ± 2s tolerance

# 测试 2: step at t=-100s（advisor 建议：仅 -30s 不能区分窗口扩到 -50 还是真扩到 -120）
result = compute_cusum_n_d_with_time(synthetic_z_step_at_neg100, lambda_thresh=10, ...)
assert result.t_onset_sec is not None
assert -102.0 <= result.t_onset_sec <= -98.0   # ≈ -100s ± 2s tolerance

# 测试 3: noise-only baseline（不应触发）
result = compute_cusum_n_d_with_time(synthetic_z_pure_noise, lambda_thresh=10, ...)
assert result.t_onset_sec is None and result.frame_idx is None

# 若任一失败 → 整套 atlas 判失败；尤其测试 2 失败说明窗口扩展不彻底
```

### 8.3 JSON schema 门
- 16 个 v2.3 JSON：
  - 顶层 `schema_version == "pr_t3_1_layer_a_v2_3_timing"`
  - 顶层 `detection_window_sec == [-120.0, 30.0]`
  - 每 `seizure_records[i].channel_onsets` keys ⊇ `per_er[band].r_sz` keys
  - 每 onset entry 是 `{"frame_idx": int|null, "t_onset_sec": float|null}` 双键

### 8.4 Cohort 输出门
- 16 个 per-subject PNG + 全部 per-seizure PNG（cohort `--include-seizures` 模式下）生成成功
- atlas 脚本 `--schema-version-required pr_t3_1_layer_a_v2_3_timing` 在 v2.3 JSON 上不报错；在 v2.2 JSON 上明确拒绝

### 8.5 视觉门（用户人工检查 6 个 case）
1. **583**（v2.2 `(stable+concordant)` model case）：focal channel 在 per-subject 矩阵呈一致冷色（pre-clinical），nonfocal 暖色或灰
2. **916**（v2.2 `(stable+discordant)`）：focal channel 不集中矩阵顶部，可视化能直接看出 data-driven vs clinical mismatch
3. **548**（v2.2 `(unstable+concordant)`）：同 channel 跨 seizure 颜色高度漂移，验证 within-subject 变异性结论
4. **1084**（`focal_channels=[]`）：仍可生成图；trace overlay 走 producer top-5 路径，标题标 "no clinical labels"
5. **139**（v2.2 `gamma_unstable+broad_concordant`）：sort_band 自动选 broad；channel 排序符合 broad r_sz
6. **任一 v2.3 数值大幅偏离 v2.2 的 case**：在 cohort_summary 里有 diff log，便于审阅"是否真的扩窗导致 r_sz 改变"

---

## 9. 风险与回滚

| 风险 | 缓解 |
|---|---|
| Layer A 重跑 1-3h，中断留半成品 JSON | Step 0 atomic write 已经强制；重跑前 dry-run 验证写路径 OK |
| v2.3 r_sz/s_sz/tags 显著偏离 v2.2，破坏既有 archive 结论 | 备份 v2.2 至 `per_subject_v2_2/`；archive doc 加 "v2.3 重跑后变化" 章节；明确说 v2.2 不再是 active 数据 |
| ~400 PNG × ~500 KB = 200 MB | 不进 git；`.gitignore` 加 `results/data_driven_soz/.../atlas_v2_3/` |
| epilepsiae channel 数 60-130，per-subject 矩阵高度可能溢出 30 inch | 高度 clip 到 30；channel-row 高度 = `min(0.18, 30 / (2 × n_ch + 4))` 自适应 |
| 扩展检测窗后 baseline 段被吃掉（baseline_end_sec 离 onset 太近） | `resolve_baseline_window` 已实现 `eeg_onset_rel_sec` clamping；额外加 assert：`baseline_end_sec <= -120s + buffer_sec`，否则 seizure 标 `baseline_invalid` |
| **Baseline squeeze**（advisor 警告）：`pre_sec=300` + `start_floor_sec=-120` + `buffer_sec=60` ⇒ baseline 仅剩 `[t_start, ~-180s]` ≈ 120s 而 v2.2 是 ~235s | sentinel canary (Step 5.5) 验证 548 baseline 仍 ≥ 80% valid；若 baseline_invalid 比例从 v2.2 暴涨，临时把 `pre_sec` 调到 360 重做 sentinel 再决策 |

**回滚路径**：
- Step 0-2 出问题：v2.2 JSON 未动，atlas 未生成 → 0 影响
- Step 4-7 出问题：删 v2.3 JSON，从 `per_subject_v2_2/` 恢复 → 完全回到 v2.2 状态
- Step 8-15 出问题：atlas 脚本独立，删 `atlas_v2_3/` 即可

---

## 10. atlas 是诊断工具，不是 SOZ proxy

⚠️ **README + 主文档必须明确**：

> Atlas v2.3 是回答"ER-onset 时间是否走在 clinical / EEG onset 之前？"的视觉诊断工具。它**不是**新的 data-driven SOZ label，**不能**用于：
> - 替代 clinical SOZ
> - 替代 r_sz / s_sz / producer_health 的统计判断
> - 在论文 figure 里作为"data-driven SOZ 早于 clinical onset"的直接证据
>
> Atlas 用途：挑选 case、识别 within-subject 变异性模式、识别 band disagreement 模式。任何升级到 SOZ proxy 的判断必须经独立 PR + 全 cohort 统计交叉验证。

---

## 11. 决议日志（含 review revision）

| 日期 | 决议 | 替代方案 / Review 修正 |
|---|---|---|
| 2026-05-08 brainstorm | per-subject + per-seizure 都做 | 仅做其中一种 |
| 2026-05-08 brainstorm | per-subject 主体 = t_ER_onset 矩阵热图 | 多记录 z-ER 堆叠 / focal-vs-nonfocal 双点图 |
| 2026-05-08 brainstorm | per-seizure 双 band 同 PNG（左 gamma / 右 broad） | 保持 1 PNG = 1 band |
| 2026-05-08 brainstorm | per-seizure 删除中间 z-ER trace 行，仅 raw + heatmap | 全留 / 删 raw / 删 heatmap |
| 2026-05-08 brainstorm | 主时间窗 [-120, +30]s | [-200, +200]s / 主+inset |
| 2026-05-08 brainstorm | 新建独立 atlas 脚本，不动 archive | 扩展 archive 脚本 / src/ helper 拆分 |
| 2026-05-08 brainstorm | t_ER_onset 经 Layer A 持久化 | atlas 现算 / 中间 cache |
| 2026-05-08 brainstorm | channel 排序按 sort_band r_sz 升序，双 band 共用 | 永远 gamma / broad 主导 / SOZ 优先 |
| **2026-05-08 review** | **检测窗 [-5, +30] → [-120, +30]，schema v2.3，r_sz/s_sz/tags 重算** | （v1）只加 channel_onsets 字段，windows 不动 — **科学撒谎，否决** |
| **2026-05-08 review** | **schema_version 顶层（不 per-record）+ detection_window_sec 显式持久化** | （v1）schema_version 写在每 seizure_records 里 — **冗余且错位** |
| **2026-05-08 review** | **channel_onsets 双存 frame_idx + t_onset_sec** | （v1）只存 t_onset_sec — **off-by-one 难审计** |
| **2026-05-08 review** | **atomic write 提前为 Step 0** | （v1）列在 §9 风险下面 — **重跑前必须先做** |
| **2026-05-08 review** | **PNG 输出搬到 atlas_v2_3/figures/{per_subject,per_seizure}/** | （v1）atlas/per_subject + atlas/figures/README.md — **违反 AGENTS.md** |
| **2026-05-08 review** | **CLI subcommand + 与 Layer A runner flag 对齐 (--no-skip-existing / --force)** | （v1）`--cohort / --skip-existing=False` — **flag 命名不一致** |
| **2026-05-08 review** | **sort_band = stable > moderate > unstable > insufficient（tie → gamma）** | （v1）永远 gamma — **broad 主导 case 排错** |
| **2026-05-08 review** | **focal overlay 加 guard：无 focal label 时改 producer top-5 + 标题标注** | （v1）硬画 "top-5 focal" — **1084 等无 label subject 撒谎** |
| **2026-05-08 review** | **加合成 step 硬测试 (-30s step → t_ER_onset ≈ -30s)** | （v1）无 — **detection_window 真扩窗无 hard-test 验证** |

---

## 12. 后续 PR（不在本设计内）

- Yuquan extract_seizure_window 实现 → 9 yuquan subject 补跑 → atlas 自动 pickup
- 若 atlas 视觉检查发现某种 ER-onset 时序模式（例 focal channel 一致 pre-clinical 但 nonfocal post-clinical），可考虑独立 PR 把这个时序模式做成新的 data-driven SOZ proxy（与 r_sz 互补）—— 但任何这种升级必须经独立全 cohort 交叉验证
- atlas 上 cohort-level 视觉聚合（如 16 个 per-subject 矩阵 SOZ 行 z-stack）若有研究价值则单独 PR
