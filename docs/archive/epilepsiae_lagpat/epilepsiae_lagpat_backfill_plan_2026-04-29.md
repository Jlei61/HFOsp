# Epilepsiae New-Pipeline pack+lagPat Production + Legacy-lagPat Sensitivity Audit — Implementation Plan

> 创建：2026-04-29
> 前序：
> - Yuquan dual-track audit 完成（`docs/archive/yuquan_lagpat/dual_track_audit_2026-04-26.md`，6/14 strict pass + 8 refine drift）
> - Epilepsiae artifact census 完成（`docs/archive/topic3/epilepsiae_artifact_census_2026-04-27.md`，0/20 legacy gpu replay-eligible，20/20 新 pipeline `_gpu.npz`+`_refineGpu.npz` ready）
> - Topic 3 PR-2 per-channel relaxed-refine 完成（`docs/archive/topic3/epilepsiae_three_tier_pr2_2026-04-27.md`），不依赖 lagPat。
>
> **本计划不是 Track B replay**（缺老 gpu 冻结条件，不可能 strict numerical parity）。
> **本计划不是数据覆盖**（绝不写回 `/mnt/epilepsia_data/.../all_recs/`）。
> **本计划是**：用新 pipeline `_gpu.npz` + `_refineGpu.npz` 生产新 packedTimes + lagPat，输出到 `results/epilepsiae_lagpat_backfill/`，再与老 lagPat 做结构 + 下游敏感性比较。

## 0. 范围与不做什么

### 做
- **Stage A**：脚本骨架 + 输出目录约定，只读不覆盖
- **Stage B**：Epilepsiae-aware pack + lagPat 生产（CAR / variable sfreq / `.data+.head` schema）
- **Stage C**：新 vs 老 lagPat 结构对比（5 维：chn set / packedTimes / eventsBool / lagPatRaw span / lagPatRank cluster similarity）
- **Stage D**：下游敏感性（用新 lagPat 重跑 interictal_propagation PR-1 + interictal_synchrony PR4–PR6 子集）
- **Stage E**：文档落档（archive contract + Topic 1/3 主文档增补 + AGENTS.md 索引）

### 不做
- ❌ 声明数值 strict parity（缺老 gpu，不可能）
- ❌ 移植 Yuquan 5 处硬编码（bipolar reref / 800 Hz resample / 单 EDF / `.legacy_backup` / `/mnt/yuquan_data` 根）
- ❌ 改写 `/mnt/epilepsia_data/.../interilca_inter_results/`（老 lagPat 是 Topic 1 当前合同的一部分，**绝不**覆盖）
- ❌ 重做 detector / refine（已就位）
- ❌ 在 Stage D 主张"新 lagPat 替代老 lagPat"——只验证下游结论方向稳不稳

### Hypothesis tier 预声明
| 比较 | tier | 用途 |
|---|---|---|
| Stage C 结构相似度 | **descriptive sensitivity** | 报告新老差异程度，不做单调假设检验 |
| Stage D propagation stable_k 分布 | **secondary sensitivity** | 验证 PR-2 接受口径是否依赖老 lagPat |
| Stage D synchrony phase Pre/Post | **secondary sensitivity** | 验证 PR-6 null 是否依赖老 lagPat |
| 任何 metric 方向反转 | **flag for archive** | 不直接推翻 Topic 1 现有结论；触发独立讨论 |

---

## 1. Phase 1 — Orientation 已完成（不重做）

以下事实已通过 Phase 1 Explore + 本 plan 起草前的 Bash 验证确认，**不需要再扫盘**：

### 1.1 sfreq 分布（per-block，20/20 subject 全部 survey 完毕）

| 类型 | subjects | 数量 |
|---|---|---|
| 纯 1024 Hz | `1073, 1077, 1084, 1096, 1125, 1146, 1150, 442, 548, 590, 620, 635, 818, 916, 922, 958` | 16 |
| 纯 512 Hz | `253` (268 blocks) | 1 |
| 混合 256+512 | `139` (43@256 + 130@512) | 1 |
| 混合 1024+256 | `384` (65@1024 + 65@256), `583` (63@1024 + 143@256) | 2 |

256 Hz blocks 在 `scripts/run_hfo_detection.py` 中被 Nyquist gate 跳过（`src/hfo_detector.py:168` raises `ValueError` 当 `sfreq/2 < 250` for ripple band）。`results/hfo_detection/<subject>/*_gpu.npz` 的 record 数等于"非-256 blocks 数"：
- `139` → 130 records（仅 512 Hz blocks）
- `384` → 65 records（仅 1024 Hz blocks）
- `583` → 63 records（仅 1024 Hz blocks）

**含义**：新 pipeline lagPat record set ⊊ 老 pipeline lagPat record set。Stage C 比较必须先做 record-set 对齐，不能假设 1:1。

### 1.2 关键代码就位
| 函数 | 位置 | 用途 |
|---|---|---|
| `load_epilepsiae_block(...)` | `src/preprocessing.py:452` | CAR signal loader (.data+.head) |
| `_read_epilepsiae_head_for_streaming(...)` | `src/preprocessing.py:423` | 解析 UTF-16 .head 取 sfreq/n_channels/duration |
| `EpilepsiaePaths` + `_collect_raw_blocks` | `src/epilepsiae_dataset.py:75/334` | subject → raw block stems 映射 |
| `compute_centroid_matrix_spectrogram(...)` | `src/group_event_analysis.py:2145` | 通用 spectrogram centroid，dataset-agnostic |
| `lag_rank_from_centroids(...)` | `src/group_event_analysis.py:2676` | rank via argsort(argsort)，dataset-agnostic |
| `build_windows_from_detections(...)` | `src/group_event_analysis.py:389` | 从 per-channel dets 建 group event windows |
| `legacy_refine_counts_from_detection_sets(...)` | `src/group_event_analysis.py:912` | refine logic（已被 `run_hfo_detection.py` 调用过，新 `_refineGpu.npz` 已就位，本 plan **直接读 refine 通道列表**，不再 refine） |
| `save_refine_gpu_npz(...)` | `src/group_event_analysis.py:1074` | 不本计划复用：refine 已完成 |

### 1.3 老 lagPat 文件清单（每 subject 全部 records 对齐 raw blocks）
- 路径：`/mnt/epilepsia_data/interilca_inter_results/all_data_lns/<subject>/all_recs/<stem>_lagPat.npz`
- 20/20 subject 完整（含 256 Hz blocks 也有 lagPat）
- key 集：`lagPatRaw, lagPatRank, eventsBool, chnNames, start_t`，缺 `block_time_ranges`

---

## 2. Stage A — 脚本骨架 + 输出目录约定

**目标**：建立目录、CLI、读路径但不写任何 lagPat。Smoke 必须能跑 1 subject 1 record 并 dry-print sfreq + chn names + det count。

### 输出目录约定（必须遵守 AGENTS.md 命名规范）

```
results/epilepsiae_lagpat_backfill/
├── <subject>/
│   ├── <stem>_packedTimes.npy
│   ├── <stem>_lagPat.npz         (chnNames, lagPatRaw, lagPatRank, eventsBool, start_t)
│   └── _backfill_log.json        (per-record stats: sfreq, n_dets, n_events, n_picked_chns, errors)
├── cohort_summary.csv            (per-subject: n_records_done, n_skipped_nyquist, n_failed)
└── README.md                     (中文，按 Results Directory Standards)
```

> **绝不**输出到 `/mnt/epilepsia_data/...`。CI 应该有断言挡住误写。

### Files
- Create: `scripts/run_epilepsiae_lagpat_backfill.py`
- Create: `tests/test_epilepsiae_lagpat_backfill.py`
- Reuse: `src/preprocessing.py::load_epilepsiae_block`
- Reuse: `src/epilepsiae_dataset.py::_collect_raw_blocks`
- Reuse: `src/group_event_analysis.py::compute_centroid_matrix_spectrogram, lag_rank_from_centroids, build_windows_from_detections`

### Task A.1 — 脚本骨架 + 路径常量

- [ ] **Step 1**：写失败测试 `tests/test_epilepsiae_lagpat_backfill.py::test_output_dir_constant_is_results_subtree`

```python
def test_output_dir_constant_is_results_subtree():
    from scripts.run_epilepsiae_lagpat_backfill import OUTPUT_ROOT
    assert OUTPUT_ROOT == Path("results/epilepsiae_lagpat_backfill")
    # 必须在 results/ 下，绝不指向 /mnt
    assert "/mnt" not in str(OUTPUT_ROOT.resolve())
```

- [ ] **Step 2**：跑测试确认 fail（脚本不存在）

```bash
pytest tests/test_epilepsiae_lagpat_backfill.py::test_output_dir_constant_is_results_subtree -v
```

预期：FAIL with `ModuleNotFoundError`

- [ ] **Step 3**：实现最小骨架

```python
# scripts/run_epilepsiae_lagpat_backfill.py
"""Epilepsiae new-pipeline pack + lagPat backfill driver.

Reads:  results/hfo_detection/<subject>/*_gpu.npz      (whole_dets, chns_names, start_time)
        results/hfo_detection/<subject>/*_refineGpu.npz (refined channel list)
        Raw .data + .head via load_epilepsiae_block      (CAR signal, variable sfreq)
Writes: results/epilepsiae_lagpat_backfill/<subject>/<stem>_packedTimes.npy
        results/epilepsiae_lagpat_backfill/<subject>/<stem>_lagPat.npz
        results/epilepsiae_lagpat_backfill/<subject>/_backfill_log.json

NOT a Track B replay: legacy *_gpu.npz are 216-byte stubs (per artifact census 2026-04-27).
Output is a NEW pack/lag artifact for sensitivity-audit purposes only.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import argparse
import json
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.epilepsiae_dataset import EpilepsiaePaths, _collect_raw_blocks
from src.preprocessing import _read_epilepsiae_head_for_streaming

NEW_GPU_ROOT = Path("results/hfo_detection")
OUTPUT_ROOT = Path("results/epilepsiae_lagpat_backfill")
RIPPLE_BAND = (80.0, 250.0)
NYQUIST_GATE_HZ = 2.0 * RIPPLE_BAND[1]  # = 500 Hz; 256 Hz blocks fail this
SEGMENT_SEC = 200.0  # mirrors Yuquan stitched-segment legacy semantics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="Process first record of first subject only; dry-print, no writes")
    args = parser.parse_args()
    raise NotImplementedError("Stage A.1 骨架")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4**：跑测试确认 pass

```bash
pytest tests/test_epilepsiae_lagpat_backfill.py::test_output_dir_constant_is_results_subtree -v
```

预期：PASS

- [ ] **Step 5**：commit

```bash
git add scripts/run_epilepsiae_lagpat_backfill.py tests/test_epilepsiae_lagpat_backfill.py
git commit -m "feat(epilepsiae_lagpat): Stage A.1 — backfill driver skeleton + output path contract"
```

### Task A.2 — 添加 record discovery + smoke dry-print

- [ ] **Step 1**：写失败测试

```python
def test_smoke_lists_first_record(tmp_path, capsys):
    # uses real 253 subject (smallest sfreq=512 single-mode subject for fast smoke)
    from scripts.run_epilepsiae_lagpat_backfill import _discover_records
    recs = _discover_records("253")
    assert len(recs) >= 1
    first = recs[0]
    assert first["stem"].startswith("253")
    assert first["sfreq"] in (256.0, 512.0, 1024.0)
    assert first["new_gpu_path"].exists()
    assert first["raw_data_path"].exists() and first["raw_head_path"].exists()
```

- [ ] **Step 2**：跑测试确认 fail

- [ ] **Step 3**：实现 `_discover_records`

```python
def _discover_records(subject: str) -> List[Dict]:
    """Cross-reference results/hfo_detection/<subject>/*_gpu.npz with raw .data/.head.

    Returns list of dicts: stem, sfreq, new_gpu_path, raw_data_path, raw_head_path.
    Records where raw .data/.head are missing are skipped with a warning (not error)
    because new pipeline already filtered Nyquist-failing blocks.
    """
    paths = EpilepsiaePaths()
    raw_blocks = _collect_raw_blocks(subject, paths)  # stem -> RawBlockFiles
    new_gpu_dir = NEW_GPU_ROOT / subject
    if not new_gpu_dir.exists():
        raise FileNotFoundError(f"No new-pipeline gpu dir: {new_gpu_dir}")
    out = []
    for gpu_path in sorted(new_gpu_dir.glob("*_gpu.npz")):
        stem = gpu_path.stem.replace("_gpu", "")
        if stem.endswith("_refineGpu") or stem.startswith("sub_"):
            continue
        if stem not in raw_blocks:
            print(f"  WARN: {stem} has new gpu but no raw .data/.head; skip")
            continue
        head_info = _read_epilepsiae_head_for_streaming(raw_blocks[stem].head_path)
        sfreq = float(head_info.get("sample_freq", 0))
        out.append({
            "stem": stem,
            "sfreq": sfreq,
            "new_gpu_path": gpu_path,
            "raw_data_path": raw_blocks[stem].data_path,
            "raw_head_path": raw_blocks[stem].head_path,
        })
    return out
```

- [ ] **Step 4**：smoke 命令

```bash
python scripts/run_epilepsiae_lagpat_backfill.py --subject 253 --smoke
```

预期 stdout：列出第一个 record 的 stem, sfreq=512.0, n_dets, n_chns。无任何文件被写入 `OUTPUT_ROOT`。

- [ ] **Step 5**：commit

```bash
git add scripts/run_epilepsiae_lagpat_backfill.py tests/test_epilepsiae_lagpat_backfill.py
git commit -m "feat(epilepsiae_lagpat): Stage A.2 — record discovery + smoke dry-print"
```

---

## 3. Stage B — Epilepsiae-aware pack + lagPat 生产

**目标**：单 record 端到端跑通 `_gpu.npz + _refineGpu.npz + raw signal → packedTimes + lagPat`，schema 与老 lagPat key 集完全对齐（chnNames/lagPatRaw/lagPatRank/eventsBool/start_t）。

### Task B.1 — refine 通道列表读取（subject-level，一次加载全 subject 复用）

> **修正**：新 pipeline 实际产物是 `results/hfo_detection/<subject>/_refineGpu.npz`（**subject-level**，没有 `<stem>_` 前缀，也没有 `sub_` 前缀）。Schema：`{chns_names, events_count}`。已验证：
> ```
> $ python -c "import numpy as np; z=np.load('results/hfo_detection/253/_refineGpu.npz', allow_pickle=True); print(sorted(z.files))"
> ['chns_names', 'events_count']
> ```
> 一个 subject 内所有 record 共用同一个 refined channel list。**Cache once, reuse across all records**。

- [ ] **Step 1**：写测试

```python
def test_load_refine_chns_for_subject_returns_subject_level():
    from scripts.run_epilepsiae_lagpat_backfill import load_refine_chns_for_subject
    chns = load_refine_chns_for_subject("253")
    assert isinstance(chns, list)
    assert len(chns) > 0
    assert all(isinstance(c, str) for c in chns)
    # subject-level: same list returned regardless of record context
    chns2 = load_refine_chns_for_subject("253")
    assert chns == chns2

def test_refine_path_is_subject_not_per_record():
    from scripts.run_epilepsiae_lagpat_backfill import _refine_path_for_subject
    p = _refine_path_for_subject("253")
    assert p.name == "_refineGpu.npz"  # NOT "253_refineGpu.npz", NOT "sub_refineGpu.npz"
    assert p.parent.name == "253"
```

- [ ] **Step 2**：跑测试 fail

- [ ] **Step 3**：实现

```python
from functools import lru_cache

def _refine_path_for_subject(subject: str) -> Path:
    return NEW_GPU_ROOT / subject / "_refineGpu.npz"

@lru_cache(maxsize=32)
def load_refine_chns_for_subject(subject: str) -> Tuple[str, ...]:
    """Read subject-level refined channel names (cached per subject).

    Returns tuple (immutable) so lru_cache is safe; convert to list at call site
    if mutation needed.
    """
    refine_path = _refine_path_for_subject(subject)
    if not refine_path.exists():
        raise FileNotFoundError(
            f"Subject-level refine artifact missing: {refine_path}\n"
            f"Expected file produced by scripts/run_hfo_detection.py."
        )
    z = np.load(refine_path, allow_pickle=True)
    return tuple(str(c) for c in z["chns_names"])
```

- [ ] **Step 4**：测试 pass

- [ ] **Step 5**：commit

### Task B.2 — Pack: 从 whole_dets 建 group event windows

> **`whole_dets` 单位 = seconds [start, end]**。已验证：`src/hfo_detector.py:82` 注释 `Each element: array shape (n_events, 2) in seconds [start, end]`，及 `:224` `Events per channel, each shape (n_events, 2) [start, end] in seconds.`。**默认实现 NOT divide by 1000 / NOT divide by sfreq**。仅在测试中保留 probe 用于防御未来 schema drift。
>
> **`build_windows_from_detections` API 修正**：
> - 实际签名：`build_windows_from_detections(detections, window_sec=0.5, *, ext_ms=30.0, chns_thr=0.5, time_axis_hz=500.0, max_window_sec=2.0, t_max_sec=None) -> List[EventWindow]`
> - **不存在** `min_n_channels` 参数；用 `chns_thr=0.5`（fraction of active picked channels）
> - `EventWindow` 字段：`start, end, event_id`（**不是** `t_start, t_end`）

- [ ] **Step 1**：先跑 probe（不是测试，是手工 sanity check）

```bash
python -c "
import numpy as np
from pathlib import Path
import sys; sys.path.insert(0, '.')
from src.epilepsiae_dataset import EpilepsiaePaths, _collect_raw_blocks
from src.preprocessing import _read_epilepsiae_head_for_streaming

z = np.load('results/hfo_detection/253/25300102_0000_gpu.npz', allow_pickle=True)
paths = EpilepsiaePaths()
blocks = _collect_raw_blocks('253', paths)
head = _read_epilepsiae_head_for_streaming(blocks['25300102_0000'].head_path)
duration_sec = float(head['duration_in_sec'])
sfreq = float(head['sample_freq'])
print(f'duration={duration_sec:.1f}s, sfreq={sfreq}')
for arr in z['whole_dets']:
    a = np.asarray(arr)
    if a.size == 0: continue
    mx = float(a.max())
    print(f'whole_dets max value = {mx:.3f}')
    if mx < duration_sec * 1.5:
        print('  → SECONDS (matches hfo_detector.py:82 contract)')
    elif mx < duration_sec * sfreq * 1.5:
        print('  → SAMPLES (would mean contract drift, raise alarm)')
    else:
        print('  → UNKNOWN — investigate before continuing')
    break
"
```

预期输出：`SECONDS`. 若 SAMPLES，**停止本 plan**，先修 `src/hfo_detector.py` 写入合同。

- [ ] **Step 2**：写测试（hard-coded contract assertion + functional test）

```python
def test_pack_record_returns_packed_times_2d():
    from scripts.run_epilepsiae_lagpat_backfill import pack_record
    pt = pack_record("253", "25300102_0000")
    assert pt.ndim == 2
    assert pt.shape[1] == 2  # (n_events, 2) [start, end_sec]
    assert (pt[:, 1] >= pt[:, 0]).all()
    # contract: max event time < block duration (≤ ~3600 sec for 1h block)
    assert pt[:, 1].max() < 4000.0  # SECONDS, not samples

def test_whole_dets_units_are_seconds():
    """Defensive: probe whole_dets max value is in seconds, not samples."""
    z = np.load("results/hfo_detection/253/25300102_0000_gpu.npz", allow_pickle=True)
    for arr in z["whole_dets"]:
        a = np.asarray(arr)
        if a.size == 0: continue
        # 1h block at 512 Hz: samples up to ~1.84M; seconds up to ~3600
        # If max > 4000, contract drifted (probably samples now)
        assert float(a.max()) < 4000.0, "whole_dets unit drift detected"
        break
```

- [ ] **Step 3**：跑测试 fail

- [ ] **Step 4**：实现

```python
def pack_record(subject: str, stem: str) -> np.ndarray:
    """Produce (n_events, 2) packed [start_sec, end_sec] times for a single record.

    whole_dets contract: each element shape (n_dets, 2) in SECONDS [start, end]
    relative to record start. (See src/hfo_detector.py:82 / :224.)
    """
    from src.group_event_analysis import build_windows_from_detections

    gpu_path = NEW_GPU_ROOT / subject / f"{stem}_gpu.npz"
    z = np.load(gpu_path, allow_pickle=True)
    chns_names = [str(c) for c in z["chns_names"]]
    whole_dets = z["whole_dets"]  # object array; entries already in seconds

    refine_chns_set = set(load_refine_chns_for_subject(subject))

    detections: Dict[str, np.ndarray] = {}
    for i, ch in enumerate(chns_names):
        if ch not in refine_chns_set:
            continue
        arr = np.atleast_2d(np.asarray(whole_dets[i], dtype=float))
        if arr.size == 0:
            continue
        detections[ch] = arr  # NO unit conversion — already in seconds

    if not detections:
        return np.empty((0, 2), dtype=float)

    windows = build_windows_from_detections(
        detections,
        window_sec=0.5,    # ripple group-event window
        chns_thr=0.5,      # fraction of active picked channels (legacy)
        ext_ms=30.0,       # legacy default
        time_axis_hz=500.0,
    )
    if not windows:
        return np.empty((0, 2), dtype=float)
    return np.array([(w.start, w.end) for w in windows], dtype=float)
```

- [ ] **Step 5**：测试 pass

- [ ] **Step 6**：commit

### Task B.3 — Per-record lagPat 计算

> **API 修正集中说明**：
> - `load_epilepsiae_block(data_path, head_path, *, reference="car", ...)` 返回 **`PreprocessingResult`**（dataclass），**不是** `(data, ch_names)` tuple。访问字段：`pre.data, pre.sfreq, pre.ch_names, pre.start_time`。
> - `pre.start_time` 已经是 Unix epoch float（在 `load_epilepsiae_block` 内部用 `datetime.strptime(raw_info["start_ts"], "%Y-%m-%d %H:%M:%S.%f")` 解析），**不要**自己再 `float(_read_epilepsiae_head_for_streaming(blk.head_path)["start_ts"])`，那样会拿到 datetime / 字符串。
> - `EventWindow` 字段：`start, end, event_id`。
> - `compute_centroid_matrix_spectrogram` 返回 `Tuple[np.ndarray, np.ndarray]`（centroids, events_bool）。

- [ ] **Step 1**：写测试

```python
def test_compute_lagpat_record_shapes():
    from scripts.run_epilepsiae_lagpat_backfill import compute_lagpat_record
    out = compute_lagpat_record("253", "25300102_0000")
    assert set(out.keys()) >= {"lagPatRaw", "lagPatRank", "eventsBool", "chnNames", "start_t"}
    n_pick = len(out["chnNames"])
    n_ev = out["lagPatRaw"].shape[1]
    assert out["lagPatRaw"].shape == (n_pick, n_ev)
    assert out["lagPatRank"].shape == (n_pick, n_ev)
    assert out["eventsBool"].shape == (n_pick, n_ev)
    # start_t is Unix epoch (2008–2012 range for Epilepsiae 253)
    assert 1e9 < out["start_t"] < 2e9
```

- [ ] **Step 2**：跑测试 fail

- [ ] **Step 3**：实现 — Epilepsiae-specific simplified pipeline

```python
def compute_lagpat_record(subject: str, stem: str) -> Dict[str, np.ndarray]:
    """End-to-end per-record pack + lagPat for Epilepsiae new pipeline.

    NO Yuquan-style assumptions:
    - NO bipolar reref / drop (CAR already applied by load_epilepsiae_block)
    - NO 800 Hz resample (variable sfreq; use as-is)
    - NO single-EDF assumption (one .data/.head per record)
    - NO .legacy_backup writes (output to results/ only)
    """
    from src.preprocessing import load_epilepsiae_block
    from src.group_event_analysis import (
        EventWindow,
        compute_centroid_matrix_spectrogram,
        lag_rank_from_centroids,
    )
    from scipy.signal import butter, filtfilt

    paths = EpilepsiaePaths()
    raw_blocks = _collect_raw_blocks(subject, paths)
    blk = raw_blocks[stem]

    # 1. Load CAR signal — returns PreprocessingResult (dataclass)
    pre = load_epilepsiae_block(
        data_path=blk.data_path,
        head_path=blk.head_path,
        reference="car",
        # notch_freqs default already strips 50/100/150/200 Hz
    )
    sfreq = float(pre.sfreq)
    start_t_epoch = float(pre.start_time)  # Unix epoch, parsed inside load_epilepsiae_block
    sig_data = pre.data                    # (n_channels, n_samples)
    ch_names_full = list(pre.ch_names)

    # Nyquist defensive gate (detector already filters; double-check)
    if sfreq < NYQUIST_GATE_HZ:
        raise ValueError(
            f"sfreq {sfreq} < {NYQUIST_GATE_HZ} (Nyquist for ripple band) "
            f"— this record should have been skipped upstream"
        )

    # 2. Filter to refined channels (subject-level cache)
    refine_chns_set = set(load_refine_chns_for_subject(subject))
    keep_idx = [i for i, c in enumerate(ch_names_full) if c in refine_chns_set]
    if len(keep_idx) < 3:
        return _empty_lagpat_record(start_t_epoch)
    sig_pick = sig_data[keep_idx]
    pick_names = [ch_names_full[i] for i in keep_idx]

    # 3. Bandpass to ripple band
    nyq = 0.5 * sfreq
    b, a = butter(4, [RIPPLE_BAND[0]/nyq, RIPPLE_BAND[1]/nyq], btype="band")
    sig_band = filtfilt(b, a, sig_pick, axis=-1)

    # 4. Pack windows (already in seconds; see Task B.2)
    packed_times = pack_record(subject, stem)
    if packed_times.size == 0:
        return _empty_lagpat_record(start_t_epoch)

    # 5. Build per-channel detections dict (refine-filtered, in seconds)
    z_gpu = np.load(NEW_GPU_ROOT / subject / f"{stem}_gpu.npz", allow_pickle=True)
    gpu_chns = [str(c) for c in z_gpu["chns_names"]]
    detections: Dict[str, np.ndarray] = {}
    for i, ch in enumerate(gpu_chns):
        if ch not in refine_chns_set:
            continue
        arr = np.atleast_2d(np.asarray(z_gpu["whole_dets"][i], dtype=float))
        if arr.size > 0:
            detections[ch] = arr  # SECONDS

    # 6. Per-segment centroid (200s legacy segment semantic)
    n_pick = len(pick_names)
    n_ev = packed_times.shape[0]
    centroids = np.full((n_pick, n_ev), np.nan, dtype=float)
    events_bool = np.zeros((n_pick, n_ev), dtype=float)

    duration_sec = sig_band.shape[1] / sfreq
    seg_starts = np.arange(0.0, duration_sec, SEGMENT_SEC)
    seg_starts = np.append(seg_starts, duration_sec)

    for s0, s1 in zip(seg_starts[:-1], seg_starts[1:]):
        in_seg = np.where((packed_times[:, 0] >= s0) & (packed_times[:, 1] <= s1))[0]
        if in_seg.size == 0:
            continue
        i0, i1 = int(s0 * sfreq), int(s1 * sfreq)
        seg_band = sig_band[:, i0:i1]
        if seg_band.shape[1] < int(0.05 * sfreq):
            continue
        seg_windows = [
            EventWindow(
                start=float(packed_times[i, 0]),
                end=float(packed_times[i, 1]),
                event_id=int(i),
            )
            for i in in_seg
        ]
        seg_centroids, seg_evbool = compute_centroid_matrix_spectrogram(
            windows=seg_windows,
            detections=detections,
            ch_names=pick_names,
            x_band=seg_band,
            sfreq=sfreq,
            start_sec=float(s0),
        )
        for col_in_seg, ev_idx in enumerate(in_seg):
            centroids[:, ev_idx] = seg_centroids[:, col_in_seg]
            events_bool[:, ev_idx] = seg_evbool[:, col_in_seg]

    lag_raw, lag_rank = lag_rank_from_centroids(
        centroids, events_bool, align="first_centroid",
    )

    return {
        "lagPatRaw": lag_raw.astype(np.float64),
        "lagPatRank": lag_rank.astype(np.int64),
        "eventsBool": events_bool.astype(np.float64),
        "chnNames": np.array(pick_names),
        "start_t": np.float64(start_t_epoch),
    }


def _empty_lagpat_record(start_t_epoch: float) -> Dict[str, np.ndarray]:
    return {
        "lagPatRaw": np.empty((0, 0), dtype=np.float64),
        "lagPatRank": np.empty((0, 0), dtype=np.int64),
        "eventsBool": np.empty((0, 0), dtype=np.float64),
        "chnNames": np.array([], dtype=str),
        "start_t": np.float64(start_t_epoch),
    }
```

> **决策点 B.3.a**：是否输出 `block_time_ranges`？老 lagPat **没有这个 key**（census 已确认）。新 lagPat 也不输出，保持向后兼容。
> **决策点 B.3.b**：256 Hz blocks 已在 `results/hfo_detection/` 中被 detector 跳过；本脚本不会处理。
> **决策点 B.3.c**：`load_epilepsiae_block` 已经做了 CAR + notch (50/100/150/200 Hz)，本脚本只在 ripple band [80, 250] 再加 band-pass，**不**重复 reref。

- [ ] **Step 4**：跑测试 pass

- [ ] **Step 5**：commit

### Task B.4 — 渐进式 smoke ladder（在 Cohort batch 之前）

> 用户强制要求：**Stage B 不可一次跑 3700 records**。必须按以下 ladder 逐步爬升，每步停下来人审：

| Step | 目标 | 命令 | 预期产物 + 用户确认点 |
|---:|---|---|---|
| **B.4.a** | 1 subject × 1 record（最小 e2e） | `python scripts/run_epilepsiae_lagpat_backfill.py --subject 253 --stem 25300102_0000` | 写 1 个 lagPat.npz + packedTimes.npy。**用户审 lagPatRaw shape 与 chnNames 是否合理后再继续** |
| **B.4.b** | 253 全 subject（pure 512 Hz cohort smoke） | `python scripts/run_epilepsiae_lagpat_backfill.py --subject 253` | 268 records 全跑通，`_backfill_log.json` 含 per-record runtime。**用户审 n_failed=0 + 平均 runtime 后继续** |
| **B.4.c** | 1 个 1024 Hz subject（验证 sfreq 通用性）| `python scripts/run_epilepsiae_lagpat_backfill.py --subject 548` | 147 records，对比 253 的 chnNames overlap > 0（不同 subject 有不同蒙太奇是正常的）|
| **B.4.d** | 1 个 mixed-sfreq subject | `python scripts/run_epilepsiae_lagpat_backfill.py --subject 384` | 65 records（仅 1024 Hz blocks；256 Hz 已在 detector 跳过，census 数确认）|
| **B.4.e** | Cohort batch（仅在 a–d 全过后） | 见 Task B.5 | 20/20 subject 跑通 |

**用户决策门**：B.4.a 完成后必须停下来 commit + 让用户审；不要直接进 B.4.b。

- [ ] **Step 1**：B.4.a 单 record run

```bash
python scripts/run_epilepsiae_lagpat_backfill.py --subject 253 --stem 25300102_0000
ls -la results/epilepsiae_lagpat_backfill/253/25300102_0000_lagPat.npz \
       results/epilepsiae_lagpat_backfill/253/25300102_0000_packedTimes.npy
python -c "
import numpy as np
z = np.load('results/epilepsiae_lagpat_backfill/253/25300102_0000_lagPat.npz', allow_pickle=True)
print('keys:', sorted(z.files))
print('chnNames:', list(z['chnNames']))
print('lagPatRaw shape:', z['lagPatRaw'].shape)
print('eventsBool participation per event:', z['eventsBool'].sum(axis=0)[:10])
"
```

- [ ] **Step 2**：commit + 用户审

- [ ] **Step 3**：B.4.b – B.4.d 依次跑（每步 commit）

- [ ] **Step 4**：用户绿灯后进 Task B.5

### Task B.5 — Cohort batch driver（resume / skip-existing / per-subject log / parallel）

> 用户强制要求：**resume / skip-existing / per-subject logs / parallel 必须内置进 plan**，不能"留到执行阶段"。

#### 必须内置的能力

1. **Skip-existing**：默认 `--skip-existing`（默认 ON）。若 `<output>/<subject>/<stem>_lagPat.npz` 已存在且 loadable，跳过该 record。`--force` 可覆盖。
2. **Per-subject log**：每 subject 独立写 `_backfill_log.json`：
   ```json
   {
     "subject": "253",
     "started_at": "2026-04-29T12:00:00",
     "completed_at": "2026-04-29T18:30:00",
     "n_records_total": 268,
     "n_records_done": 268,
     "n_skipped_existing": 0,
     "n_failed": 0,
     "failures": [],
     "per_record_seconds": {"25300102_0000": 92.3, ...},
     "median_record_seconds": 87.5
   }
   ```
3. **Subject-level parallel**：plan 内附 GNU parallel 启动脚本（不是"运行建议"）：
   ```bash
   # scripts/run_epilepsiae_lagpat_backfill_parallel.sh
   set -euo pipefail
   N_JOBS=${N_JOBS:-5}
   SUBJECTS="${SUBJECTS:-253 548 139 384 1077 1084 442 818 916 922 958 583 590 620 635 1073 1096 1125 1146 1150}"
   parallel -j "$N_JOBS" --line-buffer \
     "python scripts/run_epilepsiae_lagpat_backfill.py --subject {} 2>&1 | tee results/epilepsiae_lagpat_backfill/{}/_console.log" \
     ::: $SUBJECTS
   python scripts/run_epilepsiae_lagpat_backfill.py --aggregate-cohort-summary
   ```
4. **Crash recovery**：每 record 跑完即写 disk + 更新 log；中途 SIGINT / SIGKILL 不会丢已完成 records。重启脚本自动 skip-existing。
5. **Per-record timeout**：单 record > 30 min 视为 fail，记入 `failures`，继续下一个。

#### Tasks

- [ ] **Step 1**：写测试

```python
def test_skip_existing_default(tmp_path, monkeypatch):
    """Verify pre-existing lagPat.npz is skipped by default."""
    from scripts.run_epilepsiae_lagpat_backfill import process_subject
    # ... seed a fake completed record; assert process_subject increments n_skipped_existing

def test_per_subject_log_written(tmp_path):
    """Verify _backfill_log.json contains required keys."""
    # ... run a 1-record subject; assert log file has all keys

def test_force_flag_overwrites():
    """--force re-runs even if output exists."""
    # ... assert n_skipped_existing == 0 when force=True
```

- [ ] **Step 2**：实现 `process_subject(subject, *, skip_existing=True, force=False, max_record_sec=1800)` + log writer

- [ ] **Step 3**：实现 `_aggregate_cohort_summary()` 走每 subject `_backfill_log.json` 汇总到 `cohort_summary.csv`

- [ ] **Step 4**：写 `scripts/run_epilepsiae_lagpat_backfill_parallel.sh`

- [ ] **Step 5**：跑全 cohort（推荐 N_JOBS=5；rough estimate 6 min/record × 3700 / 5 ≈ 74 wall hours）

```bash
bash scripts/run_epilepsiae_lagpat_backfill_parallel.sh
```

- [ ] **Step 6**：检查 cohort summary

```bash
column -t -s, results/epilepsiae_lagpat_backfill/cohort_summary.csv
# 期望：20 行，n_failed=0 全部，n_records_done == n_records_total（修正：== census new_gpu_records_total）
```

- [ ] **Step 7**：commit

---

## 4. Stage C — 结构对比（新 vs 老 lagPat）

**目标**：每 subject、每对齐的 record（见 1.1 sfreq gate），输出 5 维结构相似度。**不做** numerical parity 主张。

### 输出
```
results/epilepsiae_lagpat_backfill/
├── audit/
│   ├── per_record_audit.csv          (subject, stem, n_chns_new, n_chns_old, chn_overlap_jaccard,
│   │                                   n_events_new, n_events_old, count_ratio,
│   │                                   participation_ks_p, lag_span_diff_med,
│   │                                   rank_template_corr_med)
│   ├── per_subject_audit.csv         (median + IQR over records)
│   ├── cohort_audit_summary.json     (cohort-level distribution + decision flags)
│   └── figures/
│       ├── README.md                 中文
│       ├── chn_overlap_hist.png
│       ├── event_count_ratio_hist.png
│       └── rank_template_corr_hist.png
```

### Task C.1 — 5 维相似度计算

| 维度 | 公式 | 输出 col |
|---|---|---|
| Channel set overlap | Jaccard(chnNames_new, chnNames_old) | `chn_overlap_jaccard` |
| Event count ratio | n_events_new / n_events_old | `count_ratio` |
| Participation distribution | KS test on n_participating per event | `participation_ks_p` |
| Lag span | median |max(lagPatRaw,axis=0) - min(lagPatRaw,axis=0)| diff | `lag_span_diff_med` |
| Rank template similarity | mean Pearson r between mean rank vectors over shared chns | `rank_template_corr_med` |

- [ ] **Step 1**：写测试，准备 fixture（合成 1 个匹配 + 1 个差异显著 record）

- [ ] **Step 2**：跑测试 fail

- [ ] **Step 3**：实现 `scripts/audit_epilepsiae_lagpat_backfill.py`

> **绝对禁止**：在 audit 阶段假设新 lagPat 应该 == 老 lagPat。即使全维度都极差，也不要在脚本里 raise；只把数值写出来。

- [ ] **Step 4**：跑 cohort audit

```bash
python scripts/audit_epilepsiae_lagpat_backfill.py --all
```

- [ ] **Step 5**：commit

### Task C.2 — Per-subject bucket assignment（不只看 cohort median）

> 用户强制要求：**Stage C gate 按 subject 分桶**。Cohort median 会掩盖 "10 好 / 10 坏" 的双峰分布。

#### Bucket 定义（per subject，基于 median over the subject's records）

| bucket | chn_overlap_jaccard | count_ratio | 含义 |
|---|---|---|---|
| **stable** | ≥ 0.7 | [0.7, 1.4] | 新老 lagPat 结构高度一致；下游可比 |
| **moderate_drift** | 0.5 – 0.7 | [0.5, 2.0] (excl stable) | 显著差异但同量级；smoke sensitivity 即可 |
| **large_drift** | < 0.5 | < 0.5 或 > 2.0 | 结构换了；下游结论需重写 |

每 subject 同时算两维 bucket，**取较保守的桶**（e.g. chn_overlap stable + count_ratio large → 整体 large_drift）。

#### 输出额外字段

`per_subject_audit.csv` 新增列：
- `chn_bucket`：stable / moderate_drift / large_drift
- `count_bucket`：同上
- `subject_bucket`：min(chn_bucket, count_bucket)（保守取 large_drift if either is large_drift）

`cohort_audit_summary.json` 新增段：
```json
{
  "bucket_counts": {"stable": N1, "moderate_drift": N2, "large_drift": N3},
  "subject_lists": {
    "stable": [...],
    "moderate_drift": [...],
    "large_drift": [...]
  },
  "decision_for_stage_d": "enter_full | enter_smoke | pause"
}
```

#### `decision_for_stage_d` 规则（cohort-level，三档）

| 条件 | decision |
|---|---|
| ≥ 14/20 (70%) subjects in `stable` 且 ≤ 2/20 in `large_drift` | **enter_full**：Stage D 跑完整 PR-1 + PR-6 子集 |
| 5–13/20 in `stable` 或 3–7/20 in `large_drift` | **enter_smoke**：Stage D 只跑 1–2 个 stable subject 看方向 |
| ≥ 8/20 in `large_drift` 或 < 5/20 in `stable` | **pause**：暂停 Stage D，写 mid-term archive，让用户决定 |

→ 这些规则**由 audit 脚本自动判定**写入 JSON，但 Stage D 是否真跑由 plan 的 verification gate 决定（不强制自动 trigger）。

#### Tasks

- [ ] **Step 1**：写测试，验证 bucket 分配 + decision rule

```python
def test_subject_bucket_conservative_minimum():
    from scripts.audit_epilepsiae_lagpat_backfill import _assign_subject_bucket
    # chn stable + count large → large
    assert _assign_subject_bucket("stable", "large_drift") == "large_drift"
    # both stable → stable
    assert _assign_subject_bucket("stable", "stable") == "stable"
    # stable + moderate → moderate
    assert _assign_subject_bucket("stable", "moderate_drift") == "moderate_drift"

def test_decision_for_stage_d():
    from scripts.audit_epilepsiae_lagpat_backfill import _decide_stage_d
    assert _decide_stage_d(stable=15, moderate=4, large=1) == "enter_full"
    assert _decide_stage_d(stable=10, moderate=5, large=5) == "enter_smoke"
    assert _decide_stage_d(stable=4, moderate=4, large=12) == "pause"
```

- [ ] **Step 2**：实现 `_assign_subject_bucket(chn, count) -> str` + `_decide_stage_d(stable, moderate, large) -> str`

- [ ] **Step 3**：跑测试 pass

- [ ] **Step 4**：commit

---

## 5. Stage D — 下游敏感性（按 Stage C decision 分级触发）

**目标**：用新 lagPat 作为输入，验证 Topic 1 结论是否稳。**Stage D 的 scope 由 Stage C `decision_for_stage_d` 字段决定**：

| Stage C decision | Stage D scope | 说明 |
|---|---|---|
| `enter_full` | D.1 + D.2 全 stable subjects | propagation PR-1 stable_k 分布 + synchrony PR-6 phase_all/phase_e |
| `enter_smoke` | 仅 D.1.smoke + D.2.smoke（1–2 个 stable subject） | 只看方向是否同号；不出 cohort p-value |
| `pause` | **不跑 Stage D**，跳到 Stage E mid-term archive | 让用户决策是否需要先调 detector / refine 参数 |

### 必要前置代码改动

- `scripts/run_interictal_propagation.py:47` 的 `EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/...")` 当前是 module-level 硬编码常量。**改造**：加 `--epilepsiae-root` CLI argument，default 维持原值，runtime override 后写入 per-subject JSON `meta.epilepsiae_root_used` 字段以便 audit。
- `scripts/run_epilepsiae_interictal_synchrony.py` 同样加 `--lagpat-root` 参数。

### Task D.1 — Interictal propagation（仅 stable subjects）

- [ ] **Step 1**：改造 `scripts/run_interictal_propagation.py` 加 `--epilepsiae-root`

- [ ] **Step 2**：写测试 `tests/test_interictal_propagation_root_override.py` 验证 root override 正确

- [ ] **Step 3**：根据 Stage C decision 跑

```bash
# enter_full 模式
python scripts/run_interictal_propagation.py --pr1 \
  --epilepsiae-root results/epilepsiae_lagpat_backfill \
  --epilepsiae-subjects "<stable_subjects_from_audit>" \
  --output-root results/interictal_propagation/sensitivity_new_lagpat

# enter_smoke 模式（仅 1 个 stable subject e.g. 253 if it's stable）
python scripts/run_interictal_propagation.py --pr1 \
  --epilepsiae-root results/epilepsiae_lagpat_backfill \
  --epilepsiae-subjects "253" \
  --output-root results/interictal_propagation/sensitivity_new_lagpat_smoke
```

- [ ] **Step 4**：对比 `stable_k` 分布

| metric | 老 lagPat 结论 (Topic 1 PR-2) | 新 lagPat 结论 |
|---|---|---|
| `stable_k` 分布 (Epilepsiae) | TBD（从 `results/interictal_propagation/cohort_summary.json` 取 Epilepsiae 子集） | TBD |
| AMI stability median | TBD | TBD |
| MI significant fraction | TBD | TBD |

**判定**：
- enter_full：stable_k 分布 ≥ 80% 一致 → "下游稳"；< 80% → "Topic 1 Epilepsiae 结论 legacy-derived"
- enter_smoke：方向同号即可，不做 cohort 主张

### Task D.2 — Synchrony Pre/Post（仅 stable subjects）

- [ ] **Step 1**：改造 `scripts/run_epilepsiae_interictal_synchrony.py` 加 `--lagpat-root`

- [ ] **Step 2**：根据 Stage C decision 跑

```bash
python scripts/run_epilepsiae_interictal_synchrony.py \
  --lagpat-root results/epilepsiae_lagpat_backfill \
  --subjects "<stable_subjects>" \
  --output-root results/interictal_synchrony/sensitivity_new_lagpat
```

- [ ] **Step 3**：对比 phase_all / phase_e Pre vs Post p-value

### Task D.3 — Mid-term / final archive

- 若 Stage C decision = `pause` → 写 `docs/archive/epilepsiae_lagpat/midterm_pause_<date>.md`，**跳过 D.1/D.2**
- 若 enter_smoke → 写 `docs/archive/epilepsiae_lagpat/sensitivity_audit_smoke_<date>.md`
- 若 enter_full → 写 `docs/archive/epilepsiae_lagpat/sensitivity_audit_results_<date>.md`

---

## 6. Stage E — 文档落档

### Task E.1 — Contract archive

写 `docs/archive/epilepsiae_lagpat/epilepsiae_lagpat_backfill_contract_2026-XX-XX.md`：
- §1 Scope: 不是 replay
- §2 Algorithm: per-record CAR + variable sfreq + 200s segment
- §3 Stage A-E 的实际输出数值
- §4 与老 lagPat 5 维 audit 表
- §5 Stage D 下游 sensitivity 结论
- §6 Decision: 老 Topic 1 结论是否标 `legacy-derived` 或 `new-pipeline-confirmed`

### Task E.2 — 主文档增补

`docs/topic1_within_event_dynamics.md`：
- 在每个 Epilepsiae 章节顶部加一段：
  > **数据合同**：本节 Epilepsiae 结果基于 `interilca_inter_results/all_data_lns/<subject>/all_recs/*_lagPat.npz`（legacy artifact）。新 pipeline lagPat sensitivity audit 见 [`docs/archive/epilepsiae_lagpat/`](archive/epilepsiae_lagpat/)。

`docs/topic3_spatial_soz_modulation.md`：
- 不需要改动（PR-2 不依赖 lagPat）
- 但可在"历史文档索引"加一行链接到本 plan 的 archive

### Task E.3 — AGENTS.md 索引（**deferred，不在本计划内**）

> 用户决策：暂不动 AGENTS.md。AGENTS.md 是项目级入口；只有当本 audit 升级为长期 canonical contract（即"新 lagPat 是 Topic 1 默认数据源"）才动。当前 archive 链接 + Topic 1 主文档段已经足够发现路径。

如果将来 sensitivity audit 显示新 lagPat 应替代老 lagPat 成为 Topic 1 默认数据源，再单独开一个 PR 改 AGENTS.md，并把 Topic 1 主文档的 "数据合同" 段也对应翻新。

---

## 7. Critical files

### 新建
- `scripts/run_epilepsiae_lagpat_backfill.py`
- `scripts/audit_epilepsiae_lagpat_backfill.py`
- `tests/test_epilepsiae_lagpat_backfill.py`
- `results/epilepsiae_lagpat_backfill/<subject>/*_packedTimes.npy` (×~3700 records)
- `results/epilepsiae_lagpat_backfill/<subject>/*_lagPat.npz` (×~3700 records)
- `results/epilepsiae_lagpat_backfill/cohort_summary.csv`
- `results/epilepsiae_lagpat_backfill/audit/per_record_audit.csv`
- `results/epilepsiae_lagpat_backfill/audit/cohort_audit_summary.json`
- `results/epilepsiae_lagpat_backfill/audit/figures/{README.md, *.png}`
- `docs/archive/epilepsiae_lagpat/epilepsiae_lagpat_backfill_contract_2026-XX-XX.md`
- `docs/archive/epilepsiae_lagpat/sensitivity_audit_results_2026-XX-XX.md`

### 改动
- `scripts/run_interictal_propagation.py:47` — 加 `--epilepsiae-root` CLI arg（仅 Stage D 才必需）
- `scripts/run_epilepsiae_interictal_synchrony.py` — 加 `--lagpat-root` CLI arg（仅 Stage D 才必需）
- `docs/topic1_within_event_dynamics.md` — 各 Epilepsiae 章节顶部加数据合同段
- `docs/topic3_spatial_soz_modulation.md` — 历史文档索引（可选）
- ~~`AGENTS.md`~~ — **deferred**，见 Task E.3

### 复用（不改）
- `src/preprocessing.py::load_epilepsiae_block, _read_epilepsiae_head_for_streaming`
- `src/epilepsiae_dataset.py::EpilepsiaePaths, _collect_raw_blocks`
- `src/group_event_analysis.py::compute_centroid_matrix_spectrogram, lag_rank_from_centroids, build_windows_from_detections, EventWindow`

---

## 8. Verification gates（每 stage 完成才能进下一个）

### Stage A
- `pytest tests/test_epilepsiae_lagpat_backfill.py` 全绿
- `python scripts/run_epilepsiae_lagpat_backfill.py --subject 253 --smoke` 输出 record 元信息但**未**写任何文件
- `find results/epilepsiae_lagpat_backfill/ -type f` 空（smoke 不写文件）

### Stage B（按 ladder 分级）
- `pytest tests/test_epilepsiae_lagpat_backfill.py` 全绿
- **B.4.a**：单 record (`253/25300102_0000`) 写出 lagPat.npz + packedTimes.npy；shape 合理；start_t 是 Unix epoch（1e9 < t < 2e9 即 2001-09 ~ 2033-05 之间，Epilepsiae 实际范围 2008-2012）；**用户审过**
- **B.4.b**：253 全 subject (268 records) `n_failed=0`；`_backfill_log.json` median_record_seconds 合理
- **B.4.c**：548 (1024 Hz) 跑通 147 records
- **B.4.d**：384 (mixed) 跑通 65 records（仅 1024 Hz blocks）
- **B.4.e (Cohort batch)**：20/20 subject 在 `cohort_summary.csv` 中 `n_records_done == census new_gpu_records_total`，`n_failed=0`；中途 SIGINT 后重启可 skip-existing

### Stage C
- 20/20 subject audit CSV 行齐全
- per-subject median chn_overlap_jaccard 落在 [0, 1]
- `subject_bucket` 列存在，bucket 分配遵循 stable / moderate_drift / large_drift 规则
- `cohort_audit_summary.json` 含 `bucket_counts` + `decision_for_stage_d`
- 至少 1 张 figure 含有效散布

### Stage D（按 Stage C decision 触发）
- 若 `decision_for_stage_d == pause`：跳过 D.1/D.2，写 mid-term archive，verification gate 即 archive 文件存在
- 若 `enter_smoke`：D.1/D.2 仅 1–2 个 stable subject，方向比对完成
- 若 `enter_full`：D.1/D.2 跑全 stable subjects，cohort p-value 表生成

### Stage E
- `docs/archive/epilepsiae_lagpat/` 下至少 2 个 md（本 plan + contract or sensitivity 至少其一）
- 主文档 Topic 1 各 Epilepsiae 章节顶部都有数据合同段
- ~~AGENTS.md 索引~~ deferred

---

## 9. 已规避的设计错误（沿用 Yuquan + Topic 3 经验）

1. **不要把这个 plan 当作 Track B replay**。明确写在 §0：缺老 gpu 冻结条件，**不可能** numerical parity；只做结构 + 下游 sensitivity。
2. **不要 fork `run_yuquan_lagpat_backfill.py`**。Yuquan 5 处硬编码（bipolar reref / 800 Hz resample / 单 EDF / `.legacy_backup` / `/mnt/yuquan_data`）全部不适用。新建独立脚本。
3. **不要写回 `/mnt/epilepsia_data/`**。`OUTPUT_ROOT` 必须在 `results/` 下；测试硬断言挡住误写。
4. **不要在 Stage C 把"差异大"等同于"新 pipeline bug"**。结构差异是 detector + refine 改进的预期后果（Yuquan dual-track 已证实 8/14 是真实 refine drift，不是代码 bug）。
5. **不要在 Stage D 主张"新 lagPat 替代老 lagPat"**。Topic 1 当前结论是 legacy-derived；本计划只 flag 是否要打 `legacy-derived` 标签，不直接推翻。
6. **不要把 `forward_reverse_reproduced` / `stable_k` 当作 cohort 主张证据**。CLAUDE.md §5 已明确 PR-6 H2 是 mechanism sanity not cohort claim；Stage D 同理。
7. **不要在 Stage A 写文件**。Stage A 全程 dry-print，verification gate 之一就是 `find results/ -type f` 在 OUTPUT_ROOT 下为空。

---

## 10. Open issues / 不在本 plan 范围

- **运行成本**：~3700 records × spectrogram + bandpass，估计 6 min/record，单线程总 ~370 hour。需要并行（GNU parallel / Snakemake / SLURM），**但本 plan 不规定具体方案**——交给执行时决定。
- **Yuquan 1077 epilepsiae 重检测**：1077 (epilepsiae) 已用 CPU 重跑过 detector + refine（2026-04-16）。与其他 19 subject 检测条件一致，本计划无差异。
- **inv2 / epilepsiae_3patient 子集**：census 只列 20 subject。EpilepsiaePaths 也扫了 `inv2 / epilepsiae_3patient`，但结果是空的（census 已验证）。本 plan 不处理这两个根。
- **老 lagPat 与 lagPat_withFreqCent**：老 pipeline 同时写 `*_lagPat.npz`（旧）和 `*_lagPat_withFreqCent.npz`（新）。本计划比较的是 `*_lagPat.npz`（Topic 1 当前消费的）；`withFreqCent` 不在 scope。

---

## 10b. User-pinned reservations（plan accepted with these guardrails）

> 用户在 2026-04-29 接受本 plan 作为执行口径，但留两条必须严格执行的保留意见。subagent 不得自作主张绕过：

### R1：Stage B 耗时估计是占位，不是事实
- §3 Task B.5 写的 "6 min/record × 3700 / N_JOBS=5 ≈ 74 wall hours" 是 **preliminary placeholder**，不能作为执行依据。
- **B.5 启动前必须**：拿 B.4.a（单 record）+ B.4.b（253 全 268 records）的真实 `_backfill_log.json::median_record_seconds` 重新算 cohort wall time。
- 若实测 median 与 placeholder 偏差 > 50%（即 > 9 min/record 或 < 3 min/record），停下来重审 N_JOBS 与机器资源，再开 B.5。

### R2：Stage D 不得自动触发
- §5 Stage D 的 "按 Stage C decision 分级触发" 只描述 **scope**，不授权 subagent 自动执行。
- C.2 完成后，**必须**由用户读 `cohort_audit_summary.json::decision_for_stage_d` 字段，明确给出"开跑 / 跑 smoke / pause"指令，subagent 才能进 Stage D。
- 即使 decision 字段是 `enter_full`，subagent 也**不**能直接调 `scripts/run_interictal_propagation.py` / `scripts/run_epilepsiae_interictal_synchrony.py`。Topic 1 sensitivity 是有副作用的下游分析，必须保留人审闸门。
- enter_smoke / pause 同理：subagent 不得"既然只是 smoke 就先跑了"。

---

## 11. User check-in points（强制人审，不要悄悄推进）

| 节点 | 必须停下 | 用户确认事项 |
|---|---|---|
| **Task B.2 probe** | 是 | whole_dets 单位是 SECONDS（probe 输出"SECONDS"）；若为 SAMPLES，停 plan 修 hfo_detector 合同 |
| **Task B.4.a 完成（单 record）** | 是 | lagPat shape 合理 + chnNames 与 refine 列表一致 + start_t 是 epoch；之前不要进 B.4.b |
| **Task B.5 完成（cohort batch 写完）** | 是 | `cohort_summary.csv` 全绿，无 failures；用户决定是否进 Stage C |
| **Task C.2 完成（bucket + decision）** | **强制** | 用户审 `decision_for_stage_d` 字段；用户决定是否按该 decision 跑 Stage D |
| **Stage D 完成** | 是 | 用户审 sensitivity 结果，决定是否要在 Topic 1 主文档加 `legacy-derived` 标签 |

---

## 12. Done condition

- [ ] Stage A 2 tasks complete + verification gate passed
- [ ] Stage B 5 tasks (B.1–B.5) complete + 4-step ladder (B.4.a–d) 全过 + Cohort batch (B.4.e/B.5) 完成 + 每个 user check-in 通过
- [ ] Stage C 2 tasks complete + verification gate passed + bucket 分配 + decision 字段写出
- [ ] Stage D 按 decision 跑（pause / smoke / full），3 task 中至少 1 个完成 archive 写出
- [ ] Stage E 2 tasks 完成（contract md + 主文档合同段；AGENTS.md deferred）
- [ ] `pytest tests/` 全绿（除已知 pre-existing fail）
- [ ] `docs/topic1_within_event_dynamics.md` 每 Epilepsiae 章节顶部有数据合同段

> **建议执行模式**：subagent-driven-development。每个 Task 一个 subagent。
> - Stage A 可以一气呵成
> - Stage B **必须**在 B.2 probe + B.4.a 单 record + B.5 cohort batch 完成后停下来人审
> - Stage C **必须**在 C.2 完成后停下来人审 `decision_for_stage_d`
> - Stage D 严格按 decision 跑，不要超出 scope
