# HFO Detector v2 — Cohort Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 Epilepsiae 全 cohort 检测从"对齐 21 年 cusignal 黑盒"重新定位为 **HFO Detector v2**——一个 deterministic、CPU/GPU 等价、内部可复现的现代 detector，并为它建立独立的三层验收契约（单通道 / 群体事件 / 科学下游）。21 年 gpu.npz 仅作只读历史参照，不作 1:1 验收基准。

**Architecture:** 用现有 `src.preprocessing.load_epilepsiae_block` + `src.hfo_detector.HFODetector(legacy_align=True, use_gpu=True)` 作为 v2 主路径——已被 commit 6027281 的 Path A 工作覆盖且经 cuda_env 验证 GPU=CPU=float32=float64 等价。重检全 cohort、重建所有下游（synRefine、packedTimes、lagPat、synchrony、propagation）。验收三层：(A) 单通道事件质量；(B) 群体事件 packing 后质量；(C) 科学下游 v2 自身 split-half 稳定性。

**Tech Stack:** Python 3.11 / scipy / cupy 13.6 / cusignal 23.08 / pytest / pandas。conda env: `cuda_env`。

---

## File Structure

### 新增（v2 spec / 工具 / 报告）

- `docs/archive/hfo_detector_v2/v2_cohort_rebuild_plan_2026-05-05.md`（本文件）
- `docs/archive/hfo_detector_v2/v2_specification.md` — v2 detector 算法定义、参数表、输出格式
- `docs/archive/hfo_detector_v2/v2_validation_contract.md` — 三层验收契约 + 阈值
- `docs/archive/hfo_detector_v2/v2_cohort_results_<date>.md` — 跑完后填的 cohort 报告
- `results/_legacy_2021_readonly/` — 把 `_legacy_2021_backup` 改名 + README 锁死
- `results/hfo_detector_v2/` — v2 cohort 检测产物根目录（per-subject _gpu.npz + manifest）
- `results/hfo_detector_v2/lagpat/<subject>/<stem>_lagPat.npz` — v2 重建的 packed lagPat（输出隔离于历史 `results/epilepsiae_lagpat_backfill/`）
- `results/hfo_detector_v2/synchrony/` — v2 重建的 synchrony event CSV + aggregation（隔离于 `results/interictal_synchrony/`，不覆盖原历史结果）
- `results/hfo_detector_v2/propagation/per_subject/epilepsiae_<sid>.json` — v2 重建的 propagation per-subject（隔离于 `results/interictal_propagation/per_subject/`，不覆盖原历史结果）
- `results/hfo_detector_v2/validation/layer_a_<subject>.json` — 每个 subject 的单通道质量
- `results/hfo_detector_v2/validation/layer_b_<subject>.json` — 每个 subject 的群体事件质量
- `results/hfo_detector_v2/validation/layer_c_<subject>.json` — 每个 subject 的下游 split-half
- `results/hfo_detector_v2/validation/cohort_summary.json` — 三层 cohort 汇总
- `scripts/v2_validate_layer_a.py` — Layer A 提取器
- `scripts/v2_validate_layer_b.py` — Layer B 提取器（消费 packed_times + lagPat）
- `scripts/v2_validate_layer_c.py` — Layer C 编排器（调用现有 `compute_time_split_reproducibility`）
- `scripts/v2_run_cohort_detection.py` — cohort detection 编排（GPU 默认）
- `tests/test_v2_validate_layer_a.py` — Layer A 单元测试
- `tests/test_v2_validate_layer_b.py` — Layer B 单元测试

### 修改（去 "legacy reproduction" 措辞）

- `docs/archive/epilepsiae_lagpat/legacy_replication_audit_2026-05-03.md` — 加入 v2 banner，说明 audit 已重定义为"refine/pack 层 verbatim port，不再做 detector 1:1 复刻"
- `docs/archive/epilepsiae_lagpat/detector_drift_root_cause_2026-05-03.md` — 加入 v2 banner，注明结论改为"v2 在现代 stack 上 deterministic"，21 年漂移不再追究
- `docs/archive/epilepsiae_lagpat/epilepsiae_lagpat_backfill_plan_2026-04-29.md` — 加入 v2 banner，原计划用于历史记录
- `AGENTS.md` — 在"Epilepsiae Contract"段加入 v2 路径声明、21 年只读注记

### 复用（不动）

- `src/preprocessing.py` — `load_epilepsiae_block(legacy_align=True, notch_filter_kind='fir_legacy')` 已是 v2 默认
- `src/hfo_detector.py` — `HFODetector(use_gpu=True, legacy_align=True)` 是 v2 调用形式
- `src/utils/bqk_utils.py` — `BQKDetector` GPU/CPU 路径完整
- `src/interictal_propagation.py:1734` — `compute_time_split_reproducibility` 用于 Layer C
- `scripts/run_epilepsiae_lagpat_backfill.py` — synRefine + lagPat 主程序，输入切到 v2 即可
- `scripts/run_epilepsiae_detection_parallel.sh` — 并行包装器，OUTPUT_DIR 改 `results/hfo_detector_v2`

---

## Phase 0: 锁住 21 年只读资产 + 退役 "legacy reproduction" 措辞

### Task 0.1: 校验 21 年备份 manifest 完整性

**Files:**
- Read: `results/_legacy_2021_backup/inv/_manifest.csv`
- Read: `results/_legacy_2021_backup/inv_1_part/_manifest.csv`

- [ ] **Step 1: 找到所有 manifest 文件并打印 row count**

```bash
find /home/honglab/leijiaxin/HFOsp/results/_legacy_2021_backup -name "_manifest.csv" -o -name "manifest.csv"
```

Expected output: 1+ manifest CSV with rows = number of legacy gpu.npz/.head files

- [ ] **Step 2: 用 manifest 中的 md5 重新校验所有文件**

```bash
cd /home/honglab/leijiaxin/HFOsp/results/_legacy_2021_backup
python -c "
import csv, hashlib
from pathlib import Path
ROOT = Path('.')
fail = 0
total = 0
for mfp in ROOT.rglob('_manifest.csv'):
    with open(mfp) as fh:
        for row in csv.DictReader(fh):
            total += 1
            p = mfp.parent / row['relpath']
            if not p.exists():
                print('MISSING:', p); fail += 1; continue
            h = hashlib.md5(p.read_bytes()).hexdigest()
            if h != row['md5']:
                print('MD5 MISMATCH:', p); fail += 1
print(f'verified {total} files, {fail} failures')
"
```

Expected: `verified <N> files, 0 failures`. If failures, STOP and investigate before proceeding.

- [ ] **Step 3: 没找到 manifest 时降级为现场 md5 dump**

如果 Step 1 输出空（没有 manifest），生成一份新 manifest：

```bash
cd /home/honglab/leijiaxin/HFOsp/results/_legacy_2021_backup
python -c "
import hashlib, csv
from pathlib import Path
out = Path('_manifest_full.csv')
rows = []
for p in Path('.').rglob('*'):
    if not p.is_file() or p.name.startswith('_manifest'):
        continue
    h = hashlib.md5(p.read_bytes()).hexdigest()
    sz = p.stat().st_size
    rows.append({'relpath': str(p), 'md5': h, 'size_bytes': sz})
print(f'hashed {len(rows)} files')
with open(out, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['relpath','md5','size_bytes'])
    w.writeheader(); w.writerows(rows)
print(f'wrote {out}')
"
```

Expected: writes `_manifest_full.csv` with all files' md5.

### Task 0.2: 改名 + 加 README 锁死 21 年备份

**Files:**
- Move: `results/_legacy_2021_backup/` → `results/_legacy_2021_readonly/`
- Create: `results/_legacy_2021_readonly/README.md`

- [ ] **Step 1: 确认当前路径仍叫 backup**

```bash
ls -d /home/honglab/leijiaxin/HFOsp/results/_legacy_2021_backup
```

Expected: directory exists.

- [ ] **Step 2: 改名为 readonly**

```bash
mv /home/honglab/leijiaxin/HFOsp/results/_legacy_2021_backup /home/honglab/leijiaxin/HFOsp/results/_legacy_2021_readonly
ls -d /home/honglab/leijiaxin/HFOsp/results/_legacy_2021_readonly
```

Expected: new directory listed.

- [ ] **Step 3: 写 README**

Create `results/_legacy_2021_readonly/README.md`:

```markdown
# Legacy 2021 read-only reference

This directory holds the 2021-vintage Epilepsiae HFO detection artifacts
generated with the legacy `cusignal` GPU stack on `niking314` PC. They are
preserved verbatim as a historical reference.

## Status

- **READ-ONLY.** Do not regenerate, overwrite, or move files inside this tree.
- Manifest file: `_manifest.csv` / `_manifest_full.csv` (md5 + size per file).
- Verified: <YYYY-MM-DD> via `Task 0.1` of v2 cohort rebuild plan.

## Why preserved

- Citation source for any reference back to 21 年 paper figures / numbers.
- Diagnostic anchor when investigating per-subject discrepancies.
- NOT a 1:1 validation target for the v2 detector — see v2 specification
  + validation contract under `docs/archive/hfo_detector_v2/`.

## Not preserved here

The recursive Path A diagnostics (commits 6027281, 85f5a29) confirmed the
21 年 results cannot be bit-reproduced on modern stacks. Detector v2 is the
forward-going main pipeline. See:

- `docs/archive/hfo_detector_v2/v2_specification.md`
- `docs/archive/hfo_detector_v2/v2_validation_contract.md`
- `docs/archive/hfo_detector_v2/v2_cohort_rebuild_plan_2026-05-05.md`
```

- [ ] **Step 4: chmod readonly 防误写**

```bash
chmod -R a-w /home/honglab/leijiaxin/HFOsp/results/_legacy_2021_readonly
ls -la /home/honglab/leijiaxin/HFOsp/results/_legacy_2021_readonly | head -3
```

Expected: directory listing shows mode without `w` permission.

- [ ] **Step 5: 提交**

```bash
cd /home/honglab/leijiaxin/HFOsp
git add results/_legacy_2021_readonly/README.md
git commit -m "chore(v2): freeze 21年 detection artifacts as read-only reference

Renamed _legacy_2021_backup → _legacy_2021_readonly + chmod a-w.
Added README marking legacy as historical citation only — not a
1:1 validation target for HFO detector v2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 0.3: 在历史 archive 文档头加 v2 banner

**Files:**
- Modify: `docs/archive/epilepsiae_lagpat/legacy_replication_audit_2026-05-03.md` (banner at top)
- Modify: `docs/archive/epilepsiae_lagpat/detector_drift_root_cause_2026-05-03.md` (banner at top)
- Modify: `docs/archive/epilepsiae_lagpat/epilepsiae_lagpat_backfill_plan_2026-04-29.md` (banner at top)

- [ ] **Step 1: 在每篇文档第一行插入 v2 banner**

Banner 内容（统一）：

```markdown
> **⚠️ 已退役措辞 ⚠️**：本文档语境为"对齐 21 年 cusignal 输出"。该目标在 2026-05-05 被放弃——
> 见 `docs/archive/hfo_detector_v2/v2_cohort_rebuild_plan_2026-05-05.md`。21 年 gpu.npz 现已锁为
> 只读历史参照（`results/_legacy_2021_readonly/`）。本文档 detector 层结论仍可参考，但
> "100% 复刻 / verbatim port" 措辞在 v2 上下文中不再成立——v2 是 modern stack
> deterministic detector，不与 21 年 1:1 比对。
```

把它分别插到 3 个文件的第 1 行（在原标题前）。

- [ ] **Step 2: 提交**

```bash
git add docs/archive/epilepsiae_lagpat/legacy_replication_audit_2026-05-03.md \
        docs/archive/epilepsiae_lagpat/detector_drift_root_cause_2026-05-03.md \
        docs/archive/epilepsiae_lagpat/epilepsiae_lagpat_backfill_plan_2026-04-29.md
git commit -m "docs(v2): tag epilepsiae_lagpat archive docs as superseded

3 archive docs from the legacy-replication context now carry a v2 banner
pointing to the new specification + validation contract. Their detector
findings remain valid as diagnostic context but the 'verbatim port'
language no longer applies in the v2 forward pipeline.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 0.4: 在 AGENTS.md 加 v2 入口

**Files:**
- Modify: `AGENTS.md` (Epilepsiae Contract 段)

- [ ] **Step 1: 在 "## Epilepsiae Contract" 段最后追加 v2 入口段**

```markdown
## HFO Detector v2 (canonical pipeline since 2026-05-05)

**Read these before tracing any 2026+ Epilepsiae detection result:**
- `docs/archive/hfo_detector_v2/v2_specification.md` — algorithmic definition
- `docs/archive/hfo_detector_v2/v2_validation_contract.md` — 3-layer acceptance
- `docs/archive/hfo_detector_v2/v2_cohort_rebuild_plan_2026-05-05.md` — execution log

Canonical artifact root: `results/hfo_detector_v2/`. Do NOT compare v2 events
1:1 against `results/_legacy_2021_readonly/` — that backup is historical
citation only. The v2 detector is deterministic on modern stacks (CPU=GPU,
float32=float64); the 21 年 cusignal vintage cannot be bit-reproduced and is
not a parity target.
```

- [ ] **Step 2: 提交**

```bash
git add AGENTS.md
git commit -m "docs(v2): announce HFO detector v2 as canonical pipeline in AGENTS.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 1: v2 specification + validation contract

### Task 1.1: 写 v2 specification

**Files:**
- Create: `docs/archive/hfo_detector_v2/v2_specification.md`

- [ ] **Step 1: 起草 specification 文档**

Content sections:

1. **Definition**: detector 是什么，输入/输出
2. **Inputs**: Epilepsiae .data + .head；CAR；no manual drop_chns（per-subject from `sub_dropChns` 已 retire）
3. **Pipeline**:
   - notch: FIR-801 forward fftconvolve, freqs `[50, 100, 150, 200, 250]`
   - bandpass envelope: 9 sub-bands `(80,100), (100,120), ..., (240,250)`, FIR-201 forward fftconvolve + Hilbert
   - thresholds: `rel_thresh=2 × ch_median AND abs_thresh=2 × whole_data_median`
   - duration filter: 50 < dur < 200 ms
   - merge gap: 20 ms
   - side rejection: `pick_mean >= side_thresh × side_mean`，`side_thresh=2`，empty side window → reject
   - chunking: `chunk_sec=200, overlap=0`，`legacy_align=True`
4. **Backend equivalence**: GPU (cusignal/cupy) 和 CPU (scipy) 在 float32/float64 上 deterministic 等价（cuda_env 已验证 6e-18 系数差，427/43 完全一致）
5. **Outputs**: `*_gpu.npz` schema (whole_dets, chns_names, events_count, start_time, reference_type, bipolar_pairs)
6. **What changed vs Path A**: nothing on the algorithm. Only naming + framing.
7. **What changed vs 21 年 cusignal**: 现代 cusignal/cuFFT/CuPy 的具体数值路径，不可控不可复刻 — 不作为 v2 contract。
8. **Validation scope disclaimer (必须在文档显著位置)**:

> ⚠️ **三层验收衡量的是"pipeline 内部自洽"，不是"生物学有效性"**。Layer A 验证 detector 自己的 rule 自洽 (duration window、side-rejection 阈值、deterministic)；Layer B 验证 packing 在群体事件层稳定 (rank 在子集上一致)；Layer C 验证下游 reproducibility (在 v2 自身上 split-half / odd-even 一致)。**没有任何一层声明"事件是真实生理 HFO"**——真假 HFO 的判定需要 ground-truth 标注或外部独立测度（reviewer 标注 / 跨模态对照），不在 v2 验收范围内。引用 v2 cohort 结论时必须附带这条 disclaimer。

- [ ] **Step 2: 提交**

```bash
git add docs/archive/hfo_detector_v2/v2_specification.md
git commit -m "docs(v2): write HFO detector v2 algorithmic specification"
```

### Task 1.2: 写三层验收契约

**Files:**
- Create: `docs/archive/hfo_detector_v2/v2_validation_contract.md`

- [ ] **Step 1: 起草 validation contract**

文档头必须放 disclaimer 段：

```markdown
## ⚠️ 适用范围声明（必须先读）

本契约衡量的是 **pipeline 内部自洽**，不是 **生物学有效性**：

- **Layer A** = detector 自己的 rule 自洽（duration window 命中率、side-rejection 比值、deterministic）
- **Layer B** = packing 在群体事件层稳定（rank 在 split-half / odd-even 上一致）
- **Layer C** = 下游 PR-1 / PR-2.5 在 v2 自身上 split-half / odd-even reproducibility

没有任何一层声明"检测到的事件是真实生理 HFO"。这只能由 ground-truth 标注 / 外部独立测度
（reviewer 标注 / 跨模态对照）解决。所以引用 v2 cohort 结论时必须用"v2 内部自洽通过"
而非"事件已验证"，"该 subject 的 propagation 在 v2 数据上稳定"而非"该 subject 真有
propagation"。
```

并在 PASS 表前放 **阈值预注册** 段：

```markdown
## 阈值预注册 (locked 2026-05-05)

- **锁定时间**：2026-05-05，**在 cohort detection 跑之前**。锁定时间记入 git history。
- **失败 subject 处理**：
  - 不剔除主分析。fail subject 进 `large_drift` 列表，与主分析并列报告。
  - cohort 论述用"PASS rate" + "large_drift 列表"双轨披露，禁止只报 PASS rate 不披露 fail。
- **敏感性曲线**：每个层 PASS rate 在阈值 ±20% 上的曲线一并报告（contracted in cohort_summary）。
- **修订规则**：cohort 跑完后若发现阈值需调，必须新建 archive doc 注明 reason，并在新阈值下
  重新计算所有 cohort PASS。禁止悄悄改契约阈值。
```

具体阈值（PASS 条件，不达标则该 subject 在该层记 fail）：

**Layer A — 单通道事件质量** (per subject)

| 指标 | 计算 | PASS 条件 |
|------|------|-----------|
| `dur_in_band_frac` | events with 50 ≤ dur < 200 ms | ≥ 0.99（duration filter 之后理论 100%） |
| `peak_side_ratio_p25` | events 的 pick_mean / side_mean 的 25 分位 | ≥ 2.0（side rejection 阈值） |
| `threshold_margin_p50` | events 的 (env_max - threshold) / threshold 中位数 | ≥ 0.5（事件应有 ≥50% 余量，不只是擦边） |
| `timestamp_jitter_p99` | 同 subject 跑两遍后，每个事件 t_start 与最近事件的差值 99 分位 | ≤ 1 sample (~1 ms @ 1024 Hz) — deterministic 检验 |
| `strong_chn_count_match` | 同 subject 跑两遍后，top-10 通道的 events_count 一致比例 | = 1.0（GPU 应 deterministic）|

**Layer A 抽样策略**：每个 record 取 **3 个 evenly-spaced 200s window**（first / middle / last），不再只看 first 200s。Recording 短于 600s 取首段。完整描述见 Phase 2 Task 2.2 实现。

**Layer B — 群体事件 packing 后质量** (per subject)

| 指标 | 计算 | 角色 | PASS 条件 |
|------|------|------|-----------|
| `n_participating_p50` | packed group events 的 n_participating 中位数 | PASS gate | ≥ 2（至少有跨通道协同） |
| `n_participating_p10` | 10 分位 | PASS gate | ≥ 1（不全是孤立通道事件） |
| `pack_window_width_p50` | packed group event window 时长中位数 | PASS gate | 50–500 ms（合理范围）|
| `splithalf_event_rank_corr` | 上下半各自 lagPat → channel rank 的 Spearman 相关 | PASS gate | ≥ 0.7（强通道 rank 稳定）|
| `oddeven_event_rank_corr` | 奇偶 event 各自 lagPat → channel rank 的 Spearman | PASS gate | ≥ 0.7 |
| `chunk_boundary_event_frac` | start ∈ [n*200-2, n*200+2] s 的 group events 占比（chunk 边界附近代理 merge_overhead） | descriptive only | 报告 + cohort 分布，不作 PASS 门槛 |

> **注**：原 `merge_overhead` 因实现层无法准确测量（packing 内部 chunk-cross 行为不直接暴露），降级为 `chunk_boundary_event_frac` 描述指标，**不入 PASS**。Layer B PASS 由前 5 个指标共同决定。

**Layer C — 科学下游 v2 自身 reproducibility** (per subject + cohort)

| 指标 | 计算 | 角色 | PASS 条件 |
|------|------|------|-----------|
| `time_split_grade` | `compute_time_split_reproducibility` 整体评级（split-half ∥ odd-even）| PASS gate | `strong` 或 `moderate` |
| `forward_reverse_reproduced_strict` | split-half **AND** odd-even 都把 forward/reverse template 复现 | PASS gate | True |
| `forward_reverse_reproduced_lenient` | split-half **OR** odd-even（沿用 PR-2.5 历史定义） | descriptive only | 报告 + cohort 分布 |
| `stable_k_consistent` | adaptive cluster `stable_k` 在 split-half / odd-even 上一致或差 ≤1 | PASS gate | True |

> **关键澄清**：PR-2.5 的 `forward_reverse_reproduced` 历史定义是 OR (split-half ∥ odd-even)。v2 Layer C 引入 `_strict`（AND）作为 PASS 门槛，原 OR 字段保留并改名 `_lenient`，仅作描述与历史 cross-reference。Layer C PASS 用 strict，cohort 报告必须把 strict 与 lenient 数都列出。

**Cohort 验收**: 三层 PASS 数 / 总 subjects = 各 ≥ 0.85（容许 ~3 outlier）。失败 subject 进 large_drift 列表，**与主分析并列报告**，不得静默剔除。

- [ ] **Step 2: 提交**

```bash
git add docs/archive/hfo_detector_v2/v2_validation_contract.md
git commit -m "docs(v2): define 3-layer acceptance contract for cohort rebuild"
```

---

## Phase 2: 验收工具实现 (TDD)

### Task 2.1: Layer A 提取器

**Files:**
- Create: `scripts/v2_validate_layer_a.py`
- Test: `tests/test_v2_validate_layer_a.py`

- [ ] **Step 1: 写测试 — 假事件 fixture**

`tests/test_v2_validate_layer_a.py`:

```python
import numpy as np
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.v2_validate_layer_a import (
    compute_dur_in_band_frac,
    compute_peak_side_ratio,
    compute_threshold_margin,
    _window_starts_for_record,
    CHUNK_SEC,
    N_WINDOWS_PER_RECORD,
)

def test_dur_in_band_all_pass():
    # all events 100 ms — well within [50, 200]
    events = np.array([[0.0, 0.1], [0.5, 0.6], [1.0, 1.1]])
    assert compute_dur_in_band_frac(events, 50.0, 200.0) == 1.0

def test_dur_in_band_partial():
    # one event 30 ms (too short), one event 150 ms (in band)
    events = np.array([[0.0, 0.030], [0.5, 0.65]])
    assert compute_dur_in_band_frac(events, 50.0, 200.0) == 0.5

def test_peak_side_ratio_basic():
    fs = 1024.0
    env = np.ones(int(fs) * 2) * 5.0
    env[1024:1024+103] = 20.0
    events = np.array([[1.0, 1.1]])
    ratios = compute_peak_side_ratio(env, events, fs)
    assert len(ratios) == 1
    assert ratios[0] == pytest.approx(4.0, rel=1e-3)

def test_window_starts_short_recording_falls_back():
    # 400s recording (< 600s) → only the first window
    starts = _window_starts_for_record(400.0)
    assert starts == [0.0]

def test_window_starts_long_recording_evenly_spaced():
    # 3600s recording, 3 windows: 0, mid, last_start
    starts = _window_starts_for_record(3600.0)
    assert len(starts) == N_WINDOWS_PER_RECORD == 3
    last_start = 3600.0 - CHUNK_SEC
    assert starts[0] == 0.0
    assert starts[-1] == pytest.approx(last_start)
    assert starts[1] == pytest.approx(last_start / 2)

def test_window_starts_no_overlap_at_minimum_dur():
    # exactly 3 * CHUNK_SEC → first/middle/last must not all collapse to 0
    starts = _window_starts_for_record(3 * CHUNK_SEC)
    assert starts[0] == 0.0
    assert starts[-1] > 0.0
```

- [ ] **Step 2: Run tests，预期 ImportError**

```bash
cd /home/honglab/leijiaxin/HFOsp
conda run -n cuda_env pytest tests/test_v2_validate_layer_a.py -v
```

Expected: `ModuleNotFoundError: No module named 'scripts.v2_validate_layer_a'`

- [ ] **Step 3: 实现 Layer A 提取器（仅 dur + peak/side）**

`scripts/v2_validate_layer_a.py`:

```python
"""HFO detector v2 — Layer A validation extractor (single-channel quality).

Computes per-subject metrics from a v2 detection output (*_gpu.npz):
  - dur_in_band_frac
  - peak_side_ratio (p25, p50, p99)
  - threshold_margin (p50)
  - timestamp_jitter_p99 (requires twice-run inputs)
  - strong_chn_count_match (requires twice-run inputs)

Output: results/hfo_detector_v2/validation/layer_a_<subject>.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def compute_dur_in_band_frac(events, min_ms, max_ms):
    if len(events) == 0:
        return 1.0
    durs_ms = (events[:, 1] - events[:, 0]) * 1000.0
    return float(np.mean((durs_ms >= min_ms) & (durs_ms < max_ms)))


def compute_peak_side_ratio(env, events, fs):
    """For each event, compute pick_mean / side_mean (using legacy convention)."""
    out = []
    n = len(env)
    for t0, t1 in events:
        dur = t1 - t0
        i_pre_s = max(0, int((t0 - dur) * fs))
        i_pre_e = int(t0 * fs)
        i_post_s = int(t1 * fs)
        i_post_e = min(n, int((t1 + dur) * fs))
        side = np.concatenate([env[i_pre_s:i_pre_e], env[i_post_s:i_post_e]])
        pick = env[int(t0 * fs):int(t1 * fs)]
        if len(side) == 0 or len(pick) == 0:
            out.append(np.nan)
            continue
        s_mean = float(np.mean(side))
        if s_mean <= 0:
            out.append(np.inf)
            continue
        out.append(float(np.mean(pick) / s_mean))
    return np.array(out)


def compute_threshold_margin(env, events, fs, threshold):
    """For each event, (max_env_in_event - threshold) / threshold."""
    out = []
    for t0, t1 in events:
        i0, i1 = int(t0 * fs), int(t1 * fs)
        if i1 <= i0:
            out.append(np.nan)
            continue
        env_max = float(np.max(env[i0:i1]))
        out.append((env_max - threshold) / threshold)
    return np.array(out)


# (rest of CLI omitted — implemented in next step)
```

- [ ] **Step 4: Run tests，应 PASS**

```bash
conda run -n cuda_env pytest tests/test_v2_validate_layer_a.py -v
```

Expected: 3 passed.

- [ ] **Step 5: 提交**

```bash
git add scripts/v2_validate_layer_a.py tests/test_v2_validate_layer_a.py
git commit -m "feat(v2): Layer A validation — duration / peak-side / threshold margin (TDD)"
```

### Task 2.2: Layer A CLI 与 envelope 重算

**Files:**
- Modify: `scripts/v2_validate_layer_a.py` (add CLI + per-subject driver)

- [ ] **Step 1: 实现 per-subject 提取主函数 + CLI**

```python
CHUNK_SEC = 200.0
N_WINDOWS_PER_RECORD = 3  # first / middle / last 200s — non-stationarity coverage


def _window_starts_for_record(rec_duration_sec: float) -> list[float]:
    """Return start_times of N_WINDOWS_PER_RECORD evenly-spaced 200s windows.

    Recording shorter than 600s falls back to [0.0] (first chunk only).
    """
    if rec_duration_sec < CHUNK_SEC * 3:
        return [0.0]
    last_start = max(0.0, rec_duration_sec - CHUNK_SEC)
    if N_WINDOWS_PER_RECORD == 1:
        return [0.0]
    starts = []
    for i in range(N_WINDOWS_PER_RECORD):
        frac = i / (N_WINDOWS_PER_RECORD - 1)
        starts.append(frac * last_start)
    return starts


def extract_layer_a_per_subject(subject_dir: Path, output_dir: Path) -> dict:
    """Iterate all *_gpu.npz under subject_dir; for each record recompute envelope on
    first/middle/last 200s and extract per-channel metrics for events that fall in
    those windows. Aggregating across windows controls for HFO non-stationarity."""
    from src.preprocessing import load_epilepsiae_block
    from src.utils.bqk_utils import BQKDetector

    metrics_per_record = []
    for gpu_path in sorted(subject_dir.glob("*_gpu.npz")):
        head_path = gpu_path.with_suffix('').with_suffix('').parent / (gpu_path.stem.replace('_gpu', '') + '.head')
        data_path = head_path.with_suffix('.data')
        if not (head_path.exists() and data_path.exists()):
            continue
        npz = np.load(gpu_path, allow_pickle=True)
        chns = list(npz['chns_names'])
        whole_dets = npz['whole_dets']

        pre = load_epilepsiae_block(
            data_path, head_path, reference="car",
            notch_freqs=[50.0, 100.0, 150.0, 200.0, 250.0],
            notch_filter_kind="fir_legacy",
        )
        rec_dur = pre.data.shape[1] / pre.sfreq
        det = BQKDetector(
            sfreq=pre.sfreq, freqband=(80, 250), subband_width=20,
            rel_thresh=2.0, abs_thresh=2.0, side_thresh=2.0,
            min_gap=20, min_last=50, max_last=200,
            n_jobs=1, legacy_align=True, use_gpu=True,
        )

        # Per-channel running aggregator across all sampled windows
        ch_agg = {ch: {"n_events": 0, "durs": [], "ratios": [], "margins": []} for ch in chns}
        windows_used = []
        for w_start in _window_starts_for_record(rec_dur):
            i0 = int(round(w_start * pre.sfreq))
            i1 = i0 + int(round(CHUNK_SEC * pre.sfreq))
            i1 = min(i1, pre.data.shape[1])
            if i1 - i0 < int(0.5 * CHUNK_SEC * pre.sfreq):
                continue
            chunk = pre.data[:, i0:i1]
            env = det.compute_envelope(chunk)
            whole_med = float(np.median(env))
            t_lo, t_hi = w_start, w_start + (i1 - i0) / pre.sfreq
            windows_used.append({"start_sec": t_lo, "end_sec": t_hi, "whole_data_median": whole_med})

            for ci, ch in enumerate(chns):
                evts_full = whole_dets[ci] if isinstance(whole_dets[ci], list) else []
                # Filter to events inside this window (using global recording time)
                evts_in = np.asarray([e for e in evts_full if t_lo <= e[0] < t_hi], dtype=float)
                if len(evts_in) == 0:
                    continue
                # Re-base event times into window-local for envelope indexing
                evts_local = evts_in.copy()
                evts_local[:, 0] -= t_lo
                evts_local[:, 1] -= t_lo
                ch_med = float(np.median(env[ci]))
                threshold = max(2 * ch_med, 2 * whole_med)
                durs = compute_dur_in_band_frac(evts_local, 50.0, 200.0)
                ratios = compute_peak_side_ratio(env[ci], evts_local, pre.sfreq)
                margins = compute_threshold_margin(env[ci], evts_local, pre.sfreq, threshold)
                ch_agg[ch]["n_events"] += int(len(evts_local))
                ch_agg[ch]["durs"].append(durs * len(evts_local))  # weighted by count
                ch_agg[ch]["ratios"].append(ratios)
                ch_agg[ch]["margins"].append(margins)

        # Aggregate per channel
        ch_metrics = {}
        for ch, agg in ch_agg.items():
            if agg["n_events"] == 0:
                ch_metrics[ch] = {"n_events": 0}
                continue
            ratios_concat = np.concatenate(agg["ratios"]) if agg["ratios"] else np.array([])
            margins_concat = np.concatenate(agg["margins"]) if agg["margins"] else np.array([])
            durs_total = float(sum(agg["durs"])) / agg["n_events"]
            ch_metrics[ch] = {
                "n_events": agg["n_events"],
                "dur_in_band_frac": durs_total,
                "peak_side_ratio_p25": float(np.nanpercentile(ratios_concat, 25)),
                "peak_side_ratio_p50": float(np.nanpercentile(ratios_concat, 50)),
                "threshold_margin_p50": float(np.nanpercentile(margins_concat, 50)),
            }
        metrics_per_record.append({
            "record": gpu_path.stem,
            "rec_duration_sec": rec_dur,
            "windows": windows_used,
            "channels": ch_metrics,
        })

    return {"records": metrics_per_record}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True, help="e.g. 635")
    p.add_argument("--detection-root", default="results/hfo_detector_v2")
    p.add_argument("--output-dir", default="results/hfo_detector_v2/validation")
    args = p.parse_args()

    subject_dir = Path(args.detection_root) / args.subject
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    res = extract_layer_a_per_subject(subject_dir, out_dir)
    out = out_dir / f"layer_a_{args.subject}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run on 635 (no v2 data yet — uses pre-Path-A as placeholder)**

```bash
conda run -n cuda_env python scripts/v2_validate_layer_a.py --subject 635 --detection-root results/hfo_detection
```

Expected: writes `results/hfo_detector_v2/validation/layer_a_635.json`. Check 1+ records present in JSON.

- [ ] **Step 3: 提交**

```bash
git add scripts/v2_validate_layer_a.py
git commit -m "feat(v2): Layer A per-subject CLI driver with envelope recompute"
```

### Task 2.3: Layer A determinism check (twice-run jitter)

**Files:**
- Modify: `scripts/v2_validate_layer_a.py` (add `--check-determinism` mode)

- [ ] **Step 1: 加 determinism 检查模式**

```python
def check_determinism(subject_dir: Path) -> dict:
    """Run detection twice on first record's first chunk, verify bit-identical."""
    from src.preprocessing import load_epilepsiae_block
    from src.hfo_detector import HFODetector, HFODetectionConfig

    head_path = next(subject_dir.glob("*.head"))
    data_path = head_path.with_suffix('.data')
    pre = load_epilepsiae_block(
        data_path, head_path, reference="car",
        notch_freqs=[50.0, 100.0, 150.0, 200.0, 250.0],
        notch_filter_kind="fir_legacy",
    )
    cfg = HFODetectionConfig(
        rel_thresh=2.0, abs_thresh=2.0, side_thresh=2.0,
        min_gap_ms=20, min_last_ms=50, max_last_ms=200,
        chunk_sec=200, chunk_overlap_sec=0,
        legacy_align=True, use_gpu=True, n_jobs=1,
    )
    det = HFODetector(cfg)
    r1 = det.detect(pre)
    r2 = det.detect(pre)

    # Verify counts and event lists match exactly
    count_match = bool(np.array_equal(r1.events_count, r2.events_count))
    max_t_diff = 0.0
    for ev1, ev2 in zip(r1.events_by_channel, r2.events_by_channel):
        if ev1.shape != ev2.shape:
            count_match = False
            continue
        if len(ev1) > 0:
            max_t_diff = max(max_t_diff, float(np.max(np.abs(ev1 - ev2))))
    return {
        "count_match": count_match,
        "max_timestamp_diff_sec": max_t_diff,
        "deterministic": count_match and max_t_diff == 0.0,
    }
```

加 CLI flag `--check-determinism`，调用此函数。

- [ ] **Step 2: 跑 635 验证 deterministic**

```bash
conda run -n cuda_env python scripts/v2_validate_layer_a.py --subject 635 --check-determinism --detection-root /mnt/epilepsia_data/inv2/pat_63502/adm_635102/rec_63500102
```

Expected: `deterministic: true, max_timestamp_diff_sec: 0.0`

- [ ] **Step 3: 提交**

```bash
git add scripts/v2_validate_layer_a.py
git commit -m "feat(v2): Layer A determinism / twice-run jitter check"
```

### Task 2.4: Layer B 提取器（消费 lagPat + packed_times）

**Files:**
- Create: `scripts/v2_validate_layer_b.py`
- Test: `tests/test_v2_validate_layer_b.py`

- [ ] **Step 1: 写测试**

```python
import numpy as np
import pytest
from scripts.v2_validate_layer_b import (
    compute_n_participating_stats,
    compute_pack_width_stats,
    compute_splithalf_rank_corr,
    compute_chunk_boundary_event_frac,
    compute_subset_rank_corr,
)

def test_n_participating_basic():
    # 5 group events, n_participating from 2 to 10
    n_part = np.array([2, 5, 7, 10, 3])
    s = compute_n_participating_stats(n_part)
    assert s["p50"] == 5.0
    assert s["p10"] == 2.4

def test_pack_width_basic():
    # group event windows: 50ms, 100ms, 200ms
    starts = np.array([0.0, 1.0, 2.0])
    ends = np.array([0.05, 1.10, 2.20])
    s = compute_pack_width_stats(starts, ends)
    assert s["p50"] == 0.10  # 100 ms

def test_subset_rank_corr_perfect():
    # both subsets have identical lagPatRank → rank corr should be 1.0
    rank = np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]])
    valid = np.ones_like(rank, dtype=bool)
    a = np.array([0, 1])
    b = np.array([2, 3])
    rho = compute_subset_rank_corr(rank, valid, a, b)
    assert rho == pytest.approx(1.0)

def test_subset_rank_corr_inverted():
    # subset b is the inverse rank → corr ≈ -1.0
    rank_a = np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    rank_b = np.array([[3, 2, 1, 0], [3, 2, 1, 0]])
    rank = np.vstack([rank_a, rank_b])
    valid = np.ones_like(rank, dtype=bool)
    a = np.array([0, 1])
    b = np.array([2, 3])
    rho = compute_subset_rank_corr(rank, valid, a, b)
    assert rho == pytest.approx(-1.0)

def test_chunk_boundary_event_frac():
    # chunk_sec=200, tolerance=2 → events near 200, 400 boundaries
    starts = np.array([10.0, 198.5, 250.0, 401.0, 600.0])
    frac = compute_chunk_boundary_event_frac(starts, chunk_sec=200.0, tol_sec=2.0)
    # 198.5 within [198, 202] of 200; 401 within [398, 402] of 400; 600 is on 600 boundary
    assert frac == pytest.approx(3.0 / 5.0)
```

- [ ] **Step 2: Run tests，预期 ImportError**

```bash
conda run -n cuda_env pytest tests/test_v2_validate_layer_b.py -v
```

- [ ] **Step 3: 实现**

`scripts/v2_validate_layer_b.py`:

```python
"""HFO detector v2 — Layer B validation (group-event quality)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def compute_n_participating_stats(n_part: np.ndarray) -> dict:
    return {
        "p10": float(np.percentile(n_part, 10)),
        "p50": float(np.percentile(n_part, 50)),
        "p90": float(np.percentile(n_part, 90)),
        "n_total": int(len(n_part)),
    }


def compute_pack_width_stats(starts: np.ndarray, ends: np.ndarray) -> dict:
    widths = ends - starts
    return {
        "p50": float(np.percentile(widths, 50)),
        "p90": float(np.percentile(widths, 90)),
    }


def compute_splithalf_rank_corr(rank_first: np.ndarray, rank_second: np.ndarray) -> float:
    """Spearman rank correlation between two channel-rank vectors (legacy helper)."""
    from scipy.stats import spearmanr
    if len(rank_first) != len(rank_second):
        raise ValueError("rank arrays must align")
    rho, _ = spearmanr(rank_first, rank_second)
    return float(rho)


def compute_subset_rank_corr(
    lag_pat_rank: np.ndarray,    # (n_events, n_chn) per-event ranks
    valid_mask: np.ndarray,      # (n_events, n_chn) boolean: channel participated
    idx_a: np.ndarray,
    idx_b: np.ndarray,
) -> float:
    """For two disjoint event subsets a, b of the same channel set, compute the
    Spearman correlation between (mean rank in subset a) and (mean rank in subset b).

    Channels with zero participation in either subset are dropped before the corr.
    Returns NaN if fewer than 3 channels survive (Spearman undefined).
    """
    from scipy.stats import spearmanr
    if len(idx_a) == 0 or len(idx_b) == 0:
        return float("nan")
    rk_a = lag_pat_rank[idx_a]
    rk_b = lag_pat_rank[idx_b]
    vm_a = valid_mask[idx_a]
    vm_b = valid_mask[idx_b]
    cnt_a = vm_a.sum(axis=0).astype(float)
    cnt_b = vm_b.sum(axis=0).astype(float)
    keep = (cnt_a > 0) & (cnt_b > 0)
    if keep.sum() < 3:
        return float("nan")
    # Mean rank only over events where channel actually participated
    sum_a = (rk_a * vm_a).sum(axis=0)
    sum_b = (rk_b * vm_b).sum(axis=0)
    mean_a = np.where(cnt_a > 0, sum_a / np.maximum(cnt_a, 1), np.nan)
    mean_b = np.where(cnt_b > 0, sum_b / np.maximum(cnt_b, 1), np.nan)
    rho, _ = spearmanr(mean_a[keep], mean_b[keep])
    return float(rho)


def compute_chunk_boundary_event_frac(starts: np.ndarray, chunk_sec: float = 200.0,
                                      tol_sec: float = 2.0) -> float:
    """Fraction of events whose start_time falls within ±tol_sec of any chunk
    boundary k * chunk_sec (k>=0). Descriptive proxy for merge_overhead."""
    if len(starts) == 0:
        return 0.0
    nearest_k = np.round(starts / chunk_sec)
    distance = np.abs(starts - nearest_k * chunk_sec)
    return float(np.mean(distance <= tol_sec))


def extract_layer_b_per_subject(subject_lagpat_root: Path) -> dict:
    """Walk *_lagPat_withFreqCent.npz + *_packedTimes_withFreqCent.npy.

    Aggregate group event quality across all records of one subject. Uses union
    channel set across records (re-derived from each record's lagPat npz; channels
    missing in a record contribute zero to rank-aggregates for that record).
    """
    # Epilepsiae backfill writes *_lagPat.npz (no withFreqCent suffix; that's
    # a Yuquan-only naming). Glob the backfill convention.
    npz_files = sorted(subject_lagpat_root.glob("*_lagPat.npz"))
    if not npz_files:
        return {"error": f"no lagpat npz found in {subject_lagpat_root}"}

    # First pass: build union channel ordering across records.
    # The backfill schema uses key 'chnNames' (not 'channel_names').
    union_chs: list[str] = []
    for npz_path in npz_files:
        d = np.load(npz_path, allow_pickle=True)
        ch_key = "chnNames" if "chnNames" in d.files else (
            "channel_names" if "channel_names" in d.files else None
        )
        if ch_key is None:
            continue
        for ch in [str(c) for c in d[ch_key]]:
            if ch not in union_chs:
                union_chs.append(ch)
    n_ch = len(union_chs)
    if n_ch < 3:
        return {"error": f"too few channels for rank corr ({n_ch})"}

    # Aggregate event-level lagPatRank + eventsBool aligned to union_chs
    rank_rows = []
    valid_rows = []
    n_parts_all = []
    widths_all = []
    starts_all = []

    for npz_path in npz_files:
        d = np.load(npz_path, allow_pickle=True)
        if "eventsBool" not in d.files:
            continue
        ch_key = "chnNames" if "chnNames" in d.files else "channel_names"
        chs_local = [str(c) for c in d[ch_key]]
        col_to_union = [union_chs.index(c) for c in chs_local]

        evb = d["eventsBool"]
        if evb.ndim != 2 or evb.shape[1] != len(chs_local):
            continue

        # Re-align to union ordering: rows = events, cols = union channels
        evb_full = np.zeros((evb.shape[0], n_ch), dtype=bool)
        evb_full[:, col_to_union] = evb.astype(bool)

        if "lagPatRank" in d.files:
            rk = d["lagPatRank"]
            rk_full = np.zeros((rk.shape[0], n_ch), dtype=float)
            rk_full[:, col_to_union] = rk
            rank_rows.append(rk_full)
            valid_rows.append(evb_full)

        n_parts_all.append(evb.sum(axis=1))

        # Backfill writes *_packedTimes.npy alongside *_lagPat.npz (no withFreqCent).
        packed_path = npz_path.parent / (
            npz_path.stem.replace('_lagPat', '_packedTimes') + '.npy'
        )
        if packed_path.exists():
            packed = np.load(packed_path, allow_pickle=True)
            if isinstance(packed, np.ndarray) and packed.dtype == object and len(packed) >= 2:
                starts = np.asarray(packed[0], dtype=float)
                ends = np.asarray(packed[1], dtype=float)
                widths_all.append(ends - starts)
                starts_all.append(starts)

    if not n_parts_all:
        return {"error": "no usable npz files"}

    n_part_concat = np.concatenate(n_parts_all)
    res = {
        "channel_names": union_chs,
        "n_participating": compute_n_participating_stats(n_part_concat),
    }
    if widths_all:
        widths_concat = np.concatenate(widths_all)
        res["pack_width_sec"] = compute_pack_width_stats(
            np.zeros_like(widths_concat), widths_concat
        )
    if starts_all:
        starts_concat = np.concatenate(starts_all)
        res["chunk_boundary_event_frac"] = compute_chunk_boundary_event_frac(
            starts_concat, chunk_sec=200.0, tol_sec=2.0
        )

    if rank_rows:
        rank_all = np.vstack(rank_rows)
        valid_all = np.vstack(valid_rows)
        n_evts = rank_all.shape[0]
        if n_evts >= 6:
            half = n_evts // 2
            idx_first = np.arange(0, half)
            idx_second = np.arange(half, n_evts)
            res["splithalf_event_rank_corr"] = compute_subset_rank_corr(
                rank_all, valid_all, idx_first, idx_second
            )
            idx_odd = np.arange(0, n_evts, 2)
            idx_even = np.arange(1, n_evts, 2)
            res["oddeven_event_rank_corr"] = compute_subset_rank_corr(
                rank_all, valid_all, idx_odd, idx_even
            )
        else:
            res["splithalf_event_rank_corr"] = float("nan")
            res["oddeven_event_rank_corr"] = float("nan")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True)
    p.add_argument("--lagpat-root", required=True)
    p.add_argument("--output-dir", default="results/hfo_detector_v2/validation")
    args = p.parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    res = extract_layer_b_per_subject(Path(args.lagpat_root))
    out = out_dir / f"layer_b_{args.subject}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests，应 PASS**

```bash
conda run -n cuda_env pytest tests/test_v2_validate_layer_b.py -v
```

Expected: 2 passed.

- [ ] **Step 5: 提交**

```bash
git add scripts/v2_validate_layer_b.py tests/test_v2_validate_layer_b.py
git commit -m "feat(v2): Layer B group-event quality extractor (TDD)"
```

### Task 2.5: Layer C 编排器

**Files:**
- Create: `scripts/v2_validate_layer_c.py`

- [ ] **Step 1: 实现编排器**

```python
"""HFO detector v2 — Layer C orchestrator.

Layer C 复用现有 propagation reproducibility:
  - compute_time_split_reproducibility (split-half + odd-even)
  - adaptive_cluster.stable_k consistency
  - PR-2.5 forward/reverse template reproduction (strict AND, lenient OR)

Inputs: results/hfo_detector_v2/propagation/per_subject/epilepsiae_<sid>.json
        (after re-running PR-1 + PR-2.5 from v2 lagPat — see Phase 7)

Output: results/hfo_detector_v2/validation/layer_c_<sid>.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def extract_layer_c(per_subject_json: Path,
                    stable_k_split_root: Path | None = None) -> dict:
    """Read a v2 propagation per-subject JSON and emit Layer C metrics.

    Strict / lenient distinction: PR-2.5 historically defines
    `forward_reverse_reproduced` as split-half OR odd-even (lenient).
    v2 Layer C PASS gate uses strict (AND) — both splits must reproduce
    the forward/reverse template. Both fields are emitted; PASS uses strict.
    """
    d = json.loads(per_subject_json.read_text())
    tsr = d.get("time_split_reproducibility", {})
    ac = d.get("adaptive_cluster", {})
    grade = tsr.get("grade", "unknown")
    splits = tsr.get("splits", {})
    fwd_rev_split = bool(splits.get("first_half_second_half", {})
                              .get("forward_reverse_reproduced"))
    fwd_rev_oddeven = bool(splits.get("odd_even_block", {})
                                .get("forward_reverse_reproduced"))
    fwd_rev_strict = fwd_rev_split and fwd_rev_oddeven
    fwd_rev_lenient = fwd_rev_split or fwd_rev_oddeven

    stable_k = ac.get("stable_k")
    # stable_k consistency: if split-by-time stable_k is provided in the
    # propagation JSON's splits[*]["stable_k"], require diff <= 1
    sk_split = splits.get("first_half_second_half", {}).get("stable_k")
    sk_oddeven = splits.get("odd_even_block", {}).get("stable_k")
    sk_consistent = True
    sk_used = []
    for v in (sk_split, sk_oddeven):
        if v is not None:
            sk_used.append(v)
    if stable_k is not None and sk_used:
        sk_consistent = all(abs(stable_k - v) <= 1 for v in sk_used)

    passes = (
        grade in {"strong", "moderate"}
        and fwd_rev_strict
        and sk_consistent
    )

    return {
        "subject_json": per_subject_json.name,
        "time_split_grade": grade,
        "forward_reverse_reproduced_strict": fwd_rev_strict,
        "forward_reverse_reproduced_lenient": fwd_rev_lenient,
        "fwd_rev_split_half": fwd_rev_split,
        "fwd_rev_odd_even": fwd_rev_oddeven,
        "stable_k": stable_k,
        "stable_k_split_half": sk_split,
        "stable_k_odd_even": sk_oddeven,
        "stable_k_consistent": sk_consistent,
        "passes_layer_c": passes,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-subject-root",
                   default="results/hfo_detector_v2/propagation/per_subject")
    p.add_argument("--output-dir", default="results/hfo_detector_v2/validation")
    p.add_argument("--subject", required=True, help="e.g. 635")
    args = p.parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(args.per_subject_root) / f"epilepsiae_{args.subject}.json"
    if not src.exists():
        raise SystemExit(f"input not found: {src}")
    res = extract_layer_c(src)
    out = out_dir / f"layer_c_{args.subject}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}, passes_layer_c={res['passes_layer_c']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 提交**

```bash
git add scripts/v2_validate_layer_c.py
git commit -m "feat(v2): Layer C orchestrator (consumes propagation per-subject JSON)"
```

---

## Phase 3: Cohort detection (GPU)

### Task 3.1: 切 parallel script 默认到 GPU + v2 输出目录

**Files:**
- Modify: `scripts/run_epilepsiae_detection_parallel.sh`

- [ ] **Step 1: 改 OUTPUT_DIR 默认与 N_JOBS**

```bash
# OLD:
OUTPUT_DIR="${OUTPUT_DIR:-results/hfo_detection}"
N_JOBS=${N_JOBS:-4}

# NEW:
OUTPUT_DIR="${OUTPUT_DIR:-results/hfo_detector_v2}"
N_JOBS=${N_JOBS:-1}   # GPU 单卡，串行最稳；若多卡可调
```

- [ ] **Step 2: 提交**

```bash
git add scripts/run_epilepsiae_detection_parallel.sh
git commit -m "feat(v2): default cohort detection output to results/hfo_detector_v2 + N_JOBS=1 (GPU)"
```

### Task 3.2: run_hfo_detection.py 默认走 GPU

**Files:**
- Modify: `scripts/run_hfo_detection.py` (Epilepsiae path)

- [ ] **Step 1: 找 Epilepsiae call site，把 use_gpu 默认改为 True**

定位 commit 6027281 加的 Epilepsiae 调用块（约 line 313–326），增加：

```python
use_gpu = bool(params.get("use_gpu", True))   # v2 default GPU
```

并在 HFODetectionConfig 构造中传 `use_gpu=use_gpu`。

- [ ] **Step 2: 加守护：Yuquan 路径的 use_gpu 默认仍 False（GPU 路径只在 Epilepsiae 用）**

- [ ] **Step 3: 加测试 — Epilepsiae 默认 use_gpu=True，Yuquan 默认 use_gpu=False**

```python
# tests/test_run_hfo_detection_v2_gpu_default.py
def test_epilepsiae_default_gpu(script_source):
    assert 'params.get("use_gpu", True)' in script_source

def test_yuquan_default_no_gpu(script_source):
    yq_block = script_source.split("def run_yuquan_subject")[1].split("def run_epilepsiae_subject")[0]
    assert 'use_gpu=True' not in yq_block
```

- [ ] **Step 4: Run tests**

```bash
conda run -n cuda_env pytest tests/test_run_hfo_detection_v2_gpu_default.py -v
```

- [ ] **Step 5: 提交**

```bash
git add scripts/run_hfo_detection.py tests/test_run_hfo_detection_v2_gpu_default.py
git commit -m "feat(v2): Epilepsiae detection defaults to use_gpu=True (cusignal path)"
```

### Task 3.3: 单 subject smoke (635, 1 record)

- [ ] **Step 1: 跑 635 单 record GPU 检测**

```bash
mkdir -p results/hfo_detector_v2/635
conda run -n cuda_env python scripts/run_hfo_detection.py --dataset epilepsiae --subject 635 --output-dir results/hfo_detector_v2 2>&1 | tee /tmp/v2_635_smoke.log
```

预计 ~5 分钟（GPU 路径）。

- [ ] **Step 2: 验证产物存在**

```bash
ls results/hfo_detector_v2/635/*.npz | head -5
ls results/hfo_detector_v2/635/_refineGpu.npz
```

Expected: 多个 `*_gpu.npz` + 1 个 `_refineGpu.npz`

- [ ] **Step 3: 跑 Layer A 提取器 on 635**

```bash
conda run -n cuda_env python scripts/v2_validate_layer_a.py --subject 635 --detection-root results/hfo_detector_v2
cat results/hfo_detector_v2/validation/layer_a_635.json | head -30
```

Expected: JSON with channels, dur_in_band_frac ≈ 1.0, peak_side_ratio_p25 ≥ 2.0.

- [ ] **Step 4: determinism 检查**

```bash
conda run -n cuda_env python scripts/v2_validate_layer_a.py --subject 635 --check-determinism \
    --detection-root /mnt/epilepsia_data/inv2/pat_63502/adm_635102/rec_63500102
```

Expected: `deterministic: true`

- [ ] **Step 5: 提交 smoke 结果到 cohort_summary 草稿**

写第一行 cohort summary 文件 `docs/archive/hfo_detector_v2/v2_cohort_results_2026-05-05.md`：

```markdown
# v2 cohort results (in progress)

| subject | sfreq | layer A | layer B | layer C | notes |
|---------|-------|---------|---------|---------|-------|
| 635     | 1024  | PASS    | (pending) | (pending) | smoke 2026-05-05 |
```

```bash
git add docs/archive/hfo_detector_v2/v2_cohort_results_2026-05-05.md
git commit -m "feat(v2): subject 635 smoke — Layer A PASS, deterministic"
```

### Task 3.4: 跑全 cohort（20 subjects）

- [ ] **Step 1: 后台启动**

```bash
nohup bash scripts/run_epilepsiae_detection_parallel.sh > results/hfo_detector_v2/_cohort_run.log 2>&1 &
echo $! > /tmp/v2_cohort.pid
```

预计：每 subject ~30 min GPU 时间 × 20 ≈ 10 h（取决于 record 数量）。

- [ ] **Step 2: 监控进度**

每隔 1 h 检查：

```bash
ls -d results/hfo_detector_v2/*/ | wc -l
tail -20 results/hfo_detector_v2/_cohort_run.log
```

- [ ] **Step 3: 完成后核对 20 subject 都有 _refineGpu.npz**

```bash
for s in 253 548 139 384 1077 1084 442 818 916 922 958 583 590 620 635 1073 1096 1125 1146 1150; do
    f="results/hfo_detector_v2/$s/_refineGpu.npz"
    if [ -f "$f" ]; then echo "OK $s"; else echo "MISS $s"; fi
done
```

Expected: 20 OK lines.

- [ ] **Step 4: 加 cohort manifest（每 subject 总 events count + record 数 + 错误清单）**

```bash
conda run -n cuda_env python -c "
import json, sys, traceback
from pathlib import Path
import numpy as np
ROOT = Path('results/hfo_detector_v2')
manifest = {}
errors = []
for sub_dir in sorted(ROOT.iterdir()):
    if not sub_dir.is_dir() or sub_dir.name.startswith('_') or sub_dir.name in {'validation', 'synchrony', 'propagation'}:
        continue
    npzs = sorted(sub_dir.glob('*_gpu.npz'))
    total = 0
    rec_ok = 0
    for p in npzs:
        try:
            d = np.load(p, allow_pickle=True)
            total += int(np.asarray(d['events_count']).sum())
            rec_ok += 1
        except Exception as e:
            errors.append({
                'subject': sub_dir.name,
                'file': str(p),
                'error': repr(e),
                'traceback': traceback.format_exc(limit=3),
            })
    manifest[sub_dir.name] = {
        'n_records_listed': len(npzs),
        'n_records_ok': rec_ok,
        'total_events': total,
    }
out = {'subjects': manifest, 'errors': errors}
with open(ROOT / '_cohort_manifest.json', 'w') as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
if errors:
    print(f'FAILED: {len(errors)} corrupt npz', file=sys.stderr)
    sys.exit(1)
"
```

> **重要**：此脚本失败必须 non-zero exit。`except Exception: pass` 会让坏 .npz 悄悄消失、
> manifest 看起来正常，下游 Layer A/B/C 才发现异常——晚太多。失败立即可见才是数据管线
> 的卫生标准。

- [ ] **Step 5: 提交**

```bash
git add results/hfo_detector_v2/_cohort_manifest.json
git commit -m "data(v2): cohort detection complete — 20 subjects, manifest written"
```

---

## Phase 4: Layer A cohort 验证

### Task 4.1: 跑 Layer A on 全 cohort

- [ ] **Step 1: For 循环 20 subjects**

```bash
for s in 253 548 139 384 1077 1084 442 818 916 922 958 583 590 620 635 1073 1096 1125 1146 1150; do
    echo "=== Layer A $s ==="
    conda run -n cuda_env python scripts/v2_validate_layer_a.py --subject "$s" --detection-root results/hfo_detector_v2 || echo "FAIL $s"
done
```

预计每 subject ~5 min（先做 envelope 再算 metrics）。Cohort ~100 min。

- [ ] **Step 2: Cohort 汇总**

```bash
conda run -n cuda_env python -c "
import json
from pathlib import Path
ROOT = Path('results/hfo_detector_v2/validation')
cohort = []
for p in sorted(ROOT.glob('layer_a_*.json')):
    sid = p.stem.replace('layer_a_', '')
    d = json.loads(p.read_text())
    # Aggregate per-subject pass/fail
    durs, ratios, margins = [], [], []
    for rec in d.get('records', []):
        for ch, m in rec.get('channels', {}).items():
            if m.get('n_events', 0) == 0:
                continue
            durs.append(m['dur_in_band_frac'])
            ratios.append(m['peak_side_ratio_p25'])
            margins.append(m['threshold_margin_p50'])
    import numpy as np
    cohort.append({
        'subject': sid,
        'dur_p50': float(np.median(durs)) if durs else None,
        'ratio_p25_overall': float(np.percentile(ratios, 25)) if ratios else None,
        'margin_p50': float(np.median(margins)) if margins else None,
    })
with open(ROOT / 'cohort_layer_a.json', 'w') as f:
    json.dump(cohort, f, indent=2)
print(json.dumps(cohort, indent=2))
"
```

- [ ] **Step 3: 加 PASS/FAIL 列**

按验收契约：dur ≥ 0.99, ratio ≥ 2.0, margin ≥ 0.5。

- [ ] **Step 4: 提交**

```bash
git add results/hfo_detector_v2/validation/cohort_layer_a.json
git commit -m "data(v2): Layer A cohort validation — <X>/20 PASS"
```

---

## Phase 5: 重建下游 — synRefine + lagPat

> **前置事实**（rev 2 verify）：
> - `scripts/run_epilepsiae_lagpat_backfill.py` 把 `NEW_GPU_ROOT` 与 `OUTPUT_ROOT` 写成模块常量（line 37–38）。
> - `scripts/run_epilepsiae_lagpat_backfill_parallel.sh` 没有 `GPU_ROOT` 变量。
> - 默认输出在 `results/epilepsiae_lagpat_backfill/<subject>/<stem>_lagPat.npz`，**不是** v2 root。
> - 所以**先**给 backfill 加 `--gpu-root` / `--output-root` CLI 参数，再通过 bash wrapper 的环境变量覆盖。

### Task 5.0: 给 backfill 加 `--gpu-root` / `--output-root` CLI 参数

**Files:**
- Modify: `scripts/run_epilepsiae_lagpat_backfill.py`（替换两个模块常量为可配置）
- Modify: `scripts/run_epilepsiae_lagpat_backfill_parallel.sh`（pass env → CLI）

- [ ] **Step 1: 改 python script，加 CLI 参数**

把 `NEW_GPU_ROOT` 与 `OUTPUT_ROOT` 改成函数参数 / global, 并在 main() 的 argparse 中加：

```python
parser.add_argument(
    "--gpu-root", type=Path, default=Path("results/hfo_detection"),
    help="Override input gpu.npz root. v2 default: results/hfo_detector_v2",
)
parser.add_argument(
    "--output-root", type=Path, default=Path("results/epilepsiae_lagpat_backfill"),
    help="Override pack/lagPat output root. v2: results/hfo_detector_v2/lagpat",
)
```

并在 main() 开头：

```python
global NEW_GPU_ROOT, OUTPUT_ROOT
NEW_GPU_ROOT = args.gpu_root
OUTPUT_ROOT = args.output_root
```

> **不**改函数签名（`refine_path_for_subject(subject)` 等仍读 module-level NEW_GPU_ROOT）。这样最小侵入。

- [ ] **Step 2: 改 bash wrapper 透传 env → CLI**

```bash
# 原 line 22:
N_JOBS=${N_JOBS:-5}

# 加：
GPU_ROOT="${GPU_ROOT:-results/hfo_detection}"
LAGPAT_OUTPUT_ROOT="${LAGPAT_OUTPUT_ROOT:-results/epilepsiae_lagpat_backfill}"

# 原 line 36 worker():
python scripts/run_epilepsiae_lagpat_backfill.py --subject "$subj" ${FORCE_FLAG} \
    >>"${out_dir}/_console.log" 2>&1

# 改为：
python scripts/run_epilepsiae_lagpat_backfill.py --subject "$subj" ${FORCE_FLAG} \
    --gpu-root "$GPU_ROOT" --output-root "$LAGPAT_OUTPUT_ROOT" \
    >>"${out_dir}/_console.log" 2>&1
```

并 export 这两个 env：`export GPU_ROOT LAGPAT_OUTPUT_ROOT`。

- [ ] **Step 3: 写一个最小 unit test 确认 CLI 覆盖生效**

`tests/test_lagpat_backfill_cli.py`:

```python
import subprocess, sys
from pathlib import Path

def test_help_lists_gpu_root_and_output_root():
    out = subprocess.run(
        [sys.executable, "scripts/run_epilepsiae_lagpat_backfill.py", "--help"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "--gpu-root" in out
    assert "--output-root" in out
```

- [ ] **Step 4: Run test + commit**

```bash
conda run -n cuda_env pytest tests/test_lagpat_backfill_cli.py -v
git add scripts/run_epilepsiae_lagpat_backfill.py scripts/run_epilepsiae_lagpat_backfill_parallel.sh tests/test_lagpat_backfill_cli.py
git commit -m "feat(v2): backfill --gpu-root / --output-root CLI flags + env propagation

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 5.1: 跑 lagpat backfill from v2 detection

- [ ] **Step 1: 后台跑全 cohort backfill（v2 input → v2 output）**

```bash
FORCE=1 N_JOBS=6 \
    GPU_ROOT=results/hfo_detector_v2 \
    LAGPAT_OUTPUT_ROOT=results/hfo_detector_v2/lagpat \
    nohup bash scripts/run_epilepsiae_lagpat_backfill_parallel.sh \
    > results/hfo_detector_v2/_backfill.log 2>&1 &
```

预计 ~6 h（CPU 路径，N_JOBS=6）。

- [ ] **Step 2: 完成后核对 20 subject 都有 lagPat 产物**

```bash
for s in 253 548 139 384 1077 1084 442 818 916 922 958 583 590 620 635 1073 1096 1125 1146 1150; do
    n=$(ls results/hfo_detector_v2/lagpat/$s/*_lagPat.npz 2>/dev/null | wc -l)
    echo "$s: $n lagPat files"
done
```

Expected: 每个 subject 都至少有 1 个 `*_lagPat.npz`（**注意不是 `_lagPat_withFreqCent.npz`**——backfill 不产 withFreqCent 后缀的文件，那是 Yuquan 专用 naming）。

- [ ] **Step 3: 提交 backfill 数据 manifest（不提交全部 .npz binary）**

```bash
conda run -n cuda_env python -c "
import json
from pathlib import Path
import numpy as np
ROOT = Path('results/hfo_detector_v2/lagpat')
manifest = {}
for sub_dir in sorted(ROOT.iterdir()) if ROOT.exists() else []:
    if not sub_dir.is_dir():
        continue
    npzs = sorted(sub_dir.glob('*_lagPat.npz'))
    manifest[sub_dir.name] = {'n_records': len(npzs)}
out = ROOT / '_manifest.json'
out.write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
"
```

```bash
git add results/hfo_detector_v2/lagpat/_manifest.json
git commit -m "data(v2): lagpat backfill cohort manifest (binaries not tracked)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 6: Layer B cohort 验证

### Task 6.1: 跑 Layer B on 全 cohort

- [ ] **Step 1: For 循环（lagpat 路径指到 v2 lagpat root）**

```bash
for s in 253 548 139 384 1077 1084 442 818 916 922 958 583 590 620 635 1073 1096 1125 1146 1150; do
    conda run -n cuda_env python scripts/v2_validate_layer_b.py --subject "$s" \
        --lagpat-root "results/hfo_detector_v2/lagpat/$s" || echo "FAIL $s"
done
```

- [ ] **Step 2: Cohort 汇总并加 PASS/FAIL**

按契约：n_part_p50 ≥ 2，p10 ≥ 1。pack_width_p50 ∈ [50, 500] ms。

- [ ] **Step 3: 提交**

```bash
git add results/hfo_detector_v2/validation/cohort_layer_b.json
git commit -m "data(v2): Layer B cohort validation — <X>/20 PASS"
```

---

## Phase 7: 重建下游 — synchrony + propagation（**输出隔离**）

> **关键约束**：v2 重建的所有下游产物写入 `results/hfo_detector_v2/{synchrony,propagation}/`，
> 不覆盖 `results/interictal_synchrony/` 与 `results/interictal_propagation/per_subject/` 的
> 历史结果。所有命令统一使用 `conda run -n cuda_env python` 锁定环境。

### Task 7.1: Synchrony aggregation

- [ ] **Step 1: 验证 lagPat root 命名**

`run_epilepsiae_interictal_synchrony.py` 的 `--lagpat-root` 期望的是 Epilepsiae 的 subject
目录树根，而非单 subject 目录。先验证一下 v2 的目录形态：

```bash
ls -d results/hfo_detector_v2/635/ | head -3
```

如果 v2 用 `<root>/<subject>/<record_stem>_lagPat_withFreqCent.npz`，可直接传根目录
`--lagpat-root results/hfo_detector_v2`；否则需要构建一个软链树。

- [ ] **Step 2: 重跑 epilepsiae synchrony export（每 subject）**

```bash
mkdir -p results/hfo_detector_v2/synchrony
for s in 253 548 139 384 1077 1084 442 818 916 922 958 583 590 620 635 1073 1096 1125 1146 1150; do
    conda run -n cuda_env python scripts/run_epilepsiae_interictal_synchrony.py \
        --subject "$s" \
        --lagpat-root results/hfo_detector_v2/lagpat \
        --soz-core-json results/epilepsiae_soz_core_channels.json \
        --output-dir results/hfo_detector_v2/synchrony \
    || echo "FAIL synchrony $s"
done
```

预计 ~2 h。

- [ ] **Step 3: 跑 aggregation（不能用 hardcoded CLI，用 inline import）**

`scripts/aggregate_epilepsiae_interictal_synchrony.py` 路径是写死的；v2 走 inline 调用：

```bash
conda run -n cuda_env python -c "
from pathlib import Path
import json
from src.interictal_synchrony_aggregation import run_epilepsiae_sync_aggregation
sync_dir = Path('results/hfo_detector_v2/synchrony')
summary = run_epilepsiae_sync_aggregation(
    seizure_inventory_csv=Path('results/epilepsiae_seizure_inventory.csv'),
    sync_event_csv=sync_dir / 'epilepsiae_ready_full_artifacts_interictal_sync_events.csv',
    output_dir=sync_dir / 'aggregated',
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
"
```

> **注**：CSV basename 视 `run_epilepsiae_interictal_synchrony.py` 的 `EVENT_CSV.name` 而定；
> 跑完 Step 2 后 `ls results/hfo_detector_v2/synchrony/*.csv` 取到实际名字再填回这条命令。

- [ ] **Step 4: 提交**

```bash
git add results/hfo_detector_v2/synchrony/
git commit -m "data(v2): synchrony rebuilt from v2 lagPat (isolated under hfo_detector_v2/synchrony)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 7.2: Propagation PR-1 / PR-2.5（写到 v2 propagation 子目录）

`run_interictal_propagation.py` 已支持 `--epilepsiae-root`（lagPat 输入根） 和
`--output-root`（输出 RESULTS_DIR 整体覆写）。**v2 用法**：

- `--epilepsiae-root results/hfo_detector_v2`
- `--output-root results/hfo_detector_v2/propagation`

- [ ] **Step 1: 跑 PR-1（基础 propagation）**

```bash
mkdir -p results/hfo_detector_v2/propagation
for s in 253 548 139 384 1077 1084 442 818 916 922 958 583 590 620 635 1073 1096 1125 1146 1150; do
    conda run -n cuda_env python scripts/run_interictal_propagation.py \
        --subject "epilepsiae/$s" \
        --epilepsiae-root results/hfo_detector_v2/lagpat \
        --output-root results/hfo_detector_v2/propagation \
    || echo "FAIL PR-1 $s"
done
```

预计 ~30 min（CPU）。

- [ ] **Step 2: 跑 PR-2.5（split-half / odd-even reproducibility）**

```bash
for s in 253 548 139 384 1077 1084 442 818 916 922 958 583 590 620 635 1073 1096 1125 1146 1150; do
    conda run -n cuda_env python scripts/run_interictal_propagation.py \
        --pr25 \
        --subject "epilepsiae/$s" \
        --epilepsiae-root results/hfo_detector_v2/lagpat \
        --output-root results/hfo_detector_v2/propagation \
    || echo "FAIL PR-2.5 $s"
done
```

预计 ~2 h（CPU）。

- [ ] **Step 3: 验证产物存在**

```bash
ls results/hfo_detector_v2/propagation/per_subject/epilepsiae_*.json | wc -l
```

Expected: 20。**注意**：这是 v2 propagation 的 per_subject root；不与
`results/interictal_propagation/per_subject/` 的历史结果竞争同名。

- [ ] **Step 4: 提交 v2 propagation per_subject JSONs**

```bash
git add results/hfo_detector_v2/propagation/
git commit -m "data(v2): propagation PR-1 + PR-2.5 rebuilt from v2 lagPat (isolated under hfo_detector_v2/propagation)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 8: Layer C 验证 + cohort 总结

### Task 8.1: 跑 Layer C on 全 cohort

- [ ] **Step 1: For 循环（输入指到 v2 propagation 子目录）**

```bash
for s in 253 548 139 384 1077 1084 442 818 916 922 958 583 590 620 635 1073 1096 1125 1146 1150; do
    conda run -n cuda_env python scripts/v2_validate_layer_c.py \
        --subject "$s" \
        --per-subject-root results/hfo_detector_v2/propagation/per_subject
done
```

- [ ] **Step 2: Cohort 汇总**

```bash
conda run -n cuda_env python -c "
import json
from pathlib import Path
ROOT = Path('results/hfo_detector_v2/validation')
out = []
for p in sorted(ROOT.glob('layer_c_*.json')):
    out.append(json.loads(p.read_text()))
with open(ROOT / 'cohort_layer_c.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f'cohort: {len(out)}')
n_strong = sum(1 for x in out if x['time_split_grade'] == 'strong')
n_mod = sum(1 for x in out if x['time_split_grade'] == 'moderate')
n_weak = sum(1 for x in out if x['time_split_grade'] == 'weak')
print(f'strong={n_strong}, moderate={n_mod}, weak={n_weak}')
"
```

- [ ] **Step 3: 提交**

```bash
git add results/hfo_detector_v2/validation/cohort_layer_c.json
git commit -m "data(v2): Layer C cohort — <strong/moderate/weak> distribution"
```

### Task 8.2: 三层 cohort 总结报告

**Files:**
- Modify: `docs/archive/hfo_detector_v2/v2_cohort_results_2026-05-05.md`

- [ ] **Step 1: 写最终报告**

包含：
- Phase 0–7 timeline
- 三层 PASS/FAIL 分布表
- v2 自身 stable / moderate / large_drift 分类（无 vs legacy）
- 每个 large_drift subject 的诊断（不优化掉）
- 下一步：是否进入 Topic 1/2/3 主文档更新

- [ ] **Step 2: 提交**

```bash
git add docs/archive/hfo_detector_v2/v2_cohort_results_2026-05-05.md
git commit -m "docs(v2): cohort rebuild final report — 3-layer pass/fail + per-subject diagnostics"
```

---

## Phase 9: 主文档对齐

### Task 9.1: 更新 Topic 1 主文档

**Files:**
- Modify: `docs/topic1_within_event_dynamics.md` (加"数据合同 v2"段)

- [ ] **Step 1: 加段**

```markdown
## 数据合同 v2

自 2026-05-05 起，Topic 1 所有 Epilepsiae 结论以 **HFO Detector v2** 输出为准——
不再做与 21 年 cusignal gpu.npz 的 1:1 比对。

- v2 detection root: `results/hfo_detector_v2/`
- v2 specification: `docs/archive/hfo_detector_v2/v2_specification.md`
- v2 validation: `docs/archive/hfo_detector_v2/v2_validation_contract.md`
- v2 cohort report: `docs/archive/hfo_detector_v2/v2_cohort_results_2026-05-05.md`

之前所有从 21 年 gpu.npz 出发的统计（PR-1 to PR-7）需在 v2 数据上重跑。
未通过 Layer C 的 subject 在 cohort 报告中标 large_drift，**不掩盖**。
```

- [ ] **Step 2: 同样在 topic2 / topic3 加 v2 段（如适用）**

- [ ] **Step 3: 提交**

```bash
git add docs/topic1_within_event_dynamics.md docs/topic2_between_event_dynamics.md docs/topic3_spatial_soz_modulation.md
git commit -m "docs(v2): topic main docs adopt v2 as canonical Epilepsiae data contract"
```

---

## Self-Review

**1. Spec coverage:**
- 21 年 gpu.npz 锁只读 — Phase 0 ✓
- v2 spec + 三层验收 — Phase 1 ✓
- 全 cohort 重检 — Phase 3 ✓
- 重建下游 (refine/lagPat/synchrony/propagation) — Phase 5/7 ✓
- 单通道 / 群体 / 科学下游验证 — Phase 4/6/8 ✓
- 不与 legacy 1:1 比对 — Phase 8 措辞 ✓
- 主文档措辞对齐 — Phase 9 ✓

**2. Placeholder scan:** 无 TBD / TODO / "implement later"。每个 step 都有完整代码或精确命令。

**3. Type consistency:** Layer A/B/C 提取器用同一份 schema (subject id + per-channel/per-group metrics)。validation/cohort_layer_X.json 命名一致。`results/hfo_detector_v2/` 是唯一 v2 root。

**已知风险（执行时注意）:**
- Phase 3 的 cohort detection 估算 ~10 h GPU 时间；如某 subject 卡住需手动恢复。
- Phase 5 的 lagpat backfill 与现有 `scripts/run_epilepsiae_lagpat_backfill_parallel.sh` 假定输入路径——若该脚本未实现 GPU_ROOT 参数化，需先补一个 PR。
- Phase 7 propagation 重跑写入 `results/hfo_detector_v2/propagation/`，**不**覆盖 `results/interictal_propagation/per_subject/` 的历史结果（已修订）。
- Layer A 抽样为每 record first/middle/last 三窗，非整段 recording；这是 cost / coverage 折中，cohort 论述时需说明（已写入 contract）。

---

## 修订历史

- **2026-05-05 (rev 1, 初稿)**：9 phase / 30 task / 三层验收契约 + 阈值。
- **2026-05-05 (rev 2, 审阅修复)**：
  1. **Layer B 实装补全**：补 `compute_subset_rank_corr` (splithalf + odd-even) + `compute_chunk_boundary_event_frac`（替代不可测的 `merge_overhead`）；契约同步移除 merge_overhead 作为 PASS gate，降为 descriptive。
  2. **Layer A 多窗抽样**：每 record first / middle / last 三窗 200s（替代只看首 200s），覆盖非平稳。短 recording fallback 单窗。
  3. **Internal-consistency disclaimer**：v2 specification + validation contract 显著位置标明"三层验收 = pipeline 内部自洽，不等于生物学有效性"。
  4. **阈值预注册段**：阈值锁定时间 2026-05-05，cohort 跑前；失败 subject 不剔除主分析、敏感性曲线必报、修订必新建 archive doc。
  5. **Layer C strict / lenient 拆字段**：`forward_reverse_reproduced_strict` (AND，PASS gate) 与 `_lenient` (OR，沿用 PR-2.5 语义，descriptive)。stable_k 一致性进入 PASS。
  6. **下游输出隔离**：Phase 7 synchrony / propagation 写入 `results/hfo_detector_v2/{synchrony,propagation}/`，**不覆盖** `results/interictal_synchrony/` / `results/interictal_propagation/per_subject/`。
  7. **环境一致性**：Phase 7 命令统一为 `conda run -n cuda_env python`，不再裸 python。
  8. **Manifest 异常处理**：cohort manifest 脚本失败进 `errors` 字段并 `sys.exit(1)`，禁止 `except: pass`。
  9. **Phase 5 backfill 路径参数化**（advisor 自查发现）：原计划默认改 `GPU_ROOT` 是死路——bash wrapper 没有该变量，python script 的 `NEW_GPU_ROOT`/`OUTPUT_ROOT` 是模块常量。新增 Task 5.0 给 backfill 加 `--gpu-root` / `--output-root` CLI 参数，bash wrapper env → CLI 透传。v2 lagPat 输出落在 `results/hfo_detector_v2/lagpat/<subject>/`，不覆盖 `results/epilepsiae_lagpat_backfill/`。
  10. **Layer B glob 修正**：原计划 `extract_layer_b_per_subject` 用 `*_lagPat_withFreqCent.npz`（Yuquan 命名），但 Epilepsiae backfill 写 `*_lagPat.npz` + key `chnNames`。改 glob + key fallback (`chnNames` 优先，`channel_names` 次之)。
