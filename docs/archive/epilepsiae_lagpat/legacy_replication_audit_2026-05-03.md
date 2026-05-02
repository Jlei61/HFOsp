# Legacy 复刻审计 — Stage C 差异根因定位（2026-05-03）

> 计划：[`epilepsiae_lagpat_backfill_plan_2026-04-29.md`](./epilepsiae_lagpat_backfill_plan_2026-04-29.md) §0 + §3 B
> 上一阶段：[`stage_d_smoke_2026-05-02.md`](./stage_d_smoke_2026-05-02.md)
> 起因：用户拒绝把 Stage C "12 stable / 4 moderate / 4 large_drift / decision = enter_smoke" 当作完全通过。要求复刻 `/home/honglab/leijiaxin/HFOsp/ReplayIED/inter_events/epilepsiae_interictal/` 的老代码逐步比对。

## 执行摘要

**Detector 层 OK**（per-record `gpu.npz` events_count 与 legacy 在 ~5% 误差内）。
**所有差异来自 refine + pack + lagPat 的下游链路**，共定位 **5 处 legacy 合同违反**。

```
subject 253 单 record（25300102_0000）events_count 对比：
  HLC1   legacy=1338  new=1341  ratio=1.00
  HRB1   legacy=559   new=547   ratio=0.98
  HRB2   legacy=404   new=395   ratio=0.98
  HRA5   legacy=256   new=237   ratio=0.93
  ... 29/29 channels shared, 0 added, 0 removed
  total: legacy=4127, new=3942 (95% match)
```

但 **subject-level `sub_refineGpu.npz` events_count** 完全不同（同样 29 chns）：

```
                legacy            new
  rank 1     HRB2  97622       HLC1  66589
  rank 2     HRB1  81665       HLB2  65800
  rank 3     HRC2  52333       HLB1  64332
  rank 4     HLC1  43648       HLA2  47504
  rank 5     HRC1  41254       HRB2  35950
```

→ 差异 **不在 detector 层**，**在 detector 之后**。下面 5 处定位。

## 5 处 Legacy 合同违反（按影响优先级）

### Δ1. `refine_packedEvents_byAllBool(thresh=0.7)` 同步噪声拒绝完全缺失（最严重）

**Legacy 实现** (`hfo_net.py:559-581`)：

```python
def refine_packedEvents_byAllBool(allEvents_times, packedEventsTimes, fs, thresh=0.7):
    # 对每个 packed window，计算所有通道里有多少个有事件
    bool_sum = np.sum(bool_matrix, axis=0)
    # 如果 ≥70% 通道都有事件 -> synchronized noise → drop
    keep_index = np.where(bool_sum < (thresh * len(allEvents_times)))[0]
    return packedEventsTimes[keep_index]
```

**Legacy 调用 2 次**：
- `epilepsiae_synRefine_supressAllSyn.py:96`：在 synRefine 阶段，pack 之后、reHist 之前
- `epilepsiae_packGroupEvents_supressAllSyn_withFreqCenter.py:297`：在 lagPat 生产阶段，pack 之后、pick_noOverlap 之前

**新 pipeline**：完全没有这个函数，也没有任何等价逻辑。`grep -rn "refine_packedEvents_byAllBool|0\.7" src/` 全部空。

**影响**：所有"全通道同步爆发"的 packed window 在 legacy 里被丢弃，在新 pipeline 里**保留**。这些 windows 大概率是噪声/动作伪迹，会污染 lagPat。

### Δ2. 每 subject 调优的 `pickChn_thresh` 缺失

**Legacy** (`epilepsiae_packGroupEvents_supressAllSyn_withFreqCenter.py:459-460`)：

```python
sub_pickT_list = {
    '1096':1, '1084':1, '958':1, '922':1, '590':1, '1150':1, '442':1,
    '1073':1.5,  # 较严
    '253':0.2,   # 极松
    '1146':1, '916':1,
    '620':0.5, '583':0.6, '548':1, '384':1,
    '139':0.5, '1125':1, '1077':0.5, '818':1, '635':0.5
}
```

范围：0.2 ~ 1.5。**这是逐 subject 调出来的 pack-stage core-channel 阈值**。

**新 pipeline**：在 `scripts/run_epilepsiae_lagpat_backfill.py` 中所有 subject 一律用 `mean + 1*std`。`config/subject_params.json` 的 `epilepsiae` 块**没有 `pick_k` 或 `pack_pick_k` 字段**——只有 `drop_channels`。

**影响**：例如 subject 253 用 0.2 vs 1.0 会选完全不同的 channel set（虽然单这一项不能解释左右翻转，见 Δ5）。

### Δ3. 每 subject 调优的 `packWinLen` 缺失

**Legacy** (`epilepsiae_packGroupEvents_supressAllSyn_withFreqCenter.py:462-463`)：

```python
sub_packWL_list = {
    '1096':180e-3, '1084':180e-3, '958':250e-3, '922':200e-3,
    '590':500e-3, '1150':300e-3, '442':300e-3, '1073':110e-3,
    '253':300e-3, '1146':250e-3, '916':150e-3, '620':220e-3,
    '583':220e-3, '548':300e-3, '384':400e-3, '139':200e-3,
    '1125':150e-3, '1077':200e-3, '818':250e-3, '635':200e-3
}
```

范围：110 ~ 500ms（4.5x 跨度）。**这是逐 subject 调出来的 pack-stage 窗长**。

**新 pipeline** (`scripts/run_epilepsiae_lagpat_backfill.py:203`)：`window_sec=0.5` 固定值。`config/subject_params.json` 的 `epilepsiae` 块**没有 `pack_win_sec` 字段**。

**影响**：1073 用 110ms vs 新固定 500ms → packing window 大 4.5 倍 → packed times 数量、形状全部不同。

### Δ4. synRefine 阶段 `refine_window_sec` 错值

**Legacy** (`epilepsiae_synRefine_supressAllSyn.py:32`)：`packWinLen=400e-3`（synRefine 阶段固定 400ms）

**新 pipeline** (`src/group_event_analysis.py:912` `legacy_refine_counts_from_detection_sets`)：`refine_window_sec: float = 0.3`（默认 300ms）

**影响**：synRefine 阶段的 pack window 比 legacy 短 100ms → 不同的 packed_times → 不同的 reHist count → `_refineGpu.npz` events_count 不同。

### Δ5. 检验：detector 层是否真的相同？(待二次验证)

**Detector 层定量验证**（subject 253，single record 25300102_0000）：

```
parameter           legacy   new
rel_thresh           2.0     2.0     ✓
abs_thresh           2.0     2.0     ✓
side_thresh          2.0     2.0     ✓
min_gap_ms           20      20      ✓
min_last_ms          50      50      ✓
max_last_ms          200     200     ✓
band                [80,250][80,250] ✓
notch                50/100/150/200/250  ✓
reference            CAR     CAR     ✓
multi_band_envelope  yes     yes     ✓ (sub_band 20Hz)
drop_channels (253)  ['HRC5'] ['HRC5'] ✓
```

per-record events_count ratios all ∈ [0.9, 1.07] for top-15 channels；total events 4127 vs 3942 (95%)。**Detector 已 OK，问题不在这里**。

但是 5 个 outlier subjects 中，**253 的左右翻转无法仅由 Δ1-Δ4 解释**：在 detector 层 (legacy gpu.npz)，HLC1 仍然是 #1（1338），HRB1/HRB2 才 559/404。但 legacy `sub_refineGpu.npz` 里 HRB2 跳到 #1（97622）、HLC1 降到 #4（43648）。

→ 说明 **legacy 在 sum-then-pack-then-rehist 流程中，HRB2 类右侧通道在很多 records 上落入了 packed_window 但 HLC1 没落入**。这只能由 packed_times 的形状解释。

**假设**：subject 253 的 packing 极松（pickChn_thresh=0.2 → 9 chns 进 pack），导致 packed_windows 数量爆炸；HRB2 在很多 records 上被反复 hist；而 HLC1 因为本身是单通道高密度 detector，每个 detection 是独立尖峰，**反而较少在多通道同时活动 → 较少落入 packed_window**。

如果这个假设成立，**Δ2 + Δ3（subject 253 用 pickChn_thresh=0.2 + packWinLen=300ms）就能解释 sub_refineGpu.npz 翻转**。但需要实现 Δ1-Δ4 后用 sub 253 实测重跑验证，不能光说。

## 5 处合同违反对 Stage C outliers 的归因

| Subject | bucket | 现象 | 主要可疑根因 |
|---------|--------|------|------|
| 253 | large_drift | jaccard 0.091, 左右翻转 | **Δ2+Δ3+Δ4+Δ1** 联合 — pickChn_thresh=0.2 极松 + packWinLen=300ms + synRefine pack 400ms + 同步噪声未拒绝。需重跑验证。 |
| 1150 | large_drift | count_ratio 6.6 | **Δ1+Δ3** — 同步噪声未拒绝产生大量虚假 packed → count 膨胀；packWinLen=300ms 缺失. |
| 1073 | large_drift | count_ratio 0.498 | **Δ3** — packWinLen=110ms（最短的 subject）vs 新 500ms → 老 pack 极窄 → 老 count 反而少 → 新/老比偏低. |
| 139 | large_drift | jaccard 0.571 | **Δ2** — pickChn_thresh=0.5 vs 新 1.0；可能 + Δ3. |
| 1146 | moderate_drift | count_ratio 0.689 | **Δ3** — packWinLen=250ms vs 新 500ms. |
| 1096 | moderate_drift | count_ratio 0.611 | **Δ3** — packWinLen=180ms vs 新 500ms. |
| 384 | moderate_drift | count_ratio 0.628 | **Δ3** — packWinLen=400ms vs 新 500ms. |
| 635 | moderate_drift | count_ratio 1.395 | **Δ2** — pickChn_thresh=0.5 vs 新 1.0；count 因松松松而变多. |

**12 stable subjects** 大部分 `pickChn_thresh=1` 且 `packWinLen` 接近 500ms，所以 Δ1-Δ4 影响相对小，bucket 进 stable。但 **不代表无影响**：`refine_packedEvents_byAllBool` 缺失会让所有 stable subjects 的 packed_times 含有同步噪声 windows（可能在 ~3-10% 量级），即便 chnNames 对得上，lagPat 内容也会被污染。

## 修复路径（建议）

### Step 1：补 `refine_packedEvents_byAllBool` (Δ1)

最关键，影响所有 20 subjects。在 `src/group_event_analysis.py` 加 `_legacy_refine_packedEvents_byAllBool(all_events, packed_times, fs, thresh=0.7)`，在 `legacy_refine_counts_from_detection_sets` 和 `pack_record` 两处都用。新增 TDD test。

### Step 2：补 `synRefine` 阶段 `refine_window_sec` 默认 0.4 (Δ4)

把 `legacy_refine_counts_from_detection_sets` 的 default 从 0.3 改 0.4。新 TDD test。

### Step 3：把 `pickChn_thresh` 和 `packWinLen` 落入 `config/subject_params.json` 的 `epilepsiae` 块 (Δ2 + Δ3)

按 legacy `sub_pickT_list` 和 `sub_packWL_list` 抄一份到 `config/subject_params.json`。`scripts/run_epilepsiae_lagpat_backfill.py` 读取 per-subject value，fallback 到 dataset _defaults。新 TDD test。

### Step 4：整 cohort 重跑 + Stage C 三跑

`FORCE=1 N_JOBS=8 bash scripts/run_epilepsiae_lagpat_backfill_parallel.sh` → `python scripts/audit_epilepsiae_lagpat_backfill.py --all` → 比对：
- chn_overlap_jaccard cohort median: 当前 1.0 → 应该保持
- count_ratio cohort median: 当前 0.915 → 应该保持或更接近 1.0
- 4 个 large_drift subjects (253, 139, 1073, 1150) 是否回到 stable / moderate
- bucket 分布是否变成 ≥14 stable

### Step 5：判定

- 如果 Step 4 后 stable ≥ 14 + large ≤ 2 → decision_for_stage_d = enter_full → Stage D 用 stable subjects 全跑 D.1 + D.2
- 如果仍有 ≥3 large_drift → 那一两个 outlier subject 可能是**真**信号差异（detector 层差 5% 累积出来的，不是合同 bug），单独标注后排除主分析

## 关键守则

- **不要先 commit 修复**：先把 Step 1-3 写成 PR 候选，跑 Step 4 看效果，再决定 commit。
- **不要在 Stage D 之前调任何全局阈值**：合同复刻 ≠ 把 mismatch 调没。如果 Step 4 后 outliers 仍存在，那就是真信号差异，应当报告而非掩盖。
- **detector 层不动**：已验证 95% 一致，没必要动。

## 工程文件清单（待改）

- `src/group_event_analysis.py`：补 `refine_packedEvents_byAllBool`（Δ1）；改 `legacy_refine_counts_from_detection_sets` default 0.3→0.4（Δ4）
- `scripts/run_epilepsiae_lagpat_backfill.py`：从 `subject_params.json` 读 per-subject `pick_k` + `pack_win_sec`（Δ2+Δ3）；在 pack_record 调用 `refine_packedEvents_byAllBool`（Δ1）
- `config/subject_params.json` `epilepsiae` 块：每个 subject 加 `pick_k` + `pack_win_sec` 字段，按 legacy 抄
- `tests/test_epilepsiae_lagpat_backfill.py`：补 4 类 TDD（Δ1-Δ4 各一个）
- `docs/archive/epilepsiae_lagpat/legacy_replication_audit_2026-05-03.md`（本文）

## 数据对照（不再 commit，只做 audit）

- `/mnt/epilepsia_data/inv_1_part/pat_25302/adm_253102/rec_25300102/25300102_0000_gpu.npz`（legacy 原始 raw 路径，108KB intact）
- `/mnt/epilepsia_data/interilca_inter_results/all_data_lns/253/all_recs/sub_refineGpu.npz`（legacy 后处理）
- `results/hfo_detection/253/25300102_0000_gpu.npz`（新 pipeline detector 输出）
- `results/hfo_detection/253/_refineGpu.npz`（新 pipeline 后处理）

# 关键代码定位（legacy）

- `ReplayIED/inter_events/epilepsiae_interictal/epilepsiae_detectHFOs.py` — detector
- `ReplayIED/inter_events/epilepsiae_interictal/epilepsiae_synRefine_supressAllSyn.py` — synRefine 阶段
- `ReplayIED/inter_events/epilepsiae_interictal/epilepsiae_packGroupEvents_supressAllSyn_withFreqCenter.py` — lagPat 生产
- `ReplayIED/inter_events/epilepsiae_interictal/hfo_net.py:559-581` — `refine_packedEvents_byAllBool`
- `ReplayIED/inter_events/epilepsiae_interictal/sub_dropChns.py` — 已迁入新 config
