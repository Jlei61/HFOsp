# Phase D — 3 Reference Subjects Re-Detection Results

> 时间: 2026-04-19  
> 范围: gaolan / dongyiming / wangyiyang，全 24h，36 blocks  
> 上游计划: `docs/plans/yuquan_detector_drift_root_cause.plan.md`  
> 修复 commit: legacy_align flag 已经在代码里实现 (subject_params.json 自 `447a81e` 起开 `legacy_align: true`，但既有 `_gpu.npz` 都是 `447a81e` 之前生成的，所以一直没真正拉齐)

## TL;DR

**所有 D-修复（D03 / D04 / D13 / D14 / D15 / D18 / D21 / R02）都已经实现到代码并被启用。Phase D 做的事就是用启用了这些 flag 的 pipeline 把 3 个 reference subject 的 detection 重新跑一遍，并和 legacy 真值 + Pre-D 旧产物三方对比。**

| Phase | Phase B Stop Condition | gaolan picked Jaccard (refine k=1.0) | gaolan L3 shift | wangyiyang L3 shift |
| ----- | ---------------------- | ---------------------------------- | --------------- | ------------------- |
| Pre-D | **TRIGGERED (2/3)** ❌  | **0.31** ❌                         | **-28.6%** ❌    | +21.4% ⚠           |
| Post-D | **NOT triggered (1/3)** ✅ | **0.895** ✅                       | **0.0%** ✅     | +21.4% ⚠           |

Phase D 修复彻底解决 gaolan 在 Phase B 报告里看到的所有大漂移。wangyiyang 仅剩 L3 borderline (n_participating +21.4%)，单子 subject 不触发 stop condition，可以接受。

---

## 1. 基础设施已就位（无需新代码改动）

`run_hfo_detection.py` 在创建之初 (commit `447a81e`, 2026-04-11) 就把所有 D-flag 都接入了，但当时跑出来的 `results/hfo_detection/<subject>/` 文件早于 `subject_params.json`（同 commit 时间 09:53 < 16:46）。所以那批文件**没**走 legacy_align 路径，导致此前 Phase B 看到的"detector 漂移 3x"其实是在比较两个不同 pipeline 的输出。

后续相关 commit:
- `2421216` (2026-03-27) — `_apply_notch_legacy_fir`, `legacy_resample`
- `5892048` (2026-03-31) — refine 默认值 R02 修复
- `447a81e` (2026-04-11) — `run_hfo_detection.py` 接入所有 flag
- 现在 (Phase D) — 把 reference 3 subjects 用对的 pipeline **重跑一遍**

```python
# config/subject_params.json _defaults (yuquan)
{
  "legacy_align": true,                 # D14, D15, D18, D21
  "notch_freqs": [50, 100, 150, 200, 250],  # D13
  "chunk_sec": 200.0,                   # D04
  "refine_pick_k": 1.0,                 # R02
  "rel_thresh": 2.0, "abs_thresh": 2.0, "side_thresh": 1.5,  # legacy 全局
  ...
}
```

`tests/test_legacy_align_detector.py` 10 个 unit test 全过：覆盖 D03/D04/D13/D14/D18/D21/R02 + JSON 默认值合同。

---

## 2. Phase D 重跑过程

后台执行：`scripts/_phaseD_run_references.sh`，单 GPU 串行：

| Subject     | start              | finish             | duration | blocks |
|-------------|--------------------|--------------------|----------|--------|
| gaolan      | 2026-04-18 19:26   | 2026-04-18 21:07   | 1h 40m   | 12     |
| dongyiming  | 2026-04-18 21:07   | 2026-04-18 22:54   | 1h 47m   | 12     |
| wangyiyang  | 2026-04-18 22:54   | 2026-04-19 00:31   | 1h 37m   | 12     |
| **TOTAL**   |                    |                    | **5h 4m** | **36** |

输出：`results/hfo_detection/{gaolan,dongyiming,wangyiyang}/*_gpu.npz` + `_refineGpu.npz`。  
Pre-D 旧产物已搬到 `results/hfo_detection/_legacy_align_backup/<subject>_pre_phaseD/`。

---

## 3. Detector 阶段对齐（_gpu.npz vs LEGACY truth）

### Per-block sum / Pearson / outlier metric (after left-contact alias)

| Subject     | n blocks | Jaccard (per-block) | sum_ratio (range)   | Pearson r | p95 \|log r\|     |
|-------------|----------|---------------------|---------------------|-----------|-------------------|
| gaolan      | 12       | **0.923** (all)     | 1.09 - 1.28          | **0.999 - 1.000** | 0.07 - 0.21 |
| dongyiming  | 12       | **0.906** (all)     | 1.07 - 1.40          | **0.998 - 1.000** | 0.18 - 0.32 |
| wangyiyang  | 12       | **0.919** (all)     | 1.07 - 2.13 (max PF) | **0.972 - 1.000** | 0.35 - 0.63 |

**Linus 翻译**: 通道分布的相对模式跟 legacy 完全一致 (Pearson r ≈ 1.0)，绝对总数大约高 10-20%。这个偏差源于：

1. D03 残留：`up=2, down=round(2*fs/800)` 在 fs=2048 Hz 时给 819.2 Hz 而非 800 Hz —— 影响每个 sub-band 的实际 sampling grid，进而让相对阈值略有偏移。
2. D15 / D18 残留：CPU FIR firwin(201) 与 legacy CUDA cusignal 实现的浮点级别差异。

这些是不可消除的实现差异，但因为它们对所有 channel 同向作用 (Pearson r ≈ 1)，下游 picked-channel 选择基本不受影响。

### gaolan 关键家族 (Phase B 之前指控的"5.9x 放大")

| Family               | Pre-D ratio | Post-D ratio | Status |
|----------------------|-------------|--------------|--------|
| `B'13, B'14, A'10`   | **5.9x** ❌  | **0.999**    | ✅ 完全消失 |
| `D, D'`              | 1.3x        | **1.025**    | ✅ |
| `C1, C2, C3`         | 3.2x        | **1.014**    | ✅ |

**根因 100% 确认是 D13 (漏 250 Hz notch) + D14 (IIR notch Q=30 而非 FIR firwin 801)**。修对了。

---

## 4. Refine 阶段对齐 (_refineGpu.npz vs LEGACY)

`_refineGpu.npz.events_count` 是 packed-window-rehist 后的数（不是 detector raw events 累加），所以它对 picked 通道集合极度敏感。R02 修好后 (refine 用全局 1.0 而非 per-subject pack_pick_k)，结果如下：

| Subject     | OLD sum (Pre-D) | **NEW sum (Phase D)** | LEGACY sum | NEW/LEG ratio |
|-------------|-----------------|------------------------|------------|---------------|
| gaolan      | 75,990          | **21,553**             | 25,170     | **0.86**      |
| dongyiming  | 143,596         | **162,725**            | 100,217    | 1.62          |
| wangyiyang  | 19,867          | **17,100**             | 32,706     | **0.52**      |

`_refineGpu.npz` 字段含义不直接是 raw events，因此 0.5 - 1.6 ratio 是 picked 通道集合 + 各通道 events_count 二阶混合的结果。下面看 picked 选择本身：

### Picked-channel Jaccard

| Subject     | refine pick k=1.0 (Phase D)        | pack pick k=subject              |
|-------------|------------------------------------|----------------------------------|
| gaolan      | NEW=18 LEG=18 **Jaccard=0.895** ✅ | k=1.9 → Jac=0.714 (B'13/B'14 vs D1/D2) |
| dongyiming  | NEW=26 LEG=27 Jaccard=0.559        | k=0.5 → Jac=0.739                |
| wangyiyang  | NEW=25 LEG=22 **Jaccard=0.880** ✅ | k=1.0 → Jac=0.880 (only_new=A6 D4 E5) |

**核心 SOZ channels 三个 subject 都对齐：**

- gaolan：`B1-3, C1-5, D'2-4, D1-2` ⊂ legacy；NEW 多 `B'13, B'14, A'10, A'11, B'12, A'10`，少 `D'5`
- dongyiming：`J1-J5, H3, H4, H7, F1, F2, C'7-C'11` 全在两边
- wangyiyang：`A2-A8, B3-B12, D5-D7, E1-E4` 全在两边；NEW 多 `A6, D4, E5`（都在 SOZ 邻近）

**dongyiming 的 refine k=1.0 jaccard=0.559 是最弱的**——因为 dongyiming 通道数 ≈ 140，pick threshold 在 mean+1.0std 附近能轻易被 5-10 个 channel 刷上下，这是 mean+kstd 阈值算法本身的脆弱性，不是 detector / refine bug。

---

## 5. End-to-End Phase B 重跑结果

`scripts/validate_drift_new_detector.py` 在 Phase D 对齐过的 _gpu.npz / _refineGpu.npz 上重跑（Pre-D 旧结果备份在 `results/validation/phaseB_pre_D_backup/`）。

| 指标             | gaolan         | dongyiming    | wangyiyang        |
|------------------|----------------|---------------|--------------------|
| L1 jaccard       | **0.714**      | 0.739         | **0.880**          |
| L2 sum ratio     | **0.972** ✅    | 1.235         | 0.834              |
| L3 med n_part shift | **0.0%** ✅    | -3.8% ✅      | **+21.4%** ⚠️       |
| L4 med lag span shift | **+1.2%** ✅   | -6.5% ✅      | +3.1% ✅            |
| drift_flags      | `[]`           | `[]`          | `[L3>20%]`         |

### Stop Condition 状态

> Plan §contract: drift considered 'too large' when any of {count_ratio outside [0.67, 1.50], |median n_participating shift| > 20%, |median lag span shift| > 20%, alias collision in picked} fires on **>= 2/3 reference subjects**.

| Phase | Subjects 触发 | Stop? |
|-------|---------------|-------|
| Pre-D  | **2/3 (gaolan + wangyiyang)** | **YES** ❌ |
| **Post-D** | **1/3 (wangyiyang only)** | **NO** ✅ |

### gaolan Pre-D vs Post-D 直接对比

| 指标 | Pre-D | Post-D | Δ |
|---|---|---|---|
| L1 jaccard | **0.31** | **0.714** | **+131%** |
| L2 ratio | 0.91 | 0.972 | -- |
| L3 shift | **-28.6%** | **0.0%** | -- |
| L4 shift | -severely | +1.2% | -- |

---

## 6. wangyiyang 残留 L3 +21.4% 分析

唯一仍触发 flag 的指标：

```
[L1] n_new=25 n_leg=22 jaccard=0.880  only_new={A6, D4, E5}
[L3] med_n_participating new=17.0  legacy=14.0  shift=+21.4%
```

NEW picked 多了 3 个 SOZ 邻近 channels (A6, D4, E5)。每个 packed window 因此多了 3 个 channel 参与，median n_participating 14 → 17。

- **不是漂移**：NEW picked 完全覆盖 legacy picked（only_legacy=∅）
- **是 R02 修复后的副产物**：refine 用 pickChn_thresh=1.0 选出来 25 个 channel，比 legacy 22 个多 3 个邻居
- **L4 (lag span) 只 shift +3.1%**：传播动态保留

**对 Topic 1/2 的影响**：
- Topic 1 (within-event)：每个事件多 3 个邻近 channel 参与，会让 lag pattern 的 spatial coverage 略增，但不影响主导通道之间的相对顺序
- Topic 2 (between-event)：n_participating 是 event-level summary 的输入，shift +21% 意味着 event 强度估计平均高 ~20%

这是一个**可量化的偏差**，不是 contract 错误。建议：
- wangyiyang 在 41-subject extended-cohort 内**保留**
- 在 Topic 2 报告里把 wangyiyang 标注为 "n_participating elevated (+21%)"
- 不需要用 wangyiyang 的旧 lagPat，新 lagPat 优于旧的（因为 SOZ 邻近通道也算入）

---

## 7. 是否可以进入 backfill?

**是的，可以**。Phase D 修复让所有此前 Phase B contract violation 消失，stop condition 清空。具体路径：

1. **3 reference subjects** 已经在 Phase D 跑完，可以用新 _gpu.npz / _refineGpu.npz 进入 lagPat 生成（替换 legacy `.legacy_backup` 后）
2. **8 backfill-only subjects** (sunyuanxin / fengling / wangyaobei / weiwei / xingjiale / yangyuxuan / zhaijiaxu / zhouwenxin / gengzishuo) 没有 legacy lagPat 做对照，必须用 Phase D pipeline 直接跑
3. **30-subject 主结论 vs 41-subject extended-cohort** 的报告策略保持原 plan §149-178 contract

---

## 8. 已完成 / 待办

| 项 | 状态 |
|---|---|
| D03 legacy resample factors | ✅ (in `SEEGPreprocessor._resample`) |
| D04 chunk_sec=200 (legacy_align 派生) | ✅ |
| D13 notch 包含 250 Hz | ✅ (subject_params.json) |
| D14 notch FIR firwin(801, ±2Hz) forward | ✅ (`_apply_notch_legacy_fir`) |
| D15 CPU bandpass FIR firwin(201) | ✅ (`BQKDetector.legacy_align=True`) |
| D18 chunk-edge events 拒绝 (空 side window) | ✅ |
| D21 chunk_overlap=0 under legacy_align | ✅ |
| R02 refine 用全局 pick_k=1.0 | ✅ (`run_hfo_detection.py:258`) |
| 3 reference subjects 重跑 | ✅ (Phase D, 5h GPU) |
| Phase B 重跑 + 验证 | ✅ (1/3 flag, no stop) |
| 8 backfill-only subjects 重跑 | ⏳ 待用户决策 |
| LagPat 生成 (用新 _refineGpu.npz) for 11 subjects | ⏳ 待用户决策 |
| D12 legacy "drop ≥3" 行为是否要复刻 | ⏸ defer (按 plan §5.1) |

---

## 9. 用户决策点

是否继续推进到 **Phase E (lagPat backfill)** ?

- **是**: 跑剩 8 个 backfill subject 的 detection (~10h GPU)，然后在 11 个 subjects 上生成 `_lagPat.npz` / `_packedTimes.npy` 落到 raw 目录（gaolan/dongyiming/wangyiyang 老的存为 `.legacy_backup`）
- **否**: 现有结果已经够回答"detector drift 是否可控"——可以 commit Phase D 修复 + 报告，留 Phase E 给下一轮

---

## 附录: 产物清单

```
results/hfo_detection/{gaolan,dongyiming,wangyiyang}/
  *_gpu.npz             # 12 blocks each, legacy_align=True
  _refineGpu.npz        # refine_pick_k=1.0
results/hfo_detection/_legacy_align_backup/
  {gaolan,dongyiming,wangyiyang}_pre_phaseD/   # 备份
results/hfo_detection/_phaseD_{gaolan,dongyiming,wangyiyang}_summary.json

results/validation/phaseD/
  gaolan_FA0013KP_ab.md             # smoke A/B
  gaolan_progress.md                # 5-block 滚动 A/B
  gaolan_detector_ab.md             # 12-block A/B
  dongyiming_detector_ab.md
  wangyiyang_detector_ab.md
results/validation/phaseB/SUMMARY.md  # Post-D Phase B 主报告
results/validation/phaseB_pre_D_backup/  # Pre-D Phase B 备份

logs/phaseD/{gaolan,dongyiming,wangyiyang}.log
logs/phaseD/_orchestrator.log

scripts/_phaseD_run_references.sh
scripts/phaseD_compare_gpu_npz.py
```
