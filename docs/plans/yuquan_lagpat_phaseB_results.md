# Phase B — End-to-End Drift Validation 结果

> Date: 2026-04-17
> Reference set: `gaolan` (12 blocks), `dongyiming` (9 common blocks), `wangyiyang` (9 common blocks)
> 总共比较: 30 blocks
> 验证脚本: `scripts/validate_drift_new_detector.py`
> 算法路径: 与 Phase A 完全一致（legacy-clone packing + stitched-spectrogram centroid，centroid_power=3.0）
> 唯一变量: `_gpu.npz` + `_refineGpu.npz` 从 `/mnt/yuquan_data/yuquan_24h_edf/<sub>/` 换到 `results/hfo_detection/<sub>/`
> 数据落盘: `results/validation/phaseB/SUMMARY.md` + `<sub>__SUBJECT.json` + `<sub>__<block>.json`
> 调研笔记: [`yuquan_phaseB_gaolan_picked_drift_notes.md`](./yuquan_phaseB_gaolan_picked_drift_notes.md)

---

## TL;DR

- Phase B **触发了 plan 的停止条件**：`|median n_participating shift| > 20%` 在 gaolan 和 wangyiyang 上都触发，命中 2/3 阈值。
- 但 3 个 subject 的 drift 结构**完全不同质**：
  - **gaolan**: 严重漂移。picked 通道大变（Jaccard 0.31），L3 −28.6%，L4 −29.7%，两个 flag 都中。
  - **wangyiyang**: 边界漂移。picked 通道高重叠（Jaccard 0.88），L3 +21.4%（刚过线），L4 +2.2%（健康）。
  - **dongyiming**: 无漂移。picked 高重叠（Jaccard 0.83），L3/L4 都在阈值内一半以下。
- 漂移的根因不是 packing/centroid 算法，是 **新 detector 的 events_count 总数和方差结构与老 detector 不同**，被 `mean + pick_k * std` 阈值法二次放大成 picked 集差异。详见 `yuquan_phaseB_gaolan_picked_drift_notes.md`。
- 结论：**不允许把 41-subject 当 legacy-equivalent cohort 直接替换 30-subject 主结论**；可以推进 11-subject 的 backfill 写盘，但产物只允许进入 extended-cohort sensitivity 通道。

---

## Cohort-level 数值（30 blocks 聚合）

| 指标 | gaolan | dongyiming | wangyiyang | 阈值 |
|---|---|---|---|---|
| L1 picked Jaccard | **0.312** | 0.826 | 0.880 | (无硬阈值) |
| L1 n_new vs n_legacy | 9 vs 12 | 43 vs 41 | 25 vs 22 | — |
| L1 only_new | `["A'10", 'A9', "B'13", "B'14"]` | `['G10','G13','H2','H8','J7']` | `['A6','D4','E5']` | — |
| L1 only_legacy | `['C4','C5',"D'2","D'3","D'4",'D1','D2']` | `["B'4",'C3','C4']` | `[]` | — |
| L1 alias collisions in picked | 0 | 0 | 0 | 0 |
| L2 sum n_new / sum n_legacy | 7593 / 7451 | 3574 / 2988 | 1661 / 1919 | — |
| **L2 cohort ratio** | 1.019 | 1.196 | 0.866 | [0.67, 1.50] |
| **L3 median n_participating shift** | **−28.6%** | +7.7% | **+21.4%** | ≤ 20% |
| L3 median (new / legacy) | 5.00 / 7.00 | 28.00 / 26.00 | 17.00 / 14.00 | — |
| **L4 median lag span shift** | **−29.7%** | −2.7% | +2.2% | ≤ 20% |
| L4 median lag span (ms, new / legacy) | 43.22 / 61.51 | 77.92 / 80.08 | 74.94 / 73.31 | — |
| L4 p95 lag span (ms, new / legacy) | 140.57 / 133.39 | 131.27 / 134.35 | 127.54 / 126.42 | — |
| **drift_flags** | L3, L4 | (none) | L3 | — |

## Plan stop-condition 命中情况

| 条件 | 触发 subject | 是否 ≥ 2/3 |
|---|---|---|
| L2 cohort ratio outside `[0.67, 1.50]` | (none) | 0/3 |
| L3 \|median n_participating shift\| > 20% | gaolan, wangyiyang | **2/3 ✓** |
| L4 \|median lag span shift\| > 20% | gaolan | 1/3 |
| alias collision in picked | (none) | 0/3 |

Plan §544: "以下任一项在 ≥ 2/3 参考 subject 上发生，即判定为'漂移过大'" → **触发**。

Plan §530-540: 漂移过大时的处置：
- 允许生成资产进入 staging；
- 不允许直接把 11 个 subject 并入正式 Topic 1/2 cohort summary。

---

## 三个 subject 的差异为什么这么大

不是采样噪声，是**结构性 detector 灵敏度差异**：

| 指标 | gaolan 老 → 新 | dongyiming 老 → 新 | wangyiyang 老 → 新 |
|---|---|---|---|
| events_count 总数 | 25,170 → 75,990 (3.0x) | 较平稳 | 较平稳 |
| events_count std | 338 → 1040 (3.1x) | mean 1033, std 899 | mean 160, std 207 |
| pick_k 阈值 | 852 → 2560 (3.0x) | 1480 | 367 |

`gaolan` 在新 detector 上的 events_count **极度非均匀放大**（B'13/B'14/A'10 涨 6x，C 系列涨 3x，D / D' 系列只涨 1.3x）。被 `mean + 1.9 * std` 一过滤，D / D' 全部跌出 picked，B'13/B'14 这种之前在 picked 边缘的高频通道挤进来。

`dongyiming` 和 `wangyiyang` 没有这种结构性放大，picked 集只有边角变化（边界附近 ±5 通道），所以 L1 / L3 / L4 都比较温和。

详细数据见 [`yuquan_phaseB_gaolan_picked_drift_notes.md`](./yuquan_phaseB_gaolan_picked_drift_notes.md)。

---

## 解读

### 哪些是真"漂移"，哪些是测量噪声

`gaolan` 是真漂移：
- 12 个 block 里 8 个 "busy" block 上 L3 都稳定在 −28% 到 −38%，L4 稳定在 −22% 到 −38%。
- 这不是个别 block 的偏差，是整 24h 都有的系统性偏移。
- 物理意义：用新 detector 的 picked 集（B' 高频通道 + A 系列深点）跑出来的 group event 平均参与通道数从 7-8 跌到 5，传播 lag span 从 60+ms 缩到 40+ms。这意味着新 picked 集捕到的 events **空间更集中、传播更短**，跟老 picked 集（D 系列 + 浅 C 触点）捕到的 events 不是同一类东西。

`wangyiyang` 是边界情况：
- L3 +21.4% 刚过线，但 9 个 block 里多个 block 是 +21.4% 同一个值（说明是中位数刚好提了 1）。
- L4 +2.2%，几乎不变。
- picked Jaccard 0.88，只多 3 个通道。
- 物理意义：新 picked 多包含了 3 个边角通道（A6, D4, E5），导致每事件 median n_participating +1。但 lag span / 通道空间结构基本不变。这种漂移**不会改变 Topic 1/2 的核心结论**。

`dongyiming` 是干净：
- 所有 9 个 block L3 在 +3% 到 +8%，L4 在 −5% 到 +2%。
- L2 ratio 1.0 到 1.35，绝大多数 block 1.10-1.25。
- picked Jaccard 0.83，对称地多 5 个少 3 个。

### 是否可以接受

按 plan，drift 过大触发后**仍然允许 backfill 写盘**，但限制 41-subject 的科学使用方式。这件事不能在脚本层解决，只能在报告层声明：

- 41-subject 结果**只能作为 extended-cohort sensitivity**，不替换 30-subject 主结论。
- 任何"trend with N subjects"的图都必须同时给 30 / 41 两条线，并标明 11 个新增 subject 走的是 detector source 与老 cohort 不同的链路。
- 任何 subject-level 个例展示，gaolan 必须特别标注"new detector picked set ≠ legacy"。

### 是否要回头修 detector

不在本次范围。gaolan 的 picked drift 提示**新 detector 在 ripple 频段（80-250Hz）的灵敏度结构与老 detector 不同**——可能是 detection threshold、preprocessing、resample factor 中的某一个变了。但这是另一个 PR：

> Plan §175: 如果要改 contract，那是另一个 PR，必须全 cohort 重算，不准夹带私货。

如果以后要让 41-subject 替换 30-subject 主结论，就必须：
1. 找到 detector source drift 的根因
2. 重跑全 41 subject 在同一 detector 下
3. 而不是回填 11 个，让 30 个用旧 detector，11 个用新 detector

---

## 与 Phase A 一起看

| 阶段 | 目标 | 结果 |
|---|---|---|
| Phase A 算法同一性 | 同一 legacy 输入 → 同一输出 | A1-A4 全过；A5 严格 rank 失败但 sub-ms 量级 |
| Phase B 端到端漂移 | 同一新 detector 输入 → 与老 lagPat 比 | 触发停止条件，gaolan 漂移大，dongyiming/wangyiyang 边界 |

**Phase A** 证明了"算法本身没改"；**Phase B** 证明了"换 detector 之后输出确实变了，并且不均匀"。两件事不矛盾，两个都是真信号。

合在一起的科学含义：
- 11-subject backfill 用的是新 detector + legacy-clone packing/centroid 路径，这是一条**新的 end-to-end 链路**，不是老链路的延伸。
- 这条新链路在不同 subject 上稳定性差异很大（gaolan 严重，dongyiming 干净）。

---

## 推荐处置

按 plan 的"漂移过大处置 + 推荐执行顺序"，下一步：

1. **可以推进的事**
   - 可以正式批量回填 11 个 backfill subject 写盘（按 plan §555 的 5 项前置条件继续走）。
     - 现状：Phase A ✓ + Phase B 触发 stop condition (允许 staging)。
     - 还缺：`pack_win_sec` 进 `config/subject_params.json`、alias QC 实现、原子写入实现。
   - QC 报告强制必填：每个 subject 的 picked Jaccard vs 老 cohort（如有）、events_count 总数倍率、L3/L4 与同 cohort 中位数的 z-score。
2. **不允许做的事**
   - 不允许直接拿 41-subject 替换 30-subject 主结论。
   - 不允许在 Topic 1/2 报告里把"backfill subject"和"原 cohort subject"混着画在同一条 trend 上而不区分。
3. **需要决策的事**
   - 是否把 gaolan **从 backfill cohort 中剔除**？理由：它的 picked drift 远超其它 subject，包含进 cohort 后会成为离群点干扰统计。
   - 是否在 Topic 1/2 之前做 L5/L6（Topic 1/2 single-subject summary drift）？plan 列了这两项是"必做"的，但成本不小，需要先有 single-subject summary 入口。
   - 是否回过头看新 detector 与老 detector 的差异源头？这件事独立于本次 backfill，但会影响后续是否要重跑全 41-subject。

下一轮如果要继续，建议优先级：
1. **（高优）剔除 gaolan 的决策**：单 subject 的 drift 不应污染整个 backfill cohort 的统计。
2. **（高优）补 plan §555 的前置条件**：subject_params.json 加 pack_win_sec、alias QC、atomic write。
3. **（中优）L5/L6 端到端**：要做但成本大，可以等 backfill 写完之后用新 lagPat 直接跑 Topic 1/2 single-subject summary。
4. **（低优 / 另一个 PR）detector source 调研**：不属于本次 backfill 范围，但需要单独立项。

---

## 附：每 subject 每 block 数值

完整 per-block 数据见 `results/validation/phaseB/<subject>__<block>.json`；下表给出每 subject 全 block 的 L2/L3/L4 趋势（busy block 才统计）。

### gaolan (12 blocks，busy block 8 个)

| block | n_new / n_leg | L3 shift | L4 shift |
|---|---|---|---|
| FA0013KP | 1138/1224 | -37.5% | -34.9% |
| FA0013KQ | 1068/1114 | -28.6% | -38.3% |
| FA0013KR | 1107/1119 | -28.6% | -35.2% |
| FA0013KS | 895/816 | -28.6% | -22.1% |
| FA0013KT | 667/550 | -28.6% | -21.5% |
| FA0013KW | 722/722 | -28.6% | -24.0% |
| FA0013L1 | 1589/1682 | -37.5% | -34.1% |

L3 在 8 个 busy block 上**全部稳定 ≤ −20%**。L4 在 7/8 个 busy block 上 ≤ −20%。

### dongyiming (9 blocks)

| block | n_new / n_leg | L3 shift | L4 shift |
|---|---|---|---|
| FA134D2R | 593/438 | +5.9% | -2.9% |
| FA134D2S | 627/537 | +7.7% | -1.6% |
| FA134D2T | 253/219 | +7.7% | -5.0% |
| FA134D2Y | 119/96 | +7.7% | +0.2% |
| FA134D2Z | 744/672 | +3.7% | -2.5% |
| FA134D33 | 658/550 | +7.4% | -5.1% |
| FA134D34 | 572/471 | +3.7% | -3.3% |

所有 block L3/L4 都在 ±10% 内。

### wangyiyang (9 blocks)

| block | n_new / n_leg | L3 shift | L4 shift |
|---|---|---|---|
| FA0012P5 | 113/130 | +21.4% | -3.9% |
| FA0012P6 | 250/291 | +23.1% | +2.9% |
| FA0012P7 | 437/533 | +21.4% | +3.4% |
| FA0012P8 | 331/377 | +21.4% | +1.2% |
| FA0012P9 | 246/271 | +21.4% | +2.1% |
| FA0012PC | 80/82 | +9.4% | +0.0% |
| FA0012PD | 203/231 | +13.3% | +2.1% |

L3 在 5/9 个 block 上 ≥ +21%（中位数刚好被新 picked 多一个通道顶上去）；L4 一律 ≤ +5%。
