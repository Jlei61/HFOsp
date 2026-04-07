# 间期群体事件周期性分析

> 状态：**完成** — Phase 4 鲁棒性检验 + Phase 5 伪影定位 均已完成
> 创建日期：2026-04-04
> Phase 4 完成日期：2026-04-04
> Phase 5 完成日期：2026-04-05
> 关联论文图：Figure 3, Figure S7, Figure S13
> 系统审阅：见 `docs/event_periodicity_phase2_review_2026-04-05.md`
> 方法学叙事更新：见 `docs/interictal_population_event_methodological_review.md`

---

## 1. 现象定义（老论文声称）

老论文声称癫痫间期 HFO 群体事件（population events）表现出稳定的准周期性：

1. **PSD 周期峰**：二值脉冲序列 Welch PSD 在 ~2Hz 处有周期性峰值
2. **IEI 幂律分布**：事件间隔在 log-log 空间近似直线（幂律）
3. **跨 subject 一致性**：多个 subject 和数据集均可观察到

**验证结论：以上三项均被推翻。** 详见 §5。

---

## 2. 老代码计算链路

### 2.1 端到端流水线

```
Step 1: *_gpu.npz + *_lagPat.npz
        → plotting_figAdd_qusiPeriod.py
        → {sub}_perChn_events_histNpsd.npz

Step 2: *_gpu.npz + *_lagPat.npz + *_packedTimes.npy
        → plotting_figAdd_qusiPeriod_group.py
        → {sub}_group_histNpsd.npz

Step 3: 两个 histNpsd
        → plotting_figAdd_qusiPSDdecomp.py (FOOOF)
        → {sub}_psd_hist_fitRes.npz

Step 4: fitRes → plotting_figAdd_qusiPeriod_plotting.py       → Fig S7
Step 5: fitRes → plotting_figAdd_qusiPeriod_plotting_piled_stats.py → Fig 3C
```

### 2.2 关键计算细节

**IEI 计算（per-channel）**：
- 从 `*_gpu.npz` 取 `whole_dets`（逐通道事件起止时间对）
- 通道集合由 `lagPat.npz` 的 `chnNames` 决定（pack 通道集）
- 多 record 事件拼到绝对时间轴后排序
- IEI = `np.diff(event_starts)`

**IEI 计算（group）**：
- 从 `*_packedTimes.npy` + `*_lagPat.npz` 做时间校准
- 校准：`lagPatRaw` 的 min/max 映射到 packedTimes 窗口偏移
- IEI = `np.diff(packedTimes_stamp[:, 0])`

**脉冲序列 PSD**：
- 100Hz 二值数组，事件区间=1，其余=0
- `scipy.signal.welch(arr, fs=100, nperseg=50000)`（500s 窗）
- 截取 f <= 10Hz

**FOOOF 分解**：
- PSD: `FOOOF(aperiodic_mode='knee', max_n_peaks=2, min_peak_height=0.1, peak_width_limits=(0.6,12.))`, [0.5,10] Hz
- IEI hist: `FOOOF(aperiodic_mode='fixed', max_n_peaks=0)`, [1, min_zero]

**显著性（老方法，已知问题）**：
- `ttest_1samp(psd_minus_apfit, peak_psd)` — 频率 bin 不独立，检验无效

### 2.3 老代码位置

| 脚本 | Yuquan | Epilepsiae |
|------|--------|------------|
| per-chn histNpsd | `ReplayIED/.../yuquan.../plotting_figAdd_qusiPeriod.py` | `ReplayIED/.../epilepsiae.../plotting_figAdd_qusiPeriod.py` |
| group histNpsd | `...plotting_figAdd_qusiPeriod_group.py` | 同 |
| FOOOF 分解 | `...plotting_figAdd_qusiPSDdecomp.py` | 同 |
| 幂律+R2 图 | `...plotting_figAdd_qusiPeriod_plotting.py` | 同 |
| 堆叠 PSD 图 | `...plotting_figAdd_qusiPeriod_plotting_piled_stats.py` | 同 |

---

## 3. 方法论风险清单

详见 `.cursor/plans/population_event_periodicity_*.plan.md`，共 9 项风险。

Tier 1（必须通过）：
- Gamma renewal process 零假设（不应期伪峰排除）
- per-channel vs group 一致性（packing 伪影排除）
- ISI-shuffle surrogate（替代无效 t 检验）

Tier 2（发表前必须完成）：
- SOZ vs non-SOZ 分层
- power-law vs lognormal 模型选择（MLE + LLR）
- 数据缺口处理

Tier 3（敏感性分析）：
- delta-spike vs rectangle-pulse
- Welch nperseg 扫描
- day vs night 分层

---

## 4. 新代码模块

- `src/event_periodicity.py` — 核心计算
- `scripts/run_event_periodicity.py` — 双数据集驱动
- `tests/test_event_periodicity.py` — 单元测试

---

## 5. 结果

### 5.1 Phase 3 — 复现

PSD 周期峰可在两个数据集中复现：
- Yuquan：11 subjects 有数据（7 无 lagPat），其中 7/11 有 specparam 检测到的周期峰
- Epilepsiae：20 subjects 全有数据，11/20 有周期峰
- 峰频范围 1.37–3.40 Hz，与老论文一致

### 5.2 Phase 4 — 鲁棒性检验（核心结论）

#### 5.2.1 Surrogate 显著性检验（Tier 1 — 决定性）

对 30 subjects 的 group-level PSD 峰做两种 surrogate 检验（每种 200 次）：

| Verdict | N | 含义 |
|---------|---|------|
| **refractory** | **15** | Gamma renewal p≥0.05 — 峰完全由不应期解释 |
| no peak | 11 | specparam 未检测到显著周期峰 |
| dist-artifact | 2 | ISI-shuffle p≥0.05 — 峰由 IEI 分布形态产生 |
| **GENUINE** | **1** | 两项检验均 p<0.05 — 真正的周期性 |

**结论：30 个 subject 中仅 1 个（huanghanwen, n=484 事件）通过双重 surrogate 检验。**
该 subject 事件数极少，结论可靠性存疑。其余有峰的 subject 均可由以下机制解释：

1. **不应期效应**（15/30）：事件之间存在 ~0.3-0.5s 最小间隔（检测窗重叠/
   packing 窗限制），导致事件率点过程的 PSD 在 1/min_IEI 附近必然出现峰值。
   Gamma renewal process（匹配实际发放率和最小 IEI）产生的峰 **远强于**
   真实数据（null peak power ~0.77 vs real ~0.17）。

2. **IEI 分布形态**（2/30）：ISI-shuffle 后峰值保持，说明峰不依赖于
   事件的时序结构，仅由 IEI 的统计分布决定。

#### 5.2.2 IEI 分布模型比较（Tier 2）

用 MLE + 对数似然比检验比较 power-law vs lognormal：

| 指标 | 结果 |
|------|------|
| N(lognormal 显著优于 power-law, p<0.05) | **30/30** |
| N(power-law 优于 lognormal) | **0/30** |
| Alpha 范围 | 1.21–2.84 |
| LLR R 范围 | -103.3 to -3.2（全部 <0） |

**结论：IEI 分布服从 lognormal 而非 power-law。老论文的"幂律"结论不成立。**

#### 5.2.3 敏感性分析（Tier 3）

| 检查项 | 结果 | 结论 |
|--------|------|------|
| delta-spike vs rectangle-pulse | 峰频无变化 | 排除脉冲形状伪影 |
| Welch nperseg 50-1000s | 峰频稳定 1.57-1.58 Hz | 排除参数敏感性 |
| per-channel vs group | 通道峰频中位数 3.54Hz vs group 1.58Hz | 不一致，group 峰可能受 packing 影响 |
| day vs night (548/E14) | 白天 1.81Hz, 夜晚 1.90Hz | 两者均存在，夜间更强 |

### 5.3 Phase 5 — 伪影来源精确定位（Phase 2 plan）

5 组实验在 30 subjects（Yuquan 10 + Epilepsiae 20）上运行。

#### 5.3.1 实验 1：PackWinLen 参数扫描

对 10 个 Yuquan subjects，用 9 种窗口大小 W ∈ [100, 150, 200, 300, 400, 500, 600, 800, 1000] ms 重新 packing，观察 f_peak 变化。

| 发现 | 详情 |
|------|------|
| f_peak ≠ 1/W | 若 f_peak 完全由窗口量化决定，应 f_peak ≈ 1/W。实测完全不符。例：chengshuai W=100ms → f_peak=2.10（理论 10.0）|
| iei_min 跟随 W | 最小 IEI 确实等于 W（确认窗口强制最小间隔），但这不直接创建 PSD 峰 |
| 多数 W 无峰 | 仅 2-5/9 种 W 有 specparam 检测到的峰，峰检测对参数敏感 |
| f_peak ≈ 1.88 反复出现 | 7/10 subjects 在 W=800ms 时 f_peak ≈ 1.88Hz，不依赖 W 选择 |

**结论：packing 窗口大小不是峰频的直接控制因素。**

#### 5.3.2 实验 2：质心旁路

对 30 subjects 用三种事件时间定义计算 PSD：
- (a) Window Start：`packedTimes[:, 0]`
- (b) Mean Centroid：`lagPatRaw` 参与通道均值
- (c) Ignition Centroid：`lagPatRaw` 最早通道

| 指标 | 结果 |
|------|------|
| 3 方法峰频差 < 0.1Hz | 13/30 subjects |
| 解释边界 | 当前比较仍依赖 legacy `lagPatRaw -> 绝对时间` 映射，不是真正独立的 timestamp 重建 |

**降级后的结论：** 对多数检出 peak 的 subject，在当前 legacy 一致的时间映射框架内，
窗口内锚点的改变通常不会显著改变峰频。

这说明峰频**不是简单的窗口起点选择伪影**；但这**不等于**已经完全脱离 packing /
lagPatRaw 生成链路。要彻底回答这个问题，仍需要从原始 envelope / spectrogram
独立重建绝对事件时间。

#### 5.3.3 实验 3：IEI Hazard Function

对 30 subjects 计算 H(t) = f(t) / (1-F(t))。

典型模式：
- t < min_IEI：H(t) ≈ 0（强制死区，对应 packing 窗口/检测不应期）
- t > min_IEI：H(t) 急剧上升后缓慢衰减
- 三种事件定义的 H(t) 形态一致

这是经典的 refractory renewal process 特征。

#### 5.3.4 实验 4：IEI Return Map（关键新发现）

对 30 subjects 计算连续 **log-IEI** 的 serial correlation
`corr(log IEI[n], log IEI[n+1])`。

| 指标 | 结果 |
|------|------|
| 正相关 subjects | **30/30**（100%） |
| Subject-level sign test | **p = 9.31e-10** |
| Mean r | 0.318 (range 0.124–0.511) |

**这是一个重要的新发现：**
- **正** serial correlation 意味着连续间隔趋同（长-长，短-短），指向事件率的慢调制
  （如昼夜节律、睡眠阶段变化、发作间歇状态漂移）
- 若存在振荡器驱动的周期性，应观察到 **负** serial correlation（短-长交替）
- 这进一步排除了内在振荡机制，指向 **非平稳率过程** 的正确解释

**口径注意：**
- 当前代码内部还会输出 Pearson `p` 值，但那只是描述性数字，不能作为正式
  subject-level 推断，因为相邻 IEI 对之间并不独立。
- 正式报告应优先使用“30/30 方向一致 + sign test”这一层级。

#### 5.3.5 实验 5：传播立体型分析

对 30 subjects 计算事件间通道激活顺序的 Kendall tau 一致性。

| 分组 | Mean tau | N subjects |
|------|----------|------------|
| 全部事件 | 0.126 (0.014–0.322) | 30 |
| SOZ 事件 | 0.119 | 25 (有 SOZ 的) |
| non-SOZ 事件 | 0.048 | 12 (有 non-SOZ 的) |
| sign test (SOZ > non-SOZ) | p = 0.073 | 12 pairs |
| one-sided Wilcoxon | p = 0.039 | 12 pairs |

- 高 tau 个体：548/E14 (0.322), 922 (0.309), 818 (0.298) — 传播路径相对固定
- 低 tau 个体：635 (0.014), 958 (0.032), 620 (0.035) — 近乎随机传播
- SOZ 事件的传播比 non-SOZ 更有规律，但当前仍属于**探索性证据**

### 5.4 最终科学结论

**间期群体事件的 ~2Hz PSD 周期峰是不应期/检测窗限制的数学伪影，
不是由网络内在振荡器驱动的真正节律。**

Phase 4 证据链：
1. Gamma renewal null (匹配率+不应期) 产生更强的峰 → 不应期效应足以完全解释
2. ISI-shuffle 后峰保持 → 峰不依赖于事件时序结构
3. IEI 分布为 lognormal 而非 power-law → "无标度"结论不成立
4. Per-channel 和 group 峰频不一致 → packing 引入额外伪影

Phase 5 证据链（伪影源定位）：
5. f_peak ≠ 1/W（PackWinLen 扫描）→ 窗口大小不是峰频的直接来源
6. 质心旁路对峰频无影响 → 窗口栅格量化不是来源
7. Hazard function 显示经典不应期模式 → 支持 renewal process 解释
8. **IEI serial correlation 全部为正** → 排除振荡器，指向率慢调制
9. 传播立体型：SOZ 事件路径更固定，但目前是探索性空间规律而非定论

**建议**：
- 在最终论文中不宜将 PSD 周期峰作为正面发现报告
- IEI 正 serial correlation（率慢调制）是一个值得讨论的方法学发现，正式统计口径应使用 subject-level direction consistency
- 传播立体型（SOZ vs non-SOZ）可作为空间组织规律的候选发现，
  但需更大样本验证（当前仅有探索性单侧证据）

---

## 6. 代码清单

### Phase 1 (复现 + 鲁棒性)
- `src/event_periodicity.py` — 核心计算（PSD、specparam、IEI fit、surrogate）
- `scripts/run_event_periodicity.py` — 双数据集驱动
- `scripts/run_surrogates_batch.py` — surrogate 批量运行
- `scripts/plot_event_periodicity.py` — Phase 1 可视化

### Phase 2 (伪影定位 + 正确工具)
- `src/event_periodicity.py` — 新增：
  - `load_raw_detections_yuquan_per_block()` — 逐块加载 raw detections
  - `repack_and_analyze()` — 给定 W 重新 packing + PSD
  - `run_packing_sweep()` — 实验 1 驱动
  - `load_centroid_event_times()` — 质心旁路
  - `run_centroid_bypass()` — 实验 2 驱动
  - `compute_hazard_function()` — Hazard function
  - `compute_iei_return_map()` — Return map / Poincaré
  - `compute_propagation_stereotypy()` — Kendall tau 一致性
- `scripts/run_periodicity_phase2.py` — Phase 2 批量驱动
- `scripts/plot_periodicity_phase2.py` — Phase 2 可视化

### 结果文件
- `results/event_periodicity/phase2/exp1_packing_sweep.json`
- `results/event_periodicity/phase2/exp2_centroid_bypass.json`
- `results/event_periodicity/phase2/exp3_hazard.json`
- `results/event_periodicity/phase2/exp4_return_map.json`
- `results/event_periodicity/phase2/exp5_stereotypy.json`
- `results/event_periodicity/phase2/figures/` — 可视化图
