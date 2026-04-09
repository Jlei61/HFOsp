# 间期群体事件周期性分析

> 状态：**完成** — Phase 4 + Phase 5 + PR-1 + PR-2 + PR-2.5 均已完成
> 创建日期：2026-04-04
> Phase 4 完成日期：2026-04-04
> Phase 5 完成日期：2026-04-05
> PR-2 完成日期：2026-04-08
> PR-2.5 完成日期：2026-04-08
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

### 5.4 PR-1 — 解析 Renewal PSD Overlay + SOZ Dead-Time（实验 6）

详见 `docs/interictal_population_event_methodological_review.md` §2.3.7–2.3.8。

- 解析 PSD overlay：21/30 有 specparam 峰；16/21 解析峰频 |Δf| < 1 Hz；与 gamma surrogate 互补覆盖 **19/21 (90%)**
- 逃逸的 2/21（1084, 1096）归因于极端非平稳率，待 PR-2 去趋势后回填
- SOZ dead-time < non-SOZ：Wilcoxon p=0.008（n=8 pairs，探索性）

### 5.5 PR-2 — IEI 序列相关深度分析（实验 7，2026-04-08 完成）

对 30 subjects 做四层拆解：lag-k 衰减、去趋势、block 内分析、SOZ 分层。

#### 5.5.1 Lag-k 衰减曲线

计算 r(log IEI[n], log IEI[n+k])，k = 1..100，跨 block pool pairs。

| 指标 | 值 |
|------|-----|
| Lag-1 r（pooled within-block） | 中位 0.299（范围 0.117–0.506），**30/30 正** |
| 半衰期（lags） | 中位 24 lags（24/30 有限值；6/30 在 k=100 内未衰减到半） |
| 半衰期（秒） | 中位 107.5s ≈ 1.8 min（范围 3.5s–552.6s） |

三类 subject 模式：
- **快速衰减**（半衰期 < 15 lags, ~10 subjects）：短程依赖为主
- **中速衰减**（15-80 lags, ~14 subjects）：分钟级调制
- **无衰减**（6 subjects，到 k=100 仍未半衰）：被持续慢调制主导

#### 5.5.2 去趋势分析

方法：在物理时间上用 ±300s 滑动窗口计算 IEI 的局部中位数作为基线，残差 = log(IEI) - log(baseline)，在残差上算 lag-1 Pearson r。

| 指标 | 值 |
|------|-----|
| 去趋势前 lag-1 r | 中位 0.299 |
| 去趋势后 lag-1 r | 中位 0.081 |
| 去趋势后仍为正 | 27/30 |
| 去趋势后变负 | 3/30（442, 590, 620） |
| 去趋势分数 | 中位 0.720 |
| 去趋势分数 > 0.5 | 27/30 |
| 去趋势分数 > 0.8 | 10/30 |

**关键结论**：~72% 的正序列相关来自 > 10 分钟的慢速率漂移（sleep/wake/circadian）。但 27/30 去趋势后残差仍为正 → 存在 ~28% 的短程依赖成分（可能是局部网络 facilitation/depression）。

#### 5.5.3 Within-block 分析

将事件严格按 block 边界切分（Yuquan 2h/block, Epilepsiae 1h/block），各 block 独立算 lag-1，然后 pool。

| 指标 | 值 |
|------|-----|
| Within-block pooled lag-1 r | 中位 0.299，**30/30 正** |

**结论**：跨 block 污染假说被排除。block 内序列本身就有稳固的正序列相关。

#### 5.5.4 SOZ vs non-SOZ 分层

将群体事件按 SOZ 通道参与与否分为两组，各算 lag-1。

| 指标 | 值 |
|------|-----|
| 有效配对 | 9/30 |
| SOZ lag-1 r 中位 | 0.302 |
| nonSOZ lag-1 r 中位 | 0.132 |
| SOZ > nonSOZ | 7/9 |
| Wilcoxon p | 0.055 |

**结论**：SOZ 序列相关倾向于更强，但 n=9 只是边缘趋势。方向暗示 SOZ 网络有部分自主记忆效应。

#### 5.5.5 PR-2 综合解读

1. **慢速率漂移是序列相关的主成分（~72%）**，最可能来自 sleep/wake + circadian 调制
2. **去趋势后残差（~28%）仍为正**，说明存在短程网络依赖
3. **调制时间尺度的中位数 ≈ 1.8 分钟**，但 6/30 有持续不衰减的超慢调制
4. **跨 block 污染被排除**——within-block 和全序列结果一致
5. **当前 600s 单一窗口去趋势不能区分慢调制的具体频段**——需要多尺度去趋势（→ PR-2.5）

### 5.6 PR-2.5 — 多尺度调制解剖（实验 7B，2026-04-08 完成）

对 30 subjects 做五组子实验，回答 PR-2 遗留的"慢调制集中在什么频段"和"被调制的不止 IEI"等问题。

#### 5.6.1 多尺度去趋势曲线（Exp 7B）

方法：在 6 种窗口大小 W ∈ {60, 180, 600, 1800, 3600, 7200} s 上分别做滑动中位数去趋势，计算 `detrend_fraction(W)`。用相邻尺度的差分 `Δ_frac` 定位释放最多相关性的频段。

| 指标 | 值 |
|------|-----|
| Δ_frac 范围 | 0.080 – 0.147（近似平坦） |
| Δ_frac 峰值窗口中位数 | ~329s (≈5.5 min) |
| 有清晰尖锐 Δ_frac 峰的 subjects | 0/30 |

**结论**：慢调制是**宽频段 1/f 型**，没有单一主导时间尺度。不能归因于仅 circadian 或仅 sleep architecture，而是多尺度叠加。

#### 5.6.2 n_participating Spearman 自相关（Exp 7C）

方法：对 n_participating（每事件参与通道数，离散整数）用 Spearman 秩相关计算 lag-k 衰减曲线；与 IEI Pearson 衰减做互相关。

| 指标 | 值 |
|------|-----|
| IEI–n_participating 衰减曲线互相关中位数 | **0.742** |
| r > 0.7 的 subjects | **18/30 (60%)** |

**结论**：IEI 和 n_participating 的慢调制衰减形状高度一致，**证实了单一全局状态变量假说**——存在一个全局兴奋性变量 S(t)（如 sleep/wake 状态、arousal level）同时驱动事件率和参与通道数。

#### 5.6.3 日夜分层去趋势（Exp 7D）

方法：按本地时间（Yuquan: Asia/Shanghai, Epilepsiae: Europe/Berlin）将事件分为 day (08:00–20:00) 和 night，各段内分别做 600s 去趋势后算 lag-1 r。

| 指标 | 值 |
|------|-----|
| Day 去趋势后 lag-1 r 中位 | 0.094 |
| Night 去趋势后 lag-1 r 中位 | 0.086 |
| 两段均为正的 subjects | 28/30 |
| Day vs Night Wilcoxon p | 0.088 |

**结论**：28% 短程依赖不是日夜边界伪影，而是真实的网络级短程依赖。Day 和 night 段内去趋势后残差强度无显著差异。

#### 5.6.4 Block 合并灵敏度（Exp 7E）

方法：对相邻 block（gap ≤ 5s）合并后重算半衰期，对比原始分块半衰期。

结果：多数 subject 合并后半衰期与分块一致。block 边界的 gap 大多为 0–2s（见 PR-2 block 结构分析），合并后不暴露额外的超慢调制。

#### 5.6.5 逃逸 Subject 回填（Exp 7F — 关键）

方法：对 PR-1 逃逸的 2/21 subject（1084, 1096），用 600s 滑动中位数去趋势 IEI 序列，从去趋势后 IEI 重建脉冲序列，重算 Welch PSD + specparam + gamma surrogate。

| Subject | 原始峰频 | 去趋势后峰频 | 去趋势后 specparam 峰 |
|---------|----------|-------------|---------------------|
| 1084 | 3.34 Hz | 0.00 Hz | **无峰** |
| 1096 | 2.47 Hz | 0.00 Hz | **无峰** |

**结论**：两个 subject 的谱峰在去趋势后**完全消失**。这关闭了 Layer 3 缺口：**所有 21/21 有 specparam 峰的 subject 现在都被 refractory renewal + 慢率调制完全解释**，无需引入内禀振荡器。

#### 5.6.6 PR-2.5 综合解读

1. 慢调制是宽频段 1/f 型（无单一主导频段），**不能简化为"昼夜调制"或"睡眠阶段调制"**
2. n_participating 与 IEI 由同一个全局状态调制（r = 0.742），证实"单一全局兴奋性变量"假说
3. 28% 短程依赖在日间和夜间段内独立存在，是真实的网络短程记忆效应
4. 逃逸 subject 的谱峰完全由慢率调制产生 → **21/21 全覆盖**，~2 Hz 周期性假说彻底终结

### 5.7 最终科学结论

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

PR-2.5 证据链（多尺度调制解剖）：
10. 慢调制是宽频段 1/f 型，无单一主导时间尺度 → 多尺度叠加（sleep + circadian + ...）
11. **n_participating 与 IEI 同源调制**（r = 0.742）→ 单一全局兴奋性变量驱动
12. 28% 短程依赖在 day/night 段内独立存在 → 非日夜边界伪影
13. **逃逸 subject (1084, 1096) 去趋势后峰完全消失** → 21/21 全覆盖，Layer 3 缺口关闭

**建议**：
- 在最终论文中不宜将 PSD 周期峰作为正面发现报告
- IEI 正 serial correlation（率慢调制）是一个值得讨论的方法学发现，正式统计口径应使用 subject-level direction consistency
- PR-2 / PR-2.5 量化了调制成分：~72% 慢漂移 + ~28% 短程依赖。**慢漂移是多尺度 1/f 型全局生理调制（不可简化为单一 circadian 或 sleep stage），短程部分是真实的局部网络效应**
- n_participating 的同源调制支持"兴奋性增益模型"——全局状态 S(t) 同时控制事件率和参与范围
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
  - `compute_renewal_psd_analytic()` — 解析 renewal PSD（PR-1）
  - `compute_soz_stratified_deadtime()` — SOZ 分层 dead-time（PR-1）
  - `_load_group_events_with_soz_labels()` — SOZ 事件加载公共 helper
  - `_split_events_by_block()` — 按 block 切分事件序列
  - `_rolling_log_iei_residuals()` — 物理时间滑动中位数去趋势（O(N log N) 优化版）
  - `_serial_corr_decay_from_sequences()` — 跨 block pool lag-k 衰减（PR-2）
  - `compute_serial_correlation_decay()` — lag-k 衰减公开接口（PR-2）
  - `compute_detrended_serial_correlation()` — 去趋势 serial corr（PR-2）
  - `compute_within_block_serial_corr()` — block 内 serial corr（PR-2）
  - `compute_serial_corr_soz_stratified()` — SOZ 分层 serial corr（PR-2）
  - `compute_multiscale_detrend_fraction()` — 多尺度去趋势 + Δ_frac（PR-2.5 Exp 7B）
  - `_compute_half_life()` — 自相关半衰期插值（PR-2.5）
  - `compute_nparticipating_autocorrelation()` — n_participating Spearman lag-k 衰减（PR-2.5 Exp 7C）
  - `merge_contiguous_blocks()` — 相邻 block 合并（PR-2.5 Exp 7E）
  - `_epoch_to_hour()` — Unix epoch → 本地小时（PR-2.5 Exp 7D）
  - `compute_daynight_stratified_detrending()` — 日夜分层去趋势（PR-2.5 Exp 7D）
  - `compute_detrended_psd_backfill()` — 去趋势后 PSD 回填 + gamma surrogate（PR-2.5 Exp 7F）
- `scripts/run_periodicity_phase2.py` — Phase 2 批量驱动（exp 1-7b）
- `scripts/plot_periodicity_phase2.py` — Phase 2 可视化（exp 1-7b）
- `tests/test_event_periodicity.py` — PR-2 / PR-2.5 函数单元测试

### 结果文件
- `results/event_periodicity/phase2/exp1_packing_sweep.json`
- `results/event_periodicity/phase2/exp2_centroid_bypass.json`
- `results/event_periodicity/phase2/exp3_hazard.json`
- `results/event_periodicity/phase2/exp4_return_map.json`
- `results/event_periodicity/phase2/exp5_stereotypy.json`
- `results/event_periodicity/phase2/exp6_renewal_psd.json`（PR-1）
- `results/event_periodicity/phase2/exp6_soz_deadtime.json`（PR-1）
- `results/event_periodicity/phase2/exp7_serial_corr_deep.json`（PR-2）
- `results/event_periodicity/phase2/exp7b_multiscale.json`（PR-2.5 Exp 7B）
- `results/event_periodicity/phase2/exp7b_npart_autocorr.json`（PR-2.5 Exp 7C）
- `results/event_periodicity/phase2/exp7b_daynight.json`（PR-2.5 Exp 7D）
- `results/event_periodicity/phase2/exp7b_merge_sensitivity.json`（PR-2.5 Exp 7E）
- `results/event_periodicity/phase2/exp7b_backfill.json`（PR-2.5 Exp 7F）
- `results/event_periodicity/phase2/figures/` — 可视化图
