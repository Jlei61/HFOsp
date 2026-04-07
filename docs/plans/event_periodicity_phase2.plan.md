# 间期群体事件周期性 — Phase 2 工作计划

> 前置：Phase 1 (复现 + 鲁棒性检验) 已完成，结论为"2Hz 峰是不应期伪影"。
> 本计划目标：(1) 精确定位伪影来源 (2) 用正确的工具寻找真实的时序规律。

---

## 背景：为什么需要这些实验

### 当前证据链
1. Gamma renewal surrogate p=1.0（15/30）→ 不应期效应可解释峰值
2. IEI 分布为 lognormal（30/30）→ "幂律"声称不成立
3. per-channel vs group 峰频不一致 → packing 可能引入额外伪影

### 未解的关键问题
- **Q1**：2Hz 峰究竟是 packing 窗口造成的，还是 IEI 分布固有的？
  - Gamma renewal 只证明"不应期效应足以解释"，但不应期的来源可能是 packing 窗口（算法参数），
    也可能是神经网络真正的不应期（生理学）
  - 需要分离这两个因素
- **Q2**：如果消除 packing 伪影后，群体事件还有没有可检测的时序结构？
- **Q3**：正确的时序分析工具（Hazard Function、Return Map）能揭示什么？

---

## 实验 1：PackWinLen 参数扫描（最高优先级）

### 科学目的
在同一 subject 上，仅改变 packing 窗口大小 W，观察 PSD 峰频是否随 W 系统性变化。

### 设计
- **自变量**：W ∈ [100, 150, 200, 300, 400, 500, 600, 800, 1000] ms
- **因变量**：specparam 检测到的最强周期峰频率 f_peak
- **受试者**：chengshuai（有完整 _gpu.npz，原始 W=500ms）
  + huangwanling（W=300ms，事件最多）
- **控制**：IEI 下界的理论值 f_theory = 1/W 作为参考线

### 实现
1. 从 `_gpu.npz` 加载逐通道 raw detections
2. 调用 `build_windows_from_detections(dets, window_sec=W)` 重新 packing
3. 对新的 packed 事件序列计算 PSD + specparam
4. 绘制 W vs f_peak 散点图，叠加 y=1/x 参考线

### 预判
- 如果 f_peak 与 W 强相关（单调递减）：**packing 窗口是伪影的直接来源**
- 如果 f_peak 不随 W 变化：**峰有非算法来源**（但仍需 surrogate 确认是否为真振荡）

---

## 实验 2：质心旁路实验（Centroid Bypass）

### 科学目的
完全绕过 packing 算法，用 lagPatRaw 质心直接定义事件时间戳，检验周期峰是否消失。

### 设计
对每个 subject，从 `lagPatRaw` 提取三种事件时间戳：
- (a) **Window Start**：`packedTimes[:, 0]`（原始方法）
- (b) **Mean Centroid**：`mean(lagPatRaw, axis=0)` + block_start
- (c) **Ignition Centroid**：`min(lagPatRaw[:, i])` for each event i（最早参与通道）

对三种序列分别计算 PSD + specparam + ISI-shuffle surrogate。

### 预判
- 如果 (a) 有峰但 (b)(c) 无峰：**峰 100% 来自 packing 窗口量化**
- 如果三者都有峰：**峰非 packing 伪影，需要 Gamma surrogate 进一步检验**
- 如果 (c) 比 (b) 信噪比更高：**点火模式包含更多时间结构**

### 实现
1. 从已有 `lagPatRaw` 提取质心（不需要 _gpu.npz）
2. 用质心时间 + block start_t 构建绝对时间序列
3. 以 delta 脉冲模式（非 rectangle）构建 pulse train
4. 计算 PSD + specparam

---

## 实验 3：IEI Hazard Function 分析

### 科学目的
用点过程的正确工具描述事件间隔动力学，不再假设平稳周期性。

### 方法
- **Hazard Function**：H(t) = f(t) / (1 - F(t))
  - f(t) = IEI 概率密度
  - F(t) = IEI 累积分布
  - H(t) 的物理含义：已经等待 t 秒后，下一瞬间事件发生的条件概率
- 核密度估计 f(t)，避免 bin 依赖

### 预期发现
- 对于 packing 事件：H(t<W)≈0（强制死区），H(t>W) 快速上升后衰减
- 对于质心事件：死区缩短或消失，暴露真实的不应期
- SOZ 通道的 H(t) 与 non-SOZ 的 dead-time 长度是否不同？

### 实现
1. `compute_hazard_function(iei, bandwidth=None)` → freqs, hazard
2. 对 per-channel 和 group 分别计算
3. 比较 window-start vs centroid 事件的 H(t) 差异

---

## 实验 4：IEI Return Map（Poincaré Plot）

### 科学目的
用 IEI[n] vs IEI[n+1] 散点图诊断事件间隔的时间结构。

### 预期模式
- **独立同分布（IID / 续变过程）**：均匀圆形/椭圆形分布
- **确定性振荡**：紧密聚集在一个点 (T, T) 附近
- **不应期 + 随机**：L 型分布（短-长和长-短交替）
- **状态切换（burst/pause）**：多个聚集簇

### 实现
1. 绘制 IEI[n] vs IEI[n+1]，log-log 轴
2. 计算 serial correlation corr(IEI[n], IEI[n+1])
3. 对比 window-start, centroid, per-channel 三种事件定义

---

## 实验 5：传播立体型分析（Propagation Stereotypy）

### 科学目的
寻找真正的空间-时间规律：SOZ 起源的群体事件是否有固定传播路径？

### 方法
- 从 lagPatRank 提取每个事件的通道激活顺序
- 计算 rank consistency：`mean pairwise Kendall tau across events`
- 分层：SOZ-参与 events vs non-SOZ-only events

### 预期
- SOZ events 的传播路径高度固定（stereotypical）→ 这是真正的"规律"
- non-SOZ events 的传播路径随机 → 无固定模式
- 这比频域 PSD 分析更能揭示致痫网络的组织原则

---

## 优先级排序

| 优先级 | 实验 | 理由 |
|--------|------|------|
| P0 | 1. PackWinLen 扫描 | 直接定位伪影来源，是否 f_peak ~ 1/W |
| P0 | 2. 质心旁路 | 最快验证 packing 是否是唯一伪影源 |
| P1 | 3. Hazard Function | 正确的点过程工具，替代 PSD |
| P1 | 4. Return Map | 简单、直观、信息量大 |
| P2 | 5. 传播立体型 | 寻找真正的空间-时间规律 |

---

## 工程实现

### 新增/修改文件
- `src/event_periodicity.py` — 增加：
  - `build_population_events_from_detections(dets, window_sec)` — 用于 sweep
  - `load_centroid_event_times(lagpat_files)` — 质心旁路
  - `compute_hazard_function(iei)` — hazard function
  - `compute_iei_return_map(iei)` — return map statistics
- `scripts/run_periodicity_phase2.py` — Phase 2 批量驱动
- `scripts/plot_periodicity_phase2.py` — 可视化

### 数据依赖
- 实验 1：需要 `_gpu.npz`（仅 Yuquan 可用）
- 实验 2-4：仅需 `lagPatRaw` + `packedTimes`（两个数据集均可用）
- 实验 5：需 `lagPatRank`（两个数据集均可用）
