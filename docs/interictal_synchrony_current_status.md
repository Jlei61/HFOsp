# Interictal Synchrony 当前状态与关键发现

**更新时间**：2026-04-03

## 【核心判断】

这条线现在最大的进展，不是“又多画了几张图”，而是把分析主语和统计层级纠正了。

- `event-level` 已经成为主链；`block summary` 只保留兼容派生视图。
- `Epilepsiae` 已经不只是调研对象，而是同步性分析和检测器泛化约束的真实验证基线。
- 群体统计上，当前数据**还不能支持**“post-ictal reset -> gradually resynchronize -> pre-ictal peak”作为普适规律。

## 现在做到了什么

### 1. 时间轴与数据契约已经钉死

- `Yuquan`：PR1 已把 EDF annotation、归一化 seizure interval、时区和 header-driven timeline 钉死。
- `Epilepsiae`：PR1.5 已把 `*.data + *.head + SQL` 合同、inventory、manifest、day/night 规则、统一时间轴钉死。
- `Epilepsiae` 当前主分析队列不是“20 个 subject”，而是 `ready_full_artifacts = 16` 个可直接进入同步性分析的 subject。

### 2. 同步性主链已经改成 event-first

- PR4 主产物已经切到 `event rows`，不再把 block 平均值当分析主语。
- PR5 注释和聚合已经按 `event_start/end` 做完整包含判断，跨 seizure / phase / day-night / gap 边界的 event 直接排除。
- `block summary` 现在只是兼容输出，不再是统计主口径。

### 3. PR6 的统计和作图框架已经落地，但方法论被硬修正过一次

- Figure A–E、固定窗、归一化轨迹、cohort summary 这套骨架已经有了。
- 图层语义做过一轮收口：Figure A 不再允许折线穿过 seizure 时间；残留 `block` 误导词和旧回退逻辑已被清理。
- 最关键的修正不是画图，而是统计层级：`pooled trajectory` 被证明确实会产生 Simpson's paradox 伪影。

### 4. 发作检测器方向被证明改对了，但还没到验收线

- `channels -> mean -> feature -> threshold` 这条老思路被判死刑。
- `channels -> bipolar per-channel LL -> per-channel z -> active-channel fraction -> threshold` 才是物理上讲得通的主线。
- `MAD x 1.4826` 修掉后，真正带来巨大改观的不是阈值，而是**双极参考消掉共参考噪声**。
- 在 `litengsheng` / `sunyuanxin` 上，`v3 = bipolar LL + MAD x 1.4826 + frac=0.40 + dur=30` 已把 FP 从三位数/近千量级打到 `16 / 39`，但这仍然只是“方向正确”，不是跨患者验收通过。

## 关键科学发现

### 1. 单病人的漂亮曲线不能外推成 cohort 规律

- `Epilepsiae` 的 `Subject 916` 是一个很强的正例：
  - `legacy` trajectory: `rho = +0.175`, `p = 0.001`
  - `phase` trajectory: `rho = +0.185`, `p = 0.0007`
- 这说明“某些 subject / interval 确实可能存在 reset -> resynchronize”。
- 但这**不等于**整个 cohort 共享同样规律。

### 2. pooled trajectory 是错方法，不是“一个可接受的近似”

- 旧做法把 `subject x interval` 的所有观测点混进一个池子做 Spearman。
- 这在数据结构上就是错的，因为假设本来就是 **within-interval** 的，而不是 across-interval pooled 的。
- 结果上也已经出事：cohort pooled `legacy` 一度给出显著负趋势，和 `Subject 916` 的正趋势正面冲突。
- 这不是“数据太吵”，这是**统计层级错了**，典型 Simpson's paradox。

### 3. 正确层级下，当前 cohort 结论是 null

- 把检验改成 `within-interval` 之后，三个主指标的 interval-level 趋势分布都围绕 0 摆动。
- fixed-window 的 Post vs Pre 配对检验也没有稳定显著差异。
- 当前最老实的结论只有一个：
  - **假设在部分 interval / 个体上成立，但在当前 cohort 中还不是稳定、可复现的群体规律。**

### 4. 真问题是异质性，不是“再多调几个阈值”

- 如果某些 interval 强正、某些强负、总体接近 0，说明问题不在“全体一起涨不涨”，而在**什么条件下会涨**。
- 最值得怀疑的异质性来源：
  - core-only vs all channels
  - long interval vs short interval
  - seizure 完整标注质量
  - gap burden
  - patient-specific anatomy / electrode coverage

## 现阶段结论

### 值得保留的结论

- `event-level` 主链这件事是对的，而且必须坚持。
- `Epilepsiae` 可以进入主分析，但必须走自己的数据契约，不能套 Yuquan EDF 语义。
- 双极化的空间招募检测器是对的方向，继续做有价值。

### 不能再夸大的结论

- 不能说 PR2.5 已经验收通过；现在只是候选 detector 核心成形。
- 不能说 PR6 已经证明“发作前同步性上升”是 cohort-level 规律；当前证据不支持。
- 不能再用 pooled trajectory 当主检验；那是在解决错误的问题。

## 第一性原理下的破局点

### 1. 先尊重数据层级，再谈统计显著

- 分析主语必须是 `event -> interval -> subject`，不是把所有点倒进一个桶。
- 下一步主检验应该围绕 `interval-level slope / effect` 展开，再做 subject / cohort 聚合。
- 如果需要群体建模，优先考虑 hierarchical / mixed-effects，而不是 pooled correlation。

### 2. 先找异质性分层，再谈“总平均”

- 真正可能的科学突破，不是证明“所有人都一样”，而是识别**哪些 interval 会发生 resynchronization**。
- 最该先分层的不是花哨模型，而是：
  - `legacyComparableLongIntervals` vs `shortIntervals`
  - `core_only` vs `all`
  - `ready_full_artifacts` 内高 coverage subject vs 边缘 subject
  - 完整 EEG seizure interval vs 标注边界可疑 interval

### 3. 检测器别再做患者特异补丁

- 真正的验收口径已经很清楚：
  - 同一 detector core 同时支持 `Yuquan EDF` 和 `Epilepsiae *.data + *.head`
  - 用 labeled patients 做 held-out subject 验证
  - 统一输出审计表、时间线图、误差图
- 如果还靠 `ignore_initial` 之类补丁过关，那就是在骗自己。

### 4. 现在最值钱的工程工作，不是继续加图，而是补齐可判别数据

- `Yuquan`：完成 PR3，把 `subject -> ordered seizures -> seizure_intervals` 和 artifact tier 真正落盘。
- `Epilepsiae`：用现有 `ready_full_artifacts` 继续做 interval-level 分层分析；对 `missing_interictal_artifacts` 再决定是复用老资产还是从原始块重建。
- `PR7/PR8`：只在 smoke 和 legacy 对齐过关后推进全链补跑，不要拿“能跑”冒充“可比”。

## 现在最该做什么

1. 把 PR6 的主报告口径固定为 `within-interval`，彻底禁用 pooled trajectory 作为主结论。
2. 在 `Epilepsiae ready_full_artifacts` 上做 interval-level 分层：长/短间期、core/all、coverage/gap burden。
3. 推进 PR2.5 的跨数据集 held-out 验证，让检测器和同步性分析共用同一套真实验证基线。
4. 补齐 PR3 与 PR7/PR8，减少“只有一部分病人有中间产物”造成的样本偏差。

## 一句话总结

我们现在已经把**方法论上的大坑**挖出来并填掉了一半：主语从 block 改成了 event，统计从 pooled 改回了 within-interval。  
接下来真正的破局点，不是继续润色图，而是找出 **“什么条件下同步性真的会重新上升”**，以及 **“同一个检测器能否跨 Yuquan / Epilepsiae 泛化”**。
