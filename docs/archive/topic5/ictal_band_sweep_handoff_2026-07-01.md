# Handoff — 发作能量 ↔ 间期 HFO 几何：频带扫描调查（文献研究用）

date 2026-07-01 · 状态：brainstorm 中，待文献后定频带集 · 主体=扫频带（第一优先）

> 这份 handoff 给你带去查文献用。三段：①我们这轮厘清/新发现了什么 ②现有 field similarity 的真实数值（含 per-subject）③要做的事 + 文献清单。

---

## 1. 我们这轮厘清的东西（朴素话）

一条推理链，每一步都核过代码：

1. **发作"场"喂进统计的量 = 宽带功率的能量，不是时序。** `bb_zt = baseline_robust_z(band_power_trace(signal, 1–45Hz))` —— 每触点"功率比发作前基线高几个 sigma"，是**群体放电/招募的代理量**，不是 z-ER 的峰值先后。
2. **间期"轴"= 高频事件的时序几何。** 间期模板（typical_rank / lagPat）是**间期 HFO 事件**的峰值传播先后顺序 —— 是**高频事件**定义出来的一条刻板通路。
3. **当前 field similarity（`align_maxab`）= 逐窗、逐发作算的，不是平均发作场。** 每个时间窗把该窗的发作能量场（平滑到 2D 网格）和**间期顺序几何**（固定）求 |空间相关|，A/B 取 max；逐窗 → 跨发作取 median / per-subject 统计。
4. **数值上**：发作能量（宽带和 HFA 都）中等程度贴间期几何（cohort 中位 |corr| 0.5–0.7，见 §2）。
5. **核心命题（refined）**：我们的发现 = **癫痫间期 HFO 事件的刻板传播路径 ≈ 发作早期的能量传播路径**（空间共定位）。
6. **关键张力 —— 为什么必须扫频带**：间期几何来自 **HFO（高频事件）**；可发作相似度我们是在**宽带 + HFA** 上算的，两边频带没对齐。扫频带就是判别器：
   - **HFA / ripple 特异峰** → 发作真的用高频事件重走间期 HFO 通路 →（机制命中，高优先级建模靶子）
   - **跨频带平铺、普遍低** → 全场同步均一化增强（无聊，"贴轴"是少数高基线/SOZ 触点造的假象）
   - **低 / 中频峰** → 发作主频节律 / 长程输入驱动

---

## 2. 现有 field similarity 数值（per-subject，你要的）

⚠️ **三个 caveat 先讲死**：
- (a) 这是**裸 |corr|，没有空间 null** —— 两个平滑场的空间自相关本身就会把 |corr| 抬高，**不能当"显著"读**，只能描述性。扫频带必须配 null。
- (b) broad / narrow 两队列**轴的构造不同**（broad = swap / template-earliest 集合；narrow = compact-core 端点），**不能 1:1 比**。
- (c) 是绝对值（不分能量高在早端还是晚端）。

**口径**：early-ictal onset 滑窗（10s 窗、ictal_fraction≥0.5）；每发作对窗取 median，再对被试取发作 median。

### BROAD（n=9）
| subject | n_sz | bb per-sz-med | hfa per-sz-med |
|---|---|---|---|
| epilepsiae_1077 | 8 | 0.75 | 0.52 |
| epilepsiae_1096 | 8 | 0.53 | 0.69 |
| epilepsiae_1125 | 13 | 0.54 | 0.43 |
| epilepsiae_1150 | 8 | 0.53 | 0.59 |
| epilepsiae_139 | 4 | 0.70 | 0.53 |
| epilepsiae_253 | 6 | 0.76 | 0.81 |
| epilepsiae_620 | 6 | 0.49 | 0.56 |
| epilepsiae_635 | 16 | 0.53 | 0.68 |
| epilepsiae_916 | 37 | 0.74 | 0.63 |
| **cohort median** | | **0.54** | **0.59** |

### NARROW（n=7，轴=compact-core 端点）
| subject | n_sz | bb per-sz-med | hfa per-sz-med |
|---|---|---|---|
| epilepsiae_1096 | 8 | 0.59 | 0.89 |
| epilepsiae_1125 | 13 | 0.75 | 0.73 |
| epilepsiae_1146 | 23 | 0.78 | 0.57 |
| epilepsiae_253 | 6 | 0.58 | 0.71 |
| epilepsiae_384 | 10 | 0.56 | 0.63 |
| epilepsiae_442 | 20 | 0.55 | 0.60 |
| epilepsiae_958 | 10 | 0.61 | 0.75 |
| **cohort median** | | **0.59** | **0.71** |

**初步迹象（不是结论）**：两队列的 HFA cohort 中位都 ≥ 宽带（broad 0.59≥0.54；narrow 0.71≥0.59），暗示这个一致可能偏高频。但**只有两个频带点、没 null、没细分**，不能下结论 —— 这正是扫频带要钉死的第一件事。

来源：`results/topic5_ictal_recruitment/field_dynamics{,_narrow}/per_seizure_metrics.csv`，列 `align_maxab`。

---

## 3. 要做的事 + 文献清单

### 设计骨架（已锁的）
- 参考几何 = **间期 timing 顺序场，固定不变**（不用间期同频带能量场 —— 我们没定义它）。
- 扫**发作能量**跨频带 vs 这条固定几何；每频带配**空间 null**（旋转/通道置换）+ **宽带 1/f 控制**。
- 判别器：平低=全场同步；低/中峰=节律/输入；**HF 特异峰=HFO 重走几何（目标）**。
- 两条必须连上（你点名）：(a) 基本间期事件能量假设；(b) 发作早期频带特异模式。

### 技术约束（已查）
- 加频带 = `band_power_trace(band=(lo,hi))` 重算进 cache，FFT-bin 口径，无滤波器设计问题。
- 采样率：多数 1024Hz（可到 ~500Hz），`139 / 253` 是 512Hz（只到 ~250Hz）。**全队列共同天花板 ~250Hz（ripple 可及）；fast-ripple（>250）只能在 1024Hz 子集，会牺牲 139/253。**

### 文献清单（你去查，每条都绑一个设计决定）

**A. 发作起始的频带特异模式**（→ 定 band grid + "发作早期频带特异"该对标什么）
- ictal onset patterns：低电压快活动 LVFA（fast/γ，>~20–30Hz）vs 高同步 hypersynchronous（低频棘波）vs 其他。
- 关键词/作者：Perucca 2014 *Brain*（ictal onset patterns 分类）；Gnatkovsky & de Curtis（onset 快活动 + 抑制机制）；Wendling；Velasco。
- 要回答：早期"有组织的招募"住在哪个频带？LVFA→γ/fast；hypersync→低频。决定我们重点扫哪段。

**B. HFO（ripple 80–250 / fast-ripple 250–500）与致痫网络**（→ 锚"HFO 重走"假设 + 定 HF 天花板）
- interictal HFO 作 SOZ biomarker：Jacobs、Zijlmans、Bragin、Worrell。
- ictal HFO 是否沿 interictal HFO 网络复现？
- 我们的间期轴**就是 HFO 传播** → 该匹配的应是 **ripple 带**。要不要冲 fast-ripple（需 >250 → 牺牲 139/253）就看这条文献值不值。

**C. 宽带 1/f shift = 放电率代理**（→ 方法学控制，否则"HF 特异"可能只是宽带溢出）
- Manning 2009 *J Neurosci*；Miller（broadband power-law shift = 群体放电率）；Ray & Maunsell（high-γ vs broadband）。

**D. 发作招募 / 行波 / ictal core**（→ 锚"能量传播路径" + 为什么时间结构难）
- Schevon 2012 *Nat Commun*（ictal core vs penumbra, multiunit）；Smith 2016；Liou（traveling waves）。
- "系统变得过度同步非线性"= ictal core 饱和招募 → 时间结构（沿几何渐次）为什么难看、别人怎么处理 → 决定时间结构那块只做探索。

**E. 间期→发作：发作是否沿间期通路**（→ 锚核心命题本身）
- spike/HFO 网络预测发作传播；epileptogenic network 稳定基质。

### 决定项 status
- [LOCKED] 参考几何 = 间期 timing 顺序场。
- [LOCKED] field similarity 逐窗逐发作（非平均场）。
- [OPEN，待文献] 频带集 + HF 天花板（ripple 250 全队列 vs fast-ripple 牺牲 139/253）。
- [OPEN，待文献] 主分析窗（发作早期；具体对标 onset-pattern 文献）。
- [探索] 时间结构沿几何渐次（过度同步 → 只做探索）。
- [探索] 能量跟随 geometry vs SOZ（broad+narrow 合并；SOZ 非关键）。

---

**回来后**：你查完文献定了频带集 + HF 天花板 + 主窗，我把它写成正式 design spec（`docs/superpowers/specs/`）→ 实现计划。
