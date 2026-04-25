我先仔细把你的四个观察和问题分解过一遍，再做几次有针对性的文献检索，最后给出分层讨论 + PR-6 计划建议。# PR-6 候选方向分析与讨论

先做一次整理，再给出分项判读。下面的讨论我尽量锚定到已发表文献，尤其是 Smith et al. 2022 eLife（IED 双向行波）和 Magalhães et al. 2025（FCD 小鼠 IED rate post-ictal 双峰升高，无 pre-ictal 变化）这两篇，因为它们**直接对应了你观察 1 的 forward/reverse 模板和我们 PR-5 的 post-ictal 招募信号**——这提示我们已经部分在复现经典现象，但还没有把解释做完。

---

## 1. 总体分类判断

| 观察 | 主 topic | 跨 topic 桥 | 建议新命名 |
|---|---|---|---|
| Obs 1（两种模板的语义 / 发作时序对比 / 距离空间） | **Topic 1** 深化 | Topic 3（模板 vs seizure onset propagation） | PR-6-A + PR-6-B |
| Obs 2（rate burst × seizure 耦合、cluster 化）| **Topic 2** 扩展 | Topic 1（burst 内模板占比）| PR-6-E |
| Obs 3（identity-bias 核心节点 + 二次筛选）| **Topic 3** 正式扩展 | Topic 1（二次筛选后的 PR-4 重跑）| PR-6-C |
| Obs 4（两模板的物理关系 / 配对 / 节律）| **Topic 1** 深化 | Topic 2（模板序列的时间结构）| PR-6-D |

**我的建议是不开 Topic 4**。Obs 2 虽然看似跨尺度，但它回答的仍是"事件之间的时间结构是什么、与发作如何耦合"——这正是 Topic 2 原本的范围，只是把"slow modulation"扩展到了"burst-level coupling with seizures"。把它作为 Topic 2 的 PR-6-E 更合适，避免 topic 爆炸。

Obs 1c（距离空间）其实是个方法论升级，可以作为新 metric 被 PR-6-A/B/D 共用，而不必自己成为一个独立 PR。

---

## 2. 核心文献锚定（影响结论与实验设计的关键参考）

以下几篇直接决定了哪些假设已经有先验支持、哪些是全新主张，**实验设计必须显式体现这些边界**：

Smith et al. 2022 eLife: "The majority of IEDs are traveling waves, traversing the same path as ictal discharges during seizures, and with a fixed direction relative to seizure propagation. Moreover, the majority of IEDs, like ictal discharges, were bidirectional, with one predominant and a second, less frequent antipodal direction." — **这和我们 k=2 + 9 个 forward/reverse subject 的 inter-cluster r < -0.5 在几何上是同一现象的两种测量**。Smith 用 MEA 看 μm-scale 行波；我们用 HFO group event rank 看 mm-scale 通道序。这是 Obs 1 和 Obs 4 最强的外部锚。

Korzeniewska et al. 2014: "In patients with focal ictal onsets, the patterns of propagation recorded during pre-ictal... and interictal... intervals were very similar to those recorded during seizures." — 用 SdDTF 在 HFO band 展示 interictal propagation recapitulates ictal。这支持"dominant template ≈ seizure propagation"的可检验假设。

Magalhães et al. 2025 (FCD mouse): "IED rate increases significantly after seizures, indicating a postictal effect rather than a preictal one, with the post-ictal phase displaying a two-peak pattern... an initial peak in IED rate emerged 5–10 minutes post-seizure." — **这正是我们 PR-5-B 的 post-ictal absolute rate elevation (+65 ev/h) 在另一个物种 / 另一个模型上的复现**。说明我们不是发现了新现象，而是把 Magalhães 在小鼠上说的东西做到了人 HFO cohort。

Tomlinson et al. 2025 Brain Communications: "The irritative neocortex comprises a network of semi-independent, highly cohesive communities with stereotyped local spike propagation patterns. Distributed coupling of spikes between communities was driven by outflow from the seizure onset zone and fluctuated over time in association with... temporal proximity to seizures." — 刚好是我们 identity-bias 86% 的网络学解释：模板是 cohesive communities，核心通道即 outflow hubs。直接支持 Obs 3。

Koubeissi 2016 (Epilepsiae cohort): "Seizure clusters were defined by the occurrence of at least two consecutive seizures with interseizure intervals of <4 h." — Obs 2 的 seizure cluster 定义的合理默认值。

Vargas-Irwin et al. 2015 (SSIMS): "We present a method to evaluate the relative similarity of neural spiking patterns by combining spike train distance metrics with dimensionality reduction... Vectors of pair-wise distances are used to represent the intrinsic relationships between multiple activity patterns at the level of single units or neuronal ensembles." — 直接支持 Obs 1c 的距离嵌入思路，是成熟的方法论。

---

## 3. 逐观察讨论

### 3.1 Obs 1 — 两种稳定时序模式（Topic 1 深化 + Topic 3 桥接）

这是四个观察里**最接近已发表文献预测**的一条，也是 PR-6 最该启动的入口。

#### 3.1a 跨 subject 语义对齐

k=2 聚类的两个 template 标签（C0 / C1）是任意的——KMeans 初始化随便怎么跑都可能翻。要在 cohort 层面说 "template A 普遍有某性质"，必须先定义一个**subject-agnostic 的语义轴**。你列的三个候选（出现次数、发作邻近占比、与发作时序相关性）各自有问题：

- "raw count 大的是 T0" 最简单但**语义空洞**——只告诉我们哪个模式在 baseline 更被招募，不承诺任何机制
- "post-ictal 主导的是 T_recruit"——可以用 PR-5-B 的 post-ictal 占比定义，但**有循环论证风险**：PR-5-B 本身说的是 dominant template 在 post-ictal 更多，这里又用 post-ictal 主导定义 dominant，会把结果内置进定义
- "与 ictal onset 通道序列相关的是 T_ictal-aligned"——**最接近机制语义**，而且直接对应 Smith 2022 的双向预测

**推荐的对齐规则（分三层，带 fallback）**：

1. **主语义（Smith-aligned）**：对每个 subject，从 ictal EEG 提取每次 seizure 的 channel-wise earliest activation rank（在我们参与的 n_participating 通道集合上），求 median seizure onset rank vector `r_sz`。两个 template centroid 分别和 `r_sz` 算 Spearman ρ：ρ 高者为 **T_ictal-aligned**（"forward"），另一个为 **T_antipodal**（"reverse"）。
2. **备选语义（Recruitment-anchored）**：当 ictal onset rank 不可用或 n_participating 太小时（预期少数 subject），用 per-template baseline rate 较高者为 **T_dominant**。
3. **门槛与失败合同**：若 `|ρ_T0_sz - ρ_T1_sz| < 0.2`（两模板对发作起始序列几乎等价），标记为 `ictal_unresolved`，该 subject 在 T_ictal-aligned 分析中退出但保留在 T_dominant 分析中。

这里值得强调：**我们的合同里，ictal onset 通道排序是可从原始 data 拿到的**——Epilepsiae 有 seizure onset 标注，每次 seizure 的 per-channel 最早时刻可以直接提取。这比大部分文献的处境好（他们通常只有二值的 SOZ 标签）。

#### 3.1b 模板 vs seizure onset propagation

**科学问题**：刻板 interictal 模板（尤其是 T_ictal-aligned）在通道序列上是否"重现"发作起始的传播路径？

**假设**：

- **H1 (Smith 2022 prediction)**：对每个 subject，至少有一个 template 与 seizure onset rank 呈显著正相关；forward/reverse subject 中另一个 template 呈显著负相关（反向重现）
- **H0 (null)**：两个 template 与 seizure onset rank 的相关都在 |ρ| < 0.3 范围内，cohort 层面看不到方向性

**实验合同**：

- 输入：对每个 subject，(a) 两个 cluster centroid ranks；(b) 每次 seizure 的 channel-wise earliest activation rank（从 Epilepsiae onset window 内的 HFO / spike detection）；(c) 按 PR-5-A 的 retained cohort 做（main n=23 / aux n=22）
- 主指标：`ρ_T_aligned_sz`（T_ictal-aligned 与 median seizure onset rank 的 Spearman）
- Cohort 级：sign test（正相关数），one-sample Wilcoxon against 0
- 对 9 个 forward/reverse subject 单独跑：检查 T_ictal-aligned 与 T_antipodal 是否在 `ρ_sz` 上呈现显著相反方向（这是 Smith 2022 最直接的预测）

**Falsification**：

- 如果 `ρ_T_aligned_sz` cohort 中位数在 ±0.2 内，且 sign test p > 0.1 → 没有"interictal 模板重现 ictal 通路"证据，Smith 2022 在 HFO / population event 层面不复现
- 如果 forward/reverse subject 中 T_ictal-aligned 与 T_antipodal 在 `ρ_sz` 上同号（而不是相反），反向传播的"行波"解释在我们数据里不成立
- 如果 `ρ_sz` 和 SOZ 通道占比高度相关但 ictal onset rank 信号消失，说明我们看到的是"模板富集 SOZ 通道"的粗结果，不是真正的序列匹配——这对应 "Spike onset and seizure onset seemed to be distinct networks in most cases" 这一负面文献，我们要诚实报告。

**关键边界**：Epilepsiae 的 seizure onset annotation 是二值的（onset / not onset），channel-wise **earliest activation time** 需要我们重新从 raw EEG 的 ictal window 里提取，而且需要一致的 detection threshold。这是一块实实在在的工程量，建议只在 PR-6-A P0 里对 Epilepsiae 做（Yuquan 的 ictal window 数据质量我们之前已经知道更差）。

#### 3.1c 距离空间与 embedding（方法论升级）

Kendall τ / MI 都是**rank-based 相似度**，不是欧式距离，而且它们不定义事件到"两模板之间"的位置。你的直觉对了：把每个事件嵌入到一个可解释的距离空间，能一次性解决可视化、insight、后续 metric 的构造。

**推荐方案**：参考 SSIMS (Vargas-Irwin 2015) 的思路但不完全照搬。具体：

1. 对每个 subject，每个事件 i 都有一个 rank vector `r_i ∈ R^C`（C 通道）
2. 计算事件-事件的 Kendall τ 距离矩阵 `D_{ij} = 1 - τ(r_i, r_j)`（距离式度量，在 [0, 2] 上）
3. 用 **classical MDS** 而不是 UMAP——MDS 保留全局距离结构，这里我们要的就是"每个事件离两个 template centroid 有多远"这种全局量；UMAP 强调 local topology，可能把两个 template 塞得过分远或过分近
4. 在 MDS 空间里直接投影：新 metric `π_i = (d_i_T0 - d_i_T1) / (d_i_T0 + d_i_T1)` ∈ [-1, 1]，表示事件相对 T0 / T1 的位置极化度
5. `π_i` 是一个**连续变量**，可以代替现有所有基于 cluster label 的离散分析：PR-4C 可以用 `π` 做 pre vs post 的连续调制检验，PR-5-B 也可以复核

**为什么这比 τ 好用**：

- 现有的"binary cluster label + within-cluster τ"是二阶统计量，把丰富的几何信息压缩掉了
- `π` 承载了"这个事件是 pure T0 / pure T1 / 混合中间态"的连续谱；如果大部分事件是中间态，我们过去的 cluster label 其实在"硬切一个连续分布"
- `π_i` 的时间序列本身是可分析对象：ACF、与 rate 的互相关、seizure-triggered average 都能跑

**要注意的陷阱**：

- MDS 在 n_events = 193k（如 1073）时 O(n²) 计算量爆炸。必须用 **landmark MDS** 或先在 2000 event 子采样上定 template 锚，再把所有事件投影上去
- 低 n_participating 的事件 rank vector 含大量 ties / missing，距离会不稳。必须有门槛（e.g. `n_part ≥ 5`）并在二次分析中报告

#### 建议：PR-6-A（语义对齐 + seizure onset 比较）和 PR-6-B（距离空间）合并成一个 P0 工作包

它们数据依赖相同、工程链一致，分开只会增加协调成本。

---

### 3.2 Obs 3 — Identity-bias 核心节点（Topic 3 正式扩展）

先处理这条，因为它是四个观察里**最能直接落到 where 问题、最有 surgical 相关性**的一条，而且我们 Topic 3 的 Epilepsiae per-channel 基础设施刚解锁。

#### 3.2a 你的诊断是对的

Topic 1 主文档 §3.2 说 86% 的 within-cluster τ 来自 identity ordering。换句话说：**模板的"刻板"主要是"少数几个通道永远排在前面 / 后面"贡献的**。这和 Tomlinson 2025 的 "cohesive communities with stereotyped local spike propagation patterns... driven by outflow from the seizure onset zone" 是同一现象——我们看到的不是全通道的整齐序列，而是少数 hub 通道的稳定极化。

你问的"这些通道是不是跨电极分布"很关键。Refine 筛 HFO 高发通道时，如果这些 hub 通道全来自一根电极触点，说明我们其实在看"电极末端的局部现象"；如果跨电极，说明是真网络级的 source/sink。

#### 3.2b 核心节点的形式定义

**推荐定义（多指标联合，不单挑 rank 稳定性）**：

对每个 subject 的每个 stable cluster，对每个通道 ch 算：

1. `rank_stability_ch`：该通道在该 cluster 内所有事件 ranks 的 1/std（或 IQR 的倒数）
2. `rank_polarity_ch`：`|median_rank_ch - center_rank|` / `(C-1)/2`，衡量极化程度
3. `participation_ch`：该通道参与 cluster 内事件的比例
4. 合成 `coreness_ch = rank_stability * rank_polarity * participation`

取 `coreness` 前 20% 为 **core source**（median rank 最小的那一端）和 **core sink**（median rank 最大的那一端）。再加一层 `n_participating ≥ 3` 的硬门槛——否则"稳定的早"可能就是"总是参与、还总是在场景里的那个通道"。

#### 3.2c 解剖锚定

对每个 subject，核心 source 和核心 sink 通道：

- 与 SOZ 标注（Yuquan 二值 / Epilepsiae 三值 focus_rel i/l/e）做 Fisher's exact test
- 跨电极分布：核心通道分属几根电极？是否 >1？
- 对 9 个 forward/reverse subject，**关键检验**：T0 的 source 是否就是 T1 的 sink，反之亦然？这是 Smith 2022 bidirectional traveling wave 预测的直接体现——同一个 pathway 被两个方向遍历

#### 3.2d 二次筛选后重跑 Topic 1 分析

你提的这个很值得做：如果把每个 subject 只保留其 core source + core sink 通道（典型约 6-10 个），重跑 PR-2（cluster templates）、PR-4B (rate × stereotypy)、PR-4C (seizure proximity)：

**预期（可证伪）**：

- core-only τ 应该显著高于 full-channel τ（因为去掉了参与度低、贡献 noise 的通道）
- identity-bias fraction 应该下降（因为剩下的通道都是 hub，identity 贡献被归一化）
- PR-4C 的发作邻近信号可能从 null 变成可见——之前被"几十个非 hub 通道的随机参与"稀释了

**Falsification**：

- 如果 core-only τ 跟 full τ 差不多，说明"hub 通道集合"不是 τ 的主要驱动，identity-bias 解释要调整
- 如果 PR-4C 在 core-only 仍然全 null，那 Topic 1 主文档 §3.4 的"模板内部几何无发作邻近调制"结论进一步坚实

**重要警告**：二次筛选 = 数据依赖的选择偏差。必须用 **held-out seizure** 或 **split-half** 来定义 core 再在另一半上测试 PR-4C——否则就是在同一份数据上先挑通道再算统计量，p 值没意义。这个合同要写死。

#### 建议：PR-6-C 独立 P0，先跑 Yuquan（标注干净、n=9 有 forward/reverse 对应）再扩到 Epilepsiae

---

### 3.3 Obs 4 — 两种模板的物理关系（Topic 1 深化）

这条我觉得最 exciting，但也最容易走偏——需要先把可证伪的假设列清楚。

#### 3.3a 候选假设（三选一，互斥）

- **H_homologous**：同一生理 generator 的两种 state（例如 "interictal up state vs down state"），通道参与率相似但 rank 几何相反
- **H_antagonistic**：两个独立 generator（不同网络），竞争占用 event 时间窗口；一个活跃时另一个被抑制
- **H_bidirectional**（Smith 2022 预测）：同一 pathway，两个方向的行波；和 H_homologous 的差别在于它是明确的 traveling wave geometry，不是 mere state 切换

#### 3.3b 可分辨的实验

三个假设在**事件序列**和**通道参与**上有不同预测：

| 指标 | H_homologous | H_antagonistic | H_bidirectional |
|---|---|---|---|
| 事件序列 transition matrix P(T_{n+1} \| T_n) | symmetric, near independence | asymmetric, self-reinforcing clumps | symmetric, may have refractory after transition |
| Cross-template IEI vs within-template IEI | 相似 | within-template 短（bursts of same template） | 若 bidirectional 有"换向延迟"，cross 更长 |
| 两 template 的 n_participating 分布 | overlap 高 | 可能 disjoint（不同 generators） | overlap 极高（同 pathway） |
| 通道极化反转的 lag 分布 | 无结构 | 无结构 | 单峰（ms 级 propagation time） |

**具体实验链**：

1. **事件级 transition 分析**：把每个 subject 的事件序列按时间排列，提取 template label 序列。算一阶 Markov transition matrix。用 permutation null（shuffle 标签保留 rate）检验是否显著非独立。
2. **Cross-template IEI 分布**：分别计算 within-T0, within-T1, T0→T1, T1→T0 的 IEI 分布。若 H_antagonistic 正确，within 应显著短于 cross；若 H_homologous 正确，三种 IEI 应统计一致。
3. **通道参与集合 Jaccard**：两个 template 的 typical participating channel set 的 Jaccard 相似度。H_bidirectional 预测 Jaccard 极高（>0.8）；H_antagonistic 预测低（<0.5）。
4. **连续 `π` 分析（用 3.1c 的 embedding）**：`π` 时间序列的 ACF 和分布形态——若双峰 U 型分布，对应 H_antagonistic（系统在两 state 间切换，少有中间态）；若单峰宽分布，对应 H_homologous。

#### 3.3c 这条线的 falsification

若 transition matrix 接近独立（`P(T_{n+1} \| T_n) ≈ P(T_{n+1})`），cross-IEI ≈ within-IEI，Jaccard ≈ channel overlap 期望——三种假设全部都不成立，我们观察到的"两个模板"其实只是 KMeans 在连续 `π` 分布上的硬切。这是 **非常重要的 null 结果**，说明 Obs 4 的"两种模式是离散存在"的前提本身需要修正——这和 3.1c 的连续 π 视角非常契合。

#### 建议：PR-6-D 独立 P1，可以和 PR-6-B 共享 embedding 代码

---

### 3.4 Obs 2 — Rate burst × Seizure 耦合（Topic 2 扩展）

这是最大胆的一条，也最需要小心，因为 preictal biomarker 文献里失败的工作太多了。

#### 3.4a IGE-burst 是否存在 + 与 Topic 2 的关系

Topic 2 §3.5 已经说 slow modulation 是**宽频段、近 1/f** 的。你观察到的"rhythmic burst"需要和这个结论对齐：

- 如果 burst 就是 1/f 过程的高振幅尾，那 "burst" 不是独立现象，只是 rate 分布的重尾被我们可视化放大了
- 如果 burst 有**特征时间尺度**（e.g. ~8-12h 周期），那它是 Topic 2 之外的新维度

**检验合同**：

1. 形式定义 burst：`rate_1h > 2 × median_24h` 连续 ≥2h。或用 Koubeissi 2019 的 BCG/ICG 类定义迁移到 event rate。
2. Burst duration / ISI 分布：若是 1/f 过程，duration 应呈 power-law；若有节律，应有 characteristic scale（Gamma / lognormal）
3. Burst 起始时间的 PSD：若有昼夜（已被 PR-4A cohort null 排除）之外的节律，应在特定频率出峰
4. **Cohort 分层**：按 burst 规律度分层（e.g. `burst_count / total_hours` 和 duration CV），分"有明显 burst" vs "平滑 rate"两组，后续 seizure 耦合分析分层看

#### 3.4b IGE-burst vs Seizure 的时间方向

这是最容易被"post-ictal 效应"污染的一条。**Magalhães 2025 在小鼠明确说 IED rate 升高是 post-ictal 而不是 pre-ictal**。我们 PR-5-B 也是 post-ictal 信号。如果你看到"burst 之后常有 seizure"，必须分辨：

- (a) Burst → Seizure（burst 是 preictal 信号）
- (b) Seizure → Burst（burst 是 postictal 信号）
- (c) Shared driver → both（都是某个慢状态变量的表现）

**检验合同**：

1. 对每个 burst peak t=0，看 seizure 到达的时间分布：若质量集中在 t>0（burst 之后），是 (a) 或 (c) 的 signature；若集中在 t<0，是 (b)
2. 条件化：只看"前一次 seizure 已经过去 >12h"的 burst（避免 postictal 污染），这些 burst 之后 6h 内 seizure 密度是否仍显著高于 baseline
3. Shuffled null：打乱 seizure 时间戳保留 burst 结构，重复测试
4. Granger-style：burst indicator 时间序列的 Granger causality → seizure rate

若只有 (b) 而无 (a)，PR-6-E 的 "preictal" 叙事立刻落空，只剩 "post-ictal rebound" ——这其实是对 PR-5-B 的扩展验证，价值仍然有。

#### 3.4c Seizure cluster vs isolated seizure

Koubeissi 2016 的 <4h ISI 定义是 Epilepsiae cohort 的自然选择。重跑 PR-4C 和 PR-5-B，对 cluster 内 seizure 和 isolated seizure 分别做 peri-ictal average。**预期**：

- Cluster 内第二次及之后的 seizure：preictal window 和 "another seizure 的 postictal window" 重叠，信号会被污染 → 应该把这些 seizure 从 preictal 分析中踢出，或用 cluster-first seizure 作为分析单元
- Isolated seizure：干净的 peri-ictal window，是 preictal signal 最可能出现的地方

这实际上是 PR-4C / PR-5-B 结果的**敏感性分析**——如果 cluster-first seizure 上 preictal 信号浮现，而全 seizure 分析下 null，我们就有理由相信之前把 cluster 内 seizure 一起统计稀释了信号。

#### 3.4d 建议

PR-6-E 作为 P1，硬前置是 PR-6-A（至少完成语义对齐，才能谈 "burst 期间哪种 template 占比更高" 这种 Obs 2 子问题）。

---

## 4. 建议的 PR-6 计划

正式替换 Topic 1 §7.9 的 KONWAC v2 deferred placeholder：

| 子 PR | 命名 | Topic 归属 | 优先级 | 硬前置 | 预期工作量 |
|---|---|---|---|---|---|
| PR-6-A | Template semantic alignment via seizure onset propagation | Topic 1 × 3 | **P0** | PR-5 PASS ✓；需要 Epilepsiae ictal channel-wise onset rank 提取（新代码） | 2-3 周 |
| PR-6-B | Continuous template-affinity embedding (π metric) | Topic 1 方法 | **P0**（与 A 并行） | 无 | 1-2 周 |
| PR-6-C | Core source/sink identification + held-out PR-4 replay | Topic 3 × 1 | **P0** | 无，Yuquan 先行 | 2-3 周 |
| PR-6-D | Template pair physics: homologous vs antagonistic vs bidirectional | Topic 1 | **P1** | PR-6-B（用 π） | 1-2 周 |
| PR-6-E | IGE-burst × seizure coupling with cluster stratification | Topic 2 | **P1** | PR-6-A（burst 内模板占比需要） | 2-3 周 |

**统一的 guardrails**：

1. 所有 PR-6 子项在 **PR-5-A retained cohort**（main n=23 / aux n=22）上跑，与 PR-5-B 的结果口径保持连续。
2. 新引入的 `π` 连续变量要与老的 cluster label 并列报告（至少一期），避免方法迁移造成结论断层。
3. 任何"二次筛选后重跑 PR-4C"都必须用 held-out 数据定义筛选，否则结论废。
4. PR-6-A 里 `ρ_T_aligned_sz` 的 cohort 阴性结论要被接受为"Smith 2022 在 HFO / population event 层面不复现"——这是诚实的、有价值的阴性，不要包装。

**科学收益的 bet 大小**：

- 如果 PR-6-A 正向，我们就把 Smith 2022 的 μm-scale traveling wave 几何延伸到了 mm-scale HFO population events，同时解释了我们自己的 forward/reverse dichotomy——这是 topic 1 + 3 的 capstone 结论
- 如果 PR-6-C 正向，我们就给 Topic 3 "where" 问题第一个干净的答案（core source/sink 锚定 SOZ），且自动给 PR-4 做了灵敏度提升
- PR-6-D 正向（找到 H_bidirectional 的硬证据），与 PR-6-A 联动能构成最强 story
- PR-6-E 正向或 cleanly 阴性都是 Topic 2 的重要 update
- 整体负向（全部 null）——我们依然有 PR-5-B 这个已经 Bonferroni-pass 的硬结果作为主叙事

---

## 5. 对你原文几个具体问题的直接回应

> **Obs 1: MI/Kendall tau 可以理解为 rank 之间的距离，这个距离是欧式的吗？**

不是欧式。Kendall τ ∈ [-1,1] 是相似度，`d = 1-τ` 是 dissimilarity measure，满足非负、对称、`d=0 iff same rank`，但**三角不等式在一般情况下不严格满足**（这是 rank metric 的已知性质）。不过这不影响 classical MDS 的应用——MDS 不要求严格度量，只要对称半正定 double-centered matrix 有足够大的正特征值就行。用 SSIMS 思路，MDS 输出一个**近似欧式空间**，你可以直接在里面算欧式距离、定义连续 `π` 坐标。建议具体实现用 `sklearn.manifold.MDS` 或直接 `scipy.spatial.distance` + 自己做 classical MDS 的特征分解（更快、更可控）。

> **Obs 3: 少部分电极贡献了 rank 时序刻板性，lagpat 筛选的高 HFO 数量的 channel 中，很多是跨电极的**

这个诊断对 Topic 3 PR-1 的负面结论有直接含义。旧 lagPat 看到的 SOZ 优势部分是"HFO 高发 + 跨电极选择"的联合效应，PR-6-C 的 core node 二次筛选应该更严格地锁定 hub，用更小、更干净的通道集替代大而笼统的 HFO high-rate 集合。我预期 core node 集合在 SOZ 上的富集度**高于** legacy refine 通道集合——这是可量化的对比。

> **在非聚类的视野下，两种模板时序模式在事件级的时间尺度是否存在跟随或者配对？**

这正是 PR-6-D 的核心。具体答案我们只能从数据里看。但我愿意**提前下一个方向判断**：我预期 T0→T1 和 T1→T0 的 transition 会呈现弱但显著的**正 clumping**（burst 内同 template 倾向连续），而不是完美的 bidirectional 行波配对。原因是如果真是 Smith 2022 意义上的 bidirectional wave，应该在**极短的 IEI**（ms 级）内配对；我们的 population event 尺度是秒级，单次 wave 只会产出一个 population event，两个方向分属不同 events。如果数据里真看到 T0→T1 的短 IEI 富集（相对 T0→T0），那就是惊喜。

---

需要我接下来具体起草 PR-6-A 或 PR-6-C 的详细实验合同 md（对标 `pr5_template_recruitment_plan_2026-04-20.md` 的格式，包含 hypothesis / fail contract / TDD cases / 代码入口）吗？或者你想先和我进一步辩论上面的某一条分类判断？