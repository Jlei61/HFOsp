# 间期群体事件分析的方法学再审视

> 状态：历史方法学总文档
> 当前正式入口：`docs/paper_overview.md`、`docs/topic2_between_event_dynamics.md`
> 用途：保留合作者叙事、历史批判和分 topic 的方法学推理；它不再承担当前正式总论入口的角色。

> 目的：基于 Phase 4 / Phase 5 的实证结果，系统说明老论文（Wang et al., medRxiv 2024）在 Fig 1, 2, 3, S7, S13 上的若干结论需要更新；并对每一项给出"老论文做了什么 → 哪些站得住 / 哪些不站得住 → 我们补了什么分析及其方法细节 → 这些分析的可信度边界 → 还需要补什么"的完整链条。
>
> 阅读对象：合作者，假设熟悉 SEEG / HFO / 点过程分析。

---

## 0. 一句话总结

**老论文的三条核心叙事——(i) SOZ 内群体事件存在刻板传播序列，(ii) 群体事件以约 2 Hz 为内禀频率周期发放，(iii) IEI 服从幂律分布——在当前数据上只有第一条部分成立、第二条几乎完全不成立、第三条已被证伪。** 更重要的是，IEI 的相邻正相关（30/30 subjects 方向一致）暴露出一个老论文完全没有讨论的现象：**群体事件不是平稳点过程，而是被慢时间尺度的状态调制驱动的点过程**。这件事本身可能比"准周期"更接近真正的生物学。

下面三节按话题展开。

---

## 话题 1：群体事件内部时序刻板性（对应论文 Fig 1, Fig 2）

> 2026-04-11 注：这一块现在已从 `event_periodicity` 线中独立出来，作为单独主题
> "interictal group-event internal propagation" 维护。实现、结果和图请优先看
> `docs/interictal_group_event_internal_propagation.md` 与
> `results/interictal_propagation/`；本节保留方法学叙事与问题定义。

### 1.1 老论文及代码做了什么

老论文用 80–250 Hz 带通滤波 + Hilbert 包络，在每个通道上检测 HFO/spike-coupled HFO 作为 IE。相邻通道的 IE 在时间上重叠时被合并为"群体事件"。每个群体事件中，参与通道的"激活时间"由该通道在事件窗内 80–250 Hz 频谱图（spectrogram）的**时频质心** $(t_c, f_c)$ 中的 $t_c$ 给出（公式 2）。把 $t_c$ 排序就得到该事件的传播 rank 向量。

刻板性的量化用 Matching Index：

$$\text{MI} = \frac{N_{\text{consistent pairs}} - N_{\text{opposite pairs}}}{N(N-1)/2}$$

老论文计算每个事件相对于"平均传播模式"的 MI，得到一个 MI 分布；用其**中位数**作为该 subject 的刻板性指标，并通过对每个事件的 rank 做 200 次随机置换得到 null 分布，做单侧检验。结论：Yuquan 17/18、Epilepsiae 20/20 显著。

### 1.2 哪些足够 / 哪些不适当

**足够之处**

- 时频质心是 HFO 这种短时高频事件的合理时间锚点，避免了简单峰值取点的噪声敏感性。
- MI 本身在概念上等价于 Kendall τ 的一个变体（去除 ties 后两者只差归一化常数）；它确实在度量"两个 rank 序列共多少对方向一致"。
- 跨数据集复现（Yuquan + Epilepsiae）、跨被试一致性高，是这条结论里**最硬**的部分。

**不适当之处**

1. **MI 假设存在唯一的"平均传播模式"**。这个假设被论文自己的 Fig 5（patient E3 的两类反向传播）证伪。当一个 subject 实际上有多个 stereotype（mixture），用单一 mean pattern 做 MI 会**人为压低** MI 中位数；而置换 null 同样被压低，所以显著性是否仍然成立完全靠运气。论文对 E3 用了 ad hoc 聚类后再算，但**没有把这件事制度化**：对其他 subject 是否也存在隐藏 mixture 没有检查。

2. **MI 只看 sign of pair difference，不看 magnitude**。在 ~3 ms 中位数传播延迟、~1 ms 时间分辨率的尺度上，rank 序列对噪声极其敏感：两个真实差异 0.5 ms 的通道，在不同事件里可能因为噪声反向，但这两个事件的"真实传播"其实没有差别。MI 把这种 1 ms 级噪声误判为"不一致对"。

3. **置换 null 太弱**。论文的 null 是把每个事件的 rank 向量整体打乱后重算。这只破坏了"事件 i 的 rank 向量"内部结构，**保留了 marginal rank distribution**；如果某些通道在所有事件里都倾向先放电（例如因为检测阈值偏低），这种"identity bias"会被既计入 real 又计入 null，互相抵消——但同时它也会让 real MI 看起来很高，**而实际上它根本不是 propagation，是 detection ordering**。

4. **没有按 n_participating 分层**。当 n=3 时 MI 只能取 {-1, -1/3, 1/3, 1} 四个值，n=4 时 6 个值。低 participation 事件的 MI 是离散化噪声主导的；高 participation 事件才是真正能区分 stereotype 与 noise 的。论文在汇总时没区分这两类事件——这与 PR4 阶段在同步性指标上看到的"0.6 wall"是同一类问题。

5. **没有 SOZ vs non-SOZ 对照**。论文已经声称传播路径在 SOZ 内，但**没有把 non-SOZ-only 事件**作为阴性对照来证明 stereotype 不是检测/合并管线的统计副产物。如果 non-SOZ 事件也呈现同样高 MI，那 stereotype 是 pipeline artifact 而不是 SOZ 网络属性。

6. **时频质心的 1/f bias**：论文自己在 §1 末尾承认 centroid frequency 因为 1/f 谱形而被低估。同样的 bias 也会作用在 centroid time 上：当事件包含大幅度低频成分时，$t_c$ 被向低频成分的中心拉拽，而不是反映 80–250 Hz 真正的能量爆发瞬间。这会污染 rank。

### 1.3 我们补充了什么 + 方法细节

在 Periodicity Phase 5 的实验 5 里，我已经做了一个最初步的"传播立体型"分析。具体方法：

- 对每个 subject，从 `lagPatRank` 提取每个事件的通道激活 rank 向量。
- **随机抽 200 个事件**（避免大 subject 上 O(n²) 爆炸），对样本内事件两两计算 Kendall τ，要求两事件**共同参与的通道数 ≥ 3**。
- 取所有事件对的平均 τ 作为该 subject 的刻板性强度。
- 分层：所有事件 / SOZ 参与事件 / non-SOZ-only 事件，分别计算并比较。

**结果（30 subjects，Yuquan 10 + Epilepsiae 20）**

| 分组 | mean τ | 范围 |
|---|---|---|
| 全部事件 | 0.126 | 0.014 – 0.322 |
| SOZ 事件 | 0.119 | – |
| non-SOZ 事件 | 0.048 | – |

12 个 subject 同时存在 SOZ 与 non-SOZ 事件可作配对：9/12 SOZ > non-SOZ，sign test p = 0.073；单侧 Wilcoxon p = 0.039。

**关于 Kendall τ 是否合适作为补充指标**——你问到的核心问题。

τ 的好处是：(i) 与 MI 在数学上几乎一致（同样基于 concordant/discordant pair count），但 τ 有完整成熟的统计理论（已知方差、可做 Fisher z 变换）；(ii) 有 weighted τ 变体，可以根据 rank gap magnitude 加权，从而对 1 ms 级噪声不那么敏感；(iii) 容易做被试间汇总。

τ 的局限和 MI 一样：仍然假设存在单一参考序列。要真正解决 mixture 问题，需要的不是换 τ 而是换框架（见 1.5）。

### 1.4 现有证据强度

- **可以坚定地说**：在某些个体（Y1, E10, E14, E7）上，群体事件的传播 rank 在数小时到数十小时尺度上稳定。这部分论文是对的。
- **可以谨慎地说**：SOZ 参与事件比 non-SOZ-only 事件更刻板（单侧 p = 0.039），方向与论文叙事一致，但**只是探索性证据**，不能写成"definitive"。
- **不能说**：所有 17/18 + 20/20 显著的 subject 都通过了严格意义上的刻板性检验。论文的 17/18 这个数字，在 (i) 多 stereotype mixture 检查、(ii) 按 n_participating 分层、(iii) 与 detection ordering bias 的分离 后，几乎肯定会缩水。

### 1.5 仍需补充才能给出定量定性的判断

**优先级 P0**

1. **Mixture 检测**：对每个 subject 的事件 rank 矩阵做谱聚类（spectral clustering on pairwise τ matrix）或 finite mixture model，先回答"是否存在唯一 stereotype"。E3 不是孤例的可能性需要排除。一个简单的检验：如果 rank pattern 的 silhouette score 在 k=2 显著高于 k=1，那么 k=1 的 mean pattern 不再有意义，老论文的 MI 中位数需要按 cluster 重算。

2. **Identity bias vs sequence**：把每个事件的 rank 向量做"中心化" rank（每个通道减去其在所有事件中的平均 rank），再算 τ。中心化前后 τ 的差异告诉你多少 stereotype 来自"propagation"，多少来自"某些通道总是先放电（detection ordering）"。后者不是 propagation，是 channel-level bias。

3. **n_participating 分层 + 混合效应模型**：以事件为单位，τ 或 pairwise concordance 作为响应变量，n_participating 作为协变量，SOZ 状态作为 fixed effect，subject 作为 random effect。报告 SOZ 系数的 CI 而不只是 sign test。

**优先级 P1**

4. **重抽样稳健性**：现在 τ 是 200-event 子采样的估计量，对长记录欠采样。需要 multi-seed bootstrap 报告 CI；对 long-record subjects 做分层抽样而不是简单随机。

5. **时间分辨率敏感性**：人为把 centroid time 加上 0.5 ms / 1 ms / 2 ms 高斯抖动后重算 τ，看 τ 衰减曲线。如果 1 ms 抖动就让 τ 接近 0，说明老论文宣称的"propagation pattern"在 SEEG 检测的物理分辨率上**没有真信号**。

6. **Detection ordering 反向控制**：在每个事件中，**人为打乱 80–250 Hz 包络阈值的先后顺序**（保持时频质心位置），再走全部 pipeline，看 τ 是否还显著。这是对 pipeline artifact 的真正阴性对照。

**优先级 P2**

7. **空间结构可视化**：把 stereotype 投影到电极三维坐标上，画"传播波前"。论文 Supplementary Fig 6 已经做了一点，但只在 group level；个体 level 的传播波前可视化能直接告诉合作者"这看起来像一个传播波"还是"看起来像随机噪声"。

8. **SOZ-internal 子集分析**：在临床 SOZ contacts 之间重做 τ。如果 SOZ-internal τ 比"all SOZ-participating events" τ 更高，说明 stereotype 真正集中在 SOZ 子图。

---

## 话题 2：群体事件是否真有"周期性"（对应 Fig 3, Fig S7, Fig S13）

这一节是本次最重要的、需要让合作者接受立场更新的部分。

### 2.1 老论文及代码做了什么

事件时间戳被打成 100 Hz 的二值脉冲序列（事件窗内 = 1，否则 = 0），然后用 `scipy.signal.welch(arr, fs=100, nperseg=50000)` 计算 PSD（即 500 s 窗口）。FOOOF（specparam）拟合 PSD 在 0.5–10 Hz 范围内的 (1/f aperiodic + 高斯峰) 分解，提取周期峰频率。多数被试在 ~2 Hz 出现峰，被报告为"interictal events 的内禀周期"。

显著性检验（论文 Fig S13）：把 FOOOF 拟合后的 aperiodic component 减掉，得到去趋势的 PSD 残差；把"峰位 PSD 值"与"该频段所有 bin 的去趋势 PSD"做 **one-sample t-test**，报告 z 和 p。

IEI 分布（论文 Fig S7）：log-log scale 上做线性回归拟合 power-law 指数 γ，报告 R² 作为拟合质量。

### 2.2 哪些足够 / 哪些不适当

**足够之处**

- 用 binary pulse train + Welch + FOOOF 提取候选周期峰是合理起点。
- 跨被试一致地在 1.5–2.5 Hz 看到峰，说明这个现象**确实存在**——只是它的"是不是真振荡"另说。

**严重不适当之处（按重要性排序）**

1. **Fig S13 的 t-test 是无效的**。它把 FOOOF aperiodic-detrended PSD 当成 i.i.d. 噪声，对比"峰 bin"和"其他 bin"的均值。但 (i) Welch 相邻 bin 高度相关（频率分辨率有限）；(ii) FOOOF 残差不是 i.i.d.，残差结构本身有 autocorrelation；(iii) 用同一份数据既选峰又检验，已经 double-dipping。这个 t-test 的 z=-256 不能按字面读，**它只反映了"峰比 baseline 大"，而不能拒绝任何零假设**。

2. **没有 refractory null**：这是核心问题。一个**带不应期的更新过程**（refractory renewal process），即使没有任何振荡器机制，其事件率序列的 PSD 也**必然**在 ~ 1/(平均不应期) 附近出现峰。这是 spike train 分析里的经典现象，参见 Bair, Koch, Newsome (1994), Brunel (2000)。论文完全没考虑这个 null model，等于没有做"是不是振荡"的检验。

3. **Power-law 用 log-log 线性回归是错的**。Clauset, Shalizi & Newman (2009) 已经在文献里反复强调：(i) ML 估计才是无偏的；(ii) 必须用 Kolmogorov-Smirnov 距离做 goodness-of-fit；(iii) 必须**与替代分布（lognormal, exponential, stretched exponential）做 LLR 比较**才能宣称"power-law"。光看 log-log R² 不能区分 power-law 和 lognormal——lognormal 在中段也是近似线性的。

4. **没有控制 packing window**：群体事件由 packing 窗口（默认 500 ms）合并产生，这意味着任何两个相邻群体事件的 IEI **下界硬性等于 W**。这立即在 IEI 分布中创造一个 dead zone，并在 PSD 中创造一个 ~1/W 附近的峰。论文从未论证这个峰不是 W 的产物。

5. **per-channel 与 group 峰频不一致但被一并报告**：论文 Fig 3c 把两者并列，没有解释为什么 per-channel 峰频中位数 ~3.5 Hz 而 group ~1.6–2.1 Hz。如果两者反映的是同一个生理振荡，峰频应该一致；不一致意味着至少其一是 pipeline 产物。

### 2.3 我们补充了什么 + 方法细节

#### 2.3.1 Gamma renewal surrogate（核心）

**目的**：检验"PSD 峰是否完全可以由不应期 + 平稳率解释"。

**方法**：
1. 估计真实 IEI 序列的均值 $\mu$、标准差 $\sigma$、最小值 $\tau_r$（作为不应期估计）。
2. 拟合一个 **shifted gamma distribution**：$\text{IEI} \sim \tau_r + \text{Gamma}(k, \theta)$，参数从矩匹配得到（$k = (\mu - \tau_r)^2 / \sigma^2$, $\theta = \sigma^2 / (\mu - \tau_r)$）。
3. 从该分布生成 200 套 surrogate IEI 序列，长度等于真实序列。
4. 对每套 surrogate 重建 binary pulse train，跑同一份 Welch + FOOOF pipeline，记录峰功率。
5. surrogate p-value = 真实峰功率 ≤ surrogate 峰功率的比例。

**含义**：Gamma renewal process 是"无任何振荡器、纯不应期 + 随机率"的标准模型。如果真实数据的峰**不显著大于** surrogate，那么峰可以被这个零模型完全解释，根本不需要"内禀振荡器"。

**结果**：30 个 group-level subjects 中，**15 个 surrogate p ≥ 0.05**（峰被不应期完全解释）。更尖锐的是，gamma surrogate 的 mean peak power ≈ 0.77，而真实数据 ≈ 0.17——**真实数据的峰比"纯不应期 null"产生的峰还要弱**。这是一个非常强的反向证据。

#### 2.3.2 ISI shuffle surrogate

**目的**：检验"峰是否依赖于事件的时间顺序"。

**方法**：把真实 IEI 序列**随机打乱顺序**（保持 marginal IEI 分布完全不变），重建脉冲序列，重算 PSD。如果峰仍然存在，说明峰的来源是 IEI 分布的形状本身，而不是事件之间的时间依赖关系（比如振荡器锁相）。

**结果**：30 个里 2 个在 shuffle 后峰仍然显著保留（dist-artifact），13 个 shuffle 后峰消失（说明序列结构有作用），但其中绝大多数同时被 Gamma surrogate 解释为不应期产物。**两个 surrogate 都通过的"genuine"只有 1/30**。

#### 2.3.3 PackWinLen 参数扫描（实验 1）

**目的**：检验峰频是否被 packing 窗口大小直接控制。

**方法**：对 10 个 Yuquan subjects（有 `_gpu.npz` 原始 detections），用 W ∈ {100, 150, 200, 300, 400, 500, 600, 800, 1000} ms 重新做 packing → 重新提取群体事件 → 重新算 PSD 和 specparam 峰。

**结果**：$f_\text{peak}$ **不**跟 $1/W$ 走（chengshuai 在 W=100ms 给出 $f_\text{peak}$ ≈ 2.10 Hz 而非 10 Hz）；多数 W 下 specparam 检不出峰；**这说明简单的"窗口栅格量化"不是峰的直接来源**。但要注意：iei_min 确实跟 W 走（验证了硬性死区是 W 强加的），所以 packing 仍然在塑造 dead-time，只不过不直接控制峰频。

#### 2.3.4 Centroid bypass（实验 2）

**目的**：在窗口内部更换时间锚点，看峰频是否变化。

**方法**：对 30 subjects，用三种事件时间戳重算 PSD：(a) `packedTimes[:, 0]`（窗口起点）；(b) `mean(lagPatRaw)`（参与通道的时频质心均值）；(c) `min(lagPatRaw)`（最早通道的质心，"点火时间"）。

**结果**：13/15 有峰的 subjects 三种方法峰频差 < 0.1 Hz。**结论降级版**：在当前 legacy `lagPatRaw → 绝对时间`映射框架内，窗口内锚点选择对峰频不敏感。**注意**：这不等于"已经彻底脱离 packing 影响"——bypass 仍然依附于已经 packed 的窗口，没有从原始 envelope 重新独立重建事件时间。`litengsheng` 在三种方法间的差异达 1.235 Hz，说明 outlier 存在。

#### 2.3.5 IEI hazard function（实验 3）

**目的**：用经典的 survival analysis 工具直接看 dead-time 结构。

**方法**：用 KDE 估计 IEI 的 pdf $f(t)$ 和 CDF $F(t)$，计算 $H(t) = f(t) / (1 - F(t))$。$H(t)$ 的物理含义：已经等待 $t$ 秒后，下一瞬间事件发生的条件概率密度。

**结果**：所有 subjects 的 $H(t)$ 在 $t < \text{min IEI}$ 处接近 0（强制死区），之后急剧上升再缓慢衰减——**这是经典的 refractory renewal 形态**。但当前 KDE 实现不能拿来做严格统计推断（见 §2.4 警告）。

#### 2.3.6 Power-law vs lognormal（MLE + LLR）

**方法**：用 `powerlaw` 包（基于 Clauset 方法）做 MLE 拟合 power-law 与 lognormal，再用 Vuong's test（log-likelihood ratio）比较两者。

**结果**：**30/30 subjects 的 LLR R 都为负（lognormal 显著优于 power-law），p < 0.05**。`R` 范围 -103.3 到 -3.2。这把论文 Fig S7 的"幂律"叙事彻底推翻。

#### 2.3.7 解析 Renewal PSD Overlay（实验 6A — PR-1，2026-04-07 完成）

**目的**：把 shifted-gamma renewal process 的理论 PSD **直接画在**经验 PSD 上，用零自由参数的解析预测来证明"~2 Hz 峰是不应期的必然产物"。

**方法**：

1. 对每个 subject，从已有群体事件计算 IEI 序列。
2. 估计 shifted-gamma 参数：$\tau_r$ = percentile(IEI, 2)，$k = (\mu - \tau_r)^2 / \sigma^2$，$\theta = \sigma^2 / (\mu - \tau_r)$，$\lambda = 1/\mu$。
3. 解析特征函数 $\varphi(\omega) = e^{i\omega\tau_r} (1 - i\omega\theta)^{-k}$，PSD 公式 $S(f) = \lambda \cdot \text{Re}\left[\frac{1+\varphi}{1-\varphi}\right]$。
4. 重新计算 delta-train（脉冲模式）Welch PSD 作为经验对照（避免 rectangle 模式的 sinc² 效应——尽管实际事件时长 ~0.058 s 使 sinc² 修正 < 5%，科学上可忽略）。
5. 两条曲线在 0.5–8 Hz 内做 **min-max 归一化**后叠图。经验 PSD 做 Savitzky-Golay 平滑（window=101, order=3）以暴露 bump 结构。
6. 在 0.5–5 Hz 内分别取 argmax，作为 empirical peak freq / analytic peak freq，报告 |Δf|。

**关于 shifted-gamma vs lognormal 的建模选择**：Phase 1/5 已经证明 IEI 是 lognormal 而非 power-law（30/30）。此处仍用 shifted-gamma 是因为解析 PSD 需要闭合形式的特征函数；lognormal 没有。shifted-gamma 在此仅作为**局部近似**以预测 1–5 Hz 频段的 PSD 峰位，不是对 IEI 全分布的声明。

**结果**：

30 个 subject 全量运行（Yuquan 10 + Epilepsiae 20）。对有 specparam 峰的 21 个 subject：

| 判据 | 覆盖 |
|---|---|
| Gamma surrogate p ≥ 0.05（Monte Carlo 路径） | 15/21 |
| 解析峰频 \|Δf\| < 1 Hz（解析路径） | 16/21 |
| **两者至少一个成立** | **19/21 (90%)** |
| 两者都不成立 | 2/21 (1084, 1096) |

- |Δf| 中位数 = 0.64 Hz；9/21 在 0.5 Hz 以内，16/21 在 1 Hz 以内。
- Cohort scatter (analytic vs empirical peak freq)：r = 0.34, p = 0.13。
- 解析曲线呈现出清晰的**谐波梳齿结构**（fundamental + harmonics），这是 renewal 过程的数学必然；经验 PSD 中谐波被 τ_r 变异性 wash out。这本身是一条有信息量的观察：**个体内 τ_r 不是常数而是有分布的，进一步支持"非平稳"解读**。

**逃逸的 2 个 subject 的特征**：

- **1084**：mean/median ratio = 16.1（极端 bursty），n = 11149。IEI 分布严重右偏。
- **1096**：iei_min = 0.064（所有 subject 中最短 dead-time），n = 223394（最多事件），mean/median = 3.09。

两者共同特征：**高度非平稳事件率**。gamma surrogate（假设平稳率）会低估真实 PSD 峰功率 → p 偏低 → 误判为"峰是真的"；解析公式（假设单一 τ_r + 平稳率）也会偏移峰频。

**关键解读**：这不意味着 1084 和 1096 有内禀振荡器。它意味着**平稳 renewal null 模型对这两个 subject 不够用**——需要非平稳 renewal（即 PR-2 的 detrending 分析）才能完成解释。换言之，Layer 3 不是"另一种物理机制"，而是同一个 refractory renewal 机制在更强的慢调制下的表现。

**结论**：解析 PSD overlay 不是当初设想的"杀手锏图"（cohort r = 0.34 不够强），但作为 gamma surrogate 的**理论补充**是充分的。两条路径互补覆盖 90% 的有峰 subject。

#### 2.3.8 SOZ vs non-SOZ Dead-Time 分层（实验 6B — PR-1，2026-04-07 完成）

**目的**：比较 SOZ 参与事件与 non-SOZ-only 事件的不应期特征，回答"SOZ 兴奋性是否更高"。

**方法**：

对每个 subject，按 `eventsBool` 中是否有 SOZ 通道参与将群体事件分为两组，分别计算 IEI 统计量（iei_min, iei_p02, iei_median, iei_mean）。要求两组事件数都 ≥ 50 才纳入配对比较。

**SOZ 定义来源**：Epilepsiae 用 `electrode.focus_rel == 'i'` → `results/epilepsiae_soz_core_channels.json`；Yuquan 用 `p16_subs_info.py` 手工标注 → `results/yuquan_soz_core_channels.json`。

**结果**：

只有 **8/30** subject 可以形成有效配对。原因分解：

| 排除原因 | 数量 |
|---|---|
| SOZ JSON 中无该 subject 定义 | 5 |
| non-SOZ 群体事件 < 50 | 17 |
| 有效配对 | 8 |

17/30 subject 的群体事件**几乎全部由 SOZ 通道参与**（non-SOZ 事件为 0 或个位数）。这本身是一条重要发现：**群体事件是 SOZ 驱动的现象**，纯 non-SOZ 群体事件极其稀少。

在 8 个有效配对中：

| 指标 | Wilcoxon p | 方向 |
|---|---|---|
| IEI 2nd percentile（dead-time proxy） | **0.008** | SOZ < non-SOZ |
| IEI median | **0.016** | SOZ < non-SOZ |

**SOZ 参与事件的 dead-time 和 IEI 中位数都显著短于 non-SOZ 事件**。方向与"SOZ 兴奋性更高 / 恢复更快"的 FHN/HR 框架一致。

**可信度边界**：n = 8 对，且多数配对中两组事件数极度不平衡（如 958：165484 SOZ vs 93 non-SOZ），结论为**探索性**。要扩大样本量，需要改用连续变量（SOZ 参与度比例）或降到 per-channel 层面比较。

### 2.4 这些分析的可信度边界（更新版）

| 结论 | 强度 | 边界 |
|---|---|---|
| Gamma surrogate 完全可以解释 15/21 的峰 | **强** | 仅当 shifted gamma 是合理的 refractory null 时 |
| IEI 是 lognormal 而非 power-law（30/30） | **强** | MLE + LLR 是该领域标准方法 |
| 解析 PSD + gamma surrogate 互补覆盖 19/21 (90%) | **中-强** | 解析部分用 shifted-gamma 近似，峰位对齐 < 1 Hz 但形状相关弱 (r=0.34) |
| 逃逸的 2/21 归因于非平稳调制 → **已验证** | **强** | PR-2.5 Exp 7F：去趋势后峰完全消失，21/21 全覆盖 |
| 简单 $1/W$ 量化不是峰频来源（PackWinLen） | **中-强** | 仅 Yuquan |
| Centroid bypass 不改变峰频 | **中** | 仍依附于 legacy 映射 |
| Hazard function 与 refractory renewal 一致 | **定性** | KDE-based，不做参数推断 |
| SOZ dead-time < non-SOZ dead-time | **探索性** | n=8 配对，组间事件数严重不平衡 |
| 群体事件几乎全部由 SOZ 参与 (22/30) | **描述性，强** | 直接从 eventsBool 计数，无统计推断需求 |

### 2.5 仍需补充（更新版）

1. ~~**解析 PSD 写出来**~~ → **已完成**（实验 6A）。不是"杀手锏"但与 gamma surrogate 互补达 90% 覆盖。

2. **真正独立的事件时间重建**：从原始 80–250 Hz envelope 直接定义事件时间（比如 envelope 局部最大值的时间），完全不走 packing/lagPatRaw 链路。如果这条独立链路下峰频还在 ~2 Hz，centroid bypass 的结论才能升级到"已彻底排除 packing"。

3. **替换 Fig S13 的 t-test**：将来如果还要发表周期性图，必须用 Gamma surrogate p-value 替代 t-test。

4. ~~**SOZ vs non-SOZ 分层 hazard**~~ → **已完成**（实验 6B）。SOZ dead-time < non-SOZ，p=0.008，但 n=8 且探索性。**下一步**：用 SOZ 参与度连续变量（soz_fraction）或 per-channel 层面扩大样本量。

5. ~~**对逃逸的 2/21 (1084, 1096) 做非平稳控制**~~ → **已完成**（PR-2.5 Exp 7F）。去趋势后 1084 和 1096 的峰**完全消失**。Layer 3 缺口已关闭。

6. **对 huanghanwen 这个唯一"genuine"做敏感性分析**：把不同 surrogate 参数（gamma shape、shift、shuffle 类型）扫一遍，看它是否真的稳健。n=484 的样本量本来就在边缘。

### 2.6 动力学含义：Kuramoto 还是 FHN/HR？——这是叙事更新的核心

合作者目前接受老论文的 Kuramoto / Hopfield-Kuramoto / adaptive Kuramoto 框架，是因为他们相信 ~2 Hz 是"内禀振荡频率"。**这个相信的基础已经被上面的分析抽掉了**。需要把动力学叙事更新到正确的模型类。

**Kuramoto 模型的核心假设**：每个节点是一个有内禀角频率 $\omega_i$ 的极限环振荡器，相位 $\theta_i$ 是 $S^1$ 上的变量，没有"静息态 vs 激活态"的区分，没有不应期。Kuramoto 节点不会"等待几秒再放电"——它一直在以 $\omega_i$ 转。

**FHN（FitzHugh-Nagumo）/ Hindmarsh-Rose / Morris-Lecar / QIF 等可激发模型**：有快变量（膜电位/活性变量）和慢恢复变量（refractory variable），系统在静息状态可以无限期等待，受到扰动后产生一次脉冲，然后**强制进入恢复期**。这正是真实神经元和我们看到的 IEI 数据的形态。

**两类模型的关键鉴别特征**：

| 特征 | Kuramoto | FHN/HR/QIF |
|---|---|---|
| 静息态存在 | ✗ | ✓ |
| 不应期 | ✗ | ✓ |
| IEI 存在硬性下界 | ✗ | ✓ |
| 自发产生宽 IEI 分布 | ✗（需外加噪声） | ✓ |
| Hazard function dead zone | ✗ | ✓ |
| Serial correlation 来源 | 锁相（短-长交替） | 慢变量调制（长-长 / 短-短聚集） |

我们在数据里看到的：**dead zone, lognormal IEI, hazard 上有明显死区, serial correlation 全部为正**——**每一项都指向 FHN/HR 类，没有一项指向 Kuramoto 类**。

**关于"Kuramoto 能不能加不应期"**——你问得很好，这是论文叙事可不可救的关键。

- 标准 Kuramoto 不能。$\theta \in S^1$ 没有恢复变量。
- **theta neuron model**（Ermentrout-Kopell, 1986）是 SNIC bifurcation 附近的 Type-1 兴奋性神经元的相位简化，形式上确实写在 $S^1$ 上，但它**有静息点**（在某个 $\theta^*$ 处停下不动），需要扰动才能放出一个脉冲。它在数学上和 QIF 等价，**已经不是 Kuramoto**——它更像一个穿在圆上的兴奋性单元。
- **Mirollo-Strogatz pulse-coupled oscillators** 引入了脉冲耦合后的延迟和重置，可以产生 dead time，但这个模型也已经远离了 Kuramoto 的弱耦合相位假设。
- **FHN 在 Hopf 分岔附近的 phase reduction** 可以得到 Kuramoto 形式的相位方程——但这个简化只在**极限环已经存在**的参数区间（Hopf 上方）才成立；下方是兴奋性的（无极限环），phase reduction 不适用。论文数据里 IEI 分布形态明显属于"Hopf 下方"的兴奋性区域。

所以，**老论文用 Kuramoto 是模型类选错了**。可以救的版本是：保留"网络耦合 + Hebbian / 对称塑性"的核心思想，但把节点动力学换成 FHN / HR / QIF / theta neuron。这样：
- (i) 不应期自然存在，不需要外加约束；
- (ii) IEI 分布的 lognormal/重尾形态可以由"OU 噪声驱动的 SNIC 兴奋性单元"自然产生（这是文献里熟知的结果，参见 Lindner 等的工作）；
- (iii) Hopfield-style 的 Hebbian 编码可以在 FHN 网络上做（已有文献，比如 Aoyagi 和后续工作的某些扩展）；
- (iv) "interictal → ictal 的相变"叙事可以保留，但相变机制变成"兴奋性单元从亚阈值到阈上的 SNIC bifurcation 集体跨越"，而不是 Kuramoto 的 splay→sync 相变。

**对生理学的含义**：你问到 SOZ 内外不应期是否有差别——这是一个很好的待回答的实证问题。如果 SOZ 内 dead-time 显著更短（更高 hazard），说明 SOZ 的兴奋性更高、恢复更快；如果 dead-time 一致而事件率显著不同，说明 SOZ 不是"恢复更快"而是"更容易被触发"（噪声 driven 而非 refractory bound）。这两种情况对应的微观机制完全不同（GABAergic 抑制损失 vs 谷氨酸能驱动增强）。这是一个可以在两个数据集上立刻做的实验。

---

## 话题 3：IEI 相邻正相关——可能是本次最有价值的新发现

### 3.1 发现概要

对 30 个 subjects 的群体事件 IEI 序列，计算 $\text{corr}(\log \text{IEI}_n, \log \text{IEI}_{n+1})$：

- **30/30 方向为正**
- log-IEI Pearson r 中位数 ≈ 0.31，IQR ≈ [0.25, 0.38]，范围 0.12 – 0.51
- subject-level sign test：p ≈ 9.3 × 10⁻¹⁰

需要注意的方法学口径修正（在审阅报告里已经记录）：每个 subject 内部的 Pearson p 值不能按独立样本读，因为相邻 IEI 对之间存在依赖。**正确的报告方式是 subject-level sign test，而不是 within-subject p < 0.001**。但**方向一致性这个事实本身**是非常硬的。

### 3.2 这个事实排除什么、支持什么

**排除**：稳定振荡器驱动。如果存在一个 ~2 Hz 的内禀振荡器，相邻 IEI 应该呈现负相关（短-长交替，因为相位锁定）或零相关（独立振荡周期）。**正相关与振荡器假设直接矛盾**。

**支持**：**慢时间尺度的事件率调制**。当某个调制变量 $r(t)$ 缓慢变化时，在高 $r$ 的窗口里 IEI 整体偏短（连续几个短 IEI），在低 $r$ 的窗口里 IEI 整体偏长。这就产生 long-long / short-short 的聚集，即正 serial correlation。这是一个**非平稳点过程的特征签名**。

数学上，这等价于：把事件过程模型化为 inhomogeneous Poisson with rate $r(t)$，正 serial correlation 的强度直接对应 $r(t)$ 的自相关时间尺度——慢调制 → 强正相关；快调制（高频白噪声率）→ 弱正相关。

### 3.3 你提的四个引发思考的问题，逐条回答

#### Q1：除了 IEI 还有什么被调制了？为什么 IEI 表现得最明显？

候选被调制量（按可测性排序）：

- **n_participating per event**（每事件参与通道数）——可以直接从已有数据计算 serial correlation，应该也呈正相关（如果调制是系统级的"网络兴奋性"）
- **传播 stereotype 强度 / pairwise τ**——是否在高发生率窗口里事件传播更刻板？
- **dead-time 长度本身**——是否随时间漂移？
- **事件 amplitude / total energy**——系统兴奋性高时单次释放能量更大？
- **同步性指标（legacy/phase/span）**——你们 PR4 已经有 event-level 数据，可以直接对 sync metric 做同样的 serial correlation 分析

IEI 表现最明显是因为它是所有这些调制效应在时间维度上的累积投影——任何让"下一个事件什么时候来"变化的因素都会折射进 IEI。

#### Q2：什么产生的调制？

候选源（按时间尺度从快到慢）：

- **几十秒到几分钟**：呼吸、心率、皮层 up/down state、微觉醒
- **几十分钟到几小时**：睡眠阶段（NREM N1/N2/N3 vs REM）、警觉度
- **数小时到一天**：昼夜节律、药物血药浓度（特别是 levetiracetam, lamotrigine 等）、手术后炎症反应
- **数天**：multi-day cycles（参见 Karoly et al. 2018, Baud et al. 2018，论文也引用了）
- **状态依赖**：发作前 / 发作后状态、interictal-to-ictal transition 的 critical slowing

要区分这些时间尺度，做 corr(IEI[n], IEI[n+k]) for k = 1, 2, 5, 10, 50, 200 这条衰减曲线。衰减时间常数直接告诉你主导调制的时间尺度。

#### Q3：哪里被调制？SOZ 内外有差别吗？

这是**最可操作、最值得立刻做**的实验。

具体做法：
1. 每个 subject，分别对 SOZ-only 通道事件序列、non-SOZ-only 通道事件序列、所有通道事件序列计算 serial correlation。
2. 比较三者：
   - 如果只有 SOZ 通道呈正相关 → 调制作用在 SOZ 网络核心
   - 如果三者都呈正相关 → 调制是全局的（最可能是兴奋性增益，比如觉醒/睡眠）
   - 如果 non-SOZ 反而更强 → 调制可能是非癫痫性的生理调制（比如背景脑活动），而 SOZ 因为已经被自身病理"clamp"住反而对全局调制不敏感
3. 用混合效应模型形式化检验。

**这个实验本身可能比所有"周期性"分析加在一起更有诊断价值**，因为它直接问"病灶网络是不是有自己独立的状态"。

#### Q4：动力学图景是什么？

**最简洁的候选模型**：FHN / HR 类兴奋性单元 + OU 噪声 + 一个慢变量 $u(t)$（slow drift）调制兴奋性参数。

具体形式（示意）：

$$\dot{v} = f(v, w) + I_0 + u(t) + \xi(t)$$
$$\dot{w} = \epsilon (v - \gamma w)$$
$$\dot{u} = -u/\tau_u + \eta(t)$$

其中 $u(t)$ 是慢 OU 过程（$\tau_u$ ~ 数十分钟到数小时），$\xi$ 是快噪声。这个系统会自然产生：
- 不应期（来自 $w$ 的恢复时间）
- lognormal-like IEI（OU 噪声驱动 SNIC 的标准结果）
- 正 serial correlation（来自 $u$ 的慢漂移）
- 多时间尺度（fast spiking + slow modulation）

**和 Epileptor 的关系**：Epileptor (Jirsa et al. 2014) 已经是 fast-slow + permittivity variable 的框架，从动力学类型上属于同一家族（兴奋性单元 + 慢变量），它的"permittivity"对应这里的 $u$。Epileptor 的目标是建模 ictal transitions（permittivity 越过临界值）；我们这里是建模 interictal events 的随机时序——两者完全可以拼起来：低 $u$ → 间期事件被激活，$u$ 跨过阈值 → 进入 ictal。

**这给老论文叙事的更新建议**：

老论文的 Hopfield-Kuramoto 部分（编码刻板传播）的核心数学**不需要丢**——Hebbian-like 对称耦合可以在 FHN 网络里同样实现，已经有文献做过。但概念叙事要从"oscillator network with phase entrainment"转到"excitable network with refractory + slow modulation, structurally constrained by Hebbian-shaped connectivity"。Adaptive Kuramoto 那部分的"first-order phase transition"叙事也可以保留，但相变机制变成 SNIC（saddle-node on invariant cycle）类的兴奋性集体激活，而不是 Kuramoto 的 splay→sync。

### 3.4 PR-2 已完成的深度分析（2026-04-08）

上述 P0 第 1–3 条和 P1 第 5 条已在 PR-2 中完成（实验 7，30 subjects 全量）。

#### 3.4.1 Lag-k 衰减曲线（P0.1 → 完成）

r(log IEI[n], log IEI[n+k])，k=1..100，在 block 内 IEI 序列上 pool pairs。

- **30/30 正方向**（与 exp4 一致），lag-1 中位 0.299（0.117–0.506）
- **半衰期中位 107.5 秒 ≈ 1.8 分钟**（24/30 有限值，范围 3.5s–552.6s）
- **6/30 在 k=100 内未衰减到半**：持续慢调制主导（chengshuai, huangwanling, 384, 922, 253, 1146）
- 三类模式：快衰减（~10）、中速衰减（~14）、不衰减（~6）

#### 3.4.2 去趋势检验（P0.3 → 完成）

物理时间 ±300s 滑动窗口中位数作为局部基线，残差 = log(IEI) − log(baseline)。

- **去趋势分数中位 0.720** → ~72% 的正相关来自 > 10 分钟的慢率漂移
- **27/30 去趋势后仍为正**（残差 lag-1 r 中位 0.081）→ 存在 ~28% 的短程网络依赖
- 3/30 变负（442: −0.021, 590: −0.047, 620: −0.011）→ 这些 subject 的正相关完全是慢漂移
- 去趋势分数 > 0.8 的 10/30 subject 几乎全由慢漂移主导

#### 3.4.3 Block 内分析（P0.2 → 完成）

将事件严格按 block 边界切分（Yuquan 2h, Epilepsiae 1h），各 block 独立算 lag-1 后 pool。

- **Within-block pooled lag-1 r：中位 0.299，30/30 正**
- 结论：跨 block 污染假说被排除。Block 内序列本身有稳固的正序列相关。
- 真实的 block 间隔结构：Yuquan 所有 gap=0（12×2h 完美连续 24h）；Epilepsiae 91.7% 的 gap ∈ [0,2]s（本质连续），4.2% 负 gap（overlap/block_dur 过估），2.1% 真实 >1h 中断

#### 3.4.4 SOZ vs non-SOZ 分层（P1.5 → 完成）

- **有效配对**：9/30（多数 subject non-SOZ 事件 < 50）
- SOZ lag-1 r 中位 0.302，nonSOZ 中位 0.132
- SOZ > nonSOZ：7/9，Wilcoxon p = 0.055
- 方向暗示 SOZ 网络有自主记忆效应，但 n=9 只是边缘趋势

#### 3.4.5 PR-2 综合解读

1. **慢速率漂移是序列相关的主成分（~72%）**，最可能来自 sleep/wake + circadian 全局生理调制
2. **去趋势后残差（~28%）仍为正**，说明存在短程网络依赖（facilitation/depression），不完全是全局调制
3. **调制时间尺度中位数 ≈ 1.8 分钟**，但当前 600s 单一窗口不能区分具体频段
4. **SOZ 倾向于更强的序列相关**，但 n 太小不能定论
5. **跨 block 污染被完全排除**

#### 3.4.6 PR-2 的局限 → PR-2.5 动机（已解决）

PR-2 回答了"有多少 / 方向如何"，但**没有回答"慢调制集中在什么频段"**：

- 600s 去趋势是一刀切。那 72% 是集中在 20 分钟还是 12 小时？
- 短程残差的 28% 是 1 秒还是 5 分钟？
- 除了 IEI，慢调制是否同时改变了 n_participating（参与通道数）？
- Day vs night 是否是主导分割点？

→ **PR-2.5 已全部完成**，见 §3.4.7。

#### 3.4.7 PR-2.5 多尺度调制解剖（2026-04-08 完成）

**Exp 7B：多尺度去趋势曲线**

在 6 种窗口 W ∈ {60, 180, 600, 1800, 3600, 7200}s 上计算 detrend_fraction(W)，用差分 Δ_frac 定位释放最多相关性的频段。

- Δ_frac 近似平坦（0.080–0.147），峰值中位数在 ~329s（≈5.5 min），但不尖锐
- **结论**：慢调制是宽频段 1/f 型，不能归因于单一生理过程

**Exp 7C：n_participating Spearman 自相关**

对每事件参与通道数（离散整数）用 Spearman 秩相关计算 lag-k 衰减，与 IEI Pearson 衰减做互相关。

- IEI–n_participating 衰减曲线互相关中位数 = **0.742**，18/30 subject > 0.7
- **结论**：**证实单一全局状态变量假说**。IEI 和 n_participating 由同一个缓慢变化的全局兴奋性 S(t) 驱动

**Exp 7D：日夜分层去趋势**

按本地时间（Yuquan: Asia/Shanghai, Epilepsiae: Europe/Berlin）将事件分为 day/night，各段内独立做 600s 去趋势。

- Day 去趋势后 lag-1 r 中位 = 0.094，Night = 0.086
- 28/30 subject 在两段中均为正
- Wilcoxon day vs night p = 0.088（无显著差异）
- **结论**：28% 短程依赖不是日夜边界伪影，是真实的网络级短程记忆

**Exp 7E：Block 合并灵敏度**

相邻 block（gap ≤ 5s）合并后重算半衰期。结果与分块一致，block 边界不隐藏额外的超慢调制。

**Exp 7F：逃逸 Subject 回填（决定性）**

对 PR-1 逃逸的 1084 和 1096，用 600s 去趋势 IEI → 重建脉冲序列 → 重算 PSD + specparam + gamma surrogate。

| Subject | 原始峰频 | 去趋势后峰频 |
|---------|----------|-------------|
| 1084 | 3.34 Hz | **0.00 Hz（无峰）** |
| 1096 | 2.47 Hz | **0.00 Hz（无峰）** |

**结论**：谱峰在去趋势后完全消失 → **Layer 3 缺口关闭**。21/21 有 specparam 峰的 subject 全部被 refractory renewal + 慢率调制解释。

#### 3.4.8 PR-2.6 连续长时程分析（2026-04-09 完成）

PR-2.5 仍有两个语义缺口：

- “慢调制”主要是通过事件索引 lag-k 和 60s–7200s 去趋势窗间接推断出来的，还没有真正落到**连续 24h 时间轴**
- day/night 分层虽然方向上足够，但本质还是**标签池化**，还不能严谨地写成“连续白天段 / 连续夜晚段内部仍有短程相关”

PR-2.6 的做法很克制：不引入新的强模型，只把分析主轴改成**连续时间**。

**连续时间率过程重建**

- 先将相邻 block（gap ≤ 5s）合并为连续观测段
- 在真实时间轴上构建 5 分钟 bin 的 rate trace
- 再做 0.5h / 1h / 2h / 4h / 8h 的平滑摘要和连续时间自相关

覆盖情况：

| 数据集 | N | 最长连续段中位数 | 总观测时长中位数 | near-24h continuous |
|---|---:|---:|---:|---:|
| Yuquan | 10 | **24.0h** | **24.0h** | **10/10** |
| Epilepsiae | 20 | **75.1h** | **158.4h** | **20/20** |

**连续时间多小时调制**

| 数据集 | 0.5h fluct | 8h fluct | 0.5h rate acorr | 8h rate acorr |
|---|---:|---:|---:|---:|
| Yuquan | 1.067 | 0.442 | 0.251 | -0.058 |
| Epilepsiae | 2.243 | 1.322 | 0.493 | 0.108 |

解读：

- 宏观事件率（5 分钟 bin rate trace）的慢漂移在**真实连续时间轴**上确实延伸到多小时，而不只是去趋势窗里的数学产物。注意这是 binned rate 的时间自相关，不等同于 IEI 事件级 serial correlation 的直接延伸
- Epilepsiae 的多小时调制更强、更持久；Yuquan 也有明确的多小时起伏，但 cohort-median rate autocorr 到 8h 已接近 0

**连续 day/night 段分析**

这次不再把所有白天点/夜晚点拼成 pooled 子序列，而是先切出**连续白天段**和**连续夜晚段**，每段内独立做 600s 去趋势，再汇总。

| 数据集 | Day pooled detrended r 中位 | Night pooled detrended r 中位 | day/night 两侧都为正 |
|---|---:|---:|---:|
| Yuquan | **0.0937** | **0.0629** | **9/10** |
| Epilepsiae | **0.0823** | **0.0823** | **17/20** |

**结论**：

- PR-2.5 的方向没有变，但现在语义更硬了：短程依赖是在**连续 day/night 段内部**仍然存在
- PR-2.6 没有推翻 PR-2 / PR-2.5，而是把“慢调制”从间接推断升级为真实时间轴上的直接观察
- 因此，对外口径可以收敛为：**宏观事件率的慢漂移未见单一主导时间尺度，且在连续时间轴上延伸到多小时**。IEI 事件级 serial correlation（半衰期 ~1.8 min）与多小时率漂移是同一个慢过程的两个尺度上的投影，但它们是不同层级的量测

### 3.5 仍需补充（更远期）

~~**PR-2.5（多尺度解剖 + 被调制量验证）**~~ → **已完成**，见 §3.4.7。核心发现：(i) 慢调制是宽频段 1/f 型（Δ_frac 近似平坦）；(ii) n_participating 与 IEI 同源调制（r=0.742）→ 单一全局状态变量；(iii) 28% 短程依赖在日夜段内独立存在；(iv) 逃逸 subject 谱峰去趋势后消失 → 21/21 全覆盖。

~~**PR-2.6（连续长时程 + 连续 day/night 段）**~~ → **已完成**，见 §3.4.8。核心发现：(i) 慢调制在真实连续时间轴上延伸到多小时；(ii) Yuquan 10/10 为 near-24h continuous，Epilepsiae 最长连续段中位数 75.1h；(iii) 连续 day/night 段内部仍保留短程依赖（Yuquan 9/10，Epilepsiae 17/20）。

~~**PR-2.7（Rate-Trace 谱特征刻画 + 发作邻近调制）**~~ → **已完成**，见 `docs/event_periodicity_analysis.md` §5.9。修正实现后，核心发现应收紧为：(i) Rate-trace PSD 谱指数 β 中位 0.64（30 subjects），确认宽频段长程依赖但非严格 1/f；(ii) rate × n_participating 连续时间相干中位仅 0.358（4/26 > 0.5），不支持“强单一全局状态变量”；(iii) **seizure-triggered rate average 支持 seizure-centered broad rate elevation（pre vs earlier baseline p=0.019，16/21 同方向），但 post > pre，尚不能把它定性为纯 pre-ictal ramp**。

**给 reviewer 的意见（当前最稳妥口径）**

如果 reviewer 问“这是不是已经说明 HFO rate 是 pre-ictal biomarker？”，当前最严谨的回答应该是否定的。我们现在能 defend 的只有三点：

1. `Exp 7G` 给出了一个 **subject-level seizure-linked signal**：`[-6h,-1h] > [-12h,-6h]`，`p=0.019`，`16/21` subjects 同方向。
2. 但 `post > pre`，所以它更像 **seizure-centered broad elevation**，而不是已经锁定的纯 pre-ictal buildup。
3. 因此，这个结果是一个需要继续追到底的现象，不是已经可以写成机制定论的结论。

如果 reviewer 问“PR-2.5 的 global state variable 假说在 PR-2.7 里是否被确认？”，答案也应该收紧：**只得到弱到中等频域支持，不能再做强版陈述。**

**更远期**

4. **time-rescaled 残差检验**：在估计的 $\hat{r}(t)$ 下做时间尺度变换 $\tilde{t} = \int_0^t \hat{r}(s) ds$。在 $\tilde{t}$ 上 IEI 应该是单位指数分布且独立。KS 检验残差的指数性 + 残差独立性。这是点过程拟合优度的金标准（参见 Brown, Barbieri, Eden, Frank 2002）。

6. **与 sleep / circadian 标签关联**：Epilepsiae SQL 有 vigilance 字段，PR1.5 已经做了 day/night 标签。把每个 IEI 标注上其所在窗口的 day/night（或 NREM/REM 如果有），看 serial correlation 是否在状态切换处出现 break。

7. ~~**与发作距离关联**~~：由 PR-2.7 exp 7G (seizure-triggered rate average) 部分覆盖。若 PR-2.7 发现系统性 rate 变化，进一步做 IEI serial correlation 在发作前后的 break 分析。

8. **联合模型拟合**：拟合一个 inhomogeneous renewal process with hidden slow rate（state-space model 或 latent-Gaussian process），看拟合质量。

---

## 给合作者的总结（2026-04-09 更新）

我认为可以接受的更新过的 narrative 是这样的：

> 间期 HFO 群体事件的产生不是由网络内禀振荡器驱动的——之前 Fig 3 的 ~2 Hz 谱峰在 (i) refractory renewal null model, (ii) ISI shuffle null model 下都不再显著，并且 IEI 分布是 lognormal 而不是 power-law。这些事件更符合一个**带不应期的兴奋性点过程**（FHN / HR / theta-neuron 类），其时序由 (a) 局部兴奋性恢复动力学和 (b) 慢时间尺度的状态调制共同塑造。后者的存在被 IEI 相邻正相关（30/30 subjects）直接证明，而 PR-2.6 进一步显示**宏观事件率的慢漂移确实延伸到多小时连续时间尺度**（5 分钟 bin rate trace 自相关在 Epilepsiae 到 8h 仍为正），并不是事件索引统计上的假象。PR-2.7 直接测量了率过程的谱指数（β 中位 0.64，sub-pink-noise），但修正实现后，rate × n_participating 的连续时间相干只剩弱到中等支持（中位 0.358），因此不应再把“单一全局状态变量”说得过满。另一方面，PR-2.7 的 seizure-centered analysis 确实发现 **发作邻近的 broad rate elevation**：pre vs earlier-baseline Wilcoxon p=0.019，16/21 subjects 同方向，但 post window 甚至更高，因此目前更合理的解释是“peri-ictal broad elevation”，而不是已经锁定的纯 pre-ictal biomarker。
>
> 在空间维度上，群体事件的传播 rank 在某些个体上确实呈现出可重复的 stereotype，并且 SOZ 参与的事件比 non-SOZ 事件更刻板（探索性证据，单侧 p = 0.039）。这是论文 Fig 1, 2 叙事中**仍然站得住的那一部分**——但需要在 mixture 检测、identity bias 控制（注意 centering 不能抹掉 SOZ 作为真实源节点的传播拓扑）、按 n_participating 分层之后才能给出最终的定量结论。
>
> 老论文的 Hopfield-Kuramoto 框架在数学层面（Hebbian-like 对称连接编码 stereotype）可以保留，但节点动力学需要从 Kuramoto 振荡器换成 FHN / HR / theta-neuron 类兴奋性单元。这样既能继续解释 stereotype 的稳定性，也能自然容纳不应期、宽 IEI 分布、和慢调制驱动的事件率漂移。发作前率升高的发现进一步连接了 interictal 时序动力学与 pre-ictal 状态。

### 已完成的验证工作

| 工作项 | 状态 | 核心数字 |
|---|---|---|
| Gamma surrogate 检验 ~2 Hz 峰 | ✅ Phase 5 | 15/21 有峰 subject 的峰被平稳 renewal 完全解释 |
| 解析 renewal PSD overlay | ✅ PR-1 (exp 6A) | 16/21 |Δf| < 1 Hz；两条路径互补覆盖 19/21 (90%) |
| ISI shuffle surrogate | ✅ Phase 5 | 仅 1/30 同时通过两重检验（huanghanwen, n=484, 可能假阳性） |
| Power-law vs lognormal (MLE + LLR) | ✅ Phase 4 | 30/30 lognormal 显著优于 power-law |
| Packing window 扫参 | ✅ Phase 2 exp 1 | f_peak ≠ 1/W；dead-zone 受 W 控制但峰频不受 |
| Centroid bypass | ✅ Phase 2 exp 2 | 13/15 峰频差 < 0.1 Hz |
| SOZ vs non-SOZ dead-time | ✅ PR-1 (exp 6B) | SOZ < non-SOZ, p=0.008 (n=8 pairs, 探索性) |
| IEI serial correlation (lag-1) | ✅ Phase 2 exp 4 | 30/30 正相关，r 中位数 ≈ 0.31 |
| 传播刻板性 SOZ vs non-SOZ | ✅ Phase 2 exp 5 | SOZ > non-SOZ, p=0.039 (n=12 pairs, 探索性) |
| **PR-2: lag-k 衰减 + 去趋势 + block 内 + SOZ 分层** | **✅ exp 7** | **半衰期 107s; 72% 慢漂移 + 28% 短程; within-block 30/30 正; SOZ > nonSOZ p=0.055 (n=9)** |
| **PR-2.5: 多尺度调制 + n_part + day/night + 回填** | **✅ exp 7b** | **Δ_frac 平坦(1/f); n_part 互相关 0.742; day/night 28/30 正; 1084+1096 峰消失→21/21 全覆盖** |
| **PR-2.6: 连续长时程 + 连续 day/night 段** | **✅ exp 7c** | **Yuquan 10/10 near-24h continuous; Epilepsiae 最长连续段中位 75.1h; binned rate autocorr 到 8h 仍正 (Epilepsiae); 连续 day/night 段内 pooled detrended r 仍正 (26/30)** |
| **PR-2.7: Rate-trace 谱特征 + 发作邻近调制** | **✅ exp 7d** | **β 中位 0.64; coherence 中位 0.358; seizure STA: pre > earlier baseline p=0.019 (16/21), but post > pre → seizure-centered broad elevation** |

### 已完成项的状态更新（2026-04-09）

1. **PR-2：IEI serial correlation 深层分析** — **已完成**。详见 §3.4。核心发现：(i) 调制半衰期 ~1.8 分钟；(ii) 72% 慢漂移 + 28% 短程依赖；(iii) 跨 block 污染排除；(iv) SOZ 倾向于更强（p=0.055，边缘）。

2. **PR-2.5：多尺度调制解剖 + 逃逸 subject 回填** — **已完成**。详见 §3.4.7。核心发现：(i) 慢调制是宽频段 1/f 型（Δ_frac 平坦），无单一主导时间尺度；(ii) n_participating 与 IEI 同源调制（互相关 0.742），证实全局状态变量假说；(iii) 28% 短程依赖在日夜段内独立存在；(iv) **逃逸 subject 1084, 1096 的谱峰去趋势后完全消失 → 21/21 全覆盖，~2 Hz 周期性假说彻底终结**。

3. **PR-2.6：连续长时程调制分析** — **已完成**。详见 §3.4.8。核心发现：(i) 宏观事件率（binned rate trace）的慢漂移在真实连续时间轴上延伸到多小时（注意这不等于 IEI serial correlation 本身持续到多小时）；(ii) Yuquan 的 24h 优势被真正利用，10/10 near-24h continuous；(iii) 连续 day/night 段内部仍保留短程依赖（subject 内同标签段 pooled），说明此前结论不是 pooled 标签伪影。

4. **PR-2.7：Rate-trace 谱特征 + 发作邻近调制** — **已完成**。详见 `docs/event_periodicity_analysis.md` §5.9。修复多段谱平均 bug 后，核心发现应更新为：(i) Rate-trace PSD 谱指数 β 中位 0.64（范围 0.04–1.62），确认率过程具有超越白噪声的长程依赖但非严格 1/f；(ii) rate × n_participating 连续时间相干中位仅 0.358，频域耦合偏弱，不支持强版全局状态变量；(iii) **seizure-triggered rate average: pre-ictal [-6h,-1h] vs baseline [-12h,-6h] Wilcoxon p=0.019, 16/21 pre > baseline，但 post > pre (p=0.016)**。因此更准确的结论是：存在 seizure-centered broad rate elevation，pre-window 已可见升高，但尚不能把它定性为纯 pre-ictal ramp。

### 最值得继续做的事（按性价比，更新版）

1. **PR-3：传播 stereotype 稳健性检验**（话题 1 P0）。Hartigan dip test 检测 mixture stereotype；centered rank tau 控制 detection ordering bias；按 n_participating 分层去除离散化噪声。让 Fig 2 的 17/18 + 20/20 变得真正可信。**注意：centering 存在"把婴儿和洗澡水一起倒掉"的风险——SOZ 通道因高兴奋性天然点火最早，均值扣除可能抹掉真实的 SOZ 驱动的传播拓扑。计划中已加入必选诊断：对比 center 前后的 top-3 源通道列表，如 SOZ 源节点被抹掉则报告 `soz_source_erased`，并侧报 raw vs centered tau 而非仅报 centered。**

2. **对 huanghanwen 这个唯一"genuine"做敏感性分析**：把不同 surrogate 参数（gamma shape、shift、shuffle 类型）扫一遍，看它是否真的稳健。n=484 的样本量本来就在边缘。

3. **Seizure-triggered rate 的后续深化**：PR-2.7 目前最值得追的不是“继续喊 pre-ictal marker”，而是分解这个 broad elevation 到底有多少是真 pre-ictal buildup、多少是 post-ictal carryover、多少是 seizure clustering / circadian baseline。至少要做：(i) matched control windows；(ii) 去 overlap / 去 cluster 的 seizure subsampling；(iii) 与 PR4–PR6 synchrony 在同一 seizure-centered windows 上正面对比。

整篇论文的叙事现在更稳妥的版本应是 **"interictal events as a refractory excitable point process under multi-timescale state modulation, with seizure-centered broad rate elevation and SOZ-specific propagation stereotypy"**。这比“~2 Hz oscillator”强得多，也比“已经抓到 pre-ictal biomarker”诚实得多。下一步若要把临床相关性做实，优先级已经从 PR-3 并列转向对 seizure-centered rate elevation 的机制拆解。

### PR-2.9 计划（2026-04-10 草案，先写入不提交）

这件事值不值得做？**值得。** 但前提是别再做“多画几张 seizure-centered 图”的伪进展。PR-2.9 的唯一意义，是把 PR-2.7 里那个 `p=0.019` 的信号拆干净，回答下面这个真问题：

> 这到底是 **true pre-ictal buildup**，还是 **broad peri-ictal elevation / cluster contamination / circadian baseline**？

如果回答不了这个问题，继续谈 biomarker 就是在自欺欺人。

#### 核心判断

- **值得做**：因为 `Exp 7G` 已经给出了一个真实的 subject-level seizure-linked signal，但解释仍然被三个混杂卡死了。
- **最大风险**：把 repeated seizures 当独立样本，或者继续使用 overlap 污染的窗口，最后得到看似更显著、其实更假的结果。
- **设计原则**：统计单位必须仍是 **subject**；所有新增分析都必须明确消掉一种歧义，而不是加描述性图。

#### PR-2.9 应该只做三件事

1. **分段轨迹（piecewise trajectory）**
   把 `[-12,-6] / [-6,-1] / [+1,+6]` 这种粗窗口改成非重叠分段：`[-12,-6]`、`[-6,-3]`、`[-3,-1]`、`[-1,0]`、`[0,1]`、`[1,3]`、`[3,6]`、`[6,12]`。  
   目的不是多做统计，而是判断形状：是逐渐爬升到 0 点，还是前后一起抬高。

2. **隔离 / cluster 分层**
   明确算每次 seizure 的 `time since previous seizure` 和 `time until next seizure`。  
   至少比较三组：
   - isolated: `[-12h,+12h]` 内没有别的 seizure
   - pre-clean: `[-12h,0]` 内没有 prior seizure
   - clustered: `[-12h,0]` 内有 prior seizure

   如果所谓 pre-elevation 在 isolated / pre-clean 里消失，只在 clustered seizure 里存在，那它八成就是 post-ictal carryover 或 cluster state，不是什么 prediction。

3. **matched control + synchrony 正面对比**
   对每个 seizure pre-window `[-6h,-1h]`，在同 subject 里找 **同 local clock、同 day/night、且离任意 seizure 足够远** 的 control windows。  
   然后在**同一批窗口**上比较：
   - rate: seizure-pre vs matched control
   - synchrony: seizure-pre vs matched control

   这一步是关键。否则你没资格说“rate 比 synchrony 更敏感”，因为之前根本不是同窗比较。

#### Reviewer-facing 预期口径

PR-2.9 做完后，结论只能落在三种之一：

1. **True pre-ictal buildup**
   isolated seizures 仍保留渐进式 pre-onset rise，matched controls 没有，且 synchrony 仍为 null。

2. **Broad peri-ictal elevation**
   前后都高，只是 pre-window 先开始抬升；更像 seizure-centered state shift，不该写成 prediction marker。

3. **Cluster / baseline artifact**
   一旦做 isolation / matching，效应基本消失。那就该老老实实承认 PR-2.7 抓到的是混杂。

#### 我的审稿式结论

这个 PR-2.9 方案现在是**可靠且合理的**，因为它直接打三个真正的漏洞：形状、cluster、baseline。  
不合理的做法只有两种：

1. 把 seizure 当独立样本继续堆显著性。
2. 不做 matched controls，就继续喊 pre-ictal biomarker。

那种做法是垃圾，会把一个本来很有价值的现象做废。
