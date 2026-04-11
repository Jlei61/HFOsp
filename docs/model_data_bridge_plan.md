# 数据与模型之间的逻辑缝合方案

> 衔接：本文作为 `interictal_population_event_methodological_review.md` 的续篇。
> 目的：在不大动 KONWAC 模型代码的前提下，把 PR-2.5 / 2.6 / 2.7 的实证发现（refractory point process + 多时间尺度慢调制 + peri-ictal broad elevation + 部分弱化的 single state variable）和老论文的 Hopfield-Kuramoto / adaptive Kuramoto 模型缝成一个自洽的故事。
> 阅读对象：合作者（已读过上一份审视报告）。

---

## 0. 你的诊断是对的，但要更精确

你说的"data 和 model 是两张皮"确实是病。但我想先把这个病拆得更细——只有看清三个层面的撕裂在哪里，缝合方案才能有的放矢：

1. **本体论撕裂（model class 不匹配）**
   数据：refractory excitable point process（兴奋性单元 + 不应期 + 静息态）
   模型：phase oscillators on $S^1$（无静息态，无不应期，永远在转）
   这是上一份报告里我反复强调的"FHN/HR vs Kuramoto"的问题。

2. **驱动机制撕裂（slow variable 在模型里没有位置）**
   数据：~72% 的 IEI 序列相关来自 600s 以上的慢漂移；rate trace 在多小时连续时间轴上仍有自相关；peri-ictal broad rate elevation。
   模型：α 和 β 是被外部"扫"出来的参数，没有任何动力学；adaptive plasticity 是 fast 的（time constant 与节点动力学同尺度），不能扮演 slow modulation 的角色。
   这是 Gemini 抓到的那一层。

3. **时间尺度撕裂**
   数据：相关的尺度是分钟到小时（half-life ~108s，rate 多小时仍正相关）。
   模型：Kuramoto 的内禀时间尺度是 ~1/ω ~ 0.5s。
   即使让 α 跟 S(t) 走，如果 S(t) 的时间尺度不引入到模型里，整个模型仍然只能讲秒级的故事。

**Gemini 的方案打中了第 2 层，绕过了第 1 层，部分覆盖了第 3 层。** 这就是为什么它读起来"差点意思"——它把 slow variable 接到了 α 上，但没有解决"Kuramoto 节点本身就不是兴奋性单元"这个更深的麻烦。如果直接照它的版本写进论文，审稿人一眼就能看出来：你画了一个慢驱动的相位振子网络，然后说它复现了不应期和正序列相关——可问题是你的振子根本就没有不应期，从哪里来的不应期？

下面我提出一个更稳的版本，关键是**先把 Kuramoto 的本体论换掉，再把 slow variable 接进去**。这两步做完，故事才闭环。

---

## 1. 关键的一步：把 Kuramoto 从"微观神经元模型"改写成"群体释放周期的粗粒化模型"

这是缝合方案的真正基石。下面把它拆开讲，因为这个 reframing 对整个 narrative 影响很大，而且**完全不需要改一行代码**——它只是改变了 $\theta_i$ 和 $\omega_i$ 的物理解释。

### 1.1 老论文的隐含解读 vs 新解读

**老论文的隐含解读**（虽然没明说）：每个 Kuramoto 振子 = 一个 SEEG contact 附近的神经元集合，θ_i 表示这个集合的"瞬时相位"，ω_i ~ 2π × 2 Hz 是它的内禀振荡频率。这等同于声称：每个 contact 周围有一个 ~2 Hz 的真实振荡器。**这个声称已经被数据反驳。**

**新解读**：每个 Kuramoto 振子 = 一个局部群体（assembly），θ_i 表示这个群体在其**慢释放周期**（slow release cycle）上的位置；当 θ_i 跨过 2π（或某个阈值 ε）时，这个群体释放一次 detected HFO 事件。ω_i 是这个集群在静息背景下的平均释放速率，**远小于 2 Hz**——可能在 0.1–0.5 Hz 量级。不应期不是被显式建模的，而是被**吸收进 θ_i 在 [0, 2π] 区间内行走所需的时间**——你可以把走完 [0, 2π] 所需的时间理解成"集群从一次释放到下一次完整恢复 + 整备 + 触发的总耗时"。

### 1.2 这个 reframing 解决了什么

| 之前的麻烦 | reframing 之后 |
|---|---|
| Kuramoto 振子没有静息态 | θ_i 是慢周期变量，"静息"等同于"在 [0, 2π] 中段行走"，不需要显式静息点 |
| Kuramoto 振子没有不应期 | 不应期 = 最近一次跨 2π 之后到下一次需要走完一整圈的时间下限 |
| ω_i ~ 2 Hz 直接被数据证伪 | ω_i 现在是亚赫兹的"释放节律"，与 ~2 Hz 的伪 PSD 峰彻底脱钩 |
| Kuramoto 不能产生宽尾 IEI 分布 | ω_i 的跨节点异质性 + S(t) 慢调制 → 自然产生 lognormal-ish 形态 |
| Kuramoto 的"events"是周期性的 | events 不再是单一振子的周期穿越，而是**多振子同时跨阈值的 coincidence**（见 §1.3） |

### 1.3 如何在 reframing 之后定义"事件"

这是 reframing 的灵魂部分。**不要**把单个振子的 2π 跨越直接当成一次 IE——那样产生的事件会被振子自身的周期性污染。

正确的做法：定义一个 **coincidence-based event detector**：

> 在模型时间 $t$，如果有 $\ge n_\text{thr}$ 个振子在 $[t, t + \Delta]$ 时间窗内同时跨 2π，则记录一次群体事件。

这模仿了真实数据 pipeline 里的"多通道同步检测"步骤。这样定义出来的事件序列：

- IEI 不再是 $1/\omega_i$，而是由多振子跨阈值过程的统计决定，自然得到右偏分布
- 参与振子集合（即 model 端的 n_participating）随 S(t) 变化而变化
- 当所有振子被 Hebbian 耦合 + S(t) 推到接近全锁相时，coincidence 频率爆涨 → 模型的 ictal 状态

这个改动是**post-processing**，不需要改 Kuramoto 求解代码本身，只在 trajectory 输出上加一层 event extraction。

### 1.4 这个 reframing 是不是诚实的？

是的，而且它是 EEG/MEG 建模文献里的主流用法。Breakspear、Deco、Jirsa 那一脉的所有 large-scale brain network model 都把每个 Kuramoto 节点当成"一个脑区的群体节律"，从来不当成单个神经元。我们这里只是把这个标准做法**显式写出来**而已。老论文之所以撞墙，是因为它把这一层默认前提给省略了，结果合作者和审稿人会不自觉地按"每个 Kuramoto 节点 = 一个 SEEG contact 上的实际振荡器"去读，于是出现"为什么 PSD 没有真振荡？"的尴尬。

把这一层讲透之后，原来对 Kuramoto 的所有"为什么没有不应期"的质疑都会自动失效——因为我们承认 Kuramoto 是 coarse-grained 的，refractory dynamics 已经在更细的尺度被吸收掉了。

---

## 2. 引入慢变量 S(t)：Gemini 方案的修正版

reframing 完成之后，再接 Gemini 的"加 OU 慢驱动"那一步就顺理成章了。但有几处我会修改。

### 2.1 应该让 S(t) 调制什么？

Gemini 建议让 $\alpha(t) = \alpha_0 + \gamma S(t)$，即调制相位延迟。这没错，但**不是性价比最高的入口**。

我建议让 S(t) 同时（或主要）调制 **基础角频率** $\omega_0$：

$$\omega_i(t) = \omega_i^{(0)} \cdot (1 + \beta_\omega S(t))$$

理由：

1. **数据告诉我们的是事件率在变，不是相位延迟在变。** PR-2.7 的 rate trace + n_participating coherence 直接说的是"群体释放速率"被慢调制。最直接的对应就是 ω。
2. **调制 ω 自动产生 IEI 序列相关。** ω 升高 → 相邻 events 间隔缩短 → log IEI[n] 与 log IEI[n+1] 都偏小 → 正相关。这就是数据里 30/30 看到的现象。如果只调制 α，需要再绕一道才能到 IEI 的统计量。
3. **调制 α 的必要性留给 ictal transition**，因为 α 是控制 splay→sync 相变的关键参数。所以把两件事分开：S(t) 调制 ω 解释间期慢漂移；S(t) 极端值（或其衍生量）通过短时可塑性间接推动 α 漂移到临界点 → 发作。

更完整的方案是**两个慢变量**：

$$\omega_i(t) = \omega_i^{(0)}(1 + \beta_\omega S(t))$$
$$\alpha(t) = \alpha_0 + \beta_\alpha S(t) + \text{(adaptive plasticity dynamics)}$$
$$\dot{S} = -S/\tau_S + \sigma_S \eta(t)$$

其中 $\tau_S$ 直接从 PR-2 的数据校准（半衰期 ~108s → $\tau_S \approx 150$s 的 OU 过程；如果想匹配多小时尺度，则需要叠加两个不同 $\tau_S$ 的 OU 分量，参考 PR-2.5 的"宽频段 1/f 型"发现）。

### 2.2 为什么是 OU 而不是别的？

OU 是最简单的、有有限自相关时间的平稳 Gauss 过程。它不需要任何额外假设。如果你想认真起来，**用两个 OU 加和**（一个 $\tau_1 \sim 100$s，一个 $\tau_2 \sim$ 几小时）来匹配 PR-2.5 看到的"宽频段 1/f 型 detrend fraction"分布——这是一个非常便宜但很有信息量的升级。

### 2.3 "S(t) 是什么"——给 reviewer 的物理解释

S(t) 不是抽象数学符号，要给它生理学命名。三个候选，按可辩护性排序：

1. **整体兴奋性（global excitability）/ permittivity-like variable**（最稳）。直接借用 Epileptor 框架的 permittivity 概念，但承认我们这里的 S 是经验抽象，不必满足 Epileptor 全部假设。
2. **觉醒/睡眠状态的连续代理**。PR-2.5 的 day/night 分层 + 28/30 两段都为正说明 S 至少包含一个昼夜成分。
3. **皮层 E/I 平衡漂移**。最有机制感但最难证明。

我建议在论文里写法 1，在 discussion 里 hint 法 2 和 3，避免过度承诺。

---

## 3. 这套修正版能讲什么故事？

这是关键部分。下面是我建议的叙事主线，按时间尺度从短到长组织。可以直接套到 discussion 的开头几段。

### 3.1 微观尺度（< 1 s）：单个事件的释放

每个 SOZ assembly 处于亚阈值兴奋性状态，由噪声 + 上游输入触发一次 HFO 释放。这部分不在 Kuramoto 模型显式描述，但在模型里被吸收进 θ_i 跨过 2π 的瞬间。

### 3.2 中观尺度（1 s – 1 min）：群体事件的形成

Hebbian-shaped 连接矩阵 $K_{ij}$ 让 assemblies 之间的"释放节律"相位锁定，于是当多个 assemblies 在短时间窗内同时跨阈值时，它们的 rank 顺序高度可重复——这就是数据 Fig 1, 2 看到的 stereotype propagation。模型这一部分**完全不变**：原来 Hopfield-Kuramoto 怎么编码 phase pattern，现在还是怎么编码。

### 3.3 介观尺度（1 min – 多小时）：S(t) 调制的事件率漂移

S(t) 缓慢漂移 → 所有 ω_i 同步升降 → 事件率随 S(t) 同步升降 → 相邻 IEI 出现正序列相关 → 同时 n_participating 也随之同步升降（因为 ω_i 的整体上升使得多振子 coincidence 更容易发生）。

这一节的每一个预测都直接对应我们已经测出来的数据：

- 正序列相关 → PR-2 exp 4：30/30 方向一致
- IEI 半衰期 ~分钟 → PR-2 exp 7：lag-1 半衰期中位 108s
- 慢漂移宽频段 → PR-2.5 Δ_frac 平坦 + PR-2.6 多小时 rate autocorr
- IEI 与 n_participating 同源 → PR-2.5 互相关 0.742（注意 PR-2.7 修正后的频域 coherence 只有 0.358，这是个**好消息**，因为它说明 S(t) 不是完美驱动，振子 ω_i 的异质性给 rate 和 n_participating 解耦留了空间——模型也应该预测这种部分耦合而不是完美耦合）

### 3.4 宏观尺度（接近发作）：splay → sync 的临界跨越

S(t) 漂移到极端高值时（peri-ictal broad rate elevation 阶段，对应 PR-2.7 exp 7G 看到的 pre-window > baseline），通过 $\beta_\alpha S(t)$ 把 α 推到 splay 失稳的临界点附近；此时 fast adaptive plasticity（原来 KONWAC 模型里那一套）接管，提供正反馈，把系统从 splay 推过临界点，进入 sync 状态——即一次发作。

这部分的模型代码也**完全不变**——原来是手动扫 α 看相变，现在改成"由慢变量自动驱动 α 漂移到临界"。

### 3.5 这套故事的好处

1. **承认数据的胜利**。refractory point process + slow modulation 不再是模型的对手，而是被合并进了模型的解释力之内。
2. **保留模型的全部数学**。Hebbian 编码、splay 状态、first-order 相变全部保留，只是被赋予了新的物理解释。
3. **新增的可证伪预测**。modelling 部分现在可以做"模型预测的 IEI 序列相关 vs 数据观测的"对照、"模型预测的 n_participating 与 rate 共调制 vs 数据"的对照，等等。这是论文新的 quantitative validation 章节。
4. **直接解释 SOZ vs non-SOZ dead-time 差异**。SOZ assemblies 的 ω_i 更高（兴奋性更强），所以平均跨阈值更频繁、IEI 中位数和 dead-time 都更短。这正好对应 PR-1 exp 6B 的发现。

---

## 4. Torus 动力学图景：把它变成一句精确的话

你提到 torus 直觉，Gemini 也强调了这一点。我同意它是缝合的高光，但要把它说精确，否则很容易听起来像隐喻。

### 4.1 精确版本

N 个 Kuramoto 振子的状态空间是 $T^N = (S^1)^N$，N 维环面。在这个环面上：

- **Splay state (interictal stereotype)** = 一个 1 维子流形。这个子流形参数化为 $(\theta_1, \theta_2, \ldots, \theta_N) = (\phi + \delta_1, \phi + \delta_2, \ldots, \phi + \delta_N)$，其中 $\phi$ 是整体相位（沿子流形流动），$\delta_i$ 是 Hebbian 编码的固定 phase offset。每次群体事件发生，trajectory 都沿同一个 1 维子流形走一遍 → stereotype。

- **Sync state (ictal)** = 对角线 $(\phi, \phi, \ldots, \phi)$，也是 1 维子流形，但 $\delta_i$ 全部为 0。

- **Slow variable S(t)** = 改变 $T^N$ 上向量场倾斜度的外部参数。低 S 时 splay 子流形稳定；高 S 时 splay 子流形失去稳定性（saddle-node bifurcation of the manifold），trajectory 滑落到 sync 子流形 → 发作。

- **数据中的 IEI 序列相关** = 沿 splay 子流形的"前进速度"被 S(t) 调制，所以连续几次穿过参考截面（events）的间隔系统性地短-短或长-长聚集。

### 4.2 你引用的 Nature 小鼠论文的对接

那篇论文（基于你的描述，是 mouse 发作期同步用 torus 动力学描述）很可能展示了发作期 trajectory 在低维 torus 上的具体几何形态。我们的数据没有那么细的单细胞分辨率，但可以做一个**类似精神**的可视化：

- 用 PCA 或 LDA 把 N 维 lagPatRank 投到 2D
- 每个事件在 2D 上是一个点
- 同一个 stereotype cluster 的事件应该聚成一团（splay 子流形的投影）
- 如果有数据接近发作，可以看到这些点的 spread 是否随 S(t) 系统性变化

这是 PR-3 之外的一个**完全免费的可视化**，可以放在 Fig 4 或 Fig 5。

### 4.3 在 discussion 里怎么写

我建议直接抄一段（你可以润色）：

> The interictal stereotyped propagation is not a fixed firing chain but a stable splay trajectory on a Hebbian-structured submanifold of the network's $N$-torus phase space. The slow global excitability variable, which manifests empirically as the multi-timescale drift in event rates and the concurrent modulation of $n_\text{participating}$, acts as a bifurcation parameter that gradually distorts the vector field on this torus. As the slow variable approaches its critical range — empirically observed as a peri-ictal broad rate elevation — the splay submanifold loses stability via the same first-order bifurcation captured by the adaptive Kuramoto reduction, and the trajectory collapses onto the synchronized diagonal of the torus, manifesting as a clinical seizure.

这一段把数据发现的每一项（rate 调制、n_participating 共变、peri-ictal elevation、splay→sync）都串到 torus 几何上。

---

## 5. 模型部分需要补充的具体实验（性价比从高到低）

下面几个实验都是在已有 Kuramoto 求解代码上加 post-processing 或加一个外部驱动方程，**不需要重写模型**。

### 5.1 Exp M1：S(t) 驱动的间期模拟（最高优先级）

**目的**：证明 reframing + S(t) 慢驱动能定量复现 PR-2 / PR-2.5 / PR-2.6 的间期统计特征。

**做法**：
1. 用现有 Hopfield-Kuramoto 代码，设置网络结构 = 某 subject 的 Hebbian-encoded 连接矩阵
2. 加一个 OU 驱动方程 $\dot{S} = -S/\tau_S + \sigma_S \eta$，$\tau_S = 150$s
3. 把 $\omega_i$ 改成 $\omega_i^{(0)}(1 + \beta_\omega S(t))$，$\beta_\omega \sim 0.3$
4. 跑 24 小时模型时间
5. 用 coincidence-based detector 提取群体事件，得到模型 IEI 序列
6. 在模型 IEI 上重跑数据 pipeline 中的所有分析：
   - lag-1 serial correlation（应当 ~0.3）
   - lag-k 衰减半衰期（应当 ~分钟级）
   - IEI 分布（log-likelihood ratio test 应当也是 lognormal 优于 power-law）
   - 模型 PSD（应当看不到尖锐的 ~2 Hz 峰，或者峰能被 gamma surrogate 解释）

**预期结果**：模型再现数据的所有间期统计特征。这是新增的 Fig 5 或 Fig 6 panel。

**这一个实验可能比所有现有 model figures 加起来更有说服力**，因为它把一个"慢驱动 + 释放周期 + Hebbian 拓扑"的 minimal model 与真实数据做了 quantitative head-to-head。

### 5.2 Exp M2：S(t) 调制 → ictal transition

**目的**：把"慢变量驱动 splay→sync 相变"做成连续动力学，而不是手扫参数。

**做法**：
1. 在 Exp M1 基础上，让 S(t) 的 OU 驱动有一个偶发的"大偏移"（比如混合一个稀疏的大幅扰动），代表 peri-ictal 状态
2. 让 $\alpha(t) = \alpha_0 + \beta_\alpha S(t)$
3. 跑模型，观察大偏移期间是否触发 splay→sync 相变（即一次模拟的发作）
4. 在模拟发作前的窗口内重做 Exp 7G 的分析：模型的 rate trace 在发作前是否有 broad elevation？n_participating 是否同步上升？

**预期结果**：模型在发作前 ~1h 出现 rate elevation，参与节点数上升，最终崩塌到 sync。这直接对接 PR-2.7 exp 7G 的实证发现，让模型 prediction 和数据 prediction 在同一张图上对照。

### 5.3 Exp M3：模型层面的 SOZ vs non-SOZ

**目的**：再现 PR-1 exp 6B 的 SOZ dead-time 更短现象。

**做法**：
1. 在模型里给 SOZ-内节点更高的 $\omega_i^{(0)}$（更强兴奋性）
2. 跑同样模拟
3. 分别从 SOZ 子集和 non-SOZ 子集提取事件
4. 比较两者的 dead-time（IEI 2nd percentile）

**预期结果**：SOZ 子集 dead-time 系统性更短，再现 p=0.008 的方向。

### 5.4 Exp M4：torus 投影可视化

**目的**：把 stereotype = splay submanifold on torus 这件事做成视觉证据。

**做法**：
1. 模型跑一段间期（无相变）
2. 在每次模型事件发生时记录 N 维相位向量 $(\theta_1, \ldots, \theta_N)$
3. 用 PCA 或 ISOMAP 投到 2D
4. 同样的事情在真实数据上做：每次群体事件的 lagPatRank 向量投到 2D
5. 两张图并列，应该都呈现紧致的低维 cluster（splay submanifold 的投影）

**预期结果**：模型与数据的 2D 投影在拓扑上相似，给"两者都生活在 splay 子流形上"提供视觉证据。

### 5.5 优先级建议

| 实验 | 优先级 | 理由 |
|---|---|---|
| Exp M1 | **P0** | 直接缝合数据和模型；新增最大说服力 panel |
| Exp M2 | **P0** | 把 ictal transition 从"参数扫描"升级到"由慢变量自然触发"，直接对接 PR-2.7 |
| Exp M4 | P1 | torus 几何可视化，叙事高光，工程量小 |
| Exp M3 | P2 | nice-to-have，但 PR-1 exp 6B 已经有数据侧的支持，不是唯一证据 |

---

## 6. 这套方案不能解决什么——保持诚实

我不想给你画一张完美的饼。这个 reframing 方案有几个仍然不解决的弱点，建议你在 discussion 里提前自己说出来，不要等审稿人挑：

1. **Coarse-grained Kuramoto 不是从微观推导出来的。** 我们说"每个振子代表一个 assembly"，但没有给出从 microscopic FHN/HR 网络到 mean-field Kuramoto 的严格推导。可以引用 Breakspear 等人的 mean-field reduction 文献，但那些推导通常只在 Hopf 邻域成立——我们的数据在 Hopf 之外。**这是一个理论上的妥协**，要承认。

2. **"Events"的定义在模型和数据之间不完全等价。** 数据 pipeline 是 80–250 Hz envelope thresholding + packing；模型 pipeline 是 multi-oscillator coincidence。两者只能在统计性质上对比，不能逐 event 一一对应。

3. **Slow variable S(t) 是经验抽象，不是从生理学第一性原理写出来的。** 我们没有独立测量"全局兴奋性"，只是观察到 IEI 和 n_participating 的共调制。OU 形式是数学上最简洁的选择，但真实生理调制可能更复杂（多时间尺度叠加、状态切换等）。

4. **Coherence 0.358 vs 0.742 的张力还在。** PR-2.5 的 lag-cross-correlation 给 0.742，PR-2.7 的频域 coherence 只给 0.358。这两个量从概念上不完全等价（前者是 rank 相关的衰减形状对比，后者是连续频域线性耦合），但合作者会问。我建议在论文里**明确说**这两个量不等价、各自的物理含义不同，然后采用一个保守口径："IEI 和 n_participating 受同一个慢调制影响，但耦合是部分的，振子异质性留下了解耦的余地"——这恰好与模型 Exp M1 应当预测的部分耦合相符。

5. **Adaptive Kuramoto 的 plasticity 部分仍然解释不清。** 老论文的 STDP-like 短时可塑性在生理学上对应什么？这是原 paper 就有的弱点，reframing 没有补救。建议把它降格成"a phenomenological positive-feedback term that captures the fast network reorganization observed at seizure onset (e.g., Huberfeld 2011, Dzhala & Staley 2003)"，不要硬讲机制。

---

## 7. 给合作者的一句话总结

> 我们之前的 narrative 是：interictal 群体事件 = 内禀振荡器 + Hebbian 编码 → 间期周期性传播 → 发作前 splay→sync 相变。
> 现在更稳的 narrative 是：interictal 群体事件 = 一个被慢变量调制的、Hebbian 拓扑约束下的群体释放过程。我们的数据揭示了这个慢变量的多时间尺度统计特征（PR-2 / 2.5 / 2.6 / 2.7）和它在 peri-ictal 阶段的 broad elevation。我们的模型（重新解读为 coarse-grained release-cycle network 之后）在加入这个慢驱动后，能定量复现间期统计特征（Exp M1），并把发作转换从"手动扫参数"升级为"慢变量自然推动相变"（Exp M2）。Splay 与 sync 之间的几何关系，可以用 torus 上的 1 维子流形 vs 对角线统一描述。

这句话的关键是：**慢变量从"数据里观察到的现象"变成了"模型动力学的一个组成部分"**。一旦做到这一点，"data 和 model 是两张皮"的指控就站不住了——它们现在共享同一个核心动力学对象（S(t)），只是在不同尺度被观测。

---

## 8. 与上一份方法学审视的衔接

| 上一份报告的问题 | 本文方案 |
|---|---|
| 老论文用 Kuramoto 是 model class 选错了 | 通过 reframing 为 coarse-grained release-cycle model 化解，不需要换成 FHN/HR |
| ~2 Hz 不是真振荡 | 模型现在不再依赖 ~2 Hz 内禀频率；ω_i 是亚赫兹释放节律 |
| IEI 正序列相关排除振荡器假设 | 模型加入 S(t) 后能预测正序列相关，与数据一致 |
| n_participating 与 rate 同源调制 | Exp M1 直接验证模型也能再现部分耦合 |
| Peri-ictal broad rate elevation | Exp M2 由 S(t) 大偏移自然产生 |
| SOZ dead-time 更短 | Exp M3 通过给 SOZ 节点更高 ω_i 再现 |
| 传播 stereotype（仍然站得住的部分） | 模型这一部分**不变**，原来 Hopfield-Kuramoto 的 Hebbian encoding 完整保留 |

---

## 9. 最后：要不要在论文里 explicitly 提 FHN/HR 替代方案？

这是一个策略问题。我的建议：

**提一句，不展开。** 在 discussion 的方法学反思部分写一段类似：

> A more microscopically faithful approach would replace the phase oscillators with explicit excitable units (e.g., FitzHugh-Nagumo or theta-neuron model) and treat the refractory dynamics from first principles. We adopt the coarse-grained Kuramoto formulation here because (i) it admits the Hebbian-style phase encoding required to reproduce the stereotyped propagation, (ii) the slow modulation observed empirically operates at a population level that is naturally captured by the mean-field phase variable, and (iii) it preserves a tractable analytical handle on the ictal phase transition. A more refined excitable-network model is a natural next step.

这样既承认了模型的局限，又给未来工作留了 hook，同时不会让 reviewer 觉得"那你为什么不直接用 FHN？"。

---

## 总结：要做的事情清单

**最少必须做（让 narrative 闭环）**：

1. 在论文 model 方法节加一段，明确把 Kuramoto 重新解读为 coarse-grained release-cycle model，而不是 microscopic neuron model
2. 加入 S(t) 的 OU 驱动方程，并把 $\omega_i$ 写成 $\omega_i^{(0)}(1 + \beta_\omega S(t))$
3. 跑 Exp M1，得到一张"模型 vs 数据"的 IEI 统计对照图（lag-1 r、半衰期、lognormal fit、PSD）
4. 跑 Exp M2，得到一张"模型由慢变量驱动的 ictal transition"图，叠加 PR-2.7 的 seizure-triggered rate average 作为对照
5. 在 discussion 加一段 torus 几何描述，把 splay = 1D submanifold、sync = diagonal 写清楚

**值得加做（拔高）**：

6. Exp M4 的 torus 投影可视化（PCA/ISOMAP on 模型相位 + 真实 lagPatRank）
7. Exp M3 的 SOZ vs non-SOZ 模型再现

**可以暂时不做**：

- 完全推翻 Kuramoto 换 FHN（成本太高，性价比低）
- 严格的从 microscopic 到 mean-field 推导（理论工作，超出当前论文 scope）
- 多时间尺度 OU 叠加（除非 reviewer 强烈要求）

**关键认识**：你不需要为了配合数据而重写模型。你只需要承认 Kuramoto 是 mean-field 描述（这是它在文献里的标准用法），加一个慢驱动方程（一行代码），定义 coincidence-based events（post-processing），跑两个仿真（M1 + M2）。整个修补的代码工作量是 1–2 天，narrative 收益是从"两张皮"变成"统一动力学图景"。

这就是我给的方案。Gemini 的方向对，但只补到了第 2 层（slow variable），没补第 1 层（Kuramoto 的本体论解读）。把第 1 层补上，整个故事才真正闭合。
