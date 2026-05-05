# 分层模型框架：承认尺度分离，不强行统一

## 0. 你的诊断是对的，我上一版错在哪

我上一版提的 "Kuramoto 整体 reframe 成 release-cycle" 方案有一个根本毛病：它为了缝合 inter-event 的慢调制现象，**动了 intra-event 老论文还站得住的那部分**。你敏锐地指出：老论文 Fig 1-2 的 "每个 contact = 一个振子，Hebbian 编码稳定相位差" 这套叙事，在 per-channel 证伪极其困难的前提下，**是可以接受的现状**。

所以正确的 move 不是"把 Kuramoto 换一种解读"，而是**承认我们面对的是两个不同尺度、不同物理对象的现象，应该用两个分层模型，而不是一个大一统模型**。

这正好对应物理学里的标准做法：你不会用量子电动力学去解释流体力学，即使两者原则上同源。

---

## 1. 把现象严格分层

### Layer A（微观，秒级，SOZ 内部）：事件**内部**时序稳定性

| 维度 | 内容 |
|---|---|
| 经验现象 | Fig 1–2 的 stereotyped propagation，SOZ 内 contacts 之间 ~ms 级稳定 rank |
| 时间尺度 | 单次事件内部，< 100 ms |
| 空间尺度 | SOZ 内部 contacts 之间，< 几 cm |
| 驱动机制假设 | 长时程 Hebbian 可塑性雕刻的固定连接拓扑 |
| 物理对象 | 每个 contact 近场的局部 assembly 的 ~2 Hz 快振荡相位 |
| 对应模型 | **老论文的 Hopfield-Kuramoto**（现状保留） |
| 证伪难度 | 很难。需要 per-channel 级严格排除"每个 contact 有内禀振荡器"假设 |
| 状态 | **接受现状**，PR-3 做 mixture / identity bias / n_part 分层的稳健性补丁 |

### Layer B（介观到宏观，分钟到小时，SOZ 为主但可能波及外部）：事件**之间**率调制

| 维度 | 内容 |
|---|---|
| 经验现象 | IEI 正序列相关 30/30；IEI 和 n_part 共同被慢变量驱动；rate 在发作邻域抬升；PSD 2 Hz "峰"是不应期伪影 |
| 时间尺度 | 分钟（PR-2 半衰期 ~108 s）到小时（PR-2.6 rate autocorr 到 8 h 仍正） |
| 空间尺度 | 主要在 SOZ 内（PR-1 exp 6B：SOZ dead-time 更短），但可能反映全脑状态 |
| 驱动机制假设 | 慢全局兴奋性 $S(t)$ 对兴奋性单元阈值/恢复时间的调制 |
| 物理对象 | SOZ 作为一个整体，它的"群体事件触发率"被慢变量调制 |
| 对应模型 | **现在是空的**。老论文的 adaptive Kuramoto 讲的是秒级 splay→sync，无法承载分钟-小时尺度的慢调制叙事 |
| 证伪难度 | 已经被 PR-2 / 2.5 / 2.6 / 2.7 证实 |
| 状态 | **需要新引入一个 Layer-B 模型** |

### 关键观察

Layer A 和 Layer B 的**物理对象本来就不是同一个东西**：

- Layer A 的 "振子"  = contact-level 的局部近场节律（秒级）
- Layer B 的 "事件发生机制"  = SOZ 作为一个 coarse-grained 兴奋性节点（分钟级）

一旦承认它们是**两层**，"两张皮"问题就变成"两层如何耦合"，而不是"两层能否合并"。合并是伪问题——它们根本就在不同的尺度本体论里。分层才是真问题。

---

## 2. 两层之间的耦合：Layer B 如何作用到 Layer A？

这是新叙事的逻辑关键。我提两个候选答案，它们对应不同的物理图像和不同的可检验预测。

### 2.1 Hypothesis H1：Layer B 是 Layer A 的"触发门"

**物理图像**：Layer A 的 Hebbian 编码网络一直在那里，它是**结构性**的（长时程可塑性雕刻的连接拓扑），但它不会自发放电。每次群体事件的**触发**由 Layer B 的慢变量 $S(t)$ 控制——$S(t)$ 高时触发率高，$S(t)$ 低时触发率低。触发后，Layer A 的 Hebbian 结构决定了 rank 顺序（stereotype propagation）。

**数学表述**：

$$\text{rate of ignition} = \lambda_0 \cdot g(S(t))$$

$g$ 是某个单调函数，$S(t)$ 是 OU-like 慢过程。每次 ignition 后，Layer A 的 Hopfield-Kuramoto 被一个短时 impulse 激活，跑一遍 splay trajectory（即一次 stereotype propagation），然后 Layer A 回到静息。

**这个图像下的预测**：
- IEI 统计由 $S(t)$ 决定（与 Layer A 结构无关）
- stereotype 强度由 Hebbian 矩阵决定（与 $S(t)$ 无关）
- **两者应该相互独立** —— 如果数据支持，这是对 H1 的强证据

**证伪方法**：检查"高 rate 窗口"和"低 rate 窗口"的 stereotype 强度（pairwise τ）是否相同。如果没差别 → H1 成立。

### 2.2 Hypothesis H2：Layer B 同时调制 Layer A 的耦合强度

**物理图像**：$S(t)$ 不只是触发门，还直接调制 Layer A 的耦合强度 $K(t) = K_0 \cdot (1 + \beta S(t))$。高 $S(t)$ 时 Layer A 更容易被锁相 → 参与通道数增加 → n_participating 上升，stereotype 更紧密。

**这个图像下的预测**：
- IEI 和 n_participating 共同上升（与数据一致）
- **stereotype 强度也随 $S(t)$ 上升** —— 这是 H2 的独特预测
- $S(t)$ 极端高值 → $K(t)$ 足够大 → Layer A 的 splay 失稳 → ictal transition

**证伪方法**：同上，检查高低 rate 窗口间 stereotype 是否有系统差异。如果 stereotype 随 rate 上升 → H2 成立。

### 2.3 H1 vs H2：这是一个可以做的实验

两者的判别只需要一个分析：

> 对每个 subject，把 IEI 按中位数分成"高率"和"低率"两段，分别计算 pairwise τ，做 paired Wilcoxon。

如果 τ 在两段无差别 → H1；如果 τ 在高率段显著更高 → H2。这是**一张图的事**，但它决定整个 narrative。我强烈建议在 PR-3 里把这个 analysis 加进去。

### 2.4 我的猜测

基于已有数据，我猜 H2 部分成立（stereotype 确实在高率窗口稍强，但差异不会很大），因为：
- PR-2.5 的 n_part ~ IEI 共调制强（0.742），说明 $S(t)$ 确实在影响 Layer A 的参与度
- 但 Fig 2 stereotype 跨 24h 稳定的事实说明 Hebbian 结构是主导的

所以真实图像可能是 **"H1 为主 + H2 的弱混合"**：$S(t)$ 主要扮演触发门，次要扮演耦合调制器。

---

## 3. Layer B 需要一个什么样的模型？

这是新叙事的高光部分，也是老论文真正缺的那块。

### 3.1 设计约束

Layer B 模型必须满足：

1. **物理对象是 SOZ 整体而非单 contact** —— 所以不再是 N 节点的 Kuramoto，而是少数状态变量
2. **时间尺度是分钟到小时** —— 所以动力学必须有慢变量
3. **状态空间包含 interictal / peri-ictal / ictal** —— 所以必须有 bifurcation 结构
4. **可以用 $S(t)$ 同时解释 rate 升高和 n_part 升高** —— 所以状态必须和 event 生成过程耦合

### 3.2 候选模型：最低限度的 fast-slow excitable unit

我建议用**一个**兴奋性单元（FHN / 或 Epileptor 的简化版）来代表 SOZ 整体，不是 N 个。形式上：

$$\dot{x} = f(x, y) + I_0 + \eta(t) \quad \text{(fast activity variable)}$$
$$\dot{y} = \epsilon(x, y) \quad \text{(refractory / recovery variable, medium)}$$
$$\dot{z} = (\bar{z} - z)/\tau_z + \sigma_z \xi(t) \quad \text{(slow excitability variable } S(t)\text{)}$$

其中 $z$ 调制 $f$ 的 excitability（比如作为 $I_0$ 的偏置）。每当 $x$ 跨过阈值就记录一次"群体事件触发"，然后交给 Layer A 去走 stereotype。

**这个模型是 fast-slow 结构的**，属于 Epileptor 家族的简化版本。它天然产生：
- 不应期（$y$ 的恢复时间）
- 非平稳 rate（$z$ 慢漂移）
- IEI 正序列相关（$z$ 的自相关直接映射到触发时间统计）
- Peri-ictal rate elevation（$z$ 漂移到高值区间）
- 可选的 ictal transition（$z$ 越过 saddle-node 临界点）

**重要的是 $\tau_z$ 可以直接从 PR-2 数据校准到 ~100 s 或多层叠加**。

### 3.3 Layer A 和 Layer B 的接口

接口简单：

- Layer B 产生"触发事件时间戳" $\{t_k\}$，由 $x(t)$ 的阈值穿越决定
- 每个 $t_k$ 触发一次 Layer A 的短时激活：给 Hopfield-Kuramoto 一个 impulse，它跑一次 splay trajectory，产生一个 rank 向量
- （可选 H2）Layer B 的 $z(t_k)$ 被读出，调制 Layer A 在该次事件中的耦合强度，影响参与通道数和 stereotype 精度

就这样。两层各管各的，通过触发时间戳耦合。

---

## 4. 新的叙事主线

```
Layer A (老论文，微观): 秒级、SOZ 内部、Hebbian 结构
   → 解释：为什么每次事件的 rank 顺序稳定
   → 机制：长时程可塑性雕刻的连接拓扑
   → 证据：Fig 1-2 + PR-3 稳健性补丁

Layer B (新引入，介观-宏观): 分钟到小时、SOZ 整体、兴奋性慢调制
   → 解释：为什么事件什么时候来、为什么发作前 rate 升高
   → 机制：慢变量 S(t) 调制 SOZ 的触发率
   → 证据：PR-2 / 2.5 / 2.6 / 2.7 全部

Layer A × Layer B 耦合 (H1 / H2):
   → S(t) 主要作为触发门，次要调制耦合强度
   → 可通过 PR-3 的 "high-rate vs low-rate τ 对比" 直接检验
```

这套叙事的好处：

1. **承认现状**。老论文 Layer A 不需要动，你省去了合作者接受"推翻旧模型"的成本
2. **尊重数据**。Layer B 是新数据的自然归宿
3. **避免 FHN 重写风险**。我不再要求把 Kuramoto 换掉——Layer A 保持是 Kuramoto，Layer B 是独立的兴奋性单元，两者通过接口耦合
4. **直接连接 Epileptor 传统**。Layer B 是 Epileptor 简化版，这让整个新 narrative 嵌入已有文献，审稿人熟悉
5. **Torus 直觉被保留但降级**。Torus 是 Layer A 的状态空间（老论文本来就是这样），不强求用它来解释 Layer B 的慢调制——后者本来就不在 torus 上

---

## 5. 具体要做的事

**Layer A (PR-3，已在计划中)**
- mixture detection (spectral clustering on pairwise τ)
- centered rank control（加 SOZ source node erasure 诊断）
- n_participating 分层 + mixed effects model

**Layer A × Layer B 耦合判别 (新增)**
- 高率 vs 低率窗口的 τ 对比 → H1/H2 判别，一张图

**Layer B 新建模 (新章节)**
- 单兴奋性单元 fast-slow model（FHN + OU slow $z$），参数从 PR-2 校准
- 仿真得到 IEI 序列，跑 PR-2 / 2.5 分析，与真实数据对照
- 让 $z$ 偶发大偏移，再现 peri-ictal rate elevation（PR-2.7 exp 7G）

**Discussion 加一段 Torus 几何**
- 仅用于解释 Layer A 的 splay submanifold，不强行拉 Layer B

**最小工程量**：PR-3 已在日程；Layer B 模型是单变量 ODE + OU，一天就能跑；H1/H2 判别是一个 analysis 函数。整个补充章节 1 周内可完成。

---

## 6. 诚实地说这套方案的局限

1. **两层之间的接口是唯象的**，不是从第一性原理推导。这是 multi-scale modeling 的普遍问题，不独有
2. **Layer B 的 $S(t)$ 仍然是经验抽象**。生理学对应待定（觉醒态 / 血药浓度 / E/I 漂移）
3. **不解释 non-SOZ 事件**。Layer B 模型针对 SOZ；non-SOZ 要不要独立建模，看数据是否支持
4. **H1 和 H2 可能都部分成立**，不一定能干净判别。但至少可以给出权重估计

---

## 7. 一句话总结

> 不要强行把 intra-event 和 inter-event 两套现象缝成一层。它们是不同尺度、不同物理对象。保留老论文 Layer A 的 Kuramoto 叙事（现状可接受），新增 Layer B 的 fast-slow 兴奋性单元叙事（承载 PR-2 / 2.5 / 2.6 / 2.7 的全部慢调制发现），两层通过"触发时间戳 + 可选耦合调制"接口耦合。一个简单的 "high-rate vs low-rate τ 对比" 分析可以判别两层之间耦合的类型（H1 纯触发 vs H2 触发+耦合调制）。
