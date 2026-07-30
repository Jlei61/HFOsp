# Topic 5 / Figure 6：结构化 low-rank RNN 的跨状态动力学解释（v0.5）

**日期**：2026-07-25
**状态**：执行合同
**继承**：v0.4 的34人数据、masked rank、事件构造、80/20拆分、发作target与
泄漏控制保持不变
**取代**：v0.4 中“必须先证明全秩GRU超过非递归模型，才允许进入low-rank”的
模型层级与成功门

---

## 1. 唯一主问题

本分析不是为了证明RNN预测性能优越，也不是为了提高AUC。主问题是：

> 一个受约束、低维、可解释的递归动力系统，能否复现患者特异的间期触点传播
> 分布；该系统内部的低维模式及其触点读出，能否同时解释 clinical onset 附近
> 的静态发作早期能量场？

主证据链固定为：

```text
interictal contact-rank events
    → structured low-rank recurrent dynamics
    → free-running reproduction of contact propagation fields
    → identifiable latent modes / trajectories / mode lesions
    → frozen mapping to early-ictal static energy field
```

---

## 2. 模型层级

### 2.1 全秩GRU只作参考

当前正在运行的全秩GRU保留，用途仅限：

- 验证数据与训练器能够学习合法的 next-set / STOP 任务；
- 给自由生成分布提供 unconstrained reference；
- 给 low-rank 模型提供“是否明显损失拟合能力”的上界参考。

全秩GRU不承担以下主结论：

- RNN比经验分布更有用；
- RNN比所有静态模型预测更准；
- 更高的预测指标等同于更真实的动力学。

### 2.2 主模型改为 low-rank leaky RNN

第一版主模型不再使用GRU gate，而使用可直接分析的离散 leaky RNN：

```text
h[t+1]
  = (1 - alpha) h[t]
    + alpha * tanh(
        D h[t]
        + U V^T h[t]
        + B x[t]
        + b
      )
```

其中：

- `x[t]` 是当前 recruitment set 的 permutation-invariant pooled contact
  embedding；
- `D` 是受约束的稳定 self-decay / diagonal term；
- `U V^T` 是 rank `r` 的共享递归相互作用；
- `r ∈ {0,1,2,3,4}`；
- contact-query decoder 与 v0.4 的 next-set / STOP 输出保持一致；
- recurrence 只沿单个间期事件内部的 pseudo-time 演化，不跨事件。

`rank=0` 是只有稳定衰减、没有共享递归模式的必要对照。

### 2.3 生物结构版本

只有 low-rank 主模型能复现间期分布并保留跨状态解释时，再加入第二版结构：

- Dale-sign 或 E/I unit partition；
- 显式负向稳定/self-inhibitory term；
- 局部抑制强度或稀疏结构。

这一步回答“哪种受约束结构足以产生观察到的传播场”，不能声称恢复了真实细胞
类型或真实微回路。

---

## 3. 训练任务的角色

next-set / STOP 仍是自监督训练任务，因为它能迫使模型学习单个间期事件内部的
传播规则。但一步预测性能只是：

- 训练是否工作；
- 模型是否保留局部条件结构；
- 排除完全退化模型。

它不是论文的最终科学终点。

不再要求：

- low-rank RNN必须超过直接经验分布；
- full-rank RNN必须显著超过所有非递归baseline；
- 用AUC或分类准确率作为主成功标准。

---

## 4. 间期侧主成功标准

### 4.1 自由生成复现

每名患者、每个seed自由生成事件，比较：

- 每触点参与概率；
- conditional rank distribution；
- early / middle / late rank preference；
- pairwise precedence matrix；
- 参与触点数与事件长度分布；
- event × contact rank matrix 的多路径结构。

主判断是模型误差是否落在患者自身经验分布的 split-half variability 内，而
不是是否超过经验分布。

### 4.2 最小充分rank

对 `r ∈ {0,1,2,3,4}`：

1. 先判断间期分布复现是否达到经验变异范围；
2. 再判断跨seed稳定性；
3. 选择满足上述条件的最小rank；
4. full-rank GRU只作为拟合上界。

若 rank 1–2 已足够，这是主要的简约动力学结果；不能把rank数直接等同于A/B
模板数。

---

## 5. 内部动力学分析

至少输出以下四类分析。

### 5.1 隐状态轨迹

- teacher-forced真实事件轨迹；
- free-running生成轨迹；
- 按事件路径无监督着色；
- 不以A/B label监督训练。

检查大量事件是否经过少数稳定通道、分支或慢方向。

### 5.2 递归模式与触点读出

对每个低秩模式保存：

- `U` 与 `V` mode；
- mode time course；
- contact input loading；
- contact output/readout loading；
- 与触点参与率、平均rank、precedence axis 的关系。

### 5.3 模式消融

逐个将某个 low-rank mode 置零，再自由生成，量化：

- participation field 改变；
- rank distribution 改变；
- 传播路径与precedence改变；
- 跨状态发作能量解释改变。

只有同一mode同时影响间期传播复现和发作期读出，才允许称为 cross-state
explanatory mode。

### 5.4 吸引结构边界

允许描述稳定通道、分支、低维流形或固定点附近动力学，但必须由真实轨迹、
Jacobian / local stability 或干预结果支持。不能仅凭二维投影把轨迹称为
attractor。

---

## 6. 发作期如何进入

发作侧仍使用冻结的：

```text
clinical onset [0,10] s
BB 1–150 Hz baseline-robust-z static energy field
```

这不是逐秒传播序列。

### 6.1 Primary：冻结模式的静态场解释

在读取发作target前冻结：

- low-rank recurrence；
- patient-local contact offsets；
- contact rank distribution；
- mode-to-contact readout/loading。

比较每次seizure的静态能量场与：

- low-rank模型生成的 early-rank field；
- 各mode的contact loading field；
- 模式消融前后的生成field；
- empirical interictal rank field；
- participation-only与geometry controls。

### 6.2 允许的结论

成功时允许写：

> A small number of recurrent modes was sufficient to reproduce
> patient-specific interictal propagation fields, and the corresponding
> frozen contact loadings explained the static early-ictal energy field.

不要求low-rank模型击败经验分布。若经验分布与模型均能解释发作场，而low-rank
模型用少数模式复现并提供可干预的内部动力学，仍然是成功结果。

---

## 7. 新的go / no-go

| 层级 | Go条件 | 失败后的处理 |
|---|---|---|
| 训练可用 | next-set / STOP不退化，free rollout可结束且不重复触点 | 修训练器，不谈科学 |
| 间期复现 | 至少一个 `r>0` 在主要分布指标上进入经验split-half范围 | low-rank动力学分支停止 |
| 低维充分 | 某个小rank跨seed稳定且不明显劣于全秩参考 | 选择最小充分rank |
| 跨状态解释 | 冻结mode/contact field超过label-shuffle并保留于mode lesion | 支持共享低维组织 |
| 生物结构 | E/I或抑制约束保留上述三层结果 | 只称结构充分，不称真实回路恢复 |

full-rank GRU是否显著超过非递归模型，不再是启动low-rank的硬门。

---

## 8. 当前执行裁决

- 让正在运行的全秩GRU多seed任务完成，作为参考上界；
- monitor继续只监控工程状态，不用预测增益决定提前停止；
- 同时实现 low-rank leaky RNN；
- 全秩参考完成后立即并行运行 `rank={0,1,2,3,4}`；
- 在 low-rank间期分布与mode artifacts冻结前，不读取发作target；
- paper-ready主图围绕“低秩模式—间期传播场—发作早期静态场—mode lesion”
  组织，不围绕AUC或模型排行榜组织。
