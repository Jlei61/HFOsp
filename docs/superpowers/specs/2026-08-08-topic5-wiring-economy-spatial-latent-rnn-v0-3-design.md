# Topic 5 — Wiring-Economy Spatial Latent RNN (WE-SLP-RNN v0.3) 设计合同

状态：LOCKED 2026-08-08。延续 SLP-RNN v0.1（`docs/archive/topic5/spatial_latent_propagation_rnn_v0_1_2026-08-06.md`）
与 SPO-RNN v0.2（`docs/archive/topic5/spatial_propagation_operator_rnn_v0_2_2026-08-07.md`）。
分支 `codex/topic5-wiring-economy-slp-rnn-v0-3`，工作树 `.worktrees/topic5-we-slp-rnn`。

---

## 0. 这一版在问什么（朴素话）

我们给一个通用循环网络一份**有限的连线预算**——全部可能的单元对里只准开 10%——把它的每个单元钉在患者自己那张二维组织平面上的固定位置，然后只用一件事训练它：
间期事件里哪个触点先响、哪个后响。训练过程中不断掐掉最弱的连接，再补回同样数量的新连接，
补的时候**离得近的更容易被试到**，但远连接不禁止——对任务有用就能活下来。

对照是同一个网络、同样的连接数、同样的读出、同样的训练预算，只把"补连接时偏向近处"换成"完全随机补"。

问题：**在任务性能和空间布线成本之间联合优化，会不会自己长出一张稀疏、分块、有分工、
并且真的支撑间期传播序列预测的循环拓扑？**

英文表述：*Do generic spatial wiring constraints shape a task-optimized recurrent network into an
interpretable topology that supports interictal propagation sequences?*

### 0.1 结论层级已从"边身份"下调

v0.1 的恢复门（在合成数据上，真答案已知）判定：**哪些连接存在认不出**（AUC 0.482，门槛 0.60）、
**活动整体往哪走认不出**（7 比 3）、**各组织块往前推的相对排序认得出**（7/7，p=0.008）。
把观测比 0.25→0.75、连线成本 0→10、隐藏维翻 4 倍，前两层都停在随机水平。

因此本版**不把"恢复患者真实的每一条连接"当成功判据**。结论层级固定在：

    连接倾向 → 网络拓扑 → 功能模块 → 序列计算

### 0.2 上一版三条硬约束，本版必须继承

1. **观测足印是先天限制。** 一个触点对半径 ≥6mm 的组织取加权平均（核宽 σ = max(2mm, 半个最近邻触点间距)，
   硬截断 3σ）。核宽扫描证实：读 12mm 时方向相反的两个生成器 92% 名次相同；缩到 1.5mm
   （1/4 足印）方向仍只占 7.5%。→ **任何靠"整体行进方向"承重的读数在这套观测下先天不成立。**
2. **固定训练轮数 = 比谁收敛得快，不是比模型。** v0.1 给所有臂 95 轮，而组织场臂每位患者要跑
   157–316 轮；被截断的恰好是承载否定结论的那条臂，"组织场跟静态基线没区别"因此作废
   （+0.0001 → +0.0530）。
3. **患者内 vs 患者间的相似性必做未训练对照。** 节点位置本身按患者采样，任何从这些位置算出的量
   天然造出正的患者内外差距（v0.1：学到的图 +0.1015，未训练的图在同样位置上就有 +0.0591，
   两者之差 p=0.070 不过线）。

---

## 1. 冻结边界

### 1.1 必须保留

| 内容 | 状态 |
|---|---|
| 状态在组织单元上，不在触点上 | 必须 |
| 局部 tied `Hᵀ` 注入 / `H` 读出，无触点直通 | 必须 |
| 标准通用循环单元（不是简化 tissue cell） | 必须 |
| 固定稀疏资源（edge count 训练全程不变） | 必须 |
| SET prune / regrow | 必须 |
| 距离偏置生长 | 必须 |
| wiring-cost 损失项 | 必须 |
| next-rank + STOP 任务 | 必须 |
| same-start autoregressive rollout | 必须 |
| topology / functional clustering / module lesion 分析 | 必须 |

### 1.2 不加入

发作数据、early-ictal loss、间期—发作状态转换、patient-specific latent substate、dynamic routing、
E/I 分群、慢变量、SNN 方程、SOZ 或 A/B 作为训练输入、可学习的 connection-rule generator、
**exact-edge recovery 作为成功 gate**、modularity/small-world/A-B-separation/homophily/lesion 作为损失项。

### 1.3 仅作 sensitivity（本轮不做）

ordinal distance delay（需要每个触点可靠的 peak latency）、seRNN communicability loss、
Akarca GNM cost–value 拟合（post-hoc supplementary）。

### 1.4 明确不加的自由读出臂

v0.2 收官时记录的"下一版第一问"是给空间受限模型配自由读出、量化差距归因。
**本版明确不做**（用户 2026-08-08 决定）。读出严格锁为 `固定核 × 单标量增益`。
后果：图里"不受约束 GRU"那一格只能作背景标注，**不能**用来归因差距来源。
写作时禁止出现"差距主要来自空间约束"这类未经该实验支持的归因。

---

## 2. 队列与几何

### 2.1 冻结队列 n=21

来自 v0.1 `INPUT_MANIFEST.json` 的 narrow 几何树（细 montage 严格包含粗 montage，15/21）：

```
epilepsiae_{1084,1146,1150,253,384,442,548,590,620,922,958}
yuquan_{chengshuai,huanghanwen,litengsheng,liyouran,pengzihang,songzishuo,xuxinyi,zhangbichen,zhangkexuan,zhaochenxi}
```

**所有平面都是用整段记录估的 → 回溯性。禁止写成"这套几何在记录前就能知道"。**

### 2.2 平面选择：共线者用共用平面，非共线者用各自平面

每位患者的两套间期传播模式各有一条"从早到晚"的方向轴 `u_A`、`u_B`。
共线判据沿用已冻结的宽松主定义 `|cos(u_A,u_B)| ≥ 0.50`（夹角 ≤ 60°），
判定与平面均来自现成 artifact：

    results/interictal_propagation_masked/template_gradient_fields/per_subject/<subject>.json
      .interictal_field.planes.{own_a, own_b, shared}

| 组 | n | 平面 | 每位患者拟合数 |
|---|---|---|---|
| 共线 | **11**（1084, 1146, 384, 548, 590, 958, chengshuai, huanghanwen, litengsheng, pengzihang, zhaochenxi） | `planes.shared` | 1（**训练用两模式全部事件**） |
| 非共线 | **10**（1150, 253, 442, 620, 922, liyouran, songzishuo, xuxinyi, zhangbichen, zhangkexuan） | `planes.own_a` / `planes.own_b` | 2（**各自只训练本模式事件**） |

**每条臂 31 个拟合。** 每个训练单元记录 `fit_scope ∈ {shared_all, own_a, own_b}`。

**统计纪律（硬合同）**：任何队列级统计的单位是**患者**。非共线患者的两个拟合
**必须先在患者内取平均，再进入队列检验**；否则这 10 人拿到双倍权重。
不同 `fit_scope` 的绝对分数**不可互比**（训练集大小与任务难度不同）；
只允许比较同一拟合内的臂间配对差。

### 2.3 跨模式面板的分母固定为 11

"同一张连线图既能生成 A 又能生成 B"和"删掉一个模块对 A 和 B 的伤害不一样"
**只在一个模型内部才成立**。非共线患者拆成两个模型后，删 A 模型的模块当然只影响 A。
→ **主图第二格（分模式 same-start rollout）与第六格（分模式 module lesion）的分母固定为 n=11**，
写死在结果 JSON 的 `n_cross_mode_patients` 字段里。拆两个模型买到的是第三、四、五格。

### 2.4 观测算子与单元几何：沿用 v0.1 口径

平面坐标 `planes.*.points` 是归一化的，真实毫米 = `points × scale_mm`。

- 核宽 `σ = max(2.0 mm, 0.5 × 最近邻触点间距中位数)`，硬截断 `3σ`，行归一化高斯 → `H ∈ R^{C×M}`。
- 单元数 `M = min(64, max(24, 4C))`，最远点采样于"距任一触点 ≤ 3σ"的区域内，
  长到每个触点至少看到 3 个单元为止，上限 192。`NODE_SEED = 20260808`。
- **不使用** gradient-field artifact 自带的 `sigma`。理由：观测足印是 v0.1/v0.2 已经量化过的量
  （半径 ≥6mm，与场位移 5.4mm 的比值 0.54，17/21 <1），这一轮不该在它上面再叠一个没量过的变化。

触点集合：数据集触点 ∩ 平面 `contact_order`。**已逐个核对：21/21 两边集合完全相同。**

### 2.5 事件与切分

来源 `results/topic5_interictal_rank_distribution/dataset_v0_4`（封条：`target_values_read=False`、
`ab_or_kmeans_labels_read=False`；加载函数在清单不合规时抛错）。
名次密化后丢弃 `n_ranks < 2` 的事件。切分沿用 v0.1：train80 内按时间序 `development_split(0.15, 0.15)`
→ train / validation / test。**旧 heldout20 已被更早的 RNN 开发烧掉，本版不用。**

### 2.6 A/B 标签接入（封条外，硬门）

标签只用于**训练后**的分层评价与可视化，不进损失、不进输入。接入路径已验证：

```
bools        = load_subject_propagation_events(<lagPat dir>)['bools']        # (C_all, N_all)
valid_idx    = _valid_event_indices(bools, min_participating=3)              # 长度 = len(labels)
label_full   = full(N_all, -1);  label_full[valid_idx] = adaptive_cluster.labels
label_ds     = label_full[dataset.event_source_index]
```

**硬门（不通过则大声失败，不得回退）**：
1. `len(valid_idx) == len(adaptive_cluster.labels)`（已验证 21/21 全过）；
2. `dataset.event_source_index` 无重复、且 `max < N_all`；
3. 选中 block 的事件数之和 == 数据集事件数；
4. 报告 `label_ds >= 0` 的比例，低于 0.98 记入单元 JSON 并在收官表标红。

⚠️ **`adaptive_cluster.labels` 索引的是"有效事件"子集，不是全部事件**；直接
`labels[event_source_index]` 在 21 人里有 3 人越界、其余人静默错位。这是本版最易产生
"看起来完全合理的假标签"的地方。

模式编号：`0`/`1` 映射到 A/B 的规则 = 与 `planes.own_a` 使用的模板一致；
每位患者在 provenance 里存 `mode_to_template = {"0": "a", "1": "b"}` 或其反向，并核对
`adaptive_cluster.clusters[k].template_rank` 与 artifact `rank_a/rank_b` 的 Spearman 相关取最大者。

---

## 3. 模型

### 3.1 主循环单元：掩码 leaky RNN

`h_t ∈ R^M`（**每个组织单元一个标量**），`x_t ∈ {0,1}^C`，注入 `u_t = Hᵀ x_t`。

```
pre_t = a ⊙ u_t + (M ⊙ W) h_{t-1} + b
h_t   = (1 - κ) h_{t-1} + κ · tanh(pre_t)
```

- `M ∈ {0,1}^{M×M}`：learned sparse recurrent mask，无自环。
- `W ∈ R^{M×M}`：唯一的循环矩阵 → **掩码乘上去就是那张图**。
- `a ∈ R^M`：逐位置输入增益。**输入层不允许 dense M×M 矩阵**，所有跨单元信息必须经 `M⊙W`。
- `κ = sigmoid(κ_logit)`：单个可学标量泄漏项（离散 CTRNN 形式，稳定性用）。

**选择理由（记录在案）**：这一轮的产品是"学出来的那张图长什么样"。门控网络有三个循环矩阵，
一条连接得用三个数的平方和开方合成强度，"一条边"混着三种角色。普通 RNN 只有一个矩阵，
模块划分、边长分布、删模块每一句话都指着同一个对象。事件步数 ≤ 触点数（多数 8–16 步），
梯度问题在这个长度上不发作。空间嵌入 RNN 那篇（Achterberg et al. 2023）用的也是普通 RNN。

### 3.2 复核循环单元：掩码 GRU

```
r_t = σ[a_r ⊙ u_t + (M⊙U_r) h_{t-1} + b_r]
z_t = σ[a_z ⊙ u_t + (M⊙U_z) h_{t-1} + b_z]
ĥ_t = tanh[a_h ⊙ u_t + (M⊙U_h)(r_t ⊙ h_{t-1}) + b_h]
h_t = (1-z_t) ⊙ h_{t-1} + z_t ⊙ ĥ_t
```

**同一个 `M`** 作用于三个循环矩阵。

### 3.3 边强度

| 单元 | `S_ij` |
|---|---|
| RNN | `|W_ij|` |
| GRU | `sqrt(U_r,ij² + U_z,ij² + U_h,ij²)` |

- 边是否存在由 `M_ij` 决定；prune 按 `S_ij` 排序；
- 无权拓扑分析用 `M_ij`；加权分析用 `M_ij · S_ij`。
- **不把任一矩阵元的正负号解释为生理兴奋或抑制。**

### 3.4 读出与停止

```
ℓ_{t+1} = b_contact + α_out · (H h_t)          # 已招募触点置 -inf
ℓ_STOP  = f_STOP([mean(h_t), max(h_t), t_norm, recruited_fraction])
```

`α_out` 是**单个标量**。`f_STOP` 是 2 层小 MLP（隐藏 16）。
`b_contact` 每个触点一个标量偏置（与 v0.1 一致，`use_contact_bias=True`）。

### 3.5 隐状态维度：正式锁 1 维

正式模型每个组织单元 1 维。**开发探针**：8 位预注册患者上跑一次每单元 2 维
（掩码按块施加 `M ⊗ 1_{2×2}`，图约束仍只限制跨单元通信），唯一作用是排除
"四个模型全打平是因为状态太窄"这一解释。探针结果不进主张。

---

## 4. 图的形成：SET + wiring economy

### 4.1 固定稀疏资源

密度 `ρ = 10%`（活跃边数 = `round(ρ · M(M-1))`，训练全程不变）。
**`ρ` 是明确的算法容量设置，不是生理连接密度估计。禁止把"平均每单元约 N 条边"
解释成生理事实。** 主值 a priori 冻结在 10%（Zhang et al. 2025 的 recurrent 实验用 <10%）。

### 4.2 prune / regrow

每个 epoch（warmup 之后、freeze 之前）：

1. 按 `S_ij` 升序删除活跃边中最小的 `ζ` 比例；
2. 在非活跃边中按 `P_grow(j→i) ∝ 1/(d_ij + ε)` 采样等量新边（`ε = 0.1 mm`）；
3. 新边权重初始化为 **0**（梯度仍非零，可自行长出来）；
4. 活跃边总数恒定。

`ζ` 从 `ζ0 = 0.20` 按余弦退火到 **0**；退火结束后掩码冻结，**之后才开始早停判定**。

| 臂 | 初始掩码 | regrow 采样 |
|---|---|---|
| `SPATIAL_SET` | 按 `1/(d+ε)` 采样 | 按 `1/(d+ε)` |
| `RANDOM_SET` | 均匀随机 | 均匀随机 |

### 4.3 wiring cost

```
C_wiring = (1/|E|) Σ_ij M_ij S_ij (d_ij / d_0),     d_0 = 10 mm（全队列固定）
```

`d_ij` 是患者二维平面上的真实毫米距离（`points × scale_mm`）。
**不按患者自己的中位距离归一化** —— 1mm 在所有患者中代表同样物理长度。

### 4.4 总损失

```
L = L_next_rank + λ_STOP · L_STOP + η · C_wiring
```

`λ_STOP = 1.0`（沿用 v0.1）。`η` 在 8 位开发患者的 **validation** 切分上从
`{0.003, 0.01, 0.03, 0.1, 0.3}` 按布线-性能拐点选定后冻结，全队列共用。
**只用 validation 选，test 不参与选择。**

`RANDOM_SET` 也带同样的 `η · C_wiring`（否则两臂差的就不止一件事）。
`DENSE_TISSUE` 不带（无稀疏资源，wiring cost 会直接把它压成稀疏的）。

---

## 5. 模型矩阵

| 模型 | 循环层 | 作用 |
|---|---|---|
| `STATIC_CONTACT` | 无 recurrence | 触点长期参与率地板 |
| `DENSE_TISSUE` | dense 标准循环 + 同一个 `H` | tissue representation 能力上界 |
| `RANDOM_SET` | 固定稀疏、均匀随机 regrow + wiring cost | 稀疏 recurrence 基线 |
| **`SPATIAL_SET`** | 固定稀疏、距离偏置 regrow + wiring cost | **主模型** |

**核心比较**：`SPATIAL_SET − RANDOM_SET` — 在完全相同的循环单元、边密度、局部 `H`、
训练预算和任务下，空间布线经济是否改善性能并改变循环拓扑？

**第二比较**：`SPATIAL_SET − DENSE_TISSUE` — 空间组织的稀疏网络能否以更低布线成本
保持接近稠密网络的性能？

不扩展 contact graph / 固定局部 graph / SPO / E-I graph / architecture zoo。

---

## 6. 对照（本版主张的承重点）

按 `1/d` 生长的二维图**天生**高聚类、高模块度、边短、模块空间连成片。
不加对照，"涌现出模块化"不可证伪。三个对照全部必需：

| 对照 | 成本 | 回答 |
|---|---|---|
| **C1 初始同规则图**（训练前的 `M`） | 免费 | 生长规则本身能长出多少 |
| **C2 打乱目标的同规则训练**（`SPATIAL_SET_SHUFFLED`，标签在事件内随机置换名次） | 31 单元 | 删补动力学本身能长出多少 |
| **C3 保边长分布的重连**（最终图，保住每单元进出度 + 边长分布分箱，只打乱谁连谁） | 免费（纯分析） | 同样几何开销下任务额外买到多少 |

**判据：只有同时超过 C1 和 C2，才能说"任务塑造了拓扑"。只超过 `RANDOM_SET`
只能说"距离偏好塑造了拓扑"——那是生长规则的直接后果，不是发现。**

模块删除的随机对照必须是**同样大小、同样空间连续的一块区域**（在二维平面上做
区域生长采样），不能是散点——模块在平面上就是一块连续区域，删散点当然更不痛。

---

## 7. 任务与评价

训练：`x_{1:t} → x_{t+1} + STOP`，teacher-forced next-rank BCE + STOP BCE。

评价（**test 切分**）：

- **主指标**：held-out next-rank NLL（每个可用步的 masked BCE，patient-paired）。
- same-start autoregressive rollout，全事件长度。
- first-to-last rank profile。
- contact participation field。
- A/B held-out event 分层（分母见 §2.3）。

**生成纪律（v0.2 缺陷 3、5）**：
- 自由生成用 **argmax**，不用固定 0.5 阈值。v0.2 上固定阈值 0.5 配校准概率 ~0.07 导致
  "生成塌成长度 1"，那是阈值的行为被当成了模型的行为。
- 两套生成模式之间加**硬守卫**：`≥15%` 名次不同才算两套。守卫不过时，
  记录 `generator_degenerate=true`，且**该患者不进跨模式面板**——
  从没信息的生成器得出的"不可恢复"描述的是采样器，不是模型。

---

## 8. 训练纪律

| 项 | 值 / 规则 | 来源 |
|---|---|---|
| batch | `min(1024, ceil(n_train / 8))`，每 epoch 至少 8 次更新 | v0.1 缺陷 1 |
| **batch 不得当显存旋钮** | 同一患者不同臂 batch 必须相同；资源问题只能调并发 | v0.1 缺陷 1 |
| 设备 | 全队列同一设备。开跑前 benchmark CPU vs GPU，选定后写入 `RUN_CONTRACT.json`，中途不换 | v0.2 工程 |
| epochs | warmup 10 / rewire 40（`ζ` 退火）/ freeze ≤ 350 | — |
| 早停 | patience 12，`min_relative_improvement=1e-4`，**只在掩码冻结后计时** | v0.1 缺陷 |
| **收敛是进入分析的前置条件** | 撞上 epoch 上限的单元 `converged=false`，**不进任何分析**，在收官表单独列出 | v0.1 最大坑 |
| 种子 | `STATIC` 1、`DENSE_TISSUE` 1、`RANDOM_SET` 3、`SPATIAL_SET` 3（seed 0/1/2） | — |
| 原子写 | 先写 `.tmp` 再 rename；`DONE.json` 最后写 | v0.2 |
| 重复启动守卫 | 建清单时按**绝对路径**查在飞进程 + 输出目录 | v0.2（收紧过三次） |
| **不得编辑正在运行的 bash 脚本** | bash 按字节偏移恢复；插入行会让其后全部移位 | v0.2 缺陷 7 |
| 新鲜度检查 | 每个分析阶段核对输入 `DONE.json` 的 mtime 晚于上游、且覆盖当前队列全集 | v0.2 缺陷 7 |

### 8.1 单元账

| 批次 | 单元数 |
|---|---|
| η 扫描（8 开发患者 × 5 个 η，仅 `SPATIAL_SET`，1 种子） | 40 |
| RNN 主队列（31 拟合 × [1 static + 1 dense + 3 random + 3 spatial]） | 248 |
| C2 打乱目标对照（31 × 1） | 31 |
| GRU 复核（31 × 3 个循环模型 × 1 种子） | 93 |
| 2 维容量探针（8 患者 × 1） | 8 |
| **合计** | **420** |

可选敏感性（时间允许再做）：密度 5% / 20% × `{SPATIAL, RANDOM}` × 31 = 124。

---

## 9. 训练后分析

### 9.1 布线-性能 Pareto（描述性，不需要显著性）

每个拟合报告 `C_wiring` vs held-out next-rank NLL。
**跨患者必须画患者内配对差，不画绝对分数**——这个指标的绝对水平与触点数负相关
（ρ = −0.622，触点越多"大多数不参与"越好猜；v0.1 指标陷阱）。

结论形态：「用 X% 的连接、Y% 的总布线长度，达到稠密网络 Z 的预测水平」。

### 9.2 涌现拓扑（对照见 §6）

modularity Q（Louvain）、clustering coefficient、small-worldness、边长分布、
连接强度—距离关系、participation coefficient、connector nodes、长程边比例。
每一项都同时对 C1 / C2 / C3 / `RANDOM_SET` 报告。

### 9.3 功能分工

对 held-out 事件中每个组织单元的隐状态构造
`f_i = [rank-1 响应, …, rank-R 响应, Mode-A 偏好, Mode-B 偏好]`（Mode 分量仅 11 位患者有）。

三个检验，**每个都要未训练对照**：
- **空间聚集**：功能相似的单元在平面上是否更靠近（空间置换 null）；
- **结构聚集**：同一 Louvain 模块内的单元是否 early/late 偏好、A/B 偏好、招募场偏好更相似；
- **连接倾向**：**控制距离后**，相连的单元对是否功能相似性更高。

### 9.4 模块删除

Louvain 识别模块 → 删除整个模块的单元或其循环连接 → **不重训** →
测 next-rank NLL、STOP、rollout fidelity → 分别评价 Mode A / Mode B（n=11）→
与**同大小、同空间连续、同度数、同总权重、同平均距离**匹配的随机删除比较。

### 9.5 连接倾向的 post-hoc 描述

```
logit P(M_ij = 1) = β0 + β_d log d_ij + β_f S^func_ij + u_patient
```

⚠️ **必须报告为"给定被提议后的存活率"，不是原始连边概率。**
`SPATIAL_SET` 的边按 `1/d` 被提议，所以 `β_d > 0` 部分是构造出来的、不是发现的。
零假设 = "被提议的边随机存活"，即以生长提议分布为 offset。
`RANDOM_SET` 的提议分布不同，两者的最终图**不能直接比 β_d**（混淆了提议差异与存活差异）。

Akarca et al. 的 `P_ij ∝ D_ij^η K_ij^γ` 只作 Supplementary 的 learned graph 总结，不参与训练。

### 9.6 跨患者呈现

坐标归一化到 `s, h ∈ [-1, 1]`。每位患者输出 edge-density field、local/long-range map、
module map、rank-selectivity map、A/B preference map（n=11）、wiring-cost–performance 点、
modularity/clustering、`β_d`、`β_f`。
队列层展示 cohort mean / variance / per-patient points / across-seed reliability /
untrained 与 random-geometry 对照。
**统计单位始终是患者，不把 edge pairs 或 seeds 当独立样本。**

---

## 10. 预注册判词

| 门 | 内容 | 判据 |
|---|---|---|
| **G1** | 布线经济免费吗 | 描述性：`SPATIAL_SET` 的 `C_wiring` 与 NLL 相对 `DENSE_TISSUE` |
| **G2** | 空间偏好本身有用吗 | `SPATIAL_SET − RANDOM_SET` 患者内配对 Wilcoxon（n=21） |
| **G3** | 长出来的形状超出几何了吗 | 必须**同时**超过 C1 与 C2；只超过 `RANDOM_SET` 不算 |
| **G4** | 模块是必需的吗 | 模块删除 vs 同大小同连续区域随机删除，患者内配对 |
| **G5** | 功能分工存在吗 | §9.3 三项，每项对未训练对照 |

**打平时的措辞（预注册）**：
> 在同样的连接数下，让近处更容易连上既没有帮助也没有代价——这条约束是免费的，但不是必要的。

**禁止措辞**：
- ❌「空间不重要」/「间期传播不是空间的」——注入和读出都过空间核，
  `RANDOM_SET` 也不是空间盲的（v0.2 已进措辞守卫）。
- ❌「我们恢复了患者真实的传播 connectome」——v0.1 恢复门已判定边身份认不出。
- ❌「这张 graph 在发作时被复用」/「发作是 RNN state switching」——本轮不碰发作数据。
- ❌「差距主要来自空间约束」——自由读出臂本版不做（§1.4），归因实验不存在。
- ❌ 把 `ρ = 10%` 或"平均每单元 N 条边"说成生理事实。

**允许的最强结论**：
> Generic wiring-economy constraints shaped a spatially embedded recurrent network into
> task-relevant topological and functional organization supporting interictal propagation sequences.

---

## 11. 与 early-ictal 结果的关系

已有结果显示间期传播场与发作开始后 0–10s 的能量场有空间对应。这**只作为研究间期循环组织的上游动机**。

```
静态 field 分析：  间期与 early-ictal 空间结构相似
RNN 补充实验：     这种间期空间结构可由 wiring-constrained recurrence 产生
SNN：              具体生物物理机制可以产生类似传播
```

三者互相支持，**RNN 不承担跨状态证明**。主图不放 early-ictal 面板。

---

## 12. 主图六格

| 格 | 内容 | 分母 |
|---|---|---|
| **A** | 模型与 wiring rule 示意 | — |
| **B** | 间期自回归复现：观测 A / 生成 A / 观测 B / 生成 B（same starts）+ first-to-last rank profile | **11** |
| **C** | 群体预测与 wiring Pareto：Static / Dense / Random SET / Spatial SET（+ 自由 GRU 作背景标注） | 21 |
| **D** | 代表患者的涌现拓扑：初始图 / 训练中 / 最终图 / Louvain 模块 / 局部骨架与长程 connector | 1 |
| **E** | 跨患者连接倾向与功能组织：强度-距离、距离匹配的相连 vs 未相连功能相似性、modularity 与患者方差、rank/A-B selectivity map | 21（A/B 分量 11） |
| **F** | 模块删除 vs 匹配随机删除，整体 + Mode A/B fidelity | 21（A/B 分量 11） |

叙事线：`generic wiring rule → interictal prediction → emergent topology → functional specialization → module necessity`

---

## 13. 输出目录

```
results/topic5_wiring_economy_slp_rnn_v0_3/
├── RUN_CONTRACT.json           设备、冻结超参、队列、平面分组、η、密度
├── INPUT_MANIFEST.json         输入 SHA-256、A/B 标签接入核验
├── cache/<fit_id>/             plane / nodes / H / events / labels / provenance
├── per_subject/<fit_id>/<arm>/seed<k>/   metrics.json + graph.npz + DONE.json
├── analysis/                   pareto / topology / function / lesion / tendency
├── figures/
│   └── README.md               必须存在，中文逐图说明
└── CLOSEOUT.md
```

`fit_id = <subject>__<scope>`，`scope ∈ {shared, own_a, own_b}`。
