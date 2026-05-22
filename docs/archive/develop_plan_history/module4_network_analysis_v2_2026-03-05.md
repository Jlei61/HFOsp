# Module 4: src/network_analysis.py v2 Plan (Archived 2026-05-22)

> **归档说明**：这是 `docs/DEVELOP_PLAN.md` §4 模块4 (旧 L678–1688) 的 forward plan，撰写于 2026-03-05 左右。Phase A/B/C 均**未启动实施**——项目科学焦点已转向 Topic 1 (within-event dynamics)、Topic 2 (between-event timing)、Topic 3 (spatial SOZ modulation)，network_analysis 模块未被纳入任何 active PR。
>
> 完整 1010 行的 "宽建图 → 精剪枝 → 定方向" v2 设计（Simpson Index、XYZ 多维剪枝、方向注入、Stability 权重、Phase A/B/C 路线图、源空间远景）保留在此供未来 resurrect 时参考。**主 doc 只保留 5 行 pointer。**
>
> 当前主 doc (`DEVELOP_PLAN.md`) 对应位置：§4 模块4 pointer line。

---

### 模块4: src/network_analysis.py (开发中)

> **核心目标**：从 HFO 群体事件构建下一代癫痫网络，实现 SOZ 定位与传播路径预测。

---

#### 4.0 设计哲学与批判性前提

**"宽建图 → 精剪枝 → 定方向" 策略 (Build-Prune-Direct Strategy)**

```
┌────────────────────────────────────────────────────────────────────────┐
│                    癫痫网络构建流水线 v2                                 │
│               "宽建图 → 精剪枝 → 定方向"                                │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  [全通道池] (n_all ≈ 120)                                              │
│      │                                                                 │
│      ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │  Step 1: 宽建图 (Broad Graph Construction)                  │       │
│  │  边权 = Simpson Index (归一化共激活)                         │       │
│  │  "不再用原始共激活计数——校正基础率偏差"                       │       │
│  │  + Surrogate 显著性检验 → 剔除随机重合                      │       │
│  └──────────────────────────┬──────────────────────────────────┘       │
│                             │                                          │
│                             ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │  Step 2: XYZ 多维剪枝 (Multi-Dimensional Pruning)          │       │
│  │  X = HFO Rate (活跃度) → 节点是病理活动发生者               │       │
│  │  Y = Connection Entropy (特异性) → 剔除全脑噪声/参考伪迹    │       │
│  │  Z = FR/R Ratio (致痫性) 或 谱聚类(XYZ距离度量)            │       │
│  └──────────────────────────┬──────────────────────────────────┘       │
│                             │                                          │
│                             ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │  Step 3: 方向注入 (Direction Injection)                     │       │
│  │  Wilcoxon + 一致性检验 → 有向边                             │       │
│  └──────────────────────────┬──────────────────────────────────┘       │
│                             │                                          │
│                             ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │  Step 4: 复合权重 + 物理约束                                │       │
│  │  Simpson × Consistency × Stability                          │       │
│  │  + 容积传导剔除 (<10mm, Phase B)                            │       │
│  │  + 传播速度验证 (0.1-10 m/s, Phase B)                       │       │
│  └──────────────────────────┬──────────────────────────────────┘       │
│                             │                                          │
│                             ▼                                          │
│  [加权有向图 G(V, E, W)]  →  图论指标  →  SOZ/传播路径                  │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

**为什么是 "宽建图 → 精剪枝" 而不是旧版 "选节点 → 建骨架"？**

| 策略 | 致命缺陷 |
|------|---------|
| 先选节点再建边 | 选节点用的 co-activation 本身被基础率偏差污染，垃圾进垃圾出 |
| 先建骨架再选节点 | 骨架的边权（原始 count/ratio）无法区分"真同步"和"随机重合" |
| **宽建图 → 精剪枝** | Simpson 归一化消除率偏差 → XYZ 多维独立剪枝，每步可审计可回溯 |

---

#### 4.1 现有资产盘点 (Asset Inventory)

**✅ 已有数据（groupAnalysis.npz）**

| 数据 | 形状 | 物理意义 | 网络用途 |
|------|------|----------|---------|
| `ch_names` | (n_ch,) | 核心通道名 | 节点标识 |
| `coact_all_ch_names` | (n_all,) | **全通道名** | 扩大节点池 |
| `coact_event_ratio` | (n_ch, n_ch) | 共激活概率 | **骨架构建** |
| `coact_all_event_ratio` | (n_all, n_all) | **全通道共激活** | 扩大节点池 |
| `lag_raw` | (n_ch, n_events) | 质心时间（相对窗口起点） | **方向计算** |
| `events_bool` | (n_ch, n_events) | 通道参与mask | 事件过滤 |
| `event_windows` | (n_events, 2) | 事件窗口 [start, end] | 时间分段 |

**关键数据结构洞察**：

```python
# lag_raw 存储的是每通道相对于窗口起点的质心时间
# 要获得通道对 (i, j) 在事件 k 中的时滞：
lag_ij_k = lag_raw[i, k] - lag_raw[j, k]  # 负值 = i 领先 j

# 这是紧凑存储：O(n_ch × n_events) vs O(n_ch² × n_events)
# 运行时计算差值，空间换时间
```

**⚠️ 缺失数据（需要扩展）**

| 数据 | 形状 | 来源 | 优先级 | 用途 |
|------|------|------|--------|------|
| `electrode_distance` | (n_all, n_all) | MNI坐标计算 | **Phase B 阻塞** | 容积传导剔除、传播速度验证 |
| `hfo_type_per_event` | (n_ch, n_events) | 检测器输出 | **Phase B** | 病理加权 (FR 比例) |
| `tissue_label` | (n_all,) | FreeSurfer | Phase C | 灰/白质过滤（不硬剔除） |
| `mni_coords` | (n_all, 3) | 配准结果 | **Phase B 阻塞** | 3D可视化、距离矩阵 |
| `lead_field_matrix` | (n_ch, n_sources) | BEM 前向建模 | Phase C | 源空间 LFM 概率投影 |
| `sc_matrix` | (n_regions, n_regions) | HCP tractography | Phase C | SC-FC 耦合图 |

---

#### 4.2 宽建图 (Broad Graph Construction) — Simpson Index 归一化共激活

> "建图宽进，剪枝严出。" 先把所有有意义的连接保留下来，用统计学上正确的指标度量，再在下一步精确剪枝。

##### 4.2.1 为什么不能直接用 Co-activation Count 建边？

**致命缺陷：基础率偏差 (Base Rate Bias)**

假设节点 A 只有 10 次 HFO，节点 B 有 1000 次。A 的 10 次**全部**伴随 B 发生（100% 必然跟随）：

| 指标 | 计算 | 结果 | 问题 |
|------|------|------|------|
| Raw Count | $|E_A \cap E_B| = 10$ | 10 | 被 B 的 1000 次淹没，看起来"不重要" |
| Jaccard | $\frac{10}{10 + 1000 - 10}$ | 1% | 分母被 B 的规模稀释 |
| Dice | $\frac{2 \times 10}{10 + 1000}$ | 2% | 同上，稍好但仍被稀释 |
| **Simpson** | $\frac{10}{\min(10, 1000)}$ | **100%** | 完美捕捉"A 必然跟随 B" |

**在癫痫网络中，"必然跟随"比"共同活跃"更重要**：
- 真正的"起搏器"可能发放率不高，但每次发放都必然带动下游
- 传播通路节点的特征是：它的每次 HFO 都伴随上游 Source 发放
- Simpson Index 天然捕捉这种不对称的包含关系

##### 4.2.2 推荐边权指标：Simpson Index

$$W_{ij}^{Simpson} = \frac{|E_i \cap E_j|}{\min(|E_i|, |E_j|)}$$

**备选**（供对比验证）：

$$W_{ij}^{Dice} = \frac{2 \cdot |E_i \cap E_j|}{|E_i| + |E_j|}$$

| 指标 | 公式 | 偏向 | 适用场景 |
|------|------|------|----------|
| **Simpson** (推荐) | $\frac{|E_i \cap E_j|}{\min(|E_i|, |E_j|)}$ | 捕捉包含/跟随关系 | 癫痫传播网络（不对称耦合） |
| Dice (备选) | $\frac{2|E_i \cap E_j|}{|E_i| + |E_j|}$ | 对称，温和归一化 | 一般共激活网络 |
| Jaccard | $\frac{|E_i \cap E_j|}{|E_i \cup E_j|}$ | 惩罚不对称对 | ❌ 不推荐：稀释低频节点 |
| Raw Count/Ratio | $|E_i \cap E_j|$ 或 $/ N$ | 随率缩放 | ❌ 不推荐：高频节点主导 |

**默认选择 Simpson 的理由**：
1. 癫痫网络的核心问题是识别"谁跟随谁"，Simpson 正是度量包含关系的指标
2. Simpson 的不对称偏差会被 Step 2 的 HFO Rate (X) 剪枝校正 — 率太低的节点会被剔除
3. Simpson 对"沉默的共犯"友好 — 低频但 100% 跟随的节点不会被遗漏

##### 4.2.3 数据来源与向量化实现

**关键洞察**：所有需要的数据已存在于 `*_groupAnalysis.npz`：

```python
# 数据来源映射
intersection = coact_all_event_count[i, j]   # |E_i ∩ E_j|
event_count_i = coact_all_event_count[i, i]  # |E_i| (对角线 = 自身事件数)
event_count_j = coact_all_event_count[j, j]  # |E_j|
```

**向量化实现** (N=120, <1ms)：

```python
def build_broad_graph(
    coact_event_count: np.ndarray,    # (n_all, n_all) 共激活事件计数矩阵
    method: str = 'simpson',          # 'simpson' | 'dice'
    significance_mask: Optional[np.ndarray] = None,  # surrogate 检验结果
) -> np.ndarray:
    """
    从共激活计数矩阵构建归一化边权图。

    Simpson: W_ij = |E_i ∩ E_j| / min(|E_i|, |E_j|)
    Dice:    W_ij = 2|E_i ∩ E_j| / (|E_i| + |E_j|)

    Returns: (n_all, n_all) 对称边权矩阵, 值域 [0, 1]
    """
    intersection = coact_event_count.astype(np.float64).copy()
    events_count = np.diag(coact_event_count).astype(np.float64)  # |E_i|
    np.fill_diagonal(intersection, 0.0)

    if method == 'simpson':
        denom = np.minimum.outer(events_count, events_count)
    elif method == 'dice':
        denom = np.add.outer(events_count, events_count) / 2.0
    else:
        raise ValueError(f"Unknown method: {method}. Use 'simpson' or 'dice'.")

    W = np.divide(
        intersection, denom,
        out=np.zeros_like(intersection),
        where=denom > 0,
    )

    # 对称化（Simpson 可能因浮点不完全对称）
    W = np.maximum(W, W.T)

    # 显著性门控（可选）
    if significance_mask is not None:
        W[~significance_mask] = 0.0

    np.fill_diagonal(W, 0.0)
    return W
```

##### 4.2.4 Surrogate 显著性检验（保留，逻辑不变）

> 共激活的"统计显著"不等于"物理真实"。即使用了 Simpson 归一化，也必须验证观测值是否显著高于随机。

```python
def surrogate_significance_test(
    events_bool: np.ndarray,       # (n_ch, n_events) 参与mask
    n_surrogates: int = 200,       # 替代数据集数量
    p_threshold: float = 0.05,     # 显著性阈值
) -> np.ndarray:
    """
    独立循环平移各通道事件序列生成替代数据集，
    验证真实 Simpson/Dice 共激活是否显著高于随机预期。

    Returns: (n_ch, n_ch) bool — 显著性 mask
    """
    # 实现逻辑同之前：circular shift → 重算 → p-value
    ...
```

##### 4.2.5 宽建图的设计约束

**⚠️ 关键原则**：

- ✅ **宽进**：此步不做任何节点剔除，保留所有有 HFO 的通道
- ✅ **归一化**：Simpson/Dice 消除基础率偏差
- ✅ **统计门控**：Surrogate 剔除随机重合边（可选但推荐）
- ❌ **不做阈值剪枝**：不设 `min_coact` — 那是 Step 2 的活
- ❌ **不做节点选择**：不在这里用谱聚类 — 那也是 Step 2 的活
- ❌ **不做距离约束**：Phase A 无 MNI 坐标，Phase B 再加

**输出**：`W_broad` — (n_all, n_all) 归一化的对称边权矩阵，值域 [0, 1]

---

#### 4.3 XYZ 多维剪枝 (Multi-Dimensional Pruning) — 从广泛图中提取病理网络

> "建图宽进，剪枝严出。" 三个正交维度，每个维度瞄准一类特定的噪声源。

##### 4.3.1 三维度框架总览

```
              高                    ┌──────────────────────┐
               │                    │  SOZ 核心 (保留)      │
               │       ┌────────────┤  高率 + 低熵 + 高 Z   │
    X: HFO    │       │            └──────────────────────┘
    Rate       │       │
   (活跃度)    │       │    ┌──────────────────────┐
               │       │    │  参考伪迹 (剔除)      │
               │       │    │  高率 + 高熵           │
              低       │    └──────────────────────┘
               ─────────┼──────────────────────────────→
              低        │                            高
                 Y: Connection Entropy (特异性)
```

| 维度 | 指标 | 物理意义 | 剪枝方向 | Phase |
|------|------|----------|----------|-------|
| **X (Activity)** | HFO Rate ($events/min$) | 节点是否是病理活动的活跃发生者 | 保留 $X > X_{min}$ | **A** |
| **Y (Specificity)** | Connection Entropy $\hat{H}_i$ | 连接是特异性的还是全脑弥散的 | 保留 $\hat{H} < H_{max}$ | **A** |
| **Z (Epileptogenicity)** | FR/R Ratio 或 谱聚类(XYZ距离) | 节点的致痫性特异度 | Phase A: 谱聚类; Phase B: FR比例 | **A/B** |

##### 4.3.2 维度 X — HFO Rate (活跃度)

$$X_i = \frac{|E_i|}{T_{recording}} \quad (\text{events/min})$$

- **物理意义**：节点是否产生足够多的 HFO 来被纳入网络分析
- **剪枝逻辑**：$X_i \geq X_{min}$
- **默认阈值**：`min_rate = 0.5 events/min`（每2分钟至少1次 HFO）
- **⚠️ 不要设太高**：真正的"起搏器"可能发放率不高但每次必然带动下游（Simpson 已捕捉这种关系）

##### 4.3.3 维度 Y — Connection Entropy (特异性) 🔑 核心创新

**定义**：给定节点 $i$ 在宽建图 $W$ 中的连接权重分布：

$$p_{ij} = \frac{W_{ij}}{\sum_{k \neq i} W_{ik}}, \quad H_i = -\sum_{j \neq i} p_{ij} \ln p_{ij}$$

**归一化熵**（映射到 [0, 1]）：

$$\hat{H}_i = \frac{H_i}{\ln(N_{neighbors,i})}$$

其中 $N_{neighbors,i}$ = 节点 $i$ 的非零连接数。

**物理解释**：

| $\hat{H}_i$ | 含义 | 网络角色 | 判定 |
|---|---|---|---|
| **≈ 0** | 连接集中于1-2个节点 | 高度特异的"共犯关系" | ✅ 保留（局灶性传播通路） |
| **0.3 - 0.6** | 中等分散 | 有选择性的 Hub | ✅ 保留 |
| **≈ 1.0** | 均匀连接所有节点 | 全脑同步（伪迹/噪声） | ❌ 剔除 |

**为什么 Connection Entropy 是剔除 Global Artifacts 的"神技"**：

Reference contamination 的数学特征：一个通道因共参考电极而与所有通道产生虚假"共激活"。在 Simpson 空间中，这个通道与每个其他通道的 Simpson 值都 > 0（因为它的每次 HFO 都"伴随"很多通道）。**但它的连接分布接近均匀** → $\hat{H} \approx 1.0$。

真正的病理通道只与网络内的特定"共犯"高度同步 → $\hat{H}$ 显著低于 1.0。

这比传统的"剔除与太多通道连接的节点"更精确——它不关心你连了多少通道，而关心你的连接是否有**选择性**。

**剪枝逻辑**：$\hat{H}_i < H_{max}$，默认 `max_entropy = 0.85`

```python
def compute_connection_entropy(W: np.ndarray) -> np.ndarray:
    """
    计算每个节点的归一化连接熵。

    Parameters
    ----------
    W : (n, n) 边权矩阵 (Simpson/Dice，对角线为0)

    Returns
    -------
    H_norm : (n,) 归一化熵，0=极度特异，1=均匀弥散
    """
    n = W.shape[0]
    H_norm = np.ones(n, dtype=np.float64)  # 默认最大熵（最坏情况）

    for i in range(n):
        w_i = W[i].copy()
        w_i[i] = 0.0
        total = w_i.sum()
        if total < 1e-10:
            continue  # 孤立节点，保持默认
        p = w_i / total
        nonzero = p > 0
        n_neighbors = nonzero.sum()
        if n_neighbors < 2:
            H_norm[i] = 0.0  # 只有1个连接 = 最大特异性
            continue
        H = -np.sum(p[nonzero] * np.log(p[nonzero]))
        H_max = np.log(n_neighbors)
        H_norm[i] = H / H_max if H_max > 0 else 1.0

    return H_norm
```

##### 4.3.4 维度 Z — Epileptogenicity (致痫性)

**Phase A（无 FR 分类数据）**：

在 X-Y 空间中用谱聚类，以 Simpson 连接权重为亲和度、以 XY 特征为辅助距离度量：

$$A_{ij}^{cluster} = W_{ij}^{Simpson} \times \exp\left(-\frac{(\hat{X}_i - \hat{X}_j)^2 + (\hat{Y}_i - \hat{Y}_j)^2}{2\sigma^2}\right)$$

- 谱聚类在此作为"自适应社区发现"工具
- Eigengap 自动确定聚类数（不硬编码 N=8）
- 小于 `min_cluster_size` 的孤立簇被标记为噪声

**Phase B（有 FR 分类数据后）**：

$$Z_i = \frac{N_{FR,i}}{N_{Ripple,i} + N_{FR,i}}$$

**更激进的 XYZ 距离度量**（Phase B）：

$$d_{ij}^{XYZ} = \sqrt{w_X(\hat{X}_i - \hat{X}_j)^2 + w_Y(\hat{Y}_i - \hat{Y}_j)^2 + w_Z(\hat{Z}_i - \hat{Z}_j)^2}$$

谱聚类使用 $A_{ij} = W_{ij}^{Simpson} \times \exp(-d_{ij}^{XYZ}/2\sigma^2)$ 作为亲和矩阵，同时编码**连接强度**和**病理特征相似性**。

##### 4.3.5 完整剪枝 API

```python
def compute_node_xyz(
    W_broad: np.ndarray,               # (n_all, n_all) Simpson 宽建图
    events_count: np.ndarray,           # (n_all,) 每通道 HFO 事件数
    recording_duration_min: float,      # 记录时长（分钟）
    fr_ratio: Optional[np.ndarray] = None,  # (n_all,) FR/(R+FR) (Phase B)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算每个节点的 XYZ 三维病理特征。

    Returns
    -------
    X : (n_all,) HFO Rate (events/min)
    Y : (n_all,) Normalized Connection Entropy (0=specific, 1=diffuse)
    Z : (n_all,) Epileptogenicity (Phase A: zeros; Phase B: FR ratio)
    """
    X = events_count.astype(np.float64) / max(recording_duration_min, 1e-6)
    Y = compute_connection_entropy(W_broad)
    Z = fr_ratio.copy() if fr_ratio is not None else np.zeros_like(X)
    return X, Y, Z


def prune_network(
    W_broad: np.ndarray,               # (n_all, n_all) 宽建图
    X: np.ndarray,                     # HFO Rate
    Y: np.ndarray,                     # Connection Entropy
    Z: np.ndarray,                     # Epileptogenicity
    *,
    min_rate: float = 0.5,             # X: 最低 HFO Rate (events/min)
    max_entropy: float = 0.85,         # Y: 最高归一化连接熵
    use_spectral: bool = True,         # Z: 在 XY+Simpson 空间做谱聚类
    min_cluster_size: int = 3,
    n_clusters: Optional[int] = None,  # None = Eigengap 自动
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    XYZ 多维剪枝：从宽建图中提取病理网络核心。

    Pipeline:
    1. X 门控 → 剔除低活跃度节点
    2. Y 门控 → 剔除高熵（全脑弥散）节点
    3. Z 门控 → 谱聚类 (Phase A) 或 FR 比例筛选 (Phase B)

    Returns
    -------
    selected_idx : (n_sel,) 入选节点的全通道池索引
    W_pruned : (n_sel, n_sel) 剪枝后的边权子图
    cluster_labels : (n_all,) 聚类标签 (-1=剔除)
    """
    n = W_broad.shape[0]
    labels = np.full(n, -1, dtype=np.int32)

    # Step 1: X 门控 — 活跃度
    x_pass = X >= min_rate

    # Step 2: Y 门控 — 特异性（剔除高熵 = 伪迹/弥散噪声）
    y_pass = Y <= max_entropy

    # 联合 mask
    node_mask = x_pass & y_pass
    candidate_idx = np.where(node_mask)[0]

    if len(candidate_idx) < min_cluster_size + 1:
        # 候选太少，退化为全部保留
        selected_idx = candidate_idx
        labels[candidate_idx] = 0
    elif use_spectral:
        # Step 3: 谱聚类 → 识别病理网络社区，剔除孤立噪声
        W_sub = W_broad[np.ix_(candidate_idx, candidate_idx)]
        sub_labels, _ = extract_network_clusters(
            W_sub, min_cluster_size, n_clusters,
        )
        # 映射回全通道索引
        for si, ci in enumerate(candidate_idx):
            labels[ci] = sub_labels[si]
        selected_idx = candidate_idx[sub_labels >= 0]
    else:
        selected_idx = candidate_idx
        labels[candidate_idx] = 0

    W_pruned = W_broad[np.ix_(selected_idx, selected_idx)]
    return selected_idx, W_pruned, labels
```

##### 4.3.6 典型案例：XYZ 如何区分真网络与伪迹

| 场景 | HFO Rate (X) | Entropy (Y) | FR Ratio (Z) | 判定 |
|------|---|---|---|---|
| SOZ 核心 | 高 (8/min) | **低** (0.2) | 高 (0.6) | ✅ 保留 — 高活跃、特异连接、高致痫性 |
| 传播通路 | 中 (2/min) | **低** (0.3) | 中 (0.3) | ✅ 保留 — 活跃且有选择性 |
| 参考伪迹 | 高 (10/min) | **极高** (0.95) | 低 (0.1) | ❌ Y 剔除 — 与所有通道均匀连接 |
| 生理性 HFO | 中 (1.5/min) | 中 (0.5) | **极低** (0.02) | ⚠️ X 保留, Y 保留, Z 低 → 被谱聚类标记为边缘/噪声 |
| 安静关键节点 | **低** (0.3/min) | **极低** (0.1) | 高 (0.5) | ❌ X 剔除 — 活跃度不足（Simpson 已记录其跟随关系，后续可回溯） |

##### 4.3.7 工程约束与退化策略

**⚠️ 关键约束**：
- ❌ 不要仅用白质标签剔除 — 灰质异位/脑室周围结节位于深部白质但 HFO 高发
- ❌ 不要硬编码通道数 — 让谱聚类的 Eigengap 或 XY 阈值自适应决定
- ✅ 所有阈值（min_rate, max_entropy）必须可配置 — 患者间差异大
- ✅ Eigengap 不稳定时，`n_clusters` 可手动覆盖
- ✅ 被剔除的节点信息保留在 `cluster_labels` 中，可随时回溯

**退化策略**（当 XYZ 剪枝过于激进时）：

```python
# 保底：只用 X 门控 + 弱 Y 门控
selected = np.where((X >= min_rate) & (Y <= 0.95))[0]
```

---

#### 4.4 方向注入 (Direction Injection) — 剪枝图升级为有向图

**核心改进：统计鲁棒性**

> **批判**：直接用中位数 Lag 定向是危险的。多峰分布（直接通路 5ms + 间接通路 25ms）的中位数 15ms 在物理上没有意义。

**鲁棒方向判定流程**：

```python
from scipy.stats import wilcoxon

def inject_direction(
    W_pruned: np.ndarray,           # (n_sel, n_sel) 剪枝后的 Simpson 边权图
    lag_raw: np.ndarray,            # (n, n_events) 质心时间
    events_bool: np.ndarray,        # (n, n_events) 参与mask
    min_events: int = 5,            # 最小样本量
    lag_thresh_ms: float = 5.0,     # 零滞后阈值
    consistency_thresh: float = 0.6, # 方向一致性阈值
    p_value_thresh: float = 0.05,   # 显著性阈值
) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回:
        adj_directed: (n, n) 有向邻接矩阵，A[i,j] = i→j 的权重
        edge_stats: dict 包含每条边的统计信息
    """
    n = skeleton.shape[0]
    adj = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        for j in range(i+1, n):
            if not skeleton[i, j]:
                continue
            
            # 提取共同事件的 lag
            mask = events_bool[i] & events_bool[j]
            if mask.sum() < min_events:
                continue
            
            lags = lag_raw[i, mask] - lag_raw[j, mask]  # 负 = i 领先
            
            # === 统计检验 ===
            
            # 1. Wilcoxon 检验：Lag 是否显著异于 0？
            try:
                _, p_val = wilcoxon(lags)
            except ValueError:  # 全零或样本太少
                continue
            
            if p_val > p_value_thresh:
                continue  # 零滞后同步，不定向
            
            # 2. 方向一致性检验
            median_lag = np.median(lags)
            if abs(median_lag) < lag_thresh_ms * 1e-3:
                continue  # 太接近零，不定向
            
            direction = np.sign(median_lag)
            consistency = np.mean(np.sign(lags) == direction)
            
            if consistency < consistency_thresh:
                continue  # 方向太乱，视为湍流
            
            # 3. 赋予方向
            if median_lag < 0:  # i 领先 j
                adj[i, j] = consistency
            else:              # j 领先 i
                adj[j, i] = consistency
    
    return adj
```

**关键统计保护**：

| 检验 | 目的 | 失败处理 |
|------|------|---------|
| Wilcoxon | Lag ≠ 0？ | 不定向（视为同步） |
| 一致性 | 方向稳定？ | 不定向（视为湍流） |
| 样本量 | n ≥ 5？ | 不建边（数据不足） |

---

#### 4.5 权重定义 (Weight Definition) — 多维复合权重

> 单一权重无法捕捉致痫网络的复杂性。必须融合因果性、稳定性与病理特异性。

##### 4.5.1 三维权重模型

$$W_{ij} = \underbrace{\text{Simpson}_{ij} \times \text{Consistency}_{ij}}_{\text{Causality（因果性）}} \times \underbrace{(1 - \text{CV}_{time}^{ij})}_{\text{Stability（稳定性）}} \times \underbrace{\left(1 + \alpha \cdot \frac{N_{FR}^{ij}}{N_{total}^{ij}}\right)}_{\text{Pathology（病理性）}}$$

| 维度 | 定义 | 数据来源 | Phase |
|------|------|----------|-------|
| **Causality** | $\text{Simpson}_{ij} \times \text{Consistency}_{ij}$ — Simpson 归一化共激活 × 方向一致性 | `W_pruned` + `lag_raw` | **A (立即可做)** |
| **Stability** | $1 - \text{CV}(\text{Connectivity}(t))$ — 连接的时间鲁棒性 | `event_windows` 按时间窗切片 | **A (立即可做)** |
| **Pathology** | $1 + \alpha \cdot \frac{N_{FR}}{N_{total}}$ — Fast Ripple 比例加权 | `hfo_type_per_event` | **B (需分类数据)** |

##### 4.5.2 Stability（稳定性）维度 — 时间鲁棒性

**核心思想**：癫痫网络应具有刻板性（Stereotypical），随机出现的连接是噪声。

```python
def compute_stability_weights(
    lag_raw: np.ndarray,           # (n_ch, n_events) 质心时间
    events_bool: np.ndarray,       # (n_ch, n_events) 参与mask
    event_times: np.ndarray,       # (n_events,) 事件时间戳
    window_sec: float = 300.0,     # 5分钟时间窗
    min_windows: int = 3,          # 最少窗口数
) -> np.ndarray:
    """
    计算每条边在多个时间窗内的连接方向一致性。
    
    Stability = 1 - CV(consistency_per_window)
    高稳定性 = 固定的病理通路；低稳定性 = 瞬态噪声
    """
    n_ch = lag_raw.shape[0]
    stability = np.full((n_ch, n_ch), np.nan)
    
    # 按时间窗切片
    t_min, t_max = event_times.min(), event_times.max()
    edges = np.arange(t_min, t_max, window_sec)
    if len(edges) < min_windows:
        return np.ones((n_ch, n_ch))  # 数据不够，退化为权重1
    
    window_consistencies = []
    for t_start in edges:
        t_end = t_start + window_sec
        win_mask = (event_times >= t_start) & (event_times < t_end)
        if win_mask.sum() < 5:
            continue
        
        # 每个时间窗内计算方向一致性
        cons = np.zeros((n_ch, n_ch))
        for i in range(n_ch):
            for j in range(i+1, n_ch):
                both = events_bool[i, win_mask] & events_bool[j, win_mask]
                if both.sum() < 3:
                    continue
                lags = lag_raw[i, win_mask][both] - lag_raw[j, win_mask][both]
                med = np.median(lags)
                cons[i, j] = np.mean(np.sign(lags) == np.sign(med))
                cons[j, i] = cons[i, j]
        window_consistencies.append(cons)
    
    if len(window_consistencies) < min_windows:
        return np.ones((n_ch, n_ch))
    
    stacked = np.stack(window_consistencies)
    mean_cons = np.nanmean(stacked, axis=0)
    std_cons = np.nanstd(stacked, axis=0)
    cv = np.where(mean_cons > 0, std_cons / mean_cons, 1.0)
    stability = 1.0 - np.clip(cv, 0, 1)
    
    return stability
```

##### 4.5.3 Pathology（病理性）维度 — 频率特异性

**设计理由**（参考文献：SpikewHFO更重要.pdf）：
- 叠加 HFO 的 Spike 比单纯 Spike 更能定位 SOZ
- Fast Ripple 比 Ripple 更具病理特异性
- 给高病理性传播事件更高投票权

**Phase A（立即可做）**：用 Coact × Consistency × Stability 三维权重

**Phase B（需 FR 分类数据后）**：加入 $(1 + \alpha \cdot FR_{ratio})$ 因子，$\alpha$ 建议 0.5-2.0

##### 4.5.4 进阶方向：频谱因果性（Phase C 研究前沿）

> 用频谱格兰杰因果 (Spectral GC) 或偏定向相干 (PDC) 替代 Lag-based 因果推断。

$$W_{ij}^{advanced} = \underbrace{\text{PDC}_{ij}(f_{HFO})}_{\text{频域因果}} \times \underbrace{(1 - \text{CV}_{time})}_{\text{稳定性}} \times \underbrace{\frac{SC_{ij}}{SC_{max}}}_{\text{解剖先验}} \times \underbrace{\text{PathScore}_i}_{\text{节点病理分}}$$

**为什么列为 Phase C**：PDC 需要模型阶数选择（AIC/BIC）、平稳性检验、$O(N^2 \times T \times p)$ 计算。对 50 通道 × 2h 数据虽然可行但调参复杂。先用 Lag-based 方法验证整体流程，再考虑替换为 PDC。

---

#### 4.6 图论指标计算 (Metric Calculation)

**使用 `networkx` 库**：

| 指标 | 公式 | 临床意义 |
|------|------|---------|
| **Net Outflow Index** | $\frac{OutDegree - InDegree}{OutDegree + InDegree}$ | **SOZ 定位**：值接近 +1 = Source |
| **Outflow Volatility** | $\text{Var}(\text{NetOutflow}_t)$ | 真正 SOZ 往往发作前突然爆发 |
| **Local Efficiency** | $E_{loc}(i) = \frac{1}{k_i(k_i-1)} \sum_{j,h \in N_i} \frac{1}{d_{jh}}$ | 致痫灶的紧密程度 |
| **Shortest Path Tree** | 从 Source 出发的最短路径 | 传播路径预测 |

```python
import networkx as nx

def compute_network_metrics(adj: np.ndarray, ch_names: List[str]) -> Dict:
    """
    计算核心图论指标。
    """
    G = nx.DiGraph()
    n = adj.shape[0]
    
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                G.add_edge(ch_names[i], ch_names[j], weight=adj[i, j])
    
    metrics = {}
    
    # Net Outflow Index
    for node in G.nodes():
        out_deg = G.out_degree(node, weight='weight')
        in_deg = G.in_degree(node, weight='weight')
        total = out_deg + in_deg
        metrics[f'{node}_outflow'] = (out_deg - in_deg) / total if total > 0 else 0
    
    # Local Efficiency (需要转无向图)
    G_undirected = G.to_undirected()
    metrics['local_efficiency'] = nx.local_efficiency(G_undirected)
    
    # Betweenness Centrality
    metrics['betweenness'] = nx.betweenness_centrality(G, weight='weight')
    
    return metrics
```

---

#### 4.7 关键陷阱与防护 (Critical Pitfalls)

**陷阱1：容积传导的幽灵 (Volume Conduction)**

| 现象 | 物理距离 <10mm，Lag ≈ 0，Co-activation 极高 |
|------|-------------------------------------------|
| 原因 | 电场直接传导，非神经元传播 |
| 危害 | 网络被无意义短边主导 |
| **防护** | 强制剔除 `dist_matrix < 10mm` 的边 |
| **反向利用** | 保留局部连接强度作为 "Local Recruitment Score" |
| **零滞后陷阱** | 深部强源被两个远距电极同时记录 → 高同步但零延迟<br>必须用 PLI/wPLI（对零滞后不敏感）或 Wilcoxon 检验过滤 |

**陷阱2：采样偏差 (Sampling Bias)**

| 现象 | SEEG 仅覆盖不到 1% 的脑体积 |
|------|------------------------------|
| 危害 | 真正的源在未采样区，中继站被误判为源 |
| **防护** | 结论必须谨慎：<br>"在被监测的网络中，节点 X 表现出源的特征" |

**陷阱3：中位数陷阱 (The Median Trap)**

| 现象 | Lag 分布多峰（直接通路 5ms + 间接通路 25ms） |
|------|-------------------------------------------|
| 危害 | 中位数 15ms 在物理上不存在 |
| **防护** | 单峰性检验 (Hartigan's dip test) 或方差检查<br>高方差边标记为 "Unstable Connection"，降低权重 |

**陷阱4：静态网络的局限**

| 现象 | 24小时平均图抹杀时间维度 |
|------|-------------------------|
| 危害 | 间歇性喷发的 SOZ 被持续活跃的中继站掩盖 |
| **进阶方向** | 动态切片：每 5 分钟或每 100 事件计算一次<br>比较 Pre-ictal vs Interictal 网络拓扑 |

**陷阱5：生理性 HFO 混淆 (Physiological HFO Contamination)** 🔴 新增

| 现象 | 视觉/运动皮层和海马在 NREM 期间产生高发放率生理性 HFO |
|------|--------------------------------------------------|
| 危害 | 功能区被误判为致痫灶 → 手术导致功能缺损 |
| **防护** | 谱聚类 + 共激活过滤 — 生理性 HFO 往往是孤立的局部功能柱活动，不形成大尺度同步网络<br>Stability 权重 — 生理性 HFO 是任务/状态相关的瞬态，病理性更持续 |
| **补充** | 结合 Spike-HFO 共现特征：叠加 Spike 的 HFO 病理特异性更高 |

**陷阱6：Sink/Source 反转 (Sink Trap)** 🔴 新增

| 现象 | SOZ 在发作间期可能表现为 Sink（被抑制），发作期转为 Source |
|------|------------------------------------------------------|
| 危害 | 仅分析发作间期数据会将 SOZ 误判为"接收节点" |
| **防护** | 必须结合 Ictal 数据验证：寻找"间期 Sink → 发作期 Source"的动态反转节点<br>这种反转本身就是 EZ 的"指纹"特征 |
| **指标** | $\Delta \text{Outflow} = \text{Outflow}_{ictal} - \text{Outflow}_{interictal}$ — 反转幅度最大的节点 |

**陷阱7：SEEG 行波假设失效 (Traveling Wave Caveat)** 🔴 新增

| 现象 | HFO/IED 在皮层上表现为行波（Traveling Waves） |
|------|---------------------------------------------|
| 危害 | 在 Grid 电极上可直接拟合波峰梯度场计算传播速度矢量<br>**但 SEEG 是棒状深部电极**，穿过不同皮层层级，2D 平面波假设失效 |
| **防护** | 在 SEEG 中必须沿电极轴向（Axial）和跨电极（Cross-electrode）分别计算延迟<br>不可盲目拟合平面波 |
| **方向反转** | IED 传播方向通常**指向**致痫灶（Sink 特征）<br>Ictal Discharge 通常**背离**致痫灶传播<br>这一方向反转是重要的定位特征 |

---

#### 4.8 三阶段实施路线图 (Three-Phase Roadmap)

> "Theory and practice sometimes clash. Theory loses. Every single time." — 先用现有数据跑通全流程，再逐步加入高级特征。

##### Phase A：Channel-Scale MVP（数据已就绪，立即可做）

| Step | 任务 | 输入 | 输出 | 新增依赖 | 状态 |
|------|------|------|------|----------|------|
| A.1 | **宽建图 (Simpson Index)** | `coact_all_event_count` | `W_simpson/W_dice` (n_all, n_all) | — | ✅ |
| A.2 | 替代数据显著性检验 | `events_bool` | `sig_mask` | — | ✅ |
| A.3 | **XYZ 特征计算** | `W_broad`, `events_count`, `duration` | `X, Y, Z` per node | — | ✅ |
| A.4 | **XYZ 多维剪枝** | `W_broad`, `X`, `Y`, `Z` | `selected_idx`, `W_pruned` | `sklearn` (谱聚类) | ✅ |
| A.5 | 方向注入（Wilcoxon+一致性） | `W_pruned`, `lag_raw` | `adj_directed` | `scipy.stats` | ✅ |
| A.6 | Stability 权重 | `lag_raw`, `event_windows` | `stability_matrix` | — | ✅ |
| A.7 | 复合/融合权重（direction-first） | `assoc/B/D/S/lag` | `adj_weighted` | — | ✅ |
| A.8 | 图论指标 | `adj_weighted` | `metrics_dict` | `networkx` | ✅ |
| A.9 | 2D 网络拓扑图 + XY 散点诊断图 | `metrics`, `X`, `Y` | `network_plot.png` | `matplotlib` | ✅ |

**Phase A 的交付物**：
1. 一个完整的 Channel-scale 有向加权癫痫网络（Simpson 归一化 + XYZ 剪枝）
2. XY 散点诊断图：直观展示哪些节点被保留/剔除及原因
3. Net Outflow 排名（Source-Sink 预测）

##### Phase B：Channel-Scale + Geometry（需 MNI 坐标）

| Step | 任务 | 输入 | 输出 | 阻塞条件 | 状态 |
|------|------|------|------|----------|------|
| B.0 | 电极坐标获取 | MNI 配准结果 | `mni_coords.npy`, `dist_matrix.npy` | **需临床数据** | ⬜ |
| B.1 | 空间约束骨架 | `dist_matrix`, `coact_ratio` | `skeleton_spatial` | B.0 | ⬜ |
| B.2 | 容积传导剔除 | `dist_matrix < 10mm` | `skeleton_clean` | B.0 | ⬜ |
| B.3 | 传播速度验证 | `dist_matrix`, `lag_raw` | `velocity diagnostics` (0.1-10 m/s) | B.0 | 🔄 |
| B.4 | 病理加权（FR 比例） | `hfo_type_per_event` | `pathology_weight` | **需 FR 分类** | ⬜ |
| B.5 | 3D 脑图 | `metrics`, `mni_coords` | `outflow_brain_3d.html` | B.0 | ⬜ |
| B.6 | Ictal vs Interictal 对比 | `event_windows`, `seizure_onsets` | `delta_outflow` | — | 🔄 |

**Phase B 的交付物**：物理约束后的网络 + 3D 可视化 + Sink/Source 反转分析。

##### Phase C：Source-Scale 研究前沿（需影像学流水线）

| Step | 任务 | 输入 | 阻塞条件 | 状态 |
|------|------|------|----------|------|
| C.1 | 前向模型(BEM) | FreeSurfer 输出, 电极坐标 | 需 MRI 分割 + 配准 | ⬜ |
| C.2 | 导联场矩阵(LFM) | BEM 模型 | C.1 | ⬜ |
| C.3 | LFM 概率投影 | `LFM`, `channel_metrics` | C.2 | ⬜ |
| C.4 | SC-FC 耦合图 | HCP tractography | 需 DWI 数据 | ⬜ |
| C.5 | PDC/频谱格兰杰 | 原始时间序列 | 计算密集 | ⬜ |
| C.6 | NMM 验证 | 连接矩阵 | 独立研究课题 | ⬜ |

**Phase C 的交付物**：源空间级别的病理网络重构（研究论文级别）。

---

**核心 API 设计**：

```python
# src/network_analysis.py

@dataclass
class NetworkResult:
    """癫痫网络分析结果 (v3: direction-first causal)."""
    adj: np.ndarray
    node_names: List[str]
    node_weights: np.ndarray
    W_simpson: np.ndarray
    W_dice: np.ndarray
    W_pruned: np.ndarray
    pool_names: List[str]
    selected_idx: np.ndarray
    node_xyz: Dict[str, np.ndarray]
    skeleton: np.ndarray
    direction_mask: np.ndarray
    stability: np.ndarray
    cluster_labels: np.ndarray
    metrics: Dict[str, Any]
    edge_stats: List[Dict]
    n_pool_channels: int
    n_selected: int
    params: Dict[str, Any]

def build_hfo_network(
    group_analysis_npz: str,
    dist_matrix: Optional[np.ndarray] = None,
    *,
    detections_npz_path: Optional[str] = None,
    # — Step 1: 宽建图 —
    edge_method: str = 'simpson',     # 'simpson' | 'dice'
    run_surrogate: bool = True,
    n_surrogates: int = 200,
    # — Step 2: XYZ 剪枝 —
    min_rate: float = 0.5,            # X: 最低 HFO Rate (events/min)
    max_entropy: float = 0.85,        # Y: 最高归一化连接熵
    use_spectral: bool = True,        # Z: 谱聚类进一步剪枝
    min_cluster_size: int = 3,
    n_clusters: Optional[int] = None, # None = Eigengap 自动
    # — Step 3: 方向注入 —
    min_events: int = 5,
    lag_thresh_ms: float = 5.0,
    consistency_thresh: float = 0.6,
    p_value_thresh: float = 0.05,
    # — Step 4: 稳定性 / direction-first 融合 —
    stability_window_sec: float = 300.0,
    assoc_window_ms: float = 40.0,
    min_pair_events: int = 5,
    tau_assoc: float = 20.0,
    tau_lag_ms: float = 10.0,
    fusion_w_b: float = 0.35,
    fusion_w_d: float = 0.45,
    fusion_w_s: float = 0.20,
    d_strong: float = 0.35,
    b_min: float = 0.2,
    min_dist_mm: float = 10.0,
    lag_vc_ms: float = 3.0,
    sample_cap_per_edge: int = 50000,
) -> NetworkResult:
    """
    一站式构建癫痫网络 v3（direction-first causal）。

    流程：pairwise association → direction injection → fusion + physics →
         final pruning + node rescue → metrics

    Returns
    -------
    NetworkResult : 包含有向加权邻接矩阵、XYZ 特征和图论指标
    """
    ...
```

**可视化终极目标**：

```python
def plot_outflow_brain_map(
    network_result: NetworkResult,
    mni_coords: np.ndarray,
    output_path: str,
):
    """
    3D 脑图（Phase B 交付物）：
    - 节点颜色 = Net Outflow (红=Source, 蓝=Sink)
    - 节点大小 = Local Efficiency
    - 边颜色 = 传播方向
    - 边粗细 = 权重

    这是直接对话临床医生的"终极图表"。
    """
    ...

def plot_network_topology_2d(
    network_result: NetworkResult,
    output_path: str,
    layout: str = 'spring',
):
    """
    2D 网络拓扑图（Phase A 交付物）：
    - 节点颜色 = Net Outflow
    - 节点大小 = Node Weight (EigenCentrality × Rate)
    - 边粗细 = 复合权重
    - 布局 = spring / circular / spectral

    不需要 MNI 坐标，Phase A 即可生成。
    """
    ...
```

---

#### 4.9 功能清单 (Feature Checklist)

**Phase A — Channel-Scale MVP（立即可做）**

| 功能 | 说明 | 依赖 | 状态 |
|------|------|------|------|
| A.1 宽建图 (Simpson) | Simpson Index 归一化共激活 → 宽边权图 | `coact_all_event_count` | ✅ |
| A.2 替代数据检验 | Surrogate test 验证共激活显著性 | `events_bool` | ✅ |
| A.3 XYZ 特征计算 | X=Rate, Y=Connection Entropy, Z=placeholder | `W_broad` | ✅ |
| A.4 XYZ 多维剪枝 | X门控 + Y门控 + 谱聚类(XY距离) | `sklearn` | ✅ |
| A.5 方向注入 | Wilcoxon + 一致性 + 零滞后过滤 | `lag_raw`, `scipy.stats` | ✅ |
| A.6 Stability 权重 | 时间窗切片 + CV 计算 | `event_windows` | ✅ |
| A.7 复合权重 | direction-first 融合权重 | — | ✅ |
| A.8 图论指标 | Net Outflow, Local Efficiency, Betweenness | `networkx` | ✅ |
| A.9 2D 网络拓扑图 | Spring 布局 + XY 散点诊断图 | `matplotlib` | ✅ |
| A.10 传播路径 | 最短路径树 | `adj` | ✅ |

**Phase B — Channel + Geometry（需 MNI 坐标）**

| 功能 | 说明 | 阻塞条件 | 状态 |
|------|------|----------|------|
| B.1 空间约束骨架 | 距离惩罚 + 容积传导剔除 | MNI coords | ⬜ |
| B.2 传播速度验证 | 0.1-10 m/s 生理范围检查 | MNI coords | 🔄 |
| B.3 病理加权 | FR 比例加权 | HFO type 分类 | ⬜ |
| B.4 3D 脑图 | Outflow 颜色映射 | MNI coords | ⬜ |
| B.5 动态切片 | Pre-ictal vs Interictal 网络对比 | Seizure onsets | ⬜ |
| B.6 Sink/Source 反转 | $\Delta$Outflow (ictal - interictal) | B.5 | 🔄 |

**Phase C — Source Space 研究前沿**

| 功能 | 说明 | 阻塞条件 | 状态 |
|------|------|----------|------|
| C.1 前向模型(BEM/FEM) | 患者个性化导联场 | FreeSurfer + MRI | ⬜ |
| C.2 LFM 概率投影 | 灵敏度加权映射 | C.1 | ⬜ |
| C.3 SC-FC 耦合图 | 解剖先验约束 | HCP tractography | ⬜ |
| C.4 PDC/频谱格兰杰 | 频域因果性 | 模型阶数选择 | ⬜ |
| C.5 NMM 验证 | 分析-综合闭环 | 独立研究课题 | ⬜ |

---

#### 4.10 源空间构建远景 (Source Space Vision) — Phase C 理论基础

> 本节记录 Source-Scale 网络构建的理论基础和工程路径。**当前不实现**，作为研究前沿参考。

##### 4.10.1 粒度困境 (Granularity Dilemma)

| 尺度 | 分辨率 | 失效原因 |
|------|--------|---------|
| **脑区 (AAL/DK)** | ~100 区域 | HFO 生成器 <2mm，脑区平均化彻底淹没病理信号，SNR 指数下降 |
| **顶点 (Vertex)** | ~20k/半球 | SEEG 仅 100-200 触点 → 极度欠定逆问题 → 无数据区域的"插值幻觉" |
| **传感体积 (VOI)** | **5mm 半径** | ✅ 匹配 SEEG 宏电极传感半径 (~3-5mm)<br>✅ 包含 HFO 微观发生结构<br>✅ 避免过度插值 |

**结论**：源空间节点应定义为以电极触点为中心的 5mm VOI（Virtual Voxels），非均匀全脑网格。

##### 4.10.2 LFM 概率投影 (Lead-Field Weighted Projection)

> 无需求解复杂的源成像逆问题。利用导联场矩阵作为几何先验进行概率映射。

**核心公式**：

$$W_{ji} = \frac{L_{ij}^2}{\sum_{k \in \text{Channels}} L_{kj}^2}$$

$$\text{SourceMetric}_j = \sum_{i} W_{ji} \cdot \text{ChannelMetric}_i$$

其中 $L_{ij}$ 是导联场矩阵中源 $j$ 对通道 $i$ 的贡献（包含距离衰减和偶极子方向信息）。使用平方是因为功率/能量随距离平方衰减。

**优势**：
- 计算高效：一次性线性变换，非迭代反演
- 物理合理：自动处理距离加权 + 方向性
- 避免 Double Counting：归一化权重自然分配重叠区域

**对共激活矩阵的映射扩展**：

$$\text{SourceCoAct}_{jk} = \sum_{m,n} W_{jm} \cdot \text{ChannelCoAct}_{mn} \cdot W_{kn}$$

##### 4.10.3 SC-FC 耦合图 (Structure-Function Coupled Graph)

> 在源空间定义传播路径时，必须引入 HCP SC 作为贝叶斯先验。

$$P(E_{A \to B} | \text{Data}) \propto \text{FC}_{A \to B} \times \text{SC}_{A \to B}$$

**物理意义**：如果源 A 到源 B 的功能连接（FC）很强，但无白质纤维束直接连接（SC ≈ 0），则该"连接"极可能是间接的或虚假的。

**工程依赖链**：
1. FreeSurfer 皮层重建 → 高分辨率 mesh
2. BEM/FEM 前向建模 → 导联场矩阵 $G$
3. 电极定位 (LeadDBS/iElectrodes) → MNI 坐标
4. HCP tractography → 结构连接矩阵 SC
5. LFM 概率投影 → 源空间指标
6. SC × FC → 耦合图

**现实评估**：这是一条 6 个月的工程路径，每个环节都需要独立验证。但一旦建成，可以实现从电生理到解剖的无缝对接，这是最终的临床转化目标。
</content>
</invoke>