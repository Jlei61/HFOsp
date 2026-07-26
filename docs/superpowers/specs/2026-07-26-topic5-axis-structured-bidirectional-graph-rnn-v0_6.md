# Topic 5 / Figure 6：患者轴约束的双向 graph RNN（v0.6）

**日期**：2026-07-26
**状态**：执行合同
**继承**：34人、masked rank、chronological 80/20、LOSO、clinical onset
`[0,10] s`、BB `1–150 Hz`、发作 target 封存合同
**降级**：v0.5 的自由 `D + UV^T` low-rank leaky RNN 只保留为 sensitivity

---

## 1. 核心科学问题

本分析不再问“无约束低秩矩阵能否自动浮现传播结构”，而问：

> 如果把真实数据与 SNN 都支持的患者特异轴向底物、两端 source core、正反向
> 传播和共享抑制写成模型的结构先验，一个最小的方向性 latent system 是否足以
> 生成患者的间期传播事件；冻结后的同一结构是否解释 clinical onset 附近的
> 发作早期静态能量场？

主证据链固定为：

```text
train-80% interictal rank events
  → unsigned patient axis + two endpoint cores
  → paired forward/reverse contact graphs
  → structured rank-r graph RNN
  → held-out/free-running interictal propagation fields
  → frozen directional-mode lesions
  → clinical-onset [0,10] s static ictal field
```

---

## 2. 当前 v0.5 的地位

自由 `D + UV^T` 模型不再承担主结论。它只回答：

- 无结构 low-rank 附加项是否带来增益；
- 模型是否会把事件压到低维 progress axis；
- 与结构化 graph RNN 相比，图归纳偏置是否必要。

v0.5 的 rank 0 含32个独立 diagonal memories，不是真正的无递归对照；因此不能
用它推断“正 low-rank mode 不存在”。

---

## 3. 患者轴先验

### 3.1 数据边界

每名患者只使用 chronological train 80% 的 masked contact-rank events 构造先验。
held-out 20% 和全部发作数据禁止读取。

### 3.2 构造

1. 对 train 80% 事件做 masked KMeans `k=2`；非参与触点填事件中点 `0.5`。
2. 得到两条 contact-rank template，但模板编号与正负号均视为任意。
3. 用两模板的 contact rank difference 构造无符号轴 `s_c ∈ [-1,1]`。
4. 轴两端各取 `k=min(3, floor(n_contact/4))` 个可靠触点作为 endpoint source
   cores。
5. 沿 `s_c` 连接相邻触点，构造：
   - `A_forward`：由负端向正端；
   - `A_reverse=A_forward^T`：由正端向负端。
6. 正式模型同时读取两张图，因此交换模板标签或翻转轴符号不改变模型语义。

### 3.3 已完成审计

训练内轴先验审计输出：

`results/topic5_structured_axis_graph/axis_prior_v1_fast/`

- 34/34 可构图；
- split-half 轴 `|rho|>=0.5`：31/34；
- KMeans-seed 轴 `|rho|>=0.5`：31/34；
- cluster 与事件极性两侧均有事件：34/34；
- 稳定轴 + 稳定初始化 + 双侧事件的高稳定层：28/34。

主间期分母仍为34；28人为预先定义的轴稳定性 sensitivity。模板严格反向程度只作
连续协变量，不能事后删人。

---

## 4. 模型：axis-structured bidirectional graph RNN

### 4.1 状态

每个触点只有 `r` 个结构化 latent channels：

```text
H_t ∈ R^(n_contact × r)
q_t ∈ R
```

`q_t` 是共享抑制状态。不存在额外32维自由 hidden state，也不存在稠密可学习
contact-to-contact recurrence。

rank 2 以上的两个方向通道使用参数共享、符号固定的对称相互抑制。这样可以避免
forward/reverse 两通道在相加 readout 下退化成 rank 1 的对称图；交换轴符号和
两通道后方程仍保持不变。

### 4.2 Rank层级

这里的 rank 是承重 latent channel 数，不是自由矩阵分解：

| rank | 结构 | 科学含义 |
|---:|---|---|
| 0 | 无 recurrent state | 真正静态/无历史对照 |
| 1 | `A_sym=(A_forward+A_reverse)/2` | 有轴向传播，但不区分方向 |
| 2 | channel 1=`A_forward`; channel 2=`A_reverse` | 主假设：显式正反向双通道 |
| 3 | rank 2 + global recruitment channel | 检查是否需要整体招募状态 |
| 4 | rank 3 + local surround-inhibitory channel | 检查是否需要局部抑制细化 |

预期的最小充分模型是 rank 2，但这是待检验假设，不是结果标签。

### 4.3 更新方程

对每个 channel：

```text
h_(c,k,t+1)
 = (1-alpha_k) h_(c,k,t)
 + alpha_k tanh(
     g_in,k x_(c,k,t)
     + g_prop,k [A_k (h_(·,k,t)+x_(·,k,t))]_c
     - d_k h_(c,k,t)
     - beta_k q_t
     - gamma_dir E_(opposite,t)
     + b_k
   )
```

共享抑制：

```text
q_(t+1)
 = (1-alpha_I) q_t
 + alpha_I tanh(g_I mean_c,k relu(h_(c,k,t)))
```

约束：

- `g_in, g_prop, d, beta, g_I, gamma_dir >= 0`；
- `0 < alpha < 1`；
- excitatory propagation 与 inhibitory feedback 符号固定；
- rank 2 的 forward/reverse graph 互为转置；
- local patient offset 只进入静态 contact excitability/readout，不能进入 graph
  或 latent transition。

### 4.4 端点输入

- 负端当前 recruitment 优先驱动 forward channel；
- 正端当前 recruitment 优先驱动 reverse channel；
- 中间触点按轴位置连续混合；
- rank 1 只能使用无方向的 symmetric graph。

模型不学习或预测 A/B label；方向由事件当前 source 位置与固定双向图共同决定。

### 4.5 输出

与 v0.5 相同：

- next recruitment set；
- STOP；
- free-running contact participation；
- conditional rank distribution；
- pairwise precedence；
- 完整 event × contact path distribution。

---

## 5. 训练

### 5.1 数据划分

- outer LOSO：33人共享训练，1人完全 held out；
- 每名患者 chronological first 80% 用于轴构造与 local readout calibration；
- last 20% 只评估；
- 三 seeds；
- patient-level collapse 后统计。

### 5.2 Loss

首轮结构 sanity 后依据预先封存的 train80/heldout20 审计修订为：

```text
L = L_next_set_STOP
  + lambda_stop L_stop_calibration
```

两端 source-core 富集在 train80 为24/34、heldout20为25/34，属于队列层面的弱倾向，
不是每位患者都成立。因此 `L_endpoint_source` 从主损失移除，只保留为 sensitivity；
endpoint bias 从接近0初始化并由 train80 决定是否保留。不得加入 A/B 分类 loss。

新观察到的 rank set 必须在同一次状态更新中沿固定图播散，即使用
`A_k(h_t+x_t)`；若只使用 `A_k h_t`，图信息会晚一个 rank，无法用于预测紧随其后的
下一个 rank set。

若 free rollout 暴露出明显 exposure collapse，第二版才加入预注册的 soft
self-feeding consistency；不能根据发作结果调 loss。

---

## 6. 必要对照

1. **rank 0**：真正无历史；
2. **rank 1**：轴存在但方向合并；
3. **axis shuffle**：在患者内打乱轴坐标并重建等度数图；
4. **endpoint lesion**：去掉两端 source bias；
5. **direction lesion**：rank 2 分别去掉 forward 或 reverse channel；
6. **inhibition lesion**：去掉 shared inhibitory pool；
7. **v0.5 free low-rank**：无图结构 sensitivity；
8. **empirical distribution**：数据上界，不是必须击败的模型。

axis null 不能简单翻转轴，因为正式双向图对全局翻转不变。主 null 应是患者内轴
坐标重排；有 shaft 信息时优先 within-shaft shuffle。

---

## 7. 间期成功标准

### 7.1 训练可用

- rollout 能合法 STOP；
- 不重复触点；
- 无单一 endpoint 或固定事件长度 collapse。

### 7.2 主结构证据

rank 2 相对 rank 0/1 和 axis-shuffle：

- pairwise precedence 更接近 held-out；
- label-free whole-path distance 更小；
- participation/rank/precedence 进入患者 split-half variability；
- 跨 seed 稳定。

不要求 rank 2 的一步 NLL 击败所有模型。

### 7.3 Rank选择

- rank 2 达标而 rank 1 不达标：支持显式正反向双通道；
- rank 1 已达标：只支持无方向轴传播；
- rank 3/4 才达标：需要额外 global/local inhibitory state；
- 所有结构 rank 均失败：轴图假设停止，不读发作 target。

---

## 8. 发作期读取

只有结构 rank 通过间期门并冻结后，才读取：

```text
clinical onset [0,10] s
BB 1–150 Hz
baseline-robust-z static energy field
```

Primary比较：

- frozen graph-RNN free-running early-rank / susceptibility field；
- frozen forward/reverse channel contact loading；
- empirical interictal rank field；
- axis-prior-only baseline；
- participation-only baseline；
- within-shaft axis shuffle；
- forward/reverse channel lesion；
- inhibition lesion。

成功结论必须依赖同一结构同时：

1. 复现 held-out 间期传播；
2. 产生可重复的方向性内部模式；
3. 冻结后解释发作早期静态能量场；
4. mode lesion 同时破坏间期和发作读出。

仅轴先验本身与发作场相关时，只能写“interictal scaffold reuse”，不能把 graph RNN
写成额外机制证明。

---

## 9. 当前执行顺序

1. 保留 v0.5 为 sensitivity；
2. 使用已审计的34人 train-only axis priors；
3. 先跑 rank 0/1/2 的小规模三患者 sanity；
4. 通过后扩到 rank 0–4、34人、三 seeds；
5. 完成 direction/endpoint/inhibition/axis-shuffle lesions；
6. 冻结最小充分结构；
7. 最后读取 clinical-onset 发作早期静态场。
