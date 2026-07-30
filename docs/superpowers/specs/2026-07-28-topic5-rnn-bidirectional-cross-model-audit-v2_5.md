# Topic 5 RNN 双向性与跨模型静态迁移审计 v2.5

## 1. 目的

本轮不把经验 A/B 轴当作金标准，也不再以“RNN 是否恢复 A/B 轴”作为主要结果。
需要回答两个更基础的问题：

1. train-only 转移分解选择的无符号方向，是否在 held-out 间期事件中同时包含轴两侧
   起点的传播信息；
2. 普通 full-history GRU 与结构化传播 RNN 学到的 contact-level rank distribution，
   与 clinical onset 后 `[0,10] s`、1–150 Hz 静态能量场有多大绝对相似度。

经验 A/B 的 same/reversed/different 关系只作外部描述，不参与患者筛选、模型选择或
主要统计。

## 2. 模型对象

### 2.1 普通 full-history GRU

- 输入：截至当前 rank 的有序 contact-set prefix；
- 任务：自监督预测下一 rank set 与 STOP；
- 允许自由 hidden mixing；
- 输出：free rollout 后每个 contact 的参与概率、条件 rank 分布及其统计量；
- 对照：
  - `static_contact_hazard`：只用 contact 基础频率，不看事件历史；
  - `unordered_prefix`：看已出现 contact，但不使用其顺序；
  - `last_set_first_order`：只看最后一个 rank set；
  - `rank_shuffle_gru`：保留模型容量，但训练时破坏 rank 顺序。

### 2.2 结构化 competitive-propagation RNN

- 固定输入图：3D contact geometry；
- 轴：仅由 train80 转移残差在 32 个候选方向中选择，RNN 不端到端自由学习 3D 轴；
- 状态：传播 trace 与较慢竞争 trace；
- 输出：下一 contact categorical probability；
- 禁止 dense contact-to-contact bypass；
- 对照：
  - `node_only`：只用 contact 基础频率；
  - `no_history`：传播和竞争 persistence 设为 0，只保留当前 rank；
  - `local_isotropic`：只保留各向同性局部核；
  - `axis_no_source`：保留轴向核与两状态，但移除 source-direction 项；
  - `full`：轴向核、两状态与 source-direction 项全部保留。

注意：旧 v2.4 的 `full - local_isotropic` 同时改变轴与 source 项，不能单独命名为
“axis contribution”。v2.5 必须补出 `axis_no_source`，用
`axis_no_source - local_isotropic` 隔离轴向核。

## 3. 间期双向性

队列为 v2.3 冻结的 22 名 geometry-complete、development-excluded 患者。

每位患者使用同一条 train80-selected sign-free axis。heldout20 事件按第一 rank set
在轴投影上的位置分为 negative-source 与 positive-source；每侧至少 20 个事件才进入
双侧检验。

对两侧分别计算：

1. 从 source 到后续 contact 的归一化 inward displacement；
2. `full - isotropic` heldout next-contact NLL benefit；
3. `full - axis_no_source` source-term benefit。

患者级双向分数取两侧较小值，避免一侧强、另一侧无效仍被称为“双向”。

几何/选轴 null 使用同一 32 个候选方向：在每个候选方向上重复 source-side 划分与
heldout displacement，比较 train-selected axis 相对候选方向中位数的 margin。
该 null 不使用 A/B 标签。

经验 A/B 仅在结果完成后作连续角度和 relation 分层描述，不设 90° 或其他硬阈值。

## 4. Early-ictal 静态迁移

### 4.1 主要队列

- 严格 clinical-onset、strict broadband 1–150 Hz；
- 16 名 Epilepsiae、106 次发作；
- 每次发作 `[0,10] s` baseline-robust-z contact energy；
- 每次发作先计算 contact-wise 相似度，再按 seizure median、seed median、
  patient-first 折叠。

两名只有 EEG-onset target 的 Yuquan 不混入该队列，只作单独 sensitivity。

### 4.2 Contact fields

从每个模型/经验 rank distribution 固定导出：

1. participation probability；
2. early joint mass（参与且位于前 30% rank）；
3. late joint mass（参与且位于后 30% rank）；
4. bidirectional endpoint mass（early + late）；
5. participation-weighted earliness。

每一项报告 signed Spearman rho 与 absolute rho。另报告在五项中取最大 absolute rho
的 omnibus score；每次置换必须重新选择最大项，控制 readout 选择成本。

### 4.3 Null 与统计

- primary null：患者内 all-contact label permutation；
- 同一次 permutation 对该患者所有 seizure、seed 和 readout field 一致应用；
- 每次 permutation 重新计算 five-field maximum；
- 5000 draws；
- 先 seizure，再 seed，再 patient 折叠；
- cohort 检验比较 patient observed 与 patient null median；
- 不要求每位患者单独超过其 95% null；
- within-shaft null 仅作更严格的解剖敏感性，不取代 all-contact primary null。

### 4.4 比较

主要报告：

- empirical heldout rank distribution；
- full-history GRU；
- structured full RNN。

对照/敏感性报告：

- ordinary static/no-order/last-set/rank-shuffle controls；
- structured node/no-history/isotropic/axis-no-source controls；
- low-rank rank-0 sensitivity。

结构化模型只在 strict clinical cohort 与 physical-axis cohort 的交集报告，与普通 GRU
的 16 人结果分母分开；共同 11 人用于 paired model-family comparison。

## 5. 代表性患者图

患者不能按 early-ictal transfer 结果挑选。固定：

- `epilepsiae_1084`：多发作、分母大；
- `epilepsiae_958`：既有双向传播背景；
- `epilepsiae_1096`：用于展示模型与数据高度一致的对照。

每位患者展示：

1. empirical contact × normalized-rank distribution；
2. full-history GRU distribution；
3. structured full distribution；
4. participation/earliness/endpoint fields；
5. strict clinical-onset seizure-median energy field；
6. model field 与能量的直接 scatter。

图的 contact ordering 由 empirical distribution 冻结，不按发作期 target 重排。

## 6. 结论边界

本轮可支持的最高结论是：

> 间期 rank sequence 中的 contact-level 分布及其双侧传播信息，能否被普通或结构化
> RNN 保留，并以 source-free 的方式与发作早期静态能量场相对应。

不能把：

- A/B alignment 当金标准；
- seed stability 当轴可辨识；
- next-contact 增益当作机制证明；
- source-free static similarity 当作发作动态传播预测；
- EEG-onset sensitivity 写成 clinical-onset 结果。

逐发作 source-conditioned 动态迁移仍需 exact clinical-onset contact set；在该 metadata
缺失时保持 `BLOCKED`，不能用 energy-top contacts 反推 source。
