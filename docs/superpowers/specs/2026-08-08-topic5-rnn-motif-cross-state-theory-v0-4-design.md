# Topic 5 — RNN motif benchmark、跨状态场复用与计算机制 v0.4

状态：**LOCKED FOR EXECUTION 2026-08-09**  
日期：2026-08-09  
承接：`2026-08-08-topic5-wiring-economy-spatial-latent-rnn-v0-3-design.md`  
建议结果根目录：`results/topic5_rnn_motif_cross_state_benchmark_v0_4/`

---

## 0. 一句话目的

只用患者自己的间期 contact-rank sequences 训练一组连接结构不同、但任务和读出完全相同的 RNN；
模型冻结后，用它们自由生成的 contact-space 传播场去检验论文已有的
**间期传播场—发作早期 broadband energy field 对应**，最后分析哪些有效连接与计算 motif
能以较少布线稳定地产生这种癫痫样传播。

本轮不是寻找单一“最好的 RNN”，而是辨识一个**充分模型集合**：

\[
\text{连接约束}
\rightarrow
\text{间期传播计算}
\rightarrow
\text{冻结模型生成的空间场}
\rightarrow
\text{early-ictal 外部对应}
\rightarrow
\text{有效计算 motif}.
\]

---

## 1. 三个科学问题必须分开回答

### Q1｜任务充分性

\[
\boxed{\text{哪些 recurrent connectivity motifs 足以学习 held-out 间期传播？}}
\]

这里的“学会”同时看：

1. teacher-forced next-rank contact NLL；
2. 只给 held-out event 第一个 rank set 后的 same-start free rollout；
3. 相对 no-recurrence 和 order-shuffle 的患者内增量；
4. 事件长度与 STOP 不发生灾难性塌缩。

目标不是证明所有模型相同，而是画出不同连接结构在**传播保真度—布线成本**上的充分性边界。

### Q2｜跨状态外部对应

\[
\boxed{
\text{冻结 RNN 从间期事件生成的 contact field，是否复现数据中的}
\ \text{interictal–early-ictal spatial correspondence？}
}
\]

early-ictal target 不参与训练、early stopping、seed 选择、wiring strength 选择或模型增删。
它只是所有模型冻结后的共同外部 benchmark。

这里预测的是**患者级候选空间场**，不是：

- clinical onset 时间；
- 某一次 seizure 是否发生；
- 单次发作逐触点传播顺序；
- 距离发作还有多久。

### Q3｜有效计算 motif

\[
\boxed{
\text{哪些加权后的有效连接组织，使网络更容易以经济布线生成癫痫样传播？}
}
\]

本题分析的是 task-trained network 的**有效影响和干预响应**，不是把二值 mask 当作患者真实 anatomical connectome。

---

## 2. 本轮最重要的三项设计修正

### 2.1 leaky RNN 是主 benchmark，GRU 是架构复现

v0.3 已证明空间布线优势在 plain leaky RNN 中存在，但在 GRU 中没有复现。
因此本轮不能把所有主模型静默改成 GRU。

- **Primary cell family**：masked leaky RNN；
- **Architecture replication**：masked GRU；
- 只有两个 cell family 方向一致时，才写“architecture-general”；
- 只在 leaky RNN 中成立时，写成 cell-specific computational result。

### 2.2 不再用科学 gate 阻止 early-ictal benchmark

所有预先定义且工程合格的模型都会完成间期评价、场冻结和 early-ictal 外部评分。
“task adequate”只决定它能承载多强的机制解释，不决定是否允许看 target。

唯一硬停条件是工程/数据完整性问题：泄漏、错位、非有限值、未收敛、产物缺失或 target seal 顺序错误。

### 2.3 理论分析以有效影响图为主，不以二值拓扑为主

v0.3 显示：order-shuffle 后 binary topology 基本不变，但生成性能明显下降。
因此最合理的解释是：

\[
\text{先验决定可用骨架，训练数据主要塑造权重和有效计算。}
\]

本轮承重对象固定为：

- weighted recurrent graph；
- contact-to-contact effective influence；
- transient amplification / persistence；
- matched perturbation / lesion 后的传播损失。

binary modularity、clustering 与 edge count 只作结构描述，不能单独承载“模型学到病理连接”的结论。

---

## 3. 数据与统计单位

### 3.1 间期 cohort

沿用 v0.3 冻结的 21 名患者、31 个 fit：

- 11 名 shared-plane 患者：每人一个 `shared_all` fit，模型内同时包含两种事件；
- 10 名 non-collinear 患者：每人 `own_a`、`own_b` 两个 fit；
- patient 是唯一 cohort 统计单位，fit、event、seed 和 edge 都不是独立生物学样本。

fit→patient 的聚合按问题区分：

- **Q1 性能**：non-collinear 患者先平均 `own_a/own_b` 的同名性能量；
- **Q2 场**：不得预先平均。`own_a` 保留为 `F_A`，`own_b` 保留为 `F_B`，
  逐 seizure 做 `maxAB` 后才进入 patient median；
- shared 患者由同一个 fit 按 post-hoc A/B labels 分别形成 `F_A/F_B`。

这一映射必须写入 `FIT_TO_PATIENT_AGGREGATION_CONTRACT.json`，逐患者列出 field producer。

所有 rank 输入必须来自 phantom-rank 修复后的 masked dataset，未参与 contact 不得携带伪 rank。

### 3.2 A/B 标签

A/B 标签只用于 held-out 事件的训练后分层和场构建：

- 不进输入；
- 不进 loss；
- 不参与 epoch、seed 或 motif 选择；
- 必须沿用 v0.3 的 `valid_idx → event_source_index → mode` 对齐审计。

### 3.3 early-ictal 外部 cohort

沿用论文当前冻结合同：

- strict clinical onset；
- onset 后 `[0,10] s`；
- 1–150 Hz baseline-normalized broadband contact energy；
- exact contact-name join；
- seizure 内先算、patient-first 聚合；
- 预期 strict broadband 为 16 人 / 106 seizures；
- `epilepsiae_1146` 因既往 development / target-seal incident 仅作 supportive；
- primary external benchmark 预期 15 人。

运行前只能做 metadata inventory 和哈希核对。若实际交集不是预期 15 人，必须先报告具体 join 差异，
不能静默改变分母。

### 3.4 target 独立性边界

该 early-ictal target 在项目更早阶段已经被查看过，因此本轮是**target-free training 后的内部外部基准**，
不是全新独立数据集确认。必须做到：

1. v0.4 模型矩阵、训练配置和统计合同先冻结；
2. 全部 interictal checkpoints 与 model-generated fields 先写 manifest；
3. 再读取 early-ictal 数值；
4. 读取后不修改模型、超参数、field builder 或 exemplar。

### 3.5 geometry 状态

本轮所有 arm 使用同一几何，因此模型间比较公平；但 tissue plane 来自完整记录的既有传播几何，
不是仅用当前 train split 重估。`RUN_CONTRACT.json` 和 Figure A 图注必须写：

```text
GEOMETRY_STATUS: RETROSPECTIVE_TEST_INFORMED_PROPAGATION_PLANE
```

因此不得写“模型从训练事件独立发现了患者传播轴”。可在部分患者做 train-only plane sensitivity，
但不以此阻断主矩阵。

---

## 4. 共同计算合同

所有模型共用：

- 同一患者 tissue plane、node positions 和毫米尺度；
- 同一 tied local operator `H^T` 注入 / `H` 读出；
- 同一 contact bias、STOP head 和 eligible-contact mask；
- 同一 train/validation/test split；
- 同一 next-rank + STOP loss；
- 同一 optimizer、batch、训练上限、早停逻辑与 convergence audit；
- 同一 seed 集 `0/1/2`；
- 同一 same-start deterministic set decoder；
- 同一患者内聚合和统计。

`t_norm` 只能由当前 rank index 和固定 contact denominator 构造；禁止使用事件最终长度、
最终参与数或任何未来 prefix 信息。

禁止：

- dense contact-to-contact skip；
- A/B、SOZ、seizure 或 early-ictal 特征作为输入；
- 对 early-ictal target 训练 readout；
- 根据 early-ictal 结果挑 epoch、seed 或 `eta`；
- 把 hidden unit 或 recurrent weight 的正负号命名为 E/I；
- 把 RNN graph 称为真实 anatomical connectivity。

### 4.1 主 cell：leaky RNN

沿用 v0.3：

\[
u_t=H^\top x_t,
\]

\[
h_t=(1-\kappa)h_{t-1}+\kappa\tanh\left(a\odot u_t+(M\odot W)h_{t-1}+b\right),
\]

\[
\ell_{t+1}=b_{contact}+\alpha Hh_t.
\]

每个 tissue node 一个状态标量。`M⊙W` 是唯一跨 node recurrent pathway。

### 4.2 GRU replication

沿用 v0.3 的 shared mask GRU：同一个 node mask 约束 reset、update、candidate 三个 recurrent matrices。
GRU 与 leaky RNN 只在同一 cell family 内比较 motif；不以参数量不同直接解释生物优劣。

### 4.3 训练参数

沿用 v0.3 已审计配置，不再做新超参数搜索：

- density `rho=0.10`；
- `d0=10 mm`；
- warmup 10 epochs；
- rewire 40 epochs，`zeta: 0.20→0`；
- mask 冻结后才允许 early stopping；
- patience 12，`min_relative_improvement=1e-4`；
- freeze phase 上限 3,000 epochs（沿用 v0.3 收官时为消除静态基线欠收敛而统一抬高的上限；
  recurrent units 仍由早停结束）；
- batch `min(1024, ceil(n_train/8))`，每 epoch 至少 8 updates；
- 每 epoch 最多 120 batches；
- optimizer `Adam`，learning rate `0.006`，gradient-norm clip `5.0`；
- batch 不能作为显存旋钮，资源不足只调 worker 并发。

`eta` 不是 target-driven sweep。固定为三个预先定义的连续点：

\[
\eta\in\{0.01,0.03,0.10\}.
\]

三档全部报告，不从 early-ictal 结果中挑一档；`0.03` 是 v0.3 已冻结中档。

### 4.4 wiring-cost 跨模型归一化

每个 active node pair 的 edge magnitude 固定为：

\[
S_{ij}^{RNN}=|W_{ij}|,
\qquad
S_{ij}^{GRU}=\sqrt{(U_{r,ij}^2+U_{z,ij}^2+U_{h,ij}^2)/3}.
\]

即 GRU 使用 gate-RMS，而不是三个矩阵直接求和。之后：

\[
C_{wire}=\frac{1}{|E|}\sum_{ij\in E}S_{ij}\frac{d_{ij}}{d_0}.
\]

这样成本已经按 active edge count、gate 数和固定物理尺度 `d0` 归一化。每个 run 在 initialization、
rewire midpoint、mask freeze、final 保存：

\[
r_{cost}=\frac{\eta C_{wire}}{L_{task}}.
\]

数值相同的 `eta` 仍不等于两个 cell family 受到完全相同的功能压力；跨 cell 只比较实际 `r_cost` 分布。

### 4.5 free-rollout decoder：预测 set size，不读取真实未来

next rank 可以包含多个 contacts，因此单一 argmax 不是正式生成合同。本轮选择低容量 **size-head decoder**：

1. recurrent checkpoint 按原 next-rank + STOP 任务训练；
2. checkpoint 冻结后，只用 interictal **train split** 的 teacher-forced states 拟合共同结构的 size head；
3. validation split 只用于 size-head early stopping；test set 不参与 calibration；
4. size head 输入固定为 `[mean(h_t), max(h_t), t/C_p, recruited_fraction]`；
5. 输出 `K=1,...,C_p` 的 categorical distribution；M0 的 hidden summaries 固定为 0；
6. 所有 arms 使用相同的 `4→16→C_p` MLP、optimizer、训练预算和 early stopping。

自由生成每一步固定执行：

```text
if p_STOP >= 0.5: terminate
else:
    K = argmax p(K | current state)
    K = min(K, number of eligible contacts)
    choose top-K eligible contact logits
```

- STOP 优先；允许 seed 后立即停止；
- 已招募 contacts 永久 mask；
- 不读取 observed next-set size；
- contact-logit 或 size-logit 精确并列时按冻结 `contact_order` / 较小 K 处理；
- 最大 rank steps 为 `C_p`，因为每个非终止 step 至少新增一个 contact；
- conditional contact NLL 可以读取 observed `K`，但只作诊断，绝不能驱动 free rollout。

运行前写 `ROLLOUT_DECODER_CONTRACT.json`，记录 size-head config、STOP precedence、mask、tie、
训练/验证 keys、代码与 calibration hashes。v0.3 recurrent checkpoints 只有在 §5.4 全部 hash 一致时才能复用，
但每个复用 checkpoint 仍必须新拟合该 decoder。

---

## 5. 模型矩阵

### 5.1 leaky RNN 主矩阵

| ID | recurrent connectivity | 目的 |
|---|---|---|
| `M0_NO_REC` | 无 recurrence；contact prior + STOP | 无循环基线 |
| `M1_DENSE` | all-to-all tissue recurrence | 高容量参照 |
| `M2_UNIFORM_SET` | 10% edges；uniform prune/regrow；无 wiring cost | 普通稀疏 recurrence |
| `M3_FIXED_LOCAL` | degree-balanced connected local graph；10% edges；不 rewire；无 cost | 公平的纯局部骨架 |
| `M4_SPATIAL_GROWTH` | distance-biased prune/regrow；`eta=0` | 距离偏置生长本身 |
| `M5_SPATIAL_LOW` | M4 + `eta=0.01` | 弱布线成本 |
| `M6_SPATIAL_MID` | M4 + `eta=0.03` | v0.3 主配置 |
| `M7_SPATIAL_HIGH` | M4 + `eta=0.10` | 强布线成本 |
| `M8_UNIFORM_COST_MID` | uniform regrow + `eta=0.03` | 生长规则×成本的 factorial sidecar |

所有 `M0–M8` 正式模型均运行 seeds `0/1/2`。M0 只有在 smoke 中证明三 seed 的训练轨迹和输出逐位一致后，
才允许在正式调度中物理运行一次并将结果确定性复用到三 seed；否则必须实际运行三次。

### 5.1a 核心 2×2 factorial

核心连接实验固定为：

| | `eta=0` | `eta=0.03` |
|---|---:|---:|
| uniform regrowth | M2 | M8 |
| spatial regrowth | M4 | M6 |

M2/M4/M8/M6 使用完全相同的 density、初始化资源、weakest-edge pruning、`zeta` schedule、
rewire epoch 和训练器。唯一变化是 regrowth proposal 与 `eta`。

对任何已转成“越大越好”的患者级 endpoint `Y`（NLL 使用 `Y=-NLL`）预定义：

\[
\Delta_{growth,\eta=0}=Y_{M4}-Y_{M2},
\]

\[
\Delta_{growth,\eta=mid}=Y_{M6}-Y_{M8},
\]

\[
\Delta_{cost,uniform}=Y_{M8}-Y_{M2},
\]

\[
\Delta_{cost,spatial}=Y_{M6}-Y_{M4},
\]

\[
\Delta_{interaction}=(Y_{M6}-Y_{M4})-(Y_{M8}-Y_{M2}).
\]

五项同时用于 interictal fidelity、wiring cost 和 early-ictal null-relative margin；统计先 seed 内平均、
再 fit→patient 聚合。

### 5.1b `M3_FIXED_LOCAL` 的公平构图

禁止使用“全局最短 10% pairs”。固定算法：

1. 计算 edge budget `E=round(0.10 M(M-1))`；
2. 先加入 Euclidean minimum-spanning tree 的双向 edges，保证所有 nodes 强可达；
3. 在剩余预算内按 node 当前 out-degree 从低到高轮转，为其加入最近的尚未连接邻居；
4. out-degree 最大差不超过 1；若 distance tie，按冻结 node index；
5. mask 全程固定，不允许 self-loop。

验收：

```text
minimum_in_degree >= 1
minimum_out_degree >= 1
number_weak_components == 1
number_strong_components == 1
fraction_H_supported_nodes_in_main_component == 1
edge_count == sparse_budget
mask invariant across training
```

不同 seed 只改变权重初始化，不改变 mask。

### 5.2 训练顺序对照

`C_ORDER_SHUFFLED`：对 `M6_SPATIAL_MID` 固定真实 rank 1，仅将 rank 2 到 rank T 的完整 rank sets
随机置换；每个 set 内 contacts 不变。保持 source distribution、参与 contacts、event size、split、
geometry、optimizer 与 seeds 不变。每名患者的 shuffle mapping 预先固定并写 hash。

`C_FULL_RANK_SHUFFLED` 将 rank 1 也置换，只作一 seed sensitivity，不进入 primary `G_order`。

两者都是信息对照，不是 biological motif。

### 5.3 GRU replication 矩阵

只复现五个核心点：

`M0_NO_REC`、`M1_DENSE`、`M2_UNIFORM_SET`、`M3_FIXED_LOCAL`、`M6_SPATIAL_MID`。

不在 GRU 中重复 `eta` 剂量和 factorial sidecar，避免把架构复现扩成新模型 zoo。

### 5.4 checkpoint 复用

v0.3 产物只有在以下字段逐项一致时才能复用：

- source-code hash；
- dataset / plane / `H` hash；
- split；
- model config；
- optimizer / batch / epochs / early stopping；
- seed；
- observation-scale policy。

任一字段不同就训练新单元。复用与新训练必须在 manifest 中明确标记。

---

## 6. Q1：间期任务充分性

### 6.1 四个患者级核心量

令 `R_p^m` 为 same-start free rollout 的 held-out propagation-rank Spearman correlation：

- 只给真实第一个 rank set；
- 后续完全使用 §4.5 冻结的 set-size + STOP decoder；
- 评分时从观察与生成两侧都删除白送的 seed contacts；
- 每 event 先算，再在 fit、seed、patient 层依次聚合。

定义：

\[
G_p^{rec}=R_p^m-R_p^{M0},
\]

\[
G_p^{order}=R_p^{M6}-R_p^{C_{shuffle}},
\]

\[
G_p^{spatial}=L_p^{M2}-L_p^{M6},
\]

\[
D_p^{dense}=L_p^{M6}-L_p^{M1}.
\]

其中 `L` 是 held-out next-rank contact NLL；正的 `G_spatial` 表示 spatial-mid 更好。

同时报告：

- conditional contact NLL（已知 continue 和 observed next-set size）；
- STOP NLL；
- generated / observed length ratio；
- top-1 contact accuracy。

这样可以区分模型学到“传播到哪里”还是只学到“何时停止”。

### 6.2 患者内 bootstrap

对同一批 held-out events 做 paired bootstrap，输出每名患者：

- `R_p^m`；
- `G_rec` 95% CI；
- `G_order` 95% CI；
- 描述性 `R>0`；
- 严格 `CI(G_rec)>0` / `CI(G_order)>0`。

主文不能只报多少患者 individually significant；必须同时给 21 个患者原始点、患者级 median/CI 和正向人数。

### 6.3 dense benefit retention

固定 `minimum_dense_benefit=0.01 nats/decision`。仅当

\[
L_p^{M0}-L_p^{M1}>0.01
\]

时定义患者级 retention：

\[
B_p^m=\frac{L_p^{M0}-L_p^m}{L_p^{M0}-L_p^{M1}}.
\]

`B=1` 表示保留 dense recurrence 的全部 NLL benefit；`B=0.9` 表示保留 90%。
报告患者级分布与 bootstrap CI。90% 是预先固定的解释线，不是显著性 gate。

其余患者 `B` 标为 unavailable，但原始 NLL 和 rollout 仍保留。

正式 non-inferiority 不依赖该比值。用预注册的 8 个 development fits、只依据 interictal validation，冻结：

\[
\delta_{NI}=0.10\times
\operatorname{median}_{dev}\left(L^{M0}-L^{M1}\right)_{positive}.
\]

对模型 `m` 检验患者级 `L_m-L_M1` 的 bootstrap upper 95% CI 是否小于 `delta_NI`。
`delta_NI` 必须在 full-cohort test scoring 和 target unseal 前写入合同。

### 6.4 noise ceiling 的正确用途

held-out event-pair reliability `rho_p` 只提供一个测量参照，确定性 predictor 的经典参照为 `sqrt(max(rho_p,0))`。

- 报告 `R_model / sqrt(rho)` 和两者差；
- 不得把“与 ceiling 差异不显著”写成“达到天花板”；
- 只有预先定义 equivalence margin 并通过等价检验，才能使用“statistically equivalent”。

本轮不把 ceiling 当通过门。

### 6.5 task-adequacy tier 不是 stop gate

每个模型标记：

- `ADEQUATE_STRONG`：cohort `G_rec>0`、rollout 不塌缩，且相对 dense 通过上述 non-inferiority；
- `ADEQUATE_PARTIAL`：cohort `G_rec>0` 且 rollout 不塌缩，但未通过 non-inferiority；
- `INADEQUATE`：无正 recurrence gain 或生成明显塌缩；
- `ENGINEERING_INVALID`：未收敛、NaN、错位或产物不完整。

前三类都进入 early-ictal benchmark；只有 `ENGINEERING_INVALID` 不进入。

---

## 7. 冻结模型生成的 contact fields

### 7.1 为什么不用 raw adjacency

raw adjacency 不在 contact space，且受到 hidden basis、gating、mask prior 和权重尺度影响。
跨状态比较固定使用模型的**可观察输出**：same-start free rollout 的 contact recruitment field。

### 7.2 两个冻结 field endpoint

对 held-out event `e`，只给真实第一个 rank set，生成后续顺序。两个 field 合同并列冻结，不能在看到 target 后互换主次。

#### `FIELD_CANONICAL_FULL`｜跨 Human–RNN–SNN 的 primary

真实 seed contacts 作为生成序列 rank 1，随后生成 rank 2...T；被生成 contact 的分数为：

\[
s^{full}_{e,c}=1-\frac{\hat r_{e,c}-1}{\max(\hat T_e-1,1)},
\]

未生成 contact 为 0。该场与 empirical data / SNN 的 canonical full-event field builder 同构，
用于 primary cross-modal comparison。

#### `FIELD_SEED_REMOVED`｜recurrence-specific key secondary

删除白送的 seed contacts，将后续生成 rank 重新从 1 编号：

\[
s^{rec}_{e,c}=
\begin{cases}
1-\dfrac{\hat r^{postseed}_{e,c}-1}{\max(\hat T^{postseed}_e-1,1)}, & c\text{ 在 seed 后被生成};\\
0, & c\text{ 未被生成};\\
\mathrm{missing}, & c\text{ 是 seed}.
\end{cases}
\]

对 contact `c` 的聚合分母固定为“该模式下 `c` 不是 seed 的 held-out events 数”，并保存该分母；
不得把 frequent-source contact 的缺测自动补零。它检验跨状态对应是否超越已提供起点。

M0 no-recurrence 和 `C_ORDER_SHUFFLED` 在两个 endpoint 中都保留，用于拆解 canonical full-field 对应
是否仅由起点频率或静态 scaffold 驱动。

### 7.3 A/B、common 与 contrast field

用训练后 A/B 标签分组 held-out events，分别对 `full` 与 `seed_removed` 计算：

\[
F_A^m(c)=\operatorname{mean}_{e\in A}s_{e,c},\qquad
F_B^m(c)=\operatorname{mean}_{e\in B}s_{e,c}.
\]

\[
F_{common}^m=\frac{F_A^m+F_B^m}{2},\qquad
F_{contrast}^m=F_A^m-F_B^m.
\]

另外保存生成 participation probability 作为诊断量，不允许用 observed future participation 补模型场。

为避免不同模型通过缩小 support 得到更高相关，primary R3 evaluation support **不能由各模型的生成 participation 决定**。
每名患者、每个 A/B candidate 使用同一份 target-free common support：exact-joined scored contacts 的几何核支持
与冻结 empirical interictal candidate support 的交集。该 support 在所有模型间逐位相同。
model-derived participation support 只作 sensitivity 和生成质量检查。

### 7.4 target-free empirical fidelity

读取 early-ictal 数值之前，比较模型场与冻结 empirical interictal A/B timing fields：

- A/B matched field fidelity；
- common-field fidelity；
- contrast-field fidelity；
- mode collapse：`corr(F_A,F_B)` 与 empirical A/B separation；
- split-half / seed stability。

shared 11 人单独报告一个模型内的 A/B retention；non-collinear 10 人只能报告各自 fit 的 field fidelity，
不得写成“同一个模型产生两种模式”。

Q2 中 non-collinear 患者固定：

\[
F_{p,A}=F_{p,own\_a},\qquad F_{p,B}=F_{p,own\_b}.
\]

在每次 seizure 上先对这两个 candidates 做 maxAB，再取 patient median。禁止在 maxAB 前把两个 fit
平均成 common field。shared 患者的 A/B candidates 来自同一 fit 的 post-hoc event groups。

### 7.5 field manifest

target unseal 前必须冻结：

- checkpoint hashes；
- event keys、seed contacts 与 A/B labels；
- `FIELD_CANONICAL_FULL` 和 `FIELD_SEED_REMOVED` 各自的 `F_A/F_B/common/contrast` contact vectors；
- participation support；
- contact order、geometry、sigma、grid bounds；
- model/seed aggregation rule；
- representative patient；
- plotting order；
- source-code hashes。

同时冻结 `FIT_TO_PATIENT_AGGREGATION_CONTRACT.json`，逐患者写明 shared-single-fit 或 own_a/own_b producer，
并以单元测试验证 Q1 与 Q2 使用不同的合法聚合顺序。

---

## 8. Q2：early-ictal external benchmark

### 8.1 与论文相同的 primary statistic

对每个冻结模型场，使用现有 R3 dense-grid engine。`FIELD_CANONICAL_FULL` 是 cross-modal primary，
`FIELD_SEED_REMOVED` 是 recurrence-specific key secondary；两者使用同一 scorer：

1. model contact values + 模型间共同冻结的 target-free evaluation support 重建 interictal candidate field；
2. early-ictal contact energy 重建 ictal field；
3. identity / transverse-mirror 分别做 support gate；
4. 对 A、B 分别选择 absolute best mirror candidate；
5. `score_maxAB=max(score_A, score_B)`；
6. seizure 内先算，patient-first 聚合。

Primary activation：clinical onset `[0,10] s`、1–150 Hz broadband。

### 8.2 主 null

Primary null 为患者内 coherent all-contact label shuffle，`n_perm=5000`。

每个 draw 必须完整重做：

`permute activation → rebuild field/support → mirror candidates → A/B maxAB`。

同一 patient×seizure×draw 的 permutation mapping 在所有模型、A/B 与 empirical reference 间共享。

pure within-shaft shuffle 只作更严格解剖 sensitivity；不能替代 primary all-contact null，
也不能在不合法时回退成全患者 shuffle。

### 8.3 每模型输出

每名患者、每个模型输出：

- observed `maxAB`；
- all-contact null median；
- `margin = observed-null median`；
- empirical interictal A/B field 的 reference score；
- no-recurrence、dense、uniform sparse、fixed-local、spatial continuum 的 paired contrasts；
- common-field score；
- canonical-full 与 seed-removed score / margin；
- within-shaft sensitivity 的实际 denominator。

### 8.4 common 与 contrast 必须并列报告

`maxAB` 允许模型只匹配 A 或 B 中更像的一张，可能掩盖 mode collapse。因此两个 field endpoint 内都报告：

- `maxAB`：论文同构的 primary cross-state statistic；
- `common-field concordance`：共同招募骨架；
- `contrast fidelity`：A/B 区分是否保留，仅作 secondary；
- 不得用单一 `maxAB` 宣称模型恢复了两种模式。

### 8.5 预设比较

按患者配对，统一报告效应量、bootstrap CI、positive count、Wilcoxon 与同步 permutation p：

1. 每个模型 `margin>0`；
2. recurrent motif vs `M0_NO_REC`；
3. `M6_SPATIAL_MID` vs `M2_UNIFORM_SET`；
4. `M6_SPATIAL_MID` vs `M1_DENSE`；
5. `eta={0.01,0.03,0.10}` 的有序趋势；
6. leaky RNN vs matched GRU。

2×2 factorial 的五个 contrasts（§5.1a）是 core family；M5/M7 只进入预定义 dose trend。

核心模型间的直接 contrasts 用 Holm correction；所有模型的原始患者点不隐藏。

### 8.6 控制间期拟合后的 model effect

模型 early-ictal margin 不能仅因它把 interictal task 做得更好而被解释为特殊 inductive bias。
除 raw paired contrasts 外，对 primary 15 人拟合 patient-blocked model：

\[
E_{p,m}=\alpha_p+\beta I_{p,m}+\gamma_m+\epsilon_{p,m},
\]

其中 `E` 是 early-ictal null-relative margin，`I` 是 target-free interictal field fidelity
（预先固定 primary 为 model-vs-empirical A/B canonical-field fidelity），`alpha_p` 是 patient intercept。

推断使用 10,000 次 patient-cluster bootstrap 和 patient-level label/sign permutation，不使用普通独立残差标准误。
同时报告 raw paired contrasts。只有 `gamma_m` 在控制 `I` 后仍有稳定差异，才可写：

> 该 wiring constraint 不只是提高间期拟合，还形成了更符合跨状态场的 inductive bias。

### 8.7 允许和禁止的解释

允许：

> 某类 target-blind、task-trained recurrent connectivity constraint 生成的患者级间期场，
> 在同一患者内与 early-ictal broadband energy field 保持了超过 all-contact shuffle 的空间对应。

禁止：

- “模型预测了这一次 seizure”；
- “RNN state 触发了发作”；
- “模型恢复了患者真实连接组”；
- “间期事件按原顺序在发作中 replay”；
- “某类 RNN 是真实 E/I 回路”。

---

## 9. Q3：效果较好模型的理论分析

### 9.1 分析集合

target unseal 前冻结 primary theory set：

```text
M1_DENSE
M2_UNIFORM_SET
M3_FIXED_LOCAL
M4_SPATIAL_GROWTH
M6_SPATIAL_MID
M8_UNIFORM_COST_MID
C_ORDER_SHUFFLED
```

这些模型全部做 effective influence、local/connector summary、geometry/proposal null 和 core matched lesion，
不根据 early-ictal 排名增删。

每个模型仍放在 interictal fidelity、early-ictal margin、wiring length 三轴上展示 Pareto frontier，
但 frontier 只决定 `M5/M7` 是否进入额外的 `TARGET_INFORMED_EXPLORATORY_THEORY`，不能改变 primary theory set，
也不能用探索模型再次证明选择它的 target。

### 9.2 contact-space effective influence

不能把 k>1 的 teacher-forced derivative 与 open-loop propagation 混在一起。固定两个量。

#### A. lag-1 local teacher-forced Jacobian

在真实 held-out prefix 上，对 contact probability 求导：

\[
I^{TF,1}_{ij}=
\mathbb E_{prefix}
\left[
\frac{\partial P(x_{j,t+1}=1)}{\partial x_{i,t}}
\right].
\]

binary input 以 `[0,1]` 连续松弛求局部导数，只解释 lag-1 sensitivity。

#### B. lag-1/2/3 open-loop finite pulse response

对一个真实 prefix 的当前 tissue input 加标准 pulse：

\[
\Delta u_i=A_p\frac{H^\top e_i}{\|H^\top e_i\|_2},
\]

其中 `A_p` 是该患者 train split 中 `||H^T x_t||_2` 的中位数，确保每个 contact 的 tissue-input
总能量一致。之后不输入真实 future ranks，base 与 pulse 两臂都用 §4.5 deterministic decoder open-loop，
记录每步选择前的 contact probabilities：

\[
\Delta P^{(k)}_{ij}=P^{pulse}_{j,t+k}-P^{base}_{j,t+k},\qquad k=1,2,3.
\]

只对尚未招募 contact 施加 pulse。自动微分 lag-1 与有限差分 lag-1 必须在 toy/held-out subset 上方向一致；
k=2/3 不穿过 argmax 反传。

两个量都输出 signed/absolute influence、decay、same/cross shaft、distance bins、A/B-axis aligned/transverse、
seed 与 split-half stability。probability change 是跨 leaky/GRU 的共同尺度。

### 9.3 计算动力学

主文级 Q3 只冻结三项：

1. lag-1/2/3 contact-space effective reach；
2. local-backbone / long-range connector organization；
3. targeted vs matched lesion specificity。

spectral radius、singular/transient gain、non-normality、path diversity 和 hidden eigenmodes 全部降为
exploratory/Supplementary，且 hidden-state Jacobian 只能在同一 cell family 内比较。
rank-step persistence 不解释成秒级生物时间常数。

### 9.4 候选 network motif

在 target unseal 前预先冻结候选：

> 局部 recurrent backbone + 少量高影响的跨区/长程 connector edges，
> 是否以较低布线支持广范围、可双向启动的传播。

连续量是 primary；为 lesion 选边时固定阈值：

- `local backbone edge`：长度不超过该患者所有 candidate pairs 的 Q50，且 absolute pulse influence
  不低于 active local edges 的 Q75；
- `long-range high-influence edge`：长度不低于 candidate-pair Q75，且 absolute pulse influence
  不低于全部 active edges 的 Q90；
- `connector node`：与上述长程边相连，且 weighted participation coefficient 不低于患者 Q75。

若某患者任一集合少于 3 条 edges / 2 个 nodes，记为 `motif_not_estimable`，不移动阈值。

每个候选 motif 必须同时满足：

1. 超过 matched geometry / proposal null；
2. 跨 seed 稳定；
3. 与 held-out propagation fidelity 有患者级关系；
4. 删除后性能下降超过 matched deletion；
5. 不是二值生长规则直接写入的平凡结果。

### 9.5 matched lesions

不重训，分别删除：

- highest effective-influence edges；
- long-range high-influence connectors；
- local backbone edges；
- high participation-coefficient nodes。

每个 lesion 与随机对照匹配：

- edge/node 数；
- 总绝对权重；
- in/out degree；
- edge-length distribution；
- spatial extent。

固定匹配合同：

```text
target_draws = 500
minimum_valid_matched_draws = 200
sampling_without_replacement_within_draw = true
sampling_across_draws = with_replacement
edge_count = exact
total_abs_weight_caliper = ±10%
mean_edge_length_caliper = ±10%
mean_endpoint_in_out_degree_caliper = ±1
spatial_extent_caliper = ±10%
```

node lesion 使用 node count exact、incident absolute weight ±10%、mean degree ±1、spatial radius ±10%。
不足 200 个合法 draws 时，患者级 lesion inference 标为 unavailable，只保留描述性 targeted effect，
不放宽 caliper。

评价：next-rank NLL、same-start rollout、event length、A/B/common field fidelity。
early-ictal target 本身不参与 lesion 选择。lesion 后冻结生成场可重新送入既有 early-ictal scorer，
`E_intact-E_lesioned` 只作 secondary cross-state perturbation readout。

### 9.6 training trajectory

保存初始化、rewire 中段、mask freeze、收敛四个 checkpoint，追踪：

- binary mask；
- weighted graph；
- effective influence；
- interictal task fidelity；
- wiring cost。

只有在全部 checkpoint 预先冻结后，才可事后画 early-ictal concordance trajectory；该轨迹是诊断图，
不能用来选 epoch。

### 9.7 何时能说“更容易产生癫痫样传播”

至少需要：

- 同等训练预算下更快/更稳定达到 task-adequate；
- 相同或更少 edge/wiring 下保持传播 fidelity；
- matched lesion 对关键 motif 产生特异损害；
- 多 seed / 患者方向一致；
- early-ictal external benchmark 同向，但不是训练来源。

若只有 topology enrichment 而无 perturbation 效应，只能说“伴随”，不能说“更容易产生”。

---

## 10. 与 human data 和 SNN 的统一可比接口

不做 edge-to-edge 对照，只比较同层级的 mesoscopic observables：

| observable | Human data | RNN | SNN |
|---|---|---|---|
| interictal rank field | empirical A/B full events | `FIELD_CANONICAL_FULL` held-out rollout | simulated full interictal event |
| bidirectionality | A/B endpoint reversal | opposite seed / A-B field | opposite ignition site |
| spatial reach | recruited contacts | generated contacts | recruited neural populations |
| common scaffold | A/B common field | `F_common` | shared spatial substrate |
| early recruitment | 0–10 s energy field | frozen candidate field concordance | early seizure-like recruitment field |
| perturbation | 不作因果声称 | matched lesion/pulse | existing SNN perturbation |

SNN 不作为本轮 gate，也不重新做参数恢复。RNN 负责计算充分性和可解释 readout，SNN 负责生物物理机制。

---

## 11. 统计合同

- biological unit：patient；
- shared-mode claim：n=11；
- interictal benchmark：n=21；
- early-ictal primary：expected n=15，E1146 supportive；
- non-collinear 聚合：Q1 先平均 own_a/own_b 性能；Q2 保留两个 candidate 到 per-seizure maxAB 后再聚合；
- seeds 先模型内平均，不当独立样本；
- paired patient bootstrap：10,000 draws；
- early-ictal all-contact null：5,000 synchronized draws；
- 主效应同时报告 median、95% CI、positive count、exact/tie-aware Wilcoxon；
- model-family direct contrasts 做 Holm correction；
- 不用“一个显著、一个不显著”推断两模型不同；
- 不以 P 值替代效应量和 patient raw points。

factorial 五 contrasts、raw early-ictal model effects 和控制 interictal fidelity 后的 model effects
必须进入同一统计产物，不能只展示最显著的一项。

---

## 12. 工程验收

每个新训练单元必须有：

- `config.json`；
- `input_hashes.json`；
- epoch log；
- convergence trace；
- `SNAPSHOT_INIT`、`SNAPSHOT_REWIRE_MID`、`SNAPSHOT_MASK_FREEZE`、`SNAPSHOT_FINAL`；
- best/final checkpoint；
- held-out predictions / rollouts；
- `graph.npz`；
- `DONE.json` 最后原子写入。

合法复用的 v0.3 final checkpoint 可缺少历史 snapshots，但必须标记 `snapshot_missing_reused_checkpoint`，
不得进入 training-trajectory inference。M6 true-order 与 order-shuffle 若缺 snapshots 必须重跑，保证主 trajectory 对照完整。

全链必须检查：

- 0 OOM / NaN / Inf；
- target 未进入训练对象；
- event/contact/A-B 对齐；
- identical config 可逐位复现；
- chunked / unchunked 评价一致；
- reused checkpoints hash 完全匹配；
- target unseal 发生在 field manifest 之后；
- figures 的每个点能追溯到 source CSV/JSON；
- `figures/README.md` 用中文逐图说明科学含义。

### 12.1 immutable execution worktree

正式执行不得在当前含 figure 脏改动的 worktree 中开始。锁定后：

1. 只提交 spec/plan 与经过验收的实现，不夹带旧 figure 修改；
2. 从明确 base commit 新建干净 branch/worktree `codex/topic5-rnn-motif-cross-state-v0-4`；
3. 将 launcher、model、trainer、decoder、scorer、field builder 复制到
   `results/.../run_snapshot/`；
4. active run 只执行 snapshot，禁止原地编辑；
5. 每个 stage 产物写入 `producer_code_hash`、`producer_config_hash`、`input_manifest_hash`、`created_at`；
6. aggregate 前核对 freshness、cohort revision 和 producer hashes；
7. launcher 变化必须产生新 run revision，不能续写旧目录。

仅保存 `git status` 不是充分保护。

---

## 13. 预期 Figure 6 结构

| Panel | 科学问题 |
|---|---|
| A | 同一 tissue/readout 下的 connectivity motif ladder |
| B | representative observed vs generated A/B events，明确 seed 与 free rollout |
| C | cohort interictal sufficiency：rollout、recurrence/order gain、NLL |
| D | task fidelity–wiring Pareto，不同 motif 的充分性集合 |
| E | canonical-full primary + seed-removed secondary 与 early-ictal maxAB/null-relative concordance |
| F | effective influence motif + matched lesion，解释哪种组织承载传播 |

图面不堆 gate 文字；统计放 raw patient points、简洁 CI 和星号，完整数值进 source data 与 README。

---

## 14. 预注册结论梯度

### Level 1｜多种 recurrence 足以学习间期传播

多个模型 `G_rec>0`，且 free rollout 不塌缩。

### Level 2｜某些连接约束更经济

在相近传播 fidelity 下，使用更少 edges / wiring length；不要求绝对超过 dense。

### Level 3｜跨状态场对应具有 motif 选择性

某些冻结模型相对 no-recurrence / matched alternatives 有更高 early-ictal null-relative concordance。

### Level 4｜可干预的计算 motif

有效影响结构跨 seed 稳定，matched lesion 特异损害传播，并与经济性和跨状态对应同向。

任何较低 Level 成立都可独立报告；不使用一个总 hard gate 把整条线压成 PASS/FAIL。
