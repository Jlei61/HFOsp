# Topic 5.1 Local-Backbone Selective-Shortcut RNN（LBSS-RNN）科学设计 v0.2

> 状态：已按 2026-08-11 审阅修订并锁定执行。除工程 bug、OOM 并发和已定义暂停条件外，不在真实结果出来后修改科学合同。

## 1. 一句话目标

在同一患者、同一局部 recurrent backbone、同一训练任务和匹配的新增边预算下，检验少量由真实间期有序传播选择的 nonlocal shortcuts，是否比额外局部边或随机 nonlocal shortcuts 更能生成远端间期传播，并使冻结模型场更接近同患者发作早期 broadband energy field。

## 2. 为什么需要这一分支

v0.4 已经证明：

1. recurrence 与真实 rank order 对间期传播生成很重要；
2. 多种 dense/sparse/local/spatial topology 都能完成任务；
3. wiring cost 可显著降低连接资源，但未显示 early-ictal 特异优势；
4. 局部有效影响富集稳定，但 wiring-cost 网络没有产生可确认的长程 connector motif。

因此，wiring economy 保留为通用效率 benchmark，不再被解释为癫痫网络形成原则。新分支直接测试更贴合病理传播的双尺度假设：

\[
\boxed{
\text{stable local recurrent backbone}
+
\text{few task-selected nonlocal shortcuts}
}
\]

本分支仍是计算充分性和跨状态一致性，不反演真实白质连接，也不声称间期事件诱导了人脑突触可塑性。

## 3. 固定不变的合同

LBSS 只改变 recurrent mask 的组成，以下全部继承 v0.4：

- 21 位物理坐标模型患者、31 fits；
- shared 患者单 fit，non-collinear 患者 own_a/own_b 两 fit；
- retrospective/test-informed propagation plane；
- `geometry_status=RETROSPECTIVE_TEST_INFORMED`；
- `edge_time_status=ORDINAL_NO_PHYSICAL_DELAY`：所有 recurrent edges 在一个 rank update 内生效，不估计传导速度或物理轴突延迟；
- tissue nodes、`Hᵀ/H` observation operator；
- leaky RNN cell、state dimension、input/readout/STOP head；
- chronological train/validation/test split；
- next-rank + STOP 自监督 loss；
- free-rollout size-head decoder，不读取真实下一 rank set size；
- supplied rank 1 从 primary rollout metric 两侧删除；
- empirical/model field builder、mirror/maxAB、patient-first statistics；
- early-ictal endpoint：clinical onset 后 0–10 s、1–150 Hz broadband energy；
- synchronized 5,000-draw all-contact shuffle primary null；
- canonical-full primary、seed-removed key secondary。

不新增 GRU、Transformer、GAT、Dale RNN、hidden-size sweep 或发作期训练支路。

## 4. 模型矩阵

| ID | 固定局部 backbone | 额外 K 条边 | 额外边是否任务选择 | 科学作用 |
|---|---|---|---|---|
| `L0_LOCAL_ONLY` | 相同 | 无 | — | 局部传播基线 |
| `L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL` | 相同 | 近局部 candidate pool | 是 | 控制相同新增容量与选择算法 |
| `L2_LOCAL_PLUS_RANDOM_LR` | 相同 | nonlocal candidate pool | 否，固定随机 | 任意 small-world shortcut 对照 |
| `L3_LOCAL_PLUS_LEARNED_LR` | 相同 | nonlocal candidate pool | 是 | 主要模型 |
| `C_L3_ORDER_SHUFFLED` | 相同 | nonlocal candidate pool | 是 | 真实 rank order 信息控制 |

现有 v0.4 `NO_RECURRENCE` 和 `DENSE` 只作外部边界参照；只有通过 checkpoint/config/hash 等价审计时才复用，否则不混入 primary matched contrasts。

### 4.1 公平比较约束

L1–L3 必须满足：

- 同一 local backbone 逐位相同；
- 新增 active edge count 均为 K；
- 相同 trainable weight 数量；
- 相同初始化 seed、optimizer、训练预算和 batch 顺序；
- 相同普通 weight regularization；
- 不对 nonlocal edges 额外施加 distance cost；
- L2 与 L3 每个 seed 使用同一初始 nonlocal active mask；L2 固定，L3 允许任务驱动重连。

这样：

\[
L3-L1
\]

检验 nonlocal 位置是否优于同等任务选择的局部容量；

\[
L3-L2
\]

检验任务选择的患者特异 nonlocal shortcut 是否优于相同初始 shortcut 的固定随机版本；

\[
L3-L0
\]

检验长程增量总体价值。

## 5. Local backbone

使用 **symmetrized kNN candidate mask + 独立有向权重**，而不是全图最短边：

\[
\{i,j\}\in G_{\mathrm{local}}
\iff
i\in k\mathrm{NN}(j)\ \lor\ j\in k\mathrm{NN}(i).
\]

构造要求：

1. 每个允许的无向 pair 同时加入 `i→j` 和 `j→i`；两个方向权重独立训练；
2. 每个 tissue node 至少 1 条入边和 1 条出边；
3. 无 self-loop；
4. 所有被 H 读取的 tissue nodes 位于同一个 strongly connected component；
5. `contact_supported_pairwise_reachability=1.0`；
6. topology 在训练中固定，mask 不预设传播方向；
7. k 使用 v0.4 `M3_FIXED_LOCAL` 的已冻结 edge-density matching 规则。

只有现有 M3 mask 满足上述 strong-connectivity 合同时才允许复用；weak connectivity 不足以构成本分支的 local backbone。

## 6. Extra-local 与 nonlocal candidate pools

设患者 local-backbone edge length 中位数为 `d_local_med`，冻结：

\[
r_{\mathrm{local}}=2d_{\mathrm{local\_med}}.
\]

### 6.1 Extra-local pool

\[
\mathcal E_{\mathrm{extra-local}}
=
\{(i,j)\notin G_{\mathrm{local}}: d_{ij}\le r_{\mathrm{local}}\}.
\]

### 6.2 Long-range pool

\[
\mathcal E_{\mathrm{LR}}
=
\{(i,j)\notin G_{\mathrm{local}}: d_{ij}>r_{\mathrm{local}}\}.
\]

两个 pool 都在训练前仅由几何和 local mask 冻结；不读取 A/B label 或 early-ictal target。

`LR` 在本合同中统一解释为 **relative-to-local-backbone nonlocal shortcut**，不是白质解剖长程连接。每位患者必须报告 local、extra-local 和 LR 的 edge-length 分布、pool size、重叠检查和毫米尺度；若两 pool 的物理距离高度重叠，只能称不同 candidate pools。

### 6.3 新增边数量

Primary 只使用一个固定剂量：

\[
K=\max\left(1,\operatorname{round}(0.10|E_{\mathrm{local}}|)\right).
\]

v0.2 不做 5%/20% sweep，避免再次形成 dose/architecture zoo。若某患者 candidate pool 小于 K，预先记录为 geometry-ineligible，不以重复采样补边。

运行前冻结：

- `candidate_pool_size`；
- `candidate_pool_size_per_source_node`；
- `unique_candidates_ever_activated`；
- `candidate_exposure_fraction`；
- source/target proposal frequency。

L1 与 L3 使用相同 rewiring interval、删除数、新增数和总 proposal 数。Regrowth 先均匀抽 eligible source，再从该 source 的 candidates 中均匀抽 target，避免候选多的中央节点自动变成 source hub。

## 7. 任务选择 nonlocal shortcuts

L3 只在 `E_LR` 内运行 SET：

1. local backbone 永不 prune；
2. nonlocal pool 始终恰好 K 条 active edges；
3. 按冻结周期删除 recurrent magnitude 最弱的 LR edges；
4. 从 inactive LR pool 补充相同数量；
5. 新边权重置零，不继承旧边；
6. 新生 edge 至少存活一个完整 rewiring interval，之后才允许再次 prune；
7. mask freeze 后只优化权重；
8. checkpoint、resume 与 snapshot 遵守 §7.1。

Leaky RNN edge strength：

\[
S_{ij}=|W_{ij}|.
\]

L1 使用完全相同的算法，但 candidate pool 换成 `E_extra-local`。

Order-shuffle 固定 rank 1，仅置换 **train/validation events** 中 rank 2…T 的完整 tie sets；保持事件长度、参与集合、split 和 patient-specific shuffle mapping。Held-out test events 始终保持真实顺序，使 true-order 与 shuffle 模型在完全相同的真实测试决策上评分。当 `T−1≥2` 时使用 derangement，保证后续 rank 均不留在原位；`T=2` 保持原样并单独计数。必须输出有效打乱事件比例、因长度 2 未改变比例、相对真实顺序的平均 Kendall distance，以及 held-out test ranks 前后逐位一致的 hash。

### 7.1 Checkpoint、snapshot 与 resume 合同

对 L1、L3 和 order-shuffle：

```text
eligible_best_checkpoint_epoch >= lr_mask_freeze_epoch
```

L0、L2 也只能从与 L3 相同的最早 structural-phase epoch 之后选择 checkpoint，避免不同训练预算。固定快照语义为：

```text
SNAPSHOT_INIT
SNAPSHOT_AFTER_WARMUP
SNAPSHOT_REWIRE_1_3
SNAPSHOT_REWIRE_2_3
SNAPSHOT_MASK_FREEZE
SNAPSHOT_FINAL
```

Resume 必须逐位恢复 optimizer state、RNG state、active added-edge mask、edge age、rewiring counter 和 freeze status。

## 8. 间期 primary endpoints

### 8.1 全体传播充分性

继续报告：

- heldout next-contact NLL；
- STOP BCE；
- seed-removed free-rollout Spearman；
- Kendall τb、normalized rank MAE、participation Jaccard（诊断）；
- event length ratio；
- empirical interictal field fidelity。

### 8.2 距离分层 primary

对 heldout transition，先定义真正的新招募 contacts：

\[
N_{e,t+1}=S_{e,t+1}\setminus\bigcup_{\tau\le t}S_{e,\tau}.
\]

Primary 前沿距离：

\[
d_{e,t}=\operatorname{median}_{c\in N_{e,t+1}}
\min_{j\in S_{e,t}}\|\mathbf r_c-\mathbf r_j\|_2.
\]

若 `N_{e,t+1}` 为空，该 transition 不进入 distance-stratified analysis。

阈值只用该患者 train transitions 冻结：

- local：0–50%；
- intermediate：50–80%；
- distal：80–100%。

每个患者、每个 bin 至少 20 个 heldout transitions 才进入患者级分箱推断；不足则标记 `DESCRIPTIVE_ONLY`，但不排除总体 propagation/field 分析。必须报告三个 bin 的事件数和实际毫米范围。

Primary contrast：

\[
\Delta L_{\mathrm{distal}}
=
L_{L0}-L_{L3}.
\]

Key secondary：

\[
L3-L1,
\qquad
L3-L2.
\]

所有距离分层比较使用完全相同的 heldout decisions、candidate masks 和患者集合。总体 NLL 仍报告，但不要求 L3 在所有 transition 上大幅胜出。

无需分箱的 secondary：先在每位患者内拟合 `model gain ~ transition distance` 斜率，再做患者级统计。同步保存 Human–RNN–SNN common observables：主轴推进 `Δs` 和横向扩散 `Δh`；它们是描述性共同量，不替代前沿距离 primary。

## 8.3 Functional-class detectability positive control

在 2–3 个真实患者几何上固定同一 local backbone，植入已知 K 条 nonlocal effective shortcuts 生成 rank events，再拟合 L0–L3 与 shuffle。成功标准不是 exact-edge recovery，而是：L3 选择性改善 distal transitions、L3 added-edge attenuation 选择性损害 distal transitions、true order 优于 shuffle。该结果进入最终报告，但不是决定是否运行真实数据的 gate。

## 9. Pathway 形成与可辨识性

对 L3 和 order-shuffle 在每个快照报告：

- LR edge survival probability；
- edge magnitude；
- contact-space effective influence；
- across-seed source/target density-grid similarity；
- source/target 沿 empirical interictal early/late field 的位置；
- distal-transition contribution。

Primary 承重对象依次为：contact-space effective influence、source/target endpoint density field、distal reach contribution、attenuation response、沿 empirical early/late field 的 endpoint enrichment。Exact edge overlap、edge survival、raw weight 和 binary mask similarity 仅为 secondary。

“Consensus pathway”固定定义为跨 seed 稳定的空间 endpoint/effective-influence pattern，不是完全相同的 edge set。

Candidate-proposal null 由实际 proposal/exposure logs 构建：在相同 source-first proposal 机会下，按每条 candidate 的暴露次数形成期望 source/target density 与 influence denominator。Claim C 比较的是超出这一机会分布的训练后 coarse pattern，并同时与 order-shuffle 比较；不能把候选多的节点自然更常出现解释成 task selection。

允许的模型内表述：

> Ordered interictal sequences selected and strengthened a small set of effective nonlocal shortcuts in the trained model.

不允许写：

> IEDs strengthened the corresponding human synapses.

## 10. 连续 attenuation 干预

每个模型只 attenuate 自己实际拥有的 active added edges：

\[
W_{\mathrm{target}}
\leftarrow
(1-\alpha)W_{\mathrm{target}},
\qquad
\alpha\in\{0.25,0.5,0.75,1.0\}.
\]

Target sets：

1. `A_L1`：L1 的 K 条 learned extra-local edges，在 L1 内 attenuate；
2. `A_L2`：L2 的 K 条 fixed random nonlocal edges，在 L2 内 attenuate；
3. `A_L3`：L3 的 K 条 task-selected nonlocal edges，在 L3 内 attenuate；
4. L3 内 K 条 local-backbone edges，匹配总绝对权重、endpoint in/out degree、空间覆盖和 tissue support；不匹配 edge length，因为 local/nonlocal 正是被检验因素。

评价：

- all-transition NLL；
- distal-transition NLL；
- rollout reach；
- empirical interictal field fidelity；
- frozen early-ictal concordance。

定义：

\[
\Delta^m_{p,b}(\alpha)=L^{m,\mathrm{atten}}_{p,b}(\alpha)-L^{m,\mathrm{intact}}_{p,b},
\qquad
S^m_p(\alpha)=\Delta^m_{p,\mathrm{distal}}-\Delta^m_{p,\mathrm{local}}.
\]

正式比较 `S^L3−S^L2` 与 `S^L3−S^L1`。L3 内的双重解离为：

\[
\mathrm{DD}_p(\alpha)=
[\Delta_{\mathrm{distal}}^{LR}-\Delta_{\mathrm{local}}^{LR}]
-[\Delta_{\mathrm{distal}}^{local}-\Delta_{\mathrm{local}}^{local}].
\]

Primary attenuation statistic 是四档 dose-response 的患者级 slope/AUC，不为每个 alpha 分别追逐 P 值。至少 200 个合法 matched draws 的要求只用于 L3 local-backbone subset controls；L1/L2/L3 各自的 K 条 added edges 是唯一预定义 target set，不抽 200 次。

理想双重解离：

- local backbone attenuation 主要破坏局部连续传播和总体 rollout；
- learned LR attenuation 选择性破坏 distal recruitment；
- matched random LR 损害较小；
- extra-local attenuation 不特异破坏 distal recruitment。

所有 attenuated rollouts、A/B/common/contrast fields 必须在 target unseal 前生成，写入 `ATTENUATED_FIELD_MANIFEST.json`，记录 checkpoint、edge-target、alpha、rollout keys、contact vectors、support、field-builder hash 和 `target_access_count=0`。Target unseal 后 scorer 只能读取冻结 vectors，不能重新运行模型或生成 field。

## 11. Early-ictal benchmark

### 11.1 数据边界

Target 已在项目历史中被读取，因此本分支不能称全新 blind validation。状态固定为：

```text
TARGET_KNOWN_TO_PROJECT_BUT_WITHHELD_FROM_LBSS_TRAINING_AND_SELECTION
```

所有 LBSS 模型、fields、pathway definitions 和 attenuation sets 先冻结，之后才运行 scorer。不得根据 early-ictal 结果调整 K、r_local、rewiring、checkpoint 或 representative patient。

### 11.2 Cohort

沿用 v0.4 physical-coordinate model cohort 与 target 的 exact primary intersection：10 人、24 seizures；E1146 为 supportive。缺失的 5 位 strict-target 患者不事后降低 8-contact model threshold。

### 11.3 Endpoints

Primary：`FIELD_CANONICAL_FULL`。

Key secondary：`FIELD_SEED_REMOVED`。

每个 endpoint 同时报告 A/B/common/contrast、all-contact null-relative margin 和 within-shaft sensitivity。

Primary model contrasts：

\[
E_{L3}-E_{L0},
\quad
E_{L3}-E_{L1},
\quad
E_{L3}-E_{L2}.
\]

同时按 v0.4 方法控制 interictal field fidelity，报告 patient-cluster bootstrap；不把 `P≈0.05` 写成显著。

Claim D 分两级：

- D1 cross-state correspondence：canonical-full field 相对 null；
- D2 shortcut-specific contribution：L3 在 seed-removed field 上超过 L0/L1/L2，或 attenuating `A_L3` 对 early-ictal concordance 产生单调负向 dose response。

只有 D2 支持时，才允许说 selective nonlocal shortcut 对跨状态对应有贡献。Panel E 同时加入 empirical interictal field reference。

## 12. 统计单位与 claim 分层

生物统计单位始终是患者：

1. event → fit/seed；
2. seed → fit；
3. own_a/own_b → patient（Q1 性能可合并，Q2 A/B fields 在 maxAB 前保持分开）；
4. patient-level paired tests/bootstraps。

不设置一个总 hard gate。每个 claim family 单独做 Holm 校正，不跨不同科学问题做一次全局校正。

### Claim A｜Local backbone sufficient

L0 能显著优于 no-recurrence、维持 seed-removed 自由 rollout，且 local mask 通过 strong-connectivity 合同。

### Claim B1｜Nonlocal increment

L3 对 distal endpoint 优于 L0。

### Claim B2｜Selective nonlocal benefit

L3 对 distal endpoint 同时优于 L1 与 L2；Claim B family 为 L3−L0/L1/L2 三个 paired contrasts。

### Claim C｜True order selects functional shortcut organization

True-order 与 shuffle 的 coarse endpoint/effective-influence pattern 不同、distal benefit 更高、attenuation 更具 distal specificity，并超出 candidate-proposal null；不要求 exact edge 一致。

Claim C family 固定为 coarse endpoint/effective-influence pattern、distal benefit 和 attenuation distal specificity 三项，并在该 family 内做 Holm 校正。

### Claim D1 / D2｜Cross-state correspondence / shortcut-specific contribution

按 §11.3 分别裁决 canonical-full correspondence 与 seed-removed/attenuation shortcut increment。

Claim D family 固定为 canonical-full、seed-removed 和 attenuation dose-response 三项，并在该 family 内做 Holm 校正。

Claim B/C 阳性而 D 阴性时，结论限定为“nonlocal shortcuts support distal interictal propagation”；不能升级为跨状态复用。

## 13. Figure 设计

| Panel | 内容 | 承重问题 |
|---|---|---|
| A | 真实患者几何上的固定 local backbone 与 K 条 nonlocal shortcuts | 模型合同 |
| B | 同患者 observed/generated local 与 distal transitions | 生成是否直观成立 |
| C | 全 cohort L0/L1/L2/L3 distal NLL 与 rollout reach | learned nonlocal edges 是否选择性改善远端传播 |
| D | L3 true-order vs shuffle 的 endpoint density/effective influence/distal-reach trajectory | 真实顺序是否选择 functional shortcut organization |
| E | 冻结 L0/L1/L2/L3 fields 与 early-ictal field、患者级统计 | 跨状态是否 motif-specific |
| F | local vs learned-nonlocal attenuation AUC 与双重解离 | 计算功能是否可干预 |

图内不堆长解释；沿用当前 paper-ready field renderer、TA 红/TB 蓝语义、较大轴标签和独立 legend。每个 figure 目录同步写中文 README。

## 14. 最终解释矩阵

| 结果 | 解释 |
|---|---|
| L0 已饱和 | 局部 recurrence 已足够 |
| L1≈L3 | 增益来自额外容量，不支持长程特异性 |
| L2≈L3 | 任意 shortcut 足够，不支持患者特异 pathway |
| L3 只改善 distal interictal | task-selected nonlocal shortcut 支撑远程间期传播，跨状态未确认 |
| L3 同时改善 distal 与 early-ictal | 支持双尺度 motif 的跨状态一致性 |
| L3>L2 且 targeted attenuation 特异 | 最强的模型内 pathway 功能证据 |

## 15. Definition of done

1. 5 个 arms × 31 fits × 3 seeds 的所有有效单元完成；不可恢复 invalid units 逐项说明；
2. 0 unresolved OOM、0 unresolved nonfinite；所有恢复 OOM/retries 显式记录；
3. local mask、candidate pools、K 和初始 LR mask 均有 hash；
4. 距离分层、rollout、field 与 attenuation 全部 patient-first；
5. 模型/field/pathway/attenuation/attenuated-field manifests 在 scorer 前冻结；
6. Figure A–F + source CSV/JSON + 中文 README；
7. 逐项写“可以说/不能说”；
8. 不改写或覆盖 v0.4 结果。
9. 所有产物 freshness、producer/config/input hashes、cohort revision 和 patient-level figure rows 可追溯；target unseal 后 source snapshot 不变，scorer 不生成新 field。
