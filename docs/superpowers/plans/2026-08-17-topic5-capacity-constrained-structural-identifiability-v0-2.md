# Topic 5.2D 容量约束的有序历史空间子空间与共享动力学可辨识性 v0.2：执行计划草案

> 对应 spec：`docs/superpowers/specs/2026-08-17-topic5-capacity-constrained-structural-identifiability-v0-2-design.md`  
> 状态：**AUTHORIZED FOR FORMAL EXECUTION（用户于 2026-08-17 正式授权；冻结科学合同不变，仅执行）**  
> 修订依据：2026-08-17 科学审阅；正式矩阵已改为 Core 1/Core 2 + capacity/learning/time 的分数设计。  
> 候选结果根：`results/topic5_capacity_constrained_history_motif_v0_2/`

## 0. 执行原则

1. 本计划复用现有 28 人 motif cache 与两位 ECoG cache，不重新定义事件。
2. 所有有序历史只能经过低维 state；`U_MINIMAL/U_FULL_SET` 两级无序 baseline 在 ordered model 训练前分别冻结。
3. 患者对齐和假结构同时约束 encoder 与 readout；direct-horizon 只作预测上界，autonomous family 必须共享同一个状态算子与 contact readout。
4. SEEG 与 ECoG 分开运行、分开报告、分开作图。
5. 科学阴性或 cohort 异质不停止后续预设分析。
6. 仅工程错误阻止对应单元；其余情况记录实际 denominator。
7. SEEG 的顺序扰动和 ordered-path ablation 是主要 use-phase 实验；basis swap 只称 transplant cost。ECoG graph swap 可单独称 runtime graph swap。
8. 当前 `fig6_interictal_crossstate_response_r5_candidate` 是 Figure 6 正式候选，必须保持不变；容量约束实验的新图默认进入 Supplementary / Extended Data，只有经用户另行审阅确认后才可能改动主图槽位。

---

## Phase A｜工作区保护与冻结 manifest

### A1. Worktree 范围

在任何修改前保存：

```text
WORKTREE_STATUS_BEFORE.txt
WORKTREE_SCOPE.json
PARENT_ARTIFACT_MANIFEST.json
```

只允许新增本阶段的 source、scripts、tests、docs 和结果根。不得整理、提交或覆盖现有 v0.1/v0.2 未提交修改。

### A2. Parent 数据对拍

逐患者核对：

- contact names/order；
- event IDs；
- ranks 与 participation；
- split 0/1/2/-1；
- geometry/shaft；
- model-unseen 与 parent held-out 完全一致；
- 28 位患者 census。

输出：

```text
INPUT_CENSUS.csv
SPLIT_HASH_AUDIT.json
CONTACT_GEOMETRY_AUDIT.csv
```

### A3. ECoG 独立 manifest

逐位复用 E958/E1084：

- cache hash；
- 64/63 contact order；
- block split；
- four-neighbour true/permuted/rewired graph；
- E1084 `GC1` 缺失处理；
- microsteps 合同。

输出到独立子根：

```text
ecog_construct_validity/INPUT_MANIFEST.json
```

---

## Phase B｜无序 baseline 与 target builder

### B1. 新模块

建议新增：

```text
src/topic5_strict_history_motif_v0_2.py
src/topic5_strict_history_data_v0_2.py
src/topic5_structural_identifiability_v0_2.py
src/topic5_ecog_graph_capacity_v0_2.py
```

复用现有 event/cache loader、checkpoint hashing、patient aggregation 和 exact subset mask；不复制整套 parent pipeline。

### B2. Prefix/horizon 数据构建

每个样本保存：

```text
event_id
prefix_len
start_set
cumulative_unordered_set
ordered_rank_sets[0:prefix_len]
target_rank_sets[h=1..5]
target_available_mask[h]
target_cardinality[h]
suffix5_participation_field
full_suffix_participation_field
late_field_centroid
spectral_centroid_latency_proxy[h]
split
recording_block
```

目标构建一次后写不可变 cache。不得在每个 worker 中重新解释 event suffix。

正式 horizon cache 改为：

```text
h=1,2,3 primary family
h=4,5 long-event sensitivity
suffix all events with a future suffix
```

逐 horizon 保存实际 event denominator；target builder 同时生成 exact-subset normalization 所需的 available-contact mask。

### B3. 两级无序 baseline

分别训练：

```text
U_MINIMAL  = start set + prefix length + recruited fraction + contact intercept
U_FULL_SET = U_MINIMAL + cumulative unordered contact set
```

每一级都：

1. 只用 split 0 训练；
2. split 1 选 baseline rank 与 epoch；
3. 冻结并写 per-patient hash；
4. 用 1,000 个随机 prefix-order permutations 验证输出逐位不变；
5. 用故意暴露 current rank 的 bug-injection test 证明审计能失败；
6. 预计算 cardinality logits、contact logits 与 suffix baseline，chunk 保存；
7. 后续模型只读对应 baseline，禁止共同微调。

输出：

```text
baseline/U_MINIMAL/<patient>/checkpoint.pt
baseline/U_FULL_SET/<patient>/checkpoint.pt
baseline/<level>/<patient>/logits_split{0,1,2,-1}.npz
baseline/UNORDERED_INVARIANCE_AUDIT.json
```

### B4. STOP 分离验证

训练脚本中空间 checkpoint selector 只接受 `L_space`。`L_space` 使用 cardinality NLL + exact subset NLL；suffix 使用 event-balanced Brier/BCE。加入回归测试：把 STOP loss 乘以极大常数时，选中的 spatial checkpoint 不变。

---

## Phase C｜结构 basis 与 null family

### C0. Basis ceiling 与 horizon census

在训练 ordered models 前，先对 `U_MINIMAL/U_FULL_SET` 两套 split-2 residual fields 分别计算冻结 basis 的最佳投影误差：

```text
GEOMETRY_LAYOUT
SHAFT_GRADIENT
PATIENT_ALIGNED
ANGLE_ROTATED_AXIS
IDENTITY_PERMUTED
TRAIN_ONLY_FREE_PCA
```

同时输出 principal angles、逐 horizon/suffix denominator 和 residual effective rank。该分析只回答“表示上是否可能”，不选择是否训练任何模型。

输出：

```text
basis/BASIS_CEILING_PER_PATIENT.csv
basis/BASIS_PRINCIPAL_ANGLES.csv
basis/HORIZON_DENOMINATOR_CENSUS.csv
```

### C1. Geometry、shaft 与患者对齐 basis

实现 spec §4.5：

- split-0 train-only unsigned axis；
- distance/shaft-aware local kernel；
- \(K_0,K_+,K_-\) 与两步字典；
- 投影常数/shaft scaffold；
- SVD basis \(Q_{r}\)，`r=1/2/4/8`；
- deterministic sign/order；
- 一维患者单独标签。

同时建立：

- `H1_GEOMETRY_LAYOUT`：只使用 contact/shaft geometry 与 \([K_0,K_0^2]\)；
- `H1_SHAFT_GRADIENT`：只使用 shaft 内线性坐标、shaft identity 与相邻关系；
- `H1_PATIENT_ALIGNED`：使用允许的 train-only late-field axis。

学习曲线保存两套 basis manifest：

```text
END_TO_END_BASIS_CURVE      Q_25 / Q_50 / Q_100
BASIS_PRETRAINED_CURVE      fixed Q_100 + 25/50/100% dynamics events
```

两者都不得读取 split 1/2/-1 构造 basis。

每个 basis 保存输入 hash、奇异值、orthogonality error、shaft projection、2D eligibility 和 source code hash。

### C2. Null family：主 null 与 sensitivity 分层

在看任何模型结果前，每位患者生成：

```text
8 × ANGLE_ROTATED_AXIS
4 × IDENTITY_PERMUTED
4 × LOCALITY_REWIRED（只用于 r=4 / 100% / U_FULL_SET）
```

Angle-rotated 只对冻结二维 geometry 合格患者生成；近一维患者记录 `ANGLE_NULL_INELIGIBLE`，不以 identity/rewired 替代 angle primary。

每个 null 写：

- rank、orthogonality；
- singular values；
- axis rotation angle 或 degree；
- edge-length histogram；
- within/cross-shaft；
- connectedness；
- 与 aligned 的 contact identity overlap；
- 所有未精确匹配项。

不得根据训练结果删除 null。angle rotations 在结果前按几何可行角固定；identity/rewired 选择规则不能看 loss。容量和 learning curves 使用 spec/plan 中预设的小型 angle-null subset，不在每个组合重复完整 null family。

### C3. 反重参数化测试

测试至少包括：

1. aligned/angle/permuted/rewired 同 rank、同参数量；
2. encoder/readout 都经过同一 \(Q\)；
3. 替换任一处为自由 contact matrix 时测试明确失败；
4. 完整 \(F\) 和 readout 作正交协变变换后 logits atol ≤ `1e-6`；
5. `Q^TQ≈I`；
6. `r=1/2/4/8` 的截断严格嵌套；
7. angle-rotated basis 保持 geometry、kernel、rank 与 anisotropy strength；
8. shaft basis 不读取任何事件结果。

输出：

```text
basis/STRUCTURE_BASIS_MANIFEST.csv
basis/NULL_MATCH_AUDIT.json
basis/REPARAMETERIZATION_AUDIT.json
```

---

## Phase D｜模型实现与测试

### D1. 正式模型

实现：

```text
H0_UNORDERED_ONLY
H1_GEOMETRY_LAYOUT
H1_SHAFT_GRADIENT
H1_PATIENT_ALIGNED
H1_ALIGNED_ORDERLESS_BAG
H1_ANGLE_ROTATED_AXIS
H1_IDENTITY_PERMUTED
H1_LOCALITY_REWIRED
H1_FREE_LOW_RANK
```

除 `H0` 与 `ALIGNED_ORDERLESS_BAG` 外，每个 ordered model 分别实现：

```text
DIRECT_HORIZON_UPPER_BOUND
AUTONOMOUS_SHARED_OPERATOR
```

Direct family 允许 horizon-specific low-dimensional readouts；autonomous family 必须共享一个完整 `r×r F` 和一个 contact readout。完整 `F` 为 primary；diagonal、bandwidth-1、stable-normal 与 low-dimensional tanh 只作 sensitivity。

所有 ordered models 共享：

- frozen baseline；
- prefix/horizon targets；
- loss weights；
- optimizer family；
- batch schedule；
- early stopping；
- seed registry；
- contact availability mask。

### D2. Unit tests

至少覆盖：

1. unordered baseline 的 rank-order invariance；
2. free/ordered state 对顺序改变敏感，orderless bag 不敏感；
3. future rank 不进入 prefix state；
4. horizon 不存在时正确 mask 且 denominator 独立；
5. cardinality NLL 与 exact subset NLL 通过枚举小系统对拍；
6. recruited contacts 不进入候选输出；
7. autonomous family 不含 horizon-specific contact/cardinality readout 或独立 suffix primary；
8. autonomous suffix 等于逐 horizon probability 的 no-repeat 累积；
9. aligned/null 的 trainable parameter count 匹配；
10. free model 参数量单独报告；
11. prefix-order perturbation 保持 start/set/length/cardinality 并使 baseline 逐位不变；
12. ordered-path ablation 只删除 ordered residual；
13. basis transplant/graph swap 不调用 optimizer；
14. swap 前后 checkpoint hash 不变；
15. full-F covariant rotation exact equivalence；
16. STOP 不进入 spatial checkpoint；
17. end-to-end 与 fixed-basis learning-curve event/basis contracts；
18. nested 25/50/100% event IDs；
19. split -1 只可由 compact confirmation scorer 读取；
20. patient-first aggregation；
21. one-dimensional patients 不进入 2D claim denominator。

### D3. Smoke run

只验证运行链，不看科学方向。固定 4 位不同规模患者：小 contact、中 contact、大 contact、近一维各一位。每位跑：

```text
r=2
U_MINIMAL + U_FULL_SET
aligned + one angle-rotated + orderless-bag + free
direct + autonomous
1 epoch
h=1..5 + suffix
one prefix-order perturbation
one ordered-path ablation
one basis transplant
```

输出 shape、内存、速度、resume 和 hash。科学结果无论正负都进入下一阶段。

---

## Phase E｜Synthetic identifiability 并行实验

### E0. Canonical correctness

固定少量同型 cells：

```text
effect = 0
aligned direct teacher
aligned autonomous teacher with known F
identity-permuted teacher
low/high bypass
```

验证 false-positive、结构排序、已知 `F`、orderless-vs-ordered、prefix-order cost 和 z-ablation direction。E0 只验证实现，不作为真实数据科学 gate。

### E1. Empirical-power surface

选择 6 类代表 montage，不做全笛卡尔积：

```text
small / medium / large contacts
few / many shafts
near-1D / 2D
source-near / source-far
effect = 0 / medium / strong
bypass = low / high
noise = 2–3 levels
direct / autonomous
```

输出 false-positive、`P(aligned > angle-null)`、orderless-vs-ordered recovery 与 autonomous recovery。

### E2. Misspecification LHS

Latin-hypercube 扫描 unobserved nodes、extra state、direction jitter、bypass、noise 与 source distance。28 个真实 montage masks 只在一组冻结 canonical teachers 上全部运行，生成 patient-specific detectability descriptors；不与所有参数做笛卡尔积。

另行比较 shaft-like、random 和 source-avoiding masks。所有 synthetic 结果只校准解释范围，不决定真实患者纳入或后续实验是否执行。

---

## Phase F｜SEEG 正式运行矩阵

### F1. 计算预算

为避免把预算消耗在低 rank、低 data 的第 8 张 rewired null 上，正式矩阵采用分数设计。Direct 与 autonomous 只在中心比较中完整并列；外围 capacity/learning/time 以 autonomous 为主，direct 仅保留预测上界所需的紧凑组合。

#### Core 1｜中心结构比较

全 28 位患者，`r=4`、100% data、`U_FULL_SET`：

```text
geometry layout                 3 seeds
patient aligned                3 seeds
free low-rank                  3 seeds
shaft gradient                 1 paired seed
aligned orderless bag          1 paired seed
angle-rotated                  8 nulls × 1 paired seed
identity-permuted              4 nulls × 1 paired seed
locality-rewired               4 nulls × 1 paired seed
```

完整 null family 只集中在 Core 1。

Core 1 中除 `aligned orderless bag` 只运行一次 orderless control 外，其余结构均运行 direct 与 autonomous 两个 family。

#### Core 2｜无序旁路交互

全 28 位患者，`r=4`、100% data、`U_MINIMAL`：

```text
geometry / shaft / aligned / free / aligned-orderless-bag
4 angle-rotated nulls
one paired seed
aligned and free add a second seed
```

与 Core 1 共同计算 `I_{bypass}`。不重复 identity/rewired full family。

Core 2 的完整结构集合运行 autonomous；direct 只运行 aligned、free 和 patient-median angle-null 作为 bypass 条件下的预测上界。orderless bag 运行一次。

#### Capacity curve

`r=1/2/8`、100% data、`U_FULL_SET`：

```text
geometry / shaft / aligned / free
2 angle-rotated nulls
one paired seed
```

`r=4` 复用 Core 1。

Capacity curve 以 autonomous 为正式曲线，不为每个外围 rank 重复 direct family；direct 的 rank-4 ceiling 复用 Core 1。

#### Learning curves

`r=4`、25/50% data、`U_FULL_SET`：

```text
geometry / shaft / aligned / free
4 angle-rotated nulls
one paired seed
END_TO_END_BASIS_CURVE
BASIS_PRETRAINED_CURVE
```

100% 复用 Core 1。locality-rewired 只保留在 100%。

两条 learning curves 都只运行 autonomous family；direct learning curve 不重复，因为它不回答共享算子的样本效率。

#### Time proxy

`r=4`、100% data、`U_FULL_SET`、split 2：

```text
geometry / aligned / patient-median angle-rotated / free
one paired seed
```

time head 预测 `h=1/2/3` 的累计 spectral-centroid latency proxy；空间和时间分开评分，不解释为传导速度。time proxy 不访问 split -1。

Time proxy 只挂在 autonomous family；direct time 上界留作结果明确后的非正式探索，不进入冻结矩阵。

`H0_UNORDERED_ONLY` 的两个 baseline 各训练一次，不按 rank 重训。正式 unit 数由 manifest 按 direct/autonomous、basis curve 和实际 null eligibility 展开并冻结；不得在运行中因初步结果增删组合。

按所有 28 位患者均满足全部 null 资格估算，SEEG 上限为：

```text
Core 1       53 ordered units / patient
Core 2       14 ordered units / patient
Capacity     18 ordered units / patient
Learning     32 ordered units / patient
Time          4 ordered units / patient
---------------------------------------
Total       121 × 28 = 3388 ordered units
Baselines     2 × 28 =   56 units
SEEG total upper bound  = 3444 training units
```

该数不含低成本 scoring/ablation、ECoG 与 synthetic；正式启动值由 manifest 重新计算。任何差异必须来自预注册 eligibility，而不能来自结果方向。

### F2. 运行顺序

1. G0：先完成 basis ceiling、horizon census 与两级 baseline；
2. G1/G2：全患者跑 free/aligned 的 direct 与 autonomous，验证任务和模型 family；
3. G3：完成 Core 1 的 geometry/shaft/orderless/angle/identity/rewired；
4. G4：完成 `U_MINIMAL` Core 2；
5. G5：完成 capacity 与两条 learning curves；
6. G6：对冻结 checkpoint 运行 prefix-order、ordered-path ablation 和 basis transplant；
7. G7：汇总 horizon/suffix/endpoint/time，STOP 在 spatial freeze 后单独拟合；
8. G8/G9：完成 compact synthetic 与 stochastic rollout；
9. 所有选择与 split-2 分析冻结后，只解封 compact split -1 scorer；
10. rollout 只对每患者 validation-median seed 与 ensemble 运行。

该顺序只为资源和故障隔离，不根据科学结果停止后续 family。

### F3. 稳定运行与 OOM 规则

- 小模型默认 CPU 多 worker；GPU 只用于 profile 后确认能提高吞吐的 batch；
- 每个 worker 单次只加载一个患者、一个 rank、一个 structure；
- event cache 使用 memory-map/chunk，不把所有 null 和 horizon 展开到内存；
- GPU worker 数按实测峰值显存自动上限，预留至少 20% 显存；
- 捕获 OOM 后只允许降低 batch/chunk 并从同一 checkpoint 恢复，不改变模型或事件；
- 每个 unit 原子写 `status.json`、`metrics.json`、checkpoint 和 hash；
- 长任务在 `tmux`/`nohup` 中运行，watcher 只读 manifest，不依赖桌面会话或网络连接；
- 失败单元最多自动重试 2 次；第三次记录为 unresolved，不静默跳过；
- worker 不写共享汇总文件，汇总由单独 reducer 原子生成。

### F4. Use-phase 与 transplant

#### SEEG primary

对 Core 1/Core 2 冻结 checkpoint 运行：

```text
PREFIX_ORDER_COST
ORDERED_PATH_ABLATION_COST
```

prefix-order 操作必须保持 start、cumulative set、prefix length 与 cardinality；ordered-path ablation 只能令 `z=0`/ordered residual 为零。两项均逐 horizon、suffix、endpoint 和 STOP 保存。

#### SEEG basis transplant sensitivity

只对 Core 1 aligned–patient-median angle-null 配对生成 `AA/AN/NA/NN`，完整保存：

```text
delta_test_given_A
delta_test_given_N
delta_train_given_A
delta_train_given_N
transplant_interaction
immediate_logit_norm_change
future_field_change
STOP_change
```

统一命名 `basis_transplant_cost`，不写 `delta_use`。split -1 不运行完整 transplant family。

`immediate_logit_norm_change` 只作诊断/协变量，不用超级匹配删除患者。

### F5. Split -1 compact confirmation

只在所有选择冻结后评分：

```text
r=4 / 100% data / U_FULL_SET
AUTONOMOUS_SHARED_OPERATOR
patient aligned vs patient-median angle-rotated null
free low-rank vs U_FULL_SET
DIRECT_HORIZON_UPPER_BOUND as predictive ceiling
```

angle comparison 只使用冻结二维合格 denominator；其他患者仍完成 free-vs-baseline confirmation。其他组合的 split -1 access 必须触发审计失败。

---

## Phase G｜ECoG 构造效度 case series

### G1. 复用而非覆盖

保留现有 ECoG v0.1 全部训练和结论。新结果根：

```text
results/topic5_capacity_constrained_history_motif_v0_2/ecog_construct_validity/
```

### G2. 新训练矩阵

对 E958/E1084 分别运行：

```text
capacity = G1, G2, G3
structure = observed grid, identity-permuted, degree+distance rewired
train fraction = 25,50,100%
baseline = U_MINIMAL and U_FULL_SET at G2/100% core; U_FULL_SET for remaining curves
model family = direct, autonomous
```

observed grid 3 seeds；null family 使用现有 31 张图中的预冻结 8 张 × 1 paired seed，完整 31 张用于低成本 test-swap/scoring sensitivity。既有逐边自由 G4 结果作为高容量参照，不为本阶段重跑 31×3 大矩阵，除非实现审计证明任务不可比。

### G3. 任务统一

- 使用与 SEEG 相同的两级 unordered baseline；
- primary 预测 horizon 1–3，horizon 4–5 为 long-event sensitivity；
- direct 与 autonomous 分开；
- STOP 单独拟合；
- microsteps=2 primary，microsteps=1 sensitivity；
- runtime graph swap 不重校准，节点 identity 和 contact readout 保持不变。

### G4. 报告

E958 与 E1084 各自画完整 capacity/data/use curves。不得计算二患者 pooled p，不得写 replicated unless both pre-specified directions and effects agree, and even then only “two-case consistency”。

---

## Phase H｜患者级汇总与 evidence matrix

### H1. 固定聚合顺序

```text
decision/horizon
→ event
→ seed/null basis
→ patient
→ cohort
```

每位患者的 aligned effect 与该患者 null median 比，不把 null graph 当患者。

### H2. 输出表

```text
PER_PATIENT_BASIS_CEILING.csv
PER_PATIENT_DIRECT_VS_AUTONOMOUS.csv
PER_PATIENT_CAPACITY_CURVE.csv
PER_PATIENT_END_TO_END_DATA_CURVE.csv
PER_PATIENT_FIXED_BASIS_DATA_CURVE.csv
PER_PATIENT_BYPASS_INTERACTION.csv
PER_PATIENT_ORDER_AND_PATH_ABLATION.csv
PER_PATIENT_BASIS_TRANSPLANT.csv
PER_PATIENT_ENDPOINT_MATRIX.csv
PER_PATIENT_COVERAGE_DESCRIPTORS.csv
COHORT_EVIDENCE_MATRIX.json
ECOG_CASE_SERIES_MATRIX.json
SYNTHETIC_IDENTIFIABILITY_SURFACE.npz
```

### H3. 结果解释顺序

每位患者与 cohort 都按以下顺序读：

1. candidate basis 在 representation ceiling 上是否可能；
2. free ordered state 是否优于对应 unordered baseline，并对 prefix order/z ablation 敏感；
3. direct 与 autonomous 哪一类学得会；
4. aligned 是否优于 geometry、shaft、angle-rotated、identity 与 rewired；
5. aligned ordered 是否超过 aligned orderless bag；
6. 结构优势是否在 `U_MINIMAL` 下增强；
7. 优势是否随 rank、end-to-end data 或 fixed-basis data 变化；
8. basis transplant cost 与 ECoG graph swap 分别说明什么；
9. 改善发生在哪个 horizon、suffix、endpoint、time 或 STOP；
10. synthetic 在该覆盖条件下是否有辨识力。

不先看 p 值决定故事。

---

## Phase I｜Paper-ready 图与报告

### I1. 当前 Figure 6 正式候选

不改动现有：

```text
results/paper-ready-figure/fig6_interictal_crossstate_response_r5_candidate/
```

该资产已由用户于 2026-08-17 确认为当前 Figure 6 正式候选。后续实验不得覆盖其 PNG/PDF/SVG、source data、README 或元数据。`fig6_dynamical_motif_rnn_v0_2` 保留为历史诊断资产，不占当前正式候选身份。

### I2. 新 Supplementary / Extended Data figure candidate

候选路径：

```text
results/paper-ready-figure/supp_fig6_strict_history_motif_v0_2/figures/
```

按 spec §11 生成 A–F。该图默认是补充图候选，不自动替换当前 Figure 6。Panel A 必须先单独导出草图给用户审阅，再拼整图；不得先做完整画布后靠加解释文字修科学语义。

图形执行顺序：

1. 先画 A：observed ranks → low-dimensional state；上方 direct readout，下方 autonomous shared operator；灰色支路显示 minimal/full-set baseline；
2. 冻结 geometry、shaft、patient-trained、direction-rotated、free 的读者名称、颜色和图例；
3. 画患者例子 B：真实 suffix、aligned direct、aligned autonomous、angle-null autonomous、free autonomous；
4. 画 C：basis ceiling；D：direct vs autonomous；E：bypass interaction；F：order/path use 与容量/样本效率；
5. render PNG，逐 panel 目视；
6. 修布局后从同一状态生成 PDF/SVG；
7. PDF 首页栅格化与 PNG 对拍；
8. 检查最小字号、裁切、legend、轴含义、患者点和不确定性；
9. 写中文 `README.md` 和 figure caption；
10. 生成 source-data 与 metadata hashes。

### I3. 报告双版本

最终必须同时生成：

```text
docs/archive/topic5/capacity_constrained_history_motif_v0_2_plain_report_<date>.md
docs/archive/topic5/capacity_constrained_history_motif_v0_2_technical_closeout_<date>.md
```

白话版按七个问题写：

1. 我们限制了什么；
2. 低维模型是否学得会；
3. 训练后的模型是否真的使用 rank 顺序和 ordered path；
4. 未来只是可直接解码，还是能由共享算子自主推进；
5. 患者对齐结构是否比 geometry、shaft 和错位方向省容量/省数据；
6. 无序 contact-set 旁路减弱后，结构优势是否增强；
7. 部分电极覆盖使结果能解释到哪里。

技术版记录完整方程、split、unit 分母、null matching、swap、synthetic、患者效应、允许/禁止措辞和工程验收。

---

## Phase J｜Closeout 审计

### J1. 工程审计

- 全部 unit census；
- failed/nonfinite/OOM/retry；
- checkpoint、basis、baseline、event 和 code hashes；
- split -1 access log；
- prefix-order、ordered-path ablation、basis transplant 与 ECoG graph swap 的参数不变；
- denominator 与 exclusion reasons；
- tests；
- resume/replay；
- background workers 全退出。

### J2. 科学合同审计

- 两级 baseline 是否真正无序且旁路定义不同；
- direct/autonomous 是否严格分离，autonomous 是否共享 \(F\) 与 readout；
- encoder/readout 是否共享结构 basis；
- aligned-orderless bag 是否确实不读取 rank 顺序；
- cardinality + exact subset law 是否与 decoder 合同一致；
- null 是否按实际匹配项报告；
- STOP 是否与空间选择分离；
- SEEG/ECoG 是否分母分离；
- coverage 是否只作描述；
- synthetic 是否被当解释相图而非 gate；
- claim 是否停在 observed montage/effective computation。

### J3. 图形审计

- PNG/PDF/SVG 同状态；
- source-data 可重画；
- README/caption 与 panel 一致；
- Panel A 无内部术语和防御性长句；
- C–F 含 patient points 和 uncertainty；
- ECoG 不混入 SEEG cohort 统计；
- 当前 `fig6_interictal_crossstate_response_r5_candidate` 未被覆盖、改名或重写。

输出：

```text
CLOSEOUT_AUDIT.json
SCIENTIFIC_CONTRACT_AUDIT.json
FIGURE_VISUAL_QA.json
FINAL_EVIDENCE_MATRIX.json
```

---

## 计划完成条件

本阶段完成不要求任何结构阳性。完成定义为：

1. 预设 SEEG 与 ECoG 实验按实际可用分母完成；
2. 所有失败、低覆盖和 null matching 缺口透明记录；
3. representation ceiling、ordered use、direct/autonomous、structure、bypass、capacity/data、use-phase 和 synthetic 证据齐全；
4. 白话/技术报告与 paper-ready candidate 同源；
5. 允许与禁止结论经过审计；
6. 用户完成科学与视觉终审。
