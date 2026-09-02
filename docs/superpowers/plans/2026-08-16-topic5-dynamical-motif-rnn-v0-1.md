# Topic 5.2 动力学 motif RNN v0.1-r2 执行计划

> 对应 spec：`docs/superpowers/specs/2026-08-16-topic5-dynamical-motif-rnn-v0-1-design.md`
> 状态：**AUTHORIZED AND EXECUTING（用户 2026-08-16 授权）**
> 结果根：`results/topic5_dynamical_motif_rnn_v0_1/`

## ERRATUM 2026-08-16

1. **§B6 / §唯一停止条件的第 6 条作废**：用户明确覆盖 "Phase B 结束后等待用户审阅
   `DESIGN_FREEZE_AMENDMENT`" 的暂停要求。Agent 完成 Phase B 设计实验后自行写入
   `DESIGN_FREEZE_AMENDMENT.json` 并继续全队列。
2. **`CORE_FRAME` 预先确定为 `GEOMETRY_ONLY_PCA2`**（28 个患者级 fit），
   `PARENT_FROZEN_FRAME` 为 6–8 位患者的并行敏感性。选择规则**不看显著性**：
   即使两个 frame 效应不同，也继续跑 geometry-only 全队列，并把 frame dependence
   本身作为 G0 结果。理由见 spec ERRATUM P0-1（parent frame 的 x 轴就是传播轴）。
3. **`PRIMARY_M2_GATE = M2-2RANK`**；`M2-3RANK` / `M2-ONLINE` 做设计实验 + 全队列 gate replay。
4. 设计与实现修订 11 条见 spec ERRATUM 与
   `results/topic5_dynamical_motif_rnn_v0_1/SCIENTIFIC_DESIGN_AUDIT.md`。
5. `self-feeding` 只做设计实验，不自动扩展全队列（与原计划一致）。

## 0｜执行原则

本计划把旧连接结构比较移入 Supplementary，正式实验研究：

```text
局部扩散
→ 超越植入布局的传播走廊
→ 早期状态控制的局部方向偏置
→ 沿轴前馈的有限时程扩展
```

计划不再用一串科学结果作为停止条件。实现正确后，G0–G6均运行并分别报告。只有输入错配、label leakage、嵌套等价失败、closed-loop错误、NaN/Inf或checkpoint损坏阻止相应工程阶段。

正式训练前增加一个6–8位患者的设计实验，把 frame、M2算子、训练方式和方向门控的影响拆开。该实验不挑“最显著”的实现；它用来说明后续结果对实现选择是否稳健。

## Phase A｜最小完整性审计

### A1. Parent 与 worktree provenance

1. 记录当前 worktree dirty state，保护用户已有修改；
2. 冻结 parent code/results路径、commit、SHA256和42-fit census；
3. 核对14位 `own_a/own_b` 双视图患者与14位 `shared` 患者；
4. 逐位证明双视图使用相同 event IDs和contact set；
5. 明确旧 L0/L1/L2m/L3/C-suffix checkpoints只读，不直接初始化新motif模型。

输出：

```text
PARENT_INPUT_MANIFEST.json
PARENT_VIEW_CENSUS.csv
WORKTREE_SCOPE_RECORD.json
```

### A2. 两种 frame 的可执行性

#### Parent frame

- 逐位复用原 tissue nodes、`H`、plane和local support；
- 保存 `own_a/own_b/shared` view identity；
- patient-level折叠规则固定。

#### Geometry-only frame

- 仅从contact三维坐标建立PCA2；
- component顺序/sign按spec固定；
- 合并双视图并重建patient-level mesh、`H`、距离和local support；
- 对两个近一维患者建立line-mesh；
- 比较node count、`H` rank、zero-H nodes和局部覆盖。

输出：

```text
FRAME_FEASIBILITY.json
GEOMETRY_ONLY_FIT_CENSUS.csv
frame_cache/<frame>/<fit>/provenance.json
```

两种frame行为不同不是工程失败，记录为G0结果。

### A3. Split 与 label leakage

1. 从原始event manifest逐位核对split 0/1/2/-1；
2. 证明split -1对应parent held-out events；
3. 冻结每层event IDs和hashes；
4. train-only重建TA/TB centroids；
5. 对model forward/loss、direction gate、checkpoint selection做dependency audit；
6. seizure target access在interictal basis冻结前必须为0。

输出：

```text
SPLIT_PROVENANCE_AUDIT.json
MODEL_UNSEEN_EVENT_MANIFEST.csv
LABEL_DEPENDENCY_AUDIT.json
```

### A4. Decoder 与复现合同

实现并测试：

- exact fixed-cardinality subset sampling；
- sampled size不读取未来集合大小；
- repeat mask；
- STOP precedence和absorbing STOP；
- full `q=(h,r,k,s)` replay；
- common random numbers；
- checkpoint/optimizer/RNG/resume一致；
- deterministic replay和stochastic distribution replay。

输出：

```text
DECODER_CONTRACT.json
REPLAY_AUDIT.json
```

### A5. 并行 sidecar QC

以下QC与模型实现并行，不阻止interictal主实验：

- `event_lag_raw`覆盖、分辨率和distance–lag关系；
- static seizure 0–10 s重提可行性；
- same-block pseudo-onset数量；
- dynamic BB150可靠性；
- exact contact join、Nyquist、坏段和gap。

输出：

```text
TIMING_SIDECAR_FEASIBILITY.json
SEIZURE_REUSE_METADATA_FEASIBILITY.json
```

## Phase B｜6–8位患者的设计实验

### B0. 实现骨架与回归测试

新增模块：

```text
src/topic5_dynamical_motif_rnn_v0_1.py
src/topic5_dynamical_motif_rollout_v0_1.py
src/topic5_dynamical_motif_analysis_v0_1.py
```

新增脚本至少包括cache准备、单元训练、随机rollout、counterfactual、seizure增量评分、汇总和审计。正式命名前先检查仓库现有命名，避免覆盖旧`M0...M8`代码ID。

回归测试至少覆盖：

1. 两种frame的event/contact provenance；
2. M0/M1 kernel、M2 local support和非负权重；
3. `eta=0/beta=0/gamma=0`逐位等价；
4. `s_1=0`且所有gate只读取当前prefix；
5. M3严格三角、symmetric和axis-shuffled contracts；
6. gain/memory参数方向正确；
7. exact subset sampler、STOP、repeat mask和full-state replay；
8. teacher forcing、gate replay和component-isolation replay不读未来；
9. common RNG和checkpoint/resume；
10. TA/TB/seizure dependency为0。

实现错误阻止正式训练；toy弱信号恢复率不阻止。

### B1. Sentinel patients

在看新结果前按dataset、contact数、frame类型和几何维度固定：

```text
epilepsiae_1077       小contact、双parent views
epilepsiae_1146       中等contact、shared view
epilepsiae_253        小contact、双parent views
epilepsiae_139        近一维
yuquan_songzishuo     大contact、双parent views
yuquan_zhangbichen    最大contact、双parent views
yuquan_zhangjiaqi     近一维
yuquan_zhaochenxi     大contact、shared view
```

若某位因输入损坏不可运行，只能按同一预冻结分层规则替换，不能按模型结果替换。

### B2. Frame experiment

在相同events和common RNG下运行：

```text
PARENT_FROZEN_FRAME × M0/M1/M2
GEOMETRY_ONLY_PCA2 × M0/M1/M2
```

每种一个固定seed，双parent views先在患者内折叠。比较：

- M0 prediction/calibration；
- M1−M0、M2−M1 paired effect；
- free axis vs contact-cloud 3D PCA1与dominant-shaft axes；
- seed/block stability的可估性；
- mesh/`H`变化。

输出：

```text
FRAME_EXPERIMENT_PER_PATIENT.csv
FRAME_EXPERIMENT_SUMMARY.json
```

Phase B结束后由用户审阅并冻结 `CORE_FRAME`。当前336+84资源公式只覆盖28-fit patient-level frame；若选择42 parent views全队列，必须另写预算修订，不能自动进入Phase C。

### B3. M2 operator 与方向承诺时间

在两种frame上比较：

```text
LOCAL_POSITIVE_DIRECTIONAL_KERNEL
ORTHOGONAL_ROTATION_SENSITIVITY
```

局部主算子平行运行：

```text
M2-2RANK
M2-3RANK
M2-ONLINE
```

输出：

- prefix length 2/3/4/5的future-field和endpoint prediction；
- `s_k` stability和方向翻转率；
- one-step hidden/output norm；
- fixed-H与FULL_STOP rollout；
- 与kinematic和event-vector directional baselines的差值。

该实验回答方向是早期选定还是逐步累积，不作为是否允许训练M3的gate。全队列暂定M2-2RANK，任何改变必须在Phase C前写入冻结补充条款。

### B4. Joint fit、freeze replay与self-feeding

使用平衡的fractional design，而不是把所有因素全部交叉：

1. anchored joint fine-tuning；
2. component-isolation fit/replay；
3. teacher forcing；
4. 3-step sampled self-feeding sensitivity。

对M0–M3比较：

- next-rank calibration；
- fixed-H3/H5 field和spread；
- FULL_STOP length；
- seed stability；
- shared-parameter drift。

teacher forcing始终是正式主训练。self-feeding结果只决定是否值得另开后续训练实验，不影响v0.1正式主链。

### B5. Synthetic identifiability maps

系统扫描：

```text
eta × beta × gamma
contact count × event count × rank count
noise × tie size × shaft-like geometry × STOP variability
```

每个cell保存ground truth、recovery probability、bias和model confusion。弱信号不可恢复不阻止真实数据实验；只修复generator、equivalence或实现错误。

输出：

```text
toy_identifiability/IDENTIFIABILITY_GRID.csv
toy_identifiability/IDENTIFIABILITY_SUMMARY.json
```

### B6. 设计冻结记录

Phase B只冻结实现选择，不根据科学显著性删实验：

```text
CORE_FRAME
PRIMARY_M2_GATE
anchor penalty and learning-rate ratio
teacher-forcing budget
decoder temperatures procedure
Monte Carlo precision thresholds
```

输出：

```text
DESIGN_CHOICE_EFFECTS.csv
DESIGN_FREEZE_AMENDMENT.json
PILOT_RESOURCE_AUDIT.json
```

该 amendment 必须由用户审阅后才进入正式训练。

## Phase C｜全队列四个主模型：336 units

前提是用户已冻结一个28-fit `CORE_FRAME`：

```text
28 patients × 4 models × 3 seeds = 336 units
```

模型：

```text
DM0_ISOTROPIC
DM1_FREE_AXIS
DM2_LOCAL_DIRECTIONAL
DM3_AXIS_FEEDFORWARD_TRANSIENT
```

### C1. 每条seed chain

1. M0从头训练；
2. M1从M0 warm start，anchored joint fine-tune；
3. M2从M1 warm start，使用冻结continuous gate；
4. M3从M2 warm start，`gamma=0`初始化；
5. 所有模型teacher-forced训练；
6. split 1选择checkpoint与decoder calibration；
7. split 2只做development QA；
8. split -1保持封存；
9. 保存上一层checkpoint、parameter drift和zero-component replay。

每个M2 checkpoint另做不重训的`3RANK/ONLINE gate replay`，使用相同随机数并明确标记为implementation sensitivity。

### C2. 低成本 baselines

对每个M0/M1 chain运行：

- `LAYOUT_AXIS_ANISOTROPY`：contact-cloud PCA1与dominant-shaft固定轴的scalar/grid fit；
- `EARLY_DISPLACEMENT_KINEMATIC`：early-displacement regression/classifier；
- `EVENT_VECTOR_DIRECTIONAL`：M0上的event-vector directional scalar/grid fit；
- `STATIC_READOUT`；
- one-step gain-matched M1/M2 sensitivity。

它们不算完整RNN unit，但必须有配置、hash、train/calibration score和model-unseen结果。

### C3. 单元产物

每unit必须有：

```text
config.json
provenance.json
checkpoint.pt
decoder.pt
training_curve.csv
validation_metrics.json
parameter_drift.json
numerical_audit.json
parameter_hashes.json
DONE.json or FAILED.json
```

有限但不返回低活动状态的模型保存为有效checkpoint并标记 `FINITE_NONRETURNING`；只有nonfinite或无法执行才是失败。

## Phase D｜M3替代机制：84 units

所有controls从预先固定的M2 seed chain出发，不根据validation挑seed：

```text
28 × DM3_GAIN_MEMORY × 1 seed
28 × DM3_SYMMETRIC_MATCHED × 1 seed
28 × DM3_AXIS_SHUFFLED_TRIANGULAR × 1 seed
= 84 units
```

### D1. Gain/memory control

- 不加入`F`；
- 允许更大recurrent gain和更慢leak；
- 容量不小于M3；
- 同时保存STOP probability、fixed-H spread和FULL_STOP length。

### D2. Symmetric control

- 使用与M3相同node pairs、distance distribution和weight scale；
- 将方向前馈替换为对称耦合；
- 与M3使用相同训练预算和anchor规则。

### D3. Axis-shuffled triangular control

- primary permutation由patient ID和冻结seed hash生成；
- 在distance/degree bins内打乱ordering；
- 保存三角性、非零数和权重分布审计；
- 另做7个仅重新拟合scalar strength的permutation sensitivities，形成患者内null分布。

## Phase E｜Model-unseen随机生成与动力学分析

### E1. Development QA

在split 2验证：

- sampler终止与不重复contact；
- length/contact-count calibration；
- fixed-H与FULL_STOP结果分离；
- energy score、coverage和covariance数值；
- common RNG一致；
- 32/128/256 draws的Monte Carlo误差。

只修实现错误，不按split 2模型胜负修改科学endpoint。

### E2. 打开split -1

代码、模型、decoder、reference-event manifest和统计脚本冻结后，计算：

- next-rank/STOP/cardinality prediction；
- H3/H5 endpoint、spread和field；
- FULL_STOP endpoint、length和terminal field；
- `r_last`和`r_late`；
- multivariate/contact-field energy score；
- train-only mode Brier/log score；
- distribution coverage和covariance alignment。

Monte Carlo：

1. 所有events先32 draws；
2. 每位患者20–30个hash-selected reference events扩到128；
3. 只有冻结精度标准未满足者扩到256；
4. 所有模型共用随机数表。

### E3. G0–G3比较表

建立patient-level paired tables：

```text
DM1_FREE_AXIS - LAYOUT_AXIS_ANISOTROPY
LAYOUT_AXIS_ANISOTROPY - DM0_ISOTROPIC
DM2_LOCAL_DIRECTIONAL - DM1_FREE_AXIS
DM2_LOCAL_DIRECTIONAL - EARLY_DISPLACEMENT_KINEMATIC
DM2_LOCAL_DIRECTIONAL - EVENT_VECTOR_DIRECTIONAL
DM3_AXIS_FEEDFORWARD_TRANSIENT - DM2_LOCAL_DIRECTIONAL
DM3_AXIS_FEEDFORWARD_TRANSIENT - DM3_GAIN_MEMORY
DM3_AXIS_FEEDFORWARD_TRANSIENT - DM3_SYMMETRIC_MATCHED
DM3_AXIS_FEEDFORWARD_TRANSIENT - DM3_AXIS_SHUFFLED_TRIANGULAR
```

每项分开报告prediction、fixed-H distribution、FULL_STOP、gain和uncertainty，不压成一个总判决。

同时生成：

- M1 free axis跨seed稳定性；
- 冻结shared checkpoint后、仅重拟合`theta/eta`的train-block profile；
- free axis与contact-cloud PCA1、dominant shaft的夹角；
- M2 `s_k` emergence、翻转率和gate replay；
- M3及三类controls的fixed-H、FULL_STOP和一步响应分解。

输出：

```text
AXIS_LAYOUT_PER_PATIENT.csv
DIRECTION_GATE_PER_PATIENT.csv
MODEL_UNSEEN_DISTRIBUTION_PER_PATIENT.csv
M3_CONTROL_COMPARISON_PER_PATIENT.csv
```

## Phase F｜输入counterfactual与latent解释

### F1. Reference states

每位患者预冻结：

- early 4个；
- middle 4个；
- late 4个。

按recording block、起点区域、prefix长度和mode uncertainty分层；不能按扰动效果挑选。每branch先64 draws，covariance不稳定时扩到128。

### F2. 输入空间 primary experiments

#### 轴向 contact substitution

生成`+axis/-axis/orthogonal`匹配替换；保存shaft、distance和train-support matching quality。

#### Tie/order editing

运行adjacent swap、tie merge和supported tie split；分别评价mode稳定性、`r_late`、field和微观序列变化。

#### Extent editing

在方向已稳定prefix中增加/删除一个轴中段真实contact；评价H3/H5 spread、FULL_STOP length和mode identity。

未匹配患者不删除，单独报告每类实验的实际分母和失败原因。

### F3. Latent/Jacobian secondary

1. 计算真实轨迹H=1...10的state/output gain；
2. 记录peak time、峰后变化和长期无输入状态；
3. 沿`±v1`做小剂量hidden perturbation；
4. 比较random、phase-shuffled和immediate-output-matched controls；
5. 将latent response与输入counterfactual产生的真实输出位移对齐。

不要求完整双重解离。方向、范围或长度中任何独立效应均保留。

输出：

```text
PREFIX_COUNTERFACTUAL_PER_PATIENT.csv
TIE_ORDER_RESPONSE_PER_PATIENT.csv
EXTENT_RESPONSE_PER_PATIENT.csv
FINITE_HORIZON_GAIN_PER_PATIENT.csv
LATENT_RESPONSE_PER_PATIENT.csv
```

## Phase G｜Seizure工程并行、评分后解封

### G1. 可提前并行完成的工程工作

- static 0–10 s真实onset field重提与旧target等价审计；
- same-block pseudo-onset静态fields；
- baseline robust-z；
- shaft/contact leave-out folds；
- dynamic BB150可靠性QC；
- contact join、Nyquist、gap、bad segment审计。

这一步可以与Phase C/D并行，但不能读取模型选择结果来改seizure pipeline。

### G2. Target-free IED basis冻结

全队列interictal结果完成后：

1. 按预冻结规则选择一个interictal motif，不看seizure；
2. 用split 0起点分布生成IED rollout fields；
3. 用split 0 start-removed mean participation构造primary static field `m_p`，STATIC_READOUT期望场为sensitivity；
4. 回归掉`[1,m_p,shaft/geometry]`；
5. 只用residual IED fields选择1–2维`U_p`；
6. 冻结basis、维数、sign、spatial CV、pseudo-onset和scorer hashes。

### G3. S1 static incremental reuse

比较：

```text
S0: static participation only
S1: static participation + residual IED basis
```

主endpoint：leave-one-shaft-out或预冻结LOCO的`Delta E/DeltaNLL`。先在每个seizure/pseudo-onset内部评分，再折叠到患者。真实onset与same-block pseudo-onset比较。

A/Q作为二维辅助结果完整展示，不作为conjunctive gate。

### G4. S2 dynamic branch

若time-resolved BB150可估，运行early coefficient→late field prediction。不可估时记录`NOT_IDENTIFIABLE`，不影响S1。

输出：

```text
SEIZURE_STATIC_FIELD_AUDIT.json
PSEUDO_ONSET_MANIFEST.csv
IED_RESIDUAL_BASIS_MANIFEST.json
SEIZURE_INCREMENTAL_REUSE_PER_EVENT.csv
SEIZURE_INCREMENTAL_REUSE_PER_PATIENT.csv
SEIZURE_DYNAMIC_REUSE_PER_PATIENT.csv
```

## Phase H｜G6探索、evidence matrix与图文收口

### H1. G6低成本探索

无论G1–G5结果如何都运行：

- recurrent response与readout residual的rank spectrum；
- M3后剩余field covariance；
- local-path distance与`event_lag_raw` residual；
- across-block repeatability。

这些sidecars只决定未来M4/M5是否值得另写spec，不在本阶段训练low-rank或delay model，也不称白质连接。

### H2. Evidence matrix

为G0–G6分别填写：

```text
predictive effect
distribution effect
dynamical signature
alternative-control comparison
observable perturbation consequence
denominator / CI / uncertainty
```

每格使用：

```text
SUPPORTED
PARTIAL
NOT_DETECTED
UNDERPOWERED
NOT_IDENTIFIABLE
```

不生成串联总分，不因某个Goal阴性删除后续结果。

### H3. Figure 6与Supplementary

Figure 6只展示实际有信息量的结果，候选顺序：

```text
真实rank-set输入和随机输出
→ 模板均值与模板内方差
→ layout axis与free axis
→ local directional bias
→ feedforward与三类替代机制
→ observable prefix counterfactual
→ static-only vs static+IED seizure increment
```

旧connection arms、orthogonal M2、M3N、frame细节、toy maps、self-feeding和latent controls进入Supplementary。每个含图目录同时生成中文`figures/README.md`，并对PNG/PDF/SVG做同状态与目视QA。

### H4. 最终产物

```text
EVIDENCE_MATRIX.json
MODEL_UNSEEN_PER_PATIENT.csv
TOPIC5_DYNAMICAL_MOTIF_CLOSEOUT.md
FIGURE6_SOURCE_DATA/
CLOSEOUT_AUDIT.json
```

用户终审前不commit/push。

## 正式资源总表

| 部分 | 数量 | 是否进入420 |
|---|---:|---|
| 四个主模型，3 seeds | 336 | 是 |
| 三个M3替代机制，1 fixed seed | 84 | 是 |
| layout/kinematic/event-vector/static baselines | 低成本fit | 否 |
| 7个axis-shuffled scalar sensitivities | 标量重拟合 | 否 |
| Phase B frame/operator/training experiments | pilot单列 | 否 |
| stochastic rollout与扰动 | evaluation branches | 否 |

正式RNN训练预算合计420 units。Phase B资源在pilot后单独归档，不混入正式完成率。

## Spec–plan 对照表

| Spec合同 | Plan执行位置 |
|---|---|
| §2 frame experiment | A2、B2、B6 |
| §3 split/label/static baseline | A1、A3、C2 |
| §4 RNN与stochastic decoder | A4、B0、E1 |
| §5.1–5.4 四个主模型 | B3、C |
| §5.5 三个M3替代机制 | D |
| §6 joint/isolation/self-feeding | B4、C1 |
| §7 fixed-H、FULL_STOP、Monte Carlo | E |
| §8 observable/latent perturbation | F |
| §9 identifiability maps | B5 |
| §10 static/dynamic seizure reuse | A5、G |
| §11 evidence matrix与G6 | H1、H2 |
| §13 336+84资源 | C、D、正式资源总表 |
| §14 Figure 6/Supplementary | H3 |
| §15 engineering stops | A、B0、唯一停止条件 |

## 执行顺序

```text
A minimal integrity + parallel metadata QC
→ B 6–8 patient design experiments + identifiability maps
→ user review and DESIGN_FREEZE_AMENDMENT
→ C 336 main units
→ D 84 mechanism-control units
→ E model-unseen stochastic evaluation
→ F observable counterfactuals, then latent analysis
→ G static incremental seizure reuse; dynamic branch if estimable
→ H G6 exploration + evidence matrix + Figure 6/Supplementary
```

## 唯一停止条件

- provenance、event、split或contact identity无法解析；
- label/future/seizure leakage；
- zero-component嵌套等价失败；
- sampler、STOP、repeat mask或closed-loop replay错误；
- nonfinite、shape/device、checkpoint或RNG/resume故障；
- Phase B后尚未取得用户对`DESIGN_FREEZE_AMENDMENT`的审阅。

轴不稳定、M2/M3阴性、gain不回落、输入扰动无效、toy弱信号不可恢复或dynamic seizure不可估，全部作为结果继续收集，不停止其他预注册实验。
