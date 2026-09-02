# Topic 5.2 患者特异 latent propagation landscape 与 perturbation response v0.2 执行计划

> 对应 spec：
> `docs/superpowers/specs/2026-08-14-topic5-latent-propagation-landscape-v0-2-design.md`
>
> 状态：**IMPLEMENTATION PLAN — AUTHORIZED, EXECUTED AND CLOSED；2026-08-14。**
> 执行结果：`docs/archive/topic5/latent_propagation_landscape_v0_2_closeout_2026-08-14.md`；
> machine audit：`results/topic5_latent_propagation_landscape_v0_2/CLOSEOUT_AUDIT.json`。
>
> 本计划实现三个相互独立判定的 goals：5.2A latent geometry/dynamical transport、
> 5.2B axis-specific perturbation、5.2C patient-specific spatial/cross-model convergence。
> 三者有数据依赖顺序，但没有 `P` 值驱动的停止树。

## 0. 交付目标与执行原则

最终交付以下七层证据，而不是单一 progress gate：

```text
C1 two-coordinate geometry
C2 dynamical transport
C3 axis-specific perturbation
C4 topology convergence
C5 patient-specific data alignment
C6 SNN cross-model convergence, eligibility-dependent
C7 early-ictal exploratory alignment
```

执行原则：

1. 保留 Topic 5.1 的 28 patients / 42 fits / 630 checkpoint cells，不重训、不选 topology；
2. raw hidden state 是 primary，observable residualization 只是 sensitivity；
3. progress 与 continuous future-field coordinates 平行研究；
4. 所有预注册科学分支都执行，前一分支阴性只限制措辞，不删除后一分支；
5. 只有 engineering integrity 和单次 perturbation numerical validity 是 hard gates；
6. Goals 5.2A/B 不读取 SNN/early-ictal values；RNN/data fields 冻结后才依次解封 SNN、early-ictal；
7. 工程完成、输入可用、科学支持和论文图资格分别裁定。

阶段完成标记固定为：

```text
LATENT_GEOMETRY_COMPLETE
PERTURBATION_RESPONSE_COMPLETE
SPATIAL_CONTROL_FIELD_COMPLETE
```

它们表示实验闭合，不表示结果阳性。

## 1. 代码、结果与只读边界

### 1.1 计划新增文件

```text
src/topic5_latent_landscape_v0_2.py

scripts/audit_topic5_latent_landscape_inputs_v0_2.py
scripts/stream_topic5_latent_system_id_v0_2.py
scripts/fit_topic5_latent_geometry_v0_2.py
scripts/analyse_topic5_latent_transport_v0_2.py
scripts/freeze_topic5_latent_reference_states_v0_2.py
scripts/run_topic5_axis_perturbations_v0_2.py
scripts/build_topic5_spatial_response_fields_v0_2.py
scripts/audit_topic5_snn_alignment_inputs_v0_2.py
scripts/align_topic5_rnn_snn_fields_v0_2.py
scripts/score_topic5_latent_early_ictal_v0_2.py
scripts/adjudicate_topic5_latent_landscape_v0_2.py
scripts/plot_topic5_latent_landscape_v0_2.py
scripts/audit_topic5_latent_landscape_closeout_v0_2.py

tests/test_topic5_latent_landscape_v0_2.py
tests/test_topic5_latent_landscape_pipeline_v0_2.py
```

如果现有 helper 满足合同，优先复用：

- `src/topic5_wiring_economy_rnn.py` 的 hidden update/readout/STOP；
- `src/topic5_rnn_motif_v0_4.py` 的冻结 size decoder；
- Topic 5.1 的 checkpoint resolver、fit manifest、field producer、patient aggregation 和 target guard；
- 既有 contact/plane/shaft geometry utilities。

禁止修改 frozen v0.5 producer、checkpoint 或结果。必要的新 adapter 写在 v0.2 module 中，并以 parity test
证明与 parent producer 一致。

### 1.2 新结果根

```text
results/topic5_latent_propagation_landscape_v0_2/
```

最低文件合同：

```text
CONTRACT.json
CHECKPOINT_MANIFEST.csv
INPUT_AUDIT.json
RESOURCE_BUDGET.json
PASS1_STREAMING_MANIFEST.json
LATENT_GEOMETRY_SUMMARY.json
DYNAMICAL_TRANSPORT_SUMMARY.json
REFERENCE_STATE_MANIFEST.csv
PASS2_PERTURBATION_MANIFEST.json
PERTURBATION_RESPONSE_MATRIX.json
FINITE_TIME_RESPONSE_FIELDS.npz
SPATIAL_PATCH_CONTROL_FIELDS.npz
DATA_ALIGNMENT_SUMMARY.json
SNN_INPUT_ELIGIBILITY.json
SNN_ALIGNMENT_SUMMARY.json
EARLY_ICTAL_EXPLORATORY_SUMMARY.json
COHORT_PATIENT_TABLE.csv
CLAIM_LADDER_ADJUDICATION.json
CLOSEOUT_AUDIT.json
figures/README.md
```

只有实际生成 figures 后才创建 `figures/README.md`。大数组可按 unit 使用 chunked Zarr/HDF5 或分块 NPZ，
但上表的 cohort index 和 hash manifest 必须存在。

## 2. Phase 0：parent lock、输入资格与 prefreeze

### 2.1 Parent lock

读取并记录 Topic 5.1 v0.5：

- `CLOSEOUT_AUDIT.json`；
- `FINAL_CLAIM_ADJUDICATION.json`；
- run contract 与 model/cache manifests；
- 42-fit split、H、node mask、contact order、decoder artifacts；
- train-only TA/TB/event-field manifest；
- parent commit/tag 和当前 worktree diff hash。

为每个 `(patient, fit, arm, seed)` 写入：

```text
checkpoint_path, checkpoint_sha256, checkpoint_source,
config_sha256, split_sha256, H_sha256, node_mask_sha256,
contact_order_sha256, decoder_sha256, n_nodes, n_contacts,
n_events_axis_train, n_events_axis_validation, n_events_test
```

`checkpoint_source` 只能是：

```text
V0_5_FORMAL_UNIT
V0_3_EXACT_REUSE
```

630 cells 必须精确解析为 531 formal + 99 exact reuse。禁止 nearest-path fallback、复制 seed 或缩小分母。

### 2.2 Split 和 target seal

沿用 frozen chronological split。没有独立 validation 的 checkpoint，将原 train80 固定拆为
`axis_train60/axis_validation20`，heldout20 不动。冻结：

- patient/fit/event IDs；
- event-to-field producer 和 start-removed/full masks；
- phase bins、PCA dimensions、spline grid、regularization grid；
- reference-state sampling rule；
- perturbation doses、controls、support thresholds；
- spatial nulls 和 cross-patient geometry mappings；
- target embargo globs 与 value-access log。

Goals 5.2A/B 及 5.2C data/patch 分支禁止读取 SNN/early-ictal values。SNN phase 0 只读 metadata；
early-ictal 连 metadata join 之外的 target values 都不读。

### 2.3 SNN metadata-only eligibility pre-audit

只登记：producer path/hash、baseline/engine ID、runtime mode、duration、replication、late-runaway、natural-mode
status、patient denominator 和 field schema。不得读取 field arrays 或 alignment values。

当前 candidate 不预设 cohort 资格；资格状态只允许：

```text
LOCKED_ACCEPTED
CANDIDATE_REPLICATED
DIAGNOSTIC_ONLY
```

### 2.4 Gate E0-A

`INPUT_AUDIT.json` 必须满足：

```text
resolved_cells = 630
target_values_read = false
parent_hashes_complete = true
split_and_ordering_consistent = true
status = PASS
```

失败时只修 input/replay 合同，不进入科学分析。

## 3. Milestone A：完整 decoder state replay 与资源预检

### 3.1 `q=(h,r,k)` parity replay

每个 analysis cell 确定性选择 train/validation/test 的短、中、长事件和早、中、晚 rank states，从零状态
重放 teacher-forced trajectory，核对：

- hidden state；
- contact logits；
- STOP logits/probability/decision；
- size logits/decision；
- recruited/repeat mask；
- tie break、STOP precedence、maximum-rank 和 absorbing STOP；
- generated next set。

同 dtype/device 以 exact 或 frozen numerical tolerance 比较。CPU/GPU 差异必须形成显式数值合同，不能事后
放宽 tolerance。

反例测试固定相同 `h`、改变 `r` 或 `k`，应能改变合法 decoder decision，防止代码退化为错误的 `G(h)`。

### 3.2 两遍资源预算

先按 manifest 估算：

```text
Pass 1 streaming sufficient statistics
Pass 1 low-dimensional projections
Pass 2 selected full q states
axis/control rollout branches
patch branches
temporary files and resume overhead
```

要求：

- 不生成全 cohort monolithic hidden/logit trajectory archive；
- unit 原子写入、SHA256、resume sentinel；
- 峰值 RAM/VRAM/disk 有实测 sentinel；
- 预计 disk 小于 preflight free space 的 50%；
- patch 扫描超过预算时，只能启用预冻结的 geometry-only coarsening，不能按 response 选中心。

### 3.3 Gate E0-B

只有 replay parity、full-state clone、parameter hash 和 resource preflight 全部通过，才启动 Pass 1。

## 4. Milestone B：Pass 1 streaming system identification

### 4.1 流式提取

逐 cell 确定性 replay，不保存全部高维 trajectories。通过可重复的多遍 streaming 计算：

1. node-wise robust center/scale；
2. event-first、phase-balanced incremental PCA sufficient statistics；
3. phase-binned state mean/covariance 和 local residual covariance；
4. observables `o_k`、`s_step` 和 `s_contact`；
5. start-removed/full event fields 与 train-only continuous `u_e`，并冻结 canonical/shared 与 generic/own
   identifiability tier；
6. raw-state、observable-only、observable+raw 和 residual-state projections；
7. reference-state candidate IDs 和 replay offsets。

事件少于 2 个 rank sets 时不能定义 `s_step`，记录 `PROGRESS_PHASE_UNDEFINED`，不以零除或人工复制。

在任何 hidden response 提取前先写 `PASS1_EVENT_SAMPLE_MANIFEST.csv`：split 内按 frozen event identity hash
排序，train/validation/test caps 分别为 `1024/512/512`，小 fit 全纳入；同 fit 跨 arm/seed 共用，且同一患者
`own_a/own_b` 共用 event identities。manifest
记录全部 eligible denominator、inclusion fraction 与 hash。future-field axis 使用全部 train event fields，不受
hidden replay cap 影响。Pass 2 的 reference-event cap 为 frozen heldout sample 内 64 events/fit，同样只按 identity
hash，不读 mode、`u_e`、effect 或 response。

### 4.2 Future-field axis freeze

每个 `(patient, fit)` 只用 axis-train events：

1. 复用 parent train-mode alignment，并先审计每个 fit 是 `CANONICAL_AB_SHARED` 还是
   `WITHIN_FIT_MODE_ONLY`；
2. shared fit 构造 start-removed `A−B` centroid contrast；own fit 构造 start-removed
   `mode1−mode0` contrast，后者不得命名为 A↔B；
   centroid 的 train label 必须读取 parent `full_train_mode`，不得读取 `prefix_mode`；
3. center、normalize 并冻结方向；
4. 计算 event-level continuous `u_e`；
5. 审计 axis norm、train/validation split stability、eligible events/contacts；
6. 只有 mode/common-contact 缺失、nonfinite 或数值退化才标记 `FIELD_AXIS_NOT_IDENTIFIABLE`；
7. 小分母或 split 不稳定标记 `FIELD_AXIS_LOW_RELIABILITY`，保留 effect/uncertainty，不按 `P` 值删除。

binary TA/TB 仅在 shared fits 用于 sensitivity 和图示；continuous `u_e` 是 primary future-field label。
heldout binary sensitivity 必须由 frozen train full-event centers 分配 full-event mode；prefix-only mode 只能作为
prefix classifier diagnostic，不能充当独立 future outcome。
all-fit generic 与 shared-fit canonical 结果必须分别报告。禁止跨 `own_a/own_b` node spaces 拼接 hidden axis、
tangent 或 state chord；两者只能在各自产生 contact-space response 后做 patient-level 汇总。
axis-train fields 与 heldout-test empirical fields 分开写 manifest；test A/B 方向沿用 train alignment，禁止
在 test 上重排 mode label。

### 4.3 Pass 1 可复现性

同一个 sentinel cell 至少以两种 chunk order 重跑，要求 robust statistics、PCA subspace、`u_e` 和 selected
reference IDs 在冻结 tolerance 内一致。输出 `PASS1_STREAMING_MANIFEST.json`。

## 5. Milestone C：Goal 5.2A latent geometry

### 5.1 Raw-state primary 与 comparisons

每个 `(patient, fit, arm, seed)` 拟合容量匹配模型：

```text
O  = observable-only
P  = gamma(s)
PF = gamma(s) + u*b(s)
PF-null = gamma(s) + shuffled(u)*b(s)
```

流程固定为：

1. raw standardized hidden state primary；
2. observable-only comparison；
3. observable + hidden incremental decoder；
4. observable-residualized hidden sensitivity。

PCA `d={2,4,8}`、spline knots、smoothness 和 decoder regularization 只由 axis-validation 选择；heldout test
不允许重新选 dimension、phase、sign 或 model family。

primary phase 为 `s_step=(k-1)/(K-1)`；`s_contact` 只作 sensitivity。

incremental decoder 在每个 phase bin 只用 train states 拟合 `Z~O`，以 frozen `Z_resid` 构造 `[O,Z_resid]`；
dimension/ridge 在同一 phase 的 validation states 选择。heldout negative `R2` 不截断、不回看重选。

### 5.2 Future-field emergence

在 frozen prefix-step/phase bins 上，用 partial hidden state 预测 train-defined `u_e`，输出 heldout：

- continuous `R2_future_field(k)`；
- calibration/error；
- binary TA/TB Brier/AUC sensitivity；
- first phase crossing a validation-frozen practical-effect threshold；
- L0/L1/L2m/L3 与 C-suffix 的整条 emergence curve contrast。

C-suffix 允许拥有 progress axis；关键问题是它的 future-field commitment 是否减弱或延迟。不得只挑最有利
rank step。

本节输出两套预注册 denominator：所有可辨识 fits 的 generic within-fit emergence，以及 shared fits 的
canonical A/B emergence。只有后一套可写 TA/TB commitment；前者不得通过增加 fit 数量替代前者。

### 5.3 Geometry summary

保存：

- `gamma(s)`、`b(s)` 及其 raw-metric backprojection；
- progress/field axis norm、angle、collinearity；
- within-mode convergence、between-field separation；
- heldout O/P/PF/PF-null effects；
- raw/residual sensitivity；
- patient/fit/arm/seed denominators。

若 field axis 与 progress tangent 近共线，只标记对应 phase `FIELD_AXIS_COLLINEAR`；不旋转到结果更好的方向。

## 6. Milestone D：Goal 5.2A dynamical transport

### 6.1 Teacher-forced Jacobian summaries

对 heldout states 计算真实 next input 条件下的 local Jacobian-vector products，不显式保存全部 `N×N`
Jacobian。评估：

- progress tangent transport；
- future-field tangent transport；
- local tangent gain；
- conditional normal-direction response；
- local normal singular spectrum；
- event-to-curve residual 的一步变化。

controls 固定为 phase-shuffled/event-shuffled axes、high-variance PCA directions 和 C-suffix 对应 axes。

### 6.2 Closed-loop projected transition field

从完整 `q` 自由 rollout，把访问状态投影到 `(z_prog,z_field)`，估计 phase-conditional
`E[Delta z | z,o]`。teacher-forced 和 closed-loop 分开保存；不把它叫 autonomous `h` flow。

### 6.3 C1/C2 adjudication

按 spec 的 C1/C2 family 计算 effect、CI、patient signs 和 Holm adjustment。输出：

```text
LATENT_GEOMETRY_SUMMARY.json
DYNAMICAL_TRANSPORT_SUMMARY.json
LATENT_GEOMETRY_COMPLETE
```

C1/C2 supported 或 unsupported 都继续 Milestone E；只有 axis 数值不可定义时才跳过对应 axis 的
perturbation，并保留明确 denominator。

## 7. Milestone E：冻结 reference states、axes 与 controls

### 7.1 Reference states

在读取 heldout perturbation response 前，按 event IDs 和 geometry-only rules 冻结：

- phases：最接近 `s={0.25,0.50,0.75}` 的合法 state；
- 每 event 每 phase 最多一个 state；
- event-first sampling，避免长事件占权；
- axes：train/validation frozen `a_prog(s)`、`a_field(s)`；
- doses：`0.25/0.5/1.0 local residual SD`，`0.5` primary；
- tau：open-loop `1..3`，closed-loop 到 STOP/max rank；
- support q95 和 node-wise bounds；
- empirical chord matching thresholds。

不得按 C1/C2 effect、患者方向或 heldout output 选择 reference states。

### 7.2 Control families

每个数值合法 reference state 预生成：

- 8 个 norm-matched local-normal directions；
- phase-shuffled progress/field axes；
- 前 3 个可辨识 PCA high-variance axes；
- C-suffix 对应 axes；
- matched-observable small-`u` empirical chords；
- empirical A→B field chords，最多 5 个最近合法 pairs，实际数全报。

primary axis-control 只硬匹配 perturbation norm 与同一 local-support rule。即时 logit/STOP change 和
`||J delta h||` 全保存，作为连续 covariates 和 sensitivity strata，不设 20-control 超级 caliper。

### 7.3 Freeze artifact

`REFERENCE_STATE_MANIFEST.csv` 至少包含：

```text
patient, fit, arm, seed, event_id, step, phase,
q_replay_key, axis_status, dose, support_neighbors,
control_family, control_ids, transplant_pair_ids,
all_input_hashes
```

manifest hash 写入后，Pass 2 不能增删 state，只能记录 numerical failure。

## 8. Milestone F：Pass 2 / Goal 5.2B perturbation

### 8.1 Selected-state extraction

只为冻结 reference IDs 重放并保存：

```text
q=(h,r,k)
current/future teacher-forced inputs
logits, STOP, size state
raw-metric local axes and normal basis
conditional-support neighbors
empirical chord endpoints
```

提取后复核 parameter/input hashes 与 Pass 1 一致。

### 8.2 Perturbation branches

从相同 `q` 分叉运行：

1. `h +/- epsilon*a_prog(s)`；
2. `h +/- lambda*a_field(s)`；
3. `h_A + eta*(h_B-h_A)`，`eta={0.25,0.5,1.0}`；
4. 所有预冻结 control families；
5. unperturbed replay。

只改变 `h`，`r,k` 不变。每个 branch 独立通过 Gate N0：node bounds、conditional kNN、manifold residual、
finite logits/state、decoder executable。失败不 clipping、不降 dose、不换 control，只写 reason code。

### 8.3 Open-loop

未来 `tau=1..3` 使用完全相同的真实 rank inputs。保存：

- hidden/logits/STOP probability；
- `z_prog/z_field`；
- finite-time contact response；
- immediate `tau=0` audit。

`tau=0` 不进入 primary functional response。

### 8.4 Closed-loop

从同一完整 `q` 开始，仅使用冻结 decoder，保存：

- per-step contact logits/probabilities；
- generated sets 和 cumulative fields；
- STOP probability trajectory；
- discrete STOP length；
- terminal continuous field/mode score；
- full `z_prog/z_field` trajectory。

branch STOP 后进入 absorbing state，不复制最后一个活动 logit 作为未来响应。
per-step response 只在正负两支均 active 的 risk set 上汇总并逐 `tau` 报 denominator；terminal cumulative
field 与 STOP trajectory 另行汇总。

### 8.5 Functional response matrix

使用 train-only、跨表征的 contact-space progress/field axes 计算；它们独立于 hidden coordinate metric，
但 field axis 仍来自同一 axis-train data contract，不称独立数据复制。continuous response 统一使用
repeat-mask 前的 finite contact logits；post-mask logits/availability 和 discrete decisions 另存。

```text
R_prog<-prog
R_prog<-field
R_field<-prog
R_field<-field
```

形成 co-primary：

```text
D_prog  = R_prog<-prog  - abs(R_field<-prog)
D_field = R_field<-field - abs(R_prog<-field)
```

`D_prog/D_field` 进入两检验 Holm。`z` persistence、STOP/remaining length 是 diagnostics/secondary。
empirical A→B chord 单独检验 output field 是否朝 `u_B-u_A` 改变。

### 8.6 C3/C4 adjudication

按 event→phase/tau→seed→arm/fit→patient 聚合，输出：

- 完整 2×2 matrix；
- doses/time courses；
- control-family contrasts；
- immediate-output/gain adjusted sensitivity；
- per-arm response 和 arm heterogeneity；
- L0/L1/L2m/L3 convergence 与 C-suffix difference；
- actual scheduled/eligible/completed denominator。

无论 supported/unsupported 都写：

```text
PERTURBATION_RESPONSE_COMPLETE
```

并继续 Goal 5.2C。

## 9. Milestone G：Goal 5.2C finite-time spatial fields

### 9.1 Contact response fields

对每个 fit/arm/seed/phase/tau 按 event-first 聚合 progress 和 future-field central-difference contact logits，
生成 `FINITE_TIME_RESPONSE_FIELDS.npz`。同时保存：

- sign/orientation provenance；
- contact order/mask；
- start-removed/full data-field variants；
- seed/arm stability；
- immediate `H a` sensitivity，但不以其替代 finite-time response。

### 9.2 Data-field alignment

只用从未参与 axis/model/sign 选择的 heldout-test events 构造 evaluation common/contrast fields，并在本患者
exact common contacts 上检验：

- `g_RNN_prog` vs train-only `(TA+TB)/2` common field；
- `g_RNN_field` vs train-only `TA−TB` contrast field；
- signed Spearman/cosine；
- synchronized all-contact、shaft/distance、variogram/autocorrelation-preserving nulls；
- fit/seed/arm robustness。

方向由 train-only axes 冻结，不做 max-absolute 或按 heldout 翻符号。
heldout mode/common-contact 缺失、nonfinite 或 norm 数值退化时标记 `DATA_FIELD_NOT_IDENTIFIABLE`；事件少或
reliability 低标记 `DATA_FIELD_LOW_RELIABILITY` 并报告 uncertainty。不得用 axis-train field 替代 C5 验证。

### 9.3 Cross-patient identity null

field values 解封前先生成 geometry-only `q→p` mappings。mapping 只能读取 contact coordinates、shaft、spacing
和 frozen plane normalization，不得用 alignment 选择 reflection/rotation/scale/contact pairs。

通过 registration audit 的 patient 计算 same-patient minus other-patient median margin。映射不可辨识时输出
`CROSS_PATIENT_IDENTITY_NOT_IDENTIFIABLE`，只保留 within-patient alignment，禁止写 patient-specific。

### 9.4 Tissue patch perturbation

沿用相同冻结 reference states，在 tissue grid 上做 Gaussian positive/negative central differences：

- width 使用 local node spacing 的冻结倍数；
- center 全网格或 preflight-frozen geometry-only coarse grid；
- Gaussian node vector 先做 raw-metric unit normalization，再乘与 axis perturbation 相同定义的 local-SD dose；
- endpoint 使用 finite-time pre-repeat-mask contact-logit response 对 train-defined progress/field axes 的投影；
- 与 axis perturbation 相同 Gate N0；
- 输出 progress-control 与 field-control susceptibility maps；
- 保存 sign、phase、tau、seed、arm consistency；
- 区分 immediate readout 与 finite-time response。

axis perturbation 阴性不取消 patch scan。patch 仍是 model-internal field，不称 stimulation map。

### 9.5 C5 adjudication

C5 分开报告 within-patient alignment、identity margin、spatial null 和 denominator。完成标记：

```text
SPATIAL_CONTROL_FIELD_COMPLETE
```

## 10. Milestone H：SNN eligibility 与 cross-model convergence

### 10.1 Value-access unlock

只有以下文件 hash 冻结后，才可读取 SNN values：

```text
REFERENCE_STATE_MANIFEST.csv
PERTURBATION_RESPONSE_MATRIX.json
FINITE_TIME_RESPONSE_FIELDS.npz
SPATIAL_PATCH_CONTROL_FIELDS.npz
DATA_ALIGNMENT_SUMMARY.json
cross-patient geometry mapping manifest
```

`audit_topic5_snn_alignment_inputs_v0_2.py` 生成 `SNN_INPUT_ELIGIBILITY.json`，逐 field 裁定：

- `LOCKED_ACCEPTED + adequate denominator`：允许 cohort C6；
- `CANDIDATE_REPLICATED/small denominator`：case-series exploratory；
- `DIAGNOSTIC_ONLY/unreplicated/runaway/missing provenance`：visual audit only。

当前 data-driven SNN candidate 不因 RNN alignment 好看而升级。

### 10.2 Frozen mappings 与 comparisons

在 alignment 计算前冻结：

- RNN↔SNN geometry/contact/tissue mapping；
- mode sign；
- core/susceptibility definition；
- phase/tau aggregation；
- same-patient 和 cross-patient null；
- spatial autocorrelation/shaft null。

比较：

```text
RNN future-field response <-> SNN mode field
abs(RNN patch field)      <-> SNN core/susceptibility field
RNN progress response     <-> SNN propagation field
```

同患者减 geometry-mapped other-patient 为 patient-specific metric。共享患者数据/几何，因此只称
cross-model convergence，不称 independent replication。

### 10.3 C6 status

输出以下之一：

```text
SNN_ALIGNMENT_COHORT_ELIGIBLE
SNN_ALIGNMENT_CASE_SERIES_ONLY
SNN_ALIGNMENT_NOT_IDENTIFIABLE
```

SNN 不可用不影响 C1–C5，也不阻止 early-ictal exploratory phase。

## 11. Milestone I：frozen early-ictal exploratory alignment

### 11.1 最终解封条件

只有 RNN response/patch fields、data/SNN mappings、phase/tau contracts、所有 null maps 和 hashes 全部冻结，
才读取既有 17 patients / 167 seizures 的 clinical-onset 后 0–10 s、1–150 Hz broadband energy values。

### 11.2 预注册评分

- progress/field functional response vs early-ictal signed field correspondence；
- seizure→patient patient-first aggregation；
- synchronized all-contact primary null；
- geometry-eligible shaft/distance/variogram sensitivities；
- cross-patient identity sensitivity；
- 不允许 best-axis/best-phase oracle；
- 若做 omnibus selection，每个 null draw 必须完整重选并明确标为 exploratory。

target 已在项目历史中看过，C7 永远是 locked internal exploratory，不升 confirmatory，也不改变 C1–C6。

## 12. Milestone J：统计、claim ladder 与 stop logic

### 12.1 聚合顺序

```text
control draws within reference state
-> event
-> phase/tau
-> seed within arm/fit
-> arm-specific result
-> fit within patient
-> patient-level inference
```

event、step、seed、arm、control、patch、`own_a/own_b` 都不作为独立 cohort samples。

### 12.2 Claim families

| Claim | Primary question | Status |
|---|---|---|
| C1 | progress + continuous field geometry/emergence | Holm family |
| C2 | tangent transport + transverse contraction | Holm family |
| C3 | progress/field selective perturbation | two-test Holm |
| C4 | topology convergence | secondary family |
| C5 | data-field alignment + identity null | Holm family |
| C6 | SNN convergence | eligibility-dependent |
| C7 | early-ictal alignment | exploratory only |

每一层独立输出 `SUPPORTED/UNSUPPORTED/NOT_IDENTIFIABLE/NOT_ELIGIBLE`，不以一层的结果决定另一层是否运行。

C1/C3 中所有 future-field endpoints 额外按 `generic all-identifiable` 与 `canonical A/B shared-only` 两层
裁定；两层均按 patient 聚合，`own_a/own_b` 不作为独立样本。

### 12.3 Denominators

每个 branch 报 scheduled/eligible/completed patients、fits、events、states、controls 和 exclusions。不设置跨
branches 通用的患者数阈值；只要 estimand 可计算就输出 effect、exact/permutation uncertainty 和实际分母。
claim 强度由 CI、source eligibility、patient coverage 与 estimand 可辨识性共同裁定，不能用 nominal `P`
把原本只有 case-series/diagnostic 资格的 source 升级。

### 12.4 仅两个 hard gates

```text
E0: input/replay/hash/target-seal/resource engineering integrity
N0: per-branch local support/finite state/decoder numerical validity
```

E0 阻断对应执行阶段；N0 只排除单个 state/dose/control/patch。没有 G1→G4 科学 stop tree。

## 13. Figure、source data 与 closeout

### 13.1 Candidate figure contract

只有 source tables 完整后才画图，建议 panel 顺序：

```text
A full-state RNN and two-coordinate hypothesis
B raw-state progress/future-field geometry
C future-field emergence: real order vs C-suffix
D tangent transport and transverse response
E 2x2 perturbation response matrix
F finite-time patient-specific contact response fields
G data identity-null alignment
H SNN convergence by eligibility tier
I early-ictal exploratory alignment
```

若 SNN 不 eligible，Panel H 显示 eligibility/status，不以单患者漂亮图替代 cohort inference。该图仅为 candidate；
不得自动占用主文 Figure 6 或修改 semantic registry。

### 13.2 Source data

每个 panel 必须能从 `COHORT_PATIENT_TABLE.csv` 和 panel-specific source table 重建，含：

- planned/eligible/completed denominator；
- raw patient effects；
- null draws/summary；
- exclusion/status reason；
- input hashes；
- exact plotting filters。

图生成后才写中文 `figures/README.md`，逐图 2–4 句并以 `**关注点**：` 收尾。PNG/PDF/SVG 做同状态目视
核对，machine audit 不能替代视觉验收。

### 13.3 Closeout

`CLAIM_LADDER_ADJUDICATION.json` 必须逐 C1–C7 写：

```text
status
estimand
denominator
effect/CI/P_adjusted
null/control family
allowed language
forbidden language
source artifacts
```

`CLOSEOUT_AUDIT.json` 分开裁定：

```text
engineering_complete
scientific_experiments_complete
claim_support_level
snn_input_eligibility
early_ictal_status
figure_visual_acceptance
main_registry_status
```

最终不输出一个吞并所有结果的 PASS/FAIL。

## 14. 必须实现的测试

1. 630-cell resolver 与 531+99 provenance；
2. checkpoint/config/split/H/node-mask/contact/decoder hashes；
3. full-state `q` clone 和 `same h, different r/k` decoder counterexample；
4. teacher-forced hidden/logit/STOP/size replay；
5. closed-loop 不读取真实 future suffix/set size；
6. event-first/phase-balanced weights；
7. `K<2` phase undefined handling；
8. raw-state primary/residual sensitivity 路由；
9. `s_step` primary/`s_contact` sensitivity；
10. future-field axis 只读取 train events；
11. heldout `u_e` 不进入 RNN input；
12. train-only field sign freeze；
13. O/P/PF/PF-null capacity matching；
14. heldout 不重选 PCA/spline/phase/sign；
15. fit-specific node-space guard；
16. Jacobian-vector product 使用相同真实 next input；
17. raw-metric tangent/field backprojection；
18. local normal basis 来自 conditional residual states；
19. reference state manifest 不按 effect 选择；
20. local perturbation 保持 `r/k`；
21. transplantation pair 使用冻结 observable matching；
22. conditional support、node bounds 和 fail-without-clipping；
23. `tau=0` 不进入 finite-time endpoint；
24. open-loop branches 使用相同 future inputs；
25. absorbing STOP behavior；
26. 2×2 response orientation 和 denominator；
27. immediate-output/gain 是 sensitivity covariates；
28. cross-patient mapping 不读取 field values；
29. patch coarsening 只用 geometry/resource metadata；
30. SNN values 在 RNN/data fields freeze 前不可读；
31. SNN eligibility 不由 alignment result 升级；
32. SNN mode/core/mapping sign 在 alignment 前冻结；
33. early-ictal values 在所有 target-free artifacts/nulls freeze 前不可读；
34. patient aggregation 不重复 event/seed/arm/fit；
35. parameter hashes before/after 一致；
36. figures/source tables/claim ladder 一一对应。

## 15. 执行 DAG 与预计资源

### 15.1 DAG

```text
Phase 0 parent/input lock
  -> E0 replay/resource preflight
  -> Pass 1 streaming statistics
  -> Goal 5.2A geometry
  -> Goal 5.2A dynamical transport
  -> freeze reference states/controls
  -> Pass 2 Goal 5.2B perturbations
  -> Goal 5.2C response fields/data identity/patches
  -> SNN eligibility adjudication and eligible alignment
  -> frozen early-ictal exploratory scoring
  -> C1-C7 adjudication, figures, visual QA, closeout
```

箭头表示输入冻结依赖，不表示前一节点必须统计显著。

### 15.2 分阶段资源

正式运行前用 1 个大 fit × 5 arms × 3 seeds 做 engineering sentinel，实测：

- Pass 1 seconds/event-step；
- JVP seconds/state；
- Pass 2 branches/state；
- patch branches/fit；
- peak RAM/VRAM；
- output bytes/unit。

随后将预算写入 `RESOURCE_BUDGET.json`。资源不足时允许：

1. 多遍 deterministic replay 代替全量缓存；
2. 减少并行度；
3. 使用预冻结 geometry-only patch coarsening；
4. 将 optional diagnostic tau/dose 分批运行。

不允许按中间科学结果删患者、删 arm、删 axis 或只保留正向 phase。

## 16. 审阅后才允许执行的命令顺序

本版只交付 spec/plan，不执行下列命令。用户审阅授权后再依次：

```bash
pytest -q tests/test_topic5_latent_landscape_v0_2.py tests/test_topic5_latent_landscape_pipeline_v0_2.py
python scripts/audit_topic5_latent_landscape_inputs_v0_2.py --write
python scripts/stream_topic5_latent_system_id_v0_2.py --sentinel
python scripts/fit_topic5_latent_geometry_v0_2.py --sentinel
python scripts/analyse_topic5_latent_transport_v0_2.py --sentinel
python scripts/freeze_topic5_latent_reference_states_v0_2.py --sentinel
python scripts/run_topic5_axis_perturbations_v0_2.py --sentinel
python scripts/build_topic5_spatial_response_fields_v0_2.py --sentinel
python scripts/audit_topic5_latent_landscape_closeout_v0_2.py --stage sentinel
```

sentinel 工程验收后，才使用 manifest scheduler 扩展 630 cells；SNN 与 early-ictal 脚本必须保持独立解封步骤。

## 17. 本计划相对旧版的关键变化

- 从 progress-only gate 改为 progress + continuous future-field 双坐标；
- raw hidden primary，residualization 降为 sensitivity；
- 增加 future-field emergence、tangent transport、transverse contraction；
- perturbation endpoint 从同轴 persistence 改为独立 finite-time contact field；
- 增加 local field-axis intervention、empirical hidden chord 和 2×2 response matrix；
- controls 从超级匹配单族改为多种含义明确的 families；
- 增加 spatial patch、cross-patient identity null 和 eligibility-aware SNN convergence；
- early-ictal 保留为最后解封的 locked exploratory；
- 两遍 extraction 取代全量 trajectory archive；
- 删除所有统计驱动的顺序停止条件，只保留 E0/N0。
