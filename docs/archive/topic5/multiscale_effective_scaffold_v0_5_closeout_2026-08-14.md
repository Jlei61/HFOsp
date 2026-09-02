# Topic 5.1 v0.5 患者特异多尺度有效传播 scaffold：收口报告

日期：2026-08-14  
状态：**SCIENTIFIC CLOSEOUT COMPLETE。A–H 全流程、17/167 locked internal benchmark、
Figure 6、source-data export 与 machine closeout 均已完成；仅余 commit/push 和主线 figure registry 集成。**

## 0. 最终裁定

本轮可以科学收口，不再补模型、拓扑、seed、target endpoint 或 post-hoc subgroup。最终安全结论为：

> 患者内真实 prefix–suffix association 为 heldout 间期序列预测提供稳定信息，冻结 recurrent models
> 也能生成带患者空间结构的间期传播场；但当前数据没有辨识出 task-selected nonlocal topology 的
> 普遍优势，也没有支持 shortcut-specific early-ictal cross-state contribution。

early-ictal 端只能写为 all-contact null 下的正向但未支持方向，而且该方向在严格空间 null 可辨识
子集中不稳。任何 `L3-C-suffix` cross-state 表述必须同时给出 Holm 后 `P=0.0574-0.2295`，不能只报
原始 `P=0.01913`。本报告 primary 数值统一以 `INTERICTAL_V0_5_SUMMARY.json` 与
`FINAL_CLAIM_ADJUDICATION.json` 为准；后续 fixed sensitivity 只作 sensitivity，不替换主记录。

## 1. 这一轮在回答什么

本轮不寻找一张可被解释成真实 connectome 的 RNN 矩阵。它检验的是：在同一个覆盖组织平面的
局部 recurrent backbone 上，由患者有序间期事件选择出的少量 nonlocal effective shortcuts，是否
比宏观图统计完全匹配、但 source–target pairing 被打乱并从头训练的 `L2m` 更能解释 distal
propagation；而且这种增益是否随患者偏离局部传播的程度 `J_lat` 增大。

唯一 target-free primary 是：

\[
\rho_S\left(J_p^{lat},\,M^{II,distal}_{p,L3}-M^{II,distal}_{p,L2m}\right)>0.
\]

后续 early-ictal 部分只问冻结模型场与同患者 0–10 s、1–150 Hz broadband energy field 的空间对应，
不把 broadband energy 当作 contact arrival、临床 recruitment 或 ictal core 招募次序。

## 2. 分母、训练与 target seal

- parent：34 位 masked-rank K=2 患者；
- full-tissue spatial：28 位患者、42 fits、每 fit 6–52 个 exact joint contacts；
- early-ictal routing：17 位患者、167 次 seizures；
- formal training：531/531 units；
- arms：L0 93、L1 93、L2m 126、L3 93、C-suffix 126；
- 531/531 满足冻结的 validation early-stop 收敛口径，0 ceiling，0 nonfinite，0 unresolved OOM；
- 所有 best checkpoints 均位于 topology/mask freeze 之后；
- 输入 rank 来自 masked-rank `dataset_v0_4` 的 `event_group_ids`，不是 legacy `lagPatRank`；
  42 fits、28 位 spatial patients 的缓存逐位重建检查要求：原始 participation mask 零不一致、
  cache 零不一致、所有事件 tie-group 从 0 稠密编号，未参与触点固定为 `-1`；
- Stage A–F 运行空间中 early-ictal energy 文件物理不可见；target-free 计算期间没有 unseal
  authorization 或 target read。
- parent v0.3 已由远端 annotated tag `topic5-lbss-full-tissue-v0.3-closeout` 固定到
  `bd9d86217eb6bed013661b0f6d8aa8f397f6c986`；原 `CLOSEOUT_AUDIT.json` 为 `PASS`、0 errors，
  其 SHA256 与 tag provenance 均另存于 `PARENT_V0_3_PROVENANCE.json`。

这里的 full-tissue 不是“只在观测触点附近放 RNN 节点”：42/42 fits 的 tissue nodes 均多于
joint contacts；每 fit 有 16–318 个 `H` 列全零、不能被 contact 直接注入或读出的 latent nodes，
zero-H fraction 为 0.225–0.919。所有 H-supported nodes 仍处于同一个强连通分量。

这里对 zero-H nodes 的结论限于 **diagnostic engagement**：既有逐 rank clamp 会使 heldout NLL
变差，但该操作不是 matched lesion，也没有重新训练补偿，因此不能写成这些 nodes 对任务具有已证明的
必要性。

L2m 也不是对训练后 L3 做一次 frozen rewiring：126/126 L2m units 使用 macro-matched random
nonlocal mask 从头优化，并逐单元精确复制对应 L3 初始化 added weights 的 multiset。Free rollout
只读取真实第一 rank；共享 size head 只在 interictal train decisions 拟合、由 validation 选择，STOP
先于 contact selection，未来真实 set size 从不进入 decoder。

工程验收需保留一个边界：`PHASE_1_EXECUTION.csv` 与 `PHASE_2_EXECUTION.csv` 完整记录了 531 个
单元的完成状态、return code 和 attempt，因此可以确认 0 unresolved OOM、0 unresolved failure；
但这两张执行表没有保存逐单元 peak VRAM。当前只能用运行时 worker smoke、并发上限和实际
`nvidia-smi` 监测证明本轮采用了显存保护，不能声称“531 个单元均具有可追溯的峰值显存遥测”。
这是一项工程证据缺口，不改变训练产物，也不值得为补遥测而重跑 531 个模型。

完整 `runner.log` 还保留了正式冻结执行之前的一次 fail-closed launch：206 个单元累计出现 401 次
`KeyError: 'n_contacts'`，原因是旧训练器期待 legacy provenance key；修复 cache/provenance 合同后，
最终冻结的 531-unit phase execution 全部在自身 attempt 0 完成，0 non-zero return code、0 OOM。
`TRAINING_EXECUTION_HISTORY_AUDIT.json/.csv` 将这两层事实分开保存。因此只能写“最终正式执行
531/531 成功且无 OOM”，不能写“整个开发过程从未出现失败”。

531 而不是早期草案中的 321，是 full-parent census 与终审后严格重训/复用规则的确定结果，不是
看结果后扩展模型：31 个 mandatory fits × 5 arms × 3 seeds，加 11 个 exact-reuse fits 的
C-suffix/L2m × 3 seeds。

## 3. Target-free 结果

### 3.1 真实 suffix association 有稳定总体信息

L3 相对 split-matched、跨事件 suffix reassignment：

- all-transition contact NLL 增益中位 `+0.023683 nats`；
- 24/28 患者为正；
- 单侧 patient-level paired test `P=3.159e-5`。

但 distal-only 增量中位约 `+0.00167 nats`，14/28 为正，`P=0.407`。因此允许写：

> 真实 prefix–suffix association 为间期序列预测提供稳定的总体信息。

不能写：

> 真实顺序特异解释了 distal jumps。

### 3.2 Task-selected nonlocal topology 的特异优势未确认

这里的 primary `J_lat` 采用 **event-mean sparse exceedance burden**。原登记的 event-median
burden 在 28/28 患者中精确为零，因此在任何正式 RNN 训练和 early-ictal target 解封之前，按
`J_ESTIMAND_PREFREEZE_REPAIR.json` 做了预冻结 degeneracy repair；原 event-median、10 个连续时间
block 的 median 以及 nonzero-event fraction 均保留为 sensitivity。该修订没有读取模型结果或发作
target，但不能包装成未经修订的原始 preregistered estimand。

- L3−L2m distal gain 中位 `-0.005091 nats`，11/28 正向，`P=0.899`；
- primary interaction：Spearman `rho=0.168041`，单侧 permutation `P=0.194108`，bootstrap
  95% CI `[-0.272649, 0.573713]`；
- 去除 6–7-contact patients：n=21，`rho=0.032`，`P=0.445`；
- 去除最高 J 患者：n=27，`rho=0.282`，`P=0.075`；
- 去除两位近一维几何：n=26，`rho=0.148`，`P=0.233`。

固定模型、严格 recording-block-heldout **evaluation sensitivity** 方向一致：保留 40/42 fits、
115,834/133,482 test events；L3−suffix 的总体增益仍为 `+0.022519 nats`（22/27，`P=8.12e-5`），
而 L3−L2m distal 和 `J × gain` interaction 仍未确认。该分析只在原冻结模型上评价严格 unseen-block
test subset，**不是 block-heldout refit 或新的 block-CV 训练确认**。

因此当前最准确的 target-free 结论是：

> 患者内有序间期序列可被 recurrent computation 学习，但当前数据没有显示由任务选择的具体
> nonlocal pairing 普遍优于宏观图统计匹配的随机 nonlocal scaffold，也没有显示这种优势随
> cross-fitted nonlocality J 稳定增强。

### 3.3 Prefix template 与 model-generated interictal fields

- RNN 相对 train-only prefix-template 的 NLL advantage 中位 `+0.316845 nats`；
- 该优势不随 prefix uncertainty 稳定增强：患者级 rho 中位 `0.0132`，`P=0.416`；
- L3 model-generated canonical interictal field 与 empirical interictal field 的患者级相关中位
  `0.278`，22/28 为正；seed-removed 中位 `0.189`，21/28 为正。

这说明 RNN 不是单纯复制一个 prefix template，且其 rollout 可以生成带患者空间结构的间期场；
但目前不支持把该场的 nonlocal edge identity 当作可辨识的真实解剖连接。

### 3.4 当前机制边界

- synthetic functional-class detectability 仅有限：3 个真实几何中，L3 distal 优于 L0 为 1/3，
  L3 attenuation 产生预期 distal harm 为 2/3；
- L1/L3 candidate opportunity 在 15/42 fits 严重不平衡，只有 16/28 患者全部 fits 可承担严格
  L3−L1 机制比较；
- shared-fit mode-flow 的 same-vs-cross distal selectivity 中位 `-0.001208 nats`，`P=0.452`；
- mode-flow matched-random control 0/14 患者满足匹配合同，状态应写
  `NOT_IDENTIFIABLE`，不能写成阴性；
- heldout trajectory finite-horizon gain 在 L3、L2m、suffix 间相近，没有明显的整体 gain-scale
  混淆。

这里必须再限定一次分母：`own_a/own_b` 两个 geometry fits 实际都训练该患者的全部合格事件，
并非按 A/B event family 过滤。它们各自产生一个 geometry-specific all-event candidate field；只有
14 位 shared-fit 患者能在**同一张 RNN**内比较两个 train-only modes 的 route usage。因此
same-vs-cross mode-flow 只对 shared-fit n=14 可辨识，不能外推到全部 28 人。该事实已写入
`MECHANISM_SCOPE_ADJUDICATION.json`，并纠正早期 incident 记录中过时的“event-family filtered”措辞。

## 4. Stage F 与 locked early-ictal benchmark 合同

Stage F 与 locked early-ictal 的最终数值统一写入 §8。执行顺序固定为先冻结：

1. L1/L2m/L3 各自 active added edges 的 dose attenuation；
2. L3 内实际 active local edges 的 matched-local attenuation；
3. gain-adjusted sensitivity fields；
4. intact、template、mixture、suffix、attenuated 与 gain-adjusted fields；
5. synchronized all-contact primary null 与预定义空间 robustness null maps。

只有 `STAGE_F_TARGET_FREE_COMPLETE.json` 和 `TARGET_UNSEAL_AUTHORIZATION.json` 顺序成立后，
scorer 才可读取 17 人/167 seizures target。最终验收覆盖：

- signed best-mode Spearman oracle repertoire correspondence；
- train-prevalence mixture non-oracle correspondence；
- primary `rho(J, C_L3-C_L2m)`；
- L3 与 all-contact channel shuffle；
- attenuation AUC 与 cross-state dose response；
- empirical interictal field reference 和 spatial robustness nulls。

解封前审计还识别并修正了一个只影响 non-oracle sensitivity 的聚合语义：对 non-collinear
患者，oracle A/B 必须继续保留为 `own_a` 与 `own_b` 两个 geometry candidates，不能在 maxAB 前
平均；但 train-prevalence mixture 不能把两个 **all-event geometry fields** 直接按 mode prevalence
加权。最终实现从 `own_a` 中仅聚合与 A 对齐的 train-only mode、从 `own_b` 中仅聚合与 B 对齐的
train-only mode，再按 train-only prevalence 形成 mixture。修复覆盖 14 位患者 × 5 arms = 70 个
patient-arm fields；所有 checkpoint、oracle A/B vectors、target-free primary、attenuation 与 gain
fields 均不变，且整个修复在物理 embargo 内完成。最终 PASS 必须由
`TRAIN_PREVALENCE_MIXTURE_REPAIR_COMPLETE.json` 证明 target access=false、70/70 完整且 oracle
A/B hash 未变。

## 5. 解封前允许与禁止的论文表述

解封前已允许：

> Patient-specific interictal rank sequences contained robust prefix–suffix information that could be
> learned by recurrent networks and expressed as model-generated interictal fields.

解封前不允许写：

- task-selected nonlocal shortcuts 已被证明是患者真实白质 pathway；
- L3 普遍优于 matched random nonlocal topology；
- distal jumps 的患者越多，L3 优势越大；
- 在 Stage G 完成前宣称 early-ictal field 已迁移成功；
- v0.5 是独立外部确认。

## 6. 最终验收范围

- Stage F 504 个 arm-target attenuation units 与 126 个 matched-local searches 的完整性；
- target authorization、unlock 与 17/167 scoring 的先后关系；
- Figure 6 最终 PNG/PDF/SVG、source tables 与逐 panel QA；
- full closeout audit 与最终 `git diff --check`；
- 文档/index/figure registry 的最终同步；
- 用户终审后才允许 commit/push。

## 6.1 A–H 完成证据矩阵

最终验收不以单个 `PASS` marker 代替证据，而按下表逐项核对：

| 阶段 | 要证明的内容 | 权威证据 | 当前状态 |
| --- | --- | --- | --- |
| A | 自动 full-parent census、28 人/42 fits、17 人/167 seizures routing | `FULL_PARENT_FIT_CENSUS.csv`、`FULL_PARENT_PATIENT_ATTRITION.csv`、`EARLY_ICTAL_ROUTING_METADATA.csv` | 已完成 |
| B | local strong connectivity、cache/data hash、train-only mode 与 suffix-null 合同 | `INPUT_CACHE_MANIFEST.json`、`TRAIN_ONLY_MODE_FIT_CENSUS.csv`、`SUFFIX_NULL_DESTRUCTION_AUDIT.csv` | 已完成 |
| C | L2m macro matching、candidate exposure、functional-class detectability | `L2M_GRAPH_CONTROL_MANIFEST.csv`、`CANDIDATE_*AUDIT.csv`、`functional_shortcut_detectability/` | 已完成；detectability 有限 |
| D | cross-fitted `J_lat`、不删除 local-wave-unsupported 患者 | `CROSSFIT_NONLOCALITY_*SUMMARY.csv`、`J_ESTIMAND_PREFREEZE_REPAIR.json` | 已完成 |
| E | 531-unit 正式训练、patient-first interictal inference、block-heldout sensitivity | `FORMAL_TRAINING_SCHEDULE.csv`、`INTERICTAL_*`、`PHASE_*_EXECUTION.csv` | 已完成 |
| F | mechanism、arm-specific attenuation、gain-adjusted fields 在 target 前冻结 | `STAGE_F_TARGET_FREE_COMPLETE.json` 与五套 field/metric manifests | 已完成 |
| G | authorization 后读取 17/167 broadband target，patient-first locked benchmark | `TARGET_UNSEAL_AUTHORIZATION.json`、`TARGET_UNLOCK_RECORD.json`、`EARLY_ICTAL_*` | 已完成 |
| H | Figure 6、source data、逐项 closeout audit、中文结论 | `FIGURE6_*`、`CLOSEOUT_AUDIT.json`、本文 §8 | 已完成 |

## 7. 解封前工程验收加固（2026-08-14）

在 target 仍物理封存时，已完成以下不改变模型、endpoint 或统计的收口准备：

- 相关测试固定为 4 个测试文件；冻结时最低基线为 124 项，本次最终运行因新增 3 项回归测试为
  **127/127 passed**。完整 stdout 与 SHA256 已写入
  `PREFINAL_RELATED_PYTEST.log` 和 `PREFINAL_RELATED_PYTEST_EVIDENCE.json`；
- 独立重算 `seed median -> fit -> patient` 的间期聚合，与正式表最大绝对误差不超过
  `1.14e-13`；
- 三套执行快照（formal training、Stage F、posttraining）当前 source hash 均无漂移；这里的
  不可变性只称 `hash-verified`，不称 filesystem read-only；
- 自动 target unseal guard 已替换为 fail-closed mixture-repair guard：Stage F 完成后先在 embargo
  内修复 non-oracle train-mixture、重新冻结 model-field manifest 与分析 metric hashes，之后才恢复
  主进程；任何一步失败都终止主进程而不是带错场继续评分；
- `SCORER_CONTRACT_PREFREEZE_REPAIR.json` 固定了非可辨识 mode、有限 null 分母、不可变首读记录、
  完整条件清单和 joint patient-label/spatial-null interaction 合同；
- scorer 的 coherent spatial-null 修复合法更新了 Panel E decision 后，静态审计发现 Figure finalizer
  manifest 仍指向修复前哈希。此时 target authorization 与 unlock record 均不存在，因此只重新运行
  `--freeze-contract` 刷新该输入哈希；finalizer source、scorer、endpoint、estimand 和模型场均未改变。
  `FIGURE6_FINALIZER_MANIFEST_REFRESH.json` 保存新旧 manifest hash 与 target-free 状态，避免该问题在
  解封后的最后制图步骤才 fail-closed；
- `CLOSEOUT_TOOLING_PREFREEZE_MANIFEST.json` 进一步冻结 guard、claim adjudicator、最终制图、
  source-data exporter、报告 finalizer、独立 closeout audit 和文档同步器，避免 target 解封后漂移；
- Figure 6 Panel C 已在解封前冻结为 28 人的 `true suffix vs split-matched reassigned suffix`，
  不再允许混入旧 34 人 contact-space RNN；Panel I 已冻结为 L2m/L3 heldout finite-horizon gain，
  不把 0/14 可匹配的 mode-flow random control 画成零效应；
- 最终 verifier 已升级为逐项重算：531-unit completeness、L2m matching、block-heldout sensitivity、
  target authorization/unlock 时序、五套 field/null manifest payload hash、间期/attenuation/gain/
  early-ictal patient-first 聚合、Figure source rows、Panel C/I contract、600-dpi raster 尺寸、单页
  PDF、prefreeze decision hashes 和持久化测试证据。
- target 解封后，verifier 会把 17 人/167 次发作的 derived vectors 按 contact name 逐值回算到
  `bb150_auc__*` 源 cache，并记录源 NPZ/JSON hash；同时要求源 metadata 为 line-noise-masked
  1–150 Hz、clinical-onset `[0,10] s`、mean baseline-robust-z energy。

Target 解封后的第一次 locked scorer 已读取 energy values，但在写出正式结果表之前，因清单校验器
错误要求每位患者都具备可选的 `L3_MATCHED_LOCAL` attenuation condition 而 fail-closed。Stage F
预冻结证据已明确：该 matched-local control 只有 17/28 患者可构造，early-ictal 队列中的
`epilepsiae_590` 与 `yuquan_xuxinyi` 均为零个合法 matched draws；所有 primary L3/L2m/J/intact
conditions 完整。修复只让 scorer 按 `ATTENUATED_FIELD_MANIFEST.csv` 中已冻结的逐患者可用条件进行
清单验证，没有补值、重建 field/null、改变 cohort、endpoint 或 primary estimand。原 authorization
未改，第一次失败及后续 recovery 均保留在 `diagnostic_archives/`，完整边界记录于
`TARGET_UNSEAL_ENGINEERING_AMENDMENT.json`。因此本轮仍是 locked internal follow-up，但不能声称
“target 解封后执行代码从未发生工程修复”。

最终 closeout 还暴露两项不影响科学产物的 verifier 假设：cross-fitted J 的 target-free 状态记录在
Stage-D marker 与冻结表 hash 中，而不是逐 CSV 行重复；相关测试数量也不应写死为恰好 124。
两项均只修正验收逻辑，未重算任何 endpoint，并在同一 amendment 中留痕。修复后独立 closeout audit
为 **50/50 PASS**。

Attenuation 的 rollout Spearman 只在至少 3 个 post-seed contacts 具有可比较顺序时定义。强衰减使
某些 rollout 不再满足该 denominator 时，`NaN` 被保留为“生成塌缩”，不能改写成零；但对应的
contact/local/distal NLL 必须保持有限。最终 verifier 会同时检查四档剂量、实际 draw 数、mask hash、
eligibility、NLL 有限性，并单独报告 rollout 不可定义比例。

这些检查只提高最终 PASS 的证据门槛，不改变任何正在运行的 Stage F 科学计算。由于每个 matched-local
单元要先构造 500 个候选匹配集，再对 16 个有效 draws × 4 档 attenuation 运行自由生成，Stage F 是
本轮最慢阶段。Attenuation 使用 8-worker 资源合同完成；gain-adjusted sensitivity 属于另一类负载：
每个 prefix 都要计算 exact finite-horizon spectral-norm SVD。8 个独立 CUDA 进程会争用同一个求解器并
降低总吞吐。在 target authorization 尚不存在、且 gain 最终 marker/结果表均未生成时，旧进程树被
停止；随后只把该步骤的进程并发从 8 降到 1 并重新启动。Exact estimator、每 split 冻结的 32 个
prefixes、126 对 L2m/L3 模型、seeds、endpoint 和 field construction 均未改变。最大 latent grid
（346 tissue nodes）的一 worker 诊断将 exact median G3 复现到 `1e-5`；诊断用 batched 实现没有进入
正式生产，正式计算仍使用原 scalar exact 路径。`STAGE_F_GAIN_WORKER_CONTENTION_REPAIR.json` 固定了
新旧 source hash、运行时观察和 target access=false。这是工程资源修复，不是超参数或 endpoint 调整。
Stage F 的 step log 使用 append 模式，因此文件中保留了重启前单 worker 试运行的 `10/20/30` 三条
进度记录；新进程随后从 `10` 重新计数。Gain producer 不写逐 pair partial table，只有 126/126 全部
完成才原子形成正式结果表和 completion marker，所以旧进度行不对应任何被聚合的半成品；最终完成度
只由 `GAIN_ADJUSTED_SENSITIVITY_COMPLETE.json` 与 252/56 行正式表验收，不能用累计日志行相加。

运行时进一步确认，自由生成是“冻结模型 + 真实第一 rank”的确定性函数，而每 fit 的大量 test events
只对应少量唯一第一 rank（中位 1,016.5 个 test events、10 个 unique starts）。因此在不减少任何
arm、seed、draw 或 dose 的前提下，增加了 target-free exact-equivalent hotfill：每个 unique start 只
生成一次，再逐事件展开回原 schema。启用前已在 E1146 正式 L3 seed0 的 1,492 个 heldout events 上
与原正式 rollout 逐事件比较，15 个 unique starts 展开后 1,492/1,492 完全一致、0 mismatch。最终
verifier 要求 hotfill producer hash、完整 cache 清单和这份 parity 证据同时成立；否则不能 PASS。

Stage F 的缓存交接另由 `STAGE_F_CACHE_HANDOFF_AUDIT.json` 独立审计：42 个 fits × 4 个
attenuation targets × 3 seeds 共 504 个 unit-target caches 全部存在且身份唯一，425 个由上述
exact-equivalent 去重生产器生成，79 个由原始 executor 完成；全部 cache 明示
`target_values_read=false`，不存在临时/半写文件，且交接审计时 target authorization 仍不存在。
去重生产器并未减少 arm、dose、matched-local draws 或统计行，只复用“同一冻结模型、同一首 rank
必然得到同一确定性 rollout”这一逐事件验证过的计算等价性；Stage F 汇总器随后从同一标准 cache
schema 读取完整 504 单元。原始长时 executor 仅在完整 cache grid 和原子 completion marker 已经形成
后停止，避免继续重复计算，不删除或覆盖任何已验收 cache。

## 8. 最终锁定结果（自动收口区）

> **读 §8.2 前必读**：该自动区逐个列出的单侧 `P` 全部是**未经多重校正的原始值**。spec §10.3 与
> plan §G 要求 early-ictal secondary/robustness 按预定义 claim family 报 Holm 或联合区间、
> 不为每个 endpoint 单独追星号，而冻结 adjudicator 只对两项 D2 family 做了 Holm。校正后的判定见
> §9.1（`SECONDARY_REPORTING_ADDENDUM.json`）。§8.2 中唯一 `P<0.05` 的次要项在任何族定义下都不成立。

<!-- FINAL_RESULTS_BEGIN -->
### 8.1 Stage F target-free 扰动与冻结

- 504/504 arm-target attenuation units 完成；matched-local control 的严格患者级可推断人数为
  `17`；
- 四类 attenuation AUC 的 inferential denominator：
  `L1=28`、`L2m=28`、
  `L3=28`、`L3-matched-local=17`；
- intact、template、mixture、attenuated、gain-adjusted fields 和 synchronized null maps 均在
  target authorization 前冻结；本节所有 perturbation fields 都没有读取 early-ictal values。

### 8.2 Locked internal early-ictal benchmark

目标固定为 17 位患者、167 次发作、clinical onset 后 0–10 s、1–150 Hz broadband energy；
这是项目内已经看过 target 后锁定的 mechanistic follow-up，不是 independent confirmation。

- D1：L3 canonical-full signed best-mode field 相对 synchronized all-contact null：中位 `0.2118`，11/17 正向，单侧患者级 `P=0.2019`；
- D2-direct：L3−L2m seed-removed signed field correspondence：中位 `0`，4/17 正向，单侧患者级 `P=0.5781`；
- D2-attenuation：削弱 L3 selected shortcuts 后 seed-removed concordance damage AUC：中位 `-0.007955`，6/17 正向，单侧患者级 `P=0.8511`；
- 非 oracle：train-prevalence mixture 相对 synchronized all-contact null：中位 `0.1786`，10/17 正向，单侧患者级 `P=0.153`；
- L3 相对 split-matched suffix reassignment 的 cross-state 增量：中位 `0.03571`，11/17 正向，单侧患者级 `P=0.01913`；
- train-only oracle template 相对 synchronized all-contact null：中位 `0.2`，10/17 正向，单侧患者级 `P=0.153`；

Primary early-ictal interaction 为 `rho=0.07653`，单侧 patient-label
permutation `P=0.3858`，coherent synchronized spatial-null
interaction `P=0.6843`；联合主判据取较大值
`P=0.6843`，bootstrap 95% CI
`[-0.4463, 0.6481]`。

以上两个 `J` interaction 使用的是正式训练和 target 解封前冻结的 event-mean sparse exceedance
burden。原 event-median estimand 在 28/28 患者中精确为零，已按
`J_ESTIMAND_PREFREEZE_REPAIR.json` 标记为退化 sensitivity；因此不能将本结果包装成未经修订的
原始 preregistered `J`。

### 8.3 预冻结 claim adjudication

- target-free `(L3−L2m) × J`：**未支持**；
- 总体 prefix–suffix information：**支持**；
- distal-specific suffix information：**未支持**；
- D1 cross-state field correspondence：**未支持**；
- early-ictal `(L3−L2m) × J`：**未支持**；
- D2 shortcut-specific cross-state contribution（两项 Holm family）：
  **未支持**。

允许表述必须以 `FINAL_CLAIM_ADJUDICATION.json` 为准。无论数值方向如何，都不能把 effective
scaffold 写成 anatomical/white-matter connectivity，也不能把 broadband energy field 写成 arrival
time 或 recruitment order。按预冻结决策树，下一条单一机制扩展为：
`E3_SMOOTH_SUSCEPTIBILITY`。
<!-- FINAL_RESULTS_END -->

## 9. 解封后 reporting-only 校正（2026-08-14 复审）

本节由 `scripts/report_topic5_multiscale_secondary_reporting_v0_5.py` 生成
`SECONDARY_REPORTING_ADDENDUM.json`，只重读已冻结的表，不重训、不重算 field/null、不改 estimand、
cohort 或 endpoint。三项校正的方向都是**单调收紧**，不可能制造出阳性，因此在 target 解封后执行
不构成 fishing。它补的是冻结 adjudicator 漏掉的三条 spec 条款。

### 9.1 spec §10.3：early-ictal secondary/robustness family 的 Holm 校正

冻结 adjudicator 只对两项 D2 family 做了 Holm，其余 10 个非 primary 终点在 §8.2 里都是原始单侧 `P`。
按 spec §10.3 补做 Holm，三种族定义并列报告（不事后挑对自己有利的一种）：

| 族定义 | m | `L3−C-suffix` 原始 P | Holm P | 该族是否有任一项成立 |
|---|---:|---:|---:|---|
| 报告口径：全部非 primary 终点，剔除已自带 Holm 的 D2 两项 | 10 | 0.01913 | **0.1913** | 否 |
| 最宽：全部 12 个非 primary 终点 | 12 | 0.01913 | **0.2295** | 否 |
| 最窄：3 个 cross-state arm 对比（vs L2m / vs C-suffix / vs train-only template） | 3 | 0.01913 | **0.0574** | 否 |

因此 `L3_minus_suffix_full_signed_oracle`（中位 `+0.0357`，11 正 3 负 3 并列）**在任何族定义下都不
成立**。它只能写成 hypothesis-generating 的方向性观察，并且必须与 Holm 后的 `P` 一起给出；
不得以 `P=0.019` 的形态单独呈现，也不得称其为"值得保留的次要阳性"。

补充核对：`EARLY_ICTAL_PER_PATIENT.csv` 能逐值复现该终点的患者级向量（中位数与 11/3/3 计数完全
一致）；`P` 的 `0.01764` 与冻结的 `0.01913` 之差仅来自 Wilcoxon 精确检验与带并列校正的正态近似之
别，两者同属 `~0.018–0.019`，不影响上述判定。以冻结值为准。

### 9.2 spec §9.5：四套几何合格 robustness spatial null 下的 D1

这四套零模型此前逐患者算了但从未汇总。每套只在通过对应 montage 几何 QC 的子集上可辨识
（spec §3.3），所以必须与**同一批患者**的 all-contact null 比，否则会把"零模型更严"和"患者子集不同"
混为一谈：

| Robustness null | n | 该 null 下 margin 中位 | P | 同批患者 all-contact margin 中位 | P |
|---|---:|---:|---:|---:|---:|
| within-shaft | 15 | +0.0731 | 0.244 | +0.1817 | 0.381 |
| distance-bin | 10 | +0.0662 | 0.348 | +0.1087 | 0.385 |
| spectral surrogate | 8 | −0.0677 | 0.473 | −0.0635 | 0.578 |
| variogram-matched | 5 | +0.0321 | 0.500 | +0.0357 | 0.500 |

结论有两层，必须分开写：

1. D1 在四套 robustness null 下**全部未支持**；
2. margin 从全队列的 `+0.212` 掉到 `+0.03…+0.07`（spectral 下转负），**主要来自可用患者子集不同**，
   不是零模型更严——同批患者的 all-contact margin 也同样掉下去。也就是说：D1 的正向方向由那批
   几何自由度不足、撑不起严格空间零模型的患者承载。这条比"正向趋势但未确认"更受限，
   必须一并写出。

### 9.3 spec §7.3：修复后 `J` 仍严重贴地，限制唯一 primary 检验的分辨率

`J` 从 event-median 修成 event-mean 的 pre-freeze repair 已在 §3.2 记录，但修完之后的分辨率没写：

| 队列 | n | 恰好为 0 的患者 | 不同取值个数 | 中位 | 最大 | 小于最大值 1% 的比例 |
|---|---:|---:|---:|---:|---:|---:|
| target-free primary | 28 | **10** | 19 | 0.00046 | 0.600 | 71.4% |
| locked early-ictal | 17 | 3 | 15 | 0.00471 | 0.600 | 58.8% |

整个 target-free family 的唯一 primary 就是对 `J` 的 Spearman 秩相关。28 人里有 10 人恰好为零、
共享同一个 mid-rank，另有 20/28 落在最大值的 1% 以下，只有 19 个不同取值。因此
`rho=0.168, P=0.194` 这个阴性**受 moderator 分辨率上限约束**，只能写"在当前 `J` 的分辨率下没看到
这种耦合"，不得写成"不存在这种耦合"，也不得据此说 nonlocality 与 shortcut 收益无关。
（同一方向的旁证：去掉唯一的高 `J` 患者后 `rho` 反而升到 0.282、`P=0.075`。）

## 10. Figure 6 视觉修复（2026-08-14 复审，render-only）

r2 finalizer 与 panel producer `plot_topic5_figure6_multiscale_scaffold_v0_5.py` 都被哈希锁定
（前者自校验、后者由 `POSTTRAINING_PIPELINE_SNAPSHOT.json` 固定），因此两者保持逐字节不变；全部修复
由新增的 `scripts/finalize_topic5_figure6_multiscale_scaffold_v0_5_r3.py` 在运行时组合上去。
未改动任何 panel 的 estimand、分母、cohort、null、患者数、坐标轴范围或显著性星号。

| # | 原问题 | 后果 | 修复 |
|---|---|---|---|
| 1 | Panel D 第三条 colorbar 的刻度数字压在 Panel E 的 `Signed field correlation` 轴标签上，且离能量场很远 | 读者看不清 E 的 y 轴，也判不出该色标属于哪张场 | 右侧留 gutter，色标标题写 `Energy z` |
| 2 | Panel D 三张场图都没有 x 轴名 | 违反 Topic 5 场图样式锁（Fig3-C 要求每个 panel 写轴名） | 三张均写 `Propagation axis (mm)` |
| 3 | Panel D 的 TA/TB 用了两条一模一样的 Early/Late 色标 | 同一构件画两遍（CLAUDE.md §7 + 样式锁"一个共享 colorbar"） | 合并为一条共享 `RNN rank` 色标 |
| 4 | Panel A 只画了 107 条 task-selected shortcut 中最强的 3 条，图上无说明 | 读者会以为模型只有 3 条长程边 | legend 改为 `Selected shortcut (3 of 107 drawn)` |
| 5 | Panel A 左右两条竖条无任何标签 | 看起来像杂散 colorbar，无法解读 | 标为 `Input rank` / `Generated rank` |
| 6 | Panel B `Generated` 30 列里只有 TA 10 条、TB 9 条互不相同 | 自由生成对「冻结模型 + 真实第一 rank」是确定性的，而这些事件只有约 10 个不同首 rank；读者会当成 30 次独立复现 | 该列下方标注真实的 distinct 条数 |
| 7 | Panel E 左/中两格视觉分离很强，但患者级 `P=0.202 / 0.153` | 只看图会读成阳性 | 加 `n.s. (P=0.20)` / `n.s. (P=0.15)`；仍不给任何星号，符合冻结的 Panel E 规则 |
| 8 | Panel E 最右格 6 个刻度互相压字、x 轴名跑出画布 | 主统计那一格反而最不可读 | 保留同一坐标变换与范围，只画 4 个刻度、轴名缩短为 `Nonlocality J (sqrt scale)` |
| 9 | Panel G 的 `Local` 指 L0 纯局部模型臂，Panel H 的 `Local` 指 L3 内部匹配的局部 backbone 边 | 相邻两格同词异义 | H 改为 `Local in L3` |
| 10 | Panel G y 轴 `Selected benefit (nats)` 未说明只是 distal 对比 | 全 transition 的 `L3−L2m` 是 `+0.00096`、distal 是 `−0.00509`，符号相反，会被误读为总体收益 | y 轴改 `Selected benefit, distal transitions (nats)`，x 刻度改 `vs Local / vs Nearby / vs Matched` |

重跑后独立 closeout audit 仍为 **50/50 PASS**（含 `hash_verified_execution_snapshots`、
`figure_prefreeze_contract`、`figure_package`），4 个相关测试文件仍为 **127 passed**。

**仍未修、留给用户判断的一项**：Fig3-C 样式锁要求间期侧色标标题为 `ranks` 并显示真实 propagation
rank。Panel D 的 TA/TB 是 RNN 生成的归一化 earlyness（不是整数 rank），标成整数 rank 会失真，
因此保留 `Early/Late` 端点标注并在此记录该偏离。

Figure package 的同状态 PNG/PDF/SVG 与 source tables 已通过视觉和 machine audit，但当前主线
`docs/paper_figure_registry.md` 尚未为该包分配 canonical `asset_id/paper_slot/status/canonical_path`。
因此当前身份固定为 `CANDIDATE_PENDING_MAIN_REGISTRY_INTEGRATION`；这不影响 Topic 5.1 科学收口，
但在主线登记完成前不能仅凭 `fig6_multiscale_scaffold_v0_5` 目录名将其称为 canonical Figure 6。
