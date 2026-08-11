# Topic 4 rev9-L: mode realizability and learnability audit

**状态：** `REV9L_DEVELOPMENT_AUDIT_COMPLETE`
**日期：** 2026-08-10
**上游：** rev8.1 patient-training KMeans fit；rev9 frozen-field node-edge factorial
**结果目录：** `results/topic4_sef_hfo/data_driven_core_field_rev9_learnability/`

## 1. 科学问题和边界

rev9 只检验了一个由 local-response calibration 选出的 scalar edge 参数，且没有按患者双模式目标重新训练 edge。
因此 Edge-only 的阴性结果不能写成“患者模式学不到”。rev9-L 依次区分四件事：

1. **realizability：** 当前 SNN family 中是否存在同时表达 mode A/B 的参数；
2. **shared-substrate capacity：** 同一参数能否跨 network realizations 保持两种模式；
3. **optimization learnability：** 好参数存在时，当前 objective 和 optimizer 能否找到；
4. **identifiability：** 输出相似时，恢复的 node/edge 参数是否稳定且唯一。

本分支是 development/oracle audit。它不读取旧 patient held-out 分数，不产生 patient-generalization claim，不修改
Z/M/adaptation 或 slow lifecycle 方程，也不把 forced source 当成自发间期事件。

## 2. 执行纪律

只保留工程硬错误：输入 hash/shape 不符、forced-spike 注入不符合冻结合同、连接结构守恒被破坏、状态非有限或 producer
失败。event support、模式距离、OOD、runaway、跨 seed 离散和 optimizer miss 都作为结果连续报告，不设置层层 blocker。

每一层只在科学上需要时进入下一层：

```text
L0 objective replay
  -> L1 forced-initiation capacity
     -> scalar edge propagation 不足时才做 L2 relaxed edge
        -> 已知好解或 oracle 比较时才做 L3 optimizer/shared audit
           -> forced shared capacity 成立后才做 L4 spontaneous confirmation
```

长任务统一使用 `systemd-run --user -> /usr/bin/nohup`，保存 `.log/.status`，结束时发送桌面通知。正式 producer 先提交再运行。

## 3. L0：availability-aware objective replay

### 3.1 现存数据能回答什么

rev8.1 checkpoint 保存了 48 个候选的全局距离、KMeans prototype similarity、cluster support 和事件数，但没有保存每个候选
的逐事件 recruitment、precedence、rank matrix 或 event cloud。因此 L0 分两层：

- **48-candidate replay：** 只重算历史产物真实保留的 `D_global`、mode A/B prototype loss、support 和 Pareto 排序；
- **full-descriptor replay：** 只对保存了逐事件数组的冻结 final candidate 和 rev9 arms 计算 recruitment、precedence、profile
  与 event distribution。缺失值显式标记 `not_retained`，不得由 summary 反推或补造。

历史 mode proxy 定义为：

```text
D_A = (1-rho_A)/2
D_B = (1-rho_B)/2
D_weak(tau) = tau*log((exp(D_A/tau)+exp(D_B/tau))/2), tau=0.25
```

`D_weak` 只用于诊断旧 objective 是否保护最弱模式，不替代后续完整 mode-conditioned objective。

### 3.2 训练候选与 selection 候选分开

48 个 fit candidates 的 Pareto front 只说明训练库中是否有值得重放的候选。只有已经在 selection network seeds 上评估过的
三个候选，才可用于判断 selection 是否漏选；fit-only candidate 不能因训练分数好就标记 `OLD_OBJECTIVE_OR_SELECTION_MISS`。

### 3.3 Patient training reliability

从 frozen recording-block split 重建曲线，只读取 training index。用冻结的 full-training KMeans 标签定义 A/B；每个 block 的
mode prototype 与 leave-one-block-out prototype 比较，并报告：

- 每个 mode 的 block-to-complement Spearman 分布和 95% bootstrap interval；
- 每个 block 的 mode proportion 变化；
- frozen embedding 中 within-block 与 between-block dispersion；
- 每项实际 block/event support。

这些是 target reliability 诊断，不是 held-out ceiling，也不是新的患者盲检。

## 4. L1：forced-initiation propagation capacity

### 4.1 要分离的量

自发输出分解为 `p(y)=integral p(y|s,theta_prop)p(s|theta_ignite)ds`。L1 固定 source packet，直接测
`p(y|s,theta_prop)`，从而把“点不着”和“传不对”分开。

Primary source mapping 使用当前 onset association：

- patient mode A <- Gaussian component 2；
- patient mode B <- Gaussian component 1。

Secondary scan 包括 components 1/2/3 和三个预冻结 matched off-field sites。component source 按该分量的 raw Gaussian
contribution 取前 N 个 E neurons；不能直接取 responsibility，因为远离全部 Gaussian 的尾部也可能得到虚高相对责任。
off-field control 按到冻结 control center 的距离取最近 N 个 E neurons。两者 packet 数相同，位置不按输出结果移动。

### 4.2 Forced packet

在 `t=100 ms` 对 source set 注入一次 deterministic spike packet，之后完全回到原 SNN 方程。Node canary 只比较
`0.5%/1%/2% * N_E`，冻结最小的“产生可读 downstream recruitment 且没有大面积非有限/runaway”的 packet size。
所有 arm/seed 共用 source neuron identity、时刻和随机数流；sham 只是不注入 packet。

正式四臂为 Null、Node、Edge、Node+Edge，运行窗 400 ms。primary rank readout 固定在 `[100,250] ms`，先计算
`clip(contact_envelope_forced-contact_envelope_sham,0,infinity)`，再走同一个 `read_event`；这样 propagation capacity 不以
detector 是否形成完整 returned event 为前提。raw forced trajectory 上与 trigger 重叠、onset latency 不超过 40 ms 的 returned
event 作为 secondary duration/return endpoint；后发自发事件不能冒充 triggered event。本层不用 KMeans 发现 mode，也不优化
mode proportion；source identity 决定比较的 patient target。

Instrument canary 已在 seeds 1001-1003 完成：三个候选 packet size 均对 component 1/2 产生可读 rank curve；按“最小可读”
规则冻结 `0.5% * N_E`（本网络为 160 个 E neurons）。18/18 paired runs 的 trigger 前 spike bit-identical，runaway 为 0。
该结果只冻结刺激强度，不是患者模式 capacity 证据。正式 fit 使用独立 config 锁定 canary JSON hash，并在 6 个 fit networks 上
运行四臂和全部六个 source。

### 4.3 Readout

每个 source/arm/network 报告：

- propagation success probability、source re-spiking、downstream recruitment；
- mode-conditioned recruitment probability、pairwise precedence、mean rank profile、event-cloud distance；
- `r50/r90`、duration、size、OOD 和 return；
- sham-subtracted response 和 runaway/non-finite 状态。

四层距离保留各自量纲和 patient-training reference，分别报告，不在观察 formal fit 后临时拼成等权综合分数。source-to-mode
prototype Spearman 和 intended-minus-cross margin 只作直观方向性摘要，不作为新的正式 acceptance gate；L2 是否值得进入依据
scalar Edge 的整套 forced readout 与 source-oracle 结构共同判断。

网络 cache 的 provenance 使用冻结 connectivity producer commit 作为 cache key。缺失 cache 时仍以该 commit 写入目标 key，避免
四个 arm 因当前 HEAD 不同而重复建图；launcher commit 与实际运行模块 hash 分别记录，分支在长跑期间前移不能静默改写 producer。

L1 formal 在 fit seeds 1004-1009 上 24/24 完成。Null、Node、Edge 的 source-mode matrix 近乎相同：component 1 对 B 的
prototype rho 约 `0.97`，component 2 对 A 仅约 `0.23`；component 3 和三个 control source 均无可读 curve。Edge 相对 Null
增加下游 spike mass、轻微缩短 `r90`，但没有改善 mode A 的 recruitment、precedence、profile、event distribution 或 prototype。
因此当前证据是 source-location specificity 与 baseline-scaffold directionality，不是 scalar node/edge 恢复患者双模式。Node+Edge
在 seed 1006 的 A 方向翻转，提示共享网络稳定性仍需 L3。注入 frame inclusive sensitivity 不改变上述中位结论。

## 5. L2：最小 relaxed component-pair edge oracle

仅当 L1 显示 scalar `exp(alpha*h_t*h_s)` 的 forced propagation 不足时进入。冻结 `alpha=0.75`，并用
`h_i` 加权 Gaussian responsibility，把远场质量留给 background；不能把几乎为零的 Gaussian 尾部强制归一化成 component。
对 target groups `a in {C1,C2,BG}` 和 source groups `b in {C1,C2}` 定义六参数 residual：

```text
log M_ts = 0.75*h_t*h_s
         + sum_a r_a(t) * [gamma_a1*r_C1(s) + gamma_a2*r_C2(s)]
```

参数方向固定为 source `s` 到 target `t`。仍执行 per-target normalization，保持 topology、delay、incoming-E、E->I 和 GABA。
background/C3 source 是 residual 零参照，component 3 target 固定为零负对照；`gamma=0` 必须逐数组退化为冻结 scalar edge。
六个 `gamma` 的边界为 `[-log(4),+log(4)]`。先用 13 个中心有限差分候选做方向和实现审计，再在 fit seeds
`1004--1009` 上运行 64-point scrambled Sobol；所有候选保留完整结果，只有最终 edge ratio 全在 `[0.25,4]` 且无 runaway
的候选进入排序。

正式搜索目标只读取 patient training。每个模式从六个不同 recording blocks 各取一个事件，重复 512 次，得到与六个 model
network events 同计数的 recruitment、precedence、mean-rank profile 和 event-cloud floors。每项用
`z=(D-floor_median)/floor_IQR` 标准化并取 `softplus(z)`；每模式另加 `2*(1-readable_fraction)`，最后用 `tau=0.25` 的
smooth worst-mode 汇总并加 `0.10*OOD_fraction`。prototype Spearman、response mass、edge KL/ratio/delay 只作解释和 Pareto
distortion 轴，不混入 shape objective。fit 前 8 个候选与 `gamma=0` scalar-edge 基线一起在 selection seeds
`1011--1013` 做 out-of-fit sanity；只有改善在 selection networks 保留后才进入 bounded local refinement。不读
patient held-out，不直接开放全 edge matrix，也不先增加 radial `beta`。

selection 只有 3 个 model events/mode，必须另建 3-event matched patient floor；fit 的 6-event floor 不得用于 selection 的绝对
评分。执行结果为：64-point fit `384/384` workers 成功、57 个结构可容许；top-8 加 baseline 的 selection `27/27` 成功。
count-corrected selection 最优 `sobol_004` 相对 scalar 只改善 `2.27%`，mode A 四项仍全部高于 patient floor 95% 分位，且 A
rank curve 只在 `1/3` selection networks 改变。因此冻结为
`L2_COMPONENT_PAIR_SEARCH_NO_SHARED_MODE_A_RESTORATION`，不做 local refinement、不启 CMA-ES 对比、不开 beta。该结论只覆盖
当前边界和 64-point 有限搜索，不是 family 全空间不可能性证明。

## 6. L3：network 与 optimizer audit

### 6.1 L3a：现有单事件 route-shape surrogate

每个 L2 candidate/network/source 目前只有一次 forced event，因此不能直接把四个分布 descriptor 当成 per-network capacity。先用冻结
classifier PCA 空间中 intended source 到 intended patient centroid 的距离做零新增仿真的探索性 surrogate：

```text
d_A,r(theta) = ||z(y_component2,r) - mu_A||
d_B,r(theta) = ||z(y_component1,r) - mu_B||
S_r(theta) = max(d_A,r(theta), d_B,r(theta))

C_per_net^(1) = median_r min_theta S_r(theta)
C_shared^(1)  = min_theta median_r S_r(theta)
Delta_network^(1) = C_shared^(1) - C_per_net^(1)
```

只纳入 L2 结构可容许候选。上标 `(1)` 明确表示每 source/network 只有一个 event；它只能回答当前 64 点库是否能移动单次 route
shape，不能建立 recruitment、precedence 或 event-distribution capacity，也不使用 patient floor 设置 PASS/FAIL。若 L3a 选出的
shared-surrogate minimum 有并列，二级规则固定为既有 L2 full fit objective 最低、再按 candidate id；不得另看 selection 数据解并列。
若该 candidate 未在 selection seeds 上评估，先冻结该 candidate，再只补三张 selection 网络作 out-of-fit sanity。

现有 57 个结构可容许候选 x 6 张 fit 网络的重放得到 `C_per_net^(1)=0.775`、`C_shared^(1)=1.057`、
`Delta_network^(1)=0.282`。每张网络都有候选优于 scalar 的单事件 weak distance，但 shared minimum 有 20 个候选并列，说明该
surrogate 高度离散，不能识别唯一参数。按预定二级规则冻结 `sobol_058`，只在 seeds 1011--1013 补做 selection sanity；同时计算
route surrogate 与 matched `n=3` 完整 weakest-mode objective，confirmation seeds 仍不读取。

selection sanity 的 3/3 workers 成功。`sobol_058` 的单事件 weak centroid distance 在 seeds 1011/1013 改善、1012 不变，但
matched `n=3` 完整 objective 从 scalar 的 2.717 上升到 2.864，恶化 5.42%；mode A precedence、profile 和 event-cloud 均变差。
因此单事件 surrogate 只能作 route-shape 诊断，不能提供已知好解，`sobol_058` 不进入 optimizer 或 spontaneous confirmation。

### 6.2 L3b：重复事件正式 oracle 与 optimizer 条件

正式分布级审计先只用 6 个 fit network seeds。worker 增加独立 `dynamics_seed`：`network_seed` 继续决定 Params、位置、topology、delay
和 cache key，`dynamics_seed` 只重置 paired sham/forced 的外源噪声 RNG；默认未提供时等于 network seed，必须逐数组复现旧 worker。
冻结 dynamics seeds `31001/31002/31003`，所有候选使用 common random numbers。57 个 L2 结构可容许候选 x 6 networks x 3
dynamics repeats 共 1026 workers，每个 repeat 同时产生 intended A/B source response。每个 candidate/network 尝试产生 3
events/mode，并计算完整 recruitment、precedence、profile 和 event-cloud weakest-mode objective `J(theta,r)`。

正式运行中 `2051/2052` 个 source responses 可读；唯一短缺为 `sobol_017/network1004/component_2` 的 `2/3`。初版聚合错误地把这两个
事件仍按 `n=3` floor 标准化。该问题发现于任何 out-of-fit selection 或后续 optimizer 运行前，随后使用同一 patient-training producer
补建 `n=2` floor；最终按每个 mode
实际可读的 2 或 3 个事件选择 count-matched floor，并保留原来的 readable-fraction penalty。worker commit `217f9982` 与校正聚合
commit `3d654fff` 分开记录；SNN 数组没有重跑或改写。

默认 seed 路径的 parity canary 已通过：`sobol_000/network_seed=1004` 的新旧 23 个 NPZ 数组逐项完全一致，文件 SHA256 均为
`76dbd65036f10ffe9585a45c771af2c90de71b3e65427bfe4ecb8f86a01aeff7`。

```text
C_per_net = median_r min_theta J(theta,r)
C_shared  = min_theta median_r J(theta,r)
Delta_network = C_shared - C_per_net
```

除 L2 已冻结的结构可容许集合外不增加新 gate。fit 只选择候选；3 个 selection networks 仅在 shared candidate 冻结后复核，3 个
confirmation networks 继续不读。每个 mode 有 2 或 3 次可读时使用相应 count-matched floor；只有 0 或 1 次可读、无法构成受支持的
分布描述量时，该矩阵格固定记 `J=100` 并保留 failure reason，不中止整批任务，也不删除该候选。L3b 的 3 repeats 是最小分布级探索，
不能冒充高精度 capacity ceiling。

1026/1026 workers 成功，无 runaway、无 pre-trigger mismatch。count-matched 聚合得到
`C_per_net=2.6560`、`C_shared=2.6755`、`Delta_network=0.0195`；有限库 shared minimum 为 `sobol_002`。逐网络 oracle 在 6/6
networks 都比 scalar 小，paired gain 中位数 `0.0599`，六张网络的 argmin 是六个不同的 residual。更关键的是，每张网络 oracle 的
mode A recruitment、precedence、rank profile 和 event-cloud 四项误差仍全部高于对应 patient-training `q95`。因此这些是有限库
中的小幅 route-shape 改善，不是 mode A realizability。

以下五点是 2026-08-11 复审补入的口径校正，全部由 `src/topic4_rev9l_capacity_audit.py` 从冻结 payload 重算，不再手写：

- **objective 的绝对尺度。** 同样只有 3 个事件的 patient-training 子样本，落在自己 floor 中位时得 `0.6931`、落在自己 floor
  `q95` 时得 `1.6891`。全库 342 个 candidate x network 的 objective 中位数为 `2.8231`、最小 `2.6113`，即中位数高出 patient
  `q95` 参考 `1.1339`。逐网络 paired gain `0.0599` 相当于该差距的 `5.3%`；旧稿的「scalar 中位数的 2.20%」是以零点任意的
  objective 作分母，不再使用。
- **recruitment 不是「固定值」。** mode A recruitment MAE 取 21 个不同值、范围 `0.3517–0.5436`；`0.35173` 是全库最小值，
  由 scalar baseline 在 6/6 networks 达到，252/342 行与之并列，另有 90 行更差。正确表述是「该 family 从未把 recruitment 压到
  scalar 以下，只会变差」，不是「该描述量不动」——后者会被读成仪器失灵而非 capacity 失败。四个描述量里只有 recruitment
  （A 与 B 皆然）具有这一性质，precedence / rank profile / event cloud 都被库中某些候选改善过。
- **失败集中在 mode A 的形状量，recruitment 是两个 mode 共同的读出上限。** 6/6 networks 的逐网络 oracle 中，mode B 的三个形状
  量（precedence `0.59–0.63`、rank profile `0.60–0.77`、event cloud `0.78–0.84` 倍 `q95`）全部落在 patient `q95` 以内，只有
  recruitment 超标（`1.35–1.52`）；mode A 四项全部超标（`1.14–1.35`）。触点 `0/2/4/7` 在 57 个候选 x 6 张网络 x 两个 mode 的
  全部组合里从未被招募，而患者在这四个触点上的招募概率为 A `0.598–0.799`、B `0.740–0.913`；它们独占最优 recruitment 误差的
  A `54.5%`、B `61.4%`。把这四个触点补上后 mode A recruitment MAE 会降到 `0.1602`。因此 recruitment 那一项是 forced 读出的
  几何上限，不能计入 component-pair edge family 的 capacity 证据。
- **`Delta_network >= 0` 是恒等式。** 逐网络最小值的中位数不可能超过任何单一候选的中位数，所以 `Delta_network` 恒非负；
  同理，库中含 scalar baseline 时逐网络 oracle 必然 `<=` scalar，「6/6 networks 改善」不是独立发现，只有幅度是。本轮没有跑
  repeat-level noise null，因此 `Delta=0.0195` 只能作为描述量，不能读成「六张网络需要不同 residual」。
- **描述量 support 不对称，方向对结论有利。** 模型 mode A 的 precedence 只在 `42–110/210` 对触点上有支持、rank profile 只在
  `7–11/15` 个触点上有支持，而 patient floor 建立在 `210/210` 与 `15/15` 上。模型形状误差是在更小、更容易的 support 上取平均后
  与全 support 的 floor 比较，即被系统性低估；本节的阴性结论因此是保守的。

shared `sobol_002` 只在 2/6 networks 改善，paired gain 中位数约 `-6.6e-5`，mean gain `-0.0117`；其 mean objective
`2.7616` 反而差于 scalar 的 `2.7499`。据此冻结
`FINITE_LIBRARY_MODE_A_CAPACITY_NOT_OBSERVED` 和 `STOP_NO_SHARED_FORCED_CAPACITY`。不运行 selection sanity、optimizer benchmark
或 confirmation；该阴性结论只覆盖冻结 scaffold 上测试的有限 bounded static-edge library，不推广为患者模式一般不可学习。

confirmation seeds 只在参数和 objective 完全冻结后读取。只有 oracle 或 deterministic search 已知存在好解时，才以相同 evaluation
budget、common random numbers 比较 Sobol/local refinement 与 CMA-ES 三次 restart。若多组相距很远的参数产生同等输出，结论是
`OUTPUT_REALIZABLE_MECHANISM_NONIDENTIFIABLE`，不是恢复了唯一 core。

本轮没有已知好的 full shared solution，因此 optimizer 状态冻结为
`NOT_TESTED_NO_KNOWN_GOOD_SHARED_SOLUTION`；当前结果不能归因于 CMA-ES 或其他 optimizer failure。既有候选库中参数到粗输出的多对一仍然
成立，所以也不能从小幅 objective 改善识别唯一 edge/core 机制。

## 7. L4：条件性 spontaneous confirmation

仅在 forced shared capacity 成立后恢复无外部 trigger。候选、detector、objective 和 source mapping 全部冻结；primary 使用 frozen
mode classifier，de novo KMeans 只作 secondary。forced pass / spontaneous fail 归为 ignition 或 mode occupancy 问题，不能归为
传播 family failure。

L3b 未建立 forced shared capacity，因此 L4 未运行，patient blind 未打开。当前 ignition 结论仍只限于已知 component source 可以触发
返回基线的传播；spontaneous mode occupancy、完整间期活动和发作生命周期均未测试。

## 8. 唯一决策产物和允许表述

根目录只保留一个 `decision.json`，分别给出 `target_objective`、`ignition`、`propagation_family`、`network_realization`、
`optimizer`、`identifiability` 六个字段；未执行层写 `not_yet_tested`，不能用一个综合 PASS/FAIL 覆盖。

允许的阴性结论最窄到：

- `SCALAR_EDGE_PARAMETERIZATION_FAIL`；
- `FROZEN_SCAFFOLD_STATIC_FAMILY_CAPACITY_FAIL`；
- `SHARED_SUBSTRATE_ACROSS_NETWORKS_FAIL`；
- `OPTIMIZER_SEARCH_FAIL`。

禁止写 `patient interictal modes are unlearnable`。即使 forced 与 spontaneous 均成功，也只能说当前 SNN family 能复现所测的
双模式传播表型；真实 core、完整间期活动和发作生命周期仍需独立证据。
