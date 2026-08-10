# Topic 4 rev9-L: mode realizability and learnability audit

**状态：** `L1_COMPLETE_L2_ACTIVE`
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

仅当 L1 显示 scalar `exp(alpha*h_t*h_s)` 的 forced propagation 不足时进入。定义 Gaussian soft responsibility `r_c(i)`：

```text
log M_ts = eta11*r1(t)r1(s) + eta22*r2(t)r2(s)
         + eta1_from_2*r1(t)r2(s) + eta2_from_1*r2(t)r1(s)
```

参数方向固定为 source `s` 到 target `t`。仍执行 per-target normalization，保持 topology、delay、incoming-E、E->I 和 GABA。
component 3 固定为零，作为负对照。先做正负 finite-difference response Jacobian，再由受约束 ridge 生成候选，最多 16 个完整
SNN candidates；不直接开放 edge matrix，也不先增加 radial `beta`。

## 6. L3：network 与 optimizer audit

使用 12 个与 rev8/rev9 不重叠的新 network seeds：6 fit、3 selection、3 confirmation。分别报告：

```text
C_per_net = median_r min_theta max(D_A,r, D_B,r)
C_shared  = min_theta median_r max(D_A,r, D_B,r)
Delta_network = C_shared - C_per_net
```

confirmation seeds 只在参数和 objective 完全冻结后读取。只有 oracle 或 deterministic search 已知存在好解时，才以相同 evaluation
budget、common random numbers 比较 Sobol/local refinement 与 CMA-ES 三次 restart。若多组相距很远的参数产生同等输出，结论是
`OUTPUT_REALIZABLE_MECHANISM_NONIDENTIFIABLE`，不是恢复了唯一 core。

## 7. L4：条件性 spontaneous confirmation

仅在 forced shared capacity 成立后恢复无外部 trigger。候选、detector、objective 和 source mapping 全部冻结；primary 使用 frozen
mode classifier，de novo KMeans 只作 secondary。forced pass / spontaneous fail 归为 ignition 或 mode occupancy 问题，不能归为
传播 family failure。

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
