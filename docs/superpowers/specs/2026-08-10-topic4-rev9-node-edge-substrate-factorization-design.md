# Topic 4 rev9: data-driven node-edge substrate factorization

**状态：** `REV9_REVIEW_CORRECTED / REV9L_CAPACITY_AUDIT_ACTIVE`
**日期：** 2026-08-10
**上游：** rev8.1 frozen field；旧 patient held-out 已消费，不能再称 blind

## 1. 科学问题

rev8.1 表明自由连续场可以改善全局事件剖面距离，并形成稳定的双模式符号结构，但最弱模式仍未匹配患者，
也没有胜过 rigid controls。rev9 不再扩展自由场参数，而冻结同一个 `h(x,y)`，问一个更窄的问题：

> 同一空间易感底物能否分别由神经元阈值调制（node）和局部 E->E 权重重分配（edge）实现；二者单独或联合时，
> 分别控制局部响应、起点、模式比例和传播几何的哪一部分？

node 的单位是 mV，edge 参数是无量纲重分配强度，所以两者不比较“参数预算”。先用真实 SNN 的 paired
kick-sham response 描述功能对应，再用 Null / Node / Edge / Node+Edge 四臂做二因素分解。

这是探索性机制实验，不是 patient generalization，也不是完整发作生命周期证明。

## 2. 执行原则：只保留三个硬停止

### 2.1 硬停止

1. **Node 重建错误：** `gamma_node=1` 不能从原 quantile seed 重建 rev8.1 阈值，或 hash/order 不一致。
2. **Edge 结构错误：** 出现 NaN/Inf/负权重，或 topology/delay/E->I/GABA 被改变，或任一非零入边 E target
   的 incoming-E 总量未在 `1e-9` 内守恒。
3. **单次仿真错误：** integrator/worker 报错、状态非有限，或达到既有 runaway early-stop。该 paired run 终止并记录；
   runaway 是结果，不通过删除样本来提高响应匹配。

### 2.2 其余量均为探索性诊断

edge ratio、KL、ESS、baseline rate、SNR、response correlation、mode support、OOD、独立 seed 重复性和 lesion
特异性全部连续报告。旧建议中的数值范围保留为图上的 reference bands / warnings，不阻止后续候选运行，也不在看完
结果后改名为 acceptance gate。

探索轮允许得到阴性或不稳定结果。结论强度由效应、置信区间和跨 seed 一致性决定，不由通过多少 gate 决定。

## 3. 冻结输入和结论边界

- 冻结 rev8.1 的 K=3、17 个场参数、预算投影、`N_core_manual=1129`、神经元位置顺序和 candidate hash。
- `d_i` 是 rev8.1 从原始 quantile seed 独立生成的 signed threshold-depth vector；不能从已调制 `Vtheta` 除以 `h` 反推。
- connectivity axis 继承患者 rank-derived 模板；rev9 不声称独立恢复轴。
- patient train 只用于开发 mode readout 和距离。旧 held-out 仅作历史 read-back，不参与 rev9 选择。
- 长仿真统一使用 `systemd-run --user -> /usr/bin/nohup`，写 `.log/.status`，结束时尝试 `notify-send`。
- response worker 按一个 `arm-alpha-seed` 一进程分片；本机 80 logical CPUs、约 246 GiB available memory 下最多同时 18 个，
  分批运行。缺失 network 只在 `params/connectivity/connectivity_rot` 与冻结来源逐文件 hash 一致时生成一次并供同 seed 复用。

## 4. Node 实现与重建合同

对 E neuron `i`：

```text
d_i = frozen_signed_depth(original_quantile_seed, original_neuron_order)
Vtheta_i(gamma_node) = Vtheta_0 - gamma_node * h_i * d_i
DeltaVtheta_i = -h_i*d_i
```

I neurons 保持 `Vtheta_0`。rev9 node-only 固定 `gamma_node=1`。

正式 preflight 保存：

- `d_vector_sha256`、`h_vector_sha256`；
- `vtheta_rev8_sha256`、`vtheta_rev9_reconstructed_sha256`；
- `max_abs_reconstruction_error`；
- quantile seed、NE 顺序和位置 hash。

如果旧 figdata 没有原始 `d_i`，必须从 seed 和原 producer 重建。`h_i` 接近零的位置不允许通过除法恢复 `d_i`。

## 5. Edge 实现与真实 SNN 方程

`ampa_by_delay[delta]` 固定为行=target `t`、列=E source `s`。权重已经包含 target 的 AMPA jump factor。
primary family 为：

```text
S_t = sum_(s,delta) W^(delta)_(t,s)
u^(delta)_(t,s) = log W^(delta)_(t,s) + alpha*h_t*h_s
W'^(delta)_(t,s) = S_t * softmax_(s,delta)[u^(delta)_(t,s)]
```

这是真正的 source-target pair interaction，不是会在 per-target normalization 中抵消的 `a(h_s)b(h_t)`。
只修改 E target 的 E source edge：topology、delay bin、E->I、GABA、位置和外部输入不变。

实现要求：

- normalization 跨该 target 的全部 delay bins，用 log-sum-exp；
- `S_t=0` 的 target 原样保留，不进入 KL/ESS，单列计数；
- `alpha=beta=0` 走逐值精确 no-op；
- 修改后失效所有已注册或命名为 AMPA-derived 的 cache；GABA cache 保留；
- 报告 edge ratio、target KL、ESS、全部 target、`h>0`、h top 10% 和 active-threshold target 的分层统计。

可选 radial term 仅作为 secondary exploration：

```text
kappa_(t,s) = exp[-r_(t,s)^2/(2*l_EE^2)]
kappa_tilde_(t,s) = target-wise weighted z-score(kappa)
log M_(t,s) = alpha*h_t*h_s + beta*h_t*h_s*kappa_tilde_(t,s)
```

`beta>=0` 只增加径向集中，不能修复一般的轴向或非径向几何错误；weighted centering 也不等于 alpha/beta 已正交。

## 6. 主实验臂和二因素量

每个 network/noise seed 内配对：

| arm | threshold | E->E edge |
|---|---|---|
| Null | uniform `Vtheta_0` | original |
| Node | `Vtheta_0-hd` | original |
| Edge | uniform `Vtheta_0` | `W'(alpha)` |
| Node+Edge | `Vtheta_0-hd` | 同一个 `W'(alpha)` |

对每个 endpoint `Y` 报告：

```text
Delta_N = Y_N - Y_0
Delta_E = Y_E - Y_0
Delta_NE = Y_NE - Y_0
I_Y = Y_NE - Y_N - Y_E + Y_0
```

`I_Y` 是 interaction；没有它不能称 synergy。分类按 endpoint 给出，并允许 `unidentifiable`，不强制整套模型只有一个标签。

## 7. 冻结场的零仿真诊断

### 7.1 Component responsibility

除最近中心外，在每个 event source centroid `x_e` 计算未加 EPS 的 soft responsibility：

```text
r_(e,c) = q_c(x_e) / sum_j q_j(x_e)
```

报告每个 mode 的责任分布、最大责任、最近中心一致率，并提供两个 sidecar null：

- mode-label permutation：mode 间 responsibility 差异；
- pooled source-location resampling：当前 mode-source 关系相对 pooled source cloud 的差异。

它只测关联，不把 component 命名为 causal core。

### 7.2 双 mode readout

任何 intervention 前冻结两套读出：

1. **Primary frozen classifier：** 冻结 rev8.1 embedding、standardization、KMeans centroids、A/B 对 patient-train
   prototype 的对应，以及 baseline centroid-distance p99 OOD threshold。新事件只 assignment，不重新拟合。
2. **Secondary de novo KMeans：** 每个 arm 固定 `K=2,n_init=100,random_state=0`，另报 seeds 0--9 consensus、
   与 frozen labels 的 AMI、cluster support 和 OOD fraction。

第一套回答原模式发生了什么；第二套检查新模式、mode collapse 或聚类边界漂移。两者不互相替代。

## 8. Small-kick instrument

instrument 直接采用 `src/snn_engine/kick_probe.py` 的实际语义：

- E-only top-hat disk；`r_kick=1.0 mm`；
- 初始 pulse onset `100 ms`，duration `18 ms`。若 Node sham 在该响应区间出现自发事件，先仅用 sham trajectories
  在预冻结候选 onset 上选择跨 seeds 的全局安静时段，再冻结新 onset；不得用 edge response 或 patient score 选择 timing；
- 在 pulse 内给 disk 内 E neurons 增加外部 Poisson rate `KICK_BOOST`，单位 `1/ms`；
- amplitude 首轮为 `{0.25,0.5,1.0} * nu_theta`；安静 onset canary 若多数 run 仍直接触发 detector event，
  用 `{0.05,0.10,0.20} * nu_theta` 做一次 threshold mapping。最终局部线性 instrument 的最大幅度取所有 canary
  site-seed 共同保持非事件的最大已测值，另两档固定为其 `1/2` 和 `1/4`；`nu_theta=compute_nu_theta(params)[0]`；
- kick/sham 使用相同 network、OU、Poisson seed；sham 的额外 rate 为 0；
- response 使用 `5 ms` 空间-时间 bins；pulse end 后 `[0,10] ms` 预定义为 first-generation primary window，
  `[10,20]`、`[20,40] ms` 保留为后续传播诊断，不再按最大 downstream mass 选择主窗；
- source 是 top-hat disk 内 E cells；downstream 是 disk 外 E cells；
- signed response 是每个 bin/cell 的 `spike_mass_kick-spike_mass_sham`；空间形状另用 positive clip；
- 每个 origin 对三个 amplitude 做带截距 OLS，得到 source/downstream response-amplitude slope；
- 每个窗口分别报告跨 site/seed median downstream positive mass，但不据此改变 primary window；
- SNR、排除数和 kick/sham detector event 全部报告，但不设通过阈值。

如果任一 arm 的 detector event 与某个窗口重叠，该 site-seed 只在该窗口的跨 arm 线性 slope 比较中成组剔除，同时保存
触发 arm、事件数和 exclusion imbalance。不能用从 kick onset 到最晚窗口结束的 broad interval event 排除全部窗口。
runaway 立即终止该 run 并作为 nonlinear/destabilizing outcome，不静默丢弃。

**首轮 instrument 结果：** `t_kick=100 ms` 的 Node canary 在 seeds 901--903 的 sham 中均有响应区间事件，导致
`0/18` site-seed 可用于线性窗口选择；54 个 kick runs 无 runaway。原始响应保留为启动过渡诊断，但 20--40 ms 的原始
自动窗口已作废。下一步只做 sham-only onset scan，找到全局安静 onset 后再重跑 Node canary；此时不得启动 edge alpha
动力学扫描。

sham-only onset scan 对 `[100,160,220,280,340] ms` 的无事件 seed 数为 `[0,1,3,2,2]`，因此按预定规则冻结
`t_kick=220 ms`。第二轮 Node canary 必须复用首轮 e785 commit 的同三张 network cache，只改变 timing；新结果写独立
`t220` artifact，不覆盖首轮失败证据。

**第二轮 instrument 结果：** `t_kick=220 ms` 后 3/3 sham 的响应区间均无事件，说明 timing 修复成立；但
`{0.25,0.5,1.0} * nu_theta` 的 54 个 kick 中 42 个触发 detector event。只有 `3/18` site-seed 对三个剂量均保持
非事件响应，而且全部来自 seed 903。自动选择的 20--40 ms 因此只是单网络 canary candidate，不能称跨网络冻结窗口。
下一步复用完全相同的三张 network 和 onset，仅将剂量降为 `{0.05,0.10,0.20} * nu_theta`。有效 seed 覆盖作为连续诊断；
覆盖不足不令脚本失败，也不启动当前高剂量下的 edge response matching。

**Threshold mapping 结果：** 54 个 kick 中 16 个触发 detector event，0 runaway；`6/18` site-seed 对三个剂量均为
非事件，覆盖 3/3 networks，但 field component 的全剂量有效响应仍主要来自 seed 903，不能据此冻结跨网络 field response。
另一方面，最低档 `0.05 * nu_theta` 在 `18/18` site-seed 均保持非事件。因此最终线性 canary 固定为
`{0.0125,0.025,0.05} * nu_theta`，只复用 Node canary 的非事件上限，不读取 edge 或 patient score。threshold mapping 中
component 比 control 更早进入 detector event 是一个探索性 ignition-threshold endpoint，不替代局部响应斜率。

**Eligibility 实现纠错：** threshold-mapping artifact 的旧 producer 曾用整个响应区间的 event flag 排除三个窗口，并按
downstream positive mass 最大值选择 20--40 ms。逐窗口重算得到可用 site-seed 数依次为 `16/18`、`11/18`、`7/18`，
说明旧逻辑过度排除早期响应且偏向晚期放大。rev9 从此固定 0--10 ms 为 primary first-generation window；旧 raw artifact
保留，正式解释使用带 source hash 的 window-reconciled sidecar。这是指标语义修复，不增加新的停止条件。

**最终 Node instrument：** `{0.0125,0.025,0.05} * nu_theta` raw canary 无 runaway。逐窗口 sidecar 的可用 site-seed
为 `17/18`、`16/18`、`15/18`，primary 0--10 ms 覆盖 3/3 networks。近临界网络的 event crossing 对 amplitude 并非严格
单调，因此不再继续降幅追求 18/18；event overlap 和 ignition threshold 作为结果报告。

该长跑启动于 commit `944293a2`，但旧 producer 在进程结束时才读取 HEAD/文件 hash，期间的 window-eligibility 修复导致
raw JSON 错标为后来的 commit。raw arrays 保留；正式 sidecar 必须用 `git show 944293a2` 重建 producer/config blob，核对
wrapper、systemd unit 和产物中的 seed/amplitude/window/site 语义。producer 从此在任何仿真前快照 provenance，不能在结束时
重新读取可变工作区状态。

## 9. Alpha 探索和四臂运行

### 9.1 结构与局部响应

- alpha coarse grid：`{0,0.25,0.5,1,2,4}`；结构审计先跑且所有候选均保留诊断。
- canary seeds `901--903` 只冻结 response window 和检查量级。
- exploratory selection seeds `901--906` 对 alpha 排序；可以在最佳相邻区间做至多两轮 midpoint。
- `911--922` 在不改 instrument/alpha 的情况下做 out-of-selection 描述性复测，然后进入同 seeds 的四臂长仿真。

排序分数只用于选择一个便于四臂比较的参考 `alpha_star`，不称 equivalence test：

```text
L_pair = mean(
    pseudoHuber((Edge-Node)/robust_Node_scale)
        for [source slope, downstream slope, r90, log(axis ratio)],
    normalized_sqrt_JS(positive response maps)^2)

J_cal = median(L_pair)
        + (1 - paired_coverage among Node-eligible site-seeds)
        + 0.5 * median(L_pair for matched off-field controls)
        + 0.25 * median(sham baseline shift)
```

`robust_Node_scale=max(1.4826*MAD, SD fallback, feature floor)`；sham baseline shift 联合比较 active-fraction floor、peak
的 log ratio 和 event-count relative difference。配对单位是同 seed、同 site，且 Node/Edge 的 primary 0--10 ms 窗均无
overlapping event。各权重在读取 alpha grid 聚合值前冻结；所有分项、原始标量、pair rows 和缺失比例同时保存。

每项同时画原始单位。ratio/KL/ESS reference bands、response-map rho、gain ratio、baseline 变化和有效 paired 数都作为
诊断侧栏呈现。`alpha_star` 是 response-objective selected candidate，不表示 edge 和 node 响应等效或机制相同。

**Coarse alpha 结果：** 冻结 `J_cal` 在 `{0,0.25,0.5,1,2,4}` 上分别为
`{0.604,0.584,0.624,0.485,0.633,0.990}`。`alpha=0.25` 的纯 response loss 最低，`alpha=1.0` 因 sham baseline
shift 更小而成为 coarse minimum；这不是稳定等效证据。按预定 midpoint 规则只补 `0.75` 和 `1.5`，公式、权重、instrument
和 seeds 均不改变，之后冻结一个 response-objective selected candidate。

**Midpoint 后冻结：** `alpha={0.75,1.5}` 的 `J_cal={0.434,0.472}`，因此冻结 `alpha_star=0.75`。该点的 primary
paired coverage 为 `28/34`；source/downstream slope 的配对 Spearman 分别为 `0.61/0.85`，但 field-component source
仍有明显 identity 偏离。安全口径是“选择了一个便于四臂比较的 Edge reference”，不能写 Node/Edge 等效或收敛到同一机制。
正式诊断图为 `node_edge_calibration/figures/rev9_edge_alpha_calibration.{png,pdf}`。

**Out-of-selection 描述性复测：** 冻结 instrument、权重、尺度和 `alpha_star=0.75` 后，在 seeds `911--922` 得到
`J_eval=0.510`、paired coverage `48/53=0.906`、response loss `0.307`；selection seeds 对应值为
`0.434`、`28/34=0.824`、`0.166`。source/downstream slope 的 Spearman 为 `0.82/0.74`，但 seed bootstrap 95% 区间
分别为 `[0.64,0.91]` 和 `[0.49,0.88]`，`J_eval` 区间为 `[0.211,1.533]`。因此可写“局部响应的跨位置排序关系在未参与选择的
network seeds 上仍可见”，不能写绝对响应等效；response loss 的升高必须在四臂结果中保留。正式诊断图为
`node_edge_calibration/figures/rev9_alpha_star_out_of_selection.{png,pdf}`。这一步没有读取 patient held-out，也不构成患者盲检。

### 9.2 四臂 endpoint

至少包括：baseline/event rate、runaway、earliest density、frozen/de novo mode proportion、OOD、recruitment、
pairwise precedence、mean rank profile、event-cloud distance、duration、size 和 return status。每个 endpoint 用 paired-network
bootstrap 给 Delta 和 `I_Y` 的区间；CI 宽时写 `unidentifiable`。

**探索性四臂结果：** 冻结 `alpha_star=0.75`，在同一组 seeds `911--922` 上完成 Null / Node / Edge / Node+Edge
共 `48/48` 个 worker；无运行失败、OOM 或 runaway。四臂的 detected/usable event 数分别为 `83/13`、`138/121`、
`82/18` 和 `176/169`。Null 与 Edge 的 pooled OOD 分别为 `0.923/0.833`，而 Node 与 Node+Edge 为
`0.033/0.053`。这说明单独重分配 E->E incoming weight 不能替代 `Vtheta=Vtheta0-hd` 产生患者训练分布附近的事件族。

de novo KMeans 的 frozen-assignment AMI 在 Node 和 Node+Edge 均为 `1.000`，10 个 random states 的 consensus AMI 也均为
`1.000`；对应患者训练 prototype 的 Spearman 矩阵分别为

```text
Node       [[+0.264, -0.594],
            [-0.763, +0.977]]

Node+Edge  [[+0.235, -0.576],
            [-0.799, +0.967]]
```

matched mean 为 `0.620/0.601`，因此 Node+Edge 保留了 Node 产生的两模式几何，但没有提高模式匹配。Edge-only 的 matched
mean 仅 `0.228`，且一个 crossed correlation 为正，不支持 edge-only 双模式恢复。event-cloud distance 为 Null `0.571`、Node
`0.595`、Edge `0.436`、Node+Edge `0.557`；该 pooled 描述量受各臂事件数和 OOD 组成影响，不能单独用来宣布 Edge 更接近患者。

在 12 个配对 network seeds 上，usable-event-rate 的 `Delta_Node=+1.125 Hz [0.823,1.438]`，
`Delta_Node+Edge=+1.625 Hz [1.208,2.031]`，interaction 为 `+0.448 Hz [-0.156,1.011]`。participants 的
Node+Edge interaction 为 `+2.917 [1.083,5.000]`，但因为 Null 仅 6 个 seeds 有 usable events，该 endpoint 只有 `n=6`
配对单位。故安全口径是：Node excitability 是稳定双模式几何及患者训练 prototype 定向的主要生成底物；其中 mode A
相关仅 `0.264`，不能说两个模式都已匹配。冻结 Edge redistribution 在 Node 背景上提高事件产出和参与范围，是调制项而不是
等效 core 机制。interaction 的 event-rate 区间跨零，不能写成稳定协同效应。

正式聚合为 `node_edge_factorial/factorial_summary.{json,npz}`；图为
`node_edge_factorial/figures/rev9_factorial_direct_waveforms.{png,pdf}` 和
`node_edge_factorial/figures/rev9_factorial_kmeans_modes.{png,pdf}`。这些 seeds 已用于 local-response out-of-selection 描述，
所以本轮仍是探索性 network-seed factorial，不是新的 network blind，也没有读取 patient held-out。

### 9.3 审阅后零仿真纠错

原始 worker、selection 和 factorial artifacts 保持只读；纠错写入独立 `review_audit_20260810/` sidecar。该审计只读取
patient training blocks，不计算旧 patient held-out 分数。

- **Node 重建：** rev9 `Vtheta0-hd` 与 rev8.1 冻结向量在原 dtype 下逐值一致，最大绝对误差为 0；double multiplication
  排除。
- **Alpha 身份：** `alpha_star=0.75` 通过结构参考带，但逐 site 正响应诊断不足以建立 local-response equivalence。三个 field
  components 的有效 paired seeds 分别为 `8/12、9/12、5/12`；source gain ratio 中位数为 `0.25、0.679、0.50`，response-map
  rho 为 `0.568、0.601、0.736`。因此正式状态为 `LOCAL_RESPONSE_EQUIVALENCE_UNRESOLVED`。`alpha=4` 的最小 edge ratio
  为 `0.085<0.25`，仅标为 `STRUCTURALLY_INADMISSIBLE_EXPLORATORY_ONLY`，不参与 reference 解释。
- **统一 detector 敏感性：** 用 12 个 Node arm-specific thresholds 的中位数 `0.01957` 作事后公共绝对阈值，在
  `0.8/1.0/1.2` 倍下，Null/Node/Edge/Node+Edge 的总事件数分别为 `3/116/4/222`、`3/137/3/221`、
  `2/135/2/324`。高阈值会切分宽事件，故事件数不作为唯一结论。threshold-free mean active fraction 分别为
  `0.000144/0.001917/0.000157/0.003891`，更直接支持 Edge 在 Node 背景上的条件性放大，而不是 Edge-only 点火。
- **网络是统计单位：** Node 和 Node+Edge 均有 `12/12` 网络同时产生 in-support A/B 两种模式；cluster label 与 network
  identity 的 AMI 仅 `0.015/0.007`，leave-one-network-out KMeans 中位 AMI 为 `1.0/1.0`。这排除了“不同网络各自产生一个
  mode”的替代解释。equal-network B fraction 为 `0.493/0.538`，仍低于 patient-training 的 `0.691`。
- **模式矩阵可评价性：** Null 和 Edge-only 的 in-support counts 为 `0/1` 和 `0/3`，状态为 `NOT_EVALUABLE`；其 KMeans
  相关矩阵只保留作描述，不能进入患者模式排名。Node/Node+Edge 可评价。
- **分层模式 readout：** hierarchical profile rho 的 Node A/B 为 `0.238 [0.141,0.341] / 0.977 [0.977,0.986]`，
  Node+Edge 为 `0.227 [0.161,0.279] / 0.986`。mode-conditioned event-cloud distance 中，Node A/B 为
  `1.086/0.486`，对应 patient matched floor p95 为 `0.239/0.185`；Node+Edge 为 `1.138/0.421`，对应 floor
  `0.214/0.153`。因此即使 mode B rank geometry 很强，完整 mode distribution 仍未恢复，mode A 是主要缺口。

审阅后的安全口径是：冻结的 patient-data-constrained Node field 在新网络上稳定产生低 OOD 的双簇传播 repertoire，Edge
在 Node 已点火后增加活动负荷；当前结果没有复现患者完整间期活动，也没有证明 response-equivalent edge 或 causal core。

## 10. Frozen-field causal exploration

三个 Gaussian component 全做：

- direct lesion：删除 `q_c`，保持原 projection lambda，不补预算；
- matched relocation：同 weight/covariance 旋转到预冻结 control location，再对整体 raw q 重新求 budget projection lambda；
- relocation 同时报告剩余 components 的有效峰值、支撑面积、field overlap、total variation 和 top-N neuron identity change，
  避免把 projection collateral change 当位置效应。

`d_i` arms：original、global shuffle、1 mm neighborhood shuffle、positive-only、negative-only；两类 shuffle 各用 5 个预冻结
permutation seeds。positive/negative 同时报 raw 和 `sum h|d|` matched 版本。

阴性 lesion/relocation 不自动阻止后续工程探索，但名称降为 `data_driven_spatial_parameterization`。只有稳定的位置特异性
证据才允许称 `mode-specific core components`。单个 d shuffle 的阴性结果只能写 realization sensitivity 未见明显变化。

## 11. Patient-training objective 的开发版

本轮先实现并校验，不在旧 held-out 上做 formal optimization。每个 mode 联合比较 recruitment、precedence、mean profile
和 event-cloud distribution，最弱模式使用 smooth worst-mode term保护。

有限样本合同改为：

- assignment/proportion pool 固定 `n=50`；`D_JS` 使用全部 50 events；
- shape distances 从每 mode 10 events 的 hash-locked balanced subsets 计算，并重复多次取平均；
- patient split-half floor 使用完全相同的 event count、per-mode count、block hierarchy、missing-contact rule、PCA transform
  和 repeated-subsample estimator；
- frozen-vs-de novo mode assignment 均保存 assignment margin；支持不足只标记不确定，不制造一个漂亮连续分数。

没有新 blind recording unit 时，只能输出 `DEVELOPMENT_ONLY_NO_BLIND_UNIT`。

## 12. Provenance、产物和允许表述

每个 producer 保存 git commit/status、producer/config/imported-module SHA256、Python executable/version、package lock、
systemd unit、network/OU/Poisson/readout seeds 和 paired-unit id。`.status` 原子写 `RUNNING`、`SUCCESS exit_code=0` 或
`FAILED exit_code=N`；`notify-send` 失败不改变任务退出码。

主产物：

- frozen field responsibility/null sidecar；
- edge structure audit；
- instrument JSON 和 local-response surfaces；
- endpoint-level four-arm factorial table；
- Fig4-style direct waveforms + frozen/de novo KMeans 图；
- component/d interaction diagnostics；
- 可重建的 node-edge substrate bundle。

允许表述“在冻结 field 下，node/edge 对局部响应和模式 endpoint 呈现某种效应或交互”。没有新 blind validation 时禁止写
“恢复了真实病人 core”。静态 h/alpha 不替换 Z/M/adaptation；后续 lifecycle 仍须单独检查 entry、bounded carrier、exit、
postictal protection 和 return/recovery。只有 `alpha_star` 可标记为当前 `response-objective selected candidate`，其他相图
alpha 点均是新的探索性 substrate。

rev9-L 的 L0 replay 进一步显示：旧 `joint_loss` 与 mode-A loss 无关联（Spearman `rho=-0.082,p=0.704`），但 fit library
和三个 selection-evaluated candidates 都没有双优 dominator。因此可判定旧 objective 不保护 mode A，不能判定 optimizer
漏掉了一个已知好解。下一步先用 frozen forced source 分离 ignition 与 propagation capacity；随后做冻结 Node 场的 component
lesion / matched relocation 和 `d_i` interaction audit。只有得到可实现的 forced/oracle 解后，才用同预算比较 weakest-mode
objective、multi-restart CMA-ES 与 Sobol/local refinement。当前不开 beta：它只改变径向集中，不能直接修复 source gain、
response-map 或 mode-A 几何；若后续缺口被明确定位为径向连接尺度，再开小型 beta 网格。

## 13. rev9-L forced-source 结果与下一轮连接 oracle

### 13.1 已完成的 forced-source 分解

packet-size canary 只在 Node seeds `1001--1003` 上选择最小可读剂量 `0.005 * N_E=160` cells；正式 fit 改用互不重叠的
seeds `1004--1009`。四臂、六个 source、同一 packet 和同一 sham 均冻结。主 readout 在生成 contact envelope 前，将
`100 ms` 注入帧的 source cells 恢复为 sham 值；包含注入帧的版本只作敏感性。正式结果不运行 KMeans，不读取 patient
held-out，source identity 在仿真前固定为 `component_2 -> A`、`component_1 -> B`。

24/24 arm-network workers 完成，pre-trigger mismatch 和 runaway 均为 0。四臂的 source-conditioned route shape 几乎不变：
含/不含 deterministic packet frame 的 prototype correlation 最大差值仅 `0.018`，因此下表不是注入帧直接读出的假象。

| arm | component 2 vs patient A | component 1 vs patient B |
|---|---:|---:|
| Null | 0.230 | 0.967 |
| Node | 0.230 | 0.972 |
| Edge | 0.230 | 0.967 |
| Node+Edge | 0.230 | 0.967 |

component 3 和三个 off-field controls 均不能形成可用的全 contact rank curve。四层 readout 同样显示 A 是限制项：A 的
prototype Spearman 约 `0.09--0.24`、precedence MAE 约 `0.46`、event-cloud distance 约 `1.07--1.21`；B 的 prototype
Spearman 约 `0.966--0.970`、precedence MAE 约 `0.20--0.21`、event-cloud distance 约 `0.66--0.69`。因此 forced
initiation 证明 C1/C2 是可访问 source，但没有恢复患者 mode A 的具体传播几何。

Edge 相对 Null 的 paired downstream positive-spike mass 增量在 C2 和 C1 分别为 `4583 [4114,4916]` 和
`5400 [4850,5839]`，同时 source re-spike mass 增加；rank shape 基本不变。这支持 `conditional relay/amplifier`，不支持
`Edge 恢复了第二个患者模式`。Node/Node+Edge 在部分 seeds 的 sham trigger window 内已有事件，故其 forced-minus-sham mass
和 factorial interaction 只作描述，不解释为真实抑制或拮抗；sham-clear sensitivity 单独保存。

结构 sidecar 在 12 张网络上进一步显示：C1/C2 self-flow 增加约 `13.9%/11.6%`，BG-source 到 C1/C2-target 减少约
`8.7%/6.8%`，C1/C2 source-group outgoing influence 增加约 `7.6%/7.0%`，对应 target 的 weighted incoming delay
提前约 `0.22/0.18 ms`。但逐细胞 `h` 与 outgoing log-ratio 的 Spearman 为 `-0.656`，所以该 mapper 是群组层面的
field-assortative redistribution，不是单调 high-h boost，也不是严格局部径向 core。

当前诊断必须分层：

- **objective limitation：已证实。** 旧 objective 不保护 mode A；
- **optimizer limitation：未裁定。** 搜索只有有限 restart 且未收敛，但 archive 中不存在 known-good 双优解；
- **scalar edge / frozen scaffold limitation：已有证据。** 绕过 ignition 后，四臂 route shape 仍几乎相同且 A 仍弱；
- **完整患者间期活动：未复现。** 当前只支持自发 event-like activity、双 repertoire 和部分传播表型，不能扩展到临床波形、
  频谱、持续时间、事件率或完整事件分布。

### 13.2 L2 component-pair residual edge oracle

下一轮先保持 `h`、`d`、`Vtheta`、topology、delay labels、E->I/GABA 和 `alpha=0.75` 全部冻结。用 soft membership
`r_a(i)`（`a in {C1,C2,BG}`，C3 保留为未调制负对照）增加六个可辨识 residual：

```text
log M_t,s = alpha * h_t * h_s
            + sum_{a in {C1,C2,BG}} r_a(t)
              * [gamma_a1 * r_C1(s) + gamma_a2 * r_C2(s)]

W'_t,s,delta = W_t,s,delta * M_t,s
               * sum_s,delta W_t,s,delta
               / sum_s,delta [W_t,s,delta * M_t,s]
```

BG-source 是每个 target-group 的参考水平，因此没有不可辨识的整行常数。该 family 可以分别改变 C1/C2 self relay、交叉 relay
和 component-to-background influence，同时仍保持每个 target incoming-E 总量。它是用于判断 family capacity 的 relaxed oracle，
不是最终患者机制。参数范围首先由旧 ratio `[0.25,4]`、KL 和 ESS 诊断截断；越界点只标为 exploratory inadmissible，不据此
反复增加 blocker。六个 `gamma` 的 Sobol box 固定为 `[-log(4), +log(4)]`；每个候选仍在真实稀疏矩阵上重算 edge ratio，
box 本身不替代结构审计。

训练只用 forced sources C2->A 和 C1->B。对 mode `k` 和 descriptor
`q in {recruitment, precedence, profile, event-cloud}`，先用患者训练集做与模型每 mode 数量匹配的 hash-locked bootstrap，保存
floor median `m_qk` 和 IQR `s_qk`，然后定义：

```text
Z_qk = (D_qk - m_qk) / (s_qk + eps)
D_k  = mean_q softplus(Z_qk) + 2 * (1 - readable_fraction_k)
J_shape = tau * log[exp(D_A/tau) + exp(D_B/tau)]
J_L2 = J_shape + 0.10 * OOD_fraction
```

`J_shape` 保护最弱模式；intended-minus-cross Spearman、response mass、r50/r90 和 return 只作辅助曲线，不替代四层主目标。
ratio、KL、ESS 和 weighted-delay change 单独形成 distortion axis，与 `J_L2` 画 Pareto，不再用任意权重混入患者形状目标。
首轮是探索性 capacity scan：在 fit seeds `1004--1009` 上用 common random numbers 做 64 个 Sobol points；保留 Pareto 前 8 个做
小范围 local refinement，再在未参与筛选的 `1011--1013` 上复核。不得读取 patient held-out，也不得根据结果修改 source mapping。

### 13.3 optimizer 与 beta 的裁定实验

只有 L2 或其他明确 oracle 找到同时改善 A/B 的已知解，才进入 optimizer 归因：

1. **synthetic-teacher recovery：** 用已知 oracle 参数产生 hash-locked synthetic target，检查优化器能否在同预算恢复 route
   descriptors；失败才是直接的 optimizer 证据。
2. **同预算 head-to-head：** weakest-mode objective、common random numbers 和相同 simulation calls 下比较 multi-restart
   CMA-ES 与 Sobol + top-k local refinement；报告 best-of-budget、restart variance 和 known-solution regret。
3. **objective ablation：** 同一候选集分别按旧 global objective、均值 mode objective 和 worst-mode objective 排序，确认改善
   来自目标定义而不是额外预算。
4. **family decision：** 若 L2 仍不能提高 A，则停止调 scalar alpha/beta，转而检查 source-target directional scaffold 或 montage
   observation contract；不得把失败继续归因于 CMA-ES。

当前 `beta` 保持 0。只有候选已经同时恢复 A/B 的 recruitment、precedence 和 profile，而剩余误差主要表现为 r50/r90 或
weighted-delay radial scale 时，才做小型 beta 网格。现在 Edge 已轻微缩短 r90，而 A route shape 未改善，直接开 beta 的方向不对。

正式产物：

- `edge_structure_detail/edge_structure_detail_summary.json` 与结构图；
- `forced_source_capacity/formal_fit/forced_source_capacity_summary.json`；
- `forced_source_capacity/formal_fit/review_audit/rev9l_l1_review_audit.json` 与审阅图。
