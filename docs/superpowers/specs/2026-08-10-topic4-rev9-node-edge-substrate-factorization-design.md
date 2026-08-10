# Topic 4 rev9: data-driven node-edge substrate factorization

**状态：** `EXPLORATORY_EXECUTION_ACTIVE`
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
- response 使用 `5 ms` 空间-时间 bins；窗口从 pulse end 后 `[0,10]`、`[10,20]`、`[20,40] ms` 中选；
- source 是 top-hat disk 内 E cells；downstream 是 disk 外 E cells；
- signed response 是每个 bin/cell 的 `spike_mass_kick-spike_mass_sham`；空间形状另用 positive clip；
- 每个 origin 对三个 amplitude 做带截距 OLS，得到 source/downstream response-amplitude slope；
- 全局窗口只看 Node canary 的跨 site/seed median downstream positive mass，最大者为主窗，平局取最早窗；
- SNR、排除数和 kick/sham detector event 全部报告，但不设通过阈值。

如果任一 arm 在窗口中触发 detector event，该 site-seed 在所有 arm 的线性 slope 比较中成组剔除，同时保存触发 arm、
事件数和 exclusion imbalance。runaway 立即终止该 run 并作为 nonlinear/destabilizing outcome，不静默丢弃。

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

## 9. Alpha 探索和四臂运行

### 9.1 结构与局部响应

- alpha coarse grid：`{0,0.25,0.5,1,2,4}`；结构审计先跑且所有候选均保留诊断。
- canary seeds `901--903` 只冻结 response window 和检查量级。
- exploratory selection seeds `901--906` 对 alpha 排序；可以在最佳相邻区间做至多两轮 midpoint。
- `911--922` 在不改 instrument/alpha 的情况下做 out-of-selection 描述性复测，然后进入同 seeds 的四臂长仿真。

排序分数只用于选择一个便于四臂比较的参考 `alpha_star`，不称 equivalence test：

```text
J_cal = median site-seed standardized squared difference of
        [source slope, downstream slope, r90, axis ratio, response map]
        + off-field response penalty + baseline-shift penalty
```

每项同时画原始单位。ratio/KL/ESS reference bands、response-map rho、gain ratio、baseline 变化和有效 paired 数都作为
诊断侧栏呈现。`alpha_star` 是 response-matched reference point，不表示 edge 和 node 机制相同。

### 9.2 四臂 endpoint

至少包括：baseline/event rate、runaway、earliest density、frozen/de novo mode proportion、OOD、recruitment、
pairwise precedence、mean rank profile、event-cloud distance、duration、size 和 return status。每个 endpoint 用 paired-network
bootstrap 给 Delta 和 `I_Y` 的区间；CI 宽时写 `unidentifiable`。

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
postictal protection 和 return/recovery。只有 `alpha_star` 可标记为当前 `response-matched reference`，其他相图 alpha 点均是
新的探索性 substrate。
