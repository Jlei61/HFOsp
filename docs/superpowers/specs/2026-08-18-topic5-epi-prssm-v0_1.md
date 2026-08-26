# Epi-PRSSM v0.1 科学与模型规范

**全称：** Epilepsy Physiology- and Repertoire-Constrained Recurrent State-Space Model

**中文名：** 生理资源锚定、传播 repertoire 约束的癫痫慢状态模型

**状态：** `EXPLORATORY_IMPLEMENTATION_READY`；允许并行开展探索性实验，不代表任何新结果成立

**日期：** 2026-08-18；根据同日科学审阅修订

**适用范围：** 间期事件序列、事件内部 contact 传播、发作前状态和早期发作募集

**对应 plan：** [`2026-08-18-topic5-epi-prssm-v0_1.md`](../plans/2026-08-18-topic5-epi-prssm-v0_1.md)

**自主执行 prompt：** [`2026-08-18-topic5-epi-prssm-autonomous-agent-prompt.md`](../plans/2026-08-18-topic5-epi-prssm-autonomous-agent-prompt.md)

**图形合同：** [`2026-08-18-topic5-epi-prssm-figure-contract.md`](2026-08-18-topic5-epi-prssm-figure-contract.md)

## 0. 一句话论证与研究重心

> 在患者特异传播 scaffold 上，本项目检验一个由长历史 observer 推断、由图约束 recurrent generator 自主演化、并具有有界 resource-like 锚点的慢状态，能否预测未来 IED repertoire、调制相似前缀事件的后缀分布，并在发作前发生可重复位移。累计 IED exposure 是否进一步更新该状态，是独立的嵌套机制实验，不是 H1/H2 成立的前提。

本项目是探索性模型研究。核心主轴恢复为：

\[
\boxed{
\text{慢状态存在}
\rightarrow
\text{慢状态改变事件分布}
\rightarrow
\text{慢状态连接间期与发作}
\rightarrow
\text{IED exposure 是否反向更新慢状态}
}
\]

因此：

- H3 很重要，但不支配整个项目；H3 阴性不降低已经独立成立的 H1、H2a 或 H2b。
- 科学实验阴性通常不停止后续实验，而是自动降低对应 claim；只有数据完整性、冻结纪律和正式 test 完整性构成硬门。
- `shape epileptic network` 在本规范中只指后续功能性传播 repertoire / effective state 改变，不指 anatomical rewiring、synaptic plasticity 或 connectome remodeling。
- 本规范不回改 V2.7、V3.0、V3.1 或 slow-state v4.0 的冻结结果；新模型族与旧阴性结果并列保留。

## 1. 四个独立科学问题

### 1.1 H1：是否存在可自主预测的慢状态

对患者 \(p\)，固定传播 scaffold 为

\[
\mathcal G_p=(\mathcal V_p,\mathcal E_p).
\]

第 \(e\) 场事件前的慢状态 \(\mathbf z_{p,e}^{-}\) 应在 observer correction 关闭后仍能预测 future IED repertoire：

\[
\mathbf z_p(t)\rightarrow p(\mathcal E_{p,e+1:e+H}\mid \mathcal G_p).
\]

H1 不等于 latent trajectory 平滑，也不要求 H3 成立。

### 1.2 H2a：慢状态是否调制单场事件分布

\[
\mathbf z_{p,e}^{-}
\rightarrow
p(\mathrm{suffix}_{p,e}\mid \mathrm{prefix}_{p,e},\mathcal G_p).
\]

全队列主对象是完整事件的 masked participation/rank/order/STOP distribution。ambiguous-prefix suffix 是支持充分患者中的高特异性 targeted analysis，不是全队列 eligibility gate。

### 1.3 H2b：慢状态是否连接间期与发作

模型只用间期事件学习并冻结后，再检验：

\[
\mathbf z_p(t)
\rightarrow
\text{preictal shift, seizure susceptibility, early recruitment}.
\]

TA/TB、seizure label 和 time-to-seizure 不得进入 interictal generator、observer、状态维数、时间常数或 checkpoint 选择。没有外源干预时，只能称 candidate driver 或 state link。

### 1.4 H3：IED 暴露是否反过来更新慢状态

H3 分成两个独立层次：

- **H3a，interictal functional-state updating：** IED exposure 是否改善未来间期 repertoire 的 open-loop 预测，并通过 innovation/directionality controls。
- **H3b，transition participation：** 只有 H3a 和 H2b 都有支持且方向一致时，才检验 exposure-related update 是否参与 interictal-to-ictal transition。

允许出现以下结果分支：

| H2b | H3a | 安全解释 |
| --- | --- | --- |
| 阳性 | 阴性 | 慢状态连接间期与发作，但未支持 IED exposure 是其驱动源 |
| 阴性 | 阳性 | IED exposure 可能更新间期功能状态，但该状态未与发作转换建立联系 |
| 阳性 | 阳性、方向一致 | 可进一步检验 H3b，不自动等于因果 shaping |
| 阴性 | 阴性 | 保留 H1/H2a 的独立结论和明确阴性机制边界 |

## 2. 术语与状态隔离

| 规范术语 | 符号 | 含义 | 禁止替代说法 |
| --- | --- | --- | --- |
| fast event state | \(\mathbf s_{p,e,k}\) | 当前单场事件已走过哪些 contacts | slow state、network state |
| slow generative state | \(\mathbf z_{p,e}\) | 跨事件持续的生成状态 | GRU hidden state |
| observer state | \(\mathbf c_{p,e}\) | 从历史观测积累的推断记忆 | physiological state |
| graph-drive field | \(\mathbf H_p(t)\) | contact-level effective drive | anatomical current source |
| resource-like state | \(r_p(t)\) | 有界的 inhibitory/homeostatic-resource-like scalar | ATP、K\(^+\)、泵状态 |
| exposure state | \(x_{p,\tau}(t)\) | 因果时间核积分后的 IED load | seizure countdown |
| patient baseline | \(\boldsymbol\mu_p\) | train-only 固定 repertoire | dynamic slow state |

代码、日志、图和论文必须保持：

\[
\boxed{\mathbf s_{p,e,k}\neq \mathbf z_{p,e}\neq \mathbf c_{p,e}}.
\]

observer correction、physical generator transition 和 exposure forcing 必须使用不同变量、函数和日志字段。

## 3. 数据对象、患者基线与泄漏合同

患者 \(p\) 有 \(N_p\) 个 SEEG contacts。第 \(e\) 场事件表示为 tied-rank contact sets：

\[
\mathcal E_{p,e}=(\mathcal S_{p,e,1},\ldots,\mathcal S_{p,e,K_e}).
\]

v0.1 node marks 为：

\[
\mathbf Y_{p,e,i}=
[\text{participation}_i,\ \text{normalized rank}_i,\ \text{onset indicator}_i].
\]

硬约束：

- ties 来自显式 group identity，不从相等 rank 反推；
- 非参与 contact 必须 mask，不允许 phantom rank；
- 当前事件 suffix 只能读取事件前状态 \(\mathbf z_{p,e}^{-}\)；
- 完整事件不能先进入 observer 再预测同一事件；
- TA/TB、ictal、SOZ、SNN、time-to-seizure 和 test suffix 不进入主状态学习；
- source/session、recording gap、event silence 和真实 \(\Delta t\) 分别记录；
- cohort 推断单位是患者，不是事件、窗口或 seizure。

### 3.1 固定 repertoire 与动态 residual 分离

event decoder 必须显式写成：

\[
\operatorname{logit}p(\mathcal E_{p,e})
=
\underbrace{\boldsymbol\mu_p}_{\text{train-only fixed repertoire}}
+
\underbrace{D_\psi(\mathbf z_{p,e}^{-})}_{\text{within-patient dynamic residual}}.
\]

要求：

- \(\boldsymbol\mu_p\) 只由该患者 train events 估计并冻结；
- state 只解释围绕 \(\boldsymbol\mu_p\) 的动态偏移；
- state swap 优先在患者内或 baseline-matched 患者之间进行；
- patient/site ID 不得由 observer hidden state 暗中重建为动态状态。

### 3.2 latent gauge

冻结以下 gauge：

- graph state 在 patient/source 内中心化并固定均方范数；
- state-to-adapter 增益有显式正则和上限；
- resource baseline 固定为 \(r=1\)；
- observer correction 和 generator state 使用分别归一化的尺度。

这样避免 \(\mathbf H\times 10\) 与 adapter weight \(/10\) 产生同一输出却让 latent 数值失去可比意义。

## 4. 患者图与 generator 模型阶梯

每位患者区分：

1. directed propagation support \(\mathbf A_p\)，来自 train-only 间期事件；
2. symmetric geometry support/Laplacian \(\mathbf L_p\)，来自固定 contact 几何和/或对称化 scaffold。

图只约束哪些 contacts 可以交换信息以及距离/方向如何进入 message，不代表已知真实 recurrent weight。

### 4.1 Primary：node-level shared graph cell

Primary state 直接定义在 contacts 上：

\[
\mathbf H_p(t)\in\mathbb R^{N_p\times d_H}.
\]

共享参数的 graph message passing 只沿 \(\mathcal G_p\) 支持发生。不同患者节点数通过 patient-wise scan 或 padded graph batch 处理，不要求跨患者 eigenmode 对齐。

### 4.2 G0–G3 模型阶梯

所有模型共享 event marks、patient baseline、observer 接口、state dimension budget 和 split。

#### G0：continuous-time leaky baseline

\[
\dot{\mathbf H}=-(\mathbf H-\mathbf H_\infty)/\tau_H.
\]

G0 是强基线，只能称 leaky state / graph-smooth EWMA，不能称 graph recurrent generator。

#### G1：stable graph-CLDS

\[
\dot{\mathbf H}
=
-\mathbf D_\theta(r)\mathbf H
+\mathbf M_\theta(r;\mathcal G_p)\mathbf H
+\mathbf b_\theta(r),
\]

其中 \(\mathbf M_\theta\) 由共享 graph filter/message function 产生，稳定性通过正阻尼、谱范数约束或解析稳定参数化保证；不学习自由 \(N_p\times N_p\) 矩阵。

#### G2：bounded graph-GRU-ODE

\[
\dot{\mathbf H}
=F_\theta^{\mathrm{graph}}(\mathbf H,r;\mathcal G_p),
\]

使用有界门控、受限 message passing 和数值稳定积分，检验是否需要非线性 recurrent dynamics。

#### G3：graph recurrent generator + autonomous resource

G3 在 G2 或最优稳定 recurrent family 上加入有界 resource anchor。resource 先只包含恢复和 latent-activity consumption，不包含 IED exposure forcing。

### 4.3 Spectral 版本的定位

低频表示

\[
\mathbf H_p=\mathbf U_p^{(K)}\boldsymbol\alpha_p
\]

只作压缩、可视化或 sensitivity，不是唯一 primary 实现。若使用 spectral recurrence，必须额外审计 eigenvector sign、近重复特征值的 subspace rotation、mode permutation 和跨患者 alignment；共享参数优先写成 eigenvalue-conditioned filter，而不是固定 mode-index weight。

## 5. Resource 与 observer 的可辨识接口

### 5.1 resource 必须进入 generator

Primary 路径为：

\[
r
\longrightarrow
\text{generator damping/recurrent gain/readiness}
\longrightarrow
\mathbf H
\longrightarrow
\text{event distribution}.
\]

resource 不得通过大型 MLP 任意改变 contact logits。允许一个 sensitivity arm 让 \(r\) 额外调制全局 continuation/STOP bias，但不是 primary。

### 5.2 primary observer 不逐事件改写 resource

- burn-in history 推断 \(\mathbf H_0\) 和 \(r_0\)；
- event-level observer 可以低维修正 graph-drive estimate \(\widehat{\mathbf H}\)；
- primary observer 不允许每场事件直接写入 \(r_e^+\)；
- 之后 \(r\) 只按 generator 方程恢复、消耗和 exposure arm 演化。

设置 flexible-resource-correction control：允许 observer 对 \(r\) 做低幅度、有惩罚修正。若只有该 control 有效，只能说数据需要额外 latent coordinate，不能说建立了 resource dynamics。

### 5.3 persistent causal observer

\[
\mathbf c_{p,e}=\operatorname{GRU}_\omega(\mathbf c_{p,e-1},\mathbf v_{p,e}).
\]

observer 在 source 内持续；TBPTT 只截断梯度，不重置 forward state。必须记录 correction energy，并分别报告 correction-on filtered 与 correction-off open-loop 结果。

## 6. IED exposure 的嵌套模型阶梯

IED exposure 不是整个 generator 的定义，而是在冻结的 base generator 上增加的机制分支。

### R0：无 resource

对应 G0–G2 中不含 \(r\) 的版本。

### R1：autonomous resource

\[
\dot r=(1-r)/\tau_r-\gamma_q q(\mathbf H)r.
\]

先只用 interictal T1 数据估计并冻结 \(\tau_r\)。

### R2：single-event depletion

\[
r_e^+=r_e^-\exp(-\gamma_L L_e).
\]

它检验单事件 impulse sensitivity；阴性不阻止 R3，也不等于没有累计效应。

### R3：integrated exposure

在冻结 \(\tau_r\) 后定义：

\[
x_{\tau,e}^{-}=x_{\tau,e-1}^{+}e^{-\Delta t_e/\tau_x},
\qquad
x_{\tau,e}^{+}=x_{\tau,e}^{-}+\widetilde L_e,
\]

并仅在 matched T2 中加入：

\[
\dot r=\dot r_{R1}-\gamma_x\bar x_{\tau_x}r,
\qquad \gamma_x\ge 0.
\]

T1 固定 \(\gamma_x=0\)。T1/T2 共享 graph、observer、decoder、state dimension、split、seed 和优化预算。

在 H3 比较中，`T1/R1` 指含 autonomous resource、但不含 event-load forcing 的 matched base arm；`T2/R2` 只增加 single-event depletion；`T2/R3` 在冻结 \(\tau_r\) 后增加 integrated exposure。命名不得把 R1 自主资源和 R3 累积暴露混为一类。

### 6.1 时间尺度策略

为减少 \(\tau_r\) 与 \(\tau_x\) 的不可辨识：

1. R1 先冻结 \(\tau_r\)；
2. R2 单独测试 impulse；
3. R3 development primary 只测试 metadata 支持的 fast / medium / slow 三档；
4. 完整 \(\{5,15,30,60,120\}\) min 和 event-count \(\{5,10,20,40,80\}\) 只作 sensitivity；
5. 多档不可区分时报告可辨识区间，不伪造精确时间常数。

### 6.2 event load 与非同义评价端点

Primary load 仍为 participating-contact fraction：

\[
L_e=\#\text{participating contacts}/N_p.
\]

但 H3a primary outcome 至少包含一个不与 load 同义的端点：

- masked contact order/rank；
- suffix branch；
- propagation direction；
- 控制 participation 后的 repertoire residual。

participation 和 extent 只作 secondary H3 outcome。

### 6.3 innovation challenge

expected load 只使用冻结 T1/R1 state 做 blocked cross-fit：

\[
\widehat L_e=
\mathbb E[L_e\mid \mathbf z_{e,T1}^{-},\mathrm{IEI},\mathrm{rate},\mathrm{source},\mathrm{time\ of\ day}],
\qquad
\eta_e=L_e-\widehat L_e.
\]

不得用 T2 学到的 state residualize T2 自己的 load。raw-load T2 阳性而 innovation challenge 阴性时，只能称 history-dependent predictor 改善。

## 7. State-conditioned event readout

固定 contact-RNN baseline 后比较容量递增的 adapter：

1. no-state adapter；
2. event-RNN initial-state adapter；
3. Node FiLM；
4. 受限低秩 graph edge gate。

不做所有 adapter 的全因子叠加。readout 为：

\[
p_\psi(c_{e,k+1},\mathrm{STOP}mid
c_{e,1:k},\mathcal G_p,\boldsymbol\mu_p,\mathbf H_e^{-},r_e^{-}).
\]

H2a 分三层证据：

1. **全队列：** full-event masked rank/order/participation/STOP distribution；
2. **support-rich targeted：** ambiguous-prefix suffix；
3. **全患者 counterfactual：** correct-state vs matched-state swap。

ambiguous-prefix 支持不足记为 `not_eligible_for_targeted_analysis`，不记为 H2a 失败。

## 8. Open-loop 与发作对齐合同

### 8.1 interictal open-loop

从 anchor event 完成最后一次 observer update 后：

1. 关闭后续 correction；
2. 只提供未来事件时间，不提供未来 marks；
3. generator 自主 rollout；
4. exposure arm 对未来 load 使用 expected-load 或 stochastic sampled load，不读取真实未来 load；
5. 报告 H5/H10/H20/H40；
6. filtered 与 open-loop 分开。

若 correction-off 后立即失效，只能称 event filter。该结果会降低 H1 claim，但不阻止继续定位 event readout、H2b 或 H3 失败来源。

### 8.2 seizure-aligned open-loop

读取 seizure labels 前必须冻结 interictal model family、checkpoint、normalization、主要 endpoints 和 nuisance protocol。最后一场可用 IED 后关闭 observer，自主积分到 clinical onset；不得在发作附近继续用未来事件 correction。

H2b 报告：

- seizure-aligned trajectory 与 matched pseudo-onset；
- leave-seizure-out；
- rate、IEI、source position、time of day、可用 sleep/vigilance controls；
- time-in-warning、patient effect distribution；
- onset state 对 early-ictal order/field/extent 的预测。

## 9. Just-in-time synthetic

synthetic 是各科学问题的单元测试，不是一次性总前置工程。

| 进入哪个 Goal 前 | 只做哪些 synthetic |
| --- | --- |
| H1 / generator | no-state、leaky state、graph recurrent state、observer-overpowering |
| H2a / readout | state-conditioned ambiguous suffix、no-state false adapter、state swap |
| H2b / seizure link | latent preictal drift、event-rate-only confound |
| H3 / exposure | T1、R2 impulse、R3 integrated、hidden common cause、event-count-only、switching |

synthetic 失败只限制对应模型解释或提示修模；不阻塞其他独立 Goal。反复修改同一 synthetic truth 时必须保留版本和 holdout truth，避免对 synthetic test 过拟合。

## 10. Baseline 与 control 菜单

所有实验不必一次运行全部 controls；按被裁决的问题选择最小充分集合。

### H1 最小集合

- static patient repertoire；
- event-index EWMA；
- continuous-time EWMA / G0；
- persistent unconstrained GRU；
- G1/G2/G3 ladder；
- correction-off、state reset、\(\Delta t\) shuffle。

### H2a 最小集合

- no-state adapter；
- correct-state vs matched-state swap；
- rank/participation shuffle；
- patient baseline-only readout；
- support-stratified ambiguous-prefix analysis。

### H2b 最小集合

- matched pseudo-onset；
- leave-seizure-out；
- event rate、IEI、source、time-of-day 和可用 sleep controls；
- last-observation open-loop；
- fixed repertoire 与 recent-event baselines。

### H3 最小集合

- matched T1/R1、R2、R3；
- state-matched load shuffle；
- frozen-T1 cross-fitted innovation；
- time reversal；
- event-count control；
- hidden-common-cause synthetic；
- 至少一个非 participation/extent outcome。

## 11. 三个硬门；其余自动降级

### Hard Gate A：数据与泄漏完整性

必须满足 source/session 可追溯、channel mapping 正确、tied rank 与 non-participation 正确、forbidden inputs fail closed、chronological split 无泄漏。失败的 run 无科学解释资格，修复前不能继续消费该数据。

### Hard Gate B：读取 seizure labels 前冻结 interictal model

冻结文件至少包含待检验 model family、checkpoint、state dimension、normalization、主要 endpoints、nuisance protocol 和 planned contrasts。允许冻结多个预先定义的模型阶梯代表，不要求只留一个赢家；不得根据 seizure outcome 回选。

### Hard Gate C：正式结论使用 untouched test

探索阶段可在 development train/validation 上反复实验。模型族、主要 endpoint 和统计合同确定后，正式主张只能释放一次 untouched test；若继续据 test 调参，后续结果自动回到 exploratory。

以下都不是全项目 blocker：

- G1/G2 未超过 G0；
- open-loop 某 horizon 阴性；
- ambiguous-prefix 支持不足；
- H2b 阴性；
- R2 或 R3 阴性；
- 某个 null 未过。

这些结果只改变对应 evidence card 的结论，不阻止其他实验继续运行。

## 12. 独立证据卡与允许措辞

本项目不再使用一个 Level 0–5 总阶梯，也不设 H3 joint gate 统领全文。每个问题单独出 evidence card。

| Evidence card | 关键结果 | 允许措辞 |
| --- | --- | --- |
| H1-G0 | 只有 G0/EWMA 有效 | leaky history state / observer tracking |
| H1-G1 | G1 稳定超过 G0 | structured graph recurrent slow state |
| H1-G2 | G2 稳定超过 G1 | nonlinear graph recurrent dynamics 有增量 |
| H1-G3 | G3 稳定超过 G2 | bounded resource anchor 有预测增量 |
| H2a | full-event/state-swap 支持；ambiguous suffix 可选加固 | slow state modulates event distribution |
| H2b | frozen state 超过 pseudo-onset/nuisance | slow state links interictal and ictal transition |
| H3a-predictive | R2/R3 只改善 raw prediction | exposure-aware history model improves prediction |
| H3a-mechanistic | predictive + innovation/directionality 同时支持 | IED exposure may participate in functional-state updating |
| H3b | H3a 与冻结 H2b endpoint 同向 | exposure-related updating is consistent with participation in transition |

即使 H3b 成立，也不得写 anatomical remodeling 或已证明 IED 导致 seizure。

## 13. 统计与实验规模

- source/session 内保持 chronology；
- burn-in、TBPTT、horizon 和 eligibility 在相应 formal test 前冻结；
- broad screen 可用 3 seeds；shortlisted/formal model 至少 5 seeds；
- seed 先在患者内聚合，再做患者级推断；
- 同时报告 denominator flow、dataset/support strata 和 unresolved 原因；
- 不把窗口、事件或 seizure 数当患者数；
- patient effect、置信区间和方向分布优先于 pooled P；
- exploratory multiplicity 完整记录，不用单个 nominal P 选择故事。

## 14. 工程边界与输出

建议代码根：

```text
src/topic5_epi_prssm/
├── contracts.py
├── sessions.py
├── graph_templates.py
├── event_marks.py
├── patient_baseline.py
├── graph_cells.py
├── resource_dynamics.py
├── exposure_kernels.py
├── observer.py
├── state_adapter.py
├── event_decoder.py
├── rollout.py
├── synthetic_truths.py
├── trainer.py
└── manifests.py
```

结果根：

```text
results/epi_prssm/v0_1/
├── manifests/
├── data_audit/
├── synthetic/
├── generator_ladder/
├── event_distribution/
├── seizure_link/
├── exposure_mechanism/
└── figures/
```

每个 run 保存 config、code/input hash、split、seed、checkpoint、数值稳定性、state/resource boundary、correction energy、open-loop manifest 和 failure reason。

强制单元测试至少覆盖：状态类型隔离、pre-event leakage、patient baseline train-only、observer 不逐事件改 resource、graph support、latent gauge、resource bound、解析 exposure update、observer-off no-future-mark、future load 不泄漏、TBPTT state carry、source boundary 和 forbidden input。

在作者明确 paper slot 且更新 `docs/paper_figure_registry.md` 前，不得覆盖当前 `results/paper-ready-figure/fig1`–`fig4`。

## 15. 必须同步保留的条款

1. fast event state、slow generative state 和 observer state 永远分开。
2. generator 必须在没有新观测时自主演化。
3. G0 明确称 leaky baseline，不冒充 graph RNN。
4. Primary generator 使用 node-level graph message passing；spectral 版本作受审计 sensitivity。
5. patient static repertoire 与 dynamic residual 显式分开。
6. primary observer 不逐事件直接改写 resource。
7. resource 主要调制 generator dynamics，不任意直连 contact logits。
8. TA/TB 只作冻结后的下游解释，不进入状态学习。
9. ambiguous-prefix 是高特异性 targeted analysis，不是全队列硬门。
10. 发作标签不得反向训练或选择 interictal generator。
11. H3a 与 H3b 分开；H3 不作为 H1/H2 的总 gate。
12. \(\tau_r\) 先冻结，再比较 impulse 与 integrated exposure。
13. H3a 的主评价至少包含一个与 load 不同义的 outcome。
14. raw-load 阳性、innovation 阴性时不得写 event-driven shaping。
15. `shape network` 只指 functional repertoire/effective state。

## 16. 统一接口公式

\[
\boxed{
\begin{aligned}
\mathbf z_{p,e}^{-}
&=\Phi_{G_m}(\mathbf z_{p,e-1}^{+},\Delta t_{p,e};\mathcal G_p,r_p,x_{p,\tau}),\\
\operatorname{logit}p_\psi(\mathcal E_{p,e})
&=\boldsymbol\mu_p+D_\psi(\mathrm{prefix}_{p,e},\mathcal G_p,\mathbf z_{p,e}^{-}),\\
\widehat{\mathbf H}_{p,e}^{+}
&=U_\omega(\widehat{\mathbf H}_{p,e}^{-},\mathcal E_{p,e}),\\
r_{p,e}^{+}
&=\Phi_{R_j}(r_{p,e}^{-},\mathbf H_{p,e},L_{p,e},x_{p,\tau}),\\
\gamma_x&=0\ (T1),\qquad \gamma_x\ge0\ (T2/R3).
\end{aligned}}
\]

其中 \(G_m\in\{G0,G1,G2,G3\}\)，\(R_j\in\{R0,R1,R2,R3\}\)。模型阶梯的目的不是强行证明最复杂模型，而是确定数据实际支持到哪一层。
