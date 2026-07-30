# Topic 5 / Figure 6：间期单事件传播算子到发作早期 rollout（v0.2）

**版本**：v0.2
**日期**：2026-07-24
**状态**：**已被 v0.3 取代，不得按本合同启动正式训练**
**取代范围**：取代 v0.1 中以 `32 个间期事件历史 → 未来事件分布/发作场` 为主任务的设计；不覆盖已经完成的 Fit1/Fit2 结果和旧 Gate 2 阴性审计。

> **2026-07-24 发作侧更正**：真实时序复核显示，0–10 s 内的触点空间场主要是
> 患者特异的稳定场，且 clinical onset 前已经可见；现有结果不支持把它解释成
> 可监督的逐秒接触点招募顺序。因此本版的 `1 s seed → 1–10 s rollout` 已撤回。
> 当前合同见
> `docs/superpowers/specs/2026-07-24-topic5-interictal-operator-static-ictal-readout-design.md`；
> 数据裁决见
> `docs/archive/topic5/fig6_ictal_target_temporal_adjudication_2026-07-24.md`。

## 1. 一句话科学问题

本分析只回答：

> 仅用单个间期群体事件内部的 contact-rank 轨迹学到的 recurrent propagation
> operator，在给定发作 clinical onset 后极短 seed 的条件下，能否预测其余
> 早期发作期的 contact × time BB 1–150 Hz 招募场，并超过使用同一静态
> scaffold 和同一 seed 的非递归模型？

主箭头固定为：

```text
interictal within-event rank trajectories
    → frozen recurrent propagation operator
    → seed-conditioned early-ictal rollout
```

第一版不再把多个间期事件之间的顺序当成主 recurrence。event history
只保留为 Stage C 可选扩展；Stage A–B 即使在 Stage C 阴性时也必须能够独立成立。

## 2. 与论文主线的关系和结论边界

### 2.1 已有经验结果不改

下列 parent contract 原样保留：

- 发作时间锚为 **clinical onset**，不是 EEG onset。
- 发作主频带为 line-noise-bin-masked **BB 1–150 Hz**。
- 已接受的静态读出为 TA/TB `maxAB`，每次置换重新选择 A/B 与 mirror。
- 静态主 null 为 **all-contact channel-label shuffle**。
- 先按 seizure，再按 subject，最后按 cohort 折叠。
- pure within-shaft shuffle 只作 anatomy-controlled sensitivity。
- Fit1 的 full-record 静态 benchmark 和 Fit2 的 prefix-only scaffold retention
  保持原结论，不因本模型重构而重新解释。

Fit1/Fit2 的 target-driven A/B/mirror `maxAB` 是已经接受的**回顾性 scaffold
expression readout**，不是可以直接喂给预测器的标签选择步骤。Stage B 的
预测分支不得查看 target 后再选择 A/B 或 mirror；它必须使用 swap-invariant
static context，在 outer-training patients 上学习 target-blind 映射。

旧 event-history RNN 的 Gate 2 阴性仍然是旧任务的有效结果，但它不等价于本
v0.2 的 single-event operator test。

### 2.2 v0.2 成功时允许的结论

只有 Stage A 和 Stage B 都通过时，才允许写：

> A recurrent propagation operator learned from within-event interictal rank
> trajectories transferred to seed-conditioned early-ictal recruitment.

即使成功，也只能解释为患者病理传播 scaffold 的**可迁移计算算子**，不能解释为：

- 发作时间预测或无预警的 seizure forecasting；
- 逐接触点精确生理重放；
- 因果致痫机制；
- 真实神经元级 E/I 参数恢复；
- 间期事件之间已经存在 latent-state transition。

由于模型读取 clinical-onset seed，本任务是 onset 后 rollout，不是临床发作预警器。

## 3. 冻结的数据对象

### 3.1 患者和触点集合

Epilepsiae 的候选母队列从已经通过 Fit2 prefix-scaffold gate 的患者和 strict
BB150 clinical-onset seizures 出发。v0.2 不因模型表现重新选择患者、发作、
频带或时间窗。

每名 Epilepsiae 患者的主触点集合为：

```text
C_main
  = C_interictal_masked_rank
  ∩ C_time_resolved_ictal_BB150
  ∩ C_exact_channel_name_join
```

规则：

- 主分析只预测 `C_main`，不填补 ictal-only contact。
- channel-name join 必须 exact、fail closed，并核对 channel ordering。
- 坐标缺失只影响需要 geometry 的模型分支；不能把缺失坐标的触点静默重排。
- all-ictal-contact coordinate-query extrapolation 只可作为后续 sensitivity，
  不能替代 common-contact primary。

### 3.2 单个间期事件轨迹

对每个事件 \(e\)：

- 从 raw `lagPat` 和 `eventsBool` 构造参与 mask \(m_{e,c}\)；
- 只在 `m_{e,c}=1` 的触点内重新计算 rank，禁止读取 phantom-contaminated
  legacy finite ranks；
- 将相同或无法可靠区分的 rank 合并为一个 recruitment set；
- 得到事件内 pseudo-time 序列
  \(G_{e,1},G_{e,2},\ldots,G_{e,L_e}\)，其中每个 \(G_{e,\tau}\)
  是本步新招募触点集合；
- 未参与触点不是“最后招募”，而是 masked/non-participating。

事件纳入的最低要求在 Phase 0 只按输入质量冻结，不得查看 Stage B 性能。
建议起始值为：`|C_main| >= 6`，单事件至少 3 个参与触点、至少 2 个
pseudo-time steps。

第一版不输入：

- IEI；
- event rate；
- clock time；
- time-to-seizure；
- 距离 RESET 的事件数；
- acquisition block ID；
- 发作标签或发作能量。

### 3.3 患者静态 context

患者静态 context 记为：

```text
S_p = (T_A, T_B, participation support, contact geometry, valid masks)
```

它只能来自该患者 chronological definite-interictal prefix。A/B 无顺序意义，
结构上使用 swap-invariant 表示，例如：

```text
g_AB = rho(
    phi(T_A) + phi(T_B),
    abs(phi(T_A) - phi(T_B))
)
```

不再用 soft A/B swap penalty 补救一个本身依赖 A/B 标签顺序的结构。

### 3.4 时间分辨的发作目标

新建 time-resolved BB150 cache，但必须复用 accepted producer 的以下定义：

- clinical onset 对齐；
- line-noise-bin-masked 1–150 Hz log power；
- 相同的 spectrogram、baseline window、guard 和 baseline robust-z；
- 原始分析窗 `[0,10] s`。

主时间离散固定为 1 s：

```text
seed:   [0,1) s
target: [1,2), [2,3), ..., [9,10] s
```

每个触点、每个时间 bin 保存：

```text
Y_energy[p,s,c,t] = mean baseline-robust-z BB150 power
Y_rank[p,s,c,t]   = within-seizure quantile rank of Y_energy over C_main
```

`Y_rank` 是主输出，`Y_energy` 是强制辅助输出。能量近似相同的触点对不应被
强迫排序。pairwise rank loss 的权重固定为目标能量差的单调截断函数；
near-tie noise floor 只从 outer-training patients 的 baseline/重复 frame
波动估计，不能根据 ictal prediction performance 调参。

另保存两个不参与模型选择的 summary：

- `[1,10] s` integrated-energy contact rank；
- time-resolved recruitment trajectory。

0.5 s seed/bin 只可作为预注册 sensitivity。不能在看过模型表现后在
0.5 s 和 1 s 中择优。

### 3.5 新 cache 的强制 parity

对每个既有 eligible seizure：

1. 将 time-resolved cache 的 `[0,10] s` frames 按 accepted 权重重新聚合；
2. 与既有 `bb150_auc__<seizure_idx>` 逐触点比较；
3. channel order、finite mask 和数值必须在预先声明的浮点容差内一致；
4. parity 失败则停止，不训练模型。

新 cache 不能用另一个滤波器、另一个 baseline 或另一套 onset 提取近似代替。

## 4. 训练集、验证集和测试集

### 4.1 外层评估

Stage B 使用 outer leave-one-Epilepsiae-subject-out：

- outer-train：其余 Epilepsiae 患者的 seizures 和 interictal prefix；
- outer-test：held-out 患者的全部 eligible strict-BB150 seizures；
- held-out seizure 的任何 target frame 均不得进入训练、早停、归一化、
  near-tie 阈值或超参数选择。

这是带 target-free patient calibration 的 LOSO：

- held-out 患者的间期 prefix 是模型的合法输入；
- 它可以用于构造 \(S_p\)，并按预先冻结的 epoch 数执行 Stage-A-only
  self-supervised calibration；
- 不得用 held-out seizure target 决定 calibration epoch、checkpoint、
  learning rate 或模型 rank。

另做一个 strict-inductive sensitivity：held-out 患者不更新任何 core 参数，
只输入其冻结的 \(S_p\)。

### 4.2 内层选择

所有以下选择只在 outer-training patients 内做 inner patient-level validation：

- hidden size；
- loss 权重；
- Stage-A calibration epoch 数；
- early stopping；
- low-rank \(r\)；
- optional gain 数量；
- optimizer 和 weight decay。

不能以 seizure 为单位随机拆 train/validation。

### 4.3 Stage A 数据池

Stage A 的共享初始化使用：

- 当前 outer fold 的 Epilepsiae training patients；
- 全部合格 Yuquan interictal events。

Yuquan 约束：

- 只进入 Stage A，无 ictal label，不进入 Stage B 统计；
- 先 dataset-balanced，再 patient-balanced，最后 event-balanced；
- 与 Epilepsiae 使用完全相同的 mask、rank、坐标和事件质量定义；
- 强制报告 Epilepsiae-only Stage-A sensitivity；
- 加入 Yuquan 若只改善 Stage-A reconstruction、未改善 Stage-B transfer，
  不得写成主要模型优势。

每名患者的 prefix events 按时间前 80% / 后 20% 划分：

- 前 80%：Stage-A patient calibration；
- 后 20%：held-out within-event suffix evaluation。

Stage-A gate 中使用的 TA/TB/static baseline 也只能由前 80% 构建；禁止把后
20% 的评估事件先并入模板再预测它们。

比例在看任何模型结果前冻结。Stage B 使用的最终 core 可在完成 Stage-A
gate 评估后，按固定训练步数在该患者全部合法 prefix events 上重拟合；仍不得
读取 seizure target。

### 4.4 Stage B sampling 和统计单位

训练采样顺序为：

```text
uniform patient → uniform eligible seizure
```

防止多发作患者主导梯度。测试折叠顺序固定为：

```text
time/contact metric per seizure
→ median over model seeds within seizure
→ median over seizures within patient
→ cohort statistic over patients
```

model seed、contact、time bin 和 seizure 都不是独立统计样本。

## 5. Stage A：学习单事件传播算子

### 5.1 任务

对每个事件随机选择合法截断点 \(\tau<L_e\)，输入：

```text
G_e,1, ..., G_e,tau
```

输出：

- 下一 recruitment set \(G_{e,\tau+1}\)；
- `STOP` 概率；
- 每个尚未出现触点最终是否参与；
- 被 mask 的 suffix relative rank。

损失为：

```text
L_A
  = L_next_set
  + λ_stop * L_stop
  + λ_remain * L_remaining_participation
  + λ_rank * L_masked_suffix_rank
```

同 rank tie group 使用 set likelihood/multilabel target，不任意拆成伪顺序。
事件按患者等权，不能让事件数多的患者主导。

### 5.2 Stage-A baseline

至少包括：

1. patient static TA/TB mixture；
2. first-order Markov next-contact model；
3. unordered prefix DeepSets；
4. 同 participation mask 的 within-event rank shuffle；
5. matched-capacity feed-forward contact-query model。

Stage-A 主指标为 held-out patient 的 next-set negative log-likelihood；top-k
accuracy、STOP calibration、remaining-participation AUROC 和 suffix-rank
concordance 为辅助指标。

## 6. Stage B：冻结 core 的发作早期 rollout

### 6.1 先冻结 static branch

每个 outer fold 先只用 outer-training patients 训练：

```text
Y_static_energy = f_static(S_p, contact, time)
u_static_rank    = g_static(S_p, contact, time)
```

并完全冻结。rank 使用 contact utility，不对 quantile ranks 直接做任意减法：

```text
P(i ranks above j) = sigmoid(u_i - u_j)
```

static branch 训练完成后才定义 residual：

```text
R_energy = Y_energy - Y_static_energy
u_total  = u_static_rank + Δu_dynamic
```

static 和 dynamic branch 禁止联合从头训练。
`D_energy` 只拟合冻结后得到的 `R_energy`；rank residual utility 在 contact
维度中心化并使用最小范数正则，防止 dynamic branch 无约束地重建整套 static
scaffold。

### 6.2 seed adapter、rollout 和 decoder

seed adapter 读取：

```text
(Y_energy[:, 0:1 s], Y_rank[:, 0:1 s], S_p, valid masks)
```

得到 recurrent 初始状态 \(h_0\)。随后：

```text
h_t     = F_IED(h_{t-1}, predicted_contact_activity_{t-1})
ΔY_t    = D_energy(h_t, S_p, contact)
Δu_t    = D_rank(h_t, S_p, contact)
Yhat_t  = Y_static_t + ΔY_t
uhat_t  = u_static_t + Δu_t
```

约束：

- \(F_{\mathrm{IED}}\) 在 Stage B 完全冻结；
- 只训练 seed adapter、contact-query residual decoder；
- 如确有必要，最多允许 1–2 个全局 low-dimensional rollout gain；
- gain 数量由 inner LOSO 选择，不能变成逐患者自由参数；
- 主训练和测试均从 seed 纯 closed-loop rollout，不把真实 `[1,10] s`
  frame teacher-force 回模型；
- teacher-forcing 结果只能作诊断。

### 6.3 Stage-B loss

```text
L_B
  = L_pair_rank
  + λ_energy * L_robust_energy
  + λ_temporal * L_temporal_difference
```

其中：

- `L_pair_rank`：带 near-tie mask/weight 的 Bradley–Terry pair loss；
- `L_robust_energy`：对 normalized energy residual 的 Huber loss；
- `L_temporal_difference`：只约束一阶变化误差，不直接把输出平滑成静态场。

损失权重仅由 inner patient-level validation 确定。论文主 endpoint 是
contact-rank concordance；energy error/correlation 必须同时报告，不能只报
rank。

## 7. 模型架构

### 7.1 V1：contact-query GRU

V1 是先行可用模型，用来回答 recurrence 是否有必要。

- contact encoder 读取 valid mask、中心化 geometry/shaft position 和当前
  recruitment state；
- 同步 recruitment set 先用 permutation-invariant pooling 编成一个 token；
- GRU 逐个 within-event pseudo-time step 更新 hidden state；
- next-set/STOP/remaining-participation 由 contact-query decoder 输出；
- Stage B 中 seed adapter 将 1 s ictal field 映射到同一 hidden state，
  后续使用预测 contact activity 闭环驱动 frozen GRU。

主候选 hidden size 为 `{32, 64}`，最终由 inner LOSO 选择。所有 patient ID、
channel string embedding 和 seizure ID 均禁止输入。

V1 必须先通过 Stage A 和 Stage B，才允许进入 V2 生物约束模型。若 V1
不通过，不得用更复杂 low-rank/EI 参数搜索挽救。

### 7.2 V2：within-event low-rank E/I recurrent core

V2 与 V1 使用相同输入、target、split、decoder capacity 和统计合同，只替换
core。每个 contact 具有最小 E/I surrogate state：

```text
E_next = leak_E * E + phi_pos(W_EE E - W_EI I + input)
I_next = leak_I * I + phi_pos(W_IE E - W_II I)
```

结构约束：

- excitatory contribution 非负，inhibitory contribution 以负号进入 E；
- local term 仅由同杆邻近或冻结的空间 kernel 构造；
- recurrent non-local term 为 rank-\(r\) factorization；
- `r ∈ {0,1,2,3,4}`，`r=0` 是必要 control；
- contact 数变化通过 masked node states 处理，不用 padding 数量编码 event rate；
- pseudo-time step 只解释为传播更新，不拟合成真实毫秒级膜时间常数。

Stage B 只允许一个共享 rollout-step gain 将 event pseudo-time 更新映射到 1 s
bin；因此可以讨论“同一低维传播算子是否可迁移”，不能声称恢复了真实物理
时间常数。

只有 low-rank gate 通过后，才允许讨论少数 recurrent modes。E/I 命名也只表示
within-event/early-ictal 的 sign-constrained local inhibition surrogate，不代表
真实细胞类型计数。

## 8. Stage-B baseline 和 null

所有 comparator 使用完全相同的 outer/inner split、seed、static branch、
common contacts、训练预算和 patient-level fold。

强制 baseline：

1. frozen static scaffold only；
2. frozen static + seizure seed 的非递归 residual MLP；
3. random frozen core + 相同 seed adapter/decoder；
4. 在 within-event rank-shuffled data 上训练的 frozen core；
5. seizure-only matched GRU：不做 Stage A，直接从 seed 学 rollout；
6. matched-capacity unconstrained recurrent core；
7. V2 的 rank-0 core。

可行时增加 other-patient core；若 contact topology 无法无歧义匹配，只报告为
不可定义，不做事后近似。

Null 分层：

- accepted static scaffold 的主 null 仍是 all-contact channel shuffle，并在每次
  draw 重算 A/B、mirror 和 `maxAB`；
- within-shaft shuffle 仍只作 anatomy sensitivity；
- Stage-A operator null 保持 participation mask，只打乱单事件内 rank；
- seed-channel shuffle 和 core-label shuffle 只作模型诊断，不能替代 matched
  baseline comparison。

模型主检验是相对 frozen static + same seed 的增量，不是单独“超过随机均值”。

## 9. 训练器和可复现性

统一使用自定义 PyTorch trainer：

- optimizer：AdamW；
- gradient clipping：global norm 1.0；
- mixed precision：仅在 float32 parity test 通过后启用；
- early stopping：inner-patient validation，patience 建议 15；
- 最大 epoch：Stage A 200、Stage B 200；实际 checkpoint 由 inner validation
  选择；
- 三个预先冻结 seeds，seed 先在患者内折叠；
- one-standard-error rule 选择满足性能的最小 hidden size/最小 low rank；
- batch 按总 contact-token 数限制，避免由最大 contact patient 决定显存；
- 每个 cell 保存 config、input fingerprint、checkpoint、epoch log、
  predictions、resource log 和 `DONE.json`；
- 训练可恢复但不得静默覆盖已有 cell；
- 记录 CPU RAM、GPU memory/utilization、swap 和磁盘余量；OOM 后只能缩小
  batch/token budget，不能改变科学 cohort。

所有归一化统计、near-tie threshold 和 early stopping state 都必须保存在
outer-fold checkpoint 中。

## 10. 四道科学门

| Gate | 预注册条件 | 通过后允许的结论 |
|---|---|---|
| Event dynamics | single-event V1 在 held-out patients 的 next-set NLL 上超过 strongest static/first-order baseline，且超过 within-event rank-shuffle；patient bootstrap 95% CI 不跨 0 | 间期 rank event 内存在可学习的递归传播规律 |
| Ictal transfer | frozen IED core + seed 在 patient-level rank endpoint 上同时超过 static+same-seed、random/shuffled core 和 seizure-only matched GRU；primary CI 不跨 0，energy secondary 方向一致 | 间期传播算子可迁移到 seed-conditioned early-ictal rollout |
| History state | 仅 Stage C：ordered history 超过 DeepSets/reversed/shuffle，并通过同患者 correct history–seizure pairing | 近期事件顺序含 seizure-specific state 信息 |
| Low rank | rank 1–3 在 one-SE 下匹配 V1/full GRU、超过 rank 0，且关键 mode lesion 降低 Stage-B transfer | 传播重用可由少数 recurrent modes 描述 |

Gate 按顺序执行：

1. Event dynamics 未通过：停止 Stage B 正式训练。
2. Ictal transfer 未通过：停止 V2 机制解释。
3. History state 未通过：不影响 Stage A–B，但禁止 latent-state-transition
   wording。
4. Low rank 未通过：保留 V1 预测结果，禁止 low-rank mode/EI 机制结论。

“工程完成”不等于 gate 通过。

## 11. 主要统计

### 11.1 Primary endpoint

每次 seizure 在 `[1,10] s` 上计算：

- 每 bin contact-rank concordance；
- 跨 bin 的预注册中位数；
- `[1,10] s` integrated-energy contact-rank concordance。

primary model increment：

```text
patient median(
    score[frozen IED core + seed]
    - score[frozen static + same seed]
)
```

cohort 使用 paired one-sided Wilcoxon 和 patient bootstrap 95% CI。Gate 以
bootstrap CI 为硬条件，P 值作配对证据。所有 baseline 的统计方向和折叠顺序
预先固定。

### 11.2 Mandatory secondary

- normalized-energy Huber error；
- contact-wise energy correlation；
- time-resolved rank score curve；
- STOP/recruitment calibration；
- Epilepsiae-only Stage A；
- 0.5 s seed/bin；
- strict-inductive no-heldout-calibration；
- within-shaft null。

secondary 不可替代失败的 primary gate。

## 12. Stage C：可选 event-history gain

Stage A–B 完成后，才允许把固定数目 \(K\) 的间期事件编码为：

```text
z_history = H(e_-K, ..., e_-1)
```

它只能调节 1–2 个 low-rank mode gain，不能重新训练或替换 core。主分析固定
`K`，不使用 variable padding；不够 `K` 的样本只进入明确标记的 sensitivity。

强制比较：

- ordered GRU history；
- DeepSets history；
- last event；
- reversed order；
- block-preserving shuffle；
- 同患者 correct vs wrong history–seizure pairing。

只有 History-state gate 通过，才允许出现 latent state transition 的论文表述。

## 13. 通过 gate 后的机制分析

以下分析一律在对应 gate 通过后执行：

- 将事后 A/B 模板标签投到 hidden trajectory，检验是否形成同一低维空间中的
  相反或不同初始条件轨迹；A/B 不参与 Stage-A 监督；
- 比较 interictal suffix trajectory 与 early-ictal rollout 的 canonical
  subspace overlap；
- low-rank mode lesion 和 mode gain；
- local-inhibition ablation；
- hidden-state transition/Jacobian 只作计算描述，不写成因果生理机制。

不得从单个漂亮患者、单个 seed 或训练集 latent plot 先选择 mode 再做统计。

## 14. Phase 0 输入审计和执行顺序

### Phase 0A：输入与 time-resolved cache

1. 构建 BB150 contact × time cache。
2. 完成 `[0,10] s` 聚合 parity。
3. 生成 exact common-contact 和 attrition 表。
4. 核对每患者 contact 数、event 数、pseudo-time 长度、tie 率、seed finite
   coverage、每患者 seizure 数。
5. 不查看任何模型性能，冻结最终 eligibility。

### Phase 0B：toy sanity

必须在合成数据上证明：

- 已知 forward/reverse 两种轨迹不会被平均成无序；
- tie set、STOP 和 non-participation mask 正确；
- rank shuffle 能破坏 next-contact 结构；
- static branch 冻结后 dynamic residual 不能重新吸收完整 static scaffold；
- closed-loop rollout 不读取 future target。

### Phase 1：V1 Stage A

完成 outer-fold Stage-A training、patient calibration、held-out suffix evaluation
和 Event-dynamics gate。未通过即停止。

### Phase 2：V1 Stage B

先训练/冻结 static，再训练 seed adapter 和 residual decoder，运行完整 baseline
与 Ictal-transfer gate。未通过即停止。

### Phase 3：V2 low-rank/EI

仅在 V1 Ictal-transfer 通过后运行 ranks 0–4、one-SE 选择、ablation 和
Low-rank gate。

### Phase 4：Stage C history

仅作为可选扩展，不能反过来改变 Stage A–B 的 cohort、target 或模型选择。

## 15. 预期输出

建议结果根目录：

```text
results/topic5_interictal_operator_rollout/
├── input_audit/
├── stage_a_event_operator/
├── stage_b_ictal_rollout/
├── stage_c_history_sensitivity/
├── per_subject/
├── figures/
│   └── README.md
└── run_logs/
```

正式六块图只在相应 gate 通过后生成：

1. single-event prefix → suffix operator 任务；
2. Stage-A held-out next-contact/suffix performance；
3. onset seed → frozen-core rollout 示意与真实示例；
4. patient-level model vs matched baselines；
5. time-resolved rank 与 energy transfer；
6. low-rank latent modes/lesion；若 Low-rank gate 不过，改为透明的
   rank-selection/negative panel，不画机制示意。

图目录必须在图实际生成后补中文 `figures/README.md`，并同时保存 machine-readable
cohort summary、per-subject predictions、config hash 和输入 fingerprint。

## 16. v0.1 → v0.2 的实质修改

| v0.1 | v0.2 |
|---|---|
| 32 个事件 history 是主 recurrence | 单事件内部 rank pseudo-time 是主 recurrence |
| 预测未来 8 个事件的 pair-order marginal | 预测当前事件 next set、STOP 和 suffix rank |
| 0–10 s 聚合 contact field | 1 s seed 后预测 contact × time BB150 field |
| 无 onset seed，接近 preictal pattern prediction | 短 seed 选择初始条件，回答 onset 后 rollout |
| event-indexed E/I state | within-event/early-ictal E/I surrogate |
| static/dynamic 可联合分工 | static 先训练并冻结，dynamic 只学 residual |
| GRU > DeepSets 即倾向 latent-state wording | history 降为 Stage C，须 correct-pairing gate |
| Yuquan 可混入通用自监督 | Yuquan 只进入 dataset-balanced Stage A |
| 预测所有可见 ictal contacts | exact common contacts 为 primary，外推仅 sensitivity |

## 17. 当前冻结建议

本 v0.2 建议先冻结以下三个具体选择：

1. `rank` 为 primary、`energy` 为 mandatory secondary，并保留 contact × time。
2. Yuquan 进入无标签 Stage A，同时强制 Epilepsiae-only sensitivity。
3. exact-joined common contacts 为第一版 primary。

同时冻结：

- 1 s onset seed；
- 1 s rollout bins；
- target-free held-out-patient Stage-A calibration；
- Stage C history 不参与第一版成败。

若要改变以上任一项，应在 Phase 0 看模型性能前形成 v0.3；正式训练启动后不再
通过窗口、seed 或 contact coverage 搜索改写主结论。
