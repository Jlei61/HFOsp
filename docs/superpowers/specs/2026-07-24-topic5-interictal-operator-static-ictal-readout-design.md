# Topic 5 / Figure 6：间期传播算子到静态 peri-onset 场读出（v0.3）

**版本**：v0.3
**日期**：2026-07-24
**状态**：**已被 v0.4 取代，不得据此启动新的正式训练**；已完成的 v0.3
Stage-A 结果仅保留为 engineering diagnostic
**数据裁决**：`docs/archive/topic5/fig6_ictal_target_temporal_adjudication_2026-07-24.md`

> 新版执行候选：
> `docs/superpowers/specs/2026-07-25-topic5-interictal-rank-distribution-cross-state-readout-design.md`。
> v0.4 改为 34 人 self-supervised next-set / STOP、free-running contact rank
> distributions 和不区分 A/B 的 frozen cross-state readout。

## 1. 核心问题

本分析只回答：

> 一个只从单个间期群体事件内部 contact-rank 轨迹学习的低维 recurrent
> propagation operator，能否生成患者特异的两个静态 mode fields，并在完全
> 不使用发作数据训练该 operator 的情况下，复现 clinical-onset `[0,10] s`
> BB 1–150 Hz 能量场对同一 scaffold 的表达？

主箭头：

```text
within-IED rank trajectories
    → recurrent propagation operator
    → two mode-conditioned static susceptibility/rank fields
    → clinical-onset 0–10 s BB150 static field readout
```

发作侧没有 recurrence、没有 seizure seed、没有逐秒 rollout。

## 2. 论文 parent contract

原样保留：

- Epilepsiae clinical onset；
- strict broadband 1–150 Hz；
- `[0,10] s` per-contact baseline-robust-z energy；
- TA/TB 或模型两 mode 的 polarity-free `maxAB`；
- 每次 permutation 重新选择 mode/A/B 与 mirror；
- 主 null 为 all-contact channel-label shuffle；
- within-shaft 只作 anatomy-controlled sensitivity；
- 先 seizure，再 subject，最后 cohort；
- Fit1 full-record 和 Fit2 prefix-only scaffold retention 不重新裁决。

`[0,10] s` 在本合同中是标准化静态 measurement window，不表示模式在 onset
后才出现，也不表示窗内存在传播顺序。

## 3. 输入和触点集合

发作 readout 的候选母池固定为 Fit2 prefix-scaffold 已通过的 13 名患者、71 次
strict-BB150 clinical-onset seizures。最终分母只能因预先声明的 exact-contact、
finite-target 或 input-quality gate 变化，不能因模型表现变化。

主触点：

```text
C_main
  = C_interictal_masked_rank
  ∩ C_accepted_BB150
  ∩ C_exact_channel_name_join
```

每个 interictal event 必须：

- 从 raw lag 和 `eventsBool` 重新构造 participating mask；
- 仅在参与触点内重算 rank；
- phantom ranks 完全 mask；
- tie contacts 组成同一个 recruitment set；
- 未参与触点保持 missing，不放到 rank 尾部。

第一版不输入：

- IEI、event rate、clock time；
- time-to-seizure；
- seizure seed 或任何 ictal frame；
- EEG onset；
- patient/seizure/channel string ID。

患者静态 side information 只允许：

- exact contact mask；
- centered geometry/shaft position；
- prefix-only participation support。

TA/TB 标签不作为 Stage-A 监督；只在训练完成后用于解释和对齐模型内生 modes。

## 4. 数据拆分

每个 outer fold 留一名 Epilepsiae 患者：

- shared initialization：其余 Epilepsiae patients + Yuquan interictal events；
- Yuquan 只进入无标签 Stage A，dataset-balanced 后再 patient-balanced；
- held-out 患者的 prefix IED 前 80% 可按固定 epoch 数做 target-free
  patient calibration；
- held-out 患者 prefix IED 后 20% 只做 suffix/generalization evaluation；
- held-out 患者的 seizure data 完全不参与模型、rank、checkpoint 或
  hyperparameter 选择。

Stage-A gate 完成后，可按固定训练步数用该患者全部合法 prefix IED 重拟合其
最终 operator，再生成静态 fields。该过程仍不读取 seizure data。

inner patient-level validation 只根据 Stage-A self-supervised performance 选择：

- hidden size；
- learning rate/weight decay；
- calibration epoch；
- low rank；
- loss weights。

禁止根据 ictal readout 选择 architecture 或 checkpoint。这样 primary Stage B
是真正的 zero-shot cross-domain validation。

## 5. Stage A：单事件传播学习

### 5.1 输入与输出

一个 event 的 recruitment sets 为：

```text
G_1, G_2, ..., G_L
```

随机给出 prefix `G_1 ... G_tau`，模型输出：

- next recruitment set；
- STOP；
- remaining participation；
- suffix contact utility/rank。

```text
L_A
  = L_next_set
  + λ_stop L_stop
  + λ_participation L_remaining
  + λ_rank L_suffix_rank
```

同 rank ties 用 set target；不制造任意伪顺序。

### 5.2 V1 和 baseline

V1 为 contact-query GRU：

- set token 使用 permutation-invariant pooling；
- recurrent step 对应 event 内 pseudo-time；
- contact-query decoder 保持 variable-contact mask；
- hidden size 只在 `{32,64}` 内选择。

强制 baseline：

1. empirical static TA/TB prefix；
2. first-order Markov；
3. unordered DeepSets；
4. support/participation-only；
5. within-event rank shuffle，保持 participation mask；
6. matched-capacity feed-forward contact query。

Event-dynamics gate：

> held-out patient next-set NLL 和 suffix-rank concordance 同时超过 strongest
> static/first-order baseline，并超过 participation-preserving rank shuffle。

patient bootstrap 95% CI 必须方向正确且不跨 0。未通过即停止，不进入 ictal
readout。

## 6. 从 frozen operator 生成静态 mode fields

### 6.1 两个内生 modes

对每名患者：

1. 冻结 Stage-A core；
2. Mode-recovery gate 中，只在后 20% held-out IED 上提取 event
   trajectories；对应 empirical templates 也只由这 20% 独立构建；
3. 只用这些 interictal trajectories 固定 `K=2` 聚类；
4. 不使用 TA/TB 或 seizure energy 决定聚类；
5. 聚类后才用 Hungarian matching 将两个 model modes 对齐到 held-out empirical
   TA/TB，供解释和 template-recovery gate 使用。

Mode-recovery gate 通过后，才允许按固定步数用全部 prefix IED 重拟合 operator，
生成最终 static fields；此时 full-prefix TA/TB 只用于 post-hoc 命名和 empirical
reference，不参与 mode 形成。

若 frozen operator 不能形成两个可复现 modes，该患者保留为 Stage-A 结果，但
不允许硬造 A/B field。

### 6.2 每个 mode 的输出

每个 mode \(m\in\{1,2\}\)、每个 contact \(c\) 输出：

```text
q_hat[p,m,c]   = model-predicted relative recruitment utility/rank
pi_hat[p,m,c]  = model-predicted participation probability
```

`q_hat` 是 primary timing/order field；`pi_hat` 是 participation/support
sidecar。二者必须分开保存，禁止将 participation 同时作为 value 和 smoothing
weight 双重计数。

使用与 accepted TA/TB field 完全相同的：

- contact plane；
- support rule；
- smoothing kernel；
- mirror handling；
- valid-contact mask；

将 `q_hat[p,1]`、`q_hat[p,2]` 构成一个**无序的两候选静态 field set**。

模型不输出 `contact × time`，而是：

```text
model_rank_fields          [2, contact]
model_participation_fields [2, contact]
```

## 7. 发作侧 target 和 primary score

每次 seizure 的 target 原样使用：

```text
Y_energy[p,s,c]
  = accepted clinical-onset [0,10] s BB1–150 baseline-robust-z energy

Y_rank[p,s,c]
  = within-seizure contact rank of Y_energy over C_main
```

primary paper-contract score：

```text
M_model[p,s]
  = max over {model mode 1, model mode 2, allowed mirror}
      field_similarity(model_rank_field, Y_energy[p,s])
```

每次 all-contact shuffle 都必须重新计算两 modes、mirror 和 max，承担与 observed
相同的 selection cost。within-shaft 为 sensitivity。

raw-contact Spearman/pairwise rank concordance 是强制 secondary，用于证明结果
不是仅由 2D smoothing 产生。

## 8. 三层 comparator

### 8.1 Accepted empirical reference

empirical prefix TA/TB `maxAB` 是 parent benchmark/ceiling。模型不要求显著
超过它；模型要证明的是低维 operator 可以在 held-out IED 上恢复并保留该读出。

冻结一个非劣界值后检验：

```text
M_model - M_empirical_TATB > -δ
```

`δ` 必须在正式 readout 前由 empirical split-half variability 冻结，不能根据
模型结果选择。

### 8.2 Operator-specific controls

- participation-preserving within-event rank-shuffle core；
- support-only field；
- first-order Markov-derived fields；
- random frozen core；
- V2 rank-0 core。

模型的 ictal readout 必须超过 rank-shuffle core 和 support-only field，才能说明
结果包含 event order/operator 信息，而不只是 HFO-rate topography。

### 8.3 Optional energy calibration

normalized energy magnitude 只作 secondary。若需要，从：

```text
(q_hat, pi_hat, geometry) → Y_energy
```

训练一个 outer-training-patient-only 的 Ridge/monotonic contact-query readout。
不使用深 ictal decoder，不允许该辅助分支决定 operator rank 或 primary gate。

## 9. 四道科学门

| Gate | 条件 | 允许结论 |
|---|---|---|
| Event dynamics | V1 超过 static/Markov/rank-shuffle | 单个 IED 内存在可学习的递归传播规律 |
| Mode recovery | 两个 model modes 在 held-out IED 上恢复 TA/TB rank fields，超过 shuffled core | operator 生成两个稳定的患者传播 modes |
| Static ictal readout | model fields 超过 all-contact null、rank-shuffle core 和 support-only，并对 empirical TA/TB 非劣 | IED operator 的静态 mode fields 在 peri-onset BB150 场中被表达 |
| Low rank | rank 1–3 在 one-SE 下匹配 V1、超过 rank 0，mode lesion 同时破坏 IED recovery 和 ictal readout | 跨状态共享结构可由少数 recurrent modes 描述 |

任何 gate 失败都保留其上游结果，但不得跳级解释。

## 10. V2 low-rank E/I core

只有 V1 通过前三道门后才运行 V2。

V2 的 recurrent step 仍只对应 IED 内 pseudo-time：

```text
E_next = leak_E E + phi_pos(W_EE E - W_EI I + input)
I_next = leak_I I + phi_pos(W_IE E - W_II I)
```

- local term 由冻结的同杆邻近/空间 kernel 构造；
- non-local term 为 rank `r ∈ {0,1,2,3,4}`；
- rank 0 是必要 control；
- 不把该 E/I state 延伸到 seizure seconds；
- 不声称真实细胞类型数量或真实生理时间常数。

mode lesion 必须同时降低：

1. held-out IED suffix/template recovery；
2. static ictal field readout。

只降低其中一个不能称为跨状态共享 mode。

## 11. 时间窗只作冻结 sensitivity

模型 static fields 训练完成后，原样评估：

- primary：clinical onset `[0,10] s` strict BB150；
- sensitivity：`[0,5]`、`[5,10]`、`[-10,0]`；
- negative-control/context：distal `[-120,-90]`。

所有窗口使用同一 frozen model fields，不重新训练、不重新选 rank、不选择最显著
窗口。

若 pre/onset/post 均相近，结论是 persistent patient-specific scaffold。只有
独立、预注册的 post > pre 检验通过，才允许讨论 onset amplification；仍不能
讨论接触点顺序重放。

## 12. 统计折叠

```text
每个 seizure 得到 M_model 与各 comparator
→ 患者内 seizure median
→ training random seed 在患者内 median
→ cohort patient-level paired statistic
```

- Wilcoxon 和 patient bootstrap CI；
- training random seed、seizure、contact、mode 都不是独立 N；
- all-contact null 为 primary；
- within-shaft 为 sensitivity；
- model vs empirical 使用 paired non-inferiority；
- model vs shuffled/support 使用 paired superiority。

## 13. 训练器

- PyTorch + AdamW；
- gradient clipping 1.0；
- Stage A max 200 epochs、inner-patient early stopping；
- 三个冻结 seeds；
- one-SE 选择最小 hidden size/最小 rank；
- dataset-balanced → patient-balanced → event sampling；
- 每个 cell 保存 config、input fingerprint、checkpoint、epoch log、
  predictions、resource log、`DONE.json`；
- OOM 只能调整 batch/token budget，不能改 cohort 或 target。

## 14. 执行顺序和 stop rule

### Phase 0

- exact contact/order audit；
- event tie/mask audit；
- accepted BB150 aggregate artifact fingerprint；
- synthetic forward/reverse mode sanity；
- participant-preserving rank-shuffle sanity。

### Phase 1

V1 Stage A + Event-dynamics gate。失败即停止。

### Phase 2

生成 two mode fields + held-out IED Mode-recovery gate。失败即停止。

### Phase 3

锁定所有 model outputs 后，首次读取 ictal target，运行 Static-ictal-readout
gate。禁止根据结果回改 operator。

### Phase 4

仅前三门通过后运行 V2 low-rank/EI 和 mode lesion。

### Phase 5

冻结窗口 sensitivity；event-history latent state 不属于 v0.3 主分析。

## 15. 六块图候选

1. 单个 IED rank prefix → suffix/STOP 任务；
2. held-out IED next-contact 与 suffix-rank performance；
3. 两个内生 latent trajectories 及其 model-derived static fields；
4. model modes 与 empirical TA/TB 的 held-out recovery；
5. model static fields vs clinical-onset 0–10 s BB150，配对 Data–Null；
6. low-rank selection + mode lesion；若 low-rank gate 失败，透明展示 negative
   rank result，不画机制图。

图目录必须在出图后写中文 `figures/README.md`。

## 16. 允许和禁止措辞

允许：

> A low-dimensional recurrent operator learned from within-event interictal
> trajectories generated two stable patient-specific mode fields whose static
> spatial expression was retained around clinical seizure onset.

禁止：

- seizure rollout；
- ictal contact-order replay；
- onset-emergent recruitment；
- preictal seizure forecasting；
- 相同物理时间尺度；
- causal E/I seizure mechanism。

## 17. v0.2 → v0.3

| v0.2 | v0.3 |
|---|---|
| 0–1 s seizure seed | 无 seizure input |
| 预测 1–10 s 九个 time bins | 预测两张 `[2, contact]` static mode fields |
| IED pseudo-time core 在 seizure seconds rollout | recurrence 只存在于 IED 内 |
| contact × time rank 主 target | accepted 0–10 s BB150 static field |
| static + dynamic residual decoder | zero-shot operator-derived field readout |
| 试图证明 temporal reuse | 检验 patient-specific static scaffold expression |
| 容易误写成 replay | 明确禁止 replay/onset-emergent wording |
