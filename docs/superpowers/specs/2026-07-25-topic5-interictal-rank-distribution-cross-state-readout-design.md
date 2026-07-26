# Topic 5 / Figure 6：间期自监督 rank 分布与冻结跨状态读出（v0.4）

**版本**：v0.4
**日期**：2026-07-25
**状态**：新版执行候选，待实现前审阅
**取代范围**：取代 v0.3 中的 direct suffix、K=2 mode recovery 和
mode-conditioned ictal readout；v0.3 已完成结果保留为诊断，不覆盖、不删除
**发作 target 裁决**：
`docs/archive/topic5/fig6_ictal_target_temporal_adjudication_2026-07-24.md`

---

## 1. 核心科学问题

本分析只回答两个连续问题：

1. 只看单个间期群体事件内部已经发生的 recruitment sets，RNN 能否预测
   **下一个 rank set 或事件结束**，并在自由生成时恢复每个触点的参与概率和
   完整 rank 分布？
2. 这些在发作数据完全不可见时学到并冻结的患者特异触点分布，能否直接解释或
   预测 clinical onset 附近的静态发作早期能量场？

主箭头固定为：

```text
all eligible masked interictal events
    → self-supervised next-set / STOP learning
    → frozen patient-specific contact rank distributions
    → zero-shot early-ictal static field comparison
    → optional outer-patient-only simple readout
```

本合同不要求模型学习 A/B 标签。A/B 只是经验 rank 分布中可能自然出现的主要
结构，不是监督目标、架构参数、患者选择门或发作 readout 的 max 操作。

---

## 2. 与论文主线的关系

目标不是用深度模型重新做 KMeans，而是检验：

> 大量间期群体事件中反复出现的触点级 recruitment probability 与 rank
> distribution，是否构成一个可学习、可生成、并能跨到发作早期静态能量场的
> 患者特异病理组织。

成功时最多允许写：

> A self-supervised recurrent model recovered patient-specific contact
> recruitment-rank distributions from interictal population events, and the
> frozen distributions retained predictive information about the static
> early-ictal energy field.

不能写：

- RNN 预测了发作时间；
- 发作按间期触点顺序逐个重放；
- A/B 是真实离散神经状态；
- 模型证明了致痫因果机制；
- pseudo-time 与发作秒级时间具有相同物理意义。

---

## 3. 队列合同

### 3.1 Stage A 间期主队列：34 人

主队列固定读取：

```text
results/topic4_attractor_masked/step0_audit.csv
    where eligible_for_main == true
```

冻结结果：

- 34 名患者；
- 18 Epilepsiae；
- 16 Yuquan；
- 每名患者至少 6 个 union contacts；
- 在既有审计定义下至少 100 个 eligible events。

这里使用 34 人是为了沿用当前论文的合格主队列和数据质量门。`stable_k=2`、
cluster label、TA/TB label、template rank 和 attractor metric **均不输入模型**，
也不作为 v0.4 的成功条件。

其余 6 人不进入 primary 34 人统计。可在主分析冻结后作为 generalization
sensitivity，不能用于改变模型或阈值。

### 3.2 Stage B 发作子队列

发作侧候选母池继续使用已冻结的：

- 13 名 Fit2 prefix-scaffold 患者；
- 71 次 strict-BB150 clinical-onset seizures。

13 人只是 34 人中的发作读出子集，不再决定 Stage-A 模型选择或间期结论。
最终发作分母只能因预先声明的 exact-contact、finite-target 或 input-quality
gate 变化，不能因模型表现变化。

### 3.3 两层数据范围

- **Stage-A distribution learning**：34 人各自全部合格 definite-interictal
  events。
- **Stage-B zero-shot primary**：患者自身冻结的 prefix-only interictal
  distribution；full-record distribution 只作 Fit1-aligned sensitivity。

这样既不浪费 34 人间期数据，也不让发作后的间期事件进入 primary
prefix-to-ictal 检验。

---

## 4. 间期事件输入

### 4.1 事件构造

每个事件从 raw lag、`eventsBool` 和 exact channel order 重新构造：

```text
G_1, G_2, ..., G_L
```

其中 `G_t` 是同一 rank 的新 recruitment contact set。

硬规则：

- `eventsBool` 定义参与；
- 只在参与触点内重新 rank；
- phantom finite ranks 完全屏蔽；
- 非参与触点保持 missing，不放到 rank 尾部；
- exact ties 保持 set，不任意拆序；
- primary event 至少 3 个参与触点、至少 2 个 recruitment sets；
- 34 人资格审计使用的 `n_participating >= 6` 事件门保留为 sensitivity，
  不用于删除主模型中仍有合法顺序信息的 3–5-contact events。

主数据不得输入：

- IEI 或跨事件历史；
- event rate、clock time、block ID；
- time-to-seizure；
- seizure seed、clinical/EEG onset 或 ictal energy；
- A/B、KMeans label；
- patient、seizure 或 channel string ID。

### 4.2 触点表示

共享 contact encoder 只允许读取：

- exact contact mask；
- centered geometry；
- within-shaft position；
- prefix-only participation support；
- geometry-missing indicator。

另外允许每名患者有一个低维 **patient-local contact offset**：

- 只在该患者前 80% 间期事件上拟合；
- 不按字符串 ID 跨患者共享；
- 后 20% 评估前冻结；
- matched static baseline 必须拥有完全相同的 local offset 容量。

local offset 用于表达患者内部每个触点不同的静态 recruitment propensity。
RNN 是否有价值只能由它在相同 local offsets 之上利用有序 prefix 所带来的增量
决定。

---

## 5. 数据拆分与泄漏控制

### 5.1 34 人外层评估

Stage A 使用 leave-one-subject-out：

- shared training：其余 33 人；
- held-out patient calibration：该患者按时间前 80% 的间期事件；
- held-out evaluation：该患者按时间后 20% 的间期事件；
- 后 20% 不参与 checkpoint、epoch、hidden size 或 loss 选择。

shared training 先 dataset-balanced，再 patient-balanced。每个事件在 loss
折叠时等权，不能让 recruitment steps 多或事件数多的患者主导训练。

训练 loader 必须为每名患者维护 without-replacement event queue。一个
coverage cycle 内，该患者前 80% 的全部合法 events 各进入一次；第一个完整
coverage cycle 完成前不得 early stop 或选择 checkpoint。患者等权通过 loss
weight 实现，不能通过丢弃高事件数患者的大量合法 events 实现。

### 5.2 held-out 患者适配

Primary 只允许在 held-out 患者前 80% 上更新：

- patient-local contact offsets；
- patient-local initial state / small adapter。

共享 GRU core 保持冻结。

允许一个 sensitivity 在前 80% 上微调整个 core，但不能替代 primary。

### 5.3 超参数选择

inner patient-level validation 只使用间期 self-supervised loss，选择：

- hidden size；
- learning rate、weight decay；
- local-offset dimension；
- calibration steps；
- optional low rank。

发作 target 在 Stage-A 和 distribution gate 完成、所有模型输出和 fingerprint
冻结之前不得读取。

---

## 6. 自监督训练任务

### 6.1 唯一主任务：next set / STOP

对一个完整事件，模型依次读取：

```text
empty → G_1 → G_2 → ... → G_L
```

在每一步预测：

- 下一个 recruitment set；
- 或 `STOP`。

尚未出现的所有真实记录触点持续作为候选。模型选择 `STOP` 后仍未出现的触点，
自然成为该生成事件的 non-participants。

因此 v0.4 删除：

- direct suffix-rank head；
- oracle-conditioned suffix concordance；
- separate remaining-participation head；
- K=2 mode classification loss。

### 6.2 全事件训练

每个事件的所有合法 prefix 都进入同一次 sequence loss，而不是每次只抽一个
随机 `tau`：

```text
L_event
  = mean over steps [
        L_next_set_or_STOP
    ]
```

再按：

```text
mean over events
→ mean over patients
→ mean over datasets
```

折叠。

这样每个事件等权，长事件不会因为 step 多而获得更大权重。

若一个 rank 含多个 exact-tie contacts，使用 permutation-invariant set
likelihood；不得拆成任意 contact 顺序。

### 6.3 V1 模型

V1 保留最小 contact-query GRU：

- 每个 recruitment set 先做 permutation-invariant pooling；
- GRU 只沿单个事件的 pseudo-time 更新；
- contact-query decoder 对尚未出现触点输出 next-set probability；
- 独立输出 STOP probability；
- hidden size 候选仅 `{32, 64}`；
- recurrence 不跨事件。

第一版不加入 Transformer、LSTM、离散 A/B latent 或深 decoder。

---

## 7. 强制对照

所有模型使用同一触点输入、同一 patient-local offsets、同一训练预算和同一
held-out events。

至少包括：

1. **Empirical rank distribution**：前 80% 直接估计每个触点的参与概率和
   rank histogram；
2. **Static contact hazard**：知道患者和触点，但不看 event prefix；
3. **Unordered prefix**：知道哪些触点已经出现，但不知道出现顺序；
4. **Last-set / first-order**：只看最近一个 recruitment set；
5. **Full-history GRU**：读取完整有序 prefix；
6. **Participation-preserving within-event rank shuffle**。

如果 full-history GRU 不能超过 1–4 中最强对照，不能用“模型能拟合真实数据”
替代 recurrence-value 结论。

---

## 8. 从 RNN 得到触点 rank distribution

### 8.1 自由生成

Stage-A checkpoint 冻结后，每名患者、每个 seed：

1. 从 empty prefix 开始；
2. 按模型概率采样下一个 set 或 STOP；
3. 已出现触点立即 mask，禁止重复；
4. STOP 或达到 contact 数上限时结束；
5. 生成固定 5,000 个 events。

这些是 free-running rollouts，不使用真实 prefix teacher forcing。

### 8.2 每个触点必须保存的量

对患者 \(p\)、触点 \(c\)：

```text
pi[p,c]
  = P(contact c participates)

p_rank[p,c,b]
  = P(normalized rank in bin b | participates)

mean_rank[p,c]
rank_variance[p,c]
rank_quantiles[p,c,{0.1,0.5,0.9}]
```

non-participation mass 与 conditional rank distribution 分开保存，禁止把
non-participation 当成 late rank。

主 rank grid 固定为 10 个 normalized bins；5 和 20 bins 作冻结 sensitivity。
模型训练不依赖 binning，bin 只用于保存、统计和画图。

### 8.3 多路径表示

不强制聚成 A/B。模型生成结构用两种无标签量描述：

- event × contact rollout-rank matrix；
- pairwise precedence matrix
  `P(contact i precedes contact j | both participate)`。

若 A/B 是主要结构，它们可以在这些输出中自然出现；若存在连续谱或更多路径，
也必须保留。

---

## 9. Stage-A 评估与科学门

### 9.1 一步预测

Primary：

```text
held-out next-set / STOP negative log-likelihood
```

GRU 与每个对照逐患者比较。辅助报告：

- top-k next-set accuracy；
- STOP calibration；
- event-length calibration。

### 9.2 分布恢复

在 held-out 后 20% 上比较真实与 free-running model distributions：

1. participation probability error；
2. 每触点 rank distribution 的 1-Wasserstein distance；
3. pairwise precedence matrix error/correlation；
4. participant-count 和 event-length distribution error。

所有 contact metric 先在患者内折叠，再做 34 人统计。

### 9.3 两道 Stage-A gate

| Gate | 条件 | 允许结论 |
|---|---|---|
| Next-step gate | GRU 的患者级 NLL 优于 strongest non-recurrent control，bootstrap 95% CI 方向正确且不跨 0 | 有序 prefix 提供局部下一招募信息 |
| Distribution gate | GRU rollout 明显优于 rank-shuffle，且相对 empirical rank distribution 达到预先冻结的 non-inferiority margin | RNN 可自由生成真实的患者触点 rank 分布 |

non-inferiority margin 必须由前 80% 的 split-half empirical variability 冻结，
不能根据 GRU 结果决定。

Stage-A gate 失败时：

- 若 empirical distribution 本身稳定，保留经验 rank 分布进入发作读出；
- RNN 分支停止，不进入 low-rank / E-I；
- 不把 RNN 阴性升级为论文核心假设阴性。

---

## 10. Stage B：发作数据如何进入

### 10.1 发作 target

发作侧保持 parent contract：

```text
Y_energy[p,s,c]
  = clinical-onset [0,10] s
    line-noise-masked BB 1–150 Hz
    per-contact baseline-robust-z energy
```

`[0,10] s` 是静态 measurement window，不表示窗内存在接触点传播顺序。

exact common contact set：

```text
C_main
  = C_frozen_interictal_distribution
  ∩ C_accepted_BB150
  ∩ C_exact_channel_name_join
```

### 10.2 Primary：零参数冻结比较

在读取任何发作 target 前，从 prefix-only interictal rollouts 冻结：

```text
S_rank[p,c] = 1 - mean_rank[p,c]
S_part[p,c] = pi[p,c]
```

primary rank readout 是 `S_rank` 与每次 seizure `Y_energy` 的逐触点
rank/field similarity。`S_part` 作为独立 participation control，不与
`S_rank` 相乘，不让参与率与顺序贡献混在一起。

每次 seizure 得到一个 score：

```text
seizure score
→ patient median
→ cohort patient-level statistic
```

主 null：

- all-contact channel-label shuffle。

within-shaft shuffle：

- anatomy-controlled sensitivity。

所有 permutation 都保持 ictal target、触点数和 observed score 计算完全同构。

### 10.3 RNN 是否增加价值

同一发作 target 必须同时比较：

1. RNN rollout rank distribution；
2. 前 80% empirical rank distribution；
3. participation-only field；
4. geometry-only field；
5. rank-shuffle RNN。

只有 RNN readout 高于 empirical distribution，才能说 RNN 提供了增量。

若 RNN 与 empirical 相当，而两者都超过 null，安全结论是：

> 间期触点 rank distribution 跨状态保留；RNN 没有提供必要的额外增益。

### 10.4 Secondary：简单跨患者 readout

可选的 learned readout 使用：

```text
(pi, 10-bin rank distribution, rank variance, geometry)
    → Y_energy
```

限制：

- RNN 完全冻结；
- 只允许 Ridge 或单调线性 readout；
- outer-training patients 训练；
- held-out patient 的任何 seizure 不参与训练、早停或归一化；
- 不能让 learned readout 反向决定 Stage-A 模型或 checkpoint；
- 不能使用深 ictal decoder。

这属于 frozen cross-state transfer。用 ictal data 微调整个 RNN 只可作为
exploratory sensitivity，不能支持“间期结构被重用”的主结论。

---

## 11. 发作侧科学门

| Gate | 条件 | 允许结论 |
|---|---|---|
| Frozen zero-shot | frozen `S_rank` 超过 all-contact null，并高于 rank-shuffle / participation-only | 间期 rank 分布在发作早期静态场中被表达 |
| RNN incremental | RNN rollout readout 患者级优于 empirical rank distribution | recurrent conditional structure增加跨状态信息 |
| Simple transfer | outer-patient-only simple readout 优于 geometry / participation / empirical baselines | 完整间期 rank distribution 可跨患者预测发作早期能量 |

如果只有 ictal-fine-tuned RNN 有效，主 gate 仍判失败。

---

## 12. Low-rank / 生物约束模型

只有以下全部通过后才启动：

1. 34 人 next-step gate；
2. 34 人 distribution gate；
3. frozen zero-shot gate；
4. RNN incremental gate。

V2 与 V1 使用完全相同的 next-set/STOP 任务、split、contact offsets、baselines
和 readout，只替换 recurrent core。

第一版 rank 候选 `{0,1,2,3,4}`：

- rank 0 是必要 control；
- one-SE 选择最小 rank；
- 不预设两个 modes；
- 不把低 rank 等同于 A/B；
- 不把 surrogate E/I state 解释为真实细胞类型恢复。

low-rank claim 必须同时要求：

- held-out next-step 不劣于 V1；
- rollout distribution 不劣于 V1；
- ictal zero-shot readout 保留；
- mode/rank lesion 同时损害间期分布和跨状态 readout。

---

## 13. 执行顺序与资源门

### Phase 0：v0.4 数据审计

- 34 人 cohort exact match；
- channel order / phantom mask；
- full eligible event pool 与 prefix-only pool 分开；
- chronological 80/20；
- non-participation 不进入 rank bins；
- exact-tie set likelihood synthetic test；
- free-rollout STOP / no-repeat synthetic test；
- ictal target 保持 sealed。

### Phase 1：三患者收敛 pilot

- 选择低、中、高事件数各一名患者；
- 检查 full-event loss 是否下降；
- 检查 STOP 不塌缩；
- 检查 rollout event length、参与数不过早归零或跑满；
- 检查无 OOM、NaN、显存泄漏。

只回答工程可用性，不做科学 gate。

### Phase 2：34 人 × 1 seed cheap screen

必须跑全 baseline 和 free rollout。

继续到正式三 seed 的最低条件：

- next-step gain 对 strongest non-recurrent 的患者中位数 > 0；
- rank-shuffle 分布明显更差；
- RNN rollout 对 empirical distribution 的差异不超过预冻结 non-inferiority
  margin；
- 无系统性 STOP / participant-count calibration failure。

方向不满足即停，不用增加 seeds 挽救。

### Phase 3：34 人 × 3 seeds 正式 Stage A

- patient-level bootstrap；
- seed 先在患者内折叠；
- 冻结全部模型、rollouts、distribution artifacts 和 fingerprints。

### Phase 4：首次读取发作 target

- zero-shot direct rank readout；
- empirical / participation / geometry / shuffle controls；
- 13 人候选子集按真实 eligible 分母统计；
- 不因结果修改 Stage A。

### Phase 5：simple cross-patient readout

只在 Phase 4 输出锁定后运行。不能替代 zero-shot。

### Phase 6：low-rank / E-I

仅四个前置 gate 全通过时运行。

---

## 14. 训练器与可复现性

- PyTorch + AdamW；
- gradient clipping 1.0；
- hidden size `{32,64}`；
- 三个冻结 seeds；
- inner patient-level early stopping；
- dataset-balanced → patient-balanced → event-balanced；
- full-event loss 在 event 内先平均；
- 每个 run 保存：
  - config；
  - source/input SHA256；
  - checkpoint；
  - epoch log；
  - held-out step predictions；
  - free rollouts；
  - per-contact distributions；
  - resource log；
  - `DONE.json`。

OOM 只允许调整 batch/token budget，不允许改变 cohort、event gate、target 或
评价单位。

建议新路径：

```text
config/topic5_interictal_rank_distribution_v0_4.yaml

results/topic5_interictal_rank_distribution/
    dataset_v0_4/
    runs/
    stage_a/
    cross_state_readout/
```

v0.3 目录保持只读历史：

```text
results/topic5_interictal_operator_static_readout/
```

---

## 15. Figure 6 六块正式图合同

正式图只使用真实数据或真实模型输出，不再用大面积流程框图。

### A. 单事件逐步预测

- 一个 held-out 真实事件；
- 每一步显示已出现触点、模型下一个触点概率和真实下一 set；
- 科学问题：模型是否根据完整 prefix 更新预测。

### B. 真实与 RNN 触点 rank distributions

- representative patient；
- 行为固定 contact order；
- 列为 normalized rank bins；
- 独立 non-participation column；
- 同一 panel 并列 `observed | RNN | RNN − observed`；
- 科学问题：RNN 自由生成是否恢复每个节点的真实 rank distribution。

### C. 无标签的多路径结构

- representative patient 的 free-rollout event × contact rank matrix；
- events 只按连续的 rank similarity 排序，不强制 A/B；
- 可并列 pairwise precedence matrix；
- 科学问题：模型分布中是否自然出现多条传播路径。

### D. 空间节点读出

- contact color = conditional mean rank；
- point size = participation probability；
- uncertainty ring = rank spread；
- 科学问题：RNN 学到的节点分布如何落在患者真实电极空间。

### E. 34 人分布恢复

- 34 人 next-step gain；
- 34 人 rollout-to-heldout distribution distance；
- RNN、empirical、unordered、last-set 和 rank-shuffle 使用同一患者级统计语法；
- 科学问题：RNN 是否在主队列中稳定恢复间期 rank distribution。

### F. 冻结间期分布到发作早期场

- 同一患者 interictal `S_rank` 与 clinical-onset BB150 energy field；
- cohort Data–Null；
- RNN vs empirical distribution 的患者级 paired comparison；
- 科学问题：跨状态保留是否存在，RNN 是否增加价值。

每个 panel 只回答一个问题。正式脚本放：

```text
scripts/paper_figures/
```

正式产物放：

```text
results/paper-ready-figure/
    fig6_interictal_rank_distribution_cross_state/
        figures/
            README.md
```

当前
`results/paper-ready-figure/fig6_stagea_operator_screen/`
降级为 v0.3 engineering diagnostic，不作为 v0.4 paper figure。

---

## 16. v0.3 → v0.4 变更摘要

| v0.3 | v0.4 |
|---|---|
| 13 人决定 Stage-A screen | 34 人为 Stage-A 正式主队列 |
| 随机一个 prefix `tau` | 每个事件全部 steps 进入 loss |
| next + STOP + remaining + direct suffix | 只保留 next set / STOP |
| suffix 排序时使用真实未来参与集合 | non-participation 由模型 STOP 自然产生 |
| K=2 mode recovery | 无标签 free-rollout rank distribution |
| A/B-conditioned fields | 每触点 participation +完整 rank distribution |
| mode `maxAB` ictal readout | 不分 A/B 的 frozen rank-field readout |
| RNN 与静态模型只比局部指标 | 同时比 next-step、free-rollout distribution 和 ictal增量 |
| suffix gate 阴性阻断整条线 | suffix 任务撤销；由 next-step + distribution gates 决定 |
