# Group-Event State v0.3.5 执行计划

## Dynamic baseline, full-event state and step-wise frozen-decoder modulation

**状态：** `ACTIVE_FULL_EXECUTION`  
**日期：** 2026-09-03  
**Spec：** `group_event_state_v0_3_5_dynamic_baseline_state_spec_2026-09-03.md`

> **2026-09-04 执行修订：**observed-support 覆盖合同保持不变，但训练分成两层。各 horizon 的计数基线独立训练，只负责 L0 难度/可估性；完整状态不再按 horizon 各训一套。下一阶段必须训练一个跨 2/6/8 h 共享并冻结的 `S_N` producer，以及一个跨 next-event 与 30 min/2 h/6 h grammar 目标共享并冻结的 `S_G` producer。horizon-specific evaluator/head 仍可独立。原 long supervisor 的 per-horizon full-state 阶段已在 0 个 GPU 作业启动时停用；84 个计数基线保留。

## 0. 执行原则

- W0–W6 均为本 goal 的承诺范围；不得用 MVP、接口 smoke、synthetic 或单患者结果宣布整体完成。
- 每个工作包在机器 scope manifest 中只能标为 `PENDING/RUNNING/PARTIAL/COMPLETE/NOT_ESTIMABLE`；`PARTIAL` 与 `NOT_ESTIMABLE` 都不等于整体完成。

- 先把静态偏置扩展成可因果追踪的动态基线，再判断 event-content state 是否增加信息。
- 先修 contact decoder 的状态接口，再扩大患者；当前 `h0-only` 阴性不外推到逐步调制。
- rate state 是合法科学结果，不把它当 nuisance 消掉；但形态状态必须报告相对 rate state 的增量。
- 训练、选择、评价按真实时间分离；不再用同一 selection 子样本选 checkpoint 又报结果。
- 少 gate、多探索；灵敏度不足只限制该 patient×endpoint 的解释。

## Phase 0：冻结旧结果并建立可估性表

### 任务

1. 将 v0.3.4 WE decoder/state 结果冻结为 `h0_only_diagnostic`。
2. 对当前候选患者读取 Topic 2 的 5 min rate trace、连续 coverage 与 0.5/1/2/4/8 h autocorrelation。
3. 为 5 min、30 min、2 h、6–8 h 生成 per-patient 完整 future-block 和有效独立块表。
4. 明确每位患者 decoder calibration、state TRAIN、rolling-inner、selection、development 的时间边界；任何 future target 均不得跨界。

### 首轮患者

优先使用：

- E253：长连续记录、44 个独立 30 min WE 评价窗；
- E922：rate drift 强、事件多，但必须先解决 WE 评价覆盖空洞；
- E1096：0.5–8 h rate autocorrelation 很强，适合动态 baseline；
- E548：现有 oracle 可检出、30 h 连续段；
- E583：较低事件数但 0.5–8 h rate autocorrelation 持续；
- E1146：保留为 assay-limited 对照，不承担阴性结论。

E384、E1125 作为第二波覆盖/异质性患者。患者进入某个 horizon 只依据记录支持，不依据结果方向。

### 交付

- `timescale_estimability.csv`
- `split_and_decoder_provenance.json`
- 一张 rate trace + static mean + causal multiscale baseline 诊断图

## Phase 1：动态负荷基线 `q(t)`

### 1.1 先做确定性强基线

在 1 min 有效观测网格构造因果特征：

- time since last event；
- rate EWMA：2/10/30 min、2/8 h；
- extent/STOP 与参与负荷的同尺度 EWMA；
- valid exposure、clock、session position；
- IED-core-masked 连续背景 SEEG 的低容量辅助摘要/embedding，并保留 event-time-only 嵌套臂。

以患者级负二项 count model 预测 5/30/120 min future count。静态患者截距由 calibration/TRAIN 早段拟合；所有动态项只使用 anchor 之前的信息。

### 1.2 再做小型 learned residual

在确定性 baseline 上加小型 residual state-space update。搜索只覆盖：

- residual depth 1/2；
- width small/medium；
- learning rate 三档；
- residual gate 初始化两档。

实现锁定为六个有界配方：默认、浅层、宽层、低学习率、高学习率和较开放 gate。八位登记患者各跑三个 seed；配方只用 FIT/INNER 的按患者秩选择，SELECTION 在配方冻结后只打开一次。若选中配方改变 event-time baseline，则 background residual 必须在该配方上重训，不能沿用 provisional base 的结果。

不搜索自由 tau；时间 bank 固定。每个患者至少 3 seeds，E253/E922/E1096 使用 5 seeds。

### 1.3 决定性比较

同一 anchor、同一 likelihood 比较：

```text
static calibration
causal multiscale baseline
causal baseline + learned residual
block-shifted dynamic baseline
```

主要输出不是“是否显著”的单开关，而是每位患者在 5/30/120 min 的 out-of-time count skill、校准曲线和有效独立块区间。

### 完成定义

- 静态模型是动态模型所有时变权重为零时的逐位特例；
- 缺失区间不被计作 silence；
- 不同患者的 raw count 经 exposure/dispersion 标准化，不共享无意义的绝对尺度；
- selection 只选超参数，报告期不再选模型。

## Phase 2：冻结 contact decoder 的逐步状态 adapter

### 2.1 保持 decoder 合同不变

使用 v0.3.4 已按真实时间重训的成熟 contact-sequence decoder；不在本阶段切换 exact subset objective。先完成：

- 零 adapter 输出逐位等于 frozen decoder；
- 旧 `h0-only` 路径逐位复现；
- 所有 decoder 参数 `requires_grad=False`，但输入状态梯度非零。

### 2.2 实现三个逐步调制点

1. per-node hidden FiLM；
2. contact-specific low-rank logit shift；
3. independent continue/STOP shift。

主实验使用三者组合的低秩 adapter。`edge_gate` 只作后续 sensitivity，不进入首轮主搜索。

### 2.3 容量等价的训练阶梯

1. `static_step_adapter`：同一 adapter 输入训练期常数；
2. `rate_step_adapter`：输入 `q(t_e-)`；
3. `rate_mark_step_adapter`：输入 `[q(t_e-),m(t_e-)]`；
4. `block_shift`；
5. `future_oracle`。

每条动态臂都从同一个已收敛 static adapter 开始，避免 state 替未校准 decoder 补均值。

### 完成定义

- 状态对每一步 contact logits 与 STOP logits 都有非零、contact-specific 的 Jacobian；
- 相同状态在不同 prefix step 可产生不同调制；
- future-oracle 在 E253/E548/E583 至少保持当前灵敏度，不因接口改写而丢失；
- 人体动态结果无论方向都进入报告。

## Phase 3：完整事件内容状态 `m(t)` 与多 horizon 训练

### 3.0 共享状态合同

Phase 3 不再从 `observed_support_2h/6h/8h/12h` 分别产生四套状态。每位患者登记：

1. 一个共享 `S_N` producer checkpoint；
2. 一个共享 `S_G` producer checkpoint；
3. 两条从记录起点因果 replay 得到的冻结轨迹；
4. 每个 horizon/endpoint 自己的轻量 evaluator checkpoint。

producer 的训练样本使用各 horizon 自己的 FIT mask，但 checkpoint 由标准化后等权的 multi-horizon INNER objective 统一选择一次；任何 horizon 的 SELECTION 都不能参与 producer 选择。若患者只支持 2 h 与 6 h，它仍可训练一个共享 producer，但不能据此宣称 8 h persistence。跨尺度结论只来自同一冻结轨迹在多个真实 horizon 的 held-out 增量。

### 3.1 输入接口

完成 full-event token：participation、tied groups、连续 lag、bipolar/CAR 波形 embedding、per-contact multiband energy/peak/cross-band lag、contact scaffold 和 mask。

### 3.2 训练目标

联合但分项记录：

- future count：5/30/120 min；
- next 1/5/20 events 的 contact sequence 与 STOP；
- future 5/30/120 min conditional contact-sequence score；
- lag/energy/waveform heads 在 Phase 4 接入前先保留独立 loss 字段，不让缺失 target 改变 contact 主损失权重。

训练拆成两个并行目标族：

- `S_N`：未来 count/silence 的 5 min、30 min、2 h、6 h、8 h 多尺度 likelihood；
- `S_G`：条件于未来块有事件及其 event count 后，预测 local continue/STOP、positive group size、contact subset、later continuation、multiband expression，以及 community occupancy、cross-community coupling 和 repertoire mixture。

community 与 repertoire dictionary 只用 calibration/FIT 构造并冻结。块级目标同时提供离散 mixture 与连续 embedding-distribution 两种读出，避免结论只由某个聚类数决定。

anchor 后的真实事件不能写回 cross-event state；只允许 frozen within-event decoder 看目标事件自己的早期 prefix。

### 3.3 训练与搜索

- session-preserving chronological replay；
- next-1/5/20-event 读出从 anchor post-state 出发，按目标事件的精确真实 `dt` 开环演化，中间事件不 teacher-force；
- chunk 边界 carry state 后 detach，不 reset；
- chunk 同时受最大真实时长与最大事件数限制；
- burn-in 只重建状态，不计 loss；
- state/adapter 学习率、encoder depth/width、residual blocks 与初始化做充分搜索；
- pilot 每患者 3–5 seeds；OOM 时降低 batch/concurrency，不改科学输入与 target。

### 3.4 核心解释

分别报告：

- `q(t)` 对 count 的解释；
- `q(t)` 是否同时解释 contact morphology；
- `m(t)` 在 `q(t)` 之外的 conditional morphology 增量；
- correct-time 相对 block-shift；
- 增量随 1/5/20 events 与 5/30/120 min 是否持续。
- 同一 `S_N` 是否跨 2/6/8 h 保留 burden 增量；
- 同一 `S_G` 是否表现为稳定局部 grammar 上的 community occupancy、coupling 或 repertoire mixture 慢变化；
- `S_N -> grammar`、`S_G -> burden` 与 `S_N+S_G` 的 cross-transfer，避免把两个名字当成未经验证的解耦。

## Phase 4：H2a 形态与 same-prefix

### 4.1 contact sequence 主任务

在事件 anchor 使用 pre-event state，预测：

- later recruitment；
- continue/STOP；
- extent；
- same-prefix continuation。

### 4.2 精确传播与能量任务

在成熟 tissue decoder 的 per-step hidden 上分别预训练并冻结：

- continuous contact lag/direction head；
- per-contact multiband energy/peak-time head；
- cross-band lag 与 waveform-expression head。

每个 head 先只在 calibration/TRAIN 上证明自身能预测目标，再接入同一逐步 state adapter。这样不会把“新 head 没学会”误写成“状态没有该信息”。

### 4.3 承重结果

主图画：

- future count skill；
- conditional contact morphology skill；
- same-prefix later-path gain；
- 1/5/20-event 与 5/30/120-min persistence curve；
- correct-time 与 block-shift。

rate-only 与 mark-residual 分开着色，不把事件率变化冒充传播状态。

## Phase 5：跨任务与反馈

只有前四阶段形成可用、已冻结的 `q/m` registry 后启动；不要求所有端点阳性。

### 5.1 H2b

1. 冻结 `q/m`，在 5 min grid 训练单调离散 survival hazard；
2. 预测下一次发作距离；
3. 在 5 min/30 min/2 h/6 h 预测 early ictal contact energy/recruitment field/path；
4. 比较历史/临床 baseline、`q only`、`m only`、`q+m`；
5. 发作梯度禁止回流到间期 producer。

若只有 `q` 有效，结论是 rate-linked susceptibility；若 `m` 在 `q` 外增加 early field/path，才称 network-expression susceptibility。

### 5.2 H3

按已修复的 coverage/独立窗口/截距合同比较 M0 common-drive、M1 burden feedback、M2 mark feedback。先做具有真实独立物理时间块的患者与尺度，不重复旧的一万事件滑窗伪样本设计。

## Phase 6：扩大队列

第一波 E253/E922/E1096/E548/E583/E1146 形成完整训练卡后，再按事前 coverage 表扩展到其余患者。扩展顺序由：

1. decoder 可用性；
2. 连续记录时长；
3. 5/30/120 min future-block 支持；
4. event waveform/multiband coverage；

决定，不由 pilot 结果方向决定。

## 资源与持久运行

- 独立 worktree 与 `/data/hfosp_group_event_state_v0_3_5/` 结果根；
- 每个 `(subject, arm, seed)` 单独原子目录；
- `nohup`/`setsid` 或 tmux 脱离会话，队列与监控分离；
- 每卡先测安全 batch，再并行填满 GPU；
- OOM 只降低同 job batch 或同卡并发，保留 effective config 与 retry history；
- `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`NUMEXPR_NUM_THREADS=1`；
- supervisor 每 15–30 min 更新 completed/failed/OOM/ETA、GPU 利用率、训练边界、科学卡片；
- 新结果只追加，不覆盖旧版本。

## 预计工作包

| 工作包 | 产出 | 预计顺序 |
|---|---|---|
| W0 | timescale/coverage/decoder provenance | 首先 |
| W1 | static→dynamic rate baseline | 首先并行 |
| W2 | per-step frozen-decoder adapter | 与 W1 并行 |
| W3 | full-event `m(t)` + multi-horizon pilot | W1/W2 接口完成后 |
| W4 | lag/energy/waveform heads + same-prefix | W3 学动后立即 |
| W5 | frozen H2b | W3/W4 状态登记后 |
| W6 | H3 explicit feedback | W5 可并行，但必须在状态登记与可估性后 |

## 最终交付

1. 白话报告：静态校准、动态 rate state、mark-residual state、H2a/H2b/H3 分层结论；
2. 技术报告：逐患者/seed/horizon/endpoint、训练曲线、覆盖与独立块、provenance；
3. 机器报告：`dynamic_baseline.json`、`stepwise_decoder.json`、`h1_h2a.json`、`h2b.json`、`h3.json`；
4. 四张核心图的 PNG、vector PDF、metadata 与中文 `figures/README.md`。

## 完整执行后 H3 收尾补充（2026-09-04）

W6 的全部已注册人体单元在统一数值可采信规则下重算：新增反馈槽若 INNER 或 SELECTION MSE 相对嵌套父模型超过 4 倍，保留原始拟合但不纳入效应汇总。重算必须覆盖全部 7 位可训练患者 × 3 seed，不以单个异常患者的事后修图代替。随后重新生成 H3 长表、支持度面板、白话报告和技术报告，并运行全套 v0.3.5 回归测试与四图目视验收。

## 审阅修正后的重跑（2026-09-04）

代码审阅发现 `q(t)` 的段位置特征使用了覆盖段结束时刻（多数等于下一次发作起点），属 spec §11 全局停止条件 2。处理：

- 修正 `dynamic_rate._causal_features`、`feedback_models._common_time_features`（因果段位置），`feedback_models._nested_arm_admissibility`（双向 4 倍），以及四处 block-shift 对照的同锚点打分；`tests/test_group_event_state_v035.py` 新增回归测试。
- `scripts/supervise_group_event_state_v035_causal_rerun.py` 在 `/data/hfosp_group_event_state_v0_3_5_causal/` 重跑 W1（含背景 SEEG）→ W2 → W3（锁定 `compact` 配方，SELECTION 训练结束后开一次）→ W4–W6 → finalizer；冻结 decoder 与 future-oracle 正对照沿用原运行。
- finalizer 通过 `HFOSP_GES_V035_OUTPUT_ROOT` / `HFOSP_GES_V035_REPORT_TAG` 区分原始产物与因果重跑产物，两份报告顶部均带审阅修正说明；原始报告仅供对照。
