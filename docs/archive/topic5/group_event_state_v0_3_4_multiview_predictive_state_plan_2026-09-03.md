# Group-Event State v0.3.4 执行计划

## Multi-view Interictal Predictive State

**状态：** `V0_3_4_S_P_TUNING_ACTIVE`  
**日期：** 2026-09-03  
**Spec：** `group_event_state_v0_3_4_multiview_predictive_state_spec_2026-09-03.md`  
**原则：** 少 gate、多探索；三条科学线并行，但共享数据、时间边界、checkpoint registry 与评分合同。单端点阴性不停止其他线。

## 2026-09-03 评价偏移复审后的立即修订

旧 `spatial_state_seedfixed` 结果保留为优化与 L1 阶段水平证据，不再承担 L2/L3 结论。新运行写入独立 `spatial_state_recalibrated` 根，按以下顺序执行：

1. 实现并单测 TRAIN 内 fit/rolling-inner、30 min embargo、无状态 `train_mean_adapter`；验证改变 `STATE_SELECTION` target 不会改变 selected checkpoint。
   - `train_mean_adapter` 使用独立固定 LR `3e-3`，确定性 seed；训练卡必须报告是否在预算边界。边界未收敛者先延长 baseline，不进入 state 对比。
   - TRAIN-inner 的 30 min target 也必须完整结束在 `STATE_SELECTION` 前；回归测试应改写报告期独占 targets，并证明 selected checkpoint/hash/history 逐项不变。
2. 卡片用完整 `STATE_SELECTION` 同表输出无状态重标定、因果 rolling level、非因果阶段常数、correct-time 与 wrong-time；分别给出总增益、常数之外增益和错时代价。
3. 重做 synthetic suite：dynamic truth、constant truth、no-state truth；旧“相对初始化 >0.001”不再作为恢复标准。synthetic 失败只阻止引用该仪器，不停止其他 S_N/S_F/H2b 探索。
4. 先以原 recipe 重跑 E548 ×5 seeds 和 E922 ×5 seeds（E922 2700 steps）以量化修复影响；并在 E253 重新做小范围 recipe 选择。E1146 只保留首步最优诊断，E583 作为低支持描述，不作 recipe 决策。
5. 有空闲 GPU 时并行启动 E253 修正版搜索与 E548/E922 多 seed；OOM 时只降低同一 job 的 batch/concurrency，不改变输入、目标、split 或优化器。

本修订不把“常数可解释”写成“没有状态”。当前允许的最强说法是：L1 阶段/超慢水平信息成立；是否存在超出训练期重标定且能在每个时刻因果估计的 L2/L3 状态，需由修正版实验回答。

## 0. 本轮不再做什么

- 不再围绕旧 `S_N` count 结果继续扩大患者、tau 或 reset 网格；当前 6/12/24 h 慢库完成后归档。
- 不消费旧 9 人未使用的 development 来给旧失准 count 模型做“最终裁决”。
- optimism gap 只使用 TRAIN rolling inner-validation → STATE_SELECTION 的差，加上已存在的 E253/E916 两条 development 先例；不新增旧模型的 development 读取。
- 不把 synthetic recovery 当人体可训练性的替代。
- 不把 selection-period 常数称为泄漏或无信息；明确标成非因果阶段信息上界。
- 不同时更换 decoder objective 与状态架构后再声称复现旧结果。

## 1. Phase 0：v0.3.3 收口与注册表

1. 等当前 `slow_bank_arms_on_mark` 完成，原样归档，不再扩展。
2. 冻结 v0.3.3 的 schema-9 汇总、训练卡、offset/recalibration/slow-bank 产物与哈希。
3. 在 closeout 加入本 spec §14 的五层快照，改掉“没有额外信息”的过强措辞：
   - L1 阶段/超慢信息存在；
   - L2 有患者级 learned-history 候选；
   - L3–L5 未建立。
4. 建立 `checkpoint_registry.json`，登记 `B_multiscale`、`P_local`、`P_slow/S_N`、`S_P`、`S_F`，不只保留一个 winner。

**完成条件：**旧结果只读、路径和 hash 可追溯；当前后台任务归档；无 development/sealed 新读取。

## 2. Phase 1：共同数据与评估合同

### 2.1 事件 token

实现并逐事件核对：

- participation/tied groups；
- continuous centroid lag；
- per-contact multiband energy、peak time、cross-band lag；
- bipolar/CAR waveform embedding；
- coords/shaft/coverage/mask；
- background SEEG auxiliary observation。

新增 phantom-rank 回归测试：任何 rank 特征在非参与触点必须 masked；随机改写 phantom ranks 不改变 token。

### 2.2 固定时间 anchor 与 target

- 默认每 5 min anchor；
- Core 为未来 5/30 min；120 min/6 h 仅在独立块可估时作 exploratory；
- event anchor 的 next 1/5/20 events；
- target 不跨 split/session/gap/seizure；split 间 embargo ≥ max horizon。

用 cumulative/sparse arrays 按需产生 future targets，不复制完整触点张量。

### 2.3 `B_multiscale`

实现 spec §6 的多尺度 rate + mark + clock/session/coverage 基线。TRAIN 内 rolling inner-validation 选形式、窗口与 shrinkage；development calibration 只读报告，不作 eligibility gate。

同时实现：

- `train_mean_adapter`；
- `rolling_prefix_level`；
- `selection_period_mean`（明确 `noncausal_input_oracle=true`）；
- random/times-only/mark-shuffle/block-shift。

**完成条件：**每个 anchor 可在同一 evaluator 中并排评分；因果/非因果 provenance 机器可读。

## 3. Phase 2：预训练 decoder 与状态 producer

### 3.1 Contact decoder 审计与接口

1. 审计旧 decoder 的 normalization、vocabulary/order、patient adapter、calibration、tied-group statistics、detector template、checkpoint selection 的时间来源。
2. zero state adapter 对旧 checkpoint 做逐事件 parity。
3. 保持旧 scoring 的 pilot parity；另立 exact objective 重预训练分支，不混跑。
4. 在 continue、positive size、subset/order、energy heads 接入低秩 state adapter；decoder 冻结但保留对 state 的梯度。

### 3.2 producer 的先后顺序

- `S_N`：作为辅助负荷视图，复用已知实现但改用新基线和固定时间 anchors；
- `S_P`：唯一人体 Core 状态。先完成同构 synthetic recovery，再做人体验证；
- `S_F`：先完成 token/target/frozen-probe 接口；只有 `S_P` 在人体 tiny-overfit 和 rolling inner-validation 上确认学动后，才启动探索性训练，不占用 Core 算力。
- `P_local` 与 multi-horizon `P_slow` 并列登记。

### 3.3 训练搜索

先在 E253/E916：

- 36–48 个宽覆盖配置；
- rung `300 → 900 → 2700`；只有曲线仍下降才到 8100；
- 搜索 LR、optimizer、schedule、width/depth、residual、normalization、初始化、state/write scale；
- top 3 recipe × 5 seeds；
- tiny-overfit、梯度/update、blocked generalization 分开验。

配方冻结后，`S_P` 在 E1146/E583/E548/E922 运行，不按结果剔除患者；这四位仅对 30 min conditional grammar 合格，不是通用 count/H2b 队列。并行度由实测显存决定：先用 canary 测单作业峰值，再逐级增加并发并持续观察显存与利用率；2026-09-03 的 rung-300 实测支持每 GPU 5 个并发（约 2.4 GiB/卡、无 OOM）。OOM 只允许同 job 降 batch/concurrency 重试，记录 effective config。

**完成条件：**每个视图至少有训练充分与未训练充分的明确分类；不能用“synthetic 能学”代替人体曲线。

### 3.4 2026-09-03 启动记录

- `S_P` 同构 synthetic recovery：PASS；selection gain `+0.1644`，selected step `25`，梯度非零且参数发生更新；该结果只证明实现和目标在合成数据上可学习。
- E916 真人 tiny canary：PASS；梯度非零、参数发生更新，未读取 development、seizure outcome 或 sealed partition。
- E253/E916 的首批人体搜索已用 `nohup + setsid` 启动：每人 5 个独立学习率 cell，rung `300`，两张 GPU 各 5 个并发。
- 首批已经返回的 cell 显示：E253 两个配方的 selection gain 分别为 `+0.0435`、`+0.2295`，selected step 为 `250/275`；E916 对应两个配方虽有梯度和参数更新，但 selected step 均为 `0`。因此“训练链路能更新”与“人体选择期有可泛化增量”已被明确拆开，其余配方完成前不作患者结论。
- 首批 10 个 cell 完成后，已启动完整首阶搜索：2 位患者 × 4 组 width/depth × 5 组独立学习率，共 40 个 rung-300 cell；监督器跳过已完成的 10 个并自动补跑余下 30 个。队列状态原子写入 `spatial_state/manifests/spatial_search_status.json`，进程与终端解耦。
- 完整 rung-300 已于当日结束：40/40 完成，0 失败、0 OOM。E253 为 20/20 gain>0、0/20 选择 step 0，中位 gain `+0.2614`；E916 为 0/20 gain>0、20/20 选择 step 0。E253 的中位增量几乎全部来自冻结 legacy grammar adapter（grammar `+0.2606`），独立 subset/continue/extent/lag 头尚未给出同等级增量。该表只证明不同患者上的可训练性差异，尚未经过常数、rolling-prefix 与 correct-time 对照，不能提前升级为 H1/H2a 结果。
- 已按首阶结果启动 rung-900：E253 top 3 recipe 进入 5 seeds；E916 因所有首阶 cell 并列为 0，选取三个宽深度不同的代表配方作延长预算诊断，并另做代表配方多 seed，而不是把单 seed 的 0 当科学阴性。
- rung-900 已完成：E253 三个 recipe 均为 5/5 seeds 正增量，中位 gain 分别为 `+0.3139`、`+0.3073`、`+0.2826`；E916 的代表配方 5/5 seeds 仍为 selected step 0，另两种宽深度代表也为 0。只有 E253 的一个 seed 在 step 900 仍取新最优，故仅该 run 升到 rung-2700；不把其余已早停 cell 机械扩预算。
- `pre-seed-fix` 阶段原暂定进入四位端点评价患者的 recipe 为 E253 上跨 5 次运行中位最优的 `width=64, depth=4, lr_encoder=1e-3, lr_state_adapter=3e-3, lr_auxiliary=1e-3`；下述 seed 主审更正后该暂定冻结作废，必须由 seed-fixed 产物重选。评价发布仍使用独立 locked-recipe gate，E1146/E583/E548/E922 不参与配方再选择。

**主审更正：**上述首轮搜索随后发现 seed 设置发生在模型构造之后；因此命令行 seed 没有控制初始化，相同 seed 的 rung-900/rung-2700 公共前缀不能复现。旧搜索完整保留但统一标为 `pre_seed_fix_optimization_diagnostic`，不得承担 recipe 冻结或多-seed 证据。代码已改为在任何模型构造前同时固定 Python/NumPy/Torch/CUDA RNG，并把 `seed_contract=python_numpy_torch_seeded_before_model_construction_v2` 写入训练卡；同 seed 初始化 hash 一致、异 seed 不同的回归测试已通过。修复版 synthetic 与真人 canary 均 PASS，40-cell 搜索在独立 `spatial_state_seedfixed` 根目录重跑；本段前两条数值只作为发现该问题的审计记录，最终 recipe 必须由 seed-fixed 产物重新选择。

**seed-fixed 更新：**40/40 rung-300 cell 完成，0 失败、0 OOM。E253 为 20/20 gain>0；E916 为 3/20 极小 gain>0、17/20 仍选 step 0。E253 top 3 配方进入 rung-900 ×5 seeds 后均为 5/5 正增量，最终按中位 gain 锁定 `width=64, depth=4, lr_encoder=3e-4, lr_state_adapter=3e-3, lr_auxiliary=3e-4`（中位 `+0.3165`）。E916 代表配方跨 5 seeds 仅 2/5 极小正增量，中位为 0。rung-300 与 rung-900 同 seed 公共 12 个评估步的初始化 hash 与全部 history row 已逐项精确一致。四位评价患者已用该配方、独立 gate 与 calibration-prefix frozen decoder 启动；这些仍是训练/选择层证据，常数、rolling-prefix、block-shift 和 future-block 对照完成前不升级为 H1/H2a。

## 4. Phase 3A：H1/H2a

### 4.1 H1 future-block

对三种 state view 在固定时间 anchors 运行：

```text
B_multiscale
B_multiscale + P_local
B_multiscale + P_slow
correct-time P_slow
block-shifted P_slow
```

分别给 count、conditional repertoire、participation/extent、propagation、multiband/waveform proper score。Core 主图是 5/30 min；120 min/6 h 降到 exploratory，并必须同时给出独立块数。

### 4.2 H2a same-prefix

在冻结 contact decoder 上，按相同 early prefix 比较 baseline 与 baseline + frozen state，输出：

- continue/STOP；
- positive extent；
- later contact subset/order；
- later lag/direction；
- later multiband/waveform；
- next 1/5/20-event 预测。

**判读：**L1–L4 分层独立报告。常数胜出不取消 event-history 候选；random/times-only/mark-shuffle 只定位来源。

## 5. Phase 3B：H2b（与 3A 并行）

状态在读取任何 seizure label 前冻结。先按端点核对可估性：当前四位 S_P 哨兵中，E548 有 9 次、E922 有 5 次 development 发作；E1146/E583 为 0，不能进入 H2b 分母。early-field 与 subtype 另做 target-specific support 表，不能沿用 risk 分母。两个任务同时实施：

1. 固定 5 min grid 的离散 survival，评价 Brier skill、log score、calibration；
2. 5 min/30 min/2 h/6 h lead 的 early 5–10 s ictal contact field/path。

基线包括 patient mean、最近 IED、`B_multiscale`、time since last seizure、postictal/cluster 和可用临床背景。按 seizure pattern 分层；采用 rolling-origin/held-out seizures。

不要求 H1 先显著才运行 H2b，因为跨任务迁移本身是状态科学价值的直接测试。

## 6. Phase 3C：H3（独立并行）

人体启动前先运行 H3 estimability audit：真实 coverage segment 内构造 exposure/future block；报告完整非重叠窗口、有效独立窗口与 exposure overlap；旧的 1–2 个有效窗口结果标 `not_estimable`。在通过的 `(subject, scale)` cell 和已登记的最佳可训练 producer 上比较：

- `M0_common_drive`；
- `M1_burden_feedback`；
- `M2_mark_feedback`。

三臂必须截距匹配并共享同一个因果 `rolling_prefix_level`；M0 获得与反馈臂等容量的截距/慢水平项。`selection_period_mean` 只作非因果上界。若 M1/M2 仍只赢未匹配的 no-edge，不算 feedback evidence。

先做 100/1000/10000-event 三档中有独立 support 的档位，固定相同 pre-state 与 future block。分别回答 burden effect 和 content effect：

- burden 比较不得匹配掉 exposure count；
- content 比较保持事件数/时刻，替换 mark。

输出 held-out future-block score 与 signed functional impulse response。结果不依赖 H2b 阳性，也不升级为人体因果结论。

**2026-09-03 实现状态：**H3 截距/滚动慢水平/覆盖段审计代码与 9 项 synthetic canary 已完成，synthetic 为 9/9 PASS。按当前操作性下限（STATE_TRAIN ≥8 个、inner-validation ≥3 个互不重叠块），27 人中 N=100 的 5/30 min future 分别有 26/23 人可估，N=1000 为 14/12 人，N=10000 为 0/0；2 h physical exposure 为 9/9 人，6 h 为 4/2 人。该下限只是首轮运算资格，不是经过 power calibration 的检出力保证；人体 M0/M1/M2 必须绑定可训练的已登记状态 producer 后才启动。

## 7. Phase 4：共享信息与 replication

1. 冻结 `S_N/S_P/S_F`，做 within-view 与 cross-view transfer；CCA/RRR 只作 latent 诊断。
2. shared producer 仅在 cross-view 有可复现功能增量后训练。
3. 锁定模型后，按预先登记 eligibility 在 E1073/E1077/E818/E958 上运行；不可估者按 coverage/support 退出，不按方向退出。
4. 正式 sealed partition 仍不打开，除非用户另行书面授权。

## 8. 三张核心图与机器接口

### Figure A：H1/H2a predictive state

- Core 横轴：5/30 min；120 min/6 h 放 exploratory inset；
- 纵轴：相对 `B_multiscale` 的 future-block proper-score gain；
- 分 count 与 conditional content；
- same-prefix continuation 作为右侧决定性 panel；
- correct-time 与 block-shift 同图。

### Figure B：H2b seizure transfer

- risk/Brier skill 随 lead time；
- early ictal field/path gain 随 lead time；
- 每点以 held-out seizure/患者为分母。

### Figure C：H3 feedback

- M0/M1/M2 future-block score；
- burden 与 mark-specific signed impulse response；
- 100/1000/10000 events 的真实支持范围。

接口固定为 per-subject long table + cohort JSON；允许当前不显著值进入，不用模拟显著占位。

## 9. 工程执行合同

- 新代码根：`src/topic5_group_event_state/v034_*`；
- 新结果根：`/data/hfosp_group_event_state_v0_3_4/`；
- 每个 run 独立输出目录、原子 manifest、checkpoint/config/input/split/evaluator hash；
- `nohup`/`setsid` 脱离会话，可断点续跑，重复启动幂等；
- `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`；
- 每个 seed 的 payload/checkpoint hash 必须不同；
- 训练、评价、画图 producer 分离；图目录实际生成后补中文 `README.md`。

## 10. 只保留四类全局硬停

1. sealed partition 被读取；
2. normalization/target/checkpoint selection 使用了未来评价期；
3. target 跨 session/gap/seizure/split；
4. canonical evaluator 对同一对象给出不一致结果或产物被并发污染。

优化困难、单患者阴性、某视图失败、H1/H2b/H3 彼此不支持都不是全局停止条件；记录后继续其他探索。

## 11. 最终交付

- 白话报告：五层证据、三假设结论、患者级异质性、可以/不可以写什么；
- 技术报告：完整 per-subject/horizon/endpoint/seed 表、训练曲线、资源与 provenance；
- 机器报告：`evidence_levels.json`、`h1_h2a_results.json`、`h2b_results.json`、`h3_feedback_results.json`；
- 三张核心图 PNG + vector PDF + metadata + `figures/README.md`。
