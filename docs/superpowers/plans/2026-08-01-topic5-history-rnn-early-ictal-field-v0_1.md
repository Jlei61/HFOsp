# Topic 5 跨事件历史 RNN 到发作早期能量场 v0.1 执行计划

对应 spec：`docs/superpowers/specs/2026-08-01-topic5-history-rnn-early-ictal-field-v0_1.md`

## Milestone A：旧线收口与命名修正

1. 把现有 event-reset RNN 结果登记为 `INTERICTAL_WITHIN_EVENT_SEQUENCE_ENCODER_QUALIFICATION`。
2. 把既有 early-ictal exporter/readback 登记为 `WITHIN_EVENT_ORDER_BRIDGE_PILOT`。
3. 文档中删除“旧 hidden state 是跨事件 history state”的表述。
4. 冻结既有 training-sufficiency 结果；不再为本任务重跑完整事件生成。

验收：旧结果保留但不能再充当 G1/G2/G3。

## Milestone B：G0 target-blind 因果时间轴审计

新增 target-blind audit，输入只允许：

- masked rank dataset metadata；
- 原始 event absolute time / block / recording identity；
- seizure inventory 的 clinical onset/offset；
- accepted target 的 subject、seizure index、contact 名称和 key inventory；
- 不反序列化 target field 数值。

输出：

- 逐事件 timeline index；
- 逐发作 causal history inventory；
- segment/reset reason；
- 10 min guard 后可用事件数、跨度和 last-event gap；
- exact contact join；
- 重复 causal states；
- G0 verdict 与数据 fingerprint。

必须显式复核：

- 16/106 strict clinical-onset denominator 是否漂移；
- 哪些 seizure 在 guard 前没有可用 history；
- 哪些 patient 有至少 2/3 个不同 history states；
- selected fail-closed blocks 造成的 coverage gap；
- recording discontinuity 与 event-sparse interval 的区别。

验收：target 数值读取标志必须为 false；否则 G0 失败。

## Milestone C：双级 encoder 实现与单元测试

实现 `src/topic5_history_rnn.py`：

1. `encode_within_event()`：从冻结 `LinearStateSequenceRNN` 导出 \(u_e\)；
2. `TimeDecayHistoryGRU`：真实 IEI 衰减 + 跨事件 GRU；
3. `MatchedUnorderedHistory`：相同 events 的 mean/max + last event + scalar context；
4. `NextEventContactReadout`：participation + relative-rank；
5. `EarlyIctalContactReadout`：contact-centered shared readout；
6. causal sequence chunking，segment 开头严格 reset；
7. across-event order shuffle 与 within-event rank shuffle。

单元测试至少覆盖：

- EventRNN 每场事件 reset；
- HistoryRNN 在连续事件间不 reset；
- `delta_t=0` 不衰减，较大 `delta_t` 衰减更强；
- segment boundary 精确归零；
- event-order permutation 改变 HistoryRNN、但不改变 matched unordered pooling；
- causal prefix 不包含 guard 之后事件；
- heldout target 不参与参数拟合；
- contact join 一一对应；
- target-centered readout 对全局 shift 不变；
- chunked recurrence 与完整 recurrence 数值一致；
- CPU/GPU smoke 无 NaN/OOM。

## Milestone D：development smoke 与冻结

development patients 固定为既有三人：

- `epilepsiae_1073`；
- `epilepsiae_1146`；
- `yuquan_chenziyang`。

三位都只用于 G1 工程、loss 与超参数选择；`1146` 还可用于 target bridge 的 shape/leakage smoke，但不能据其 G2 科学结果修改模型。三位均排除于 G1/G2 的确认性 primary inference。允许冻结：

- history hidden size：16/32 二选一；
- decay 参数初始化：0.5/2/6 h 半衰期；
- BPTT chunk：128/256；
- learning rate：`1e-3/3e-4`；
- early stopping；
- participation/rank loss 固定权重。

Development 采用预定义的 staged one-factor audit，而非事后 architecture zoo：先以 `hidden=32, half-life=2 h, chunk=256, 3 cycles, lr=3e-4` 为中心，分别检查 `lr=1e-3`、`half-life=0.5/6 h`、`hidden=16`、`chunk=128` 与 `6 cycles`。只有这组允许项完成后才冻结正式 G1；early-ictal target 在整个选择过程中保持封存。

不得根据这三人的 G2 科学效果反复修改 target readout。冻结后写 config hash。

## Milestone E：G1 自监督确认

1. 在全 34 人上建立 chronological next-event decisions；31 位 development-excluded patients 构成 primary inference cohort，34 人全体为 supportive。
2. 外层逐 patient 留出 shared encoder/history model。
3. heldout patient 只做 interictal local calibration；评价其后续 heldout events。
4. 运行：
   - matched unordered；
   - chronological HistoryRNN；
   - causal-prefix-matched across-event order shuffle（固定相同 prefix event set 和 last-event embedding，只置换最近 64 个 observed events 中更早的 63 个；开发期整段置换不作 gate）；
   - within-event rank shuffle。
5. 保存逐 event/contact loss、逐 patient aggregate、两数据集分层、seed 和 checkpoint。

G1 先单 seed 全 cohort；若 primary direction 为正，再补 3 seeds。G1 失败则写 bounded-negative，停止 G2/G3。

## Milestone F：G2 strict LOSO early-ictal bridge

仅在 G1 通过后打开 target values。

对 16 位 strict clinical-onset target patients 逐一；primary inference 排除 development patient `epilepsiae_1146`，因此主分母为 15 人，16 人全体为 supportive：

1. outer train = 其余 target patients；outer test = heldout patient；
2. shared EventRNN/HistoryRNN 不能看 heldout patient 的 ictal target；
3. 每次 seizure 独立构建 onset minus 10 min causal state；
4. 训练并比较 `M0/M1/M2`；
5. heldout patient 的所有 target 仅最终评分一次；
6. 对重复 state 的 seizures 先折叠，再 patient-first 汇总。

Primary：LOSO `rho(M2)-rho(M1)`。

Secondary：centered MSE、cosine、5/30 min guard sensitivity、within-event rank-shuffle M2、across-event order-shuffle M2。

## Milestone G：G3 state–seizure pairing

只在至少两个不同 causal states 的患者中运行：

- correct state–seizure pairing；
- same-patient wrong pairing；
- circular shift；
- 对至少三条不同 state 且 residual reliability 合格者做 LOSO residual analysis。

不以 pooled seizure rows 做显著性；patient-first。

## Milestone H：复现、监控与图

运行规范：

- 长任务用 `tmux` 或 `nohup`；
- 每个 fold 独立目录，写 `RUNNING.json` / `DONE.json` / `FAILED.json`；
- 每 30–60 s 写 heartbeat、GPU memory、CPU RSS、当前 fold/epoch/loss；
- 每 GPU 先 1 个 worker，实测余量后增加，不用进程数堆满显存；
- batch/chunk 自适应减小只能写进日志，不能静默改变科学 config；
- NaN/OOM 自动停止当前 fold，保留 traceback，不吞错；
- 所有输出带 input/config/code hash。

图目录：`results/topic5_history_rnn_early_ictal_field/figures/`，生成图后同时写中文 `README.md`，逐图说明科学含义与 claim 边界。

## 执行停止点

1. 当前立即执行至 G0。
2. G0 通过后实现并 smoke 双级模型。
3. G1 全 cohort 通过后才解封 early-ictal target。
4. G2/G3 结束后才决定能否进入主文；不因阴性结果临时扩展 architecture zoo。
