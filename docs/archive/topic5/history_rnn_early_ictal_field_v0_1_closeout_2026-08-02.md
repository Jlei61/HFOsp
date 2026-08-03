# Topic 5 HistoryRNN v0.1 next-event proxy 暂定收口

日期：2026-08-02

状态：`PROVISIONAL_BOUNDED_NEGATIVE_FOR_CURRENT_G1_TASK`

## 1. 一句话判断

在控制患者特异 static contact prior、无序事件集合、last event、事件数和真实时间跨度后，当前带 IEI 衰减的跨事件 HistoryRNN **没有在三 seed、patient-first 检验中提供稳定的 next-event contact-field 增量**。因此 G1 未通过，clinical-onset `[0,10] s`、`1–150 Hz` early-ictal contact-energy target 按冻结规则保持封存，G2/G3 没有运行。

需要与上一条分开读的是：把同一段 causal prefix 里较早事件的先后打乱（事件集合、时间槽、last event 全部不动）**确实稳定地损害了 chronological 模型自己**（31 人中位 +0.000160，26/31 为正，单侧 p=3.9e-05，三 seed 与两队列方向一致）。这两件事不矛盾——它说明这个 state 确实依赖输入顺序，但这种顺序依赖并没有转化成对"完全看不到顺序"的匹配模型的优势。顺序敏感性也可能只反映递归网络没见过乱序输入，因此**不能**用顺序对照反推"顺序带来了预测信息"；唯一直接检验这一点的是 `M2−M1`，而它为零。

这不是“early-ictal field 不可预测”的阴性结果；它只说明当前自监督 next-event objective 没有辨识出可复现的 chronology-specific 增量。原合同把代理任务设为 direct transfer 的硬门并不充分，direct early-ictal transfer 将在独立 v0.2 合同中运行。

在 synthetic recoverability、局部收敛和 state-utilization 审计完成前，本结果不升级为模型类的正式阴性。

## 2. 本轮真正检验的问题

本轮不再把事件内 next-rank RNN 当作发作状态。模型明确分成两层：

\[
\text{one-event rank field}
\xrightarrow{\mathrm{EventRNN\;reset}}
u_e,
\]

\[
z_e=\mathrm{GRUCell}\!\left(
u_e,
\exp[-\mathrm{softplus}(\gamma)\Delta t_e]z_{e-1}
\right).
\]

- `EventRNN` 每场事件重置，只编码该事件的 contact-rank field；
- `HistoryRNN` 只在真实连续记录 segment 内跨事件持续，并使用真实 IEI；
- 记录中断、缺块和 postictal reset 会清零 history state；
- G1 完全不读取 ictal target value。

G1 primary 比较为：

\[
\mathrm{BCE}(M_1)-\mathrm{BCE}(M_2),
\]

其中 `M1` 和 `M2` 使用同一 contact、同一 causal prefix、同一 static prior、mean/max pooling、last event 和 scalar context；`M2` 唯一多出的信息是按真实时间顺序积累的持续状态。二者 residual 参数量近似匹配：`M1=3496`，`M2=3440`。

## 3. 数据与 target seal

### 3.1 G0 metadata audit

- early-ictal endpoint 预先固定为 `clinical_onset`、`[0,10] s`、`1–150 Hz`；
- onset guard 为 10 min，postictal reset 为 70 min；
- strict inventory 为 16 位患者、106 次发作；
- 16/16 患者 target key 和 exact contact join 完整；
- metadata-only audit 得到 33 个 G2-eligible causal histories、11 位 G3 pairing-eligible 患者和 3 位 residual candidate；
- G0 只反序列化 target channel names，没有读取任何能量数组。

### 3.2 G1 数据合同

- 34 位患者均进入 target-sealed LOSO 训练；
- development 固定为 `epilepsiae_1073`、`epilepsiae_1146`、`yuquan_chenziyang`；
- primary inference 为其余 31 人，34 人全体只作 supportive；
- heldout 患者 static prior 只用其 chronological train80；
- shared normalization 只用外层训练患者的 train80；
- heldout20 只用于 next-event 最终评分；
- 所有统计先按患者聚合，不以事件或 contact 伪增大样本量。

## 4. Development 与冻结配置

三位 development patients 共运行 21 个 target-blind 配置单元，最终冻结：

```text
history dimension      16
initial half-life      2 h
learning rate          3e-4
optimizer              AdamW, weight decay 0
matched/history cycles 3 / 3
BPTT chunk             256 events
segment batch          16
```

冻结配置在 development 中为 2/3 患者 chronological increment 正向，工程性全局顺序置换为 3/3 正向。该结果只用于选择配置，不充当正式科学 gate。

## 5. 正式 G1 结果

### 5.1 执行完整性

- 3 seeds × 34 LOSO folds = 102/102 完成；
- 102/102 causal-prefix-matched order controls 完成；
- 0 failed folds、0 OOM、0 NaN、0 traceback；
- 一次会话中断由 `DONE.json` 断点恢复，没有结果丢失；
- 复现 manifest 对 102 个 DONE、102 个 order-control JSON 和 102 个 checkpoint 逐一记录 SHA256。

### 5.2 Primary 31 人

| 对比 | patient median | 95% bootstrap CI | 正向患者 | 单侧 Wilcoxon p | 判定 |
|---|---:|---:|---:|---:|---|
| static → matched unordered | +0.003497 | [−0.000443, +0.005875] | 19/31 | 0.0040 | signed-rank 偏正但中位 bootstrap CI 跨零；见 §7 归因边界，不能读成"最近发生了哪些事件带有信息" |
| matched unordered → chronological | **−0.000186** | [−0.000565, +0.001509] | 15/31 | **0.1012** | G1 primary fail |
| strict order shuffle − true order | **+0.000160** | [+0.000069, +0.000250] | **26/31** | **3.9e-05** | order control 通过；只证明 state 依赖顺序，不证明顺序有增量信息 |
| relative-rank increment | −0.001201 | [−0.001713, −0.000502] | 7/31 | 0.9996 | 不支持 |
| within-event rank shuffle − true | −0.000757 | [−0.001521, +0.000152] | 14/31 | 0.7779 | 不支持 |

这里的 BCE 单位是每 contact-decision 的 nats；正值表示右侧模型更好。

### 5.3 数据集与 seed 稳定性

| 分层 | chronological median | strict order-shuffle median |
|---|---:|---:|
| Epilepsiae（n=16） | +0.000315 | +0.000127（14/16 为正） |
| Yuquan（n=15） | −0.000237 | +0.000173（12/15 为正） |
| seed 20260725 | +0.000810 | +0.000145 |
| seed 20260726 | −0.000186 | +0.000236 |
| seed 20260727 | −0.000363 | +0.000250 |

primary contrast 上，seed1 的近阈值正向没有在两个确认 seed 中复现，数据集方向也不一致，因此不能用单 seed 或单一队列把该状态写成阳性。顺序对照则相反：三 seed、两队列同向，是本轮唯一稳健的正向信号，但它约束的是"state 是否依赖顺序"，不是"顺序是否有用"。

### 5.4 decay 参数边界

学习后的维度中位 half-life 在患者×seed 层面的中位为 1.977 h，IQR 1.971–1.986 h，全范围 1.878–2.137 h。它基本停留在 2 h 初始化附近，不能解释为从人体数据辨识出的生物学恢复时间常数。

### 5.5 全 34 人 supportive（不进入 gate）

| 对比 | patient median | 正向患者 | 单侧 Wilcoxon p |
|---|---:|---:|---:|
| static → matched unordered | +0.003350 | 21/34 | 0.0034 |
| matched unordered → chronological | −0.000096 | 16/34 | 0.0924 |
| strict order shuffle − true order | +0.000167 | 29/34 | 9.6e-06 |
| relative-rank increment | −0.001481 | 7/34 | ≈1 |
| within-event rank shuffle − true | −0.000740 | 15/34 | 0.8198 |

spec §5 与 plan Milestone E 都要求报告这一层；加入三位 development patients 不改变任何方向或判定。数值写入 `FINAL_CLOSEOUT.json::g1_supportive_34` 与 `G1_MULTI_SEED_SUMMARY.json::supportive_all_34`。

### 5.6 顺序对照的组成修正（2026-08-02 审阅）

首版顺序对照有一个模型组成错误：被打乱的那一臂在预测里多加了一个 unordered residual 读出项，而它要对比的 chronological 模型没有这一项。因此那个差值同时包含"打乱顺序"和"多了一个模型分支"两件事。

诊断方式是把**真实顺序**（不打乱）喂进同一段代码：正确组成下它必须逐 decision 复现 chronological 模型的 BCE。修正后误差为 1.3e-09（等价），修正前偏差 −0.004182 nats——而被报告的顺序效应量只有 1e-4 量级，逐患者偏差范围 −0.002 至 +0.011。也就是说旧数字里绝大部分不是顺序。

修正后重跑全部 102 个 fold；`chronological_increment`、`static_to_matched_gain`、`rank_increment`、`within_event_rank_shuffle_cost` 与学到的 half-life 逐位不变，只有 `prefix_matched_order_shuffle_cost` 改变（Epilepsiae 由 6/16 为正、中位 −0.001098 变为 14/16 为正、中位 +0.000127）。G1 判定不变，但停机理由由 5 条减为 4 条（顺序对照两条不再是失败项）。

- 回归测试：`tests/test_topic5_history_rnn.py::test_prefix_matched_order_control_uses_the_chronological_model_composition`（`window=1` 时对照必须逐 decision 等于 chronological 臂；重新引入任何额外读出项即失败）；
- 被推翻的产物按仓库惯例保留：每个 fold 的 `ORDER_CONTROLS.superseded_composition_bug.json`，汇总表在 `g1_sequential_formal_v0_1/superseded_composition_bug_2026-08-02/`。

## 6. G2/G3 为什么没有运行

实际执行的 gate（`summarize_topic5_history_rnn_gate1_multiseed_v0_1.py::pass_gate`，六个条件全部满足才放行）与结果：

| # | 条件 | 结果 |
|---|---|---|
| 1 | 31 人 `M2−M1` 中位 >0 | **未满足**（−0.000186） |
| 2 | 单侧 patient-level p<0.05 | **未满足**（0.1012） |
| 3 | Epilepsiae 和 Yuquan 的 `M2−M1` 都为正 | **未满足**（Yuquan −0.000237） |
| 4 | 同一 causal prefix 的顺序置换中位 >0 | 满足（+0.000160） |
| 5 | 顺序置换单侧 p<0.05 | 满足（3.9e-05） |
| 6 | 三个冻结 seed 的第 1、4 项方向一致 | **未满足**（seed 26/27 的 `M2−M1` 为负） |

spec §5 只写了前四类条件；第 5、6 条是进入 multi-seed 阶段时按 plan Milestone E "三 seed 确认"落地的实现，方向上只会收紧 gate，且第 1–3 条在没有它们时也已经失败。

第 1、2、3、6 项未满足，因此 G2 watcher 读取最终状态后退出，未创建任何 G2 fold。最终 closeout 和 reproducibility manifest 均记录：

```text
target_values_read: false
g2_g3_status: LOCKED_NOT_RUN
g2_folds: 0
```

因此当前没有 early-ictal 阳性或阴性统计，也没有 state–seizure pairing 结果。

## 7. 与论文核心科学目标的关系

### 可以保留的结论

1. 一个能学习 contact 读出、并且看得到近期事件汇总的模型，比原始的静态参与率先验预测得好（中位 +0.003497，19/31）。**归因边界**：本轮没有"能学习读出但看不到任何事件内容"的对照臂，因此这个差值同时包含"学到了更好的 contact 读出"和"近期发生了哪些事件"两部分，不能单独归给后者；而且它的中位 bootstrap CI 跨零，只算小幅、异质的辅助结果；
2. 跨事件 state 确实依赖输入顺序：同 prefix 打乱较早事件顺序会稳定损害它（26/31，p=3.9e-05，三 seed 两队列同向）。这是本轮唯一稳健的正向信号；
3. 旧线已经确认的事件内短程 rank 顺序信息不被本结果推翻；
4. 本轮确实把"何时进入发作状态"写成了真实时间、跨事件持续的模型，而不是把 event-reset RNN 误称为慢状态；
5. 在当前输入、容量和自监督任务下，这种顺序依赖没有转化成对匹配无序模型的优势，即没有辨识出可稳定复现的 ordered inter-event state。

### 不能写的结论

- RNN latent state 能预测 early-ictal field；
- early-ictal field 不能由间期历史预测；
- 癫痫患者不存在跨事件慢状态；
- 2 h 是人体病理状态的时间常数；
- seed1 的近阈值结果证明 history 有效；
- **顺序置换显著 ⇒ 真实事件顺序携带预测信息**（顺序对照只说明模型对顺序敏感，也可能是递归网络遇到未见过的乱序输入；直接检验是 `M2−M1`，为零）；
- **"最近发生了哪些事件带有额外信息"**（`static → matched` 的对照不干净，见上一节归因边界）；
- G2/G3 是 `FAIL`（正确状态是 `LOCKED_NOT_RUN`）。

## 8. 论文定位

本结果不应作为主文 Figure 6 的阳性机制终点。若需要保留，适合作为 Supplementary bounded-negative，回答：

> 在稳定接触点招募倾向和无序事件集合之外，真实时间顺序能否形成一个可迁移的跨事件状态？

当前答案是：在预先冻结的 next-event contact-field objective 下，HistoryRNN 没有稳定提供 chronology-specific 增量。主文仍应由已确认的 static interictal–early-ictal scaffold 和 SNN 机制层承重；direct latent-state → early-ictal field 尚未在 v0.1 中测试，将由 v0.2 独立回答。

## 9. 关键产物

- 统计收口：`results/topic5_history_rnn_early_ictal_field/FINAL_CLOSEOUT.json`
- 3-seed gate：`results/topic5_history_rnn_early_ictal_field/g1_sequential_formal_v0_1/G1_MULTI_SEED_SUMMARY.json`
- patient 表：`results/topic5_history_rnn_early_ictal_field/g1_sequential_formal_v0_1/g1_multiseed_patient_metrics.csv`
- 复现清单：`results/topic5_history_rnn_early_ictal_field/REPRODUCIBILITY_MANIFEST.json`
- 被组成 bug 推翻的旧顺序对照（保留为证据，不再引用）：`results/topic5_history_rnn_early_ictal_field/g1_sequential_formal_v0_1/superseded_composition_bug_2026-08-02/` 与各 fold 的 `ORDER_CONTROLS.superseded_composition_bug.json`
- 最终图：`results/topic5_history_rnn_early_ictal_field/figures/topic5_history_rnn_early_ictal_field_v0_1.{png,pdf,json}`
- 图说明：`results/topic5_history_rnn_early_ictal_field/figures/README.md`
- spec：`docs/superpowers/specs/2026-08-01-topic5-history-rnn-early-ictal-field-v0_1.md`
- plan：`docs/superpowers/plans/2026-08-01-topic5-history-rnn-early-ictal-field-v0_1.md`

## 10. 最终验收

- 科学合同执行至预注册停止点：**100%**；
- G0/G1 与 target seal：**PASS**；
- G1 scientific gate：**FAIL**（primary `M2−M1` 层面；顺序对照本身通过）；
- G2/G3：**LOCKED_NOT_RUN**；
- 可复现性审计：**PASS**；
- 顺序对照组成 bug 已修复并重跑 102/102（§5.6），回归测试锁住组成一致性；
- 最终 PNG 已完成三轮目视检查，修复了模型框重叠、统计注释遮挡、误导性 early-ictal 标题，以及把 p=3.9e-05 打印成 `p=0.000` 的取整；PDF/JSON/README 同步生成。

本线在 v0.1 冻结结束，不再通过增加 seeds、改变 half-life、扩 hidden size 或读取 early-ictal target 来追逐阳性。

## 11. 审阅后状态分层

```text
ENGINEERING_EXECUTION: PASS
G1_NEXT_EVENT_PROXY: PROVISIONAL_BOUNDED_NEGATIVE
CHRONOLOGY_SPECIFIC_STATE: NOT_SUPPORTED_UNDER_CURRENT_OBJECTIVE
LATENT_STATE_TO_EARLY_ICTAL_FIELD: NOT_TESTED_IN_V0.1
HISTORY_DEPENDENT_NETWORK_RECONFIGURATION: NOT_TESTED
CAUSAL_NETWORK_SHAPING: NOT_ESTABLISHED
```

下一合同：`docs/superpowers/specs/2026-08-02-topic5-history-rnn-direct-early-ictal-transfer-v0_2.md`。
