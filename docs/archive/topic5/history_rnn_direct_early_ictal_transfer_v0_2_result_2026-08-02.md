# HistoryRNN direct early-ictal transfer v0.2 最终结果与验收

**日期**：2026-08-02（同日 code review 后修订，见 §5.4）

**最终状态**：`ACCEPTED_SUPPLEMENTARY_TRAINING_SENSITIVE_BOUNDARY`
**定位**：reused-target internal cross-state validation；Supplementary bounded result

## 1. 核心问题

本轮真正检验的是：

```text
某次发作前、按真实时间累积的间期事件历史
→ target-blind HistoryRNN state
→ 随后 clinical-onset [0,10] s、1–150 Hz early-ictal contact energy field
```

它不是 next-contact prediction、完整间期事件生成、自动恢复 A/B 轴，也不是发作时间预测。上一版 next-event G1 只负责 target-blind representation learning 和训练诊断，不再拥有锁死 early-ictal target 的权力。

## 2. 一句话结论

最长冻结训练预算 c30 下，HistoryRNN 相对 `static + unordered history` 的 early-ictal 场增量未通过患者级检验，真实顺序也没有优于严格顺序打乱，非零状态没有优于 zero-state；RNN 预测场本身未超过全通道打乱，正确发作前状态也不比同患者错误发作状态更匹配 target。

c10 仍保留相对增量和 zero-state 两项阳性，但真实顺序优于严格顺序打乱这一项，在按合同实现的全前缀打乱下 **c10 与 c30 均未通过**（早期版本在 c10 观察到的顺序阳性来自只打乱最近 64 个事件的对照实现，已撤回，见 §5.4）。因此最终结论是：

> 当前 frozen self-supervised HistoryRNN 没有提供训练预算稳定的 early-ictal field 预测证据。两种训练预算都一致不支持真实事件顺序、绝对空间预测和 seizure-specific state。

这不推翻论文既有的 sign-free static interictal–ictal morphology correspondence；它只说明不能把稳定患者 scaffold 升级成当前模型可读出的跨事件动态状态。

## 3. 数据与防泄漏合同

- target：clinical onset 后 `[0,10] s`、baseline-normalized broadband `1–150 Hz` contact energy field；
- history guard：所有动态输入截止于 onset 前 10 min；
- causal prefix：仅使用同一连续记录片段、且位于上一次 postictal 结束后的间期事件；
- exact join：按 contact name 连接 frozen interictal scaffold、rank dataset 与 early-ictal target；
- evaluation：target-patient LOSO；held-out patient 的 target 不进入 readout 拟合；
- primary：排除 development patient `epilepsiae_1146`，15 人、31 次合格发作；
- supportive：16 人、33 次合格发作；
- target inventory 原有 16 人、106 次 strict clinical-onset 发作；只有 33 次满足同一连续记录段内至少 8 个 causal events；
- patient-first：先逐 seizure 计算 contact-wise Spearman，再对同一患者取 seizure median；
- **空间分母**：每患者的 contact field 只覆盖冻结间期骨架与 rank 数据集按名字精确取交后的 `6–16` 个触点（中位 `9`），不是完整 SEEG 蒙太奇；该分母由上游 `topic5_interictal_rank_distribution/dataset_v0_4` 的 per-subject 触点集决定。所有 Spearman、channel-shuffle null 与患者级对比都建立在这个分母上，因此阳性与阴性都是粗分辨率结论（6 触点患者只有 720 种不同通道排列）。逐患者分母写入 `DIRECT_TRANSFER_SUMMARY.json::contact_denominator`；
- absolute null：每患者 5000 次 all-contact target-label shuffle，同一批 permutation 对所有模型复用；
- target reuse：frozen static scaffold 和 early-ictal target 已被既往分析使用，因此本轮是 internal validation，不是独立复制或临床前瞻预测。

## 4. 比较模型

| 模型 | 信息 |
|---|---|
| M0 | causal participation prior、geometry、frozen TA/TB scaffold |
| M1 | M0 + unordered history、last event、event count/span/gap |
| E0.5/E2/E6 | M1 + 0.5/2/6 h causal EWMA activity field |
| EM | M1 + multi-horizon EWMA field |
| R2 | M1 + target-blind HistoryRNN contact field |
| controls | R2 strict order shuffle、R2 zero-state、EWMA time-slot shuffle、同患者错误 seizure-state pairing |

Primary direct contrast 是 `R2 − M1`。绝对 channel-null、order、zero-state 和 pairing 分别回答不同层级，不能互相替代。

## 5. 训练是否充分

### 5.1 c30 执行

- 16/16 target-patient checkpoints 完成；
- history cycles：30；
- learning rate：`3e-4`；
- hidden state：16；
- BPTT chunk：256；
- segment batch：16；
- 0 failed、0 OOM；
- encoder 全程 `target_values_read=false`。

### 5.2 c30 state branch 确实被使用

对最终 16 个 c30 checkpoint 的审计显示：

- 16/16 history readout norm 非零；
- participation readout norm 中位 `3.890`；
- G1 上相对 zero-state/base 的中位 gain `0.00855`；
- G1 上相对 capacity-matched M1 的中位 gain `0.00528`；
- learned half-life 相对 2 h 初始化的绝对漂移中位 `0.316 h`；
- 最后一轮相对前一轮 train loss 变化中位 `−1.01×10⁻5`；
- direct causal-history trajectory variance 中位 `0.0402`。

因此 c30 阴性不能归因于 history branch 没接上、状态恒为零或完全没有训练。它仍不能证明 state 是可辨识的生物慢变量。

### 5.3 训练预算仍影响科学标志

c10 与 c30 的五个预设标志（均为 §5.4 修正后重跑的值）：

| 标志 | c10 | c30 |
|---|---:|---:|
| R2 相对 M1 增量 | 通过 | 未通过 |
| true order 优于 strict shuffle | 未通过 | 未通过 |
| nonzero state 优于 zero-state | 通过 | 未通过 |
| absolute field 超过 channel null | 未通过 | 未通过 |
| correct seizure pairing | 未通过 | 未通过 |

仍有两项标志在两个预算之间翻转，因此 `DIRECT_TRAINING_BUDGET_COMPARISON` 保持 `TRAINING_BUDGET_SENSITIVE_SCIENTIFIC_VERDICT`。不能按 early-ictal target 在 c10 与 c30 中挑较好者，也不能写“已收敛”。

但翻转的内容与修正前不同：现在对训练预算敏感的只有“相对 matched baseline 的增量”和“非零状态优于 zero-state”两项，**真实事件顺序在两个预算下一致地没有优势**。翻转的两项问的都是“状态分支是否带来可学的额外容量”，不是“事件顺序是否携带信息”。

### 5.4 同日 code review 触发的两处方法学修正

1. **顺序对照的作用域**（P0）。合同 §5.1 要求把 last event 之前的**整段** causal prefix 重新分配到既有时间槽；先前实现只打乱最近 64 个事件（`_history_final_order_control` 的 `window=64` 默认值）。33 次合格发作中有 23 次的 causal prefix 超过 64 个事件，实际被打乱的中位比例只有 `20.6%`，最长的一段（6125 事件）只有 `1.0%`；按事件时间算，被打乱的窗口中位只跨 `0.9 h`，而状态半衰期约 2 h。修正为全前缀打乱后重跑 c10 与 c30 全部 32 个 fold。RNN 与 EWMA 两个时间对照共用同一置换，修正同时作用于两者。
2. **含并列患者的 Wilcoxon 口径**（P0/P1）。患者级对比是 6–16 触点上的 Spearman 差，天然产生大量精确并列以及 `~1e-17` 的浮点残差。先前实现把 `5.55e-17` 计为“为正”；并且在存在精确零时 SciPy 的 `method="auto"` 会在内部丢弃零之后退回正态近似（n=4 全为正时给 `P=0.034`，精确检验为 `P=0.0625`）。现统一以 `1e-9` 为并列带，先剔除并列再做单侧 Wilcoxon（此时 `auto` 会选精确零分布），并在每个对比中同时报告正 / 负 / 并列三个计数。

`R2 − M1`、`R2 − zero-state`、绝对场、channel-null margin 与 target headroom 不受修正 1 影响（这些对比不使用打乱状态），数值与修正前逐位一致；变化的只有各对照臂以及含并列的 P 值。修正前的全部产物保留在 `diagnostic_archives/c30_window64_order_control_2026-08-02/` 与 `results/topic5_history_rnn_direct_c10_candidate_v0_2/diagnostic_archives/c10_window64_order_control_2026-08-02/`。

缺陷来源：64 这个窗口是从 v0.1 的 next-event 合同继承来的。v0.1 spec §（正式 null 段）**明确要求**只置换最近 64 个事件、并明确禁止整段置换，理由是 next-event 任务里每个 decision 有自己的 prefix，整段置换会改变每个 decision 的 prefix event set。这个理由对 v0.2 不成立：direct transfer 每次发作只有一个预测点，整段置换不改变事件集合，只改变身份到时间槽的分配——这正是 v0.2 spec §5.1 要求的。默认值被沿用时没有重新对照 v0.2 的合同，属于 CLAUDE.md §6.1“helper 复用要按问题匹配、不能按签名匹配”的典型形态。

因此 v0.1 的 next-event proxy 结论不受本次修正影响：那里窗口是合同规定的，且被显式记录为 `prefix_order_window_events=64`、模型名 `prefix_matched_order_shuffle_k64`；同时那里的主张方向是“打乱**会**变差 ⇒ 状态依赖顺序”，弱扰动只会让该结论更保守。v0.1 draft 的正文与图注已补上“最近 64 个事件”这一限定。

## 6. c30 最终 direct 结果

### 6.1 RNN 没有稳定超过 matched baseline

`R2 − M1`：

- median Δρ = `+0.0571`；
- bootstrap 95% CI `[-0.0357, 0.1429]`；
- 8 正 / 7 负 / 0 并列（n=15）；
- one-sided Wilcoxon `P=0.1384`。

效应方向不为负，但患者级证据不足，primary contrast 未通过。

### 6.2 真实顺序未提供任何证据，非零状态只在较短预算下有优势

| 对照 | median Δρ | 正 / 负 / 并列 | P |
|---|---:|---:|---:|
| R2 − strict order shuffle（c30） | `0.0000` | 6 / 5 / 4 | 0.2065 |
| R2 − strict order shuffle（c10） | `0.0000` | 6 / 7 / 2 | 0.3677 |
| R2 − zero-state（c30） | `+0.0571` | 8 / 7 / 0 | 0.1384 |
| R2 − zero-state（c10） | `+0.0667` | 12 / 3 / 0 | 0.0085 |

按合同实现的全前缀顺序打乱下，**两个训练预算都没有出现真实顺序优于打乱**，因此不能支持 chronology-specific state。zero-state 对照在 c10 通过、c30 未通过：这只说明较短预算下 RNN 的状态分支带来了可学的额外读出容量，与“顺序携带信息”是两回事——同一批 c10 checkpoint 在顺序对照上是 6 正 / 7 负。

### 6.3 绝对 early-ictal 场预测未超过全通道 null

R2 absolute patient-median ρ：

- median ρ = `−0.0250`；
- 7 正 / 8 负（n=15）；
- observed ρ − channel-null median = `−0.0214`；
- channel-null margin `P=0.5980`；
- 2/15 患者高于各自 5000 次 null 的 p95；
- 2/15 患者 permutation `P<0.05`。

所以当前模型不是一个可用的 early-ictal contact-field predictor。c10 也未通过这一门（margin median `+0.0357`、`P=0.445`，同样仅 2/15 超过各自 p95），这是跨预算一致的边界。

需要一并记录的边界：**整个模型阶梯的绝对场都没有超过零**——M0 中位 ρ = `−0.2143`（5/15 为正）、M1 = `0.0000`、E0.5/E2/E6 = `−0.0221 / −0.0500 / −0.0706`、EM = `0.0000`、R2 = `−0.0250`。也就是说，连显式吃进论文冻结 TA/TB 骨架的 M0，在这个跨患者共享的**有符号** ridge readout 下也系统性地反向。这与论文既有 sign-free 结论不矛盾（见 §8），但它意味着：本轮阴性的适用范围是“这一族有符号、跨患者共享的 readout”，而不是“间期历史里没有信息”。`R2 − M1` 是两个都不预测的模型之间的相对比较。

### 6.4 简单 EWMA activity integrator 也未建立

E0.5/E2/E6/EM 相对 M1 的 BH-FDR q 均为 `0.382`；各模型 absolute field 也未超过 channel null。因此不能改写为“简单活动负荷已经预测 early-ictal field”。

E2 的 time-slot shuffle 是唯一在两个预算下方向一致的对照：c30 `median 0.0000`、5 正 / 1 负 / 9 并列、`P=0.0469`；c10 `median 0.0000`、5 正 / 0 负 / 10 并列、`P=0.0312`。但它仍然不能升级为 activity-state 阳性，理由有三：E2 相对 M1 与绝对场都阴性（合同 §7 要求对应模型绝对 ρ>0 才算有效预测）；患者中位为 0，2/3 的患者精确并列；两个 time-slot 对照（E2 与 EM）之间没有做多重性校正，而 EM 的同类对照是 `P=0.170 / 0.248`。它最多只是一个“值得在更大分母上复查”的次要敏感性。

### 6.5 发作特异性未通过

R2 正确 history–seizure pairing 相对同患者错误 pairing：

- n=10 patients；
- median Δρ = `−0.00417`；
- 3 正 / 5 负 / 2 并列；
- `P=0.5273`。

只有 3 名 primary patients 具备 seizure-specific residual 的可靠分母；R2 residual median ρ 为 `−0.0591`。因此不能声称 state 区分了同一患者的不同发作。

### 6.6 target 本身有 headroom

10 名重复发作患者中：

- seizure–seizure early-ictal field 相关中位 `0.470`；
- leave-one-seizure-out patient-mean oracle 中位 `0.517`。

所以阴性不能简单归因于 target 完全随机。更准确的解释是：当前 target-blind history representation 与跨患者共享 readout 没有取出这部分稳定或发作特异信息。

## 7. c3/c10/c30 如何一起解释

- c3：只在旧的 64-事件截断对照下跑过，其 order sensitivity 读数按 §5.4 撤回，仅作训练不足的存档；
- c10：R2−M1 与 zero-state 两项阳性；order、绝对 channel-null、seizure pairing 阴性；
- c30：R2−M1、zero-state、order 均未通过；绝对 channel-null 和 seizure pairing 仍阴性。

因此最可靠的不是“RNN 有小幅 early-ictal 增量”，而是：

1. 相对增量（以及与之同源的 zero-state 对照）对训练预算敏感，不能作为论文稳定发现；
2. **真实事件顺序的优势在两个预算下都不存在**——这是本轮最干净的一条阴性，因为 c10 与 c30 方向一致；
3. 绝对 early-ictal field prediction 和 seizure specificity 在 c10/c30 均不支持；
4. 当前模型没有把患者稳定 scaffold 提升为可复现的动态发作前状态。

## 8. 与论文核心主线的关系

本轮已经回到正确科学终点，没有继续停留在 next-contact 或完整事件生成。但结果是有边界的阴性：

```text
稳定的 interictal pathological scaffold ↔ early-ictal morphology
```

仍由既有 sign-free static analysis 支持；而更强的命题：

```text
按真实时间积累的 interictal RNN state → subsequent early-ictal field
```

在当前 frozen representation、15 人 primary 和 reused target 下未建立。

RNN 单独也不能证明“间期活动因果塑造病理网络”。若论文保留本线，应作为 Supplementary boundary result，用来说明动态状态升级经过严格检验但未获稳定支持。

## 9. Paper-ready 六联图

- A：直接跨状态任务与 causal guard；
- B：early-ictal target headroom；
- C：绝对预测相对 5000 次 all-contact null；
- D：EWMA/RNN 相对 static+unordered 的增量；
- E：order、zero-state、time-slot 对照；
- F：within-patient seizure pairing 与 residual 边界。

图：`results/topic5_history_rnn_direct_early_ictal_transfer_v0_2/figures/topic5_history_to_early_ictal_direct_transfer_v0_2.{png,pdf}`

## 10. 已知边界与未修复项

以下几条在同日 code review 中被记录，但**没有**在本轮改动——它们要么是预注册设计的一部分（事后改动等于看着结果调对照），要么是上游数据的限制。它们共同决定了这条阴性的适用范围。

1. **空间分母只有 6–16 个触点**（见 §3）。这是上游间期 rank 数据集的 per-subject 触点集决定的，不是本轮引入的。它同时限制阳性与阴性的分辨率：n=6 时 Spearman 只能取少数离散值，这也是本轮大量患者级对比精确并列的直接原因。
2. **M1 的“last-event gap”协变量是上一次事件间隔，不是最后一次事件到 cutoff 的间隔**。该协变量沿用 next-event proxy 的 `_causal_unordered_summary`，其中第三个标量是 `log1p(previous_iei)`。而各次发作的“最后事件距 onset”实际跨 `0.17–5.5 h`。R2 与 M1 共用同一协变量，因此 `R2 − M1` 不受影响；受影响的是绝对预测。
3. **RNN 状态读在最后一个事件上，没有再衰减到 cutoff；EWMA 臂则是衰减到 cutoff 的**。`causal_ewma_contact_fields` 用 `age = cutoff − t_event`，而 `_history_final` 只在事件之间衰减。在 2 h 半衰期下，最后事件到 cutoff 的 `0.17–5.5 h` 间隔意味着 R2 与 EWMA 两臂的“时间参照点”不同。primary contrast（R2 vs M1）不受影响，受影响的是 §4 的并列机制比较（R2 vs EWMA）。
4. **synthetic positive control 的作用域是短程顺序**。注入信号是 `(t−1, t)` 有序对、序列长 32。它证明的是“这套架构+优化器能恢复一个无序摘要拿不到的**近程**顺序信号”，不能外推成“能在 10–6125 个事件、跨数小时实时衰减的尺度上恢复长程状态”。因此 `SYNTHETIC_RECOVERABILITY: PASS` 不足以把 c30 阴性归因于“信息不存在而非模型学不到”。
5. **训练预算尚未到平台**。`REAL_CONVERGENCE_SUMMARY.json` 里 next-event chronological increment 随预算单调上升（c3 `0.00106` → c10 `0.00214` → c30 `0.00443`），`direction_stable_10_to_30=false`。c30 是本轮审计过的最长冻结预算，不是收敛点。
6. **顺序对照每次发作只抽一个置换**。合同没有规定抽样次数；单次抽样让 `R2 − shuffle` 带有可观的蒙特卡洛噪声。若将来复查这一项，应改为每次发作多抽若干置换取均值。
7. **训练患者的特征来自“见过它们”的 checkpoint，留出患者的特征来自没见过它的 checkpoint**。RNN 按留出患者做 LOSO，但 readout 必须在其余患者上拟合，因此 readout 是在 in-sample 表征上标定、在 out-of-sample 表征上使用。R2 与 M1 两臂同样受此影响，相对对比大体抵消，绝对预测不抵消。
8. **ridge 的 alpha 用未加权 MSE 内层选择**（`_fit_model`），触点多 / 发作多的患者在选参时权重更大；最终拟合本身是 patient-balanced 的。

## 11. 最终可写与不可写口径

### 可以写

> A target-blind recurrent representation of causal interictal history did not provide training-budget-stable prediction of the subsequent early-ictal contact-energy field. Under a strict order control that reassigns the entire causal prefix across the observed timestamp slots, true event order was not better than shuffled order at either training budget. Relative gains over the matched non-recurrent baseline appeared only at the intermediate budget and were not retained at the longest frozen budget, while absolute channel-shuffle and within-patient seizure-pairing tests were negative at both budgets. Fields were scored on 6–16 scaffold contacts per patient, so the analysis is coarse in both directions.

### 不可以写

- RNN 学到了发作前慢状态；
- RNN 能可靠预测发作早期场；
- c10 的相对增量是稳定阳性；
- **c10 出现过顺序敏感性**（该读数来自 64-事件截断对照，已按 §5.4 撤回）；
- c30 已证明所有 RNN 都学不到；
- state decay 是生物时间常数；
- RNN 证明间期事件因果塑造网络；
- 本轮是独立或前瞻性临床验证；
- 本轮阴性说明间期历史里没有信息（它只覆盖“这一族有符号、跨患者共享的 readout”，见 §6.3）。

## 12. 关键产物

- final acceptance：`results/topic5_history_rnn_direct_early_ictal_transfer_v0_2/FINAL_ACCEPTANCE.json`
- c30 summary：`results/topic5_history_rnn_direct_early_ictal_transfer_v0_2/DIRECT_TRANSFER_SUMMARY.json`
- c10/c30 comparison：`results/topic5_history_rnn_direct_early_ictal_transfer_v0_2/training_budget_comparison_c10_to_c30/`
- c30 state audit：`results/topic5_history_rnn_direct_early_ictal_transfer_v0_2/g1_diagnostics/checkpoint_utilization/`
- c30 refit：`results/topic5_history_rnn_direct_early_ictal_transfer_v0_2/g1_refit_c30/`
- c10 candidate：`results/topic5_history_rnn_direct_c10_candidate_v0_2/`
- archived c3 direct：`results/topic5_history_rnn_direct_early_ictal_transfer_v0_2/diagnostic_archives/c3_direct_complete_2026-08-02/`
- 修正前 c30 产物：`results/topic5_history_rnn_direct_early_ictal_transfer_v0_2/diagnostic_archives/c30_window64_order_control_2026-08-02/`
- 修正前 c10 产物：`results/topic5_history_rnn_direct_c10_candidate_v0_2/diagnostic_archives/c10_window64_order_control_2026-08-02/`
- figure：`results/topic5_history_rnn_direct_early_ictal_transfer_v0_2/figures/`

## 13. 最终验收

```text
ENGINEERING_EXECUTION: PASS
G1_NEXT_EVENT_PROXY: PROVISIONAL_BOUNDED_NEGATIVE_FOR_CURRENT_G1_TASK
SYNTHETIC_RECOVERABILITY: PASS_SHORT_RANGE_ORDER_SCOPE_ONLY
FINAL_C30_STATE_BRANCH: ACTIVE
STRICT_ORDER_CONTROL: NOT_SUPPORTED_AT_BOTH_BUDGETS
DIRECT_TRAINING_BUDGET_ROBUSTNESS: FAIL
LATENT_STATE_TO_EARLY_ICTAL_FIELD: NOT_SUPPORTED
SEIZURE_SPECIFIC_STATE: NOT_SUPPORTED
CAUSAL_NETWORK_SHAPING: NOT_ESTABLISHED
OVERALL: ACCEPTED_SUPPLEMENTARY_TRAINING_SENSITIVE_BOUNDARY
```
