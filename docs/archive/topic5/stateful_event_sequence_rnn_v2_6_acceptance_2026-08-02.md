# 审阅结论

## 1. 一句话判断

RNNv2.6 已经正确实现“以整场间期事件为时间步、hidden state 在 source 内连续传递”的核心模型，并完成 34 人全队列验收。它建立的是**状态跟踪**：稳定 repertoire 的短期表达存在约 10–20 场事件尺度的可预测变化；它没有检验并证明“事件塑造网络”。训练结果没有超越简单 recency smoothing，最合理的模型收缩是低维 leaky state，而不是终止 evolving-state 这条科学线。

> **正式验收状态：`ACCEPTED_AS_STATE_TRACKING_PRECURSOR_WITH_KNOWN_TRAINING_BIAS`。** 验收已经完成；90/100 表示证据与实现仍有已知限制，不表示状态未裁决。

> 2026-08-03 代码与数值复核：所有承重数字已逐条对着冻结产物重算，除 RNN vs static 的 CI 下界（已订正）与 validation candidate 计数（已订正）外全部一致；四条结论方向全部成立。本稿已按复核结果修订，修订点在 §2、§3 P1-1/P1-4/P1-5/P1-6、§4.3、§4.4、§4.5、§4.7、§5、§6、§8。

## 2. 完成程度

> **完成度：90/100**（2026-08-03 复核后从自评的 96 下调，见下）

已完成：

- 34/34 患者 validation-only profile selection；
- RNN、GRU、LSTM 作为同一 recurrent hypothesis 的数值参数化；
- learning rate、optimizer、normalization、hidden size、layers、TBPTT 和 update batch 的患者级筛选；
- 760 个 validation candidate fits（748 个 architecture+refinement 筛选 fits，加 epoch-boundary audit 在 4 位患者上追加的 12 个 100-epoch fits）；
- epoch-boundary audit，以及 profile 与训练预算共同冻结；
- 34/34 untouched test，3 seeds/patient，共 102 个 checkpoint；
- source-coherent block-order null；
- source-level time-reversal null；
- dense-anchor、state-reset、memory-curve 和 H=40 sensitivity；
- synthetic long-memory、state-carry、chunk parity 和 target-construction tests。

未计满分的原因：

1. 当前仍属于同一队列内的 exploratory nested validation/test；旧 `heldout20` 已排除，但还没有一个完全未参与 Topic 5 决策的独立 replication cohort。
2. 2026-08-03 复核发现的四项缺口，已在本稿修复或标注：RNN vs static 此前没有冻结产物且 CI 下界写错（P0，已补产物并订正）、承诺的支持度分层没有落盘（P1-1，已补）、early stopping 以静态初始化为基准压低训练预算（P1-6，已量化，代码留待下一版）、以及 within-source / 块间顺序两处覆盖范围未写明（P1-4、P1-5，已补）。

## 3. P0 / P1 关键问题

### P0（2026-08-03 复核发现，已修复）

复核前本节写的是「没有剩余 P0」。实际存在一个：**结论 3「RNN 超过静态 repertoire」是四条结论里唯一没有任何冻结产物支撑的。** `STATEFUL_TEST_STATE.json` 只写了 RNN 对 EWMA 的三组统计，RNN 对 static 的中位差、计数、CI 与 p 值是临时算出后直接写进文档的；其中 bootstrap CI 下界 `−0.1403` 与从冻结 per-patient JSON 重算得到的 `−0.1629` 不符（中位 −0.0619、25/34、`p=0.00385` 三项复算完全一致，结论方向不受影响）。

已修复：`scripts/accept_topic5_stateful_event_rnn_v2_6.py` 只读冻结 per-patient 产物重新推导 RNN vs static 并落盘到 `acceptance/ACCEPTANCE_STATE.json`；`tests/test_topic5_stateful_event_rnn_v2_6_acceptance.py` 里有一条把派生统计逐位锁到 `STATEFUL_TEST_STATE.json` 主端点，另一条锁住 state-tracking / state-shaping 的最终机器判决，保证派生层与冻结层同源。§4.4 的 CI 已订正。

其余 P0 面检查通过：34 位患者均有完整 JSON、预测 artifact 和 3 个 checkpoint；34×7 组 per-subject 产物的 `contract_checks` 全部为 true；所有正式状态文件均为 complete；config / core module / primary runner 的 sha256 与 `FROZEN_VALIDATION_STATE.json` 仍逐位一致；旧 `heldout20`、A/B/axis、geometry、SOZ、ictal 和 SNN 输入均未进入模型。

附件提出的 null provenance 疑问也已逐项核清：冻结 block-shuffle 与 reversal 结果分别由
`scripts/run_topic5_stateful_event_rnn_v2_6_block_null.py` 和
`scripts/run_topic5_stateful_event_rnn_v2_6_reversal_null.py` 生成；两个 runner 的当前 SHA-256
与对应完成状态文件逐位一致。旧的 row-wise `shuffled_histories()` 与 target-only
`circularly_shift_targets()` 仍存在于 v2.3 时代的模块中，但 v2.6 正式 null 没有调用它们，
因此这里不存在报告与正式实现不一致的 P0。

### P1-1：正式 test 支持度差异很大，且低支持患者独自撑起 RNN−EWMA 的负号

H=20 formal test 共 6,490 个非重叠 target windows，但患者间从 1 到 1,096 个不等（中位 56.5）。4 位患者只有 1–3 个窗口（`yuquan_huanghanwen` 1、`yuquan_litengsheng` 2、`yuquan_sunyuanxin` 2、`yuquan_songzishuo` 3），却与有 1,096 个窗口的患者在队列统计里等权。

按支持度分层后（`acceptance/ACCEPTANCE_STATE.json::support_strata`），全 34 人 RNN−EWMA 的轻微负中位数**由少数低支持患者独自撑起**：只要去掉 5 位窗口数不足 10 的患者，方向就翻正，再往上继续分层方向不变：

| 最少 formal windows | n | RNN−EWMA 中位差 | RNN 更好 | Wilcoxon p |
|---:|---:|---:|---:|---:|
| ≥1（全队列） | 34 | −0.0248 | 18/34 | 0.076 |
| ≥10 | 29 | **+0.0312** | 13/29 | 0.375 |
| ≥20 | 27 | **+0.0378** | 11/27 | 0.607 |
| ≥50 | 20 | **+0.0402** | 8/20 | 0.622 |

同一分层下 RNN−static 的方向与显著性反而单调增强（≥50：中位 −0.0826，17/20，p=0.00029），说明分层没有削弱阳性结论，只削弱了 RNN 相对 EWMA 的表面优势。

处理方式：正式报告同时给出 formal endpoint、dense sensitivity 和支持度分层；「RNN 未稳定超过 EWMA」按最强证据写，不能只引用 p=0.076 让读者以为差一点就显著。

### P1-2：两个数据集表现不同

Yuquan 的 dense RNN−EWMA propagation 中位差为 −0.172，Epilepsiae 为 +0.041。但 Yuquan 在 block-shuffle null 中 RNN 的相对优势更大，说明该差异不能解释为真实 chronology 被辨识，更像 RNN 相对 EWMA 的正则化或数据结构差异。

处理方式：两个数据集分别报告，不把 Yuquan 的表面优势升级成 chronology claim。

### P1-3：不能写成 network shaping

模型证明的是过去事件对未来 repertoire prediction 有短程增量。它没有观测或辨识 anatomical graph change，也没有排除 sleep、medication、recording state 等未观测变量。

允许写 `short-range event-history state` 或 `recency-dependent repertoire modulation`；不允许写 activity-dependent plasticity、network formation 或 causal shaping。

这里要进一步区分 observer 与 updater。当前模型支持的是：

\[
E_{\le e}\rightarrow \widehat z_e\rightarrow D(E_{e+1:e+H}),
\]

即近期事件帮助估计当前慢状态。未观测状态完全可能同时生成当前事件与未来事件，所以
“最近事件提供预测信息”不能改写成“最近事件影响了未来网络”。真正的 updater 检验必须先用
过去事件估计 event 前状态，再问当前事件超出该状态预测的 innovation 是否解释随后状态残差。

### P1-4：可检验的历史长度被 source 边界硬性封顶，「无长程」只覆盖单段记录之内

按合同，hidden state 在每个 source 开头清零，训练时 source 之间还会整体打乱顺序。因此模型**在结构上根本无法跨 source 积累历史**：任何跨记录段、跨天的过程都不在被检验的模型族里。

每位患者的 source 长度中位数（跨 34 人取中位）为 294 个事件，最长 source 的中位为 818 个事件。所以本轮「长程 chronology 没有额外信息」的真实覆盖范围是**单个 source 之内的事件块排列**，不是整段住院记录的时间轴。

处理方式：凡是写「no long-range chronology」的地方必须带上 within-source 限定；把它当作否定 multi-day network shaping 的证据是越界的（这也是 P1-3 之外的独立理由）。

### P1-5：block-shuffle null 的块长恰好等于 horizon 和 anchor 间距，只打乱「块与块之间」的顺序

`block_size = horizon = 20`，而 formal anchors 落在 19、39、59…（间距 20）。这三者相等意味着每个 formal target window 恰好是一整块，打乱后这些窗口作为**完整单元被整体搬家**：窗口内部的事件顺序、窗口内部的组成全部原样保留，被破坏的只有「哪一块跟在哪一块后面」。

这让该 null 成为一个偏保守的 chronology null：它能否定「块间顺序携带信息」，**不能**否定「窗口内部顺序携带信息」。后者本轮没有测。

另外必须披露反向尾：`true − shuffled` 中位 +0.0176、13/34 真实顺序更有利，注册方向 Wilcoxon p=0.967，而**反方向 p=0.035**——打乱顺序反而让 RNN 相对 EWMA 的优势变大。这不能读成「真实顺序有害」，只能读成两臂不可交换：打乱同时改变了 RNN 和 EWMA 两条腿，差分对比里剩下的不是纯 chronology 效应。

### P1-6：early stopping 以 epoch −1 静态初始化为基准，8/102 正式 run 被压到最低预算

`fit_stateful_event_rnn` 的 patience 计数器初始化在 epoch −1 静态模型的 validation 分数上。后果是：一个前 8 个 epoch 打不过自己静态初始化的 profile，会在 `minimum_epochs=8` 处直接停掉，而不是用满 40 个 epoch。

实测影响：102 个正式 test run 里 8 个（分布在 6 位患者）落在这个路径；748 个筛选 fits 里 79 个（10.6%）如此，其中 69 个的最佳 epoch 停在 0。这既压低了 trained 臂，也让筛选阶段各 profile 的训练预算不等。

方向审计（`acceptance/ACCEPTANCE_STATE.json::training_budget_audit`）：
- 对 **RNN > static 这个阳性结论是保守的**——截断只会让 RNN 更差；剔除这 6 位患者后阳性反而更强（中位 −0.0818，21/28，p=0.0055）。
- 对 **RNN 未超过 EWMA 这个阴性结论不是天然保守的**，所以必须做剔除检验：剔除这 6 位患者后中位为 **+0.0088**（14/28，p=0.226），仍然没有 RNN 优势。阴性结论在这个缺陷下站得住。

处理方式：本轮冻结产物不重跑（改动 `src/topic5_stateful_event_rnn_v2_6.py` 会让 102 个 checkpoint 和 9 个 state 文件的 hash 校验全部失效）。下一版必须把 patience 计数器改成跟踪 trained checkpoint 自身的最优值，静态 epoch −1 只作报告用的 fallback，不作停机基准。

## 4. 科学性验收

### 4.1 稳定 repertoire 是前提

既有 split-half 和 odd/even 结果支持同一患者的 propagation repertoire 跨时间稳定。v2.6 不再否定这个前提，而是检验稳定 backbone 的表达是否具有跨事件状态。

### 4.2 模型确实使用了历史

把 hidden state 在每场事件后清零，propagation score 中位恶化 +0.0257；25/34 患者变差，单侧 Wilcoxon `p=0.0051`。在至少 20 个 formal test windows 的 27 位患者中，21/27 变差，`p=0.0021`。

因此当前 RNN 不是伪装成 recurrent 的单事件模型。

### 4.3 训练好的 state 所整合的历史长度约 10–20 场事件

同一 checkpoint 的 inference-only reset curve：

| reset interval | propagation penalty，中位数 | penalty 恰为 0 的患者 |
|---:|---:|---:|
| 1 event | +0.02572 | 0/34 |
| 5 events | +0.00946 | 0/34 |
| 10 events | +0.00452 | 0/34 |
| 20 events | +0.000155 | 2/34 |
| 50 events | +0.000092 | 3/34 |
| 100 events | 0 | 5/34 |

读法有两个必须写明的限定：

1. **1/5/10/20 这四档是干净的，50/100 两档不是。** formal anchors 落在 19、39、59…，而 20 是 1/5/10/20 的整数倍，所以在这四档下每个 anchor 拿到的上下文**恰好**是 1/5/10/20 场事件。50 和 100 不整除 20，anchor 拿到的上下文在 10–50 与 20–100 之间循环变化；而且对 source 短于该间隔的患者根本不发生 reset，penalty 恒为 0（分别 3 位和 5 位患者）。所以「20 之后饱和」的证据来自 1→5→10→20 这段，不来自 50/100 两档。
2. **这是训练好的模型的 state 属性，不是对数据本身记忆时间常数的独立估计。** 如果模型压根没学会长程结构，曲线同样会早早饱和。可以写「该 recurrent state 主要整合最近约 10–20 场事件」，不能写「间期事件序列的记忆长度就是 10–20 场」。

这个尺度与固定 EWMA `decay=0.95` 的有效记忆长度接近——这也正是 §4.4 里 RNN 打不过 EWMA 的机制层面解释。

### 4.4 RNN 超过静态 repertoire，但没有超过 EWMA

H=20 formal endpoint（全部由 `acceptance/ACCEPTANCE_STATE.json::comparisons` 冻结，`reproduces_frozen_primary_endpoint=true`）：

- RNN vs static：中位差 −0.0619，25/34 更好，bootstrap 95% CI `[−0.1629, −0.0063]`，Wilcoxon `p=0.00385`；
- RNN vs EWMA：中位差 −0.0248，18/34 更好，CI `[−0.1389, 0.0425]`，`p=0.0764`；
- dense-anchor sensitivity：RNN vs EWMA 中位差 +0.0294，16/34 更好，CI `[−0.1240, 0.0514]`，`p=0.163`；
- 参考量 EWMA vs static：中位差 −0.0858，22/34 更好，`p=0.228`——EWMA 相对静态的优势本身也没到显著。

validation RNN−EWMA 与 dense test RNN−EWMA 相关 `rho=0.505, p=0.0023`，说明训练信号不是随机；但它没有产生稳定超越 EWMA 的 test increment。

两条必须一起写的强度限定：

- **支持度分层后 RNN−EWMA 的负号消失**（见 P1-1 表）：≥20 windows 的 27 位患者中位为 +0.0378、11/27。所以 `p=0.0764` 不应被读成「差一点就显著」。
- **效应量小于 seed 噪声**：患者内跨 seed 的 propagation 分数标准差中位为 0.0285，大于队列中位差的绝对值 0.0248。每位患者取 3 个 seed 的中位数已经压掉一部分噪声，队列层统计仍然成立，但单个患者的正负号是 seed-labile 的，不能拿单患者方向讲故事。

RNN vs static 这一条此前没有任何冻结产物，本轮由 `scripts/accept_topic5_stateful_event_rnn_v2_6.py` 从冻结的 per-patient JSON 重新推导并落盘；旧稿写的 CI 下界 `−0.1403` 与产物不符，已订正为 `−0.1629`。

### 4.5 单段记录之内的块间顺序和时间方向没有额外信息

source-coherent block-order null（块长 = 20 = horizon = anchor 间距，覆盖范围见 P1-5）：

- `true gain − shuffled gain` 中位数 +0.0176；
- 13/34 患者真实顺序更有利；
- CI `[−0.0097, 0.0918]`；
- 注册方向 Wilcoxon `p=0.967`；**反方向 `p=0.035`**。

source-level time reversal：

- `true gain − reversed gain` 中位数 −0.0088；
- 19/34 患者真实方向更有利；
- CI `[−0.0545, 0.0690]`；
- 注册方向 Wilcoxon `p=0.513`；反方向 `p=0.493`（对称，无异常）。

真实 chronology 既没有超过 block shuffle，也没有稳定超过反转时间方向。

但 block-shuffle 那条的反方向名义显著必须写出来，并且不能升级成「真实顺序有害」：`true − null` 是一个差分对比，打乱同时改变了 RNN 和 EWMA 两条腿，反向信号更可能来自「EWMA 在打乱数据上退化得比 RNN 多」或「打乱起到了训练数据增广的作用」，而不是真实时间顺序本身是负担。它的正确用途是提示两臂不可交换——这个 null 只能支持「RNN 相对 EWMA 的优势不来自块间顺序」这一条，不能支持任何关于真实顺序方向的主张。

### 4.6 更长预测 horizon 没有改变结论

H=40 sensitivity 使用同一 source split、冻结 profile 和训练预算：32 位 eligible、2 位 ineligible。RNN−EWMA propagation 中位差 +0.0037，16/32 更好，CI `[−0.1331, 0.0746]`，`p=0.211`。

因此不触发 H40 null 扩展。

### 4.7 当前安全结论

> 同一患者的间期事件反复采样一个稳定传播 repertoire。以完整事件为时间步的连续 recurrent model 能够利用最近约 10–20 场事件的历史改善相对于静态 repertoire 的未来传播预测，但这一增量可由简单 recency smoothing 解释，并不依赖**同一段记录之内**更长的事件块排列或时间方向。

这是一个有边界的阳性结果：短程 event-history state tracking 成立；单段记录之内的长程 chronology-specific recurrent computation 当前不成立。EWMA 本身就是
`z_{e+1}=\rho z_e+(1-\rho)y_e` 形式的最小状态模型，所以“RNN 未超过 EWMA”说明当前可检测状态
接近简单的线性泄漏积分，不等于 evolving-state 假设失败。

边界要连着讲，缺一条就会被读成更强的结论：

- 「长程」只到一个 source 之内（患者中位 294 个事件），跨 source / 跨天从未进入模型（P1-4）；
- 「块排列」只指 20 事件块之间的顺序，块内顺序未测（P1-5）；
- 「RNN 未超过 EWMA」在支持度分层和训练预算剔除两个检验下都站得住（P1-1、P1-6），但效应量在 seed 噪声量级，不能反过来说「EWMA 优于 RNN」。

### 4.8 与最初科学问题的分层验收

| 证据层级 | v2.6 判决 |
|---|---|
| 稳定患者特异传播 repertoire | 已由既有 split-half / odd-even 建立，作为本轮前提 |
| repertoire 表达并非完全静止 | 支持 |
| 过去事件历史预测未来 repertoire | 支持 |
| 信息超越简单 recency observer | 未支持 |
| 当前事件 innovation 定向预测后续状态更新 | 未检验 |
| activity-dependent network shaping | 未建立 |

因此 v2.6 不是 network-shaping 阴性终点。它完成了 shaping 假设所需的 state-observability 与
state-tracking 前提，并把下一问收窄为一个可辨识的 event-innovation update test。

这也不重复 v2.2：v2.2 使用未按稳定 repertoire 分层的 block-mean rank/participation field，
并以 block 内 `late-half - early-half` 预测下一 block，最终只有 2 位 eligible pilot。下一合同
改为 source-continuous event-level observer，在 34 人已建立的短程状态基础上，使用 pre-event
filtered state 定义单事件 innovation；它不是再次运行旧 block-delta 分支。

## 5. 工程性验收

- 数据：691,314 个 train80 events；旧 heldout20 未进入（全部使用索引经 `all_indices_train80_only` 断言）。
- targets：dense train/validation/test 分别 385,744 / 127,206 / 127,811；formal validation/test 分别 6,450 / 6,490。
- **anchor 口径**：训练与 **validation 模型选择/early stopping 都用 dense causal anchors**；间距为 H 的 formal anchors 定义 primary test endpoint，并在 validation 与 test 两侧都通过了 `_formal_targets_nonoverlap` 非重叠断言。spec 原文「Validation and test use anchors spaced by H」易被读成 validation 也按 formal 打分，已在 spec §3 订正。这不是泄漏（validation 仍只用 validation 数据），但选择准则与主端点口径不同，是 formal 与 dense 两个方向不一致的来源之一。
- profile selection：760 个 validation candidate fits（748 筛选 + 12 boundary 延长）；GRU 15、LSTM 12、tanh-RNN 7；hidden size 8/16/32/64 均有进入冻结结果。
- training choices：TBPTT 20/50/100、AdamW/RMSprop、none/zscore/robust normalization、1/2 layers、update batch 1/4/8 均有 validation 选择实例。
- boundary audit：4 位触发 100-epoch 延长；2 位真正改选。`epilepsiae_548` validation score 0.8554→0.7728；`yuquan_sunyuanxin` 0.21857→0.21778。
- 训练数值：102/102 final runs finite；test hidden-state max norm 5.65；患者内 seed score SD 中位数 0.0285。
- **训练预算审计**：102 个正式 run 中 8 个（6 位患者）因 early stopping 以 epoch −1 静态初始化为基准而停在最低预算；筛选阶段 79/748 同路径。方向影响与剔除检验见 P1-6。
- checkpoints：102；patient JSON：34；prediction NPZ：34；`nested` checkpoint 实为 `min(epoch −1 静态, 最优 trained epoch)`，34 人中 30 人与 trained 相同，**不是**「原样的 epoch −1 静态初始化」——static 对照另由 `_static_predictions` 直接从 train future-repertoire 均值构造，不依赖该 checkpoint。
- 测试：核心 `7 passed`（target vectorization、RNN/GRU/LSTM chunk parity、long-memory synthetic、state-carry failure control）＋派生层 `5 passed`（合计 `12 passed`）；派生测试同时锁定冻结主端点和最终科学判决边界。
- frozen hashes：config、core module 和 primary runner 均写入 `FROZEN_VALIDATION_STATE.json`；派生层只读冻结产物、不训练，因此不改动这些 hash。

## 6. 最小修改路线

1. V2.7 只修 early stopping 基准（P1-6），使用同一数据、模型族、参数网格、split、seed 与 endpoint，在 `results/topic5_stateful_event_sequence_rnn/v2_7/` 平行重跑；v2.6 原样保留。
2. 不再扩大 cell 或 architecture zoo；V2.7 是训练公平性复跑，不承担新科学问题。
3. V2.7 的正式数值用于 manuscript-facing state-tracking 结果；无论它是否改变精确效应量，都不能把 prediction 写成 shaping。
4. 新科学合同直接比较 fixed backbone、autonomous drift、leaky observer、离散切换和 event-innovation low-rank update。核心问题是 innovation 对后续状态残差的增量，不是 GRU 是否更强。
5. 新模型必须把 observer 的状态估计更新与 generative state transition 分开；否则 `B\nu_e` 仍可能只是滤波器在修正自己，不能解释为网络更新。
6. chronology 对照使用 source-coherent block-size sweep `1/2/5/10/20`、state-matched innovation permutation 和 future-versus-past direction control；不再只依赖块长恰好等于 horizon 的 null。
7. independent cohort 仍是最终确认要求，但不再把“RNN 必须先超过 EWMA”当作是否允许研究 state update 的进入门；EWMA 已经是最小 evolving-state observer。

## 7. 下一步建议

这轮核心目标已经完成：我们不再问同一事件内部“下一 contact 是谁”，而是以完整事件为序列单位，证明患者内 stable repertoire 的短期表达存在可跟踪的跨事件状态。结果不是“RNN 失败”，也不是“网络塑造成立”；它把最小可解释动力学收缩为最近约 10–20 场事件的 leaky observer。

下一阶段不是继续救 GRU，而是检验最初缺失的箭头：控制 pre-event state 后，单事件 innovation 是否与随后低维传播状态更新相关。V2.7 负责清理训练公平性，V3.0 负责这个新的科学问题；两者不能混成一次重跑。论文中 v2.6/v2.7 定位仍是稳定传播 repertoire 上的短程 history-dependent modulation，它目前不能承担 network-shaping bridge。

## 8. 关键产物

- 主冻结状态：`results/topic5_stateful_event_sequence_rnn/v2_6/validation_screen/FROZEN_VALIDATION_STATE.json`
- 正式 test：`results/topic5_stateful_event_sequence_rnn/v2_6/STATEFUL_TEST_STATE.json`
- block null：`results/topic5_stateful_event_sequence_rnn/v2_6/chronology_null/block_shuffle/BLOCK_NULL_STATE.json`
- reversal null：`results/topic5_stateful_event_sequence_rnn/v2_6/chronology_null/time_reversal/TIME_REVERSAL_STATE.json`
- dense sensitivity：`results/topic5_stateful_event_sequence_rnn/v2_6/dense_test_sensitivity/DENSE_TEST_STATE.json`
- state reset：`results/topic5_stateful_event_sequence_rnn/v2_6/state_reset_ablation/STATE_RESET_STATE.json`
- memory curve：`results/topic5_stateful_event_sequence_rnn/v2_6/memory_curve/MEMORY_CURVE_STATE.json`
- H40 sensitivity：`results/topic5_stateful_event_sequence_rnn/v2_6/h40_sensitivity/H40_STATE.json`
- **派生验收层（2026-08-03 补）**：`results/topic5_stateful_event_sequence_rnn/v2_6/acceptance/{ACCEPTANCE_STATE.json, patient_summary.csv}`，由 `scripts/accept_topic5_stateful_event_rnn_v2_6.py` 只读上述冻结产物推导，含 RNN vs static、支持度分层、两个 null 的双尾、seed 离散度与训练预算审计；`reproduces_frozen_primary_endpoint` 字段把它与 `STATEFUL_TEST_STATE.json` 逐位对齐。
