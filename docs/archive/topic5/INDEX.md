# Topic 5 Archive Index

- **2026-08-18 Epi-PRSSM v0.1 新合同（审阅后修订）**：
  [`scientific spec`](../../superpowers/specs/2026-08-18-topic5-epi-prssm-v0_1.md) ·
  [`figure contract`](../../superpowers/specs/2026-08-18-topic5-epi-prssm-figure-contract.md) ·
  [`implementation plan`](../../superpowers/plans/2026-08-18-topic5-epi-prssm-v0_1.md) ·
  [`autonomous agent prompt`](../../superpowers/plans/2026-08-18-topic5-epi-prssm-autonomous-agent-prompt.md)。
  主轴改为 H1=可 open-loop 的慢状态、H2a=状态调制事件分布、H2b=冻结状态连接
  preictal/early-ictal、H3a/b=IED exposure 是否更新间期功能状态及是否与发作转换方向一致。H3 是独立机制扩展，
  不再作为 H1/H2 的总 gate；generator 改为 G0 leaky → G1 graph-CLDS → G2 graph-GRU-ODE →
  G3 resource-anchored 阶梯，primary observer 不逐事件改写 resource。全项目只保留数据/泄漏完整性、
  seizure-label 前 interictal freeze、正式 untouched test 三个硬门，其余阴性结果自动降低对应 claim，实验继续。

- **2026-08-18 Epi-PRSSM v0.1 首轮探索性执行结果**：
  [`白话版报告`](epi_prssm_v0_1_plain_chinese_report_2026-08-18.md) ·
  [`技术报告`](epi_prssm_v0_1_technical_report_2026-08-18.md) ·
  机器可读汇总 `results/epi_prssm/v0_1/FINAL_RUN_SUMMARY.json` ·
  接手说明 `results/epi_prssm/v0_1/CURRENT_HANDOFF.md`。
  34 位患者、864,163 次间期事件、2,097 个记录块；Hard Gate A 通过，与上一代冻结管线的
  通道/事件映射 34/34 逐元素一致（`results/epi_prssm/v0_1/baseline/CONTACT_RNN_PARITY.md`）。
  **全项目尺子：逐患者中位数只有 3.8% 的参与度方差是随时间变的**，其余是各自固定的 repertoire。
  三处由 just-in-time synthetic 或代码审计抓出、且会让结论作废的问题已修并重跑：
  (1) H1 第一级台阶原本与「只有固定习惯」的臂相比，增益主要来自适配器逐触点参数，已改为与
  容量配平的冻结状态臂相比；(2) 资源类 synthetic 真值原本走 spec 禁止的「直接改触点兴奋性」
  通路，已改写为调制潜状态到读出的增益；(3) 状态时间常数写成 `softplus(log τ)` 时实际初始化在
  5.7 秒且训练预算内最多到约 20 秒，**模型在结构上无法表示慢状态**，已改为对数空间指数参数化，
  受影响运行归档在 `results/epi_prssm/v0_1/_invalidated_tau_parametrisation/`。
  H2b 的分母瓶颈是数据取法（间期事件按确定无疑的间期时段挑选，发作附近被排除，
  最后一个间期事件通常在数小时以前）；early-ictal transfer 未运行（盲法临床起始触点 0/71）。
  正式未触碰检验分区**一次都未开启**，本轮全部为开发分区结果。

- `event_innovation_v3_0_execution_handoff_2026-08-03.md` — V3.0 已完成边界、实时进程、剩余 cumulative aggregate / acceptance / 归档步骤与禁止偏离项。**已闭环**，执行结果见下方 `event_innovation_v3_0_acceptance_2026-08-03.md`。

## 当前后续合同（2026-08-03）

- `docs/superpowers/specs/2026-08-03-topic5-stateful-event-sequence-rnn-v2_7-repair.md`
  + matching plan：只修 early stopping，按原合同平行重跑 34 人，不增加模型容量。
- `docs/superpowers/specs/2026-08-03-topic5-event-innovation-low-rank-state-space-v3_0.md`
  + matching plan：以 rank/precedence 为主状态，依次检验 innovation 有效性、单事件多时距脉冲响应和重复 innovation 的累积/抵消；这是 association/local-projection 层，不把 observer correction 提前写成 state transition。
- `docs/superpowers/specs/2026-08-03-topic5-event-innovation-recurrent-transition-v3_1.md`
  + matching plan：observer-only 与 event-driven transition 共享 observation/filter/state dimension，唯一额外项位于 latent transition；synthetic 可并行，人体检验由冻结 v3.0 train/validation handoff 触发。34 人仍为 exploratory，独立复现后才允许 `supports activity-dependent shaping`。

- `stable_repertoire_event_history_v2_4_acceptance_2026-08-02.md` — event-indexed
  history v2.4 P0 repair, six-patient development freeze, locked 28-patient extension and
  final bounded-negative recurrent-state verdict.

> **主入口**：`docs/topic5_seizure_subtyping.md`（§5 历史文档索引含完整 backlink）
> **范围**：以 ictal seizure 本身为研究对象（subtype / pre-ictal / propagation / outcome）。
> **不属于**：interictal 事件内部传播（topic1）、IEI/PSD（topic2）、SOZ 空间归因（topic3）、模型层（topic4）。

## 主线（network-axis pivot）

### `event_innovation_v3_0_followup_census_and_drift_2026-08-04.md` — **收口后续：一半人"看不清"是记录太短；实测效应只有可探测下限的一半；骨架自身按钟表在漂**
- **不改** V3.0 冻结产物与 Level 1 判决，**不反开** V3.1；骨架漂移那一项的预注册档位是探索性描述，不是队列主张。
- **覆盖诊断**：有效 17 人中位记录 96.7 h / 17,196 事件；"历史不够"9 人只有 21.4 h / 1,535 事件（差 4.5 倍时长、11 倍事件量）→ 这一半是**队列覆盖问题**不是生理结论。"残差仍可被过去预测"8 人不是长度问题（19.4 h 但 7,191 事件、触点 15.5），与前一组**不可合并**。有效组触点数（8）反而低于两未解组（16 / 15.5）——门槛对高维病人不利，是设计特性。
- **跨记录可行性**：两侧各 ≥100 事件、中间静默 ≥24 h 的病人**只有 1 位**（`epilepsiae_1146`）；≥12 h 只有 4 位（2 位有有效偏差）；≥6 h 有 11 位（7 位有效）。→ **"跨天骨架"合同在这批数据上不可行，不要为它写 spec**；唯一还有体量的尺度是 ≥6 h。
- **效力下界**：累加路线可探测下限 δ80 = +0.0069~0.0083，实测中位仅 +0.0041 → **实测约为下限的一半到六成**，`p=0.098` 的正确读法是"这么小的效应本来就抓不住"。单场路线 δ80 = 1.1 个病人间散布 SD。⚠️只约束**队列汇总**那一步，不约束病人内部估计环节。
- **新测量（骨架漂移）**：同等事件间隔下，隔得秒数更多的两块次序更不像（控住块粗糙度+共享触点后中位 −0.0335、22/30 为负、`p=0.017`）；跨录制段也更不像（限定支持度达标后 −0.0137、18/30、`p=0.0086`）。⚠️**这两条不是互相独立的证据**（跨段本来就隔得更久）。
- **昼夜复核已做**（口径取 `AGENTS.md` 锁定值，复用仓库已有 `_classify_day_night` / `epoch_to_local_hour`）：限定同相位后负号保住（−0.0335、24/30、`p=0.0066`）→ 时段解释不了它。**但真正原因是该混淆本来就几乎不存在**——块不跨录制段而 Epilepsiae 录制约 1 h 一段，同段配对最大间隔中位仅 **0.99 h**、跨相位配对占比中位 **0**、只有 7 位 Yuquan 病人有任何跨相位配对；真能检验的只有 6 人（5/6 同向但 `p=0.156`）。
- **⚠️连带把"钟表漂移"收窄了**：该读数实际只覆盖**同一段约 1 小时记录内、典型相隔半小时**的两块（各人中位 1,646–2,099 s，全队列上限 1.98 h）。**不是"骨架在天尺度上漂移"**；超过约 2 小时的尺度本轮没有段内证据。
- **两次同源执行事故已留痕**：`event_source_index` 是逐事件行指针不是段标签（`CLAUDE.md` §6.2 层级错配），误用版本产物整体移入 `_superseded_*` 目录并在新规则文件写明 supersedes；另修复秩基偏相关在控制变量吸干自变量时返回虚假 ±1（残差只剩浮点噪声会相关成 ±1），并单独记录残余变化占比使"没效力"与"效应为零"可区分。
- 仍待签核才可做：真数据注入的效力下界（需定注入语义）、≥6 h 静默的跨段新合同。scoped `pytest` **157 passed**。

### `event_innovation_v3_0_acceptance_2026-08-03.md` — **Level 1 leaky observer：能追踪当前次序状态，但一场事件的"新意"不带未来信息**
- 朴素说法：一个时间步 = 一整场完整间期事件。用过去若干场估出"这个病人目前习惯的触点先后次序"，再问某一场里**没被过去猜到的那部分偏差**能否预告后面几场次序怎么变。两条路线（单场偏差→20 场后次序；连续 20 场偏差累加→次序挪动量），各自对照纯惯性外推、状态配对的替身偏差、往回看同样长的一段。
- 判据在看结果前冻结并锁 SHA：一条路线三个中位数全为正 **且** 主指标病人层双尾 Wilcoxon `p≤0.05`。
- **Goal 2 local**：主指标中位 −0.0007759、7/17、`p=0.329`（`true_minus_matched` +7.39e-6、10/17；`future_minus_past` +1.21e-4、12/17、`p=0.071` — 非主指标，禁单取包装阳性）→ 不成立。
- **Goal 3 cumulative**：三中位数全正（主 +0.004148、11/17；`true_minus_matched` +1.83e-5、9/17；`alignment` +7.36e-4、10/17），但主指标 `p=0.098` 未过 0.05 → 不成立。**且支持度分层反转**：锚点 ≥100 的 9 人中位翻负 −0.000847、4/9、`p=0.910` → 必须读成"没看出预告作用"，**禁读成"差一点就阳性"**。
- **有效 innovation 仅 17/34**（Epilepsiae 15/18、Yuquan 2/16）；另 17 人为"历史长度不够"或"残差仍可被过去预测"，是**未解**不是阴性证据。禁把 17 人结果讲成 34 人队列结论。
- Epilepsiae 子集 cumulative `p=0.035` 是**事后子组**，预注册端点是全部 17 人，仅留档不作主张。dense moving-block 两路线（local −0.000695、`p=0.159`；cumulative +0.004689、`p=0.064`）**只作敏感性**，重叠锚点数不得当独立样本量。
- 最终等级由已验收的 V2.7 state tracking 托底至 **Level 1 `leaky_observer`**。允许："最近发生的完整事件有助于追踪当前 repertoire 状态，但有效 innovation 不提供路线一致的未来传播信息"。禁止：`event-driven transition identified` / `activity-dependent shaping` / `causal plasticity` / `within-event next-rank mechanism`。
- **V3.1 人体 transition 保持 `NOT_TRIGGERED`**，由 validation-only 结果在人体 test 释放前冻结（哈希被 release 记录并每次验收重校验）；本轮任何结果都不能反开，也不允许 model-capacity rescue。SNN 与本线独立，不互为 Gate 或 ground truth。
- 工程：scoped `pytest` 104 passed；冻结主分析三件套 + 验收脚本 + release/handoff 六个 SHA256 逐位一致；`ACCEPTANCE_RULE_STATE.json` 未被验收运行改写。

### `stateful_event_sequence_rnn_v2_7_acceptance_2026-08-03.md` — **repair-only 最终验收：短程 state tracking 保留，EWMA 与 chronology 阴性不变**
- 34/34 validation 与 boundary audit、34×3 formal runs、dense/reset/memory/两类 chronology null 和 H40 全部闭环；102/102 trained checkpoints finite，epoch −1 不再参与 trained patience。
- RNN 相对 static 中位 −0.0619、25/34、`p=0.00385`，但相对 EWMA formal `p=0.076`、dense `p=0.163`；block shuffle、time reversal 与 H40 均未提供额外支持。
- v2.7 与 v2.6 患者级主效应 34/34 逐位相同：修复了训练充分性偏差，但未改变最佳 trained checkpoint。最终只保留 within-recording short-range state tracking，不支持 chronology-specific state 或 shaping。

### `stateful_event_sequence_rnn_v2_6_acceptance_2026-08-02.md` — **验收为 state-tracking precursor：短程 leaky state 成立，state shaping 未检验**
- 34/34 患者完成 validation-only profile 冻结（760 个 candidate fits）与 untouched test（3 seeds、102 个 checkpoint 全 finite）；旧 heldout20、A/B/轴、几何、SOZ、ictal、SNN 均未进入。
- **阳性**：每场事件清零 hidden state 后 25/34 变差（`p=0.0051`），说明模型确实在用历史；RNN 超过静态 repertoire 中位 −0.0619、25/34、`p=0.00385`，且该阳性随支持度分层单调增强（≥50 windows：−0.0826、17/20、`p=0.00029`）。
- **阴性**：没有稳定超过固定 EWMA（formal 中位 −0.0248、`p=0.076`；≥20 windows 分层后翻正为 +0.0378、`p=0.607`；dense +0.0294、`p=0.163`）。block shuffle 与 time reversal 两个 coherent null 均未被真实顺序超过；H=40 未释放额外优势。
- **三条必带边界**：state 每个 source 开头清零 → 「长程」只到单段记录之内（每位患者 source 长度中位数，跨 34 人再取中位 = 294 个事件），跨 source / 跨天过程不在模型族里；block 长度 = horizon = anchor 间距 → 只测块间顺序、未测块内顺序；正式 test 支持度 1–1,096 极不均衡，任何队列陈述必须同时给分层。
- **禁止写**：activity-dependent plasticity / network formation / causal shaping；也不得把 block-shuffle 的反向名义显著（`p=0.035`）读成「真实顺序有害」。
- **阶段判决**：`ACCEPTED_AS_STATE_TRACKING_PRECURSOR_WITH_KNOWN_TRAINING_BIAS`。EWMA 本身是最小 leaky state model，因此“不超过 EWMA”收缩了动力学复杂度，但不终止 evolving-state 问题。
- **后续合同**：v2.7 只修 epoch-minus-one early stopping 并平行重跑，且不作为 v3.0 的科学 Gate；v3.0 使用 rank/precedence primary state，先验证 innovation，再做单事件脉冲和重复事件累积；v3.1 单独承担 shared-filter transition identification，不重复 v2.2 block-delta 分支。
- 派生验收层 `results/topic5_stateful_event_sequence_rnn/v2_6/acceptance/` 由 `scripts/accept_topic5_stateful_event_rnn_v2_6.py` 只读冻结产物推导，含 RNN vs static、支持度分层、双尾 null、seed 离散度与训练预算审计。

### `stable_repertoire_event_rnn_v2_3_1_acceptance_2026-08-01.md` — **完整事件为时间步；长历史分布有信息，线性顺序状态呈患者异质性**
- 恢复 split-half / odd-even 稳定模板作为 RNN 前提；六患者 train-only `K=2` repertoire read-back 6/6，旧 heldout20、A/B/轴/几何/SNN 均未进入。
- 同信息集比较显示：无序 80 场历史在 H=20/H=40 分别 5/6、4/6 优于最近窗口；嵌套线性顺序增量跨两个 horizon 在 3 位患者同向，六患者总体 sign test 不显著。
- 嵌套 GRU 仅 1/6 超过线性 correction；全队列只开放冻结的 static/recent/unordered/nested-linear/null 阶梯，不再扩 GRU。

### `event_indexed_evolving_rank_field_v2_2_review_2026-08-01.md` — **event-indexed 动态可观测，但 event-history 增量 bounded negative**
- 正式把长期时间轴从事件内 rank step 改为完整事件 chronology；pilot 6/6 与全 inventory 34/34 的 time/source/block/rank/tie 字段审计通过。
- 六人 Phase 0 中 5 人 block reliable 且动态超过噪声，严格 full+middle low-rank Gate 后 2 人进入 Phase 1；matched fixed/persistence/drift/switching/time-IEI 比较中 0/2 通过 chronology null。
- 结论只支持多数 pilot 存在 block-wise field variation，不支持 event-history shaping；停止 ELR/RNN，不扩全队列，SNN 独立。

### `stable_interaction_identifiability_v2_1_multiround_2026-07-31.md` — **RNNv2.1 五轮结构审计：single fixed graph 在 4 位可辨识患者中 bounded negative**
- 修正 test endpoint-specific oracle Gate，不增加模型容量，完成 D1 baseline/envelope/diversity、D2 M2 operator、D4 unseen-start、D3 real-minus-null split stability 和 D0 patient-matched sensitivity+specificity。
- D0 为 4/6 PASS；这 4 位 human real-minus-null stability 全为负。另 2 位分别因 sensitivity 与 fixed-vs-mixture specificity 不足保持未裁决。
- Unseen-start NLL 5/6 改善，但 precedence 仅 2/6；不授权 shared-backbone modulation、event drive、process noise 或 full 34-patient 扩展。SNN 不参与 Gate。

### `stable_interaction_graph_rnn_v2_development_2026-07-31.md` — **SIG-RNN v2 development：feedback 有增量；稳定结构尚未裁决**
- 通用 12-contact synthetic feedback graph 在独立 graph/event seeds、9,600 个训练事件和未改阈值下通过工程校准；首轮 2,400-event G0-A 失败仍原样保留，未事后改门。
- 六患者中 SIG1 相对匹配 phase-only noGraph 的 NLL 与自由生成 precedence 均为 6/6 改善；但相对每位患者最强的 phase-matched mixture 或 latent time template，两端点同时改善仅 1/6。
- 旧 `G1` 实际是在 development test 上分别为 NLL 和 precedence 选择最小值的 endpoint-specific oracle stress test；它只能说明 current single fixed graph 未取得已见分布预测优势，不能裁决 stable structure。
- v2.1 已重开 patient-matched identifiability、chronological observable stability 和 unseen-start/compositional generalization；在这些结构特异实验完成前不扩 34 人，也不加 event drive/process noise。SNN 与 RNN 独立，SNN 不在任何 RNN Gate 中。

### `shared_propagation_field_rnn_multiround_review_2026-07-31.md` — **v0.1 七轮复核：输出不反馈的 autonomous trajectory 未被选择**
- 六患者 development 中，M4 相对 M3 在 10%–100% nested learning curve 六档均为 0/6，`d={2,4,6}` 也均为 0/6；该结果只约束由第一 rank 初始化、生成 contact 不反馈 latent state 的 deterministic autonomous latent-trajectory null。
- 既有 SNN Round 5 只作 exploratory compatibility check：legacy artifacts 未满足同条件 nested event-count / `N_min` 合同，且 first-rank lookup 已形成捷径，因此不能据此判 G0 正或负。
- v0.1 按窄合同完成 bounded negative；不解释 latent weight，不否定稳定 contact interaction，也不参与 human-to-SNN mechanism mapping。

### `shared_propagation_field_rnn_ladder_pilot_2026-07-30.md` — **RNNv2 自主 shared-field 六患者公平比较 bounded negative**
- 六名 target-blind development patients × 3 seeds × 8 models 全部训练充分；M4 虽超过 static 6/6、stationary M1 5/6，但相对 phase-matched mixture 与低维时间模板均为 0/6，M4-phase 也只在 2/6 超过模板。
- 旧 train80 内新增独立 development test、checkpoint/provenance、低学习率自动复核、重复 IWAE/prior-predictive 与多次自由 rollout；旧 heldout20 未读取。按 stop rule 不扩到 34 人。
- SNN 仅审计既有 source/sink/paired artifacts，未重跑；方向由低阈值 kernel/core 位置产生，isotropic 不是方向消失 null。该审计后续已从 RNN Gate 中删除，不能评分 G0。

### `rnn_training_and_objective_sufficiency_v0_1_report_2026-07-30.md` — **训练充分性关闭；整场生成阴性拆成"训练不足"+"读出方式"两个成分**
- 1,068 个单元零失败。上一轮冻结训练预算比延长训练预算差 **0.134 nats/decision（34/34 人，P=1.16e-10，LOSO 结构确认）**，约为既有顺序增量（0.0257）的 5 倍；容量、优化器家族、权重衰减、显存分块均已排除为限制因素，学习率仍是限制（3e-4 优于上一轮的 1e-3，位于预注册网格边缘）。跑到预注册上限 8 遍仍未满足连续两遍改善 <0.002 的收敛判据，故只能写"接近收敛"，**不得写"已收敛"**。
- **测试的三种 rollout-aware 目标不支持曝光偏差作为主要解释**（不等于普遍排除）：它们（每 2 步 / 每 3 步自喂、渐增 schedule）在 development 与外层留出上都同时损害局部预测与整场生成，呈单调剂量反应，一步预测护栏全部失守。
- **方法学更正**：上一轮用来评价生成的复合发生器（静态骨架＋顺序残差＋经验终止）**随模型变好而系统性变差**——外层留出上收敛模型经它读出后成对先后相关塌到 0.014（不用历史的静态对照 0.184）。改用模型自身联合分布读出，六个整场端点全部显著改善（30–33/34，两队列同向）；无重训的匹配消融显示这一改善**部分**来自事件内顺序（相对身份打乱对照，五端点显著），但成对先后的绝对水平主要由静态解码结构承担（相对冻结状态仅 +0.034，p=0.121）。**两台发生器是两个不同被估量**：模型自身＝模型生成能力主读出，复合＝顺序残差分解的敏感性。
- **预注册整场门槛仍未达到**：用上一轮自己的标准重评分，上一轮 9/34（本轮精确复现）、静态对照 10/34、延长训练+模型自身读出 13/34、rollout-aware+模型自身读出 14/34，门槛 17/34。**不得写"RNN 自由生成了真实的完整双向传播事件"。**

### `rnn_stage_acceptance_and_training_sufficiency_2026-07-30.md` — **当前 RNN 阶段性总入口：科学验收通过，训练充分性仍开放**
- 已接受对象为稳定 static contact scaffold + 短程 within-event ordered information；linear-state 可改善 heldout next-contact，并在自由生成中恢复局部 transition fingerprint。
- 完整 suffix rank/precedence 和双向 axis read-back 未恢复，但该阴性目前只限 frozen teacher-forced training contract。正式模型仅一轮 exact coverage，最终 linear-state 未独立调 learning rate / training budget，也未用 self-fed rollout loss。
- 下一轮只允许做 convergence 与 objective-sufficiency 审计；执行 prompt 见 `docs/superpowers/plans/2026-07-30-topic5-rnn-training-sufficiency-agent-prompt.md`。

### `minimal_sequence_kernel_closeout_v0_2_report_2026-07-30.md` — **where / how / when 分层后的最小序列结构最终验收**
- 34 人 × 3 seeds 的同分母重评分将 heldout likelihood 精确拆为 contact choice 与 STOP：contact identity 的增量集中在当前和前一 rank，第三 rank 主要改善 STOP；更早历史无额外 contact 信息。
- 可识别对象改为 linear-state 的输入—输出 lag kernel \(K_k=CA^kB\)。显式 FIR-H3 未优于无序基线，固定方向跨数据集确认失败；patient-mean early-ictal association 仍主要来自 static scaffold。
- `when` 已隔离为新分支：exact 1–150 Hz seizure residual 可靠性当前不可辨识，IEI-aware Gate 1 仅有未跨两数据集复现的 cohort feasibility signal。本轮定位固定为 Extended Data / Supplementary bounded result。

### `ordered_history_architecture_audit_v0_1_report_2026-07-29.md` — **最新 RNN 条件信息、架构与跨状态综合验收**
- 34 人 × 3 seeds 的 target-blind 架构审计显示：linear-state 相对 unordered-prefix 的患者中位 NLL 增益为 0.0257（26/34，7-family maxT P=0.00032），相对同架构 rank-shuffle 为 0.0419（31/34）；容量匹配后 linear 结果保留。
- 但 7 个预注册递归家族仅 linear-state 通过 family-wise inference，故顺序证据具有架构依赖性；clinical-onset `[0,10] s`、`1–150 Hz` reused target 上，ordered residual 超越 static + unordered 与 matched shuffle 的条件增量均未建立。当前只进入 supplementary sequence-identification / boundary result，不支持脑流形、真实时间慢变量或逐发作预测。

### `rnn_overall_integrated_acceptance_2026-07-28.md` — **上一版 RNN 总验收基线（已由 2026-07-29 架构审计细化）**
- 统一收口 full-rank、low-rank、persistent path、symmetric-axis、competitive/source、internal-state、fixed early-ictal readout 与 H1/H2/H3 history-necessity 全部分支。
- 最终接受对象为 target-blind 的稳定 interictal participation scaffold + 最近 2–3 个 rank set 的有序短历史；early-ictal 只保留 reused-target 的 sign-free static morphology。full history、正 low-rank mode、path/axis/competition/source 和 GRU-specific static transfer 均未建立。
- 机器可读状态：`results/topic5_rnn_overall_acceptance/FINAL_ACCEPTANCE.json`；论文层级固定为 `SUPPLEMENTARY_BOUNDED_COMPUTATIONAL_RESULT`。

### `interictal_scaffold_reliability_history_necessity_v0_1_report_2026-07-28.md` — **34人 target-blind 静态可靠性与 H1/H2/H3 历史必要性**
- train80–heldout20 participation field Spearman 中位 0.893，34/34 为正；约 200 个事件时 30/34 已接近 full train80 estimate。
- H2>H1、H3>H2、ordered H3>matched H3 shuffle 均通过；full history 不超过 H3。accepted sequence reference 因此锁为 H3，不再扩大 GRU 历史。

### `static_scaffold_fixed_readout_validation_v0_1_report_2026-07-28.md` — **静态 contact topography 分项验收：跨状态形态保留，GRU-specific 增量未建立**
- strict clinical-onset 16 人/106 seizures 的 participation readout、signed primary、强空间 null、target-free regularized baseline、teacher/free 分解和 baseline-power confound audit 已全部完成。
- orientation-free contact morphology 在 within-shaft 与 geometry-smooth null 下保留；真实顺序相对 rank-shuffle 有 heldout 增益，但 positive signed direction、unbounded-history necessity 和 GRU-specific static increment 均未建立。等待真正独立 clinical-onset patient cohort 复制。

### `static_contact_topography_claim_consistency_audit_2026-07-28.md` — **当前论文口径全文一致性审计**
- 扫描 21 个 manuscript-facing 文件，20 处敏感表述全部归入明确边界/否定、其他经验合同或历史模型阶段，`UNSAFE_CURRENT_CLAIM=0`。
- 当前唯一 Figure 6 source 为 `docs/paper-draft/figure6_static_contact_topography_bounded_result.md`；早期 RNN 版本只保留为 provenance。

### `rnn_postreview_closeout_and_static_scaffold_goal_2026-07-28.md` — **RNN 审阅后收口与 fixed signed scaffold 新 goal**
- 上一 goal 经复核拆成 static contact topography、interictal order sensitivity 和 target-reused state read-back 三层；Figure 降为补充探索性候选，structured-axis RNN 冻结。
- 新 fixed participation Phase 1 显示 absolute morphology margin 稳定，但 positive signed margin 在 all-contact、within-shaft 与 geometry-smooth null 下均有明显患者异质性；已启动正则化非递归 baseline、teacher-forced/free-rollout 和 confound 分解。

### `rnn_internal_state_reduction_v0_1_report_2026-07-28.md` — **静态 scaffold、ordered-history 诊断与探索性 state read-back 分层**
- 冻结 34 人 × 3 seeds 既有 GRU，102/102 hidden extraction、扰动和随机子空间单元全部完成；真实 prefix 顺序打乱对 ordered GRU 的 NLL 影响显著大于 rank-shuffle，对应 32/34 患者为正。
- strict clinical-onset 16 人/106 seizures 中，固定 participation 支持静态 contact scaffold，但 full GRU 未稳定超过 static/unordered/rank-shuffle；去 participation 后的 PC1/PC2 迁移仅作 target-reused exploratory candidate。下一步先做 fixed signed readout、强空间 null 和正则化非递归基线。

### `symmetric_axis_competitive_propagation_rnn_v2_3_result_2026-07-27.md` — **categorical RNN 可预测，但 physical-axis 机制门失败**
- 22 人 × 3 seeds × 5 conditions 的 330 个正式模型全部完成；full 相对 node bias 22/22 为正，history 相对 instantaneous 18/22 为正。
- delayed competition、matched physical axis 与 source-conditioned direction 均未过冻结门；模型约恢复 ordered-history Markov cohort-median benefit 的 58%，不开放 latent-state 解释或 early-ictal transfer。

### `interictal_transition_signal_decomposition_v0_1_result_2026-07-27.md` — **Markov transition signal 分解与 v2.3 开发许可**
- 31 人显示 symmetric、跨局部几何且依赖 ordered multi-step history 的 heldout transition signal；22 人 physical-axis residual通过，但 source-conditioned增益很小且 14/22 axis coefficient 为负。
- 冻结决策为允许起草最小 v2.3 recurrent observation model，不是 shared anatomical axis 或 early-ictal transfer 的机制结论。

### `symmetric_axis_propagation_state_v2_2_1_closeout_2026-07-27.md` — **symmetric-axis propagation-state RNN 按预注册停止点收口**
- 66/66 formal runs 审计完整；Claim 1/2 为已执行失败，Claim 3/4 为 `LOCKED_NOT_RUN`，early-ictal transfer 同时受间期 gate 与 0 exact source metadata 阻断。
- 在同一 22 人 heldout 合同下，Markov 21/22 优于 node-bias，而 full/isotropic 均仅 1/22 为正；校准显示 next-set size 1.00 被预测为约 1.65，local/axis kernel 中位 Frobenius cosine 0.979。结论只限“当前非负线性单状态 observation mapping 不足”，不否定共享病理轴。

### `persistent_path_mode_rnn_closeout_and_v2_pivot_2026-07-26.md` — **Figure 6 旧 RNN 收口与 v2.2 propagation-state 转向**
- 冻结 v0.7/v0.9/v1.0：局部历史可学，但离散、event-persistent path mode 不是合适科学对象；不再调 K、hidden size 或开放发作期 target。
- 下一版改为共同的近似对称 effective scaffold + observed source + 单一 propagation state + scalar STOP；三位 geometry-complete development 后以 22 人做 physical-axis formal，跨状态主任务回到 clinical-onset early-ictal energy field。

### `persistent_path_mode_rnn_formal_result_2026-07-26.md` — **Figure 6 正式 34 人 structured graph RNN bounded-negative**
- 34 人 × 3 seeds × 5 conditions 的 510 个 LOSO runs 全部完成；局部 next-set NLL 可学，但 participation 与完整 rank distribution 的自由生成主门未通过。
- graph / mode-collapse 结构必要性未成立，path-direction posterior 保持高熵；按预注册合同不读取 clinical-onset 发作期 target。给出可保留口径、禁止口径和若重开时必须修改的训练目标。

### `fig6_interictal_operator_phase0_stagea_pilot_2026-07-24.md` — **Figure 6 计算桥：selected h64 的 Stage-A engineering screen 停止于 suffix-static gate**
- 以 masked contact-rank 的单事件 prefix→suffix/STOP 任务训练 contact-query GRU；40/40 患者、532,793 个间期事件通过数据与泄漏审计，Stage A 不读取 ictal values。
- 13 折 target-free one-SE 选择 h64；13 人 one-seed screen 中 next-set 对 strongest static 的患者级 CI 为正，但 suffix 中位为负且 CI 跨 0；两项均 13/13 超过 rank-shuffle。按 stop rule 不启动正式三 seed gate、Mode recovery 或 ictal readout。

### `fig3_ictal_gradient_r3_full_recompute_handoff_2026-07-18.md` — **当前执行合同：Figure 3 发作相关 gradient R3 全量重算**
- 以正式 n=17 / 167 seizures 为唯一母清单，统一使用 outcome-independent adaptive 81×81 gradient grid、subject-fixed sigma、corrected mirror abs-max、shared-else-own maxAB 与 coherent all-contact null；R2 只作同输入 paired sensitivity。
- Stage 1 重算 cohort Data-vs-Null 与七频带 inheritance/specificity；Stage 2 更新 Fig3-B R3 score provenance，并在 Fig3-C 仍保留时同步重算 7 名 shared-only 轨迹和 spatial null。配套回填表 `fig3_ictal_gradient_r3_full_recompute_run_form_2026-07-18.md`。

### `field_concordance_multiband_unified_handoff_2026-07-18.md` — **SUPERSEDED：旧 R2 七频带合同**
- 该版本虽锁定 n=17 / 167 parent cohort、共同 permutation 与 subject-first fold，但 primary metric 仍是 contact-evaluated R2；只保留 provenance，不再执行。旧 form 同样作废。

### `fig3a_raw_spectral_context_acceptance_2026-07-18.md` — **Fig3-A 正式画图合同与验收**
- 锁定 E1146 seizure 7 的 raw SEEG + SCL9 TFR + 四频带 2×2 布局、严格时间轴对齐、row-shared y 轴、clinical-onset shading 和可/不可报告边界。
- canonical producer：`scripts/paper_figures/plot_fig3_raw_spectral_context.py`；输出：`results/paper-ready-figure/fig3a_raw_spectral_context/figures/`。

### `axis_alignment_AB_result_2026-06-14.md` — **现阶段主线结果**：间期传播轴 ↔ 发作早期激活的轴对齐（A 线 primary + B 线 secondary）
- 18 Epilepsiae 队列：粗"共享网络主轴"稳（broadband 稳赢全通道 null，FDR + LOSO 扛住）；细对齐仅快活动（hfa）稳（过最严 joint）；符号自由共线，非逐点重放。
- 含完整方法 / 定稿数值表 / 工件清单 / handoff。计划全貌：`network_axis_pivot_plan_2026-06-13.md`（A/B 段已标 ✅ 执行）。
- 定稿表 `results/topic5_ictal_recruitment/axis_alignment/axis_alignment_FINAL.md`。

### `hfa_joint_confirm_2026-06-15.md` — hfa×joint 冻结复验（split-half + 负对照）
- 唯一过最严 joint null 的 hfa 细对齐：full 干净复现（Wilcox=0.022）但**奇数半不显著（0.078）→ 非 split-half 稳健**；负对照四层全部非显著=非假阳性。
- 结论 = real-but-not-robust，**维持灵敏度档、不升 primary**；升格须独立第二队列。主线粗骨架不受影响。

### `v3p_preictal_nonaxis_trajectory_2026-07-05.md` — V3p preictal-only 非轴向轨迹完整硬门阴性
- 只看 EEG onset 前 −120~−10 s；narrow、`broad_expanded`、`broad_core` 分层报告，三层均 tier 0，完整个体支持为 0。
- broad 的少数 single-null nominal hits 被 rate / lag1 / phase / block / 双 span 预设硬门筛掉，不算潜在阳性。
- 结论边界：未支持稳定一致的 preictal non-axis ramp；不等于发作前没有任何 state change，也不裁决 onset 后变化。

### `contact_similarity_ladder_2026-07-01.md` — 触点相似性几何阶梯（R1 无几何 / R2 同平面触点核 / R3 场）
- n=18（两激活量），场统计量数值抬高主要来自平面几何平滑；但平滑同时抬高信号与零假设，超零假设被试数反而随 R1→R3 下降。网格步无可分辨增益；R3 与 A 线主统计逐位一致。
- R2b native-3D sensitivity 与 2D plane 等价通过。定位是灵敏度/稳健性复核，不是新的队列级主张；主线粗骨架结论不受影响。

## PR 系列

### `pr1_seizure_clustering/` — Per-subject seizure subtyping (z-ER tensor + 1−Spearman + UPGMA)
- `pr1_zer_cohort_2026-05-10.md` — **主结果文档**：cohort z-ER subtyping，含 sentinel 视觉裁定、audit fix 历史、over_split 规则演化
- 见 `results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/figures/README.md` 的 cohort 视觉骨架
- 计划档：`docs/superpowers/specs/topic5_pr1_seizure_clustering.md`（plan v2）

### PR-0：v2.3 Layer A ictal ER timing atlas
（追授 topic5 PR-0；详细 spec 见 `docs/superpowers/specs/`）

### Bridge → Ictal-template echo 谱系（Topic 1 × Topic 5：间期传播模板是否在发作期复演）
- `bridge_q1/bridge_q1_results_2026-05-10.md` — Q1 cohort（verdict NULL-locked, n=9, power floor）
- `bridge_q1prime/bridge_q1prime_results_2026-05-10.md` + `q1prime_overnight_exploration_2026-05-10.md` — Q1′ case-series（INDETERMINATE）
- `echo_gate/stage1_proxy_triage_2026-06-08.md` — **Stage 1** ER 代理 echo gate：= 共享粗锚，非 specific-path-replay
- `dynamic_echo/stage2b_sentinel_2026-06-12.md` — **Stage 2b** early-ictal 动态模板 echo sentinel：**gate NOT PASSED**（B=500 n=3）；有模板相关结构但非稳定早期路径复演 → 粗解剖/杆级锚为主；未进 cohort。（Stage 2 first-onset recruitment 量错失败，未单独归档，见此文档"谱系"段）
### `constructive_event_generation_sufficiency_v0_1_report_2026-07-30.md` — **局部 transition 可生成，但完整双向事件充分性 Gate 失败**
- 34 人 × 3 seeds 的 102 个 source-conditioned free-running 单元全部完成；history 改善 first-order transition fingerprint，但不改善 suffix rank/precedence，只有 9/34 人至少两项达到人体 split-half 经验范围。
- 独立 rank-progress STOP 在 34/34 人必要；22 位 train-only 双模态+物理轴合格患者中，history 未改善 template 或 signed-axis fidelity。Gate C 与 SNN bridge 按合同锁定。
# Continuous marked-state R1.2b 收口与下一阶段

- `continuous_marked_state_r1_2b_closeout_2026-08-25.md`：冻结 R1.2b 为有限
  target-alignment 诊断，规定 persistent-vs-memoryless、完整 raw R1.3、精确
  H2a 与长尺度最小 T2-S1 的执行边界。
- `continuous_marked_state_r1_3_preregistered_contract_2026-08-25.md`：正式固定
  三人×三 seed 的 paired explicit/full-raw 训练、inner selection、H2a 端点与
  梯度/更新量验收口径；smoke 不进入正式聚合。
- `continuous_marked_state_t2_s1_long_scale_human_contract_2026-08-25.md`：把 H3
  的 `N=100` 降为短尺度参照，预注册 `N=1000` 为当前两位可测患者的主探索；
  `N=10000` 只在张家齐等高事件量患者具备同合同 T1 后执行，历史严格不跨记录缺口。
- `continuous_marked_state_t2_long_total_effect_contract_2026-08-26.md`：把 H3 从
  下一事件 residual edge 扩展为长窗 total-effect 候选；固定张家齐 `N=10000`
  与约 6 小时两尺度、occurrence+load 双输入、no-edge 与因果延迟 1000 次
  counterfactual，并以冻结 T1 decoder 空间为主评分。
- `continuous_marked_state_long_t1_triage_contract_2026-08-26.md`：在三位事前固定
  长记录患者上完成 target-trained R1.3 三 seed 分诊，并把 H3 独立支持改为
  real window 与 causal-delayed 额外 1,000 events 的联合区间。9/9 T1 完成，
  但无人同时满足 2/3 persistent T1 与 TRAIN/validation 各至少 3 个完整不重叠窗；
  因此新人体 H3 按合同 0 作业，判为不可检验而非生物学阴性。权威报告位于
  `results/epi_prssm/continuous_marked_state/r1/r1_3_long_triage_goal_report/`。

### 2026-08-26 近期 goals 综合复审

- 白话版：`results/epi_prssm/continuous_marked_state/r1/final_reports/recent_goals_integrated_review_plain_2026-08-26.md`
- 技术版：`results/epi_prssm/continuous_marked_state/r1/final_reports/recent_goals_integrated_review_technical_2026-08-26.md`
- 两版统一审阅 2026-08-21 至 2026-08-26 的 Raw-SEEG、R1.2/R1.2b/R1.3 与短至超长尺度 H3，并已吸收第二轮代码复审：H2b 当前重算为 +0.4582 SD、21/27、p=0.0059、339 次可评分发作，但仪器修复后仍需重跑；六小时 boxcar 为 7/7 拟合发散、不可估计；H3 资格按真实记录覆盖段而不是 `event_session` 计算。两版明确区分 accepted development evidence、结构零、发散/不可测人体结果与尚未打开的正式检验分区。
- 第二轮更正记录仅作审计链，正文不再依赖读者另行覆盖：`recent_goals_integrated_review_post_review_corrections_2026-08-26.md`
  - 机器重算：`results/epi_prssm/continuous_marked_state/r1/final_reports/recent_goals_post_review_audit.json`
- 前一轮长尺度更正记录：`continuous_marked_state_t2_long_total_post_review_corrections_2026-08-26.md`
- 最终下一版合同：`continuous_marked_state_r1_4_t2_r2_0_contract_2026-08-27.md`
  - 六患者 R1.4 复现 H1/H2a；H3 主实验回到 H3-S0 支持最稳定的 N=100 一步 generator-edge；N=1,000–2,000 与六小时 boxcar 退出当前主线。
- R1.4/T2-R2.0 完成后的下一阶段合同：`continuous_marked_state_r1_5_h3_long_contract_2026-08-27.md`
  - 以三位 exact-model 未见患者加三位旧长记录校准患者做 5-seed R1.5；H3 另行探索 N=1,000/3,000/10,000 的 TRAIN-only event innovation。N=100 阴性不 gate 长尺度探索，但没有稳定 T1 时只允许报告长历史 antecedent association。
- R1.5/H3-long 完成报告：
  - 白话版：`continuous_marked_state_r1_5_h3_long_plain_2026-08-27.md`
  - 技术版：`continuous_marked_state_r1_5_h3_long_technical_2026-08-27.md`
  - 机器审计快照：`continuous_marked_state_r1_5_h3_long_machine_audit_2026-08-27.json`
  - R1.5 在三位新增患者中仅张克轩达到稳定、正确时刻特异的候选状态标准，增量落在 first subset 而非 continuation。H3-long 的 16 个完整对照与 10 个边界患者-source-尺度组合均无患者级支持；唯一稳定 T1 患者的独立 validation 支持不足，因此结论是未支持且仍未决，不是长尺度效应的生物学阴性。
- R1.6 优化器与可识别性诊断：
  - 合同：`continuous_marked_state_optimizer_identifiability_r1_6_contract_2026-08-27.md`
  - 白话版：`continuous_marked_state_optimizer_identifiability_r1_6_plain_2026-08-27.md`
  - 技术版：`continuous_marked_state_optimizer_identifiability_r1_6_technical_2026-08-27.md`
  - 旧结论更正边界：`continuous_marked_state_optimizer_identifiability_r1_6_correction_boundary_2026-08-27.md`
  - 机器审计：`continuous_marked_state_optimizer_identifiability_r1_6_machine_audit_2026-08-27.json`
  - 推荐配置：`continuous_marked_state_optimizer_identifiability_r1_6_recommended_config_2026-08-27.json`
  - 修复 R1.5 epoch 0 已见过 inner-validation 的不公平选择，并确认旧 H3 `ZERO_GRADIENT` 来自零 state readout。synthetic 与六患者短段过拟合证明模型可训练；公共配置五 seed 确认仅 E384 达到稳健 persistent+correct-time development 支持。E384 的最小 H3 6/6 未过完整控制且只有 2 个独立 validation 单元，因此 H3 仍未决，不作生物学阴性。
- R1.7A / T2-R2.0 前瞻性 development 复现：
  - 冻结合同：`continuous_marked_state_r1_7a_r2_0_contract_2026-08-27.md`
  - R1.5/H3-long 退役说明：`continuous_marked_state_r1_5_retirement_2026-08-27.md`
  - 正式验收 R1.6 的优化器/可识别性诊断与 E384 单患者支持；R1.5 的选择偏差结果以及 H3-long 的 N=1,000/3,000/10,000、六小时 boxcar 均退出当前证据主线。
  - 在不读取模型结果的前提下，从未参与旧决策的 development 患者中按记录与事件支持各取 Epilepsiae/Yuquan 5 人；development validation 按真实记录时长冻结为 D_state 60% 与 D_mechanism 40%。先做五 seed H1/H2a，仅对合格患者在 D_mechanism 运行无自由 exposure 截距的 N=100 T2。
  - **执行结果（2026-08-27 收口）**：
    - 白话版：`continuous_marked_state_r1_7a_r2_0_plain_2026-08-27.md`
    - 技术版：`continuous_marked_state_r1_7a_r2_0_technical_2026-08-27.md`
    - 机器审计：`results/epi_prssm/continuous_marked_state/r1/r1_7a/reports/machine_audit.json`
    - 50/50 R1.7A cells、38/38 T2 cells 全部完成；R1 与 T2 各自 source payload 唯一；
      formal/sealed/seizure/paper-ready 全程关闭；无 N≥1000、六小时、物理时间产物。
    - **H1/H2a**：按事前判据（≥3/5 seeds 同时通过 persistent 与 correct-time）**4/10 患者复现**
      （上一轮 1/6）；但按 TRAIN 决定长度的时间块 bootstrap 区间，只有 `epilepsiae_1125`
      在充足独立块（37/35）上两层都不跨零；`epilepsiae_253` 仅 correct-time 成立、
      `yuquan_liyouran` 仅 persistence 成立、`yuquan_zhangbichen` 两层成立但独立块仅 5 个。
      反例：`epilepsiae_1073` 两个差值恒为 0（状态完全未参与）；
      `yuquan_zhaochenxi` persistent 显著劣于 memoryless（时刻编码但无跨窗口留存）。
    - **H2a 端点**：增益集中在 selecting group size 与 later continuation；
      `first subset` 方向不一致（最强患者上区间显著不利）。
      合同三个主端点中 `same_prefix_continuation` 在本队列与 `continuation` **逐位等值**，
      实际只有两个独立端点。
    - **H3（T2-R2.0）**：8 个 patient×source 单元中冻结聚合器判 2 个为 support，但复核后
      `yuquan_zhangbichen`/load 由 **placebo−no_edge=+0.0286 的安慰剂退化**驱动
      （real−no_edge 仅 −0.00088、种子符号不一致、独立块 8 个），不予采信；
      `epilepsiae_1125`/participation 内部一致但**缺独立时间块确认**（事件平均基于 2904 行、
      独立块仅 33 个，产物中无逐块 contrast）。**8/8 未达上一轮自身证据标准，H3 仍未决。**
    - **三项方法学发现**：(1) T2 无逐块 contrast；(2) `real_edge_estimable` 把拟合结果
      `edge_left_zero_initialisation` 混入可估计性闸门，导致"主动判定无边"的 seed 被移出分母，
      **低报阴性**；(3) `patient_source_support` 不要求 real 相对 no_edge 有实质量级，
      退化的 placebo 即可让判据通过。
    - **运行事件**：5 cells 非有限梯度（`epilepsiae_1077` 4、`yuquan_zhaochenxi` 1）按 R1.6
      先例显式记录为仪器失败并保留在分母；`r1_7_t2.py:89` 漏传 `state_permutation` 致该 T2
      路径此前从未跑通，已修并加 AST 静态回归测试；`yuquan_zhangbichen`（52 contacts）
      3 cells OOM，按合同以降低并发处理、未改 batch/chunk 故数值不变。测试 126 passed。
