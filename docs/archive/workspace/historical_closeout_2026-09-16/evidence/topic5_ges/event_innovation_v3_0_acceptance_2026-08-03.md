# Topic 5 V3.0 跨事件 innovation — 最终验收报告（2026-08-03）

判决：**Level 1 — leaky observer**（`evidence_level.json` 状态 `EVIDENCE_LEVEL_FROZEN`）。
V3.1 人体 transition 分支保持 `NOT_TRIGGERED`，这是看结果**之前**就定下的，不因本轮任何数字改动。

---

## 0. 朴素话摘要（不看代号也能复述）

**我们看的是什么。** 同一个病人在一段脑电记录里，一场接一场地发生间期放电事件。每一场事件里，参与放电的触点有个先后次序 —— 谁先响、谁后响。把过去若干场的次序攒起来，就能估出"这个病人目前习惯的先后次序"。我们要问的是：某一场事件里**没能被过去猜到的那一部分次序偏差**，对**后面几场**的次序怎么变，有没有预告作用。

注意时间步的定义：一个时间步 = 一整场完整事件。我们不做"同一场事件内部下一个触点是谁"这种预测。

**我们怎么测的。** 走了两条路线。第一条：拿单独一场事件的那点偏差，看能不能让"20 场之后的次序"比"完全按过去惯性往前推"猜得更准。第二条：把连续 20 场事件的偏差累加起来，看累加得越多、方向越一致，后面次序挪动得是不是越大。

每条路线都要跟三种对照比：(a) 什么都不用，纯按惯性外推；(b) 把真实偏差换成"状态相近但不是这一场"的替身偏差；(c) 把"往后看 20 场"换成"往回看同样长的一段"。

判据在看到任何结果之前就写死并存了哈希：一条路线上三个指标的中位数**全部**为正，**并且**该路线主指标在病人层面的双尾符号秩检验 p ≤ 0.05，才算"偏差能预告未来"。

**我们看到了什么。** 两条路线都没跨过这条线。

单场路线的主指标中位数是负的（17 人中只有 7 人为正，p=0.33）——单场偏差没让后面的次序更好猜。

累加路线三个中位数确实都是正的，但主指标 p=0.098，没到 0.05。更要紧的是我们额外查的一件事：这个正号是被"锚点少"的那半批病人撑起来的。只看锚点 ≥ 100 的那 9 位，中位数翻成负的、9 人里只有 4 人为正、p=0.91。所以**不能把"三个中位数都为正"读成"差一点就阳性"**——数据量越足的病人，方向反而反过来。

同时，上一轮（V2.7）已经单独确认过一件相邻但不同的事：**最近发生了哪些事件，确实能帮着追踪"当前的次序状态"**。所以整体停在"会漏的观察者"这一档：模型能追踪状态，但一场事件带来的"新意"不额外携带未来信息。

**这一轮没有回答的问题。** 有效偏差只在 34 人中的 17 人身上成立（Yuquan 只有 2 人）；另外 17 人是"历史长度不够"或"残差仍能被过去预测"，那是**没看清**，不是"看清了没有"。

（内部归档代号：V3.0 Goal 2 local impulse response / Goal 3 cumulative accumulation, `propagation_gain_standardized`, `true_minus_state_matched_null_gain`, `future_minus_past_state_gain`, `dose_alignment.alignment_coefficient`, `INNOVATION_VALID`, `UNRESOLVED_INSUFFICIENT_HISTORY`, `UNRESOLVED_STATE_RESIDUAL`, evidence Level 1 `leaky_observer`, V2.7 `ACCEPTED_REPAIR_ONLY_STATE_TRACKING_FINAL`）

---

## 1. V2.7 是已完成的前置，不是未验收挂账

`results/topic5_stateful_event_sequence_rnn/v2_7/acceptance/ACCEPTANCE_STATE.json` 判决
`ACCEPTED_REPAIR_ONLY_STATE_TRACKING_FINAL`，34 人 × 3 seed、dense、state reset、memory curve、
block shuffle、time reversal、H40 全部闭环。

该验收允许的措辞是 event-history state tracking、与 static / 固定 EWMA 观察者的比较、
within-recording memory 与 chronology 对照；禁止 event-driven network shaping、
evolving-graph identification、causal plasticity、within-event next-rank mechanism。

V2.7 在 V3.0 中的角色有两条，不能混：

- 它**不是** V3.0 人体 test 的释放条件（spec §14 明写 "V2.7 completion is not on this checklist"，
  `HUMAN_TEST_RELEASE_STATE.json` 亦记录 `v2_7_completion_is_release_condition=false`）；
- 它**是** evidence ladder 的兜底档来源：当两条路线都不满足 Level 2 时，
  已验收的 state tracking 把最终等级托在 Level 1 而不是 Level 0。

## 2. 有效 innovation 的人数与不合格原因

`innovation_validation_only/innovation_validity.json`：34 人请求、34 人完成、0 失败。

| 状态 | 人数 | 含义 |
| --- | --- | --- |
| `INNOVATION_VALID` | 17 | 残差通过 blocked cross-fitting 后仍不可被过去预测，可进入两条 test 路线 |
| `UNRESOLVED_INSUFFICIENT_HISTORY` | 9 | 缺少所需的 pre-window 历史长度 |
| `UNRESOLVED_STATE_RESIDUAL` | 8 | 残差仍能被过去预测，未分离出独立 innovation |

数据集拆分：**Epilepsiae 15/18 有效，Yuquan 仅 2/16 有效**（`yuquan_zhangjiaqi`、`yuquan_zhangkexuan`）。

两类 `UNRESOLVED` 按 spec §15 是**显式的未解状态，既不是病人失败，也不算 innovation 通过**。
任何队列陈述必须同时给出"17/34"这个分母；不得把 17 人的结果讲成 34 人队列结论。

## 3. 两条 primary 路线的连续效应（非重叠锚点，每人一个效应）

两条路线均 `HUMAN_TEST_ROUTE_COMPLETE`、`n_completed=34`、`n_failed=0`、`n_eligible=17`、
`test_dependent_selection=false`、`within_event_next_rank_model_fit=false`。

### 3.1 Goal 2 local（`local/LOCAL_TEST_STATE.json`）

| 指标 | 中位数 | bootstrap 95% CI | favorable | Wilcoxon 双尾 | sign test 双尾 |
| --- | --- | --- | --- | --- | --- |
| `propagation_gain`（主） | −0.0007759218 | [−0.0029691577, +0.0011044352] | 7/17 | 0.328949 | 0.629059 |
| `true_minus_matched` | +0.0000073893 | [−0.0000722488, +0.0002648225] | 10/17 | 0.430679 | 0.629059 |
| `future_minus_past` | +0.0001214024 | [−0.0000097166, +0.0003262951] | 12/17 | 0.071411 | 0.143463 |

主指标中位数为负 → 该路线在冻结规则下不成立。
`future_minus_past` 方向为正但 p=0.071，且它**不是**该路线主指标；
单取这一项包装成阳性是被 handoff 明确禁止的操作。

数据集方向（记录用，非分层主张）：Epilepsiae n=15 主指标 −0.0007759218（6/15，p=0.421）；
Yuquan n=2 主指标 −0.0044540427（1/2）。

### 3.2 Goal 3 cumulative（`cumulative/CUMULATIVE_TEST_STATE.json`）

| 指标 | 中位数 | bootstrap 95% CI | favorable | Wilcoxon 双尾 | sign test 双尾 |
| --- | --- | --- | --- | --- | --- |
| `cumulative_gain`（主） | +0.0041483373 | [−0.0020690926, +0.0083331617] | 11/17 | 0.098373 | 0.332306 |
| `true_minus_matched` | +0.0000182563 | [−0.0003044796, +0.0004408064] | 9/17 | 0.889969 | 1.000000 |
| `alignment` | +0.0007358903 | [−0.0040382006, +0.0094913599] | 10/17 | 0.430679 | 0.629059 |

三个中位数**全为正**，满足 Level 2 规则的前半段；但主指标 p=0.098373 **未过** 0.05，
故规则后半段不满足，Level 2 不成立。三个指标的 bootstrap CI 均跨 0。

数据集方向（记录用，非分层主张）：Epilepsiae n=15 主指标 +0.0047465871（10/15，p=0.035339）；
Yuquan n=2 主指标 −0.0048313716（1/2）。
**Epilepsiae 子集那个 p=0.035 不是预注册端点**——冻结判据用的是全部 17 人；
按数据集拆开后再挑显著的一半，属于事后子组选择，不得作为主张，仅按 spec §11
"dataset-specific directions are reported without result-defined subtypes" 留档。

### 3.3 支持度分层（事后描述，不改判决，方向是收紧不是放松）

cumulative 主指标的正中位数由低支持度病人承载。按每人 test 锚点数分层：

| 分层 | n | 中位数 | favorable | Wilcoxon 双尾 |
| --- | --- | --- | --- | --- |
| 全部 eligible | 17 | +0.004148 | 11/17 | 0.098 |
| 锚点 ≥ 50 | 16 | +0.004447 | 11/16 | 0.074 |
| 锚点 ≥ 100 | 9 | **−0.000847** | 4/9 | 0.910 |
| 锚点 ≥ 250 | 6 | +0.001651 | 3/6 | 0.844 |

锚点分布：17 人 min=37、median=152、max=1115（两条路线锚点数基本相同）。
数据量最足的一半病人方向**反转**。因此对 3.2 的正确读法是"没看出预告作用"，
**不是**"差一点就阳性"；这一条与 V2.6/V2.7 验收里"支持度分层可以翻转符号"的已知教训一致。

此分层是事后描述性检查，不在冻结判据内，也**无法**提高等级（Level 1 已由冻结规则锁定）；
它只能收紧结论，故如实登记。local 主指标在同样分层下始终为负，无反转。

## 4. dense moving-block sensitivity（只作敏感性，不替代 primary）

独立 runner（`scripts/run_topic5_event_innovation_v3_0_dense_bootstrap.py` +
`src/topic5_event_innovation_bootstrap_v3_0.py`），**未改动**冻结主 runner。
两条路线均 34/34，状态 `DENSE_BOOTSTRAP_ROUTE_COMPLETE`、`sensitivity_only=true`。

| 路线 | 中位数 | bootstrap 95% CI | favorable | Wilcoxon 双尾 | sign test 双尾 |
| --- | --- | --- | --- | --- | --- |
| local dense | −0.0006950206 | [−0.0014816020, +0.0009903123] | 6/17 | 0.159378 | 0.332306 |
| cumulative dense | +0.0046887388 | [−0.0008479218, +0.0079809581] | 11/17 | 0.063828 | 0.332306 |

按 spec §10，dense 锚点重叠，其数量**不得**当作独立样本量。
这两条只是"换一种取锚方式，结论方向是否翻转"的稳健性检查，方向与 primary 一致（同样未过线），
既不替代非重叠 primary，也不构成反开 V3.1 的依据。

## 5. 最终 evidence level 与措辞边界

冻结判据（`ACCEPTANCE_RULE_STATE.json`，状态 `PRE_AGGREGATE_ACCEPTANCE_RULE_FROZEN`，
在任一路线汇总产生**之前**写入，并锁定验收脚本 SHA256 `3fa4ffa2…`）：

> Level 2 requires all three frozen cohort medians to be positive in either Goal 2 or Goal 3
> and the route's primary gain to have a patient-level two-sided Wilcoxon p-value <= 0.05;
> otherwise accepted V2.7 state tracking gives Level 1, else Level 0.

逐条比对：

- Goal 2：三中位数并非全正（主指标为负）→ 不成立；
- Goal 3：三中位数全正，但主指标 p=0.098373 > 0.05 → 不成立；
- V2.7 判决为 `ACCEPTED_REPAIR_ONLY_STATE_TRACKING_FINAL` → 落 **Level 1**。

`evidence_level.json`：`level=1`、`level_name=leaky_observer`、`level2_supporting_routes=[]`。

**允许的措辞**（冻结原文）：

> Recent complete events help track the current repertoire state, but valid event innovations
> add no route-consistent future propagation information.

**禁止的措辞**（冻结原文）：`event-driven transition identified`、`activity-dependent shaping`、
`causal plasticity`、`within-event next-rank mechanism`。

另外三条本报告自加的边界，来自本轮实际数据：

- 不得把 Goal 3 的"三中位数全正"写成"接近阳性 / 趋势性阳性"——高支持度分层方向反转（§3.3）；
- 不得把 Epilepsiae 子集 p=0.035 提为结论——那是事后子组，预注册端点是全部 17 人（§3.2）；
- 不得把 17 人结果讲成 34 人队列结论——另外 17 人是未解状态，不是阴性证据（§2）。

## 6. V3.1 人体 transition 保持 `NOT_TRIGGERED`（预 test 决定）

`V3_1_HANDOFF_STATE.json`：`status=NOT_TRIGGERED`、`v3_1_human_execution_allowed=false`、
`human_test_outcomes_read=false`、`capacity_rescue_allowed_if_closed=false`。

该状态由 **validation-only** 的 Goal 2 / Goal 3 结果决定（两者均 `NOT_OPEN`），
在人体 test 释放**之前**就已冻结，其 SHA256 `ecf0f295…` 被 `HUMAN_TEST_RELEASE_STATE.json`
的 `inputs_sha256.handoff` 记录，验收脚本每次运行都会重新校验这一致性。

验收层显式记录 `human_test_cannot_reopen_v3_1=true`。本轮人体 test 的任何结果
（包括 Goal 3 三个正中位数）都**不能**把它改成 `OPEN`，也不允许任何 model-capacity rescue。
若将来要做 V3.1 人体 transition，那是一份新合同，不是本轮的事后敏感性或补救。

## 7. SNN 与这条 RNN 线相互独立

按 spec §13：SNN **不是**本线的 Gate、目标、先验、标签来源或 ground truth。
两边各自冻结之后，其可观测的传播原则**可以**在讨论层做一次收敛性对照，
但不得把任一方的结论当作另一方的支持证据。

`config/topic5_event_innovation_v3_0.yaml` 的 `forbidden_inputs` 在实现层锁死了
old heldout20、A/B 或轴标签、几何或 SOZ、ictal 或 SNN 四类输入，
`innovation_validity.json` 记录 `forbidden_inputs_read=false`。

## 8. 冻结完整性与工程验收

人体 test 释放后未被改动（本轮逐个重算 SHA256，与 release 记录逐位一致）：

| 对象 | SHA256 | 校验 |
| --- | --- | --- |
| `scripts/run_topic5_event_innovation_v3_0_human_test.py` | `75eb2d44…` | 与 release `implementation_sha256.human_runner` 一致 |
| `src/topic5_event_innovation_test_v3_0.py` | `ebce595e…` | 与 release `implementation_sha256.test_helpers` 一致 |
| `config/topic5_event_innovation_v3_0.yaml` | `50bc1c4d…` | 与 release `config_sha256` 一致 |
| `scripts/accept_topic5_event_innovation_v3_0.py` | `3fa4ffa2…` | 与规则文件 `runner_sha256` 一致（验收脚本自校验） |
| `HUMAN_TEST_RELEASE_STATE.json` | `665ac88f…` | 与规则文件 `release_state_sha256` 一致 |
| `V3_1_HANDOFF_STATE.json` | `ecf0f295…` | 与 release `inputs_sha256.handoff` 一致 |

验收脚本运行**未**改写 `ACCEPTANCE_RULE_STATE.json`（mtime 保持在汇总之前，
`route_aggregate_outcomes_read` 仍为 `false`），符合"规则先于结果"的设计。

scoped 测试：`pytest -q` 覆盖 25 个 `topic5_event_innovation` / `topic5_stateful_event_rnn_v2_7`
测试文件，**104 passed**（此前 101 + dense bootstrap 新增 3）。

`git diff --check`：handoff 指定的五文件命令返回干净，但需注明这五个文件当前在 git 中是
**未跟踪**状态（`??`），未跟踪文件不进 diff，该命令对它们是空跑。
另用 `git diff --check --no-index /dev/null <file>` 逐个实查，五个文件空白字符均干净。
仓库其余部分存在用户既有改动（含一个 matplotlib 生成的 SVG 有 trailing whitespace），
按 handoff 要求未做任何清理。

## 9. 产物清单

```
results/topic5_event_innovation_impulse_response/v3_0/
├── HUMAN_TEST_RELEASE_STATE.json          HUMAN_TEST_RELEASED
├── V3_1_HANDOFF_STATE.json                NOT_TRIGGERED（预 test 冻结）
├── innovation_validation_only/
│   └── innovation_validity.json           17 valid / 9 无历史 / 8 残差未分离
└── human_exploratory/
    ├── ACCEPTANCE_RULE_STATE.json         PRE_AGGREGATE_ACCEPTANCE_RULE_FROZEN
    ├── HUMAN_EXPLORATORY_STATE.json       HUMAN_EXPLORATORY_COMPLETE
    ├── cohort_inference.json              两路线 cohort + dataset-specific
    ├── evidence_level.json                level=1, leaky_observer
    ├── patient_summary.csv                34 行 × 14 列
    ├── local/LOCAL_TEST_STATE.json        HUMAN_TEST_ROUTE_COMPLETE, 34/34
    ├── cumulative/CUMULATIVE_TEST_STATE.json  HUMAN_TEST_ROUTE_COMPLETE, 34/34
    └── dense_bootstrap/{local,cumulative}/*_DENSE_BOOTSTRAP_STATE.json  sensitivity_only
```

## 10. 本轮闭环范围

- 科学对象自始至终是"一整场完整事件 = 一个时间步"，两条路线与验收层均记录
  `one_step_is_one_complete_event=true`、`within_event_next_rank_model_fit=false`；
- 未做 architecture sweep、未增加模型容量、未从 latent 权重解释触点图；
- 未把预测关联写成 activity-dependent shaping 或 causal plasticity；
- 冻结 spec / plan / release 哈希对象正文未回改，执行结果只记录在本报告。

上游合同：`docs/superpowers/specs/2026-08-03-topic5-event-innovation-low-rank-state-space-v3_0.md`
（§7 Goal 2、§8 Goal 3、§9 V3.1 边界、§12 evidence ladder、§15/§16 冻结修订与 test 语义）。
执行 handoff：`event_innovation_v3_0_execution_handoff_2026-08-03.md`。
前置验收：`stateful_event_sequence_rnn_v2_7_acceptance_2026-08-03.md`。
