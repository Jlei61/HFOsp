# Epi-PRSSM v0.1 探索性实施计划

**状态：** `GO_FOR_EXPLORATORY_IMPLEMENTATION`

**日期：** 2026-08-18；根据科学审阅修订

**科学 spec：** [`2026-08-18-topic5-epi-prssm-v0_1.md`](../specs/2026-08-18-topic5-epi-prssm-v0_1.md)

**图形合同：** [`2026-08-18-topic5-epi-prssm-figure-contract.md`](../specs/2026-08-18-topic5-epi-prssm-figure-contract.md)

**自主执行 prompt：** [`2026-08-18-topic5-epi-prssm-autonomous-agent-prompt.md`](2026-08-18-topic5-epi-prssm-autonomous-agent-prompt.md)

## 0. 当前判断

### 能不能开始

可以。现在可以同时启动：

- 数据合同和现有 baseline 复核；
- G0–G3 generator 广度筛选；
- H2a event adapter 与 ambiguous-prefix inventory；
- 分问题的 just-in-time synthetic；
- H3/R0–R3 的 interictal development。

不需要等一个庞大的 synthetic 总网格全部通过，也不需要等 H3 成立后才运行 H2。

### 当前安全主张

现有工作支持稳定 patient-specific repertoire 和有限的短程 ordered-history 信息，但尚未建立分钟至小时 autonomous graph recurrent slow state，也未建立 IED exposure 驱动该状态。Epi-PRSSM 是新探索，不是旧阴性结果的改名。

### 最大科学缺口

需要把以下替代解释拆开：

1. 只有 patient baseline；
2. 只有 leaky recency / observer tracking；
3. 有真实 graph recurrent slow state；
4. 慢状态调制当前事件；
5. 慢状态连接间期与发作；
6. IED exposure 反向更新慢状态。

### 本轮执行原则

科学阴性不是项目 blocker。除三个硬门外，实验继续运行并单独输出 evidence card，让阴性结果定位失败来自 generator、readout、seizure link 还是 exposure mechanism。

## 1. 冻结的科学问题与工作流

| Goal | 科学问题 | 是否依赖 H3 | 主输出 |
| --- | --- | --- | --- |
| Goal 0 | 数据对象、patient baseline 和现有 baseline 是什么 | 否 | inventory、parity、variance decomposition |
| Goal 1 / H1 | 慢状态是否存在；需要 G0/G1/G2/G3 哪一层 | 否 | filtered/open-loop model ladder |
| Goal 2 / H2a | 慢状态是否改变事件分布 | 否 | full-event、state-swap、ambiguous suffix |
| Goal 3 / H2b | 慢状态是否连接间期与发作 | 否 | preictal/pseudo-onset/early-ictal |
| Goal 4 / H3a-b | IED exposure 是否更新状态并可能参与转换 | 独立嵌套 | R0–R3、innovation、directionality |
| Goal 5 | learned event encoder 是否增加信息 | 否 | representation sensitivity |

运行依赖是：

```text
Goal 0 ──> Goal 1 ──> Goal 3 (先过 interictal freeze)
   │          ├────> Goal 2
   │          └────> Goal 4 interictal
   └───────────────> Goal 2 inventory / synthetic

Goal 4 H3b 只读取冻结后的 Goal 3 endpoint；
Goal 4 不阻塞 Goal 1–3。
```

## 2. 运行规模与资源策略

为兼顾“多做实验”和避免无意义全因子爆炸，分四级运行：

| 级别 | 分母 | seeds | 目的 |
| --- | --- | --- | --- |
| smoke | 每个数据源 1 名、支持完整 | 1 | shape、mask、积分、日志和显存 |
| breadth pilot | 6–8 名 support-stratified development patients | 3 | 广泛比较 G0–G3、adapter、R0–R3 |
| development cohort | 全部 eligible train/validation patients | 3 | 选择小型候选族和主要 endpoint |
| formal confirmation | 全部 eligible untouched test | 至少 5 | 正式 evidence card；每个 test 只释放一次 |

development patients 只能按 train-only event count、contact count、source continuity、dataset 和 ambiguous-prefix support 分层选择，不能按 H1/H2/H3 effect 选择。

广度筛选保留所有结果，不只保存赢家。每个被淘汰模型也写明：数值失败、容量无增量、open-loop 崩溃、observer 吞噬或资源塌缩。

## 3. 输出根与不可覆盖边界

代码根：

```text
src/topic5_epi_prssm/
```

脚本根：

```text
scripts/topic5_epi_prssm/
```

结果根：

```text
results/epi_prssm/v0_1/
```

建议目录：

```text
results/epi_prssm/v0_1/
├── manifests/
├── data_audit/
├── baseline/
├── synthetic/
│   ├── generator/
│   ├── event_distribution/
│   ├── seizure_link/
│   └── exposure/
├── generator_ladder/
├── event_distribution/
├── seizure_link/
├── exposure_mechanism/
├── learned_encoder/
└── figures/
```

本计划不得覆盖现有 `results/paper-ready-figure/fig1`–`fig4`。任何未来 paper slot 必须先更新 `docs/paper_figure_registry.md`。

## 4. Goal 0：冻结数据对象与现有 baseline

这一步不设科学性能 gate；目的是让后续每个阴性和阳性都可解释。

### Task 0.1：建立 immutable run manifest

新增：

- `src/topic5_epi_prssm/contracts.py`
- `src/topic5_epi_prssm/manifests.py`
- `scripts/topic5_epi_prssm/build_data_manifest.py`

manifest 至少记录：

- dataset、patient、source、session、recording block；
- raw/artifact hash、code revision、config hash；
- contact order、channel mapping、coordinate/scaffold provenance；
- event timestamp、real \(\Delta t\)、gap type；
- train/validation/test chronology；
- seizure labels 是否仍为 `sealed`；
- forbidden-field schema。

输出：

```text
results/epi_prssm/v0_1/manifests/DATA_MANIFEST.json
results/epi_prssm/v0_1/manifests/SPLIT_MANIFEST.json
results/epi_prssm/v0_1/manifests/FORBIDDEN_INPUT_AUDIT.json
```

### Task 0.2：reconcile 可复用 v4.0 组件

逐项标记 `reuse / adapt / reject`：

- session join 和真实 \(\Delta t\)；
- window/repertoire/scale 工具；
- 既有 event descriptor；
- observer/checkpoint 逻辑；
- 旧 result roots 与未跟踪模块。

不能因为函数存在就视为科学验收。输出：

```text
results/epi_prssm/v0_1/data_audit/V4_RECONCILIATION.md
```

### Task 0.3：冻结 event representation

实现并测试：

- tied rank 使用显式 group identity；
- non-participation 为 mask；
- raw lagPat channel order 与 canonical order 对齐；
- phantom rank fail closed；
- node marks 为 participation、normalized rank、onset indicator；
- primary load 为 participating-contact fraction。

单元测试文件：

```text
tests/topic5_epi_prssm/test_event_marks.py
tests/topic5_epi_prssm/test_channel_order.py
tests/topic5_epi_prssm/test_forbidden_inputs.py
```

### Task 0.4：估计 patient baseline repertoire

新增 `patient_baseline.py`，只用 train events 估计 \(\boldsymbol\mu_p\)。输出：

- static participation/rank/order/STOP baseline；
- train–validation stability；
- patient baseline 与 within-patient residual 的方差分解；
- patient/site ID 预测审计。

主表：

```text
results/epi_prssm/v0_1/baseline/patient_repertoire_variance.csv
results/epi_prssm/v0_1/baseline/patient_baseline_summary.json
```

### Task 0.5：建立支持度 inventory

统计每位患者：

- event/source/session 数；
- IEI 和 clock-time coverage；
- H5/H10/H20/H40 可用 anchors；
- ambiguous-prefix 候选数、每个 prefix 的 suffix 支持；
- seizure 数、last-IED-to-onset gap，仅在 Gate B 后释放给 Goal 3。

输出：

```text
results/epi_prssm/v0_1/data_audit/support_inventory.csv
results/epi_prssm/v0_1/data_audit/ambiguous_prefix_inventory_train_only.csv
```

### Hard Gate A：数据与泄漏完整性

只检查数据能否被科学解释：source/session 可追溯、channel mapping 正确、mask/ties 正确、forbidden inputs fail closed、chronological split 无泄漏。

单患者失败只排除该患者并记录原因；只有系统性 schema 错误才暂停相关数据源。baseline 性能差不属于 Gate A。

## 5. Goal 1 / H1：慢状态是否存在，需要哪一层 generator

### Task 1.0：just-in-time generator synthetic

只实现四类 truth：

1. no-state；
2. leaky state；
3. graph recurrent state；
4. observer-overpowering。

每类至少包含 irregular IEI、missing source 和不同 contact count。此 synthetic 只校准：

- no-state false positive；
- G0 与 graph recurrence 的可区分性；
- observer correction 是否吞掉 generator；
- open-loop 实现是否偷读 future marks。

失败时修对应模块或降低解释，不阻止 Goal 2 inventory 和 Goal 3 protocol 准备。

### Task 1.1：实现共同 generator 接口

新增：

```text
src/topic5_epi_prssm/graph_cells.py
src/topic5_epi_prssm/resource_dynamics.py
src/topic5_epi_prssm/observer.py
src/topic5_epi_prssm/rollout.py
```

统一接口：

```python
state_minus = generator.propagate(state_plus, delta_t, graph)
state_est = observer.correct_graph_state(state_minus, event_marks)
pred = decoder.predict(prefix, patient_baseline, state_minus)
```

primary observer API 中不得出现 `correct_resource_every_event()`。

### Task 1.2：实现 G0–G3

| 模型 | 实现 | 关键约束 |
| --- | --- | --- |
| G0 | CT leaky/EWMA | 明确标为 baseline |
| G1 | stable graph-CLDS | 正阻尼或谱稳定参数化；无自由 \(N\times N\) 权重 |
| G2 | graph-GRU-ODE | node-level shared cell、有界门控、真实 \(\Delta t\) |
| G3 | G2/最佳 recurrent + autonomous resource | resource 只进 damping/gain/readiness |

同时实现：

- node-level primary；
- spectral compressed sensitivity；
- flexible observer-resource correction control；
- unconstrained persistent GRU baseline。

### Task 1.3：强制工程测试

至少覆盖：

1. G0 不调用 message passing；
2. G1/G2 message 只沿 graph support；
3. 不同 patient node count 可训练；
4. state gauge 固定；
5. resource 在 \((0,1)\)；
6. primary observer 不逐事件改 resource；
7. correction 与 physical transition 分日志；
8. TBPTT 不重置 forward state；
9. correction-off 后不读 future marks；
10. 数值积分在最大真实 gap 下稳定。

### Task 1.4：breadth pilot

在 6–8 名 support-stratified development patients 上运行：

```text
static
event-index EWMA
CT-EWMA / G0
unconstrained persistent GRU
G1 graph-CLDS
G2 graph-GRU-ODE
G3 graph recurrent + autonomous resource
G3-flexible-resource-correction control
```

每个 3 seeds，统一 state dimension budget、observer budget 和 event descriptor。

记录而不提前淘汰：

- filtered H1 loss；
- correction-off H5/H10/H20/H40；
- state reset curve；
- \(\Delta t\) shuffle；
- correction energy；
- time constants、stability margin；
- patient baseline residual variance explained；
- wall time、显存和 numerical failures。

### Task 1.5：development cohort 扩展

将所有稳定运行的模型扩到全部 eligible development patients。宽筛阶段 3 seeds；每个结构层最多保留一个代表进入 5-seed confirmation，使用 one-standard-error 和稳定性共同选择，不只看最低平均 loss。

输出：

```text
results/epi_prssm/v0_1/generator_ladder/model_runs.csv
results/epi_prssm/v0_1/generator_ladder/patient_effects.csv
results/epi_prssm/v0_1/generator_ladder/open_loop_horizon.csv
results/epi_prssm/v0_1/generator_ladder/state_reset.csv
results/epi_prssm/v0_1/generator_ladder/GENERATOR_EVIDENCE_CARD.json
```

### H1 自动解释分支

- 只有 G0 有效：`leaky observer/state tracking`；继续 Goal 2–4，以定位其它信号。
- G1 超过 G0：`structured graph recurrent state`。
- G2 超过 G1：`nonlinear graph recurrent increment`。
- G3 超过 G2：`bounded resource anchor adds predictive value`。
- open-loop 阴性：不称 autonomous predictor，但仍继续 filtered/state-conditioned、preictal 和 exposure 诊断。

这里没有停止门。

## 6. Goal 2 / H2a：慢状态是否改变事件分布

Goal 2 的 inventory、synthetic 和 baseline caching 与 Goal 1 并行；正式 adapter 比较读取 Goal 1 的预定义 model representatives，不要求 H1 全部阳性。

### Task 2.0：just-in-time readout synthetic

只做：

- state-conditioned ambiguous suffix truth；
- no-state false adapter；
- correct-state vs matched-state swap；
- patient-baseline-only truth。

### Task 2.1：冻结 base contact RNN

复现现有 next-contact、STOP、free generation 和 TA/TB downstream readout。复现差异写 parity report；若只是旧模型性能阴性，保留为 baseline。只有 channel/event mapping 不一致才回到 Gate A。

输出：

```text
results/epi_prssm/v0_1/baseline/CONTACT_RNN_PARITY.md
```

### Task 2.2：adapter 容量阶梯

在同一 base RNN 上依次比较：

1. no state；
2. initial-state adapter；
3. Node FiLM；
4. restricted low-rank graph edge gate。

不做四者全组合。每种 adapter 都运行 static state、G0、G1、G2、G3 代表，以回答增量来自 state 还是 adapter 容量。

### Task 2.3：全队列 full-event endpoints

Primary：

- masked rank/order distribution；
- STOP/continuation；
- participation-residualized repertoire；
- correct-state vs patient-internal matched-state swap。

Secondary：participation、extent、平均 next-contact NLL。

### Task 2.4：ambiguous-prefix targeted analysis

只纳入 train inventory 支持充分的 prefix family。对每个 eligible patient 报告：

- prefix support；
- suffix entropy；
- state-conditioned suffix NLL；
- correct-state vs matched-state swap；
- state-stratified suffix distribution。

不 eligible 的患者保留在 full-event analysis，不算阴性。

### Task 2.5：冻结后 TA/TB 投影

TA/TB 只读冻结 state 和 event predictions，作为解释性 downstream panel；不得用于 adapter、state dimension 或 checkpoint 选择。

输出：

```text
results/epi_prssm/v0_1/event_distribution/adapter_ladder.csv
results/epi_prssm/v0_1/event_distribution/full_event_effects.csv
results/epi_prssm/v0_1/event_distribution/state_swap_effects.csv
results/epi_prssm/v0_1/event_distribution/ambiguous_prefix_effects.csv
results/epi_prssm/v0_1/event_distribution/H2A_EVIDENCE_CARD.json
```

H2a 阴性不阻止 Goal 3 或 Goal 4。

## 7. Goal 3 / H2b：慢状态是否连接间期与发作

### Task 3.0：just-in-time seizure-link synthetic

只做：

- latent preictal drift；
- event-rate-only drift；
- last-observation gap；
- pseudo-onset matched null。

目标是验证统计管线，不要求 synthetic 成为人体分析的总前置门。

### Hard Gate B：冻结 interictal family 后释放 seizure labels

冻结文件：

```text
results/epi_prssm/v0_1/manifests/INTERICTAL_MODEL_FREEZE.json
```

至少记录：

- G0–G3 中计划进入 H2b 的代表及 checkpoints；
- patient baseline、observer、adapter、state dimension；
- normalization、open-loop anchor 和 last-observation 规则；
- preictal windows、pseudo-onset matching、nuisance set；
- primary/secondary endpoints；
- planned patient-level contrasts。

允许冻结多个结构层代表，不允许读 seizure outcome 后回选模型或 checkpoint。

### Task 3.1：seizure-aligned state trajectory

对每场 seizure：

1. 找最后一场允许进入 observer 的间期事件；
2. 关闭 observer；
3. 用真实 elapsed time 将 generator 自主积分到 onset；
4. 保存 graph-state summary、resource、uncertainty 和 last-observation gap；
5. 与同 patient/source/time-of-day/rate/IEI 匹配的 pseudo-onset 比较。

### Task 3.2：leave-seizure-out 与 nuisance 增量

比较：

- state-only；
- rate/IEI/source/time-of-day；
- 可用 sleep/vigilance；
- recent-event descriptor；
- fixed patient repertoire；
- nuisance + frozen state。

主报告为 patient effect distribution、time-in-warning 和 leave-seizure-out；不把 pooled AUC 当唯一结果。

### Task 3.3：early-ictal transfer

使用 onset 前冻结 state 预测：

- early-ictal masked order/rank；
- activation field；
- extent；
- frozen TA/TB projection。

不得用 early-ictal outcome 更新 interictal state 或 checkpoint。

输出：

```text
results/epi_prssm/v0_1/seizure_link/seizure_aligned_states.csv
results/epi_prssm/v0_1/seizure_link/pseudo_onset_effects.csv
results/epi_prssm/v0_1/seizure_link/warning_metrics.csv
results/epi_prssm/v0_1/seizure_link/early_ictal_transfer.csv
results/epi_prssm/v0_1/seizure_link/H2B_EVIDENCE_CARD.json
```

Goal 3 即使阴性，也继续 Goal 4 H3a；只关闭 H3b 的 transition interpretation。

## 8. Goal 4 / H3：resource 与 IED exposure 是否更新状态

Goal 4 在纯 interictal数据上可与 Goal 2 并行启动，不等待 H2b。H3b 一致性只在 Goal 3 冻结结果完成后追加。

### Task 4.0：just-in-time exposure synthetic

分批而不是一次大网格：

1. T1/R1 autonomous resource；
2. R2 impulse depletion；
3. R3 integrated exposure；
4. hidden common cause；
5. event-count-only；
6. switching state；
7. observer-resource substitution。

每个 truth 只比较与它相邻的模型，避免全模型×全 truth 笛卡尔积。

### Task 4.1：R0/R1 估计并冻结 \(\tau_r\)

在不含 exposure forcing 的 T1 上比较 no-resource 与 autonomous resource。使用 train/validation 冻结：

- \(\tau_r\) 或可辨识区间；
- latent-activity consumption \(\gamma_q\)；
- generator 中 resource 调制 damping/gain 的接口；
- flexible observer-resource correction control 的预算。

若 primary resource 无增量但 flexible control 有效，记录为“额外 latent coordinate”，仍允许运行 R2/R3 sensitivity，但不作资源机制主张。

### Task 4.2：R2 single-event depletion

在完全匹配的 T1/R1 上只增加 event impulse 路径。报告：

- immediate state/readout response；
- recovery curve；
- non-load endpoints；
- event-load shuffle；
- open-loop H5/H10/H20/H40。

R2 阴性不阻止 R3。

### Task 4.3：R3 integrated exposure

固定 \(\tau_r\) 后，breadth pilot 先跑 metadata 支持的 fast/medium/slow 三档 clock kernel；每档 3 seeds。

shortlist 后再扩展：

- clock sensitivity：5/15/30/60/120 min；
- event-count sensitivity：5/10/20/40/80 events；
- expected-load deterministic rollout；
- stochastic load rollout。

不以单个最佳 \(\tau\) 的 nominal P 定义机制；报告 effect-vs-timescale curve 和可辨识区间。

### Task 4.4：H3a interictal predictive leg

T1/R1、R2、R3 使用同 graph、observer、decoder、adapter、state dimension、split、seed 和优化预算。primary outcome 至少包括：

- masked order/rank；
- suffix branch 或 state-swap residual；
- participation-residualized repertoire。

participation/extent 只作 secondary，避免 load-target 同义反复。

### Task 4.5：H3a innovation/directionality leg

用冻结 T1 state 构建 blocked cross-fitted expected load：

```text
expected load <- T1 state + IEI + local rate + source + time of day
innovation    <- observed load - expected load
```

比较：

- cumulative innovation；
- state-matched innovation shuffle；
- time reversal；
- event-count control；
- source-coherent block shuffle。

不得用 T2 state 生成 expected load。

### Task 4.6：H3b transition consistency

只有 Goal 3 已冻结 endpoint 后才运行，不回选 T2 或 \(\tau_x\)。检查：

- interictal H3a effect；
- preictal state effect；
- early-ictal transfer；
- patient-level direction consistency。

H3a 可以在 H2b 阴性时独立报告；H3b 需要 H3a、H2b 和方向一致。它不是全项目联合 gate。

输出：

```text
results/epi_prssm/v0_1/exposure_mechanism/resource_ladder.csv
results/epi_prssm/v0_1/exposure_mechanism/exposure_timescale_curve.csv
results/epi_prssm/v0_1/exposure_mechanism/t1_t2_patient_effects.csv
results/epi_prssm/v0_1/exposure_mechanism/innovation_controls.csv
results/epi_prssm/v0_1/exposure_mechanism/H3A_EVIDENCE_CARD.json
results/epi_prssm/v0_1/exposure_mechanism/H3B_EVIDENCE_CARD.json
```

## 9. Goal 5：learned event encoder

这不是科学 gate，只是后续表示学习扩展。完成显式 marks 的 broad model ladder 后即可按资源优先级启动：

1. explicit marks；
2. frozen event-internal encoder；
3. frozen encoder + state model；
4. 小学习率 joint fine-tuning；
5. raw waveform encoder 最后。

每次升级重新检查：

- patient/site ID 是否进入 latent；
- state gauge 是否保持；
- resource 是否塌缩；
- observer 是否吞掉 generator；
- open-loop 是否真实改善。

## 10. 三个硬门与非 blocker 清单

### Hard Gate A

数据、mapping、mask、chronology 或 forbidden-input integrity 失败。修复前相关 run 无资格解释。

### Hard Gate B

读取 seizure labels 前没有冻结 interictal model family/checkpoint/endpoints。未冻结则不能作 H2b/H3b 正式分析。

### Hard Gate C

主要模型和 endpoint 确定后，正式 claim 必须来自 untouched test。test 被用于调参后，该轮自动降为 exploratory，但不禁止继续实验。

### 明确不是 blocker

- baseline parity 的某个性能值未复现，但数据 mapping 正确；
- synthetic 的某个 truth 不可辨识；
- G1/G2/G3 没有超过 G0；
- H20 阴性但 H5/H10 有信号；
- ambiguous-prefix 分母不足；
- H2a、H2b、H3a 或 H3b 任一阴性；
- 单个 control/null 阴性；
- resource 参数塌缩。

这些情况都要继续相关诊断或其它独立 Goal，并把 claim 自动降级。

## 11. Hard Gate C 的正式释放流程

在 formal test 前生成：

```text
results/epi_prssm/v0_1/manifests/FORMAL_TEST_RELEASE.json
```

必须列出：

- 每个 Goal 的 model representatives；
- primary/secondary endpoints；
- patient eligibility；
- seeds；
- planned contrasts 与 multiplicity family；
- figure panel contract；
- 允许的 claim 分支。

正式 test 释放后只允许：

- 修复明确工程 bug，并用新版本和新 test 标识重跑；
- 运行预先声明的 sensitivity；
- 生成不改变 estimand 的图。

不允许根据 test outcome 回选 \(\tau\)、checkpoint、adapter 或 patient subgroup 后仍称 confirmatory。

## 12. 图形生产顺序

按独立科学问题生产，不等 H3 决定整套图：

1. **Figure A：** 数据对象、三类状态、G0–G3 与 R0–R3 架构；工程实现后即可出 methods candidate。
2. **Figure B：** H1 generator ladder；development 完成即可出 exploratory，formal test 后替换 cohort statistics。
3. **Figure C：** H2a event distribution；full-event、state-swap 和 support-rich ambiguous suffix。
4. **Figure D：** H2b interictal-to-ictal link；只在 Gate B 后生成。
5. **Figure E：** H3 exposure mechanism；作为独立扩展，阴性也可画成 model discrimination/falsification，不影响 A–D。

每个 figure package 必须同一次运行生成 PNG、PDF、metadata JSON 和 `figures/README.md`。完整视觉合同见 figure contract。

## 13. 监控、失败恢复与运行纪律

- broad pilot 每个模型先 smoke，再并行到 patients/seeds；
- 单个 run numerical failure 不重启整批，记录 failure reason 后重跑该单元；
- OOM 优先 patient-wise scan、gradient accumulation 和 checkpointing，不改变科学 batch object；
- 每个阶段保存 completed/failed/pending denominators；
- 不覆盖 active worker 输出；每个 run 使用唯一 run ID；
- 每完成一个 breadth cohort，立即生成汇总 CSV/JSON 和诊断图，不等全部 Goals 结束；
- 图实际生成后再写 README，不提前放空模板。

## 14. 最终验收产物

```text
results/epi_prssm/v0_1/
├── manifests/
│   ├── DATA_MANIFEST.json
│   ├── SPLIT_MANIFEST.json
│   ├── FORBIDDEN_INPUT_AUDIT.json
│   ├── INTERICTAL_MODEL_FREEZE.json
│   └── FORMAL_TEST_RELEASE.json
├── data_audit/
│   ├── V4_RECONCILIATION.md
│   ├── support_inventory.csv
│   └── ambiguous_prefix_inventory_train_only.csv
├── baseline/
│   ├── patient_repertoire_variance.csv
│   └── CONTACT_RNN_PARITY.md
├── generator_ladder/
│   ├── model_runs.csv
│   ├── open_loop_horizon.csv
│   └── GENERATOR_EVIDENCE_CARD.json
├── event_distribution/
│   ├── full_event_effects.csv
│   ├── state_swap_effects.csv
│   ├── ambiguous_prefix_effects.csv
│   └── H2A_EVIDENCE_CARD.json
├── seizure_link/
│   ├── pseudo_onset_effects.csv
│   ├── early_ictal_transfer.csv
│   └── H2B_EVIDENCE_CARD.json
├── exposure_mechanism/
│   ├── exposure_timescale_curve.csv
│   ├── innovation_controls.csv
│   ├── H3A_EVIDENCE_CARD.json
│   └── H3B_EVIDENCE_CARD.json
└── figures/
    ├── README.md
    └── <asset_id>/
        ├── figures/
        │   ├── README.md
        │   ├── <asset_id>.png
        │   └── <asset_id>.pdf
        └── <asset_id>_metadata.json
```

## 15. 第一批可立即执行的任务

按以下顺序开始，但允许并行：

1. 建 `contracts.py`、manifest 和 forbidden-input tests。
2. 生成 patient/source/event/time-scale/ambiguous-prefix inventory。
3. 建 train-only patient baseline \(\boldsymbol\mu_p\) 和 variance decomposition。
4. 实现 node-level G0/G1 及共同 observer/rollout API；先完成 generator synthetic。
5. 并行缓存 contact-RNN baseline 和 H2a adapter synthetic。
6. 扩展 G2/G3，跑 6–8 人 × 3 seeds breadth pilot。
7. 不等 G3 成为赢家，直接让 G0–G3 代表进入 H2a adapter ladder。
8. 在纯 interictal数据上冻结 R1 的 \(\tau_r\)，依次跑 R2、R3；H3a 独立输出。
9. 写 `INTERICTAL_MODEL_FREEZE.json` 后才释放 seizure labels，运行 H2b。
10. 最后把冻结 H3a 与 H2b 只读组合成 H3b evidence card，不用 H3b 决定其它 Goals 是否成立。

这条路线的目标是广泛、可解释地定位数据支持哪一层模型，而不是把所有实验压缩成一次超级验收。
