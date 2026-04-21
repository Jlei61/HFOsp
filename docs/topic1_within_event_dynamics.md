# Topic 1：间期事件内部时序结构

> 状态：当前正式入口
> 范围：只讨论单个间期群体事件内部的时序组织，包括传播刻板性与事件级同步性。

---

## 1. 这个 topic 只回答什么问题

本 topic 只回答两个问题：

1. 单个群体事件内部，不同通道的激活顺序是否稳定、是否刻板、是否存在多种主要传播模式。
2. 单个事件内部的同步性指标在发作前后是否表现出系统性变化。

它**不**回答：

- 事件与事件之间的 IEI / PSD / rate modulation：那是 `docs/topic2_between_event_dynamics.md`
- 慢调制发生在 SOZ 还是 non-SOZ：那是 `docs/topic3_spatial_soz_modulation.md`

---

## 2. 一句话当前结论

- **传播刻板性**：内部传播真实存在但不是单一模板；`k=2` 是主导压缩，少数 subject 有 `k=4`/`k=6` 多模态；模板在 split-half / blockwise 上 `23/30 strong` + `7/30 moderate` + `0 weak`。Identity-bias 在簇内高达 86%，必须并列报告 raw 与 centered。
- **慢调制（PR-4B）**：模板混合（L1）与模板内顺序一致性（L2）cohort 全 null；模板内相对时延结构（L3）在全 cohort 上证据不足，仅在 8 个高置信子集（`dom_r > 0.7`）的 Pearson r 上探索性显著（p=0.016, 7/8）。详见 `docs/archive/topic1/interictal_group_event_internal_propagation.md`。
- **发作邻近（PR-4C）**：propagation pattern 五指标 cohort Wilcoxon 在主+辅两配置下均 null（主 1/15 / 辅 1/15 名义显著且跨配置不一致）→ **模板内部几何无稳健发作邻近调制，正式封板为阴性**。唯一稳健信号在 `rate_by_template`（post_ictal vs baseline 主 p=0.0009、辅 p=0.0067）。详见 `docs/archive/topic1/pr4c_seizure_proximity_review_2026-04-17.md` §9。
- **模板招募（PR-5）— 核心科学结论（已验收 2026-04-20）**：在 PR-5-A novel-template gate 已 PASS（main n=23 / aux n=22，未观察到 `H_OOD` 或 `H_assignment_drift` 的 cohort-level 证据）的前提下，PR-5-B 把 PR-4C `rate_by_template` 的描述层信号正式升级为推断结论：**dominant template 的绝对事件率（events/h）在 post-ictal 相对 baseline 出现 cohort-level 系统升高**（候选 A `dominant_global` `post_minus_baseline` median main `+65.46` / aux `+42.43` events/h；main p=0.00128 Bonferroni-pass α=0.0083，aux p=0.0115 nominal-pass，方向一致；候选 B main p=0.00214 同向支持 → §4.4 sensitivity gate `overall_strong=True`）。**§4.5 composition diagnostic 在 PR-5 合同下未复制 panel d**：`share_post_minus_baseline` 两配置都是 nominal-positive 但**与 panel d 预期方向相反**（main `+0.0156`, p=0.0149，direction-consistent 6/23；aux `+0.0328`, p=0.0301，direction-consistent 5/22）→ panel d 信号不在 PR-5 cohort 复现，且不为主结论背书。验收口径：PR-5-A PASS / PR-5-B STRONG；`pre_minus_baseline`、`post_minus_pre` 仅留为次级描述层。详见 `docs/archive/topic1/pr5_template_recruitment_plan_2026-04-20.md` §11。
- **未来模型层（§7.9）** 维持冻结：当前不绑 PR 编号，原 PR-6 编号空间已让位给 PR-6A/B/C/D/E 数据发现序列。
- **同步性**：cohort-level interictal synchrony 总体为 null。唯一探索性信号是 extra-focal `phase_e` 的 `pre > post`（p=0.012, r=0.31）。

---

## 3. 核心证据链

### 3.1 内部传播刻板性

来自 `lagPatRank + eventsBool + chnNames` 的 cluster-aware 分析：

- `30/30` subject 的 pairwise Kendall `τ` 分布呈多模态
- KMeans(`k=2`) 后，簇内 `τ` 中位数 `0.250` 显著高于整体 `τ = 0.089`
- `29/30` subject within-cluster `τ > overall τ`
- `30/30` legacy MI permutation 显著，复现老论文结论

合理口径：**一个 subject 常常有多条主要传播路径，每条路径内部仍然刻板。**

### 3.1b 数据合同、聚类稳定性与跨时间复现（PR-2a/2b/2.5）

- 全量 `30/30` subject 都找到 `stable_k`，零 fallback；`stable_k` 分布 `27 × k=2`、`2 × k=4`、`1 × k=6`
- Adaptive within-cluster `τ` 中位数 `0.252`，相对整体 uplift 中位数 `+0.100`
- PR-2.5 时间切片复现：`23/30 strong`、`7/30 moderate`、`0 weak`；split-half 中位模板相关 `0.899`，odd/even block 中位 `0.985`
- `9` 个 k=2 subject 带 `candidate_forward_reverse` 对（inter-cluster `r < -0.5`），其中 `8/9` 跨时间切片可复现互逆关系

完整数值表与算法合同见 archive `interictal_group_event_internal_propagation.md`。

### 3.1c PR-3 / PR-4A：固定模板可视化与 occupancy 漂移

- PR-3 论文级 6-panel cohort 图已固定；新增簇内 identity-bias 计算（median = 86%）
- PR-4A 在固定模板投射前提下做 day/night occupancy timeline：dominant fraction Wilcoxon `p=0.124`、entropy `p=0.245`、TV distance median `0.019`
- 模板投射 agreement 中位数 `0.888`；只有 `3/30`（`chengshuai`、`253`、`818`）低于 `0.8`
- **结论口径**：模板稳定，但占比的昼夜漂移整体较弱。**这是描述层结果，不是强机制结论**
- occupancy 在低 rate 时段天然高方差，不适合直接承担 PR-4C 的主统计读数 → PR-4D 已把这层补强成 `rate×type`

### 3.2 Identity bias 不是小问题，在簇内水平更高

| 层 | raw τ | centered τ | bias fraction |
|---|---|---|---|
| 整体 | 0.089 | 0.023 | 0.652 |
| 簇内 | 0.252 | ≈0.03 | **0.86** |

簇内 86% bias 意味着每个传播模式内部约 86% 通道排序一致性来自固有激活位置（identity ordering），仅 14% 是事件特异性传播动力学。**stereotypy 主要由结构性通道排序驱动**，不否定传播结构（identity 本身反映网络拓扑约束），但量化口径必须更新。

### 3.3 Event-level synchrony 的正式统计口径

PR4–PR6 以 seizure interval 为统计单位，主指标是 `phase`：

- `phase_all` post vs pre：`p = 0.279`
- `phase_core` post vs pre：`p = 0.967`
- within-interval trajectory：`phase_all p = 0.589`，`phase_core p = 0.643`
- event rate：`p = 0.361`

cohort level **不**支持"发作后去同步重置"或"发作前同步性恢复"。

### 3.4 Topic 1 中唯一值得继续追的 synchrony 信号

Epilepsiae 的区域分层分析中：

- `phase_i`：`p = 0.646`
- `phase_l`：`p = 0.543`
- `phase_e`：`p = 0.012`，`r = 0.31`，方向 `pre > post`，Bonferroni 校正后仍勉强保留

这是 synchrony 线中唯一可称为 exploratory-significant 的结果。

---

## 4. 当前最可信的结果

### 4.1 传播刻板性

- 多模态是普遍现象但主要压缩仍是 `k=2`
- 模板跨时间切片总体稳定（`23/30 strong`，余 `7/30 moderate`，无 weak）
- PR-3 / PR-4A 固定模板可视化已稳定，day/night occupancy 漂移弱
- forward/reverse 双模式可复现（`8/9` k=2 subject）
- legacy MI 全部显著，老论文最硬的结果站得住
- 真正可信的定量指标应是 cluster-aware `τ` 与 raw/centered 并列报告

### 4.2 同步性

- 主结论是 population-level null
- `phase` 是主指标，`legacy` 仅作兼容，`span` 仅作附录
- `phase_e` 的 `pre > post` 是唯一需要继续追的分层信号

---

## 5. 仍未解决的问题 / 风险点

- SOZ > non-SOZ 的传播优势仍偏弱，更像探索性趋势，不该写成定论
- centered rank 可能过度校正；`soz_source_erased` 仅 `3/30`，仍必须和 raw 结果并列报告
- PR-4A occupancy 已给出 day/night 描述性答案，但不能升级为正式发作邻近主结论
- `candidate_forward_reverse` 仅是描述标签（`inter-cluster Spearman r < -0.5`），不是生理机制
- 高 k subject 中 `818` 与 `zhangjinhan` 仍需 `n_participating` 匹配子样本验证
- 固定模板投射 agreement 整体够高，但 `chengshuai`、`253`、`818` 这 3 个 subject 解释时间轨迹细节时需谨慎
- synchrony 线最大风险是"把 null 写得太花"；最诚实说法是 **总体 null，局部 extra-focal 线索待验证**
- propagation 与 synchrony 是不同统计对象，文档里必须并列而非混写
- **PR-4B 高置信子集 H2 探索性支持的功效极有限**：n=8 Wilcoxon 最小可能 p = 0.004，当前 p=0.016（W=1）；不能据此做 population-level 结论。`huangwanling` 在 L3 读数上完全 ineligible（n=29）
- **PR-4C 阴性已封板**（2026-04-19，三处合同 P0 修复后复跑两套配置）。原本怀疑被合同 bug 掩盖的 `pre_ictal vs baseline raw_tau aux` 在修复后也消失（p=0.0005 → p=0.141），印证旧版那条信号是 bug 制品
- **PR-5 已完成**：完整阈值 / 失败合同 / sensitivity gate 三态以 archive `pr5_template_recruitment_plan_2026-04-20.md` 为单一来源；本文档只保留判定摘要。已知边界：PR-5-B cohort 必须严格限定在 PR-5-A retained subset（main n=23 / aux n=22）；`pre_minus_baseline` / `post_minus_pre` 主配置未通过 Bonferroni（次级描述层）；§4.5 composition diagnostic `share_post_minus_baseline` 在两配置下都 nominal-significant 但方向与 panel d 预期反向 → panel d 未在 PR-5 合同下复制，且不能为主结论背书
- **未来模型层尚未启动**：硬前置见 §7.9；当前数据发现序列尚未给出已封板的几何/招募一致性结论，模型层维持冻结，不进主线工作量

---

## 6. 传播模板受慢调控的三类读数框架

PR-4 系列的核心问题：**固定传播模板受到什么慢调控？**

| 读数类别 | 科学问题 | 因变量 | 核心指标 | 数据来源 | 当前状态 |
|---|---|---|---|---|---|
| **模板混合 / 模式选择** | 慢调制改变了哪个模板更常出现 / 总 rate 由哪个模板贡献？ | 占比或固定模板分解后绝对事件率 | occupancy fraction（PR-4A）、`rate×type` envelope + stacked count（PR-4D） | `lagPatRank` cluster labels | PR-4A / PR-4D 完成 |
| **模板内顺序一致性** | 慢调制改变了模板内部 rank 一致性？ | within-cluster `τ` | Pairwise Kendall `τ` (high vs low rate) | `lagPatRank` | PR-4B 完成（null） |
| **模板内相对时延结构** | 慢调制改变了模板内部相对时延几何？ | within-cluster lag span / Pearson `r` | fixed-template 内的 relative-lag 统计 | `lagPatRaw` | PR-4B 完成（HC 子集探索性显著） |

三类读数的**数据合同**（per-event min-subtraction、`min_participating ≥ 5` 的 Pearson r 门槛、为何不能用 MI、为何不重新聚类、L3 读数的灵敏度论证）已写入：

- `.cursor/rules/topic1-within-event-dynamics.mdc` "L1/L2/L3 three-layer modulation guardrails"
- `docs/archive/topic1/interictal_group_event_internal_propagation.md`

主文档不再重述。

---

## 7. 推荐的下一步验证

### 7.1 PR-4B：Rate state × stereotype coupling

**状态**：DONE（2026-04-14）。

**结论摘要**：
- L1（模板混合）null：dominant cluster ρ median = −0.083，13/30 正方向
- L2（模板内顺序一致性）null：raw τ p=0.349、centered τ p=0.221（86% identity-bias 限制灵敏度）
- L3 lag span 全 cohort（n=30）：p=0.135，方向一致；高置信子集（n=8）p=0.055
- L3 Pearson r 高置信子集（n=8, dom_r>0.7）：p=0.016, 7/8（**唯一显著**，但功效极有限）
- 跨指标同向性：Spearman(lag_span_Δ, pearson_r_Δ) = 0.628（p=0.0003），24/29 同号
- 综合：H2 在高置信子集**探索性支持**，全 cohort **证据不足**

完整 Step 0 / Step 1 / Step 2-3 数值表与决策合同：archive `interictal_group_event_internal_propagation.md`。

### 7.2 PR-4C：Seizure proximity 双轨口径

**状态**：DONE / CLOSED（2026-04-19）。

**结论摘要**：主 + 辅两配置全量已跑两轮。第一轮（2026-04-17）识别三处合同问题（pair-wise window usability / 候选枚举式事件归属 / gap-aware rate denominator）；第二轮（2026-04-19）完成 P0 TDD 修复后复跑（n_usable_windows 主 187→360、辅 245→370）。

- **传播模式五指标**：cohort Wilcoxon 主 1/15、辅 1/15 名义显著且跨配置不一致 → **模板内部几何无稳健发作邻近调制，正式封板为阴性**
- **唯一稳健信号**：`rate_by_template` post_ictal vs baseline 主 p=0.0009、辅 p=0.0067，方向一致

完整数值、合同问题分析、P0 修复细节：archive `pr4c_seizure_proximity_review_2026-04-17.md` §9。该信号已被 §7.8 PR-5 正式化。

### 7.3 PR-4D：Template-rate decomposition

**状态**：DONE / ACCEPTED（2026-04-16）。PR-4A 的描述层补强。

**结论摘要**：
- 不再"平滑 occupancy"。正式读数只有一个：**固定模板分解后绝对事件率**（`rate×type`）
- 每个 subject 一张图：上面板 smoothed rate envelope（events/hour），下面板同色离散计数堆叠柱
- 暴露分母按真实 `coverage_ranges` 计算；gap 必须留白
- cohort dominant template rate fraction 中位数 `0.584`（range `0.262–0.866`）；`25/30` subject 至少出现一次主导模板交叉，`17/30` 在中高 rate 区间出现，`6/30` 反复交叉
- **PR-4D 是描述层，不能单独升级为推断结论**

图契约与 guardrails 见规则文件 "PR-4D gap-aware rate×type guardrails"。

### 7.4 高 k subject 鲁棒性复核

- 对 `818`、`zhangjinhan` 做 `n_participating` 匹配子样本、raw/centered 双版本模板比较
- 与 PR-4B / PR-5 并行，不阻塞

### 7.5 优先级

1. ~~**PR-4B**（P0）~~ — DONE（2026-04-14）
2. ~~**PR-4C**（P0）~~ — DONE / CLOSED（2026-04-19）
3. ~~**PR-4D**（P1）~~ — DONE / ACCEPTED（2026-04-16）
4. **高 k 复核**（P1）：可并行
5. ~~**PR-5**（P0）~~ — DONE（2026-04-20，PR-5-A PASS + PR-5-B STRONG）；详见 §7.8
6. **§7.6 / §7.7 可选方向**：§7.6 已被 PR-5 吸收为正式分析，§7.7 仍维持 exploratory 子集
7. **未来模型层（§7.9）**：硬前置未达成，维持冻结；不绑 PR 编号

---

### 7.6 后续可选方向：模板招募频率（Topic 1 × Topic 2，已升级为 PR-5 正式分析）

> 状态：**已升级为 PR-5 正式分析**（2026-04-20）。本节保留科学问题陈述与边界，正式合同与判定见 §7.8 + archive `pr5_template_recruitment_plan_2026-04-20.md`。

**科学问题**：PR-4C 唯一稳健发作邻近信号在 `rate_by_template` 层（主 p=0.0009 / 辅 p=0.0067，方向一致）。这指向"哪个模板被招募的频率随发作邻近变化"，而非"模板内部几何如何变形"。

**判定边界**：本方向**不**回答模板内部几何（PR-4C 主分析回答 → 几何已封板 null）；只回答 recruitment frequency。

### 7.7 后续可选方向：高置信子集的窄 exploratory 分支（Topic 1 内部）

> 状态：可选 / 后续；PR-4C P0 已 CLOSED。本方向是**全 cohort 阴性已封板**之后的 exploratory 子集分析，仅辅助叙事，不作为主结论。

**科学问题**：在 8/30 dominant_r > 0.7 高置信 subject 与 8/9 forward/reverse 跨时间复现 subject 上做条件性发作邻近分析，作为机制性 case-series 呈现。

**最小工作合同**：
- subset 定义：(a) `dominant_r > 0.7`（n=8）；(b) `inter-cluster r < -0.5` 且跨时间复现的 forward/reverse 对（n=8/9）
- 复用 PR-4C 修复后主读数（`lag_span` + `pearson_r`），不引入新 metric
- 单独报告 subject-level paired 比较 + 个案展示，不做 cohort-level inferential 论述
- 必须写明 selection criterion 是 Step 0 / PR-2.5 数据质量门槛，避免 cherry-picking
- 不替代 PR-4C 主分析的 cohort 阴性结论

### 7.8 PR-5：Template Recruitment Around Seizures

> 完整合同（数据/假设/失败合同/代码入口/测试合同/工作量）+ §11 复跑结论：`docs/archive/topic1/pr5_template_recruitment_plan_2026-04-20.md`
> PR-5-A gate 中间报告：`docs/archive/topic1/pr5a_novel_template_gate_2026-04-20.md`
> 性质：正式入口。本节只保留**判定摘要**，不重述阈值与 metric 定义（避免与 archive 双源漂移）。

**当前状态（2026-04-20 复跑）**：

| 子 PR | 结论 | 关键数 |
|---|---|---|
| PR-5-A novel-template gate | `overall_pass=True` | retained main n=23 / aux n=22；max \|Δr\|=0.0088、max \|Δe/e_baseline\|=0.0169；六条对比全部满足 archive §3.5 写死阈值 |
| PR-5-B recruitment shift | `overall_strong=True` | retained main n=23 / aux n=22；候选 A `dominant_global` `post_minus_baseline` median main `+65.46` / aux `+42.43` events/h（main p=0.00128 Bonferroni-pass α=0.0083；aux p=0.0115 nominal-pass，方向一致）；候选 B `dominant_per_window` main p=0.00214 nominal-pass 同向 → §4.4 sensitivity gate 三态判定 **strong** |
| §4.5 composition diagnostic | 不复制 panel d | `share_post_minus_baseline` main p=0.0149 / aux p=0.0301，且方向**与 panel d 预期反向**（中位数 `+0.0156 / +0.0328`；direction-consistent 仅 `6/23` 与 `5/22`）→ panel d 未在 PR-5 合同下复制，且不能联动主结论 |

**已知边界（按 archive §7 风险写死）**：
- `pre_minus_baseline` 与 `post_minus_pre` 主配置未通过 Bonferroni → 维持次级描述层
- 高对称 subject `dom_agreement < 0.5`（main 7 / aux 5）按 sensitivity gate `medium` 判读路径，不剔除
- share 与 absolute rate 数学耦合 → 机制层不展开，留未来模型层（§7.9）

**与 PR-4C / PR-4D 的边界**：PR-4C 五指标几何 cohort null 保持封板；PR-4D `rate×type` 描述层保持原状。PR-5 不涉及 SOZ 解剖锚定（属于 Topic 3 §7 独立 P1 候选）。PR-4 PPT panel d（`scripts/plot_topic1_pr4_ppt.py` fig 2d）已在 `docs/topic1_pr4_ppt_figures.md` 与 archive plan §6 同步降级为"历史 motivation / 描述层"，正式归属一律指向 §4.5。

**验收意见（2026-04-20）**：

| 子 PR | 验收 | 关键证据 |
|---|---|---|
| **PR-5-A** | ✅ PASS / ACCEPTED | `overall_pass=True`；六条 gate 对比全部满足 archive §3.5 写死阈值（max \|Δr\|=0.0088，max \|Δe/e_baseline\|=0.0169，所有 Wilcoxon p ≥ 0.21）；retained subset 透明记录；49/49 测试通过 |
| **PR-5-B** | ✅ STRONG / ACCEPTED | `overall_strong=True`；候选 A 主配置 Bonferroni-pass + 辅配置同向 nominal-pass；候选 B 主配置同向 nominal-pass；composition diagnostic 反向结果按 §4.5.3 第二条诚实记录、不联动主结论；cohort filtering 按 retained subset per config 严格执行（regression test 锁死） |

**核心科学结论一句话**：剔除 novel-template / assignment-drift 假象后，dominant template 在 post-ictal 相对 baseline 的**绝对招募率**系统抬升（main +65 ev/h Bonferroni-pass，aux +42 ev/h nominal-pass 同向）；其相对**占比**变化在 PR-5 合同下不复制 panel d，所以这是"绝对招募整体增多、模板间相对权重不偏向 non-dominant"的画面，而**不是** "non-dominant template emergence"。

### 7.9 未来模型层（不绑 PR 编号）

> 性质：未来候选模型方向，不占任何 PR 编号空间，不进数据发现序列主线。本节只锁定模型层启动条件与排除清单；与数据发现序列的边界由 §7.5 优先级与 §7.10 起的数据发现条目共同界定。

**硬启动条件（模型层尚未启动）**：
- 数据发现序列必须先在主文档形成至少一条已封板的几何/招募一致性结论；
- 数据合同冻结后才允许讨论"框架是否能解释现象"；
- 在数据层证据未就位之前，模型层维持冻结，不进任何主线工作量与排期。

**为什么不在数据发现序列内做**：老一代多模板 Hebbian 框架（2024）已经把"框架能解释现象"做完。继续沿同一框架重跑没有论文价值边际。要做下一代框架，必须以 §7.1–§7.8 已钉死的事实 + 数据发现序列通过后的结果做强约束。

**几何 / manifold / persistent homology 方向**：不进数据发现序列主线；当前不立项，仅留作模型层启动后再评估的可选讨论层。

---

## 8. 代码与结果入口

### 内部传播

- 文档：`docs/archive/topic1/interictal_group_event_internal_propagation.md`
- 代码：`src/interictal_propagation.py`
- 脚本：`scripts/run_interictal_propagation.py`、`scripts/plot_interictal_propagation.py`、`scripts/plot_topic1_pr4_ppt.py`
- 结果：`results/interictal_propagation/`

### 事件级同步性

- 文档：`docs/archive/topic1/interictal_synchrony_preliminary_report_2026-04-03.md`
- 代码：`src/interictal_synchrony.py`、`src/interictal_synchrony_aggregation.py`、`src/interictal_synchrony_analysis.py`
- 脚本：`scripts/pr6_interictal_sync_figures.py`
- 结果：`results/interictal_synchrony/analysis/combined/`

---

## 9. 与其他 topic 的边界

- "`~2 Hz` 峰是不是真的"或"IEI serial correlation 说明什么" → `docs/topic2_between_event_dynamics.md`
- "SOZ 和 non-SOZ 到底差在哪里" → `docs/topic3_spatial_soz_modulation.md`
- 同时涉及"传播是否真实"和"慢调制是否发生在 SOZ" → 先分别读 topic 1 和 topic 3，不要混成一个问题

---

## 10. 历史文档索引

- `docs/archive/topic1/interictal_group_event_internal_propagation.md` — 内部传播线的详细结果与合同文档
- `docs/archive/topic1/interictal_synchrony_preliminary_report_2026-04-03.md` — PR4–PR6 的统计报告
- `docs/archive/topic1/pr4c_seizure_proximity_review_2026-04-17.md` — PR-4C 主+辅助配置全量审阅。§1-§8 是 2026-04-17 第一轮审阅（cohort 数值表 / 三处实现合同问题 / P0/P1/P2/P3 路线）；**§9 是 2026-04-19 P0 修复完成后的复跑数值与正式封板结论**。Topic 1 §3.1c / §5 / §7.2 / §7.6 / §7.7 都引用本文件。
- `docs/archive/topic1/pr5_template_recruitment_plan_2026-04-20.md` — PR-5 完整计划合同：科学问题 / 主+备择假设 / 失败合同 / PR-5-A novel-template gate / PR-5-B recruitment shift（含 §4.5 secondary composition diagnostic 独立合同）/ 9 项 TDD 测试合同 / §9 未来模型层占位（不绑 PR 编号，对应主文档 §7.9）/ §11 复跑结论。Topic 1 §5 / §7.5 / §7.6 / §7.8 / §7.9 / §10 都引用本文件。
- `docs/archive/topic1/pr5a_novel_template_gate_2026-04-20.md` — PR-5-A gate 全 cohort 跑数与判定中间报告。
- `docs/archive/topic1/pr4_ppt_per_subject_iteration_summary_2026-04-20.md` — PR-4 PPT/per-subject 综合图的对话迭代记录：版式收敛、关键病例池、以及 SBCI/TRIS 新 metric 需求定义。

这些文档保留为历史事实来源；当前正式口径以本文件为准。

---

## 11. 文档整理里程碑

- **2026-04-20**：主文档大幅瘦身（442 行 → ~270 行），按"主文档只放正式口径，过程性细节回链 archive"原则重写。所有 §-锚点保留；PR-4B / PR-4C / PR-5 的完整数值表、阈值、metric 公式、测试合同、复跑过程已下沉到对应 archive。同步把 `docs/plans/` 下已完成的 yuquan_lagpat 系列（6 个文件）归档到 `docs/archive/yuquan_lagpat/`，event_periodicity Phase 2 plan 归档到 `docs/archive/topic2/plans/`；`interictal_synchrony_analysis_v4.plan.md` 仍有 pending 项，保留在 `docs/plans/`。
