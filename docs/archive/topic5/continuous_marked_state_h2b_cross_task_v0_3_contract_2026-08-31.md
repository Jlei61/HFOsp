# H2b Cross-task Transfer v0.3 冻结合同

## 1. 定位

名称：**Frozen Interictal State, Incremental Hazard, and Seizure-entry Geometry**。

v0.3 是在看过 v0.2 development 结果后的测量工具重设计，不是独立预注册确认。v0.2 保持只读；v0.3 写入独立结果根：

```text
results/epi_prssm/continuous_marked_state/h2b_cross_task/v0_3/
```

机器合同为 `config/topic5_continuous_marked_state_h2b_v0_3.json`；运行前必须逐字节冻结为结果根的 `analysis_contract.json`。

## 2. 唯一科学问题与三个假设

一个完全由连续背景与间期事件学习、随后冻结的状态，是否保留当前窗口无法提供的历史信息；该历史信息是否在正确时间尺度上改变未来发作风险，并使状态进入可重复的 seizure-entry dynamical regime？

- **T：增量迁移。** `persistent state` 在 nuisance、history、current observation 之外增加 held-out 发作风险信息。
- **M：跨窗口记忆。** persistent history residual 优于由同一当前窗口得到的 memoryless state。
- **D：时间与动力学专属性。** 状态在自身时间常数内表现为 basin occupancy、directed approach 或 abrupt off-manifold exit，并在超出该时间尺度后减弱。

只有 T、M、D 同时成立，才允许称 `transferable persistent latent state`。只有 T 成立时，只称 `transferable representation`。

## 3. 永久边界

- state、observer、generator、IED decoder 在 seizure task 前冻结。
- seizure loss 不得更新上游，也不得用于 state qualification、latent dimension 或 manifold 选择。
- formal、sealed、H3/T2 和 physical-clock family 在 v0.3 全部关闭。
- 不把 prediction 写成机制、因果或临床预测器。
- 不使用 UMAP/CEBRA 的颜色分离承担证据；UMAP 仅可视化。
- 不重新聚类 seizure subtype 挽救 phenotype。
- sleep/wake 只能来自经过验证的 metadata；不得把 `vigilance` 当作 day/night 或 sleep 标签。

## 4. A0：合同和 attrition

在任何新 outcome fit 前冻结：患者、seizure ID、coverage、lead 定义、5 min anchor grid、30 min 主 horizon、5/15/60 min secondary、120 min descriptive、postictal 120 min、全部 design matrix、prequential 规则、null、state qualification、最小相关效应和 claim ladder。

必须审计完整链条：85 total cells → 75 checkpoint available → state cache → patient support → contrast-specific estimability。每个缺失单独归因为 checkpoint failure、无发作支持、coverage、design/cache、crosswalk、wrong-time donor 或统计不可估计；不允许把技术缺失并入科学阴性。

## 5. A1：纯间期 state instrument qualification

只使用 interictal development 数据：

1. Q1 non-collapse：有效秩、方差与 decoder-distance 超过 collapsed/shuffled null；
2. Q2 cross-window information：persistent 在 held-out interictal future IED/background 上胜 memoryless；
3. Q3 generator contribution：分开记录 generator drift、observation correction、open-loop horizon 和 reset recovery；
4. Q4 tau：在 decoder-output metric 中估计 `tau_z`，要求长于 current observation window，并能在合格连续段内观察到衰减；近似常数或区间删失记为不可辨识，不写成生理失败；
5. Q5 seed stability：比较 decoder outputs、距离矩阵或对齐轨迹，不直接比较不可识别的 raw latent 坐标；
6. Q6 not-only-clock：加入 time-of-day、合法 sleep、recording day、time since previous seizure 和 segment 后，interictal 增量仍存在。

输出 `all_frozen` 与 `state_qualified`。后者至少三 seed 通过承重 Q1–Q5，并在所有已验证可用 nuisance 上通过 Q6。缺少真实 sleep/medication metadata 只作限制和分层，不自动判 state 失败；缺少承重 instrument 证据时仍 fail closed，并留在 all-frozen 分母说明原因。

若 state-qualified 为空，停止 downstream seizure probe，转向上游 instrument 修复。

## 6. A2：半合成 assay qualification

在真实 coverage、缺失、seizure count、聚集性、时钟分布、state autocorrelation 和 control sampling 上构造 null、observation-only、persistent-state、clock-confounded、basin、approach、abrupt worlds。

- 最小相关效应固定为相对 held-out log loss 改善 5%。
- 先用 100 次 Monte Carlo 做实现 smoke；冻结全部设置后，只用最终 1000 次批次验收。
- type-I error ≤0.05，95% 上界 ≤0.075。
- power ≥0.80，95% 下界 ≥0.75。
- prequential 初始 K 只从 2/3/4/5 中由该 assay 一次性选择，不看 v0.3 真实 outcome performance。

失败输出 `ASSAY_NOT_SENSITIVE`，不得解释为 biological negative。

## 7. A3–A5：嵌套 hazard、prequential 与时间尺度

完整记录覆盖以 5 min grid 建 anchor，主 outcome 为未来 30 min 是否有 lead seizure，主指标为 full-grid held-out discrete-time hazard log loss。旧 exact onset-minus-lead risk set 仅作 v0.2 bridge sensitivity。主模型：

```text
M0 = C + H
M1 = C + H + O
M2 = C + H + O + Z_persistent
M3 = C + H + O + Z_memoryless
M4 = C + H + O + Z_memoryless + R_persistent_history
```

其中 residual `R_persistent_history` 只能在 outer-training fold 内拟合。主量为 `logloss(M2)-logloss(M1)`，记忆量为 `logloss(M4)-logloss(M3)`，负值有利。

按发作时间做 rolling-origin/prequential：只用过去发作训练并预测下一次。大量 grid anchors 不是独立发作；推断先按 lead seizure/segment、再按患者聚合，seed 不作为患者。

时间专属性使用纯间期估计的 `tau_z`，替换状态为 `Z_t`、`Z_(t-0.5tau)`、`Z_(t-tau)`、`Z_(t-2tau)`、`Z_(t-4tau)`。未来状态只作 acausal falsification。

## 8. A6：冻结状态的 OOS 流形—流场

几何只在 outer-training fold 的连续 clean interictal trajectory 上拟合，预先排除 preictal、ictal、postictal。held-out preictal/ictal 状态只能 OOS projection。

距离优先级：生理 decoder 参数 → IED 时空 decoder outputs → decoder-induced metric → robust whitening + diffusion/kNN geodesic。三类预注册 family：basin gating、directed approach、abrupt exit。

MARBLE 只有在连续采样足够密、局部邻域含多个独立 segment、flow bootstrap 稳定且半合成 dynamics 可恢复时才解锁。

流形模块不得替代 T/M。若 T 在有充分 assay power 时失败，则停止当前 checkpoint 的 claim-bearing downstream heads；all-frozen 几何只可作为明确标注的 instrument diagnostic。

## 9. A7–A8：条件解锁扩展

只有出现稳定 transfer signal，才训练完全匹配且仅用 interictal 数据的 `full`、`background-only`、`IED-shuffled` 上游对照。只有 full 稳定优于两个对照，才允许把迁移归因于 IED learning。

phenotype 只使用不看 state 冻结的连续 seizure-level target：IED–ictal reuse、early recruitment extent/entropy、propagation speed、laterality。它属于 organizational extension，不是 H3。

## 10. 停止与 claim ladder

- A1 失败：停止 downstream，修 instrument。
- A2 失败：写不可估计，不解释真实阴性。
- T 失败且 assay power 充分：拒绝当前 checkpoint transfer utility，不增加更多 head。
- T+、M−、D−：transferable representation。
- T+、M−、仅 5 min D+：acute preictal encoder。
- T+、M+、D−：slow contextual state，无 seizure-time specificity。
- T+、M+、D+：persistent interictal-to-seizure state transfer。
- 再有 full > background/shuffled：IED-specific transfer。
- 再有 phenotype bridge：状态预测 ictal recruitment 组织；仍不能称 H3 因果。

v0.3 完成前不得打开 sealed partition。

## 11. 科学路线偏移审计

每阶段开始和结束都必须机器记录：上游是否仍由间期任务学习并冻结；主比较是否在相同 `C+H+O` 上增量加入 state；memoryless 与 persistent 是否分开；时间/几何是否按纯间期时间尺度 OOS 评价；分母是否为患者与 OOF lead seizure；当前结果能否区分 representation、persistent state 与 ictal organisation；IED-specific/因果措辞是否有独立证据。

任一项失败写 `SCIENTIFIC_ROUTE_DRIFT`，只停止受影响的 claim branch 并回到最近有效模块。工程 PASS 不得替代科学支持，单一探索阴性也不得成为整个项目的总 gate。

Scientific Spec：`continuous_marked_state_h2b_cross_task_v0_3_spec_2026-08-31.md`。执行计划：`continuous_marked_state_h2b_cross_task_v0_3_plan_2026-08-31.md`。
