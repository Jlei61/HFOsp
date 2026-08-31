# H2b Cross-task Transfer v0.3 技术收口报告

## 0. 验收结论

本版本验收为：

```text
PASS_EXPLORATORY_CLOSEOUT_H2B_NOT_ESTABLISHED
```

验收对象是 development-only 的完整执行与证据分层，不是 H2b 科学阳性。A0–A6、A8 已按“少 gate、多探索”附录完成；A7 根据操作性优先规则未触发。formal、sealed、H3/T2、physical-clock family 与 paper-ready figures 未开启。

安全结论为：**现有 R1.7B 冻结 checkpoint 没有形成满足严格多维资格的 persistent state；完整时间网格上的风险迁移、跨窗口记忆和时刻专属性没有患者级一致性。OOS 几何中最一致的探索模式是 abrupt off-manifold transition（4/4 可估患者），但分母小、A1 为空且风险 assay 对 persistent transfer 严重欠功效，因此 H2b 未建立，也不能作生物学阴性。**

## 1. 科学目标与 claim ladder

唯一核心问题是：只在 interictal/background 数据上学习并冻结的表示，能否在不接受 seizure loss 梯度的前提下迁移到未来发作风险和发作入口组织。

v0.3 的三个承重量为：

- **T，incremental transfer：** frozen persistent representation 是否在 causal nuisance、显式 IED history 和 current observation 之外降低 held-out 30-min risk-set log loss；
- **M，persistent memory：** persistent history 中不能由 current memoryless code 解释的残差，是否继续降低 log loss；
- **D，time/dynamical specificity：** correct-time 是否优于 matched wrong-time，旧状态是否随 lag 退化，以及 OOS seizure-entry trajectory 是否表现为 basin gating、directed approach 或 abrupt exit。

只有 T、M、D 共同成立才允许称 `transferable persistent latent state`。A1/A2 失败只降低措辞，不阻止 development exploration；这一执行语义由 `continuous_marked_state_h2b_cross_task_v0_3_exploration_addendum_2026-08-31.md` 覆盖原合同中的串行停止条款。

## 2. 数据、分区与冻结边界

### 2.1 A0 分母

R1.7B inventory 共 85 个患者×seed 单元：

- checkpoint 可读 75；
- `NONFINITE_GRADIENT` 无 checkpoint 10；
- v0.2 旧 support-conditioned state cache 可用 46；
- 旧 attrition 中无冻结 seizure support 24；
- 主 horizon coverage 不足 5；
- 完整时间网格最终覆盖 10 位患者、46 个 checkpoint。

attrition audit 在读取任何 probe loss、coefficient、rank、p-value 或 phenotype value 前冻结。H1 既有标签只作分层，不筛患者。

### 2.2 因果边界

- state/observer/generator/IED decoder 在 seizure task 前冻结；
- 5-min anchor 由记录覆盖生成，query builder 不读取 seizure table；
- 每个 cache 的 `max_source_time_epoch <= anchor_time_epoch`；
- rolling-origin fold 只用该 cutoff 已发生、已可知的 seizure label；
- wrong-time 和 lag donor 只来自同 recording segment 的过去合法时刻；
- patient 是 inference unit，seed 先在患者内汇总；
- absolute timestamps 为 float64；
- formal/sealed 均未读。

## 3. A1：纯间期 state instrument

A1 审计 16 位患者、75 个可读 checkpoint。患者资格规则要求至少 3 个 seed 联合满足：non-collapse、persistent cross-window increment、generator contribution、可辨识 decoder-metric tau、功能 seed stability，以及可用 nuisance 下 not-only-clock。

结果：

| 分层 | 患者数 | 允许解释 |
|---|---:|---|
| strict state-qualified | 0/16 | 无患者可承担 persistent-state claim |
| scalar slow-axis candidate | 10/16 | 可作单轴探索，不是合格多维状态 |
| collapsed/unusable | 6/16 | 不能解释为持续状态 |

值得注意的是，一些患者分别通过 seed geometry、tau 或 clock-sensitive 子项，但没有患者在至少 3 个 seed 上联合通过全部承重条件。单项相似度高不能替代联合资格。

## 4. A2：半合成 assay

A2 使用真实 coverage、缺失、seizure count、时刻分布、state autocorrelation 与 control sampling。主批次 7 worlds × 100 replicates；K 选择另用 800 replicates，选定 `initial_k=2`。选定 null thresholds 为 T `0.275603`、M `0.384867`。

| 世界 | T detection | M detection | joint T+M+lag | 正确几何 family recovery |
|---|---:|---:|---:|---:|
| null | 5/100 | 6/100 | 2/100 | 不适用 |
| observation-only | 5/100 | 7/100 | 2/100 | 不适用 |
| persistent-state | 4/100 | 4/100 | 1/100 | 不适用 |
| clock-confounded | 4/99 | 5/99 | 0/99 | 不适用 |
| basin-gating | 5/100 | 7/100 | 2/100 | 99/100 |
| directed-approach | 3/100 | 3/100 | 1/100 | 100/100 |
| abrupt-transition | 11/100 | 8/100 | 5/100 | 99/100 |

结论：family classifier 对几何形态敏感，transfer contrast 的 null calibration 约为目标 5%，但对 persistent T+M+lag 基本无功效。该 assay 只允许将人体风险结果标为 `EXPLORATORY_ASSAY_NOT_SENSITIVE`，不允许 biological negative。

## 5. full-grid state extraction

完整记录覆盖每 300 s 建一个 anchor，不按 seizure support 条件抽样。

| 患者 | seeds | unique anchors | A1 exploration stratum |
|---|---:|---:|---|
| E1073 | 5 | 1,307 | collapsed/unusable |
| E1077 | 1 | 1,786 | collapsed/unusable |
| E1125 | 5 | 1,055 | scalar slow axis |
| E1146 | 5 | 902 | scalar slow axis |
| E1150 | 5 | 881 | scalar slow axis |
| E253 | 5 | 2,768 | scalar slow axis |
| E442 | 5 | 361 | scalar slow axis |
| E548 | 5 | 912 | collapsed/unusable |
| E635 | 5 | 438 | collapsed/unusable |
| Yuquan xuxinyi | 5 | 187 | scalar slow axis |
| **合计** | **46 cells** | **10,597** | 10 patients |

seed 展开后为 45,841 cell-anchor rows。队列 46/46 COMPLETE、0 failure、0 OOM。8 CPU workers，BLAS/OpenMP thread=1；36 份 streaming resource receipts 中最大 RSS `1,590,300,672` bytes（约 1.48 GiB）。

## 6. A3–A5：full-grid prequential hazard 与 lag

### 6.1 嵌套模型

同一 fold 内比较：

```text
M0 = causal clinical/time context + explicit IED history
M1 = M0 + current explicit observation
M2 = M1 + frozen persistent state
M3 = M1 + frozen memoryless state
M4 = M3 + outer-training-only residualized persistent history
```

T 定义为 `(loss(M1)-loss(M2))/loss(M1)`，正值有利。M 定义为 `(loss(M3)-loss(M4))/loss(M3)`，正值有利。M4 residualizer 只在 outer-training fold 内拟合。

### 6.2 患者级结果

| 患者 | seeds | OOF seizures 中位 | T | M | correct-time − wrong-time 的 T 改善差 |
|---|---:|---:|---:|---:|---:|
| E1073 | 5 | 7 | +0.04301 | −0.00844 | +0.17077 |
| E1077 | 1 | 6 | +0.00939 | +0.04044 | −0.07251 |
| E1125 | 5 | 10 | +0.05798 | −0.03368 | +0.41319 |
| E253 | 5 | 3 | −0.05034 | −0.12166 | −0.05627 |
| E442 | 5 | 5 | −0.04073 | −0.11884 | −0.44780 |
| E548 | 5 | 12 | +0.00098 | +0.00103 | +0.34624 |
| E1146/E1150/E635/xuxinyi | — | — | NOT_ESTIMABLE | NOT_ESTIMABLE | NOT_ESTIMABLE |

cohort direction：

- T：4/6，two-sided sign p=0.6875；
- M：2/6，p=1.0；
- correct-time better than wrong-time：3/6，p=1.0。

这些 p 值只描述患者方向，不能补偿 A2 低功效。E1077 只有一个可读 seed，须单独保留该限制。

### 6.3 lag-response

lag donor 是同 segment、严格过去、位于 `t-lag` 或更早的最近状态。

| lag | 有利患者 | 患者中位 T | donor-valid fraction 中位 |
|---|---:|---:|---:|
| 0.5 tau | 2/6 | −0.04305 | 0.903 |
| 1 tau | 3/6 | −0.01354 | 0.834 |
| 2 tau | 2/6 | −0.01832 | 0.738 |
| 4 tau | 1/6 | −0.02138 | 0.552 |

没有稳定的 correct-time advantage，也没有从 0.5 tau 到 4 tau 的单调退化曲线。4 tau donor coverage 已明显下降，不能把更远 lag 的负值单独包装。

### 6.4 support-conditioned 旧结果隔离

旧结果只在满足 seizure-support 的时刻抽状态，会把 outcome support 混入 anchor availability。它已移入：

```text
quarantine/support_conditioned_hazard_v2/
```

该层同为 46 cells、10 patients、6 estimable，T 4/6、M 3/6、correct-time 2/6。机器 closeout 强制要求该 summary 留在 quarantine 并记录 hash；任何报告不得将其与 full-grid 的 4/6、2/6、3/6 合并。

## 7. A6：common-domain OOS manifold/flow

### 7.1 拟合合同

- decoder-output geometry 只在 held-out seizure 前、排除已知 seizure 周围窗口后的 clean interictal full-grid rows 上拟合；
- PCA/whitening、two-basin centres、entry centroid 和 entry direction 均只用过去；
- held-out preictal trajectory 仅 OOS 投影；
- matched control trajectories 的 endpoint 不晚于上一场已知 seizure；
- case 与 control 使用同一 full-grid extraction domain；
- family score 是相对同患者 matched controls 的有符号百分位，范围 `[-1,+1]`。

三类 family：

- basin gating：entry-basin occupancy 与最长停留；
- directed approach：approach rate 与 flow alignment；
- abrupt transition：最大 off-manifold distance。

### 7.2 患者结果

| 患者 | complete folds | abrupt | basin | approach |
|---|---:|---:|---:|---:|
| E1077 | 5 | +1.000 | +0.450 | −0.400 |
| E1125 | 25 | +1.000 | +0.778 | +0.500 |
| E253 | 15 | +0.500 | −0.150 | −0.050 |
| E548 | 43 | +1.000 | +0.400 | 0.000 |

患者方向：abrupt 4/4、median +1.000、p=0.125；basin 3/4、median +0.425、p=0.625；approach 1/4、median −0.025、p=1.0。

其余 6 位 NOT_ESTIMABLE。cell-level 主要原因是 `collapsed_decoder_geometry` 5 cells、`insufficient_past_full_grid_rows` 5 cells；fold-level 还包括 25 个 heldout trajectory coverage 不完整、55 个过去 full-grid rows 不足、2 个 entry direction 相消和 1 个 prior entry trajectory 不足。

该结果支持下一版将 abrupt exit 作为单独复现目标，但不支持把它称作队列级 seizure-entry manifold。4/4 的 nominal sign p 仍为 0.125，且所有患者的 strict A1 均为 false。

## 8. A8：冻结连续 ictal-organisation target

target 源固定为 `field_concordance_grid_method_sensitivity/n161_frozen_per_model/parent_anchor_event.csv`，只使用连续 `r3_observed` 与 margin；没有在看 state 后重新聚类或阈值化。

源表没有 onset timestamp，因此使用冻结的 patient + seizure index chronological mapping；这不是已验证的 onset crosswalk。后段 index attrition 显式保留。

主观察 target 的 evaluative rows：

| 患者 | tier | target seizures | state − observation MSE |
|---|---|---:|---:|
| E1125 | sensitivity LOSO | 5 | −0.000994 |
| E442 | sensitivity LOSO | 4 | +0.000620 |
| E548 | sensitivity LOSO | 5 | +0.001649 |

负值有利，故 1/3 患者方向有利。E1077、E1146、E635 只有 2–3 次，保留为 descriptive case series；E253 和 xuxinyi 各 1 次，不可估计。

## 9. A7 条件决策

A7 是 matched `full/background-only/IED-shuffled` 上游重训。v0.3 没有实现并运行该昂贵分支，而是用写入 closeout 的操作性优先规则决定是否值得启动：T 和 correct-time 患者方向均需 ≥0.70，并至少有一条独立 geometry/phenotype direction ≥0.70。该 0.70 不是 p-value 或新科学阈值。

实际为 T `4/6=0.667`、correct-time `3/6=0.5`；虽 abrupt 为 4/4，前两项不满足。因此：

```text
NOT_TRIGGERED_NO_COHERENT_CROSS_DOMAIN_SIGNAL
```

这只表示当前不值得做复杂 source ablation，不表示 IED source 无作用。

## 10. 单患者收敛线索与替代解释

E1125 同时具有 T +0.05798、correct-time difference +0.41319、5/5 seed 同向、三类 geometry 均为正，并在 frozen ictal-reuse target 上 state−observation 为 −0.000994。它是本轮最强的复现候选。

但 E1125 的 M 为 −0.03368，说明 current observation 之外的 persistent residual 没有增量。可能解释包括：

1. 当前窗口的 time-specific code，而非跨窗口 carry；
2. scalar slow axis 恰好与发作背景共变；
3. preictal abrupt excursion，而非稳定的慢方向；
4. 小发作分母下的患者特异偶然性。

因此不能以该病例反向选择模型或改阈值。它只能进入冻结定义的 independent replication。

## 11. 工程与复现审计

- full-grid state queue：46 requested，31 new + 15 pre-existing，0 failure；
- hazard queue：46/46；geometry queue：46/46；phenotype full grid：46 cells、10 patients；
- query anchors：10,597 unique，45,841 cell-expanded；
- kernel OOM：false；OOM retries：0；
- resource audit：`PASS_NO_OOM_BOUNDED_WORKERS`；
- patient-first、seed-not-patient 由 machine audit 验证；
- state cache、query manifest、queue、summary、quarantine summary、producer 与 test log 均写 SHA256；
- scoped tests 的最终数字以 `logs/pytest_topic5_h2b_v03_final.log` 和 machine audit 为准；
- 机器总状态：`PASS_EXPLORATORY_CLOSEOUT_H2B_NOT_ESTABLISHED`。

执行期间发现一个并发 strict-controller 会把 exploratory 输出移走并用空循环占锁。最终 closeout 从提交 `886296cb` 建立隔离 worktree，恢复用户指定的 exploration addendum，并以 full-grid producer 重建结果；strict 版本遗留内容不进入本报告。

## 12. 科学路线审计

本轮没有偏离“interictal discovery → freeze → cross-task seizure validation”：

- 上游状态未接受 seizure loss；
- anchor grid 不由 seizure outcome 生成；
- hazard 是对 current observation 的嵌套增量，不是单独 classifier；
- persistent 与 memoryless 分开；
- geometry 是过去 interictal fit、held-out seizure OOS projection；
- phenotype target 在 state 前冻结；
- A1/A2 失败只限制措辞，没有删除真实探索结果；
- T/M/D 不一致时没有升级为 persistent state；
- 未执行 H3、T2 或 causal shaping。

因此路线对齐，但当前瓶颈明确落在 upstream state identification，而不是 seizure head 容量。

## 13. 下一版本的最小科学任务

### 13.1 R1.8 upstream state identification

在纯 interictal/background 任务中同时识别 fast observation code 与 slow carried state；模型选择必须依赖跨窗口增量、correct-time specificity、decoder-output seed geometry 和可辨识时间尺度。发作标签仍完全隔离。

### 13.2 abrupt-transition locked replication

冻结本轮的 decoder metric、clean interictal fit、lookback、matched-control 和 `max_off_manifold_z` 定义；在未用于本轮 family 选择的患者或后段 development 数据上只检验 abrupt family。不得同时优化 manifold algorithm、basin number、lead 或 family。

### 13.3 重新进入 H2b 的条件

只有新上游先在纯间期数据中产生至少一个可复算、跨 seed 合格的状态，才重跑同一 full-grid T/M/correct-time/lag/OOS geometry。不能继续在现有不合格表示上增加 seizure head，也不能把 E1125 用作超参数选择集。

## 14. 权威产物

- `results/epi_prssm/continuous_marked_state/h2b_cross_task/v0_3/reports/machine_audit.json`
- `results/epi_prssm/continuous_marked_state/h2b_cross_task/v0_3/reports/scientific_route_audit_A3_A8.json`
- `results/epi_prssm/continuous_marked_state/h2b_cross_task/v0_3/hazard_full_grid/patient_first_summary.json`
- `results/epi_prssm/continuous_marked_state/h2b_cross_task/v0_3/geometry/patient_first_summary.json`
- `results/epi_prssm/continuous_marked_state/h2b_cross_task/v0_3/phenotype_continuous/summary.json`
- `results/epi_prssm/continuous_marked_state/h2b_cross_task/v0_3/full_grid/RESOURCE_AUDIT.json`
- `results/epi_prssm/continuous_marked_state/h2b_cross_task/v0_3/logs/pytest_topic5_h2b_v03_final.log`
