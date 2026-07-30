# Figure 6 state-conditioned predictor：执行与 feasibility gate 审计

**日期**：2026-07-24
**合同**：`interictal_to_early_ictal_state_conditioned_spatial_predictor_v1`
**状态**：**已被新合同取代，不再用于主结果裁决**

> **2026-07-24 口径更正**：本审计使用 EEG onset、1–8 Hz signed field 与
> nested-LOSO static mapper，和论文已经接受的主结果合同不一致。正式基准应为
> clinical onset `[0,10] s`、strict BB 1–150、TA/TB maxAB、all-contact
> channel shuffle、seizure→subject→cohort fold；within-shaft 仅作敏感性。
> 因此本文档以下 no-go 只保留为旧任务的工程与负向探索记录，不能再解释成
> accepted 静态 scaffold 无法复现，也不能阻断新 Fit1/Fit2 与后续 RNN。

## 1. 旧合同的一句话判断

这条分析线已经完成数据合同、无泄漏队列、静态 scaffold gate、敏感性分析、模型实现、单元测试和 GPU 工程 smoke；但 prefix-only 的 TA/TB 静态 scaffold 不能在 nested LOSO 中读取早期发作期 1–8 Hz 空间场，也没有优于 geometry/support baseline，因此当前数据不支持继续训练正式低秩 RNN，更不支持“间期事件可预测发作早期空间招募”的论文结论。

这里的 no-go 只否定当前预注册任务合同，不等于证明任何 RNN 都不能预测发作，也不否定论文中已经建立的 interictal–ictal field reuse 结果。

## 2. 冻结的科学合同

- **主要目标**：EEG onset 后 `[0, 10] s` 的接触点级 1–8 Hz log-power z-score；baseline 为 `[-120, -90] s`。
- **历史输入**：每次发作前 `[-65, -5] min` 的间期事件；不允许使用发作后信息。
- **模板构建**：每位患者仅使用所有评估发作之前的 chronological definite-interictal calibration prefix；主分析为累计 12 个合格 1 h block，6 h 和 24 h 为敏感性。
- **模板方向**：固定的 target-free 定向规则；禁止根据 seizure target 做 mirror 或 template max。
- **rank 处理**：所有事件 rank 都由 `eventsBool` mask 后重新计算，禁止使用 phantom-contaminated legacy rank。
- **验证**：outer LOSO；静态 Gate 1 的映射只在其他患者上学习，held-out 患者的 seizure label 不参与拟合。
- **门控规则**：只有 T0/Gate 1 证明静态间期 scaffold 可读，才允许进入低秩 RNN 的 Gates 2–5。

配置与输入 spec 均记录 SHA256：

- 配置：`config/topic5_state_conditioned_predictor.yaml`
- config SHA256：`290d67974a698d50f073a523de4beab4d6e8c1ab688035b837284807207d2cba`
- source spec SHA256：`81ed32fdf4e806a952b4dda9240386c3ebd674751d6d4efe105dcfad29ea28d9`

## 3. Gate 0：数据与 prefix 资格

主分析共有 17 位候选患者，其中 13 位通过 Gate 0，提供 41 次具有合格历史和主要 target 的发作。

| Gate 0 状态 | 患者数 | 发作数 | 说明 |
|---|---:|---:|---|
| 通过 | 13 | 41 | 满足 prefix、事件数、轴稳定性、历史与 target 合同 |
| 排除：无 finite primary target | 1 | 0 | `epilepsiae_1146` |
| 排除：prefix split-axis 不稳定 | 1 | 0 | `epilepsiae_442`，axis correlation = 0.153 |
| 排除：prefix 事件不足 | 1 | 0 | `epilepsiae_590`，414 events < 500 |
| 排除：definite-interictal prefix block 不足 | 1 | 0 | `yuquan_xuxinyi` |

主要产物：

- `results/topic5_state_conditioned_predictor/dataset/dataset_manifest.json`
- `results/topic5_state_conditioned_predictor/dataset/gate0_attrition.csv`
- `results/topic5_state_conditioned_predictor/dataset/seizure_targets.csv`
- `results/topic5_state_conditioned_predictor/target_cache/`

一个已明确记录的输入偏差是：accepted `lagPat` artifact 不含 HFO energy，因此模型只使用单独命名的 HFO frequency centroid；没有把 frequency 偷换成 energy。

## 4. Gate 1：静态 scaffold 结果

### 4.1 主分析：12 h calibration

canonical nested-LOSO 结果为：

| 指标 | 结果 |
|---|---:|
| 患者 / 发作 | 13 / 41 |
| static TA/TB scaffold，cohort median Spearman r | -0.066 |
| geometry/support baseline，cohort median Spearman r | 0.199 |
| static 相对 geometry 的患者内 median increment | -0.095 |
| paired Wilcoxon，static > geometry | P = 0.927 |
| full contact rank-shuffle null | P = 0.596 |
| within-shaft rank-shuffle null | P = 0.548 |
| Gate 1 | **FAIL** |

静态 scaffold 不仅没有稳定超过 rank-shuffle null，也没有超过较简单的 geometry/support baseline。该结果不满足继续训练状态模型的必要前提。

直接使用固定 `-(TA + TB) / 2` 的非 canonical diagnostic 同样失败：cohort median `r = 0.077`，full-null `P = 0.277`，within-shaft-null `P = 0.432`。它只作为方向核对，不替代 nested LOSO 主结果。

### 4.2 calibration 时长敏感性

| Calibration | 患者 / 发作 | Static r | Geometry r | Median increment | Gate 1 |
|---|---:|---:|---:|---:|---|
| 6 h | 13 / 43 | 0.188 | -0.014 | -0.042 | FAIL |
| 12 h | 13 / 41 | -0.066 | 0.199 | -0.095 | FAIL |
| 24 h | 14 / 36 | 0.216 | 0.146 | 0.016 | FAIL |

6 h 的 full-shuffle nominal `P = 0.040`，但 within-shaft `P = 0.063`，且没有优于 geometry/support；24 h 对两类 null 和 geometry/support 也均未通过。因此不能通过更换 calibration 时长挽救 Gate 1。

主要产物：

- `results/topic5_state_conditioned_predictor/dataset/gate1_static_scaffold_loso/gate1_verdict.json`
- `results/topic5_state_conditioned_predictor/sensitivity/calibration_6h/gate1_static_scaffold_loso/gate1_verdict.json`
- `results/topic5_state_conditioned_predictor/sensitivity/calibration_24h/gate1_static_scaffold_loso/gate1_verdict.json`

## 5. RNN 实现与工程 smoke

已经实现的模型为 event-driven continuous-time E/I RNN：

- 12 个轴向位置，每个位置 4 个 E 单元和 1 个 I 单元，共 60 个单元；
- local E/I backbone 加 effective low-rank recurrent term；
- slow latent state；
- rank 0–4、strict-Dale、no-slow、no-local-inhibition、no-local 等变体；
- Stage A 为 masked reconstruction、next-event、future-balance 自监督任务；
- Stage B 冻结整个 recurrent core，只训练 Ridge linear probe；
- A/B swap augmentation；
- matched GRU、static scaffold、geometry/support、last-event、A/B imbalance、EWMA、linear state-space 和 history Ridge baselines；
- inner LOSO one-standard-error rank selection；
- 与真实顺序模型同初始化、单独训练的 event-order shuffle control。

单患者 GPU smoke 仅用于确认工程链可运行：

- GPU：NVIDIA RTX 3090，PyTorch 2.5.1 + CUDA 12.4；
- held-out：`epilepsiae_1077`，rank 1，seed 20260724，10 epochs；
- true-order held-out pretext loss = 10.437；
- independently shuffle-trained held-out loss = 10.657；
- shuffle minus true = +0.220；
- 峰值 GPU memory 约 449 MiB；
- process-tree RSS 约 1.54 GiB；
- 运行中可用内存始终高于 231 GiB，未发生 OOM。

这个 smoke 样本太小，既不是预测性能，也不是 event-order 科学证据，不能进入论文结果。

资源日志和完成哨兵：

- `results/topic5_state_conditioned_predictor/runs/partial_smoke_v3_resource/resource.csv`
- `results/topic5_state_conditioned_predictor/runs/partial_smoke_v3_20260724/DONE.json`
- `results/topic5_state_conditioned_predictor/runs/partial_smoke_v3_20260724/checkpoints/primary/epilepsiae_1077/rank_1/seed_20260724/DONE.json`

## 6. 验证状态

- 6 个定向单元测试通过；
- 覆盖 post-prefix seizure invariance、phantom rank mask、A/B sign equivariance、strict-Dale 列约束和梯度传播；
- target cache、Gate 0、Gate 1 主分析与两个 calibration sensitivity 均有独立 resource log；
- 每个训练 cell 写 checkpoint、epoch log 和 `DONE.json`，便于恢复和审计；
- 没有发生 CPU 内存、GPU 显存或磁盘阈值越界。

## 7. 为什么正式训练没有启动

根据冻结 spec，Gate 1 是正式低秩 RNN 的必要条件。主分析和两个 calibration sensitivity 全部失败，因此：

- 没有运行 full-cohort rank 0–4 × 3 seeds × outer LOSO；
- 没有进入 Gates 2–5；
- 没有做 RNN latent/pathway 的机制解释；
- 没有报告临床早期预测性能；
- 没有把单患者 smoke 当成绕过 gate 的替代结果。

这是预注册停止规则的执行结果，不是工程中断或资源不足。

## 8. 当前安全结论

当前可以写：

> Under a leakage-controlled prefix-only definition, the static interictal TA/TB scaffold did not generalize to the early-ictal 1–8 Hz spatial field beyond geometry/support and rank-shuffle controls; the pre-registered recurrent-model analysis was therefore not entered.

当前不能写：

- 间期 TA/TB rank 可以预测发作早期空间招募；
- 低秩 E/I RNN 学到了 interictal-to-ictal stereotyped pathway；
- RNN 优于简单 baseline；
- 当前结果具有临床发作预警价值；
- Gate 1 失败否定了已完成的 interictal–ictal field reuse 主结果。

## 9. 最小下一步

1. 若主文篇幅紧张，将本结果作为 negative feasibility audit 保留在补充材料，不把它包装成 Figure 6 的阳性结果。
2. 若仍希望构建 Figure 6，需要先提出一个新的、独立冻结的任务合同，例如改变可解释 target 或引入外部训练数据；不能在看到 Gate 1 后反复选择频段、时间窗或模板方向。
3. 可做一条新的 exploratory diagnostic：比较 prefix-only scaffold 与 full-record scaffold 的漂移，判断 no-go 来自 calibration instability、跨患者 mapping 不可迁移，还是 target 本身与 rank scaffold 不匹配。该分析必须明确标为新问题，不能回填当前预注册 pipeline。

## 10. Paper-ready 中间图

- `results/paper-ready-figure/fig6_state_conditioned_predictor/figures/fig6_feasibility_gate_audit_intermediate.png`
- 同目录提供 PDF、SVG、summary JSON 和中文 `README.md`。

图中依次展示：任务合同、Gate 0 attrition、Gate 1 paired comparison、6/12/24 h sensitivity、GPU smoke，以及预注册 no-go 决策。它是 feasibility gate audit，不是最终预测图。
