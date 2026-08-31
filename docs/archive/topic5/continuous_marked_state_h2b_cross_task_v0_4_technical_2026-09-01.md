# Continuous Marked State H2b Cross-task Transfer v0.4：技术报告

## 1. 科学问题与验收结论

本轮检验：完全不接触 seizure label 的 frozen interictal state，是否能在患者内未见发作上，提供超出 recent IED history 和 current observation 的风险与发作表现信息；同时允许同一患者不同发作由不同 seizure-entry route 驱动。

工程状态为 `PASS_COMPLETE`，科学状态为 `H2B_NOT_ESTABLISHED_DEVELOPMENT_ONLY`。安全结论是：放松“所有发作共享同一多维状态”的 gate 后，主效应仅相对 observation 小幅有利，未胜 history，双 route 未胜单轴，正确时刻、memoryless 与 phenotype 证据不一致；严格 assay power 不足，因此既不支持 H2b，也不允许生物学阴性。

## 2. 冻结合同及审阅中修复的问题

最终合同为 `h2b_v0_4_heterogeneous_seizure_entry_routes_v10`。v1–v10 的变化均在 final output 前留存为 superseded manifest；没有看真实 H2b 结果后调整主 horizon、route 数或显著性阈值。

- v1 的单中心距离对照本身是双侧非线性量，不能代表“单一方向”；v2 改为 TRAIN-only signed single axis。
- v3 把长时间网格的普通二分类 loss 改成每个发作一个 conditional risk-set loss，避免连续 negatives 淹没 held-out seizure。
- v4 移除非合同的 same-segment control 硬门；同段优先，其他有效患者内 coverage segment 可用，并显式调整 context。
- v5 按支持量固定 primary/rolling/descriptive 层级，而不是让所有患者套同一个 60/20/20 split。
- v6 取消按 history/session position 的最近邻硬匹配，改为与 history、observation、state 都独立的 deterministic control sampling；混杂量在 probe 中调整。two-route 要求至少 4 次过去发作、每 route 至少 2 次、中心相隔至少一个 TRAIN-coverage bandwidth，否则退回单 route。
- v7 将 ridge grid 冻结为 `{1, 10, 100}`，排除只在 `alpha=0.01` 出现的 null outliers。
- v8 把 assay 的承重门改为可校准的方向恢复；严格单-replicate power 单独报告且不阻塞队列，但失败时禁止生物学阴性。
- v9 补回独立 `B_history`；v10 再直接保留 `B_route_state-B_history`，避免用两个患者中位数相加回答核心问题。

主 estimand 始终是 30-min held-out equal-seizure-weight conditional log loss：

`logloss(B_route_state) - logloss(B_observation)`，负值有利。

## 3. 状态来源与分母

source inventory 从 v0.3 最终机器审计读取精确路径和 SHA256，不使用 glob。共 10 位患者、46 个 checkpoint/state-cache cells；checkpoint、state cache、state manifest 和 instrument manifest 均逐个复算。状态模型在 seizure task 前冻结，`state_model_updated=false`、`seizure_gradient_enters_state=false`。

30-min 支持如下：

| patient | mapped seizures | supported | OOF | evidence tier |
|---|---:|---:|---:|---|
| epilepsiae_1073 | 9 | 5 | 3 | rolling sensitivity |
| epilepsiae_1077 | 8 | 7 | 4 | rolling sensitivity |
| epilepsiae_1125 | 13 | 8 | 5 | rolling sensitivity |
| epilepsiae_1146 | 2 | 2 | 0 | not estimable |
| epilepsiae_1150 | 1 | 1 | 0 | not estimable |
| epilepsiae_253 | 5 | 4 | 1 | descriptive |
| epilepsiae_442 | 7 | 6 | 3 | rolling sensitivity |
| epilepsiae_548 | 16 | 10 | 4 | primary chronological |
| epilepsiae_635 | 2 | 2 | 0 | not estimable |
| yuquan_xuxinyi | 2 | 2 | 0 | not estimable |

因此 6/10 患者在 30 min 可估计，但只有 1 位属于 primary chronological。46 cells 中 26 个 `COMPLETE_DEVELOPMENT`，20 个 `NOT_ESTIMABLE_PRIMARY_LEAD`；不可估计者没有从分母删除。

## 4. Risk set、时间切分与模型

每个 OOF seizure 建一个 conditional risk set：1 个 `onset-lead` case 和 5 个患者内 controls。controls 必须有完整 recorded coverage、对应 horizon 内无发作、排除 ictal/postictal；同 coverage segment 优先。抽样 key 不使用 history、observation 或 state。outer TEST control 时间严格晚于上一发作 cutoff，TRAIN/TEST rows 不重叠；同一 risk set 的所有比较臂逐行相同。

支持层固定为：支持发作至少 10 次时前 `floor(60%)` 初始化、后 40% prequential OOF；5–9 次以前 2 次初始化作 rolling sensitivity；3–4 次仅 descriptive；少于 3 次不可估计。ridge 只在过去发作内选择。seizure 是 fold 单位，seed 先在患者内取中位，患者是 cohort 单位。

模型为：

- `B_history`：IED history 与严格因果的 seizure/session context；
- `B_observation`：`B_history` 加 current explicit observation 和 TRAIN-defined observation-route distances；
- `B_linear_state`：同一 base 加 raw frozen state；
- `B_single_axis_state`：同一 base 加过去发作定义的 signed persistent axis；
- `B_route_state`：同一 base 加最多两条 TRAIN-only persistent route distances；
- `B_route_memoryless`：同容量 memoryless route；
- `B_route_wrong_time`：训练模型固定，仅替换 TEST state；无合法同段 donor 的 fold 不进入该 contrast，绝不替换成 memoryless。

## 5. 30-min 患者优先结果

| patient | Obs−Hist | State−Hist | State−Obs | State−Mem | Two-route−Axis | Correct−Wrong |
|---|---:|---:|---:|---:|---:|---:|
| epilepsiae_1073 | +0.4951 | +0.4968 | +0.0034 | −0.0399 | NA | +0.0016 |
| epilepsiae_1077 | −0.2351 | −0.3564 | −0.0135 | +0.0017 | NA | −0.0072 |
| epilepsiae_1125 | +0.0453 | −0.2100 | −0.1855 | −0.3812 | +0.0763 | −0.5653 |
| epilepsiae_253 | +0.0304 | +0.0297 | +0.0034 | +0.0028 | NA | +0.0019 |
| epilepsiae_442 | +0.5920 | +0.5233 | −0.0132 | +0.0072 | NA | −0.0495 |
| epilepsiae_548 | +0.3043 | +0.9134 | −0.0547 | −0.1316 | +0.1505 | +0.0164 |

患者层汇总：

| contrast | n | patient median | favourable | bootstrap median 95% | exact sign p |
|---|---:|---:|---:|---|---:|
| `B_observation-B_history` | 6 | +0.1748 | 1/6 | [−0.1023, +0.5436] | 0.2188 |
| `B_route_state-B_history` | 6 | +0.2633 | 2/6 | [−0.2832, +0.7183] | 0.6875 |
| `B_route_state-B_observation` | 6 | −0.0133 | 4/6 | [−0.1201, +0.0034] | 0.6875 |
| `B_route_state-B_memoryless` | 6 | −0.0191 | 3/6 | [−0.2564, +0.0050] | 1.0000 |
| correct-time−wrong-time | 6 | −0.00284 | 3/6 | [−0.3074, +0.0091] | 1.0000 |
| two-route−single-axis | 2 | +0.1134 | 0/2 | [+0.0763, +0.1505] | 0.5000 |

主效应的轻微负值不能单独支撑 H2b：增强的 observation baseline 自身显著不稳，state 直接相对 history 仍为不利方向；persistent-vs-memoryless 和 correct-time 均只有一半患者有利；异质性专属对比在仅有的两位可估计患者中都不利。

预先定义的 R1.7B H1-stable 层只有 3 位可估计：state−observation 中位 `−0.0132`（2/3），但 state−history 中位 `+0.0297`（1/3），同样不能升级。H1 是解释层，不是运行 gate。

## 6. Lead-time sensitivity

| lead | n | State−History median | favourable | State−Observation median | favourable |
|---:|---:|---:|---:|---:|---:|
| 5 min | 6 | +0.1598 | 2/6 | +0.0024 | 3/6 |
| 15 min | 6 | +0.4337 | 2/6 | +0.0039 | 3/6 |
| 30 min | 6 | +0.2633 | 2/6 | −0.0133 | 4/6 |
| 60 min | 3 | −0.0841 | 2/3 | −0.0069 | 2/3 |
| 120 min | 0 | NA | NA | NA | NA |

30 min 保持唯一 primary。60 min 只有 3 位患者，不能事后改选为最佳 lead。

## 7. 真实 coverage 半合成 assay

使用真实 coverage、缺失、自相关和支持分母，独立 calibration/evaluation 各运行 100 replicates/world；worlds 为 null、observation-only、persistent single-route、persistent two-route、clock-confounded。8 个 process workers、每 worker 单线程。

三个负对照的 primary false-increment/type-I calibration 全部通过。方向恢复率为：

- single-route：state−observation 82%，state−memoryless 70%，correct−wrong-time 66%；
- two-route：state−observation 82%，state−memoryless 75%，two-route−single-axis 88%，correct−wrong-time 89%。

two-route 路径的承重方向检查全部通过，但 single-route correct-time 未达到预设 70%。更重要的是，基于独立 null 5% 阈值的五项 strict single-replicate power 均未达到 80%。最终 assay 状态为 `ASSAY_NOT_DIRECTIONALLY_SENSITIVE_NO_BIOLOGICAL_NEGATIVE`：可以运行真实 development 队列，但不能把真实 null/反方向解释成不存在效应。

## 8. OOS seizure-entry geometry 与 phenotype

decoder-metric OOS geometry 有 4 位患者可评分。abrupt transition 为 4/4 有利，中位 1.0，exact sign `p=0.125`；route basin gating 为 3/4，有宽区间；directed approach 仅 1/3 非零患者有利。route-minus-single geometry 的三个患者中位均为 0。结果只支持描述性的发作进入突变，不支持 route-specific transferable geometry。

phenotype target 来自既有 R3 recruitment observed/margin summary，未按 state 重聚类。共 206 probe rows、22 attrition rows、12 个可估计 patient-target rows；state−observation loss 中位 `+0.0008585`，仅 1/12 有利。route-specific phenotype 为 `NOT_ESTIMABLE_TARGET_SUPPORT_TOO_SPARSE`。因此没有 interictal-to-ictal phenotype transfer 证据。

## 9. 工程验收与资源记录

- 46/46 expected cells 收口；26 complete、20 support-limited not estimable；
- 8 CPU workers，`OMP/MKL/OPENBLAS/NUMEXPR=1`，GPU disabled；
- measured sentinel RSS 474,824 kB，最大 worker RSS 488,148 kB；
- elapsed 90.98 s，first-pass OOM 0；serial OOM retry policy 保留但未触发；
- assay 8 workers，100 calibration + 100 evaluation/world，elapsed 172.27 s；
- test：`180 passed, 5 warnings`，warning 均为 PyTorch Transformer nested-tensor 提示；
- machine audit：所有冻结、hash、因果、risk-set、split、route、wrong-time、patient-first、formal/sealed 边界检查均为 true；
- 未运行 H3、T2、physical clock；未修改 paper-ready figures。

关键 SHA256：analysis contract `9636149d…8c4fb4`；source inventory `f0ab2a5f…9904e4`；estimator `fc5428f1…12437`；assay summary `d1919a35…e3acf`；cohort summary `578c7fdb…37aad`。完整 checkpoint/cache 与 source hash 表见 `reports/machine_audit.json`。

## 10. 复现命令

在隔离 worktree `/tmp/hfosp_h2b_v04_20260831`：

```bash
PY=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
ROOT=results/epi_prssm/continuous_marked_state/h2b_cross_task/v0_4

$PY scripts/topic5_continuous_marked_state_h2b/initialize_v04.py --result-root "$ROOT"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  $PY scripts/topic5_continuous_marked_state_h2b/run_v04_queue.py \
  --result-root "$ROOT" --max-workers 8 --measured-rss-kb 474824
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  $PY scripts/topic5_continuous_marked_state_h2b/run_v04_assay.py \
  --result-root "$ROOT" --workers 8 --calibration-replicates 100 \
  --evaluation-replicates 100
LD_LIBRARY_PATH=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib \
  $PY scripts/topic5_continuous_marked_state_h2b/run_v04_phenotype.py --result-root "$ROOT"
$PY scripts/topic5_continuous_marked_state_h2b/aggregate_v04.py --result-root "$ROOT"
$PY scripts/topic5_continuous_marked_state_h2b/closeout_v04.py --result-root "$ROOT"
```

## 11. 最终判断与下一步

放松 gate 是合理的科学修正，但不是降低证据标准。结果说明：允许不同发作走不同 entry route 后，仍没有形成 history 之外、跨窗口、正确时刻特异且可预测发作表现的一致 frozen-state increment。

下一步不应打开 formal/sealed，也不应继续增加 route 数或深度 seizure classifier。应先获得更多具有至少 10 次完整 lead coverage 的患者，并提高 real-coverage assay 的 single-route time-specificity 与 strict single-replicate power；只有这些条件满足后，才值得预注册独立 confirmation。
