# Figure 6 state-conditioned predictor：Fit1/Fit2 与正式 RNN 审计

**日期**：2026-07-24
**合同**：`interictal_to_clinical_onset_bb150_scaffold_margin_predictor_v2`
**状态**：**Fit1 PASS；Fit2 PASS；Gate 2 FAIL；按预注册 stop rule 停止机制扩展**

## 1. 一句话结论

按论文已接受的 clinical onset、BB 1–150、TA/TB maxAB、all-contact
channel-shuffle 合同，full-record 静态 scaffold 被精确复现，prefix-only scaffold
也保留了患者级信号；但只有 9 名患者、11 个合格 history-target pair 的正式
nested-LOSO 实验中，frozen-core RNN 没有在患者层面可靠超过最佳简单历史
baseline 或静态 scaffold，且患者内 history-pairing 精确置换不通过。因此当前
成立的是 **prefix-only interictal scaffold retention**，不成立的是
**seizure-specific recurrent-state prediction**。

## 2. 本轮更正后的冻结合同

- 时间锚：Epilepsiae 使用 clinical onset；窗口为 `[0,10] s`。
- 主频带：strict broadband，BB 1–150 Hz。
- 静态读出：accepted TA/TB `maxAB`，每次置换重新选择 A/B 与 mirror。
- 主 null：coherent all-contact channel-label shuffle；先 seizure，再 subject，
  最后 cohort fold。
- pure within-shaft 只作第二层解剖敏感性，不参与 Fit1/Fit2 hard gate。
- Fit2 唯一允许改变的部分：把 full-record interictal scaffold 换成所有目标发作
  之前的 12 h definite-interictal prefix scaffold。
- RNN 输入：clinical onset 前 `[-65,-5] min` 的 masked-rank 事件序列。
- RNN 目标：
  `target_scaffold_margin_bb150 = observed maxAB - all-contact shuffle median`。
  该目标在 A/B 交换下不变，表示无符号 scaffold-expression strength。
- 主验证：nested leave-one-subject-out；rank `0–4`；3 seeds；one-SE 选择最小
  充分 rank；seed 先在患者内折叠。
- 主动态 comparator：每个 outer fold 内仅用训练患者，从 EWMA 与 linear
  state-space 中选择较好的一个。

配置 SHA256：
`935bcca7e97ddfbe23609d9d2b8a534def3b5c27ed5f85ecab95ef111117450b`。

旧审计
`docs/archive/topic5/fig6_state_conditioned_predictor_gate_audit_2026-07-24.md`
使用 EEG onset、1–8 Hz signed field 和另一套 LOSO static mapper，已经标记为
superseded。它的阴性结果与既往“EEG onset 不显著”一致，不能用于否定本轮
clinical-onset accepted scaffold。

## 3. Fit1：accepted full-record benchmark

Fit1 与 accepted parent producer 在 `atol=1e-12` 下逐字段一致。

| 指标 | strict BB 1–150 |
|---|---:|
| 患者 / 发作 | 16 / 106 |
| observed median maxAB | 0.784453 |
| all-contact null median | 0.780170 |
| 患者级 margin median | +0.047995 |
| observed > null | 12 / 16 |
| one-sided paired Wilcoxon | **P = 0.019318** |
| Fit1 | **PASS** |

主要产物：

- `results/topic5_state_conditioned_predictor/fit12_clinical_bb150/fit1/`
- `results/paper-ready-figure/fig6_state_conditioned_predictor/fit12/fit1/figures/`

## 4. Fit2：prefix-only scaffold

候选母清单为 17 名患者、167 次 phenotype-matched 发作。13 名患者通过
prefix-field gate。只替换 prefix scaffold 后：

| 指标 | strict BB 1–150 |
|---|---:|
| 患者 / 发作 | 13 / 71 |
| observed median maxAB | 0.821366 |
| all-contact null median | 0.786008 |
| 患者级 margin median | +0.033940 |
| observed > null | 10 / 13 |
| one-sided paired Wilcoxon | **P = 0.019897** |
| Fit2 | **PASS** |

pooled phenotype-matched 描述性结果为 13 名患者、115 次发作，margin median
`+0.034171`，11/13 为正，`P=0.006714`。RNN 主任务仍只使用预先指定的
strict-BB 目标，不用 pooled 结果替代。

主要产物：

- `results/topic5_state_conditioned_predictor/fit12_clinical_bb150/fit2/`
- `results/topic5_state_conditioned_predictor/fit12_clinical_bb150/fit12_verdict.json`
- `results/paper-ready-figure/fig6_state_conditioned_predictor/fit12/fit2/figures/`

## 5. RNN 数据资格与 attrition

71 次 strict-BB clinical-onset 发作中：

| 层级 | 患者 | 发作 |
|---|---:|---:|
| prefix scaffold | 13 | 115 |
| strict-BB target | 13 | 71 |
| `[-65,-5] min` 内至少 20 个 definite-interictal events | 9 | 11 |

其余 60/71 次发作全部只因 history events 不足而排除；没有 target、临床时间、
prefix 映射或 masked-rank 读取失败。

只有 `epilepsiae_384` 和 `epilepsiae_958` 各有 2 次合格发作，其余 7 名患者
各只有 1 次。因此：

- LOSO 可检验 held-out-subject readout；
- 不能充分检验患者内 seizure-to-seizure state prediction；
- history-pairing null 只有 `2! × 2! = 4` 种独立配对，最小可达 P 值为 0.25。

主要产物：

- `results/topic5_state_conditioned_predictor/dataset_fit2_clinical_bb150/`
- `results/topic5_state_conditioned_predictor/dataset_fit2_clinical_bb150/event_attrition.csv`

## 6. 正式训练完整性

正式网格为：

`9 outer patients × 5 ranks × 3 seeds = 135 cells`。

验收结果：

- 135/135 cell 完成，`outer × rank × seed` 唯一，无重复；
- 每个 cell 均有 `checkpoint.pt`、`predictions.csv` 和科学 `DONE.json`；
- 9 个分片的 config SHA256 完全一致；
- 27 个 `outer × seed` 组合各恰好选中一个 one-SE rank；
- 9 个 run-level `DONE.json` 全部存在；
- D–I 资源 wrapper 均为 `DONE_PROCESS.json`，无 `RESOURCE_PAUSED.json`。

A/B/C 最初由多 outer 分片启动；第一名 outer 完成后，为避免自动继续产生重复
cell，父 wrapper 被有意 SIGTERM，随后在相同 run 目录用 cell-level resume
完成 run-level 汇总。因此 A/B/C 的旧 resource wrapper 留有 `ABORTED.json`，
这是去重操作，不是训练失败；A/B/C 各自的正式 run 均为 15/15 且有
`DONE.json`。

## 7. Gate 2 正式结果

| 比较 / null | 患者级结果 |
|---|---:|
| RNN 相对 nested history baseline 的 median MAE improvement | +0.010857 |
| patient bootstrap 95% CI | **[-0.013123, +0.028086]** |
| paired Wilcoxon，improvement > 0 | P = 0.212891 |
| RNN 优于 history baseline | 5 / 9 |
| RNN 相对 static scaffold 的 median improvement | +0.000504 |
| patient bootstrap 95% CI | **[-0.005510, +0.008123]** |
| RNN 相对 matched GRU 的 median improvement | +0.002821 |
| patient bootstrap 95% CI | **[-0.031405, +0.025022]** |
| exact within-patient history-pairing null | **P = 0.50** |
| Gate 2 | **FAIL** |

history-pairing null 已改为小置换空间精确枚举。观察到的 2-patient median MAE
为 0.085634，4 种置换的 null median 为 0.087306；P=0.50。

主要产物：

- `results/topic5_state_conditioned_predictor/fit2_rnn_final_analysis/gate2_verdict.json`
- `results/topic5_state_conditioned_predictor/fit2_rnn_final_analysis/subject_level_metrics.csv`
- `results/topic5_state_conditioned_predictor/fit2_rnn_final_analysis/event_seed_predictions.csv`
- `results/topic5_state_conditioned_predictor/fit2_rnn_final_analysis/nested_history_baseline_selection.csv`

## 8. 时间顺序与 rank 结果

真实顺序模型相对独立训练的 event-order-shuffle core：

- patient median `shuffle loss - true-order loss = +0.014813`；
- 6/9 患者为正；
- 预先定义的 event-order sanity gate 通过。

这说明 Stage-A core 捕捉到一部分间期事件顺序结构，但该顺序结构没有转化为
可靠的 early-ictal scaffold-margin 动态增量。

one-SE rank 选择：

| rank | 被选次数 / 27 outer×seed |
|---:|---:|
| 0 | **25 / 27** |
| 1 | 1 / 27 |
| 2 | 0 / 27 |
| 3 | 0 / 27 |
| 4 | 1 / 27 |

rank 0 仍保留 local E/I backbone 和 slow state，但没有 effective low-rank
recurrent term。这个结果不支持“少数 low-rank modes 对任务是必要的”，也不
支持继续做 mode lesion、canonical subspace 或 strict-Dale 机制解释。

## 9. Stop rule

Gate 2 未通过，因此本轮没有进入：

- no-slow / no-local-inhibition / no-local ablation；
- strict-Dale confirmation；
- mode lesion；
- canonical latent coordinate / reduced vector field；
- lookback/cutoff information-horizon sweep；
- 连续时间 seizure forecasting。

这是按预注册停止规则结束，不是资源不足或工程中断。对当前小分母继续搜索
ablation、窗口或 seed 会把阴性结果变成事后选择。

## 10. 资源与复现

正式训练使用 9 个可恢复分片，峰值/最低资源记录为：

| 资源 | 记录 |
|---|---:|
| 单进程树最大 RSS | 1.568 GiB |
| 整卡最大显存 | 2598 MiB |
| GPU 最大利用率 | 100% |
| GPU 最高温度 | 59 °C |
| 最低 MemAvailable | 238.318 GiB |
| 最大 swap used | 0.748 GiB |
| 最低磁盘剩余 | 232.852 GiB |
| OOM / resource pause | 0 / 0 |

I 分片最初两次在普通工具 sandbox 中因 NVIDIA driver 不可见而在 cell 创建前
退出；`retry2` 在获准的 GPU 环境中完成 15/15。两次启动失败均保留
`ABORTED.json`，没有覆盖正式日志。

资源与 stdout 日志位于：

- `results/topic5_state_conditioned_predictor/runs/fit2_rnn_final_shard_*_resource_20260724*/`
- `results/topic5_state_conditioned_predictor/runs/fit2_rnn_final_analysis_resource_20260724/`
- `results/topic5_state_conditioned_predictor/runs/fit2_rnn_final_figure_resource_20260724_v3/`

验证结果：

- 相关 pytest：**18 passed**；
- 6 个 Figure 6 核心 producer/analyzer/plotter 均通过 `py_compile`；
- paper-ready PNG 已人工目视 QA 三轮。

## 11. Paper-ready 六块中间图

- `results/paper-ready-figure/fig6_state_conditioned_predictor/fit2_rnn/figures/fig6_fit2_rnn_intermediate.png`
- 同目录提供 PDF、SVG、metadata JSON 与中文 `README.md`。

六块分别展示：

1. leakage-controlled clinical-onset BB150 合同；
2. 17/167 到 9/11 的 attrition；
3. accepted full-record static benchmark；
4. prefix-only static retention；
5. held-out dynamic readout 与 Gate 2；
6. event-order sanity 和 one-SE rank 选择。

## 12. 当前安全与禁止主张

可以写：

> A clinical-onset broadband readout of the patient-specific interictal
> scaffold remained detectable when the scaffold was restricted to a
> pre-seizure calibration prefix. Interictal event order was weakly learnable,
> but a frozen recurrent readout did not show a reliable patient-level
> improvement over prespecified simple history or static-scaffold baselines.

不能写：

- 发作前间期事件已经可以预测下一次发作；
- RNN 已经解释患者内 seizure-to-seizure 差异；
- low-rank recurrent modes 是任务所必需；
- 当前模型支持 stereotyped computational pathway；
- 结果支持临床连续时间 seizure forecasting；
- all-contact channel shuffle 已排除 shaft identity 或局部空间平滑；
- EEG-onset 旧阴性结果推翻了 clinical-onset accepted scaffold。
