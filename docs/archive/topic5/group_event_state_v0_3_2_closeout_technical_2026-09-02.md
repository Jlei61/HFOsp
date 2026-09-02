# Group-Event State v0.3.2 阶段收口（技术版）

**日期：** 2026-09-02  
**状态：** `V0_3_2_INSTRUMENT_UNSTABLE_DEVELOPMENT_CLOSEOUT`  
**分区：** development-only；`sealed_partition_opened=false`。

## 1. 版本与执行范围

两条实现分支已合并到 `codex/topic5-group-event-state-v032-closeout`：

- model：`07b33e26`；
- evaluation：`81d36b74`；
- merge commits：`1d65c7e8`、`9b4f456e`。

当前执行矩阵：

| 模块 | 分母 | 状态 |
|---|---:|---|
| measurement/support/eligibility | 27 patients | complete |
| history baseline | 27/27 | complete |
| leaky-bank model | 3 patients × 3 seeds | complete |
| random-reservoir model | 3 × 3 | complete |
| frozen state registry | 9 entries | complete |
| H1 paired evaluation | 3 × 3 | complete |
| H2a frozen grammar probe | 3 × 3 | complete |
| preregistered positive synthetic | 3 replicates | complete, fail |
| H-only null synthetic | 6 replicates | complete, pass |
| added beta ladder | 3 betas × 3 replicates | complete, non-monotonic |
| repaired-RNN architecture triage | — | not run after assay failure |
| H2b/H3 | — | out of v0.3.2 scope |

相关 v0.3.2 测试共 65 项通过。leaky-bank 主模型的最大 PyTorch peak allocation 为 56.6 MB；H2a 使用每 GPU 两个 worker 并行完成，未发生 OOM。收口时无相关训练进程存活。

## 2. 数据资格

资格来自模型结果之前冻结的 `endpoint_eligibility.json`。

| subject | 5-min count | 30-min count | 120-min count | H2a |
|---|---:|---:|---:|---:|
| `epilepsiae_1146` | yes | yes | no | yes |
| `yuquan_pengzihang` | yes | no | no | yes |
| `yuquan_zhangkexuan` | yes | no | no | yes |

30 分钟非重叠块数（base-fit / inner-val / dev-val / dev-test）：

- `epilepsiae_1146`：61 / 11 / 10 / 22；
- `yuquan_pengzihang`：20 / 2 / 3 / 7；
- `yuquan_zhangkexuan`：23 / 4 / 4 / 8，base-fit 比冻结阈值少 1。

因此 30 分钟 primary 的患者分母严格为 1，不得将另外两人的开发数值纳入主统计。120 分钟主分母为 0。

## 3. 模型合同

主状态为 12 维 fixed-timescale marked leaky bank：每个 5/30/120 分钟时间尺度 4 个通道。event encoder 输出 4 维 mark innovation；时间差只进入固定指数衰减，不作为普通 event-token feature重复输入。

主任务只训练未来 30 分钟 negative-binomial count residual：

```text
log μ(H+S) = log μ(H) + α wᵀS
```

其中 `H` 为冻结的 `H_strong`，adapter 使用非零小 gate；state trajectory 以 open-loop 方式导出。H2a 读取冻结 state，不向 count-state 反传梯度。

## 4. 合成定标

### 4.1 预登记结果

| assay | 结果 | 判定 |
|---|---:|---|
| H-only null | 0/6 false positives；median gain −0.00881 nats/anchor | pass |
| residual positive, β=0.35 | 0/3 recovered | fail |

阳性三个 replicate 的 `H−correct` 均值为 +0.0193、+0.0866、+0.0020，但三者 CI 下界均未超过 0，因此 0/3 满足完整恢复规则。

### 4.2 追加效应阶梯

| β | recovered | median gain | pass |
|---:|---:|---:|---:|
| 0.35，新 seed | 2/3 | +0.0227 | yes |
| 0.70 | 3/3 | +0.1931 | yes |
| 1.40 | 1/3 | +0.2738 | no |

该结果同时暴露两项问题：同一 β 在不同 seed 组间从 0/3 变成 2/3；β 增强到 1.40 后恢复率反而下降。因此当前不是一个可给出稳定检测下限的 assay。预登记阳性未通过，追加阶梯不能事后把人体结果重新升级为 admissible。

另一个需修复的点是：当前 `r2_hidden_vs_h_state_train` 为 0.97–0.999。它是高维 `H` 上的 in-sample R²，易受 235 行对 126 维回归过拟合影响，但仍说明“植入成分未进入 H”尚未被合格的 out-of-sample residual-independence 检验确认。

## 5. 模型内部 dev-test 诊断

下表为三 seed 均值；正值表示 correct dynamic state 更好。

| subject | H−correct | fitted-intercept−correct | shifted−correct | mean−correct | random−correct |
|---|---:|---:|---:|---:|---:|
| `epilepsiae_1146` | +0.1277 | +0.1311 | −0.0272 | −0.0036 | −0.0496 |
| `yuquan_pengzihang` | −0.2380 | −0.0514 | +0.0405 | −0.2357 | +3.5157 |
| `yuquan_zhangkexuan` | −0.1632 | −0.2269 | −0.1597 | −0.1065 | +0.0213 |

`epilepsiae_1146` 相对 H 和静态截距看似有增量，但 correct 不胜 shifted、mean 或 random，因此不能称为时刻特异动态状态。另两位相对 H 为负。9 个主模型均未选择首个 validation checkpoint，也均未卡最大训练预算。

## 6. H1 paired residual evaluation

评价主设置为 `H_strong` + shared-H NB dispersion。每位患者先对三个 seed 取均值；正值支持状态。

### 6.1 5 分钟

| subject | correct vs H | correct vs shifted | dynamic vs mean | eligible |
|---|---:|---:|---:|---:|
| `epilepsiae_1146` | −0.0310 | −0.0083 | −0.0310 | yes |
| `yuquan_pengzihang` | +0.2457 | +0.1620 | +0.2457 | yes |
| `yuquan_zhangkexuan` | −0.7368 | +0.0627 | +0.1458 | yes |

只有 `yuquan_pengzihang` 同时满足三项方向，未形成跨患者一致性。

### 6.2 30 分钟 primary

| subject | correct vs H | correct vs shifted | dynamic vs mean | eligible |
|---|---:|---:|---:|---:|
| `epilepsiae_1146` | −0.3291 | −0.4433 | −0.3291 | yes |
| `yuquan_pengzihang` | +0.3796 | +0.9607 | +1.4055 | no |
| `yuquan_zhangkexuan` | −0.7877 | −0.3674 | −0.2955 | no |

唯一合格患者三项均为负。`yuquan_pengzihang` 的正向读数只能保留为低独立窗口数的开发观察。

### 6.3 120 分钟

没有合格患者。`yuquan_zhangkexuan` 虽输出数值，但所有对比约 +4.4 至 +4.8 且完全不满足预冻结资格，不进入任何主统计或 median。

**H1 技术判定：** `NOT_ESTABLISHED_ASSAY_UNSTABLE`。

## 7. H2a frozen grammar probe

每位患者共 8 arms × 3 seeds；72 个 arm 中 71 个未选择预算末端。主 endpoint 为 `subset identity | K,prefix`。每个数为 correct 相对对应 control 的 NLL gain；`best control` 是 H、mean 和 5 个 shift 中性能最好的对照，因此正值才满足完整要求。

| subject | endpoint | vs H | vs shifted mean | vs mean | vs best control |
|---|---|---:|---:|---:|---:|
| E1146 | continue | +0.00002 | +0.00002 | +0.00002 | −0.00002 |
| E1146 | size | +0.00189 | +0.00102 | +0.00188 | +0.00010 |
| E1146 | subset | −0.00242 | −0.00246 | −0.00244 | −0.00846 |
| E1146 | later continuation | −0.00009 | +0.00013 | −0.00009 | −0.00011 |
| Peng | continue | +0.00307 | +0.00251 | +0.00197 | −0.00094 |
| Peng | size | −0.00234 | +0.00703 | −0.00713 | −0.00774 |
| Peng | subset | −0.01384 | +0.01541 | +0.02115 | −0.01385 |
| Peng | later continuation | +0.00729 | +0.00552 | +0.00447 | −0.00287 |
| Zhang | continue | −0.00046 | +0.00309 | −0.00051 | −0.00189 |
| Zhang | size | +0.00486 | +0.00540 | +0.00497 | +0.00387 |
| Zhang | subset | +0.03756 | +0.01356 | +0.03793 | +0.00475 |
| Zhang | later continuation | −0.00079 | +0.00470 | −0.00089 | −0.00328 |

主 subset endpoint 对 best control 为 1/3 患者有利；三位患者的 seed range 均跨越零或明显不稳定。continue 和 later-continuation 对 best control 为 0/3；size 为 2/3，但 E1146 的 +0.00010 接近零。

**H2a 技术判定：** `NOT_ESTABLISHED_NO_STABLE_TIME_SPECIFIC_GRAMMAR_TRANSFER`。

## 8. 未运行内容的边界

- repaired RNN triage：按合同需在 positive synthetic recovery 后进行，当前未满足；遗留的单步速度试跑不构成结果。
- H2b：未运行；不能称发作迁移阴性。
- H3：未运行；不能称 IED feedback 阴性。
- sealed partition：未打开。

## 9. 图与机器产物

- summary：`results/group_event_state/v0_3_2/v0_3_2_closeout_summary.json`
- machine mirror：`/data/hfosp_group_event_state_v0_3_2/final/v0_3_2_closeout_summary.json`
- payload：`results/group_event_state/v0_3_2/core_evidence_payload_v0_3_2.json`
- state registry：`/data/hfosp_group_event_state_v0_3_2/shared/frozen_state_registry.json`
- H1：`results/group_event_state/core_evidence/figures/group_event_state_h1_future_blocks.{png,pdf}`
- H2a：`results/group_event_state/core_evidence/figures/group_event_state_h2a_repertoire.{png,pdf}`
- H2b/H3 interface：`results/group_event_state/core_evidence/figures/group_event_state_h2b_h3_transfer_feedback.{png,pdf}`
- metadata：`results/group_event_state/core_evidence/core_evidence_metadata.json`

图中实心点为预冻结资格合格结果，空心点为不合格的 development diagnostic；median 只使用实心点。正值背景只表示有利方向，不代表统计建立。

## 10. 下一版最小修复

1. 在 TRAIN 内将人工 hidden component 对 H 做 cross-fitted residualization，并在 dev blocks 报 out-of-sample R²。
2. 将 synthetic recovery 变成单调性合同：β 增强时 recovered fraction 不得下降；至少两个独立 seed batch 复现。
3. 检查 β=1.40 失败 replicate 的 count scale、NB dispersion、gradient 和早停轨迹，避免强信号导致外推/优化发散。
4. 合成恢复后，仅在两位 development 患者比较 leaky bank 与 repaired RNN；不先扩患者。
5. 锁定 architecture 后重跑 6 人 H1/H2a；再使用完全冻结 state 运行 H2b。H3 保持独立机制线。
