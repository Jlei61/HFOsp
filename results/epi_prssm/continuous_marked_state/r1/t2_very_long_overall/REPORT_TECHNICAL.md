# 超长尺度 IED→状态实验：技术审计报告

## 验收结论

- 工程完成：通过。5 preparation、35 T1、70 H3 artifacts 完成；sealed/formal test 均为 false。
- 科学验收：不通过 H3 阳性；结论为 `H3_LONG_UNRESOLVED_NO_CURRENT_SUPPORT`。
- P0 更正：原调度器只用 `selected_epochs > 0` 判定 T1 可用，导致陈子阳进入 H3。正确规则已改为：非零 epoch、filtered 胜 no-state、persistent 胜 validation-correction-off 三者同时成立。

## T1 仪器

| 患者 | 非零 epoch | predictive+persistent | correct-time 有利 | 判定 |
|---|---:|---:|---:|---|
| 程帅 | 0/7 | 0/7 | 0/7 | 退化 |
| 彭子航 | 0/7 | 0/7 | 0/7 | 退化 |
| E922 | 0/7 | 0/7 | 0/7 | 退化 |
| 陈子阳 | 7/7 | 0/7 | 2/7 | 训练动但外层验证无效 |
| 韩宇轩 | 7/7 | 7/7 | 2/7 | predictive persistent memory；时间专属性不足 |

陈子阳 filtered-minus-no-state 中位 +0.000438824，persistent-minus-validation-off 中位 +0.000439301，均为不利方向。韩宇轩相应为 -0.00656089 与 -0.00488895。

## H3 主读数

只有韩宇轩满足更正后的 predictive+persistent gate：

| kernel / window | real−intercept | 有利 seed | real−delayed | 有利 seed | TRAIN/VAL 不重叠整窗 |
|---|---:|---:|---:|---:|---:|
| boxcar / N=2000 | +4.1003 | 1/7 | -1.67407 | 6/7 | 1/1 |
| boxcar / 6 h | +94.9233 | 0/7 | +38.5744 | 0/7 | 1/1 |

N=2000 虽在 real−delayed 上多 seed 有利，但 real−intercept 失败；这表示 delayed arm 更差，不能转写为 exposure 增量。该档 7/7 个 seed 是有效估计（拟合臂相对截距对照最大 1.23 倍）。

6 h 那一档只有 0/7 个 seed 是有效估计：拟合臂内含截距对照的常数，却落到它的最大 149 倍，属于外推。因此该行应写成**不可估计**（与结构零同类），不能写成"两个主对照均失败"。

## 时间尺度审计

- generator-weighted 在所有名义长窗口中，90% 权重仍集中在约 1.8–2.0 h，只是近期记忆敏感性。
- boxcar 的 N=2000 在韩宇轩覆盖约 8.72 h 的 90% 权重，陈子阳 4,000 次覆盖约 5.60 h，确实是长记忆实现。
- 但所有 boxcar validation 条件的完整不重叠窗口数均为 1，无法形成患者内独立重复。

## 端点与限制

韩宇轩 N=2000 的 timing 在 7/7 seed 落到 scale floor，STOP 多数 seed、contact subset 多个 seed 也偏弱；因此 equal-block 总分应谨慎。next-event exact likelihood 同样未支持：N=2000 real−intercept joint NLL 中位 +3.8147e-06，6 h 为 +0.00118256。

陈子阳 4,000 次/6 h 的表面有利方向保留为“若未来 T1 修复后值得复查”的候选，不纳入当前 H3 证据。

## 文件

- machine audit: `/home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/r1/t2_very_long_overall/machine_audit.json`
- generator-weighted summary: `/home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/r1/t2_very_long_discovery/summary.json`
- boxcar summary: `/home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/r1/t2_very_long_boxcar/summary.json`
- support audit: `/home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/r1/t2_long_total_effect/cohort_support/summary.json`
