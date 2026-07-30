# Topic 5 RNN bounded closeout and replication readiness v0.1 execution plan

> **Superseded.** 当前论文收口与未来外部复制已拆成独立 v1.0 spec/plan；
> 本计划不再承担长期等待状态。

## Milestone A：冻结当前证据包

1. 运行 static-scaffold v0.1 最终 audit；
2. 固定输入 fingerprint、96 teacher-forced cells、baseline freeze 和 confound maps；
3. 更新 `RUN_STATUS.json`、archive index 和 Figure index；
4. 不再重训或重读 target 选择模型。

## Milestone B：完成 Supplementary Figure 6

六块固定为：

| Panel | 科学问题 |
|---|---|
| A | target-sealed 的固定 contact-field 跨状态合同是什么？ |
| B | full GRU 是否既使用顺序、又获得 heldout 顺序增益？ |
| C | 预设正方向的 signed correspondence 是否成立？ |
| D | sign-free morphology 是否超过 all-contact、within-shaft 和 smooth-field null？ |
| E | full GRU 是否超过 raw、regularized、first-order、rank-shuffle 和 teacher-forced？ |
| F | 单个 contact confound 控制后，sign-free morphology 是否仍高于 residual null？ |

输出 PNG、PDF、metadata 和中文 `figures/README.md`，完成 PNG 目检及 metadata 数值核对。

## Milestone C：论文文本

1. Supplementary Methods：self-supervised sequence task、free/teacher decomposition、
   baseline selection、空间 null、partial residual null；
2. Supplementary Results：严格分开 formal heldout gain、order perturbation、signed primary、
   sign-free sensitivity 和 GRU-specific increment；
3. Discussion：静态 scaffold 不等于动态 replay；
4. 主文只保留一句 bounded result，不把 RNN 作为 Figure 2/3 主结论的必要条件。

## Milestone D：replication inventory

只读审计：

1. 是否存在未参与当前 16 人 target 读取的 clinical-onset 患者；
2. exact contact join 和 seizure 数；
3. patient-level 可用分母；
4. 若不足，输出 `BLOCKED_INPUT`，不做 target reuse 型伪复制。

## Milestone E：决策

- 有独立 cohort：直接运行 frozen-field replication；
- 无独立 cohort：完成 replication-ready handoff 后停止；
- 不因当前单项阴性给出“RNN 永远无用”或“患者不存在病理轴”的结论；
- 不满足独立 heldout gain + static increment 时，不启动新的 RNN 动力学模型。
